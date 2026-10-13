"""Kaggle 2xT4 kernel: zero-shot VLM crossing-intention on JAAD, clean vs. distractor scenes.

Each model runs in its own subprocess pinned to one GPU (DataParallel on 2xT4 is slow).
Scoring is a single forward pass: P(Yes) vs P(No) from the next-token logits, so
there is no answer parsing and we get a continuous score for AUC.
Predictions are appended to /kaggle/working/preds/<model>.csv as they are produced.
"""
import csv, os, random, subprocess, sys, time
from pathlib import Path

OUT = Path("/kaggle/working")
PRED_DIR = OUT / "preds"
SMOKE = False  # True: 6 samples per model to check memory/speed
WIDTH = 896  # frames are 1920x1080; downscale before the processor sees them

PROMPTS = {
    "p0": "This image is from a car's front camera. Will the pedestrian in the red box "
          "cross the road in front of the car in the next two seconds? Answer Yes or No.",
    "p1": "You are looking through the windshield of a moving car. Is the person marked "
          "with the red box about to step into the road ahead? Answer Yes or No.",
    "p2": "Dashcam view. Question: does the highlighted pedestrian (red box) intend to "
          "cross the street in front of this vehicle? Answer Yes or No.",
}
CLIP_PROMPT = ("These are 4 consecutive frames (0.7 s apart in total) from a car's front camera. "
               "Will the pedestrian in the red box cross the road in front of the car in the "
               "next two seconds? Answer Yes or No.")

MODELS = {
    "qwen25vl7b": dict(hf="Qwen/Qwen2.5-VL-7B-Instruct", clip=True),
    "llavaov7b": dict(hf="llava-hf/llava-onevision-qwen2-7b-ov-hf", clip=False),
    "smolvlm2b": dict(hf="HuggingFaceTB/SmolVLM-Instruct", clip=False),
    # same weights as qwen25vl7b; runs the counterfactual ablations instead of the main grid
    "qwenabl": dict(hf="Qwen/Qwen2.5-VL-7B-Instruct", clip=False, ablate=True),
}
# Ablation conditions (all use prompt p2, the least "No"-biased prompt in the main run):
#   clean    - unchanged frame (re-run in the same process for a paired comparison)
#   mask     - target pedestrian inpainted out, red box kept: is the model using the person at all?
#   boxshift - same frame, box moved to a pedestrian-free spot at the same height
#   blank    - grey frame with only the box: position + language prior
#   near/far - one distractor on the target's ground line, close (< 0.2 W) or far (> 0.4 W)
#   n4       - four distractors at random positions
ABLATIONS = ["clean", "mask", "boxshift", "blank", "near", "far", "n4"]
# models run one after another, each split across both GPUs
ORDER = ["qwen25vl7b", "llavaov7b", "smolvlm2b"]


def find_input():
    hits = list(Path("/kaggle/input").rglob("samples.csv"))
    assert hits, "samples.csv not found under /kaggle/input"
    return hits[0].parent


# ---------------------------------------------------------------- images
def load_frame(src, sid, k):
    from PIL import Image
    return Image.open(src / "frames" / f"{sid}_{k}.jpg").convert("RGB")


def draw_box(img, box):
    from PIL import ImageDraw
    x1, y1, x2, y2 = box
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=6)
    return img


def overlaps(a, b, pad=40):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def add_distractors(img, target_box, donors, rng, n=2):
    """Paste n pedestrian crops from other videos, scaled to the target's height,
    at random spots that do not overlap the target. They are irrelevant to the
    target's intention, so a robust model's answer should not change."""
    th = target_box[3] - target_box[1]
    placed = [target_box]
    W, H = img.size
    for crop in rng.sample(donors, n):
        c = crop.resize((max(1, int(crop.width * th / crop.height)), int(th)))
        for _ in range(50):
            x = rng.randint(0, W - c.width)
            y = rng.randint(int(H * 0.35), max(int(H * 0.35), H - c.height))
            box = (x, y, x + c.width, y + c.height)
            if not any(overlaps(box, p) for p in placed):
                img.paste(c, (x, y))
                placed.append(box)
                break
    return img


def paste_on_ground(img, target_box, crop, rng, near):
    """Paste one distractor bottom-aligned with the target (same ground line, same height),
    horizontally close (near) or far (not near), without overlapping the target."""
    x1, y1, x2, y2 = target_box
    th, tw, W = y2 - y1, x2 - x1, img.size[0]
    c = crop.resize((max(1, int(crop.width * th / crop.height)), int(th)))
    cx = (x1 + x2) / 2
    for _ in range(200):
        d = rng.uniform(1.3 * tw, 0.2 * W) if near else rng.uniform(0.4 * W, W)
        x = int(cx + rng.choice([-1, 1]) * d - c.width / 2)
        if 0 <= x <= W - c.width and not overlaps((x, y1, x + c.width, y2), target_box, pad=5):
            img.paste(c, (x, int(y1)))
            return img, True
    return img, False


def inpaint_box(img, box, pad=6):
    import cv2, numpy as np
    a = np.array(img)[:, :, ::-1].copy()
    m = np.zeros(a.shape[:2], np.uint8)
    x1, y1, x2, y2 = box
    m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
    a = cv2.inpaint(a, m, 9, cv2.INPAINT_TELEA)
    from PIL import Image
    return Image.fromarray(a[:, :, ::-1])


def shifted_box(box, others, W, rng):
    """Same size and height as box, moved horizontally to a spot clear of all annotated people."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    for _ in range(200):
        nx = rng.randint(0, W - w)
        nb = (nx, y1, nx + w, y2)
        if abs(nx - x1) > 2 * w and not any(overlaps(nb, o, pad=10) for o in others + [box]):
            return nb
    return None


def build_ablation_inputs(src, rows, all_boxes):
    from PIL import Image
    rng = random.Random(1)
    boxes = {r["sample_id"]: [tuple(map(int, b.split(","))) for b in r["boxes"].split()] for r in rows}
    donors = {}
    for r in rows:
        x1, y1, x2, y2 = boxes[r["sample_id"]][-1]
        donors.setdefault(r["video"], []).append(load_frame(src, r["sample_id"], 3).crop((x1, y1, x2, y2)))
    out, meta, s = {}, {}, WIDTH / 1920
    size = lambda im: im.resize((WIDTH, int(im.height * s)))
    for r in rows:
        sid, b = r["sample_id"], boxes[r["sample_id"]][-1]
        pool = [c for v, cs in donors.items() if v != r["video"] for c in cs]
        frame = lambda: load_frame(src, sid, 3)
        others = all_boxes.get((r["video"], int(r["frames"].split()[-1])), [])
        d = {"clean": draw_box(frame(), b), "mask": draw_box(inpaint_box(frame(), b), b)}
        sb = shifted_box(b, others, 1920, rng)
        if sb:
            d["boxshift"] = draw_box(frame(), sb)
        d["blank"] = draw_box(Image.new("RGB", (1920, 1080), (128, 128, 128)), b)
        for k, near in (("near", True), ("far", False)):
            im, ok = paste_on_ground(frame(), b, rng.choice(pool), rng, near)
            if ok:
                d[k] = draw_box(im, b)
        d["n4"] = draw_box(add_distractors(frame(), b, pool, rng, n=4), b)
        out[sid] = {k: size(v) for k, v in d.items()}
        if len(out) <= 12:  # keep a few examples for the write-up
            ex = OUT / "examples"
            ex.mkdir(exist_ok=True)
            for k, v in out[sid].items():
                v.save(ex / f"{sid}_{k}.jpg", quality=85)
    return out


def build_inputs(src, rows):
    """Return {sample_id: dict(clean=img, distract=img, clip=[imgs])}, all downscaled."""
    rng = random.Random(0)
    boxes = {r["sample_id"]: [tuple(map(int, b.split(","))) for b in r["boxes"].split()] for r in rows}
    donors_by_video = {}
    for r in rows:
        x1, y1, x2, y2 = boxes[r["sample_id"]][-1]
        donors_by_video.setdefault(r["video"], []).append(load_frame(src, r["sample_id"], 3).crop((x1, y1, x2, y2)))
    out = {}
    s = WIDTH / 1920
    for r in rows:
        sid, bx = r["sample_id"], boxes[r["sample_id"]]
        donors = [c for v, cs in donors_by_video.items() if v != r["video"] for c in cs]
        clean = draw_box(load_frame(src, sid, 3), bx[-1])
        distract = draw_box(add_distractors(load_frame(src, sid, 3), bx[-1], donors, rng), bx[-1])
        clip = [draw_box(load_frame(src, sid, k), bx[k]) for k in range(4)]
        size = lambda im: im.resize((WIDTH, int(im.height * s)))
        out[sid] = dict(clean=size(clean), distract=size(distract), clip=[size(c) for c in clip])
    return out


# ---------------------------------------------------------------- models
def load_model(name):
    import torch
    from transformers import AutoProcessor
    cfg = MODELS[name]
    # fp16 split across both T4s; no 4-bit quantization (its load step is RAM-heavy)
    kw = dict(dtype=torch.float16, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"})
    if name.startswith("qwen"):
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        proc = AutoProcessor.from_pretrained(cfg["hf"], min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
    elif name.startswith("llava"):
        from transformers import LlavaOnevisionForConditionalGeneration as M
        proc = AutoProcessor.from_pretrained(cfg["hf"])
        # default anyres tiling gives ~7k image tokens per frame (10 s/call on T4); cap at 1x2 / 2x1 tiles
        proc.image_processor.image_grid_pinpoints = [[384, 384], [384, 768], [768, 384]]
    else:
        from transformers import AutoModelForImageTextToText as M
        proc = AutoProcessor.from_pretrained(cfg["hf"])
    model = M.from_pretrained(cfg["hf"], **kw).eval()
    if name.startswith("llava"):  # must match the processor's tiling
        model.config.image_grid_pinpoints = proc.image_processor.image_grid_pinpoints
    return model, proc


def yes_no_ids(tok):
    ids = lambda words: sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    return ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])


def score(model, proc, images, prompt, yes_ids, no_ids):
    import torch
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    y, n = torch.logsumexp(lp[yes_ids], 0), torch.logsumexp(lp[no_ids], 0)
    p_yes = torch.sigmoid(y - n).item()
    return p_yes, torch.exp(torch.logsumexp(torch.stack([y, n]), 0)).item()


def run_model(name):
    src = find_input()
    rows = list(csv.DictReader(open(src / "samples.csv")))
    if SMOKE:
        rows = rows[::50]
    if MODELS[name].get("ablate"):
        allb = {}
        abl = [p for p in [OUT / "all_boxes.csv"] if p.exists()] or list(Path("/kaggle/input").rglob("all_boxes.csv"))
        if abl:
            for r in csv.DictReader(open(abl[0])):
                allb.setdefault((r["video"], int(r["frame"])), []).append(tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2")))
        else:
            print("all_boxes.csv not found: boxshift avoids only the target", flush=True)
        inputs = build_ablation_inputs(src, rows, allb)
    else:
        inputs = build_inputs(src, rows)
    t0 = time.time()
    model, proc = load_model(name)
    print(f"[{name}] loaded in {time.time() - t0:.0f}s, "
          f"{__import__('torch').cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    if MODELS[name].get("ablate"):
        jobs = [(c, "p2") for c in ABLATIONS]
    else:
        jobs = [(c, p) for p in PROMPTS for c in ["clean"]] + [("distract", "p0")]
    if MODELS[name]["clip"]:
        jobs.append(("clip", "clip"))
    PRED_DIR.mkdir(exist_ok=True)
    path = PRED_DIR / f"{name}.csv"
    done = set()
    if path.exists():
        done = {(r["sample_id"], r["cond"], r["prompt"]) for r in csv.DictReader(open(path))}
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(["model", "sample_id", "cond", "prompt", "p_yes", "mass_yes_no", "sec"])
    for cond, pk in jobs:
        for r in rows:
            sid = r["sample_id"]
            if (sid, cond, pk) in done:
                continue
            t = time.time()
            if cond not in inputs[sid] and cond != "clip":
                continue  # e.g. no valid spot for boxshift/near/far in this frame
            imgs = inputs[sid]["clip"] if cond == "clip" else [inputs[sid][cond]]
            prompt = CLIP_PROMPT if cond == "clip" else PROMPTS[pk]
            try:
                p, mass = score(model, proc, imgs, prompt, yes_ids, no_ids)
            except Exception as e:  # keep going; record the failure
                print(f"[{name}] {sid} {cond}/{pk} failed: {e}", flush=True)
                p, mass = float("nan"), float("nan")
            w.writerow([name, sid, cond, pk, f"{p:.5f}", f"{mass:.4f}", f"{time.time() - t:.2f}"])
            f.flush()
            n_done = len(done) + 1
            done.add((sid, cond, pk))
            if n_done % 25 == 0:
                print(f"[{name}] {n_done} calls, last {time.time() - t:.2f}s", flush=True)
        print(f"[{name}] finished {cond}/{pk}", flush=True)
    f.close()


def mem_monitor(path, every=1):
    """Log host RAM, disk and GPU memory so a killed run leaves evidence."""
    import shutil, threading, torch
    def loop():
        t0 = time.time()
        with open(path, "a") as f:
            while True:
                m = dict(l.split(":") for l in open("/proc/meminfo").read().splitlines()[:3])
                avail = int(m["MemAvailable"].split()[0]) / 2**20
                disk = shutil.disk_usage("/root").free / 2**30
                gpu = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
                gpu = " ".join(f"gpu{i}_used={(t - fr) / 2**30:.1f}G" for i, (fr, t) in enumerate(gpu))
                f.write(f"{time.time() - t0:7.0f}s ram_avail={avail:.1f}G root_disk_free={disk:.0f}G {gpu}\n")
                f.flush()
                time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()


def main():
    t0 = time.time()
    import torch, transformers
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()} "
          f"transformers {transformers.__version__}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "num2words"], check=False)
    mem_monitor(OUT / "memory.log")
    codes = []
    for n in ORDER:
        t = time.time()
        codes.append(subprocess.run([sys.executable, os.path.abspath(sys.argv[0]), n]).returncode)
        print(f"[{n}] exit {codes[-1]} after {time.time() - t:.0f}s", flush=True)
        # free disk: drop this model's weights before downloading the next
        subprocess.run("rm -rf /root/.cache/huggingface/hub/models--*", shell=True)
    print("exit codes:", codes, f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_model(sys.argv[1])
    else:
        main()
