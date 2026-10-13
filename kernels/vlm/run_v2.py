"""v2 experiments on the 433-pedestrian set (samples_v2.csv from the jaad-vlm-frames-v2 kernel).

    python run_v2.py <key>

Keys (one model per process; predictions append to /kaggle/working/preds_v2/<key>.csv and resume):
  q7_v2     Qwen2.5-VL-7B: clean, mask, blank, crop, scene_crop, scene_cropmask, speed
  q7_probe  Qwen2.5-VL-7B: hidden states (last token + mean image tokens) for clean/mask/blank/crop
  q7_cot    Qwen2.5-VL-7B: step-by-step reasoning (with vehicle speed), then a scored final Yes/No
  q3_v2     Qwen2.5-VL-3B: clean, mask, blank, crop
  q3vl8_v2  Qwen3-VL-8B:   clean, mask, blank, crop
  q32_v2    Qwen2.5-VL-32B (4-bit nf4): clean, mask, blank, crop
"""
import csv, os, sys, time
from pathlib import Path
import run_vlm as R  # shared helpers: frames, boxes, inpainting, yes/no scoring

OUT = Path("/kaggle/working")
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0: smoke test on the first LIMIT samples
PRED = OUT / ("smoke_v2" if LIMIT else "preds_v2")

SCENE = ("This image is from a car's front camera. Does the highlighted pedestrian (red box) intend to "
         "cross the street in front of this vehicle in the next two seconds? Answer Yes or No.")
CROP = ("This is a close-up of a pedestrian (red box) seen from a car's front camera. Does this pedestrian "
        "intend to cross the street in front of the vehicle in the next two seconds? Answer Yes or No.")
SCENE_CROP = ("The first image is a car's front-camera view; the pedestrian of interest is in the red box. "
              "The second image is a close-up of the same pedestrian. Does this pedestrian intend to cross the "
              "street in front of the vehicle in the next two seconds? Answer Yes or No.")
SPEED = {"decelerating": "slowing down", "accelerating": "speeding up", "stopped": "stopped",
         "moving_slow": "moving slowly", "moving_fast": "moving fast"}
COT = ("This image is from a car's front camera. The car is currently {speed}. Consider the highlighted "
       "pedestrian (red box): their position relative to the road and curb, body orientation, gaze, motion, "
       "and whether there is a crosswalk or traffic signal. In at most four short sentences, reason step by step "
       "about whether they will cross the street in front of the vehicle in the next two seconds.")
FINAL = "Based on your analysis, will the pedestrian cross in front of the vehicle in the next two seconds? Answer Yes or No."

CFG = {
    "q7_v2": dict(hf="Qwen/Qwen2.5-VL-7B-Instruct",
                  conds=["clean", "mask", "blank", "crop", "scene_crop", "scene_cropmask", "speed"]),
    "q7_probe": dict(hf="Qwen/Qwen2.5-VL-7B-Instruct", probe=["clean", "mask", "blank", "crop"]),
    "q7_cot": dict(hf="Qwen/Qwen2.5-VL-7B-Instruct", cot=True),
    "q3_v2": dict(hf="Qwen/Qwen2.5-VL-3B-Instruct", conds=["clean", "mask", "blank", "crop"]),
    "q3vl8_v2": dict(hf="Qwen/Qwen3-VL-8B-Instruct", conds=["clean", "mask", "blank", "crop"]),
    "q32_v2": dict(hf="Qwen/Qwen2.5-VL-32B-Instruct", conds=["clean", "mask", "blank", "crop"], q4=True),
}
PROBE_LAYERS = (7, 14, 21, 28)


def crop_around(img, box, scale=1920):
    """Context crop: the box plus 1 box-width each side and half a box-height above/below,
    resized so the long side is 448 px."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    c = img.crop((max(0, x1 - w), max(0, y1 - h // 2), min(img.width, x2 + w), min(img.height, y2 + h // 2)))
    k = 448 / max(c.size)
    return c.resize((max(1, int(c.width * k)), max(1, int(c.height * k))))


def build(src, rows):
    from PIL import Image
    s = R.WIDTH / 1920
    size = lambda im: im.resize((R.WIDTH, int(im.height * s)))
    out = {}
    for r in rows:
        sid = r["sample_id"]
        b = tuple(map(int, r["boxes"].split()[-1].split(",")))
        frame = R.load_frame(src, sid, 3)
        masked = R.inpaint_box(frame.copy(), b)
        boxed, boxed_m = R.draw_box(frame.copy(), b), R.draw_box(masked, b)
        out[sid] = dict(
            clean=[size(boxed)], mask=[size(boxed_m)],
            blank=[size(R.draw_box(Image.new("RGB", (1920, 1080), (128, 128, 128)), b))],
            crop=[crop_around(boxed, b)],
            scene_crop=[size(boxed), crop_around(boxed, b)],
            scene_cropmask=[size(boxed_m), crop_around(boxed_m, b)])
        out[sid]["speed"] = out[sid]["clean"]
    return out


def prompt_for(cond, r):
    if cond == "crop":
        return CROP
    if cond.startswith("scene_crop"):
        return SCENE_CROP
    if cond == "speed":
        return SCENE.replace("Does the highlighted", f"The car is currently {SPEED.get(r['vehicle'], 'moving')}. Does the highlighted")
    return SCENE


def load(key):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    hf = CFG[key]["hf"]
    kw = dict(dtype=torch.float16, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"})
    if CFG[key].get("q4"):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained(hf, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28) \
        if "Qwen2.5" in hf else AutoProcessor.from_pretrained(hf)
    if "Qwen3" in hf:
        proc.image_processor.size = {"shortest_edge": 256 * 32 * 32, "longest_edge": 640 * 32 * 32}
    model = AutoModelForImageTextToText.from_pretrained(hf, **kw).eval()
    return model, proc


def inputs_for(proc, images, text, reasoning=None):
    """User turn with the images + text. With `reasoning`, adds that assistant turn and a final
    user question, so the next token is the model's Yes/No after its own reasoning."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    if reasoning is not None:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": reasoning}]},
                 {"role": "user", "content": [{"type": "text", "text": FINAL}]}]
    chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=[chat], images=images, return_tensors="pt").to("cuda:0")


def yes_prob(model, inputs, yes_ids, no_ids):
    import torch
    with torch.no_grad():
        lp = torch.log_softmax(model(**inputs).logits[0, -1].float(), -1)
    y, n = torch.logsumexp(lp[yes_ids], 0), torch.logsumexp(lp[no_ids], 0)
    return torch.sigmoid(y - n).item(), torch.exp(torch.logsumexp(torch.stack([y, n]), 0)).item()


def writer(path, header):
    done = set()
    if path.exists():
        done = {(r["sample_id"], r["cond"]) for r in csv.DictReader(open(path))}
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(header)
    return f, w, done


def main(key):
    import torch, numpy as np
    src = R.find_input()
    rows = list(csv.DictReader(open(src / "samples.csv")))
    assert "group" in rows[0], "expected the v2 sample set"
    if LIMIT:
        rows = rows[:LIMIT]
    t0 = time.time()
    imgs = build(src, rows)
    print(f"[{key}] built inputs for {len(rows)} samples in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    model, proc = load(key)
    print(f"[{key}] loaded in {time.time() - t0:.0f}s", flush=True)
    yes_ids, no_ids = R.yes_no_ids(proc.tokenizer)
    PRED.mkdir(exist_ok=True)
    cfg = CFG[key]

    if "conds" in cfg:
        f, w, done = writer(PRED / f"{key}.csv", ["model", "sample_id", "cond", "p_yes", "mass_yes_no", "sec"])
        for cond in cfg["conds"]:
            for i, r in enumerate(rows):
                if (r["sample_id"], cond) in done:
                    continue
                t = time.time()
                try:
                    p, m = yes_prob(model, inputs_for(proc, imgs[r["sample_id"]][cond], prompt_for(cond, r)), yes_ids, no_ids)
                except Exception as e:
                    print(f"[{key}] {r['sample_id']} {cond}: {e}", flush=True); p = m = float("nan")
                w.writerow([key, r["sample_id"], cond, f"{p:.5f}", f"{m:.4f}", f"{time.time() - t:.2f}"]); f.flush()
                if i % 50 == 0:
                    print(f"[{key}] {cond} {i}/{len(rows)} {time.time() - t:.2f}s", flush=True)
            print(f"[{key}] finished {cond}", flush=True)

    if "probe" in cfg:
        for cond in cfg["probe"]:
            path = PRED / f"{key}_{cond}.npz"
            if path.exists():
                continue
            last, img_mean = [], []
            for r in rows:
                x = inputs_for(proc, imgs[r["sample_id"]][cond], SCENE if cond != "crop" else CROP)
                with torch.no_grad():
                    hs = model(**x, output_hidden_states=True).hidden_states
                is_img = (x["input_ids"][0] == model.config.image_token_id)
                last.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in PROBE_LAYERS]))
                img_mean.append(np.stack([hs[l][0, is_img].float().mean(0).cpu().numpy() for l in PROBE_LAYERS]))
            np.savez_compressed(path, ids=np.array([r["sample_id"] for r in rows]), layers=np.array(PROBE_LAYERS),
                                last=np.stack(last).astype(np.float16), img_mean=np.stack(img_mean).astype(np.float16))
            print(f"[{key}] finished probe {cond}", flush=True)

    if cfg.get("cot"):
        f, w, done = writer(PRED / f"{key}.csv", ["model", "sample_id", "cond", "p_yes", "mass_yes_no", "sec", "reasoning"])
        for i, r in enumerate(rows):
            if (r["sample_id"], "cot") in done:
                continue
            t = time.time()
            q = COT.format(speed=SPEED.get(r["vehicle"], "moving"))
            try:
                x = inputs_for(proc, imgs[r["sample_id"]]["clean"], q)
                with torch.no_grad():
                    g = model.generate(**x, max_new_tokens=128, do_sample=False)
                reasoning = proc.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                p, m = yes_prob(model, inputs_for(proc, imgs[r["sample_id"]]["clean"], q, reasoning=reasoning), yes_ids, no_ids)
            except Exception as e:
                print(f"[{key}] {r['sample_id']}: {e}", flush=True); p = m = float("nan"); reasoning = ""
            w.writerow([key, r["sample_id"], "cot", f"{p:.5f}", f"{m:.4f}", f"{time.time() - t:.2f}",
                        reasoning.replace("\n", " ")]); f.flush()
            if i % 25 == 0:
                print(f"[{key}] cot {i}/{len(rows)} {time.time() - t:.2f}s", flush=True)
        print(f"[{key}] finished cot", flush=True)
    print(f"[{key}] ALL FINISHED", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
