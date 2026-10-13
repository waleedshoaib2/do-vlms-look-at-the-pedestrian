"""Pick a balanced set of JAAD crossing / not-crossing samples.

Protocol, loosely following the JAAD/PIE action-prediction benchmark
(Kotseruba et al., WACV 2021):
- uses behavioral pedestrians (id ending in 'b') from all default-split videos (zero-shot, so no train/test leakage concern)
- the event is crossing_point for crossers, and the last annotated frame otherwise
- the observation ends TTE frames before the event, TTE in {30, 45, 60} (1-2 s),
  so a pedestrian can give up to 3 samples; bootstrap by ped_id, not by sample
- 4 observation frames, 7 frames apart (~0.7 s of context)
"""
import csv, random, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent / "JAAD"
TTES, STRIDE, N_FRAMES, MIN_H, PER_CLASS, SEED = (30, 45, 60), 7, 4, 50, 150, 0


def boxes_by_ped(vid):
    root = ET.parse(ROOT / "annotations" / f"{vid}.xml").getroot()
    out = {}
    for tr in root.iter("track"):
        if tr.get("label") != "pedestrian":
            continue
        for b in tr.findall("box"):
            attrs = {a.get("name"): a.text for a in b.findall("attribute")}
            pid = attrs.get("id")
            out.setdefault(pid, {})[int(b.get("frame"))] = (
                float(b.get("xtl")), float(b.get("ytl")),
                float(b.get("xbr")), float(b.get("ybr")),
                int(b.get("occluded")), attrs.get("look"), attrs.get("action"))
    return out


def main():
    vids = sorted({v for s in ("train", "val", "test") for v in (ROOT / f"split_ids/default/{s}.txt").read_text().split()})
    cands = {0: [], 1: []}
    for vid in vids:
        attr_root = ET.parse(ROOT / "annotations_attributes" / f"{vid}_attributes.xml").getroot()
        boxes = boxes_by_ped(vid)
        for p in attr_root.iter("pedestrian"):
            pid, cross = p.get("id"), int(p.get("crossing"))
            if not pid.endswith("b") or cross not in (0, 1) or pid not in boxes:
                continue
            fr = boxes[pid]
            event = int(p.get("crossing_point")) if cross == 1 else max(fr)
            for tte in TTES:
                t = event - tte
                obs = [t - STRIDE * k for k in reversed(range(N_FRAMES))]
                if not all(f in fr for f in obs):
                    continue
                last = fr[t]
                if last[3] - last[1] < MIN_H or last[4] == 1:
                    continue
                cands[cross].append(dict(
                    sample_id=f"{vid}_{pid}_tte{tte}", video=vid, ped_id=pid, label=cross, tte=tte,
                    frames=" ".join(map(str, obs)),
                    boxes=" ".join(",".join(f"{v:.0f}" for v in fr[f][:4]) for f in obs),
                    look=last[5], action=last[6],
                    intersection=p.get("intersection"), designated=p.get("designated"),
                    signalized=p.get("signalized"), group_size=p.get("group_size")))
    print({k: len(v) for k, v in cands.items()})
    rng = random.Random(SEED)
    rows = []
    for k, v in cands.items():
        rows += rng.sample(v, min(PER_CLASS, len(v)))
    rows.sort(key=lambda r: r["sample_id"])
    with open(Path(__file__).parent / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "samples from", len({r["video"] for r in rows}), "videos")


if __name__ == "__main__":
    main()
