"""v2 sample set: one observation window per pedestrian (all samples independent).

Labels follow the frame-level 'cross' attribute, as in the JAAD_all setting of the
benchmark (Kotseruba et al., WACV 2021):
  crossers       - behavioral pedestrians ('pedestrian' tracks) that are ever labeled crossing;
                   event = first crossing frame
  beh_noncross   - behavioral pedestrians never labeled crossing; event = last frame
  bystander      - 'ped' tracks (no behavioral labels, treated as not crossing); event = last frame
Observation: 4 frames, 7 apart, ending TTE in {30, 45, 60} frames before the event (one TTE
chosen at random among the valid ones). Box height >= 50 px, not occluded at the last frame.
Vehicle speed state (JAAD annotations_vehicle) at the last observed frame is recorded for the
speed-prompt experiment.
"""
import csv, random, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent / "JAAD"
TTES, STRIDE, N_FRAMES, MIN_H, N_BYSTANDERS, SEED = (30, 45, 60), 7, 4, 50, 150, 0


def tracks(vid):
    root = ET.parse(ROOT / "annotations" / f"{vid}.xml").getroot()
    out = {}
    for tr in root.iter("track"):
        lab = tr.get("label")
        if lab not in ("pedestrian", "ped"):
            continue
        for b in tr.findall("box"):
            a = {x.get("name"): x.text for x in b.findall("attribute")}
            d = out.setdefault((lab, a["id"]), {})
            d[int(b.get("frame"))] = dict(
                box=(float(b.get("xtl")), float(b.get("ytl")), float(b.get("xbr")), float(b.get("ybr"))),
                occ=int(b.get("occluded")), cross=a.get("cross"), look=a.get("look"), action=a.get("action"))
    return out


def vehicle(vid):
    root = ET.parse(ROOT / "annotations_vehicle" / f"{vid}_vehicle.xml").getroot()
    return {int(f.get("id")): f.get("action") for f in root.iter("frame")}


def main():
    rng = random.Random(SEED)
    vids = sorted({v for s in ("train", "val", "test") for v in (ROOT / f"split_ids/default/{s}.txt").read_text().split()})
    groups = {"crosser": [], "beh_noncross": [], "bystander": []}
    for vid in vids:
        veh = vehicle(vid)
        for (lab, pid), fr in tracks(vid).items():
            crossing = [f for f in sorted(fr) if fr[f]["cross"] == "crossing"]
            label = int(bool(crossing))
            event = crossing[0] if label else max(fr)
            valid = []
            for tte in TTES:
                t = event - tte
                obs = [t - STRIDE * k for k in reversed(range(N_FRAMES))]
                if all(f in fr for f in obs) and fr[t]["box"][3] - fr[t]["box"][1] >= MIN_H and fr[t]["occ"] == 0:
                    valid.append((tte, obs))
            if not valid:
                continue
            tte, obs = rng.choice(valid)
            last = fr[obs[-1]]
            group = "bystander" if lab == "ped" else ("crosser" if label else "beh_noncross")
            groups[group].append(dict(
                sample_id=f"{vid}_{pid}_tte{tte}", video=vid, ped_id=pid, label=label, group=group, tte=tte,
                frames=" ".join(map(str, obs)),
                boxes=" ".join(",".join(f"{v:.0f}" for v in fr[f]["box"]) for f in obs),
                look=last["look"] or "", action=last["action"] or "",
                vehicle=veh.get(obs[-1], "")))
    print({k: len(v) for k, v in groups.items()})
    rows = groups["crosser"] + groups["beh_noncross"] + rng.sample(groups["bystander"], N_BYSTANDERS)
    rows.sort(key=lambda r: r["sample_id"])
    with open(Path(__file__).parent / "samples_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(len(rows), "samples,", len({r["video"] for r in rows}), "videos")


if __name__ == "__main__":
    main()
