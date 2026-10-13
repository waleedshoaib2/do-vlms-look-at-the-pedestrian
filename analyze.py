"""Score the VLM predictions. Bootstrap CIs resample pedestrians (not samples),
because one pedestrian can contribute up to 3 observation windows."""
import csv, glob, math, random, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
B, SEED = 2000, 0


def auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(y, s):
    pred = [int(v >= 0.5) for v in s]
    tp = sum(p and t for p, t in zip(pred, y))
    fp = sum(p and not t for p, t in zip(pred, y))
    fn = sum((not p) and t for p, t in zip(pred, y))
    acc = sum(p == t for p, t in zip(pred, y)) / len(y)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(acc=acc, f1=f1, auc=auc(y, s), yes_rate=sum(pred) / len(pred))


def boot(groups, fn):
    """groups: {ped: [(y, s), ...]} -> 95% CI of fn(y, s) under cluster bootstrap."""
    rng, keys, vals = random.Random(SEED), list(groups), []
    for _ in range(B):
        pick = [groups[rng.choice(keys)] for _ in keys]
        y = [a for g in pick for a, _ in g]
        s = [b for g in pick for _, b in g]
        vals.append(fn(y, s))
    vals = sorted(v for v in vals if not math.isnan(v))
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main(pred_dir):
    samples = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "samples.csv"))}
    preds = defaultdict(dict)  # (model, cond, prompt) -> {sid: p_yes}
    for path in glob.glob(f"{pred_dir}/*.csv"):
        for r in csv.DictReader(open(path)):
            if r["p_yes"] != "nan":
                preds[(r["model"], r["cond"], r["prompt"])][r["sample_id"]] = float(r["p_yes"])

    out = []
    for (m, c, p), d in sorted(preds.items()):
        sids = sorted(d)
        y = [int(samples[s]["label"]) for s in sids]
        sc = [d[s] for s in sids]
        g = defaultdict(list)
        for s in sids:
            g[samples[s]["video"] + samples[s]["ped_id"]].append((int(samples[s]["label"]), d[s]))
        mt = metrics(y, sc)
        lo, hi = boot(g, lambda yy, ss: metrics(yy, ss)["acc"])
        out.append(dict(model=m, cond=c, prompt=p, n=len(sids), **{k: round(v, 3) for k, v in mt.items()},
                        acc_ci=f"[{lo:.3f}, {hi:.3f}]"))

    # distractor effect: paired, same samples and prompt p0
    for m in sorted({k[0] for k in preds}):
        a, b = preds.get((m, "clean", "p0")), preds.get((m, "distract", "p0"))
        if not a or not b:
            continue
        sids = sorted(set(a) & set(b))
        flips = sum((a[s] >= .5) != (b[s] >= .5) for s in sids)
        g = defaultdict(list)
        for s in sids:
            g[samples[s]["video"] + samples[s]["ped_id"]].append((int(samples[s]["label"]), (a[s], b[s])))
        acc = lambda yy, ss, i: sum((x[i] >= .5) == t for t, x in zip(yy, ss)) / len(yy)
        drop = lambda yy, ss: acc(yy, ss, 0) - acc(yy, ss, 1)
        rng, keys, vals = random.Random(SEED), list(g), []
        for _ in range(B):
            pick = [g[rng.choice(keys)] for _ in keys]
            vals.append(drop([t for x in pick for t, _ in x], [v for x in pick for _, v in x]))
        vals.sort()
        y = [int(samples[s]["label"]) for s in sids]
        print(f"{m}: distractors flip {flips}/{len(sids)} answers ({flips/len(sids):.1%}); "
              f"acc drop {drop(y, [(a[s], b[s]) for s in sids]):+.3f} "
              f"95% CI [{vals[int(.025*B)]:+.3f}, {vals[int(.975*B)-1]:+.3f}]")

    w = csv.DictWriter(open(HERE / "results.csv", "w", newline=""), fieldnames=out[0].keys())
    w.writeheader(); w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "output" / "preds")
