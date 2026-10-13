"""Figures for the write-up. Works with whatever prediction files exist."""
import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output" / "preds"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"qwen25vl7b": "Qwen2.5-VL-7B", "llavaov7b": "LLaVA-OV-7B", "smolvlm2b": "SmolVLM-2B", "qwenabl": "Qwen2.5-VL-7B"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

S = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "samples.csv"))}
P = defaultdict(dict)
for path in glob.glob(str(PRED / "*.csv")):
    for r in csv.DictReader(open(path)):
        if r["p_yes"] != "nan":
            P[(r["model"], r["cond"], r["prompt"])][r["sample_id"]] = float(r["p_yes"])
groups = defaultdict(list)
for sid, r in S.items():
    groups[r["video"] + r["ped_id"]].append(sid)
rng = np.random.default_rng(0)


def auc_ci(d, B=1000):
    sids = list(d)
    y = np.array([int(S[s]["label"]) for s in sids]); s = np.array([d[x] for x in sids])
    pos = {x: i for i, x in enumerate(sids)}
    keys = [k for k in groups if any(x in pos for x in groups[k])]
    vals = []
    for _ in range(B):
        idx = [pos[x] for k in rng.choice(keys, len(keys)) for x in groups[k] if x in pos]
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return roc_auc_score(y, s), *np.percentile(vals, [2.5, 97.5])


# 1. AUC by model and prompt, against chance and the trained baselines
fig, ax = plt.subplots(figsize=(7, 3.4))
models = [m for m in ("qwen25vl7b", "llavaov7b", "smolvlm2b") if (m, "clean", "p0") in P]
conds = [("clean", "p0"), ("clean", "p1"), ("clean", "p2"), ("distract", "p0"), ("clip", "clip")]
labels = ["prompt 1", "prompt 2", "prompt 3", "+distractors", "4-frame clip"]
w = 0.8 / len(conds)
for j, c in enumerate(conds):
    xs, ys, lo, hi = [], [], [], []
    for i, m in enumerate(models):
        if (m, *c) in P:
            a, l, h = auc_ci(P[(m, *c)])
            xs.append(i + (j - len(conds) / 2 + 0.5) * w); ys.append(a); lo.append(a - l); hi.append(h - a)
    ax.bar(xs, ys, w * 0.9, yerr=[lo, hi], capsize=2, label=labels[j])
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.text(0.01, 0.5, "chance", fontsize=8, ha="left", va="bottom", transform=ax.get_yaxis_transform())
for val, name in ((0.768, "box-geometry LR (trained)"), (0.908, "annotated-cue LR (trained)")):
    ax.axhline(val, color="gray", lw=0.8, ls="--")
    ax.text(0.99, val + 0.005, name, fontsize=8, color="gray", ha="right", transform=ax.get_yaxis_transform())
ax.set_xticks(range(len(models)), [NAMES[m] for m in models])
ax.set_ylabel("AUC (crossing vs. not)"); ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)
ax.set_title("Zero-shot VLMs vs. simple trained baselines on JAAD\n(95% CI, bootstrap by pedestrian)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "auc_by_model.png", dpi=200); plt.close(fig)

# 2. Distractor shift in P(yes), paired per sample
fig, ax = plt.subplots(figsize=(6, 3))
for m in models:
    a, b = P.get((m, "clean", "p0")), P.get((m, "distract", "p0"))
    if a and b:
        d = [b[s] - a[s] for s in a if s in b]
        ax.hist(d, bins=40, alpha=0.55, label=f"{NAMES[m]} (mean {np.mean(d):+.3f})")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("P(yes) with distractors − P(yes) clean, same scene"); ax.set_ylabel("samples")
ax.legend(fontsize=8, frameon=False); ax.set_title("Irrelevant pedestrians shift the models' confidence", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "distractor_shift.png", dpi=200); plt.close(fig)

# 3. Counterfactual ablations (qwenabl)
abl = ["clean", "mask", "boxshift", "blank", "near", "far", "n4"]
alab = ["original", "target\nremoved", "box moved\nto empty", "grey frame\n+ box", "1 distractor\nnear", "1 distractor\nfar", "4\ndistractors"]
if ("qwenabl", "clean", "p2") in P:
    base = P[("qwenabl", "clean", "p2")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    aucs, shifts, keep = [], [], []
    for c, l in zip(abl, alab):
        d = P.get(("qwenabl", c, "p2"))
        if not d:
            continue
        keep.append(l); aucs.append(auc_ci(d))
        shifts.append([d[s] - base[s] for s in d if s in base])
    a1.errorbar(range(len(keep)), [x[0] for x in aucs],
                yerr=[[x[0] - x[1] for x in aucs], [x[2] - x[0] for x in aucs]], fmt="o", capsize=3)
    a1.axhline(0.5, color="k", ls=":", lw=0.8); a1.set_xticks(range(len(keep)), keep, fontsize=7)
    a1.set_ylabel("AUC"); a1.set_title("Does the answer depend on the pedestrian?", fontsize=10)
    a2.boxplot(shifts, showfliers=False); a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(range(1, len(keep) + 1), keep, fontsize=7)
    a2.set_ylabel("Δ P(yes) vs. original"); a2.set_title("How much each edit moves the score", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "ablations.png", dpi=200); plt.close(fig)

# 4. Cue weights: ground truth vs. models
cw = HERE / "cue_weights.csv"
if cw.exists():
    rows = list(csv.DictReader(open(cw)))
    cols = [c for c in rows[0] if c != "feature"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(cols)
    for j, c in enumerate(cols):
        v = np.array([float(r[c]) for r in rows]); v = v / np.linalg.norm(v)  # compare direction, not scale
        ax.barh([i + (j - len(cols) / 2 + 0.5) * w for i in range(len(rows))],
                v, w * 0.9, label="ground truth (logistic)" if c == "ground_truth" else NAMES.get(c, c))
    ax.set_yticks(range(len(rows)), [r["feature"] for r in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("standardized weight (each set scaled to unit length)")
    ax.legend(fontsize=8, frameon=False); ax.set_title("Which cues predict crossing vs. which cues move the VLM", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cue_weights.png", dpi=200); plt.close(fig)

print("figures:", sorted(p.name for p in FIG.glob("*.png")))
