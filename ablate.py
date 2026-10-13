"""Paired counterfactual analysis for the qwenabl run (prompt p2).

For each edit vs. the original frame (same sample, same model process):
  AUC, mean shift in P(yes), mean |shift|, rank correlation with the original,
  and 95% cluster-bootstrap CIs (by pedestrian) for the AUC change and the mean shift.
Also: is the 'near' distractor worse than 'far' (paired on samples that have both)?
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output" / "preds"
S = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "samples.csv"))}
P = defaultdict(dict)
for r in csv.DictReader(open(PRED / "qwenabl.csv")):
    if r["p_yes"] != "nan":
        P[r["cond"]][r["sample_id"]] = float(r["p_yes"])
base = P["clean"]
rng = np.random.default_rng(0)


def boot(sids, fn, B=2000):
    g = defaultdict(list)
    for s in sids:
        g[S[s]["video"] + S[s]["ped_id"]].append(s)
    keys, vals = list(g), []
    for _ in range(B):
        pick = [s for k in rng.choice(keys, len(keys)) for s in g[k]]
        try:
            vals.append(fn(pick))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])


y = lambda ss: [int(S[s]["label"]) for s in ss]
out = []
for c in ["clean", "mask", "boxshift", "blank", "near", "far", "n4"]:
    if c not in P:
        continue
    ss = sorted(set(P[c]) & set(base))
    a = np.array([base[s] for s in ss]); b = np.array([P[c][s] for s in ss])
    auc = roc_auc_score(y(ss), b)
    dauc = auc - roc_auc_score(y(ss), a)
    lo, hi = boot(ss, lambda p: roc_auc_score(y(p), [P[c][s] for s in p]) - roc_auc_score(y(p), [base[s] for s in p]))
    slo, shi = boot(ss, lambda p: np.mean([P[c][s] - base[s] for s in p]))
    rho = spearmanr(a, b)[0] if c != "clean" else 1.0
    out.append(dict(cond=c, n=len(ss), auc=round(auc, 3), d_auc=round(dauc, 3), d_auc_ci=f"[{lo:+.3f}, {hi:+.3f}]",
                    mean_shift=round(float(np.mean(b - a)), 4), shift_ci=f"[{slo:+.4f}, {shi:+.4f}]",
                    mean_abs_shift=round(float(np.mean(np.abs(b - a))), 4), spearman_vs_original=round(float(rho), 3)))

for r in out:
    print(r)
with open(HERE / "ablation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)

if "near" in P and "far" in P:
    ss = sorted(set(P["near"]) & set(P["far"]) & set(base))
    d = lambda p: np.mean([abs(P["near"][s] - base[s]) - abs(P["far"][s] - base[s]) for s in p])
    lo, hi = boot(ss, d)
    print(f"near vs far: mean |shift| difference {d(ss):+.4f} [{lo:+.4f}, {hi:+.4f}] on {len(ss)} paired samples")
