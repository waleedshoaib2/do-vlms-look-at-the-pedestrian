"""v2 analysis: 433 independent pedestrians (161 crossers, 122 behavioral non-crossers, 150 bystanders).

For every model key and condition: AUC overall, AUC on the hard split (crossers vs. behavioral
non-crossers) and the easy split (crossers vs. bystanders), plus paired differences against the
same model's 'clean' condition. Samples are independent, so the bootstrap resamples samples.
"""
import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output_v2" / "preds_v2"
S = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "samples_v2.csv"))}
rng = np.random.default_rng(0)
B = 2000


def split(ids, which):
    if which == "all":
        return ids
    neg = "beh_noncross" if which == "hard" else "bystander"
    return [i for i in ids if S[i]["group"] in ("crosser", neg)]


def auc(ids, d):
    return roc_auc_score([int(S[i]["label"]) for i in ids], [d[i] for i in ids])


def boot_ci(ids, fn):
    ids = np.array(ids)
    v = []
    for _ in range(B):
        s = list(rng.choice(ids, len(ids)))
        try:
            v.append(fn(s))
        except ValueError:
            pass
    return np.percentile(v, [2.5, 97.5])


P = defaultdict(dict)
for path in glob.glob(str(D / "*.csv")):
    for r in csv.DictReader(open(path)):
        if r["p_yes"] != "nan":
            P[(r["model"], r["cond"])][r["sample_id"]] = float(r["p_yes"])

# geometry baseline on v2 (trained, 5-fold CV)
def geo(r):
    b = [tuple(map(float, x.split(","))) for x in r["boxes"].split()]
    x1, y1, x2, y2 = b[-1]
    cx0, cx = (b[0][0] + b[0][2]) / 2, (x1 + x2) / 2
    return [(y2 - y1) / 1080, abs(cx - 960) / 960, y2 / 1080, (abs(cx0 - 960) - abs(cx - 960)) / 1920, int(r["tte"]) / 30]
ids = sorted(S)
X = StandardScaler().fit_transform(np.array([geo(S[i]) for i in ids])); y = np.array([int(S[i]["label"]) for i in ids])
oof = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    oof[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
G = dict(zip(ids, oof))
out = []
for which in ("all", "hard", "easy"):
    ss = split(ids, which)
    lo, hi = boot_ci(ss, lambda s: auc(s, G))
    out.append(dict(model="geometry_LR_trained", cond="-", split=which, n=len(ss), auc=round(auc(ss, G), 3),
                    ci=f"[{lo:.3f}, {hi:.3f}]", d_vs_clean="", d_ci=""))

for (m, c), d in sorted(P.items()):
    base = P.get((m, "clean"))
    for which in ("all", "hard", "easy"):
        ss = [i for i in split(ids, which) if i in d]
        if len(ss) < 20:
            continue
        lo, hi = boot_ci(ss, lambda s: auc(s, d))
        row = dict(model=m, cond=c, split=which, n=len(ss), auc=round(auc(ss, d), 3), ci=f"[{lo:.3f}, {hi:.3f}]",
                   d_vs_clean="", d_ci="")
        if base and c != "clean":
            pp = [i for i in ss if i in base]
            dl, dh = boot_ci(pp, lambda s: auc(s, d) - auc(s, base))
            row.update(d_vs_clean=round(auc(pp, d) - auc(pp, base), 3), d_ci=f"[{dl:+.3f}, {dh:+.3f}]")
        out.append(row)

with open(HERE / "results_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
for r in out:
    if r["split"] == "all" or r["model"].startswith("geometry"):
        print(r)
