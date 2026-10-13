"""What does the VLM's score follow, and what predicts the real outcome?

1. Non-VLM baseline: logistic regression on JAAD annotations + box geometry,
   5-fold CV grouped by pedestrian (no pedestrian in both train and test).
2. Cue alignment: standardized logistic coefficients for the ground truth vs.
   a linear fit to each VLM's logit(p_yes). If the VLM tracks different cues than
   the outcome does, it is "right (or wrong) for the wrong reasons".
"""
import csv, glob, math, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
FEATS = ["looking", "walking", "designated", "signalized", "intersection", "group",
         "box_h", "x_center", "y_bottom", "dx_toward_center", "tte_s"]


def features(r):
    b = [tuple(map(float, x.split(","))) for x in r["boxes"].split()]
    x1, y1, x2, y2 = b[-1]
    cx0 = (b[0][0] + b[0][2]) / 2
    cx = (x1 + x2) / 2
    toward = (abs(cx0 - 960) - abs(cx - 960)) / 1920  # moving toward image centre (the road ahead)
    return [
        float(r["look"] == "looking"), float(r["action"] == "walking"),
        float(r["designated"] == "D"), float(r["signalized"] == "S"),
        float(r["intersection"] == "yes"), float(r["group_size"] or 1),
        (y2 - y1) / 1080, abs(cx - 960) / 960, y2 / 1080, toward, int(r["tte"]) / 30,
    ]


def main(pred_dir):
    S = list(csv.DictReader(open(HERE / "samples.csv")))
    ids = [r["sample_id"] for r in S]
    X = np.array([features(r) for r in S])
    y = np.array([int(r["label"]) for r in S])
    g = np.array([r["video"] + r["ped_id"] for r in S])
    Xs = StandardScaler().fit_transform(X)

    # 1. grouped-CV baseline
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, g):
        oof[te] = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    rng = np.random.default_rng(0)
    keys = np.unique(g)
    boots = []
    for _ in range(2000):
        pick = rng.choice(keys, len(keys))
        idx = np.concatenate([np.where(g == k)[0] for k in pick])
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Cue baseline (logistic, 11 features, grouped 5-fold CV): AUC {roc_auc_score(y, oof):.3f} [{lo:.3f}, {hi:.3f}]")

    # fairer: box geometry and motion only (everything a model could get from pixels + the red box)
    geo = [FEATS.index(f) for f in ("box_h", "x_center", "y_bottom", "dx_toward_center", "tte_s")]
    oof2 = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, g):
        oof2[te] = LogisticRegression(max_iter=1000).fit(Xs[tr][:, geo], y[tr]).predict_proba(Xs[te][:, geo])[:, 1]
    b2 = []
    for _ in range(2000):
        pick = rng.choice(keys, len(keys))
        idx = np.concatenate([np.where(g == k)[0] for k in pick])
        if len(set(y[idx])) == 2:
            b2.append(roc_auc_score(y[idx], oof2[idx]))
    lo2, hi2 = np.percentile(b2, [2.5, 97.5])
    print(f"Geometry-only baseline (5 box features): AUC {roc_auc_score(y, oof2):.3f} [{lo2:.3f}, {hi2:.3f}]")

    gt = LogisticRegression(max_iter=1000).fit(Xs, y).coef_[0]
    rows = [dict(feature=f, ground_truth=round(c, 3)) for f, c in zip(FEATS, gt)]

    # 2. cue alignment per VLM (clean, prompt p2 if present else p0)
    preds = {}
    for path in glob.glob(f"{pred_dir}/*.csv"):
        for r in csv.DictReader(open(path)):
            if r["cond"] == "clean" and r["p_yes"] != "nan":
                preds.setdefault((r["model"], r["prompt"]), {})[r["sample_id"]] = float(r["p_yes"])
    for (m, p), d in sorted(preds.items()):
        if p != "p2" or not all(i in d for i in ids):
            continue
        s = np.clip(np.array([d[i] for i in ids]), 1e-4, 1 - 1e-4)
        z = np.log(s / (1 - s))
        coef = LinearRegression().fit(Xs, (z - z.mean()) / z.std()).coef_
        r = np.corrcoef(coef, gt)[0, 1]
        print(f"{m} ({p}): AUC {roc_auc_score(y, s):.3f}; cue-weight correlation with ground truth r={r:+.2f}")
        for row, c in zip(rows, coef):
            row[f"{m}"] = round(c, 3)

    with open(HERE / "cue_weights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("\nstandardized weights (ground truth = logistic; models = linear fit to logit p_yes):")
    for row in rows:
        print("  ".join(f"{v:>14}" if isinstance(v, str) else f"{v:>+14.3f}" for v in row.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "output" / "preds")
