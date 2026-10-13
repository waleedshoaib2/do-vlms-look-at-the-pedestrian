"""Linear probes on Qwen2.5-VL-7B hidden states (v2 set, 433 independent pedestrians).

Is crossing intent linearly decodable from the model's internal features, even though
its Yes/No answer barely tracks it? And does that decodability depend on the pedestrian
(clean vs. inpainted-out), or only on the box location (blank grey frame)?

Features per sample, condition and layer: the last-token hidden state and the mean of the
image-token hidden states. L2 logistic regression with the C chosen by inner CV, evaluated
with 5-fold stratified CV repeated 5 times (all samples are different pedestrians).
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output_v2" / "preds_v2"
S = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "samples_v2.csv"))}


def cv_auc(X, y, reps=5):
    aucs = []
    for rep in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=3, max_iter=2000))
            oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


rows = []
for cond in ["clean", "mask", "blank", "crop"]:
    f = D / f"q7_probe_{cond}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    ids = list(z["ids"])
    y = np.array([int(S[i]["label"]) for i in ids])
    for kind in ("last", "img_mean"):
        X_all = z[kind].astype(np.float32)
        for li, layer in enumerate(z["layers"]):
            m, s = cv_auc(X_all[:, li], y)
            rows.append(dict(cond=cond, feature=kind, layer=int(layer), auc=round(m, 3), sd=round(s, 3)))
            print(rows[-1], flush=True)

if rows:
    with open(HERE / "probe_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
