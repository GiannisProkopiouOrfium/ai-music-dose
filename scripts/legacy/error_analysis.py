import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/sonics_balanced_ablation_mert_probe/ablation_mert.csv")

FEATURES = [c for c in df.columns if "id_" in c and any(s in c for s in ["phd", "twonn"])]

df = df.dropna(subset=FEATURES)
X = StandardScaler().fit_transform(df[FEATURES])
y = (df["label"] == "fake").astype(int)

# Collect OOF probabilities
proba = np.zeros(len(df))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    proba[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

df["pred_proba"] = proba
df["pred_label"] = (proba > 0.5).astype(int)
df["correct"] = (df["pred_label"] == y).astype(int)

# Top false positives (real tracks predicted most confidently as fake)
fp = df[(y == 0) & (df["pred_label"] == 1)].copy()
fp = fp.sort_values("pred_proba", ascending=False)
print("Top false positives (real → predicted fake):")
print(fp[["track_id", "path", "pred_proba", "algorithm", "fake_label"]].head(20))

# Top false negatives (fake tracks predicted most confidently as real)
fn = df[(y == 1) & (df["pred_label"] == 0)].copy()
fn = fn.sort_values("pred_proba", ascending=True)
print("\nTop false negatives (fake → predicted real):")
print(fn[["track_id", "path", "pred_proba", "algorithm", "fake_label"]].head(20))

# Error rate by algorithm × fake_label
err_tab = (
    df.groupby(["algorithm", "fake_label"])["correct"]
    .agg(n="count", accuracy="mean", errors=lambda x: (1 - x).sum())
    .reset_index()
)
print("\nError rates by stratum:")
print(err_tab.to_string(index=False))
