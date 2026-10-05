"""Parity test: scikit-learn pipeline vs. the plain-JavaScript port (docs/demo/bias.js).

Retrains the exact pipeline from export_model.py, predicts on the whole 25% test set (1,250
articles), then runs docs/demo/bias.js + docs/demo/model.json (via Node) on the same texts and
compares labels, decision scores and calibrated probabilities.

Usage:  python tools/export_model.py && python tools/parity_test.py      (needs Node >= 18)
Exit code is non-zero if label agreement is below 99.5%.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
TARGET = 0.995

df = pd.read_csv(ROOT / "guardian_text.csv").dropna(subset=["text", "label"])
X, y = df["text"].astype(str), df["label"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
    ("svm", LinearSVC(C=1)),
]).fit(X_train, y_train)

py_pred = model.predict(X_test)
py_score = model.decision_function(X_test)
platt = json.loads((ROOT / "docs/demo/model.json").read_text())["platt"]
py_proba = 1 / (1 + np.exp(-(platt["A"] * py_score + platt["B"])))

with tempfile.TemporaryDirectory() as tmp:
    inp, outp = Path(tmp, "texts.json"), Path(tmp, "js.json")
    inp.write_text(json.dumps(X_test.tolist(), ensure_ascii=False))
    subprocess.run(["node", str(ROOT / "tools/parity_run.js"), str(inp), str(outp)], check=True)
    js = json.loads(outp.read_text())

js_pred, js_score, js_proba = np.array(js["label"]), np.array(js["score"]), np.array(js["pNeutral"])
agree = int((js_pred == py_pred).sum())
n = len(py_pred)
print(f"Test articles:           {n}")
print(f"Label agreement:         {agree}/{n} = {agree / n:.4%}")
print(f"Max |score diff|:        {np.abs(js_score - py_score).max():.2e}")
print(f"Max |P(Neutral) diff|:   {np.abs(js_proba - py_proba).max():.2e}")
print(f"Python test accuracy:    {(py_pred == y_test.values).mean():.4%}")
print(f"JavaScript test accuracy:{(js_pred == y_test.values).mean():.4%}")
for i in np.where(js_pred != py_pred)[0][:5]:
    print(f"  mismatch #{i}: py={py_pred[i]} ({py_score[i]:+.4f})  js={js_pred[i]} ({js_score[i]:+.4f})")
sys.exit(0 if agree / n >= TARGET else 1)
