"""Retrain the repo's best model and export it for the in-browser demo.

Pipeline (identical to linear_svm_model.py's grid-search winner):
    TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    LinearSVC(C=1)
    75/25 stratified split, random_state=42.

On top of that, Platt scaling (a 1-D logistic fit on the SVM decision score) is fitted on
out-of-fold training scores, so the probability shown in the demo is calibrated while the
label still comes from the exact SVM.

Outputs:
    docs/demo/model.json          model for the browser

Usage:  python tools/export_model.py [--grid]     (--grid re-runs the full grid search first)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
PARAMS = {"tfidf__max_features": 5000, "tfidf__ngram_range": (1, 2), "svm__C": 1}


def make_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("svm", LinearSVC(C=1)),
    ])


df = pd.read_csv(ROOT / "guardian_text.csv").dropna(subset=["text", "label"])
X, y = df["text"].astype(str), df["label"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

if "--grid" in sys.argv:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("svm", LinearSVC())])
    grid = GridSearchCV(pipe, {"tfidf__max_features": [5000, 10000], "tfidf__ngram_range": [(1, 1), (1, 2)],
                               "svm__C": [0.1, 1, 3, 10]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Grid best:", grid.best_params_, "CV f1_macro:", grid.best_score_)

model = make_pipe().fit(X_train, y_train)
vec, svm = model.named_steps["tfidf"], model.named_steps["svm"]
pred = model.predict(X_test)
scores = model.decision_function(X_test)
classes = list(svm.classes_)            # ['Biased', 'Neutral']; positive class = classes[1]
acc = accuracy_score(y_test, pred)
auc = roc_auc_score((y_test == classes[1]).astype(int), scores)
print(f"Python test accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}  (n={len(y_test)})")

# Platt scaling on out-of-fold decision scores of the training set -> P(Neutral | score)
oof = cross_val_predict(make_pipe(), X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="decision_function")
platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), (y_train == classes[1]).astype(int))
A, B = float(platt.coef_[0][0]), float(platt.intercept_[0])
proba = 1 / (1 + np.exp(-(A * scores + B)))
brier = float(np.mean((proba - (y_test == classes[1]).astype(int)) ** 2))
print(f"Platt: P({classes[1]}) = sigmoid({A:.4f}*score + {B:.4f}); test Brier = {brier:.4f}")

vocab = vec.vocabulary_
terms = [None] * len(vocab)
for t, i in vocab.items():
    terms[i] = t
r = lambda a, n: [round(float(v), n) for v in a]
out = {
    "classes": classes,
    "vectorizer": {
        "lowercase": vec.lowercase, "strip_accents": vec.strip_accents,
        "token_pattern": vec.token_pattern, "ngram_range": list(vec.ngram_range),
        "sublinear_tf": vec.sublinear_tf, "norm": vec.norm, "smooth_idf": vec.smooth_idf,
        "use_idf": vec.use_idf, "max_features": vec.max_features,
        "stop_words": sorted(vec.get_stop_words()),
    },
    "terms": terms,
    "idf": r(vec.idf_, 5),
    "coef": r(svm.coef_[0], 5),
    "intercept": round(float(svm.intercept_[0]), 6),
    "platt": {"A": round(A, 6), "B": round(B, 6)},
    "meta": {"test_accuracy": round(acc, 4), "roc_auc": round(auc, 4), "n_test": int(len(y_test)),
             "n_train": int(len(y_train)), "sklearn": __import__("sklearn").__version__},
}
(ROOT / "docs/demo/model.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))

print("wrote docs/demo/model.json,", (ROOT / "docs/demo/model.json").stat().st_size, "bytes")
