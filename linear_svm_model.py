import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

df = pd.read_csv("guardian_text.csv").dropna(subset=["text", "label"])
X = df["text"].astype(str)
y = df["label"].astype(str)

labels = sorted(y.unique())
if len(labels) != 2:
    raise ValueError(f"ROC/AUC needs 2 classes, found: {labels}")

pos_label = labels[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("svm", LinearSVC())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "svm__C": [0.1, 1, 3, 10]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("Best params:", grid.best_params_)
print("Best CV score (f1_macro):", grid.best_score_)

y_pred = best_model.predict(X_test)

print("\nTEST Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(values_format="d")
plt.title("Confusion Matrix - Linear SVM (TF-IDF)")
plt.show()

scores = best_model.decision_function(X_test)
y_test_bin = (y_test == pos_label).astype(int)

auc = roc_auc_score(y_test_bin, scores)
fpr, tpr, _ = roc_curve(y_test_bin, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Linear SVM (AUC = {auc:.3f})")
plt.show()

print("\nROC-AUC:", auc, "| Positive label:", pos_label)

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train, y_train,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 6),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

train_error = 1 - train_mean
val_error   = 1 - val_mean

plt.figure()
plt.plot(train_sizes, train_error, marker="o", label="Training error")
plt.plot(train_sizes, val_error, marker="o", label="Validation error")
plt.fill_between(train_sizes, (train_error - train_std), (train_error + train_std), alpha=0.2)
plt.fill_between(train_sizes, (val_error - val_std), (val_error + val_std), alpha=0.2)
plt.xlabel("Training set size")
plt.ylabel("Error (1 - F1-macro)")
plt.title("Learning Curve (Error) - Linear SVM (TF-IDF)")
plt.legend()
plt.show()

mis_idx = np.where(np.array(y_pred) != np.array(y_test))[0]
print(f"\nMisclassified: {len(mis_idx)} / {len(y_test)}")

for i in mis_idx[:3]:
    print("\n--- Misclassified example ---")
    print("True:", np.array(y_test)[i], "| Pred:", np.array(y_pred)[i])
    print("Text snippet:", X_test.iloc[i][:300].replace("\n", " "))
