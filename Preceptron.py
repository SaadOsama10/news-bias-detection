import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("guardian_palestine_preprocessed.csv")
df.head()

X = df["processed_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(
    stop_words="english", 
    max_features=5000, 
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = Perceptron(max_iter=2000, random_state=42)
model.fit(X_train_vec, y_train)

predict_model = model.predict(X_test_vec)
print("Perceptron Accuracy:", accuracy_score(y_test, predict_model))
print("Classification Report:")
print(classification_report(y_test, predict_model))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predict_model))