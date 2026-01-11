import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    csv_file = "guardian_text.csv"
    df = pd.read_csv(csv_file).dropna(subset=["text", "label"])

    print("\n=== Dataset loaded ===")
    print("File:", csv_file)
    print("Shape (rows, cols):", df.shape)
    print("\nColumns:", list(df.columns))

    print("\n=== Label distribution ===")
    print(df["label"].value_counts())

    print("\n=== Sample rows (text truncated) ===")
    sample = df[["text", "label"]].head(3).copy()
    sample["text"] = sample["text"].astype(str).str.slice(0, 120) + "..."
    print(sample.to_string(index=False))

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("\n=== Split ===")
    print("Train size:", len(X_train))
    print("Test size :", len(X_test))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n=== TF-IDF Matrix ===")
    print("X_train_vec shape:", X_train_vec.shape)
    print("X_test_vec  shape:", X_test_vec.shape)
    print("Matrix type:", type(X_train_vec))
    print("Note: It is a sparse matrix (most values are zero).")

    feature_names = vectorizer.get_feature_names_out()

    first_vec = X_train_vec[0].toarray().ravel()  # convert 1xN sparse -> dense 1D
    print("\n=== First training article (as numbers) ===")
    print("First 20 TF-IDF values:")
    print(first_vec[:20])

    print("\nFirst 20 feature names (corresponding dimensions):")
    print(feature_names[:20])

    mini_df = pd.DataFrame({
        "feature": feature_names[:20],
        "tfidf_value": first_vec[:20]
    })

    print("\n=== Mini view (first 20 dimensions) ===")
    print(mini_df.to_string(index=False))

    top_idx = np.argsort(first_vec)[::-1]
    top_idx = [i for i in top_idx if first_vec[i] > 0][:15]

    print("\n=== Top TF-IDF features for the first article ===")
    for i in top_idx:
        print(f"{feature_names[i]} : {first_vec[i]:.4f}")

    print("\nDone ")

if __name__ == "__main__":
    main()
