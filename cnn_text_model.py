# ===============================
# Configuration
# ===============================

DATA_PATH = "guardian_palestine_preprocessed.csv"

MAX_WORDS = 20000
MAX_LEN = 400
EMBEDDING_DIM = 100

BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path):
    import pandas as pd

    df = pd.read_csv(path)
    texts = df["processed_text"].astype(str)
    labels = df["label"].map({"Neutral": 0, "Biased": 1}).values

    return texts, labels



def prepare_cnn_input(texts, max_words, max_len):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return X, tokenizer



def build_cnn_model(max_words, max_len, embedding_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),

        Conv1D(filters=128, kernel_size=3, activation="relu"),
        GlobalMaxPooling1D(),

        Dense(64, activation="relu"),
        Dropout(0.5),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model



def train_model(model, X_train, y_train):
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    return history



def evaluate_model(model, X_test, y_test):
    import numpy as np
    from sklearn.metrics import classification_report, roc_auc_score

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    return y_prob



def plot_learning_curve(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("CNN Learning Curve")
    plt.savefig("CNN_LearningCurve.png")
    plt.show()



def main():
    from sklearn.model_selection import train_test_split

    texts, labels = load_data(DATA_PATH)

    X, tokenizer = prepare_cnn_input(texts, MAX_WORDS, MAX_LEN)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    model = build_cnn_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM)

    history = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_learning_curve(history)



if __name__ == "__main__":
    main()
