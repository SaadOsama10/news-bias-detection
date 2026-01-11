Bias Detection in News Articles Using Machine Learning
📌 Project Overview

This project focuses on detecting biased language in news articles using machine learning and deep learning techniques.
We address the problem as a binary text classification task, where each article is classified as either Biased or Neutral.

📂 Dataset

Source: The Guardian (via official Content API)

Topic: Israel–Palestine related articles

Labels:

Biased → Opinion articles (Comment is Free section)

Neutral → News reporting articles (World, Middle East, US News, etc.)

Dataset Characteristics:

Binary classification

Balanced classes

Articles with at least 60 words

File format: CSV

⚠️ Note: Labels are derived from section metadata, not manual annotation, which may introduce minor label noise.

⚙️ Feature Engineering

Different feature representations were used depending on the model:

Traditional Machine Learning Models

TF-IDF Vectorization

Vocabulary size: 5,000

Unigrams and bigrams

Stop-word removal

Deep Learning Model (CNN)

Tokenization and sequence encoding

Fixed sequence length with padding and truncation

Trainable word embeddings

🧠 Implemented Models
Category A – Traditional ML

Linear Support Vector Machine (SVM)

Logistic Regression

Perceptron

All trained using TF-IDF features and evaluated with 5-fold stratified cross-validation.

Category B – Deep Learning

Convolutional Neural Network (CNN)

Embedding layer

1D Convolution layer

Global Max Pooling

Fully connected layers

Dropout regularization

Binary cross-entropy loss

📊 Evaluation Metrics

Models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC Curve & ROC-AUC

Learning Curves

📈 Results Summary
Model	Accuracy	F1-score
Linear SVM	0.971	0.97
Logistic Regression	0.963	0.96
Perceptron	0.961	0.96
CNN	0.940	0.94

Linear SVM achieved the best overall performance.

CNN demonstrated strong discriminative ability (high ROC-AUC) but slightly lower accuracy due to dataset size and model complexity.

🧩 Key Insights

Linear models perform exceptionally well with TF-IDF features in high-dimensional text data.

CNNs can capture contextual and semantic patterns but require larger datasets to consistently outperform linear approaches.

Model selection should depend on data size, feature representation, and task requirements, not accuracy alone.


🚀 How to Run

Clone the repository: git clone https://github.com/ABDULRAHMANszn/Bias_Detection_in_News
Install dependencies: pip install -r requirements.txt
Run traditional ML models: python linear_svm_model.py or linear_svm_model.ipynb
python LogisticRegression.py or LogisticRegression.ipynb
python Perceptron.py or Perceptron.ipynb
Run CNN model: Open cnn_text_model.ipynb Run all cells sequentially

🔮 Future Work

Use larger and more diverse datasets

Explore transformer-based models (e.g., BERT)

Incorporate linguistic and discourse-level features

Perform manual annotation for more precise labels
