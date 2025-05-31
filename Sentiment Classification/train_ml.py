import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Sample Dataset
data = [
    ("I loved this movie, it was fantastic!", 1),
    ("What a terrible film, I hated it.", 0),
    ("It was an amazing experience, really enjoyed it.", 1),
    ("Worst movie ever, do not waste your time.", 0),
    ("Great acting and wonderful plot!", 1),
    ("The film was dull and boring.", 0),
    ("Absolutely brilliant, would watch again.", 1),
    ("Not good, the story was weak.", 0)
]

texts, labels = zip(*data)
labels = np.array(labels)

# Split data into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train Logistic Regression with higher max_iter for convergence
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, train_labels)

# Predict and evaluate
lr_preds = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(test_labels, lr_preds))
print("Logistic Regression Classification Report:")
print(classification_report(test_labels, lr_preds, zero_division=0))

# Optional: Debug classes present
print("True labels in test set:", set(test_labels))
print("Predicted labels:", set(lr_preds))
