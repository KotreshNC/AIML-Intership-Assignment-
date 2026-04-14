"""
Fake News Detector Mini App

This script trains a logistic regression model to classify news articles as real or fake using TF-IDF vectorization.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase, removing punctuation, and stripping whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def build_dataset():
    """
    Build a sample dataset of real and fake news articles.
    Returns lists of texts and corresponding labels (0=real, 1=fake).
    """
    texts = [
        "Scientists discover new planet in solar system.",  # real
        "Aliens invade Earth, government confirms.",  # fake
        "Stock market rises due to economic recovery.",  # real
        "Celebrity claims to be immortal after drinking magic potion.",  # fake
        "New vaccine shows 95% effectiveness against virus.",  # real
        "Moon landing was faked, insider reveals.",  # fake
        "Company announces record profits this quarter.",  # real
        "Zombie apocalypse starts next week, prepare now.",  # fake
        "Researchers find cure for common cold.",  # real
        "Time travel machine invented by secret lab.",  # fake
        "Election results show clear winner.",  # real
        "Dinosaurs still alive in hidden jungle.",  # fake
        "Tech giant releases new smartphone.",  # real
        "Ghosts haunt White House, president admits.",  # fake
        "Climate change impacts discussed at summit.",  # real
        "Unicorns found in backyard, video proof.",  # fake
    ]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0=real, 1=fake
    return texts, labels

def train_and_evaluate():
    """
    Train the model on the dataset and evaluate its performance.
    Returns the trained model and vectorizer.
    """
    texts, labels = build_dataset()
    processed_texts = [preprocess_text(t) for t in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(processed_texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("--- Evaluation on test split ---")
    print(classification_report(y_test, preds, target_names=['real', 'fake']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return model, vectorizer

def predict_fake_news(model, vectorizer, text):
    """
    Predict whether the given text is real or fake news.
    Returns the prediction label and confidence score.
    """
    processed = preprocess_text(text)
    X_new = vectorizer.transform([processed])
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0]
    return "Fake" if pred == 1 else "Real", prob[pred]

if __name__ == "__main__":
    print("Training Fake News Detector...")
    model, vectorizer = train_and_evaluate()

    print("\n--- Demo Predictions ---")
    test_texts = [
        "New study shows benefits of exercise.",
        "Elvis Presley spotted at grocery store.",
    ]
    for text in test_texts:
        label, confidence = predict_fake_news(model, vectorizer, text)
        print(f"Text: {text}")
        print(f"Prediction: {label} (confidence: {confidence:.2f})")
        print()