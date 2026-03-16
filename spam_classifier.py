from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def build_dataset():
    # Tiny toy dataset for spam detection
    texts = [
        "Win a FREE iPhone now! Click here to claim your prize",
        "Your account has been suspended. Verify your password immediately",
        "Get 50% off on your next purchase. Limited time offer",
        "Meeting at 3pm tomorrow, please confirm your availability",
        "Can we reschedule the call to Friday?",
        "Project update: we completed the data pipeline and tests",
        "Congratulations! You have won a lottery. Send details to collect funds",
        "Please review the attached agenda for the team call",
        "Earn money quickly from home with this simple trick",
        "Hi, here is the report from last week for your review"
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1=spam, 0=ham
    return texts, labels


def train_and_evaluate():
    texts, labels = build_dataset()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, texts, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("--- Evaluation on test split ---")
    print(classification_report(y_test, preds, target_names=['ham', 'spam']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return model, vectorizer


def predict_messages(model, vectorizer, messages):
    X_new = vectorizer.transform(messages)
    pred = model.predict(X_new)
    probs = model.predict_proba(X_new)
    print("\n--- Prediction examples ---")
    for msg, p, pr in zip(messages, pred, probs):
        label = 'spam' if p == 1 else 'ham'
        print(f"Message: {msg}")
        print(f"Predicted: {label} (spam_prob={pr[1]:.3f})\n")


def main():
    model, vectorizer = train_and_evaluate()

    new_messages = [
        "Urgent: your payment is overdue. Pay now to avoid penalty",
        "Can we meet for lunch tomorrow at the cafe?",
        "Limited-time cheap loans available with instant approval",
        "Please find attached the final invoice and let me know if any changes"
    ]
    predict_messages(model, vectorizer, new_messages)


if __name__ == '__main__':
    main()
