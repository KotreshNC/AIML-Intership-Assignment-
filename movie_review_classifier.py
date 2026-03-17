import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

reviews = [
    # Positive
    "I loved the movie, it was fantastic and full of heart.",
    "The acting was superb and the story moved me.",
    "Amazing visuals and the soundtrack was incredible.",
    "It was an excellent film with strong performances.",
    "A wonderful story and very well directed.",
    "I really enjoyed it and would watch again.",
    "The movie was good and had a few memorable scenes.",
    "A solid movie with a satisfying plot.",
    "The film had great pacing and a strong ending.",
    "A charming, heartfelt movie that made me smile.",
    # Neutral
    "It was okay overall, not great but not awful either.",
    "The movie had average performances and some good parts.",
    "I feel indifferent; the movie was neither exciting nor terrible.",
    "A middling film that had both strong and weak moments.",
    "Neutral story with average performances and some good parts.",
    "The story was fine, but nothing stood out strongly.",
    "A decent watch when you have time and nothing better.",
    # Negative
    "I disliked the movie and found it boring.",
    "The plot was weak, and the script felt recycled.",
    "Bad characters and poor direction ruined the experience.",
    "I did not connect with the film; it felt long and dull.",
    "The movie was a waste of time and predictable.",
    "Terrible acting and confusing storyline.",
    "I didn't like it; the jokes fell flat.",
    "It was just too slow and uninteresting for me.",
    "The editing was choppy and the pacing was off.",
    "The film had no emotional impact and felt flat.",
    "I walked out early because it was that bad.",
    "The script was sloppy and performances were weak.",
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))


def predict_review(text: str) -> str:
    """Return sentiment class for one sentence."""
    vect = vectorizer.transform([text])
    return model.predict(vect)[0]


def main():
    print("\n--- Sentiment Analysis (TF-IDF + Naive Bayes) ---")
    while True:
        text = input("Enter a movie review (or 'exit' to quit): ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not text:
            continue
        print("Predicted sentiment:", predict_review(text))


if __name__ == "__main__":
    # Show samples
    sample_reviews = [
        "The movie had breathtaking action and a heartfelt story.",
        "I fell asleep; it was a terrible and long movie.",
        "Some parts were okay but I felt nothing special.",
    ]
    print("Sample predictions:")
    for r in sample_reviews:
        print(f"  - {r} -> {predict_review(r)}")
    main()
