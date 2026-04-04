"""
Word Importance Explorer using TF-IDF

This program demonstrates the use of TF-IDF (Term Frequency-Inverse Document Frequency)
to identify important keywords in a collection of documents.

TF-IDF Explanation:
- TF (Term Frequency): Measures how frequently a term appears in a document.
  It's the ratio of the number of times a term appears in a document to the total
  number of terms in that document.

- IDF (Inverse Document Frequency): Measures how important a term is across the
  entire corpus. It's calculated as log(N/df), where N is the total number of
  documents, and df is the number of documents containing the term.

- TF-IDF Score: TF * IDF. Higher scores indicate terms that are frequent in a
  specific document but rare across the entire corpus, making them more important
  for that document.

This program uses 5 sample documents, computes TF-IDF scores, and identifies the
top keywords for each document.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample documents (you can replace these with your own documents)
documents = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
    "Natural language processing involves the interaction between computers and humans using natural language.",
    "Data science combines statistics, programming, and domain expertise to extract insights from data.",
    "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
    "Computer vision allows machines to interpret and understand visual information from the world."
]

# Create TF-IDF vectorizer
# - stop_words='english': Remove common English stop words
# - max_features=None: Consider all features
# - use_idf=True: Use IDF weighting
vectorizer = TfidfVectorizer(stop_words='english', max_features=None, use_idf=True)

# Fit the vectorizer to the documents and transform them into TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

print("Word Importance Explorer using TF-IDF")
print("=" * 50)
print(f"Number of documents: {len(documents)}")
print(f"Number of unique words: {len(feature_names)}")
print()

# For each document, find the top keywords
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(f"Text: {doc}")
    print("Top keywords:")

    # Get the TF-IDF scores for this document
    doc_scores = tfidf_matrix[i].toarray().flatten()

    # Get indices of top scores (sorted in descending order)
    top_indices = np.argsort(doc_scores)[::-1][:5]  # Top 5 keywords

    for rank, idx in enumerate(top_indices, 1):
        word = feature_names[idx]
        score = doc_scores[idx]
        if score > 0:  # Only print words with positive scores
            print(f"{rank}. {word}: {score:.4f}")

    print("-" * 50)
    print()

print("Explanation:")
print("The keywords listed above have the highest TF-IDF scores for each document.")
print("These words are important because they appear frequently in the specific document")
print("but are relatively rare across all documents, making them distinctive for that topic.")