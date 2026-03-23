import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Dataset of 17 sentences on college life
sentences = [
    "Students attend classes every day.",
    "College life is full of assignments and exams.",
    "Many students live in dormitories.",
    "Professors teach various subjects.",
    "Libraries are great places for studying.",
    "Friends make college life fun.",
    "Cafeterias serve delicious food.",
    "Sports teams compete in tournaments.",
    "Clubs and societies organize events.",
    "Graduation is the ultimate goal.",
    "Lectures can be boring sometimes.",
    "Group projects help in learning.",
    "Part-time jobs help students earn money.",
    "Campus is beautiful with green spaces.",
    "Networking is important for future careers.",
    "Research opportunities are available.",
    "Internships provide real-world experience.",
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Print vocabulary
print("Vocabulary:", list(model.wv.key_to_index.keys()))

# Print vector for 'students'
print("\nVector for 'students':")
print(model.wv['students'])

# Print 5 most similar words to 'studying'
print("\n5 most similar words to 'studying':")
similar = model.wv.most_similar('studying', topn=5)
for word, sim in similar:
    print(f"{word}: {sim:.3f}")

# Save the model
model.save("word2vec_college_life.model")
print("\nModel saved to word2vec_college_life.model")