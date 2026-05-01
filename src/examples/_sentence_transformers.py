# Visualizing embeddings with sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A kitten rested on the rug",
    "Python is a programming language",
    "JavaScript is used for web development",
    "The weather is sunny today",
]

# Generate embeddings
embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (5, 384)

# Calculate cosine similarity between all pairs
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print("\nSemantic Similarity Matrix:")
print(f"{'':>40}", end="")
for i in range(len(sentences)):
    print(f"  [{i}]", end="")
print()

for i, s1 in enumerate(sentences):
    print(f"[{i}] {s1[:37]:>40}", end="")
    for j, s2 in enumerate(sentences):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f" {sim:.2f}", end="")
    print()