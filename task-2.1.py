import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

matplotlib.use('Agg')

data = []
with open("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

data_sample = df.sample(frac=0.5, random_state=42)

cuisines = ["Indian", "Italian", "Chinese", "Mexican", "Japanese"]

cuisine_reviews = {cuisine: [] for cuisine in cuisines}

for index, row in data_sample.iterrows():
    for cuisine in cuisines:
        if cuisine.lower() in row['text'].lower():
            cuisine_reviews[cuisine].append(row['text'])

for cuisine in cuisines:
    cuisine_reviews[cuisine] = " ".join(cuisine_reviews[cuisine])

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(cuisine_reviews.values())

similarity_matrix = cosine_similarity(X)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, xticklabels=cuisines, yticklabels=cuisines, cmap="Blues", cbar=True)
plt.title("Cuisine Similarity Matrx")
plt.xlabel("Cuisines")
plt.ylabel("Cuisines")

plt.savefig("cuisine_similarity_matrix.png")

plt.show()
