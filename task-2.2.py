import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

data = pd.read_json("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json", lines=True)

print(data.head())
print(data.columns)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])
similarity_matrix = cosine_similarity(X)
kmeans_2 = KMeans(n_clusters=2, random_state=42)
kmeans_2_labels = kmeans_2.fit_predict(similarity_matrix)
kmeans_5 = KMeans(n_clusters=5, random_state=42)
kmeans_5_labels = kmeans_5.fit_predict(similarity_matrix)
agg_2 = AgglomerativeClustering(n_clusters=2)
agg_2_labels = agg_2.fit_predict(similarity_matrix)
agg_5 = AgglomerativeClustering(n_clusters=5)
agg_5_labels = agg_5.fit_predict(similarity_matrix)
pca = PCA(n_components=2)
similarity_pca = pca.fit_transform(similarity_matrix)
plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=kmeans_2_labels, cmap='viridis')
plt.title("K-Means Clustering (2 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("kmeans_2_clusters.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=kmeans_5_labels, cmap='viridis')
plt.title("K-Means Clustering (5 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("kmeans_5_clusters.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=agg_2_labels, cmap='viridis')
plt.title("Agglomerative Clustering (2 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("agg_2_clusters.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=agg_5_labels, cmap='viridis')
plt.title("Agglomerative Clustering (5 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("agg_5_clusters.png", bbox_inches='tight')
plt.show()
