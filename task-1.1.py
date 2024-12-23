import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import os

file_path = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"  # Update to your file path
reviews = []
with open(file_path, "r") as file:
    for line in file:
        review = json.loads(line)
        reviews.append(review)
data = pd.DataFrame(reviews)


def preprocess_reviews(data, min_stars=1, max_stars=5):
    filtered_data = data[(data['stars'] >= min_stars) & (data['stars'] <= max_stars)]
    return filtered_data['text'].tolist()

all_reviews = preprocess_reviews(data.sample(frac=0.0001, random_state=42))  # Use 60% of data
positive_reviews = preprocess_reviews(data[data['stars'] >= 4].sample(frac=0.0001, random_state=42))
negative_reviews = preprocess_reviews(data[data['stars'] <= 2].sample(frac=0.0001, random_state=42))

def perform_lda(reviews, num_topics=5, save_file="lda_vis.html"):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(reviews)

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)

    lda_vis_data = pyLDAvis.prepare(
        topic_term_dists=lda_model.components_,
        doc_topic_dists=lda_model.transform(dtm),
        doc_lengths=dtm.sum(axis=1).A1,
        vocab=vectorizer.get_feature_names_out(),
        term_frequency=dtm.sum(axis=0).A1,
        mds='tsne'
    )
    pyLDAvis.save_html(lda_vis_data, save_file)
    print(f"Visualization saved to {save_file}")

perform_lda(all_reviews, num_topics=5, save_file="all_reviews_lda.html")
perform_lda(positive_reviews, num_topics=5, save_file="positive_reviews_lda.html")
perform_lda(negative_reviews, num_topics=5, save_file="negative_reviews_lda.html")
