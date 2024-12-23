import pandas as pd
import numpy as np
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import random

def load_sample_reviews(file_path, sample_fraction=0.1):
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    sampled_reviews = random.sample(reviews, int(len(reviews) * sample_fraction))
    return pd.DataFrame(sampled_reviews)

def load_business_data(file_path):
    businesses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            businesses.append(json.loads(line))
    return pd.DataFrame(businesses)

review_file_path = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
business_file_path = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"

reviews_df = load_sample_reviews(review_file_path)
business_df = load_business_data(business_file_path)
dish_names_path = "student_dn_annotations.txt"
with open(dish_names_path, 'r', encoding='utf-8') as f:
    dish_names = [line.strip().lower() for line in f.readlines()]
def analyze_reviews(reviews_df, dish_names):
    dish_popularity = {dish: {'count': 0, 'positive': 0, 'negative': 0} for dish in dish_names}
    for _, row in reviews_df.iterrows():
        review_text = row['text'].lower()
        sentiment = TextBlob(review_text).sentiment.polarity
        for dish in dish_names:
            if dish in review_text:
                dish_popularity[dish]['count'] += 1
                if sentiment > 0:
                    dish_popularity[dish]['positive'] += 1
                elif sentiment < 0:
                    dish_popularity[dish]['negative'] += 1
    return dish_popularity
dish_popularity = analyze_reviews(reviews_df, dish_names)
dish_rankings = sorted(
    dish_popularity.items(),
    key=lambda x: (x[1]['count'], x[1]['positive']),
    reverse=True
)
def plot_popular_dishes(dish_rankings, top_n=10):
    top_dishes = dish_rankings[:top_n]
    dishes = [dish[0] for dish in top_dishes]
    counts = [dish[1]['count'] for dish in top_dishes]
    positives = [dish[1]['positive'] for dish in top_dishes]
    x = np.arange(len(dishes))
    plt.bar(x - 0.2, counts, width=0.4, label='Total Mentions')
    plt.bar(x + 0.2, positives, width=0.4, label='Positive Mentions', color='green')
    plt.xlabel('Dishes')
    plt.ylabel('Mentions')
    plt.title('Top Popular Dishes')
    plt.xticks(x, dishes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_popular_dishes(dish_rankings)
def recommend_restaurants(reviews_df, business_df, dish_name):
    restaurant_scores = {}
    for _, row in reviews_df.iterrows():
        review_text = row['text'].lower()
        if dish_name in review_text:
            business_id = row['business_id']
            sentiment = TextBlob(review_text).sentiment.polarity
            if business_id not in restaurant_scores:
                restaurant_scores[business_id] = {'count': 0, 'score': 0}
            restaurant_scores[business_id]['count'] += 1
            restaurant_scores[business_id]['score'] += sentiment
    ranked_restaurants = []
    for business_id, data in restaurant_scores.items():
        business_info = business_df[business_df['business_id'] == business_id]
        if not business_info.empty:
            address = business_info.iloc[0]['full_address']
            ranked_restaurants.append({
                'address': address,
                'count': data['count'],
                'score': data['score']
            })
    ranked_restaurants = sorted(ranked_restaurants, key=lambda x: (x['score'], x['count']), reverse=True)
    return ranked_restaurants
selected_dish = "margherita pizza"
ranked_restaurants = recommend_restaurants(reviews_df, business_df, selected_dish)
def plot_recommended_restaurants(ranked_restaurants, top_n=10):
    top_restaurants = ranked_restaurants[:top_n]
    addresses = [restaurant['address'] for restaurant in top_restaurants]
    scores = [restaurant['score'] for restaurant in top_restaurants]
    plt.barh(addresses, scores, color='blue')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Restaurant Address')
    plt.title(f'Top Restaurants for {selected_dish}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
plot_recommended_restaurants(ranked_restaurants)
