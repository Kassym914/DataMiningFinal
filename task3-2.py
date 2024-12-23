import json
import random
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import csv
nlp = spacy.load("en_core_web_sm")
reviews_file = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'

def load_reviews(file_path, percentage=1):
    with open(file_path, 'r') as f:
        reviews = [json.loads(line) for line in f]
    return random.sample(reviews, int(len(reviews) * (percentage / 100)))

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def extract_ngram_candidates(reviews, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    corpus = [preprocess_text(review['text']) for review in reviews]
    X = vectorizer.fit_transform(corpus)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)

    ngram_counts = Counter(dict(zip(ngrams, counts)))
    return ngram_counts.most_common(20)

def expand_dish_names(reviews, refined_dishes, ngram_limit=20):
    candidates = extract_ngram_candidates(reviews, n=2)
    expanded_dishes = set(refined_dishes)

    for candidate, count in candidates:
        if count >= 10 and candidate not in refined_dishes:
            expanded_dishes.add(candidate)
    return expanded_dishes

reviews_data = load_reviews(reviews_file, 1)
refined_dishes = set()
with open('Refined_Italian.label', 'r') as f:
    for line in f:
        dish, label = line.strip().split('\t')
        if int(label) == 1:
            refined_dishes.add(dish)

expanded_dishes = expand_dish_names(reviews_data, refined_dishes)


def save_expanded_dishes(expanded_dishes, filename='expanded_dishes.txt'):
    with open(filename, 'w') as f:
        for dish in expanded_dishes:
            f.write(f"{dish}\n")


def save_dish_counts(expanded_dishes, filename='dish_counts.csv'):
    dish_counts = Counter(expanded_dishes)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dish', 'Count'])  # Column headers
        for dish, count in dish_counts.most_common():
            writer.writerow([dish, count])


def visualize_dishes(expanded_dishes):
    dish_counts = Counter(expanded_dishes)
    dishes, counts = zip(*dish_counts.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.barh(dishes, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.title('Top 10 Dish Candidates from Yelp Reviews')
    plt.gca().invert_yaxis()
    plt.show()


print("Expanded list of dish names:")
for dish in expanded_dishes:
    print(dish)

save_expanded_dishes(expanded_dishes, 'expanded_dishes.txt')
save_dish_counts(expanded_dishes, 'dish_counts.csv')

visualize_dishes(expanded_dishes)
