import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import precision_recall_curve

with open('Hygiene/hygiene.dat', 'r') as file:
    reviews = file.readlines()

def clean_text(text):
    return BeautifulSoup(text, 'html.parser').get_text()
reviews_cleaned = [clean_text(review) for review in reviews]
labels = pd.read_csv('Hygiene/hygiene.dat.labels', header=None)
additional_info = pd.read_csv('Hygiene/hygiene.dat.additional', header=None)
print("Columns of additional_info:")
print(additional_info.columns)
data = pd.DataFrame({
    'review': reviews_cleaned,
    'label': labels[0],
    'cuisine': additional_info[0],
    'zip_code': additional_info[1],
    'num_reviews': additional_info[2],
    'avg_rating': additional_info[3]
})
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(data['review'])
X_additional = data[['num_reviews', 'avg_rating']].values
X = hstack([X_text, X_additional])
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)
print(f"Unique labels in y_train: {np.unique(y_train)}")
unique_labels = np.unique(y_train)
if len(unique_labels) > 1:
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=y_train)
    class_weights_dict = {label: class_weights[i] for i, label in enumerate(unique_labels)}
else:
    class_weights_dict = None
model = LogisticRegression(class_weight=class_weights_dict, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {f1_macro:.4f}")

with open('Hygiene/hygiene.dat', 'r') as file:
    test_reviews = file.readlines()[546:]

test_reviews_cleaned = [clean_text(review) for review in test_reviews]
X_test_text = vectorizer.transform(test_reviews_cleaned)
print("Columns of additional_info for the test set:")
print(additional_info[546:].columns)
X_test_additional = additional_info[546:][[2, 3]].values
X_test_combined = hstack([X_test_text, X_test_additional])
test_predictions = model.predict(X_test_combined)
output = pd.DataFrame({'Nickname': ['Kassym']*len(test_predictions), 'Label': test_predictions})
output.to_csv('predictions.csv', index=False, header=False)
print("Predictions saved to 'predictions.csv'")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
class_report = classification_report(y_test, y_pred, output_dict=True)
precision = [class_report[str(label)]['precision'] for label in unique_labels]
recall = [class_report[str(label)]['recall'] for label in unique_labels]
f1 = [class_report[str(label)]['f1-score'] for label in unique_labels]
plt.figure(figsize=(8, 5))
bar_width = 0.2
index = np.arange(len(unique_labels))
plt.bar(index, precision, bar_width, label='Precision')
plt.bar(index + bar_width, recall, bar_width, label='Recall')
plt.bar(index + 2*bar_width, f1, bar_width, label='F1 Score')
plt.xlabel('Class')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1 Score')
plt.xticks(index + bar_width, [str(label) for label in unique_labels])
plt.legend()
plt.show()

if len(unique_labels) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("ROC Curve is only supported for binary classification. Please adjust your model or use a different approach for multi-class.")
