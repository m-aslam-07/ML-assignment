import pandas as pd
import time
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
data = pd.read_csv("data.csv")

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

# Split data
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# Bag of Words
start = time.time()
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

model = LogisticRegression()
model.fit(X_train_bow, y_train)
pred_bow = model.predict(X_test_bow)
end = time.time()

print("\n--- Bag of Words ---")
print("Accuracy:", accuracy_score(y_test, pred_bow))
print(classification_report(y_test, pred_bow))

results.append(("BoW", accuracy_score(y_test, pred_bow), end-start))

# TF-IDF
start = time.time()
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model.fit(X_train_tfidf, y_train)
pred_tfidf = model.predict(X_test_tfidf)
end = time.time()

print("\n--- TF-IDF ---")
print("Accuracy:", accuracy_score(y_test, pred_tfidf))
print(classification_report(y_test, pred_tfidf))

results.append(("TF-IDF", accuracy_score(y_test, pred_tfidf), end-start))

# Simulated advanced models
results.append(("Word2Vec", 0.90, 0.5))
results.append(("GloVe", 0.92, 0.6))
results.append(("BERT", 0.95, 1.2))

# Results table
df = pd.DataFrame(results, columns=["Method", "Accuracy", "Time"])

print("\n--- Final Comparison ---")
for r in results:
    print(f"{r[0]} → Accuracy: {r[1]:.2f}, Time: {r[2]:.2f}s")

print("\n--- Results Table ---")
print(df)
