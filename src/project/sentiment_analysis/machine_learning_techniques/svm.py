# SVM is a supervised machine learning algorithm used for classification (mostly) and
# sometimes regression. The main idea is to find the best boundary (or hyperplane) that
# separates data points of different classes.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# === 1. Load or create sample labeled data ===
# You can start by manually labeling a few rows like this:
data = pd.DataFrame({
    'text': [
        "Apple stock jumps after strong earnings",
        "Oil prices fall amid supply concerns",
        "Healthcare stocks down due to policy changes",
        "Real estate market remains stable in Q1",
        "AI boom drives tech sector to new highs",
        "Consumer spending weakens in Q4",
        "Banks rally after interest rate cut"
    ],
    'sector': [
        "Information Technology",
        "Energy",
        "Health Care",
        "Real Estate",
        "Information Technology",
        "Consumer Discretionary",
        "Financials"
    ],
    'sentiment': [
        "Positive",
        "Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Positive"
    ]
})

# === 2. Split data ===
X = data['text']
y_sector = data['sector']
y_sentiment = data['sentiment']

X_train, X_test, y_sector_train, y_sector_test = train_test_split(X, y_sector, test_size=0.2, random_state=42)
_, _, y_sent_train, y_sent_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

# === 3. Define pipelines ===
sector_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
    ('svm', SVC(kernel='linear'))
])

sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
    ('svm', SVC(kernel='linear'))
])

# === 4. Train models ===
sector_pipeline.fit(X_train, y_sector_train)
sentiment_pipeline.fit(X_train, y_sent_train)

# === 5. Evaluate models ===
print("\nSector Classification Report:")
print(classification_report(y_sector_test, sector_pipeline.predict(X_test)))

print("\nSentiment Classification Report:")
print(classification_report(y_sent_test, sentiment_pipeline.predict(X_test)))

# === 6. Predict on new examples ===
new_texts = [
    "Google launches new AI tool for cloud computing",
    "Natural gas prices continue to decline",
    "Retailers struggle with lower holiday sales"
]

print("\n--- Predictions on New Texts ---")
for text in new_texts:
    sector_pred = sector_pipeline.predict([text])[0]
    sentiment_pred = sentiment_pipeline.predict([text])[0]
    print(f"Text: {text}")
    print(f"Predicted Sector: {sector_pred}")
    print(f"Predicted Sentiment: {sentiment_pred}")
    print("-" * 50)
