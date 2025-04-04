# This file has as goal to compute the sentiment scores for the newspapers and the selected subreddits.
# Naive Bayes is a supervised machine learning techniques.
# Three steps are required:
    # 1: We label our data
    # 2: We train our Naive Bayes model
    # 2: We make sentiment predictions on the remaining data

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ✅ 1. Sample data
data = {
    'text': [
        'I love this product',
        'This is the worst thing I bought',
        'Absolutely amazing experience',
        'Terrible service and rude staff',
        'Really enjoyed it a lot',
        'Not worth the money at all',
        'Best purchase I have made',
        'Extremely disappointing and bad',
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# ✅ 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# ✅ 3. Convert text to numeric features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ 4. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ 5. Predict on test set
y_pred = model.predict(X_test_vec)

# ✅ 6. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ 7. Predict a new text
new_text = ["I absolutely love it"]
new_vec = vectorizer.transform(new_text)
prediction = model.predict(new_vec)[0]
score = model.predict_proba(new_vec)[0]

label = "positive" if prediction == 1 else "negative"
confidence = round(score[prediction] * 100, 2)

print(f"\nNew text sentiment: {label} (confidence: {confidence}%)")

