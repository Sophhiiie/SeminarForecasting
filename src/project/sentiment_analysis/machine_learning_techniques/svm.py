# SVM is a supervised machine learning algorithm used for classification (mostly) and
# sometimes regression. The main idea is to find the best boundary (or hyperplane) that
# separates data points of different classes.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ✅ 1. Labeled sample data
data = {
    'text': [
        'I love this product',
        'This is the worst purchase ever',
        'Absolutely amazing and fun!',
        'Terrible customer support and experience',
        'I enjoyed every bit of it',
        'Not worth the money or time',
        'This is the best thing I’ve used',
        'I hate everything about this'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# ✅ 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# ✅ 3. Text vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ 4. Train SVM (with probability support if needed)
model = SVC(kernel='linear', probability=True)  # Use 'linear' kernel for text
model.fit(X_train_vec, y_train)

# ✅ 5. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ 6. Predict new text with confidence score
new_text = ["I really love how easy this is to use!"]
new_vec = vectorizer.transform(new_text)

pred = model.predict(new_vec)[0]
prob = model.predict_proba(new_vec)[0]
sentiment = "positive" if pred == 1 else "negative"
confidence = round(prob[pred] * 100, 2)

print(f"\nSentiment: {sentiment} (confidence: {confidence}%)")
