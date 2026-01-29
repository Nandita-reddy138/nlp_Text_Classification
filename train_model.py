# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Load dataset (MUST COME FIRST)
# -----------------------------
df = pd.read_csv("news_data.csv")
print("Dataset loaded:", df.shape)

# -----------------------------
# Text cleaning
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("Text cleaned")

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
print("TF-IDF shape:", X.shape)

# -----------------------------
# Train / Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train/Test split done")

# -----------------------------
# Train Model
# -----------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)
print("Model trained")

# -----------------------------
# Evaluation
# -----------------------------
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "text_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved")
