# train.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_and_save_model():
    # 1. Representative training dataset (Movie/Sentiment Domain)
    X_train = [
        "This movie was fantastic, amazing plot and great acting!",
        "Loved the film! Excellent direction and brilliant performance.",
        "A true masterpiece, highly recommended for everyone.",
        "Wonderful experience, thoroughly enjoyed watching it.",
        "Terrible waste of time, awful acting and boring script.",
        "Horrible product, worst film ever, completely disappointed.",
        "Bad movie, poorly directed and dreadful pacing.",
        "I hated this, completely unwatchable and waste of money."
    ]
    y_train = [1, 1, 1, 1, 0, 0, 0, 0]

    # 2. Define Pipeline (Vectorization + Model)
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
        ('clf', LogisticRegression())
    ])

    # 3. FIT THE PIPELINE (Populates idf_ and classifier coefficients)
    print("Fitting TF-IDF vectorizer and Logistic Regression classifier...")
    model_pipeline.fit(X_train, y_train)

    # 4. Save fitted pipeline
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.pkl"
    joblib.dump(model_pipeline, model_path)
    print(f"Successfully saved fitted pipeline to {model_path}")

if __name__ == "__main__":
    train_and_save_model()