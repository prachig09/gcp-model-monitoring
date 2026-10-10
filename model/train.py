import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Loading real text dataset with balanced classes...")

# Load two distinct categories to represent binary classes (0 = Tech, 1 = Science)
cats = ['comp.graphics', 'sci.space']
dataset = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))

df_text = pd.DataFrame({
    'text': dataset.data,
    'label': dataset.target
})

# Clean out empty reviews/posts
df_text = df_text[df_text['text'].str.strip() != ''].reset_index(drop=True)

# Shuffle and balance classes explicitly
df_class_0 = df_text[df_text['label'] == 0].sample(n=500, random_state=42)
df_class_1 = df_text[df_text['label'] == 1].sample(n=500, random_state=42)
df_imdb = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Extract Meta-Features for Monitoring
df_imdb['text_length'] = df_imdb['text'].apply(len)
df_imdb['word_count'] = df_imdb['text'].apply(lambda x: len(x.split()))

# Save Baseline Reference Data
df_imdb[['text_length', 'word_count', 'label']].to_csv('data/reference.csv', index=False)

print(f"Dataset prepared with class distribution:\n{df_imdb['label'].value_counts()}")

# Train Model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression())
])

pipeline.fit(df_imdb['text'], df_imdb['label'])

# Save Artifacts
joblib.dump(pipeline, 'model/model.pkl')
print("\nSuccess! Model trained and saved to model/model.pkl, baseline saved to data/reference.csv.")