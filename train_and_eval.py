import re
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\anime\playground\xnode_project\emails.csv")

def clean_email_body(text: str) -> str:
    """
    lower-cases
    strips HTML tags & angle-bracket e-mail quoting
    collapses whitespace/newlines
    """
    text = re.sub(r"<.*?>", " ", text)                 # strip tags
    text = re.sub(r"\n+", " ", text)                   # newlines -> space
    text = re.sub(r"\s{2,}", " ", text)                # collapse runs
    return text.lower().strip()

df["body_clean"] = df["body"].astype(str).apply(clean_email_body)

X = df["body_clean"]
y = df["intent"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",          # removes very frequent function words
        ngram_range=(1, 2),            # unigrams + bigrams
        min_df=2                       # ignore rare terms that appear once
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",       # helps if classes become imbalanced
        n_jobs=-1
    ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipe, X, y, cv=skf)

print("\n=== 5-fold cross-validated metrics ===")
print(classification_report(y, y_pred))

pipe.fit(X, y)

# Now this works
cm = confusion_matrix(y, y_pred, labels=pipe.classes_)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=pipe.classes_, yticklabels=pipe.classes_)
plt.title("Confusion Matrix (5-fold CV)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()

with open("email_intent_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)

with open("tfidf_vectorizer.pkl", "wb") as f_vec, open("logreg_model.pkl", "wb") as f_clf:
    pickle.dump(pipe.named_steps["tfidf"], f_vec)
    pickle.dump(pipe.named_steps["clf"],   f_clf)