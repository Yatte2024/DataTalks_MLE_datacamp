# score_v1.py
import pickle

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

# pipeline is a scikit-learn Pipeline with a DictVectorizer and LogisticRegression
proba = pipeline.predict_proba([record])[:, 1][0]

print(f"{proba:.6f}")
