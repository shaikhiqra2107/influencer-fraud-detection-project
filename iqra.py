# =========================
#  Influencer Fraud Detection
# =========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

# -------------------------
# 1. Load Data
# -------------------------
data = pd.read_csv("influencers.csv")  # 5,000 profiles
# labeled_200.csv will contain true labels (0=genuine, 1=fraud)

# -------------------------
# 2. Feature Engineering
# -------------------------
data["engagement_per_follower"] = (data["avg_likes"] + 2 * data["avg_comments"]) / data["followers"].replace(0, 1)
data["likes_per_post"] = data["avg_likes"] / data["posts_30d"].replace(0, 1)
data["comments_per_post"] = data["avg_comments"] / data["posts_30d"].replace(0, 1)
data["growth_rate_30d"] = data["follower_growth_30d"] / data["followers"].replace(0, 1)
data["likes_followers_ratio_log"] = np.log1p(data["avg_likes"]) - np.log1p(data["followers"])

features = [
    "followers", "avg_likes", "avg_comments", "posts_30d",
    "follower_growth_30d", "engagement_per_follower",
    "likes_per_post", "comments_per_post", "growth_rate_30d",
    "likes_followers_ratio_log"
]

X = data[features].fillna(0)

# -------------------------
# 3. Preprocessing
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4. Train Models
# -------------------------
iso = IsolationForest(contamination=0.1, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

iso_scores = -iso.fit(X_scaled).score_samples(X_scaled)
lof_scores = -lof.fit_predict(X_scaled)
lof_scores = (lof.negative_outlier_factor_ * -1)  # convert to positive anomaly score

# -------------------------
# 5. Ensemble Scoring
# -------------------------
iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

ensemble_score = 0.6 * iso_norm + 0.4 * lof_norm
data["fraud_score_0_100"] = (ensemble_score * 100).round(2)

# -------------------------
# 6. Choose Threshold using labeled data
# -------------------------
labeled = pd.read_csv("labeled_200.csv")
merged = data.merge(labeled, on="handle", how="inner")

best_threshold, best_precision, best_recall = 0, 0, 0
for threshold in range(50, 100):
    preds = (merged["fraud_score_0_100"] >= threshold).astype(int)
    precision = precision_score(merged["label"], preds)
    recall = recall_score(merged["label"], preds)
    if precision >= 0.85 and recall > best_recall:
        best_threshold, best_precision, best_recall = threshold, precision, recall

print(f"✅ Best threshold: {best_threshold}")
print(f"Precision: {best_precision:.2f}, Recall: {best_recall:.2f}")

# -------------------------
# 7. Final Output
# -------------------------
data.to_csv("influencer_scores.csv", index=False)
print("Fraud risk scores saved ➜ influencer_scores.csv")