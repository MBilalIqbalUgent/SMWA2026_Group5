from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# 1. PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

posts_path = DATA_DIR / "reddit_posts_raw.csv"
comments_path = DATA_DIR / "reddit_comments_raw.csv"
google_trends_path = DATA_DIR / "google_trends_raw.csv"

# -------------------------
# 2. READ FILES
# -------------------------
posts = pd.read_csv(posts_path, encoding="latin1")
comments = pd.read_csv(comments_path, encoding="latin1")

if not google_trends_path.exists():
    raise FileNotFoundError("google_trends_raw.csv not found in data/raw")

google_trends = pd.read_csv(google_trends_path, encoding="latin1")

print("\nGoogle Trends columns:")
print(list(google_trends.columns))

# -------------------------
# 3. CLEAN DATES
# -------------------------
posts["date"] = pd.to_datetime(posts["date"], errors="coerce")
comments["date"] = pd.to_datetime(comments["date"], errors="coerce")
google_trends["date"] = pd.to_datetime(google_trends["date"], errors="coerce")

# -------------------------
# 4. CLEAN TEXT
# -------------------------
posts["text"] = posts["text"].fillna("").astype(str).str.lower()
comments["text"] = comments["text"].fillna("").astype(str).str.lower()

# -------------------------
# 5. ASSIGN MOVIES
# -------------------------
def assign_movie(text: str):
    text = str(text).lower()
    if re.search(r"deadpool|wolverine", text):
        return "Deadpool & Wolverine"
    if re.search(r"\bdune\b|dune 2|dune: part two|dune part two", text):
        return "Dune 2"
    return None

posts["movie"] = posts["text"].apply(assign_movie)
comments["movie"] = comments["text"].apply(assign_movie)

posts_movies = posts.dropna(subset=["movie", "date", "text"]).copy()
comments_movies = comments.dropna(subset=["movie", "date", "text"]).copy()

print(f"\nMovie-related posts: {len(posts_movies)}")
print(f"Movie-related comments: {len(comments_movies)}")

if len(posts_movies) + len(comments_movies) == 0:
    raise ValueError("No movie mentions found in Reddit data.")

# -------------------------
# 6. SIMPLE SENTIMENT SCORING
# -------------------------
positive_words = {
    "good", "great", "amazing", "love", "liked", "awesome", "best", "fun",
    "excellent", "cool", "strong", "favorite", "enjoyed", "fantastic", "positive",
    "brilliant", "incredible", "solid", "beautiful", "impressive"
}

negative_words = {
    "bad", "terrible", "awful", "hate", "boring", "worst", "weak", "disappointing",
    "poor", "mess", "negative", "stupid", "dull", "trash", "annoying",
    "confusing", "lazy", "forgettable", "overrated", "mediocre"
}

def sentiment_score(text: str) -> int:
    words = re.findall(r"\b\w+\b", str(text).lower())
    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)
    return pos - neg

posts_movies["sentiment_score"] = posts_movies["text"].apply(sentiment_score)
comments_movies["sentiment_score"] = comments_movies["text"].apply(sentiment_score)

posts_movies["source"] = "post"
comments_movies["source"] = "comment"

posts_movies["text_length"] = posts_movies["text"].astype(str).str.len()
comments_movies["text_length"] = comments_movies["text"].astype(str).str.len()

# -------------------------
# 7. COMBINE POSTS + COMMENTS
# -------------------------
reddit_all = pd.concat([
    posts_movies[["movie", "date", "text", "source", "sentiment_score", "text_length"]],
    comments_movies[["movie", "date", "text", "source", "sentiment_score", "text_length"]]
], ignore_index=True)

# -------------------------
# 8. MONTHLY REDDIT FEATURES
# -------------------------
reddit_all["month"] = reddit_all["date"].dt.to_period("M").astype(str)

reddit_monthly = (
    reddit_all
    .groupby(["movie", "month"], as_index=False)
    .agg(
        mentions=("text", "count"),
        n_posts=("source", lambda x: (x == "post").sum()),
        n_comments=("source", lambda x: (x == "comment").sum()),
        avg_sentiment=("sentiment_score", "mean"),
        positive_share=("sentiment_score", lambda x: (x > 0).mean()),
        negative_share=("sentiment_score", lambda x: (x < 0).mean()),
        neutral_share=("sentiment_score", lambda x: (x == 0).mean()),
        avg_text_length=("text_length", "mean")
    )
)

reddit_monthly["net_sentiment"] = (
    reddit_monthly["positive_share"] - reddit_monthly["negative_share"]
)

reddit_monthly["engagement_ratio"] = (
    reddit_monthly["n_comments"] / reddit_monthly["mentions"].replace(0, np.nan)
).fillna(0)

reddit_monthly["log_mentions"] = np.log1p(reddit_monthly["mentions"])
reddit_monthly["hype_score"] = reddit_monthly["mentions"] * reddit_monthly["net_sentiment"]

reddit_monthly = reddit_monthly.sort_values(["movie", "month"]).copy()
reddit_monthly["sentiment_volatility"] = (
    reddit_monthly.groupby("movie")["avg_sentiment"].diff().abs()
).fillna(0)

reddit_monthly["month_date"] = pd.to_datetime(reddit_monthly["month"] + "-01", errors="coerce")

print("\nMonthly Reddit features preview:")
print(reddit_monthly.head())

# -------------------------
# 9. GOOGLE TRENDS CLEAN
# -------------------------
if "hits" not in google_trends.columns:
    raise ValueError("Google Trends file must contain a 'hits' column.")

google_trends["hits"] = google_trends["hits"].astype(str).replace("<1", "0")
google_trends["hits"] = pd.to_numeric(google_trends["hits"], errors="coerce")

# Choose best text column for movie matching
candidate_text_cols = ["movie", "keyword", "query", "search_term", "term"]
gt_text_col = None

for col in candidate_text_cols:
    if col in google_trends.columns:
        gt_text_col = col
        break

if gt_text_col is None:
    object_cols = [
        col for col in google_trends.columns
        if google_trends[col].dtype == "object" and col not in ["date", "hits"]
    ]
    if object_cols:
        gt_text_col = object_cols[0]

if gt_text_col is None:
    raise ValueError("No usable text column found in Google Trends file.")

print(f"\nUsing Google Trends text column for movie matching: {gt_text_col}")

google_trends["movie_label_source"] = google_trends[gt_text_col].astype(str)
google_trends["movie"] = google_trends["movie_label_source"].apply(assign_movie)

print("\nMatched Google Trends movie counts:")
print(google_trends["movie"].value_counts(dropna=False))

google_trends = google_trends.dropna(subset=["date", "movie", "hits"]).copy()
google_trends["month"] = google_trends["date"].dt.to_period("M").astype(str)

google_monthly = (
    google_trends
    .groupby(["movie", "month"], as_index=False)
    .agg(google_hits=("hits", "mean"))
)

if google_monthly.empty:
    raise ValueError("No usable Google Trends rows after cleaning.")

google_monthly["month_date"] = pd.to_datetime(google_monthly["month"] + "-01", errors="coerce")

print("\nMonthly Google Trends preview:")
print(google_monthly.head())

# -------------------------
# 10. MERGE REDDIT + GOOGLE TRENDS
# -------------------------
model_monthly = reddit_monthly.merge(
    google_monthly[["movie", "month", "google_hits"]],
    on=["movie", "month"],
    how="inner"
)

print(f"\nRows after monthly merge: {len(model_monthly)}")

if model_monthly.empty:
    raise ValueError(
        "No matching monthly rows between Reddit and Google Trends. "
        "Your time periods still do not overlap."
    )

model_monthly["month_date"] = pd.to_datetime(model_monthly["month"] + "-01", errors="coerce")
model_monthly = model_monthly.sort_values(["movie", "month_date"]).copy()

print("\nMerged monthly model data preview:")
print(model_monthly.head())

# -------------------------
# 11. CREATE FUTURE TARGET
# -------------------------
# Predict next month's Google Trends using current month's Reddit features
model_monthly["target_google_hits_next"] = (
    model_monthly.groupby("movie")["google_hits"].shift(-1)
)

model_data = model_monthly.dropna(subset=["target_google_hits_next"]).copy()

print(f"\nRows in model data after future target creation: {len(model_data)}")

if len(model_data) < 4:
    raise ValueError(
        "Too few rows after creating next-month target. "
        "Your dataset is too small for meaningful prediction."
    )

# -------------------------
# 12. FEATURES
# -------------------------
feature_cols = [
    "mentions",
    "n_comments",
    "avg_sentiment",
    "net_sentiment",
    "engagement_ratio",
    "sentiment_volatility",
    "log_mentions",
    "hype_score",
    "avg_text_length"
]

X = model_data[feature_cols]
y = model_data["target_google_hits_next"]

# -------------------------
# 13. CORRELATION TABLE
# -------------------------
corr_cols = feature_cols + ["target_google_hits_next", "google_hits"]
corr_table = model_data[corr_cols].corr(numeric_only=True)
corr_table.to_csv(OUTPUT_DIR / "correlation_table.csv")

# -------------------------
# 14. TIME-BASED TRAIN / TEST SPLIT
# -------------------------
unique_months = sorted(model_data["month_date"].dropna().unique())
split_index = int(np.floor(0.8 * len(unique_months)))

if split_index < 1 or split_index >= len(unique_months):
    raise ValueError("Not enough unique months for a valid train/test split.")

split_date = unique_months[split_index - 1]

train_data = model_data[model_data["month_date"] <= split_date].copy()
test_data = model_data[model_data["month_date"] > split_date].copy()

print(f"\nSplit month: {split_date}")
print(f"Train rows: {len(train_data)}")
print(f"Test rows: {len(test_data)}")

if len(train_data) < 3 or len(test_data) < 1:
    raise ValueError("Dataset too small after train/test split.")

X_train = train_data[feature_cols]
y_train = train_data["target_google_hits_next"]

X_test = test_data[feature_cols]
y_test = test_data["target_google_hits_next"]

# -------------------------
# 15. LINEAR REGRESSION
# -------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

test_data["pred_google_hits_lr"] = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, test_data["pred_google_hits_lr"]))
lr_mae = mean_absolute_error(y_test, test_data["pred_google_hits_lr"])
lr_r2_train = lr_model.score(X_train, y_train)
lr_r2_test = r2_score(y_test, test_data["pred_google_hits_lr"]) if len(test_data) > 1 else np.nan

lr_coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_
}).sort_values("coefficient", ascending=False)

# -------------------------
# 16. RANDOM FOREST
# -------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=4
)
rf_model.fit(X_train, y_train)

test_data["pred_google_hits_rf"] = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, test_data["pred_google_hits_rf"]))
rf_mae = mean_absolute_error(y_test, test_data["pred_google_hits_rf"])
rf_r2_train = rf_model.score(X_train, y_train)
rf_r2_test = r2_score(y_test, test_data["pred_google_hits_rf"]) if len(test_data) > 1 else np.nan

rf_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

# -------------------------
# 17. PRINT RESULTS
# -------------------------
print("\n=== LINEAR REGRESSION RESULTS ===")
print(lr_coef_df)
print(f"\nLinear Regression Train R-squared: {lr_r2_train:.4f}")
print(
    f"Linear Regression Test R-squared: {lr_r2_test:.4f}"
    if not np.isnan(lr_r2_test)
    else "Linear Regression Test R-squared: not available"
)
print(f"Linear Regression RMSE: {lr_rmse:.4f}")
print(f"Linear Regression MAE: {lr_mae:.4f}")

print("\n=== RANDOM FOREST RESULTS ===")
print(rf_importance_df)
print(f"\nRandom Forest Train R-squared: {rf_r2_train:.4f}")
print(
    f"Random Forest Test R-squared: {rf_r2_test:.4f}"
    if not np.isnan(rf_r2_test)
    else "Random Forest Test R-squared: not available"
)
print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest MAE: {rf_mae:.4f}")

# -------------------------
# 18. SAVE OUTPUTS
# -------------------------
reddit_monthly.to_csv(OUTPUT_DIR / "reddit_monthly_features.csv", index=False)
google_monthly.to_csv(OUTPUT_DIR / "google_monthly_features.csv", index=False)
model_monthly.to_csv(OUTPUT_DIR / "model_data_monthly.csv", index=False)
model_data.to_csv(OUTPUT_DIR / "model_data_monthly_future_target.csv", index=False)
test_data.to_csv(OUTPUT_DIR / "predictions_test_data.csv", index=False)
lr_coef_df.to_csv(OUTPUT_DIR / "linear_model_coefficients.csv", index=False)
rf_importance_df.to_csv(OUTPUT_DIR / "random_forest_importance.csv", index=False)

metrics_df = pd.DataFrame({
    "model": ["Linear Regression", "Random Forest"],
    "train_r2": [lr_r2_train, rf_r2_train],
    "test_r2": [lr_r2_test, rf_r2_test],
    "rmse": [lr_rmse, rf_rmse],
    "mae": [lr_mae, rf_mae]
})
metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

# -------------------------
# 19. PLOTS
# -------------------------

# Actual vs predicted - LR
plt.figure(figsize=(8, 6))
plt.scatter(test_data["target_google_hits_next"], test_data["pred_google_hits_lr"])
min_val = min(test_data["target_google_hits_next"].min(), test_data["pred_google_hits_lr"].min())
max_val = max(test_data["target_google_hits_next"].max(), test_data["pred_google_hits_lr"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.title("Actual vs Predicted Next-Month Google Trends - Linear Regression")
plt.xlabel("Actual Next-Month Google Trends")
plt.ylabel("Predicted Next-Month Google Trends")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "actual_vs_predicted_google_hits_lr.png")
plt.close()

# Actual vs predicted - RF
plt.figure(figsize=(8, 6))
plt.scatter(test_data["target_google_hits_next"], test_data["pred_google_hits_rf"])
min_val = min(test_data["target_google_hits_next"].min(), test_data["pred_google_hits_rf"].min())
max_val = max(test_data["target_google_hits_next"].max(), test_data["pred_google_hits_rf"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.title("Actual vs Predicted Next-Month Google Trends - Random Forest")
plt.xlabel("Actual Next-Month Google Trends")
plt.ylabel("Predicted Next-Month Google Trends")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "actual_vs_predicted_google_hits_rf.png")
plt.close()

# Reddit mentions vs Google Trends
plt.figure(figsize=(8, 6))
for movie, df_movie in model_monthly.groupby("movie"):
    plt.scatter(df_movie["mentions"], df_movie["google_hits"], label=movie)
plt.title("Monthly Reddit Mentions vs Monthly Google Trends")
plt.xlabel("Monthly Reddit Mentions")
plt.ylabel("Monthly Google Trends")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "reddit_mentions_vs_google_trends.png")
plt.close()

# Random forest importance
plt.figure(figsize=(10, 6))
plt.barh(rf_importance_df["feature"], rf_importance_df["importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "random_forest_feature_importance.png")
plt.close()

# -------------------------
# 20. SUMMARY NOTES
# -------------------------
print("\nPredictive analysis complete.")
print(f"Saved: {OUTPUT_DIR / 'reddit_monthly_features.csv'}")
print(f"Saved: {OUTPUT_DIR / 'google_monthly_features.csv'}")
print(f"Saved: {OUTPUT_DIR / 'model_data_monthly.csv'}")
print(f"Saved: {OUTPUT_DIR / 'model_data_monthly_future_target.csv'}")
print(f"Saved: {OUTPUT_DIR / 'predictions_test_data.csv'}")
print(f"Saved: {OUTPUT_DIR / 'linear_model_coefficients.csv'}")
print(f"Saved: {OUTPUT_DIR / 'random_forest_importance.csv'}")
print(f"Saved: {OUTPUT_DIR / 'model_metrics.csv'}")
print(f"Saved: {OUTPUT_DIR / 'correlation_table.csv'}")
print(f"Plots saved in: {PLOTS_DIR}")

print("\nInterpretation notes:")
print("- This model predicts next-month Google Trends using current-month Reddit features.")
print("- Monthly aggregation is used because Reddit data is sparse.")
print("- Results are exploratory due to the small dataset.")
print("- This is still much stronger than using a fabricated box office series.")