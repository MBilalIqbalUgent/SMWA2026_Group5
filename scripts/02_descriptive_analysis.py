from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
google_trends = pd.read_csv(google_trends_path, encoding="latin1") if google_trends_path.exists() else None

# -------------------------
# 3. CLEAN DATES
# -------------------------
posts["date"] = pd.to_datetime(posts["date"], errors="coerce")
comments["date"] = pd.to_datetime(comments["date"], errors="coerce")

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

print(f"Movie-related posts: {len(posts_movies)}")
print(f"Movie-related comments: {len(comments_movies)}")

if len(posts_movies) + len(comments_movies) == 0:
    raise ValueError("No movie mentions found in the Reddit data.")

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

# -------------------------
# 7. COMBINE POSTS + COMMENTS
# -------------------------
reddit_all = pd.concat([
    posts_movies[["movie", "date", "text", "source", "sentiment_score"]],
    comments_movies[["movie", "date", "text", "source", "sentiment_score"]]
], ignore_index=True)

# -------------------------
# 8. DAILY REDDIT FEATURES
# -------------------------
daily_features = (
    reddit_all
    .groupby(["movie", reddit_all["date"].dt.date], as_index=False)
    .agg(
        mentions=("text", "count"),
        n_posts=("source", lambda x: (x == "post").sum()),
        n_comments=("source", lambda x: (x == "comment").sum()),
        avg_sentiment=("sentiment_score", "mean"),
        positive_share=("sentiment_score", lambda x: (x > 0).mean()),
        negative_share=("sentiment_score", lambda x: (x < 0).mean()),
        neutral_share=("sentiment_score", lambda x: (x == 0).mean()),
    )
    .rename(columns={"date": "day"})
)

daily_features["net_sentiment"] = (
    daily_features["positive_share"] - daily_features["negative_share"]
)

# -------------------------
# 9. MONTHLY REDDIT FEATURES
# -------------------------
reddit_all["month"] = reddit_all["date"].dt.to_period("M").astype(str)

monthly_features = (
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
    )
)

monthly_features["net_sentiment"] = (
    monthly_features["positive_share"] - monthly_features["negative_share"]
)

# -------------------------
# 10. GOOGLE TRENDS CLEAN
# -------------------------
google_monthly = None

if google_trends is not None and {"date", "hits", "movie"}.issubset(google_trends.columns):
    google_trends["date"] = pd.to_datetime(google_trends["date"], errors="coerce")
    google_trends["hits"] = google_trends["hits"].replace("<1", "0")
    google_trends["hits"] = pd.to_numeric(google_trends["hits"], errors="coerce")
    google_trends["movie"] = google_trends["movie"].astype(str)

    google_trends = google_trends.dropna(subset=["date", "hits", "movie"]).copy()
    google_trends["month"] = google_trends["date"].dt.to_period("M").astype(str)

    google_monthly = (
        google_trends
        .groupby(["movie", "month"], as_index=False)
        .agg(google_hits=("hits", "mean"))
    )

# -------------------------
# 11. SAVE OUTPUTS
# -------------------------
daily_features.to_csv(OUTPUT_DIR / "daily_reddit_features.csv", index=False)
monthly_features.to_csv(OUTPUT_DIR / "monthly_reddit_features.csv", index=False)

if google_monthly is not None:
    google_monthly.to_csv(OUTPUT_DIR / "monthly_google_trends.csv", index=False)

movie_summary = (
    reddit_all
    .groupby("movie", as_index=False)
    .agg(
        total_mentions=("text", "count"),
        avg_sentiment=("sentiment_score", "mean")
    )
)

movie_summary.to_csv(OUTPUT_DIR / "movie_summary_descriptive.csv", index=False)

# -------------------------
# 12. PLOTS
# -------------------------

# Daily mentions
plt.figure(figsize=(10, 6))
for movie, df_movie in daily_features.groupby("movie"):
    df_movie = df_movie.sort_values("day")
    plt.plot(df_movie["day"], df_movie["mentions"], marker="o", label=movie)
plt.title("Daily Reddit Mentions by Movie")
plt.xlabel("Date")
plt.ylabel("Mentions")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "daily_mentions_by_movie.png")
plt.close()

# Daily sentiment
plt.figure(figsize=(10, 6))
for movie, df_movie in daily_features.groupby("movie"):
    df_movie = df_movie.sort_values("day")
    plt.plot(df_movie["day"], df_movie["avg_sentiment"], marker="o", label=movie)
plt.title("Daily Average Sentiment by Movie")
plt.xlabel("Date")
plt.ylabel("Average Sentiment")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "daily_sentiment_by_movie.png")
plt.close()

# Monthly mentions
plt.figure(figsize=(10, 6))
for movie, df_movie in monthly_features.groupby("movie"):
    plt.plot(df_movie["month"], df_movie["mentions"], marker="o", label=movie)
plt.title("Monthly Reddit Mentions by Movie")
plt.xlabel("Month")
plt.ylabel("Mentions")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "monthly_mentions_by_movie.png")
plt.close()

# Monthly sentiment
plt.figure(figsize=(10, 6))
for movie, df_movie in monthly_features.groupby("movie"):
    plt.plot(df_movie["month"], df_movie["avg_sentiment"], marker="o", label=movie)
plt.title("Monthly Average Sentiment by Movie")
plt.xlabel("Month")
plt.ylabel("Average Sentiment")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "monthly_sentiment_by_movie.png")
plt.close()

# Monthly Google Trends
if google_monthly is not None and not google_monthly.empty:
    plt.figure(figsize=(10, 6))
    for movie, df_movie in google_monthly.groupby("movie"):
        plt.plot(df_movie["month"], df_movie["google_hits"], marker="o", label=movie)
    plt.title("Monthly Google Trends by Movie")
    plt.xlabel("Month")
    plt.ylabel("Google Trends Hits")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_google_trends_by_movie.png")
    plt.close()

# Word clouds
for movie in reddit_all["movie"].dropna().unique():
    text_blob = " ".join(reddit_all.loc[reddit_all["movie"] == movie, "text"].astype(str))
    if text_blob.strip():
        wc = WordCloud(width=1000, height=500, background_color="white").generate(text_blob)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {movie}")
        plt.tight_layout()
        safe_name = movie.lower().replace(" ", "_").replace("&", "and")
        plt.savefig(PLOTS_DIR / f"wordcloud_{safe_name}.png")
        plt.close()

print("\nDescriptive analysis complete.")
print(f"Saved: {OUTPUT_DIR / 'daily_reddit_features.csv'}")
print(f"Saved: {OUTPUT_DIR / 'monthly_reddit_features.csv'}")
if google_monthly is not None:
    print(f"Saved: {OUTPUT_DIR / 'monthly_google_trends.csv'}")
print(f"Saved: {OUTPUT_DIR / 'movie_summary_descriptive.csv'}")
print(f"Plots saved in: {PLOTS_DIR}")