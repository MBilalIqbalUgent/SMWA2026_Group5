from pathlib import Path
import re
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


# ============================================================
# Getting folders and files ready
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_POSTS_FILE = RAW_DIR / "reddit_posts_raw.csv"
REDDIT_COMMENTS_FILE = RAW_DIR / "reddit_comments_raw.csv"
GOOGLE_TRENDS_FILE = RAW_DIR / "google_trends_raw.csv"
IMDB_FILE = RAW_DIR / "imdb_movie_metrics_raw.csv"
BOX_OFFICE_FILE = RAW_DIR / "weekly_box_office_raw.csv"

DESCRIPTIVE_PANEL_FILE = OUTPUT_DIR / "descriptive_weekly_panel.csv"


# ============================================================
# Text and stopword settings
# ============================================================
analyzer = SentimentIntensityAnalyzer()

CUSTOM_STOPWORDS = {
    "look", "had", "way", "new", "say", "see", "good", "one", "thing", "really",
    "people", "think",
    "movie", "film", "movies", "films", "just", "like", "watch", "watched",
    "scene", "scenes", "im", "ive", "dont", "didnt", "doesnt", "isnt",
    "wasnt", "theyre"
}

EXTRA_STOPWORDS = {
    "make", "know", "seen", "said", "lot", "maybe", "pretty", "want",
    "better", "fun", "bad", "action", "character", "little", "big",
    "day", "week", "month", "years", "live", "post", "list", "guy",
    "men", "things", "far", "kind", "definitely", "actually", "right",
    "come", "came", "going", "got", "get", "still", "well", "much",
    "even", "also", "around", "inside", "outside", "need",
    "probably", "sure", "yeah", "yes", "isn", "doesn", "wasn", "aren",
    "weren", "wouldn", "couldn", "shouldn", "bit", "point", "times",
    "work", "world", "place", "end", "start", "old", "real",
    "saw", "thought", "feel", "felt", "looking", "looks", "review",
    "reviews", "opinion", "favorite", "interesting", "enjoyed", "loved",
    "performance", "screen", "theater", "theatre", "release", "streaming"
}

ALL_STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)
ALL_WORDCLOUD_STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS).union(EXTRA_STOPWORDS)

MOVIE_SPECIFIC_STOPWORDS = {
    "Deadpool & Wolverine": {
        "deadpool", "wolverine", "marvel", "mcu", "spider", "avenger",
        "ryan", "reynolds", "jackman", "logan", "xmen", "men", "x"
    },
    "Dune 2": {
        "dune", "paul", "villeneuve", "chalamet", "fremen", "atreides",
        "harkonnen", "stilgar", "zendaya", "timothee", "denis", "feyd",
        "jessica", "butler", "austin", "chani"
    }
}


# ============================================================
# Small helper functions
# ============================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}\n")


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_week_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)
    return dt.dt.to_period("W-SUN").dt.start_time


def get_sentiment_score(text: str) -> float:
    score = analyzer.polarity_scores(str(text))["compound"]
    return max(-1.0, min(1.0, score))


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(4)
    return df


def safe_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return np.nan
    temp = df[[x_col, y_col]].dropna()
    if len(temp) < 2:
        return np.nan
    if temp[x_col].nunique() < 2 or temp[y_col].nunique() < 2:
        return np.nan
    return temp[x_col].corr(temp[y_col])


def save_line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
) -> None:
    valid = data.dropna(subset=[x_col, y_col, group_col]).copy()
    if valid.empty:
        return

    plt.figure(figsize=(12, 6))
    for group_value, df_group in valid.groupby(group_col):
        df_group = df_group.sort_values(x_col)
        plt.plot(df_group[x_col], df_group[y_col], marker="o", label=group_value)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def tokenize_clean(text: str, stopwords: set[str]) -> list[str]:
    text = preprocess_text(text)
    words = text.split()
    return [w for w in words if w not in stopwords and len(w) >= 3]


def remove_top_frequent_words(token_lists: list[list[str]], top_n: int = 20):
    all_tokens = [token for tokens in token_lists for token in tokens]
    freq = Counter(all_tokens)
    removed_words = {word for word, _ in freq.most_common(top_n)}

    cleaned_lists = []
    for tokens in token_lists:
        cleaned_lists.append([token for token in tokens if token not in removed_words])

    return cleaned_lists, removed_words


def generate_wordcloud_from_tokens(tokens: list[str], title: str, save_path: Path) -> None:
    text_blob = " ".join(tokens).strip()
    if not text_blob:
        print(f"Skipped {title}: no words left after cleaning.")
        return

    wc = WordCloud(
        width=1800,
        height=900,
        background_color="white",
        max_words=250,
        collocations=False
    ).generate(text_blob)

    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_words_from_lda(model, feature_names, n_top_words=10):
    rows = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append({
            "topic": f"Topic_{topic_idx + 1}",
            "top_words": ", ".join(top_features)
        })
    return pd.DataFrame(rows)


# ============================================================
# Loading all the input files
# ============================================================
print_section("LOADING INPUT FILES")

required_files = [
    REDDIT_POSTS_FILE,
    REDDIT_COMMENTS_FILE,
    GOOGLE_TRENDS_FILE,
    IMDB_FILE,
    BOX_OFFICE_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

posts = pd.read_csv(REDDIT_POSTS_FILE, encoding="latin1")
comments = pd.read_csv(REDDIT_COMMENTS_FILE, encoding="latin1")
google = pd.read_csv(GOOGLE_TRENDS_FILE, encoding="latin1")
imdb = pd.read_csv(IMDB_FILE, encoding="latin1")
box = pd.read_csv(BOX_OFFICE_FILE, encoding="latin1")

print(f"Posts rows: {len(posts)}")
print(f"Comments rows: {len(comments)}")
print(f"Google Trends rows: {len(google)}")
print(f"IMDb rows: {len(imdb)}")
print(f"Box office rows: {len(box)}")


# ============================================================
# Cleaning Reddit posts
# ============================================================
print_section("PREPARING REDDIT POSTS")

if not posts.empty:
    posts["title"] = posts["title"].fillna("").astype(str)
    posts["selftext"] = posts["selftext"].fillna("").astype(str)
    posts["text"] = (posts["title"] + " " + posts["selftext"]).str.strip().apply(preprocess_text)
    posts["date"] = safe_to_datetime(posts["date"])
    posts = posts.dropna(subset=["movie", "date"]).copy()
    posts = posts[posts["text"].str.strip() != ""].copy()
    posts["sentiment_score"] = posts["text"].apply(get_sentiment_score)
    posts["sentiment_label"] = posts["sentiment_score"].apply(sentiment_label)
    posts["source"] = "post"
    posts["week_start"] = to_week_start(posts["date"])
else:
    posts = pd.DataFrame(columns=[
        "movie", "text", "date", "sentiment_score", "sentiment_label",
        "source", "week_start", "post_id", "author"
    ])


# ============================================================
# Cleaning Reddit comments
# ============================================================
print_section("PREPARING REDDIT COMMENTS")

if not comments.empty:
    comments["comment_text"] = comments["comment_text"].fillna("").astype(str)
    comments["text"] = comments["comment_text"].apply(preprocess_text)
    comments["date"] = safe_to_datetime(comments["date"])
    comments = comments.dropna(subset=["movie", "date"]).copy()
    comments = comments[comments["text"].str.strip() != ""].copy()
    comments["sentiment_score"] = comments["text"].apply(get_sentiment_score)
    comments["sentiment_label"] = comments["sentiment_score"].apply(sentiment_label)
    comments["source"] = "comment"
    comments["week_start"] = to_week_start(comments["date"])
else:
    comments = pd.DataFrame(columns=[
        "movie", "text", "date", "sentiment_score", "sentiment_label",
        "source", "week_start", "comment_id", "author", "parent_id", "post_id"
    ])


# ============================================================
# Bringing all Reddit text together
# ============================================================
print_section("COMBINING REDDIT DATA")

reddit_all = pd.concat(
    [
        posts[["movie", "date", "week_start", "text", "source", "sentiment_score", "sentiment_label"]],
        comments[["movie", "date", "week_start", "text", "source", "sentiment_score", "sentiment_label"]],
    ],
    ignore_index=True
)

reddit_all = reddit_all.dropna(subset=["movie", "date", "week_start"]).copy()
print(f"Combined Reddit rows: {len(reddit_all)}")


# ============================================================
# Building daily Reddit features
# ============================================================
print_section("BUILDING DAILY REDDIT FEATURES")

if not reddit_all.empty:
    reddit_all["day"] = reddit_all["date"].dt.tz_convert(None).dt.floor("D")

    daily_reddit = (
        reddit_all
        .groupby(["movie", "day"], as_index=False)
        .agg(
            mentions=("text", "count"),
            n_posts=("source", lambda x: (x == "post").sum()),
            n_comments=("source", lambda x: (x == "comment").sum()),
            avg_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda x: (x == "positive").mean()),
            negative_share=("sentiment_label", lambda x: (x == "negative").mean()),
            neutral_share=("sentiment_label", lambda x: (x == "neutral").mean()),
        )
    )
    daily_reddit["net_sentiment"] = daily_reddit["positive_share"] - daily_reddit["negative_share"]
else:
    daily_reddit = pd.DataFrame()


# ============================================================
# Building weekly Reddit features
# ============================================================
print_section("BUILDING WEEKLY REDDIT FEATURES")

if not reddit_all.empty:
    weekly_reddit = (
        reddit_all
        .groupby(["movie", "week_start"], as_index=False)
        .agg(
            reddit_mentions=("text", "count"),
            reddit_posts=("source", lambda x: (x == "post").sum()),
            reddit_comments=("source", lambda x: (x == "comment").sum()),
            reddit_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda x: (x == "positive").mean()),
            negative_share=("sentiment_label", lambda x: (x == "negative").mean()),
            neutral_share=("sentiment_label", lambda x: (x == "neutral").mean()),
        )
    )
    weekly_reddit["net_sentiment"] = weekly_reddit["positive_share"] - weekly_reddit["negative_share"]
else:
    weekly_reddit = pd.DataFrame(columns=[
        "movie", "week_start", "reddit_mentions", "reddit_posts", "reddit_comments",
        "reddit_sentiment", "positive_share", "negative_share", "neutral_share", "net_sentiment"
    ])


# ============================================================
# Preparing Google Trends
# ============================================================
print_section("PREPARING GOOGLE TRENDS")

google["date"] = safe_to_datetime(google["date"])
google["trend_score"] = pd.to_numeric(google["trend_score"], errors="coerce")
google["week_start"] = to_week_start(google["date"])
google = google.dropna(subset=["movie", "date", "trend_score", "week_start"]).copy()

weekly_google = (
    google
    .groupby(["movie", "week_start"], as_index=False)
    .agg(google_trends=("trend_score", "mean"))
)


# ============================================================
# Preparing IMDb snapshot variables
# ============================================================
print_section("PREPARING IMDb SNAPSHOT")

imdb["imdb_rating"] = pd.to_numeric(imdb["imdb_rating"], errors="coerce")
imdb["imdb_votes"] = pd.to_numeric(imdb["imdb_votes"], errors="coerce")
if "imdb_runtime_minutes" in imdb.columns:
    imdb["imdb_runtime_minutes"] = pd.to_numeric(imdb["imdb_runtime_minutes"], errors="coerce")

agg_map = {
    "imdb_rating": ("imdb_rating", "mean"),
    "imdb_votes": ("imdb_votes", "mean"),
}
if "imdb_runtime_minutes" in imdb.columns:
    agg_map["imdb_runtime_minutes"] = ("imdb_runtime_minutes", "first")
if "imdb_genres" in imdb.columns:
    agg_map["imdb_genres"] = ("imdb_genres", "first")
if "imdb_primary_title" in imdb.columns:
    agg_map["imdb_primary_title"] = ("imdb_primary_title", "first")

imdb_snapshot = (
    imdb
    .dropna(subset=["movie"])
    .groupby("movie", as_index=False)
    .agg(**agg_map)
)


# ============================================================
# Preparing weekly box office
# ============================================================
print_section("PREPARING WEEKLY BOX OFFICE")

if box.empty:
    weekly_boxoffice = pd.DataFrame(columns=["movie", "week_start", "weekly_box_office", "total_box_office"])
else:
    box["date"] = safe_to_datetime(box["date"])
    box["weekly_box_office"] = pd.to_numeric(box["weekly_box_office"], errors="coerce")
    box["total_box_office"] = pd.to_numeric(box["total_box_office"], errors="coerce")
    box["week_start"] = to_week_start(box["date"])
    box = box.dropna(subset=["movie", "date", "week_start"]).copy()

    weekly_boxoffice = (
        box
        .groupby(["movie", "week_start"], as_index=False)
        .agg(
            weekly_box_office=("weekly_box_office", "sum"),
            total_box_office=("total_box_office", "max")
        )
    )


# ============================================================
# Building the descriptive weekly panel
# ============================================================
print_section("BUILDING WEEKLY DESCRIPTIVE PANEL")

weekly_panel = weekly_google.merge(weekly_reddit, on=["movie", "week_start"], how="left")
weekly_panel = weekly_panel.merge(weekly_boxoffice, on=["movie", "week_start"], how="left")
weekly_panel = weekly_panel.merge(imdb_snapshot, on="movie", how="left")

for col in ["reddit_mentions", "reddit_posts", "reddit_comments"]:
    if col in weekly_panel.columns:
        weekly_panel[col] = weekly_panel[col].fillna(0)

weekly_panel = weekly_panel.sort_values(["movie", "week_start"]).reset_index(drop=True)


# ============================================================
# Looking at simple correlations
# ============================================================
print_section("CORRELATION ANALYSIS")

corr_rows = []
for movie, df_movie in weekly_panel.groupby("movie"):
    corr_rows.append({
        "movie": movie,
        "corr_google_trends_vs_box_office": safe_corr(df_movie, "google_trends", "weekly_box_office"),
        "corr_reddit_mentions_vs_box_office": safe_corr(df_movie, "reddit_mentions", "weekly_box_office"),
        "corr_reddit_sentiment_vs_box_office": safe_corr(df_movie, "reddit_sentiment", "weekly_box_office"),
        "corr_reddit_sentiment_vs_google_trends": safe_corr(df_movie, "reddit_sentiment", "google_trends"),
    })

correlation_summary = pd.DataFrame(corr_rows)
print(correlation_summary.to_string(index=False))


# ============================================================
# Cleaning text more aggressively for word clouds
# ============================================================
print_section("IMPROVING WORD CLOUD TEXT")

wordcloud_source = pd.concat(
    [
        posts[["movie", "text"]].copy() if not posts.empty else pd.DataFrame(columns=["movie", "text"]),
        comments[["movie", "text"]].copy() if not comments.empty else pd.DataFrame(columns=["movie", "text"]),
    ],
    ignore_index=True
)

wordcloud_source = wordcloud_source.dropna(subset=["movie", "text"]).copy()
wordcloud_source = wordcloud_source[wordcloud_source["text"].str.strip() != ""].copy()

wordcloud_source["tokens_initial"] = wordcloud_source["text"].apply(
    lambda x: tokenize_clean(x, ALL_WORDCLOUD_STOPWORDS)
)

token_lists = wordcloud_source["tokens_initial"].tolist()
cleaned_token_lists, removed_top_words = remove_top_frequent_words(token_lists, top_n=40)
wordcloud_source["tokens_after_global"] = cleaned_token_lists

print("Top 40 high-frequency words removed from word clouds:")
print(sorted(removed_top_words))

final_tokens = []
for _, row in wordcloud_source.iterrows():
    movie_name = row["movie"]
    tokens = row["tokens_after_global"]
    movie_stopwords = MOVIE_SPECIFIC_STOPWORDS.get(movie_name, set())
    tokens = [token for token in tokens if token not in movie_stopwords]
    final_tokens.append(tokens)

wordcloud_source["tokens_final"] = final_tokens


# ============================================================
# Generating the cleaned word clouds
# ============================================================
print_section("GENERATING WORD CLOUDS")

overall_tokens = [token for tokens in wordcloud_source["tokens_final"] for token in tokens]
generate_wordcloud_from_tokens(
    overall_tokens,
    "Word Cloud - Overall (Improved)",
    PLOTS_DIR / "wordcloud_overall_improved.png"
)

movie_plot_titles = {
    "Dune 2": "Word Cloud - Dune 2",
    "Deadpool & Wolverine": "Word Cloud - Deadpool & Wolverine"
}

for movie_name, plot_title in movie_plot_titles.items():
    movie_tokens = [
        token
        for tokens in wordcloud_source.loc[wordcloud_source["movie"] == movie_name, "tokens_final"]
        for token in tokens
    ]
    safe_name = movie_name.lower().replace(" ", "_").replace("&", "and")
    generate_wordcloud_from_tokens(
        movie_tokens,
        plot_title,
        PLOTS_DIR / f"wordcloud_{safe_name}_improved.png"
    )


# ============================================================
# Running topic modelling with LDA
# ============================================================
print_section("TOPIC MODELING")

topic_rows = []
topic_weight_rows = []

if not reddit_all.empty:
    for movie in reddit_all["movie"].dropna().unique():
        df_movie = reddit_all[reddit_all["movie"] == movie].copy()
        docs = df_movie["text"].dropna().astype(str).tolist()

        if len(docs) < 10:
            continue

        vectorizer = CountVectorizer(
            stop_words=list(ALL_STOPWORDS),
            max_features=500,
            min_df=2,
            max_df=0.90,
        )
        X = vectorizer.fit_transform(docs)

        if X.shape[1] < 5:
            continue

        lda = LatentDirichletAllocation(
            n_components=3,
            random_state=42,
            learning_method="batch"
        )
        doc_topic_matrix = lda.fit_transform(X)

        feature_names = vectorizer.get_feature_names_out().tolist()
        movie_topics = get_top_words_from_lda(lda, feature_names, n_top_words=10)
        movie_topics["movie"] = movie
        topic_rows.append(movie_topics)

        avg_topic_weights = doc_topic_matrix.mean(axis=0)
        for i, weight in enumerate(avg_topic_weights):
            topic_weight_rows.append({
                "movie": movie,
                "topic": f"Topic_{i + 1}",
                "avg_topic_weight": float(weight)
            })

if topic_rows:
    topic_summary = pd.concat(topic_rows, ignore_index=True)
else:
    topic_summary = pd.DataFrame(columns=["topic", "top_words", "movie"])

if topic_weight_rows:
    topic_weights = pd.DataFrame(topic_weight_rows)
else:
    topic_weights = pd.DataFrame(columns=["movie", "topic", "avg_topic_weight"])


# ============================================================
# Building a simple reply network
# ============================================================
print_section("NETWORK ANALYSIS")

network_rows = []

if not comments.empty and "author" in comments.columns and "parent_id" in comments.columns:
    post_author_lookup = {}
    if not posts.empty and "post_id" in posts.columns and "author" in posts.columns:
        post_author_lookup = dict(zip(posts["post_id"], posts["author"]))

    comment_author_lookup = {}
    if "comment_id" in comments.columns and "author" in comments.columns:
        comment_author_lookup = dict(zip(comments["comment_id"], comments["author"]))

    edges = []
    for _, row in comments.iterrows():
        child_author = row.get("author")
        parent_id = row.get("parent_id")
        movie = row.get("movie")

        if pd.isna(child_author) or pd.isna(parent_id):
            continue

        parent_author = None
        parent_id = str(parent_id)

        if parent_id.startswith("t3_"):
            parent_post_id = parent_id.replace("t3_", "")
            parent_author = post_author_lookup.get(parent_post_id)
        elif parent_id.startswith("t1_"):
            parent_comment_id = parent_id.replace("t1_", "")
            parent_author = comment_author_lookup.get(parent_comment_id)

        if parent_author and parent_author != child_author:
            edges.append((child_author, parent_author, movie))

    edges_df = pd.DataFrame(edges, columns=["source_author", "target_author", "movie"])

    if not edges_df.empty:
        for movie, df_movie in edges_df.groupby("movie"):
            G = nx.DiGraph()
            for _, r in df_movie.iterrows():
                G.add_edge(r["source_author"], r["target_author"])

            if len(G.nodes) > 0:
                degree_centrality = nx.degree_centrality(G)
                betweenness = nx.betweenness_centrality(G)

                for node in G.nodes:
                    network_rows.append({
                        "movie": movie,
                        "author": node,
                        "degree_centrality": degree_centrality.get(node, 0),
                        "betweenness_centrality": betweenness.get(node, 0),
                        "in_degree": G.in_degree(node),
                        "out_degree": G.out_degree(node),
                    })

network_summary = pd.DataFrame(network_rows)
if not network_summary.empty:
    network_summary = network_summary.sort_values(
        ["movie", "betweenness_centrality", "degree_centrality"],
        ascending=[True, False, False]
    )


# ============================================================
# Making the plots
# ============================================================
print_section("CREATING PLOTS")

save_line_plot(
    weekly_google,
    x_col="week_start",
    y_col="google_trends",
    group_col="movie",
    title="Weekly Google Trends by Movie",
    xlabel="Week",
    ylabel="Trend Score",
    save_path=PLOTS_DIR / "weekly_google_trends_by_movie.png",
)

if not weekly_reddit.empty:
    save_line_plot(
        weekly_reddit,
        x_col="week_start",
        y_col="reddit_mentions",
        group_col="movie",
        title="Weekly Reddit Mentions by Movie",
        xlabel="Week",
        ylabel="Mentions",
        save_path=PLOTS_DIR / "weekly_reddit_mentions_by_movie.png",
    )

    save_line_plot(
        weekly_reddit,
        x_col="week_start",
        y_col="reddit_sentiment",
        group_col="movie",
        title="Weekly Reddit Sentiment by Movie",
        xlabel="Week",
        ylabel="Average Sentiment",
        save_path=PLOTS_DIR / "weekly_reddit_sentiment_by_movie.png",
    )

if not weekly_boxoffice.empty:
    save_line_plot(
        weekly_boxoffice,
        x_col="week_start",
        y_col="weekly_box_office",
        group_col="movie",
        title="Weekly Box Office by Movie",
        xlabel="Week",
        ylabel="Weekly Box Office",
        save_path=PLOTS_DIR / "weekly_box_office_by_movie.png",
    )

if not imdb_snapshot.empty and "imdb_rating" in imdb_snapshot.columns:
    imdb_plot_df = imdb_snapshot.dropna(subset=["movie", "imdb_rating"])
    if not imdb_plot_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(imdb_plot_df["movie"], imdb_plot_df["imdb_rating"])
        plt.title("IMDb Rating by Movie")
        plt.xlabel("Movie")
        plt.ylabel("IMDb Rating")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "imdb_rating_by_movie.png")
        plt.close()

if not topic_weights.empty:
    for movie, df_movie in topic_weights.groupby("movie"):
        plt.figure(figsize=(8, 5))
        plt.bar(df_movie["topic"], df_movie["avg_topic_weight"])
        plt.title(f"Average LDA Topic Weights - {movie}")
        plt.xlabel("Topic")
        plt.ylabel("Average Topic Weight")
        plt.tight_layout()
        safe_name = movie.lower().replace(" ", "_").replace("&", "and")
        plt.savefig(PLOTS_DIR / f"topic_weights_{safe_name}.png")
        plt.close()

if not network_summary.empty:
    for movie, df_movie in network_summary.groupby("movie"):
        top_nodes = df_movie.head(10).copy()
        plt.figure(figsize=(10, 6))
        plt.barh(top_nodes["author"], top_nodes["betweenness_centrality"])
        plt.title(f"Top Reddit Users by Betweenness Centrality - {movie}")
        plt.xlabel("Betweenness Centrality")
        plt.ylabel("Author")
        plt.tight_layout()
        safe_name = movie.lower().replace(" ", "_").replace("&", "and")
        plt.savefig(PLOTS_DIR / f"network_top_betweenness_{safe_name}.png")
        plt.close()


# ============================================================
# Saving all outputs
# ============================================================
print_section("SAVING OUTPUTS")

reddit_all = round_numeric_columns(reddit_all)
daily_reddit = round_numeric_columns(daily_reddit)
weekly_reddit = round_numeric_columns(weekly_reddit)
weekly_google = round_numeric_columns(weekly_google)
imdb_snapshot = round_numeric_columns(imdb_snapshot)
weekly_boxoffice = round_numeric_columns(weekly_boxoffice)
weekly_panel = round_numeric_columns(weekly_panel)
correlation_summary = round_numeric_columns(correlation_summary)
topic_summary = round_numeric_columns(topic_summary)
topic_weights = round_numeric_columns(topic_weights)
network_summary = round_numeric_columns(network_summary)

reddit_all.to_csv(OUTPUT_DIR / "reddit_all_scored.csv", index=False)
daily_reddit.to_csv(OUTPUT_DIR / "daily_reddit_features.csv", index=False)
weekly_reddit.to_csv(OUTPUT_DIR / "weekly_reddit_features.csv", index=False)
weekly_google.to_csv(OUTPUT_DIR / "weekly_google_trends.csv", index=False)
imdb_snapshot.to_csv(OUTPUT_DIR / "imdb_snapshot_features.csv", index=False)
weekly_boxoffice.to_csv(OUTPUT_DIR / "weekly_boxoffice_features.csv", index=False)
weekly_panel.to_csv(DESCRIPTIVE_PANEL_FILE, index=False)
correlation_summary.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)
topic_summary.to_csv(OUTPUT_DIR / "topic_summary.csv", index=False)
topic_weights.to_csv(OUTPUT_DIR / "topic_weights.csv", index=False)
network_summary.to_csv(OUTPUT_DIR / "network_summary.csv", index=False)

print("Saved descriptive outputs successfully.")