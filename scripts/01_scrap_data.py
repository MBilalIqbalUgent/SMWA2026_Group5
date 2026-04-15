from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from pytrends.request import TrendReq


# ============================================================
# Getting folders and file paths ready
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_POSTS_FILE = RAW_DIR / "reddit_posts_raw.csv"
REDDIT_COMMENTS_FILE = RAW_DIR / "reddit_comments_raw.csv"
GOOGLE_TRENDS_FILE = RAW_DIR / "google_trends_raw.csv"
IMDB_FILE = RAW_DIR / "imdb_movie_metrics_raw.csv"
BOX_OFFICE_FILE = RAW_DIR / "weekly_box_office_raw.csv"
MERGED_FILE = PROCESSED_DIR / "movie_weekly_panel.csv"


# ============================================================
# Main settings for the project
# ============================================================
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)

STUDY_START = pd.Timestamp("2024-02-01", tz="UTC")
STUDY_END = pd.Timestamp("2024-09-30 23:59:59", tz="UTC")

MOVIE_CONFIG = {
    "Deadpool & Wolverine": {
        "keywords": [
            "deadpool",
            "wolverine",
            "deadpool and wolverine",
            "deadpool & wolverine",
        ],
        "reddit_subreddits": [
            "movies",
            "boxoffice",
            "MarvelStudios",
            "comicbookmovies",
            "flicks",
        ],
        "imdb_titles": ["Deadpool & Wolverine"],
        "box_office_titles": [
            "Deadpool & Wolverine",
            "Deadpool and Wolverine",
        ],
    },
    "Dune 2": {
        "keywords": [
            "dune 2",
            "dune part two",
            "dune part 2",
            "dune: part two",
        ],
        "reddit_subreddits": [
            "movies",
            "boxoffice",
            "dune",
            "flicks",
            "Letterboxd",
        ],
        "imdb_titles": ["Dune: Part Two"],
        "box_office_titles": [
            "Dune: Part Two",
            "Dune Part Two",
            "Dune 2",
        ],
    },
}

GOOGLE_TRENDS_MOVIES = ["Deadpool & Wolverine", "Dune 2"]
GT_TIMEFRAME = "2024-02-01 2024-09-30"

REDDIT_SEARCH_LIMIT = 100
REDDIT_COMMENTS_PER_MOVIE = 12
REDDIT_COMMENT_LIMIT_PER_POST = 80
REDDIT_SLEEP = 2.0
REDDIT_MAX_RETRIES = 6
REDDIT_BACKOFF = 2.0

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"


# ============================================================
# Small helper functions
# ============================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}\n")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = text.replace(":", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_week_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)
    return dt.dt.to_period("W-SUN").dt.start_time


def parse_money(value) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "-", "n/a", "nan", "N/A"}:
        return None
    text = text.replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def get_json_with_backoff(url: str, params: dict | None = None, timeout: int = 30):
    delay = REDDIT_SLEEP

    for attempt in range(REDDIT_MAX_RETRIES):
        response = SESSION.get(url, params=params, timeout=timeout)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            print(
                f"429 hit. Sleeping {delay:.1f}s before retry "
                f"{attempt + 1}/{REDDIT_MAX_RETRIES}"
            )
            time.sleep(delay)
            delay *= REDDIT_BACKOFF
            continue

        response.raise_for_status()

    raise requests.HTTPError(f"Failed after retries: {url}")


def score_post_relevance(text: str, keywords: list[str]) -> int:
    norm = normalize_text(text)
    return sum(1 for kw in keywords if kw in norm)


def within_study_period(dt: pd.Timestamp) -> bool:
    if pd.isna(dt):
        return False
    return STUDY_START <= dt <= STUDY_END


def generate_fridays(start_date: str, end_date: str) -> list[pd.Timestamp]:
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    return [d for d in all_days if d.weekday() == 4]


# ============================================================
# Pulling Reddit posts
# ============================================================
def scrape_reddit_posts() -> pd.DataFrame:
    print_section("SCRAPING REDDIT POSTS")

    rows = []
    seen_ids = set()

    for movie, cfg in MOVIE_CONFIG.items():
        keywords_norm = [normalize_text(k) for k in cfg["keywords"]]

        for subreddit in cfg["reddit_subreddits"]:
            for keyword in cfg["keywords"]:
                print(f"Movie: {movie} | r/{subreddit} | query={keyword}")

                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": keyword,
                    "restrict_sr": 1,
                    "sort": "new",
                    "t": "all",
                    "limit": REDDIT_SEARCH_LIMIT,
                    "type": "link",
                }

                try:
                    payload = get_json_with_backoff(url, params=params)
                    children = payload.get("data", {}).get("children", [])

                    for item in children:
                        post = item.get("data", {})
                        post_id = post.get("id")
                        created_utc = post.get("created_utc")

                        if not post_id or created_utc is None:
                            continue
                        if post_id in seen_ids:
                            continue

                        post_dt = pd.to_datetime(created_utc, unit="s", utc=True)
                        if not within_study_period(post_dt):
                            continue

                        title = post.get("title", "") or ""
                        selftext = post.get("selftext", "") or ""
                        combined = f"{title} {selftext}".strip()
                        relevance = score_post_relevance(combined, keywords_norm)

                        if relevance == 0:
                            continue

                        permalink = post.get("permalink", "")
                        full_permalink = f"https://www.reddit.com{permalink}" if permalink else None

                        seen_ids.add(post_id)
                        rows.append({
                            "movie": movie,
                            "subreddit": subreddit,
                            "query_used": keyword,
                            "post_id": post_id,
                            "title": title,
                            "selftext": selftext,
                            "author": post.get("author"),
                            "score": post.get("score"),
                            "upvote_ratio": post.get("upvote_ratio"),
                            "num_comments": post.get("num_comments"),
                            "url": post.get("url"),
                            "permalink": full_permalink,
                            "date": post_dt,
                            "relevance_score": relevance,
                        })

                    time.sleep(REDDIT_SLEEP)

                except Exception as e:
                    print(f"Failed search for subreddit={subreddit}, query={keyword}: {e}")

    posts_df = pd.DataFrame(rows)

    if posts_df.empty:
        posts_df = pd.DataFrame(columns=[
            "movie", "subreddit", "query_used", "post_id", "title", "selftext",
            "author", "score", "upvote_ratio", "num_comments", "url",
            "permalink", "date", "relevance_score"
        ])
    else:
        posts_df["score"] = pd.to_numeric(posts_df["score"], errors="coerce")
        posts_df["num_comments"] = pd.to_numeric(posts_df["num_comments"], errors="coerce")
        posts_df = (
            posts_df
            .drop_duplicates(subset=["post_id"])
            .sort_values(
                ["movie", "relevance_score", "num_comments", "score", "date"],
                ascending=[True, False, False, False, False]
            )
            .reset_index(drop=True)
        )

    posts_df.to_csv(REDDIT_POSTS_FILE, index=False)
    print(f"Saved Reddit posts: {REDDIT_POSTS_FILE}")
    print(f"Rows: {len(posts_df)}")

    return posts_df


# ============================================================
# Pulling Reddit comments from the selected posts
# ============================================================
def scrape_reddit_comments(posts_df: pd.DataFrame) -> pd.DataFrame:
    print_section("SCRAPING REDDIT COMMENTS")

    if posts_df.empty:
        comments_df = pd.DataFrame(columns=[
            "movie", "subreddit", "post_id", "comment_id", "parent_id",
            "author", "comment_text", "score", "date", "permalink"
        ])
        comments_df.to_csv(REDDIT_COMMENTS_FILE, index=False)
        return comments_df

    selected = (
        posts_df
        .sort_values(
            ["movie", "relevance_score", "num_comments", "score"],
            ascending=[True, False, False, False]
        )
        .groupby("movie", group_keys=False)
        .head(REDDIT_COMMENTS_PER_MOVIE)
        .reset_index(drop=True)
    )

    rows = []
    seen_comment_ids = set()

    for _, row in selected.iterrows():
        permalink = row.get("permalink")
        if not permalink:
            continue

        json_url = permalink.rstrip("/") + ".json"

        try:
            payload = get_json_with_backoff(json_url)
            if not isinstance(payload, list) or len(payload) < 2:
                continue

            children = payload[1].get("data", {}).get("children", [])
            stack = children[:]
            flat_comments = []

            while stack:
                node = stack.pop()
                if node.get("kind") != "t1":
                    continue

                data = node.get("data", {})
                flat_comments.append(data)

                replies = data.get("replies")
                if isinstance(replies, dict):
                    nested = replies.get("data", {}).get("children", [])
                    stack.extend(nested)

            flat_comments = sorted(
                flat_comments,
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:REDDIT_COMMENT_LIMIT_PER_POST]

            for c in flat_comments:
                comment_id = c.get("id")
                created_utc = c.get("created_utc")
                body = c.get("body")

                if not comment_id or created_utc is None or not body:
                    continue
                if body in {"[deleted]", "[removed]"}:
                    continue
                if comment_id in seen_comment_ids:
                    continue

                comment_dt = pd.to_datetime(created_utc, unit="s", utc=True)
                if not within_study_period(comment_dt):
                    continue

                seen_comment_ids.add(comment_id)

                rows.append({
                    "movie": row["movie"],
                    "subreddit": row["subreddit"],
                    "post_id": row["post_id"],
                    "comment_id": comment_id,
                    "parent_id": c.get("parent_id"),
                    "author": c.get("author"),
                    "comment_text": body,
                    "score": c.get("score"),
                    "date": comment_dt,
                    "permalink": f"https://www.reddit.com{c.get('permalink')}" if c.get("permalink") else None,
                })

            time.sleep(REDDIT_SLEEP)

        except Exception as e:
            print(f"Failed comments scrape for {json_url}: {e}")

    comments_df = pd.DataFrame(rows)

    if comments_df.empty:
        comments_df = pd.DataFrame(columns=[
            "movie", "subreddit", "post_id", "comment_id", "parent_id",
            "author", "comment_text", "score", "date", "permalink"
        ])
    else:
        comments_df["score"] = pd.to_numeric(comments_df["score"], errors="coerce")
        comments_df = (
            comments_df
            .drop_duplicates(subset=["comment_id"])
            .sort_values(["movie", "date"])
            .reset_index(drop=True)
        )

    comments_df.to_csv(REDDIT_COMMENTS_FILE, index=False)
    print(f"Saved Reddit comments: {REDDIT_COMMENTS_FILE}")
    print(f"Rows: {len(comments_df)}")

    return comments_df


# ============================================================
# Pulling Google Trends
# ============================================================
def scrape_google_trends_weekly() -> pd.DataFrame:
    print_section("SCRAPING GOOGLE TRENDS")

    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(
        kw_list=GOOGLE_TRENDS_MOVIES,
        timeframe=GT_TIMEFRAME,
        geo="",
        cat=0,
        gprop="",
    )

    trends = pytrends.interest_over_time()

    if trends.empty:
        raise ValueError("Google Trends returned no data.")

    if "isPartial" in trends.columns:
        trends = trends[~trends["isPartial"]].drop(columns=["isPartial"])

    trends = trends.reset_index().melt(
        id_vars="date",
        value_vars=GOOGLE_TRENDS_MOVIES,
        var_name="movie",
        value_name="trend_score"
    )

    trends["date"] = safe_to_datetime(trends["date"])
    trends["trend_score"] = pd.to_numeric(trends["trend_score"], errors="coerce")

    trends = (
        trends
        .dropna(subset=["date", "movie", "trend_score"])
        .sort_values(["movie", "date"])
        .reset_index(drop=True)
    )

    trends.to_csv(GOOGLE_TRENDS_FILE, index=False)
    print(f"Saved Google Trends: {GOOGLE_TRENDS_FILE}")
    print(f"Rows: {len(trends)}")

    return trends


# ============================================================
# Pulling IMDb snapshot data
# ============================================================
def scrape_imdb_snapshot() -> pd.DataFrame:
    print_section("SCRAPING IMDb SNAPSHOT")

    title_map = {}
    for canonical_movie, cfg in MOVIE_CONFIG.items():
        for imdb_title in cfg["imdb_titles"]:
            title_map[normalize_text(imdb_title)] = canonical_movie

    basics_matches = []

    basics_reader = pd.read_csv(
        IMDB_BASICS_URL,
        sep="\t",
        compression="gzip",
        usecols=[
            "tconst", "titleType", "primaryTitle", "originalTitle",
            "startYear", "runtimeMinutes", "genres"
        ],
        dtype=str,
        na_values="\\N",
        keep_default_na=True,
        chunksize=200000,
        low_memory=False,
    )

    for chunk in basics_reader:
        chunk = chunk[chunk["titleType"] == "movie"].copy()
        chunk["primary_norm"] = chunk["primaryTitle"].map(normalize_text)
        chunk["original_norm"] = chunk["originalTitle"].map(normalize_text)

        mask = chunk["primary_norm"].isin(title_map) | chunk["original_norm"].isin(title_map)
        matched = chunk.loc[mask].copy()

        if not matched.empty:
            matched["movie"] = matched.apply(
                lambda x: title_map.get(x["primary_norm"], title_map.get(x["original_norm"])),
                axis=1,
            )
            basics_matches.append(
                matched[[
                    "movie", "tconst", "primaryTitle", "originalTitle",
                    "startYear", "runtimeMinutes", "genres"
                ]]
            )

    if not basics_matches:
        raise ValueError("No movie matches found in IMDb basics.")

    basics_df = pd.concat(basics_matches, ignore_index=True).drop_duplicates(subset=["movie", "tconst"])
    tconsts = set(basics_df["tconst"].dropna())

    ratings_matches = []
    ratings_reader = pd.read_csv(
        IMDB_RATINGS_URL,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype=str,
        na_values="\\N",
        keep_default_na=True,
        chunksize=200000,
        low_memory=False,
    )

    for chunk in ratings_reader:
        matched = chunk[chunk["tconst"].isin(tconsts)].copy()
        if not matched.empty:
            ratings_matches.append(matched)

    if not ratings_matches:
        raise ValueError("No movie matches found in IMDb ratings.")

    ratings_df = pd.concat(ratings_matches, ignore_index=True).drop_duplicates(subset=["tconst"])
    imdb_df = basics_df.merge(ratings_df, on="tconst", how="left")

    imdb_df["startYear"] = pd.to_numeric(imdb_df["startYear"], errors="coerce")
    imdb_df["runtimeMinutes"] = pd.to_numeric(imdb_df["runtimeMinutes"], errors="coerce")
    imdb_df["averageRating"] = pd.to_numeric(imdb_df["averageRating"], errors="coerce")
    imdb_df["numVotes"] = pd.to_numeric(imdb_df["numVotes"], errors="coerce")

    imdb_df = imdb_df.rename(columns={
        "tconst": "imdb_tconst",
        "primaryTitle": "imdb_primary_title",
        "originalTitle": "imdb_original_title",
        "startYear": "imdb_start_year",
        "runtimeMinutes": "imdb_runtime_minutes",
        "genres": "imdb_genres",
        "averageRating": "imdb_rating",
        "numVotes": "imdb_votes",
    })

    imdb_df["imdb_snapshot_date"] = pd.Timestamp.now(tz="UTC")

    imdb_df = (
        imdb_df
        .sort_values(["movie", "imdb_votes"], ascending=[True, False])
        .drop_duplicates(subset=["movie"])
        .reset_index(drop=True)
    )

    imdb_df.to_csv(IMDB_FILE, index=False)
    print(f"Saved IMDb metrics: {IMDB_FILE}")
    print(f"Rows: {len(imdb_df)}")

    return imdb_df


# ============================================================
# Finding the right weekly box office table
# ============================================================
def clean_boxoffice_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_text(c) for c in df.columns]
    return df


def find_chart_table(html: str) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None

    candidates = []

    for table in tables:
        table = clean_boxoffice_columns(table)
        title_col = next((c for c in table.columns if "title" in c), None)
        gross_cols = [c for c in table.columns if "gross" in c]

        if title_col and gross_cols:
            candidates.append(table)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
    return candidates[0]


def get_title_column(df: pd.DataFrame) -> str | None:
    priority = ["title", "movie title"]
    for col in priority:
        if col in df.columns:
            return col
    for col in df.columns:
        if "title" in col:
            return col
    return None


def get_weekly_gross_column(df: pd.DataFrame) -> str | None:
    preferred_patterns = [
        ["weekly", "gross"],
        ["week", "gross"],
        ["gross", "week"],
        ["gross"],
    ]

    cols = list(df.columns)

    for pattern_parts in preferred_patterns:
        for col in cols:
            if "total" in col or "cume" in col:
                continue
            if all(part in col for part in pattern_parts):
                return col

    return None


def get_total_gross_column(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)

    priority_patterns = [
        ["total", "gross"],
        ["cume", "gross"],
        ["gross", "to", "date"],
        ["gross", "total"],
    ]

    for pattern_parts in priority_patterns:
        for col in cols:
            if all(part in col for part in pattern_parts):
                return col

    for col in cols:
        if "total" in col or "cume" in col:
            return col

    return None


def title_matches_movie(title_norm: str, movie_variants: list[str]) -> bool:
    if not title_norm:
        return False
    for variant in movie_variants:
        if title_norm == variant:
            return True
        if variant in title_norm:
            return True
        if title_norm in variant:
            return True
    return False


# ============================================================
# Pulling weekly box office numbers
# ============================================================
def scrape_weekly_box_office() -> pd.DataFrame:
    print_section("SCRAPING WEEKLY BOX OFFICE")

    movie_variants = {
        movie: [normalize_text(v) for v in cfg["box_office_titles"]]
        for movie, cfg in MOVIE_CONFIG.items()
    }

    rows = []
    fridays = generate_fridays("2024-02-01", "2024-09-30")

    for friday in fridays:
        url = f"https://www.the-numbers.com/box-office-chart/weekly/{friday:%Y/%m/%d}"
        print(f"Fetching {url}")

        try:
            response = SESSION.get(url, timeout=30)
            response.raise_for_status()

            chart_df = find_chart_table(response.text)
            if chart_df is None or chart_df.empty:
                print("  No usable table found.")
                continue

            title_col = get_title_column(chart_df)
            weekly_col = get_weekly_gross_column(chart_df)
            total_col = get_total_gross_column(chart_df)

            if title_col is None or weekly_col is None:
                print("  Missing title or weekly gross column.")
                continue

            chart_df["title_norm"] = chart_df[title_col].astype(str).map(normalize_text)

            week_hits = 0

            for movie, variants in movie_variants.items():
                matched = chart_df[chart_df["title_norm"].apply(lambda x: title_matches_movie(x, variants))].copy()

                if matched.empty:
                    continue

                matched["weekly_box_office"] = matched[weekly_col].apply(parse_money)
                if total_col is not None:
                    matched["total_box_office"] = matched[total_col].apply(parse_money)
                else:
                    matched["total_box_office"] = None

                matched = matched.dropna(subset=["weekly_box_office"])
                if matched.empty:
                    continue

                matched = matched.sort_values("weekly_box_office", ascending=False).head(1)
                chosen = matched.iloc[0]

                rows.append({
                    "date": pd.Timestamp(friday, tz="UTC"),
                    "movie": movie,
                    "box_office_title": chosen[title_col],
                    "weekly_box_office": chosen["weekly_box_office"],
                    "total_box_office": chosen["total_box_office"],
                    "source": "the_numbers_weekly_chart",
                })
                week_hits += 1

            print(f"  Matched movies this week: {week_hits}")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Failed for {url}: {e}")

    box_df = pd.DataFrame(rows)

    if box_df.empty:
        box_df = pd.DataFrame(columns=[
            "date", "movie", "box_office_title", "weekly_box_office",
            "total_box_office", "source"
        ])
    else:
        box_df["date"] = safe_to_datetime(box_df["date"])
        box_df["weekly_box_office"] = pd.to_numeric(box_df["weekly_box_office"], errors="coerce")
        box_df["total_box_office"] = pd.to_numeric(box_df["total_box_office"], errors="coerce")
        box_df = (
            box_df
            .drop_duplicates(subset=["date", "movie"])
            .sort_values(["movie", "date"])
            .reset_index(drop=True)
        )

    box_df.to_csv(BOX_OFFICE_FILE, index=False)
    print(f"Saved box office: {BOX_OFFICE_FILE}")
    print(f"Rows: {len(box_df)}")

    if not box_df.empty:
        print("\nWeekly box office rows per movie:")
        print(box_df.groupby("movie").size().to_string())

    return box_df


# ============================================================
# Building the weekly panel
# ============================================================
def build_weekly_panel(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    imdb_df: pd.DataFrame,
    box_df: pd.DataFrame,
) -> pd.DataFrame:
    print_section("BUILDING WEEKLY PANEL")

    trends = trends_df.copy()
    trends["week_start"] = to_week_start(trends["date"])
    trends_weekly = (
        trends
        .groupby(["movie", "week_start"], as_index=False)
        .agg(google_trends=("trend_score", "mean"))
    )

    if posts_df.empty:
        posts_weekly = pd.DataFrame(columns=["movie", "week_start", "reddit_posts"])
    else:
        posts = posts_df.copy()
        posts["date"] = safe_to_datetime(posts["date"])
        posts["week_start"] = to_week_start(posts["date"])
        posts_weekly = (
            posts
            .groupby(["movie", "week_start"], as_index=False)
            .agg(reddit_posts=("post_id", "nunique"))
        )

    if comments_df.empty:
        comments_weekly = pd.DataFrame(columns=["movie", "week_start", "reddit_comments"])
    else:
        comments = comments_df.copy()
        comments["date"] = safe_to_datetime(comments["date"])
        comments["week_start"] = to_week_start(comments["date"])
        comments_weekly = (
            comments
            .groupby(["movie", "week_start"], as_index=False)
            .agg(reddit_comments=("comment_id", "nunique"))
        )

    reddit_weekly = pd.merge(posts_weekly, comments_weekly, on=["movie", "week_start"], how="outer")
    for col in ["reddit_posts", "reddit_comments"]:
        if col in reddit_weekly.columns:
            reddit_weekly[col] = reddit_weekly[col].fillna(0)
    reddit_weekly["reddit_mentions"] = reddit_weekly["reddit_posts"] + reddit_weekly["reddit_comments"]

    if box_df.empty:
        box_weekly = pd.DataFrame(columns=["movie", "week_start", "weekly_box_office", "total_box_office"])
    else:
        box = box_df.copy()
        box["week_start"] = to_week_start(box["date"])
        box_weekly = (
            box
            .groupby(["movie", "week_start"], as_index=False)
            .agg(
                weekly_box_office=("weekly_box_office", "sum"),
                total_box_office=("total_box_office", "max"),
            )
        )

    imdb_cols = [
        "movie", "imdb_tconst", "imdb_primary_title", "imdb_original_title",
        "imdb_start_year", "imdb_runtime_minutes", "imdb_genres",
        "imdb_rating", "imdb_votes", "imdb_snapshot_date"
    ]
    imdb_small = imdb_df[imdb_cols].drop_duplicates(subset=["movie"])

    panel = trends_weekly.merge(reddit_weekly, on=["movie", "week_start"], how="left")
    panel = panel.merge(box_weekly, on=["movie", "week_start"], how="left")
    panel = panel.merge(imdb_small, on="movie", how="left")

    for col in ["reddit_posts", "reddit_comments", "reddit_mentions"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    panel = panel.sort_values(["movie", "week_start"]).reset_index(drop=True)
    panel.to_csv(MERGED_FILE, index=False)

    print(f"Saved weekly panel: {MERGED_FILE}")
    print(f"Rows: {len(panel)}")

    return panel


# ============================================================
# Running the full data collection pipeline
# ============================================================
def main():
    print_section("MOVIE DATA PIPELINE")

    print(f"Study period: {STUDY_START.date()} to {STUDY_END.date()}")
    print(f"Movies: {list(MOVIE_CONFIG.keys())}")
    print(f"Raw folder: {RAW_DIR}")
    print(f"Processed folder: {PROCESSED_DIR}")

    posts_df = scrape_reddit_posts()
    comments_df = scrape_reddit_comments(posts_df)
    trends_df = scrape_google_trends_weekly()
    imdb_df = scrape_imdb_snapshot()
    box_df = scrape_weekly_box_office()
    build_weekly_panel(posts_df, comments_df, trends_df, imdb_df, box_df)

    print_section("PIPELINE COMPLETE")
    print(f"Saved: {REDDIT_POSTS_FILE}")
    print(f"Saved: {REDDIT_COMMENTS_FILE}")
    print(f"Saved: {GOOGLE_TRENDS_FILE}")
    print(f"Saved: {IMDB_FILE}")
    print(f"Saved: {BOX_OFFICE_FILE}")
    print(f"Saved: {MERGED_FILE}")


if __name__ == "__main__":
    main()