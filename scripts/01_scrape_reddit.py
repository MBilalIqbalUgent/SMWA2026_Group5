from pathlib import Path
from time import sleep
import pandas as pd
from pytrends.request import TrendReq

# ============================================
# 01_scrape_reddit.py
# Checks Reddit raw files and re-scrapes Google Trends
# for the same period as the Reddit dataset
# ============================================

# -------------------------
# 1. PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

posts_file = DATA_DIR / "reddit_posts_raw.csv"
comments_file = DATA_DIR / "reddit_comments_raw.csv"
google_file = DATA_DIR / "google_trends_raw.csv"

print("\n=== DATA COLLECTION SETUP ===\n")
print(f"Raw data folder: {DATA_DIR}\n")

# -------------------------
# 2. CHECK REDDIT FILES
# -------------------------
print("Checking Reddit raw files...\n")

if posts_file.exists():
    print("✔ reddit_posts_raw.csv found")
else:
    raise FileNotFoundError("reddit_posts_raw.csv not found in data/raw")

if comments_file.exists():
    print("✔ reddit_comments_raw.csv found")
else:
    raise FileNotFoundError("reddit_comments_raw.csv not found in data/raw")

# -------------------------
# 3. LOAD REDDIT FILES TO DETERMINE DATE RANGE
# -------------------------
posts = pd.read_csv(posts_file, encoding="latin1")
comments = pd.read_csv(comments_file, encoding="latin1")

posts["date"] = pd.to_datetime(posts["date"], errors="coerce")
comments["date"] = pd.to_datetime(comments["date"], errors="coerce")

reddit_dates = pd.concat([posts["date"], comments["date"]], ignore_index=True).dropna()

if reddit_dates.empty:
    raise ValueError("No valid dates found in Reddit raw files.")

reddit_start = reddit_dates.min().date()
reddit_end = reddit_dates.max().date()

print(f"\nReddit date range detected: {reddit_start} to {reddit_end}")

# pytrends timeframe format: YYYY-MM-DD YYYY-MM-DD
timeframe = f"{reddit_start} {reddit_end}"
print(f"Google Trends timeframe to scrape: {timeframe}")

# -------------------------
# 4. GOOGLE TRENDS SETTINGS
# -------------------------
# Change geo if your assignment needs a specific market:
# "" = worldwide
# "US" = United States
# "BE" = Belgium
# "AE" = UAE
GEO = ""
CATEGORY = 0
GPROP = ""

# Map keywords to canonical movie labels
movie_queries = {
    "Deadpool & Wolverine": [
        "Deadpool & Wolverine",
        "Deadpool Wolverine"
    ],
    "Dune 2": [
        "Dune 2",
        "Dune Part Two"
    ]
}

# -------------------------
# 5. SCRAPE GOOGLE TRENDS
# -------------------------
print("\nScraping Google Trends...\n")

pytrends = TrendReq(hl="en-US", tz=0)

all_trends = []

for movie_label, query_list in movie_queries.items():
    for query in query_list:
        print(f"Scraping query: {query}")

        try:
            pytrends.build_payload(
                kw_list=[query],
                cat=CATEGORY,
                timeframe=timeframe,
                geo=GEO,
                gprop=GPROP
            )

            interest = pytrends.interest_over_time()

            if interest.empty:
                print(f"  No Google Trends data returned for: {query}")
                sleep(2)
                continue

            interest = interest.reset_index()

            # Remove 'isPartial' if present
            if "isPartial" in interest.columns:
                interest = interest.drop(columns=["isPartial"])

            # The keyword column will have the same name as the query
            if query not in interest.columns:
                print(f"  Could not find expected query column for: {query}")
                sleep(2)
                continue

            temp = interest.rename(columns={query: "hits"}).copy()
            temp["keyword"] = query
            temp["geo"] = GEO
            temp["time"] = timeframe
            temp["gprop"] = GPROP
            temp["category"] = CATEGORY
            temp["movie"] = movie_label

            # Keep consistent columns
            temp = temp[["date", "hits", "keyword", "geo", "time", "gprop", "category", "movie"]]

            all_trends.append(temp)

            print(f"  Collected {len(temp)} rows for {query}")

            # polite pause to reduce rate-limit issues
            sleep(2)

        except Exception as e:
            print(f"  Failed for {query}: {e}")
            sleep(3)

# -------------------------
# 6. SAVE GOOGLE TRENDS
# -------------------------
if not all_trends:
    raise ValueError("No Google Trends data was collected. Nothing to save.")

google_trends_raw = pd.concat(all_trends, ignore_index=True)

# Clean and standardize
google_trends_raw["date"] = pd.to_datetime(google_trends_raw["date"], errors="coerce")
google_trends_raw["hits"] = google_trends_raw["hits"].astype(str).replace("<1", "0")
google_trends_raw["hits"] = pd.to_numeric(google_trends_raw["hits"], errors="coerce")

# Drop duplicates if any
google_trends_raw = google_trends_raw.drop_duplicates()

# Save fresh Google Trends file
google_trends_raw.to_csv(google_file, index=False)

print("\n✔ google_trends_raw.csv saved successfully")
print(f"Saved to: {google_file}")

# -------------------------
# 7. SUMMARY
# -------------------------
print("\n=== SUMMARY ===")
print(f"Reddit period used: {reddit_start} to {reddit_end}")
print(f"Google Trends rows saved: {len(google_trends_raw)}")

print("\nRows per movie:")
print(google_trends_raw["movie"].value_counts(dropna=False))

print("\nDone. Next step:")
print("Run: python scripts/02_descriptive_analysis.py")
print("Then: python scripts/03_predictive_analysis.py")