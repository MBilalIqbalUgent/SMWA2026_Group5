import requests
import pandas as pd
import time
import os
from datetime import datetime

APIFY_TOKEN = os.getenv("APIFY_TOKEN")

if not APIFY_TOKEN:
    raise ValueError("APIFY_TOKEN not found. Set it in PowerShell before running.")

REDDIT_SCRAPER_URL = "https://api.apify.com/v2/acts/apify~reddit-scraper/run-sync-get-dataset-items"
IMDB_SCRAPER_URL = "https://api.apify.com/v2/acts/john-doe~imdb-reviews-scraper/run-sync-get-dataset-items"


def process_movie_data(imdb_item, review_item=None, platform="IMDb"):
    def clean_money(value):
        if value and isinstance(value, str):
            try:
                return float(value.replace("$", "").replace(",", "")) / 1_000_000
            except Exception:
                return None
        return value

    def extract_year(timestamp):
        if timestamp:
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "")).year
            except Exception:
                return None
        return None

    budget = clean_money(imdb_item.get("budget"))
    boxoffice = clean_money(imdb_item.get("grossWorldwide"))

    roi = None
    if budget and budget != 0 and boxoffice:
        roi = round(boxoffice / budget, 2)

    return {
        "Movie Title": imdb_item.get("title"),
        "Platform": platform,
        "Review_Text": review_item.get("body") if review_item else None,
        "Year": imdb_item.get("year"),
        "Budget_M": budget,
        "BoxOffice_M": boxoffice,
        "IMDb_Rating": imdb_item.get("rating"),
        "Reviewer": review_item.get("author") if review_item else "IMDb",
        "ROI": roi,
        "Review_Year": extract_year(review_item.get("createdAt")) if review_item else None,
        "Review_Rating": review_item.get("rating") if review_item else None
    }


def extract_imdb_data(movie_list):
    print("Fetching IMDb Data...")
    imdb_results = {}

    for movie in movie_list:
        payload = {
            "searchMode": "movie",
            "searchQuery": movie
        }
        params = {"token": APIFY_TOKEN}

        try:
            response = requests.post(IMDB_SCRAPER_URL, json=payload, params=params, timeout=60)

            print(f"\n[IMDb] Movie: {movie}")
            print("Status code:", response.status_code)
            print("Response preview:", response.text[:500])

            if response.status_code == 200:
                data = response.json()
                if data:
                    imdb_results[movie] = data[0]
                    print(f"✅ IMDb fetched: {movie}")
                else:
                    print(f"⚠️ No IMDb data returned: {movie}")
            else:
                print(f"❌ IMDb failed: {movie}")

        except Exception as e:
            print(f"❌ Error IMDb ({movie}): {e}")

        time.sleep(1)

    return imdb_results


def extract_reddit_data(movie_list):
    print("\nFetching Reddit Data...")
    reddit_results = {}

    for movie in movie_list:
        payload = {
            "subreddits": ["movies", "boxoffice"],
            "searchQuery": f"{movie} review",
            "type": "comment",
            "maxItems": 20
        }
        params = {"token": APIFY_TOKEN}

        try:
            response = requests.post(REDDIT_SCRAPER_URL, json=payload, params=params, timeout=60)

            print(f"\n[Reddit] Movie: {movie}")
            print("Status code:", response.status_code)
            print("Response preview:", response.text[:500])

            if response.status_code == 200:
                data = response.json()
                reddit_results[movie] = data if isinstance(data, list) else []
                print(f"✅ Reddit fetched: {movie}")
            else:
                print(f"❌ Reddit failed: {movie}")

        except Exception as e:
            print(f"❌ Error Reddit ({movie}): {e}")

        time.sleep(1)

    return reddit_results


def merge_and_export(imdb_data, reddit_data):
    print("\nMerging Data...")
    final_rows = []

    for movie, imdb_item in imdb_data.items():
        final_rows.append(process_movie_data(imdb_item, platform="IMDb"))

        reviews = reddit_data.get(movie, [])
        for review in reviews:
            final_rows.append(process_movie_data(imdb_item, review, platform="Reddit"))

    if not final_rows:
        print("❌ No data fetched. Skipping CSV export.")
        return

    df = pd.DataFrame(final_rows)

    expected_columns = [
        "Movie Title", "Platform", "Review_Text", "Year",
        "Budget_M", "BoxOffice_M", "IMDb_Rating",
        "Reviewer", "ROI", "Review_Year", "Review_Rating"
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    df = df[expected_columns]

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/movie_data_professional.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ CSV Generated: {output_path}")


if __name__ == "__main__":
    target_movies = ["Inception", "The Dark Knight"]

    imdb_data = extract_imdb_data(target_movies)
    reddit_data = extract_reddit_data(target_movies)

    print("\nIMDb movies fetched:", list(imdb_data.keys()))
    print("Reddit movies fetched:", list(reddit_data.keys()))

    merge_and_export(imdb_data, reddit_data)

    print("\n==============================")
    print("DATA PIPELINE COMPLETE")
    print("==============================")