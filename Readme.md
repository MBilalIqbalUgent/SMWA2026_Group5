# Social Media Analytics Project

## Group 5

### Predicting Movie Attention Using Reddit and Google Trends

---

## Group Members

* Muhammad Bilal Iqbal
* Ashir Naveed
* Firuza Hajiyeva

---

## 1. Project Overview

This project investigates whether online discussions on Reddit can be used to predict public attention toward movies.

The analysis focuses on two major releases:

* Deadpool & Wolverine
* Dune 2

The performance indicator used is:

* Google Trends search interest

The project follows a complete social media analytics pipeline, including data collection, text processing, feature engineering, and predictive modeling.

---

## 2. Data Sources

### Reddit Data

* Source: Reddit (scraped using RedditExtractoR in R)
* Files:

  * `reddit_posts_raw.csv`
  * `reddit_comments_raw.csv`
* Content:

  * Posts and comments mentioning selected movies
  * Includes timestamps, text, and engagement data

### Google Trends Data

* Source: Google Trends (scraped using Python `pytrends`)
* File:

  * `google_trends_raw.csv`
* Data was collected for the same time period as the Reddit dataset to ensure alignment.

---

## 3. Project Structure

```
SMWA2026_Group5/
│
├── data/
│   └── raw/
│       ├── reddit_posts_raw.csv
│       ├── reddit_comments_raw.csv
│       └── google_trends_raw.csv
│
├── scripts/
│   ├── 01_scrape_reddit.py
│   ├── 02_descriptive_analysis.py
│   └── 03_predictive_analysis.py
│
├── outputs/
│   ├── *.csv
│   └── plots/
│
└── README.md
```

---

## 4. Workflow

### Step 1 – Data Collection

Run:

```
python scripts/01_scrape_reddit.py
```

This script:

* verifies Reddit data availability
* detects the Reddit time range
* scrapes Google Trends for the same period

---

### Step 2 – Descriptive Analysis

Run:

```
python scripts/02_descriptive_analysis.py
```

This step:

* cleans Reddit text data
* assigns movie labels
* performs sentiment analysis
* aggregates data (daily and monthly)
* generates plots:

  * mentions over time
  * sentiment trends
  * Google Trends trends
  * word clouds

---

### Step 3 – Predictive Analysis

Run:

```
python scripts/03_predictive_analysis.py
```

This step:

* aggregates data to monthly level
* merges Reddit and Google Trends
* creates lagged features
* predicts next-period Google Trends using:

  * Linear Regression
  * Random Forest

Outputs include:

* model coefficients
* feature importance
* predictions
* evaluation metrics (RMSE, MAE, R²)

---

## 5. Feature Engineering

Key features used:

* Mentions volume
* Number of comments
* Average sentiment
* Net sentiment (positive − negative)
* Engagement ratio
* Text length
* Sentiment volatility
* Log-transformed mentions
* Hype score

Lagged versions of these features are used to predict future attention.

---

## 6. Model Design

Target variable:

* Next-period Google Trends (t+1)

Approach:

* Time-based split (no random shuffling)
* Train/test split based on chronological order

Models used:

* Linear Regression (baseline)
* Random Forest (captures non-linear relationships)

---

## 7. Key Insights

* Reddit discussion volume shows a relationship with search interest
* Spikes in mentions are often followed by increased Google Trends activity
* Sentiment has a weaker and less consistent effect
* Engagement (comments vs posts) provides additional signal
* Random Forest performs better in capturing complex patterns

---

## 8. Limitations

* Small dataset size (limited observations)
* Reddit scraping constraints
* Google Trends provides relative values, not absolute demand
* Results are exploratory and not intended as precise forecasts

---

## 9. Reproducibility

To reproduce the project:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run scripts in order:

```
python scripts/01_scrape_reddit.py
python scripts/02_descriptive_analysis.py
python scripts/03_predictive_analysis.py
```

---

## 10. Conclusion

This project demonstrates a complete social media analytics workflow:

* Data collection
* Text processing
* Feature engineering
* Predictive modeling

The focus is on applying correct methodology and analytical techniques rather than maximizing predictive accuracy, in line with the course requirements.
