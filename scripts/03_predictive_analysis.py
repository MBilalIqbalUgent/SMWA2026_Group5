from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# ============================================================
# Getting folders and files ready
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTIVE_PANEL_FILE = OUTPUT_DIR / "descriptive_weekly_panel.csv"
PROCESSED_PANEL_FILE = PROCESSED_DIR / "movie_weekly_panel.csv"


# ============================================================
# Small helper functions
# ============================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}\n")


def safe_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = pd.Series(y_true).dropna()
    if len(y_true) < 2:
        return np.nan
    if y_true.nunique() < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(4)
    return df


def safe_min_max(series_a: pd.Series, series_b: pd.Series):
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if a.empty or b.empty:
        return None, None
    return min(a.min(), b.min()), max(a.max(), b.max())


# ============================================================
# Loading the final panel
# ============================================================
print_section("LOADING PANEL DATA")

if DESCRIPTIVE_PANEL_FILE.exists():
    panel_path = DESCRIPTIVE_PANEL_FILE
elif PROCESSED_PANEL_FILE.exists():
    panel_path = PROCESSED_PANEL_FILE
else:
    raise FileNotFoundError("No panel file found. Run 02_descriptive_analysis.py first.")

panel = pd.read_csv(panel_path, encoding="latin1")

if panel.empty:
    raise ValueError(f"Panel file is empty: {panel_path}")

print(f"Loaded panel: {panel_path}")
print(f"Rows: {len(panel)}")
print("\nColumns:")
print(panel.columns.tolist())


# ============================================================
# Basic cleaning before modelling
# ============================================================
required_cols = {"movie", "week_start", "weekly_box_office"}
missing_required = required_cols - set(panel.columns)
if missing_required:
    raise ValueError(f"Panel is missing required columns: {missing_required}")

panel["week_start"] = pd.to_datetime(panel["week_start"], errors="coerce")
panel = panel.dropna(subset=["movie", "week_start"]).copy()
panel = panel.sort_values(["movie", "week_start"]).reset_index(drop=True)

known_numeric_cols = [
    "google_trends",
    "reddit_mentions",
    "reddit_posts",
    "reddit_comments",
    "reddit_sentiment",
    "positive_share",
    "negative_share",
    "neutral_share",
    "net_sentiment",
    "weekly_box_office",
    "total_box_office",
    "imdb_rating",
    "imdb_votes",
    "imdb_runtime_minutes",
]

for col in known_numeric_cols:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

print("\nRows per movie:")
print(panel.groupby("movie").size())

print("\nNon-null weekly_box_office per movie:")
print(panel.groupby("movie")["weekly_box_office"].apply(lambda x: x.notna().sum()))


# ============================================================
# Keeping only usable box office rows
# ============================================================
print_section("FILTERING VALID BOX OFFICE ROWS")

panel = panel.dropna(subset=["weekly_box_office"]).copy()
panel = panel[panel["weekly_box_office"] > 0].copy()
panel = panel.sort_values(["movie", "week_start"]).reset_index(drop=True)

print(f"Rows after filtering positive weekly box office: {len(panel)}")

if len(panel) < 8:
    raise ValueError("Too few usable rows with positive weekly box office.")


# ============================================================
# Creating a smaller and cleaner feature set
# ============================================================
print_section("FEATURE ENGINEERING")

for col in ["reddit_mentions", "reddit_posts", "reddit_comments"]:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

for col in ["reddit_sentiment", "positive_share", "negative_share", "neutral_share", "net_sentiment"]:
    if col in panel.columns:
        panel[f"{col}_filled"] = panel[col].fillna(0)

lag_cols = [
    "weekly_box_office",
    "google_trends",
    "reddit_mentions",
    "reddit_sentiment_filled",
]
for col in lag_cols:
    if col in panel.columns:
        panel[f"{col}_lag1"] = panel.groupby("movie")[col].shift(1)

if "weekly_box_office" in panel.columns:
    panel["log_weekly_box_office"] = np.log1p(panel["weekly_box_office"])

if "google_trends" in panel.columns:
    panel["log_google_trends"] = np.log1p(panel["google_trends"])

if "reddit_mentions" in panel.columns:
    panel["log_reddit_mentions"] = np.log1p(panel["reddit_mentions"])

if "imdb_votes" in panel.columns:
    panel["log_imdb_votes"] = np.log1p(panel["imdb_votes"])

panel = panel.replace([np.inf, -np.inf], np.nan)
print("Feature engineering complete.")


# ============================================================
# Building the next-week target
# ============================================================
print_section("BUILDING NEXT-WEEK TARGET")

target_base_col = "weekly_box_office"
analysis_goal = "predict_next_week_box_office"

panel["target_next"] = panel.groupby("movie")[target_base_col].shift(-1)
model_data = panel.dropna(subset=["target_next"]).copy()

print(f"Analysis goal: {analysis_goal}")
print(f"Rows after next-week target creation: {len(model_data)}")

if len(model_data) < 6:
    raise ValueError("Too few usable rows after creating next-week target.")


# ============================================================
# Choosing the final feature columns
# ============================================================
print_section("SELECTING FEATURES")

candidate_features = [
    "weekly_box_office_lag1",
    "google_trends_lag1",
    "reddit_mentions_lag1",
    "reddit_sentiment_filled_lag1",
    "imdb_rating",
    "log_imdb_votes",
    "imdb_runtime_minutes",
    "log_google_trends",
    "log_reddit_mentions",
]

feature_cols = [col for col in candidate_features if col in model_data.columns]

final_features = []
for col in feature_cols:
    non_null_ratio = model_data[col].notna().mean()
    if non_null_ratio >= 0.50:
        final_features.append(col)

feature_cols = final_features

if len(feature_cols) < 3:
    raise ValueError(f"Too few usable features remain: {feature_cols}")

print("Final feature columns:")
for col in feature_cols:
    print(f"- {col}")


# ============================================================
# Saving a simple correlation table
# ============================================================
print_section("BUILDING CORRELATION TABLE")

corr_cols = feature_cols + ["target_next"]
corr_table = model_data[corr_cols].corr(numeric_only=True)
corr_table = round_numeric_columns(corr_table)
corr_table.to_csv(OUTPUT_DIR / "predictive_correlation_table.csv")


# ============================================================
# Splitting the data by time
# ============================================================
print_section("TRAIN / TEST SPLIT")

unique_weeks = sorted(model_data["week_start"].dropna().unique())

if len(unique_weeks) < 4:
    raise ValueError("Not enough unique weeks for a valid time-based split.")

split_index = int(np.floor(0.8 * len(unique_weeks)))
split_index = max(1, min(split_index, len(unique_weeks) - 1))
split_week = unique_weeks[split_index - 1]

train_data = model_data[model_data["week_start"] <= split_week].copy()
test_data = model_data[model_data["week_start"] > split_week].copy()

print(f"Split week: {split_week}")
print(f"Train rows: {len(train_data)}")
print(f"Test rows: {len(test_data)}")

if len(train_data) < 4 or len(test_data) < 1:
    raise ValueError("Dataset too small after train/test split.")

X_train = train_data[feature_cols].copy()
y_train = train_data["target_next"].copy()

X_test = test_data[feature_cols].copy()
y_test = test_data["target_next"].copy()


# ============================================================
# Building a simple baseline
# ============================================================
print_section("BASELINE MODEL")

if "weekly_box_office_lag1" in test_data.columns:
    test_data["pred_baseline"] = test_data["weekly_box_office_lag1"].fillna(y_train.mean())
else:
    test_data["pred_baseline"] = y_train.mean()

baseline_rmse = np.sqrt(mean_squared_error(y_test, test_data["pred_baseline"]))
baseline_mae = mean_absolute_error(y_test, test_data["pred_baseline"])
baseline_r2 = safe_r2(y_test, test_data["pred_baseline"])


# ============================================================
# Running linear regression
# ============================================================
print_section("LINEAR REGRESSION")

lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
test_data["pred_lr"] = lr_pipeline.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, test_data["pred_lr"]))
lr_mae = mean_absolute_error(y_test, test_data["pred_lr"])
lr_r2_train = lr_pipeline.score(X_train, y_train)
lr_r2_test = safe_r2(y_test, test_data["pred_lr"])

lr_model = lr_pipeline.named_steps["model"]
lr_coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_
}).sort_values("coefficient", ascending=False)


# ============================================================
# Running random forest
# ============================================================
print_section("RANDOM FOREST")

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=4,
        min_samples_leaf=2
    ))
])

rf_pipeline.fit(X_train, y_train)
test_data["pred_rf"] = rf_pipeline.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, test_data["pred_rf"]))
rf_mae = mean_absolute_error(y_test, test_data["pred_rf"])
rf_r2_train = rf_pipeline.score(X_train, y_train)
rf_r2_test = safe_r2(y_test, test_data["pred_rf"])

rf_model = rf_pipeline.named_steps["model"]
rf_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)


# ============================================================
# Saving model outputs
# ============================================================
print_section("SAVING OUTPUTS")

predictions_df = test_data[[
    "movie",
    "week_start",
    "target_next",
    "pred_baseline",
    "pred_lr",
    "pred_rf"
]].copy()

predictions_df["residual_baseline"] = predictions_df["target_next"] - predictions_df["pred_baseline"]
predictions_df["residual_lr"] = predictions_df["target_next"] - predictions_df["pred_lr"]
predictions_df["residual_rf"] = predictions_df["target_next"] - predictions_df["pred_rf"]

metrics_df = pd.DataFrame({
    "model": ["Baseline", "Linear Regression", "Random Forest"],
    "train_r2": [np.nan, lr_r2_train, rf_r2_train],
    "test_r2": [baseline_r2, lr_r2_test, rf_r2_test],
    "rmse": [baseline_rmse, lr_rmse, rf_rmse],
    "mae": [baseline_mae, lr_mae, rf_mae],
    "target_variable": [target_base_col, target_base_col, target_base_col],
    "analysis_goal": [analysis_goal, analysis_goal, analysis_goal],
})

panel = round_numeric_columns(panel)
model_data = round_numeric_columns(model_data)
predictions_df = round_numeric_columns(predictions_df)
lr_coef_df = round_numeric_columns(lr_coef_df)
rf_importance_df = round_numeric_columns(rf_importance_df)
metrics_df = round_numeric_columns(metrics_df)

panel.to_csv(OUTPUT_DIR / "ml_ready_features.csv", index=False)
model_data.to_csv(OUTPUT_DIR / "model_data_with_target.csv", index=False)
predictions_df.to_csv(OUTPUT_DIR / "predictions_test_data.csv", index=False)
lr_coef_df.to_csv(OUTPUT_DIR / "linear_model_coefficients.csv", index=False)
rf_importance_df.to_csv(OUTPUT_DIR / "random_forest_importance.csv", index=False)
metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)


# ============================================================
# Making predictive plots
# ============================================================
print_section("CREATING PLOTS")

if not test_data.empty:
    min_val, max_val = safe_min_max(y_test, test_data["pred_lr"])
    if min_val is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_data["pred_lr"])
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.title(f"Actual vs Predicted - Linear Regression\n{analysis_goal}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "actual_vs_predicted_lr.png")
        plt.close()

if not test_data.empty:
    min_val, max_val = safe_min_max(y_test, test_data["pred_rf"])
    if min_val is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_data["pred_rf"])
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.title(f"Actual vs Predicted - Random Forest\n{analysis_goal}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "actual_vs_predicted_rf.png")
        plt.close()

if not rf_importance_df.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(rf_importance_df["feature"], rf_importance_df["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "random_forest_feature_importance.png")
    plt.close()

if not lr_coef_df.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(lr_coef_df["feature"], lr_coef_df["coefficient"])
    plt.title("Linear Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "linear_regression_coefficients.png")
    plt.close()


# ============================================================
# Printing results
# ============================================================
print_section("RESULTS")

print("Analysis goal:")
print(analysis_goal)

print("\nTarget base column:")
print(target_base_col)

print("\n=== BASELINE RESULTS ===")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAE: {baseline_mae:.4f}")
print(
    f"Test R-squared: {baseline_r2:.4f}"
    if not np.isnan(baseline_r2)
    else "Test R-squared: not available"
)

print("\n=== LINEAR REGRESSION RESULTS ===")
print(lr_coef_df.to_string(index=False))
print(f"\nTrain R-squared: {lr_r2_train:.4f}")
print(
    f"Test R-squared: {lr_r2_test:.4f}"
    if not np.isnan(lr_r2_test)
    else "Test R-squared: not available"
)
print(f"RMSE: {lr_rmse:.4f}")
print(f"MAE: {lr_mae:.4f}")

print("\n=== RANDOM FOREST RESULTS ===")
print(rf_importance_df.to_string(index=False))
print(f"\nTrain R-squared: {rf_r2_train:.4f}")
print(
    f"Test R-squared: {rf_r2_test:.4f}"
    if not np.isnan(rf_r2_test)
    else "Test R-squared: not available"
)
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")

print("\n=== MODEL COMPARISON ===")
print(metrics_df.sort_values("rmse").to_string(index=False))


# ============================================================
# Final status message
# ============================================================
print_section("PREDICTIVE ANALYSIS COMPLETE")
print(f"Saved: {OUTPUT_DIR / 'ml_ready_features.csv'}")
print(f"Saved: {OUTPUT_DIR / 'model_data_with_target.csv'}")
print(f"Saved: {OUTPUT_DIR / 'predictions_test_data.csv'}")
print(f"Saved: {OUTPUT_DIR / 'linear_model_coefficients.csv'}")
print(f"Saved: {OUTPUT_DIR / 'random_forest_importance.csv'}")
print(f"Saved: {OUTPUT_DIR / 'model_metrics.csv'}")