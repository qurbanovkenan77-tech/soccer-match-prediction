from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

season_files = {
    "2021-22": "epl_2021_22.csv",
    "2022-23": "epl_2022_23.csv",
    "2023-24": "epl_2023_24.csv",
    "2024-25": "epl_2024_25.csv",
    "2025-26": "epl_2025_26.csv",
}

columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

season_data = []

for season, filename in season_files.items():
    df = pd.read_csv(DATA_DIR / filename)

    # Keep only the information needed for our framework.
    df = df[columns].copy()
    df["Season"] = season

    season_data.append(df)

matches = pd.concat(season_data, ignore_index=True)

# Convert dates explicitly: these files use day/month/year.
matches["Date"] = pd.to_datetime(
    matches["Date"],
    format="%d/%m/%Y",
    errors="coerce",
)

# Remove accidental spaces from team names and result labels.
for column in ["HomeTeam", "AwayTeam", "FTR"]:
    matches[column] = matches[column].astype("string").str.strip()
    matches[column] = matches[column].replace("", pd.NA)

# Ensure goals are numeric.
for column in ["FTHG", "FTAG"]:
    matches[column] = pd.to_numeric(matches[column], errors="coerce")

print("Missing or invalid values:")
print(matches[columns].isna().sum())

# Stop rather than silently remove matches that affect team histories.
if matches[columns].isna().any().any():
    problem_rows = matches[matches[columns].isna().any(axis=1)]
    print("\nRows needing review:")
    print(problem_rows.to_string(index=False))
    raise ValueError("Some required values need review before continuing.")

# Goals must be nonnegative whole numbers.
goals = matches[["FTHG", "FTAG"]]
invalid_goals = ((goals < 0) | (goals % 1 != 0)).any(axis=1)

if invalid_goals.any():
    print(matches.loc[invalid_goals].to_string(index=False))
    raise ValueError("Invalid goal values found.")

matches[["FTHG", "FTAG"]] = goals.astype(int)

# Each home/away pairing should occur once within a league season.
match_key = ["Season", "HomeTeam", "AwayTeam"]
duplicates = matches.duplicated(subset=match_key, keep=False)

print("\nRows with duplicated season/team pairings:", duplicates.sum())

if duplicates.any():
    print(matches.loc[duplicates].to_string(index=False))
    raise ValueError("Duplicate fixtures need review.")

# Verify that the recorded result agrees with the score.
expected_result = pd.Series("D", index=matches.index)
expected_result.loc[matches["FTHG"] > matches["FTAG"]] = "H"
expected_result.loc[matches["FTHG"] < matches["FTAG"]] = "A"

inconsistent_results = matches["FTR"] != expected_result

print("Results inconsistent with score:", inconsistent_results.sum())

if inconsistent_results.any():
    print(matches.loc[inconsistent_results].to_string(index=False))
    raise ValueError("Some results do not match their scores.")

# Chronological order is essential for later pre-match features.
matches = matches.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)

print("\nSeason summary:")
summary = matches.groupby("Season").agg(
    matches=("FTR", "size"),
    first_date=("Date", "min"),
    last_date=("Date", "max"),
    home_teams=("HomeTeam", "nunique"),
    away_teams=("AwayTeam", "nunique"),
)
print(summary.to_string())

print("\nCombined dataset shape:")
print(matches.shape)

print("\nFirst five rows:")
print(matches.head().to_string(index=False))

output_path = DATA_DIR / "matches_clean.csv"
matches.to_csv(output_path, index=False)

print("\nSaved cleaned data to:", output_path)

from src.features import FEATURE_COLUMNS, build_features, get_team_history

features = build_features(matches)

if features.empty:
    raise ValueError("No fixtures have enough history. Check the data.")

print("\nFeature coverage by season:")
coverage = matches.groupby("Season").size().rename("total_matches").to_frame()
coverage["eligible_matches"] = (
    features.groupby("Season").size().reindex(coverage.index, fill_value=0)
)
coverage["excluded_matches"] = (
    coverage["total_matches"] - coverage["eligible_matches"]
)
print(coverage.to_string())

print("\nMissing feature values:")
print(features[FEATURE_COLUMNS].isna().sum())

print("\nFirst three matches with features:")
preview = features.head(3).copy()
preview[FEATURE_COLUMNS] = preview[FEATURE_COLUMNS].round(3)
print(preview.to_string(index=False))

features.to_csv(DATA_DIR / "matches_features.csv", index=False)
print("\nSaved: data/matches_features.csv")

# Inspect one example so we can verify how the features were calculated.
example = features.iloc[0]

print(
    f"\nManual check: {example['HomeTeam']} vs {example['AwayTeam']}"
    f" on {example['Date'].date()}"
)

home_history = get_team_history(
    matches,
    example["HomeTeam"],
    example["Season"],
    example["Date"],
)

print("\nHome team's previous five league matches:")
print(
    home_history[
        ["Date", "HomeTeam", "AwayTeam",
         "goals_scored", "goals_conceded", "points"]
    ].tail(5).to_string(index=False)
)

print("\nCalculated home recent points:", example["home_recent_points"])

from src.eda import create_eda

RESULTS_DIR = Path(__file__).resolve().parent / "results"
create_eda(features, RESULTS_DIR)

from src.modeling import compare_models

compare_models(features, RESULTS_DIR)

from src.final_evaluation import evaluate_final_models

final_models = evaluate_final_models(features, RESULTS_DIR)

from src.predict_match import demonstrate_prediction

demonstrate_prediction(
    final_models["Logistic_regression"],
    features,
    RESULTS_DIR,
)