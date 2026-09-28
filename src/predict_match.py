import pandas as pd

from src.features import FEATURE_COLUMNS


def demonstrate_prediction(model, features, results_dir):
    # Select a match without looking at its outcome.
    test_matches = features[
        features["Season"] == "2025-26"
    ].sort_values(["Date", "HomeTeam", "AwayTeam"])

    match = test_matches.iloc[[0]]
    details = match.iloc[0]

    probabilities = model.predict_proba(
        match[FEATURE_COLUMNS]
    )[0]

    probability_by_outcome = dict(
        zip(model.classes_, probabilities)
    )

    predicted_outcome = model.predict(
        match[FEATURE_COLUMNS]
    )[0]

    labels = {
        "H": f"{details['HomeTeam']} win",
        "D": "Draw",
        "A": f"{details['AwayTeam']} win",
    }

    lines = [
        "HISTORICAL PRE-MATCH PREDICTION",
        f"Match: {details['HomeTeam']} vs {details['AwayTeam']}",
        f"Date: {details['Date'].date()}",
        "Model: Logistic Regression",
        "Model training seasons: 2021-22 through 2024-25",
        "",
    ]

    for outcome in ["H", "D", "A"]:
        lines.append(
            f"{labels[outcome]}: "
            f"{probability_by_outcome[outcome]:.1%}"
        )

    lines.extend([
        "",
        f"Predicted outcome: {labels[predicted_outcome]}",
        f"Actual outcome: {labels[details['FTR']]}",
        "",
        "These are model estimates, not guaranteed outcomes.",
        "The actual outcome was not used as a prediction input.",
    ])

    report = "\n".join(lines)
    print("\n" + report)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "example_prediction.txt").write_text(
        report,
        encoding="utf-8",
    )

    probability_table = pd.DataFrame({
        "outcome": ["H", "D", "A"],
        "label": [labels[outcome] for outcome in ["H", "D", "A"]],
        "probability": [
            probability_by_outcome[outcome]
            for outcome in ["H", "D", "A"]
        ],
    })

    probability_table.to_csv(
        results_dir / "example_probabilities.csv",
        index=False,
    )