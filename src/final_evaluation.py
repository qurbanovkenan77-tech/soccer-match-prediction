import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS


def evaluate_final_models(features, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Refit using all four seasons available before the test season.
    development = features[
        features["Season"].isin([
            "2021-22", "2022-23", "2023-24", "2024-25"
        ])
    ].copy()

    test = features[
        features["Season"] == "2025-26"
    ].copy()

    X_development = development[FEATURE_COLUMNS]
    y_development = development["FTR"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["FTR"]

    models = {
        "Baseline": DummyClassifier(strategy="prior"),
        "Logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    print("\nFINAL TEST EVALUATION")
    print("Development matches:", len(development))
    print("Test matches:", len(test))
    print("Primary model: Logistic_regression")

    scores = []
    reports = []

    for name, model in models.items():
        model.fit(X_development, y_development)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        scores.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=["H", "D", "A"],
                average="macro",
                zero_division=0,
            ),
            "log_loss": log_loss(
                y_test,
                probabilities,
                labels=model.classes_,
            ),
        })

        report = classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            target_names=["Home win", "Draw", "Away win"],
            digits=3,
            zero_division=0,
        )
        reports.append(f"\n{name}\n\n{report}")

        # Save the final confusion matrix for each model.
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            display_labels=["Home win", "Draw", "Away win"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"{name}: 2025–26 test season")
        fig.tight_layout()
        fig.savefig(
            results_dir / f"test_confusion_{name}.png",
            dpi=150,
        )
        plt.close(fig)

        prediction_table = test[
            ["Date", "HomeTeam", "AwayTeam", "FTR"]
        ].copy()
        prediction_table["prediction"] = predictions

        for index, outcome in enumerate(model.classes_):
            prediction_table[f"prob_{outcome}"] = probabilities[:, index]

        prediction_table.to_csv(
            results_dir / f"test_predictions_{name}.csv",
            index=False,
        )

    comparison = pd.DataFrame(scores).set_index("model")

    print("\nFinal test comparison:")
    print(comparison.round(4).to_string())

    report_text = "\n".join(reports)
    print(report_text)

    comparison.to_csv(results_dir / "test_comparison.csv")
    (results_dir / "test_reports.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    return models