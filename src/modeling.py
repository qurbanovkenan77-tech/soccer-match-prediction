import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS


def compare_models(features, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    train = features[
        features["Season"].isin(["2021-22", "2022-23", "2023-24"])
    ].copy()

    validation = features[
        features["Season"] == "2024-25"
    ].copy()

    y_train = train["FTR"]
    y_validation = validation["FTR"]

    form_columns = [
        "home_recent_points",
        "away_recent_points",
    ]

    form_goals_columns = form_columns + [
        "home_recent_goals_scored",
        "away_recent_goals_scored",
        "home_recent_goals_conceded",
        "away_recent_goals_conceded",
    ]

    def logistic_model():
        # Scaling is learned only from the training data.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )

    experiments = {
        "Baseline": (
            DummyClassifier(strategy="prior"),
            FEATURE_COLUMNS,
        ),
        "LR_form": (
            logistic_model(),
            form_columns,
        ),
        "LR_form_goals": (
            logistic_model(),
            form_goals_columns,
        ),
        "LR_all": (
            logistic_model(),
            FEATURE_COLUMNS,
        ),
        "Random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
            FEATURE_COLUMNS,
        ),
    }

    print("\nTraining matches:", len(train))
    print("Validation matches:", len(validation))

    scores = []
    reports = []

    for name, (model, columns) in experiments.items():
        model.fit(train[columns], y_train)
        if name == "LR_all":
            logistic = model.named_steps["logisticregression"]

            coefficients = pd.DataFrame(
                logistic.coef_.T,
                index=columns,
                columns=logistic.classes_,
            )

            # A positive value favors H over A as the feature increases,
            # holding the other features constant.
            coefficients["H_minus_A"] = (
                coefficients["H"] - coefficients["A"]
            )

            print("\nStandardized Logistic Regression coefficients:")
            print(coefficients.round(3).to_string())

            coefficients.to_csv(
                results_dir / "lr_coefficients.csv"
            )

        if name == "Random_forest":
            importance = pd.Series(
                model.feature_importances_,
                index=columns,
                name="importance",
            ).sort_values(ascending=False)

            print("\nRandom Forest training feature importance:")
            print(importance.round(3).to_string())

            importance.to_csv(
                results_dir / "rf_feature_importance.csv"
            )      
        predictions = model.predict(validation[columns])
        probabilities = model.predict_proba(validation[columns])

        scores.append({
            "model": name,
            "accuracy": accuracy_score(y_validation, predictions),
            "macro_f1": f1_score(
                y_validation,
                predictions,
                labels=["H", "D", "A"],
                average="macro",
                zero_division=0,
            ),
            "log_loss": log_loss(
                y_validation,
                probabilities,
                labels=model.classes_,
            ),
        })

        report = classification_report(
            y_validation,
            predictions,
            labels=["H", "D", "A"],
            target_names=["Home win", "Draw", "Away win"],
            digits=3,
            zero_division=0,
        )

        matrix = pd.DataFrame(
            confusion_matrix(
                y_validation,
                predictions,
                labels=["H", "D", "A"],
            ),
            index=["Actual H", "Actual D", "Actual A"],
            columns=["Predicted H", "Predicted D", "Predicted A"],
        )

        reports.append(
            f"\n{name}\n\n{report}\n"
            f"Confusion matrix:\n{matrix.to_string()}\n"
        )

        # Save individual predictions for later inspection.
        prediction_table = validation[
            ["Date", "HomeTeam", "AwayTeam", "FTR"]
        ].copy()
        prediction_table["prediction"] = predictions

        for index, outcome in enumerate(model.classes_):
            prediction_table[f"prob_{outcome}"] = probabilities[:, index]

        prediction_table.to_csv(
            results_dir / f"validation_predictions_{name}.csv",
            index=False,
        )

    comparison = pd.DataFrame(scores).set_index("model")

    print("\nValidation comparison:")
    print(comparison.round(4).to_string())

    comparison.to_csv(results_dir / "validation_comparison.csv")

    report_text = "\n".join(reports)
    (results_dir / "validation_reports.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nDetailed validation reports:")
    print(report_text)

    print("\nValidation results saved. Final test season not evaluated.")