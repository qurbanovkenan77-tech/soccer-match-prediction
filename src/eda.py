import pandas as pd
import matplotlib.pyplot as plt


def create_eda(features, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Explore only training seasons.
    training = features[
        features["Season"].isin(["2021-22", "2022-23", "2023-24"])
    ].copy()

    outcome_order = ["H", "D", "A"]
    outcome_names = ["Home win", "Draw", "Away win"]
    colors = ["steelblue", "gray", "darkorange"]

    # Graph 1: Outcome frequencies.
    counts = training["FTR"].value_counts().reindex(
        outcome_order, fill_value=0
    )
    percentages = counts / counts.sum() * 100

    print("\nTraining outcome distribution:")
    summary = pd.DataFrame({
        "matches": counts,
        "percentage": percentages.round(1),
    })
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(outcome_names, percentages, color=colors)

    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Percentage of eligible training matches")
    ax.set_title("How common is each match outcome?")
    ax.set_ylim(0, percentages.max() + 10)

    fig.tight_layout()
    fig.savefig(results_dir / "01_outcome_distribution.png", dpi=150)
    plt.close(fig)

    # Graph 2: Compare recent points between opponents.
    training["form_difference"] = (
        training["home_recent_points"]
        - training["away_recent_points"]
    )

    # Round to avoid tiny floating-point differences at boundaries.
    difference = training["form_difference"].round(6)

    training["form_group"] = "Similar form"
    training.loc[
        difference > 0.4, "form_group"
    ] = "Home stronger"
    training.loc[
        difference < -0.4, "form_group"
    ] = "Away stronger"

    group_order = ["Away stronger", "Similar form", "Home stronger"]

    table = pd.crosstab(
        training["form_group"], training["FTR"]
    ).reindex(
        index=group_order,
        columns=outcome_order,
        fill_value=0,
    )

    group_sizes = table.sum(axis=1)
    rates = table.div(group_sizes.replace(0, float("nan")), axis=0) * 100

    print("\nOutcomes by recent-form group (%):")
    report = rates.round(1).copy()
    report["matches"] = group_sizes
    print(report.to_string())

    ax = rates.plot.bar(
        figsize=(8, 5),
        color=colors,
        rot=0,
    )

    ax.set_title("Does stronger recent form relate to the result?")
    ax.set_xlabel("Recent points per match: home versus away")
    ax.set_ylabel("Percentage within each form group")
    ax.set_ylim(0, 100)
    ax.legend(outcome_names)

    ax.set_xticklabels([
        f"{group}\n(n={group_sizes[group]})"
        for group in group_order
    ])

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(results_dir / "02_recent_form_outcomes.png", dpi=150)
    plt.close(fig)

    print("\nSaved two graphs in the results folder.")