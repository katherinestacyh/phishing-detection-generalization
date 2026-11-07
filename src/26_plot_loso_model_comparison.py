from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: Set project paths
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "23_loso_model_metrics_v2.csv"
OUTPUT = ROOT / "results" / "figure_loso_f1_comparison.png"


def main():

    # STEP 2: Load our corrected evaluation results
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing results file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {"held_out_source", "model", "f1"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    # STEP 3: Organize model results by email source
    sources = [
        "CEAS_08",
        "Enron",
        "Ling",
        "SpamAssasin"
    ]

    models = [
        "Logistic Regression",
        "Linear SVM"
    ]

    comparison = df.pivot(
        index="held_out_source",
        columns="model",
        values="f1"
    )

    comparison = comparison.loc[sources, models]

    if comparison.isna().any().any():
        raise ValueError(
            "Missing F1-scores for one or more experiments."
        )

    print("Corrected LOSO F1-scores:")
    print(comparison.round(4))

    # STEP 4: Create a grouped bar chart
    x = np.arange(len(sources))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_lr = ax.bar(
        x - width / 2,
        comparison["Logistic Regression"],
        width,
        label="Logistic Regression"
    )

    bars_svm = ax.bar(
        x + width / 2,
        comparison["Linear SVM"],
        width,
        label="Linear SVM"
    )

    # STEP 5: Format our figure
    ax.set_title(
        "Leave-One-Source-Out Model Performance",
        fontsize=14
    )

    ax.set_ylabel("F1-score")
    ax.set_xlabel("Held-out Email Source")

    ax.set_xticks(x)
    ax.set_xticklabels(sources)

    ax.set_ylim(0, 1.05)

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    # Display values above each bar
    for bars in [bars_lr, bars_svm]:

        for bar in bars:

            height = bar.get_height()

            ax.annotate(
                f"{height:.3f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # STEP 6: Save graph
    fig.tight_layout()

    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nFigure saved successfully:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
