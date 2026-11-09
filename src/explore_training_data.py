"""PIP-04: describe frozen training sessions and save figures; fit no models.

Run from the repository root: python -m src.explore_training_data
The split loader validates the complete bundle for integrity, after which all
statistics and plots receive only training rows. No split is created here.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.feature_sets import (
    ALL_17,
    CATEGORICAL,
    DURATIONS,
    FEATURE_SETS,
    NUMERIC,
    PAGE_COUNTS,
)
from src.split_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DIR,
    ROOT,
    load_frozen_splits,
)

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
BLUE, ORANGE = "#246a8d", "#c06428"


def load_training_data() -> tuple[pd.DataFrame, dict[str, object]]:
    data, assignments, metadata = load_frozen_splits()
    rows = assignments.loc[assignments["partition"].eq("train"), "row_index"]
    train = data.iloc[rows.sort_values().to_numpy()].copy()
    if list(train.columns) != [*ALL_17, "Revenue"]:
        raise ValueError("Source columns differ from the feature-set contract.")
    provenance: dict[str, object] = {
        "source_sha256": metadata["source"]["sha256"],
        "train_manifest_sha256": hashlib.sha256(
            (DEFAULT_SPLIT_DIR / "train.csv").read_bytes()
        ).hexdigest(),
        "rows": len(train),
        "purchases": int(train["Revenue"].sum()),
        "scope": "Frozen training rows only; held-out rows used only by split integrity checks.",
    }
    return train, provenance


def group_summary(train: pd.DataFrame, column: str) -> pd.DataFrame:
    # Named aggregations produce a table, despite pandas' broader return annotation.
    result = cast(
        pd.DataFrame,
        train.groupby(column, observed=True)["Revenue"].agg(
            sessions="size", purchases="sum", purchase_rate="mean"
        ),
    )
    result["non_purchases"] = result["sessions"] - result["purchases"]
    return result


def build_tables(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric = cast(pd.DataFrame, train[NUMERIC])
    summary = numeric.describe(percentiles=[0.5, 0.95, 0.99]).T
    summary["zeros"] = numeric.eq(0).sum()
    summary["zero_fraction"] = numeric.eq(0).mean()
    summary["skew"] = numeric.skew()
    categories = pd.concat(
        [
            group_summary(train, col)
            .reset_index()
            .rename(columns={col: "value"})
            .assign(feature=col)
            for col in [*CATEGORICAL, "SpecialDay"]
        ],
        ignore_index=True,
    )
    categories["fewer_than_20"] = categories["sessions"].lt(20)
    flags = []
    extremes = []
    for count, duration in zip(PAGE_COUNTS, DURATIONS):
        # The audited schema has unique column names, so each name selects a Series.
        counts = cast(pd.Series, train[count])
        durations = cast(pd.Series, train[duration])
        flags.append(
            {
                "page_type": count,
                "positive_count_zero_duration": int(
                    (counts.gt(0) & durations.eq(0)).sum()
                ),
                "zero_count_positive_duration": int(
                    (counts.eq(0) & durations.gt(0)).sum()
                ),
            }
        )
        top = train.nlargest(3, duration)[[count, duration]].reset_index()
        top.columns = ["source_row_index", "page_count", "duration"]
        extremes.append(top.assign(feature=duration))
    boundaries = []
    for col in ["BounceRates", "ExitRates"]:
        values = cast(pd.Series, train[col])
        boundaries.append(
            {
                "feature": col,
                "zeros": int(values.eq(0).sum()),
                "at_0.2": int(values.eq(0.2).sum()),
                "min": values.min(),
                "max": values.max(),
            }
        )
    correlation_inputs = cast(pd.DataFrame, train[[*NUMERIC, "Revenue"]])
    return {
        "class_balance": group_summary(train, "Revenue"),
        "numeric_summary": summary,
        "month": group_summary(train, "Month").reindex(MONTHS),
        "visitor_type": group_summary(train, "VisitorType"),
        "categories": categories,
        "count_duration_flags": pd.DataFrame(flags),
        "extreme_durations": pd.concat(extremes, ignore_index=True),
        "rate_boundaries": pd.DataFrame(boundaries),
        "spearman": correlation_inputs.corr(method="spearman"),
        "pagevalues_zero": group_summary(
            train.assign(
                PageValues_group=np.where(train["PageValues"].eq(0), "Zero", "Positive")
            ),
            "PageValues_group",
        ).reindex(["Zero", "Positive"]),
    }


def save_figures(
    train: pd.DataFrame, tables: dict[str, pd.DataFrame], directory: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    n = len(train)

    def save(fig, name):
        fig.savefig(directory / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def rate_bars(ax, table, title, labels=None):
        positions = np.arange(len(table))
        ax.bar(positions, table["purchase_rate"], color=BLUE, width=0.65)
        ax.axhline(
            train["Revenue"].mean(),
            color=ORANGE,
            ls="--",
            lw=1.5,
            label=f"All training sessions: {train['Revenue'].mean():.1%}",
        )
        ax.set_xticks(positions, labels if labels is not None else table.index)
        for x, row in zip(positions, table.itertuples()):
            if pd.notna(row.sessions):
                ax.text(
                    x,
                    row.purchase_rate + 0.008,
                    f"{row.purchase_rate:.1%}\nn={int(row.sessions):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                )
            else:
                ax.text(x, 0.007, "Absent", ha="center", rotation=90, fontsize=8)
        ax.set_ylim(0, max(0.35, table["purchase_rate"].max() + 0.10))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Sessions ending in purchase")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4.8),
        gridspec_kw={"width_ratios": [1, 2.8, 1.4]},
        layout="constrained",
    )
    counts = cast(pd.Series, tables["class_balance"]["sessions"]).reindex([False, True])
    axes[0].bar(["No purchase", "Purchase"], counts, color=[BLUE, ORANGE])
    for x, value in enumerate(counts):
        axes[0].text(x, value + n * 0.015, f"{value:,}\n{value / n:.1%}", ha="center")
    axes[0].set_ylim(0, n)
    axes[0].set_ylabel("Sessions")
    axes[0].set_title("Most sessions do not buy")
    rate_bars(axes[1], tables["month"], "Month: rates and sample sizes")
    rate_bars(
        axes[2], tables["visitor_type"], "Visitor type", ["New", "Other", "Returning"]
    )
    fig.suptitle(f"Who buys in the frozen training data?  |  n={n:,}", fontsize=15)
    save(fig, "01_purchase_context.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, col in zip(axes.flat, [*PAGE_COUNTS, *DURATIONS]):
        values = cast(pd.Series, train[col])
        transformed = np.log1p(values)
        ax.hist(transformed, bins=35, color=BLUE, edgecolor="white", linewidth=0.4)
        median, p99 = values.quantile([0.5, 0.99])
        ax.set_title(
            f"{col}\nzero={values.eq(0).mean():.1%}; median={median:,.1f}; p99={p99:,.1f}",
            fontsize=10,
        )
        ticks = (
            [0, 1, 3, 10, 30, 100, 300, 1000]
            if col in PAGE_COUNTS
            else [0, 10, 100, 1000, 10000]
        )
        ticks = [value for value in ticks if value <= values.max()]
        ax.set_xticks(np.log1p(ticks), [f"{value:,}" for value in ticks])
        ax.set_xlabel(
            "Pages visited (compressed scale)"
            if col in PAGE_COUNTS
            else "Total time (source units; compressed scale)"
        )
        ax.set_ylabel("Sessions")
    fig.suptitle(
        f"Page activity has many zeros and a long upper tail  |  n={n:,}", fontsize=15
    )
    save(fig, "02_page_activity.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    zero = cast(pd.Series, tables["numeric_summary"]["zero_fraction"]).sort_values()
    axes[0, 0].barh(zero.index, zero, color=BLUE)
    for y, value in enumerate(zero):
        axes[0, 0].text(value + 0.01, y, f"{value:.1%}", va="center", fontsize=8)
    axes[0, 0].set_xlim(0, 1.08)
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].set_title("Zero is a common recorded value")
    axes[0, 0].set_xlabel("Share of training sessions")
    for ax, col in zip([axes[0, 1], axes[1, 0]], ["BounceRates", "ExitRates"]):
        ax.hist(train[col], bins=np.linspace(0, 0.2, 41), color=BLUE, edgecolor="white")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Recorded fraction (0.02 = 2%)")
        ax.set_ylabel("Sessions")
        ax.set_title(f"{col}: {train[col].eq(0.2).sum():,} sessions exactly at 20%")
    special = train["SpecialDay"].value_counts().sort_index()
    axes[1, 1].bar(special.index, special, width=0.14, color=BLUE)
    for x, count in special.items():
        axes[1, 1].text(x, count + 90, f"{count:,}", ha="center", fontsize=8)
    axes[1, 1].set_ylim(0, n)
    axes[1, 1].set_xticks(special.index)
    axes[1, 1].set_xlabel("Recorded closeness score")
    axes[1, 1].set_ylabel("Sessions")
    axes[1, 1].set_title("SpecialDay: usually zero")
    fig.suptitle(
        f"Zeros and bounded scales need careful interpretation  |  n={n:,}", fontsize=15
    )
    save(fig, "03_zeros_and_rates.png")

    corr = tables["spearman"]
    labels = [
        "Admin pages",
        "Admin time",
        "Info pages",
        "Info time",
        "Product pages",
        "Product time",
        "Bounce rate",
        "Exit rate",
        "PageValues",
        "SpecialDay",
        "Purchase",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 9), layout="constrained")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), labels, rotation=50, ha="right")
    ax.set_yticks(range(len(corr)), labels)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.65 else "#222222",
            )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title(
        f"Which measurements move together?  |  n={n:,}\nNumeric inputs plus the purchase label; category codes excluded",
        pad=15,
    )
    save(fig, "04_relationships.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rate_bars(axes[0], tables["pagevalues_zero"], "PageValues: zero versus positive")
    positive = train.loc[train["PageValues"].gt(0)]
    bins = np.linspace(0, np.log1p(positive["PageValues"].max()), 30)
    for label, name, color in [
        (False, "No purchase", BLUE),
        (True, "Purchase", ORANGE),
    ]:
        values = positive.loc[positive["Revenue"].eq(label), "PageValues"]
        axes[1].hist(
            np.log1p(values),
            bins=bins,
            weights=np.full(len(values), 1 / len(values)),
            histtype="step",
            lw=2,
            color=color,
            label=f"{name} (n={len(values):,})",
        )
    ticks = [0, 1, 5, 20, 100]
    axes[1].set_xticks(np.log1p(ticks), [str(value) for value in ticks])
    axes[1].set_xlabel("Positive PageValues (compressed scale; currency unknown)")
    axes[1].set_ylabel("Share within each positive-value class")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Positive values still overlap between classes")
    axes[1].legend(frameon=False)
    fig.suptitle(
        f"PageValues separates groups; this does not establish when it was known  |  n={n:,}",
        fontsize=13,
    )
    save(fig, "05_pagevalues.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/eda")
    args = parser.parse_args()
    source_before = DEFAULT_DATA_PATH.read_bytes()
    train, provenance = load_training_data()
    tables = build_tables(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(
            args.output_dir / f"{name}.csv",
            index=name
            not in {
                "categories",
                "count_duration_flags",
                "extreme_durations",
                "rate_boundaries",
            },
            float_format="%.8g",
        )
    save_figures(train, tables, args.output_dir)
    provenance["versions"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "feature_sets.json").write_text(
        json.dumps(FEATURE_SETS, indent=2) + "\n", encoding="utf-8"
    )
    if DEFAULT_DATA_PATH.read_bytes() != source_before:
        raise RuntimeError("Source CSV bytes changed during exploration.")
    print(
        f"EDA complete: {len(train):,} training sessions, {train['Revenue'].sum():,} purchases."
    )
    print(
        f"Saved 5 figures, {len(tables)} summary tables, provenance and feature lists to {args.output_dir}."
    )
    print(
        "Source unchanged; no preprocessing fitted, models trained or held-out performance evaluated."
    )


if __name__ == "__main__":
    main()
