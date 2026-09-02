"""Render round timelines as a human-readable strip chart.

This intentionally avoids numeric summaries and instead draws each round as a
colored horizontal band. Busy rounds read as darker, longer blocks; quiet rounds
read as pale, short blocks.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.forecasting.sessionize import sessionize


def build_round_eye_view(csv_path, output_path, gap_minutes=30):
    df = pd.read_csv(csv_path)
    sessionized = sessionize(df, gap_minutes=gap_minutes)

    summary = (
        sessionized.groupby(["user_id", "round_id"], as_index=False)
        .agg(
            round_start=("recorded_at", "min"),
            round_end=("recorded_at", "max"),
            event_count=("recorded_at", "count"),
        )
        .sort_values(["user_id", "round_start"])
        .reset_index(drop=True)
    )
    summary["round_start"] = pd.to_datetime(summary["round_start"])
    summary["round_end"] = pd.to_datetime(summary["round_end"])

    user_order = summary["user_id"].drop_duplicates().tolist()
    y_map = {user: idx for idx, user in enumerate(user_order)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(user_order) * 0.8)))
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")

    norm = plt.Normalize(summary["event_count"].min(), summary["event_count"].max())
    cmap = plt.get_cmap("Blues")

    for _, row in summary.iterrows():
        start = mdates.date2num(row["round_start"])
        width = mdates.date2num(row["round_end"]) - start
        y = y_map[row["user_id"]]
        ax.barh(
            y,
            width=width,
            left=start,
            height=0.7,
            color=cmap(norm(row["event_count"])),
            edgecolor="none",
            alpha=0.9,
        )

    ax.set_yticks(range(len(user_order)))
    ax.set_yticklabels(user_order)
    ax.set_xlabel("time")
    ax.set_ylabel("user")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.grid(axis="x", color="#dfe7f3", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    title = "Round strip view"
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(
        0.98,
        0.02,
        "Darker = busier round",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#54657a",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "examples" / "sample_recordings.csv"
    output_path = repo_root / "examples" / "round_eye_view.png"
    build_round_eye_view(csv_path, output_path, gap_minutes=30)
    print(f"Saved round-eye view to: {output_path}")
