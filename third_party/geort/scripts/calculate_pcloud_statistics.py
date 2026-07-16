"""Calculates statistics for hand point cloud data.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Custom
from utils.helpers import (
    initialize_logger,
)


DATASETS_TO_INSPECT = {
    "avp_left": [
        "dataset_left_avp_subject-RJM_run-200.npy",
        "dataset_left_avp_subject-RJM_run-201.npy",
        "dataset_left_avp_subject-RJM_run-202.npy",
        "dataset_left_avp_subject-RJM_run-203.npy",
        "dataset_left_avp_subject-RJM_run-204.npy",
        "dataset_left_avp_subject-RJM_run-205.npy",
        "dataset_left_avp_subject-RJM_run-206.npy",
        "dataset_left_avp_subject-RJM_run-207.npy",
        "dataset_left_avp_subject-RJM_run-208.npy",
    ],
    "avp_right": [
        "dataset_right_avp_subject-RJM_run-200.npy",
        "dataset_right_avp_subject-RJM_run-201.npy",
        "dataset_right_avp_subject-RJM_run-202.npy",
        "dataset_right_avp_subject-RJM_run-203.npy",
        "dataset_right_avp_subject-RJM_run-204.npy",
        "dataset_right_avp_subject-RJM_run-205.npy",
        "dataset_right_avp_subject-RJM_run-206.npy",
        "dataset_right_avp_subject-RJM_run-207.npy",
        "dataset_right_avp_subject-RJM_run-208.npy",
    ],
    "manus_left": [
        "dataset_left_manus_subject-RJM_run-200.npy",
        "dataset_left_manus_subject-RJM_run-201.npy",
        "dataset_left_manus_subject-RJM_run-202.npy",
        "dataset_left_manus_subject-RJM_run-203.npy",
        "dataset_left_manus_subject-RJM_run-204.npy",
        "dataset_left_manus_subject-RJM_run-205.npy",
        "dataset_left_manus_subject-RJM_run-206.npy",
        "dataset_left_manus_subject-RJM_run-207.npy",
        "dataset_left_manus_subject-RJM_run-208.npy",
    ],
    "manus_right": [
        "dataset_right_manus_subject-RJM_run-200.npy",
        "dataset_right_manus_subject-RJM_run-201.npy",
        "dataset_right_manus_subject-RJM_run-202.npy",
        "dataset_right_manus_subject-RJM_run-203.npy",
        "dataset_right_manus_subject-RJM_run-204.npy",
        "dataset_right_manus_subject-RJM_run-205.npy",
        "dataset_right_manus_subject-RJM_run-206.npy",
        "dataset_right_manus_subject-RJM_run-207.npy",
        "dataset_right_manus_subject-RJM_run-208.npy",
    ],
}


def calculate_pcloud_statistics(hand_pcloud_data: np.ndarray) -> dict[str, np.ndarray]:
    """Calculates statistics for the hand point cloud data.

    Args:
        hand_pcloud_data: np.ndarray of shape (N, 3) where N is number of points from mocap system.
    Returns:
        A dictionary containing mean and standard deviation of the point cloud data.
    """
    statistics = {
        "mean": np.mean(hand_pcloud_data, axis=0),
        "std_dev": np.std(hand_pcloud_data, axis=0),
    }
    return statistics


def process_and_save_benchmarks(data_groups, logger):
    """Processes groups and generates a detailed Run-ID comparison."""
    all_results = []
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    # NEW: Create output directory
    output_dir = Path("analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, filenames in data_groups.items():
        for fname in filenames:
            data_path = (data_dir / fname).resolve()
            if not data_path.exists():
                continue

            hand_pcloud_data = np.load(data_path, allow_pickle=True)
            stats = calculate_pcloud_statistics(hand_pcloud_data)

            tracker = "avp" if "avp" in fname.lower() else "manus"
            chirality = "left" if "left" in fname.lower() else "right"
            run_id = fname.split("run-")[-1].replace(".npy", "")

            max_std_mm = np.max(stats["std_dev"]) * 1000

            all_results.append(
                {
                    "Run ID": run_id,
                    "Tracker": tracker.upper(),
                    "Chirality": chirality.capitalize(),
                    "Max Std Dev (mm)": round(max_std_mm, 4),
                }
            )

    df = pd.DataFrame(all_results)

    # 1. Output the Table
    print("\n" + tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    # 2. Save to CSV in analysis_results/
    csv_path = output_dir / "run_id_precision_report.csv"
    df.to_csv(csv_path, index=False)

    # 3. Generate Plot
    generate_run_plot(df, logger, output_dir)


def generate_run_plot(df, logger, output_dir):
    """Creates a bar plot comparing each Run ID across trackers."""
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    df = df.sort_values(by="Run ID")

    # NEW: ci=None removes the black bars (error bars)
    # to keep it clean since we want to see individual labels
    plot = sns.barplot(
        data=df,
        x="Run ID",
        y="Max Std Dev (mm)",
        hue="Tracker",
        palette="muted",
        ci=None,  # Remove the black confidence interval bars
    )

    # NEW: Add values on top of bars
    for container in plot.containers:
        plot.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.title(
        "Precision Measurements: Uncertainty for Hand Trackers", fontsize=14, pad=20
    )
    plt.ylabel("Max Standard Deviation (mm)")
    plt.xlabel("Trial Run ID (different hand poses and movements)")

    # Legend handling
    plt.legend(title="Hardware Tracker", loc="upper right", frameon=True)

    plt.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="1mm Threshold")

    # Save to the new directory
    plot_path = output_dir / "precision_per_run_id.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Run-ID Comparison plot saved to {plot_path}")
    plt.show()


def main():
    logger = initialize_logger(__name__)
    process_and_save_benchmarks(DATASETS_TO_INSPECT, logger)


if __name__ == "__main__":
    main()
