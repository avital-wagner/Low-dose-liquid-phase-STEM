# Functions for radial plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

def radar_polygon_area(values):
    """
    Area of a radar polygon where values are radial coordinates in [0, 1].

    Returns normalized area:
      0 = center/no area
      1 = full outer polygon
    """

    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return float("nan")

    if np.any(~np.isfinite(values)):
        return float("nan")

    n = len(values)

    # Area of polygon in polar coordinates:
    # A = 1/2 * sin(2π/n) * sum(r_i * r_{i+1})
    raw_area = 0.5 * np.sin(2 * np.pi / n) * np.sum(
        values * np.roll(values, -1)
    )

    # Maximum area happens when all r_i = 1
    max_area = 0.5 * np.sin(2 * np.pi / n) * n

    return float(raw_area / max_area)

    
def make_radar_plot(csv_path, output_path=None, individual_dir=None, image_col="image", range_row_index=0):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Remove empty Excel/trailing-comma columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    if image_col not in df.columns:
        raise ValueError(f"Could not find image column: {image_col}")

    if output_path is None:
        output_path = csv_path.parent / "combined_radar.png"
    else:
        output_path = Path(output_path)

    if individual_dir is None:
        individual_dir = output_path.parent / "individual_radars"
    else:
        individual_dir = Path(individual_dir)

    individual_dir.mkdir(parents=True, exist_ok=True)

    range_row = df.iloc[range_row_index]

    metric_cols = []

    for c in df.columns:
        if c == image_col:
            continue

        spec = str(range_row[c]).strip()

        if spec.lower() in ["nan", "", "none"]:
            continue

        if "-" not in spec:
            continue

        metric_cols.append(c)

    axis_ranges = {}

    for metric in metric_cols:
        spec = str(range_row[metric]).strip()
        axis_start, axis_end = spec.split("-")
        axis_ranges[metric] = (float(axis_start), float(axis_end))

    sample_df = df.drop(index=df.index[range_row_index]).reset_index(drop=True)

    labels = metric_cols
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    plt.rcParams["font.family"] = "Arial"

    # -------------------------
    # Combined plot
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(9, 8),
        subplot_kw={"projection": "polar"},
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in sample_df.iterrows():
        values = []
        raw_values = []

        for metric in metric_cols:
            raw = pd.to_numeric(row[metric], errors="coerce")

            if not np.isfinite(raw):
                values.append(np.nan)
                continue

            axis_start, axis_end = axis_ranges[metric]
            denom = axis_end - axis_start

            if denom == 0:
                values.append(np.nan)
                continue

            radial_value = (raw - axis_start) / denom
            radial_value = float(np.clip(radial_value, 0, 1))

            values.append(radial_value)
            raw_values.append(raw)

        values = np.array(values, dtype=float)

        if np.all(~np.isfinite(values)):
            continue

        values = np.nan_to_num(values, nan=0.0)
        values_closed = np.r_[values, values[0]]

        area_score = radar_polygon_area(values)
        label = f"{Path(str(row[image_col])).stem}  A={area_score:.2f}"

        # Add to combined plot
        ax.plot(
            angles_closed,
            values_closed,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )

        ax.fill(
            angles_closed,
            values_closed,
            alpha=0.08,
        )

        # -------------------------
        # Individual plot
        # -------------------------
        fig_ind, ax_ind = plt.subplots(
            figsize=(9, 8),
            subplot_kw={"projection": "polar"},
        )

        ax_ind.set_theta_offset(np.pi / 2)
        ax_ind.set_theta_direction(-1)

        ax_ind.plot(
            angles_closed,
            values_closed,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax_ind.fill(
            angles_closed,
            values_closed,
            alpha=0.08,
        )

        ax_ind.set_ylim(0, 1)

        ax_ind.set_xticks(angles)
        ax_ind.set_xticklabels(labels, fontsize=16)

        ax_ind.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax_ind.set_yticklabels(
            ["25", "50", "75", "100"],
            fontsize=16,
        )

        ax_ind.grid(True, linewidth=1.1, alpha=0.45)00
        ax_ind.spines["polar"].set_alpha(0.35)

        # Add raw values next to the points
        for angle, r, raw in zip(angles, values, raw_values):
            ax_ind.text(
                angle,
                min(r - 0.10, 1.12),
                f"{raw:.3g}",
                ha="center",
                va="center",
                fontsize=14,
            )

        ax_ind.set_title(
            f"{Path(str(row[image_col])).stem}  A={area_score:.2f}",
            fontsize=14,
            pad=25,
        )

        plt.tight_layout()

        individual_path = individual_dir / (
            f"{Path(str(row[image_col])).stem}_radar.png"
        )

        plt.savefig(
            individual_path,
            dpi=600,
            bbox_inches="tight",
        )

        plt.close(fig_ind)

    # Finish combined plot
    ax.set_ylim(0, 1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=16)

    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=16)

    ax.grid(True, linewidth=1.1, alpha=0.45)
    ax.spines["polar"].set_alpha(0.35)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved combined radar plot to: {output_path}")
    print(f"Saved individual radar plots to: {individual_dir}")