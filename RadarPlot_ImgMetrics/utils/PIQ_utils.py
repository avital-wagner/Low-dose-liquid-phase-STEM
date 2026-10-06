# UTIL Functions for image metric calculation and radial plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# OWN PACKAGE
from utils.extract_metadata import extract_image_metadata

def load_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rgb_float: HxWx3 float32 in [0,1]
      alpha_mask: HxW bool mask of valid pixels (True=valid), derived from alpha if present.
                 If no alpha channel, returns all-True mask.
    """
    im = Image.open(path)
    im_np = np.array(im)

    if im_np.ndim == 2:
        rgb = np.stack([im_np, im_np, im_np], axis=-1)
        alpha_mask = np.ones(im_np.shape, dtype=bool)
    else:
        if im_np.shape[2] == 4:
            rgb = im_np[:, :, :3]
            alpha = im_np[:, :, 3]
            alpha_mask = alpha > 0
        else:
            rgb = im_np[:, :, :3]
            alpha_mask = np.ones(im_np.shape[:2], dtype=bool)

    rgb = rgb.astype(np.float32)

    if np.issubdtype(im_np.dtype, np.integer):
        rgb_float = rgb / float(np.iinfo(im_np.dtype).max)
    else:
        mx = float(rgb.max()) if rgb.size else 1.0
        rgb_float = rgb / mx if mx > 1.0 else rgb

    return rgb_float.astype(np.float32), alpha_mask


def compute_metrics_for_image(path, metrics, hyperparameters):
    metric_values = {}
    
    rgb_float, _ = load_image(path)

    # loop over metric objects and compute their corresponding values
    # hyperparameters are individually extracted
    for metric in metrics:
        print("Calculating", metric.name())

        dict_values = metric.compute(rgb_float, **hyperparameters)

        # add metric + value to dict
        metric_values.update(dict_values)

    return metric_values


def df_all_images(imgs, metrics, hyperparameters):
    # Per-image pixel size from filename metadata (optional; adds nm columns)
    pixel_size_by_name= {}
    try:
        metadata_list = extract_image_metadata(imgs)
        for m in metadata_list:
            px = float(m.get("pixel"))
            pth = m.get("path")
            name = Path(pth).name if pth is not None else None
            if name is not None:
                pixel_size_by_name[name] = px
    except Exception:
        # If metadata parsing fails, we just won't compute nm resolutions.
        pixel_size_by_name = {}
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    
    hyperparameters["device"] = device
    
    rows = []
    for img in imgs:
        try:
            print("Working on:", img.name)
    
            row = compute_metrics_for_image(img, metrics, hyperparameters)
    
            px_nm = pixel_size_by_name.get(img.name, None)
            row["pixel_size_nm"] = px_nm
    
            if px_nm is not None:
                if "res_ampl_px" in row and row["res_ampl_px"] is not None:
                    row["res_ampl_nm"] = row["res_ampl_px"] * px_nm
                if "res_pwr_px" in row and row["res_pwr_px"] is not None:
                    row["res_pwr_nm"] = row["res_pwr_px"] * px_nm
    
            row["image"] = img
            rows.append(row)
    
            print("Done with:", img.name)
        
        except Exception as e:
            print("Issue with", img.name, "-", e)
            rows.append({"image": img.name, "error": str(e)})
    
    df = pd.DataFrame(rows)

    return df


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

    
def make_radar_plot(csv_path, output_path=None, image_col="image", range_row_index=0):
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

    fig, ax = plt.subplots(
        figsize=(9, 8),
        subplot_kw={"projection": "polar"},
    )
    
    plt.rcParams["font.family"] = "Arial"

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in sample_df.iterrows():
        values = []

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

        values = np.array(values, dtype=float)

        if np.all(~np.isfinite(values)):
            continue

        values = np.nan_to_num(values, nan=0.0)
        values_closed = np.r_[values, values[0]]

        area_score = radar_polygon_area(values)
        label = f"{Path(str(row[image_col])).stem}  A={area_score:.2f}"

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

    ax.set_ylim(0, 1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=16)

    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=16)

    ax.grid(True, linewidth=1.1, alpha=0.45)
    ax.spines["polar"].set_alpha(0.35)

    ax.set_title(
        "",
        fontsize=14,
        pad=25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved combined radar plot to: {output_path}")

    return output_path