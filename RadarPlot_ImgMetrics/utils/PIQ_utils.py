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


def compute_metrics_for_image(path, metrics):
    metric_values = {}
    
    rgb_float, _ = load_image(path)

    # loop over metric objects and compute their corresponding values
    # hyperparameters are individually extracted
    for metric in metrics:
        print("Calculating", metric.name())

        dict_values = metric.compute(rgb_float) 

        # add metric + value to dict
        metric_values.update(dict_values)

    return metric_values


def df_all_images(imgs, metrics):
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
    
    rows = []
    for img in imgs:
        try:
            print("Working on:", img.name)
    
            row = compute_metrics_for_image(img, metrics)
    
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
