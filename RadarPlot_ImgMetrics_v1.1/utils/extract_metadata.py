from pathlib import Path

def extract_image_metadata(image_files):
    """Extract metadata (id, detector, current, pixel, dwell, alpha, sparse)
    from image filenames.
    Assumes filenames are formatted as: id_detector_current_pixel_dwell_alpha_sparse.tif
    Args:
        image_files: List of Path objects to image files
    Returns:
        List of dictionaries containing extracted metadata
    """
    metadata = []
    for img_path in image_files:
        img_path_obj = img_path
        parts = img_path_obj.stem.split('_')
        if len(parts) >= 7:
            img_id = parts[0]
            number = parts[1]
            detector = parts[2]
            current = parts[3]
            pixel = parts[4]
            dwell = parts[5]
            alpha = parts[6]
            sparse = parts[7]
            metadata.append({
                'path': img_path_obj,
                'id': img_id,
                'number': number,
                'detector': detector,
                'current': current,
                'pixel': pixel,
                'dwell': dwell,
                'convergence angle': alpha,
                'sparsity': sparse
            })
        else:
            metadata.append({
                'path': img_path_obj,
                'id': 'unknown',
                'number': 'unknown',
                'detector': 'unknown',
                'current': 'unknown',
                'pixel': 'unknown',
                'dwell': 'unknown',
                'convergence angle': 'unknown',
                'sparsity': 'unknown'
            })
    return metadata