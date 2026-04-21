import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clean_mask(mask, kernel_size=7):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _otsu(gray_enhanced):
    blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union between two binary masks.
    Returns None if gt_mask is None.
    """
    if gt_mask is None:
        return None
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred,  gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ─────────────────────────────────────────────
# Segmentation Strategies
# ─────────────────────────────────────────────

def segment_grayscale_otsu(img_bgr):
    """
    Strategy 1: Grayscale + CLAHE + Otsu
    Best for: broken_large, crack, hole, scratch
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enh  = _clahe(gray)
    mask = _otsu(enh)
    # Flip if background is mistakenly thresholded (defect is typically darker)
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)
    mask = _clean_mask(mask, kernel_size=7)
    return mask


def segment_diff_otsu(img_bgr, ref_good_bgr=None):
    """
    Strategy 2: Difference Image + Otsu
    Best for: broken_small, missing parts, subtle scratches
    Falls back to grayscale Otsu if no reference provided.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if ref_good_bgr is not None:
        ref  = cv2.cvtColor(ref_good_bgr, cv2.COLOR_BGR2GRAY)
        ref  = cv2.resize(ref, (gray.shape[1], gray.shape[0]))
        diff = cv2.absdiff(gray, ref)
        enh  = _clahe(diff)
    else:
        enh  = _clahe(gray)
    mask = _otsu(enh)
    mask = _clean_mask(mask, kernel_size=5)
    return mask


def segment_hsv_saturation(img_bgr):
    """
    Strategy 3: HSV Saturation Channel + Otsu
    Best for: contamination, color spots, stains, glue
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s    = hsv[:, :, 1]
    enh  = _clahe(s)
    mask = _otsu(enh)
    mask = _clean_mask(mask, kernel_size=5)
    return mask


def segment_kmeans(img_bgr, k=3):
    """
    Strategy 4: K-Means Color Clustering
    Best for: texture anomalies (carpet, tile, wood, fabric)
    """
    pixels   = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    # Defect cluster = darkest mean brightness
    gray_centers = [
        cv2.cvtColor(np.full((1, 1, 3), c, dtype=np.uint8),
                     cv2.COLOR_BGR2GRAY)[0, 0]
        for c in centers
    ]
    defect_label = int(np.argmin(gray_centers))
    mask = (labels.flatten() == defect_label).reshape(
        img_bgr.shape[:2]
    ).astype(np.uint8) * 255
    mask = _clean_mask(mask, kernel_size=5)
    return mask


# ─────────────────────────────────────────────
# Strategy Auto-Selector
# ─────────────────────────────────────────────

_COLOR_DEFECTS  = {"contamination", "color", "stain", "glue", "poke", "liquid", "oil"}
_DIFF_DEFECTS   = {"broken_small", "scratch_head", "thread_side",
                   "missing_thread", "scratch_neck", "missing_wire",
                   "missing_cable", "manipulated_front"}
_KMEANS_DEFECTS = {"cut", "fold", "metal_contamination", "thread_top",
                   "fabric_border", "fabric_interior", "rough", "combined"}


def _auto_strategy(defect_type):
    if defect_type is None:
        return "grayscale"
    dt = defect_type.lower()
    for kw in _COLOR_DEFECTS:
        if kw in dt:
            return "hsv"
    for kw in _DIFF_DEFECTS:
        if kw in dt:
            return "diff"
    for kw in _KMEANS_DEFECTS:
        if kw in dt:
            return "kmeans"
    return "grayscale"


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def segment_image(img_bgr, ref_good_bgr=None, defect_type=None, strategy=None):
    """
    Segment defects in an industrial image.

    Parameters
    ----------
    img_bgr      : np.ndarray  BGR image (H×W×3)
    ref_good_bgr : np.ndarray | None   A reference 'good' image
    defect_type  : str | None   Defect folder name (e.g. 'broken_large', 'contamination')
    strategy     : str | None   Force a strategy: 'grayscale' | 'diff' | 'hsv' | 'kmeans'
                                If None, auto-selected from defect_type.

    Returns
    -------
    np.ndarray  Binary mask (H×W), 255 = defect region
    """
    if defect_type == "good":
        h, w = img_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    if strategy is None:
        strategy = _auto_strategy(defect_type)

    if strategy == "hsv":
        return segment_hsv_saturation(img_bgr)
    elif strategy == "diff":
        return segment_diff_otsu(img_bgr, ref_good_bgr)
    elif strategy == "kmeans":
        return segment_kmeans(img_bgr)
    else:
        return segment_grayscale_otsu(img_bgr)


def print_segmentation_metrics(metrics):
    """Print segmentation evaluation results."""
    print("\n=== Segmentation Metrics ===\n")
    for defect_type, values in metrics["per_type"].items():
        print(f"[{defect_type.upper()}]")
        print(f"  IoU  : {values['mean_iou']:.4f}  (n={values['n']})")
    print(f"\n[OVERALL]")
    print(f"  Mean IoU : {metrics['overall_iou']:.4f}")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_segmentation(img_bgr, pred_mask, gt_mask=None,
                           iou=None, strategy="", title="Segmentation"):
    """
    Display: original image | predicted mask | ground truth mask (if available)
    """
    n_cols = 3 if gt_mask is not None else 2
    plt.figure(figsize=(5 * n_cols, 5))
    plt.suptitle(
        f"{title}  |  Strategy: {strategy}" +
        (f"  |  IoU: {iou:.4f}" if iou is not None else ""),
        fontsize=13
    )

    plt.subplot(1, n_cols, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, n_cols, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="hot")
    plt.axis("off")

    if gt_mask is not None:
        plt.subplot(1, n_cols, 3)
        plt.title("Ground Truth")
        plt.imshow(gt_mask, cmap="hot")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_overlay(img_bgr, pred_mask, title="Defect Overlay"):
    """Overlay predicted mask on the original image in red."""
    overlay = img_bgr.copy()
    overlay[pred_mask > 0] = [0, 0, 220]
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_strategies(img_bgr, gt_mask=None, ref_good_bgr=None, defect_type=None):
    """
    Compare all 4 strategies side-by-side on a single image.
    """
    strategies  = ["grayscale", "diff", "hsv", "kmeans"]
    strat_names = ["Grayscale + Otsu", "Diff + Otsu", "HSV Saturation", "K-Means"]

    fig, axes = plt.subplots(2, len(strategies) + 1, figsize=(22, 8))

    # Column 0: original + GT
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    if gt_mask is not None:
        axes[1, 0].imshow(gt_mask, cmap="hot")
        axes[1, 0].set_title("Ground Truth")
    else:
        axes[1, 0].axis("off")
    axes[1, 0].axis("off")

    for i, (strat, sname) in enumerate(zip(strategies, strat_names)):
        mask = segment_image(img_bgr, ref_good_bgr=ref_good_bgr,
                             defect_type=defect_type, strategy=strat)
        iou  = compute_iou(mask, gt_mask)

        axes[0, i + 1].imshow(mask, cmap="hot")
        axes[0, i + 1].set_title(
            f"{sname}\nIoU: {iou:.4f}" if iou is not None else sname
        )
        axes[0, i + 1].axis("off")

        overlay = img_bgr.copy()
        overlay[mask > 0] = [0, 0, 200]
        axes[1, i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, i + 1].set_title("Overlay")
        axes[1, i + 1].axis("off")

    plt.suptitle(
        f"Strategy Comparison — {defect_type or 'unknown'}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Batch Evaluation (all defect types)
# ─────────────────────────────────────────────

def batch_segment(image_paths, save_samples=False, sample_limit=5,
                  output_dir=None, ref_good_bgr=None, category="bottle"):
    """
    Run segmentation on multiple images and aggregate IoU metrics.
    Mirrors the batch_preprocess() interface from preprocessing.py
    """
    total_iou = 0.0
    count     = 0

    for i, img_path in enumerate(tqdm(image_paths)):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Infer defect type from path
        defect_type = os.path.basename(os.path.dirname(img_path))

        # Load GT mask if available
        gt_mask   = None
        parts     = img_path.split(os.sep)
        try:
            cat_idx   = parts.index(category)
            gt_path   = os.path.join(
                *parts[:cat_idx + 1],
                "ground_truth", defect_type,
                os.path.basename(img_path).replace(".png", "_mask.png")
            )
            if os.path.isabs(img_path):
                gt_path = "/" + gt_path
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        except (ValueError, IndexError):
            pass

        mask = segment_image(img_bgr, ref_good_bgr=ref_good_bgr,
                             defect_type=defect_type)
        iou  = compute_iou(mask, gt_mask)

        if iou is not None:
            total_iou += iou
            count     += 1

        # Save sample outputs
        if save_samples and i < sample_limit and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_dir, f"sample_{i}_mask.png"), mask
            )
            overlay = img_bgr.copy()
            overlay[mask > 0] = [0, 0, 220]
            cv2.imwrite(
                os.path.join(output_dir, f"sample_{i}_overlay.png"), overlay
            )

    avg_iou = total_iou / count if count > 0 else 0.0
    return {"mean_iou": avg_iou, "n": count}


def evaluate_segmentation(data_path, category="bottle",
                          max_images=50, ref_good_bgr=None):
    """
    Evaluate segmentation on all defect types of a category.

    Returns dict with per-type IoU and overall mean IoU.
    """
    test_path = os.path.join(data_path, category, "test")
    gt_root   = os.path.join(data_path, category, "ground_truth")

    results_per_type = {}

    for defect_type in sorted(os.listdir(test_path)):
        defect_folder = os.path.join(test_path, defect_type)
        if not os.path.isdir(defect_folder):
            continue

        img_names = [
            f for f in os.listdir(defect_folder) if f.endswith(".png")
        ][:max_images]
        ious = []

        for img_name in tqdm(img_names,
                             desc=f"  [{category}] {defect_type}", leave=False):
            img_bgr = cv2.imread(os.path.join(defect_folder, img_name))
            if img_bgr is None:
                continue

            gt_mask = None
            if defect_type != "good":
                mask_path = os.path.join(
                    gt_root, defect_type,
                    img_name.replace(".png", "_mask.png")
                )
                if os.path.exists(mask_path):
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask = segment_image(img_bgr, ref_good_bgr=ref_good_bgr,
                                 defect_type=defect_type)
            iou  = compute_iou(mask, gt_mask)
            if iou is not None:
                ious.append(iou)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        results_per_type[defect_type] = {"mean_iou": mean_iou, "n": len(ious)}
        print(f"  {defect_type:25s}  →  mean IoU: {mean_iou:.4f}  (n={len(ious)})")

    all_ious = [v["mean_iou"] for v in results_per_type.values()]
    overall  = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"\n  Overall mean IoU [{category}] : {overall:.4f}")

    return {"per_type": results_per_type, "overall_iou": overall}
