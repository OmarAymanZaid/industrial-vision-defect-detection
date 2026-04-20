"""
Module 5 — Segmentation (Improved)
====================================
Strategy per defect type:
  - broken_large  : Otsu on grayscale  (high intensity contrast)
  - broken_small  : Otsu on diff-image (subtract a "good" reference)
  - contamination : Otsu on HSV-S channel (color anomaly)
  - good          : returns empty mask

All paths use:
  1. CLAHE  → enhance local contrast
  2. Gaussian blur → reduce noise before thresholding
  3. Otsu / Adaptive threshold
  4. Morphological open+close → clean mask
  5. IoU vs ground-truth mask

Usage (inside Kaggle notebook):
    from segmentation import run_segmentation
    results = run_segmentation(img_bgr, gt_mask, reference_good_img)
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────
def _clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clean_mask(mask, kernel_size=7):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _otsu(gray_enhanced):
    blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _iou(pred_mask, gt_mask):
    """Compute IoU between binary masks (any non-zero = foreground)."""
    if gt_mask is None:
        return None
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred,  gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ──────────────────────────────────────────────
# per-strategy segmenters
# ──────────────────────────────────────────────
def _segment_broken_large(img_bgr):
    """Otsu on grayscale + CLAHE (works well for large cracks)."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enh   = _clahe(gray)
    mask  = _otsu(enh)
    # keep only the darker region (crack is darker than bottle)
    # flip if background is thresholded instead of defect
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)
    mask = _clean_mask(mask, kernel_size=7)
    return mask


def _segment_broken_small(img_bgr, ref_good_bgr):
    """
    Diff image approach:
      diff = |defective_gray - reference_gray|
      Then Otsu on diff to find anomaly region.
    Falls back to grayscale Otsu if no reference provided.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if ref_good_bgr is not None:
        ref  = cv2.cvtColor(ref_good_bgr, cv2.COLOR_BGR2GRAY)
        # resize reference to match (safety)
        ref  = cv2.resize(ref, (gray.shape[1], gray.shape[0]))
        diff = cv2.absdiff(gray, ref)
        enh  = _clahe(diff)
    else:
        enh  = _clahe(gray)
    mask = _otsu(enh)
    mask = _clean_mask(mask, kernel_size=5)
    return mask


def _segment_contamination(img_bgr):
    """
    Contamination = foreign color.
    Work in HSV → Saturation channel highlights colored blobs on
    a nearly achromatic glass bottle.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s   = hsv[:, :, 1]          # saturation channel
    enh = _clahe(s)
    mask = _otsu(enh)
    mask = _clean_mask(mask, kernel_size=5)
    return mask


# ──────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────
def run_segmentation(img_bgr, gt_mask=None, ref_good_bgr=None, defect_type=None):
    """
    Parameters
    ----------
    img_bgr      : np.ndarray  BGR image (H×W×3)
    gt_mask      : np.ndarray | None   ground-truth binary mask (H×W)
    ref_good_bgr : np.ndarray | None   a "good" reference bottle image
    defect_type  : str | None   one of 'broken_large', 'broken_small',
                                'contamination', 'good'.
                                If None the function auto-selects a strategy.

    Returns
    -------
    dict with keys:
        'pred_mask'   : predicted binary mask (np.ndarray)
        'iou'         : float | None
        'strategy'    : str  name of strategy used
    """
    if defect_type == "good":
        h, w = img_bgr.shape[:2]
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        return {"pred_mask": pred_mask, "iou": _iou(pred_mask, gt_mask), "strategy": "none (good)"}

    # auto-select strategy
    if defect_type == "contamination":
        pred_mask = _segment_contamination(img_bgr)
        strategy  = "HSV-Saturation + Otsu"
    elif defect_type == "broken_small":
        pred_mask = _segment_broken_small(img_bgr, ref_good_bgr)
        strategy  = "Diff-Image + Otsu" if ref_good_bgr is not None else "Grayscale + Otsu"
    else:
        # broken_large OR unknown → grayscale Otsu (most robust default)
        pred_mask = _segment_broken_large(img_bgr)
        strategy  = "Grayscale-CLAHE + Otsu"

    iou = _iou(pred_mask, gt_mask)
    return {"pred_mask": pred_mask, "iou": iou, "strategy": strategy}
