import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.classification import *
from modules.harris import *
from modules.preprocessing import *
from modules.pyramid import *
from modules.segmentation import *
from modules.sift_matching import *
from modules.utils import *

# -----------------------
# Clean final pipeline
# -----------------------
# img
# → preprocess(img)
# → segment(img)
# → features = extract_features(img)
# → classify(features)

# -----------------------------
# Detect environment
# -----------------------------
def get_data_path():
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/datasets/ipythonx/mvtec-ad"
    return "data"


# -----------------------------
# 1. Batch Preprocessing (Evaluation)
# -----------------------------
def run_batch_preprocessing(category="bottle"):
    DATA_PATH = get_data_path()

    print(f"\nLoading dataset: {category}")

    image_paths = load_image_paths(
        DATA_PATH,
        category=category,
        split="train",
        max_images=100
    )

    print(f"Total images loaded: {len(image_paths)}")

    output_dir = os.path.join("outputs", "preprocessing", category)

    avg_metrics = batch_preprocess(
        image_paths,
        save_samples=True,
        sample_limit=5,
        output_dir=output_dir
    )

    if avg_metrics:
        print_metrics(avg_metrics)


# -----------------------------
# 2. Single Image Pipeline (Inference)
# -----------------------------
def run_single_pipeline(defect_type, image_path, ref_image_path=None):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    # -----------------------------
    # STEP 1: Preprocessing
    # -----------------------------
    preprocessed = preprocess_image(img)
    processed_img = preprocessed["median"]

    # -----------------------------
    # STEP 2: Feature Detection
    # -----------------------------
    harris_img, harris_response = harris_detect(processed_img)
    visualize_harris(processed_img, harris_img)

    # -----------------------------
    # STEP 3: Multi-scale Analysis
    # -----------------------------
    g_pyr, l_pyr = build_pyramids(processed_img, levels=4)

    corner_counts = []
    for level in g_pyr:
        _, response = harris_detect(level)
        corners = np.sum(response > 0.01 * response.max())
        corner_counts.append(corners)

    results = {
        "corner_counts": corner_counts,
        "num_levels": len(g_pyr)
    }

    visualize_gaussian_pyramid(g_pyr)
    visualize_laplacian_pyramid(l_pyr)

    # -----------------------------
    # STEP 4: Feature Extraction & Matching (SIFT)
    # -----------------------------
    if ref_image_path is not None:
        ref_img  = cv2.imread(ref_image_path)
        ref_gray = preprocess_image(ref_img)["median"]

        sift_results = sift_compare(processed_img, ref_gray, visualize=True)

        print(f"\nSIFT Match Score : {sift_results['match_score']:.4f}")
        print(f"Keypoints (query): {len(sift_results['kp1'])}")
        print(f"Keypoints (ref)  : {len(sift_results['kp2'])}")
        print(f"Good Matches     : {len(sift_results['matches'])}")
    else:
        keypoints, descriptors = sift_extract(processed_img)
        visualize_keypoints(processed_img, keypoints, title="SIFT Keypoints")
        print(f"\nSIFT Keypoints detected: {len(keypoints)}")

    # -----------------------------
    # STEP 5: Segmentation
    # -----------------------------
    print(f"[*] Running Segmentation for: {defect_type}")
    mask = segment_image(processed_img, defect_type=defect_type)
    gt_mask = None
    gt_path = image_path.replace("test", "ground_truth").replace(".png", "_mask.png")
    if os.path.exists(gt_path):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        iou = compute_iou(mask, gt_mask)
        print(f"[+] Segmentation IoU: {iou:.4f}")
        visualize_segmentation(img, mask, gt_mask, iou=iou, title="Defect Segmentation")
    else:
        visualize_overlay(img, mask, title=f"Segmented Mask ({defect_type})")

    # -----------------------------
    # STEP 6: Classification
    # -----------------------------
    # label = classify(processed_img, mask)

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    print("Pipeline executed successfully.")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    # -------- Option 1: Evaluate preprocessing --------
    run_batch_preprocessing("bottle")

    # -------- Option 2: Run full pipeline (single image) --------
    # sample_image = "data/bottle/test/broken_large/000.png"
    # ref_image    = "data/bottle/test/good/000.png"
    # run_single_pipeline(defect_type="good", sample_image, ref_image_path=ref_image)
