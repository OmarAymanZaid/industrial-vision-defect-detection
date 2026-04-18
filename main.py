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
def run_single_pipeline(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    # -----------------------------
    # STEP 1: Preprocessing
    # -----------------------------
    preprocessed = preprocess_image(img)

    # Choose ONE version (important design choice)
    processed_img = preprocessed["median"]

    # -----------------------------
    # STEP 2: Feature Detection
    # -----------------------------
    # corners = harris_detect(processed_img)

    # -----------------------------
    # STEP 3: Multi-scale Analysis
    # -----------------------------
    # pyramid = build_pyramid(processed_img)

    # -----------------------------
    # STEP 4: Feature Extraction & Matching
    # -----------------------------
    # keypoints, descriptors = sift_extract(processed_img)

    # -----------------------------
    # STEP 5: Segmentation
    # -----------------------------
    # mask = segment(processed_img)

    # -----------------------------
    # STEP 6: Classification
    # -----------------------------
    # label = classify(processed_img, mask)

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    # print(f"Prediction: {label}")

    print("Pipeline executed (placeholders for remaining modules).")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    # -------- Option 1: Evaluate preprocessing --------
    run_batch_preprocessing("bottle")

    # -------- Option 2: Run full pipeline (single image) --------
    # sample_image = "data/bottle/test/good/000.png"
    # run_single_pipeline(sample_image)