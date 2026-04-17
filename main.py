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

# -----------------------------
# Detect environment
# -----------------------------
def get_data_path():
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/mvtec-ad"
    return "data"


# -----------------------------
# Main
# -----------------------------
def run_preprocessing(category="bottle"):
    DATA_PATH = get_data_path()

    # Pick ONE sample image (start simple)
    sample_path = os.path.join(
        DATA_PATH,
        category,
        "train",
        "good"
    )

    # Get first image
    img_name = os.listdir(sample_path)[0]
    img_path = os.path.join(sample_path, img_name)

    print(f"Processing: {img_path}")

    img = cv2.imread(img_path)

    # Run preprocessing
    results = preprocess_image(img)

    # Save outputs
    output_dir = os.path.join("outputs", "preprocessing", category)
    save_results(results, output_dir, image_name="sample")

    # Print metrics
    print_metrics(results["metrics"])


if __name__ == "__main__":
    run_preprocessing("bottle")