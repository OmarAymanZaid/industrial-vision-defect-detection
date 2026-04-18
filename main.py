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
# Main
# -----------------------------
def run_batch_preprocessing(category="bottle"):
    DATA_PATH = get_data_path()

    print(f"\nLoading dataset: {category}")

    image_paths = load_image_paths(
        DATA_PATH,
        category=category,
        split="train",   # start with train (good images)
        max_images=100   # limit for speed (you can remove later)
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
if __name__ == "__main__":
    run_batch_preprocessing("bottle")