import os
import numpy as np
from modules.preprocessing import preprocess_image
from modules.segmentation import segment_image
from modules.features import extract_features

def load_image_paths(data_path, category="bottle", split="train", max_images=None):
    """
    Returns list of image paths from dataset.
    """

    base_path = os.path.join(data_path, category, split)

    image_paths = []

    for defect_type in os.listdir(base_path):
        folder = os.path.join(base_path, defect_type)

        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            image_paths.append(img_path)

    if max_images:
        image_paths = image_paths[:max_images]

    return image_paths

# -----------------------------
# Detect environment
# -----------------------------
def get_data_path():
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/datasets/ipythonx/mvtec-ad"
    return "data"