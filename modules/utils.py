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

def build_dataset(data_path, category, max_samples=50):


    import os, cv2
    from tqdm import tqdm

    X, y = [], []

    test_path = os.path.join(data_path, category, "test")

    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label = 0 if folder == "good" else 1

        images = os.listdir(folder_path)[:max_samples]

        for img_name in tqdm(images, desc=f"{folder}", leave=False):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                # Preprocess
                pre = preprocess_image(img)
                processed = pre["median"]

                # Segment
                mask = segment_image(processed, defect_type=folder)

                # Features
                feats = extract_features(processed, mask, pre["metrics"])

                X.append(feats)
                y.append(label)

            except Exception as e:
                print(f"Error: {e}")

    return np.array(X), np.array(y)

# -----------------------------
# Detect environment
# -----------------------------
def get_data_path():
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/datasets/ipythonx/mvtec-ad"
    return "data"