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
from modules.features import *
from modules.utils import *

# -----------------------
# Clean final pipeline
# -----------------------
# img
# → preprocess(img)
# → segment(img)
# → features = extract_features(img)
# → classify(features)


# --------
# Helpers
# --------
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
#  Single Image Pipeline (Inference)
# -----------------------------
def run_single_pipeline(image_path, defect_type=None, clf_model=None, ref_image_path=None, debug=False):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    results = {}

    # -----------------------------
    # STEP 1: Preprocessing
    # -----------------------------
    preprocessed = preprocess_image(img)
    processed_img = preprocessed["median"]
    results["preprocessing_metrics"] = preprocessed["metrics"]

    # -----------------------------
    # STEP 2: Segmentation
    # -----------------------------
    mask = segment_image(processed_img, defect_type=defect_type)
    results["mask"] = mask

    # -----------------------------
    # IoU (optional)
    # -----------------------------
    gt_mask = None
    gt_path = image_path.replace("test", "ground_truth").replace(".png", "_mask.png")

    if os.path.exists(gt_path):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        iou = compute_iou(mask, gt_mask)
        results["iou"] = iou
    else:
        iou = None

    # -----------------------------
    # STEP 3: Feature Extraction
    # -----------------------------
    features = extract_features(processed_img, mask, preprocessed["metrics"])
    results["features"] = features

    # -----------------------------
    # STEP 4: Classification
    # -----------------------------
    prediction = None

    if clf_model is not None and clf_model.is_trained:
        prediction = clf_model.predict(features)
        results["prediction"] = prediction

        print("\n=== FINAL RESULT ===")
        print(f"Prediction: {prediction}")

        # Visualization
        color = (0, 255, 0) if prediction == "Non-Defective" else (0, 0, 255)

        display_img = img.copy()
        cv2.putText(display_img, f"{prediction}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title("Final Classification")
        plt.axis("off")
        plt.show()

    else:
        print("Classification skipped (model not trained).")

    # -----------------------------
    # DEBUG MODULES
    # -----------------------------
    if debug:

        # Harris
        harris_img, harris_response = harris_detect(processed_img)
        results["harris"] = harris_img
        visualize_harris(processed_img, harris_img)

        # Pyramid
        g_pyr, l_pyr = build_pyramids(processed_img, levels=4)

        corner_counts = []
        for level in g_pyr:
            _, response = harris_detect(level)
            corners = np.sum(response > 0.01 * response.max())
            corner_counts.append(corners)

        results["pyramid_corner_counts"] = corner_counts

        visualize_gaussian_pyramid(g_pyr)
        visualize_laplacian_pyramid(l_pyr)

        # SIFT
        if ref_image_path is not None:
            ref_img = cv2.imread(ref_image_path)
            ref_gray = preprocess_image(ref_img)["median"]

            sift_results = sift_compare(processed_img, ref_gray, visualize=True)

            results["sift"] = {
                "match_score": sift_results["match_score"],
                "num_matches": len(sift_results["matches"]),
            }

            print(f"\nSIFT Match Score: {sift_results['match_score']:.4f}")

        else:
            keypoints, _ = sift_extract(processed_img)
            results["sift_kp"] = len(keypoints)
            visualize_keypoints(processed_img, keypoints, title="SIFT Keypoints")

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    if iou is not None:
        print(f"IoU (Segmentation): {iou:.4f}")

    # Clean visualization
    if gt_mask is not None:
        visualize_segmentation(img, mask, gt_mask, iou=iou, title="Segmentation")
    else:
        visualize_overlay(img, mask, title="Segmentation Mask")

    return results


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    DATA_PATH = get_data_path()
    CATEGORY = "bottle"

    print("--- Training ---")
    clf = IndustrialClassifier(method='boosting')

    X, y = build_dataset(DATA_PATH, CATEGORY, max_samples=30)
    clf.train(X, y)

    print("\n--- Inference ---")
    sample_img = os.path.join(DATA_PATH, CATEGORY, "test", "broken_large", "000.png")

    run_single_pipeline(
        image_path=sample_img,
        defect_type="broken_large",
        clf_model=clf,
        debug=False
    )