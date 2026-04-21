from tqdm import tqdm
import cv2
import numpy as np
import os

# -----------------------------
# Metrics
# -----------------------------
def compute_mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)


def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# -----------------------------
# Filters
# -----------------------------
def apply_gaussian(gray, kernel_size=(5, 5)):
    return cv2.GaussianBlur(gray, kernel_size, 0)


def apply_median(gray, ksize=5):
    return cv2.medianBlur(gray, ksize)


# -----------------------------
# Core Preprocessing
# -----------------------------
def preprocess_image(img):
    """
    Input:
        img: BGR image
    Returns:
        dict with processed images + metrics
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filters
    gaussian = apply_gaussian(gray)
    median = apply_median(gray)

    # Compute metrics (compare with original gray)
    mse_gaussian = compute_mse(gray, gaussian)
    mse_median = compute_mse(gray, median)

    psnr_gaussian = compute_psnr(gray, gaussian)
    psnr_median = compute_psnr(gray, median)

    results = {
        "gray": gray,
        "gaussian": gaussian,
        "median": median,
        "metrics": {
            "gaussian": {
                "mse": mse_gaussian,
                "psnr": psnr_gaussian,
            },
            "median": {
                "mse": mse_median,
                "psnr": psnr_median,
            },
        },
    }

    return results


# -----------------------------
# Saving Utilities
# -----------------------------
def save_results(results, output_dir, image_name="image"):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f"{image_name}_gray.png"), results["gray"])
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_gaussian.png"), results["gaussian"])
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_median.png"), results["median"])


# -----------------------------
# Logging
# -----------------------------
def print_metrics(metrics):
    print("\n=== Preprocessing Metrics ===")

    for method, values in metrics.items():
        print(f"\n[{method.upper()}]")
        print(f"MSE  : {values['mse']:.4f}")
        print(f"PSNR : {values['psnr']:.4f}")


# -----------------------------
# Batch Preprocessing
# -----------------------------
def batch_preprocess(image_paths, save_samples=False, sample_limit=5, output_dir=None):
    """
    Runs preprocessing on multiple images and aggregates metrics
    """

    total_mse_gaussian = 0
    total_mse_median = 0
    total_psnr_gaussian = 0
    total_psnr_median = 0

    count = 0

    for i, img_path in enumerate(tqdm(image_paths)):
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = preprocess_image(img)
        metrics = results["metrics"]

        total_mse_gaussian += metrics["gaussian"]["mse"]
        total_mse_median += metrics["median"]["mse"]

        total_psnr_gaussian += metrics["gaussian"]["psnr"]
        total_psnr_median += metrics["median"]["psnr"]

        # Save only few samples
        if save_samples and i < sample_limit and output_dir:
            save_results(results, output_dir, image_name=f"sample_{i}")

        count += 1

    # Avoid division by zero
    if count == 0:
        return None

    avg_metrics = {
        "gaussian": {
            "mse": total_mse_gaussian / count,
            "psnr": total_psnr_gaussian / count,
        },
        "median": {
            "mse": total_mse_median / count,
            "psnr": total_psnr_median / count,
        },
    }

    return avg_metrics
