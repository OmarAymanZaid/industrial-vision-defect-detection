import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_detect(gray, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Harris Corner Detection

    Args:
        gray: grayscale image
        block_size: neighborhood size
        ksize: kernel size
        k: Harris detector free parameter
        threshold: threshold for detecting strong corners

    Returns:
        corners_img: image with corners marked
        response: raw Harris response
    """

    gray = np.float32(gray)

    # Harris response
    response = cv2.cornerHarris(gray, block_size, ksize, k)

    # expand for better visualization
    response = cv2.dilate(response, None)

    # Copy image for visualization
    corners_img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Mark corners
    corners_img[response > threshold * response.max()] = [0, 0, 255]

    return corners_img, response


def analyze_threshold_tuning(gray):
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    results = []

    for t in thresholds:
        corners_img, response = harris_detect(gray, threshold=t)
        count = np.sum(response > t * response.max())
        results.append((t, count))
        print(f"threshold={t}: {count} corners")

    return results


def plot_threshold_results(results):
    thresholds = [r[0] for r in results]
    counts = [r[1] for r in results]

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, counts, marker='o')
    plt.title("Harris Threshold Tuning")
    plt.xlabel("Threshold")
    plt.ylabel("Number of Corners")
    plt.grid()
    plt.show()


def visualize_harris(gray, corners_img, title="Harris Corner Detection"):
    
    if len(gray.shape) == 3:
        gray_vis = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    else:
        gray_vis = gray

    harris_rgb = cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(gray_vis, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Harris Corners")
    plt.imshow(harris_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
