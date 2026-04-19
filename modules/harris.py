import cv2
import numpy as np


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

    # Dilate for better visualization
    response = cv2.dilate(response, None)

    # Copy image for visualization
    corners_img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Mark corners
    corners_img[response > threshold * response.max()] = [0, 0, 255]

    return corners_img, response