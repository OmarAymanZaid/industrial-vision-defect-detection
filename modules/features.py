import numpy as np
from modules.harris import harris_detect
from modules.sift_matching import sift_extract


def extract_features(processed_img, mask, pre_metrics=None):

    mse_val = pre_metrics["median"]["mse"] if pre_metrics else 0

    # Harris
    _, harris_response = harris_detect(processed_img)
    corner_count = np.sum(harris_response > 0.01 * harris_response.max())

    # Segmentation
    defect_area = np.sum(mask > 0) / mask.size

    # SIFT (converted to usable feature)
    keypoints, descriptors = sift_extract(processed_img)
    num_kp = len(keypoints)

    return [mse_val, corner_count, defect_area, num_kp]