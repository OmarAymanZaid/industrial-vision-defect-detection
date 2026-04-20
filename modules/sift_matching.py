import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Feature Extraction
# -----------------------------
def sift_extract(gray, n_features=0):
    """
    Extract SIFT keypoints and 128-D descriptors from a grayscale image.

    Args:
        gray      : grayscale uint8 image
        n_features: max keypoints to keep (0 = all)

    Returns:
        keypoints  : list of cv2.KeyPoint
        descriptors: np.ndarray of shape (N, 128) or None
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


# -----------------------------
# Matching
# -----------------------------
def match_bf(desc1, desc2, ratio_thresh=0.75):
    """
    Brute-Force matching with Lowe's ratio test.
    """
    if desc1 is None or desc2 is None:
        return []
    bf  = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in raw if m.distance < ratio_thresh * n.distance]


def match_flann(desc1, desc2, ratio_thresh=0.75):
    """
    FLANN-based matching with Lowe's ratio test (faster for large sets).
    """
    if desc1 is None or desc2 is None:
        return []
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw   = flann.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
    return [m for m, n in raw if m.distance < ratio_thresh * n.distance]

def filter_matches_ransac(kp1, kp2, matches, min_match_count=10):
    """
    Filters matches using RANSAC to ensure geometric consistency.
    """
    if len(matches) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Keep only inliers
        matches_mask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(matches) if matches_mask[i] == 1]
        return good_matches
    return matches
# -----------------------------
# Metric
# -----------------------------
def match_score(n_matches, n_ref_keypoints):
    """
    Normalized match count: fraction of reference keypoints that found a good match.
    Higher = more similar. Lower = possible defect.
    """
    return n_matches / n_ref_keypoints if n_ref_keypoints > 0 else 0.0


# -----------------------------
# Visualization
# -----------------------------
def visualize_keypoints(gray, keypoints, title="SIFT Keypoints"):
    img_kp = cv2.drawKeypoints(
        gray, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(8, 6))
    plt.title(f"{title}  ({len(keypoints)} keypoints)")
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_matches(gray1, kp1, gray2, kp2, matches, title="Matches", max_draw=50):
    matched_img = cv2.drawMatches(
        gray1, kp1, gray2, kp2,
        matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(16, 6))
    plt.title(f"{title}  ({min(len(matches), max_draw)} shown / {len(matches)} total)")
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Full SIFT Pipeline (single pair)
# -----------------------------

def sift_compare(gray1, gray2, ratio_thresh=0.75, visualize=True):
    kp1, desc1 = sift_extract(gray1)
    kp2, desc2 = sift_extract(gray2)
    
    # 1. هات الـ matches المبدئية
    matches = match_bf(desc1, desc2, ratio_thresh=ratio_thresh)
    
    # 2. فلترها بالـ RANSAC (ده السطر الزيادة)
    matches = filter_matches_ransac(kp1, kp2, matches)
    
    # 3. احسب السكور على الـ Matches اللي اتبقت
    score   = match_score(len(matches), len(kp1))
    
    if visualize:
        visualize_matches(gray1, kp1, gray2, kp2, matches,
                          title=f"SIFT Matches (RANSAC) | score={score:.3f}")
    return {
        "kp1": kp1, "desc1": desc1,
        "kp2": kp2, "desc2": desc2,
        "matches": matches,
        "match_score": score,
    }