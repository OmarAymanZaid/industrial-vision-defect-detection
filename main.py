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


# Detect environment
if os.path.exists("/kaggle/input"):
    DATA_PATH = "/kaggle/input/mvtec-ad"
else:
    DATA_PATH = "data"

# # -----------------------------
# # Utility Functions
# # -----------------------------
# def show(title, img, cmap='gray'):
#     plt.figure()
#     if len(img.shape) == 2:
#         plt.imshow(img, cmap=cmap)
#     else:
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis('off')


# # -----------------------------
# # 1. Preprocessing
# # -----------------------------
# def preprocessing(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
#     median = cv2.medianBlur(gray, 5)

#     return gray, gaussian, median


# # -----------------------------
# # 2. Harris Corner Detection
# # -----------------------------
# def harris_detection(gray):
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#     dst = cv2.dilate(dst, None)

#     threshold = 0.01 * dst.max()
#     corners = dst > threshold

#     return corners


# # -----------------------------
# # 3. Pyramid
# # -----------------------------
# def build_pyramid(img, levels=3):
#     pyramid = [img]
#     for _ in range(levels):
#         img = cv2.pyrDown(img)
#         pyramid.append(img)
#     return pyramid


# # -----------------------------
# # 4. SIFT Feature Matching
# # -----------------------------
# def sift_features(img):
#     sift = cv2.SIFT_create()
#     kp, des = sift.detectAndCompute(img, None)
#     return kp, des


# # -----------------------------
# # 5. Segmentation (Otsu)
# # -----------------------------
# def segmentation(gray):
#     _, thresh = cv2.threshold(
#         gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )
#     return thresh


# # -----------------------------
# # MAIN PIPELINE
# # -----------------------------
# def run_pipeline(image_path):
#     if not os.path.exists(image_path):
#         print("Image not found:", image_path)
#         return

#     img = cv2.imread(image_path)

#     # 1. Preprocessing
#     gray, gaussian, median = preprocessing(img)

#     show("Original", img)
#     show("Gaussian Filter", gaussian)
#     show("Median Filter", median)

#     # 2. Harris
#     corners = harris_detection(gray)
#     harris_img = img.copy()
#     harris_img[corners] = [0, 0, 255]

#     show("Harris Corners", harris_img)

#     # 3. Pyramid
#     pyramid = build_pyramid(gray)
#     for i, level in enumerate(pyramid):
#         show(f"Pyramid Level {i}", level)

#     # 4. SIFT
#     kp, des = sift_features(gray)
#     sift_img = cv2.drawKeypoints(
#         img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
#     )
#     show("SIFT Keypoints", sift_img)

#     # 5. Segmentation
#     seg = segmentation(gray)
#     show("Segmentation (Otsu)", seg)

#     plt.show()


# # -----------------------------
# # ENTRY POINT
# # -----------------------------
# if __name__ == "__main__":
#     # Change this to your dataset image
#     sample_image = "data/sample.jpg"

#     run_pipeline(sample_image)