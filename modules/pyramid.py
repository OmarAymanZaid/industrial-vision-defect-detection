import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_gaussian_pyramid(image, levels=4):
    #Constructs a Gaussian Pyramid.
    pyramid = [image]
    temp_img = image.copy()
    
    for _ in range(levels - 1):
        temp_img = cv2.pyrDown(temp_img) # bluring,then downsampling
        pyramid.append(temp_img)
        
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    #Constructs a Laplacian Pyramid from a given Gaussian Pyramid.
    levels = len(gaussian_pyramid)
    laplacian_pyramid = []
    
    for i in range(levels - 1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i]) # upsamling
        rows, cols = gaussian_pyramid[i - 1].shape[:2]
        gaussian_expanded = cv2.resize(gaussian_expanded, (cols, rows))
        
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.reverse()
    
    return laplacian_pyramid


def build_pyramids(image, levels=4):
    g_pyr = build_gaussian_pyramid(image, levels)
    l_pyr = build_laplacian_pyramid(g_pyr)
    return g_pyr, l_pyr


def visualize_gaussian_pyramid(pyramid):
    n = len(pyramid)
    plt.figure(figsize=(15, 5))

    for i, img in enumerate(pyramid):
        plt.subplot(1, n, i + 1)
        plt.title(f"Gaussian L{i}")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_laplacian_pyramid(pyramid):
    n = len(pyramid)
    plt.figure(figsize=(15, 5))

    for i, img in enumerate(pyramid):
        img_vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_vis = img_vis.astype(np.uint8)

        plt.subplot(1, n, i + 1)
        plt.title(f"Laplacian L{i}")
        plt.imshow(img_vis, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
