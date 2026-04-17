# Industrial Vision Defect Detection

A modular computer vision pipeline for **industrial quality inspection and defect detection**, covering preprocessing, feature detection, multi-scale analysis, segmentation, feature matching, and classification.

---

## 📌 Overview

This project implements an **end-to-end classical computer vision pipeline** designed for defect detection in industrial settings. The system is built in a modular way to allow experimentation, evaluation, and extension.

> Current focus: single-object inspection (e.g., bottles) — designed to generalize to similar industrial scenarios.

---

## 🧠 Pipeline Architecture

```
Input Image
     │
     ▼
[1] Preprocessing
     │
     ▼
[2] Feature Detection (Harris)
     │
     ▼
[3] Multi-Scale Analysis (Pyramids)
     │
     ▼
[4] Feature Extraction & Matching (SIFT)
     │
     ▼
[5] Segmentation
     │
     ▼
[6] Classification
     │
     ▼
Output: Defective / Non-Defective
```

---

## ⚙️ Modules

### 1. Image Preprocessing

* Gaussian Filtering
* Median Filtering
* Noise reduction and smoothing
* Quantitative comparison using:

  * MSE
  * PSNR

---

### 2. Feature Detection

* Harris Corner Detector
* Threshold tuning analysis
* Visualization of detected corners

---

### 3. Multi-Scale Analysis

* Gaussian Pyramid
* Laplacian Pyramid
* Scale-space representation of objects

---

### 4. Feature Extraction & Matching

* SIFT keypoint detection
* Descriptor extraction
* Feature matching using BFMatcher
* Match visualization

---

### 5. Image Segmentation

* Thresholding (Otsu)
* Morphological operations
* Optional: K-means clustering
* Output: segmented defect regions

---

### 6. Classification

* Models:

  * Naive Bayes
  * AdaBoost (optional)
* Input features:

  * Texture
  * Region properties
  * Keypoint statistics

---

### 7. Final Pipeline

All modules are integrated into a single workflow for:

* Image input
* Processing
* Defect prediction

---

## 📊 Evaluation Metrics

| Task              | Metric                        |
| ----------------- | ----------------------------- |
| Preprocessing     | PSNR, MSE                     |
| Feature Detection | Corner stability              |
| Matching          | Matching accuracy             |
| Segmentation      | IoU (Intersection over Union) |
| Classification    | Accuracy, Precision, Recall   |

---

## 📂 Project Structure

```
industrial-vision-defect-detection/
│
├── data/                  # Dataset
├── outputs/               # Saved results & visualizations
│
├── preprocessing.py
├── harris.py
├── pyramid.py
├── sift_matching.py
├── segmentation.py
├── classification.py
├── pipeline.py
│
├── utils.py
├── requirements.txt
└── README.md
```

---

## 📥 Dataset

This project uses the **MVTec Anomaly Detection Dataset**.

* Public dataset for industrial defect detection
* Contains high-quality images with various defect types
* Minimum requirement: 200+ images

👉 Download from Kaggle and place inside:

```
data/
```

---

## ▶️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/industrial-vision-defect-detection.git
cd industrial-vision-defect-detection
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the pipeline

```bash
python pipeline.py
```

---

## 📸 Example Outputs

* Filter comparison (Gaussian vs Median)
* Harris corner visualization
* Multi-scale pyramid images
* SIFT keypoint matching
* Segmented defect regions
* Final classification result

---

## 🧪 Experiments & Analysis

The project emphasizes:

* Parameter tuning (e.g., Harris threshold)
* Visual vs numerical comparisons
* Multi-scale behavior of defects
* Matching robustness between normal and defective samples

---

## 🔮 Future Improvements

* Deep learning-based defect detection (CNNs / Autoencoders)
* Real-time inspection system
* Support for multiple object categories
* GUI for interactive inspection

---

## 📚 Attribution

If you use the dataset in scientific work, please cite:

> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
> *"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"*,
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

---

## 🤝 Contributing

Contributions are welcome.
Feel free to open issues or submit pull requests for improvements.

---

## 📄 License

This project is for educational and research purposes.
