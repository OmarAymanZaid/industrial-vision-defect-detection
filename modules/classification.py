import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


from modules.preprocessing import preprocess_image
from modules.harris import harris_detect
from modules.sift_matching import sift_compare
from modules.segmentation import segment_image

class IndustrialClassifier:
    def __init__(self, method='boosting'):
        """
        method: 'boosting' for AdaBoost or 'naive_bayes' for Naive Bayes
        """
        if method == 'boosting':
            self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        else:
            self.model = GaussianNB()
        self.is_trained = False

    def extract_features(self, img_bgr, ref_img_bgr=None, defect_type=None):
       
        # 1. Preprocessing
        pre_results = preprocess_image(img_bgr)
        processed_img = pre_results["median"]
        mse_val = pre_results['metrics']['median']['mse']

        # 2. Harris Corner Detection
        _, harris_response = harris_detect(processed_img)
        corner_count = np.sum(harris_response > 0.01 * harris_response.max())

        # 3. SIFT Matching 
        if ref_img_bgr is not None:
            ref_gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY)
            sift_results = sift_compare(processed_img, ref_gray, visualize=False)
            match_score = sift_results['match_score']
        else:
            match_score = 1.0 

        # 4. Segmentation
        mask = segment_image(processed_img, defect_type=defect_type)
        defect_area = np.sum(mask > 0) / mask.size

        return [mse_val, corner_count, match_score, defect_area]

    def build_train_sets(self, data_path, category, max_samples=50):
        """
        X & y 
        """
        X, y = [], []
        test_path = os.path.join(data_path, category, "test")
        
        
        ref_path = os.path.join(data_path, category, "train", "good", "000.png")
        ref_img = cv2.imread(ref_path) if os.path.exists(ref_path) else None

        print(f"[*] Extracting features for {category}...")
        
        for folder in os.listdir(test_path):
            folder_path = os.path.join(test_path, folder)
            if not os.path.isdir(folder_path) or folder == "ground_truth":
                continue
            
            label = 0 if folder == "good" else 1
            images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))][:max_samples]

            for img_name in tqdm(images, desc=f"Folder: {folder}", leave=False):
                img = cv2.imread(os.path.join(folder_path, img_name))
                if img is None: continue
                
                try:
                    features = self.extract_features(img, ref_img_bgr=ref_img, defect_type=folder)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

        return np.array(X), np.array(y)

    def train(self, X, y):
        if len(X) == 0:
            print(" No data found for training!")
            return
        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained on {len(X)} samples.")

    def predict(self, img_bgr, ref_img_bgr=None):
        """Defective or Non-Defective"""
        if not self.is_trained:
            return "Model not trained yet!"
        
        features = self.extract_features(img_bgr, ref_img_bgr)
        prediction = self.model.predict([features])[0]
        return "Defective" if prediction == 1 else "Non-Defective"

    def evaluate_model(self, X_test, y_test):
        """Accuracy"""
        y_pred = self.model.predict(X_test)
        print("\n" + "="*30)
        print("CLASSIFICATION REPORT")
        print("="*30)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Good', 'Defective']))