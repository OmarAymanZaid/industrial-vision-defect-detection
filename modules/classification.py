import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


class IndustrialClassifier:
    def __init__(self, method='boosting'):
        if method == 'boosting':
            self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        else:
            self.model = GaussianNB()

        self.is_trained = False

    def train(self, X, y):
        if len(X) == 0:
            print("No data for training!")
            return

        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained on {len(X)} samples.")

    def predict(self, features):
        if not self.is_trained:
            return "Model not trained!"

        pred = self.model.predict([features])[0]
        return "Defective" if pred == 1 else "Non-Defective"

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        print("\n=== Classification Report ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Good', 'Defective']))