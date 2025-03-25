import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # To save models

# Load dataset
dataset_path = "../dataset/"
classes = ["without_mask", "with_mask"]
X, y = [], []

for label, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))

        # Extract HOG features
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X.append(features)
        y.append(label)

X, y = np.array(X), np.array(y)

# Train SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "../models/svm_model.pkl")  # Save SVM model

# Train Neural Network (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
mlp_model.fit(X_train, y_train)
joblib.dump(mlp_model, "../models/mlp_model.pkl")  # Save MLP model

# Evaluate
svm_pred = svm_model.predict(X_test)
mlp_pred = mlp_model.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)}")
print(f"MLP Accuracy: {accuracy_score(y_test, mlp_pred)}")
