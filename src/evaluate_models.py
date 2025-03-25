import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import tensorflow as tf
import cv2
from skimage.feature import hog
import random

#  Load full dataset
X_test_full = np.load("../dataset/X_faces.npy")  
Y_test_full = np.load("../dataset/Y_masks.npy")  

#  Randomly select 500 images
num_samples = 500
random_indices = np.random.choice(len(X_test_full), num_samples, replace=False)  # Unique random indices

X_test = X_test_full[random_indices]  
Y_test = Y_test_full[random_indices]

# Ensure test images have 3 channels (convert grayscale to RGB if needed)
def preprocess_images(images, target_size=(224, 224)):
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return np.array(resized_images, dtype=np.float32) / 255.0  # Normalize to [0,1]

X_test_resized = preprocess_images(X_test)

#  Convert Segmentation Mask to Binary Labels
# Convert each segmentation mask into a single binary class (Mask or No Mask)
Y_test_binary = (Y_test.mean(axis=(1, 2)) > 0.5).astype(int)  # 1 if mask covers significant area, else 0

#  Load Models
cnn_model1 = tf.keras.models.load_model("../models/mask_detector_adam_relu.keras")
cnn_model2 = tf.keras.models.load_model("../models/mask_detector_adam_tanh.keras")
svm_model = joblib.load("../models/svm_model.pkl")
mlp_model = joblib.load("../models/mlp_model.pkl")

#  Process CNN Predictions in Batches to Prevent Memory Issues
batch_size = 32
cnn1_preds, cnn2_preds = [], []

for i in range(0, len(X_test_resized), batch_size):
    batch = X_test_resized[i : i + batch_size]
    cnn1_preds.append((cnn_model1.predict(batch) > 0.5).astype(int))
    cnn2_preds.append((cnn_model2.predict(batch) > 0.5).astype(int))

#  Flatten Predictions for Accuracy Calculation
cnn1_pred_binary = np.concatenate(cnn1_preds, axis=0).flatten()
cnn2_pred_binary = np.concatenate(cnn2_preds, axis=0).flatten()

#  Evaluate CNNs
cnn1_acc = accuracy_score(Y_test_binary, cnn1_pred_binary)
cnn2_acc = accuracy_score(Y_test_binary, cnn2_pred_binary)



#  Define a function to extract HOG features
def extract_hog_features(images):
    features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        img_resized = cv2.resize(img_gray, (64, 64))  # ✅ Resize to 64x64 like in training
        hog_feature = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feature)
    return np.array(features)

#  Extract HOG features before passing to SVM
X_test_hog = extract_hog_features(X_test_resized)  # Now shape is (num_samples, 1764)


#  Evaluate SVM (Requires Flattened Input)
svm_pred = svm_model.predict(X_test_hog)
svm_acc = accuracy_score(Y_test_binary, svm_pred)

# Evaluate MLP Classifier
mlp_pred = mlp_model.predict(X_test_hog)
mlp_acc = accuracy_score(Y_test_binary, mlp_pred)

# Print Results
print(f"CNN1 Accuracy (Adam + ReLU): {cnn1_acc:.4f}")
print(f"CNN2 Accuracy (Adam + Tanh): {cnn2_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"MLP Accuracy: {mlp_acc:.4f}")
