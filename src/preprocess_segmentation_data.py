import os
import cv2
import numpy as np

#  Define dataset paths
image_folder = "../dataset/mask_segmentation/images/"
mask_folder = "../dataset/mask_segmentation/masks/"

X, Y = [], []

# Load images and masks
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    mask_path = os.path.join(mask_folder, img_name)

    # Ensure mask exists for the image
    if not os.path.exists(mask_path):
        continue

    # Read and resize images
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize to [0,1]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128)) / 255.0  # Normalize to [0,1]

    X.append(img)
    Y.append(mask)

#  Convert lists to numpy arrays
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

#  Save as .npy files
np.save("../dataset/X_faces.npy", X)
np.save("../dataset/Y_masks.npy", Y)

print("✅ Dataset preprocessed and saved as .npy files!")
