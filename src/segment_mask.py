import cv2
import os

#  Dynamically pick the first available image
image_folder = "../dataset/with_mask/"
image_files = os.listdir(image_folder)

if not image_files:
    raise FileNotFoundError(" No images found in 'with_mask' folder. Check dataset path.")

sample_image = os.path.join(image_folder, image_files[0])
img = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"OpenCV could not read the image: {sample_image}")

#  Proceed with segmentation
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
