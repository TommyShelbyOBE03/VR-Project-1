import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

#  Step 1: Load Pretrained Models
mask_classifier = load_model("../models/mask_detector_adam_relu.keras")  # CNN classifier
mask_segmenter = load_model("../models/unet_mask_segmenter.h5")  # U-Net segmentation

#  Step 2: Initialize Face Detector
detector = dlib.get_frontal_face_detector()

#  Step 3: Start Webcam for Real-Time Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()

        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict Mask Usage
        mask_prob = mask_classifier.predict(face_resized)[0][0]
        label = "Mask" if mask_prob > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Segment Mask Region
        face_segmented = cv2.resize(face_roi, (128, 128)) / 255.0
        face_segmented = np.expand_dims(face_segmented, axis=0)
        mask_map = mask_segmenter.predict(face_segmented)[0]

        # Overlay Segmentation Map
        mask_map_resized = cv2.resize(mask_map, (face_roi.shape[1], face_roi.shape[0]))
        mask_overlay = np.uint8(mask_map_resized * 255)
        mask_overlay = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)

        # Display Results
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, mask_overlay, 0.4, 0)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
