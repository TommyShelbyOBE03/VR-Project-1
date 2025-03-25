import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate
from tensorflow.keras.models import Model
import numpy as np

#  Step 1: Define the Corrected U-Net Model
def unet_model(input_size=(128, 128, 3)):  # Ensure input is 128x128
    inputs = Input(input_size)

    # Encoder (Downsampling)
    c1 = Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(128, (3,3), activation="relu", padding="same")(p1)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, (3,3), activation="relu", padding="same")(p2)
    p3 = MaxPooling2D((2,2))(c3)

    # Decoder (Upsampling - Now matches 128x128)
    u1 = UpSampling2D((2,2))(p3)
    c4 = Conv2D(128, (3,3), activation="relu", padding="same")(u1)

    u2 = UpSampling2D((2,2))(c4)
    c5 = Conv2D(64, (3,3), activation="relu", padding="same")(u2)

    u3 = UpSampling2D((2,2))(c5)  # Added to ensure 128x128
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(u3)  # Matches input size

    return Model(inputs, outputs)

#  Step 2: Load Dataset
X = np.load("../dataset/X_faces.npy")  # Face images (128x128)
Y = np.load("../dataset/Y_masks.npy")  # Mask images (128x128)

# Ensure dataset shape matches model output
X = X.astype("float32") / 255.0
Y = Y.astype("float32") / 255.0
Y = np.expand_dims(Y, axis=-1)  # Ensure shape is (128,128,1)

# Step 3: Train the U-Net Model
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2)

#  Step 4: Save the Trained Model
model.save("../models/unet_mask_segmenter.h5")
print("✅ U-Net Model Trained and Saved Successfully!")
