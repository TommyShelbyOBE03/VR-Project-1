import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Step 1: Load Dataset
dataset_path = "../dataset/"
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=32, class_mode="binary", subset="training"
)
val_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=32, class_mode="binary", subset="validation"
)

# Step 2: Define Function to Build Model
def build_model(activation="relu"):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(128, activation=activation)(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Step 3: Train and Compare Models
for optimizer in ["adam", "sgd"]:
    for activation in ["relu", "tanh"]:
        print(f"\n🚀 Training Model: Optimizer={optimizer}, Activation={activation}")

        model = build_model(activation=activation)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model.fit(train_generator, validation_data=val_generator, epochs=10)

        # Save the trained model
        model.save(f"../models/mask_detector_{optimizer}_{activation}.keras")

print("✅ All Models Trained and Saved Successfully!")
