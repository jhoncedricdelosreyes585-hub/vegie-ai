import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import json

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_DIR = "veg_dataset"
TRAIN_MODEL = False  # Set to True to retrain

checkpoint_path = os.path.join("saved_model", "efficientnet_best.keras")
os.makedirs("saved_model", exist_ok=True)

# Map each class to its family (add as needed)
FAMILY_MAPPING = {
    "Bean": "Legume",
    "Bitter_Gourd": "Gourd",
    "Bottle_Gourd": "Gourd",
    "Brinjal": "Nightshade",
    "Broccoli": "Cruciferous",
    "Cabbage": "Cruciferous",
    "Capsicum": "Nightshade",
    "Carrot": "Root",
    "Cauliflower": "Cruciferous",
    "Cucumber": "Gourd",
    "Papaya": "Tropical Fruit",
    "Potato": "Nightshade",
    "Pumpkin": "Gourd",
    "Radish": "Root",
    "Tomato": "Nightshade",
    "Onion": "Allium"
}

VEGETABLE_INFO = {
    "Bean": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 31,
        "Key Nutrients": ["Protein", "Fiber", "Vitamin C"]
    },
    "Bitter_Gourd": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 17,
        "Key Nutrients": ["Vitamin C", "Vitamin A", "Iron"]
    },
    "Bottle_Gourd": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 14,
        "Key Nutrients": ["Vitamin C", "Calcium", "Magnesium"]
    },
    "Brinjal": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 25,
        "Key Nutrients": ["Fiber", "Vitamin B6", "Antioxidants"]
    },
    "Broccoli": {
        "Growing Season": "Cool season (Fall/Spring)",
        "Calories (per 100g)": 34,
        "Key Nutrients": ["Vitamin C", "Vitamin K", "Fiber"]
    },
    "Cabbage": {
        "Growing Season": "Cool season (Fall/Spring)",
        "Calories (per 100g)": 25,
        "Key Nutrients": ["Vitamin K", "Vitamin C", "Folate"]
    },
    "Capsicum": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 20,
        "Key Nutrients": ["Vitamin C", "Vitamin B6", "Beta-Carotene"]
    },
    "Carrot": {
        "Growing Season": "Cool season (Fall/Spring)",
        "Calories (per 100g)": 41,
        "Key Nutrients": ["Vitamin A", "Fiber", "Potassium"]
    },
    "Cauliflower": {
        "Growing Season": "Cool season (Fall/Winter)",
        "Calories (per 100g)": 25,
        "Key Nutrients": ["Vitamin C", "Vitamin K", "Folate"]
    },
    "Cucumber": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 16,
        "Key Nutrients": ["Vitamin K", "Water", "Potassium"]
    },
    "Papaya": {
        "Growing Season": "Tropical / Year-round",
        "Calories (per 100g)": 43,
        "Key Nutrients": ["Vitamin C", "Vitamin A", "Folate"]
    },
    "Potato": {
        "Growing Season": "Cool season (Spring/Fall)",
        "Calories (per 100g)": 77,
        "Key Nutrients": ["Carbohydrates", "Vitamin C", "Potassium"]
    },
    "Pumpkin": {
        "Growing Season": "Warm season (Summer/Fall)",
        "Calories (per 100g)": 26,
        "Key Nutrients": ["Vitamin A", "Fiber", "Beta-Carotene"]
    },
    "Radish": {
        "Growing Season": "Cool season (Winter)",
        "Calories (per 100g)": 16,
        "Key Nutrients": ["Vitamin C", "Folate", "Potassium"]
    },
    "Tomato": {
        "Growing Season": "Warm season (Summer)",
        "Calories (per 100g)": 18,
        "Key Nutrients": ["Vitamin C", "Lycopene", "Potassium"]
    },
    "Onion": {
        "Growing Season": "Cool to warm season",
        "Calories (per 100g)": 40,
        "Key Nutrients": ["Vitamin C", "Manganese", "Antioxidants"]
    }
}

# -------------------------------
# Dataset
# -------------------------------
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "validation")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# -------------------------------
# Model Training / Loading
# -------------------------------
if TRAIN_MODEL or not os.path.exists(checkpoint_path):
    base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    base_model.trainable = False

    inputs = layers.Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(train_gen.num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

    model.fit(train_gen, validation_data=val_gen, epochs=10,
              callbacks=[checkpoint, earlystop, reduce_lr])
    model = tf.keras.models.load_model(checkpoint_path)
else:
    model = tf.keras.models.load_model(checkpoint_path)

# Use VEGETABLE_INFO keys for class labels (ensure dataset folders match)
class_labels = list(VEGETABLE_INFO.keys())

# -------------------------------
# Prediction Function
# -------------------------------
def detect_vegetables(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Predict with TTA (original + horizontally flipped)
    pred1 = model.predict(arr, verbose=0)[0]
    pred2 = model.predict(np.flip(arr, axis=2), verbose=0)[0]
    pred = (pred1 + pred2) / 2

    idx = np.argmax(pred)
    label = class_labels[idx]
    confidence = float(np.max(pred))
    family = FAMILY_MAPPING.get(label, "Unknown")
    info = VEGETABLE_INFO.get(label, {
        "Growing Season": "Unknown",
        "Calories (per 100g)": "Unknown",
        "Key Nutrients": []
    })

    return {
        "image_path": image_path,
        "overall_prediction": label,
        "family": family,
        "confidence": round(confidence * 100, 2),
        "extra_info": info
    }

# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    test_image = "test.jpg"  # Replace with your test image path
    if not os.path.exists(test_image):
        print("❌ Please place a test.jpg image in your directory.")
    else:
        result = detect_vegetables(test_image)
        print(json.dumps(result, indent=2))
