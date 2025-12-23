import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# -----------------------------
# 1. Dataset paths
# -----------------------------
train_dir = "../data/train"
test_dir = "../data/test"

# -----------------------------
# 2. Hyperparameters
# -----------------------------
img_height = 64
img_width = 64
batch_size = 32
num_classes = 46

# -----------------------------
# 3. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# -----------------------------
# 4. Dataset loading
# -----------------------------
train_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    test_dir,
    seed=123,
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# -----------------------------
# 5. Model architecture
# -----------------------------
model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=schedules.ExponentialDecay(1e-3, 100000, 0.96)),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 6. Train
# -----------------------------
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stopping])

# -----------------------------
# 7. Save model + history
# -----------------------------
model.save("../model/hindi_cnn_model.h5")

with open("../model/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# -----------------------------
# 8. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# -----------------------------
# 9. Plot results
# -----------------------------
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.savefig("../results/accuracy_plot.png")



