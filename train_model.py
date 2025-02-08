import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Corrected dataset path
dataset_path = r"D:\Download Chrome\fer2013\fer2013.csv"

# Check if file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Load dataset
data = pd.read_csv(dataset_path)

# Extracting features and labels
X = []
y = []
for index, row in data.iterrows():
    pixels = np.array(row["pixels"].split(), dtype="float32").reshape(48, 48)
    X.append(pixels)
    y.append(row["emotion"])

# Converting to numpy arrays
X = np.array(X, dtype=np.uint8)  # Optimized memory usage
y = np.array(y, dtype=np.uint8)

# Splitting dataset before normalization (prevents data leakage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing pixel values (0 to 1)
X_train = X_train / 255.0
X_val = X_val / 255.0

# Reshaping for CNN input (Adding channel dimension)
X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)

# One-hot encoding labels
y_train = to_categorical(y_train, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)

# CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(7, activation='softmax')  
])

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Compile Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training
history = model.fit(
    X_train, y_train, 
    epochs=20,  # Ensuring it runs for 20 epochs
    batch_size=64, 
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]  # Removed EarlyStopping
)

# Save Model
model.save("emotion_model.h5")

# Print Accuracy for Each Epoch
print("\n📌 Training Completed! Below is the accuracy per epoch:\n")
for i, acc in enumerate(history.history["accuracy"]):
    val_acc = history.history["val_accuracy"][i]
    print(f"Epoch {i+1}: Train Accuracy = {acc*100:.2f}%, Validation Accuracy = {val_acc*100:.2f}%")

# Print Final Accuracy
final_train_acc = history.history["accuracy"][-1] * 100
final_val_acc = history.history["val_accuracy"][-1] * 100
print("\n🎯 Final Model Accuracy:")
print(f"✅ Train Accuracy: {final_train_acc:.2f}%")
print(f"✅ Validation Accuracy: {final_val_acc:.2f}%")

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue", marker="o")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="red", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title(f"Model Accuracy\nFinal Train: {final_train_acc:.2f}% | Final Val: {final_val_acc:.2f}%")
plt.grid()
plt.show()