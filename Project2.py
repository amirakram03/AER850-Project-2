
"""
Created on Thu Oct 30 17:55:24 2025

@author: Amir Akram
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from pathlib import Path

# For reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)


"""Step 1: Data Processing"""

train_data_dir = "./Project 2 Data/train" 
val_data_dir = "./Project 2 Data/valid" 
test_data_dir = "./Project 2 Data/test" 

#Define Image Shape
image_shape=(500,500);

train_data = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,              # Image Directory
    label_mode='categorical',         
    batch_size=32,
    image_size = image_shape,    
    shuffle=True,
    color_mode='grayscale'
)

# Load validation images
validation_data = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    label_mode='categorical',
    batch_size=32,
    image_size = image_shape,
    shuffle=False,
    color_mode='grayscale'
)

# Load test images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='categorical',
    batch_size=32,
    image_size = image_shape,
    shuffle=False,
    color_mode='grayscale'
)

# Data preprocessing
# --------------------------
# Normalize pixel values to [0,1]
rescale = layers.Rescaling(1./255)
class_names = train_data.class_names

# Data augmentation (ONLY to training data)
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.05),
])

# Apply augmentation to training data
train_data = train_data.map(lambda x, y: (data_augmentation(x), y))

# Apply only rescaling to validation and test data
validation_data = validation_data.map(lambda x, y: (rescale(x), y))
test_data = test_data.map(lambda x, y: (rescale(x), y))


print("Data processing complete.")
print("Classes:", class_names)

"""Step 2: Neural Network Architecture and Step 3: Hyperparameter Analysis"""

# Define CNN architecture
model = models.Sequential([
    # --- Convolution + Pooling Layers ---
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # --- Flatten and Fully Connected Layers ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    # --- Output Layer ---
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Print model summary
model.summary()

# Train Model
history=model.fit(
    train_data,               # Training data
    epochs=10,                # Number of training epochs
    validation_data=validation_data  
)

"""Step 4: Model Evaluation"""

# Plot training and validation accuracy & loss
plt.figure(figsize=(10,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0, 2)
plt.show()

# Evaluate the model on unseen test data
test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")





















