
"""
Created on Thu Oct 30 17:55:24 2025

@author: Amir Akram
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)


"""Step 1: Data Processing"""
# Define directories for training, validation, and test datasets
train_data_dir = "./Project 2 Data/train" 
val_data_dir = "./Project 2 Data/valid" 
test_data_dir = "./Project 2 Data/test" 

#Define Image Shape
image_shape=(500,500);

#Number of epochs for training
epochs=20;

#Load training images and their labels
train_data = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,              # Path to training folder
    label_mode='categorical',         
    batch_size=32,              # Number of images per batch
    shuffle=True,               # Randomize image order
    image_size = image_shape,   # Resize all images to this size
    color_mode='rgb'      # Convert to grayscale
)

# Load validation images
validation_data = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    label_mode='categorical',
    batch_size=32,
    shuffle=False,              # No shuffle needed for validation
    image_size = image_shape,
    color_mode='rgb'
)

# Load test images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='categorical',
    batch_size=32,
    shuffle=False,              # No shuffle needed for test
    image_size = image_shape,
    color_mode='rgb'
)

# Data preprocessing
# --------------------------
# Normalize pixel values to [0,1]
rescale = layers.Rescaling(1./255)
class_names = train_data.class_names

# Data augmentation (ONLY to training data)
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.1),
])

# Apply augmentation to training data
train_data = train_data.map(lambda x, y: (data_augmentation(x), y))

# Apply only rescaling to validation and test data
validation_data = validation_data.map(lambda x, y: (rescale(x), y))
test_data = test_data.map(lambda x, y: (rescale(x), y))


print("Data processing complete.")
print("Classes:", class_names)

"""Step 2: Neural Network Architecture and Step 3: Hyperparameter Analysis"""

# ------------ Model 1 ------------

# Define CNN architecture
model = models.Sequential([
    # --- Convolution + Pooling Layers ---
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # --- Flatten and Dense Layers ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    # --- Dropout Layer to reduce overfitting ---
    layers.Dropout(0.5),   # randomly disables 50% of neurons during training
    
    # --- Output layer for 3 classes ---
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile Model 1
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])


# Print model 1 summary 
model.summary()

# Train Model
history=model.fit(
    train_data,               # Training data
    epochs=epochs,                # Number of training epochs
    validation_data=validation_data,
)

# ------------ Model 2 ------------

# Define CNN architecture
model2 = models.Sequential([
    # --- Convolution + Pooling Layers ---
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # --- Flatten and Dense Layers ---
    layers.Flatten(),
    layers.Dense(256, activation='relu'),

     # --- Dropout Layer to reduce overfitting ---
    layers.Dropout(0.5),   # randomly disables 50% of neurons during training
    
    # --- Output layer for 3 classes ---
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile Model 2
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

# Print model 2 summary
model2.summary()

# Train Model 2
history2=model2.fit(
    train_data,               # Training data
    epochs=epochs,                # Number of training epochs
    validation_data=validation_data,
)


"""Step 4: Model Evaluation"""

# Plot training and validation accuracy & loss
def plot_history(history, title_prefix="Model"):
    plt.figure(figsize=(10,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Plotting models
plot_history(history, "Model 1")
plot_history(history2, "Model 2")

# ------------ Model Evaluation ------------

# Evaluate model #1 on unseen test data
test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nModel 1 Final Test Accuracy: {test_accuracy:.4f}")
print(f"Model 1 Final Test Loss: {test_loss:.4f}")


# Evaluate model #2 on unseen test data
test_loss2, test_accuracy2 = model2.evaluate(test_data)

print(f"\nModel 2 Final Test Accuracy: {test_accuracy2:.4f}")
print(f"Model 2 Final Test Loss: {test_loss2:.4f}")


#Saving Both Models
model.save("model1.keras")
model2.save("model2.keras")















