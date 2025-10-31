
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
    shuffle=True,
    color_mode='grayscale'
)

# Load test images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='categorical',
    batch_size=32,
    image_size = image_shape,
    shuffle=True,
    color_mode='grayscale'
)

"""Step 2: Neural Network Architecture Design"""