import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array


"""Step 5: Model Testing"""

# Load the trained models from saved files
model_1   = keras.models.load_model("model.keras") # Load Model 1
model_2 = keras.models.load_model("model2.keras")  # Load Model 2

# Define the class label
class_names = ["crack", "missing-head", "paint-off"]

# Paths to sample test images for each defect class
img_paths = {
    "paint-off":   "./Project 2 Data/test/paint-off/test_paintoff.jpg",
    "missing-head":"./Project 2 Data/test/missing-head/test_missinghead.jpg",
    "crack":       "./Project 2 Data/test/crack/test_crack.jpg",
}

# Define Image Shape
image_shape = (500, 500)

# Function to load and preprocess an image for model prediction
def load_and_prep(img_path):
    # Load the image and resize it
    img = load_img(img_path, target_size=image_shape, color_mode="grayscale")
    
    # Convert the image to array and normalize to [0,1]
    img = img_to_array(img) / 255.0
    
    return np.expand_dims(img, axis=0)

# Function to generate prediction probabilities for a single image
def predict(model, img_path):
    
    # Pass preprocessed image into the model and return prediction array
    return model.predict(load_and_prep(img_path))[0]

# Function to visualize predictions for all three classes
def show_row(model, row, title):
    for i, cls in enumerate(["paint-off", "missing-head", "crack"]):
        
        # Create subplot position
        ax = plt.subplot(2, 3, row*3 + i + 1)
        
        # Load the test image in grayscale
        img = load_img(img_paths[cls], color_mode="grayscale")
        ax.imshow(img, cmap='gray')
        ax.axis('off') # Remove axes for cleaner display
        
        # Get prediction probabilities
        probs = predict(model, img_paths[cls])
        
        # Determine predicted class (highest probability)
        pred = class_names[np.argmax(probs)]
        
        # Display predication info on image
        info = "\n".join([
            f"{name}: {p*100:.2f}%" for name, p in zip(class_names, probs)
        ] + [f"Predicted: {pred}", f"True: {cls}"])
        
        #Display text box on iamge with results
        ax.text(0.02, 0.02, info, transform=ax.transAxes,
        va='bottom', ha='left',
        fontsize=20, color='black',
        bbox=dict(boxstyle="round,pad=0.4", fc=(0, 1, 0, 0.3), ec='none'))

#Plot predictions for both models
plt.figure(figsize=(15, 9))
show_row(model_1, 0, "Model 1") # Model 1
show_row(model_2,   1, "Model 2") # Model 2
plt.tight_layout()
plt.show()
