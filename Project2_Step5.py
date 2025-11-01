import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

model_1   = keras.models.load_model("modelx.keras")
model_2 = keras.models.load_model("model2x.keras")

class_names = ["crack", "missing-head", "paint-off"]

img_paths = {
    "paint-off":   "./Project 2 Data/test/paint-off/test_paintoff.jpg",
    "missing-head":"./Project 2 Data/test/missing-head/test_missinghead.jpg",
    "crack":       "./Project 2 Data/test/crack/test_crack.jpg",
}

image_shape = (500, 500)

def load_and_prep(img_path):
    img = load_img(img_path, target_size=image_shape, color_mode="grayscale")
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img_path):
    return model.predict(load_and_prep(img_path))[0]

def show_row(model, row, title):
    for i, cls in enumerate(["paint-off", "missing-head", "crack"]):
        ax = plt.subplot(2, 3, row*3 + i + 1)
        img = load_img(img_paths[cls], color_mode="grayscale")
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{title} — {cls}", fontsize=20,fontweight='bold')
        ax.axis('off')

        probs = predict(model, img_paths[cls])
        pred = class_names[np.argmax(probs)]

        info = "\n".join([
            f"{name}: {p*100:.2f}%" for name, p in zip(class_names, probs)
        ] + [f"Predicted: {pred}", f"True: {cls}"])

        ax.text(0.02, 0.02, info, transform=ax.transAxes,
        va='bottom', ha='left',
        fontsize=20, color='black',
        bbox=dict(boxstyle="round,pad=0.4", fc=(0, 1, 0, 0.3), ec='none'))

plt.figure(figsize=(15, 9))
show_row(model_1, 0, "Model 1")
show_row(model_2,   1, "Model 2")
plt.tight_layout()
plt.show()
