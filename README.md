#Project Title:
#Aircraft Surface Defect Classification Using Deep Convolutional Neural Networks

Overview:
This project focuses on automating aircraft surface inspection using deep learning.
A custom Deep Convolutional Neural Network (DCNN) was developed to classify aircraft
skin defects into three categories: cracks, missing screw heads, and paint degradation.

The goal is to improve inspection efficiency and accuracy compared to traditional
manual inspection methods commonly used in aerospace maintenance.

Dataset:
- Image-based dataset with three defect classes:
  1. Crack
  2. Missing-head
  3. Paint-off
- Data split into training, validation, and test sets
- Input image size standardized to (500 x 500 x 3)

Methodology:
1. Data Processing
   - Image normalization and resizing
   - Data augmentation (scaling, zoom, shear) applied to training data
   - Keras image generators used for efficient data loading

2. Neural Network Architecture
   - Multiple Conv2D layers for feature extraction
   - MaxPooling layers for dimensionality reduction
   - Fully connected Dense layers with Dropout for classification
   - Softmax output layer for multi-class prediction

3. Hyperparameter Tuning
   - Explored different activation functions (ReLU, LeakyReLU, ELU)
   - Tuned number of filters, neurons, optimizers, and loss functions
   - Optimized network depth to balance accuracy and overfitting

4. Model Evaluation
   - Tracked training and validation accuracy and loss
   - Assessed overfitting using performance curves

5. Model Testing
   - Evaluated model on unseen test images
   - Predicted defect class using softmax probability outputs

Technologies Used:
- Python
- TensorFlow / Keras
- NumPy, Pandas
- ImageDataGenerator

Key Outcomes:
- Achieved reliable classification across all defect categories
- Demonstrated CNN effectiveness for aerospace visual inspection
- Identified edge cases and discussed performance limitations

Applications:
- Automated aircraft inspection
- Structural health monitoring
- Aerospace quality assurance systems

