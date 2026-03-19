import os
import cv2
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from tensorflow.keras.models import load_model

logo_folder = 'output'
output = 'genLogoOutput'
scale_factor = 0.8
rotation_angle = 30
image_size = 224  # Define the size of the image for resizing

# Load the trained model for logo classification
# Replace 'your_model_file_path.h5' with the actual file path of your trained model
model = load_model('logo_classification_model.h5')

def preprocess_image(image, image_size):
    # Preprocess the image (resize, normalization, etc.)
    resized_image = cv2.resize(image, (image_size, image_size))
    normalized_image = resized_image / 255.0  # Normalize pixel values to range [0, 1]
    return normalized_image

def predict_logo(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image, image_size)
    
    # Reshape the image to match the input shape expected by the model
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Perform prediction using the trained model
    prediction = model.predict(processed_image)
    
    # Determine whether the logo is genuine or fake based on the prediction
    if prediction[0][0] > 0.5:
        return 'Genuine'
    else:
        return 'Fake'

# Allow the user to select an image using a file dialog window
def select_image():
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename()  # Show an "Open" dialog box and return the path to the selected file
    return image_path

def main():
    # Select an image using a file dialog window
    image_path = select_image()
    
    if image_path:
        # Predict whether the logo in the selected image is genuine or fake
        prediction = predict_logo(image_path)
        
        # Display the prediction result
        print(f"The predicted label for the selected logo is: {prediction}")

if __name__ == "__main__":
    main()
