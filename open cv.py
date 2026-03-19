import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model for logo classification
# Replace 'your_model_file_path.h5' with the actual file path of your trained model
model = load_model('logo_classification_model.h5')

# Define the size of the image for resizing
image_size = 224

def preprocess_image(image, image_size):
    # Preprocess the image (resize, normalization, etc.)
    resized_image = cv2.resize(image, (image_size, image_size))
    normalized_image = resized_image / 255.0  # Normalize pixel values to range [0, 1]
    return normalized_image

def predict_logo(image):
    # Preprocess the image
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

def main():
    # Access the camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        
        if ret:
            # Perform logo detection on the frame
            prediction = predict_logo(frame)
            
            # Display the result on the frame
            cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Camera Feed', frame)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
