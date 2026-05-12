import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Fixed: Full path to your model file
model = load_model(r'C:\Users\A HAJIBU\OneDrive\Desktop\online fake logo detection\logo_classification_model.h5')

# Define the size of the image for resizing
image_size = 224


def preprocess_image(image, image_size):
    resized_image = cv2.resize(image, (image_size, image_size))
    normalized_image = resized_image / 255.0
    return normalized_image


def predict_logo(image):
    processed_image = preprocess_image(image, image_size)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)

    if prediction[0][0] > 0.5:
        return 'Genuine'
    else:
        return 'Fake'


def main():
    cap = cv2.VideoCapture(0)

    # Added: Check if camera opened successfully
    if not cap.isOpened():
        print("ERROR: Camera could not be opened!")
        return

    print("Camera opened successfully! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if ret:
            prediction = predict_logo(frame)

            # Display result on frame
            color = (0, 255, 0) if prediction == 'Genuine' else (0, 0, 255)  # Green=Genuine, Red=Fake
            cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Logo Detection - Camera Feed', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
