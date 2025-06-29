import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('my_model_lstm.keras')

# Define emotion labels (must match the labels used in training)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        roi_gray = gray_frame[y:y+w, x:x+h]
        
        # Resize the face to the size expected by the model (48x48)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0  # Normalize the pixel values
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Expand dimensions to match model input
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add the channel dimension (1 for grayscale)

        # Predict the emotion
        prediction = model.predict(roi_gray)
        emotion_label = class_labels[np.argmax(prediction)]  # Get the emotion with the highest probability

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Put the predicted emotion label above the face
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with face and emotion detection
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

