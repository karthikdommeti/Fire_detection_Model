import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model
model.save("fire_detection_model.h5", save_format='h5')

# Image settings
IMG_SIZE = (128, 128)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized = cv2.resize(frame, IMG_SIZE)
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=0)

    # Make prediction
    prediction = model.predict(reshaped)[0][0]

    # Display label
    label = "🔥 Fire Detected!" if prediction > 0.5 else "✅ No Fire"
    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Live Fire Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
