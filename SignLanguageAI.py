import cv2
import mediapipe as mp
from sklearn.svm import SVC
from joblib import load
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV camera
cap = cv2.VideoCapture(1)  # Change the camera index if needed

# Load the trained SVM model from HandLanguageTraining.py
svm_clf = load('svm_model.joblib')

def extract_features(hand_landmarks):
    # Initialize a list to store extracted features
    features = []

    # Check if hand landmarks are detected and if the required landmarks are present
    if hand_landmarks and len(hand_landmarks.landmark) > mp_hands.HandLandmark.PINKY_TIP:
        # Calculate distances between specific landmarks (e.g., fingertips)
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Calculate angles between fingers (e.g., thumb-index, index-middle, etc.)
        thumb_index_angle = calculate_angle(thumb_tip, index_tip)
        index_middle_angle = calculate_angle(index_tip, middle_tip)
        middle_ring_angle = calculate_angle(middle_tip, ring_tip)
        ring_pinky_angle = calculate_angle(ring_tip, pinky_tip)

        # Add the angles as features
        features.extend([thumb_index_angle, index_middle_angle, middle_ring_angle, ring_pinky_angle])

    return np.array(features).reshape(1, -1) if features else None  # Return None if features are not extracted

def calculate_angle(point1, point2):
    # Calculate angle between two points (in radians)
    angle_radians = np.arctan2(point2.y - point1.y, point2.x - point1.x)
    return angle_radians

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to greyscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the greyscale image back to RGB
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features from hand landmarks within the bounding box
            features = extract_features(hand_landmarks)

            # Check if features are sufficient for prediction
            if features is not None:
                # Prepare landmarks for bounding box calculation
                landmarks_xy = np.array([(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in hand_landmarks.landmark], dtype=np.float32)

                # Draw bounding box around the hand
                bbox = cv2.boundingRect(landmarks_xy)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

                # Display the predicted label with decision function values on the frame
                decision_values = svm_clf.decision_function(features)
                predicted_label = svm_clf.predict(features)

                cv2.putText(frame, f'Predicted Label: {predicted_label} ', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
