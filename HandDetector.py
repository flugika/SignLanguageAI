import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV camera
cap = cv2.VideoCapture(1)

# Create a folder to store captured images and labels
data_folder = 'training_data'
image_folder = os.path.join(data_folder, 'images')
label_folder = os.path.join(data_folder, 'labels')
os.makedirs(image_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

# Count the number of files in the image folder to adjust image_count
lenImage = len(os.listdir(image_folder))
image_count = 0

# Flag to indicate if we should capture an image
capture_image = False

while cap.isOpened() and image_count < 100:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks.append((x, y))

            # Calculate bounding box coordinates
            x_min = min(landmarks, key=lambda x: x[0])[0]
            x_max = max(landmarks, key=lambda x: x[0])[0]
            y_min = min(landmarks, key=lambda x: x[1])[1]
            y_max = max(landmarks, key=lambda x: x[1])[1]

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Check if the space bar is pressed to capture an image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        capture_image = True

    # Check if we should capture an image
    if capture_image:
        # Save the captured image
        image_count += 1
        image_path = os.path.join(image_folder, f'Image_{image_count + lenImage}.jpg')
        cv2.imwrite(image_path, frame)
        print(f'Image captured and saved at {image_path}')

        if image_count == 100:
            # Prompt user to input label for the captured image
            label = input(f"Enter the label for image {(image_count + lenImage - 99)}-{image_count + lenImage}: ")
            # Save the label in a text file
            label_path = os.path.join(label_folder, f'label_{(image_count + lenImage)//100}.txt')
            labels_path = os.path.join(label_folder, f'labels.txt')
            with open(label_path, 'w') as f:
                f.write(label)
            with open(labels_path, 'a') as f:
                f.write(f'\n{label}')


    # Break the loop when 'q' key is pressed or when capturing 100 images
    if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= 100:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
