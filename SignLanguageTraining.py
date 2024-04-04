import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import math
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Folder containing the data
data_folder = 'training_data'
image_folder = os.path.join(data_folder, 'images')
label_folder = os.path.join(data_folder, 'labels')

# Lists to store features and labels
features = []
labels = []
errors = []

# Load images and labels
for image_file in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
    # Load images and convert to RGB
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load labels
    image_number = int(image_file.split('_')[1].split('.')[0])
    label_number = ((image_number - 1) // 100) + 1
    label_file = f'label_{label_number}.txt'
    
    label_path = os.path.join(label_folder, label_file)
    with open(label_path, 'r') as f:
        label = f.read().strip()
    
    # Print the image file name and its corresponding label
    print(label, image_file)

    # Process the image with MediaPipe Hands to get hand landmarks
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Prepare landmarks for bounding box calculation
            landmarks_xy = np.array([(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in hand_landmarks.landmark], dtype=np.float32)

            # Calculate bounding box coordinates
            bbox = cv2.boundingRect(landmarks_xy)

            # Crop the image within the bounding box
            cropped_image = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

            # Check if the cropped image is not empty or invalid
            if cropped_image is not None and cropped_image.size > 0:
                # Convert the cropped image to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Resize the grayscale image to desired dimensions
                resized_image = cv2.resize(gray_image, (224, 224))

                # Calculate finger distances and add to features
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                thumb_index_dist = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                index_middle_dist = math.sqrt((index.x - middle.x)**2 + (index.y - middle.y)**2)
                middle_ring_dist = math.sqrt((middle.x - ring.x)**2 + (middle.y - ring.y)**2)
                ring_pinky_dist = math.sqrt((ring.x - pinky.x)**2 + (ring.y - pinky.y)**2)

                features.append([thumb_index_dist, index_middle_dist, middle_ring_dist, ring_pinky_dist])

                # Add label to the list
                labels.append(label)
            else:
                errors.append(f"{image_file}")

# Convert lists to NumPy arrays
X_data = np.array(features)
y_labels = np.array(labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Print the unique labels used for training and testing
print(f"Unique Labels for Training: {len(np.unique(y_train))}", np.unique(y_train))
print(f"Unique Labels for Testing: {len(np.unique(y_test))}", np.unique(y_test))

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# Initialize the SVM classifier
svm_clf = SVC()

# Perform grid search to find the best parameters
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its parameters
best_svm_clf = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the accuracy for each combination of kernel and C parameter
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print(f"Accuracy: {mean:.3f} (+/- {std * 2:.3f}) for {params}")

# Train the best model on the entire training data
best_svm_clf.fit(X_train_scaled, y_train)

# Evaluate the best model
y_pred = best_svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nBest Kernel:", best_params['kernel'])
print("Best C Parameter:", best_params['C'])
print("Model Accuracy:", round(accuracy, 2), "%")

# Input for choosing kernel and C parameter
selected_kernel = input("Enter the kernel (linear, poly, rbf, sigmoid): ")
selected_C = float(input("Enter the C parameter value: "))

# Initialize the SVM classifier with selected kernel and C parameter
svm_clf = SVC(kernel=selected_kernel, C=selected_C)

# Train the model on the training data
svm_clf.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nSelected Kernel:", selected_kernel)
print("Selected C Parameter:", selected_C)
print("Model Accuracy:", round(accuracy, 2), "%")

# Handle errors
if errors:
    print("Errors encountered for the following images:")
    for error in errors:
        print(error)

# Save the best trained model
# dump(best_svm_clf, 'svm_model.joblib')
dump(svm_clf, 'svm_model.joblib')
