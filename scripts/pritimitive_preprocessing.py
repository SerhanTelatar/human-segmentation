import os
import cv2
import numpy as np

# Define directory paths
primitive_data_dir = 'data/primitive_data/'
raw_data_dir = 'data/raw/'
image_size = (400, 400)  # Resize all faces to this size

# Create necessary directories if they don't exist
os.makedirs(raw_data_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect face and return it
def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        return image[y:y+h, x:x+w]  # Return the face as a region of interest (ROI)
    return None  # If no face is detected, return None

# Function to save the detected faces to the corresponding person's folder
def save_face_to_folder(face_image, person_folder):
    # Create folder for person if it doesn't exist
    person_path = os.path.join(raw_data_dir, person_folder)
    os.makedirs(person_path, exist_ok=True)

    # Save the face with a unique name
    face_filename = f'face_{len(os.listdir(person_path)) + 1}.jpg'
    face_image_resized = cv2.resize(face_image, image_size)
    cv2.imwrite(os.path.join(person_path, face_filename), face_image_resized)
    print(f'Saved face to {person_path}/{face_filename}')

# Process each folder from the primitive data directory
for folder in os.listdir(primitive_data_dir):
    folder_path = os.path.join(primitive_data_dir, folder)

    if os.path.isdir(folder_path):  # Only process directories
        print(f"Processing folder: {folder}")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            # Extract the face
            face = extract_face(image)

            if face is not None:
                save_face_to_folder(face, folder)  # Save to the same folder in `raw`
            else:
                print(f'No face detected in {image_name}')

print("Process completed.")
