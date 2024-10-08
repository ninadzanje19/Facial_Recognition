import face_recognition
import os
from face_recognition.face_detection_cli import test_image
from variables import *

def detect_face(train_set_address, test_image_address):
    # Directory containing known faces
    known_faces_dir = train_set_address
    known_face_encodings = []
    known_face_names = []

    # Load and encode known faces
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]

            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name

    # Load the input image for recognition
    input_image_path = test_image_address  # Replace with your image path
    input_image = face_recognition.load_image_file(input_image_path)
    input_encoding = face_recognition.face_encodings(input_image)

    if not input_encoding:
        print("No faces found in the input image.")
    else:
        input_encoding = input_encoding[0]  # Get the first encoding

        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, input_encoding)

        # Determine name based on matches
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        print(f"Recognized: {name}")
        return f"Recognized: {name}"


detect_face(train_data_dir, r"G:\Projects\Facial_Recognition\Facial_Recognition\data\test\Rohit_Sharma_Test.jpg")
detect_face(train_data_dir, r"G:/Projects/Facial_Recognition/Facial_Recognition/data/test/Sachin_Tendulkar_Test.jpg")
detect_face(train_data_dir, r"G:\Projects\Facial_Recognition\Facial_Recognition\data\test\Zaheer_Khan_Test.jpg")
