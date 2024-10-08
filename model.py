import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load images and labels
faces = []
labels = []

for filename in os.listdir('datasets'):
    img_path = os.path.join('datasets', filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    label = int(filename.split('.')[1])  # Extract label from filename
    labels.append(label)

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))