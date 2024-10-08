import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Create a directory for storing images
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f'datasets/user.{count}.jpg', gray[y:y+h, x:x+w])

    cv2.imshow('Face Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:  # Capture 30 images
        break



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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = model.predict(gray[y:y+h, x:x+w])
        if confidence < 100:
            name = f'User {id_}'
        else:
            name = 'Unknown'

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
