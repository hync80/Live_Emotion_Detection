import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Load FER-2013 images
def load_video_data(base_path='data/ferr'):
    emotion_labels = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        for emotion, label in emotion_labels.items():
            emotion_path = os.path.join(split_path, emotion)
            if os.path.exists(emotion_path):  # Agar folder hai
                for img_file in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:  # Image load hui ya nahi
                        img = cv2.resize(img, (48, 48)) / 255.0  # 48x48 resize aur normalize
                        img = img.reshape(48, 48, 1)
                        if split == 'train':
                            X_train.append(img)
                            y_train.append(label)
                        else:
                            X_test.append(img)
                            y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train), num_classes=7)
    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test), num_classes=7)
    print("Training data:", X_train.shape, "Test data:", X_test.shape)
    return X_train, y_train, X_test, y_test

# Build Video Model
def build_video_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    model.save('video_emotion_model.h5')
    print("Video model trained and saved!")

# Test Model
def test_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Real-Time Detection
def detect_emotions():
    model = tf.keras.models.load_model('video_emotion_model.h5')
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam not working!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = roi.reshape(1, 48, 48, 1)
            pred = model.predict(roi)
            label = emotions[np.argmax(pred)]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_video_data()
    model = build_video_model()
    train_model(model, X_train, y_train)
    test_model(model, X_test, y_test)
    detect_emotions()