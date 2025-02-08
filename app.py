import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the trained emotion detection model
model = tf.keras.models.load_model("emotion_model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize camera
camera = None

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        
        # Convert to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=0).reshape(-1, 48, 48, 1) / 255.0
            
            # Predict emotion
            predictions = model.predict(roi_gray)
            emotion = emotion_labels[np.argmax(predictions)]

            # Draw rectangle and put emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Video Stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to Start Camera
@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return "Camera Started"

# Route to Stop Camera
@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
    return "Camera Stopped"

if __name__ == "__main__":
    app.run(debug=True)
