import tkinter as tk
from tkinter import Label, Frame
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageTk

# Load the model
model = tf.keras.models.load_model("emotion_detection.h5")

# Load the face detection Haar Cascade
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the window
window = tk.Tk()
window.title("Emotion Detection")
window.attributes('-fullscreen', True)  # Fullscreen mode
window.configure(bg='#121212')  # Dark theme background

# Function to exit fullscreen
window.bind("<Escape>", lambda event: window.quit())

# Header Label
header_label = Label(window, text="Emotion Detection", font=("Helvetica", 24, "bold"), fg="#FFFFFF", bg="#121212")
header_label.pack(pady=20)

# Frame for Video Feed
video_frame = Frame(window, bg="#000000", bd=5, relief="ridge")
video_frame.pack(pady=20)
video_label = Label(video_frame, bg="#000000")
video_label.pack()

# Initialize Video Capture (Global Variable)
cap = None

def update_frame():
    global cap
    if cap is None or not cap.isOpened():
        return  # Exit if cap is None or not opened
    
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    height, width, _ = img.shape

    # Detect faces and predict emotions
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

        predictions = model.predict(roi_gray)
        max_index = np.argmax(predictions[0])
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprised", "Neutral"]
        emotion_prediction = emotions[max_index]
        cv2.putText(img, emotion_prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Convert the frame to ImageTk format and update the label
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    video_label.img_tk = img_tk  # Keep reference to avoid garbage collection
    video_label.configure(image=img_tk)

    if cap.isOpened():
        window.after(10, update_frame)

def start_video():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
    update_frame()
    start_button.config(state="disabled")
    stop_button.config(state="normal")

def stop_video():
    global cap
    if cap and cap.isOpened():
        cap.release()
    cap = None
    video_label.configure(image='')  # Clear the label
    start_button.config(state="normal")
    stop_button.config(state="disabled")

def close_app():
    stop_video()
    window.quit()

# Button Styling
button_style = {"font": ("Helvetica", 14), "fg": "#FFFFFF", "bg": "#333333", "activebackground": "#555555", "bd": 2, "relief": "ridge", "width": 12}

# Buttons Frame
button_frame = Frame(window, bg="#121212")
button_frame.pack(pady=20)

# Start Button
start_button = tk.Button(button_frame, text="Start Video", command=start_video, **button_style)
start_button.grid(row=0, column=0, padx=10, pady=10)

# Stop Button
stop_button = tk.Button(button_frame, text="Stop Video", command=stop_video, state="disabled", **button_style)
stop_button.grid(row=0, column=1, padx=10, pady=10)

# Exit Button
exit_button = tk.Button(button_frame, text="Exit", command=close_app, **button_style)
exit_button.grid(row=0, column=2, padx=10, pady=10)

# Run the Tkinter event loop
window.mainloop()

# Cleanup
if cap and cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
