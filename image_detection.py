import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog
import tkinter as tk


model = load_model("emotion_model.h5")


emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Error: Haarcascade file not found")
    exit()

root = tk.Tk()
root.withdraw()


image_path = filedialog.askopenfilename(
    title="Select Face Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)


if image_path == "":
    print("No image selected")
    exit()


image = cv2.imread(image_path)


if image is None:
    print("Error: Unable to read image")
    exit()


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)


if len(faces) == 0:
    print("No face detected")
    

for (x, y, w, h) in faces:

    face = gray[y:y+h, x:x+w]


    face = cv2.resize(
        face,
        (48, 48)
    )


    face = face / 255.0


    face = np.reshape(
        face,
        (1, 48, 48, 1)
    )


    prediction = model.predict(face, verbose=0)


    index = np.argmax(prediction)

    emotion = emotions[index]

    confidence = np.max(prediction) * 100


    cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )


    cv2.putText(
        image,
        f"{emotion} {confidence:.1f}%",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


cv2.imshow(
    "Emotion Detection",
    image
)


cv2.waitKey(0)

cv2.destroyAllWindows()