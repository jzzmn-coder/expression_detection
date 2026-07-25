import cv2
import numpy as np
from tensorflow.keras.models import load_model


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


camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )


    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (48,48)
        )

        face = face / 255.0

        face = np.reshape(
            face,
            (1,48,48,1)
        )


        prediction = model.predict(face)

        index = np.argmax(prediction)

        emotion = emotions[index]

        confidence = np.max(prediction)*100


        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            f"{emotion} {confidence:.1f}%",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


    cv2.imshow(
        "Live Emotion Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



camera.release()
cv2.destroyAllWindows()