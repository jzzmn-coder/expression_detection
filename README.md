😊 Face Emotion Detection using Deep Learning

A Deep Learning-based Facial Emotion Recognition system that detects human emotions from both **static images** and **live webcam video**. The project uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras for emotion classification and **OpenCV Haar Cascade** for face detection.

## 🚀 Features

* Detect emotions from uploaded images
* Real-time emotion recognition using a webcam
* CNN model trained on facial expression datasets
* Automatic face detection with OpenCV
* Supports multiple facial expressions:

  * 😠 Angry
  * 🤢 Disgust
  * 😨 Fear
  * 😀 Happy
  * 😐 Neutral
  * 😢 Sad
  * 😲 Surprise

 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Tkinter
* Haar Cascade Classifier

 📂 Project Structure

```text
Face-Emotion-Detection/
│── image_detection.py        # Train the CNN model
│── live_camera.py            # Real-time emotion detection
│── face.py                   # Detect emotion from an image
│── emotion_model.h5          # Trained model
│── haarcascade_frontalface_default.xml
│── train/                    # Training dataset
│── test/                     # Testing dataset
│── requirements.txt
│── README.md
```

 ⚙️ Installation

```bash
git clone https://github.com/your-username/Face-Emotion-Detection.git
cd Face-Emotion-Detection

pip install -r requirements.txt
```

 ▶️ Usage

 Train the Model

```bash
python image_detection.py
```

 Detect Emotion from an Image

```bash
python face.py
```

 Live Webcam Detection

```bash
python live_camera.py
```

 🎯 How It Works

1. Detects faces using OpenCV Haar Cascade.
2. Preprocesses the detected face image.
3. Passes the image to the trained CNN model.
4. Predicts one of the seven emotion classes.
5. Displays the predicted emotion on the image or live video feed.

 📌 Future Improvements

* Improve model accuracy with transfer learning
* Support multiple face tracking
* Add confidence score for predictions
* Deploy as a Flask web application
* Mobile-friendly implementation



