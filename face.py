from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255)

train = train_data.flow_from_directory(
    "train",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

test_data = ImageDataGenerator(rescale=1./255)

test = test_data.flow_from_directory(
    "test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train,
    validation_data=test,
    epochs=10
)
model.save("emotion_model.h5")
print("Model Saved Successfully")