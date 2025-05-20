import cv2
import numpy as np
import os
import tensorflow as tf

# Load and preprocess images
def load_images(folder):
    data = []
    labels = []

    for i in range(10):
        path = os.path.join(folder, f"{i}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Image {i}.png not found")
            continue

        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        data.append(img)
        labels.append(i)

    return np.array(data), np.array(labels)

# Load images
x, y = load_images('my_digits') # Images for training stored in my_digits folder
x = x.reshape(-1, 28, 28)

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=20)

# Save the model
model.save('model.keras')
print("Model trained and saved as model.keras")
