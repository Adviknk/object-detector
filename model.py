import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


model = tf.keras.applications.MobileNetV2(weights='imagenet')

image_path = '/test_image.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# Resize the image to match the input size of the model
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0)  # Add a batch dimension
image = tf.keras.applications.mobilenet.preprocess_input(
    image)  # Preprocess the image


predictions = model.predict(image)
decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(
    predictions, top=5)[0]


for _, label, confidence in decoded_predictions:
    print(f'{label}: {confidence * 100:.2f}%')

plt.imshow(image[0])
plt.axis('off')
plt.show()
