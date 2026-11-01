import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the pretrained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

print("Pretrained model loaded successfully!")

# Find all images inside the images folder
image_folder = "images"
image_files = os.listdir(image_folder)

print("Images found:", image_files)

# Go through each image
for image_file in image_files:

    # Only process image files
    if not image_file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    print()
    print("-----------------------------------")
    print("Image:", image_file)

    # Create the image path
    image_path = os.path.join(
        image_folder,
        image_file
    )

    # Load and resize the image
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    print("Image loaded successfully!")

    # Convert the image into a NumPy array
    img_array = image.img_to_array(img)

    # Add an extra dimension for the model
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    print("Image converted successfully!")

    # Preprocess the image for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    print("Image preprocessed successfully!")

    # Make a prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Convert prediction into a readable label
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=1
    )

    label = decoded_predictions[0][0][1]
    confidence = decoded_predictions[0][0][2] * 100

    print("Predicted label:", label)
    print("Confidence:", round(confidence, 2), "%")

print()
print("===================================")
print("Classification completed!")
print("===================================")