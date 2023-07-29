import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template

app = Flask(__name__)

def train_model():
    # Set the paths to your dataset folders
    train_dir = 'training'
    valid_dir = 'validation'
    # Parameters for training
    batch_size = 32
    img_height, img_width = 224, 224
    num_epochs = 10
    # Load pre-trained MobileNetV2 model without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # Add custom top classification layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    # Freeze all layers in the base model to avoid overfitting
    for layer in base_model.layers:
        layer.trainable = False
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # Rescale validation data
    valid_datagen = ImageDataGenerator(rescale=1./255)
    # Load the training and validation data
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
    valid_data = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
    # Train the model
    model.fit(
        train_data,
        steps_per_epoch=train_data.samples // batch_size,
        validation_data=valid_data,
        validation_steps=valid_data.samples // batch_size,
        epochs=num_epochs
    )
    # Save the model
    model.save('wrinkle_classifier_model.h5')

# Train the model if it doesn't exist already
if not os.path.isfile("wrinkle_classifier_model.h5"):
    train_model()


# Load the trained model from the saved file
model = load_model('wrinkle_classifier_model.h5')

# Parameters for image preprocessing and classification
img_height, img_width = 224, 224

# Function to preprocess and classify the uploaded image
def classify_wrinkles(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_height, img_width))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    # Predict the image (Wrinkle or Non-wrinkle)
    prediction = model.predict(img)
    if prediction[0][0] >= 0.5:
        return "Wrinkle"
    else:
        return "Non-wrinkle"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image selected!"
        image = request.files['image']
        if image.filename == '':
            return "No image selected!"
        result = classify_wrinkles(image)
        return result  # Return the classification result as plain text
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)