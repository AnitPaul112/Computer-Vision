import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

def build_vgg16(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the VGG-16 model
vgg16_model = build_vgg16()

# Load the weights pre-trained on ImageNet
vgg16_model.load_weights(tf.keras.utils.get_file(
    'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    'https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'))


image_path = r"D:\CSE463 Homework 03\Ques 3.jpg"


def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    image_array = img_to_array(image)                    # Convert to array
    image_array = np.expand_dims(image_array, axis=0)    # Add batch dimension
    image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    return image_array

input_image = preprocess_image(image_path)


predictions = vgg16_model.predict(input_image)
decoded_predictions = tf.keras.applications.vgg16.decode_predictions(predictions, top=3)

print("Predictions:")
for prediction in decoded_predictions[0]:
    label = prediction[1]
    score = prediction[2] * 100  # Convert to percentage
    print(f"{label}: {score:.2f}%")


'''
VGG-16 processes input images resized to 224x224x3 for consistency.
Convolutional layers with small 3x3 filters extract features like edges and textures while minimizing parameters and computational cost.
ReLU activation introduces non-linearity, helping the model learn complex patterns. 
MaxPooling layers reduce feature map size, retaining key details and lowering computation. 
The Flatten layer converts 3D data into 1D, ready for the fully connected layers, which perform classification. 
VGG-16 uses small filters because they are efficient, capture better features, and enable a deeper network while maintaining flexibility and high accuracy, outperforming architectures with larger filters.
'''