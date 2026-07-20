import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras import layers 
import os
import urllib.request

MODEL_PATH = 'garbage_classifier.keras'
MODEL_URL = 'https://drive.google.com/uc?export=download&id=11hh8UaKZERZwbMUIcP7ntIeWdcTTt41o'

def load_garbage_model():
    """Builds and loads weights into the model ONCE at startup."""

    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights... this might take a moment.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")
        
    number_classes = 6 

    resnet_input = tf.keras.Input(shape=(224, 224, 3), name='resnet_input')
    vgg_input = tf.keras.Input(shape=(224, 224, 3), name='vgg_input')

    resnet_prepped_output = layers.Lambda(resnet_preprocess, name='resnet_prepped')(resnet_input)
    vgg_prepped_output = layers.Lambda(vgg_preprocess, name='vgg_prepped')(vgg_input)

    base_resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_prepped_output)
    base_vgg = VGG16(weights='imagenet', include_top=False, input_tensor=vgg_prepped_output)

    base_resnet.trainable = False
    base_vgg.trainable = False

    resnet_features = GlobalAveragePooling2D()(base_resnet.output)
    vgg_features = GlobalAveragePooling2D()(base_vgg.output)

    combined_features = Concatenate()([resnet_features, vgg_features])
    combined_features = Dense(128, activation='relu')(combined_features)
    combined_features = layers.BatchNormalization()(combined_features) 
    combined_features = Dropout(0.5)(combined_features)
    output = Dense(number_classes, activation='softmax')(combined_features)

    model = Model(inputs=[resnet_input, vgg_input], outputs=output)
    model.load_weights('garbage_classifier.keras')
    return model

model_loaded = load_garbage_model()

def get_classification(img_bytes):
    """Takes image bytes, processes them, and returns the class index."""
    # Convert uploaded raw bytes directly into a TensorFlow tensor
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [224, 224])
    img_array = tf.expand_dims(img, 0)  

    predictions = model_loaded.predict({'resnet_input': img_array, 'vgg_input': img_array})
    return int(np.argmax(predictions)) # Cast to standard python int for JSON serialization
