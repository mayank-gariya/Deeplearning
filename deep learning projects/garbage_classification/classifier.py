import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras import layers 

def get_classification(image):
    number_classes = 6 

    resnet_input = tf.keras.Input(shape=(224, 224, 3), name='resnet_input')
    vgg_input = tf.keras.Input(shape=(224, 224, 3), name='vgg_input')

    resnet_prepped_output = layers.Lambda(resnet_preprocess, name='resnet_prepped')(resnet_input)
    vgg_prepped_output = layers.Lambda(vgg_preprocess, name='vgg_prepped')(vgg_input)

    base_resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_prepped_output)
    base_vgg = VGG16(weights='imagenet', include_top=False, input_tensor=vgg_prepped_output)

    base_resnet.trainable = False
    base_vgg.trainable = False

    resnet_features = base_resnet.output
    vgg_features = base_vgg.output

    resnet_features = GlobalAveragePooling2D()(resnet_features)
    vgg_features = GlobalAveragePooling2D()(vgg_features)

    combined_features = Concatenate()([resnet_features, vgg_features])
    combined_features = Dense(128, activation='relu')(combined_features)
    combined_features = layers.BatchNormalization()(combined_features) 
    combined_features = Dropout(0.5)(combined_features)
    output = Dense(number_classes, activation='softmax')(combined_features)

    model_loaded = Model(inputs=[resnet_input, vgg_input], outputs=output)

    model_loaded.load_weights('garbage_classifier.keras')

    img_path = image
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  


    predictions = model_loaded.predict({'resnet_input': img_array, 'vgg_input': img_array})
    return np.argmax(predictions)