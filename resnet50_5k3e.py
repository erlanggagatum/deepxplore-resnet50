'''
LeNet-1
'''

# usage: python MNISTModel1.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf

from configs import bcolors


def ResNet50_5k3e(input_tensor=None, train=False):

    # def feature_extractor(inputs):

    #     feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
    #                                                 include_top=False,
    #                                                 weights='imagenet')(inputs)
    #     return feature_extractor
    
    
    # def classifier(inputs):
    #     x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    #     x = tf.keras.layers.Flatten()(x)
    #     x = tf.keras.layers.Dense(1024, activation="relu")(x)
    #     x = tf.keras.layers.Dense(512, activation="relu")(x)
    #     x = tf.keras.layers.Dense(10, activation=None, name="before_classification")(x)
    #     x = tf.keras.layers.Activation(activation="softmax", name="classification")(x)
    #     return x
    
            
    # def final_model(inputs):

    #     resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    #     resnet_feature_extractor = feature_extractor(resize)
    #     classification_output = classifier(resnet_feature_extractor)

    #     return classification_output
    
            
    # def define_compile_model(inputs):
    #     # inputs = tf.keras.layers.Input(shape=(32,32,3))
        
    #     classification_output = final_model(inputs) 
    #     model = tf.keras.Model(inputs=inputs, outputs = classification_output)
        
    #     model.compile(optimizer='SGD', 
    #                     loss='sparse_categorical_crossentropy',
    #                     metrics = ['accuracy'])
        
    #     return model
    
    def compile_model(input_tensor):
        
        input = tf.keras.layers.UpSampling2D(size=(7,7))(input_tensor)
        
        # feature_extractor
        resnet50 = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                    include_top=False,
                                                    weights='imagenet')(input)
        
        # classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(resnet50)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(10, activation=None, name="before_classification")(x)
        x = tf.keras.layers.Activation(activation="softmax", name="classification")(x)
        
        model = tf.keras.Model(inputs=input_tensor, outputs = x)
        
        model.compile(optimizer='SGD', 
                        loss='sparse_categorical_crossentropy',
                        metrics = ['accuracy'])
        return model
    
    # model = define_compile_model(input_tensor)
    # model.load_weights('./resnet50_5k3e.h5')
    model = compile_model(input_tensor=input_tensor)
    model.load_weights('./resnet50_5k3e.h5')
    
    print(bcolors.OKBLUE + 'Model resnet50_5k3e loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    ResNet50_5k3e(train=True)
