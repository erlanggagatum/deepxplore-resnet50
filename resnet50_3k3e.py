'''
ResNet-50 with custom classification head (transfer learning)
'''

from __future__ import print_function

import tensorflow as tf

from configs import bcolors


def ResNet50_3k3e(input_tensor=None, train=False):
    
    def compile_model(input_tensor):
        
        input = tf.keras.layers.UpSampling2D(size=(7,7))(input_tensor)
        
        # resnet layer feature extractor
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
    
    # compile model
    model = compile_model(input_tensor=input_tensor)
    
    # load weights
    model.load_weights('./resnet50_3k3e.h5')
    
    print(bcolors.OKBLUE + 'Model resnet50_3k3e loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    ResNet50_3k3e(train=True)
