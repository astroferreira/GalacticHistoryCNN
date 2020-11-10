from tensorflow.keras.layers import Activation,Input, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Flatten, Dropout, Reshape, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam,RMSprop, Adamax
from tensorflow.keras.regularizers import *
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, RandomContrast
import tensorflow as tf
print(tf.__version__)

def base_model():

    input_img = Input(shape=(128, 128,4))
    #x = RandomContrast([0.1, 1])(input_img)
    
    x = RandomFlip()(input_img)
    x = Conv2D(16, 11)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 9)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 7)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 5)(x)    
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, 'relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='linear', name='class_dense')(x)

    model = Model(inputs=input_img, outputs=output)
    
    return model
