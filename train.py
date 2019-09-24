from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model, to_categorical
from keras.utils import Sequence

from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model

import tensorflow as tf
import random as rn
import numpy as np
import keras

from PIL import Image
from skimage import transform

K.set_image_data_format('channels_last')

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np

def gen_arcface(generator):
    while True:
        data = generator.__next__()
        yield [data[0], data[1]], data[1]

if __name__ == '__main__':
    # Data Generator
    TRAIN_PATH = ''
    VAL_PATH = ''
    train_datagen = ImageDataGenerator(rescale=1./255.)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True
        )
    val_gen = train_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
        )

    print(train_gen.class_indices)

    STEP_SIZE_TRAIN=(train_gen.n//train_gen.batch_size)+1
    STEP_SIZE_VALID=(val_gen.n//val_gen.batch_size)+1

    # Fit
    #for layer in base_model.layers:
    #    layer.trainable = False
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
    history = model.fit_generator(generator=gen_arcface(train_gen),
                            validation_data=gen_arcface(val_gen),
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=20)