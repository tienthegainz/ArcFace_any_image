from keras_efficientnets import EfficientNetB0, EfficientNetB3, EfficientNetB6
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Layer

from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model, to_categorical
from keras.utils import Sequence

K.set_image_data_format('channels_last')
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from arcface import *

def build_efficent_nets(n_classes, type_eff='b0', regulizer = None, target_size = (224, 224)):
    # Clear memory for new model
    K.clear_session()
    img_input = Input(shape=(target_size[0], target_size[1], 3))
    label = Input(shape=(n_classes,))

    if type_eff == 'b0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=img_input)
    elif type_eff == 'b3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=img_input)
    elif type_eff == 'b6':
        base_model = EfficientNetB6(weights='imagenet', include_top=False, input_tensor=img_input)
    # Custom top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = ArcFace(n_classes=11)([x, label])

    return Model([img_input, label], output)