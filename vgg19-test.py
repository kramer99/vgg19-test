# -*- coding: utf-8 -*-

import scipy.io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Flatten

def _weights(layer, expected_layer_name):
    """
    Return the weights and bias from the VGG model for a given layer.
    Credit: deeplearning.ai
    """
    wb = vgg_layers[0][layer][0][0][2]
    W = wb[0][0]
    b = wb[0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b

def _set_weights_on_layer(keras_layer, vgg_layer):
    W, b = _weights(vgg_layer, model.layers[keras_layer].name)
    b = np.reshape(b, (b.size))
    model.layers[keras_layer].set_weights([W, b])

def _set_weights_on_dense_layer(keras_layer, vgg_layer):
    W, b = _weights(vgg_layer, model.layers[keras_layer].name)
    b = np.reshape(b, (b.size))
    W = np.reshape(W, (-1, W.shape[-1]))     # squish from 4 dimensions to 2
    model.layers[keras_layer].set_weights([W, b])


vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
vgg_layers = vgg['layers']

img = imageio.imread("elephant.jpg")
plt.imshow(img)
plt.show()
img = np.reshape(img, ((1,) + img.shape))   # first layer expects 4D input

model = Sequential([
    #Convolution2D(64, (3, 3), name='conv1_1', padding='same', activation='relu', input_shape=(300,400,3)),
    Convolution2D(64, (3, 3), name='conv1_1', padding='same', activation='relu', input_shape=(224,224,3)),
    Convolution2D(64, (3, 3), name='conv1_2', padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same'),
    Convolution2D(128, (3, 3), name='conv2_1', padding='same', activation='relu'),
    Convolution2D(128, (3, 3), name='conv2_2', padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same'),
    Convolution2D(256, (3, 3), name='conv3_1', padding='same', activation='relu'),
    Convolution2D(256, (3, 3), name='conv3_2', padding='same', activation='relu'),
    Convolution2D(256, (3, 3), name='conv3_3', padding='same', activation='relu'),
    Convolution2D(256, (3, 3), name='conv3_4', padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3', padding='same'),
    Convolution2D(512, (3, 3), name='conv4_1', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv4_2', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv4_3', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv4_4', padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4', padding='same'),
    Convolution2D(512, (3, 3), name='conv5_1', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv5_2', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv5_3', padding='same', activation='relu'),
    Convolution2D(512, (3, 3), name='conv5_4', padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5', padding='same'),
    #GlobalAveragePooling2D(name='avgpool5'),
    Flatten(),
    Dense(4096, name='fc6', activation='relu'),
    Dense(4096, name='fc7', activation='relu'),
    Dense(1000, name='fc8', activation='softmax'),
])

_set_weights_on_layer(0, 0)
_set_weights_on_layer(1, 2)
_set_weights_on_layer(3, 5)
_set_weights_on_layer(4, 7)
_set_weights_on_layer(6, 10)
_set_weights_on_layer(7, 12)
_set_weights_on_layer(8, 14)
_set_weights_on_layer(9, 16)
_set_weights_on_layer(11, 19)
_set_weights_on_layer(12, 21)
_set_weights_on_layer(13, 23)
_set_weights_on_layer(14, 25)
_set_weights_on_layer(16, 28)
_set_weights_on_layer(17, 30)
_set_weights_on_layer(18, 32)
_set_weights_on_layer(19, 34)

_set_weights_on_dense_layer(22, 37)
_set_weights_on_dense_layer(23, 39)
_set_weights_on_dense_layer(24, 41)

#model.summary()

class_predictions = model.predict_proba(img)

# bar graph of the prediction intensities
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 6
plt.bar(np.arange(len(class_predictions[0])), class_predictions[0], width=1.5)
plt.show()

top_five = np.argpartition(-class_predictions[0], 5)
print('Top five highest likely predictions:')
class_names = vgg['meta'][0][0][1][0][0][1][0]      # no idea why they are embedded so deeply
for i in top_five[:5]:
    print(i, ': ', class_names[i][0], '(', class_predictions[0][i], ')')



