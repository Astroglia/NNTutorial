import keras
import tensorflow
import numpy
import sys
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.convolutional import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

### TODO: update to tf.keras.layers for tensorflow 2.0 implementations.

############################################################################ UNET parameters & layers, guides, explanations: ###########################################################################

# Explanations
    #OG paper: https://arxiv.org/pdf/1505.04597.pdf
    #Intro explanation: https://towardsdatascience.com/u-net-b229b32b4a71
        #it has a lot of words.
    #Layer explanation: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
        #only use this if you're confused by what one of the layers does.

# Parameters

    #CONVOLUTIONAL LAYERS: 

        # -> {he_normal kernel initialization}: gradient variance stays the same in each filter of the layer you define. This could possibly be good if you want your filters
            # to have less dependence on the input (a possible normalizer), but who knows.
        # https://github.com/keras-team/keras/issues/52#issuecomment-93902209 is a nice summary.

        # -> { (3, 3) }: filter kernel size. Just look at this: https://qph.fs.quoracdn.net/main-qimg-84e2348d6dc2712ff2bb17262486232b
                # -> you can also modify the stride length to move every two pixels, etc. *however this affects filter outputs and you have to think more*

        # -> {padding='same'}: this is what is used most often, it's used in this UNET so you don't have to mess with the output shape of your filter too much.
    
    #DROPOUT: 
        # No idea why, but less dropout in the encoding block and more dropout in the decoding block resulted in higher accuracy for both my kidney scans and my retina images.
        # Usually you don't want to add dropout in the initial input filter because the first set of filters are *usually* generic/linear filters (like simple thresholding, gamma correction...)

    #MAX POOLING 2D:
        # since we want to condense our image and look at it from different size perspectives, we need to have a way to actually do this - max pooling 2d is essentially image resizing, but
        # for neural network filters.
        

## a generalized unet that allows any [ 2^n , 2^n ] image size input. adjusts filter scales accordingly (with the bottleneck filter @ the 2^n image resolution)
## Note that this implementation is using the default theano version, with channels being the first parameter.
    # TODO: allow any input dataset and just flip around axes if it doesn't match correctly.
def GEN_UNET(size, image_size, channels):
    
    inputs = Input((channels, image_size , image_size))

    div16 = int(size/16)#int(size/div_arr[3])
    div8  = int(size/8)#int(size/div_arr[2])
    div4  = int(size/4)#int(size/div_arr[1])
    div2  = int(size/2)#int(size/div_arr[0])


    ######################### ENCODING BLOCK

    conv1 = Conv2D(div16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #no dropout here!
    conv1 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(.2)(conv2)
    conv2 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(.2)(conv3)
    conv3 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Dropout(.2)(conv4)
    conv4 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    ######################### [END] ENCODING BLOCK
    ######################### BOTTLENECK

    conv5 = Conv2D(size, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Dropout(.3)(conv5)
    conv5 = Conv2D(size, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    ######################### [END] BOTTLENECK
    ######################### DECODING BLOCK

    up6 = concatenate([Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv5), conv4], axis=1)
    conv6 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Dropout(.3)(conv6)
    conv6 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(div2, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv6), conv3], axis=1)
    conv7 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Dropout(.3)(conv7)
    conv7 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(div4, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv7), conv2], axis=1)
    conv8 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Dropout(.3)(conv8)
    conv8 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(div8, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv8), conv1], axis=1)
    conv9 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Dropout(.3)(conv9)
    conv9 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)
    
    ######################### [END] DECODING BLOCK

    ##change to Conv2D(n, (1, 1) .............) for the amount of segments you want, where n = amount of segments you want as output.
    outputs = Conv2D(3, (1, 1), activation='sigmoid',kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model