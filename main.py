import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #
import numpy as np
import time
import tensorflow as tf



#Figure out if you have GPU's available, and run on gpus if they are available.
print("#######################")
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
#print(" ################### :::  ", tf.test.is_gpu_available() )

######################################## DATA LOADING ########################################
##### TODO: data loading functions for something to actually run through
##### For now, just make some dummy datasets

data = np.random.rand( 30,  15 , 256 , 256  ) #30 datasets, 15 images each, resolution==256x256
data_test = np.random.rand( 10,  15 , 256 , 256  ) #Test dataset. I like large testing sets (30 to 50% the size of my training set),
                                          # but it can change based off how generalized you need your network to be.

labels = np.random.rand( 30,  15 , 256 , 256  ) #we want segmentations, so the output will be equal to the input.
labels_test = np.random.rand( 10,  15 , 256 , 256  )        

#NOTE: sklearn.train_test_split can be used instead if you have all data in one big dataset, though some precautions have to be considered: 
# you have to be careful for the network, though - if you have an MRI images with 60 images, you want all 60 images to be in either training or testing, 
# not some images in training, and some images in testing. Therefore, run train_test_split on something like:
    # split_base_images = [ 100, 60, 256 , 256 ], where 100 = the amount of MRI images
        # After train_test_split: 
            # train = [ 80, 60 , 256 , 256 ]
            # test =  [ 20, 60,  256 , 256 ]
#This issue also arises with time series data in the same sense (signal input from the same "run" should not be in both the training and testing set.)

######################################## DATA NETWORK CONFIGURATION ########################################
#normalize data to range between zero and one. zero and one are typically used as limiters due to gradients tending to explode if you
#normalize to larger ranges 
import data_modification as DMOD
data = DMOD.normalize_numpy_array(data)
data_test = DMOD.normalize_numpy_array(data_test)
labels = DMOD.normalize_numpy_array(labels)
labels_test = DMOD.normalize_numpy_array(labels_test)

# For slice by slice training, chuck all images to one dimension ( [80, 60, 256, 256] ) --> ( [80*60, 256, 256] ) 
# If using stack by stack, you'd have to use Conv3d, which I have never used. It probably uses a lot of memory.

data = np.reshape( data, [ int(data.shape[0]*data.shape[1]), data.shape[2], data.shape[3] ]) #generalized form
data_test = np.reshape( data_test, [ int(10*15), 256, 256  ] ) #hardcoded form
labels = np.reshape( labels, [ int(30*15), 256, 256  ] ) #hardcoded form
labels_test = np.reshape( labels_test, [ int(10*15), 256, 256  ] ) #hardcoded form


#Add dimension for tensor type of input. I do not know the reason behind needing an extra dimension for network input.
#one important aspect for keras/tensorflow: by default, inputs to things such as a 2d convolutional network for keras are channels last:
                    #  (batch, rows, cols, channels)
        # note that batch is the amount of images the neural network trains on before doing a backward pass to compute gradients:
          #  https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network


data = np.expand_dims(data , axis=3)
data_test = np.expand_dims(data_test, axis=3)
labels = np.expand_dims(labels, axis=3)
labels_test =np.expand_dims(labels_test, axis=3)

#convert all data to floats - networks typically take float32 as inputs (int8/float16 can be used for quantized networks on embedded devices).
#using float instead of np.float can also speed things up, but may affect accuracy.
data = data.astype(np.float)
data_test = data_test.astype(np.float)
labels = labels.astype(np.float)
labels_test = labels_test.astype(np.float)

print(data.shape)
print(data_test.shape)
print(labels.shape)
print(labels_test.shape)

### SHOULD OUTPUT:
#(450, 256, 256, 1)
#(150, 256, 256, 1)
#(450, 256, 256, 1)
#(150, 256, 256, 1)

############################################ NETWORK TRAINING ########################################
