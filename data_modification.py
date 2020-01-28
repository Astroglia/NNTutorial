## works on any dimensional image.
import numpy as np


#normalizes an n dimensional numpy image to a range between zero and one.
def normalize_numpy_array(numpy_img, cap_outliers=False):
    if(cap_outliers):
        #TODO: option to do mean based off of row/column - may be useful in certain segmentation types.
        mean = np.mean(numpy_img.flatten()) #Get mean/sdev
        sdev = np.std(numpy_img.flatten())

        thresh = mean + sdev + sdev
        numpy_img_threshold = numpy_img > thresh #Exclude large values > 2 sdev
        numpy_img[numpy_img_threshold] = thresh
    
    numpy_img = (numpy_img-np.min(numpy_img))/(np.max(numpy_img)-np.min(numpy_img))    
    return numpy_img