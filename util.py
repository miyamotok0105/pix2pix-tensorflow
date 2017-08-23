# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import imread, imresize, imsave

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
    
def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = imresize(img, (256, 256))
    # img = np.transpose(img, (2, 0, 1))
    return img

