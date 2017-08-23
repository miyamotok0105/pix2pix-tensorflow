# -*- coding: utf-8 -*-
import math
import json
import random
# import scipy.misc
import numpy as np
from time import gmtime, strftime
from os import listdir
from os.path import join
from util import is_image_file, load_img

class DatasetFromFolder():
    def __init__(self, image_dir):
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.a_path, self.image_filenames[index]))
        target = load_img(join(self.b_path, self.image_filenames[index]))
        return input, target

    def __len__(self):
        return len(self.image_filenames)

