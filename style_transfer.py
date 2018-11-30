from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

from keras.applications import vgg19
from keras import backend as K


if __name__ == "__main__":
    print("hello world")