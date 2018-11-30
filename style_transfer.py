from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

from keras.applications import vgg19
from keras import backend as K



# def preprocess_image(image_path):
#     load_img()
#     img = load_img(image_path, target_size=(img_nrows, img_ncols))
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = vgg19.preprocess_input(img)
#     return img



if __name__ == "__main__":
    print("hello world")
    img = load_img("data/style/")
    print(img)
    print(img.size)

