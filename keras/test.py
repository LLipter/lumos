from model import loss_net
from util import get_file_paths, preprocess_image, deprocess_image
from conf import *
import numpy as np
import os
import tensorflow as tf
import time
from model import overall_net
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg16

if __name__ == "__main__":

    # loss_model = loss_net()
    # train_filenames = get_file_paths(train_image_dirpath)[:100]
    # imgs = []
    # for i, filename in enumerate(train_filenames, start=1):
    #     filepath = os.path.join(train_image_dirpath, filename)
    #     img = preprocess_image(filepath)
    #     imgs.append(img)
    # imgs = np.vstack(imgs)
    # global graph
    # graph = tf.get_default_graph()
    # t1 = time.time()
    # with graph.as_default():
    #     loss_model.predict(imgs)
    # t2 = time.time()
    # print(t2 - t1)

    img = load_img("img/test/raw/test1.jpg")
    img_size = img.size
    print(img_size)
    if img_size[0] % 4 == 0:
        diff1 = 0
    else:
        diff1 = 4 - (img_size[0] % 4)
    if img_size[1] % 4 == 0:
        diff2 = 0
    else:
        diff2 = 4 - (img_size[1] % 4)
    img = img.resize((img_size[0] + diff1, img_size[1] + diff2))
    img_size = img.size
    print(img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    overall, trans, loss = overall_net(img_size[1], img_size[0])
    trans.load_weights("model/mosaic.hdf5")
    img = trans.predict(img)
    img = img.reshape((img_size[1], img_size[0], 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    save_img("test4.jpg", img)
