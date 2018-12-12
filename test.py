from model import loss_net
from util import get_file_paths, preprocess_image
from conf import *
import numpy as np
import os
import tensorflow as tf
import time

if __name__ == "__main__":

    loss_model = loss_net()
    train_filenames = get_file_paths(train_image_dirpath)[:100]
    imgs = []
    for i, filename in enumerate(train_filenames, start=1):
        filepath = os.path.join(train_image_dirpath, filename)
        img = preprocess_image(filepath)
        imgs.append(img)
    imgs = np.vstack(imgs)
    global graph
    graph = tf.get_default_graph()
    t1 = time.time()
    with graph.as_default():
        loss_model.predict(imgs)
    t2 = time.time()
    print(t2 - t1)
