from model import transform_net, loss_net, overall_net
import os
from util import preprocess_image, load_data
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from conf import style_feature_layers, content_feature_layers, style
# from loss import content_loss_func, style_loss_func, tv_loss_func


def get_file_paths(path):
    paths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            paths.append(file)
    return paths

def generator(batch_size):
    # check whether style feature has already been computed
    


    path = "img/train/raw"
    train_filenames = get_file_paths(path)
    length = len(train_filenames)
    while True:
        index = np.random.randint(length, size=(batch_size,))
        imgs = []
        for i in index:
            filepath = os.path.join(path, train_filenames[i])
            img = preprocess_image(filepath)
            imgs.append(img)
        imgs = np.vstack(imgs)
        yield imgs


if __name__ == "__main__":
    print("hello world")
    # img_paths = []
    # for root, dirs, files in os.walk("img/test"):
    #     for filename in files:
    #         img_paths.append(os.path.join(root, filename))
    # train_data = load_data(img_paths)
    # style_data = preprocess_image("img/style/candy.jpg")
    # style_data = K.variable(style_data)
    # style_data = K.repeat_elements(style_data, train_data.shape[0], axis=0)
    #
    # model = overall_net()
    # model.summary()
    # opt = Adam()
    # model.compile(opt, loss=lambda y_pred, y_true: y_pred)

    a = get_file_paths("img/style/raw")
    print(a)

    x = generator(10)
    for i in x:
        print(i.shape)
