from model import transform_net, loss_net, overall_net
import os
from util import preprocess_image, load_data
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from conf import *


overall_model, loss_net = overall_net()

def get_file_paths(path):
    paths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            paths.append(file)
    return paths


def get_style_feature():
    # check whether style name is valid
    style_image_path = os.path.join(style_image_dirpath, style_name) + ".jpg"
    assert os.path.exists(style_image_path)

    # check whether feature directory exists
    if not os.path.exists(style_feature_dirpath):
        os.mkdir(style_feature_dirpath)
    assert os.path.isdir(style_feature_dirpath)

    # check whether style feature has already been computed
    # if not, recompute it
    feature_path = os.path.join(style_feature_dirpath, style_name) + ".npz"
    if not os.path.exists(feature_path):
        style_image = preprocess_image(style_image_path)
        feature = loss_net.predict(style_image)
        feature = feature[len(content_feature_layers):]
        feature_dict = {}
        for i, layer_name in enumerate(style_feature_layers):
            feature_dict[layer_name] = feature[i]
        np.savez(feature_path, **feature_dict)

    return np.load(feature_path)


def get_content_feature(filename):
    # check whether image path is valid
    image_path = os.path.join(train_image_dirpath, filename)
    assert os.path.exists(image_path)

    # check whether feature directory exists
    if not os.path.exists(train_feature_dirpath):
        os.mkdir(train_feature_dirpath)
    assert os.path.isdir(train_feature_dirpath)

    # check whether train feature has already been computed
    # if not, recompute it
    feature_path = os.path.join(train_feature_dirpath, filename) + ".npz"
    if not os.path.exists(feature_path):
        train_image = preprocess_image(image_path)
        feature = loss_net.predict(train_image)
        feature = feature[:len(content_feature_layers)]
        feature_dict = {}
        for i, layer_name in enumerate(content_feature_layers):
            feature_dict[layer_name] = feature[i]
        np.savez(feature_path, **feature_dict)

    return np.load(feature_path)


def generator(batch_size):
    style_features = get_style_feature()
    style_features = [np.repeat(style_features[layer_name], batch_size, axis=0) for layer_name in style_feature_layers]

    train_filenames = get_file_paths(train_image_dirpath)
    length = len(train_filenames)

    while True:
        index = np.random.randint(length, size=(batch_size,))
        imgs = []
        content_features = []
        for i in index:
            filepath = os.path.join(train_image_dirpath, train_filenames[i])
            content_features.append(get_content_feature(train_filenames[i]))
            img = preprocess_image(filepath)
            imgs.append(img)
        imgs = np.vstack(imgs)
        content_features = [np.vstack(
                                    [content_features[i][layer_name] for i in range(len(content_features))]
                                    )
                            for layer_name in content_feature_layers]
        yield [imgs] + content_features + style_features


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

    # model.summary()
    # opt = Adam()
    # model.compile(opt, loss=lambda y_pred, y_true: y_pred)

    # a = get_file_paths("img/style/raw")
    # print(a)
    #
    x = generator(10)
    for i in x:
        for a in i:
            print(a.shape)



    # style_image = preprocess_image("img/style/raw/candy.jpg")
    # print(style_image)
    #
    # feature = loss_net.predict(style_image)
    # feature = feature[len(content_feature_layers):]
    # feature_dict = {}
    # for i, layer_name in enumerate(style_feature_layers):
    #     feature_dict[layer_name] = feature[i]
    # print(feature_dict.keys())
    #
    # np.savez("img/style/feature/candy.npz", **feature_dict)


