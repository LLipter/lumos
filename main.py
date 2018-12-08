from model import transform_net, loss_net, overall_net
import os
from util import preprocess_image, deprocess_image
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint
import numpy as np
from conf import *
import matplotlib.pyplot as plt
import time


loss_model = None

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
        feature = loss_model.predict(style_image)
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
        feature = loss_model.predict(train_image)
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
        yield ([imgs] + content_features + style_features, np.zeros((batch_size,)))


def transform_test_image(epoch, logs):
    filenames = get_file_paths(test_image_dirpath)
    filenames = sorted(filenames)
    row = len(filenames) / 5
    col = 5
    plt.figure(figsize=(16, 8))
    for i, filename in enumerate(filenames, start=1):
        filepath = os.path.join(test_image_dirpath, filename)
        img = preprocess_image(filepath)
        img = deprocess_image(img)
        plt.subplot(row, col, i)
        plt.imshow(img)
    save_path = os.path.join(test_image_savepath, "%d-%d.png" % (time.time(), epoch))
    plt.savefig(save_path)


if __name__ == "__main__":
    print("hello lumos!")
    loss_model = loss_net()
    loss_model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
    loss_model._make_predict_function()
    overall_model = overall_net()
    overall_model.summary()
    overall_model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
    predict_callback = LambdaCallback(on_epoch_end=transform_test_image)

    model_path = os.path.join(model_dirpath, style_name) + ".hdf5"
    checkpointer = ModelCheckpoint(filepath=model_dirpath)
    overall_model.fit_generator(generator(batch_size=4),
                                steps_per_epoch=250,
                                epochs=100,
                                callbacks=[predict_callback, checkpointer])

