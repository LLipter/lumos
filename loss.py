from keras import backend as K
from keras.layers import Flatten

from conf import content_weight, style_weight, tv_weight


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    size = int(x.shape[0] * x.shape[1] * x.shape[2])
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features)) / size
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(x):
    style, combination = x[0], x[1]
    print(style.shape)
    print(combination.shape)

    x = K.reshape(style, shape=(-1, int(style.shape[1]*style.shape[2]), int(style.shape[3])))
    print(x.shape)

    x = K.reshape(combination, shape=(combination.shape[0], -1, combination.shape[3]))
    print(x.shape)

    assert K.ndim(style) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    loss = style_weight * K.sum(K.square(S - C))
    return K.reshape(loss, shape=(1,))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(x):
    base, combination = x[0], x[1]
    # print(base.shape)
    # print(combination.shape)
    assert K.ndim(base) == 4
    size = int(base.shape[1] * base.shape[2] * base.shape[3])
    loss = content_weight * K.sum(K.square(combination - base), axis=[1, 2, 3]) / size
    loss = K.reshape(loss, shape=(-1, 1))
    # print(loss.shape)
    return loss


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    # print(x.shape)
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :x.shape[2] - 1, :x.shape[3] - 1] - x[:, :, :, 1:, :x.shape[3] - 1])
        b = K.square(
            x[:, :, :x.shape[2] - 1, :x.shape[3] - 1] - x[:, :, :, :x.shape[2] - 1, 1:])
    else:
        a = K.square(
            x[:, :x.shape[1] - 1, :x.shape[2] - 1, :] - x[:, 1:, :x.shape[2] - 1, :])
        b = K.square(
            x[:, :x.shape[1] - 1, :x.shape[2] - 1, :] - x[:, :x.shape[1] - 1, 1:, :])
    loss = tv_weight * K.sum(K.sqrt(a + b), axis=[1, 2, 3])
    loss = K.reshape(loss, shape=(-1, 1))
    # print(loss.shape)
    return loss

# def content_loss_func(y_pred, y_true):
#     base = y_pred[0, :, :, :]
#     transformed = y_pred[2, :, :, :]
#     return content_weight * content_loss(base, transformed)
#
#
# def style_loss_func(y_pred, y_true):
#     style = y_pred[1, :, :, :]
#     transformed = y_pred[2, :, :, :]
#     return style_weight * style_loss(style, transformed)
#
#
# def tv_loss_func(y_pred, y_true):
#     return tv_weight * total_variation_loss(y_pred)


# def loss_function(y_pred, y_true):
#     loss = K.variable(0.0)
#     # content loss
#     for i in range(len(content_feature_layers)):
#         base = y_pred[i][0, :, :, :]
#         transformed = y_pred[i][2, :, :, :]
#         loss.assign_add(content_weight * content_loss(base, transformed))
#     # style loss
#     for i in range(len(style_feature_layers)):
#         style = y_pred[i + len(content_feature_layers)][1, :, :, :]
#         transformed = y_pred[i + len(content_feature_layers)][2, :, :, :]
#         loss.assign_add(style_weight * style_loss(style, transformed))
#     # total variation loss
#     loss.assign_add(tv_weight * total_variation_loss(y_pred[-1]))
