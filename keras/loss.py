from keras import backend as K

from conf import content_weight, style_weight, tv_weight


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 4
    int_shape = K.int_shape(x)
    height = int_shape[1]
    width = int_shape[2]
    channel = int_shape[3]
    size = height * width * channel
    features = K.permute_dimensions(x, (0, 3, 1, 2))
    features = K.reshape(features, shape=(-1,
                                          channel,
                                          height * width))
    # print(features.shape)
    # print(K.permute_dimensions(features, (0, 2, 1)).shape)
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1))) / size / 2
    # print("gram size", gram.shape)
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(x):
    style, combination = x[0], x[1]
    assert K.ndim(style) == 4
    S = gram_matrix(style)
    C = gram_matrix(combination)
    loss = style_weight * K.sum(K.square(S - C), axis=[1, 2])
    loss = K.reshape(loss, shape=(-1, 1))
    # print("loss size", loss.shape)
    return loss


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(x):
    base, combination = x[0], x[1]
    # print(base.shape)
    # print(combination.shape)
    assert K.ndim(base) == 4
    int_shape = K.int_shape(base)
    size = int_shape[1] * int_shape[2] * int_shape[3]
    loss = content_weight * K.sum(K.square(combination - base), axis=[1, 2, 3]) / size / 2
    loss = K.reshape(loss, shape=(-1, 1))
    # print(loss.shape)
    return loss


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    # print(x.shape)
    assert K.ndim(x) == 4
    int_shape = K.int_shape(x)
    if K.image_data_format() == 'channels_first':
        height = int_shape[2]
        width = int_shape[3]
        a = K.square(
            x[:, :, :height - 1, :width - 1] - x[:, :, :, 1:, :width - 1])
        b = K.square(
            x[:, :, :height - 1, :width - 1] - x[:, :, :, :height - 1, 1:])
    else:
        height = int_shape[1]
        width = int_shape[2]
        a = K.square(
            x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
        b = K.square(
            x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    loss = tv_weight * K.sum(K.pow(a + b, 1.25), axis=[1, 2, 3])
    loss = K.reshape(loss, shape=(-1, 1))
    # print(loss.shape)
    return loss
