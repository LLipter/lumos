from keras import backend as K


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    size = x.shape[0] * x.shape[1] * x.shape[2]
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
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert style.shape == combination.shape
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return K.sum(K.square(S - C))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    assert K.ndim(base) == 3
    assert base.shape == combination.shape
    size = base.shape[0] * base.shape[1] * base.shape[2]
    return K.sum(K.square(combination - base)) / size


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :x.shape[1] - 1, :x.shape[2] - 1] - x[:, :, 1:, :x.shape[2] - 1])
        b = K.square(
            x[:, :x.shape[1] - 1, :x.shape[2] - 1] - x[:, :, :x.shape[1] - 1, 1:])
    else:
        a = K.square(
            x[:x.shape[0] - 1, :x.shape[1] - 1, :] - x[:, 1:, :x.shape[1] - 1, :])
        b = K.square(
            x[:x.shape[0] - 1, :x.shape[1] - 1, :] - x[:, :x.shape[0] - 1, 1:, :])
    return K.sum(K.sqrt(a + b))
