from keras import backend as K

from conf import img_nrows, img_ncols

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
    assert K.ndim(base) == 4
    assert base.shape == combination.shape
    size = base.shape[1] * base.shape[2] * base.shape[3]
    return K.sum(K.square(combination - base), axis=0) / size


# # the 3rd loss function, total variation loss,
# # designed to keep the generated image locally coherent
# def total_variation_loss(x):
#     assert K.ndim(x) == 4
#     if K.image_data_format() == 'channels_first':
#         a = K.square(
#             x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
#         b = K.square(
#             x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
#     else:
#         a = K.square(
#             x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
#         b = K.square(
#             x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
#     return K.sum(K.pow(a + b, 1.25))
#
#
#
#
