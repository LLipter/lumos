from keras import backend as K


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    size = int(x.shape[1] * x.shape[2] * x.shape[3])
    if K.image_data_format() == 'channels_first':
        shape = (-1, x.shape[1], x.shape[2] * x.shape[3])
        features = K.reshape(x, shape)
    else:
        shape = (-1, x.shape[3], x.shape[1] * x.shape[2])
        features = K.reshape(K.permute_dimensions(x, (0, 3, 1, 2)), shape)
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1))) / size
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return K.sum(K.square(S - C))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    size = int(base.shape[1] * base.shape[2] * base.shape[3])
    return K.sum(K.square(combination - base), axis=[1, 2, 3]) / size

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
