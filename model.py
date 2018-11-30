from keras import backend as K
from keras.layers import Input, Conv2D, Add, Conv2DTranspose, Activation, Lambda
from keras.models import Model
from keras.utils import plot_model
from keras_contrib.layers import InstanceNormalization

from conf import img_nrows, img_ncols


def downSampling(x, filters, kernel_size, strides, padding="same", activation="relu"):
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  data_format=K.image_data_format())(x)
    if K.image_data_format() == 'channels_first':
        instance_normalized = InstanceNormalization(axis=1)(conv)
    else:
        instance_normalized = InstanceNormalization(axis=3)(conv)
    activated = Activation(activation)(instance_normalized)
    return activated


def residul(x):
    conv1 = Conv2D(filters=128,
                   kernel_size=3,
                   strides=1,
                   padding="same",
                   data_format=K.image_data_format())(x)
    if K.image_data_format() == 'channels_first':
        instance_normalized1 = InstanceNormalization(axis=1)(conv1)
    else:
        instance_normalized1 = InstanceNormalization(axis=3)(conv1)
    relu = Activation("relu")(instance_normalized1)
    conv2 = Conv2D(filters=128,
                   kernel_size=3,
                   strides=1,
                   padding="same",
                   data_format=K.image_data_format())(relu)
    if K.image_data_format() == 'channels_first':
        instance_normalized2 = InstanceNormalization(axis=1)(conv2)
    else:
        instance_normalized2 = InstanceNormalization(axis=3)(conv2)
    return Add()([instance_normalized2, x])


def upSampling(x, filters, kernel_size, strides, padding="same", activation="relu"):
    deconv = Conv2DTranspose(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=K.image_data_format())(x)
    if K.image_data_format() == 'channels_first':
        instance_normalized = InstanceNormalization(axis=1)(deconv)
    else:
        instance_normalized = InstanceNormalization(axis=3)(deconv)
    activated = Activation(activation)(instance_normalized)
    return activated


def transform_model():
    if K.image_data_format() == 'channels_first':
        input = Input(shape=(3, img_nrows, img_ncols))
    else:
        input = Input(shape=(img_nrows, img_ncols, 3))

    conv1 = downSampling(input, 32, 9, 1)
    conv2 = downSampling(conv1, 64, 3, 2)
    conv3 = downSampling(conv2, 128, 3, 2)

    resi1 = residul(conv3)
    resi2 = residul(resi1)
    resi3 = residul(resi2)
    resi4 = residul(resi3)
    resi5 = residul(resi4)

    deconv1 = upSampling(resi5, 64, 3, 2)
    deconv2 = upSampling(deconv1, 32, 3, 2)
    deconv3 = upSampling(deconv2, 3, 9, 1)

    output = Lambda(lambda x: x * 128)(deconv3)

    transformNet = Model(inputs=input, outputs=output)
    plot_model(transformNet, to_file="img/model/transformNet.png", show_shapes=True)
    return transformNet
