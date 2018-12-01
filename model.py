from keras import backend as K
from keras.layers import Input, Conv2D, Add, Conv2DTranspose, Activation, Lambda, Concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.applications import vgg19
from keras_contrib.layers import InstanceNormalization

from conf import img_nrows, img_ncols, style_feature_layers, content_feature_layers


def down_sampling(x, filters, kernel_size, strides, padding="same", activation="relu"):
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


def up_sampling(x, filters, kernel_size, strides, padding="same", activation="relu"):
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


def transform_net():
    if K.image_data_format() == 'channels_first':
        input_tensor = Input(shape=(3, img_nrows, img_ncols))
    else:
        input_tensor = Input(shape=(img_nrows, img_ncols, 3))

    conv1 = down_sampling(input_tensor, 32, 9, 1)
    conv2 = down_sampling(conv1, 64, 3, 2)
    conv3 = down_sampling(conv2, 128, 3, 2)

    resi1 = residul(conv3)
    resi2 = residul(resi1)
    resi3 = residul(resi2)
    resi4 = residul(resi3)
    resi5 = residul(resi4)

    deconv1 = up_sampling(resi5, 64, 3, 2)
    deconv2 = up_sampling(deconv1, 32, 3, 2)
    deconv3 = up_sampling(deconv2, 3, 9, 1)

    output = Lambda(lambda x: x * 128)(deconv3)

    transform_model = Model(inputs=input_tensor, outputs=output)
    plot_model(transform_model, to_file="img/model/transform_net.png", show_shapes=True)
    return transform_model


def loss_net():
    # get tensor representations of our images
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_nrows, img_ncols)
    else:
        input_shape = (img_nrows, img_ncols, 3)

    # build the VGG19 network with pre-trained ImageNet weights loaded
    input_tensor = Input(shape=input_shape)
    vgg19_model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    plot_model(vgg19_model, to_file="img/model/vgg19.png", show_shapes=True)

    # freeze weights
    for layer in vgg19_model.layers:
        layer.trainable = False

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg19_model.layers])

    output_tensors = []
    for layer_name in content_feature_layers:
        output_tensors.append(outputs_dict[layer_name])
    for layer_name in style_feature_layers:
        output_tensors.append(outputs_dict[layer_name])
    return Model(inputs=input_tensor, outputs=output_tensors)


def overall_net():
    trans_net = transform_net()
    los_net = loss_net()

    if K.image_data_format() == 'channels_first':
        base_image = Input(shape=(3, img_nrows, img_ncols))
        style_image = Input(shape=(3, img_nrows, img_ncols))
    else:
        base_image = Input(shape=(img_nrows, img_ncols, 3))
        style_image = Input(shape=(img_nrows, img_ncols, 3))
    transformed_image = trans_net(base_image)

    input_tensor = Concatenate(axis=0)([base_image,
                                        style_image,
                                        transformed_image])
    output_tensor = los_net(input_tensor)
    output_tensor.append(transformed_image)

    overall_model = Model(inputs=[base_image, style_image], outputs=output_tensor)
    plot_model(overall_model, to_file="img/model/overall.png", show_shapes=True)
    return overall_model
