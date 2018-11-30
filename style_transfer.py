from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np

from keras.applications import vgg19
from keras import backend as K
from keras.layers import Input, Conv2D, Add, Conv2DTranspose, Activation, Lambda
from keras.models import Model
from keras.utils import plot_model
from keras_contrib.layers import InstanceNormalization

img_nrows = 256
img_ncols = 256


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


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


if __name__ == "__main__":
    print("hello world")
    # img = preprocess_image("img/style/candy.jpg")
    # print(img)
    # print(img.shape)

    # input_tensor = K.placeholder((1, img_nrows, img_ncols, 3))
    # model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    # print('Model loaded.')
    # outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # for key in outputs_dict.keys():
    #     print(key)

    transformNet = transform_model()
    print(K.image_data_format())
