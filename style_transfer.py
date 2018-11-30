from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np

from keras.applications import vgg19
from keras import backend as K
from keras.layers import Input, Conv2D, ReLU
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


def downSampling(x, filters, kernel_size, strides, padding):
    conv1 = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   data_format=K.image_data_format())(x)
    if K.image_data_format() == 'channels_first':
        instance_normalized = InstanceNormalization(axis=1)(conv1)
    else:
        instance_normalized = InstanceNormalization(axis=3)(conv1)
    relu = ReLU()(instance_normalized)
    return relu


def transform_model():
    if K.image_data_format() == 'channels_first':
        input = Input(shape=(3, img_nrows, img_ncols))
    else:
        input = Input(shape=(img_nrows, img_ncols, 3))

    conv1 = downSampling(input, 32, 9, 1, "same")
    conv2 = downSampling(conv1, 64, 3, 2, "valid")
    conv3 = downSampling(conv2, 128, 3, 2, "valid")

    transformNet = Model(inputs=input, outputs=conv3)
    plot_model(transformNet, to_file="img/model/transform.png", show_shapes=True)
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



