from model import transform_net, loss_net, overall_net
import os
from util import preprocess_image, load_data
from keras import backend as K
from keras.optimizers import Adam
# from conf import style_feature_layers, content_feature_layers
# from loss import content_loss_func, style_loss_func, tv_loss_func

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

    img_paths = []
    for root, dirs, files in os.walk("img/test"):
        for filename in files:
            img_paths.append(os.path.join(root, filename))
    train_data = load_data(img_paths)
    style_data = preprocess_image("img/style/candy.jpg")
    style_data = K.variable(style_data)
    style_data = K.repeat_elements(style_data, train_data.shape[0], axis=0)

    model = overall_net()
    opt = Adam()
    # loss_func = []
    # for _ in range(len(content_feature_layers)):
    #     loss_func.append(content_loss_func)
    # for _ in range(len(style_feature_layers)):
    #     loss_func.append(style_loss_func)
    # loss_func.append(tv_loss_func)
    model.compile(opt, loss=lambda y_pred, y_true: y_pred)
    # model.fit(x=[train_data, style_data], batch_size=4, epochs=10)
