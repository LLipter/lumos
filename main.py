from model import transform_net, loss_net, overall_net

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

    model = overall_net()
    print(model)
