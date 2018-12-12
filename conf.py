img_nrows = 256
img_ncols = 256
content_weight = 1
style_weight = 5
tv_weight = 1e-6
learning_rate = 1e-3
style_feature_layers = ['block1_conv2', 'block2_conv2',
                        'block3_conv3', 'block4_conv3']
content_feature_layers = ['block2_conv2']
style_name = "candy"
style_image_dirpath = "img/style/raw"
style_feature_dirpath = "img/style/feature"
train_image_dirpath = "img/train/raw"
train_feature_dirpath = "img/train/feature"
test_image_dirpath = "img/test/raw"
test_image_savepath = "img/test/transformed"
model_dirpath = "model"
