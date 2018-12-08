img_nrows = 256
img_ncols = 256
content_weight = 0.025
style_weight = 5
tv_weight = 1
style_feature_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1',
                        'block5_conv1']
content_feature_layers = ['block5_conv1']
style_name = "candy"
style_image_dirpath = "img/style/raw"
style_feature_dirpath = "img/style/feature"
train_image_dirpath = "img/train/raw"
train_feature_dirpath = "img/train/feature"
test_image_dirpath = "img/test/raw"
test_image_savepath = "img/test/transformed"
model_dirpath = "model"
