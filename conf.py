img_nrows = 256
img_ncols = 256
content_weight = 1
style_weight = 8
tv_weight = 1e-6
style_feature_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1',
                        'block5_conv1']
content_feature_layers = ['block5_conv2']
style = "candy"
