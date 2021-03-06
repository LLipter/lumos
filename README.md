# 我们要做什么

我们要做一个视频风格转换的程序，有点类似一个视频的滤镜。比如说将一个正常的视频转化为漫画风格视频。



# 怎么做

深度学习算法中已经有人提出了一种模型，能够做得到类似的事情，我们可以有效的加以利用。

这种算法接受两个输入图片，一张作为风格模板，一张作为原图。对原图进行修改，输出一张参照风格模板修改过后的图片。具体网络是怎么实现的，结构是怎么样的我都不清楚，没有仔细了解过，这部分需要大家自己学习一下，可以搜索关键词“深度学习”、“机器学习”、“风格转换”等等。 好像听说是运用到了一种叫做GAN（对抗生成网络）的结构，可能是GAN的一种变体吧。

拿到可以使用的模型之后，一个最直观的思路就是把视频分成一帧一帧的图片，然后分别丢到模型里面去得到转化后的图片，然后再拼起来得到修改后的视频。

但是这样显然效率太低，考虑到一个长度为5分钟的视频，以每秒20帧计算，那么就会有6000张图片需要进行处理。这类模型的运算效率现在我不清楚，但猜测不会很快，这样的话转换视频所需的时间就会长到无法接受。我目前想到了一种优化思路，那就是借鉴视频压缩的技术。因为相邻两帧视频之间会有很大的相似度，所以完全可以不把每一帧图片都扔到模型中训练，而是比如说选取1000帧（等距分布在整个视频中）扔到模型中转化，由这1000转化后的图片推测剩下的5000帧，形成连贯的视频。我记得上数字图像处理课的时候好像讲过相似的东西。实际上视频就会利用这样的手段进行压缩。可能会用到一个叫残差的概念。



# 开发平台以及所用到的开源组件

主开发语言是`python3`

### 机器学习模型的训练和修改

`tensorflow + keras`

`tensorflow`是一个很出名的机器学习框架，也挺好用的，安装也还算方便。但有一点要注意的是，为了提高模型的训练速度，最好安装`tensorflow`的GPU版本，GPU版本相对于CPU版本会有很大的性能提升，缩短训练模型需要的时间。安装GPU版本的时候确实很麻烦，有很多坑需要注意。大体而言我们需要准备显卡驱动、`cuda`和`cuDNN`三个东西。这些东西全都正确配置了之后才能让GPU版本的`tensorflow`运行起来。显卡驱动是根据自己的电脑型号而定的，可以去官网搜索下载。`cuda`和`cudnn`是运用GPU进行高效计算的库，他们之间的版本必须正确匹配才能发挥作用。具体的安装还是参考百度上各种博客吧，**有很多坑**，可能要踩一阵子才能装成功。另外在`ubuntu`上装应该会相对而言简单一点，出错少一点。

`keras`是一个基于`tensorflow`的库。本质上就是再把`tensorflow`封装了一遍，各种模型实现起来和搭积木一样非常方便。在看关于这部分博客的时候也重点看利用`keras`实现的。

### 数字图像处理

`OpenCV`

这是一个开源的图像处理库，里面支持各种关于图像处理的算法、函数等等。

# 难点

1. 因为借鉴到了深度学习的模型，这类模型的训练需要大量的时间，在自己电脑性能不佳的情况下更是进展缓慢。
2. 在算法优化部分，一定会需要对数字图像的底层有深入的掌握，也就是说一定会需要运用数字图像处理这门课讲到的知识，并且大概率会比较深入，晦涩难懂。



#  步骤

1. 了解该机器学习模型的基本原理之后，寻找适合的模型，加以调整、训练。可以疯狂从`github`上找
2. 学习数字图像处理知识，实现我所设想的算法优化。
3. 进一步的优化、以及可能的模型优化，实现更加复杂的风格转换。



# Reference

- 机器学习
  1. [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)(2015.8.26)
  2. **[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)**(2016.3.27)
  3. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)(2016.7.27)
  4. [jcjohnson/fast-neural-style](https://github.com/jcjohnson/fast-neural-style)(文献2作者提供的github代码仓库)
    5. **[基于深度学习的图像风格转换](https://blog.csdn.net/u013805360/article/details/73543229)**（文献2的中文解读）
    6. [wisewong/ImageStyleTransform](https://github.com/wisewong/ImageStyleTransform)(文献4的代码实现)
    7. **[keras官方给的风格转移示例程序](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)**（基于文献1的实现）
    8. **[【啄米日常】5：keras示例程序解析（2）：图像风格转移](https://zhuanlan.zhihu.com/p/23479658)**（文献5的代码中文解读）
  9. [Keras Documentation](https://keras.io/)
  10. [深度残差网络RESNET](https://blog.csdn.net/qq_31050167/article/details/79161077)
  11. [图像的上采样（upsampling）与下采样（subsampled）](https://blog.csdn.net/stf1065716904/article/details/78450997)
  12. [使用上采样加卷积的方法代替反卷积](https://blog.csdn.net/MOOLLLLLLLLLLL/article/details/80221613?utm_source=blogkpcl4)
  13. [Deconvolution and Checkerboard Artifacts (cn)](https://www.jianshu.com/p/36ff39344de5)
  14. [Deconvolution and Checkerboard Artifacts (en)](https://distill.pub/2016/deconv-checkerboard/)
  15. [VGG16学习笔记](https://blog.csdn.net/dta0502/article/details/79654931)
  16. [Total variation denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)
  17. [Texture Networks + Instance normalization: Feed-forward Synthesis of Textures and Stylized Images](https://github.com/DmitryUlyanov/texture_nets)
- 数字图像技术
  1. [JPEG压缩编码算法原理](https://blog.csdn.net/u013752202/article/details/78551592)
  2. [视频压缩算法的相关知识](https://www.cnblogs.com/mengfanrong/p/3827052.html?tdsourcetag=s_pctim_aiomsg)
  3. [转：MPEG-2压缩编码技术原理应用](https://www.cnblogs.com/xkfz007/articles/2615192.html?tdsourcetag=s_pctim_aiomsg)
  4. [MPEG1压缩编码原理](http://blog.chinaunix.net/uid-31409925-id-5754943.html)
  5. **[Video compression picture types](https://en.wikipedia.org/wiki/Video_compression_picture_types)**
  6. **[how prediction is calulated](https://softwareengineering.stackexchange.com/questions/165872/what-are-mpeg-i-p-and-b-frames)**
    7. [【官方双语】形象展示傅里叶变换](https://www.bilibili.com/video/av19141078?from=search&seid=11382747690788550091)
  8. [【官方双语】欧拉公式与初等群论](https://www.bilibili.com/video/av11339177)

