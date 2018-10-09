---
title: 'DeepLearning.ai笔记:(4-2)-- 深度卷积网络实例探究（Deep convolutional models:case studies）'
id: dl-ai-4-2
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-10-09 17:17:04
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

本周主要讲了深度卷积网络的一些模型：LeNet,AlexNet,VGGNet,ResNet,Inception,1×1卷积，迁移学习等。

<!--more-->

# 经典的卷积网络

经典的卷及网络有三种：LeNet、AlexNet、VGGNet。



**LeNet-5**

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgc3e4j20ox0c5t9k.jpg)

LeNet-5主要是单通道的手写字体的识别，这是80年代提出的算法，当时没有用padding，而且pooling用的是average pooling，但是现在大家都用max pooling了。

论文中的最后预测用的是sigmoid和tanh，而现在都用了softmax。



**AlexNet**

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgltf9j20p20d9q3w.jpg)

AlexNet是2012年提出的算法。用来对彩色的图片进行处理，其实大致的结构和LeNet-5是很相似的，但是网络更大，参数更多了。

这个时候已经用Relu来作为激活函数了，而且用了多GPU进行计算。



**VGG-16**

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgmomwj217i0n4di7.jpg)

VGG-16是2015的论文，比较简化的是，卷积层和池化层都是用相同的卷积核大小，卷积核都是3×3，stride=1，same padding，池化层用的maxpooling，为2×2，stride=2。只是在卷积的时候改变了每一层的通道数。

网络很大，参数有1.38亿个参数。



**建议阅读论文顺序：AlexNet->VGG->LeNet**



# Residual Network(残差网络)

残差网络是由若干个残差块组成的。

因为在非常深的网络中会存在梯度消失和梯度爆炸的问题，为此，引入了**Skip Connection**来解决，也就是残差网络的实现。



![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgavrvj20bc05sdfw.jpg)

上图即为一个残差块的基本原理，在原本的传播过程(称为主线)中，加上了$a^{[l]}$到$z^{[l+2]}$的连接，成为'short cut'或者'skip connetction'。

所以输出的表达式变成了:$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgcjiqj20yv0c7gml.jpg)



残差网络是由多个残差块组成的：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgdlpfj20xc08gq3p.jpg)



没有残差网络和加上残差网络的效果对比，可以看到，随着layers的增加，ResNet表现的更好：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgd9gaj20yo0anq3e.jpg)



# ResNet为何有用？

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgwuudj20io05vq50.jpg)

假设我们已经经过了一个很大的神经网络Big NN,得到了$a^{[l]}$

那么这个时候再经过两层的神经网络得到$a^{[l+2]}$,那么表达式为：

$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]}) = g(W^{[l+2]} a^{[l+2]} + b^{[l+2]} + a^{[l]})$$

如果加上正则化，那么权值就会很小，假设$W^{[l+2]},b^{[l+2]} = 0$， 因为激活函数是Relu，所以

$$a^{[l+2]} = g(a^{[l]}) = a^{[l]}$$

所以可以看到，加上残差块以后，更深的网络最差也只是和前面的效果一样，何况还有可能更好。

如果只是普通的两层网络，那么结果可能更好，也可能更差。

注意的是$a^{[l+2]}$要和$a^{[l]}$的维度一样，可以使用same padding，来保持维度。



# 1×1卷积

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgeo2fj20hs09vq2z.jpg)

用1×1的卷积核可以来减少通道数，从而减少参数个数。



# Inception Network

Inception的主要好处就是不需要人工来选择filter的大小和是否要添加池化层的问题。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgr815j20nj0bo756.jpg)

如图可以一次性把各个卷积核的大小和max pool一起加进去，然后让机器自己学习里面的参数。



但是这样有一个问题，就是计算量太大了，假设是上面的$5 \times 5 \times 192$的卷积核，有32个，这样一共要进行$28\times\28\times32\times5\times5\times192=120M$的乘法次数，运算量是很大的。

如何解决这个问题呢？就需要用到前面的1×1的卷积核了。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgtwmtj20og0dqgmd.jpg)

可以看到经过维度压缩，计算次数少了十倍。



# Inception 网络

单个的inception模块如下：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgsv81j215g0m70w1.jpg)



构成的google net如下：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kh9sd1j21eh0quk1o.jpg)



# 使用开源的实现方案

别人已经实现的网络已经很厉害了，我觉得重复造轮子很没有必要，而且浪费时间，何况你水平也没有别人高。。还不如直接用别人的网络，然后稍加改造，这样可以很快的实现你的想法。

在GitHub上找到自己感兴趣的网络结构fork过来，好好研究！

# 迁移学习

之前已经讲过迁移学习了，也就是用别人训练好的网络，固定他们已经训练好的网络参数，然后套到自己的训练集上，完成训练。



如果你只有很少的数据集，那么，改变已有网络的最后一层softmax就可以了，比如原来别人的模型是有1000个分类，现在你只需要有3个分类。然后freeze冻结前面隐藏层的所有参数不变。这样就好像是你自己在训练一个很浅的神经网络，把隐藏层看做一个函数来映射，只需要训练最后的softmax层就可以了。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgqbpqj20no062aca.jpg)



如果你有一定量的数据，那么freeze的范围可以减少，你可以训练后面的几层隐藏层，或者自己设计后面的隐藏层。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fw26kgiq0xj20nm04wabg.jpg)



# 数据扩充

数据不够的话，进行数据扩充是很有用的。

可以采用

- 镜像
- 随机裁剪
- 色彩转换color shifting（如三通道：R+20,G-20,B+20）等等





**tips:**

在数据比赛中

- ensembling：训练多个网络模型，然后平均结果，或者加权平均
- 测试时使用muti-crop，也就是在把单张测试图片用数据扩充的形式变成很多张，然后运行分类器，得到的结果进行平均。



