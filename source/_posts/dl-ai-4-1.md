---
title: 'DeepLearning.ai笔记:(4-1)-- 卷积神经网络（Foundations of CNN）'
id: dl-ai-4-1
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-30 10:20:54
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

第四门课开始就学习深度学习关于计算机视觉的重要应用---卷积神经网络。

第一周主要是对卷积神经网络的基本构造和原理做了介绍。

<!--more-->



# 计算机视觉

计算机视觉是深度学习的一个非常重要的应用。比如图像分类，目标检测，图片风格迁移等。

用传统的深度学习算法，假设你有一张$64×64$的猫片，又有RGB三通道，那么这个时候是$64×64×3=12288$，input layer的维度就是12288，这样其实也还可以，因为图片很小。那么如果你有$1000×1000$的照片呢，你的向量就会有300万！假设有1000个隐藏神经元，那么就是第一层的参数矩阵$W$有30亿个参数！算到地老天荒。所以用传统的深度学习算法是不现实的。



# 边缘检测

如图，这些边缘检测中，用水平检测和垂直检测会得到不同的结果。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v3yw2j20hg0813z5.jpg)



垂直检测如下图，用一个$3×3$的过滤器（filter），也叫卷积核，在原图片$6×6$的对应地方按元素相乘，得到$4×4$的图片。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1whcopj20x40iln2i.jpg)



可以看到，用垂直边缘的filter可以将原图片中间的边缘区分出来，也就是得到了最右图中最亮的部分即为检测到的边缘。



当然，如果左图的亮暗分界线反过来，则输出图片中最暗的部分表示边缘。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v5z60j216s0o2q4z.jpg)





也自然有水平的边缘分类器。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v2vcqj20t509ajrh.jpg)



还有更复杂的，但是我们不需要进行人工的决定这些filter是什么，因为我们可以通过训练，让机器自己学到这些参数。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v72odj21450mh0v0.jpg)





# padding

padding是填充的意思。

- 我们可以从之前的例子看到，每经过一次卷积运算，图片的像素都会变小，从$6×6 ---> 4×4$，这样子图片就会越来越小，后面就毛都不剩了。

- 还有一点就是，从卷积的运算方法来看，边缘和角落的位置卷积的次数少，会丢失有用信息。



所以就有padding的想法了，也就是在图片四周填补上像素。



![填充后从$6\times6 ->8\times8$，经过$3\times3$卷积后，还是$6\times6$](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v7wyoj20c70avju6.jpg)

计算方法如下，

原数据是$n \times n$，filter为$f \times f$,padding为$p \times p$，

那么得到的矩阵大小是$(n + 2p -f +1)\times(n + 2p -f +1)$



padding有两种：

- valid：也就是不填充
- same：输入与输出大小相同的图片, $p=(f - 1) / 2$，一般padding为奇数，因为filter是奇数





# stride（步长）



卷积的步长也就是每一次运算后平移的距离，之前使用都是stride=1。

假设stride=2，就会得到：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1wj691j20x60if78d.jpg)



得到的矩阵大小是

$$\lfloor \frac{n+2p-f}{s}+1\rfloor \times \lfloor \frac{n+2p-f}{s}+1\rfloor$$

向下取整: 59/60 = 0



# 立体卷积

之前都是单通道的图片进行卷积，如果有RGB三种颜色的话，就要使用立体卷积了。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1v6zxyj211k0lt413.jpg)



这个时候的卷积核就变成了$3 \times 3 \times 3$的三维卷积核，一共27个参数，每次对应着原图片上的RGB一共27个像素运算，然后求和得到输出图片的一个像素。因为只有一个卷积核，这个时候输出的还是$4 \times 4 \times 1$的图片。



**多个卷积核**

因为不同的卷积核可以提取不同的图片特征，所以可以有很多个卷积核，同时提取图片的特征，如分别提取图片的水平和垂直边缘特征。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1vbgemj213t0m6gnu.jpg)

因为有了两个卷积核，这时候输出的图片就是有两通道的图片$4\times 4 \times 2$。

这里要搞清两个概念，卷积核的通道数和个数：

- 通道数channel：即卷积核要作用在原图片上，原图片的通道处$n_c$，卷积核的通道数必须和原图片通道数相同
- 个数：即要使用多少个这样的卷积核，使用$n_{c}^{\prime}$表示，卷积核的个数也就是输出图片的通道数，如有两个卷积核，那么生成了$4\times 4 \times 2$的图片，2  就是卷积核的个数
- 即 $n \times n \times n_c$ ，乘上的$n_{c}^{\prime}$个卷积核 $ f \times f \times n_c$，得到$(n -f +1)\times (n - f +1 ) \times n_{c}^{\prime}$的新图片



# 卷积神经网络

**单层卷积网络**

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1x8wxmj21ba0qd4nz.jpg)

如图是单层卷积的基本过程，先经过两个卷积核，然后再加上bias进行relu激活函数。

那么假设某层卷积层有10个$3 \times 3 \times 3$的卷积核，那么一共有$(3\times3\times3+1) \times10=280$个参数，加1是加上了bias

在这里总结了各个参数的表示方法：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1whr44j20m50bx43e.jpg)



**简单神经网络**

一般卷积神经网络层的类型有：

- convolution卷积层
- pool池化层
- fully connected全连接层



# 池化层



pooling 的作用就是用来压缩数据，加速运算，提高提取特征的鲁棒性



**Max pooling**

在范围内取最大值

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1whjhbj20ra0g3wkp.jpg)



**Average Pooling**

取平均值

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1vp1w0j20pf0cbtaq.jpg)



# 卷积神经网络示例



![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1w3xmij20z00jl76m.jpg)

一般conv后都会进行pooling，所以可以把conv和pooling当做一层。

如上图就是$conv-pool-conv-pool-fc-fc-fc-softmax$的卷积神经网络结构。

各个层的参数是这样的：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrm1vv63zj20lv0bcdjg.jpg)

可以看到，在卷积层的参数非常少，池化层没有参数，大量的参数在全连接层。



# 为何用卷积神经网络？

这里给出了两点主要原因：

- 参数共享：卷积核的参数是原图片中各个像素之间共享的，所以大大减少了参数
- 连接的稀疏性：每个输出值，实际上只取决于很少量的输入而已。

