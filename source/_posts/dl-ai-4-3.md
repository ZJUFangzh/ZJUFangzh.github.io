---
title: 'DeepLearning.ai笔记:(4-3)-- 目标检测（Object detection）'
id: dl-ai-4-3
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-10-11 17:02:09
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

这一周主要讲了卷积神经网络的进一步应用：目标检测。

主要内容有：目标定位、特征点检测、目标检测、滑动窗口、Bounding Box，IOU，NMS，Anchor Boxes，Yolo算法。

<!--more-->

# 目标定位（Object localization）

在进行目标检测之前，需要先了解一下目标定位。

因为进行目标检测的时候需要预测出目标的具体位置，所以在训练的时候需要先标定一下这个目标的实际位置。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktch0hzj20ot0e1juw.jpg)

假设我们需要分类的类别有3个，行人，汽车，自行车，如果什么都没有，那么就是背景。可以看到，y一共有8个参数：

- $P_c$：是否有目标
- $b_x,b_y,b_h,b_w$：目标的位置x,y，高宽h,w
- $c_1,c_2,c_3$：行人、汽车、自行车

如果$P_c = 0$那么表示没有目标，那么我们就不关心后面的其他参数了。



# 特征点检测(Landmark detection)

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktegta7j20p50e0tga.jpg)

如果是要检测人脸，那么可以在人脸上标注若干个特征点，假设有64个特征点，那么这个时候就有128个参数了，再加上判断是否有人脸，就有129个参数。

假设要检测的是人体肢体的动作，那么同样也可以标注若干个肢体上的特征点。

注意，这些都是需要**人工进行标注**的。



# 目标检测

**滑动窗口**

目标检测通常采用的是滑动窗口的方法来检测的。也就是用一定窗口的大小，按照指定的步长，遍历整个图像；而后再选取更大的窗口，再次遍历，依次循环。这样子，每个窗口都相当于一张小图片，对这个小图片进行图像识别，从而得到预测结果。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktcgv82j20oo0dy0wz.jpg)



但是这个方法有个很明显的问题，就是每个窗口都要进行一次图像识别，速度太慢了。因此就有了滑动窗口的卷积实现。在此之前，我们需要知道如何把全连接层变为卷积层。



**全连接层转化为卷积层**

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb7cdvj20ov0dx0tx.jpg)

如图，在经过Max pool后，我们得到了$5 \times 5 \times 16$的图像，经过第一个FC层后变成了400个节点。

而此时我们可以使用400个$5\times5$的卷积核，进行卷积，得到了$1\times1\times400$

而后再使用400个$1\times1$的卷积核，再得到了$1\times1\times400$矩阵，所以我们就将全连接层转化成了卷积层。



**卷积滑动窗口的实现**

因为之前的滑动窗口每一次都要进行一次计算，太慢了。所以利用上面的全连接层转化为卷积层的做法，可以一次性把滑动窗口的结果都计算出来。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb94djj20ns09k0vh.jpg)

为了方面观察，这里把三维图像画成了平面。

假设滑动的窗口是$14\times14\times3$，原图像大小是$16\times\times16\times3$。

蓝色表示滑动窗口，如果步数是2的话，很容易可以得到$2\times2$的图像，不难看出，在图中最后输出的左上角的蓝色部分就是原图中蓝色部分的计算结果，以此类推。

也就是说，只需要原图进行一次运算，就可以一次性得到多个滑动窗口的输出值。



具体例子如下图：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb8z1tj20no0cdwhr.jpg)

可以看到，原图为$28\times28$，最后得到了$8\times8 = 64$个滑动窗口。



# Bounding Box

上面介绍的滑动窗口的方法有一个问题，就是很多情况下并不能检测出窗口的精确位置。

那么如何找到这个准确的边界框呢？有一个很快的算法叫做YOLO(you only look once)，只需要计算一次便可以检测出物体的位置。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktcya16j20p10btq8q.jpg)

如图，首先，将图片分为$n \times n$个部分，如图是划分成了$3\times3=9$份，而每一份都由一个向量y来表示。

因此最终得到了$3\times3\times8$的矩阵。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktblfnuj20h70cg414.jpg)

要得到这个$3\times3\times8$的矩阵，只要选择适当的卷积神经网络，让输出矩阵为这个矩阵就行，而每一个小图像都有一个目标标签y，这个时候y中的$b_x,b_y$都是这个小图像的相对位置，在0-1之间，而$b_h,b_w$是可以大于1的，因为整个大目标有可能在框框外。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb8qxxj20v606i0v4.jpg)





在实际过程中可以选用更精细的划分，如$19\times19$。



# 交并比(Intersection over Union, IoU)

如何判断框框是否正确呢？

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4kte96wxj20ay0at7af.jpg)

如图红色为车身的框，而紫色为检测到的框，那么紫色的框到底算不算有车呢？

这个时候可以用交并比来判断，也就是两个框框的交集和并集之比。

$$IoU = \frac{交集面积}{并集面积}$$

一般来说 $IoU  \geq 0.5$，那么说明检测正确，当然，这个阈值可以自己设定。



# 非最大值抑制（NMS）

在实际过程中，很可能很多个框框都检测出同一个物体，那么如何判断这些边界框检测的是同一个对象呢？

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktdewjvj20a6088tdv.jpg)

- 首先，每一个框都会返回一个概率$P_c$，我们需要先去掉那些概率比较低的框，如去掉$P_c \leq 0.55$的框。
- 而后，在$P_c$中找到概率最大的框，然后用算法遍历其他的边框，找出并取消掉和这个边框IoU大于0.5的框（因为如果IoU大于0.5，我们可以认为是同一个物体）
- 循环第二步的操作



如果有多个目标类别的检测，那么对每个类别分别进行上面的NMS算法。



# Anchor Box

如果一张格子中有多个目标，那怎么办？这时候就需要Anchor Box了，可以同时检测出多个对象。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktbwaixj2064065mzy.jpg)

我们预先定义了两个不同形状的Anchor box，如比较高的来检测人，比较宽的来检测汽车，然后重新定义了目标向量y：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb6okuj20e50bmq38.jpg)





这个时候最后输出的矩阵从原来的$3\times3\times8$变成了$3\times3\times16$，也可以是$3\times3\times2\times8$

在计算的时候就可以根据不同的box输出了，？号表示我们不关系这个值。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktb77nnj20dn0dtmxs.jpg)

**问题：**

- 如果使用两个Box，那么如果出现3个目标怎么办，这时候需要别的手段了
- 如果同一个格子的两个对象的box相同怎么办，那也需要别的手段来处理了。

因为这两种情况出现的几率都比较少，所以问题不大。



**注意：**

- Anchor box的形状都是人工指定的，一般可以选择5-10种不同的形状，来涵盖我们想要检测的不同对象
- 更高级一点的使用k-means聚类算法，将不同的对象形状进行聚类，然后得到一组比较具有代表性的boxes



# YOLO算法

假设我们需要检测三种目标：行人、汽车、摩托车，使用两种anchor box

 **在训练集中：**

- 输入同样大小的图片X
- 每张图片的输出Y是$3\times3\times16$的矩阵
- 人工标定输出Y

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktdbjbij20om0de42t.jpg)



**预测：**

输入图片和训练集大小相同，得到$3\times3\times16$的输出结果

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktc1dooj20p00cjadn.jpg)

这个时候得到了很多个框框，如果是两个Anchor box，那么就有$2\times9=18$个预测框框，那么就需要把没用的框框都去掉，也就用到了上面的NMS非最大值抑制算法。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktd13lnj208v08rn2c.jpg)



**进行NMS算法：**

- 去掉$P_c$小于某个阈值的框框
- 对于每个对象分别使用NMS算法得到最终的边界框。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktdzoa1j209z09w461.jpg)





# 候选区域

这里还介绍了其他的目标检测算法，不过貌似都是比较慢的。

**R-CNN：**

原本的滑动窗口，只有在少部分的区域是可以检测到目标的，很多区域都是背景，所以运算很慢，用R-CNN后，只选择一些候选的窗口，不需要对整个图片进行滑动。

R-CNN使用的是图像分割算法，将图片分割成很多个色块，从而减少了窗口数量。

是对每个候选区域进行分类，输出的标签和bounding box

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fw4ktfhue7j20xn08lwo3.jpg)



**Fast R-CNN：**

候选区域，使用滑动窗口在区分所有的候选区域。

**Faster R-CNN：**

使用卷积神经网络而不是图像分割来获得候选区域。