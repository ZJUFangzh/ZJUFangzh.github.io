---
title: 'DeepLearning.ai笔记:(3-2)-- 机器学习策略(2)(ML strategy)'
id: 2018092017
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-20 17:59:04
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

这周继续讲了机器学习策略,包括误差分析、错误样本清楚、数据分布不同、迁移学习、多任务学习等。

<!--more-->



# 误差分析

对于训练后的模型，如果不进行误差分析，那么很难提升精度。所以应该在验证集中，找到标记错误的那些样本，统计一下都是因为什么原因出现的错误，如是不是照片模糊，还是本来是猫把它标记成狗了等等。



# 清除错误标记样本



如果是随机的误差，也就是人为标记样本出现了随机错误，那么没有关系，因为算法对随即误差还是很有鲁棒性的。

如果是系统误差，那没办法了。



比如说总体误差是10%，然后发现因为人工错误标记引起的误差是0.6%，那么其他原因造成的误差就是9.4%，这个时候应该集中精力去找那9.4%的误差原因，并进行修正。



# 快速搭建系统

对于一个项目来说，我们一开始不要想得太复杂，先快速搭建一个基本的系统，进行迭代，然后在慢慢分析，逐步提高，不要想着一步到位，这样子往往会难以入手。



# 不同分布的训练和测试

假设你在网上找到了20万张照片去分析，但是我们实际上要测试的是用户在手机拍摄情况下的准确度。但是问题是手机上拍摄的数据不足，假设只有1万张。也就是训练集和测试集不是在同一分布，那么怎么办呢？

显然，如果把21万张照片加在一起，重新分配，是不合理的，因为这样子你验证集和测试集上的数据显然很少是手机拍摄的。

所以，应该用20万张照片，再加上5000张照片作为训练集，然后把剩下来的5000张照片对半分为验证集和测试集，那样子才更为符合实际情况。



# 不同分布的偏差和方差

如上述情况，你的训练集和验证测试集不同一分布的，假设training error：1%，dev error：10%，那么这个时候能说是方差太大吗，显然是不合理的，因为不是同一分布的。

那么这个时候应该重新定义一个集合，叫做训练验证集：train-dev

也就是在训练集中拿出一部分数据，跟验证集合在一起，不参与训练，这样我们就得到了：training error：1%，training-dev error：9%，dev error：10%，如果是这种情况，这样才能说是方差问题。

如果是training error：1%，training-dev error：1.5%，dev error：10%，那么，显然不是因为方差问题，而是因为分布不同而导致的。



如何解决呢？

- 进行人工误差分析，看一看训练集和测试集的差别到底在哪里，比如是不是有噪音、照片模糊等等
- 然后把训练集搞得更像测试集，也就是多收集点类似于测试集的数据，或者通过人工合成技术，把噪声加上去。



# 迁移学习

如果我们现在训练了一个猫的分类器，然后这个时候有了新任务，要识别红绿灯，问题是，我们没有那么多红绿灯的照片，没有那么多的数据，那怎么办？这时候就可以把这个猫分类器学习的参数迁移到红绿灯分类器中，只要输出层的微调就行了。因为图像识别的神经网络，在前面的网络大多是进行一些特征提取，所以如果进行图像识别的迁移，还是很有帮助的！



但是迁移学习有限制：

- 必须是相关的类型，比如都是图像识别，都是语音识别
- A的数据远大于B，如果B的数据够多，那自己从头开始学不就好了



# Muti-task多任务学习

假设在自动驾驶中，需要同时检测很多物体，比如人、红绿灯，汽车等等。

那么就可以把这些都写到一个向量中：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrlrv0owzj211i0ky422.jpg)

如图，$y = [0 1 1 0]$即表示同时**有车和停车标志**。

这个又和softmax不同，softmax一次只识别一种物体，而多任务学习一次可以识别多种物体。

这个时候的loss funtion 和logistic是一样的：

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrlruy2ndj20hy0250sn.jpg)

如果在标注样本中，只标注了每张图片的一部分，比如说图片中有行人和车，只标注的行人，有没有车是不知道的，那么可以设为问号$y = [1 0 ? 0]$，这样也是可以训练的，但是在计算loss的时候，要把这个未标记的部分扣除，不要计算在内。



# 端到端学习



假如我们进行公司门禁，需要刷脸进入，那么这时候算法需要分成两步，

- 首先检测到你这个人，然后找到人脸的位置
- 把人脸图像方法，然后在放入模型中计算是否匹配

而端到端学习则直接忽略的这个过程，直接拍一张照片放入模型，输出结果。



再比如说语音识别的时候，在数据少的情况下，我们可能需要

- 提取声音
- 分析语法
- 切分成一个个发声字母
- 组成句子
- 翻译

而端到端学习直接是：提取声音--->翻译

就不需要人为的过多干预了，因为机器可以学到的比人为规定的还要好。



但是注意一点是，需要很大量的数据的时候才能进行端到端学习；如果数据很少，那么还是手动干预，设计一些组件效果会好一点。

