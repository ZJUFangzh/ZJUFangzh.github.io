---
title: 'DeepLearning.ai笔记:(2-3)-- 超参数调试（Hyperparameter tuning）'
id: 2018091720
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-17 20:19:55
---


![](http://peu31tfv4.bkt.clouddn.com/dl.ai1.png)



这周主要讲了这些超参数调试的方法以及batch norm，还有softmax多分类函数的使用。

<!--more-->



# 调试处理

之前提到的超参数有：

- <font color=#FF0000 >$\alpha$</font>
- <font color=#AE8F00 >hidden units</font>
- <font color=#AE8F00 >minibatch size</font>
- <font color=#AE8F00 >$\beta$</font>(Momentum)
- layers
- learning rate decay
- $\beta_1,\beta_2,\epsilon$

颜色代表重要性。



在调参中，常用的方式是在网格中取不同的点，然后计算这些点中的最佳值，

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-1.png)

但是左边是均匀的选点，这样有可能导致在某一个参数上变化很小，浪费计算时间，所以应该更推荐右边的选点方法，即随机选点。



而后，当随机选点选到几个结果比较好的点时，逐步缩小范围，进行更精细的选取。

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-2.png)



# 超参数的合适范围

当然，随机采样并不是在轴上均匀的采样。

比如说$\alpha = 0.001  --- 1$，这样子，那么在$0.1-1$的部分占了90%的概率，显然是不合理的，所以应该将区间对数化，转化成$[0.001,0.01],[0.01,0.1],[0.1,1]$的区间，这样更为合理。思路是：$10^{-3} = 0.001$，所以取值从$[10^{-3},10^{0}]$，我们只要将指数随机就可以了。

```python
r = -3*np.random.rand() # rand()表示在 [0，1]随机取样，再乘以系数，就可以得到[-3,0]
a = 10**r
```

同理,$\beta = 0.9 ,.....,0.999$

通过$1-\beta = 0.1,....,0.001$，所以$1-\beta = 10^{r}$，$\beta = 1-10^{r}$



# 归一化网络的激活函数

我们之前是将输入的数据X归一化，可以加速训练，其实在神经网络中，也可以同样归一化，一般是对$z^{[l]}$归一化。



这个方法叫做 batch norm

公式是：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-3.png)

加上$\epsilon$是为了不至于除以0



而一般标准化后还会加上两个参数，来表示新的方差$\gamma$和均值$\beta$：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-4.png)

$\gamma$和$\beta$也是参数，和$w,b$一样，可以在学习中进行更新。



# 将batch norm 放入神经网络



![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-5.png)



可以看到，

先求的$z^{[1]}$，再进行batch norm，加上参数$\beta^{[1]},\gamma^{[1]}$，得到${\tilde{z}}^{[1]}$,再根据activation function得到$a^{[1]}$

batch norm同样适用于Momentum、RMSprop 、Adam的梯度下降法来进行更新。



# Batch Norm为什么有用？

如果我们的图片中训练的都是黑猫，这个时候给你一些橘猫的图片，那么大概率是训练不好的。因为相当于样本集合的分布改变了，batch norm就可以解决这个问题。





![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-6.png)





如果这个时候要计算第三层，那么很显然计算结果是依赖第二层的数据的。但是如果我们对第二层的数据进行了归一化，那么就可以将第二层的均值和方差都限制在同一分布，而且这两个参数是自动学习的。也就是归一化后的数据可以减弱前层参数的作用与后层参数的作用之间的联系，它使得网络每层都可以自己学习。



还有就是batch norm在某种程度上有正则化的效果，因为归一化会使各个层之间的依赖性降低，而且归一化有带来一定的噪声，有点像dropout。



# 测试集的batch norm

batch norm是在训练集上得到的，那么怎么把它应用在测试集呢？

这个时候可以直接从训练集中拿到$\mu$和$\sigma^{2}$

使用指数加权平均，在每一步中保留$\mu$和$\sigma^{2}$，就可以得到训练后的$\mu$和$\sigma^{2}$



# softmax

之前说的都是二分类问题，如何解决多分类问题呢？

可以用softmax算法来解决。

前面的步骤都一样，而到了最后一层output layer，你想要分为多少类，就用多少个神经元。



这个时候，最后一层的activation function就变成了：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-7.png)

$a^{[l]}_i$就表示了每一个分类的概率。



计算例子如图：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-8.png)



而它的损失函数用的也是cross-entropy：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-9.png)



最终得到一个关于Y的矩阵：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-3-10.png)



其实是可以证明，当分类为2时，softmax就是logistic regression











