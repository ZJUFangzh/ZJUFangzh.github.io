---
title: 'DeepLearning.ai笔记:(2-1)-- 深度学习的实践层面（Practical aspects of Deep Learning）'
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-15 13:37:15
id: 20180901513
---

![](http://peu31tfv4.bkt.clouddn.com/dl.ai1.png)

第二门课主要讲的是如何改善神经网络，通过超参数的调试、正则化以及优化。



第一周主要是说了一些之前机器学习里面涉及到的数据集的划分，以及初始化，正则化的方法，还有梯度的验证。

<!--more-->

# 训练、验证、测试集的划分

这些在之前的机器学习课程中都讲过了，这里简单说一下。

训练集也就是你训练的样本；验证集是你训练之后的参数放到这些数据中做验证；而最后做的测试集则是相当于用来最终的测试。



一般来说，划分比例为60%/20%/20%就可以了，但是当数据越来越大，变成上百万，上千万的时候，那么验证集和测试集就没必要占那么大比重了，因为太过浪费，一般在0.5%-3%左右就可以。



需要注意的是，验证集和测试集的数据要来源相同，同分布，也就是同一类的数据，不能验证集是网上的，测试集是你自己拍的照片，这样误差会很大。



# bias and variance（偏差和方差）



![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-1.png)



high bias 表示的是高偏差，一般出现在欠拟合(under fitting)的情况下，

high variance表示高方差，一般出现在overfitting情况下。



如何解决呢：

- high bias
  - 更多的隐藏层
  - 每一层更多的神经元
- high variance
  - 增加数据
  - 正则化



![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-2.png)



从左到右4种情况即是： high variance ;  high bias ; high bias and high variance ; low bias and low variance



# regularization（正则化）



high variance可以使用正则化来解决。

我们知道，在logistic regression中的正则化项，是在损失函数后面加上：

L2 正则：$\frac{\lambda}{2m}||w||^{2}_{2} = \frac{\lambda}{2m}\sum_{j=1}^{n_{x}}{|w|} =  \frac{\lambda}{2m} w^T w$

L1正则：$\frac{\lambda}{2m}||w||_{1} = \frac{\lambda}{2m}\sum_{j=1}^{n_{x}}{|w|}$



一般用L2正则来做。



在neural network中，

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-3.png)



可以看到后面的正则式是从第1层累加到了第L层的所有神经网络的权重$||W^{[l]}||_{F}$的平方。

而我们知道这个W是一个$n^{[l]} * n^{[l-1]}$的矩阵，那么

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-4.png)

它表示矩阵中所有元素的平方和。也就这一项嵌套了3层的$\sum$。



那么，如何实现这个范数的梯度下降呢？



在原本的backprop中,加上的正则项的导数，$dJ / dW$

$$dW^{[l]} = (form backprop) + \frac{\lambda}{m}W^{[l]}$$

代入

$$W^{[l]} = W^{[l]} - \alpha dW^{[l]}$$

得到：

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-5.png)

可以看到，$(1 - \frac{\alpha \lambda}{m}) < 1$，所以每一次都会让W变小，因此L2范数正则化也成为“权重衰减”



## 正则化如何防止过拟合？

直观理解是在代价函数加入正则项后，如果$\lambda$非常大，为了满足代价函数最小化，那么$w^{[l]}$这一项必须非常接近于0，所以就等价于很多神经元都没有作用了，从原本的非线性结构变成了近似的线性结构，自然就不会过拟合了。

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-6.png)



我们再来直观感受一下，



![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-7.png)



假设是一个tanh()函数，那么$z = wx + b$，当w非常接近于0时，z也接近于0，也就是在坐标轴上0附近范围内，这个时候斜率接近于线性，那么整个神经网络也非常接近于线性的网络，那么就不会发生过拟合了。





## dropout 正则化



dropout(随机失活)，也是正则化的一种，顾名思义，是让神经网络中的神经元按照一定的概率随机失活。

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-8.png)

**实现dropout：inverted dropout（反向随机失活）**

实现dropout有好几种，但是最常用的还是这个inverted dropout

假设是一个3层的神经网络，keepprob表示保留节点的概率

```python
keepprob = 0.8
#d3是矩阵，每个元素有true,false,在python中代表1和0
d3 = np.random.rand(a3.shape[0],a3.shape[1]) < keepprob
a3 = np.multiply(a3,d3)
a3 /= keepprob
```

其中第4式 $a3 /= keepprob$

假设第三层有50个神经元 a3.shape[0] = 50，一共有 $50 * m$维，m是样本数，这样子就会有平均10个神经元被删除，因为$z^{[4]} = w^{[4]} a^{[3]} + b^{[4]}$，那么这个时候$z^{[4]}$的期望值就少了20%,所以在每个神经元上都除以keepprob的值，刚好弥补的之前的损失。



**注意**

在test阶段，就不需要再使用dropout了，而是像之前一样，直接乘以各个层的权重，得出预测值就可以。



## 理解dropout



直观上，因为神经元有可能会被随机清除，这样子在训练中，就不会过分依赖某一个神经元或者特征的权重。



当然可以设置不同层有不同的dropout概率。



计算机视觉领域非常喜欢用这个dropout。



但是这个东西的一大缺点就是代价函数J不能再被明确定义，每次都会随机移除一些节点，所以很难进行复查。如果需要调试的话，通常会关闭dropout，设置为1，这样再来debug。





# 归一化



归一化数据可以加速神经网络的训练速度。

一般有两个步骤：

- 零均值
- 归一化方差



![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-9.png)



这样子在gradient的时候就会走的顺畅一点：

![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-10.png)







# 参数初始化



合理的参数初始化可以有效的加快神经网络的训练速度。

一般呢$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n$，一般希望z不要太大也不要太小。所以呢，希望n越大，w越小才好。最合理的就是方差 $w = \frac{1}{n}$，所以：

```
WL = np.random.randn(WL.shape[0],WL.shape[1])* np.sqrt(1/n)
```

这个$n$即$n^{[l-1]}$



如果是relu函数，

那么 $w = \frac{2}{n}$比较好，也就是`np.sqrt(2/n)`



# 梯度的数值逼近

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$

微积分的常识，用$\varepsilon$来逼近梯度。



# 梯度检验



用梯度检验可以来检查在反向传播中的算法有没有错误。

这个时候，可以把$W^{[1]},b^{[1]},......W^{[l]},b^{[l]}$变成一个向量，这样可以得到一个代价函数$J(\theta)$，然后$dW,db$也可以转换成一个向量，用$d\theta$表示，和$\theta$有相同的维度。



![](http://peu31tfv4.bkt.clouddn.com/dl-ai-2-4-11.png)

再对每一个$d\theta_{approx}[i]$求上面的双边梯度逼近，然后也用导数求得每一个$d\theta[i]$，然后根据图上的cheak公式。求梯度逼近的时候，设置两边的$\varepsilon = 10^{-7}$，最终求得的值如果是$10^{-7}$，那么很正常，$10^{-3}$就是错了的，如果是$10^{-5}$，那么就需要斟酌一下了。

**注意**

- 不要在训练中用梯度检验，因为很慢
- 如果发现有问题，那么定位到误差比较大的那一层查看
- 如果有正则化，记得加入正则项
- 不要和dropout一起使用，因为dropout本来就不容易计算梯度。







