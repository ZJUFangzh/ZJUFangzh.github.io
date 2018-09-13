---
title: 'DeepLearning.ai笔记:(1-4)-- 深层神经网络（Deep neural networks）'
id: 2018091316
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-13 16:54:18
---


![](http://peu31tfv4.bkt.clouddn.com/dl.ai1.png)

这一周主要讲了深层的神经网络搭建。

<!--more-->



# 深层神经网络的符号表示

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-1.jpg)



在深层的神经网络中，

- $L$表示神经网络的层数 $L = 4$
- $n^{[l]}$表示第$l$层的神经网络个数
- $W^{[l]}: (n^{[l]},n^{l-1})$
- $dW^{[l]}: (n^{[l]},n^{l-1})$
- $b^{[l]}: (n^{[l]},1)$
- $db^{[l]}: (n^{[l]},1)$
- $z^{[l]}:(n^{[l]},1)$
- $a^{[l]}:(n^{[l]},1)$



# 前向传播和反向传播



**前向传播**

input $a^{[l-1]}$

output $a^{[l]},cache (z^{[l]})$ ，其中cache也顺便把 $W^{[l]},  b^{[l]}$也保存下来了

所以，前向传播的公式可以写作：

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$



**维度**

假设有m个样本，那么$Z^{[l]}$ 维度就是 $(n^{[l]}, m)$ ，$A^{[l]}$的维度和$Z^{[l]}$一样。

那么 $ W^{[l]} A^{[l-1]}$维度就是 $(n^{[l]},n^{l-1})  *  (n^{[l-1]},m)$  也就是  $(n^{[l]}, m)$，这个时候，还需要加上$b^{[l]}$，而$b^{[l]}$本身的维度是$(n^{[l]},1)$，借助python的广播，扩充到了m个维度。



**反向传播**

input $da^{[l]}$

output $da^{[l-1]} , dW^{[l]} , db^{[l]}$

公式：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-5.jpg)

向量化：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-6.jpg)



正向传播和反向传播如图：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-2.jpg)





具体过程为，第一层和第二层用Relu函数，第三层输出用sigmoid，这个时候的输出值是$a^{[3]}$

而首先进行反向传播的时候先求得$da^{[3]} = - \frac{y}{a} - \frac{1-y}{1-a}$，然后再包括之前存在cache里面的$z^{[3]}$,反向传播可以得到$dw^{[3]}, db^{[3]},da^{[2]}$，然后继续反向，知道得到了$dw^{[1]},db^{[1]}$后，更新一下w，b的参数，然后继续做前向传播、反向传播，不断循环。



# Why Deep？



![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-3.jpg)

如图直观上感觉，比如第一层，它会先识别出一些边缘信息；第二层则将这些边缘进行整合，得到一些五官信息，如眼睛、嘴巴等；到了第三层，就可以将这些信息整合起来，输出一张人脸了。

如果网络层数不够深的话，可以组合的情况就很少，或者需要类似门电路那样，用单层很多个特征才能得到和深层神经网络类似的效果。

# 搭建深层神经网络块



![](http://pexm7md4m.bkt.clouddn.com/dl-ai-1-4-4.jpg)



和之前说的一样，一个网络块中包含了前向传播和反向传播。

前向输入$a^{[l-1]}$，经过神经网络的计算，$g^{[l]}(w^{[l]}a^{[l-1]} + b^{[l]})$得到$a^{[l]}$

反向传播，输入$da^{[l]}$，再有之前在cache的$z^{[l]}$,即可得到$dw^{[l]},db^{[l]}$还有上一层的$da^{[l-1]}$



# 参数与超参数



超参数就是你自己调的，玄学参数：

- learning_rate
- iterations
- L = len(hidden layer)
- $n^{[l]}$
- activation function
- mini batch size（最小的计算批）
- regularization（正则）
- momentum（动量）