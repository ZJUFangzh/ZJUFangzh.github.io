---
title: 'DeepLearning.ai笔记:(2-2)-- 优化算法（Optimization algorithms）'
id: 2018091621
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-09-16 21:42:33
---


![](http://peu31tfv4.bkt.clouddn.com/dl.ai1.png)



这周学习了优化算法，可以让神经网络运行的更快。

<!--more-->

主要有:

- mini-batch
- 动量梯度下降(momentum)
- RMSprop
- Adam优化算法
- 学习率衰减



# mini-batch(小批量)

原本的梯度下降算法，在每一次的迭代中，要把所有的数据都进行计算再取平均，那如果你的数据量特别大的话，每进行一次迭代就会耗费大量的时间。

所以就有了mini-batch，做小批量的计算迭代。也就是把训练集划分成n等分，比如数据量有500万个的时候，以1000为单位，将数据集划分为5000份，
$$x =  {x^{\lbrace 1 \rbrace},x^{\lbrace 2 \rbrace},x^{\lbrace 3 \rbrace},.....,x^{\lbrace 5000 \rbrace}}$$

用大括弧表示每一份的mini-batch，其中每一份$x^{\lbrace t \rbrace}$都是1000个样本。

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-1.png)

这个时候引入epoch的概念，1个epoch相当于是遍历了一次数据集，比如用mini-batch，1个epoch就可以进行5000次迭代，而传统的batch把数据集都一起计算，相当于1个epoch只进行了1次迭代。

具体计算步骤是：

- 先划分好每一个mini-batch
- `for t in range(5000)`，循环每次迭代
  - 循环里面和之前的计算过程一样，前向传播，但每次计算量是1000个样本
  - 计算损失函数
  - 反向传播
  - 更新参数



batch和mini-batch的对比如图：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-2.png)



- 如果mini-batch的样本为m的话，其实就是**batch gradient descent**，缺点是如果样本量太大的话，每一次迭代的时间会比较长，但是优点是每一次迭代的损失函数都是下降的，比较平稳。
- mini-batch样本为1的话，那就是**随机梯度下降（Stochastic gradient descent）**,也就是每次迭代只选择其中一个样本进行迭代，但是这样会失去了样本向量化带来的计算加速效果，损失函数总体是下降的，但是局部会很抖动，很可能无法达到全局最小点。
- 所以选择一个合适的size很重要，$1 < size < m$，可以实现快速的计算效果，也能够享受向量化带来的加速。

![三种下降对比，蓝色为batch，紫色为Stochastic，绿色为mini-batch](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-3.png)

**mini-batch size的选择**

因为电脑的内存和使用方式都是二进制的，而且是2的n次方，所以之前选1000也不太合理，可以选1024，但是1024也比较少见，一般是从64到512。也就是$64、128、256、512$



# 指数加权平均(Exponentially weighted averages )



![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-4.png)



蓝色的点是每一天的气温，可以看到是非常抖动的，那如果可以把它平均一下，比如把10天内的气温平均一下，就可以得到如红色的曲线。

但是如果是单纯的把前面的10天气温一起平均的话，那么这样你就需要把前10天的气温全部储存记录下来，这样子虽然会更准一点，但是很浪费储存空间，所以就有了**指数加权平均**这样的概念。方法如下：

$$V_0 = 0$$

$$V_1 = \beta * V_0 + (1 - \beta) \theta_1$$

$……$

$$V_t = \beta * V_{t-1} + (1 - \beta) \theta_t$$

其中，$\theta_t$表示第t天的温度，而$V_t$表示指数加权平均后的第t天温度，$\beta$这个参数表示$\frac{1}{1-\beta}$天的平均，也就是，$\beta = 0.9$，表示10天内的平均，$\beta = 0.98$，表示50天内的平均。

![黄、红、绿线依次表示$\beta = 0.5,0.9,0.98，即2、10、50天的平均$](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-5.png)



# 理解指数加权平均

我们再来看一下公式：

$$v_t = \beta v_{t-1} + (1 - \beta) \theta_t$$

假设$\beta = 0.9$，那么

$$v_{100} = 0.9v_{99} + 0.1\theta_{100}$$

$$v_{99} = 0.9v_{98} + 0.1\theta_{99}$$

$$v_{98} = 0.9v_{97} + 0.1\theta_{98}$$

展开一下，得到：

$$ v_{100} = 0.1 \theta_{100} + 0.1 \times 0.9 \times \theta_{99} +  0.1 \times 0.9^2  \times \theta_{98} + ......$$

看到没有，每一项都会乘以0.9，这样就是指数加权的意思了，那么为什么表示的是10天内的平均值呢？明明是10天以前的数据都有加进去的才对，其实是因为$0.9^{10} \approx 0.35 \approx \frac{1}{e}$，也就是10天以前的权重只占了三分之一左右，已经很小了，所以我们就可以认为这个权重就是10天内的温度平均，其实有详细的数学证明的，这里就不要证明了，反正理解了$(1-\epsilon)^{\frac{1}{\epsilon}} \approx \frac{1}{e}$，$\epsilon$为0.02的时候，就代表了50天内的数据。

因为指数加权平均不需要知道前面n个数据，只要一步一步进行迭代，知道当前的数据就行，所以非常节省空间。

# 指数加权平均的偏差修正

如果你细心一点，你就会发现其实这个公式有问题，

$$V_0 = 0$$

$$V_1 = \beta * V_0 + (1 - \beta) \theta_1$$

$……$

$$V_t = \beta * V_{t-1} + (1 - \beta) \theta_t$$

如果第一天的温度是40摄氏度，那么$V_1 = 0.1 * 40 = 4$，显然是不合理的。因为初始值$V_0 = 0$，也就是前面几天的数据都会普遍偏低。所以特别是在估测初期，需要进行一些修正，这个时候就不要用$v_t$了，而是用$\frac{v_t}{1-\beta^t}$来代表第t天的温度平均，你会发现随着t的增加，$\beta^t$接近于0，所以偏差修正几乎就没有用了，而t比较小的时候，就非常有效果。

![紫色线为修正前，绿色线为修正后的效果](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-6.png)



不过在大部分机器学习中，一般也不需要修正，因为只是前面的初始时期比较有偏差而已，到后面就基本不会有偏差了，所以也不太用。



# 动量梯度下降法 (Gradient descent with Momentum )

用动量梯度下降法运行速度总是比标准的梯度下降法要来的快。它的基本思想是计算梯度的指数加权平均数，然后用该梯度来更新权重。



效果如图：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-7.png)



使用动量梯度下降法后，在竖直方向上的抖动减少了，而在水平方向上的运动反而加速了。

算法公式：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-8.png)



可以发现，就是根据指数平均计算出了$v_{dW}$，然后更新参数时把$dW$换成了$v_{dw}$，$\beta$一般的取值是0.9。可以发现，在纵向的波动经过平均以后，变得非常小了，而因为在横向上，每一次的微分分量都是指向低点，所以平均后的值一直朝着低点前进。

物理意义：

- 个人的理解是大概这个公式也很像动量的公式$m v = m_1 v_1 + m_2 v_2$，也就是把两个物体合并了得到新物体的质量和速度的意思
- 理解成速度和加速度，把$v_{dW}$看成速度，$dW$看成加速度，这样每次因为有速度的存在，加速度只能影响到速度的大小而不能够立刻改变速度的方向。



# RMSprop（root mean square prop）

均方根传播。这是另一种梯度下降的优化算法。

顾名思义，先平方再开根号。

其实和动量梯度下降法公式差不多：

![](http://pexm7md4m.bkt.clouddn.com/dl-ai-2-2-9.png)

在更新参数的分母项加了一项$\epsilon = 10^{-8}$,来确保算法不会除以0



# Adam算法

Adam算法其实就是结合了Momentum和RMSprop ，注意这个时候要加上偏差修正：

- 初始化参数：$v_{dW} = 0$，$S_{dW} =0$，$v_{db} = 0$，$S_{db} =0$
- 在第$t$次迭代中，
  - 计算mini-batch的dW,db
  - Momentum: $v_{dW}= \beta_{1}v_{dW} + ( 1 - \beta_{1})dW$，$v_{db}= \beta_{1}v_{db} + ( 1 -\beta_{1} ){db}$
  - RMSprop:$S_{dW}=\beta_{2}S_{dW} + ( 1 - \beta_{2}){(dW)}^{2}$，$S_{db} =\beta_{2}S_{db} + \left( 1 - \beta_{2} \right){(db)}^{2}$
  - $v_{dW}^{\text{corrected}}= \frac{v_{dW}}{1 - \beta_{1}^{t}}$，$v_{db}^{\text{corrected}} =\frac{v_{db}}{1 -\beta_{1}^{t}}$
  - $S_{dW}^{\text{corrected}} =\frac{S_{dW}}{1 - \beta_{2}^{t}}$，$S_{db}^{\text{corrected}} =\frac{S_{db}}{1 - \beta_{2}^{t}}$
  - $W:= W - \frac{a v_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}} +\varepsilon}$



超参数有$\alpha,\beta_1,\beta_2,\epsilon$，一般$\beta_1 = 0.9,\beta_2 = 0.999,\epsilon = 10^{-8}$



# 学习率衰减



在梯度下降时，如果是固定的学习率$\alpha$，在到达最小值附近的时候，可能不会精确收敛，会很抖动，因此很难达到最小值，所以可以考虑学习率衰减，在迭代过程中，逐渐减小$\alpha$，这样一开始比较快，后来慢慢的变慢。

常用的是：

$$a= \frac{1}{1 + decayrate * \text{epoch_num}} a_{0}$$

$$a =\frac{k}{\sqrt{\text{epoch_num}}}a_{0}$$

$$a =\frac{k}{\sqrt{t}}a_{0}$$



