---
title: 'DeepLearning.ai笔记:(5-1)-- 循环神经网络（Recurrent Neural Networks）'
id: dl-ai-5-1
tags:
  - dl.ai
categories:
  - AI
  - Deep Learning
date: 2018-10-18 10:26:52
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

第五门课讲的是序列模型，主要是对RNN算法的应用，如GRU，LSTM算法，应用在词嵌入模型，情感分类，语音识别等领域。

第一周讲的是RNN的基本算法。

<!--more-->

# 序列模型的应用

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokszq4j20og0dkq54.jpg)

序列模型用在了很多的地方，如语音识别，音乐生成，情感分类，DNA序列分析，机器翻译，视频内容检测，名字检测等等。



# 数学符号

先讲一下NG在课程中主要用到的数学符号。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokftypj20o507kwes.jpg)

对于输入一个$x$的句子序列，可以细分为一个个的词，每一个词记为$x^{<t\>}$，对应的输出$y$记为$y^{<t\>}$

其中，输入x的序列长度为 $T_x$，输出$y$的序列长度为$T_y$

而针对很多个不同的序列，$X^{(i)<t\>}$表示第$i$个样本的第t的词。

那么如何用数学的形式表示这个$x^{<t\>}$呢？这里用到了one-hot编码，假设词表中一共有10000个词汇，那么$x^{<t\>}$就是一个长度为10000的向量，在这之中只有一个维度是1，其他都是0

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokgq4pj20o80dkjsc.jpg)



# 循环神经网络

如果用传统的神经网络，经过一个N层的神经网络得到输出y。

效果并不是很好，因为：

- 输入和输出在不同的样本中是可以不同长度的（每个句子可以有不同的长度）
- 这种朴素的神经网络结果并不能共享从文本不同位置所学习到的特征。（如卷积神经网络中学到的特征的快速地推广到图片其他位置）

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokgsxfj20n40cfdgj.jpg)



所以循环神经网络采用每一个时间步来计算，输入一个$x^{<t\>}$和前面留下来的记忆$a^{<t-1\>}$，来得到这一层的输出$y^{<t\>}$和下一层的记忆$a^{<t\>}$

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokk8tvj20n108s0tp.jpg)

这里需要注意在零时刻，我们需要编造一个激活值，通常输入一个零向量，有的研究人员会使用随机的方法对该初始激活向量进行初始化。同时，上图中右边的循环神经网络的绘制结构与左边是等价的。

循环神经网络是从左到右扫描数据的，同时共享每个时间步的参数。

- $W_{ax}$管理从输入$x^{<t\>}$到隐藏层的连接，每个时间步都使用相同的$W_{ax}$，同下；
- $W_{aa}$管理激活值$a^{<t\>}$到隐藏层的连接；
- $W_{ya}$管理隐藏层到激活值$y^{<t\>}$的连接。



**RNN的前向传播**

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokk24pj20np0ckmya.jpg)

前向传播公式如图，这里可以把$W_{aa}，W_{ax}$合并成一项，为$W_a$，而后将$[a^{<t-1\>},x^{<t\>}]$合并成一项。



**RNN的反向传播**

定义一个loss function，然后倒回去计算。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgoksqpcj20nf0cxdhj.jpg)



# 不同类型的RNN

对于RNN，不同的问题需要不同的输入输出结构。

- One to many：如音乐生成，输入一个音乐类型或者空值，生成一段音乐
- Many to one：如情感分类问题，输入某个序列，输出一个值来判断得分。
- many to many（$T_x = T_y$）：输入和输出的序列长度相同
- many to many（$T_x != T_y$）：如机器翻译这种，先输入一段，然后自己生成一段，输入和输出长度不一定相同的。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokh75rj20nm0d274y.jpg)



# 语言模型和序列生成

**什么是语言模型？**

对于下面的例子，两句话有相似的发音，但是想表达的意义和正确性却不相同，如何让我们的构建的语音识别系统能够输出正确地给出想要的输出。也就是对于语言模型来说，从输入的句子中，评估各个句子中各个单词出现的可能性，进而给出整个句子出现的可能性。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokqwy9j20ln0b83yw.jpg)

 **使用RNN构建语言模型：**

- 训练集：一个很大的语言文本语料库；
- Tokenize：将句子使用字典库标记化；其中，未出现在字典库中的词使用“UNK”来表示；
- 第一步：使用零向量对输出进行预测，即预测第一个单词是某个单词的可能性；
- 第二步：通过前面的输入，逐步预测后面一个单词出现的概率；

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokk6ckj20n70cx75i.jpg)



# 对新序列采样

当我们训练得到了一个模型之后，如果我们想知道这个模型学到了些什么，一个非正式的方法就是对新序列进行采样。具体方法如下：

在每一步输出$y$时，通常使用 softmax 作为激活函数，然后根据输出的分布，随机选择一个值，也就是对应的一个字或者英文单词。

然后将这个值作为下一个单元的x输入进去(即$x^{<t\>}=y^{<t−1\>}$), 直到我们输出了终结符，或者输出长度超过了提前的预设值n才停止采样。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokk4rpj20ic063jre.jpg)



# RNN的梯度消失

RNN存在一个梯度消失问题，如：

- The cat, which already ate ………..，was full；
- The cats, which already ate ………..，were full.

cat 和 cats要经过很长的一系列词汇后，才对应 was 和 were，但是我们在传递过程中$a^{<t\>}$很难记住前面这么多词汇的内容，往往只和前面最近几个词汇有关而已。

当然，也有可能是每一层的梯度都很大，导致的梯度爆炸问题，不过这个问题可以通过设置阈值来解决，关键是要解决梯度消失问题。我们知道一旦神经网络层次很多时，反向传播很难影响前面层次的参数。



# GRU(Gated Recurrent Unit)

那么如何解决梯度消失问题了，使用GRU单元可以有效的捕捉到更深层次的连接，来改善梯度消失问题。

原本的RNN单元如图：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokkehij20m00aiglw.jpg)



而GRU单元多了一个c（memory cell）变量，用来提供长期的记忆能力。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgoktgchj20n20cwjtf.jpg)

具体过程为：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokn137j20pg04m0t4.jpg)

完整的GRU还存在另一个门，用来控制$\bar c$和 $c^{<t-1\>}$之间的联系强弱：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokn4qrj20dp06maaj.jpg)



# LSTM(Long short-term memory)

GRU能够让我们在序列中学习到更深的联系，长短期记忆（long short-term memory, LSTM）对捕捉序列中更深层次的联系要比GRU更加有效。

GRU只有两个门，而LSTM有三个门，分别是更新门、遗忘门、输出门：$\Gamma_u,\Gamma_f, \Gamma_o$

![GRU和LSTM公式对比](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgoko16mj20n50c174n.jpg)

更新门：用来决定是否更新$\bar c^{<t\>}$

遗忘门：来决定是否遗忘上一个$c^{<t-1\>}$

输出门：来决定是否输出$c^{<t\>}$

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgolrcslj20wl0idjth.jpg)



# 双向RNN

双向RNN（bidirectional RNNs）模型能够让我们在序列的某处，不仅可以获取之间的信息，还可以获取未来的信息。

对于下图的单向RNN的例子中，无论我们的RNN单元是基本的RNN单元，还是GRU，或者LSTM单元，对于例子中第三个单词”Teddy”很难判断是否是人名，仅仅使用前面的两个单词是不够的，需要后面的信息来进行判断，但是单向RNN就无法实现获取未来的信息。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokp4z4j20mn0boaas.jpg)



而双向RNN则可以解决单向RNN存在的弊端。在BRNN中，不仅有从左向右的前向连接层，还存在一个从右向左的反向连接层。

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgolueabj20n20cwq4i.jpg)



# Deep RNN

与深层的基本神经网络结构相似，深层RNNs模型具有多层的循环结构，但不同的是，在传统的神经网络中，我们可能会拥有很多层，几十层上百层，但是对与RNN来说，三层的网络结构就已经很多了，因为RNN存在时间的维度，所以其结构已经足够的庞大。如下图所示：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgokxta0j20mu0cuab9.jpg)

