---
title: Brief Introduciton of Deep Learning
date: 2018/9/03 13:18:00
tags:
- deep learning
---



## 1. Fully Connect Feedforword Network



![1535627390133](/tmp/1535725823569.png)

 



input layer(input)  -----  hidden layer  --------- output layer  --- output

**Deep = Many hidden layers **

Use Matrix Operation to compute

![1535628199642](/tmp/1535628199642.png)



Output Layer = Muti-class classifier(softmax)

Decide the network structure to let a good funciton in your funciton set



1. Q: How many layers?  How many neurons for each layer?

​       A: Trial and Error + Intuition(直觉), let machine find the best feature

2. Q: Can the structure be auto determined?

   A: Yes



**Loss Function: Use cross entropy**

![1535725902947](/tmp/1535725902947.png)

**Total Loss:**



![1535628956644](/tmp/1535725914899.png)



Use Gradient Descent to find the loss function

**Backpropagation**: an efficient way to compute dL/dw in neural network



![1535629138279](/tmp/1535725942025.png)











## 2. Convolutional Neural Network(CNN)



CNN 是 Fully Connect Feedforword Network 简化版

CNN usually used for **image**



**Why CNN for Image**

- not need to see the whole image to discover the pattern

![1535629804697](/tmp/1535726022077.png)

- same patterns appear in different regions

  ![1535629883697](/tmp/1535726032493.png)

- subsampling the pixels will not change the object

![1535629973112](/tmp/1535726051850.png)









![1535630799292](/tmp/1535726078309.png)



![1535630887333](/tmp/1535726091088.png)





### **Convolution**





![1535631303309](/tmp/1535726113388.png)





![1535726124613](/tmp/1535726124613.png)



The 4 * 4 image as a new image with 2 feature in each pixel, like the rgb color channel





#### **CNN v.s. Fully Connected**



- like the filter connect some feacture,not all feacture 

![1535633570069](/tmp/1535726276531.png)



- less parameters,they shared weights in neurons

  ![1535633701642](/tmp/1535726290079.png)







### **Max Pooling**

only save the max neurons  or  select some top neurons(选一个或多个)

![1535633822693](/tmp/1535726179581.png)









![1535726221527](/tmp/1535726221527.png)

### **Flatten**

![1535634241565](/tmp/1535726238593.png)





**CNN in Keras**



convolution + max pooling

![1535634706353](/tmp/1535726552323.png)



flatten + fully connected feedforword

![1535634775622](/tmp/1535726570206.png)





Look at the First layer pixels's output



![1535635639412](/tmp/1535726621484.png)









