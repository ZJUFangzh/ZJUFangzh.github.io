---
title: Linux安装anaconda
date: 2018-09-05 21:52:53
id: 2018090521
tags:
- Linux
- install
- python
categories:
- 日常技术
- Linux
---
![](http://peu31tfv4.bkt.clouddn.com/ana.jpg)
Anaconda是python的一个很好的发行版，安装了anaconda就可以解决很多python第三方库的问题。

<!--more-->
首先，检查一下电脑中的python版本。

```
$ which python3

/usr/bin/python3
```

此时调用的python3版本在`/usr/bin/`中。

## 1. Download Anaconda

[Download Anaconda](https://www.anaconda.com/download/#linux)

## 2. 安装 Anaconda

这里选择你下载的那个文件（可以用tab自动补全）

```
bash ~/Download/Anaconda3-5.2.0-Linux-x86_64.sh
```

## 3. 添加入path

输入：

```
source ~/.bashrc
```

自动添加完毕。

如果不行，可以手动添加（慎用）

```
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

这个时候，pip已经可以使用了。用`which pip`可以显示在anaconda的pip。

输入 python3，也显示的是anaconda的python3。

这时候如果需要调用系统自带的python

则需要输入

```
sudo python3   # 3.6.5

#或者

sudo python   # 2.7


```


具体可以查看[anaconda的使用帮助](http://docs.anaconda.com/anaconda/install/linux/)。