---
title: Linux安装selenium+chromedriver
date: 2018-09-05 21:51:41
id: 2018090520
tags: 
- Linux
- install
- python
categories:
- 日常技术
- Linux
---
![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstrje1j20xc0himxw.jpg)


Selenium是爬虫中用来模拟JS的利器。

下面介绍一下Linux安装selenium和chromedriver的具体做法。

<!--more-->
## 1. install selenium

首先确保已经安装了pip命令，接下来：

```
sudo pip install -U selenium
```

## 2. install chromedriver

在[Chromedriver网站](http://chromedriver.storage.googleapis.com/index.html)上找到对应的版本，一般是最新版，如果你选的版本和电脑上的Chrome不互相匹配的话，在运行爬虫的时候会报错。（在网站里面的LATEST_RELEASE中可以找到最新版，不一定按那个序号来的）

找到后，把下面的`2.41`改成你要安装的版本。
```
wget -N http://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip
```

然后

```
unzip chromedriver_linux64.zip #解压你下载的那个包
chmod +x chromedriver   #修改用户权限为可执行
sudo mv -f chromedriver /usr/local/share/chromedriver #将解压后的文件移动到指定目录

#在指定目录link到别的目录
sudo ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver 
sudo ln -s /usr/local/share/chromedriver /usr/bin/chromedriver

```

一通操作后，你的selenium和chromedriver应该可以正常使用了。

```py
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://www.baidu.com/')
print('打开浏览器')
print(driver.title)
driver.quit()

```