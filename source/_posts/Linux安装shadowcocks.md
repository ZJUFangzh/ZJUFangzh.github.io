---
title: Linux安装shadowcocks
date: 2018-09-05 21:53:29
id: 2018090522
tags:
- Linux
- install
- shadowsocks
categories:
- 日常技术
- Linux
---
![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstv140j20zk0nmaar.jpg)



对于windows来说，只要下载一个shadowsocks的应用程序就行了。

github上一大堆[shadowsocks-windows](https://github.com/shadowsocks/shadowsocks-windows)

---
Linux上，可以用shell命令行解决的，绝不用GUI。
<!--more-->

```
sudo apt-get install python-pip

pip install shadowsocks
```

接下来配置文件 shadowsocks.json，随便找个地方，你记得住的地方保存。
```json

{

  	"server":"my_server_ip",
  
	"local_address": "127.0.0.1",

	"local_port":1080,

	"server_port":my_server_port,
  
	"password":"my_password",
  
	"timeout":300,

  	"method":"aes-256-cfb"

}

```

- my_server_ip:你的账户ip
- my_server_port:你的账户端口
- my_password:你的账户密码
- method:输入你账户的加密方式

配置完成后，分前端启动和后端启动

**前端启动**就是你那个窗口得一直开着

后面这一段是你刚才建立的json文件地址
```
 sudo sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json
```

**后端启动**在后端自己挂着（推荐）
```
sudo sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json -d start
```

**后端停止**
```
sudo sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json -d stop
```

**重启**（修改配置后要重启才能生效）

```
sudo sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json -d restart
```


在此，建议把命令行做成一个.sh文件，放在桌面，想开的时候就可以随时执行
shadowsocks.sh

```
#! /bin/bash

sudo sslocal -c /home/xx/Software/ShadowsocksConfig/shadowsocks.json -d start

```

---

配置好后，还需要在chrome浏览器中配置switchomega（插件），如果没有，自己去下一个。因为我们肯定是希望在指定的国外网站进行科学上网，而在国内的网站，就不需要用shadowsocks做转发了，这样很慢。所以配置一个有一定规则的列表，是很有必要的。详细的switchomega配置过程网上一大堆，这里就不详细说明了。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrksts6ifj21060gtwfj.jpg)

---

当然，如果你嫌麻烦，觉得以上用shell配置shadowsocks的方法太复杂，那直接下一个linux下的[shadowsocks-Qt5](https://github.com/shadowsocks/shadowsocks-qt5)吧。

---

还有安卓版的：

[shadowsocks-android](https://github.com/shadowsocks/shadowsocks-android/releases)