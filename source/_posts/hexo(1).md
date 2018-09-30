---
title: hexo教程：github page+独立域名搭建(1)
date: 2018-09-05 13:38:44
id: 2018090514
tags:
- hexo
- blog
- 教程
categories:
- 日常技术
- 博客搭建
---

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstce5fj20ki0b4wey.jpg)

>喜欢写Blog的人，会经历三个阶段。

>- 第一阶段，刚接触Blog，觉得很新鲜，试着选择一个免费空间来写。

>- 第二阶段，发现免费空间限制太多，就自己购买域名和空间，搭建独立博客。

>- 第三阶段，觉得独立博客的管理太麻烦，最好在保留控制权的前提下，让别人来管，自己只负责写文章。            ------阮一峰


现在市面上的博客很多，如CSDN，博客园，简书等平台，可以直接在上面发表，用户交互做的好，写的文章百度也能搜索的到。缺点是比较不自由，会受到平台的各种限制和恶心的广告。

而自己购买域名和服务器，搭建博客的成本实在是太高了，不光是说这些购买成本，单单是花力气去自己搭这么一个网站，还要定期的维护它，对于我们大多数人来说，实在是没有这样的精力和时间。

那么就有第三种选择，直接在github page平台上托管我们的博客。这样就可以安心的来写作，又不需要定期维护，而且hexo作为一个快速简洁的博客框架，用它来搭建博客真的非常容易。

# Hexo简介
Hexo是一款基于Node.js的静态博客框架，依赖少易于安装使用，可以方便的生成静态网页托管在GitHub和Coding上，是搭建博客的首选框架。大家可以进入[hexo官网](https://hexo.io/zh-cn/)进行详细查看，因为Hexo的创建者是台湾人，对中文的支持很友好，可以选择中文进行查看。


# Hexo搭建步骤

1. 安装Git
2. 安装Node.js
3. 安装Hexo
4. GitHub创建个人仓库
5. 生成SSH添加到GitHub
6. 将hexo部署到GitHub
7. 设置个人域名
8. 发布文章

# 1. 安装Git

Git是目前世界上最先进的分布式版本控制系统，可以有效、高速的处理从很小到非常大的项目版本管理。也就是用来管理你的hexo博客文章，上传到GitHub的工具。Git非常强大，我觉得建议每个人都去了解一下。廖雪峰老师的Git教程写的非常好，大家可以了解一下。[Git教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

windows：到git官网上下载,[Download git](https://gitforwindows.org/),下载后会有一个Git Bash的命令行工具，以后就用这个工具来使用git。

linux：对linux来说实在是太简单了，因为最早的git就是在linux上编写的，只需要一行代码


```
sudo apt-get install git
```

安装好后，用`git --version` 来查看一下版本


# 2. 安装nodejs

Hexo是基于nodeJS编写的，所以需要安装一下nodeJs和里面的npm工具。

windows：[nodejs](https://nodejs.org/en/download/)选择LTS版本就行了。

linux：
```shell
sudo apt-get install nodejs
sudo apt-get install npm
```

安装完后，打开命令行
```
node -v
npm -v
```
检查一下有没有安装成功 


顺便说一下，windows在git安装完后，就可以直接使用git bash来敲命令行了，不用自带的cmd，cmd有点难用。

# 3. 安装hexo

前面git和nodejs安装好后，就可以安装hexo了，你可以先创建一个文件夹blog，然后`cd`到这个文件夹下（或者在这个文件夹下直接右键git bash打开）。

输入命令

```
npm install -g hexo-cli
```
依旧用`hexo -v`查看一下版本

至此就全部安装完了。


接下来初始化一下hexo

```
hexo init myblog
```
这个myblog可以自己取什么名字都行，然后
```bash
cd myblog //进入这个myblog文件夹
npm install
```

新建完成后，指定文件夹目录下有：

- node_modules: 依赖包
- public：存放生成的页面
- scaffolds：生成文章的一些模板
- source：用来存放你的文章
- themes：主题
- ** _config.yml: 博客的配置文件**


```
hexo g
hexo server
```
打开hexo的服务，在浏览器输入localhost:4000就可以看到你生成的博客了。

大概长这样：
![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrksvj6e0j211c0f2n60.jpg)
使用ctrl+c可以把服务关掉。

# 4. GitHub创建个人仓库

首先，你先要有一个GitHub账户，去注册一个吧。

注册完登录后，在GitHub.com中看到一个New repository，新建仓库
![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstcm7ej20ei0c1aah.jpg)


创建一个和你用户名相同的仓库，后面加.github.io，只有这样，将来要部署到GitHub page的时候，才会被识别，也就是xxxx.github.io，其中xxx就是你注册GitHub的用户名。我这里是已经建过了。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstusrdj20iw0o4myp.jpg)

点击create repository。



# 5. 生成SSH添加到GitHub

回到你的git bash中，
```
git config --global user.name "yourname"
git config --global user.email "youremail"
```
这里的yourname输入你的GitHub用户名，youremail输入你GitHub的邮箱。这样GitHub才能知道你是不是对应它的账户。

可以用以下两条，检查一下你有没有输对
```
git config user.name
git config user.email
```

然后创建SSH,一路回车
```
ssh-keygen -t rsa -C "youremail"
```

这个时候它会告诉你已经生成了.ssh的文件夹。在你的电脑中找到这个文件夹。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstd106j20kb073gll.jpg)

ssh，简单来讲，就是一个秘钥，其中，`id_rsa`是你这台电脑的私人秘钥，不能给别人看的，`id_rsa.pub`是公共秘钥，可以随便给别人看。把这个公钥放在GitHub上，这样当你链接GitHub自己的账户时，它就会根据公钥匹配你的私钥，当能够相互匹配时，才能够顺利的通过git上传你的文件到GitHub上。


而后在GitHub的setting中，找到SSH keys的设置选项，点击`New SSH key`
把你的`id_rsa.pub`里面的信息复制进去。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstdifaj210s0gfjrz.jpg)


在gitbash中，查看是否成功
```
ssh -T git@github.com
```

# 6. 将hexo部署到GitHub

这一步，我们就可以将hexo和GitHub关联起来，也就是将hexo生成的文章部署到GitHub上，打开站点配置文件 `_config.yml`，翻到最后，修改为
YourgithubName就是你的GitHub账户
```
deploy:
  type: git
  repo: https://github.com/YourgithubName/YourgithubName.github.io.git
  branch: master
```


这个时候需要先安装deploy-git ，也就是部署的命令,这样你才能用命令部署到GitHub。
```
npm install hexo-deployer-git --save
```

然后
```
hexo clean
hexo generate
hexo deploy
```
其中 `hexo clean`清除了你之前生成的东西，也可以不加。
`hexo generate` 顾名思义，生成静态文章，可以用 `hexo g`缩写
`hexo deploy` 部署文章，可以用`hexo d`缩写

注意deploy时可能要你输入username和password。

得到下图就说明部署成功了，过一会儿就可以在`http://yourname.github.io` 这个网站看到你的博客了！！
![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstbtvfj20lq01u3yd.jpg)

# 7. 设置个人域名

现在你的个人网站的地址是 `yourname.github.io`，如果觉得这个网址逼格不太够，这就需要你设置个人域名了。但是需要花钱。

注册一个阿里云账户,在[阿里云](https://wanwang.aliyun.com/?spm=5176.8142029.digitalization.2.e9396d3e46JCc5)上买一个域名，我买的是 `fangzh.top`，各个后缀的价格不太一样，比如最广泛的.com就比较贵，看个人喜好咯。


你需要先去进行实名认证,然后在域名控制台中，看到你购买的域名。

点**解析**进去，添加解析。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstcu8xj20d607wdfw.jpg)


其中，192.30.252.153 和 192.30.252.154 是GitHub的服务器地址。
**注意，解析线路选择默认**，不要像我一样选境外。这个境外是后面来做国内外分流用的,在后面的博客中会讲到。记得现在选择**默认**！！

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstf8unj20ob05b0sq.jpg)


登录GitHub，进入之前创建的仓库，点击settings，设置Custom domain，输入你的域名`fangzh.top`

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstghklj20as04mt8n.jpg)


然后在你的博客文件source中创建一个名为CNAME文件，不要后缀。写上你的域名。

![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrkstgsyrj208806aq2z.jpg)


最后，在gitbash中，输入
```
hexo clean
hexo g
hexo d
```
过不了多久，再打开你的浏览器，输入你自己的域名，就可以看到搭建的网站啦！

接下来你就可以正式开始写文章了。

```
hexo new newpapername
```

然后在source/_post中打开markdown文件，就可以开始编辑了。当你写完的时候，再
```
hexo clean
hexo g
hexo d
```
就可以看到更新了。

至于更换网站主题，还有添加各种各样的功能等等，在往后的系列博客中，再进行介绍。