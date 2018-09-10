---
title: 'hexo教程:基本配置+更换主题+多终端工作+coding page部署分流(2)'
id: 2018090715
tags:
  - hexo
  - blog
  - 教程
categories:
  - 日常技术
  - 博客搭建
date: 2018-09-07 15:18:31
---


![](http://peu31tfv4.bkt.clouddn.com/1.jpg)



上次介绍了hexo的基本搭建和部署。但是还有很多事情没有解决，这次先来看看hexo的基本配置文件，还有如何在多平台部署实现国内外分流，以及换电脑后如何无缝的衔接工作。

<!--more-->



# 1. hexo基本配置



在文件根目录下的`_config.yml`，就是整个hexo框架的配置文件了。可以在里面修改大部分的配置。详细可参考[官方的配置](https://hexo.io/zh-cn/docs/configuration)描述。

### 网站

| 参数          | 描述                                                         |
| ------------- | ------------------------------------------------------------ |
| `title`       | 网站标题                                                     |
| `subtitle`    | 网站副标题                                                   |
| `description` | 网站描述                                                     |
| `author`      | 您的名字                                                     |
| `language`    | 网站使用的语言                                               |
| `timezone`    | 网站时区。Hexo 默认使用您电脑的时区。[时区列表](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)。比如说：`America/New_York`, `Japan`, 和 `UTC` 。 |

其中，`description`主要用于SEO，告诉搜索引擎一个关于您站点的简单描述，通常建议在其中包含您网站的关键词。`author`参数用于主题显示文章的作者。



### 网址

| 参数                 | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| `url`                | 网址                                                         |
| `root`               | 网站根目录                                                   |
| `permalink`          | 文章的 [永久链接](https://hexo.io/zh-cn/docs/permalinks) 格式 |
| `permalink_defaults` | 永久链接中各部分的默认值                                     |

 

在这里，你需要把`url`改成你的网站域名。

permalink，也就是你生成某个文章时的那个链接格式。

比如我新建一个文章叫`temp.md`，那么这个时候他自动生成的地址就是`http://yoursite.com/2018/09/05/temp`。

以下是官方给出的示例，关于链接的变量还有很多，需要的可以去官网上查找 [永久链接](https://hexo.io/zh-cn/docs/permalinks) 。

| 参数                            | 结果                        |
| ------------------------------- | --------------------------- |
| `:year/:month/:day/:title/`     | 2013/07/14/hello-world      |
| `:year-:month-:day-:title.html` | 2013-07-14-hello-world.html |
| `:category/:title`              | foo/bar/hello-world         |



再往下翻，中间这些都默认就好了。



```
theme: landscape

# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo: <repository url>
  branch: [branch]

```

`theme`就是选择什么主题，也就是在`theme`这个文件夹下，在官网上有很多个主题，默认给你安装的是`lanscape`这个主题。当你需要更换主题时，在官网上下载，把主题的文件放在`theme`文件夹下，再修改这个参数就可以了。



接下来这个`deploy`就是网站的部署的，`repo`就是仓库(`Repository`)的简写。`branch`选择仓库的哪个分支。这个在之前进行github page部署的时候已经修改过了，不再赘述。而这个在后面进行双平台部署的时候会再次用到。



### Front-matter

Front-matter 是文件最上方以 `---` 分隔的区域，用于指定个别文件的变量，举例来说：

```
title: Hello World
date: 2013/7/13 20:46:25
---
```

下是预先定义的参数，您可在模板中使用这些参数值并加以利用。

| 参数         | 描述                 |
| ------------ | -------------------- |
| `layout`     | 布局                 |
| `title`      | 标题                 |
| `date`       | 建立日期             |
| `updated`    | 更新日期             |
| `comments`   | 开启文章的评论功能   |
| `tags`       | 标签（不适用于分页） |
| `categories` | 分类（不适用于分页） |
| `permalink`  | 覆盖文章网址         |



其中，分类和标签需要区别一下，分类具有顺序性和层次性，也就是说 `Foo, Bar` 不等于 `Bar, Foo`；而标签没有顺序和层次。

```
categories:
- Diary
tags:
- PS3
- Games
```

### layout（布局）

当你每一次使用代码

```
hexo new paper
```

它其实默认使用的是`post`这个布局，也就是在`source`文件夹下的`_post`里面。

Hexo 有三种默认布局：`post`、`page` 和 `draft`，它们分别对应不同的路径，而您自定义的其他布局和 `post` 相同，都将储存到 `source/_posts` 文件夹。

| 布局    | 路径             |
| ------- | ---------------- |
| `post`  | `source/_posts`  |
| `page`  | `source`         |
| `draft` | `source/_drafts` |

而new这个命令其实是：

```
hexo new [layout] <title>
```

只不过这个layout默认是post罢了。



#### page

如果你想另起一页，那么可以使用

```
hexo new page board
```

系统会自动给你在source文件夹下创建一个board文件夹，以及board文件夹中的index.md，这样你访问的board对应的链接就是`http://xxx.xxx/board`

#### draft

draft是草稿的意思，也就是你如果想写文章，又不希望被看到，那么可以

```
hexo new draft newpage
```

这样会在source/_draft中新建一个newpage.md文件，如果你的草稿文件写的过程中，想要预览一下，那么可以使用

```
hexo server --draft
```

在本地端口中开启服务预览。



如果你的草稿文件写完了，想要发表到post中，

```
hexo publish draft newpage
```

就会自动把newpage.md发送到post中。



---

# 2. 更换主题

到这一步，如果你觉得默认的`landscape`主题不好看，那么可以在官网的主题中，选择你喜欢的一个主题进行修改就可以啦。[点这里](https://hexo.io/themes/)

![](http://peu31tfv4.bkt.clouddn.com/2hexo1.png)

这里有200多个主题可以选。不过最受欢迎的就是那么几个，比如[NexT主题](https://github.com/theme-next/hexo-theme-next)，非常的简洁好看，大多数人都选择这个，关于这个的教程也比较多。不过我选择的是[hueman](https://github.com/ppoffice/hexo-theme-hueman)这个主题，好像是从WordPress移植过来的，展示效果如下：

![](http://peu31tfv4.bkt.clouddn.com/2hexo13.png)



不管怎么样，至少是符合我个人的审美。



直接在github链接上下载下来，然后放到`theme`文件夹下就行了，然后再在刚才说的配置文件中把`theme`换成那个主题文件夹的名字，它就会自动在`theme`文件夹中搜索你配置的主题。



而后进入`hueman`这个文件夹，可以看到里面也有一个配置文件`_config.xml`，貌似它默认是`_config.xml.example`，把它复制一份，重命名为`_config.xml`就可以了。这个配置文件是修改你整个主题的配置文件。

### menu（菜单栏）

也就是上面菜单栏上的这些东西。

![](http://peu31tfv4.bkt.clouddn.com/2hexo2.png)

其中，About这个你是找不到网页的，因为你的文章中没有about这个东西。如果你想要的话，可以执行命令

```
hexo new page about
```

它就会在根目录下`source`文件夹中新建了一个`about`文件夹，以及index.md，在index.md中写上你想要写的东西，就可以在网站上展示出来了。

如果你想要自己再自定义一个菜单栏的选项，那么就

```
hexo new page yourdiy
```

然后在主题配置文件的menu菜单栏添加一个 `Yourdiy : /yourdiy`，注意冒号后面要有空格，以及前面的空格要和menu中默认的保持整齐。然后在`languages`文件夹中，找到`zh-CN.yml`，在index中添加`yourdiy: '中文意思'`就可以显示中文了。



### customize(定制)

在这里可以修改你的个人logo，默认是那个hueman，在`source/css/images`文件夹中放入自己要的logo，再改一下`url`的链接名字就可以了。

`favicon`是网站中出现的那个小图标的icon，找一张你喜欢的logo，然后转换成ico格式，放在images文件夹下，配置一下路径就行。

`social_links` ，可以显示你的社交链接，而且是有logo的。

**tips:**

在这里可以添加一个rss功能，也就是那个符号像wifi一样的东西。

### 添加RSS

**1. 什么是RSS？**

RSS也就是订阅功能，你可以理解为类似与订阅公众号的功能，来订阅各种博客，杂志等等。

**2. 为什么要用RSS？**

就如同订阅公众号一样，你对某个公众号感兴趣，你总不可能一直时不时搜索这个公众号来看它的文章吧。博客也是一样，如果你喜欢某个博主，或者某个平台的内容，你可以通过RSS订阅它们，然后在RSS阅读器上可以实时推送这些消息。现在网上的垃圾消息太多了，如果你每一天都在看这些消息中度过，漫无目的的浏览，只会让你的时间一点一点的流逝，太不值得了。如果你关注的博主每次都发的消息都是精华，而且不是每一天十几条几十条的轰炸你，那么这个博主就值得你的关注，你就可以通过RSS订阅他。

在我的理解中，如果你不想每天都被那些没有质量的消息轰炸，只想安安静静的关注几个博主，每天看一些有质量的内容也不用太多，那么RSS订阅值得你的拥有。

**3. 添加RSS功能**

先安装RSS插件

```
npm i hexo-generator-feed
```

而后在你整个项目的`_config.yml`中找到Extensions，添加：

```
# Extensions
## Plugins: https://hexo.io/plugins/
#RSS订阅
plugin:
- hexo-generator-feed
#Feed Atom
feed:
  type: atom
  path: atom.xml
  limit: 20
```

这个时候你的RSS链接就是  域名`/atom.xml`了。

所以，在主题配置文件中的这个`social links`，开启RSS的页面功能，这样你网站上就有那个像wifi一样符号的RSS logo了，注意空格。

```
rss: /atom.xml
```

**4. 如何关注RSS？**

首先，你需要一个RSS阅读器，在这里我推荐`inoreader`，宇宙第一RSS阅读器，而且中文支持的挺好。不过它没有PC端的程序，只有网页版，chrome上有插件。在官网上用google账号或者自己注册账号登录，就可以开始你的关注之旅了。

每次需要关注某个博主时，就点开他的RSS链接，把链接复制到`inoreader`上，就能关注了，当然，如果是比较大众化的很厉害的博主，你直接搜名字也可以的，比如每个人都非常佩服的阮一峰大师，直接在阅读器上搜索`阮一峰`，应该就能出来了。

我关注的比如，阮一峰的网络日志，月光博客，知乎精选等，都很不错。当然，还有我！！赶快关注我吧！你值得拥有：http://fangzh.top/atom.xml

在安卓端，inoreader也有下载，不过因为国内google是登录不了的，你需要在inoreader官网上把你的密码修改了，然后就可以用账户名和密码登录了。

在IOS端，没用过，好像是reader 3可以支持inoreader账户，还有个readon也不错，可以去试试。



### widgets(侧边栏)

侧边栏的小标签，如果你想自己增加一个，比如我增加了一个联系方式，那么我把`communication`写在上面，在`zh-CN.yml`中的`sidebar`，添加`communication: '中文'`。

然后在`hueman/layout/widget`中添加一个`communicaiton.ejs`，填入模板：

```js
<% if (site.posts.length) { %>
    <div class="widget-wrap widget-list">
        <h3 class="widget-title"><%= __('sidebar.communiation') %></h3>
        <div class="widget">
            <!--这里添加你要写的内容-->
        </div>
    </div>
<% } %>
```



### search(搜索框)

默认搜索框是不能够用的，

> you need to install `hexo-generator-json-content` before using Insight Search

它已经告诉你了，如果想要使用，就安装这个插件。



### comment(评论系统)

这里的多数都是国外的，基本用不了。这个`valine`好像不错，还能统计文章阅读量，可以自己试一试，[链接](https://valine.js.org/quickstart.html#npm)。



### miscellaneous(其他)

这里我就改了一个`links`，可以添加友链。注意空格要对！不然会报错！



### 总结：

整个主题看起来好像很复杂的样子，但是仔细捋一捋其实也比较流畅，

- languages: 顾名思义
- layout：布局文件，其实后期想要修改自定义网站上的东西，添加各种各样的信息，主要是在这里修改，其中`comment`是评论系统，`common`是常规的布局，最常修改的在这里面，比如修改页面`head`和`footer`的内容。
- scripts：js脚本，暂时没什么用
- source：里面放了一些css的样式，以及图片



---

# 3. git分支进行多终端工作

问题来了，如果你现在在自己的笔记本上写的博客，部署在了网站上，那么你在家里用台式机，或者实验室的台式机，发现你电脑里面没有博客的文件，或者要换电脑了，最后不知道怎么移动文件，怎么办？

在这里我们就可以利用git的分支系统进行多终端工作了，这样每次打开不一样的电脑，只需要进行简单的配置和在github上把文件同步下来，就可以无缝操作了。





### 机制

机制是这样的，由于`hexo d`上传部署到github的其实是hexo编译后的文件，是用来生成网页的，不包含源文件。

![可以看到，并没有source等源文件在内](http://peu31tfv4.bkt.clouddn.com/2hexo4.png)



也就是上传的是在本地目录里自动生成的`.deploy_git`里面。

其他文件 ，包括我们写在source 里面的，和配置文件，主题文件，都没有上传到github

![](http://peu31tfv4.bkt.clouddn.com/2hexo3.png)





所以可以利用git的分支管理，将源文件上传到github的另一个分支即可。



### 上传分支

首先，先在github上新建一个hexo分支，如图：

![](http://peu31tfv4.bkt.clouddn.com/2hexo8.png)



然后在这个仓库的settings中，选择默认分支为hexo分支（这样每次同步的时候就不用指定分支，比较方便）。

![](http://peu31tfv4.bkt.clouddn.com/2hexo9.png)



然后在本地的任意目录下，打开git bash，

```shell
git clone git@github.com:ZJUFangzh/ZJUFangzh.github.io.git
```

将其克隆到本地，因为默认分支已经设成了hexo，所以clone时只clone了hexo。



接下来在克隆到本地的`ZJUFangzh.github.io`中，把除了.git 文件夹外的所有文件都删掉

 把之前我们写的博客源文件全部复制过来，除了`.deploy_git`。这里应该说一句，复制过来的源文件应该有一个`.gitignore`，用来忽略一些不需要的文件，如果没有的话，自己新建一个，在里面写上如下，表示这些类型文件不需要git：

```
.DS_Store
Thumbs.db
db.json
*.log
node_modules/
public/
.deploy*/
```

注意，如果你之前克隆过theme中的主题文件，那么应该把主题文件中的`.git`文件夹删掉，因为git不能嵌套上传，最好是显示隐藏文件，检查一下有没有，否则上传的时候会出错，导致你的主题文件无法上传，这样你的配置在别的电脑上就用不了了。

而后

```shell
git add .
git commit –m "add branch"
git push 
```

这样就上传完了，可以去你的github上看一看hexo分支有没有上传上去，其中`node_modules`、`public`、`db.json`已经被忽略掉了，没有关系，不需要上传的，因为在别的电脑上需要重新输入命令安装 。

![](http://peu31tfv4.bkt.clouddn.com/2hexo7.png)



这样就上传完了。



### 更换电脑操作

一样的，跟之前的环境搭建一样，

- 安装git

```
sudo apt-get install git
```

- 设置git全局邮箱和用户名

```
git config --global user.name "yourgithubname"
git config --global user.email "yourgithubemail"
```

- 设置ssh key

```
ssh-keygen -t rsa -C "youremail"
#生成后填到github和coding上（有coding平台的话）
#验证是否成功
ssh -T git@github.com
ssh -T git@git.coding.net #(有coding平台的话)
```

- 安装nodejs

```
sudo apt-get install nodejs
sudo apt-get install npm
```

-  安装hexo  

```
sudo npm install hexo-cli -g
```

但是已经不需要初始化了，

直接在任意文件夹下，

```
git clone git@………………
```

然后进入克隆到的文件夹：

```
cd xxx.github.io
npm install
npm install hexo-deployer-git --save
```

生成，部署：

```
hexo g
hexo d
```



然后就可以开始写你的新博客了

```
hexo new newpage
```



**Tips:**

1. 不要忘了，每次写完最好都把源文件上传一下

```
git add .
git commit –m "xxxx"
git push 
```

2. 如果是在已经编辑过的电脑上，已经有clone文件夹了，那么，每次只要和远端同步一下就行了

```
git pull
```



---

# 4. coding page上部署实现国内外分流

之前我们已经把hexo托管在github了，但是github是国外的，而且百度的爬虫是不能够爬取github的，所以如果你希望你做的博客能够在百度引擎上被收录，而且想要更快的访问，那么可以在国内的coding page做一个托管，这样在国内访问就是coding page，国外就走github page。



**1. 申请coding账户，新建项目**

先申请一个账户，然后创建新的项目，这一步项目名称应该是随意的。

**2.  添加ssh key**

这一步跟github一样。

添加后，检查一下是不是添加成功

```
ssh -T git@git.coding.net
```

**3. 修改_config.yml**

hexo官方文档是这样的：

```
deploy:
  type: git
  message: [message]
  repo:
    github: <repository url>,[branch]
    coding: <repository url>,[branch] 
```

那么，我们只需要：

```
deploy:
  type: git
  repo: 
    coding: git@git.coding.net:ZJUFangzh/ZJUFangzh.git,master
    github: git@github.com:ZJUFangzh/ZJUFangzh.github.io.git,master
```

**4. 部署**

保存一下，直接

```
hexo g
hexo d
```

这样就可以在coding的项目上看到你部署的文件了。

**5. 开启coding pages服务，绑定域名**

如图：

![](http://peu31tfv4.bkt.clouddn.com/2hexo11.png)

**6. 阿里云添加解析**

![](http://peu31tfv4.bkt.clouddn.com/2hexo5.png)

这个时候就可以把之前github的解析改成境外，把coding的解析设为默认了。

**7. 去除coding page的跳转广告**

coding page的一个比较恶心人的地方就是，你只是银牌会员的话，访问会先跳转到一个广告，再到你自己的域名。那么它也给出了消除的办法。右上角切换到coding的旧版界面，默认新版是不行的。然后再来到`pages服务`这里。

这里：

![](http://peu31tfv4.bkt.clouddn.com/2hexo10.png)

只要你在页面上添加一行文字，写`Hosted by Coding Pages`，然后点下面的小勾勾，两个工作日内它就会审核通过了。

```
<p>Hosted by <a href="https://pages.coding.me" style="font-weight: bold">Coding Pages</a></p>
```

我的选择是把这一行代码放在主题文件夹`/layout/common/footer.ejs`里面，也就是本来在页面中看到的页脚部分。

![](http://peu31tfv4.bkt.clouddn.com/2hexo6.png)

当然，为了统一，我又在后面加上了and **Github**哈哈，可以不加。

```
<p><span>Hosted by <a href="https://pages.coding.me" style="font-weight: bold">Coding Pages</a></span> and <span><a href="https://github.com" style="font-weight: bold">Github</a></span></p>
```

这是最终加上去的代码。



至此，关于hexo的基本文件配置，主题更换，多终端同步，多平台部署已经介绍完了。

这一次就先到这里了，下回再讲讲如何优化网站的SEO、以及在主题中添加评论系统、阅读量统计等等，谢谢大家。