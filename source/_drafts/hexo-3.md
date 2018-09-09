---
title: hexo教程:搜索SEO+阅读量统计+访问量统计+评论系统
date: 2018-09-09 13:09:38
id: 2018090913
tags:
- hexo
- blog
- 教程
categories:
- 日常技术
- 博客搭建
---

![]()



网站做完之后，可以为网站添加一些常用的功能，如能被搜索引擎收录的SEO优化，网站访问量和文章阅读量统计，以及评论系统。

<!--more-->

本文参考了: [visugar.com](http://visugar.com/2017/08/01/20170801HexoPlugins/)这里面说的很详细了。

# 1. SEO优化 

推广是很麻烦的事情，怎么样别人才能知道我们呢，首先需要让搜索引擎收录你的这个网站，别人才能搜索的到。那么这就需要SEO优化了。

> SEO是由英文Search Engine Optimization缩写而来， 中文意译为“搜索引擎优化”。SEO是指通过站内优化比如网站结构调整、网站内容建设、网站代码优化等以及站外优化。
>
> 

### 百度seo

刚建站的时候是没有搜索引擎收录我们的网站的。可以在搜索引擎中输入`site:<域名>`

来查看一下。



**1. 登录百度站长平台添加网站**

登录[百度站长平台](https://ziyuan.baidu.com/linksubmit/index?)，在站点管理中添加你自己的网站。

验证网站有三种方式：文件验证、HTML标签验证、CNAME验证。

第三种方式最简单，只要将它提供给你的那个xxxxx使用CNAME解析到xxx.baidu.com就可以了。也就是登录你的阿里云，把这个解析填进去就OK了。

**2. 提交链接**

我们需要使用npm自动生成网站的sitemap，然后将生成的sitemap提交到百度和其他搜索引擎

```
npm install hexo-generator-sitemap --save     
npm install hexo-generator-baidu-sitemap --save
```

这时候你需要在你的根目录下`_config.xml`中看看url有没有改成你自己的：

![]()



重新部署后，就可以在public文件夹下看到生成的sitemap.xml和baidusitemap.xml了。

然后就可以向百度提交你的站点地图了。

这里建议使用自动提交。

![]()

自动提交又分为三种：主动推送、自动推送、sitemap。

可以三个一起提交不要紧，我选择的是后两种。

- 自动推送：把百度生成的自动推送代码，放在主题文件`/layout/common/head.ejs`的适当位置，然后验证一下就可以了。
- sitemap：把两个sitemap地址，提交上去，看到状态正常就OK了。

![]()



**ps:** 百度收录比较慢，慢慢等个十天半个月再去`site:<域名>`看看有没有被收录。



### google的SEO



流程一样，google更简单，而且收录更快，进入[google站点地图](https://search.google.com/search-console/sitemaps?resource_id=http://fangzh.top/&hl=zh-CN)，提交网站和sitemap.xml，就可以了。

如果你这个域名在google这里出了问题，那你就提交 yourname.github.io，这个链接，效果是一样的。

不出意外的话一天内google就能收录你的网站了。

![]()



其他的搜索，如搜狗搜索，360搜索，流程是一样的，这里就不再赘述。



# 2. 评论系统



评论系统有很多，但是很多都是墙外的用不了，之前说过这个valine好像集成在hueman和next主题里面了，但是我还没有研究过，我看的是[visugar](http://visugar.com/2017/08/01/20170801HexoPlugins/)这个博主用的来比力评论系统，感觉也还不错。



[来比力官网](https://livere.com/)，注册好后，点击管理页面，在`代码管理`中找到安装代码：

![]()



获取安装代码后，在主题的comment下新建一个文件放入刚刚那段代码，再找到article文件，找到如下代码，若没有则直接在footer后面添加即可。livebe即为刚刚所创文件名称。

```
<%- partial('comment/livebe') %>
```



然后可以自己设置一些东西：

![]()

还可以设置评论提醒，这样别人评论你的时候就可以及时知道了。



# 3. 添加百度统计



百度统计可以在后台上看到你网站的访问数，浏览量，浏览链接分布等很重要的信息。所以添加百度统计能更有效的让你掌握你的网站情况。

[百度统计](https://tongji.baidu.com)，注册一下，这里的账号好像和百度账号不是一起的。



![]()

照样把代码复制到`head.ejs`文件中，然后再进行一下安装检查，半小时左右就可以在百度统计里面看到自己的网站信息了。



# 4. 文章阅读量统计leanCloud



[leanCloud](https://leancloud.cn/)，进去后注册一下，进入后创建一个应用：

![]()

在`存储`中创建Class，命名为Counter,

![]()



然后在设置页面看到你的`应用Key`，在主题的配置文件中：

```
leancloud_visitors:
  enable: true
  app_id: 你的id
  app_key: 你的key
```



在`article.ejs`中适当的位置添加如下，这要看你让文章的阅读量统计显示在哪个地方了，

```
阅读数量:<span id="<%= url_for(post.path) %>" class="leancloud_visitors" data-flag-title="<%- post.title %>"></span>次
```



然后在`footer.ejs`的最后，添加：

```
<script src="//cdn1.lncld.net/static/js/2.5.0/av-min.js"></script>
<script>
    var APP_ID = '你的app id';
    var APP_KEY = '你的app key';
    AV.init({
        appId: APP_ID,
        appKey: APP_KEY
    });
    // 显示次数
    function showTime(Counter) {
        var query = new AV.Query("Counter");
        if($(".leancloud_visitors").length > 0){
            var url = $(".leancloud_visitors").attr('id').trim();
            // where field
            query.equalTo("words", url);
            // count
            query.count().then(function (number) {
                // There are number instances of MyClass where words equals url.
                $(document.getElementById(url)).text(number?  number : '--');
            }, function (error) {
                // error is an instance of AVError.
            });
        }
    }
    // 追加pv
    function addCount(Counter) {
        var url = $(".leancloud_visitors").length > 0 ? $(".leancloud_visitors").attr('id').trim() : 'icafebolger.com';
        var Counter = AV.Object.extend("Counter");
        var query = new Counter;
        query.save({
            words: url
        }).then(function (object) {
        })
    }
    $(function () {
        var Counter = AV.Object.extend("Counter");
        addCount(Counter);
        showTime(Counter);
    });
</script>
```



重新部署后就可以了。



# 5. 引入不蒜子访问量和访问人次统计

不蒜子的添加非常非常方便，[不蒜子](http://busuanzi.ibruce.info/)

在`footer.ejs`中的合适位置，看你要显示在哪个地方，添加：

```
<!--这一段是不蒜子的访问量统计代码-->
<script async src="//dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js"></script>
<span id="busuanzi_container_site_pv">本站总访问量<span id="busuanzi_value_site_pv"></span>次 &nbsp;   </span>
<span id="busuanzi_container_site_uv">访客数<span id="busuanzi_value_site_uv"></span>人次</span>
```

就可以了。



# 总结

到这里就基本做完了。其实都是参考别的博主的设置的，不一定仅限于hueman主题，其他主题的设置也是大体相同的，所以如果你希望设置别的主题，那么仔细看一下这个主题的代码结构，也能够把上边的功能添加进去。



多看看别的博主的那些功能，如果有你能找到自己喜欢的功能，那么好好发动搜索技能，很快就能找到怎么做了。加油吧！