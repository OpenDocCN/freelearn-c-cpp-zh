# 前言

本书是关于 Godot 游戏引擎及其新版本 3.0 的入门介绍。Godot 3.0 拥有大量新功能和能力，使其成为比商业游戏引擎更强大的替代品。对于初学者来说，它提供了一个友好的学习游戏开发技术的途径。对于更有经验的开发者，Godot 是一个强大且可定制的工具，可以将创意变为现实。

本书将采用基于项目的教学方法。它包含五个项目，这些项目将帮助开发者深入理解如何使用 Godot 引擎来构建游戏。

# 本书面向的对象

这本书适合任何想要学习如何使用现代游戏引擎制作游戏的人。无论是新用户还是经验丰富的开发者，都会发现这是一本有用的资源。建议具备一些编程经验。

# 本书涵盖的内容

本书是使用 Godot 游戏引擎的项目式入门介绍。每个游戏项目都是在前一个项目的基础上构建的。

第一章，*简介*，介绍了游戏引擎的一般概念以及 Godot 的具体情况，包括如何下载 Godot 并在您的计算机上安装它。

第二章，*金币冲刺*，处理一个小游戏，演示了如何创建场景以及与 Godot 的节点架构一起工作。

第三章，*逃离迷宫*，包含一个基于俯视迷宫游戏的项目，将展示如何使用 Godot 强大的继承特性和节点来处理瓦片地图和精灵动画。

第四章，*太空岩石*，展示了如何使用物理体来创建类似《小行星》风格的太空游戏。

第五章，*丛林跳跃*，涉及类似《超级马里奥兄弟》的侧滚动平台游戏。您将了解运动学体、动画状态和视差背景。

第六章，*3D 迷你高尔夫*，将前面的概念扩展到三维空间。您将处理网格、光照和相机控制。

第七章，*附加主题*，涵盖了在掌握前几章材料后可以探索的更多主题。

# 为了充分利用本书

为了最好地理解本书中的示例代码，您应该具备编程的一般知识，最好是现代动态类型语言，如 Python 或 JavaScript。如果您是编程新手，在深入本书的游戏项目之前，您可能希望先回顾一个初学者 Python 教程。

Godot 可以在运行 Windows、macOS 或 Linux 操作系统的任何相对现代的 PC 上运行。您的显卡必须支持 OpenGL ES 3.0。

# 下载示例代码文件

您可以从 [www.packtpub.com](http://www.packtpub.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载和勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

书籍的代码包也托管在 GitHub 上 [`github.com/PacktPublishing/Godot-Game-Engine-Projects/issues`](https://github.com/PacktPublishing/Godot-Game-Engine-Projects)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有来自我们丰富的书籍和视频目录的其他代码包，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 获取。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/GodotEngineGameDevelopmentProjects_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/GodotEngineGameDevelopmentProjects_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的 `WebStorm-10*.dmg` 磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块设置如下：

```cpp
extends Area2D

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)
```

任何命令行输入或输出都如下所示：

```cpp
adb install dodge.apk
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“编辑器窗口的主要部分是 Viewport。”

警告或重要注意事项如下所示。

小贴士和技巧看起来是这样的。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：请将邮件发送至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送邮件给我们。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误表提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为何不在你购买它的网站上留下评论呢？潜在读者可以查看并使用你的客观意见来做出购买决定，Packt 公司可以了解你对我们的产品有何看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
