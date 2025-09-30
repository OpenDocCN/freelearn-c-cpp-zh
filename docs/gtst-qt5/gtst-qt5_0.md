# 前言

在计算领域有许多流行词汇，其中大多数都与各种软件技术和概念有关。浏览器已成为获取信息和消费各种数据的首选方式。但仍然有一个空白，只能由必须安装和运行在操作系统上的独立应用程序来填补。浏览器本身作为一个应用程序，不能通过浏览器访问，这也证明了这一论断。

VLC、Adobe Photoshop、Google Earth 和 QGIS 等应用程序是直接在操作系统上运行的应用程序的一些例子。有趣的是，这些知名的软件品牌是用 Qt 构建的。

Qt（发音为“cute”）是一个跨平台的应用程序框架和控件工具包，用于创建在多种不同的硬件和操作系统上运行的图形用户界面应用程序。上述应用程序就是使用这个相同的工具包编写的。

本书的主要目的是向读者介绍 Qt。通过使用简单易懂的例子，它将引导用户从一个概念过渡到下一个，而不太关注理论。本书的篇幅要求我们在材料展示上要简洁。结合所提供的丰富例子，我们希望缩短理解和使用 Qt 的学习路径。

# 这本书面向谁

任何想要开始开发图形用户界面应用程序的人都会发现这本书很有用。为了理解这本书，不需要对其他工具包有先前的接触。然而，拥有这些技能将会很有用。

然而，本书假设您对 C++的使用有实际的知识。如果您能在开发算法和使用面向对象编程中表达自己的思想，您会发现内容很容易消化。

拥有 Qt 知识的专家或中级人员应寻求更多详细的外部材料。这本书不是参考指南，而应仅用作入门材料。

# 为了最大限度地利用这本书

每章的开始将包含一些理论，这应该有助于巩固您的理解。之后，一系列的例子被用来解释概念，并帮助读者更好地掌握主题。

本书也避免了继续使用前一章的例子。每个章节的例子都很短，不需要读者了解前一章的内容。这样，您可以挑选任何感兴趣的章节并开始学习。

已经提供了适当的链接来在 Windows 上设置环境。Linux 和 macOS 平台在本书中也得到了直接的支持。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择支持选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Getting-Started-with-Qt-5`](https://github.com/PacktPublishing/Getting-Started-with-Qt-5)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包可供下载，请访问 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载： [`www.packtpub.com/sites/default/files/downloads/9781789956030_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789956030_ColorImages.pdf)。

# 约定使用

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“要将密码设置为`connection`参数，发出代码片段，`db_conn.setPassword("")`。”

代码块应如下设置：

```cpp
QSqlDatabase db_conn =
        QSqlDatabase::addDatabase("QMYSQL", "contact_db");

db_conn.setHostName("127.0.0.1");
db_conn.setDatabaseName("contact_db");
db_conn.setUserName("root");
db_conn.setPassword("");
db_conn.setPort(3306);

```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

任何命令行输入或输出都应如下所示：

```cpp
% mkdir helloWorld
% ./run_executable
```

**粗体**: 表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“它在一个标签中显示文本 Hello world !。”

警告或重要说明如下所示。

小技巧如下所示。

# 联系我们

我们读者的反馈总是受欢迎的。

**一般反馈**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书籍标题，并通过 `customercare@packtpub.com` 邮箱联系我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为何不在你购买它的网站上留下评论呢？潜在读者可以查看并使用你的客观意见来做出购买决定，Packt 公司可以了解你对我们的产品有何看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
