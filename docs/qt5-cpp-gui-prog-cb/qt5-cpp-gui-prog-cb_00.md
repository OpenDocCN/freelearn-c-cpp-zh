# 前言

计算机软件市场的持续增长导致了一个竞争激烈和具有挑战性的时代。你的软件不仅需要功能强大且易于使用，还必须对用户具有吸引力和专业性。为了在市场上获得竞争优势，产品的外观和感觉至关重要，并且应该在生产阶段早期予以关注。在本书中，我们将教你如何使用 Qt5 开发平台创建功能强大、吸引人且用户友好的软件。

# 本书涵盖了什么

第一章 *外观和感觉定制*，展示了如何使用 Qt Designer 和 Qt Quick Designer 设计程序的用户界面。

第二章 *状态和动画*，解释了如何通过使用状态机框架和动画框架来为用户界面小部件添加动画效果。

第三章 *QPainter 和 2D 图形*，介绍了如何使用 Qt 的内置类在屏幕上绘制矢量形状和位图图像。

第四章 *OpenGL 实现*，演示了如何通过在 Qt 项目中集成 OpenGL 来渲染程序中的 3D 图形。

第五章 *使用 Qt5 构建触摸屏应用程序*，解释了如何创建适用于触摸屏设备的程序。

第六章 *简化 XML 解析*，展示了如何处理 XML 格式的数据，并与 Google 地理编码 API 一起使用，以创建一个简单的地址查找器。

第七章 *转换库*，介绍了如何使用 Qt 的内置类以及第三方程序在不同变量类型、图像格式和视频格式之间进行转换。

第八章 *访问数据库*，解释了如何使用 Qt 将程序连接到 SQL 数据库。

第九章 *使用 Qt Web 引擎开发 Web 应用程序*，介绍了如何使用 Qt 提供的 Web 渲染引擎，并开发利用 Web 技术的程序。

# 本书需要什么

以下是本书的先决条件：

1.  Qt5（适用于所有章节）

1.  FFmpeg（用于第七章 *转换库*）

1.  XAMPP（用于第八章 *访问数据库*）

# 本书适合谁

本书旨在为那些想使用 Qt5 开发软件的人提供帮助。如果你想提高软件应用的视觉质量和内容呈现，这本书将最适合你。

# 部分

在本书中，你会经常看到几个标题（准备工作，如何做，它是如何工作的，还有更多，另请参阅）。

为了清晰地说明如何完成一个配方，我们使用以下这些部分：

## 准备工作

本节告诉你在配方中可以期待什么，并描述了为配方设置任何软件或任何预备设置所需的步骤。

## 如何做...

本节包含了遵循配方所需的步骤。

## 它是如何工作的...

本节通常包括对上一节内容的详细解释。

## 还有更多...

本节包含有关配方的附加信息，以使读者更加了解配方。

## 另请参阅

本节为配方提供了其他有用信息的链接。 

# 约定

在本书中，你会发现一些区分不同信息类型的文本样式。以下是一些样式的示例及其含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下："在`mylabel.cpp`源文件中，定义一个名为`SetMyObject()`的函数来保存对象指针。"

代码块设置如下：

```cpp
QSpinBox::down-button
{
  image: url(:/images/spindown.png);
  subcontrol-origin: padding;
  subcontrol-position: right bottom;
}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
QSpinBox::down-button
{
 image: url(:/images/spindown.png);
  subcontrol-origin: padding;
  subcontrol-position: right bottom;
}
```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的单词，比如菜单或对话框中的单词，会以这样的方式出现在文本中："转到**Library**窗口中的**Imports**标签，并向您的项目添加一个名为**QtQuick.Controls**的 Qt Quick 模块。"

### 注意

警告或重要提示会显示在这样的框中。

### 提示

提示和技巧会显示为这样。

# 读者反馈

我们始终欢迎读者的反馈。让我们知道您对这本书的看法——您喜欢或不喜欢什么。读者的反馈对我们很重要，因为它有助于我们开发您真正能从中获益的标题。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在主题中提及书名。

如果您在某个专题上有专业知识，并且有兴趣撰写或为一本书做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您是 Packt 书籍的自豪所有者，我们有很多东西可以帮助您充分利用您的购买。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便直接将文件发送到您的电子邮件。

您可以按照以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册到我们的网站。

1.  将鼠标指针悬停在顶部的**SUPPORT**标签上。

1.  点击**Code Downloads & Errata**。

1.  在**Search**框中输入书名。

1.  选择您要下载代码文件的书籍。

1.  从下拉菜单中选择您购买这本书的地点。

1.  点击**Code Download**。

您还可以通过在 Packt Publishing 网站上的书籍网页上点击**Code Files**按钮来下载代码文件。可以通过在**Search**框中输入书名来访问该页面。请注意，您需要登录到您的 Packt 账户。

下载文件后，请确保使用最新版本的解压缩软件解压缩文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Qt5-C-GUI-Programming-Cookbook`](https://github.com/PacktPublishing/Qt5-C-GUI-Programming-Cookbook)。我们还有来自丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快去看看吧！

## 下载本书的彩色图片

我们还为您提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。彩色图片将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/Qt5C++GUIProgrammingCookbook_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/Qt5C++GUIProgrammingCookbook_ColorImages.pdf)下载此文件。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误是难免的。如果您在我们的书籍中发现错误——可能是文本或代码中的错误，我们将不胜感激地希望您向我们报告。通过这样做，您可以帮助其他读者避免挫折，并帮助我们改进本书的后续版本。如果您发现任何勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表**链接，并输入您的勘误详情。一旦您的勘误经过验证，您的提交将被接受，并且勘误将被上传到我们的网站或添加到该书籍的勘误部分下的任何现有勘误列表中。

要查看先前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)并在搜索框中输入书名。所需信息将显示在**勘误表**部分下。

## 盗版

互联网上的盗版行为是跨所有媒体持续存在的问题。在 Packt，我们非常重视版权和许可的保护。如果您在互联网上发现我们作品的任何非法副本，请立即向我们提供位置地址或网站名称，以便我们采取补救措施。

请通过`<copyright@packtpub.com>`与我们联系，并附上涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们提供有价值内容的能力方面的帮助。

## 问题

如果您对本书的任何方面有问题，可以通过`<questions@packtpub.com>`与我们联系，我们将尽力解决问题。
