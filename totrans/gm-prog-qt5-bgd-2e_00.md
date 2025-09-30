# 前言

作为所有重要桌面、移动和嵌入式平台的主要跨平台工具包，Qt 正变得越来越受欢迎。本书将帮助你学习 Qt 的细节，并为你提供构建应用程序和游戏的必要工具集。本书旨在作为入门指南，将 Qt 新手从基础，如对象、核心类、小部件和 5.9 版本的新特性，引导到能够使用 Qt 的最佳实践创建自定义应用程序的水平。

从简要介绍如何创建应用程序并为桌面和移动平台准备工作环境开始，我们将深入探讨创建图形界面和 Qt 数据处理和显示的核心概念。随着你通过章节的进展，你将学会通过实现网络连接和采用脚本丰富你的游戏。深入研究 Qt Quick、OpenGL 和其他工具，以添加游戏逻辑、设计动画、添加游戏物理、处理游戏手柄输入以及为游戏构建惊人的用户界面。本书的后期，你将学会利用移动设备功能，如传感器和地理位置服务，来构建引人入胜的用户体验。

# 本书面向的对象

本书对具有 C++基本知识的程序员和应用程序及 UI 开发者来说，既有趣又实用。此外，Qt 的一些部分允许你使用 JavaScript，因此对该语言的基本了解也将有所帮助。不需要有 Qt 的先前经验。拥有最多一年 Qt 经验的开发者也将从本书涵盖的主题中受益。

# 为了最大限度地利用本书

在开始使用本书之前，你不需要拥有或安装任何特定的软件。一个常见的 Windows、Linux 或 MacOS 系统就足够了。第二章，*安装*，包含了如何下载和设置所需所有内容的详细说明。

在本书中，你将发现一些频繁出现的标题：

+   **行动时间**部分包含了完成一个程序或任务的具体指导。

+   **发生了什么？**部分解释了你刚刚完成的任务或指令的工作原理。

+   **尝试英雄**部分包含了一些实际挑战，这些挑战能给你提供实验你所学内容的灵感。

+   **快速问答**部分包含了一些简短的单选题，旨在帮助你测试自己的理解。你将在本书的末尾找到答案。

在阅读章节时，您将看到多个游戏和其他项目，以及如何创建它们的详细描述。我们建议您尝试使用我们提供的说明自己创建这些项目。如果在任何时候您在遵循说明或不知道如何执行某个步骤时遇到困难，您应该查看示例代码文件以了解如何操作。然而，学习最重要的和最激动人心的部分是决定您想要实现什么，然后找到实现它的方法，因此请注意“英雄试炼”部分或考虑您自己的方法来改进每个项目。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保使用最新版本的以下软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Game-Programming-Using-Qt-5-Beginners-Guide-Second-Edition`](https://github.com/PacktPublishing/Game-Programming-Using-Qt-5-Beginners-Guide-Second-Edition)。我们还有其他丰富的图书和视频代码包可供选择，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。请查看它们！

# 约定使用

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“此 API 以`QNetworkAccessManager`为中心，该管理器处理您的游戏与互联网之间的完整通信。”

代码块设置如下：

```cpp
QNetworkRequest request;
request.setUrl(QUrl("http://localhost/version.txt"));
request.setHeader(QNetworkRequest::UserAgentHeader, "MyGame");
m_manager->get(request);
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
void FileDownload::downloadFinished(QNetworkReply *reply) {
    const QByteArray content = reply->readAll();
    _edit->setPlainText(content);
    reply->deleteLater();
}
```

**粗体**：表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的文字如下所示。以下是一个示例：“在“选择目标位置”屏幕上，点击“下一步”以接受默认目标。”

警告或重要提示如下所示。

小贴士和技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：请发送邮件至`feedback@packtpub.com`，并在邮件主题中提及书名。如果您对本书的任何方面有疑问，请发送邮件至`questions@packtpub.com`。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将非常感激您能提供位置地址或网站名称。请通过链接至材料的方式与我们联系至`copyright@packtpub.com`。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，并且我们的作者可以查看他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
