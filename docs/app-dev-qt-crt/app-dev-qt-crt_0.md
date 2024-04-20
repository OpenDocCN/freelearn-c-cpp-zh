# 前言

无论您是刚开始学习编程，还是已经确定 Qt 是您项目的 GUI 工具包，Qt Creator 都是一个很好的集成开发环境（IDE）的选择！在本书中，我们努力帮助您充分利用 Qt Creator，向您展示使用 Qt Creator 的几乎每个方面，从配置到编译和调试应用程序，以及众多的技巧和窍门。在这个过程中，您不仅会获得 Qt Creator 作为 IDE 的宝贵经验，还会获得 Qt 和 Qt Quick 的宝贵经验。阅读完本书后，您将能够：

+   使用 Qt Creator 编辑、编译、调试和运行 C++应用程序，为使用 Qt 和标准模板库（STL）构建最先进的控制台和 GUI 应用程序打开了一条道路

+   使用 Qt Creator 编辑、编译、调试和运行 Qt Quick 应用程序，让您可以访问最先进的声明式 GUI 创作环境之一

+   使用 Qt Designer 设计 GUI 应用程序，构建传统的基于小部件或 Qt Quick 应用程序

+   分析 Qt 应用程序的内存和运行时性能，并进行改进和缺陷修复

+   提供应用程序的本地化版本，以便您可以在世界各地以不同语言部署它

+   使用 Qt Quick 和 Qt Widgets 为诸如 Google Android 等平台编写移动应用程序

# 本书涵盖了什么内容

本书分为七章，您应该按顺序阅读，特别是如果您对 Qt Creator 和 Qt 编程不熟悉的话。这些章节包括：

第一章，“使用 Qt Creator 入门”，解释了如何下载和安装 Qt Creator，以及编辑简单的应用程序来测试您的安装。

第二章，“使用 Qt Creator 构建应用程序”，解释了如何使用 Qt Creator 编译，运行和调试应用程序。您将学习 Qt Creator 如何与 GNU 调试器和 Microsoft 控制台调试器集成，以提供断点、内存检查和其他调试帮助。

第三章，“使用 Qt Designer 设计您的应用程序”，解释了如何使用 Qt Creator 中的拖放 GUI 设计工具来构建 Qt 基于小部件和 Qt Quick 应用程序。

第四章，“使用 Qt Linguist 本地化您的应用程序”，解释了如何管理不同区域设置的资源字符串，让您可以在不同区域设置中使用不同语言构建应用程序。

第五章，“使用 Qt Creator 进行性能优化”，解释了如何使用 Qt Creator 来检查 Qt Quick 应用程序的运行时性能，以及如何使用 Valgrind 进行应用程序的内存分析，Valgrind 是一个开源的诊断工具。

第六章，“使用 Qt Creator 开发移动应用程序”，介绍了移动软件开发的激动人心的领域，并展示了如何利用本书中关于 Qt 和 Qt Creator 的知识来为诸如 Google Android 等平台编写应用程序。

第七章，“Qt 技巧和技巧”，涵盖了使用 Qt 和 Qt Creator 的技巧，这将帮助您高效地使用 Qt 框架和 Qt Creator IDE。

# 本书需要什么

Qt 和 Qt Creator 是跨平台工具。无论您使用的是 Windows 机器、运行 Mac OS X 的 Macintosh，还是运行 Linux 的工作站，您可能已经拥有所需的一切。您应该有足够的磁盘空间（大约 10GB 就足够了）来安装整个 Qt Creator IDE 和 Qt 库，与任何软件开发环境一样，您拥有的 RAM 越多越好（尽管我曾在运行 Ubuntu 的上网本上运行 Qt Creator，只有 1GB 的 RAM 也能正常运行！）。

您应该对计算机编程有基本的了解，并且应该准备用 C++编写代码。如果您对使用 Qt Quick 进行编程感兴趣，那么对 JavaScript 的基本了解会有所帮助，但您可以在学习过程中轻松掌握。

# 这本书适合谁

我写这本书是为了那些对 Qt 和 Qt Creator 没有或很少经验的人，他们可能是第一次在大学课程、开源项目中使用它，或者只是想尝试一下这个平台和 IDE。

我特别鼓励您阅读这本书，如果您是一名在大学 C++编程课程中使用 Qt Creator 的学生！您应该专注于前两章，以及您课程所需的其余部分。

# 约定

在这本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“对于名称，输入`HelloWorldConsole`，并选择对您有意义的路径（或接受默认设置）。”

代码块设置如下：

```cpp
#include <QCoreApplication>
#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
  QCoreApplication a(argc, argv);
  cout << "Hello world!";
  return a.exec();
}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会以粗体显示：

```cpp
import QtQuick 2.0
Rectangle {
  width: 360
  height: 360
  Text {
    text: qsTr("Hello World")
    anchors.centerIn: parent
  }
  MouseArea {
    anchors.fill: parent
    onClicked: {
      Qt.quit();
    }
  }
}
```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，比如菜单或对话框中的单词，会以这样的方式出现在文本中：“在**此处输入**的位置，右键单击并选择**删除菜单栏**。”

### 注意

警告或重要提示会以这样的方式出现在一个框中。

### 提示

提示和技巧会以这样的方式出现。

# 读者反馈

我们始终欢迎来自读者的反馈。让我们知道您对这本书的看法——您喜欢或不喜欢的地方。读者的反馈对我们开发您真正受益的标题非常重要。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在您的消息主题中提到书名。

如果您在某个专题上有专业知识，并且有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南，网址为`<www.packtpub.com/authors>`。

# 客户支持

现在您是 Packt 图书的自豪所有者，我们有很多东西可以帮助您充分利用您的购买。

## 下载示例代码

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)账户中购买的所有 Packt 图书中下载示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，将文件直接发送到您的电子邮件中。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误还是会发生。如果您在我们的书籍中发现错误——可能是文本或代码中的错误——我们将不胜感激地希望您向我们报告。通过这样做，您可以帮助其他读者避免挫折，并帮助我们改进本书的后续版本。如果您发现任何勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)报告，选择您的书籍，点击**勘误提交表**链接，并输入您的勘误详情。一旦您的勘误经过验证，您的提交将被接受，并且勘误将被上传到我们的网站上，或者添加到该标题的勘误列表中的任何现有勘误下的勘误部分。您可以通过访问[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有的勘误。

## 盗版

互联网上的版权盗版是所有媒体都面临的持续问题。在 Packt，我们非常重视对我们的版权和许可的保护。如果您在互联网上发现我们作品的任何非法副本，请立即向我们提供位置地址或网站名称，以便我们采取补救措施。

请通过`<copyright@packtpub.com>`与我们联系，并附上涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您在阅读本书的任何方面遇到问题，可以通过`<questions@packtpub.com>`与我们联系，我们将尽力解决。
