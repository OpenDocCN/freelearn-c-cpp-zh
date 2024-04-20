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