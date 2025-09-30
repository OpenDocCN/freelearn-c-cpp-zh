# 介绍 Qt 5

Qt 为开发者提供了一个出色的工具箱，可以轻松地创建令人惊叹且实用的应用程序，而无需承受太多压力，您很快就会发现这一点。在本章中，我们将介绍 Qt 并描述如何在机器上设置它。到本章结束时，您应该能够做到以下内容：

+   安装 Qt

+   使用 Qt 编写一个简单的程序

+   编译并运行一个 Qt 程序

目标保持简单直接。那么，让我们开始吧！

# 在 Linux 上安装 Qt

Ubuntu 操作系统使安装 Qt 5 变得相对容易。输入以下命令来设置您的环境：

```cpp
sudo apt-get install qt5-default
```

安装后，Qt 程序将从命令行编译和运行。在 第六章 “连接 Qt 与数据库”中，我们将展示如何使用 Qt 连接到数据库。输入以下命令以确保安装了 Qt 运行所需的库。我们将连接到的是 MySQL 数据库：

```cpp
sudo apt-get install libqt5sql5-mysql
```

# 在 macOS 上安装 Qt

在 Mac 上安装 Qt 有多种方法。要开始安装 Qt 5 到您的 Mac，您需要在您的机器上安装 Xcode。在终端中输入以下命令：

```cpp
xcode-select --install
```

如果您得到以下输出，那么您就可以进行下一步了：

```cpp
xcode-select: error: command line tools are already installed, use "Software Update" to install updates
```

*HomeBrew* 是一种软件包管理工具，它允许您轻松安装随 macOS 不附带安装的 Unix 工具。

如果您的机器上还没有安装，您可以在终端中输入以下命令进行安装：

```cpp
 /user/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

之后，您应该在终端中输入另一组命令来安装 Qt：

```cpp
curl -O https://raw.githubusercontent.com/Homebrew/homebrew-core/fdfc724dd532345f5c6cdf47dc43e99654e6a5fd/Formula/qt5.rb

brew install ./qt5.rb
```

在接下来的几章中，我们将使用 MySql 数据库。要配置 Qt 5 与 MySql，请输入以下命令：

```cpp
brew install ./qt5 --with-mysql
```

此命令可能需要一段时间才能完成，如果一切顺利，您就可以编写 Qt 程序了。

# Windows 上的安装

对于使用 Windows 的读者，安装过程仍然简单，尽管稍微不那么直接。我们可以先访问 [`download.qt.io`](http://download.qt.io)。

选择 `official_releases/`，然后 `online_installers/`，并选择下载 `qt-unified-windows-x86-online.exe`。

运行程序并选择创建账户。点击通过选择安装文件夹，并且不要忘记在选择需要安装的组件时选择 MinGW 5.3.0 32 位选项作为编译器。

本书中的大多数命令都应该在这个 IDE 中运行。

# 什么是 Qt？

现在我们已经设置了开发环境，让我们来构建一个“Hello World”示例。然而，首先让我们先简要地了解一下。

Qt 是一个用于创建 **图形用户界面**（**GUI**）以及跨平台应用程序的工具包。GUI 应用程序是使用鼠标向计算机发出命令以执行程序的应用程序。虽然 Qt 在某些情况下可以不使用鼠标操作，但这正是其用途所在。

在编写 GUI 应用程序时，试图在多个操作系统上实现相同的外观、感觉和功能是一项很大的挑战。Qt 通过提供一种只需编写一次代码并确保它在大多数操作系统上运行而无需进行太多或任何更改的方法，完全消除了这一障碍。

Qt 使用了一些模块。这些模块将相关的功能组合在一起。以下列出了一些模块及其功能：

+   `QtCore`：正如其名所示，这些模块包含 Qt 框架的核心和重要类。这包括容器、事件和线程管理等功能。

+   `QtWidgets`和`QtGui`：此模块包含用于调用控件的类。控件是构成图形界面大部分组件的元素。这包括按钮、文本框和标签。

+   `QtWebkit`：此模块使得在 Qt 应用程序中使用网页和应用成为可能。

+   `QtNetwork`：此模块提供连接到并通信网络资源的类。

+   `QtXML`：为了解析 XML 文档，此模块包含有用的类。

+   `QtSQL`：此模块具有丰富的类和驱动程序，允许连接到数据库，包括 MySQL、PostgreSQL 和 SQLite。

# Qt 中的“Hello World”

在本节中，我们将组合一个非常简单的“Hello World”程序。程序将在窗口中显示一个简单的按钮。在新建的名为`hello_world`的文件夹中创建一个名为`hello.cpp`的文件。打开文件并插入以下代码：

```cpp
#include <QApplication>
#include <QLabel>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QLabel label("Hello world !");
   label.show();
   return app.exec();
}
```

这看起来像是一个普通的 C++程序，除了使用了不熟悉的类。

就像任何常规程序一样，`int main()`函数是应用程序的入口点。

创建了一个`QApplication`类的实例，名为`app`，并将传递给`main()`函数的参数。`app`对象是必需的，因为它触发了`事件循环`，该循环会一直运行，直到我们关闭应用程序。没有`QApplication`对象，实际上无法创建 Qt GUI 应用程序。

然而，可以在不创建`QApplication`实例的情况下使用 Qt 的某些功能。

此外，`QApplication`的构造函数要求我们向其传递`argc`和`argv`。

我们实例化了一个`QLabel`类的对象，名为`label`。我们将`"Hello World!"`字符串传递给其构造函数。`QLabel`代表我们所说的控件，这是一个用来描述屏幕上视觉元素的术语。标签用于显示文本。

默认情况下，创建的控件是隐藏的。要显示它们，必须调用`show()`函数。

要启动`事件循环`，需要执行`app.exec()`这一行代码。这会将应用程序的控制权交给 Qt。

`return`关键字将一个整数返回给操作系统，表示应用程序关闭或退出时的状态。

要编译和运行我们的程序，导航到存储`hello.cpp`的文件夹。在终端中输入以下命令：

```cpp
% qmake -project
```

这将创建`hello_world.pro`文件。`hello_world`这个名字是`hello.cpp`所在的文件夹的名字。生成的文件将根据你存储`hello.cpp`文件的路径而变化。

使用你选择的任何文本编辑器打开`hello_world.pro`文件。以下几行需要一些解释：

```cpp
TEMPLATE = app
```

这里，`app`的值意味着项目的最终输出将是一个应用程序。或者，它可能是一个库或子目录：

```cpp
TARGET = hello_world
```

这里，`hello_world`的名字是应用程序或（库）的名称，它将被执行：

```cpp
SOURCES += hello.cpp
```

由于`hello.cpp`是我们项目中的唯一源文件，它被添加到`SOURCES`变量中。

我们需要生成一个`Makefile`，它将详细说明编译我们的 hello world 程序所需的步骤。这个自动生成的`Makefile`的好处是它消除了我们了解在不同操作系统上编译程序的各种细微差别所需的必要性。

在同一项目目录下，执行以下命令：

```cpp
% qmake
```

这将在目录中生成一个`Makefile`。

现在，执行以下命令来编译程序：

```cpp
% make
```

当运行`make`命令时，会产生以下错误（以及更多信息）作为输出：

```cpp
#include <QApplication>
        ^~~~~~~~~~~~
```

之前我们提到，各种组件和类被打包到模块中。`QApplication`正在我们的应用程序中使用，但正确的模块尚未包含。在编译过程中，这种遗漏会导致错误。

为了解决这个问题，打开`hello_world.pro`文件，并在该行之后插入以下几行：

```cpp
INCLUDEPATH += .
QT += widgets
```

这将添加`QtWidget`模块，以及`QtCore`模块，到编译程序中。添加了正确的模块后，再次在命令行上运行`make`命令：

```cpp
% make
```

在同一文件夹中会生成一个`hello_world`文件。按照以下方式从命令行运行此文件：

```cpp
% ./hello_world
```

在 macOS 上，可执行文件的完整路径将从以下命令行路径指定：

```cpp
./hello_world.app/Contents/MacOS/hello_world
```

这应该会产生以下输出：

![](img/2c834541-f8a2-4f5c-b1e7-2c8f97c37a3e.png)

好的，这就是我们的第一个 GUI 程序。它在一个标签中显示 Hello world !。要关闭应用程序，请点击窗口的关闭按钮。

让我们添加一些**Qt 样式表**（**QSS**）来给我们的标签添加一点效果！

按照以下方式修改`hello.cpp`文件：

```cpp
#include <QApplication>
#include <QLabel>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QLabel label("Hello world !");
   label.setStyleSheet("QLabel:hover { color: rgb(60, 179, 113)}");
   label.show();
   return app.exec();
}
```

这里的唯一变化是`label.setStyleSheet("QLabel:hover { color: rgb(60, 179, 113)}");`。

一个 QSS 规则作为参数传递给`label`对象的`setStyleSheet`方法。该规则设置我们应用程序中的每个标签，当光标悬停在其上时显示绿色。

运行以下命令重新编译应用程序并运行它：

```cpp
% make
% ./hello_world
```

程序应该看起来像以下截图。当鼠标放在标签上时，标签变为绿色：

![](img/e2178602-669c-4509-90db-06cb03263baa.png)

# 摘要

本章为了解 Qt 及其用途奠定了基础。概述了在 macOS 和 Linux 上安装 Qt 的步骤。编写并编译了一个简单的“Hello World”应用程序，所有操作均通过命令行完成，无需任何集成开发环境（IDE）。这意味着我们还了解了导致最终程序的各种步骤。

最后，将“Hello World”应用程序修改为使用 QSS，以展示可以对小部件进行哪些其他操作。

在第二章“创建小部件和布局”中，我们将探索 Qt 中的更多小部件以及如何组织和分组它们。
