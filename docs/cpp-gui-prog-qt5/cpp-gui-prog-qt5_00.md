# 前言

Qt 5 是 Qt 的最新版本，它使您能够为多个目标开发具有复杂用户界面的应用程序。它为您提供了更快速、更智能的方式来创建现代 UI 和多平台应用程序。本书将教您如何设计和构建功能齐全、吸引人和用户友好的图形用户界面。

通过本书，您将成功学习高端 GUI 应用程序，并能够构建更多功能强大的跨平台应用程序。

# 本书适合对象

本书适合希望构建基于 GUI 的应用程序的开发人员和程序员。需要基本的 C++知识，了解 Qt 的基础知识会有所帮助。

# 充分利用本书

为了成功执行本书中的所有代码和指令，您需要以下内容：

+   基本的 PC/笔记本电脑

+   工作的互联网连接

+   Qt 5.10

+   MariaDB 10.2（或 MySQL Connector）

+   Filezilla Server 0.9

我们将在每一章中处理安装过程和详细信息。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中为本书下载示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

文件下载后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/Windows 7-Zip

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-GUI-Programming-with-CPP-and-Qt5`](https://github.com/PacktPublishing/Hands-On-GUI-Programming-with-CPP-and-Qt5)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnGUIProgrammingwithCPPandQt5_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnGUIProgrammingwithCPPandQt5_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。这是一个例子：“我们在`MainWindow`构造函数中调用`test()`函数。”

代码块设置如下：

```cpp
void MainWindow::test() 
{ 
   int amount = 100; 
   amount -= 10; 
   qDebug() << "You have obtained" << amount << "apples!"; 
} 
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目以粗体设置：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 
   test(); 
} 
```

任何命令行输入或输出都以以下方式编写：

```cpp
********* Start testing of MainWindow ********* 
Config: Using QtTest library 5.9.1, Qt 5.9.1 (i386-little_endian-ilp32 shared (dynamic) debug build; by GCC 5.3.0) 
PASS   : MainWindow::initTestCase() 
PASS   : MainWindow::_q_showIfNotHidden() 
PASS   : MainWindow::testString() 
PASS   : MainWindow::testGui() 
PASS   : MainWindow::cleanupTestCase() 
Totals: 5 passed, 0 failed, 0 skipped, 0 blacklisted, 880ms 
********* Finished testing of MainWindow ********* 
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这样的方式出现在文本中。这是一个例子：“第三个选项是切换书签，它允许您为自己设置书签。”

警告或重要说明会以这种方式出现。

提示和技巧会以这种方式出现。
