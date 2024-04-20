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

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：发送电子邮件至`feedback@packtpub.com`，并在主题中提及书名。如果您对本书的任何方面有疑问，请发送电子邮件至`questions@packtpub.com`。

勘误：尽管我们已经尽最大努力确保内容的准确性，但错误确实会发生。如果您在本书中发现错误，我们将不胜感激，如果您能向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书，点击勘误提交表格链接，并输入详细信息。

盗版：如果您在互联网上发现我们作品的任何形式的非法副本，我们将不胜感激，如果您能向我们提供位置地址或网站名称。请通过`copyright@packtpub.com`与我们联系，并提供材料链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 在 Qt 中发现工具

Qt 配备了一套工具，使程序员的生活更加轻松。其中一个工具是 Qt Creator（如下截图所示），它是一个**集成开发环境**（IDE），包括代码编辑器和**图形用户界面**（GUI）设计师，与其他 Qt 工具如编译器、调试器等紧密配合。其中最吸引人的工具当然是 GUI 设计师，它带有两种不同类型的编辑器：一个用于基于小部件的应用程序，称为 Qt Designer，另一个用于 Qt 快速应用程序，称为 Qt 快速设计师。当您打开相关文件格式时，这两个工具都可以直接在 Qt Creator 中访问。Qt Creator 还包括一个内置的文档查看器，称为 Qt Assistant。这真的很方便，因为您可以通过简单地将鼠标悬停在源代码中的类名上，并按下 *F1* 键来查找关于某个 Qt 类或函数的解释。然后 Qt Assistant 将被打开，并显示与 Qt 类或函数相关的文档。

![](img/0983f1e7-423c-40b7-9965-101a2c0a1be1.png)

# 评论

请留下评论。一旦您阅读并使用了本书，为什么不在购买它的网站上留下评论呢？潜在的读者可以看到并使用您的客观意见来做出购买决定，我们在 Packt 可以了解您对我们产品的看法，我们的作者可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packtpub.com](https://www.packtpub.com/)。
