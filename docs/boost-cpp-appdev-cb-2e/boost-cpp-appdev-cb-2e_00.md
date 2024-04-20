# 前言

如果您想充分利用 Boost 和 C++的真正力量，并避免在不同情况下使用哪个库的困惑，那么这本书适合您。

从 Boost C++的基础知识开始，您将学习如何利用 Boost 库简化应用程序开发。您将学会将数据转换，例如将字符串转换为数字，数字转换为字符串，数字转换为数字等。资源管理将变得轻而易举。您将了解可以在编译时完成哪些工作以及 Boost 容器的功能。您将学会为高质量、快速和可移植的应用程序开发所需的一切。只需编写一次程序，然后就可以在 Linux、Windows、macOS 和 Android 操作系统上使用。从操作图像到图形、目录、定时器、文件和网络，每个人都会找到一个有趣的主题。

请注意，本书的知识不会过时，因为越来越多的 Boost 库成为 C++标准的一部分。

# 本书涵盖的内容

第一章，“开始编写您的应用程序”，介绍了日常使用的库。我们将看到如何从不同来源获取配置选项，以及使用 Boost 库作者引入的一些数据类型可以做些什么。

第二章，“资源管理”，涉及由 Boost 库引入的数据类型，主要关注指针的使用。我们将看到如何轻松管理资源，以及如何使用能够存储任何功能对象、函数和 lambda 表达式的数据类型。阅读完本章后，您的代码将变得更加可靠，内存泄漏将成为历史。

第三章，“转换和强制转换”，描述了如何将字符串、数字和用户定义的类型相互转换，如何安全地转换多态类型，以及如何在 C++源文件中编写小型和大型解析器。涵盖了日常使用和罕见情况下数据转换的多种方式。

第四章，“编译时技巧”，描述了 Boost 库的一些基本示例，可以用于调整算法的编译时检查，以及其他元编程任务。没有理解这些内容，就无法理解 Boost 源代码和其他类似 Boost 的库。

第五章，“多线程”，着重介绍了多线程编程的基础知识以及与之相关的所有内容。

第六章，“任务操作”，展示了将功能对象称为任务。本章的主要思想是，我们可以将所有处理、计算和交互分解为函数对象（任务），并几乎独立地处理每个任务。此外，我们可以不阻塞一些慢操作（例如从套接字接收数据或等待超时），而是提供一个回调任务，并继续处理其他任务。一旦操作系统完成慢操作，我们的回调将被执行。

第七章，“字符串操作”，展示了改变、搜索和表示字符串的不同方面。我们将看到如何使用 Boost 库轻松完成一些常见的与字符串相关的任务。它涉及非常常见的字符串操作任务。

第八章，“元编程”，介绍了一些酷而难以理解的元编程方法。在本章中，我们将深入了解如何将多种类型打包成单个类似元组的类型。我们将创建用于操作类型集合的函数，看到如何改变编译时集合的类型，以及如何将编译时技巧与运行时混合使用。

第九章《容器》介绍了 boost 容器及与之直接相关的内容。本章提供了关于 Boost 类的信息，这些类可以在日常编程中使用，可以使您的代码运行速度更快，开发新应用程序更容易。

第十章《收集平台和编译器信息》描述了用于检测编译器、平台和 Boost 特性的不同辅助宏--这些宏广泛用于 boost 库，并且对于编写能够使用任何编译器标志的可移植代码至关重要。

第十一章《与系统一起工作》提供了对文件系统的更详细的了解，以及如何创建和删除文件。我们将看到数据如何在不同的系统进程之间传递，如何以最大速度读取文件，以及如何执行其他技巧。

第十二章《触类旁通》致力于一些大型库，并为您提供一些入门基础知识。

# 您需要为本书做好准备

您需要一个现代的 C++编译器，Boost 库（任何版本都可以，建议使用 1.65 或更高版本），以及 QtCreator/qmake，或者只需访问[`apolukhin.GitHub.io/Boost-Cookbook/`](http://apolukhin.github.io/Boost-Cookbook/)在线运行和实验示例。

# 这本书适用对象

这本书适用于希望提高对 Boost 的了解并希望简化其应用程序开发流程的开发人员。假定具有先前的 C++知识和标准库的基本知识。

# 章节

在本书中，您将经常看到几个标题（准备工作，如何做...，它是如何工作的...，还有更多...，另请参阅）。为了清晰地说明如何完成配方，我们使用这些部分如下：

# 准备工作

本节告诉您配方中可以期望的内容，并描述了为配方设置任何软件或任何预备设置所需的步骤。

# 如何做...

本节包含遵循该配方所需的步骤。

# 它是如何工作...

本节通常包括对前一节发生的事情的详细解释。

# 还有更多...

本节包含有关配方的其他信息，以使读者更加了解配方。

# 另请参阅

本节提供了有关配方的其他有用信息的链接。

# 约定

在本书中，您将找到一些区分不同类型信息的文本样式。以下是这些样式的一些示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：

“请记住，这个库不仅仅是一个头文件，所以您的程序必须链接到`libboost_program_options`库”。

代码块设置如下：

```cpp
#include <boost/program_options.hpp> 
#include <iostream>
namespace opt = boost::program_options; 
int main(int argc, char *argv[])
{
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目以粗体显示：

```cpp
#include <boost/program_options.hpp> 
#include <iostream>
namespace opt = boost::program_options; 
int main(int argc, char *argv[])
```

任何命令行输入或输出都以以下方式编写：

```cpp
 $ ./our_program.exe --apples=10 --oranges=20
Fruits count: 30
```

**新术语**和**重要单词**以粗体显示。

警告或重要说明会以这样的形式出现在一个框中。

提示和技巧会出现在这样的形式中。

# 读者反馈

我们的读者的反馈总是受欢迎的。让我们知道您对本书的看法--您喜欢或不喜欢的内容。读者的反馈对我们很重要，因为它可以帮助我们开发出您真正喜欢的标题。要向我们发送一般反馈，只需发送电子邮件至`feedback@packtpub.com`，并在您的消息主题中提及书名。如果您在某个主题上有专业知识，并且有兴趣编写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您是 Packt 书籍的自豪所有者，我们有很多东西可以帮助您充分利用您的购买。

# 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。您可以按照以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册到我们的网站。

1.  将鼠标指针悬停在顶部的“支持”选项卡上。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名。

1.  选择您要下载代码文件的书籍。

1.  从下拉菜单中选择您购买这本书的地方。

1.  点击“代码下载”。

下载文件后，请确保使用最新版本的以下软件解压缩文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`GitHub.com/PacktPublishing/Boost-Cpp-Application-Development-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Boost-Cpp-Application-Development-Cookbook-Second-Edition)。我们还有来自丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快去看看吧！

本食谱中介绍的示例源代码文件也托管在作者的 GitHub 存储库中。您可以访问作者的存储库[`GitHub.com/apolukhin/Boost-Cookbook`](https://github.com/apolukhin/Boost-Cookbook)获取代码的最新版本。

# 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误确实会发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——我们将不胜感激，如果您能向我们报告。通过这样做，您可以帮助其他读者避免挫折，并帮助我们改进本书的后续版本。如果您发现任何勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)报告，选择您的书，点击“勘误提交表”链接，并输入您的勘误详情。一旦您的勘误被验证，您的提交将被接受，并且勘误将被上传到我们的网站或添加到该标题的勘误部分下的任何现有勘误列表中。

要查看先前提交的勘误表，请转到[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)并在搜索框中输入书名。所需信息将出现在勘误部分下。

# 盗版

互联网上的版权材料盗版是所有媒体的持续问题。在 Packt，我们非常重视保护我们的版权和许可。如果您在互联网上发现我们作品的任何非法副本，请立即向我们提供位置地址或网站名称，以便我们采取补救措施。

请通过`copyright@packtpub.com`与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您帮助保护我们的作者和我们提供有价值内容的能力。

# 问题

如果您对本书的任何方面有问题，可以通过`questions@packtpub.com`与我们联系，我们将尽力解决问题。