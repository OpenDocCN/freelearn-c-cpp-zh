# 前言

本书将向读者提供关于 C++17 和 C++20 标准的 C++程序的细节，以及它们是如何编译、链接和执行的。它还将涵盖内存管理的工作原理，内存管理问题的最佳实践，类的实现方式，编译器如何优化代码，以及编译器在支持类继承、虚函数和模板方面的方法。

本书还将告诉读者如何将内存管理、面向对象编程、并发和设计模式应用于创建面向世界的生产应用程序。

读者将学习高效数据结构和算法的内部细节，并将了解如何衡量和比较它们，以选择最适合特定问题的内容。

本书将帮助读者将系统设计技能与基本设计模式结合到 C++应用程序中。

此外，本书还介绍了人工智能世界，包括使用 C++编程语言进行机器学习的基础知识。

本书结束时，读者应该有足够的信心，能够使用高效的数据结构和算法设计和构建真实的可扩展 C++应用程序。

# 本书的读者对象

寻求了解与语言和程序结构相关的细节，或者试图通过深入程序的本质来提高自己的专业知识，以设计可重用、可扩展架构的 C++开发人员将从本书中受益。那些打算使用 C++17 和 C++20 的新特性设计高效数据结构和算法的开发人员也将受益。

# 本书内容

[第一章]《构建 C++应用程序简介》包括对 C++世界、其应用程序以及语言标准的最新更新的介绍。本章还对 C++涵盖的主题进行了良好的概述，并介绍了代码编译、链接和执行阶段。

[第二章]《使用 C++进行低级编程》专注于讨论 C++数据类型、数组、指针、指针的寻址和操作，以及条件语句、循环、函数、函数指针和结构的低级细节。本章还包括对结构（structs）的介绍。

[第三章]《面向对象编程的细节》深入探讨了类和对象的结构，以及编译器如何实现对象的生命周期。本章结束时，读者将了解继承和虚函数的实现细节，以及 C++中面向对象编程的基本内部细节。

[第四章]《理解和设计模板》介绍了 C++模板、模板函数的示例、模板类、模板特化以及一般的模板元编程。特性和元编程将融入 C++应用程序的魔力。

[第五章]《内存管理和智能指针》深入探讨了内存部分、分配和管理的细节，包括使用智能指针来避免潜在的内存泄漏。

[第六章]《深入 STL 中的数据结构和算法》介绍了数据结构及其 STL 实现。本章还包括对数据结构的比较，并讨论了实际应用的适当性，提供了真实世界的例子。

第七章，*函数式编程*，着重于函数式编程，这是一种不同的编程范式，使读者能够专注于代码的“功能”而不是“物理”结构。掌握函数式编程为开发人员提供了一种新的技能，有助于为问题提供更好的解决方案。

第八章，*并发和多线程*，着重于如何通过利用并发使程序运行更快。当高效的数据结构和高效的算法达到程序性能的极限时，并发就会发挥作用。

第九章，*设计并发数据结构*，着重利用数据结构和并发性来设计基于锁和无锁的并发数据结构。

第十章，*设计面向世界的应用程序*，着重于将前几章学到的知识融入到使用设计模式设计健壮的现实世界应用程序中。本章还包括了理解和应用领域驱动设计，通过设计一个亚马逊克隆来实现。

第十一章，*使用设计模式设计策略游戏*，将前几章学到的知识融入到使用设计模式和最佳实践设计策略游戏中。

第十二章，*网络和安全*，介绍了 C++中的网络编程以及如何利用网络编程技能构建一个 dropbox 后端克隆。本章还包括了如何确保编码最佳实践的讨论。

第十三章，*调试和测试*，着重于调试 C++应用程序和避免代码中的错误的最佳实践，应用静态代码分析以减少程序中的问题，介绍和应用测试驱动开发和行为驱动开发。本章还包括了行为驱动开发和 TDD 之间的区别以及使用案例。

第十四章，*使用 Qt 进行图形用户界面*，介绍了 Qt 库及其主要组件。本章还包括了对 Qt 跨平台性质的理解，通过构建一个简单的桌面客户端来延续 dropbox 示例。

第十五章，*在机器学习任务中使用 C++*，涉及了人工智能概念和领域的最新发展的简要介绍。本章还包括了机器学习和诸如回归分析和聚类等任务的介绍，以及如何构建一个简单的神经网络。

第十六章，*实现基于对话框的搜索引擎*，涉及将之前章节的知识应用到设计一个高效的搜索引擎中，描述为*基于对话框*，因为它通过询问（和学习）用户的相应问题来找到正确的文档。

# 为了充分利用本书

基本的 C++经验，包括对内存管理、面向对象编程和基本数据结构和算法的熟悉，将是一个很大的优势。如果你渴望了解这个复杂程序在幕后是如何工作的，以及想要理解 C++应用设计的编程概念和最佳实践的细节，那么你应该继续阅读本书。

| **书中涉及的软件/硬件** | **操作系统要求** |
| --- | --- |
| g++编译器 | Ubuntu Linux 是一个优势，但不是必需的 |

你还需要在计算机上安装 Qt 框架。相关细节在相关章节中有介绍。

在撰写本书时，并非所有 C++编译器都支持所有新的 C++20 功能，考虑使用最新版本的编译器以测试本章介绍的更多功能。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在“搜索”框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩软件解压文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`static.packt-cdn.com/downloads/9781838552657_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781838552657_ColorImages.pdf)

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子："前面的代码声明了两个`readonly`属性，并分配了值"。

代码块设置如下：

```cpp
Range book = 1..4;
var res = Books[book] ;
Console.WriteLine($"\tElement of array using Range: Books[{book}] => {Books[book]}");
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
private static readonly int num1=5;
private static readonly int num2=6;
```

任何命令行输入或输出都以以下形式书写：

```cpp
dotnet --info
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中出现。这是一个例子："从管理面板中选择系统信息"。

警告或重要提示会以这种形式出现。

提示和技巧会以这种形式出现。

# 联系我们

我们始终欢迎读者的反馈。

一般反馈：如果您对本书的任何方面有疑问，请在消息主题中提及书名，并发送电子邮件至`customercare@packtpub.com`。

勘误：尽管我们已经尽最大努力确保内容的准确性，但错误确实会发生。如果您在本书中发现错误，我们将不胜感激您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，点击勘误提交表格链接，并输入详细信息。

盗版：如果您在互联网上发现我们作品的任何非法副本，请向我们提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在购买书籍的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决定，我们在 Packt 可以了解您对我们产品的看法，我们的作者可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。