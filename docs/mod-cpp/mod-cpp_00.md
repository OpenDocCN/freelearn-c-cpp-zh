# 前言

C++是最广泛使用的编程语言之一。它快速，灵活，高效，用于解决许多编程问题。

这个学习路径的目标是使您熟悉并熟悉 C++。通过学习语言结构，函数和类，您将熟悉 C++编程的构造，这将帮助您识别代码中的执行流程。您将探索并了解 C++标准库的重要性，以及为编写更好，更快的程序而进行的内存分配。

这个学习路径还涉及理解高级 C++编程所面临的挑战。您将学习高级主题，如多线程，网络，并发性，性能，元编程，lambda 表达式，正则表达式，测试等，以食谱的形式。

通过这个学习路径的结束，您将成为 C++的专家。

# 这本书适合谁

这个学习路径是为想要在 C++中建立坚实基础的开发人员设计的。一台计算机，一个互联网连接，以及学习如何在 C++中编码的愿望就是您开始这个学习路径所需要的一切。

# 这本书涵盖了什么

第一章《理解语言特性》涵盖了 C++语句和表达式，常量，变量，运算符，以及如何控制应用程序中的执行流程。

第二章《使用内存，数组和指针》涵盖了在 C++应用程序中如何分配和使用内存，如何使用内置数组，C++引用的作用，以及如何使用 C++指针来访问内存。

第三章《使用函数》解释了如何定义函数，如何通过引用和按值传递参数，使用可变数量的参数，创建和使用函数指针，以及定义模板函数和重载运算符。

第四章《类》描述了如何通过类定义新类型以及类中使用的各种特殊函数，如何将类实例化为对象以及如何销毁它们，以及如何通过指针访问对象以及如何编写模板类。

第五章《使用标准库容器》涵盖了所有 C++标准库容器类以及如何使用它们与迭代器和标准算法，以便您可以操作容器中的数据。

第六章《使用字符串》描述了标准 C++字符串类的特性，数字数据和字符串之间的转换，国际化字符串，以及使用正则表达式来搜索和操作字符串。

第七章《诊断和调试》解释了如何准备您的代码以提供诊断，并使其能够进行调试，应用程序是如何终止的，是突然还是优雅地，以及如何使用 C++异常。

第八章《学习现代核心语言特性》教授了现代核心语言特性，包括类型推断，统一初始化，作用域枚举，基于范围的 for 循环，结构化绑定等。

第九章《使用数字和字符串》讨论了如何在数字和字符串之间进行转换，生成伪随机数，使用正则表达式以及各种类型的字符串。

第十章《探索函数》深入探讨了默认和删除的函数，可变模板，lambda 表达式和高阶函数。

第十一章，*标准库容器、算法和迭代器*，向您介绍了几种标准容器，许多算法，并教您如何编写自己的随机访问迭代器。

第十二章，*数学问题*，包含一系列数学练习，为您做好准备，以应对接下来章节中更具挑战性的问题。

第十三章，*语言特性*，提出了一些问题供您练习运算符重载，移动语义，用户定义的文字，以及模板元编程方面的问题，如可变函数，折叠表达式和类型特征。

第十四章，*字符串和正则表达式*，存在一些字符串操作问题，例如在字符串和其他数据类型之间转换，拆分和连接字符串，以及处理正则表达式。

第十五章，*流和文件系统*，涵盖了输出流操作和使用 C++17 `filesystem`库处理文件和目录。

第十六章，*日期和时间*，为即将到来的 C++20 对`chrono`库的扩展做准备，其中包含了几个日历和时区问题，您可以使用`date`库解决这些问题，新的标准扩展就是基于这个库的。

第十七章，*算法和数据结构*，是最大的章节之一，包含各种问题，您需要利用现有的标准算法；另一些问题是您需要实现自己的通用算法或数据结构，比如循环缓冲区和优先队列。本章以两个相当有趣的问题结束，分别是道金斯的鼠鼠程序和康威的生命游戏程序，您可以从中了解到进化算法和细胞自动机。

# 为了充分利用本书

读者应该具备以下环境配置：

1.  C++11（英特尔、IBM、Sun、苹果和微软，以及开源 GCC）

1.  Visual C++ 2017 社区版

1.  Windows 上的 VC++ 2017

1.  在 Linux 和 Mac 上使用 GCC 7.0 或 Clang 5.0

如果您没有最新版本的编译器，或者想尝试另一个编译器，您可以使用在线可用的编译器。虽然有各种在线平台可供您使用，但我推荐使用[`wandbox.org/`](https://wandbox.org/)来使用 GCC 和 Clang，以及[`webcompiler.cloudapp.net/`](http://webcompiler.cloudapp.net/)来使用 VC++。

在使用支持 C++17 的编译器时，您将需要一个所需库的完整列表。

# 如何为 Visual Studio 2017 生成项目

为了生成 Visual Studio 2017 项目以定位 x86 平台，请按照以下步骤进行：

1.  打开命令提示符并转到源代码根文件夹中的`build`目录。

1.  执行以下 CMake 命令：

``cmake -G "Visual Studio 15 2017" .. -DCMAKE_USE_WINSSL=ON -DCURL_WINDOWS_SSPI=ON -DCURL_LIBRARY=libcurl -DCURL_INCLUDE_DIR=..\libs\curl\include -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DUSE_MANUAL=OFF``

1.  完成后，可以在`build/cppchallenger.sln`找到 Visual Studio 解决方案。

如果要将目标定为 x64 平台，可以使用名为`"Visual Studio 15 2017 Win64"`的生成器。Visual Studio 2017 15.4 同时支持`filesystem`（作为实验性库）和`std::optional`。如果您使用之前的版本，或者只想使用 Boost 库，可以使用以下命令生成项目，前提是您已经正确安装了 Boost：

```cpp
cmake -G "Visual Studio 15 2017" .. -DCMAKE_USE_WINSSL=ON -DCURL_WINDOWS_SSPI=ON -DCURL_LIBRARY=libcurl -DCURL_INCLUDE_DIR=..\libs\curl\include -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DUSE_MANUAL=OFF -DBOOST_FILESYSTEM=ON -DBOOST_OPTIONAL=ON -DBOOST_INCLUDE_DIR=<path_to_headers> -DBOOST_LIB_DIR=<path_to_libs>
```

确保头文件和静态库文件的路径不包含尾随反斜杠（即`\`）。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  请在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明进行操作。

文件下载后，请确保使用最新版本的解压缩软件解压缩文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Modern-C-plus-plus-Efficient-and-Scalable-Application-Development`](https://github.com/PacktPublishing/Modern-C-plus-plus-Efficient-and-Scalable-Application-Development)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的书籍和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/Cplusplus_Efficient_and_Scalable_Application_Development.pdf`](https://www.packtpub.com/sites/default/files/downloads/Cplusplus_Efficient_and_Scalable_Application_Development.pdf)

# 使用的约定

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“作者打算输入`c = a + 8 / b + 1;`，但是他们按逗号键而不是斜杠键。”

代码块设置如下：

```cpp
inline auto mult(int lhs, int rhs) -> int 
    { 
        return lhs * rhs; 
    }
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
if (op == ',' || op == '.' || op < '+' || op > '/') 
    { 
        cout << endl << "operator not recognized" << endl; 
        usage(); 
        return 1; 
    }
```

任何命令行输入或输出都以以下方式编写：

```cpp
C:\Beginning_C++Chapter_02\cl /EHsc calc.cpp
```

**粗体**：新术语和重要单词以粗体显示。例如，屏幕上看到的单词，例如菜单或对话框中的单词，会以这种方式出现在文本中：“函数的**调用约定**决定了调用函数或被调用函数负责执行此操作。”

警告或重要说明会以这种方式出现。

技巧和窍门会以这种方式出现。