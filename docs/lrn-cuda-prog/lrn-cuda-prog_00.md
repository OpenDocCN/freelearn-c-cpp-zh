# 前言

不要在第一次胜利后就休息。因为如果你在第二次失败，会有更多的人说你的第一次胜利只是运气。

- A. P. J. Abdul Kalam

传统上，计算需求与中央处理单元（CPU）相关联，CPU 已经从单核发展到现在的多核。每一代新的 CPU 都提供了更多的性能，但科学和高性能计算社区每年都要求更多的性能，导致应用程序需求与硬件/软件堆栈提供的计算之间存在差距。与此同时，传统上用于视频图形的新架构也进入了科学领域。图形处理单元（GPU）——基本上是用于加速计算机图形的并行计算处理器——在 2007 年 CUDA 推出时进入了 HPC 领域。CUDA 成为了使用 GPU 进行通用计算的事实标准；即非图形应用程序。

自 CUDA 诞生以来，已经发布了许多版本，现在 CUDA 的版本为 10.x。每个版本都提供支持新硬件架构的新功能。本书旨在帮助您学习 GPU 并行编程，并指导您在现代应用中的应用。在它的帮助下，您将能够发现现代 GPU 架构的 CUDA 编程方法。本书不仅将指导您了解 GPU 功能、工具和 API，还将帮助您了解如何使用示例并行编程算法来分析性能。本书将确保您获得丰富的优化经验和对 CUDA 编程平台的洞察，包括各种库、开放加速器（OpenACC）和其他语言。随着您的进步，您将发现如何在一个盒子或多个盒子中利用多个 GPU 生成额外的计算能力。最后，您将探索 CUDA 如何加速深度学习算法，包括卷积神经网络（CNN）和循环神经网络（RNN）。

本书旨在成为任何新手或初学者开发者的入门点。但到最后，您将能够为不同领域编写优化的 CUDA 代码，包括人工智能。

如果您符合以下情况，这本书将是一个有用的资源：

+   您是 HPC 或并行计算的新手

+   您有代码，并希望通过将并行计算应用于 GPU 来提高其性能

+   您是深度学习专家，想利用 GPU 加速深度学习算法，如 CNN 和 RNN

+   您想学习优化代码和分析 GPU 应用性能的技巧和窍门，并发现优化策略

+   您想了解最新的 GPU 功能，以及高效的、分布式的多 GPU 编程。

如果您觉得自己属于以上任何一类，请加入我们一起踏上这段旅程。

# 这本书适合谁

这本初学者级别的书适用于希望深入研究并行计算、成为高性能计算社区的一部分并构建现代应用程序的程序员。假定具有基本的 C 和 C++编程经验。对于深度学习爱好者，本书涵盖了 Python InterOps、DL 库以及性能估计的实际示例。

# 为了充分利用这本书

本书适用于完全初学者和刚开始学习并行计算的人。除了计算机体系结构的基础知识外，不需要任何特定的知识，假定具有 C/C++编程经验。对于深度学习爱好者，在[第十章]（d0e9e8ff-bc17-4031-bb0e-1cfd310aff6f.xhtml）中，还提供了基于 Python 的示例代码，因此该章节需要一些 Python 知识。

本书的代码主要是在 Linux 环境中开发和测试的。因此，熟悉 Linux 环境是有帮助的。任何最新的 Linux 版本，如 CentOS 或 Ubuntu，都可以。代码可以使用 makefile 或命令行进行编译。本书主要使用免费软件堆栈，因此无需购买任何软件许可证。本书中将始终使用的两个关键软件是 CUDA 工具包和 PGI 社区版。

由于本书主要涵盖了利用 CUDA 10.x 的最新 GPU 功能，为了充分利用所有培训材料，最新的 GPU 架构（Pascal 及更高版本）将是有益的。虽然并非所有章节都需要最新的 GPU，但拥有最新的 GPU 将有助于您重现本书中实现的结果。每一章都有一个关于首选或必备 GPU 架构的部分，位于*技术要求*部分。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择支持选项卡。

1.  点击“代码下载”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保您使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learn-CUDA-Programming`](https://github.com/PacktPublishing/Learn-CUDA-Programming)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。快来看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：`static.packt-cdn.com/downloads/9781788996242_ColorImages.pdf`。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个例子：“请注意，`cudaMemcpy`有一个异步的替代方案。”

代码块设置如下：

```cpp
#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] \
        From device\n", threadIdx.x,blockIdx.x);
}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
int main(void) {
    printf("Hello World from host!\n");
    print_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

任何命令行输入或输出都以如下形式编写：

```cpp
$ nvcc -o hello_world hello_world.cu
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单中的单词或对话框中的单词会以这种形式出现在文本中。以下是一个例子：“对于 Windows 用户，在 VS 项目属性对话框中，您可以在 CUDA C/C++ | Device | Code Generation 中指定您的 GPU 的计算能力。”

警告或重要说明如下。

提示和技巧如下。

# 联系我们

我们随时欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在消息主题中提及书名，并发送电子邮件至`customercare@packtpub.com`。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误确实会发生。如果您在本书中发现错误，我们将不胜感激。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法复制，请向我们提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料链接。

**如果您有兴趣成为作者**：如果您在某个专业领域有专长，并且有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。当您阅读并使用了这本书之后，为什么不在购买它的网站上留下评论呢？潜在的读者可以看到并使用您的客观意见来做出购买决定，我们在 Packt 可以了解您对我们产品的看法，我们的作者也可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
