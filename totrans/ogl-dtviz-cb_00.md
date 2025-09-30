# 前言

OpenGL 是一个多平台、跨语言、硬件加速的应用程序编程接口，用于高性能渲染 2D 和 3D 图形。OpenGL 的一种新兴用途是在从医学成像、建筑和工程中的模拟或建模，到前沿的移动/可穿戴计算等领域的实时、高性能数据可视化应用开发。确实，随着数据集变得更大和更复杂，尤其是在大数据的演变中，使用传统的非图形硬件加速方法进行数据可视化变得越来越具有挑战性。从移动设备到复杂的高性能计算集群，OpenGL 库为开发者提供了一个易于使用的接口，以创建实时、3D 的惊人视觉效果，适用于广泛的交互式应用。

本书包含一系列针对经验不足的初学者和希望探索最先进技术的更高级用户定制的动手实践配方。我们从第一章到第三章通过演示如何在 Windows、Mac OS X 和 Linux 中设置环境，以及如何使用原语渲染基本的 2D 数据集和交互式地学习更复杂的 3D 体积数据集开始，对 OpenGL 进行基本介绍。这部分只需要 OpenGL 2.0 或更高版本，以便即使拥有较老图形硬件的读者也可以尝试代码。在第四章到第六章中，我们过渡到更高级的技术（需要 OpenGL 3.2 或更高版本），例如用于图像/视频处理的纹理映射、从 3D 范围感应相机渲染深度传感器数据的点云渲染，以及立体 3D 渲染。最后，在第七章到第九章中，我们通过介绍在日益强大的移动（基于 Android）计算平台上使用 OpenGL ES 3.0 以及在移动设备上开发高度交互式、增强现实应用来结束本书。

本书中的每个配方都为读者提供了一组可以导入现有项目中的标准函数，这些函数可以成为创建各种实时、交互式数据可视化应用的基础。本书还利用了一系列流行的开源库，如 GLFW、GLM、Assimp 和 OpenCV，以简化应用程序开发，并通过启用 OpenGL 上下文管理和 3D 模型加载以及使用最先进的计算机视觉算法进行图像/视频处理来扩展 OpenGL 的功能。

# 本书涵盖内容

第一章, *开始使用 OpenGL*，介绍了创建基于 OpenGL 的数据可视化应用所需的必需开发工具，并提供了在 Windows、Mac OS X 和 Linux 中设置我们第一个 OpenGL 演示应用的逐步教程。

第二章, *OpenGL 原语和 2D 数据可视化*，专注于 OpenGL 2.0 原语的使用，如点、线和三角形，以实现数据的基本 2D 可视化，包括时间序列，如心电图（ECG）。

第三章, *交互式 3D 数据可视化*，基于之前讨论的基本概念，并将演示扩展到包含更复杂的 OpenGL 功能以进行 3D 渲染。

第四章, *使用纹理映射渲染 2D 图像和视频*，介绍了 OpenGL 技术来可视化另一类重要的数据集——涉及图像或视频的数据集。这类数据集在许多领域都很常见，包括医学成像应用。

第五章, *为 3D 范围感应相机渲染点云数据*，介绍了用于可视化另一类有趣且新兴的数据类——来自 3D 范围感应相机的深度信息的技术。

第六章, *使用 OpenGL 渲染立体 3D 模型*，展示了如何使用 OpenGL 这项令人惊叹的立体 3D 技术来可视化数据。OpenGL 本身并不提供加载、保存或操作 3D 模型的机制。因此，为了支持这一点，我们将集成一个名为 Assimp 的新库到我们的代码中。

第七章, *使用 OpenGL ES 3.0 在移动平台上进行实时图形渲染的介绍*，通过展示如何设置 Android 开发环境并在最新的移动设备上创建第一个基于 Android 的应用程序（从智能手机到平板电脑），使用嵌入式系统 OpenGL（OpenGL ES）过渡到一个越来越强大且无处不在的计算平台。

第八章, *移动设备上的交互式实时数据可视化*，展示了如何通过使用内置的运动传感器，称为惯性测量单元（IMUs），以及移动设备上发现的多点触控界面来交互式地可视化数据。

第九章, *基于增强现实在移动或可穿戴平台上的可视化*，介绍了在基于 Android 的通用移动设备上创建第一个基于 AR 的应用程序所需的基本构建块：OpenCV 用于计算机视觉，OpenGL 用于图形渲染，以及 Android 的传感器框架用于交互。

# 你需要这本书的内容

本书支持广泛的平台和开源库，从基于 Windows、Mac OS X 或 Linux 的桌面应用程序到基于 Android 的便携式移动应用程序。您需要对 C/C++ 编程有基本的了解，并对基本线性代数和几何模型有背景知识。

第一章到第三章的要求如下：

+   **OpenGL 版本**: 2.0 或更高版本（易于在旧版图形硬件上进行测试）。

+   **平台**: Windows、Mac OS X 或 Linux。

+   **库**: GLFW 用于 OpenGL 窗口上下文管理和处理用户输入。不需要额外的库，这使得它很容易集成到现有项目中。

+   **开发工具**: Windows Visual Studio 或 Xcode、CMake 和 gcc。

第四章到第六章的要求如下：

+   **OpenGL 版本**: 3.2 或更高版本。

+   **平台**: Windows、Mac OS X 或 Linux。

+   **库**: Assimp 用于 3D 模型加载，SOIL 用于图像和纹理加载，GLEW 用于运行时 OpenGL 扩展支持，GLM 用于矩阵运算，以及 OpenCV 用于图像处理

+   **开发工具**: Windows Visual Studio 或 Xcode、CMake 和 gcc。

第七章到第九章的要求如下：

+   **OpenGL 版本**: OpenGL ES 3.0

+   **平台**: 开发时使用 Linux 或 Mac OS X，部署时使用 Android OS 4.3 及更高版本（API 18 及更高版本）

+   **库**: OpenCV for Android 和 GLM

+   **开发工具**: Mac OS X 或 Linux 中的 Android SDK、Android NDK 和 Apache Ant

更多信息，请注意，本书中的代码是在所有支持的平台上的以下库和开发工具中构建和测试的：

+   OpenCV 2.4.9 ([`opencv.org/downloads.html`](http://opencv.org/downloads.html))

+   OpenCV 3.0.0 for Android ([`opencv.org/downloads.html`](http://opencv.org/downloads.html))

+   SOIL ([`www.lonesock.net/soil.html`](http://www.lonesock.net/soil.html))

+   GLEW 1.12.0 ([`glew.sourceforge.net/`](http://glew.sourceforge.net/))

+   GLFW 3.0.4 ([`www.glfw.org/download.html`](http://www.glfw.org/download.html))

+   GLM 0.9.5.4 ([`glm.g-truc.net/0.9.5/index.html`](http://glm.g-truc.net/0.9.5/index.html))

+   Assimp 3.0 ([`assimp.sourceforge.net/main_downloads.html`](http://assimp.sourceforge.net/main_downloads.html))

+   Android SDK r24.3.3 ([`developer.android.com/sdk/index.html`](https://developer.android.com/sdk/index.html))

+   Android NDK r10e ([`developer.android.com/ndk/downloads/index.html`](https://developer.android.com/ndk/downloads/index.html))

+   Windows Visual Studio 2013 ([`www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx`](https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx))

+   CMake 3.2.1 ([`www.cmake.org/download/`](http://www.cmake.org/download/))

# 本书面向的对象

这本书的目标是面向任何对使用现代图形硬件创建令人印象深刻的可视化工具感兴趣的人。无论你是开发者、工程师还是科学家，如果你对探索 OpenGL 在数据可视化方面的强大功能感兴趣，这本书就是为你准备的。虽然推荐熟悉 C/C++，但不需要有 OpenGL 的先验经验。

# 部分

在这本书中，你会发现一些频繁出现的标题（准备就绪、如何操作、工作原理、更多内容、相关内容）。

为了清楚地说明如何完成食谱，我们使用以下这些部分：

## 准备就绪

本节告诉你在食谱中可以期待什么，并描述了如何设置任何软件或任何为食谱所需的初步设置。

## 如何操作...

本节包含遵循食谱所需的步骤。

## 工作原理...

本节通常包含对前一个章节发生情况的详细解释。

## 更多内容...

本节包含有关食谱的附加信息，以便使读者对食谱有更多的了解。

## 相关内容

本节提供了对其他有用信息的链接，这些信息对食谱很有帮助。

# 惯例

在这本书中，你会发现许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示：“我们假设所有文件都保存在名为`code`的顶级目录中，而`main.cpp`文件保存在`/code/Tutorial1`子目录中。”

代码块设置如下：

```cpp
typedef struct
{
  GLfloat x, y, z;
} Data;
```

任何命令行输入或输出都按如下方式编写：

```cpp
sudo port install glfw

```

**新术语**和**重要词汇**以粗体显示。屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“选择**空项目**选项，然后点击**完成**。”

### 注意

警告或重要注意事项以如下框中的方式出现。

### 小贴士

小技巧和技巧看起来像这样。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们你对这本书的看法——你喜欢什么或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出你真正能从中获得最大收益的标题。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书籍的标题。

如果你在一个领域有专业知识，并且你对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助你从购买中获得最大收益。

## 下载示例代码

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)的账户中下载示例代码文件，适用于您购买的所有 Packt Publishing 书籍。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 下载本书的彩色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。这些彩色图像将帮助您更好地理解输出的变化。您可以从以下链接下载此文件：[`www.packtpub.com/sites/default/files/downloads/9727OS.pdf`](https://www.packtpub.com/sites/default/files/downloads/9727OS.pdf)。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下现有的勘误列表中。

要查看之前提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分下。

## 盗版

互联网上对版权材料的盗版是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现任何形式的非法复制我们的作品，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过`<copyright@packtpub.com>`与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过`<questions@packtpub.com>`联系我们，我们将尽力解决问题。
