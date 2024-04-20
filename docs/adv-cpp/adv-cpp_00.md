# 前言

## 关于

本节简要介绍了作者、本书的内容、开始所需的技术技能以及完成所有包含的活动和练习所需的硬件和软件要求。

## 关于本书

C++是最广泛使用的编程语言之一，应用于各种领域，从游戏到图形用户界面（GUI）编程，甚至操作系统。如果您希望扩展职业机会，掌握 C++的高级特性至关重要。

该书从高级 C++概念开始，帮助您解析复杂的 C++类型系统，并了解编译的各个阶段如何将源代码转换为目标代码。然后，您将学习如何识别需要使用的工具，以控制执行流程，捕获数据并传递数据。通过创建小模型，您甚至会发现如何使用高级 lambda 和捕获，并在 C++中表达常见的 API 设计模式。随着后续章节的学习，您将探索通过学习内存对齐、缓存访问以及程序运行所需的时间来优化代码的方法。最后一章将帮助您通过了解现代 CPU 分支预测以及如何使您的代码对缓存友好来最大化性能。

通过本书，您将发展出与其他 C++程序员不同的编程技能。

### 关于作者

加齐汗·阿兰库斯（Gazihan Alankus）在华盛顿大学获得计算机科学博士学位。目前，他是土耳其伊兹密尔经济大学的助理教授。他在游戏开发、移动应用开发和人机交互方面进行教学和研究。他是 Dart 的 Google 开发专家，并与他在 2019 年创立的公司 Gbot 的学生一起开发 Flutter 应用程序。

奥莉娜·利津娜（Olena Lizina）是一名拥有 5 年 C++开发经验的软件开发人员。她具有为国际产品公司开发用于监控和管理远程计算机的系统的实际知识，该系统有大量用户。在过去的 4 年中，她一直在国际外包公司为知名汽车公司的汽车项目工作。她参与了不同项目的复杂和高性能应用程序的开发，如 HMI（人机界面）、导航以及与传感器工作的应用程序。

拉克什·马内（Rakesh Mane）在软件行业拥有 18 年的经验。他曾与来自印度、美国和新加坡的熟练程序员合作。他主要使用 C++、Python、shell 脚本和数据库进行工作。在业余时间，他喜欢听音乐和旅行。此外，他喜欢使用软件工具和代码玩耍、实验和破坏东西。

维韦克·纳加拉贾（Vivek Nagarajan）是一名自学成才的程序员，他在上世纪 80 年代开始使用 8 位系统。他曾参与大量软件项目，并拥有 14 年的 C++专业经验。此外，他还在多年间使用了各种语言和框架。他是一名业余举重运动员、DIY 爱好者和摩托车赛手。他目前是一名独立软件顾问。

布赖恩·普莱斯（Brian Price）在各种语言、项目和行业中拥有 30 多年的工作经验，其中包括 20 多年的 C++经验。他曾参与电站模拟器、SCADA 系统和医疗设备的开发。他目前正在为下一代医疗设备开发 C++、CMake 和 Python 软件。他喜欢用各种语言解决难题和欧拉项目。

### 学习目标

通过本书，您将能够：

+   深入了解 C++的解剖和工作流程

+   研究在 C++中编码的不同方法的优缺点

+   测试、运行和调试您的程序

+   将目标文件链接为动态库

+   使用模板、SFINAE、constexpr if 表达式和可变模板

+   应用最佳实践进行资源管理

### 观众

如果您已经使用 C++但想要学习如何充分利用这种语言，特别是对于大型项目，那么这本书适合您。必须具备对编程的一般理解，并且必须具备使用编辑器在项目目录中生成代码文件的知识。还建议具备一些使用强类型语言（如 C 和 C++）的经验。

### 方法

这本快节奏的书旨在通过描述性图形和具有挑战性的练习快速教授您概念。该书将包含“标注”，其中包括关键要点和最常见的陷阱，以保持您的兴趣，同时将主题分解为可管理的部分。

### 硬件要求

为了获得最佳的学生体验，我们建议以下硬件配置：

+   任何具有 Windows、Linux 或 macOS 的入门级 PC/Mac 都足够

+   处理器：双核或等效

+   内存：4 GB RAM（建议 8 GB）

+   存储：35 GB 的可用空间

### 软件要求

您还需要提前安装以下软件：

+   操作系统：Windows 7 SP1 32/64 位，Windows 8.1 32/64 位，或 Windows 10 32/64 位，Ubuntu 14.04 或更高版本，或 macOS Sierra 或更高版本

+   浏览器：Google Chrome 或 Mozilla Firefox

### 安装和设置

在开始阅读本书之前，您需要安装本书中使用的以下库。您将在此处找到安装这些库的步骤。

**安装 CMake**

我们将使用 CMake 版本 3.12.1 或更高版本。我们有两种安装选项。

选项 1：

如果您使用的是 Ubuntu 18.10，可以使用以下命令全局安装 CMake：

```cpp
sudo apt install cmake
```

当您运行以下命令时：

```cpp
cmake –version
```

您应该看到以下输出：

```cpp
cmake version 3.12.1
CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

如果您在此处看到的版本低于 3.12.1（例如 3.10），则应按照以下说明在本地安装 CMake。

选项 2：

如果您使用的是较旧的 Linux 版本，则可能会获得低于 3.12.1 的 CMake 版本。然后，您需要在本地安装它。使用以下命令：

```cpp
wget \
https://github.com/Kitware/CMake/releases/download/v3.15.1/cmake-3.15.1-Linux-x86_64.sh
sh cmake-3.15.1-Linux-x86_64.sh
```

当您看到软件许可证时，请输入*y*并按*Enter*。当询问安装位置时，请输入*y*并再次按 Enter。这应该将其安装到系统中的一个新文件夹中。

现在，我们将将该文件夹添加到我们的路径中。输入以下内容。请注意，第一行有点太长，而且在本文档中换行。您应该将其写成一行，如下所示：

```cpp
echo "export PATH=\"$HOME/cmake-3.15.1-Linux-x86_64/bin:$PATH\"" >> .bash_profile
source .profile
```

现在，当您输入以下内容时：

```cpp
cmake –version
```

您应该看到以下输出：

```cpp
cmake version 3.15.1
CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

在撰写本文时，3.15.1 是当前最新版本。由于它比 3.12.1 更新，这对我们的目的足够了。

**安装 Git**

通过输入以下内容来测试当前安装情况：

```cpp
git --version
```

您应该看到以下行：

```cpp
git version 2.17.1
```

如果您看到以下行，则需要安装`git`：

```cpp
command 'git' not found
```

以下是如何在 Ubuntu 中安装`git`：

```cpp
sudo apt install git
```

**安装 g++**

通过输入以下内容来测试当前安装情况：

```cpp
g++ --version
```

您应该看到以下输出：

```cpp
g++ (Ubuntu 7.4.0-1ubuntu1~18.04) 7.4.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions. There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

如果尚未安装，请输入以下代码进行安装：

```cpp
sudo apt install g++
```

**安装 Ninja**

通过输入以下内容来测试当前安装情况：

```cpp
ninja --version
```

您应该看到以下输出：

```cpp
1.8.2
```

如果尚未安装，请输入以下代码进行安装：

```cpp
sudo apt install ninja-build
```

**安装 Eclipse CDT 和 cmake4eclipse**

有多种安装 Eclipse CDT 的方法。为了获得最新的稳定版本，我们将使用官方安装程序。转到此网站并下载 Linux 安装程序：[`www.eclipse.org/downloads/packages/installer`](https://www.eclipse.org/downloads/packages/installer)。

按照那里的说明并安装**Eclipse IDE for C/C++ Developers**。安装完成后，运行 Eclipse 可执行文件。如果您没有更改默认配置，在终端中输入以下命令将运行它：

```cpp
~/eclipse/cpp-2019-03/eclipse/eclipse
```

您将选择一个工作区文件夹，然后将在主 Eclipse 窗口中看到一个**欢迎**选项卡。

现在，我们将安装`cmake4eclipse`。一个简单的方法是访问该网站，并将**安装**图标拖到 Eclipse 窗口中：[`github.com/15knots/cmake4eclipse#installation`](https://github.com/15knots/cmake4eclipse#installation)。它会要求您重新启动 Eclipse，之后您就可以修改 CMake 项目以在 Eclipse 中使用了。

**安装 GoogleTest**

我们将在系统中安装`GoogleTest`，这也将安装其他依赖于它的软件包。写入以下命令：

```cpp
sudo apt install libgtest-dev google-mock
```

这个命令安装了`GoogleTest`的包含文件和源文件。现在，我们需要构建已安装的源文件以创建`GoogleTest`库。运行以下命令来完成这个步骤：

```cpp
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib
```

### 安装代码包

将该课程的代码包复制到`C:/Code`文件夹中。

### 附加资源

本书的代码包也托管在 GitHub 上，网址为[`github.com/TrainingByPackt/Advanced-CPlusPlus`](https://github.com/TrainingByPackt/Advanced-CPlusPlus)。

我们还有其他代码包来自我们丰富的图书和视频目录，可以在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。去看看吧！
