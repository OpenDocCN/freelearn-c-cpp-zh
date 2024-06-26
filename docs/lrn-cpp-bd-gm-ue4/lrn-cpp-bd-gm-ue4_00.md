# 前言

因此，您想要使用 Unreal Engine 4（UE4）编写自己的游戏。您有很多理由这样做：UE4 功能强大——UE4 提供了一些最先进、美丽和逼真的光照和物理效果，这些效果是 AAA 工作室使用的类型。

UE4 是设备无关的：为 UE4 编写的代码将在 Windows 台式机、Mac 台式机、所有主要游戏主机（如果您是官方开发人员）、Android 设备和 iOS 设备上运行（在撰写本书时——将来可能支持更多设备！）。因此，您可以使用 UE4 一次编写游戏的主要部分，然后在不经过任何麻烦的情况下部署到 iOS 和 Android 市场。当然，会有一些小问题：iOS 和 Android 应用内购买和通知将需要单独编程，还可能存在其他差异。

# 本书适合对象

本书适合任何想学习游戏编程的人。我们将逐步创建一个简单的游戏，因此您将对整个过程有一个很好的了解。

本书也适合任何想学习 C++，特别是 C++17 的人。我们将介绍 C++的基础知识以及如何在其中编程，并介绍最新 C++版本中的一些新功能。

最后，本书适合任何想学习 UE4 的人。我们将使用它来创建我们的游戏。我们将主要关注 C++方面，但也会涉及一些基本的蓝图开发。

# 本书涵盖内容

第一章，“使用 C++17 入门”，介绍了如何在 Visual Studio Community 2017 或 Xcode 中创建您的第一个 C++项目。我们将创建我们的第一个简单的 C++程序。

第二章，“变量和内存”，涵盖了不同类型的变量，C++中存储数据的基本方法，以及指针、命名空间和控制台应用程序中的基本输入和输出。

第三章，“If、Else 和 Switch”，涵盖了 C++中的基本逻辑语句，允许您根据变量中的值在代码中做出选择。

第四章，“循环”，介绍了如何运行一段代码一定次数，或者直到条件为真。它还涵盖了逻辑运算符，并且我们将看到 UE4 中的第一个代码示例。

第五章，“函数和宏”，介绍了如何设置可以从代码的其他部分调用的代码部分。我们还将介绍如何传递值或获取返回值，并涉及与变量相关的一些更高级的主题。

第六章，“对象、类和继承”，介绍了 C++中的对象，它们是将数据成员和成员函数绑定在一起形成的代码片段，称为类或结构。我们将学习封装以及如何更轻松、更高效地编程对象，使其保持自己的内部状态。

第七章，“动态内存分配”，讨论了动态内存分配以及如何为对象组在内存中创建空间。本章介绍了 C 和 C++风格的数组和向量。在大多数 UE4 代码中，您将使用 UE4 编辑器内置的集合类。

第八章，“角色和棋子”，介绍了如何创建角色并在屏幕上显示它，使用轴绑定控制角色，并创建并显示可以向 HUD 发布消息的 NPC。

第九章，“模板和常用容器”，介绍了如何在 C++中使用模板，并讨论了在 UE4 和 C++标准模板库中可用的基于模板的数据结构。

第十章，库存系统和拾取物品，我们将为玩家编写和设计一个背包来存放物品。当用户按下*I*键时，我们将显示玩家携带的物品。我们将学习如何为玩家设置多个拾取物品。

第十一章，怪物，介绍了如何添加一个景观。玩家将沿着为他们雕刻出的路径行走，然后他们将遇到一支军队。您将学习如何在屏幕上实例化怪物，让它们追逐玩家并攻击他们。

第十二章，使用高级人工智能构建更智能的怪物，介绍了人工智能的基础知识。我们将学习如何使用 NavMesh、行为树和其他人工智能技术，使你的怪物看起来更聪明。

第十三章，魔法书，介绍了如何在游戏中创建防御法术，以及用于可视化显示法术的粒子系统。

第十四章，使用 UMG 和音频改进 UI 反馈，介绍了如何使用新的 UMG UI 系统向用户显示游戏信息。我们将使用 UMG 更新您的库存窗口，使其更简单、更美观，并提供创建自己 UI 的技巧。还介绍了如何添加基本音频以增强游戏体验。

第十五章，虚拟现实及更多，概述了 UE4 在 VR、AR、过程式编程、附加组件和不同平台上的能力。

# 要充分利用本书

在本书中，我们不假设您具有任何编程背景，因此如果您是完全初学者，也没关系！但是最好了解一些关于计算机的知识，以及一些基本的游戏概念。当然，如果您想编写游戏，那么您很可能至少玩过几款游戏！

我们将运行 Unreal Engine 4 和 Visual Studio 2017（或者如果您使用 Mac，则是 Xcode），因此您可能希望确保您的计算机是最新的、性能较强的计算机（如果您想进行 VR，则确保您的计算机已准备好 VR）。

另外，请做好准备！UE4 使用 C++，您可以很快学会基础知识（我们将在这里学到），但要真正掌握这门语言可能需要很长时间。如果您正在寻找一个快速简单的方式来创建游戏，还有其他工具可供选择，但如果您真的想学习能够带来编程游戏职业技能，这是一个很好的起点！

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)注册并直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“SUPPORT”选项卡。

1.  单击“Code Downloads & Errata”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压软件解压文件夹：

+   Windows 系统使用 WinRAR/7-Zip

+   Mac 系统使用 Zipeg/iZip/UnRarX

+   Linux 系统使用 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learning-Cpp-by-Building-Games-with-Unreal-Engine-4-Second-Edition`](https://github.com/PacktPublishing/Learning-Cpp-by-Building-Games-with-Unreal-Engine-4-Second-Edition)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788476249_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788476249_ColorImages.pdf)。

# 本书使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子："我们看到的第一件事是一个`#include`语句。我们要求 C++复制并粘贴另一个 C++源文件的内容，名为`<iostream>`。"

代码块设置如下：

```cpp
#include <iostream>
using namespace std;  
int main() 
{ 
  cout << "Hello, world" << endl; 
  cout << "I am now a C++ programmer." << endl; 
  return 0;
} 
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目将以粗体显示：

```cpp
string name; 
int goldPieces; 
float hp; 
```

**粗体**：表示一个新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中出现。这是一个例子："打开 Epic Games Launcher 应用程序。选择启动 Unreal Engine 4.20.X。"

警告或重要说明看起来像这样。

提示和技巧看起来像这样。