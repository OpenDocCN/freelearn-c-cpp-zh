# 前言

现代游戏动画有点像黑魔法。没有太多资源详细介绍如何构建基于轨道驱动的动画系统，或者高级主题，比如双四元数蒙皮。这本书的目标就是填补这个空白。本书的目标是为动画编程的黑魔法投下一些光，使这个主题对每个人都变得可接近。

本书采用“理论到实现”的方法，您将首先学习每个讨论主题的理论。一旦您理解了理论，就可以实施它以获得实际经验。

本书着重于动画编程的概念和实现细节，而不是所使用的语言或图形 API。通过专注于这些基本概念，您将能够实现一个动画系统，而不受语言或图形 API 的限制。

# 本书适合的读者

本书适用于想要学习如何构建现代动画系统的程序员。跟随本书的唯一要求是对 C++有一定的了解。除此之外，本书涵盖了从如何打开一个新窗口，到创建一个 OpenGL 上下文，渲染一个动画模型，以及高级动画技术的所有内容。

# 本书涵盖的内容

[*第一章*]（B16191_01_Final_JC_ePub.xhtml#_idTextAnchor013）*，创建游戏窗口*，解释了如何创建一个新的 Visual Studio 项目，创建一个 Win32 窗口，设置一个 OpenGL 3.3 渲染上下文，并启用垂直同步。本书的代码示例是针对 OpenGL 3.3 编译的。所有 OpenGL 代码都与最新版本的 OpenGL 和 OpenGL 4.6 兼容。

[*第二章*]（B16191_02_Final_JC_ePub.xhtml#_idTextAnchor026）*，实现向量*，涵盖了游戏动画编程中的向量数学。

[*第三章*]（B16191_03_Final_JC_ePub.xhtml#_idTextAnchor048）*，实现矩阵*，讨论了游戏动画编程中的矩阵数学。

[*第四章*]（B16191_04_Final_JC_ePub.xhtml#_idTextAnchor069）*，实现四元数*，解释了如何在游戏动画编程中使用四元数数学。

[*第五章*]（B16191_05_Final_JC_ePub.xhtml#_idTextAnchor094）*，实现变换*，解释了如何将位置、旋转和缩放组合成一个变换对象。这些变换对象可以按层次排列。

[*第六章*]（B16191_06_Final_JC_ePub.xhtml#_idTextAnchor104）*，构建抽象渲染器*，向您展示如何在 OpenGL 3.3 之上创建一个抽象层。本书的其余部分将使用这个抽象层进行渲染。通过使用抽象层，我们可以专注于动画编程的核心概念，而不是用于实现它的 API。抽象层针对 OpenGL 3.3，但代码也适用于 OpenGL 4.6。

[*第七章*]（B16191_07_Final_JC_ePub.xhtml#_idTextAnchor128）*，了解 glTF 文件格式*，介绍了 glTF 文件格式。glTF 是一种标准的开放文件格式，受大多数 3D 内容创建工具支持。能够加载一个通用格式将让您加载几乎任何创建工具中制作的动画。

[*第八章*]（B16191_08_Final_JC_ePub.xhtml#_idTextAnchor142）*创建曲线、帧和轨道*，介绍了如何插值曲线以及曲线如何用于动画存储在层次结构中的变换。

[*第九章*]（B16191_09_Final_JC_ePub.xhtml#_idTextAnchor155）*，实现动画片段*，解释了如何实现动画片段。动画片段会随时间修改变换层次结构。

[*第十章*]（B16191_10_Final_JC_ePub.xhtml#_idTextAnchor167）*，网格蒙皮*，介绍了如何变形网格，使其与采样动画片段生成的姿势相匹配。

[*第十一章*]（B16191_11_Final_JC_ePub.xhtml#_idTextAnchor185）*，优化动画管道*，向您展示如何优化动画管道的部分，使其更快速和更适合生产。

*第十二章**，动画之间的混合*，解释了如何混合两个动画姿势。这种技术可以用来平滑地切换两个动画，而不会出现任何视觉跳动。

*第十三章**，实现逆运动学*，介绍了如何使用逆运动学使动画与环境互动。例如，您将学习如何使动画角色的脚在不平坦的地形上不穿透地面。

*第十四章**，使用双四元数进行蒙皮*，介绍了游戏动画中的双四元数数学。双四元数可用于避免在动画关节处出现捏合。

*第十五章**，渲染实例化人群*，展示了如何将动画数据编码到纹理中，并将姿势生成移入顶点着色器。您将使用这种技术来使用实例化渲染大型人群。

# 为了充分利用本书

为了充分利用本书，需要一些 C++的经验。您不必是一个经验丰富的 C++大师，但您应该能够调试简单的 C++问题。有一些 OpenGL 经验是一个加分项，但不是必需的。没有使用高级 C++特性。提供的代码针对 C++ 11 或最新版本进行编译。

本书中的代码是针对 OpenGL 3.3 Core 编写的。本书中呈现的 OpenGL 代码是向前兼容的；在出版时，OpenGL 的最高兼容版本是 4.6。在*第六章*，构建抽象渲染器，您将在 OpenGL 之上实现一个薄的抽象层。在本书的其余部分，您将针对这个抽象层进行编码，而不是直接针对 OpenGL。

本书中呈现的代码应该可以在运行 Windows 10 或更高版本的任何笔记本电脑上编译和运行。跟随本书的唯一硬件要求是能够运行 Visual Studio 2019 或更高版本的计算机。

Visual Studio 2019 的最低硬件要求是：

+   Windows 10，版本 1703 或更高版本

+   1.8 GHz 或更快的处理器

+   2GB 的 RAM

这些要求可以在以下网址找到：[`docs.microsoft.com/en-us/visualstudio/releases/2019/system-requirements`](https://docs.microsoft.com/en-us/visualstudio/releases/2019/system-requirements)

下载示例代码文件

您可以从[`www.packt.com`](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packt.com/support`](http://www.packt.com/support)并注册，文件将直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[`www.packt.com`](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的以下软件解压或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip / UnRarX

+   Linux 上的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Game-Animation-Programming`](https://github.com/PacktPublishing/Game-Animation-Programming)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快去看看吧！

## 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

代码块设置如下：

```cpp
public:
    Pose();
    Pose(const Pose& p);
    Pose& operator=(const Pose& p);
    Pose(unsigned int numJoints);
```

任何命令行输入或输出都会以以下方式书写：

```cpp
# cp /usr/src/asterisk-addons/configs/cdr_mysql.conf.sample
     /etc/asterisk/cdr_mysql.conf
```

**粗体**：表示一个新术语、一个重要词或者屏幕上看到的词，例如在菜单或对话框中，也会在文本中显示为这样。例如：“从管理面板中选择**系统信息**。”

注意

警告或重要说明会显示在这样。

提示和技巧会显示在这样。