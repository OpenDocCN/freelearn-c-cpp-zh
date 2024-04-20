# 前言

WebAssembly 是一项将在未来几年改变网络的技术。WebAssembly 承诺了一个世界，网络应用程序以接近本机速度运行。这是一个你可以用任何喜欢的语言为网络编写应用程序，并将其编译为本机平台以及网络的世界。对于 WebAssembly 来说，现在还处于早期阶段，但这项技术已经像火箭一样起飞。如果你对网络的未来和现在一样感兴趣，那就继续阅读吧！

我写这本书是为了反映我喜欢学习新技能的方式。我将带领你通过使用 WebAssembly 及其所有相关技术开发游戏。我是一名长期从事游戏和网络开发的人，我一直喜欢通过编写游戏来学习新的编程语言。在这本书中，我们将使用与 WebAssembly 紧密相关的网络和游戏开发工具涵盖许多主题。我们将学习如何使用各种编程语言和工具编写针对 WebAssembly 的游戏，包括 Emscripten、C/C++、WebGL、OpenGL、JavaScript、HTML5 和 CSS。作为一家专门从事网络游戏开发的独立游戏开发工作室的老板，我发现了解网络和游戏技术是至关重要的，我在这本书中充满了这些技术。你将学习一系列技能，重点是如何使用 WebAssembly 快速启动应用程序。如果你想学习如何使用 WebAssembly 开发游戏，或者想创建运行速度极快的基于网络的应用程序，这本书适合你。

# 这本书是为谁写的

这本书不是编程入门。它适用于至少掌握一种编程语言的人。了解一些网络技术，如 HTML，会有所帮助，但并非绝对必要。这本书包含了如何在 Windows 或 Ubuntu Linux 上安装所需工具的说明，如果两者中选择一个，我建议使用 Ubuntu，因为它的安装过程要简单得多。

# 这本书涵盖了什么

第一章，*WebAssembly 和 Emscripten 简介*，介绍了 WebAssembly，为什么网络需要它，以及为什么它比 JavaScript 快得多。我们将介绍 Emscripten，为什么我们需要它进行 WebAssembly 开发，以及如何安装它。我们还将讨论与 WebAssembly 相关的技术，如 asm.js、LLVM 和 WebAssembly Text。

第二章，*HTML5 和 WebAssembly*，讨论了 WebAssembly 模块如何使用 JavaScript“粘合代码”与 HTML 集成。我们将学习如何创建自己的 Emscripten HTML 外壳文件，以及如何在我们将用 C 编写的 WebAssembly 模块中进行调用和调用。最后，我们将学习如何编译和运行与我们的 WebAssembly 模块交互的 HTML 页面，以及如何使用 Emscripten 构建一个简单的 HTML5 Canvas 应用程序。

第三章，*WebGL 简介*，介绍了 WebGL 及支持它的新画布上下文。我们将学习着色器是什么，以及 WebGL 如何使用它们将几何图形渲染到画布上。我们将学习如何使用 WebGL 和 JavaScript 将精灵绘制到画布上。最后，我们将编写一个应用程序，集成了 WebAssembly、JavaScript 和 WebGL，显示一个精灵并在画布上移动。

第四章，*在 WebAssembly 中使用 SDL 进行精灵动画*，教你关于 SDL 库以及我们如何使用它来简化从 WebAssembly 到 WebGL 的调用。我们将学习如何使用 SDL 在 HTML5 画布上渲染、动画化和移动精灵。

第五章，“键盘输入”，介绍了如何从 JavaScript 中接收键盘输入并调用 WebAssembly 模块。我们还将学习如何在 WebAssembly 模块内使用 SDL 接受键盘输入，并使用输入来移动 HTML5 画布上的精灵。

第六章，“游戏对象和游戏循环”，探讨了一些基本的游戏设计。我们将学习游戏循环，以及 WebAssembly 中的游戏循环与其他游戏的不同之处。我们还将学习游戏对象以及如何在游戏内部创建对象池。我们将通过编写游戏的开头来结束本章，其中有两艘太空船在画布上移动并互相射击。

第七章，“碰撞检测”，将碰撞检测引入我们的游戏中。我们将探讨 2D 碰撞检测的类型，实现基本的碰撞检测系统，并学习一些使其工作的三角学知识。我们将修改我们的游戏，使得当抛射物相撞时太空船被摧毁。

第八章，“基本粒子系统”，介绍了粒子系统，并讨论了它们如何可以在视觉上改善我们的游戏。我们将讨论虚拟文件系统，并学习如何通过网页向其中添加文件。我们将简要介绍 SVG 和矢量图形，以及如何将它们用于数据可视化。我们还将进一步讨论三角学以及我们将如何在粒子系统中使用它。我们将构建一个新的 HTML5 WebAssembly 应用程序，帮助我们配置和测试稍后将添加到我们的游戏中的粒子系统。

第九章，“改进的粒子系统”，着手改进我们的粒子系统配置工具，添加了粒子缩放、旋转、动画和颜色过渡。我们将修改工具以允许粒子系统循环，并添加爆发效果。然后，我们将更新我们的游戏以支持粒子系统，并为我们的引擎排气和爆炸添加粒子系统效果。

第十章，“AI 和转向行为”，介绍了 AI 和游戏 AI 的概念，并讨论了它们之间的区别。我们将讨论有限状态机、自主代理和转向行为的 AI 概念，并在敌方 AI 中实现这些行为，使其能够避开障碍物并与玩家作战。

第十一章，“设计 2D 摄像头”，引入了 2D 摄像头设计的概念。我们将首先向我们的游戏添加一个渲染管理器，并创建一个锁定在玩家太空船上的摄像头，跟随它在扩展的游戏区域周围移动。然后，我们将添加投影焦点和摄像头吸引器的高级 2D 摄像头功能。

第十二章，“音效”，涵盖了在我们的游戏中使用 SDL 音频。我们将讨论从在线获取音效的位置，以及如何将这些声音包含在我们的 WebAssembly 模块中。然后，我们将向我们的游戏添加音效。

第十三章，“游戏物理”，介绍了计算机游戏中的物理概念。我们将在我们的游戏对象之间添加弹性碰撞。我们将在游戏的物理中添加牛顿第三定律，即当太空船发射抛射物时的后坐力。我们将在吸引太空船的星球上添加一个重力场。

第十四章，“UI 和鼠标输入”，讨论在我们的 WebAssembly 模块中添加要管理和呈现的用户界面。我们将收集要求并将其转化为我们游戏中的新屏幕。我们将添加一个新的按钮对象，并学习如何使用 SDL 从我们的 WebAssembly 模块内管理鼠标输入。

第十五章，“着色器和 2D 照明”，深入探讨了如何创建一个混合 OpenGL 和 SDL 的新应用程序。我们将创建一个新的着色器，加载并渲染多个纹理到一个四边形上。我们将学习法线贴图，以及如何使用法线贴图来在 2D 中近似冯氏光照模型，使用 OpenGL 在我们的 WebAssembly 模块中。

第十六章，“调试和优化”，介绍了调试和优化 WebAssembly 模块的基本方法。我们将从 WebAssembly 的调试宏和堆栈跟踪开始。我们将介绍源映射的概念，以及 Web 浏览器如何使用它们来调试 WebAssembly 模块。我们将学习使用优化标志来优化 WebAssembly 代码。我们将讨论使用分析器来优化我们的 WebAssembly 代码。

# 充分利用本书

您必须了解计算机编程的基础知识。

了解 HTML 和 CSS 等网络技术的基础知识将有所帮助。

# 下载示例代码文件

您可以从这里下载本书的代码包：[`github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly`](https://github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly)。

我们还有来自我们丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781838644659_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781838644659_ColorImages.pdf)。

# 使用的约定

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[**https://github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly**](https://github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自我们丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码字，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如：“我们将复制`basic_particle_shell.html`文件到一个新的外壳文件，我们将其称为`advanced_particle_shell.html`。”

代码块设置如下：

```cpp
<label class="ccontainer"><span class="label">loop:</span>
<input type="checkbox" id="loop" checked="checked">
<span class="checkmark"></span>
</label>
<br/>
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
<label class="ccontainer"><span class="label">loop:</span>
<input type="checkbox" id="loop" checked="checked">
<span class="checkmark"></span>
</label>
<br/>
```

任何命令行输入或输出都以以下方式编写：

```cpp
emrun --list_browsers
```

**粗体**：表示新术语，重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种形式出现在文本中。例如：“从管理面板中选择系统信息。”

警告或重要提示会以这种形式出现。

提示和技巧会出现在这样的形式中。

# 联系我们

我们的读者的反馈总是受欢迎的。

**一般反馈**：如果您对本书的任何方面有疑问，请在您的消息主题中提及书名，并发送电子邮件至`customercare@packtpub.com`。

勘误：尽管我们已经尽最大努力确保内容的准确性，但错误确实会发生。如果您在本书中发现错误，我们将不胜感激地希望您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，点击勘误提交表格链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，请向我们提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或为一本书做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。在阅读并使用本书后，为什么不在购买书籍的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决定，我们在 Packt 可以了解您对我们产品的看法，我们的作者也可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
