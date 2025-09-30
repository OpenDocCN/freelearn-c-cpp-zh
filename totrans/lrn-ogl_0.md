# 前言

OpenGL 是世界上最受欢迎的图形库；大多数移动游戏都使用 OpenGL，许多其他应用程序也是如此。在这本书中，你将了解构成我们玩的游戏和它们背后的游戏引擎的基础知识。通过逐步过程展示从设置 OpenGL 到其基本现代功能的一切。你将深入了解以下概念：使用 GLFW、SDL 和 SFML 在 Windows 和 Mac 上设置，2D 绘图，3D 绘图，纹理，光照，3D 渲染，着色器/GLSL，模型加载，和立方体贴图。

# 本书面向的对象

《学习 OpenGL》适合任何对创建游戏、了解游戏引擎工作原理感兴趣的人，最重要的是，对于任何对学习 OpenGL 感兴趣的人。这本书的理想读者是那些对学习游戏开发充满热情或寻找 OpenGL 参考指南的人。你在本书中学到的技能将适用于你所有的游戏开发需求。你需要有扎实的 C++基础来理解和应用本书中的概念。

# 本书涵盖的内容

第一章，*设置 OpenGL*，在这一章中，你将学习如何使用各种库设置 OpenGL：GLFW、GLEW、SDL 和 SFML。我们将学习如何在 Windows 和 Mac 上设置我们的 OpenGL 项目。我们还讨论了如何使用绝对或相对链接将库链接到你的项目中，并最终创建渲染窗口来显示 OpenGL 图形。

第二章，*绘制形状和应用纹理*，将引导你通过着色器绘制各种形状。我们将从绘制一个三角形开始，并学习如何为其添加颜色。然后，我们将使用三角形的概念来绘制我们的四边形，并学习如何为其添加纹理。

第三章，*变换、投影和摄像机*，这一章在上一章的基础上进一步展开。你将学会如何将旋转和变换等变换应用到我们的形状上，并学习如何绘制一个立方体并为其添加纹理。然后，我们将探讨投影（透视和正交）的概念，并在我们的游戏世界中实现这些概念。

第四章，*光照、材质和光照贴图的效果*，在这一章中，我们将学习如何为我们的对象添加颜色，以及如何在游戏世界中创建光源，例如灯。然后，我们将研究光照对对象的影响。你将了解不同的光照技术：环境光、漫反射、镜面反射。我们还将探索各种真实世界的材质，并观察光照对材质的影响。你还将在本章中学习关于光照贴图的内容。

第五章, *光源类型和灯光组合*, 本章将讨论不同类型的光源，如方向光、点光源和聚光灯。我们还将学习如何组合游戏世界中的灯光效果和光源。

第六章, *使用立方体贴图实现天空盒*, 在本章中，您将使用立方体贴图生成天空盒。您将学习如何将纹理应用到天空盒上，并创建一个单独的纹理文件，以便在代码中更容易地加载纹理。您还将学习如何绘制天空盒，并使用它来创建我们的游戏世界。

[在线章节](https://www.packtpub.com/sites/default/files/downloads/ModelLoading.pdf), *模型加载*, 这是可在线获取的额外章节，地址为[`www.packtpub.com/sites/default/files/downloads/ModelLoading.pdf`](https://www.packtpub.com/sites/default/files/downloads/ModelLoading.pdf)。在本章中，您将学习如何使用 CMake 在 Windows 上设置 Assimp（Open Asset Import Library），以满足我们所有模型加载的需求。我们还将介绍如何在 Mac OS X 上设置 Assimp，并创建一个跨平台网格类。然后我们将探讨如何将 3D 模型加载到我们的游戏中。您还将学习如何创建一个模型类来处理我们模型的加载。

# 要充分利用本书

对于本书，您对 C++有良好的基础非常重要，因为您将在此书中使用 OpenGL 与 C++结合。OpenGL 并不容易，如果您是第一次编码或编码时间不长，建议您先掌握 C++，然后再继续阅读本书。

# 免责声明

本书使用的插图仅用于说明目的。我们不推荐您以任何方式滥用这些插图。有关更多信息，请参阅此处提到的出版商的条款和条件。

任天堂 : [`www.nintendo.com/terms-of-use`](https://www.nintendo.com/terms-of-use)

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择支持选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保您使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为**[`github.com/PacktPublishing/Learn-OpenGL`](https://github.com/PacktPublishing/Learn-OpenGL)**。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。去看看吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/LearnOpenGL_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/LearnOpenGL_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“在`外部库`文件夹中提取 GLEW 和 GLFW 的库文件。”

代码块设置如下：

```cpp
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3); 
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3); 
   SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8); 
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3); 
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3); 
   SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8); 
```

任何命令行输入或输出都按以下方式编写：

```cpp
brew install glfw3 
brew install glew
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“打开 Xcode 并点击创建新 Xcode 项目选项。”

警告或重要注意事项如下所示。

技巧和窍门如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`给我们发邮件。

**勘误表**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，点击勘误表提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上遇到我们作品的任何形式的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为本书做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，并且我们的作者可以查看他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
