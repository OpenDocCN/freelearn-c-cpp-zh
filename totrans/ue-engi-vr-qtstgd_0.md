# 前言

对于我们许多开发者来说，**虚拟现实**（**VR**）代表着一个相对未被充分挖掘的独特游戏市场，这些游戏能够利用令人惊叹的新技术。VR 有能力将我们的玩家直接带入我们的数字世界，并为他们提供在其他地方无法获得的经验。然而，采用这项新技术并创建这些世界的技能尚未得到广泛传播且不易获得。我们的目标是改变这一现状，并帮助传播关于 VR 力量的信息。

Epic Games 一直是 VR 的长期支持者。在过去的几个版本中，Unreal Engine 4 扩展了对 VR 的支持，并继续优化其软件，以便更多开发者能够完成令人惊叹的工作。在硬件方面，市场上的制造商数量和 VR 头盔的功能都在不断增加。许多开发者正在为 Oculus Rift 和 HTC Vive 开发应用程序，尽管还有其他选择可供选择，包括 PlayStation VR、Samsung Gear VR 和 Windows 混合现实头盔。

无论你选择哪个，本书都能帮助你踏上与 VR 一起工作的旅程。在本书的整个过程中，我们将探讨如何为 VR 设计。我们将为这个独特环境编写灵活的交互系统，创建用户界面元素，并讨论该媒体的具体游戏艺术需求。最后，我们将完成一个游戏原型并为其准备分发。

# 本书面向的对象

本书是为对 Unreal Engine 4 有兴趣的中级到高级用户编写的，他们希望使用 VR 技术工作。这些用户熟悉游戏引擎，但尚未探索如何在 VR 中创建游戏和应用。

# 为了充分利用本书

+   阅读本书需要具备对 Unreal 游戏引擎的中级知识

+   需要安装 Unreal Engine 4.20.x 版本

+   需要一个虚拟现实头盔以及能够运行它的计算机硬件

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误表”。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

下载文件后，请确保您使用最新版本的软件解压缩或提取文件夹：

+   Windows 系统的 WinRAR/7-Zip

+   Mac 系统的 Zipeg/iZip/UnRarX

+   Linux 系统的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Unreal-Engine-Virtual-Reality-Quick-Start-Guide`](https://github.com/PacktPublishing/Unreal-Engine-Virtual-Reality-Quick-Start-Guide)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包可供在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**. 查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789617405_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789617405_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“点击创建项目按钮，让我们继续！现在看看界面。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过 `customercare@packtpub.com` 发送邮件给我们。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，我们将不胜感激，如果您能向我们提供位置地址或网站名称。请通过 `copyright@packt.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
