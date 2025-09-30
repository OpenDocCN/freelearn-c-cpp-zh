# 前言

*英特尔爱迪生项目*旨在帮助初学者掌握英特尔爱迪生并探索其功能。英特尔爱迪生是一个嵌入式计算平台，它允许我们探索物联网、嵌入式系统和机器人的领域。

本书将带你了解各种概念，每一章都有一个你可以执行的项目。它涵盖了多个主题，包括传感器数据采集并将其推送到云端以通过互联网控制设备，以及从图像处理到自主和手动机器人的主题。

在每一章中，本书首先介绍该主题的一些理论方面，包括一些小段代码和最小硬件设置。本章的其余部分致力于项目的实践方面。

本书讨论的项目尽可能只需要最少的硬件，并且每个章节的项目都包括在内，以确保你理解基础知识。

# 本书涵盖的内容

第一章，*设置英特尔爱迪生*，涵盖了设置英特尔爱迪生的初始步骤，包括刷写和设置开发环境。

第二章，*气象站（物联网）*，介绍了物联网，并使用一个简单的气象站案例，其中我们使用温度、烟雾水平和声音水平并将数据推送到云端进行可视化。

第三章，*英特尔爱迪生和物联网（家庭自动化）*，涵盖了一个家庭自动化的案例，其中我们使用英特尔爱迪生控制电气负载。

第四章，*英特尔爱迪生和安全系统*，涵盖了英特尔爱迪生的语音和图像处理。

第五章，*使用英特尔爱迪生的自主机器人*，探讨了机器人领域，其中我们使用英特尔爱迪生和相关算法开发了一条形机器人。

第六章，*使用英特尔爱迪生的手动机器人*，探讨了无人地面车辆（UGV），并指导你通过开发控制器软件的过程。

# 你需要为本书准备什么

本书的强制性先决条件是配备 Windows/Linux/Mac OS 的英特尔爱迪生。软件要求如下：

+   Arduino IDE

+   Visual Studio

+   FileZilla

+   Notepad++

+   PuTTY

+   Intel XDK

# 本书面向的对象

如果你是一名爱好者、机器人工程师、物联网爱好者、程序员或开发者，希望使用英特尔爱迪生创建自主项目，那么这本书适合你。具备先前的编程知识将有所帮助。

# 术语约定

在本书中，你会发现许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称都按照以下方式显示：“我们可以通过使用`include`指令来包含其他上下文。”

代码块按照以下方式设置：

```cpp
int a = analogRead(tempPin ); float R = 1023.0/((float)a)-1.0;
R = 100000.0*R;

float temperature=1.0/(log(R/100000.0)/B+1/298.15)-273.15; Serial.print("temperature = "); Serial.println(temperature);
delay(500);

```

当我们希望将你的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```cpp
string res = textBox.Text; if(string.IsNullOrEmpty(res))
  {
 MessageBox.Show("No text entered. Please enter again");  }
else
  {
    textBlock.Text = res;

```

任何命令行输入或输出都按照以下方式编写：

```cpp
npm install mqtt

```

**新术语**和**重要词汇**以粗体显示。你会在屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“点击确定后，工具将自动解压缩文件。”

警告或重要注意事项以如下方式显示在框中。

小贴士和技巧看起来像这样。

# 读者反馈

我们欢迎读者的反馈。告诉我们你对这本书的看法——你喜欢什么或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出你真正能从中获得最大价值的标题。

要发送一般反馈，只需发送电子邮件到`feedback@packtpub.com`，并在邮件的主题中提及书的标题。

如果你在某个领域有专业知识，并且对撰写或参与一本书籍感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助你从购买中获得最大价值。

# 下载示例代码

你可以从你的账户中下载这本书的示例代码文件[`www.packtpub.com`](http://www.packtpub.com)。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便直接将文件通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

1.  使用你的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的“支持”选项卡上。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书的名称。

1.  选择你想要下载代码文件的书籍。

1.  从下拉菜单中选择你购买这本书的地方。

1.  点击“代码下载”。

文件下载完成后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

这本书的代码包也托管在 GitHub 上[`github.com/PacktPublishing/Intel-Edison-Projects`](https://github.com/PacktPublishing/Intel-Edison-Projects)。我们还有来自我们丰富的图书和视频目录的其他代码包可供选择，在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。查看它们吧！

# 下载这本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些彩色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/IntelEdisonProjects_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/IntelEdisonProjects_ColorImages.pdf)下载此文件。

# 勘误

尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一情况，我们将不胜感激。通过这样做，您可以避免其他读者感到沮丧，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在勘误部分下。

# 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过`copyright@packtpub.com`与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

# 询问

如果您对本书的任何方面有问题，您可以通过`questions@packtpub.com`与我们联系，我们将尽力解决问题。
