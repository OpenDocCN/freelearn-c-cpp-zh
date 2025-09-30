# 前言

本书向读者介绍了 WebAssembly，这是一种新的、令人兴奋的技术，能够在浏览器中执行除 JavaScript 之外的语言。本书描述了如何从头开始构建使用 WebAssembly 的 C/JavaScript 应用，以及如何使用 Emscripten 将现有的 C++代码库移植到浏览器中运行的过程。

WebAssembly 代表了 Web 平台的一个重要转变。作为 C、C++和 Rust 等语言的编译目标，它提供了构建一类新应用的能力。WebAssembly 得到了所有主要浏览器厂商的支持，代表了协作努力的结果。

在这本书中，我们将描述构成 WebAssembly 的元素，以及它的起源。我们将逐步介绍安装所需工具、设置开发环境以及与 WebAssembly 交互的过程。我们将通过简单的示例，逐步深入到更高级的应用场景。到这本书的结尾，你将能够充分准备在你的 C、C++或 JavaScript 项目中使用 WebAssembly。

# 本书面向的对象

如果你是一名希望为 Web 构建应用的 C/C++程序员，或者是一名希望提高其 JavaScript 应用性能的 Web 开发者，那么这本书就是为你准备的。这本书旨在为熟悉 JavaScript 的开发者提供，他们不会介意学习一些 C 和 C++（反之亦然）。这本书通过提供两个示例应用，为 C/C++程序员和 JavaScript 程序员都提供了便利。

# 本书涵盖的内容

第一章，*什么是 WebAssembly？*，描述了 WebAssembly 的起源，并提供了该技术的高级概述。它涵盖了 WebAssembly 的用途、支持的编程语言以及其当前的限制。

第二章，*WebAssembly 的元素 – Wat、Wasm 和 JavaScript API*，概述了构成 WebAssembly 的元素。它详细解释了文本和二进制格式，以及相应的 JavaScript 和 Web API。

第三章，*设置开发环境*，介绍了用于开发 WebAssembly 的工具。它为每个平台提供了安装说明，并提供了改进开发体验的建议。

第四章，*安装所需依赖项*，提供了为每个平台安装工具链要求的说明。到本章结束时，你将能够将 C 和 C++编译成 WebAssembly 模块。

第五章，*创建和加载 WebAssembly 模块*，解释了如何使用 Emscripten 生成 WebAssembly 模块以及编译器接收到的标志如何影响输出结果。它描述了在浏览器中加载 WebAssembly 模块的技术。

第六章，*与 JavaScript 交互和调试*，详细说明了 Emscripten 的 Module 对象与浏览器全局 WebAssembly 对象之间的区别。本章描述了 Emscripten 提供的功能以及生成源图的说明。

第七章，*从头创建应用程序*，介绍了创建一个与 WebAssembly 模块交互的 JavaScript 会计应用程序的过程。我们将编写 C 代码来计算会计交易的价值，并在 JavaScript 和编译的 WebAssembly 模块之间传递数据。

第八章，*使用 Emscripten 移植游戏*，以逐步方法将现有的 C++游戏移植到 WebAssembly。在审查现有的 C++代码库后，对适当的文件进行修改，以便游戏能在浏览器中运行。

第九章，*与 Node.js 集成*，展示了如何在使用 WebAssembly 的服务器和客户端中使用 Node.js 和 npm。本章涵盖了在 Express 应用程序中使用 WebAssembly、将 WebAssembly 与 webpack 集成以及使用 Jest 测试 WebAssembly 模块。

第十章，*高级工具和即将推出的功能*，涵盖了正在标准化过程中的高级工具、用例和新 WebAssembly 功能。本章描述了 WABT、Binaryen 以及在线可用的工具。在本章中，你将学习如何使用 LLVM 编译 WebAssembly 模块以及如何使用 Web Workers 与 WebAssembly 模块交互。本章以标准化过程的描述和对即将添加到规范中的某些令人兴奋的功能的回顾结束。

# 要充分利用本书

你应该有一些编程经验，并理解变量和函数等概念。如果你从未见过 JavaScript 或 C/C++代码，你可能需要在阅读本书中的示例之前做一些初步研究。我选择使用 JavaScript ES6/7 功能，如解构和箭头函数，所以如果你在过去 3-4 年内没有使用过 JavaScript，语法可能看起来略有不同。

# 下载示例代码文件

你可以从 [www.packtpub.com](http://www.packtpub.com) 的账户下载本书的示例代码文件。如果你在其他地方购买了本书，你可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

文件下载完成后，请确保您使用最新版本的以下软件解压或提取文件夹：

+   WinRAR/7-Zip（适用于 Windows）

+   Zipeg/iZip/UnRarX（适用于 Mac）

+   7-Zip/PeaZip（适用于 Linux）

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learn-WebAssembly`](https://github.com/PacktPublishing/Learn-WebAssembly)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还提供其他代码包，这些代码包来自我们丰富的书籍和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788997379_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788997379_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“`instantiate()`是编译和实例化 WebAssembly 代码的主要 API。”

代码块按以下方式设置：

```cpp
int addTwo(int num) {
 return num + 2;
}
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
int calculate(int firstVal, int secondVal) {
return firstVal - secondVal;
}
```

任何命令行输入或输出都按以下方式编写：

```cpp
npm install -g webassembly
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“您可以通过按下“开始”菜单按钮，然后右键单击“命令提示符”应用程序并选择“以管理员身份运行”来完成此操作。”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至`feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`发送电子邮件给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
