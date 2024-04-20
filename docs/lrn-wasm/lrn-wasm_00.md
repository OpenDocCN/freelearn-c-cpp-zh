# 前言

本书介绍了 WebAssembly，这是一项新颖而令人兴奋的技术，能够在浏览器中执行除 JavaScript 以外的其他语言。本书描述了如何从头开始构建一个 C/JavaScript 应用程序，使用 WebAssembly，并将现有的 C++代码库移植到浏览器中运行的过程，借助 Emscripten 的帮助。

WebAssembly 代表了 Web 平台的重要转变。作为诸如 C、C++和 Rust 等语言的编译目标，它提供了构建新型应用程序的能力。WebAssembly 得到了所有主要浏览器供应商的支持，并代表了一项协作努力。

在本书中，我们将描述构成 WebAssembly 的元素及其起源。我们将介绍安装所需工具、设置开发环境以及与 WebAssembly 交互的过程。我们将通过简单示例并逐渐深入的用例来工作。通过本书结束时，您将能够在 C、C++或 JavaScript 项目中充分利用 WebAssembly。

# 本书适合对象

如果您是希望为 Web 构建应用程序的 C/C++程序员，或者是希望改进其 JavaScript 应用程序性能的 Web 开发人员，那么本书适合您。本书面向熟悉 JavaScript 的开发人员，他们不介意学习一些 C 和 C++（反之亦然）。本书通过提供两个示例应用程序，同时考虑到了 C/C++程序员和 JavaScript 程序员的需求。

# 本书涵盖内容

第一章，*什么是 WebAssembly？*，描述了 WebAssembly 的起源，并提供了对该技术的高级概述。它涵盖了 WebAssembly 的用途，支持哪些编程语言以及当前的限制。

第二章，*WebAssembly 的元素- Wat、Wasm 和 JavaScript API*，概述了构成 WebAssembly 的元素。它详细解释了文本和二进制格式，以及相应的 JavaScript 和 Web API。

第三章，*设置开发环境*，介绍了用于开发 WebAssembly 的工具。它提供了每个平台的安装说明，并提供了改进开发体验的建议。

第四章，*安装所需的依赖项*，提供了每个平台安装工具链要求的说明。通过本章结束时，您将能够将 C 和 C++编译为 WebAssembly 模块。

第五章，*创建和加载 WebAssembly 模块*，解释了如何使用 Emscripten 生成 WebAssembly 模块，以及传递给编译器的标志如何影响生成的输出。它描述了在浏览器中加载 WebAssembly 模块的技术。

第六章，*与 JavaScript 交互和调试*，详细介绍了 Emscripten 的 Module 对象和浏览器的全局 WebAssembly 对象之间的区别。本章描述了 Emscripten 提供的功能，以及生成源映射的说明。

第七章，*从头开始创建应用程序*，介绍了创建一个与 WebAssembly 模块交互的 JavaScript 会计应用程序的过程。我们将编写 C 代码来计算会计交易的值，并在 JavaScript 和编译后的 WebAssembly 模块之间传递数据。

第八章，*使用 Emscripten 移植游戏*，采用逐步方法将现有的 C++游戏移植到 WebAssembly 上，使用 Emscripten。在审查现有的 C++代码库之后，对适当的文件进行更改，以使游戏能够在浏览器中运行。

第九章，*与 Node.js 集成*，演示了如何在服务器端和客户端使用 Node.js 和 npm 与 WebAssembly。本章涵盖了在 Express 应用程序中使用 WebAssembly，将 WebAssembly 与 webpack 集成以及使用 Jest 测试 WebAssembly 模块。

第十章，*高级工具和即将推出的功能*，涵盖了正在标准化过程中的高级工具，用例和新的 WebAssembly 功能。本章描述了 WABT，Binaryen 和在线可用的工具。在本章中，您将学习如何使用 LLVM 编译 WebAssembly 模块，以及如何将 WebAssembly 模块与 Web Workers 一起使用。本章最后描述了标准化过程，并审查了一些正在添加到规范中的令人兴奋的功能。

# 充分利用本书

您应该具有一些编程经验，并了解变量和函数等概念。如果您从未见过 JavaScript 或 C/C++代码，您可能需要在阅读本书的示例之前进行一些初步研究。我选择使用 JavaScript ES6/7 功能，如解构和箭头函数，因此如果您在过去 3-4 年内没有使用 JavaScript，语法可能会有些不同。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learn-WebAssembly`](https://github.com/PacktPublishing/Learn-WebAssembly)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包来自我们丰富的书籍和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/9781788997379_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788997379_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如："`instantiate()` 是编译和实例化 WebAssembly 代码的主要 API。"

代码块设置如下：

```cpp
int addTwo(int num) {
 return num + 2;
}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
int calculate(int firstVal, int secondVal) {
return firstVal - secondVal;
}
```

任何命令行输入或输出都将按照以下格式编写：

```cpp
npm install -g webassembly
```

**粗体**：表示新术语，重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中出现。例如：“您可以通过按下“开始”菜单按钮，右键单击“命令提示符”应用程序并选择“以管理员身份运行”来执行此操作。”

警告或重要说明会出现在这样的地方。

提示和技巧会出现在这样的地方。