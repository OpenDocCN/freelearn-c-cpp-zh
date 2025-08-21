# 前言

现代 C++允许您在高级语言中编写高性能应用程序，而不会牺牲可读性和可维护性。不过，软件架构不仅仅是语言。我们将向您展示如何设计和构建健壮且可扩展且性能良好的应用程序。

本书包括基本概念的逐步解释、实际示例和自我评估问题，您将首先了解架构的重要性，看看一个实际应用程序的案例研究。

您将学习如何在单个应用程序级别使用已建立的设计模式，探索如何使您的应用程序健壮、安全、高性能和可维护。然后，您将构建连接多个单个应用程序的更高级别服务，使用诸如面向服务的架构、微服务、容器和无服务器技术等模式。

通过本书，您将能够使用现代 C++和相关工具构建分布式服务，以提供客户推荐的解决方案。

您是否有兴趣成为软件架构师或者想要了解现代架构的更多趋势？如果是的话，这本书应该会帮助您！

# 本书适合对象

使用现代 C++的开发人员将能够通过本实用指南将他们的知识付诸实践。本书采用了实践方法，涉及实施和相关方法论，将让您迅速上手并提高工作效率。

# 本书涵盖内容

第一章《软件架构的重要性和优秀设计原则》探讨了我们首先为什么设计软件。

第二章《架构风格》涵盖了在架构方面可以采取的不同方法。

第三章《功能和非功能需求》探讨了理解客户需求。

第四章《架构和系统设计》是关于创建有效的软件解决方案。

第五章《利用 C++语言特性》让您能够流利地使用 C++。

第六章《设计模式和 C++》专注于现代 C++习语和有用的代码构造。

第七章《构建和打包》是关于将代码部署到生产环境中。

第八章《可测试代码编写》教会您如何在客户发现之前找到错误。

第九章《持续集成和持续部署》介绍了自动化软件发布的现代方式。

第十章《代码和部署中的安全性》是您将学习如何确保系统难以被破坏的地方。

第十一章《性能》关注性能（当然！）。C++应该快速-它能更快吗？

第十二章《面向服务的架构》让您基于服务构建系统。

第十三章《设计微服务》专注于只做一件事情-设计微服务。

第十四章《容器》为您提供了一个统一的界面来构建、打包和运行应用程序。

第十五章《云原生设计》超越了传统基础设施，探索了云原生设计。

# 充分利用本书

本书中的代码示例大多是为 GCC 10 编写的。它们也应该适用于 Clang 或 Microsoft Visual C++，尽管在较旧版本的编译器中可能缺少 C++20 的某些功能。为了尽可能接近作者的开发环境，我们建议您在类似 Linux 的环境中使用 Nix ([`nixos.org/download.html`](https://nixos.org/download.html))和 direnv ([`direnv.net/`](https://direnv.net/))。如果您在包含示例的目录中运行`direnv allow`，这两个工具应该会为您配置编译器和支持包。

如果没有 Nix 和 direnv，我们无法保证示例将正常工作。如果您使用的是 macOS，Nix 应该可以正常工作。如果您使用的是 Windows，Windows 子系统适用于 Linux 2 是一个很好的方式，可以使用 Nix 创建一个 Linux 开发环境。

要安装这两个工具，您必须运行以下命令：

```cpp
# Install Nix
curl -L https://nixos.org/nix/install | sh
# Configure Nix in the current shell
. $HOME/.nix-profile/etc/profile.d/nix.sh
# Install direnv
nix-env -i direnv
# Download the code examples
git clone https://github.com/PacktPublishing/Hands-On-Software-Architecture-with-Cpp.git
# Change directory to the one with examples
cd Hands-On-Software-Architecture-with-Cpp
# Allow direnv and Nix to manage your development environment
direnv allow
```

执行前面的命令后，Nix 应该下载并安装所有必要的依赖项。这可能需要一些时间，但它有助于确保我们使用的工具完全相同。

**如果您使用本书的数字版本，我们建议您自己输入代码或通过 GitHub 存储库（链接在下一节中提供）访问代码。这样做将有助于避免与复制和粘贴代码相关的任何潜在错误。**

## 下载示例代码文件

您可以从 GitHub 上下载本书的示例代码文件，网址为[`github.com/PacktPublishing/Software-Architecture-with-Cpp`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp)。如果代码有更新，它将在现有的 GitHub 存储库中更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到！快来看看吧！

## 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：`static.packt-cdn.com/downloads/9781838554590_ColorImages.pdf`。

## 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这里有一个例子：“前两个字段（`openapi`和`info`）是描述文档的元数据。”

代码块设置如下：

```cpp
using namespace CppUnit;
using namespace std;
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这里有一个例子：“从管理面板中选择系统信息。”

警告或重要说明会出现在这样的地方。

提示和技巧会出现在这样的地方。
