# 第一章：安装 LLVM

要了解如何使用 LLVM，最好从源代码编译 LLVM 开始。LLVM 是一个综合项目，其 GitHub 存储库包含属于 LLVM 的所有项目的源代码。每个 LLVM 项目都在存储库的顶级目录中。除了克隆存储库外，您的系统还必须安装构建系统所需的所有工具。

在本章中，您将了解以下主题：

+   准备先决条件，将向您展示如何设置构建系统。

+   使用 CMake 构建，将介绍如何使用 CMake 和 Ninja 编译和安装 LLVM 核心库和 Clang。

+   定制构建过程，将讨论我们可以影响构建过程的各种方式。

# 准备先决条件

要使用 LLVM，您的开发系统必须运行常见的操作系统，如 Linux，FreeBSD，macOS 或 Windows。启用调试符号构建 LLVM 和 Clang 很容易需要数十 GB 的磁盘空间，因此请确保您的系统有足够的磁盘空间可用-在这种情况下，您应该有 30GB 的可用空间。

所需的磁盘空间严重依赖于所选择的构建选项。例如，仅在发布模式下构建 LLVM 核心库，同时仅针对一个平台，大约需要 2GB 的可用磁盘空间，这是所需的最低限度。为了减少编译时间，快速的 CPU（例如 2.5GHz 时钟速度的四核 CPU）和快速的 SSD 也会有所帮助。

甚至可以在树莓派等小型设备上构建 LLVM-只是需要花费很长时间。我在一台配有 Intel 四核 CPU，时钟速度为 2.7GHz，40GB RAM 和 2.5TB SSD 磁盘空间的笔记本电脑上开发了本书中的示例。这个系统非常适合手头的开发任务。

您的开发系统必须安装一些先决条件软件。让我们回顾一下这些软件包的最低要求版本。

注意

Linux 发行版通常包含可以使用的更新版本。版本号适用于 LLVM 12。LLVM 的较新版本可能需要这里提到的软件包的更新版本。

要从**GitHub**检出源代码，您需要**git** ([`git-scm.com/`](https://git-scm.com/))。没有特定版本的要求。GitHub 帮助页面建议至少使用版本 1.17.10。

LLVM 项目使用**CMake** ([`cmake.org/`](https://cmake.org/)) 作为构建文件生成器。至少需要版本 3.13.4。CMake 可以为各种构建系统生成构建文件。在本书中，使用**Ninja** ([`ninja-build.org/`](https://ninja-build.org/))，因为它快速且在所有平台上都可用。建议使用最新版本 1.9.0。

显然，您还需要一个**C/C++编译器**。LLVM 项目是用现代 C++编写的，基于 C++14 标准。需要符合的编译器和标准库。已知以下编译器与 LLVM 12 兼容：

+   gcc 5.1.0 或更高版本

+   Clang 3.5 或更高版本

+   Apple Clang 6.0 或更高版本

+   Visual Studio 2017 或更高版本

请注意，随着 LLVM 项目的进一步发展，编译器的要求很可能会发生变化。在撰写本文时，有讨论要使用 C++17 并放弃对 Visual Studio 2017 的支持。一般来说，您应该使用系统中可用的最新编译器版本。

**Python** ([`python.org/`](https://python.org/)) 用于生成构建文件和运行测试套件。它应至少是 3.6 版本。

尽管本书未涉及，但您可能有理由需要使用 Make 而不是 Ninja。在这种情况下，您需要在每个命令中使用`make`和本书中描述的场景。

要安装先决条件软件，最简单的方法是使用操作系统的软件包管理器。在接下来的部分中，将显示安装最受欢迎操作系统的软件所需输入的命令。

## Ubuntu

Ubuntu 20.04 使用 APT 软件包管理器。大多数基本实用程序已经安装好了；只有开发工具缺失。要一次安装所有软件包，请键入以下内容：

```cpp
$ sudo apt install –y gcc g++ git cmake ninja-build
```

## Fedora 和 RedHat

Fedora 33 和 RedHat Enterprise Linux 8.3 的软件包管理器称为**DNF**。与 Ubuntu 一样，大多数基本实用程序已经安装好了。要一次安装所有软件包，请键入以下内容：

```cpp
$ sudo dnf install –y gcc gcc-c++ git cmake ninja-build
```

## FreeBSD

在 FreeBSD 12 或更高版本上，必须使用 PKG 软件包管理器。FreeBSD 与基于 Linux 的系统不同，它更喜欢使用 Clang 编译器。要一次安装所有软件包，请键入以下内容：

```cpp
$ sudo pkg install –y clang git cmake ninja
```

## OS X

在 OS X 上进行开发时，最好从 Apple 商店安装**Xcode**。虽然本书中没有使用 XCode IDE，但它带有所需的 C/C++编译器和支持工具。要安装其他工具，可以使用 Homebrew 软件包管理器（https://brew.sh/）。要一次安装所有软件包，请键入以下内容：

```cpp
$ brew install git cmake ninja
```

## Windows

与 OS X 一样，Windows 没有软件包管理器。安装所有软件的最简单方法是使用**Chocolately**（[`chocolatey.org/`](https://chocolatey.org/)）软件包管理器。要一次安装所有软件包，请键入以下内容：

```cpp
$ choco install visualstudio2019buildtools cmake ninja git\
  gzip bzip2 gnuwin32-coreutils.install
```

请注意，这只安装了来自`package visualstudio2019community`而不是`visualstudio2019buildtools`的构建工具。Visual Studio 2019 安装的一部分是 x64 Native Tools Command Prompt for VS 2019。使用此命令提示时，编译器会自动添加到搜索路径中。

## 配置 Git

LLVM 项目使用 Git 进行版本控制。如果您以前没有使用过 Git，则应该在继续之前对 Git 进行一些基本配置；也就是说，设置用户名和电子邮件地址。如果您提交更改，这两个信息都会被使用。在以下命令中，将`Jane`替换为您的姓名，`jane@email.org`替换为您的电子邮件：

```cpp
$ git config --global user.email "jane@email.org"
$ git config --global user.name "Jane"
```

默认情况下，Git 使用**vi**编辑器进行提交消息。如果您希望使用其他编辑器，则可以以类似的方式更改配置。要使用**nano**编辑器，请键入以下内容：

```cpp
$ git config --global core.editor nano
```

有关 git 的更多信息，请参阅 Packt Publishing 的*Git Version Control Cookbook - Second Edition*（[`www.packtpub.com/product/git-version-control-cookbook/9781782168454`](https://www.packtpub.com/product/git-version-control-cookbook/9781782168454)）。

# 使用 CMake 构建

准备好构建工具后，您现在可以从 GitHub 检出所有 LLVM 项目。执行此操作的命令在所有平台上基本相同。但是，在 Windows 上，建议关闭行结束的自动翻译。

让我们分三部分回顾这个过程：克隆存储库，创建构建目录和生成构建系统文件。

## 克隆存储库

在所有非 Windows 平台上，键入以下命令以克隆存储库：

```cpp
$ git clone https://github.com/llvm/llvm-project.git
```

在 Windows 上，您必须添加选项以禁用自动翻译行结束。在这里，键入以下内容：

```cpp
$ git clone --config core.autocrlf=false\  https://github.com/llvm/llvm-project.git
```

这个`git`命令将最新的源代码从 GitHub 克隆到名为`llvm-project`的本地目录中。现在，使用以下命令将当前目录更改为新的`llvm-project`目录：

```cpp
$ cd llvm-project
```

在目录中包含了所有 LLVM 项目，每个项目都在自己的目录中。值得注意的是，LLVM 核心库位于`llvm`子目录中。LLVM 项目使用分支进行后续发布的开发（“release/12.x”）和标记（“llvmorg-12.0.0”）来标记特定的发布。使用前面的`clone`命令，您可以获得当前的开发状态。本书使用 LLVM 12。要检出 LLVM 12 的第一个发布版本，请键入以下内容：

```cpp
$ git checkout -b llvmorg-12.0.0
```

有了这个，你已经克隆了整个存储库并检出了一个标签。这是最灵活的方法。

Git 还允许你只克隆一个分支或一个标签（包括历史记录）。使用`git clone --branch llvmorg-12.0.0 https://github.com/llvm/llvm-project`，你检出了与之前相同的标签，但只克隆了该标签的历史记录。通过额外的`--depth=1`选项，你可以防止克隆历史记录。这样可以节省时间和空间，但显然会限制你在本地可以做什么。

下一步是创建一个构建目录。

## 创建一个构建目录

与许多其他项目不同，LLVM 不支持内联构建，需要一个单独的`build`目录。这可以很容易地在`llvm-project`目录内创建。使用以下命令切换到此目录：

```cpp
$ cd llvm-project
```

然后，为简单起见，创建一个名为`build`的构建目录。在这里，Unix 和 Windows 系统的命令不同。在类 Unix 系统上，你应该使用以下命令：

```cpp
$ mkdir build
```

在 Windows 上，你应该使用以下命令：

```cpp
$ md build
```

然后，切换到`build`目录：

```cpp
$ cd build
```

现在，你已经准备好在这个目录中使用 CMake 工具创建构建系统文件。

## 生成构建系统文件

要生成使用 Ninja 编译 LLVM 和 Clang 的构建系统文件，请运行以下命令：

```cpp
$ cmake –G Ninja -DLLVM_ENABLE_PROJECTS=clang ../llvm
```

提示

在 Windows 上，反斜杠字符`\`是目录名称分隔符。在 Windows 上，CMake 会自动将 Unix 分隔符`/`转换为 Windows 分隔符。

`-G`选项告诉 CMake 为哪个系统生成构建文件。最常用的选项如下：

+   `Ninja`：对于 Ninja 构建系统

+   `Unix Makefiles`：对于 GNU Make

+   `Visual Studio 15 VS2017`和`Visual Studio 16 VS2019`：对于 Visual Studio 和 MS Build

+   `Xcode`：对于 XCode 项目

生成过程可以通过使用`-D`选项设置各种变量来进行影响。通常，它们以`CMAKE_`（如果由 CMake 定义）或`LLVM_`（如果由 LLVM 定义）为前缀。通过设置`LLVM_ENABLE_PROJECTS=clang`变量，CMake 会生成 Clang 的构建文件，除了 LLVM。命令的最后一部分告诉 CMake 在哪里找到 LLVM 核心库源代码。关于这一点，我们将在下一节详细介绍。

一旦构建文件生成，LLVM 和 Clang 可以使用以下命令编译：

```cpp
$ ninja
```

根据硬件资源的不同，这个命令需要花费 15 分钟（具有大量 CPU 核心和内存以及快速存储的服务器）到几个小时（双核 Windows 笔记本，内存有限）不等。默认情况下，Ninja 利用所有可用的 CPU 核心。这对于编译速度很好，但可能会阻止其他任务运行。例如，在基于 Windows 的笔记本上，几乎不可能在 Ninja 运行时上网冲浪。幸运的是，你可以使用`-j`选项限制资源使用。

假设你有四个 CPU 核心可用，而 Ninja 只应该使用两个（因为你有并行任务要运行）。在这里，你应该使用以下命令进行编译：

```cpp
$ ninja –j2
```

一旦编译完成，最佳实践是运行测试套件，以检查一切是否按预期工作：

```cpp
$ ninja check-all
```

这个命令的运行时间因可用的硬件资源而变化很大。Ninja `check-all`目标运行所有测试用例。为包含测试用例的每个目录生成目标。使用`check-llvm`而不是`check-all`运行 LLVM 测试但不运行 Clang 测试；`check-llvm-codegen`只运行 LLVM 的`CodeGen`目录中的测试（即`llvm/test/CodeGen`目录）。

你也可以进行快速手动检查。你将使用的 LLVM 应用程序之一是`-version`选项，它显示它的 LLVM 版本，它的主机 CPU 以及所有支持的架构：

```cpp
$ bin/llc -version
```

如果您在编译 LLVM 时遇到问题，应该查阅*Getting Started with the LLVM System*文档的*Common Problems*部分（[`llvm.org/docs/GettingStarted.html#common-problems`](https://llvm.org/docs/GettingStarted.html#common-problems)）以解决常见问题。

最后，安装二进制文件：

```cpp
$ ninja install
```

在类 Unix 系统上，安装目录为`/usr/local`。在 Windows 上，使用`C:\Program Files\LLVM`。当然可以更改。下一节将解释如何更改。

# 自定义构建过程

CMake 系统使用`CMakeLists.txt`文件中的项目描述。顶层文件位于`llvm`目录中；即`llvm/CMakeLists.txt`。其他目录也包含`CMakeLists.txt`文件，在构建文件生成期间递归包含。

根据项目描述中提供的信息，CMake 检查已安装的编译器，检测库和符号，并创建构建系统文件，例如`build.ninja`或`Makefile`（取决于选择的生成器）。还可以定义可重用的模块，例如检测 LLVM 是否已安装的函数。这些脚本放置在特殊的`cmake`目录（`llvm/cmake`），在生成过程中会自动搜索。

构建过程可以通过定义 CMake 变量进行自定义。使用`-D`命令行选项设置变量的值。这些变量在 CMake 脚本中使用。CMake 本身定义的变量几乎总是以`CMAKE_`为前缀，并且这些变量可以在所有项目中使用。LLVM 定义的变量以`LLVM_`为前缀，但只能在项目定义中包括 LLVM 使用时使用。

## CMake 定义的变量

一些变量使用环境变量的值进行初始化。最显著的是`CC`和`CXX`，它们定义了用于构建的 C 和 C++编译器。CMake 会尝试自动定位 C 和 C++编译器，使用当前的 shell 搜索路径。它会选择找到的第一个编译器。如果安装了多个编译器，例如 gcc 和 Clang 或不同版本的 Clang，则这可能不是您要用于构建 LLVM 的编译器。

假设您想将`clang9`用作 C 编译器，将`clang++9`用作 C++编译器。在 Unix shell 中，可以按以下方式调用 CMake：

```cpp
$ CC=clang9 CXX=clang++9 cmake ../llvm
```

这将设置`cmake`调用时环境变量的值。如果需要，您可以为编译器可执行文件指定绝对路径。

`CC`是`CMAKE_C_COMPILER` CMake 变量的默认值，而`CXX`是`CMAKE_CXX_COMPILER` CMake 变量的默认值。您可以直接设置 CMake 变量，而不是使用环境变量。这相当于前面的调用：

```cpp
$ cmake –DCMAKE_C_COMPILER=clang9\
  -DCMAKE_CXX_COMPILER=clang++9 ../llvm
```

CMake 定义的其他有用变量如下：

+   `CMAKE_INSTALL_PREFIX`：安装期间添加到每个路径前面的路径前缀。Unix 上默认为`/usr/local`，Windows 上为`C:\Program Files\<Project>`。要在`/opt/llvm`目录中安装 LLVM，必须指定`-DCMAKE_INSTALL_PREFIX=/opt/llvm`。二进制文件将被复制到`/opt/llvm/bin`，库文件将被复制到`/opt/llvm/lib`，依此类推。

+   `CMAKE_BUILD_TYPE`：不同类型的构建需要不同的设置。例如，调试构建需要指定生成调试符号的选项，并且通常链接到系统库的调试版本。相比之下，发布构建使用优化标志，并链接到库的生产版本。此变量仅用于只能处理一种构建类型的构建系统，例如 Ninja 或 Make。对于 IDE 构建系统，会生成所有变体，您必须使用 IDE 的机制在构建类型之间切换。一些可能的值如下：

`DEBUG`：带有调试符号的构建

`RELEASE`：用于速度优化的构建

`RELWITHDEBINFO`：带有调试符号的发布版本

`MINSIZEREL`：针对大小进行优化的构建

默认的构建类型是`DEBUG`。要为发布构建生成构建文件，必须指定`-DCMAKE_BUILD_TYPE=RELEASE`。

+   `CMAKE_C_FLAGS`和`CMAKE_CXX_FLAGS`：这些是在编译 C 和 C++源文件时使用的额外标志。初始值取自`CFLAGS`和`CXXFLAGS`环境变量，可以用作替代。

+   `CMAKE_MODULE_PATH`：指定要在 CMake 模块中搜索的附加目录。指定的目录将在默认目录之前搜索。该值是一个用分号分隔的目录列表。

+   `PYTHON_EXECUTABLE`：如果找不到 Python 解释器，或者如果安装了多个版本并选择了错误的版本，则可以将此变量设置为 Python 二进制文件的路径。只有在包含 CMake 的 Python 模块时，此变量才会生效（这是 LLVM 的情况）。

CMake 为变量提供了内置帮助。`--help-variable var`选项会打印`var`变量的帮助信息。例如，您可以输入以下内容以获取`CMAKE_BUILD_TYPE`的帮助：

```cpp
$ cmake --help-variable CMAKE_BUILD_TYPE
```

您还可以使用以下命令列出所有变量：

```cpp
$ cmake --help-variablelist
```

此列表非常长。您可能希望将输出导入`more`或类似的程序。

## LLVM 定义的变量

LLVM 定义的变量与 CMake 定义的变量的工作方式相同，只是没有内置帮助。最有用的变量如下：

+   `LLVM_TARGETS_TO_BUILD`：LLVM 支持不同 CPU 架构的代码生成。默认情况下，会构建所有这些目标。使用此变量指定要构建的目标列表，用分号分隔。当前的目标有`AArch64`、`AMDGPU`、`ARM`、`BPF`、`Hexagon`、`Lanai`、`Mips`、`MSP430`、`NVPTX`、`PowerPC`、`RISCV`、`Sparc`、`SystemZ`、`WebAssembly`、`X86`和`XCore`。`all`可以用作所有目标的简写。名称区分大小写。要仅启用 PowerPC 和 System Z 目标，必须指定`-DLLVM_TARGETS_TO_BUILD="PowerPC;SystemZ"`。

+   `LLVM_ENABLE_PROJECTS`：这是要构建的项目列表，用分号分隔。项目的源代码必须与`llvm`目录处于同一级别（并排布局）。当前列表包括`clang`、`clang-tools-extra`、`compiler-rt`、`debuginfo-tests`、`lib`、`libclc`、`libcxx`、`libcxxabi`、`libunwind`、`lld`、`lldb`、`llgo`、`mlir`、`openmp`、`parallel-libs`、`polly`和`pstl`。`all`可以用作此列表中所有项目的简写。要与 LLVM 一起构建 Clang 和 llgo，必须指定`-DLLVM_ENABLE_PROJECT="clang;llgo"`。

+   `LLVM_ENABLE_ASSERTIONS`：如果设置为`ON`，则启用断言检查。这些检查有助于发现错误，在开发过程中非常有用。对于`DEBUG`构建，默认值为`ON`，否则为`OFF`。要打开断言检查（例如，对于`RELEASE`构建），必须指定`–DLLVM_ENABLE_ASSERTIONS=ON`。

+   `LLVM_ENABLE_EXPENSIVE_CHECKS`：这将启用一些可能会显著减慢编译速度或消耗大量内存的昂贵检查。默认值为`OFF`。要打开这些检查，必须指定`-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON`。

+   `LLVM_APPEND_VC_REV`：LLVM 工具（如`llc`）显示它们所基于的 LLVM 版本，以及其他信息（如果提供了`--version`命令行选项）。此版本信息基于`LLVM_REVISION` C 宏。默认情况下，版本信息不仅包括 LLVM 版本，还包括最新提交的 Git 哈希。如果您正在跟踪主分支的开发，这很方便，因为它清楚地指出了工具所基于的 Git 提交。如果不需要这个信息，则可以使用`–DLLVM_APPEND_VC_REV=OFF`关闭。

+   `LLVM_ENABLE_THREADS`：如果检测到线程库（通常是 pthread 库），LLVM 会自动包含线程支持。此外，在这种情况下，LLVM 假定编译器支持`-DLLVM_ENABLE_THREADS=OFF`。

+   `LLVM_ENABLE_EH`：LLVM 项目不使用 C++异常处理，因此默认情况下关闭异常支持。此设置可能与您的项目链接的其他库不兼容。如果需要，可以通过指定`–DLLVM_ENABLE_EH=ON`来启用异常支持。

+   `LLVM_ENABLE_RTTI`：LVM 使用了一个轻量级的、自建的运行时类型信息系统。默认情况下，生成 C++ RTTI 是关闭的。与异常处理支持一样，这可能与其他库不兼容。要打开 C++ RTTI 的生成，必须指定`–DLLVM_ENABLE_RTTI=ON`。

+   `LLVM_ENABLE_WARNINGS`：编译 LLVM 应尽可能不生成警告消息。因此，默认情况下打印警告消息的选项是打开的。要关闭它，必须指定`–DLLVM_ENABLE_WARNINGS=OFF`。

+   `LLVM_ENABLE_PEDANTIC`：LLVM 源代码应符合 C/C++语言标准；因此，默认情况下启用源代码的严格检查。如果可能，还会禁用特定于编译器的扩展。要取消此设置，必须指定`–DLLVM_ENABLE_PEDANTIC=OFF`。

+   `LLVM_ENABLE_WERROR`：如果设置为`ON`，则所有警告都被视为错误-一旦发现警告，编译就会中止。它有助于找到源代码中所有剩余的警告。默认情况下，它是关闭的。要打开它，必须指定`–DLLVM_ENABLE_WERROR=ON`。

+   `LLVM_OPTIMIZED_TABLEGEN`：通常，tablegen 工具与 LLVM 的其他部分使用相同的选项构建。同时，tablegen 用于生成代码生成器的大部分代码。因此，在调试构建中，tablegen 的速度要慢得多，从而显著增加了编译时间。如果将此选项设置为`ON`，则即使在调试构建中，tablegen 也将使用优化进行编译，可能会减少编译时间。默认为`OFF`。要打开它，必须指定`–DLLVM_OPTIMIZED_TABLEGEN=ON`。

+   `LLVM_USE_SPLIT_DWARF`：如果构建编译器是 gcc 或 Clang，则打开此选项将指示编译器将 DWARF 调试信息生成到单独的文件中。对象文件的减小尺寸显著减少了调试构建的链接时间。默认为`OFF`。要打开它，必须指定`-LLVM_USE_SPLIT_DWARF=ON`。

LLVM 定义了许多更多的 CMake 变量。您可以在 LLVM CMake 文档中找到完整的列表([`releases.llvm.org/12.0.0/docs/CMake.html#llvm-specific-variables`](https://releases.llvm.org/12.0.0/docs/CMake.html#llvm-specific-variables))。前面的列表只包含您可能需要的变量。

# 总结

在本章中，您准备好了开发机器来编译 LLVM。您克隆了 LLVM GitHub 存储库，并编译了自己的 LLVM 和 Clang 版本。构建过程可以使用 CMake 变量进行自定义。您还了解了有用的变量以及如何更改它们。掌握了这些知识，您可以根据自己的需求调整 LLVM。

在下一章中，我们将更仔细地查看 LLVM 单一存储库的内容。您将了解其中包含哪些项目以及这些项目的结构。然后，您将使用这些信息来使用 LLVM 库创建自己的项目。最后，您将学习如何为不同的 CPU 架构编译 LLVM。
