# 1

# 安装 LLVM

为了学习如何与 LLVM 一起工作，最好从源代码编译 LLVM 开始。LLVM 是一个伞形项目，GitHub 存储库包含 LLVM 所属所有项目的源代码。每个 LLVM 项目都在存储库的顶级目录中。除了克隆存储库外，您的系统还必须安装构建系统所需的全部工具。在本章中，您将学习以下主题：

+   准备先决条件，这将向您展示如何设置您的构建系统

+   克隆存储库并从源代码构建，这将涵盖如何获取 LLVM 源代码，以及如何使用 CMake 和 Ninja 编译和安装 LLVM 核心库和 clang

+   自定义构建过程，这将讨论影响构建过程的多种可能性

# 编译 LLVM 与安装二进制文件

您可以从各种来源安装 LLVM 二进制文件。如果您使用 Linux，那么您的发行版包含 LLVM 库。为什么还要自己编译 LLVM 呢？

首先，并非所有安装包都包含开发 LLVM 所需的所有文件。自己编译和安装 LLVM 可以防止这个问题。另一个原因源于 LLVM 的高度可定制性。通过构建 LLVM，您将学习如何自定义 LLVM，这将使您能够诊断在将您的 LLVM 应用程序带到另一个平台时可能出现的任何问题。最后，在本书的第三部分，您将扩展 LLVM 本身，为此，您需要自己构建 LLVM 的技能。

然而，在第一步避免编译 LLVM 是完全可以接受的。如果您想走这条路，那么您只需要安装下一节中描述的先决条件。

注意

许多 Linux 发行版将 LLVM 分割成几个包。请确保您安装了开发包。例如，在 Ubuntu 上，您需要安装 `llvm-dev` 包。请确保您安装了 LLVM 17。对于其他版本，本书中的示例可能需要更改。

# 准备先决条件

要使用 LLVM，您的开发系统应运行常见的操作系统，例如 Linux、FreeBSD、macOS 或 Windows。您可以在不同的模式下构建 LLVM 和 clang。启用调试符号的构建可能需要高达 30 GB 的空间。所需的磁盘空间很大程度上取决于选择的构建选项。例如，仅以发布模式构建 LLVM 核心库，针对单一平台，需要大约 2 GB 的空闲磁盘空间，这是最低需求。

为了减少编译时间，一个快速的 CPU（例如，2.5 GHz 时钟速度的四核 CPU）和快速的 SSD 也是很有帮助的。甚至可以在像 Raspberry Pi 这样的小型设备上构建 LLVM – 它只需要很多时间。本书中的示例是在一个配备英特尔四核 CPU，运行在 2.7 GHz 时钟速度，40 GB RAM 和 2.5 TB SSD 磁盘空间的笔记本电脑上开发的。这个系统非常适合开发任务。

您的开发系统必须安装一些先决软件。让我们回顾这些软件包的最小所需版本。

要从 GitHub 检出源代码，你需要 **Git** ([`git-scm.com/`](https://git-scm.com/))。没有特定版本的要求。GitHub 帮助页面建议使用至少版本 1.17.10。由于过去发现的安全问题，建议使用最新的可用版本，即写作时的 2.39.1。

LLVM 项目使用 **CMake** ([`cmake.org/`](https://cmake.org/)) 作为构建文件生成器。至少需要 3.20.0 版本。CMake 可以为各种构建系统生成构建文件。本书中使用 **Ninja** ([`ninja-build.org/`](https://ninja-build.org/))，因为它速度快且适用于所有平台。建议使用最新版本，1.11.1。

显然，你还需要一个 **C/C++ 编译器**。LLVM 项目是用现代 C++ 编写的，基于 C++17 标准。需要一个符合标准的编译器和标准库。以下编译器已知与 LLVM 17 兼容：

+   gcc 7.1.0 或更高版本

+   clang 5.0 或更高版本

+   Apple clang 10.0 或更高版本

+   Visual Studio 2019 16.7 或更高版本

小贴士

请注意，随着 LLVM 项目的进一步发展，编译器的需求很可能会发生变化。一般来说，你应该使用适用于您系统的最新编译器版本。

**Python** ([`python.org/`](https://python.org/)) 在生成构建文件和运行测试套件时使用。它至少应该是 3.8 版本。

虽然本书没有涉及，但可能有原因需要使用 Make 而不是 Ninja。在这种情况下，对于以下描述的场景，需要在每个命令中使用 `make` 和 `ninja`。

LLVM 还依赖于 `zlib` 库 ([`www.zlib.net/`](https://www.zlib.net/))。你应该至少安装了 1.2.3.4 版本。像往常一样，我们建议使用最新版本，1.2.13。

要安装先决软件，最简单的方法是使用操作系统的包管理器。在以下章节中，将展示为最流行的操作系统安装软件所需的命令。

## Ubuntu

Ubuntu 22.04 使用 `apt` 包管理器。大多数基本实用工具已经安装；只有开发工具缺失。要一次性安装所有包，请输入以下命令：

```cpp

$ sudo apt -y install gcc g++ git cmake ninja-build zlib1g-dev
```

## Fedora 和 RedHat

Fedora 37 和 RedHat Enterprise Linux 9 的包管理器称为 `dnf`。和 Ubuntu 一样，大多数基本工具已经安装。要一次性安装所有包，你可以输入以下命令：

```cpp

$ sudo dnf –y install gcc gcc-c++ git cmake ninja-build \
  zlib-devel
```

## FreeBSD

在 FreeBSD 13 或更高版本上，你必须使用 `pkg` 包管理器。FreeBSD 与基于 Linux 的系统不同，因为 clang 编译器已经安装。要一次性安装所有其他包，你可以输入以下命令：

```cpp

$ sudo pkg install –y git cmake ninja zlib-ng
```

## OS X

对于 OS X 上的开发，最好从 Apple Store 安装 **Xcode**。虽然本书中没有使用 Xcode IDE，但它包含了所需的 C/C++ 编译器和支持工具。对于其他工具的安装，可以使用包管理器 **Homebrew** ([`brew.sh/`](https://brew.sh/))。要一次性安装所有包，你可以输入以下命令：

```cpp

$ brew install git cmake ninja zlib
```

## Windows

和 OS X 一样，Windows 没有自带包管理器。对于 C/C++ 编译器，你需要下载 **Visual Studio Community 2022** ([`visualstudio.microsoft.com/vs/community/`](https://visualstudio.microsoft.com/vs/community/))，这是个人使用的免费软件。请确保你安装了名为 **Desktop Development with C++** 的工作负载。你可以使用包管理器 **Scoop** ([`scoop.sh/`](https://scoop.sh/)) 来安装其他包。按照网站上的说明安装 Scoop 后，从你的 Windows 菜单中打开 **x64 Native Tools 命令提示符 for VS 2022**。要安装所需的包，你可以输入以下命令：

```cpp

$ scoop install git cmake ninja python gzip bzip2 coreutils
$ scoop bucket add extras
$ scoop install zlib
```

请密切关注 Scoop 的输出。对于 Python 和 `zlib` 包，它会建议添加一些注册表键。这些条目是必需的，以便其他软件可以找到这些包。要添加注册表键，你最好复制并粘贴 Scoop 的输出，如下所示：

```cpp

$ %HOMEPATH%\scoop\apps\python\current\install-pep-514.reg
$ %HOMEPATH%\scoop\apps\zlib\current\register.reg
```

每个命令之后，注册表编辑器会弹出一个消息窗口询问你是否真的想要导入那些注册表键。你需要点击 **是** 来完成导入。现在所有先决条件都已安装。

对于本书中的所有示例，你必须使用 VS 2022 的 **x64 Native Tools 命令提示符**。使用此命令提示符，编译器会自动添加到搜索路径。

小贴士

LLVM 代码库非常大。为了舒适地导航源代码，我们建议使用一个允许你跳转到类定义并搜索源代码的 IDE。我们发现 **Visual Studio Code** ([`code.visualstudio.com/download`](https://code.visualstudio.com/download))，这是一个可扩展的跨平台 IDE，非常易于使用。然而，这并不是遵循本书中示例的必要条件。

# 克隆仓库并从源代码构建。

准备好构建工具后，你现在可以从 GitHub 检出所有 LLVM 项目并构建 LLVM。这个过程在所有平台上基本上是相同的：

1.  配置 Git。

1.  克隆仓库。

1.  创建构建目录。

1.  生成构建系统文件。

1.  最后，构建并安装 LLVM。

让我们从配置 Git 开始。

## 配置 Git

LLVM 项目使用 Git 进行版本控制。如果您之前没有使用过 Git，那么在继续之前，您应该先进行一些基本的 Git 配置：设置用户名和电子邮件地址。这两项信息在提交更改时都会使用。

您可以使用以下命令检查是否已经在 Git 中配置了之前的电子邮件和用户名：

```cpp

$ git config user.email
$ git config user.name
```

前面的命令将输出您在使用 Git 时已经设置的相应电子邮件和用户名。然而，如果您是第一次设置用户名和电子邮件，可以输入以下命令进行首次配置。在以下命令中，您可以将`Jane`替换为您自己的名字，将`jane@email.org`替换为您自己的电子邮件：

```cpp

$ git config --global user.email "jane@email.org"
$ git config --global user.name "Jane"
```

这些命令会更改全局 Git 配置。在 Git 仓库内部，您可以通过不指定`--global`选项来本地覆盖这些值。

默认情况下，Git 使用**vi**编辑器来编辑提交信息。如果您更喜欢其他编辑器，那么您可以通过类似的方式更改配置。要使用**nano**编辑器，您需要输入以下命令：

```cpp

$ git config --global core.editor nano
```

关于 Git 的更多信息，请参阅*Git 版本控制* *食谱* ([`www.packtpub.com/product/git-version-control-cookbook-second-edition/9781789137545`](https://www.packtpub.com/product/git-version-control-cookbook-second-edition/9781789137545))。

现在您已经准备好从 GitHub 克隆 LLVM 仓库了。

## 克隆仓库

克隆仓库的命令在所有平台上基本上是相同的。只有在 Windows 上，建议关闭自动转换行结束符的功能。

在所有非 Windows 平台上，您需要输入以下命令来克隆仓库：

```cpp

$ git clone https://github.com/llvm/llvm-project.git
```

只有在 Windows 上，才需要添加禁用自动转换行结束符的选项。这里，您需要输入以下命令：

```cpp

$ git clone --config core.autocrlf=false \
  https://github.com/llvm/llvm-project.git
```

这个 Git 命令将最新的源代码从 GitHub 克隆到名为`llvm-project`的本地目录中。现在使用以下命令将当前目录切换到新的`llvm-project`目录：

```cpp

$ cd llvm-project
```

目录内部包含所有 LLVM 项目，每个项目都在自己的目录中。最值得注意的是，LLVM 核心库位于`llvm`子目录中。LLVM 项目使用分支进行后续的发布开发（“release/17.x”）和标签（“llvmorg-17.0.1”）来标记特定的发布。使用前面的克隆命令，您将获得当前的开发状态。本书使用 LLVM 17。要将 LLVM 17 的第一个发布版本检出到一个名为`llvm-17`的分支，您需要输入以下命令：

```cpp

$ git checkout -b llvm-17 llvmorg-17.0.1
```

通过前面的步骤，您已经克隆了整个仓库并从标签创建了一个分支。这是最灵活的方法。

Git 还允许你仅克隆一个分支或一个标签（包括历史记录）。使用 `git clone --branch release/17.x https://github.com/llvm/llvm-project`，你只克隆 `release/17.x` 分支及其历史记录。这样，你就拥有了 LLVM 17 发布分支的最新状态，因此如果你需要确切的发布版本，你只需像以前一样从发布标签创建一个分支即可。使用额外的 `–-depth=1` 选项，这被称为 Git 的**浅克隆**，你还可以防止克隆历史记录。这节省了时间和空间，但显然限制了你在本地可以做的事情，包括基于发布标签检出分支。

## 创建构建目录

与许多其他项目不同，LLVM 不支持内联构建并需要一个单独的构建目录。最简单的方法是在 `llvm-project` 目录内创建，这是你的当前目录。为了简单起见，让我们将构建目录命名为 `build`。在这里，Unix 和 Windows 系统的命令不同。在类 Unix 系统上，你使用以下命令：

```cpp

$ mkdir build
```

在 Windows 上，使用以下命令：

```cpp

$ md build
```

现在，你已准备好在这个目录内使用 CMake 工具创建构建系统文件。

## 生成构建系统文件

为了生成使用 Ninja 编译 LLVM 和 clang 的构建系统文件，你运行以下命令：

```cpp

$ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=clang -B build -S llvm
```

`-G` 选项告诉 CMake 为哪个系统生成构建文件。该选项常用的值如下：

+   `Ninja` – 用于 Ninja 构建系统

+   `Unix Makefiles` – 用于 GNU Make

+   `Visual Studio 17 VS2022` – 用于 Visual Studio 和 MS Build

+   `Xcode` – 用于 Xcode 项目

使用 `–B` 选项，你告诉 CMake 构建目录的路径。同样，使用 `–S` 选项指定源目录。生成过程可以通过设置 `–D` 选项中的各种变量来影响。通常，它们以 `CMAKE_`（如果由 CMake 定义）或 `LLVM_`（如果由 LLVM 定义）为前缀。

如前所述，我们还对在 LLVM 旁边编译 clang 感兴趣。通过设置 `LLVM_ENABLE_PROJECTS=clang` 变量，这允许 CMake 生成 clang 的构建文件，除了 LLVM。此外，`CMAKE_BUILD_TYPE=Release` 变量告诉 CMake 应该生成发布构建的构建文件。

`–G` 选项的默认值取决于你的平台，构建类型的默认值取决于工具链。然而，你可以使用环境变量定义自己的偏好。`CMAKE_GENERATOR` 变量控制生成器，而 `CMAKE_BUILD_TYPE` 变量指定构建类型。如果你使用 **bash** 或类似的 shell，那么你可以使用以下方式设置变量：

```cpp

$ export CMAKE_GENERATOR=Ninja
$ export CMAKE_BUILD_TYPE=Release
```

如果你使用 Windows 命令提示符，那么你可以使用以下方式设置变量：

```cpp

$ set CMAKE_GENERATOR=Ninja
$ set CMAKE_BUILD_TYPE=Release
```

使用这些设置，创建构建系统文件的命令变为以下内容，这更容易输入：

```cpp

$ cmake -DLLVM_ENABLE_PROJECTS=clang -B build -S llvm
```

你可以在 *自定义构建过程* 部分找到更多关于 CMake 变量的信息。

## 编译和安装 LLVM

在生成构建文件后，可以使用以下方式编译 LLVM 和 clang：

```cpp

$ cmake –-build build
```

此命令在底层运行 Ninja，因为我们告诉 CMake 在配置步骤中生成 Ninja 文件。然而，如果您为支持多个构建配置的系统（如 Visual Studio）生成构建文件，则需要使用 `--config` 选项指定用于构建的配置。根据硬件资源，此命令的运行时间在 15 分钟（具有大量 CPU 核心、内存和快速存储的服务器）到数小时（双核 Windows 笔记本，内存有限）之间。

默认情况下，Ninja 会利用所有可用的 CPU 核心。这对于编译速度是有好处的，但可能会阻止其他任务运行；例如，在基于 Windows 的笔记本电脑上，当 Ninja 运行时几乎无法上网。幸运的是，您可以使用 `--j` 选项限制资源使用。

假设您有四个 CPU 核心可用，而 Ninja 应仅使用两个（因为您有并行任务要运行）；然后您使用此命令进行编译：

```cpp

$ cmake --build build –j2
```

编译完成后，一个最佳实践是运行测试套件以检查是否一切按预期工作：

```cpp

$ cmake --build build --target check-all
```

再次强调，此命令的运行时间会因可用硬件资源而大不相同。`check-all` Ninja 目标会运行所有测试用例。为包含测试用例的每个目录生成目标。使用 `check-llvm` 而不是 `check-all` 将运行 LLVM 测试但不运行 clang 测试；`check-llvm-codegen` 仅运行 LLVM 的 `CodeGen` 目录中的测试（即 `llvm/test/CodeGen` 目录）。

您还可以进行快速的手动检查。LLVM 应用程序中的一个选项是 `-version`，它显示 LLVM 版本、主机 CPU 和所有支持的架构：

```cpp

$ build/bin/llc --version
```

如果您在编译 LLVM 时遇到问题，那么您应该查阅 *Getting Started with the LLVM System* 文档中的 *常见问题* 部分 https://releases.llvm.org/17.0.1/docs/GettingStarted.html#common-problems) 以获取典型问题的解决方案。

作为最后一步，您可以安装二进制文件：

```cpp

$ cmake --install build
```

在类 Unix 系统上，安装目录是 `/usr/local`。在 Windows 上，使用 `C:\Program Files\LLVM`。当然，这也可以更改。下一节将解释如何更改。

# 自定义构建过程

CMake 系统使用 `CMakeLists.txt` 文件中的项目描述。顶级文件位于 `llvm` 目录中，`llvm/CMakeLists.txt`。其他目录也有 `CMakeLists.txt` 文件，在生成过程中递归包含。

根据项目描述中提供的信息，CMake 会检查已安装的编译器，检测库和符号，并创建构建系统文件，例如`build.ninja`或`Makefile`（取决于选择的生成器）。还可能定义可重用的模块，例如检测 LLVM 是否已安装的函数。这些脚本放置在特殊的`cmake`目录（`llvm/cmake`）中，在生成过程中会自动搜索。

构建过程可以通过 CMake 变量的定义进行自定义。命令行选项`–D`用于将变量设置为一个值。变量在 CMake 脚本中使用。由 CMake 本身定义的变量几乎总是以`CMAKE_`为前缀，并且这些变量可以在所有项目中使用。由 LLVM 定义的变量以`LLVM_`为前缀，但只有在项目定义中包含了对 LLVM 的使用时才能使用。

## 由 CMake 定义的变量

一些变量使用环境变量的值进行初始化。最显著的是`CC`和`CXX`，它们定义了用于构建的 C 和 C++编译器。CMake 会尝试自动定位 C 和 C++编译器，使用当前 shell 搜索路径。它会选择找到的第一个编译器。如果你安装了多个编译器，例如 gcc 和 clang 或不同版本的 clang，那么这可能不是你用于构建 LLVM 的编译器。

假设你希望使用 clang17 作为 C 编译器，clang++17 作为 C++编译器。那么，你可以在 Unix shell 中以以下方式调用 CMake：

```cpp

$ CC=clang17 CXX=clang++17 cmake –B build –S llvm
```

这只为`cmake`的调用设置环境变量的值。如果需要，你可以指定编译器可执行文件的绝对路径。

`CC`是`CMAKE_C_COMPILER` CMake 变量的默认值，`CXX`是`CMAKE_CXX_COMPILER` CMake 变量的默认值。而不是使用环境变量，你可以直接设置 CMake 变量。这相当于前面的调用：

```cpp

$ cmake –DCMAKE_C_COMPILER=clang17 \
  -DCMAKE_CXX_COMPILER=clang++17 –B build –S llvm
```

CMake 定义的其他有用变量如下：

| **变量名** | **用途** |
| --- | --- |
| `CMAKE_INSTALL_PREFIX` | 这是一个路径前缀，在安装过程中会添加到每个路径之前。在 Unix 上默认为`/usr/local`，在 Windows 上默认为`C:\Program Files\<Project>`。要在`/opt/llvm`目录中安装 LLVM，你指定`-DCMAKE_INSTALL_PREFIX=/opt/llvm`。二进制文件会复制到`/opt/llvm/bin`，库文件到`/opt/llvm/lib`，等等。 |
| `CMAKE_BUILD_TYPE` | 不同的构建类型需要不同的设置。例如，调试构建需要指定生成调试符号的选项，通常链接到系统库的调试版本。相比之下，发布构建使用优化标志并链接到库的生产版本。此变量仅用于只能处理一种构建类型的构建系统，例如 Ninja 或 Make。对于 IDE 构建系统，所有变体都会生成，您必须使用 IDE 的机制在构建类型之间切换。可能的值如下：`DEBUG`：带有调试符号的构建`RELEASE`：优化速度的构建`RELWITHDEBINFO`：带有调试符号的发布构建`MINSIZEREL`：优化大小的构建默认的构建类型是从 `CMAKE_BUILD_TYPE` 环境变量中获取的。如果此变量未设置，则默认值取决于使用的工具链，通常为空。为了生成发布构建的构建文件，您指定 `-DCMAKE_BUILD_TYPE=RELEASE`。 |
| `CMAKE_C_FLAGS` | `CMAKE_CXX_FLAGS` | 这些是在编译 C 和 C++ 源文件时使用的额外标志。初始值是从 `CFLAGS` 和 `CXXFLAGS` 环境变量中获取的，也可以用作替代。 |
| `CMAKE_MODULE_PATH` | 这指定了搜索 CMake 模块的附加目录。指定的目录在默认目录之前被搜索。该值是一个以分号分隔的目录列表。 |
| `PYTHON_EXECUTABLE` | 如果未找到 Python 解释器或您安装了多个版本而选择了错误的版本，您可以设置此变量为 Python 二进制的路径。此变量仅在包含 CMake Python 模块（对于 LLVM 是这种情况）时才有效。 |

表 1.1 - CMake 提供的附加有用变量

CMake 为变量提供了内置的帮助。`--help-variable var` 选项打印 `var` 变量的帮助。例如，您可以输入以下内容以获取 `CMAKE_BUILD_TYPE` 的帮助：

```cpp

$ cmake --help-variable CMAKE_BUILD_TYPE
```

您也可以使用以下命令列出所有变量：

```cpp

$ cmake --help-variable-list
```

此列表非常长。您可能希望将输出通过 `more` 或类似程序管道输出。

## 使用由 LLVM 定义的构建配置变量

由 LLVM 定义的构建配置变量与由 CMake 定义的变量工作方式相同，只是没有内置的帮助。最有用的变量可以在以下表中找到，其中它们被分为对首次安装 LLVM 的用户有用的变量，以及更高级的 LLVM 用户使用的变量。

### 对首次安装 LLVM 的用户有用的变量

| **变量名称** | **用途** |
| --- | --- |
| `LLVM_TARGETS_TO_BUILD` | LLVM 支持为不同的 CPU 架构生成代码。默认情况下，构建所有这些目标。使用此变量来指定要构建的目标列表，由分号分隔。当前的目标包括 `AArch64`、`AMDGPU`、`ARM`、`AVR`、`BPF`、`Hexagon`、`Lanai`、`LoongArch`、`Mips`、`MSP430`、`NVPTX`、`PowerPC`、`RISCV`、`Sparc`、`SystemZ`、`VE`、`WebAssembly`、`X86` 和 `XCore`。`all` 可以用作所有目标的简称。名称是区分大小写的。要仅启用 PowerPC 和 System Z 目标，你指定 `-DLLVM_TARGETS_TO_BUILD="PowerPC;SystemZ"`。 |
| `LLVM_EXPERIMENTAL_TARGETS_TO_BUILD` | 除了官方的目标之外，LLVM 源代码树还包含实验性目标。这些目标处于开发中，通常还不支持后端的所有功能。当前实验性目标的列表包括 `ARC`、`CSKY`、`DirectX`、`M68k`、`SPIRV` 和 `Xtensa`。要构建 `M68k` 目标，你指定 `-D LLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M68k`。 |
| `LLVM_ENABLE_PROJECTS` | 这是你要构建的项目列表，由分号分隔。项目的源必须在 `llvm` 目录同一级别（并排布局）。当前的列表包括 `bolt`、`clang`、`clang-tools-extra`、`compiler-rt`、`cross-project-tests`、`libc`、`libclc`、`lld`、`lldb`、`mlir`、`openmp`、`polly` 和 `pstl`。`all` 可以用作此列表中所有项目的简称。此外，你还可以在此处指定 `flang` 项目。由于一些特殊的构建要求，它目前还不是 `all` 列表的一部分。要一起构建 clang 和 bolt 与 LLVM，你指定 `-DLLVM_ENABLE_PROJECT="clang;bolt"`。 |

表 1.2 - 首次使用 LLVM 用户的有用变量

### LLVM 的高级用户变量

| `LLVM_ENABLE_ASSERTIONS` | 如果设置为 `ON`，则启用断言检查。这些检查有助于查找错误，在开发期间非常有用。对于 `DEBUG` 构建默认值为 `ON`，否则为 `OFF`。要启用断言检查（例如，对于 `RELEASE` 构建），你指定 `–DLLVM_ENABLE_ASSERTIONS=ON`。 |
| --- | --- |
| `LLVM_ENABLE_EXPENSIVE_CHECKS` | 这将启用一些昂贵的检查，这些检查可能会真正减慢编译速度或消耗大量内存。默认值是 `OFF`。要启用这些检查，你指定 `-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON`。 |
| `LLVM_APPEND_VC_REV` | 如果提供了 `-version` 命令行选项，LLVM 工具（如 `llc`）除了显示其他信息外，还会显示它们基于的 LLVM 版本。这个版本信息基于 `LLVM_REVISION` C 宏。默认情况下，LLVM 版本以及当前的 Git 哈希值都是版本信息的一部分。如果你正在跟踪 master 分支的开发，这很有用，因为它清楚地表明工具基于哪个 Git 提交。如果不需要，则可以使用 `–DLLVM_APPEND_VC_REV=OFF` 来关闭它。 |
| `LLVM_ENABLE_THREADS` | 如果检测到线程库（通常是`pthreads`库），LLVM 会自动包含线程支持。此外，在这种情况下，LLVM 假设编译器支持`-DLLVM_ENABLE_THREADS=OFF`。 |
| `LLVM_ENABLE_EH` | LLVM 项目不使用 C++异常处理，因此默认关闭异常支持。此设置可能与项目链接的其他库不兼容。如果需要，可以通过指定`–DLLVM_ENABLE_EH=ON`来启用异常支持。 |
| `LLVM_ENABLE_RTTI` | LLVM 使用一个轻量级、自建的运行时类型信息系统。默认情况下关闭 C++ RTTI 的生成。与异常处理支持一样，这可能与其他库不兼容。要启用 C++ RTTI 的生成，请指定`–DLLVM_ENABLE_RTTI=ON`。 |
| `LLVM_ENABLE_WARNINGS` | 如果可能，编译 LLVM 不应生成警告消息。因此，默认情况下启用了打印警告消息的选项。要关闭它，请指定`–DLLVM_ENABLE_WARNINGS=OFF`。 |
| `LLVM_ENABLE_PEDANTIC` | LLVM 源代码应遵循 C/C++语言标准；因此，默认情况下启用了源代码的严格检查。如果可能，也会禁用特定编译器的扩展。要反转此设置，请指定`–DLLVM_ENABLE_PEDANTIC=OFF`。 |
| `LLVM_ENABLE_WERROR` | 如果设置为`ON`，则所有警告都视为错误——一旦发现警告，编译就会终止。这有助于在源代码中找到所有剩余的警告。默认情况下是关闭的。要启用它，请指定`–DLLVM_ENABLE_WERROR=ON`。 |
| `LLVM_OPTIMIZED_TABLEGEN` | 通常，tablegen 工具会使用与 LLVM 其他部分相同的选项进行构建。同时，tablegen 用于生成代码生成器的大部分代码。因此，在调试构建中，tablegen 的速度会明显减慢，从而增加编译时间。如果此选项设置为`ON`，则即使在调试构建中，tablegen 也会启用优化进行编译，这可能会减少编译时间。默认是`OFF`。要启用它，请指定`–DLLVM_OPTIMIZED_TABLEGEN=ON`。 |
| `LLVM_USE_SPLIT_DWARF` | 如果构建编译器是 gcc 或 clang，则启用此选项将指示编译器在单独的文件中生成 DWARF 调试信息。对象文件大小的减少可以显著减少调试构建的链接时间。默认是`OFF`。要启用它，请指定`-LLVM_USE_SPLIT_DWARF=ON`。 |

表 1.3 - 高级 LLVM 用户的有用变量

注意

LLVM 定义了许多其他 CMake 变量。您可以在 LLVM 关于 CMake 的文档中找到完整的列表[`releases.llvm.org/17.0.1/docs/CMake.html#llvm-specific-variables`](https://releases.llvm.org/17.0.1/docs/CMake.html#llvm-specific-variables)。上述列表仅包含您最可能需要的变量。

# 摘要

在本章中，你已准备好你的开发机器以编译 LLVM。你已克隆了 GitHub 仓库并编译了你自己的 LLVM 和 clang 版本。构建过程可以通过 CMake 变量进行自定义。你了解了有用的变量以及如何更改它们。掌握了这些知识，你可以根据需要调整 LLVM。

在下一节中，我们将更深入地探讨编译器的结构。我们将探讨编译器内部的不同组件，以及在其中发生的不同类型的分析——特别是词法、语法和语义分析。最后，我们还将简要介绍与用于代码生成的 LLVM 后端进行接口连接。
