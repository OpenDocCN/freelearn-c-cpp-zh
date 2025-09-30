# *第一章*：构建 LLVM 时节省资源

LLVM 是许多令人惊叹的工业和学术项目采用的先进编译器优化和代码生成框架，例如 JavaScript 引擎中的 **即时编译器**（**JIT**）和 **机器学习**（**ML**）框架。它是构建编程语言和二进制文件工具的有用工具箱。然而，尽管该项目非常稳健，但其学习资源分散，而且文档也不是最好的。正因为如此，即使是有些 LLVM 经验的开发者，其学习曲线也相当陡峭。本书旨在通过以实用方式向您提供 LLVM 中常见和重要领域知识来解决这些问题——向您展示一些有用的工程技巧，指出一些不太为人所知但实用的功能，并举例说明有用的示例。

作为 **LLVM** 开发者，从源代码构建 LLVM 总是您应该做的第一件事。鉴于 LLVM 当前的规模，这项任务可能需要数小时才能完成。更糟糕的是，重建项目以反映更改也可能需要很长时间，从而阻碍您的生产力。因此，了解如何使用正确的工具以及如何为您的项目找到最佳的构建配置，以节省各种资源，尤其是您宝贵的时间，这一点至关重要。

在本章中，我们将涵盖以下主题：

+   通过更好的工具减少构建资源

+   通过调整 CMake 参数节省构建资源

+   学习如何使用 GN，一个替代的 LLVM 构建系统，以及其优缺点

# 技术要求

在撰写本书时，LLVM 只有一些软件要求：

+   支持 C++14 的 C/C++ 编译器

+   CMake

+   CMake 支持的构建系统之一，例如 GNU Make 或 Ninja

+   Python（2.7 也行，但我强烈建议使用 3.x）

+   zlib

这些项目的确切版本会不时发生变化。有关更多详细信息，请参阅 [`llvm.org/docs/GettingStarted.html#software`](https://llvm.org/docs/GettingStarted.html#software)。

本章假设您之前已经构建过 LLVM。如果不是这样，请执行以下步骤：

1.  从 GitHub 获取 LLVM 源代码树副本：

    ```cpp
    $ git clone https://github.com/llvm/llvm-project
    ```

1.  通常，默认分支应该无错误地构建。如果您想使用更稳定的发布版本，例如 10.x 版本的发布版本，请使用以下命令：

    ```cpp
    $ git clone -b release/10.x https://github.com/llvm/llvm-project
    ```

1.  最后，您应该创建一个构建文件夹，您将在其中调用 CMake 命令。所有构建工件也将放置在这个文件夹中。可以使用以下命令完成此操作：

    ```cpp
    $ mkdir .my_build
    $ cd .my_build
    ```

# 通过更好的工具减少构建资源

如本章开头所述，如果您使用默认（CMake）配置构建 LLVM，通过以下方式调用 **CMake** 并构建项目，整个过程可能需要 *数小时* 才能完成：

```cpp
$ cmake ../llvm
$ make all
```

这可以通过简单地使用更好的工具和更改一些环境来避免。在本节中，我们将介绍一些指导原则，以帮助您选择正确的工具和配置，这些工具和配置既可以加快您的构建时间，又可以改善内存占用。

## 用 Ninja 替换 GNU Make

我们可以做的第一个改进是使用 **Ninja** 构建工具 ([`ninja-build.org`](https://ninja-build.org)) 而不是 GNU Make，这是 CMake 在主要 Linux/Unix 平台上生成的默认构建系统。

这里有一些步骤可以帮助你在系统上设置 Ninja：

1.  例如，在 Ubuntu 上，你可以使用以下命令安装 Ninja：

    ```cpp
    $ sudo apt install ninja-build
    ```

    Ninja 也适用于大多数 Linux 发行版。

1.  然后，当你在构建 LLVM 时调用 CMake，请添加一个额外的参数：

    ```cpp
    $ cmake -G "Ninja" ../llvm
    ```

1.  最后，使用以下构建命令代替：

    ```cpp
    $ ninja all
    ```

在大型代码库如 LLVM 上，Ninja 比 GNU Make 运行得**显著**更快。Ninja 运行速度极快的一个秘密是，尽管大多数构建脚本如 `Makefile` 都是设计为手动编写的，但 Ninja 的构建脚本 `build.ninja` 的语法更类似于汇编代码，这应该**不应该**由开发者编辑，而应该由其他高级构建系统如 CMake 生成。Ninja 使用类似汇编的构建脚本的事实使得它能够在幕后进行许多优化，并消除许多冗余，例如在调用构建时的较慢解析速度。Ninja 在生成构建目标之间的依赖关系方面也有很好的声誉。

Ninja 在其**并行化程度**方面做出了聪明的决策；也就是说，你想要并行执行多少个作业。所以，通常你不需要担心这一点。如果你想显式地分配工作线程的数量，GNU Make 使用的相同命令行选项在这里仍然有效：

```cpp
$ ninja -j8 all
```

现在我们来看看如何避免使用 BFD 链接器。

## 避免使用 BFD 链接器

我们可以做的第二个改进是使用**除了** BFD 链接器之外的链接器，这是大多数 Linux 系统中使用的默认链接器。尽管 BFD 链接器是 Unix/Linux 系统上最成熟的链接器，但它并不是针对速度或内存消耗进行优化的。这会创建一个性能瓶颈，尤其是在像 LLVM 这样的大型项目中。这是因为，与编译阶段不同，链接阶段很难在文件级别上进行并行化。更不用说 BFD 链接器在构建 LLVM 时的峰值内存消耗通常约为 20 GB，这会给内存较少的计算机带来负担。幸运的是，至少有两种链接器在野外提供良好的单线程性能和低内存消耗：**GNU gold 链接器**和 LLVM 自带的链接器 **LLD**。

金链接器最初由谷歌开发，捐赠给了 GNU 的`binutils`。在现代 Linux 发行版中，您应该默认在`binutils`软件包中找到它。LLD 是 LLVM 的子项目之一，具有更快的链接速度和实验性的并行链接技术。一些 Linux 发行版（例如较新的 Ubuntu 版本）已经在其软件仓库中包含了 LLD。您也可以从 LLVM 的官方网站下载预构建版本。

要使用 gold 链接器或 LLD 构建您的 LLVM 源代码树，请添加一个额外的 CMake 参数，指定您想要使用的链接器名称。

对于 gold 链接器，使用以下命令：

```cpp
$ cmake -G "Ninja" -DLLVM_USE_LINKER=gold ../llvm
```

类似地，对于 LLD，使用以下命令：

```cpp
$ cmake -G "Ninja" -DLLVM_USE_LINKER=lld ../llvm
```

限制链接的并行线程数量

限制链接的并行线程数量是减少（峰值）内存消耗的另一种方法。您可以通过分配`LLVM_PARALLEL_LINK_JOBS=<N>` CMake 变量来实现这一点，其中`N`是期望的工作线程数。

通过使用不同的工具，我们可以显著减少构建时间。在下一节中，我们将通过调整 LLVM 的 CMake 参数来提高构建速度。

# 调整 CMake 参数

本节将向您展示 LLVM 构建系统中的一些最常见 CMake 参数，这些参数可以帮助您自定义构建并实现最大效率。

在我们开始之前，您应该有一个已经通过 CMake 配置的构建文件夹。以下大部分子部分将修改构建文件夹中的一个文件；即`CMakeCache.txt`文件。

## 选择正确的构建类型

LLVM 使用 CMake 提供的几个预定义的构建类型。其中最常见的是以下几种：

+   `Release`：如果您没有指定任何构建类型，这是默认的构建类型。它将采用最高的优化级别（通常是-O3）并消除大部分调试信息。通常，这种构建类型会使构建速度略微变慢。

+   `Debug`：这种构建类型将不应用任何优化（即-O0）。它保留所有调试信息。请注意，这将生成大量的工件，通常需要占用约 20GB 的空间，因此在使用此构建类型时，请确保您有足够的存储空间。由于没有进行优化，这通常会使构建速度略微加快。

+   `RelWithDebInfo`：这种构建类型尽可能多地应用编译器优化（通常是-O2）并保留所有调试信息。这是一个在空间消耗、运行时速度和可调试性之间取得平衡的选项。

您可以使用`CMAKE_BUILD_TYPE` CMake 变量选择其中之一。例如，要使用`RelWithDebInfo`类型，可以使用以下命令：

```cpp
$ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo …
```

建议首先使用`RelWithDebInfo`（如果你打算稍后调试 LLVM）。现代编译器在优化程序二进制中的调试信息质量方面已经取得了长足的进步。因此，始终先尝试它以避免不必要的存储浪费；如果事情没有按预期进行，你始终可以回到`Debug`类型。

除了配置构建类型外，`LLVM_ENABLE_ASSERTIONS`是另一个控制是否启用断言（即`assert(bool predicate)`函数，如果谓词参数不为真，则终止程序）的 CMake（布尔）参数。默认情况下，此标志仅在构建类型为`Debug`时为真，但你始终可以手动将其打开以强制执行更严格的检查，即使在其他构建类型中也是如此。

## 避免构建所有目标

在过去几年中，LLVM 支持的硬件目标数量迅速增长。在撰写本书时，有近 20 个官方支持的目标。每个目标都处理非平凡的任务，例如原生代码生成，因此构建需要花费相当多的时间。然而，你同时处理**所有**这些目标的几率很低。因此，你可以使用`LLVM_TARGETS_TO_BUILD` CMake 参数选择构建目标的一个子集。例如，要仅构建 X86 目标，我们可以使用以下命令：

```cpp
$ cmake -DLLVM_TARGETS_TO_BUILD="X86" …
```

你还可以使用分号分隔的列表指定多个目标，如下所示：

```cpp
$ cmake -DLLVM_TARGETS_TO_BUILD="X86;AArch64;AMDGPU" …
```

用双引号括起目标列表！

在某些 shell 中，例如`BASH`，分号是命令的结束符号。所以，如果你不用双引号括起目标列表，CMake 命令的其余部分将被截断。

让我们看看构建共享库如何帮助调整 CMake 参数。

## 构建为共享库

LLVM 最标志性的特性之一是其 Unix/Linux 中的`*.a`和 Windows 中的`*.lib`。然而，在这种情况下，静态库有以下缺点：

+   静态库的链接通常比动态库（Unix/Linux 中的`*.so`和 Windows 中的`*.dll`）的链接花费更多时间。

+   如果多个可执行文件链接到同一组库，例如许多 LLVM 工具所做的那样，当你采用静态库方法时，与动态库对应方法相比，这些可执行文件的总大小将**显著**更大。这是因为每个可执行文件都有这些库的副本。

+   当你使用调试器（例如 GDB）调试 LLVM 程序时，它们通常会在开始时花费相当多的时间加载静态链接的可执行文件，这会阻碍调试体验。

因此，建议在开发阶段使用`BUILD_SHARED_LIBS` CMake 参数将每个 LLVM 组件构建为动态库：

```cpp
$ cmake -DBUILD_SHARED_LIBS=ON …
```

这将为您节省大量的存储空间并加快构建过程。

## 分离调试信息

当你在调试模式下构建程序时——例如，使用 GCC 和 Clang 时添加`-g`标志——默认情况下，生成的二进制文件包含一个存储`cm``AKE_BUILD_TYPE=Debug`变量的部分——编译的库和可执行文件附带大量调试信息，这些信息占据了大量的磁盘空间。这导致以下问题：

+   由于 C/C++的设计，相同的调试信息可能会嵌入到不同的对象文件中（例如，头文件的调试信息可能嵌入到包含它的每个库中），这浪费了大量的磁盘空间。

+   链接器需要在链接阶段将对象文件及其相关的调试信息加载到内存中，这意味着如果对象文件包含非平凡的调试信息量，内存压力将会增加。

为了解决这些问题，LLVM 的构建系统提供了一种方法，允许我们将调试信息从原始对象文件中*分割*到单独的文件中。通过将调试信息从对象文件中分离出来，同一源文件的调试信息被压缩到一个地方，从而避免了不必要的重复创建并节省了大量磁盘空间。此外，由于调试信息不再是对象文件的一部分，链接器不再需要将它们加载到内存中，从而节省了大量内存资源。最后但同样重要的是，这个特性还可以提高我们的*增量*构建速度——即，在（小的）代码更改后重新构建项目——因为我们只需要更新单个地方的修改后的调试信息。

要使用此功能，请使用`LLVM_USE_SPLIT_DWARF` CMake 变量：

```cpp
$ cmake -DcmAKE_BUILD_TYPE=Debug -DLLVM_USE_SPLIT_DWARF=ON …
```

注意，这个 CMake 变量仅适用于使用 DWARF 调试格式的编译器，包括 GCC 和 Clang。

## 构建优化版本的`llvm-tblgen`

`llvm-tblgen`。换句话说，`llvm-tblgen`的运行时间将影响 LLVM 本身的构建时间。因此，如果你没有开发 TableGen 部分，无论全局构建类型（即`CMAKE_BUILD_TYPE`）如何，始终构建一个优化版本的`llvm-tblgen`都是一个好主意，这样可以使`llvm-tblgen`运行得更快，并缩短整体构建时间。

例如，以下 CMake 命令将创建构建配置，构建除`llvm-tblgen`可执行文件外的所有内容的调试版本，该可执行文件将作为优化版本构建：

```cpp
$ cmake -DLLVM_OPTIMIZED_TABLEGEN=ON -DCMAKE_BUILD_TYPE=Debug …
```

最后，你将看到如何使用 Clang 和新的 PassManager。

## 使用新的 PassManager 和 Clang

**Clang**是 LLVM 的官方 C 族前端（包括 C、C++和 Objective-C）。它使用 LLVM 的库生成机器代码，这些代码由 LLVM 中最重要的子系统之一——**PassManager**组织。PassManager 将所有优化和代码生成所需的任务（即 Passes）组合在一起。

在*第九章* *与 PassManager 和 AnalysisManager 一起工作*中，将介绍 LLVM 的*新* PassManager，它从头开始构建，以在未来某个时候替换现有的 PassManager。与传统的 PassManager 相比，新的 PassManager 具有更快的运行速度。这种优势间接地为 Clang 带来了更好的运行性能。因此，这里的想法非常简单：如果我们使用 Clang 并启用新的 PassManager 来构建 LLVM 的源代码树，编译速度将会更快。大多数主流 Linux 发行版的软件包仓库已经包含了 Clang。如果您想获得更稳定的 PassManager 实现，建议使用 Clang 6.0 或更高版本。使用`LLVM_USE_NEWPM` CMake 变量来使用新的 PassManager 构建 LLVM，如下所示：

```cpp
$ env CC=`which clang` CXX=`which clang++` \
  cmake -DLLVM_USE_NEWPM=ON …
```

LLVM 是一个庞大的项目，构建它需要花费很多时间。前两节介绍了一些提高其构建速度的有用技巧和提示。在下一节中，我们将介绍一个*替代*的构建系统来构建 LLVM。它相对于默认的 CMake 构建系统有一些优势，这意味着在某些场景下它将更加适合。

# 使用 GN 以获得更快的周转时间

CMake 是可移植和灵活的，并且已经被许多工业项目所实战检验。然而，在重新配置方面，它有一些严重的问题。正如我们在前几节中看到的，一旦构建文件生成，您可以通过编辑构建文件夹中的`CMakeCache.txt`文件来修改一些 CMake 参数。当您再次调用`build`命令时，CMake 将重新配置构建文件。如果您编辑源文件夹中的`CMakeLists.txt`文件，相同的重新配置也会启动。CMake 的重新配置过程主要有两个缺点：

+   在某些系统中，CMake 配置过程相当慢。即使是重新配置，理论上只运行部分过程，有时仍然需要很长时间。

+   有时，CMake 将无法解决不同变量和构建目标之间的依赖关系，因此您的更改将不会反映出来。在最坏的情况下，它将默默地失败，让您花费很长时间来找出问题。

**生成 Ninja**，也称为**GN**，是 Google 许多项目（如 Chromium）使用的构建文件生成器。GN 从其自己的描述语言生成 Ninja 文件。它因其快速的配置时间和可靠的参数管理而享有良好的声誉。自 2018 年底（大约版本 8.0.0）以来，LLVM 已经引入了 GN 支持，作为一种（实验性的）替代构建方法。如果您的开发更改了构建文件，或者您想在短时间内尝试不同的构建选项，GN 特别有用。

使用 GN 构建 LLVM 的步骤如下：

1.  LLVM 的 GN 支持位于`llvm/utils/gn`文件夹中。切换到该文件夹后，运行以下`get.py`脚本来本地下载 GN 的可执行文件：

    ```cpp
    get.py, simply put your version into the system's PATH. If you are wondering what other GN versions are available, you might want to check out the instructions for installing depot_tools at https://dev.chromium.org/developers/how-tos/install-depot-tools.
    ```

1.  在同一文件夹中使用 `gn.py` 生成构建文件（本地的 `gn.py` 只是真实 `gn` 的包装，用于设置基本环境）：

    ```cpp
    out/x64.release is the name of the build folder. Usually, GN users will name the build folder in <architecture>.<build type>.<other features> format.
    ```

1.  最后，您可以切换到构建文件夹并启动 Ninja：

    ```cpp
    $ cd out/x64.release
    $ ninja <build target>
    ```

1.  或者，您可以使用 `-C` Ninja 选项：

    ```cpp
    $ ninja -C out/x64.release <build target>
    ```

您可能已经知道，初始构建文件生成过程非常快。现在，如果您想更改一些构建参数，请导航到构建文件夹下的 `args.gn` 文件（在这个例子中是 `out/x64.release/args.gn`）；例如，如果您想将构建类型更改为 `debug` 并将目标构建（即 `LLVM_TARGETS_TO_BUILD` CMake 参数）改为 `X86` 和 `AArch64`。建议使用以下命令来启动编辑器编辑 `args.gn`：

```cpp
$ ./gn.py args out/x64.release
```

在 `args.gn` 编辑器中输入以下内容：

```cpp
# Inside args.gn
is_debug = true
llvm_targets_to_build = ["X86", "AArch64"]
```

保存并退出编辑器后，GN 将进行一些语法检查并重新生成构建文件（当然，您可以在不使用 `gn` 命令的情况下编辑 `args.gn`，并且构建文件不会重新生成，直到您调用 `ninja` 命令）。这种重新生成/重新配置也将很快。最重要的是，不会有任何不一致的行为。多亏了 GN 的语言设计，不同构建参数之间的关系可以很容易地分析，几乎没有歧义。

通过运行此命令可以找到 GN 的构建参数列表：

```cpp
$ ./gn.py args --list out/x64.release
```

不幸的是，在撰写本书时，仍有大量 CMake 参数尚未移植到 GN。GN 并非 LLVM 现有 CMake 构建系统的替代品，而是一个替代方案。尽管如此，如果您在涉及许多构建配置更改的开发中希望快速迭代，GN 仍然是一个不错的构建方法。

# 摘要

当涉及到构建用于代码优化和代码生成的工具时，LLVM 是一个有用的框架。然而，其代码库的大小和复杂性导致构建时间相当可观。本章提供了一些加快 LLVM 源树构建时间的技巧，包括使用不同的构建工具、选择正确的 CMake 参数，甚至采用除 CMake 之外的构建系统。这些技能减少了不必要的资源浪费，并在使用 LLVM 进行开发时提高了您的生产力。

在下一章中，我们将深入探讨基于 CMake 的 LLVM 构建基础设施，并展示如何构建在许多不同开发环境中至关重要的系统特性和指南。

# 进一步阅读

+   您可以在 [`llvm.org/docs/CMake.html#frequently-used-CMake-variables`](https://llvm.org/docs/CMake.html#frequently-used-CMake-variables) 查看由 LLVM 使用的完整 CMake 变量列表。

    你可以在[`gn.googlesource.com/gn`](https://gn.googlesource.com/gn)了解更多关于 GN 的信息。[`gn.googlesource.com/gn/+/master/docs/quick_start.md`](https://gn.googlesource.com/gn/+/master/docs/quick_start.md)上的快速入门指南也非常有帮助。
