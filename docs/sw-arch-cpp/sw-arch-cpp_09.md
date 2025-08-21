# 第七章：构建和打包

作为架构师，您需要了解构建过程的所有要素。本章将解释构建过程的所有要素。从编译器标志到自动化脚本等，我们将指导您到每个可能的模块、服务和构件都被版本化并存储在一个中央位置，准备部署。我们将主要关注 CMake。

在本章中，您将了解以下内容：

+   您应该考虑使用哪些编译器标志

+   如何基于现代 CMake 创建构建系统

+   如何构建可重用的组件

+   如何在 CMake 中清洁地使用外部代码

+   如何使用 CPack 创建 DEB 和 RPM 软件包，以及 NSIS 安装程序

+   如何使用 Conan 软件包管理器来安装您的依赖项并创建您自己的软件包

阅读完本章后，您将了解如何编写最先进的代码来构建和打包您的项目。

# 技术要求

要复制本章中的示例，您应安装最新版本的**GCC**和**Clang**，**CMake 3.15**或更高版本，**Conan**和**Boost 1.69**。

本章的源代码片段可以在[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter07`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter07)找到。

# 充分利用编译器

编译器是每个程序员工作室中最重要的工具之一。这就是为什么充分了解它们可以在许多不同的场合帮助您的原因。在本节中，我们将描述一些有效使用它们的技巧。这只是冰山一角，因为整本书都可以写关于这些工具及其广泛的可用标志、优化、功能和其他具体内容。GCC 甚至有一个关于编译器书籍的维基页面！您可以在本章末尾的*进一步阅读*部分找到它。

## 使用多个编译器

在构建过程中应考虑的一件事是使用多个编译器而不仅仅是一个，原因是它带来的几个好处。其中之一是它们可以检测代码中的不同问题。例如，MSVC 默认启用了符号检查。使用多个编译器可以帮助您解决将来可能遇到的潜在可移植性问题，特别是当决定在不同操作系统上编译代码时，例如从 Linux 迁移到 Windows 或反之。为了使这样的努力不花费任何成本，您应该努力编写可移植的、符合 ISO C++标准的代码。**Clang**的一个好处是它比 GCC 更注重符合 C++标准。如果您使用**MSVC**，请尝试添加`/permissive-`选项（自 Visual Studio 17 起可用；对于使用版本 15.5+创建的项目，默认启用）。对于**GCC**，在为代码选择 C++标准时，尽量不要使用 GNU 变体（例如，更喜欢`-std=c++17`而不是`-std=gnu++17`）。如果性能是您的目标，能够使用多种编译器构建软件还将使您能够选择为特定用例提供最快二进制文件的编译器。

无论您选择哪个编译器进行发布构建，都应考虑在开发中使用 Clang。它可以在 macOS、Linux 和 Windows 上运行，支持与 GCC 相同的一组标志，并旨在提供最快的构建时间和简洁的编译错误。

如果您使用 CMake，有两种常见的方法可以添加另一个编译器。一种是在调用 CMake 时传递适当的编译器，如下所示：

```cpp
mkdir build-release-gcc
cd build-release-gcc
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ 
```

也可以在调用 CMake 之前设置 CC 和 CXX，但这些变量并非在所有平台上都受到尊重（例如 macOS）。

另一种方法是使用工具链文件。如果你只需要使用不同的编译器，这可能有点过度，但当你想要交叉编译时，这是一个常用的解决方案。要使用工具链文件，你应该将其作为 CMake 参数传递：`-DCMAKE_TOOLCHAIN_FILE=toolchain.cmake`。

## 减少构建时间

每年，程序员们花费无数时间等待他们的构建完成。减少构建时间是提高整个团队生产力的简单方法，所以让我们讨论一下几种方法来做到这一点。

### 使用一个快速编译器

有时使构建更快的最简单方法之一是升级你的编译器。例如，通过将 Clang 升级到 7.0.0，你可以减少高达 30%的构建时间，使用**预编译头**（**PCH**）文件。自 Clang 9 以来，它已经获得了`-ftime-trace`选项，它可以为你提供有关它处理的所有文件的编译时间的信息。其他编译器也有类似的开关：比如查看 GCC 的`-ftime-report`或 MSVC 的`/Bt`和`/d2cgsummary`。通常情况下，通过切换编译器可以获得更快的编译速度，这在你的开发机器上尤其有用；例如，Clang 通常比 GCC 更快地编译代码。

一旦你有了一个快速的编译器，让我们看看它需要编译什么。

### 重新思考模板

编译过程的不同部分需要不同的时间来完成。这对于编译时构造尤为重要。Odin Holmes 的一个实习生 Chiel Douwes 基于对各种模板操作的编译时成本进行基准测试，创造了所谓的 Chiel 规则。这个规则以及其他基于类型的模板元编程技巧可以在 Odin Holmes 的*基于类型的模板元编程并没有死*讲座中看到。从最快到最慢，它们如下：

+   查找一个记忆化类型（例如，一个模板实例化）

+   向别名调用添加一个参数

+   添加一个参数到一个类型

+   调用一个别名

+   实例化一个类型

+   实例化一个函数模板

+   使用**SFINAE**（**替换失败不是错误**）

为了证明这个规则，考虑以下代码：

```cpp
template<bool>
 struct conditional {
     template<typename T, typename F>
     using type = F;
 };

 template<>
 struct conditional<true> {
     template<typename T, typename F>
     using type = T;
 };

 template<bool B, typename T, typename F>
 using conditional_t = conditional<B>::template type<T, F>;
```

它定义了一个`conditional`模板别名，它存储一个类型，如果条件`B`为真，则解析为`T`，否则解析为`F`。编写这样一个实用程序的传统方式如下：

```cpp
template<bool B, class T, class F>
 struct conditional {
     using type = T;
 };

 template<class T, class F>
 struct conditional<false, T, F> {
     using type = F;
 };

 template<bool B, class T, class F>
 using conditional_t = conditional<B,T,F>::type;
```

然而，这第二种方法比第一种编译速度慢，因为它依赖于创建模板实例而不是类型别名。

现在让我们看看你可以使用哪些工具及其特性来保持编译时间低。

### 利用工具

一个常见的技术，可以使你的构建更快，就是使用**单一编译单元构建**，或者**统一构建**。它不会加速每个项目，但如果你的头文件中有大量代码，这可能值得一试。统一构建通过将所有`.cpp`文件包含在一个翻译单元中来工作。另一个类似的想法是使用预编译头文件。像 CMake 的 Cotire 这样的插件将为你处理这两种技术。CMake 3.16 还增加了对统一构建的本机支持，你可以通过为一个目标启用它，`set_target_properties(<target> PROPERTIES UNITY_BUILD ON`，或者通过将`CMAKE_UNITY_BUILD`设置为`true`来全局启用。如果你只想要 PCHs，你可能需要查看 CMake 3.16 的`target_precompile_headers`。

如果你觉得你在 C++文件中包含了太多内容，考虑使用一个名为**include-what-you-use**的工具来整理它们。更倾向于前向声明类型和函数而不是包含头文件也可以在减少编译时间方面走得更远。

如果您的项目链接需要很长时间，也有一些应对方法。使用不同的链接器，例如 LLVM 的 LLD 或 GNU 的 Gold，可以帮助很多，特别是因为它们允许多线程链接。如果您负担不起使用不同的链接器，您可以尝试使用诸如`-fvisibility-hidden`或`-fvisibility-inlines-hidden`等标志，并在源代码中仅标记您希望在共享库中可见的函数。这样，链接器将有更少的工作要做。如果您正在使用链接时优化，尝试仅对性能关键的构建进行优化：计划进行性能分析和用于生产的构建。否则，您可能只会浪费开发人员的时间。

如果您正在使用 CMake 并且没有绑定到特定的生成器（例如，CLion 需要使用`Code::Blocks`生成器），您可以用更快的生成器替换默认的 Make 生成器。**Ninja**是一个很好的选择，因为它是专门用于减少构建时间而创建的。要使用它，只需在调用 CMake 时传递`-G Ninja`。

还有两个很棒的工具，肯定会给您带来帮助。其中一个是**Ccache**。它是一个运行其 C 和 C++编译输出缓存的工具。如果您尝试两次构建相同的东西，它将从缓存中获取结果，而不是运行编译。它保留统计信息，如缓存命中和未命中，可以记住在编译特定文件时应发出的警告，并具有许多配置选项，可以存储在`~/.ccache/ccache.conf`文件中。要获取其统计信息，只需运行`ccache --show-stats`。

第二个工具是**IceCC**（或 Icecream）。这是 distcc 的一个分支，本质上是一个工具，可以在多台主机上分发您的构建。使用 IceCC，更容易使用自定义工具链。它在每台主机上运行 iceccd 守护程序和一个管理整个集群的 icecc-scheduler 服务。调度程序与 distcc 不同，它确保仅使用每台机器上的空闲周期，因此您不会过载其他人的工作站。

要在 CMake 构建中同时使用 IceCC 和 Ccache，只需在 CMake 调用中添加`-DCMAKE_C_COMPILER_LAUNCHER="ccache;icecc" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache;icecc"`。如果您在 Windows 上编译，您可以使用 clcache 和 Incredibuild，或者寻找其他替代方案，而不是最后两个工具。

现在您知道如何快速构建，让我们继续另一个重要的主题。

## 查找潜在的代码问题

即使最快的构建也不值得，如果你的代码有错误。有数十个标志可以警告您代码中的潜在问题。本节将尝试回答您应该考虑启用哪些标志。

首先，让我们从一个略有不同的问题开始：如何避免收到来自其他库代码的问题警告。收到无法真正修复的问题警告是没有用的。幸运的是，有编译器开关可以禁用此类警告。例如，在 GCC 中，您有两种类型的`include`文件：常规文件（使用`-I`传递）和系统文件（使用`-isystem`传递）。如果您使用后者指定一个目录，您将不会收到它包含的头文件的警告。MSVC 有一个等效于`-isystem`的选项：`/external:I`。此外，它还有其他用于处理外部包含的标志，例如`/external:anglebrackets`，告诉编译器将使用尖括号包含的所有文件视为外部文件，从而禁用对它们的警告。您可以为外部文件指定警告级别。您还可以保留由您的代码引起的模板实例化产生的警告，使用`/external:templates-`。如果您正在寻找一种将`include`路径标记为系统/外部路径的便携方式，并且正在使用 CMake，您可以在`target_include_directories`指令中添加`SYSTEM`关键字。

谈到可移植性，如果您想符合 C++标准（您应该这样做），请考虑为 GCC 或 Clang 的编译选项添加-pedantic，或者为 MSVC 添加/permissive-选项。这样，您将得到关于您可能正在使用的每个非标准扩展的信息。如果您使用 CMake，请为每个目标添加以下行，set_target_properties(<target> PROPERTIES CXX_EXTENSIONS OFF)，以禁用特定于编译器的扩展。

如果您正在使用 MSVC，请努力使用/W4 编译代码，因为它启用了大部分重要的警告。对于 GCC 和 Clang，请尝试使用-Wall -Wextra -Wconversion -Wsign-conversion。第一个尽管名字是这样，但只启用了一些常见的警告。然而，第二个添加了另一堆警告。第三个基于 Scott Meyers 的一本名为《Effective C++》的好书中的建议（这是一组很好的警告，但请检查它是否对您的需求太吵闹）。最后两个是关于类型转换和符号转换的。所有这些标志一起创建了一个理智的安全网，但您当然可以寻找更多要启用的标志。Clang 有一个-Weverything 标志。尝试定期使用它运行构建，以发现可能值得在您的代码库中启用的新的潜在警告。您可能会对使用此标志获得多少消息感到惊讶，尽管启用一些警告标志可能不值得麻烦。MSVC 的替代方案名为/Wall。看一下以下表格，看看之前未启用的其他一些有趣的选项：

GCC/Clang:

| Flag | 意义 |
| --- | --- |
| -Wduplicated-cond | 当在 if 和 else-if 块中使用相同条件时发出警告。 |
| -Wduplicated-branches | 如果两个分支包含相同的源代码，则发出警告。 |
| -Wlogical-op | 当逻辑操作中的操作数相同时发出警告，并且应使用位操作符时发出警告。 |
| -Wnon-virtual-dtor | 当一个类有虚函数但没有虚析构函数时发出警告。 |
| -Wnull-dereference | 警告空指针解引用。此检查可能在未经优化的构建中处于非活动状态。 |
| -Wuseless-cast | 当转换为相同类型时发出警告。 |
| -Wshadow | 一系列关于声明遮蔽其他先前声明的警告。 |

MSVC:

| Flag | 意义 |
| --- | --- |
| /w44640 | 警告非线程安全的静态成员初始化。 |

最后值得一提的是一个问题：是否使用-Werror（或 MSVC 上的/WX）？这实际上取决于您的个人偏好，因为发出错误而不是警告有其利弊。好的一面是，您不会让任何已启用的警告溜走。您的 CI 构建将失败，您的代码将无法编译。在运行多线程构建时，您不会在快速通过的编译消息中丢失任何警告。然而，也有一些坏处。如果编译器启用了任何新的警告或只是检测到更多问题，您将无法升级编译器。对于依赖项也是一样，它们可能会废弃一些提供的函数。如果您的代码被项目的其他部分使用，您将无法废弃其中的任何内容。幸运的是，您总是可以使用混合解决方案：努力使用-Werror 进行编译，但在需要执行它所禁止的操作时将其禁用。这需要纪律，因为如果有任何新的警告滑入，您可能会很难消除它们。

## 使用以编译器为中心的工具

现在，编译器允许您做的事情比几年前多得多。这归功于 LLVM 和 Clang 的引入。通过提供 API 和模块化架构，使得诸如消毒剂、自动重构或代码完成引擎等工具得以蓬勃发展。您应该考虑利用这个编译器基础设施所提供的优势。使用 clang-format 确保代码库中的所有代码符合给定的标准。考虑使用 pre-commit 工具添加预提交挂钩，在提交之前重新格式化新代码。您还可以将 Python 和 CMake 格式化程序添加到其中。使用 clang-tidy 对代码进行静态分析——这是一个实际理解您的代码而不仅仅是推理的工具。这个工具可以为您执行大量不同的检查，所以一定要根据您的特定需求自定义列表和选项。您还可以在启用消毒剂的情况下每晚或每周运行软件测试。这样，您可以检测线程问题、未定义行为、内存访问、管理问题等。如果您的发布版本禁用了断言，使用调试版本运行测试也可能有价值。

如果您认为还可以做更多，您可以考虑使用 Clang 的基础设施编写自己的代码重构。如果您想看看如何创建一个基于 LLVM 的工具，已经有了一个`clang-rename`工具。对于 clang-tidy 的额外检查和修复也不难创建，它们可以为您节省数小时的手动劳动。

您可以将许多工具整合到您的构建过程中。现在让我们讨论这个过程的核心：构建系统。

# 摘要构建过程

在本节中，我们将深入研究 CMake 脚本，这是全球 C++项目中使用的事实标准构建系统生成器。

## 介绍 CMake

CMake 是构建系统生成器而不是构建系统本身意味着什么？简单地说，CMake 可以用来生成各种类型的构建系统。您可以使用它来生成 Visual Studio 项目、Makefile 项目、基于 Ninja 的项目、Sublime、Eclipse 和其他一些项目。

CMake 还配备了一系列其他工具，如用于执行测试的 CTest 和用于打包和创建安装程序的 CPack。CMake 本身也允许导出和安装目标。

CMake 的生成器可以是单配置的，比如 Make 或 NMAKE，也可以是多配置的，比如 Visual Studio。对于单配置的生成器，在首次在文件夹中运行生成时，应传递`CMAKE_BUILD_TYPE`标志。例如，要配置调试构建，您可以运行`cmake <project_directory> -DCMAKE_BUILD_TYPE=Debug`。其他预定义的配置有`Release`、`RelWithDebInfo`（带有调试符号的发布）和`MinSizeRel`（最小二进制大小的发布优化）。为了保持源目录清洁，始终创建一个单独的构建文件夹，并从那里运行 CMake 生成。

虽然可以添加自己的构建类型，但您真的应该尽量避免这样做，因为这会使一些 IDE 的使用变得更加困难，而且不具有可扩展性。一个更好的选择是使用`option`。

CMake 文件可以以两种风格编写：一种是基于变量的过时风格，另一种是基于目标的现代 CMake 风格。我们这里只关注后者。尽量遏制通过全局变量设置事物，因为这会在您想要重用目标时引起问题。

### 创建 CMake 项目

每个 CMake 项目的顶层`CMakeLists.txt`文件中应包含以下行：

```cpp
cmake_minimum_required(VERSION 3.15...3.19)

project(
   Customer
   VERSION 0.0.1
   LANGUAGES CXX)
```

设置最低和最大支持的版本很重要，因为它会影响 CMake 的行为，通过设置策略。如果需要，您也可以手动设置它们。

我们项目的定义指定了它的名称、版本（将用于填充一些变量）和 CMake 将用于构建项目的编程语言（这将填充更多变量并找到所需的工具）。

一个典型的 C++项目有以下目录：

+   `cmake`：用于 CMake 脚本

+   `include`：用于公共头文件，通常带有一个项目名称的子文件夹

+   `src`：用于源文件和私有头文件

+   `test`：用于测试

你可以使用 CMake 目录来存储你的自定义 CMake 模块。为了方便从这个目录访问脚本，你可以将它添加到 CMake 的`include()`搜索路径中，就像这样：

```cpp
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake"
```

在包含 CMake 模块时，你可以省略`.cmake`后缀。这意味着`include(CommonCompileFlags.cmake)`等同于`include(CommonCompileFlags)`。

### 区分 CMake 目录变量

在 CMake 中浏览目录有一个常见的陷阱，不是每个人都意识到。在编写 CMake 脚本时，尝试区分以下内置变量：

+   `PROJECT_SOURCE_DIR`：`project`命令最后一次从 CMake 脚本中调用的目录。

+   `PROJECT_BINARY_DIR`：与前一个相同，但用于构建目录树。

+   `CMAKE_SOURCE_DIR`：顶层源目录（这可能是另一个项目，只是将我们作为依赖项/子目录添加进来）。

+   `CMAKE_BINARY_DIR`：与`CMAKE_SOURCE_DIR`相同，但用于构建目录树。

+   `CMAKE_CURRENT_SOURCE_DIR`：对应于当前处理的`CMakeLists.txt`文件的源目录。

+   `CMAKE_CURRENT_BINARY_DIR`：与`CMAKE_CURRENT_SOURCE_DIR`匹配的二进制（构建）目录。

+   `CMAKE_CURRENT_LIST_DIR`：`CMAKE_CURRENT_LIST_FILE`的目录。如果当前的 CMake 脚本是从另一个脚本中包含的（对于被包含的 CMake 模块来说很常见），它可能与当前源目录不同。

搞清楚了这一点，现在让我们开始浏览这些目录。

在你的顶层`CMakeLists.txt`文件中，你可能想要调用`add_subdirectory(src)`，这样 CMake 将处理那个目录。

### 指定 CMake 目标

在`src`目录中，你应该有另一个`CMakeLists.txt`文件，这次可能定义了一个或两个目标。让我们为我们之前在书中提到的多米尼加展会系统添加一个客户微服务的可执行文件：

```cpp
add_executable(customer main.cpp)
```

源文件可以像前面的代码行那样指定，也可以稍后使用`target_sources`添加。

一个常见的 CMake 反模式是使用通配符来指定源文件。使用它们的一个很大的缺点是，CMake 不会知道文件是否被添加，直到重新运行生成。这样做的一个常见后果是，如果你从存储库中拉取更改然后简单地构建，你可能会错过编译和运行新的单元测试或其他代码。即使你使用了`CONFIGURE_DEPENDS`和通配符，构建时间也会变长，因为通配符必须作为每次构建的一部分进行检查。此外，该标志可能无法可靠地与所有生成器一起使用。即使 CMake 的作者也不鼓励使用它，而是更倾向于明确声明源文件。

好的，我们定义了我们的源代码。现在让我们指定我们的目标需要编译器支持 C++17：

```cpp
target_compile_features(customer PRIVATE cxx_std_17)
```

`PRIVATE`关键字指定这是一个内部要求，即只对这个特定目标可见，而不对依赖于它的任何目标可见。如果你正在编写一个提供用户 C++17 API 的库，你可以使用`INTERFACE`关键字。要同时指定接口和内部要求，你可以使用`PUBLIC`关键字。当使用者链接到我们的目标时，CMake 将自动要求它也支持 C++17。如果你正在编写一个不被构建的目标（即一个仅包含头文件的库或一个导入的目标），通常使用`INTERFACE`关键字就足够了。

你还应该注意，指定我们的目标要使用 C++17 特性并不强制执行 C++标准或禁止编译器扩展。要这样做，你应该调用以下命令：

```cpp
set_target_properties(customer PROPERTIES
     CXX_STANDARD 17
     CXX_STANDARD_REQUIRED YES
     CXX_EXTENSIONS NO
 )
```

如果你想要一组编译器标志传递给每个目标，你可以将它们存储在一个变量中，并在想要创建一个具有这些标志设置为`INTERFACE`的目标时调用以下命令，并且没有任何源并且使用这个目标在`target_link_libraries`中：

```cpp
target_compile_options(customer PRIVATE ${BASE_COMPILE_FLAGS})
```

该命令会自动传播包含目录、选项、宏和其他属性，而不仅仅是添加链接器标志。说到链接，让我们创建一个库，我们将与之链接：

```cpp
add_library(libcustomer lib.cpp)
add_library(domifair::libcustomer ALIAS libcustomer)
set_target_properties(libcustomer PROPERTIES OUTPUT_NAME customer)
# ...
target_link_libraries(customer PRIVATE libcustomer)
```

`add_library`可用于创建静态、共享、对象和接口（考虑头文件）库，以及定义任何导入的库。

它的**`ALIAS`**版本创建了一个命名空间目标，有助于调试许多 CMake 问题，是一种推荐的现代 CMake 实践。

因为我们已经给我们的目标添加了`lib`前缀，所以我们将输出名称设置为**`libcustomer.a`**而不是`liblibcustomer.a`。

最后，我们将我们的可执行文件与添加的库链接起来。尽量始终为`target_link_libraries`命令指定`PUBLIC`、`PRIVATE`或`INTERFACE`关键字，因为这对于 CMake 有效地管理目标依赖关系的传递性至关重要。

### 指定输出目录

一旦您使用`cmake --build .`等命令构建代码，您可能想知道在哪里找到构建产物。默认情况下，CMake 会将它们创建在与它们定义的源目录匹配的目录中。例如，如果您有一个带有`add_executable`指令的`src/CMakeLists.txt`文件，那么二进制文件将默认放在构建目录的`src`子目录中。我们可以使用以下代码来覆盖这一点：

```cpp
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
```

这样，二进制文件和 DLL 文件将放在项目构建目录的`bin`子目录中，而静态和共享 Linux 库将放在`lib`子目录中。

## 使用生成器表达式

以一种既支持单配置生成器又支持多配置生成器的方式设置编译标志可能会很棘手，因为 CMake 在配置时间执行`if`语句和许多其他结构，而不是在构建/安装时间执行。

这意味着以下是 CMake 的反模式：

```cpp
if(CMAKE_BUILD_TYPE STREQUAL Release)
   target_compile_definitions(libcustomer PRIVATE RUN_FAST)
endif()
```

相反，生成器表达式是实现相同目标的正确方式，因为它们在稍后的时间被处理。让我们看一个实际使用它们的例子。假设您想为您的`Release`配置添加一个预处理器定义，您可以编写以下内容：

```cpp
target_compile_definitions(libcustomer PRIVATE "$<$<CONFIG:Release>:RUN_FAST>")
```

这将仅在构建所选的配置时解析为`RUN_FAST`。对于其他配置，它将解析为空值。它适用于单配置和多配置生成器。然而，这并不是生成器表达式的唯一用例。

在构建期间由我们的项目使用时，我们的目标的某些方面可能会有所不同，并且在安装目标时由其他项目使用时也会有所不同。一个很好的例子是**包含目录**。在 CMake 中处理这个问题的常见方法如下：

```cpp
target_include_directories(
   libcustomer PUBLIC $<INSTALL_INTERFACE:include>
                      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
```

在这种情况下，我们有两个生成器表达式。第一个告诉我们，当安装时，可以在`include`目录中找到包含文件，相对于安装前缀（安装的根目录）。如果我们不安装，这个表达式将变为空。这就是为什么我们有另一个用于构建的表达式。这将解析为上次使用`project()`找到的目录的`include`子目录。

不要在模块之外的路径上使用`target_include_directories`。如果这样做，您就是**偷**别人的头文件，而不是明确声明库/目标依赖关系。这是 CMake 的反模式。

CMake 定义了许多生成器表达式，您可以使用这些表达式来查询编译器和平台，以及目标（例如完整名称、对象文件列表、任何属性值等）。除此之外，还有运行布尔操作、if 语句、字符串比较等表达式。

现在，举一个更复杂的例子，假设您想要有一组编译标志，您可以在所有目标上使用，并且这些标志取决于所使用的编译器，您可以定义如下：

```cpp
list(
   APPEND
   BASE_COMPILE_FLAGS
   "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall;-Wextra;-pedantic;-Werror>"
   "$<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>")
```

如果编译器是 Clang 或 AppleClang 或 GCC，则会附加一组标志，如果使用的是 MSVC，则会附加另一组标志。请注意，我们使用分号分隔标志，因为这是 CMake 在列表中分隔元素的方式。

现在让我们看看如何为我们的项目添加外部代码供其使用。

# 使用外部模块

有几种方法可以获取您所依赖的外部项目。例如，您可以将它们添加为 Conan 依赖项，使用 CMake 的`find_package`来查找操作系统提供的版本或以其他方式安装的版本，或者自行获取和编译依赖项。

本节的关键信息是：如果可以的话，应该使用 Conan。这样，您将最终使用与您的项目及其依赖项要求相匹配的依赖项版本。

如果您的目标是支持多个平台，甚至是同一发行版的多个版本，使用 Conan 或自行编译都是可行的方法。这样，无论您在哪个操作系统上编译，都将使用相同的依赖项版本。

让我们讨论一下 CMake 本身提供的几种抓取依赖项的方法，然后转而使用名为 Conan 的多平台包管理器。

## 获取依赖项

使用 CMake 内置的`FetchContent`模块从源代码准备依赖项的一种可能的方法是。它将为您下载依赖项，然后像常规目标一样构建它们。

该功能在 CMake 3.11 中推出。它是`ExternalProject`模块的替代品，后者有许多缺陷。其中之一是它在构建时克隆了外部存储库，因此 CMake 无法理解外部项目定义的目标，以及它们的依赖关系。这使得许多项目不得不手动定义这些外部目标的`include`目录和库路径，并完全忽略它们所需的接口编译标志和依赖关系。`FetchContent`没有这样的问题，因此建议您使用它。

在展示如何使用之前，您必须知道`FetchContent`和`ExternalProject`（以及使用 Git 子模块和类似方法）都有一个重要的缺陷。如果您有许多依赖项使用同一个第三方库，您可能最终会得到同一项目的多个版本，例如几个版本的 Boost。使用 Conan 等包管理器可以帮助您避免这种问题。

举个例子，让我们演示如何使用上述的`FetchContent`功能将**GTest**集成到您的项目中。首先，创建一个`FetchGTest.cmake`文件，并将其放在我们源代码树中的`cmake`目录中。我们的`FetchGTest`脚本将定义如下：

```cpp
include(FetchContent)

 FetchContent_Declare(
   googletest
   GIT_REPOSITORY https://github.com/google/googletest.git
   GIT_TAG dcc92d0ab6c4ce022162a23566d44f673251eee4)

 FetchContent_GetProperties(googletest)
 if(NOT googletest_POPULATED)
   FetchContent_Populate(googletest)
   add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}
                    EXCLUDE_FROM_ALL)
 endif()

 message(STATUS "GTest binaries are present at ${googletest_BINARY_DIR}")

```

首先，我们包含内置的`FetchContent`模块。一旦加载了该模块，我们就可以使用`FetchContent_Declare`来声明依赖项。现在，让我们命名我们的依赖项，并指定 CMake 将克隆的存储库以及它将检出的修订版本。

现在，我们可以读取我们外部库的属性并填充（即检出）它（如果尚未完成）。一旦我们有了源代码，我们可以使用`add_subdirectory`来处理它们。`EXCLUDE_FROM_ALL`选项将告诉 CMake 在运行诸如`make all`这样的命令时，如果其他目标不需要它们，就不要构建这些目标。在成功处理目录后，我们的脚本将打印一条消息，指示 GTests 库在构建后将位于哪个目录中。

如果您不喜欢将依赖项与项目一起构建，也许下一种集成依赖项的方式更适合您。

## 使用查找脚本

假设你的依赖项在主机的某个地方可用，你可以调用`find_package`来尝试搜索它。如果你的依赖项提供了配置或目标文件（稍后会详细介绍），那么只需编写这一个简单的命令就足够了。当然，前提是依赖项已经在你的机器上可用。如果没有，你需要在运行 CMake 之前安装它们。

要创建前面的文件，你的依赖项需要使用 CMake，但这并不总是情况。那么，你该如何处理那些不使用 CMake 的库呢？如果这个库很受欢迎，很可能已经有人为你创建了一个查找脚本。版本早于 1.70 的 Boost 库就是这种方法的一个常见例子。CMake 自带一个`FindBoost`模块，你可以通过运行`find_package(Boost)`来执行它。

要使用前面的模块找到 Boost，你首先需要在系统上安装它。之后，在你的 CMake 列表中，你应该设置任何你认为合理的选项。例如，要使用动态和多线程 Boost 库，而不是静态链接到 C++运行时，指定如下：

```cpp
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
```

然后，你需要实际搜索库，如下所示：

```cpp
find_package(Boost 1.69 EXACT REQUIRED COMPONENTS Beast)
```

在这里，我们指定我们只想使用 Beast，这是 Boost 的一部分，一个很棒的网络库。一旦找到，你可以将它链接到你的目标，如下所示：

```cpp
target_link_libraries(MyTarget PUBLIC Boost::Beast)
```

现在你知道如何正确使用查找脚本了，让我们学习如何自己编写一个。

## 编写查找脚本

如果你的依赖项既没有提供配置和目标文件，也没有人为其编写查找模块，你总是可以自己编写这样的模块。

这不是你经常做的事情，所以我们会尽量简要地介绍一下这个主题。如果你想深入了解，你还应该阅读官方 CMake 文档中的指南（在*进一步阅读*部分中链接），或者查看 CMake 安装的一些查找模块（通常在 Unix 系统的`/usr/share/cmake-3.17/Modules`等目录中）。为简单起见，我们假设你只想找到你的依赖项的一个配置，但也可以分别找到`Release`和`Debug`二进制文件。这将导致设置不同的目标和相关变量。

脚本名称决定了你将传递给`find_package`的参数；例如，如果你希望最终得到`find_package(Foo)`，那么你的脚本应该命名为`FindFoo.cmake`。

良好的做法是从一个`reStructuredText`部分开始编写脚本，描述你的脚本实际要做什么，它将设置哪些变量等等。这样的描述示例可能如下：

```cpp
 #.rst:
 # FindMyDep
 # ----------
 #
 # Find my favourite external dependency (MyDep).
 #
 # Imported targets
 # ^^^^^^^^^^^^^^^^
 #
 # This module defines the following :prop_tgt:`IMPORTED` target:
 #
 # ``MyDep::MyDep``
 #   The MyDep library, if found.
 #
```

通常，你还会想描述一下你的脚本将设置的变量：

```cpp
 # Result variables
 # ^^^^^^^^^^^^^^^^
 #
 # This module will set the following variables in your project:
 #
 # ``MyDep_FOUND``
 #   whether MyDep was found or not
 # ``MyDep_VERSION_STRING``
 #   the found version of MyDep
```

如果`MyDep`本身有任何依赖项，现在就是找到它们的时候了：

```cpp
find_package(Boost REQUIRED)
```

现在我们可以开始搜索库了。一个常见的方法是使用`pkg-config`：

```cpp
find_package(PkgConfig)
pkg_check_modules(PC_MyDep QUIET MyDep)
```

如果`pkg-config`有关于我们的依赖项的信息，它将设置一些我们可以用来找到它的变量。

一个好主意可能是让我们的脚本用户设置一个变量，指向库的位置。按照 CMake 的约定，它应该被命名为`MyDep_ROOT_DIR`。用户可以通过在构建目录中调用`-DMyDep_ROOT_DIR=some/path`来提供这个变量给 CMake，修改`CMakeCache.txt`中的变量，或者使用`ccmake`或`cmake-gui`程序。

现在，我们可以使用前面提到的路径实际搜索我们的依赖项的头文件和库：

```cpp
find_path(MyDep_INCLUDE_DIR
   NAMES MyDep.h
   PATHS "${MyDep_ROOT_DIR}/include" "${PC_MyDep_INCLUDE_DIRS}"
   PATH_SUFFIXES MyDep
 )

 find_library(MyDep_LIBRARY
   NAMES mydep
   PATHS "${MyDep_ROOT_DIR}/lib" "${PC_MyDep_LIBRARY_DIRS}"
 )
```

然后，我们还需要设置找到的版本，就像我们在脚本头部承诺的那样。要使用从`pkg-config`找到的版本，我们可以编写如下内容：

```cpp
set(MyDep_VERSION ${PC_MyDep_VERSION})
```

或者，我们可以手动从头文件的内容、库路径的组件或使用其他任何方法中提取版本。完成后，让我们利用 CMake 的内置脚本来决定库是否成功找到，同时处理`find_package`调用的所有可能参数：

```cpp
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MyDep
         FOUND_VAR MyDep_FOUND
         REQUIRED_VARS
         MyDep_LIBRARY
         MyDep_INCLUDE_DIR
         VERSION_VAR MyDep_VERSION
         )
```

由于我们决定提供一个目标而不仅仅是一堆变量，现在是定义它的时候了：

```cpp
if(MyDep_FOUND AND NOT TARGET MyDep::MyDep)
     add_library(MyDep::MyDep UNKNOWN IMPORTED)
     set_target_properties(MyDep::MyDep PROPERTIES
             IMPORTED_LOCATION "${MyDep_LIBRARY}"
             INTERFACE_COMPILE_OPTIONS "${PC_MyDep_CFLAGS_OTHER}"
             INTERFACE_INCLUDE_DIRECTORIES "${MyDep_INCLUDE_DIR}"
             INTERFACE_LINK_LIBRARIES Boost::boost
             )
endif()
```

最后，让我们隐藏我们内部使用的变量，以免让不想处理它们的用户看到：

```cpp
mark_as_advanced(
 MyDep_INCLUDE_DIR
 MyDep_LIBRARY
 )
```

现在，我们有了一个完整的查找模块，我们可以按以下方式使用它：

```cpp
find_package(MyDep REQUIRED)
target_link_libraries(MyTarget PRIVATE MyDep::MyDep)
```

这就是您可以自己编写查找模块的方法。

不要为您自己的包编写`Find\*.cmake`模块。这些模块是为不支持 CMake 的包而设计的。相反，编写一个`Config\*.cmake`模块（如本章后面所述）。

现在让我们展示如何使用一个合适的包管理器，而不是自己来处理繁重的工作。

## 使用 Conan 包管理器

Conan 是一个开源的、去中心化的本地包管理器。它支持多个平台和编译器。它还可以与多个构建系统集成。

如果某个包在您的环境中尚未构建，Conan 将在您的计算机上处理构建它，而不是下载已构建的版本。构建完成后，您可以将其上传到公共存储库、您自己的`conan_server`实例，或者 Artifactory 服务器。

### 准备 Conan 配置文件

如果这是您第一次运行 Conan，它将根据您的环境创建一个默认配置文件。您可能希望通过创建新配置文件或更新默认配置文件来修改其中的一些设置。假设我们正在使用 Linux，并且希望使用 GCC 9.x 编译所有内容，我们可以运行以下命令：

```cpp
 conan profile new hosacpp
 conan profile update settings.compiler=gcc hosacpp
 conan profile update settings.compiler.libcxx=libstdc++11 hosacpp
 conan profile update settings.compiler.version=10 hosacpp
 conan profile update settings.arch=x86_64 hosacpp
 conan profile update settings.os=Linux hosacpp
```

如果我们的依赖来自于默认存储库之外的其他存储库，我们可以使用`conan remote add <repo> <repo_url>`来添加它们。例如，您可能希望使用这个来配置您公司的存储库。

现在我们已经设置好了 Conan，让我们展示如何使用 Conan 获取我们的依赖，并将所有这些集成到我们的 CMake 脚本中。

### 指定 Conan 依赖

我们的项目依赖于 C++ REST SDK。为了告诉 Conan 这一点，我们需要创建一个名为`conanfile.txt`的文件。在我们的情况下，它将包含以下内容：

```cpp
 [requires]
 cpprestsdk/2.10.18

 [generators]
 CMakeDeps
```

您可以在这里指定尽可能多的依赖。每个依赖可以有一个固定的版本、一系列固定版本，或者像**latest**这样的标签。在`@`符号之后，您可以找到拥有该包的公司以及允许您选择特定变体的通道（通常是稳定和测试）。

**生成器**部分是您指定要使用的构建系统的地方。对于 CMake 项目，您应该使用`CMakeDeps`。您还可以生成许多其他生成器，包括用于生成编译器参数、CMake 工具链文件、Python 虚拟环境等等。

在我们的情况下，我们没有指定任何其他选项，但您可以轻松添加此部分，并为您的包和它们的依赖项配置变量。例如，要将我们的依赖项编译为静态库，我们可以编写以下内容：

```cpp
 [options]
 cpprestsdk:shared=False
```

一旦我们放置了`conanfile.txt`，让我们告诉 Conan 使用它。

### 安装 Conan 依赖

要在 CMake 代码中使用我们的 Conan 包，我们必须先安装它们。在 Conan 中，这意味着下载源代码并构建它们，或者下载预构建的二进制文件，并创建我们将在 CMake 中使用的配置文件。在我们创建了构建目录后，让 Conan 在我们之后处理这些，我们应该`cd`进入它，然后简单地运行以下命令：

```cpp
conan install path/to/directory/containing/conanfile.txt --build=missing -s build_type=Release -pr=hosacpp
```

默认情况下，Conan 希望下载所有依赖项作为预构建的二进制文件。如果服务器没有预构建它们，Conan 将构建它们，而不是像我们传递了`--build=missing`标志那样退出。我们告诉它抓取使用与我们配置文件中相同的编译器和环境构建的发布版本。您可以通过简单地使用`build_type`设置为其他 CMake 构建类型的另一个命令来为多个构建类型安装软件包。如果需要，这可以帮助您快速切换。如果要使用默认配置文件（Conan 可以自动检测到的配置文件），只需不传递`-pr`标志。

如果我们计划使用的 CMake 生成器没有在`conanfile.txt`中指定，我们可以将其附加到前面的命令中。例如，要使用`compiler_args`生成器，我们应该附加`--generator compiler_args`。稍后，您可以通过将`@conanbuildinfo.args`传递给编译器调用来使用它生成的内容。

### 使用 CMake 中的 Conan 目标

一旦 Conan 完成下载、构建和配置我们的依赖关系，我们需要告诉 CMake 使用它们。

如果您正在使用带有`CMakeDeps`生成器的 Conan，请确保指定`CMAKE_BUILD_TYPE`值。否则，CMake 将无法使用 Conan 配置的软件包。例如调用（从您运行 Conan 的相同目录）可能如下所示：

```cpp
cmake path/to/directory/containing/CMakeLists.txt -DCMAKE_BUILD_TYPE=Release
```

这样，我们将以发布模式构建我们的项目；我们必须使用 Conan 安装的类型之一。要找到我们的依赖关系，我们可以使用 CMake 的`find_package`：

```cpp
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
find_package(cpprestsdk CONFIG REQUIRED)
```

首先，我们将根构建目录添加到 CMake 将尝试在其中查找软件包配置文件的路径中。然后，我们找到 Conan 生成的软件包配置文件。

要将 Conan 定义的目标作为我们目标的依赖项传递，最好使用命名空间目标名称：

```cpp
 target_link_libraries(libcustomer PUBLIC cpprestsdk::cpprest)
```

这样，当找不到包时，我们将在 CMake 的配置期间收到错误。如果没有别名，我们在尝试链接时会收到错误。

现在我们已经按照我们想要的方式编译和链接了我们的目标，是时候进行测试了。

## 添加测试

CMake 有自己的测试驱动程序，名为`CTest`。很容易从您的`CMakeLists`中添加新的测试套件，无论是自己还是使用测试框架提供的许多集成。在本书的后面，我们将深入讨论测试，但首先让我们展示如何快速而干净地基于 GoogleTest 或 GTest 测试框架添加单元测试。

通常，要在 CMake 中定义您的测试，您会想要编写以下内容：

```cpp
 if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   include(CTest)
   if(BUILD_TESTING)
     add_subdirectory(test)
   endif()
 endif()
```

前面的片段将首先检查我们是否是正在构建的主项目。通常，您只想为您的项目运行测试，并且甚至不想为您使用的任何第三方组件构建测试。这就是为什么项目名称是`checked`。

如果我们要运行我们的测试，我们包括`CTest`模块。这将加载 CTest 提供的整个测试基础设施，定义其附加目标，并调用一个名为`enable_testing`的 CMake 函数，该函数将在其他事项中启用`BUILD_TESTING`标志。此标志是缓存的，因此您可以通过在生成构建系统时简单地传递`-DBUILD_TESTING=OFF`参数来禁用所有测试来构建您的项目。

所有这些缓存变量实际上都存储在名为`CMakeCache.txt`的文本文件中，位于您的构建目录中。随意修改那里的变量以更改 CMake 的操作；直到您删除该文件，它才不会覆盖那里的设置。您可以使用`ccmake`、`cmake-gui`，或者手动进行修改。

如果`BUILD_TESTING`为 true，我们只需处理我们测试目录中的`CMakeLists.txt`文件。可能看起来像这样：

```cpp
 include(FetchGTest)
 include(GoogleTest)

 add_subdirectory(customer)
```

第一个 include 调用了我们之前描述的提供 GTest 的脚本。在获取了 GTest 之后，我们当前的`CMakeLists.txt`通过调用`include(GoogleTest)`加载了 GoogleTest CMake 模块中定义的一些辅助函数。这将使我们更容易地将我们的测试集成到 CTest 中。最后，让我们告诉 CMake 进入一个包含一些测试的目录，通过调用`add_subdirectory(customer)`。

`test/customer/CMakeLists.txt`文件将简单地添加一个使用我们预定义的标志编译的带有测试的可执行文件，并链接到被测试的模块和 GTest。然后，我们调用 CTest 辅助函数来发现已定义的测试。所有这些只是四行 CMake 代码：

```cpp
 add_executable(unittests unit.cpp)
 target_compile_options(unittests PRIVATE ${BASE_COMPILE_FLAGS})
 target_link_libraries(unittests PRIVATE domifair::libcustomer gtest_main)
 gtest_discover_tests(unittests)
```

大功告成！

现在，您可以通过简单地转到`build`目录并调用以下命令来构建和执行您的测试：

```cpp
 cmake --build . --target unittests
 ctest # or cmake --build . --target test
```

您可以为 CTest 传递一个`-j`标志。它的工作方式与 Make 或 Ninja 调用相同-并行化测试执行。如果您想要一个更短的构建命令，只需运行您的构建系统，也就是通过调用`make`。

在脚本中，通常最好使用命令的较长形式；这将使您的脚本独立于所使用的构建系统。

一旦您的测试通过了，现在我们可以考虑向更广泛的受众提供它们。

# 重用优质代码

CMake 具有内置的实用程序，当涉及到分发构建结果时，这些实用程序可以走得更远。本节将描述安装和导出实用程序以及它们之间的区别。后续章节将向您展示如何使用 CPack 打包您的代码，以及如何使用 Conan 进行打包。

安装和导出对于微服务本身并不那么重要，但如果您要为其他人提供库以供重用，这将非常有用。

## 安装

如果您编写或使用过 Makefiles，您很可能在某个时候调用了`make install`，并看到项目的交付成果被安装在操作系统目录或您选择的其他目录中。如果您正在使用`make`与 CMake，使用本节的步骤将使您能够以相同的方式安装交付成果。如果没有，您仍然可以调用安装目标。除此之外，在这两种情况下，您将有一个简单的方法来利用 CPack 来创建基于您的安装命令的软件包。

如果您在 Linux 上，预设一些基于操作系统约定的安装目录可能是一个不错的主意，通过调用以下命令：

```cpp
include(GNUInstallDirs)
```

这将使安装程序使用由`bin`、`lib`和其他类似目录组成的目录结构。这些目录也可以使用一些 CMake 变量手动设置。

创建安装目标包括一些更多的步骤。首先，首要的是定义我们要安装的目标，这在我们的情况下将是以下内容：

```cpp
install(
   TARGETS libcustomer customer
   EXPORT CustomerTargets
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

这告诉 CMake 使用我们在本章前面定义的库和可执行文件作为`CustomerTargets`公开，使用我们之前设置的目录。

如果您计划将您的库的不同配置安装到不同的文件夹中，您可以使用前面命令的几次调用，就像这样：

```cpp
 install(TARGETS libcustomer customer
         CONFIGURATIONS Debug
         # destinations for other components go here...
         RUNTIME DESTINATION Debug/bin)
 install(TARGETS libcustomer customer
         CONFIGURATIONS Release
         # destinations for other components go here...
         RUNTIME DESTINATION Release/bin)
```

您可以注意到我们为可执行文件和库指定了目录，但没有包含文件。我们需要在另一个命令中提供它们，就像这样：

```cpp
 install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
         DESTINATION include)
```

这意味着顶层包含目录的内容将被安装在安装根目录下的包含目录中。第一个路径后面的斜杠修复了一些路径问题，所以请注意使用它。

所以，我们有了一组目标；现在我们需要生成一个文件，另一个 CMake 项目可以读取以了解我们的目标。可以通过以下方式完成：

```cpp
 install(
     EXPORT CustomerTargets
     FILE CustomerTargets.cmake
     NAMESPACE domifair::
     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Customer)
```

此命令将获取我们的目标集并创建一个`CustomerTargets.cmake`文件，其中将包含有关我们的目标及其要求的所有信息。我们的每个目标都将使用命名空间进行前缀处理；例如，`customer`将变成`domifair::customer`。生成的文件将安装在我们安装树中库文件夹的子目录中。

为了允许依赖项目使用 CMake 的`find_package`命令找到我们的目标，我们需要提供一个`CustomerConfig.cmake`文件。如果您的目标没有任何依赖项，您可以直接将前面的目标导出到该文件中，而不是`targets`文件。否则，您应该编写自己的配置文件，其中将包括前面的`targets`文件。

在我们的情况下，我们想要重用一些 CMake 变量，因此我们需要创建一个模板，并使用`configure_file`命令来填充它：

```cpp
  configure_file(${PROJECT_SOURCE_DIR}/cmake/CustomerConfig.cmake.in
                  CustomerConfig.cmake @ONLY)
```

我们的`CustomerConfig.cmake.in`文件将首先处理我们的依赖项：

```cpp
 include(CMakeFindDependencyMacro)

 find_dependency(cpprestsdk 2.10.18 REQUIRED)
```

`find_dependency`宏是`find_package`的包装器，旨在在配置文件中使用。尽管我们依赖 Conan 在`conanfile.txt`中定义的 C++ REST SDK 2.10.18，但在这里我们需要再次指定依赖关系。我们的软件包可以在另一台机器上使用，因此我们要求我们的依赖项也在那里安装。如果您想在目标机器上使用 Conan，可以按以下方式安装 C++ REST SDK：

```cpp
conan install cpprestsdk/2.10.18
```

处理完依赖项后，我们的配置文件模板将包括我们之前创建的`targets`文件：

```cpp
if(NOT TARGET domifair::@PROJECT_NAME@)
   include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")
endif()
```

当`configure_file`执行时，它将用项目中定义的`${VARIABLES}`的内容替换所有这些`@VARIABLES@`。这样，基于我们的`CustomerConfig.cmake.in`文件模板，CMake 将创建一个`CustomerConfig.cmake`文件。

在使用`find_package`查找依赖项时，通常需要指定要查找的软件包的版本。为了在我们的软件包中支持这一点，我们必须创建一个`CustomerConfigVersion.cmake`文件。CMake 为我们提供了一个辅助函数，可以为我们创建此文件。让我们按照以下方式使用它：

```cpp
 include(CMakePackageConfigHelpers)
 write_basic_package_version_file(
   CustomerConfigVersion.cmake
   VERSION ${PACKAGE_VERSION}
   COMPATIBILITY AnyNewerVersion)
```

`PACKAGE_VERSION`变量将根据我们在调用顶层`CMakeLists.txt`文件顶部的`project`时传递的`VERSION`参数进行填充。

`AnyNewerVersion COMPATIBILITY`表示如果我们的软件包比请求的版本更新或相同，它将被任何软件包搜索接受。其他选项包括`SameMajorVersion`，`SameMinorVersion`和`ExactVersion`。

一旦我们创建了我们的配置和配置版本文件，让我们告诉 CMake 它们应该与二进制文件和我们的目标文件一起安装：

```cpp
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CustomerConfig.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/CustomerConfigVersion.cmake
         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Customer)
```

我们应该安装的最后一件事是我们项目的许可证。我们将利用 CMake 的安装文件的命令将它们放在我们的文档目录中：

```cpp
install(
   FILES ${PROJECT_SOURCE_DIR}/LICENSE
   DESTINATION ${CMAKE_INSTALL_DOCDIR})
```

这就是您成功在操作系统根目录中创建安装目标所需了解的全部内容。您可能会问如何将软件包安装到另一个目录，比如仅供当前用户使用。要这样做，您需要设置`CMAKE_INSTALL_PREFIX`变量，例如，在生成构建系统时。

请注意，如果我们不安装到 Unix 树的根目录，我们将不得不为依赖项目提供安装目录的路径，例如通过设置`CMAKE_PREFIX_PATH`。

现在让我们看看另一种您可以重用刚刚构建的东西的方法。

## 导出

导出是一种将您在本地构建的软件包的信息添加到 CMake 的软件包注册表中的技术。当您希望您的目标可以直接从它们的构建目录中看到，即使没有安装时，这将非常有用。导出的常见用途是当您在开发机器上检出了几个项目并在本地构建它们时。

从您的`CMakeLists.txt`文件中添加对此机制的支持非常容易。在我们的情况下，可以这样做：

```cpp
export(
   TARGETS libcustomer customer
   NAMESPACE domifair::
   FILE CustomerTargets.cmake)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE domifair)
```

这样，CMake 将创建一个类似于*Installing*部分中的目标文件，定义我们在提供的命名空间中的库和可执行目标。从 CMake 3.15 开始，默认情况下禁用软件包注册表，因此我们需要通过设置适当的前置变量来启用它。然后，通过导出我们的软件包，我们可以将有关我们的目标的信息直接放入注册表中。

请注意，现在我们有一个没有匹配配置文件的`targets`文件。这意味着如果我们的目标依赖于任何外部库，它们必须在我们的软件包被找到之前被找到。在我们的情况下，调用必须按照以下方式排序：

```cpp
 find_package(cpprestsdk 2.10.18)
 find_package(domifair)
```

首先，我们找到 C++ REST SDK，然后再寻找依赖于它的软件包。这就是你需要知道的一切，就可以开始导出你的目标了。比安装它们要容易得多，不是吗？

现在让我们继续介绍第三种将您的目标暴露给外部世界的方法。

## 使用 CPack

在本节中，我们将描述如何使用 CMake 附带的打包工具 CPack。

CPack 允许您轻松创建各种格式的软件包，从 ZIP 和 TGZ 存档到 DEB 和 RPM 软件包，甚至安装向导，如 NSIS 或一些特定于 OS X 的软件包。一旦您安装逻辑就位，集成工具并不难。让我们展示如何使用 CPack 来打包我们的项目。

首先，我们需要指定 CPack 在创建软件包时将使用的变量：

```cpp
 set(CPACK_PACKAGE_VENDOR "Authors")
 set(CPACK_PACKAGE_CONTACT "author@example.com")
 set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
     "Library and app for the Customer microservice")
```

我们需要手动提供一些信息，但是一些变量可以根据我们在定义项目时指定的项目版本来填充。CPack 变量还有很多，您可以在本章末尾的*进一步阅读*部分的 CPack 链接中阅读所有这些变量。其中一些对所有软件包生成器都是通用的，而另一些则特定于其中的一些。例如，如果您计划使用安装程序，您可以设置以下两个：

`set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")`

`set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")`

一旦您设置了所有有趣的变量，就该选择 CPack 要使用的生成器了。让我们从在`CPACK_GENERATOR`中放置一些基本的生成器开始，这是 CPack 依赖的一个变量：

`list(APPEND CPACK_GENERATOR TGZ ZIP)`

这将导致 CPack 基于我们在本章前面定义的安装步骤生成这两种类型的存档。

你可以根据许多因素选择不同的软件包生成器，例如，正在运行的机器上可用的工具。例如，在 Windows 上构建时创建 Windows 安装程序，在 Linux 上构建时使用适当的工具安装 DEB 或 RPM 软件包。例如，如果你正在运行 Linux，你可以检查是否安装了`dpkg`，如果是，则创建 DEB 软件包：

```cpp
 if(UNIX)
   find_program(DPKG_PROGRAM dpkg)
   if(DPKG_PROGRAM)
     list(APPEND CPACK_GENERATOR DEB)
     set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS} libcpprest2.10 (>= 2.10.2-6)")
     set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
   else()
     message(STATUS "dpkg not found - won't be able to create DEB packages")
   endif()
```

我们使用了`CPACK_DEBIAN_PACKAGE_DEPENDS`变量，使 DEB 软件包要求首先安装 C++ REST SDK。

对于 RPM 软件包，您可以手动检查`rpmbuild`：

```cpp
 find_program(RPMBUILD_PROGRAM rpmbuild)
   if(RPMBUILD_PROGRAM)
     list(APPEND CPACK_GENERATOR RPM)
     set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES} cpprest >= 2.10.2-6")
   else()
     message(STATUS "rpmbuild not found - won't be able to create RPM packages")
   endif()
 endif()
```

很巧妙，对吧？

这些生成器提供了大量其他有用的变量，所以如果您需要比这里描述的基本需求更多的东西，请随时查看 CMake 的文档。

当涉及到变量时，最后一件事是，您也可以使用它们来避免意外打包不需要的文件。这可以通过以下方式完成：

`set(CPACK_SOURCE_IGNORE_FILES /.git /dist /.*build.* /\\\\.DS_Store)`

一旦我们把所有这些都放在位子上，我们可以从我们的 CMake 列表中包含 CPack 本身：

`include(CPack)`

记住，始终将此作为最后一步进行，因为 CMake 不会将您稍后使用的任何变量传播给 CPack。

要运行它，直接调用`cpack`或更长的形式，它还会检查是否需要首先重新构建任何内容：`cmake --build . --target package`。您可以轻松地通过`-G`标志覆盖生成器，例如，`-G DEB`只需构建 DEB 软件包，`-G WIX -C Release`打包一个发布的 MSI 可执行文件，或`-G DragNDrop`获取 DMG 安装程序。

现在让我们讨论一种更原始的构建软件包的方法。

# 使用 Conan 打包

我们已经展示了如何使用 Conan 安装我们的依赖项。现在，让我们深入了解如何创建我们自己的 Conan 软件包。

让我们在我们的项目中创建一个新的顶级目录，简单地命名为`conan`，在那里我们将使用这个工具打包所需的文件：一个用于构建我们的软件包的脚本和一个用于测试的环境。

## 创建 conanfile.py 脚本

所有 Conan 软件包所需的最重要的文件是`conanfile.py`。在我们的情况下，我们将使用 CMake 变量填写一些细节，所以我们将创建一个`conanfile.py.in`文件。我们将使用它来通过将以下内容添加到我们的`CMakeLists.txt`文件来创建前一个文件：

```cpp
configure_file(${PROJECT_SOURCE_DIR}/conan/conanfile.py.in
                ${CMAKE_CURRENT_BINARY_DIR}/conan/conanfile.py @ONLY)
```

我们的文件将以一些无聊的 Python 导入开始，例如 Conan 对于 CMake 项目所需的导入：

```cpp
 import os
 from conans import ConanFile, CMake
```

现在我们需要创建一个定义我们软件包的类：

```cpp
class CustomerConan(ConanFile):
     name = "customer"
     version = "@PROJECT_VERSION@"
     license = "MIT"
     author = "Authors"
     description = "Library and app for the Customer microservice"
     topics = ("Customer", "domifair")
```

首先，我们从我们的 CMake 代码中获取一堆通用变量。通常，描述将是一个多行字符串。主题对于在 JFrog 的 Artifactory 等网站上找到我们的库非常有用，并且可以告诉读者我们的软件包是关于什么的。现在让我们浏览其他变量：

```cpp
     homepage = "https://example.com"
     url = "https://github.com/PacktPublishing/Hands-On-Software-Architecture-with-Cpp/"
```

`homepage`应该指向项目的主页：文档、教程、常见问题解答等内容的所在地。另一方面，`url`是软件包存储库的位置。许多开源库将其代码放在一个存储库中，将打包代码放在另一个存储库中。一个常见情况是软件包由中央 Conan 软件包服务器构建。在这种情况下，`url`应该指向`https://github.com/conan-io/conan-center-index`。

接下来，我们现在可以指定我们的软件包是如何构建的：

```cpp
     settings = "os", "compiler", "build_type", "arch"
     options = {"shared": [True, False], "fPIC": [True, False]}
     default_options = {"shared": False, "fPIC": True}
     generators = "CMakeDeps"
     keep_imports = True  # useful for repackaging, e.g. of licenses
```

`settings`将确定软件包是否需要构建，还是可以下载已构建的版本。

`options`和`default_options`的值可以是任何你喜欢的。`shared`和`fPIC`是大多数软件包提供的两个选项，所以让我们遵循这个约定。

现在我们已经定义了我们的变量，让我们开始编写 Conan 将用于打包我们软件的方法。首先，我们指定我们的库，消费我们软件包的人应该链接到：

```cpp
    def package_info(self):
         self.cpp_info.libs = ["customer"]
```

`self.cpp_info`对象允许设置更多内容，但这是最低限度。请随意查看 Conan 文档中的其他属性。

接下来，让我们指定其他需要的软件包：

```cpp
    def requirements(self):
         self.requires.add('cpprestsdk/2.10.18')
```

这一次，我们直接从 Conan 中获取 C++ REST SDK，而不是指定 OS 的软件包管理器应该依赖哪些软件包。现在，让我们指定 CMake 应该如何（以及在哪里）生成我们的构建系统：

```cpp
    def _configure_cmake(self):
         cmake = CMake(self)
         cmake.configure(source_folder="@CMAKE_SOURCE_DIR@")
         return cmake
```

在我们的情况下，我们只需将其指向源目录。一旦配置了构建系统，我们将需要实际构建我们的项目：

```cpp
    def build(self):
         cmake = self._configure_cmake()
         cmake.build()
```

Conan 还支持非基于 CMake 的构建系统。构建我们的软件包之后，就是打包时间，这需要我们提供另一种方法：

```cpp
    def package(self):
         cmake = self._configure_cmake()
         cmake.install()
         self.copy("license*", ignore_case=True, keep_path=True)
```

请注意，我们正在使用相同的`_configure_cmake()`函数来构建和打包我们的项目。除了安装二进制文件之外，我们还指定许可证应该部署的位置。最后，让我们告诉 Conan 在安装我们的软件包时应该复制什么：

```cpp
    def imports(self):
         self.copy("license*", dst="licenses", folder=True, ignore_case=True)

         # Use the following for the cmake_multi generator on Windows and/or Mac OS to copy libs to the right directory.
         # Invoke Conan like so:
         #   conan install . -e CONAN_IMPORT_PATH=Release -g cmake_multi
         dest = os.getenv("CONAN_IMPORT_PATH", "bin")
         self.copy("*.dll", dst=dest, src="img/bin")
         self.copy("*.dylib*", dst=dest, src="img/lib")
```

前面的代码指定了在安装库时解压许可文件、库和可执行文件的位置。

现在我们知道如何构建一个 Conan 软件包，让我们也看看如何测试它是否按预期工作。

## 测试我们的 Conan 软件包

一旦 Conan 构建我们的包，它应该测试它是否被正确构建。为了做到这一点，让我们首先在我们的`conan`目录中创建一个`test_package`子目录。

它还将包含一个`conanfile.py`脚本，但这次是一个更短的脚本。它应该从以下内容开始：

```cpp
import os

from conans import ConanFile, CMake, tools

```

```cpp
class CustomerTestConan(ConanFile):
     settings = "os", "compiler", "build_type", "arch"
     generators = "CMakeDeps"
```

这里没有太多花哨的东西。现在，我们应该提供构建测试包的逻辑：

```cpp
    def build(self):
        cmake = CMake(self)
        # Current dir is "test_package/build/<build_id>" and 
        # CMakeLists.txt is in "test_package"
        cmake.configure()
        cmake.build()
```

我们将在一秒钟内编写我们的`CMakeLists.txt`文件。但首先，让我们写两件事：`imports`方法和`test`方法。`imports`方法可以编写如下：

```cpp
    def imports(self):
        self.copy("*.dll", dst="bin", src="img/bin")
        self.copy("*.dylib*", dst="bin", src="img/lib")
        self.copy('*.so*', dst='bin', src='lib')
```

然后我们有我们的包测试逻辑的核心 - `test`方法：

```cpp
    def test(self):
         if not tools.cross_building(self.settings):
             self.run(".%sexample" % os.sep)
```

我们只希望在为本机架构构建时运行它。否则，我们很可能无法运行已编译的可执行文件。

现在让我们定义我们的`CMakeLists.txt`文件：

```cpp
 cmake_minimum_required(VERSION 3.12)
 project(PackageTest CXX)

 list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

 find_package(customer CONFIG REQUIRED)

 add_executable(example example.cpp)
 target_link_libraries(example customer::customer)

 # CTest tests can be added here
```

就这么简单。我们链接到所有提供的 Conan 库（在我们的情况下，只有我们的 Customer 库）。

最后，让我们编写我们的`example.cpp`文件，其中包含足够的逻辑来检查包是否成功创建：

```cpp
 #include <customer/customer.h>

 int main() { responder{}.prepare_response("Conan"); }
```

在我们开始运行所有这些之前，我们需要在我们的 CMake 列表的主树中进行一些小的更改。现在让我们看看如何正确从我们的 CMake 文件中导出 Conan 目标。

## 将 Conan 打包代码添加到我们的 CMakeLists

记得我们在*重用优质代码*部分编写的安装逻辑吗？如果您依赖 Conan 进行打包，您可能不需要运行裸的 CMake 导出和安装逻辑。假设您只想在不使用 Conan 时导出和安装，您需要修改您的`CMakeLists`中的*安装*子部分，使其类似于以下内容：

```cpp
if(NOT CONAN_EXPORTED)
   install(
     EXPORT CustomerTargets
     FILE CustomerTargets.cmake
     NAMESPACE domifair::
     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Customer)

   configure_file(${PROJECT_SOURCE_DIR}/cmake/CustomerConfig.cmake.in
                  CustomerConfig.cmake @ONLY)

   include(CMakePackageConfigHelpers)
   write_basic_package_version_file(
     CustomerConfigVersion.cmake
     VERSION ${PACKAGE_VERSION}
     COMPATIBILITY AnyNewerVersion)

   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CustomerConfig.cmake
                 ${CMAKE_CURRENT_BINARY_DIR}/CustomerConfigVersion.cmake
           DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Customer)
 endif()

 install(
   FILES ${PROJECT_SOURCE_DIR}/LICENSE
   DESTINATION $<IF:$<BOOL:${CONAN_EXPORTED}>,licenses,${CMAKE_INSTALL_DOCDIR}>)
```

添加 if 语句和生成器表达式是为了获得干净的包，这就是我们需要做的一切。

最后一件事是让我们的生活变得更轻松 - 一个我们可以**构建**以创建 Conan 包的目标。我们可以定义如下：

```cpp
add_custom_target(
   conan
   COMMAND
     ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/conan/test_package/
     ${CMAKE_CURRENT_BINARY_DIR}/conan/test_package
   COMMAND conan create . customer/testing -s build_type=$<CONFIG>
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conan
   VERBATIM)
```

现在，当我们运行`cmake --build . --target conan`（或者如果我们使用该生成器并且想要一个简短的调用，则为`ninja conan`），CMake 将把我们的`test_package`目录复制到`build`文件夹中，构建我们的 Conan 包，并使用复制的文件进行测试。

全部完成！

这是冰山一角，关于创建 Conan 包的更多信息，请参考 Conan 的文档。您可以在*进一步阅读*部分找到链接。

# 总结

在本章中，您已经学到了很多关于构建和打包代码的知识。您现在能够编写更快构建的模板代码，知道如何选择工具来更快地编译代码（您将在下一章中了解更多关于工具的知识），并知道何时使用前向声明而不是`#include`指令。

除此之外，您现在可以使用现代 CMake 定义构建目标和测试套件，使用查找模块和`FetchContent`管理外部依赖项，以各种格式创建包和安装程序，最重要的是，使用 Conan 安装依赖项并创建自己的构件。

在下一章中，我们将看看如何编写易于测试的代码。持续集成和持续部署只有在有很好的测试覆盖率时才有用。没有全面测试的持续部署将使您更快地向生产中引入新的错误。当我们设计软件架构时，这不是我们的目标。

# 问题

1.  在 CMake 中安装和导出目标有什么区别？

1.  如何使您的模板代码编译更快？

1.  如何在 Conan 中使用多个编译器？

1.  如果您想使用预 C++11 GCC ABI 编译您的 Conan 依赖项，该怎么办？

1.  如何确保在 CMake 中强制使用特定的 C++标准？

1.  如何在 CMake 中构建文档并将其与您的 RPM 包一起发布？

# 进一步阅读

+   GCC 维基上的编译器书籍列表：[`gcc.gnu.org/wiki/ListOfCompilerBooks`](https://gcc.gnu.org/wiki/ListOfCompilerBooks)

+   基于类型的模板元编程并没有消亡，Odin Holmes 在 C++Now 2017 上的演讲：[`www.youtube.com/watch?v=EtU4RDCCsiU`](https://www.youtube.com/watch?v=EtU4RDCCsiU)

+   现代 CMake 在线书籍：[`cliutils.gitlab.io/modern-cmake`](https://cliutils.gitlab.io/modern-cmake)

+   Conan 文档：[`docs.conan.io/en/latest/`](https://docs.conan.io/en/latest/)

+   CMake 关于创建查找脚本的文档：[`cmake.org/cmake/help/v3.17/manual/cmake-developer.7.html?highlight=find#a-sample-find-module`](https://cmake.org/cmake/help/v3.17/manual/cmake-developer.7.html?highlight=find#a-sample-find-module)
