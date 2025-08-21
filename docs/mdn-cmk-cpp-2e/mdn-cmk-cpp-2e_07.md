# 第七章：使用 CMake 编译 C++ 源码

简单的编译场景通常由工具链的默认配置或者集成开发环境（**IDE**）提供。然而，在专业环境中，业务需求经常需要更高级的功能。可能需要更高的性能、更小的二进制文件、更强的可移植性、自动化测试或者更多的调试能力 – 不胜枚举。在一个一致、未来可靠的方式中管理所有这些很快就变成了一个复杂、纠结的问题（尤其是在需要支持多个平台时）。

编译的过程通常在 C++ 的书籍中解释得不够详细（像虚拟基类这样的深入主题似乎更有趣）。在本章中，我们将通过讨论编译的不同方面来解决这个问题：我们将了解编译的工作原理、它的内部阶段以及它们如何影响二进制输出。

之后，我们将专注于先决条件 – 我们将讨论可以用于微调编译过程的命令，如何从编译器要求特定功能，以及如何正确地告知编译器处理哪些输入文件。

然后，我们将专注于编译的第一阶段 – 预处理器。我们将提供包含头文件的路径，并学习如何通过预处理器定义从 CMake 和构建环境中插入变量。我们将涵盖最有趣的用例，并学习如何公开 CMake 变量以便从 C++ 代码中访问。

在此之后，我们将讨论优化器及其如何通过不同的标志影响性能。我们还将讨论优化的成本，特别是它如何影响生成的二进制文件的调试能力，以及如果不需要这些影响时应该怎么做。

最后，我们将解释如何通过使用预编译头文件和统一构建来管理编译过程，以减少编译时间。我们将学习如何调试构建过程并找出可能存在的任何错误。

在本章中，我们将涵盖以下主要主题：

+   编译的基础知识

+   预处理器的配置

+   配置优化器

+   管理编译过程

# 技术要求

你可以在 GitHub 上找到本章节中存在的代码文件，链接在[`github.com/PacktPublishing/Modern-CMake-for-Cpp-2E/tree/main/examples/ch07`](https://github.com/PacktPublishing/Modern-CMake-for-Cpp-2E/tree/main/examples/ch07)。

要构建本书提供的示例，请始终使用推荐的命令：

```cpp
cmake -B <build tree> -S <source tree>
cmake --build <build tree> 
```

请确保用适当的路径替换 `<build tree>` 和 `<source tree>` 占位符。作为提醒：**build tree** 是指目标/输出目录的路径，**source tree** 是指源代码所在的路径。

# 编译的基础知识

编译可以大致描述为将用高级编程语言编写的指令转换为低级机器码的过程。这使我们能够使用诸如类和对象等抽象概念来创建应用程序，而不必费力处理处理器特定的汇编语言。我们不需要直接操作 CPU 寄存器，考虑短跳或长跳，或管理堆栈帧。编译型语言更具表现力、可读性和安全性，并鼓励编写可维护的代码，同时尽可能提供最佳性能。

在 C++中，我们使用静态编译——这意味着整个程序必须在执行之前先被翻译成本地代码。这与像 Java 或 Python 这样的语言不同，后者每次用户运行程序时都会即时解释和编译程序。每种方法都有其独特的优点。C++旨在提供多种高级工具，同时提供本地性能。C++编译器可以为几乎所有架构生成一个自包含的应用程序。

创建并运行 C++程序涉及多个步骤：

1.  **设计你的应用程序**：这包括规划应用程序的功能、结构和行为。一旦设计完成，按照代码可读性和可维护性的最佳实践，仔细编写源代码。

1.  **编译单个.cpp 实现文件，也称为翻译单元，成目标文件**：这一步涉及将您编写的高级语言代码转换为低级机器码。

1.  **将链接** **目标文件合并成单个可执行文件**：在此步骤中，所有其他依赖项，包括动态库和静态库，也会被链接。这一过程创建了一个可以在预定平台上运行的可执行文件。

要运行程序，**操作系统**（**OS**）将使用一种名为**加载器**的工具，将程序的机器码和所有所需的动态库映射到虚拟内存中。加载器随后读取程序头部，以确定执行应从哪里开始，并开始运行指令。

在这个阶段，程序的启动代码开始发挥作用。系统 C 库提供的一个特殊函数`_start`被调用。`_start`函数收集命令行参数和环境变量，启动线程，初始化静态符号，并注册清理回调函数。只有在此之后，它才会调用`main()`，这是程序员填入自己代码的函数。

如你所见，在幕后发生了大量工作。本章重点讲解早期列表中的第二步。通过考虑整体情况，我们可以更好地理解潜在问题可能来自哪里。尽管软件开发中的复杂性看起来似乎无法逾越，但开发中并不存在“魔法”。一切都有解释和原因。我们需要理解，由于我们如何编译程序，程序在运行时可能会出现问题，即使编译步骤本身看似成功。编译器不可能在其操作过程中检查所有边界情况。因此，让我们深入了解当编译器执行其工作时，实际发生了什么。

## 编译如何工作

如前所述，编译是将高级语言翻译成低级语言的过程。具体来说，这涉及生成机器代码，这些机器代码是特定处理器可以直接执行的指令，格式为平台独有的二进制**目标文件**。在 Linux 上，最常用的格式是**可执行与可链接格式**（**ELF**）。Windows 使用 PE/COFF 格式规范，而在 macOS 上，我们会遇到 Mach 对象（Mach-O 格式）。

**目标文件**是单个源文件的直接翻译。每个文件必须单独编译，然后由链接器将其合并成一个可执行文件或库。这个模块化过程在修改代码时可以显著节省时间，因为只有程序员更新的文件需要重新编译。

编译器必须执行以下阶段才能创建**目标文件**：

+   预处理

+   语言分析

+   汇编

+   优化

+   代码生成

让我们更详细地解释一下它们。

**预处理**，虽然大多数编译器自动调用，但被视为实际编译之前的准备步骤。它的作用是对源代码进行基本的操作；执行`#include`指令、通过`#define`指令和`-D`标志替换标识符为已定义的值、调用简单的宏，并根据`#if`、`#elif`和`#endif`指令有条件地包含或排除部分代码。预处理器对实际的 C++代码毫不知情。从本质上讲，它充当一个高级的查找和替换工具。

然而，预处理器在构建高级程序中的作用至关重要。将代码分割成多个部分并在多个翻译单元之间共享声明的能力是代码可重用性的基础。

接下来是**语言分析**，在这一阶段，编译器进行更复杂的操作。它逐字符扫描预处理后的文件（现在已包含由预处理器插入的所有头文件）。通过一种称为词法分析的过程，它将字符分组为有意义的记号——这些记号可能是关键字、运算符、变量名等。

然后，令牌会被组装成链并进行检查，以验证它们的顺序和存在是否符合 C++的语法规则——这一过程称为语法分析或解析。通常，这是生成大多数错误信息的阶段，因为它识别了语法问题。

最后，编译器进行语义分析。在这个阶段，编译器检查文件中的语句是否在逻辑上是合理的。例如，它确保所有类型正确性检查都已满足（你不能将整数赋值给字符串变量）。这一分析确保程序在编程语言的规则范围内是合乎逻辑的。

**汇编**阶段本质上是将这些令牌翻译成基于平台可用指令集的 CPU 特定指令。有些编译器实际上生成汇出文件，然后传递给专门的汇编程序。该程序生成 CPU 可以执行的机器代码。其他编译器直接在内存中生成机器代码。通常，这些编译器还提供生成可供人类阅读的汇编代码的选项。然而，尽管这些代码是可以阅读的，但并不意味着它们容易理解或值得这么做。

**优化**并不仅仅局限于编译过程中的某一个步骤，而是在每个阶段逐步进行的。然而，在初步汇编生成后，有一个独立的阶段，专注于最小化寄存器使用并消除冗余代码。

一个有趣且值得注意的优化技术是内联展开或*内联*。在这个过程中，编译器有效地将函数体“剪切”并将其“粘贴”到函数调用的位置。C++标准并没有明确定义何时进行这种操作——它是依赖于实现的。内联展开可以提高执行速度并减少内存使用，但它也会对调试产生重大影响，因为执行的代码不再与源代码中的原始行对应。

**代码生成**阶段涉及将优化后的机器代码写入一个与目标平台规范对齐的*目标文件*中。然而，这个*目标文件*尚未准备好执行——它需要传递给链中的下一个工具：链接器。链接器的工作是适当地重新定位我们的*目标文件*的各个部分，并解决对外部符号的引用，有效地为文件的执行做准备。此步骤标志着**美国信息交换标准代码**（**ASCII**）源代码转化为*二进制可执行文件*，这些文件可以直接由 CPU 处理。

这些阶段每个都非常重要，并且可以配置以满足我们的特定需求。让我们看看如何使用 CMake 来管理这个过程。

## 初始配置

CMake 提供了多个命令，可以影响编译过程中的每个阶段。

+   `target_compile_features()`: 这需要一个具有特定功能的编译器来编译此目标。

+   `target_sources()`: 该命令将源文件添加到已定义的目标中。

+   `target_include_directories()`: 该命令设置预处理器 *包含路径*。

+   `target_compile_definitions()`: 该命令设置预处理器定义。

+   `target_compile_options()`: 该命令设置编译器特定的命令行选项。

+   `target_precompile_headers()`: 该命令设置外部头文件以便进行预编译优化。

每个命令接受类似格式的参数：

```cpp
target_...(<target name> <INTERFACE|PUBLIC|PRIVATE> <arguments>) 
```

这意味着使用该命令设置的属性通过传递的使用要求传播，如 *第五章*，*与目标一起工作* 中的 *什么是传递使用要求？* 部分所讨论的，可以用于可执行文件和库。另外，值得注意的是，所有这些命令都支持生成器表达式。

### 需要从编译器中获取特定的功能

如 *第四章*，*设置你的第一个 CMake 项目* 中的 *检查支持的编译器功能* 部分所述，预见问题并确保在出现错误时给用户清晰的信息至关重要——例如，当一个可用的编译器 X 不提供所需的功能 Y 时。这种方法比让用户解读不兼容工具链所产生的错误更为友好。我们不希望用户将不兼容问题归咎于我们的代码，而是他们过时的环境。

你可以使用以下命令来指定目标构建所需的所有功能：

```cpp
target_compile_features(<target> <PRIVATE|PUBLIC|INTERFACE>
                        <feature> [...]) 
```

CMake 支持以下 `compiler_ids` 的 C++ 标准和编译器功能：

+   `AppleClang`: 用于 Xcode 版本 4.4+ 的 Apple Clang

+   `Clang`: Clang 编译器版本 2.9+

+   `GNU`: GNU 编译器版本 4.4+

+   `MSVC`: Microsoft Visual Studio 版本 2010+

+   `SunPro`: Oracle Solaris Studio 版本 12.4+

+   `Intel`: Intel 编译器版本 12.1+

CMake 支持超过 60 个功能，你可以在官方文档中找到完整列表，详见解释 `CMAKE_CXX_KNOWN_FEATURES` 变量的页面。不过，除非你在寻找某个非常具体的功能，否则我建议选择一个表示一般 C++ 标准的高级元功能：

+   `cxx_std_14`

+   `cxx_std_17`

+   `cxx_std_20`

+   `cxx_std_23`

+   `cxx_std_26`

查看以下示例：

```cpp
target_compile_features(my_target PUBLIC cxx_std_26) 
```

这基本上等同于在 *第四章*，*设置你的第一个 CMake 项目* 中引入的 `set(CMAKE_CXX_STANDARD 26)` 和 `set(CMAKE_CXX_STANDARD_REQUIRED ON)`。然而，区别在于 `target_compile_features()` 是按目标处理的，而不是为整个项目全局处理，这在你需要为项目中的所有目标添加时可能会显得麻烦。

在官方手册中查看 CMake 的 *支持的编译器* 的更多详细信息（请参见 *进一步阅读* 部分获取网址）。

## 管理目标的源文件

我们已经知道如何告诉 CMake 哪些源文件构成一个目标，无论它是可执行文件还是库。我们通过在使用`add_executable()`或`add_library()`命令时提供一个文件列表来做到这一点。

随着您的解决方案扩展，每个目标的文件列表也在增长。这可能会导致一些相当冗长的`add_...()`命令。我们该如何处理呢？一种诱人的方法可能是使用`file()`命令的`GLOB`模式，这样可以从子目录中收集所有文件并将它们存储在一个变量中。我们可以将其作为参数传递给目标声明，再也不需要关心文件列表了：

```cpp
file(GLOB helloworld_SRC "*.h" "*.cpp")
add_executable(helloworld ${helloworld_SRC}) 
```

然而，这种方法并不推荐。让我们理解一下为什么。CMake 根据列表文件中的更改生成构建系统。所以，如果没有检测到任何更改，您的构建可能会在没有任何警告的情况下失败（这是开发者的噩梦）。此外，省略目标声明中的所有源代码可能会破坏像 CLion 这样的 IDE 中的代码检查，因为它知道如何解析某些 CMake 命令来理解您的项目。

在目标声明中使用变量是不建议的，原因是：它会创建一个间接层，导致开发者在阅读项目时必须解包目标定义。为了遵循这个建议，我们又面临另一个问题：如何有条件地添加源文件？这在处理特定平台的实现文件时是一个常见场景，例如`gui_linux.cpp`和`gui_windows.cpp`。

`target_sources()`命令允许我们将源文件附加到之前创建的目标：

**ch07/01-sources/CMakeLists.txt**

```cpp
add_executable(main main.cpp)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(main PRIVATE gui_linux.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_sources(main PRIVATE gui_windows.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_sources(main PRIVATE gui_macos.cpp)
else()
  message(FATAL_ERROR "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME} not supported.")
endif() 
```

这样，每个平台都会得到一组兼容的文件。这很好，但如果源文件列表很长怎么办？嗯，我们只能接受某些事情尚不完美，并继续手动添加它们。如果您正在与一个非常长的列表作斗争，那么您很可能在项目结构上做错了什么：也许可以考虑将源文件划分为库。

现在我们已经涵盖了编译的基本知识，让我们深入了解第一步——预处理。像所有计算机科学的事物一样，细节决定成败。

# 配置预处理器

预处理器在构建过程中扮演着巨大的角色。也许这有点令人惊讶，因为它的功能看起来相当直接和有限。在接下来的章节中，我们将介绍如何提供包含文件的路径和使用预处理器定义。我们还将解释如何使用 CMake 配置包含的头文件。

## 提供包含文件的路径

预处理器的最基本功能是能够使用`#include`指令包含`.h`和`.hpp`头文件，这有两种形式：

+   尖括号形式：`#include <path-spec>`

+   引号形式：`#include "path-spec"`

如我们所知，预处理器将把这些指令替换为 `path-spec` 中指定文件的内容。查找这些文件可能会很有挑战性。应该搜索哪些目录，以及按什么顺序搜索？不幸的是，C++ 标准并未明确规定这一点。我们必须查看所使用编译器的手册。

通常，尖括号形式将检查标准的 *包含目录*，这些目录包括系统中存储标准 C++ 库和标准 C 库头文件的目录。

引号形式首先会在当前文件的目录中搜索被包含的文件，然后再检查尖括号形式的目录。

CMake 提供了一条命令来操作搜索包含文件的路径：

```cpp
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
                           <INTERFACE|PUBLIC|PRIVATE> [item1...]
                          [<INTERFACE|PUBLIC|PRIVATE> [item2...]
...]) 
```

这使我们能够添加希望编译器扫描的自定义路径。CMake 将在生成的构建系统中将它们添加到编译器调用中，并为特定编译器提供适当的标志（通常是 `-I`）。

`target_include_directories()` 命令通过在目标的 `INCLUDE_DIRECTORIES` 属性中附加或预附加目录来修改它，具体取决于是否使用 `AFTER` 或 `BEFORE` 关键字。然而，是否在默认目录之前或之后检查这些目录，仍然由编译器决定（通常是在之前）。

`SYSTEM` 关键字表示编译器应将给定的目录视为标准系统目录（用于尖括号形式）。对于许多编译器，这些目录是通过 `-isystem` 标志传递的。

## 预处理器定义

回想一下之前讨论的编译阶段中的预处理器 `#define` 和 `#if`、`#elif` 以及 `#endif` 指令。让我们看一下以下示例：

**ch07/02-definitions/definitions.cpp**

```cpp
#include <iostream>
int main() {
#if defined(ABC)
    std::cout << "ABC is defined!" << std::endl;
#endif
#if (DEF > 2*4-3)
    std::cout << "DEF is greater than 5!" << std::endl;
#endif
} 
```

如此一来，这个例子没有任何效果，因为 `ABC` 和 `DEF` 都没有被定义（在这个例子中，`DEF` 会默认为 `0`）。我们可以通过在代码的顶部添加两行来轻松改变这一点：

```cpp
#define ABC
#define DEF 8 
```

编译并执行此代码后，我们可以在控制台中看到两条消息：

```cpp
ABC is defined!
DEF is greater than 5! 
```

这看起来似乎足够简单，但如果我们想根据外部因素（如操作系统、架构或其他因素）来条件化这些部分怎么办？好消息是，你可以将值从 CMake 传递给 C++ 编译器，而且这并不复杂。

`target_compile_definitions()` 命令就足够了：

**ch07/02-definitions/CMakeLists.txt**

```cpp
set(VAR 8)
add_executable(defined definitions.cpp)
target_compile_definitions(defined PRIVATE ABC "DEF=${VAR}") 
```

前面的代码将与两个 `#define` 语句的行为完全相同，但我们有灵活性使用 CMake 的变量和生成器表达式，并且可以将命令放入条件块中。

传统上，这些定义通过 `-D` 标志传递给编译器（例如，`-DFOO=1`），有些程序员仍然在这个命令中继续使用这个标志：

```cpp
target_compile_definitions(hello PRIVATE -DFOO) 
```

CMake 能识别这一点，并会自动移除任何前导的 `-D` 标志。它还会忽略空字符串，因此以下命令是完全有效的：

```cpp
target_compile_definitions(hello PRIVATE -D FOO) 
```

在这种情况下，`-D` 是一个独立的参数，移除后会变成空字符串，并随后被忽略，从而确保行为正确。

### 避免在单元测试中访问私有类字段

一些在线资源建议结合使用特定的 `-D` 定义与 `#ifdef/ifndef` 指令，用于单元测试。此方法最直接的应用是将 `public` 访问控制符包含在条件包含中，当 `UNIT_TEST` 被定义时，使所有字段都变为公共（默认情况下，类字段是私有的）：

```cpp
class X {
#ifdef UNIT_TEST
  public:
#endif
  int x_;
} 
```

尽管这种技术提供了便利（允许测试直接访问私有成员），但它并不会产生干净的代码。理想情况下，单元测试应该专注于验证公共接口内方法的功能，将底层实现视为黑盒。因此，我建议仅在不得已时使用这种方法。

### 使用 Git 提交跟踪已编译版本

让我们思考一些可以从了解环境或文件系统细节中受益的用例。一个典型的例子可能是在专业环境中，传递用于构建二进制文件的修订或提交 `SHA`。可以通过以下方式实现：

**ch07/03-git/CMakeLists.txt**

```cpp
add_executable(print_commit print_commit.cpp)
execute_process(COMMAND git log -1 --pretty=format:%h
                OUTPUT_VARIABLE SHA)
target_compile_definitions(print_commit
                           PRIVATE "SHA=${SHA}") 
```

然后，SHA 可以在我们的应用中按如下方式使用：

**ch07/03-git/print_commit.cpp**

```cpp
#include <iostream>
// special macros to convert definitions into c-strings:
#define str(s) #s
#define xstr(s) str(s)
int main()
{
#if defined(SHA)
    std::cout << "GIT commit: " << xstr(SHA) << std::endl;
#endif
} 
```

当然，前面的代码要求用户安装并在其 `PATH` 中能够访问 Git。这个功能在生产服务器上运行的程序是通过持续集成/部署流水线构建的情况下特别有用。如果我们的软件出现问题，可以迅速检查到底是哪个 Git 提交被用来构建有问题的产品。

跟踪确切的提交对于调试非常有帮助。将单个变量传递给 C++ 代码非常简单，但当需要将几十个变量传递给头文件时，我们该如何处理呢？

## 配置头文件

通过 `target_compile_definitions()` 传递定义可能会变得繁琐，尤其是当变量众多时。难道提供一个带有占位符的头文件，引用这些变量，并让 CMake 来填充它们，不更简单吗？绝对可以！

CMake 的 `configure_file(<input> <output>)` 命令允许你从模板生成新文件，示例如下：

**ch07/04-configure/configure.h.in**

```cpp
#cmakedefine FOO_ENABLE
#cmakedefine FOO_STRING1 "@FOO_STRING1@"
#cmakedefine FOO_STRING2 "${FOO_STRING2}"
#cmakedefine FOO_UNDEFINED "@FOO_UNDEFINED@" 
```

你可以按如下方式使用此命令：

**ch07/04-configure/CMakeLists.txt**

```cpp
add_executable(configure configure.cpp)
set(FOO_ENABLE ON)
set(FOO_STRING1 "abc")
set(FOO_STRING2 "def")
configure_file(configure.h.in configured/configure.h)
target_include_directories(configure PRIVATE
                           ${CMAKE_CURRENT_BINARY_DIR}) 
```

CMake 然后会生成一个类似以下的输出文件：

**ch07/04-configure/<build_tree>/configured/configure.h**

```cpp
#define FOO_ENABLE
#define FOO_STRING1 "abc"
#define FOO_STRING2 "def"
/* #undef FOO_UNDEFINED */ 
```

如你所见，`@VAR@` 和 `${VAR}` 变量占位符已被 CMake 列表文件中的值替换。此外，`#cmakedefine` 被已定义变量的 `#define` 和未定义变量的 `/* #undef VAR */` 所取代。如果你需要显式的 `#define 1` 或 `#define 0` 用于 `#if` 块，请改用 `#cmakedefine01`。

你可以通过简单地在实现文件中包含这个配置好的头文件，将其集成到你的应用程序中：

**ch07/04-configure/configure.cpp**

```cpp
#include <iostream>
#include "configured/configure.h"
// special macros to convert definitions into c-strings:
#define str(s) #s
#define xstr(s) str(s)
using namespace std;
int main()
{
#ifdef FOO_ENABLE
  cout << "FOO_ENABLE: ON" << endl;
#endif
  cout << "FOO_STRING1: " << xstr(FOO_STRING1) << endl;
  cout << "FOO_STRING2: " << xstr(FOO_STRING2) << endl;
  cout << "FOO_UNDEFINED: " << xstr(FOO_UNDEFINED) << endl;
} 
```

通过将二叉树添加到我们的*包含路径*中，并使用 `target_include_directories()` 命令，我们可以编译示例，并接收来自 CMake 的输出：

```cpp
FOO_ENABLE: ON
FOO_STRING1: "abc"
FOO_STRING2: "def"
FOO_UNDEFINED: FOO_UNDEFINED 
```

`configure_file()` 命令还包括一系列格式化和文件权限选项，由于篇幅限制，我们不会在此深入探讨。如果你感兴趣，可以参考在线文档获取更多细节（请参阅本章的 *进一步阅读* 部分）。

在准备好完整的头文件和源文件编译后，让我们讨论在后续步骤中输出代码是如何形成的。尽管我们无法直接影响语言分析或汇编（因为这些步骤遵循严格的标准），但我们可以调整优化器的配置。让我们来探索一下这种配置如何影响最终结果。

# 配置优化器

优化器将分析前一阶段的输出，并使用多种策略，程序员通常不会直接使用这些策略，因为它们不符合干净代码原则。但这没关系——优化器的核心作用是提高代码性能，追求低 CPU 使用率、最小化寄存器使用和减少内存占用。当优化器遍历源代码时，它会将代码重构为几乎无法辨认的形式，专门为目标 CPU 量身定制。

优化器不仅会决定哪些函数可以删除或压缩，它还会重新排列代码，甚至大规模复制代码！如果它能够确定某些代码行是多余的，它会将这些行从重要函数中间删除（你甚至不会注意到）。它会回收内存，让多个变量在不同时间占用相同的位置。它甚至可以将你的控制结构重塑成完全不同的形式，如果这样做能节省几次 CPU 周期的话。

如果程序员手动将上述技术应用到源代码中，它将把代码变成一团糟，既难写又难理解。然而，当编译器应用这些技术时，它们是有益的，因为编译器严格遵循给定的指令。优化器是一只无情的野兽，服务的唯一目的就是加速执行速度，无论输出变得多么扭曲。这样的输出可能包含一些调试信息，如果我们在测试环境中运行它，或者可能不包含调试信息，以防止未授权的人篡改。

每个编译器都有自己独特的技巧，这与它支持的平台和所遵循的哲学一致。我们将查看 GNU GCC 和 LLVM Clang 中最常见的一些，以便了解哪些是实际可行的。

事情是这样的——许多编译器默认不会启用任何优化（包括 GCC）。在某些情况下这样没问题，但在其他情况下就不行了。为什么要慢呢，当你可以更快？为了解决这个问题，我们可以使用 `target_compile_options()` 命令，明确表达我们对编译器的期望。

该命令的语法与本章中的其他命令类似：

```cpp
target_compile_options(<target> [BEFORE]
                       <INTERFACE|PUBLIC|PRIVATE> [items1...]
                      [<INTERFACE|PUBLIC|PRIVATE> [items2...]
...]) 
```

我们提供命令行选项，在构建目标时使用，并且还指定了传播关键字。当执行时，CMake 会将给定的选项附加到目标的适当 `COMPILE_OPTIONS` 变量中。如果我们希望将它们放在前面，可以使用可选的 `BEFORE` 关键字。在某些场景中，顺序可能很重要，因此能够选择顺序是有益的。

请注意，`target_compile_options()` 是一个通用命令。它也可以用于为编译器提供其他参数，例如 `-D` 定义，CMake 还提供了 `target_compile_definition()` 命令。建议尽可能使用最专业的 CMake 命令，因为它们在所有支持的编译器中保证以相同的方式工作。

现在是讨论细节的时候了。接下来的部分将介绍可以在大多数编译器中启用的各种优化。

## 一般级别

优化器的所有不同行为可以通过特定的标志来深入配置，这些标志我们可以作为 *编译选项* 传递。如果我们只是想要一个在大多数情况下都能很好工作的最佳解决方案，该怎么办？我们可以选择一个通用的解决方案——一个优化级别说明符。

大多数编译器提供四个基本的优化级别，从 `0` 到 `3`。我们通过 `-O<level>` 选项来指定它们。`-O0` 意味着 *没有优化*，通常这是编译器的默认级别。另一方面，`-O2` 被认为是 *完全优化*，它生成高度优化的代码，但代价是最慢的编译时间。

还有一个中间的 `-O1` 级别，这个级别（根据你的需求）可能是一个不错的折衷——它启用了合理的优化机制，同时不会过多地减慢编译速度。

最后，我们可以选择 `-O3`，这是*完全优化*，类似于 `-O2`，但采用更激进的子程序内联和循环向量化方法。

还有一些优化的变体，它们优化的是生成文件的大小（不一定是速度）——`-Os`。有一种超激进的优化 `-Ofast`，它是 `-O3` 优化，但不严格遵守 C++ 标准。最明显的区别是使用了 `-ffast-math` 和 `-ffinite-math` 标志，这意味着如果你的程序涉及精确计算（大多数程序都是），你可能希望避免使用它。

CMake 知道并非所有编译器都是一样的，因此它通过为编译器提供一些默认标志来标准化开发者的体验。这些标志存储在系统范围内（而非特定目标）的变量中，用于所使用的语言（`CXX` 用于 C++）和构建配置（`DEBUG` 或 `RELEASE`）：

+   `CMAKE_CXX_FLAGS_DEBUG` 等于 `-g`

+   `CMAKE_CXX_FLAGS_RELEASE` 等于 `-O3 -DNDEBUG`

正如你所看到的，调试配置不会启用任何优化，而发布配置则直接使用 `O3`。如果你愿意，你可以通过 `set()` 命令直接更改它们，或者只需添加目标编译选项，这将覆盖默认行为。另两个标志（`-g,` `-DNDEBUG`）与调试相关——我们将在本章的 *为调试器提供信息* 部分讨论它们。

像 `CMAKE_<LANG>_FLAGS_<CONFIG>` 这样的变量是全局的——它们适用于所有目标。建议通过属性和命令（如 `target_compile_options()`）来配置目标，而不是依赖于全局变量。这样，你可以更细粒度地控制你的目标。

通过选择优化级别 `-O<level>`，我们间接设置了一长串标志，每个标志控制着特定的优化行为。然后，我们可以通过追加更多标志来微调优化，如下所示：

+   使用 `-f` 选项启用它们：`-finline-functions`。

+   使用 `-fno` 选项禁用它们：`-fno-inline-functions`。

这些标志中的一些值得更好地理解，因为它们会影响你的程序的运行方式以及你如何调试它。让我们来看看。

## 函数内联

正如你可能记得的那样，编译器可以通过*在类的*声明*块中定义*一个函数，或通过显式使用 `inline` 关键字来鼓励内联一些函数：

```cpp
struct X {
  void im_inlined(){ cout << "hi\n"; };
  void me_too();
};
**inline** void X::me_too() { cout << "bye\n"; }; 
```

内联一个函数的决定最终由编译器做出。如果启用了内联，并且该函数仅在一个位置使用（或是一个在少数地方使用的相对较小的函数），那么内联很可能会发生。

函数内联是一种有趣的优化技术。它通过将目标函数的代码提取出来并嵌入到所有调用该函数的位置来工作。这个过程替换了原始的调用，并节省了宝贵的 CPU 周期。

让我们考虑以下使用我们刚刚定义的类的示例：

```cpp
int main() {
  X x;
  x.im_inlined();
  x.me_too();
  return 0;
} 
```

如果没有内联，代码将在`main()`框架中执行，直到方法调用为止。然后，它会为`im_inlined()`创建一个新框架，在一个单独的作用域中执行，并返回到`main()`框架。`me_too()`方法也会发生同样的情况。

然而，当发生内联时，编译器会替换调用，类似这样：

```cpp
int main() {
  X x;
  cout << "hi\n";
  cout << "bye\n";
  return 0;
} 
```

这并不是精确的表示，因为内联发生在汇编或机器代码的层面（而非源代码层面），但它提供了一个大致的概念。

编译器使用内联来节省时间。它跳过了创建和销毁新调用框架的过程，避免了查找下一个要执行的指令地址（并返回）的需求，并且增强了指令缓存，因为它们彼此非常接近。

然而，内联确实带来了一些显著的副作用。如果一个函数被多次使用，它必须复制到所有调用位置，从而导致文件大小增大和内存使用增加。尽管今天这可能不像以前那么关键，但它仍然相关，尤其是在为低端设备（内存有限）开发软件时。

此外，内联对调试产生了重大影响。内联代码不再出现在原始的行号位置，这使得追踪变得更加困难，有时甚至变得不可能。这就是为什么在内联的函数上设置调试断点时，永远不会被触发（即使代码仍然以某种方式被执行）。为了解决这个问题，你需要在调试版本中禁用内联（这意味着无法测试完全相同的发布版本）。

我们可以通过为目标指定`-O0`（o-zero）级别，或直接修改负责内联的标志来实现：

+   `-finline-functions-called-once`：仅适用于 GCC。

+   `-finline-functions`：适用于 Clang 和 GCC。

+   `-finline-hint-functions`：仅适用于 Clang。

内联可以通过`-fno-inline-...`显式禁用，但是，若要了解详细信息，建议查阅特定编译器版本的文档。

## 循环展开

循环展开，也称为循环解开，是一种优化技术。该策略旨在将循环转换为一系列实现相同结果的语句。因此，这种方法将程序的小体积换成了执行速度，因为它消除了循环控制指令、指针运算和循环结束检查。

请看以下示例：

```cpp
void func() {
  for(int i = 0; i < 3; i++)
    cout << "hello\n";
} 
```

上述代码将被转换为类似如下内容：

```cpp
void func() {
    cout << "hello\n";
    cout << "hello\n";
    cout << "hello\n";
} 
```

结果将是一样的，但我们不再需要分配 `i` 变量、递增它或将其与值 `3` 比较三次。如果在程序的生命周期内多次调用 `func()`，即使是展开如此短小的函数，也会产生显著的差异。

然而，理解两个限制因素是很重要的。首先，循环展开只有在编译器知道或能够准确估计迭代次数时才有效。其次，循环展开可能会对现代 CPU 产生不良影响，因为增加的代码大小可能会妨碍有效的缓存。

每个编译器提供的此标志的版本略有不同：

+   `-floop-unroll`：这是用于 GCC 的选项。

+   `-funroll-loops`：这是用于 Clang 的选项。

如果你不确定，广泛测试此标志是否影响你特定的程序，并显式地启用或禁用它。请注意，在 GCC 中，它在 `-O3` 下隐式启用，作为隐式启用的 `-floop-unroll-and-jam` 标志的一部分。

## 循环向量化

被称为**单指令多数据**（**SIMD**）的机制是在 1960 年代初期开发的，目的是实现并行性。顾名思义，它旨在同时对多个数据执行相同的操作。让我们通过以下示例来实际了解这一点：

```cpp
int a[128];
int b[128];
// initialize b
for (i = 0; i<128; i++)
  a[i] = b[i] + 5; 
```

通常，这样的代码会循环 128 次，但在具备能力的 CPU 上，通过同时计算两个或更多的数组元素，代码的执行可以显著加速。这是因为连续元素之间没有依赖关系，且数组之间的数据没有重叠。聪明的编译器可以将前面的循环转换为如下形式（这发生在汇编级别）：

```cpp
for (i = 0; i<32; i+=4) {
  a[ i ] = b[ i ] + 5;
  a[i+1] = b[i+1] + 5;
  a[i+2] = b[i+2] + 5;
  a[i+3] = b[i+3] + 5;
} 
```

GCC 在 `-O3` 下会启用这种自动循环向量化。Clang 默认启用它。两种编译器都提供不同的标志来启用/禁用特定的向量化：

+   `-ftree-vectorize -ftree-slp-vectorize`：这是用于启用 GCC 中向量化的选项。

+   `-fno-vectorize -fno-slp-vectorize`：这是用于在 Clang 中禁用向量化的选项。

向量化的效率源于利用 CPU 制造商提供的特殊指令，而不仅仅是将原始的循环形式替换为展开的版本。因此，手动实现相同的性能水平是不可行的（此外，这也不会导致*简洁的代码*）。

优化器在提高程序运行时性能方面发挥着至关重要的作用。通过有效地利用其策略，我们可以获得更多的效益。效率不仅在编码完成后很重要，在软件开发过程中同样如此。如果编译时间过长，我们可以通过更好地管理过程来改进它。

# 管理编译过程

作为程序员和构建工程师，我们还必须考虑编译过程中的其他方面，例如完成时间以及在解决方案构建过程中识别和修正错误的便捷性。

## 降低编译时间

在需要频繁重新编译的繁忙项目中（可能每小时多次），确保编译过程尽可能快速是至关重要的。这不仅影响你的代码编译测试循环的效率，还会影响你的专注力和工作流程。

幸运的是，C++已经相当擅长管理编译时间，这要归功于分离的翻译单元。CMake 会确保只重新编译受到最近更改影响的源文件。然而，如果我们需要进一步改善，有几种技术可以使用：头文件预编译和统一构建。

### 头文件的预编译

头文件（`.h`）由预处理器在实际编译开始之前包含到翻译单元中。这意味着每当`.cpp`实现文件发生变化时，它们必须重新编译。此外，如果多个翻译单元使用相同的共享头文件，每次包含时都必须编译一次。这是低效的，但它已经是长期以来的标准做法。

幸运的是，从 CMake 3.16 版本开始，CMake 提供了一个命令来启用头文件预编译。这使得编译器可以将头文件与实现文件分开处理，从而加速编译过程。以下是该命令的语法：

```cpp
target_precompile_headers(<target>
                          <INTERFACE|PUBLIC|PRIVATE> [header1...]
                         [<INTERFACE|PUBLIC|PRIVATE> [header2...]
...]) 
```

添加的头文件列表存储在`PRECOMPILE_HEADERS`目标属性中。正如我们在*第五章*、*与目标的协作*中讨论的，在*什么是传递的使用要求？*部分，我们可以使用传播的属性，通过选择`PUBLIC`或`INTERFACE`关键字，将头文件与任何依赖目标共享；然而，对于使用`install()`命令导出的目标，不应这样做。其他项目不应被强迫使用我们的预编译头文件，因为这并不是一种常规做法。

使用在*第六章*、*使用生成器表达式*中描述的`$<BUILD_INTERFACE:...>`生成器表达式，防止预编译头文件出现在目标的使用要求中，尤其是在它们被安装时。然而，它们仍会被添加到通过`export()`命令从构建树中导出的目标中。如果现在这看起来有点困惑，不用担心——在*第十四章*、*安装与打包*中会做详细说明。

CMake 会将所有头文件的名称放入一个`cmake_pch.h`或`cmake_pch.hxx`文件中，然后将该文件预编译为一个特定于编译器的二进制文件，扩展名为`.pch`、`.gch`或`.pchi`。

我们可以在我们的列表文件中像这样使用它：

**ch07/06-precompile/CMakeLists.txt**

```cpp
add_executable(precompiled hello.cpp)
target_precompile_headers(precompiled PRIVATE <iostream>) 
```

我们也可以在对应的源文件中使用它：

**ch07/06-precompile/hello.cpp**

```cpp
int main() {
  std::cout << "hello world" << std::endl;
} 
```

请注意，在我们的`main.cpp`文件中，我们不需要包含`cmake_pch.h`或任何其他头文件——它将由 CMake 使用特定于编译器的命令行选项包含进来。

在前面的例子中，我使用了一个内置头文件；然而，你可以轻松地添加自己的包含类或函数定义的头文件。可以使用两种形式之一来引用头文件：

+   `header.h`（直接路径）被解释为相对于当前源目录的路径，并将以绝对路径包含。

+   `[["header.h"]]`（双括号和引号）的路径将根据目标的 `INCLUDE_DIRECTORIES` 属性进行扫描，该属性可以通过 `target_include_directiories()` 配置。

一些在线参考资料可能会建议避免预编译那些不是标准库的一部分的头文件，比如 `<iostream>`，或者完全不使用预编译头文件。这是因为修改列表或编辑自定义头文件将导致目标中的所有翻译单元重新编译。使用 CMake 时，这个问题就没有那么严重，尤其是当你正确地组织项目（将项目结构划分为相对较小、聚焦于特定领域的目标）时。每个目标都有一个独立的预编译头文件，这样可以限制头文件更改的影响。

如果你的头文件被认为相对稳定，你可以决定在目标中重用预编译头文件。为此，CMake 提供了一个方便的命令：

```cpp
target_precompile_headers(<target> REUSE_FROM <other_target>) 
```

这会设置目标的 `PRECOMPILE_HEADERS_REUSE_FROM` 属性，重用头文件，并在这些目标之间创建依赖关系。使用这种方法，消费目标将无法再指定自己的预编译头文件。此外，所有的*编译选项*、*编译标志*和*编译定义*必须在目标之间匹配。

注意要求，尤其是如果你有任何使用双括号格式（`[["header.h"]]`）的头文件。两个目标都需要适当设置它们的*包含路径*，以确保编译器能够找到这些头文件。

### Unity 构建

CMake 3.16 引入了另一种编译时间优化功能——Unity 构建，也被称为*统一构建*或*超大构建*。Unity 构建通过利用 `#include` 指令将多个实现源文件合并。这有一些有趣的影响，其中一些是有利的，而另一些可能是有害的。

最明显的优势是，当 CMake 创建统一构建文件时，避免了不同翻译单元中头文件的重新编译：

```cpp
#include "source_a.cpp"
#include "source_b.cpp" 
```

当两个源文件中都有 `#include "header.h"` 行时，参考的文件只会被解析一次，得益于*包含保护*（假设它们已正确添加）。虽然不如预编译头文件精细，但这也是一种替代方案。

这种构建方式的第二个好处是，优化器现在可以在更大的范围内工作，优化所有捆绑源代码之间的过程间调用。这类似于我们在*第四章*、*设置你的第一个 CMake 项目*中的*过程间优化*部分讨论的链接时间优化。

然而，这些好处是有权衡的。由于我们减少了*目标文件*和处理步骤的数量，我们也增加了处理较大文件所需的内存量。此外，我们减少了可并行工作的数量。编译器在多线程编译方面并不特别擅长，因为它们通常不需要这样做——构建系统通常会启动许多编译任务，以便在不同的线程上同时执行所有文件。将所有文件分组在一起会使这一过程变得复杂，因为 CMake 现在需要并行编译的文件变少了。

使用 Unity 构建时，你还需要考虑一些可能不容易察觉的 C++ 语义影响——匿名命名空间隐藏跨文件的符号，现在这些符号的作用域局限于 Unity 文件，而不是单独的翻译单元。静态全局变量、函数和宏定义也会发生同样的情况。这可能会导致名称冲突，或执行错误的函数重载。

Jumbo 构建在重新编译时表现不佳，因为它们会编译比实际需要的更多文件。它们最适合用于代码需要尽可能快地编译所有文件的情况。在 Qt Creator（一个流行的 GUI 库）上进行的测试表明，你可以期望性能提高 20%到 50%之间（具体取决于使用的编译器）。

要启用 Unity 构建，我们有两个选择：

+   将`CMAKE_UNITY_BUILD`变量设置为`true`——它将初始化随后定义的每个目标上的`UNITY_BUILD`属性。

+   手动将`UNITY_BUILD`目标属性设置为`true`，用于所有应使用 Unity 构建的目标。

第二种选择通过调用以下内容来实现：

```cpp
set_target_properties(<target1> <target2> ...
                      PROPERTIES UNITY_BUILD true) 
```

在许多目标上手动设置这些属性当然需要更多的工作，并增加了维护成本，但你可能需要这样做，以便更精细地控制这一设置。

默认情况下，CMake 会创建包含八个源文件的构建，这些源文件由目标的`UNITY_BUILD_BATCH_SIZE`属性指定（该属性在目标创建时从`CMAKE_UNITY_BUILD_BATCH_SIZE`变量复制）。你可以更改目标属性或默认变量。

从版本 3.18 开始，你可以明确地定义文件应如何与命名组捆绑。为此，请将目标的`UNITY_BUILD_MODE`属性更改为`GROUP`（默认值是`BATCH`）。然后，通过将源文件的`UNITY_GROUP`属性设置为你选择的名称来将它们分配到组中：

```cpp
set_property(SOURCE <src1> <src2> PROPERTY UNITY_GROUP "GroupA") 
```

然后，CMake 将忽略`UNITY_BUILD_BATCH_SIZE`并将该组中的所有文件添加到一个 Unity 构建中。

CMake 的文档建议默认情况下不要为公共项目启用统一构建。推荐的做法是，应用程序的最终用户应该能够决定是否希望使用 jumbo 构建，可以通过提供`-DCMAKE_UNITY_BUILD`命令行参数来实现。如果统一构建由于代码编写方式引发问题，你应该明确地将目标的属性设置为 false。然而，你可以自由地为内部使用的代码启用此功能，例如公司内部的代码或你自己的私人项目。

这些是使用 CMake 减少编译时间的最重要方面。编程中还有其他常常让我们浪费大量时间的因素——其中最臭名昭著的就是调试。让我们看看如何在这方面改进。

## 查找错误

作为程序员，我们花费大量时间在寻找 bug 上。不幸的是，这是我们职业的一个事实。识别错误并修复它们的过程常常让人焦躁不安，尤其是当修复需要长时间工作时。当我们缺乏必要的工具来帮助我们在这些困难的情况下航行时，这个难度会大大增加。正因如此，我们必须特别注意如何配置环境，使得这一过程变得更加简化，尽可能轻松和耐受。一种实现这一目标的方法是通过`target_compile_options()`配置编译器。那么，哪些*编译选项*可以帮助我们实现这一目标呢？

### 配置错误和警告

软件开发中有很多令人头疼的事情——在深夜修复关键性 bug，处理大型系统中的高可见度和高成本故障，或者面对恼人的编译错误。一些错误难以理解，而另一些则是繁琐且具有挑战性的修复任务。在你努力简化工作并减少失败的机会时，你会发现很多关于如何配置编译器警告的建议。

其中一个值得注意的建议是默认启用`-Werror`标志进行所有构建。从表面上看，这个标志的功能看起来很简单——它将所有警告视为错误，直到你解决每个警告，代码才会继续编译。虽然看起来似乎是一种有益的方法，但它通常并非如此。

你看，警告之所以不被归类为错误，是有原因的：它们的设计目的是提醒你。如何处理这些警告由你自己决定。特别是在你进行实验或原型开发时，能够忽视某些警告往往是非常宝贵的。

另一方面，如果你有一段完美的、没有警告的、无懈可击的代码，似乎不应该让将来的修改破坏这种完美的状态。启用它并保持在那里，似乎也没有什么坏处，至少在你的编译器没有升级之前是这样。新版本的编译器通常对已弃用的特性更加严格，或者在提供改进建议方面更加高效。虽然这在警告仍然是警告时有益，但它可能导致在代码没有更改的情况下出现意外的构建失败，或者更让人沮丧的是，当你需要快速修复与新警告无关的问题时。

那么，什么时候启用所有可能的警告是可以接受的呢？简短的答案是，当你在创建一个公共库时。在这种情况下，你会希望预防那些因环境比你严格而导致的代码问题的工单。如果你选择启用这个设置，请确保及时更新新的编译器版本及其引入的警告。还需要特别管理这个更新过程，与代码变更的管理分开进行。

否则，让警告保持原样，集中精力处理错误。如果你觉得有必要强求严格，可以使用`-Wpedantic`标志。这个特定的标志会启用严格的 ISO C 和 ISO C++标准要求的所有警告。然而，请记住，这个标志并不能确认标准的符合性；它只是标识出那些需要诊断消息的非 ISO 做法。

更宽容且脚踏实地的程序员将会满足于`-Wall`，可以选择与`-Wextra`搭配使用，增加一些精致的警告，这样就足够了。这些警告被认为是真正有用的，当有时间时，你应该在代码中处理这些警告。

根据你的项目类型，还有许多其他警告标志可能会有用。我建议你阅读所选编译器的手册，看看有哪些可用的选项。

### 调试构建

偶尔，编译会失败。这通常发生在我们尝试重构大量代码或清理我们的构建系统时。有时问题可以很容易解决；然而，也有一些复杂的问题需要深入调查配置步骤。我们已经知道如何打印更详细的 CMake 输出（如在*第一章*中讨论的《CMake 的第一步》），但我们如何分析每个阶段实际上发生了什么？

#### 调试各个阶段

`-save-temps`，可以传递给 GCC 和 Clang 编译器，允许我们调试编译的各个阶段。这个标志会指示编译器将某些编译阶段的输出存储在文件中，而不是存储在内存中。

**ch07/07-debug/CMakeLists.txt**

```cpp
add_executable(debug hello.cpp)
target_compile_options(debug PRIVATE **-save-temps=obj**) 
```

启用此选项将在每个翻译单元中生成两个额外的文件（`.ii` 和 `.s`）。

第一个文件，`<build-tree>/CMakeFiles/<target>.dir/<source>.ii`，存储预处理阶段的输出，并附有注释，解释每部分源代码的来源：

```cpp
# 1 "/root/examples/ch07/06-debug/hello.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# / / / ... removed for brevity ... / / /
# 252 "/usr/include/x86_64-linux-
  gnu/c++/9/bits/c++config.h" 3
namespace std
{
  typedef long unsigned int size_t;
  typedef long int ptrdiff_t;
  typedef decltype(nullptr) nullptr_t;
}
... 
```

第二个文件，`<build-tree>/CMakeFiles/<target>.dir/<source>.s`，包含语言分析阶段的输出，已准备好进入汇编阶段：

```cpp
 .file   "hello.cpp"
        .text
        .section        .rodata
        .type   _ZStL19piecewise_construct, @object
        .size   _ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
        .zero   1
        .local  _ZStL8__ioinit
        .comm   _ZStL8__ioinit,1,1
.LC0:
        .string "hello world"
        .text
        .globl  main
        .type   main, @function
main:
( ... ) 
```

根据问题的类型，我们通常可以揭示实际问题。例如，预处理器的输出可以帮助我们识别错误，如错误的*包含路径*（可能提供错误版本的库），或定义中的错误导致的`#ifdef`评估错误。

与此同时，语言分析的输出对于针对特定处理器和解决关键优化问题尤其有益。

#### 调试头文件包含问题

调试错误的包含文件可能是一个具有挑战性的任务。我应该知道——在我第一份公司工作时，我曾经需要将整个代码库从一个构建系统迁移到另一个。如果你发现自己处于一个需要精确理解用于包含所请求头文件的路径的情况，可以考虑使用`-H`编译选项：

**ch07/07-debug/CMakeLists.txt**

```cpp
add_executable(debug hello.cpp)
target_compile_options(debug PRIVATE **-H**) 
```

产生的输出将类似于以下内容：

```cpp
[ 25%] Building CXX object
  CMakeFiles/inclusion.dir/hello.cpp.o
. /usr/include/c++/9/iostream
.. /usr/include/x86_64-linux-gnu/c++/9/bits/c++config.h
... /usr/include/x86_64-linux-gnu/c++/9/bits/os_defines.h
.... /usr/include/features.h
-- removed for brevity --
.. /usr/include/c++/9/ostream 
```

在*目标文件*的名称后，每一行输出都包含一个头文件路径。在这个例子中，行首的单个点表示顶级包含（`#include`指令位于`hello.cpp`中）。两个点表示此文件由后续文件（`<iostream>`）包含。每增加一个点，表示嵌套的层级增加。

在此输出的末尾，你还可能会看到一些关于如何改进代码的建议：

```cpp
Multiple include guards may be useful for:
/usr/include/c++/9/clocale
/usr/include/c++/9/cstdio
/usr/include/c++/9/cstdlib 
```

虽然你不需要解决标准库中的问题，但你可能会看到一些你自己编写的头文件被列出。在这种情况下，你可能需要考虑进行修正。

### 为调试器提供信息

机器代码是一组神秘的指令和数据，以二进制格式编码。它并没有传达更深层次的意义或目标。这是因为 CPU 并不关心程序的目标是什么，或者所有指令的含义。唯一的要求是代码的正确性。编译器会将上述所有内容翻译成 CPU 指令的数字标识符，存储数据以初始化所需的内存，并提供成千上万的内存地址。换句话说，最终的二进制文件不需要包含实际的源代码、变量名、函数签名或程序员关心的任何其他细节。这就是编译器的默认输出——原始且裸露。

这样做主要是为了节省空间并减少过多的开销。巧合的是，我们也在一定程度上保护了我们的应用程序免受逆向工程的攻击。是的，即使没有源代码，你也可以理解每个 CPU 指令的作用（例如，将这个值复制到那个寄存器）。但是，即使是最基础的程序也包含太多这样的指令，难以理清它们的逻辑。

如果你是一个特别有动力的人，你可以使用一个叫做**反汇编器**的工具，通过大量的知识（和一点运气），你将能够解读可能发生的事情。然而，这种方法并不太实际，因为反汇编的代码没有原始符号，这使得解读程序的逻辑变得非常困难且缓慢。

相反，我们可以要求编译器将源代码与编译后代码与原始代码之间的引用映射一起存储到生成的二进制文件中。然后，我们可以将调试器附加到正在运行的程序上，并查看在任何时刻正在执行哪个源代码行。当我们在编写新功能或修复错误等代码时，这一点是不可或缺的。

这两个用例是两个构建配置的原因：`Debug` 和 `Release`。正如我们之前所看到的，CMake 默认会向编译器提供一些标志来管理此过程，并首先将它们存储在全局变量中：

+   `CMAKE_CXX_FLAGS_DEBUG` 包含 `-g`

+   `CMAKE_CXX_FLAGS_RELEASE`包含 `-DNDEBUG`

`-g`标志的意思是“添加调试信息”。它以操作系统的本地格式提供：stabs、COFF、XCOFF 或 DWARF。这些格式可以被像 `gdb`（GNU 调试器）这样的调试器访问。通常，这对于像 CLion 这样的集成开发环境（IDE）来说是足够的，因为它们在后台使用 `gdb`。在其他情况下，请参考所提供调试器的手册，检查适用于您所选择编译器的正确标志。

对于 `Release` 配置，CMake 会添加 `-DNDEBUG` 标志。这是一个预处理器定义，简单来说就是“不是调试构建”。一些面向调试的宏将被故意禁用，其中之一就是在 `<assert.h>` 头文件中可用的 `assert`。如果你决定在生产代码中使用断言，它们将不起作用：

```cpp
int main(void)
{
    **assert****(****false****)**;
    std::cout << "This shouldn't run. \n";
    return 0;
} 
```

在 `Release` 配置中，`assert(false)` 调用不会产生任何效果，但在 `Debug` 配置中，它会正常停止执行。如果你正在实践断言编程，并且仍然需要在发布版本中使用 `assert()`，你可以选择更改 CMake 提供的默认设置（从 `CMAKE_CXX_FLAGS_RELEASE` 中移除 `NDEBUG`），或者在包含头文件之前实现硬编码的覆盖，方法是取消定义该宏：

```cpp
#undef NDEBUG
#include <assert.h> 
```

更多信息请参见断言参考：[`en.cppreference.com/w/c/error/assert`](https://en.cppreference.com/w/c/error/assert)。

如果您的断言可以在编译时完成，您可以考虑用 C++11 中引入的 `static_assert()` 替代 `assert()`，因为该函数不像 `assert()` 那样被 `#ifndef(NDEBUG)` 预处理器指令保护。

到这里，我们已经学会了如何管理编译过程。

# 总结

我们又完成了一个章节！毫无疑问，编译是一个复杂的过程。由于它的各种边界情况和特定要求，在没有强大工具的支持下很难管理。幸运的是，CMake 在这方面做得非常出色。

那么，到目前为止我们学到了什么呢？我们从讨论编译是什么以及它在构建和运行操作系统中的应用程序这一更广泛叙述中所处的位置开始。然后，我们检查了编译的各个阶段以及管理这些阶段的内部工具。这种理解对于解决我们未来可能遇到的复杂问题是非常宝贵的。

接下来，我们探索了如何使用 CMake 来验证主机上可用的编译器是否满足构建我们代码所需的所有必要要求。正如我们已经确立的那样，对于我们的解决方案的用户来说，看到一条友好的消息，提示他们升级编译器，远比看到由无法处理新语言特性的过时编译器打印出来的晦涩错误信息要好得多。

我们简要讨论了如何将源文件添加到已经定义的目标中，然后继续讲解了预处理器的配置。这是一个相当重要的主题，因为这一阶段将所有代码片段汇集在一起，并决定哪些部分会被忽略。我们谈到了如何提供文件路径并单独或批量添加自定义定义（以及一些用例）。接着，我们讨论了优化器；我们探讨了所有常见的优化级别以及它们隐式添加的标志。我们还详细讲解了一些标志——`finline`、`floop-unroll` 和 `ftree-vectorize`。

最后，是时候回顾更大的图景，并研究如何管理编译的可行性了。我们在这里解决了两个主要方面——减少编译时间（从而帮助保持程序员的专注力）和发现错误。后者对于识别哪些地方出了问题以及为什么会出问题至关重要。正确配置工具并理解事情发生的原因，有助于确保代码的质量（也有助于维护我们的心理健康）。

在下一章，我们将学习链接以及在构建库并在项目中使用它们时需要考虑的所有事项。

# 进一步阅读

欲了解更多信息，您可以参考以下资源：

+   CMake 支持的编译特性和编译器：[`cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers`](https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers)

+   管理目标的源文件: [`stackoverflow.com/questions/32411963/why-is-cmake-file-glob-evil`](https://stackoverflow.com/questions/32411963/why-is-cmake-file-glob-evil), [`cmake.org/cmake/help/latest/command/target_sources.html`](https://cmake.org/cmake/help/latest/command/target_sources.html)

+   `include` 关键字: [`en.cppreference.com/w/cpp/preprocessor/include`](https://en.cppreference.com/w/cpp/preprocessor/include)

+   提供包含文件的路径: [`cmake.org/cmake/help/latest/command/target_include_directories.html`](https://cmake.org/cmake/help/latest/command/target_include_directories.html)

+   配置头文件: [`cmake.org/cmake/help/latest/command/configure_file.html`](https://cmake.org/cmake/help/latest/command/configure_file.html)

+   头文件预编译: [`cmake.org/cmake/help/latest/command/target_precompile_headers.html`](https://cmake.org/cmake/help/latest/command/target_precompile_headers.html)

+   Unity 构建: [`cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html`](https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html)

+   预编译头文件和 Unity 构建: [`www.qt.io/blog/2019/08/01/precompiled-headers-and-unity-jumbo-builds-in-upcoming-cmake`](https://www.qt.io/blog/2019/08/01/precompiled-headers-and-unity-jumbo-builds-in-upcoming-cmake)

+   查找错误 – 编译器标志: [`interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags`](https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags)

+   为什么使用库而不是目标文件: [`stackoverflow.com/questions/23615282/object-files-vs-library-files-and-why`](https://stackoverflow.com/questions/23615282/object-files-vs-library-files-and-why)

+   职责分离[:https://nalexn.github.io/separation-of-concerns/](https://nalexn.github.io/separation-of-concerns/)

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`discord.com/invite/vXN53A7ZcA`](https://discord.com/invite/vXN53A7ZcA)

![](img/QR_Code94081075213645359.png)
