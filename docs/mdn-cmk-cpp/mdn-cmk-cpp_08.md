# 第五章：使用 CMake 编译 C++源代码

简单的编译场景通常由工具链的默认配置或直接由 IDE 提供。然而，在专业环境中，业务需求往往需要更高级的东西。可能是对更高性能、更小二进制文件、更可移植性、测试支持或广泛的调试功能的需求——您说得都对。以一种连贯、未来无忧的方式管理所有这些，很快就会变得复杂、纠缠不清（尤其是在需要支持多个平台的情况下）。

编译过程在 C++书籍中往往没有解释得足够清楚（像虚拟基类这样的深入主题似乎更有趣）。在本章中，我们将回顾基础知识，以确保事情不如预期时能取得成功。我们将发现编译是如何工作的，它的内部阶段是什么，以及它们如何影响二进制输出。

之后，我们将重点关注先决条件——我们将讨论我们可以使用哪些命令来调整编译，如何从编译器那里要求特定的功能，以及如何向编译器提供必须处理的输入文件。

然后，我们将重点关注编译的第一阶段——预处理器。我们将提供包含头文件的路径，并研究如何插入 CMake 和环境预处理器定义。我们将涵盖一些有趣的用例，并学习如何大量暴露 CMake 变量给 C++代码。

紧接着，我们将讨论优化器以及不同标志如何影响性能。我们还将痛苦地意识到优化的代价——调试被破坏的代码有多困难。

最后，我们将解释如何通过使用预编译头和单元编译来减少编译时间，为发现错误做准备，调试构建，以及在最终二进制文件中存储调试信息。

在本章中，我们将涵盖以下主要主题：

+   编译的基础

+   预处理器配置

+   配置优化器

+   管理编译过程

# 技术要求

您可以在 GitHub 上找到本章中存在的代码文件，地址为[`github.com/PacktPublishing/Modern-CMake-for-Cpp/tree/main/examples/chapter05`](https://github.com/PacktPublishing/Modern-CMake-for-Cpp/tree/main/examples/chapter05)。

构建本书提供的示例时，始终使用建议的命令：

```cpp
cmake -B <build tree> -S <source tree>
cmake --build <build tree>
```

请确保将占位符`<build tree>`和`<source tree>`替换为适当的路径。作为提醒：**build tree**是目标/输出目录的路径，**source tree**是您的源代码所在的路径。

# 编译的基础

编译可以大致描述为将用高级编程语言编写的指令翻译成低级机器代码的过程。这允许我们使用类和对象等抽象概念来创建应用程序，而无需关心处理器特定汇编语言的繁琐细节。我们不需要直接与 CPU 寄存器打交道，考虑短跳或长跳，以及管理堆栈帧。编译语言更有表现力、可读性、更安全，并促进更易维护的代码（但性能尽可能）。

在 C++中，我们依赖于静态编译——整个程序必须在执行之前翻译成本地代码。这是 Java 或 Python 等语言的替代方法，这些语言每次用户运行时都使用特殊的、独立的解释器编译程序。每种方法都有其优点。C++的政策是为尽可能多的提供高级工具，同时仍能以完整的、自包含的应用程序的形式，为几乎所有的架构提供本地性能。

创建并运行一个 C++程序需要几个步骤：

1.  设计你的应用程序并仔细编写源代码。

1.  将单个`.cpp`实现文件（称为翻译单元）编译成*目标文件*。

1.  将*目标文件*链接成单个可执行文件，并添加所有其他依赖项——动态和静态库。

1.  要运行程序，操作系统将使用一个名为*加载器*的工具将它的机器代码和所有必需的动态库映射到虚拟内存。加载器然后读取头文件以检查程序从哪里开始，并将控制权交给代码。

1.  启动 C++运行时；执行特殊的`_start`函数来收集命令行参数和环境变量。它开始线程，初始化静态符号，并注册清理回调。然后它调用由程序员编写的`main()`函数。

正如你所见，幕后发生了相当多的工作。本章讨论的是前述列表中的第二步。从整体的角度考虑，我们可以更好地理解一些可能问题的来源。毕竟，软件中没有黑魔法（即使难以理解的复杂性让它看起来像是那样）。一切都有解释和原因。程序运行时可能会失败，是因为我们如何编译它（即使编译步骤本身已经成功完成）。编译器在其工作中检查所有边缘情况是不可能的。

## 编译是如何工作的

如前所述，编译是将高级语言翻译成低级语言的过程——具体来说，是通过产生特定处理器可以直接执行的机器代码，以二进制**对象文件**格式生成，该格式特定于给定平台。在 Linux 上，最流行的格式是**可执行和可链接格式**（**ELF**）。Windows 使用 PE/COFF 格式规范。在 macOS 上，我们会找到 Mach 对象（Mach-O 格式）。

对象文件**是单个源文件的直接翻译。每一个对象文件都需要单独编译，之后链接器将它们合并成一个可执行文件或库。正因为如此，当你修改了代码，只需重新编译受影响的文件，就能节省时间。

编译器必须执行以下阶段来创建一个**对象文件**：

+   预处理

+   语言分析

+   汇编

+   优化

+   代码生成

`#include`指令，用定义的值替换标识符（`#define`指令和`-D`标志），调用简单的宏，并根据`#if`、`#elif`和`#endif`指令有条件地包含或排除代码的一部分。预处理器对实际的 C++代码一无所知，通常只是一个更高级的查找和替换工具。然而，它在构建高级程序中的工作至关重要；将代码分成部分并在多个翻译单元之间共享声明是代码可重用的基础。

接下来是**语言分析**。在这里，更有趣的事情会发生。编译器将逐字符扫描文件（包含预处理器包含的所有头文件），并进行词法分析，将它们分组成有意义的标记——关键字、操作符、变量名等。然后，标记被分组成标记链，并检查它们的顺序和存在是否遵循 C++的规则——这个过程称为语法分析或解析（通常，在打印错误方面，它是声音最大的部分）。最后，进行语义分析——编译器尝试检测文件中的语句是否真的有意义。例如，它们必须满足类型正确性检查（你不能将整数赋值给字符串变量）。

**汇编**不过是将这些标记翻译成基于平台可用指令集的 CPU 特定指令。一些编译器实际上会创建一个汇编输出文件，之后再传递给专门的汇编器程序，以产生 CPU 可执行的机器代码。其他的编译器直接从内存中产生相同的机器代码。通常，这类编译器包括一个选项，以产生人类可读的汇编代码文本输出（尽管，仅仅因为你能读它，并不意味着它值得这么做）。

**优化**在整个编译过程中逐步进行，一点一点地，在每个阶段。在生成第一个汇编版本之后有一个明确的阶段，负责最小化寄存器的使用和删除未使用的代码。一个有趣且重要的优化是在线扩展或*内联*。编译器将“剪切”函数的主体并“粘贴”代替其调用（标准未定义这种情况发生在哪些情况下——这取决于编译器的实现）。这个过程加快了执行速度并减少了内存使用，但对调试有重大缺点（执行的代码不再在原始行上）。

**代码发射**包括根据目标平台指定的格式将优化后的机器代码写入*对象文件*。这个*对象文件*不能直接执行——它必须传递给下一个工具，链接器，它将适当移动我们*对象文件*的各个部分并解决对外部符号的引用。这是从 ASCII 源代码到可被处理器处理的二进制*对象文件*的转换。

每个阶段都具有重要意义，可以根据我们的特定需求进行配置。让我们看看如何使用 CMake 管理这个过程。

## 初始配置

CMake 提供了多个命令来影响每个阶段：

+   `target_compile_features()`：要求具有特定特性的编译器编译此目标。

+   `target_sources()`：向已定义的目标添加源文件。

+   `target_include_directories()`：设置预处理器*包含路径*。

+   `target_compile_definitions()`：设置预处理器定义。

+   `target_compile_options()`：命令行上的编译器特定选项。

+   `target_precompile_headers()`：优化外部头的编译。

所有上述命令都接受类似的参数：

```cpp
target_...(<target name> <INTERFACE|PUBLIC|PRIVATE>
  <value>)
```

这意味着它们支持属性传播，如前章所讨论的，既可以用于可执行文件也可以用于库。顺便提一下——所有这些命令都支持生成器表达式。

### 要求编译器具有特定的特性

如第三章“*设置你的第一个 CMake 项目*”中讨论的，*检查支持的编译器特性*，为使用你的软件的用户准备可能出错的事情，并努力提供清晰的消息——**可用的编译器 X 没有提供所需的特性 Y**。这比用户可能拥有的不兼容的工具链产生的任何错误都要好。我们不希望用户假设是你的代码出了问题，而不是他们过时的环境。

以下命令允许你指定构建目标所需的所有特性：

```cpp
target_compile_features(<target> <PRIVATE|PUBLIC|INTERFACE>
                        <feature> [...])
```

CMake 理解 C++标准和这些`compiler_ids`所支持的编译器特性：

+   `AppleClang`：Xcode 版本 4.4+的 Apple Clang

+   `Clang`：Clang 编译器版本 2.9+

+   `GNU`: GNU 编译器 4.4+版本

+   `MSVC`: Microsoft Visual Studio 2010+版本

+   `SunPro`: Oracle Solaris Studio 12.4+版本

+   `Intel`: Intel 编译器 12.1+版本

    重要提示

    当然，您可以使用任何`CMAKE_CXX_KNOWN_FEATURES`变量，但我建议坚持使用通用 C++标准——`cxx_std_98`、`cxx_std_11`、`cxx_std_14`、`cxx_std_17`、`cxx_std_20`或`cxx_std_23`。查看*进阶阅读*部分以获取更多详细信息。

## 管理目标源代码

我们已经知道如何告诉 CMake 哪些源文件组成一个目标——一个可执行文件或一个库。我们在使用`add_executable()`或`add_library()`时提供文件列表。

随着解决方案的增长，每个目标的文件列表也在增长。我们可能会得到一些非常长的`add_...()`命令。我们如何处理呢？一种诱惑可能是使用`GLOB`模式的`file()`命令——它可以收集子目录中的所有文件并将它们存储在一个变量中。我们将其作为目标声明的参数传递，并不再担心列表文件：

```cpp
file(GLOB helloworld_SRC "*.h" "*.cpp")
add_executable(helloworld ${helloworld_SRC})
```

然而，前面提到的方法并不推荐。让我们找出原因。CMake 根据列表文件的变化生成构建系统，因此如果没有进行任何更改，构建可能会在没有警告的情况下失败（我们知道，在花费了长时间进行调试后，这种类型的失败是最糟糕的）。除此之外，不在目标声明中列出所有源代码将导致代码审查在 IDE（如 CLion）中失败（CLion 只解析一些命令以理解您的项目）。

如果不建议在目标声明中使用变量，我们如何才能在例如处理特定平台的实现文件（如`gui_linux.cpp`和`gui_windows.cpp`）时条件性地添加源文件呢？

我们可以使用`target_sources()`命令将文件追加到先前创建的目标：

chapter05/01-sources/CMakeLists.txt

```cpp
add_executable(main main.cpp)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(main PRIVATE gui_linux.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_sources(main PRIVATE gui_windows.cpp)
endif()
```

这样，每个平台都可以获得自己的兼容文件集合。很好，但是长文件列表怎么办呢？嗯，我们只能接受有些事情目前还不完美，并继续手动添加它们。

既然我们已经确立了编译的关键事实，让我们更仔细地看看第一步——预处理。与计算机科学中的所有事情一样，细节是魔鬼。

# 预处理器配置

预处理器在构建过程中的作用非常大。这可能有点令人惊讶，因为它的功能多么简单和有限。在接下来的部分，我们将介绍为包含文件提供路径和使用预处理器定义。我们还将解释如何使用 CMake 配置包含的头文件。

## 为包含文件提供路径

预处理器最基本的功能是使用`#include`指令包含`.h`/`.hpp`头文件。它有两种形式：

+   `#include <path-spec>`: 尖括号形式

+   `#include "path-spec"`: 引号形式

正如我们所知，预处理器将这些指令替换为`path-spec`中指定的文件的正文。找到这些文件可能是个问题。我们搜索哪些目录以及按什么顺序？不幸的是，C++标准并没有确切指定；我们需要查看我们使用的编译器的手册。

通常，尖括号形式将检查标准*包含目录*，包括系统中存储标准 C++库和标准 C 库头文件的目录。

引号形式将开始在当前文件的目录中搜索包含的文件，然后检查尖括号形式的目录。

CMake 提供了一个命令，用于操作搜索包含文件所需的路径：

```cpp
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [item1...]
  [<INTERFACE|PUBLIC|PRIVATE> [item2...] ...])
```

我们可以添加自定义路径，我们希望编译器检查。CMake 将在生成的构建系统中为编译器调用添加它们。它们将用适合特定编译器的标志提供（通常是`-I`）。

使用`BEFORE`或`AFTER`确定路径应该附加到目标`INCLUDE_DIRECTORIES`属性之前还是之后。是否检查这里提供的目录还是默认目录之前还是之后（通常，是之前）仍然由编译器决定。

`SYSTEM`关键字通知编译器，提供的目录是作为标准系统目录（与尖括号形式一起使用）。对于许多编译器，这个值将作为`-isystem`标志提供。

## 预处理器定义

记得我提到预处理器的`#define`和`#if`、`#elif`、`#endif`指令时描述编译阶段吗？让我们考虑以下示例：

chapter05/02-definitions/definitions.cpp

```cpp
#include <iostream>
int main() {
#if defined(ABC)
    std::cout << "ABC is defined!" << std::endl;
#endif
#if (DEF < 2*4-3)
    std::cout << "DEF is greater than 5!" << std::endl;
#endif
}
```

如它所示，这个例子什么也不做；在这个例子中`ABC`和`DEF`都没有定义（在这个例子中`DEF`将默认为`0`）。我们可以在这个代码顶部添加两行轻松地改变这一点：

```cpp
#define ABC
#define DEF 8
```

编译并执行此代码后，我们可以在控制台看到两条消息：

```cpp
ABC is defined!
DEF is greater than 5!
```

这看起来很简单，但如果我们想根据外部因素（如操作系统、体系结构或其他内容）来条件这些部分，会发生什么情况呢？好消息！您可以将值从 CMake 传递给 C++编译器，而且一点也不复杂。

`target_compile_definitions()`命令将解决这个问题：

chapter05/02-definitions/CMakeLists.txt

```cpp
set(VAR 8)
add_executable(defined definitions.cpp)
target_compile_definitions(defined PRIVATE ABC
  "DEF=${VAR}")
```

前面的代码将与两个`#define`声明完全一样，但我们有自由使用 CMake 的变量和生成表达式，并且可以将命令放在条件块中。

重要提示

这些定义传统上通过`-D`标志传递给编译器——`-DFOO=1`——一些程序员仍然在这个命令中使用这个标志：

`target_compile_definitions(hello PRIVATE -DFOO)`

CMake 识别这一点，并将移除任何前面的`-D`标志。它还会忽略空字符串，所以即使写如下内容也是可以的：

`target_compile_definitions(hello PRIVATE -D FOO)`

`-D`是一个独立的参数；移除后它将变成一个空字符串，然后正确地被忽略。

### 单元测试私有类字段时的常见陷阱

一些在线资源建议在单元测试中使用特定的`-D`定义与`#ifdef/ifndef`指令的组合。最简单的可能方法是将访问修饰符包裹在条件包含中，并在定义`UNIT_TEST`时忽略它们：

```cpp
class X {
#ifndef UNIT_TEST
 private: 
#endif
  int x_;
}
```

虽然这种用例非常方便（它允许测试直接访问私有成员），但这不是非常整洁的代码。单元测试应该只测试公共接口中方法是否如预期工作，并将底层实现视为黑盒机制。我建议你只在万不得已时使用这个方法。

### 使用 git 提交跟踪编译版本

让我们考虑一下在了解环境或文件系统详情方面有益的用例。一个在专业环境中可能很好的例子是传递用于构建二进制的修订版或提交`SHA`：

chapter05/03-git/CMakeLists.txt

```cpp
add_executable(print_commit print_commit.cpp)
execute_process(COMMAND git log -1 --pretty=format:%h
                OUTPUT_VARIABLE SHA)
target_compile_definitions(print_commit PRIVATE
  "SHA=${SHA}")
```

我们可以在应用程序中如此使用它：

chapter05/03-git/print_commit.cpp

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

当然，上述代码需要用户在他们的`PATH`中安装并可访问`git`。这对于运行在我们生产主机上的程序来自持续集成/部署管道很有用。如果我们的软件有问题时，我们可以快速检查用于构建有缺陷产品的确切 Git 提交。

跟踪确切的提交对调试非常有用。对于一个变量来说，这不是很多工作，但是当我们想要将数十个变量传递给我们的头文件时会发生什么？

## 配置头文件

如果我们有多个变量，通过`target_compile_definitions()`传递定义可能会有些繁琐。我们不能提供一个带有引用各种变量的占位符的头文件，并让 CMake 填充它们吗？

当然我们可以！使用`configure_file(<input> <output>)`命令，我们可以从模板生成新的文件，就像这个一样：

chapter05/04-configure/configure.h.in

```cpp
#cmakedefine FOO_ENABLE
#cmakedefine FOO_STRING1 "@FOO_STRING@"
#cmakedefine FOO_STRING2 "${FOO_STRING}"
#cmakedefine FOO_UNDEFINED "@FOO_UNDEFINED@"
```

我们可以使用命令，像这样：

chapter05/04-configure/CMakeLists.txt

```cpp
add_executable(configure configure.cpp)
set(FOO_ENABLE ON)
set(FOO_STRING1 "abc")
set(FOO_STRING2 "def")
configure_file(configure.h.in configured/configure.h)
target_include_directories(configure PRIVATE 
                           ${CMAKE_CURRENT_BINARY_DIR})
```

我们可以让 CMake 生成一个输出文件，像这样：

chapter05/04-configure/<build_tree>/configure.h

```cpp
#define FOO_ENABLE
#define FOO_STRING1 "abc"
#define FOO_STRING2 "def"
/* #undef FOO_UNDEFINED "@FOO_UNDEFINED@" */
```

正如你所见，`@VAR@`和`${VAR}`变量占位符被替换成了 CMake 列表文件中的值。此外，`#cmakedefine`被替换成了`#define`给已定义的变量，对于未定义的变量则替换成`/* #undef VAR */`。

如果你需要为`#if`块提供显式的`#define 1`或`#define 0`，请使用`#cmakedefine01`。

我们如何在应用程序中使用这样的配置头文件？我们可以在实现文件中简单地包含它：

chapter05/04-configure/configure.cpp

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
  cout << "FOO_ENABLE1: " << xstr(FOO_ENABLE1) << endl;
  cout << "FOO_ENABLE2: " << xstr(FOO_ENABLE2) << endl;
  cout << "FOO_UNDEFINED: " << xstr(FOO_UNDEFINED) << endl;
}
```

由于我们已使用`target_include_directories()`命令将二叉树添加到了我们的*包含路径*中，因此我们可以编译示例并从 CMake 接收填充好的输出：

```cpp
FOO_ENABLE: ON
FOO_ENABLE1: FOO_ENABLE1
FOO_ENABLE2: FOO_ENABLE2
FOO_UNDEFINED: FOO_UNDEFINED
```

`configure_file()`命令还具有许多格式化和文件权限选项。在这里描述它们可能会稍显冗长。如果你有兴趣，可以查看在线文档以获取详细信息（链接在*进一步阅读*部分）。

在准备好我们头文件和源文件的完整组合后，我们可以讨论在下一步中输出代码是如何形成的。由于我们无法直接影响语言分析或汇编（这些步骤遵循严格的标准），我们肯定可以访问优化器的配置。让我们了解它如何影响最终结果。

# 配置优化器

优化器将分析前阶段的结果，并使用多种程序员认为不整洁的技巧，因为它们不符合整洁代码原则。没关系——优化器的关键作用是使代码具有高性能（即，使用较少的 CPU 周期、较少的寄存器和较少的内存）。当优化器遍历源代码时，它会对其进行大量转换，以至于它几乎变得无法辨认。它变成了针对目标 CPU 的特殊准备版本。

优化器不仅会决定哪些函数可以被删除或压缩；它还会移动代码或甚至显著地重复它！如果它可以完全确定某些代码行是没有意义的，它就会从重要函数的中间抹去它们（你甚至都注意不到）。它会重复利用内存，所以众多变量在不同时间段可以占据同一个槽位。如果这意味着它可以节省一些周期，它还会将你的控制结构转换成完全不同的结构。

这里描述的技术，如果由程序员手动应用到源代码中，将会使其变得可怕、难以阅读。编写和推理将会困难。另一方面，如果由编译器应用，那就非常棒了，因为编译器将严格遵循所写的内容。优化器是一种无情的野兽，只服务于一个目的：使执行快速，无论输出会变得多么糟糕。如果我们在测试环境中运行它，输出可能包含一些调试信息，或者它可能不包含，以便让未授权的人难以篡改。

每个编译器都有自己的技巧，与它所遵循的平台和哲学相一致。我们将查看 GNU GCC 和 LLVM Clang 中可用的最常见的一些，以便我们可以了解什么是有用和可能的。

问题是——许多编译器默认不会启用任何优化（包括 GCC）。这在某些情况下是可以的，但在其他情况下则不然。为什么要慢慢来，当你可以快速前进时呢？要改变事物，我们可以使用`target_compile_options()`命令，并精确指定我们想从编译器那里得到什么。

这个命令的语法与本章中的其他命令相似：

```cpp
target_compile_options(<target> [BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```

我们提供`target`命令行选项以添加，并指定传播关键字。当执行此命令时，CMake 将在目标相应的`COMPILE_OPTIONS`变量中附加给定选项。可选的`BEFORE`关键字可用于指定我们想要在它们之前添加它们。在某些情况下，顺序很重要，因此能够选择是件好事。

重要提示

`target_compile_options()`是一个通用命令。它也可以用来为类似编译器的`-D`定义提供其他参数，对于这些参数，CMake 还提供了`target_compile_definition()`命令。始终建议尽可能使用 CMake 命令，因为它们在所有支持的编译器上都是一致的。

是讨论细节的时候了。接下来的章节将介绍您可以在大多数编译器中启用的各种优化方法。

## 通用级别

优化器的所有不同行为都可以通过我们作为*编译选项*传递的具体标志进行深度配置。了解它们需要花费大量时间，并需要深入了解编译器、处理器和内存的内部工作原理。如果我们只想在大多数情况下都能良好工作的最佳可能场景怎么办？我们可以寻求一个通用解决方案——一个优化级别指定符。

大多数编译器提供四个基本级别的优化，从`0`到`3`。我们使用`-O<level>`选项指定它们。`-O0`意味着*没有优化*，通常，这是编译器的默认级别。另一方面，`-O2`被认为是*完全优化*，它生成高度优化的代码，但编译时间最慢。

有一个中间的`-O1`级别，根据您的需求，它可以是一个很好的折中方案——它启用了适量的优化机制，而不会使编译速度变得太慢。

最后，我们可以使用`-O3`，这是*完全优化*，类似于`-O2`，但它在子程序内联和循环向量化方面采取了更为激进的方法。

还有一些优化变体，它们将优化生成文件的大小（不一定是速度）——`-Os`。还有一个超级激进的优化，`-Ofast`，它是不严格符合 C++标准的`-O3`优化。最明显的区别是使用`-ffast-math`和`-ffinite-math`标志，这意味着如果你的程序是关于精确计算（像大多数程序一样），你可能想避免使用它。

CMake 知道并非所有的编译器都平等，因此，为了提供一致的开发体验，它为编译器提供了一些默认标志。这些标志存储在系统级（非目标特定）变量中，用于指定使用的语言（`CXX`用于 C++）和构建配置（`DEBUG`或`RELEASE`）：

+   `CMAKE_CXX_FLAGS_DEBUG`等于`-g`。

+   `CMAKE_CXX_FLAGS_RELEASE`等于`-O3 -DNDEBUG`。

正如你所看到的，调试配置没有启用任何优化，而发布配置直接选择了`O3`。如果你愿意，你可以直接使用`set()`命令更改它们，或者只是添加一个目标编译选项，这将覆盖这个默认行为。另外两个标志（`-g,` `-DNDEBUG`）与调试有关——我们将在*为调试器提供信息*部分讨论它们。

诸如`CMAKE_<LANG>_FLAGS_<CONFIG>`之类的变量是全局的——它们适用于所有目标。建议通过`target_compile_options()`等属性和命令来配置目标，而不是依赖全局变量。这样，你可以更精细地控制你的目标。

通过使用`-O<level>`选择优化级别，我们间接设置了一系列标志，每个标志控制一个特定的优化行为。然后，我们可以通过添加更多标志来微调优化：

+   使用`-f`选项启用它们：`-finline-functions`。

+   使用`-fno`选项禁用它们：`-fno-inline-functions`。

其中一些标志值得更深入地了解，因为它们通常会影响你的程序如何运行以及你如何可以调试它。让我们来看看。

## 函数内联

正如你所回忆的，编译器可以被鼓励内联某些函数，要么在类声明块内*定义*一个函数，要么明确使用`inline`关键字：

```cpp
struct X {
  void im_inlined(){ cout << "hi\n"; }; 
  void me_too();
};
inline void X::me_too() { cout << "bye\n"; };
```

是否内联函数由编译器决定。如果启用了内联并且函数在一个地方使用（或者是一个在几个地方使用的小函数），那么很可能会发生内联。

这是一种非常有趣的优化技术。它通过从所述函数中提取代码，并将它放在函数被调用的所有地方，替换原始调用并节省宝贵的 CPU 周期来工作。

让我们考虑一下我们刚刚定义的类以下示例：

```cpp
int main() {
  X x;
  x.im_inlined();
  x.me_too();
  return 0;
}
```

如果没有内联，代码将在`main()`帧中执行，直到一个方法调用。然后，它会为`im_inlined()`创建一个新帧，在一个单独的作用域中执行，并返回到`main()`帧。对`me_too()`方法也会发生同样的事情。

然而，当内联发生时，编译器将替换这些调用，如下所示：

```cpp
int main() {
  X x;
  cout << "hi\n";
  cout << "bye\n";
  return 0;
}
```

这不是一个精确的表示，因为内联是在汇编语言或机器代码级别（而不是源代码级别）发生的，但它传达了一个大致的画面。

编译器这样做是为了节省时间；它不必经历新调用帧的创建和销毁，不必查找下一条要执行（并返回）的指令地址，而且因为它们彼此相邻，编译器可以更好地缓存这些指令。

当然，内联有一些重要的副作用；如果函数使用不止一次，它必须被复制到所有地方（意味着文件大小更大，使用的内存更多）。如今，这可能不像过去那么关键，但仍然相关，因为我们不断开发必须在内存有限的高端设备上运行的软件。

除此之外，当我们调试自己编写的代码时，它对我们的影响尤为关键。内联代码不再位于其最初编写的行号，因此跟踪起来不再那么容易（有时甚至不可能），这就是为什么在调试器中放置的断点永远不会被击中（尽管代码以某种方式被执行）。为了避免这个问题，我们只能禁用调试构建中的内联功能（代价是不再测试与发布构建完全相同的版本）。

我们可以通过为目标指定`-O0`级别或直接针对负责的标志：

+   `-finline-functions-called-once`：仅 GCC 支持

+   `-finline-functions`：Clang 和 GCC

+   `-finline-hint-functions`：仅 Clang 支持

+   `-finline-functions-called-once`：仅 GCC 支持

你可以使用`-fno-inline-...`显式禁用内联。无论如何，对于详细信息，请参阅您编译器的特定版本的文档。

## 循环展开

循环展开是一种优化技术，也被称为循环展开。通用方法是将循环转换为一组实现相同效果的语句。这样做，我们将用程序的大小换取执行速度，因为我们减少了或消除了控制循环的指令——指针算术或循环末端测试。

请考虑以下示例：

```cpp
void func() {
  for(int i = 0; i < 3; i++)
    cout << "hello\n";
}
```

之前的代码将被转换为类似这样的内容：

```cpp
void func() {
    cout << "hello\n";
    cout << "hello\n";
    cout << "hello\n";
} 
```

结果将相同，但我们不再需要分配`i`变量，增加它，或三次将其与`3`进行比较。如果我们程序运行期间调用`func()`足够多次，即使是对这样一个简短且小的函数进行展开，也会产生显著的差异。

然而，理解两个限制因素很重要。循环展开只有在编译器知道或可以有效估计迭代次数时才能工作。其次，循环展开可能会对现代 CPU 产生不希望的效果，因为代码尺寸的增加可能会阻止有效缓存。

每个编译器提供这个标志的略有不同的版本：

+   `-floop-unroll`：GCC

+   `-funroll-loops`：Clang

如果你有疑问，广泛测试这个标志是否影响你的特定程序，并显式启用或禁用它。请注意，在 GCC 上，`-O3`作为隐式启用的`-floop-unroll-and-jam`标志的一部分隐式启用。

## 循环向量化

**单指令多数据**（**SIMD**）是 20 世纪 60 年代初为实现并行化而开发的一种机制。它的工作方式正如其名称所暗示的那样；它可以同时对多块信息执行相同的操作。实际意味着什么？让我们考虑以下示例：

```cpp
int a[128];
int b[128];
// initialize b
for (i = 0; i<128; i++)
  a[i] = b[i] + 5;
```

通常，前面的代码会循环 128 次，但是有了性能强大的 CPU，我们可以通过同时计算数组中的两个或更多元素来大大加快代码的执行速度。这之所以可行，是因为连续元素之间没有依赖性，数组之间的数据也没有重叠。智能编译器可以将前面的循环转换成类似于此的东西（这发生在汇编语言级别）：

```cpp
for (i = 0; i<32; i+=4) {
  a[ i ] = b[ i ] + 5;
  a[i+1] = b[i+1] + 5;
  a[i+2] = b[i+2] + 5;
  a[i+3] = b[i+3] + 5;
}
```

GCC 会在`-O3`时启用循环的自动向量化。Clang 默认启用。这两个编译器提供了不同的标志来启用/禁用向量化：

+   `-ftree-vectorize -ftree-slp-vectorize` 在 GCC 中启用

+   `-fno-vectorize -fno-slp-vectorize` 在 Clang 中禁用（如果东西坏了）

向量化性能的提升来自于利用 CPU 制造商提供的特殊指令，而不仅仅是简单地将循环的原始形式替换为展开版本。因此，手动操作是无法达到相同性能水平的（而且代码也不太整洁）。

优化器在提高程序运行时的性能方面起着重要作用。通过有效地运用其策略，我们可以物有所值。效率的重要性不仅在于编码完成后，还在于我们开发软件的过程中。如果编译时间过长，我们可以通过更好地管理编译过程来改进它们。

# 管理编译过程

作为程序员和构建工程师，我们需要考虑编译的其他方面——完成所需的时间，以及如何容易地发现和修复在构建解决方案过程中犯的错误。

## 减少编译时间

在需要每天（或每小时）进行许多十几个重新编译的繁忙项目中，编译速度尽可能快是至关重要的。这不仅影响了你的代码-编译-测试循环的紧密程度，还影响了你的注意力和工作流程。幸运的是，C++在管理编译时间方面已经相当出色，这要归功于独立的翻译单元。CMake 会处理仅重新编译最近更改影响的源代码。然而，如果我们需要进一步改进，我们可以使用一些技术——头文件预编译和单元构建：

### 头文件预编译

头文件（`.h`）在实际编译开始前由预处理器包含在翻译单元中。这意味着每当`.cpp`实现文件发生变化时，它们都必须重新编译。此外，如果多个翻译文件使用相同的共享头文件，每次包含时都必须重新编译。这是浪费，但长期以来一直是这样。*

幸运的是，自从版本 3.16 以来，CMake 提供了一个命令来启用头文件预编译。这使得编译器可以单独处理头文件和实现文件，从而加快编译速度。提供命令的语法如下：*

```cpp
target_precompile_headers(<target>
  <INTERFACE|PUBLIC|PRIVATE> [header1...]
  [<INTERFACE|PUBLIC|PRIVATE> [header2...] ...])
```

添加的头文件列表存储在`PRECOMPILE_HEADERS`目标属性中。正如你在第四章，《使用目标》中了解到的，*我们可以使用传播属性通过使用`PUBLIC`或`INTERFACE`关键字将头文件与任何依赖的目标共享；然而，对于使用`install()`命令导出的目标，不应该这样做。其他项目不应当被迫消耗我们的预编译头文件（因为这不符合常规）。*

重要提示：*

如果你需要内部预编译头文件但仍然希望安装导出目标，那么第四章，《使用目标》中描述的`$<BUILD_INTERFACE:...>`生成器表达式将防止头文件出现在使用要求中。然而，它们仍然会被添加到使用`export()`命令从构建树导出的目标中。*

CMake 会将所有头文件的名称放入一个`cmake_pch.h|xx`文件中，然后预编译为具有`.pch`、`.gch`或`.pchi`扩展名的特定于编译器的二进制文件。*

我们可以像这样使用它：*

chapter05/06-precompile/CMakeLists.txt*

```cpp
add_executable(precompiled hello.cpp)
target_precompile_headers(precompiled PRIVATE <iostream>)
```

chapter05/06-precompile/hello.cpp*

```cpp
int main() {
  std::cout << "hello world" << std::endl;
} 
```

请注意，在我们的`main.cpp`文件中，我们不需要包含`cmake_pch.h`或其他任何头文件——CMake 会使用特定的命令行选项强制包含它们。*

在前一个示例中，我使用了一个内置的头文件；然而，你可以很容易地添加自己的头文件，带有类或函数定义：*

+   `header.h`被视为相对于当前源目录的，并将使用绝对路径包含进来。*

+   `[["header.h"]]`根据编译器的实现来解释，通常可以在`INCLUDE_DIRECTORIES`变量中找到。使用`target_include_directiories()`来配置它。*

一些在线参考资料将不鼓励预编译不属于标准库的头文件，如`<iostream>`，或使用预编译头文件。这是因为更改列表或编辑自定义头文件会导致目标中所有翻译单元的重新编译。使用 CMake，你不需要担心这么多，尤其是如果你正确地组织你的项目（具有相对较小的目标，专注于狭窄的领域）。每个目标都有一个单独的预编译头文件，限制了头文件更改的扩散。*

另一方面，如果你的头文件被认为相当稳定，你可能会决定从一个小目标中重复使用预编译的头文件到另一个目标中。CMake 为此目的提供了一个方便的命令：

```cpp
target_precompile_headers(<target> REUSE_FROM
  <other_target>)
```

这设置了使用头文件的目标的`PRECOMPILE_HEADERS_REUSE_FROM`属性，并在这些目标之间创建了一个依赖关系。使用这种方法，消费目标无法再指定自己的预编译头文件。另外，所有*编译选项*、*编译标志*和*编译定义*必须在目标之间匹配。注意要求，特别是如果你有任何使用双括号格式的头文件（`[["header.h"]]`）。两个目标都需要适当地设置它们的*包含路径*，以确保编译器能够找到这些头文件。

### Unity 构建

CMake 3.16 还引入了另一个编译时间优化功能——统一构建，也称为*统一构建*或*巨构建*。统一构建将多个实现源文件与`#include`指令结合在一起（毕竟，编译器不知道它是在包含头文件还是实现）。这带来了一些有趣的含义——有些是非常有用的，而其他的是潜在有害的。

让我们从最明显的一个开始——避免在 CMake 创建统一构建文件时在不同翻译单元中重新编译头文件：

```cpp
#include "source_a.cpp"
#include "source_b.cpp"
```

当这两个源中都包含`#include "header.h"`行时，多亏了*包含守卫*（假设我们没有忘记添加那些），它只会被解析一次。这不如预编译头文件优雅，但这是一个选项。

这种构建方式的第二个好处是，优化器现在可以更大规模地作用，并优化所有捆绑源之间的跨过程调用。这类似于我们在第二章《CMake 语言》中讨论的链接时间优化。

然而，这些好处是有代价的。因为我们减少了*对象文件*的数量和处理步骤，我们也增加了处理更大文件所需的内存量。此外，我们减少了并行化工作量。编译器并不是真正那么擅长多线程编译，因为它们不需要——构建系统通常会启动许多编译任务，以便在不同的线程上同时执行所有文件。当我们把所有文件放在一起时，我们会使它变得困难得多，因为 CMake 现在会在我们创建的多个巨构建之间安排并行构建。

在使用统一构建时，你还需要考虑一些可能不是那么明显捕捉到的 C++语义含义——匿名命名空间跨文件隐藏符号现在被分组到一组中。静态全局变量、函数和宏定义也是如此。这可能会导致名称冲突，或者执行不正确的函数重载。

在重新编译时，巨构构建不受欢迎，因为它们会编译比所需更多的文件。当代码旨在尽可能快地整体编译所有文件时，它们效果最佳。在 Qt Creator 上进行的测试表明，您可以期待性能提升在 20%到 50%之间（取决于所使用的编译器）。

启用统一构建，我们有两个选项：

+   将`CMAKE_UNITY_BUILD`变量设置为`true`——它将在定义后的每个目标上初始化`UNITY_BUILD`属性。

+   手动将`UNITY_BUILD`设置为每个应使用统一构建的目标的`true`。

第二个选项是通过以下方式实现的：

```cpp
set_target_properties(<target1> <target2> ... 
                      PROPERTIES UNITY_BUILD true)
```

默认情况下，CMake 将创建包含八个源文件的构建，这是由目标的`UNITY_BUILD_BATCH_SIZE`属性指定的（在创建目标时从`CMAKE_UNITY_BUILD_BATCH_SIZE`变量复制）。您可以更改目标属性或默认变量。

自版本 3.18 起，你可以选择明确地定义文件如何与命名组一起打包。为此，将目标的`UNITY_BUILD_MODE`属性更改为`GROUP`（默认值始终为`BATCH`）。然后，你需要通过将他们的`UNITY_GROUP`属性设置为你选择的名称来为源文件分配组：

```cpp
set_property(SOURCE <src1> <src2>... 
             PROPERTY UNITY_GROUP "GroupA") 
```

然后，CMake 将忽略`UNITY_BUILD_BATCH_SIZE`，并将组中的所有文件添加到单个巨构构建中。

CMake 的文档建议不要默认启用公共项目的统一构建。建议您的应用程序的最终用户能够通过提供`DCMAKE_UNITY_BUILD`命令行参数来决定他们是否需要巨构构建。更重要的是，如果由于您的代码编写方式而引起问题，您应该明确将目标属性设置为`false`。然而，这并不妨碍您为内部使用的代码启用此功能，例如在公司内部或为您私人项目使用。

### 不支持的 C++20 模块

如果你密切关注 C++标准的发布，你会知道 C++20 引入了一个新特性——模块。这是一个重大的变革。它允许你避免使用头文件时的许多烦恼，减少构建时间，并使得代码更简洁、更易于导航和推理。

本质上，我们可以创建一个带有模块声明的单文件，而不是创建一个单独的头部和实现文件：

```cpp
export module hello_world;
import <iostream>; 
export void hello() {
    std::cout << "Hello world!\n";
}
```

然后，你可以在代码中简单地导入它：

```cpp
import hello_world;
int main() {
    hello();
}
```

注意我们不再依赖预处理器；模块有自己的关键字——`import`、`export`和`module`。最受欢迎的编译器最新版本已经可以执行所有必要的任务，以支持模块作为编写和构建 C++解决方案的新方法。我原本希望在本章开始时，CMake 已经提供了对模块的早期支持。不幸的是，这一点尚未实现。

然而，到你购买这本书的时候（或不久之后）可能就有了。有一些非常好的指标；Kitware 开发者已经创建（并在 3.20 中发布）了一个新的实验性特性，以支持 C++20 模块依赖项扫描对 Ninja 生成器的支持。现在，它只打算供编译器编写者使用，这样他们就可以在开发过程中测试他们的依赖项扫描工具。

当这个备受期待的特性完成并在一个稳定的版本中可用时，我建议彻底研究它。我预计它将简化并大大加快编译速度，超过今天可用的任何东西。

## 查找错误。

作为程序员，我们花了很多时间寻找 bug。这是一个悲哀的事实。查找并解决错误常常会让我们感到不舒服，尤其是如果它需要长时间的话。如果我们没有仪器帮助我们导航暴风雨，盲目飞行会更困难。这就是为什么我们应该非常小心地设置我们的环境，使这个过程尽可能容易和可忍受。我们通过使用`target_compile_options()`配置编译器来实现这一点。那么*编译选项*能帮助我们什么呢？

### 配置错误和警告。

软件开发中有许多令人压力很大的事情——比如在半夜修复关键错误、在高知名度的大型系统中处理昂贵的失败、以及处理那些令人烦恼的编译错误，尤其是那些难以理解或修复起来极其繁琐的错误。当研究一个主题以简化你的工作并减少失败的可能性时，你会发现有很多关于如何配置编译器警告的建议。

一条这样的好建议就是为所有构建启用`-Werror`标志作为默认设置。这个标志做的简单而无辜的事情是——所有警告都被视为错误，除非你解决所有问题，否则代码不会编译。虽然这可能看起来是个好主意，但几乎从来不是。

你看，警告之所以不是错误，是有原因的。它们是用来警告你的。决定如何处理这是你的事。拥有忽视警告的自由，尤其是在你实验和原型化解决方案时，通常是一种祝福。

另一方面，如果你有一个完美无瑕、没有警告、闪闪发光的代码，允许未来的更改破坏这种情况真是太可惜了。启用它并只是保持在那里会有什么害处呢？表面上看起来没有。至少在你升级编译器之前是这样。编译器的新版本往往对弃用的特性更加严格，或者更好地提出改进建议。当你不将所有警告视为错误时，这很好，但当你这样做时，有一天你会发现你的构建开始在没有代码更改的情况下失败，或者更令人沮丧的是，当你需要快速修复一个与新警告完全无关的问题时。

那么，“几乎不”是什么意思，当你实际上应该启用所有可能的警告时？快速答案是当你编写一个公共库时。这时，你真的想避免因为你的代码在一个比你的环境更严格的编译器中编译而产生问题报告。如果你决定启用它，请确保你对编译器的新版本和它引入的警告了如指掌。

否则，让警告就是警告，专注于错误。如果你觉得自己有必要吹毛求疵，可以使用`-Wpedantic`标志。这是一个有趣的选择——它启用了所有严格遵循 ISO C 和 ISO C++所要求的警告。请注意，使用此标志并不能检查代码是否符合标准——它只能找到需要诊断信息的非 ISO 实践。

更加宽容和脚踏实地的程序员会对`-Wall`感到满意，可选地加上`-Wextra`，以获得那种额外的华丽感觉。这些被认为是有实际用处和意义的警告，当你有空时应该修复你的代码中的这些问题。

还有许多其他的警告标志，这取决于项目的类型可能会有所帮助。我建议你阅读你选择的编译器的手册，看看有什么可用。

### 调试构建过程

偶尔，编译会失败。这通常发生在我们试图重构一堆代码或清理我们的构建系统时。有时，事情很容易解决，但随后会有更复杂的问题，需要深入分析配置的每个步骤。我们已经知道如何打印更详细的 CMake 输出（如在第一章中讨论的，*CMake 的初步步骤*），但我们如何分析在每个阶段实际发生的情况呢？

#### 调试单个阶段

我们可以向编译器传递`-save-temps`标志（GCC 和 Clang 都有这个标志），它将强制将每个阶段的输出存储在文件中，而不是内存中：

chapter05/07-debug/CMakeLists.txt

```cpp
add_executable(debug hello.cpp)
target_compile_options(debug PRIVATE -save-temps=obj)
```

前面的片段通常会产生两个额外的文件：

+   `<build-tree>/CMakeFiles/<target>.dir/<source>.ii`：存储预处理阶段的输出，带有注释解释源代码的每一部分来自哪里：

    ```cpp
    # 1 "/root/examples/chapter05/06-debug/hello.cpp"
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

+   `<build-tree>/CMakeFiles/<target>.dir/<source>.s`：语言分析阶段的输出，准备进入汇编阶段：

    ```cpp
            .file   "hello.cpp"
            .text
            .section        .rodata
            .type   _ZStL19piecewise_construct, @object
            .size   _ZStL19piecewise_construct, 1
    _ZStL19piecewise_construct:
            .zero   1
            .local  _ZStL8__ioinit
            .comm   _ZStL8__ioinit,1,1
    .LC0:
            .string "hello world"
            .text
            .globl  main
            .type   main, @function
    main:
    ( ... )
    ```

根据问题的性质，我们通常可以发现实际的问题所在。预处理器的输出对于发现诸如不正确的*include 路径*（提供错误版本的库）以及导致错误`#ifdef`评估的定义错误等 bug 很有帮助。

语言分析阶段的输出对于针对特定处理器和解决关键优化问题很有用。

#### 解决头文件包含的调试问题

错误地包含的文件可能是一个真正难以调试的问题。我应该知道——我的第一份企业工作就是将整个代码库从一种构建系统移植到另一种。如果你发现自己需要精确了解正在使用哪些路径来包含请求的头文件，可以使用`-H`：

chapter05/07-debug/CMakeLists.txt

```cpp
add_executable(debug hello.cpp)
target_compile_options(debug PRIVATE -H)
```

打印出的输出将类似于这样：

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

在`object file`的名称之后，输出中的每一行都包含一个头文件的路径。行首的一个点表示顶级包含（`#include`指令在`hello.cpp`中）。两个点意味着这个文件被`<iostream>`包含。进一步的点表示嵌套的又一层。

在这个输出的末尾，你也许还会找到对代码可能的改进建议：

```cpp
Multiple include guards may be useful for:
/usr/include/c++/9/clocale
/usr/include/c++/9/cstdio
/usr/include/c++/9/cstdlib
```

你不必修复标准库，但可能会看到一些自己的头文件。你可能想修正它们。

### 提供调试器信息

机器代码是一系列用二进制格式编码的指令和数据，它不传达任何意义或目标。这是因为 CPU 不关心程序的目标是什么，或者所有指令的含义是什么。唯一的要求是代码的正确性。编译器会将所有内容转换成 CPU 指令的数值标识符、一些用于初始化内存的数据以及成千上万的内存地址。换句话说，最终的二进制文件不需要包含实际的源代码、变量名、函数签名或程序员关心的任何其他细节。这就是编译器的默认输出——原始且干燥。

这样做主要是为了节省空间并在执行时尽量减少开销。巧合的是，我们也在一定程度上（somewhat）保护了我们的应用程序免受逆向工程。是的，即使没有源代码，你也可以理解每个 CPU 指令做什么（例如，将这个整数复制到那个寄存器）。但最终，即使是基本程序也包含太多这样的指令，很难思考大局。

如果你是一个特别有驱动力的人，你可以使用一个名为**反汇编器**的工具，并且凭借大量的知识（还有一点运气），你将能够理解可能发生了什么。这种方法并不非常实用，因为反汇编代码没有原始符号，所以很难且缓慢地弄清楚哪些部分应该放在哪里。

相反，我们可以要求编译器将源代码存储在生成的二进制文件中，并与包含编译后和原始代码之间引用关系的映射一起存储。然后，我们可以将调试器连接到运行中的程序，并查看任何给定时刻正在执行哪一行源代码。当我们编写代码时，例如编写新功能或修正错误，这是不可或缺的。

这两个用例是两个配置文件（`Debug`和`Release`）的原因。正如我们之前看到的，CMake 会默认提供一些标志给编译器来管理这个过程，首先将它们存储在全局变量中：

+   `CMAKE_CXX_FLAGS_DEBUG`包含了`-g`。

+   `CMAKE_CXX_FLAGS_RELEASE`包含了`-DNDEBUG`。

`-g`标志的意思是*添加调试信息*。它以操作系统的本地格式提供——stabs、COFF、XCOFF 或 DWARF。这些格式随后可以被诸如`gdb`（GNU 调试器）之类的调试器访问。通常，这对于像 CLion 这样的 IDE 来说已经足够好了（因为它们在底层使用`gdb`）。在其他情况下，请参考提供的调试器的手册，并检查对于您选择的编译器，适当的标志是什么。

对于`RELEASE`配置，CMake 将添加`-DNDEBUG`标志。这是一个预处理器定义，简单意味着*不是调试构建*。当启用此选项时，一些面向调试的宏可能不会工作。其中之一就是`assert`，它在`<assert.h>`头文件中可用。如果你决定在你的生产代码中使用断言，它们将根本不会工作：

```cpp
int main(void)
{
    bool my_boolean = false;
    assert(my_boolean); 
    std::cout << "This shouldn't run. \n"; 
    return 0;
}
```

在`Release`配置中，`assert(my_boolean)`调用将不会产生任何效果，但在`Debug`模式下它会正常工作。如果你在实践断言性编程的同时还需要在发布构建中使用`assert()`，你会怎么做？要么更改 CMake 提供的默认设置（从`CMAKE_CXX_FLAGS_RELEASE`中移除`NDEBUG`），要么通过在包含头文件前取消定义宏来实现硬编码覆盖：

```cpp
#undef NDEBUG
#include <assert.h>
```

有关`assert`的更多信息，请参考：[`en.cppreference.com/w/c/error/assert`](https://en.cppreference.com/w/c/error/assert)。

# 总结

我们已经完成了又一章！毫无疑问，编译是一个复杂的过程。有了所有的边缘情况和特定要求，如果没有一个好工具，管理起来可能会很困难。幸运的是，CMake 在这方面做得很好。

到目前为止，我们学到了什么？我们首先讨论了编译是什么以及它在操作系统中构建和运行应用程序的更广泛故事中的位置。然后，我们研究了编译的阶段以及管理它们的内部工具。这对于解决我们将来可能会遇到的更高级别案例中的所有问题非常有用。

然后，我们探讨了如何让 CMake 验证宿主上可用的编译器是否满足我们代码构建的所有必要要求。正如我们之前所确定的，对于我们的解决方案的用户来说，看到一个友好的消息要求他们升级，而不是由一个混淆于语言新特性的旧编译器打印出的某些神秘错误，会是一个更好的体验。

我们简要讨论了如何向已定义的目标添加源代码，然后转向预处理器配置。这是一个相当大的主题，因为这一阶段将所有的代码片段汇集在一起，决定哪些将被忽略。我们谈论了提供文件路径以及作为单个参数和批量（还有一些用例）添加自定义定义。

然后，我们讨论了优化器；我们探索了所有通用优化级别的优化以及它们隐含的标志，但我们也详细讨论了其中的一些——`finline`、`floop-unroll`和`ftree-vectorize`。

最后，是再次研究整体编译流程和如何管理编译可行性的时候了。在这里我们解决了两个主要问题——减少编译时间（从而加强程序员的注意力集中）和查找错误。后者对于发现什么坏了和如何坏是非常重要的。正确设置工具并了解事情为何如此发生，在确保代码质量（以及我们的心理健康）方面起着很长的作用。

在下一章中，我们将学习链接知识，以及我们需要考虑的所有构建库和使用它们的项目中的事情。

## 进一步阅读

+   关于本章涵盖的更多信息，你可以参考以下内容：*CMake 支持的编译特性和编译器：* [`cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers`](https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers)

+   *管理目标源文件：*

    +   [Stack Overflow 讨论：为什么 CMake 的文件匹配功能这么“邪恶”？](https://stackoverflow.com/questions/32411963/why-is-cmake-file-glob-evil)

    +   [CMake 官方文档：target_sources 命令](https://cmake.org/cmake/help/latest/command/target_sources.html)

+   *提供包含文件的路径：*

    +   [C++参考：预处理器中的#include 指令](https://en.cppreference.com/w/cpp/preprocessor/include)

    +   [CMake 官方文档：target_include_directories 命令](https://cmake.org/cmake/help/latest/command/target_include_directories.html)

+   *配置头文件：* [CMake 官方文档：configure_file 命令](https://cmake.org/cmake/help/latest/command/configure_file.html)

+   *预编译头文件：* [CMake 官方文档：target_precompile_headers 命令](https://cmake.org/cmake/help/latest/command/target_precompile_headers.html)

+   *统一构建：*

    +   [CMake 官方文档：UNITY_BUILD 属性](https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html)

    +   [Qt 官方博客：关于即将到来的 CMake 中的预编译头文件和大型构建](https://www.qt.io/blog/2019/08/01/precompiled-headers-and-unity-jumbo-builds-in-upcoming-cmake)

+   *查找错误——编译器标志：* [`interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags`](https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags)

+   *为什么使用库而不是对象文件：* [`stackoverflow.com/questions/23615282/object-files-vs-library-files-and-why`](https://stackoverflow.com/questions/23615282/object-files-vs-library-files-and-why)

+   分离关注点：*[https://nalexn.github.io/separation-of-concerns/](https://nalexn.github.io/separation-of-concerns/*
