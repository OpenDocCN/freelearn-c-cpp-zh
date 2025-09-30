# *第二章*：探索 LLVM 的构建系统功能

在上一章中，我们了解到 LLVM 的构建系统是一个庞然大物：它包含数百个构建文件，有成千上万的交错构建依赖。更不用说，它还包含需要为异构源文件定制构建指令的目标。这些复杂性驱使 LLVM 采用了一些高级构建系统特性，更重要的是，一个更结构化的构建系统设计。在本章中，我们的目标将是了解一些重要的指令，以便在树内和树外进行 LLVM 开发时编写更简洁和更具表现力的构建文件。

在本章中，我们将涵盖以下主要主题：

+   探索 LLVM 重要 CMake 指令的词汇表

+   在树外项目中通过 CMake 集成 LLVM

# 技术要求

与*第一章*的*构建 LLVM 时的资源节约*类似，你可能想要有一个从源代码构建的 LLVM 副本。可选地，由于本章将涉及大量的 CMake 构建文件，你可能希望为`CMakeLists.txt`准备一个语法高亮插件（例如，VSCode 的*CMake Tools*插件）。所有主流 IDE 和编辑器都应该有现成的。此外，熟悉基本的`CMakeLists.txt`语法是首选的。

本章中所有的代码示例都可以在这个书的 GitHub 仓库中找到：[`github.com/PacktPublishing/LLVM-Techniques-Tips-and-Best-Practices/tree/main/Chapter02`](https://github.com/PacktPublishing/LLVM-Techniques-Tips-and-Best-Practices/tree/main/Chapter02).

# 探索 LLVM 重要 CMake 指令的词汇表

由于在选择底层构建系统方面的更高灵活性，LLVM 已经从**GNU autoconf**切换到**CMake**。从那时起，LLVM 已经提出了许多自定义的 CMake 函数、宏和规则来优化其自身的使用。本节将为您概述其中最重要的和最常用的几个。我们将学习如何以及何时使用它们。

## 使用 CMake 函数添加新库

库是 LLVM 框架的构建块。然而，在为新的库编写`CMakeLists.txt`时，你不应该使用在正常的`CMakeLists.txt`文件中出现的普通`add_library`指令，如下所示：

```cpp
# In an in-tree CMakeLists.txt file…
add_library(MyLLVMPass SHARED
  MyPass.cpp) # Do NOT do this to add a new LLVM library
```

在这里使用普通的`add_library`有几个缺点，如下所示：

+   如*第一章*所示，*构建 LLVM 时的资源节约*，LLVM 更倾向于使用全局 CMake 参数（即`BUILD_SHARED_LIBS`）来控制其所有组件库是否应该静态或动态构建。使用内置指令来做这一点相当困难。

+   与前一点类似，LLVM 更倾向于使用全局 CMake 参数来控制一些编译标志，例如是否在代码库中启用**运行时类型信息**（**RTTI**）和**C++异常处理**。

+   通过使用自定义 CMake 函数/宏，LLVM 可以创建自己的组件系统，这为开发者提供了更高层次的抽象，以便以更简单的方式指定构建目标依赖项。

因此，你应该始终使用这里所示的 `add_llvm_component_library` CMake 函数：

```cpp
# In a CMakeLists.txt
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp)
```

这里，`LLVMFancyOpt` 是最终的库名称，而 `FancyOpt.cpp` 是源文件。

在常规的 CMake 脚本中，你可以使用 `target_link_libraries` 来指定给定目标的库依赖项，然后使用 `add_dependencies` 来在不同构建目标之间分配依赖关系，以创建明确的构建顺序。当你使用 LLVM 的自定义 CMake 函数创建库目标时，有更简单的方法来完成这些任务。

通过在 `add_llvm_component_library`（或 `add_llvm_library`，这是前者的底层实现）中使用 `LINK_COMPONENTS` 参数，你可以指定目标的链接组件：

```cpp
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp
  LINK_COMPONENTS
  Analysis ScalarOpts)
```

或者，你可以使用在函数调用之前定义的 `LLVM_LINK_COMPONENTS` 变量来完成相同的事情：

```cpp
set(LLVM_LINK_COMPONENTS
    Analysis ScalarOpts)
add_llvm_component_library(LLVMFancyOpt
   FancyOpt.cpp)
```

组件库只是具有特殊意义的普通库，当涉及到你可以使用的 *LLVM 构建块* 时。如果你选择构建它，它们也包含在庞大的 `libLLVM` 库中。组件名称与真实库名称略有不同。如果你需要从组件名称到库名称的映射，你可以使用以下 CMake 函数：

```cpp
llvm_map_components_to_libnames(output_lib_names
  <list of component names>)
```

如果你想要直接链接到一个 *普通* 库（非 LLVM 组件的库），你可以使用 `LINK_LIBS` 参数：

```cpp
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp
  LINK_LIBS
  ${BOOST_LIBRARY})
```

要将一般构建目标依赖项分配给库目标（相当于 `add_dependencies`），你可以使用 `DEPENDS` 参数：

```cpp
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp
  DEPENDS
  intrinsics_gen)
```

`intrinsics_gen` 是一个表示生成包含 LLVM 内置定义的头文件过程的通用目标。

### 每个文件夹添加一个构建目标

许多 LLVM 自定义 CMake 函数存在一个涉及源文件检测的陷阱。假设你有一个如下的目录结构：

```cpp
/FancyOpt
  |___ FancyOpt.cpp
  |___ AggressiveFancyOpt.cpp
  |___ CMakeLists.txt
```

这里，你有两个源文件，`FancyOpt.cpp` 和 `AggressiveFancyOpt.cpp`。正如它们的名称所暗示的，`FancyOpt.cpp` 是这种优化的基本版本，而 `AggressiveFancyOpt.cpp` 是相同功能的替代、更激进的版本。自然地，你将希望将它们分成单独的库，以便用户可以选择是否在他们的正常工作量中包含更激进的版本。因此，你可能编写一个 `CMakeLists.txt` 文件如下：

```cpp
# In /FancyOpt/CMakeLists.txt
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp)
add_llvm_component_library(LLVMAggressiveFancyOpt
  AggressiveFancyOpt.cpp)
```

不幸的是，这会在处理第一个 `add_llvm_component_library` 语句时生成错误消息，告诉你类似 `Found unknown source AggressiveFancyOpt.cpp …` 的事情。

LLVM 的构建系统强制执行更严格的规则，以确保同一文件夹中的所有 C/C++ 源文件都被添加到同一个库、可执行文件或插件中。为了解决这个问题，有必要将其中一个文件拆分到一个单独的文件夹中，如下所示：

```cpp
/FancyOpt
  |___ FancyOpt.cpp
  |___ CMakeLists.txt
  |___ /AggressiveFancyOpt
       |___ AggressiveFancyOpt.cpp
       |___ CMakeLists.txt
```

在 `/FancyOpt/CMakeLists.txt` 中，我们有以下内容：

```cpp
add_llvm_component_library(LLVMFancyOpt
  FancyOpt.cpp)
add_subdirectory(AggressiveFancyOpt)
```

最后，在 `/FancyOpt/AggressiveFancyOpt/CMakeLists.txt` 文件中，我们有以下内容：

```cpp
add_llvm_component_library(LLVMAggressiveFancyOpt
  AggressiveFancyOpt.cpp)
```

这些是使用 LLVM 的自定义 CMake 指令添加（组件）库构建目标的基本要素。在接下来的两个部分中，我们将向您展示如何使用一组不同的 LLVM 特定 CMake 指令添加可执行文件和 Pass 插件构建目标。

## 使用 CMake 函数添加可执行文件和工具

与 `add_llvm_component_library` 类似，要添加新的可执行目标，我们可以使用 `add_llvm_executable` 或 `add_llvm_tool`：

```cpp
add_llvm_tool(myLittleTool
  MyLittleTool.cpp)
```

这两个函数具有相同的语法。然而，只有由 `add_llvm_tool` 创建的目标才会包含在安装中。还有一个全局 CMake 变量 `LLVM_BUILD_TOOLS`，它启用/禁用这些 LLVM 工具目标。

这两个函数也可以使用 `DEPENDS` 参数来指定依赖项，类似于我们之前介绍的 `add_llvm_library`。然而，您只能使用 `LLVM_LINK_COMPONENTS` 变量来指定要链接的组件。

## 使用 CMake 函数添加 Pass 插件

尽管我们将在本书的后面部分介绍 Pass 插件开发，但添加 Pass 插件的构建目标现在（与早期仍使用带有一些特殊参数的 `add_llvm_library` 的 LLVM 版本相比）不可能更简单了。我们可以简单地使用以下命令：

```cpp
add_llvm_pass_plugin(MyPass
   HelloWorldPass.cpp)
```

`LINK_COMPONENTS`、`LINK_LIBS` 和 `DEPENDS` 参数也在这里可用，其用法和功能与 `add_llvm_component_library` 中相同。

这些是一些最常见且最重要的 LLVM 特定 CMake 指令。使用这些指令不仅可以使您的 CMake 代码更加简洁，还可以帮助它与 LLVM 的自身构建系统同步，以防您想进行一些树内开发。在下一节中，我们将向您展示如何将 LLVM 集成到树外 CMake 项目中，并利用我们在本章中学到的知识。

树内与树外开发

在本书中，*树内* 开发意味着直接向 LLVM 项目贡献代码，例如修复 LLVM 缺陷或向现有的 LLVM 库添加新功能。另一方面，*树外* 开发可能代表为 LLVM 创建扩展（例如编写 LLVM Pass）或在其他项目中使用 LLVM 库（例如使用 LLVM 的代码生成库来实现您自己的编程语言）。

# 理解树外项目的 CMake 集成

在树内项目中实现你的功能对于原型设计是有益的，因为大部分基础设施已经存在。然而，与创建一个 **树外项目** 并将其链接到 LLVM 库相比，有许多场景将整个 LLVM 源代码树拉入你的代码库并不是最佳选择。例如，你可能只想创建一个使用 LLVM 功能的小型代码重构工具并在 GitHub 上开源，那么让 GitHub 上的开发者下载与你小巧的工具一起的多吉字节 LLVM 源代码树可能不会是一个愉快的体验。

配置树外项目以链接到 LLVM 至少有两种方式：

+   使用 `llvm-config` 工具

+   使用 LLVM 的 CMake 模块

这两种方法都有助于你整理所有细节，包括头文件和库路径。然而，后者创建的 CMake 脚本更简洁、更易读，这对于已经使用 CMake 的项目来说更可取。本节将展示使用 LLVM 的 CMake 模块将 LLVM 集成到树外 CMake 项目的必要步骤。

首先，我们需要准备一个树外（C/C++）CMake 项目。我们在上一节中讨论的核心 CMake 函数/宏将帮助我们完成这项工作。让我们看看我们的步骤：

1.  我们假设你已经为需要链接到 LLVM 库的项目准备好了以下 `CMakeLists.txt` 框架：

    ```cpp
    project(MagicCLITool)
    set(SOURCE_FILES
        main.cpp)
    add_executable(magic-cli
      ${SOURCE_FILES})
    ```

    无论你是在尝试创建一个生成可执行文件的项目，就像我们在前面的代码块中看到的那样，还是其他如库或甚至 LLVM Pass 插件等工件，现在最大的问题是如何获取 `包含路径` 以及 `库路径`。

1.  为了解决 `包含路径` 和 `库路径`，LLVM 为你提供了标准的 CMake 包接口，你可以使用 `find_package` CMake 指令导入各种配置，如下所示：

    ```cpp
    project(MagicCLITool)
    find_package trick work, you need to supply the LLVM_DIR CMake variable while invoking the CMake command for this project:

    ```

    LLVM 安装路径下的 `lib/cmake/llvm` 子目录。

    ```cpp

    ```

1.  在解决包含路径和库之后，是时候将主可执行文件链接到 LLVM 的库上了。LLVM 的自定义 CMake 函数（例如，`add_llvm_executable`）在这里将非常有用。但首先，CMake 需要能够 *找到* 这些函数。

    以下片段导入了 LLVM 的 CMake 模块（更具体地说，是 `AddLLVM` CMake 模块），其中包含我们在上一节中介绍过的那些 LLVM 特定函数/宏：

    ```cpp
    find_package(LLVM REQUIRED CONFIG)
    …
    list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
    include(AddLLVM)
    ```

1.  以下片段使用我们在上一节中介绍过的 CMake 函数添加了可执行文件的构建目标：

    ```cpp
    find_package(LLVM REQUIRED CONFIG)
    …
    include(AddLLVM)
    set(LLVM_LINK_COMPONENTS
      Support
      Analysis)
    add_llvm_executable(magic-cli
      main.cpp)
    ```

1.  添加库目标没有区别：

    ```cpp
    find_package(LLVM REQUIRED CONFIG)
    …
    include(AddLLVM)
    add_llvm_library(MyMagicLibrary
      lib.cpp
      LINK_COMPONENTS
      Support Analysis)
    ```

1.  最后，添加 LLVM Pass 插件：

    ```cpp
    find_package(LLVM REQUIRED CONFIG)
    …
    include(AddLLVM)
    add_llvm_pass_plugin(MyMagicPass
      ThePass.cpp)
    ```

1.  在实践中，你还需要注意 **LLVM 特定定义** 和 RTTI 设置：

    ```cpp
    find_package(LLVM REQUIRED CONFIG)
    …
    add_definitions(${LLVM_DEFINITIONS})
    if(NOT ${LLVM_ENABLE_RTTI})
      # For non-MSVC compilers
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif()
    add_llvm_xxx(source.cpp)
    ```

    这对于 RTTI 部分尤其如此，因为默认情况下，LLVM 并未构建 RTTI 支持，而正常的 C++ 应用程序是支持的。如果你的代码和 LLVM 库之间存在 RTTI 不匹配，将会抛出编译错误。

尽管在 LLVM 源树内开发很方便，但有时将整个 LLVM 源代码包含在你的项目中可能并不可行。因此，我们必须创建一个树外项目，并将 LLVM 作为库进行集成。本节展示了如何将 LLVM 集成到基于 CMake 的树外项目中，并充分利用我们在“探索 LLVM 重要 CMake 指令词汇表”部分学到的 LLVM 特定 CMake 指令。

# 摘要

本章深入探讨了 LLVM 的 CMake 构建系统。我们看到了如何使用 LLVM 自己的 CMake 指令来编写简洁有效的构建脚本，无论是树内还是树外开发。掌握这些 CMake 技能可以使你的 LLVM 开发更加高效，并为你提供更多与现有代码库或自定义逻辑交互 LLVM 功能的选择。

在下一章中，我们将介绍 LLVM 项目中另一个重要的基础设施，称为 LLVM LIT，这是一个易于使用且通用的框架，用于运行各种类型的测试。
