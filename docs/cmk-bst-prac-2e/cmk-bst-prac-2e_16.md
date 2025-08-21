# 第十三章：重用 CMake 代码

为项目编写构建系统代码并非易事。项目维护者和开发人员在编写 CMake 代码时会花费大量精力来配置编译器标志、项目构建变体、第三方库和工具集成。从头开始编写用于配置项目无关细节的 CMake 代码，在处理多个 CMake 项目时可能会开始带来显著的负担。为项目编写的这些 CMake 代码中的大部分内容，都是可以在多个项目之间重用的。考虑到这一点，我们有必要制定一个策略，使我们的 CMake 代码易于重用。解决这个问题的直接方法是将 CMake 代码视为常规代码，并应用一些最基本的编码原则：**不要重复自己**（**DRY**）原则和**单一责任**（**SRP**）原则。

如果按照重用性的思路来结构化 CMake 代码，那么 CMake 代码是可以轻松重用的。实现基本的重用性非常简单：将 CMake 代码拆分成模块和函数。你可能已经意识到，使 CMake 代码可重用的方法与使软件代码可重用的方法是一样的。记住——毕竟 CMake 本身就是一种脚本语言。所以，将 CMake 代码视为常规代码并应用软件设计原则是完全自然的。像任何功能性脚本语言一样，CMake 拥有以下基本重用能力：

+   能够包含其他 CMake 文件

+   函数/宏

+   可移植性

在本章中，我们将学习如何编写具有重用性思想的 CMake 代码，并在 CMake 项目中重用 CMake 代码。我们还将讨论如何在项目之间进行版本管理和共享常用的 CMake 代码。

为了理解本章分享的技能，我们将涵盖以下主要主题：

+   什么是 CMake 模块？

+   模块的基本构建块——函数和宏

+   编写你的第一个 CMake 模块

我们从技术要求开始。

# 技术要求

在深入本章之前，建议先阅读 *第一章*，*快速入门 CMake*。本章采用以示例教学的方法，因此建议从这里获取本章的示例内容：[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/tree/main/chapter13`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/tree/main/chapter13)。对于所有示例，假设你将使用该项目提供的容器：[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/)。

让我们首先了解一些 CMake 中重用性的基本知识。

# 什么是 CMake 模块？

一个`Find*.cmake`模块)。CMake 默认提供的模块列表可以在[`cmake.org/cmake/help/latest/manual/cmake-modules.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html)查看。官方的 CMake 文档将模块分为以下两大类：

+   工具模块

+   查找模块

正如它们的名字所示，工具模块提供工具，而查找模块则旨在搜索系统中的第三方软件。如你所记得，我们在*第四章*《*CMake 项目的打包、部署与安装*》和*第五章*《*集成第三方库*及依赖管理》中详细讨论了查找模块。因此，在本章中我们将专注于工具模块。你会记得，在前几章中我们使用了一些 CMake 提供的工具模块。我们使用的一些模块有`GNUInstallDirs`、`CPack`、`FetchContent`和`ExternalProject`。这些模块位于`CMake`安装目录下。

为了更好地理解工具模块的概念，让我们从研究 CMake 提供的一个简单的工具模块开始。为此，我们将研究`ProcessorCount`工具模块。你可以在[`github.com/Kitware/CMake/blob/master/Modules/ProcessorCount.cmake`](https://github.com/Kitware/CMake/blob/master/Modules/ProcessorCount.cmake)找到该模块的源文件。`ProcessorCount`模块是一个允许在 CMake 代码中获取系统 CPU 核心数的模块。`ProcessorCount.cmake`文件定义了一个名为`ProcessorCount`的 CMake 函数，它接受一个名为`var`的参数。该函数的实现大致如下：

```cpp
function(ProcessorCount var)
  # Unknown:
  set(count 0)
  if(WIN32)
    set(count "$ENV{NUMBER_OF_PROCESSORS}")
  endif()
  if(NOT count)
    # Mac, FreeBSD, OpenBSD (systems with sysctl):
    # … mac-specific approach … #
  endif()
  if(NOT count)
    # Linux (systems with nproc):
    # … linux-specific approach … #
  endif()
# … Other platforms, alternative fallback methods … #
# Lastly:
set(${var} ${count} PARENT_SCOPE)
endfunction()
```

`ProcessorCount`函数尝试多种不同的方法来获取主机机器的 CPU 核心数。使用`ProcessorCount`模块非常简单，如下所示：

```cpp
   include(ProcessorCount)
   ProcessorCount(CORE_COUNT)
   message(STATUS "Core count: ${CORE_COUNT}")
```

如你在上面的示例中所看到的，使用 CMake 模块就像将模块包含到所需的 CMake 文件中一样简单。`include()`函数是递归的，因此`include`行之后的代码可以使用模块中包含的所有 CMake 定义。

我们现在大致了解了一个工具模块的样子。接下来，继续学习更多关于工具模块的基本构建块：函数和宏。

# 模块的基本构建块——函数和宏

很明显，我们需要一些基本的构建块来创建工具模块。工具模块的最基本构建块是函数和宏，因此掌握它们的工作原理至关重要。让我们从学习函数开始。

## 函数

让我们回顾一下在 *第一章*《启动 CMake》中学到的关于函数的内容。`function(…)` 拥有一个包含 CMake 命令的函数体，并以 `endfunction()` CMake 命令结束。`function()` 命令的第一个参数需要是函数名，其后可以有可选的函数参数名，如下所示：

```cpp
function(<name> [<arg1> ...])
  <commands>
endfunction()
```

函数定义了一个新的变量作用域，因此对 CMake 变量的修改仅在函数体内可见。独立作用域是函数的最重要特性。拥有新的作用域意味着我们无法意外地泄露变量给调用者或修改调用者的变量，*除非我们愿意这么做*。大多数情况下，我们希望将修改限制在函数的作用域内，并仅将函数的结果反映给调用者。由于 CMake 不支持返回值，我们将采取*在调用者作用域中定义变量*的方法来返回函数结果给调用者。

为了说明这种方法，我们定义一个简单的函数来一起获取当前的 Git 分支名称：

```cpp
function(git_get_branch_name result_var_name)
  execute_process(
        COMMAND git symbolic-ref -q --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE git_current_branch_name
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(${result_var_name} ${git_current_branch_name}
      PARENT_SCOPE)
endfunction()
```

`git_get_branch_name` 函数接受一个名为 `result_var_name` 的单一参数。该参数是将被定义在调用者作用域中的变量名，用于返回 Git 分支名称给调用者。或者，我们可以使用一个常量变量名，比如 `GIT_CURRENT_BRANCH_NAME`，并去掉 `result_var_name` 参数，但如果项目已经使用了 `GIT_CURRENT_BRANCH_NAME` 这个名字，这可能会导致问题。

这里的经验法则是将命名留给调用者，因为这可以提供最大的灵活性和可移植性。为了获取当前的 Git 分支名称，我们通过 `execute_process()` 调用了 `git symbolic-ref -q --short HEAD` 命令。命令的结果存储在函数作用域内的 `git_current_branch_name` 变量中。由于该变量处于函数作用域内，调用者无法看到 `git_current_branch_name` 变量。因此，我们使用 `set(${result_var_name} ${git_current_branch_name} PARENT_SCOPE)` 来在调用者的作用域内定义一个变量，使用的是本地 `git_current_branch_name` 变量的值。

`PARENT_SCOPE` 参数改变了 `set(…)` 命令的作用域，使得它在调用者的作用域中定义变量，而不是在函数作用域中定义。`git_get_branch_name` 函数的用法如下：

```cpp
git_get_branch_name(branch_n)
message(STATUS "Current git branch name is: ${branch_n}")
```

接下来我们来看看宏。

## 宏

如果函数的作用域对你的使用场景是个难题，你可以考虑改用宏。宏以 `macro(…)` 开始，以 `endmacro()` 结束。函数和宏在各个方面表现得很相似，但有一个区别：宏不会定义新的变量作用域。回到我们的 Git 分支示例，考虑到 `execute_process(…)` 已经有了 `OUTPUT_VARIABLE` 参数，定义 `git_get_branch_name` 为宏而非函数会更方便，这样就可以避免在结尾使用 `set(… PARENT_SCOPE)`：

```cpp
macro(git_get_branch_name_m result_var_name)
  execute_process(
        COMMAND git symbolic-ref -q --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE ${result_var_name}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endmacro()
```

`git_get_branch_name_m`宏的使用与`git_get_branch_name()`函数相同：

```cpp
git_get_branch_name_m(branch_nn)
message(STATUS "Current git branch name is: ${branch_nn}")
```

我们已经学习了如何在需要时定义函数或宏。接下来，我们将一起定义第一个 CMake 模块。

# 编写你自己的第一个 CMake 模块

在上一节中，我们学习了如何使用函数和宏为 CMake 项目提供有用的工具。现在，我们将学习如何将这些函数和宏移到一个单独的 CMake 模块中。

创建和使用一个基本的 CMake 模块文件非常简单：

1.  在你的项目中创建一个`<module_name>.cmake`文件。

1.  在`<module_name>.cmake`文件中定义任何宏/函数。

1.  在需要的文件中包含`<module_name>.cmake`。

好的——让我们按照这些步骤一起创建一个模块。作为我们之前 Git 分支名称示例的后续，我们将扩大范围，编写一个 CMake 模块，提供通过`git`命令获取分支名称、头部提交哈希值、当前作者名称和当前作者电子邮件信息的能力。对于这一部分，我们将参考`chapter13/ex01_git_utility`的示例。示例文件夹包含一个`CMakeLists.txt`文件和一个位于`.cmake`文件夹下的`git.cmake`文件。让我们首先来看一下`.cmake/git.cmake`文件，文件内容如下：

```cpp
# …
include_guard(DIRECTORY)
macro(git_get_branch_name result_var_name)
    execute_process(
        COMMAND git symbolic-ref -q --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE ${result_var_name}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endmacro()
# … git_get_head_commit_hash(), git_get_config_value()
```

`git.cmake`文件是一个 CMake 实用模块文件，其中包含三个宏，分别为`git_get_branch_name`、`git_get_head_commit_hash`和`git_get_config_value`。此外，文件顶部有一个`include_guard(DIRECTORY)`行。这类似于 C/C++中的`#pragma once`预处理指令，防止该文件被多次包含。`DIRECTORY`参数表示`include_guard`在目录范围内定义，意味着该文件在当前目录及其下属目录中最多只能被包含一次。或者，也可以指定`GLOBAL`参数代替`DIRECTORY`，限制该文件无论作用域如何只被包含一次。

为了了解如何使用`git.cmake`模块文件，让我们一起查看`chapter13/ex01_git_utility`的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.21)
project(
  ch13_ex01_git_module
  VERSION 1.0
  DESCRIPTION "Chapter 13 Example 01, git utility module
    example"
  LANGUAGES CXX)
# Include the git.cmake module.
# Full relative path is given, since .cmake/ is not in the
  CMAKE_MODULE_PATH
include(.cmake/git.cmake)
git_get_branch_name(current_branch_name)
git_get_head_commit_hash(current_head)
git_get_config_value("user.name" current_user_name)
git_get_config_value("user.email" current_user_email)
message(STATUS "-----------------------------------------")
message(STATUS "VCS (git) info:")
message(STATUS "\tBranch: ${current_branch_name}")
message(STATUS "\tCommit hash: ${current_head}")
message(STATUS "\tAuthor name: ${current_user_name}")
message(STATUS "\tAuthor e-mail: ${current_user_email}")
message(STATUS "-----------------------------------------")
```

`CMakeLists.txt`文件通过指定模块文件的完整相对路径来包含`git.cmake`文件。该模块提供的`git_get_branch_name`、`git_get_head_commit_hash`和`git_get_config_value`宏分别用于获取分支名称、提交哈希值、作者名称和电子邮件地址，并将其存储到`current_branch_name`、`current_head`、`current_user_name`和`current_user_email`变量中。最后，通过`message(…)`命令将这些变量打印到屏幕上。让我们配置示例项目，看看我们刚刚编写的 Git 模块是否按预期工作：

```cpp
cd chapter13/ex01_git_utility/
cmake -S ./ -B ./build
```

命令的输出应类似于以下内容：

```cpp
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- -------------------------------------------
-- VCS (git) info:
--      Branch: chapter-development/chapter-13
--      Commit hash: 1d5a32649e74e4132e7b66292ab23aae
          ed327fdc
--      Author name: Mustafa Kemal GILOR
--      Author e-mail: mustafagilor@gmail.com
-- -------------------------------------------
-- Configuring done
-- Generating done
-- Build files have been written to:
/home/toor/workspace/ CMake-Best-Practices---2nd-Edition/chapter13
/ex01_git_utility/build
```

如我们所见，我们成功地从`git`命令中获取了信息。我们的第一个 CMake 模块按预期工作。

## 案例研究——处理项目元数据文件

让我们继续另一个例子。假设我们有一个环境文件，其中每行包含一个键值对。在项目中包含外部文件以存储有关项目的一些元数据（例如项目版本和依赖关系）并不罕见。该文件可以有不同的格式，例如 JSON 格式或换行分隔的键值对，如我们在此示例中所见。当前任务是创建一个工具模块，读取环境变量文件并为文件中的每个键值对定义一个 CMake 变量。文件的内容将类似于以下内容：

```cpp
KEY1="Value1"
KEY2="Value2"
```

对于本节，我们将参考`chapter13/ex02_envfile_utility`示例。让我们从检查`.cmake/envfile-utils.cmake`的内容开始：

```cpp
include_guard(DIRECTORY)
function(read_environment_file ENVIRONMENT_FILE_NAME)
    file(STRINGS ${ENVIRONMENT_FILE_NAME} KVP_LIST ENCODING
      UTF-8)
    foreach(ENV_VAR_DECL IN LISTS KVP_LIST)
        string(STRIP ENV_VAR_DECL ${ENV_VAR_DECL})
        string(LENGTH ENV_VAR_DECL ENV_VAR_DECL_LEN)
        if(ENV_VAR_DECL_LEN EQUAL 0)
            continue()
        endif()
        string(SUBSTRING ${ENV_VAR_DECL} 0 1
          ENV_VAR_DECL_FC)
        if(ENV_VAR_DECL_FC STREQUAL "#")
            continue()
        endif()
        string(REPLACE "=" ";" ENV_VAR_SPLIT
          ${ENV_VAR_DECL})
        list(GET ENV_VAR_SPLIT 0 ENV_VAR_NAME)
        list(GET ENV_VAR_SPLIT 1 ENV_VAR_VALUE)
        string(REPLACE "\"" "" ENV_VAR_VALUE
          ${ENV_VAR_VALUE})
        set(${ENV_VAR_NAME} ${ENV_VAR_VALUE} PARENT_SCOPE)
    endforeach()
endfunction()
```

`envfile-utils.cmake`工具模块包含一个函数`read_environment_file`，该函数读取一个键值对列表格式的环境文件。该函数将文件中的所有行读取到`KVP_LIST`变量中，然后遍历所有行。每一行都通过（`=`）等号符号进行分割，等号左边的部分作为变量名，右边的部分作为变量值，将每个键值对定义为一个 CMake 变量。空行和注释行会被跳过。至于模块的使用情况，让我们看看`chapter13/ex02_envfile_utility/CMakeLists.txt`文件：

```cpp
# Add .cmake folder to the module path, so subsequent
  include() calls
# can directly include modules under .cmake/ folder by
  specifying the name only.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${PROJECT_SOURCE_DIR}/.cmake/)
add_subdirectory(test-executable)
```

你可能已经注意到，`.cmake`文件夹被添加到了`CMAKE_MODULE_PATH`变量中。`CMAKE_MODULE_PATH`变量是`include(…)`指令将在其中搜索的路径集合。默认情况下，它是空的。这允许我们直接按名称在当前和子`CMakeLists.txt`文件中包含`envfile-utils`模块。最后，让我们看一下`chapter13/ex02_envfile_utility/test-executable/CMakeLists.txt`文件：

```cpp
# ....
# Include the module by name
include(envfile-utils)
read_environment_file("${PROJECT_SOURCE_DIR}/
  variables.env")
add_executable(ch13_ex02_envfile_utility_test)
target_sources(ch13_ex02_envfile_utility_test PRIVATE
  test.cpp)
target_compile_features(ch13_ex02_envfile_utility_test
  PRIVATE cxx_std_11)
target_compile_definitions(ch13_ex02_envfile_utility_test
  PRIVATE TEST_PROJECT_VERSION="${TEST_PROJECT_VERSION}"
    TEST_PROJECT_AUTHOR="${TEST_PROJECT_AUTHOR}")
```

如你所见，`envfile-utils`环境文件读取模块按名称被包含。这是因为包含`envfile-utils.cmake`文件的文件夹之前已经添加到`CMAKE_MODULE_PATH`变量中。`read_environment_file()`函数被调用来读取同一文件夹中的`variables.env`文件。`variables.env`文件包含以下键值对：

```cpp
# This file contains some metadata about the project
TEST_PROJECT_VERSION="1.0.2"
TEST_PROJECT_AUTHOR="CBP Authors"
```

因此，在调用`read_environment_file()`函数之后，我们期望`TEST_PROJECT_VERSION`和`TEST_PROJECT_AUTHOR`变量在当前 CMake 作用域中定义，并且它们的相应值在文件中指定。为了验证这一点，定义了一个名为`ch13_ex02_envfile_utility_test`的可执行目标，并将`TEST_PROJECT_VERSION`和`TEST_PROJECT_AUTHOR`变量作为宏定义传递给该目标。最后，目标的源文件`test.cpp`将`TEST_PROJECT_VERSION`和`TEST_PROJECT_AUTHOR`宏定义打印到控制台：

```cpp
#include <cstdio>
int main(void) {
    std::printf("Version `%s`, author `%s`\n",
      TEST_PROJECT_VERSION, TEST_PROJECT_AUTHOR);
}
```

好的——让我们编译并运行应用程序，看看它是否有效：

```cpp
cd chapter13/ex02_envfile_utility
cmake -S ./ -B ./build
cmake --build build
./build/test-executable/ch13_ex02_envfile_utility_test
# Will output: Version `1.0.2`, author `CBP Authors`
```

正如我们所看到的，我们已成功地从源代码树中读取了一个键值对格式的文件，并将每个键值对定义为 CMake 变量，然后将这些变量作为宏定义暴露给我们的应用程序。

虽然编写 CMake 模块非常直接，但还是有一些额外的建议需要考虑：

+   始终为函数/宏使用唯一的名称

+   为所有模块函数/宏使用一个共同的前缀

+   避免为非函数范围的变量使用常量名称

+   使用 `include_guard()` 来保护您的模块

+   如果您的模块输出消息，请为模块提供静默模式

+   不要暴露模块的内部实现

+   对于简单的命令包装器使用宏，对于其他情况使用函数

说到这里，我们结束了本章这一部分的内容。接下来，我们将探讨如何在项目间共享 CMake 模块。

## 关于项目间共享 CMake 模块的建议

共享 CMake 模块的推荐方式是维护一个独立的 CMake 模块项目，然后将该项目作为外部资源引入，可以通过 Git 子模块/子树或 CMake 的`FetchContent`进行，如在*第五章*中所描述，*集成第三方库和依赖管理*。在使用 `FetchContent` 时，可以通过设置 `SOURCE_DIR` 属性，并将 `CMAKE_MODULE_PATH` 设置为指定路径，轻松集成外部模块。这样，所有可重用的 CMake 工具可以集中在一个项目中进行维护，并可以传播到所有下游项目。将 CMake 模块放入在线 Git 托管平台（如 GitHub 或 GitLab）中的仓库，将使大多数人方便使用该模块。由于 CMake 支持直接从 Git 拉取内容，使用共享模块将变得非常简单。

为了演示如何使用外部 CMake 模块项目，我们将使用一个名为 `hadouken` 的开源 CMake 工具模块项目（[`github.com/mustafakemalgilor/hadouken`](https://github.com/mustafakemalgilor/hadouken)）。该项目包含用于工具集成、目标创建和特性检查的 CMake 工具模块。

对于这一部分，我们将按照`chapter13/ex03_external_cmake_module`示例进行操作。该示例将获取`hadouken`：

```cpp
include(FetchContent)
FetchContent_Declare(hadouken
    GIT_REPOSITORY https://github.com/mustafakemalgilor
      /hadouken.git
    GIT_TAG        7d0447fcadf8e93d25f242b9bb251ecbcf67f8cb
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.hadouken"
)
FetchContent_MakeAvailable(hadouken)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.hadouken/cmake/modules)
include(core/MakeTarget)
```

在前面的示例中，我们使用了 `FetchContent_Declare` 和 `FetchContent_MakeAvailable` 将 `hadouken` 拉取到我们的项目中，并将其放置在 `.hadouken` 文件夹中的构建目录里。然后，将 `hadouken` 项目的模块目录添加到 `CMAKE_MODULE_PATH` 中，通过 `include(…)` 指令使用 `hadouken` 项目的 CMake 工具模块。这样，我们就能访问由导入的 CMake 模块提供的 `make_target()` 宏。我们已经共同完成了这一章的内容。接下来，我们将总结本章所学的知识，并展望下一章的内容。

# 总结

在本章中，我们学习了如何构建 CMake 项目以支持可重用性。我们学习了如何实现 CMake 工具模块，如何共享它们，以及如何使用别人编写的工具模块。能够利用 CMake 模块使我们能够更好地组织项目，并更有效地与团队成员协作。掌握这些知识后，CMake 项目将变得更易于维护。CMake 项目中常用的可重用代码将发展成一个庞大的有用模块库，使得使用 CMake 编写项目变得更容易。

我想提醒您，CMake 是一种脚本语言，应该像对待脚本语言一样来使用。采用软件设计原则和模式，使 CMake 代码更具可维护性。将 CMake 代码组织成函数和模块。尽可能地重用和共享 CMake 代码。请不要忽视您的构建系统代码，否则您可能需要从头开始编写。

在下一章中，我们将学习如何优化和维护 CMake 项目。

下一章见！

# 问题

完成本章后，您应该能够回答以下问题：

1.  在 CMake 中，可重用性的最基本构建模块是什么？

1.  什么是 CMake 模块？

1.  如何使用 CMake 模块？

1.  `CMAKE_MODULE_PATH`变量的用途是什么？

1.  分享 CMake 模块给不同项目的一种方式是什么？

1.  在 CMake 中，函数和宏的主要区别是什么？

# 答案

1.  函数和宏。

1.  CMake 模块是一个逻辑实体，包含 CMake 代码、函数和宏，用于特定目的。

1.  通过将其包含在所需的作用域中。

1.  要将额外的路径添加到`include(…)`指令的搜索路径中。

1.  通过使用 Git 子模块/子树或 CMake 的`FetchContent`/`ExternalProject`模块。

1.  函数定义了一个新的变量作用域；宏则没有。
