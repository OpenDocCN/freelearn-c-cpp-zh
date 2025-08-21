# 第五章：创建和运行测试

在本章中，我们将介绍以下内容：

+   创建一个简单的单元测试

+   使用 Catch2 库定义单元测试

+   定义单元测试并链接到 Google Test

+   定义单元测试并链接到 Boost 测试

+   使用动态分析检测内存缺陷

+   测试预期失败

+   为长时间测试设置超时

+   并行运行测试

+   运行测试的子集

+   使用测试夹具

# 引言

测试是代码开发工具箱的核心组成部分。通过使用单元和集成测试进行自动化测试，不仅可以帮助开发者在早期检测功能回归，还可以作为新加入项目的开发者的起点。它可以帮助新开发者提交代码变更，并确保预期的功能得以保留。对于代码的用户来说，自动化测试在验证安装是否保留了代码功能方面至关重要。从一开始就为单元、模块或库使用测试的一个好处是，它可以引导程序员编写更加模块化和不那么复杂的代码结构，采用纯粹的、函数式的风格，最小化并局部化全局变量和全局状态。

在本章中，我们将演示如何将测试集成到 CMake 构建结构中，使用流行的测试库和框架，并牢记以下目标：

+   让用户、开发者和持续集成服务轻松运行测试套件。在使用 Unix Makefiles 时，应该简单到只需输入`make test`。

+   通过最小化总测试时间来高效运行测试，以最大化测试经常运行的概率——理想情况下，每次代码更改后都进行测试。

# 创建一个简单的单元测试

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-01)找到，并包含一个 C++示例。该示例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在本食谱中，我们将介绍使用 CTest 进行单元测试，CTest 是作为 CMake 一部分分发的测试工具。为了保持对 CMake/CTest 方面的关注并最小化认知负荷，我们希望尽可能简化要测试的代码。我们的计划是编写并测试能够求和整数的代码，仅此而已。就像在小学时，我们在学会加法后学习乘法和除法一样，此时，我们的示例代码只会加法，并且只会理解整数；它不需要处理浮点数。而且，就像年轻的卡尔·弗里德里希·高斯被他的老师测试从 1 到 100 求和所有自然数一样，我们将要求我们的代码做同样的事情——尽管没有使用高斯所用的聪明分组技巧。为了展示 CMake 对实现实际测试的语言没有任何限制，我们将不仅使用 C++可执行文件，还使用 Python 脚本和 shell 脚本来测试我们的代码。为了简单起见，我们将不使用任何测试库来完成这个任务，但我们将在本章后面的食谱中介绍 C++测试框架。

# 准备就绪

我们的代码示例包含三个文件。实现源文件`sum_integers.cpp`负责对整数向量进行求和，并返回总和：

```cpp
#include "sum_integers.hpp"

#include <vector>

int sum_integers(const std::vector<int> integers) {
  auto sum = 0;
  for (auto i : integers) {
    sum += i;
  }
  return sum;
}
```

对于这个例子，无论这是否是最优雅的向量求和实现方式都无关紧要。接口被导出到我们的示例库中的`sum_integers.hpp`，如下所示：

```cpp
#pragma once

#include <vector>

int sum_integers(const std::vector<int> integers);
```

最后，`main.cpp`中定义了主函数，它从`argv[]`收集命令行参数，将它们转换成一个整数向量，调用`sum_integers`函数，并将结果打印到输出：

```cpp
#include "sum_integers.hpp"

#include <iostream>
#include <string>
#include <vector>

// we assume all arguments are integers and we sum them up
// for simplicity we do not verify the type of arguments
int main(int argc, char *argv[]) {

  std::vector<int> integers;
  for (auto i = 1; i < argc; i++) {
    integers.push_back(std::stoi(argv[i]));
  }
  auto sum = sum_integers(integers);

  std::cout << sum << std::endl;
}
```

我们的目标是使用 C++可执行文件（`test.cpp`）、Bash shell 脚本（`test.sh`）和 Python 脚本（`test.py`）来测试这段代码，以证明 CMake 并不真正关心我们偏好哪种编程或脚本语言，只要实现能够返回零或非零值，CMake 可以将其解释为成功或失败，分别。

在 C++示例（`test.cpp`）中，我们通过调用`sum_integers`验证 1 + 2 + 3 + 4 + 5 等于 15：

```cpp
#include "sum_integers.hpp"

#include <vector>

int main() {
  auto integers = {1, 2, 3, 4, 5};

  if (sum_integers(integers) == 15) {
    return 0;
  } else {
    return 1;
  }
}
```

Bash shell 脚本测试示例调用可执行文件，该文件作为位置参数接收：

```cpp
#!/usr/bin/env bash

EXECUTABLE=$1

OUTPUT=$($EXECUTABLE 1 2 3 4)

if [ "$OUTPUT" = "10" ]
then
    exit 0
else
    exit 1
fi
```

此外，Python 测试脚本直接调用可执行文件（使用`--executable`命令行参数传递），并允许它使用`--short`命令行参数执行：

```cpp
import subprocess
import argparse

# test script expects the executable as argument
parser = argparse.ArgumentParser()
parser.add_argument('--executable',
                    help='full path to executable')
parser.add_argument('--short',
                    default=False,
                    action='store_true',
                    help='run a shorter test')
args = parser.parse_args()

def execute_cpp_code(integers):
    result = subprocess.check_output([args.executable] + integers)
    return int(result)

if args.short:
    # we collect [1, 2, ..., 100] as a list of strings
    result = execute_cpp_code([str(i) for i in range(1, 101)])
    assert result == 5050, 'summing up to 100 failed'
else:
    # we collect [1, 2, ..., 1000] as a list of strings
    result = execute_cpp_code([str(i) for i in range(1, 1001)])
    assert result == 500500, 'summing up to 1000 failed'
```

# 如何操作

现在我们将逐步描述如何为我们的项目设置测试，如下所示：

1.  对于这个例子，我们需要 C++11 支持、一个可用的 Python 解释器以及 Bash shell：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonInterp REQUIRED)
find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
```

1.  然后我们定义了库、主可执行文件的依赖项以及测试可执行文件：

```cpp
# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)
```

```cpp
# testing binary
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test sum_integers)
```

1.  最后，我们开启测试功能并定义了四个测试。最后两个测试调用同一个 Python 脚本；首先是没有任何命令行参数，然后是使用`--short`：

```cpp
enable_testing()

add_test(
  NAME bash_test
  COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.sh $<TARGET_FILE:sum_up>
  )

add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )

add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --executable $<TARGET_FILE:sum_up>
  )

add_test(
  NAME python_test_short
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --short --executable $<TARGET_FILE:sum_up>
  )
```

1.  现在，我们准备好配置和构建代码了。首先，我们手动测试它：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./sum_up 1 2 3 4 5

15
```

1.  然后，我们可以使用`ctest`运行测试集。

```cpp
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-01/cxx-example/build
    Start 1: bash_test
1/4 Test #1: bash_test ........................ Passed 0.01 sec
    Start 2: cpp_test
2/4 Test #2: cpp_test ......................... Passed 0.00 sec
    Start 3: python_test_long
3/4 Test #3: python_test_long ................. Passed 0.06 sec
    Start 4: python_test_short
4/4 Test #4: python_test_short ................ Passed 0.05 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) = 0.12 sec
```

1.  您还应该尝试破坏实现，以验证测试集是否捕获了更改。

# 它是如何工作的

这里的两个关键命令是`enable_testing()`，它为这个目录及其所有子文件夹（在本例中，整个项目，因为我们将其放在主`CMakeLists.txt`中）启用测试，以及`add_test()`，它定义一个新测试并设置测试名称和运行命令；例如：

```cpp
add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

在前面的示例中，我们使用了一个生成器表达式：`$<TARGET_FILE:cpp_test>`。生成器表达式是在**构建系统生成时间**评估的表达式。我们将在第五章，*配置时间和构建时间操作*，第 9 个配方，*使用生成器表达式微调配置和编译*中更详细地返回生成器表达式。目前，我们可以声明`$<TARGET_FILE:cpp_test>`将被替换为`cpp_test`可执行目标的完整路径。

生成器表达式在定义测试的上下文中非常方便，因为我们不必将可执行文件的位置和名称硬编码到测试定义中。以可移植的方式实现这一点将非常繁琐，因为可执行文件的位置和可执行文件后缀（例如，Windows 上的`.exe`后缀）可能在操作系统、构建类型和生成器之间有所不同。使用生成器表达式，我们不必明确知道位置和名称。

还可以向测试命令传递参数以运行；例如：

```cpp
add_test(
  NAME python_test_short
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --short --executable $<TARGET_FILE:sum_up>
  )
```

在本例中，我们按顺序运行测试（第 8 个配方，*并行运行测试*，将向您展示如何通过并行执行测试来缩短总测试时间），并且测试按定义的顺序执行（第 9 个配方，*运行测试子集*，将向您展示如何更改顺序或运行测试子集）。程序员负责定义实际的测试命令，该命令可以用操作系统环境支持的任何语言编程。CTest 唯一关心的是决定测试是否通过或失败的测试命令的返回代码。CTest 遵循标准约定，即零返回代码表示成功，非零返回代码表示失败。任何可以返回零或非零的脚本都可以用来实现测试用例。

既然我们知道如何定义和执行测试，了解如何诊断测试失败也很重要。为此，我们可以向代码中引入一个错误，并让所有测试失败：

```cpp
    Start 1: bash_test
1/4 Test #1: bash_test ........................***Failed 0.01 sec
    Start 2: cpp_test
2/4 Test #2: cpp_test .........................***Failed 0.00 sec
    Start 3: python_test_long
3/4 Test #3: python_test_long .................***Failed 0.06 sec
    Start 4: python_test_short
4/4 Test #4: python_test_short ................***Failed 0.06 sec

0% tests passed, 4 tests failed out of 4

Total Test time (real) = 0.13 sec
The following tests FAILED:
    1 - bash_test (Failed)
    2 - cpp_test (Failed)
    3 - python_test_long (Failed)
    4 - python_test_short (Failed)
Errors while running CTest
```

如果我们希望了解更多信息，可以检查文件`Testing/Temporary/LastTestsFailed.log`。该文件包含测试命令的完整输出，是进行事后分析时的第一个查看地点。通过使用以下 CLI 开关，可以从 CTest 获得更详细的测试输出：

+   `--output-on-failure`：如果测试失败，将打印测试程序产生的任何内容到屏幕上。

+   `-V`：将启用测试的详细输出。

+   `-VV`：启用更详细的测试输出。

CTest 提供了一个非常方便的快捷方式，可以仅重新运行先前失败的测试；使用的 CLI 开关是`--rerun-failed`，这在调试过程中证明极其有用。

# 还有更多内容。

考虑以下定义：

```cpp
add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --executable $<TARGET_FILE:sum_up>
  )
```

前面的定义可以通过显式指定脚本将在其中运行的`WORKING_DIRECTORY`来重新表达，如下所示：

```cpp
add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} test.py --executable $<TARGET_FILE:sum_up>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

我们还将提到，测试名称可以包含`/`字符，这在按名称组织相关测试时可能很有用；例如：

```cpp
add_test(
  NAME python/long
  COMMAND ${PYTHON_EXECUTABLE} test.py --executable $<TARGET_FILE:sum_up>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

有时，我们需要为测试脚本设置环境变量。这可以通过`set_tests_properties`实现。

```cpp
set_tests_properties(python_test
  PROPERTIES 
    ENVIRONMENT
      ACCOUNT_MODULE_PATH=${CMAKE_CURRENT_SOURCE_DIR}
      ACCOUNT_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/account/account.h
      ACCOUNT_LIBRARY_FILE=$<TARGET_FILE:account>
  )
```

这种方法可能并不总是跨不同平台都健壮，但 CMake 提供了一种绕过这种潜在健壮性不足的方法。以下代码片段等同于上述代码片段，并通过`CMAKE_COMMAND`预先添加环境变量，然后执行实际的 Python 测试脚本：

```cpp
add_test(
  NAME
    python_test
  COMMAND
    ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=${CMAKE_CURRENT_SOURCE_DIR}
                            ACCOUNT_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/account/account.h
                            ACCOUNT_LIBRARY_FILE=$<TARGET_FILE:account>
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/account/test.py
  )
```

再次注意，使用生成器表达式`$<TARGET_FILE:account>`来传递库文件的位置，而无需显式硬编码路径。

我们使用`ctest`命令执行了测试集，但 CMake 还将为生成器创建目标（对于 Unix Makefile 生成器使用`make test`，对于 Ninja 工具使用`ninja test`，或对于 Visual Studio 使用`RUN_TESTS`）。这意味着还有另一种（几乎）便携的方式来运行测试步骤：

```cpp
$ cmake --build . --target test
```

不幸的是，在使用 Visual Studio 生成器时这会失败，我们必须使用`RUN_TESTS`代替：

```cpp
$ cmake --build . --target RUN_TESTS
```

`ctest`命令提供了丰富的命令行参数。其中一些将在后面的食谱中探讨。要获取完整列表，请尝试`ctest --help`。命令`cmake --help-manual ctest`将输出完整的 CTest 手册到屏幕上。

# 使用 Catch2 库定义单元测试

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-02)获取，并包含一个 C++示例。该食谱适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在前一个配方中，我们在`test.cpp`中使用整数返回码来表示成功或失败。这对于简单的测试来说是可以的，但通常我们希望使用一个提供基础设施的测试框架，以便运行更复杂的测试，包括固定装置、与数值容差的比较，以及如果测试失败时更好的错误报告。一个现代且流行的测试库是 Catch2（[`github.com/catchorg/Catch2`](https://github.com/catchorg/Catch2)）。这个测试框架的一个很好的特点是它可以作为单个头文件库包含在你的项目中，这使得编译和更新框架特别容易。在本配方中，我们将使用 CMake 与 Catch2 结合，测试在前一个配方中介绍的求和代码。

# 准备就绪

我们将保持`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前的配方不变，但将更新`test.cpp`：

```cpp
#include "sum_integers.hpp"

// this tells catch to provide a main()
// only do this in one cpp file
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>

TEST_CASE("Sum of integers for a short vector", "[short]") {
  auto integers = {1, 2, 3, 4, 5};
  REQUIRE(sum_integers(integers) == 15);
}

TEST_CASE("Sum of integers for a longer vector", "[long]") {
  std::vector<int> integers;
  for (int i = 1; i < 1001; ++i) {
    integers.push_back(i);
  }
  REQUIRE(sum_integers(integers) == 500500);
}
```

我们还需要`catch.hpp`头文件，可以从[`github.com/catchorg/Catch2`](https://github.com/catchorg/Catch2)（我们使用了 2.0.1 版本）下载，并将其放置在项目根目录中，与`test.cpp`并列。

# 如何做

为了使用 Catch2 库，我们将修改前一个配方的`CMakeLists.txt`，执行以下步骤：

1.  我们可以保持`CMakeLists.txt`的大部分内容不变：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-02 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)

# testing binary
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test sum_integers)
```

1.  与前一个配方相比，唯一的改变是删除所有测试，只保留一个，并重命名它（以明确我们改变了什么）。请注意，我们向我们的单元测试可执行文件传递了`--success`选项。这是 Catch2 的一个选项，即使在成功时也会从测试中产生输出：

```cpp
enable_testing()

add_test(
  NAME catch_test
  COMMAND $<TARGET_FILE:cpp_test> --success
  )
```

1.  就这样！让我们配置、构建并测试。测试将使用 CTest 中的`-VV`选项运行，以从单元测试可执行文件获取输出：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest -V

UpdateCTestConfiguration from :/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/DartConfiguration.tcl
UpdateCTestConfiguration from :/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/DartConfiguration.tcl
Test project /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
 Start 1: catch_test

1: Test command: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/cpp_test "--success"
1: Test timeout computed to be: 10000000
1: 
1: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1: cpp_test is a Catch v2.0.1 host application.
1: Run with -? for options
1: 
1: ----------------------------------------------------------------
1: Sum of integers for a short vector
1: ----------------------------------------------------------------
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:10
1: ...................................................................
1: 
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:12: 
1: PASSED:
1: REQUIRE( sum_integers(integers) == 15 )
1: with expansion:
1: 15 == 15
1: 
1: ----------------------------------------------------------------
1: Sum of integers for a longer vector
1: ----------------------------------------------------------------
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:15
1: ...................................................................
1: 
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:20: 
1: PASSED:
1: REQUIRE( sum_integers(integers) == 500500 )
1: with expansion:
1: 500500 (0x7a314) == 500500 (0x7a314)
1: 
1: ===================================================================
1: All tests passed (2 assertions in 2 test cases)
1:
1/1 Test #1: catch_test ....................... Passed 0.00 s

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.00 sec
```

1.  我们也可以直接尝试运行`cpp_test`二进制文件，并直接从 Catch2 看到输出：

```cpp
$ ./cpp_test --success

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cpp_test is a Catch v2.0.1 host application.
Run with -? for options

-------------------------------------------------------------------
Sum of integers for a short vector
-------------------------------------------------------------------
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:10
...................................................................

/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:12: 
PASSED:
  REQUIRE( sum_integers(integers) == 15 )
with expansion:
  15 == 15

-------------------------------------------------------------------
Sum of integers for a longer vector
-------------------------------------------------------------------
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:15
...................................................................

/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:20: 
PASSED:
  REQUIRE( sum_integers(integers) == 500500 )
with expansion:
  500500 (0x7a314) == 500500 (0x7a314)

===================================================================
All tests passed (2 assertions in 2 test cases)
```

1.  Catch 将生成一个具有命令行界面的可执行文件。我们邀请你也尝试执行以下命令，以探索单元测试框架提供的选项：

```cpp
$ ./cpp_test --help
```

# 它是如何工作的

由于 Catch2 是一个单头文件框架，因此不需要定义和构建额外的目标。我们只需要确保 CMake 能够找到`catch.hpp`来构建`test.cpp`。为了方便，我们将其放置在与`test.cpp`相同的目录中，但我们也可以选择不同的位置，并使用`target_include_directories`指示该位置。另一种方法是将头文件包装成一个`INTERFACE`库。这可以按照 Catch2 文档中的说明进行（`https://github.com/catchorg/Catch2/blob/master/docs/build-systems.md#cmake`）：

```cpp
# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})
```

那么我们将按照以下方式链接库：

```cpp
target_link_libraries(cpp_test Catch)
```

我们从第一章，*从简单可执行文件到库*中的食谱 3，*构建和链接静态和共享库*的讨论中回忆起，`INTERFACE`库是 CMake 提供的伪目标，对于指定项目外部的目标使用要求非常有用。

# 还有更多

这是一个简单的例子，重点在于 CMake。当然，Catch2 提供了更多功能。要获取 Catch2 框架的完整文档，请访问[`github.com/catchorg/Catch2`](https://github.com/catchorg/Catch2)。

# 另请参阅

Catch2 代码仓库包含一个由贡献的 CMake 函数，用于解析 Catch 测试并自动创建 CMake 测试，而无需显式键入`add_test()`函数；请参阅[`github.com/catchorg/Catch2/blob/master/contrib/ParseAndAddCatchTests.cmake`](https://github.com/catchorg/Catch2/blob/master/contrib/ParseAndAddCatchTests.cmake)。

# 定义单元测试并链接 Google Test

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-03)找到，并包含一个 C++示例。本食谱适用于 CMake 版本 3.11（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。代码仓库还包含一个与 CMake 3.5 兼容的示例。

在本食谱中，我们将演示如何使用 CMake 和 Google Test 框架实现单元测试。与之前的食谱不同，Google Test 框架不仅仅是一个头文件；它是一个包含多个需要构建和链接的文件的库。我们可以将这些文件与我们的代码项目放在一起，但为了让代码项目更轻量级，我们将在配置时下载 Google Test 源代码的明确定义版本，然后构建框架并与之链接。我们将使用相对较新的`FetchContent`模块（自 CMake 版本 3.11 起可用）。我们将在第八章，*超级构建模式*中重新讨论`FetchContent`，在那里我们将讨论模块在幕后是如何工作的，以及我们还将说明如何使用`ExternalProject_Add`来模拟它。本食谱的灵感来自（并改编自）[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)的示例。

# 准备工作

我们将保持`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前的食谱不变，但将更新`test.cpp`源代码，如下所示：

```cpp
#include "sum_integers.hpp"
#include "gtest/gtest.h"

#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(example, sum_zero) {
  auto integers = {1, -1, 2, -2, 3, -3};
  auto result = sum_integers(integers);
  ASSERT_EQ(result, 0);
}

TEST(example, sum_five) {
  auto integers = {1, 2, 3, 4, 5};
  auto result = sum_integers(integers);
  ASSERT_EQ(result, 15);
}
```

如前述代码所示，我们选择不在我们的代码项目仓库中显式放置`gtest.h`或其他 Google Test 源文件，而是通过使用`FetchContent`模块在配置时下载它们。

# 如何操作

以下步骤描述了如何逐步设置`CMakeLists.txt`，以使用 GTest 编译可执行文件及其相应的测试：

1.  `CMakeLists.txt`的开头与前两个配方相比大部分未变，只是我们需要 CMake 3.11 以访问`FetchContent`模块：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# project name and language
project(recipe-03 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)
```

1.  然后我们引入了一个 if 语句，检查`ENABLE_UNIT_TESTS`。默认情况下它是`ON`，但我们希望有可能将其关闭，以防我们没有网络下载 Google Test 源码：

```cpp
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if(ENABLE_UNIT_TESTS)
  # all the remaining CMake code will be placed here
endif()
```

1.  在 if 语句内部，我们首先包含`FetchContent`模块，声明一个新的要获取的内容，并查询其属性：

```cpp
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)

FetchContent_GetProperties(googletest)
```

1.  如果内容尚未填充（获取），我们获取并配置它。这将添加一些我们可以链接的目标。在本例中，我们对`gtest_main`感兴趣。该示例还包含一些使用 Visual Studio 编译的解决方法：

```cpp
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Prevent GoogleTest from using PThreads
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

  # adds the targers: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )

  # Silence std::tr1 warning on MSVC
  if(MSVC)
    foreach(_tgt gtest gtest_main gmock gmock_main)
      target_compile_definitions(${_tgt}
        PRIVATE
          "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
        )
    endforeach()
  endif()
endif()
```

1.  然后我们定义了`cpp_test`可执行目标，并使用`target_sources`命令指定其源文件，使用`target_link_libraries`命令指定其链接库：

```cpp
add_executable(cpp_test "")

target_sources(cpp_test
  PRIVATE
    test.cpp
  )

target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    gtest_main
  )
```

1.  最后，我们使用熟悉的`enable_testing`和`add_test`命令来定义单元测试：

```cpp
enable_testing()

add_test(
  NAME google_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

1.  现在，我们准备好配置、构建和测试项目了：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-cookbook/chapter-04/recipe-03/cxx-example/build
    Start 1: google_test
1/1 Test #1: google_test ...................... Passed 0.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.00 sec
```

1.  我们也可以尝试直接运行`cpp_test`，如下所示：

```cpp
$ ./cpp_test

[==========] Running 2 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 2 tests from example
[ RUN      ] example.sum_zero
[       OK ] example.sum_zero (0 ms)
[ RUN      ] example.sum_five
[       OK ] example.sum_five (0 ms)
[----------] 2 tests from example (0 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test case ran. (0 ms total)
[  PASSED  ] 2 tests.
```

# 它是如何工作的

`FetchContent`模块允许在配置时填充内容，*通过*任何`ExternalProject`模块支持的方法，并且已成为 CMake 3.11 版本的标准部分。而`ExternalProject_Add()`在构建时下载（如第八章，*超级构建模式*所示），`FetchContent`模块使内容立即可用，以便主项目和获取的外部项目（在本例中为 Google Test）可以在 CMake 首次调用时处理，并且可以使用`add_subdirectory`嵌套。

为了获取 Google Test 源码，我们首先声明了外部内容：

```cpp
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)
```

在这种情况下，我们获取了一个带有特定标签（`release-1.8.0`）的 Git 仓库，但我们也可以从 Subversion、Mercurial 或 HTTP(S)源获取外部项目。有关可用选项，请参阅[`cmake.org/cmake/help/v3.11/module/ExternalProject.html`](https://cmake.org/cmake/help/v3.11/module/ExternalProject.html)上相应`ExternalProject_Add`命令的选项。

我们在调用`FetchContent_Populate()`之前使用`FetchContent_GetProperties()`命令检查内容填充是否已经处理；否则，如果`FetchContent_Populate()`被调用多次，它会抛出一个错误。

`FetchContent_Populate(googletest)`命令填充源码并定义`googletest_SOURCE_DIR`和`googletest_BINARY_DIR`，我们可以使用它们来处理 Google Test 项目（使用`add_subdirectory()`，因为它恰好也是一个 CMake 项目）：

```cpp
add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
  )
```

上述定义了以下目标：`gtest`、`gtest_main`、`gmock`和`gmock_main`。在本示例中，我们只对`gtest_main`目标感兴趣，作为单元测试示例的库依赖项：

```cpp
target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    gtest_main
  )
```

在构建我们的代码时，我们可以看到它如何正确地触发了 Google Test 的配置和构建步骤。有一天，我们可能希望升级到更新的 Google Test 版本，我们可能需要更改的唯一一行是详细说明`GIT_TAG`的那一行。

# 还有更多

我们已经初步了解了`FetchContent`及其构建时的表亲`ExternalProject_Add`，我们将在第八章，*超级构建模式*中重新审视这些命令。对于可用选项的详细讨论，请参考[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)。

在本示例中，我们在配置时获取了源代码，但我们也可以在系统环境中安装它们，并使用`FindGTest`模块来检测库和头文件（[`cmake.org/cmake/help/v3.5/module/FindGTest.html`](https://cmake.org/cmake/help/v3.5/module/FindGTest.html)）。从版本 3.9 开始，CMake 还提供了一个`GoogleTest`模块（[`cmake.org/cmake/help/v3.9/module/GoogleTest.html`](https://cmake.org/cmake/help/v3.9/module/GoogleTest.html)），该模块提供了一个`gtest_add_tests`函数。这个函数可以用来自动添加测试，通过扫描源代码中的 Google Test 宏。

# 另请参阅

显然，Google Test 有许多超出本示例范围的功能，如[`github.com/google/googletest`](https://github.com/google/googletest)所列。

# 定义单元测试并链接到 Boost 测试

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-04)找到，并包含一个 C++示例。本示例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

Boost 测试是 C++社区中另一个非常流行的单元测试框架，在本示例中，我们将演示如何使用 Boost 测试对我们的熟悉求和示例代码进行单元测试。

# 准备工作

我们将保持`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前的示例不变，但我们将更新`test.cpp`作为使用 Boost 测试库的单元测试的简单示例：

```cpp
#include "sum_integers.hpp"

#include <vector>

#define BOOST_TEST_MODULE example_test_suite
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(add_example) {
  auto integers = {1, 2, 3, 4, 5};
  auto result = sum_integers(integers);
  BOOST_REQUIRE(result == 15);
}
```

# 如何操作

以下是使用 Boost 测试构建我们项目的步骤：

1.  我们从熟悉的`CMakeLists.txt`结构开始：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-04 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)
```

1.  我们检测 Boost 库并链接`cpp_test`：

```cpp
find_package(Boost 1.54 REQUIRED COMPONENTS unit_test_framework)

add_executable(cpp_test test.cpp)

target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    Boost::unit_test_framework
  )

# avoid undefined reference to "main" in test.cpp
target_compile_definitions(cpp_test
  PRIVATE
    BOOST_TEST_DYN_LINK
  )
```

1.  最后，我们定义单元测试：

```cpp
enable_testing()

add_test(
  NAME boost_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

1.  以下是我们需要配置、构建和测试代码的所有内容：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-04/cxx-example/build
    Start 1: boost_test
1/1 Test #1: boost_test ....................... Passed 0.01 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.01 sec

$ ./cpp_test

Running 1 test case...

*** No errors detected
```

# 工作原理

我们使用了`find_package`来检测 Boost 的`unit_test_framework`组件（请参阅第三章，*检测外部库和程序*，第八部分，*检测 Boost 库*）。我们坚持认为这个组件是`REQUIRED`，如果无法在系统环境中找到，配置将停止。`cpp_test`目标需要知道在哪里找到 Boost 头文件，并需要链接到相应的库；这两者都由`IMPORTED`库目标`Boost::unit_test_framework`提供，该目标由成功的`find_package`调用设置。我们从第一章，*从简单可执行文件到库*中的第三部分，*构建和链接静态和共享库*的讨论中回忆起，`IMPORTED`库是 CMake 提供的伪目标，用于表示预先存在的依赖关系及其使用要求。

# 还有更多内容

在本节中，我们假设 Boost 已安装在系统上。或者，我们可以在编译时获取并构建 Boost 依赖项（请参阅第八章，*超级构建模式*，第二部分，*使用超级构建管理依赖项：I. Boost 库*）。然而，Boost 不是一个轻量级依赖项。在我们的示例代码中，我们仅使用了最基本的基础设施，但 Boost 提供了丰富的功能和选项，我们将引导感兴趣的读者访问[`www.boost.org/doc/libs/1_65_1/libs/test/doc/html/index.html`](http://www.boost.org/doc/libs/1_65_1/libs/test/doc/html/index.html)。

# 使用动态分析检测内存缺陷

本节的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-05)找到，并提供了一个 C++示例。本节适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

内存缺陷，例如越界写入或读取内存，或者内存泄漏（已分配但从未释放的内存），可能会产生难以追踪的讨厌错误，因此尽早检测它们是有用的。Valgrind（[`valgrind.org`](http://valgrind.org)）是一个流行且多功能的工具，用于检测内存缺陷和内存泄漏，在本节中，我们将使用 Valgrind 来提醒我们使用 CMake/CTest 运行测试时的内存问题（请参阅第十四章，*测试仪表板*，以讨论相关的`AddressSanitizer`和`ThreadSanitizer`）。

# 准备就绪

对于本节，我们需要三个文件。第一个是我们希望测试的实现（我们可以将文件称为`leaky_implementation.cpp`）：

```cpp
#include "leaky_implementation.hpp"

int do_some_work() {

  // we allocate an array
  double *my_array = new double[1000];

  // do some work
  // ...

  // we forget to deallocate it
  // delete[] my_array;

  return 0;
}
```

我们还需要相应的头文件（`leaky_implementation.hpp`）：

```cpp
#pragma once

int do_some_work();
```

我们需要测试文件（`test.cpp`）：

```cpp
#include "leaky_implementation.hpp"

int main() {
  int return_code = do_some_work();

  return return_code;
}
```

我们期望测试通过，因为`return_code`被硬编码为`0`。然而，我们也希望检测到内存泄漏，因为我们忘记了释放`my_array`。

# 如何操作

以下是如何设置`CMakeLists.txt`以执行代码的动态分析：

1.  我们首先定义了最低 CMake 版本、项目名称、语言、目标和依赖项：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(example_library leaky_implementation.cpp)

```

```cpp

add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test example_library)
```

1.  然后，我们不仅定义了测试，还定义了`MEMORYCHECK_COMMAND`：

```cpp
find_program(MEMORYCHECK_COMMAND NAMES valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")

# add memcheck test action
include(CTest)

enable_testing()

add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

1.  运行测试集报告测试通过，如下所示：

```cpp
$ ctest 
Test project /home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build
    Start 1: cpp_test
1/1 Test #1: cpp_test ......................... Passed 0.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.00 sec
```

1.  现在，我们希望检查内存缺陷，并可以观察到内存泄漏被检测到：

```cpp
$ ctest -T memcheck

   Site: myhost
   Build name: Linux-c++
Create new tag: 20171127-1717 - Experimental
Memory check project /home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build
    Start 1: cpp_test
1/1 MemCheck #1: cpp_test ......................... Passed 0.40 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.40 sec
-- Processing memory checking output:
1/1 MemCheck: #1: cpp_test ......................... Defects: 1
MemCheck log files can be found here: ( * corresponds to test number)
/home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build/Testing/Temporary/MemoryChecker.*.log
Memory checking results:
Memory Leak - 1
```

1.  作为最后一步，你应该尝试修复内存泄漏，并验证`ctest -T memcheck`报告没有错误。

# 工作原理

我们使用`find_program(MEMORYCHECK_COMMAND NAMES valgrind)`来查找 Valgrind 并将其完整路径设置为`MEMORYCHECK_COMMAND`。我们还需要显式包含`CTest`模块以启用`memcheck`测试动作，我们可以通过使用`ctest -T memcheck`来使用它。此外，请注意我们能够使用`set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")`将选项传递给 Valgrind。内存检查步骤创建一个日志文件，可用于详细检查内存缺陷。

一些工具，如代码覆盖率和静态分析工具，可以类似地设置。然而，使用其中一些工具更为复杂，因为需要专门的构建和工具链。Sanitizers 就是一个例子。有关更多信息，请参阅[`github.com/arsenm/sanitizers-cmake`](https://github.com/arsenm/sanitizers-cmake)。此外，请查看第十四章，*测试仪表板*，以讨论`AddressSanitizer`和`ThreadSanitizer`。

# 还有更多

本食谱可用于向夜间测试仪表板报告内存缺陷，但我们在这里演示了此功能也可以独立于测试仪表板使用。我们将在第十四章，*测试仪表板*中重新讨论与 CDash 结合使用的情况。

# 另请参阅

有关 Valgrind 及其功能和选项的文档，请参阅[`valgrind.org`](http://valgrind.org)。

# 测试预期失败

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-06`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-06)找到。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

理想情况下，我们希望我们的所有测试在每个平台上都能始终通过。然而，我们可能想要测试在受控环境中是否会发生预期的失败或异常，在这种情况下，我们将预期的失败定义为成功的结果。我们相信，通常这应该是测试框架（如 Catch2 或 Google Test）的任务，它应该检查预期的失败并将成功报告给 CMake。但是，可能会有情况，你希望将测试的非零返回代码定义为成功；换句话说，你可能想要反转成功和失败的定义。在本节中，我们将展示这样的情况。

# 准备工作

本节的成分将是一个微小的 Python 脚本（`test.py`），它总是返回`1`，CMake 将其解释为失败：

```cpp
import sys

# simulate a failing test
sys.exit(1)
```

# 如何操作

逐步地，这是如何编写`CMakeLists.txt`来完成我们的任务：

1.  在本节中，我们不需要 CMake 提供任何语言支持，但我们需要找到一个可用的 Python 解释器：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES NONE)

find_package(PythonInterp REQUIRED)
```

1.  然后我们定义测试并告诉 CMake 我们期望它失败：

```cpp
enable_testing()

add_test(example ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py)

set_tests_properties(example PROPERTIES WILL_FAIL true)
```

1.  最后，我们验证它被报告为成功的测试，如下所示：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-06/example/build
    Start 1: example
1/1 Test #1: example .......................... Passed 0.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.01 sec
```

# 它是如何工作的

使用`set_tests_properties(example PROPERTIES WILL_FAIL true)`，我们将属性`WILL_FAIL`设置为`true`，这会反转成功/失败的状态。然而，这个功能不应该用来临时修复损坏的测试。

# 还有更多

如果你需要更多的灵活性，你可以结合使用测试属性`PASS_REGULAR_EXPRESSION`和`FAIL_REGULAR_EXPRESSION`与`set_tests_properties`。如果设置了这些属性，测试输出将被检查与作为参数给出的正则表达式列表进行匹配，如果至少有一个正则表达式匹配，则测试分别通过或失败。还有许多其他属性可以设置在测试上。可以在[`cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#properties-on-tests`](https://cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#properties-on-tests)找到所有可用属性的完整列表。

# 为长时间测试设置超时

本节的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-07`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-07)找到。本节适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

理想情况下，测试集应该只需要很短的时间，以激励开发者频繁运行测试集，并使得对每次提交（变更集）进行测试成为可能（或更容易）。然而，有些测试可能会耗时较长或卡住（例如，由于高文件 I/O 负载），我们可能需要实施超时机制来终止超时的测试，以免它们堆积起来延迟整个测试和部署流水线。在本节中，我们将展示一种实施超时的方法，可以为每个测试单独调整。

# 准备工作

本食谱的成分将是一个微小的 Python 脚本（`test.py`），它总是返回`0`。为了保持超级简单并专注于 CMake 方面，测试脚本除了等待两秒钟之外不做任何事情；但是，我们可以想象在现实生活中，这个测试脚本会执行更有意义的工作：

```cpp
import sys
import time

# wait for 2 seconds
time.sleep(2)

# report success
sys.exit(0)
```

# 如何操作

我们需要通知 CTest，如果测试超时，需要终止测试，如下所示：

1.  我们定义项目名称，启用测试，并定义测试：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-07 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()

# we expect this test to run for 2 seconds
add_test(example ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py)
```

1.  此外，我们为测试指定了一个`TIMEOUT`，并将其设置为 10 秒：

```cpp
set_tests_properties(example PROPERTIES TIMEOUT 10)
```

1.  我们知道如何配置和构建，我们期望测试通过：

```cpp
$ ctest 
Test project /home/user/cmake-recipes/chapter-04/recipe-07/example/build
    Start 1: example
1/1 Test #1: example .......................... Passed 2.01 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 2.01 sec
```

1.  现在，为了验证`TIMEOUT`是否有效，我们将`test.py`中的睡眠命令增加到 11 秒，并重新运行测试：

```cpp
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-07/example/build
    Start 1: example
1/1 Test #1: example ..........................***Timeout 10.01 sec

0% tests passed, 1 tests failed out of 1

Total Test time (real) = 10.01 sec

The following tests FAILED:
          1 - example (Timeout)
Errors while running CTest
```

# 工作原理

`TIMEOUT`是一个方便的属性，可用于通过使用`set_tests_properties`为单个测试指定超时。如果测试超过该时间，无论出于何种原因（测试停滞或机器太慢），测试都会被终止并标记为失败。

# 并行运行测试

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-08`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-08)找到。该食谱适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

大多数现代计算机都有四个或更多的 CPU 核心。CTest 的一个很棒的功能是，如果你有多个核心可用，它可以并行运行测试。这可以显著减少总测试时间，减少总测试时间才是真正重要的，以激励开发者频繁测试。在这个食谱中，我们将演示这个功能，并讨论如何优化你的测试定义以获得最大性能。

# 准备就绪

让我们假设我们的测试集包含标记为*a, b, ..., j*的测试，每个测试都有特定的持续时间：

| 测试 | 持续时间（以时间单位计） |
| --- | --- |
| *a, b, c, d* | 0.5 |
| *e, f, g* | 1.5 |
| *h* | 2.5 |
| *i* | 3.5 |
| *j* | 4.5 |

时间单位可以是分钟，但为了保持简单和短，我们将使用秒。为了简单起见，我们可以用一个 Python 脚本来表示消耗 0.5 时间单位的测试*a*：

```cpp
import sys
import time

# wait for 0.5 seconds
time.sleep(0.5)

# finally report success
sys.exit(0)
```

其他测试可以相应地表示。我们将把这些脚本放在`CMakeLists.txt`下面的一个目录中，目录名为`test`。

# 如何操作

对于这个食谱，我们需要声明一个测试列表，如下所示：

1.  `CMakeLists.txt`非常简短：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-08 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()

add_test(a ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/a.py)
add_test(b ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/b.py)
add_test(c ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/c.py)
add_test(d ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/d.py)
add_test(e ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/e.py)
add_test(f ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/f.py)
add_test(g ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/g.py)
add_test(h ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/h.py)
add_test(i ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/i.py)
add_test(j ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/j.py)
```

1.  我们可以使用`ctest`配置项目并运行测试，总共需要 17 秒：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

      Start 1: a
 1/10 Test #1: a ................................ Passed 0.51 sec
      Start 2: b
 2/10 Test #2: b ................................ Passed 0.51 sec
      Start 3: c
 3/10 Test #3: c ................................ Passed 0.51 sec
      Start 4: d
 4/10 Test #4: d ................................ Passed 0.51 sec
      Start 5: e
 5/10 Test #5: e ................................ Passed 1.51 sec
      Start 6: f
 6/10 Test #6: f ................................ Passed 1.51 sec
      Start 7: g
 7/10 Test #7: g ................................ Passed 1.51 sec
      Start 8: h
 8/10 Test #8: h ................................ Passed 2.51 sec
      Start 9: i
 9/10 Test #9: i ................................ Passed 3.51 sec
      Start 10: j
10/10 Test #10: j ................................ Passed 4.51 sec

100% tests passed, 0 tests failed out of 10

Total Test time (real) = 17.11 sec
```

1.  现在，如果我们碰巧有四个核心可用，我们可以在不到五秒的时间内将测试集运行在四个核心上：

```cpp
$ ctest --parallel 4

      Start 10: j
      Start 9: i
      Start 8: h
      Start 5: e
 1/10 Test #5: e ................................ Passed 1.51 sec
      Start 7: g
 2/10 Test #8: h ................................ Passed 2.51 sec
      Start 6: f
 3/10 Test #7: g ................................ Passed 1.51 sec
      Start 3: c
 4/10 Test #9: i ................................ Passed 3.63 sec
 5/10 Test #3: c ................................ Passed 0.60 sec
      Start 2: b
      Start 4: d
 6/10 Test #6: f ................................ Passed 1.51 sec
 7/10 Test #4: d ................................ Passed 0.59 sec
 8/10 Test #2: b ................................ Passed 0.59 sec
      Start 1: a
 9/10 Test #10: j ................................ Passed 4.51 sec
10/10 Test #1: a ................................ Passed 0.51 sec

100% tests passed, 0 tests failed out of 10

Total Test time (real) = 4.74 sec
```

# 工作原理

我们可以看到，在并行情况下，测试*j, i, h*和*e*同时开始。并行运行时总测试时间的减少可能是显著的。查看`ctest --parallel 4`的输出，我们可以看到并行测试运行从最长的测试开始，并在最后运行最短的测试。从最长的测试开始是一个非常好的策略。这就像打包搬家箱子：我们从较大的物品开始，然后用较小的物品填充空隙。比较在四个核心上从最长测试开始的*a-j*测试的堆叠，看起来如下：

```cpp
        --> time
core 1: jjjjjjjjj
core 2: iiiiiiibd
core 3: hhhhhggg
core 4: eeefffac
```

按照定义的顺序运行测试看起来如下：

```cpp
        --> time
core 1: aeeeiiiiiii
core 2: bfffjjjjjjjjj
core 3: cggg
core 4: dhhhhh
```

按照定义的顺序运行测试总体上需要更多时间，因为它让两个核心大部分时间处于空闲状态（这里，核心 3 和 4）。CMake 是如何知道哪些测试需要最长的时间？CMake 知道每个测试的时间成本，因为我们首先按顺序运行了测试，这记录了每个测试的成本数据在文件`Testing/Temporary/CTestCostData.txt`中，看起来如下：

```cpp
a 1 0.506776
b 1 0.507882
c 1 0.508175
d 1 0.504618
e 1 1.51006
f 1 1.50975
g 1 1.50648
h 1 2.51032
i 1 3.50475
j 1 4.51111
```

如果我们刚配置完项目就立即开始并行测试，它将按照定义的顺序运行测试，并且在四个核心上，总测试时间会明显更长。这对我们意味着什么？这是否意味着我们应该根据递减的时间成本来排序测试？这是一个选项，但事实证明还有另一种方法；我们可以自行指示每个测试的时间成本：

```cpp
add_test(a ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/a.py)
add_test(b ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/b.py)
add_test(c ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/c.py)
add_test(d ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/d.py)
set_tests_properties(a b c d PROPERTIES COST 0.5)

add_test(e ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/e.py)
add_test(f ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/f.py)
add_test(g ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/g.py)
set_tests_properties(e f g PROPERTIES COST 1.5)

add_test(h ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/h.py)
set_tests_properties(h PROPERTIES COST 2.5)

add_test(i ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/i.py)
set_tests_properties(i PROPERTIES COST 3.5)

add_test(j ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/j.py)
set_tests_properties(j PROPERTIES COST 4.5)
```

`COST`参数可以是估计值或从`Testing/Temporary/CTestCostData.txt`提取。

# 还有更多内容。

除了使用`ctest --parallel N`，你还可以使用环境变量`CTEST_PARALLEL_LEVEL`，并将其设置为所需的级别。

# 运行测试子集

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-09`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-09)找到。本示例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在前面的示例中，我们学习了如何借助 CMake 并行运行测试，并讨论了从最长的测试开始的优势。虽然这种策略可以最小化总测试时间，但在特定功能的代码开发或调试过程中，我们可能不希望运行整个测试集。我们可能更倾向于从最长的测试开始，特别是在调试由短测试执行的功能时。对于调试和代码开发，我们需要能够仅运行选定的测试子集。在本示例中，我们将介绍实现这一目标的策略。

# 准备工作

在本例中，我们假设总共有六个测试；前三个测试较短，名称分别为`feature-a`、`feature-b`和`feature-c`。我们还有三个较长的测试，名称分别为`feature-d`、`benchmark-a`和`benchmark-b`。在本例中，我们可以使用 Python 脚本来表示这些测试，其中我们可以调整睡眠时间：

```cpp
import sys
import time

# wait for 0.1 seconds
time.sleep(0.1)

# finally report success
sys.exit(0)
```

# 如何操作

以下是对我们的`CMakeLists.txt`内容的详细分解：

1.  我们从一个相对紧凑的`CMakeLists.txt`开始，定义了六个测试：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-09 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()

add_test(
  NAME feature-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-a.py
  )
add_test(
  NAME feature-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-b.py
  )
add_test(
  NAME feature-c
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-c.py
  )
add_test(
  NAME feature-d
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-d.py
  )

add_test(
  NAME benchmark-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark-a.py
  )
```

```cpp
add_test(
  NAME benchmark-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark-b.py
  )
```

1.  此外，我们将较短的测试标记为`"quick"`，将较长的测试标记为`"long"`：

```cpp
set_tests_properties(
  feature-a
  feature-b
  feature-c
  PROPERTIES
    LABELS "quick"
  )

set_tests_properties(
  feature-d
  benchmark-a
  benchmark-b
  PROPERTIES
    LABELS "long"
  )
```

1.  我们现在准备运行测试集，如下所示：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

    Start 1: feature-a
1/6 Test #1: feature-a ........................ Passed 0.11 sec
    Start 2: feature-b
2/6 Test #2: feature-b ........................ Passed 0.11 sec
    Start 3: feature-c
3/6 Test #3: feature-c ........................ Passed 0.11 sec
    Start 4: feature-d
4/6 Test #4: feature-d ........................ Passed 0.51 sec
    Start 5: benchmark-a
5/6 Test #5: benchmark-a ...................... Passed 0.51 sec
    Start 6: benchmark-b
6/6 Test #6: benchmark-b ...................... Passed 0.51 sec
```

```cpp
100% tests passed, 0 tests failed out of 6

Label Time Summary:
long = 1.54 sec*proc (3 tests)
quick = 0.33 sec*proc (3 tests)

Total Test time (real) = 1.87 sec
```

# 工作原理

现在每个测试都有一个名称和一个标签。在 CMake 中，所有测试都有编号，因此它们也具有唯一编号。定义了测试标签后，我们现在可以运行整个集合，也可以根据测试的名称（使用正则表达式）、标签或编号来运行测试。

通过名称运行测试（这里，我们运行所有名称匹配`feature`的测试）如下所示：

```cpp
$ ctest -R feature

    Start 1: feature-a
1/4 Test #1: feature-a ........................ Passed 0.11 sec
    Start 2: feature-b
2/4 Test #2: feature-b ........................ Passed 0.11 sec
    Start 3: feature-c
3/4 Test #3: feature-c ........................ Passed 0.11 sec
    Start 4: feature-d
4/4 Test #4: feature-d ........................ Passed 0.51 sec

100% tests passed, 0 tests failed out of 4
```

通过标签运行测试（这里，我们运行所有`long`测试）产生：

```cpp
$ ctest -L long

    Start 4: feature-d
1/3 Test #4: feature-d ........................ Passed 0.51 sec
    Start 5: benchmark-a
2/3 Test #5: benchmark-a ...................... Passed 0.51 sec
    Start 6: benchmark-b
3/3 Test #6: benchmark-b ...................... Passed 0.51 sec

100% tests passed, 0 tests failed out of 3
```

通过编号运行测试（这里，我们运行第 2 到第 4 个测试）得到：

```cpp
$ ctest -I 2,4

    Start 2: feature-b
1/3 Test #2: feature-b ........................ Passed 0.11 sec
    Start 3: feature-c
2/3 Test #3: feature-c ........................ Passed 0.11 sec
    Start 4: feature-d
3/3 Test #4: feature-d ........................ Passed 0.51 sec

100% tests passed, 0 tests failed out of 3
```

# 不仅如此

尝试使用`**$ ctest --help**`，您将看到大量可供选择的选项来定制您的测试。

# 使用测试夹具

本例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-10`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-10)找到。本例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

本例灵感来源于 Craig Scott 的工作，我们建议读者也参考相应的博客文章以获取更多背景信息，网址为[`crascit.com/2016/10/18/test-fixtures-with-cmake-ctest/`](https://crascit.com/2016/10/18/test-fixtures-with-cmake-ctest/)。本例的动机是展示如何使用测试夹具。对于需要测试前设置动作和测试后清理动作的更复杂的测试来说，这些夹具非常有用（例如创建示例数据库、设置连接、断开连接、清理测试数据库等）。我们希望确保运行需要设置或清理动作的测试时，这些步骤能以可预测和稳健的方式自动触发，而不会引入代码重复。这些设置和清理步骤可以委托给测试框架，如 Google Test 或 Catch2，但在这里，我们展示了如何在 CMake 级别实现测试夹具。

# 准备就绪

我们将准备四个小型 Python 脚本，并将它们放置在`test`目录下：`setup.py`、`feature-a.py`、`feature-b.py`和`cleanup.py`。

# 如何操作

我们从熟悉的`CMakeLists.txt`结构开始，并添加了一些额外的步骤，如下所示：

1.  我们准备好了熟悉的基础设施：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-10 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()
```

1.  然后，我们定义了四个测试步骤并将它们与一个固定装置绑定：

```cpp
add_test(
  NAME setup
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/setup.py
  )
set_tests_properties(
  setup
  PROPERTIES
    FIXTURES_SETUP my-fixture
  )

add_test(
  NAME feature-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-a.py
  )
add_test(
  NAME feature-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-b.py
  )
set_tests_properties(
  feature-a
  feature-b
  PROPERTIES
    FIXTURES_REQUIRED my-fixture
  )

add_test(
  NAME cleanup
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/cleanup.py
  )
set_tests_properties(
  cleanup
  PROPERTIES
    FIXTURES_CLEANUP my-fixture
  )
```

1.  运行整个集合并不会带来任何惊喜，正如以下输出所示：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

    Start 1: setup
1/4 Test #1: setup ............................ Passed 0.01 sec
    Start 2: feature-a
2/4 Test #2: feature-a ........................ Passed 0.01 sec
    Start 3: feature-b
3/4 Test #3: feature-b ........................ Passed 0.00 sec
    Start 4: cleanup
4/4 Test #4: cleanup .......................... Passed 0.01 sec

100% tests passed, 0 tests failed out of 4
```

1.  然而，有趣的部分在于当我们尝试单独运行测试`feature-a`时。它正确地调用了`setup`步骤和`cleanup`步骤：

```cpp
$ ctest -R feature-a

 Start 1: setup
1/3 Test #1: setup ............................ Passed 0.01 sec
 Start 2: feature-a
```

```cpp
2/3 Test #2: feature-a ........................ Passed 0.00 sec
 Start 4: cleanup
3/3 Test #4: cleanup .......................... Passed 0.01 sec

100% tests passed, 0 tests failed out of 3
```

# 工作原理

在本例中，我们定义了一个文本固定装置并将其命名为`my-fixture`。我们为设置测试赋予了`FIXTURES_SETUP`属性，为清理测试赋予了`FIXTURES_CLEANUP`属性，并且使用`FIXTURES_REQUIRED`确保测试`feature-a`和`feature-b`都需要设置和清理步骤才能运行。将这些绑定在一起，确保我们始终以明确定义的状态进入和退出步骤。

# 还有更多内容

如需了解更多背景信息以及使用此技术进行固定装置的出色动机，请参阅[`crascit.com/2016/10/18/test-fixtures-with-cmake-ctest/`](https://crascit.com/2016/10/18/test-fixtures-with-cmake-ctest/)。
