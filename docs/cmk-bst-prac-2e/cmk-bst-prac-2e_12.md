# 第十章：在超级构建中处理分布式仓库和依赖项

正如我们现在应该已经了解的，每个大项目都有自己的依赖项。处理这些依赖项最简单的方法是使用包管理器，如**Conan**或**vcpkg**。但是，使用包管理器并不总是可行的，可能是由于公司政策、项目需求或资源不足。因此，项目作者可能会考虑使用传统的老式方式来处理依赖项。处理这些依赖项的常见方式可能包括将所有依赖项嵌入到仓库的构建代码中。或者，项目作者可能决定让最终用户从头开始处理依赖项。这两种方式都不太清晰，各有缺点。如果我告诉你有一个折衷方案呢？欢迎使用*超级构建*方法。

超级构建是一种可以用于将满足依赖关系所需的逻辑与项目代码解耦的方法，就像包管理器的工作原理一样。事实上，我们可以把这种方法称为*穷人的包管理器*。将依赖逻辑与项目代码分离，使我们能够拥有更灵活、更易于维护的项目结构。在本章中，我们将详细学习如何实现这一点。

为了理解本章分享的技巧，我们将涵盖以下主要内容：

+   超级构建的要求和前提条件

+   跨多个代码仓库构建

+   确保超级构建中的版本一致性

让我们从技术要求开始。

# 技术要求

在深入本章之前，你应该已经掌握了*第五章*的内容，*整合第三方库* *和依赖管理*。本章将采用以实例教学的方式，因此建议从[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/tree/main/chapter10`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/tree/main/chapter10)获取本章的示例内容。所有示例都假设你将使用项目中提供的容器，项目地址为[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition)。

让我们从检查超级构建的前提和要求开始学习。

# 超级构建的要求和前提条件

超级构建可以结构化为一个大型构建，构建多个项目，或者作为一个项目内部的子模块来处理依赖。因此，获取仓库的手段是必须的。幸运的是，CMake 提供了稳定且成熟的方式来实现这一点。举几个例子，`ExternalProject` 和 `FetchContent` 是处理外部依赖的最流行的 CMake 模块。在我们的示例中，我们将使用 `FetchContent` CMake 模块，因为它更简洁且易于处理。请注意，使用 CMake 提供的手段并不是强制要求，而是一种便捷方式。超级构建也可以通过使用版本控制系统工具来结构化，比如 `git submodule` 或 `git subtree`。由于本书的重点是 CMake，而 Git 对 `FetchContent` 的支持也相当不错，我们倾向于使用它。

现在就到这里。让我们继续学习如何构建跨多个代码仓库的项目。

# 跨多个代码仓库进行构建

软件项目，无论是直接的还是间接的，都涉及多个代码仓库。处理本地项目代码是最简单的，但软件项目很少是独立的。如果没有合适的依赖管理策略，事情可能会很快变得复杂。本章的第一个建议是*如果可能的话，使用包管理器或依赖提供者*。正如在 *第五章* 中所描述的，*集成第三方库和依赖管理*，包管理器大大减少了在依赖管理上花费的精力。如果你不能使用预构建的包管理器，你可能需要为你的项目创建一个专门的迷你包管理器，这就是所谓的 **超级构建**。

超级构建主要用于使项目在依赖方面自给自足，也就是说，项目能够在不需要用户干预的情况下满足自身的依赖。拥有这样的能力对所有使用者来说都非常方便。为了演示这种技术，我们将从一个这种场景的示例开始。让我们开始吧。

## 创建超级构建的推荐方式 – FetchContent

我们将按照 `Chapter 10``, Example 01` 来进行这一部分的学习。让我们像往常一样，首先检查 `Chapter 10``, Example 01` 的 `CMakeLists.txt` 文件。为了简便起见，前七行被省略了：

```cpp
if(CH10_EX01_USE_SUPERBUILD)
  include(superbuild.cmake)
else()
  find_package(GTest 1.10.0 REQUIRED)
  find_package(benchmark 1.6.1 REQUIRED)
endif()
add_executable(ch10_ex01_tests)
target_sources(ch10_ex01_tests PRIVATE src/tests.cpp)
target_link_libraries(ch10_ex01_tests PRIVATE GTest::Main)
add_executable(ch10_ex01_benchmarks)
target_sources(ch10_ex01_benchmarks PRIVATE src
  /benchmarks.cpp)
target_link_libraries(ch10_ex01_benchmarks PRIVATE
  benchmark::benchmark)
```

如我们所见，这是一个简单的 `CMakeLists.txt` 文件，定义了两个目标，分别命名为 `ch10_ex01_tests` 和 `ch10_ex01_benchmarks`。这两个目标分别依赖于 Google Test 和 Google Benchmark 库。这些库通过超级构建或 `find_package(…)` 调用来查找和定义，具体取决于 `CH10_EX01_USE_SUPERBUILD` 变量。`find_package(…)` 路径是我们到目前为止所采用的方式。让我们一起检查超级构建文件 `superbuild.cmake`：

```cpp
include(FetchContent)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.6.1
)
FetchContent_Declare(GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(GTest benchmark)
add_library(GTest::Main ALIAS gtest_main)
```

在第一行中，我们包含了`FetchContent` CMake 模块，因为我们将利用它来处理依赖关系。在接下来的六行中，使用`FetchContent_Declare`函数声明了两个外部目标，`benchmark`和`GTest`，并指示它们通过 Git 获取。因此，调用了`FetchContent_MakeAvailable(…)`函数以使声明的目标可用。最后，调用`add_library(…)`来定义一个名为`GTest::Main`的别名目标，指向`gtest_main`目标。这样做是为了保持`find_package(…)`和超级构建目标名称之间的兼容性。对于`benchmark`，没有定义别名目标，因为它的`find_package(…)`和超级构建目标名称已经兼容。

让我们通过调用以下命令来配置和构建示例：

```cpp
cd chapter_10/ex01_external_deps
cmake -S ./ -B build -DCH10_EX01_USE_SUPERBUILD:BOOL=ON
cmake --build build/ --parallel $(nproc)
```

在前两行中，我们进入`example10/ex01`文件夹并配置项目。请注意，我们将`CH10_EX01_USE_SUPERBUILD`变量设置为`ON`，以启用超级构建代码。在最后一行，我们通过*N*个并行作业构建项目，其中*N*是`nproc`命令的结果。

由于替代的`find_package(...)`路径，构建在没有启用超级构建的情况下也能正常工作，前提是环境中有`google test >= 1.10.0`和`google benchmark >= 1.6.1`。这将允许包维护者在不修补项目的情况下更改依赖项版本。像这样的细小定制点对于可移植性和可重复性非常重要。

接下来，我们将查看一个使用`ExternalProject`模块而不是`FetchContent`模块的超级构建示例。

## 传统方法 – ExternalProject_Add

在`FetchContent`出现之前，大多数人通过使用`ExternalProject_Add` CMake 函数实现了超级构建方法。该函数由`ExternalProject` CMake 模块提供。在本节中，我们将通过`ExternalProject_Add`查看一个超级构建示例，以了解它与使用`FetchContent`模块的区别。

让我们一起看看`Chapter 10`, Example 02`中的`CMakeLists.txt`文件（注释和项目指令已省略）：

```cpp
# ...
include(superbuild.cmake)
add_executable(ch10_ex02_tests)
target_sources(ch10_ex02_tests PRIVATE src/tests.cpp)
target_link_libraries(ch10_ex02_tests PRIVATE catch2)
```

同样，这个项目是一个单元测试项目，包含一个 C++源文件，不过这次使用的是`catch2`而不是 Google Test。`CMakeLists.txt`文件直接包含了`superbuild.cmake`文件，定义了一个可执行目标，并将`Catch2`库链接到该目标。你可能已经注意到，这个示例没有使用`FindPackage(...)`来发现`Catch2`库。原因在于，与`FetchContent`不同，`ExternalProject`是在构建时获取并构建外部依赖项。由于`Catch2`库的内容在配置时不可用，我们无法在此使用`FindPackage(...)`。`FindPackage(…)`在配置时运行，并需要包文件存在。让我们也看看`superbuild.cmake`：

```cpp
include(ExternalProject)
ExternalProject_Add(catch2_download
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v2.13.9
    INSTALL_COMMAND ""
    # For disabling the warning that treated as an error
    CMAKE_ARGS -DCMAKE_CXX_FLAGS="-Wno-error=pragmas"
)
SET(CATCH2_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}
  /catch2_download-
  prefix/src/catch2_download/single_include)
file(MAKE_DIRECTORY ${CATCH2_INCLUDE_DIR})
add_library(catch2 IMPORTED INTERFACE GLOBAL)
add_dependencies(catch2 catch2_download)
set_target_properties(catch2 PROPERTIES "INTERFACE_INCLUDE_
  DIRECTORIES" "${CATCH2_INCLUDE_DIR}")
```

`superbuild.cmake` 模块包含了 `ExternalProject` CMake 模块。代码调用了 `ExternalProject_Add` 函数来声明一个名为 `catch2_download` 的目标，并指定 `GIT_REPOSITORY`、`GIT_TAG`、`INSTALL_COMMAND` 和 `CMAKE_ARGS` 参数。如你从前面的章节中回忆的那样，`ExternalProject_Add` 函数可以从不同的源获取依赖项。我们的示例是尝试通过 Git 获取依赖项。`GIT_REPOSITORY` 和 `GIT_TAG` 参数分别用于指定目标 Git 仓库的 URL 和 `git clone` 后需要签出的标签。由于 `Catch2` 是一个 CMake 项目，因此我们需要提供给 `ExternalProject_Add` 函数的参数最少。`ExternalProject_Add` 函数默认知道如何配置、构建和安装一个 CMake 项目，因此无需 `CONFIGURE_COMMAND` 或 `BUILD_COMMAND` 参数。空的 `INSTALL_COMMAND` 参数用于禁用并安装构建后的依赖项。最后一个参数 `CMAKE_ARGS` 用于向外部项目的配置步骤传递 CMake 参数。我们用它来抑制一个关于 `Catch2` 编译中遗留 pragma 的 GCC 警告（将其视为错误）。

`ExternalProject_Add` 命令将所需的库拉取到一个前缀路径并进行构建。因此，要使用已获取的内容，我们首先需要将其导入到项目中。由于我们不能使用 `FindPackage(...)` 让 CMake 来处理库的导入工作，因此我们需要做一些手动操作。其中一项工作是定义 `Catch2` 目标的 `include` 目录。由于 `Catch2` 是一个仅包含头文件的库，定义一个包含头文件的接口目标就足够了。我们声明了 `CATCH2_INCLUDE_DIR` 变量来设置包含 `Catch2` 头文件的目录。我们使用该变量来设置在此示例中创建的导入目标的 `INTERFACE_INCLUDE_DIRECTORIES` 属性。接下来，调用文件（`MAKE_DIRECTORY ${CATCH2_INCLUDE_DIR}`）的 CMake 命令来创建包含目录。之所以这么做，是因为 `ExternalProject_Add` 的工作方式，`Catch2` 的内容直到构建步骤执行时才会出现。设置目标的 `INTERFACE_INCLUDE_DIRECTORIES` 需要确保给定的目录已经存在，所以我们通过这种小技巧来解决这个问题。在最后三行中，我们为 `Catch2` 声明了一个 `IMPORTED INTERFACE` 库，使该库依赖于 `catch2_download` 目标，并设置了导入库的 `INTERFACE_INCLUDE_DIRECTORIES`。

让我们尝试配置并构建我们的示例，检查它是否能够正常工作：

```cpp
cd chapter_10/ex02_external_deps_with_extproject
cmake -S ./ -B build
cmake --build build/ --parallel $(nproc)
```

如果一切顺利，你应该会看到类似于以下的输出：

```cpp
[ 10%] Creating directories for 'catch2_download'
[ 20%] Performing download step (git clone) for
  'catch2_download'
Cloning into 'catch2_download'...
HEAD is now at 62fd6605 v2.13.9
[ 30%] Performing update step for 'catch2_download'
[ 40%] No patch step for 'catch2_download'
[ 50%] Performing configure step for 'catch2_download'
/* ... */
[ 60%] Performing build step for 'catch2_download'
/* ... */
[ 70%] No install step for 'catch2_download'
[ 80%] Completed 'catch2_download'
[ 80%] Built target catch2_download
/* ... */
[100%] Built target ch10_ex02_tests
```

好的，看来我们已经成功构建了测试可执行文件。让我们运行它，检查它是否正常工作，通过运行 `./``build/ch10_ex02_tests` 可执行文件：

```cpp
===========================================================
All tests passed (4 assertions in 1 test case)
```

接下来，我们将看到一个简单的 Qt 应用程序，使用来自超构建的 Qt 框架。

## 奖励 – 使用 Qt 6 框架与超构建

到目前为止，我们已经处理了那些占用空间较小的库。现在让我们尝试一些更复杂的内容，比如在超级构建中使用像 Qt 这样的框架。在这一部分，我们将跟随`第十章`，示例`03`进行操作。

重要提示

如果你打算在提供的 Docker 容器外尝试这个示例，可能需要安装一些 Qt 运行时所需的附加依赖项。Debian 类系统所需的包如下：`libgl1-mesa-dev libglu1-mesa-dev '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev` `libxkbcommon-dev libxkbcommon-x11-dev`。

该示例包含一个源文件`main.cpp`，它输出一个简单的 Qt 窗口应用程序，并带有一条消息。实现如下：

```cpp
#include <qapplication.h>
#include <qpushbutton.h>
int main( int argc, char **argv )
{
    QApplication a( argc, argv );
    QPushButton hello( "Hello from CMake Best Practices!",
      0 );
    hello.resize( 250, 30 );
    hello.show();
    return a.exec();
}
```

我们的目标是能够编译这个 Qt 应用程序，而不需要用户自己安装 Qt 框架。超级构建应该自动安装 Qt 6 框架，并且应用程序应该能够使用它。接下来，让我们像往常一样看看这个示例的 `CMakeLists.txt` 文件：

```cpp
if(CH10_EX03_USE_SUPERBUILD)
  include(superbuild.cmake)
else()
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
endif()
add_executable(ch10_ex03_simple_qt_app main.cpp)
target_compile_features(ch10_ex03_simple_qt_app PRIVATE
  cxx_std_11)
target_link_libraries(ch10_ex03_simple_qt_app Qt6::Core
  Qt6::Widgets)
```

和第一个示例一样，`CMakeLists.txt` 文件根据一个 `option` 标志包含了 `superbuild.cmake` 文件。如果用户选择使用超级构建，例如，它将包含超级构建模块。否则，依赖项将通过 `find_package(...)` 尝试在系统中找到。在最后的三行中，定义了一个可执行目标，为该目标设置了 C++ 标准，并将该目标与 `QT6::Core` 和 `QT6::Widgets` 目标链接。这些目标要么由超级构建定义，要么通过 `find_package(...)` 调用找到，具体取决于用户是否选择使用超级构建。接下来，让我们继续看看 `superbuild.cmake` 文件：

```cpp
include(FetchContent)
message(STATUS "Chapter 10, example 03 superbuild enabled.
  Will try to satisfy dependencies for the example.")
set(FETCHCONTENT_QUIET FALSE) # Enable message output for
  FetchContent commands
set(QT_BUILD_SUBMODULES "qtbase" CACHE STRING "Submodules
  to build")
set(QT_WILL_BUILD_TOOLS on)
set(QT_FEATURE_sql off)
set(QT_FEATURE_network off)
set(QT_FEATURE_dbus off)
set(QT_FEATURE_opengl off)
set(QT_FEATURE_testlib off)
set(QT_BUILD_STANDALONE_TESTS off)
set(QT_BUILD_EXAMPLES off)
set(QT_BUILD_TESTS off)
FetchContent_Declare(qt6
    GIT_REPOSITORY https://github.com/qt/qt5.git
    GIT_TAG        v6.3.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE # Since the clone process is lengthy,
      show progress of download
    GIT_SUBMODULES qtbase # The only QT submodule we need
)
FetchContent_MakeAvailable(qt6)
```

`superbuild.cmake` 文件使用 `FetchContent` 模块来获取 Qt 依赖项。由于获取和准备 Qt 可能需要较长时间，因此禁用了某些未使用的 Qt 框架功能。启用了 `FetchContent` 消息输出，以便更好地跟踪进度。让我们通过运行以下命令来配置并编译示例：

```cpp
cd chapter_10/ex03_simple_qt_app/
cmake -S ./ -B build -DCH10_EX03_USE_SUPERBUILD:BOOL=ON
cmake --build build/ --parallel $(nproc)
```

如果一切如预期那样进行，你应该看到类似于这里展示的输出：

```cpp
/*...*/
[ 11%] Creating directories for 'qt6-populate'
[ 22%] Performing download step (git clone) for
  'qt6-populate'
Cloning into 'qt6-src'...
/*...*/
[100%] Completed 'qt6-populate'
[100%] Built target qt6-populate
/*...*/
-- Configuring done
-- Generating done
/*...*/
[  0%] Generating ../../mkspecs/modules
  /qt_lib_widgets_private.pri
[  0%] Generating ../../mkspecs/modules
  /qt_lib_gui_private.pri
[  0%] Generating ../../mkspecs/modules/qt_lib_core_private.pri
/* ... */
[ 98%] Linking CXX executable ch10_ex03_simple_qt_app
[ 98%] Built target ch10_ex03_simple_qt_app
/*...*/
```

如果一切顺利，你已经成功编译了示例。让我们通过运行以下命令检查它是否正常工作：

```cpp
./build/ch10_ex03_simple_qt_app
```

如果一切正常，一个小的图形界面窗口应该会弹出。该窗口应类似于下图所示：

![图 10.1 – 简单的 Qt 应用程序窗口](img/B30947_10_01.jpg)

图 10.1 – 简单的 Qt 应用程序窗口

解决了这个问题后，我们已经完成了如何在 CMake 项目中使用超级构建的讲解。接下来，我们将探讨如何确保超级构建中的版本一致性。

# 确保超级构建中的版本一致性

版本一致性是所有软件项目中的一个重要方面。正如你现在应该已经了解到的，软件世界中的一切都不是一成不变的。软件随着时间的发展而演变和变化。这些变化往往需要提前得到确认，要么通过对新版本运行一系列测试，要么通过对消费代码本身进行修改。理想情况下，上游代码的变化不应该影响现有构建的复现，除非我们希望它们产生影响。如果软件验证和测试已经针对某一组合完成，那么一个项目的`x.y`版本应该始终与`z.q`依赖版本一起构建。原因在于，即便没有 API 或 ABI 的变化，上游依赖中的最小变动也可能影响你的软件行为。如果没有提供版本一致性，你的软件将没有明确定义的行为。因此，提供版本一致性的方法非常关键。

在超构建中确保版本一致性取决于超构建的组织方式。对于通过版本控制系统获取的代码库来说，相对容易。与其克隆项目并按原样使用，不如切换到特定的分支或标签。如果没有这些锚点，可以切换到特定的提交。这样可以确保你的超构建具备前瞻性。但即使这样也可能不足够。标签可能被覆盖，分支可能被强制推送，历史可能被重写。为了降低这种风险，你可以选择分叉项目并使用该分叉作为上游。这样，你就可以完全控制上游内容。但请记住，这种方法带来了维护的负担。

这个故事的寓意是，不要盲目跟踪上游。始终关注最新的变化。对于作为归档文件使用的第三方依赖，始终检查它们的哈希摘要。通过这种方式，你可以确保你确实使用了项目的目标版本，如果有任何变化，你必须手动确认它。

# 总结

本章简要介绍了超构建的概念，以及如何利用超构建进行依赖管理。超构建是一种非侵入性且强大的依赖管理方式，适用于缺少包管理器的情况。

在下一章中，我们将详细探讨如何为苹果生态系统构建软件。由于苹果的封闭性以及 macOS 和 iOS 的紧密集成，在针对这些平台时需要考虑一些事项。

# 问题

完成本章后，你应该能够回答以下问题：

1.  什么是超构建（super-build）？

1.  我们在哪些主要场景下可以使用超构建（super-build）？

1.  为了在超构建中实现版本一致性，可以采取哪些措施？

# 答案

1.  超构建是一种构建软件项目的方法，跨越多个代码库。

1.  在没有包管理器的情况下，我们希望让项目能够满足自身的依赖关系。

1.  使用锚点，如分支、标签或提交哈希值。
