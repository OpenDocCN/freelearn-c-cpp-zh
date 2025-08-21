# 第九章：超级构建模式

在本章中，我们将涵盖以下内容：

+   使用超级构建模式

+   使用超级构建管理依赖：I. Boost 库

+   使用超级构建管理依赖：II. FFTW 库

+   使用超级构建管理依赖：III. Google Test 框架

+   将项目作为超级构建进行管理

# 引言

每个项目都必须处理依赖关系，而 CMake 使得在配置项目的系统上查找这些依赖关系变得相对容易。第三章，*检测外部库和程序*，展示了如何在系统上找到已安装的依赖项，并且到目前为止我们一直使用相同的模式。然而，如果依赖关系未得到满足，我们最多只能导致配置失败并告知用户失败的原因。但是，使用 CMake，我们可以组织项目，以便在系统上找不到依赖项时自动获取和构建它们。本章将介绍和分析`ExternalProject.cmake`和`FetchContent.cmake`标准模块以及它们在*超级构建模式*中的使用。前者允许我们在*构建时间*获取项目的依赖项，并且长期以来一直是 CMake 的一部分。后者模块是在 CMake 3.11 版本中添加的，允许我们在*配置时间*获取依赖项。通过超级构建模式，我们可以有效地利用 CMake 作为高级包管理器：在您的项目中，您将以相同的方式处理依赖项，无论它们是否已经在系统上可用，或者它们是否需要从头开始构建。接下来的五个示例将引导您了解该模式，并展示如何使用它来获取和构建几乎任何依赖项。

两个模块都在网上有详尽的文档。对于`ExternalProject.cmake`，我们建议读者参考[`cmake.org/cmake/help/v3.5/module/ExternalProject.html`](https://cmake.org/cmake/help/v3.5/module/ExternalProject.html)。对于`FetchContent.cmake`，我们建议读者参考[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)。

# 使用超级构建模式

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-01)找到，并包含一个 C++示例。该示例适用于 CMake 3.5（及以上）版本，并在 GNU/Linux、macOS 和 Windows 上进行了测试。

本示例将通过一个非常简单的示例介绍超级构建模式。我们将展示如何使用`ExternalProject_Add`命令来构建一个简单的“Hello, World”程序。

# 准备工作

本示例将构建以下源代码（`hello-world.cpp`）中的“Hello, World”可执行文件：

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() { return std::string("Hello, CMake superbuild world!"); }

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

项目结构如下，包含一个根目录`CMakeLists.txt`和一个`src/CMakeLists.txt`文件：

```cpp
.
├── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── hello-world.cpp
```

# 如何操作

首先让我们看一下根文件夹中的`CMakeLists.txt`：

1.  我们声明一个 C++11 项目，并指定最低要求的 CMake 版本：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  我们为当前和任何底层目录设置`EP_BASE`目录属性。这将在稍后讨论：

```cpp
set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/subprojects)
```

1.  我们包含`ExternalProject.cmake`标准模块。该模块提供了`ExternalProject_Add`函数：

```cpp
include(ExternalProject)
```

1.  通过调用`ExternalProject_Add`函数，将我们的“Hello, World”示例的源代码作为外部项目添加。外部项目的名称为`recipe-01_core`：

```cpp
ExternalProject_Add(${PROJECT_NAME}_core
```

1.  我们使用`SOURCE_DIR`选项设置外部项目的源目录：

```cpp
SOURCE_DIR
${CMAKE_CURRENT_LIST_DIR}/src
```

1.  `src`子目录包含一个完整的 CMake 项目。为了配置和构建它，我们通过`CMAKE_ARGS`选项将适当的 CMake 选项传递给外部项目。在我们的情况下，我们只需要传递 C++编译器和对 C++标准的要求：

```cpp
CMAKE_ARGS
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
  -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
```

1.  我们还设置了 C++编译器标志。这些标志通过`CMAKE_CACHE_ARGS`选项传递给`ExternalProject_Add`命令：

```cpp
CMAKE_CACHE_ARGS
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
```

1.  我们配置外部项目，使其始终处于构建状态：

```cpp
BUILD_ALWAYS
  1
```

1.  安装步骤不会执行任何操作（我们将在第 4 个配方中重新讨论安装，即“编写安装程序”中的“安装超级构建”）：

```cpp
INSTALL_COMMAND
  ""
)
```

现在让我们转向`src/CMakeLists.txt`。由于我们将“Hello, World”源代码作为外部项目添加，这是一个完整的`CMakeLists.txt`文件，用于独立项目：

1.  同样，这里我们声明了最低要求的 CMake 版本：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

1.  我们声明一个 C++项目：

```cpp
project(recipe-01_core LANGUAGES CXX)
```

1.  最后，我们从`hello-world.cpp`源文件添加一个可执行目标，即`hello-world`：

```cpp
add_executable(hello-world hello-world.cpp)
```

配置和构建我们的项目按照常规方式进行：

```cpp
$ mkdir -p build
$ cmake ..
$ cmake --build .
```

构建目录的结构现在稍微复杂一些。特别是，我们注意到`subprojects`文件夹及其内容：

```cpp
build/subprojects/
├── Build
│   └── recipe-01_core
│       ├── CMakeCache.txt
│       ├── CMakeFiles
│       ├── cmake_install.cmake
│       ├── hello-world
│       └── Makefile
├── Download
│   └── recipe-01_core
├── Install
│   └── recipe-01_core
├── Stamp
│   └── recipe-01_core
│       ├── recipe-01_core-configure
│       ├── recipe-01_core-done
│       ├── recipe-01_core-download
│       ├── recipe-01_core-install
│       ├── recipe-01_core-mkdir
│       ├── recipe-01_core-patch
│       └── recipe-01_core-update
└── tmp
    └── recipe-01_core
        ├── recipe-01_core-cache-.cmake
        ├── recipe-01_core-cfgcmd.txt
        └── recipe-01_core-cfgcmd.txt.in
```

`recipe-01_core`已构建到`build/subprojects`的子目录中，称为`Build/recipe-01_core`，这是我们设置的`EP_BASE`。

`hello-world`可执行文件已在`Build/recipe-01_core`下创建。额外的子文件夹`tmp/recipe-01_core`和`Stamp/recipe-01_core`包含临时文件，例如 CMake 缓存脚本`recipe-01_core-cache-.cmake`，以及 CMake 为构建外部项目执行的各种步骤的标记文件。

# 它是如何工作的

`ExternalProject_Add`命令可用于添加第三方源代码。然而，我们的第一个示例展示了如何将我们自己的项目作为不同 CMake 项目的集合来管理。在这个示例中，根目录和叶目录的`CMakeLists.txt`都声明了一个 CMake 项目，即它们都使用了`project`命令。

`ExternalProject_Add`有许多选项，可用于微调外部项目的配置和编译的所有方面。这些选项可以分为以下几类：

+   **目录**选项：这些用于调整外部项目的源代码和构建目录的结构。在我们的例子中，我们使用了 `SOURCE_DIR` 选项让 CMake 知道源代码可在 `${CMAKE_CURRENT_LIST_DIR}/src` 文件夹中找到，因此不应从其他地方获取。构建项目和存储临时文件的目录也可以在此类选项中指定，或者作为目录属性指定。我们通过设置 `EP_BASE` 目录属性遵循了后者的方式。CMake 将为各种子项目设置所有目录，布局如下：

```cpp
TMP_DIR      = <EP_BASE>/tmp/<name>
STAMP_DIR    = <EP_BASE>/Stamp/<name>
DOWNLOAD_DIR = <EP_BASE>/Download/<name>
SOURCE_DIR   = <EP_BASE>/Source/<name>
BINARY_DIR   = <EP_BASE>/Build/<name>
INSTALL_DIR  = <EP_BASE>/Install/<name>
```

+   **下载**选项：外部项目的代码可能需要从在线存储库或资源下载。此类选项允许您控制此步骤的所有方面。

+   **更新**和**补丁**选项：这类选项可用于定义如何更新外部项目的源代码或如何应用补丁。

+   **配置**选项：默认情况下，CMake 假设外部项目本身使用 CMake 进行配置。然而，正如后续章节将展示的，我们并不局限于这种情况。如果外部项目是 CMake 项目，`ExternalProject_Add` 将调用 CMake 可执行文件并传递选项给它。对于我们当前的示例，我们通过 `CMAKE_ARGS` 和 `CMAKE_CACHE_ARGS` 选项传递配置参数。前者直接作为命令行参数传递，而后者通过 CMake 脚本文件传递。在我们的示例中，脚本文件位于 `build/subprojects/tmp/recipe-01_core/recipe-01_core-cache-.cmake`。配置将如下所示：

```cpp
$ cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_STANDARD=11 
-DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_STANDARD_REQUIRED=ON 
-C/home/roberto/Workspace/robertodr/cmake-cookbook/chapter-08/recipe-01/cxx-example/build/subprojects/tmp/recipe-01_core/recipe-01_core-cache-.cmake "-GUnix Makefiles" /home/roberto/Workspace/robertodr/cmake-cookbook/chapter-08/recipe-01/cxx-example/src
```

+   **构建**选项：这类选项可用于调整外部项目的实际编译。我们的示例使用了 `BUILD_ALWAYS` 选项以确保外部项目总是被新鲜构建。

+   **安装**选项：这些是配置外部项目应如何安装的选项。我们的示例将 `INSTALL_COMMAND` 留空，我们将在 第十章，*编写安装程序*中更详细地讨论使用 CMake 进行安装。

+   **测试**选项：对于从源代码构建的任何软件，运行测试总是一个好主意。`ExternalProject_Add` 的这类选项就是为了这个目的。我们的示例没有使用这些选项，因为“Hello, World”示例没有任何测试，但在第五章，*将您的项目作为超级构建管理*中，我们将触发测试步骤。

`ExternalProject.cmake` 定义了命令 `ExternalProject_Get_Property`，顾名思义，这对于检索外部项目的属性非常有用。外部项目的属性在首次调用 `ExternalProject_Add` 命令时设置。例如，检索配置 `recipe-01_core` 时传递给 CMake 的参数可以通过以下方式实现：

```cpp
ExternalProject_Get_Property(${PROJECT_NAME}_core CMAKE_ARGS)
message(STATUS "CMAKE_ARGS of ${PROJECT_NAME}_core ${CMAKE_ARGS}")
```

`ExternalProject_Add`的完整选项列表可以在 CMake 文档中找到：[`cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add`](https://cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add)

# 还有更多

我们将在以下配方中详细探讨`ExternalProject_Add`命令的灵活性。然而，有时我们想要使用的外部项目可能需要执行额外的、非标准的步骤。为此，`ExternalProject.cmake`模块定义了以下附加命令：

1.  `ExternalProject_Add_Step`。一旦添加了外部项目，此命令允许将附加命令作为自定义步骤附加到该项目上。另请参见：[`cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_step`](https://cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_step)

1.  `ExternalProject_Add_StepTargets`。它允许您在任何外部项目中定义步骤，例如构建和测试步骤，作为单独的目标。这意味着可以从完整的外部项目中单独触发这些步骤，并允许对项目内的复杂依赖关系进行精细控制。另请参见：[`cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_steptargets`](https://cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_steptargets)

1.  `ExternalProject_Add_StepDependencies`。有时外部项目的步骤可能依赖于项目之外的目标，此命令旨在处理这些情况。另请参见：[`cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_stepdependencies`](https://cmake.org/cmake/help/v3.5/module/ExternalProject.html#command:externalproject_add_stepdependencies)

# 使用超级构建管理依赖项：I. Boost 库

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-02) 获取，并包含一个 C++示例。该配方适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS、Windows（使用 MSYS Makefiles 和 Ninja）上进行了测试。

Boost 库提供了丰富的 C++编程基础设施，并且受到 C++开发者的欢迎。我们已经在第三章，*检测外部库和程序*中展示了如何在系统上找到 Boost 库。然而，有时您的项目所需的 Boost 版本可能不在系统上。本食谱将展示如何利用超级构建模式来确保缺少的依赖不会阻止配置。我们将重用来自第三章，*检测外部库和程序*中第 8 个食谱，*检测 Boost 库*的代码示例，但将其重新组织为超级构建的形式。这将是项目的布局：

```cpp
.
├── CMakeLists.txt
├── external
│   └── upstream
│       ├── boost
│       │   └── CMakeLists.txt
│       └── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── path-info.cpp
```

您会注意到项目源代码树中有四个`CMakeLists.txt`文件。以下部分将引导您了解这些文件。

# 如何操作

我们将从根`CMakeLists.txt`开始：

1.  我们像往常一样声明一个 C++11 项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  我们设置`EP_BASE`目录属性：

```cpp
set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/subprojects)
```

1.  我们设置`STAGED_INSTALL_PREFIX`变量。该目录将用于在我们的构建树中安装依赖项：

```cpp
set(STAGED_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage)
message(STATUS "${PROJECT_NAME} staged install: ${STAGED_INSTALL_PREFIX}")
```

1.  我们的项目需要 Boost 库的文件系统和系统组件。我们声明一个列表变量来保存此信息，并设置所需的最小 Boost 版本：

```cpp
list(APPEND BOOST_COMPONENTS_REQUIRED filesystem system)
set(Boost_MINIMUM_REQUIRED 1.61)
```

1.  我们添加`external/upstream`子目录，它将依次添加`external/upstream/boost`子目录：

```cpp
add_subdirectory(external/upstream)
```

1.  然后，我们包含`ExternalProject.cmake`标准 CMake 模块。这定义了，除其他外，`ExternalProject_Add`命令，这是协调超级构建的关键：

```cpp
include(ExternalProject)
```

1.  我们的项目位于`src`子目录下，并将其作为外部项目添加。我们使用`CMAKE_ARGS`和`CMAKE_CACHE_ARGS`传递 CMake 选项：

```cpp
ExternalProject_Add(${PROJECT_NAME}_core
  DEPENDS
    boost_external
  SOURCE_DIR
    ${CMAKE_CURRENT_LIST_DIR}/src
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_INCLUDE_PATH:PATH=${BOOST_INCLUDEDIR}
    -DCMAKE_LIBRARY_PATH:PATH=${BOOST_LIBRARYDIR}
```

```cpp
  BUILD_ALWAYS
    1
  INSTALL_COMMAND
    ""
  )
```

现在让我们看看`external/upstream`中的`CMakeLists.txt`文件。该文件只是将`boost`文件夹添加为附加目录：

```cpp
add_subdirectory(boost)
```

`external/upstream/boost`中的`CMakeLists.txt`描述了满足对 Boost 依赖所需的操作。我们的目标很简单，如果所需版本未安装，下载源代码存档并构建它：

1.  首先，我们尝试找到所需的最小版本的 Boost 组件：

```cpp
find_package(Boost ${Boost_MINIMUM_REQUIRED} QUIET COMPONENTS "${BOOST_COMPONENTS_REQUIRED}")
```

1.  如果找到这些选项，我们会添加一个接口库，`boost_external`。这是一个虚拟目标，用于在我们的超级构建中正确处理构建顺序：

```cpp
if(Boost_FOUND)
  message(STATUS "Found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
  add_library(boost_external INTERFACE)
else()    
  # ... discussed below
endif()
```

1.  如果`find_package`不成功或者我们强制进行超级构建，我们需要设置一个本地的 Boost 构建，为此，我们进入前一个条件语句的 else 部分：

```cpp
else()
  message(STATUS "Boost ${Boost_MINIMUM_REQUIRED} could not be located, Building Boost 1.61.0 instead.")
```

1.  由于这些库不使用 CMake，我们需要为它们的原生构建工具链准备参数。首先，我们设置要使用的编译器：

```cpp
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(APPLE)
      set(_toolset "darwin")
    else()
      set(_toolset "gcc")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(_toolset "clang")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if(APPLE)
      set(_toolset "intel-darwin")
    else()
      set(_toolset "intel-linux")
    endif()
  endif()
```

1.  我们根据所需组件准备要构建的库列表。我们定义了一些列表变量：`_build_byproducts`，用于包含将要构建的库的绝对路径；`_b2_select_libraries`，用于包含我们想要构建的库列表；以及`_bootstrap_select_libraries`，这是一个内容相同但格式不同的字符串：

```cpp
  if(NOT "${BOOST_COMPONENTS_REQUIRED}" STREQUAL "")
    # Replace unit_test_framework (used by CMake's find_package) with test (understood by Boost build toolchain)
    string(REPLACE "unit_test_framework" "test" _b2_needed_components "${BOOST_COMPONENTS_REQUIRED}")
    # Generate argument for BUILD_BYPRODUCTS
    set(_build_byproducts)
    set(_b2_select_libraries)
    foreach(_lib IN LISTS _b2_needed_components)
      list(APPEND _build_byproducts ${STAGED_INSTALL_PREFIX}/boost/lib/libboost_${_lib}${CMAKE_SHARED_LIBRARY_SUFFIX})
      list(APPEND _b2_select_libraries --with-${_lib})
    endforeach()
    # Transform the ;-separated list to a ,-separated list (digested by the Boost build toolchain!)
    string(REPLACE ";" "," _b2_needed_components "${_b2_needed_components}")
    set(_bootstrap_select_libraries "--with-libraries=${_b2_needed_components}")
    string(REPLACE ";" ", " printout "${BOOST_COMPONENTS_REQUIRED}")
    message(STATUS "  Libraries to be built: ${printout}")
  endif()
```

1.  我们现在可以将 Boost 项目作为外部项目添加。首先，我们在**下载**选项类中指定下载 URL 和校验和。将`DOWNLOAD_NO_PROGRESS`设置为`1`以抑制打印下载进度信息：

```cpp
include(ExternalProject)
ExternalProject_Add(boost_external
  URL
    https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.zip
  URL_HASH
    SHA256=02d420e6908016d4ac74dfc712eec7d9616a7fc0da78b0a1b5b937536b2e01e8
  DOWNLOAD_NO_PROGRESS
    1
```

1.  接下来，我们设置**更新/修补**和**配置**选项：

```cpp
 UPDATE_COMMAND
   ""
 CONFIGURE_COMMAND
   <SOURCE_DIR>/bootstrap.sh
     --with-toolset=${_toolset}
     --prefix=${STAGED_INSTALL_PREFIX}/boost
     ${_bootstrap_select_libraries}
```

1.  使用`BUILD_COMMAND`指令设置构建选项。将`BUILD_IN_SOURCE`设置为`1`以指示构建将在源目录内发生。此外，我们将`LOG_BUILD`设置为`1`以将构建脚本的输出记录到文件中：

```cpp
  BUILD_COMMAND
    <SOURCE_DIR>/b2 -q
         link=shared
         threading=multi
         variant=release
         toolset=${_toolset}
         ${_b2_select_libraries}
  LOG_BUILD
    1
  BUILD_IN_SOURCE
    1
```

1.  使用`INSTALL_COMMAND`指令设置安装选项。注意使用`LOG_INSTALL`选项也将安装步骤记录到文件中：

```cpp
  INSTALL_COMMAND
    <SOURCE_DIR>/b2 -q install
         link=shared
         threading=multi
         variant=release
         toolset=${_toolset}
         ${_b2_select_libraries}
  LOG_INSTALL
    1
```

1.  最后，我们将我们的库列为`BUILD_BYPRODUCTS`并关闭`ExternalProject_Add`命令：

```cpp
  BUILD_BYPRODUCTS
    "${_build_byproducts}"
  )
```

1.  我们设置了一些对指导新安装的 Boost 检测有用的变量：

```cpp
set(
  BOOST_ROOT ${STAGED_INSTALL_PREFIX}/boost
  CACHE PATH "Path to internally built Boost installation root"
  FORCE
  )
set(
  BOOST_INCLUDEDIR ${BOOST_ROOT}/include
  CACHE PATH "Path to internally built Boost include directories"
  FORCE
  )
set(
  BOOST_LIBRARYDIR ${BOOST_ROOT}/lib
  CACHE PATH "Path to internally built Boost library directories"
  FORCE
  )
```

1.  在条件分支的最后执行的操作是取消设置所有内部变量：

```cpp
  unset(_toolset)
  unset(_b2_needed_components)
  unset(_build_byproducts)
  unset(_b2_select_libraries)
  unset(_boostrap_select_libraries)
```

最后，让我们看看`src/CMakeLists.txt`。该文件描述了一个独立项目：

1.  我们声明一个 C++项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-02_core LANGUAGES CXX)
```

1.  项目依赖于 Boost，我们调用`find_package`。从根目录的`CMakeLists.txt`配置项目保证了依赖项始终得到满足，无论是使用系统上预装的 Boost 还是我们作为子项目构建的 Boost：

```cpp
find_package(Boost 1.61 REQUIRED COMPONENTS filesystem)
```

1.  我们添加我们的示例可执行目标，描述其链接库：

```cpp
add_executable(path-info path-info.cpp)

target_link_libraries(path-info
  PUBLIC
    Boost::filesystem
  )
```

虽然导入目标的使用很整洁，但并不能保证对任意 Boost 和 CMake 版本组合都能正常工作。这是因为 CMake 的`FindBoost.cmake`模块手动创建了导入目标，所以如果 CMake 发布时不知道 Boost 版本，将会有`Boost_LIBRARIES`和`Boost_INCLUDE_DIRS`，但没有导入目标（另请参见[`stackoverflow.com/questions/42123509/cmake-finds-boost-but-the-imported-targets-not-available-for-boost-version`](https://stackoverflow.com/questions/42123509/cmake-finds-boost-but-the-imported-targets-not-available-for-boost-version)）。

# 工作原理

本食谱展示了如何利用超级构建模式来集结项目的依赖项。让我们再次审视项目的布局：

```cpp
.
├── CMakeLists.txt
├── external
│   └── upstream
│       ├── boost
│       │   └── CMakeLists.txt
│       └── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── path-info.cpp
```

我们在项目源树中引入了四个`CMakeLists.txt`文件：

1.  根目录的`CMakeLists.txt`将协调超级构建。

1.  位于`external/upstream`的文件将引导我们到`boost`叶目录。

1.  `external/upstream/boost/CMakeLists.txt`将负责处理 Boost 依赖项。

1.  最后，位于`src`下的`CMakeLists.txt`将构建我们的示例代码，该代码依赖于 Boost。

让我们从`external/upstream/boost/CMakeLists.txt`文件开始讨论。Boost 使用自己的构建系统，因此我们需要在`ExternalProject_Add`中稍微详细一些，以确保一切正确设置：

1.  我们保留**目录**选项的默认值。

1.  **下载**步骤将从 Boost 的在线服务器下载所需版本的存档。因此，我们设置了`URL`和`URL_HASH`。后者用于检查下载存档的完整性。由于我们不希望看到下载的进度报告，我们还设置了`DOWNLOAD_NO_PROGRESS`选项为 true。

1.  **更新**步骤留空。如果需要重新构建，我们不希望再次下载 Boost。

1.  **配置**步骤将使用 Boost 提供的本地配置工具，在`CONFIGURE_COMMAND`中。由于我们希望超级构建是跨平台的，我们使用`<SOURCE_DIR>`变量来引用解压源代码的位置：

```cpp
CONFIGURE_COMMAND
  <SOURCE_DIR>/bootstrap.sh
  --with-toolset=${_toolset}
  --prefix=${STAGED_INSTALL_PREFIX}/boost
  ${_bootstrap_select_libraries}
```

1.  **构建**选项声明了一个*源码内*构建，通过将`BUILD_IN_SOURCE`选项设置为 true。`BUILD_COMMAND`使用 Boost 的本地构建工具`b2`。由于我们将进行源码内构建，我们再次使用`<SOURCE_DIR>`变量来引用解压源代码的位置。

1.  接下来，我们转向**安装**选项。Boost 使用相同的本地构建工具进行管理。实际上，构建和安装命令可以很容易地合并为一个。

1.  **输出**日志选项`LOG_BUILD`和`LOG_INSTALL`指示`ExternalProject_Add`为构建和安装操作编写日志文件，而不是输出到屏幕。

1.  最后，`BUILD_BYPRODUCTS`选项允许`ExternalProject_Add`在后续构建中跟踪新近构建的 Boost 库，即使它们的修改时间可能不会更新。

Boost 构建完成后，构建目录中的`${STAGED_INSTALL_PREFIX}/boost`文件夹将包含我们所需的库。我们需要将此信息传递给我们的项目，其构建系统在`src/CMakeLists.txt`中生成。为了实现这一目标，我们在根`CMakeLists.txt`中的`ExternalProject_Add`中传递两个额外的`CMAKE_CACHE_ARGS`：

1.  `CMAKE_INCLUDE_PATH`：CMake 查找 C/C++头文件的路径

1.  `CMAKE_LIBRARY_PATH`：CMake 查找库的路径

通过将这些变量设置为我们新近构建的 Boost 安装，我们确保依赖项将被正确地检测到。

在配置项目时将`CMAKE_DISABLE_FIND_PACKAGE_Boost`设置为`ON`，将跳过 Boost 库的检测并始终执行超级构建。请参阅文档：[`cmake.org/cmake/help/v3.5/variable/CMAKE_DISABLE_FIND_PACKAGE_PackageName.html`](https://cmake.org/cmake/help/v3.5/variable/CMAKE_DISABLE_FIND_PACKAGE_PackageName.html)

# 使用超级构建管理依赖项：II. FFTW 库

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-03)找到，并包含一个 C 语言示例。该示例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

超级构建模式可用于管理 CMake 支持的所有语言项目的相当复杂的依赖关系。如前一示例所示，各个子项目并非必须由 CMake 管理。与前一示例相反，本示例中的外部子项目将是一个 CMake 项目，并将展示如何使用超级构建下载、构建和安装 FFTW 库。FFTW 是一个快速傅里叶变换库，可免费在[`www.fftw.org`](http://www.fftw.org/)获取。

# 准备就绪

本示例的目录布局展示了超级构建的熟悉结构：

```cpp
.
├── CMakeLists.txt
├── external
│   └── upstream
│       ├── CMakeLists.txt
│       └── fftw3
│           └── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── fftw_example.c
```

我们项目的代码`fftw_example.c`位于`src`子目录中，并将计算源代码中定义的函数的傅里叶变换。

# 如何操作

让我们从根`CMakeLists.txt`开始。此文件组合了整个超级构建过程：

1.  我们声明一个 C99 项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
```

1.  与前一示例一样，我们设置`EP_BASE`目录属性和暂存安装前缀：

```cpp
set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/subprojects)

set(STAGED_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage)
message(STATUS "${PROJECT_NAME} staged install: ${STAGED_INSTALL_PREFIX}")
```

1.  FFTW 的依赖关系在`external/upstream`子目录中进行检查，我们继续将此子目录添加到构建系统中：

```cpp
add_subdirectory(external/upstream)
```

1.  我们包含`ExternalProject.cmake`模块：

```cpp
include(ExternalProject)
```

1.  我们声明`recipe-03_core`外部项目。该项目的源代码位于`${CMAKE_CURRENT_LIST_DIR}/src`文件夹中。该项目设置为使用`FFTW3_DIR`选项选择正确的 FFTW 库：

```cpp
ExternalProject_Add(${PROJECT_NAME}_core
  DEPENDS
    fftw3_external
  SOURCE_DIR
    ${CMAKE_CURRENT_LIST_DIR}/src
  CMAKE_ARGS
    -DFFTW3_DIR=${FFTW3_DIR}
    -DCMAKE_C_STANDARD=${CMAKE_C_STANDARD}
    -DCMAKE_C_EXTENSIONS=${CMAKE_C_EXTENSIONS}
    -DCMAKE_C_STANDARD_REQUIRED=${CMAKE_C_STANDARD_REQUIRED}
  CMAKE_CACHE_ARGS
    -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
  BUILD_ALWAYS
    1
  INSTALL_COMMAND
    ""
  )
```

在`external/upstream`子目录中还包含一个`CMakeLists.txt`：

1.  在此文件中，我们将`fftw3`文件夹添加为构建系统中的另一个子目录：

```cpp
add_subdirectory(fftw3)
```

`external/upstream/fftw3`中的`CMakeLists.txt`负责我们的依赖关系：

1.  首先，我们尝试在系统上查找 FFTW3 库。请注意，我们使用了`find_package`的`CONFIG`参数：

```cpp
find_package(FFTW3 CONFIG QUIET)
```

1.  如果找到了库，我们可以使用导入的目标`FFTW3::fftw3`与之链接。我们向用户打印一条消息，显示库的位置。我们添加一个虚拟的`INTERFACE`库`fftw3_external`。这在超级构建中子项目之间的依赖树正确修复时是必需的：

```cpp
find_package(FFTW3 CONFIG QUIET)

if(FFTW3_FOUND)
  get_property(_loc TARGET FFTW3::fftw3 PROPERTY LOCATION)
  message(STATUS "Found FFTW3: ${_loc} (found version ${FFTW3_VERSION})")
  add_library(fftw3_external INTERFACE) # dummy
else()
  # this branch will be discussed below
endif()
```

1.  如果 CMake 无法找到预安装的 FFTW 版本，我们进入条件语句的 else 分支，在其中我们使用`ExternalProject_Add`下载、构建和安装它。外部项目的名称为`fftw3_external`。`fftw3_external`项目将从官方在线档案下载。下载的完整性将使用 MD5 校验和进行检查：

```cpp
message(STATUS "Suitable FFTW3 could not be located. Downloading and building!")

include(ExternalProject)
ExternalProject_Add(fftw3_external
  URL
    http://www.fftw.org/fftw-3.3.8.tar.gz
  URL_HASH
    MD5=8aac833c943d8e90d51b697b27d4384d
```

1.  我们禁用下载的进度打印，并将更新命令定义为空：

```cpp
  DOWNLOAD_NO_PROGRESS
    1
  UPDATE_COMMAND
    ""
```

1.  配置、构建和安装输出将被记录到文件中：

```cpp
  LOG_CONFIGURE
    1
  LOG_BUILD
    1
  LOG_INSTALL
    1
```

1.  我们将`fftw3_external`项目的安装前缀设置为之前定义的`STAGED_INSTALL_PREFIX`目录，并关闭 FFTW3 的测试套件构建：

```cpp
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${STAGED_INSTALL_PREFIX}
    -DBUILD_TESTS=OFF
```

1.  如果我们在 Windows 上构建，我们通过生成表达式设置`WITH_OUR_MALLOC`预处理器选项，并关闭`ExternalProject_Add`命令：

```cpp
  CMAKE_CACHE_ARGS
    -DCMAKE_C_FLAGS:STRING=$<$<BOOL:WIN32>:-DWITH_OUR_MALLOC>
  )
```

1.  最后，我们定义了`FFTW3_DIR`变量并将其缓存。该变量将由 CMake 用作导出的`FFTW3::fftw3`目标的搜索目录：

```cpp
include(GNUInstallDirs)

set(
  FFTW3_DIR ${STAGED_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/fftw3
  CACHE PATH "Path to internally built FFTW3Config.cmake"
  FORCE
  )
```

位于`src`文件夹中的`CMakeLists.txt`文件相当简洁：

1.  同样在这个文件中，我们声明了一个 C 项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03_core LANGUAGES C)
```

1.  我们调用`find_package`来检测 FFTW 库。再次使用`CONFIG`检测模式：

```cpp
find_package(FFTW3 CONFIG REQUIRED)
get_property(_loc TARGET FFTW3::fftw3 PROPERTY LOCATION)
message(STATUS "Found FFTW3: ${_loc} (found version ${FFTW3_VERSION})")
```

1.  我们将`fftw_example.c`源文件添加到可执行目标`fftw_example`中：

```cpp
add_executable(fftw_example fftw_example.c)
```

1.  我们为目标可执行文件设置链接库：

```cpp
target_link_libraries(fftw_example
  PRIVATE
    FFTW3::fftw3
  )
```

# 工作原理

本示例展示了如何下载、构建和安装由 CMake 管理的构建系统的外部项目。与之前的示例不同，那里必须使用自定义构建系统，这种超级构建设置相对简洁。值得注意的是，`find_package`命令使用了`CONFIG`选项；这告诉 CMake 首先查找`FFTW3Config.cmake`文件以定位 FFTW3 库。这样的文件将库作为目标导出，供第三方项目使用。目标包含版本、配置和库的位置，即有关目标如何配置和构建的完整信息。如果系统上未安装该库，我们需要告诉 CMake`FFTW3Config.cmake`文件的位置。这可以通过设置`FFTW3_DIR`变量来完成。这是在`external/upstream/fftw3/CMakeLists.txt`文件的最后一步，通过使用`GNUInstallDirs.cmake`模块，我们将`FFTW3_DIR`设置为缓存变量，以便稍后在超级构建中被拾取。

在配置项目时将`CMAKE_DISABLE_FIND_PACKAGE_FFTW3`设置为`ON`，将跳过 FFTW 库的检测并始终执行超级构建。请参阅文档：[`cmake.org/cmake/help/v3.5/variable/CMAKE_DISABLE_FIND_PACKAGE_PackageName.html`](https://cmake.org/cmake/help/v3.5/variable/CMAKE_DISABLE_FIND_PACKAGE_PackageName.html)

# 使用超级构建管理依赖项：III. Google Test 框架

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-04)找到，并包含一个 C++示例。该示例适用于 CMake 版本 3.11（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。代码仓库还包含一个与 CMake 3.5 兼容的示例。

在第四章，*创建和运行测试*，第 3 个菜谱，*定义单元测试并链接到 Google Test*，我们使用 Google Test 框架实现了单元测试，并在配置时使用相对较新的`FetchContent`模块（自 CMake 3.11 起可用）获取了 Google Test 源码。在本章中，我们将重温这个菜谱，减少对测试方面的关注，并深入探讨`FetchContent`，它提供了一个紧凑且多功能的模块，用于在配置时组装项目依赖。为了获得更多见解，以及对于 CMake 3.11 以下的版本，我们还将讨论如何使用`ExternalProject_Add` *在配置时*模拟`FetchContent`。

# 准备工作

在本菜谱中，我们将构建并测试与第四章，*创建和运行测试*，第 3 个菜谱，*定义单元测试并链接到 Google Test*中相同的源文件，`main.cpp`、`sum_integers.cpp`、`sum_integers.hpp`和`test.cpp`。我们将使用`FetchContent`或`ExternalProject_Add`在配置时下载所有必需的 Google Test 源码，并且在本菜谱中只关注在配置时获取依赖，而不是实际的源码及其单元测试。

# 如何操作

在本菜谱中，我们将只关注如何获取 Google Test 源码以构建`gtest_main`目标。关于如何使用该目标测试示例源码的讨论，我们请读者参考第四章，*创建和运行测试*，第 3 个菜谱，*定义单元测试并链接到 Google Test*：

1.  我们首先包含`FetchContent`模块，它将提供我们所需的函数来声明、查询和填充依赖：

```cpp
include(FetchContent)
```

1.  接着，我们声明内容——其名称、仓库位置以及要获取的确切版本：

```cpp
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)
```

1.  然后我们查询内容是否已经被获取/填充：

```cpp
FetchContent_GetProperties(googletest)
```

1.  之前的函数调用定义了`googletest_POPULATED`。如果内容尚未填充，我们将获取内容并配置子项目：

```cpp
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # ...

  # adds the targets: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )

  # ...

endif()
```

1.  注意内容是在配置时获取的：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
```

1.  这将生成以下构建目录树。Google Test 源码现在已就位，可以由 CMake 处理并提供所需的目标：

```cpp
build/
├── ...
├── _deps
│   ├── googletest-build
│   │   ├── ...
│   │   └── ...
│   ├── googletest-src
│   │   ├── ...
│   │   └── ...
│   └── googletest-subbuild
│       ├── ...
│       └── ...
└── ...
```

# 它是如何工作的

`FetchContent`模块允许在配置时填充内容。在我们的例子中，我们获取了一个带有明确 Git 标签的 Git 仓库：

```cpp
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)
```

`FetchContent`模块支持通过`ExternalProject`模块支持的任何方法*获取*内容 - 换句话说，*通过*Subversion、Mercurial、CVS 或 HTTP(S)。内容名称“googletest”是我们的选择，有了这个，我们将能够在查询其属性、填充目录以及稍后配置子项目时引用内容。在填充项目之前，我们检查内容是否已经获取，否则如果`FetchContent_Populate()`被调用超过一次，它将抛出错误：

```cpp
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # ...

endif()
```

只有在那时我们才配置了子目录，我们可以通过`googletest_SOURCE_DIR`和`googletest_BINARY_DIR`变量来引用它。这些变量是由`FetchContent_Populate(googletest)`设置的，并根据我们在声明内容时给出的项目名称构建的。

```cpp
add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
  )
```

`FetchContent`模块有许多选项（参见[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)），这里我们可以展示一个：如何更改外部项目将被放置的默认路径。之前，我们看到默认情况下内容被保存到`${CMAKE_BINARY_DIR}/_deps`。我们可以通过设置`FETCHCONTENT_BASE_DIR`来更改此位置：

```cpp
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/custom)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)
```

`FetchContent`已成为 CMake 3.11 版本中的标准部分。在下面的代码中，我们将尝试在*配置时间*使用`ExternalProject_Add`来模拟`FetchContent`。这不仅对旧版本的 CMake 实用，而且有望让我们更深入地了解`FetchContent`层下面发生的事情，并提供一个有趣的替代方案，以替代使用`ExternalProject_Add`在构建时间获取项目的典型方式。我们的目标是编写一个`fetch_git_repo`宏，并将其放置在`fetch_git_repo.cmake`中，以便我们可以这样获取内容：

```cpp
include(fetch_git_repo.cmake)

fetch_git_repo(
  googletest
  ${CMAKE_BINARY_DIR}/_deps
  https://github.com/google/googletest.git
  release-1.8.0
)

# ...

# adds the targets: gtest, gtest_main, gmock, gmock_main
add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
  )

# ...
```

这感觉类似于使用`FetchContent`。在幕后，我们将使用`ExternalProject_Add`。现在让我们揭开盖子，检查`fetch_git_repo`在`fetch_git_repo.cmake`中的定义：

```cpp
macro(fetch_git_repo _project_name _download_root _git_url _git_tag)

  set(${_project_name}_SOURCE_DIR ${_download_root}/${_project_name}-src)
  set(${_project_name}_BINARY_DIR ${_download_root}/${_project_name}-build)

  # variables used configuring fetch_git_repo_sub.cmake
  set(FETCH_PROJECT_NAME ${_project_name})
  set(FETCH_SOURCE_DIR ${${_project_name}_SOURCE_DIR})
  set(FETCH_BINARY_DIR ${${_project_name}_BINARY_DIR})
  set(FETCH_GIT_REPOSITORY ${_git_url})
  set(FETCH_GIT_TAG ${_git_tag})

  configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/fetch_at_configure_step.in
    ${_download_root}/CMakeLists.txt
    @ONLY
    )

  # undefine them again
  unset(FETCH_PROJECT_NAME)
  unset(FETCH_SOURCE_DIR)
  unset(FETCH_BINARY_DIR)
  unset(FETCH_GIT_REPOSITORY)
  unset(FETCH_GIT_TAG)

  # configure sub-project
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY
      ${_download_root}
    )
  # build sub-project which triggers ExternalProject_Add
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY
      ${_download_root}
    )
endmacro()
```

宏接收项目名称、下载根目录、Git 仓库 URL 和 Git 标签。宏定义了`${_project_name}_SOURCE_DIR`和`${_project_name}_BINARY_DIR`，我们使用宏而不是函数，因为`${_project_name}_SOURCE_DIR`和`${_project_name}_BINARY_DIR`需要在`fetch_git_repo`的作用域之外存活，因为我们稍后在主作用域中使用它们来配置子目录：

```cpp
add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
  )
```

在`fetch_git_repo`宏内部，我们希望使用`ExternalProject_Add`在*配置时间*获取外部项目，我们通过一个三步的技巧来实现这一点：

1.  首先，我们配置`fetch_at_configure_step.in`：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(fetch_git_repo_sub LANGUAGES NONE)

include(ExternalProject)

ExternalProject_Add(
  @FETCH_PROJECT_NAME@
  SOURCE_DIR "@FETCH_SOURCE_DIR@"
  BINARY_DIR "@FETCH_BINARY_DIR@"
  GIT_REPOSITORY
    @FETCH_GIT_REPOSITORY@
  GIT_TAG
    @FETCH_GIT_TAG@
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
  )
```

使用`configure_file`，我们生成一个`CMakeLists.txt`文件，其中之前的占位符被替换为在`fetch_git_repo.cmake`中定义的值。注意，之前的`ExternalProject_Add`命令被构造为仅获取，而不进行配置、构建、安装或测试。

1.  其次，我们在配置时间（从根项目的角度）使用配置步骤触发`ExternalProject_Add`：

```cpp
# configure sub-project
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . 
  WORKING_DIRECTORY
    ${_download_root}
  ) 
```

1.  第三个也是最后一个技巧在`fetch_git_repo.cmake`中触发配置时间构建步骤：

```cpp
# build sub-project which triggers ExternalProject_Add
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" --build . 
  WORKING_DIRECTORY
    ${_download_root}
  )
```

这个解决方案的一个很好的方面是，由于外部依赖项不是由`ExternalProject_Add`配置的，我们不需要通过`ExternalProject_Add`调用将任何配置设置传递给项目。我们可以使用`add_subdirectory`配置和构建模块，就好像外部依赖项是我们项目源代码树的一部分一样。巧妙的伪装！

# 另请参阅

有关可用的`FetchContent`选项的详细讨论，请咨询[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)。

配置时间`ExternalProject_Add`解决方案的灵感来自 Craig Scott 的工作和博客文章：[`crascit.com/2015/07/25/cmake-gtest/`](https://crascit.com/2015/07/25/cmake-gtest/)。

# 将您的项目作为超级构建进行管理

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-08/recipe-05)获取，并且有一个 C++示例。本示例适用于 CMake 版本 3.6（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

`ExternalProject`和`FetchContent`是 CMake 工具箱中的两个非常强大的工具。之前的示例应该已经说服了您超级构建方法在管理具有复杂依赖关系的项目方面的多功能性。到目前为止，我们已经展示了如何使用`ExternalProject`来处理以下内容：

+   存储在您的源代码树中的源代码

+   从在线服务器上的档案中检索来源

之前的示例展示了如何使用`FetchContent`来处理来自开源 Git 存储库的依赖项。本示例将展示如何使用`ExternalProject`达到相同的效果。最后一个示例将介绍一个将在第 4 个示例中重复使用的示例，即*安装超级构建*，在第十章，*编写安装程序*。

# 准备工作

这个超级构建的源代码树现在应该感觉很熟悉：

```cpp
.
├── CMakeLists.txt
├── external
│   └── upstream
│       ├── CMakeLists.txt
│       └── message
│           └── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── use_message.cpp
```

根目录有一个`CMakeLists.txt`，我们已经知道它将协调超级构建。叶目录`src`和`external`托管我们自己的源代码和满足对`message`库的依赖所需的 CMake 指令，我们将在本示例中构建该库。

# 如何操作

到目前为止，设置超级构建的过程应该感觉很熟悉。让我们再次看一下必要的步骤，从根`CMakeLists.txt`开始：

1.  我们声明了一个具有相同默认构建类型的 C++11 项目：

```cpp
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(recipe-05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}")
```

1.  设置了`EP_BASE`目录属性。这将固定由`ExternalProject`管理的所有子项目的布局：

```cpp
set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/subprojects)
```

1.  我们设置了`STAGED_INSTALL_PREFIX`。与之前一样，此位置将用作构建树中依赖项的安装前缀：

```cpp
set(STAGED_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage)
message(STATUS "${PROJECT_NAME} staged install: ${STAGED_INSTALL_PREFIX}")
```

1.  我们添加`external/upstream`子目录：

```cpp
add_subdirectory(external/upstream)
```

1.  我们自己的项目也将由超级构建管理，因此使用`ExternalProject_Add`添加：

```cpp
include(ExternalProject)
ExternalProject_Add(${PROJECT_NAME}_core
  DEPENDS
    message_external
  SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
    -Dmessage_DIR=${message_DIR}
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
  BUILD_ALWAYS
    1
  INSTALL_COMMAND
    ""
  )
```

`external/upstream`中的`CMakeLists.txt`只包含一个命令：

```cpp
add_subdirectory(message)
```

跳转到`message`文件夹，我们再次看到管理我们对`message`库依赖的常用命令：

1.  首先，我们调用`find_package`来找到一个合适的库版本：

```cpp
find_package(message 1 CONFIG QUIET)
```

1.  如果找到，我们通知用户并添加一个虚拟的`INTERFACE`库：

```cpp
get_property(_loc TARGET message::message-shared PROPERTY LOCATION)
message(STATUS "Found message: ${_loc} (found version ${message_VERSION})")
add_library(message_external INTERFACE) # dummy
```

1.  如果未找到，我们再次通知用户并继续使用`ExternalProject_Add`：

```cpp
message(STATUS "Suitable message could not be located, Building message instead.")
```

1.  该项目托管在一个公共 Git 仓库中，我们使用`GIT_TAG`选项来指定下载哪个分支。像之前一样，我们让`UPDATE_COMMAND`选项保持空白：

```cpp
include(ExternalProject)
ExternalProject_Add(message_external
  GIT_REPOSITORY
    https://github.com/dev-cafe/message.git
  GIT_TAG
    master
  UPDATE_COMMAND
    ""
```

1.  外部项目使用 CMake 进行配置和构建。我们传递所有必要的构建选项：

```cpp
 CMAKE_ARGS
   -DCMAKE_INSTALL_PREFIX=${STAGED_INSTALL_PREFIX}
   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
   -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
   -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
   -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
 CMAKE_CACHE_ARGS
   -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
```

1.  我们决定在项目安装后进行测试：

```cpp
  TEST_AFTER_INSTALL
    1
```

1.  我们不希望看到下载进度，也不希望屏幕上显示配置、构建和安装的信息，我们关闭`ExternalProject_Add`命令：

```cpp
  DOWNLOAD_NO_PROGRESS
    1
  LOG_CONFIGURE
    1
  LOG_BUILD
    1
  LOG_INSTALL
    1
  )
```

1.  为了确保子项目在超级构建的其余部分中可被发现，我们设置`message_DIR`目录：

```cpp
if(WIN32 AND NOT CYGWIN)
  set(DEF_message_DIR ${STAGED_INSTALL_PREFIX}/CMake)
else()
  set(DEF_message_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/message)
endif()

file(TO_NATIVE_PATH "${DEF_message_DIR}" DEF_message_DIR)
set(message_DIR ${DEF_message_DIR}
    CACHE PATH "Path to internally built messageConfig.cmake" FORCE)
```

最后，让我们看看`src`文件夹中的`CMakeLists.txt`：

1.  再次，我们声明一个 C++11 项目：

```cpp
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(recipe-05_core
  LANGUAGES CXX
  )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  这个项目需要`message`库：

```cpp
find_package(message 1 CONFIG REQUIRED)
get_property(_loc TARGET message::message-shared PROPERTY LOCATION)
message(STATUS "Found message: ${_loc} (found version ${message_VERSION})")
```

1.  我们声明一个可执行目标，并将其链接到我们依赖项提供的`message-shared`库：

```cpp
add_executable(use_message use_message.cpp)

target_link_libraries(use_message
  PUBLIC
    message::message-shared
  )
```

# 它是如何工作的

这个配方突出了`ExternalProject_Add`命令的一些新选项：

1.  `GIT_REPOSITORY`：这可以用来指定包含我们依赖源代码的仓库的 URL。CMake 还可以使用其他版本控制系统，如 CVS（`CVS_REPOSITORY`）、SVN（`SVN_REPOSITORY`）或 Mercurial（`HG_REPOSITORY`）。

1.  `GIT_TAG`：默认情况下，CMake 将检出给定仓库的默认分支。然而，依赖于一个已知稳定的定义良好的版本是更可取的。这可以通过这个选项来指定，它可以接受 Git 识别为“版本”信息的任何标识符，如 Git 提交 SHA、Git 标签，或者仅仅是一个分支名称。对于 CMake 理解的其他版本控制系统，也有类似的选项。

1.  `TEST_AFTER_INSTALL`：很可能，你的依赖项有自己的测试套件，你可能想要运行测试套件以确保超级构建过程中一切顺利。这个选项将在安装步骤之后立即运行测试。

下面是`ExternalProject_Add`理解的额外**测试**选项：

+   `TEST_BEFORE_INSTALL`，它将在安装步骤*之前*运行测试套件

+   `TEST_EXCLUDE_FROM_MAIN`，我们可以使用它从测试套件中移除对外部项目主要目标的依赖

这些选项假设外部项目使用 CTest 管理测试。如果外部项目不使用 CTest 管理测试，我们可以设置`TEST_COMMAND`选项来执行测试。

引入超级构建模式，即使对于项目中包含的模块，也会带来额外的层次，重新声明小型 CMake 项目，并通过`ExternalProject_Add`显式传递配置设置。引入这一额外层次的好处是变量和目标作用域的清晰分离，这有助于管理复杂性、依赖关系和由多个组件组成的项目的命名空间，这些组件可以是内部的或外部的，并通过 CMake 组合在一起。
