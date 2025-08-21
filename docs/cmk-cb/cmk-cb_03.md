# 第三章：检测环境

在本章中，我们将介绍以下食谱：

+   发现操作系统

+   处理依赖于平台的源代码

+   处理依赖于编译器的源代码

+   发现主机处理器架构

+   发现主机处理器指令集

+   为 Eigen 库启用矢量化

# 引言

尽管 CMake 是跨平台的，在我们的项目中我们努力使源代码能够在不同平台、操作系统和编译器之间移植，但有时源代码并不完全可移植；例如，当使用依赖于供应商的扩展时，我们可能会发现有必要根据平台以略有不同的方式配置和/或构建代码。这对于遗留代码或交叉编译尤其相关，我们将在第十三章，*替代生成器和交叉编译*中回到这个话题。了解处理器指令集以针对特定目标平台优化性能也是有利的。本章提供了检测此类环境的食谱，并提供了如何实施此类解决方案的建议。

# 发现操作系统

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-01)找到。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

尽管 CMake 是一套跨平台的工具，但了解配置或构建步骤在哪个操作系统（OS）上执行仍然非常有用。这种操作系统检测可以用来调整 CMake 代码以适应特定的操作系统，根据操作系统启用条件编译，或者在可用或必要时使用编译器特定的扩展。在本食谱中，我们将展示如何使用 CMake 来检测操作系统，并通过一个不需要编译任何源代码的示例来说明。为了简单起见，我们只考虑配置步骤。

# 如何操作

我们将通过一个非常简单的`CMakeLists.txt`来演示操作系统检测：

1.  我们首先定义最小 CMake 版本和项目名称。请注意，我们的语言要求是`NONE`：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES NONE)

```

1.  然后我们希望根据检测到的操作系统打印一条自定义消息：

```cpp
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "Configuring on/for macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring on/for Windows")
elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  message(STATUS "Configuring on/for IBM AIX")
else()
  message(STATUS "Configuring on/for ${CMAKE_SYSTEM_NAME}")
endif()
```

在尝试之前，首先检查前面的代码块，并考虑你期望在你的系统上看到的行为。

1.  现在我们准备测试并配置项目：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
```

1.  在 CMake 的输出中，有一行在这里很有趣——在 Linux 系统上，这是感兴趣的行（在其他系统上，输出可能会有所不同）：

```cpp
-- Configuring on/for Linux
```

# 它是如何工作的

CMake 正确地为目标操作系统定义了`CMAKE_SYSTEM_NAME`，因此通常不需要使用自定义命令、工具或脚本来查询此信息。该变量的值随后可用于实现操作系统特定的条件和解决方法。在具有`uname`命令的系统上，该变量设置为`uname -s`的输出。在 macOS 上，该变量设置为"Darwin"。在 Linux 和 Windows 上，它分别评估为"Linux"和"Windows"。现在我们知道，如果我们需要在特定操作系统上执行特定的 CMake 代码，该如何操作。当然，我们应该尽量减少这种定制，以便简化迁移到新平台的过程。

为了在从一个平台迁移到另一个平台时尽量减少麻烦，应避免直接使用 Shell 命令，并避免使用显式的路径分隔符（Linux 和 macOS 上的正斜杠和 Windows 上的反斜杠）。在 CMake 代码中只使用正斜杠作为路径分隔符，CMake 会自动为所涉及的操作系统环境进行转换。

# 处理平台依赖的源代码

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-02)找到，并包含一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

理想情况下，我们应该避免平台依赖的源代码，但有时我们别无选择——尤其是当我们被给予需要配置和编译的代码时，而这些代码并非我们自己编写的。在本食谱中，我们将演示如何使用 CMake 根据操作系统有条件地编译源代码。

# 准备工作

对于这个例子，我们将修改来自第一章，*从简单可执行文件到库*，食谱 1，*将单个源文件编译成可执行文件*的`hello-world.cpp`示例代码：

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() {
#ifdef IS_WINDOWS
  return std::string("Hello from Windows!");
#elif IS_LINUX
  return std::string("Hello from Linux!");
#elif IS_MACOS
  return std::string("Hello from macOS!");
#else
  return std::string("Hello from an unknown system!");
#endif
}

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

# 如何操作

让我们构建一个对应的`CMakeLists.txt`实例，这将使我们能够根据目标操作系统有条件地编译源代码：

1.  我们首先设置最小 CMake 版本、项目名称和支持的语言：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-02 LANGUAGES CXX)
```

1.  然后我们定义可执行文件及其对应的源文件：

```cpp
add_executable(hello-world hello-world.cpp)
```

1.  然后我们通过定义以下目标编译定义来让预处理器知道系统名称：

```cpp
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(hello-world PUBLIC "IS_LINUX")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_compile_definitions(hello-world PUBLIC "IS_MACOS")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_compile_definitions(hello-world PUBLIC "IS_WINDOWS")
endif()
```

在继续之前，先检查前面的表达式并考虑在你的系统上你期望的行为。

1.  现在我们准备测试并配置项目：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./hello-world

Hello from Linux!
```

在 Windows 系统上，你会看到`Hello from Windows!`；其他操作系统将产生不同的输出。

# 工作原理

在`hello-world.cpp`示例中，有趣的部分是基于预处理器定义`IS_WINDOWS`、`IS_LINUX`或`IS_MACOS`的条件编译：

```cpp
std::string say_hello() {
#ifdef IS_WINDOWS
  return std::string("Hello from Windows!");
#elif IS_LINUX
  return std::string("Hello from Linux!");
#elif IS_MACOS
  return std::string("Hello from macOS!");
#else
  return std::string("Hello from an unknown system!");
#endif
}
```

这些定义在配置时由 CMake 在`CMakeLists.txt`中使用`target_compile_definitions`定义，然后传递给预处理器。我们可以实现一个更紧凑的表达式，而不重复`if-endif`语句，我们将在下一个食谱中演示这种重构。我们还可以将`if-endif`语句合并为一个`if-elseif-elseif-endif`语句。

在这一点上，我们应该指出，我们可以使用`add_definitions(-DIS_LINUX)`（当然，根据所讨论的平台调整定义）而不是使用`target_compile_definitions`来设置定义。使用`add_definitions`的缺点是它修改了整个项目的编译定义，而`target_compile_definitions`给了我们限制定义范围到特定目标的可能性，以及通过使用`PRIVATE`、`PUBLIC`或`INTERFACE`限定符限制这些定义的可见性。这些限定符具有与编译器标志相同的含义，正如我们在第一章，*从简单的可执行文件到库*，第 8 个食谱，*控制编译器标志*中已经看到的：

+   使用`PRIVATE`限定符，编译定义将仅应用于给定目标，而不会被其他消费目标应用。

+   使用`INTERFACE`限定符，编译定义将仅应用于消费该定义的目标。

+   使用`PUBLIC`限定符，编译定义将应用于给定目标以及所有其他消费目标。

尽量减少项目中依赖于平台的源代码，以便更容易移植。

# 处理依赖于编译器的源代码

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-03)找到，并包含 C++和 Fortran 示例。本食谱适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

本食谱与前一个食谱类似，因为我们使用 CMake 来适应依赖于环境的条件源代码的编译：在这种情况下，它将依赖于所选的编译器。同样，为了便携性，这是我们在编写新代码时尽量避免的情况，但这也是我们几乎肯定会在某个时候遇到的情况，尤其是在使用遗留代码或处理依赖于编译器的工具（如 sanitizers）时。从本章和前一章的食谱中，我们已经具备了实现这一点的所有要素。尽管如此，讨论处理依赖于编译器的源代码的问题仍然很有用，因为我们有机会介绍一些新的 CMake 方面。

# 准备就绪

在本配方中，我们将从 C++示例开始，稍后我们将展示一个 Fortran 示例，并尝试重构和简化 CMake 代码。

让我们考虑以下`hello-world.cpp`源代码：

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() {
#ifdef IS_INTEL_CXX_COMPILER
  // only compiled when Intel compiler is selected
  // such compiler will not compile the other branches
  return std::string("Hello Intel compiler!");
#elif IS_GNU_CXX_COMPILER
  // only compiled when GNU compiler is selected
  // such compiler will not compile the other branches
  return std::string("Hello GNU compiler!");
#elif IS_PGI_CXX_COMPILER
  // etc.
  return std::string("Hello PGI compiler!");
#elif IS_XL_CXX_COMPILER
  return std::string("Hello XL compiler!");
#else
  return std::string("Hello unknown compiler - have we met before?");
#endif
}

int main() {
  std::cout << say_hello() << std::endl;
  std::cout << "compiler name is " COMPILER_NAME << std::endl;
  return EXIT_SUCCESS;
}
```

我们还将使用相应的 Fortran 示例（`hello-world.F90`）：

```cpp
program hello

  implicit none

#ifdef IS_Intel_FORTRAN_COMPILER
  print *, 'Hello Intel compiler!'
#elif IS_GNU_FORTRAN_COMPILER
  print *, 'Hello GNU compiler!'
#elif IS_PGI_FORTRAN_COMPILER
  print *, 'Hello PGI compiler!'
#elif IS_XL_FORTRAN_COMPILER
  print *, 'Hello XL compiler!'
#else
  print *, 'Hello unknown compiler - have we met before?'
#endif

end program
```

# 如何做到这一点

我们将在转向 Fortran 示例之前从 C++示例开始：

1.  在`CMakeLists.txt`文件中，我们定义了现在熟悉的最低版本、项目名称和支持的语言：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03 LANGUAGES CXX)
```

1.  然后我们定义可执行目标及其对应的源文件：

```cpp
add_executable(hello-world hello-world.cpp)
```

1.  然后我们通过定义以下目标编译定义，让预处理器了解编译器名称和供应商：

```cpp
target_compile_definitions(hello-world PUBLIC "COMPILER_NAME=\"${CMAKE_CXX_COMPILER_ID}\"")

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    target_compile_definitions(hello-world PUBLIC "IS_INTEL_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_definitions(hello-world PUBLIC "IS_GNU_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES PGI)
    target_compile_definitions(hello-world PUBLIC "IS_PGI_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES XL)
    target_compile_definitions(hello-world PUBLIC "IS_XL_CXX_COMPILER")
endif()
```

之前的配方已经训练了我们的眼睛，现在我们甚至可以预见到结果：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./hello-world

Hello GNU compiler!
```

如果您使用的是不同的编译器供应商，那么此示例代码将提供不同的问候。

在前面的示例和之前的配方中的`CMakeLists.txt`文件中的`if`语句似乎是重复的，作为程序员，我们不喜欢重复自己。我们能更简洁地表达这一点吗？确实可以！为此，让我们转向 Fortran 示例。

在 Fortran 示例的`CMakeLists.txt`文件中，我们需要执行以下操作：

1.  我们需要将语言调整为 Fortran：

```cpp
project(recipe-03 LANGUAGES Fortran)
```

1.  然后我们定义可执行文件及其对应的源文件；在这种情况下，使用大写的`.F90`后缀：

```cpp
add_executable(hello-world hello-world.F90)
```

1.  然后我们通过定义以下目标编译定义，让预处理器非常简洁地了解编译器供应商：

```cpp
target_compile_definitions(hello-world
  PUBLIC "IS_${CMAKE_Fortran_COMPILER_ID}_FORTRAN_COMPILER"
)
```

剩余的 Fortran 示例行为与 C++示例相同。

# 它是如何工作的

预处理器定义是在配置时由 CMake 在`CMakeLists.txt`中定义的，并传递给预处理器。Fortran 示例包含一个非常紧凑的表达式，我们使用`CMAKE_Fortran_COMPILER_ID`变量来构造预处理器定义，使用`target_compile_definitions`。为了适应这一点，我们不得不将“Intel”的案例从`IS_INTEL_CXX_COMPILER`更改为`IS_Intel_FORTRAN_COMPILER`。我们可以通过使用相应的`CMAKE_C_COMPILER_ID`和`CMAKE_CXX_COMPILER_ID`变量为 C 或 C++实现相同的效果。但是请注意，`CMAKE_<LANG>_COMPILER_ID`*并不保证*为所有编译器或语言定义。

对于应该被预处理的 Fortran 代码，使用`.F90`后缀，对于不应该被预处理的代码，使用`.f90`后缀。

# 探索主机处理器架构

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-04)获取，并包含一个 C++示例。该配方适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

20 世纪 70 年代超级计算中 64 位整数运算的出现以及 21 世纪初个人计算机中 64 位寻址的出现扩大了内存寻址范围，并且投入了大量资源将硬编码为 32 位架构的代码移植到支持 64 位寻址。许多博客文章，例如[`www.viva64.com/en/a/0004/`](https://www.viva64.com/en/a/0004/)，都致力于讨论在将 C++代码移植到 64 位平台时遇到的典型问题和解决方案。非常建议以避免明确硬编码限制的方式编程，但您可能处于需要容纳硬编码限制的代码配置与 CMake 的情况，在本菜谱中，我们希望讨论检测宿主处理器架构的选项。

# 准备工作

我们将使用以下`arch-dependent.cpp`示例源代码：

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

std::string say_hello() {
  std::string arch_info(TOSTRING(ARCHITECTURE));
  arch_info += std::string(" architecture. ");
#ifdef IS_32_BIT_ARCH
  return arch_info + std::string("Compiled on a 32 bit host processor.");
#elif IS_64_BIT_ARCH
  return arch_info + std::string("Compiled on a 64 bit host processor.");
#else
  return arch_info + std::string("Neither 32 nor 64 bit, puzzling ...");
#endif
}
int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

# 如何操作

现在让我们转向 CMake 方面。在`CMakeLists.txt`文件中，我们需要应用以下内容：

1.  我们首先定义可执行文件及其源文件依赖项：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-04 LANGUAGES CXX)

add_executable(arch-dependent arch-dependent.cpp)
```

1.  我们检查`void`指针类型的大小。这在`CMAKE_SIZEOF_VOID_P` CMake 变量中定义，并将告诉我们 CPU 是 32 位还是 64 位。我们通过状态消息让用户知道检测到的大小，并设置一个预处理器定义：

```cpp
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_definitions(arch-dependent PUBLIC "IS_64_BIT_ARCH")
  message(STATUS "Target is 64 bits")
else()
  target_compile_definitions(arch-dependent PUBLIC "IS_32_BIT_ARCH")
  message(STATUS "Target is 32 bits")
endif()
```

1.  然后我们通过定义以下目标编译定义让预处理器知道宿主处理器架构，同时在配置期间打印状态消息：

```cpp
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
  message(STATUS "i386 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686")
  message(STATUS "i686 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "x86_64 architecture detected")
else()
  message(STATUS "host processor architecture is unknown")
endif()

target_compile_definitions(arch-dependent
  PUBLIC "ARCHITECTURE=${CMAKE_HOST_SYSTEM_PROCESSOR}"
  )
```

1.  我们配置项目并记录状态消息（当然，确切的消息可能会发生变化）：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Target is 64 bits
-- x86_64 architecture detected
...
```

1.  最后，我们构建并执行代码（实际输出将取决于宿主处理器架构）：

```cpp
$ cmake --build .
$ ./arch-dependent

x86_64 architecture. Compiled on a 64 bit host processor.
```

# 它是如何工作的

CMake 定义了`CMAKE_HOST_SYSTEM_PROCESSOR`变量，其中包含当前正在运行的处理器的名称。这可以设置为“i386”、“i686”、“x86_64”、“AMD64”等，当然，这取决于当前的 CPU。`CMAKE_SIZEOF_VOID_P`被定义为持有指向`void`类型的指针的大小。我们可以在 CMake 级别查询这两个变量，以便修改目标或目标编译定义。使用预处理器定义，我们可以根据检测到的宿主处理器架构分支源代码编译。正如在前面的菜谱中讨论的那样，在编写新代码时应避免这种定制，但在处理遗留代码或进行交叉编译时，有时是有用的，这是第十三章，*替代生成器和交叉编译*的主题。

使用`CMAKE_SIZEOF_VOID_P`是检查当前 CPU 是 32 位还是 64 位架构的唯一真正可移植的方法。

# 还有更多内容

除了`CMAKE_HOST_SYSTEM_PROCESSOR`，CMake 还定义了`CMAKE_SYSTEM_PROCESSOR`变量。前者包含 CMake**当前正在运行**的 CPU 的名称，后者将包含我们**当前正在构建**的 CPU 的名称。这是一个微妙的区别，在交叉编译时起着非常基本的作用。我们将在第十三章，*替代生成器和交叉编译*中了解更多关于交叉编译的信息。

让 CMake 检测主机处理器架构的替代方法是使用 C 或 C++中定义的符号，并使用 CMake 的`try_run`函数来构建并尝试执行源代码（参见第五章，*配置时间和构建时间操作*，第 8 个配方，*探测执行*），该操作由预处理器符号分支。这会返回可以在 CMake 侧捕获的定义良好的错误（此策略的灵感来自[`github.com/axr/solar-cmake/blob/master/TargetArch.cmake`](https://github.com/axr/solar-cmake/blob/master/TargetArch.cmake)）：

```cpp
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
    #error cmake_arch i386
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
    #error cmake_arch x86_64
#endif
```

此策略也是检测目标处理器架构的首选方法，其中 CMake 似乎没有提供便携式内置解决方案。

还存在另一种替代方案。它将仅使用 CMake，完全摆脱预处理器，代价是每个情况都有一个不同的源文件，然后使用`target_sources` CMake 命令将其设置为可执行目标`arch-dependent`的源文件：

```cpp
add_executable(arch-dependent "")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
  message(STATUS "i386 architecture detected")
  target_sources(arch-dependent
    PRIVATE
      arch-dependent-i386.cpp
    )
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686")
  message(STATUS "i686 architecture detected")
  target_sources(arch-dependent
    PRIVATE
      arch-dependent-i686.cpp
    )
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "x86_64 architecture detected")
  target_sources(arch-dependent
    PRIVATE
      arch-dependent-x86_64.cpp
    )
else()
  message(STATUS "host processor architecture is unknown")
endif()
```

这种方法显然需要对现有项目进行更多工作，因为源文件需要分开。此外，不同源文件之间的代码重复可能确实成为一个问题。

# 发现主机处理器指令集

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-05)获取，并包含一个 C++示例。该配方适用于 CMake 版本 3.10（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在本配方中，我们将讨论如何借助 CMake 发现主机处理器指令集。此功能相对较新地添加到 CMake 中，并需要 CMake 3.10 或更高版本。检测到的主机系统信息可用于设置相应的编译器标志，或根据主机系统实现可选的源代码编译或源代码生成。在本配方中，我们的目标是检测主机系统信息，使用预处理器定义将其传递给 C++源代码，并将信息打印到输出。

# 准备就绪

我们的示例 C++源文件（`processor-info.cpp`）包含以下内容：

```cpp
#include "config.h"

#include <cstdlib>
#include <iostream>

int main() {
  std::cout << "Number of logical cores: "
            << NUMBER_OF_LOGICAL_CORES << std::endl;
  std::cout << "Number of physical cores: "
            << NUMBER_OF_PHYSICAL_CORES << std::endl;

  std::cout << "Total virtual memory in megabytes: "
            << TOTAL_VIRTUAL_MEMORY << std::endl;
  std::cout << "Available virtual memory in megabytes: "
            << AVAILABLE_VIRTUAL_MEMORY << std::endl;
  std::cout << "Total physical memory in megabytes: "
            << TOTAL_PHYSICAL_MEMORY << std::endl;
  std::cout << "Available physical memory in megabytes: "
            << AVAILABLE_PHYSICAL_MEMORY << std::endl;

  std::cout << "Processor is 64Bit: "
            << IS_64BIT << std::endl;
  std::cout << "Processor has floating point unit: "
            << HAS_FPU << std::endl;
  std::cout << "Processor supports MMX instructions: "
            << HAS_MMX << std::endl;
  std::cout << "Processor supports Ext. MMX instructions: "
            << HAS_MMX_PLUS << std::endl;
  std::cout << "Processor supports SSE instructions: "
            << HAS_SSE << std::endl;
  std::cout << "Processor supports SSE2 instructions: "
            << HAS_SSE2 << std::endl;
  std::cout << "Processor supports SSE FP instructions: "
            << HAS_SSE_FP << std::endl;
  std::cout << "Processor supports SSE MMX instructions: "
            << HAS_SSE_MMX << std::endl;
  std::cout << "Processor supports 3DNow instructions: "
            << HAS_AMD_3DNOW << std::endl;
  std::cout << "Processor supports 3DNow+ instructions: "
            << HAS_AMD_3DNOW_PLUS << std::endl;
  std::cout << "IA64 processor emulating x86 : "
            << HAS_IA64 << std::endl;

  std::cout << "OS name: "
            << OS_NAME << std::endl;
  std::cout << "OS sub-type: "
            << OS_RELEASE << std::endl;
  std::cout << "OS build ID: "
            << OS_VERSION << std::endl;
  std::cout << "OS platform: "
            << OS_PLATFORM << std::endl;

  return EXIT_SUCCESS;
}
```

该文件包含`config.h`，我们将从`config.h.in`生成，如下所示：

```cpp
#pragma once

#define NUMBER_OF_LOGICAL_CORES @_NUMBER_OF_LOGICAL_CORES@
#define NUMBER_OF_PHYSICAL_CORES @_NUMBER_OF_PHYSICAL_CORES@
#define TOTAL_VIRTUAL_MEMORY @_TOTAL_VIRTUAL_MEMORY@
#define AVAILABLE_VIRTUAL_MEMORY @_AVAILABLE_VIRTUAL_MEMORY@
#define TOTAL_PHYSICAL_MEMORY @_TOTAL_PHYSICAL_MEMORY@
#define AVAILABLE_PHYSICAL_MEMORY @_AVAILABLE_PHYSICAL_MEMORY@
#define IS_64BIT @_IS_64BIT@
#define HAS_FPU @_HAS_FPU@
#define HAS_MMX @_HAS_MMX@
#define HAS_MMX_PLUS @_HAS_MMX_PLUS@
#define HAS_SSE @_HAS_SSE@
#define HAS_SSE2 @_HAS_SSE2@
#define HAS_SSE_FP @_HAS_SSE_FP@
#define HAS_SSE_MMX @_HAS_SSE_MMX@
#define HAS_AMD_3DNOW @_HAS_AMD_3DNOW@
#define HAS_AMD_3DNOW_PLUS @_HAS_AMD_3DNOW_PLUS@
#define HAS_IA64 @_HAS_IA64@
#define OS_NAME "@_OS_NAME@"
#define OS_RELEASE "@_OS_RELEASE@"
#define OS_VERSION "@_OS_VERSION@"
#define OS_PLATFORM "@_OS_PLATFORM@"
```

# 如何做到这一点

我们将使用 CMake 来填充`config.h`中对我们平台有意义的定义，并将我们的示例源文件编译成可执行文件：

1.  首先，我们定义最小 CMake 版本、项目名称和项目语言：

```cpp
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(recipe-05 CXX)
```

1.  然后，我们定义目标可执行文件、其源文件和包含目录：

```cpp
add_executable(processor-info "")

target_sources(processor-info
  PRIVATE
    processor-info.cpp
  )

target_include_directories(processor-info
  PRIVATE
    ${PROJECT_BINARY_DIR}
  )
```

1.  然后，我们继续查询主机系统信息的一系列键：

```cpp
foreach(key
  IN ITEMS
    NUMBER_OF_LOGICAL_CORES
    NUMBER_OF_PHYSICAL_CORES
    TOTAL_VIRTUAL_MEMORY
    AVAILABLE_VIRTUAL_MEMORY
    TOTAL_PHYSICAL_MEMORY
    AVAILABLE_PHYSICAL_MEMORY
    IS_64BIT
    HAS_FPU
    HAS_MMX
    HAS_MMX_PLUS
    HAS_SSE
    HAS_SSE2
    HAS_SSE_FP
    HAS_SSE_MMX
    HAS_AMD_3DNOW
```

```cpp
    HAS_AMD_3DNOW_PLUS
    HAS_IA64
    OS_NAME
    OS_RELEASE
    OS_VERSION
    OS_PLATFORM
  )
  cmake_host_system_information(RESULT _${key} QUERY ${key})
endforeach()
```

1.  定义了相应的变量后，我们配置`config.h`：

```cpp
configure_file(config.h.in config.h @ONLY)
```

1.  现在我们准备好配置、构建和测试项目了：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./processor-info

Number of logical cores: 4
Number of physical cores: 2
Total virtual memory in megabytes: 15258
Available virtual memory in megabytes: 14678
Total physical memory in megabytes: 7858
Available physical memory in megabytes: 4072
Processor is 64Bit: 1
Processor has floating point unit: 1
Processor supports MMX instructions: 1
Processor supports Ext. MMX instructions: 0
Processor supports SSE instructions: 1
Processor supports SSE2 instructions: 1
Processor supports SSE FP instructions: 0
Processor supports SSE MMX instructions: 0
Processor supports 3DNow instructions: 0
Processor supports 3DNow+ instructions: 0
IA64 processor emulating x86 : 0
OS name: Linux
OS sub-type: 4.16.7-1-ARCH
OS build ID: #1 SMP PREEMPT Wed May 2 21:12:36 UTC 2018
OS platform: x86_64
```

1.  输出当然会根据处理器而变化。

# 它是如何工作的

在`CMakeLists.txt`中的`foreach`循环查询多个键的值，并定义相应的变量。本食谱的核心功能是`cmake_host_system_information`，它查询 CMake 运行所在的主机系统的系统信息。此函数可以一次调用多个键，但在这种情况下，我们为每个键使用一次函数调用。然后，我们使用这些变量来配置`config.h.in`中的占位符，并生成`config.h`。此配置是通过`configure_file`命令完成的。最后，`config.h`被包含在`processor-info.cpp`中，一旦编译，它将打印值到屏幕上。我们将在第五章，*配置时间和构建时间操作*，和第六章，*生成源代码*中重新审视这种方法。

# 还有更多

对于更精细的处理器指令集检测，请考虑使用此模块：[`github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake`](https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake)。我们还想指出，有时构建代码的主机可能与运行代码的主机不同。这在计算集群中很常见，登录节点的架构可能与计算节点的架构不同。解决此问题的一种方法是提交配置和编译作为计算步骤，并将其部署到计算节点。

我们没有使用`cmake_host_system_information`中的所有可用键。为此，请参考[`cmake.org/cmake/help/latest/command/cmake_host_system_information.html`](https://cmake.org/cmake/help/latest/command/cmake_host_system_information.html)。

# 为 Eigen 库启用矢量化

此食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-06`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-06)找到，并包含一个 C++示例。该食谱适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

现代处理器架构的向量能力可以显著提高代码的性能。对于某些类型的操作，这一点尤其明显，而线性代数是其中最突出的。本食谱将展示如何启用向量化以加速使用 Eigen C++库进行线性代数的简单可执行文件。

# 准备就绪

我们将使用 Eigen C++模板库进行线性代数运算，并展示如何设置编译器标志以启用向量化。本食谱的源代码是`linear-algebra.cpp`文件：

```cpp
#include <chrono>
#include <iostream>

#include <Eigen/Dense>

EIGEN_DONT_INLINE
double simple_function(Eigen::VectorXd &va, Eigen::VectorXd &vb) {
  // this simple function computes the dot product of two vectors
  // of course it could be expressed more compactly
  double d = va.dot(vb);
  return d;
}

int main() {
  int len = 1000000;
  int num_repetitions = 100;

  // generate two random vectors
  Eigen::VectorXd va = Eigen::VectorXd::Random(len);
  Eigen::VectorXd vb = Eigen::VectorXd::Random(len);

  double result;
  auto start = std::chrono::system_clock::now();
  for (auto i = 0; i < num_repetitions; i++) {
    result = simple_function(va, vb);
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed_seconds = end - start;

  std::cout << "result: " << result << std::endl;
  std::cout << "elapsed seconds: " << elapsed_seconds.count() << std::endl;
}
```

我们期望向量化能够加速`simple_function`中点积操作的执行。

# 如何操作

根据 Eigen 库的文档，只需设置适当的编译器标志即可启用向量化代码的生成。让我们看看`CMakeLists.txt`：

1.  我们声明一个 C++11 项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  由于我们希望使用 Eigen 库，因此我们需要在系统上找到其头文件：

```cpp
find_package(Eigen3 3.3 REQUIRED CONFIG)
```

1.  我们包含`CheckCXXCompilerFlag.cmake`标准模块文件：

```cpp
include(CheckCXXCompilerFlag)
```

1.  我们检查`-march=native`编译器标志是否有效：

```cpp
check_cxx_compiler_flag("-march=native" _march_native_works)
```

1.  我们还检查了替代的`-xHost`编译器标志：

```cpp
check_cxx_compiler_flag("-xHost" _xhost_works)
```

1.  我们设置一个空变量`_CXX_FLAGS`，以保存我们刚刚检查的两个标志中找到的一个有效标志。如果我们看到`_march_native_works`，我们将`_CXX_FLAGS`设置为`-march=native`。如果我们看到`_xhost_works`，我们将`_CXX_FLAGS`设置为`-xHost`。如果两者都不起作用，我们将保持`_CXX_FLAGS`为空，向量化将被禁用：

```cpp
set(_CXX_FLAGS)
if(_march_native_works)
  message(STATUS "Using processor's vector instructions (-march=native compiler flag set)")
  set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
  message(STATUS "Using processor's vector instructions (-xHost compiler flag set)")
  set(_CXX_FLAGS "-xHost")
else()
  message(STATUS "No suitable compiler flag found for vectorization")
endif()
```

1.  为了进行比较，我们还为未优化的版本定义了一个可执行目标，其中我们不使用前面的优化标志：

```cpp
add_executable(linear-algebra-unoptimized linear-algebra.cpp)

target_link_libraries(linear-algebra-unoptimized
  PRIVATE
    Eigen3::Eigen
  )
```

1.  此外，我们还定义了一个优化版本：

```cpp
add_executable(linear-algebra linear-algebra.cpp)

target_compile_options(linear-algebra
  PRIVATE
    ${_CXX_FLAGS}
  )

target_link_libraries(linear-algebra
  PRIVATE
    Eigen3::Eigen
  )
```

1.  让我们比较这两个可执行文件——首先我们进行配置（在这种情况下，`-march=native_works`）：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Performing Test _march_native_works
-- Performing Test _march_native_works - Success
```

```cpp
-- Performing Test _xhost_works
-- Performing Test _xhost_works - Failed
-- Using processor's vector instructions (-march=native compiler flag set)
...
```

1.  最后，让我们编译并比较时间：

```cpp
$ cmake --build .

$ ./linear-algebra-unoptimized 
result: -261.505
elapsed seconds: 1.97964

$ ./linear-algebra 
result: -261.505
elapsed seconds: 1.05048
```

# 工作原理

大多数现代处理器提供向量指令集。精心编写的代码可以利用这些指令集，并在与非向量化代码相比时实现增强的性能。Eigen 库在编写时就明确考虑了向量化，因为线性代数操作可以从中大大受益。我们所需要做的就是指示编译器为我们检查处理器，并为当前架构生成原生指令集。不同的编译器供应商使用不同的标志来实现这一点：GNU 编译器通过`-march=native`标志实现这一点，而 Intel 编译器使用`-xHost`标志。然后我们使用`CheckCXXCompilerFlag.cmake`模块提供的`check_cxx_compiler_flag`函数：

```cpp
check_cxx_compiler_flag("-march=native" _march_native_works)
```

该函数接受两个参数：第一个是要检查的编译器标志，第二个是用于存储检查结果的变量，即`true`或`false`。如果检查结果为正，我们将工作标志添加到`_CXX_FLAGS`变量中，然后该变量将用于设置我们可执行目标的编译器标志。

# 还有更多

这个配方可以与之前的配方结合使用；可以使用`cmake_host_system_information`查询处理器能力。
