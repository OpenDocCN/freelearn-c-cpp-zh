# 第十章：混合语言项目

在本章中，我们将涵盖以下示例：

+   构建使用 C/C++库的 Fortran 项目

+   构建使用 Fortran 库的 C/C++项目

+   使用 Cython 构建 C++和 Python 项目

+   使用 Boost.Python 构建 C++和 Python 项目

+   使用 pybind11 构建 C++和 Python 项目

+   使用 Python CFFI 混合 C、C++、Fortran 和 Python

# 引言

有许多现有的库在特定任务上表现出色。通常，在我们的代码库中重用这些库是一个非常好的主意，因为我们可以依赖其他专家团队多年的经验。随着计算机架构和编译器的演变，编程语言也在发展。过去，大多数科学软件都是用 Fortran 编写的，而现在，C、C++和解释型语言——尤其是 Python——正占据主导地位。将编译型语言编写的代码与解释型语言的绑定相结合变得越来越普遍，因为它提供了以下好处：

+   终端用户可以自定义和扩展代码本身提供的能力，以完全满足他们的需求。

+   人们可以将 Python 等语言的表达力与编译型语言的性能相结合，这种编译型语言在内存寻址方面更接近“硬件层面”，从而获得两者的最佳效果。

正如我们在之前的各个示例中一直展示的那样，`project`命令可以通过`LANGUAGES`关键字来设置项目中使用的语言。CMake 支持多种编译型编程语言，但并非全部。截至 CMake 3.5 版本，各种汇编语言（如 ASM-ATT、ASM、ASM-MASM 和 ASM-NASM）、C、C++、Fortran、Java、RC（Windows 资源编译器）和 Swift 都是有效选项。CMake 3.8 版本增加了对两种新语言的支持：C#和 CUDA（详见此处发布说明：[`cmake.org/cmake/help/v3.8/release/3.8.html#languages`](https://cmake.org/cmake/help/v3.8/release/3.8.html#languages)）。

在本章中，我们将展示如何将用不同编译型（C、C++和 Fortran）和解释型（Python）语言编写的代码集成到一个可移植和跨平台的解决方案中。我们将展示如何利用 CMake 和不同编程语言固有的工具来实现集成。

# 构建使用 C/C++库的 Fortran 项目

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-01)找到，并包含两个示例：一个是混合 Fortran 和 C，另一个是混合 Fortran 和 C++。该示例适用于 CMake 3.5 版本（及以上）。两个版本的示例都已在 GNU/Linux 和 macOS 上进行了测试。

Fortran 作为高性能计算语言有着悠久的历史。许多数值线性代数库仍然主要用 Fortran 编写，许多需要与过去几十年积累的遗留代码保持兼容的大型数字处理软件包也是如此。虽然 Fortran 在处理数值数组时提供了非常自然的语法，但在与操作系统交互时却显得不足，主要是因为直到 Fortran 2003 标准发布时，才强制要求与 C 语言（计算机编程的*事实上的通用语言*）的互操作层。本食谱将展示如何将 Fortran 代码与 C 系统库和自定义 C 代码接口。

# 准备工作

如第七章，*项目结构化*所示，我们将把项目结构化为树状。每个子目录都有一个`CMakeLists.txt`文件，其中包含与该目录相关的指令。这使我们能够尽可能地将信息限制在叶目录中，如下例所示：

```cpp
.
├── CMakeLists.txt
└── src
    ├── bt-randomgen-example.f90
    ├── CMakeLists.txt
    ├── interfaces
    │   ├── CMakeLists.txt
    │   ├── interface_backtrace.f90
    │   ├── interface_randomgen.f90
    │   └── randomgen.c
    └── utils
        ├── CMakeLists.txt
        └── util_strings.f90
```

在我们的例子中，我们有一个包含源代码的`src`子目录，包括我们的可执行文件`bt-randomgen-example.f90`。另外两个子目录，`interfaces`和`utils`，包含将被编译成库的更多源代码。

在`interfaces`子目录中的源代码展示了如何封装 backtrace C 系统库。例如，`interface_backtrace.f90`包含：

```cpp
module interface_backtrace

  implicit none

  interface
    function backtrace(buffer, size) result(bt) bind(C, name="backtrace")
      use, intrinsic :: iso_c_binding, only: c_int, c_ptr
      type(c_ptr) :: buffer
      integer(c_int), value :: size
      integer(c_int) :: bt
    end function

    subroutine backtrace_symbols_fd(buffer, size, fd) bind(C, name="backtrace_symbols_fd")
      use, intrinsic :: iso_c_binding, only: c_int, c_ptr
      type(c_ptr) :: buffer
      integer(c_int), value :: size, fd
    end subroutine
  end interface

end module
```

上述示例展示了以下用法：

+   内置的`iso_c_binding`模块，确保了 Fortran 和 C 类型及函数的互操作性。

+   `interface`声明，它将函数绑定到单独库中的符号。

+   `bind(C)`属性，它固定了声明函数的名称混淆。

这个子目录包含另外两个源文件：

+   `randomgen.c`，这是一个 C 源文件，它使用 C 标准的`rand`函数公开一个函数，用于在区间内生成随机整数。

+   `interface_randomgen.f90`，它封装了用于 Fortran 可执行文件中的 C 函数。

# 如何操作

我们有四个`CMakeLists.txt`实例需要查看：一个根目录和三个叶目录。让我们从根目录的`CMakeLists.txt`开始：

1.  我们声明了一个混合语言的 Fortran 和 C 项目：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES Fortran C)
```

1.  我们指示 CMake 在构建目录的`lib`子目录下保存静态和共享库。可执行文件将保存在`bin`下，而 Fortran 编译模块文件将保存在`modules`下：

```cpp
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_Fortran_MODULE_DIRECTORY
  ${CMAKE_CURRENT_BINARY_DIR}/modules)
```

1.  接下来，我们转到第一个叶目录，通过添加`src`子目录来编辑`CMakeLists.txt`：

```cpp
add_subdirectory(src)
```

1.  `src/CMakeLists.txt`文件添加了另外两个子目录：

```cpp
add_subdirectory(interfaces)
add_subdirectory(utils)
```

在`interfaces`子目录中，我们执行以下操作：

1.  我们包含了`FortranCInterface.cmake`模块，并验证 C 和 Fortran 编译器可以正确地相互通信：

```cpp
include(FortranCInterface)
FortranCInterface_VERIFY()
```

1.  接下来，我们找到 backtrace 系统库，因为我们想在 Fortran 代码中使用它：

```cpp
find_package(Backtrace REQUIRED)
```

1.  然后，我们使用回溯包装器、随机数生成器及其 Fortran 包装器的源文件创建一个共享库目标：

```cpp
add_library(bt-randomgen-wrap SHARED "")

target_sources(bt-randomgen-wrap
  PRIVATE
    interface_backtrace.f90
    interface_randomgen.f90
    randomgen.c
  )
```

1.  我们还为新生成的库目标设置了链接库。我们使用`PUBLIC`属性，以便链接我们的库的其他目标能够正确看到依赖关系：

```cpp
target_link_libraries(bt-randomgen-wrap
  PUBLIC
    ${Backtrace_LIBRARIES}
  )
```

在`utils`子目录中，我们还有一个`CMakeLists.txt`。这是一个一行代码：我们创建一个新的库目标，该子目录中的源文件将被编译到这个目标中。这个目标没有依赖关系：

```cpp
add_library(utils SHARED util_strings.f90)
```

让我们回到`src/CMakeLists.txt`：

1.  我们添加一个可执行目标，使用`bt-randomgen-example.f90`作为源文件：

```cpp
add_executable(bt-randomgen-example bt-randomgen-example.f90)
```

1.  最后，我们将`CMakeLists.txt`叶中生成的库目标链接到我们的可执行目标：

```cpp
target_link_libraries(bt-randomgen-example
  PRIVATE
    bt-randomgen-wrap
    utils
  )
```

# **它是如何工作的**

在确定了要链接的正确库之后，我们需要确保我们的程序能够正确调用它们定义的函数。每个编译器在生成机器代码时都会执行名称重整，不幸的是，这项操作的约定并不是通用的，而是依赖于编译器。我们已经在《第三章》（c1fec057-4e5f-4a9b-b404-30dc74f5d7b7.xhtml），*检测外部库和程序*，第 4 个配方，*检测 BLAS 和 LAPACK 数学库*中遇到的`FortranCInterface`，检查所选 C 编译器与 Fortran 编译器的兼容性。对于我们当前的目的，名称重整并不是真正的问题。Fortran 2003 标准为函数和子程序定义了一个`bind`属性，它接受一个可选的`name`参数。如果提供了这个参数，编译器将使用程序员固定的名称为这些子程序和函数生成符号。例如，回溯函数可以从 C 暴露给 Fortran，保留名称，如下所示：

```cpp
function backtrace(buffer, size) result(bt) bind(C, name="backtrace")
```

# **还有更多**

在`interfaces/CMakeLists.txt`中的 CMake 代码也表明，可以从不同语言的源文件创建库。显然，CMake 能够执行以下操作：

+   确定使用哪个编译器从列出的源文件获取目标文件。

+   选择适当的链接器来从这些目标文件构建库（或可执行文件）。

CMake 如何确定使用哪个编译器？通过在`project`命令中指定`LANGUAGES`选项，CMake 将检查您的系统上是否存在适用于给定语言的工作编译器。当添加目标并列出源文件时，CMake 将根据文件扩展名适当地确定编译器。因此，以`.c`结尾的文件将使用已确定的 C 编译器编译为对象文件，而以`.f90`（或需要预处理的`.F90`）结尾的文件将使用工作的 Fortran 编译器进行编译。同样，对于 C++，`.cpp`或`.cxx`扩展名将触发使用 C++编译器。我们仅列出了 C、C++和 Fortran 语言的一些可能的有效文件扩展名，但 CMake 可以识别更多。如果项目中的文件扩展名由于任何原因不在识别的扩展名之列，该怎么办？可以使用`LANGUAGE`源文件属性来告诉 CMake 在特定源文件上使用哪个编译器，如下所示：

```cpp
set_source_files_properties(my_source_file.axx
  PROPERTIES
    LANGUAGE CXX
  )
```

最后，链接器呢？CMake 如何确定目标的链接器语言？对于**不混合**编程语言的目标，选择很简单：通过用于生成对象文件的编译器命令调用链接器。如果目标**确实混合**了编程语言，如我们的示例，链接器语言的选择基于在语言混合中偏好值最高的那个。在我们的示例中混合了 Fortran 和 C，Fortran 语言的偏好高于 C 语言，因此被用作链接器语言。当混合 Fortran 和 C++时，后者具有更高的偏好，因此被用作链接器语言。与编译器语言一样，我们可以通过在目标上设置相应的`LINKER_LANGUAGE`属性来强制 CMake 为我们的目标使用特定的链接器语言：

```cpp
set_target_properties(my_target
   PROPERTIES
     LINKER_LANGUAGE Fortran
   )
```

# 构建使用 Fortran 库的 C/C++项目

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-02)找到，并提供了一个混合 C++、C 和 Fortran 的示例。该配方适用于 CMake 版本 3.5（及以上），并在 GNU/Linux 和 macOS 上进行了测试。

第三章的配方 4，*检测 BLAS 和 LAPACK 数学库*，在第三章，*检测外部库和程序*，展示了如何检测用 Fortran 编写的 BLAS 和 LAPACK 线性代数库，以及如何在 C++代码中使用它们。在这里，我们将重新审视这个配方，但这次从不同的角度出发：更少关注检测外部库，而是更深入地讨论混合 C++和 Fortran 以及名称修饰的方面。

# 准备工作

在本食谱中，我们将重用来自第三章，*检测外部库和程序*，食谱 4，*检测 BLAS 和 LAPACK 数学库*的源代码。尽管我们不会修改实际的实现源文件或头文件，但我们将根据第七章，*项目结构*中讨论的建议修改项目树结构，并得出以下源代码结构：

```cpp
.
├── CMakeLists.txt
├── README.md
└── src
    ├── CMakeLists.txt
    ├── linear-algebra.cpp
    └── math
        ├── CMakeLists.txt
        ├── CxxBLAS.cpp
        ├── CxxBLAS.hpp
        ├── CxxLAPACK.cpp
        └── CxxLAPACK.hpp
```

这里我们收集了所有 BLAS 和 LAPACK 的包装器，它们在`src/math`下提供了`math`库。主程序是`linear-algebra.cpp`。所有源文件都组织在`src`子目录下。为了限定范围，我们将 CMake 代码拆分到三个`CMakeLists.txt`文件中，现在我们将讨论这些文件。

# 如何操作

这个项目混合了 C++（主程序的语言）、Fortran（因为这是库所写的语言）和 C（需要用来包装 Fortran 子例程）。在根`CMakeLists.txt`文件中，我们需要执行以下操作：

1.  将项目声明为混合语言并设置 C++标准：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-02 LANGUAGES CXX C Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  我们使用`GNUInstallDirs`模块来指导 CMake 将静态和共享库以及可执行文件保存到标准目录中。我们还指示 CMake 将 Fortran 编译的模块文件放置在`modules`下：

```cpp
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/modules)
```

1.  然后我们转到下一个叶子子目录：

```cpp
add_subdirectory(src)
```

在`src/CMakeLists.txt`文件中，我们添加了另一个子目录`math`，其中包含了线性代数包装器。在`src/math/CMakeLists.txt`中，我们需要执行以下操作：

1.  我们调用`find_package`来获取 BLAS 和 LAPACK 库的位置：

```cpp
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
```

1.  我们包含`FortranCInterface.cmake`模块，并验证 Fortran、C 和 C++编译器是否兼容：

```cpp
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
```

1.  我们还需要生成预处理器宏来处理 BLAS 和 LAPACK 子例程的名称修饰。再次，`FortranCInterface`通过在当前构建目录中生成一个名为`fc_mangle.h`的头文件来提供帮助：

```cpp
FortranCInterface_HEADER(
  fc_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOLS DSCAL DGESV
  )
```

1.  接下来，我们为 BLAS 和 LAPACK 包装器添加一个库，并指定头文件和库所在的目录。注意`PUBLIC`属性，它将允许依赖于`math`的其他目标正确获取其依赖项：

```cpp
add_library(math "")

target_sources(math
  PRIVATE
    CxxBLAS.cpp
    CxxLAPACK.cpp
  )

target_include_directories(math
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

target_link_libraries(math
  PUBLIC
    ${LAPACK_LIBRARIES}
  )
```

回到`src/CMakeLists.txt`，我们最终添加了一个可执行目标，并将其链接到我们的 BLAS/LAPACK 包装器的`math`库：

```cpp
add_executable(linear-algebra "")

target_sources(linear-algebra
  PRIVATE
    linear-algebra.cpp
  )

target_link_libraries(linear-algebra
  PRIVATE
    math
  )
```

# 它是如何工作的

使用`find_package`，我们已经确定了要链接的正确库。与之前的食谱一样，我们需要确保我们的程序能够正确调用它们定义的函数。在第三章，*检测外部库和程序*，第 4 个食谱，*检测 BLAS 和 LAPACK 数学库*，我们面临编译器依赖的符号修饰问题。我们使用`FortranCInterface` CMake 模块来检查所选 C 和 C++编译器与 Fortran 编译器的兼容性。我们还使用`FortranCInterface_HEADER`函数来生成包含宏的头文件，以处理 Fortran 子程序的符号修饰。这是通过以下代码实现的：

```cpp
FortranCInterface_HEADER(
  fc_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOLS DSCAL DGESV
  )
```

此命令将生成包含符号修饰宏的`fc_mangle.h`头文件，如 Fortran 编译器所推断，并将其保存到当前二进制目录`CMAKE_CURRENT_BINARY_DIR`。我们小心地将`CMAKE_CURRENT_BINARY_DIR`设置为`math`目标的包含路径。考虑以下生成的`fc_mangle.h`：

```cpp
#ifndef FC_HEADER_INCLUDED
#define FC_HEADER_INCLUDED

/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name,NAME) name##_

/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL_(name,NAME) name##_

/* Mangling for Fortran module symbols without underscores. */
#define FC_MODULE(mod_name,name, mod_NAME,NAME) __##mod_name##_MOD_##name

/* Mangling for Fortran module symbols with underscores. */
#define FC_MODULE_(mod_name,name, mod_NAME,NAME) __##mod_name##_MOD_##name

/* Mangle some symbols automatically. */
#define DSCAL FC_GLOBAL(dscal, DSCAL)
#define DGESV FC_GLOBAL(dgesv, DGESV)

#endif
```

本示例中的编译器使用下划线进行符号修饰。由于 Fortran 不区分大小写，子程序可能以小写或大写形式出现，因此需要将两种情况都传递给宏。请注意，CMake 还将为隐藏在 Fortran 模块后面的符号生成修饰宏。

如今，许多 BLAS 和 LAPACK 的实现都附带了一个围绕 Fortran 子程序的薄 C 层包装器。这些包装器多年来已经标准化，并分别称为 CBLAS 和 LAPACKE。

由于我们已将源文件仔细组织成一个库目标和一个可执行目标，我们应该对目标的`PUBLIC`、`INTERFACE`和`PRIVATE`可见性属性进行注释。这些对于清晰的 CMake 项目结构至关重要。与源文件一样，包含目录、编译定义和选项，当与`target_link_libraries`一起使用时，这些属性的含义保持不变：

+   使用`PRIVATE`属性，库将仅被链接到当前目标，而不会被链接到以它作为依赖的其他目标。

+   使用`INTERFACE`属性，库将仅被链接到以当前目标作为依赖的目标。

+   使用`PUBLIC`属性，库将被链接到当前目标以及任何以它作为依赖的其他目标。

# 使用 Cython 构建 C++和 Python 项目

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-03)找到，并包含一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

Cython 是一个优化的静态编译器，允许为 Python 编写 C 扩展。Cython 是一个非常强大的工具，使用基于 Pyrex 的扩展 Cython 编程语言。Cython 的一个典型用例是加速 Python 代码，但它也可以用于通过 Cython 层将 C/C++与 Python 接口。在本食谱中，我们将专注于后一种用例，并演示如何使用 CMake 帮助下的 Cython 将 C/C++和 Python 接口。

# 准备就绪

作为一个例子，我们将使用以下 C++代码（`account.cpp`）：

```cpp
#include "account.hpp"

Account::Account() : balance(0.0) {}

Account::~Account() {}

void Account::deposit(const double amount) { balance += amount; }

void Account::withdraw(const double amount) { balance -= amount; }

double Account::get_balance() const { return balance; }
```

这段代码提供了以下接口（`account.hpp`）：

```cpp
#pragma once

class Account {
public:
  Account();
  ~Account();

  void deposit(const double amount);
  void withdraw(const double amount);
  double get_balance() const;

private:
  double balance;
};
```

使用这段示例代码，我们可以创建起始余额为零的银行账户。我们可以向账户存款和取款，也可以使用`get_balance()`查询账户余额。余额本身是`Account`类的私有成员。

我们的目标是能够直接从 Python 与这个 C++类交互——换句话说，在 Python 方面，我们希望能够这样做：

```cpp
account = Account()

account.deposit(100.0)
account.withdraw(50.0)

balance = account.get_balance()
```

为了实现这一点，我们需要一个 Cython 接口文件（我们将称这个文件为`account.pyx`）：

```cpp
# describe the c++ interface
cdef extern from "account.hpp":
    cdef cppclass Account:
        Account() except +
        void deposit(double)
        void withdraw(double)
        double get_balance()

# describe the python interface
cdef class pyAccount:
    cdef Account *thisptr
    def __cinit__(self):
        self.thisptr = new Account()
    def __dealloc__(self):
        del self.thisptr
    def deposit(self, amount):
        self.thisptr.deposit(amount)
    def withdraw(self, amount):
        self.thisptr.withdraw(amount)
    def get_balance(self):
        return self.thisptr.get_balance()
```

# 如何操作

让我们看看如何生成 Python 接口：

1.  我们的`CMakeLists.txt`开始定义 CMake 依赖项、项目名称和语言：

```cpp
# define minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and supported language
project(recipe-03 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  在 Windows 上，最好不要让构建类型未定义，这样我们就可以使此项目的构建类型与 Python 环境的构建类型相匹配。这里我们默认使用`Release`构建类型：

```cpp
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
```

1.  在本食谱中，我们还将需要 Python 解释器：

```cpp
find_package(PythonInterp REQUIRED)
```

1.  以下 CMake 代码将允许我们构建 Python 模块：

```cpp
# directory cointaining UseCython.cmake and FindCython.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-cython)

# this defines cython_add_module
include(UseCython)

# tells UseCython to compile this file as a c++ file
set_source_files_properties(account.pyx PROPERTIES CYTHON_IS_CXX TRUE)

# create python module
cython_add_module(account account.pyx account.cpp)

# location of account.hpp
target_include_directories(account
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

1.  现在我们定义一个测试：

```cpp
# turn on testing
enable_testing()

# define test
add_test(
  NAME
    python_test
  COMMAND
    ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=$<TARGET_FILE_DIR:account>
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  )
```

1.  `python_test`执行`test.py`，在其中我们进行了几次存款和取款，并验证了余额：

```cpp
import os
import sys
sys.path.append(os.getenv('ACCOUNT_MODULE_PATH'))

from account import pyAccount as Account

account1 = Account()

account1.deposit(100.0)
account1.deposit(100.0)

account2 = Account()

account2.deposit(200.0)
account2.deposit(200.0)

account1.withdraw(50.0)

assert account1.get_balance() == 150.0
assert account2.get_balance() == 400.0
```

1.  有了这些，我们就可以配置、构建和测试代码了：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

 Start 1: python_test
1/1 Test #1: python_test ...................... Passed 0.03 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.03 sec
```

# 工作原理

在本食谱中，我们通过一个相对紧凑的`CMakeLists.txt`文件实现了 Python 与 C++的接口，但我们通过使用`FindCython.cmake`和`UseCython.cmake`模块实现了这一点，这些模块被放置在`cmake-cython`下。这些模块通过以下代码包含：

```cpp
# directory contains UseCython.cmake and FindCython.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-cython)

# this defines cython_add_module
include(UseCython)
```

`FindCython.cmake`包含在`UseCython.cmake`中，并定位和定义`${CYTHON_EXECUTABLE}`。后一个模块定义了`cython_add_module`和`cython_add_standalone_executable`函数，这些函数可用于创建 Python 模块和独立可执行文件。这两个模块都已从[`github.com/thewtex/cython-cmake-example/tree/master/cmake`](https://github.com/thewtex/cython-cmake-example/tree/master/cmake)下载。

在本食谱中，我们使用`cython_add_module`来创建一个 Python 模块库。请注意，我们将非标准的`CYTHON_IS_CXX`源文件属性设置为`TRUE`，这样`cython_add_module`函数就会知道将`pyx`文件编译为 C++文件：

```cpp
# tells UseCython to compile this file as a c++ file
set_source_files_properties(account.pyx PROPERTIES CYTHON_IS_CXX TRUE)

# create python module
cython_add_module(account account.pyx account.cpp)
```

Python 模块在`${CMAKE_CURRENT_BINARY_DIR}`内部创建，为了让 Python `test.py`脚本能够找到它，我们通过自定义环境变量传递相关路径，该变量在`test.py`内部用于设置`PATH`变量。注意`COMMAND`是如何设置为调用 CMake 可执行文件本身以在执行 Python 脚本之前正确设置本地环境的。这为我们提供了平台独立性，并避免了用无关变量污染环境：

```cpp
add_test(
  NAME
    python_test
  COMMAND
    ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=$<TARGET_FILE_DIR:account>
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  )
```

我们还应该查看`account.pyx`文件，它是 Python 和 C++之间的接口文件，描述了 C++接口：

```cpp
# describe the c++ interface
cdef extern from "account.hpp":
    cdef cppclass Account:
        Account() except +
        void deposit(double)
        void withdraw(double)
        double get_balance()
```

在`Account`类构造函数中可以看到`except +`。这个指令允许 Cython 处理由 C++代码引发的异常。

`account.pyx`接口文件还描述了 Python 接口：

```cpp
# describe the python interface
cdef class pyAccount:
    cdef Account *thisptr
    def __cinit__(self):
        self.thisptr = new Account()
    def __dealloc__(self):
        del self.thisptr
    def deposit(self, amount):
        self.thisptr.deposit(amount)
    def withdraw(self, amount):
        self.thisptr.withdraw(amount)
    def get_balance(self):
        return self.thisptr.get_balance()
```

我们可以看到`cinit`构造函数、`__dealloc__`析构函数以及`deposit`和`withdraw`方法是如何与相应的 C++实现对应部分匹配的。

总结一下，我们找到了一种通过引入对 Cython 模块的依赖来结合 Python 和 C++的机制。这个模块可以通过`pip`安装到虚拟环境或 Pipenv 中，或者使用 Anaconda 安装。

# 还有更多内容

C 也可以类似地耦合。如果我们希望利用构造函数和析构函数，我们可以围绕 C 接口编写一个薄的 C++层。

Typed Memoryviews 提供了有趣的功能，可以直接在 Python 中映射和访问由 C/C++分配的内存缓冲区，而不会产生任何开销：[`cython.readthedocs.io/en/latest/src/userguide/memoryviews.html`](http://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)。它们使得可以直接将 NumPy 数组映射到 C++数组。

# 使用 Boost.Python 构建 C++和 Python 项目

本节的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-04)找到，并包含一个 C++示例。本节适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

Boost 库提供了另一种流行的选择，用于将 C++代码与 Python 接口。本节将展示如何使用 CMake 为依赖于 Boost.Python 的 C++项目构建，以便将它们的功能作为 Python 模块暴露出来。我们将重用前一节的示例，并尝试与 Cython 示例中的相同 C++实现(`account.cpp`)进行交互。

# 准备工作

虽然我们保持`account.cpp`不变，但我们修改了前一节的接口文件(`account.hpp`)：

```cpp
#pragma once

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>

class Account {
public:
  Account();
  ~Account();

  void deposit(const double amount);
  void withdraw(const double amount);
  double get_balance() const;

private:
  double balance;
};

namespace py = boost::python;

BOOST_PYTHON_MODULE(account) {
  py::class_<Account>("Account")
      .def("deposit", &Account::deposit)
      .def("withdraw", &Account::withdraw)
      .def("get_balance", &Account::get_balance);
}
```

# 如何操作

以下是使用 Boost.Python 与您的 C++项目所需的步骤：

1.  与前一节一样，我们首先定义最小版本、项目名称、支持的语言和默认构建类型：

```cpp
# define minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and supported language
project(recipe-04 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# we default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
```

1.  在本配方中，我们依赖于 Python 和 Boost 库以及 Python 解释器进行测试。Boost.Python 组件的名称取决于 Boost 版本和 Python 版本，因此我们探测几个可能的组件名称：

```cpp
# for testing we will need the python interpreter
find_package(PythonInterp REQUIRED)

# we require python development headers
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
```

```cpp
# now search for the boost component
# depending on the boost version it is called either python,
# python2, python27, python3, python36, python37, ...

list(
  APPEND _components
    python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
    python${PYTHON_VERSION_MAJOR}
    python
  )

set(_boost_component_found "")

foreach(_component IN ITEMS ${_components})
  find_package(Boost COMPONENTS ${_component})
  if(Boost_FOUND)
    set(_boost_component_found ${_component})
    break()
  endif()
endforeach()

if(_boost_component_found STREQUAL "")
  message(FATAL_ERROR "No matching Boost.Python component found")
endif()
```

1.  使用以下命令，我们定义了 Python 模块及其依赖项：

```cpp
# create python module
add_library(account
  MODULE
    account.cpp
  )

target_link_libraries(account
  PUBLIC
    Boost::${_boost_component_found}
    ${PYTHON_LIBRARIES}
  )

target_include_directories(account
  PRIVATE
    ${PYTHON_INCLUDE_DIRS}
  )
```

```cpp
# prevent cmake from creating a "lib" prefix
set_target_properties(account
  PROPERTIES
    PREFIX ""
  )

if(WIN32)
  # python will not import dll but expects pyd
  set_target_properties(account
    PROPERTIES
      SUFFIX ".pyd"
    )
endif()
```

1.  最后，我们为这个实现定义了一个测试：

```cpp
# turn on testing
enable_testing()

# define test
add_test(
  NAME
    python_test
  COMMAND
    ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=$<TARGET_FILE_DIR:account>
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  )
```

1.  现在可以配置、编译和测试代码：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

    Start 1: python_test
1/1 Test #1: python_test ......................   Passed    0.10 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.11 sec
```

# 它是如何工作的

与依赖 Cython 模块不同，本配方现在依赖于在系统上定位 Boost 库，以及 Python 开发头文件和库。

使用以下命令搜索 Python 开发头文件和库：

```cpp
find_package(PythonInterp REQUIRED)

find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
```

请注意，我们首先搜索解释器，然后搜索开发头文件和库。此外，对`PythonLibs`的搜索要求开发头文件和库的相同主要和次要版本与解释器发现的版本相同。这是为了确保在整个项目中使用一致的解释器和库版本。然而，这种命令组合并不能保证会找到完全匹配的两个版本。

在定位 Boost.Python 组件时，我们遇到了一个难题，即我们尝试定位的组件名称取决于 Boost 版本和我们的 Python 环境。根据 Boost 版本，组件可以称为`python`、`python2`、`python3`、`python27`、`python36`、`python37`等。我们通过从特定到更通用的名称进行搜索，并且只有在找不到匹配项时才失败来解决这个问题：

```cpp
list(
  APPEND _components
    python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
    python${PYTHON_VERSION_MAJOR}
    python
  )

set(_boost_component_found "")

foreach(_component IN ITEMS ${_components})
  find_package(Boost COMPONENTS ${_component})
  if(Boost_FOUND)
    set(_boost_component_found ${_component})
    break()
  endif()
endforeach()
if(_boost_component_found STREQUAL "")
  message(FATAL_ERROR "No matching Boost.Python component found")
endif()
```

可以通过设置额外的 CMake 变量来调整 Boost 库的发现和使用。例如，CMake 提供以下选项：

+   `Boost_USE_STATIC_LIBS`可以设置为`ON`以强制使用 Boost 库的静态版本。

+   `Boost_USE_MULTITHREADED`可以设置为`ON`以确保选择并使用多线程版本。

+   `Boost_USE_STATIC_RUNTIME`可以设置为`ON`，以便我们的目标将使用链接 C++运行时静态的 Boost 变体。

本配方引入的另一个新方面是在`add_library`命令中使用`MODULE`选项。我们从第 3 个配方，*构建和链接共享和静态库*，在第一章，*从简单可执行文件到库*中已经知道，CMake 接受以下选项作为`add_library`的第二个有效参数：

+   `STATIC`，用于创建静态库；即，用于链接其他目标（如可执行文件）的对象文件的档案

+   `SHARED`，用于创建共享库；即，可以在运行时动态链接和加载的库

+   `OBJECT`，用于创建对象库；即，不将对象文件归档到静态库中，也不将它们链接成共享对象

这里引入的`MODULE`选项将生成一个*插件库*；也就是说，一个动态共享对象（DSO），它不会被动态链接到任何可执行文件中，但仍然可以在运行时加载。由于我们正在用自己编写的 C++功能扩展 Python，Python 解释器将需要在运行时能够加载我们的库。这可以通过使用`add_library`的`MODULE`选项并阻止在我们的库目标名称中添加任何前缀（例如，Unix 系统上的`lib`）来实现。后者操作是通过设置适当的 target 属性来完成的，如下所示：

```cpp
set_target_properties(account
  PROPERTIES
    PREFIX ""
  )
```

所有展示 Python 和 C++接口的示例都有一个共同点，那就是我们需要向 Python 代码描述如何与 C++层连接，并列出应该对 Python 可见的符号。我们还可以（重新）命名这些符号。在前面的示例中，我们在一个单独的`account.pyx`文件中完成了这一点。当使用`Boost.Python`时，我们直接在 C++代码中描述接口，最好靠近我们希望接口的类或函数的定义：

```cpp
BOOST_PYTHON_MODULE(account) {
  py::class_<Account>("Account")
      .def("deposit", &Account::deposit)
      .def("withdraw", &Account::withdraw)
      .def("get_balance", &Account::get_balance);
}
```

`BOOST_PYTHON_MODULE`模板包含在`<boost/python.hpp>`中，负责创建 Python 接口。该模块将暴露一个`Account` Python 类，该类映射到 C++类。在这种情况下，我们不必显式声明构造函数和析构函数——这些会为我们自动创建，并在 Python 对象创建时自动调用：

```cpp
myaccount = Account()
```

当对象超出作用域并被 Python 垃圾回收机制收集时，析构函数会被调用。同时，注意`BOOST_PYTHON_MODULE`是如何暴露`deposit`、`withdraw`和`get_balance`这些函数，并将它们映射到相应的 C++类方法上的。

这样，编译后的模块可以在`PYTHONPATH`中找到。在本示例中，我们实现了 Python 和 C++层之间相对干净的分离。Python 代码在功能上不受限制，不需要类型注释或重命名，并且保持了*pythonic*：

```cpp
from account import Account

account1 = Account()

account1.deposit(100.0)
account1.deposit(100.0)

account2 = Account()

account2.deposit(200.0)
account2.deposit(200.0)

```

```cpp
account1.withdraw(50.0)

assert account1.get_balance() == 150.0
assert account2.get_balance() == 400.0
```

# 还有更多内容

在本示例中，我们依赖于系统上已安装的 Boost，因此 CMake 代码尝试检测相应的库。或者，我们可以将 Boost 源代码与我们的项目一起打包，并将此依赖项作为项目的一部分进行构建。Boost 是一种便携式的方式，用于将 Python 与 C++接口。然而，考虑到编译器支持和 C++标准的可移植性，Boost.Python 并不是一个轻量级的依赖。在下面的示例中，我们将讨论 Boost.Python 的一个轻量级替代方案。

# 使用 pybind11 构建 C++和 Python 项目

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-05)找到，并包含一个 C++示例。该示例适用于 CMake 版本 3.11（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在前一个示例中，我们使用了 Boost.Python 来实现 Python 与 C(++)的接口。在这个示例中，我们将尝试使用 pybind11 作为轻量级替代方案，该方案利用了 C++11 特性，因此需要支持 C++11 的编译器。与前一个示例相比，我们将展示如何在配置时获取 pybind11 依赖项，并使用我们在第四章，*创建和运行测试*，示例 3，*定义单元测试并与 Google Test 链接*中遇到的 FetchContent 方法构建我们的项目，包括 Python 接口，并在第八章，*超级构建模式*，示例 4，*使用超级构建管理依赖项：III. Google Test 框架*中进行了讨论。在第十一章，*打包项目*，示例 2，*通过 PyPI 分发使用 CMake/pybind11 构建的 C++/Python 项目*中，我们将重新访问此示例，并展示如何打包它并通过 pip 安装。

# 准备就绪

我们将保持`account.cpp`相对于前两个示例不变，只修改`account.hpp`：

```cpp
#pragma once

#include <pybind11/pybind11.h>

class Account {
public:
  Account();
  ~Account();

  void deposit(const double amount);
  void withdraw(const double amount);
  double get_balance() const;

private:
  double balance;
};

namespace py = pybind11;

PYBIND11_MODULE(account, m) {
  py::class_<Account>(m, "Account")
      .def(py::init())
      .def("deposit", &Account::deposit)
      .def("withdraw", &Account::withdraw)
      .def("get_balance", &Account::get_balance);
}
```

我们将遵循 pybind11 文档中的“使用 CMake 构建”指南（[`pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake`](https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake)），并介绍使用`add_subdirectory`添加 pybind11 的 CMake 代码。然而，我们不会将 pybind11 源代码明确放入我们的项目目录中，而是演示如何在配置时使用`FetchContent`（[`cmake.org/cmake/help/v3.11/module/FetchContent.html`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html)）获取 pybind11 源代码。

为了在下一个示例中更好地重用代码，我们还将所有源代码放入子目录中，并使用以下项目布局：

```cpp
.
├── account
│   ├── account.cpp
│   ├── account.hpp
│   ├── CMakeLists.txt
│   └── test.py
└── CMakeLists.txt
```

# 如何操作

让我们详细分析这个项目中各个`CMakeLists.txt`文件的内容：

1.  根目录的`CMakeLists.txt`文件包含熟悉的头部信息：

```cpp
# define minimum cmake version
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# project name and supported language
project(recipe-05 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  在此文件中，我们还查询将用于测试的 Python 解释器：

```cpp
find_package(PythonInterp REQUIRED)
```

1.  然后，我们包含账户子目录：

```cpp
add_subdirectory(account)
```

1.  之后，我们定义单元测试：

```cpp
# turn on testing
enable_testing()

# define test
add_test(
  NAME
    python_test
  COMMAND
    ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=$<TARGET_FILE_DIR:account>
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/account/test.py
  )
```

1.  在`account/CMakeLists.txt`文件中，我们在配置时获取 pybind11 源代码：

```cpp
include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.2
)

FetchContent_GetProperties(pybind11_sources)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)

  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()
```

1.  最后，我们定义 Python 模块。再次使用`add_library`的`MODULE`选项。我们还为我们的库目标设置前缀和后缀属性为`PYTHON_MODULE_PREFIX`和`PYTHON_MODULE_EXTENSION`，这些属性由 pybind11 适当地推断出来：

```cpp
add_library(account
  MODULE
    account.cpp
  )

target_link_libraries(account
  PUBLIC
    pybind11::module
  )

set_target_properties(account
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
```

1.  让我们测试一下：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

 Start 1: python_test
1/1 Test #1: python_test ...................... Passed 0.04 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.04 sec
```

# 它是如何工作的

pybind11 的功能和使用与 Boost.Python 非常相似，不同的是 pybind11 是一个更轻量级的依赖项——尽管我们需要编译器的 C++11 支持。在`account.hpp`中的接口定义与前一个示例中的定义相当相似：

```cpp
#include <pybind11/pybind11.h>

// ...

namespace py = pybind11;

PYBIND11_MODULE(account, m) {
  py::class_<Account>(m, "Account")
      .def(py::init())
      .def("deposit", &Account::deposit)
      .def("withdraw", &Account::withdraw)
      .def("get_balance", &Account::get_balance);
}
```

再次，我们可以清楚地看到 Python 方法是如何映射到 C++函数的。解释`PYBIND11_MODULE`的库在导入的目标`pybind11::module`中定义，我们使用以下方式包含它：

```cpp
add_subdirectory(
  ${pybind11_sources_SOURCE_DIR}
  ${pybind11_sources_BINARY_DIR}
  )
```

与前一个配方相比，有两个不同之处：

+   我们不要求系统上安装了 pybind11，因此不会尝试定位它。

+   在项目开始构建时，包含 pybind11 `CMakeLists.txt`的`${pybind11_sources_SOURCE_DIR}`子目录并不存在。

解决此挑战的一种方法是使用`FetchContent`模块，该模块在配置时获取 pybind11 源代码和 CMake 基础设施，以便我们可以使用`add_subdirectory`引用它。采用`FetchContent`模式，我们现在可以假设 pybind11 在构建树中可用，这使得我们能够构建并链接 Python 模块。

```cpp
add_library(account
  MODULE
    account.cpp
  )

target_link_libraries(account
  PUBLIC
    pybind11::module
  )
```

我们使用以下命令确保 Python 模块库获得一个与 Python 环境兼容的定义良好的前缀和后缀：

```cpp
set_target_properties(account
  PROPERTIES
    PREFIX ${PYTHON_MODULE_PREFIX}
    SUFFIX ${PYTHON_MODULE_EXTENSION}
  )
```

顶级`CMakeLists.txt`文件的其余部分用于测试（我们使用与前一个配方相同的`test.py`）。

# 还有更多

我们可以将 pybind11 源代码作为项目源代码仓库的一部分，这将简化 CMake 结构并消除在编译时需要网络访问 pybind11 源代码的要求。或者，我们可以将 pybind11 源路径定义为 Git 子模块（[`git-scm.com/book/en/v2/Git-Tools-Submodules`](https://git-scm.com/book/en/v2/Git-Tools-Submodules)），以简化更新 pybind11 源依赖关系。

在本例中，我们使用`FetchContent`解决了这个问题，它提供了一种非常紧凑的方法来引用 CMake 子项目，而无需显式跟踪其源代码。此外，我们还可以使用所谓的超级构建方法来解决这个问题（参见第八章，*The Superbuild Pattern*）。

# 另请参阅

若想了解如何暴露简单函数、定义文档字符串、映射内存缓冲区以及获取更多阅读材料，请参考 pybind11 文档：[`pybind11.readthedocs.io`](https://pybind11.readthedocs.io)。

# 使用 Python CFFI 混合 C、C++、Fortran 和 Python

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-06`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-06)找到，并包含 C++和 Fortran 示例。这些配方适用于 CMake 版本 3.5（及更高版本）。这两个版本的配方已在 GNU/Linux、macOS 和 Windows 上进行了测试。

在前三个菜谱中，我们讨论了 Cython、Boost.Python 和 pybind11 作为连接 Python 和 C++的工具，提供了一种现代且清晰的方法。在前面的菜谱中，主要接口是 C++接口。然而，我们可能会遇到没有 C++接口可供连接的情况，这时我们可能希望将 Python 与 Fortran 或其他语言连接起来。

在本菜谱中，我们将展示一种使用 Python C Foreign Function Interface（CFFI；另见[`cffi.readthedocs.io`](https://cffi.readthedocs.io)）的替代方法来连接 Python。由于 C 是编程语言的*通用语*，大多数编程语言（包括 Fortran）都能够与 C 接口通信，Python CFFI 是一种将 Python 与大量语言连接的工具。Python CFFI 的一个非常好的特点是，生成的接口是薄的且不侵入的，这意味着它既不限制 Python 层的语言特性，也不对 C 层以下的代码施加任何限制，除了需要一个 C 接口。

在本菜谱中，我们将应用 Python CFFI 通过 C 接口将 Python 和 C++连接起来，使用在前述菜谱中介绍的银行账户示例。我们的目标是实现一个上下文感知的接口，可以实例化多个银行账户，每个账户都携带其内部状态。我们将通过本菜谱结束时对如何使用 Python CFFI 将 Python 与 Fortran 连接进行评论。在第十一章，*打包项目*，菜谱 3，*通过 CMake/CFFI 构建的 C/Fortran/Python 项目通过 PyPI 分发*，我们将重新审视这个示例，并展示如何打包它，使其可以通过 pip 安装。

# 准备工作

我们将需要几个文件来完成这个菜谱。让我们从 C++实现和接口开始。我们将把这些文件放在一个名为`account/implementation`的子目录中。实现文件（`cpp_implementation.cpp`）与之前的菜谱类似，但包含了额外的`assert`语句，因为我们将在一个不透明的句柄中保持对象的状态，并且我们必须确保在尝试访问它之前创建了对象：

```cpp
#include "cpp_implementation.hpp"

#include <cassert>

Account::Account() {
  balance = 0.0;
  is_initialized = true;
}

Account::~Account() {
  assert(is_initialized);
  is_initialized = false;
}

void Account::deposit(const double amount) {
  assert(is_initialized);
  balance += amount;
}

void Account::withdraw(const double amount) {
  assert(is_initialized);
  balance -= amount;
}

double Account::get_balance() const {
  assert(is_initialized);
  return balance;
}
```

接口文件（`cpp_implementation.hpp`）包含以下内容：

```cpp
#pragma once

class Account {
public:
  Account();
  ~Account();

  void deposit(const double amount);
  void withdraw(const double amount);
  double get_balance() const;

private:
  double balance;
  bool is_initialized;
};
```

此外，我们隔离了一个 C—C++接口（`c_cpp_interface.cpp`）。这将是我们尝试使用 Python CFFI 连接的接口：

```cpp
#include "account.h"
#include "cpp_implementation.hpp"

#define AS_TYPE(Type, Obj) reinterpret_cast<Type *>(Obj)
#define AS_CTYPE(Type, Obj) reinterpret_cast<const Type *>(Obj)

account_context_t *account_new() {
  return AS_TYPE(account_context_t, new Account());
}

void account_free(account_context_t *context) { delete AS_TYPE(Account, context); }

void account_deposit(account_context_t *context, const double amount) {
  return AS_TYPE(Account, context)->deposit(amount);
}

void account_withdraw(account_context_t *context, const double amount) {
  return AS_TYPE(Account, context)->withdraw(amount);
}

double account_get_balance(const account_context_t *context) {
  return AS_CTYPE(Account, context)->get_balance();
}
```

在`account`目录下，我们描述了 C 接口（`account.h`）：

```cpp
/* CFFI would issue warning with pragma once */
#ifndef ACCOUNT_H_INCLUDED
#define ACCOUNT_H_INCLUDED

#ifndef ACCOUNT_API
#include "account_export.h"
#define ACCOUNT_API ACCOUNT_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct account_context;
typedef struct account_context account_context_t;

ACCOUNT_API
account_context_t *account_new();

ACCOUNT_API
void account_free(account_context_t *context);

ACCOUNT_API
void account_deposit(account_context_t *context, const double amount);

ACCOUNT_API
void account_withdraw(account_context_t *context, const double amount);

ACCOUNT_API
double account_get_balance(const account_context_t *context);

#ifdef __cplusplus
}
#endif

#endif /* ACCOUNT_H_INCLUDED */
```

我们还描述了 Python 接口，我们将在下面进行评论（`__init__.py`）：

```cpp
from subprocess import check_output
from cffi import FFI
import os
import sys
from configparser import ConfigParser
from pathlib import Path

def get_lib_handle(definitions, header_file, library_file):
    ffi = FFI()
    command = ['cc', '-E'] + definitions + [header_file]
    interface = check_output(command).decode('utf-8')

    # remove possible \r characters on windows which
    # would confuse cdef
    _interface = [l.strip('\r') for l in interface.split('\n')]

    ffi.cdef('\n'.join(_interface))
    lib = ffi.dlopen(library_file)
    return lib

# this interface requires the header file and library file
# and these can be either provided by interface_file_names.cfg
# in the same path as this file
# or if this is not found then using environment variables
_this_path = Path(os.path.dirname(os.path.realpath(__file__)))
_cfg_file = _this_path / 'interface_file_names.cfg'
if _cfg_file.exists():
    config = ConfigParser()
    config.read(_cfg_file)
    header_file_name = config.get('configuration', 'header_file_name')
    _header_file = _this_path / 'include' / header_file_name
    _header_file = str(_header_file)
    library_file_name = config.get('configuration', 'library_file_name')
    _library_file = _this_path / 'lib' / library_file_name
    _library_file = str(_library_file)
else:
    _header_file = os.getenv('ACCOUNT_HEADER_FILE')
    assert _header_file is not None
    _library_file = os.getenv('ACCOUNT_LIBRARY_FILE')
    assert _library_file is not None

_lib = get_lib_handle(definitions=['-DACCOUNT_API=', '-DACCOUNT_NOINCLUDE'],
                      header_file=_header_file,
                      library_file=_library_file)

# we change names to obtain a more pythonic API
new = _lib.account_new
free = _lib.account_free
deposit = _lib.account_deposit
withdraw = _lib.account_withdraw
get_balance = _lib.account_get_balance

__all__ = [
    '__version__',
    'new',
    'free',
    'deposit',
    'withdraw',
    'get_balance',
]
```

这是一堆文件，但是，正如我们将看到的，大部分接口工作是通用的和可重用的，实际的接口相当薄。总之，这是我们项目的布局：

```cpp
.
├── account
│   ├── account.h
│   ├── CMakeLists.txt
│   ├── implementation
│   │   ├── c_cpp_interface.cpp
│   │   ├── cpp_implementation.cpp
│   │   └── cpp_implementation.hpp
│   ├── __init__.py
│   └── test.py
└── CMakeLists.txt
```

# 如何操作

现在让我们使用 CMake 将这些文件组合成一个 Python 模块：

1.  顶层`CMakeLists.txt`文件包含一个熟悉的标题。此外，我们还根据 GNU 标准设置了编译库的位置：

```cpp
# define minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and supported language
project(recipe-06 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# specify where to place libraries
include(GNUInstallDirs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
```

1.  第二步是在 `account` 子目录下包含接口定义和实现源代码，我们将在下面详细介绍：

```cpp
# interface and sources
add_subdirectory(account)
```

1.  顶层的 `CMakeLists.txt` 文件以定义测试（需要 Python 解释器）结束：

```cpp
# turn on testing
enable_testing()

# require python
find_package(PythonInterp REQUIRED)

# define test
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

1.  包含的 `account/CMakeLists.txt` 定义了共享库：

```cpp
add_library(account
  SHARED
    implementation/c_cpp_interface.cpp
    implementation/cpp_implementation.cpp
  )

target_include_directories(account
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
```

1.  然后我们生成一个可移植的导出头文件：

```cpp
include(GenerateExportHeader)
generate_export_header(account
  BASE_NAME account
  )
```

1.  现在我们准备好了对 Python—C 接口进行测试：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

    Start 1: python_test
1/1 Test #1: python_test ...................... Passed 0.14 sec

100% tests passed, 0 tests failed out of 1
```

# 它是如何工作的

虽然前面的示例要求我们显式声明 Python—C 接口并将 Python 名称映射到 C(++) 符号，但 Python CFFI 会根据 C 头文件（在我们的例子中是 `account.h`）自动推断此映射。我们只需要向 Python CFFI 层提供描述 C 接口的头文件和包含符号的共享库。我们已经在主 `CMakeLists.txt` 文件中使用环境变量完成了此操作，并在 `__init__.py` 中查询了这些环境变量：

```cpp
# ...

def get_lib_handle(definitions, header_file, library_file):
    ffi = FFI()
    command = ['cc', '-E'] + definitions + [header_file]
    interface = check_output(command).decode('utf-8')

    # remove possible \r characters on windows which
    # would confuse cdef
    _interface = [l.strip('\r') for l in interface.split('\n')]

    ffi.cdef('\n'.join(_interface))
    lib = ffi.dlopen(library_file)
    return lib

# ...

_this_path = Path(os.path.dirname(os.path.realpath(__file__)))
_cfg_file = _this_path / 'interface_file_names.cfg'
if _cfg_file.exists():
    # we will discuss this section in chapter 11, recipe 3
else:
    _header_file = os.getenv('ACCOUNT_HEADER_FILE')
    assert _header_file is not None
    _library_file = os.getenv('ACCOUNT_LIBRARY_FILE')
    assert _library_file is not None

_lib = get_lib_handle(definitions=['-DACCOUNT_API=', '-DACCOUNT_NOINCLUDE'],
                      header_file=_header_file,
                      library_file=_library_file)

# ...
```

`get_lib_handle` 函数打开并解析头文件（使用 `ffi.cdef`），加载库（使用 `ffi.dlopen`），并返回库对象。前面的文件原则上具有通用性，可以不经修改地重用于其他连接 Python 和 C 或其他使用 Python CFFI 语言的项目。

`_lib` 库对象可以直接导出，但我们又多做了一步，以便在 Python 端使用时 Python 接口感觉更 *pythonic*：

```cpp
# we change names to obtain a more pythonic API
new = _lib.account_new
free = _lib.account_free
deposit = _lib.account_deposit
withdraw = _lib.account_withdraw
get_balance = _lib.account_get_balance

__all__ = [
    '__version__',
    'new',
    'free',
    'deposit',
    'withdraw',
    'get_balance',
]
```

有了这个改动，我们可以这样写：

```cpp
import account

account1 = account.new()

account.deposit(account1, 100.0)
```

另一种方法则不那么直观：

```cpp
from account import lib

account1 = lib.account_new()

lib.account_deposit(account1, 100.0)
```

请注意，我们能够使用上下文感知的 API 实例化和跟踪隔离的上下文：

```cpp
account1 = account.new()
account.deposit(account1, 10.0)

account2 = account.new()
account.withdraw(account1, 5.0)
account.deposit(account2, 5.0)
```

为了导入 `account` Python 模块，我们需要提供 `ACCOUNT_HEADER_FILE` 和 `ACCOUNT_LIBRARY_FILE` 环境变量，就像我们为测试所做的那样：

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

在 第十一章《打包项目》中，我们将讨论如何创建一个可以使用 pip 安装的 Python 包，其中头文件和库文件将安装在定义良好的位置，这样我们就不必定义任何环境变量来使用 Python 模块。

讨论了接口的 Python 方面之后，现在让我们考虑接口的 C 方面。`account.h` 的本质是这一部分：

```cpp
struct account_context;
typedef struct account_context account_context_t;

ACCOUNT_API
account_context_t *account_new();

ACCOUNT_API
void account_free(account_context_t *context);

ACCOUNT_API
void account_deposit(account_context_t *context, const double amount);

ACCOUNT_API
void account_withdraw(account_context_t *context, const double amount);

ACCOUNT_API
double account_get_balance(const account_context_t *context);
```

不透明的句柄 `account_context` 保存对象的状态。`ACCOUNT_API` 在 `account_export.h` 中定义，该文件由 CMake 在 `account/interface/CMakeLists.txt` 中生成：

```cpp
include(GenerateExportHeader)
generate_export_header(account
  BASE_NAME account
  )
```

`account_export.h` 导出头文件定义了接口函数的可见性，并确保以可移植的方式完成。我们将在 第十章《编写安装程序》中更详细地讨论这一点。实际的实现可以在 `cpp_implementation.cpp` 中找到。它包含 `is_initialized` 布尔值，我们可以检查该值以确保 API 函数按预期顺序调用：上下文不应在创建之前或释放之后被访问。

# 还有更多内容

在设计 Python-C 接口时，重要的是要仔细考虑在哪一侧分配数组：数组可以在 Python 侧分配并传递给 C(++)实现，或者可以在 C(++)实现中分配并返回一个指针。后一种方法在缓冲区大小*事先*未知的情况下很方便。然而，从 C(++)-侧返回分配的数组指针可能会导致内存泄漏，因为 Python 的垃圾回收不会“看到”已分配的数组。我们建议设计 C API，使得数组可以在外部分配并传递给 C 实现。然后，这些数组可以在`__init__.py`内部分配，如本例所示：

```cpp
from cffi import FFI
import numpy as np

_ffi = FFI()

def return_array(context, array_len):

    # create numpy array
    array_np = np.zeros(array_len, dtype=np.float64)

    # cast a pointer to its data
    array_p = _ffi.cast("double *", array_np.ctypes.data)

    # pass the pointer
    _lib.mylib_myfunction(context, array_len, array_p)

    # return the array as a list
    return array_np.tolist()
```

`return_array`函数返回一个 Python 列表。由于我们已经在 Python 侧完成了所有的分配工作，因此我们不必担心内存泄漏，可以将清理工作留给垃圾回收。

对于 Fortran 示例，我们建议读者参考以下配方仓库：[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-06/fortran-example`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-09/recipe-06/fortran-example)。与 C++实现的主要区别在于，账户库是由 Fortran 90 源文件编译而成，我们在`account/CMakeLists.txt`中对此进行了考虑：

```cpp
add_library(account
  SHARED
    implementation/fortran_implementation.f90
  )
```

上下文保存在用户定义的类型中：

```cpp
type :: account
  private
  real(c_double) :: balance
  logical :: is_initialized = .false.
end type
```

Fortran 实现能够通过使用`iso_c_binding`模块解析未更改的`account.h`中定义的符号和方法：

```cpp
module account_implementation

  use, intrinsic :: iso_c_binding, only: c_double, c_ptr

  implicit none

  private

  public account_new
  public account_free
  public account_deposit
  public account_withdraw
  public account_get_balance

  type :: account
    private
    real(c_double) :: balance
    logical :: is_initialized = .false.
  end type

contains

  type(c_ptr) function account_new() bind (c)
    use, intrinsic :: iso_c_binding, only: c_loc
    type(account), pointer :: f_context
    type(c_ptr) :: context

    allocate(f_context)
    context = c_loc(f_context)
    account_new = context
    f_context%balance = 0.0d0
    f_context%is_initialized = .true.
  end function

  subroutine account_free(context) bind (c)
    use, intrinsic :: iso_c_binding, only: c_f_pointer
    type(c_ptr), value :: context
    type(account), pointer :: f_context

    call c_f_pointer(context, f_context)
    call check_valid_context(f_context)
    f_context%balance = 0.0d0
    f_context%is_initialized = .false.
    deallocate(f_context)
  end subroutine

  subroutine check_valid_context(f_context)
    type(account), pointer, intent(in) :: f_context
    if (.not. associated(f_context)) then
        print *, 'ERROR: context is not associated'
        stop 1
    end if
    if (.not. f_context%is_initialized) then
        print *, 'ERROR: context is not initialized'
        stop 1
    end if
  end subroutine

  subroutine account_withdraw(context, amount) bind (c)
    use, intrinsic :: iso_c_binding, only: c_f_pointer
    type(c_ptr), value :: context
    real(c_double), value :: amount
    type(account), pointer :: f_context

    call c_f_pointer(context, f_context)
    call check_valid_context(f_context)
    f_context%balance = f_context%balance - amount
  end subroutine

  subroutine account_deposit(context, amount) bind (c)
    use, intrinsic :: iso_c_binding, only: c_f_pointer
    type(c_ptr), value :: context
    real(c_double), value :: amount
    type(account), pointer :: f_context

    call c_f_pointer(context, f_context)
    call check_valid_context(f_context)
    f_context%balance = f_context%balance + amount
  end subroutine

  real(c_double) function account_get_balance(context) bind (c)
    use, intrinsic :: iso_c_binding, only: c_f_pointer
    type(c_ptr), value, intent(in) :: context
    type(account), pointer :: f_context

    call c_f_pointer(context, f_context)
    call check_valid_context(f_context)
    account_get_balance = f_context%balance
  end function

end module
```

# 另请参阅

本配方和解决方案的灵感来源于 Armin Ronacher 的帖子“Beautiful Native Libraries”，[`lucumr.pocoo.org/2013/8/18/beautiful-native-libraries/`](http://lucumr.pocoo.org/2013/8/18/beautiful-native-libraries/)。
