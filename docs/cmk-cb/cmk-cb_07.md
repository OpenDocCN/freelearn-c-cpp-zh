# 第七章：生成源代码

在本章中，我们将介绍以下配方：

+   在配置时生成源代码

+   使用 Python 在配置时生成源代码

+   使用 Python 在构建时生成源代码

+   记录项目版本信息以确保可复现性

+   从文件记录项目版本

+   在配置时记录 Git 哈希

+   在构建时记录 Git 哈希

# 引言

对于大多数项目，源代码是通过版本控制系统进行跟踪的；它通常作为构建系统的输入，构建系统将其转换为对象、库和可执行文件。在某些情况下，我们使用构建系统在配置或构建步骤中生成源代码。这可以用于根据在配置步骤中收集的信息来微调源代码，或者自动化原本容易出错的重复代码的机械生成。生成源代码的另一个常见用例是记录配置或编译信息以确保可复现性。在本章中，我们将展示使用 CMake 提供的强大工具生成源代码的各种策略。

# 在配置时生成源代码

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-01)找到，包括一个 Fortran/C 示例。该配方适用于 CMake 版本 3.10（及以上），并在 GNU/Linux、macOS 和 Windows（使用 MSYS Makefiles）上进行了测试。

最直接的代码生成发生在配置时。例如，CMake 可以检测操作系统和支持的库；基于这些信息，我们可以定制构建哪些源代码，以向我们的库或程序的最终用户提供最佳性能。在本章和后续的一些配方中，我们将展示如何生成一个简单的源文件，该文件定义了一个函数来报告构建系统配置。

# 准备就绪

本配方的代码示例是 Fortran 和 C 语言的，为第九章，*混合语言项目*，其中将讨论混合语言编程。主程序是一个简单的 Fortran 可执行文件，它调用一个 C 函数`print_info()`，该函数将打印配置信息。值得注意的是，使用 Fortran 2003，编译器将处理名称重整（给定 C 函数的适当接口声明），正如我们在简单的`example.f90`源文件中看到的：

```cpp
program hello_world

  implicit none

  interface
    subroutine print_info() bind(c, name="print_info")
    end subroutine
  end interface

  call print_info()

end program
```

`print_info()` C 函数在模板文件`print_info.c.in`中定义。以`@`开始和结束的变量将在配置时被替换为其实际值：

```cpp
#include <stdio.h>
#include <unistd.h>

void print_info(void) {
  printf("\n");
  printf("Configuration and build information\n");
  printf("-----------------------------------\n");
  printf("\n");
  printf("Who compiled | %s\n", "@_user_name@");
  printf("Compilation hostname | %s\n", "@_host_name@");
  printf("Fully qualified domain name | %s\n", "@_fqdn@");
  printf("Operating system | %s\n",
         "@_os_name@, @_os_release@, @_os_version@");
  printf("Platform | %s\n", "@_os_platform@");
  printf("Processor info | %s\n",
         "@_processor_name@, @_processor_description@");
  printf("CMake version | %s\n", "@CMAKE_VERSION@");
  printf("CMake generator | %s\n", "@CMAKE_GENERATOR@");
  printf("Configuration time | %s\n", "@_configuration_time@");
  printf("Fortran compiler | %s\n", "@CMAKE_Fortran_COMPILER@");
  printf("C compiler | %s\n", "@CMAKE_C_COMPILER@");
  printf("\n");

  fflush(stdout);
}
```

# 如何操作

在我们的`CMakeLists.txt`中，我们首先必须收集配置选项，然后可以用它们的值替换`print_info.c.in`中相应的占位符；我们将 Fortran 和 C 源文件编译成一个可执行文件：

1.  我们创建一个混合 Fortran-C 项目，如下所示：

```cpp
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(recipe-01 LANGUAGES Fortran C)
```

1.  我们通过使用`execute_process`获得配置项目的用户的用户名：

```cpp
execute_process(
  COMMAND
    whoami
  TIMEOUT
    1
  OUTPUT_VARIABLE
    _user_name
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
```

1.  使用`cmake_host_system_information()`函数（我们在第二章，*检测环境*，第 5 个配方，*发现主机处理器指令集*中已经遇到过），我们可以查询更多系统信息：

```cpp
# host name information
cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
cmake_host_system_information(RESULT _fqdn QUERY FQDN)

# processor information
cmake_host_system_information(RESULT _processor_name QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT _processor_description QUERY PROCESSOR_DESCRIPTION)

# os information
cmake_host_system_information(RESULT _os_name QUERY OS_NAME)
cmake_host_system_information(RESULT _os_release QUERY OS_RELEASE)
cmake_host_system_information(RESULT _os_version QUERY OS_VERSION)
cmake_host_system_information(RESULT _os_platform QUERY OS_PLATFORM)
```

1.  我们还通过使用字符串操作函数获得配置的时间戳：

```cpp
string(TIMESTAMP _configuration_time "%Y-%m-%d %H:%M:%S [UTC]" UTC)
```

1.  我们现在准备通过使用 CMake 自己的`configure_file`函数来配置模板文件`print_info.c.in`。请注意，我们只要求以`@`开始和结束的字符串被替换：

```cpp
configure_file(print_info.c.in print_info.c @ONLY)
```

1.  最后，我们添加一个可执行目标并定义目标源，如下所示：

```cpp
add_executable(example "")

target_sources(example
  PRIVATE
    example.f90
    ${CMAKE_CURRENT_BINARY_DIR}/print_info.c
  )
```

1.  以下是示例输出：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

Configuration and build information
-----------------------------------

Who compiled                | somebody
Compilation hostname        | laptop
Fully qualified domain name | laptop
Operating system            | Linux, 4.16.13-1-ARCH, #1 SMP PREEMPT Thu May 31 23:29:29 UTC 2018
Platform                    | x86_64
Processor info              | Unknown P6 family, 2 core Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
CMake version               | 3.11.3
CMake generator             | Unix Makefiles
Configuration time          | 2018-06-25 15:38:03 [UTC]
Fortran compiler            | /usr/bin/f95
C compiler                  | /usr/bin/cc
```

# 它是如何工作的

`configure_file`命令可以复制文件并将它们的內容替换为变量值。在我们的示例中，我们使用`configure_file`来修改我们的模板文件的内容，并将其复制到一个可以编译到我们的可执行文件的位置。让我们看看我们对`configure_file`的调用：

```cpp
configure_file(print_info.c.in print_info.c @ONLY)
```

第一个参数是脚手架的名称：`print_info.c.in`。CMake 假设输入文件位于相对于项目根目录的位置；也就是说，在`${CMAKE_CURRENT_SOURCE_DIR}/print_info.c.in`中。第二个参数是我们选择的配置文件的名称，即`print_info.c`。输出文件假设位于相对于项目构建目录的位置；也就是说，在`${CMAKE_CURRENT_BINARY_DIR}/print_info.c`中。

当仅限制为两个参数，即输入和输出文件时，CMake 不仅会配置形如`@VAR@`的变量，还会配置形如`${VAR}`的变量。当`${VAR}`是语法的一部分且不应被修改时（例如在 shell 脚本中），这可能会造成不便。为了在这方面指导 CMake，应该将选项`@ONLY`传递给`configure_file`的调用，正如我们之前所展示的。

# 还有更多

请注意，将占位符替换为值时，期望 CMake 中的变量名与待配置文件中使用的变量名完全相同，并且位于`@`标记之间。在调用`configure_file`时定义的任何 CMake 变量都可以使用。这包括所有内置的 CMake 变量，例如`CMAKE_VERSION`或`CMAKE_GENERATOR`，在我们的示例中。此外，每当模板文件被修改时，重新构建代码将触发构建系统的重新生成。这样，配置的文件将始终保持最新。

完整的内部 CMake 变量列表可以通过使用`cmake --help-variable-list`从 CMake 手册中获得。

`file(GENERATE ...)`命令提供了一个有趣的替代`configure_file`的方法，因为它允许生成器表达式作为配置文件的一部分进行评估。然而，`file(GENERATE ...)`每次运行 CMake 时都会更新输出文件，这迫使所有依赖于该输出的目标重新构建。另请参见[`crascit.com/2017/04/18/generated-sources-in-cmake-builds/`](https://crascit.com/2017/04/18/generated-sources-in-cmake-builds/)。

# 使用 Python 在配置时生成源代码

本方法的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-02)找到，包括一个 Fortran/C 示例。本方法适用于 CMake 版本 3.10（及以上），并在 GNU/Linux、macOS 和 Windows 上使用 MSYS Makefiles 进行了测试。

在本方法中，我们将回顾之前的示例，并再次从模板`print_info.c.in`生成`print_info.c`。然而，这一次，我们将假设 CMake 函数`configure_file()`尚未被发明，并将使用 Python 脚本来模拟它。本方法的目标是学习如何通过使用一个熟悉的示例在配置时生成源代码。当然，在实际项目中，我们可能会更倾向于使用`configure_file()`，但是当我们面临在配置时使用 Python 生成源代码的挑战时，我们将知道如何操作。

我们应该指出，这个方法有一个严重的局限性，无法完全模拟`configure_file()`。我们在这里介绍的方法无法生成自动依赖项，该依赖项会在构建时重新生成`print_info.c`。换句话说，如果在配置步骤后删除了生成的`print_info.c`，该文件将不会被重新生成，构建步骤将会失败。为了正确模仿`configure_file()`的行为，我们需要使用`add_custom_command()`和`add_custom_target()`，我们将在接下来的第 3 个方法中使用，即“使用 Python 在构建时生成源代码”，在那里我们将克服这个限制。

在本方法中，我们将使用一个相对简单的 Python 脚本，下面我们将详细介绍。该脚本将读取`print_info.c.in`，并使用从 CMake 传递给 Python 脚本的参数替换文件中的占位符。对于更复杂的模板，我们推荐使用外部工具，如 Jinja（参见[`jinja.pocoo.org`](http://jinja.pocoo.org/)）。

# 准备工作

`print_info.c.in`和`example.f90`文件与前一个方法相比没有变化。此外，我们将使用一个 Python 脚本`configurator.py`，它提供了一个函数：

```cpp
def configure_file(input_file, output_file, vars_dict):

    with input_file.open('r') as f:
        template = f.read()

    for var in vars_dict:
        template = template.replace('@' + var + '@', vars_dict[var])

    with output_file.open('w') as f:
        f.write(template)
```

该函数读取一个输入文件，遍历`vars_dict`字典的所有键，将模式`@key@`替换为其对应值，并将结果写入输出文件。键值对将由 CMake 提供。

# 如何操作

与上一个配方类似，我们需要配置一个模板文件，但这次，我们将用 Python 脚本来模拟`configure_file()`函数。我们基本上保持`CMakeLists.txt`不变，但我们用一组命令替换了`configure_file(print_info.c.in print_info.c @ONLY)`，我们将逐步介绍这些命令：

1.  首先，我们构造一个变量，`_config_script`，它将保存我们稍后要执行的 Python 脚本：

```cpp
set(_config_script
"
from pathlib import Path
source_dir = Path('${CMAKE_CURRENT_SOURCE_DIR}')
binary_dir = Path('${CMAKE_CURRENT_BINARY_DIR}')
input_file = source_dir / 'print_info.c.in'
output_file = binary_dir / 'print_info.c'

import sys
sys.path.insert(0, str(source_dir))

from configurator import configure_file
vars_dict = {
    '_user_name':             '${_user_name}',
    '_host_name':             '${_host_name}',
    '_fqdn':                  '${_fqdn}',
    '_processor_name':        '${_processor_name}',
    '_processor_description': '${_processor_description}',
    '_os_name':               '${_os_name}',
    '_os_release':            '${_os_release}',
    '_os_version':            '${_os_version}',
    '_os_platform':           '${_os_platform}',
    '_configuration_time':    '${_configuration_time}',
    'CMAKE_VERSION':          '${CMAKE_VERSION}',
    'CMAKE_GENERATOR':        '${CMAKE_GENERATOR}',
    'CMAKE_Fortran_COMPILER': '${CMAKE_Fortran_COMPILER}',
    'CMAKE_C_COMPILER':       '${CMAKE_C_COMPILER}',
}
configure_file(input_file, output_file, vars_dict)
")
```

1.  然后，我们使用`find_package`来确保 CMake 可以使用 Python 解释器：

```cpp
find_package(PythonInterp QUIET REQUIRED)
```

1.  如果找到了 Python 解释器，我们可以在 CMake 内部执行`_config_script`，以生成`print_info.c`文件：

```cpp
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} "-c" ${_config_script}
  )
```

1.  之后，我们定义了可执行目标和依赖项，但这与上一个配方中的相同。同样，得到的输出也没有变化。

# 工作原理

让我们通过倒叙的方式来审视我们对`CMakeLists.txt`所做的更改。

我们执行了一个生成`print_info.c`的 Python 脚本。为了运行 Python 脚本，我们首先必须检测 Python 并构造 Python 脚本。Python 脚本导入了我们在`configurator.py`中定义的`configure_file`函数。它要求我们提供读写文件的位置，以及一个保存 CMake 变量及其值作为键值对的字典。

这个配方展示了一种生成配置报告的替代方法，该报告可以编译成可执行文件，甚至是一个库目标，通过将源的生成委托给外部脚本。我们在上一个配方中讨论的第一个方法更干净、更简单，但通过本配方中提出的方法，我们可以在原则上实现 Python（或其他语言）允许的任何配置时步骤。使用当前的方法，我们可以执行超出`cmake_host_system_information()`当前提供的功能的操作。

然而，我们需要记住这种方法的局限性，它无法生成自动依赖项，以便在构建时重新生成`print_info.c`。在下一个配方中，我们将克服这个限制。

# 还有更多

可以更简洁地表达这个配方。我们不必显式地构造`vars_dict`，这感觉有些重复，而是可以使用`get_cmake_property(_vars VARIABLES)`来获取此时定义的所有变量的列表，并可以遍历`_vars`的所有元素来访问它们的值：

```cpp
get_cmake_property(_vars VARIABLES)
foreach(_var IN ITEMS ${_vars})
  message("variable ${_var} has the value ${${_var}}")
endforeach()
```

采用这种方法，可以隐式地构建`vars_dict`。然而，必须注意转义包含诸如"`；`"这类字符的值，因为 Python 会将其解释为终止指令。

# 使用 Python 在构建时生成源代码

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-03)找到，包括一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

能够在构建时生成源代码是实用开发者工具箱中的一个强大功能，他们希望根据某些规则生成可能冗长且重复的代码，同时避免在源代码仓库中显式跟踪生成的代码。例如，我们可以想象根据检测到的平台或架构生成不同的源代码。或者，我们可以使用 Python 的简单性在构建时根据配置步骤中收集的输入生成明确且高效的 C++代码。其他相关的例子包括解析器生成器，如 Flex（[`github.com/westes/flex`](https://github.com/westes/flex)）和 Bison（[`www.gnu.org/software/bison/`](https://www.gnu.org/software/bison/)），元对象编译器，如 Qt moc（[`doc.qt.io/qt-5/moc.html`](http://doc.qt.io/qt-5/moc.html)），以及序列化框架，如 Google protobuf（[`developers.google.com/protocol-buffers/`](https://developers.google.com/protocol-buffers/)）。

# 准备工作

为了提供一个具体的例子，我们设想需要编写一段代码来验证一个数是否为质数。存在许多算法，例如，我们可以使用埃拉托色尼筛法来区分质数和非质数。如果我们需要验证很多数，我们不希望为每一个数都运行埃拉托色尼筛法算法。相反，我们希望一次性列出所有质数，直到某个上限，并使用查表法来验证大量数字。

在这个例子中，我们将使用 Python 在编译时生成查找表（一个质数向量）的 C++代码。当然，为了解决这个特定的编程问题，我们也可以使用 C++在运行时生成查找表。

让我们从一个名为`generate.py`的 Python 脚本开始。这个脚本接受两个命令行参数——一个将限制搜索的整数和一个输出文件名：

```cpp
"""
Generates C++ vector of prime numbers up to max_number
using sieve of Eratosthenes.
"""
import pathlib
import sys

# for simplicity we do not verify argument list
max_number = int(sys.argv[-2])
output_file_name = pathlib.Path(sys.argv[-1])

numbers = range(2, max_number + 1)
is_prime = {number: True for number in numbers}

for number in numbers:
    current_position = number
    if is_prime[current_position]:
        while current_position <= max_number:
            current_position += number
            is_prime[current_position] = False

primes = (number for number in numbers if is_prime[number])
code = """#pragma once

#include <vector>

const std::size_t max_number = {max_number};

std::vector<int> & primes() {{
  static std::vector<int> primes;

{push_back}

  return primes;
}}
"""
push_back = '\n'.join(['  primes.push_back({:d});'.format(x) for x in primes])
output_file_name.write_text(
    code.format(max_number=max_number, push_back=push_back))
```

我们的目标是生成一个头文件`primes.hpp`，在编译时生成，并在以下示例代码中包含它：

```cpp
#include "primes.hpp"

#include <iostream>
#include <vector>

int main() {
  std::cout << "all prime numbers up to " << max_number << ":";

  for (auto prime : primes())
    std::cout << " " << prime;

  std::cout << std::endl;

  return 0;
}
```

# 如何实现

以下是对`CMakeLists.txt`中命令的分解：

1.  首先，我们需要定义项目并检测 Python 解释器，如下所示：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonInterp QUIET REQUIRED)
```

1.  我们决定将待生成的代码放在`${CMAKE_CURRENT_BINARY_DIR}/generated`下，我们需要指示 CMake 创建这个目录：

```cpp
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
```

1.  这个 Python 脚本期望得到一个质数的上限，通过以下命令，我们可以设置一个默认值：

```cpp
set(MAX_NUMBER "100" CACHE STRING "Upper bound for primes")
```

1.  接下来，我们定义一个自定义命令来生成头文件：

```cpp
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  COMMAND
    ${PYTHON_EXECUTABLE} generate.py ${MAX_NUMBER} ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    generate.py
  )
```

1.  最后，我们定义了可执行文件及其目标，包括目录和依赖项：

```cpp
add_executable(example "")

target_sources(example
  PRIVATE
    example.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  )

target_include_directories(example
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

1.  我们现在准备测试实现，如下所示：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

all prime numbers up to 100: 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97
```

# 它是如何工作的

为了生成头文件，我们定义了一个自定义命令，该命令执行`generate.py`脚本，并接受`${MAX_NUMBER}`和文件路径（`${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp`）作为参数：

```cpp
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  COMMAND
    ${PYTHON_EXECUTABLE} generate.py ${MAX_NUMBER} ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    generate.py
  )
```

为了触发源代码生成，我们需要在可执行文件的定义中将其添加为源代码依赖项，这一任务可以通过`target_sources`轻松实现：

```cpp
target_sources(example
  PRIVATE
    example.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  )
```

在前述代码中，我们不必定义一个新的自定义目标。头文件将作为`example`的依赖项生成，并且每当`generate.py`脚本更改时都会重新构建。如果代码生成脚本生成多个源文件，重要的是所有生成的文件都被列为某个目标的依赖项。

# 还有更多内容

我们提到所有生成的文件都应该被列为某个目标的依赖项。然而，我们可能会遇到这样的情况：我们不知道这些文件的列表，因为它是根据我们提供给配置的输入由生成文件的脚本决定的。在这种情况下，我们可能会倾向于使用`file(GLOB ...)`来收集生成的文件到一个列表中（参见[`cmake.org/cmake/help/v3.5/command/file.html`](https://cmake.org/cmake/help/v3.5/command/file.html)）。

然而，请记住，`file(GLOB ...)`是在配置时执行的，而代码生成发生在构建时。因此，我们可能需要一个额外的间接层，将`file(GLOB ...)`命令放在一个单独的 CMake 脚本中，我们使用`${CMAKE_COMMAND} -P`执行该脚本，以便在构建时获取生成的文件列表。

# 记录项目版本信息以确保可重复性

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-04)找到，包括 C 和 Fortran 示例。本配方适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

代码版本不仅对可重复性很重要，而且对于记录 API 能力或简化支持请求和错误报告也很重要。源代码通常在某种版本控制下，并且可以使用 Git 标签等附加语义版本号（参见例如[`semver.org`](https://semver.org)）。然而，不仅源代码需要版本化，可执行文件也需要记录项目版本，以便它可以打印到代码输出或用户界面。

在本例中，我们将在 CMake 源代码中定义版本号。我们的目标是记录程序版本，以便在配置项目时将其记录到头文件中。生成的头文件随后可以在代码中的正确位置和时间被包含，以便将代码版本打印到输出文件或屏幕上。

# 准备就绪

我们将使用以下 C 文件（`example.c`）来打印版本信息：

```cpp
#include "version.h"

#include <stdio.h>

int main() {
  printf("This is output from code %s\n", PROJECT_VERSION);
  printf("Major version number: %i\n", PROJECT_VERSION_MAJOR);
  printf("Minor version number: %i\n", PROJECT_VERSION_MINOR);

  printf("Hello CMake world!\n");
}
```

在这里，我们假设`version.h`中定义了`PROJECT_VERSION_MAJOR`，`PROJECT_VERSION_MINOR`和`PROJECT_VERSION`。我们的目标是根据以下骨架生成`version.h`，即`version.h.in`：

```cpp
#pragma once

#define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define PROJECT_VERSION_PATCH @PROJECT_VERSION_PATCH@

#define PROJECT_VERSION "v@PROJECT_VERSION@"
```

我们将使用预处理器定义，但也可以使用字符串或整数常量以获得更多类型安全性（我们稍后将演示）。从 CMake 的角度来看，方法是一样的。

# 如何操作

我们将按照以下步骤在我们的模板头文件中注册版本：

1.  为了追踪代码版本，我们可以在`CMakeLists.txt`中调用 CMake 的`project`命令时定义项目版本：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-04 VERSION 2.0.1 LANGUAGES C)
```

1.  我们随后根据`version.h.in`配置`version.h`：

```cpp
configure_file(
  version.h.in
  generated/version.h
  @ONLY
  )
```

1.  最后，我们定义可执行文件并提供目标包含路径：

```cpp
add_executable(example example.c)

target_include_directories(example
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

# 工作原理

当使用`VERSION`参数调用 CMake 的`project`命令时，CMake 将为我们的项目设置`PROJECT_VERSION_MAJOR`，`PROJECT_VERSION_MINOR`和`PROJECT_VERSION_PATCH`。本食谱中的关键命令是`configure_file`，它接受一个输入文件（在这种情况下，`version.h.in`）并生成一个输出文件（在这种情况下，`generated/version.h`），通过将所有`@`之间的占位符扩展为其对应的 CMake 变量。它将`@PROJECT_VERSION_MAJOR@`替换为`2`，以此类推。使用关键字`@ONLY`，我们限制`configure_file`仅扩展`@variables@`，但不触及`${variables}`。后一种形式在`version.h.in`中没有使用，但它们经常出现在使用 CMake 配置 shell 脚本时。

生成的头文件可以包含在我们的示例代码中，并且版本信息可供打印：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

This is output from code v2.0.1
Major version number: 2
Minor version number: 0
Hello CMake world!
```

CMake 理解以`X.Y.Z.t`格式给出的版本号，并将设置`PROJECT_VERSION`和`<project-name>_VERSION`变量为传入的值。此外，`PROJECT_VERSION_MAJOR`（`<project-name>_VERSION_MAJOR`），`PROJECT_VERSION_MINOR`（`<project-name>_VERSION_MINOR`），`PROJECT_VERSION_PATCH`（`<project-name>_VERSION_PATCH`），和`PROJECT_VERSION_TWEAK`（`<project-name>_VERSION_TWEAK`）将被设置为`X`，`Y`，`Z`，和`t`，分别。

# 还有更多

为了确保预处理器变量仅在 CMake 变量被视为真常量时定义，可以在即将配置的头文件中使用`#cmakedefine`而不是`#define`，通过使用`configure_file`。

根据 CMake 变量是否被定义并且评估为真常量，`#cmakedefine YOUR_VARIABLE`将被替换为`#define YOUR_VARIABLE ...`或`/* #undef YOUR_VARIABLE */`。还有`#cmakedefine01`，它将根据变量是否定义将变量设置为`0`或`1`。

# 从文件记录项目版本

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-05)找到，包括一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

本食谱的目标与前一个相似，但起点不同；我们的计划是从文件中读取版本信息，而不是在`CMakeLists.txt`内部设置它。将版本信息保存在 CMake 源代码之外的单独文件中的动机是允许其他构建框架或开发工具使用该信息，独立于 CMake，而不在几个文件中重复信息。您可能希望与 CMake 并行使用的构建框架的一个例子是 Sphinx 文档框架，它生成文档并将其部署到 Read the Docs 服务以在线提供您的代码文档。

# 准备工作

我们将从一个名为`VERSION`的文件开始，其中包含以下内容：

```cpp
2.0.1-rc-2
```

这一次，我们将选择更注重类型安全，并将`PROGRAM_VERSION`定义为`version.hpp.in`中的字符串常量：

```cpp
#pragma once

#include <string>

const std::string PROGRAM_VERSION = "@PROGRAM_VERSION@";
```

我们将在下面的示例源代码（`example.cpp`）中包含生成的`version.hpp`：

```cpp
// provides PROGRAM_VERSION
#include "version.hpp"

#include <iostream>

int main() {
  std::cout << "This is output from code v" << PROGRAM_VERSION
                                            << std::endl;

  std::cout << "Hello CMake world!" << std::endl;
}
```

# 如何操作

以下展示了我们如何一步步完成任务：

1.  `CMakeLists.txt`定义了最低版本、项目名称、语言和标准：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  我们按照以下方式从文件中读取版本信息：

```cpp
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROGRAM_VERSION)
  string(STRIP "${PROGRAM_VERSION}" PROGRAM_VERSION)
else()
  message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/VERSION not found")
endif()
```

1.  然后我们配置头文件：

```cpp
configure_file(
  version.hpp.in
  generated/version.hpp
  @ONLY
  )
```

1.  最后，我们定义了可执行文件及其依赖项：

```cpp
add_executable(example example.cpp)

target_include_directories(example
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

1.  然后我们准备测试它：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

This is output from code v2.0.1-rc-2
Hello CMake world!
```

# 它是如何工作的

我们使用了以下结构从名为`VERSION`的文件中读取版本字符串：

```cpp
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROGRAM_VERSION)
  string(STRIP "${PROGRAM_VERSION}" PROGRAM_VERSION)
else()
  message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/VERSION not found")
endif()
```

在这里，我们首先检查该文件是否存在，如果不存在则发出错误消息。如果存在，我们将文件内容读入名为`PROGRAM_VERSION`的变量中，并去除任何尾随空格。一旦设置了变量`PROGRAM_VERSION`，就可以用来配置`version.hpp.in`以生成`generated/version.hpp`，如下所示：

```cpp
configure_file(
  version.hpp.in
  generated/version.hpp
  @ONLY
  )
```

# 在配置时记录 Git 哈希

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-06`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-06)找到，包括一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

大多数现代源代码仓库都使用 Git 作为版本控制系统进行跟踪，这一事实可以归因于仓库托管平台 GitHub 的巨大流行。因此，在本食谱中，我们将使用 Git；然而，动机和实现将适用于其他版本控制系统。如果我们以 Git 为例，一个提交的 Git 哈希值唯一地确定了源代码的状态。因此，为了唯一地标记可执行文件，我们将尝试通过在头文件中记录哈希字符串来将 Git 哈希值烧录到可执行文件中，该头文件可以在代码中的正确位置包含和使用。

# 准备工作

我们需要两个源文件，都与之前的食谱非常相似。一个将使用记录的哈希值进行配置（`version.hpp.in`），如下所示：

```cpp
#pragma once

#include <string>

const std::string GIT_HASH = "@GIT_HASH@";
```

我们还需要一个示例源文件（`example.cpp`），它将打印哈希值到屏幕上：

```cpp
#include "version.hpp"

#include <iostream>

int main() {
  std::cout << "This code has been configured from version " << GIT_HASH
            << std::endl;
}
```

这个食谱还假设我们处于至少有一个提交的 Git 仓库中。因此，使用 `git init` 初始化这个示例，并通过 `git add <filename>` 和 `git commit` 创建提交，以获得有意义的示例。

# 如何操作

以下步骤说明了如何从 Git 记录版本信息：

1.  在 `CMakeLists.txt` 中，我们首先定义项目和语言支持：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

1.  然后，我们使用以下代码片段来定义一个变量，`GIT_HASH`：

```cpp
# in case Git is not available, we default to "unknown"
set(GIT_HASH "unknown")

# find Git and if available set GIT_HASH variable
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

message(STATUS "Git hash is ${GIT_HASH}")
```

1.  其余的 `CMakeLists.txt` 与之前的食谱中的相似：

```cpp
# generate file version.hpp based on version.hpp.in
configure_file(
  version.hpp.in
  generated/version.hpp
  @ONLY
  )

# example code
add_executable(example example.cpp)

# needs to find the generated header file
target_include_directories(example
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

1.  我们可以通过以下方式验证输出（哈希值会有所不同）：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

This code has been configured from version d58c64f
```

# 工作原理

我们使用 `find_package(Git QUIET)` 来检测系统上是否安装了 Git。如果安装了（如果 `GIT_FOUND` 为真），我们运行一个 Git 命令：`${GIT_EXECUTABLE} log -1 --pretty=format:%h`。这个命令给我们提供了当前提交哈希的简短版本。当然，我们完全有灵活性来运行另一个 Git 命令，而不是这个。我们要求 `execute_process` 命令将命令的结果放入一个名为 `GIT_HASH` 的变量中，然后我们去除任何尾随的空白。使用 `ERROR_QUIET`，我们要求命令在 Git 命令由于某种原因失败时不停止配置。

由于 Git 命令可能会失败（源代码可能已经在 Git 仓库之外分发）或者系统上甚至可能没有安装 Git，我们希望为变量设置一个默认值，如下所示：

```cpp
set(GIT_HASH "unknown")
```

这个食谱的一个问题是 Git 哈希值是在配置时记录的，而不是在构建时。在下一个食谱中，我们将演示如何实现后一种方法。

# 在构建时记录 Git 哈希值

这个食谱的代码可以在 [`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-07`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-06/recipe-07) 找到，包括一个 C++ 示例。这个食谱适用于 CMake 版本 3.5（及更高版本），并且已经在 GNU/Linux、macOS 和 Windows 上进行了测试。

在之前的配方中，我们在配置时记录了代码仓库的状态（Git 哈希），并且在可执行文件中记录仓库状态非常有用。然而，之前方法的一个不满意之处是，如果我们更改分支或提交更改后配置代码，源代码中包含的版本记录可能会指向错误的 Git 哈希。在本配方中，我们希望更进一步，并演示如何在构建时记录 Git 哈希（或一般而言，执行其他操作），以确保每次我们构建代码时都会运行这些操作，因为我们可能只配置一次，但构建多次。

# 准备工作

我们将使用与之前配方相同的`version.hpp.in`，并且只会对`example.cpp`文件进行最小限度的修改，以确保它打印出构建时的 Git 哈希值：

```cpp
#include "version.hpp"

#include <iostream>

int main() {
  std::cout << "This code has been built from version " << GIT_HASH << std::endl;
}
```

# 如何操作

在构建时将 Git 信息保存到`version.hpp`头文件将需要以下操作：

1.  我们将把之前配方中`CMakeLists.txt`的大部分代码移动到一个单独的文件中，并将其命名为`git-hash.cmake`：

```cpp
# in case Git is not available, we default to "unknown"
set(GIT_HASH "unknown")

# find Git and if available set GIT_HASH variable
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    )
endif()

message(STATUS "Git hash is ${GIT_HASH}")

# generate file version.hpp based on version.hpp.in
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/version.hpp.in
  ${TARGET_DIR}/generated/version.hpp
  @ONLY
  )
```

1.  `CMakeLists.txt`现在剩下我们非常熟悉的部分：

```cpp
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-07 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# example code
add_executable(example example.cpp)

# needs to find the generated header file
target_include_directories(example
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

1.  `CMakeLists.txt`的剩余部分记录了每次我们构建代码时 Git 哈希值，如下所示：

```cpp
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    ALL
  COMMAND
    ${CMAKE_COMMAND} -D TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/git-hash.cmake
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

# rebuild version.hpp every time
add_custom_target(
  get_git_hash
  ALL
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
  )

# version.hpp has to be generated
# before we start building example
add_dependencies(example get_git_hash)
```

# 它是如何工作的

在本配方中，我们实现了在构建时执行 CMake 代码。为此，我们定义了一个自定义命令：

```cpp
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    ALL
  COMMAND
    ${CMAKE_COMMAND} -D TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/git-hash.cmake
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

我们还定义了一个自定义目标，如下所示：

```cpp
add_custom_target(
  get_git_hash
  ALL
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
  )
```

自定义命令调用 CMake 执行`git-hash.cmake`CMake 脚本。这是通过使用`-P`CLI 开关来实现的，以传递脚本的位置。请注意，我们可以使用`-D`CLI 开关传递选项，就像我们通常所做的那样。`git-hash.cmake`脚本生成`${TARGET_DIR}/generated/version.hpp`。自定义目标添加到`ALL`目标，并依赖于自定义命令的输出。换句话说，当我们构建默认目标时，我们确保自定义命令被执行。此外，请注意自定义命令将`ALL`目标作为输出。这样，我们确保每次都会生成`version.hpp`。

# 还有更多

我们可以增强配方，以便在记录的 Git 哈希之外包含额外信息。检测构建环境是否“脏”（即是否包含未提交的更改和未跟踪的文件）或“干净”并不罕见。可以使用`git describe --abbrev=7 --long --always --dirty --tags`检测此信息。根据可重复性的雄心，甚至可以将`git status`的完整输出记录到头文件中，但我们将其作为练习留给读者。
