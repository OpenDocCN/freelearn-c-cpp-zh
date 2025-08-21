# 第二章：从简单的可执行文件到库

在本章中，我们将介绍以下内容：

+   将单个源文件编译成可执行文件

+   切换生成器

+   构建和链接静态和共享库

+   使用条件控制编译

+   向用户展示选项

+   指定编译器

+   切换构建类型

+   控制编译器标志

+   设置语言标准

+   使用控制流结构

# 引言

本章中的示例将引导您完成构建代码所需的基本任务：编译可执行文件、编译库、根据用户输入执行构建操作等。CMake 是一个*构建系统生成器*，特别适合于平台和编译器无关。我们努力在本章中展示这一方面。除非另有说明，所有示例都与操作系统无关；它们可以在不加修改的情况下在 GNU/Linux、macOS 和 Windows 上运行。

本书中的示例主要针对 C++项目，并使用 C++示例进行演示，但 CMake 也可用于其他语言的项目，包括 C 和 Fortran。对于任何给定的示例，只要合理，我们都尝试包括 C++、C 和 Fortran 的示例。这样，您就可以选择您喜欢的语言的示例。有些示例是专门为突出特定语言选择时需要克服的挑战而定制的。

# 将单个源文件编译成可执行文件

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-01`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-01)找到，并提供了 C++、C 和 Fortran 的示例。本示例适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在本示例中，我们将演示如何运行 CMake 来配置和构建一个简单的项目。该项目由单个源文件组成，用于单个可执行文件。我们将讨论 C++项目，但 GitHub 存储库中提供了 C 和 Fortran 的示例。

# 准备工作

我们希望将以下源代码编译成一个单独的可执行文件：

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() { return std::string("Hello, CMake world!"); }

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

# 如何操作

除了源文件外，我们还需要向 CMake 提供一个描述，说明如何为构建工具配置项目。描述使用 CMake 语言完成，其全面的文档可以在[`cmake.org/cmake/help/latest/`](https://cmake.org/cmake/help/latest/)在线找到。我们将把 CMake 指令放入一个名为`CMakeLists.txt`的文件中。

文件名是*区分大小写*的；它必须被称为`CMakeLists.txt`，以便 CMake 能够解析它。

详细来说，以下是遵循的步骤：

1.  使用您喜欢的编辑器打开一个文本文件。该文件将被命名为`CMakeLists.txt`。

1.  第一行设置 CMake 的最低要求版本。如果使用的 CMake 版本低于该版本，将发出致命错误：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

1.  第二行声明了项目的名称（recipe-01）和支持的语言（`CXX`代表 C++）：

```cpp
project(recipe-01 LANGUAGES CXX)
```

1.  我们指示 CMake 创建一个新的*目标*：可执行文件`hello-world`。这个可执行文件是通过编译和链接源文件`hello-world.cpp`生成的。CMake 将使用所选编译器和构建自动化工具的默认设置：

```cpp
add_executable(hello-world hello-world.cpp)
```

1.  将文件保存在与源文件`hello-world.cpp`相同的目录中。请记住，它只能被命名为`CMakeLists.txt`。

1.  我们现在准备通过创建并进入构建目录来配置项目：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-01/cxx-example/build
```

1.  如果一切顺利，项目配置已经在构建目录中生成。我们现在可以编译可执行文件了：

```cpp
$ cmake --build .

Scanning dependencies of target hello-world
[ 50%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world
```

# 它是如何工作的

在这个示例中，我们使用了一个简单的`CMakeLists.txt`来构建一个“Hello world”可执行文件：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES CXX)

add_executable(hello-world hello-world.cpp)
```

CMake 语言是不区分大小写的，但参数是区分大小写的。

CMake 中，C++是默认的编程语言。然而，我们建议始终在`project`命令中使用`LANGUAGES`选项明确声明项目的语言。

为了配置项目并生成其构建系统，我们必须通过命令行界面（CLI）运行 CMake。CMake CLI 提供了许多开关，`cmake --help`将输出屏幕上列出所有可用开关的完整帮助菜单。我们将在本书中了解更多关于它们的信息。正如您将从`cmake --help`的输出中注意到的，大多数开关将允许您访问 CMake 手册。生成构建系统的典型命令序列如下：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
```

在这里，我们创建了一个目录，`build`，其中将生成构建系统，我们进入了`build`目录，并通过指向`CMakeLists.txt`的位置调用了 CMake（在这种情况下位于父目录中）。可以使用以下调用来实现相同的效果：

```cpp
$ cmake -H. -Bbuild
```

这个调用是跨平台的，并引入了`-H`和`-B`CLI 开关。使用`-H.`我们指示 CMake 在当前目录中搜索根`CMakeLists.txt`文件。`-Bbuild`告诉 CMake 在名为`build`的目录中生成所有文件。

注意，`cmake -H. -Bbuild`调用 CMake 仍在进行标准化：[`cmake.org/pipermail/cmake-developers/2018-January/030520.html`](https://cmake.org/pipermail/cmake-developers/2018-January/030520.html)。这就是为什么我们在这本书中将使用传统方法（创建一个构建目录，进入它，并通过指向`CMakeLists.txt`的位置来配置项目）。

运行`cmake`命令会输出一系列状态消息来通知您配置情况：

```cpp
$ cmake ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-01/cxx-example/build
```

在`CMakeLists.txt`所在的同一目录中运行`cmake .`原则上足以配置项目。但是，CMake 随后会将所有生成的文件写入项目的**根目录**。这将是一个*源内构建*，通常是不希望的，因为它混合了项目的源代码和构建树。我们演示的*源外构建*是首选实践。

CMake 是一个*生成器*构建系统。您描述了构建系统（如 Unix Makefiles、Ninja、Visual Studio 等）需要执行的操作类型，以便编译您的代码。然后，CMake 为所选的构建系统*生成*相应的指令。默认情况下，在 GNU/Linux 和 macOS 系统上，CMake 使用 Unix Makefiles 生成器。在 Windows 上，Visual Studio 是默认生成器。我们将在下一个配方中更详细地了解生成器，并在第十三章，*替代生成器和交叉编译*中重新审视生成器。

在 GNU/Linux 上，CMake 将默认生成 Unix Makefiles 以构建项目：

+   `Makefile`：`make`将运行以构建项目的一组指令。

+   `CMakeFiles`：该目录包含 CMake 用于检测操作系统、编译器等的临时文件。此外，根据所选的*生成器*，它还包含特定于项目的文件。

+   `cmake_install.cmake`：一个 CMake 脚本，用于处理安装规则，在安装时使用。

+   `CMakeCache.txt`：正如文件名所示，这是 CMake 的缓存文件。在重新运行配置时，CMake 会使用此文件。

要构建示例项目，我们运行了以下命令：

```cpp
$ cmake --build .
```

此命令是一个通用的跨平台包装器，用于所选*生成器*的本地构建命令，在本例中为`make`。我们不应忘记测试我们的示例可执行文件：

```cpp
$ ./hello-world

Hello, CMake world!
```

最后，我们应该指出，CMake 不强制要求特定的名称或特定的位置用于构建目录。我们可以将其完全放置在项目路径之外。这将同样有效：

```cpp
$ mkdir -p /tmp/someplace
$ cd /tmp/someplace
$ cmake /path/to/source
$ cmake --build .
```

# 还有更多

官方文档位于[`cmake.org/runningcmake/`](https://cmake.org/runningcmake/)，提供了运行 CMake 的简明概述。由 CMake 生成的构建系统，在上面的示例中为`Makefile`，将包含构建给定项目的对象文件、可执行文件和库的目标和规则。在当前示例中，`hello-world`可执行文件是我们唯一的目标，但是运行命令：

```cpp
$ cmake --build . --target help

The following are some of the valid targets for this Makefile:
... all (the default if no target is provided)
... clean
... depend
... rebuild_cache
... hello-world
... edit_cache
... hello-world.o
... hello-world.i
... hello-world.s
```

揭示了 CMake 生成的目标比仅构建可执行文件本身所需的目标更多。可以使用`cmake --build . --target <target-name>`语法选择这些目标，并实现以下目标：

+   `all`（或使用 Visual Studio 生成器时的`ALL_BUILD`）是默认目标，将构建项目中的所有其他目标。

+   `clean`，是选择删除所有生成的文件的目标。

+   `depend`，将调用 CMake 为源文件生成任何依赖项。

+   `rebuild_cache`，将再次调用 CMake 来重建`CMakeCache.txt`。如果需要从源代码中添加新条目，这是必要的。

+   `edit_cache`，这个目标将允许你直接编辑缓存条目。

对于更复杂的项目，包括测试阶段和安装规则，CMake 将生成额外的便利目标：

+   `test`（或使用 Visual Studio 生成器时的`RUN_TESTS`）将使用 CTest 运行测试套件。我们将在第四章，*创建和运行测试*中详细讨论测试和 CTest。

+   `install`，将执行项目的安装规则。我们将在第十章，*编写安装程序*中讨论安装规则。

+   `package`，这个目标将调用 CPack 来为项目生成可重新分发的包。打包和 CPack 将在第十一章，*打包项目*中讨论。

# 切换生成器

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-02`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-02)找到，并提供了 C++、C 和 Fortran 的示例。本配方适用于 CMake 3.5（及以上）版本，并在 GNU/Linux、macOS 和 Windows 上进行了测试。

CMake 是一个构建系统生成器，单个`CMakeLists.txt`可以用于为不同平台上的不同工具链配置项目。你可以在`CMakeLists.txt`中描述构建系统需要运行的操作来配置和编译你的代码。基于这些指令，CMake 将为所选构建系统（Unix Makefiles、Ninja、Visual Studio 等）生成相应的指令。我们将在第十三章，*替代生成器和交叉编译*中重新讨论生成器。

# 准备工作

CMake 支持大量不同平台的原生构建工具。无论是命令行工具，如 Unix Makefiles 和 Ninja，还是集成开发环境（IDE）工具，都得到支持。你可以通过运行以下命令来获取你平台和已安装的 CMake 版本上可用的生成器的最新列表：

```cpp
$ cmake --help
```

此命令的输出将列出 CMake 命令行界面的所有选项。在底部，你将找到可用生成器的列表。例如，这是在安装了 CMake 3.11.2 的 GNU/Linux 机器上的输出：

```cpp
Generators

The following generators are available on this platform:
  Unix Makefiles = Generates standard UNIX makefiles.
  Ninja = Generates build.ninja files.
  Watcom WMake = Generates Watcom WMake makefiles.
  CodeBlocks - Ninja = Generates CodeBlocks project files.

  CodeBlocks - Unix Makefiles = Generates CodeBlocks project files.
  CodeLite - Ninja = Generates CodeLite project files.
  CodeLite - Unix Makefiles = Generates CodeLite project files.
  Sublime Text 2 - Ninja = Generates Sublime Text 2 project files.
  Sublime Text 2 - Unix Makefiles = Generates Sublime Text 2 project files.
  Kate - Ninja = Generates Kate project files.
  Kate - Unix Makefiles = Generates Kate project files.
  Eclipse CDT4 - Ninja = Generates Eclipse CDT 4.0 project files.
  Eclipse CDT4 - Unix Makefiles= Generates Eclipse CDT 4.0 project files.
```

通过本配方，我们将展示为同一项目切换生成器是多么容易。

# 如何操作

我们将重用之前的配方中的`hello-world.cpp`和`CMakeLists.txt`。唯一的区别在于 CMake 的调用方式，因为我们现在必须使用`-G`命令行开关显式传递生成器。

1.  首先，我们使用以下命令配置项目：

```cpp
$ mkdir -p build
$ cd build
$ cmake -G Ninja ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-02/cxx-example/build
```

1.  在第二步，我们构建项目：

```cpp
$ cmake --build .

[2/2] Linking CXX executable hello-world
```

# 它是如何工作的

我们已经看到，配置步骤的输出与之前的配方相比没有变化。然而，编译步骤的输出和构建目录的内容将会有所不同，因为每个生成器都有其特定的文件集：

+   `build.ninja` 和 `rules.ninja`：包含 Ninja 的所有构建语句和构建规则。

+   `CMakeCache.txt`：无论选择哪种生成器，CMake 总是会在此文件中生成自己的缓存。

+   `CMakeFiles`：包含 CMake 在配置过程中生成的临时文件。

+   `cmake_install.cmake`：处理安装规则的 CMake 脚本，用于安装时使用。

注意 `cmake --build .` 是如何将 `ninja` 命令包装在一个统一的跨平台接口中的。

# 另请参阅

我们将在 第十三章，*替代生成器和交叉编译*中讨论替代生成器和交叉编译。

CMake 文档是了解生成器的良好起点：[`cmake.org/cmake/help/latest/manual/cmake-generators.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)。

# 构建和链接静态和共享库

本配方的代码可在 [`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-03`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-03) 获取，并提供了 C++ 和 Fortran 的示例。本配方适用于 CMake 3.5（及以上）版本，并在 GNU/Linux、macOS 和 Windows 上进行了测试。

一个项目几乎总是由多个源文件构建的单个可执行文件组成。项目被拆分到多个源文件中，通常分布在源树的不同子目录中。这种做法不仅有助于在项目中组织源代码，而且极大地促进了模块化、代码重用和关注点分离，因为可以将常见任务分组到库中。这种分离还简化了项目开发过程中的重新编译并加快了速度。在本配方中，我们将展示如何将源分组到库中，以及如何将目标链接到这些库。

# 准备工作

让我们回到最初的例子。然而，我们不再使用单一的源文件来编译可执行文件，而是引入一个类来封装要打印到屏幕的消息。这是我们更新的 `hello-world.cpp`：

```cpp
#include "Message.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  Message say_hello("Hello, CMake World!");

  std::cout << say_hello << std::endl;

  Message say_goodbye("Goodbye, CMake World");

  std::cout << say_goodbye << std::endl;

  return EXIT_SUCCESS;
}
```

`Message` 类封装了一个字符串，提供了对 `<<` 操作符的重载，并由两个源文件组成：`Message.hpp` 头文件和相应的 `Message.cpp` 源文件。`Message.hpp` 接口文件包含以下内容：

```cpp
#pragma once

#include <iosfwd>
#include <string>

class Message {
public:
  Message(const std::string &m) : message_(m) {}

  friend std::ostream &operator<<(std::ostream &os, Message &obj) {
    return obj.printObject(os);
  }

private:
  std::string message_;
  std::ostream &printObject(std::ostream &os);
};
```

相应的实现包含在 `Message.cpp` 中：

```cpp
#include "Message.hpp"

#include <iostream>
#include <string>

std::ostream &Message::printObject(std::ostream &os) {
  os << "This is my very nice message: " << std::endl;
  os << message_;

  return os;
}
```

# 如何操作

这两个新文件也需要编译，我们需要相应地修改 `CMakeLists.txt`。然而，在这个例子中，我们希望先将它们编译成一个库，而不是直接编译成可执行文件：

1.  创建一个新的 *目标*，这次是静态库。库的名称将是目标的名称，源代码列表如下：

```cpp
add_library(message 
  STATIC
    Message.hpp
    Message.cpp
  )
```

1.  创建 `hello-world` 可执行文件的目标未作修改：

```cpp
add_executable(hello-world hello-world.cpp) 
```

1.  最后，告诉 CMake 库目标需要链接到可执行目标：

```cpp
target_link_libraries(hello-world message)
```

1.  我们可以使用与之前相同的命令进行配置和构建。这次将编译一个库，与 `hello-world` 可执行文件一起：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .

Scanning dependencies of target message
[ 25%] Building CXX object CMakeFiles/message.dir/Message.cpp.o
[ 50%] Linking CXX static library libmessage.a
[ 50%] Built target message
Scanning dependencies of target hello-world
[ 75%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world

$ ./hello-world

This is my very nice message: 
Hello, CMake World!
This is my very nice message: 
Goodbye, CMake World
```

# 工作原理

前面的示例介绍了两个新命令：

+   `add_library(message STATIC Message.hpp Message.cpp)`：这将生成将指定源代码编译成库所需的构建工具指令。`add_library` 的第一个参数是目标的名称。在整个 `CMakeLists.txt` 中可以使用相同的名称来引用该库。生成的库的实际名称将由 CMake 通过在前面添加前缀 `lib` 和作为后缀的适当扩展名来形成。库扩展名是根据第二个参数 `STATIC` 或 `SHARED` 以及操作系统来确定的。

+   `target_link_libraries(hello-world message)`：将库链接到可执行文件。此命令还将确保 `hello-world` 可执行文件正确依赖于消息库。因此，我们确保消息库总是在我们尝试将其链接到 `hello-world` 可执行文件之前构建。

成功编译后，构建目录将包含 `libmessage.a` 静态库（在 GNU/Linux 上）和 `hello-world` 可执行文件。

CMake 接受 `add_library` 的第二个参数的其他有效值，我们将在本书的其余部分遇到所有这些值：

+   `STATIC`，我们已经遇到过，将用于创建静态库，即用于链接其他目标（如可执行文件）的对象文件的归档。

+   `SHARED` 将用于创建共享库，即可以在运行时动态链接和加载的库。从静态库切换到动态共享对象（DSO）就像在 `CMakeLists.txt` 中使用 `add_library(message SHARED Message.hpp Message.cpp)` 一样简单。

+   `OBJECT` 可用于将传递给 `add_library` 的列表中的源代码编译成目标文件，但不将它们归档到静态库中，也不将它们链接到共享对象中。如果需要一次性创建静态库和共享库，使用对象库尤其有用。我们将在本示例中演示这一点。

+   `MODULE` 库再次是 DSOs。与 `SHARED` 库不同，它们不在项目内链接到任何其他目标，但可能会在以后动态加载。这是构建运行时插件时要使用的参数。

CMake 还能够生成特殊类型的库。这些库在构建系统中不产生输出，但在组织目标之间的依赖关系和构建要求方面非常有帮助：

+   `IMPORTED`，这种类型的库目标代表位于项目*外部*的库。这种类型的库的主要用途是模拟项目上游包提供的预先存在的依赖项。因此，`IMPORTED`库应被视为不可变的。我们将在本书的其余部分展示使用`IMPORTED`库的示例。另请参见：[`cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets`](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets)

+   `INTERFACE`，这种特殊的 CMake 库类型类似于`IMPORTED`库，但它是可变的，没有位置。它的主要用例是模拟项目外部目标的使用要求。我们将在第 5 个配方中展示`INTERFACE`库的使用案例，即*将依赖项作为 Conda 包分发项目*，在第十一章，*打包项目*中。另请参见：[`cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries`](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries)

+   `ALIAS`，顾名思义，这种类型的库为目标定义了一个别名。因此，不可能为`IMPORTED`库选择别名。另请参见：[`cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#alias-libraries`](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#alias-libraries)

在本例中，我们直接使用`add_library`收集源文件。在后面的章节中，我们将展示使用`target_sources`CMake 命令来收集源文件，特别是在第七章，*项目结构化*中。也可以参考 Craig Scott 的这篇精彩博文：[`crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/`](https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)，它进一步说明了使用`target_sources`命令的动机。

# 还有更多

现在让我们展示 CMake 中提供的对象库功能的使用。我们将使用相同的源文件，但修改`CMakeLists.txt`：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03 LANGUAGES CXX)

add_library(message-objs
  OBJECT
    Message.hpp
    Message.cpp
  )

# this is only needed for older compilers
# but doesn't hurt either to have it
set_target_properties(message-objs
  PROPERTIES
    POSITION_INDEPENDENT_CODE 1
  )

add_library(message-shared
  SHARED
    $<TARGET_OBJECTS:message-objs>
  )

add_library(message-static
  STATIC
    $<TARGET_OBJECTS:message-objs>
  )

add_executable(hello-world hello-world.cpp)

target_link_libraries(hello-world message-static)
```

首先，注意`add_library`命令已更改为`add_library(message-objs OBJECT Message.hpp Message.cpp)`。此外，我们必须确保编译为对象文件生成位置无关代码。这是通过使用`set_target_properties`命令设置`message-objs`目标的相应*属性*来完成的。

对于目标显式设置`POSITION_INDEPENDENT_CODE`属性的需求可能只在某些平台和/或使用旧编译器时才会出现。

现在，这个对象库可以用来获取静态库（称为`message-static`）和共享库（称为`message-shared`）。需要注意的是，用于引用对象库的*生成器表达式语法*：`$<TARGET_OBJECTS:message-objs>`。生成器表达式是 CMake 在生成时（即配置时间之后）评估的构造，以产生特定于配置的构建输出。另请参阅：[`cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html)。我们将在第五章，*配置时间和构建时间操作*中深入探讨生成器表达式。最后，`hello-world`可执行文件与`message`库的静态版本链接。

是否可以让 CMake 生成两个同名的库？换句话说，它们是否可以都称为`message`而不是`message-static`和`message-shared`？我们需要修改这两个目标的属性：

```cpp
add_library(message-shared
  SHARED
    $<TARGET_OBJECTS:message-objs>
  )
set_target_properties(message-shared
  PROPERTIES
    OUTPUT_NAME "message"
  )

add_library(message-static
  STATIC
    $<TARGET_OBJECTS:message-objs>
  )
set_target_properties(message-static
  PROPERTIES
    OUTPUT_NAME "message"
  )
```

我们可以链接 DSO 吗？这取决于操作系统和编译器：

1.  在 GNU/Linux 和 macOS 上，无论选择哪个编译器，它都能正常工作。

1.  在 Windows 上，它无法与 Visual Studio 配合使用，但可以与 MinGW 和 MSYS2 配合使用。

为什么？生成好的 DSO 需要程序员限制*符号可见性*。这是通过编译器的帮助实现的，但在不同的操作系统和编译器上约定不同。CMake 有一个强大的机制来处理这个问题，我们将在第十章，*编写安装程序*中解释它是如何工作的。

# 使用条件控制编译

本节代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-04`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-04)找到，并包含一个 C++示例。本节适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

到目前为止，我们研究的项目相对简单，CMake 的执行流程是线性的：从一组源文件到一个单一的可执行文件，可能*通过*静态或共享库。为了确保对项目构建过程中所有步骤的执行流程有完全的控制，包括配置、编译和链接，CMake 提供了自己的语言。在本节中，我们将探讨使用条件结构`if-elseif-else-endif`。

CMake 语言相当庞大，包括基本控制结构、CMake 特定命令以及用于模块化扩展语言的新函数的基础设施。完整的概述可以在线找到：[`cmake.org/cmake/help/latest/manual/cmake-language.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)。

# 如何操作

让我们从与上一个配方相同的源代码开始。我们希望能够在这两种行为之间切换：

1.  将`Message.hpp`和`Message.cpp`编译成一个库，无论是静态还是共享，然后将生成的库链接到`hello-world`可执行文件中。

1.  将`Message.hpp`、`Message.cpp`和`hello-world.cpp`编译成一个单一的可执行文件，不生成库。

让我们构建`CMakeLists.txt`以实现这一点：

1.  我们首先定义最小 CMake 版本、项目名称和支持的语言：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-04 LANGUAGES CXX)
```

1.  我们引入了一个新变量，`USE_LIBRARY`。这是一个逻辑变量，其值将被设置为`OFF`。我们还打印其值供用户查看：

```cpp
set(USE_LIBRARY OFF)

message(STATUS "Compile sources into a library? ${USE_LIBRARY}")
```

1.  将 CMake 中定义的`BUILD_SHARED_LIBS`全局变量设置为`OFF`。调用`add_library`并省略第二个参数将构建一个静态库：

```cpp
set(BUILD_SHARED_LIBS OFF)
```

1.  然后，我们引入一个变量`_sources`，列出`Message.hpp`和`Message.cpp`：

```cpp
list(APPEND _sources Message.hpp Message.cpp)
```

1.  然后，我们根据`USE_LIBRARY`的值引入一个`if-else`语句。如果逻辑开关为真，`Message.hpp`和`Message.cpp`将被打包成一个库：

```cpp
if(USE_LIBRARY)
  # add_library will create a static library
  # since BUILD_SHARED_LIBS is OFF
  add_library(message ${_sources})

  add_executable(hello-world hello-world.cpp)

  target_link_libraries(hello-world message)
else()
  add_executable(hello-world hello-world.cpp ${_sources})
endif()
```

1.  我们可以再次使用相同的命令集进行构建。由于`USE_LIBRARY`设置为`OFF`，所有源文件将被编译成`hello-world`可执行文件。这可以通过在 GNU/Linux 上运行`objdump -x`命令来验证。

# 工作原理

我们引入了两个变量：`USE_LIBRARY`和`BUILD_SHARED_LIBS`。两者都设置为`OFF`。正如 CMake 语言文档中所详述的，真或假值可以用多种方式表达：

+   逻辑变量在以下情况下为真：设置为`1`、`ON`、`YES`、`TRUE`、`Y`或非零数字。

+   逻辑变量在以下情况下为假：设置为`0`、`OFF`、`NO`、`FALSE`、`N`、`IGNORE`、`NOTFOUND`、空字符串或以`-NOTFOUND`结尾。

`USE_LIBRARY`变量将在第一种和第二种行为之间切换。`BUILD_SHARED_LIBS`是 CMake 提供的一个全局标志。记住，`add_library`命令可以在不传递`STATIC`/`SHARED`/`OBJECT`参数的情况下调用。这是因为，内部会查找`BUILD_SHARED_LIBS`全局变量；如果为假或未定义，将生成一个静态库。

这个例子展示了在 CMake 中引入条件语句以控制执行流程是可能的。然而，当前的设置不允许从外部设置开关，也就是说，不通过手动修改`CMakeLists.txt`。原则上，我们希望将所有开关暴露给用户，以便在不修改构建系统代码的情况下调整配置。我们将在稍后展示如何做到这一点。

`else()`和`endif()`中的`()`可能会在你开始阅读和编写 CMake 代码时让你感到惊讶。这些的历史原因是能够指示作用域。例如，如果这有助于读者理解，可以使用`if(USE_LIBRARY) ... else(USE_LIBRARY) ... endif(USE_LIBRARY)`。这是一个品味问题。

在引入`_sources`变量时，我们向代码的读者表明这是一个不应在当前作用域外使用的局部变量，方法是将其前缀加上一个下划线。

# 向用户展示选项

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-05`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-05)找到，并包含一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在上一食谱中，我们以相当僵硬的方式引入了条件：通过引入具有硬编码真值的变量。有时这可能很有用，但它阻止了代码用户轻松切换这些变量。僵硬方法的另一个缺点是，CMake 代码没有向读者传达这是一个预期从外部修改的值。在项目构建系统生成中切换行为的推荐方法是使用`option()`命令在`CMakeLists.txt`中将逻辑开关作为选项呈现。本食谱将向您展示如何使用此命令。

# 如何操作

让我们回顾一下上一食谱中的静态/共享库示例。我们不再将`USE_LIBRARY`硬编码为`ON`或`OFF`，而是更倾向于将其作为具有默认值的选项公开，该默认值可以从外部更改：

1.  将上一食谱中的`set(USE_LIBRARY OFF)`命令替换为具有相同名称和默认值为`OFF`的选项。

```cpp
option(USE_LIBRARY "Compile sources into a library" OFF)
```

1.  现在，我们可以通过将信息传递给 CMake 的`-D` CLI 选项来切换库的生成：

```cpp
$ mkdir -p build
$ cd build
$ cmake -D USE_LIBRARY=ON ..

-- ...
-- Compile sources into a library? ON
-- ...

$ cmake --build .

Scanning dependencies of target message
[ 25%] Building CXX object CMakeFiles/message.dir/Message.cpp.o
[ 50%] Linking CXX static library libmessage.a
[ 50%] Built target message
Scanning dependencies of target hello-world
[ 75%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world
```

`-D`开关用于为 CMake 设置任何类型的变量：逻辑值、路径等。

# 工作原理

`option`命令接受三个参数：

```cpp
 option(<option_variable> "help string" [initial value])
```

+   `<option_variable>`是代表选项的变量名。

+   `"帮助字符串"`是记录选项的字符串。此文档在 CMake 的终端或图形用户界面中可见。

+   `[初始值]`是选项的默认值，可以是`ON`或`OFF`。

# 还有更多

有时需要引入依赖于其他选项值的选项。在我们的示例中，我们可能希望提供生成静态或共享库的选项。但是，如果`USE_LIBRARY`逻辑未设置为`ON`，则此选项将没有意义。CMake 提供了`cmake_dependent_option()`命令来定义依赖于其他选项的选项：

```cpp
include(CMakeDependentOption)

# second option depends on the value of the first
cmake_dependent_option(
  MAKE_STATIC_LIBRARY "Compile sources into a static library" OFF
  "USE_LIBRARY" ON
  )

# third option depends on the value of the first
cmake_dependent_option(
  MAKE_SHARED_LIBRARY "Compile sources into a shared library" ON
  "USE_LIBRARY" ON
  )
```

如果`USE_LIBRARY`设置为`ON`，则`MAKE_STATIC_LIBRARY`默认为`OFF`，而`MAKE_SHARED_LIBRARY`默认为`ON`。因此，我们可以运行以下命令：

```cpp
$ cmake -D USE_LIBRARY=OFF -D MAKE_SHARED_LIBRARY=ON ..
```

这仍然不会构建库，因为`USE_LIBRARY`仍然设置为`OFF`。

如前所述，CMake 通过包含*模块*来扩展其语法和功能，这些模块可以是 CMake 自带的，也可以是自定义的。在这种情况下，我们包含了一个名为`CMakeDependentOption`的模块。如果没有包含语句，`cmake_dependent_option()`命令将不可用。另请参阅[`cmake.org/cmake/help/latest/module/CMakeDependentOption.html`](https://cmake.org/cmake/help/latest/module/CMakeDependentOption.html)。

任何模块的手册页也可以使用`cmake --help-module <name-of-module>`从命令行访问。例如，`cmake --help-option CMakeDependentOption`将打印刚刚讨论的模块的手册页。

# 指定编译器

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-06`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-06)获取，并包含一个 C++/C 示例。该食谱适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

到目前为止，我们没有过多考虑的一个方面是编译器的选择。CMake 足够复杂，可以根据平台和生成器选择最合适的编译器。CMake 还能够将编译器标志设置为一组合理的默认值。然而，我们通常希望控制编译器的选择，在本食谱中，我们将展示如何做到这一点。在后面的食谱中，我们还将考虑构建类型的选择，并展示如何控制编译器标志。

# 如何操作

我们如何选择特定的编译器？例如，如果我们想使用 Intel 或 Portland Group 编译器怎么办？CMake 为每种语言的编译器存储在`CMAKE_<LANG>_COMPILER`变量中，其中`<LANG>`是任何受支持的语言，对我们来说，`CXX`、`C`或`Fortran`。用户可以通过以下两种方式之一设置此变量：

1.  通过在 CLI 中使用`-D`选项，例如：

```cpp
$ cmake -D CMAKE_CXX_COMPILER=clang++ ..
```

1.  通过导出环境变量`CXX`用于 C++编译器，`CC`用于 C 编译器，`FC`用于 Fortran 编译器。例如，使用此命令将 clang++作为 C++编译器：

```cpp
$ env CXX=clang++ cmake ..
```

到目前为止讨论的任何配方都可以通过传递适当的选项配置为与任何其他编译器一起使用。

CMake 了解环境，并且许多选项可以通过其 CLI 的`-D`开关*或*通过环境变量设置。前者机制覆盖后者，但我们建议始终使用`-D`显式设置选项。**显式优于隐式**，因为环境变量可能设置为不适合当前项目的值。

我们在这里假设额外的编译器在 CMake 进行查找的标准路径中可用。如果不是这种情况，用户需要传递编译器可执行文件或包装器的*完整路径*。

我们建议使用`-D CMAKE_<LANG>_COMPILER` CLI 选项设置编译器，而不是导出`CXX`，`CC`和`FC`。这是唯一保证跨平台兼容且与非 POSIX shell 兼容的方法。它还可以避免用可能影响与项目一起构建的外部库的环境的变量污染您的环境。

# 它是如何工作的

在配置时，CMake 执行一系列平台测试，以确定哪些编译器可用，以及它们是否适合手头的项目。合适的编译器不仅由我们工作的平台决定，还由我们要使用的生成器决定。CMake 执行的第一个测试基于项目语言的编译器名称。例如，如果`cc`是一个工作的 C 编译器，那么它将用作 C 项目的默认编译器。在 GNU / Linux 上，使用 Unix Makefiles 或 Ninja，GCC 家族的编译器将最有可能被默认选择用于 C ++，C 和 Fortran。在 Microsoft Windows 上，如果选择 Visual Studio 作为生成器，则将选择 Visual Studio 中的 C ++和 C 编译器。如果选择 MinGW 或 MSYS Makefiles 作为生成器，则默认使用 MinGW 编译器。

# 还有更多

我们可以在哪里找到 CMake 将为我们平台选择哪些默认编译器和编译器标志？CMake 提供了`--system-information`标志，该标志会将有关您系统的所有信息转储到屏幕或文件中。要查看此信息，请尝试以下操作：

```cpp
$ cmake --system-information information.txt
```

在文件（在本例中为`information.txt`）中搜索，您将找到`CMAKE_CXX_COMPILER`，`CMAKE_C_COMPILER`和`CMAKE_Fortran_COMPILER`选项的默认值，以及它们的默认标志。我们将在下一个配方中查看这些标志。

CMake 提供了其他变量来与编译器交互：

+   `CMAKE_<LANG>_COMPILER_LOADED`：如果为项目启用了语言`<LANG>`，则设置为`TRUE`。

+   `CMAKE_<LANG>_COMPILER_ID`：编译器识别字符串，对于编译器供应商是唯一的。例如，对于 GNU 编译器集合，这是`GCC`，对于 macOS 上的 Clang，这是`AppleClang`，对于 Microsoft Visual Studio 编译器，这是`MSVC`。但是请注意，不能保证此变量对所有编译器或语言都定义。

+   `CMAKE_COMPILER_IS_GNU<LANG>`：如果语言`<LANG>`的编译器是 GNU 编译器集合的一部分，则此逻辑变量设置为`TRUE`。请注意，变量名称的`<LANG>`部分遵循 GNU 约定：对于 C 语言，它将是`CC`，对于 C ++语言，它将是`CXX`，对于 Fortran 语言，它将是`G77`。

+   `CMAKE_<LANG>_COMPILER_VERSION`：此变量包含给定语言的编译器版本的字符串。版本信息以`major[.minor[.patch[.tweak]]]`格式给出。但是，与`CMAKE_<LANG>_COMPILER_ID`一样，不能保证此变量对所有编译器或语言都定义。

我们可以尝试使用不同的编译器配置以下示例`CMakeLists.txt`。在这个例子中，我们将使用 CMake 变量来探测我们正在使用的编译器及其版本：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES C CXX)

message(STATUS "Is the C++ compiler loaded? ${CMAKE_CXX_COMPILER_LOADED}")
if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "Is the C++ from GNU? ${CMAKE_COMPILER_IS_GNUCXX}")
  message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

message(STATUS "Is the C compiler loaded? ${CMAKE_C_COMPILER_LOADED}")
if(CMAKE_C_COMPILER_LOADED)
  message(STATUS "The C compiler ID is: ${CMAKE_C_COMPILER_ID}")
  message(STATUS "Is the C from GNU? ${CMAKE_COMPILER_IS_GNUCC}")
  message(STATUS "The C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
endif()
```

请注意，此示例不包含任何目标，因此没有要构建的内容，我们只关注配置步骤：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Is the C++ compiler loaded? 1
-- The C++ compiler ID is: GNU
-- Is the C++ from GNU? 1
-- The C++ compiler version is: 8.1.0
-- Is the C compiler loaded? 1
-- The C compiler ID is: GNU
-- Is the C from GNU? 1
-- The C compiler version is: 8.1.0
...
```

输出当然取决于可用和选择的编译器以及编译器版本。

# 切换构建类型

本配方的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-07`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-07)找到，并包含一个 C++/C 示例。该配方适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

CMake 具有构建类型或配置的概念，例如`Debug`、`Release`等。在一种配置中，可以收集相关选项或属性，例如编译器和链接器标志，用于`Debug`或`Release`构建。控制生成构建系统时使用的配置的变量是`CMAKE_BUILD_TYPE`。该变量默认情况下为空，CMake 识别的值包括：

1.  `Debug` 用于构建您的库或可执行文件，不带优化且带有调试符号，

1.  `Release` 用于构建您的库或可执行文件，带有优化且不带调试符号，

1.  `RelWithDebInfo` 用于构建您的库或可执行文件，具有较不激进的优化和调试符号，

1.  `MinSizeRel` 用于构建您的库或可执行文件，优化不会增加对象代码大小。

# 如何操作

在本配方中，我们将展示如何为示例项目设置构建类型：

1.  我们首先定义了最小 CMake 版本、项目名称和支持的语言：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-07 LANGUAGES C CXX)
```

1.  然后，我们设置了一个默认构建类型（在这种情况下，`Release`），并将其打印在消息中供用户查看。请注意，该变量被设置为`CACHE`变量，以便随后可以通过缓存进行编辑：

```cpp
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
```

1.  最后，我们打印出由 CMake 根据构建类型设置的相应编译标志：

```cpp
message(STATUS "C flags, Debug configuration: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C flags, Release configuration: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "C flags, Release configuration with Debug info: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message(STATUS "C flags, minimal Release configuration: ${CMAKE_C_FLAGS_MINSIZEREL}")

message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
```

1.  现在让我们验证默认配置的输出：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Build type: Release
-- C flags, Debug configuration: -g
-- C flags, Release configuration: -O3 -DNDEBUG
-- C flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C flags, minimal Release configuration: -Os -DNDEBUG
-- C++ flags, Debug configuration: -g
-- C++ flags, Release configuration: -O3 -DNDEBUG
-- C++ flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C++ flags, minimal Release configuration: -Os -DNDEBUG
```

1.  现在，让我们切换构建类型：

```cpp
$ cmake -D CMAKE_BUILD_TYPE=Debug ..

-- Build type: Debug
-- C flags, Debug configuration: -g
-- C flags, Release configuration: -O3 -DNDEBUG
-- C flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C flags, minimal Release configuration: -Os -DNDEBUG
-- C++ flags, Debug configuration: -g
-- C++ flags, Release configuration: -O3 -DNDEBUG
-- C++ flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C++ flags, minimal Release configuration: -Os -DNDEBUG
```

# 它是如何工作的

我们已经演示了如何设置默认构建类型以及如何从命令行覆盖它。通过这种方式，我们可以控制项目是使用优化标志构建，还是关闭所有优化并启用调试信息。我们还看到了不同可用配置使用的标志类型，这取决于所选的编译器。除了在 CMake 运行期间明确打印标志外，还可以通过运行`cmake --system-information`来查看当前平台、默认编译器和语言组合的预设。在下一个配方中，我们将讨论如何为不同的编译器和不同的构建类型扩展或调整编译器标志。

# 还有更多

我们已经展示了`CMAKE_BUILD_TYPE`变量（文档链接：[`cmake.org/cmake/help/v3.5/variable/CMAKE_BUILD_TYPE.html`](https://cmake.org/cmake/help/v3.5/variable/CMAKE_BUILD_TYPE.html)）如何定义生成的构建系统的配置。在评估编译器优化级别的影响时，例如，构建项目的`Release`和`Debug`配置通常很有帮助。对于*单配置*生成器，如 Unix Makefiles、MSYS Makefiles 或 Ninja，这需要运行 CMake 两次，即对项目进行完全重新配置。然而，CMake 还支持*多配置*生成器。这些通常是由集成开发环境提供的项目文件，最著名的是 Visual Studio 和 Xcode，它们可以同时处理多个配置。这些生成器的可用配置类型可以通过`CMAKE_CONFIGURATION_TYPES`变量进行调整，该变量将接受一个值列表（文档链接：[`cmake.org/cmake/help/v3.5/variable/CMAKE_CONFIGURATION_TYPES.html`](https://cmake.org/cmake/help/v3.5/variable/CMAKE_CONFIGURATION_TYPES.html)）。

以下是使用 Visual Studio 的 CMake 调用：

```cpp
$ mkdir -p build
$ cd build
$ cmake .. -G"Visual Studio 12 2017 Win64" -D CMAKE_CONFIGURATION_TYPES="Release;Debug"
```

将生成`Release`和`Debug`配置的构建树。然后，您可以使用`--config`标志决定构建哪一个：

```cpp
$ cmake --build . --config Release
```

当使用单配置生成器开发代码时，为`Release`和`Debug`构建类型创建单独的构建目录，两者都配置相同的源代码。这样，您可以在两者之间切换，而不会触发完全重新配置和重新编译。

# 控制编译器标志

本示例的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08)找到，并包含一个 C++示例。本示例适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

之前的示例展示了如何向 CMake 查询有关编译器的信息，以及如何调整项目中所有目标的编译器优化。后一项任务是控制项目中使用哪些编译器标志的一般需求的一个子集。CMake 提供了调整或扩展编译器标志的很大灵活性，您可以选择两种主要方法之一：

+   CMake 将编译选项视为目标的属性。因此，可以在不覆盖 CMake 默认设置的情况下，为每个目标设置编译选项。

+   通过使用`-D` CLI 开关，您可以直接修改`CMAKE_<LANG>_FLAGS_<CONFIG>`变量。这些变量将影响项目中的所有目标，并覆盖或扩展 CMake 的默认设置。

在本示例中，我们将展示这两种方法。

# 准备工作

我们将编译一个计算不同几何形状面积的示例程序。代码在名为`compute-areas.cpp`的文件中有一个`main`函数：

```cpp
#include "geometry_circle.hpp"
#include "geometry_polygon.hpp"
#include "geometry_rhombus.hpp"
#include "geometry_square.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  using namespace geometry;

  double radius = 2.5293;
  double A_circle = area::circle(radius);
  std::cout << "A circle of radius " << radius << " has an area of " << A_circle
            << std::endl;

  int nSides = 19;
  double side = 1.29312;
  double A_polygon = area::polygon(nSides, side);
  std::cout << "A regular polygon of " << nSides << " sides of length " << side
            << " has an area of " << A_polygon << std::endl;

  double d1 = 5.0;
  double d2 = 7.8912;
  double A_rhombus = area::rhombus(d1, d2);
  std::cout << "A rhombus of major diagonal " << d1 << " and minor diagonal " << d2
            << " has an area of " << A_rhombus << std::endl;

  double l = 10.0;
  double A_square = area::square(l);
  std::cout << "A square of side " << l << " has an area of " << A_square
            << std::endl;

  return EXIT_SUCCESS;
}
```

各种函数的实现包含在其他文件中：每个几何形状都有一个头文件和一个对应的源文件。总共，我们有四个头文件和五个源文件需要编译：

```cpp
.
├── CMakeLists.txt
├── compute-areas.cpp
├── geometry_circle.cpp
├── geometry_circle.hpp
├── geometry_polygon.cpp
├── geometry_polygon.hpp
├── geometry_rhombus.cpp
├── geometry_rhombus.hpp
├── geometry_square.cpp
└── geometry_square.hpp
```

我们不会为所有这些文件提供列表，而是引导读者参考[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08)。

# 如何操作

现在我们有了源文件，我们的目标将是配置项目并尝试使用编译器标志：

1.  我们设置 CMake 的最低要求版本：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

1.  我们声明项目的名称和语言：

```cpp
project(recipe-08 LANGUAGES CXX)
```

1.  然后，我们打印当前的编译器标志集。CMake 将使用这些标志来编译所有 C++目标：

```cpp
message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
```

1.  我们为我们的目标准备了一份标志列表。其中一些在 Windows 上可能不可用，我们确保考虑到这种情况：

```cpp
list(APPEND flags "-fPIC" "-Wall")
if(NOT WIN32)
  list(APPEND flags "-Wextra" "-Wpedantic")
endif()
```

1.  我们添加一个新的目标，`geometry`库及其源依赖项：

```cpp
add_library(geometry
  STATIC
    geometry_circle.cpp
    geometry_circle.hpp
    geometry_polygon.cpp
    geometry_polygon.hpp
    geometry_rhombus.cpp
    geometry_rhombus.hpp
    geometry_square.cpp
    geometry_square.hpp
  )
```

1.  我们为这个库目标设置编译选项：

```cpp
target_compile_options(geometry
  PRIVATE
    ${flags}
  )
```

1.  然后，我们为`compute-areas`可执行文件添加一个目标：

```cpp
add_executable(compute-areas compute-areas.cpp)
```

1.  我们还为可执行目标设置编译选项：

```cpp
target_compile_options(compute-areas
  PRIVATE
    "-fPIC"
  )
```

1.  最后，我们将可执行文件链接到`geometry`库：

```cpp
target_link_libraries(compute-areas geometry)
```

# 它是如何工作的

在这个例子中，警告标志`-Wall`、`-Wextra`和`-Wpedantic`将被添加到`geometry`目标的编译选项中；`compute-areas`和`geometry`目标都将使用`-fPIC`标志。编译选项可以通过三种可见性级别添加：`INTERFACE`、`PUBLIC`和`PRIVATE`。

可见性级别具有以下含义：

+   使用`PRIVATE`属性，编译选项将仅应用于给定目标，而不会应用于其他消费它的目标。在我们的示例中，设置在`geometry`目标上的编译器选项不会被`compute-areas`继承，尽管`compute-areas`会链接到`geometry`库。

+   使用`INTERFACE`属性，给定目标的编译选项将仅应用于消费它的目标。

+   使用`PUBLIC`属性，编译选项将应用于给定目标以及所有其他消费它的目标。

目标属性的可见性级别是现代 CMake 使用的核心，我们将在本书中经常并广泛地回顾这个主题。以这种方式添加编译选项不会污染`CMAKE_<LANG>_FLAGS_<CONFIG>`全局 CMake 变量，并给你对哪些选项用于哪些目标的精细控制。

我们如何验证标志是否如我们所愿正确使用？换句话说，你如何发现一个 CMake 项目实际上使用了哪些编译标志？一种方法是使用 CMake 传递额外的参数，在这种情况下是环境变量`VERBOSE=1`，给本地构建工具：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build . -- VERBOSE=1

... lots of output ...

[ 14%] Building CXX object CMakeFiles/geometry.dir/geometry_circle.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_circle.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_circle.cpp
[ 28%] Building CXX object CMakeFiles/geometry.dir/geometry_polygon.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_polygon.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_polygon.cpp
[ 42%] Building CXX object CMakeFiles/geometry.dir/geometry_rhombus.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_rhombus.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_rhombus.cpp
[ 57%] Building CXX object CMakeFiles/geometry.dir/geometry_square.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_square.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_square.cpp

... more output ...

[ 85%] Building CXX object CMakeFiles/compute-areas.dir/compute-areas.cpp.o
/usr/bin/c++ -fPIC -o CMakeFiles/compute-areas.dir/compute-areas.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/compute-areas.cpp

... more output ...
```

前面的输出确认编译标志根据我们的指示正确设置。

控制编译器标志的第二种方法不涉及对`CMakeLists.txt`的任何修改。如果想要为该项目中的`geometry`和`compute-areas`目标修改编译器选项，只需使用一个额外的参数调用 CMake 即可。

```cpp
$ cmake -D CMAKE_CXX_FLAGS="-fno-exceptions -fno-rtti" ..
```

正如你可能已经猜到的，这个命令将编译项目，禁用异常和运行时类型识别（RTTI）。

这两种方法也可以结合使用。可以使用一组基本的标志全局设置，同时保持对每个目标发生的情况的控制。我们可以使用`CMakeLists.txt`并运行这个命令：

```cpp
$ cmake -D CMAKE_CXX_FLAGS="-fno-exceptions -fno-rtti" ..
```

这将使用`-fno-exceptions -fno-rtti -fPIC -Wall -Wextra -Wpedantic`配置`geometry`目标，同时使用`-fno-exceptions -fno-rtti -fPIC`配置`compute-areas`。

在本书的其余部分，我们通常会为每个目标设置编译器标志，这是我们推荐您项目采用的做法。使用 `target_compile_options()` 不仅允许对编译选项进行细粒度控制，而且还更好地与 CMake 的更高级功能集成。

# 还有更多

大多数情况下，标志是编译器特定的。我们当前的示例仅适用于 GCC 和 Clang；其他供应商的编译器将不理解许多，如果不是全部，这些标志。显然，如果一个项目旨在真正跨平台，这个问题必须解决。有三种方法可以解决这个问题。

最典型的方法是将所需的一组编译器标志附加到每个配置类型的 CMake 变量，即 `CMAKE_<LANG>_FLAGS_<CONFIG>`。这些标志设置为已知适用于给定编译器供应商的内容，因此将包含在

`if-endif` 子句检查 `CMAKE_<LANG>_COMPILER_ID` 变量，例如：

```cpp
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND CMAKE_CXX_FLAGS "-fno-rtti" "-fno-exceptions")
  list(APPEND CMAKE_CXX_FLAGS_DEBUG "-Wsuggest-final-types" "-Wsuggest-final-methods" "-Wsuggest-override")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)  
  list(APPEND CMAKE_CXX_FLAGS "-fno-rtti" "-fno-exceptions" "-Qunused-arguments" "-fcolor-diagnostics")
  list(APPEND CMAKE_CXX_FLAGS_DEBUG "-Wdocumentation")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()
```

一种更精细的方法根本不修改 `CMAKE_<LANG>_FLAGS_<CONFIG>` 变量，而是定义项目特定的标志列表：

```cpp
set(COMPILER_FLAGS)
set(COMPILER_FLAGS_DEBUG)
set(COMPILER_FLAGS_RELEASE)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND CXX_FLAGS "-fno-rtti" "-fno-exceptions")
  list(APPEND CXX_FLAGS_DEBUG "-Wsuggest-final-types" "-Wsuggest-final-methods" "-Wsuggest-override")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)  
  list(APPEND CXX_FLAGS "-fno-rtti" "-fno-exceptions" "-Qunused-arguments" "-fcolor-diagnostics")
  list(APPEND CXX_FLAGS_DEBUG "-Wdocumentation")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()
```

稍后，它使用生成器表达式以每个配置和每个目标为基础设置编译器标志：

```cpp
target_compile_option(compute-areas
  PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
  )
```

我们在当前的配方中展示了这两种方法，并明确推荐后者（项目特定变量和 `target_compile_options()`）而不是前者（CMake 变量）。

这两种方法都有效，并在许多项目中广泛使用。然而，它们也有缺点。正如我们已经提到的，`CMAKE_<LANG>_COMPILER_ID`并不保证为所有编译器供应商定义。此外，某些标志可能会被弃用，或者可能在编译器的较新版本中引入。与`CMAKE_<LANG>_COMPILER_ID`类似，`CMAKE_<LANG>_COMPILER_VERSION`变量并不保证为所有语言和供应商定义。尽管检查这些变量非常流行，但我们认为更稳健的替代方案是检查给定编译器是否支持所需的标志集，以便仅在项目中实际使用有效的标志。结合使用项目特定变量、`target_compile_options`和生成器表达式，这种方法非常强大。我们将在第 3 个示例中展示如何使用这种检查和设置模式，即第七章中的“编写一个函数来测试和设置编译器标志”。

# 设置语言标准

本示例代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09)获取，包含 C++和 Fortran 示例。本示例适用于 CMake 版本 3.5（及更高版本），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

编程语言有不同的标准可供选择，即提供新改进语言结构的不同版本。启用新标准是通过设置适当的编译器标志来实现的。我们在前面的示例中展示了如何做到这一点，无论是针对特定目标还是全局设置。CMake 3.1 版本引入了针对 C++和 C 语言标准的平台和编译器无关机制：为目标设置`<LANG>_STANDARD`属性。

# 准备工作

对于以下示例，我们将要求 C++编译器符合 C++14 标准或更高版本。本示例代码定义了一个动物的多态层次结构。我们在层次结构的基类中使用`std::unique_ptr`：

```cpp
std::unique_ptr<Animal> cat = Cat("Simon");
std::unique_ptr<Animal> dog = Dog("Marlowe);
```

我们没有明确使用各种子类型的构造函数，而是使用工厂方法的实现。工厂使用 C++11 的*可变参数模板*实现。它保存了继承层次结构中每个对象的创建函数映射：

```cpp
typedef std::function<std::unique_ptr<Animal>(const std::string &)> CreateAnimal;
```

它根据预先分配的标签进行分派，以便对象的创建将如下所示：

```cpp
std::unique_ptr<Animal> simon = farm.create("CAT", "Simon");
std::unique_ptr<Animal> marlowe = farm.create("DOG", "Marlowe");
```

在工厂使用之前，将标签和创建函数注册到工厂：

```cpp
Factory<CreateAnimal> farm;
farm.subscribe("CAT", [](const std::string & n) { return std::make_unique<Cat>(n); });
farm.subscribe("DOG", [](const std::string & n) { return std::make_unique<Dog>(n); });
```

我们使用 C++11 的*lambda*函数定义创建函数。注意使用`std::make_unique`来避免引入裸`new`操作符。这个辅助函数是在 C++14 中引入的。

此 CMake 功能是在版本 3.1 中添加的，并且一直在不断发展。CMake 的后续版本为 C++标准的后续版本和不同的编译器提供了越来越好的支持。我们建议您检查您的首选编译器是否受支持，请访问文档网页：[`cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers`](https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compilers)。

# 如何做到这一点

我们将逐步构建 `CMakeLists.txt` 并展示如何要求特定的标准（在本例中为 C++14）：

1.  我们声明了所需的最低 CMake 版本、项目名称和语言：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-09 LANGUAGES CXX)
```

1.  我们要求在 Windows 上导出所有库符号：

```cpp
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
```

1.  我们需要为库添加一个目标。这将编译源代码成一个共享库：

```cpp
add_library(animals
  SHARED
    Animal.cpp
    Animal.hpp
    Cat.cpp
    Cat.hpp
    Dog.cpp
    Dog.hpp
    Factory.hpp
  )
```

1.  现在我们为目标设置 `CXX_STANDARD`、`CXX_EXTENSIONS` 和 `CXX_STANDARD_REQUIRED` 属性。我们还设置了 `POSITION_INDEPENDENT_CODE` 属性，以避免在某些编译器上构建 DSO 时出现问题：

```cpp
set_target_properties(animals
  PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE 1
  )
```

1.  然后，我们为 `animal-farm` 可执行文件添加一个新的目标并设置其属性：

```cpp
add_executable(animal-farm animal-farm.cpp)

set_target_properties(animal-farm
  PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
  )
```

1.  最后，我们将可执行文件链接到库：

```cpp
target_link_libraries(animal-farm animals)
```

1.  让我们也检查一下我们的例子中的猫和狗有什么要说的：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./animal-farm

I'm Simon the cat!
I'm Marlowe the dog!
```

# 它是如何工作的

在步骤 4 和 5 中，我们为 `animals` 和 `animal-farm` 目标设置了一系列属性：

+   `CXX_STANDARD` 规定了我们希望采用的标准。

+   `CXX_EXTENSIONS` 告诉 CMake 只使用将启用 ISO C++标准的编译器标志，而不使用编译器特定的扩展。

+   `CXX_STANDARD_REQUIRED` 指定所选标准版本是必需的。如果该版本不可用，CMake 将以错误停止配置。当此属性设置为 `OFF` 时，CMake 将查找下一个最新的标准版本，直到设置了适当的标志。这意味着首先查找 C++14，然后是 C++11，然后是 C++98。

在撰写本文时，还没有 `Fortran_STANDARD` 属性可用，但可以使用 `target_compile_options` 设置标准；请参阅 [`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09)。

如果语言标准是所有目标共享的全局属性，您可以将 `CMAKE_<LANG>_STANDARD`、`CMAKE_<LANG>_EXTENSIONS` 和 `CMAKE_<LANG>_STANDARD_REQUIRED` 变量设置为所需值。所有目标上的相应属性将使用这些值进行设置。

# 还有更多

CMake 通过引入*编译特性*的概念，提供了对语言标准的更精细控制。这些特性是由语言标准引入的，例如 C++11 中的可变参数模板和 lambda，以及 C++14 中的自动返回类型推导。您可以通过`target_compile_features()`命令要求特定目标支持某些特性，CMake 会自动为该标准设置正确的编译器标志。CMake 还可以为可选的编译器特性生成兼容性头文件。

我们建议阅读`cmake-compile-features`的在线文档，以全面了解 CMake 如何处理编译特性和语言标准：[`cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html)。

# 使用控制流结构

本食谱的代码可在[`github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-10`](https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-10)找到，并附有一个 C++示例。该食谱适用于 CMake 版本 3.5（及以上），并在 GNU/Linux、macOS 和 Windows 上进行了测试。

在本章之前的食谱中，我们已经使用了`if-elseif-endif`结构。CMake 还提供了创建循环的语言设施：`foreach-endforeach`和`while-endwhile`。两者都可以与`break`结合使用，以提前从封闭循环中跳出。本食谱将向您展示如何使用`foreach`遍历源文件列表。我们将对一组源文件应用这样的循环，以降低编译器优化，而不引入新的目标。

# 准备就绪

我们将重用本章第 8 个食谱中引入的`geometry`示例，*控制编译器标志*。我们的目标是通过将它们收集到一个列表中，对一些源文件的编译器优化进行微调。

# 如何操作

以下是在`CMakeLists.txt`中需要遵循的详细步骤：

1.  与第 8 个食谱，*控制编译器标志*一样，我们指定了所需的最低 CMake 版本、项目名称和语言，并声明了`geometry`库目标：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-10 LANGUAGES CXX)

add_library(geometry
  STATIC
    geometry_circle.cpp
    geometry_circle.hpp
    geometry_polygon.cpp
    geometry_polygon.hpp
    geometry_rhombus.cpp
    geometry_rhombus.hpp
    geometry_square.cpp
    geometry_square.hpp
  )
```

1.  我们决定以`-O3`编译器优化级别编译库。这作为目标的`PRIVATE`编译选项设置：

```cpp
target_compile_options(geometry
  PRIVATE
    -O3
  )
```

1.  然后，我们生成一份需要以较低优化级别编译的源文件列表：

```cpp
list(
  APPEND sources_with_lower_optimization
    geometry_circle.cpp
    geometry_rhombus.cpp
  )
```

1.  我们遍历这些源文件，将它们的优化级别调整至`-O2`。这是通过使用它们的源文件属性来完成的：

```cpp
message(STATUS "Setting source properties using IN LISTS syntax:")
foreach(_source IN LISTS sources_with_lower_optimization)
  set_source_files_properties(${_source} PROPERTIES COMPILE_FLAGS -O2)
  message(STATUS "Appending -O2 flag for ${_source}")
endforeach()
```

1.  为了确保设置了源属性，我们再次遍历并打印每个源的`COMPILE_FLAGS`属性：

```cpp
message(STATUS "Querying sources properties using plain syntax:")
foreach(_source ${sources_with_lower_optimization})
  get_source_file_property(_flags ${_source} COMPILE_FLAGS)
  message(STATUS "Source ${_source} has the following extra COMPILE_FLAGS: ${_flags}")
endforeach()
```

1.  最后，我们添加了`compute-areas`可执行目标，并将其与`geometry`库链接：

```cpp
add_executable(compute-areas compute-areas.cpp)

target_link_libraries(compute-areas geometry)
```

1.  让我们验证在配置步骤中标志是否正确设置：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Setting source properties using IN LISTS syntax:
-- Appending -O2 flag for geometry_circle.cpp
-- Appending -O2 flag for geometry_rhombus.cpp
-- Querying sources properties using plain syntax:
-- Source geometry_circle.cpp has the following extra COMPILE_FLAGS: -O2
-- Source geometry_rhombus.cpp has the following extra COMPILE_FLAGS: -O2
```

1.  最后，使用`VERBOSE=1`检查构建步骤。您将看到`-O2`标志被附加到`-O3`标志上，但最终的优化级别标志（在这种情况下是`-O2`）“胜出”：

```cpp
$ cmake --build . -- VERBOSE=1
```

# 它是如何工作的

`foreach-endforeach`语法可以用来表达对一组变量的重复任务。在我们的例子中，我们使用它来操作、设置和获取项目中特定文件的编译器标志。这个 CMake 代码片段引入了两个额外的新的命令：

+   `set_source_files_properties(file PROPERTIES property value)`，它为给定文件设置属性到传递的值。与目标类似，文件在 CMake 中也有属性。这允许对构建系统生成进行极其精细的控制。源文件可用属性的列表可以在这里找到：[`cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#source-file-properties`](https://cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#source-file-properties)。

+   `get_source_file_property(VAR file property)`，它检索给定文件的所需属性的值，并将其存储在 CMake 的`VAR`变量中。

在 CMake 中，列表是由分号分隔的字符串组。列表可以通过`list`命令或`set`命令创建。例如，`set(var a b c d e)`和`list(APPEND a b c d e)`都创建了列表`a;b;c;d;e`。

为了降低一组文件的优化级别，将它们收集到一个单独的目标（库）中，并为该目标显式设置优化级别，而不是附加一个标志，这可能更清晰。但在本例中，我们的重点是`foreach-endforeach`。

# 还有更多

`foreach()`构造可以以四种不同的方式使用：

+   `foreach(loop_var arg1 arg2 ...)`：提供了一个循环变量和一个明确的项列表。当打印`sources_with_lower_optimization`中项的编译器标志集时，使用了这种形式。请注意，如果项列表在一个变量中，它必须被显式展开；也就是说，必须将`${sources_with_lower_optimization}`作为参数传递。

+   作为对整数的循环，通过指定一个范围，例如`foreach(loop_var RANGE total)`，或者替代地

    `foreach(loop_var RANGE start stop [step])`。

+   作为对列表值变量的循环，例如`foreach(loop_var IN LISTS [list1 [...]])`。参数被解释为列表，并且它们的内含物会自动相应地展开。

+   作为对项的循环，例如`foreach(loop_var IN ITEMS [item1 [...]])`。参数的内容不会展开。
