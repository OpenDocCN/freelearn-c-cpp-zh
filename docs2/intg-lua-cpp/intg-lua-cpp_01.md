

# 使您的 C++ 项目 Lua 准备就绪

在本书的整个过程中，您将学习如何将 Lua 集成到您的 C++ 项目中。每一章都将基于前一章学到的知识。本章将教您如何准备一个 C++ 项目以集成 Lua，并介绍本书中使用的工具，以便您更好地理解示例。如果您已经知道如何使用一些工具，请随意跳过这些部分。如果您不知道，请在阅读本章后进行更深入的学习。

在本章中，我们将涵盖以下主题：

+   编译 Lua 源代码

+   使用 Lua 库构建 C++ 项目

+   使用 Lua 源代码构建 C++ 项目

+   执行简单的 Lua 脚本

+   其他工具链选项

# 技术要求

要跟随本章和本书，您将需要以下内容：

+   一个可工作的 C++ 编译器，最好是 **GNU C++ 编译器** 或 **Clang/LLVM**

+   一个构建自动化工具，最好是 **Make**

+   您选择的代码编辑器

+   Lua 源代码

+   本章的源代码：[`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter01`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter01)

您不需要先前的 Lua 编程知识来理解本章内容。如果您对本章中 Lua 代码示例有任何疑问，那没关系；您可以将其视为 C++ 代码阅读，尽管存在语法差异。您将在阅读本书的过程中学习 Lua。虽然成为 Lua 专家会很有益，但如果您的重点是 C++ 方面，您不需要成为 Lua 专家。

我们决定使用开源编译器和构建工具来处理本书中的代码示例，因为它们对每个人来说都很容易获得，并且也是大多数大型项目中的首选工具。

如果你正在使用 Linux 或 Mac 开发机器，GNU C++ 编译器（或 Clang/LLVM）和 Make 应该已经安装。如果没有，请安装您系统支持的版本。如果您是 Windows 用户，您可以首先查看本章的最后部分：*其他* *工具链选项*。

使用的构建工具称为 Make。在实际项目中，您可能会使用其他构建工具。但 Make 是一个没有其他依赖的基本选项，其他构建工具具有类似的思想，这使得它适合本书的目的。如果您愿意，您可以调整本书中的示例以适应您选择的另一个构建工具。

您可以从 [`www.lua.org/download.xhtml`](https://www.lua.org/download.xhtml) 下载 Lua 源代码。您可以选择特定的版本，但您很可能会希望使用最新版本。您也可以从 Lua 的 Git 仓库克隆源代码，这里：[`www.github.com/lua/lua`](https://www.github.com/lua/lua)。然而，这并不是官方推荐的，因为 Lua 稳定且紧凑，变化不频繁。

# 编译 Lua 源代码

访问 Lua 语言的方式有很多。如果你使用 Linux，你可以通过发行版的包管理器安装 Lua 进行开发。对于 Windows，你也可以找到预构建的二进制文件。然而，由于我们的目标是把 Lua 集成到你的 C++ 项目中，而不是将其作为独立的解释器使用，最好是自己从源代码构建。在学习 Lua 的过程中，这将帮助你更好地了解 Lua。例如，在现代代码编辑器中，包括 *Visual Studio Code*，你可以轻松地检查 Lua 库函数的声明和实现。

在本节中，我们将专注于从源代码编译 Lua。解压缩下载的 Lua 源代码存档。大多数压缩工具都支持这一点，Lua 下载网站也提供了说明。当你完成这个步骤后，你会看到一个简单的文件夹结构：

```cpp
lua-5.4.6 % ls
Makefile README doc src
```

在下一节中，我们将学习前面的代码块做了什么。

Lua 源代码包具有典型的 **POSIX**（想想 Linux 和 Unix）文件夹结构。

+   `Makefile` 是包的根 `Makefile`

+   `src` 子文件夹包含源代码

+   `doc` 子文件夹包含文档

## 介绍 shell

在本节中，我们将学习 POSIX 机器上的 `zsh`（`Z` shell）。另一个流行的 shell 是 `bash`，使用它你也可以直接运行本书中的示例。即使你使用 **集成开发环境**（**IDE**）并手动将示例适配到你的 IDE，了解 shell 命令的基本知识也能帮助你更好地理解示例。所有 IDE 内部都使用各种命令行程序来完成它们的工作，这与我们在 shell 中将要做的类似。

简单来说，shell 为系统提供了一个命令行界面。当你交互式地访问 shell 时，也可以说你在访问一个终端。shell 和终端这两个词有时可以互换使用，尽管在技术上它们是不同的事物。幸运的是，我们在这里不需要担心术语上的差异。所有具有图形用户界面的系统也会提供一个应用程序来启动 shell。通常，这些应用程序被称为 **终端** 或 **控制台**。

你可以启动一个 shell 并尝试以下命令。要找出你正在使用的 shell，请使用这个命令：

```cpp
% echo $SHELL
/bin/zsh
```

输出 `/bin/zsh` 表示正在使用的 shell 是 `Z` shell。

要进入一个目录，请使用以下命令：

```cpp
cd ~/Downloads/lua-5.4.6
lua-5.4.6 %
```

`cd` 是用于更改当前工作目录的命令。这会进入 Lua 源代码文件夹。

而且，正如你之前看到的，`ls` 是列出目录内容的命令：

```cpp
lua-5.4.6 % ls
Makefile README doc src
```

另一个重要的事情是 `%` 符号。它表示 shell 提示符，不同的 shell 或用户角色可能会看到不同的符号。`%` 前面的部分是当前工作目录，`%` 后面的部分是你将在终端中输入的命令。

本节仅提供一个简要说明。如果你遇到你不知道的 shell 命令，可以在网上查找。

## 构建 Lua

在 shell 终端中，进入未解压的 Lua 源代码文件夹，并执行`make all test`。如果你的工具链被检测为正常工作，你将已编译 Lua 库和命令行解释器。现在，让我们检查感兴趣的文件：

```cpp
lua-5.4.6 % ls src/*.a src/lua
src/liblua.a src/lua
```

`liblua.a`是你可以链接的 Lua 库。`lua`是 Lua 命令行解释器。

让我们现在尝试解释器，看看我们是否可以成功运行它：

```cpp
lua-5.4.6 % src/lua
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> 1+1
2
> os.exit()
```

在终端中，执行`src/lua`以启动交互式 Lua 解释器。首先，输入`1+1`：Lua 将返回结果`2`。然后输入`os.exit()`以退出 Lua 解释器。

你现在已成功从源代码编译了 Lua 库。接下来，我们将看看如何在你的项目中使用它。

# 使用 Lua 库构建 C++项目

使用 Lua 库构建你的 C++项目的好处是不必在你的项目和源代码管理系统中包含 100 多个 Lua 源文件。然而，它也有一些缺点。例如，如果你的项目需要支持多个平台，你需要维护多个预编译的库。在这种情况下，从 Lua 源代码构建可能更容易。

## 创建一个与 Lua 库一起工作的项目

在上一节中，我们从源代码构建了 Lua 库。现在，让我们将其提取出来以在我们的项目中使用。

记得源代码文件夹根目录下的`Makefile`吗？打开它，你会看到这里显示的两行：

```cpp
TO_INC= lua.h luaconf.h lualib.h lauxlib.h lua.hpp
TO_LIB= liblua.a
```

这些是你需要的头文件和静态库。

为你的项目创建一个文件夹。在其内部，创建一个名为`main.cpp`的空源文件、一个空的`Makefile`以及两个名为`include`和`lib`的空文件夹。将头文件复制到`include`文件夹，将库文件复制到`lib`文件夹。项目文件夹应看起来像这样：

```cpp
project-with-lua-lib % tree
.
├── Makefile
├── lua
│   ├── include
│   │   ├── lauxlib.h
│   │   ├── lua.h
│   │   ├── lua.hpp
│   │   ├── luaconf.h
│   │   └── lualib.h
│   └── lib
│       └── liblua.a
└── main.cpp
```

如果你使用 Linux，`tree`是一个用于打印文件夹层次结构的 shell 程序。如果你没有安装`tree`，无需担心。你也可以检查你喜欢的 IDE 中的文件夹结构。

## 编写 C++代码

我们将编写一个简单的 C++程序来测试我们是否可以链接到 Lua 库。在 C++中，你只需要包含一个 Lua 的头文件：`lua.hpp`。

按照以下内容编写`main.cpp`：

```cpp
#include <iostream>
#include <lua.hpp>
int main()
{
    lua_State *L = luaL_newstate();
    std::cout << "Lua version number is "
              << lua_version(L)
              << std::endl;
    lua_close(L);
    return 0;
}
```

上述源代码打开 Lua，打印其构建号，然后关闭 Lua。

## 编写 Makefile

作为第一个项目的部分，编写`Makefile`非常简单。让我们利用这个机会来学习更多关于`Makefile`的细节，如果你还没有很好地理解它的话。更多信息，你可以查看官方网站[`www.gnu.org/software/make/`](https://www.gnu.org/software/make/).

按照以下代码编写`Makefile`：

```cpp
project-with-lua-lib: main.cpp
    g++ -o project-with-lua-lib main.cpp -Ilua/include \
        -Llua/lib -llua
```

这是一个非常基础的 `Makefile`。在实际项目中，你需要一个更复杂的 `Makefile` 来使其更加灵活。你将在本章后面看到更灵活的示例。在这里，重点是第一次接触时的简单性。这个初始的 `Makefile` 包含以下元素：

+   `project-with-lua-lib` 是一个 `Makefile` 目标。你可以在一个 `Makefile` 中定义任意数量的目标。当你不指定明确的目标而调用 `make` 时，它将执行文件中定义的第一个目标。

+   `main.cpp` 是目标的依赖项。你可以依赖于另一个目标或文件。你可以根据需要添加任意数量的依赖项。

+   目标调用 `g++` 命令来编译 `main.cpp` 并将其与 Lua 库链接。在目标命令之前，你需要使用制表符而不是空格。

+   `-o project-with-lua-lib` 指定了编译后的可执行文件名称。你可以将其更改为任何你想要的名称。

+   `-Ilua/include` 将 `lua/include` 添加到头文件的搜索路径。

+   `-Llua/lib` 将 `lua/lib` 添加到库的链接搜索路径。

+   `-llua` 告诉链接器链接到静态 Lua 库：`liblua.a`。

## 测试项目

在终端中，执行 `make` 来构建项目。然后执行 `./project-with-lua-lib` 来运行编译后的项目：

```cpp
project-with-lua-lib % make
project-with-lua-lib % ./project-with-lua-lib
Lua version number is 504
```

如前述代码所示，C++ 程序将执行并打印：`Lua 版本号` `是 504`。

恭喜！你已经通过链接预编译的 Lua 库完成了第一个 C++ 项目。在下一节中，我们将探讨如何直接使用 Lua 源代码来避免本节开头提到的缺点。

# 使用 Lua 源代码构建 C++ 项目

使用 Lua 源代码构建你的 C++ 项目的好处是它始终作为项目的一部分进行编译，并且不存在由于编译器不兼容性而产生的意外情况。

与链接预编译的 Lua 库相比的主要区别是，我们现在将首先从源代码编译 Lua 库。最好在不修改它或只将一些选定的文件复制到新的文件夹层次结构中时使用源代码包。这将在未来帮助你，如果你需要使用更新的 Lua 版本。在这种情况下，你只需要用新版本替换 Lua 源代码文件夹。

## 创建一个用于与 Lua 源代码一起工作的项目

要使用 Lua 源代码创建一个项目，我们需要回到 Lua 源代码包：

```cpp
lua-5.4.6 % ls
Makefile README doc src
```

你需要 `src` 子文件夹。

为新项目创建一个文件夹。在其内部，创建一个空的 `main.cpp` 和一个空的 `Makefile`，并将前面 shell 输出中显示的 `src` 子文件夹作为项目文件夹中的 `lua` 子文件夹复制过来。项目结构应如下所示：

```cpp
project-with-lua-src % tree
.
├── Makefile
├── lua
│   ├── Makefile
│   ├── lapi.c
│   ├── ...
│   ├── lzio.c
│   └── lzio.h
└── main.cpp
```

## 编写 C++ 代码

你可以以与使用 Lua 库构建时完全相同的方式编写 C++ 代码。

## 编写 Makefile

让我们比上一个项目更进一步，为 `Makefile` 编写两个目标，如下所示：

```cpp
project-with-lua-src: main.cpp
    cd lua && make
    g++ -o project-with-lua-src main.cpp -Ilua -Llua -llua
clean:
    rm -f project-with-lua-src
    cd lua && make clean
```

这个 `Makefile` 首先进入 `lua` 子文件夹，然后构建 Lua 库。之后，它编译 C++ 代码并将其与 Lua 库链接。`lua` 子文件夹是 Lua 源代码存档中 `src` 文件夹的副本。如果你不小心将整个存档复制到那里，你可能会看到一些编译错误。

`Makefile` 还包括一个 `clean` 目标。这将删除编译文件。通常，所有构建系统都会实现一个 `clean` 目标。

## 测试项目

在终端中输入 `make` 来构建项目。然后输入 `./project-with-lua-src` 来执行编译后的项目：

```cpp
project-with-lua-src % make
project-with-lua-src % ./project-with-lua-arc
Lua version number is 504
```

C++ 程序将执行并打印：`Lua 版本号` `是 504`。

## 测试 clean 目标

由于我们已经实现了 `clean` 目标，让我们也测试一下：

```cpp
project-with-lua-src % make clean
rm -f project-with-lua-src
cd lua && make clean
rm -f liblua.a lua luac lapi.o lcode.o lctype.o ldebug.o
ldo.o ldump.o lfunc.o lgc.o llex.o lmem.o lobject.o
lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o
lundump.o lvm.o lzio.o lauxlib.o lbaselib.o lcorolib.o
ldblib.o liolib.o lmathlib.o loadlib.o loslib.o lstrlib.o
ltablib.o lutf8lib.o linit.o lua.o luac.o
```

这通过删除编译器生成的文件来清理工作文件夹。在一个生产就绪的项目中，你会在构建过程中首先将所有中间文件输出到单独的文件夹中，最可能命名为 `build` 或 `output`，并将该文件夹排除在源代码控制系统之外。

到目前为止，我们已经学习了两种将 Lua 集成到 C++ 项目中的方法。接下来，让我们学习如何从 C++ 中执行实际的 Lua 脚本。

# 执行简单的 Lua 脚本

要执行 Lua 脚本，你可以选择使用 Lua 库或 Lua 源代码。对于生产项目，我个人推荐使用源代码。对于学习目的，两种方式都可以。本书的其余部分我们将使用 Lua 源代码。

你能注意到这个吗？

即使你选择使用 Lua 源代码，在 `Makefile` 中，你首先将 Lua 源代码构建成 Lua 库，然后让你的项目链接到该库。与直接使用 Lua 库相比，使用 Lua 源代码只是在你项目中多进行一步操作。你可以更多地关注它们的相似之处，而不是差异。

现在，让我们看看一个更通用的项目结构。

## 创建项目

正如所说，接下来的章节中会有更复杂的项目。现在，我们将探索一个更通用的项目结构。我们将在一个共享位置构建和链接 Lua，而不是为每个项目创建副本。

以下是在其父文件夹中显示的项目结构：

```cpp
% tree
.
├── Chapter01
│   ├── execute-lua-script
│   │   ├── Makefile
│   │   ├── main.cpp
│   │   └── script.lua
│   ├── project-with-lua-lib
│   └── project-with-lua-src
└── lua
     ├── Makefile
     ├── README
     ├── doc
     └── src
```

本项目相关的两个文件夹如下：

+   `execute-lua-script` 文件夹包含主项目，其中包含一个 C++ 源文件、一个 `Makefile` 和一个 Lua 脚本文件

+   `lua` 文件夹包含 Lua 源代码包，它是未解压的，直接使用。

显示的其他文件夹表明了本书源代码的组织方式——首先按章节，然后按项目。遵循确切的结构是可选的，只要你能让它工作。

## 编写 Makefile

我们在前两个项目中看到了两个简单的 `Makefile`。让我们编写一个更灵活的 `Makefile`，如下所示：

```cpp
LUA_PATH = ../../lua
CXX = g++
CXXFLAGS = -Wall -Werror
CPPFLAGS = -I ${LUA_PATH}/src
LDFLAGS = -L ${LUA_PATH}/src
EXECUTABLE = executable
all: lua project
lua:
    @cd  ${LUA_PATH} && make
project: main.cpp
    $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) \
        -o $(EXECUTABLE) main.cpp -llua
clean:
    rm -f $(EXECUTABLE)
```

这个 `Makefile` 更灵活，应该足够用作学习目的的模板。它在以下方面与之前的版本不同：

+   在文件开头定义了一些变量。好处是在实际项目中，你可能需要编译多个文件。这样，你就不需要在每个目标中重复自己。这也更易于阅读。

+   默认目标，传统上命名为 `all`，依赖于两个其他目标。

+   在 `lua` 目标中，命令前有一个 `@` 符号。当 `make` 执行时，这将阻止在终端中打印出命令内容。

+   `LUA_PATH` 是相对于此 `Makefile` 所在文件夹的 Lua 源代码路径。

+   `CXX` 是定义 C++ 编译器程序的常规变量名。使用 `CC` 作为 C 编译器。

+   `CXXFLAGS` 定义了提供给 C++ 编译器的参数。使用 `CFLAGS` 作为 C 编译器。

+   `CPPFLAGS` 定义了提供给 C 预处理器的参数，C++ 与 C 共享相同的预处理器。

+   `LDFLAGS` 定义了提供给链接器的参数。在某些开发系统中，你可能需要在 `-o $(EXECUTABLE)` 后放置 `LDFLAGS`。

`Makefile` 准备就绪后，让我们编写 C++ 代码。

## 编写 C++ 代码

要执行 Lua 脚本，我们需要一些实际的操作，我们可以通过调用一些 Lua 库函数来获取这些操作。按照以下方式编写 `main.cpp`：

```cpp
#include <iostream>
#include <lua.hpp>
int main()
{
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    if (luaL_loadfile(L, "script.lua") ||
        lua_pcall(L, 0, 0, 0))
    {
        std::cout << "Error: "
                  << lua_tostring(L, -1)
                  << std::endl;
        lua_pop(L, 1);
    }
    lua_close(L);
    return 0;
}
```

代码执行以下操作：

1.  使用 `luaL_newstate` 打开 Lua 并创建 Lua 状态。

1.  使用 `luaL_openlibs` 打开 Lua 标准库。

1.  使用 `luaL_loadfile` 加载名为 `script.lua` 的 Lua 脚本，并使用 `lua_pcall` 执行它。我们很快就会编写 `script.lua`。

1.  如果脚本执行失败，则输出错误。这是在 `if` 子句中完成的。

1.  使用 `lua_close` 关闭 Lua。

这里使用的 Lua 函数以及 Lua 状态将在 *第三章* 中详细解释。

## 测试项目

如果你创建了一个空的 `script.lua`，请删除它。编译并运行项目：

```cpp
execute-lua-script % make
execute-lua-script % ./executable
Error: cannot open script.lua: No such file or directory
```

如预期的那样，它说 `script.lua` 未找到。别担心。你将在下一节编写它。需要注意的是，你已经完成了项目的 C++ 部分的编码，并且已经编译了项目。

## 编写 Lua 脚本

按照下面显示的单一行编写 `script.lua`：

```cpp
print("Hello C++!")
```

这将打印 `Hello C++!`。

再次执行项目，而无需重新编译 C++ 代码：

```cpp
execute-lua-script % ./executable
Hello C++!
```

恭喜！你现在已经从一个 C++ 程序中执行了 Lua 脚本，并在编译后改变了 C++ 程序的行为。

下一节提供了一些设置不同开发环境的想法。

# 其他工具链选项

如果你没有访问原生 POSIX 系统的权限，有许多其他工具链选项。这里我们给出了两个例子。因为你的开发平台可能不同，操作系统更新和情况会变化，这些只提供了一些想法。你总是可以在线研究并实验，以获得适合自己的舒适设置。

## 使用 Visual Studio 或 Xcode

Lua 源代码是用 C 编写的，不需要其他依赖。你可以将 Lua 源代码包中的 `src` 文件夹复制到 Visual Studio 或 Xcode 中，直接将其添加到你的项目中，或者将其配置为你的主项目所依赖的 Lua 项目。根据需要调整项目设置。这是完全可行的。

无论你选择使用哪个 IDE，记得检查其许可证，看看你是否可以使用该 IDE 来满足你的需求。

## 使用 Cygwin

如果你使用 Windows，你可以获取 Cygwin 来获得 POSIX 的体验：

1.  从 [`sourceware.org/cygwin/`](https://sourceware.org/cygwin/) 下载 Cygwin 安装程序并运行它。

1.  在选择软件包时，搜索名为 `make` 和 `gcc-g++` 的两个软件包。选择它们进行安装。

1.  稍微修改所有与 Lua 相关的 shell 命令和项目 `Makefiles`。你需要明确构建库的 Linux 版本。例如，将 `cd lua && make` 改为 `cd lua && make linux`。这是因为 Lua 的 `Makefile` 无法检测 Cygwin 是 Linux 版本。

1.  打开 Cygwin 终端，你可以像本章中的示例一样构建和运行项目。

# 摘要

在本章中，我们学习了如何编译 Lua 源代码，如何链接到 Lua 库，以及如何直接将 Lua 源代码包含到你的项目中。最后，我们从 C++ 代码中执行了一个 Lua 脚本。通过亲自遵循这些步骤，你应该能够舒适和自信地将 Lua 包含到你的 C++ 项目中，并为更复杂的工作做好准备。

在下一章中，我们将学习 Lua 编程语言的基础知识。如果你已经熟悉 Lua 编程语言，可以自由地跳过*第二章*，即“Lua 基础”。我们将在*第三章*，即“如何从 C++ 调用 Lua”中回到 Lua 与 C++ 之间的通信问题。
