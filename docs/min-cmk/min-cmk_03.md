# 第二章：Hello, CMake!

我们现在开始使用 CMake。首先，我们将介绍在终端中频繁使用的命令，然后是我们在 CMake 脚本中编写的命令。我们将通过启动一个*Hello, CMake!* 应用程序（回顾每个人最喜欢的 *Hello, World!* 程序），并用一个最小的 CMake 脚本进行引导，深入探讨我们使用的每个 CMake 命令。很快，这些命令将成为你的第二天性，让你轻松构建代码。

CMake 拥有丰富的功能集，但幸运的是，开始时只需要学习很少的内容就能提高生产力。它有很多选项可以处理复杂的使用场景；不过幸运的是，暂时我们不需要担心这些。知道它们在那儿就好，但不要觉得一开始就需要了解所有有关命令或 CMake 语言的知识。随着项目的推进，你将有足够的时间去学习这些。

本章将涵盖以下主题：

+   从命令行使用 CMake

+   检查我们的第一个 `CMakeLists.txt` 文件

+   CMake 生成器

+   项目下一步

+   添加另一个文件

# 技术要求

为了跟上进度，请确保你已满足*第一章*《入门》的要求。包括以下内容：

+   一个具有最新 **操作系统** (**OS**) 的 Windows、Mac 或 Linux 机器

+   一个工作中的 C/C++ 编译器（如果你还没有，建议使用每个平台的系统默认编译器）

本章中的代码示例可以通过以下链接找到：[`github.com/PacktPublishing/Minimal-`](https://github.com/PacktPublishing/Minimal-CMake)CMake。

# 从命令行使用 CMake

在深入了解第一个 CMake 脚本的内容之前，先克隆书中代码示例的仓库。可以通过打开终端并运行以下命令来执行此操作。

## Linux/macOS

如果你在 Linux/macOS 上工作，请运行以下命令：

```cpp
cd ~ # User's home directory on Linux/macOS (feel free to pick another location)
mkdir minimal-cmake
cd minimal-cmake
git clone https://github.com/PacktPublishing/Minimal-CMake.git .
```

现在你已经准备好在 macOS 或 Linux 上探索书中的代码仓库。

## Windows

如果你在 Windows 上工作，请运行以下命令：

```cpp
cd C:\Users\%USERNAME% # User's home directory on Windows (feel free to pick another location)
mkdir minimal-cmake
cd minimal-cmake
git clone https://github.com/PacktPublishing/Minimal-CMake.git .
```

现在你已经准备好在 Windows 上探索书中的代码仓库。

## 探索仓库

克隆仓库后，导航到*第二章*的第一个代码示例：

```cpp
cd ch2/part-1
```

从这里开始，输入 `ls`（如果你在 Windows 上且没有使用 Git Bash 或类似工具，请将 `ls` 替换为 `dir`）。显示的文件夹内容如下：

```cpp
CMakeLists.txt
main.c
```

`CMakeLists.txt` 文件显示我们处于 CMake 项目的根目录。所有 CMake 项目在其根目录都有这个文件，正是从这里我们可以要求 CMake 为我们的平台生成构建文件。

## 调用 CMake

让我们运行第一个 CMake 命令：

```cpp
cmake -B build
```

这是你将会学会并喜爱的最重要的 CMake 命令之一。它通常是在克隆一个使用 CMake 的仓库后你第一个运行的命令。运行这个命令时，你应该看到类似以下的输出（下面是 macOS 输出）：

```cpp
-- The C compiler identification is AppleClang 15.0.0.15000100
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info – done
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features – done
-- Configuring done (3.4s)
-- Generating done (0.0s)
-- Build files have been written to: /path/to/minimal-cmake/ch2/part-1/build
```

让我们简要分析一下我们使用的命令（`cmake -B build`）。命令的第一部分（`cmake`）是 CMake 可执行文件。如果我们在没有任何参数的情况下调用它，CMake 无法获取足够的信息来知道我们想要做什么；我们只会看到用法说明：

```cpp
Usage
  cmake [options] <path-to-source>
  cmake [options] <path-to-existing-build>
  cmake [options] -S <path-to-source> -B <path-to-build>
Specify a source directory to (re-)generate a build system for it in the current working directory. Specify an existing build directory to re-generate its build system.
```

在我们的情况下，我们希望 CMake 为我们的目标平台生成构建文件。为此，我们使用 `-B` 选项来指定一个文件夹来存放构建文件。该文件夹的名称和位置是任意的（我们可以写 `cmake -B my-built-files` 或 `cmake -B ../../build-output`），但通常会使用位于项目根目录下的 `build` 文件夹作为约定。

由于我们不想将这些文件提交到源代码控制中，通常会在 `.gitignore` 文件中添加某种形式的 `build`，这样我们就不会不小心开始跟踪这些文件（有些项目选择使用 `bin` 代替；不过，这种做法相对较少见）。这种做法是从源代码文件夹中使用 `cmake .` 的变体。如果这样做，增加了构建文件被意外添加到源代码控制中的风险，并且使得管理不同的构建类型变得繁琐。

显式指定源目录

如果 CMake 不是从与 `CMakeLists.txt` 文件相同的文件夹中调用的，可以提供一个单独的命令行参数 `-S`，并指定该文件所在的路径（当从构建自动化脚本如 GitHub Actions 调用 CMake 时，这一点尤其有用，这样就不需要切换目录）。如果在相同的文件夹中，您可以通过使用 `cmake -S . -B build` 来显式指定，但这在技术上是多余的，省略它是完全可以的。

CMake 的一大优点和一大缺点是，它在幕后为我们做出了很多猜测和假设，这些假设在没有仔细检查的情况下并不显而易见。稍后在本章中，我们将介绍更重要的选项，但可以简单地说，CMake 选择了一些合理的默认设置，我们可能需要稍后进行调整。

## 使用 CMake 构建

我们现在已经生成了一些构建文件（具体细节不重要），但还没有进行构建。就我们当前需要理解的部分，使用 CMake 是一个两步过程（第一步*严格来说*可以分解为两个进一步的步骤，称为*配置*和*生成*，但这两个步骤都会在运行 `cmake -B build` 时完成，所以我们现在可以将它们视为一个步骤）。构建步骤需要一个新命令：

```cpp
cmake --build build
```

前面的命令处理了在第一步中由 CMake 调用的底层构建系统。我们使用 `--build` 作为命令，而 `build` 只是我们在先前命令中指定的文件夹。

构建系统可以被看作是一种软件，它协调多个低级应用程序（例如编译器和链接器）在目标平台上生成某种输出（通常是应用程序或库）。在 macOS 和 Linux 的情况下，默认的底层构建系统将是 Make。

`CMakeLists.txt`）并将其映射到 Make 命令（以及我们即将学习的其他许多构建系统）。

直接调用构建系统

如果你知道底层的构建系统是 Make，你可以选择运行 `make -C build`，它的效果与 `cmake --build build` 相同。不幸的是，这并不具有可移植性（如果我们有一个构建脚本，在其他平台上选择了不同的构建系统，它将无法很好地工作）。坚持使用 CMake 命令可以保持一致的抽象层次，避免将来与特定的构建系统耦合。

Windows 上的情况略有不同，现在值得讨论。`cmake -B build` 和 `cmake --build build` 仍然会为我们生成构建文件并构建我们的代码，但底层的构建系统会有所不同。在 Windows 上，尤其是如果你按照 *第一章* 中的步骤，*入门*，生成的可能是 Visual Studio/MSBuild 项目文件，并且这些文件随后会被构建。

在 Windows 和 macOS/Linux 之间切换时的一个障碍是这两个独立的构建系统（Make 和 Visual Studio）具有稍微不同的行为（这是一种不幸的巧合）。Make 被称为**单配置**，而 Visual Studio 是**多配置**。我们尚未涉及配置的概念，但让我们先看看它们之间的可观察差异。

在 macOS 或 Linux 上，运行了两个 CMake 命令（配置和构建）后，我们可以通过运行以下命令启动我们的可执行文件：

```cpp
./build/minimal-cmake
```

奖励是标准的 *Hello,* *World!* 程序的变体：

```cpp
Hello, CMake!
```

如前所述，不幸的是，这在 Windows 上无法正常工作。相反，我们必须指定配置目录：

```cpp
build\Debug\minimal-cmake.exe
```

通过这个小的修改，我们将在 Windows 上看到 `Hello, CMake!` 被打印出来。

我们将在本章后面更详细地讨论配置以及单配置和多配置之间的差异，但现在，我们知道它们的存在及其主要差异。

另一个有用的提示是，一旦你运行了配置命令（`cmake -B build`），即使修改了 `CMakeLists.txt` 文件，也不必再次运行它。只需运行 `cmake --build build`，CMake 会检查是否有任何更改，并自动重新运行配置步骤。这避免了每次更改时反复运行两个命令。

# 检查我们的第一个 CMakeLists.txt 文件

既然我们已经使用 CMake 构建了我们的项目，让我们看看位于项目根目录下的 `CMakeLists.txt` 文件中的命令：

```cpp
cmake_minimum_required(VERSION 3.28)
project(minimal-cmake LANGUAGES C)
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.c)
target_compile_features(${PROJECT_NAME} PRIVATE c_std_17)
```

前述代码是制作 CMake 项目时可以采用的最低配置。`project` 还有一些其他可选参数，我们稍后会讲到，我们或许能够在不指定 `target_compile_features` 的语言版本的情况下进行设置（这样做的弊端是我们就会依赖平台上编译器的默认设置，而这些设置可能并非我们想要的。这也有可能使我们的 `CMakeLists.txt` 文件在跨平台时变得不太便携，因为不同平台或编译器的默认设置可能不同）。

大写或小写命令

在实际使用中，看到 CMake 命令全大写（例如 `ADD_EXECUTABLE` 而不是 `add_executable`）并不罕见。在 CMake 的早期版本中，命令必须使用大写字母，但今天 CMake 命令实际上是大小写不敏感的（`aDD_eXecuTAble` 技术上可以工作，但不推荐模仿）。现代的做法倾向于使用小写命令，这是本书中贯穿使用的风格。值得简要提到的是，CMake 变量（与命令不同）是区分大小写的，并且通常按照惯例使用大写字母。

让我们逐一分析每一行语句，了解它的作用以及为什么需要它。

## 设置最低版本

首先让我们来看一下如何设置可以与我们项目一起使用的最低（或最旧）版本的 CMake：

```cpp
cmake_minimum_required(VERSION 3.28)
```

每个 `CMakeLists.txt` 文件必须以前述语句开始，以告诉 CMake 在运行时，执行文件所需的最低 CMake 版本号是什么。版本越高，可用的功能就越多（同时也会有警告，提示可能已经被弃用或从旧版本中删除的内容）。在指定较高版本（拥有所有最新功能）和略旧版本（更多人可能使用的版本）的之间，需要取得平衡。例如，如果某个使用旧版本 CMake 的人尝试生成我们的项目，当他们尝试配置时，会看到以下错误消息：

```cpp
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
    CMake 3.28 or higher is required.  You are running version 3.15.5
```

如果你正在开发一个你自己或一个小团队将要构建的应用程序，指定最新的版本（至少是你已经安装的版本，在我们的例子中是 `3.28`）是可以的，也是个好主意。另一方面，如果你正在创建一个希望其他项目轻松采用的库，选择一个稍微旧一点的版本可能会更容易使用（如果你能够放弃一些新功能的话）。例如，在我们的例子中，我们可以轻松将所需版本号降至 `3.5`，而一切仍然能够正常工作（即使我们实际使用的是 `3.28`）。然而，如果我们将版本号降至 `2.8`，就会看到这个警告：

```cpp
Compatibility with CMake < 3.5 will be removed from a future version of CMake.
```

随着时间的推移，逐渐增加版本号是很重要的，这样可以保持 `CMakeLists.txt` 文件与 CMake 最新的更改和改进兼容。一个例子是 CMake `3.19` 和 `3.20` 之间的变化。在 CMake `3.20` 之前，在列出 `target_sources` 中的文件时，可以省略引用文件的扩展名。所以我们会使用如下代码：

```cpp
target_sources(${PROJECT_NAME} PRIVATE main)
```

这与以下代码是相同的：

```cpp
target_sources(${PROJECT_NAME} PRIVATE main.c)
```

如果 CMake 找不到完全匹配的文件，它会尝试附加一个潜在的扩展列表，看看是否有适合的扩展。这个行为容易出错，并可能导致潜在的 bug，因此被修复了。如果你尝试使用版本大于或等于 `3.20` 的 CMake 配置一个项目，而该项目的要求版本是 `3.19` 或更低版本，你将看到以下警告信息：

```cpp
CMake Warning (dev) at CMakeLists.txt:4 (target_sources):
  Policy CMP0115 is not set: Source file extensions must be explicit.  Run "cmake --help-policy CMP0115" for policy details.  Use the cmake_policy command to set the policy and suppress this warning.
  File: /path/to/main.c
```

我们还没有涉及到策略，所以暂时跳过详细信息，但本质上它们是 CMake 维护者为了避免在发布新版本的 CMake 时破坏项目兼容性的一种方式。

如果你将 `cmake_minimum_required(VERSION 3.19)` 更新为 `cmake_minimum_required(VERSION 3.20)`，但没有为 `main` 文件添加显式的扩展名，那么尝试配置时将产生一个硬错误：

```cpp
CMake Error at CMakeLists.txt:4 (target_sources):
  Cannot find source file: main
```

这有点偏题，但目的是强调为什么 `cmake_minimum_required` 非常重要，必须包括。通常来说，涉及 CMake 时最好是明确指定，而不是依赖于可能会根据平台或未来版本变化的隐式行为。

## 为项目命名

接下来让我们看看如何给我们的项目命名：

```cpp
project(minimal-cmake LANGUAGES C)
```

`project` 是所有 `CMakeLists.txt` 文件必须提供的第二个必需命令。如果你省略它，你会得到一个有用的错误信息：

```cpp
CMake Warning (dev) in CMakeLists.txt:
No project() command is present.  The top-level CMakeLists.txt file must contain a literal, direct call to the project() command.  Add a line of code such as
    project(ProjectName)
near the top of the file, but after cmake_minimum_required().
CMake is pretending there is a "project(Project)" command on the first line.
```

`project` 命令允许你为顶级项目指定一个有意义的名称，该项目可能是一个库和/或应用程序的集合。`project` 命令提供了许多附加选项，这些选项可能在指定时非常有用。在我们的示例中，我们提供了 `LANGUAGES C` 来让 CMake 知道项目包含哪种类型的源文件。这是可选的，但通常是良好的实践，因为它可以防止 CMake 做不必要的工作。如果我们没有在此情况下仅指定 C，CMake 将会搜索 C 和 C++ 编译器（CMake 脚本中使用 CXX 来表示 C++，以避免与不同上下文中的 `+` 运算符产生歧义）。

其他 `project` 选项包括：

+   `VERSION`

+   `DESCRIPTION`

+   `HOMEPAGE_URL`

这些选项的有用性可能因项目而异。对于小型本地项目，它们可能过于复杂，但如果一个项目开始获得关注并被更广泛使用，那么添加这些选项对于新用户可能是有帮助的。如需了解更多关于 CMake `project` 命令的信息，请参阅 [`cmake.org/cmake/help/latest/command/project.html#options`](https://cmake.org/cmake/help/latest/command/project.html#options)。

## 声明应用程序

设置好最低版本要求并命名我们的项目后，我们可以请求 CMake 创建我们的第一个可执行文件：

```cpp
add_executable(${PROJECT_NAME})
```

`add_executable` 很重要，因为这是我们项目中执行特定操作的第一行代码。调用此命令将创建 CMake 所称的 **目标**。

目标通常是一个可执行文件（如这里所示）或一个库（你还可以创建特殊的自定义目标命令）。CMake 提供了命令来直接获取和设置目标的值，而不会相互影响，或影响全局的 CMake 状态。目标是一个非常有用的概念，它使得可以将一组属性和行为封装在一起。可以把目标看作是 CMake 项目中的一个独立单元。它们使得我们能够轻松拥有多个可执行文件或库，并且每个都具有独特的属性，并且可以相互依赖。我们将在本书的其余部分频繁使用目标。

在之前的 `add_executable` 示例中，我们使用了一个已经为我们创建的现有 CMake 变量。

有两个重要的问题需要解决：

+   我们是如何知道要使用 `PROJECT_NAME` 的？

+   为什么我们需要在 `PROJECT_NAME` 周围使用 `${}`？

第一个问题的答案可以通过访问 [`cmake.org/cmake/help/latest/manual/cmake-variables.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html) 来解决。这个页面是一个有用的资源，列出了所有当前的 CMake 变量。如果我们向下滚动页面，我们会找到 **PROJECT_NAME** ([`cmake.org/cmake/help/latest/variable/PROJECT_NAME.html`](https://cmake.org/cmake/help/latest/variable/PROJECT_NAME.html))，并看到如下描述：

这是当前目录范围或更高范围内最近调用的 `project()` 命令所赋予的名称。

在我们的简单示例中，使用这个作为我们正在创建的目标的名称是足够的，因为目标和项目本质上是同一个东西。未来，在创建可能包含多个目标的较大 CMake 项目时，最好为目标名称创建一个单独的变量（例如，`${MY_EXECUTABLE}`），或者直接使用字面值（例如，`my_executable`）。我们稍后会讲解如何定义变量。

我们尚未回答的第二个问题是关于稍微奇怪的 `${}` 语法。CMake 变量遵循与系统环境变量类似的模式，你可能以前以某种形式遇到过这些变量。为了访问存储在变量中的值，我们需要用 `${}` 将其括起来，以有效地取消引用或解包存储的值。举个简单的例子，如果我们在终端中输入 `echo PATH`，我们将看到打印出的 `PATH`。然而，如果我们输入 `echo ${PATH}`（或者在 Windows 上输入 `echo %PATH%`），我们将看到 `PATH` 变量的内容（在 macOS 上，这通常是类似 `/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin...` 的内容）。CMake 也是一样。做个简单的测试，让我们添加一个调试语句来确认 `PROJECT_NAME` 的值。我们可以通过在 `CMakeLists.txt` 文件的底部添加以下命令来实现：

```cpp
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
```

当我们运行 `cmake -B build` 时，我们将在控制台中看到 `PROJECT_NAME: minimal-cmake` 被打印出来。

使用 `${PROJECT_NAME}` 作为我们的目标名的一个小优势是，我们保持了 `CMakeLists.txt` 文件的简洁性，没有引入额外的复杂性。另一个优势是，如果我们决定更改项目的名称，我们只需要在一个地方进行更改（遵循通常的建议，`${PROJECT_NAME}` 会自动反映新值）。

## 添加源文件

现在让我们理解如何指定我们要构建的文件：

```cpp
target_sources(${PROJECT_NAME} PRIVATE main.c)
```

现在通过 `add_executable` 定义了一个目标后，我们可以通过该目标的名称（在我们的例子中是 `${PROJECT_NAME}`，它解包为 `minimal-cmake`）在其他与目标相关的命令中引用它。这些命令通常以 `target_` 为前缀，方便我们识别。与之前的 CMake 命令相比，这些命令的巨大优势在于它们消除了大量潜在问题和关于 CMake 命令作用范围的混淆。在过去，某个 `CMakeLists.txt` 文件中定义的设置可能会无意中泄露到另一个文件中，往往带来痛苦的后果。通过更加规范地使用目标，我们可以为该目标指定特定的属性和设置，从而避免影响其他目标。

关于 `target_sources` 命令，这是我们为目标指定要构建的源文件的地方。紧跟在 `main.c` 之前的参数控制源文件的可见性或作用范围。在大多数情况下，我们希望在这里使用 `PRIVATE`，以便只有这个目标构建源文件。其他作用范围参数有 `PUBLIC`（该目标和依赖它的其他目标使用）和 `INTERFACE`（仅供依赖的目标使用）。我们将在以后回到这些关键字（当我们讨论库时），因为它们出现在所有的 `target_` 命令中，并且有多种用途。

#### 设置语言特性

最后，让我们确保明确指定我们正在使用的语言版本：

```cpp
target_compile_features(${PROJECT_NAME} PRIVATE c_std_17)
```

我们的 `CMakeLists.txt` 文件中的最后一条命令是 `target_compile_features`。这是指定我们希望使用的语言版本的便捷方法，在本例中为 `C17`。我们也可以更精细地选择特定的语言特性（例如，`c_restrict`），但选择语言版本更加清晰简洁。你可以在这里查看 C 语言的可用模式和特性：[`cmake.org/cmake/help/latest/prop_gbl/CMAKE_C_KNOWN_FEATURES.html`](https://cmake.org/cmake/help/latest/prop_gbl/CMAKE_C_KNOWN_FEATURES.html)

我们也可以选择另一种方式，使用 `set(CMAKE_C_STANDARD 17)`。这会在整个项目中应用此设置。我们可能希望这种行为，但在我们的情况下，我们坚持采用更具目标导向的方法，因此只有 `minimal-cmake` 目标会受到影响。

就构建小型应用程序而言，这大致涵盖了我们在使用 CMake 时所需的一切。单独来看，这已经非常有用，因为我们现在有了一种完全便携的方式，可以在 Windows、macOS 和 Linux 上运行我们的代码。这使得代码更容易共享和协作。如果其他平台的用户或开发者想查看我们的项目，只要他们安装了 CMake（很可能还需要 Git），他们可以通过几条命令轻松完成。如果你分享的是 Xcode、Visual Studio，甚至是 Make 项目，他们就需要做更多的工作。好消息是，即使用户希望使用 Visual Studio 或 Xcode 来测试或修改代码，他们仍然可以这样做。这将我们引向了使用 CMake 的下一个重要部分：生成器。

# CMake 生成器

在 *调用 CMake* 部分，我们略过了运行 `cmake -B build` 时发生了什么。当我们运行 `cmake -B build` 时，我们要求 CMake 为我们生成构建文件，但到底是什么构建文件呢？CMake 会尽力选择平台的默认值；在 Windows 上是 Visual Studio，而在 macOS 和 Linux 上是 Make。所有潜在生成器的列表可以通过访问 [`cmake.org/cmake/help/latest/manual/cmake-generators.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html) 或运行 `cmake --help` 命令找到（默认生成器会用星号标出）。如果你不确定正在使用哪个生成器，可以打开 `build/` 文件夹中的 `CMakeCache.txt` 文件并搜索 `CMAKE_GENERATOR`。你应该能找到类似下面的行：

```cpp
INTERNAL, so we shouldn’t depend on this in our scripts, but as a debugging aid it’s sometimes useful to check.
			Specifying a generator
			If we would like more control over the generator CMake uses, we can specify this explicitly by using the `-G` argument, `cmake -B build -G <generator>`, as in this example:

```

cmake -B build -G Ninja

```cpp

			Here, we’ve referenced the Ninja build system generator ([`ninja-build.org/`](https://ninja-build.org/)), a build tool designed to run builds as fast as possible. Unfortunately, if we try and run this command on macOS or Linux, we’ll get an error as we currently do not have Ninja installed (fortunately on Windows, Ninja comes bundled with Visual Studio, and if we’re using the Developer Command Prompt or have run `VsDevCmd.bat`, we’ll have it in our path).
			Ninja can be downloaded from GitHub ([`github.com/ninja-build/ninja/releases`](https://github.com/ninja-build/ninja/releases)), and once the executable is on your machine, you can add it to your `PATH` or move it to an appropriate folder such as `/usr/local/bin` or `/opt/bin`.
			Security settings for macOS
			On macOS, you may need to open **System Settings** and navigate to **Privacy and Security** to allow Ninja to run because it is not from an identified developer.
			It may also be easier to acquire Ninja through a package manager, particularly on Linux (e.g., `apt-get` `install ninja-build`).
			Ninja advantages
			Ninja is designed to be fast, so it’s well worth setting it up for use with future chapters when we start building larger third-party dependencies. Ninja will take full advantage of all system cores by default, and this really shows when comparing build times against other generators. Ninja’s multi-config generator support is also useful.
			One thing to mention is even with this change to the generator behind the scenes, we can still use `cmake --build build` to build our project; there is no need to memorize any other build-specific commands. This consistency is invaluable as it reduces the cognitive load when working with different build systems, they’re largely abstracted away from us and we can focus on our project.
			If you have generated some build artifacts using one generator and would like to switch to another, this requires deleting the build folder and starting over (e.g., `rm -rf build` or `cmake -B build –G <new-generator>`). If you aren’t switching generators, a useful argument to be aware of (added in CMake `3.24`) is `--fresh`:

```

cmake -B build -G <new-generator> --fresh

```cpp

			Using `--fresh` will remove the existing `CMakeCache.txt` and `CMakeFiles/` directory and restore them to the state they’d be if you were doing the first configure.
			CMake configs
			Now that we know how to specify a generator, we can talk about the one remaining topic in this chapter, configs (a concept inextricably linked to generators themselves). Generators come in two varieties, either single-config or multi-config. We’ve actually already encountered one of each already. Make is a single-config generator, and the default config we built without specifying anything was `Debug`. Visual Studio is a multi-config generator, which is why when we ran our earlier example on Windows, we had to specify the `Debug/` folder inside the `build/` folder instead of only the `build/` folder (`build\Debug\minimal-cmake.exe` versus `build/minimal-cmake`).
			Single-config generators
			With a single-config generator, when we run `cmake -B build`, we can pass an additional argument to set a CMake variable called `CMAKE_BUILD_TYPE`. We do this with `-D` to define a CMake variable and override the default value (one set by CMake or us in our `CMakeLists.txt` file). To be explicit about the config/build type, we’d write the following:

```

cmake -B build -DCMAKE_BUILD_TYPE=Debug

```cpp

			Usually, there are at least three build types: `Debug`, `Release`, and `RelWithDebInfo` (there’s also `MinSizeRel` with Visual Studio). These build types essentially control what underlying compiler flags are set for things such as optimization, debugging, and logging through defines. When developing code, we usually want to use the `Debug` configuration to allow us to easily step through our code in a debugger. When we’re ready to share our project with users, we use the `Release` configuration to get maximum performance. `RelWithDebInfo` is a happy medium. Some optimizations may be disabled compared to `Release`, but performance will be similar. Debug symbols are also created to make debugging `Release` builds easier.
			The defaults are more than sufficient for our purposes but, in advanced cases, it is possible to create your own build types (this is easier said than done as you need to know the compiler flags to use across a host of platforms/compilers, but if you ever did need to do this, you can).
			One thing to be aware of when changing `CMAKE_BUILD_TYPE` is the artifacts in your build folder will be completely rebuilt depending on the build type. So, for example, if you have a larger project, and you normally have `-DCMAKE_BUILD_TYPE=Release` set, if you run `cmake -B build -DCMAKE_BUILD_TYPE=Debug` and run `cmake --build build`, the release files will be overwritten, and so switching back again to `Release` will wipe out all the `Debug` build files. For this reason, it is wise to use different folders for the different configurations to make this switching back and forward more efficient. To illustrate, we could have the following:

```

cmake -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug

cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release

```cpp

			To build each config, you’d then use either `cmake --build build-debug` or `cmake --build build-release`. You could also group the different configurations under the build folder (e.g., `build/debug` or `build/release`), but remember each subfolder is completely distinct and nothing is shared between the two when using single-config generators.
			Let’s now explore multi-config generators.
			Multi-config generators
			With a multi-config generator, `CMAKE_BUILD_TYPE` goes away and instead, the config is specified at build time rather than configuration time. It also handles the case described earlier where different build types can overwrite one another.
			With a multi-config generator, you’d configure it in this way:

```

cmake -B build -G "Visual Studio 17 2022" # Windows

# 为了简洁起见，年份可以省略。

cmake -B build -G "Visual Studio 17" # Windows

cmake -B build -G Xcode # macOS

cmake -B build -G "Ninja Multi-Config" # Linux

```cpp

			Then, when building, you pass an additional argument, `--config`, along with the config type:

```

cmake --build build --config Debug

cmake --build build --config Release

```cpp

			Multi-config generators will create subdirectories inside the build folder you specified. In the case of Ninja Multi-Config, this will be `Debug`, `Release`, and `RelWithDebInfo` (no `MinSizeRel`). Multi-config generators are a good choice to stick with and, in later chapters, we’ll cover a couple more reasons why to prefer them.
			That covers the most essential operations you’ll perform when working with CMake on a daily basis. There are many more options and tools available to streamline usage and simplify project configuration, but you could survive with what we’ve covered here for some time.
			Project next steps
			Now we’ve been through our first `CMakeLists.txt` file and are more familiar with build types (configs) and generators, it’s time to look at a real program and see how we can start to evolve it with CMake’s help.
			Staying with the book’s sample code, navigate to `ch2/part-2` in your terminal and run the commands we’re now intimately familiar with, `cmake -B build` (feel free to specify a generator of your choosing such as `-G "Ninja Multi-Config"`), followed by `cmake --``build build`.
			After configuring and building, we can run the sample application by typing `./build/Debug/minimal-cmake_game-of-life` on macOS and Linux, or `build\Debug\minimal-cmake_game-of-life.exe` on Windows (for brevity, we’ll use the POSIX path convention from macOS and Linux going forward; this is one reason to recommend using Git Bash from within Terminal on Windows as the experience will be more consistent).
			You should see the following printed (several blank lines omitted here):

```

****************************************

*********@******************************

*******@*@******************************

********@@******************************

****************************************

```cpp

			Press *Enter* on your keyboard and you’ll see the pattern denoted by the `@` symbols update (hitting *Enter* repeatedly will cause the scene to keep updating).
			What you are seeing is an incredibly simple implementation of John Horton Conway’s *Game of Life*. *Game of Life* is an example of cellular automaton. Conway’s *Game of Life* is represented as a grid, with each cell in either an on or off state. A set of rules is processed for each update to decide which cells turn on, which turn off, and which stay the same. The topic is vast; if you would like to learn more about it, please check out the Wikipedia pages about both Conway’s *Game of Life* ([`en.wikipedia.org/wiki/Conway%27s_Game_of_Life`](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) and cellular automaton more generally ([`en.wikipedia.org/wiki/Cellular_automaton`](https://en.wikipedia.org/wiki/Cellular_automaton)).
			For our purposes, we’d just like something interesting to look at so we can start to evolve it over time. The implementation is written in C and the `CMakeLists.txt` file differs from the first one we looked at by only the name (the *Game of Life* implementation lives in `main.c`).
			In the book’s repository (available from [`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake)), every `ch<n>/part-<n>` section in each chapter builds on the last in some small way. To help make sense of these incremental changes, see the following callout about using Visual Studio Code to make visualizing these differences easier.
			Visual Studio Code compare
			A useful feature in Visual Studio Code is the `code .` from your terminal will help with this, so all related files can be easily accessed). It’s then simple to highlight what has changed between versions of our `CMakeLists.txt` files without needing to switch back and forth between them. Focusing on the changes instead of reviewing an entire file, which may be very similar to the previous one, is an efficient strategy.
			Don’t worry too much about the code. It’s not super important how it works; what is important is how CMake can start to help us organize and enhance our application.
			Adding another file
			Before we wrap up, let’s make one small addition to our application. We’d like to improve the performance of our update logic in our current implementation of *Game of Life*. One subtlety of implementing *Game of Life* is we can’t change the board we’re reading from at the same time. If we do, then the cells from the row we’re on will have changed from their earlier state by the time we get to the next row, which will mean the simulation won’t run correctly. In the implementation in `ch2/part2` (a reminder to refer to [`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake) to find this), we simply make a copy of the whole board, read from that in `update_board` (see line 72 in `ch2/part-2/main.c`) and write back to the original board. This is okay, but if most cells don’t change, it’s wasteful. A better approach is to record the cells that change, and then write back to the original board at the end. By doing this, we only need to allocate memory for cells that change instead of the whole board.
			Adding a dynamic array
			Let’s add a simple data structure to make this possible. C unfortunately doesn’t have a built-in dynamic array, which would be particularly useful in this case, so let’s add one.
			Moving to `ch2/part3` from the book’s GitHub repository, there are two new files, `array.h` and `array.c`. To keep them grouped logically together, they’ve been added to a folder called `array`. The interface provided by `array.h` is like that of `std::vector` from C++. It’s a little trickier to use as C doesn’t support generics/templates, but for our purposes, it’ll be a huge help.
			With this file added, we need to ensure CMake knows about it; otherwise, it won’t be built. To do this, we simply add `array/array.c` to the existing `target_sources` command from earlier:

```

target_sources(${PROJECT_NAME} PRIVATE main.c cmake --build build again (不需要重新配置)。

            忘记添加一个文件

            如果我们没有将 `array.c` 添加到 `CMakeLists.txt` 文件中，而是添加了对 `array` 的使用代码，并尝试编译（`cmake --build build`），那会很有用。编译是可以通过的，但我们会遇到大家都熟悉的问题：链接器错误。以下输出展示了这一点：

```cpp
ld: Undefined symbols:
  _array_free, referenced from:
      _update_board in main.c.o
  _array_size, referenced from:
      _update_board in main.c.o
      _update_board in main.c.o
      _update_board in main.c.o
  _internal_array_grow, referenced from:
      _update_board in main.c.o
      _update_board in main.c.o
```

            这是因为链接器找不到列出的函数的实现（例如，`_array_size`）。输出文件，在 macOS/Linux 上是 `array.c.o`，在 Windows 上是 `array.c.obj` 或 `array.obj`，不会被创建（你可以通过进入 `build/CMakeFiles/minimal-cmake_game-of-life.dir/Debug` 来查看这些文件是否存在，如果使用的是 Ninja Multi-Config 生成器，其他生成器会将其放在类似位置）。

            这是使用 CMake 时常见的早期问题（创建了文件但忘记将它们添加到 `CMakeLists.txt` 中）。

            是否使用 GLOB

            到这个时候，值得提到一个常常在 CMake 中出现的话题，那就是是否像前面的例子一样明确列出要构建的文件，还是使用一种 `GLOB`（有效地搜索）每个文件夹层级中的所有源文件的技术。就像软件工程和计算机科学中的一切一样，这里面有权衡。有些情况下，使用 `GLOB` 会更简单快捷。这可能看起来像下面这样：

```cpp
file(GLOB sources CONFIGURE_DEPENDS *.c)
target_sources(foobar PRIVATE ${sources})
```

            这可能在你和你的环境中运行得很好，但也有一系列风险。在 `CONFIGURE_DEPENDS`（CMake `3.12` 中新增）出现之前，如果你添加了源文件（例如，从版本控制系统中拉取最新代码）而没有进行配置，运行 `cmake --build build` 时会遇到问题。在这种情况下，CMake 构建会失败。指定 `CONFIGURE_DEPENDS` 可以避免这种情况，但不能保证它与所有生成器兼容，对于更大的项目，可能会引发性能问题。CMake 的维护者仍然建议明确指定要构建的源文件，这是我们在本书中一直遵循的做法。它减少了不小心构建不想要的文件的风险，并且对 `CMakeLists.txt` 文件所做的更改有助于在版本控制中跟踪。前面提到的链接器错误一开始确实让人沮丧，但你很快就会适应，添加新文件也会变得自然而然。

            在添加了新的 `array.c` 文件后，我们可以更改更新函数以使用新的逻辑，并提高代码的性能（`ch2/part-3` 中有一个稍微更激动人心的棋盘配置，值得一看）。

            在 target_sources 中引用接口文件

            最后一个值得提及的点是`array.h`怎么办？由于我们在`main.c`中相对引用了这个文件（使用`#include "array/array.h"`而不是`#include <array/array.h>`），我们不需要在`CMakeLists.txt`文件中明确提到任何包含目录（当我们涉及到库时，这一点会更重要）。如果你使用的是一种生成工具，能够生成一个可以在独立工具中打开的项目或解决方案（例如集成开发环境，如 Visual Studio 或 Xcode），那么你可以像下面这样将`array.h`添加到`target_sources`中：

```cpp
target_sources(
  ${PROJECT_NAME} PRIVATE main.c array/array.h array/array.c)
```

            这样，它会出现在项目视图中，这对于维护可能很有用；不过，它并不是构建代码所必需的。由于我们在大多数示例中将使用 Visual Studio Code 和文件夹项目视图，为了简洁起见，我们会省略头文件。指定头文件还有一个好处，那就是如果文件被意外删除，或者无法从源控制中获取，CMake 会在配置步骤中提前失败，而不是在构建时。增加的维护成本可能是值得的，特别是在团队较大的情况下。

            总结

            非常棒，你已经走到了这一步；我们已经覆盖了很多内容！我们从熟悉如何通过终端使用 CMake 开始（`cmake -B build`和`cmake --build build`应该已经深深记在你的脑海中了）。接着，我们通过一个简单的`CMakeLists.txt`文件，检查了最重要的命令以及它们为何需要。然后，我们深入探讨了生成器，研究了单配置生成器和多配置生成器之间的一些差异，以及如何在每种情况下指定构建类型。最后，我们看了我们项目的种子，康威的*生命游戏*实现，并了解了如何在扩展功能时，逐步向现有项目中添加更多文件。

            在下一章中，我们将探讨如何将外部依赖项引入我们的项目。这将使我们能够增强和改善应用程序的功能以及代码的可维护性。这正是 CMake 的强大之处，它帮助我们集成现有的库，而无需从头开始实现一切。

```cpp

```
