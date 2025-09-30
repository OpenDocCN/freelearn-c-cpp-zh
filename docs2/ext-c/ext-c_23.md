# 第二十三章

# 构建系统

对于我们程序员来说，构建项目并运行其各种组件是开发新功能或修复项目中报告的错误的第一个步骤。实际上，这不仅仅限于 C 或 C++；几乎任何包含用编译型编程语言（如 C、C++、Java 或 Go）编写的组件的项目，都需要首先进行构建。

因此，能够快速轻松地构建软件项目是几乎任何在软件生产流程中工作的一方的基本需求，无论是开发者、测试人员、集成人员、DevOps 工程师，甚至是客户支持人员。

更重要的是，当你作为一个新手加入一个团队时，你做的第一件事就是构建你将要工作的代码库。考虑到所有这些，很明显，解决构建软件项目的能力是合理的，鉴于它在软件开发过程中的重要性。

程序员需要频繁地构建代码库以查看他们更改的结果。仅使用少量源文件构建项目似乎既简单又快捷，但当源文件数量增加（相信我，这种情况会发生）时，频繁构建代码库就变成了开发任务的真正障碍。因此，一个适当的软件项目构建机制至关重要。

人们过去常常编写 shell 脚本来构建大量的源文件。尽管它有效，但需要大量的努力和维护来保持脚本足够通用，以便在各种软件项目中使用。随后，大约在 1976 年，贝尔实验室开发了第一个（或者至少是其中之一）名为 *Make* 的 *构建系统*，并在内部项目中使用。

此后，Make 在所有 C 和 C++ 项目中得到了大规模的应用，甚至在其他 C/C++ 不是主要语言的项目中也是如此。

在本章中，我们将讨论广泛使用的 C 和 C++ 项目的 *构建系统* 和 *构建脚本生成器*。作为本章的一部分，我们将讨论以下主题：

+   首先，我们将探讨什么是构建系统以及它们有什么好处。

+   然后，我们将介绍 Make 是什么以及如何使用 Makefile。

+   CMake 是下一个主题。你将了解构建脚本生成器，并学习如何编写简单的 `CMakeLists.txt` 文件。

+   我们将了解 Ninja 是什么以及它与 Make 的区别。

+   本章还将探讨如何使用 CMake 生成 Ninja 构建脚本。

+   我们将深入研究 Bazel 是什么以及如何使用它。你将了解 `WORKSPACE` 和 `BUILD` 文件，以及在一个简单的用例中应该如何编写它们。

+   最后，你将获得一些已发布的各种构建系统比较的链接。

注意，本章中使用的所有构建工具都需要事先安装在你的系统上。由于这些构建工具正在大规模使用，因此互联网上应有适当资源和文档。

在第一部分，我们将探讨构建系统实际上是什么。

# 什么是构建系统？

简而言之，构建系统是一组程序和配套的文本文件，它们共同构建一个软件代码库。如今，每种编程语言都有自己的构建系统。例如，在 Java 中，有*Ant*、*Maven*、*Gradle*等等。但“构建代码库”究竟是什么意思呢？

构建代码库意味着从源文件中生成最终产品。例如，对于一个 C 代码库，最终产品可以是可执行文件、共享对象文件或静态库，而 C 构建系统的目标就是从代码库中找到的 C 源文件生成这些产品。为此目的所需的操作细节在很大程度上取决于编程语言或代码库中涉及的语言。

许多现代构建系统，尤其是在用*JVM 语言*（如 Java 或 Scala）编写的项目中，提供额外的服务。

它们也进行*依赖管理*。这意味着构建系统检测目标代码库的依赖关系，并下载所有这些依赖关系，在*构建过程*中使用下载的工件。这非常方便，尤其是在项目中有很多依赖关系的情况下，这在大型代码库中通常是常见的情况。

例如，*Maven*是 Java 项目中最著名的构建系统之一；它使用 XML 文件并支持依赖管理。不幸的是，我们在 C/C++项目中没有很好的依赖管理工具。为什么我们还没有得到类似 Maven 的构建系统，这是一个值得讨论的问题，但它们尚未开发的事实可能表明我们并不需要它们。

构建系统的另一个方面是能够构建包含多个模块的大型项目。当然，这可以通过使用 shell 脚本和编写递归的*Makefiles*来实现，这些 Makefiles 可以遍历任何级别的模块，但我们谈论的是对这种需求的原生支持。不幸的是，Make 并不提供这种原生支持。另一个著名的构建工具 CMake 则提供了这种支持。我们将在专门介绍 CMake 的章节中进一步讨论这个问题。

到目前为止，许多项目仍然使用 Make 作为它们的默认构建系统，然而，通过使用 CMake。事实上，这是使 CMake 非常重要的一个点，在加入 C/C++项目之前，你需要学习它。请注意，CMake 不仅限于 C 和 C++，也可以用于使用各种编程语言的项目。

在下一节中，我们将讨论 Make 构建系统以及它是如何构建项目的。我们将给出一个多模块 C 项目的示例，并在本章中用它来展示如何使用各种构建系统构建这个项目。

# Make

Make 构建系统使用 Makefile。Makefile 是一个名为 "Makefile"（确切地说是这个名字，没有任何扩展名）的文本文件，位于源目录中，它包含 *构建目标* 和命令，告诉 Make 如何构建当前的代码库。

让我们从简单的多模块 C 项目开始，并为其配备 Make。以下 shell box 显示了项目中的文件和目录。如您所见，它有一个名为 `calc` 的模块，还有一个名为 `exec` 的模块正在使用它。

`calc`模块的输出将是一个静态对象库，而`exec`模块的输出是一个可执行文件：

```cpp
$ tree ex23_1
ex23_1/
├── calc
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    └── main.c
2 directories, 5 files 
$
```

Shell Box 23-1：目标项目中的文件和目录

如果我们想在没有使用构建系统的情况下构建上述项目，我们必须按以下顺序运行以下命令。请注意，我们已将 Linux 作为此项目的目标平台：

```cpp
$ mkdir -p out
$ gcc -c calc/add.c -o out/add.o
$ gcc -c calc/multiply.c -o out/multiply.o
$ gcc -c calc/subtract.c -o out/subtract.o
$ ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
$ gcc -c -Icalc exec/main.c -o out/main.o
$ gcc -Lout out/main.o -lcalc -o out/ex23_1.out
$
```

Shell Box 23-2: 构建目标项目

如您所见，项目有两个工件：一个静态库，`libcalc.a`，和一个可执行文件，`ex23_1.out`。如果您不知道如何编译 C 项目，或者前面的命令对您来说很陌生，请阅读 *第二章*，*编译和链接*，以及 *第三章*，*目标文件*。

*Shell Box 23-2* 中的第一个命令创建了一个名为 out 的目录。这个目录应该包含所有可重定位目标文件和最终产品。

接着，接下来的三个命令使用 `gcc` 编译 `calc` 目录中的源文件，并生成它们相应的可重定位目标文件。然后，这些目标文件在第五个命令中使用，以生成静态库 `libcalc.a`。

最后，最后两个命令从 exec 目录编译文件 `main.c`，并将其与 `libcalc.a` 链接在一起，生成最终的执行文件，`ex23_1.out`。请注意，所有这些文件都放在 out 目录内。

前面的命令会随着源文件数量的增加而增长。我们可以将前面的命令保存在一个名为 *build script* 的 shell 脚本文件中，但有一些方面我们在事先应该考虑：

+   我们是否将在所有平台上运行相同的命令？不同的编译器和环境中有一些细节是不同的；因此，命令可能因系统而异。在最简单的情况下，我们应该为不同的平台维护不同的 shell 脚本。那么，这意味着我们的脚本不是 *可移植的*。

+   当项目添加新的目录或新的模块时会发生什么？我们需要更改构建脚本吗？

+   如果我们添加新的源文件，构建脚本会发生什么？

+   如果我们需要一个新的产品，比如一个新的库或一个新的可执行文件，会发生什么？

一个好的构建系统应该处理上述所有或大多数情况。让我们展示我们的第一个 Makefile。此文件将构建上述项目并生成其产品。本节和以下各节中编写的所有用于构建系统的文件都可以用来构建这个特定的项目，而不会涉及更多。

以下代码框显示了我们可以为上述项目编写的最简单的 Makefile 的内容：

```cpp
build:
    mkdir -p out
    gcc -c calc/add.c -o out/add.o
    gcc -c calc/multiply.c -o out/multiply.o
    gcc -c calc/subtract.c -o out/subtract.o
    ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
    gcc -c -Icalc exec/main.c -o out/main.o
    gcc -Lout -lcalc out/main.o -o out/ex23_1.out
clean:
    rm -rfv out
```

代码框 23-1 [Makefile-very-simple]：为特定项目编写的非常简单的 Makefile

前面的 Makefile 包含两个目标：`build` 和 `clean`。目标有一组命令，当调用目标时应该执行这些命令。这组命令被称为目标的 *配方*。

为了运行 Makefile 中的命令，我们需要使用 `make` 命令。你需要告诉 `make` 命令要运行哪个目标，但如果留空，make 总是执行第一个目标。

要使用 Makefile 构建前面的项目，只需将 *代码框 23-1* 中的行复制到名为 `Makefile` 的文件中，并将其放在项目的根目录下。项目的目录内容应类似于以下 shell 框中所示：

```cpp
$ tree ex23_1
ex23_1/
├── Makefile
├── calc
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    └── main.c
2 directories, 6 files 
$
```

Shell 框 23-3：在添加 Makefile 后在目标项目中找到的文件和目录

之后，你只需运行 make 命令。`make` 程序会自动在当前目录中查找 `Makefile` 文件并执行其第一个目标。如果我们想运行 `clean` 目标，我们必须使用 `make clean` 命令。`clean` 目标可以用来删除构建过程中产生的文件，这样我们就可以从头开始进行全新的构建。

以下 shell 框显示了运行 `make` 命令的结果：

```cpp
$ cd ex23_1
$ make
mkdir -p out
gcc -c -Icalc exec/main.c -o out/main.o
gcc -c calc/add.c -o out/add.o
gcc -c calc/multiply.c -o out/multiply.o
gcc -c calc/subtract.c -o out/subtract.o
ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
gcc -Lout -lcalc out/main.o -o out/ex23_1.out
$
```

Shell 框 23-4：使用非常简单的 Makefile 构建目标项目

你可能会问，“构建脚本（用 shell 脚本编写的）和上面的 Makefile 之间有什么区别？”你提出这个问题是正确的！前面的 Makefile 并不代表我们通常使用 Make 构建项目的方式。

实际上，前面的 Makefile 是对 Make 构建系统的天真使用，它没有从 Make 提供的已知特性中受益。

换句话说，到目前为止，Makefile 与 shell 脚本非常相似，我们仍然可以使用 shell 脚本（尽管当然这会涉及更多的工作）。现在我们到了 Makefile 变得有趣并且真正不同的地方。

下面的 Makefile 仍然很简单，但它介绍了我们感兴趣的 Make 构建系统的更多方面：

```cpp
CC = gcc
build: prereq out/main.o out/libcalc.a
    ${CC} -Lout -lcalc out/main.o -o out/ex23_1.out
prereq:
    mkdir -p out
out/libcalc.a: out/add.o out/multiply.o out/subtract.o
    ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
out/main.o: exec/main.c calc/calc.h
    ${CC} -c -Icalc exec/main.c -o out/main.o
out/add.o: calc/add.c calc/calc.h
    ${CC} -c calc/add.c -o out/add.o
out/subtract.o: calc/subtract.c calc/calc.h
    ${CC} -c calc/subtract.c -o out/subtract.o
out/multiply.o: calc/multiply.c calc/calc.h
    ${CC} -c calc/multiply.c -o out/multiply.o
clean: out
    rm -rf out
```

代码框 23-2 [Makefile-simple]：为特定项目编写的新的但仍然简单的 Makefile

正如你所看到的，我们可以在 Makefile 中声明一个变量并在多个地方使用它，就像我们在先前的代码框中声明的 CC 一样。变量，加上 Makefile 中的条件，允许我们用比编写一个能够实现相同灵活性的 shell 脚本更少的努力来编写灵活的构建指令。

Makefile 的另一个酷特性是能够包含其他 Makefile。这样，你可以从你之前项目中编写的现有 Makefile 中受益。

正如你在先前的 Makefile 中看到的，每个 Makefile 可以有多个目标。目标从行的开头开始，以冒号“:”结束。必须使用一个制表符字符来缩进目标（即配方）内的所有指令，以便让`make`程序能够识别。关于目标的一个酷地方是：它们可以依赖于其他目标。

例如，在先前的 Makefile 中，`build`目标依赖于`prereq`、`out /main.o`和`out/libcalc.a`目标。然后，每当调用`build`目标时，首先会检查其依赖的目标，如果它们尚未生成，那么这些目标将首先被调用。现在，如果你更仔细地观察先前的 Makefile 中的目标，你应该能够看到目标之间的执行流程。

这绝对是我们在一个 shell 脚本中缺少的东西；为了使 shell 脚本像这样工作，我们需要很多控制流机制（循环、条件等等）。Makefile 比 shell 脚本更简洁，更声明式，这就是我们使用它的原因。我们只想声明需要构建的内容，而不需要知道构建路径。虽然使用 Make 并不能完全实现这一点，但它是一个拥有完整功能的构建系统的起点。

Makefile 中目标的另一个特性是，如果它们引用的是磁盘上的文件或目录，例如`out/multiply.o`，`make`程序会检查该文件或目录的最新修改，如果没有自上次构建以来的修改，它将跳过该目标。这也适用于`out/multiply.o`的依赖项`calc/multiply.c`。如果源文件`calc/multiply.c`最近没有更改并且已经编译过，再次编译它就没有意义了。这又是一个你不能仅仅通过编写 shell 脚本就能获得的功能。

通过这个特性，你只编译自上次构建以来已修改的源文件，这大大减少了自上次构建以来未更改的源文件的编译量。当然，这个特性在至少编译整个项目一次之后才会工作。之后，只有修改过的源文件才会触发编译或链接。

前面的 Makefile 中还有一个关键点，即`calc/calc.h`目标。正如你所见，有多个目标，主要是源文件，依赖于头文件`calc/calc.h`。因此，根据我们之前解释的功能，对头文件的简单修改可以触发依赖于该头文件的源文件的多次编译。

这正是我们试图在源文件中只包含所需的头文件，并在可能的情况下使用前向声明而不是包含的原因。前向声明通常不在源文件中制作，因为在那里，我们通常需要访问结构或函数的实际定义，但在头文件中可以轻松完成。

头文件之间有很多依赖通常会导致构建灾难。即使是包含在许多其他头文件中，最终被许多源文件包含的一个头文件的微小修改，也可能触发整个项目或类似规模的构建。这将有效降低开发质量，并导致开发者需要在构建之间等待数分钟。

前面的 Makefile 仍然过于冗长。每当我们添加一个新的源文件时，我们必须更改目标。我们期望在添加新的源文件时更改 Makefile，而不是通过添加新的目标并改变 Makefile 的整体结构。这实际上阻止了我们重用相同的 Makefile 在另一个类似当前项目的项目中。

更重要的是，许多目标遵循相同的模式，我们可以利用 Make 中可用的*模式匹配*功能来减少目标数量，并在 Makefile 中编写更少的代码。这是 Make 的另一个超级特性，其效果你很难通过编写 shell 脚本轻易实现。

以下 Makefile 将是本项目中的最后一个，但仍然不是一位 Make 专家能写出的最佳 Makefile：

```cpp
BUILD_DIR = out
OBJ = ${BUILD_DIR}/calc/add.o \
                ${BUILD_DIR}/calc/subtract.o \
                ${BUILD_DIR}/calc/multiply.o \
                ${BUILD_DIR}/exec/main.o
CC = gcc
HEADER_DIRS = -Icalc
LIBCALCNAME = calc
LIBCALC = ${BUILD_DIR}/lib${LIBCALCNAME}.a
EXEC = ${BUILD_DIR}/ex23_1.out
build: prereq ${BUILD_DIR}/exec/main.o ${LIBCALC}
    ${CC} -L${BUILD_DIR} -l${LIBCALCNAME} ${BUILD_DIR}/exec/main.o -o ${EXEC}
prereq:
    mkdir -p ${BUILD_DIR}
    mkdir -p ${BUILD_DIR}/calc
    mkdir -p ${BUILD_DIR}/exec
${LIBCALC}: ${OBJ}
    ar rcs ${LIBCALC} ${OBJ}
${BUILD_DIR}/calc/%.o: calc/%.c
    ${CC} -c ${HEADER_DIRS} $< -o $@
${BUILD_DIR}/exec/%.o: exec/%.c
    ${CC} -c ${HEADER_DIRS} $< -o $@
clean: ${BUILD_DIR}
    rm -rf ${BUILD_DIR}
```

代码框 23-3 [Makefile-by-pattern]：为针对目标项目编写的新 Makefile，使用模式匹配

前面的 Makefile 在其目标中使用模式匹配。变量`OBJ`保存预期可重定位目标文件的列表，并在需要目标文件列表的所有其他地方使用。

这不是一本关于 Make 的模式匹配如何工作的书，但你可以看到，有一些通配符，如`%`、`$<`和`$@`，在模式中使用。

运行前面的 Makefile 将产生与其他 Makefile 相同的结果，但我们可以从 Make 提供的各种优秀功能中受益，并最终拥有一个可重用和维护的 Make 脚本。

下面的 shell 框展示了如何运行前面的 Makefile 以及输出结果：

```cpp
$ make
mkdir -p out
mkdir -p out/calc
mkdir -p out/exec
gcc -c -Icalc exec/main.c -o out/exec/main.o
gcc -c -Icalc calc/add.c -o out/calc/add.o
gcc -c -Icalc calc/subtract.c -o out/calc/subtract.o
gcc -c -Icalc calc/multiply.c -o out/calc/multiply.o
ar rcs out/libcalc.a out/calc/add.o out/calc/subtract.o out/calc/multiply.o out/exec/main.o
gcc -Lout -lcalc out/exec/main.o -o out/ex23_1.out
$
```

Shell 框 23-5：使用最终 Makefile 构建目标项目

在接下来的章节中，我们将讨论 CMake，这是一个用于生成真正的 Makefiles 的出色工具。事实上，在 Make 变得流行之后，新一代的构建工具出现了，*构建脚本生成器*，可以从给定的描述中生成 Makefiles 或其他构建系统的脚本。CMake 就是其中之一，它可能是最受欢迎的。

**注意**：

这里是阅读更多关于 GNU Make 的主要链接，它是为 GNU 项目制作的 Make 的实现：[GNU Make：https://www.gnu.org/software/make/manual/html_node/index.html](https://www.gnu.org/software/make/manual/html_node/index.html)/html_node/index.html。

# CMake – 并非一个构建系统！

CMake 是一个构建脚本生成器，并作为其他构建系统（如 Make 和 Ninja）的生成器。编写有效的跨平台 Makefiles 是一项繁琐且复杂的工作。CMake 或类似工具，如*Autotools*，被开发出来以提供精心调校的跨平台构建脚本，如 Makefiles 或 Ninja 构建文件。请注意，Ninja 是另一个构建系统，将在下一节中介绍。

**注意**：

你可以在这里阅读更多关于 Autotools 的信息：[Autotools：https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html)。

依赖管理也很重要，这不是通过 Makefiles 实现的。这些生成工具还可以检查已安装的依赖项，如果系统中缺少所需的依赖项，则不会生成构建脚本。检查编译器和它们的版本，以及找到它们的位置、它们支持的功能等等，都是这些工具在生成构建脚本之前所做的工作的一部分。

类似于 Make，它会寻找名为`Makefile`的文件，CMake 会寻找名为`CMakeLists.txt`的文件。无论你在项目中找到这个文件的位置，都意味着 CMake 可以使用来生成适当的 Makefiles。幸运的是，与 Make 不同，CMake 支持嵌套模块。换句话说，你可以在其他目录中拥有多个`CMakeLists.txt`文件作为项目的一部分，并且只需在项目根目录中运行 CMake，就可以找到它们并为它们生成适当的 Makefiles。

让我们通过添加 CMake 支持到我们的示例项目来继续本节。为此，我们添加了三个`CMakeLists.txt`文件。接下来，你可以看到添加这些文件后项目的层次结构：

```cpp
$ tree ex23_1
ex23_1/
├── CMakeLists.txt
├── calc
│   ├── CMakeLists.txt
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    ├── CMakeLists.txt
    └── main.c
2 directories, 8 files
$
```

Shell Box 23-6：引入三个 CMakeLists.txt 文件后的项目层次结构

如你所见，我们有三个`CMakeLists.txt`文件：一个在根目录中，一个在`calc`目录中，另一个在`exec`目录中。下面的代码框显示了在根目录中找到的`CMakeLists.txt`文件的内容。如你所见，它添加了`calc`和`exec`的子目录。

这些子目录必须包含一个`CMakeLists.txt`文件，实际上，根据我们的设置，它们确实包含：

```cpp
cmake_minimum_required(VERSION 3.8)
include_directories(calc)
add_subdirectory(calc)
add_subdirectory(exec)
```

代码框 23-4 [CMakeLists.txt]: 在项目根目录中找到的 CMakeLists.txt 文件

前面的 CMake 文件将 `calc` 目录添加到编译源文件时编译器将使用的 `include` 目录中。正如我们之前所说的，它还添加了两个子目录：`calc` 和 `exec`。这些目录有自己的 `CMakeLists.txt` 文件，解释了如何编译它们的内容。以下是在 `calc` 目录中找到的 `CMakeLists.txt` 文件：

```cpp
add_library(calc STATIC
  add.c
  subtract.c
  multiply.c
)
```

代码框 23-5 [calc/CMakeLists.txt]: 在 calc 目录中找到的 CMakeLists.txt 文件

如您所见，它只是一个简单的 *目标声明*，针对 `calc` 目标，这意味着我们需要一个名为 `calc` 的静态库（实际上在构建后为 `libcalc.a`），该库应包含对应于源文件 `add.c`、`subtract.c` 和 `multiply.c` 的可重定位目标文件。请注意，CMake 目标通常代表代码库的最终产品。因此，对于 `calc` 模块，我们只有一个产品，即一个静态库。

如您所见，对于 `calc` 目标没有指定其他内容。例如，我们没有指定静态库的扩展名或库的文件名（尽管我们可以）。构建此模块所需的所有其他配置要么是从父 `CMakeLists.txt` 文件继承的，要么是从 CMake 本身的默认配置中获得的。

例如，我们知道在 Linux 和 macOS 上共享对象文件的扩展名不同。因此，如果目标是共享库，就没有必要在目标声明中指定扩展名。CMake 能够处理这种非常平台特定的差异，并且最终共享对象文件将根据构建的平台具有正确的扩展名。

以下 `CMakeLists.txt` 文件是在 `exec` 目录中找到的：

```cpp
add_executable(ex23_1.out
  main.c
)
target_link_libraries(ex23_1.out
  calc
)
```

代码框 23-6 [exec/CMakeLists.txt]: 在 exec 目录中找到的 CMakeLists.txt 文件

如您所见，前面 `CMakeLists.txt` 中声明的目标是可执行文件，并且它应该链接到另一个 `CMakeLists.txt` 文件中已经声明的 `calc` 目标。

这实际上给了您在项目的某个角落创建库，并在另一个角落使用它们的能力，只需编写一些指令即可。

现在是时候向您展示如何根据根目录中找到的 `CMakeLists.txt` 文件生成 Makefile。请注意，我们在名为 `build` 的单独目录中这样做，以便将生成的可重定位和最终目标文件与实际源文件保持分离。

如果您使用的是 **源代码管理**（**SCM**）系统如 *git*，您可以忽略 `build` 目录，因为它应该在每个平台上单独生成。唯一重要的文件是 `CMakeLists.txt` 文件，这些文件始终保存在源代码控制仓库中。

以下 shell box 演示了如何为根目录中找到的`CMakeLists.txt`文件生成构建脚本（在这种情况下，是一个 Makefile）：

```cpp
$ cd ex23_1
$ mkdir -p build
$ cd build
$ rm -rfv *
...
$ cmake ..
-- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: .../extreme_c/ch23/ex23_1/build
$
```

Shell Box 23-7：基于根目录中找到的 CMakeLists.txt 文件生成 Makefile

如您从输出中看到的，CMake 命令已经能够检测到工作编译器、它们的 ABI 信息（关于 ABI 的更多信息，请参阅*第三章*，*目标文件*）、它们的功能等等，最后它在`build`目录中生成了一个 Makefile。

**注意**：

在*Shell Box 23-7*中，我们假设我们可以在`build`目录中；因此，我们首先删除了其所有内容。

您可以看到`build`目录的内容和生成的 Makefile：

```cpp
$ ls
CMakeCache.txt  CMakeFiles  Makefile  calc  cmake_install.cmake  exec
$
```

Shell Box 23-8：在 build 目录中生成的 Makefile

现在您已经在`build`目录中有了 Makefile，您可以自由地运行 make 命令。它将负责编译，并为您优雅地显示进度。

注意，在运行`make`命令之前，您应该在`build`目录中：

```cpp
$ make
Scanning dependencies of target calc
[ 16%] Building C object calc/CMakeFiles/calc.dir/add.c.o
[ 33%] Building C object calc/CMakeFiles/calc.dir/subtract.c.o
[ 50%] Building C object calc/CMakeFiles/calc.dir/multiply.c.o
[ 66%] Linking C static library libcalc.a
[ 66%] Built target calc
Scanning dependencies of target ex23_1.out
[ 83%] Building C object exec/CMakeFiles/ex23_1.out.dir/main.c.o
[100%] Linking C executable ex23_1.out
[100%] Built target ex23_1.out
$
```

Shell Box 23-9：执行生成的 Makefile

目前，许多大型项目使用 CMake，您可以使用我们在之前的 shell boxes 中展示的大致相同的命令来构建它们的源代码。"Vim"就是这样一个项目。甚至 CMake 本身也是在使用由 Autotools 构建的最小 CMake 系统之后，用 CMake 构建的！CMake 现在有很多版本和功能，要详细讨论它们需要一本书。

**注意**：

以下链接是最新版本 CMake 的官方文档，它可以帮助您了解它[如何工作以及它有哪些功能：https://cmake.](https://cmake.org/cmake/help/latest/index.html)org/cmake/help/latest/index.html。

在本节的最后，CMake 可以为 Microsoft Visual Studio、Apple 的 Xcode 和其他开发环境创建构建脚本文件。

在下一节中，我们将讨论 Ninja 构建系统，这是一个比 Make 更快的替代方案，最近正逐渐流行起来。我们还将解释如何使用 CMake 生成 Ninja 构建脚本文件而不是 Makefile。

# Ninja

Ninja 是 Make 的替代品。我犹豫是否称它为替代品，但它是更快的替代品。它通过移除 Make 提供的一些功能（如字符串操作、循环和模式匹配）来实现高性能。

通过移除这些功能，Ninja 减少了开销，因此从头开始编写 Ninja 构建脚本并不是明智之举。

编写 Ninja 脚本可以与编写 shell 脚本相比较，我们之前章节中解释了其缺点。这就是为什么建议与 CMake 这样的构建脚本生成工具一起使用。

在本节中，我们将展示当 Ninja 构建脚本由 CMake 生成时如何使用 Ninja。因此，在本节中，我们不会像对 Makefile 那样介绍 Ninja 的语法。这是因为我们不会自己编写它们；相反，我们将要求 CMake 为我们生成它们。

**注意**：

想了解更多关于 Ninja 语法的知识，请点击此链接：[`ninja-build.org/manual.html#_writing_your_own_ninja_files`](https://ninja-build.org/manual.html#_writing_your_own_ninja_files).

正如我们之前所解释的，最好使用构建脚本生成器来生成 Ninja 构建脚本文件。在下面的 shell 框中，您可以查看如何使用 CMake 生成 Ninja 构建脚本，`build.ninja`，而不是为我们的目标项目生成 Makefile：

```cpp
$ cd ex23_1
$ mkdir -p build
$ cd build
$ rm -rfv *
...
$ cmake -GNinja ..
-- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: .../extreme_c/ch23/ex23_1/build
$
```

Shell Box 23-10：基于根目录中找到的 CMakeLists.txt 生成 build.ninja

如您所见，我们已经传递了选项`-GNinja`来让 CMake 知道我们要求的是 Ninja 构建脚本文件而不是 Makefile。CMake 生成`build.ninja`文件，您可以在以下`build`目录中找到它：

```cpp
$ ls
CMakeCache.txt  CMakeFiles  build.ninja  calc  cmake_install.cmake  exec  rules.ninja
$
```

Shell Box 23-11：在 build 目录中生成的 build.ninja

要编译项目，只需运行以下`ninja`命令即可。请注意，就像`make`程序在当前目录中查找`Makefile`一样，`ninja`程序在当前目录中查找`build.ninja`：

```cpp
$ ninja
[6/6] Linking C executable exec/ex23_1.out
$
```

Shell Box 23-12：执行生成的 build.ninja

在以下部分，我们将讨论*Bazel*，这是另一个可以用于构建 C 和 C++项目的构建系统。

# Bazel

Bazel 是 Google 开发的一个构建系统，旨在解决内部需要有一个快速且可扩展的构建系统，无论编程语言是什么，都能构建任何项目。Bazel 支持构建 C、C++、Java、Go 和 Objective-C 项目。不仅如此，它还可以用于构建 Android 和 iOS 项目。

Bazel 大约在 2015 年成为开源软件。它是一个构建系统，因此它可以与 Make 和 Ninja 进行比较，但不能与 CMake 相比。几乎所有的 Google 开源项目都使用 Bazel 进行构建。例如，我们可以提到*Bazel*本身、*gRPC*、*Angular*、*Kubernetes*和*TensorFlow*。

Bazel 是用 Java 编写的。它以并行和可扩展的构建而闻名，在大项目中确实能带来很大的差异。Make 和 Ninja 都支持并行构建，通过传递`-j`选项（Ninja 默认是并行的）。

**注意**：

Bazel 的官方文档可以在以下链接找到：[`docs.bazel.build/versions/master/bazel-overview.html`](https://docs.bazel.build/versions/master/bazel-overview.html).

使用 Bazel 的方式与我们对 Make 和 Ninja 所做的方式类似。Bazel 需要在一个项目中存在两种类型的文件：`WORKSPACE` 和 `BUILD` 文件。`WORKSPACE` 文件应该在根目录中，而 `BUILD` 文件应该放入作为同一工作区（或项目）一部分的模块中。这在大约程度上类似于 CMake 的情况，我们有三份 `CMakeLists.txt` 文件分布在项目中，但请注意，在这里，Bazel 本身是构建系统，我们不会为另一个构建系统生成任何构建脚本。

如果我们想将 Bazel 支持添加到我们的项目中，我们应该在项目中获得以下层次结构：

```cpp
$ tree ex23_1
ex23_1/
├── WORKSPACE
├── calc
│   ├── BUILD
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    ├── BUILD
    └── main.c
2 directories, 8 files
$
```

Shell 框 23-13：引入 Bazel 文件后的项目层次结构

在我们的示例中，`WORKSPACE` 文件的内容将是空的。它通常用于指示代码库的根目录。请注意，如果您有更多嵌套和更深的模块，您需要参考文档以了解这些文件（`WORKSPACE` 和 `BUILD`）应该如何在整个代码库中传播。

`BUILD` 文件的内容表明了在该目录（或模块）中应该构建的目标。以下代码框显示了 `calc` 模块的 `BUILD` 文件：

```cpp
c_library(
  name = "calc",
  srcs = ["add.c", "subtract.c", "multiply.c"],
  hdrs = ["calc.h"],
  linkstatic = True,
  visibility = ["//exec:__pkg__"]
)
```

代码框 23-7 [calc/BUILD]：calc 目录中找到的 BUILD 文件

正如您所看到的，一个新的目标 `calc` 被声明。它是一个静态库，包含目录中找到的三个源文件。该库对 `exec` 目录中的目标也是可见的。

让我们看看 exec 目录中的 `BUILD` 文件：

```cpp
cc_binary(
  name = "ex23_1.out",
  srcs = ["main.c"],
  deps = [
    "//calc:calc"
  ],
  copts = ["-Icalc"]
)
```

代码框 23-8 [exec/BUILD]：exec 目录中找到的 BUILD 文件

在这些文件放置到位后，我们现在可以运行 Bazel 并构建项目。您需要进入项目的根目录。请注意，与 CMake 一样，我们不需要有构建目录：

```cpp
$ cd ex23_1
$ bazel build //...
INFO: Analyzed 2 targets (14 packages loaded, 71 targets configured).
INFO: Found 2 targets...
INFO: Elapsed time: 1.067s, Critical Path: 0.15s
INFO: 6 processes: 6 linux-sandbox.
INFO: Build completed successfully, 11 total actions
$
```

Shell 框 23-14：使用 Bazel 构建示例项目

现在，如果您查看根目录中找到的 `bazel-bin` 目录，您应该能够找到产品：

```cpp
$ tree bazel-bin
bazel-bin
├── calc
│   ├── _objs
│   │   └── calc
│   │       ├── add.pic.d
│   │       ├── add.pic.o
│   │       ├── multiply.pic.d
│   │       ├── multiply.pic.o
│   │       ├── subtract.pic.d
│   │       └── subtract.pic.o
│   ├── libcalc.a
│   └── libcalc.a-2.params
└── exec
    ├── _objs
    │   └── ex23_1.out
    │       ├── main.pic.d
    │       └── main.pic.o
    ├── ex23_1.out
    ├── ex23_1.out-2.params
    ├── ex23_1.out.runfiles
    │   ├── MANIFEST
    │   └── __main__
    │       └── exec
    │           └── ex23_1.out -> .../bin/exec/ex23_1.out
    └── ex23_1.out.runfiles_manifest
9 directories, 15 files
$
```

Shell 框 23-15：构建后 bazel-bin 的内容

如您在前面的列表中所见，项目已成功构建，产品已被定位。

在下一节中，我们将结束本章的讨论，并比较现有的 C 和 C++ 项目构建系统。

# 比较构建系统

在本章中，我们尝试介绍了三种最著名和最广泛使用的构建系统。我们还介绍了 CMake 作为构建脚本生成器。您应该知道还有其他构建系统可以用于构建 C 和 C++ 项目。

请注意，您对构建系统的选择应被视为一项长期承诺；如果您以特定的构建系统开始一个项目，将其更改为另一个系统将需要大量的努力。

构建系统可以根据各种属性进行比较。依赖管理、能够处理复杂的嵌套项目层次结构、构建速度、可扩展性、与现有服务的集成、添加新逻辑的灵活性等等，都可以用来进行公平的比较。我不会用构建系统的比较来结束这本书，因为这是一项繁琐的工作，而且更重要的是，已经有了一些关于这个主题的优秀在线文章。

在 Bitbucket 上有一个很好的 Wiki 页面，对可用的构建系统进行了优缺点比较，以及构建脚本生成系统，可以在这里找到：[`bitbucket.org/scons/scons/wiki/SconsVsOtherBuildTools`](https://bitbucket.org/scons/scons/wiki/SconsVsOtherBuildTools)。

注意，比较的结果可能因人而异。你应该根据你项目的需求和可用的资源来选择构建系统。以下链接提供了可用于进一步研究和比较的补充资源：[`www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/`](https://www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/)

[`www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/`](https://www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/)[i](https://github.com/LoopPerfect/buckaroo/wiki/Build-Systems-Comparison)[ew_of_build_systems_mostly_for_c_projects/](https://github.com/LoopPerfect/buckaroo/wiki/Build-Systems-Comparison)

[`github.com/LoopPer`](https://github.com/LoopPerfect/buckaroo/wiki/Build-Systems-Comparison)f[ect/buckaroo/wiki/Build-Systems-Comparison](https://medium.com/@julienjorge/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444)

[`medium.com/@julienjorge/an-overview-of-build`](https://medium.com/@julienjorge/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444)-systems-mostly-for-c-projects-ac9931494444

# 摘要

在本章中，我们讨论了用于构建 C 或 C++项目的常用构建工具。作为本章的一部分：

+   我们讨论了构建系统的必要性。

+   我们介绍了 Make，这是可用于 C 和 C++项目的最古老的构建系统之一。

+   我们介绍了 Autotools 和 CMake，两种著名的构建脚本生成器。

+   我们展示了如何使用 CMake 生成所需的 Makefiles。

+   我们讨论了 Ninja，并展示了如何使用 CMake 生成 Ninja 构建脚本。

+   我们展示了如何使用 Bazel 构建 C 项目。

+   最后，我们提供了一些链接，指向关于各种构建系统比较的在线讨论。

# 结语

最后的话 ...

如果你正在阅读这篇文档，这意味着我们的旅程已经结束！作为本书的一部分，我们探讨了几个主题和概念，我希望这次旅程能让你成为一个更好的 C 程序员。当然，它不能给你带来经验；你必须通过参与各种项目来获得。本书中讨论的方法和技巧将提升你的专业水平，这将使你能够参与更复杂的项目。现在你对软件系统有了更深入的了解，从更广阔的角度来看，并且对内部运作有了顶尖的知识。

尽管这本书比你的常规阅读更厚重、更长，但它仍然无法涵盖 C、C++和系统编程中所有的话题。因此，我肩上仍然有一份责任；旅程还没有结束！我希望能继续研究更多极端话题，也许更具体的领域，比如异步 I/O、高级数据结构、套接字编程、分布式系统、内核开发和函数式编程，在适当的时候。

希望在下次旅程中再次见到你！

Kamran
