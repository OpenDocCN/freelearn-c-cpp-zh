# 构建 C++应用程序的简介

编程语言通过其程序执行模型而有所不同；最常见的是解释型语言和编译型语言。编译器将源代码转换为机器代码，计算机可以在没有中介支持系统的情况下运行。另一方面，解释型语言代码需要支持系统、解释器和虚拟环境才能工作。

C++是一种编译型语言，使得程序运行速度比解释型语言更快。虽然 C++程序应该为每个平台进行编译，但解释型程序可以跨平台操作。

我们将讨论程序构建过程的细节，从编译器处理源代码的阶段开始，到可执行文件的细节（编译器的输出）结束。我们还将了解为什么为一个平台构建的程序在另一个平台上无法运行。

本章将涵盖以下主题：

+   C++20 简介

+   C++预处理器的细节

+   源代码编译的底层细节

+   理解链接器及其功能

+   可执行文件的加载和运行过程

# 技术要求

使用选项`-std=c++2a`的 g++编译器用于编译本章中的示例。您可以在本章中找到使用的源文件[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# C++20 简介

C++经过多年的发展，现在有了全新的版本，C++20。自 C++11 以来，C++标准大大扩展了语言特性集。让我们来看看新的 C++20 标准中的显著特性。

# 概念

概念是 C++20 中的一个重要特性，它为类型提供了一组要求。概念背后的基本思想是对模板参数进行编译时验证。例如，要指定模板参数必须具有默认构造函数，我们可以如下使用`default_constructible`概念：

```cpp
template <default_constructible T>
void make_T() { return T(); }
```

在上述代码中，我们错过了`typename`关键字。相反，我们设置了描述`template`函数的`T`参数的概念。

我们可以说概念是描述其他类型的类型 - 元类型，可以这么说。它们允许在类型属性的基础上对模板参数进行编译时验证以及函数调用。我们将在第三章和第四章中详细讨论概念，*面向对象编程的细节*和*理解和设计模板*。

# 协程

协程是特殊的函数，能够在任何定义的执行点停止并稍后恢复。协程通过以下新关键字扩展了语言：

+   `co_await` 暂停协程的执行。

+   `co_yield` 暂停协程的执行，同时返回一个值。

+   `co_return` 类似于常规的`return`关键字；它结束协程并返回一个值。看看以下经典示例：

```cpp
generator<int> step_by_step(int n = 0) {
  while (true) {
    co_yield n++;
  }
}
```

协程与`promise`对象相关联。`promise`对象存储和警报协程的*状态*。我们将在第八章中深入讨论协程，*并发和多线程*。

# 范围

`ranges`库提供了一种新的处理元素范围的方式。要使用它们，您应该包含`<ranges>`头文件。让我们通过一个例子来看`ranges`。范围是具有开始和结束的元素序列。它提供了一个`begin`迭代器和一个`end`哨兵。考虑以下整数向量：

```cpp
import <vector>

int main()
{
  std::vector<int> elements{0, 1, 2, 3, 4, 5, 6};
}
```

范围和范围适配器（`|`运算符）提供了处理一系列元素的强大功能。例如，查看以下代码：

```cpp
import <vector>
import <ranges>

int main()
{
  std::vector<int> elements{0, 1, 2, 3, 4, 5, 6};
  for (int current : elements | ranges::view::filter([](int e) { return 
   e % 2 == 0; }))
  {
    std::cout << current << " ";
  }
}
```

在前面的代码中，我们使用`ranges::view::filter()`过滤了偶数整数的范围。注意应用于元素向量的范围适配器`|`。我们将在第七章中讨论范围及其强大的功能，*函数式编程*。

# 更多的 C++20 功能

C++20 是 C++语言的一个新的重大发布。它包含许多使语言更复杂和灵活的功能。**概念**、**范围**和**协程**是本书中将讨论的许多功能之一。

最受期待的功能之一是**模块**，它提供了在模块内声明模块并导出类型和值的能力。您可以将模块视为带有现在多余的包含保护的头文件的改进版本。我们将在本章中介绍 C++20 模块。

除了 C++20 中添加的显着功能之外，还有一系列其他功能，我们将在整本书中讨论：

+   太空船操作符：`operator<=>()`。现在可以通过利用`operator<=>()`来控制运算符重载的冗长。

+   `constexpr`在语言中占据越来越多的空间。C++20 现在有了`consteval`函数，`constexpr std::vector`和`std::string`，以及许多其他功能。

+   数学常数，例如`std::number::pi`和`std::number::log2e`。

+   线程库的重大更新，包括停止令牌和加入线程。

+   迭代器概念。

+   移动视图和其他功能。

为了更好地理解一些新功能，并深入了解语言的本质，我们将从以前的版本开始介绍语言的核心。这将帮助我们找到比旧版本更好的新功能的用途，并且还将有助于支持旧版 C++代码。现在让我们开始了解 C++应用程序的构建过程。

# 构建和运行程序

您可以使用任何文本编辑器来编写代码，因为最终，代码只是文本。要编写代码，您可以自由选择简单的文本编辑器，如*Vim*，或者高级的**集成开发环境**（**IDE**），如*MS Visual Studio*。情书和源代码之间唯一的区别是后者可能会被称为**编译器**的特殊程序解释（而情书无法编译成程序，它可能会让您心跳加速）。

为了区分纯文本文件和源代码，使用了特殊的文件扩展名。C++使用`.cpp`和`.h`扩展名（您可能偶尔也会遇到`.cxx`和`.hpp`）。在深入细节之前，将编译器视为将源代码转换为可运行程序（称为可执行文件或**可执行文件**）的工具。从源代码生成可执行文件的过程称为**编译**。编译 C++程序是一系列复杂任务的序列，最终产生机器代码。**机器代码**是计算机的本机语言，这就是为什么它被称为机器代码。

通常，C++编译器会解析和分析源代码，然后生成中间代码，对其进行优化，最后生成一个名为**目标文件**的机器代码文件。您可能已经遇到过目标文件；它们在 Linux 中有单独的扩展名`.o`，在 Windows 中有单独的扩展名`.obj`。创建的目标文件包含不仅可以由计算机运行的机器代码。编译通常涉及多个源文件，编译每个源文件会产生一个单独的目标文件。然后，这些目标文件由一个称为**链接器**的工具链接在一起，形成一个单独的可执行文件。链接器使用存储在目标文件中的附加信息来正确地链接它们（链接将在本章后面讨论）。

以下图表描述了程序构建的阶段：

![](img/c5d04f78-da90-4c83-88ac-d152f1e9d2b1.png)

C++应用程序构建过程包括三个主要步骤：**预处理**、**编译**和**链接**。所有这些步骤都使用不同的工具完成，但现代编译器将它们封装在一个单一的工具中，为程序员提供了一个更简单的接口。

生成的可执行文件保存在计算机的硬盘上。为了运行它，应将其复制到主内存 RAM 中。复制由另一个名为**加载器**的工具完成。加载器是操作系统的一部分，它知道应从可执行文件的内容中复制什么和复制到哪里。将可执行文件加载到主内存后，原始可执行文件不会从硬盘中删除。

程序的加载和运行由操作系统（OS）完成。操作系统管理程序的执行，优先级高于其他程序，在完成后卸载程序等。程序的运行副本称为进程。进程是可执行文件的一个实例。

# 理解预处理

**预处理器**旨在处理源文件，使其准备好进行编译。预处理器使用预处理器**指令**，如`#define`、`#include`等。指令不代表程序语句，而是预处理器的命令，告诉它如何处理源文件的文本。编译器无法识别这些指令，因此每当在代码中使用预处理器指令时，预处理器会在实际编译代码之前相应地解析它们。例如，编译器开始编译之前，以下代码将被更改：

```cpp
#define NUMBER 41 
int main() { 
  int a = NUMBER + 1; 
  return 0; 
}
```

使用`#define`指令定义的所有内容都称为**宏**。经过预处理后，编译器以这种形式获得转换后的源代码：

```cpp
int main() { 
  int a = 41 + 1; 
  return 0;
}
```

如前所述，预处理器只是处理文本，不关心语言规则或其语法。特别是使用宏定义的预处理器指令，如前面的例子中的`#define NUMBER 41`，除非你意识到预处理器只是简单地将`NUMBER`的任何出现替换为`41`，而不将`41`解释为整数。对于预处理器来说，以下行都是有效的：

```cpp
int b = NUMBER + 1; 
struct T {}; // user-defined type 
T t = NUMBER; // preprocessed successfully, but compile error 
```

这将产生以下代码：

```cpp
int b = 41 + 1
struct T {};
T t = 41; // error line
```

当编译器开始编译时，它会发现赋值`t = 41`是错误的，因为从'int'到'T'没有可行的转换。

甚至使用在语法上正确但存在逻辑错误的宏也是危险的：

```cpp
#define DOUBLE_IT(arg) (arg * arg) 
```

预处理器将任何`DOUBLE_IT(arg)`的出现替换为`(arg * arg)`，因此以下代码将输出`16`：

```cpp
int st = DOUBLE_IT(4);
std::cout << st;
```

编译器将接收到这段代码：

```cpp
int st = (4 * 4);
std::cout << st;
```

当我们将复杂表达式用作宏参数时会出现问题：

```cpp
int bad_result = DOUBLE_IT(4 + 1); 
std::cout << bad_result;
```

直观上，这段代码将产生`25`，但事实上预处理器只是进行文本处理，而在这种情况下，它会这样替换宏：

```cpp
int bad_result = (4 + 1 * 4 + 1);
std::cout << bad_result;
```

这将输出`9`，显然`9`不是`25`。

要修复宏定义，需要用额外的括号括住宏参数：

```cpp
#define DOUBLE_IT(arg) ((arg) * (arg)) 
```

现在表达式将采用这种形式：

```cpp
int bad_result = ((4 + 1) * (4 + 1)); 
```

强烈建议在适用的地方使用`const`声明，而不是宏定义。

一般来说，应避免使用宏定义。宏容易出错，而 C++提供了一组构造，使得宏的使用已经过时。

如果我们使用`constexpr`函数，同样的前面的例子将在编译时进行类型检查和处理：

```cpp
constexpr int double_it(int arg) { return arg * arg; } 
int bad_result = double_it(4 + 1); 
```

使用`constexpr`限定符使得能够在编译时评估函数的返回值（或变量的值）。使用`const`变量重新编写`NUMBER`定义的示例会更好：

```cpp
const int NUMBER = 41; 
```

# 头文件

预处理器最常见的用法是`#include`指令，用于在源代码中包含头文件。头文件包含函数、类等的定义：

```cpp
// file: main.cpp 
#include <iostream> 
#include "rect.h"
int main() { 
  Rect r(3.1, 4.05) 
  std::cout << r.get_area() << std::endl;
}
```

假设头文件 `rect.h` 定义如下：

```cpp
// file: rect.h
struct Rect  
{
private:
  double side1_;
  double side2_;
public:
  Rect(double s1, double s2);
  const double get_area() const;
};
```

实现包含在 `rect.cpp` 中：

```cpp
// file: rect.cpp
#include "rect.h"

Rect::Rect(double s1, double s2)
  : side1_(s1), side2_(s2)
{}

const double Rect::get_area() const {
  return side1_ * side2_;
}
```

预处理器检查 `main.cpp` 和 `rect.cpp` 后，将用 `main.cpp` 的 `#include` 指令替换为 `iostream` 和 `rect.h` 的相应内容，用 `rect.cpp` 的 `#include` 指令替换为 `rect.h`。C++17 引入了 `__has_include` 预处理器常量表达式。如果找到指定名称的文件，`__has_include` 的值为 `1`，否则为 `0`：

```cpp
#if __has_include("custom_io_stream.h")
#include "custom_io_stream.h"
#else
#include <iostream>
#endif
```

在声明头文件时，强烈建议使用所谓的*包含保护*（`#ifndef, #define, #endif`）来避免双重声明错误。我们将很快介绍这种技术。这些又是预处理指令，允许我们避免以下情况：`Square` 类在 `square*.*h` 中定义，它包含 `rect.h` 以从 `Rect` 派生 `Square`：

```cpp
// file: square.h
#include "rect.h"
struct Square : Rect {
  Square(double s);
};
```

在 `main.cpp` 中包含 `square.h` 和 `rect.h` 会导致 `rect.h` 被包含两次：

```cpp
// file: main.cpp
#include <iostream> 
#include "rect.h" 
#include "square.h"
/* 
  preprocessor replaces the following with the contents of square.h
*/
// code omitted for brevity
```

预处理后，编译器将以以下形式接收 `main.cpp`：

```cpp
// contents of the iostream file omitted for brevity 
struct Rect {
  // code omitted for brevity
};
struct Rect {
  // code omitted for brevity
};
struct Square : Rect {
  // code omitted for brevity
};
int main() {
  // code omitted for brevity
}
```

然后编译器会产生一个错误，因为它遇到了两个类型为 `Rect` 的声明。头文件应该通过使用包含保护来防止多次包含，方法如下：

```cpp
#ifndef RECT_H 
#define RECT_H 
struct Rect { ... }; // code omitted for brevity  
#endif // RECT_H 

```

当预处理器第一次遇到头文件时，`RECT_H` 未定义，`#ifndef` 和 `#endif` 之间的所有内容都将被相应处理，包括 `RECT_H` 的定义。当预处理器在同一源文件中第二次包含相同的头文件时，它将省略内容，因为 `RECT_H` 已经被定义。

这些包含保护是控制源文件部分编译的指令的一部分。所有条件编译指令都是 `#if`, `#ifdef`, `#ifndef`, `#else`, `#elif`, 和 `#endif`。

条件编译在许多情况下都很有用；其中之一是在所谓的**调试**模式下记录函数调用。在发布程序之前，建议调试程序并测试逻辑缺陷。您可能想要查看在调用某个函数后代码中发生了什么，例如：

```cpp
void foo() {
  log("foo() called");
  // do some useful job
}
void start() {
  log("start() called");
  foo();
  // do some useful job
}
```

每个函数调用 `log()` 函数，其实现如下：

```cpp
void log(const std::string& msg) {
#if DEBUG
  std::cout << msg << std::endl;
#endif
}
```

如果定义了 `DEBUG`，`log()` 函数将打印 `msg`。如果编译项目时启用了 `DEBUG`（使用编译器标志，如 g++ 中的 `-D`），那么 `log()` 函数将打印传递给它的字符串；否则，它将什么也不做。

# 在 C++20 中使用模块

模块修复了头文件中令人讨厌的包含保护问题。我们现在可以摆脱预处理宏。模块包括两个关键字，`import` 和 `export`。要使用一个模块，我们使用 `import`。要声明一个模块及其导出的属性，我们使用 `export`。在列出使用模块的好处之前，让我们看一个简单的使用示例。以下代码声明了一个模块：

```cpp
export module test;

export int twice(int a) { return a * a; }
```

第一行声明了名为 `test` 的模块。接下来，我们声明了 `twice()` 函数并将其设置为 `export`。这意味着我们可以有未导出的函数和其他实体，因此它们在模块外部将是私有的。通过导出实体，我们将其设置为模块用户的 `public`。要使用 `module`，我们像以下代码中那样导入它：

```cpp
import test;

int main()
{
  twice(21);
}
```

模块是 C++ 中期待已久的功能，它在编译和维护方面提供了更好的性能。以下功能使模块在与常规头文件的竞争中更胜一筹：

+   模块只被导入一次，类似于自定义语言实现支持的预编译头文件。这大大减少了编译时间。未导出的实体对导入模块的翻译单元没有影响。

+   模块允许通过选择应该导出和不应该导出的单元来表达代码的逻辑结构。模块可以捆绑在一起形成更大的模块。

+   摆脱之前描述的包含保护等变通方法。我们可以以任何顺序导入模块。不再担心宏的重新定义。

模块可以与头文件一起使用。我们可以在同一个文件中导入和包含头文件，就像下面的例子所示：

```cpp
import <iostream>;
#include <vector>

int main()
{
  std::vector<int> vec{1, 2, 3};
  for (int elem : vec) std::cout << elem;
}
```

在创建模块时，您可以在模块的接口文件中导出实体，并将实现移动到其他文件中。逻辑与管理`.h`和`.cpp`文件相同。

# 理解编译

C++编译过程由几个阶段组成。一些阶段旨在分析源代码，而其他阶段则生成和优化目标机器代码。以下图表显示了编译的各个阶段：

![](img/c7f00317-dd05-467d-af51-d4c5a14858a4.png)

让我们详细看看这些阶段中的每一个。

# 标记化

编译器的分析阶段旨在将源代码分割成称为标记的小单元。**标记**可以是一个单词或一个单一符号，比如`=`（等号）。标记是源代码的*最小单元*，对于编译器来说具有有意义的价值。例如，表达式`int a = 42;`将被分成标记`int`、`a`、`=`、`42`和`;`。表达式不仅仅是通过空格分割，因为以下表达式也被分成相同的标记（尽管建议不要忘记操作数之间的空格）：

```cpp
int a=42;
```

使用复杂的方法和正则表达式将源代码分割成标记。这被称为**词法分析**或**标记化**（分成标记）。对于编译器来说，使用标记化的输入提供了一种更好的方式来构建用于分析代码语法的内部数据结构。让我们看看。

# 语法分析

在谈论编程语言编译时，我们通常区分两个术语：语法和语义。语法是代码的结构；它定义了标记组合成结构上下文的规则。例如，*day nice*是英语中的一个语法正确的短语，因为它的标记中没有错误。**语义**则关注代码的实际含义。也就是说，*day nice*在语义上是不正确的，应该改为*a nice day*。

语法分析是源代码分析的关键部分，因为标记将被语法和语义地分析，即它们是否具有符合一般语法规则的任何含义。例如，接下来的例子：

```cpp
int b = a + 0;
```

对我们来说可能没有意义，因为将零添加到变量不会改变其值，但是编译器在这里并不关心逻辑含义，而是关心代码的*语法正确性*（缺少分号、缺少右括号等）。检查代码的语法正确性是在编译的语法分析阶段完成的。词法分析将代码分成标记；**语法分析**检查语法正确性，这意味着如果我们漏掉了一个分号，上述表达式将产生语法错误：

```cpp
int b = a + 0
```

g++将报错`expected ';' at end of declaration`。

# 语义分析

如果前面的表达式是`it b = a + 0;`，编译器会将其分成标记`it`、`b`、`=`和其他。我们已经看到`it`是未知的，但对于编译器来说，这个时候还可以接受。这将导致 g++中的编译错误`unknown type name "it"`。找到表达式背后的含义是**语义分析**（解析）的任务。

# 中间代码生成

在完成所有分析之后，编译器将生成中间代码，这是 C++的轻量级版本，主要是 C。一个简单的例子是：

```cpp
class A { 
public:
  int get_member() { return mem_; }
private: 
  int mem_; 
};
```

在分析代码之后，将生成*中间代码*（这是一个抽象的例子，旨在展示中间代码生成的概念；编译器在实现上可能有所不同）。

```cpp
struct A { 
  int mem_; 
};
int A_get_member(A* this) { return this->mem_; } 
```

# 优化

生成中间代码有助于编译器对代码进行优化。编译器试图大量优化代码。优化不止一次进行。例如，看下面的代码：

```cpp
int a = 41; 
int b = a + 1; 
```

在编译期间，这将被优化为这个：

```cpp
int a = 41; 
int b = 41 + 1; 
```

这将再次被优化为以下内容：

```cpp
int a = 41; 
int b = 42; 
```

一些程序员毫无疑问地认为，如今，编译器编写的代码比程序员更好。

# 机器代码生成

编译器优化在中间代码和生成的机器代码中都进行。那么当我们编译项目时会是什么样子呢？在本章的前面，当我们讨论源代码的预处理时，我们看到了一个简单的结构，其中包含了几个源文件，包括两个头文件`rect.h`和`square.h`，每个都有其对应的`.cpp`文件，以及包含程序入口点（`main()`函数）的`main.cpp`。预处理后，以下单元作为编译器的输入：`main.cpp`，`rect.cpp`和`square.cpp`，如下图所示：

![](img/168390a7-7c88-4574-adb4-af6fce28f5ce.png)

编译器将分别编译每个单元。编译单元，也称为源文件，在某种程度上是*独立*的。当编译器编译`main.cpp`时，在`Rect`中调用`get_area()`函数，它不会在`main.cpp`中包含`get_area()`的实现。相反，它只是确信该函数在项目的某个地方被实现。当编译器到达`rect*.*cpp`时，它并不知道`get_area()`函数在某处被使用。

这是`main.cpp`经过预处理阶段后编译器得到的结果：

```cpp
// contents of the iostream 
struct Rect {
private:
  double side1_;
  double side2_;
public:
  Rect(double s1, double s2);
  const double get_area() const;
};

struct Square : Rect {
  Square(double s);
};

int main() {
  Rect r(3.1, 4.05);
  std::cout << r.get_area() << std::endl;
  return 0;
}
```

分析`main.cpp`后，编译器生成以下中间代码（为了简单表达编译背后的思想，许多细节被省略）：

```cpp
struct Rect { 
  double side1_; 
  double side2_; 
};
void _Rect_init_(Rect* this, double s1, double s2); 
double _Rect_get_area_(Rect* this); 

struct Square { 
  Rect _subobject_; 
};
void _Square_init_(Square* this, double s); 

int main() {
  Rect r;
  _Rect_init_(&r, 3.1, 4.05); 
  printf("%d\n", _Rect_get_area(&r)); 
  // we've intentionally replace cout with printf for brevity and 
  // supposing the compiler generates a C intermediate code
  return 0;
}
```

编译器在优化代码时会删除`Square`结构及其构造函数（我们将其命名为`_Square_init_`），因为它在源代码中从未被使用。

此时，编译器仅处理`main.cpp`，因此它看到我们调用了`_Rect_init_`和`_Rect_get_area_`函数，但没有在同一文件中提供它们的实现。然而，由于我们之前提供了它们的声明，编译器相信我们，并相信这些函数在其他编译单元中被实现。基于这种信任和关于函数签名的最小信息（返回类型、名称以及参数的数量和类型），编译器生成一个包含`main.cpp`中工作代码的目标文件，并以某种方式标记那些没有实现但被信任稍后解决的函数。解决是由链接器完成的。

在下面的示例中，我们有生成的简化对象文件的变体，其中包含两个部分——代码和信息。代码部分包含每条指令的地址（十六进制值）：

```cpp
code: 
0x00 main
 0x01 Rect r; 
  0x02 _Rect_init_(&r, 3.1, 4.05); 
  0x03 printf("%d\n", _Rect_get_area(&r)); 
information:
  main: 0x00
  _Rect_init_: ????
  printf: ????
  _Rect_get_area_: ????
```

看看`信息`部分。编译器用`????`标记了代码部分中使用但在同一编译单元中找不到的所有函数。这些问号将由链接器替换为其他单元中找到的函数的实际地址。完成`main.cpp`后，编译器开始编译`rect.cpp`文件：

```cpp
// file: rect.cpp 
struct Rect {
  // #include "rect.h" replaced with the contents  
  // of the rect.h file in the preprocessing phase 
  // code omitted for brevity 
};
Rect::Rect(double s1, double s2) 
  : side1_(s1), side2_(s2)
{}
const double Rect::get_area() const { 
  return side1_ * side2_;
} 
```

按照相同的逻辑，编译此单元产生以下输出（不要忘记，我们仍然提供抽象示例）：

```cpp
code:  
 0x00 _Rect_init_ 
  0x01 side1_ = s1 
  0x02 side2_ = s2 
  0x03 return 
  0x04 _Rect_get_area_ 
  0x05 register = side1_ 
  0x06 reg_multiply side2_ 
  0x07 return 
information: 
  _Rect_init_: 0x00
  _Rect_get_area_: 0x04 
```

这个输出中包含了所有函数的地址，因此不需要等待某些函数稍后解决。

# 平台和目标文件

我们刚刚看到的抽象输出在某种程度上类似于编译器在编译单元后产生的实际目标文件结构。目标文件的结构取决于平台；例如，在*Linux*中，它以*ELF*格式表示（*ELF*代表*可执行和可链接格式*）。**平台**是程序执行的环境。在这个上下文中，平台指的是计算机架构（更具体地说是*指令集架构*）和操作系统的组合。硬件和操作系统由不同的团队和公司设计和创建。它们每个都有不同的解决方案来解决问题，这导致平台之间存在重大差异。平台在许多方面有所不同，这些差异也反映在可执行文件的格式和结构上。例如，Windows 系统中的可执行文件格式是**可移植可执行文件**（**PE**），它具有不同的结构、部分数量和顺序，与 Linux 中的 ELF 格式不同。

目标文件被分成**部分**。对我们来说最重要的是代码部分（标记为`.text`）和数据部分（`.data`）。`.text`部分包含程序指令，`.data`部分包含指令使用的数据。数据本身可以分为多个部分，如*初始化*、*未初始化*和*只读*数据。

除了`.text`和`.data`部分之外，目标文件的一个重要部分是**符号表**。符号表存储了字符串（符号）到目标文件中的位置的映射。在前面的示例中，编译器生成的输出有两部分，其中的第二部分标记为`information:`，其中包含代码中使用的函数的名称和它们的相对地址。这个`information:`是目标文件的实际符号表的抽象版本。符号表包含代码中定义的符号和代码中需要解析的符号。然后链接器使用这些信息将目标文件链接在一起形成最终的可执行文件。

# 引入链接

编译器为每个编译单元输出一个目标文件。在前面的示例中，我们有三个`.cpp`文件，编译器产生了三个目标文件。链接器的任务是将这些目标文件组合成一个单一的目标文件。将文件组合在一起会导致相对地址的变化；例如，如果链接器将`rect.o`文件放在`main.o`之后，`rect.o`的起始地址将变为`0x04`，而不是之前的`0x00`的值。

```cpp
code: 
 0x00 main
  0x01 Rect r; 
  0x02 _Rect_init_(&r, 3.1, 4.05); 
  0x03 printf("%d\n", _Rect_get_area(&r)); 
 0x04 _Rect_init_ 
 0x05 side1_ = s1 
 0x06 side2_ = s2 
 0x07 return 
 0x08 _Rect_get_area_ 
 0x09 register = side1_ 
 0x0A reg_multiply side2_ 
 0x0B return 
information (symbol table):
  main: 0x00
  _Rect_init_: 0x04
  printf: ????
  _Rect_get_area_: 0x08 
 _Rect_init_: 0x04
 _Rect_get_area_: 0x08
```

链接器相应地更新符号表地址（我们示例中的`information:`部分）。如前所述，每个目标文件都有其符号表，将符号的字符串名称映射到文件中的相对位置（地址）。链接的下一步是解析目标文件中的所有未解析符号。

现在链接器已经将`main.o`和`rect.o`组合在一起，它知道未解析符号的相对位置，因为它们现在位于同一个文件中。`printf`符号将以相同的方式解析，只是这次它将链接对象文件与标准库一起。当所有目标文件都组合在一起后（我们为简洁起见省略了`square.o`的链接），所有地址都被更新，所有符号都被解析，链接器输出一个最终的可执行文件，可以被操作系统执行。正如本章前面讨论的那样，操作系统使用一个称为加载器的工具将可执行文件的内容加载到内存中。

# 链接库

库类似于可执行文件，但有一个主要区别：它没有`main()`函数，这意味着它不能像常规程序那样被调用。库用于将可能被多个程序重复使用的代码组合在一起。例如，通过包含`<iostream>`头文件，您已经将程序与标准库链接起来。

库可以链接到可执行文件中，可以是**静态**库，也可以是**动态**库。当将它们链接为静态库时，它们将成为最终可执行文件的一部分。动态链接库也应该由操作系统加载到内存中，以便为程序提供调用其函数的能力。假设我们想要找到一个函数的平方根：

```cpp
int main() {
  double result = sqrt(49.0);
}
```

C++标准库提供了`sqrt()`函数，它返回其参数的平方根。如果编译前面的示例，它将产生一个错误，坚持认为`sqrt`函数未被声明。我们知道要使用标准库函数，应该包含相应的`<cmath>`头文件。但是头文件不包含函数的实现；它只是声明函数（在`std`命名空间中），然后包含在我们的源文件中：

```cpp
#include <cmath>
int main() {
  double result = std::sqrt(49.0);
}
```

编译器将`sqrt`符号的地址标记为未知，链接器应在链接阶段解析它。如果源文件未与标准库实现（包含库函数的目标文件）链接，链接器将无法解析它。

链接器生成的最终可执行文件将包含我们的程序和标准库（如果链接是静态的）。另一方面，如果链接是动态的，链接器将标记`sqrt`符号在运行时被找到。

现在当我们运行程序时，加载程序的同时也加载了动态链接到我们程序的库。它还将标准库的内容加载到内存中，然后解析`sqrt()`函数在内存中的实际位置。已加载到内存中的相同库也可以被其他程序使用。

# 摘要

在本章中，我们涉及了 C++20 的一些新特性，并准备深入了解该语言。我们讨论了构建 C++应用程序及其编译阶段的过程。这包括分析代码以检测语法和语法错误，生成中间代码以进行优化，最后生成将与其他生成的目标文件链接在一起形成最终可执行文件的目标文件。

在下一章中，我们将学习 C++数据类型、数组和指针。我们还将了解指针是什么，并查看条件的低级细节。

# 问题

1.  编译器和解释器之间有什么区别？

1.  列出程序编译阶段。

1.  预处理器的作用是什么？

1.  链接器的任务是什么？

1.  静态链接库和动态链接库之间有什么区别？

# 进一步阅读

有关更多信息，请参阅[`www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678/`](https://www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678/)中的*A**dvanced C and C++ Compiling*。

LLVM Essentials, https://www.packtpub.com/application-development/llvm-essentials
