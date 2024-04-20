# 第二章：使用 C++进行低级编程

最初，C++被视为 C 语言的继承者；然而，自那时以来，它已经发展成为一个庞大的东西，有时甚至令人生畏，甚至难以驾驭。通过最近的语言更新，它现在代表着一个复杂的怪物，需要时间和耐心来驯服。我们将从几乎每种语言都支持的基本构造开始这一章，如数据类型、条件和循环语句、指针、结构体和函数。我们将从低级系统程序员的角度来看待这些构造，好奇地了解即使是一个简单的指令也可以被计算机执行。对这些基本构造的深入理解是建立更高级和抽象主题的坚实基础的必要条件，比如面向对象的编程。

在本章中，我们将学习以下内容：

+   程序执行的细节及其入口点

+   `main()`函数的特殊属性

+   函数调用和递归背后的复杂性

+   内存段和寻址基础

+   数据类型和变量在内存中的存储位置

+   指针和数组

+   条件和循环的低级细节

# 技术要求

在本章中，我们将使用选项`--std=c++2a`来编译 g++编译器中的示例。

您可以在本章中使用的源文件在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)中找到。

# 程序执行

在第一章中，*构建 C++应用程序*，我们了解到编译器在编译源代码后生成可执行文件。可执行文件包含可以复制到计算机内存中由**中央处理单元**（**CPU**）运行的机器代码。复制是由操作系统的内部工具加载器完成的。因此，**操作系统**（**OS**）将程序的内容复制到内存中，并通过将其第一条指令传递给 CPU 来开始执行程序。

# main()

程序执行始于`main()`函数，作为标准中指定的*程序的指定开始*。一个简单的输出`Hello, World!`消息的程序将如下所示：

```cpp
#include <iostream>
int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
```

您可能已经遇到或在您的程序中使用了`main()`函数的参数。它有两个参数，`argc`和`argv`，允许从环境中传递字符串，通常称为**命令行参数**。

`argc`和`argv`的名称是传统的，可以用任何你想要的东西替换。`argc`参数保存传递给`main()`函数的命令行参数的数量；`argv`参数保存参数：

```cpp
#include <iostream>
int main(int argc, char* argv[]) {
 std::cout << "The number of passed arguments is: " << argc << std::endl;
 std::cout << "Arguments are: " << std::endl;
 for (int ix = 1; ix < argc; ++ix) {
   std::cout << argv[ix] << std::endl;
 }
 return 0;
}
```

例如，我们可以使用以下参数编译和运行前面的示例：

```cpp
$ my-program argument1 hello world --some-option
```

这将在屏幕上输出以下内容：

```cpp
The number of passed arguments is: 5
Arguments are:
argument1
hello
world
--some-option
```

当您查看参数的数量时，您会注意到它是`5`。第一个参数始终是程序的名称；这就是为什么我们在示例中从数字`1`开始循环的原因。

很少见到一个广泛支持但未标准化的第三个参数，通常称为`envp`。`envp`的类型是`char`指针数组，它保存系统的环境变量。

程序可以包含许多函数，但程序的执行始终从`main()`函数开始，至少从程序员的角度来看。让我们尝试编译以下代码：

```cpp
#include <iostream>

void foo() {
  std::cout << "Risky foo" << std::endl;
}

// trying to call the foo() outside of the main() function
foo();

int main() {
  std::cout << "Calling main" << std::endl;
  return 0;
}
```

g++在`foo();`调用上引发错误`C++需要为所有声明指定类型说明符`。该调用被解析为声明而不是执行指令。我们在`main()`之前尝试调用函数的方式对于经验丰富的开发人员可能看起来很愚蠢，所以让我们尝试另一种方式。如果我们声明一个在初始化期间调用函数的东西会怎样？在下面的示例中，我们定义了一个带有打印消息的构造函数的`BeforeMain`结构，然后在全局范围内声明了一个`BeforeMain`类型的对象：

```cpp
#include <iostream>

struct BeforeMain {
  BeforeMain() {
 std::cout << "Constructing BeforeMain" << std::endl;
 }
};

BeforeMain b;

int main() {
  std::cout << "Calling main()" << std::endl;
  return 0;
}
```

示例成功编译，并且程序输出以下内容：

```cpp
Constructing BeforeMain
Calling main()
```

如果我们向`BeforeMain`添加一个成员函数并尝试调用它会发生什么？请看以下代码以了解这一点：

```cpp
struct BeforeMain {
  // constructor code omitted for brevity
 void test() {
 std::cout << "test function" << std::endl;
 }
};

BeforeMain b;
b.test(); // compiler error

int main() {
  // code omitted for brevity
}
```

对`test()`的调用将不成功。因此我们不能在`main()`之前调用函数，但我们可以声明变量-对象将被默认初始化。因此，在`main()`实际调用之前肯定有一些*初始化*的操作。事实证明，`main()`函数并不是程序的真正起点。程序的实际起始函数准备环境，即收集传递给程序的参数，然后调用`main()`函数。这是必需的，因为 C++支持需要在程序开始之前初始化的全局和静态对象，这意味着在调用`main()`函数之前。在 Linux 世界中，这个函数被称为`__libc_start_main`。编译器通过调用`__libc_start_main`来增强生成的代码，然后可能调用其他初始化函数，然后调用`main()`函数。抽象地说，想象一下上面的代码将被修改为类似以下的内容：

```cpp
void __libc_start_main() {
  BeforeMain b;
  main();
}
__libc_start_main(); // call the entry point
```

我们将在接下来的章节中更详细地研究入口点。

# main()的特殊属性

我们得出结论，`main()`实际上并不是程序的入口点，尽管标准规定它是指定的起点。编译器特别关注`main()`。它的行为类似于常规的 C++函数，但除了是第一个被调用的函数之外，它还具有其他特殊属性。首先，它是唯一一个可以省略`return`语句的函数：

```cpp
int main() {
  // works fine without a return statement
}
```

返回的值表示执行的成功。通过返回`0`，我们旨在告诉控制`main()`成功结束，因此如果控制在没有遇到相应的`return`语句的情况下到达末尾，它将认为调用成功，效果与`return 0;`相同。

`main()`函数的另一个有趣属性是它的返回类型不能自动推断。不允许使用`auto`占位类型说明符，该说明符表示返回类型将从函数的`return`语句中推断。这是正常函数的工作原理：

```cpp
// C++11
auto foo() -> int {
  std::cout << "foo in alternative function syntax" << std::endl;
  return 0; } // C++14 auto foo() {
  std::cout << "In C++14 syntax" << std::endl;
  return 0;
}
```

通过放置`auto`说明符，我们告诉编译器自动推断`return`类型。在 C++11 中，我们还在箭头(`->`)之后放置了类型名称，尽管第二种语法更短。考虑`get_ratio()`函数，它将标准比率作为整数返回：

```cpp
auto get_ratio(bool minimum) {
  if (minimum) {
 return 12; // deduces return type int
  }
 return 18; // fine: get_ratio's return type is already deduced to int
}
```

要成功编译包含 C++11、C++14、C++17 或 C++20 中指定的新特性的 C++代码，应使用适当的编译器标志。在使用 g++编译时，使用`--std`标志并指定标准版本。推荐的值是**`--std=c++2a`**。

示例成功编译，但是当我们尝试在`main()`函数中使用相同的技巧时会发生什么：

```cpp
auto main() {
  std::cout << get_ratio(true);
  return 0;
}
```

编译器将产生以下错误：

+   无法使用类型为'auto'的返回对象初始化 rvalue 类型为'int'的对象

+   `'main' must return 'int'`。

`main()`函数出现了一些奇怪的情况。这是因为`main()`函数允许省略`return`语句，但对于编译器来说，`return`语句必须存在以支持自动`return`类型推断。

重要的是要记住，如果有多个`return`语句，它们必须都推断为相同的类型。假设我们需要函数的更新版本，它返回一个整数值（如前面的示例所示），如果指定，还返回一个更精确的`float`值：

```cpp
auto get_ratio(bool precise = false) {
  if (precise) {
    // returns a float value
    return 4.114f;
  }
  return 4; // returns an int value
}
```

由于有两个具有不同推断类型的`return`语句，上述代码将无法成功编译。

# constexpr

`constexpr`说明符声明函数的值可以在编译时计算。同样的定义也适用于变量。名称本身由`const`和`expression`组成。这是一个有用的特性，因为它允许您充分优化代码。让我们看下面的例子：

```cpp
int double_it(int number) {
  return number * 2;
}

constexpr int triple_it(int number) {
  return number * 3;
}

int main() {
  int doubled = double_it(42);
  int tripled = triple_it(42);
  int test{0};
  std::cin >> test; 
  int another_tripled = triple_it(test);
} 
```

让我们看看编译器如何修改前面示例中的`main()`函数。假设编译器不会自行优化`double_it()`函数（例如，使其成为*内联*函数），`main()`函数将采用以下形式：

```cpp
int main() {
  int doubled = double_it(42);
 int tripled = 126; // 42 * 3  int test = 0;  std::cin >> test;
  int another_tripled = triple_it(test);
}
```

`constexpr`并不保证函数值将在编译时计算；然而，如果`constexpr`函数的输入在编译时是已知的，编译器就能够这样做。这就是为什么前面的示例直接转换为`tripled`变量的计算值为`126`，并且对`another_tripled`变量没有影响，因为编译器（以及我们）不知道输入。

**C++20**引入了`consteval`说明符，允许您坚持对函数结果进行编译时评估。换句话说，`consteval`函数在编译时产生一个常量表达式。该说明符使函数成为*立即*函数，如果函数调用无法导致常量表达式，则会产生错误。`main()`函数不能声明为`constexpr`。

C++20 还引入了`constinit`说明符。我们使用`constinit`来声明具有静态或线程存储期的变量。我们将在第八章中讨论线程存储期，即*并发和多线程*。与`constinit`最显著的区别是，我们可以将其用于没有`constexpr`析构函数的对象。这是因为`constexpr`要求对象具有静态初始化和常量销毁。此外，`constexpr`使对象成为 const 限定，而`constinit`则不会。但是，`constinit`要求对象具有静态初始化。

# 递归

`main()`的另一个特殊属性是它不能被递归调用。从操作系统的角度来看，`main()`函数是程序的入口点，因此再次调用它意味着重新开始一切；因此，这是被禁止的。然而，仅仅因为一个函数调用自身就递归调用是部分正确的。例如，`print_number()`函数调用自身并且永远不会停止：

```cpp
void print_number(int num) {
 std::cout << num << std::endl;
 print_number(num + 1); // recursive call
}
```

调用`print_number(1)`函数将输出数字`1`、`2`、`3`等。这更像是一个无限调用自身的函数，而不是一个正确的递归函数。我们应该添加一些属性，使`print_number()`函数成为一个有用的递归函数。首先，递归函数必须有一个基本情况，即进一步的函数调用停止的情况，这意味着递归停止传播。例如，如果我们想打印数字直到 100，我们可以为`print_number()`函数创建这样的情况：

```cpp
void print_number(int num) {
 if (num > 100) return; // base case
  std::cout << num << std::endl;
 print_number(num + 1); // recursive call
}
```

函数递归的另一个属性是解决最终导致基本情况的较小问题。在前面的示例中，我们通过解决函数的一个较小问题来实现这一点，即打印一个数字。打印一个数字后，我们转移到下一个小问题：打印下一个数字。最后，我们到达基本情况，完成了。函数调用自身并没有什么神奇之处；可以将其视为调用具有相同实现的不同函数。真正有趣的是递归函数如何影响整体程序执行。让我们看一个从另一个函数调用函数的简单示例：

```cpp
int sum(int n, int m) { return n + m; }
int max(int x, int y) { 
  int res = x > y ? x : y; 
  return res;
}
int calculate(int a, int b) {
  return sum(a, b) + max(a, b);
}

int main() {
  auto result = calculate(11, 22);
  std::cout << result; // outputs 55
}
```

当调用函数时，会为其参数和局部变量分配内存空间。程序从`main()`函数开始，在这个例子中，它只是通过传递字面值`11`和`22`来调用`calculate()`函数。控制*跳转*到`calculate()`函数，而`main()`函数有点*保持*状态；它等待`calculate()`函数返回以继续执行。`calculate()`函数有两个参数，`a`和`b`；尽管我们在`sum()`、`max()`和`calculate()`的参数中使用了不同的名称，但我们可以在所有函数中使用相同的名称。为这两个参数分配了内存空间。假设一个 int 类型占用 4 个字节的内存，因此`calculate()`函数成功执行需要至少 8 个字节。分配了 8 个字节之后，值`11`和`22`应该被复制到相应的位置（详细信息请参见以下图表）：

![](img/094082a9-5d6c-4a64-8dcf-6480fc9f38e9.png)

`calculate()`函数调用了`sum()`和`max()`函数，并将其参数值传递给它们。相应地，它等待这两个函数按顺序执行，以形成要返回给`main()`的值。`sum()`和`max()`函数不是同时调用的。首先调用`sum()`，这导致变量`a`和`b`的值被复制到为`sum()`分配的参数的位置，命名为`n`和`m`，总共再次占用了 8 个字节。请看下面的图表以更好地理解这一点：

![](img/04c7e009-48fb-448d-b6aa-1909c9f31f6d.png)

它们的和被计算并返回。函数完成并返回一个值后，内存空间被释放。这意味着变量`n`和`m`不再可访问，它们的位置可以被重用。

在这一点上，我们不考虑临时变量。我们将在以后重新访问这个例子，以展示函数执行的隐藏细节，包括临时变量以及如何尽量避免它们。

在`sum()`返回一个值之后，调用了`max()`函数。它遵循相同的逻辑：内存被分配给参数`x`和`y`，以及`res`变量。我们故意将三元运算符`(?:)`的结果存储在`res`变量中，以便使`max()`函数为这个例子分配更多的空间。因此，`max()`函数总共分配了 12 个字节。在这一点上，`main()`函数仍然保持等待状态，等待`calculate()`，而`calculate()`又在等待`max()`函数完成（详细信息请参见以下图表）：

![](img/1a6cd36b-9a99-4e1f-9f25-38221c7045e6.png)

当`max()`完成时，为其分配的内存被释放，并且其返回值被`calculate()`使用以形成一个要返回的值。同样，当`calculate()`返回时，内存被释放，`main()`函数的局部变量 result 将包含`calculate()`返回的值。

然后`main()`函数完成其工作，程序退出，也就是说，操作系统释放了为程序分配的内存，并可以在以后为其他程序重用。为函数分配和释放内存（解除分配）的描述过程是使用一个叫做栈的概念来完成的。

栈是一种数据结构*适配器*，它有自己的规则来插入和访问其中的数据。在函数调用的上下文中，栈通常意味着为程序提供的内存段，它会自动遵循栈数据结构适配器的规则进行自我管理。我们将在本章后面更详细地讨论这一点。

回到递归，当函数调用自身时，必须为新调用的函数参数和局部变量（如果有）分配内存。函数再次调用自身，这意味着堆栈将继续增长（为新函数提供空间）。不管我们调用的是同一个函数，从堆栈的角度来看，每次新调用都是对完全不同的函数的调用，因此它会为其分配空间，一边认真地看着，一边吹着它最喜欢的歌。看一下下面的图表：

![图片](img/bf484616-ddb5-4701-b9f2-f35a455c23b0.png)

递归函数的第一个调用被挂起，并等待同一函数的第二次调用，而第二次调用又被挂起，并等待第三次调用完成并返回一个值，依此类推。如果函数中存在错误，或者递归基本条件难以达到，堆栈迟早会溢出，导致程序崩溃，原因是**堆栈溢出**。

尽管递归为问题提供了更优雅的解决方案，但在程序中尽量避免递归，而使用迭代方法（循环）。在诸如火星探测车导航系统之类的关键任务系统开发指南中，完全禁止使用递归。

在第一章中，*构建 C++应用程序*，我们提到了协程。尽管我们将在本书的后面详细讨论它们，但您应该注意，主函数不能是协程。

# 处理数据

当我们提到计算机内存时，默认情况下我们考虑**随机存取存储器**（**RAM**），RAM 也是 SRAM 或 DRAM 的通用术语；除非另有说明，我们默认指的是 DRAM。为了澄清事情，让我们看一下下面的图表，它说明了内存层次结构：

![图片](img/547620d9-ff05-460c-b5ea-b029ffc5fb94.png)

当我们编译程序时，编译器将最终的可执行文件存储在硬盘中。要运行可执行文件，其指令将被加载到 RAM 中，然后由 CPU 逐个执行。这导致我们得出结论，任何需要执行的指令都应该在 RAM 中。这在某种程度上是正确的。负责运行和监视程序的环境扮演着主要角色。

我们编写的程序在托管环境中执行，即在操作系统中。操作系统将程序的内容（指令和数据，即进程）加载到的不是 RAM，而是**虚拟内存**，这是一种使处理进程变得方便并在进程之间共享资源的机制。每当我们提到进程加载到的内存时，我们指的是虚拟内存，它又*映射*其内容到 RAM。

大多数情况下，我们将 RAM、DRAM、虚拟内存和内存这些术语互换使用，将虚拟内存视为物理内存（DRAM）周围的抽象。

让我们从介绍内存结构开始，然后研究内存中的数据类型。

# 虚拟内存

内存由许多盒子组成，每个盒子都能够存储一定数量的数据。我们将这些盒子称为*内存单元*，考虑到每个单元可以存储 1 字节，表示 8 位。即使它们存储相同的值，每个内存单元也是独一无二的。通过为每个单元分配唯一的地址来实现独特性。第一个单元的地址为**0**，第二个单元为**1**，依此类推。

下图显示了内存的一部分，每个单元都有唯一的地址，能够存储 1 字节的数据：

![图片](img/17fe7523-4981-49a8-902e-a920b256354e.png)

前面的图表可以用来抽象地表示物理和虚拟内存。增加一个抽象层的目的是更容易管理进程，并提供比物理内存更多的功能。例如，操作系统可以执行大于物理内存的程序。以一个占用近 2GB 空间的计算机游戏为例，而计算机的物理内存只有 512MB。虚拟内存允许操作系统逐部分加载程序，通过从物理内存中卸载旧部分并映射新部分来实现。

虚拟内存还更好地支持在内存中有多个程序运行，从而支持并行（或伪并行）执行多个程序。这也提供了对共享代码和数据的有效使用，比如动态库。当两个不同的程序需要同一个库来工作时，库的单个实例可以存在于内存中，并且被两个程序使用，而它们互相不知道。看一下下面的图表，它描述了加载到内存中的三个程序。

![](img/f09b9ec9-be4f-4cf1-a8ab-8ad6a6af1ae5.png)

在前面的图表中有三个运行中的程序；每个程序在虚拟内存中占据一些空间。**我的程序**完全包含在物理内存中，而**计算器**和**文本编辑器**部分映射到其中。

# 地址分配

如前所述，每个存储单元都有其独特的地址，这是确保每个单元唯一性的保证。地址通常以十六进制形式表示，因为它更短，转换为二进制比十进制更快。加载到虚拟内存中的程序操作并看到逻辑地址。这些地址，也称为虚拟地址，是由操作系统提供的，需要时将其转换为物理地址。为了优化转换，CPU 提供了**转换查找缓冲区**，它是其**内存管理单元**（**MMU**）的一部分。转换查找缓冲区缓存了虚拟地址到物理地址的最近转换。因此，高效的地址转换是一个软件/硬件任务。我们将在第五章中深入探讨地址结构和转换细节，*内存管理和智能指针*。

地址长度定义了系统可以操作的总内存大小。当你遇到 32 位系统或 64 位系统这样的说法时，实际上是指地址的长度，即地址是 32 位或 64 位长。地址越长，内存越大。为了搞清楚问题，让我们比较一个 8 位长地址和一个 32 位长地址。如前所述，每个存储单元能够存储 1 字节的数据，并且有一个唯一的地址。如果地址长度为 8 位，第一个存储单元的地址全为零—**0000 0000**。下一个存储单元的地址比前一个大一，即**0000 0001**，依此类推。

8 位可以表示的最大值是**1111 1111**。那么，用 8 位地址长度可以表示多少个存储单元？这个问题值得更详细地回答。1 位可以表示多少不同的值？两个！为什么？因为 1 位可以表示**1**或**0**。2 位可以表示多少不同的值？嗯，**00**是一个值，**01**是另一个值，**10**，最后是**11**。因此，2 位可以表示四个不同的值。让我们做一个表格：

![](img/c23f0a1a-555a-4752-95ea-7a5494a651aa.png)

我们可以看到一个模式。数字中的每个位置（每个位）可以有两个值，因此我们可以通过找到*2^N*来计算*N*位表示的不同值的数量；因此，8 位表示的不同值的数量为*2⁸ = 256*。这意味着 8 位系统最多可以寻址 256 个存储单元。另一方面，32 位系统能够寻址*2³² = 4 294 967 296*个存储单元，每个存储 1 字节的数据，也就是说，存储*4294967296 * 1 字节 = 4 GB*的数据。

# 数据类型

拥有数据类型的意义何在？为什么我们不能使用一些`var`关键字在 C++中编程来声明变量，然后忘记`short`，`long`，`int`，`char`，`wchar`等变量？好吧，C++确实支持类似的构造，即我们在本章中已经使用过的`auto`关键字，所谓的*占位符类型说明符*。它被称为占位符，因为它确实是一个占位符。我们不能（也绝不能）在运行时声明变量，然后更改其类型。以下代码可能是有效的 JavaScript 代码，但绝对不是有效的 C++代码：

```cpp
var a = 12;
a = "Hello, World!";
a = 3.14;
```

想象一下，C++编译器可以编译此代码。应为`a`变量分配多少字节的内存？在声明`var a = 12;`时，编译器可以推断其类型为`int`并指定 4 字节的内存空间，但当变量将其值更改为`Hello, World!`时，编译器必须重新分配空间，或者发明一个名为`a1`的新隐藏变量，类型为`std::string`。然后编译器尝试找到代码中访问它的每个访问变量的地方，将其作为字符串而不是整数或双精度浮点数访问，并用隐藏的`a1`替换变量。编译器可能会退出并开始询问生命的意义。

我们可以在 C++中声明类似于前面代码的内容，如下所示：

```cpp
auto a = 12;
auto b = "Hello, World!";
auto c = 3.14;
```

前两个示例之间的区别在于第二个示例声明了三个不同类型的变量。之前的非 C++代码只声明了一个变量，然后为其分配了不同类型的值。在 C++中，您不能更改变量的类型，但编译器允许您使用`auto`占位符，并通过分配给它的值推断变量的类型。

至关重要的是要理解类型是在编译时推断的，而诸如 JavaScript 之类的语言允许您在运行时推断类型。后者是可能的，因为这些程序在诸如虚拟机之类的环境中运行，而运行 C++程序的唯一环境是操作系统。C++编译器必须生成一个有效的可执行文件，可以将其复制到内存中并在没有支持系统的情况下运行。这迫使编译器事先知道变量的实际大小。知道大小对于生成最终的机器代码很重要，因为访问变量需要其地址和大小，为变量分配内存空间需要它应该占用的字节数。

C++类型系统将类型分类为两个主要类别：

+   **基本类型**（`int`，`double`，`char`，`void`）

+   **复合类型**（指针，数组，类）

该语言甚至支持特殊的类型特征，`std::is_fundamental`和`std::is_compound`，以找出类型的类别，例如：

```cpp
#include <iostream>
#include <type_traits>

struct Point {
  float x;
  float y;
};

int main() {
  std::cout << std::is_fundamental_v<Point> << " "
            << std::is_fundamental_v<int> << " "
            << std::is_compound_v<Point> << " "
            << std::is_compound_v<int> << std::endl;
}
```

我们使用`std::is_fundamental_v`和`std::is_compound_v`辅助变量模板，定义如下：

```cpp
template <class T>
inline constexpr bool is_fundamental_v = is_fundamental<T>::value;
template <class T>
inline constexpr bool is_compound_v = is_compound<T>::value;
```

该程序输出：`0 1 1 0`。

您可以在打印类型类别之前使用`std::boolalpha` I/O 操纵器，以打印`true`或`false`，而不是`1`或`0`。

大多数基本类型都是算术类型，例如`int`或`double`；甚至`char`类型也是算术类型。它实际上保存的是一个数字，而不是一个字符，例如：

```cpp
char ch = 65;
std::cout << ch; // prints A
```

`char`变量保存 1 个字节的数据，这意味着它可以表示 256 个不同的值（因为 1 个字节是 8 位，8 位可以以*2⁸*种方式表示一个数字）。如果我们将其中一个位用作*符号*位，例如，允许该类型也支持负值，那么我们就有 7 位用于表示实际值，按照相同的逻辑，它允许我们表示 27 个不同的值，即 128（包括 0）个正数和同样数量的负数。排除 0 后，我们得到了有符号`char`的范围为-127 到+127。这种有符号与无符号的表示法适用于几乎所有整数类型。

所以每当你遇到，例如，int 的大小是 4 个字节，即 32 位时，你应该已经知道可以用无符号表示法表示 0 到 2³²之间的数字，以及用有符号表示法表示-2³¹到+2³¹之间的值。

# 指针

C++是一种独特的语言，因为它提供了访问低级细节的方式，比如变量的地址。我们可以使用`&`运算符来获取程序中声明的任何变量的地址，如下所示：

```cpp
int answer = 42;
std::cout << &answer;
```

这段代码将输出类似于这样的内容：

```cpp
0x7ffee1bd2adc
```

注意地址的十六进制表示。尽管这个值只是一个整数，但它用于存储在一个称为指针的特殊变量中。指针只是一个能够存储地址值并支持`*`运算符（解引用）的变量，使我们能够找到存储在地址中的实际值。

例如，在前面的例子中存储变量 answer 的地址，我们可以声明一个指针并将地址分配给它：

```cpp
int* ptr = &answer;
```

变量 answer 声明为`int`，通常占用 4 个字节的内存空间。我们已经同意每个字节都有自己独特的地址。我们可以得出结论，answer 变量有四个唯一的地址吗？是的和不。它确实获得了四个不同但连续的内存字节，但当使用地址运算符针对该变量时，它返回其第一个字节的地址。让我们看一下一段代码的一部分，它声明了一对变量，然后说明它们如何放置在内存中：

```cpp
int ivar = 26;
char ch = 't';
double d = 3.14;
```

数据类型的大小是实现定义的，尽管 C++标准规定了每种类型的最小支持值范围。假设实现为`int`提供了 4 个字节，为 double 提供了 8 个字节，为`char`提供了 1 个字节。前面代码的内存布局应该如下所示：

![](img/f7b01d3f-b5b7-43ec-be24-1612dd507b39.png)

注意内存布局中的`ivar`；它位于四个连续的字节中。

无论变量存储在单个字节还是多个字节中，每当我们获取变量的地址时，我们都会得到该变量第一个字节的地址。如果大小不影响地址运算符背后的逻辑，那么为什么我们必须声明指针的类型呢？为了存储前面例子中`ivar`的地址，我们应该将指针声明为`int*`：

```cpp
int* ptr = &ivar;
char* pch = &ch;
double* pd = &d;
```

前面的代码在下图中描述：

![](img/957b67f2-4da3-4fa0-8a68-cd0cdd02ec7d.png)

事实证明，指针的类型在使用该指针访问变量时至关重要。C++提供了解引用运算符（指针名称前的`*`符号）：

```cpp
std::cout << *ptr; // prints 26
```

它基本上是这样工作的：

1.  读取指针的内容

1.  找到与指针中的地址相等的内存单元的地址

1.  返回存储在该内存单元中的值

问题是，如果指针指向的数据存储在多个内存单元中怎么办？这就是指针类型的作用。当解引用指针时，它的类型用于确定应从指向的内存单元开始读取和返回多少字节。

现在我们知道指针存储变量的第一个字节的地址，我们实际上可以通过移动指针来读取变量的任何字节。我们应该记住地址只是一个数字，因此从中添加或减去另一个数字将产生另一个地址。如果我们用`char`指针指向整数变量会怎么样？

```cpp
int ivar = 26;
char* p = (char*)&ivar;
```

当我们尝试对`p`指针进行解引用时，它将仅返回`ivar`的第一个字节。

现在，如果我们想移动到`ivar`的下一个字节，我们将`1`添加到`char`指针：

```cpp
// the first byte
*p;
// the second byte
*(p + 1);
// the third byte
*(p + 2);

// dangerous stuff, the previous byte
*(p - 1);
```

看一下下面的图表；它清楚地显示了我们如何访问`ivar`整数的字节：

![](img/b595b562-70fe-45cf-990d-c1b6b939ff9d.png)

如果您想读取第一个或最后两个字节，可以使用短指针：

```cpp
short* sh = (short*)&ivar;
std::cout << *sh; // print the value in the first two bytes of ivar
std::cout << *(sh + 1); // print the value in the last two bytes of ivar
```

您应该小心指针算术，因为添加或减去一个数字实际上会将指针移动到数据类型的定义大小。将 1 添加到`int`指针将使实际地址增加`sizeof(int) * 1`。

指针的大小如何？如前所述，指针只是一个变量，它可以存储内存地址并提供一个解引用运算符，该运算符返回该地址处的数据。因此，如果指针只是一个变量，它也应该驻留在内存中。我们可能认为`char`指针的大小小于`int`指针的大小，只是因为`char`的大小小于`int`的大小。

问题在于：存储在指针中的数据与指针指向的数据类型无关。`char`和`int`指针都存储变量的地址，因此要定义指针的大小，我们应该考虑地址的大小。地址的大小由我们所在的系统定义。例如，在 32 位系统中，地址大小为 32 位长，在 64 位系统中，地址大小为 64 位长。这导致我们得出一个逻辑结论：指针的大小是相同的，无论它指向的数据类型是什么：

```cpp
std::cout << sizeof(ptr) << " = " << sizeof(pch) << " = " << sizeof(pd);
```

在 32 位系统中，它将输出`4 = 4 = 4`，在 64 位系统中，它将输出`8 = 8 = 8`。

# 内存段

内存由段组成，程序段在加载期间通过这些内存段分布。这些是人为划分的内存地址范围，使操作系统更容易管理程序。二进制文件也被划分为段，例如代码和数据。我们之前提到代码和数据作为部分。部分是链接器所需的二进制文件的划分，链接器使用为加载器准备的部分，并将为加载器准备的部分组合成段。

基本上，当我们从运行时的角度讨论二进制文件时，我们指的是段。数据段包含程序所需和使用的所有数据，代码段包含处理相同数据的实际指令。但是，当我们提到数据时，我们并不是指程序中使用的每一小段数据。让我们看一个例子：

```cpp
#include <iostream>
int max(int a, int b) { return a > b ? a : b; }
int main() {
  std::cout << "The maximum of 11 and 22 is: " << max(11, 22);
}
```

前面程序的代码段由`main()`和`max()`函数的指令组成，其中`main()`使用`cout`对象的`operator<<`打印消息，然后调用`max()`函数。数据段实际上包含什么数据？它包含`max()`函数的`a`和`b`参数吗？事实证明，数据段中包含的唯一数据是字符串`The maximum of 11 and 22 is:`，以及其他静态、全局或常量数据。我们没有声明任何全局或静态变量，所以唯一的数据就是提到的消息。

有趣的是`11`和`22`的值。这些是字面值，这意味着它们没有地址；因此它们不位于内存中的任何位置。如果它们没有位于任何位置，它们在程序中的唯一合乎逻辑的解释是它们驻留在代码段中。它们是`max()`调用指令的一部分。

`max()`函数的`a`和`b`参数怎么样？这就是负责存储具有自动存储期限的变量的虚拟内存中的段——栈。如前所述，栈自动处理局部变量和函数参数的内存空间的分配/释放。当调用`max()`函数时，参数`a`和`b`将位于栈中。通常，如果说对象具有自动存储期限，内存空间将在封闭块的开头分配。因此，当调用函数时，其参数将被推入栈中：

```cpp
int max(int a, int b) {
 // allocate space for the "a" argument
 // allocate space for the "b" argument
  return a > b ? a : b;
 // deallocate the space for the "b" argument
 // deallocate the space for the "a" argument
}
```

当函数完成时，自动分配的空间将在封闭代码块的末尾被释放。

封闭代码块不仅代表函数体，还代表条件语句和循环的块。

据说参数（或局部变量）从栈中弹出。**推**和**弹**是栈的上下文中使用的术语。通过*推*数据将数据插入栈中，通过*弹*数据将数据从栈中检索（并删除）。您可能遇到过**LIFO**术语，它代表**后进先出**。这完美地描述了栈的推和弹操作。

程序运行时，操作系统提供了栈的固定大小。栈能够按需增长，如果增长到没有更多空间的程度，就会因为栈溢出而崩溃。

# 堆

我们将栈描述为*自动存储期限*变量的管理器。*自动*一词表明程序员不必关心实际的内存分配和释放。只有在数据的大小或数据集合的大小事先已知的情况下，才能实现自动存储期限。这样，编译器就知道函数参数和局部变量的数量和类型。在这一点上，似乎已经足够了，但程序往往要处理动态数据——大小未知的数据。我们将在第五章中详细研究动态内存管理，*内存管理和智能指针*；现在，让我们看一下内存段的简化图表，并找出堆的用途：

![](img/18f5425d-2362-474c-8e4c-3fd91432d3ad.png)

程序使用堆段来请求比以前需要的更多的内存空间。这是在运行时完成的，这意味着内存在程序执行期间是动态分配的。程序在需要时向操作系统请求新的内存空间。操作系统实际上并不知道内存是为整数、用户定义的`Point`还是用户定义的`Point`数组而请求的。程序通过传递实际需要的字节数来请求内存。例如，要为`Point`类型的对象请求空间，可以使用`malloc()`函数如下：

```cpp
#include <cstdlib>
struct Point {
  float x;
  float y;
};

int main() {
 std::malloc(sizeof(Point));
}
```

`malloc()`函数来自 C 语言，使用它需要包含`<cstdlib>`头文件。

`malloc()`函数分配了`sizeof(Point)`字节的连续内存空间，比如说 8 字节。然后它返回该内存的第一个字节的地址，因为这是提供访问空间的唯一方式。而且，`malloc()`实际上并不知道我们是为`Point`对象还是`int`请求了内存空间，它只是简单地返回`void*`。`void*`存储了分配内存的第一个字节的地址，但它绝对不能用于通过解引用指针来获取实际数据，因为`void`没有定义数据的大小。看一下下面的图示；它显示了`malloc`在堆上分配内存：

![](img/762fa3db-fc1e-4137-b71c-ee80cffd43f5.png)

要实际使用内存空间，我们需要将`void`指针转换为所需的类型：

```cpp
void* raw = std::malloc(sizeof(Point)); Point* p = static_cast<Point*>(raw); 
```

或者，只需声明并初始化指针并进行类型转换：

```cpp
Point* p = static_cast<Point*>(std::malloc(sizeof(Point))); 
```

C++通过引入`new`运算符来解决这个头疼的问题，该运算符自动获取要分配的内存空间的大小，并将结果转换为所需的类型：

```cpp
Point* p = new Point;
```

动态内存管理是一个手动过程；没有类似于堆栈的构造，可以在不再需要时自动释放内存空间。为了正确管理内存资源，我们应该在想要释放空间时使用`delete`运算符。我们将在第五章中了解细节，*内存管理和智能指针*。

当我们访问`p`指向的`Point`对象的成员时会发生什么？对`p`进行解引用会返回完整的`Point`对象，所以要更改成员`x`的值，我们应该这样做：

```cpp
(*p).x = 0.24;
```

或者更好的方法是使用箭头运算符访问它：

```cpp
p->x = 0.24;
```

我们将在第三章中特别深入讨论用户定义类型和结构体，*面向对象编程的细节。*

# 数组

数组是一种基本的数据结构，它提供了在内存中连续存储的数据集合。许多适配器，如堆栈，都是使用数组实现的。它们的独特之处在于数组元素都是相同类型的，这在访问数组元素中起着关键作用。例如，以下声明创建了一个包含 10 个整数的数组：

```cpp
int arr[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
```

数组的名称会衰减为指向其第一个元素的指针。考虑到数组元素都是相同类型，我们可以通过推进指针到其第一个元素来访问数组的任何元素。例如，以下代码打印了数组的第三个元素：

```cpp
std::cout << *(arr + 2);
```

第一个元素也是如此；以下三行代码都在做同样的事情：

```cpp
std::cout << *(arr + 0);
std::cout << *arr;
std::cout << arr[0];
```

为了确保`arr[2]`和`*(arr + 2)`做了完全相同的事情，我们可以这样做：

```cpp
std::cout << *(2 + arr);
```

将`2`移到`+`的后面不会影响结果，所以下面的代码也是有效的：

```cpp
std::cout << 2[arr];
```

然后它会打印数组的第三个元素。

数组元素的访问时间是恒定的，这意味着访问数组的第一个和最后一个元素需要相同的时间。这是因为每次访问数组元素时，我们都会执行以下操作：

1.  通过添加相应的数值来推进指针

1.  读取结果指针所指的内存单元的内容

数组的类型指示应读取（或写入）多少个内存单元。以下图示了访问的过程：

![](img/cbb9a70d-2cfc-471f-ae05-4d6e28d2ffca.png)

这个想法在创建动态数组时至关重要，动态数组位于堆而不是堆栈中。正如我们已经知道的，从堆中分配内存会给出其第一个字节的地址，所以除了第一个元素之外，访问其他元素的唯一机会就是使用指针算术：

```cpp
int* arr = new int[10];
arr[4] = 2; // the same as *(arr + 4) = 2 
```

我们将在第六章中更多地讨论数组的结构和其他数据结构，*深入数据结构和 STL 中的算法。*

# 控制流

几乎任何编程语言的最基本概念都是条件语句和循环。我们将详细探讨它们。

# 条件语句

很难想象一个不包含条件语句的程序。检查函数的输入参数以确保它们的安全执行几乎成了一种习惯。例如，`divide()`函数接受两个参数，将一个除以另一个，并返回结果。很明显，我们需要确保除数不为零：

```cpp
int divide(int a, int b) {
 if (b == 0) {
    throw std::invalid_argument("The divisor is zero");
  }
  return a / b;
}
```

条件语句是编程语言的核心；毕竟，程序是一系列动作和决策。例如，以下代码使用条件语句来找出两个输入参数中的最大值：

```cpp
int max(int a, int b) {
  int max;
 if (a > b) {
    // the if block
    max = a;
 } else {
    // the else block
    max = b;
  }
  return max;
}
```

前面的示例是故意过于简化，以表达`if`-`else`语句的使用。然而，最让我们感兴趣的是这样一个条件语句的实现。当编译器遇到`if`语句时会生成什么？CPU 按顺序逐个执行指令，指令是简单的命令，只能执行一个操作。在高级编程语言（如 C++）中，我们可以在一行中使用复杂表达式，而汇编指令是简单的命令，每个周期只能执行一个简单操作：`move`、`add`、`subtract`等。

CPU 从代码存储段中获取指令，对其进行解码以找出它应该做什么（移动数据，加法，减法），然后执行命令。

为了以最快的速度运行，CPU 将操作数和执行结果存储在称为**寄存器**的存储单元中。您可以将寄存器视为 CPU 的临时变量。寄存器是位于 CPU 内部的物理内存单元，因此访问速度比 RAM 快得多。要从汇编语言程序中访问寄存器，我们使用它们的指定名称，如`rax`、`rbx`、`rdx`等。CPU 命令操作寄存器而不是 RAM 单元；这就是 CPU 必须将变量的内容从内存复制到寄存器，执行操作并将结果存储在寄存器中，然后将寄存器的值复制回内存单元的原因。

例如，以下 C++表达式只需要一行代码：

```cpp
a = b + 2 * c - 1;
```

它看起来类似于以下汇编表示（分号后添加了注释）：

```cpp
mov rax, b; copy the contents of "b" 
          ; located in the memory to the register rax
mov rbx, c; the same for the "c" to be able to calculate 2 * c
mul rbx, 2; multiply the value of the rbx register with 
          ; immediate value 2 (2 * c)
add rax, rbx; add rax (b) with rbx (2*c) and store back in the rax
sub rax, 1; subtract 1 from rax
mov a, rax; copy the contents of rax to the "a" located in the memory
```

条件语句表明应跳过代码的一部分。例如，调用`max(11, 22)`意味着`if`块将被省略。为了在汇编语言中表达这一点，使用了跳转的概念。我们比较两个值，并根据结果跳转到代码的指定部分。我们使用标签来标记部分，以便找到一组指令。例如，要跳过将`42`添加到寄存器`rbx`，我们可以使用无条件跳转指令`jpm`*跳转*到标记为`UNANSWERED`的部分，如下所示：

```cpp
mov rax, 2
mov rbx, 0
jmp UNANSWERED
add rbx, 42; will be skipped
UNANSWERED:
  add rax, 1
  ; ...
```

`jmp`指令执行无条件跳转；这意味着它开始执行指定标签下的第一条指令，而不进行任何条件检查。好消息是，CPU 也提供了条件跳转。`max()`函数的主体将转换为以下汇编代码（简化），其中`jg`和`jle`命令被解释为*如果大于*和*如果小于或等于*，分别（基于使用`cmp`指令进行比较的结果）：

```cpp
mov rax, max; copy the "max" into the rax register
mov rbx, a
mov rdx, b
cmp rbx, rdx; compare the values of rbx and rdx (a and b)
jg GREATER; jump if rbx is greater than rdx (a > b)
jl LESSOREQUAL; jump if rbx is lesser than
GREATER:
  mov rax, rbx; max = a
LESSOREQUAL:
  mov rax, rdx; max = b
```

在前面的代码中，标签`GREATER`和`LESSOREQUAL`代表先前实现的`max()`函数的`if`和`else`子句。

# `switch`语句

诸如`switch`语句之类的条件语句使用与上述相同的逻辑：

```cpp
switch (age) {
case 18:
  can_drink = false;
  can_code = true;
  break;
case 21: 
  can_drink = true;
  can_code = true;
 break;
default: 
  can_drink = false;
}
```

假设`rax`表示年龄，`rbx`表示`can_drink`，`rdx`表示`can_code`。前面的示例将转换为以下汇编指令（简化以表达基本思想）：

```cpp
cmp rax, 18
je CASE_18
cmp rax, 21
je CASE_21
je CASE_DEFAULT
CASE_18:
  mov rbx, 0; cannot drink
  mov rdx, 1; can code
  jmp BEYOND_SWITCH; break
CASE_21:
 mov rbx, 1
 mov rdx, 1
 jmp BEYOND_SWITCH
CASE_DEFAULT:
 mov rbx, 0
BEYOND_SWITCH:
  ; ....
```

每个`break`语句都会转换为跳转到`BEYOND_SWITCH`标签，所以如果我们忘记了`break`关键字，例如在`age`为`18`的情况下，执行将会通过`CASE_21`。这就是为什么你不应该忘记`break`语句。

让我们找到一种方法来避免在源代码中使用条件语句，以使代码更短，可能更快。我们将使用函数指针。

# 用函数指针替换条件语句

之前，我们看过内存段，其中最重要的一个是代码段（也称为文本段）。这个段包含程序图像，也就是应该执行的程序指令。指令通常被分组成函数，这些函数提供了一个唯一的名称，允许我们从其他函数中调用它们。函数驻留在可执行文件的代码段中。

一个函数有它的地址。我们可以声明一个指针，取函数的地址，然后稍后使用它来调用该函数：

```cpp
int get_answer() { return 42; }
int (*fp)() = &get_answer;
// int (*fp)() = get_answer; same as &get_answer
```

函数指针可以像原始函数一样被调用：

```cpp
get_answer(); // returns 42
fp(); // returns 42
```

假设我们正在编写一个程序，从输入中获取两个数字和一个字符，并对数字执行算术运算。操作由字符指定，无论是`+`，`-`，`*`，还是`/`。我们实现四个函数，`add()`，`subtract()`，`multiply()`和`divide()`，并根据字符输入的值调用其中一个。

而不是在一堆`if`语句或`switch`语句中检查字符的值，我们将使用哈希表将操作的类型映射到指定的函数：

```cpp
#include <unordered_map>
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }
int divide(int a, int b) { return (b == 0) ? 0 : a / b; }

int main() {
 std::unordered_map<char, int (*)(int, int)> operations;
 operations['+'] = &add;
 operations['-'] = &subtract;
 operations['*'] = &multiply;
 operations['/'] = &divide;
  // read the input 
  char op;
  int num1, num2;
  std::cin >> num1 >> num2 >> op;
  // perform the operation, as follows
 operationsop;
}

```

正如你所看到的，`std::unordered_map`将`char`映射到一个函数指针，定义为`(*)(int, int)`。也就是说，它可以指向任何接受两个整数并返回一个整数的函数。

哈希表由`<unordered_map>`头文件中定义的`std::unordered_map`表示。我们将在第六章中详细讨论它，*深入 STL 中的数据结构和算法*

现在我们不需要写以下内容：

```cpp
if (op == '+') {
  add(num1, num2);
} else if (op == '-') {
  subtract(num1, num2);
} else if (op == '*') {
  ...
```

相反，我们只需调用由字符映射的函数：

```cpp
operationsop;
```

虽然使用哈希表更加美观，看起来更专业，但你应该注意意外情况，比如无效的用户输入。

# 函数作为类型

`unordered_map`的第二个参数是`int (*)(int, int)`，它字面上意味着指向接受两个整数并返回一个整数的函数的指针。C++支持类模板`std::function`作为通用函数包装器，允许我们存储可调用对象，包括普通函数、lambda 表达式、函数对象等。存储的对象被称为`std::function`的目标，如果它没有目标，那么在调用时将抛出`std::bad_function_call`异常。这不仅帮助我们使`operations`哈希表接受任何可调用对象作为它的第二个参数，还帮助我们处理异常情况，比如前面提到的无效字符输入。

以下代码块说明了这一点：

```cpp
#include <functional>
#include <unordered_map>
// add, subtract, multiply and divide declarations omitted for brevity
int main() {
  std::unordered_map<char, std::function<int(int, int)> > operations;
  operations['+'] = &add;
  // ...
}
```

注意`std::function`的参数；它的形式是`int(int, int)`而不是`int(*)(int, int)`。使用`std::function`帮助我们处理异常情况。例如，调用`operations'x';`将导致创建一个空的`std::function`映射到字符`x`。

调用它将抛出异常，因此我们可以通过正确处理调用来确保代码的安全性：

```cpp
// code omitted for brevity
std::cin >> num1 >> num2 >> op;
try {
 operationsop;
} catch (std::bad_function_call e) {
  // handle the exception
  std::cout << "Invalid operation";
}
```

最后，我们可以使用*lambda 表达式* - 在现场构造的未命名函数，能够捕获范围内的变量。例如，我们可以在将其插入哈希表之前创建一个 lambda 表达式，而不是声明前面的函数然后将其插入哈希表中：

```cpp
std::unordered_map<char, std::function<int(int, int)> > operations;
operations['+'] = [](int a, int b) { return a + b; }
operations['-'] = [](int a, int b) { return a * b; }
// ...
std::cin >> num1 >> num2 >> op;
try {
  operationsop;
} catch (std::bad_functional_call e) {
  // ...
}
```

Lambda 表达式将在整本书中涵盖。

# 循环

循环可以被视为可重复的`if`语句，再次应该被转换为 CPU 比较和跳转指令。例如，我们可以使用`while`循环计算从 0 到 10 的数字的和：

```cpp
auto num = 0;
auto sum = 0;
while (num <= 10) {
  sum += num;
  ++num;
}
```

这将转换为以下汇编代码（简化）：

```cpp
mov rax, 0; the sum
mov rcx, 0; the num
LOOP:
  cmp rbx, 10
  jg END; jump to the END if num is greater than 10
  add rax, rcx; add to sum
  inc rcx; increment num
  jmp LOOP; repeat
END:
  ...
```

C++17 引入了可以在条件语句和循环中使用的 init 语句。现在，在`while`循环之外声明的`num`变量可以移入循环中：

```cpp
auto sum = 0;
while (auto num = 0; num <= 10) {
  sum += num;
  ++num;
}
```

相同的规则适用于`if`语句，例如：

```cpp
int get_absolute(int num) {
  if (int neg = -num; neg < 0) {
    return -neg;
  }
  return num;
}
```

C++11 引入了基于范围的`for`循环，使语法更加清晰。例如，让我们使用新的`for`循环调用我们之前定义的所有算术操作：

```cpp
for (auto& op: operations) {
  std::cout << op.second(num1, num2);
}
```

迭代`unordered_map`返回一个带有第一个和第二个成员的 pair，第一个是键，第二个是映射到该键的值。C++17 进一步推动我们，允许我们将相同的循环写成如下形式：

```cpp
for (auto& [op, func]: operations) {
  std::cout << func(num1, num2);
}
```

了解编译器实际生成的内容对于设计和实现高效软件至关重要。我们涉及了条件语句和循环的低级细节，这些细节是几乎每个程序的基础。

# 总结

在本章中，我们介绍了程序执行的细节。我们讨论了函数和`main()`函数及其一些特殊属性。我们了解了递归的工作原理以及`main()`函数不能递归调用。

由于 C++是为数不多支持低级编程概念（例如通过地址访问内存字节）的高级语言之一，我们研究了数据驻留在内存中的方式以及如何在访问数据时可以使用指针。了解这些细节对于专业的 C++程序员来说是必不可少的。

最后，我们从汇编语言的角度讨论了条件语句和循环的主题。在整个章节中，我们介绍了 C++20 的特性。

在下一章中，我们将更多地了解**面向对象编程**（**OOP**），包括语言对象模型的内部细节。我们将深入研究虚函数的细节，并了解如何使用多态性。

# 问题

1.  `main()`函数有多少个参数？

1.  `constexpr`限定符用于什么目的？

1.  为什么建议使用迭代而不是递归？

1.  堆栈和堆之间有什么区别？

1.  如果声明为`int*`，`ptr`的大小是多少？

1.  为什么访问数组元素被认为是一个常数时间操作？

1.  如果我们在`switch`语句的任何情况下忘记了`break`关键字会发生什么？

1.  如何将算术操作示例中的`multiply()`和`divide()`函数实现为 lambda 表达式？

# 进一步阅读

您可以参考以下书籍，了解本章涵盖的主题的更多信息：*C++ High Performance*，作者 Viktor Sehr 和 Bjorn Andrist（[`www.amazon.com/gp/product/1787120953`](https://www.amazon.com/gp/product/1787120953)）。
