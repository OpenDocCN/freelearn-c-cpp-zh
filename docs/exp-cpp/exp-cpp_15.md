# 第十三章：调试和测试

调试和测试在软件开发过程的流水线中扮演着极其重要的角色。测试帮助我们发现问题，而调试修复问题。然而，如果我们在实施阶段遵循一定的规则，就可以预防许多潜在的缺陷。此外，由于测试过程非常昂贵，如果我们能在需要人工测试之前使用某些工具自动分析软件，那将是非常好的。此外，关于软件何时、如何以及应该测试什么也是很重要的。

在本章中，我们将涵盖以下主题：

+   了解问题的根本原因

+   调试 C++程序

+   了解静态和动态分析

+   探索单元测试、TDD 和 BDD

在本章中，我们将学习如何分析软件缺陷，如何使用 GNU 调试器（GDB）工具来调试程序，以及如何使用工具自动分析软件。我们还将学习单元测试、测试驱动开发（TDD）和行为驱动开发（BDD）的概念，以及如何在软件工程开发过程中进行实践。

# 技术要求

本章的代码可以在本书的 GitHub 存储库中找到：[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# 了解问题的根本原因

在医学中，一个好的医生需要理解治疗症状和治愈疾病之间的区别。例如，给一个断臂的病人止痛药只会消除症状；手术可能是帮助骨骼逐渐愈合的正确方式。

根本原因分析（RCA）是一种系统性的过程，用于确定问题的根本原因。借助适当的工具，它试图使用一组特定的步骤来确定问题的根本原因的起源。通过这样做，我们可以确定以下内容：

+   发生了什么？

+   它是如何发生的？

+   为什么会发生？

+   应该采用什么适当的方法来防止或减少它，使其永远不再发生？

RCA 假设一个地方的行动会触发另一个地方的行动，依此类推。通过追溯行动链到开始，我们可以发现问题的根源以及它如何演变成我们所拥有的症状。啊哈！这正是我们应该遵循的修复或减少软件缺陷的过程。在接下来的小节中，我们将学习基本的 RCA 步骤，如何应用 RCA 过程来检测软件缺陷，以及 C++开发人员应该遵循哪些规则来防止软件中出现这样的缺陷。

# RCA 概述

通常，RCA 过程包括以下五个步骤：

1.  定义问题：在这个阶段，我们可能会找到以下问题的答案：发生了什么？问题的症状是什么？问题发生在什么环境或条件下？

1.  收集数据：为了制作因果因素图，我们需要收集足够的数据。这一步可能既昂贵又耗时。

1.  制作因果因素图：因果因素图提供了一个可视化结构，我们可以用它来组织和分析收集到的数据。因果因素图只是一个带有逻辑测试的序列图，解释了导致症状发生的事件。这个图表过程应该驱动数据收集过程，直到调查人员对图表的彻底性感到满意。

1.  确定根本原因：通过检查因果因素图，我们可以制作一个决策图，称为根本原因图，以确定根本原因或原因。

1.  **推荐和实施解决方案**：一旦确定了根本原因或多个原因，以下问题的答案可以帮助我们找到解决方案：我们可以采取什么措施防止问题再次发生？解决方案将如何实施？谁将负责？实施解决方案的成本或风险是什么？

RCA 树图是软件工程行业中最流行的因素图之一。以下是一个示例结构：

![](img/e9263dd4-03a3-4449-90b4-6e7d6d187b5d.png)

假设我们有一个问题，它有**A**、**B**和**C**三种症状。症状**A**可能是由事件**A1**或**A2**引起的，症状**B**可能是由事件**B1**和**B2**或**B3**和**B4**引起的，而症状**C**是由事件**C1**和**C2**引起的。在收集数据后，我们发现症状**A**和**C**从未出现，而我们只有症状**B**。进一步的分析显示，在问题发生时，事件**B1**和**B2**并未涉及，因此我们可以确定这个问题的根本原因是由于事件**B3**或**B4**的发生。

如果软件存在缺陷，我们应该对其应用 RCA，并调查问题的原始根本原因。然后，问题的根本原因可以追溯到需求、设计、实施、验证和/或测试规划和输入数据。当找到并修复了根本原因时，软件的质量可以得到改善，因此维护费用将大大降低。

我们刚刚学会了如何找到问题的根本原因，但请记住，“最好的防御是进攻”。因此，我们可以预防问题的发生，而不是分析和修复问题。

# 预防胜于治疗——良好的编码行为

从成本的角度来看，IBM 的一项研究表明，假设需求和设计的总成本为 1X，那么实施和编码过程将需要 5X，单元和集成测试将需要约 10X，全面的客户测试成本将需要约 15X，而在产品发布后修复错误的成本将占约 30X！因此，最小化代码缺陷是降低生产成本的最有效方法之一。

尽管找到软件缺陷的根本原因的通用方法非常重要，但如果我们能在实施阶段预防一些缺陷，那将更好。为此，我们需要有良好的编码行为，这意味着必须遵循某些规则。这些规则可以分为低级和高级。低级规则可能包括以下内容：

+   未初始化变量

+   整数除法

+   错误地使用`=`而不是`==`

+   可能将有符号变量分配给无符号变量

+   在`switch`语句中缺少`break`

+   复合表达式或函数调用中的副作用

至于高级规则，我们有以下相关主题：

+   接口

+   资源管理

+   内存管理

+   并发

B. Stroustrup 和 H. Sutter 在他们的实时文档*C++ Core Guidelines (Release 0.8)*中建议遵循这些规则，其中强调了静态类型安全和资源安全。他们还强调了范围检查的可能性，以避免解引用空指针、悬空指针和异常的系统使用。如果开发人员遵循这些规则，它将使他/她的代码在静态类型上是安全的，没有任何资源泄漏。此外，它不仅可以捕获更多的编程逻辑错误，而且还可以运行得更快。

由于页面限制，本小节只会介绍一些示例。如果您想查看更多示例，请访问[`isocpp.github.io/CppCoreGuidelines`](https://isocpp.github.io/CppCoreGuidelines)。

# 未初始化变量问题

未初始化的变量是程序员可能犯的最常见的错误之一。当我们声明一个变量时，将为其分配一定数量的连续内存。如果未初始化，它仍然具有一些值，但没有确定性地预测它的方法。因此，当我们执行程序时，会出现不可预测的行为：

```cpp
//ch13_rca_uninit_variable.cpp
#include <iostream>
int main()
{
  int32_t x;
  // ... //do something else but not assign value to x
  if (x>0) {
    std::cout << "do A, x=" << x << std::endl;
  }
  else {
    std::cout << "do B, x=" << x << std::endl;
  }
  return 0;
}
```

在上面的代码中，当声明`x`时，操作系统将为其分配 4 个字节的未使用内存，这意味着`x`的值是驻留在该内存中的任何值。每次运行此程序时，`x`的地址和值可能都不同。此外，一些编译器（如 Visual Studio）将在调试版本中将`x`的值初始化为`0`，但在发布版本中将其保持未初始化。在这种情况下，调试版本和发布版本的输出完全不同。

# 复合表达式中的副作用

当运算符、表达式、语句或函数完成评估后，它可能会被延长或者可能会持续存在于其复合体内。这种持续存在会产生一些副作用，可能导致一些未定义的行为。让我们看一下以下代码来理解这一点：

```cpp
//ch13_rca_compound.cpp
#include <iostream>
int f(int x, int y)
{
  return x*y;
}

int main()
{
  int x = 3;
  std::cout << f(++x, x) << std::endl; //bad,f(4,4) or f(4,3)?
}
```

由于操作数的评估顺序的未定义行为，上述代码的结果可能是 16 或 12。

# 混合有符号和无符号问题

通常，二进制运算符（`+`，`-`，`*`，`/`，`%`，`<`，`<=`，`>`，`>=`，`==`，`!=`，`&&`，`||`，`!`，`&`，`|`，`<<`，`>>`，`~`，`^`，`=`，`+=`，`-=`，`*=`，`/=`，和`%=`）要求两个操作数具有相同的类型。如果两个操作数的类型不同，则一个将被提升为与另一个相同的类型。粗略地说，C 标准转换规则在子条款 6.3.1.1 [ISO/IEC 9899:2011]中给出。

+   当我们混合相同等级的类型时，有符号的类型将被提升为无符号类型。

+   当我们混合不同等级的类型时，如果较低等级的一方的所有值都可以由较高等级的一方表示，那么较低等级的一方将被提升为较高等级的类型。

+   如果在上述情况下较低等级类型的所有值都不能由较高等级类型表示，则将使用较高等级类型的无符号版本。

现在，让我们来看一下传统的有符号整数减去无符号整数的问题：

```cpp
//ch13_rca_mix_sign_unsigned.cpp
#include <iostream>
using namespace std;
int main()
{
 int32_t x = 10;
 uint32_t y = 20;
 uint32_t z = x - y; //z=(uint32_t)x - y
 cout << z << endl; //z=4294967286\. 
}
```

在上面的例子中，有符号的`int`将自动转换为无符号的`int`，结果将是`uint32_t z` = `-10`。另一方面，因为`−10`不能表示为无符号的`int`值，它的十六进制值`0xFFFFFFF6`将被解释为`UINT_MAX - 9`（即`4294967286`）在补码机器上。

# 评估顺序问题

以下示例涉及构造函数中类成员的初始化顺序。由于初始化顺序是类成员在类定义中出现的顺序，因此将每个成员的声明分开到不同的行是一个好的做法：

```cpp
//ch13_rca_order_of_evaluation.cpp
#include <iostream>
using namespace std;

class A {
public:
  A(int x) : v2(v1), v1(x) {
  };
  void print() {
    cout << "v1=" << v1 << ",v2=" << v2 << endl;
  };
protected:
  //bad: the order of the class member is confusing, better
  //separate it into two lines for non-ambiguity order declare   
  int v1, v2; 
};

class B {
public:
  //good: since the initialization order is: v1 -> v2, 
  //after this we have: v1==x, v2==x.
  B(int x) : v1(x), v2(v1) {};

  //wrong: since the initialization order is: v1 -> v2, 
  //after this we have: v1==uninitialized, v2==x. 
  B(float x) : v2(x), v1(v2) {};
  void print() {
    cout << "v1=" << v1 << ", v2=" << v2 << endl;
  };

protected:
  int v1; //good, here the declaration order is clear
  int v2;
};

int main()
{
  A a(10);
  B b1(10), b2(3.0f);
  a.print();  //v1=10,v2=10,v3=10 for both debug and release
  b1.print(); //v1=10, v2=10 for both debug and release
  b2.print(); //v1=-858993460,v2=3 for debug; v1=0,v2=3 for release.
}
```

在类`A`中，尽管声明顺序是`v1 -> v2`，但将它们放在一行中会使其他开发人员感到困惑。在类`B`的第一个构造函数中，`v1`将被初始化为`x`，然后`v2`将被初始化为`v1`，因为其声明顺序是`v1->v2`。然而，在其第二个构造函数中，`v1`将首先被初始化为`v2`（此时，`v2`尚未初始化！），然后`v2`将被`x`初始化。这导致调试版本和发布版本中`v1`的不同输出值。

# 编译时检查与运行时检查

以下示例显示，运行时检查（整数类型变量云的位数）可以被编译时检查替换：

```cpp
//check # of bits for int
//courtesy: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
int nBits = 0; // don't: avoidable code
for (int i = 1; i; i <<= 1){
     ++nBits;
}
if (nBits < 32){
    cerr << "int too small\n";
}
```

由于`int`可以是 16 位或 32 位，这个例子取决于操作系统，无法实现其目标。我们应该使用`int32_t`或者用以下内容替换它：

```cpp
 static_assert(sizeof(int) >= 4); //compile-time check
```

另一个例子涉及将 n 个整数的最大数量读入一维数组中：

```cpp
void read_into(int* p, int n); // a function to read max n integers into *p
...
int v[10];
read_into(v, 100); //bad, off the end, but the compile cannot catch this error.
```

这可以使用`span<int>`来修复：

```cpp
void read_into( span<int> buf); // read into a range of integers
...
int v[10];
read_into(v); //better, the compiler will figure out the number of elements
```

这里的一般规则是尽可能在编译时进行分析，而不要推迟到运行时。

# 避免内存泄漏

内存泄漏意味着分配的动态内存永远无法释放。在 C 中，我们使用`malloc()`和/或`calloc()`来分配内存，使用`free()`来释放它。在 C++中，使用`new`运算符和`delete`或`delete[]`运算符来动态管理内存。尽管智能指针和 RAII 的帮助可以减少内存泄漏的风险，但如果我们希望构建高质量的代码，仍然有一些规则需要遵循。

首先，最简单的内存管理方式是你的代码从未分配的内存。例如，每当你可以写`T x;`时，不要写`T* x = new T();`或`shared_ptr<T> x(new T());`。

接下来，不要使用自己的代码管理内存，如下所示：

```cpp
void f_bad(){
 T* p = new T() ;
  ...                 //do something with p
 delete p ;           //leak if throw or return before reaching this line 
}
```

相反，尝试使用 RAII，如下所示：

```cpp
void f_better()
{
 std::auto_ptr<T> p(new T()) ; //other smart pointers is ok also
 ...                           //do something with p
 //will not leak regardless whether this point is reached or not
}
```

然后，使用`unique_ptr`替换`shared_ptr`，除非你需要共享它的所有权，如下所示：

```cpp
void f_bad()
{
 shared_ptr<Base> b = make_shared<Derived>();
 ...            
} //b will be destroyed at here
```

由于`b`在本地使用而没有复制它，它的`refcount`将始终为`1`。这意味着我们可以使用`unique_ptr`来替换它：

```cpp
void f_better()
{
 unique_ptr<Base> b = make_unique<Derived>();
 ...            //use b locally
}               //b will be destroyed at here
```

最后，即使你真的需要自己动态管理内存，如果有`std container`库类可用，不要手动分配内存。

在本节中，我们学习了如何使用 RCA 定位问题以及如何通过编码最佳实践来预防问题。接下来，我们将学习如何使用调试器工具来控制程序的逐行执行，并在运行时检查变量和表达式的值。

# 调试 C++程序

调试是找到并解决程序问题或缺陷的过程。这可能包括交互式调试、数据/控制流分析以及单元和集成测试。在本节中，我们只关注交互式调试，这是逐行执行源代码并显示正在使用的变量的值及其相应内存地址的过程。

# 调试 C/C++程序的工具

根据你的开发环境，在 C++社区中有很多可用的工具。以下列表显示了不同平台上最受欢迎的工具。

+   Linux/Unix：

+   GDB：一个免费的开源命令行界面（CLI）调试器。

+   Eclipse：一个免费的开源集成开发环境（IDE）。它不仅支持调试，还支持编译、性能分析和智能编辑。

+   Valgrind：另一个开源的动态分析工具；它适用于调试内存泄漏和线程错误。

+   Affinic：一个商业的图形用户界面（GUI）工具，专为 GDB、LLDB 和 LLVM 调试器构建。

+   DDD：一个用于 GDB、DBX、JDB、XDB 和 Python 的开源数据显示调试器，它将数据结构显示为图形。

+   Emacs 模式下的 GDB：一个使用 GNU Emacs 查看和编辑源代码的开源 GUI 工具，用于与 GDB 一起调试。

+   KDevelop：一个用于 C/C++、Objective-等编程语言的免费开源 IDE 和调试器工具。

+   Nemiver：一个在 GNOME 桌面环境中运行良好的开源工具。

+   SlickEdit：一个用于调试多线程和多处理器代码的好工具。

+   Windows：

+   Visual Studio：一个商业工具，社区版本免费提供 GUI。

+   GDB：这也可以在 Windows 上运行，借助 Cygwin 或 MinGW 的帮助。

+   Eclipse：它的 C++开发工具（CDT）可以在 Windows 上使用 MinGW GCC 编译器的工具链进行安装。

+   macOS：

+   LLDB：这是 macOS 上 Xcode 的默认调试器，支持桌面和 iOS 设备及其模拟器上的 C/C++和 Objective-C。

+   GDB：这个 CLI 调试器也被用于 macOS 和 iOS 系统。

+   **Eclipse**：这个使用 GCC 的免费 IDE 适用于 macOS。

由于 GDB 可以在所有平台上运行，我们将在以下子节中向您展示如何使用 GDB。

# GDB 概述

GDB 代表 GNU 调试器，允许开发人员在另一个程序执行时看到*内部发生了什么，或者在另一个程序崩溃时它正在做什么*。GDB 可以做以下四件事情：

+   启动程序并指定可能影响其行为的任何内容。

+   使程序在给定条件下停止。

+   检查程序停止时发生了什么。

+   在运行程序时更改变量的值。这意味着我们可以尝试纠正一个 bug 的影响和/或继续学习另一个 bug 的副作用。

请注意，涉及两个程序或可执行文件：一个是 GDB，另一个是要调试的程序。由于这两个程序可以在同一台机器上或不同的机器上运行，因此我们可能有三种调试类别，如下所示：

+   **本地调试**：两个程序在同一台机器上运行。

+   **远程调试**：GDB 在主机上运行，而调试的程序在远程机器上运行。

+   **模拟器调试**：GDB 在主机上运行，而调试的程序在模拟器上运行。

根据撰写本书时的最新版本（GDB v8.3），GDB 支持的语言包括 C、C++、Objective-C、Ada、Assembly、D、Fortran、Go、OpenCL、Modula-2、Pascal 和 Rust。

由于 GDB 是调试行业中的一种先进工具，功能复杂且功能丰富，因此在本节中不可能学习所有其功能。相反，我们将通过示例来学习最有用的功能。

# GDB 示例

在练习这些示例之前，我们需要通过运行以下代码来检查系统上是否已安装`gdb`：

```cpp
~wus1/chapter-13$ gdb --help 
```

如果显示以下类型的信息，我们将准备好开始：

```cpp
This is the GNU debugger. Usage:
 gdb [options] [executable-file [core-file or process-id]]
 gdb [options] --args executable-file [inferior-arguments ...]

 Selection of debuggee and its files:
 --args Arguments after executable-file are passed to inferior
 --core=COREFILE Analyze the core dump COREFILE.
 --exec=EXECFILE Use EXECFILE as the executable.
 ...
```

否则，我们需要安装它。让我们看看如何在不同的操作系统上安装它：

+   对于基于 Debian 的 Linux：

```cpp
~wus1/chapter-13$ s*udo apt-get install build-essential* 
```

+   对于基于 Redhat 的 Linux：

```cpp
~wus1/chapter-13$***sudo yum install  build-essential***
```

+   对于 macOS：

```cpp
~wus1/chapter-13$***brew install gdb***
```

Windows 用户可以通过 MinGW 发行版安装 GDB。macOS 将需要 taskgated 配置。

然后，再次输入`gdb --help`来检查是否成功安装。

# 设置断点和检查变量值

在以下示例中，我们将学习如何设置断点，继续，步入或跳过函数，打印变量的值，以及如何在`gdb`中使用帮助。源代码如下：

```cpp
//ch13_gdb_1.cpp
#include <iostream>
float multiple(float x, float y);
int main()
{
 float x = 10, y = 20;
 float z = multiple(x, y);
 printf("x=%f, y=%f, x*y = %f\n", x, y, z);
 return 0;
}

float multiple(float x, float y)
{
 float ret = x + y; //bug, should be: ret = x * y;
 return ret;
}
```

正如我们在第三章中提到的*面向对象编程的细节*，让我们以调试模式构建此程序，如下所示：

```cpp
~wus1/chapter-13$ g++ -g ch13_gdb_1.cpp -o ch13_gdb_1.out
```

请注意，对于 g++，`-g`选项意味着调试信息将包含在输出的二进制文件中。如果我们运行此程序，它将显示以下输出：

```cpp
x=10.000000, y=20.000000, x*y = 30.000000
```

现在，让我们使用`gdb`来查看 bug 在哪里。为此，我们需要执行以下命令行：

```cpp
~wus1/chapter-13$ gdb ch13_gdb_1.out
```

通过这样做，我们将看到以下输出：

```cpp
GNU gdb (Ubuntu 8.1-0ubuntu3) 8.1.0.20180409-git
 Copyright (C) 2018 Free Software Foundation, Inc.
 License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
 This is free software: you are free to change and redistribute it.
 There is NO WARRANTY, to the extent permitted by law. Type "show copying"
 and "show warranty" for details.
 This GDB was configured as "aarch64-linux-gnu".
 Type "show configuration" for configuration details.
 For bug reporting instructions, please see:
 <http://www.gnu.org/software/gdb/bugs/>.
 Find the GDB manual and other documentation resources online at:
 <http://www.gnu.org/software/gdb/documentation/>.
 For help, type "help".
 Type "apropos word" to search for commands related to "word"...
 Reading symbols from a.out...done.
 (gdb) 
```

现在，让我们详细了解各种命令：

+   `break`和`run`：如果我们输入`b main`或`break main`并按*Enter*，则会在主函数中插入一个`breakpoint`。然后，我们可以输入`run`或`r`来开始调试程序。在终端窗口中将显示以下信息。在这里，我们可以看到我们的第一个`breakpoint`在源代码的第六行，调试程序已经暂停以等待新命令：

```cpp
(gdb) b main
Breakpoint 1 at 0x8ac: file ch13_gdb_1.cpp, line 6.
(gdb) r
Starting program: /home/nvidia/wus1/Chapter-13/a.out
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1". 

Breakpoint 1, main () at ch13_gdb_1.cpp:6
6 float x = 10, y = 20;
```

+   `next`，`print`和`quit`：`n`或`next`命令将转到代码的下一行。如果该行调用子例程，则不会进入子例程；相反，它会跳过调用并将其视为单个源行。如果我们想显示变量的值，我们可以使用`p`或`print`命令，后跟变量的名称。最后，如果我们想退出`gdb`，可以使用`q`或`quit`命令。运行这些操作后，以下是终端窗口的输出：

```cpp
(gdb) n
 7 float z = multiple(x, y);
 (gdb) p z
 $1 = 0
 (gdb) n
 8 printf("x=%f, y=%f, x*y = %f\n", x, y, z);
 (gdb) p z
 $2 = 30
 (gdb) q
 A debugging session is active.
 Inferior 1 [process 29187] will be killed.
 Quit anyway? (y or n) y
 ~/wus1/Chapter-13$
```

+   `step`：现在让我们学习如何进入`multiple()`函数并找到错误。为此，我们需要使用`b`、`r`和`n`命令首先到达第 7 行。然后，我们可以使用`s`或`step`命令进入`multiple()`函数。接下来，我们使用`n`命令到达第 14 行，使用`p`打印`ret`变量的值，即 30。到目前为止，我们已经发现，通过使用`ahha the bug is at line 14!:`，而不是`x*y`，我们有一个拼写错误，即`x+y`。以下代码块是这些命令的相应输出：

```cpp
~/wus1/Chapter-13$gdb ch13_gdb_1.out
 ...
 (gdb) b main
 Breakpoint 1 at 0x8ac: file ch13_gdb_1.cpp, line 6.
 (gdb) r
 The program being debugged has been started already.
 Start it from the beginning? (y or n) y
 Starting program: /home/nvidia/wus1/Chapter-13/a.out
 [Thread debugging using libthread_db enabled]
 Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".                                                                                Breakpoint 1, main () at ch13_gdb_1.cpp:6
 6 float x = 10, y = 20;
 (gdb) n
 7 float z = multiple(x, y);
 (gdb) s
 multiple (x=10, y=20) at ch13_gdb_1.cpp:14
 14 float s = x + y;
 (gdb) n
 15 return s;
 (gdb) p s
 $1 = 30
```

+   `help`：最后，让我们学习如何使用`help`命令来结束这个小例子。当启动`gdb`时，我们可以使用`help`或`h`命令来获取特定命令的使用信息。例如，以下终端窗口总结了我们到目前为止学到的内容：

```cpp
(gdb) h b      
 Set breakpoint at specified location.
 break [PROBE_MODIFIER] [LOCATION] [thread THREADNUM] [if CONDITION]
 PROBE_MODIFIER shall be present if the command is to be placed in a
 probe point. Accepted values are `-probe' (for a generic, automatically
 guessed probe type), `-probe-stap' (for a SystemTap probe) or
 `-probe-dtrace' (for a DTrace probe).
 LOCATION may be a linespec, address, or explicit location as described
 below.
  ....

 (gdb) h r
 Start debugged program.
 You may specify arguments to give it.
 Args may include "*", or "[...]"; they are expanded using the
 shell that will start the program (specified by the "$SHELL" environment
 variable). Input and output redirection with ">", "<", or ">>"
 are also allowed.

 (gdb) h s
 Step program until it reaches a different source line.
 Usage: step [N]
 Argument N means step N times (or till program stops for another reason).

 (gdb) h n
 Step program, proceeding through subroutine calls.
 Usage: next [N]
 Unlike "step", if the current source line calls a subroutine,
 this command does not enter the subroutine, but instead steps over
 the call, in effect treating it as a single source line.

 (gdb) h p
 Print value of expression EXP.
 Variables accessible are those of the lexical environment of the selected
 stack frame, plus all those whose scope is global or an entire file.

 (gdb) h h
 Print list of commands.
 (gdb) h help
 Print list of commands.
 (gdb) help h
 Print list of commands.
 (gdb) help help
 Print list of commands.
```

到目前为止，我们已经学习了一些基本命令，可以用来调试程序。这些命令是`break`、`run`、`next`、`print`、`quit`、`step`和`help`。我们将在下一小节学习函数和条件断点、观察点，以及`continue`和`finish`命令。

# 函数断点、条件断点、观察点，以及继续和完成命令

在这个例子中，我们将学习如何设置函数断点、条件断点，并使用`continue`命令。然后，我们将学习如何在不需要逐步执行所有代码行的情况下完成函数调用。源代码如下：

```cpp
//ch13_gdb_2.cpp
#include <iostream>

float dotproduct( const float *x, const float *y, const int n);
int main()
{
 float sxx,sxy;
 float x[] = {1,2,3,4,5};
 float y[] = {0,1,1,1,1};

 sxx = dotproduct( x, x, 5);
 sxy = dotproduct( x, y, 5);
 printf( "dot(x,x) = %f\n", sxx );
 printf( "dot(x,y) = %f\n", sxy );
 return 0;
}

float dotproduct( const float *x, const float *y, const int n )
{
 const float *p = x;
 const float *q = x;  //bug: replace x by y
 float s = 0;
 for(int i=0; i<n; ++i, ++p, ++q){
        s += (*p) * (*q);
 }
 return s;
}
```

再次，构建并运行`ch13_gdb_2.cpp`后，我们得到以下输出：

```cpp
~/wus1/Chapter-13$ g++ -g ch13_gdb_2.cpp -o ch13_gdb_2.out
~/wus1/Chapter-13$ ./ch13_gdb_2.out
dot(x,x) = 55.000000
dot(x,y) = 55.000000
```

由于`dot(x,x)`和`dot(x,y)`都给我们相同的结果，这里一定有问题。现在，让我们通过学习如何在`dot()`函数中设置断点来调试它：

+   **函数断点**：要在函数的开头设置断点，我们可以使用`b function_name`命令。和往常一样，在输入时可以使用制表符补全。例如，假设我们输入以下内容：

```cpp
(gdb) b dot<Press TAB Key>
```

如果我们这样做，以下命令行将自动弹出：

```cpp
(gdb) b dotproduct(float const*, float const*, int)
```

如果它是一个类的成员函数，它的类名应该包括在内，如下所示：

```cpp
(gdb) b MyClass::foo(<Press TAB key>
```

+   **条件断点**：有几种设置条件断点的方法：

```cpp
(gdb) b f.cpp:26 if s==0 //set a breakpoint in f.cpp, line 26 if s==0
(gdb) b f.cpp:20 if ((int)strcmp(y, "hello")) == 0 
```

+   **列出和删除断点**：一旦我们设置了几个断点，我们可以列出或删除它们，如下所示：

```cpp
(gdb) i b (gdb) delete breakpoints 1 (gdb) delete breakpoints 2-5
```

+   **删除使断点无条件**：由于每个断点都有一个编号，我们可以删除断点的条件，如下所示：

```cpp
(gdb) cond 1         //break point 1 is unconditional now
```

+   **观察点**：观察点可以在表达式的值发生变化时停止执行，而不必预测它可能发生的位置（在哪一行）。有三种观察点：

+   `watch`：当写入发生时，`gdb`将中断。

+   `rwatch`：当读取发生时，`gdb`将中断。

+   `awatch`：当发生写入或读取时，`gdb`将中断。

以下代码显示了一个例子：

```cpp
(gdb) watch v                 //watch the value of variable v
(gdb) watch *(int*)0x12345678 //watch an int value pointed by an address
(gdb) watch a*b + c/d         // watch an arbitrarily complex expression
```

+   **继续**：当我们在断点处检查变量的值后，我们可以使用`continue`或`c`命令来继续程序执行，直到调试器遇到断点、信号、错误或正常进程终止。

+   **完成**：一旦我们进入一个函数，我们可能希望连续执行它，直到返回到其调用行。这可以使用`finish`命令来完成。

现在，让我们将这些命令组合在一起来调试`ch13_gdb_2.cpp`。以下是我们终端窗口的输出。为了方便起见，我们将其分为三部分：

```cpp
//gdb output of example ch13_gdb_2.out -- part 1
~/wus1/Chapter-13$ gdb ch13_gdb_2.out                     //cmd 1
 ...
 Reading symbols from ch13_gdb_2.out ... done.

 (gdb) b dotproduct(float const*, float const*, int)      //cmd 2
 Breakpoint 1 at 0xa5c: file ch13_gdb_2.cpp, line 20.
 (gdb) b ch13_gdb_2.cpp:24 if i==1                        //cmd 3
 Breakpoint 2 at 0xa84: file ch13_gdb_2.cpp, line 24.
 (gdb) i b                                                //cmd 4
 Num Type Disp Enb Address What
 1 breakpoint keep y 0x0000000000000a5c in dotproduct(float const*, float const*, int) at ch13_gdb_2.cpp:20
 2 breakpoint keep y 0x0000000000000a84 in dotproduct(float const*, float const*, int) at ch13_gdb_2.cpp:24
 stop only if i==1
 (gdb) cond 2                                            //cmd 5
 Breakpoint 2 now unconditional.
 (gdb) i b                                               //cmd 6
 Num Type Disp Enb Address What
 1 breakpoint keep y 0x0000000000000a5c in dotproduct(float const*, float const*, int) at ch13_gdb_2.cpp:20
 2 breakpoint keep y 0x0000000000000a84 in dotproduct(float const*, float const*, int) at ch13_gdb_2.cpp:24 
```

第一部分，我们有以下六个命令：

+   `cmd 1`：我们使用构建的可执行文件`ch13_gdb_2.out`启动`gdb`。这简要显示了它的版本和文档和使用信息，然后告诉我们读取符号的过程已经完成，并且正在等待下一个命令。

+   `cmd 2`：我们设置了一个`断点`函数（在`dotproduct()`处）。

+   `cmd 3`：设置了一个条件`断点`。

+   `cmd 4`: 它列出了关于断点的信息，并告诉我们有两个断点。

+   `cmd 5`: 我们将`breakpoint 2`设置为`无条件的`。

+   `cmd 6`: 再次列出断点信息。此时，我们可以看到两个断点。它们分别位于`ch13_gdb_2.cp`文件的第 20 行和第 24 行。

接下来，让我们看看第二部分的`gdb`输出：

```cpp
//gdb output of example ch13_gdb_2.out -- part 2 
(gdb) r                                                //cmd 7
 Starting program: /home/nvidia/wus1/Chapter-13/ch13_gdb_2.out
 [Thread debugging using libthread_db enabled]
 Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".

 Breakpoint 1, dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:20
 20 const float *p = x;
 (gdb) p x                                            //cmd 8
 $1 = (const float *) 0x7fffffed68
 (gdb) c                                              //cmd 9 
 Continuing.

 Breakpoint 2, dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:24
 24 s += (*p) * (*q);
 (gdb) p i                                           //cmd 10
 $2 = 0
 (gdb) n                                             //cmd 11
 23 for(int i=0; i<n; ++i, ++p, ++q){
 (gdb) n                                             //cmd 12

 Breakpoint 2, dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:24
 24 s += (*p) * (*q);
 (gdb) p s                                           //cmd 13 
 $4 = 1
 (gdb) watch s                                       //cmd 14 
 Hardware watchpoint 3: s
```

第二部分有以下命令：

+   `cmd 7`: 通过给出`run`命令，程序开始运行，并在第 20 行的第一个断点处停止。

+   `cmd 8`: 我们打印`x`的值，显示其地址。

+   `cmd 9`: 我们继续程序。一旦继续，它会在第 24 行的第二个断点处停止。

+   `cmd 10`: 打印`i`的值，为`0`。

+   `cmd 11-12`: 我们两次使用`next`命令。在这一点上，执行`s += (*p) * (*q)`语句。

+   `cmd 13`: 打印`s`的值，为`1`。

+   `cmd 14`: 我们打印`s`的值。

最后，第三部分如下：

```cpp
//gdb output of example ch13_gdb_2.out -- part 3 
(gdb) n                                             //cmd 15 
  Hardware watchpoint 3: s

 Old value = 1
 New value = 5
 dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:23
 23 for(int i=0; i<n; ++i, ++p, ++q){
 (gdb) finish                                       //cmd 16
 Run till exit from #0 dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:23

 Breakpoint 2, dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:24
 24 s += (*p) * (*q);
 (gdb) delete breakpoints 1-3                       //cmd 17
 (gdb) c                                            //cmd 18
 Continuing.

 dot(x,x) = 55.000000
 dot(x,y) = 55.000000
 [Inferior 1 (process 31901) exited normally]
 [Inferior 1 (process 31901) exited normally]
 (gdb) q                                           //cmd 19
 ~/wus1/Chapter-13$
```

在这一部分，我们有以下命令：

+   `cmd 15`: 我们使用`next`命令来查看如果执行下一次迭代时`s`的值是多少。它显示旧值为`s`为`1`（s = 1*1），新值为`5`（s=1*1+2*2）。到目前为止，一切顺利！

+   `cmd 16`: 使用`finish`命令继续运行程序，直到退出函数。

+   `cmd 17`: 删除断点 1 到 3。

+   `cmd 18`: 使用`continue`命令。

+   `cmd 19`: 我们退出`gdb`，回到终端窗口。

# 将 gdb 记录到文本文件中

处理长堆栈跟踪或多线程堆栈跟踪时，从终端窗口查看和分析`gdb`输出可能会不方便。然而，我们可以先将整个会话或特定输出记录到文本文件中，然后稍后离线使用其他文本编辑工具进行浏览。为此，我们需要使用以下命令：

```cpp
(gdb) set logging on
```

当我们执行此命令时，`gdb`将把所有终端窗口输出保存到名为`gdb.txt`的文本文件中，该文件位于当前运行的`gdb`文件夹中。如果我们想停止记录，只需输入以下内容：

```cpp
(gdb) set logging off
```

关于 GDB 的一大好处是，我们可以随意多次打开和关闭日志记录命令，而不必担心转储文件名。这是因为所有输出都被连接到`gdb.txt`文件中。

以下是返回`ch13_gdb_2.out`并将`gdb`输出转储的示例：

```cpp
~/wus1/Chapter-13$ gdb ch13_gdb_2.out           //cmd 1
 ...
Reading symbols from ch13_gdb_2.out...done.
 (gdb) set logging on                           //cmd 2
 Copying output to gdb.txt.
 (gdb) b ch13_gdb_2.cpp:24 if i==1              //cmd 3 
 Breakpoint 1 at 0xa84: file ch13_gdb_2.cpp, line 24.
 (gdb) r                                        //cmd 4 
 ...
 Breakpoint 1, dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:24
 24 s += (*p) * (*q);
 (gdb) p i                                      //cmd 5 
 $1 = 1
 (gdb) p s                                      //cmd 6 
 $2 = 1
 (gdb) finish                                   //cmd 7 
 Run till exit from #0 dotproduct (x=0x7fffffed68, y=0x7fffffed68, n=5) at ch13_gdb_2.cpp:24
 0x00000055555559e0 in main () at ch13_gdb_2.cpp:11
 11 sxx = dotproduct( x, x, 5);
 Value returned is $3 = 55
 (gdb) delete breakpoints 1                    //cmd 8
 (gdb) set logging off                         //cmd 9
 Done logging to gdb.txt.
 (gdb) c                                       //cmd 10 
 Continuing.
 dot(x,x) = 55.000000
 dot(x,y) = 55.000000
 [Inferior 1 (process 386) exited normally]
 (gdb) q                                      //cmd 11
 ~/wus1/Chapter-13$ cat gdb.txt               //cmd 12
```

在前面的代码中使用的命令如下：

+   `cmd 1`: 启动`gdb`。

+   `cmd 2`: 我们将日志标志设置为打开。此时，`gdb`表示输出将被复制到`gdb.txt`文件中。

+   `cmd 3`: 设置条件`断点`。

+   `cmd 4`: 我们运行程序，当它到达第 24 行的条件`断点`时停止。

+   `cmd 5`和`cmd 6`: 我们分别打印`i`和`s`的值。

+   `cmd 7`: 通过执行函数步出命令，显示`sxx`为`55`（在调用`sxx=dotproduct(x, x, 5))`后），程序停在`sxy *=* dotproduct(x, y, 5)`行。

+   `cmd 8`: 我们删除`breakpoint 1`。

+   `cmd 9`: 我们将日志标志设置为关闭。

+   `cmd 10`: 一旦给出继续指令，它就会从`main`函数中运行出来，`gdb`等待新命令。

+   `cmd 11`: 我们输入`q`退出`gdb`。

+   `cmd 12`: 当返回到终端窗口时，通过在操作系统中运行`cat`命令打印已记录的`gdb.txt`文件的内容。

到目前为止，我们已经学会了足够的 GDB 命令来调试程序。正如你可能已经注意到的，这是耗时的，因此非常昂贵。有时，由于在错误的地方调试，情况变得更糟。为了高效地调试，我们需要遵循正确的策略。我们将在下一小节中介绍这一点。

# 实用调试策略

由于调试是软件开发生命周期中成本最高的阶段，发现错误并修复它们是不可行的，特别是对于大型复杂系统。然而，有一些策略可以在实际过程中使用，其中一些如下：

+   **使用 printf()或 std::cout**：这是一种老式的做法。通过将一些信息打印到终端，我们可以检查变量的值，并执行进一步分析的位置和时间种类的日志配置文件。

+   **使用调试器**：虽然学习使用 GDB 这类调试器工具不是一蹴而就的事情，但它可以节省大量时间。因此，逐步熟悉它，并逐渐掌握。

+   **重现错误**：每当在现场报告错误时，记录运行环境和输入数据。

+   **转储日志文件**：应用程序应将日志消息转储到文本文件中。发生崩溃时，我们应首先检查日志文件，以查看是否发生异常事件。

+   **猜测**：粗略猜测错误的位置，然后证明它是对还是错。

+   **分而治之**：即使在最糟糕的情况下，我们对存在什么错误一无所知，我们仍然可以使用**二分搜索**策略设置断点，然后缩小范围，最终定位它们。

+   **简化**：始终从最简化的情景开始，逐渐添加外围设备、输入模块等，直到可以重现错误。

+   **源代码版本控制**：如果一个错误突然出现在一个发布版上，但之前运行正常，首先检查源代码树。可能有人做了改变！

+   **不要放弃**：有些错误真的很难定位和/或修复，特别是对于复杂和多团队参与的系统。暂时搁置它们，回家的路上重新思考一下，也许会有*灵光一现*。

到目前为止，我们已经学习了如何使用 RCA 进行宏观问题定位，以及我们可以遵循的良好编码实践，以防止问题发生。此外，通过使用诸如 GDB 之类的最先进的调试器工具，我们可以逐行控制程序的执行，以便我们可以在微观级别分析和解决问题。所有这些活动都是程序员集中和手动的。是否有任何自动工具可以帮助我们诊断程序的潜在缺陷？我们将在下一节中看一下静态和动态分析。

# 理解静态和动态分析

在前几节中，我们学习了根本原因分析过程以及如何使用 GDB 调试缺陷。本节将讨论如何分析程序，无论是否执行。前者称为动态分析，而后者称为静态分析。

# 静态分析

静态分析评估计算机程序的质量，而无需执行它。虽然通常可以通过自动工具和代码审查/检查来完成，但本节我们只关注自动工具。

自动静态代码分析工具旨在分析一组代码与一个或多个编码规则或指南。通常，人们可以互换使用静态代码分析、静态分析或源代码分析。通过扫描每个可能的代码执行路径的整个代码库，我们可以在测试阶段之前找到许多潜在的错误。然而，它也有一些限制，如下所示：

+   它可能会产生错误的阳性和阴性警报。

+   它只应用于扫描算法内部实施的规则，其中一些可能会被主观解释。

+   它无法找到在运行时环境中引入的漏洞。

+   它可能会产生一种虚假的安全感，认为一切都在得到解决。

在商业和免费开源类别下，有大约 30 个自动 C/C++代码分析工具[9]。这些工具的名称包括 Clang、Clion、CppCheck、Eclipse、Visual Studio 和 GNU g++等。作为示例，我们想介绍内置于 GNU 编译器 g++[10]中的`**-**Wall`、`-Weffcc++`和`-Wextra`选项：

+   `-Wall`：启用所有构造警告，对于某些用户来说是有问题的。这些警告很容易避免或修改，即使与宏一起使用。它还启用了一些在 C++方言选项和 Objective-C/C++方言选项中描述的特定于语言的警告。

+   `-Wextra`：正如其名称所示，它检查一些`-Wall`未检查的额外警告标志。将打印以下任何情况的警告消息：

+   将指针与整数零使用`<`、`<=`、`>`或`>=`操作数进行比较。

+   非枚举和枚举在条件表达式中出现。

+   虚拟基类不明确。

+   对`register`类型数组进行下标操作。

+   使用`register`类型变量的地址。

+   派生类的复制构造函数未初始化其基类。注意（b）-（f）仅适用于 C++。

+   `-Weffc++`：它检查了 Scott Meyers 所著的*Effective and More Effective C++*中建议的一些准则的违反。这些准则包括以下内容：

+   为具有动态分配内存的类定义复制构造函数和赋值运算符。

+   在构造函数中，优先使用初始化而不是赋值。

+   在基类中使析构函数虚拟。

+   使`=`运算符返回对`*this`的引用。

+   当必须返回对象时，不要尝试返回引用。

+   区分增量和减量运算符的前缀和后缀形式。

+   永远不要重载`&&`、`||`或`,`。

为了探索这三个选项，让我们看下面的例子：

```cpp
//ch13_static_analysis.cpp
#include <iostream>
int *getPointer(void)
{
    return 0;
}

int &getVal() {
    int x = 5;
    return x;
}

int main()
{
    int *x = getPointer();
    if( x> 0 ){
        *x = 5;
   }
   else{
       std::cout << "x is null" << std::endl;
   }

   int &y = getVal();
   std::cout << y << std::endl;
   return 0;
}
```

首先，让我们不使用任何选项来构建它：

```cpp
g++ -o ch13_static.out ch13_static_analysis.cpp 
```

这可以成功构建，但是如果我们运行它，预期会出现**段错误**（**核心已转储**）消息。

接下来，让我们添加`-Wall`、`-Weffc++`和`-Wextra`选项并重新构建它：

```cpp
g++ -Wall -o ch13_static.out ch13_static_analysis.cpp
g++ -Weffc++ -o ch13_static.out ch13_static_analysis.cpp
g++ -Wextra -o ch13_static.out ch13_static_analysis.cpp
```

`-Wall`和`-Weffc++`都给出了以下消息：

```cpp
ch13_static_analysis.cpp: In function ‘int& getVal()’:
ch13_static_analysis.cpp:9:6: warning: reference to local variable ‘x’ returned [-Wreturn-local-addr]
int x = 5;
 ^
```

在这里，它抱怨在`int & getVal()`函数（`cpp`文件的第 9 行）中返回了对局部变量的引用。这不起作用，因为一旦程序退出函数，`x`就是垃圾（`x`的生命周期仅限于函数的范围内）。引用一个已经失效的变量是没有意义的。

`-Wextra`给出了以下消息：

```cpp
 ch13_static_analysis.cpp: In function ‘int& getVal()’:
 ch13_static_analysis.cpp:9:6: warning: reference to local variable ‘x’ returned [-Wreturn-local-addr]
 int x = 5;
 ^
 ch13_static_analysis.cpp: In function ‘int main()’:
 ch13_static_analysis.cpp:16:10: warning: ordered comparison of pointer with integer zero [-Wextra]
 if( x> 0 ){
 ^
```

前面的输出显示，`*-*Wextra`不仅给出了`-Wall`的警告，还检查了我们之前提到的六件事。在这个例子中，它警告我们代码的第 16 行存在指针和整数零的比较。

现在我们知道了如何在编译时使用静态分析选项，我们将通过执行程序来了解动态分析。

# 动态分析

*动态分析*是*动态程序分析*的简称，它通过在真实或虚拟处理器上执行软件程序来分析软件程序的性能。与静态分析类似，动态分析也可以自动或手动完成。例如，单元测试、集成测试、系统测试和验收测试通常是人为参与的动态分析过程。另一方面，内存调试、内存泄漏检测和 IBM purify、Valgrind 和 Clang sanitizers 等性能分析工具是自动动态分析工具。在本小节中，我们将重点关注自动动态分析工具。

动态分析过程包括准备输入数据、启动测试程序、收集必要的参数和分析其输出等步骤。粗略地说，动态分析工具的机制是它们使用代码插装和/或模拟环境来对分析的代码进行检查。我们可以通过以下方式与程序交互：

+   源代码插装：在编译之前，将特殊的代码段插入原始源代码中。

+   **目标代码插装**：将特殊的二进制代码直接添加到可执行文件中。

+   **编译阶段插装**：通过特殊的编译器开关添加检查代码。

+   它不会改变源代码。相反，它使用特殊的执行阶段库来检测错误。

动态分析有以下优点：

+   没有错误预测的模型，因此不会出现假阳性或假阴性结果。

+   它不需要源代码，这意味着专有代码可以由第三方组织进行测试。

动态分析的缺点如下：

+   它只能检测与输入数据相关的路径上的缺陷。其他缺陷可能无法被发现。

+   它一次只能检查一个执行路径。为了获得完整的图片，我们需要尽可能多地运行测试。这需要大量的计算资源。

+   它无法检查代码的正确性。可能会从错误的操作中得到正确的结果。

+   在真实处理器上执行不正确的代码可能会产生意想不到的结果。

现在，让我们使用 Valgrind 来找出以下示例中给出的内存泄漏和越界问题：

```cpp
//ch13_dynamic_analysis.cpp
#include <iostream>
int main()
{
    int n=10;
    float *p = (float *)malloc(n * sizeof(float));
    for( int i=0; i<n; ++i){
        std::cout << p[i] << std::endl;
    }
    //free(p);  //leak: free() is not called
    return 0;
}
```

要使用 Valgrind 进行动态分析，需要执行以下步骤：

1.  首先，我们需要安装`valgrind`。我们可以使用以下命令来完成：

```cpp
sudo apt install valgrind //for Ubuntu, Debian, etc.
```

1.  安装成功后，我们可以通过传递可执行文件作为参数以及其他参数来运行`valgrind`，如下所示：

```cpp
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
 --verbose --log-file=valgrind-out.txt ./myExeFile myArgumentList
```

1.  接下来，让我们构建这个程序，如下所示：

```cpp
g++ -o ch13_dyn -std=c++11 -Wall ch13_dynamic_analysis.cpp
```

1.  然后，我们运行`valgrind`，如下所示：

```cpp
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
 --verbose --log-file=log.txt ./ch13_dyn
```

最后，我们可以检查`log.txt`的内容。粗体和斜体行表示内存泄漏的位置和大小。通过检查地址（`0x4844BFC`）及其对应的函数名（`main()`），我们可以看到这个`malloc`在`main()`函数中：

```cpp
... //ignore many lines at begining
 by 0x108A47: main (in /home/nvidia/wus1/Chapter-13/ch13_dyn)
 ==18930== Uninitialised value was created by a heap allocation
 ==18930== at 0x4844BFC: malloc (in /usr/lib/valgrind/vgpreload_memcheck-arm64-linux.so)
 ... //ignore many lines in middle
 ==18930== HEAP SUMMARY:
 ==18930== in use at exit: 40 bytes in 1 blocks
 ==18930== total heap usage: 3 allocs, 2 frees, 73,768 bytes allocated
 ==18930==
 ==18930== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
 ==18930== at 0x4844BFC: malloc (in /usr/lib/valgrind/vgpreload_memcheck-arm64-linux.so)
 ==18930==
 ==18930== LEAK SUMMARY:
 ==18930== definitely lost: 40 bytes in 1 blocks
 ==18930== indirectly lost: 0 bytes in 0 blocks
 ==18930== possibly lost: 0 bytes in 0 blocks
 ==18930== still reachable: 0 bytes in 0 blocks
 ==18930== suppressed: 0 bytes in 0 blocks
```

在这里，我们可以看到`malloc()`被调用来在地址`0x4844BFC`分配一些内存。堆摘要部分表明我们在`0x4844BFC`处有 40 字节的内存泄漏。最后，泄漏摘要部分显示肯定有一个 40 字节的内存泄漏。通过在`log.txt`文件中搜索`0x4844BFC`的地址值，我们最终发现原始代码中没有调用`free(p)`行。取消注释此行后，我们重新进行`valgrind`分析，以便泄漏问题现在已经不在报告中。

总之，借助静态和动态分析工具，程序的潜在缺陷可以自动大大减少。然而，为了确保软件的质量，人类必须参与最终的测试和评估。现在，我们将探讨软件工程中的单元测试、测试驱动开发和行为驱动开发概念。

# 探索单元测试、TDD 和 BDD

在上一节中，我们了解了自动静态和动态程序分析。本节将重点介绍人为参与（准备测试代码）的测试，这是动态分析的另一部分。这些是单元测试、测试驱动开发和行为驱动开发。

单元测试假设如果我们已经有了单个代码单元，那么我们需要编写一个测试驱动程序并准备输入数据来检查其输出是否正确。之后，我们进行集成测试来测试多个单元一起，然后进行验收测试，测试整个应用程序。由于集成和验收测试比单元测试更难维护且与项目更相关，因此在本书中很难覆盖它们。有兴趣的人可以通过访问[`www.iso.org/standard/45142.html`](https://www.iso.org/standard/45142.html)了解更多信息。

与单元测试相比，TDD 认为我们应该先有测试代码和数据，快速开发一些代码并使其通过，最后进行重构，直到客户满意。另一方面，BDD 认为我们不应该测试程序的实现，而是测试其期望的行为。为此，BDD 强调应该建立一个软件生产相关人员之间的沟通平台和语言。

我们将在以下小节中详细讨论这些方法。

# 单元测试

单元是更大或更复杂应用程序中的一个单独组件。通常，一个单元有自己的用户界面，例如函数、类或整个模块。单元测试是一种软件测试方法，用于确定代码单元是否按照其设计要求的预期行为。单元测试的主要特点如下：

+   它小巧简单，编写和运行速度快，因此可以在早期开发周期中发现问题，因此问题可以很容易地修复。

+   由于它与依赖项隔离，因此每个测试用例都可以并行运行。

+   单元测试驱动程序帮助我们理解单元接口。

+   当测试单元后集成时，它极大地帮助集成和验收测试。

+   通常由开发人员准备和执行。

虽然我们可以从头开始编写一个单元测试包，但社区中已经开发了许多**单元测试框架**（**UTFs**）。Boost.Test、CppUnit、GoogleTest、Unit++和 CxxTest 是最受欢迎的。这些 UTF 通常提供以下功能：

+   只需要最少的工作来设置一个新的测试。

+   它们依赖于标准库并支持跨平台，这意味着它们易于移植和修改。

+   它们支持测试固定装置，允许我们为多个不同的测试重用相同的对象配置。

+   它们很好地处理异常和崩溃。这意味着 UTF 可以报告异常，但不能崩溃。

+   它们具有良好的断言功能。每当断言失败时，应打印其源代码位置和变量的值。

+   它们支持不同的输出，这些输出可以方便地由人类或其他工具进行分析。

+   它们支持测试套件，每个套件可能包含多个测试用例。

现在，让我们来看一个 Boost UTF 的例子（自 v1.59.0 起）。它支持三种不同的使用变体：仅单头文件变体、静态库变体和共享库变体。它包括四种类型的测试用例：无参数的测试用例、数据驱动的测试用例、模板测试用例和参数化的测试用例。

它还有七种检查工具：`BOOST_TEST()`、`BOOST_CHECK()`、`BOOST_REQUIRE(`)、`BOOST_ERROR()`、`BOOST_FAIL()`、`BOOST_CHECK_MESSAGE( )`和`BOOST_CHECK_EQUAL()`。它还支持固定装置，并以多种方式控制测试输出。编写测试模块时，我们需要遵循以下步骤：

1.  定义我们的测试程序的名称。这将在输出消息中使用。

1.  选择一个使用变体：仅头文件、链接静态文件或作为共享库。

1.  选择并添加一个测试用例到测试套件中。

1.  对被测试代码执行正确性检查。

1.  在每个测试用例之前初始化被测试的代码。

1.  自定义测试失败报告的方式。

1.  控制构建测试模块的运行时行为，也称为运行时配置。

例如，以下示例涵盖了*步骤 1-4*。如果您感兴趣，可以在[`www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html`](https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html)获取*步骤 5-7*的示例：

```cpp
//ch13_unit_test1.cpp
#define BOOST_TEST_MODULE my_test //item 1, "my_test" is module name
#include <boost/test/included/unit_test.hpp> //item 2, header-only

//declare we begin a test suite and name it "my_suite "
BOOST_AUTO_TEST_SUITE( my_suite ) 

//item 3, add a test case into test suit, here we choose 
//        BOOST_AUTO_TEST_CASE and name it "test_case1" 
BOOST_AUTO_TEST_CASE(test_case1) {
 char x = 'a';
 BOOST_TEST(x);        //item 4, checks if c is non-zero
 BOOST_TEST(x == 'a'); //item 4, checks if c has value 'a'
 BOOST_TEST(x == 'b'); //item 4, checks if c has value 'b'
}

//item 3, add the 2nd test case
BOOST_AUTO_TEST_CASE( test_case2 )
{
  BOOST_TEST( true );
}

//item 3, add the 3rd test case
BOOST_AUTO_TEST_CASE( test_case3 )
{
  BOOST_TEST( false );
}

BOOST_AUTO_TEST_SUITE_END() //declare we end test suite
```

为了构建这个，我们可能需要安装 boost，如下所示：

```cpp
sudo apt-get install libboost-all-dev
```

然后，我们可以构建并运行它，如下所示：

```cpp
~/wus1/Chapter-13$ g++ -g  ch13_unit_test1.cpp 
~/wus1/Chapter-13$ ./a.out
```

上述代码的结果如下：

```cpp
Running 3 test cases...
 ch13_unit_test1.cpp(13): error: in "my_suite/test_case1": check x == 'b' has failed ['a' != 'b']
 ch13_unit_test1.cpp(25): error: in "my_suite/test_case3": check false has failed

 *** 2 failures are detected in the test module "my_test"
```

在这里，我们可以看到`test_case1`和`test_case3`中存在失败。特别是在`test_case1`中，`x`的值不等于`b`，显然在`test_case3`中，一个错误的检查无法通过测试。

# TDD

如下图所示，TDD 流程从编写失败的测试代码开始，然后添加/修改代码使测试通过。之后，我们对测试计划和代码进行重构，直到满足所有要求[16,17]。让我们看看下面的图表：

![](img/48d849fe-1729-4c98-8497-c0500b7f3080.png)

*步骤 1*是编写一个失败的测试。TDD 不是先开发代码，而是开始编写测试代码。因为我们还没有代码，所以我们知道，如果我们运行测试，它会失败。在这个阶段，定义测试数据格式和接口，并想象代码实现细节。

*步骤 2*的目标是尽快使测试通过，开发工作量最小。我们不想完美地实现一切；我们只希望它通过测试。一旦测试通过，我们就有东西可以展示给客户，并告诉客户，此时客户可能在看到初始产品后完善需求。然后，我们进入下一个阶段。

第三阶段是重构。在这个阶段，我们可能会进去，看看，看看我们想要改变什么以及如何改变它。

对于传统的开发人员来说，TDD 最困难的地方是从编码->测试模式转变为测试->编码模式的心态变化。为了对测试套件有一个模糊的概念，J. Hartikainen 建议开发人员考虑以下五个步骤[18]来开始：

1.  首先确定输入和输出。

1.  选择类/函数签名。

1.  只决定功能的一个小方面进行测试。

1.  实现测试。

1.  实现代码。

一旦我们完成了这个迭代，我们可以逐渐重构它，直到实现整体的综合目标。

# TDD 的例子

接下来，我们将通过实施一个案例研究来演示 TDD 过程。在这个研究中，我们将开发一个 Mat 类来执行 2D 矩阵代数，就像我们在 Matlab 中所做的那样。这是一个类模板，可以容纳所有数据类型的 m×n 矩阵。矩阵代数包括矩阵的加法、减法、乘法和除法，它还具有元素操作能力。

让我们开始吧。

# 步骤 1 - 编写一个失败的测试

首先，我们只需要以下内容：

+   从给定的行数和列数创建一个`Mat`对象（默认应为 0×0，即空矩阵）。

+   按行打印其元素。

+   从`rows()`和`cols()`获取矩阵大小。

根据这些要求，我们可以有失败的单元测试代码来提升 UTF，如下所示：

```cpp
// ch13_tdd_boost_UTF1.cpp
#define BOOST_TEST_MODULE tdd_test
#include <boost/test/included/unit_test.hpp>
#include "ch13_tdd_v1.h"

BOOST_AUTO_TEST_SUITE(tdd_suite)  //begin a test suite: "tdd_suite"

BOOST_AUTO_TEST_CASE(test_case1) {
  Mat<int> x(2, 3);            //create a 2 x 3 int matrix
  x.print("int x=");
  BOOST_TEST(2 == x.rows());
  BOOST_TEST(3 == x.cols());

  Mat<float> y;              //create a 0 x 0 empty float matrix
  y.print("float y=");
  BOOST_TEST(0 == y.rows());
  BOOST_TEST(0 == y.cols());

  Mat<char> z(1,10);       //create a 1 x 10 char matrix
  z.print("char z=");
  BOOST_TEST(1 == z.rows());
  BOOST_TEST(10 == z.cols());
}
BOOST_AUTO_TEST_SUITE_END() //end test suite
```

现在我们的测试代码准备好了，我们准备开发代码。

# 步骤 2 - 开发代码使测试通过

实现一个最小的代码段来通过前面的测试，如下所示：

```cpp
//file: ch13_tdd_v1.h
#ifndef __ch13_TDD_V1__
#define __ch13_TDD_V1__
#include <iostream>
#include <assert.h>
template< class T>
class Mat {
public:
  Mat(const uint32_t m=0, const uint32_t n=0);
  Mat(const Mat<T> &rhs) = delete;
  ~Mat();

  Mat<T>& operator = (const Mat<T> &x) = delete;

  uint32_t rows() { return m_rows; }
  uint32_t cols() { return m_cols; }
  void print(const char* str) const;

private:
  void creatBuf();
  void deleteBuf();
  uint32_t m_rows; //# of rows
  uint32_t m_cols; //# of cols
  T* m_buf;
};
#include "ch13_tdd_v1.cpp"
#endif
```

一旦我们有了前面的头文件，我们就可以开发其相应的`cpp`文件，如下所示：

```cpp
//file: ch13_tdd_v1.cpp
#include "ch13_tdd_v1.h"
using namespace std;

template< class T>
Mat<T>::Mat(const uint32_t m, const uint32_t n)
 : m_rows(m)
 , m_cols(n)
 , m_buf(NULL)
{
 creatBuf();
}

template< class T>
Mat<T> :: ~Mat()
{ 
 deleteBuf(); 
}

template< class T>
void Mat<T>::creatBuf()
{
 uint32_t sz = m_rows * m_cols;
 if (sz > 0) {
 if (m_buf) { deleteBuf();}
 m_buf = new T[sz];
 assert(m_buf);
 }
 else {
 m_buf = NULL;
 }
}

template< class T>
void Mat<T>::deleteBuf()
{
 if (m_buf) {
 delete[] m_buf;
 m_buf = NULL;
 }
}

template< class T>
void Mat<T> ::print(const char* str) const
{
 cout << str << endl;
 cout << m_rows << " x " << m_cols << "[" << endl;
 const T *p = m_buf;
 for (uint32_t i = 0; i<m_rows; i++) {
 for (uint32_t j = 0; j < m_cols; j++) {
 cout << *p++ << ", ";
 }
 cout << "\n";
 }
 cout << "]\n";
}
```

假设我们使用支持`-std=c++11`或更高版本的 g++进行构建和执行：

```cpp
~/wus1/Chapter-13$ g++ -g ch13_tdd_boost_UTF1.cpp~/wus1/Chapter-13$ a.out 
```

这将导致以下输出：

```cpp
Running 1 test case...
 int x=2 x 3[
 1060438054, 1, 4348032,
 0, 4582960, 0,
 ]
 float y=0 x 0[
 ]
 char z=1 x 10[
 s,s,s,s,s,s,s,s,s,s,
 ]
```

在`test_case1`中，我们创建了三个矩阵并测试了`rows()`，`cols()`和`print()`函数。第一个是一个 2x3 的`int`类型矩阵。由于它没有初始化，其元素的值是不可预测的，这就是为什么我们可以从`print()`中看到这些随机数。在这一点上，我们也通过了`rows()`和`cols()`的测试（两个`BOOST_TEST()`调用没有错误）。第二个是一个空的浮点类型矩阵；它的`print()`函数什么也不输出，它的`cols()`和`rows()`都是零。最后，第三个是一个 1x10 的`char`类型未初始化矩阵。同样，这三个函数的所有输出都是预期的。

# 步骤 3 - 重构

到目前为止，一切顺利 - 我们通过了测试！然而，在向客户展示前面的结果后，他/她可能会要求我们添加另外两个接口，如下所示：

+   为所有元素创建一个给定初始值的 m x n 矩阵。

+   添加`numel()`以返回矩阵的总元素数。

+   添加`empty()`，如果矩阵既有零行又有零列，则返回 true，否则返回 false。

一旦我们向测试套件添加了第二个测试用例，整体重构后的测试代码将如下所示：

```cpp
// ch13_tdd_Boost_UTF2.cpp
#define BOOST_TEST_MODULE tdd_test
#include <boost/test/included/unit_test.hpp>
#include "ch13_tdd_v2.h"

//declare we begin a test suite and name it "tdd_suite"
BOOST_AUTO_TEST_SUITE(tdd_suite)

//add the 1st test case
BOOST_AUTO_TEST_CASE(test_case1) {
  Mat<int> x(2, 3);
  x.print("int x=");
  BOOST_TEST(2 == x.rows());
  BOOST_TEST(3 == x.cols());

  Mat<float> y;
  BOOST_TEST(0 == y.rows());
  BOOST_TEST(0 == y.cols());

  Mat<char> z(1, 10);
  BOOST_TEST(1 == z.rows());
  BOOST_TEST(10 == z.cols());
}

//add the 2nd test case
BOOST_AUTO_TEST_CASE(test_case2)
{
  Mat<int> x(2, 3, 10);
  x.print("int x=");
  BOOST_TEST( 6 == x.numel() );
  BOOST_TEST( false == x.empty() );

  Mat<float> y;
  BOOST_TEST( 0 == y.numel() );
  BOOST_TEST( x.empty() ); //bug x --> y 
}

BOOST_AUTO_TEST_SUITE_END() //declare we end test suite
```

下一步是修改代码以通过这个新的测试计划。为了简洁起见，我们不会在这里打印`ch13_tdd_v2.h`和`ch13_tdd_v2.cpp`文件。您可以从本书的[GitHub](https://github.com/PacktPublishing/Expert-CPP)存储库中下载它们。构建并执行`ch13_tdd_Boost_UTF2.cpp`后，我们得到以下输出：

```cpp
Running 2 test cases...
 int x=2x3[
 1057685542, 1, 1005696,
 0, 1240624, 0,
 ]
 int x=2x3[
 10, 10, 10,
 10, 10, 10,
 ]
 ../Chapter-13/ch13_tdd_Boost_UTF2.cpp(34): error: in "tdd_suite/test_case2": che
 ck x.empty() has failed [(bool)0 is false]
```

在第一个输出中，由于我们只定义了一个 2x3 的整数矩阵，并且没有在`test_case1`中初始化它，所以会打印出未定义的行为 - 也就是六个随机数。第二个输出来自`test_case2`，其中`x`的所有六个元素都初始化为`10`。在我们展示了前面的结果之后，我们的客户可能会要求我们添加其他新功能或修改当前存在的功能。但是，经过几次迭代，最终我们会达到*快乐点*并停止因式分解。

现在我们已经了解了 TDD，我们将讨论 BDD。

# BDD

软件开发最困难的部分是与业务参与者、开发人员和质量分析团队进行沟通。由于误解或模糊的需求、技术争论和缓慢的反馈周期，项目很容易超出预算、错过截止日期或完全失败。

(BDD) [20]是一种敏捷开发过程，具有一套旨在减少沟通障碍和其他浪费活动的实践。它还鼓励团队成员在生产生命周期中不断地使用真实世界的例子进行沟通。

BDD 包含两个主要部分：故意发现和 TDD。为了让不同组织和团队的人了解开发软件的正确行为，故意发现阶段引入了*示例映射*技术，通过具体的例子让不同角色的人进行对话。这些例子将成为系统行为的自动化测试和实时文档。在其 TDD 阶段，BDD 规定任何软件单元的测试应该以该单元的期望行为为基础。

有几种 BDD 框架工具（JBehave、RBehave、Fitnesse、Cucumber [21]等）适用于不同的平台和编程语言。一般来说，这些框架执行以下步骤：

1.  在故意发现阶段，阅读由业务分析师准备的规范格式文档。

1.  将文档转换为有意义的条款。每个单独的条款都可以被设置为质量保证的测试用例。开发人员也可以根据条款实现源代码。

1.  自动执行每个条款场景的测试。

总之，我们已经了解了关于应用开发流程中什么、何时以及如何进行测试的策略。如下图所示，传统的 V 形[2]模型强调需求->设计->编码->测试的模式。TDD 认为开发过程应该由测试驱动，而 BDD 将来自不同背景和角色的人之间的沟通加入到 TDD 框架中，并侧重于行为测试：

![](img/fcf1a324-da5d-4573-9146-c831408113a7.png)

此外，单元测试强调在编码完成后测试单个组件。TDD 更注重如何在编写代码之前编写测试，然后通过下一级测试计划添加/修改代码。BDD 鼓励客户、业务分析师、开发人员和质量保证分析师之间的合作。虽然我们可以单独使用每一个，但在这个敏捷软件开发时代，我们真的应该将它们结合起来以获得最佳结果。

# 总结

在本章中，我们简要介绍了软件开发过程中与测试和调试相关的主题。测试可以发现问题，根本原因分析有助于在宏观层面上定位问题。然而，良好的编程实践可以在早期阶段防止软件缺陷。此外，命令行界面调试工具 GDB 可以帮助我们设置断点，并在程序运行时逐行执行程序并打印变量的值。

我们还讨论了自动分析工具和人工测试过程。静态分析评估程序的性能而不执行它。另一方面，动态分析工具可以通过执行程序来发现缺陷。最后，我们了解了测试过程在软件开发流程中应该如何、何时以及如何参与的策略。单元测试强调在编码完成后测试单个组件。TDD 更注重如何在开发代码之前编写测试，然后通过下一级测试计划重复这个过程。BDD 鼓励客户、业务分析师、开发人员和质量保证分析师之间的合作。

在下一章中，我们将学习如何使用 Qt 创建跨平台应用程序的图形用户界面（GUI）程序，这些程序可以在 Linux、Windows、iOS 和 Android 系统上运行。首先，我们将深入了解跨平台 GUI 编程的基本概念。然后我们将介绍 Qt 及其小部件的概述。最后，通过一个案例研究示例，我们将学习如何使用 Qt 设计和实现网络应用程序。

# 进一步阅读

+   J. Rooney 和 L. Vanden Heuvel，《初学者的根本原因分析》，Quality Progress，2004 年 7 月，第 45-53 页。

+   T. Kataoka，K. Furuto 和 T. Matsumoto，《软件问题根本原因分析方法》，SEI Tech. Rev.，第 73 期，2011 年第 81 页。

+   K. A. Briski 等，《减少代码缺陷以提高软件质量和降低开发成本》，IBM Rational Software Analyzer 和 IBM Rational PurifyPlus 软件。

+   [`www.learncpp.com/cpp-programming/eight-c-programming-mistakes-the-compiler-wont-catch`](https://www.learncpp.com/cpp-programming/eight-c-programming-mistakes-the-compiler-wont-catch)。

+   B. Stroustrup 和 H. Sutter，《C++核心指南》：[`isocpp.github.io/CppCoreGuidelines`](https://isocpp.github.io/CppCoreGuidelines)。

+   [`www.gnu.org/software/gdb/`](https://www.gnu.org/software/gdb/)。

+   [`www.fayewilliams.com/2014/02/21/debugging-for-beginners/`](https://www.fayewilliams.com/2014/02/21/debugging-for-beginners/)。

+   [`www.perforce.com/blog/qac/what-static-code-analysis`](https://www.perforce.com/blog/qac/what-static-code-analysis)。

+   [`linux.die.net/man/1/g++`](https://linux.die.net/man/1/g++)。

+   [`www.embedded.com/static-vs-dynamic-analysis-for-secure-code-development-part-2/`](https://www.embedded.com/static-vs-dynamic-analysis-for-secure-code-development-part-2/)。

+   ISO/IEC/IEEE 29119-1:2013《软件和系统工程-软件测试》[`www.iso.org/standard/45142.html`](https://www.iso.org/standard/45142.html)。

+   [`gamesfromwithin.com/exploring-the-c-unit-testing-framework-jungle`](http://gamesfromwithin.com/exploring-the-c-unit-testing-framework-jungle)。

+   [`www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html`](https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html)。

+   K. Beck，《通过示例进行测试驱动开发》, Addison Wesley 出版，ISBN 978-0321146533。

+   H. Erdogmus,  T. Morisio, [*关于编程的测试优先方法的有效性*](https://ieeexplore.ieee.org/document/1423994), IEEE 软件工程交易会议录, 31(1). 2005 年 1 月。

+   [`codeutopia.net/blog/2015/03/01/unit-testing-tdd-and-bdd`](https://codeutopia.net/blog/2015/03/01/unit-testing-tdd-and-bdd).

+   [`cucumber.io/blog/intro-to-bdd-and-tdd/`](https://cucumber.io/blog/intro-to-bdd-and-tdd/).

+   D. North,  Introducing BDD,  [`dannorth.net/introducing-bdd/`](https://dannorth.net/introducing-bdd/)  (2006 年 3 月)。

+   D. North, E. Keogh, et. al, "[jbehave.org/team-list](https://jbehave.org/)",  May 2019.

除此之外，你还可以查看以下来源（这些在本章中没有直接提到）：

+   B. Stroustrup 和 H. Sutter, *C++ 核心指南*: [`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)[.](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

+   G. Rozental 和 R. Enficiaud, *Boost.Test*: [`www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html`](https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html)

+   D. North,* Introducing BDD*: [`dannorth.net/introducing-bdd/`](https://dannorth.net/introducing-bdd/)

# 练习和问题

1.  使用 `gdb` 函数断点、条件断点和 `watchpoint`、`continue`、`finish` 命令，调试 `ch13_gdb_2.cpp`。

1.  使用 `g++ -c -Wall -Weffc++ -Wextra  x.cpp -o x.out` 来构建 `cpp` 文件 `ch13_rca*.cpp`。你从他们的警告输出中看到了什么？

1.  为什么静态分析会产生误报，而动态分析不会呢？

1.  下载 `ch13_tdd_v2.h/.cpp` 并执行下一阶段的重构。在这个阶段，我们将添加一个拷贝构造函数、赋值运算符，以及诸如 `+`、`-`、`*`、`/` 等的逐元素操作运算符。更具体地，我们需要做以下事情：

1.  将第三个测试用例添加到我们的测试套件中，即 `ch13_tdd_Boost_UTF2.cpp`。

1.  将这些函数的实现添加到文件中；例如，`ch13_tdd_v2.h/.cpp`。

1.  运行测试套件来测试它们。
