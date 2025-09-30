# 第三章

# 对象文件

本章详细介绍了 C/C++ 项目可能产生的各种产品。可能的产品包括可重定位对象文件、可执行对象文件、静态库和共享对象文件。然而，可重定位对象文件被认为是临时产品，它们作为制作其他最终产品的原料。

看起来，在今天的 C 语言中，进一步讨论各种类型的对象文件及其内部结构至关重要。大多数 C 语言书籍只讨论 C 语法和语言本身；但在现实生活中，你需要更深入的知识才能成为一名成功的 C 语言程序员。

当你创建软件时，不仅仅是关于开发和编程语言。实际上，它是关于整个流程：编写代码、编译、优化、生产正确的产品，以及进一步的后续步骤，以便在目标平台上运行和维护这些产品。

你应该对这些中间步骤有所了解，以便能够解决你可能会遇到的问题。这对于嵌入式开发来说尤其严重，因为硬件架构和指令集可能具有挑战性和非典型性。

本章分为以下几部分：

1.  **应用程序二进制接口**：在这里，我们首先将讨论 **应用程序二进制接口**（**ABI**）及其重要性。

1.  **对象文件格式**：在本节中，我们讨论今天存在或在过去几年中变得过时的各种对象文件格式。我们还介绍了 ELF 作为 Unix-like 系统中最常用的对象文件格式。

1.  **可重定位对象文件**：在这里，我们讨论可重定位对象文件以及 C 项目的第一个产品。我们深入 ELF 可重定位对象文件内部，看看我们能找到什么。

1.  **可执行对象文件**：作为本节的一部分，我们讨论可执行对象文件。我们还解释了它们是如何从多个可重定位对象文件中创建的。我们讨论了 ELF 可重定位对象文件和可执行对象文件在内部结构上的差异。

1.  **静态库**：在本节中，我们讨论静态库以及如何创建它们。我们还演示了如何编写程序并使用已经构建的静态库。

1.  **动态库**：在这里，我们讨论共享对象文件。我们演示了如何从多个可重定位对象文件中创建它们，以及如何在程序中使用它们。我们还简要地讨论了 ELF 共享对象文件的内部结构。

本章的讨论将主要围绕 Unix-like 系统，但我们也会讨论与其他操作系统（如 Microsoft Windows）的一些差异。

**注意**：

在继续阅读本章之前，你需要熟悉构建 C 项目所需的基本思想和步骤。你需要知道什么是翻译单元以及链接与编译的不同之处。请在继续阅读本章之前先阅读上一章。

让我们从介绍 ABI（应用程序二进制接口）开始本章。

# 应用程序二进制接口 (ABI)

如你所知，每个库或框架，无论使用的技术或编程语言如何，都暴露了一组特定的功能，这被称为其**应用程序编程接口**（**API**）。如果一个库应该被其他代码使用，那么消费者代码应该使用提供的 API。为了清楚起见，除了 API 之外，不应使用任何其他东西来使用库，因为它是库的公共接口，其他所有内容都被视为黑盒，因此不能使用。

现在假设经过一段时间后，库的 API 进行了某些修改。为了使消费者代码能够继续使用库的新版本，代码必须适应新的 API；否则，它将无法再使用它。消费者代码可以坚持使用库的某个版本（可能是一个旧版本）并忽略新版本，但让我们假设有一个升级到库最新版本的愿望。

简而言之，API 就像两个软件组件之间接受（或标准）的约定。ABI 与 API 非常相似，但处于不同的层面。虽然 API 保证了两个软件组件在功能合作方面的兼容性，但 ABI 保证了两个程序在机器级指令及其相应的目标文件级别上的兼容性。

例如，一个程序不能使用具有不同 ABI 的动态或静态库。或许更糟糕的是，一个可执行文件（实际上是一个目标文件）不能在支持与可执行文件构建的 ABI 不同的系统上运行。许多关键且明显的系统功能，如*动态链接*、*加载可执行文件*和*函数调用约定*，必须精确地按照约定的 ABI 执行。

一个 ABI 通常涵盖以下内容：

+   目标架构的指令集，包括处理器指令、内存布局、字节序、寄存器等。

+   现有的数据类型、它们的大小和对齐策略。

+   函数调用约定描述了函数应该如何被调用。例如，*栈帧*的结构和参数的推入顺序都属于其中。

+   定义在类 Unix 系统中如何调用*系统调用*。

+   使用*目标文件格式*，我们将在下一节中解释，以拥有*可重定位、可执行*和*共享目标文件*。

+   关于由 C++ 编译器生成的目标文件，*名称编码*、*虚函数表布局*是 ABI 的一部分。

*System V ABI* 是在 Linux 和 BSD 等类 Unix 操作系统中最广泛使用的 ABI 标准。**可执行和链接格式**（**ELF**）是 System V ABI 中使用的标准目标文件格式。

**注意**：

以下链接是 AMD 64 位架构的 System V ABI：[`www.uclibc.org/docs/psABI-x86_64.pdf`](https://www.uclibc.org/docs/psABI-x86_64.pdf)。您可以查看目录列表，并了解它涵盖的领域。

在下一节中，我们将讨论目标文件格式，特别是 ELF。

# 目标文件格式

正如我们在上一章，即 *第二章，编译和链接* 中所解释的，在平台上，目标文件有其自己的特定格式来存储机器级指令。请注意，这是关于目标文件的结构，这与每个架构都有自己的指令集的事实不同。正如我们从之前的讨论中了解到的，这两个变体是平台中 ABI 的不同部分；目标文件格式和架构的指令集。

在本节中，我们将简要介绍一些广为人知的目标文件格式。首先，让我们看看各种操作系统中使用的某些目标文件格式：

+   **ELF** 被 Linux 和许多其他类 Unix 操作系统使用

+   在 OS X（macOS 和 iOS）系统中使用的 **Mach-O**

+   在 Microsoft Windows 中使用的 **PE**（可移植执行）格式

为了提供关于当前和过去目标文件格式的历史和背景信息，我们可以这样说，今天存在的所有目标文件格式都是旧 `a.out` 目标文件格式的继承者。它是为 Unix 的早期版本设计的。

术语 **a.out** 代表 **汇编器输出**。尽管今天该文件格式已经过时，但该名称仍然被用作大多数链接器生成的可执行文件的默认文件名。您应该记得在本书的第一章中看到过 `a.out`。

然而，`a.out` 格式很快就被 **COFF** 或 **通用目标文件格式** 所取代。COFF 是 ELF 的基础——我们在大多数类 Unix 系统中使用的目标文件格式。苹果公司也用 Mach-O 替换了 `a.out` 作为 OS/X 的一部分。Windows 使用 **PE** 或 **可移植执行** 文件格式来处理其目标文件，该格式基于 COFF。

**注意**：

更深入的目标文件格式历史可以在这里找到：[`en.wikipedia.org/wiki/COFF#History`](https://en.wikipedia.org/wiki/COFF#History)。了解特定主题的历史将有助于您更好地理解其演变路径以及当前和过去的特点。

如您所见，今天所有的主流目标文件格式都基于历史目标文件格式 `a.out`，然后是 COFF，并且在很多方面都拥有相同的血统。

ELF 是 Linux 和大多数类 Unix 操作系统中使用的标准对象文件格式。实际上，ELF 是作为 System V ABI 的一部分使用的对象文件格式，在大多数 Unix 系统中得到广泛使用。如今，它是操作系统使用最广泛的对象文件格式。

ELF 是包括但不限于以下操作系统的标准二进制文件格式：

+   Linux

+   FreeBSD

+   NetBSD

+   Solaris

这意味着，只要它们下面的架构保持不变，为这些操作系统之一创建的 ELF 对象文件可以在其他操作系统中运行和使用。ELF，像所有其他 *文件格式* 一样，有一个结构，我们将在接下来的章节中简要描述。

**注意**：

更多关于 ELF 及其细节的信息可以在以下链接找到：[`www.uclibc.org/docs/psABI`](https://www.uclibc.org/docs/psABI-x86_64.pdf)。请注意，此链接指的是 AMD 64 位（`amd64`）架构的 System V ABI。

您还可以在此处阅读 System V ABI 的 HTML 版本：[`www.sco.com/developers/gabi/2003-12-17/ch4.intro.html`](http://www.sco.com/developers/gabi/2003-12-17/ch4.intro.html)。

在接下来的章节中，我们将讨论 C 项目的临时和最终产品。我们从可重定位对象文件开始。

# 可重定位对象文件

在本节中，我们将讨论可重定位对象文件。正如我们在上一章中解释的，这些对象文件是 C 编译管道中汇编步骤的输出。这些文件被认为是 C 项目的临时产品，并且是生产进一步和最终产品的主要成分。因此，深入了解它们并查看我们可以在可重定位对象文件中找到什么将是有用的。

在可重定位对象文件中，我们可以找到以下关于编译的翻译单元的项目：

+   为翻译单元中找到的函数生成的机器级指令（代码）。

+   在翻译单元（数据）中声明的初始化全局变量的值。

+   包含翻译单元中找到的所有定义和引用符号的 *符号表*。

以下是在任何可重定位对象文件中可以找到的关键项目。当然，它们的组合方式取决于对象文件格式，但使用适当的工具，你应该能够从可重定位对象文件中提取这些项目。我们将很快对 ELF 可重定位对象文件进行这样的操作。

但在深入示例之前，让我们谈谈为什么可重定位对象文件会这样命名。换句话说，*可重定位*究竟意味着什么？原因来自于链接器执行的过程，以便将一些可重定位对象文件组合在一起，形成一个更大的对象文件——可执行对象文件或共享对象文件。

我们将在下一节讨论可执行文件中可以找到的内容，但到目前为止，我们应该知道我们在可执行对象文件中找到的项目是所有构成可重定位对象文件中找到的项目之和。让我们只谈谈机器级指令。

在一个可重定位对象文件中找到的机器级指令应该放在来自另一个可重定位对象文件的机器级指令旁边。这意味着指令应该是容易 *移动* 或 *重定位* 的。为了实现这一点，指令在可重定位对象文件中没有地址，并且它们只在链接步骤后获得地址。这就是我们称这些对象文件为可重定位的主要原因。为了更详细地说明这一点，我们需要在真实示例中展示。

*示例 3.1* 是关于两个源文件，一个包含两个函数的定义，`max` 和 `max_3`，另一个源文件包含使用声明的函数 `max` 和 `max_3` 的 `main` 函数。接下来，您可以看到第一个源文件的内容：

```cpp
int max(int a, int b) {
  return a > b ? a : b;
}
int max_3(int a, int b, int c) {
  int temp = max(a, b);
  return c > temp ? c : temp;
}
```

代码框 3-1 [ExtremeC_examples_chapter3_1_funcs.c]：包含两个函数定义的源文件

第二个源文件看起来像以下代码框：

```cpp
int max(int, int);
int max_3(int, int, int);
int a = 5;
int b = 10;
int main(int argc, char** argv) {
  int m1 = max(a, b);
  int m2 = max_3(5, 8, -1);
  return 0;
}
```

代码框 3-2 [ExtremeC_examples_chapter3_1.c]：使用已声明的函数的 `main` 函数。定义被放入单独的源文件中。

让我们为前面的源文件生成可重定位对象文件。这样，我们可以调查内容和之前解释的内容。请注意，由于我们在 Linux 机器上编译这些源文件，我们期望看到 ELF 对象文件作为结果：

```cpp
$ gcc -c ExtremeC_examples_chapter3_1_funcs.c  -o funcs.o
$ gcc -c ExtremeC_examples_chapter3_1.c -o main.o
$
```

Shell 框 3-1：将源文件编译成相应的可重定位对象文件

`funcs.o` 和 `main.o` 都是可重定位的 ELF 对象文件。在 ELF 对象文件中，描述为可重定位对象文件中的项目被放入多个部分中。为了查看先前可重定位对象文件中的当前部分，我们可以使用以下 `readelf` 工具：

```cpp
$ readelf -hSl funcs.o
[7/7]
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              REL (Relocatable file)
  Machine:                           Advanced Micro Devices X86-64
...
  Number of section headers:         12
  Section header string table index: 11
Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .text             PROGBITS         0000000000000000  00000040
       0000000000000045  0000000000000000  AX       0     0     1
...
  [ 3] .data             PROGBITS         0000000000000000  00000085
       0000000000000000  0000000000000000  WA       0     0     1
  [ 4] .bss              NOBITS           0000000000000000  00000085
       0000000000000000  0000000000000000  WA       0     0     1
...
  [ 9] .symtab           SYMTAB           0000000000000000  00000110
       00000000000000f0  0000000000000018          10     8     8
  [10] .strtab           STRTAB           0000000000000000  00000200
       0000000000000030  0000000000000000           0     0     1
  [11] .shstrtab         STRTAB           0000000000000000  00000278
       0000000000000059  0000000000000000           0     0     1
...
$
```

Shell 框 3-2：`funcs.o` 对象文件的 ELF 内容

如您在前面的 shell 框中看到的，可重定位对象文件有 11 个部分。粗体字体的部分是我们介绍为存在于对象文件中的项目。`.text` 部分包含翻译单元的所有机器级指令。`.data` 和 `.bss` 部分包含初始化全局变量的值，以及未初始化全局变量所需的字节数。`.symtab` 部分包含符号表。

注意，前两个对象文件中存在的部分是相同的，但它们的内容是不同的。因此，我们不显示其他可重定位对象文件的部分。

正如我们之前提到的，ELF 对象文件中的一个部分包含符号表。在上一章中，我们对符号表及其条目进行了详细讨论。我们描述了链接器如何使用它来生成可执行和共享对象文件。在这里，我们想提醒您注意我们在上一章中没有讨论的符号表的一些内容。这将是根据我们关于为什么可重定位对象文件以这种方式命名的解释。

让我们导出 `funcs.o` 的符号表。在上一章中，我们使用了 `objdump`，但现在，我们将使用 `readelf` 来完成：

```cpp
$ readelf -s funcs.o
Symbol table '.symtab' contains 10 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
...
     6: 0000000000000000     0 SECTION LOCAL  DEFAULT    7
     7: 0000000000000000     0 SECTION LOCAL  DEFAULT    5
     8: 0000000000000000    22 FUNC    GLOBAL DEFAULT    1 max
     9: 0000000000000016    47 FUNC    GLOBAL DEFAULT    1 max_3
$
```

Shell Box 3-3：funcs.o 对象文件的符号表

如您在 `Value` 列中看到的，分配给 `max` 的地址是 `0`，分配给 `max_3` 的地址是 `22`（十六进制 `16`）。这意味着与这些符号相关的指令是相邻的，并且它们的地址从 0 开始。这些符号及其对应的机器级指令已准备好被重新定位到最终可执行文件中的其他位置。让我们看看 `main.o` 的符号表：

```cpp
$ readelf -s main.o
Symbol table '.symtab' contains 14 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
...
     8: 0000000000000000     4 OBJECT  GLOBAL DEFAULT    3 a
     9: 0000000000000004     4 OBJECT  GLOBAL DEFAULT    3 b
    10: 0000000000000000    69 FUNC    GLOBAL DEFAULT    1 main
    11: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND _GLOBAL_OFFSET_TABLE_
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND max
    13: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND max_3
$
```

Shell Box 3-4：main.o 对象文件的符号表

如您所见，与全局变量 `a` 和 `b` 相关的符号，以及 `main` 函数的符号被放置在似乎不是它们应该放置的最终地址上。这是可重定位对象文件的标志。正如我们之前所说的，可重定位对象文件中的符号没有最终和绝对地址，它们的地址将在链接步骤中确定。

在下一节中，我们继续从前面的可重定位对象文件中生成可执行文件。您将看到符号表是不同的。

# 可执行对象文件

现在，是时候讨论可执行对象文件了。您现在应该知道，可执行对象文件是 C 项目的最终产品之一。与可重定位对象文件一样，它们在：; 机器级指令、初始化全局变量的值、符号表中有相同的项；然而，它们的排列可能不同。我们可以通过 ELF 可执行对象文件来展示这一点，因为它们很容易生成并研究其内部结构。

为了生成可执行的 ELF 对象文件，我们继续进行 *示例 3.1*。在上一节中，我们为示例中的两个源生成了可重定位对象文件，在本节中，我们将它们链接起来形成一个可执行文件。

以下命令为您完成这些操作，如前一章所述：

```cpp
$ gcc funcs.o main.o -o ex3_1.out
$ 
```

Shell Box 3-5：在示例 3.1 中链接之前构建的可重定位对象文件

在上一节中，我们讨论了存在于 ELF 对象文件中的段。我们应该指出，ELF 可执行对象文件中存在更多的段，但与一些段一起。每个 ELF 可执行对象文件，正如你将在本章后面看到的那样，每个 ELF 共享对象文件，除了段之外，还有许多*段*。每个段由多个段（零个或更多）组成，并且根据其内容将段放入段中。

例如，所有包含机器级指令的段都放入同一个段中。你将在*第四章*，*进程内存结构*中看到，这些段很好地映射到运行进程内存布局中找到的静态*内存段*。

让我们看看可执行文件的内容，并了解这些段。同样，对于可重定位对象文件，我们可以使用相同的命令来显示段，以及在一个可执行 ELF 对象文件中找到的段。

```cpp
$ readelf -hSl ex3_1.out
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file) 
  Machine:                           Advanced Micro Devices X86-64 
  Version:                           0x1
  Entry point address:               0x4f0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          6576 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         28
  Section header string table index: 27
Section Headers: 
  [Nr] Name              Type             Address           Offset 
       Size              EntSize          Flags  Link  Info  Align 
  [ 0]                   NULL             0000000000000000  00000000 
      0000000000000000  0000000000000000           0     0     0 
  [ 1] .interp           PROGBITS         0000000000000238  00000238 
       000000000000001c  0000000000000000   A       0     0     1 
  [ 2] .note.ABI-tag     NOTE             0000000000000254  00000254 
       0000000000000020  0000000000000000   A       0     0     4 
  [ 3] .note.gnu.build-i NOTE             0000000000000274  00000274 
       0000000000000024  0000000000000000   A       0     0     4 
... 
  [26] .strtab           STRTAB           0000000000000000  00001678 
       0000000000000239  0000000000000000           0     0     1 
  [27] .shstrtab         STRTAB           0000000000000000  000018b1 
       00000000000000f9  0000000000000000           0     0     1 
Key to Flags: 
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info), 
  L (link order), O (extra OS processing required), G (group), T (TLS), 
  C (compressed), x (unknown), o (OS specific), E (exclude), 
  l (large), p (processor specific) 
Program Headers: 
  Type           Offset             VirtAddr           PhysAddr 
                 FileSiz            MemSiz              Flags  Align 
  PHDR           0x0000000000000040 0x0000000000000040 0x0000000000000040 
                 0x00000000000001f8 0x00000000000001f8  R      0x8 
  INTERP         0x0000000000000238 0x0000000000000238 0x0000000000000238 
                 0x000000000000001c 0x000000000000001c  R      0x1 
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2] 
... 
  GNU_EH_FRAME   0x0000000000000714 0x0000000000000714 0x0000000000000714 
                 0x000000000000004c 0x000000000000004c  R      0x4 
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000 
                 0x0000000000000000 0x0000000000000000  RW     0x10 
  GNU_RELRO      0x0000000000000df0 0x0000000000200df0 0x0000000000200df0 
                 0x0000000000000210 0x0000000000000210  R      0x1 
Section to Segment mapping: 
  Segment Sections... 
   00 
   01     .interp 
   02     .interp .note.ABI-tag .note.gnu.build-id .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .init .plt .plt.got .text .fini .rodata .eh_frame_hdr .eh_frame 
   03     .init_array .fini_array .dynamic .got .data .bss 
   04     .dynamic 
   05     .note.ABI-tag .note.gnu.build-id 
   06     .eh_frame_hdr 
   07 
   08     .init_array .fini_array .dynamic .got 
$
```

Shell Box 3-6：ex3_1.out 可执行对象文件的 ELF 内容

关于上述输出有以下几点说明：

+   从 ELF 的角度来看，我们可以看到对象文件的类型是共享对象文件。换句话说，在 ELF 中，可执行对象文件是一种具有特定段（如`INTERP`）的共享对象文件。这个段（实际上是此段引用的`.interp`段）由加载程序用于加载和执行可执行对象文件。

+   我们将四个段加粗。第一个指的是上一条项目符号中解释的`INTERP`段。第二个是`TEXT`段，它包含所有包含机器级指令的段。第三个是`DATA`段，它包含所有用于初始化全局变量和其他早期结构的值。第四个段指的是可以找到*动态链接*相关信息的段。例如，需要作为执行部分加载的共享对象文件。

+   如你所见，与可重定位共享对象相比，我们得到了更多的段，可能填充了加载和执行对象文件所需的数据。

如我们在上一节所述，可重定位目标文件的符号表中找到的符号没有任何绝对和确定的地址。这是因为包含机器级指令的部分尚未链接。

在更深层次上，链接多个可重定位对象文件实际上是将给定可重定位对象文件中的所有类似段收集起来，并将它们组合成一个更大的段，最后将生成的段放入输出可执行文件或共享对象文件中。因此，只有在这个步骤之后，符号才能最终确定并获得不会改变的地址。在可执行对象文件中，地址是绝对的，而在共享对象文件中，相对地址是绝对的。我们将在专门讨论动态库的章节中进一步讨论这个问题。

让我们看看在可执行文件`ex3_1.out`中找到的符号表。请注意，符号表有很多条目，这就是为什么以下 Shell Box 中的输出没有完全显示：

```cpp
$ readelf -s ex3_1.out
Symbol table '.dynsym' contains 6 entries: 
   Num:    Value          Size Type    Bind   Vis      Ndx Name 
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
... 
     5: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@GLIBC_2.2.5 (2) 
Symbol table '.symtab' contains 66 entries: 
   Num:    Value          Size Type    Bind   Vis      Ndx Name 
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
... 
    45: 0000000000201000     0 NOTYPE  WEAK   DEFAULT   22 data_start 
    46: 0000000000000610    47 FUNC    GLOBAL DEFAULT   13 max_3 
    47: 0000000000201014     4 OBJECT  GLOBAL DEFAULT   22 b 
    48: 0000000000201018     0 NOTYPE  GLOBAL DEFAULT   22 _edata 
    49: 0000000000000704     0 FUNC    GLOBAL DEFAULT   14 _fini 
    50: 00000000000005fa    22 FUNC    GLOBAL DEFAULT   13 max 
    51: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@@GLIBC_ 
... 
    64: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@@GLIBC_2.2 
    65: 00000000000004b8     0 FUNC    GLOBAL DEFAULT   10 _init 
$
```

Shell Box 3-7：在`ex3_1.out`可执行目标文件中找到的符号表

如前述 Shell Box 所示，一个可执行目标文件中有两个不同的符号表。第一个，`.dynsym`，包含在加载可执行文件时应解析的符号，但第二个符号表`.symtab`包含所有解析出的符号，以及从动态符号表中带来的未解析符号。换句话说，符号表包含来自动态表的未解析符号。

正如你所见，符号表中解析出的符号具有在链接步骤后获得的绝对对应地址。`max`和`max_3`符号的地址以粗体显示。

在本节中，我们简要地了解了可执行目标文件。在下一节中，我们将讨论静态库。

# 静态库

如我们之前所解释的，静态库是 C 项目可能的产物之一。在本节中，我们将讨论静态库以及它们的创建和使用方式。然后，我们将在下一节中通过介绍动态库继续这一讨论。

静态库简单地说就是由可重定位目标文件组成的 Unix 归档。这样的库通常与其他目标文件链接在一起，形成一个可执行目标文件。

注意，静态库本身不被视为对象文件，而是一个容器。换句话说，静态库在 Linux 系统中不是 ELF 文件，在 macOS 系统中也不是 Mach-O 文件。它们只是由 Unix 的`ar`实用程序创建的归档文件。

当链接器在链接步骤中准备使用静态库时，它首先尝试从中提取可重定位的目标文件，然后开始查找并解析可能存在于其中的一些未定义符号。

现在，是时候为具有多个源文件的项目创建一个静态库了。第一步是创建一些可重定位的目标文件。一旦你编译了一个 C/C++项目中的所有源文件，你就可以使用 Unix 归档工具`ar`来创建静态库的归档文件。

在 Unix 系统中，静态库通常根据一个公认且广泛使用的约定命名。名称以`lib`开头，并以`.a`扩展名结尾。在其他操作系统中可能会有所不同；例如，在 Microsoft Windows 中，静态库带有`.lib`扩展名。

假设在一个虚构的 C 项目中，你拥有源文件 `aa.c`、`bb.c`，一直到 `zz.c`。为了生成可重定位的目标文件，你需要以类似以下命令的方式编译源文件。注意，编译过程已经在上一章中进行了详细解释：

```cpp
$ gcc -c aa.c -o aa.o
$ gcc -c bb.c -o bb.o
.
.
.
$ gcc -c zz.c -o zz.o
$
```

Shell Box 3-8：将多个源文件编译成相应的可重定位目标文件

通过运行前面的命令，我们将得到所有必需的可重定位目标文件。注意，如果项目很大且包含成千上万的源文件，这可能需要相当长的时间。当然，拥有一个强大的构建机器，以及并行运行编译任务，可以显著减少构建时间。

当涉及到创建静态库文件时，我们只需运行以下命令：

```cpp
$ ar crs libexample.a aa.o bb.o ... zz.o
$
```

Shell Box 3-9：从多个可重定位目标文件中制作静态库的一般方法

因此，生成了 `libexample.a` 库，其中包含了前面所有的可重定位目标文件作为一个单独的归档。解释传递给 `ar` 命令的 `crs` 选项超出了本章的范围，但在以下链接中，你可以阅读关于其含义的说明：[关于其含义：https://stackoverflow.com/questions/29714300/what-does-th](https://stackoverflow.com/questions/29714300/what-does-the-rcs-option-in-ar-do)e-rcs-option-in-ar-do.

**注意**：

`ar` 命令不一定创建一个 *压缩* 的归档文件。它仅用于将文件组合在一起形成一个包含所有这些文件的单一文件。工具 `ar` 是通用的，你可以用它将任何类型的文件组合在一起，并从中创建自己的归档。

现在我们已经知道了如何创建一个静态库，我们将创建一个真实的静态库作为 *示例 3.2* 的一部分。

首先，我们将假设 *示例 3.2* 是一个关于几何学的 C 项目。该示例由三个源文件和一个头文件组成。库的目的是定义一组可用于其他应用的几何相关函数。

要做到这一点，我们需要从三个源文件中创建一个名为 `libgeometry.a` 的静态库文件。通过拥有静态库，我们可以将头文件和静态库文件一起使用，以便编写另一个使用库中定义的几何函数的程序。

以下代码框是源文件和头文件的内容。第一个文件 `ExtremeC_examples_chapter3_2_geometry.h` 包含了需要从我们的几何库中导出的所有声明。这些声明将被未来使用该库的应用程序使用。

**注意**：

提供的所有用于创建目标文件的命令都在 Linux 上运行并测试过。如果你要在不同的操作系统上执行它们，可能需要进行一些修改。

我们需要注意，未来的应用程序*必须*只依赖于声明，而完全不依赖于定义。因此，首先，让我们看看几何库的声明：

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_3_2_H
#define EXTREME_C_EXAMPLES_CHAPTER_3_2_H
#define PI 3.14159265359
typedef struct {
  double x;
  double y;
} cartesian_pos_2d_t;
typedef struct {
  double length;
  // in degrees
  double theta;
} polar_pos_2d_t;
typedef struct {
  double x;
  double y;
  double z;
} cartesian_pos_3d_t;
typedef struct {
  double length;
  // in degrees
  double theta;
  // in degrees
  double phi;
} polar_pos_3d_t;
double to_radian(double deg);
double to_degree(double rad);
double cos_deg(double deg);
double acos_deg(double deg);
double sin_deg(double deg);
double asin_deg(double deg);
cartesian_pos_2d_t convert_to_2d_cartesian_pos(
        const polar_pos_2d_t* polar_pos);
polar_pos_2d_t convert_to_2d_polar_pos(
        const cartesian_pos_2d_t* cartesian_pos);
cartesian_pos_3d_t convert_to_3d_cartesian_pos(
        const polar_pos_3d_t* polar_pos);
polar_pos_3d_t convert_to_3d_polar_pos(
        const cartesian_pos_3d_t* cartesian_pos);
#endif
```

代码框 3-3 [ExtremeC_examples_chapter3_2_geometry.h]: 示例 3.2 的头文件

第二个文件是一个源文件，包含三角函数的定义，这是在前面头文件中声明的六个函数中的前六个：

```cpp
#include <math.h>
// We need to include the header file since
// we want to use the macro PI
#include "ExtremeC_examples_chapter3_2_geometry.h"
double to_radian(double deg) {
  return (PI * deg) / 180;
}
double to_degree(double rad) {
  return (180 * rad) / PI;
}
double cos_deg(double deg) {
  return cos(to_radian(deg));
}
double acos_deg(double deg) {
  return acos(to_radian(deg));
}
double sin_deg(double deg) {
  return sin(to_radian(deg));
}
double asin_deg(double deg) {
  return asin(to_radian(deg));
}
```

代码框 3-4 [ExtremeC_examples_chapter3_2_trigon.c]: 包含三角函数定义的源文件

注意，源文件不需要包含头文件，除非它们将要使用头文件中声明的像 `PI` 或 `to_degree` 这样的声明。

第三个文件，同样是一个源文件，包含所有 2D 几何函数的定义：

```cpp
#include <math.h>
// We need to include the header file since we want
// to use the types polar_pos_2d_t, cartesian_pos_2d_t,
// etc and the trigonometry functions implemented in
// another source file.
#include "ExtremeC_examples_chapter3_2_geometry.h"
cartesian_pos_2d_t convert_to_2d_cartesian_pos(
        const polar_pos_2d_t* polar_pos) {
  cartesian_pos_2d_t result;
  result.x = polar_pos->length * cos_deg(polar_pos->theta);
  result.y = polar_pos->length * sin_deg(polar_pos->theta);
  return result;
}
polar_pos_2d_t convert_to_2d_polar_pos(
        const cartesian_pos_2d_t* cartesian_pos) {
  polar_pos_2d_t result;
  result.length = sqrt(cartesian_pos->x * cartesian_pos->x +
    cartesian_pos->y * cartesian_pos->y);
  result.theta =
      to_degree(atan(cartesian_pos->y / cartesian_pos->x));
  return result;
}
```

代码框 3-5 [ExtremeC_examples_chapter3_2_2d.c]: 包含 2D 函数定义的源文件

最后，包含 3D 几何函数定义的第四个文件：

```cpp
#include <math.h>
// We need to include the header file since we want to
// use the types polar_pos_3d_t, cartesian_pos_3d_t,
// etc and the trigonometry functions implemented in
// another source file.
#include "ExtremeC_examples_chapter3_2_geometry.h"
cartesian_pos_3d_t convert_to_3d_cartesian_pos(
        const polar_pos_3d_t* polar_pos) {
  cartesian_pos_3d_t result;
  result.x = polar_pos->length *
      sin_deg(polar_pos->theta) * cos_deg(polar_pos->phi);
  result.y = polar_pos->length *
      sin_deg(polar_pos->theta) * sin_deg(polar_pos->phi);
  result.z = polar_pos->length * cos_deg(polar_pos->theta);
  return result;
}
polar_pos_3d_t convert_to_3d_polar_pos(
        const cartesian_pos_3d_t* cartesian_pos) {
  polar_pos_3d_t result;
  result.length = sqrt(cartesian_pos->x * cartesian_pos->x +
    cartesian_pos->y * cartesian_pos->y +
    cartesian_pos->z * cartesian_pos->z);
  result.theta =
      to_degree(acos(cartesian_pos->z / result.length));
  result.phi =
      to_degree(atan(cartesian_pos->y / cartesian_pos->x));
  return result;
}
```

代码框 3-6 [ExtremeC_examples_chapter3_2_3d.c]: 包含 3D 函数定义的源文件

现在我们将创建静态库文件。为此，首先我们需要将前面的源文件编译成它们对应的目标文件。需要注意的是，源文件不需要包含头文件，除非它们将要使用像 `PI` 或 `to_degree` 这样的声明，这些声明在头文件中声明。

在本节中，我们选择将它们存档以创建一个静态库文件。以下命令将在 Linux 系统上执行编译：

```cpp
$ gcc -c ExtremeC_examples_chapter3_2_trigon.c -o trigon.o
$ gcc -c ExtremeC_examples_chapter3_2_2d.c -o 2d.o
$ gcc -c ExtremeC_examples_chapter3_2_3d.c -o 3d.o
$
```

命令行框 3-10：将源文件编译成对应的目标文件

当涉及到将这些目标文件存档到静态库文件中时，我们需要运行以下命令：

```cpp
$ ar crs libgeometry.a trigon.o 2d.o 3d.o
$ mkdir -p /opt/geometry
$ mv libgeometry.a /opt/geometry
$
```

命令行框 3-11：从可重定位的目标文件创建静态库文件

如我们所见，已经创建了文件 `libgeometry.a`。如您所见，我们已经将库文件移动到 `/opt/geometry` 目录，以便其他任何程序都能轻松找到。再次使用 `ar` 命令，并通过传递 `t` 选项，我们可以查看存档文件的内容：

```cpp
$ ar t /opt/geometry/libgeometry.a
trigon.o
2d.o
3d.o
$
```

命令行框 3-12：列出静态库文件的内容

如前所述的命令行框所示，静态库文件包含三个可重定位的目标文件，正如我们预期的。下一步是使用静态库文件。

现在我们已经为我们的几何示例 *example 3.2* 创建了一个静态库，我们将将其用于一个新的应用程序。在使用 C 库时，我们需要访问库公开的声明以及与其静态库文件一起的声明。这些声明被认为是库的 *公共接口*，或者更常见的是，库的 API。

在编译阶段，我们需要声明，当编译器需要了解类型、函数签名等信息的存在时。头文件就起到这个作用。在后续阶段，如链接和加载，还需要其他详细信息，例如类型大小和函数地址。

正如我们之前所说的，我们通常将 C API（由 C 库公开的 API）作为一个头文件组找到。因此，*example 3.2* 的头文件和创建的静态库文件 `libgeometry.a` 就足够我们编写一个新的程序，该程序使用我们的几何库。

当涉及到使用静态库时，我们需要编写一个新的源文件，该文件包含库的 API 并使用其函数。我们将新代码作为一个新的示例，*example 3.3*。以下代码是 *example 3.3* 的源代码：

```cpp
#include <stdio.h>
#include "ExtremeC_examples_chapter3_2_geometry.h"
int main(int argc, char** argv) {
  cartesian_pos_2d_t cartesian_pos;
  cartesian_pos.x = 100;
  cartesian_pos.y = 200;
  polar_pos_2d_t polar_pos =
      convert_to_2d_polar_pos(&cartesian_pos);
  printf("Polar Position: Length: %f, Theta: %f (deg)\n",
    polar_pos.length, polar_pos.theta);
  return 0;
}
```

Code Box 3-7 [ExtremeC_examples_chapter3_3.c]：测试一些几何函数的主函数

如你所见，*example 3.3* 包含了 *example 3.2* 的头文件。它这样做是因为它需要使用到的函数的声明。

现在，我们需要编译前面的源文件，在 Linux 系统中创建其对应的可重定位目标文件：

```cpp
$ gcc -c ExtremeC_examples_chapter3_3.c -o main.o
$
```

Shell Box 3-13：编译示例 3.3

在完成这些之后，我们需要将其与为 *example 3.2* 创建的静态库进行链接。在这种情况下，我们假设文件 `libgeometry.a` 位于 `/opt/geometry` 目录中，就像我们在 *Shell Box 3-11* 中做的那样。以下命令将通过执行链接步骤并创建可执行目标文件 *ex3_3.out* 来完成构建：

```cpp
$ gcc main.o -L/opt/geometry -lgeometry -lm -o ex3_3.out
$
```

Shell Box 3-14：使用示例 3.2 中创建的静态库进行链接

为了解释前面的命令，我们将分别解释每个传递选项：

+   `-L/opt/geometry` 告诉 `gcc` 将目录 `/opt/geometry` 视为静态和共享库可能存在的多个位置之一。默认情况下，链接器会在已知路径如 `/usr/lib` 或 `/usr/local/lib` 中搜索库文件。如果你没有指定 `-L` 选项，链接器只会在默认路径中搜索。

+   `-lgeometry` 告诉 `gcc` 查找文件 `libgeometry.a` 或 `libgeometry.so`。以 `.so` 结尾的文件是共享对象文件，我们将在下一节中解释。注意使用的约定。例如，如果你传递 `-lxyz` 选项，链接器将在默认和指定目录中搜索文件 `libxyz.a` 或 `libxyz.so`。如果找不到文件，链接器将停止并生成错误。

+   `-lm` 告诉 `gcc` 查找另一个名为 `libm.a` 或 `libm.so` 的库。这个库保存了 *glibc* 中的数学函数定义。我们需要它来使用 `cos`、`sin` 和 `acos` 函数。请注意，我们正在 Linux 机器上构建 *example 3.3*，它使用 *glibc* 作为其默认 C 库的实现。在 macOS 和可能的一些其他类 Unix 系统中，你不需要指定此选项。

+   `-o ex3_3.out` 告诉 `gcc` 输出的可执行文件应该命名为 `ex3_3.out`。

在运行前面的命令后，如果一切顺利，你将有一个包含在静态库 `libgeometry.a` 中找到的所有可重定位目标文件以及 `main.o` 的可执行二进制文件。

注意，在链接之后，不会对静态库文件的存在有任何依赖，因为所有内容都*嵌入*在可执行文件本身中。换句话说，最终的可执行文件可以独立运行，无需静态库存在。

然而，从许多静态库的链接中产生的可执行文件通常具有很大的体积。静态库越多，其中的可重定位目标文件越多，最终可执行文件的体积就越大。有时它可以达到几百兆字节，甚至几吉字节。

这是在二进制文件的大小和它可能具有的依赖项之间的一种权衡。你可以拥有一个更小的二进制文件，但通过使用共享库。这意味着最终的二进制文件并不完整，如果外部共享库不存在或找不到，则无法运行。我们将在接下来的章节中更多地讨论这个问题。

在本节中，我们描述了静态库是什么，以及它们应该如何创建和使用。我们还演示了另一个程序如何使用公开的 API 并将其链接到现有的静态库。在下一节中，我们将讨论动态库以及如何从 *example 3.2* 的源代码中生成共享对象文件（动态库），而不是使用静态库。

# 动态库

动态库，或共享库，是生成可重用库的另一种方式。正如其名所示，与静态库不同，动态库不是最终可执行文件本身的一部分。相反，它们应该在加载用于执行的过程时加载和引入。

由于静态库是可执行文件的一部分，链接器会将给定可重定位文件中找到的所有内容放入最终的可执行文件中。换句话说，链接器检测未定义的符号和所需的定义，并尝试在给定的可重定位目标文件中找到它们，然后将它们全部放入输出可执行文件中。

只有当所有未定义的符号都被找到时，最终产品才会被生成。从独特的角度来看，我们在链接时检测所有依赖关系并解决它们。至于动态库，在链接时可能存在未解决的未定义符号。这些符号将在可执行产品即将加载并开始执行时被搜索。

换句话说，当你有未定义的动态符号时，需要不同的链接步骤。一个*动态链接器*，或者简单地称为*加载器*，通常在加载可执行文件并准备将其作为进程运行时执行链接操作。

由于未定义的动态符号没有在可执行文件中找到，它们应该在别处找到。这些符号应该从共享对象文件中加载。这些文件是静态库文件的姐妹文件。虽然静态库文件在其名称中有`.a`扩展名，但共享对象文件在大多数类 Unix 系统中携带`.so`扩展名。在 macOS 中，它们有`.dylib`扩展名。

当加载一个进程并即将启动时，一个共享对象文件将被加载并映射到一个进程可访问的内存区域。这个步骤是由动态链接器（或加载器）完成的，它加载并执行可执行文件。

正如我们在专门讨论可执行对象文件的章节中所说的那样，ELF 可执行文件和共享对象文件在其 ELF 结构中都有段。每个段包含零个或多个节。ELF 可执行对象文件和 ELF 共享对象文件之间有两个主要区别。首先，符号具有相对绝对地址，这使得它们可以作为许多进程的一部分同时加载。

这意味着虽然每个进程中的每条指令的地址都不同，但两条指令之间的距离保持固定。换句话说，地址相对于偏移量是固定的。这是因为可重定位对象文件是*位置无关的*。我们将在本章的最后部分更多地讨论这一点。

例如，如果两条指令在一个进程中的地址是 100 和 200，在另一个进程中它们可能在 140 和 240，在另一个进程中可能是 323 和 423。相关的地址是绝对的，但实际地址可以改变。这两条指令之间的距离始终是 100 个地址。

第二个区别是，与加载 ELF 可执行对象文件相关的某些段在共享对象文件中不存在。这实际上意味着共享对象文件不能被执行。

在详细介绍如何从不同的进程访问共享对象之前，我们需要展示一个示例，说明它们是如何创建和使用的。因此，我们将为我们在上一节中工作的相同几何库，*示例 3.2*，创建动态库。

在上一节中，我们为几何库创建了一个静态库。在本节中，我们想要再次编译源代码，以便从中创建一个共享对象文件。以下命令显示了如何将三个源代码编译成它们对应的位置无关可重定位目标文件，与我们在 *示例 3.2* 中所做的工作只有一个不同。在以下命令中，请注意传递给 `gcc` 的 `-fPIC` 选项：

```cpp
$ gcc -c ExtremeC_examples_chapter3_2_2d.c -fPIC -o 2d.o
$ gcc -c ExtremeC_examples_chapter3_2_3d.c -fPIC -o 3d.o
$ gcc -c ExtremeC_examples_chapter3_2_trigon.c -fPIC -o trigon.o
$
```

Shell Box 3-15：将示例 3.2 的源代码编译成相应的位置无关可重定位目标文件

观察命令，你可以看到我们在编译源代码时传递了一个额外的选项 `-fPIC` 给 `gcc`。如果你打算从一些可重定位目标文件中创建共享对象文件，这个选项是**强制性的**。**PIC**代表**位置无关代码**。正如我们之前解释的，如果一个可重定位目标文件是位置无关的，那么它仅仅意味着其中的指令没有固定的地址。相反，它们有相对地址；因此，它们可以在不同的进程中获得不同的地址。这是由于我们使用共享对象文件的方式所要求的。

没有保证加载程序将在不同进程中以相同的地址加载共享对象文件。实际上，加载程序为共享对象文件创建内存映射，这些映射的地址范围可能不同。如果指令地址是绝对的，我们就不能同时在不同的进程和不同的内存区域中加载相同的共享对象文件。

**注意**：

要获取有关程序和共享对象文件动态加载工作方式的更详细信息，你可以查看以下资源：

+   [`software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf`](https://software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf)

+   [`www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html`](https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html)

要创建共享对象文件，你需要再次使用编译器，在这种情况下，是 `gcc`。与是一个简单的存档的静态库文件不同，共享对象文件本身就是一个目标文件。因此，它们应该由相同的链接程序创建，例如我们用来生成可重定位目标文件的 `ld`。

我们知道，在大多数类 Unix 系统中，`ld` 会这样做。然而，强烈建议不要直接使用 `ld` 链接对象文件，原因我们在上一章中已经解释过。

以下命令显示了如何从使用 `-fPIC` 选项编译的多个可重定位目标文件中创建共享对象文件：

```cpp
$ gcc -shared 2d.o 3d.o trigon.o -o libgeometry.so
$ mkdir -p /opt/geometry
$ mv libgeometry.so /opt/geometry
$
```

Shell Box 3-16：从可重定位目标文件中创建共享对象文件

如您在第一个命令中看到的，我们传递了 `-shared` 选项，要求 `gcc` 从可重定位目标文件中创建一个共享对象文件。结果是名为 `libgeometry.so` 的共享对象文件。我们已经将共享对象文件移动到 `/opt/geometry` 以便其他愿意使用它的程序可以轻松访问。下一步是再次编译和链接 *示例 3.3*。

之前，我们使用创建的静态库文件 `libgeometry.a` 编译并链接了 *示例 3.3*。这里，我们将做同样的事情，但我们将使用 `libgeometry.so`，一个动态库来链接它。

虽然一切似乎都一样，特别是命令，但事实上它们是不同的。这次，我们将用 `libgeometry.so` 而不是 `libgeometry.a` 链接 *示例 3.3*，而且不仅如此，动态库不会嵌入到最终的执行文件中，而是在执行时加载库。在练习这个过程中，确保在再次链接 *示例 3.3* 之前，你已经从 `/opt/geometry` 中移除了静态库文件 `libgeometry.a`：

```cpp
$ rm -fv /opt/geometry/libgeometry.a
$ gcc -c ExtremeC_examples_chapter3_3.c -o main.o
$ gcc main.o -L/opt/geometry-lgeometry -lm -o ex3_3.out
$
```

Shell Box 3-17：将示例 3.3 链接到构建的共享对象文件

正如我们之前解释的那样，选项 `-lgeometry` 告诉编译器查找并使用一个库，无论是静态的还是共享的，以便将其与其它目标文件链接。由于我们已经移除了静态库文件，所以选择了共享对象文件。如果定义的库既有静态库文件又有共享对象文件，那么 `gcc` 会优先选择共享对象文件并将其与程序链接。

如果你现在尝试运行可执行文件 `ex3_3.out`，你很可能会遇到以下错误：

```cpp
$ ./ex3_3.out
./ex3_3.out: error while loading shared libraries: libgeometry.so: cannot open shared object file: No such file or directory
$
```

Shell Box 3-18：尝试运行示例 3.3

我们之前没有看到这个错误，因为我们使用的是静态链接和静态库。但现在，通过引入动态库，如果我们打算运行一个具有 *动态依赖性* 的程序，我们应该提供所需的动态库以便它能够运行。但发生了什么，为什么我们会收到错误信息？

可执行文件 `ex3_3.out` 依赖于 `libgeometry.so` 库。这是因为它需要的某些定义只能在该共享对象文件中找到。我们应该注意，这并不适用于静态库 `libgeometry.a`。与静态库链接的可执行文件可以作为独立可执行文件运行，因为它已经从静态库文件中复制了所有内容，因此不再依赖于其存在。

对于共享对象文件来说，情况并非如此。我们收到错误是因为程序加载器（动态链接器）在其默认搜索路径中找不到 `libgeometry.so`。因此，我们需要将 `/opt/geometry` 添加到其搜索路径中，以便在那里找到 `libgeometry.so` 文件。为此，我们将更新环境变量 `LD_LIBRARY_PATH` 以指向当前目录。

加载程序将检查这个环境变量的值，并将在指定的路径中搜索所需的共享库。请注意，在这个环境变量中可以指定多个路径（使用冒号 `:` 作为分隔符）。

```cpp
$ export LD_LIBRARY_PATH=/opt/geometry 
$ ./ex3_3.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-19：通过指定 LD_LIBRARY_PATH 运行示例 3.3

这次，程序已经成功运行！这意味着程序加载器已经找到了共享对象文件，动态链接器已经成功从其中加载所需的符号。

注意，在前面的 shell box 中，我们使用了 `export` 命令来更改 `LD_LIBRARY_PATH`。然而，将环境变量与执行命令一起设置是很常见的。你可以在下面的 shell box 中看到这一点。两种用法的结果将是相同的：

```cpp
$ LD_LIBRARY_PATH=/opt/geometry ./ex3_3.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-20：通过指定 LD_LIBRARY_PATH 作为同一命令的一部分运行示例 3.3

通过将可执行文件与多个共享对象文件链接，就像我们之前做的那样，我们告诉系统这个可执行文件需要找到并加载在运行时所需的多个共享库。因此，在运行可执行文件之前，加载程序会自动搜索这些共享对象文件，并将所需的符号映射到进程可访问的正确地址。只有在这种情况下，处理器才能开始执行。

## 手动加载共享库

共享对象文件也可以以不同的方式加载和使用，在这种情况下，它们不是由加载程序（动态链接器）自动加载的。相反，程序员将使用一系列函数在需要使用共享库中可找到的一些符号（函数）之前手动加载共享对象文件。这种手动加载机制有一些应用，一旦我们讨论了本节中将要查看的示例，我们就会谈到它们。

*Example 3.4* 展示了如何懒加载或手动加载共享对象文件，而不在链接步骤中包含它。这个例子借鉴了 *example 3.3* 的相同逻辑，但不同之处在于它手动在程序内部加载共享对象文件 `libgeometry.so`。

在进入 *example 3.4* 之前，我们需要以不同的方式生成 `libgeometry.so`，以便 *example 3.4* 能够工作。为此，我们必须在 Linux 中使用以下命令：

```cpp
$ gcc -shared 2d.o 3d.o trigon.o -lm -o libgeometry.so
$
```

Shell Box 3-21：将几何共享对象文件链接到标准数学库

查看前面的命令，你可以看到一个新选项 `-lm`，它告诉链接器将共享对象文件链接到标准数学库 `libm.so`。这样做是因为当我们手动加载 `libgeometry.so` 时，它的依赖项应该以某种方式自动加载。如果不是这样，那么我们将得到关于 `libgeometry.so` 本身所需的符号的错误，例如 `cos` 或 `sqrt`。请注意，我们不会将最终的可执行文件与数学标准库链接，它将在加载 `libgeometry.so` 时由加载程序自动解析。

现在我们有了链接的共享对象文件，我们可以继续进行*示例 3.4*：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "ExtremeC_examples_chapter3_2_geometry.h"
polar_pos_2d_t (*func_ptr)(cartesian_pos_2d_t*);
int main(int argc, char** argv) {
  void* handle = dlopen ("/opt/geometry/libgeometry.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  func_ptr = dlsym(handle, "convert_to_2d_polar_pos");
  if (!func_ptr)  {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  cartesian_pos_2d_t cartesian_pos;
  cartesian_pos.x = 100;
  cartesian_pos.y = 200;
  polar_pos_2d_t polar_pos = func_ptr(&cartesian_pos);
  printf("Polar Position: Length: %f, Theta: %f (deg)\n",
    polar_pos.length, polar_pos.theta);
  return 0;
}
```

Code Box 3-8 [ExtremeC_examples_chapter3_4.c]：示例 3.4 手动加载几何共享对象文件

通过查看前面的代码，你可以看到我们是如何使用`dlopen`和`dlsym`函数来加载共享对象文件，并在其中找到`convert_to_2d_polar_pos`符号。`dlsym`函数返回一个函数指针，可以用来调用目标函数。

值得注意的是，前面的代码在`/opt/geometry`中搜索共享对象文件，如果没有这样的对象文件，则会显示错误信息。请注意，在 macOS 中，共享对象文件的扩展名以`.dylib`结尾。因此，前面的代码应该修改为加载具有正确扩展名的文件。

以下命令编译了前面的代码并运行可执行文件：

```cpp
$ gcc ExtremeC_examples_chapter3_4.c -ldl -o ex3_4.out
$ ./ex3_4.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-22：运行示例 3.4

正如你所见，我们没有将程序与`libgeometry.so`文件链接。我们没有这样做是因为我们希望在需要时手动加载它。这种方法通常被称为共享对象文件的*延迟加载*。尽管名称如此，但在某些场景下，延迟加载共享对象文件可能非常有用。

其中一个例子是当你有不同的共享对象文件用于同一库的不同实现或版本时。延迟加载让你有更大的自由度，可以根据自己的逻辑和需要加载所需的共享对象，而不是在加载时自动加载，那时你对它们的控制较少。

# 摘要

本章主要讨论了各种类型的对象文件，它们是 C/C++项目构建后的产物。作为本章的一部分，我们涵盖了以下内容：

+   我们讨论了 API 和 ABI，以及它们之间的区别。

+   我们探讨了各种对象文件格式，并简要回顾了它们的历史。它们都有相同的祖先，但它们在特定的路径上发生了变化，成为了今天的模样。

+   我们讨论了可重定位对象文件及其内部结构，特别是关于 ELF 可重定位对象文件。

+   我们讨论了可执行对象文件，以及它们与可重定位对象文件之间的区别。我们还查看了一个 ELF 可执行对象文件。

+   我们展示了静态和动态符号表，以及如何使用一些命令行工具读取它们的内容。

+   我们讨论了静态链接和动态链接，以及如何查找各种符号表以生成最终的二进制文件或执行程序。

+   我们讨论了静态库文件，以及它们实际上是包含多个可重定位对象文件的归档文件。

+   讨论了共享对象文件（动态库），并演示了如何将多个可重定位对象文件组合成它们。

+   我们解释了什么是位置无关代码以及为什么参与创建共享库的可重定位目标文件必须是位置无关的。

在下一章中，我们将探讨进程的内存结构；这是 C/C++编程中的另一个关键主题。下一章将描述各种内存段，我们将看到如何编写没有内存问题的代码。
