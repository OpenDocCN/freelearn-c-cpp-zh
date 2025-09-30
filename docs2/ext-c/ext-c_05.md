# 第五章

# 栈和堆

在前一章中，我们对正在运行的进程的内存布局进行了调查。如果不了解足够的内存结构和其各个段，进行系统编程就像在不知道人体解剖学的情况下进行手术一样。前一章只是提供了关于进程内存布局中不同段的基本信息，但本章希望我们只关注最常用的段：栈和堆。

作为程序员，你大部分时间都在忙于处理栈和堆段。其他段，如数据或 BSS 段，使用较少，或者你对它们控制较少。这基本上是因为数据和 BSS 段是由编译器生成的，通常，在进程的生命周期中，它们只占用整个内存的一小部分。这并不意味着它们不重要，实际上，有一些问题直接与这些段相关。但因为你大部分时间都在处理栈和堆，所以大多数内存问题都源于这些段。

作为本章的一部分，你将学习：

+   如何探测栈段以及为此目的所需的工具

+   栈段是如何自动进行内存管理的

+   栈段的各项特性

+   关于如何使用栈段的指南和最佳实践

+   如何探测堆段

+   如何分配和释放堆内存块

+   关于堆段使用的指南和最佳实践

+   内存受限环境和性能环境中的内存调整

让我们通过更详细地讨论栈段来开始我们的探索之旅。

# 栈

一个进程可以在没有堆段的情况下继续工作，但不能在没有栈段的情况下工作。这说明了很多。栈是进程代谢的主要部分，没有它就无法继续执行。原因隐藏在驱动函数调用的机制背后。正如前一章简要解释的，调用函数只能通过使用栈段来完成。没有栈段，就无法进行函数调用，这意味着根本无法执行。

如此一来，栈段及其内容被精心设计，以确保过程的健康执行。因此，干扰栈内容可能会破坏执行并停止进程。从栈段分配内存速度快，且不需要任何特殊函数调用。更重要的是，释放内存和所有内存管理任务都是自动发生的。所有这些事实都非常诱人，并鼓励你过度使用栈。

你应该对此保持警惕。使用栈段会带来自己的复杂性。栈并不大，因此你无法在其中存储大对象。此外，栈内容的错误使用可能导致执行中断并引发崩溃。以下代码片段展示了这一点：

```cpp
#include <string.h>
int main(int argc, char** argv) {
  char str[10];
  strcpy(str, "akjsdhkhqiueryo34928739r27yeiwuyfiusdciuti7twe79ye");
  return 0;
}
```

代码框 5-1：缓冲区溢出情况。strcpy 函数将覆盖栈的内容

当运行前面的代码时，程序很可能会崩溃。这是因为`strcpy`正在覆盖栈的内容，或者如通常所说的，*破坏*栈。正如你在*代码框 5-1*中看到的，`str`数组有`10`个字符，但`strcpy`正在向`str`数组写入超过 10 个字符。正如你很快就会看到的，这实际上是在写入之前推入的变量和栈帧，程序在从`main`函数返回后会跳转到错误的指令。这最终使得程序无法继续执行。

希望前面的例子已经帮助你理解了栈段的微妙之处。在本章的前半部分，我们将更深入地研究栈，并对其进行仔细检查。我们首先从探测栈开始。

## 探测栈

在了解更多关于栈的信息之前，我们需要能够读取它，也许还能修改它。正如前一章所述，栈段是只有所有者进程才有权读取和修改的私有内存。如果我们打算读取栈或更改它，我们需要成为拥有栈的进程的一部分。

这就是一套新工具的用武之地：*调试器*。调试器是一种程序，它可以附加到另一个进程上以对其进行*调试*。在调试进程时，一个常见的任务就是观察和操作各种内存段。只有在调试进程时，我们才能读取和修改私有内存块。作为调试的一部分，还可以控制程序指令的执行顺序。在本节中，我们将通过示例展示如何使用调试器来完成这些任务。

让我们从例子开始。在*示例 5.1*中，我们展示了如何编译程序并使其准备好进行调试。然后，我们演示了如何使用`gdb`（GNU 调试器）来运行程序并读取栈内存。此示例声明了一个在栈顶分配的字符数组，并用一些字符填充其元素，如下面的代码框所示：

```cpp
#include <stdio.h>
int main(int argc, char** argv) {
  char arr[4];
  arr[0] = 'A';
  arr[1] = 'B';
  arr[2] = 'C';
  arr[3] = 'D';
  return 0;
}
```

代码框 5-2 [ExtremeC_examples_chapter5_1.c]：在栈顶分配的数组声明

程序简单易懂，但内存内部发生的事情很有趣。首先，`arr`数组所需的内存是从栈中分配的，因为它不是从堆段分配的，我们没有使用`malloc`函数。记住，栈段是分配变量和数组的默认位置。

为了从堆中分配一些内存，应该通过调用`malloc`或其他类似函数来获取它，例如`calloc`。否则，内存将从栈中分配，更确切地说，是在栈顶。

为了能够调试一个程序，二进制文件必须为调试目的而构建。这意味着我们必须告诉编译器我们想要一个包含*调试* *符号*的二进制文件。这些符号将用于找到正在执行的代码行或导致崩溃的代码行。让我们编译*example 5.1*并创建一个包含调试符号的可执行目标文件。

首先，我们构建示例。我们在 Linux 环境中进行编译：

```cpp
$ gcc -g ExtremeC_examples_chapter5_1.c -o ex5_1_dbg.out
$
```

Shell Box 5-1：使用调试选项-g 编译 example 5.1

`-g`选项告诉编译器最终的可执行目标文件必须包含调试信息。当使用和不使用调试选项编译源代码时，二进制文件的大小也会不同。接下来，你可以看到两个可执行目标文件大小的差异，第一个是未使用`-g`选项构建的，第二个是使用`-g`选项构建的：

```cpp
$ gcc ExtremeC_examples_chapter2_10.c -o ex5_1.out
$ ls -al ex5_1.out
-rwxrwxr-x 1 kamranamini kamranamini 8640 jul 24 13:55 ex5_1.out
$ gcc -g ExtremeC_examples_chapter2_10.c -o ex5_1_dbg.out
$ ls -al ex5_1.out
-rwxrwxr-x 1 kamranamini kamranamini 9864 jul 24 13:56 ex5_1_dbg.out
$
```

Shell Box 5-2：带有和不带有`-g`选项的输出可执行目标文件的大小

现在我们有一个包含调试符号的可执行文件，我们可以使用调试器来运行程序。在这个例子中，我们将使用`gdb`来调试*example 5.1*。接下来，你可以找到启动调试器的命令：

```cpp
$ gdb ex5_1_dbg.out
```

Shell Box 5-3：启动 example 5.1 的调试器

**注意**：

`gdb`通常作为`build-essentials`包的一部分安装在 Linux 系统上。在 macOS 系统上，可以使用`brew`包管理器安装，如下所示：`brew install gdb`。

运行调试器后，输出将类似于以下 Shell Box：

```cpp
$ gdb ex5_1_dbg.out
GNU gdb (Ubuntu 7.11.1-0ubuntu1~16.5) 7.11.1
Copyright (C) 2016 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later http://gnu.org/licenses/gpl.html
...
Reading symbols from ex5_1_dbg.out...done.
(gdb)
```

Shell Box 5-4：启动调试器后的输出

如你所注意到的，我已在 Linux 机器上运行了前面的命令。`gdb`有一个命令行界面，允许你发出调试命令。输入`r`（或`run`）命令以执行调试器指定的可执行目标文件。以下 Shell Box 显示了`run`命令如何执行程序：

```cpp
...
Reading symbols from ex5_1_dbg.out...done.
(gdb) run
Starting program: .../extreme_c/5.1/ex5_1_dbg.out
[Inferior 1 (process 9742) exited normally]
(gdb)
```

Shell Box 5-5：发出`run`命令后调试器的输出

在前面的 Shell Box 中，在发出`run`命令后，`gdb`已经启动了进程，附加到它上面，并让程序执行其指令并退出。它没有中断程序，因为我们没有设置*断点*。断点是一个指示器，告诉`gdb`暂停程序执行并等待进一步的指令。你可以设置任意多的断点。

接下来，我们使用`b`（或`break`）命令在`main`函数上设置断点。设置断点后，当程序进入`main`函数时，`gdb`会暂停执行。以下 Shell Box 显示了如何在`main`函数上设置断点：

```cpp
(gdb) break main
Breakpoint 1 at 0x400555: file ExtremeC_examples_chapter5_1.c, line 4.
(gdb)
```

Shell Box 5-6：在 gdb 中设置 main 函数的断点

现在，我们再次运行程序。这会创建一个新的进程，并且`gdb`会附加到它上面。接下来，你可以找到结果：

```cpp
(gdb) r
Starting program: .../extreme_c/5.1/ex5_1_dbg.out
Breakpoint 1, main (argc=1, argv=0x7fffffffcbd8) at ExtremeC_examples_chapter5_1.c:3
3       int main(int argc, char** argv) {
(gdb)
```

Shell Box 5-7：设置断点后再次运行程序

正如你所看到的，执行暂停在第 3 行，这正是 `main` 函数的行。然后，调试器等待下一个命令。现在，我们可以要求 `gdb` 运行下一行代码并再次暂停。换句话说，我们逐行逐行地运行程序。这样，你有足够的时间四处查看并检查内存中变量及其值。实际上，这是我们用来探测堆栈和堆段的技巧。

在下面的 Shell Box 中，你可以看到如何使用 `n`（或 `next`）命令来运行下一行代码：

```cpp
(gdb) n
5         arr[0] = 'A';
(gdb) n
6         arr[1] = 'B';
(gdb) next
7        arr[2] = 'C';
(gdb) next
8        arr[3] = 'D';
(gdb) next
9        return 0;
(gdb)
```

Shell Box 5-8：使用 n（或 next）命令执行即将到来的代码行

现在，如果你在调试器中输入 `print arr` 命令，它将显示数组的内容作为一个字符串：

```cpp
(gdb) print arr
$1 = "ABCD"
(gdb)
```

Shell Box 5-9：使用 gdb 打印 arr 数组的内容

为了回到主题，我们介绍了 `gdb` 以能够查看堆栈内存。现在，我们可以做到了。我们有一个具有堆栈段的进程，它是暂停的，并且我们有一个 `gdb` 命令行来探索其内存。让我们开始并打印 `arr` 数组分配的内存：

```cpp
(gdb) x/4b arr
0x7fffffffcae0: 0x41    0x42    0x43    0x44
(gdb) x/8b arr
0x7fffffffcae0: 0x41    0x42    0x43    0x44    0xff    0x7f    0x00    0x00
(gdb)
```

Shell Box 5-10：从 arr 数组开始打印内存字节

第一条命令 `x/4b` 显示了从 `arr` 所指向的位置开始的 4 个字节。记住，`arr` 是一个指针，实际上它指向数组的第一个元素，因此它可以用来在内存中移动。

第二条命令 `x/8b` 在 `arr` 之后打印 8 个字节。根据为 *example 5.1* 编写的代码，并在 *Code Box 5-2* 中找到，值 `A`、`B`、`C` 和 `D` 存储在数组 `arr` 中。你应该知道，ASCII 值存储在数组中，而不是真正的字符。`A` 的 ASCII 值是十进制的 `65` 或十六进制的 `0x41`。对于 `B`，它是 `66` 或 `0x42`。正如你所看到的，`gdb` 输出中打印的值就是我们刚刚存储在 `arr` 数组中的值。

第二条命令中的其他 4 个字节是什么？它们是堆栈的一部分，并且它们可能包含在调用 `main` 函数时放在堆栈顶部的最近堆栈帧中的数据。

注意，与其他段相比，堆栈段是以相反的方式填充的。

其他内存区域是从较小的地址开始填充，并向前移动到较大的地址，但堆栈段的情况并非如此。

堆栈段是从较大的地址开始填充，并向后移动到较小的地址。这种设计背后的原因部分在于现代计算机的开发历史，部分在于堆栈段的功能，它表现得像一个堆栈数据结构。

说了这么多，如果你像我们在*Shell Box 5-10*中做的那样，从地址段向更大的地址读取 Stack 段，你实际上是在将已推入的内容作为 Stack 段的一部分来读取，如果你尝试更改这些字节，你就是在更改 Stack，这是不好的。我们将在未来的段落中演示为什么这是危险的以及如何做到这一点。

为什么我们能看到比`arr`数组大小更多的内容？因为`gdb`会遍历我们请求的内存中的字节数。`x`命令不关心数组的边界。它只需要一个起始地址和要打印的字节数。

如果你想要更改 Stack 中的值，你必须使用`set`命令。这允许你修改现有的内存单元。在这种情况下，内存单元指的是`arr`数组中的单个字节：

```cpp
(gdb) x/4b arr
0x7fffffffcae0: 0x41    0x42    0x43    0x44
(gdb) set arr[1] = 'F'
(gdb) x/4b arr
0x7fffffffcae0: 0x41    0x46    0x43    0x44
(gdb) print arr
$2 = "AFCD"
(gdb)
```

Shell Box 5-11：使用 set 命令更改数组中的单个字节

如你所见，使用`set`命令，我们已经将`arr`数组的第二个元素设置为`F`。如果你打算更改不在你的数组边界内的地址，仍然可以通过`gdb`来实现。

请仔细观察以下修改。现在，我们想要修改一个位于比`arr`大得多的地址的字节，正如我们之前解释的，我们将更改 Stack 中已推入的内容。记住，Stack 内存的填充方式与其他段相反：

```cpp
(gdb) x/20x arr
0x7fffffffcae0: 0x41    0x42    0x43    0x44    0xff    0x7f    0x00    0x00
0x7fffffffcae8: 0x00    0x96    0xea    0x5d    0xf0    0x31    0xea    0x73
0x7fffffffcaf0: 0x90    0x05    0x40    0x00
(gdb) set *(0x7fffffffcaed) = 0xff
(gdb) x/20x arr
0x7fffffffcae0: 0x41    0x42    0x43    0x44    0xff    0x7f    0x00    0x00
0x7fffffffcae8: 0x00    0x96    0xea    0x5d    0xf0    0xff    0x00    0x00
0x7fffffffcaf0: 0x00    0x05    0x40    0x00
(gdb)
```

Shell Box 5-12：在数组边界之外更改单个字节

那就是全部。我们只是在`0x7fffffffcaed`地址中写入了值`0xff`，这个地址超出了`arr`数组的边界，可能是在进入`main`函数之前推入的栈帧中的某个字节。

如果我们继续执行会发生什么？如果我们修改了 Stack 中的关键字节，我们预计会看到崩溃，或者至少通过某种机制检测到这种修改，并使程序执行停止。`c`（或`continue`）命令将在`gdb`中继续进程的执行，正如你接下来可以看到的：

```cpp
(gdb) c
Continuing.
*** stack smashing detected ***: .../extreme_c/5.1/ex5_1_dbg.out terminated
Program received signal SIGABRT, Aborted.
0x00007ffff7a42428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/Unix/sysv/linux/raise.c:54
54      ../sysdeps/Unix/sysv/linux/raise.c: No such file or directory.
(gdb)
```

Shell Box 5-13：在 Stack 中更改关键字节会终止进程

如前所述的 shell box 所示，我们刚刚破坏了 Stack！在未分配给你的地址中修改 Stack 的内容，即使只修改 1 个字节，也可能非常危险，通常会导致崩溃或突然终止。

正如我们之前所说的，与程序执行相关的许多重要过程都是在 Stack 内存中完成的。因此，你在写入 Stack 变量时应该非常小心。你不应该在变量和数组的定义边界之外写入任何值，仅仅因为 Stack 内存中的地址是向后增长的，这使得覆盖已写入的字节变得很可能会发生。

当你完成调试，准备离开`gdb`时，你可以简单地使用命令`q`（或`quit`）。现在，你应该已经离开了调试器，回到了终端。

另外需要注意的是，将未经检查的值写入在栈顶（而非堆）分配的*缓冲区*（字节数组或字符数组的另一种称呼）被视为一个漏洞。攻击者可以精心设计一个字节数组并将其提供给程序，以控制程序。这通常被称为*漏洞利用*，因为它涉及到*缓冲区溢出*攻击。

以下程序展示了这个漏洞：

```cpp
int main(int argc, char** argv) {
  char str[10];
  strcpy(str, argv[1]);
  printf("Hello %s!\n", str);
}
```

代码框 5-3：展示缓冲区溢出漏洞的程序

前面的代码没有检查`argv[1]`输入的内容和大小，直接将其复制到在栈顶分配的`str`数组中。

如果你很幸运，这可能导致崩溃，但在一些罕见但危险的情况下，这可能导致漏洞利用攻击。

## 使用栈内存的要点

现在你对栈段及其工作原理有了更好的理解，我们可以讨论最佳实践和你应该注意的要点。你应该熟悉*作用域*的概念。每个栈变量都有自己的作用域，作用域决定了变量的生命周期。这意味着栈变量在其生命周期开始于一个作用域，并在该作用域消失时结束。换句话说，作用域决定了栈变量的生命周期。

我们对栈变量也有自动的内存分配和释放，这仅适用于栈变量。这个特性，自动内存管理，来源于栈段的本质。

每次你声明一个栈变量时，它都会被分配在栈段顶部。分配是自动发生的，这可以标记为其生命周期的开始。在此之后，许多更多的变量和其他栈帧被放在栈的顶部。只要变量存在于栈中，并且有其他变量在其上方，它就会存活并继续存在。

然而，这些内容最终会从栈中弹出，因为将来某个时刻程序必须结束，此时栈应该是空的。因此，在未来的某个时刻，这个变量应该从栈中弹出。因此，释放或弹出是自动发生的，这可以标记为变量生命周期的结束。这基本上是我们说我们对栈变量具有自动内存管理，这种管理不由程序员控制的原因。

假设你在`main`函数中定义了一个变量，如下面的代码框所示：

```cpp
int main(int argc, char** argv) {
  int a;
  ...
  return 0;
}
```

代码框 5-4：在栈上声明变量

这个变量将保留在栈中，直到`main`函数返回。换句话说，变量存在直到其作用域（`main`函数）有效。由于`main`函数是所有程序运行的函数，因此变量的生命周期几乎像一个在整个程序运行期间声明的全局变量。

它像一个全局变量，但又不完全一样，因为变量会在某个时刻从栈中弹出，而全局变量即使在`main`函数完成并且程序正在最终化时，其内存仍然存在。请注意，在`main`函数之前和之后运行了两段代码，分别是程序的引导和最终化。作为另一个注意事项，全局变量是从不同的段分配的，如数据或 BSS 段，它不像栈段那样表现。

让我们现在看看一个非常常见的错误的例子。这通常发生在编写第一个 C 程序时的业余程序员身上。它涉及到在函数内部返回局部变量的地址。

以下代码框显示了*示例 5.2*：

```cpp
int* get_integer() {
  int var = 10;
  return &var;
}
int main(int argc, char** argv) {
  int* ptr = get_integer();
  *ptr = 5;
  return 0;
}
```

代码框 5-5 [ExtremeC_examples_chapter5_2.c]：在栈顶声明一个变量

`get_integer`函数返回一个指向在`get_integer`函数作用域内声明的局部变量`var`的地址。`get_integer`函数返回局部变量的地址。然后，`main`函数尝试解引用接收到的指针并访问其后的内存区域。以下是在 Linux 系统上编译上述代码时`gcc`编译器的输出：

```cpp
$ gcc ExtremeC_examples_chapter5_2.c -o ex5_2.out
ExtremeC_examples_chapter5_2.c: In function 'get_integer':
ExtremeC_examples_chapter5_2.c:3:11: warning: function returns address of local variable [-Wreturn-local-addr]
   return &var;
          ^~~~
$
```

Shell Box 5-14：在 Linux 中编译示例 5.2

如您所见，我们收到了一个警告消息。由于返回局部变量的地址是一个常见的错误，编译器已经知道这一点，并且会显示一个清晰的警告消息，如`warning: function returns address of a local variable`。

这就是程序执行时发生的情况：

```cpp
$ ./ex5_2.out
Segmentation fault (core dumped)
$
```

Shell Box 5-15：在 Linux 中执行示例 5.2

如您在*Shell Box 5-15*中看到的那样，发生了段错误。它可以被翻译为崩溃。这通常是因为对某个在某个时刻已分配但现在已经取消分配的内存区域的无效访问。

**注意**：

应该将一些警告视为错误。例如，前面的警告应该是一个错误，因为它通常会导致崩溃。如果您想将所有警告都视为错误，只需将`-Werror`选项传递给`gcc`编译器即可。如果您只想将一个特定的警告视为错误，例如前面的警告，只需传递`-Werror=return-local-addr`选项即可。

如果您使用`gdb`运行程序，您将看到有关崩溃的更多详细信息。但请记住，您需要使用`-g`选项编译程序，否则`gdb`不会那么有帮助。

如果你打算使用`gdb`或其他调试工具（如`valgrind`）来调试程序，那么始终必须使用`-g`选项编译源代码。以下 shell 窗口演示了如何在调试器中编译和运行*示例 5.2*：

```cpp
$ gcc -g ExtremeC_examples_chapter5_2.c -o ex5_2_dbg.out
ExtremeC_examples_chapter5_2.c: In function 'get_integer':
ExtremeC_examples_chapter5_2.c:3:11: warning: function returns address of local variable [-Wreturn-local-addr]
   return &var;
          ^~~~
$ gdb ex5_2_dbg.out
GNU gdb (Ubuntu 8.1-0ubuntu3) 8.1.0.20180409-git
...
Reading symbols from ex5_2_dbg.out...done.
(gdb) run
Starting program: .../extreme_c/5.2/ex5_2_dbg.out
Program received signal SIGSEGV, Segmentation fault.
0x00005555555546c4 in main (argc=1, argv=0x7fffffffdf88) at ExtremeC_examples_chapter5_2.c:8
8    *ptr = 5;
(gdb) quit
$
```

Shell Box 5-16：在调试器中运行示例 5.2

如`gdb`输出所示，崩溃的来源位于`main`函数的第 8 行，正好是程序尝试通过解引用返回的指针来写入返回地址的地方。但是，`var`变量已经成为`get_integer`函数的局部变量，并且它不再存在，仅仅因为我们在第 8 行已经从`get_integer`函数及其作用域返回，以及所有变量，都已经消失。因此，返回的指针是一个*悬垂指针*。

通常，将指向当前作用域内变量的指针传递给其他函数，而不是相反，是一种常见的做法，因为只要当前作用域有效，变量就在那里。进一步的函数调用只会将更多东西放在栈段顶部，并且当前作用域不会在它们之前结束。

注意，上述关于并发程序的说法并不是一个好的实践，因为在将来，如果另一个并发任务想要使用指向当前作用域内变量的接收到的指针，当前作用域可能已经不存在了。

为了结束这一节，并对栈段得出结论，我们可以从到目前为止所解释的内容中提取以下要点：

+   栈内存的大小有限；因此，它不是存储大对象的好地方。

+   栈段的地址向后增长，因此，在栈内存中向前读取意味着读取已经推入的字节。

+   栈具有自动内存管理，包括分配和释放。

+   每个栈变量都有一个作用域，它决定了其生命周期。你应该根据这个生命周期来设计你的逻辑。你无法控制它。

+   指针应该只指向那些仍然在作用域内的栈变量。

+   当作用域即将结束时，栈变量的内存释放是自动完成的，你无法控制它。

+   只有当我们确信当前作用域在调用函数中的代码即将使用该指针时仍然存在时，才能将指向当前作用域内变量的指针作为参数传递给其他函数。在具有并发逻辑的情况下，这种条件可能会被打破。

在下一节中，我们将讨论堆段及其各种特性。

# 堆

几乎任何编程语言编写的代码都会以某种方式使用堆内存。这是因为堆有一些独特的优势，这些优势是使用栈无法实现的。

另一方面，它也有一些缺点；例如，与栈内存中的类似区域相比，分配堆内存区域要慢。

在本节中，我们将更详细地讨论堆本身以及在使用堆内存时应注意的指南。

堆内存之所以重要，是因为它具有独特的属性。并非所有这些属性都是有益的，事实上，其中一些可以被视为应该减轻的风险。一个伟大的工具总有优点和缺点，如果你要正确使用它，你必须非常了解这两方面。

在这里，我们将列出这些特性，并看看哪些是有益的，哪些是有风险的：

1.  **堆中没有自动分配的内存块**。相反，程序员必须使用`malloc`或类似函数逐个获取堆内存块。实际上，这可以被视为栈内存的弱点，而堆内存则解决了这个问题。栈内存可以包含栈帧，这些栈帧不是由程序员分配和推送的，而是由于函数调用以自动方式产生的。

1.  **堆具有很大的内存大小**。虽然栈的大小有限，并且不适合存储大对象，但堆允许存储非常大的对象，甚至可以存储数十个 GB 大小的对象。随着堆大小的增长，分配器需要从操作系统请求更多的堆页面，堆内存块在这些页面之间分散。请注意，与栈段不同，堆内存中的分配地址是向前移动到更大的地址。

1.  **堆内存中的内存分配和释放由程序员管理**。这意味着程序员是唯一负责分配内存并在不再需要时释放它的实体。在许多现代编程语言中，释放分配的堆块是由一个称为垃圾回收器的并行组件自动完成的。但在 C 和 C++中，我们没有这样的概念，释放堆块应该手动完成。这确实是一种风险，C/C++程序员在使用堆内存时应该非常小心。未能释放分配的堆块通常会导致**内存泄漏**，这在大多数情况下都是致命的。

1.  **从堆中分配的变量没有作用域**，这与栈中的变量不同。

1.  这是一种风险，因为它使得内存管理变得更加困难。你不知道何时需要释放变量，你必须提出一些新的定义来有效地进行内存管理，包括内存块的**作用域**和**所有者**。一些这些方法将在接下来的章节中介绍。

1.  **我们只能使用指针来访问堆内存块**。换句话说，没有所谓的堆变量。堆区域是通过指针来访问的。

1.  **由于堆段对其所有者进程是私有的，我们需要使用调试器来探测它**。幸运的是，C 指针与堆内存块的工作方式与与栈内存块的工作方式完全相同。C 在这方面做得很好，因此我们可以使用相同的指针来访问这两种内存。因此，我们可以使用检查栈的方法来探测堆内存。

在下一节中，我们将讨论如何分配和释放堆内存块。

## 堆内存分配和释放

正如我们在上一节中说的，堆内存应该手动获取和释放。这意味着程序员应该使用一组函数或 API（C 标准库的内存分配函数）来在堆中分配或释放内存块。

这些函数确实存在，并且它们在头文件`stdlib.h`中定义。用于获取堆内存块的函数有`malloc`、`calloc`和`realloc`，而用于释放堆内存块的唯一函数是`free`。*示例 5.3* 展示了如何使用这些函数中的一些。

**注意**：

在某些文本中，动态内存被用来指代堆内存。*动态内存分配*是堆内存分配的同义词。

以下代码框显示了 *示例 5.3* 的源代码。它分配了两个堆内存块，然后打印出其内存映射：

```cpp
#include <stdio.h>  // For printf function
#include <stdlib.h> // For C library's heap memory functions
void print_mem_maps() {
#ifdef __linux__
  FILE* fd = fopen("/proc/self/maps", "r");
  if (!fd) {
    printf("Could not open maps file.\n");
    exit(1);
  }
  char line[1024];
  while (!feof(fd)) {
    fgets(line, 1024, fd);
    printf("> %s", line);
  }
  fclose(fd);
#endif
}
int main(int argc, char** argv) {
  // Allocate 10 bytes without initialization
  char* ptr1 = (char*)malloc(10 * sizeof(char));
  printf("Address of ptr1: %p\n", (void*)&ptr1);
  printf("Memory allocated by malloc at %p: ", (void*)ptr1);
  for (int i = 0; i < 10; i++) {
    printf("0x%02x ", (unsigned char)ptr1[i]);
  }
  printf("\n");
  // Allocation 10 bytes all initialized to zero
  char* ptr2 = (char*)calloc(10, sizeof(char));
  printf("Address of ptr2: %p\n", (void*)&ptr2);
  printf("Memory allocated by calloc at %p: ", (void*)ptr2);
  for (int i = 0; i < 10; i++) {
    printf("0x%02x ", (unsigned char)ptr2[i]);
  }
  printf("\n");
  print_mem_maps();
  free(ptr1);
  free(ptr2);
  return 0;
}
```

代码框 5-6 [ExtremeC_examples_chapter5_3.c]: 分配两个堆内存块后的内存映射示例 5.3

前面的代码是跨平台的，您可以在大多数类 Unix 操作系统上编译它。但是，`print_mem_maps`函数仅在 Linux 上工作，因为`__linux__`宏仅在 Linux 环境中定义。因此，在 macOS 上，您可以编译代码，但`print_mem_maps`函数不会做任何事情。

以下 shell 框是 Linux 环境中运行示例的结果：

```cpp
$ gcc ExtremeC_examples_chapter5_3.c -o ex5_3.out
$ ./ex5_3.out
Address of ptr1: 0x7ffe0ad75c38
Memory allocated by malloc at 0x564c03977260: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 
Address of ptr2: 0x7ffe0ad75c40
Memory allocated by calloc at 0x564c03977690: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 
> 564c01978000-564c01979000 r-xp 00000000 08:01 5898436                    /home/kamranamini/extreme_c/5.3/ex5_3.out
> 564c01b79000-564c01b7a000 r--p 00001000 08:01 5898436                    /home/kamranamini/extreme_c/5.3/ex5_3.out
> 564c01b7a000-564c01b7b000 rw-p 00002000 08:01 5898436                    /home/kamranamini/extreme_c/5.3/ex5_3.out
> 564c03977000-564c03998000 rw-p 00000000 00:00 0           [heap]
> 7f31978ec000-7f3197ad3000 r-xp 00000000 08:01 5247803     /lib/x86_64-linux-gnu/libc-2.27.so
...
> 7f3197eef000-7f3197ef1000 rw-p 00000000 00:00 0 
> 7f3197f04000-7f3197f05000 r--p 00027000 08:01 5247775     /lib/x86_64-linux-gnu/ld-2.27.so
> 7f3197f05000-7f3197f06000 rw-p 00028000 08:01 5247775     /lib/x86_64-linux-gnu/ld-2.27.so
> 7f3197f06000-7f3197f07000 rw-p 00000000 00:00 0 
> 7ffe0ad57000-7ffe0ad78000 rw-p 00000000 00:00 0           [stack]
> 7ffe0adc2000-7ffe0adc5000 r--p 00000000 00:00 0           [vvar]
> 7ffe0adc5000-7ffe0adc7000 r-xp 00000000 00:00 0           [vdso]
> ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0   [vsyscall]
$
```

shell 框 5-17：Linux 环境中示例 5.3 的输出

前面的输出有很多要说的。程序打印了指针`ptr1`和`ptr2`的地址。如果您在打印的内存映射中找到栈段的内存映射，您会看到栈区域从`0x7ffe0ad57000`开始，到`0x7ffe0ad78000`结束。指针位于这个范围内。

这意味着指针是从栈中分配的，但它们指向栈段之外的一个内存区域，在这种情况下，是堆段。使用栈指针来访问堆内存块是非常常见的。

请记住，`ptr1`和`ptr2`指针具有相同的范围，它们将在`main`函数返回时被释放，但堆内存块没有范围。它们将保留分配状态，直到程序手动释放它们。您可以看到，在从`main`函数返回之前，使用指向它们的指针和`free`函数释放了两个内存块。

关于上述示例的进一步说明，我们可以看到`malloc`和`calloc`函数返回的地址位于堆段内部。这可以通过比较返回的地址和描述为`[heap]`的内存映射来调查。标记为堆的区域从`0x564c03977000`开始，到`0x564c03998000`结束。`ptr1`指针指向地址`0x564c03977260`，而`ptr2`指针指向地址`0x564c03977690`，它们都在堆区域内部。

关于堆分配函数，正如它们的名称所暗示的，`calloc`代表**清除并分配**，而`malloc`代表**内存分配**。这意味着`calloc`在分配后清除内存块，而`malloc`则将其保留为未初始化状态，直到程序在必要时自行初始化。

**注意**：

在 C++中，`new`和`delete`关键字分别与`malloc`和`free`相同。此外，新操作符从操作数类型推断分配的内存块的大小，并自动将返回的指针转换为操作数类型。

但如果你查看两个分配的块中的字节，它们都有零字节。所以，看起来`malloc`在分配后也初始化了内存块。但根据 C 规范中`malloc`的描述，`malloc`不会初始化分配的内存块。那么，这是为什么？为了进一步探讨，让我们在 macOS 环境中运行示例：

```cpp
$ clang ExtremeC_examples_chapter5_3.c -o ex5_3.out
$ ./ ex5_3.out
Address of ptr1: 0x7ffee66b2888
Memory allocated by malloc at 0x7fc628c00370: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x80 0x00 0x00
Address of ptr2: 0x7ffee66b2878
Memory allocated by calloc at 0x7fc628c02740: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
$
```

Shell 框 5-18：在 macOS 上示例 5.3 的输出

如果你仔细观察，你可以看到`malloc`分配的内存块中有一些非零字节，但`calloc`分配的内存块全部为零。那么，我们应该怎么做？我们应该假设 Linux 中`malloc`分配的内存块总是零吗？

如果你打算编写一个跨平台程序，始终要与 C 规范保持一致。规范说明`malloc`不会初始化分配的内存块。

即使你只为 Linux 编写程序，而不是为其他操作系统编写，也要注意未来的编译器可能会有不同的行为。因此，根据 C 规范，我们必须始终假设由`malloc`分配的内存块未初始化，如果需要，应手动初始化。

注意，由于`malloc`不会初始化分配的内存，它通常比`calloc`更快。在某些实现中，`malloc`实际上不会分配内存块，而是在内存块被访问（无论是读取还是写入）时才延迟分配。这样，内存分配会更快。

如果你打算在`malloc`之后初始化内存，可以使用`memset`函数。以下是一个示例：

```cpp
#include <stdlib.h> // For malloc
#include <string.h> // For memset
int main(int argc, char** argv) {
  char* ptr = (char*)malloc(16 * sizeof(char));
  memset(ptr, 0, 16 * sizeof(char));    // Fill with 0
  memset(ptr, 0xff, 16 * sizeof(char)); // Fill with 0xff
  ...
  free(ptr);
  return 0;
}
```

代码框 5-7：使用`memset`函数初始化内存块

`realloc`函数是作为堆分配函数的一部分引入的另一个函数。它没有在*example 5.3*中使用。实际上，它通过调整已分配内存块的大小来重新分配内存。以下是一个例子：

```cpp
int main(int argc, char** argv) {
  char* ptr = (char*)malloc(16 * sizeof(char));
  ...
  ptr = (char*)realloc(32 * sizeof(char));
  ...
  free(ptr);
  return 0;
}
```

代码框 5-8：使用 realloc 函数改变已分配块的尺寸

`realloc`函数不会更改旧块中的数据，而只是将已分配的块扩展到新块。如果由于*碎片化*无法扩展当前分配的块，它将找到另一个足够大的块，并将旧块中的数据复制到新块中。在这种情况下，它也会释放旧块。正如你所看到的，在某些情况下，重新分配并不是一个便宜的操作，因为它涉及许多步骤，因此应该谨慎使用。

关于*example 5.3*的最后一点是关于`free`函数的。实际上，它通过传递块地址作为指针来释放已经分配的堆内存块。正如之前所说，任何已分配的堆块在不再需要时都应该被释放。未能这样做会导致*内存泄漏*。使用一个新的例子，*example 5.4*，我们将向您展示如何使用`valgrind`工具检测内存泄漏。

让我们先在*example 5.4*中产生一些内存泄漏：

```cpp
#include <stdlib.h> // For heap memory functions
int main(int argc, char** argv) {
  char* ptr = (char*)malloc(16 * sizeof(char));
  return 0;
}
```

代码框 5-9：在从 main 函数返回时未释放分配的块产生内存泄漏

前一个程序存在内存泄漏，因为当程序结束时，我们分配了`16`字节堆内存但没有释放。这个例子非常简单，但当源代码增长并且涉及更多组件时，通过肉眼检测它就会变得非常困难，甚至不可能。

内存分析器是有用的程序，可以检测运行中的进程中的内存问题。著名的`valgrind`工具是最为人所知的之一。

为了使用`valgrind`分析*example 5.4*，首先我们需要使用调试选项`-g`构建示例。然后，我们应该使用`valgrind`运行它。在运行给定的可执行目标文件时，`valgrind`记录所有的内存分配和释放。最后，当执行完成或发生崩溃时，`valgrind`会打印出分配和释放的摘要以及未释放的内存量。这样，它可以让你知道在给定程序的执行过程中产生了多少内存泄漏。

下面的 shell box 演示了如何编译和使用`valgrind`来分析*example 5.4*：

```cpp
$ gcc -g ExtremeC_examples_chapter5_4.c -o ex5_4.out
$ valgrind ./ex5_4.out
==12022== Memcheck, a memory error detector
==12022== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==12022== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==12022== Command: ./ex5_4.out
==12022== 
==12022== 
==12022== HEAP SUMMARY:
==12022==     in use at exit: 16 bytes in 1 blocks
==12022==   total heap usage: 1 allocs, 0 frees, 16 bytes allocated
==12022== 
==12022== LEAK SUMMARY:
==12022==    definitely lost: 16 bytes in 1 blocks
==12022==    indirectly lost: 0 bytes in 0 blocks
==12022==      possibly lost: 0 bytes in 0 blocks
==12022==    still reachable: 0 bytes in 0 blocks
==12022==         suppressed: 0 bytes in 0 blocks
==12022== Rerun with --leak-chck=full to see details of leaked memory
==12022== 
==12022== For counts of detected and suppressed errors, rerun with: -v
==12022== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
$
```

Shell Box 5-19：valgrind 输出的输出，显示了作为 example 5.4 执行一部分的 16 字节内存泄漏

如果你查看*Shell Box 5-19*中的`HEAP SUMMARY`部分，你可以看到我们进行了`1`次分配和`0`次释放，并且在退出时还保留了`16`字节的分配。如果你向下滚动一点到`LEAK SUMMARY`部分，它指出`16`字节肯定丢失了，这意味着存在内存泄漏！

如果您想确切知道提到的泄漏内存块是在哪一行分配的，您可以使用专门为此设计的`valgrind`特殊选项。在下面的 shell 框中，您将看到如何使用`valgrind`找到实际分配的责任行：

```cpp
$ gcc -g ExtremeC_examples_chapter5_4.c -o ex5_4.out
$ valgrind --leak-check=full ./ex5_4.out
==12144== Memcheck, a memory error detector
==12144== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==12144== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==12144== Command: ./ex5_4.out
==12144== 
==12144== 
==12144== HEAP SUMMARY:
==12144==     in use at exit: 16 bytes in 1 blocks
==12144==   total heap usage: 1 allocs, 0 frees, 16 bytes allocated
==12144== 
==12144== 16 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12144==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12144==    by 0x108662: main (ExtremeC_examples_chapter5_4.c:4)
==12144== 
==12144== LEAK SUMMARY:
==12144==    definitely lost: 16 bytes in 1 blocks
==12144==    indirectly lost: 0 bytes in 0 blocks
==12144==      possibly lost: 0 bytes in 0 blocks
==12144==    still reachable: 0 bytes in 0 blocks
==12144==         suppressed: 0 bytes in 0 blocks
==12144== 
==12144== For counts of detected and suppressed errors, rerun with : -v
==12144== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
$
```

Shell Box 5-20：valgrind 输出的实际分配责任行的输出

如您所见，我们已经将`--leak-check=full`选项传递给了`valgrind`，现在它显示了负责泄漏堆内存的代码行。它清楚地显示了*代码框 5-9*中的第 4 行，这是一个`malloc`调用，泄漏的堆块就是在这里分配的。这可以帮助您进一步追踪并找到应该释放提到的泄漏块的正确位置。

好的，让我们修改前面的示例，使其释放分配的内存。我们只需要在`return`语句之前添加`free(ptr)`指令，就像我们在这里看到的那样：

```cpp
#include <stdlib.h> // For heap memory functions
int main(int argc, char** argv) {
  char* ptr = (char*)malloc(16 * sizeof(char));
  free(ptr);
  return 0;
}
```

代码框 5-10：作为示例 5.4 的一部分释放分配的内存块

现在经过这个修改，唯一的分配堆块已经被释放。让我们再次构建并运行`valgrind`：

```cpp
$ gcc -g ExtremeC_examples_chapter5_4.c -o ex5_4.out
$ valgrind --leak-check=full ./ex5_4.out
==12175== Memcheck, a memory error detector
==12175== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==12175== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==12175== Command: ./ex5_4.out
==12175== 
==12175== 
==12175== HEAP SUMMARY:
==12175==     in use at exit: 0 bytes in 0 blocks
==12175==   total heap usage: 1 allocs, 1 frees, 16 bytes allocated
==12175== 
==12175== All heap blocks were freed -- no leaks are possible
==12175== 
==12175== For counts of detected and suppressed errors, rerun with  -v
==12175== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
$
```

Shell Box 5-20：释放分配的内存块后的 valgrind 输出

如您所见，`valgrind`表示“所有堆块都已释放”，这实际上意味着我们的程序中没有进一步的内存泄漏。使用`valgrind`运行程序可能会将它们的速度降低 10 到 50 倍，但它可以帮助您非常容易地发现内存问题。让您的程序在内存分析器中运行并尽快捕获内存泄漏是一个好习惯。

内存泄漏可以被视为*技术债务*，如果您有一个导致泄漏的糟糕设计，或者作为*风险*，我们知道我们有一个泄漏，但我们不知道如果泄漏继续增长会发生什么。但在我看来，它们应该被视为*错误*；否则，您需要一段时间才能回顾它们。通常，在团队中，内存泄漏被视为应该尽快修复的错误。

除了`valgrind`之外，还有其他内存分析器。**LLVM Address Sanitizer**（或**ASAN**）和**MemProf**也是其他知名的内存分析器。内存分析器可以使用各种方法来分析内存使用和分配。接下来，我们将讨论其中的一些：

+   一些分析器可以像沙盒一样运行，在沙盒内运行目标程序并监控所有内存活动。我们已经使用这种方法在`valgrind`沙盒中运行了*示例 5.4*。这种方法不需要您重新编译代码。

+   另一种方法是使用一些内存分析器提供的库，这些库包装了内存相关的系统调用。这样，最终的二进制文件将包含用于分析任务的全部逻辑。

    `valgrind`和 ASAN 可以作为内存分析库链接到最终的可执行对象文件。这种方法需要重新编译你的目标源代码，甚至需要对源代码进行一些修改。

+   程序也可以*预加载*不同的库，而不是默认的 C 标准库，这些库包含 C 库标准内存分配函数的*函数替换*。这样，你不需要编译你的目标源代码。你只需要在`LD_PRELOAD`环境变量中指定这样的分析库，以便预加载，而不是默认的`libc`库。`MemProf`使用这种方法。

**注意**：

*函数替换*是在目标动态库之前加载的动态库中定义的包装函数，它将调用传播到目标函数。可以使用`LD_PRELOAD`环境变量预加载动态库。

## 堆内存原则

如前所述，堆内存与栈内存有几个不同之处。因此，堆内存有自己的内存管理指南。在本节中，我们将关注这些差异，并提出一些我们在处理堆空间时应考虑的“应该做”和“不应该做”的事项。

栈中的每个内存块（或变量）都有一个作用域。因此，根据其作用域定义内存块的生存期是一个简单的任务。每次我们超出作用域时，该作用域中的所有变量都会消失。但是这与堆内存不同，并且更加复杂。

堆内存块没有作用域，因此其生存期不明确，应该重新定义。这就是为什么在像 Java 这样的现代语言中，有手动释放或*代际* *垃圾回收*的原因。堆的生存期不能由程序本身或使用的 C 库来确定，程序员是唯一定义堆内存块生存期的人。

当讨论到程序员的决策时，尤其是在这种情况下，情况变得复杂，很难提出一个通用的银弹解决方案。每个观点都是可辩论的，并且可能导致权衡。

解决堆生存期复杂性的最佳策略之一，当然这不是一个完整的解决方案，是为内存块定义一个*所有者*，而不是让作用域包含内存块。

所有者是唯一负责管理堆内存块生存期的实体，也是最初分配该块并在不再需要时释放它的人。

有许多经典的例子展示了如何使用这种策略。大多数知名的 C 库都使用这种策略来处理它们的堆内存分配。*示例 5.5* 是这种方法的一个非常简单的实现，用于管理用 C 编写的队列对象的生存期。下面的代码框试图展示*所有权*策略：

```cpp
#include <stdio.h> // For printf function
#include <stdlib.h> // For heap memory functions
#define QUEUE_MAX_SIZE 100
typedef struct {
  int front;
  int rear;
  double* arr;
} queue_t;
void init(queue_t* q) {
  q->front = q->rear = 0;
  // The heap memory block allocated here is owned
  // by the queue object.
  q->arr = (double*)malloc(QUEUE_MAX_SIZE * sizeof(double));
}
void destroy(queue_t* q) {
  free(q->arr);
}
int size(queue_t* q) {
  return q->rear - q->front;
}
void enqueue(queue_t* q, double item) {
  q->arr[q->rear] = item;
  q->rear++;
}
double dequeue(queue_t* q) {
  double item = q->arr[q->front];
  q->front++;
  return item;
}
int main(int argc, char** argv) {
  // The heap memory block allocated here is owned
  // by the function main
  queue_t* q = (queue_t*)malloc(sizeof(queue_t));
  // Allocate needed memory for the queue object
  init(q);
  enqueue(q, 6.5);
  enqueue(q, 1.3);
  enqueue(q, 2.4);
  printf("%f\n", dequeue(q));
  printf("%f\n", dequeue(q));
  printf("%f\n", dequeue(q));
  // Release resources acquired by the queue object
  destroy(q);
  // Free the memory allocated for the queue object
  // acquired by the function main
  free(q);
  return 0;
}
```

代码框 5-11 [ExtremeC_examples_chapter5_5.c]：演示堆生命周期管理所有权策略的例子 5.5

前面的例子包含两种不同的所有权，每种所有权都拥有一个特定的对象。第一种所有权是关于由`queue_t`结构中的`arr`指针指向的堆内存块，该内存块由队列对象拥有。只要队列对象存在，这个内存块就必须保持在原地并分配。

第二种所有权是关于由`main`函数获取的堆内存块，作为队列对象`q`的占位符，该对象由`main`函数本身拥有。区分队列对象拥有的堆内存块和`main`函数拥有的堆内存块非常重要，因为释放其中一个并不会释放另一个。

为了演示在前面代码中内存泄漏是如何发生的，假设你忘记在队列对象上调用`destroy`函数。这肯定会引起内存泄漏，因为`init`函数内部获取的堆内存块仍然会被分配，而不会被释放。

注意，如果一个实体（一个对象、函数等）拥有一个堆内存块，应该在注释中表达出来。如果没有拥有该内存块，则不应该有任何东西释放堆内存块。

注意，对同一堆内存块进行多次释放会导致*双重释放*的情况。双重释放是一个内存损坏问题，就像任何其他内存损坏问题一样，应该在检测到后尽快处理和解决。否则，它可能会产生严重的后果，如突然崩溃。

除了所有权策略之外，还可以使用垃圾回收器。垃圾回收器是一种嵌入到程序中的自动机制，它试图收集没有任何指针指向它们的内存块。C 语言中一个老牌且广为人知的垃圾回收器是*Boehm-Demers-Weiser 保守垃圾回收器*，它提供了一组内存分配函数，应该代替`malloc`和其他标准 C 内存分配函数来调用。

**进一步阅读**：

更多关于 Boehm-Demers-Weiser 垃圾回收器的信息可以在这里找到：[`www.hboehm.info/gc/`](http://www.hboehm.info/gc/)。

管理堆块生命周期的另一种技术是使用 RAII 对象。**RAII**代表**资源获取即初始化**。这意味着我们可以将资源的生命周期（可能是一个堆分配的内存块）绑定到对象的生命周期。换句话说，我们使用一个对象，在它的构造时初始化资源，在它的销毁时释放资源。不幸的是，这种技术在 C 语言中不能使用，因为我们没有得到关于对象销毁的通知。但在 C++中，使用析构函数，这种技术可以有效地使用。在 RAII 对象中，资源初始化发生在构造函数中，而用于反初始化资源的代码被放入析构函数中。请注意，在 C++中，当对象超出作用域或被删除时，析构函数会自动调用。

作为结论，当与堆内存一起工作时，以下指南非常重要：

+   堆内存分配不是免费的，它有自己的成本。并非所有内存分配函数的成本相同，通常`malloc`是最便宜的。

+   从堆空间分配的所有内存块都必须在不再需要时立即释放，或者在程序结束前释放。

+   由于堆内存块没有作用域，程序必须能够管理内存，以避免任何可能的泄漏。

+   对于每个堆内存块坚持使用所选的内存管理策略似乎是必要的。

+   所选策略及其假设应该在代码中记录下来，无论在何处访问该块，以便未来的程序员了解它。

+   在像 C++这样的某些编程语言中，我们可以使用 RAII 对象来管理资源，可能是一个堆内存块。

到目前为止，我们假设我们有足够的内存来存储大对象并运行任何类型的程序。但在下一节中，我们将对可用的内存施加一些限制，并讨论内存低或增加额外内存存储（在金钱、时间、性能等方面）成本高的环境。在这种情况下，我们需要以最有效的方式使用可用的内存。

# 受限环境中的内存管理

在某些环境中，内存是一种宝贵的资源，而且通常有限。也有其他环境，其中性能是一个关键因素，程序应该快速运行，不管我们有多少内存。关于内存管理，每个这样的环境都需要特定的技术来克服内存短缺和性能下降。首先，我们需要知道什么是受限环境。

限制性环境不一定具有低内存容量。通常有一些*限制*会限制程序的内存使用。这些限制可以是客户对内存使用的硬性限制，也可能是由于提供低内存容量的硬件，或者可能是由于不支持更大内存的操作系统（例如，MS-DOS）。

即使没有限制或硬件限制，我们作为程序员也会尽力使用尽可能少的内存，并以最优的方式使用它。内存消耗是一个项目中关键的*非功能性需求*之一，应该被仔细监控和调整。

在本节中，我们将首先介绍在低内存环境中用于克服短缺问题的技术，然后我们将讨论在性能环境中通常使用的内存技术，以提升运行程序的性能。

## 内存受限环境

在这些环境中，有限的内存总是个约束，算法应该设计成能够应对内存短缺。具有数十到数百兆字节内存大小的嵌入式系统通常属于这一类。关于这种环境下的内存管理有一些小贴士，但它们都不如有一个调校得很好的算法来得有效。在这种情况下，通常使用内存复杂度低的算法。这些算法通常具有更高的*时间复杂度*，需要与它们的低内存使用进行权衡。

为了更详细地说明这一点，每个算法都有特定的*时间*和*内存*复杂度。时间复杂度描述了输入大小与算法完成所需时间之间的关系。同样，内存复杂度描述了输入大小与算法完成任务所消耗的内存之间的关系。这些复杂度通常用*大 O 函数*表示，我们不想在本节中处理这些。我们的讨论是定性的，因此我们不需要任何数学来讨论内存受限环境。

一个算法理想情况下应该具有低时间复杂度和低内存复杂度。换句话说，拥有一个快速且内存消耗低的算法是非常理想的，但这种“两者兼得”的情况很少见。同时，一个内存消耗高但性能不佳的算法也是令人意外的。

大多数时候，我们在内存和速度之间进行权衡，这代表了时间。例如，一个比另一个算法更快的排序算法通常会消耗比另一个更多的内存，尽管这两个算法都完成了相同的工作。

在编写程序时，即使我们知道最终的生产环境将有足够的内存，假设我们正在为内存受限的系统编写代码，这是一种很好的但保守的做法。我们做出这个假设是因为我们希望减轻过度消耗内存的风险。

注意，推动这个假设的动力应根据对最终设置中平均内存可用性的相当准确的估计进行控制和调整，包括大小。为内存受限环境设计的算法本质上较慢，您应该小心这个陷阱。

在接下来的章节中，我们将介绍一些可以帮助我们收集一些浪费的内存或在使用内存受限环境中使用更少内存的技术。

### 压缩结构

使用压缩结构是减少内存消耗的最简单方法之一。压缩结构放弃了内存对齐，并且它们有更紧凑的内存布局来存储它们的字段。

使用压缩结构实际上是一种权衡。你消耗更少的内存，因为你放弃了内存对齐，最终在加载结构变量时会有更多的内存读取时间。这将导致程序运行速度变慢。

这种方法简单，但不适用于所有程序。有关此方法的更多信息，您可以阅读*第一章*，*基本特性*中找到的*结构*部分。

### 压缩

这是一种有效的技术，尤其是对于需要在内存中保留的大量文本数据的程序。与二进制数据相比，文本数据具有很高的*压缩比*。这种技术允许程序存储压缩形式而不是实际的文本数据，从而获得巨大的内存回报。

然而，节省内存并非没有代价；由于压缩算法是*CPU 密集型*的，程序最终的性能会变差。这种方法对于需要保存不常使用的文本数据的程序来说很理想；否则，需要大量的压缩/解压缩操作，程序最终几乎无法使用。

### 外部数据存储

使用网络服务、云基础设施或简单的硬盘作为外部数据存储形式，是一种非常常见且有用的技术，用于解决内存不足的问题。由于通常认为程序可能在有限的或内存较低的环境中运行，因此有很多示例使用这种方法，即使在有足够内存的环境中也能消耗更少的内存。

这种技术通常假设内存不是主要存储，而是作为*缓存*内存。另一个假设是我们不能将所有数据都保存在内存中，在任何时刻，只能将部分数据或一个*页面*的数据加载到内存中。

这些算法并不是直接解决低内存问题，而是在尝试解决另一个问题：慢速的外部数据存储。与主内存相比，外部数据存储总是太慢。因此，算法应该平衡从外部数据存储的读取和它们的内部内存。所有数据库服务，如 PostgreSQL 和 Oracle，都使用这种技术。

在大多数项目中，从头开始设计和编写这些算法并不是一个非常明智的选择，因为这些算法并不那么简单和容易编写。SQLite 等著名库背后的团队已经修复了多年的错误。

如果你需要在具有低内存占用的情况下访问外部数据存储，如文件、数据库或网络上的主机，总有适合你的选择。

## 性能环境

如我们在前几节关于算法的时间和内存复杂度的解释中所述，通常期望在想要获得更快的算法时消耗更多的内存。因此，在本节中，我们期望为了提高性能而消耗更多的内存。

这个陈述的一个直观例子是使用缓存来提高性能。缓存数据意味着消耗更多的内存，但如果缓存使用得当，我们预计可以获得更好的性能。

但增加额外的内存并不总是提高性能的最佳方式。还有其他直接或间接与内存相关的方法，可以对算法的性能产生重大影响。在跳到这些方法之前，让我们先谈谈缓存。

### 缓存

缓存是计算机系统中涉及不同读写速度的两个数据存储时使用的所有类似技术的通用术语。例如，CPU 有几个内部寄存器，在读写操作方面速度很快。此外，CPU 还需要从主内存中获取数据，其速度比寄存器慢得多。这里需要一个缓存机制；否则，主内存的较低速度将占主导地位，并掩盖 CPU 的高计算速度。

与数据库文件一起工作是另一个例子。数据库文件通常存储在外部硬盘上，其速度比主内存慢得多。毫无疑问，这里需要一个缓存机制；否则，最慢的速度将占主导地位，并决定整个系统的速度。

缓存及其相关细节值得有一个专门的章节，因为这里有一些抽象模型和特定的术语需要解释。

使用这些模型，可以预测缓存的表现如何以及引入缓存后可以期望获得多少*性能提升*。在这里，我们试图以简单直观的方式解释缓存。

假设你有一种可以包含许多项目的慢速存储。你还有一个快速的存储，但它只能包含有限数量的项目。这是一个明显的权衡。我们可以将更快但更小的存储称为*缓存*。如果你将项目从慢速存储带入快速存储并在此处理它们，这将是合理的，因为这样可以更快。

不时地，你必须前往慢速存储以带来更多的项目。显然，你不会只从慢速存储中带来一个项目，因为这会非常低效。相反，你将一桶项目带入更快的存储中。通常，人们会说项目被缓存到更快的存储中。

假设你正在处理一个需要从慢速存储中加载其他项目的项目。首先想到的事情是在当前缓存存储中的最近带来的桶内搜索所需的项目。

如果你能在缓存中找到项目，就没有必要从慢速存储中检索它，这被称为*命中*。如果项目缺失于缓存存储中，你必须前往慢速存储并读取另一个桶中的项目到缓存内存中。这被称为*未命中*。很明显，你观察到的命中越多，你获得的表现就越好。

上述描述可以应用于 CPU 缓存和主存储。CPU 缓存存储从主存储中读取的最近指令和数据，而与 CPU 缓存内存相比，主存储较慢。

在下一节中，我们讨论缓存友好代码，并观察为什么缓存友好代码可以由 CPU 更快地执行。

#### 缓存友好代码

当 CPU 执行指令时，它必须首先获取所有所需的数据。数据位于由指令确定的特定地址的主存储中。

在进行任何计算之前，数据必须被传输到 CPU 寄存器。但 CPU 通常携带比预期要获取的更多块，并将它们放入其缓存中。

下次，如果需要在先前地址的*附近*的值，它应该存在于缓存中，CPU 可以使用缓存而不是主存储，这比从主存储中读取要快得多。正如我们在上一节中解释的，这被称为*缓存命中*。如果地址未在 CPU 缓存中找到，则称为*缓存未命中*，CPU 必须访问主存储以读取目标地址并带来所需的数据，这相当慢。一般来说，更高的命中率会导致更快的执行。

但为什么 CPU 会从地址周围的相邻地址（邻近性）中获取数据？这是因为 *局部性原理*。在计算机系统中，通常观察到位于相同区域的数据被更频繁地访问。因此，CPU 根据这个原理行事，并从局部引用中获取更多数据。如果一个算法可以利用这种行为，它就可以被 CPU 更快地执行。这就是我们为什么称这样的算法为 *缓存友好算法*。

*示例 5.6* 展示了缓存友好代码和非缓存友好代码性能的差异：

```cpp
#include <stdio.h>  // For printf function
#include <stdlib.h> // For heap memory functions
#include <string.h> // For strcmp function
void fill(int* matrix, int rows, int columns) {
  int counter = 1;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      *(matrix + i * columns + j) = counter;
    }
    counter++;
  }
}
void print_matrix(int* matrix, int rows, int columns) {
  int counter = 1;
  printf("Matrix:\n");
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%d ", *(matrix + i * columns + j));
    }
    printf("\n");
  }
}
void print_flat(int* matrix, int rows, int columns) {
  printf("Flat matrix: ");
  for (int i = 0; i < (rows * columns); i++) {
    printf("%d ", *(matrix + i));
  }
  printf("\n");
}
int friendly_sum(int* matrix, int rows, int columns) {
  int sum = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      sum += *(matrix + i * columns + j);
    }
  }
  return sum;
}
int not_friendly_sum(int* matrix, int rows, int columns) {
  int sum = 0;
  for (int j = 0; j < columns; j++) {
    for (int i = 0; i < rows; i++) {
      sum += *(matrix + i * columns + j);
    }
  }
  return sum;
}
int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s [print|friendly-sum|not-friendly-sum] ");
    printf("[number-of-rows] [number-of-columns]\n", argv[0]);
    exit(1);
  }
  char* operation = argv[1];
  int rows = atol(argv[2]);
  int columns = atol(argv[3]);
  int* matrix = (int*)malloc(rows * columns * sizeof(int));
  fill(matrix, rows, columns);
  if (strcmp(operation, "print") == 0) {
    print_matrix(matrix, rows, columns);
    print_flat(matrix, rows, columns);
  }
  else if (strcmp(operation, "friendly-sum") == 0) {
    int sum = friendly_sum(matrix, rows, columns);
    printf("Friendly sum: %d\n", sum);
  }
  else if (strcmp(operation, "not-friendly-sum") == 0) {
    int sum = not_friendly_sum(matrix, rows, columns);
    printf("Not friendly sum: %d\n", sum);
  }
  else {
    printf("FATAL: Not supported operation!\n");
    exit(1);
  }
  free(matrix);
  return 0;
}
```

代码框 5-12 [ExtremeC_examples_chapter5_6.c]: 示例 5.6 展示了缓存友好代码和非缓存友好代码的性能

前面的程序计算并打印矩阵中所有元素的总和，但它做的不仅仅是这些。

用户可以向此程序传递选项，这会改变其行为。假设我们想打印一个由 `fill` 函数编写的算法初始化的 2 行 3 列矩阵。用户必须传递 `print` 选项以及所需的行数和列数。接下来，你可以看到这些选项是如何传递给最终的可执行二进制文件的：

```cpp
$ gcc ExtremeC_examples_chapter5_6.c -o ex5_6.out
$ ./ex5_6.out print 2 3
Matrix:
1 1 1
2 2 2
Flat matrix: 1 1 1 2 2 2
$
```

Shell Box 5-21: 示例 5.6 的输出，显示一个 2 行 3 列的矩阵

输出包括矩阵的两个不同打印。第一个是矩阵的二维表示，第二个是相同矩阵的 *扁平* 表示。正如你所看到的，矩阵在内存中按 *行主序顺序* 存储。这意味着我们按行存储它。所以，如果 CPU 获取了某行的数据，那么该行中的所有元素很可能也会被获取。因此，我们最好按行主序而不是 *列主序* 进行求和。

如果你再次查看代码，你可以看到在 `friendly_sum` 函数中执行的是行主序求和，而在 `not_friendly_sum` 函数中执行的是列主序求和。接下来，我们可以比较执行 20,000 行和 20,000 列矩阵求和所需的时间。正如你所看到的，差异非常明显：

```cpp
$ time ./ex5_6.out friendly-sum 20000 20000
Friendly sum: 1585447424
real   0m5.192s
user   0m3.142s
sys    0m1.765s
$ time ./ex5_6.out not-friendly-sum 20000 20000
Not friendly sum: 1585447424
real   0m15.372s
user   0m14.031s
sys    0m0.791s
$
```

Shell Box 5-22: 列主序和行主序矩阵求和算法的时间差异演示

测量时间的差异大约是 10 秒！程序是在 macOS 机器上使用 `clang` 编译器编译的。这种差异意味着相同的逻辑，使用相同数量的内存，可能需要更长的时间——仅仅是通过选择不同的矩阵元素访问顺序！这个例子清楚地展示了缓存友好代码的影响。

**注意**：

`time` 工具在所有类 Unix 操作系统中都可用。它可以用来测量程序完成所需的时间。

在继续到下一个技术之前，我们应该更多地讨论一下分配和释放成本。

### 分配和释放成本

在这里，我们想特别谈谈堆内存分配和释放的成本。如果你意识到堆内存分配和释放操作既耗时又耗内存，并且通常很昂贵，特别是当你需要每秒多次分配和释放堆内存块时，这可能会让你感到有些惊讶。

与相对快速且分配本身不需要额外内存的栈分配不同，堆分配需要找到足够大小的空闲内存块，这可能很昂贵。

设计了许多用于内存分配和释放的算法，并且在分配和释放操作之间总是存在权衡。如果你想快速分配，就必须在分配算法中消耗更多的内存；反之，如果你想减少内存消耗，可以选择花费更多时间进行较慢的分配。

除了通过`malloc`和`free`函数提供的默认 C 标准库之外，还有其他用于 C 语言的内存分配器。这些内存分配器库包括`ptmalloc`、`tcmalloc`、`Haord`和`dlmalloc`。

在这里详细介绍所有分配器超出了本章的范围，但对你来说，亲自尝试它们并体验一下是一个好的实践。

这个无声问题的解决方案是什么？很简单：减少分配和释放的频率。在某些程序中，这可能看起来是不可能的，因为这些程序需要以高频率进行堆内存分配。这些程序通常分配一大块堆内存，并尝试自行管理它。这就像在堆内存的大块上又增加了一层分配和释放逻辑（可能比`malloc`和`free`的实现简单）。

还有另一种方法，即使用*内存池*。在我们结束本章之前，我们将简要解释这项技术。

### 内存池

正如我们在上一节中描述的，内存分配和释放是昂贵的。使用预先分配的固定大小堆内存块池是一种有效的方法来减少分配次数并提高一些性能。池中的每个块通常都有一个标识符，可以通过为池管理设计的 API 获取。此外，当不再需要时，块也可以被释放。由于分配的内存量几乎保持不变，这对于希望在内存受限环境中具有确定性行为的算法来说是一个极佳的选择。

详细描述内存池超出了本书的范围；如果您想了解更多关于这个主题的信息，网上有许多有用的资源。

# 摘要

作为本章的一部分，我们主要介绍了栈和堆段以及它们的使用方式。之后，我们简要讨论了内存受限环境，并看到了缓存和内存池等技术如何提高性能。

在本章中：

+   我们讨论了用于探测栈和堆段的工具和技术。

+   我们介绍了调试器，并使用 `gdb` 作为我们的主要调试器来排查与内存相关的问题。

+   我们讨论了内存分析器，并使用 `valgrind` 来查找运行时发生的内存问题，如泄漏或悬垂指针。

+   我们比较了栈变量和堆块的生存期，并解释了如何判断此类内存块的生存期。

+   我们看到，栈变量的内存管理是自动的，但堆块的内存管理是完全手动的。

+   我们回顾了处理栈变量时常见的错误。

+   我们讨论了受限环境，并展示了在这些环境中如何进行内存调优。

+   我们讨论了高效的环境以及可以使用哪些技术来提高性能。

接下来的四章一起涵盖了 C 语言中的面向对象。乍一看，这似乎与 C 语言无关，但实际上，这是在 C 语言中编写面向对象代码的正确方式。作为这些章节的一部分，你将了解如何以面向对象的方式设计和解决问题，并且你将通过编写可读且正确的 C 代码获得指导。

下一章通过提供必要的理论讨论和示例来探讨所讨论的主题，涵盖了封装和面向对象编程的基础。
