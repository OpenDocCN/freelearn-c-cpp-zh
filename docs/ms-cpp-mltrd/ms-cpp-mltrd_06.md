# 第六章：调试多线程代码

理想情况下，自己的代码第一次就能正常工作，并且不包含等待崩溃应用程序、损坏数据或引起其他问题的隐藏错误。当然，这是不可能的。因此，开发了一些工具，使得检查和调试多线程应用程序变得容易。

在本章中，我们将研究其中一些内容，包括常规调试器以及 Valgrind 套件的一些工具，特别是 Helgrind 和 DRD。我们还将研究如何对多线程应用程序进行性能分析，以找出设计中的热点和潜在问题。

本章涵盖的主题包括以下内容：

+   介绍 Valgrind 工具套件

+   使用 Helgrind 和 DRD 工具

+   解释 Helgrind 和 DRD 分析结果

+   分析应用程序的性能，并分析结果

# 何时开始调试

理想情况下，每次达到某个里程碑时，无论是针对单个模块、多个模块还是整个应用程序，都应该测试和验证自己的代码。重要的是要确定自己的假设是否与最终功能相匹配。

特别是在多线程代码中，巧合的因素很大，特定的错误状态不能保证在每次运行应用程序时都会达到。实现不正确的多线程应用程序的迹象可能导致似乎随机崩溃的症状。

当应用程序崩溃时，留下核心转储时，可能会得到的第一个提示是，有些地方不正确。这是一个包含应用程序在崩溃时的内存内容的文件，包括堆栈。

这个核心转储可以以几乎与运行进程调试器相同的方式使用。它特别有用的是检查我们崩溃的代码位置以及线程。我们也可以通过这种方式检查内存内容。

处理多线程问题时最好的指标之一是应用程序从未在相同位置崩溃（不同的堆栈跟踪），或者总是在执行互斥操作的地方崩溃，例如操作全局数据结构。

首先，我们将首先更深入地研究使用调试器进行诊断和调试，然后再深入 Valgrind 工具套件。

# 谦逊的调试器

开发人员可能会有的所有问题中，“为什么我的应用程序刚刚崩溃？”这个问题可能是最重要的之一。这也是最容易用调试器回答的问题之一。无论是实时调试进程还是分析崩溃进程的核心转储，调试器都可以（希望）生成回溯，也称为堆栈跟踪。这个跟踪包含自应用程序启动以来调用的所有函数的时间顺序列表，就像它们在堆栈上找到的那样（有关堆栈工作原理的详细信息，请参见第二章，*处理器和操作系统上的多线程实现*）。

这个回溯的最后几个条目将告诉我们代码的哪个部分出了问题。如果调试信息被编译到二进制文件中，或者提供给调试器，我们还可以看到该行的代码以及变量的名称。

更好的是，因为我们正在查看堆栈帧，我们还可以检查该堆栈帧中的变量。这意味着传递给函数的参数以及任何局部变量及其值。

为了使调试信息（符号）可用，必须使用适当的编译器标志编译源代码。对于 GCC，可以选择一系列调试信息级别和类型。最常见的是使用`-g`标志，后面附加一个指定调试级别的整数，如下所示：

+   `-g0`：不生成调试信息（否定`-g`）

+   `-g1`：函数描述和外部变量的最小信息

+   `-g3`：包括宏定义在内的所有信息

此标志指示 GCC 以 OS 的本机格式生成调试信息。也可以使用不同的标志以特定格式生成调试信息；但是，这对于 GCC 的调试器（GDB）以及 Valgrind 工具并不是必需的。

GDB 和 Valgrind 都将使用这些调试信息。虽然在没有调试信息的情况下使用两者是技术上可能的，但最好留给真正绝望的时候去练习。

# GDB

用于基于 C 和 C++的代码的最常用的调试器之一是 GNU 调试器，简称 GDB。在下面的示例中，我们将使用这个调试器，因为它被广泛使用并且免费提供。最初于 1986 年编写，现在与各种编程语言一起使用，并且已成为个人和专业使用中最常用的调试器。

GDB 的最基本接口是命令行 shell，但它可以与图形前端一起使用，这些前端还包括一些 IDE，如 Qt Creator、Dev-C++和 Code::Blocks。这些前端和 IDE 可以使管理断点、设置监视变量和执行其他常见操作变得更加容易和直观。但是，它们的使用并不是必需的。

在 Linux 和 BSD 发行版上，gdb 可以轻松从软件包中安装，就像在 Windows 上使用 MSYS2 和类似的类 UNIX 环境一样。对于 OS X/MacOS，可能需要使用 Homebrew 等第三方软件包管理器安装 gdb。

由于在 MacOS 上通常不对 gdb 进行代码签名，因此它无法获得正常操作所需的系统级访问权限。在这里，可以以 root 身份运行 gdb（不建议），或者按照与您的 MacOS 版本相关的教程进行操作。

# 调试多线程代码

如前所述，有两种使用调试器的方法，一种是从调试器内启动应用程序（或附加到运行中的进程），另一种是加载核心转储文件。在调试会话中，可以中断运行进程（使用*Ctrl*+*C*发送`SIGINT`信号），或者加载加载的核心转储的调试符号。之后，我们可以检查此帧中的活动线程：

```cpp
Thread 1 received signal SIGINT, Interrupt.
0x00007fff8a3fff72 in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
(gdb) info threads
Id   Target Id         Frame 
* 1    Thread 0x1703 of process 72492 0x00007fff8a3fff72 in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
3    Thread 0x1a03 of process 72492 0x00007fff8a406efa in kevent_qos () from /usr/lib/system/libsystem_kernel.dylib
10   Thread 0x2063 of process 72492 0x00007fff8a3fff72 in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylibs
14   Thread 0x1e0f of process 72492 0x00007fff8a405d3e in __pselect () from /usr/lib/system/libsystem_kernel.dylib
(gdb) c
Continuing.

```

在前面的代码中，我们可以看到在向应用程序发送`SIGINT`信号后（在 OS X 上运行的基于 Qt 的应用程序），我们请求此时存在的所有线程的列表，以及它们的线程编号、ID 和它们当前正在执行的函数。根据后者的信息，这也清楚地显示了哪些线程可能正在等待，这在像这样的图形用户界面应用程序中经常发生。在这里，我们还看到当前活动的线程在应用程序中由其编号前的星号标记（线程 1）。

我们还可以使用`thread <ID>`命令随意在线程之间切换，并在线程的堆栈帧之间移动`up`和`down`。这使我们能够检查每个线程的每个方面。

当完整的调试信息可用时，通常还会看到线程正在执行的确切代码行。这意味着在应用程序的开发阶段，有尽可能多的调试信息是有意义的，以使调试变得更加容易。

# 断点

对于我们在第四章中查看的调度器代码，*线程同步和通信*，我们可以设置断点以允许我们检查活动线程：

```cpp
$ gdb dispatcher_demo.exe 
GNU gdb (GDB) 7.9 
Copyright (C) 2015 Free Software Foundation, Inc. 
Reading symbols from dispatcher_demo.exe...done. 
(gdb) break main.cpp:67 
Breakpoint 1 at 0x4017af: file main.cpp, line 67\. 
(gdb) run 
Starting program: dispatcher_demo.exe 
[New Thread 10264.0x2a90] 
[New Thread 10264.0x2bac] 
[New Thread 10264.0x2914] 
[New Thread 10264.0x1b80] 
[New Thread 10264.0x213c] 
[New Thread 10264.0x2228] 
[New Thread 10264.0x2338] 
[New Thread 10264.0x270c] 
[New Thread 10264.0x14ac] 
[New Thread 10264.0x24f8] 
[New Thread 10264.0x1a90] 

```

正如我们在上面的命令行输出中所看到的，我们使用应用程序的名称作为参数启动 GDB，这里是在 Windows 下的 Bash shell 中。之后，我们可以在这里设置一个断点，使用源文件的文件名和我们希望在其后中断的行号，然后运行应用程序。接着是由 GDB 报告的由调度程序创建的新线程的列表。

接下来，我们等待直到断点被触发：

```cpp
Breakpoint 1, main () at main.cpp:67 
67              this_thread::sleep_for(chrono::seconds(5)); 
(gdb) info threads 
Id   Target Id         Frame 
11   Thread 10264.0x1a90 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
10   Thread 10264.0x24f8 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
9    Thread 10264.0x14ac 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
8    Thread 10264.0x270c 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
7    Thread 10264.0x2338 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
6    Thread 10264.0x2228 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
5    Thread 10264.0x213c 0x00000000775ec2ea in ntdll!ZwWaitForMultipleObjects () from /c/Windows/SYSTEM32/ntdll.dll 
4    Thread 10264.0x1b80 0x0000000064942eaf in ?? () from /mingw64/bin/libwinpthread-1.dll 
3    Thread 10264.0x2914 0x00000000775c2385 in ntdll!LdrUnloadDll () from /c/Windows/SYSTEM32/ntdll.dll 
2    Thread 10264.0x2bac 0x00000000775c2385 in ntdll!LdrUnloadDll () from /c/Windows/SYSTEM32/ntdll.dll 
* 1    Thread 10264.0x2a90 main () at main.cpp:67 
(gdb) bt 
#0  main () at main.cpp:67 
(gdb) c 
Continuing. 

```

达到断点后，*info threads*命令列出了活动线程。在这里，我们可以清楚地看到条件变量的使用，其中一个线程在`ntdll!ZwWaitForMultipleObjects()`中等待。正如第三章中所介绍的，*C++多线程 API*，这是 Windows 使用其本地多线程 API 实现条件变量的一部分。

当我们创建一个回溯（`bt`命令）时，我们可以看到线程 1（当前线程）的当前堆栈只有一个帧，只有主方法，因为我们从这个起始点的这一行没有调用其他函数。

# 回溯

在正常的应用程序执行过程中，比如我们之前看过的 GUI 应用程序，向应用程序发送`SIGINT`也可以跟随着创建回溯的命令，就像这样：

```cpp
Thread 1 received signal SIGINT, Interrupt.
0x00007fff8a3fff72 in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
(gdb) bt
#0  0x00007fff8a3fff72 in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
#1  0x00007fff8a3ff3b3 in mach_msg () from /usr/lib/system/libsystem_kernel.dylib
#2  0x00007fff99f37124 in __CFRunLoopServiceMachPort () from /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
#3  0x00007fff99f365ec in __CFRunLoopRun () from /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
#4  0x00007fff99f35e38 in CFRunLoopRunSpecific () from /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
#5  0x00007fff97b73935 in RunCurrentEventLoopInMode ()
from /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/HIToolbox
#6  0x00007fff97b7376f in ReceiveNextEventCommon ()
from /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/HIToolbox
#7  0x00007fff97b735af in _BlockUntilNextEventMatchingListInModeWithFilter ()
from /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/HIToolbox
#8  0x00007fff9ed3cdf6 in _DPSNextEvent () from /System/Library/Frameworks/AppKit.framework/Versions/C/AppKit
#9  0x00007fff9ed3c226 in -[NSApplication _nextEventMatchingEventMask:untilDate:inMode:dequeue:] ()
from /System/Library/Frameworks/AppKit.framework/Versions/C/AppKit
#10 0x00007fff9ed30d80 in -[NSApplication run] () from /System/Library/Frameworks/AppKit.framework/Versions/C/AppKit
#11 0x0000000102a25143 in qt_plugin_instance () from /usr/local/Cellar/qt/5.8.0_1/plugins/platforms/libqcocoa.dylib
#12 0x0000000100cd3811 in QEventLoop::exec(QFlags<QEventLoop::ProcessEventsFlag>) () from /usr/local/opt/qt5/lib/QtCore.framework/Versions/5/QtCore
#13 0x0000000100cd80a7 in QCoreApplication::exec() () from /usr/local/opt/qt5/lib/QtCore.framework/Versions/5/QtCore
#14 0x0000000100003956 in main (argc=<optimized out>, argv=<optimized out>) at main.cpp:10
(gdb) c
Continuing.

```

在上述代码中，我们可以看到线程 ID 1 的执行，从创建开始，通过入口点（main）。每个后续的函数调用都被添加到堆栈中。当一个函数结束时，它就会从堆栈中移除。这既是一个好处，也是一个缺点。虽然它确实保持了回溯的干净整洁，但也意味着在最后一个函数调用之前发生的历史已经不复存在。

如果我们使用核心转储文件创建一个回溯，没有这些历史信息可能会非常恼人，并且可能会让人在试图缩小崩溃原因的范围时陷入困境。这意味着需要一定水平的经验才能成功调试。

在应用程序崩溃的情况下，调试器会将我们带到遭受崩溃的线程。通常，这是有问题代码的线程，但也可能是真正的错误在于另一个线程执行的代码，甚至是变量的不安全使用。如果一个线程改变了另一个线程当前正在读取的信息，后者可能会得到垃圾数据。这可能导致崩溃，甚至更糟糕的是，在应用程序的后续过程中出现损坏。

最坏的情况是堆栈被覆盖，例如被野指针。在这种情况下，堆栈上的缓冲区或类似的东西被写入超出其限制，从而通过填充新数据来擦除堆栈的部分。这就是缓冲区溢出，可能导致应用程序崩溃，或者（恶意）利用应用程序。

# 动态分析工具

尽管调试器的价值难以忽视，但有时需要不同类型的工具来回答关于内存使用、泄漏以及诊断或预防线程问题等问题。这就是 Valgrind 动态分析工具套件中的工具可以提供极大帮助的地方。作为构建动态分析工具的框架，Valgrind 发行版目前包含以下我们感兴趣的工具：

+   Memcheck

+   Helgrind

+   DRD

Memcheck 是一种内存错误检测器，它允许我们发现内存泄漏、非法读写，以及分配、释放等与内存相关的问题。

Helgrind 和 DRD 都是线程错误检测器。这基本上意味着它们将尝试检测任何多线程问题，如数据竞争和互斥锁的不正确使用。它们的区别在于 Helgrind 可以检测锁定顺序的违规，而 DRD 支持分离的线程，同时使用的内存比 Helgrind 少。

# 限制

动态分析工具的一个主要限制是它们需要与主机操作系统紧密集成。这是 Valgrind 专注于 POSIX 线程的主要原因，目前不适用于 Windows。

Valgrind 网站（[`valgrind.org/info/platforms.html`](http://valgrind.org/info/platforms.html)）描述了该问题如下：

“Windows 不在考虑范围之内，因为将其移植到 Windows 将需要很多更改，几乎可以成为一个单独的项目。（但是，Valgrind + Wine 可以通过一些努力来实现。）此外，非开源操作系统很难处理；能够看到操作系统和相关（libc）源代码使事情变得更容易。但是，Valgrind 在与 Wine 一起使用时非常有用，这意味着可以通过一些努力在 Valgrind 下运行 Windows 程序。”

基本上，这意味着 Windows 应用程序可以在 Linux 下使用 Valgrind 进行调试，但是使用 Windows 作为操作系统不会很快发生。

Valgrind 在 OS X/macOS 上可以工作，从 OS X 10.8（Mountain Lion）开始。由于苹果公司所做的更改，对最新版本的 macOS 的支持可能会有些不完整。与 Valgrind 的 Linux 版本一样，最好始终使用最新版本的 Valgrind。与 gdb 一样，使用发行版的软件包管理器，或者在 MacOS 上使用 Homebrew 等第三方软件包管理器。

# 替代方案

在 Windows 和其他平台上替代 Valgrind 工具的选择包括以下表中列出的工具：

| **名称** | **类型** | **平台** | **许可证** |
| --- | --- | --- | --- |
| Dr. Memory | 内存检查器 | 所有主要平台 | 开源 |
| gperftools（Google） | 堆，CPU 和调用分析器 | Linux（x86） | 开源 |
| Visual Leak Detector | 内存检查器 | Windows（Visual Studio） | 开源 |
| Intel Inspector | 内存和线程调试器 | Windows，Linux | 专有 |
| PurifyPlus | 内存，性能 | Windows，Linux | 专有 |
| Parasoft Insure++ | 内存和线程调试器 | Windows，Solaris，Linux，AIX | 专有 |

# Memcheck

当没有在其可执行文件的参数中指定其他工具时，Memcheck 是默认的 Valgrind 工具。 Memcheck 本身是一个内存错误检测器，能够检测以下类型的问题：

+   访问已分配边界之外的内存，堆栈溢出以及访问先前释放的内存块

+   使用未定义值，即未初始化的变量

+   错误释放堆内存，包括重复释放块

+   不匹配使用 C 和 C++风格的内存分配，以及数组分配器和解除分配器（`new[]`和`delete[]`）

+   在诸如`memcpy`之类的函数中重叠源和目标指针

+   将无效值（例如负值）作为`malloc`或类似函数的大小参数传递

+   内存泄漏；即，没有任何有效引用的堆块

使用调试器或简单的任务管理器，几乎不可能检测到前面列表中给出的问题。 Memcheck 的价值在于能够在开发的早期检测和修复问题，否则可能会导致数据损坏和神秘崩溃。

# 基本用法

使用 Memcheck 非常容易。如果我们使用第四章中创建的演示应用程序，*线程同步和通信*，我们知道通常我们使用以下方式启动它：

```cpp
$ ./dispatcher_demo

```

要使用默认的 Memcheck 工具运行 Valgrind，并将生成的输出记录到日志文件中，我们将如下启动它：

```cpp
$ valgrind --log-file=dispatcher.log --read-var-info=yes --leak-check=full ./dispatcher_demo

```

通过上述命令，我们将 Memcheck 的输出记录到一个名为`dispatcher.log`的文件中，并且还启用了对内存泄漏的全面检查，包括详细报告这些泄漏发生的位置，使用二进制文件中可用的调试信息。通过读取变量信息（`--read-var-info=yes`），我们可以获得更详细的关于内存泄漏发生位置的信息。

不能将日志记录到文件中，但除非是一个非常简单的应用程序，否则 Valgrind 生成的输出很可能太多，可能无法适应终端缓冲区。将输出作为文件允许我们稍后使用它作为参考，并使用比终端通常提供的更高级的工具进行搜索。

运行完这个命令后，我们可以按照以下方式检查生成的日志文件的内容：

```cpp
==5764== Memcheck, a memory error detector
==5764== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==5764== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==5764== Command: ./dispatcher_demo
==5764== Parent PID: 2838
==5764==
==5764==
==5764== HEAP SUMMARY:
==5764==     in use at exit: 75,184 bytes in 71 blocks
==5764==   total heap usage: 260 allocs, 189 frees, 88,678 bytes allocated
==5764==
==5764== 80 bytes in 10 blocks are definitely lost in loss record 1 of 5
==5764==    at 0x4C2E0EF: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==5764==    by 0x402EFD: Dispatcher::init(int) (dispatcher.cpp:40)
==5764==    by 0x409300: main (main.cpp:51)
==5764==
==5764== 960 bytes in 40 blocks are definitely lost in loss record 3 of 5
==5764==    at 0x4C2E0EF: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==5764==    by 0x409338: main (main.cpp:60)
==5764==
==5764== 1,440 (1,200 direct, 240 indirect) bytes in 10 blocks are definitely lost in loss record 4 of 5
==5764==    at 0x4C2E0EF: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==5764==    by 0x402EBB: Dispatcher::init(int) (dispatcher.cpp:38)
==5764==    by 0x409300: main (main.cpp:51)
==5764==
==5764== LEAK SUMMARY:
==5764==    definitely lost: 2,240 bytes in 60 blocks
==5764==    indirectly lost: 240 bytes in 10 blocks
==5764==      possibly lost: 0 bytes in 0 blocks
==5764==    still reachable: 72,704 bytes in 1 blocks
==5764==         suppressed: 0 bytes in 0 blocks
==5764== Reachable blocks (those to which a pointer was found) are not shown.
==5764== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==5764==
==5764== For counts of detected and suppressed errors, rerun with: -v
==5764== ERROR SUMMARY: 3 errors from 3 contexts (suppressed: 0 from 0) 

```

在这里，我们可以看到我们总共有三个内存泄漏。其中两个是在第 38 行和第 40 行的`dispatcher`类中分配的：

```cpp
w = new Worker; 

```

另一个问题是：

```cpp
t = new thread(&Worker::run, w); 

```

我们还看到了在`main.cpp`的第 60 行分配的内存泄漏：

```cpp
rq = new Request(); 

```

虽然这些分配本身没有问题，但是如果我们在应用程序生命周期中跟踪它们，我们会注意到我们从未在这些对象上调用`delete`。如果我们要修复这些内存泄漏，我们需要在完成后删除这些`Request`实例，并在`dispatcher`类的析构函数中清理`Worker`和`thread`实例。

由于在这个演示应用程序中，整个应用程序在运行结束时由操作系统终止并清理，所以这并不是一个真正的问题。对于一个应用程序，在这个应用程序中，同一个调度程序以一种不断生成和添加新请求的方式被使用，同时可能还动态地扩展工作线程的数量，这将是一个真正的问题。在这种情况下，必须小心解决这些内存泄漏问题。

# 错误类型

Memcheck 可以检测到各种与内存相关的问题。以下部分总结了这些错误及其含义。

# 非法读取/非法写入错误

这些错误通常以以下格式报告：

```cpp
Invalid read of size <bytes>
at 0x<memory address>: (location)
by 0x<memory address>: (location)
by 0x<memory address>: (location)
Address 0x<memory address> <error description>

```

在前面错误消息的第一行将告诉我们是无效的读取还是写入访问。接下来的几行将是一个回溯，详细说明了发生无效读取或写入的位置（可能还包括源文件中的行），以及调用该代码的位置。

最后一行将详细说明发生的非法访问类型，比如读取已经释放的内存块。

这种类型的错误表明写入或读取一个不应该访问的内存部分。这可能是因为访问了一个野指针（即引用一个随机的内存地址），或者是由于代码中的早期问题导致了错误的内存地址计算，或者是没有尊重内存边界，读取了数组或类似结构的边界之外。

通常情况下，当报告这种类型的错误时，应该非常重视，因为它表明存在一个基本问题，不仅可能导致数据损坏和崩溃，还可能导致其他人可以利用的错误。

# 使用未初始化的值

简而言之，这是一个变量的值在没有被赋值的情况下被使用的问题。在这一点上，很可能这些内容只是刚刚分配的 RAM 部分中的任意字节。因此，每当使用或访问这些内容时，可能会导致不可预测的行为。

当遇到时，Memcheck 会抛出类似于这样的错误：

```cpp
$ valgrind --read-var-info=yes --leak-check=full ./unval
==6822== Memcheck, a memory error detector
==6822== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==6822== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==6822== Command: ./unval
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E87B83: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Use of uninitialised value of size 8
==6822==    at 0x4E8476B: _itoa_word (_itoa.c:179)
==6822==    by 0x4E8812C: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E84775: _itoa_word (_itoa.c:179)
==6822==    by 0x4E8812C: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E881AF: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E87C59: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E8841A: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E87CAB: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== Conditional jump or move depends on uninitialised value(s)
==6822==    at 0x4E87CE2: vfprintf (vfprintf.c:1631)
==6822==    by 0x4E8F898: printf (printf.c:33)
==6822==    by 0x400541: main (unval.cpp:6)
==6822== 
==6822== 
==6822== HEAP SUMMARY:
==6822==     in use at exit: 0 bytes in 0 blocks
==6822==   total heap usage: 1 allocs, 1 frees, 1,024 bytes allocated
==6822== 
==6822== All heap blocks were freed -- no leaks are possible
==6822== 
==6822== For counts of detected and suppressed errors, rerun with: -v
==6822== Use --track-origins=yes to see where uninitialised values come from
==6822== ERROR SUMMARY: 8 errors from 8 contexts (suppressed: 0 from 0)

```

这一系列特定的错误是由以下一小段代码引起的：

```cpp
#include <cstring>
 #include <cstdio>

 int main() {
    int x;  
    printf ("x = %d\n", x); 
    return 0;
 } 

```

正如我们在前面的代码中所看到的，我们从未初始化我们的变量，这将被设置为任意的随机值。如果幸运的话，它将被设置为零，或者一个同样（希望如此）无害的值。这段代码展示了我们的任何未初始化的变量如何进入库代码。

未初始化变量的使用是否有害很难说，这在很大程度上取决于变量的类型和受影响的代码。然而，简单地分配一个安全的默认值要比追踪和调试可能由未初始化变量（随机）引起的神秘问题要容易得多。

要了解未初始化变量的来源，可以向 Memcheck 传递`-track-origins=yes`标志。这将告诉它为每个变量保留更多信息，这将使追踪此类问题变得更容易。

# 未初始化或不可寻址的系统调用值

每当调用一个函数时，可能会传递未初始化的值作为参数，甚至是指向不可寻址的缓冲区的指针。在任何一种情况下，Memcheck 都会记录这一点：

```cpp
$ valgrind --read-var-info=yes --leak-check=full ./unsyscall
==6848== Memcheck, a memory error detector
==6848== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==6848== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==6848== Command: ./unsyscall
==6848== 
==6848== Syscall param write(buf) points to uninitialised byte(s)
==6848==    at 0x4F306E0: __write_nocancel (syscall-template.S:84)
==6848==    by 0x4005EF: main (unsyscall.cpp:7)
==6848==  Address 0x5203040 is 0 bytes inside a block of size 10 alloc'd
==6848==    at 0x4C2DB8F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==6848==    by 0x4005C7: main (unsyscall.cpp:5)
==6848== 
==6848== Syscall param exit_group(status) contains uninitialised byte(s)
==6848==    at 0x4F05B98: _Exit (_exit.c:31)
==6848==    by 0x4E73FAA: __run_exit_handlers (exit.c:97)
==6848==    by 0x4E74044: exit (exit.c:104)
==6848==    by 0x4005FC: main (unsyscall.cpp:8)
==6848== 
==6848== 
==6848== HEAP SUMMARY:
==6848==     in use at exit: 14 bytes in 2 blocks
==6848==   total heap usage: 2 allocs, 0 frees, 14 bytes allocated
==6848== 
==6848== LEAK SUMMARY:
==6848==    definitely lost: 0 bytes in 0 blocks
==6848==    indirectly lost: 0 bytes in 0 blocks
==6848==      possibly lost: 0 bytes in 0 blocks
==6848==    still reachable: 14 bytes in 2 blocks
==6848==         suppressed: 0 bytes in 0 blocks
==6848== Reachable blocks (those to which a pointer was found) are not shown.
==6848== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==6848== 
==6848== For counts of detected and suppressed errors, rerun with: -v
==6848== Use --track-origins=yes to see where uninitialised values come from
==6848== ERROR SUMMARY: 2 errors from 2 contexts (suppressed: 0 from 0)

```

前面的日志是由这段代码生成的：

```cpp
#include <cstdlib>
 #include <unistd.h> 

 int main() {  
    char* arr  = (char*) malloc(10);  
    int*  arr2 = (int*) malloc(sizeof(int));  
    write(1, arr, 10 ); 
    exit(arr2[0]);
 } 

```

与前一节详细介绍的未初始化值的一般用法类似，传递未初始化或其他可疑的参数，至少是有风险的，最坏的情况下可能导致崩溃、数据损坏或更糟的情况。

# 非法释放

非法的释放或删除通常是指尝试重复调用`free()`或`delete()`来释放已经释放的内存块。虽然不一定有害，但这表明设计不良，绝对需要修复。

当尝试使用不指向该内存块开头的指针释放内存块时，也可能会发生这种情况。这是为什么我们永远不应该对从`malloc()`或`new()`调用获得的原始指针进行指针算术运算，而是使用副本的主要原因之一。

# 不匹配的释放

内存块的分配和释放应该始终使用匹配的函数。这意味着当我们使用 C 风格函数分配时，我们使用相同 API 的匹配函数释放。对于 C++风格的分配和释放也是如此。

简而言之，这意味着以下内容：

+   如果我们使用`malloc`、`calloc`、`valloc`、`realloc`或`memalign`分配，我们使用`free`释放

+   如果我们使用 new 分配，我们使用`delete`释放

+   如果我们使用`new[]`分配，我们使用`delete[]`释放

混淆这些不一定会导致问题，但这样做是未定义的行为。后一种类型的分配和释放是特定于数组的。对使用`new[]`分配的数组不使用`delete[]`可能会导致内存泄漏，甚至更糟。

# 重叠的源和目的地

这种类型的错误表明传递给源和目的地内存块的指针重叠（基于预期大小）。这种错误的结果通常是一种形式的损坏或系统崩溃。

# 可疑的参数值

对于内存分配函数，Memcheck 会验证传递给它们的参数是否真的有意义。其中一个例子是传递负大小，或者它将远远超出合理的分配大小：例如，请求分配一百万兆字节的内存。这些值很可能是代码中早期错误计算的结果。

Memcheck 会像 Memcheck 手册中的这个例子一样报告这个错误：

```cpp
==32233== Argument 'size' of function malloc has a fishy (possibly negative) value: -3
==32233==    at 0x4C2CFA7: malloc (vg_replace_malloc.c:298)
==32233==    by 0x400555: foo (fishy.c:15)
==32233==    by 0x400583: main (fishy.c:23)

```

在这里，尝试将值-3 传递给`malloc`，这显然没有多大意义。由于这显然是一个荒谬的操作，这表明代码中存在严重的错误。

# 内存泄漏检测

对于 Memcheck 报告的内存泄漏，最重要的是，许多报告的*泄漏*实际上可能并不是泄漏。这反映在 Memcheck 报告它发现的任何潜在问题的方式如下：

+   绝对丢失

+   间接丢失

+   可能丢失

在三种可能的报告类型中，**绝对丢失**类型是唯一一种绝对确定所涉及的内存块不再可达的情况，没有剩余的指针或引用，这使得应用程序永远无法释放内存。

在**间接丢失**类型的情况下，我们并没有丢失这些内存块本身的指针，而是丢失了指向这些块的结构的指针。例如，当我们直接丢失对数据结构的根节点（如红/黑树或二叉树）的访问权限时，就会发生这种情况。结果，我们也失去了访问任何子节点的能力。

最后，**可能丢失**是一个综合类型，Memcheck 并不完全确定是否仍然有对内存块的引用。这可能发生在存在内部指针的情况下，比如特定类型的数组分配的情况。它也可能通过多重继承的使用发生，其中 C++对象使用自引用。

如同之前在 Memcheck 的基本用法部分提到的，建议始终使用`--leak-check=full`来运行 Memcheck，以获取关于内存泄漏发生的具体位置的详细信息。

# Helgrind

Helgrind 的目的是检测多线程应用程序中同步实现的问题。它可以检测对 POSIX 线程的错误使用，由于错误的锁定顺序而导致的潜在死锁问题，以及数据竞争--在没有线程同步的情况下读取或写入数据。

# 基本用法

我们以以下方式启动 Helgrind 应用程序：

```cpp
$ valgrind --tool=helgrind --read-var-info=yes --log-file=dispatcher_helgrind.log ./dispatcher_demo

```

与运行 Memcheck 类似，这将运行应用程序并将所有生成的输出记录到日志文件中，同时明确使用二进制中所有可用的调试信息。

运行应用程序后，我们检查生成的日志文件：

```cpp
==6417== Helgrind, a thread error detector
==6417== Copyright (C) 2007-2015, and GNU GPL'd, by OpenWorks LLP et al.
==6417== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==6417== Command: ./dispatcher_demo
==6417== Parent PID: 2838
==6417== 
==6417== ---Thread-Announcement------------------------------------------
==6417== 
==6417== Thread #1 is the program's root thread 

```

在关于应用程序和 Valgrind 版本的初始基本信息之后，我们被告知根线程已经创建：

```cpp
==6417== 
==6417== ---Thread-Announcement------------------------------------------
==6417== 
==6417== Thread #2 was created
==6417==    at 0x56FB7EE: clone (clone.S:74)
==6417==    by 0x53DE149: create_thread (createthread.c:102)
==6417==    by 0x53DFE83: pthread_create@@GLIBC_2.2.5 (pthread_create.c:679)
==6417==    by 0x4C34BB7: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x4EF8DC2: std::thread::_M_start_thread(std::shared_ptr<std::thread::_Impl_base>, void (*)()) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x403AD7: std::thread::thread<void (Worker::*)(), Worker*&>(void (Worker::*&&)(), Worker*&) (thread:137)
==6417==    by 0x4030E6: Dispatcher::init(int) (dispatcher.cpp:40)
==6417==    by 0x4090A0: main (main.cpp:51)
==6417== 
==6417== ----------------------------------------------------------------

```

第一个线程是由调度程序创建并记录的。接下来我们得到了第一个警告：

```cpp
==6417== 
==6417==  Lock at 0x60F4A0 was first observed
==6417==    at 0x4C321BC: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x401CD1: __gthread_mutex_lock(pthread_mutex_t*) (gthr-default.h:748)
==6417==    by 0x402103: std::mutex::lock() (mutex:135)
==6417==    by 0x40337E: Dispatcher::addWorker(Worker*) (dispatcher.cpp:108)
==6417==    by 0x401DF9: Worker::run() (worker.cpp:49)
==6417==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6417==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6417==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6417==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6417==    by 0x4EF8C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x4C34DB6: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x53DF6B9: start_thread (pthread_create.c:333)
==6417==  Address 0x60f4a0 is 0 bytes inside data symbol "_ZN10Dispatcher12workersMutexE"
==6417== 
==6417== Possible data race during write of size 1 at 0x5CD9261 by thread #1
==6417== Locks held: 1, at address 0x60F4A0
==6417==    at 0x403650: Worker::setRequest(AbstractRequest*) (worker.h:38)
==6417==    by 0x403253: Dispatcher::addRequest(AbstractRequest*) (dispatcher.cpp:70)
==6417==    by 0x409132: main (main.cpp:63)
==6417== 
==6417== This conflicts with a previous read of size 1 by thread #2
==6417== Locks held: none
==6417==    at 0x401E02: Worker::run() (worker.cpp:51)
==6417==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6417==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6417==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6417==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6417==    by 0x4EF8C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x4C34DB6: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x53DF6B9: start_thread (pthread_create.c:333)
==6417==  Address 0x5cd9261 is 97 bytes inside a block of size 104 alloc'd
==6417==    at 0x4C2F50F: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x40308F: Dispatcher::init(int) (dispatcher.cpp:38)
==6417==    by 0x4090A0: main (main.cpp:51)
==6417==  Block was alloc'd by thread #1
==6417== 
==6417== ----------------------------------------------------------------

```

在前面的警告中，Helgrind 告诉我们线程 ID 1 和 2 之间发生了大小为 1 的冲突读取。由于 C++11 线程 API 使用了大量模板，跟踪可能有些难以阅读。关键在于以下几行：

```cpp
==6417==    at 0x403650: Worker::setRequest(AbstractRequest*) (worker.h:38) ==6417==    at 0x401E02: Worker::run() (worker.cpp:51) 

```

这对应以下代码行：

```cpp
void setRequest(AbstractRequest* request) { this->request = request; ready = true; }
while (!ready && running) { 

```

这些代码行中唯一大小为 1 的变量是布尔变量`ready`。由于这是一个布尔变量，我们知道这是一个原子操作（详见﻿第八章，*原子操作-与硬件一起工作*）。因此，我们可以忽略这个警告。

接下来，我们为这个线程得到另一个警告：

```cpp
==6417== Possible data race during write of size 1 at 0x5CD9260 by thread #1
==6417== Locks held: none
==6417==    at 0x40362C: Worker::stop() (worker.h:37)
==6417==    by 0x403184: Dispatcher::stop() (dispatcher.cpp:50)
==6417==    by 0x409163: main (main.cpp:70)
==6417== 
==6417== This conflicts with a previous read of size 1 by thread #2 ==6417== Locks held: none
==6417==    at 0x401E0E: Worker::run() (worker.cpp:51)
==6417==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6417==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6417==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6417==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6417==    by 0x4EF8C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x4C34DB6: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x53DF6B9: start_thread (pthread_create.c:333)
==6417==  Address 0x5cd9260 is 96 bytes inside a block of size 104 alloc'd
==6417==    at 0x4C2F50F: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x40308F: Dispatcher::init(int) (dispatcher.cpp:38)
==6417==    by 0x4090A0: main (main.cpp:51)
==6417==  Block was alloc'd by thread #1 

```

与第一个警告类似，这也涉及一个布尔变量--这里是`Worker`实例中的`running`变量。由于这也是一个原子操作，我们可以再次忽略这个警告。

在这个警告之后，我们看到其他线程重复了这些警告。我们还多次看到了这个警告的重复：

```cpp
==6417==  Lock at 0x60F540 was first observed
==6417==    at 0x4C321BC: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==    by 0x401CD1: __gthread_mutex_lock(pthread_mutex_t*) (gthr-default.h:748)
==6417==    by 0x402103: std::mutex::lock() (mutex:135)
==6417==    by 0x409044: logFnc(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) (main.cpp:40)
==6417==    by 0x40283E: Request::process() (request.cpp:19)
==6417==    by 0x401DCE: Worker::run() (worker.cpp:44)
==6417==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6417==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6417==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6417==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6417==    by 0x4EF8C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x4C34DB6: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
==6417==  Address 0x60f540 is 0 bytes inside data symbol "logMutex"
==6417== 
==6417== Possible data race during read of size 8 at 0x60F238 by thread #1
==6417== Locks held: none
==6417==    at 0x4F4ED6F: std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x4F4F236: std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x403199: Dispatcher::stop() (dispatcher.cpp:53)
==6417==    by 0x409163: main (main.cpp:70)
==6417== 
==6417== This conflicts with a previous write of size 8 by thread #7
==6417== Locks held: 1, at address 0x60F540
==6417==    at 0x4F4EE25: std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6417==    by 0x409055: logFnc(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) (main.cpp:41)
==6417==    by 0x402916: Request::finish() (request.cpp:27)
==6417==    by 0x401DED: Worker::run() (worker.cpp:45)
==6417==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6417==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6417==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6417==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6417==  Address 0x60f238 is 24 bytes inside data symbol "_ZSt4cout@@GLIBCXX_3.4"  

```

这个警告是由于在线程之间没有使用标准输出同步引起的。尽管这个演示应用程序的日志记录函数使用互斥锁来同步工作线程记录的文本，但在一些位置我们也以不安全的方式写入标准输出。

通过使用一个中央、线程安全的日志记录函数，这相对容易修复。尽管这不太可能引起任何稳定性问题，但很可能会导致任何日志输出最终变成一团乱七八糟、无法使用的混乱。

# 滥用 pthreads API

Helgrind 检测到了大量涉及 pthreads API 的错误，如其手册所总结的，并列在下面：

+   解锁无效的互斥锁

+   解锁未锁定的互斥锁

+   解锁由不同线程持有的互斥锁

+   销毁无效或锁定的互斥锁

+   递归锁定非递归互斥锁

+   释放包含锁定互斥锁的内存

+   将互斥锁参数传递给期望读写锁参数的函数，反之亦然

+   POSIX pthread 函数的失败会返回必须处理的错误代码

+   线程在仍持有锁定的锁时退出

+   使用未锁定的互斥锁、无效的互斥锁或被不同线程锁定的互斥锁调用`pthread_cond_wait`

+   条件变量与其关联的互斥锁之间的不一致绑定

+   无效或重复初始化 pthread 屏障

+   在仍有线程等待的 pthread 屏障上进行初始化

+   销毁从未初始化或仍有线程等待的 pthread 屏障对象

+   等待未初始化的 pthread 屏障

此外，如果 Helgrind 本身没有检测到错误，但 pthread 库本身对 Helgrind 拦截的每个函数返回错误，那么 Helgrind 也会报告错误。

# 锁定顺序问题

锁定顺序检测使用的假设是一旦一系列锁以特定顺序被访问，它们将永远以这种顺序使用。例如，想象一下，一个由两个锁保护的资源。正如我们在第四章中看到的调度程序演示，*线程同步和通信*，我们在其 Dispatcher 类中使用两个互斥锁，一个用于管理对工作线程的访问，另一个用于请求实例。

在该代码的正确实现中，我们始终确保在尝试获取另一个互斥锁之前解锁一个互斥锁，因为另一个线程可能已经获得了对第二个互斥锁的访问权，并尝试获取对第一个互斥锁的访问权，从而创建死锁情况。

虽然有用，但重要的是要意识到，在某些领域，这种检测算法目前还不完善。这在使用条件变量时尤为明显，条件变量自然使用的锁定顺序往往会被 Helgrind 报告为*错误*。

这里的要点是，人们必须检查这些日志消息并判断它们的价值，但与多线程 API 的直接误用不同，报告的问题是否是误报还不够清晰。

# 数据竞争

实质上，数据竞争是指两个或更多线程在没有任何同步机制的情况下尝试读取或写入相同的资源。在这里，只有并发读取和写入，或两个同时写入，才会真正有害；因此，只有这两种类型的访问会被报告。

在早期关于基本 Helgrind 使用的部分中，我们在日志中看到了这种类型错误的一些示例。这里涉及到对变量的同时写入和读取。正如我们在该部分中也提到的，Helgrind 并不关心写入或读取是否是原子的，而只是报告潜在问题。

与锁定顺序问题类似，这意味着人们必须根据每个数据竞争报告的价值来判断，因为许多报告可能是误报。

# DRD

DRD 与 Helgrind 非常相似，因为它也检测应用程序中的线程和同步问题。DRD 与 Helgrind 的主要区别在于：

+   DRD 使用的内存较少

+   DRD 无法检测锁定顺序违规

+   DRD 支持分离线程

通常，人们希望同时运行 DRD 和 Helgrind，以便将两者的输出进行比较。由于许多潜在问题非常不确定，使用两种工具通常有助于确定最严重的问题。

# 基本用法

启动 DRD 与启动其他工具非常相似-我们只需指定我们想要的工具，如下所示：

```cpp
$ valgrind --tool=drd --log-file=dispatcher_drd.log --read-var-info=yes ./dispatcher_demo

```

应用程序完成后，我们检查生成的日志文件内容。

```cpp
==6576== drd, a thread error detector
==6576== Copyright (C) 2006-2015, and GNU GPL'd, by Bart Van Assche.
==6576== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==6576== Command: ./dispatcher_demo
==6576== Parent PID: 2838
==6576== 
==6576== Conflicting store by thread 1 at 0x05ce51b1 size 1
==6576==    at 0x403650: Worker::setRequest(AbstractRequest*) (worker.h:38)
==6576==    by 0x403253: Dispatcher::addRequest(AbstractRequest*) (dispatcher.cpp:70)
==6576==    by 0x409132: main (main.cpp:63)
==6576== Address 0x5ce51b1 is at offset 97 from 0x5ce5150\. Allocation context:
==6576==    at 0x4C3150F: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_drd-amd64-linux.so)
==6576==    by 0x40308F: Dispatcher::init(int) (dispatcher.cpp:38)
==6576==    by 0x4090A0: main (main.cpp:51)
==6576== Other segment start (thread 2)
==6576==    at 0x4C3818C: pthread_mutex_unlock (in /usr/lib/valgrind/vgpreload_drd-amd64-linux.so)
==6576==    by 0x401D00: __gthread_mutex_unlock(pthread_mutex_t*) (gthr-default.h:778)
==6576==    by 0x402131: std::mutex::unlock() (mutex:153)
==6576==    by 0x403399: Dispatcher::addWorker(Worker*) (dispatcher.cpp:110)
==6576==    by 0x401DF9: Worker::run() (worker.cpp:49)
==6576==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6576==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6576==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6576==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6576==    by 0x4F04C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6576==    by 0x4C3458B: ??? (in /usr/lib/valgrind/vgpreload_drd-amd64-linux.so)
==6576==    by 0x53EB6B9: start_thread (pthread_create.c:333)
==6576== Other segment end (thread 2)
==6576==    at 0x4C3725B: pthread_mutex_lock (in /usr/lib/valgrind/vgpreload_drd-amd64-linux.so)
==6576==    by 0x401CD1: __gthread_mutex_lock(pthread_mutex_t*) (gthr-default.h:748)
==6576==    by 0x402103: std::mutex::lock() (mutex:135)
==6576==    by 0x4023F8: std::unique_lock<std::mutex>::lock() (mutex:485)
==6576==    by 0x40219D: std::unique_lock<std::mutex>::unique_lock(std::mutex&) (mutex:415)
==6576==    by 0x401E33: Worker::run() (worker.cpp:52)
==6576==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
==6576==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
==6576==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
==6576==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
==6576==    by 0x4F04C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==6576==    by 0x4C3458B: ??? (in /usr/lib/valgrind/vgpreload_drd-amd64-linux.so) 

```

前面的总结基本上重复了我们在 Helgrind 日志中看到的内容。我们看到了相同的数据竞争报告（冲突存储），由于原子性，我们可以安全地忽略它。至少对于这个特定的代码，使用 DRD 并没有增加任何我们从使用 Helgrind 已经知道的东西。

无论如何，最好同时使用两种工具，以防一种工具发现了另一种工具没有发现的问题。

# 特性

DRD 将检测以下错误：

+   数据竞争

+   锁争用（死锁和延迟）

+   错误使用 pthreads API

对于第三点，根据 DRD 手册，DRD 检测到的错误列表与 Helgrind 的非常相似：

+   将一种类型的同步对象（例如互斥锁）的地址传递给期望指向另一种类型同步对象（例如条件变量）的 POSIX API 调用

+   尝试解锁未被锁定的互斥锁

+   尝试解锁被另一个线程锁定的互斥锁

+   尝试递归锁定`PTHREAD_MUTEX_NORMAL`类型的互斥锁或自旋锁

+   销毁或释放被锁定的互斥锁

+   在与条件变量相关的互斥锁未被锁定时向条件变量发送信号

+   在未锁定的互斥锁上调用`pthread_cond_wait`，即由另一个线程锁定，或者已被递归锁定

+   通过`pthread_cond_wait`将两个不同的互斥锁与条件变量关联

+   销毁或释放正在等待的条件变量

+   销毁或释放被锁定的读写同步对象

+   尝试解锁被调用线程未锁定的读写同步对象

+   尝试递归锁定独占的读写同步对象

+   尝试将用户定义的读写同步对象的地址传递给 POSIX 线程函数

+   尝试将 POSIX 读写同步对象的地址传递给用户定义的读写同步对象的注释之一

+   重新初始化互斥锁、条件变量、读写锁、信号量或屏障

+   销毁或释放正在等待的信号量或屏障

+   屏障等待和屏障销毁之间的同步丢失

+   在不先解锁由该线程锁定的自旋锁、互斥锁或读写同步对象的情况下退出线程

+   向`pthread_join`或`pthread_cancel`传递无效的线程 ID

如前所述，DRD 还支持分离线程，这里有帮助。锁定顺序检查是否重要取决于应用程序。

# C++11 线程支持

DRD 手册中包含了关于 C++11 线程支持的这一部分。

如果要使用`c++11`类`std::thread`，则需要对该类实现中使用的`std::shared_ptr<>`对象进行注释：

+   在公共头文件的开头或每个源文件的开头添加以下代码，然后再包含任何 C++头文件：

```cpp
    #include <valgrind/drd.h>
    #define _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(addr)
    ANNOTATE_HAPPENS_BEFORE(addr)
    #define _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(addr)
    ANNOTATE_HAPPENS_AFTER(addr)

```

+   下载 GCC 源代码，从源文件`libstdc++-v3/src/c++11/thread.cc`中复制`execute_native_thread_routine()`和`std::thread::_M_start_thread()`函数的实现到一个与应用程序链接的源文件中。确保在此源文件中也正确定义`_GLIBCXX_SYNCHRONIZATION_HAPPENS_*()`宏。

使用 DRD 与使用 C++11 线程 API 的应用程序可能会出现许多误报，这将通过前面的*修复*来解决。

然而，当使用 GCC 5.4 和 Valgrind 3.11（可能也适用于旧版本）时，这个问题似乎不再存在。然而，当使用 C++11 线程 API 时，如果突然看到 DRD 输出中出现大量误报，这是需要记住的事情。

# 总结

在本章中，我们看了如何处理多线程应用程序的调试。我们探讨了在多线程环境中使用调试器的基础知识。接下来，我们看到了如何使用 Valgrind 框架中的三种工具，这些工具可以帮助我们追踪多线程和其他关键问题。

在这一点上，我们可以拿取应用程序，使用前面章节中的信息，并分析它们是否存在需要修复的问题，包括内存泄漏和不正确使用同步机制。

在下一章中，我们将运用我们所学到的知识，探讨多线程编程和一般开发中的一些最佳实践。
