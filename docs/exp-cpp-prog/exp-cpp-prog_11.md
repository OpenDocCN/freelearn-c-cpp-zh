# 第十一章：处理器和操作系统上的多线程实现

任何多线程应用程序的基础都是由处理器的硬件实现所需功能以及这些功能如何被操作系统转换为应用程序使用的 API 所形成的。了解这个基础对于开发对多线程应用程序的最佳实现方式至关重要。

本章涵盖的主题包括以下内容：

+   操作系统如何改变使用这些硬件功能

+   各种架构中内存安全和内存模型背后的概念

+   操作系统的各种进程和线程模型之间的差异

+   并发

# 介绍 POSIX pthreads

Unix、Linux 和 macOS 在很大程度上符合 POSIX 标准。**Unix 可移植操作系统接口**（**POSIX**）是一个 IEEE 标准，它帮助所有 Unix 和类 Unix 操作系统，即 Linux 和 macOS，通过一个统一的接口进行通信。

有趣的是，POSIX 也受到符合 POSIX 的工具的支持--Cygwin、MinGW 和 Windows 子系统用于 Linux--它们在 Windows 平台上提供了一个伪 Unix 样的运行时和开发环境。

请注意，`pthread` 是一个在 Unix、Linux 和 macOS 中使用的符合 POSIX 标准的 C 库。从 C++11 开始，C++通过 C++线程支持库和并发库本地支持线程。在本章中，我们将了解如何以面向对象的方式使用 pthread、线程支持和并发库。此外，我们将讨论使用本地 C++线程支持和并发库与使用 POSIX pthreads 或其他第三方线程框架的优点。

# 使用 pthread 库创建线程

让我们直入主题。您需要了解我们将讨论的 pthread API，以便开始动手。首先，此函数用于创建一个新线程：

```cpp
 #include <pthread.h>
 int pthread_create(
              pthread_t *thread,
              const pthread_attr_t *attr,
              void *(*start_routine)(void*),
              void *arg
 )
```

以下表格简要解释了前述函数中使用的参数：

| **API 参数** | **注释** |
| --- | --- |
| `pthread_t *thread` | 线程句柄指针 |
| `pthread_attr_t *attr` | 线程属性 |
| `void *(*start_routine)(void*)` | 线程函数指针 |
| `void * arg` | 线程参数 |

此函数会阻塞调用线程，直到第一个参数中传递的线程退出，如下所示：

```cpp
int pthread_join ( pthread_t *thread, void **retval )
```

以下表格简要描述了前述函数中的参数：

| **API 参数** | **注释** |
| --- | --- |
| `pthread_t thread` | 线程句柄 |
| `void **retval` | 输出参数，指示线程过程的退出代码 |

接下来的函数应该在线程上下文中使用。在这里，`retval` 是调用此函数的线程的退出代码，表示调用此函数的线程的退出代码：

```cpp
int pthread_exit ( void *retval )
```

这是在此函数中使用的参数：

| **API 参数** | **注释** |
| --- | --- |
| `void *retval` | 线程过程的退出代码 |

以下函数返回线程 ID：

```cpp
pthread_t pthread_self(void)
```

让我们编写我们的第一个多线程应用程序：

```cpp
#include <pthread.h>
#include <iostream>

using namespace std;

void* threadProc ( void *param ) {
  for (int count=0; count<3; ++count)
    cout << "Message " << count << " from " << pthread_self()
         << endl;
  pthread_exit(0);
}

int main() {
  pthread_t thread1, thread2, thread3;

  pthread_create ( &thread1, NULL, threadProc, NULL );
  pthread_create ( &thread2, NULL, threadProc, NULL );
  pthread_create ( &thread3, NULL, threadProc, NULL );

  pthread_join( thread1, NULL );
  pthread_join( thread2, NULL );

  pthread_join( thread3, NULL );

  return 0;

}
```

# 如何编译和运行

该程序可以使用以下命令编译：

```cpp
g++ main.cpp -lpthread
```

如您所见，我们需要动态链接 POSIX `pthread` 库。

查看以下截图并可视化多线程程序的输出：

![](img/ffc7c770-a884-446e-bb63-8f5ef4b1485e.png)

在 ThreadProc 中编写的代码在线程上下文中运行。前面的程序总共有四个线程，包括主线程。我使用`pthread_join`阻塞了主线程，强制它等待其他三个线程先完成它们的任务，否则主线程会在它们之前退出。当主线程退出时，应用程序也会退出，这会过早地销毁新创建的线程。

尽管我们按照相应的顺序创建了`thread1`、`thread2`和`thread3`，但不能保证它们将按照创建的确切顺序启动。

操作系统调度程序根据操作系统调度程序使用的算法决定必须启动线程的顺序。有趣的是，在同一系统的不同运行中，线程启动的顺序可能会有所不同。

# C++是否本地支持线程？

从 C++11 开始，C++确实本地支持线程，并且通常被称为 C++线程支持库。C++线程支持库提供了对 POSIX pthreads C 库的抽象。随着时间的推移，C++本机线程支持已经得到了很大的改善。

我强烈建议您使用 C++本机线程而不是 pthread。C++线程支持库在所有平台上都受支持，因为它是标准 C++的正式部分，而不是仅在 Unix、Linux 和 macOS 上直接支持的 POSIX `pthread`库。

最好的部分是 C++17 中的线程支持已经成熟到了一个新的水平，并且有望在 C++20 中达到下一个水平。因此，在项目中考虑使用 C++线程支持库是一个好主意。

# 定义进程和线程

基本上，对于**操作系统**（**OS**），进程由一个或多个线程组成，每个线程处理自己的状态和变量。可以将其视为分层配置，操作系统作为基础，为（用户）进程的运行提供支持。然后，每个进程由一个或多个线程组成。进程之间的通信由操作系统提供的**进程间通信**（**IPC**）处理。

在图形视图中，这看起来像下面的样子：

![](img/e2a11c2f-0b17-424f-ab3e-4db6ff9bdc62.png)

操作系统中的每个进程都有自己的状态，进程中的每个线程也有自己的状态，与该进程中的其他线程相关。虽然 IPC 允许进程彼此通信，但线程可以以各种方式与同一进程中的其他线程通信，我们将在接下来的章节中更深入地探讨这一点。这通常涉及线程之间的某种共享内存。

应用程序是从二进制数据中加载的，格式为特定的可执行文件格式，例如**可执行和可链接格式**（**ELF**），通常用于 Linux 和许多其他操作系统。对于 ELF 二进制文件，应始终存在以下数量的部分：

+   `.bss`

+   `.data`

+   `.rodata`

+   `.text`

`.bss`部分基本上是使用未初始化的内存分配的，包括空数组，因此在二进制文件中不占用任何空间，因为在可执行文件中存储纯零行没有意义。类似地，还有`.data`部分包含初始化数据。其中包含全局表、变量等。最后，`.rodata`部分类似于`.data`，但正如其名称所示，是只读的。其中包含硬编码的字符串。

在`.text`部分，我们找到实际的应用程序指令（代码），这些指令将由处理器执行。整个这些将被操作系统加载，从而创建一个进程。这样的进程布局看起来像下面的图表：

![](img/078ec7eb-400d-41cb-85af-c612e8612b9b.png)

这是从 ELF 格式二进制文件启动时进程的样子，尽管在内存中的最终格式在任何操作系统中基本上都是一样的，包括从 PE 格式二进制文件启动的 Windows 进程。二进制文件中的每个部分都加载到它们各自的部分中，BSS 部分分配给指定的大小。`.text`部分与其他部分一起加载，并且一旦完成，将执行其初始指令，从而启动进程。

在诸如 C++之类的系统语言中，可以看到在这样的进程中，变量和其他程序状态信息是如何存储在堆栈（变量存在于作用域内）和堆（使用 new 运算符）中的。堆栈是内存的一部分（每个线程分配一个），其大小取决于操作系统及其配置。在创建新线程时，通常也可以通过编程方式设置堆栈大小。

在操作系统中，一个进程由一块内存地址组成，其大小是恒定的，并受其内存指针的大小限制。对于 32 位操作系统，这将限制该块为 4GB。在这个虚拟内存空间中，操作系统分配了一个基本的堆栈和堆，两者都可以增长，直到所有内存地址都被耗尽，并且进程进一步尝试分配更多内存将被拒绝。

堆栈对于操作系统和硬件都是一个概念。本质上，它是一组所谓的堆栈帧（stack frames），每个堆栈帧由与任务的执行框架相关的变量、指令和其他数据组成。

在硬件术语中，堆栈是任务（x86）或进程状态（ARM）的一部分，这是处理器定义执行实例（程序或线程）的方式。这个硬件定义的实体包含了一个线程的整个执行状态。有关此内容的更多详细信息，请参见以下各节。

# x86（32 位和 64 位）中的任务

在 Intel IA-32 系统编程指南第 3A 卷中，任务定义如下：

“任务是处理器可以分派、执行和挂起的工作单元。它可以用于执行程序、任务或进程、操作系统服务实用程序、中断或异常处理程序，或内核或执行实用程序。”

“IA-32 架构提供了一种保存任务状态、分派任务执行以及从一个任务切换到另一个任务的机制。在保护模式下，所有处理器执行都是从一个任务中进行的。即使是简单的系统也必须定义至少一个任务。更复杂的系统可以使用处理器的任务管理设施来支持多任务应用程序。”

IA-32（Intel x86）手册中的这段摘录总结了硬件如何支持和实现对操作系统、进程以及这些进程之间的切换的支持。

在这里重要的是要意识到，对于处理器来说，没有进程或线程这样的东西。它所知道的只是执行线程，定义为一系列指令。这些指令被加载到内存的某个地方，并且当前位置在这些指令中以及正在创建的变量数据（变量）的跟踪，作为应用程序在进程的数据部分中执行。

每个任务还在硬件定义的保护环中运行，OS 的任务通常在环 0 上运行，用户任务在环 3 上运行。环 1 和 2 很少使用，除了在 x86 架构的现代操作系统中的特定用例。这些环是硬件强制执行的特权级别，例如严格分离内核和用户级任务。

32 位和 64 位任务的任务结构在概念上非常相似。它的官方名称是**任务状态结构**（**TSS**）。它对 32 位 x86 CPU 的布局如下：

![](img/fdb56c2a-af43-4d41-b70a-c98b2b018900.png)

以下是字段：

+   SS0：第一个堆栈段选择器字段

+   **ESP0**：第一个 SP 字段

对于 64 位 x86_64 CPU，TSS 布局看起来有些不同，因为在这种模式下不支持基于硬件的任务切换：

![](img/94cc164c-5fd6-4eda-b974-1f7ba35c245c.png)

在这里，我们有类似的相关字段，只是名称不同：

+   **RSPn**：特权级别 0 到 2 的 SP

+   **ISTn**：中断堆栈表指针

即使在 32 位模式下，x86 CPU 支持任务之间的硬件切换，大多数操作系统仍然会在每个 CPU 上使用单个 TSS 结构，而不管模式如何，并且在软件中实际执行任务之间的切换。这在一定程度上是由于效率原因（仅交换变化的指针），部分原因是由于只有这种方式才可能的功能，例如测量进程/线程使用的 CPU 时间，并调整线程或进程的优先级。在软件中执行这些操作还简化了代码在 64 位和 32 位系统之间的可移植性，因为前者不支持基于硬件的任务切换。

在软件基础的任务切换（通常通过中断）期间，ESP/RSP 等存储在内存中，并用下一个计划任务的值替换。这意味着一旦执行恢复，TSS 结构现在将具有新任务的**堆栈指针**（**SP**），段指针，寄存器内容和所有其他细节。

中断的来源可以是硬件或软件。硬件中断通常由设备使用，以向 CPU 发出它们需要操作系统关注的信号。调用硬件中断的行为称为中断请求，或 IRQ。

软件中断可能是由 CPU 本身的异常条件引起的，也可能是 CPU 指令集的特性。操作系统内核通过触发软件中断来执行任务切换的操作。

# ARM 中的进程状态

在 ARM 架构中，应用程序通常在非特权的**异常级别 0**（**EL0**）级别上运行，这与 x86 架构上的 ring 3 相当，而 OS 内核在 EL1 级别上。ARMv7（AArch32，32 位）架构将 SP 放在通用寄存器 13 中。对于 ARMv8（AArch64，64 位），每个异常级别都实现了专用的 SP 寄存器：`SP_EL0`，`SP_EL1`等。

对于 ARM 架构的任务状态，使用**程序状态寄存器**（**PSR**）实例来表示**当前程序状态寄存器**（**CPSR**）或**保存的程序状态寄存器**（**SPSR**）的程序状态寄存器。PSR 是**进程状态**（**PSTATE**）的一部分，它是进程状态信息的抽象。

虽然 ARM 架构与 x86 架构有很大不同，但在使用基于软件的任务切换时，基本原则并未改变：保存当前任务的 SP，寄存器状态，并在恢复处理之前将下一个任务的详细信息放在其中。

# 堆栈

正如我们在前面的部分中看到的，堆栈与 CPU 寄存器一起定义了一个任务。正如前面提到的，这个堆栈由堆栈帧组成，每个堆栈帧定义了该特定任务执行实例的（局部）变量，参数，数据和指令。值得注意的是，尽管堆栈和堆栈帧主要是软件概念，但它是任何现代操作系统的重要特性，在许多 CPU 指令集中都有硬件支持。从图形上看，可以像下面这样进行可视化：

![](img/88a7ab24-ba2b-42c1-97b3-78275eccd380.png)

SP（x86 上的 ESP）指向堆栈顶部，另一个指针（x86 上的**扩展基指针**（**EBP**））。每个帧包含对前一个帧的引用（调用者返回地址），由操作系统设置。

当使用调试器调试 C++应用程序时，当请求回溯时，基本上就是看到了堆栈的各个帧--显示了一直到当前帧的初始堆栈帧。在这里，可以检查每个单独帧的细节。

# 多线程定义

在过去的几十年中，与计算机处理任务方式相关的许多不同术语已经被创造并广泛使用。其中许多术语也被交替使用，正确与否。其中一个例子是多线程与多处理的比较。

在这里，后者意味着在具有多个物理处理器的系统中每个处理器运行一个任务，而前者意味着在单个处理器上同时运行多个任务，从而产生它们都在同时执行的错觉：

![](img/d8f34726-99a5-498c-a2fe-e7c50f6de467.png)

多处理和多任务之间的另一个有趣区别是，后者使用时间片来在单个处理器核心上运行多个线程。这与多线程不同，因为在多任务系统中，没有任务会在同一个 CPU 核心上并发运行，尽管任务仍然可以被中断。

从软件的角度来看，进程和进程内的线程之间共享的内存空间的概念是多线程系统的核心。尽管硬件通常不知道这一点--只看到操作系统中的单个任务。但是，这样的多线程进程包含两个或多个线程。然后，每个线程执行自己的一系列任务。

在其他实现中，例如英特尔的 x86 处理器上的**超线程**（**HT**），这种多线程是在硬件中实现的，通常被称为 SMT（有关详细信息，请参见*同时多线程（SMT）*部分）。当启用 HT 时，每个物理 CPU 核心被呈现给操作系统为两个核心。硬件本身将尝试同时执行分配给这些所谓的虚拟核心的任务，调度可以同时使用处理核心的不同元素的操作。实际上，这可以在不需要操作系统或应用程序进行任何类型的优化的情况下显着提高性能。

当然，操作系统仍然可以进行自己的调度，以进一步优化任务的执行，因为硬件对执行的指令的许多细节并不了解。

启用 HT 的外观如下所示：

![](img/6f9d96fd-abee-4e3b-9ab3-08c46e683f2f.png)

在上述图形中，我们看到内存（RAM）中四个不同任务的指令。其中两个任务（线程）正在同时执行，CPU 的调度程序（在前端）试图安排指令，以便尽可能多地并行执行指令。在这种情况下不可能的情况下，执行硬件空闲时会出现所谓的流水线气泡（白色）。

与内部 CPU 优化一起，这导致指令的吞吐量非常高，也称为**每秒指令数**（**IPC**）。与 CPU 的 GHz 评级不同，这个 IPC 数字通常更重要，用于确定 CPU 的性能。

# 弗林分类法

不同类型的计算机架构使用一种系统进行分类，这个系统最早是由迈克尔·J·弗林在 1966 年提出的。这个分类系统有四个类别，根据处理硬件的输入和输出流的数量来定义处理硬件的能力：

+   **单指令，单数据**（**SISD**）：获取单个指令来操作单个数据流。这是 CPU 的传统模型。

+   **单指令，多数据**（**SIMD**）：使用这种模型，单个指令可以并行操作多个数据流。这是矢量处理器（如**图形处理单元**（**GPU**））使用的模型。

+   **多指令，单数据**（**MISD**）：这种模型最常用于冗余系统，通过不同的处理单元对相同的数据执行相同的操作，最终验证结果以检测硬件故障。这通常由航空电子系统等使用。

+   **多指令，多数据**（**MIMD**）：对于这种模型，多处理系统非常适用。多个处理器上的多个线程处理多个数据流。这些线程不是相同的，就像 SIMD 的情况一样。

需要注意的一点是，这些类别都是根据多处理来定义的，这意味着它们指的是硬件的固有能力。使用软件技术，几乎可以在常规的 SISD 架构上近似任何方法。然而，这也是多线程的一部分。

# 对称与非对称多处理

在过去的几十年里，许多系统都包含了多个处理单元。这些可以大致分为对称多处理（SMP）和非对称多处理（AMP）系统。

AMP 的主要特征是第二个处理器作为外围连接到主 CPU。这意味着它不能运行控制软件，而只能运行用户应用程序。这种方法也被用于连接使用不同架构的 CPU，以允许在 Amiga、68k 系统上运行 x86 应用程序，例如。

在 SMP 系统中，每个 CPU 都是对等的，可以访问相同的硬件资源，并以合作的方式设置。最初，SMP 系统涉及多个物理 CPU，但后来，多个处理器核心集成在单个 CPU 芯片上：

![](img/58c33dca-c958-4806-beb3-15969b592057.png)

随着多核 CPU 的普及，SMP 是嵌入式开发之外最常见的处理类型，而在嵌入式开发中，单处理（单核，单处理器）仍然非常常见。

从技术上讲，系统中的声音、网络和图形处理器可以被认为是与 CPU 相关的非对称处理器。随着通用 GPU 处理的增加，AMP 变得更加相关。

# 松散和紧密耦合的多处理

多处理系统不一定要在单个系统内实现，而可以由多个连接在网络中的系统组成。这样的集群被称为松散耦合的多处理系统。我们将在第九章《分布式计算中的多线程》中介绍分布式计算。

这与紧密耦合的多处理系统形成对比，紧密耦合的多处理系统是指系统集成在单个印刷电路板（PCB）上，使用相同的低级别、高速总线或类似的方式。

# 将多处理与多线程结合

几乎任何现代系统都结合了多处理和多线程，这要归功于多核 CPU，它在单个处理器芯片上结合了两个或更多处理核心。对于操作系统来说，这意味着它必须在多个处理核心之间调度任务，同时也必须在特定核心上调度它们，以提取最大性能。

这是任务调度器的领域，我们将在一会儿看一下。可以说这是一个值得一本书的话题。

# 多线程类型

与多处理一样，多线程也不是单一的实现，而是两种主要的实现。它们之间的主要区别是处理器在单个周期内可以同时执行的线程的最大数量。多线程实现的主要目标是尽可能接近 100%的处理器硬件利用率。多线程利用线程级和进程级并行性来实现这一目标。

有两种类型的多线程，我们将在以下部分进行介绍。

# 时间多线程

也被称为超线程，时间多线程（TMT）的主要子类型是粗粒度和细粒度（或交错）。前者在不同任务之间快速切换，保存每个任务的上下文，然后切换到另一个任务的上下文。后者在每个周期中切换任务，导致 CPU 流水线包含来自各种任务的指令，从中得到“交错”这个术语。

细粒度类型是在桶处理器中实现的。它们比 x86 和其他架构有优势，因为它们可以保证特定的定时（对于硬实时嵌入式系统很有用），而且由于可以做出一些假设，实现起来更不复杂。

# 同时多线程（SMT）

SMT 是在超标量 CPU 上实现的（实现指令级并行性），其中包括 x86 和 ARM 架构。SMT 的定义特征也由其名称指出，特别是其能够在每个核心中并行执行多个线程的能力。

通常，每个核心有两个线程是常见的，但一些设计支持每个核心最多八个并发线程。这样做的主要优势是能够在线程之间共享资源，明显的缺点是多个线程的冲突需求，这必须加以管理。另一个优势是由于缺乏硬件资源复制，使得结果 CPU 更加节能。

英特尔的 HT 技术本质上是英特尔的 SMT 实现，从 2002 年的一些奔腾 4 CPU 开始提供基本的双线程 SMT 引擎。

# 调度程序

存在许多任务调度算法，每个算法都专注于不同的目标。有些可能寻求最大化吞吐量，其他人则寻求最小化延迟，而另一些可能寻求最大化响应时间。哪种调度程序是最佳选择完全取决于系统所用于的应用。

对于桌面系统，调度程序通常尽可能保持通用，通常优先处理前台应用程序，以便为用户提供最佳的桌面体验。

对于嵌入式系统，特别是在实时工业应用中，通常会寻求保证定时。这允许进程在恰好正确的时间执行，这在驱动机械、机器人或化工过程中至关重要，即使延迟几毫秒也可能造成巨大成本甚至是致命的。

调度程序类型还取决于操作系统的多任务状态——合作式多任务系统无法提供关于何时可以切换运行中进程的许多保证，因为这取决于活动进程何时让出。

使用抢占式调度程序，进程在不知情的情况下进行切换，允许调度程序更多地控制进程在哪个时间点运行。

基于 Windows NT 的操作系统（Windows NT、2000、XP 等）使用所谓的多级反馈队列，具有 32 个优先级级别。这种类型的优先级调度程序允许优先处理某些任务，从而使结果体验得到精细调整。

Linux 最初（内核 2.4）也使用了基于多级反馈队列的优先级调度程序，类似于 Windows NT 的 O(n)调度程序。从 2.6 版本开始，这被 O(1)调度程序取代，允许在恒定时间内安排进程。从 Linux 内核 2.6.23 开始，默认调度程序是**完全公平调度程序**（**CFS**），它确保所有任务获得可比较的 CPU 时间份额。

以下是一些常用或知名操作系统使用的调度算法类型：

| **操作系统** | **抢占** | **算法** |
| --- | --- | --- |
| Amiga OS | 是 | 优先级轮转调度 |
| FreeBSD | 是 | 多级反馈队列 |
| Linux kernel 2.6.0 之前 | 是 | 多级反馈队列 |
| Linux kernel 2.6.0-2.6.23 | 是 | O(1)调度程序 |
| Linux kernel 2.6.23 之后 | 是 | 完全公平调度程序 |
| 经典 Mac OS 9 之前 | 无 | 合作调度程序 |
| Mac OS 9 | 一些 | 用于 MP 任务的抢占式调度程序，以及用于进程和线程的合作调度程序 |
| OS X/macOS | 是 | 多级反馈队列 |
| NetBSD | 是 | 多级反馈队列 |
| Solaris | 是 | 多级反馈队列 |
| Windows 3.1x | 无 | 合作调度程序 |
| Windows 95, 98, Me | Half | 32 位进程的抢占式调度程序，16 位进程的协作式调度程序 |
| Windows NT（包括 2000、XP、Vista、7 和 Server） | 是 | 多级反馈队列 |

（来源：[`en.wikipedia.org/wiki/Scheduling_(computing)`](https://en.wikipedia.org/wiki/Scheduling_(computing)））

抢占式列指示调度程序是否是抢占式的，下一列提供了更多细节。可以看到，抢占式调度程序非常常见，并且被所有现代桌面操作系统使用。

# 跟踪演示应用程序

在第一章“重温多线程”的演示代码中，我们看了一个简单的`c++11`应用程序，它使用四个线程来执行一些处理。在本节中，我们将从硬件和操作系统的角度来看同一个应用程序。

当我们查看`main`函数中代码的开头时，我们看到我们创建了一个包含单个（整数）值的数据结构：

```cpp
int main() {
    values.push_back(42);
```

在操作系统创建新任务和相关的堆栈结构之后，堆栈上分配了一个向量数据结构的实例（针对整数类型进行了定制）。这个大小在二进制文件的全局数据部分（ELF 的 BSS）中指定。

当应用程序使用其入口函数（默认为`main()`）启动执行时，数据结构被修改为包含新的整数值。

接下来，我们创建四个线程，为每个线程提供一些初始数据：

```cpp
    thread tr1(threadFnc, 1);
    thread tr2(threadFnc, 2);
    thread tr3(threadFnc, 3);
    thread tr4(threadFnc, 4);
```

对于操作系统来说，这意味着创建新的数据结构，并为每个新线程分配一个堆栈。对于硬件来说，如果不使用基于硬件的任务切换，最初不会改变任何东西。

在这一点上，操作系统的调度程序和 CPU 可以结合起来尽可能高效和快速地执行这组任务（线程），利用硬件的特性，包括 SMP、SMT 等等。

之后，主线程等待，直到其他线程停止执行：

```cpp
    tr1.join();
    tr2.join();
    tr3.join();
    tr4.join();
```

这些是阻塞调用，它们标记主线程被阻塞，直到这四个线程（任务）完成执行。在这一点上，操作系统的调度程序将恢复主线程的执行。

在每个新创建的线程中，我们首先在标准输出上输出一个字符串，确保我们锁定互斥锁以确保同步访问：

```cpp
void threadFnc(int tid) {
    cout_mtx.lock();
    cout << "Starting thread " << tid << ".n";
    cout_mtx.unlock();
```

互斥锁本质上是一个存储在堆栈或堆上的单个值，然后使用原子操作访问。这意味着需要某种形式的硬件支持。使用这个，任务可以检查它是否被允许继续，或者必须等待并再次尝试。

在代码的最后一个特定部分，这个互斥锁允许我们在标准的 C++输出流上输出，而不会受到其他线程的干扰。

之后，我们将向量中的初始值复制到一个局部变量中，再次确保它是同步完成的：

```cpp
    values_mtx.lock();
    int val = values[0];
    values_mtx.unlock();
```

在这里发生的事情是一样的，只是现在互斥锁允许我们读取向量中的第一个值，而不会在我们使用它时冒险另一个线程访问或甚至更改它。

接下来是生成随机数如下：

```cpp
    int rval = randGen(0, 10);
    val += rval;
```

这使用了`randGen()`方法，如下所示：

```cpp
int randGen(const int& min, const int& max) {
    static thread_local mt19937 generator(hash<thread::id>() (this_thread::get_id()));
    uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}
```

这种方法之所以有趣，是因为它使用了线程本地变量。线程本地存储是线程特定的内存部分，用于全局变量，但必须保持限制在特定线程内。

这对于像这里使用的静态变量非常有用。`generator`实例是静态的，因为我们不希望每次使用这个方法时都重新初始化它，但我们也不希望在所有线程之间共享这个实例。通过使用线程本地的静态实例，我们可以实现这两个目标。静态实例被创建和使用，但对于每个线程是分开的。

`Thread`函数最后以相同的一系列互斥锁结束，并将新值复制到数组中。

```cpp
    cout_mtx.lock();
    cout << "Thread " << tid << " adding " << rval << ". New value: " << val << ".n";
    cout_mtx.unlock();

    values_mtx.lock();
    values.push_back(val);
    values_mtx.unlock();
}
```

在这里，我们看到对标准输出流的同步访问，然后是对值数据结构的同步访问。

# 互斥锁实现

互斥排斥是多线程应用程序中数据的线程安全访问的基本原则。可以在硬件和软件中都实现这一点。**互斥排斥**（**mutex**）是大多数实现中这种功能的最基本形式。

# 硬件

在单处理器（单处理器核心），非 SMT 系统上最简单的基于硬件的实现是禁用中断，从而防止任务被更改。更常见的是采用所谓的忙等待原则。这是互斥锁背后的基本原理--由于处理器如何获取数据，只有一个任务可以获取和读/写共享内存中的原子值，这意味着，一个变量的大小与 CPU 的寄存器相同（或更小）。这在第十五章中进一步详细说明，*原子操作 - 与硬件一起工作*。

当我们的代码尝试锁定互斥锁时，它所做的是读取这样一个原子内存区域的值，并尝试将其设置为其锁定值。由于这是一个单操作，因此在任何给定时间只有一个任务可以更改该值。其他任务将不得不等待，直到它们可以在这个忙等待周期中获得访问，如图所示：

![](img/9abfe9d9-ee51-4508-b46d-cbb6ac2c97c1.png)

# 软件

软件定义的互斥锁实现都基于忙等待。一个例子是**Dekker**算法，它定义了一个系统，其中两个进程可以同步，利用忙等待等待另一个进程离开临界区。

该算法的伪代码如下：

```cpp
    variables
        wants_to_enter : array of 2 booleans
        turn : integer

    wants_to_enter[0] ← false
    wants_to_enter[1] ← false
    turn ← 0 // or 1
p0:
    wants_to_enter[0] ← true
    while wants_to_enter[1] {
        if turn ≠ 0 {
            wants_to_enter[0] ← false
            while turn ≠ 0 {
                // busy wait
            }
            wants_to_enter[0] ← true
        }
    }
    // critical section
    ...
    turn ← 1
    wants_to_enter[0] ← false
    // remainder section
p1:
    wants_to_enter[1] ← true
    while wants_to_enter[0] {
        if turn ≠ 1 {
            wants_to_enter[1] ← false
            while turn ≠ 1 {
                // busy wait
            }
            wants_to_enter[1] ← true
        }
    }
    // critical section
    ...
    turn ← 0
    wants_to_enter[1] ← false
    // remainder section
```

(引用自：[`en.wikipedia.org/wiki/Dekker's_algorithm`](https://en.wikipedia.org/wiki/Dekker's_algorithm))

在前面的算法中，进程指示意图进入临界区，检查是否轮到它们（使用进程 ID），然后在它们进入后将它们的意图设置为 false。只有一旦进程再次将其意图设置为 true，它才会再次进入临界区。如果它希望进入，但`turn`与其进程 ID 不匹配，它将忙等待，直到条件变为真。

基于软件的互斥排斥算法的一个主要缺点是，它们只在禁用**乱序**（**OoO**）执行代码时才起作用。OoO 意味着硬件积极重新排序传入的指令，以优化它们的执行，从而改变它们的顺序。由于这些算法要求各种步骤按顺序执行，因此它们不再适用于 OoO 处理器。

# 并发性

每种现代编程语言都支持并发性，提供高级 API，允许同时执行许多任务。C++支持并发性，从 C++11 开始，更复杂的 API 在 C++14 和 C++17 中进一步添加。虽然 C++线程支持库允许多线程，但它需要编写复杂的同步代码；然而，并发性让我们能够执行独立的任务--甚至循环迭代可以在不编写复杂代码的情况下并发运行。底线是，并行化通过并发性变得更加容易。

并发支持库是 C++线程支持库的补充。这两个强大库的结合使用使并发编程在 C++中变得更加容易。

让我们在以下名为`main.cpp`的文件中使用 C++并发编写一个简单的`Hello World`程序：

```cpp
#include <iostream>
#include <future>
using namespace std;

void sayHello( ) {
  cout << endl << "Hello Concurrency support library!" << endl;
}

int main ( ) {
  future<void> futureObj = async ( launch::async, sayHello );
  futureObj.wait( );

  return 0;
}
```

让我们试着理解`main()`函数。Future 是并发模块的一个对象，它帮助调用者函数以异步的方式检索线程传递的消息。`future<void>`中的 void 表示`sayHello()`线程函数不会向调用者即`main`线程函数传递任何消息。`async`类让我们以`launch::async`或`launch::deferred`模式执行函数。

`launch::async`模式允许`async`对象在单独的线程中启动`sayHello()`方法，而`launch::deferred`模式允许`async`对象在不创建单独线程的情况下调用`sayHello()`函数。在`launch::deferred`模式下，直到调用者线程调用`future::get()`方法之前，`sayHello()`方法的调用将不同。

`futureObj.wait()`方法用于阻塞主线程，让`sayHello()`函数完成其任务。`future::wait()`函数类似于线程支持库中的`thread::join()`。

# 如何编译和运行

让我们继续使用以下命令编译程序：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

让我们启动`concurrency.exe`，并了解它的工作原理：

![](img/1d316a09-3388-4996-b3b0-22c8fe6f15ea.png)

# 使用并发支持库进行异步消息传递

让我们稍微修改`main.cpp`，即我们在上一节中编写的 Hello World 程序。让我们了解如何从`Thread`函数异步地向调用者函数传递消息：

```cpp
#include <iostream>
#include <future>
using namespace std;

void sayHello( promise<string> promise_ ) {
  promise_.set_value ( "Hello Concurrency support library!" );
}

int main ( ) {
  promise<string> promiseObj;

  future<string> futureObj = promiseObj.get_future( );
  async ( launch::async, sayHello, move( promiseObj ) );
  cout << futureObj.get( ) << endl;

  return 0;
}
```

在上一个程序中，`promiseObj`被`sayHello()`线程函数用来异步地向主线程传递消息。注意`promise<string>`意味着`sayHello()`函数预期传递一个字符串消息，因此主线程检索`future<string>`。`future::get()`函数调用将被阻塞，直到`sayHello()`线程函数调用`promise::set_value()`方法。

然而，重要的是要理解`future::get()`只能被调用一次，因为在调用`future::get()`方法之后，相应的`promise`对象将被销毁。

你注意到了`std::move()`函数的使用吗？`std::move()`函数基本上将`promiseObj`的所有权转移给了`sayHello()`线程函数，因此在调用`std::move()`后，`promiseObj`不能从`main`线程中访问。

# 如何编译和运行

让我们继续使用以下命令编译程序：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

通过启动`concurrency.exe`应用程序来观察`concurrency.exe`应用程序的工作情况：

![](img/7276bc6e-0b48-4e8c-878b-987bff2ea20d.png)

正如你可能已经猜到的那样，这个程序的输出与我们之前的版本完全相同。但是这个程序的版本使用了 promise 和 future 对象，而之前的版本不支持消息传递。

# 并发任务

并发支持模块支持一个称为**task**的概念。任务是跨线程并发发生的工作。可以使用`packaged_task`类创建并发任务。`packaged_task`类方便地连接了`thread`函数、相应的 promise 和 feature 对象。

让我们通过一个简单的例子来了解`packaged_task`的用法。以下程序让我们有机会尝试一些函数式编程的味道，使用 lambda 表达式和函数：

```cpp
#include <iostream>
#include <future>
#include <promise>
#include <thread>
#include <functional>
using namespace std;

int main ( ) {
     packaged_task<int (int, int)>
        addTask ( [] ( int firstInput, int secondInput ) {
              return firstInput + secondInput;
     } );

     future<int> output = addTask.get_future( );
     addTask ( 15, 10 );

     cout << "The sum of 15 + 10 is " << output.get() << endl;
     return 0;
}
```

在之前展示的程序中，我创建了一个名为`addTask`的`packaged_task`实例。`packaged_task< int (int,int)>`实例意味着 add 任务将返回一个整数并接受两个整数参数：

```cpp
addTask ( [] ( int firstInput, int secondInput ) {
              return firstInput + secondInput;
}); 
```

前面的代码片段表明它是一个匿名定义的 lambda 函数。

有趣的是，`main.cpp`中的`addTask()`调用看起来像是普通的函数调用。`future<int>`对象是从`packaged_task`实例`addTask`中提取出来的，然后用于通过 future 对象实例检索`addTask`的输出，即`get()`方法。

# 如何编译和运行

让我们继续使用以下命令编译程序：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

让我们快速启动`concurrency.exe`并观察下面显示的输出：

![](img/f2649f2e-71d5-4bec-9889-a120e26d7844.png)

太棒了！您学会了如何在并发支持库中使用 lambda 函数。

# 使用带有线程支持库的任务

在上一节中，您学会了如何以一种优雅的方式使用`packaged_task`。我非常喜欢 lambda 函数。它们看起来很像数学。但并不是每个人都喜欢 lambda 函数，因为它们在一定程度上降低了可读性。因此，如果您不喜欢 lambda 函数，使用并发任务时不一定要使用 lambda 函数。在本节中，您将了解如何使用线程支持库的并发任务，如下所示：

```cpp
#include <iostream>
#include <future>
#include <thread>
#include <functional>
using namespace std;

int add ( int firstInput, int secondInput ) {
  return firstInput + secondInput;
}

int main ( ) {
  packaged_task<int (int, int)> addTask( add);

  future<int> output = addTask.get_future( );

  thread addThread ( move(addTask), 15, 10 );

  addThread.join( );

  cout << "The sum of 15 + 10 is " << output.get() << endl;

  return 0;
}
```

# 如何编译和运行

让我们继续使用以下命令编译程序：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

让我们启动`concurrency.exe`，如下截图所示，并了解前一个程序和当前版本之间的区别：

![](img/2c5aae5a-3b84-4b4e-b194-8ee35667ad7a.png)

是的，输出与上一节相同，因为我们只是重构了代码。

太棒了！您刚刚学会了如何将 C++线程支持库与并发组件集成。

# 将线程过程及其输入绑定到 packaged_task

在本节中，您将学习如何将`thread`函数及其相应的参数与`packaged_task`绑定。

让我们从上一节的代码中取出并修改以了解绑定功能，如下所示：

```cpp
#include <iostream>
#include <future>
#include <string>
using namespace std;

int add ( int firstInput, int secondInput ) {
  return firstInput + secondInput;
}

int main ( ) {

  packaged_task<int (int,int)> addTask( add );
  future<int> output = addTask.get_future();
  thread addThread ( move(addTask), 15, 10);
  addThread.join();
  cout << "The sum of 15 + 10 is " << output.get() << endl;
  return 0;
}
```

`std::bind()`函数将`thread`函数及其参数与相应的任务绑定。由于参数是预先绑定的，因此无需再次提供输入参数 15 或 10。这些是`packaged_task`在 C++中可以使用的一些便利方式。

# 如何编译和运行

让我们继续使用以下命令编译程序：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

让我们启动`concurrency.exe`，如下截图所示，并了解前一个程序和当前版本之间的区别：

![](img/e2d19c8b-835e-49ee-992e-d5c11b2d7b13.png)

恭喜！到目前为止，您已经学到了很多关于 C++并发的知识。

# 使用并发库处理异常

并发支持库还支持通过 future 对象传递异常。

让我们通过一个简单的例子了解异常并发处理机制，如下所示：

```cpp
#include <iostream>
#include <future>
#include <promise>
using namespace std;

void add ( int firstInput, int secondInput, promise<int> output ) {
  try {
         if ( ( INT_MAX == firstInput ) || ( INT_MAX == secondInput ) )
             output.set_exception( current_exception() ) ;
        }
  catch(...) {}

       output.set_value( firstInput + secondInput ) ;

}

int main ( ) {

     try {
    promise<int> promise_;
          future<int> output = promise_.get_future();
    async ( launch::deferred, add, INT_MAX, INT_MAX, move(promise_) );
          cout << "The sum of INT_MAX + INT_MAX is " << output.get ( ) << endl;
     }
     catch( exception e ) {
  cerr << "Exception occured" << endl;
     }
}

```

就像我们将输出消息传递给调用者函数/线程一样，并发支持库还允许您设置任务或异步函数中发生的异常。当调用线程调用`future::get()`方法时，相同的异常将被抛出，因此异常通信变得很容易。

# 如何编译和运行

让我们继续使用以下命令编译程序。叔叔水果和尤达的麦芽：

```cpp
g++ main.cpp -o concurrency.exe -std=c++17 -lpthread
```

![](img/98871bec-2f2e-40dc-bd84-088519ea8196.png)

# 您学到了什么？

让我总结一下要点：

+   并发支持库提供了高级组件，可以并发执行多个任务

+   Future 对象让调用线程检索异步函数的输出

+   承诺对象由异步函数用于设置输出或异常

+   `FUTURE`和`PROMISE`对象的类型必须与异步函数设置的值的类型相同

+   并发组件可以与 C++线程支持库无缝结合使用

+   Lambda 函数和表达式可以与并发支持库一起使用

# 总结

在本章中，我们看到了进程和线程是如何在操作系统和硬件中实现的。我们还研究了处理器硬件的各种配置以及调度中涉及的操作系统元素，以了解它们如何提供各种类型的任务处理。

最后，我们拿上一章的多线程程序示例，再次运行它，这次考虑的是在执行过程中操作系统和处理器发生了什么。

在下一章中，我们将看一下通过操作系统和基于库的实现提供的各种多线程 API，以及比较这些 API 的示例。
