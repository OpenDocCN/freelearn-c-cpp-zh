# 11

# 并发

在上一章中涵盖了惰性求值和代理对象之后，我们现在将探讨如何使用共享内存在 C++中编写并发程序。我们将探讨如何通过编写没有数据竞争和死锁的程序来使并发程序正确运行。本章还将包含关于如何使并发程序以低延迟和高吞吐量运行的建议。

在继续之前，你应该知道本章不是并发编程的完整介绍，也不会涵盖 C++中所有并发的细节。相反，本章是 C++中编写并发程序的核心构建块的介绍，结合了一些与性能相关的指导方针。如果你以前没有编写过并发程序，最好通过一些入门材料来了解并发编程的理论方面。死锁、临界区、条件变量和互斥锁等概念将会被简要讨论，但这将更像是一个复习而不是对概念的彻底介绍。

本章涵盖以下内容：

+   并发编程的基础知识，包括并行执行、共享内存、数据竞争和死锁

+   C++线程支持库、原子库和 C++内存模型的介绍

+   无锁编程的简短示例

+   性能指南

# 理解并发编程的基础知识

并发程序可以同时执行多个任务。并发编程一般比顺序编程更难，但有几个原因可以使程序从并发中受益：

+   **效率**：今天的智能手机和台式电脑都有多个 CPU 核心，可以并行执行多个任务。如果你成功地将一个大任务分割成可以并行运行的子任务，理论上可以将大任务的运行时间除以 CPU 核心数。对于在单核机器上运行的程序，如果一个任务是 I/O 绑定的，仍然可以获得性能上的提升。当一个子任务在等待 I/O 时，其他子任务仍然可以在 CPU 上执行有用的工作。

+   **响应性和低延迟环境**：对于具有图形用户界面的应用程序，重要的是永远不要阻塞 UI，以免应用程序变得无响应。为了防止无响应，通常会让长时间运行的任务（如从磁盘加载文件或从网络获取数据）在单独的后台线程中执行，以便负责 UI 的线程永远不会被长时间运行的任务阻塞。低延迟很重要的另一个例子是实时音频。负责生成音频数据缓冲区的函数在单独的高优先级线程中执行，而程序的其余部分可以在低优先级线程中运行，以处理 UI 等。

+   **模拟**：并发可以使模拟现实世界中并发系统变得更容易。毕竟，我们周围的大多数事情都是同时发生的，有时很难用顺序编程模型来建模并发流。本书不会专注于模拟，而是专注于并发的性能相关方面。

并发为我们解决了许多问题，但也引入了新问题，接下来我们将讨论这些问题。

# 并发编程为何如此困难？

有许多原因使并发编程变得困难，如果你以前编写过并发程序，你很可能已经遇到了以下列出的原因：

+   以安全的方式在多个线程之间共享状态是困难的。每当我们有可以同时读写的数据时，我们需要一些方法来保护这些数据免受数据竞争的影响。稍后你将看到许多这样的例子。

+   由于多个并行执行流，并发程序通常更难推理。

+   并发使调试变得复杂。由于数据竞争而导致的错误可能非常难以调试，因为它们依赖于线程的调度方式。这类错误很难复现，并且在最坏的情况下，甚至在使用调试器运行程序时可能会消失。有时，对控制台的无辜调试跟踪可能会改变多线程程序的行为方式，并使错误暂时消失。你已经被警告了！

在我们开始使用 C++进行并发编程之前，将介绍一些与并发和并行编程相关的一般概念。

# 并发和并行

**并发**和**并行**是有时可以互换使用的两个术语。然而，它们并不相同，重要的是要理解它们之间的区别。如果程序在重叠的时间段内具有多个单独的控制流运行，则称其并发运行。在 C++中，每个单独的控制流由一个线程表示。这些线程可能会或可能不会同时执行。如果它们同时执行，就称为并行执行。要使并发程序并行运行，需要在支持指令并行执行的机器上执行它；也就是说，具有多个 CPU 核心的机器。

乍一看，似乎很明显我们总是希望并发程序尽可能并行运行，出于效率原因。然而，这并不一定总是正确的。本章涵盖的许多同步原语（如互斥锁）仅需要支持线程的并行执行。不在并行运行的并发任务不需要相同的锁定机制，可能更容易推理。

## 时间片

你可能会问，“在只有一个 CPU 核心的机器上如何执行并发线程？”答案是**时间片**。这是操作系统用来支持进程并发执行的相同机制。为了理解时间片，让我们假设我们有两个应该同时执行的独立指令序列，如下图所示：

！[](img/B15619_11_01.png)

图 11.1：两个独立的指令序列

编号的方框表示指令。每个指令序列在一个单独的线程中执行，标记为**T1**和**T2**。操作系统将安排每个线程在 CPU 上有一定的时间，然后执行上下文切换。上下文切换将存储正在运行的线程的当前状态，并加载应该执行的线程的状态。这样做的频率足够高，以至于看起来好像线程在同时运行。然而，上下文切换是耗时的，并且每次新线程在 CPU 核心上执行时很可能会产生大量的缓存未命中。因此，我们不希望上下文切换发生得太频繁。

下图显示了两个线程在单个 CPU 上调度的可能执行顺序：

！[](img/B15619_11_02.png)

图 11.2：两个线程的可能执行。点表示上下文切换

T1 线程的第一条指令开始执行，然后进行上下文切换，让 T2 线程执行前两条指令。作为程序员，我们必须确保程序可以按预期运行，无论操作系统调度程序如何调度任务。如果某个序列因某种原因无效，有方法可以通过使用锁来控制指令执行的顺序，这将在后面介绍。

如果一台机器有多个 CPU 核心，就有可能并行执行两个线程。然而，并没有保证（甚至是不太可能）这两个线程在程序的整个生命周期中都会在各自的核心上执行。整个系统共享 CPU 的时间，所以调度程序也会让其他进程执行。这就是为什么线程不会被调度到专用核心上的原因之一。

*图 11.3*显示了相同的两个线程的执行情况，但现在它们在一个有两个 CPU 核心的机器上运行。正如你所看到的，第一个线程的第二和第三条指令（白色框）与另一个线程同时执行 - 两个线程在并行执行：

![](img/B15619_11_03.png)

图 11.3：两个线程在多核机器上执行。这使得两个线程可以并行执行。

接下来让我们讨论共享内存。

## 共享内存

在同一进程中创建的线程共享相同的虚拟内存。这意味着一个线程可以访问进程内可寻址的任何数据。操作系统使用虚拟内存在进程之间保护内存，但对于意外访问进程内未打算在不同线程之间共享的内存，操作系统不会提供保护。虚拟内存只保护我们免受访问分配给我们自己的不同进程中的内存的影响。

在多个线程之间共享内存可以是处理线程间通信的一种非常有效的方式。然而，在 C++中编写并发程序时，以安全的方式在线程之间共享内存是一个主要挑战之一。我们应该始终努力将线程之间共享的资源数量最小化。

幸运的是，并非所有内存默认都是共享的。每个线程都有自己的堆栈，用于存储本地变量和处理函数调用所需的其他数据。除非一个线程将本地变量的引用或指针传递给其他线程，否则其他线程将无法访问该线程的堆栈。这是尽可能使用堆栈的另一个原因（如果你在阅读*第七章*，*内存管理*后还不相信堆栈是一个好地方存储数据）。

还有**线程本地存储**，有时缩写为**TLS**，它可以用来存储在线程上下文中是全局的，但在不同线程之间不共享的变量。线程本地变量可以被视为每个线程都有自己副本的全局变量。

其他所有内容默认情况下都是共享的；即堆上分配的动态内存、全局变量和静态局部变量。每当你有被某个线程改变的共享数据时，你需要确保没有其他线程同时访问该数据，否则就会出现数据竞争。

还记得*第七章*，*内存管理*中*进程内存*部分的图示吗？这里再次展示，但修改后显示了当一个进程包含多个线程时的情况。如下图所示，每个线程都有自己的堆栈内存，但所有线程只有一个堆：

![](img/B15619_11_04.png)

图 11.4：进程的虚拟地址空间的可能布局

在这个例子中，进程包含三个线程。堆内存默认情况下被所有线程共享。

## 数据竞争

**数据竞争**发生在两个线程同时访问同一内存且至少一个线程正在改变数据时。如果你的程序有数据竞争，这意味着你的程序有未定义的行为。编译器和优化器会*假设*你的代码中没有数据竞争，并在这个假设下对其进行优化。这可能导致崩溃或其他完全令人惊讶的行为。换句话说，你绝对不能允许程序中出现数据竞争。编译器通常不会在编译时警告你有数据竞争，因为它们很难在编译时检测到。

调试数据竞争可能是一个真正的挑战，有时需要像**ThreadSanitizer**（来自 Clang）或**Concurrency Visualizer**（Visual Studio 扩展）这样的工具。这些工具通常会对代码进行插装，以便运行时库可以在调试程序运行时检测、警告或可视化潜在的数据竞争。

### 例子：数据竞争

*图 11.5*显示了两个线程要更新一个名为`counter`的整数。想象一下，这些线程都在使用指令`++counter`来增加一个全局计数器变量。事实证明，增加一个`int`可能涉及多个 CPU 指令。这可以在不同的 CPU 上以不同的方式完成，但假设`++counter`生成以下虚构的机器指令：

+   **R**：从内存中读取 counter

+   **+1**：增加 counter

+   **W**：将新的 counter 值写入内存

现在，如果我们有两个线程要更新`counter`的值，初始值为 42，我们期望在这两个线程运行后它变成 44。然而，如下图所示，没有保证指令会按顺序执行以确保`counter`变量的正确增加。

![](img/B15619_11_05.png)

图 11.5：两个线程都在增加相同的共享变量

没有数据竞争，counter 本应该达到值 44，但实际上只达到了 43。

在这个例子中，两个线程都读取值 42 并将该值增加到 43。然后，它们都写入新值 43，这意味着我们永远不会达到正确的答案 44。如果第一个线程能够在下一个线程开始读取之前写入值 43，我们最终会得到 44。还要注意，即使只有一个 CPU 核心，这也是可能的。调度程序可以以类似的方式安排这两个线程，以便在任何写入之前执行两个读取指令。

再次强调，这只是一种可能的情况，但重要的是行为是未定义的。当程序存在数据竞争时，任何事情都可能发生。其中一个例子是**tearing**，这是**torn reads**和**torn writes**的常用术语。当一个线程在另一个线程同时读取值时向内存写入值的部分，因此最终得到一个损坏的值时，就会发生这种情况。

### 避免数据竞争

我们如何避免数据竞争？有两个主要选项：

+   使用原子数据类型而不是`int`。这将告诉编译器以原子方式执行读取、增加和写入。我们将在本章后面花更多时间讨论原子数据类型。

+   使用互斥锁（mutex）来保证多个线程永远不会同时执行关键部分。**关键部分**是代码中不得同时执行的地方，因为它更新或读取可能会产生数据竞争的共享内存。

值得强调的是，不可变数据结构——永远不会改变的数据结构——可以被多个线程访问而不会有任何数据竞争的风险。减少可变对象的使用有很多好处，但在编写并发程序时变得更加重要。一个常见的模式是总是创建新的不可变对象，而不是改变现有对象。当新对象完全构建并表示新状态时，它可以与旧对象交换。这样，我们可以最小化代码的关键部分。只有交换是一个关键部分，因此需要通过原子操作或互斥体来保护。

## 互斥体

**互斥锁**，简称**互斥锁**，是用于避免数据竞争的同步原语。需要进入临界区的线程首先需要锁定互斥锁（有时锁定也称为获取互斥锁）。这意味着在持有锁的第一个线程解锁互斥锁之前，没有其他线程可以锁定相同的互斥锁。这样，互斥锁保证一次只有一个线程在临界区内部。

在*图 11.6*中，您可以看到在*数据竞争示例*部分演示的竞争条件是如何通过使用互斥锁来避免的。标记为**L**的指令是锁定指令，标记为**U**的指令是解锁指令。在核心 0 上执行的第一个线程首先到达临界区并在读取计数器的值之前锁定互斥锁。然后，它将 1 添加到计数器并将其写回内存。之后，它释放锁。

第二个线程，在核心 1 上执行，在第一个线程获取互斥锁后立即到达临界区。由于互斥锁已经被锁定，线程被阻塞，直到第一个线程无干扰地更新计数器并释放互斥锁：

![](img/B15619_11_06.png)

图 11.6：互斥锁保护临界区，避免计数器变量的数据竞争

结果是，两个线程可以以安全和正确的方式更新可变的共享变量。然而，这也意味着这两个线程不能再并行运行。如果一个线程大部分工作都不能在不串行化的情况下完成，从性能的角度来看，使用线程就没有意义了。

第二个线程被第一个线程阻塞的状态称为**争用**。这是我们努力最小化的东西，因为它会影响并发程序的可伸缩性。如果争用程度很高，增加 CPU 核心数量将不会提高性能。

## 死锁

使用互斥锁保护共享资源时，存在陷入**死锁**状态的风险。当两个线程互相等待对方释放锁时，就会发生死锁。两个线程都无法继续进行，它们陷入了死锁状态。死锁发生的一个条件是，已经持有一个锁的线程尝试获取另一个锁。当系统增长并变得更大时，跟踪系统中所有线程可能使用的所有锁变得越来越困难。这是始终努力最小化使用共享资源的一个原因，也说明了对独占锁的需求。

*图 11.7*显示了两个线程处于等待状态，试图获取另一个线程持有的锁：

![](img/B15619_11_07.png)

图 11.7：死锁状态的示例

接下来让我们讨论同步和异步任务。

## 同步和异步任务

在本章中，我将提到**同步任务**和**异步任务**。同步任务就像普通的 C++函数。当同步任务完成其任务后，它将控制权返回给任务的调用者。任务的调用者在等待或被阻塞，直到同步任务完成。

另一方面，异步任务将立即将控制权返回给调用者，并同时执行其工作。

*图 11.8*中的序列显示了分别调用同步和异步任务之间的区别：

![](img/B15619_11_08.png)

图 11.8：同步与异步调用。异步任务立即返回，但在调用者重新获得控制权后继续工作。

如果您以前没有见过异步任务，它们可能一开始看起来很奇怪，因为在 C++中，普通函数遇到返回语句或到达函数体末尾时总是停止执行。然而，异步 API 变得越来越常见，很可能您以前已经遇到过，例如在使用异步 JavaScript 时。

有时，我们使用术语**阻塞**来表示阻塞调用者的操作；也就是说，使调用者等待操作完成。

在对并发性进行了一般介绍之后，现在是时候探索 C++中的线程编程支持了。

# C++中的并发编程

C++中的并发支持使程序能够同时执行多个任务。正如前面提到的，编写正确的并发 C++程序通常比在一个线程中依次执行所有任务的程序要困难得多。本节还将演示一些常见的陷阱，以使您了解编写并发程序所涉及的所有困难。

并发支持首次出现在 C++11 中，并在 C++14、C++17 和 C++20 中得到扩展。在并发成为语言的一部分之前，它是通过操作系统的本机并发支持、**POSIX 线程**（**pthreads**）或其他一些库来实现的。

有了 C++语言中的并发支持，我们可以编写跨平台的并发程序，这很棒！然而，当处理平台上的并发时，有时必须使用特定于平台的功能。例如，在 C++标准库中没有支持设置线程优先级、配置 CPU 亲和性（CPU 绑定）或设置新线程的堆栈大小。

还应该说一下，随着 C++20 的发布，线程支持库得到了相当大的扩展，未来版本的语言很可能会添加更多功能。由于硬件的发展方式，对良好的并发支持的需求正在增加，而在高度并发程序的效率、可伸缩性和正确性方面还有很多待发现的地方。

## 线程支持库

我们现在将通过 C++线程支持库进行一次介绍，并涵盖其最重要的组件。

### 线程

运行中的程序至少包含一个线程。当调用主函数时，它会在通常被称为**主线程**的线程上执行。每个线程都有一个标识符，在调试并发程序时可能会有用。以下程序打印主线程的线程标识符：

```cpp
int main() { 
  std::cout << "Thread ID: " <<  std::this_thread::get_id() << '\n'; 
} 
```

运行上述程序可能会产生类似以下的输出：

```cpp
 Thread ID: 0x1001553c0 
```

线程可以休眠。在生产代码中很少使用休眠，但在调试过程中可能非常有用。例如，如果您有一个只在罕见情况下发生的数据竞争，向代码中添加休眠可能会使其更频繁地出现。以下是使当前运行的线程休眠一秒钟的方法：

```cpp
std::this_thread::sleep_for(std::chrono::seconds{1}); 
```

在您的程序中插入随机休眠后，程序不应该暴露任何数据竞争。在添加休眠后，您的程序可能无法正常工作；缓冲区可能变满，UI 可能会出现延迟等，但它应该始终以可预测和定义的方式行为。我们无法控制线程的调度，随机休眠模拟了不太可能但可能发生的调度场景。

现在，让我们使用`<thread>`头文件中的`std::thread`类创建一个额外的线程。它表示一个执行线程，并且通常是操作系统线程的包装器。`print()`函数将从我们显式创建的线程中调用：

```cpp
void print() { 
  std::this_thread::sleep_for(std::chrono::seconds{1}); 
  std::cout << "Thread ID: "<<  std::this_thread::get_id() << '\n'; 
} 

int main() { 
  auto t1 = std::thread{print}; 
  t1.join(); 
  std::cout << "Thread ID: "<<  std::this_thread::get_id() << '\n'; 
} 
```

在创建线程时，我们传递一个可调用对象（函数、lambda 或函数对象），线程将在 CPU 上获得调度时间时开始执行。我添加了一个调用 sleep，以明显地说明为什么我们需要在线程上调用`join()`。当`std::thread`对象被销毁时，它必须已经*加入*或*分离*，否则将导致程序调用`std::terminate()`，默认情况下将调用`std::abort()`，如果我们没有安装自定义的`std::terminate_handler`。

在前面的例子中，`join()`函数是阻塞的——它会等待线程运行结束。因此，在前面的例子中，`main()`函数将在线程`t1`运行结束之前不会返回。考虑以下一行：

```cpp
t1.join(); 
```

假设我们通过以下一行替换前面的行来分离线程`t1`：

```cpp
t1.detach(); 
```

在这种情况下，我们的主函数将在线程`t1`唤醒打印消息之前结束，因此程序将（很可能）只输出主线程的线程 ID。请记住，我们无法控制线程的调度，可能但非常不太可能，主线程将在`print()`函数有时间休眠、唤醒并打印其线程 ID 之后输出其消息。

在这个例子中，使用`detach()`而不是`join()`也引入了另一个问题。我们在两个线程中都使用了`std::cout`，而没有任何同步，而且由于`main()`不再等待线程`t1`完成，它们两者理论上都可以并行使用`std::cout`。幸运的是，`std::cout`是线程安全的，可以从多个线程中使用而不会引入数据竞争，因此没有未定义的行为。但是，仍然有可能线程生成的输出是交错的，导致类似以下的结果：

```cpp
Thread ID: Thread ID: 0x1003a93400x700004fd4000 
```

如果我们想避免交错输出，我们需要将字符的输出视为临界区，并同步访问`std::cout`。我们将在稍后更多地讨论临界区和竞争条件，但首先，让我们先了解一些关于`std::thread`的细节。

### 线程状态

在我们继续之前，您应该对`std::thread`对象的真正表示以及它可能处于的状态有一个很好的理解。我们还没有讨论在执行 C++程序的系统中通常有哪些类型的线程。

在下图中，您可以看到一个假设运行中系统的快照。

![](img/B15619_11_09.png)

图 11.9：假设运行中系统的快照

从底部开始，图中显示了 CPU 及其**硬件线程**。这些是 CPU 上的执行单元。在这个例子中，CPU 提供了四个硬件线程。通常这意味着它有四个核心，但也可能是其他配置；例如，一些核心可以执行两个硬件线程。这通常被称为**超线程**。硬件线程的总数可以在运行时打印出来：

```cpp
 std::cout << std::thread::hardware_concurrency() << '\n';
  // Possible output: 4 
```

在运行平台上无法确定硬件线程的数量时，前面的代码也可能输出`0`。

在硬件线程上面的一层包含了**操作系统线程**。这些是实际的软件线程。操作系统调度程序确定操作系统线程由硬件线程执行的时间和持续时间。在*图 11.9*中，目前有六个软件线程中的三个正在执行。

图中最上层包含了`std::thread`对象。`std::thread`对象只是一个普通的 C++对象，可能与底层操作系统线程相关联，也可能不相关联。两个`std::thread`实例不能与同一个底层线程相关联。在图中，您可以看到程序当前有三个`std::thread`实例；两个与线程相关联，一个没有。可以使用`std::thread::joinable`属性来查找`std::thread`对象的状态。如果它已经：

+   默认构造；也就是说，如果它没有任何要执行的内容

+   从中移动（其关联的运行线程已被转移到另一个`std::thread`对象）

+   通过调用`detach()`分离

+   通过调用`join()`已连接

否则，`std::thread`对象处于可连接状态。请记住，当`std::thread`对象被销毁时，它不能再处于可连接状态，否则程序将终止。

### 可连接的线程

C++20 引入了一个名为`std::jthread`的新线程类。它与`std::thread`非常相似，但有一些重要的补充：

+   `std::jthread`支持使用停止令牌停止线程。在 C++20 之前，使用`std::thread`时，我们必须手动实现这一点。

+   在非可连接状态下销毁应用程序时，`std::jthread`的析构函数将发送一个停止请求并在销毁时加入线程。

接下来我将说明后一点。首先，我们将使用如下定义的`print()`函数：

```cpp
void print() {
  std::this_thread::sleep_for(std::chrono::seconds{1});
  std::cout << "Thread ID: "<<  std::this_thread::get_id() << '\n';
} 
```

它休眠一秒，然后打印当前线程标识符：

```cpp
int main() {
  std::cout << "main begin\n"; 
  auto joinable_thread = std::jthread{print};  
  std::cout << "main end\n";
} // OK: jthread will join automatically 
```

在我的机器上运行代码时，产生了以下输出：

```cpp
main begin
main end
Thread ID: 0x1004553c0 
```

现在让我们改变我们的`print()`函数，使其在循环中连续输出消息。然后我们需要一些方法来通知`print()`函数何时停止。`std::jthread`（而不是`std::thread`）通过使用停止令牌内置支持这一点。当`std::jthread`调用`print()`函数时，如果`print()`函数接受这样的参数，它可以传递一个`std::stop_token`的实例。以下是我们如何使用停止令牌来实现这个新的`print()`函数的示例：

```cpp
void print(std::stop_token stoken) {
  while (!stoken.stop_requested()) { 
    std::cout << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds{1});
  }
  std::cout << "Stop requested\n";
} 
```

`while`循环在每次迭代时检查函数是否已被调用`stop_requested()`请求停止。现在，从我们的`main()`函数中，可以通过在我们的`std::jthread`实例上调用`request_stop()`来请求停止：

```cpp
int main() {
  auto joinable_thread = std::jthread(print);
  std::cout << "main: goes to sleep\n";
  std::this_thread::sleep_for(std::chrono::seconds{3});
  std::cout << "main: request jthread to stop\n";
  joinable_thread.request_stop();
} 
```

当我运行这个程序时，它生成了以下输出：

```cpp
main: goes to sleep
Thread ID: 0x70000f7e1000
Thread ID: 0x70000f7e1000
Thread ID: 0x70000f7e1000
main: request jthread to stop
Stop requested 
```

在这个例子中，我们本可以省略对`request_stop()`的显式调用，因为`jthread`在销毁时会自动调用`request_stop()`。

新的`jthread`类是 C++线程库中的一个受欢迎的补充，当在 C++中寻找线程类时，它应该是第一选择。

### 保护关键部分

正如我之前提到的，我们的代码不能包含任何数据竞争。不幸的是，编写带有数据竞争的代码非常容易。在使用线程编写并发程序时，找到关键部分并用锁保护它们是我们不断需要考虑的事情。

C++为我们提供了一个`std::mutex`类，可以用于保护关键部分并避免数据竞争。我将演示如何使用互斥锁来处理一个经典的例子，其中多个线程更新了一个共享的可变计数器变量。

首先，我们定义一个全局可变变量和一个增加计数器的函数：

```cpp
auto counter = 0; // Warning! Global mutable variable
void increment_counter(int n) {
  for (int i = 0; i < n; ++i)
    ++counter;
} 
```

接下来的`main()`函数创建了两个线程，它们都将执行`increment_counter()`函数。在这个例子中还可以看到如何向线程调用的函数传递参数。我们可以向线程构造函数传递任意数量的参数，以匹配要调用的函数签名中的参数。最后，我们断言如果程序没有数据竞争，计数器的值将符合我们的预期：

```cpp
int main() {
  constexpr auto n = int{100'000'000};
  {
    auto t1 = std::jthread{increment_counter, n};
    auto t2 = std::jthread{increment_counter, n};
  }
  std::cout << counter << '\n';
  // If we don't have a data race, this assert should hold:
  assert(counter == (n * 2));
} 
```

这个程序很可能会失败。`assert()`函数不起作用，因为程序当前包含竞争条件。当我反复运行程序时，计数器的值会不同。我最终得到的不是达到值`200000000`，而是最多只有`137182234`。这个例子与本章前面所举的数据竞争例子非常相似。

带有表达式`++counter`的那一行是一个关键部分——它使用了一个共享的可变变量，并由多个线程执行。为了保护这个关键部分，我们现在将使用`<mutex>`头文件中包含的`std::mutex`。稍后，您将看到我们如何通过使用原子操作来避免这个例子中的数据竞争，但现在我们将使用锁。

首先，在`counter`旁边添加全局`std::mutex`对象：

```cpp
auto counter = 0; // Counter will be protected by counter_mutex
auto counter_mutex = std::mutex{}; 
```

但是，`std::mutex`对象本身不是一个可变的共享变量吗？如果被多个线程使用，它不会产生数据竞争吗？是的，它是一个可变的共享变量，但不会产生数据竞争。C++线程库中的同步原语，如`std::mutex`，是为了这个特定目的而设计的。在这方面，它们非常特殊，并使用硬件指令或者平台上必要的任何东西来保证它们自己不会产生数据竞争。

现在我们需要在读取和更新计数器变量的关键部分使用互斥锁。我们可以在`counter_mutex`上使用`lock()`和`unlock()`成员函数，但更倾向于更安全的方法是始终使用 RAII 来处理互斥锁。把互斥锁看作一个资源，当我们使用完毕时总是需要解锁。线程库为我们提供了一些有用的 RAII 类模板来处理锁定。在这里，我们将使用`std::scoped_lock<Mutex>`模板来确保我们安全地释放互斥锁。下面是更新后的`increment_counter()`函数，现在受到互斥锁的保护：

```cpp
void increment_counter(int n) {
  for (int i = 0; i < n; ++i) {
    auto lock = std::scoped_lock{counter_mutex};
    ++counter;
  }
} 
```

程序现在摆脱了数据竞争，并且按预期工作。如果我们再次运行它，`assert()`函数中的条件现在将成立。

### 避免死锁

只要一个线程一次只获取一个锁，就不会有死锁的风险。然而，有时需要在已经持有先前获取的锁的情况下获取另一个锁。在这种情况下，通过同时抓住两个锁来避免死锁的风险。C++有一种方法可以通过使用`std::lock()`函数来做到这一点，该函数获取任意数量的锁，并在所有锁都被获取之前阻塞。

以下是一个在账户之间转账的示例。在交易期间需要保护两个账户，因此我们需要同时获取两个锁。操作如下：

```cpp
struct Account { 
  Account() {} 
  int balance_{0}; 
  std::mutex m_{}; 
}; 

void transfer_money(Account& from, Account& to, int amount) { 
   auto lock1 = std::unique_lock<std::mutex>{from.m_, std::defer_lock}; 
   auto lock2 = std::unique_lock<std::mutex>{to.m_, std::defer_lock}; 

   // Lock both unique_locks at the same time 
   std::lock(lock1, lock2); 

   from.balance_ -= amount; 
   to.balance_ += amount; 
} 
```

我们再次使用 RAII 类模板来确保每当这个函数返回时我们都释放锁。在这种情况下，我们使用`std::unique_lock`，它为我们提供了推迟锁定互斥锁的可能性。然后，我们通过使用`std::lock()`函数同时显式锁定两个互斥锁。

### 条件变量

**条件变量**使线程能够等待直到某个特定条件得到满足。线程还可以使用条件变量向其他线程发出条件已经改变的信号。

并发程序中的一个常见模式是有一个或多个线程在等待数据以某种方式被消耗。这些线程通常被称为**消费者**。另一组线程负责生成准备好被消耗的数据。这些生成数据的线程被称为**生产者**，如果只有一个线程，则称为**生产者**。

生产者和消费者模式可以使用条件变量来实现。我们可以结合使用`std::condition_variable`和`std::unique_lock`来实现这个目的。让我们看一个生产者和消费者的示例，使它们不那么抽象：

```cpp
auto cv = std::condition_variable{}; 
auto q = std::queue<int>{}; 
auto mtx = std::mutex{};     // Protects the shared queue 
constexpr int sentinel = -1; // Value to signal that we are done 

void print_ints() { 
  auto i = 0; 
  while (i != sentinel) { 
    { 
      auto lock = std::unique_lock<std::mutex>{mtx}; 
      while (q.empty()) {
        cv.wait(lock); // The lock is released while waiting 
      }
      i = q.front(); 
      q.pop(); 
    } 
    if (i != sentinel) { 
      std::cout << "Got: " << i << '\n'; 
    } 
  } 
} 

auto generate_ints() { 
  for (auto i : {1, 2, 3, sentinel}) { 
    std::this_thread::sleep_for(std::chrono::seconds(1)); 
    { 
      auto lock = std::scoped_lock{mtx}; 
      q.push(i); 
    } 
    cv.notify_one(); 
  } 
} 

int main() { 
   auto producer = std::jthread{generate_ints}; 
   auto consumer = std::jthread{print_ints}; 
} 
```

我们创建了两个线程：一个`consumer`线程和一个`producer`线程。`producer`线程生成一系列整数，并在每秒钟将它们推送到全局`std::queue<int>`中。每当向队列添加元素时，生产者都会使用`notify_one()`来发出条件已经改变的信号。

程序检查队列中是否有数据可供消费者线程使用。还要注意的是，在通知条件变量时不需要持有锁。

消费者线程负责将数据（即整数）打印到控制台。它使用条件变量等待空队列发生变化。当消费者调用`cv.wait(lock)`时，线程会进入睡眠状态，让出 CPU 给其他线程执行。重要的是要理解为什么在调用`wait()`时需要传递变量`lock`。除了让线程进入睡眠状态，`wait()`在睡眠时也会释放互斥锁，然后在返回之前重新获取互斥锁。如果`wait()`没有释放互斥锁，生产者将无法向队列中添加元素。

为什么消费者在条件变量上等待时使用`while`循环而不是`if`语句？这是一个常见的模式，有时我们需要这样做，因为可能有其他消费者在我们之前被唤醒并清空了队列。在我们的程序中，我们只有一个消费者线程，所以这种情况不会发生。但是，消费者可能会在等待时被唤醒，即使生产者线程没有发出信号。这种现象称为**虚假唤醒**，导致这种情况发生的原因超出了本书的范围。

作为使用`while`循环的替代方案，我们可以使用`wait()`的重载版本，该版本接受一个谓词。这个`wait()`版本检查谓词是否满足，并为我们执行循环。在我们的示例中，它看起来像这样：

```cpp
// ...
auto lock = std::unique_lock<std::mutex>{mtx}; 
cv.wait(lock, [] { return !q.empty(); });
// ... 
```

您可以在 Anthony Williams 的*C++ Concurrency in Action*，*Second Edition*中找到有关虚假唤醒的更多信息。您现在至少知道如何处理可能发生虚假唤醒的情况：始终在 while 循环中检查条件，或者使用接受谓词的`wait()`的重载版本。

条件变量和互斥锁是自从 C++引入线程以来就可用的同步原语。C++20 还提供了额外的有用的类模板，用于同步线程，即`std::counting_semaphore`、`std::barrier`和`std::latch`。我们将在后面介绍这些新的原语。首先，我们将花一些时间讨论返回值和错误处理。

### 返回数据和处理错误

到目前为止，在本章中所呈现的示例都使用了共享变量来在线程之间通信状态。我们使用互斥锁来确保避免数据竞争。在程序规模增大时，使用互斥锁的共享数据可能会非常难以正确实现。在代码库中分散使用显式锁定也需要大量工作。跟踪共享内存和显式锁定使我们远离我们编写程序时真正想要实现和花时间的目标。

此外，我们还没有处理错误处理。如果一个线程需要向另一个线程报告错误怎么办？当函数需要报告运行时错误时，我们通常使用异常，那么我们如何使用异常来做到这一点呢？

在标准库的`<future>`头文件中，我们可以找到一些类模板，可以帮助我们编写并发代码，而无需全局变量和锁，并且可以在线程之间传递异常以处理错误。我现在将介绍**future**和**promise**，它们代表值的两个方面。future 是值的接收方，promise 是值的返回方。

以下是使用`std::promise`将结果返回给调用者的示例：

```cpp
auto divide(int a, int b, std::promise<int>& p) { 
  if (b == 0) { 
    auto e = std::runtime_error{"Divide by zero exception"}; 
    p.set_exception(std::make_exception_ptr(e)); 
  } 
  else { 
    const auto result = a / b; 
    p.set_value(result); 
  } 
} 

int main() { 
   auto p = std::promise<int>{}; 
   std::thread(divide, 45, 5, std::ref(p)).detach(); 

   auto f = p.get_future(); 
   try { 
     const auto& result = f.get(); // Blocks until ready 
     std::cout << "Result: " << result << '\n'; 
   } 
   catch (const std::exception& e) { 
     std::cout << "Caught exception: " << e.what() << '\n'; 
   } 
} 
```

调用者（`main()`函数）创建`std::promise`对象并将其传递给`divide()`函数。我们需要使用`<functional>`中的`std::ref`，以便引用可以通过`std::thread`正确地转发到`compute()`。

当`divide()`函数计算出结果时，通过调用`set_value()`函数通过 promise 传递返回值。如果`divide()`函数发生错误，则在 promise 上调用`set_exception()`函数。

future 代表可能已经计算或尚未计算的计算值。由于 future 是一个普通对象，我们可以将其传递给需要计算值的其他对象。最后，当某个客户端需要该值时，它调用`get()`来获取实际值。如果在那时没有计算，调用`get()`将阻塞，直到完成。

还要注意的是，我们成功地进行了适当的错误处理来回传递数据，而没有使用任何共享全局数据，并且没有显式锁定。promise 为我们处理了这一切，我们可以专注于实现程序的基本逻辑。

### 任务

通过 future 和 promise，我们成功摆脱了显式锁定和共享全局数据。在可能的情况下，我们的代码将受益于使用更高级的抽象。在这里，我们将进一步探索自动为我们设置未来和承诺的类。您还将看到我们如何摆脱手动管理线程，并将其留给库。

在许多情况下，我们并不需要管理线程；相反，我们真正需要的是能够异步执行任务，并使该任务与程序的其余部分同时执行，然后最终将结果或错误传达给需要它的程序部分。任务应该在隔离环境中执行，以最小化争用和数据竞争的风险。

我们将从重写我们之前的例子开始，该例子将两个数字相除。这一次，我们将使用`<future>`中的`std::packaged_task`，它为我们设置 promise 的所有工作都是正确的：

```cpp
int divide(int a, int b) { // No need to pass a promise ref here! 
  if (b == 0) { 
    throw std::runtime_error{"Divide by zero exception"}; 
  } 
  return a / b; 
} 

int main() { 
  auto task = std::packaged_task<decltype(divide)>{divide}; 
  auto f = task.get_future(); 
  std::thread{std::move(task), 45, 5}.detach(); 

  // The code below is unchanged from the previous example 
  try { 
    const auto& result = f.get(); // Blocks until ready 
    std::cout << "Result: " << result << '\n'; 
  } 
  catch (const std::exception& e) { 
    std::cout << "Caught exception: " << e.what() << '\n'; 
  } 
  return 0; 
} 
```

`std::packaged_task`本身是一个可调用对象，可以移动到我们正在创建的`std::thread`对象中。正如你所看到的，`std::packaged_task`现在为我们做了大部分工作：我们不必自己创建 promise。但更重要的是，我们可以像编写普通函数一样编写我们的`divide()`函数，而不需要通过 promise 显式返回值或异常；`std::packaged_task`会为我们做这些。

在本节的最后一步，我们还希望摆脱手动线程管理。创建线程并不是免费的，您将在后面看到，程序中的线程数量会影响性能。似乎是否为我们的`divide()`函数创建一个新线程并不一定由`divide()`的调用者决定。库再次通过提供另一个有用的函数模板`std::async()`来帮助我们。在我们的`divide()`示例中，我们唯一需要做的是用一个简单的调用`std::async()`替换创建`std::packaged_task`和`std::thread`对象的代码：

```cpp
 auto f = std::async(divide, 45, 5); 
```

我们现在已经从基于线程的编程模型切换到了基于任务的模型。完整的基于任务的示例现在看起来是这样的：

```cpp
int divide(int a, int b) { 
  if (b == 0) { 
    throw std::runtime_error{"Divide by zero exception"}; 
  } 
  return a / b; 
} 

int main() { 
  auto future = std::async(divide, 45, 5); 
  try { 
    const auto& result = future.get(); 
    std::cout << "Result: " << result << '\n'; 
  } 
  catch (const std::exception& e) { 
    std::cout << "Caught exception: " << e.what() << '\n'; 
  } 
} 
```

这里真的只剩下很少的代码来处理并发。异步调用函数的推荐方式是使用`std::async()`。关于为什么以及何时首选`std::async()`的更深入讨论，我强烈推荐 Scott Meyers 的*Effective Modern C++*中的*并发*章节。

## C++20 中的额外同步原语

C++20 带来了一些额外的同步原语，即`std::latch`、`std::barrier`和`std::counting_semaphore`（以及模板特化`std::binary_semaphore`）。本节将概述这些新类型以及它们可以有用的一些典型场景。我们将从`std::latch`开始。

### 使用门闩

门闩是一种同步原语，可用于同步多个线程。它创建一个同步点，所有线程都必须到达。您可以将门闩视为递减计数器。通常，所有线程都会递减计数器一次，然后等待门闩达到零，然后再继续。

门闩是通过传递内部计数器的初始值来构造的：

```cpp
auto lat = std::latch{8}; // Construct a latch initialized with 8 
```

然后线程可以使用`count_down()`递减计数器：

```cpp
lat.count_down(); // Decrement but don't wait 
```

线程可以等待在门闩上达到零：

```cpp
lat.wait(); // Block until zero 
```

还可以检查（不阻塞）计数器是否已经达到零：

```cpp
if (lat.try_wait()) { 
  // All threads have arrived ...
} 
```

通常在递减计数器后立即等待门闩达到零，如下所示：

```cpp
lat.count_down();
lat.wait(); 
```

事实上，这种用法很常见，值得一个定制的成员函数；`arrive_and_wait()`递减门闩，然后等待门闩达到零：

```cpp
lat.arrive_and_wait(); // Decrement and block while not zero 
```

在并发工作时，加入一组分叉任务是一种常见情况。如果任务只需要在最后加入，我们可以使用一个未来对象数组（等待）或者只等待所有线程完成。但在其他情况下，我们希望一组异步任务到达一个共同的同步点，然后让任务继续运行。这些情况通常发生在多个工作线程开始实际工作之前需要某种初始化的情况下。

#### 示例：使用 std::latch 初始化线程

以下示例演示了当多个工作线程需要在开始工作之前运行一些初始化代码时，如何使用`std::latch`。

当创建一个线程时，会为堆栈分配一块连续的内存。通常，当首次在虚拟地址空间中分配内存时，这块内存尚未驻留在物理内存中。相反，当堆栈被使用时，将生成*页错误*，以便将虚拟内存映射到物理内存。操作系统会为我们处理映射，这是一种在需要时懒惰地映射内存的有效方式。通常，这正是我们想要的：我们尽可能晚地支付映射内存的成本，只有在需要时才会支付。然而，在低延迟很重要的情况下，例如在实时代码中，可能需要完全避免页错误。堆栈内存不太可能被操作系统分页出去，因此通常只需运行一些代码，生成页错误，从而将虚拟堆栈内存映射到物理内存。这个过程称为**预缓存**。

没有一种可移植的方法来设置或获取 C++线程的堆栈大小，所以这里我们只是假设堆栈至少为 500 KB。以下代码尝试预先分配堆栈的前 500 KB：

```cpp
void prefault_stack() {
  // We don't know the size of the stack
  constexpr auto stack_size = 500u * 1024u; 
  // Make volatile to avoid optimization
  volatile unsigned char mem[stack_size]; 
  std::fill(std::begin(mem), std::end(mem), 0);
} 
```

这里的想法是在堆栈上分配一个数组，它将占用大量的堆栈内存。然后，为了生成页面错误，我们使用`std::fill()`写入数组中的每个元素。之前没有提到`volatile`关键字，它是 C++中一个有些令人困惑的关键字。它与并发无关；它只是在这里添加以防止编译器优化掉这段代码。通过声明`mem`数组为`volatile`，编译器不允许忽略对数组的写入。

现在，让我们专注于实际的`std::latch`。假设我们想要创建一些工作线程，只有在所有线程堆栈都被预分配后才能开始它们的工作。我们可以使用`std::latch`来实现这种同步，如下所示：

```cpp
auto do_work() { /* ... */ }
int main() {
  constexpr auto n_threads = 2;
  auto initialized = std::latch{n_threads};
  auto threads = std::vector<std::thread>{};
  for (auto i = 0; i < n_threads; ++i) {
    threads.emplace_back([&] {
      prefault_stack();
      initialized.arrive_and_wait(); 
      do_work();
    });
  }
  initialized.wait();
  std::cout << "Initialized, starting to work\n";
  for (auto&& t : threads) {
    t.join();
  }
} 
```

所有线程到达后，主线程可以开始向工作线程提交工作。在这个例子中，所有线程都在等待其他线程到达，通过在屏障上调用`arrive_and_wait()`来实现。一旦屏障达到零，就不能再重用它了。没有重置屏障的函数。如果我们有一个需要多个同步点的场景，我们可以使用`std::barrier`来代替。

### 使用屏障

屏障类似于 latch，但有两个主要的附加功能：屏障可以被*重用*，并且当所有线程到达屏障时可以运行*完成函数*。

通过传递内部计数器的初始值和完成函数来构造屏障：

```cpp
auto bar = std::barrier{8, [] {
  // Completion function
  std::cout "All threads arrived at barrier\n";
}}; 
```

线程可以以与使用 latch 相同的方式到达并等待：

```cpp
bar.arrive_and_wait(); // Decrement but don't wait 
```

每当所有线程都到达（也就是说，当屏障的内部计数器达到零时）时，会发生两件事：

+   提供给构造函数的完成函数由屏障调用。

+   完成函数返回后，内部计数器将被重置为其初始值。

屏障在基于**fork-join 模型**的并行编程算法中非常有用。通常，迭代算法包含一个可以并行运行的部分和一个需要顺序运行的部分。多个任务被分叉并并行运行。然后，当所有任务都完成并加入时，会执行一些单线程代码来确定算法是否应该继续还是结束。

![](img/B15619_11_10.png)

图 11.10：fork-join 模型的示例

遵循 fork-join 模型的并发算法将受益于使用屏障，并可以以一种优雅和高效的方式避免其他显式的锁定机制。让我们看看如何在一个简单的问题中使用屏障但有两个主要的问题。

#### 示例：使用 std::barrier 进行 fork-join

我们的下一个示例是一个玩具问题，将演示 fork-join 模型。我们将创建一个小程序，模拟一组骰子被掷出，并计算在获得所有 6 之前需要掷出的次数。掷一组骰子是我们可以并发执行的（分叉）操作。在单个线程中执行的加入步骤检查结果，并确定是重新掷骰子还是结束。

首先，我们需要实现掷骰子的代码，有六个面。为了生成 1 到 6 之间的数字，我们可以使用`<random>`头文件中的类的组合，如下所示：

```cpp
auto engine = 
  std::default_random_engine{std::random_device{}()};
auto dist = std::uniform_int_distribution<>{1, 6};
auto result = dist(engine); 
```

这里的`std::random_device`负责生成一个种子，用于产生伪随机数的引擎。为了以相等的概率选择 1 到 6 之间的整数，我们使用`std::uniform_int_distribution`。变量`result`是掷骰子的结果。

现在我们想将此代码封装到一个函数中，该函数将生成一个随机整数。生成种子并创建引擎通常很慢，我们希望避免在每次调用时都这样做。通常的做法是使用`static`持续时间声明随机引擎，以便它在整个程序的生命周期内存在。但是，`<random>`中的类不是线程安全的，因此我们需要以某种方式保护`static`引擎。我将利用这个机会演示如何使用线程本地存储，而不是使用互斥锁同步访问，这将使随机数生成器按顺序运行。

以下是如何将引擎声明为`static thread_local`对象的方法：

```cpp
auto random_int(int min, int max) {
  // One engine instance per thread
  static thread_local auto engine = 
    std::default_random_engine{std::random_device{}()};
  auto dist = std::uniform_int_distribution<>{min, max};
  return dist(engine);
} 
```

具有`thread_local`存储期的静态变量将在每个线程中创建一次；因此，可以在不使用任何同步原语的情况下同时从多个线程调用`random_int()`。有了这个小的辅助函数，我们可以继续使用`std::barrier`实现程序的其余部分：

```cpp
int main() {
  constexpr auto n = 5; // Number of dice
  auto done = false;
  auto dice = std::array<int, n>{};
  auto threads = std::vector<std::thread>{};
  auto n_turns = 0;
  auto check_result = [&] { // Completion function
    ++n_turns;
    auto is_six = [](auto i) { return i == 6; };
    done = std::all_of(dice.begin(), dice.end(), is_six); 
  };
  auto bar = std::barrier{n, check_result}; 
  for (int i = 0; i < n; ++i) {
    threads.emplace_back([&, i] {
      while (!done) {
        dice [i] = random_int(1, 6); // Roll dice        
        bar.arrive_and_wait();       // Join
      }});
  }
  for (auto&& t : threads) { 
    t.join();
  }
  std::cout << n_turns << '\n';
} 
```

lambda`check_result()`是完成函数，每次所有线程都到达屏障时都会调用它。完成函数检查每个骰子的值，并确定是否应该玩新一轮，或者我们已经完成。

传递给`std::thread`对象的 lambda 通过值捕获索引`i`，以便所有线程都具有唯一的索引。其他变量`done`、`dice`和`bar`通过引用捕获。

还要注意，我们可以在不引入任何数据竞争的情况下从不同线程中对引用捕获的变量进行突变和读取，这要归功于屏障执行的协调。

### 使用信号量进行信号传递和资源计数

**信号量**一词表示可以用于信号传递的东西，例如旗帜或灯。在接下来的示例中，您将看到我们如何使用信号量来传递其他线程可能正在等待的不同状态。

信号量还可以用于控制对资源的访问，类似于`std::mutex`限制对临界区的访问：

```cpp
class Server {
public:
  void handle(const Request& req) {
    sem_.acquire();
    // Restricted section begins here.
    // Handle at most 4 requests concurrently.
    do_handle(req);
    sem_.release();
  }
private:
  void do_handle(const Request& req) { /* ... */ }
  std::counting_semaphore<4> sem_{4};
}; 
```

在这种情况下，信号量的初始值为`4`，这意味着最多可以同时处理四个并发请求。与代码中的某个部分相互排斥的访问不同，多个线程可以访问相同的部分，但受限于当前在该部分的线程数量。

成员函数`acquire()`在信号量大于零时减少信号量。否则，`acquire()`将阻塞，直到信号量允许其减少并进入受限制的部分。`release()`在不阻塞的情况下增加计数器。如果在`release()`增加计数器之前信号量为零，则会发出信号通知等待的线程。

除了`acquire()`函数之外，还可以使用`try_acquire()`函数*无阻塞*地尝试减少计数器。如果成功减少计数器，则返回`true`，否则返回`false`。函数`try_acquire_for()`和`try_acquire_until()`可以类似地使用。但是，它们在计数器已经为零时不会立即返回`false`，而是在指定时间内自动尝试减少计数器，然后再返回给调用者。

这三个函数的模式与标准库中的其他类型相同，例如`std::timed_mutex`及其`try_lock()`、`try_lock_for()`和`try_lock_until()`成员函数。

`std::counting_semaphore`是一个模板，具有一个模板参数，接受信号量的最大值。在增加（释放）信号量超过其最大值时被认为是编程错误。

具有最大大小为 1 的`std::counting_semaphore`称为**二进制信号量**。`<semaphore>`头文件包括二进制信号量的别名声明：

```cpp
std::binary_semaphore = std::counting_semaphore<1>; 
```

二进制信号量的实现效率比具有更高最大值的计数信号量更高。

信号量的另一个重要属性是释放信号量的线程可能不是获取它的线程。这与`std::mutex`相反，后者要求获取互斥锁的线程也必须释放它。然而，使用信号量时，通常有一种类型的任务负责等待（获取），另一种类型的任务负责信号（释放）。这将在我们的下一个示例中演示。

#### 示例：使用信号量的有界缓冲区

以下示例演示了一个有界缓冲区。这是一个固定大小的缓冲区，可以有多个线程从中读取和写入。同样，这个示例演示了你已经使用条件变量看到的生产者-消费者模式。生产者线程是写入缓冲区的线程，而读取线程是从缓冲区中读取（和弹出元素）的线程。

以下图显示了缓冲区（一个固定大小的数组）和跟踪读取和写入位置的两个变量：

![](img/B15619_11_11.png)

图 11.11：有界缓冲区具有固定大小

我们将一步一步地开始，从一个专注于有界缓冲区内部逻辑的版本开始。使用信号量进行信号传递将在下一个版本中添加。在这里，初始尝试演示了读取和写入位置的使用方式：

```cpp
template <class T, int N> 
class BoundedBuffer {
  std::array<T, N> buf_;
  std::size_t read_pos_{};
  std::size_t write_pos_{};
  std::mutex m_;
  void do_push(auto&& item) {
    /* Missing: Should block if buffer is full */
    auto lock = std::unique_lock{m_};
    buf_[write_pos_] = std::forward<decltype(item)>(item);
    write_pos_ = (write_pos_ + 1) % N;
  }
public:
  void push(const T& item) { do_push(item); }
  void push(T&& item) { do_push(std::move(item)); }
  auto pop() {
    /* Missing: Should block if buffer is empty */
    auto item = std::optional<T>{};
    {
      auto lock = std::unique_lock{m_};
      item = std::move(buf_[read_pos_]);
      read_pos_ = (read_pos_ + 1) % N;
    }
    return std::move(*item);
  }
}; 
```

这个第一次尝试包含了固定大小的缓冲区，读取和写入位置，以及一个互斥锁，用于保护数据成员免受数据竞争的影响。这个实现应该能够让任意数量的线程同时调用`push()`和`pop()`。

`push()`函数重载了`const T&`和`T&&`。这是标准库容器使用的一种优化技术。`T&&`版本在调用者传递一个右值时避免了参数的复制。

为了避免重复推送操作的逻辑，一个辅助函数`do_push()`包含了实际的逻辑。通过使用转发引用（`auto&& item`）以及`std::forward`，`item`参数将根据客户端使用右值还是左值调用`push()`而进行移动分配或复制分配。

这个有界缓冲区的版本并不完整，因为它没有保护我们免受`write_pos`指向（或超出）`read_pos`的影响。同样，`read_pos`绝不能指向`write_pos`（或超出）。我们想要的是一个缓冲区，在缓冲区满时生产者线程被阻塞，而在缓冲区为空时消费者线程被阻塞。

这是使用计数信号量的完美应用。信号量*阻塞*试图将信号量减少到已经为零的线程。信号量*信号*被阻塞的线程，每当一个值为零的信号量增加时。

对于有界缓冲区，我们需要两个信号量：

+   第一个信号量`n_empty_slots`跟踪缓冲区中空槽的数量。它将以缓冲区大小的值开始。

+   第二个信号量`n_full_slots`跟踪缓冲区中满槽的数量。

确保你理解为什么需要两个计数信号量（而不是一个）。原因是有两个不同的*状态*需要被信号：当缓冲区*满*时和当缓冲区*空*时。

在添加了使用两个计数信号量进行信号处理后，有界缓冲区现在看起来像这样（在此版本中添加的行用“new”标记）：

```cpp
template <class T, int N> 
class BoundedBuffer {
  std::array<T, N> buf_;
  std::size_t read_pos_{};
  std::size_t write_pos_{};
  std::mutex m_;
  std::counting_semaphore<N> n_empty_slots_{N}; // New
  std::counting_semaphore<N> n_full_slots_{0};  // New
  void do_push(auto&& item) {
    // Take one of the empty slots (might block)
    n_empty_slots_.acquire();                   // New
    try {
      auto lock = std::unique_lock{m_};
      buf_[write_pos_] = std::forward<decltype(item)>(item);
      write_pos_ = (write_pos_ + 1) % N;
    } catch (...) {
      n_empty_slots_.release();                 // New
      throw;
    }
    // Increment and signal that there is one more full slot
    n_full_slots_.release();                    // New
  }
public:
  void push(const T& item) { do_push(item); }
  void push(T&& item) { do_push(std::move(item)); }
  auto pop() {
    // Take one of the full slots (might block)
    n_full_slots_.acquire();                // New
    auto item = std::optional<T>{};
    try {
      auto lock = std::unique_lock{m_};
      item = std::move(buf_[read_pos_]);
      read_pos_ = (read_pos_ + 1) % N;
    } catch (...) {
      n_full_slots_.release();             // New
      throw;
    }
    // Increment and signal that there is one more empty slot
    n_empty_slots_.release();              // New
    return std::move(*item);
  }
}; 
```

这个版本支持多个生产者和消费者。两个信号量的使用保证了两者都不会达到缓冲区中元素的最大数量。例如，生产者线程无法在首先检查是否有至少一个空槽之前添加值并增加`n_full_slots`信号量。

还要注意，`acquire()`和`release()`是从不同的线程调用的。例如，消费者线程正在等待（`acquire()`）`n_full_slots`信号量，而生产者线程正在对同一个信号量进行信号（`release()`）。

C++20 中添加的新同步原语是常见的线程库中常见的构造。与`std::mutex`和`std::condition_variable`相比，它们提供了方便且通常更有效的替代方案来同步对共享资源的访问。

## C++中的原子支持

标准库包含对**原子变量**的支持，有时被称为**原子**。原子变量是一种可以安全地从多个线程使用和变异而不引入数据竞争的变量。

您还记得我们之前看过的两个线程更新全局计数器的数据竞争示例吗？我们通过添加互斥锁和计数器来解决了这个问题。我们可以使用 `std::atomic<int>` 来代替显式锁：

```cpp
std::atomic<int> counter; 

auto increment_counter(int n) { 
  for (int i = 0; i < n; ++i) 
    ++counter; // Safe, counter is now an atomic<int> 
} 
```

`++counter` 是一种方便的方式，相当于 `counter.fetch_add(1)`。可以从多个线程同时调用的所有成员函数都是安全的。

原子类型来自`<atomic>`头文件。对于所有标量数据类型，都有命名为`std::atomic_int`的 typedef。这与`std::atomic<int>`相同。只要自定义类型是平凡可复制的，就可以将自定义类型包装在`std::atomic`模板中。基本上，这意味着类的对象完全由其数据成员的位描述。这样，对象可以通过例如`std::memcpy()`仅复制原始字节来复制。因此，如果一个类包含虚函数、指向动态内存的指针等，就不再可能仅仅复制对象的原始位并期望它能够工作，因此它不是平凡可复制的。这可以在编译时检查，因此如果尝试创建一个不是平凡可复制的类型的原子，将会得到编译错误：

```cpp
struct Point { 
  int x_{}; 
  int y_{}; 
}; 

auto p = std::atomic<Point>{};       // OK: Point is trivially copyable 
auto s = std::atomic<std::string>{}; // Error: Not trivially copyable 
```

还可以创建原子指针。这使得指针本身是原子的，但指向的对象不是。我们将在稍后更多地讨论原子指针和引用。

### 无锁属性

使用原子而不是用互斥锁保护变量的原因是避免使用`std::mutex`引入的性能开销。此外，互斥锁可能会阻塞线程一段非确定性的时间，并引入优先级反转（参见*线程优先级*部分），这排除了在低延迟环境中使用互斥锁。换句话说，您的代码中可能有延迟要求的部分完全禁止使用互斥锁。在这些情况下，了解原子变量是否使用互斥锁是很重要的。

原子变量可能会或可能不会使用锁来保护数据；这取决于变量的类型和平台。如果原子变量不使用锁，则称为**无锁**。您可以在运行时查询变量是否无锁：

```cpp
auto variable = std::atomic<int>{1};
assert(variable.is_lock_free());          // Runtime assert 
```

这很好，因为现在至少在运行程序时我们可以断言使用 `variable` 对象是无锁的。通常，同一类型的所有原子对象都将是无锁或有锁的，但在一些奇异的平台上，有可能两个原子对象会生成不同的答案。

通常更有趣的是知道在特定平台上是否保证了原子类型（`std::atomic<T>`）是无锁的，最好是在编译时而不是运行时知道。自 C++17 以来，还可以使用`is_always_lock_free()`在编译时验证原子特化是否是无锁的，就像这样：

```cpp
static_assert(std::atomic<int>::is_always_lock_free); 
```

如果我们的目标平台上 `atomic<int>` 不是无锁的，这段代码将生成编译错误。现在，如果我们编译一个假设 `std::atomic<int>` 不使用锁的程序，它将无法编译，这正是我们想要的。

在现代平台上，任何`std::atomic<T>`，其中`T`适合本机字大小，通常都是*始终无锁*的。在现代 x64 芯片上，甚至可以获得双倍的数量。例如，在现代英特尔 CPU 上编译的 libc++上，`std::atomic<std::complex<double>>`始终是无锁的。

### 原子标志

保证始终是无锁的原子类型是`std::atomic_flag`（无论目标平台如何）。因此，`std::atomic_flag`不提供`is_always_lock_free()`/`is_lock_free()`函数，因为它们总是返回`true`。

原子标志可以用来保护临界区，作为使用`std::mutex`的替代方案。由于锁的概念容易理解，我将在这里以此为例。但需要注意的是，我在本书中演示的锁的实现并不是生产就绪的代码，而是概念上的实现。以下示例演示了如何概念上实现一个简单的自旋锁：

```cpp
class SimpleMutex {       
  std::atomic_flag is_locked_{};           // Cleared by default
public:
  auto lock() noexcept {
    while (is_locked_.test_and_set()) {
      while (is_locked_.test());           // Spin here
    }
  } 
  auto unlock() noexcept {
    is_locked_.clear();
  }
}; 
```

`lock()`函数调用`test_and_set()`来设置标志并同时获取标志的先前值。如果`test_and_set()`返回`false`，意味着调用者成功获取了锁（在先前清除标志时设置标志）。否则，内部的`while`循环将不断使用`test()`在一个自旋循环中轮询标志的状态。我们在额外的内部循环中使用`test()`的原因是性能：`test()`不会使缓存行失效，而`test_and_set()`会。这种锁定协议称为**测试和测试并设置**。

这个自旋锁可以工作，但不太节约资源；当线程执行时，它不断使用 CPU 来一遍又一遍地检查相同的条件。我们可以在每次迭代中添加一个短暂的休眠和指数退避，但是为各种平台和场景微调这一点是很困难的。

幸运的是，C++20 为`std::atomic`添加了等待和通知 API，使线程可以等待（以一种节约资源的方式）原子变量改变其值。

### 原子等待和通知

自 C++20 以来，`std::atomic`和`std::atomic_flag`提供了等待和通知的功能。`wait()`函数阻塞当前线程，直到原子变量的值发生变化，并且其他线程通知等待线程。线程可以通过调用`notify_one()`或`notify_all()`来通知发生了变化。

有了这个新功能，我们可以避免不断轮询原子的状态，而是以更节约资源的方式等待值的改变；这类似于`std::condition_variable`允许我们等待和通知状态改变的方式。

通过使用等待和通知，前一节中实现的`SimpleMutex`可以重写如下：

```cpp
class SimpleMutex {       
  std::atomic_flag is_locked_{}; 
public:
  auto lock() noexcept {
    while (is_locked_.test_and_set())
      is_locked_.wait(true);    // Don't spin, wait
  } 
  auto unlock() noexcept {
    is_locked_.clear();
    is_locked_.notify_one();   // Notify blocked thread
  }
}; 
```

我们将旧值（`true`）传递给`wait()`。在`wait()`返回时，可以保证原子变量已经改变，不再是`true`。但不能保证我们会捕捉到*所有*变量的改变。变量可能已经从状态 A 改变到状态 B，然后再回到状态 A，而没有通知等待的线程。这是无锁编程中的一种现象，称为**ABA 问题**。

这个示例演示了使用`std::atomic_flag`的等待和通知功能。相同的等待和通知 API 也适用于`std::atomic`类模板。

请注意，本章中介绍的自旋锁不是生产就绪的代码。实现高效的锁通常涉及正确使用内存顺序（稍后讨论）和用于让出的非可移植代码，这超出了本书的范围。详细讨论可在[`timur.audio/using-locks-in-real-time-audio-processing-safely`](https://timur.audio/using-locks-in-real-time-audio-processing-safely)找到。

现在，我们将继续讨论原子指针和原子引用。

### 在多线程环境中使用 shared_ptr

`std::shared_ptr`怎么样？它能在多线程环境中使用吗？当多个线程访问由多个共享指针引用的对象时，引用计数是如何处理的？

要理解共享指针和线程安全，我们需要回顾`std::shared_ptr`通常是如何实现的（也可以参见*第七章*，*内存管理*）。考虑以下代码：

```cpp
// Thread 1 
auto p1 = std::make_shared<int>(42); 
```

代码在堆上创建了一个`int`和一个指向`int`对象的引用计数智能指针。使用`std::make_shared()`创建共享指针时，会在`int`旁边创建一个`控制块`。控制块包含引用计数等内容，每当创建指向`int`的新指针时，引用计数就会增加，每当销毁指向`int`的指针时，引用计数就会减少。总之，当执行上述代码行时，会创建三个单独的实体：

+   实际的`std::shared_ptr`对象`p1`（堆栈上的局部变量）

+   一个控制块（堆对象）

+   一个`int`（堆对象）

下图显示了三个对象：

![](img/B15619_11_12.png)

图 11.12：一个指向整数对象的 shared_ptr 实例 p1 和包含引用计数的控制块。在这种情况下，只有一个共享指针使用 int，因此引用计数为 1。

现在，考虑如果以下代码被第二个线程执行会发生什么？

```cpp
// Thread 2 
auto p2 = p1; 
```

我们正在创建一个新的指针指向`int`（和控制块）。创建`p2`指针时，我们读取`p1`，但在更新引用计数时也需要改变控制块。控制块位于堆上，并且在两个线程之间共享，因此需要同步以避免数据竞争。由于控制块是隐藏在`std::shared_ptr`接口后面的实现细节，我们无法知道如何保护它，结果发现它已经被实现照顾了。

通常，它会使用可变的原子计数器。换句话说，引用计数更新是线程安全的，因此我们可以在不担心同步引用计数的情况下，从不同线程使用多个共享指针。这是一个良好的实践，也是在设计类时需要考虑的事情。如果在客户端视角下，对变量进行了语义上只读（`const`）的方法中进行了变异，那么应该使变异变量线程安全。另一方面，客户端可以检测到的一切作为变异函数的东西应该留给类的客户端来同步。

下图显示了两个`std::shared_ptr`，`p1`和`p2`，它们都可以访问相同的对象。`int`是共享对象，控制块是`std::shared_ptr`实例之间内部共享的对象。控制块默认是线程安全的：

![](img/B15619_11_13.png)

图 11.13：两个共享指针访问相同的对象

总结：

+   在这个例子中，共享对象，即`int`，不是线程安全的，如果从多个线程访问，需要显式加锁。

+   控制块已经是线程安全的，因此引用计数机制在多线程环境中可以工作。

让我们继续保护`shared_ptr`实例。

#### 保护 shared_ptr 实例

现在只剩下一个部分：在前面的例子中，实际的`std::shared_ptr`对象`p1`和`p2`怎么样？为了理解这一点，让我们来看一个只使用一个名为`p`的全局`std::shared_ptr`对象的例子：

```cpp
// Global, how to protect? 
auto p = std::shared_ptr<int>{}; 
```

如何在多个线程中改变`p`而不引入数据竞争？一种选择是在使用`p`时用显式互斥锁保护`p`。或者，我们可以使用`std::atomic`的模板特化来处理`std::shared_ptr`（在 C++20 中引入）。换句话说，可以这样声明`p`为原子共享指针：

```cpp
// Global, protect using atomic
auto p = std::atomic<std::shared_ptr<int>>{}; 
```

这个模板特化可能是锁定的，也可能不是。您可以使用 `is_lock_free()` 成员函数来验证这一点。另一个需要注意的是，特化 `std::atomic<std::shared_ptr<T>>` 是一个例外，它违反了 `std::atomic` 只能用可以平凡复制的类型进行特化的规则。不管怎样，我们很高兴最终在标准库中拥有了这个有用的类型。

以下示例演示了如何从多个线程原子地加载和存储共享指针对象：

```cpp
// Thread T1 calls this function
auto f1() { 
  auto new_p = std::make_shared<int>(std::rand());  // ... 
  p.store(new_p);
} 

// Thread T2 calls this function
auto f2() { 
  auto local_p = p.load(); 
  // Use local_p... 
} 
```

在前面的例子中，我们假设有两个线程 `T1` 和 `T2`，分别调用函数 `f1()` 和 `f2()`。从线程 `T1` 中使用 `std::make_shared<int>()` 调用创建了新的堆分配的 `int` 对象。

在这个例子中有一个微妙的细节需要考虑：堆分配的 `int` 在哪个线程中被删除？当 `f2()` 函数中的 `local_p` 超出范围时，它可能是对 `int` 的最后一个引用（引用计数达到零）。在这种情况下，堆分配的 `int` 将从线程 `T2` 中删除。否则，当调用 `std::atomic_store()` 时，删除将从线程 `T1` 中进行。因此，答案是 `int` 的删除可以从两个线程中进行。

### 原子引用

到目前为止，您已经看到了 `std::atomc_flag` 和 `std::atomic<>` 以及许多有用的特殊化。`std::atomic` 可以用指针进行特殊化，比如 `std::atomic<T*>`，但您还没有看到如何使用引用类型的原子操作。不可能编写 `std::atomic<T&>`；相反，标准库为我们提供了一个名为 `std::atomic_ref` 的模板。

`std::atomic_ref` 模板在 C++20 中引入。它的接口与 `std::atomic` 相同，之所以有一个单独的名称是为了避免影响使用 `std::atomic<T>` 的现有通用代码的风险。

原子引用允许我们对我们拥有引用的非原子对象执行原子操作。当我们引用由客户端或一些不提供内部同步对象的第三方代码提供的对象时，这可能很方便。我们将看一个例子来演示原子引用的有用性。

#### 示例：使用原子引用

假设我们正在编写一个函数，该函数会将硬币翻转指定次数：

```cpp
void flip_coin(std::size_t n, Stats& outcomes); 
```

结果累积在类型为 `Stats` 的 `outcomes` 对象中，它看起来像这样：

```cpp
struct Stats {
  int heads_{};
  int tails_{};
};
std::ostream& operator<<(std::ostream& os, const Stats &s) {
  os << "heads: " << s.heads_ << ", tails: " << s.tails_;
  return os;
} 
```

客户端可以多次调用 `flip_coins()`，使用相同的 `Stats` 实例，翻转的结果将被添加到 `Stats` 中：

```cpp
auto outcomes = Stats{};
flip_coin(30, outcomes); 
flip_coin(10, outcomes); 
```

假设我们想要并行化 `flip_coin()` 的实现，并让多个线程改变 `Stats` 对象。此外，我们可以假设以下情况：

+   `Stats` 结构体无法更改（可能来自第三方库）。

+   我们希望客户端不知道我们的实用函数 `flip_coin()` 是并发的；也就是说，`flip_coin()` 函数的并发应该对调用者完全透明。

对于这个示例，我们将重用我们之前定义的用于生成随机数的函数。

```cpp
int random_int(int min, int max); // See implementation above 
```

现在我们准备定义我们的 `flip_coin()` 函数，它将使用两个线程来翻转硬币 `n` 次：

```cpp
void flip_coin(std::size_t n, Stats &outcomes) {
  auto flip = &outcomes {
    auto heads = std::atomic_ref<int>{outcomes.heads_};
    auto tails = std::atomic_ref<int>{outcomes.tails_};
    for (auto i = 0u; i < n; ++i) {
      random_int(0, 1) == 0 ? ++heads : ++tails;
    }
  };
  auto t1 = std::jthread{flip, n / 2};       // First half
  auto t2 = std::jthread{flip, n - (n / 2)}; // The rest
} 
```

两个线程都会在抛硬币后更新非原子结果对象。我们将创建两个 `std::atomic_ref<int>` 变量，用于原子更新结果对象的成员，而不是使用 `std::mutex`。重要的是要记住，为了保护头和尾计数器免受数据竞争的影响，所有对计数器的并发访问都需要使用 `std::atomic_ref` 进行保护。

以下小程序演示了 `flip_coin()` 函数可以在不了解 `flip_coin()` 的并发实现的情况下被调用：

```cpp
int main() {
  auto stats = Stats{};
  flip_coin(5000, stats);       // Flip 5000 times
  std::cout << stats << '\n';
  assert((stats.tails_ + stats.heads_) == 5000);
} 
```

在我的机器上运行此程序产生了以下输出：

```cpp
heads: 2592, tails: 2408 
```

这个例子结束了我们关于 C++中各种原子类模板的部分。原子操作自 C++11 以来就已经成为标准库的一部分，并且不断发展。C++20 引入了：

+   特化`std::atomic<std::shared_ptr<T>>`

+   原子引用；即`std::atomic_ref<T>`模板

+   等待和通知 API，这是使用条件变量的轻量级替代方案

我们现在将继续讨论 C++内存模型以及它与原子操作和并发编程的关系。

## C++内存模型

为什么在并发章节中我们要谈论 C++的内存模型？内存模型与并发密切相关，因为它定义了内存读写在线程之间如何可见。这是一个相当复杂的主题，涉及编译器优化和多核计算机架构。不过好消息是，如果你的程序没有数据竞争，并且使用原子库默认提供的内存顺序，你的并发程序将遵循一个直观易懂的内存模型。但是，至少了解内存模型是什么以及默认内存顺序保证是很重要的。

这一部分涵盖的概念由 Herb Sutter 在他的演讲*原子武器：C++内存模型和现代硬件 1 和 2*中得到了深入解释。这些演讲可以在[`herbsutter.com/2013/02/11/atomic-weapons-the-c-memory-model-and-modern-hardware/`](https://herbsutter.com/2013/02/11/atomic-weapons-the-c-memory-model-and-modern-hardware/)上免费获取，并且强烈推荐如果你需要更深入地了解这个主题。

### 指令重新排序

理解内存模型的重要性，首先需要了解我们编写的程序实际上是如何执行的一些背景知识。

当我们编写和运行程序时，合理地假设源代码中的指令将按照它们在源代码中出现的顺序执行。这是不正确的。我们编写的代码将在最终执行之前经过多个阶段的优化。编译器和硬件都会重新排序指令，以更有效地执行程序。这并不是新技术：编译器长期以来一直在做这个，这也是为什么优化构建比非优化构建运行得更快的原因之一。编译器（和硬件）可以自由地重新排序指令，只要在运行程序时不可观察到重新排序。程序运行时*好像*一切都按照程序顺序发生。

让我们看一个代码片段的例子：

```cpp
int a = 10;      // 1 
std::cout << a;  // 2 
int b = a;       // 3 
std::cout << b;  // 4 
// Observed output: 1010 
```

在这里，很明显第二行和第三行可以交换而不会引入任何可观察的效果：

```cpp
int a = 10;      // 1 
int b = a;       // 3 This line moved up  
std::cout << a;  // 2 This line moved down 
std::cout << b;  // 4 
// Observed output: 1010 
```

这是另一个例子，类似但不完全相同于*第四章*，*数据结构*中的例子，编译器可以在遍历二维矩阵时优化一个不友好的缓存版本：

```cpp
constexpr auto ksize = size_t{100}; 
using MatrixType = std::array<std::array<int, ksize>, ksize>; 

auto cache_thrashing(MatrixType& matrix, int v) { // 1 
  for (size_t i = 0; i < ksize; ++i)              // 2 
    for (size_t j = 0; j < ksize; ++j)            // 3 
      matrix[j][i] = v;                           // 4 
} 
```

你在*第四章*，*数据结构*中看到，类似这样的代码会产生大量的缓存未命中，从而影响性能。编译器可以通过重新排序`for`语句来优化这个问题，就像这样：

```cpp
auto cache_thrashing(MatrixType& matrix, int v) { // 1 
  for (size_t j = 0; j < ksize; ++j)              // 3 Line moved up 
    for (size_t i = 0; i < ksize; ++i)            // 2 Line moved down 
      matrix[j][i] = v;                           // 4  
} 
```

在执行程序时，无法观察到这两个版本之间的差异，但后者将运行得更快。

编译器和硬件执行的优化（包括指令流水线、分支预测和缓存层次结构）是非常复杂且不断发展的技术。幸运的是，原始程序的所有这些转换都可以看作是源代码中读写的重新排序。这也意味着无论是编译器还是硬件的某个部分执行转换都无关紧要。对于 C++程序员来说，重要的是知道指令可以被重新排序，但没有任何可观察的效果。

如果您一直在尝试调试程序的优化版本，您可能已经注意到很难逐步执行，因为重新排序。因此，通过使用调试器，重新排序在某种意义上是可观察的，但在正常运行程序时是不可观察的。

### 原子操作和内存顺序

在 C++中编写单线程程序时，不会发生数据竞争的风险。我们可以快乐地编写程序，而不必关心指令重新排序。然而，在多线程程序中涉及共享变量时，情况完全不同。编译器（和硬件）基于仅对*一个*线程为真和可观察的内容进行所有优化。编译器无法知道其他线程通过共享变量能观察到什么，因此我们作为程序员的工作就是告知编译器允许进行哪些重新排序。事实上，当我们使用原子变量或互斥锁保护我们免受数据竞争时，这正是我们所做的。

当用互斥锁保护临界区时，可以保证只有当前拥有锁的线程才能执行临界区。但是，互斥锁还在临界区周围创建内存栅栏，以通知系统在临界区边界不允许某些重新排序。在获取锁时，会添加一个“获取”栅栏，在释放锁时，会添加一个“释放”栅栏。

我将用一个例子来证明这一点。假设我们有四条指令：**i1**，**i2**，**i3**和**i4**。它们之间没有依赖关系，因此系统可以任意重新排序指令而不会产生可观察的影响。指令 i2 和 i3 使用共享数据，因此它们是需要通过互斥锁保护的临界区。在添加互斥锁的“获取”和“释放”后，现在有一些重新排序不再有效。显然，我们不能将临界区的指令移出临界区，否则它们将不再受互斥锁的保护。单向栅栏确保没有指令可以从临界区移出。i1 指令可以通过获取栅栏移入临界区，但不能超过释放栅栏。i4 指令也可以通过释放栅栏移入临界区，但不能超过获取栅栏。

下图显示了单向栅栏如何限制指令的重新排序。没有读取或写入指令可以通过获取栅栏上方，也没有任何指令可以通过释放栅栏下方：

![](img/B15619_11_14.png)

图 11.14：单向栅栏限制指令的重新排序

在获取互斥锁时，我们创建了一个获取内存栅栏。它告诉系统不能将内存访问（读取或写入）移动到获取栅栏所在的线以上。系统可以将 i4 指令移动到释放栅栏之上，超过 i3 和 i2 指令，但不能超过获取栅栏。

现在，让我们看看原子变量而不是互斥锁。当我们在程序中使用共享原子变量时，它给我们两件事：

+   **防止写入时出现撕裂**：原子变量始终以原子方式更新，因此读取者无法读取部分写入的值。

+   **通过添加足够的内存栅栏同步内存**：这可以防止某些指令重新排序，以保证原子操作指定的特定内存顺序。

如果我们的程序没有数据竞争，并且在使用原子操作时使用默认的内存顺序，C++内存模型会保证**顺序一致性**。那么，什么是顺序一致性？顺序一致性保证执行的结果与按照原始程序指定的顺序执行操作时的结果相同。线程之间指令的交错是任意的；也就是说，我们无法控制线程的调度。这一开始可能听起来很复杂，但这可能是你已经对并发程序的执行方式有所了解的方式。

顺序一致性的缺点是可能会影响性能。因此，可以使用松散的内存模型来代替原子操作。这意味着你只能获得对撕裂写入的保护，而无法获得顺序一致性提供的内存顺序保证。

我强烈建议你除了默认的顺序一致性内存顺序之外，不要使用其他任何东西，除非你非常了解更弱的内存模型可能引入的影响。

我们不会在这里进一步讨论松散的内存顺序，因为这超出了本书的范围。但值得一提的是，你可能会对知道`std::shared_ptr`中的引用计数器在增加计数时使用了松散模型（但在减少计数时没有使用）。这就是为什么在多线程环境中使用`std::shared_ptr`成员函数`use_count()`时，它只会报告大约的实际引用数量。

内存模型和原子操作非常相关的一个领域是无锁编程。接下来的部分将让你对无锁编程有所了解，并介绍一些应用场景。

# 无锁编程

无锁编程很难。我们不会在本书中花费很多时间讨论无锁编程，而是会为你提供一个非常简单的无锁数据结构的示例。有很多资源（网上和书籍中，比如之前提到的 Anthony Williams 的书）专门讨论无锁编程，这些资源会解释在编写自己的无锁数据结构之前需要理解的概念。一些你可能听说过的概念，比如**比较和交换**（**CAS**）和 ABA 问题，在本书中不会进一步讨论。

## 示例：无锁队列

在这里，你将看到一个无锁队列的示例，这是一个相对简单但有用的无锁数据结构。无锁队列可用于与无法使用锁来同步对共享数据的访问的线程进行单向通信。

由于对队列的要求有限，它只支持*一个读取*线程和*一个写入*线程。队列的容量也是固定的，在运行时无法更改。

无锁队列是一个可能在通常放弃异常的环境中使用的组件的示例。因此，后面的队列设计中没有异常，这使得 API 与本书中其他示例不同。

类模板`LockFreeQueue<T>`具有以下公共接口：

+   `push()`: 将一个元素添加到队列中，并在成功时返回`true`。这个函数只能被（唯一的）*写入线程*调用。为了避免在客户端提供右值时进行不必要的复制，`push()`重载了`const T&`和`T&&`。这种技术也在本章前面介绍的`BoundedBuffer`类中使用过。

+   `pop()`: 返回一个`std::optional<T>`，其中包含队列的第一个元素，除非队列为空。这个函数只能被（唯一的）*读取线程*调用。

+   `size()`: 返回队列的当前大小。这个函数可以被*两个线程*同时调用。

以下是队列的完整实现：

```cpp
template <class T, size_t N>
class LockFreeQueue {
  std::array<T, N> buffer_{};   // Used by both threads
  std::atomic<size_t> size_{0}; // Used by both threads
  size_t read_pos_{0};          // Used by reader thread
  size_t write_pos_{0};         // Used by writer thread
  static_assert(std::atomic<size_t>::is_always_lock_free);
  bool do_push(auto&& t) {      // Helper function
    if (size_.load() == N) { 
      return false; 
    }
    buffer_[write_pos_] = std::forward<decltype(t)>(t);
    write_pos_ = (write_pos_ + 1) % N;
    size_.fetch_add(1);
    return true;
  }
public:
  // Writer thread
  bool push(T&& t) { return do_push(std::move(t)); }
  bool push(const T& t) { return do_push(t); }
  // Reader thread
  auto pop() -> std::optional<T> {
    auto val = std::optional<T>{};    
    if (size_.load() > 0) {
      val = std::move(buffer_[read_pos_]);
      read_pos_ = (read_pos_ + 1) % N;
      size_.fetch_sub(1);
    }
    return val;
  }
  // Both threads can call size()
  auto size() const noexcept { return size_.load(); }
}; 
```

唯一需要原子访问的数据成员是`size_`变量。`read_pos_`成员仅由读取线程使用，`write_pos_`仅由写入线程使用。那么`std::array`类型的缓冲区呢？它是可变的，并且被两个线程访问？这不需要同步吗？由于算法确保两个线程永远不会同时访问数组中的相同元素，C++保证可以在没有数据竞争的情况下访问数组中的单个元素。元素有多小都没关系；即使是`char`数组也具有这一保证。

这种非阻塞队列何时会有用？一个例子是在音频编程中，当主线程上运行着一个 UI 需要与实时音频线程发送或接收数据时，实时线程在任何情况下都不能阻塞。实时线程不能使用互斥锁，分配/释放内存，或执行任何可能导致线程等待低优先级线程的操作。这些情况下需要无锁数据结构。

在`LockFreeQueue`中，读取器和写入器都是无锁的，因此我们可以有两个队列实例在主线程和音频线程之间双向通信，如下图所示：

![](img/B15619_11_15.png)

图 11.15：使用两个无锁队列在主线程和实时音频线程之间传递状态

正如前面提到的，本书只是浅尝辄止无锁编程的表面。现在是时候用一些关于编写并发程序时性能的指南来结束本章了。

# 性能指南

我无法强调在尝试提高性能之前，正确运行并发程序的重要性。此外，在应用与性能相关的任何指南之前，您首先需要建立一种可靠的方式来衡量您要改进的内容。

## 避免争用

每当多个线程使用共享数据时，就会发生争用。争用会影响性能，有时由争用引起的开销会使并行算法的工作速度比单线程替代方案更慢。

使用导致等待和上下文切换的锁是明显的性能惩罚，但同样不明显的是，锁和原子操作都会禁用编译器生成的代码中的优化，并且在 CPU 执行代码时会在运行时这样做。这是为了保证顺序一致性。但请记住，这类问题的解决方案绝不是忽略同步，从而引入数据竞争。数据竞争意味着未定义行为，拥有快速但不正确的程序不会让任何人满意。

相反，我们需要尽量减少在关键部分的时间。我们可以通过更少地进入关键部分，并通过尽量减少关键部分本身来做到这一点，以便一旦进入关键部分，我们就尽快离开它。

## 避免阻塞操作

要编写一个现代响应式 UI 应用程序，始终保持流畅运行，绝对不能阻塞主线程超过几毫秒。一个流畅运行的应用程序每秒更新其界面 60 次。这意味着如果您正在做一些阻塞 UI 线程超过 16 毫秒的事情，FPS 将会下降。

您可以在设计应用程序的内部 API 时考虑这一点。每当编写执行 I/O 或可能需要超过几毫秒的其他操作的函数时，它需要被实现为异步函数。这种模式在 iOS 和 Windows 中变得非常普遍，例如，所有网络 API 都已变成异步。

## 线程数/CPU 核心数

机器的 CPU 核心越多，您可以运行的活动线程就越多。如果您设法将顺序的 CPU 绑定任务拆分为并行版本，您可以通过多个核心并行处理任务来提高性能。

从单线程算法转变为可以由两个线程运行的算法，在最佳情况下可能会使性能翻倍。但是，添加越来越多的线程后，最终会达到一个极限，此时不会再有性能增益。超过该极限添加更多线程实际上会降低性能，因为上下文切换引起的开销会随着添加的线程数量增加而变得更加显著。

例如，I/O 密集型任务，例如等待网络数据的网络爬虫，在达到 CPU 过度订阅的极限之前需要大量线程。等待 I/O 的线程很可能会从 CPU 中切换出来，以为其他准备执行的线程腾出空间。对于 CPU 密集型任务，通常没有必要使用超过机器上核心数量的线程。

控制大型程序中的线程总数可能很困难。控制线程数量的一个好方法是使用可以根据当前硬件大小调整大小的线程池。

在*第十四章*，*并行算法*中，您将看到如何并行化算法的示例，以及如何根据 CPU 核心数量调整并发量。

## 线程优先级

线程的优先级会影响线程的调度。具有高优先级的线程可能比具有较低优先级的线程更频繁地被调度。线程优先级对降低任务的延迟很重要。

操作系统提供的线程通常具有优先级。目前，使用当前的 C++线程 API 无法设置线程的优先级。但是，通过使用`std::thread::native_handle`，您可以获取到底层操作系统线程的句柄，并使用本机 API 来设置优先级。

与线程优先级相关的一种可能会影响性能并且应该避免的现象称为**优先级反转**。当一个具有高优先级的线程正在等待获取当前由低优先级线程持有的锁时，就会发生这种情况。这种依赖关系会影响高优先级线程，因为它被阻塞，直到下一次低优先级线程被调度以释放锁。

对于实时应用程序来说，这是一个大问题。实际上，这意味着您不能使用锁来保护需要实时线程访问的任何共享资源。例如，生成实时音频的线程以最高可能的优先级运行，为了避免优先级反转，不可能让音频线程调用任何可能阻塞并引起上下文切换的函数（包括`std::malloc()`）。

## 线程亲和性

线程亲和性使得调度程序可以提示哪些线程可以受益于共享相同的 CPU 缓存。换句话说，这是对调度程序的请求，如果可能的话，一些线程应该在特定的核心上执行，以最小化缓存未命中。

为什么要让一个线程在特定的核心上执行？答案是（再次）缓存。在相同内存上操作的线程可能会受益于在同一核心上运行，从而利用热缓存。对于调度程序来说，这只是分配线程到核心时需要考虑的众多参数之一，因此这几乎不是任何保证，但是，操作系统之间的行为差异非常大。线程优先级，甚至利用所有核心（以避免过热）是现代调度程序需要考虑的要求之一。

使用当前的 C++ API 无法以便携的方式设置线程亲和性，但大多数平台支持在线程上设置亲和性掩码的某种方式。为了访问特定于平台的功能，您需要获取本机线程的句柄。接下来的示例演示了如何在 Linux 上设置线程亲和性掩码：

```cpp
#include <pthreads> // Non-portable header 
auto set_affinity(const std::thread& t, int cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  pthread_t native_thread = t.native_handle(); 
  pthread_set_affinity(native_thread, sizeof(cpu_set_t), &cpuset); 
} 
```

请注意，这不是便携式的 C++，但如果您正在进行性能关键的并发编程，很可能需要对线程进行一些不便携式的配置。

## 虚假共享

**虚假共享**，或者破坏性干扰，可能会严重降低性能。当两个线程使用一些数据（这些数据在逻辑上不共享）但碰巧位于同一个缓存行时，就会发生虚假共享。想象一下，如果两个线程在不同的核心上执行，并且不断更新位于共享缓存行上的变量，会发生什么。尽管线程之间没有真正共享数据，但它们会相互使缓存行失效。

虚假共享很可能发生在使用全局数据或动态分配的数据在线程之间共享时。一个可能发生虚假共享的例子是分配一个数组，该数组在线程之间共享，但每个线程只使用数组的一个元素。

解决这个问题的方法是对数组中的每个元素进行填充，以便相邻的两个元素不能位于同一个缓存行上。自 C++17 以来，有一种便携式的方法可以使用`<new>`中定义的`std::hardware_destructive_interference_size`常量和`alignas`说明符来实现这一点。以下示例演示了如何创建一个元素来防止虚假共享：

```cpp
struct alignas(std::hardware_destructive_interference_size) Element {
   int counter_{};
}; 

auto elements = std::vector<Element>(num_threads); 
```

现在，向量中的元素被保证位于不同的缓存行上。

# 总结

在本章中，您已经学会了如何创建可以同时执行多个线程的程序。我们还介绍了如何通过使用锁或原子操作来保护关键部分，以避免数据竞争。您了解到 C++20 带来了一些有用的同步原语：屏障、障碍和信号量。然后我们研究了执行顺序和 C++内存模型，在编写无锁程序时理解这些内容变得很重要。您还发现了不可变数据结构是线程安全的。本章最后介绍了一些改进并发应用程序性能的指南。

接下来的两章专门介绍了一个全新的 C++20 特性，称为协程，它允许我们以顺序方式编写异步代码。
