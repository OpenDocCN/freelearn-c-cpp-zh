# 5

# 原子操作

在第四章中，我们学习了基于锁的线程同步。我们学习了互斥锁、条件变量以及其他基于锁的线程同步原语，这些都是基于获取和释放锁的。这些同步机制建立在*原子类型和操作*之上，这是本章的主题。

我们将研究原子操作是什么，以及它们与基于锁的同步原语有何不同。阅读完本章后，你将具备原子操作的基本知识以及它们的一些应用。基于原子操作的锁免费（不使用锁）同步是一个非常复杂的话题，需要多年的时间来掌握，但我们将为你提供一个我们希望对主题有良好介绍的入门。

在本章中，我们将涵盖以下主要内容：

+   什么是原子操作？

+   C++内存模型简介

+   C++标准库提供了哪些原子类型和操作？

+   一些原子操作的示例，从用于收集统计信息的简单计数器到一个基本的类似互斥锁的全**单生产者单消费者**（**SPSC**）无锁有界队列

# 技术要求

你需要一个支持 C++20 的较新版本的 C++编译器。一些简短的代码示例将通过链接到非常有用的 godbolt 网站（[`godbolt.org`](https://godbolt.org)）提供。对于完整的代码示例，我们将使用书籍仓库，该仓库可在[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)找到。

示例可以在本地编译和运行。我们已经在运行 Linux（Ubuntu 24.04 LTS）的 Intel CPU 计算机上测试了代码。对于原子操作，尤其是内存排序（关于这一点将在本章后面详细说明），Intel CPU 与 Arm CPU 不同。

请注意，此处代码性能和性能分析将是第十三章的主题。我们将在本章中仅对性能做一些简要说明，以避免使内容过于冗长。

# 原子操作简介

原子操作是不可分割的（因此得名原子，源自希腊语*ἄτομος*，*atomos*，不可分割）。

在本节中，我们将介绍原子操作，它们是什么，以及使用（以及不使用！）它们的一些原因。

## 原子操作与非原子操作——示例

如果你还记得第四章中的简单计数器示例，我们需要使用同步机制（我们使用了互斥锁）来修改计数器变量，以避免竞态条件。竞态条件的原因是增加计数器需要三个操作：读取计数器值，增加它，并将修改后的计数器值写回内存。如果我们能一次性完成这些操作，就不会有竞态条件。

这正是原子操作所能实现的效果：如果我们有一种**atomic_increment**操作，每个线程都会在一个指令中读取、增加并写入计数器，从而避免竞争条件，因为在任何时刻，增加计数器都会被完全完成。当我们说完全完成时，意味着每个线程要么增加计数器，要么什么都不做，使得在计数器增加操作中途的中断成为不可能。

以下两个示例仅用于说明目的，并且不是多线程的。我们在这里只关注操作，无论是原子的还是非原子的。

让我们在代码中看看这个。对于以下示例中的 C++代码和生成的汇编语言，请参考[`godbolt.org/z/f4dTacsKW`](https://godbolt.org/z/f4dTacsKW) ：

```cpp
int counter {0};
int main() {
    counter++;
    return 0;
}
```

代码增加了一个全局计数器。现在让我们看看编译器生成的汇编代码以及 CPU 执行了哪些指令（完整的汇编代码可以在之前的链接中找到）：

```cpp
    Mov    eax, DWORD PTR counter[rip]
    Add    eax, 1
    Move    DWORD PTR counter[rip], eax
```

**[1]** 将存储在**counter**中的值复制到**eax**寄存器，**[2]** 将存储在**eax**中的值增加**1**，最后，**[3]** 将**eax**寄存器的内容复制回**counter**变量。因此，一个线程可以执行**[1]**然后被调度出去，而另一个线程在之后执行所有三个指令。当第一个线程完成增加结果后，计数器只会增加一次，因此结果将是错误的。

以下代码执行相同的操作：它增加了一个全局计数器。不过，这次它使用了原子类型和操作。要获取以下示例中的代码和生成的汇编代码，请参考[`godbolt.org/z/9hrbo31vx`](https://godbolt.org/z/9hrbo31vx)：

```cpp
#include <atomic>
std::atomic<int> counter {0};
int main() {
    counter++;
    return 0;
}
```

我们将在后面解释**std::atomic<int>**类型和原子增加操作。

生成的汇编代码如下：

```cpp
    lock add    DWORD PTR counter[rip], 1
```

只生成了一条指令来将**counter**变量中的值增加**1**。这里的**lock**前缀意味着接下来的指令（在这种情况下是**add**）将被原子执行。因此，在这个第二个示例中，一个线程在增加计数器过程中不能被中断。作为旁注，一些 Intel x64 指令是原子执行的，并且不使用**lock**前缀。

原子操作允许线程以不可分割的方式读取、修改（例如，增加一个值）和写入，也可以用作同步原语（类似于我们在*第四章*中看到的互斥锁）。实际上，我们在这本书中看到的所有基于锁的同步原语都是使用原子操作实现的。原子操作必须由 CPU 提供（如**lock** **add**指令）。

在本节中，我们介绍了原子操作，定义了它们是什么，并研究了通过查看编译器生成的汇编指令来实现的非常简单的例子。在下一节中，我们将探讨原子操作的一些优缺点。

## 何时使用（以及何时不使用）原子操作

使用原子操作是一个复杂的话题，它可能非常困难（或者至少相当棘手）要掌握。这需要大量的经验，我们参加了一些关于这个主题的课程，并被建议不要这样做！无论如何，您总是可以学习基础知识并在实践中进行实验。我们希望这本书能帮助您在学习之旅中取得进步。

原子操作可以在以下情况下使用：

+   **如果多个线程共享可变状态**：需要同步线程的情况最为常见。当然，可以使用互斥锁等锁，但在某些情况下，原子操作将提供更好的性能。请注意，然而，使用原子操作*并不*保证更好的性能。

+   **如果对共享状态的同步访问是细粒度的**：如果我们必须同步的数据是一个整数、指针或任何其他 C++内建类型的变量，那么使用原子操作可能比使用锁更好。

+   **为了提高性能**：如果您想达到最大性能，那么原子操作可以帮助减少线程上下文切换（参见*第二章*）并减少锁引入的开销，从而降低延迟。请记住，始终对代码进行性能分析以确保性能得到提升（我们将在*第十三章*中深入探讨）。

锁可以在以下情况下使用：

+   **如果受保护的数据不是细粒度的**：例如，我们正在同步访问一个大于 8 字节（在现代 CPU 上）的数据结构或对象。

+   **如果性能不是问题**：锁的使用和推理要简单得多（在某些情况下，使用锁比使用原子操作性能更好）。

+   **为了避免需要获取底层知识**：要从原子操作中获得最大性能，需要大量的底层知识。我们将在“*C++内存模型*”部分介绍其中的一些内容。

我们刚刚学习了何时使用原子操作以及何时不使用。一些应用程序，如低延迟/高频交易系统，需要最大性能并使用原子操作以实现最低的延迟。大多数应用程序通过锁同步将正常工作。

在下一节中，我们将研究阻塞和非阻塞数据结构之间的差异以及一些相关概念的定义。

# 非阻塞数据结构

在*第四章*中，我们研究了同步队列的实现。我们使用了互斥锁和条件变量作为同步原语。与锁同步的数据结构被称为**阻塞数据结构**，因为线程会被操作系统*阻塞*（等待锁变为可用）。

不使用锁的数据结构被称为**非阻塞数据结构**。大多数（但并非所有）都是无锁的。

如果每个同步操作都在有限步骤内完成，不允许无限期等待条件变为真或假，则数据结构或算法被认为是无锁的。

无锁数据结构的类型如下：

+   **无阻塞**：如果所有其他线程都处于挂起状态，则线程将在有限步骤内完成其操作。

+   **无锁**：在多个线程同时工作在数据结构上时，线程将在有限步骤内完成其操作。

+   **无等待**：在多个线程同时工作在数据结构上时，所有线程将在有限步骤内完成其操作。

实现无锁数据结构非常复杂，在实施之前，我们需要确保这是必要的。使用无锁数据结构的原因如下：

+   **实现最大并发性**：如我们之前所看到的，当数据访问同步涉及细粒度数据（如原生类型变量）时，原子操作是一个很好的选择。根据前面的定义，无锁数据结构将允许至少一个访问数据结构的线程在有限步骤内取得一些进展。无等待结构将允许所有访问数据结构的线程在有限步骤内取得一些进展。

    然而，当我们使用锁时，一个线程会拥有锁，而其他线程则只是在等待锁变为可用，因此无锁数据结构可实现的并发性可以更好。

+   **无死锁**：因为没有涉及锁，所以我们的代码中不可能有任何死锁。

+   **性能**：某些应用程序必须实现尽可能低的延迟，因此等待锁可能是不可以接受的。当线程尝试获取锁，而锁不可用时，操作系统会阻塞该线程。在线程被阻塞期间，调度器需要进行上下文切换以能够调度另一个线程进行执行。这些上下文切换需要时间，而在低延迟应用程序（如高性能网络数据包接收/处理器）中，这些时间可能太多。

我们现在已经了解了阻塞和非阻塞数据结构是什么，以及无锁代码是什么。在下一节中，我们将介绍 C++内存模型。

# C++内存模型

本节解释了 C++内存模型及其如何处理并发。C++内存模型从 C++11 开始引入，并定义了 C++内存的两大主要特性：

+   对象在内存中的布局（即结构方面）。这个主题不会在本书中介绍，因为本书是关于异步编程的。

+   内存修改顺序（即并发方面）。我们将看到内存模型中指定的不同内存修改顺序。

## 内存访问顺序

在我们解释 C++内存模型及其支持的不同的内存排序之前，让我们明确我们所说的内存排序是什么。内存排序指的是内存（即程序中的变量）被访问的顺序。内存访问可以是读取或写入（加载和存储）。但是，程序变量的实际访问顺序是什么？对于以下代码，有三个观点：所写的代码顺序、编译器生成的指令顺序，最后是 CPU 执行指令的顺序。这三个排序都可以相同，或者（更可能）不同。

第一种和最明显的排序是代码中的排序。以下代码片段是一个例子：

```cpp
void func_a(int& a, int& b) {
    a += 1;
    b += 10;
    a += 2;
}
```

**func_a**函数首先将 1 加到变量**a**上，然后加 10 到变量**b**上，最后将 2 加到变量**a**上。这是我们想要的方式，也是我们定义要执行语句的顺序。

编译器将前面的代码转换为汇编指令。如果代码执行的结果不变，编译器可以改变我们语句的顺序，以使生成的代码更高效。例如，对于前面的代码，编译器可以首先对变量**a**执行两个加法操作，然后对变量**b**执行加法操作，或者它可以直接将 3 加到**a**上，然后加 10 到**b**上。正如我们之前提到的，如果结果是相同的，编译器可以执行任何操作来优化代码。

现在让我们考虑以下代码：

```cpp
void func_a(int& a, int& b) {
    a += 1;
    b += 10 + a;
    a += 2;
}
```

在这种情况下，对**b**的操作依赖于对**a**的先前操作，因此编译器不能重新排序语句，生成的代码将与我们所写的代码（操作顺序相同）一样。

CPU（本书中使用的 CPU 是现代的 Intel x64 CPU）将运行生成的代码。它可以以不同的顺序执行编译器生成的指令。这被称为乱序执行。如果结果是正确的，CPU 可以再次这样做。

有关前例中显示的生成代码的链接：[`godbolt.org/z/Mhrcnsr9e`](https://godbolt.org/z/Mhrcnsr9e)

首先，为**func_1**生成的指令显示了优化：编译器通过在一个指令中将 3 加到变量**a**上，将两个加法操作合并为一个。其次，为**func_2**生成的指令与我们所写的 C++语句的顺序相同。在这种情况下，CPU 可以执行指令的乱序执行，因为操作之间没有依赖关系。

总结来说，我们可以这样说，CPU 将要运行的代码可能与我们所写的代码不同（再次强调，前提是执行结果与我们在程序中预期的相同）。

我们所展示的所有示例都适用于单线程运行的代码。代码指令的执行顺序可能因编译器优化和 CPU 的乱序执行而不同，但结果仍然正确。

以下代码展示了乱序执行的示例：

```cpp
    mov    eax, [var1]  ; load variable var1 into reg eax
    inc    eax          ; eax += 1
    mov    [var1], eax  ; store reg eax into var1
    xor    ecx, ecx     ; ecx = 0
    inc    ecx          ; ecx += 1
    add    eax, ecx     ; eax = eax + ecx
```

CPU 可能会按照前面代码中显示的顺序执行指令，即**load var1 [1]**。然后，在变量被读取的同时，它可能会执行一些后续指令，例如**[4]**和**[5]**，然后，一旦**var1**被读取，执行**[2]**，然后**[3]**，最后，**[6]**。指令的执行顺序不同，但结果仍然是相同的。这是一个典型的乱序执行示例：CPU 发出一个加载指令，而不是等待数据可用，它会执行一些其他指令，如果可能的话，以避免空闲并最大化性能。

我们所提到的所有优化（包括编译器和 CPU）都是在不考虑线程间交互的情况下进行的。编译器和 CPU 都不知道不同的线程。在这些情况下，我们需要告诉编译器它可以做什么，不可以做什么。原子操作和锁是实现这一点的途径。

当例如我们使用原子变量时，我们可能不仅需要操作是原子的，还需要在多线程运行时遵循一定的顺序以确保代码能够正确工作。这不能仅仅通过编译器或 CPU 来完成，因为它们都没有涉及多个线程的信息。为了指定我们想要使用的顺序，C++内存模型提供了不同的选项：

+   **宽松** **排序** : **std::memory_order_relaxed**

+   **获取和释放排序** : **std::memory_order_acquire** , **std::memory_order_release** , **std::memory_order_acq_rel** , 和 **std::memory_order_consume**

+   **顺序一致性** **排序** : **std::memory_order_seq_cst**

C++内存模型定义了一个抽象机以实现与任何特定 CPU 的独立性。然而，CPU 仍然存在，内存模型中可用的功能可能不会适用于特定的 CPU。例如，Intel x64 架构相当限制性，并强制执行相当强的内存顺序。

Intel x64 架构使用一个处理器排序的内存排序模型，可以定义为*写入排序并带有存储缓冲区转发*。在单处理器系统中，内存排序模型遵循以下原则：

+   读取不会与任何读取操作重排

+   写入不会与任何写入操作重排

+   写入不会与较旧的读取操作重排

+   读取可能与较旧的写入操作重排（如果要重排的读取和写入操作涉及不同的内存位置）

+   读取和写入操作不会与锁定（原子）指令重新排序

更多详细信息请参阅英特尔手册（见本章末尾的参考文献），但前面的原则是最相关的。

在多处理器系统中，以下原则适用：

+   每个单独的处理器使用与单处理器系统相同的序原则

+   单个处理器的写入操作被所有处理器以相同的顺序观察到

+   来自单个处理器的写入操作不会与其他处理器的写入操作进行排序

+   内存序遵循因果关系

+   除了执行写入操作的处理器之外，任何两个存储操作都以一致的顺序被其他处理器观察到

+   锁定（原子）指令具有总序

英特尔架构是强序的；每个处理器的存储操作（写指令）按照它们执行时的顺序被其他处理器观察到，并且每个处理器按照程序中出现的顺序执行存储操作。这被称为**总存储序**（**TSO**）。

ARM 架构支持**弱序**（**WO**）。以下是主要原则：

+   读取和写入可以无序执行。与 TSO 不同，正如我们所看到的，除了写入不同地址后的读取外，没有局部重新排序，ARM 架构允许局部重新排序（除非使用特殊指令另行指定）。

+   写入操作不一定能像在英特尔架构中那样同时被所有线程看到。

+   通常，这种相对非限制性的内存序允许核心更自由地重新排序指令，从而可能提高多核性能。

我们必须在这里说明，内存序越宽松，对执行代码的推理就越困难，正确同步多个线程使用原子操作就变得更加具有挑战性。此外，您应该记住，无论内存序如何，原子性总是得到保证。

在本节中，我们已经了解了访问内存时序的含义以及我们在代码中指定的序可能与 CPU 执行代码的序不同。在下一节中，我们将看到如何使用原子类型和操作强制某些序。

## 强制序

我们已经在*第四章*以及本章前面的内容中看到，来自不同线程的同一内存地址上的非原子操作可能会导致数据竞争和未定义的行为。为了强制线程间操作的序，我们将使用原子类型及其操作。本节将探讨原子在多线程代码中的使用所达到的效果。

以下简单的示例将帮助我们了解可以使用原子操作做什么：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
std::string message;
std::atomic<bool> ready{false};
void reader() {
    using namespace std::chrono::literals;
    while (!ready.load()) {
        std::this_thread::sleep_for(1ms);
    }
    std::cout << "Message received = " << message << std::endl;
}
void writer() {
    message = "Hello, World!";
    ready.store(true);
}
int main() {
    std::thread t1(reader);
    std::thread t2(writer);
    t1.join();
    t2.join();
    return 0;
}
```

在这个例子中，**reader()** 等待直到 **ready** 变量变为 **true**，然后打印由 **writer()** 设置的消息。**writer()** 函数设置消息并将 **store** 变量设置为 **true**。

原子操作为我们提供了两个特性，用于在多线程代码中强制执行特定的执行顺序：

+   **发生之前**：在先前的代码中，**[1]**（设置**message**变量）发生在**[2]**（将原子**ready**变量设置为**true**）之前。同样，**[3]**（在循环中读取**ready**变量直到其为**true**）发生在**[4]**（打印消息）之前。在这种情况下，我们使用顺序一致性内存顺序（默认内存顺序）。

+   **同步于**：这仅在原子操作之间发生。在先前的例子中，这意味着当**ready**由**[1]**设置时，其值将对后续不同线程中的读取（或写入）可见（当然，它对当前线程也是可见的），当**ready**由**[3]**读取时，更改后的值将可见。

现在我们已经看到了原子操作如何强制从不同线程执行内存访问顺序，让我们详细看看 C++内存模型提供的每个内存顺序选项。

在我们开始之前，让我们在这里记住，英特尔 x64 架构（英特尔和 AMD 的桌面处理器）在内存顺序方面相当严格，不需要任何额外的 acquire/release 指令，并且顺序一致性在性能成本方面是低廉的。

## 顺序一致性

顺序一致性保证了程序按你编写的方式执行。在 1979 年，莱斯利·兰波特将顺序一致性定义为“*执行的结果与读取和写入发生某种顺序的结果相同，并且每个处理器的操作以* *其程序指定的顺序* *出现在这个序列中。*”

在 C++中，顺序一致性通过**std::memory_order_seq_cst**选项指定。这是最严格的内存顺序，也是默认的。如果没有指定顺序选项，则将使用顺序一致性。

C++的内存模型默认确保在代码中不存在竞态条件时保持顺序一致性。将其视为一种协议：如果我们正确同步我们的程序以防止竞态条件，C++将保持程序按编写顺序执行的表象。

在此模型中，所有线程必须看到相同的操作顺序。只要计算的可见结果与无序代码的结果相同，操作仍然可以重新排序。如果读取和写入的顺序与编译代码中的顺序相同，则指令和操作可以重新排序。如果满足依赖关系，CPU 可以在读取和写入之间自由重新排序任何其他指令。由于它定义了一致的顺序，顺序一致性是最直观的排序形式。为了说明顺序一致性，让我们考虑以下示例：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
std::atomic<bool> x{ false };
std::atomic<bool> y{ false };
std::atomic<int> z{ 0 };
void write_x() {
    x.store(true, std::memory_order_seq_cst);
}
void write_y() {
    y.store(true, std::memory_order_seq_cst);
}
void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) {}
    if (y.load(std::memory_order_seq_cst)) {
        ++z;
    }
}
void read_y_then_x()
{
    while (!y.load(std::memory_order_seq_cst)) {}
    if (x.load(std::memory_order_seq_cst)) {
        ++z;
    }
}
int main() {
    std::thread t1(write_x);
    std::thread t2(write_y);
    std::thread t3(read_x_then_y);
    std::thread t4(read_y_then_x);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    if (z.load() == 0) {
        std::cout << "This will never happen\n";
    }
    {
        std::cout << "This will always happen and z = " << z << "\n";
    }
    return 0;
}
```

由于我们在运行代码时使用 **std::memory_order_seq_cst**，我们应该注意以下事项：

+   每个线程中的操作按给定顺序执行（不重新排序原子操作）。

+   **t1** 和 **t2** 按顺序更新 **x** 和 **y**，而 **t3** 和 **t4** 看到相同的顺序。如果没有这个属性，**t3** 可能会看到 **x** 和 **y** 的顺序变化，但 **t4** 可能会看到相反的顺序。

+   任何其他排序都可能打印 **This will never happen**，因为 **t3** 和 **t4** 可能会看到 **x** 和 **y** 的变化顺序相反。我们将在下一节中看到这个示例。

此例中的顺序一致性意味着以下两个事情将会发生：

+   每个存储操作都被所有线程看到；也就是说，每个存储操作与每个变量的所有加载操作同步，所有线程以相同的顺序看到这些变化。

+   每个线程的操作顺序相同（操作顺序与代码中的顺序相同）

请注意，不同线程中操作的顺序没有保证，并且来自不同线程的指令可能以任何顺序执行，因为线程可能被调度。

## 获取-释放排序

**获取-释放排序** 比顺序一致性排序更宽松。我们不会得到与顺序一致性排序相同的操作总顺序，但仍然可以进行一些同步。一般来说，随着我们增加内存排序的自由度，我们可能会看到性能提升，但推理代码的执行顺序将变得更加困难。

在此排序模型中，原子加载操作是 **std::memory_order_acquire** 操作，原子存储操作是 **std::memory_order_release** 操作，原子读-改-写操作可能是 **std::memory_order_acquire**、**std::memory_order_release** 或 **std::memory_order_acq_rel** 操作。

**获取语义**（与 **std::memory_order_acquire** 一起使用）确保源代码中出现在获取操作之后的线程中的所有读取或写入操作都在获取操作之后发生。这防止内存重新排序获取操作之后的读取和写入。

**释放语义**（与**std::memory_order_release**一起使用）确保在源代码中的释放操作之前完成的读取或写入操作在释放操作之前完成。这防止了释放操作之后的读取和写入的内存重排。

以下示例显示了与上一节关于顺序一致性的示例相同的代码，但在此情况下，我们使用原子操作的获取-释放内存顺序：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
std::atomic<bool> x{ false };
std::atomic<bool> y{ false };
std::atomic<int> z{ 0 };
void write_x() {
    x.store(true, std::memory_order_release);
}
void write_y() {
    y.store(true, std::memory_order_release);
}
void read_x_then_y() {
    while (!x.load(std::memory_order_acquire)) {}
    if (y.load(std::memory_order_acquire)) {
        ++z;
    }
}
void read_y_then_x() {
    while (!y.load(std::memory_order_acquire)) {}
    if (x.load(std::memory_order_acquire)) {
        ++z;
    }
}
int main() {
    std::thread t1(write_x);
    std::thread t2(write_y);
    std::thread t3(read_x_then_y);
    std::thread t4(read_y_then_x);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    if (z.load() == 0) {
        std::cout << "This will never happen\n";
    }
    {
        std::cout << "This will always happen and z = " << z << "\n";
    }
    return 0;
}
```

在这种情况下，**z**的值可能是 0。因为我们不再具有顺序一致性，在**t1**将**x**设置为**true**和**t2**将**y**设置为**true**之后，**t3**和**t4**可能对内存访问的不同看法。由于使用了获取-释放内存排序，**t3**可能看到**x**为**true**和**y**为**false**（记住，没有强制排序），而**t4**可能看到**x**为**false**和**y**为**true**。当这种情况发生时，**z**的值将是 0。

除了**std::memory_order_acquire**、**std::memory_order_release**和**std::memory_order_acq_rel**之外，获取-释放内存排序还包括**std::memory_order_consume**选项。我们不会对其进行描述，因为根据在线 C++参考，“*释放-消费排序的规范正在修订，std::memory_order_consume 的使用* *暂时不鼓励*。”

## 松弛内存排序

要执行具有**松弛内存排序**的原子操作，我们将**std::memory_order_relaxed**指定为内存顺序选项。

松弛内存排序是最弱形式的同步。它提供两个保证：

+   操作的原子性。

+   单个线程中同一原子变量的原子操作不会被重排。这被称为**修改顺序一致性**。然而，没有保证其他线程将以相同的顺序看到这些操作。

让我们考虑以下场景：一个线程（**th1**）将值存储到一个原子变量中。在一定的随机时间间隔后，该变量将被新的随机值覆盖。为了这个示例的目的，我们应该假设写入的顺序是 2、12、23、4、6。另一个线程，**th2**，定期读取相同的变量。第一次读取变量时，**th2**得到值 23。记住，该变量是原子的，并且加载和存储操作都是使用松弛内存顺序完成的。

如果**th2**再次读取该变量，它可以获取与之前读取的相同值或任何在之前读取值之后写入的值。它不能读取任何在之前写入的值，因为这会违反修改顺序一致性属性。在当前示例中，第二次读取可能得到 23、4 或 6，但不能得到 2 或 12。如果我们得到 4，th1 将继续写入 8、19 和 7。现在 th2 可能得到 4、6、8、19 或 7，但不能得到 4 之前的任何数字等等。

在两个或多个线程之间，没有保证任何顺序，但一旦读取了一个值，就不能再读取之前写入的值。

松弛模型不能用于线程同步，因为没有可见性顺序保证，但在操作不需要在线程之间紧密协调的场景中很有用，这可以提高性能。

当执行顺序不影响程序的正确性时，通常可以安全使用，例如用于统计的计数器或引用计数器，其中增量顺序的精确性并不重要。

在本节中，我们学习了 C++ 内存模型以及它是如何允许具有不同内存顺序约束的原子操作进行顺序和同步的。在下一节中，我们将看到 C++ 标准库提供的原子类型和操作。

# C++ 标准库原子类型和操作

现在我们将介绍 C++ 标准库提供的支持原子类型和操作的数据类型和函数。正如我们已经看到的，原子操作是一个不可分割的操作。要在 C++ 中执行原子操作，我们需要使用 C++ 标准库提供的原子类型。

## C++ 标准库原子类型

C++ 标准库提供的原子类型定义在 **<atomic>** 头文件中。

你可以在在线 C++ 参考中查看定义在 **<atomic>** 头文件中的所有原子类型的文档，你可以通过 [`en.cppreference.com/w/cpp/atomic/atomic`](https://en.cppreference.com/w/cpp/atomic/atomic) 访问。我们不会在这里包含所有内容（这就是参考的作用！），但我们将介绍主要概念和使用示例，以进一步阐述我们的解释。

C++ 标准库提供的原子类型如下：

+   **std::atomic_flag**：原子布尔类型（但与 **std::atomic<bool>** 不同）。它是唯一保证无锁的原子类型。它不提供加载或存储操作。它是所有原子类型中最基本的。我们将用它来实现一个非常简单的类似互斥锁的功能。

+   **std::atomic<T>**：这是一个用于定义原子类型的模板。所有内建类型都使用此模板定义了自己的原子类型。以下是一些这些类型的示例：

    +   **std::atomic<bool>**（及其别名 **atomic_bool**）：我们将使用此原子类型来实现从多个线程中懒加载变量的一次性初始化。

    +   **std::atomic<int>**（及其别名 **atomic_int**）：我们已经在简单的计数器示例中看到了这个原子类型。我们将在另一个示例中使用它来收集统计数据（与计数器示例非常相似）。

    +   **std::atomic<intptr_t>**（及其别名 **atomic_intptr_t**）。

    +   C++20 引入了原子智能指针：**std::atomic<std::shared_ptr<U>>** 和 **std::atomic<std::weak_ptr<U>>**。

+   自从 C++20 发布以来，出现了一种新的原子类型，**std::atomic_ref<T>** 。

在本章中，我们将重点关注 **std::atomic_flag** 和一些 **std::atomic** 类型。对于这里提到的其他原子类型，您可以使用之前的链接访问在线 C++ 参考。

在进一步解释这些类型之前，有一个非常重要的澄清需要做出：仅仅因为一个类型是 *原子* 的，并不能保证它是 *无锁* 的。在这里，我们所说的原子意味着不可分割的操作，而所说的无锁意味着有特殊的 CPU 原子指令支持。如果没有硬件支持某些原子操作，C++ 标准库将使用锁来实现这些操作。

要检查原子类型是否无锁，我们可以使用任何 **std::atomic<T>** 类型下的以下成员函数：

+   **bool is_lock_free() const noexcept**：如果此类型的所有原子操作都是无锁的，则返回 **true**，否则返回 **false**（除了 **std::atomic_flag**，它保证始终是无锁的）。其余的原子类型可以使用锁（如互斥量）来实现以保证操作的原子性。此外，某些原子类型可能只在某些情况下是无锁的。如果某个 CPU 只能无锁地访问对齐的内存，那么该原子类型的未对齐对象将使用锁来实现。

也有一个常量用来指示原子类型是否始终无锁：

+   **静态常量 bool is_always_lock_free = /* 实现定义 */**：如果原子类型始终是无锁的（例如，即使是未对齐的对象），则此常量的值将为 **true**

重要的是要意识到这一点：原子类型不保证是无锁的。**std::atomic<T>** 模板不是一个可以将所有原子类型转换为无锁原子类型的魔法机制。

## C++ 标准库原子操作

原子操作主要有两种类型：

+   **原子类型的成员函数**：例如，**std::atomic<int>** 有一个 **load()** 成员函数用于原子地读取其值

+   **自由函数**：**const std::atomic_load(const std::atomic<T>* obj)** 函数与之前的函数完全相同

您可以访问以下代码（如果您感兴趣，还可以访问生成的汇编代码）在 [`godbolt.org/z/Yhdr3Y1Y8`](https://godbolt.org/z/Yhdr3Y1Y8) 。此代码展示了成员函数和自由函数的使用：

```cpp
#include <atomic>
#include <iostream>
std::atomic<int> counter {0};
int main() {
    // Using member functions
    int count = counter.load();
    std::cout << count << std::endl;
    count++;
    counter.store(count);
    // Using free functions
    count = std::atomic_load(&counter);
    std::cout << count << std::endl;
    count++;
    std::atomic_store(&counter, count);
    return 0;
}
```

大多数原子操作函数都有一个参数来指示内存顺序。我们已经在关于 C++ 内存模型的章节中解释了内存顺序是什么，以及 C++ 提供了哪些内存排序类型。

## 示例 - 使用 C++ 原子标志实现的简单自旋锁

**std::atomic_flag** 原子类型是最基本的标准原子类型。它只有两种状态：设置和未设置（我们也可以称之为 true 和 false）。它总是无锁的，与任何其他标准原子类型形成对比。因为它如此简单，所以主要用作构建块。

这是原子标志示例的代码：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
class spin_lock {
public:
    spin_lock() = default;
    spin_lock(const spin_lock &) = delete;
    spin_lock &operator=(const spin_lock &) = delete;
    void lock() {
        while  (flag.test_and_set(std::memory_order_acquire)) {
        }
    }
    void unlock() {
        flag.clear(std::memory_order_release);
    }
private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};
```

在使用之前，我们需要初始化 **std::atomic_flag**。以下代码展示了如何进行初始化：

```cpp
std::atomic_flag flag = ATOMIC_FLAG_INIT;
```

这是初始化 **std::atomic_flag** 为确定值的唯一方法。**ATOMIC_FLAG_INIT** 的值是实现定义的。

一旦标志被初始化，我们就可以对其执行两个原子操作：

+   **clear**：这个操作原子地将标志设置为 **false**

+   **test_and_set**：这个操作原子地将标志设置为 **true** 并获取其前一个值

**clear** 函数只能使用松散、释放或顺序一致性内存顺序调用。**test_and_set** 函数只能使用松散、获取或顺序一致性调用。使用任何其他内存顺序将导致未定义行为。

现在让我们看看如何使用 **std::atomic_flag** 实现一个简单的自旋锁。首先，我们知道操作是原子的，所以线程要么清除标志，要么不清除，如果一个线程清除了标志，它就会被完全清除。线程不可能只 *半清除* 标志（记住，对于某些非原子标志这是可能的）。**test_and_set** 函数也是原子的，所以标志被设置为 **true**，并且我们一次性获得其前一个状态。

要实现基本的自旋锁，我们需要一个原子标志来原子地处理锁状态，以及两个函数：**lock()** 用于获取锁（就像我们为互斥量所做的那样）和 **unlock()** 用于释放锁。

### 简单自旋锁 unlock() 函数

我们将从 **unlock()** 开始，这是最简单的函数。它只会重置标志（通过将其设置为 false）而不再做其他操作：

```cpp
void unlock()
{
    flag.clear(std::memory_order_release);
}
```

代码很简单。如果我们省略了 **std::memory_order_seq_cst** 参数，将应用最严格的内存顺序选项，即顺序一致性。

### 简单的自旋锁 lock() 函数

锁函数有更多步骤。首先，让我们解释一下它做什么：**lock()** 必须检查原子标志是否开启。如果它是关闭的，那么就将其开启并完成。如果标志是开启的，那么就持续检查，直到另一个线程将其关闭。我们将使用 **test_and_set()** 使这个函数工作：

```cpp
void lock()
{
    while (flag.test_and_set(std::memory_order_acquire)) {}
}
```

上述代码的工作方式如下：在一个 **while** 循环中，**test_and_set** 将标志设置为 **true** 并返回前一个值。如果标志已经设置，再次设置它不会改变任何东西，函数返回 **true**，所以循环会持续设置标志。当最终 **test_and_set** 返回 **false** 时，这意味着标志已被清除，我们可以退出循环。

### 简单自旋锁问题

简单的自旋锁实现已包含在本章中，以介绍原子类型（**std::atomic_flag**，最简单的标准原子类型）和操作（**clear**和**test_and_set**）的使用，但它存在一些严重问题：

+   其中第一个问题是其性能不佳。仓库中的代码将让您进行实验。预期自旋锁的性能将远低于互斥锁。

+   线程一直在自旋等待标志被清除。这种忙等待是应该避免的，尤其是在存在线程竞争的情况下。

您可以尝试运行此示例的前述代码。当我们运行它时，我们得到了这些结果，如*表 5.1*所示。每个线程将计数器加 1 2 亿次。

|  | **std::mutex** | **自旋锁** | **原子计数器** |
| --- | --- | --- | --- |
| 一个线程 | 1.03 s | 1.33 s | 0.82 s |
| 两个线程 | 10.15 s | 39.14 s | 4.52 s |
| 四个线程 | 24.61 s | 128.84 s | 9.13 s |

表 5.1：同步原语分析结果

从上述表中，我们可以看到简单的自旋锁工作得有多差，以及它如何随着线程的增加而恶化。请注意，这个简单的示例只是为了学习，简单的**std::mutex**自旋锁和原子计数器都可以得到改进，以便原子类型表现更好。

在本节中，我们探讨了 C++标准库提供的最基本原子类型**std::atomic_flag**。有关此类型和 C++20 中添加的新功能的信息，请参阅在线 C++参考，可在[`en.cppreference.com/w/cpp/atomic/atomic_flag`](https://en.cppreference.com/w/cpp/atomic/atomic_flag)找到。

在下一节中，我们将探讨如何创建一种简单的方法，让线程告诉主线程它已处理了多少个项目。

## 示例 - 线程进度报告

有时我们想检查线程的进度或在其完成时收到通知。这可以通过不同的方式完成，例如，使用互斥锁和条件变量，或者使用由互斥锁同步的共享变量，正如我们在*第四章*中看到的。我们还在本章中看到了如何使用原子操作同步计数器。在以下示例中，我们将使用类似的计数器：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
constexpr int NUM_ITEMS{100000};
int main() {
    std::atomic<int> progress{0};
    std::thread worker([&progress] {
        for (int i = 1; i <= NUM_ITEMS; ++i) {
            progress.store(i, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    while (true) {
        int processed_items = progress.load(std::memory_order_relaxed);
        std::cout << "Progress: "
                  << processed_items << " / " << NUM_ITEMS
                  << std::endl;
        if (processed_items == NUM_ITEMS) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    worker.join();
    return 0;
}
```

上述代码实现了一个线程（**工作线程**），它处理一定数量的项目（在这里，处理是通过使线程休眠来模拟的）。每当线程处理一个项目时，它都会增加**进度**变量。主线程执行一个**while**循环，并在每次迭代中访问**进度**变量并写入进度报告（处理的项目数量）。一旦所有项目都处理完毕，循环结束。

在本例中，我们使用了**std::atomic<int>**原子类型（一个原子整数）和两个原子操作：

+   **load()** : 该原子操作检索**进度**变量的值

+   **store()**：这个原子操作修改 **progress** 变量的值

处理 **progress** 的 **worker** 线程以原子方式读取和写入，因此当两个线程访问 **progress** 变量时不会发生竞争条件。

**load()** 和 **store()** 原子操作有一个额外的参数来指示内存顺序。在这个例子中，我们使用了 **std::memory_order_relaxed**。这是一个使用松散内存顺序的典型例子：一个线程增加一个计数器，另一个线程读取它。我们需要的唯一顺序是读取递增的值，而对于这一点，松散内存顺序就足够了。

在介绍了 **load()** 和 **store()** 原子操作用于原子地读写变量之后，让我们看看另一个简单的统计收集应用的例子。

## 示例 - 简单统计

这个例子与上一个例子有相同的思想：一个线程可以使用原子操作将进度（例如，处理的项目数量）传递给另一个线程。在这个新的例子中，一个线程将生成一些数据，另一个线程将读取这些数据。我们需要同步内存访问，因为我们有两个线程共享相同的内存，并且至少有一个线程正在更改内存。与上一个例子一样，我们将使用原子操作来实现这一点。

以下代码声明了我们将要使用的原子变量，用于收集统计信息——一个用于处理的项目数量，另外两个（分别用于总处理时间和每个项目的平均处理时间）：

```cpp
std::atomic<int> processed_items{0};
std::atomic<float> total_time{0.0f};
std::atomic<double> average_time{0.0};
```

我们使用原子浮点数和双精度浮点数来表示总时间和平均时间。在完整的示例代码中，我们确保这两种类型都是无锁的，这意味着它们使用 CPU 的原子指令（所有现代 CPU 都应该有这些）。

现在我们来看看工作线程如何使用这些变量：

```cpp
processed_items.fetch_add(1, std::memory_order_relaxed);
total_time.fetch_add(elapsed_s, std::memory_order_relaxed);
average_time.store(total_time.load() / processed_items.load(), std::memory_order_relaxed);
```

第一行以原子方式将处理的项目数增加 1。**fetch_add** 函数将 **1** 添加到变量值，并返回旧值（我们在这个例子中没有使用它）。

第二行将 **elapsed_s**（处理一个项目所需的时间，以秒为单位）加到 **total_time** 变量上，我们使用这个变量来跟踪处理所有项目所需的时间。

然后，第三行通过原子地读取 **total_time** 和 **processed_items** 并将结果原子地写入 **average_time** 来计算每个项目的平均时间。或者，我们也可以使用 **fetch_add()** 调用的值来计算平均时间，但它们不包括最后处理的项目。我们也可以在主线程中计算 **average_time**，但在这里我们选择在工作线程中这样做，仅作为一个示例并练习使用原子操作。记住，我们的目标（至少在本章中）并不是速度，而是学习如何使用原子操作。

下面的代码是统计示例的完整代码：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
constexpr int NUM_ITEMS{10000};
void process() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 20);
    int sleep_duration = dis(gen);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
}
int main() {
    std::atomic<int> processed_items{0};
    std::atomic<float> total_time{0.0f};
    std::atomic<double> average_time{0.0};
    std::thread worker([&] {
        for (int i = 1; i <= NUM_ITEMS; ++i) {
            auto now = std::chrono::high_resolution_clock::now();
            process();
            auto elapsed = 
                std::chrono::high_resolution_clock::now() - now;
            float elapsed_s =
                std::chrono::duration<float>(elapsed).count();
            processed_items.fetch_add(1, std::memory_order_relaxed);
            total_time.fetch_add(elapsed_s, std::memory_order_relaxed);
            average_time.store(total_time.load() / processed_items.load(), std::memory_order_relaxed);
        }
    });
    while (true) {
        int items = processed_items.load(std::memory_order_relaxed);
        std::cout << "Progress: " << items << " / " << NUM_ITEMS << std::endl;
        float time = total_time.load(std::memory_order_relaxed);
        std::cout << "Total time: " << time << " sec" << std::endl;
        double average = average_time.load(std::memory_order_relaxed);
        std::cout << "Average time: " << average * 1000 << " ms" << std::endl;
        if (items == NUM_ITEMS) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    worker.join();
    return 0;
}
```

让我们总结一下在本节中到目前为止我们所看到的内容：

+   C++标准原子类型：我们使用 **std::atomic_flag** 实现了一个简单的自旋锁，并且我们已经使用了一些 **std::atomic<T>** 类型来实现线程间简单数据的通信。我们看到的所有原子类型都是无锁的。

+   **load()** 原子操作用于原子地读取原子变量的值。

+   **store()** 原子操作用于原子地将新值写入原子变量。

+   **clear()** 和 **test_and_set()** 是由 **std::atomic_flag** 提供的特殊原子操作。

+   **fetch_add()**，用于原子地将某个值添加到原子变量中并获取其之前的值。整数和浮点类型还实现了**fetch_sub()**，用于从原子变量中减去一定值并返回其之前的值。一些用于执行位逻辑操作的函数仅针对整数类型实现：**fetch_and()**，**fetch_or()**，和**fetch_xor()**。

以下表格总结了原子类型和操作。对于详尽的描述，请参考在线 C++参考：[`en.cppreference.com/w/cpp/atomic/atomic`](https://en.cppreference.com/w/cpp/atomic/atomic)

表格显示了三种新的操作：**exchange**，**compare_exchange_weak**，和**compare_exchange_strong**。我们将在稍后的示例中解释它们。大多数操作（即函数，而不是运算符）都有一个用于内存顺序的另一个参数。

| **操作** | **atomic_flag** | **atomic<bool>** | **atomic<integral>** | **atomic<floating-point>** | **atomic<other>** |
| --- | --- | --- | --- | --- | --- |
| **test_and_set** | YES |  |  |  |  |
| **Clear** | YES |  |  |  |  |
| **Load** |  | YES | YES | YES | YES |
| **Store** |  | YES | YES | YES | YES |
| **fetch_add, +=** |  |  | YES | YES |  |
| **fetch_sub, -=** |  |  | YES | YES |  |
| **fetch_and, &=** |  |  | YES |  |  |
| **fetch_or, | =** |  |  | YES |  |  |
| **fetch_xor, ^=** |  |  | YES |  |  |
| **++, --** |  |  | YES |  |  |
| **Exchange** |  | YES | YES | YES | YES |
| **compare_exchange_weak, compare_exchange_strong** |  | YES | YES | YES | YES |

表 5.2：原子类型和操作

让我们回顾一下 **is_lock_free()** 函数和 **is_always_lock_free** 常量。我们看到了如果 **is_lock_free()** 为真，则原子类型具有具有特殊 CPU 指令的无锁操作。原子类型可能只在某些时候是无锁的，因此 **is_always_lock_free** 常量告诉我们类型是否始终无锁。到目前为止，我们看到的所有类型都是无锁的。让我们看看当原子类型非无锁时会发生什么。

以下展示了非无锁原子类型的代码：

```cpp
#include <atomic>
#include <iostream>
struct no_lock_free {
    int a[128];
    no_lock_free() {
        for (int i = 0; i < 128; ++i) {
            a[i] = i;
        }
    }
};
int main() {
    std::atomic<no_lock_free> s;
    std::cout << "Size of no_lock_free: " << sizeof(no_lock_free) << " bytes\n";
    std::cout << "Size of std::atomic<no_lock_free>: " << sizeof(s) << " bytes\n";
    std::cout << "Is std::atomic<no_lock_free> always lock-free: " << std::boolalpha
              << std::atomic<no_lock_free>::is_always_lock_free << std::endl;
    std::cout << "Is std::atomic<no_lock_free> lock-free: " << std::boolalpha << s.is_lock_free() << std::endl;
    no_lock_free s1;
    s.store(s1);
    return 0;
}
```

当你执行代码时，你会注意到**std::atomic<no_lock_free>**类型不是无锁的。它的大小，512 字节，是导致这种情况的原因。当我们向原子变量赋值时，该值是*原子地*写入的，但这个操作没有使用 CPU 原子指令，也就是说它不是无锁的。这个操作的实现取决于编译器，但一般来说，它使用互斥锁或特殊的自旋锁（例如 Microsoft Visual C++）。

这里的教训是，所有原子类型都有原子操作，但它们并不都是无锁的。如果一个原子类型不是无锁的，那么最好使用锁来实现它。

我们了解到一些原子类型不是无锁的。现在我们将看看另一个例子，展示我们尚未覆盖的原子操作：**exchange**和**compare_exchange**操作。

## 示例 – 延迟一次性初始化

有时初始化一个对象可能会很昂贵。例如，一个特定的对象可能需要连接到数据库或服务器，建立这种连接可能需要很长时间。在这些情况下，我们应该在对象使用之前而不是在程序中定义它时初始化对象。这被称为**延迟初始化**。现在假设多个线程需要首次使用该对象。如果有多个线程初始化对象，那么将创建不同的连接，这是错误的，因为对象只打开和关闭一个连接。因此，必须避免多次初始化。为了确保对象只初始化一次，我们将利用一种称为延迟一次性初始化的方法。

下面的代码展示了延迟一次性初始化的示例：

```cpp
#include <atomic>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
constexpr int NUM_THREADS{8};
void process() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000000);
    int sleep_duration = dis(gen);
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
}
int main() {
    std::atomic<int> init_thread{0};
    auto worker = &init_thread {
        process();
        int init_value = init_thread.load(std::memory_order::seq_cst);
        if (init_value == 0) {
            int expected = 0;
            if (init_thread.compare_exchange_strong(expected, i, std::memory_order::seq_cst)) {
                std::cout << "Previous value of init_thread: " << expected << "\n";
                std::cout << "Thread " << i << " initialized\n";
            } else {
                // init_thread was already initialized
            }
        } else {
            // init_thread was already initialized
        }
    };
    std::vector<std::thread> threads;
    for (int i = 1; i <= NUM_THREADS; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto &t: threads) {
        t.join();
    }
    std::cout << "Thread: " << init_thread.load() << " initialized\n";
    return 0;
}
```

在本章前面我们看到的原子类型操作表中，有一些操作我们还没有讨论。现在我们将通过一个例子来解释**compare_exchange_strong**。在例子中，我们有一个初始值为 0 的变量。有多个线程正在运行，每个线程都有一个唯一的整数 ID（1、2、3 等等）。我们希望将变量的值设置为第一个设置它的线程的 ID，并且只初始化变量一次。在*第四章*中，我们学习了**std::once_flag**和**std::call_once**，我们可以使用它们来实现这种一次性初始化，但本章是关于原子类型和操作的，所以我们将使用这些来实现我们的目标。

为了确保**init_thread**变量的初始化只进行一次，并且避免由于多个线程的写访问导致的竞态条件，我们使用了一个原子的**int**。第**[1**]行原子地读取了**init_thread**的内容。如果值不是 0，那么这意味着它已经被初始化，并且工作线程不再做其他操作。

**init_thread** 的当前值存储在 **expected** 变量中，它代表当我们尝试初始化它时，我们期望 **init_thread** 将具有的值。现在行 **[2]** 执行以下步骤：

1.  将 **init_thread** 的当前值与 **expected** 值（再次强调，等于 0）进行比较。

1.  如果比较不成功，将 **init_thread** 的当前值复制到 **expected** 中，然后返回 **false**。

1.  如果比较成功，将 **init_thread** 的当前值复制到 **expected** 中，然后将 **init_thread** 的当前值设置为 **i** 并返回 **true**。

只有当 **compare_exchange_strong** 返回 **true** 时，当前线程才会初始化 **init_thread**。此外，请注意，我们需要再次执行比较（即使行 **[1]** 返回 0 作为 **init_thread** 的当前值）因为有可能另一个线程已经初始化了该变量。

非常重要的是要注意，如果 **compare_exchange_strong** 返回 **false**，则比较失败；如果它返回 **true**，则比较成功。这对于 **compare_exchange_strong** 总是成立的。另一方面，**compare_exchange_weak** 即使比较成功也可能失败（即返回 **false**）。使用它的原因是在某些平台上，当它在循环内部调用时，它提供了更好的性能。

关于这两个函数的更多信息，请参考在线 C++ 参考文档：[`en.cppreference.com/w/cpp/atomic/atomic/compare_exchange`](https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange)

在本节关于 C++ 标准库原子类型和操作的讨论中，我们看到了以下内容：

+   最常用的标准原子类型，例如 **std::atomic_flag** 和 **std::atomic<int>**

+   最常用的原子操作：**load()**、**store()** 和 **exchange_compare_strong()**/ **exchange_compare_weak()**

+   包含这些原子类型和操作的基本示例，包括懒加载一次性初始化和线程进度通信

我们已经多次提到，大多数原子操作（函数）允许我们选择我们想要使用的内存顺序。在下一节中，我们将实现一个无锁编程示例：一个 SPSC 无锁队列。

# SPSC 无锁队列

我们已经探讨了 C++ 标准库的原子特性，例如原子类型和操作以及内存模型和排序。现在我们将看到一个使用原子实现 SPSC 无锁队列的完整示例。

此队列的主要特性如下：

+   **SPSC**：此队列设计用于与两个线程一起工作，一个线程将元素推入队列，另一个线程从队列中获取元素。

+   **有界**：此队列具有固定大小。我们需要一种方法来检查队列何时达到其容量以及何时没有元素）。

+   **无锁**：此队列使用在现代 Intel x64 CPU 上始终无锁的原子类型。

在你开始开发队列之前，请记住，无锁不等于无等待（也要记住，无等待并不完全消除等待；它只是确保每个队列 push/pop 所需的步骤数有一个限制）。一些主要影响性能的方面将在*第十三章*中讨论。在第十三章中，我们还将优化队列的性能。现在，在本章中，我们将构建一个正确且性能良好的 SPSC 无锁队列——我们将在稍后展示如何提高其性能。

我们在*第四章*中使用了互斥锁和条件变量来创建一个 SPSC 队列，消费者和生产线程可以安全地访问。本章将使用原子操作达到相同的目标。

我们将在队列中使用相同的数据结构来存储项目：**std::vector<T>**，具有固定大小，即 2 的幂。这样，我们可以提高性能并快速找到下一个头和尾索引，而无需使用需要除法指令的模运算符。当使用无锁原子类型以获得更好的性能时，我们需要注意影响性能的每一件事。

## 为什么我们使用 2 的幂作为缓冲区大小？

我们将使用一个向量来保存队列项目。该向量将具有固定大小，比如说**N**。我们将使向量表现得像环形缓冲区，这意味着在向量中访问元素的索引将在到达末尾后循环回起点。第一个元素将跟随最后一个元素。正如我们在*第四章*中学到的，我们可以用模运算符做到这一点：

```cpp
size_t next_index = (curr_index + 1) % N;
```

如果大小是，例如，四个元素，下一个元素的索引将按照前面的代码计算。对于最后一个索引，我们有以下代码：

```cpp
next_index = (3 + 1) % 4 = 4 % 4 = 0;
```

因此，正如我们所说的，向量将是一个环形缓冲区，因为，在最后一个元素之后，我们将回到第一个，然后是第二个，依此类推。

我们可以使用这种方法为任何缓冲区大小**N**获取下一个索引。但我们为什么只使用 2 的幂的大小？答案是简单的：性能。模（**%**）运算符需要除法指令，这是昂贵的。当**N**是 2 的幂时，我们只需做以下操作：

```cpp
size_t next_index = curr_index & (N – 1);
```

这比使用模运算符要快得多。

## 缓冲区访问同步

要访问队列缓冲区，我们需要两个索引：

+   **head**：当前要读取的元素的索引

+   **tail**：下一个要写入的元素的索引

消费者线程将使用头索引进行读写。生产线程将使用尾索引进行读写。由于这个原因，我们需要同步对这些变量的访问：

+   只有一个线程（消费者）写入 **head**，这意味着它可以以松散的内存顺序读取它，因为它总是看到自己的更改。读取 **tail** 由读取器线程完成，并且它需要与生产者写入 **tail** 进行同步，因此它需要获取内存顺序。我们可以为一切使用顺序一致性，但我们希望获得最佳性能。当消费者线程写入 **head** 时，它需要与生产者读取它的操作同步，因此它需要释放内存顺序。

+   对于 **tail**，只有生产者线程写入它，因此我们可以使用松散的内存顺序来读取它，但我们需要释放内存顺序来写入它并与消费者线程的读取同步。为了与消费者线程的写入同步，我们需要获取内存顺序来读取 **head**。

队列类的成员变量如下：

```cpp
const std::size_t capacity_; // power of two buffer size
std::vector<T> buffer_; // buffer to store queue items handled like a ring buffer
std::atomic<std::size_t> head_{ 0 };
std::atomic<std::size_t> tail_{ 0 };
```

在本节中，我们看到了如何同步对队列缓冲区的访问。

## 将元素推入队列

一旦我们决定了队列的数据表示以及如何同步对其元素的访问，让我们实现将元素推入队列的函数：

```cpp
bool push(const T& item) {
    std::size_t tail =
        tail_.load(std::memory_order_relaxed);
    std::size_t next_tail =
       (tail + 1) & (capacity_ - 1);
    if (next_tail != head_.load(std::memory_order_acquire)) {
        buffer_[tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    return false;
}
```

当前尾索引，即数据项（如果可能）要推入队列的缓冲区槽位，在行 **[1]** 中原子地读取。正如我们之前提到的，这个读取可以使用 **std::memory_order_relaxed**，因为只有生产者线程更改此变量，并且它是唯一调用 push 的线程。

行 **[2]** 计算下一个索引对容量取模（记住缓冲区是一个环形）。我们需要这样做来检查队列是否已满。

我们在行 **[3]** 中执行检查。我们首先使用 **std::memory_order_acquire** 原子地读取当前头值，因为我们希望生产者线程观察到消费者线程对此变量所做的修改。然后我们将其值与下一个头索引进行比较。

如果下一个尾值等于当前头值，那么（根据我们的约定）队列已满，我们返回 **false**。

如果队列未满，行 **[4]** 将数据项复制到队列缓冲区。这里值得指出的是，数据复制不是原子的。

行 **[5]** 原子地将新的尾索引值写入 **tail_**。然后，使用 **std::memory_order_release** 使更改对使用 **std::memory_order_acquire** 原子读取此变量的消费者线程可见。

## 从队列中弹出元素

现在我们来看一下 **pop** 函数是如何实现的：

```cpp
bool pop(T& item) {
    std::size_t head =
        head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_acquire)) {
        return false;
    }
    item = buffer_[head];
    head_.store((head + 1) & (capacity_ - 1), std::memory_order_release);
    return true;
}
```

行 **[1]** 原子地读取 **head_**（下一个要读取的项目索引）的当前值。我们使用 **std::memory_order_relaxed**，因为不需要执行顺序强制，因为 **head_** 变量只由消费者线程修改，它是唯一调用 **pop** 的线程。

行 **[2]** 检查队列是否为空。如果当前 **head_** 的值与当前 **tail_** 的值相同，则队列为空，函数仅返回 **false**。我们使用 **std::memory_order_acquire** 原子地读取 **tail_** 的值，以查看生产者线程对 **tail_** 的最新更改。

行 **[3]** 将队列中的数据复制到作为 **pop** 参数传递的项目引用中。再次强调，这个复制不是原子的。

最后，行 **[4]** 更新 **head_** 的值。同样，我们使用 **std::memory_order_release** 原子地写入值，以便消费者线程可以看到消费者线程对 **head_** 的更改。

SPSC 无锁队列实现的代码如下：

```cpp
#include <atomic>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
template<typename T>
class spsc_lock_free_queue {
public:
    // capacity must be power of two to avoid using modulo operator when calculating the index
    explicit spsc_lock_free_queue(size_t capacity) : capacity_(capacity), buffer_(capacity) {
        assert((capacity & (capacity - 1)) == 0 && "capacity must be a power of 2");
    }
    spsc_lock_free_queue(const spsc_lock_free_queue &) = delete;
    spsc_lock_free_queue &operator=(const spsc_lock_free_queue &) = delete;
    bool push(const T &item) {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        std::size_t next_tail = (tail + 1) & (capacity_ - 1);
        if (next_tail != head_.load(std::memory_order_acquire)) {
            buffer_[tail] = item;
            tail_.store(next_tail, std::memory_order_release);
            return true;
        }
        return false;
    }
    bool pop(T &item) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        item = buffer_[head];
        head_.store((head + 1) & (capacity_ - 1), std::memory_order_release);
        return true;
    }
private:
    const std::size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};
```

完整示例的代码可以在以下书籍仓库中找到：[`github.com/PacktPublishing/Asynchronous-Programming-in-CPP/blob/main/Chapter_05/5x09-SPSC_lock_free_queue.cpp`](https://github.com/PacktPublishing/Asynchronous-Programming-in-CPP/blob/main/Chapter_05/5x09-SPSC_lock_free_queue.cpp)

在本节中，我们将 SPSC 无锁队列作为原子类型和操作的示例实现。在第 *第十三章*中，我们将重新审视这个实现并提高其性能。

# 摘要

本章介绍了原子类型和操作、C++ 内存模型以及 SPSC 无锁队列的基本实现。

以下是我们所查看内容的摘要：

+   C++ 标准库原子类型和操作，它们是什么，以及如何使用一些示例。

+   C++ 内存模型，特别是它定义的不同内存排序。请记住，这是一个非常复杂的话题，本节只是对其进行了基本介绍。

+   如何实现一个基本的 SPSC 无锁队列。正如我们之前提到的，我们将在*第十三章*中展示如何提高其性能。性能提升的措施包括消除虚假共享（当两个变量位于同一缓存行中，并且每个变量仅被一个线程修改时发生的情况）和减少真实共享。如果你现在不理解这些内容，请不要担心。我们将在稍后进行讲解，并演示如何运行性能测试。

这是对原子操作的基本介绍，用于同步不同线程之间的内存访问。在某些情况下，使用原子操作相当简单，类似于收集统计数据和简单的计数器。更复杂的应用，如 SPSC 无锁队列的实现，需要更深入地了解原子操作。本章我们所看到的内容有助于理解基础知识，并为进一步研究这个复杂主题打下基础。

在下一章中，我们将探讨承诺和未来，这是 C++ 异步编程的两个基本构建块。

# 进一步阅读

+   [Butenhof, 1997] David R. Butenhof，使用 POSIX 线程进行编程，Addison Wesley，1997。

+   [Williams, 2019] Anthony Williams，C++并发实战，第 2 版，Manning，2019。

+   内存模型：控制你的共享数据，Jana Machutová，[`www.youtube.com/watch?v=L5RCGDAan2Y`](https://www.youtube.com/watch?v=L5RCGDAan2Y) .

+   *C++原子操作：从基础到高级*，Fedor Pikus，[`www.youtube.com/watch?v=ZQFzMfHIxng`](https://www.youtube.com/watch?v=ZQFzMfHIxng) .

+   *Intel 64 和 IA-32 架构软件开发者手册，第 3A 卷：系统编程指南，第一部分*，英特尔公司，[`www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf`](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf) .

# 第三部分：使用承诺（Promises）、未来（Futures）和协程进行异步编程

在这部分，我们将焦点转向本书的核心主题，即异步编程，这是构建响应式、高性能应用程序的关键方面。我们将学习如何通过使用诸如承诺（promises）、未来（futures）、打包任务（packaged tasks）、`std::async`函数和协程（coroutines）等工具来并发执行任务，而不会阻塞主执行流程。协程是一种革命性的特性，它允许在不创建线程的开销下进行异步编程。我们还将介绍高级技术，用于共享未来（futures），并探讨这些概念在现实世界场景中的必要性。这些强大的机制使我们能够开发出适用于现代软件系统的有效、可扩展和可维护的异步软件。

本部分包含以下章节：

+   *第六章* ，*承诺（Promises）和未来（Futures）*

+   *第七章* ，*异步函数*

+   *第八章* ，*使用协程进行异步编程*
