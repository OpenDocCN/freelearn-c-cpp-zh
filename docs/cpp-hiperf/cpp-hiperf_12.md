# 12

# 协程和惰性生成器

计算已经成为一个等待的世界，我们需要编程语言的支持来表达*等待*。一般的想法是在当前流到达我们知道可能需要等待某些东西的点时，暂停（暂时暂停）当前流，并将执行交给其他流。我们需要等待的*某些东西*可能是网络请求、用户的点击、数据库操作，甚至是花费太长时间的内存访问。相反，我们在代码中说我们会等待，继续一些其他流，然后在准备好时回来。协程允许我们这样做。

在本章中，我们主要将关注添加到 C++20 中的协程。您将学习它们是什么，如何使用它们以及它们的性能特征。但我们也将花一些时间来更广泛地看待协程，因为这个概念在许多其他语言中都是明显的。

C++协程在标准库中的支持非常有限。为 C++23 发布添加协程的标准库支持是一个高优先级的功能。为了在日常代码中有效地使用协程，我们需要实现一些通用的抽象。本书将向您展示如何实现这些抽象，以便学习 C++协程，而不是为您提供生产就绪的代码。

了解存在的各种类型的协程，协程可以用于什么，以及是什么促使 C++添加新的语言特性来支持协程。

本章涵盖了很多内容。下一章也是关于协程，但重点是异步应用程序。总之，本章将引导您完成：

+   关于协程的一般理论，包括有栈和无栈协程之间的区别，以及编译器如何转换它们并在计算机上执行。

+   介绍 C++中无栈协程。将讨论和演示 C++20 对协程的新语言支持，包括`co_await`、`co_yield`和`co_return`。

+   使用 C++20 协程作为生成器所需的抽象。

+   一些真实世界的例子展示了使用协程的可读性和简单性的好处，以及我们如何通过使用协程编写可组合的组件，以便进行惰性评估。

如果您已经在其他语言中使用协程，那么在阅读本章的其余部分之前，您需要做好两件事：

+   对您来说，一些内容可能感觉很基础。尽管 C++协程的工作原理的细节远非微不足道，但使用示例可能对您来说感觉微不足道。

+   本章中我们将使用的一些术语（协程、生成器、任务等）可能与您当前对这些内容的看法不一致。

另一方面，如果您对协程完全不熟悉，本章的部分内容可能看起来像魔术一样，需要一些时间来理解。因此，我将首先向您展示一些使用协程时 C++代码的例子。

# 一些激励性的例子

协程是一种类似于 lambda 表达式的功能，它提供了一种完全改变我们编写和思考 C++代码的方式。这个概念非常普遍，可以以许多不同的方式应用。为了让您了解使用协程时 C++的样子，我们将在这里简要地看两个例子。

使用 yield 表达式来实现生成器——产生值序列的对象。在这个例子中，我们将使用关键字`co_yield`和`co_return`来控制流程：

```cpp
auto iota(int start) -> Generator<int> {
  for (int i = start; i < std::numeric_limits<int>::max(); ++i) {
    co_yield i;
  }
}
auto take_until(Generator<int>& gen, int value) -> Generator<int> {
  for (auto v : gen) {
    if (v == value) {
      co_return;
    }
    co_yield v;
  }
}
int main() {
  auto i = iota(2);
  auto t = take_until(i, 5);
  for (auto v : t) {          // Pull values
    std::cout << v << ", ";
  }
  return 0;
}
// Prints: 2, 3, 4 
```

在前面的示例中，`iota()`和`take_until()`是协程。`iota()`生成一个整数序列，`take_until()`在找到指定值之前产生值。`Generator`模板是一种自定义类型，我将在本章后面向您展示如何设计和实现它。

构建生成器是协程的一个常见用例，另一个是实现异步任务。下一个示例将演示我们如何使用操作符`co_await`来等待某些内容，而不会阻塞当前执行的线程：

```cpp
auto tcp_echo_server() -> Task<> {
  char data[1024];
  for (;;) {
    size_t n = co_await async_read(socket, buffer(data));
    co_await async_write(socket, buffer(data, n));
  }
} 
```

`co_await`不会阻塞，而是在异步读写函数完成并恢复执行之前暂停执行。这里介绍的示例是不完整的，因为我们不知道`Task`、`socket`、`buffer`和异步 I/O 函数是什么。但是在下一章中，当我们专注于异步任务时，我们会了解到这些内容。

如果目前还不清楚这些示例是如何工作的，不要担心——我们将在本章后面花费大量时间深入了解细节。这些示例是为了给你一个关于协程允许我们做什么的提示，如果你以前从未遇到过它们。

在深入研究 C++20 协程之前，我们需要讨论一些术语和共同的基础知识，以更好地理解为什么在 2020 年向 C++中添加一个相当复杂的语言特性的设计和动机。

# 协程抽象

现在我们将退后一步，谈论一般的协程，而不仅仅是专注于添加到 C++20 的协程。这将让你更好地理解为什么协程是有用的，以及有哪些类型的协程以及它们之间的区别。如果你已经熟悉了有栈和无栈协程以及它们是如何执行的，你可以跳过这一部分，直接转到下一部分，*C++中的协程*。

协程抽象已经存在了 60 多年，许多语言已经将某种形式的协程纳入其语法或标准库中。这意味着协程在不同的语言和环境中可能表示不同的东西。由于这是一本关于 C++的书，我将使用 C++标准中使用的术语。

协程与子例程非常相似。在 C++中，我们没有明确称为子例程的东西；相反，我们编写函数（例如自由函数或成员函数）来创建子例程。我将交替使用术语**普通函数**和**子例程**。

## 子例程和协程

为了理解协程和子例程（普通函数）之间的区别，我们将在这里专注于子例程和协程的最基本属性，即如何启动、停止、暂停和恢复它们。当程序的其他部分调用子例程时，子例程就会启动。当子例程返回到调用者时，子例程就会停止：

```cpp
auto subroutine() {
  // Sequence of statements ...

  return;     // Stop and return control to caller
}
subroutine(); // Call subroutine to start it
// subroutine has finished 
```

子例程的调用链是严格嵌套的。在接下来的图表中，子例程`f()`在子例程`g()`返回之前无法返回到`main()`：

![](img/B15619_12_01.png)

图 12.1：子例程调用和返回的链

协程也可以像子例程一样启动和停止，但它们也可以被**挂起**（暂停）和**恢复**。如果你以前没有使用过协程，这一点可能一开始看起来很奇怪。协程被挂起和恢复的地方称为**挂起/恢复点**。有些挂起点是隐式的，而其他挂起点则以某种方式在代码中明确标记。以下伪代码显示了使用`await`和`yield`标记的三个显式挂起/恢复点：

```cpp
// Pseudo code
auto coroutine() {
  value = 10;  
  await something;        // Suspend/Resume point
  // ...
  yield value++;          // Suspend/Resume point
  yield value++;          // Suspend/Resume point
  // ...
  return;
}
auto res = coroutine();    // Call
res.resume();              // Resume 
```

在 C++中，使用关键字`co_await`和`co_yield`标记显式的挂起点。下面的图表显示了协程如何从一个子例程中调用，然后稍后从代码的不同部分恢复：

![](img/B15619_12_02.png)

图 12.2：协程的调用可以挂起和恢复。协程调用在被挂起时保持其内部状态。

协程被挂起时，协程内部的局部变量状态会被保留。这些状态属于协程的某次调用。也就是说，它们不像静态局部变量那样，静态局部变量在函数的所有调用之间是全局共享的。

总之，协程是可以被挂起和恢复的子例程。另一种看待它的方式是说，子例程是无法被挂起或恢复的协程的一种特例。

从现在开始，我将在区分*调用*和*恢复*，以及*挂起*和*返回*时非常严格。它们意味着完全不同的事情。调用协程会创建一个可以被挂起和恢复的协程的新实例。从协程返回会销毁协程实例，它将无法再恢复。

要真正理解协程如何帮助我们编写高效的程序，您需要了解一些关于 C++函数通常如何转换为机器代码然后执行的低级细节。

## 在 CPU 上执行子例程和协程

在本书中，我们已经讨论了内存层次结构、缓存、虚拟内存、线程调度和其他硬件和操作系统概念。但我们并没有真正讨论指令是如何使用 CPU 寄存器和堆栈在 CPU 上执行的。当比较子例程与各种协程时，了解这些概念是很重要的。

### CPU 寄存器、指令和堆栈

本节将提供一个非常简化的 CPU 模型，以便理解上下文切换、函数调用以及关于调用堆栈的更多细节。在这种情况下，当我提到 CPU 时，我指的是一些类似于带有多个通用寄存器的 x86 系列 CPU 的 CPU。

程序包含 CPU 执行的一系列指令。指令序列存储在计算机的某个地方的内存中。CPU 通过一个称为**程序计数器**的寄存器跟踪当前执行指令的地址。这样，CPU 就知道下一个要执行的指令是什么。

CPU 包含固定数量的寄存器。寄存器类似于具有预定义名称的变量，可以存储值或内存地址。寄存器是计算机上最快的数据存储器，并且最接近 CPU。当 CPU 操作数据时，它使用寄存器。一些寄存器对 CPU 具有特殊意义，而其他寄存器可以由当前执行的程序更自由地使用。

对 CPU 具有特殊意义的两个非常重要的寄存器是：

+   **程序计数器**（**PC**）：存储当前执行指令的内存地址的寄存器。每当执行一条指令时，该值会自动递增。有时它也被称为*指令指针*。

+   **堆栈指针**（**SP**）：它存储当前使用的调用堆栈顶部的地址。分配和释放堆栈内存只是改变这个单个寄存器中存储的值的问题。

![](img/B15619_12_03.png)

图 12.3：带有寄存器的 CPU

假设寄存器被称为**R0**、**R1**、**R2**和**R3**，如前图所示。典型的算术指令可能如下所示：

```cpp
add 73, R1   // Add 73 to the value stored in R1 
```

数据也可以在寄存器和内存之间复制：

```cpp
mov SP, R2   // Copy the stack pointer address to R2
mov R2, [R1] // Copy value of R2 to memory address stored in R1 
```

一组指令隐含地指向调用堆栈。CPU 通过堆栈指针知道调用堆栈的顶部在哪里。在堆栈上分配内存只是更新堆栈指针的问题。该值增加或减少取决于堆栈是向更高地址还是更低地址增长。

以下指令使用了堆栈：

```cpp
push R1     // Push value of R1 to the top of the stack 
```

push 指令将寄存器中的值复制到由堆栈指针指向的内存位置，并递增（或递减）堆栈指针。

我们还可以使用`pop`指令从堆栈中弹出值，并读取和更新堆栈指针：

```cpp
pop R2      // Pop value from the stack into R2 
```

每当执行一条指令时，CPU 会自动递增程序计数器。但程序计数器也可以通过指令明确更新，例如`jump`指令：

```cpp
jump R3     // Set the program counter to the address in R3 
```

CPU 可以以两种模式运行：用户模式或内核模式。当 CPU 在用户模式下运行时，它以不同的方式使用 CPU 寄存器。当 CPU 在用户模式下执行时，它以无法访问硬件的受限权限运行。操作系统提供在内核模式下运行的系统调用。因此，C++库函数（例如`std::puts()`）必须进行系统调用才能完成其任务，迫使 CPU 在用户模式和内核模式之间切换。

在用户模式和内核模式之间转换是昂贵的。要理解原因，让我们再次考虑我们的示意 CPU。CPU 通过使用其寄存器高效运行，因此避免不必要地将值溢出到堆栈上。但是 CPU 是所有用户进程和操作系统之间共享的资源，每当我们需要在任务之间切换时（例如，进入内核模式时），处理器的状态，包括其所有寄存器，都需要保存在内存中，以便以后可以恢复。

### 调用和返回

现在您已经基本了解了 CPU 如何使用寄存器和堆栈，我们可以讨论子例程调用。在调用和返回子例程时涉及许多机制，我们可能会认为这是理所当然的。当编译器将 C++函数转换为高度优化的机器代码时，它们的工作非常出色。

以下列表显示了调用、执行和从子例程返回时需要考虑的方面：

+   调用和返回（在代码中跳转）。

+   传递参数——参数可以通过寄存器或堆栈传递，也可以两者兼而有之。

+   在堆栈上为局部变量分配存储空间。

+   返回值——从子例程返回的值需要存储在调用者可以找到的地方。通常，这是一个专用的 CPU 寄存器。

+   在不干扰其他函数的情况下使用寄存器——子例程使用的寄存器需要在调用子例程之前恢复到其调用之前的状态。

有关如何执行函数调用的确切细节由称为**调用约定**的东西指定。它们为调用者/被调用者提供了一个协议，以便双方就谁负责哪些部分达成一致。调用约定在 CPU 架构和编译器之间不同，并且是构成**应用程序二进制接口**（**ABI**）的主要部分之一。

当调用函数时，该函数的**调用帧**（或激活帧）被创建。调用帧包含：

+   传递给函数的*参数*。

+   函数的*局部变量*。

+   我们打算使用的寄存器的*快照*，因此需要在返回之前恢复。

+   *返回地址*，它链接回调用者从中调用函数的内存位置。

+   可选的*帧指针*，指向调用者的调用帧顶部。在检查堆栈时，帧指针对调试器很有用。我们在本书中不会进一步讨论帧指针。

由于子例程的严格嵌套性质，我们可以将子例程的调用帧有效地保存在堆栈上，以支持嵌套调用。存储在堆栈上的调用帧通常称为**堆栈帧**。

以下图表显示了调用堆栈上的多个调用帧，并突出显示了单个调用帧的内容：

![](img/B15619_12_04.png)

图 12.4：具有多个调用帧的调用堆栈。右侧的调用帧是单个调用帧的放大版本。

当子程序返回给调用者时，它使用返回地址来知道要跳转到哪里，恢复它已经改变的寄存器，并弹出（释放）整个调用帧。通过这种方式，堆栈和寄存器都恢复到调用子程序被调用之前的状态。但是，有两个例外。首先，程序计数器（PC）已经移动到调用后的指令。其次，将值返回给其调用者的子程序通常将该值存储在一个专用寄存器中，调用者知道在哪里找到它。

理解了子程序是如何通过临时使用堆栈来执行，然后在将控制返回给调用者之前恢复 CPU 寄存器，我们现在可以开始看看如何挂起和恢复协程。

### 挂起和恢复

考虑以下伪代码，定义了一个具有多个挂起/恢复点的协程：

```cpp
// Pseudo code
auto coroutine() { 
  auto x = 0;
  yield x++;       // Suspend
  g();             // Call some other function
  yield x++;       // Suspend
  return;          // Return 
}
auto co = coroutine(); // Call subroutine to start it
// ...                 // Coroutine is suspended
auto a = resume(co);   // Resume coroutine to get
auto b = resume(co);   // next value 
```

当`coroutine()`挂起时，我们无法像子程序返回给调用者时那样删除调用帧。为什么？因为我们需要保留变量`x`的当前值，并且还需要记住在协程中应该在*何处*继续执行下次协程恢复时。这些信息被放入一个称为**协程帧**的东西中。协程帧包含恢复暂停协程所需的所有信息。然而，这引发了一些新问题：

+   协程帧存储在哪里？

+   协程帧有多大？

+   当协程调用子程序时，它需要一个堆栈来管理嵌套的调用帧。如果我们尝试从嵌套的调用帧内恢复会发生什么？那么当协程恢复时，我们需要恢复整个堆栈。

+   调用和从协程返回的运行时开销是多少？

+   挂起和恢复协程的运行时开销是多少？

对这些问题的简短回答是，这取决于我们讨论的协程类型：无堆栈或有堆栈的协程。

有堆栈的协程有一个单独的侧堆栈（类似于线程），其中包含协程帧和嵌套的调用帧。这使得可以从嵌套的调用帧中挂起：

![](img/B15619_12_05.png)

图 12.5：对堆栈协程的每次调用都会创建一个具有唯一堆栈指针的单独侧堆栈

#### 挂起和恢复无堆栈协程

无堆栈协程需要在其他地方（通常在堆上）存储协程帧，然后使用当前执行线程的堆栈来存储嵌套调用帧。

但这并不是全部真相。调用者负责创建调用帧，保存返回地址（程序计数器的当前值）和堆栈上的参数。调用者不知道自己正在调用一个会挂起和恢复的协程。因此，协程本身在被调用时需要创建协程帧，并将参数和寄存器从调用帧复制到协程帧中：

![](img/B15619_12_06.png)

图 12.6：无堆栈协程具有单独的协程帧（通常在堆上），其中包含恢复协程所需的状态

当协程最初挂起时，协程的堆栈帧从堆栈中弹出，但协程帧继续存在。协程帧的内存地址（句柄/指针）被返回给调用者：

![](img/B15619_12_07.png)

图 12.7：挂起的协程。协程帧包含恢复协程所需的所有信息。

要恢复协程，调用者使用先前收到的句柄，并调用一个恢复函数，并将协程句柄作为参数传递。恢复函数使用存储在协程帧中的挂起/恢复点来继续执行协程。对恢复函数的调用也是一个普通的函数调用，将生成一个堆栈帧，如下图所示：

![](img/B15619_12_08.png)

图 12.8：恢复协程为恢复调用创建一个新的调用帧。恢复函数使用协程状态的句柄从正确的挂起点恢复。

最后，当协程返回时，通常会被挂起并最终被释放。堆栈的状态如下图所示：

![](img/B15619_12_09.png)

图 12.9：协程帧在返回时被释放

没有为每个协程调用分配单独的侧边堆栈的一个重要后果是，当无堆栈协程被挂起时，它不能在堆栈上留下任何嵌套调用帧。记住，当控制权转回调用者时，调用者的调用帧必须位于堆栈顶部。

最后要提到的是，在某些情况下，协程帧所需的内存可以在调用者的调用帧内分配。当我们查看 C++20 协程时，我们将更详细地讨论这一点。

## 无堆栈与有堆栈协程

正如前一节所述，无堆栈协程使用当前运行线程的堆栈来处理嵌套函数调用。这样做的效果是无堆栈协程永远不会从嵌套调用帧中挂起。

有时堆栈式协程被称为**纤程**，在 Go 编程语言中被称为**goroutines**。堆栈式协程让我们想起线程，每个线程管理自己的堆栈。然而，堆栈式协程（或纤程）与操作系统线程之间有两个重大区别：

+   操作系统线程由内核调度，并在两个线程之间进行切换是内核模式操作。

+   大多数操作系统**抢占式**地切换操作系统线程（线程被调度程序中断），而两个纤程之间的切换是**合作**的。运行中的纤程会一直运行，直到将控制权交给可以调度另一个纤程的管理器。

还有一类称为**用户级线程**或**绿色线程**的线程。这些是轻量级线程，不涉及内核模式切换（因为它们在用户模式下运行，因此内核不知道）。纤程是用户级线程的一个例子。但用户级线程也可以由用户库或虚拟机抢占地调度。Java 线程是抢占式用户级线程的一个例子。

无堆栈协程还允许我们编写和组合多个并发运行的任务，但不需要每个流程单独的侧边堆栈。无堆栈协程和状态机密切相关。可以将状态机转换为协程，反之亦然。为什么了解这一点很有用？首先，这让你更好地理解无堆栈协程是什么。其次，如果你已经擅长识别可以使用状态机解决的问题，你可以更容易地看到协程可能适合作为适当解决方案的地方。状态机是非常通用的抽象，可以应用于各种问题。然而，状态机通常应用的一些领域包括解析、手势识别和 I/O 多路复用等。这些都是无堆栈协程在表达和性能方面真正闪耀的领域。

### 性能成本

协程是一种抽象，使我们能够以清晰简洁的方式编写惰性评估代码和异步程序。但是，创建和销毁协程以及挂起和恢复协程都会带来性能成本。在比较无堆栈和有堆栈协程的性能成本时，需要解决两个主要方面：*内存占用*和*上下文切换*。

### 内存占用

有栈协程需要一个单独的调用栈来处理来自嵌套调用帧的挂起。因此，在调用协程时，我们需要动态分配一块内存来存储这个新的侧栈。这立即引发了一个问题：我们需要分配多大的栈？除非我们有关于协程及其嵌套调用帧可以消耗多少栈的一些策略，否则我们可能需要一个大约与线程的正常调用栈大小相同的栈。

一些实现已经尝试使用分段栈，这将允许栈在必要时增长。另一种选择是从一个小的连续栈开始，然后在需要时将栈复制到一个更大的新分配的内存区域（类似于`std::vector`的增长）。Go 语言中的协程实现（goroutines）已经从使用分段栈切换到了动态增长的连续栈。

无栈协程不需要为单独的侧栈分配内存。相反，它们需要为每个协程帧分配一个单独的内存以支持挂起和恢复。这种分配发生在调用协程时（但不是在挂起/恢复时）。当协程返回时，调用帧被释放。

总之，有栈协程需要为协程帧和侧栈进行大量的初始内存分配，或者需要支持一个增长的栈。无栈协程只需要为协程帧分配内存。调用协程的内存占用可以总结如下：

+   无栈：协程帧

+   有栈：协程帧+调用栈

性能成本的下一个方面与挂起和恢复协程有关。

### 上下文切换

上下文切换可以发生在不同的级别。一般来说，当我们需要 CPU 在两个或多个正在进行的任务之间切换时，就会发生上下文切换。即将暂停的任务需要保存 CPU 的整个状态，以便在以后恢复时可以恢复。

在不同进程和操作系统线程之间切换是相当昂贵的操作，涉及系统调用，需要 CPU 进入内核模式。内存缓存被使无效，对于进程切换，包含虚拟内存和物理内存映射的表需要被替换。

挂起和恢复协程也是一种上下文切换，因为我们在多个并发流之间切换。在协程之间切换比在进程和操作系统线程之间切换要快得多，部分原因是它不涉及需要 CPU 在内核模式下运行的任何系统调用。

然而，当在有栈协程和无栈协程之间切换时仍然存在差异。有栈协程和无栈协程的上下文切换的相对运行时性能可能取决于调用模式。但总的来说，有栈协程的上下文切换操作更昂贵，因为在挂起和恢复时需要保存和恢复更多的信息，而无栈协程的恢复类似于正常的函数调用。

关于有栈与无栈的辩论在 C++社区已经进行了好几年，我会尽力避开这场辩论，总结它们都有有效的用例——有些用例会偏向有栈协程，而其他用例会偏向无栈协程。

为了让你更好地理解协程的执行和性能，这一部分稍微偏离了一下。让我们简要回顾一下你学到的内容。

## 到目前为止你学到的内容

协程是可以挂起和恢复的函数。普通函数没有这种能力，这使得可以删除返回的函数的调用帧。然而，一个被挂起的协程需要保持调用帧活动，以便在恢复时能够恢复协程的状态。协程比子例程更强大，并且在生成的机器代码中涉及更多的簿记工作。然而，由于协程与普通函数之间的密切关系，今天的编译器非常擅长优化无堆栈协程。

堆栈式协程可以看作是非抢占式用户级线程，而无堆栈协程提供了一种以直接命令方式编写状态机的方法，使用关键字`await`和`yield`来指定挂起点。

在对协程的一般抽象介绍之后，现在是时候了解 C++中如何实现无堆栈协程。

# C++中的协程

C++20 中添加的协程是无堆栈协程。也有使用第三方库在 C++中使用堆栈式协程的选项。最知名的跨平台库是 Boost.Fiber。C++20 无堆栈协程引入了新的语言构造，而 Boost.Fiber 是一个可以在 C++11 及以后版本中使用的库。在本书中我们不会进一步讨论堆栈式协程，而是专注于 C++20 中标准化的无堆栈协程。

C++20 中的无堆栈协程设计有以下目标：

+   在内存开销方面可扩展，这使得可以有更多的协程同时存在，与可能存在的线程或堆栈式协程数量相比。

+   高效的上下文切换，这意味着挂起和恢复协程应该与普通函数调用一样廉价。

+   高度灵活。C++协程有 15 多个自定义点，这为应用程序开发人员和库编写人员提供了很大的自由度，可以根据自己的喜好配置和塑造协程。关于协程应该如何工作的决定可以由我们开发人员确定，而不是硬编码在语言规范中。一个例子是协程在被调用后是否应该直接挂起，还是继续执行到第一个显式挂起点。在其他语言中，这些问题通常是硬编码的，但在 C++中，我们可以使用自定义点来定制这种行为。

+   不要求 C++异常来处理错误。这意味着您可以在关闭异常的环境中使用协程。请记住，协程是一种低级功能，类似于普通函数，在嵌入式环境和具有实时要求的系统中非常有用。

有了这些目标，C++协程可能一开始会有点复杂。

## 标准 C++中包含了什么（以及不包含什么）？

一些 C++特性是纯库特性（例如 Ranges 库），而其他特性是纯语言特性（例如使用`auto`关键字进行类型推断）。然而，有些特性需要对核心语言和标准库进行补充。C++协程就是其中之一；它们为语言引入了新的关键字，同时也向标准库添加了新的类型。

在语言方面，总结一下，我们有以下与协程相关的关键字：

+   `co_await`：挂起当前协程的运算符

+   `co_yield`：向调用者返回一个值并挂起协程

+   `co_return`：完成协程的执行，并且可以选择返回一个值

在库方面，有一个新的`<coroutine>`头文件，其中包括以下内容：

+   `std::coroutine_handle`：引用协程状态的模板类，使协程能够挂起和恢复

+   `std::suspend_never`：一个从不挂起的平凡等待类型

+   `std::suspend_always`：一个始终暂停的平凡等待类型

+   `std::coroutine_traits`：用于定义协程的承诺类型

C++20 附带的库类型是绝对最低限度的。例如，用于协程和调用者之间通信的基础设施不是 C++标准的一部分。为了有效地在应用程序代码中使用协程，我们需要的一些类型和函数已经在新的 C++提案中提出，例如模板类`task`和`generator`以及函数`sync_wait()`和`when_all()`。C++协程的库部分很可能会在 C++23 中得到补充。

在本书中，我将提供一些简化的类型来填补这一空白，而不是使用第三方库。通过实现这些类型，您将深入了解 C++协程的工作原理。然而，设计可以与协程一起使用的健壮库组件很难在不引入生命周期问题的情况下正确实现。因此，如果您计划在当前项目中使用协程，使用第三方库可能是比从头开始实现更好的选择。在撰写本文时，**CppCoro**库是这些通用原语的事实标准。该库由 Lewis Baker 创建，可在[`github.com/lewissbaker/cppcoro`](https://github.com/lewissbaker/cppcoro)上找到。

## 什么使 C++函数成为协程？

如果 C++函数包含关键字`co_await`、`co_yield`或`co_return`，则它是一个协程。此外，编译器对协程的返回类型也有特殊要求。但是，我们需要检查定义（主体）而不仅仅是声明，才能知道我们是否面对的是协程还是普通函数。这意味着协程的调用者不需要知道它调用的是协程还是普通函数。

与普通函数相比，协程还有以下限制：

+   协程不能使用像`f(const char*...)`这样的可变参数

+   协程不能返回`auto`或概念类型：`auto f()`

+   协程不能声明为`constexpr`

+   构造函数和析构函数不能是协程

+   `main()`函数不能是协程

一旦编译器确定一个函数是协程，它就会将协程与多种类型关联起来，以使协程机制工作。以下图表突出显示了在*调用者*使用*协程*时涉及的不同组件：

![](img/B15619_12_10.png)

图 12.10：协程与其调用者之间的关系

调用者和协程是我们通常在应用程序代码中实现的实际函数。

**返回对象**是协程返回的类型，通常是为某个特定用例设计的通用类模板，例如*生成器*或*异步任务*。*调用者*与返回对象交互以恢复协程并获取从协程中发出的值。返回对象通常将其所有调用委托给协程句柄。

**协程句柄**是对**协程状态**的非拥有句柄。通过协程句柄，我们可以恢复和销毁协程状态。

**协程状态**是我之前所说的协程帧。它是一个不透明的对象，这意味着我们不知道它的大小，也不能以其他方式访问它，而只能通过句柄。协程状态存储了恢复协程的一切必要条件。协程状态还包含**Promise**。

承诺对象是协程本身间接通过关键字`co_await`、`co_yield`和`co_return`进行通信的。如果从协程提交值或错误，它们将首先到达承诺对象。承诺对象充当协程和调用者之间的通道，但它们都无法直接访问承诺。

诚然，乍一看这可能看起来相当密集。一个完整但简单的例子将帮助你更好地理解不同的部分。

## 一个简单但完整的例子

让我们从一个最小的例子开始，以便理解协程的工作原理。首先，我们实现一个小的*协程*，在返回之前被挂起和恢复：

```cpp
auto coroutine() -> Resumable {    // Initial suspend
  std::cout << "3 ";
  co_await std::suspend_always{};  // Suspend (explicit)
  std::cout << "5 ";
}                                  // Final suspend then return 
```

其次，我们创建协程的*调用者*。注意程序的输出和控制流。这里是：

```cpp
int main() {            
  std::cout << "1 ";
  auto resumable = coroutine(); // Create coroutine state
  std::cout << "2 ";
  resumable.resume();           // Resume
  std::cout << "4 ";
  resumable.resume();           // Resume
  std::cout << "6 ";
}                               // Destroy coroutine state
// Outputs: 1 2 3 4 5 6 
```

第三，协程的返回对象`Resumable`需要被定义：

```cpp
class Resumable {                // The return object
  struct Promise { /*...*/ };    // Nested class, see below
  std::coroutine_handle<Promise> h_;
  explicit Resumable(std::coroutine_handle<Promise> h) : h_{h} {}
public:
  using promise_type = Promise;
  Resumable(Resumable&& r) : h_{std::exchange(r.h_, {})} {}
  ~Resumable() { if (h_) { h_.destroy(); } }
  bool resume() {
    if (!h_.done()) { h_.resume(); }
    return !h_.done();
  }
}; 
```

最后，承诺类型被实现为`Resumable`内部的嵌套类，如下所示：

```cpp
struct Promise {
  Resumable get_return_object() {
    using Handle = std::coroutine_handle<Promise>;
    return Resumable{Handle::from_promise(*this)};
  }
  auto initial_suspend() { return std::suspend_always{}; }
  auto final_suspend() noexcept { return std::suspend_always{}; }
  void return_void() {}
  void unhandled_exception() { std::terminate(); }
}; 
```

这个例子很简单，但涉及了很多值得注意和需要理解的东西：

+   `coroutine()`函数是一个协程，因为它包含了使用`co_await`的显式挂起/恢复点

+   协程不会产生任何值，但仍然需要返回一个类型（`Resumable`），具有一定的约束，以便调用者可以恢复协程。

+   我们正在使用一个名为`std::suspend_always`的*可等待类型*。

+   `resumable`对象的`resume()`函数从协程被挂起的地方恢复协程

+   `Resumable`是协程状态的所有者。当`Resumable`对象被销毁时，它使用`coroutine_handle`销毁协程

调用者、协程、协程句柄、承诺和可恢复之间的关系如下图所示：

![](img/B15619_12_11.png)

图 12.11：可恢复示例中涉及的函数/协程和对象之间的关系

现在是时候仔细看看每个部分了。我们将从`Resumable`类型开始。

### 协程返回对象

我们的协程返回一个`Resumable`类型的对象。这个`Resumable`类非常简单。这是协程返回的对象，调用者可以使用它来恢复和销毁协程。以下是完整的定义，以供您方便查看：

```cpp
class Resumable {               // The return object
  struct Promise { /*...*/ };   // Nested class
  std::coroutine_handle<Promise> h_;
  explicit Resumable(std::coroutine_handle<Promise> h) : h_{h} {}
public:
  using promise_type = Promise;
  Resumable(Resumable&& r) : h_{std::exchange(r.h_, {})} {}
  ~Resumable() { if (h_) { h_.destroy(); } }
  bool resume() {
    if (!h_.done()) { h_.resume(); }
    return !h_.done();
  }
}; 
```

`Resumable`是一个移动类型，它是协程句柄的所有者（因此控制协程的生命周期）。移动构造函数确保通过使用`std::exchange()`在源对象中清除协程句柄。当`Resumable`对象被销毁时，如果仍然拥有它，它将销毁协程。

`resume()`成员函数将恢复调用委托给协程句柄，如果协程仍然存活。

为什么我们需要在`Resumable`内部有成员类型别名`promise_type = Promise`？对于每个协程，还有一个关联的承诺对象。当编译器看到一个协程（通过检查函数体），它需要找出关联的承诺类型。为此，编译器使用`std::coroutine_traits<T>`模板，其中`T`是您的协程的返回类型。您可以提供`std::coroutine_traits<T>`的模板特化，或者利用`std::coroutine_traits`的默认实现将在协程的返回类型`T`中查找名为`promise_type`的`public`成员类型或别名。在我们的情况下，`Resumable::promise_type`是`Promise`的别名。

### 承诺类型

承诺类型控制协程的行为。以下是完整的定义，以供您方便查看：

```cpp
struct Promise {
  auto get_return_object() { return Resumable{*this}; }
  auto initial_suspend() { return std::suspend_always{}; }
  auto final_suspend() noexcept { return std::suspend_always{}; }
  void return_void() {}
  void unhandled_exception() { std::terminate(); }
}; 
```

我们不应直接调用这些函数；相反，编译器在将协程转换为机器代码时会插入对 promise 对象的调用。如果我们不提供这些成员函数，编译器就不知道如何为我们生成代码。您可以将 promise 视为协程控制器对象，负责：

+   产生从协程调用返回的值。这由函数`get_return_object()`处理。

+   通过实现函数`initial_suspend()`和`final_supsend()`定义协程创建时和销毁前的行为。在我们的`Promise`类型中，我们通过返回`std::suspend_always`（见下一节）来表示协程应在这些点挂起。

+   自定义协程最终返回时的行为。如果协程使用带有类型`T`的值的表达式的`co_return`，则 promise 必须定义一个名为`return_value(T)`的成员函数。我们的协程不返回任何值，但 C++标准要求我们提供称为`return_void()`的定制点，我们在这里留空。

+   处理在协程体内未处理的异常。在函数`unhandled_exception()`中，我们只是调用`std::terminate()`，但在后面的示例中我们将更优雅地处理它。

还有一些代码的最后部分需要更多的关注，即`co_await`表达式和可等待类型。

### 可等待类型

我们在代码中使用`co_await`添加了一个显式的挂起点，并传递了一个可等待类型`std::suspend_always`的实例。`std::suspend_always`的实现大致如下：

```cpp
struct std::suspend_always {
  constexpr bool await_ready() const noexcept { return false; }
  constexpr void await_suspend(coroutine_handle<>) const noexcept {}
  constexpr void await_resume() const noexcept {}
}; 
```

`std::suspend_always`被称为微不足道的可等待类型，因为它总是使协程挂起，说它永远不会准备好。还有另一种微不足道的可等待类型，总是报告自己准备好的，称为`std::suspend_never`：

```cpp
struct std::suspend_never {
  constexpr bool await_ready() const noexcept { return true; }
  constexpr void await_suspend(coroutine_handle<>) const noexcept {}
  constexpr void await_resume() const noexcept {}
}; 
```

我们可以创建自己的可等待类型，这将在下一章中介绍，但现在我们可以使用这两种微不足道的标准类型。

这完成了示例。但是当我们有了`Promise`和`Resumable`类型时，我们可以进行更多的实验。让我们看看在启动的协程中我们能做些什么。

### 传递我们的协程

一旦`Resumable`对象被创建，我们可以将它传递给其他函数，并从那里恢复它。我们甚至可以将协程传递给另一个线程。下面的示例展示了一些这种灵活性：

```cpp
auto coroutine() -> Resumable {
  std::cout << "c1 ";
  co_await std::suspend_always{};
  std::cout << "c2 ";
}                                
auto coro_factory() {             // Create and return a coroutine
  auto res = coroutine();
  return res;
}
int main() {
  auto r = coro_factory();
  r.resume();                     // Resume from main
  auto t = std::jthread{[r = std::move(r)]() mutable {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2s);
    r.resume();                   // Resume from thread
  }};
} 
```

前面的示例表明，一旦我们调用了我们的协程并获得了对它的句柄，我们就可以像任何其他可移动类型一样移动它。将它传递给其他线程的能力实际上在需要避免在特定线程上对协程状态进行可能的堆分配的情况下非常有用。

## 分配协程状态

协程状态，或协程帧，是协程在挂起时存储其状态的地方。协程状态的生命周期始于协程被调用时，并在协程执行`co_return`语句（或控制流离开协程体的末尾）时被销毁，除非它在此之前通过协程句柄被销毁。

协程状态通常在堆上分配。编译器会插入一个单独的堆分配。然而，在某些情况下，可以通过将协程状态内联到调用者的帧中（可以是普通的堆栈帧或另一个协程帧）来省略这个单独的堆分配。不幸的是，永远不能保证省略堆分配。

为了使编译器能够省略堆分配，协程状态的完整生存期必须严格嵌套在调用者的生存期内。此外，编译器需要找出协程状态的总大小，并且通常需要能够看到被调用协程的主体，以便其中的部分可以内联。像虚函数调用和调用其他翻译单元或共享库中的函数的情况通常会使这种情况变得不可能。如果编译器缺少所需的信息，它将插入堆分配。

协程状态的堆分配是使用`operator new`执行的。可以在 promise 类型上提供自定义的类级`operator new`，然后将其用于全局`operator new`。因此，可以检查堆分配是否被省略。如果没有，我们可以找出协程状态需要多少内存。以下是使用我们之前定义的`Promise`类型的示例：

```cpp
struct Promise {
  /* Same as before ... */
  static void* operator new(std::size_t sz) {
    std::cout << "custom new for size " << sz << '\n';
    return ::operator new(sz);
  }
  static void operator delete(void* ptr) {
    std::cout << "custom delete called\n";
    ::operator delete(ptr);
  }
} 
```

另一个验证使用特定 promise 类型的所有协程完全省略了堆分配的技巧是声明`operator new`和`operator delete`，但不包括它们的定义。如果编译器插入了对这些操作符的调用，程序将由于未解析的符号而无法链接。

## 避免悬空引用

协程可以在我们的代码中传递，这意味着我们需要非常小心地处理传递给协程的参数的生存期，以避免悬空引用。协程帧包含通常存储在堆栈上的对象的副本，例如局部变量和传递给协程的参数。如果协程通过引用接受参数，则*引用*被复制，而不是对象。这意味着当遵循函数参数的通常指导方针时，即通过引用传递`const`对象，我们很容易遇到悬空引用。

### 向协程传递参数

以下协程使用了对`const std::string`的引用：

```cpp
auto coroutine(const std::string& str) -> Resumable { 
  std::cout << str;
  co_return;
} 
```

假设我们有一个创建并返回协程的工厂函数，就像这样：

```cpp
auto coro_factory() {
  auto str = std::string{"ABC"};
  auto res = coroutine(str);
  return res;
} 
```

最后，一个使用协程的`main()`函数：

```cpp
int main() {
  auto coro = coro_factory();
  coro.resume();
} 
```

这段代码表现出未定义的行为，因为包含字符串`"ABC"`的`std::string`对象在协程尝试访问它时已经不再存在。希望这对你来说不是什么意外。这个问题类似于让 lambda 通过引用捕获变量，然后将 lambda 传递给其他代码而不保持引用对象的生存。当传递捕获变量的 lambda 时也可以实现类似的例子：

```cpp
auto lambda_factory() {
  auto str = std::string{"ABC"};
  auto lambda = [&str]() {         // Capture str by reference
    std::cout << str;     
  };
  return lambda;                   // Ops! str in lambda becomes
}                                  // a dangling reference
int main() {
  auto f = lambda_factory();
  f();                             // Undefined behavior
} 
```

正如你所看到的，使用 lambda 也可能出现相同的问题。在*第二章*，*基本的 C++技术*中，我警告过你使用 lambda 捕获引用的问题，通常最好通过值捕获来避免这个问题。

避免悬空引用的解决方案与协程类似：在使用协程时避免通过引用传递参数。而是使用按值传递，整个参数对象将安全地放置在协程帧中：

```cpp
auto coroutine(std::string str) -> Resumable {  // OK, by value!
  std::cout << str;
  co_return;
}
auto coro_factory() {
  auto str = std::string{"ABC"};
  auto res = coroutine(str);
  return res;
}
int main() {
  auto coro = coro_factory();
  coro.resume();                                 // OK!
} 
```

参数是使用协程时生存期问题的一个重要且常见的来源，但并不是唯一的来源。现在我们将探讨一些与协程和悬空引用相关的其他陷阱。

### 作为协程的成员函数

成员函数也可以是协程。例如，我们可以在成员函数中使用`co_await`，就像下面的例子一样：

```cpp
struct Widget {
auto coroutine() -> Resumable {       // A member function 
    std::cout << i_++ << " ";         // Access data member
    co_await std::suspend_always{};
    std::cout << i_++ << " ";
  }
  int i_{};
};
int main() {
  auto w = Widget{99};
  auto coro = w.coroutine();
  coro.resume();
  coro.resume();
}
// Prints: 99 100 
```

重要的是要理解，调用者`coroutine()`（在这种情况下是`main()`）有责任确保`Widget`对象`w`在整个协程的生命周期内保持存活。协程正在访问所属对象的数据成员，但`Widget`对象本身*不*由协程保持存活。如果我们将协程传递给程序的其他部分，这很容易成为一个问题。

假设我们正在使用一些协程工厂函数，就像之前演示的那样，但是返回一个成员函数协程：

```cpp
auto widget_coro_factory() {      // Create and return a coroutine
  auto w = Widget{};
  auto coro = w.coroutine();
  return coro; 
}                                 // Object w destructs here
int main() {
  auto r = widget_coro_factory();
  r.resume();                     // Undefined behavior 
  r.resume();                  
} 
```

这段代码表现出未定义的行为，因为我们现在有一个从协程到在`widget_coro_factory()`函数中创建和销毁的`Widget`对象的悬空引用。换句话说，我们最终得到了两个具有不同生命周期的对象，其中一个对象引用另一个对象，但没有明确的所有权。

### 作为协程的 lambda

不仅成员函数可以成为协程。还可以通过在 lambda 的主体中插入`co_await`、`co_return`和/或`co_yield`来使用 lambda 表达式创建协程。

协程 lambda 可能会有一些额外的棘手问题。更好地理解协程 lambda 最常见的生命周期问题的一种方法是考虑函数对象。回想一下*第二章*，*Essential C++ Techniques*，lambda 表达式被编译器转换为函数对象。这个对象的类型是一个实现了调用运算符的类。现在，假设我们在 lambda 的主体中使用`co_return`；这意味着调用运算符`operator()()`变成了一个协程。

考虑以下使用 lambda 的代码：

```cpp
auto lambda = [](int i) -> Resumable {
  std::cout << i;
  co_return;              // Make it a coroutine
};
auto coro = lambda(42);   // Call, creates the coroutine frame
coro.resume();            // Outputs: 42 
```

lambda 对应的类型看起来像这样：

```cpp
struct LambdaType {
  auto operator()(int i) -> Resumable {  // Member function
    std::cout << i;                      // Body
    co_return;
  }
};
auto lambda = LambdaType{};
auto coro = lambda(42);
coro.resume(); 
```

这里需要注意的重要事情是，实际的协程是一个*成员函数*，即调用运算符`operator()()`。前面的部分已经展示了拥有协程成员函数的陷阱：我们需要在协程的生命周期内保持对象的存活。在前面的例子中，这意味着我们需要在协程帧存活期间保持名为`lambda`的函数对象存活。

一些 lambda 的用法很容易在协程帧被销毁之前意外销毁函数对象。例如，通过使用*立即调用 lambda*，我们很容易陷入麻烦：

```cpp
auto coro = [i = 0]() mutable -> Resumable { 
  std::cout << i++; 
  co_await std::suspend_always{};
  std::cout << i++;
}();               // Invoke lambda immediately
coro.resume();     // Undefined behavior! Function object
coro.resume();     // already destructed 
```

这段代码看起来无害；lambda 没有通过引用捕获任何东西。然而，lambda 表达式创建的函数对象是一个临时对象，一旦被调用并且协程捕获了对它的引用，它将被销毁。当协程恢复时，程序很可能会崩溃或产生垃圾。

再次，更好地理解这一点的方法是将 lambda 转换为具有定义的`operator()`的普通类：

```cpp
struct LambdaType {
  int i{0};
  auto operator()() -> Resumable {
    std::cout << i++; 
    co_await std::suspend_always{};
    std::cout << i++;
  }
};
auto coro = LambdaType{}(); // Invoke operator() on temporary object
coro.resume();              // Ops! Undefined behavior 
```

现在你可以看到，这与我们有一个成员函数是协程的情况非常相似。函数对象不会被协程帧保持存活。

### 防止悬空引用的指导方针

除非你有接受引用参数的充分理由，如果你正在编写一个协程，选择通过值接受参数。协程帧将保持你传递给它的对象的完整副本，并且保证对象在协程帧存活期间存活。

如果你正在使用 lambda 或成员函数作为协程，特别注意协程所属对象的生命周期。记住对象（或函数对象）*不*存储在协程帧中。调用协程的责任是保持其存活。

## 处理错误

有不同的方法将错误从协程传递回调用它或恢复它的代码部分。我们不必使用异常来标志错误。相反，我们可以根据需要自定义错误处理。

协程可以通过抛出异常或在客户端从协程获取值时返回错误代码，将错误传递回客户端。

如果我们使用异常并且异常从协程体中传播出来，那么承诺对象的函数`unhandled_exception()`就会被调用。这个调用发生在编译器插入的 catch 块内部，因此可以使用`std::current_exception()`来获取抛出的异常。然后可以将`std::current_exception()`的结果存储在协程中作为`std::exception_ptr`，并在以后重新抛出。在下一章中使用异步协程时，您将看到这方面的例子。

## 定制点

您已经看到了许多定制点，我认为一个有效的问题是：为什么有这么多定制点？

+   **通用性**：定制点使得可以以各种方式使用协程。对于如何使用 C++协程，几乎没有什么假设。库编写者可以定制`co_await`、`co_yield`和`co_return`的行为。

+   **效率**：一些定制点是为了根据使用情况启用可能的优化。一个例子是`await_ready()`，如果值已经计算出来，它可以返回`true`以避免不必要的暂停。

还应该说的是，我们暴露于这些定制点，是因为 C++标准没有提供任何类型（除了`std::coroutine_handle`）来与协程通信。一旦它们就位，我们就可以重用这些类型，而不用太担心其中一些定制点。然而，了解定制点对于充分理解如何有效使用 C++协程是有价值的。

# 生成器

生成器是一种向其调用者产生值的协程类型。例如，在本章开头，我演示了生成器`iota()`产生递增的整数值。通过实现一个通用的生成器类型，它可以充当迭代器，我们可以简化实现与基于范围的`for`循环、标准库算法和范围兼容的迭代器的工作。一旦我们有了生成器模板类，我们就可以重用它。

到目前为止，在本书中，您大多数时候看到的是在访问容器元素和使用标准库算法时的迭代器。然而，迭代器不一定要与容器绑定。可以编写产生值的迭代器。

## 实现生成器

我们即将实现的生成器是基于 CppCoro 库中的生成器。生成器模板旨在用作协程的返回类型，用于生成一系列值。应该可以将此类型的对象与基于范围的`for`循环和接受迭代器和范围的标准算法一起使用。为了实现这一点，我们将实现三个组件：

+   `Generator`，这是返回对象

+   `Promise`，作为协程控制器

+   `Iterator`，是客户端和`Promise`之间的接口

这三种类型紧密耦合，它们与协程状态之间的关系在下图中呈现：

![](img/B15619_12_12.png)

图 12.12：迭代器、生成器、Promise 和协程状态之间的关系

返回对象，这种情况下是`Generator`类，与`Promise`类型紧密耦合；`Promise`类型负责创建`Generator`对象，而`Generator`类型负责向编译器公开正确的`promise_type`。这是`Generator`的实现：

```cpp
template <typename T>
class Generator {
  struct Promise { /* ... */ };   // See below
  struct Sentinel {};  
  struct Iterator { /* ... */ };  // See below

  std::coroutine_handle<Promise> h_;
  explicit Generator(std::coroutine_handle<Promise> h) : h_{h} {}
public: 
  using promise_type = Promise;
  Generator(Generator&& g) : h_(std::exchange(g.h_, {})) {}
  ~Generator() { if (h_) { h_.destroy();  } }
  auto begin() {
    h_.resume();
    return Iterator{h_};
  }
  auto end() { return Sentinel{}; }
}; 
```

`Promise`和`Iterator`的实现将很快跟进。`Generator`与我们之前定义的`Resumable`类并没有太大的不同。`Generator`是协程的返回对象，也是`std::coroutine_handle`的所有者。生成器是可移动类型。在移动时，协程句柄被转移到新构造的`Generator`对象。当拥有协程句柄的生成器被销毁时，它通过在协程句柄上调用`destroy`来销毁协程状态。

`begin()`和`end()`函数使得可以在基于范围的`for`循环和接受范围的算法中使用这个生成器。`Sentinel`类型是空的——它是一个虚拟类型——`Sentinel`实例是为了能够将某些东西传递给`Iterator`类的比较运算符。`Iterator`的实现如下：

```cpp
struct Iterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  std::coroutine_handle<Promise> h_;  // Data member

  Iterator& operator++() {
    h_.resume();
    return *this;
  }
  void operator++(int) { (void)operator++(); }
  T operator*() const { return h_.promise().value_; }
  T* operator->() const { return std::addressof(operator*()); }
  bool operator==(Sentinel) const { return h_.done(); }
}; 
```

迭代器需要在数据成员中存储协程句柄，以便它可以将调用委托给协程句柄和 promise 对象：

+   当迭代器被解引用时，它返回由 promise 持有的当前值

+   当迭代器递增时，它恢复协程

+   当迭代器与哨兵值进行比较时，迭代器会忽略哨兵并将调用委托给协程句柄，协程句柄知道是否还有更多元素要生成

现在只剩下`Promise`类型需要我们实现。`Promise`的完整定义如下：

```cpp
struct Promise {
  T value_;
  auto get_return_object() -> Generator {
    using Handle = std::coroutine_handle<Promise>;
    return Generator{Handle::from_promise(*this)};
  }
  auto initial_suspend() { return std::suspend_always{}; }
  auto final_suspend() noexcept { return std::suspend_always{}; }
  void return_void() {}
  void unhandled_exception() { throw; }
  auto yield_value(T&& value) {
    value_ = std::move(value);
    return std::suspend_always{};
  }
  auto yield_value(const T& value) {
    value_ = value;
    return std::suspend_always{};
  }
}; 
```

我们的生成器的 promise 对象负责：

+   创建`Generator`对象

+   定义初始和最终挂起点达到时的行为

+   跟踪从协程中产生的最后一个值

+   处理协程主体抛出的异常

就是这样！我们现在已经把所有的部分都放在了一起。一个返回某种`Generator<T>`类型的协程现在可以使用`co_yield`来懒惰地产生值。协程的调用者与`Generator`和`Iterator`对象交互以检索值。对象之间的交互如下所示：

![](img/B15619_12_13.png)

图 12.13：调用者与生成器和迭代器对象通信，以从协程中检索值

现在，让我们看看如何使用新的`Generator`模板以及它如何简化各种迭代器的实现。

## 使用`Generator`类

这个例子受到了 Gor Nishanov 在 CppCon 2016 上的讲座*C++ Coroutines: Under the covers*的启发([`sched.co/7nKt`](https://sched.co/7nKt))。它清楚地演示了我们如何从刚刚实现的生成器类型中受益。现在可以像这样实现小型可组合的生成器：

```cpp
template <typename T>
auto seq() -> Generator<T> {
  for (T i = {};; ++i) {
    co_yield i;
  }
}
template <typename T>
auto take_until(Generator<T>& gen, T value) -> Generator<T> {
  for (auto&& v : gen) {
    if (v == value) {
      co_return;
    }
    co_yield v;
  }
}
template <typename T>
auto add(Generator<T>& gen, T adder) -> Generator<T> {
  for (auto&& v : gen) {
    co_yield v + adder;
  }
} 
```

一个小的使用示例演示了我们可以将生成器传递给基于范围的`for`循环：

```cpp
int main() {
  auto s = seq<int>();
  auto t = take_until<int>(s, 10);
  auto a = add<int>(t, 3);
  int sum = 0;
  for (auto&& v : a) {
      sum += v;
  }
  return sum; // returns 75
} 
```

生成器是惰性评估的。直到程序达到`for`循环时，才会产生值，从生成器链中拉取值。

这个程序的另一个有趣之处是，当我使用启用优化的 Clang 10 编译它时，*整个*程序的汇编代码看起来像这样：

```cpp
main:  # @main
mov  eax, 75
ret 
```

太棒了！程序简单地定义了一个返回值为`75`的主函数。换句话说，编译器优化器能够在编译时完全评估生成器链，并得出单个值`75`。

我们的`Generator`类也可以与范围算法一起使用。在下面的示例中，我们使用算法`includes()`来查看序列`{5,6,7}`是否是生成器产生的数字的子序列：

```cpp
int main() { 
  auto s = seq<int>();                           // Same as before
  auto t = take_until<int>(s, 10);
  auto a = add<int>(t, 3);
  const auto v = std::vector{5, 6, 7};
  auto is_subrange = std::ranges::includes(a, v); // True
} 
```

通过实现`Generator`模板，我们可以重用它来实现各种生成器函数。我们已经实现了一个通用且非常有用的库组件，应用代码可以在构建惰性生成器时从中受益。

### 解决生成器问题

现在我将提出一个小问题，我们将尝试使用不同的技术来解决它，以了解我们可以用生成器替换哪些编程习惯。我们即将编写一个小型实用程序，用于在起始值和停止值之间生成线性间隔序列。

如果您一直在使用 MATLAB/Octave 或 Python NumPy，您可能会认识到使用名为`linspace()`的函数生成均匀（线性）间隔数字的方式。这是一个方便的实用程序，可以在各种上下文中使用任意范围。

我们将称我们的生成器为`lin_space()`。以下是一个使用示例，在`2.0`和`3.0`之间生成五个等间距值：

```cpp
for (auto v: lin_space(2.0f, 3.0f, 5)) {
  std::cout << v << ", ";
}
// Prints: 2.0, 2.25, 2.5, 2.75, 3.0, 
```

在生成浮点值时，我们必须要小心，因为我们不能简单地计算每个步骤的大小（在前面的示例中为 0.25）并累积它，因为步长可能无法使用浮点数据类型精确表示。可能的舍入误差将在每次迭代中累积，最终我们可能会得到完全荒谬的值。相反，我们需要做的是使用线性插值在特定增量上计算开始和停止值之间的数字。

C++20 在`<cmath>`中添加了一个方便的实用程序，称为`std::lerp()`，它计算两个值之间的线性插值，并指定一个特定的量。在我们的情况下，量将是 0.0 到 1.0 之间的值；量为 0 返回`start`值，量为 1.0 返回`stop`值。以下是使用`std::lerp()`的几个示例：

```cpp
auto start = -1.0;
auto stop = 1.0;
std::lerp(start, stop, 0.0);    // -1.0
std::lerp(start, stop, 0.5);    //  0.0
std::lerp(start, stop, 1.0);    //  1.0 
```

我们即将编写的`lin_space()`函数将全部使用以下小型实用函数模板：

```cpp
template <typename T>
auto lin_value(T start, T stop, size_t index, size_t n) {  
  assert(n > 1 && index < n);
  const auto amount = static_cast<T>(index) / (n - 1);
  const auto v = std::lerp(start, stop, amount);   // C++20
  return v;
} 
```

该函数返回范围[`start`，`stop`]中线性序列中的一个值。`index`参数是我们即将生成的`n`个总数中的当前数字。

有了`lin_value()`辅助程序，我们现在可以轻松实现`lin_space()`生成器。在看到使用协程的解决方案之前，我们将研究其他常见技术。接下来的部分将探讨在实现`lin_space()`时使用的不同方法：

+   急切地生成并返回所有值

+   使用回调（惰性）

+   使用自定义迭代器（惰性）

+   使用 Ranges 库（惰性）

+   使用我们的`Generator`类的协程（惰性）

对于每个示例，都将简要反映每种方法的优缺点。

#### 一个急切的线性范围

我们将首先实现一个简单的急切版本，计算范围内的所有值并返回一个包含所有值的向量：

```cpp
template <typename T>
auto lin_space(T start, T stop, size_t n) {
  auto v = std::vector<T>{};
  for (auto i = 0u; i < n; ++i)
    v.push_back(lin_value(start, stop, i, n));
  return v;
} 
```

由于这个版本返回一个标准容器，所以可以将返回值与基于范围的`for`循环和其他标准算法一起使用：

```cpp
for (auto v : lin_space(2.0, 3.0, 5)) {
  std::cout << v << ", ";
}
// Prints: 2, 2.25, 2.5, 2.75, 3, 
```

这个版本很直接，而且相当容易阅读。缺点是我们需要分配一个向量并填充*所有*值，尽管调用者不一定对所有值感兴趣。这个版本也缺乏可组合性，因为没有办法在首先生成所有值之前过滤中间的元素。

现在让我们尝试实现`lin_space()`生成器的惰性版本。

#### 使用回调的惰性版本

在*第十章* *代理对象和惰性求值*中，我们得出结论，可以通过使用回调函数来实现惰性求值。我们将要实现的惰性版本将基于将回调传递给`lin_space()`并在发出值时调用回调函数：

```cpp
template <typename T, typename F>
requires std::invocable<F&, const T&>               // C++20 
void lin_space(T start, T stop, std::size_t n, F&& f) {
  for (auto i = 0u; i < n; ++i) {
    const auto y = lin_value(start, stop, i, n);
    f(y);
  }
} 
```

如果我们想打印生成器产生的值，可以这样调用该函数：

```cpp
auto print = [](auto v) { std::cout << v << ", "; };
lin_space(-1.f, 1.f, 5, print);
// Prints: -1, -0.5, 0, 0.5, 1, 
```

现在迭代发生在`lin_space()`函数内部。无法取消生成器，但通过一些更改，我们可以让回调函数返回一个`bool`来指示是否希望生成更多元素。

这种方法有效，但不太优雅。这种设计的问题在尝试组合生成器时变得更加明显。如果我们想要添加一个选择一些特殊值的过滤器，我们最终会有嵌套的回调函数。

我们现在将继续看如何实现基于迭代器的解决方案。

### 迭代器实现

另一种选择是实现一个符合范围概念的类型，通过暴露`begin()`和`end()`迭代器。在这里定义的类模板`LinSpace`使得可以迭代线性值的范围：

```cpp
template <typename T>
struct LinSpace {
  LinSpace(T start, T stop, std::size_t n)
      : begin_{start, stop, 0, n}, end_{n} {}
  struct Iterator {
    using difference_type = void;
    using value_type = T;
    using reference = T;
    using pointer = T*;
    using iterator_category = std::forward_iterator_tag;
    void operator++() { ++i_; }
    T operator*() { return lin_value(start_, stop_, i_, n_);}
    bool operator==(std::size_t i) const { return i_ == i; } 
    T start_{};
    T stop_{};
    std::size_t i_{};
    std::size_t n_{};
  };
  auto begin() { return begin_; }
  auto end() { return end_; }
 private:
  Iterator begin_{};
  std::size_t end_{};
};
template <typename T>
auto lin_space(T start, T stop, std::size_t n) {
  return LinSpace{start, stop, n};
} 
```

这个实现非常高效。然而，它受到大量样板代码的困扰，我们试图封装的小算法现在分散在不同的部分：`LinSpace`构造函数实现了设置起始和停止值的初始工作，而计算值所需的工作最终在`Iterator`类的成员函数中完成。与我们看到的其他版本相比，这使得算法的实现更难理解。

### 使用 Ranges 库的解决方案

另一种选择是使用 Ranges 库（C++20）中的构建模块来组合我们的算法，如下所示：

```cpp
template <typename T>
auto lin_space(T start, T stop, std::size_t n) {
  return std::views::iota(std::size_t{0}, n) |
    std::views::transform(= {
      return lin_value(start, stop, i, n);
    });
} 
```

在这里，我们将整个算法封装在一个小函数中。我们使用`std::views::iota`为我们生成索引。将索引转换为线性值是一个简单的转换，可以在`iota`视图之后链接。

这个版本高效且可组合。从`lin_space()`返回的对象是`std::ranges::view`类型的随机访问范围，可以使用基于范围的`for`循环进行迭代，或者传递给其他算法。

最后，是时候使用我们的`Generator`类来将我们的算法实现为一个协程。

#### 使用协程的解决方案

在看了不少于四个版本的同一个问题之后，我们现在已经达到了最后的解决方案。在这里，我将呈现一个使用之前实现的通用`Generator`类模板的版本：

```cpp
template <typename T> 
auto lin_space(T start, T stop, std::size_t n) -> Generator<T> {
   for (auto i = 0u; i < n; ++i) {
     co_yield lin_value(start, stop, i, n);
   }
 } 
```

它紧凑、简单明了。通过使用`co_yield`，我们可以以类似于简单的急切版本的方式编写代码，但不需要收集所有值到一个容器中。可以基于协程链式多个生成器，正如你将在本章末尾看到的那样。

这个版本也兼容基于范围的`for`循环和标准算法。然而，这个版本暴露了一个输入范围，所以不可能跳过任意数量的元素，而使用 Ranges 库的版本是可以的。

### 结论

显然，有多种方法可以做到这一点。但为什么我展示了所有这些方法呢？

首先，如果你是新手协程，希望你能开始看到在哪些情况下使用协程是有利的。

其次，`Generator`模板和使用`co_yield`允许我们以非常清晰简洁的方式实现惰性生成器。当我们将解决方案与其他版本进行比较时，这一点变得很明显。

最后，一些方法在这个例子问题中可能看起来很牵强，但在其他情境中经常被使用。C++默认是一种急切的语言，许多人（包括我自己）已经习惯于创建类似急切版本的代码。使用回调的版本可能看起来很奇怪，但在异步代码中是一个常用的模式，协程可以包装或替代那些基于回调的 API。

我们实现的生成器类型部分基于 CppCoro 库中的同步生成器模板。CppCoro 还提供了一个`async_generator`模板，它使得可以在生成器协程中使用`co_await`运算符。我在本章中提供了`Generator`模板，以演示如何实现生成器以及如何与协程交互。但是，如果您计划在代码中开始使用生成器，请考虑使用第三方库。

## 使用生成器的真实世界示例

当示例稍微复杂时，使用协程简化迭代器的示例效果非常好。使用`Generator`类的`co_yield`允许我们高效地实现和组合小算法，而无需编写大量模板代码来将它们粘合在一起。下一个示例将尝试证明这一点。

### 问题

我们将在这里通过一个示例来演示如何使用我们的`Generator`类来实现一个压缩算法，该算法可以用于搜索引擎中压缩通常存储在磁盘上的搜索索引。该示例在 Manning 等人的书籍《信息检索导论》中有详细描述，该书可以在[`nlp.stanford.edu/IR-book/`](https://nlp.stanford.edu/IR-book/)免费获取。以下是简要背景和问题的简要描述。

搜索引擎使用称为**倒排索引**的数据结构的某种变体。它类似于书末的索引。使用该索引，我们可以找到包含我们正在搜索的术语的所有页面。

现在想象一下，我们有一个充满食谱的数据库，并且我们为该数据库构建了一个倒排索引。该索引的部分可能看起来像这样：

![](img/B15619_12_14.png)

图 12.14：具有三个术语及其相应的文档引用列表的倒排索引

每个术语都与一个排序的文档标识符列表相关联。（例如，术语**苹果**包含在 ID 为**4**、**9**、**67**和**89**的食谱中。）如果我们想要查找同时包含**豆子**和**辣椒**的食谱，我们可以运行类似合并的算法来找到**豆子**和**辣椒**列表的交集：

![](img/B15619_12_15.png)

图 12.15：“豆子”和“辣椒”术语的文档列表的交集

现在想象一下，我们有一个大型数据库，并且我们选择用 32 位整数表示文档标识符。对于出现在许多文档中的术语，文档标识符列表可能会变得非常长，因此我们需要压缩这些列表。其中一种可能的方法是使用增量编码结合可变字节编码方案。

### 增量编码

由于列表是排序的，我们可以不保存文档标识符，而是存储两个相邻元素之间的**间隔**。这种技术称为**增量编码**或**间隔编码**。以下图表显示了使用文档 ID 和间隔的示例：

![](img/B15619_12_16.png)

图 12.16：间隔编码存储列表中两个相邻元素之间的间隔

间隔编码非常适合这种类型的数据；因此经常使用的术语将具有许多小间隔。真正长的列表将只包含非常小的间隔。在列表进行间隔编码之后，我们可以使用可变字节编码方案来实际压缩列表，通过使用较少的字节来表示较小的间隔。

但首先，让我们开始实现间隔编码功能。我们将首先编写两个小协程，用于执行间隔编码/解码。编码器将排序的整数序列转换为间隔序列：

```cpp
template <typename Range>
auto gap_encode(Range& ids) -> Generator<int> {
  auto last_id = 0;
  for (auto id : ids) {
    const auto gap = id - last_id;
    last_id = id;
    co_yield gap;
  }
} 
```

通过使用`co_yield`，我们无需急切地传递完整的数字列表并分配一个大的输出间隔列表。相反，协程会懒惰地处理一个数字。请注意，函数`gap_encode()`包含了有关如何将文档 ID 转换为间隔的所有信息。将其实现为传统的迭代器是可能的，但这将使逻辑分散在构造函数和迭代器操作符中。

我们可以编写一个小程序来测试我们的间隔编码器：

```cpp
int main() {
  auto ids = std::vector{10, 11, 12, 14};
  auto gaps = gap_encode();
  for (auto&& gap : gaps) {
    std::cout << gap << ", ";
  }
} // Prints: 10, 1, 1, 2, 
```

解码器则相反；它以间隔的范围作为输入，并将其转换为有序数字列表：

```cpp
template <typename Range>
auto gap_decode(Range& gaps) -> Generator<int> {
  auto last_id = 0;
  for (auto gap : gaps) {
    const auto id = gap + last_id;
    co_yield id;
    last_id = id;
  }
} 
```

通过使用间隔编码，我们平均可以存储更小的数字。但由于我们仍然使用`int`值来存储小间隔，如果将这些间隔保存到磁盘上，我们并没有真正获得任何好处。不幸的是，我们不能只使用较小的固定大小数据类型，因为仍然有可能遇到需要完整 32 位`int`的非常大的间隔。我们希望的是以更少的位数存储小间隔，如下图所示：

![](img/B15619_12_17.png)

图 12.17：小数字应该使用更少的字节

为了使这个列表在物理上更小，我们可以使用**可变字节编码**，这样小间隔可以用比大间隔更少的字节进行编码，如前图所示。

#### 可变字节编码

可变字节编码是一种非常常见的压缩技术。UTF-8 和 MIDI 消息是一些使用这种技术的众所周知的编码。为了在编码时使用可变数量的字节，我们使用每个字节的 7 位作为实际有效载荷。每个字节的第一位表示**续位**。如果还有更多字节要读取，则设置为`0`，如果是编码数字的最后一个字节，则设置为`1`。编码方案在下图中有例示：

![](img/B15619_12_18.png)

图 12.18：使用可变字节编码，只需要一个字节来存储十进制值 3，而需要两个字节来编码十进制值 1025

现在我们准备实现可变字节编码和解码方案。这比增量编码要复杂一些。编码器应该将一个数字转换为一个或多个字节的序列：

```cpp
auto vb_encode_num(int n) -> Generator<std::uint8_t> {
  for (auto cont = std::uint8_t{0}; cont == 0;) {
    auto b = static_cast<std::uint8_t>(n % 128);
    n = n / 128;
    cont = (n == 0) ? 128 : 0;
    co_yield (b + cont);
  }
} 
```

续位，代码中称为`cont`，要么是 0，要么是 128，对应的位序列是 10000000。这个例子中的细节并不重要，但为了使编码更容易，字节是以相反的顺序生成的，这样最不重要的字节首先出现并不是问题，因为我们可以在解码过程中轻松处理这个问题。

有了数字编码器，就可以轻松地对一系列数字进行编码，并将它们转换为一系列字节：

```cpp
template <typename Range>
auto vb_encode(Range& r) -> Generator<std::uint8_t> {
  for (auto n : r) {
    auto bytes = vb_encode_num(n);
    for (auto b : bytes) {
      co_yield b;
    }
  }
} 
```

解码器可能是最复杂的部分。但同样，它完全封装在一个单一函数中，并具有清晰的接口：

```cpp
template <typename Range>
auto vb_decode(Range& bytes) -> Generator<int> {
  auto n = 0;
  auto weight = 1;
  for (auto b : bytes) {
    if (b < 128) {  // Check continuation bit
      n += b * weight;
      weight *= 128;
    } 
    else {
      // Process last byte and yield
      n += (b - 128) * weight;
      co_yield n;
      n = 0;       // Reset
      weight = 1;  // Reset
    }
  }
} 
```

如您所见，这段代码中几乎没有需要的样板代码。每个协程封装了所有状态，并清楚地描述了如何一次处理一个部分。

我们还需要将间隔编码器与可变字节编码器结合起来，以压缩我们的文档标识符排序列表：

```cpp
template <typename Range>
auto compress(Range& ids) -> Generator<int> {
  auto gaps = gap_encode(ids);
  auto bytes = vb_encode(gaps);
  for (auto b : bytes) {
    co_yield b;
  }
} 
```

解压缩是`vb_decode()`后跟`gap_decode()`的简单链接：

```cpp
template <typename Range>
auto decompress(Range& bytes) -> Generator<int> {
  auto gaps = vb_decode(bytes);
  auto ids = gap_decode(gaps);
  for (auto id : ids) {
    co_yield id;
  }
} 
```

由于`Generator`类公开了迭代器，我们甚至可以进一步使用 iostreams 将值流式传输到磁盘上。 （尽管更现实的方法是使用内存映射 I/O 以获得更好的性能。）以下是两个将压缩数据写入磁盘并从磁盘读取的小函数：

```cpp
template <typename Range>
void write(const std::string& path, Range& bytes) {
  auto out = std::ofstream{path, std::ios::out | std::ofstream::binary};
  std::ranges::copy(bytes.begin(), bytes.end(),    
                    std::ostreambuf_iterator<char>(out));
}
auto read(std::string path) -> Generator<std::uint8_t> {
  auto in = std::ifstream {path, std::ios::in | std::ofstream::binary};
  auto it = std::istreambuf_iterator<char>{in};
  const auto end = std::istreambuf_iterator<char>{};
  for (; it != end; ++it) {
    co_yield *it;
  }
} 
```

一个小的测试程序将结束这个例子：

```cpp
int main() {
  {
    auto documents = std::vector{367, 438, 439, 440};
    auto bytes = compress(documents);
    write("values.bin", bytes);
  }
  {
    auto bytes = read("values.bin");
    auto documents = decompress(bytes);
    for (auto doc : documents) {
      std::cout << doc << ", ";
    }
  }
}
// Prints: 367, 438, 439, 440, 
```

这个例子旨在表明我们可以将惰性程序分成小的封装协程。C++协程的低开销使它们适合构建高效的生成器。我们最初实现的`Generator`是一个完全可重用的类，可以帮助我们最小化这类示例中的样板代码量。

这结束了关于生成器的部分。我们现在将继续讨论在使用协程时的一些一般性能考虑。

# 性能

每次创建协程（首次调用时），都会分配一个协程帧来保存协程状态。帧可以在堆上分配，或者在某些情况下在堆栈上分配。但是，并没有完全避免堆分配的保证。如果您处于禁止堆分配的情况（例如，在实时环境中），协程可以在不同的线程中创建并立即挂起，然后传递给实际需要使用协程的程序部分。挂起和恢复保证不会分配任何内存，并且具有与普通函数调用相当的成本。

在撰写本书时，编译器对协程有实验性支持。小型实验显示了与性能相关的有希望的结果，表明协程对优化器友好。但是，我不会在本书中为您提供任何协程的基准测试。相反，我向您展示了无栈协程是如何评估的，以及如何可能以最小的开销实现协程。

生成器示例表明，协程可能对编译器非常友好。我们在该示例中编写的生成器链是在运行时完全评估的。实际上，这是 C++协程的一个非常好的特性。它们使我们能够编写对编译器和人类都易于理解的代码。C++协程通常会产生易于优化的干净代码。

在同一线程上执行的协程可以共享状态，而无需使用任何锁原语，因此可以避免同步多个线程所产生的性能开销。这将在下一章中进行演示。

# 摘要

在本章中，您已经了解了如何使用 C++协程来使用关键字`co_yield`和`co_return`构建生成器。为了更好地理解 C++无栈协程与有栈协程的区别，我们对两者进行了比较，并查看了 C++协程提供的定制点。这使您深刻了解了 C++协程的灵活性，以及它们如何实现效率。无栈协程与状态机密切相关。通过将传统实现的状态机重写为使用协程的代码，我们探索了这种关系，您看到编译器如何将我们的协程转换和优化为机器语言。

在下一章中，我们将继续讨论协程，重点放在异步编程上，并加深您对`co_await`关键字的理解。
