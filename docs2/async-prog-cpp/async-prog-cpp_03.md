

# 如何在 C++中创建和管理线程

正如我们在前两章所学，线程是程序中执行的最小且最轻量级的单元。每个线程负责由操作系统调度器在分配的 CPU 资源上运行的指令序列定义的唯一任务。当管理程序中的并发性以最大化 CPU 资源利用率时，线程发挥着关键作用。

在程序的启动过程中，在内核将执行权传递给进程之后，C++运行时创建主线程并执行**main()**函数。之后，可以创建额外的线程来将程序分割成不同的任务，这些任务可以并发运行并共享资源。这样，程序可以处理多个任务，提高效率和响应速度。

在本章中，我们将学习如何使用现代 C++特性创建和管理线程的基础知识。在随后的章节中，我们将遇到关于 C++锁同步原语（互斥锁、信号量、屏障和自旋锁）、无锁同步原语（原子变量）、协调同步原语（条件变量）以及使用 C++解决或避免并发或多线程使用时潜在问题的方法的解释（竞争条件或数据竞争、死锁、活锁、饥饿、过载订阅、负载均衡和线程耗尽）。

在本章中，我们将介绍以下主要主题：

+   如何在 C++中创建、管理和取消线程

+   如何向线程传递参数并从线程获取结果

+   如何让线程休眠或让其他线程执行

+   jthread 对象是什么以及为什么它们有用

# 技术要求

在本章中，我们将使用 C++11 和 C++20 开发不同的解决方案。因此，我们需要安装**GNU 编译器集合**（**GCC**），特别是 GCC 13，以及 Clang 8（有关 C++编译器支持的详细信息，请参阅[`en.cppreference.com/w/cpp/compiler_support`](https://en.cppreference.com/w/cpp/compiler_support)）。

你可以在[`gcc.gnu.org`](https://gcc.gnu.org)找到更多关于 GCC 的信息。你可以在[`gcc.gnu.org/install/index.html`](https://gcc.gnu.org/install/index.html)找到有关如何安装 GCC 的信息。

想了解更多关于支持包括 C++在内的多种语言的编译器前端 Clang 的信息，请访问[`clang.llvm.org`](https://clang.llvm.org)。Clang 是 LLVM 编译器基础设施项目的一部分（[`llvm.org`](https://llvm.org)）。Clang 中的 C++支持在此处文档化：[`clang.llvm.org/cxx_status.html`](https://clang.llvm.org/cxx_status.html)。

在这本书中，一些代码片段没有显示包含的库。此外，一些函数，即使是主要的函数，也可能被简化，只显示相关的指令。你可以在以下 GitHub 仓库中找到所有完整的代码：[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)。

在前一个 GitHub 仓库的根目录下的 **scripts** 文件夹中，你可以找到一个名为 **install_compilers.sh** 的脚本，这个脚本可能有助于在基于 Debian 的 Linux 系统中安装所需的编译器。该脚本已在 Ubuntu 22.04 和 24.04 上进行了测试。

本章的示例位于 **Chapter_03** 文件夹下。所有源代码文件都可以使用 C++20 和 CMake 编译，如下所示：

```cpp
cmake . && cmake —build .
```

可执行文件将在 **bin** 目录下生成。

# 线程库——简介

在 C++ 中创建和管理线程的主要库是线程库。首先，让我们回顾一下线程。然后我们将深入了解线程库提供了什么。

## 什么是线程？让我们回顾一下

线程的目的是在一个进程中执行多个同时任务。

正如我们在前一章中看到的，线程有自己的堆栈、局部数据和 CPU 寄存器，如 **指令指针** ( **IP** ) 和 **堆栈指针** ( **SP** )，但与父进程共享地址空间和虚拟内存。

在用户空间中，我们可以区分 **原生线程** 和 **轻量级或虚拟线程**。原生线程是在使用某些内核 API 时由操作系统创建的。C++ 线程对象创建和管理这些类型的线程。另一方面，轻量级线程类似于原生线程，但它们是由运行时或库模拟的。如前一章所述，轻量级线程比原生线程具有更快的上下文切换。此外，多个轻量级线程可以在同一个原生线程中运行，并且可以比原生线程小得多。

在本章中，我们将开始学习原生线程。在 *第八章* 中，我们将学习以协程形式存在的轻量级线程。

## C++ 线程库

在 C++ 中，线程允许多个函数并发运行。**线程**类定义了一个对原生线程的类型安全接口。这个类在 **std::thread** 库中定义，位于 **Standard Template Library** ( **STL** ) 的 **<thread>** 头文件中。从 C++11 开始，它就可用。

在 C++ STL 中包含线程库之前，开发者使用特定平台的库，例如 Unix 或 Linux 操作系统中的 POSIX 线程（**pthread**）库，Windows NT 和 CE 系统的 **C 运行时**（**CRT**）和 Win32 库，或者第三方库如 **Boost.Threads**。在这本书中，我们将只使用现代 C++ 功能。由于 **<thread>** 可用并提供在特定平台机制之上的可移植抽象，因此不会使用或解释这些库。在 *第九章* 中，我们将介绍 **Boost.Asio**，在 *第十章* 中，**Boost.Cobalt**。这两个库都提供了处理异步 I/O 操作和协程的高级框架。

现在是时候学习不同的线程操作了。

# 线程操作

在本节中，我们将学习如何创建线程，在它们的构造过程中传递参数，从线程返回值，取消线程执行，捕获异常，以及更多。

## 线程创建

当创建一个线程时，它将立即执行。它只是被操作系统调度过程所延迟。如果没有足够的资源并行运行父线程和子线程，它们运行的顺序是不确定的。

构造函数参数定义了线程将要执行的功能或**函数**对象。这个可调用对象不应该返回任何内容，因为它的返回值将被忽略。如果由于某种原因线程执行以异常结束，除非捕获到异常，否则将调用 **std::terminate**，正如我们将在本章后面看到的那样。

在以下示例中，我们使用不同的可调用对象创建了六个线程。

**t1** 使用函数指针：

```cpp
void func() {
    std::cout << "Using function pointer\n";
}
std::thread t1(func);
```

**t2** 使用 lambda 函数：

```cpp
auto lambda_func = []() {
    std::cout << "Using lambda function\n";
};
std::thread t2(lambda_func);
```

**t3** 使用内嵌的 lambda 函数：

```cpp
std::thread t3([]() {
    std::cout << "Using embedded lambda function\n";
});
```

**t4** 使用**函数**对象，其中**operator()**被重载：

```cpp
class FuncObjectClass {
   public:
    void operator()() {
        std::cout << "Using function object class\n";
    }
};
std::thread t4{FuncObjectClass()};
```

**t5** 通过传递**成员**函数的地址和对象的地址来使用非静态**成员**函数调用**成员**函数：

```cpp
class Obj {
  public:
    void func() {
        std::cout << "Using a non-static member function"
                  << std::endl;
    }
};
Obj obj;
std::thread t5(&Obj::func, &obj);
```

**t6** 使用静态**成员**函数，其中只需要**成员**函数的地址，因为方法是静态的：

```cpp
class Obj {
  public:
    static void static_func() {
        std::cout << "Using a static member function\n";
    }
};
std::thread t6(&Obj::static_func);
```

线程创建会产生一些开销，可以通过使用线程池来减少，正如我们将在 *第四章* 中探讨的那样。

### 检查硬件并发

有效线程管理的一种策略，这与可扩展性和性能相关，并在上一章中进行了评论，是平衡线程数与可用资源以避免过度订阅。

要检索操作系统支持的并发线程数，我们可以使用 **std::thread::hardware_concurrency()** 函数：

```cpp
const auto processor_count = std::thread::hardware_concurrency();
```

此函数返回的值必须被视为仅提供有关将要并发运行的线程数的提示。它有时也不太明确，因此返回值为 **0**。

## 同步流写入

当我们使用来自两个或更多线程的**std::cout**向控制台打印消息时，输出结果可能会很混乱。这是由于输出流中发生的**竞态条件**造成的。

如前一章所述，竞态条件是并发和多线程程序中的软件错误，其行为取决于在共享资源上发生的事件序列，其中至少有一个操作不是原子的。我们将在*第四章*中了解更多如何避免它们。此外，我们还将学习如何使用 Clang 的 sanitizers 在*第十二章*中调试竞态条件。

以下代码片段显示了两个线程打印一系列数字。**t1**线程应打印包含**1 2 3 4**序列的行。**t2**线程应打印**5 6 7 8**序列。每个线程打印其序列 100 次。在主线程退出之前，它使用**join()**等待**t1**和**t2**完成。

本章后面将详细介绍如何加入线程。

```cpp
#include <iostream>
#include <thread>
int main() {
    std::thread t1([]() {
        for (int i = 0; i < 100; ++i) {
            std::cout << "1 " << "2 " << "3 " << "4 "
                      << std::endl;
        }
    });
    std::thread t2([]() {
        for (int i = 0; i < 100; ++i) {
            std::cout << "5 " << "6 " << "7 " << "8 "
                      << std::endl;
        }
    });
    t1.join();
    t2.join();
    return 0;
}
```

然而，运行前面的示例显示了一些包含以下内容的行：

```cpp
6 1 2 3 4
1 5 2 6 3 4 7 8
1 2 3 5 6 7 8
```

为了避免这些问题，我们可以简单地从一个特定的线程写入，或者使用一个**std::ostringstream**对象，该对象对**std::cout**对象进行原子调用：

```cpp
std::ostringstream oss;
oss << "1 " << "2 " << "3 " << "4 " << "\n";
std::cout << oss.str();
```

从 C++20 开始，我们还可以使用**std::osyncstream**对象。它们的行为类似于**std::cout**，但在访问同一流的线程之间具有写入同步。然而，由于只有从其内部缓冲区到输出流的传输步骤是同步的，因此每个线程都需要自己的**std::osyncstream**实例。

当流被销毁时，内部缓冲区会被转移，这是在显式调用**emit()**时发生的。

以下是一个简单的解决方案，允许在每行打印上进行同步：

```cpp
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
int main() {
    std::thread t1([]() {
        for (int i = 0; i < 100; ++i) {
            sync_cout << "1 " << "2 " << "3 " << "4 "
                      << std::endl;
        }
    });
    std::thread t2([]() {
        for (int i = 0; i < 100; ++i) {
            sync_cout << "5 " << "6 " << "7 " << "8 "
                      << std::endl;
        }
    });
    t1.join();
    t2.join();
    return 0;
}
```

这两种解决方案都将输出序列，而不进行交错。

```cpp
1 2 3 4
1 2 3 4
5 6 7 8
```

由于这种方法现在是官方 C++20 避免输出内容时发生竞态条件的官方方法，因此我们将使用**std::osyncstream**作为本书其余部分默认的方法。

## 使当前线程休眠

**std::this_thread**是一个命名空间。它提供了从当前线程访问函数以将执行权交予另一个线程或阻塞当前任务的执行并等待一段时间的功能。

**std::this_thread::sleep_for**和**std::this_thread::sleep_until**函数会阻塞线程的执行给定的时间长度。

**std::this_thread::sleep_for**至少休眠给定的时间长度。阻塞时间可能更长，这取决于操作系统调度器如何决定运行任务，或者由于某些资源争用延迟。

资源争用

当对某个共享资源的需求超过供应时，就会发生资源争用，导致性能下降。

**std::this_thread::sleep_until**与**std::this_thread::sleep_for**类似。然而，它不是睡眠一段时间，而是睡眠直到达到特定的时间点。计算时间点所使用的时钟必须满足**Clock**要求（你可以在这里找到更多信息：[`en.cppreference.com/w/cpp/named_req/Clock`](https://en.cppreference.com/w/cpp/named_req/Clock)）。标准建议使用稳定的时钟而不是系统时钟来设置持续时间。

## 线程识别

在调试多线程解决方案时，知道哪个线程正在执行给定的函数是有用的。每个线程都可以通过一个标识符来识别，这使得可以记录其值以进行跟踪和调试。

**std::thread::id**是一个轻量级类，它定义了线程对象（**std::thread**和**std::jthread**，我们将在本章后面介绍）的唯一标识符。该标识符通过使用**get_id()**函数检索。

线程标识符对象可以通过输出流进行比较、序列化和打印。它们也可以用作映射容器中的键，因为它们支持**std::hash**函数。

以下示例打印了**t**线程的标识符。在本章后面，我们将学习如何创建线程并睡眠一段时间：

```cpp
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
void func() {
    std::this_thread::sleep_for(1s);
}
int main() {
    std::thread t(func);
    std::cout << "Thread ID: " << t.get_id() << std::endl;
    t.join();
    return 0;
}
```

记住，当一个线程完成时，其标识符可以被未来的线程重用。

## 传递参数

可以通过值、引用或指针将参数传递给线程。

在这里我们可以看到如何通过值传递参数：

```cpp
void funcByValue(const std::string& str, int val) {
    sync_cout << «str: « << str << «, val: « << val
              << std::endl;
}
std::string str{"Passing by value"};
std::thread t(funcByValue, str, 1);
```

通过值传递可以避免数据竞争。然而，它的成本要高得多，因为数据需要复制。

下一个示例显示了如何通过引用传递值：

```cpp
void modifyValues(std::string& str, int& val) {
    str += " (Thread)";
    val++;
}
std::string str{"Passing by reference"};
int val = 1;
std::thread t(modifyValues, std::ref(str), std::ref(val));
```

或者作为**const-reference**：

```cpp
void printVector(const std::vector<int>& v) {
    sync_cout << "Vector: ";
    for (int num : v) {
        sync_cout << num << " ";
    }
    sync_cout << std::endl;
}
std::vector<int> v{1, 2, 3, 4, 5};
std::thread t(printVector, std::cref(v));
```

通过引用传递是通过使用**ref()**（非 const 引用）或**cref()**（const 引用）实现的。这两个都在**<functional>**头文件中定义。这允许变长模板定义线程构造函数，将参数作为引用处理。

这些辅助函数用于生成**std::reference_wrapper**对象，这些对象将引用包装在可复制和可赋值的对象中。在传递参数时缺少这些函数会使参数以值的方式传递。

你也可以按照以下方式将对象移动到线程中：

```cpp
std::thread t(printVector, std::move(v));
```

然而，请注意，在将**v**向量移动到**t**线程后，在主线程中尝试访问它会导致未定义的行为。

最后，我们还可以允许线程通过 lambda 捕获访问变量：

```cpp
std::string str{"Hello"};
std::thread t([&]() {
    sync_cout << "str: " << str << std::endl;
});
```

在这个例子中，**str**变量是通过嵌入的 lambda 函数捕获的引用被**t**线程访问的。

## 返回值

要返回线程中计算出的值，我们可以使用带有同步机制（如互斥锁、锁或原子变量）的共享变量。

在下面的代码片段中，我们可以看到如何通过使用非 const 引用传递的参数来返回线程计算出的值（使用**ref()**）。在**func**函数中，**result**变量在**t**线程中被计算。从主线程中可以看到结果值。正如我们将在下一节中学习的，**join()**函数只是等待**t**线程完成，然后让主线程继续运行，并在之后检查**result**变量：

```cpp
#include <chrono>
#include <iostream>
#include <random>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
namespace {
int result = 0;
};
void func(int& result) {
    std::this_thread::sleep_for(1s);
    result = 1 + (rand () % 10);
}
Int main() {
    std::thread t(func, std::ref(result));
    t.join();
    sync_cout << "Result: " << result << std::endl;
}
```

**reference** 参数可以是输入对象的引用，或者是我们想要存储结果的另一个变量，就像在这个示例中用**result**变量所做的那样。

我们也可以使用 lambda 捕获来返回值，如下面的示例所示：

```cpp
std::thread t([&]() { func(result); });
t.join();
sync_cout << "Result: " << result << std::endl;
```

我们也可以通过写入由互斥锁保护的共享变量来实现这一点，在执行写入操作之前锁定互斥锁（例如使用**std::lock_guard**）。然而，我们将在*第四章*中更深入地探讨这些机制：

```cpp
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
namespace {
int result = 0;
std::mutex mtx;
};
void funcWithMutex() {
    std::this_thread::sleep_for(1s);
    int localVar = 1 + (rand() % 10);
    std::lock_guard<std::mutex> lock(mtx);
    result = localVar;
}
Int main() {
    std::thread t(funcWithMutex);
    t.join();
    sync_cout << "Result: " << result << std::endl;
}
```

从线程返回值有更优雅的方法。这涉及到使用 future 和 promise，我们将在*第六章*中学习。

## 移动线程

线程可以移动但不能复制。这是为了避免有两个不同的线程对象来表示相同的硬件线程。

在下面的示例中，**t1**使用**std::move**移动到**t2**。因此，**t2**继承了**t1**移动之前的相同标识符，而**t1**不再可连接，因为它不再包含任何有效的线程：

```cpp
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
void func() {
    for (auto i=0; i<10; ++i) {
        std::this_thread::sleep_for(500ms);
    }
}
int main() {
    std::thread t1(func);
    std::thread t2 = std::move(t1);
    t2.join();
    return 0;
}
```

当一个**std::thread**对象被移动到另一个**std::thread**对象时，移动线程对象将达到一个不再表示真实线程的状态。这种情况也发生在分离或连接后由默认构造函数生成的线程对象上。我们将在下一节中介绍这些操作。

## 等待线程完成

有一些用例需要线程等待另一个线程完成，以便它可以使用后者线程计算出的结果。其他用例包括在后台运行线程，将其分离，并继续执行主线程。

### 连接线程

**join()** 函数在等待由调用**join()**函数的线程对象指定的连接线程完成时阻塞当前线程。这确保了在**join()**返回后连接线程已经终止（有关更多详细信息，请参阅*第二章*中的*线程生命周期*部分）。

很容易忘记使用**join()**函数。**Joining Thread**（**jthread**）解决了这个问题。它从 C++20 开始可用。我们将在下一节中介绍它。

### 检查线程是否可连接

如果在某个线程中没有调用 **join()** 函数，则该线程被认为是可连接的并且是活跃的。即使线程已经执行了代码但尚未连接，这也是正确的。另一方面，默认构造的线程或已经连接的线程是不可连接的。

要检查线程是否可连接，只需使用 **std::thread::joinable()** 函数。

让我们看看以下示例中 **std::thread::join()** 和 **std::thread::joinable()** 的用法：

```cpp
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
void func() {
    std::this_thread::sleep_for(100ms);
}
int main() {
    std::thread t1;
    std::cout << "Is t1 joinable? " << t1.joinable()
              << std::endl;
    std::thread t2(func);
    t1.swap(t2);
    std::cout << "Is t1 joinable? " << t1.joinable()
              << std::endl;
    std::cout << "Is t2 joinable? " << t2.joinable()
              << std::endl;
    t1.join();
    std::cout << "Is t1 joinable? " << t1.joinable()
              << std::endl;
}
```

使用默认构造函数（未指定可调用对象）构造 **t1** 后，该线程将不可连接。由于 **t2** 构造时指定了函数，**t2** 在构造后是可连接的。然而，当 **t1** 和 **t2** 交换时，**t1** 再次变为可连接的，而 **t2** 则不再可连接。然后主线程等待 **t1** 连接，因此它不再可连接。尝试连接一个不可连接的线程 **t2** 将导致未定义的行为。最后，不连接一个可连接的线程将导致资源泄漏或由于共享资源的意外使用而可能引发程序崩溃。

### 通过分离实现守护线程

如果我们希望一个线程作为守护线程在后台继续运行，但完成当前线程的执行，我们可以使用 **std::thread::detach()** 函数。守护线程是在后台执行一些不需要运行到完成的任务的线程。如果主程序退出，所有守护线程都将被终止。如前所述，线程必须在主线程终止之前连接或分离，否则程序将中止执行。

在调用 **detach** 之后，分离的线程无法通过 **std::thread** 对象进行控制或连接（因为它正在等待其完成），因为这个对象不再代表分离的线程。

以下示例展示了一个名为 **t** 的守护线程，它在构造后立即分离，在后台运行 **daemonThread()** 函数。这个函数执行三秒钟后退出，完成线程执行。同时，主线程在退出前比线程执行时间多睡一秒钟：

```cpp
#include <chrono>
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
namespace {
int timeout = 3;
}
void daemonThread() {
    sync_cout << "Daemon thread starting...\n";
    while (timeout-- > 0) {
        sync_cout << "Daemon thread is running...\n";
        std::this_thread::sleep_for(1s);
    }
    sync_cout << "Daemon thread exiting...\n";
}
int main() {
    std::thread t(daemonThread);
    t.detach();
    std::this_thread::sleep_for(
              std::chrono::seconds(timeout + 1));
    sync_cout << "Main thread exiting...\n";
    Return 0;
}
```

## 线程连接 – jthread 类

从 C++20 开始，有一个新的类：**std::jthread**。这个类类似于 **std::thread**，但增加了额外的功能，即线程在析构时自动重新连接，遵循 **资源获取即初始化**（**RAII**）技术。在某些情况下，它可以被取消或停止。

如以下示例所示，**jthread** 线程具有与 **std::thread** 相同的接口。唯一的区别是我们不需要调用 **join()** 函数来确保主线程等待 **t** 线程连接：

```cpp
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
void func() {
    std::this_thread::sleep_for(1s);
}
int main() {
    std::jthread t(func);
    sync_cout << "Thread ID: " << t.get_id() << std::endl;
    return 0;
}
```

当两个**std::jthread**被销毁时，它们的析构函数按照从构造函数相反的顺序被调用。为了演示这种行为，让我们实现一个线程包装类，当包装的线程被创建和销毁时打印一些消息：

```cpp
#include <chrono>
#include <functional>
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
class JthreadWrapper {
   public:
    JthreadWrapper(
       const std::function<void(const std::string&)>& func,
       const std::string& str)
        : t(func, str), name(str) {
        sync_cout << "Thread " << name
                  << " being created" << std::endl;
    }
    ~JthreadWrapper() {
        sync_cout << "Thread " << name
                  << " being destroyed" << std::endl;
    }
   private:
    std::jthread t;
    std::string name;
};
```

使用这个**JthreadWrapper**包装类，我们启动了三个线程来执行**func**函数。每个线程将在退出前等待一秒钟：

```cpp
void func(const std::string& name) {
    sync_cout << "Thread " << name << " starting...\n";
    std::this_thread::sleep_for(1s);
    sync_cout << "Thread " << name << " finishing...\n";
}
int main() {
    JthreadWrapper t1(func, «t1»);
    JthreadWrapper t2(func, "t2");
    JthreadWrapper t3(func, "t3");
    std::this_thread::sleep_for(2s);
    sync_cout << "Main thread exiting..." << std::endl;
    return 0;
}
```

此程序将显示以下输出：

```cpp
Thread t1 being created
Thread t1 starting...
Thread t2 being created
Thread t2 starting...
Thread t3 being created
Thread t3 starting...
Thread t1 finishing...
Thread t2 finishing...
Thread t3 finishing...
Main thread exiting...
Thread t3 being destroyed
Thread t2 being destroyed
Thread t1 being destroyed
```

如我们所见，**t1**首先被创建，然后是**t2**，最后是**t3**。析构函数遵循相反的顺序，**t3**首先被销毁，然后是**t2**，最后是**t1**。

由于 jthreads 在忘记在线程中使用**join**时可以避免陷阱，我们更倾向于使用**std::jthread**而不是**std::thread**。可能会有一些情况，我们需要显式调用**join()**来确保在移动到另一个任务之前线程已经连接并且资源已经适当释放。

## 让出线程执行

线程也可以决定暂停其执行，让实现重新调度线程的执行，并给其他线程运行的机会。

**std::this_thread::yield**方法向操作系统提供提示以重新调度另一个线程。其行为依赖于实现，取决于操作系统调度程序和系统的当前状态。

一些 Linux 实现会挂起当前线程并将其移回一个线程队列以调度具有相同优先级的所有线程。如果这个队列是空的，则让出（yield）没有效果。

以下示例显示了两个线程，**t1**和**t2**，执行相同的工作函数。它们随机选择要么做一些工作（锁定互斥锁，我们将在下一章中了解）或让出执行权给另一个线程：

```cpp
#include <iostream>
#include <random>
#include <string>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono;
namespace {
int val = 0;
std::mutex mtx;
}
int main() {
    auto work = & {
        while (true) {
            bool work_to_do = rand() % 2;
            if (work_to_do) {
                sync_cout << name << ": working\n";
                std::lock_guard<std::mutex> lock(mtx);
                for (auto start = steady_clock::now(),
                          now = start;
                          now < start + 3s;
                          now = steady_clock::now()) {
                }
            } else {
                sync_cout << name << ": yielding\n";
                std::this_thread::yield();
            }
        }
    };
    std::jthread t1(work, "t1");
    std::jthread t2(work, "t2");
    return 0;
}
```

当运行此示例时，当执行达到让出（yield）命令时，我们可以看到当前运行的线程如何停止并允许其他线程重新启动其执行。

## 线程取消

如果我们不再对线程正在计算的结果感兴趣，我们希望取消该线程并避免更多的计算成本。

杀死线程可能是一个解决方案。然而，这会留下属于线程处理者的资源，例如从该线程启动的其他线程、锁、连接等。这可能导致程序以未定义的行为结束，在互斥锁下锁定关键部分，或任何其他意外问题。

为了避免这些问题，我们需要一个无数据竞争的机制，让线程知道停止执行（请求停止）的意图，以便线程可以采取所有必要的具体步骤来取消其工作并优雅地终止。

实现这一目标的一种可能方式是使用原子变量，该变量由线程定期检查。我们将在下一章详细探讨原子变量。现在，让我们将原子变量定义为一个变量，许多线程可以无任何锁定机制或由于原子事务操作和内存模型导致的数据竞争来读写它。

作为示例，让我们创建一个**Counter**类，该类每秒调用一次回调。这是无限期进行的，直到调用者使用**stop()**函数将**running**原子变量设置为**false**：

```cpp
#include <chrono>
#include <functional>
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
class Counter {
    using Callback = std::function<void(void)>;
   public:
    Counter(const Callback &callback) {
        t = std::jthread([&]() {
            while (running.load() == true) {
                callback ();
                std::this_thread::sleep_for(1s);
            }
        });
    }
    void stop() { running.store(false); }
   private:
    std::jthread t;
    std::atomic_bool running{true};
};
```

在调用函数中，我们将如下实例化**Counter**。然后，在需要的时候（这里是在三秒后），我们将调用**stop()**函数，让**Counter**退出循环并终止线程执行：

```cpp
int main() {
    Counter counter([&]() {
        sync_cout << "Callback: Running...\n";
    });
    std::this_thread::sleep_for(3s);
    counter.stop();
}
```

自 C++20 以来，出现了一种新的线程协作中断机制。这可以通过**std::stop_token**来实现。

线程通过调用**std::stop_token::stop_requested()**函数的结果来检查是否请求了停止。

要生成**stop_token**，我们将通过**stop_source**对象使用**std::stop_source::get_token()**函数。

这种线程取消机制是通过**std::jthead**对象中**std::stop_source**类型的内部成员实现的，其中存储了共享的停止状态。**jthread**构造函数接受**std::stop_token**作为其第一个参数。这用于在执行期间请求停止。

因此，与**std::thread**对象相比，**std::jthread**暴露了一些额外的函数来管理停止令牌。这些函数是**get_stop_source()**、**get_stop_token()**和**request_stop()**。

当调用**request_stop()**时，它向内部停止状态发出停止请求，该状态原子更新以避免竞争条件（你将在*第四章*中了解更多关于原子变量的内容）。

让我们检查以下示例中所有这些函数是如何工作的。

首先，我们将定义一个模板函数来展示停止项对象（**stop_token**或**stop_source**）的属性：

```cpp
#include <chrono>
#include <iostream>
#include <string_view>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
template <typename T>
void show_stop_props(std::string_view name,
                     const T& stop_item) {
    sync_cout << std::boolalpha
              << name
              << ": stop_possible = "
              << stop_item.stop_possible()
              << ", stop_requested = "
              << stop_item.stop_requested()
              << '\n';
};
```

现在，在**main()**函数中，我们将启动一个工作线程，获取其停止令牌对象，并展示其属性：

```cpp
auto worker1 = std::jthread(func_with_stop_token);
std::stop_token stop_token = worker1.get_stop_token();
show_stop_props("stop_token", stop_token);
```

**Worker1**正在运行定义在下述代码块中的**func_with_stop_token()**函数。在这个函数中，通过使用**stop_requested()**函数来检查停止令牌。如果这个函数返回**true**，则表示请求了停止，因此函数简单地返回，终止线程执行。否则，它将运行下一个循环迭代，使当前线程休眠 300 毫秒，直到下一次停止请求检查：

```cpp
void func_with_stop_token(std::stop_token stop_token) {
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(300ms);
        if (stop_token.stop_requested()) {
            sync_cout << "stop_worker: "
                      << "Stopping as requested\n";
            return;
        }
        sync_cout << "stop_worker: Going back to sleep\n";
    }
}
```

我们可以通过使用线程对象返回的停止令牌来从主线程请求停止，如下所示：

```cpp
worker1.request_stop();
worker1.join();
show_stop_props("stop_token after request", stop_token);
```

此外，我们还可以从不同的线程请求停止。为此，我们需要传递一个**stop_source**对象。在下面的代码片段中，我们可以看到如何使用从**worker2**工作线程获取的**stop_source**对象创建一个线程停止器：

```cpp
auto worker2 = std::jthread(func_with_stop_token);
std::stop_source stop_source = worker2.get_stop_source();
show_stop_props("stop_source", stop_source);
auto stopper = std::thread( [](std::stop_source source) {
        std::this_thread::sleep_for(500ms);
        sync_cout << "Request stop for worker2 "
                  << "via source\n";
        source.request_stop();
    }, stop_source);
stopper.join();
std::this_thread::sleep_for(200ms);
show_stop_props("stop_source after request", stop_source);
```

**stopper**线程等待 0.5 秒，然后从**stop_source**对象请求停止。然后**worker2**意识到这个请求，并终止其执行，如前所述。

我们还可以注册一个回调函数，当通过停止令牌或停止源请求停止时，将调用该函数。这可以通过使用**std::stop_callback**对象来实现，如下面的代码块所示：

```cpp
std::stop_callback callback(worker1.get_stop_token(), []{
    sync_cout << "stop_callback for worker1 "
              << "executed by thread "
              << std::this_thread::get_id() << '\n';
});
sync_cout << "main_thread: "
          << std::this_thread::get_id() << '\n';
std::stop_callback callback_after_stop(
    worker2.get_stop_token(),[] {
        sync_cout << "stop_callback for worker2 "
                  << "executed by thread "
                  << std::this_thread::get_id() << '\n';
});
```

如果销毁了**std::stop_callback**对象，将阻止其执行。例如，这个作用域内的停止回调将不会执行，因为回调对象在超出作用域时被销毁：

```cpp
{
    std::stop_callback scoped_callback(
        worker2.get_stop_token(), []{
          sync_cout << "Scoped stop callback "
                    << "will not execute\n";
      }
    );
}
```

在已经请求停止之后，新的停止回调对象将立即执行。在以下示例中，如果已经为**worker2**请求了停止，**callback_after_stop**将在构造后立即执行 lambda 函数：

```cpp
sync_cout << "main_thread: "
          << std::this_thread::get_id() << '\n';
std::stop_callback callback_after_stop(
    worker2.get_stop_token(), []{
        sync_cout << "stop_callback for worker2 "
                  << "executed by thread "
                  << std::this_thread::get_id() << '\n';
    }
);
```

## 捕获异常

在线程内部抛出的任何未处理的异常都需要在该线程内部捕获。否则，C++运行时会调用**std::terminate**，导致程序突然终止。这会导致意外的行为、数据丢失，甚至程序崩溃。

一种解决方案是在线程内部使用 try-catch 块来捕获异常。然而，只有在该线程内部抛出的异常才会被捕获。异常不会传播到其他线程。

要将异常传播到另一个线程，一个线程可以捕获它并将其存储到**std::exception_ptr**对象中，然后使用共享内存技术将其传递到另一个线程，在那里将检查**std::exception_ptr**对象并在需要时重新抛出异常。

以下示例展示了这种方法：

```cpp
#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>
using namespace std::chrono_literals;
std::exception_ptr captured_exception;
std::mutex mtx;
void func() {
    try {
        std::this_thread::sleep_for(1s);
        throw std::runtime_error(
                  "Error in func used within thread");
    } catch (...) {
        std::lock_guard<std::mutex> lock(mtx);
        captured_exception = std::current_exception();
    }
}
int main() {
    std::thread t(func);
    while (!captured_exception) {
        std::this_thread::sleep_for(250ms);
        std::cout << „In main thread\n";
    }
    try {
        std::rethrow_exception(captured_exception);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught in main thread: "
                  << e.what() << std::endl;
    }
    t.join();
}
```

在这里，我们可以看到当**t**线程执行**func**函数时抛出的**std::runtime_error**异常。异常被捕获并存储在**captured_exception**中，这是一个由互斥锁保护的**std::exception_ptr**共享对象。抛出异常的类型和值是通过调用**std::current_exception()**函数确定的。

在主线程中，**while**循环会一直运行，直到捕获到异常。通过调用**std::rethrow_exception(captured_exception)**在主线程中重新抛出异常。异常再次被主线程捕获，在**catch**块中执行，通过**std::cerr**错误流向控制台打印消息。

我们将在*第六章*中学习一个更好的解决方案，通过使用 future 和 promise。

# 线程局部存储

**线程局部存储**（**TLS**）是一种内存管理技术，允许每个线程拥有自己的变量实例。这种技术允许线程存储其他线程无法访问的线程特定数据，避免竞态条件并提高性能。这是因为访问这些变量的同步机制的开销被消除了。

TLS 由操作系统实现，可以通过使用**thread_local**关键字访问，该关键字自 C++11 以来一直可用。**thread_local**提供了一种统一的方式来使用许多操作系统的 TLS 功能，并避免使用特定编译器的语言扩展来访问 TLS 功能（此类扩展的示例包括 TLS Windows API、**__declspec(thread)** MSVC 编译器语言扩展或**__thread** GCC 编译器语言扩展）。

要使用不支持 C++11 或更高版本的编译器的 TLS，请使用**Boost.Library**。这提供了**boost::thread_specific_ptr**容器，它实现了可移植的 TLS。

线程局部变量可以声明如下：

+   全局范围内

+   在命名空间中

+   作为类的静态成员变量

+   在函数内部；它具有与使用**static**关键字分配的变量相同的效果，这意味着变量在程序的生命周期内分配，其值在下一个函数调用中传递

以下示例展示了三个线程使用不同参数调用**multiplyByTwo**函数。此函数将**val**线程局部变量的值设置为参数值，将其乘以 2，并打印到控制台：

```cpp
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
thread_local int val = 0;
void setValue(int newval) { val = newval; }
void printValue() { sync_cout << val << ' '; }
void multiplyByTwo(int arg) {
    setValue(arg);
    val *= 2;
    printValue();
}
int main() {
    val = 1;  // Value in main thread
    std::thread t1(multiplyByTwo, 1);
    std::thread t2(multiplyByTwo, 2);
    std::thread t3(multiplyByTwo, 3);
    t1.join();
    t2.join();
    t3.join();
    std::cout << val << std::endl;
}
```

运行此代码片段将显示以下输出：

```cpp
2 4 6 1
```

在这里，我们可以看到每个线程都操作其输入参数，导致**t1**打印**2**，**t2**打印**4**，**t3**打印**6**。运行主函数的主线程也可以访问其线程局部变量**val**，该变量在程序开始时设置为**1**，但在主函数结束时打印到控制台之前仅用于打印，然后退出程序。

与任何技术一样，也有一些缺点。TLS 会增加内存使用量，因为每个线程都会创建一个变量，所以在资源受限的环境中可能会出现问题。此外，访问 TLS 变量可能比常规变量有一些开销。这在性能关键型软件中可能是一个问题。

使用我们迄今为止学到的许多技术，让我们构建一个计时器。

# 实现计时器

让我们实现一个接受间隔和回调函数的计时器。计时器将在每个间隔执行回调函数。此外，用户可以通过调用其**stop()**函数来停止计时器。

以下代码片段展示了计时器的实现：

```cpp
#include <chrono>
#include <functional>
#include <iostream>
#include <syncstream>
#include <thread>
#define sync_cout std::osyncstream(std::cout)
using namespace std::chrono_literals;
using namespace std::chrono;
template<typename Duration>
class Timer {
   public:
    typedef std::function<void(void)> Callback;
    Timer(const Duration interval,
          const Callback& callback) {
        auto value = duration_cast<milliseconds>(interval);
        sync_cout << "Timer: Starting with interval of "
                  << value << std::endl;
        t = std::jthread(& {
            while (!stop_token.stop_requested()) {
                sync_cout << "Timer: Running callback "
                          << val.load() << std::endl;
                val++;
                callback();
                sync_cout << "Timer: Sleeping...\n";
                std::this_thread::sleep_for(interval);
            }
            sync_cout << „Timer: Exit\n";
        });
    }
    void stop() {
        t.request_stop();
    }
   private:
    std::jthread t;
    std::atomic_int32_t val{0};
};
```

**Timer**构造函数接受一个**Callback**函数（一个**std::function<void(void)>**对象）和一个定义回调将执行的周期或间隔的**std::chrono::duration**对象。

然后使用 lambda 表达式创建一个 **std::jthread** 对象，其中循环以时间间隔调用回调函数。这个循环检查是否通过 **stop_token** 请求停止，这是通过使用 **stop()** **计时器** API 函数来实现的。当这种情况发生时，循环退出，线程终止。

这里是如何使用它的：

```cpp
int main(void) {
    sync_cout << "Main: Create timer\n";
    Timer timer(1s, [&]() {
        sync_cout << "Callback: Running...\n";
    });
    std::this_thread::sleep_for(3s);
    sync_cout << "Main thread: Stop timer\n";
    timer.stop();
    std::this_thread::sleep_for(500ms);
    sync_cout << "Main thread: Exit\n";
    return 0;
}
```

在这个示例中，我们启动了计时器，每秒将打印**回调：运行**消息。三秒后，主线程将调用**timer.stop()**函数，终止计时器线程。然后主线程等待 500 毫秒后退出。

这是输出结果：

```cpp
Main: Create timer
Timer: Starting with interval of 1000ms
Timer: Running callback 0
Callback: Running...
Timer: Sleeping...
Timer: Running callback 1
Callback: Running...
Timer: Sleeping...
Timer: Running callback 2
Callback: Running...
Timer: Sleeping...
Main thread: Stop timer
Timer: Exit
Main thread: Exit
```

作为练习，你可以稍微修改这个示例来实现一个超时类，如果给定超时间隔内没有输入事件，它会调用回调函数。在处理网络通信时，这是一个常见的模式，如果在一段时间内没有接收到数据包，则会发送数据包重放请求。

# 摘要

在这一章中，我们学习了如何创建和管理线程，如何传递参数或检索结果，如何工作 TLS，以及如何等待线程完成。我们还学习了如何使线程将控制权交给其他线程或取消其执行。如果出现问题并抛出异常，我们现在知道如何在线程之间传递异常并避免意外的程序终止。最后，我们实现了一个 **计时器** 类，该类定期运行回调函数。

在下一章中，我们将学习线程安全、互斥和原子操作。这包括互斥锁、锁定和无锁算法，以及内存同步排序，以及其他主题。这些知识将帮助我们开发线程安全的数组和算法。

# 进一步阅读

+   编译器支持：[`en.cppreference.com/w/cpp/compiler_support`](https://en.cppreference.com/w/cpp/compiler_support)

+   GCC 版本：[`gcc.gnu.org/releases.html`](https://gcc.gnu.org/releases.html)

+   Clang：[`clang.llvm.org`](https://clang.llvm.org)

+   Clang 8 文档：[`releases.llvm.org/8.0.0/tools/clang/docs/index.html`](https://releases.llvm.org/8.0.0/tools/clang/docs/index.html)

+   LLVM 项目：[`llvm.org`](https://llvm.org)

+   Boost.Threads：[`www.boost.org/doc/libs/1_78_0/doc/html/thread.html`](https://www.boost.org/doc/libs/1_78_0/doc/html/thread.html)

+   P0024 – 并行技术规范：[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0024r0.html`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0024r0.html)

+   TLS 提案：[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm)

+   *C++0X 的线程启动*：[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2184.html`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2184.html)

+   IBM 的 TLS: [`docs.oracle.com/cd/E19683-01/817-3677/chapter8-1/index.html`](https://docs.oracle.com/cd/E19683-01/817-3677/chapter8-1/index.html)

+   *线程专有的* 数据: [`www.ibm.com/docs/en/i/7.5?topic=techniques-data-that-is-private-thread`](https://www.ibm.com/docs/en/i/7.5?topic=techniques-data-that-is-private-thread)

+   **资源获取即初始化** ( **RAII** ): [`en.cppreference.com/w/cpp/language/raii`](https://en.cppreference.com/w/cpp/language/raii)

+   Bjarne Stroustrup, *C++之旅*，第三版，18.2 和 18.7。
