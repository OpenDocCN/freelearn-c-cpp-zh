# 第五章：并发和同步

在本章中，我们将学习如何正确处理 C++中的并发、同步和并行。在这里，您需要对 C++和 C++线程有一般的了解。本章很重要，因为在处理 C++时通常需要使用共享资源，如果没有正确实现线程安全，这些资源很容易变得损坏。我们将首先对`std::mutexes`进行广泛的概述，它提供了一种同步 C++线程的方法。然后我们将研究原子数据类型，它提供了另一种安全处理并行性的机制。

本章包含了演示如何处理 C++线程的不同场景的示例，包括处理`const &`、线程安全包装、阻塞与异步编程以及 C++ promises 和 futures。这是很重要的，因为在处理多个执行线程时，这些知识是至关重要的。

本章涵盖了以下示例：

+   使用互斥锁

+   使用原子数据类型

+   了解在多个线程的上下文中`const &` mutable 的含义

+   使类线程安全

+   同步包装器及其实现方法

+   阻塞操作与异步编程

+   使用 promises 和 futures

# 技术要求

要编译和运行本章中的示例，您必须具有管理权限的计算机运行 Ubuntu 18.04，并具有正常的互联网连接。在运行这些示例之前，您必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake
```

如果此软件安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

# 使用互斥锁

在本示例中，我们将学习为什么以及如何在 C++中使用互斥锁。在 C++中使用多个线程时，通常会建立线程之间共享的资源。正如我们将在本示例中演示的那样，尝试同时使用这些共享资源会导致可能损坏资源的竞争条件。

互斥锁（在 C++中写作`std::mutex`）是一个用于保护共享资源的对象，确保多个线程可以以受控的方式访问共享资源。这可以防止资源损坏。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本示例所需的正确工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试此示例：

1.  从新终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe01_example01
The answer is: 42
The answer is: 42
The answer is: 42
The
 answer is: 42
The answer is: 42
...

> ./recipe01_example02
The answer is: 42
The answer is: 42
The answer is: 42
The answer is: 42
The answer is: 42
...

> ./recipe01_example03
...

> ./recipe01_example04
The answer is: 42

> ./recipe01_example05
The answer is: 42
The answer is: 42
The answer is: 42
The answer is: 42
The answer is: 42
...

> ./recipe01_example06
The answer is: 42
The answer is: 42

> ./recipe01_example07

> ./recipe01_example08
lock acquired
lock failed
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例教授的课程的关系。

# 工作原理...

在本示例中，我们将学习如何使用`std::mutex`来保护共享资源，防止其损坏。首先，让我们首先回顾一下当多个线程同时访问资源时资源如何变得损坏：

```cpp
#include <thread>
#include <string>
#include <iostream>

void foo()
{
    static std::string msg{"The answer is: 42\n"};
    while(true) {
        for (const auto &c : msg) {
            std::clog << c;
        }
    }
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    // Never reached
    return 0;
}
```

执行时，我们得到以下输出：

![](img/01192c95-b3c1-4df5-a5a4-b94be4b18090.png)

在上面的示例中，我们创建了一个在无限循环中输出到`stdout`的函数。然后我们创建了两个线程，每个线程执行先前定义的函数。正如您所看到的，当两个线程同时执行时，结果输出变得损坏。这是因为当一个线程正在将其文本输出到`stdout`时，另一个线程同时输出到`stdout`，导致一个线程的输出与另一个线程的输出混合在一起。

要解决这个问题，我们必须确保一旦其中一个线程尝试将其文本输出到`stdout`，在另一个线程能够输出之前，它应该被允许完成输出。换句话说，每个线程必须轮流输出到`stdout`。当一个线程输出时，另一个线程必须等待轮到它。为了做到这一点，我们将利用一个`std::mutex`对象。

# std::mutex

互斥锁是一个用来保护共享资源的对象，以确保对共享资源的使用不会导致损坏。为了实现这一点，`std::mutex`有一个`lock()`函数和一个`unlock()`函数。`lock()`函数*获取*对共享资源的访问（有时称为临界区）。`unlock()`*释放*先前获取的访问。任何尝试在另一个线程已经执行`lock()`之后执行`lock()`函数的操作都将导致线程必须等待，直到执行`unlock()`函数为止。

`std::mutex`的实现取决于 CPU 的架构和操作系统；但是，一般来说，互斥锁可以用一个简单的整数来实现。如果整数为`0`，`lock()`函数将把整数设置为`1`并返回，这告诉互斥锁它已被获取。如果整数为`1`，意味着互斥锁已经被获取，`lock()`函数将等待（即阻塞），直到整数变为`0`，然后它将把整数设置为`1`并返回。如何实现这种等待取决于操作系统。例如，`wait()`函数可以循环直到整数变为`0`，这被称为**自旋锁**，或者它可以执行`sleep()`函数并等待一段时间，允许其他线程和进程在互斥锁被锁定时执行。释放函数总是将整数设置为`0`，这意味着互斥锁不再被获取。确保互斥锁正常工作的诀窍是确保使用原子操作读/写整数。如果使用非原子操作，整数本身将遭受与互斥锁试图防止的相同的共享资源损坏。

例如，考虑以下情况：

```cpp
#include <mutex>
#include <thread>
#include <string>
#include <iostream>

std::mutex m{};

void foo()
{
    static std::string msg{"The answer is: 42\n"};
    while(true) {
        m.lock();
        for (const auto &c : msg) {
            std::clog << c;
        }
        m.unlock();
    }
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    // Never reached
    return 0;
}
```

此示例运行时输出以下内容：

![](img/a74a32c8-e166-46cc-b84f-774e905f34cb.png)

在前面的例子中，我们创建了一个输出到`stdout`的相同函数。不同之处在于，在我们输出到`stdout`之前，我们通过执行`lock()`函数来获取`std::mutex`。一旦我们完成了对`stdout`的输出，我们通过执行`unlock()`函数来释放互斥锁。在`lock()`和`unlock()`函数之间的代码称为**临界区**。临界区中的任何代码只能由一个线程在任何给定时间执行，确保我们对`stdout`的使用不会变得损坏。

通过控制对共享资源的访问（例如使用互斥锁）来确保共享资源不会变得损坏称为**同步**。尽管大多数需要线程同步的情况并不复杂，但有些情况可能导致需要整个大学课程来覆盖的线程同步方案。因此，线程同步被认为是计算机科学中极其困难的范式，需要正确编程。

在本教程中，我们将涵盖其中一些情况。首先，让我们讨论一下**死锁**。当一个线程在调用`lock()`函数时进入无休止的等待状态时，就会发生死锁。死锁通常非常难以调试，是由于几个原因造成的，包括以下原因：

+   由于程序员错误或获取互斥锁的线程崩溃，导致线程从未调用`unlock()`

+   同一个线程在调用`unlock()`之前多次调用`lock()`函数

+   每个线程以不同的顺序锁定多个互斥锁

为了证明这一点，让我们看一下以下例子：

```cpp
#include <mutex>
#include <thread>

std::mutex m{};

void foo()
{
    m.lock();
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    // Never reached
    return 0;
}
```

在前面的例子中，我们创建了两个线程，它们都试图锁定互斥量，但从未调用`unlock()`。结果，第一个线程获取了互斥量，然后返回而没有释放它。当第二个线程尝试获取互斥量时，它被迫等待第一个线程执行`unlock()`，但第一个线程从未执行，导致死锁（即程序永远不会返回）。

在这个例子中，死锁很容易识别和纠正；然而，在现实场景中，识别死锁要复杂得多。让我们看下面的例子：

```cpp
#include <array>
#include <mutex>
#include <thread>
#include <string>
#include <iostream>

std::mutex m{};
std::array<int,6> numbers{4,8,15,16,23,42};

int foo(int index)
{
    m.lock();
    auto element = numbers.at(index);
    m.unlock();

    return element;
}

int main(void)
{
    std::cout << "The answer is: " << foo(5) << '\n';
    return 0;
}
```

在前面的例子中，我们编写了一个函数，根据索引返回数组中的元素。此外，我们获取了保护数组的互斥量，并在返回之前释放了互斥量。挑战在于我们必须在函数可以返回的地方`unlock()`互斥量，这不仅包括从函数返回的每种可能分支，还包括抛出异常的所有可能情况。在前面的例子中，如果提供的索引大于数组大小，`std::array`对象将抛出异常，导致函数在调用`unlock()`之前返回，如果另一个线程正在共享此数组，将导致死锁。

# std::lock_guard

C++提供了`std::lock_guard`对象来简化对`std::mutex`对象的使用，而不是在代码中到处使用`try`/`catch`块来防止死锁，这假设程序员甚至能够确定每种可能发生死锁的情况而不出错。

例如，考虑以下代码：

```cpp
#include <mutex>
#include <thread>
#include <iostream>

std::mutex m{};

void foo()
{
    static std::string msg{"The answer is: 42\n"};

    while(true) {
        std::lock_guard lock(m);
        for (const auto &c : msg) {
            std::clog << c;
        }
    }
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    // Never reached
    return 0;
}
```

执行时，我们看到以下结果：

![](img/948ff65e-fbca-4f22-98f2-85b33fe28cea.png)

如前面的例子所示，当我们通常在互斥量上调用`lock()`时，使用`std::lock_guard`。`std::lock_guard`在创建时调用互斥量的`lock()`函数，然后在销毁时调用互斥量的`unlock()`函数（一种称为**资源获取即初始化**或**RAII**的习惯用法）。无论函数如何返回（无论是正常返回还是异常），互斥量都将被释放，确保死锁不可能发生，避免程序员必须准确确定函数可能返回的每种可能情况。

尽管`std::lock_guard`能够防止在从未调用`unlock()`的情况下发生死锁，但它无法防止在调用`lock()`多次之后再调用`unlock()`之前发生死锁的情况。为了处理这种情况，C++提供了`std::recursive_mutex`。

# std::recursive_mutex

递归互斥量每次同一线程调用`lock()`函数时都会增加互斥量内部存储的整数，而不会导致`lock()`函数等待。例如，如果互斥量被释放（即，互斥量中的整数为`0`），当线程`#1`调用`lock()`函数时，互斥量中的整数被设置为`1`。通常情况下，如果线程`#1`再次调用`lock()`函数，`lock()`函数会看到整数为`1`并进入等待状态，直到整数被设置为`0`。相反，递归互斥量将确定调用`lock()`函数的线程，并且如果获取互斥量的线程与调用`lock()`函数的线程相同，则使用原子操作再次增加互斥量中的整数（现在结果为`2`）。要释放互斥量，线程必须调用`unlock()`，这将使用原子操作递减整数，直到互斥量中的整数为`0`。

递归互斥锁允许同一个线程调用`lock()`函数多次，防止多次调用`lock()`函数并导致死锁，但代价是`lock()`和`unlock()`函数必须包括一个额外的函数调用来获取线程的`id()`实例，以便互斥锁可以确定是哪个线程在调用`lock()`和`unlock()`。

例如，考虑以下代码片段：

```cpp
#include <mutex>
#include <thread>
#include <string>
#include <iostream>

std::recursive_mutex m{};

void foo()
{
    m.lock();
    m.lock();

    std::cout << "The answer is: 42\n";

    m.unlock();
    m.unlock();
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    return 0;
}
```

前面的例子会导致以下结果：

![](img/fef09e3b-fb6a-479b-a600-6c482d4c8b94.png)

在前面的例子中，我们定义了一个函数，该函数调用递归互斥锁的`lock()`函数两次，输出到`stdout`，然后再调用`unlock()`函数两次。然后我们创建两个执行此函数的线程，结果是`stdout`没有腐败，也没有死锁。

# std::shared_mutex

直到这一点，我们的同步原语已经对共享资源进行了序列化访问。也就是说，每个线程在访问临界区时必须一次执行一个。虽然这确保了腐败是不可能的，但对于某些类型的场景来说效率不高。为了更好地理解这一点，我们必须研究是什么导致了腐败。

让我们考虑一个整数变量，它被两个线程同时增加。增加整数变量的过程如下：`i = i + 1`。

让我们将其写成如下形式：

```cpp
int i = 0;

auto tmp = i;
tmp++;
i = tmp; // i == 1
```

为了防止腐败，我们使用互斥锁来确保两个线程同步地增加整数：

```cpp
auto tmp_thread1 = i;
tmp_thread1++;
i = tmp_thread1; // i == 1

auto tmp_thread2 = i;
tmp_thread2++;
i = tmp_thread2; // i == 2
```

当这些操作混合在一起时（也就是说，当两个操作在不同的线程中同时执行时），就会发生腐败。例如，考虑以下代码：

```cpp
auto tmp_thread1 = i; // 0
auto tmp_thread2 = i; // 0
tmp_thread1++; // 1
tmp_thread2++; // 1
i = tmp_thread1; // i == 1
i = tmp_thread2; // i == 1
```

与整数为`2`不同，它是`1`，因为在第一个增量允许完成之前整数被读取。这种情况是可能的，因为两个线程都试图写入同一个共享资源。我们称这些类型的线程为**生产者**。

然而，如果我们创建了 100 万个同时读取共享资源的线程会发生什么。由于整数永远不会改变，无论线程以什么顺序执行，它们都会读取相同的值，因此腐败是不可能的。我们称这些线程为**消费者**。如果我们只有消费者，我们就不需要线程同步，因为腐败是不可能的。

最后，如果我们有相同的 100 万个消费者，但是我们在其中添加了一个生产者会发生什么？现在，我们必须使用线程同步，因为可能在生产者试图将一个值写入整数的过程中，消费者也试图读取，这将导致腐败的结果。为了防止这种情况发生，我们必须使用互斥锁来保护整数。然而，如果我们使用`std::mutex`，那么所有 100 万个消费者都必须互相等待，即使消费者们自己可以在不担心腐败的情况下同时执行。只有当生产者尝试执行时，我们才需要担心。

为了解决这个明显的性能问题，C++提供了`std::shared_mutex`对象。例如，考虑以下代码：

```cpp
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <iostream>

int count_rw{};
const auto &count_ro = count_rw;

std::shared_mutex m{};

void reader()
{
    while(true) {
        std::shared_lock lock(m);
        if (count_ro >= 42) {
            return;
        }
    }
}

void writer()
{
    while(true) {
        std::unique_lock lock(m);
        if (++count_rw == 100) {
            return;
        }
    }
}

int main(void)
{
    std::thread t1{reader};
    std::thread t2{reader};
    std::thread t3{reader};
    std::thread t4{reader};
    std::thread t5{writer};

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    return 0;
}
```

在前面的例子中，我们创建了一个生产者函数（称为`reader`函数）和一个消费者函数（称为`writer`函数）。生产者使用`std::unique_lock()`锁定互斥锁，而消费者使用`std::shared_lock()`锁定互斥锁。每当使用`std::unique_lock()`锁定互斥锁时，所有其他线程都必须等待（无论是生产者还是消费者）。然而，如果使用`std::shared_lock()`锁定互斥锁，使用`std::shared_lock()`再次尝试锁定互斥锁不会导致线程等待。

只有在调用`std::unique_lock()`时才需要等待。这允许消费者在不等待彼此的情况下执行。只有当生产者尝试执行时，消费者必须等待，防止消费者相互串行化，最终导致更好的性能（特别是如果消费者的数量是 100 万）。

应该注意，我们使用`const`关键字来确保消费者不是生产者。这个简单的技巧确保程序员不会在不经意间认为他们已经编写了一个消费者，而实际上他们已经创建了一个生产者，因为如果发生这种情况，编译器会警告程序员。

# std::timed_mutex

最后，我们还没有处理线程获取互斥锁后崩溃的情况。在这种情况下，任何尝试获取相同互斥锁的线程都会进入死锁状态，因为崩溃的线程永远没有机会调用`unlock()`。预防这种问题的一种方法是使用`std::timed_mutex`。

例如，考虑以下代码：

```cpp
#include <mutex>
#include <thread>
#include <iostream>

std::timed_mutex m{};

void foo()
{
    using namespace std::chrono;

    if (m.try_lock_for(seconds(1))) {
        std::cout << "lock acquired\n";
    }
    else {
        std::cout << "lock failed\n";
    }
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    return 0;
}
```

当执行这个时，我们得到以下结果：

![](img/a606b9bb-6ef7-4885-93e6-344fc3bc06e7.png)

在上面的例子中，我们告诉 C++线程只允许等待 1 秒。如果互斥锁已经被获取，并且在 1 秒后没有被释放，`try_lock_for()`函数将退出并返回 false，允许线程优雅地退出并处理错误，而不会进入死锁状态。

# 使用原子数据类型

在这个食谱中，我们将学习如何在 C++中使用原子数据类型。原子数据类型提供了读写简单数据类型（即布尔值或整数）的能力，而无需线程同步（即使用`std::mutex`和相关工具）。为了实现这一点，原子数据类型使用特殊的 CPU 指令来确保当执行操作时，它是作为单个原子操作执行的。

例如，递增一个整数可以写成如下：

```cpp
int i = 0;

auto tmp = i;
tmp++;
i = tmp; // i == 1
```

原子数据类型确保这个递增是以这样的方式执行的，即没有其他尝试同时递增整数的操作可以交错，并因此导致损坏。CPU 是如何做到这一点的超出了本书的范围。这是因为在现代的超标量、流水线化的 CPU 中，支持在多个核心和插槽上并行、乱序和推测性地执行指令，这是非常复杂的。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行此食谱中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个食谱：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe02_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行这个食谱中的每个示例：

```cpp
> ./recipe02_example01
count: 711
atomic count: 1000
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本食谱中所教授的课程的关系。

# 工作原理...

在这个食谱中，我们将学习如何使用 C++的原子数据类型。原子数据类型仅限于简单的数据类型，如整数，由于这些数据类型非常复杂，只支持简单的操作，如加法、减法、递增和递减。

让我们看一个简单的例子，不仅演示了如何在 C++中使用原子数据类型，还演示了为什么原子数据类型如此重要：

```cpp
#include <atomic>
#include <thread>
#include <iostream>

int count{};
std::atomic<int> atomic_count{};

void foo()
{
    do {
        count++;
        atomic_count++;
    }
    while (atomic_count < 99999);
}

int main(void)
{
    std::thread t1{foo};
    std::thread t2{foo};

    t1.join();
    t2.join();

    std::cout << "count: " << count << '\n';
    std::cout << "atomic count: " << atomic_count << '\n';

    return 0;
}
```

当执行这段代码时，我们得到以下结果：

![](img/216f9a16-1893-4c5f-b259-da1e2d0b4bc0.png)

在上面的示例中，我们有两个整数。第一个整数是普通的 C/C++整数类型，而第二个是原子数据类型（整数类型）。然后，我们定义一个循环，直到原子数据类型为`1000`为止。最后，我们从两个线程中执行这个函数，这意味着我们的全局整数会被两个线程同时增加。

如您所见，这个简单测试的输出显示，简单的 C/C++整数数据类型与原子数据类型的值不同，但两者都增加了相同次数。这个原因可以在这个函数的汇编中看到（在 Intel CPU 上），如下所示：

![](img/b1e3de9e-b754-49b6-a53a-d4e0bfd9cc2f.png)

要增加一个整数（未启用优化），编译器必须将内存内容移动到寄存器中，将`1`添加到寄存器中，然后将寄存器的结果写回内存。由于这段代码同时在两个不同的线程中执行，这段代码交错执行，导致损坏。原子数据类型不会遇到这个问题。这是因为增加原子数据类型的过程发生在一个单独的特殊指令中，CPU 确保执行，而不会将其内部状态与其他指令的相同内部状态交错在一起，也不会在其他 CPU 上交错。

原子数据类型通常用于实现同步原语，例如`std::mutex`（尽管在实践中，`std::mutex`是使用测试和设置指令实现的，这些指令使用类似的原理，但通常比原子指令执行得更快）。这些数据类型还可以用于实现称为无锁数据结构的特殊数据结构，这些数据结构能够在多线程环境中运行，而无需`std::mutex`。无锁数据结构的好处是在处理线程同步时没有等待状态，但会增加更复杂的 CPU 硬件和其他类型的性能惩罚（当 CPU 遇到原子指令时，大多数由硬件提供的 CPU 优化必须暂时禁用）。因此，就像计算机科学中的任何东西一样，它们都有其时机和地点。

# 在多线程的上下文中理解 const & mutable 的含义

在这个示例中，我们将学习如何处理被标记为`const`的对象，但包含必须使用`std::mutex`来确保线程同步的对象。这个示例很重要，因为将`std::mutex`存储为类的私有成员是很有用的，但是，一旦你这样做了，将这个对象的实例作为常量引用（即`const &`）传递将导致编译错误。在这个示例中，我们将演示为什么会发生这种情况以及如何克服它。

# 准备工作

在我们开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本示例中示例的正确工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个示例：

1.  从一个新的终端中，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe03_example01
The answer is: 42

> ./recipe03_example03
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 它是如何工作的...

在本示例中，我们将学习如何将`std::mutex`添加到类的私有成员中，同时仍然能够处理`const`情况。一般来说，确保对象是线程安全的有两种方法。第一种方法是将`std::mutex`放在全局级别。这样做可以确保对象可以作为常量引用传递，或者对象本身可以有一个标记为`const`的函数。

为此，请考虑以下代码示例：

```cpp
#include <mutex>
#include <thread>
#include <iostream>

std::mutex m{};

class the_answer
{
public:
    void print() const
    {
        std::lock_guard lock(m);
        std::cout << "The answer is: 42\n";
    }
};

int main(void)
{
    the_answer is;
    is.print();

    return 0;
}
```

在前面的例子中，当执行`print()`函数时，我们创建了一个对象，该对象输出到`stdout`。`print()`函数被标记为`const`，这告诉编译器`print()`函数不会修改任何类成员（即函数是只读的）。由于`std::mutex`是全局的，对象的 const 限定符被维持，代码可以编译和执行而没有问题。

全局`std::mutex`对象的问题在于，对象的每个实例都必须使用相同的`std::mutex`对象。如果用户打算这样做，那没问题，但如果您希望对象的每个实例都有自己的`std::mutex`对象（例如，当对象的相同实例可能被多个线程执行时），该怎么办？

为此，让我们看看如何使用以下示例发生的情况：

```cpp
#include <mutex>
#include <thread>
#include <iostream>

class the_answer
{
    std::mutex m{};

public:
    void print() const
    {
        std::lock_guard lock(m);
        std::cout << "The answer is: 42\n";
    }
};

int main(void)
{
    the_answer is;
    is.print();

    return 0;
}
```

如果我们尝试编译这个，我们会得到以下结果：

![](img/944a6bd9-fba1-4f70-b061-5dc7c7c4afba.png)

在前面的例子中，我们所做的只是将前面的例子中的`std::mutex`移动到类内部作为私有成员。结果是，当我们尝试编译类时，我们会得到一个编译器错误。这是因为`print()`函数被标记为`const`，这告诉编译器`print()`函数不会修改类的任何成员。问题在于，当您尝试锁定`std::mutex`时，您必须对其进行修改，从而导致编译器错误。

为了克服这个问题，我们必须告诉编译器忽略这个错误，方法是将`std::mutex`标记为 mutable。将成员标记为 mutable 告诉编译器允许修改该成员，即使对象被作为常量引用传递或对象定义了常量函数。

例如，这是`const`标记为`mutable`的代码示例：

```cpp
#include <mutex>
#include <thread>
#include <iostream>

class the_answer
{
    mutable std::mutex m{};

public:
    void print() const
    {
        std::lock_guard lock(m);
        std::cout << "The answer is: 42\n";
    }
};

int main(void)
{
    the_answer is;
    is.print();

    return 0;
}
```

如前面的例子所示，一旦我们将`std::mutex`标记为 mutable，代码就会像我们期望的那样编译和执行。值得注意的是，`std::mutex`是少数几个可以接受 mutable 使用的例子之一。mutable 关键字很容易被滥用，导致代码无法编译或操作不符合预期。

# 使类线程安全

在本示例中，我们将学习如何使一个类线程安全（即如何确保一个类的公共成员函数可以随时被任意数量的线程同时调用）。大多数类，特别是由 C++标准库提供的类，都不是线程安全的，而是假设用户会根据需要添加线程同步原语，如`std::mutex`对象。这种方法的问题在于，每个对象都有两个实例，必须在代码中进行跟踪：类本身和它的`std::mutex`。用户还必须用自定义版本包装对象的每个函数，以使用`std::mutex`保护类，结果不仅有两个必须管理的对象，还有一堆 C 风格的包装函数。

这个示例很重要，因为它将演示如何通过创建一个线程安全的类来解决代码中的这些问题，将所有内容合并到一个单一的类中。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本示例的正确工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个教程：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe04_example01
```

在接下来的部分中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它们与本教程中所教授的课程的关系。

# 它是如何工作的...

在本教程中，我们将学习如何通过实现自己的线程安全栈来制作一个线程安全的类。C++标准库不提供线程安全的数据结构，因此，如果您希望在多个线程中使用数据结构作为全局资源，您需要手动添加线程安全性。这可以通过实现包装函数或创建包装类来实现。

创建包装函数的优势在于，对于全局对象，通常所需的代码量更少，更容易理解，而线程安全类的优势在于，您可以创建类的多个实例，因为`std::mutex`是自包含的。

可以尝试以下代码示例：

```cpp
#include <mutex>
#include <stack>
#include <iostream>

template<typename T>
class my_stack
{
    std::stack<T> m_stack;
    mutable std::mutex m{};

public:

    template<typename ARG>
    void push(ARG &&arg)
    {
        std::lock_guard lock(m);
        m_stack.push(std::forward<ARG>(arg));
    }

 void pop()
    {
        std::lock_guard lock(m);
        m_stack.pop();
    }

    auto empty() const
    {
        std::lock_guard lock(m);
        return m_stack.empty();
    }
};
```

在前面的示例中，我们实现了自己的栈。这个栈有`std::stack`和`std::mutex`作为成员变量。然后，我们重新实现了`std::stack`提供的一些函数。这些函数中的每一个首先尝试获取`std::mutex`，然后调用`std::stack`中的相关函数。在`push()`函数的情况下，我们利用`std::forward`来确保传递给`push()`函数的参数被保留。

最后，我们可以像使用`std::stack`一样使用我们的自定义栈。例如，看一下以下代码：

```cpp
int main(void)
{
    my_stack<int> s;

    s.push(4);
    s.push(8);
    s.push(15);
    s.push(16);
    s.push(23);
    s.push(42);

    while(s.empty()) {
        s.pop();
    }

    return 0;
}
```

正如您所看到的，`std::stack`和我们的自定义栈之间唯一的区别是我们的栈是线程安全的。

# 同步包装器及其实现方式

在本教程中，我们将学习如何制作线程安全的同步包装器。默认情况下，C++标准库不是线程安全的，因为并非所有应用程序都需要这种功能。确保 C++标准库是线程安全的一种机制是创建一个线程安全类，它将您希望使用的数据结构以及`std::mutex`作为私有成员添加到类中，然后重新实现数据结构的函数以首先获取`std::mutex`，然后转发函数调用到数据结构。这种方法的问题在于，如果数据结构是全局资源，程序中会添加大量额外的代码，使得最终的代码难以阅读和维护。

这个教程很重要，因为它将演示如何通过制作线程安全的同步包装器来解决代码中的这些问题。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有正确的工具来编译和执行本教程中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何操作...

您需要执行以下步骤来尝试这个教程：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe05_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe05_example01
```

在接下来的部分中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它们与本教程中所教授的课程的关系。

# 它是如何工作的...

在本教程中，我们将学习如何创建线程安全的同步包装器，这允许我们向 C++标准库数据结构添加线程安全性，而默认情况下这些数据结构是不安全的。

为此，我们将为 C++标准库中的每个函数创建包装函数。这些包装函数将首先尝试获取`std::mutex`，然后将相同的函数调用转发到 C++标准库数据结构。

为此，请考虑以下代码示例：

```cpp
#include <mutex>
#include <stack>
#include <iostream>

std::mutex m{};

template<typename S, typename T>
void push(S &s, T &&t)
{
    std::lock_guard lock(m);
    s.push(std::forward<T>(t));
}

template<typename S>
void pop(S &s)
{
    std::lock_guard lock(m);
    s.pop();
}

template<typename S>
auto empty(S &s)
{
    std::lock_guard lock(m);
    return s.empty();
}
```

在前面的例子中，我们为`push()`、`pop()`和`empty()`函数创建了一个包装函数。这些函数在调用数据结构之前会尝试获取我们的全局`std::mutex`对象，这里是一个模板。使用模板创建了一个概念。我们的包装函数可以被实现了`push()`、`pop()`和`empty()`的任何数据结构使用。另外，请注意我们在`push()`函数中使用`std::forward`来确保被推送的参数的 l-valueness 和 CV 限定符保持不变。

最后，我们可以像使用数据结构的函数一样使用我们的包装器，唯一的区别是数据结构作为第一个参数传递。例如，看一下以下代码块：

```cpp
int main(void)
{
    std::stack<int> mystack;

    push(mystack, 4);
    push(mystack, 8);
    push(mystack, 15);
    push(mystack, 16);
    push(mystack, 23);
    push(mystack, 42);

    while(empty(mystack)) {
        pop(mystack);
    }

    return 0;
}
```

正如前面的例子中所示，使用我们的同步包装器是简单的，同时确保我们创建的堆栈现在是线程安全的。

# 阻塞操作与异步编程

在本示例中，我们将学习阻塞操作和异步操作之间的区别。这个示例很重要，因为阻塞操作会使每个操作在单个 CPU 上串行执行。如果每个操作的执行必须按顺序执行，这通常是可以接受的；然而，如果这些操作可以并行执行，异步编程可以是一个有用的优化，确保在一个操作等待时，其他操作仍然可以在同一个 CPU 上执行。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本示例中的示例的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个示例：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe06_examples
```

1.  源代码编译后，您可以通过运行以下命令执行本示例中的每个示例：

```cpp
> time ./recipe06_example01
999999
999999
999999
999999

real 0m1.477s
...

> time ./recipe06_example02
999999
999999
999999
999999

real 0m1.058s
...

> time ./recipe06_example03
999999
999999
999998
999999

real 0m1.140s
...
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

阻塞操作是指必须在下一个操作发生之前完成的操作。大多数程序是按顺序编写的，这意味着每个指令必须在下一个指令之前执行。然而，问题在于有些操作可以并行执行（即同时或异步执行）。串行化这些操作在最好的情况下可能会导致性能不佳，并且在某些情况下实际上可能会导致死锁（程序进入无休止的等待状态），如果阻塞的操作正在等待另一个从未有机会执行的操作。

为了演示一个阻塞操作，让我们来看一下以下内容：

```cpp
#include <vector>
#include <iostream>
#include <algorithm>

constexpr auto size = 1000000;

int main(void)
{
    std::vector<int> numbers1(size);
    std::vector<int> numbers2(size);
    std::vector<int> numbers3(size);
    std::vector<int> numbers4(size);
```

前面的代码创建了一个主函数，其中有四个`int`类型的`std::vector`对象。在接下来的步骤中，我们将使用这些向量来演示一个阻塞操作。

1.  首先，我们创建四个可以存储整数的向量：

```cpp
    std::generate(numbers1.begin(), numbers1.end(), []() {
      return rand() % size;
    });
    std::generate(numbers2.begin(), numbers2.end(), []() {
      return rand() % size;
    });
    std::generate(numbers3.begin(), numbers3.end(), []() {
      return rand() % size;
    });
    std::generate(numbers4.begin(), numbers4.end(), []() {
      return rand() % size;
    });
```

1.  接下来，我们使用`std::generate`用随机数填充每个数组，结果是一个带有数字和随机顺序的数组：

```cpp
    std::sort(numbers1.begin(), numbers1.end());
    std::sort(numbers2.begin(), numbers2.end());
    std::sort(numbers3.begin(), numbers3.end());
    std::sort(numbers4.begin(), numbers4.end());
```

1.  接下来，我们对整数数组进行排序，这是本例的主要目标，因为这个操作需要一段时间来执行：

```cpp
    std::cout << numbers1.back() << '\n';
    std::cout << numbers2.back() << '\n';
    std::cout << numbers3.back() << '\n';
    std::cout << numbers4.back() << '\n';

    return 0;
}
```

1.  最后，我们输出每个数组中的最后一个条目，通常会是`999999`（但不一定，因为数字是使用随机数生成器生成的）。

前面示例的问题在于操作可以并行执行，因为每个数组是独立的。为了解决这个问题，我们可以异步执行这些操作，这意味着数组将并行创建、填充、排序和输出。例如，考虑以下代码：

```cpp
#include <future>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>

constexpr auto size = 1000000;

int foo()
{
    std::vector<int> numbers(size);
    std::generate(numbers.begin(), numbers.end(), []() {
      return rand() % size;
    });

    std::sort(numbers.begin(), numbers.end());
    return numbers.back();
}
```

我们首先要做的是实现一个名为`foo()`的函数，该函数创建我们的向量，用随机数填充它，对列表进行排序，并返回数组中的最后一个条目（与前面的示例相同，唯一的区别是我们一次只处理一个数组，而不是`4`个）：

```cpp
int main(void)
{
    auto a1 = std::async(std::launch::async, foo);
    auto a2 = std::async(std::launch::async, foo);
    auto a3 = std::async(std::launch::async, foo);
    auto a4 = std::async(std::launch::async, foo);

    std::cout << a1.get() << '\n';
    std::cout << a2.get() << '\n';
    std::cout << a3.get() << '\n';
    std::cout << a4.get() << '\n';

    return 0;
}
```

然后，我们使用`std::async`四次执行这个`foo()`函数，得到与前面示例相同的四个数组。在这个示例中，`std::async()`函数做的事情与手动执行四个线程相同。`std::aync()`的结果是一个`std::future`对象，它在函数执行完成后存储函数的结果。在这个示例中，我们做的最后一件事是使用`get()`函数在函数准备好后返回函数的值。

如果我们计时这些函数的结果，我们会发现异步版本比阻塞版本更快。以下代码显示了这一点（`real`时间是查找时间）：

![](img/46ef0e32-b06c-4bc6-9b92-5984d00d7432.png)

`std::async()`函数也可以用来在同一个线程中异步执行我们的数组函数。例如，考虑以下代码：

```cpp
int main(void)
{
    auto a1 = std::async(std::launch::deferred, foo);
    auto a2 = std::async(std::launch::deferred, foo);
    auto a3 = std::async(std::launch::deferred, foo);
    auto a4 = std::async(std::launch::deferred, foo);

    std::cout << a1.get() << '\n';
    std::cout << a2.get() << '\n';
    std::cout << a3.get() << '\n';
    std::cout << a4.get() << '\n';

    return 0;
}
```

如前面的示例所示，我们将操作从`std::launch::async`更改为`std::launch::deferred`，这将导致每个函数在需要函数结果时执行一次（即调用`get()`函数时）。如果不确定函数是否需要执行（即仅在需要时执行函数），这将非常有用，但缺点是程序的执行速度较慢，因为线程通常不用作优化方法。

# 使用承诺和未来

在本配方中，我们将学习如何使用 C++承诺和未来。C++ `promise`是 C++线程的参数，而 C++ `future`是线程的返回值，并且可以用于手动实现`std::async`调用的相同功能。这个配方很重要，因为对`std::aync`的调用要求每个线程停止执行以获取其结果，而手动实现 C++ `promise`和`future`允许用户在线程仍在执行时获取线程的返回值。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本配方中示例所需的适当工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行示例。

# 如何做...

您需要执行以下步骤来尝试这个配方：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter05
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe07_examples
```

1.  编译源代码后，可以通过运行以下命令来执行本配方中的每个示例：

```cpp
> ./recipe07_example01
The answer is: 42

> ./recipe07_example02
The answer is: 42
```

在下一节中，我们将逐个介绍每个示例，并解释每个示例程序的作用及其与本配方中所教授的课程的关系。

# 它是如何工作的...

在本配方中，我们将学习如何手动使用 C++ `promise`和`future`来提供一个并行执行带有参数的函数，并获取函数的返回值。首先，让我们演示如何以最简单的形式完成这个操作，使用以下代码：

```cpp
#include <thread>
#include <iostream>
#include <future>

void foo(std::promise<int> promise)
{
    promise.set_value(42);
}

int main(void)
{
    std::promise<int> promise;
    auto future = promise.get_future();

    std::thread t{foo, std::move(promise)};
    t.join();

    std::cout << "The answer is: " << future.get() << '\n';

    return 0;
}
```

执行前面的示例会产生以下结果：

![](img/5313a9ee-d6f1-449f-90df-069c182a2a80.png)

正如您在上面的代码中所看到的，C++的`promise`是作为函数的参数进行线程化的。线程通过设置`promise`参数来返回其值，而`promise`又设置了一个 C++的`future`，用户可以从提供给线程的`promise`参数中获取。需要注意的是，我们使用`std::move()`来防止`promise`参数被复制（编译器会禁止，因为 C++的`promise`是一个只能移动的类）。最后，我们使用`get()`函数来获取线程的结果，就像使用`std::async`执行线程的结果一样。

手动使用`promise`和`future`的一个好处是，可以在线程完成之前获取线程的结果，从而允许线程继续工作。例如，看下面的例子：

```cpp
#include <thread>
#include <iostream>
#include <future>

void foo(std::promise<int> promise)
{
    promise.set_value(42);
    while (true);
}

int main(void)
{
    std::promise<int> promise;
    auto future = promise.get_future();

    std::thread t{foo, std::move(promise)};

    future.wait();
    std::cout << "The answer is: " << future.get() << '\n';

    t.join();

    // Never reached
    return 0;
}
```

执行时会得到以下结果：

![](img/af9f0ada-0fe3-4d17-9c75-52f61975d425.png)

在上面的例子中，我们创建了相同的线程，但在线程中无限循环，意味着线程永远不会返回。然后我们以相同的方式创建线程，但在 C++的`future`准备好时立即输出结果，我们可以使用`wait()`函数来确定。
