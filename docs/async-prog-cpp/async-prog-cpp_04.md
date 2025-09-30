

# 第四章：使用锁进行线程同步

在 *第二章* 中，我们了解到线程可以读取和写入它们所属进程共享的内存。虽然操作系统实现了进程内存访问保护，但同一进程中对共享内存的线程访问没有这种保护。来自多个线程对同一内存地址的并发内存写操作需要同步机制来避免数据竞争并确保数据完整性。

在本章中，我们将详细描述由多个线程对共享内存并发访问所引起的问题以及如何解决这些问题。我们将详细研究以下主题：

+   竞态条件 – 它们是什么以及如何发生

+   互斥作为同步机制及其在 C++中通过 **std::mutex** 的实现

+   泛型锁管理

+   条件变量是什么以及如何与互斥锁一起使用

+   使用 **std::mutex** 和 **std::condition_variable** 实现一个完全同步的队列

+   C++20 引入的新同步原语 – 信号量、屏障和闩锁

这些都是基于锁的同步机制。无锁技术是下一章的主题。

# 技术要求

本章的技术要求与上一章中解释的概念相同，要编译和运行示例，需要一个支持 C++20 的 C++编译器（用于信号量、闩锁和屏障示例）。大多数示例只需要 C++11。示例已在 Linux Ubuntu LTS 24.04 上测试。

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)

# 理解竞态条件

当程序运行的输出结果取决于其指令执行的顺序时，就会发生竞态条件。我们将从一个非常简单的例子开始，展示竞态条件是如何发生的，然后在本章的后面部分，我们将学习如何解决这个问题。

在以下代码中，**counter** 全局变量由两个并发运行的线程递增：

```cpp
#include <iostream>
#include <thread>
int counter = 0;
int main() {
    auto func = [] {
        for (int i = 0; i < 1000000; ++i) {
            counter++;
        }
    };
    std::thread t1(func);
    std::thread t2(func);
    t1.join();
    t2.join();
    std::cout << counter << std::endl;
    return 0;
}
```

运行前面的代码三次后，我们得到以下 **counter** 值：

```cpp
1056205
1217311
1167474
```

在这里，我们看到了两个主要问题：首先，**counter** 的值是不正确的；其次，每次程序执行都以不同的 **counter** 值结束。结果是不可预测的，并且大多数情况下是错误的。如果你非常幸运，可能会得到正确的值，但这非常不可能。

这种情况涉及两个线程，**t1**和**t2**，它们并发运行并修改相同的变量，本质上是一些内存区域。看起来它应该可以正常工作，因为只有一行代码增加了**计数器**的值，从而修改了内存内容（顺便说一句，我们使用后增量运算符如**counter++**或前增量运算符如**++counter**并不重要；结果都会同样错误）。

仔细观察前面的代码，让我们仔细研究以下这一行：

```cpp
        counter++;
```

它通过三个步骤增加**计数器**：

+   存储在**计数器**变量内存地址中的内容被加载到一个 CPU 寄存器中。在这种情况下，从内存中加载一个**int**数据类型到 CPU 寄存器中。

+   寄存器中的值增加一。

+   寄存器中的值存储在**计数器**变量内存地址中。

现在，让我们考虑一个可能的情况，即当两个线程尝试并发地增加计数器时。让我们看看*表 4.1*：

| **线程 1** | **线程 2** |
| --- | --- |
| [1] 将计数器值加载到寄存器 | [3] 将计数器值加载到寄存器 |
| [2] 增加寄存器值 | [5] 增加寄存器值 |
| [4] 将寄存器存储到计数器 | [6] 将寄存器存储到计数器 |

表 4.1：两个线程并发增加计数器

线程 1 执行[1]，并将计数器的当前值（假设是 1）加载到一个 CPU 寄存器中。然后，它通过[2]将寄存器中的值增加一（现在，寄存器中的值是 2）。

线程 2 被调度执行，[3]将计数器的当前值（记住——它尚未被修改，所以仍然是 1）加载到一个 CPU 寄存器中。

现在，线程 1 再次被调度执行，[4]将更新后的值存储到内存中。此时，**计数器**的值现在是二。

最后，线程 2 再次被调度，并执行[5]和[6]。寄存器值增加一，然后将值二存储在内存中。**计数器**变量只增加了一次，而它应该增加两次，其值应该是三。

之前的问题发生是因为对计数器的增量操作不是原子的。如果每个线程都能在没有被中断的情况下执行增加**计数器**变量所需的三个指令，那么**计数器**就会像预期的那样增加两次。然而，根据操作执行的顺序，结果可能会有所不同。这被称为**竞态条件**。

为了避免竞态条件，我们需要确保共享资源以受控的方式被访问和修改。实现这一目标的一种方法是通过使用锁。**锁**是一种同步原语，它允许一次只有一个线程访问共享资源。当线程想要访问共享资源时，它必须首先获取锁。一旦线程获取了锁，它就可以在没有其他线程干扰的情况下访问共享资源。当线程完成对共享资源的访问后，它必须释放锁，以便其他线程可以访问它。

另一种避免竞态条件的方法是使用**原子操作**。原子操作是一种保证在单个、不可分割的步骤中执行的操作。这意味着在操作执行期间，没有其他线程可以干扰原子操作。原子操作通常使用设计为不可分割的硬件指令来实现。原子操作将在*第五章*中解释。

在本节中，我们看到了由多线程代码引起的最常见和重要的问题：竞态条件。我们看到了根据操作执行的顺序，结果可能会有所不同。带着这个问题，我们将研究如何在下一节中解决它。

# 我们为什么需要互斥锁？

**互斥锁**是并发编程中的一个基本概念，它确保多个线程或进程不会同时访问共享资源，例如共享变量、代码的关键部分或文件或网络连接。互斥锁对于防止如前节所见到的竞态条件至关重要。

想象一家小咖啡馆，只有一台意式浓缩咖啡机。这台机器一次只能制作一杯浓缩咖啡。这意味着这台机器是一个所有咖啡师都必须共享的关键资源。

这家咖啡馆由三位咖啡师：Alice、Bob 和 Carol 负责。他们*并发*使用咖啡机，但不能同时使用，因为这可能会造成问题：Bob 将适量的新鲜研磨咖啡放入机器中，开始制作浓缩咖啡。然后，Alice 也这样做，但首先从机器中取出咖啡，认为 Bob 忘记做了。Bob 然后从机器中取出浓缩咖啡，之后，Alice 发现没有浓缩咖啡了！这是一场灾难——我们计数程序的现实版本。

为了解决咖啡馆的问题，他们可能会任命 Carol 为机器管理员。在使用机器之前，Alice 和 Bob 都会问她是否可以开始制作新的浓缩咖啡。这样就能解决问题。

回到我们的计数器程序，如果我们能允许一次只有一个线程访问**counter**（就像 Carol 在咖啡馆里做的那样），我们的软件问题也会得到解决。互斥是一种可以用来控制并发线程访问内存的机制。C++标准库提供了**std::mutex**类，这是一个同步原语，用于保护共享数据不被两个或更多线程同时访问。

我们在上节中看到的这个新版本的代码实现了两种并发增加**counter**的方式：自由访问，如前节所述，以及使用互斥同步的访问：

```cpp
#include <iostream>
#include <mutex>
#include <thread>
std::mutex mtx;
int counter = 0;
int main() {
    auto funcWithoutLocks = [] {
        for (int i = 0; i < 1000000; ++i) {
            ++counter;
        };
    };
    auto funcWithLocks = [] {
        for (int i = 0; i < 1000000; ++i) {
            mtx.lock();
            ++counter;
            mtx.unlock();
        };
    };
    {
        counter = 0;
        std::thread t1(funcWithoutLocks);
        std::thread t2(funcWithoutLocks);
        t1.join();
        t2.join();
        std::cout << "Counter without using locks: " << counter << std::endl;
    }
    {
        counter = 0;
        std::thread t1(funcWithLocks);
        std::thread t2(funcWithLocks);
        t1.join();
        t2.join();
        std::cout << "Counter using locks: " << counter << std::endl;
    }
    return 0;
}
```

当一个线程运行**funcWithLocks**时，它在增加**counter**之前使用**mtx.lock()**获取锁。一旦**counter**被增加，线程将释放锁（**mtx.unlock()**）。

锁只能被一个线程拥有。例如，如果**t1**获取了锁，然后**t2**也尝试获取它，**t2**将被阻塞并等待直到锁可用。因为任何时刻只有一个线程可以拥有锁，所以这种同步原语被称为**互斥锁**（来自“互斥”）。如果你运行这个程序几次，你总是会得到正确的结果：**2000000**。

在本节中，我们介绍了互斥的概念，并了解到 C++标准库提供了**std::mutex**类作为线程同步的原语。在下一节中，我们将详细研究**std::mutex**。

## C++标准库互斥实现

在上一节中，我们介绍了互斥和互斥锁的概念，以及为什么需要它们来同步并发内存访问。在本节中，我们将看到 C++标准库提供的用于实现互斥的类。我们还将看到 C++标准库提供的一些辅助类，使互斥锁的使用更加容易。

下表总结了 C++标准库提供的互斥锁类及其主要特性：

| **Mutex Type** | **Access** | **Recursive** | **Timeout** |
| --- | --- | --- | --- |
| **std::mutex** | EXCLUSIVE | NO | NO |
| **std::recursive_mutex** | EXCLUSIVE | YES | NO |
| **std::shared_mutex** | 1 - EXCLUSIVEN - SHARED | NO | NO |
| **std::timed_mutex** | EXCLUSIVE | NO | YES |
| **std::recursive_timed_mutex** | EXCLUSIVE | YES | YES |
| **std::shared_timed_mutex** | 1 - EXCLUSIVEN - SHARED | NO | YES |

表 4.2：C++标准库中的互斥锁类

让我们逐一探索这些类。

### std::mutex

**std::mutex**类是在 C++11 中引入的，它是 C++标准库提供的最重要的、最常使用的同步原语之一。

如我们在本章前面所见，**std::mutex**是一个同步原语，可以用来保护共享数据不被多个线程同时访问。

**std::mutex**类提供了独占、非递归的所有权语义。

**std::mutex**的主要特性如下：

+   从调用线程成功调用**lock()**或**try_lock()**到调用**unlock()**，调用线程拥有互斥锁。

+   在调用**lock()**或**try_lock(**)之前，调用线程不得拥有互斥锁。这是**std::mutex**的非递归所有权语义属性。

+   当一个线程拥有互斥锁时，所有其他线程将阻塞（在调用**lock()**时）或接收一个**false**返回值（在调用**try_lock()**时）。这是**std::mutex**的独占所有权语义。

如果一个拥有互斥锁的线程尝试再次获取它，其行为是未定义的。通常，在这种情况下会抛出一个异常，但这是由实现定义的。

如果一个线程在释放互斥锁之后，再次尝试释放它，这同样是不确定的行为（就像前一个情况一样）。

当一个线程持有互斥锁时，互斥锁被销毁，或者线程在未释放锁的情况下终止，这些都是不确定行为的原因。

**std::mutex**类有三个方法：

+   **lock()**：调用**lock()**会获取互斥锁。如果互斥锁已被锁定，则调用线程将被阻塞，直到互斥锁被解锁。从应用程序的角度来看，这就像调用线程在等待互斥锁可用一样。

+   **try_lock()**：当调用此函数时，它返回**true**，表示互斥锁已被成功锁定，或者在互斥锁已被锁定的情况下返回**false**。请注意，**try_lock**是非阻塞的，调用线程要么获取互斥锁，要么不获取，但它不会像调用**lock()**时那样被阻塞。**try_lock()**方法通常在我们不希望线程等待互斥锁可用时使用。当我们希望线程继续进行一些处理并稍后尝试获取互斥锁时，我们将调用**try_lock()**。

+   **unlock()**：调用**unlock()**会释放互斥锁。

### std::recursive_mutex

**std::mutex**类提供了独占、非递归的所有权语义。虽然对于至少一个线程来说，独占所有权语义总是需要的（毕竞它是一个互斥机制），但在某些情况下，我们可能需要递归地获取互斥锁。例如，一个递归函数可能需要获取一个互斥锁。我们也可能需要在从另一个函数**f()**中调用的函数**g()**中获取互斥锁。

**std::recursive_mutex**类提供了独占、递归语义。其主要特性如下：

+   调用线程可能多次获取相同的互斥锁。它将持有互斥锁，直到它释放互斥锁的次数与它获取的次数相同。例如，如果一个线程递归地获取一个互斥锁三次，它将持有互斥锁，直到它第三次释放它。

+   递归互斥锁可以递归获取的最大次数是不确定的，因此是实现定义的。一旦互斥锁已被获取最大次数，调用 **lock()** 将会抛出 **std::system_error**，而调用 **try_lock()** 将返回 **false**。

+   所有权与 **std::mutex** 相同：如果一个线程拥有 **std::recursive_mutex** 类，那么任何其他线程在尝试通过调用 **lock()** 获取它时都会阻塞，或者在调用 **try_lock()** 时返回 **false**。

**std::recursive_mutex** 的接口与 **std::mutex** 完全相同。

### std::shared_mutex

**std::mutex** 和 **std::shared_mutex** 都具有独占所有权的语义，在任何给定时间只有一个线程可以是互斥锁的所有者。尽管如此，也有一些情况下，我们可能需要让多个线程同时访问受保护的数据，并只给一个线程提供独占访问权限。

所需的计数器反例要求每个线程对单个变量具有独占访问权限，因为它们都在更新**counter**值。现在，如果我们有只要求读取**counter**当前值的线程，并且只有一个线程需要增加其值，那么让读取线程并发访问**counter**并将写入线程的独占访问权限会更好。

此功能是通过所谓的读者-写者锁实现的。C++ 标准库实现了具有类似（但不完全相同）功能的 **std::shared_mutex** 类。

**std::shared_mutex** 与其他互斥锁类型的主要区别在于它有两个访问级别：

+   **共享**：多个线程可以共享同一个互斥锁的所有权。共享所有权通过调用 **lock_shared()**、**try_lock_shared()** / **unlock_shared()** 来获取/释放。当至少有一个线程已经获取了对锁的共享访问权限时，没有其他线程可以获取独占访问权限，但它可以获取共享访问权限。

+   **独占**：只有一个线程可以拥有互斥锁。独占所有权通过调用**lock()**、**try_lock()** / **unlock()** 来获取/释放。当一个线程已经获取了对锁的独占访问权限时，没有其他线程可以获取共享或独占访问权限。

让我们通过一个简单的例子来看看如何使用 **std::shared_mutex**：

```cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <shared_mutex>
#include <thread>
int counter = 0;
int main() {
    using namespace std::chrono_literals;
    std::shared_mutex mutex;
    auto reader = [&] {
        for (int i = 0; i < 10; ++i) {
            mutex.lock_shared();
            // Read the counter and do something
            mutex.unlock_shared();
        }
    };
    auto writer = [&] {
        for (int i = 0; i < 10; ++i) {
            mutex.lock();
            ++counter;
            std::cout << "Counter: " << counter << std::endl;
            mutex.unlock();
            std::this_thread::sleep_for(10ms);
        }
    };
    std::thread t1(reader);
    std::thread t2(reader);
    std::thread t3(writer);
    std::thread t4(reader);
    std::thread t5(reader);
    std::thread t6(writer);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    return 0;
}
```

示例使用 **std::shared_mutex** 来同步六个线程：其中两个线程是写入者，它们增加**counter**的值并需要独占访问。其余四个线程只读取**counter**，只需要共享访问。此外，请注意，为了使用 **std::shared_mutex**，我们需要包含 **<shared_mutex>** 头文件。

### 定时互斥锁类型

我们至今所见到的互斥锁类型，当我们想要获取锁以进行独占使用时，表现方式相同：

+   **std::lock()** : 调用线程会阻塞，直到锁可用

+   **std::try_lock()**：如果锁不可用，则返回 **false**

在**std::lock()**的情况下，调用线程可能需要等待很长时间，我们可能只需要等待一段时间，然后如果线程还没有能够获取到锁，就让它继续进行一些处理。

为了实现这个目标，我们可以使用 C++标准库提供的定时互斥锁：**std::timed_mutex**、**std::recursive_timed_mutex**和**std::shared_time_mutex**。

它们与它们的非定时对应物类似，并实现了以下附加功能，以允许等待锁在特定时间段内可用：

+   **try_lock_for()**：尝试锁定互斥锁，并在指定的时间段内（超时）阻塞线程。如果在指定的时间段之前互斥锁被锁定，则返回**true**；否则，返回**false**。

    如果指定的时间段小于或等于零（**timeout_duration.zero()**），则该函数的行为与**try_lock()**完全相同。

    由于调度或竞争延迟，此函数可能会阻塞超过指定的时间段。

+   **try_lock_until()**：尝试锁定互斥锁，直到指定的超时时间或互斥锁被锁定，以先到者为准。在这种情况下，我们指定一个未来的实例作为等待的限制。

以下示例展示了如何使用**std::try_lock_for()**：

```cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
constexpr int NUM_THREADS = 8;
int counter = 0;
int failed = 0;
int main() {
    using namespace std::chrono_literals;
    std::timed_mutex tm;
    std::mutex m;
    auto worker = [&] {
        for (int i = 0; i < 10; ++i) {
            if (tm.try_lock_for(10ms)) {
                ++counter;
                std::cout << "Counter: " << counter << std::endl;
                std::this_thread::sleep_for(10ms);
                m.unlock();
            }
            else {
                m.lock();
                ++failed;
                std::cout << "Thread " << std::this_thread::get_id() << " failed to lock" << std::endl;
                m.unlock();
            }
            std::this_thread::sleep_for(12ms);
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Counter: " << counter << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    return 0;
}
```

上述代码使用了两个锁：**tm**，一个定时互斥锁，用于同步对**counter**的访问以及在成功获取**tm**的情况下向屏幕写入，以及**m**，一个非定时互斥锁，用于在未成功获取**tm**的情况下同步对**failed**的访问以及向屏幕写入。

## 使用锁时可能出现的问题

我们已经看到了仅使用互斥锁（锁）的示例。如果我们只需要一个互斥锁并且正确地获取和释放它，通常编写正确的多线程代码并不困难。一旦我们需要多个锁，代码复杂性就会增加。使用多个锁时常见的两个问题是*死锁*和*活锁*。

### 死锁

让我们考虑以下场景：为了执行某个任务，一个线程需要访问两个资源，并且两个或更多线程不能同时访问这些资源（我们需要互斥来正确同步对所需资源的访问）。每个资源都由不同的**std::mutex**类进行同步。在这种情况下，一个线程必须先获取第一个资源互斥锁，然后获取第二个资源互斥锁，最后处理资源并释放两个互斥锁。

当两个线程尝试执行上述处理时，可能会发生类似以下情况：

*线程 1* 和 *线程 2* 需要获取两个互斥锁来执行所需的处理。*线程 1* 获取第一个互斥锁，*线程 2* 获取第二个互斥锁。然后，*线程 1* 将永远阻塞等待第二个互斥锁可用，而 *线程 2* 将永远阻塞等待第一个互斥锁可用。这被称为**死锁**，因为两个线程都将永远阻塞等待对方释放所需的互斥锁。

这是在多线程代码中最常见的问题之一。在*第十一章*中，关于调试，我们将学习如何通过检查运行（死锁）程序来发现这个问题。

### 活锁

解决死锁的一个可能方案是：当线程尝试获取锁时，它将仅阻塞有限的时间，如果仍然不成功，它将释放它可能已经获得的任何锁。

例如，*线程 1* 获得了第一个锁，*线程 2* 获得了第二个锁。经过一段时间后，*线程 1* 仍然没有获得第二个锁，因此它释放了第一个锁。*线程 2* 也可能完成等待并释放它所获得的锁（在这个例子中，是第二个锁）。

这种解决方案有时可能有效，但并不正确。想象一下这个场景：*线程 1* 获得了第一个锁和第二个锁。过了一段时间后，两个线程都释放了它们已经获得的锁，然后再次获取相同的锁。然后，线程释放锁，再次获取，如此循环。

线程无法做任何事情，除了获取锁、等待、释放锁，然后再重复同样的操作。这种情况被称为**活锁**，因为线程不仅仅是永远等待（如死锁情况），它们似乎是活跃的，不断地获取和释放锁。

对于死锁和活锁的情况，最常用的解决方案是按照一致的顺序获取锁。例如，如果一个线程需要获取两个锁，它将始终先获取第一个锁，然后获取第二个锁。锁的释放将按照相反的顺序进行（首先释放第二个锁，然后是第一个）。如果第二个线程尝试获取第一个锁，它将不得不等待直到第一个线程释放了两个锁，这样就不会发生死锁。

在本节中，我们看到了 C++ 标准库提供的互斥类。我们研究了它们的主要特性和在使用多个锁时可能遇到的问题。在下一节中，我们将看到 C++ 标准库提供的机制，以使获取和释放互斥锁更加容易。

# 通用锁管理

在上一节中，我们看到了 C++ 标准库提供的不同类型的互斥量。在本节中，我们将看到提供的类，这些类使得使用互斥量更加容易。这是通过使用不同的包装器类来实现的。以下表格总结了锁管理类及其主要特性：

| **互斥量管理类** | **支持的互斥量类型** | **管理的互斥量** |
| --- | --- | --- |
| **std::lock_guard** | 所有 | 1 |
| **std::scoped_lock** | 所有 | 零个或多个 |
| **std::unique_lock** | 所有 | 1 |
| **std::shared_lock** | **std::shared_mutex** | **std::shared_timed_mutex** | 1 |

表 4.3：锁管理类及其特性

让我们看看每个互斥量管理类及其主要特性。

## std::lock_guard

**std::lock_guard** 类是一个 **资源获取即初始化** ( **RAII** ) 类，它使得使用互斥量更加容易，并保证当调用 **lock_guard** 析构函数时，互斥量将被释放。这在处理异常时非常有用，例如。

以下代码展示了 **std::lock_guard** 的使用以及它是如何使在已经获取锁的情况下处理异常变得更容易的：

```cpp
#include <format>
#include <iostream>
#include <mutex>
#include <thread>
std::mutex mtx;
uint32_t counter{};
void function_throws() { throw std::runtime_error("Error"); }
int main() {
    auto worker = [] {
        for (int i = 0; i < 1000000; ++i) {
            mtx.lock();
            counter++;
            mtx.unlock();
        }
    };
    auto worker_exceptions = [] {
        for (int i = 0; i < 1000000; ++i) {
            try {
                std::lock_guard<std::mutex> lock(mtx);
                counter++;
                function_throws();
            } catch (std::system_error& e) {
                std::cout << e.what() << std::endl;
                return;
            } catch (...) {
                return;
            }
        }
    };
    std::thread t1(worker_exceptions);
    std::thread t2(worker);
    t1.join();
    t2.join();
    std::cout << "Final counter value: " << counter << std::endl;
}
```

**function_throws()** 函数只是一个实用函数，它将抛出一个异常。

在之前的代码示例中，**worker_exceptions()** 函数由 **t1** 执行。在这种情况下，异常被处理以打印有意义的消息。锁不是显式地获取/释放。这被委托给 **lock**，一个 **std::lock_guard** 对象。当 **lock** 被构造时，它会包装互斥量并调用 **mtx.lock()**，获取锁。当 **lock** 被销毁时，互斥量将自动释放。在发生异常的情况下，互斥量也将被释放，因为 **lock** 被定义的作用域已经退出。

为 **std::lock_guard** 实现了另一个构造函数，接收一个类型为 **std::adopt_lock_t** 的参数。基本上，这个构造函数使得能够包装一个已经获取的非共享互斥量，该互斥量将在 **std::lock_guard** 析构函数中自动释放。

## std::unique_lock

**std::lock_guard** 类只是一个简单的 **std::mutex** 包装器，它在构造函数中自动获取互斥量（线程将被阻塞，等待另一个线程释放互斥量）并在析构函数中释放互斥量。这非常有用，但有时我们需要更多的控制。例如，**std::lock_guard** 将会在互斥量上调用 **lock()** 或者假设互斥量已经被获取。我们可能更喜欢或者确实需要调用 **try_lock** 。我们可能还希望 **std::mutex** 包装器在其构造函数中不获取锁；也就是说，我们可能希望在稍后的某个时刻再进行锁定。所有这些功能都是由 **std::unique_lock** 实现的。

**std::unique_lock** 构造函数接受一个标签作为其第二个参数，以指示我们想要如何处理底层的互斥量。这里有三种选项：

+   **std::defer_lock**：不获取互斥锁的所有权。构造函数中不会锁定互斥锁，如果从未获取，则析构函数中也不会解锁。

+   **std::adopt_lock**：假设互斥锁已被调用线程获取。它将在析构函数中释放。此选项也适用于**std::lock_guard**。

+   **std::try_to_lock**：尝试获取互斥锁而不阻塞。

如果我们只是将互斥锁作为唯一参数传递给**std::unique_lock**构造函数，其行为与**std::lock_guard**相同：它会阻塞直到互斥锁可用，然后获取它。它将在析构函数中释放互斥锁。

与**std::lock_guard**不同，**std::unique_lock**类允许你分别调用**lock()**和**unlock()**来获取和释放互斥锁。

## std::scoped_lock

**std::scoped_lock**类，与**std::unique_lock**一样，是一个实现 RAII 机制（记住——如果获取了互斥锁，它们将在析构函数中释放）的**std::mutex**包装器。主要区别在于，**std::unique_lock**，正如其名称所暗示的，仅包装一个互斥锁，而**std::scoped_lock**可以包装零个或多个互斥锁。此外，互斥锁的获取顺序与传递给**std::scoped_lock**构造函数的顺序相同，从而避免了死锁。

让我们看看以下代码：

```cpp
std::mutex mtx1;
std::mutex mtx2;
// Acquire both mutexes avoiding deadlock
std::scoped_lock lock(mtx1, mtx2);
// Same as doing this
// std::lock(mtx1, mtx2);
// std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
// std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
```

上述代码片段显示了我们可以非常容易地处理两个互斥锁。

## std::shared_lock

**std::shared_lock**类是另一种通用互斥锁所有权包装器。与**std::unique_lock**和**std::scoped_lock**一样，它允许延迟锁定和转移锁所有权。**std::unique_lock**和**std::shared_lock**之间的主要区别在于，后者用于以共享模式获取/释放包装的互斥锁，而前者用于以独占模式执行相同的操作。

在本节中，我们看到了互斥锁包装类及其主要功能。接下来，我们将介绍另一种同步机制：条件变量。

# 条件变量

**条件变量**是 C++标准库提供的另一种同步原语。它们允许多个线程相互通信。它们还允许多个线程等待另一个线程的通知。条件变量始终与一个互斥锁相关联。

在以下示例中，一个线程必须等待计数器等于某个特定值：

```cpp
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
int counter = 0;
int main() {
    using namespace std::chrono_literals;
    std::mutex mtx;
    std::mutex cout_mtx;
    std::condition_variable cv;
    auto increment_counter = [&] {
        for (int i = 0; i < 20; ++i) {
            std::this_thread::sleep_for(100ms);
            mtx.lock();
            ++counter;
            mtx.unlock();
            cv.notify_one();
        }
    };
    auto wait_for_counter_non_zero_mtx = [&] {
        mtx.lock();
        while (counter == 0) {
            mtx.unlock();
            std::this_thread::sleep_for(10ms);
            mtx.lock();
        }
        mtx.unlock();
        std::lock_guard<std::mutex> cout_lck(cout_mtx);
        std::cout << "Counter is non-zero" << std::endl;
    };
    auto wait_for_counter_10_cv = [&] {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck, [] { return counter == 10; });
        std::lock_guard<std::mutex> cout_lck(cout_mtx);
        std::cout << "Counter is: " << counter << std::endl;
    };
    std::thread t1(wait_for_counter_non_zero_mtx);
    std::thread t2(wait_for_counter_10_cv);
    std::thread t3(increment_counter);
    t1.join();
    t2.join();
    t3.join();
    return 0;
}
```

有两种等待特定条件的方法：一种是在循环中等待并使用互斥锁作为同步机制。这在**wait_for_counter_non_zero_mtx**函数中实现。该函数获取锁，读取**counter**中的值，然后释放锁。然后，它睡眠 10 毫秒，再次获取锁。这是在**while**循环中完成的，直到**counter**不为零。

条件变量帮助我们简化了之前的代码。**wait_for_counter_10_cv** 函数等待直到 **counter** 等于 10。线程将在 **cv** 条件变量上等待，直到它被 **t1**（在循环中增加 **counter** 的线程）通知。

**wait_for_counter_10_cv** 函数是这样工作的：一个条件变量，**cv**，在互斥锁，**mtx** 上等待。在调用 **wait()** 之后，条件变量锁定互斥锁并等待直到条件为 **true**（条件是在传递给 **wait** 函数作为第二个参数的 lambda 表达式中实现的）。如果条件不是 **true**，条件变量将保持 *等待* 状态，直到它被发出信号并释放互斥锁。一旦条件满足，条件变量结束其等待状态并再次锁定互斥锁以同步其对 **counter** 的访问。

一个重要的问题是条件变量可能被一个无关的线程发出信号。这被称为 **虚假唤醒**。为了避免由于虚假唤醒而引起的错误，条件在 **wait** 中被检查。当条件变量被发出信号时，条件再次被检查。在发生虚假唤醒且计数器为零（条件检查返回 **false** ）的情况下，等待将重新开始。

一个不同的线程通过运行 **increment_counter** 来增加计数器的值。一旦 **counter** 达到期望的值（在示例中，这个值是 10），它就会向等待的线程的条件变量发出信号。

提供了两个函数来发出条件变量信号：

+   **cv.notify_one()** : 仅向等待的线程中的一个发出信号

+   **cv.notify_all()** : 向所有等待的线程发出信号

在本节中，我们介绍了条件变量，并看到了一个使用条件变量进行同步的简单示例，以及在某些情况下它如何简化同步/等待代码。现在，让我们将注意力转向使用互斥锁和两个条件变量来实现一个同步队列。

# 实现一个线程安全的队列

在本节中，我们将看到如何实现一个简单的 **线程安全的队列**。队列将由多个线程访问，其中一些线程向其中添加元素（ **生产者线程**），而另一些线程从中移除元素（ **消费者线程**）。为了开始，我们将假设只有两个线程：一个生产者和一个消费者。

队列或**先进先出**（**FIFOs**）是线程之间通信的标准方式。例如，如果我们需要尽可能快地接收包含来自网络连接的数据的包，我们可能没有足够的时间仅在一个线程中接收所有包并处理它们。在这种情况下，我们使用第二个线程来处理第一个线程读取的包。仅使用一个消费者线程更容易同步（我们将在*第五章*中看到这一点），并且我们有保证包将被按照它们到达和被生产者线程复制到队列中的顺序进行处理。确实，包将真正按照它们被复制到队列中的顺序被读取，无论我们有多少消费者线程，但消费者线程可能被操作系统调度进和出，处理过的包的完整序列可能以不同的顺序出现。

通常，最简单的问题是一个**单生产者单消费者**（**SPSC**）队列。如果每个项目的处理成本太高，以至于仅一个线程无法处理，那么可能需要多个消费者，我们可能还有不同的数据源需要处理，并且需要多个生产者线程。本节中描述的队列将适用于所有情况。

设计队列的第一步是决定我们将使用什么数据结构来存储队列中的项目。我们希望队列包含任何类型 *T* 的元素，因此我们将它实现为一个模板类。此外，我们将限制队列的容量，以便我们可以在队列中存储的最大元素数量是固定的，并在类构造函数中设置。例如，可以使用链表并使队列无界，或者甚至使用**标准模板库**（**STL**）队列，**std::queue**，并让队列增长到任意大小。在本章中，我们将实现一个固定大小的队列。我们将在*第五章*中重新审视实现，并以非常不同的方式实现它（我们不会使用任何互斥锁或等待条件变量）。对于我们的当前实现，我们将使用 STL 向量，**std::vector<T>**，来存储队列中的项目。向量将在队列类构造函数中为所有元素分配内存，因此之后将不会有内存分配。当队列被销毁时，向量将自行销毁并释放分配的内存。这是方便的，并且简化了实现。

我们将使用向量作为**环形缓冲区**。这意味着，一旦我们在向量的末尾存储了一个元素，下一个元素将被存储在开头，因此我们将在读写元素时对这两个位置进行**循环**。

这是队列类的第一个版本，相当简单，但还没有什么用处：

```cpp
template <typename T>
class synchronized_queue {
public:
    explicit synchronized_queue(size_t size) :
        capacity_{ size }, buffer_(capacity_)
        {}
private:
    std::size_t head_{ 0 };
    std::size_t tail_{ 0 };
    std::size_t capacity_;
    std::vector<T> buffer_;
};
```

**head** 和 **tail** 变量用于指示分别读取或写入下一个元素的位置。我们还需要知道队列何时为空或满。如果队列为空，消费者线程将无法从队列中获取任何项目。如果队列已满，生产者线程将无法将任何项目放入队列中。

有不同的方式来指示队列何时为空和何时已满。在这个例子中，我们遵循以下约定：

+   如果 **tail_ == head_**，则队列是空的

+   如果 **(tail_ + 1) % capacity_ == head_**，则队列已满

另一种实现方式只需检查 **tail_ == head_** 并使用一个额外的标志来指示队列是否已满（或者使用计数器来知道队列中有多少项）。在这个例子中，我们避免使用任何额外的标志或计数器，因为标志将由消费者和生产者线程同时读写，并且我们旨在尽可能减少线程间的数据共享。此外，减少数据共享将是我们在*第五章*重新实现队列时的唯一选项。

这里有一个小问题。由于我们检查队列是否已满的方式，我们丢失了一个缓冲区槽位，因此实际容量是 **capacity_ - 1**。我们将认为队列已满，当只有一个空槽位时。由于这个原因，我们丢失了一个队列槽位（请注意，槽位将被使用，但当项目数量为 **capacity_ - 1** 时，队列仍然会显示为满）。通常情况下，这不是一个问题。

我们将要实现的队列是一个有界队列（固定大小），实现为一个环形缓冲区。

这里还有一个需要考虑的细节：**head_ + 1** 必须考虑到我们将索引回绕到缓冲区（它是一个环形缓冲区）。因此，我们必须做 **(head_ + 1) % capacity_**。模运算符计算索引值除以队列容量的余数。

以下代码展示了作为同步队列中的辅助函数实现的基本实用函数：

```cpp
template <typename T>
class synchronized_queue {
public:
    explicit synchronized_queue(size_t size) :
        capacity_{ size }, buffer_(capacity_) {
    }
private:
    std::size_t next(std::size_t index) {
        return (index + 1)% capacity_;
    }
    bool is_full() const {
        return next(tail_) == head_;
    }
    bool is_empty() const {
        return tail_ == head_;
    }
    std::size_t head_{ 0 };
    std::size_t tail_{ 0 };
    std::size_t capacity_;
    std::vector<T> buffer_;
};
```

我们实现了一些有用的函数来更新环形缓冲区的头和尾，并检查缓冲区是否已满或为空。现在，我们可以开始实现队列功能。

完整队列实现的代码位于本书的配套 GitHub 仓库中。*在这里，我们只展示重要的部分，以简化内容并专注于队列实现的同步方面*。

队列的接口具有以下两个功能：

```cpp
void push(const T& item);
void pop(T& item);
```

**push** 函数用于在队列中插入一个元素，而 **pop** 函数用于从队列中获取一个元素。

让我们从 **push** 开始。它将一个项目插入到队列中。如果队列已满，**push** 将等待直到队列至少有一个空槽（消费者从队列中移除了一个元素）。这样，生产者线程将阻塞，直到队列至少有一个空槽（满足非满条件）。

在本章前面我们已经看到，存在一种称为条件变量的同步机制，它正是这样做的。**push**函数将检查条件是否满足，如果满足，它将在队列中插入一个项目。如果条件不满足，与条件变量关联的锁将被释放，线程将等待在条件变量上，直到条件得到满足。

条件变量可能只是等待直到锁被释放。我们仍然需要检查队列是否已满，因为条件变量可能因为虚假唤醒而结束等待。这种情况发生在条件变量接收到一个通知，而这个通知并非由任何其他线程明确发送时。

我们向队列类添加以下三个成员变量：

```cpp
std::mutex mtx_;
std::condition_variable not_full_;
Std::condition_variable not_empty_;
```

我们需要两个条件变量——一个用于通知消费者队列不为满（**not_full_**），另一个用于通知生产者队列不为空（**not_empty_**）。

这是实现**push**的代码：

```cpp
void push(const T& item) {
    std::unique_lock<std::mutex> lock(mtx_);
    not_full_.wait(lock, [this]{ return !is_full(); });
    buffer_[tail_] = T;
    tail_ = increment(tail_);
    lock.unlock();
    not_empty_.notify_one();
}
```

让我们考虑一个只有一个生产者和一个消费者的场景。我们稍后会看到**pop**函数，但作为提前了解，它也同步于互斥锁/条件变量。两个线程同时尝试访问队列——生产者在插入元素时，消费者在移除元素时。

假设消费者首先获取锁。这发生在**[1]**处。条件变量需要**std::unique_lock**的使用来使用互斥锁。在**[2]**中，我们等待在条件变量上，直到**wait**函数谓词中的条件得到满足。如果条件不满足，锁将被释放，以便消费者线程能够访问队列。

一旦条件满足，锁再次被获取，队列在**[3]**处更新。更新队列后，**[4]**释放锁，然后**[5]**通知一个可能正在等待**not_empty**的消费者线程，队列现在实际上不为空。

**std::unique_lock**类可以在其析构函数中释放互斥锁，但我们需要在**[4]**处释放它，因为我们不希望在通知条件变量后释放锁。

**pop()**函数遵循类似的逻辑，如下面的代码所示：

```cpp
void pop(T& item)
{
    std::unique_lock<std::mutex> lock(mtx_);
    not_empty_.wait(lock, [this]{return !is_empty()});
    item = buffer_[head_];
    head_ = increment(head_);
    lock.unlock();
    not_full_.notify_one();
}
```

代码与**push**函数中的代码非常相似。**[1]**创建了使用**not_empty_**条件变量所需的**std::unique_lock**类。**[2]**在**not_empty_**上等待，直到它被通知队列不为空。**[3]**从队列中读取项目，将其分配给**item**变量，然后在**[4]**中释放锁。最后，在**[5]**中，通知**not_full_**条件变量，向消费者指示队列不为满。

**push** 和 **pop** 函数都是阻塞的，分别等待队列不满或不满。我们可能需要在无法插入或从队列中获取/发送消息的情况下让线程继续运行——例如，让它执行一些独立处理——然后再次尝试访问队列。

**try_push** 函数正是如此。如果互斥锁可以获取并且队列未满，那么功能与 **push** 函数相同，但在此情况下，**try_push** 不需要使用任何条件变量进行同步（但必须通知消费者）。这是 **try_push** 的代码：

```cpp
bool try_push(const T& item) {
    std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
    if (!lock || is_full()) {
        return false;
    }
    buffer_[tail_] = item;
    tail_ = next(tail_);
    lock.unlock();
    not_empty_.notify_one();
    return true;
}
```

代码是这样工作的：**[1]** 尝试获取锁并返回，而不阻塞调用线程。如果锁已经被获取，那么它将评估为 **false**。在 **[2]** 中，如果锁尚未获取或队列已满，**try_push** 返回 **false** 以指示调用者没有在队列中插入任何项，并将等待/阻塞委托给调用者。请注意，**[3]** 返回 **false** 并且函数终止。如果锁已被获取，它将在函数退出和 **std::unique_lock** 析构函数被调用时释放。

在获取锁并检查队列未满之后，然后将项插入队列，并更新 **tail_**。在 **[5]** 中，释放锁，在 **[6]** 中，通知消费者队列不再为空。这种通知是必需的，因为消费者可能会调用 **pop** 而不是 **try_pop**。

最后，函数返回 **true** 以指示调用者项已成功插入队列。

下面的代码显示了相应的 **try_pop** 函数。作为一个练习，尝试理解它是如何工作的：

```cpp
bool try_pop(T& item) {
     std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
     if (!lock || is_empty()) {
         return false;
     }
     item = buffer_[head_];
     head_ = next(head_);
     lock.unlock();
     not_empty_.notify_one();
     return true;
 }
```

这是本节中实现队列的完整代码：

```cpp
#pragma once
#include <condition_variable>
#include <mutex>
#include <vector>
namespace async_prog {
template <typename T>
class queue {
public:
    queue(std::size_t capacity) : capacity_{capacity}, buffer_(capacity) {}
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [this] { return !is_full(); });
        buffer_[tail_] = item;
        tail_ = next(tail_);
        lock.unlock();
        not_empty_.notify_one();
    }
    bool try_push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
        if (!lock || is_full()) {
            return false;
        }
        buffer_[tail_] = item;
        tail_ = next(tail_);
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }
    void pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this] { return !is_empty(); });
        item = buffer_[head_];
        head_ = next(head_);
        lock.unlock();
        not_full_.notify_one();
    }
    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
        if (!lock || is_empty()) {
            return false;
        }
        item = buffer_[head_];
        head_ = next(head_);
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }
private:
    [[nodiscard]] std::size_t next(std::size_t idx) const noexcept {
        return ((idx + 1) % capacity_);
    }
    [[nodiscard]] bool is_empty() const noexcept { return (head_ == tail_); }
    [[nodiscard]] bool is_full() const noexcept { return (next(tail_) == head_); }
   private:
    std::mutex mtx_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::size_t head_{0};
    std::size_t tail_{0};
    std::size_t capacity_;
    std::vector<T> buffer_;
};
}
```

在本节中，我们介绍了条件变量，并实现了一个与互斥锁和两个条件变量同步的基本队列，这是自 C++11 以来 C++ 标准库提供的两种基本同步原语。

队列示例展示了如何使用这些同步原语实现同步，并且可以用作更复杂工具（例如线程池）的基本构建块。

# 信号量

C++20 引入了新的同步原语来编写多线程应用程序。在本节中，我们将查看信号量。

**信号量**是一个管理可用于访问共享资源许可数的计数器。信号量可以分为两大类：

+   **二进制信号量**就像互斥锁。它只有两种状态：0 和 1。尽管二进制信号量在概念上类似于互斥锁，但二进制信号量和互斥锁之间有一些差异，我们将在本节后面看到。

+   **计数信号量**可以具有大于 1 的值，并用于控制对具有有限实例数的资源的访问。

C++20 实现了二进制和计数信号量。

## 二进制信号量

二进制信号量是一种同步原语，可用于控制对共享资源的访问。它有两个状态：0 和 1。值为 0 的信号量表示资源不可用，而值为 1 的信号量表示资源可用。

二进制信号量可用于实现互斥。这是通过使用二进制信号量来控制对资源的访问来实现的。当线程想要访问资源时，它首先检查信号量。如果信号量为 1，则线程可以访问资源。如果信号量为 0，则线程必须等待信号量变为 1，然后才能访问资源。

锁和信号量之间最显著的区别是，锁具有独占所有权，而二进制信号量则没有。只有拥有锁的线程可以释放它。信号量可以被任何线程发出信号。锁是一个临界区的锁定机制，而信号量更像是一个信号机制。在这方面，信号量比锁更接近条件变量。因此，信号量通常用于信号而不是互斥。

在 C++20 中，**std::binary_semaphore** 是 **std::counting_semaphore** 特化的别名，其 **LeastMaxValue** 为 1。

二进制信号量必须初始化为 1 或 0，例如：

```cpp
std::binary_semaphore sm1{ 0 };
std::binary_semaphore sm2{ 1 };
```

如果初始值为 **0**，获取信号量将阻塞尝试获取它的线程，并且必须在另一个线程释放它之后才能获取。获取信号量会减少计数器，而释放信号量会增加计数器。如前所述，如果计数器为 **0**，并且一个线程尝试获取锁（信号量），则该线程将被阻塞，直到信号量计数器大于 **0**。

## 计数信号量

计数信号量允许多个线程访问共享资源。计数器可以初始化为任意数值，每次线程获取信号量时，计数器将减少。作为使用计数信号量的示例，我们将修改上一节中实现的线程安全队列，并使用信号量而不是条件变量来同步对队列的访问。

新类的成员变量如下：

```cpp
template <typename T>
class queue {
 // public methods and private helper methods
private:
    std::counting_semaphore<> sem_empty_;
    std::counting_semaphore<> sem_full_;
    std::size_t head_{ 0 };
    std::size_t tail_{ 0 };
    std::size_t capacity_;
    std::vector<T> buffer_;
};
```

我们仍然需要 **head_** 和 **tail_** 来确定读取和写入元素的位置，**capacity_** 用于索引的回绕，以及 **buffer_** ，一个 **std::vector<T>** 向量。但到目前为止，我们并没有使用互斥锁，而是将使用计数信号量代替条件变量。我们将使用两个信号量：**sem_empty_** 用于计算缓冲区中的空槽位（初始设置为 **capacity_**），而 **sem_full_** 用于计算缓冲区中的非空槽位，初始设置为 0。

现在，让我们看看如何实现 **push** 函数，它是用于在队列中插入项目的函数。

在 **[1]** 中，**sem_empty_** 被获取，减少了信号量计数器。如果队列已满，则线程将阻塞，直到另一个线程通过释放（信号）**sem_empty_** 来解除阻塞。如果队列未满，则项目将被复制到缓冲区，并在 **[2]** 和 **[3]** 中更新 **tail_** 。最后，在 **[4]** 中释放 **sem_full_**，向另一个线程发出信号，表明队列不为空，且缓冲区中至少有一个项目：

```cpp
void push(const T& item) {
    sem_empty_.acquire();
    buffer_[tail_] = item;
    tail_ = next(tail_);
    sem_full_.release();
}
```

**pop** 函数用于从队列中获取元素：

```cpp
void pop(T& item) {
    sem_full_.acquire();
    item = buffer_[head_];
    head_ = next(head_);
    sem_empty_.release();
}
```

在这里，在 **[1]** 中，如果队列不为空，我们成功获取了 **sem_full_**。然后，读取项目并在 **[2]** 和 **[3]** 中分别更新 **head_**。最后，我们向消费者线程发出信号，表明队列不为空，释放 **sem_empty**。

在我们的 **push** 的第一个版本中存在几个问题。第一个也是最重要的问题是 **sem_empty_** 允许多个线程访问队列中的临界区（**[2]** 和 **[3]**）。我们需要同步这个临界区并使用互斥锁。

这里是使用互斥锁进行同步的 **push** 的新版本。

在 **[2]** 中，获取了锁（使用 **std::unique_lock**），在 **[5]** 中释放了锁。使用锁将同步临界区，防止多个线程同时访问它，并更新队列而没有任何同步：

```cpp
void push(const T& item)
{
    sem_empty_.acquire();
    std::unique_lock<std::mutex> lock(mtx_);
    buffer_[tail_] = item;
    tail_ = next(tail_);
    lock.unlock();
    sem_full_.release();
}
```

第二个问题是获取信号量是阻塞的，正如我们之前所看到的，有时调用线程可以做一些处理，而不仅仅是等待。**try_push** 函数（及其对应的 **try_pop** 函数）实现了这一功能。让我们研究一下 **try_push** 的代码。请注意，**try_push** 可能仍然会在互斥锁上阻塞：

```cpp
bool try_push(const T& item) {
    if (!sem_empty_.try acquire()) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mtx_);
    buffer_[tail_] = item;
    tail_ = next(tail_);
    lock.unlock();
    sem_full_.release();
    return true;
}
```

唯一的变化是 **[1]** 和 **[2]**。在获取信号量时，我们只是尝试获取它，如果失败，则返回 **false**。**try_acquire** 函数可能会意外失败并返回 **false**，即使信号量可以被获取（计数不是零）。

这里是使用信号量同步的队列的完整代码。

```cpp
#pragma once
#include <mutex>
#include <semaphore>
#include <vector>
namespace async_prog {
template <typename T>
class semaphore_queue {
   public:
    semaphore_queue(std::size_t capacity)
        : sem_empty_(capacity), sem_full_(0), capacity_{capacity}, buffer_(capacity)
    {}
    void push(const T& item) {
        sem_empty_.acquire();
        std::unique_lock<std::mutex> lock(mtx_);
        buffer_[tail_] = item;
        tail_ = next(tail_);
        lock.unlock();
        sem_full_.release();
    }
    bool try_push(const T& item) {
        if (!sem_empty_.try_acquire()) {
            return false;
        }
        std::unique_lock<std::mutex> lock(mtx_);
        buffer_[tail_] = item;
        tail_ = next(tail_);
        lock.unlock();
        sem_full_.release();
        return true;
    }
    void pop(T& item) {
        sem_full_.acquire();
        std::unique_lock<std::mutex> lock(mtx_);
        item = buffer_[head_];
        head_ = next(head_);
        lock.unlock();
        sem_empty_.release();
    }
    bool try_pop(T& item) {
        if (!sem_full_.try_acquire()) {
            return false;
        }
        std::unique_lock<std::mutex> lock(mtx_);
        item = buffer_[head_];
        head_ = next(head_);
        lock.unlock();
        sem_empty_.release();
        return true;
    }
private:
    [[nodiscard]] std::size_t next(std::size_t idx) const noexcept {
        return ((idx + 1) % capacity_);
    }
private:
    std::mutex mtx_;
    std::counting_semaphore<> sem_empty_;
    std::counting_semaphore<> sem_full_;
    std::size_t head_{0};
    std::size_t tail_{0};
    std::size_t capacity_;
    std::vector<T> buffer_;
};
```

在本节中，我们看到了信号量，这是自 C++20 以来包含在 C++ 标准库中的一个新的同步原语。我们学习了如何使用它们来实现我们之前实现的相同队列，但使用信号量作为同步原语。

在下一节中，我们将介绍 **屏障** 和 **闩锁**，这是自 C++20 以来包含在 C++ 标准库中的两个新的同步机制。

# 屏障和闩锁

在本节中，我们将介绍屏障和闩锁，这是 C++20 中引入的两个新的同步原语。这些机制允许线程相互等待，从而协调并发任务的执行。

## std::latch

**std::latch** 闩锁是一种同步原语，允许一个或多个线程阻塞，直到指定的操作数量完成。它是一个一次性对象，一旦计数达到零，就不能重置。

以下示例是 latch 在多线程应用程序中使用的简单说明。我们想要编写一个函数，将向量的每个元素乘以二，然后添加向量的所有元素。我们将使用三个线程将向量元素乘以二，然后使用一个线程添加向量的所有元素并获取结果。

我们需要两个闩锁。第一个闩锁将由每个乘以两个向量元素的三个线程递减。添加线程将等待此闩锁为零。然后，主线程将在第二个闩锁上等待以同步打印添加所有向量元素的结果。我们也可以等待执行加法操作的线程调用 **join**，但这也可以使用闩锁来完成。

现在，让我们分析代码的功能块。我们将在本节后面包含闩锁和屏障示例的完整代码：

```cpp
std::latch map_latch{ 3 };
auto map_thread = & {
    for (int i = start; i < end; ++i) {
        numbers[i] *= 2;
    }
    map_latch.count_down();
};
```

每个乘法线程将运行此 lambda 函数，乘以向量中一定范围内的两个元素（从 **start** 到 **end**）。一旦线程完成，它将递减 **map_latch** 计数器一次。一旦所有线程完成其任务，闩锁计数器将为零，等待在 **map_latch** 上的线程将能够继续并添加向量的所有元素。请注意，线程访问向量的不同元素，因此我们不需要同步对向量本身的访问，但我们不能开始添加数字，直到所有乘法完成。

添加线程的代码如下：

```cpp
std::latch reduce_latch{ 1 };
auto reduce_thread = & {
    map_latch.wait();
    sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    reduce_latch.count_down();
};
```

此线程将等待直到 **map_latch** 计数器降至零，然后添加向量的所有元素，并最终递减 **reduce_latch** 计数器（它将降至零），以便主线程能够打印最终结果：

```cpp
reduce_latch.wait();
std::cout << "All threads finished. The sum is: " << sum << '\n';
```

在了解了闩锁的基本应用之后，接下来，让我们学习关于屏障的内容。

## std::barrier

**std::barrier** 屏障是另一种用于同步一组线程的同步原语。**std::barrier** 屏障是可重用的。每个线程达到屏障并等待，直到所有参与线程达到相同的屏障点（就像我们使用闩锁时发生的情况）。

**std::barrier**和**std::latch**之间的主要区别是重置能力。**std::latch**是一个单次使用的 barrier，具有计数器机制，不能重置。一旦它达到零，它就会保持在零。相比之下，**std::barrier**是可重用的。所有线程都达到 barrier 后，它会重置，允许同一组线程在同一个 barrier 上多次同步。

何时使用 latches 和何时使用 barriers？当您有一个线程的一次性聚集点时，使用**std::latch**，例如在等待多个初始化完成后再继续之前。当您需要通过任务的多个阶段或迭代计算反复同步线程时，使用**std::barrier**。

我们现在将重写之前的示例，这次使用 barriers 而不是 latches。每个线程将乘以二其对应的向量元素的范围，然后将其相加。在这个例子中，主线程将使用**join()**等待处理完成，然后添加每个线程获得的结果。

工作线程的代码如下：

```cpp
std::barrier map_barrier{ 3 };
auto worker_thread = & {
    std::cout << std::format("Thread {0} is starting...\n", id);
    for (int i = start; i < end; ++i) {
        numbers[i] *= 2;
    }
    map_barrier.arrive_and_wait();
    for (int i = start; i < end; ++i) {
        sum[id] += numbers[i];
    }
    map_barrier.arrive();
};
```

代码通过一个 barrier 进行同步。当一个工作线程完成乘法运算后，它会减少**map_barrier**计数器，并等待 barrier 计数器变为零。一旦它降到零，线程结束等待并开始进行加法运算。barrier 计数器被重置，其值再次等于三。一旦加法完成，barrier 计数器再次减少，但这次线程不会等待，因为他们的任务已经完成。

当然——每个线程都可以先进行加法运算，然后再乘以二。它们不需要互相等待，因为任何线程完成的工作都不依赖于其他线程完成的工作，但这是一个很好的方法来解释 barriers 是如何通过一个简单的例子来工作的。

主线程只是通过**join**等待工作线程完成，然后打印结果：

```cpp
for (auto& t : workers) {
    t.join();
}
std::cout << std::format("The total sum is {0}\n",
                         std::accumulate(sum.begin(), sum. End(), 0));
```

这里是 latches 和 barriers 示例的完整代码：

```cpp
#include <algorithm>
#include <barrier>
#include <format>
#include <iostream>
#include <latch>
#include <numeric>
#include <thread>
#include <vector>
void multiply_add_latch() {
    const int NUM_THREADS{3};
    std::latch map_latch{NUM_THREADS};
    std::latch reduce_latch{1};
    std::vector<int> numbers(3000);
    int sum{};
    std::iota(numbers.begin(), numbers.end(), 0);
    auto map_thread = & {
        for (int i = start; i < end; ++i) {
            numbers[i] *= 2;
        }
        map_latch.count_down();
    };
    auto reduce_thread = & {
        map_latch.wait();
        sum = std::accumulate(numbers.begin(), numbers.end(), 0);
        reduce_latch.count_down();
    };
    for (int i = 0; i < NUM_THREADS; ++i) {
        std::jthread t(map_thread, std::ref(numbers), 1000 * i, 1000 * (i + 1));
    }
    std::jthread t(reduce_thread, numbers, std::ref(sum));
    reduce_latch.wait();
    std::cout << "All threads finished. The total sum is: " << sum << '\n';
}
void multiply_add_barrier() {
    const int NUM_THREADS{3};
    std::vector<int> sum(3, 0);
    std::vector<int> numbers(3000);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::barrier map_barrier{NUM_THREADS};
    auto worker_thread = & {
        std::cout << std::format("Thread {0} is starting...\n", id);
        for (int i = start; i < end; ++i) {
            numbers[i] *= 2;
        }
        map_barrier.arrive_and_wait();
        for (int i = start; i < end; ++i) {
            sum[id] += numbers[i];
        }
        map_barrier.arrive();
    };
    std::vector<std::jthread> workers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        workers.emplace_back(worker_thread, std::ref(numbers), 1000 * i,
                             1000 * (i + 1), i);
    }
    for (auto& t : workers) {
        t.join();
    }
    std::cout << std::format("All threads finished. The total sum is: {0}\n",
     std::accumulate(sum.begin(), sum.end(), 0));
}
int main() {
    std::cout << "Multiplying and reducing vector using barriers..." << std::endl;
    multiply_add_barrier();
    std::cout << "Multiplying and reducing vector using latches..." << std::endl;
    multiply_add_latch();
    return 0;
}
```

在本节中，我们看到了 barriers 和 latches。虽然它们不像 mutexes、condition variables 和 semaphores 那样常用，但了解它们总是有用的。这里提供的简单示例展示了 barriers 和 latches 的常见用法：同步在不同阶段执行处理的线程。

最后，我们将看到一个机制，即使代码从不同的线程中被多次调用，也能只执行一次。

# 只执行一次任务

有时候，我们只需要执行某个任务一次。例如，在一个多线程应用程序中，几个线程可能运行相同的函数来初始化一个变量。任何正在运行的线程都可以这样做，但我们希望初始化恰好只进行一次。

C++标准库提供了**std::once_flag**和**std::call_once**来实现这一功能。我们将在下一章中看到如何使用原子操作来实现这一功能。

以下示例将帮助我们理解如何使用**std::once_flag**和**std::call_once**在多个线程尝试执行同一任务时仅执行一次任务：

```cpp
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>
int main() {
    std::once_flag run_once_flag;
    std::once_flag run_once_exceptions_flag;
    auto thread_function = [&] {
        std::call_once(run_once_flag, []{
            std::cout << "This must run just once\n";
        });
    };
    std::jthread t1(thread_function);
    std::jthread t2(thread_function);
    std::jthread t3(thread_function);
    auto function_throws = & {
        if (throw_exception) {
            std::cout << "Throwing exception\n";
            throw std::runtime_error("runtime error");
        }
        std::cout << "No exception was thrown\n";
    };
    auto thread_function_1 = & {
        try {
            std::call_once(run_once_exceptions_flag,
                           function_throws,
                           throw_exception);
        }
        catch (...) {
        }
    };
    std::jthread t4(thread_function_1, true);
    std::jthread t5(thread_function_1, true);
    std::jthread t6(thread_function_1, false);
    return 0;
}
```

在示例的第一部分，三个线程**t1**、**t2**和**t3**运行**thread_function**函数。这个函数从**std::call_once**调用一个 lambda 表达式。如果您运行此示例，您将看到预期的消息**This must run just once**只打印一次。

在示例的第二部分，再次，三个线程**t4**、**t5**和**t6**运行**thread_function_1**函数。这个函数调用**function_throws**，该函数根据一个参数可能抛出异常或不抛出异常。此代码表明，如果从**std::call_once**调用的函数没有成功终止，则它不算作完成，并且应该再次调用**std::call_once**。只有成功的函数才算作运行函数。

本节最后展示了一种简单的机制，我们可以用它来确保即使函数被从同一线程或不同线程多次调用，该函数也只被执行一次。

# 摘要

在本章中，我们学习了如何使用 C++标准库提供的基于锁的同步原语。

我们从对竞争条件和互斥需求进行解释开始。然后，我们研究了**std::mutex**及其如何用于解决竞争条件。我们还了解了使用锁进行同步时出现的主要问题：死锁和活锁。

在学习了解锁之后，我们研究了条件变量，并使用互斥锁和条件变量实现了一个同步队列。最后，我们看到了 C++20 中引入的新同步原语：信号量、闩锁和屏障。

最后，我们研究了 C++标准库提供的机制，以运行一个函数仅一次。

在本章中，我们学习了线程同步的基本构建块以及多线程异步编程的基础。基于锁的线程同步是同步线程最常用的方法。

在下一章中，我们将研究无锁线程同步。我们将从回顾 C++20 标准库提供的原子性、原子操作和原子类型开始。我们将展示一个无锁的单生产者单消费者队列的实现。我们还将介绍 C++内存模型。

# 进一步阅读

+   大卫·R·布滕霍夫，《使用 POSIX 线程编程》，Addison Wesley，1997。

+   安东尼·威廉姆斯，《C++并发实战》，第二版，Manning，2019。
