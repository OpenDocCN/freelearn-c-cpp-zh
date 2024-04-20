# 第十四章：本地 C++线程和原语

从 2011 年的 C++标准修订版开始，多线程 API 正式成为 C++**标准模板库**（**STL**）的一部分。这意味着线程、线程原语和同步机制对于任何新的 C++应用程序都是可用的，无需安装第三方库或依赖操作系统的 API。

本章将介绍本地 API 中可用的多线程功能，直到 2014 年标准添加的功能。将展示一些示例以详细使用这些功能。

本章的主题包括以下内容：

+   C++ STL 中的多线程 API 提供的功能

+   每个功能的详细使用示例

# STL 线程 API

在第十章中，*C++多线程 API*，我们看了一下在开发多线程 C++应用程序时可用的各种 API。在第十一章中，*线程同步和通信*，我们使用本地 C++线程 API 实现了一个多线程调度程序应用程序。

# Boost.Thread API

通过包含 STL 中的`<thread>`头文件，我们可以访问`std::thread`类，该类具有由其他头文件提供的互斥（互斥锁等）设施。这个 API 本质上与`Boost.Thread`的多线程 API 相同，主要区别在于对线程的更多控制（带超时的加入，线程组和线程中断），以及在原语（如互斥锁和条件变量）之上实现的一些额外的锁类型。

一般来说，当 C++11 支持不可用时，或者这些额外的`Boost.Thread`功能是应用程序的要求，并且不容易以其他方式添加时，应该使用`Boost.Thread`作为备用。由于`Boost.Thread`建立在可用的（本地）线程支持之上，因此与 C++11 STL 实现相比，它还可能增加开销。

# 2011 年标准

C++标准的 2011 年修订版（通常称为 C++11）增加了许多新功能，其中最关键的是添加了本地多线程支持，这增加了在 C++中创建、管理和使用线程的能力，而无需使用第三方库。

这个标准为核心语言规范了内存模型，允许多个线程共存，并启用了诸如线程本地存储之类的功能。C++03 标准中已经添加了初始支持，但 C++11 标准是第一个充分利用这一特性的标准。

如前所述，实际的线程 API 本身是在 STL 中实现的。C++11（C++0x）标准的一个目标是尽可能多地将新功能放入 STL 中，而不是作为核心语言的一部分。因此，为了使用线程、互斥锁等，必须首先包含相关的 STL 头文件。

负责新多线程 API 的标准委员会各自设定了自己的目标，因此一些希望加入的功能最终未能成为标准的一部分。这包括终止另一个线程或线程取消等功能，这些功能受到 POSIX 代表的强烈反对，因为取消线程可能会导致正在销毁的线程资源清理出现问题。

以下是此 API 实现提供的功能：

+   `std::thread`

+   `std::mutex`

+   `std::recursive_mutex`

+   `std::condition_variable`

+   `std::condition_variable_any`

+   `std::lock_guard`

+   `std::unique_lock`

+   `std::packaged_task`

+   `std::async`

+   `std::future`

接下来，我们将详细介绍每个功能的示例。首先，我们将看看 C++标准的下一个修订版本添加了哪些初始功能。

# C++14

2014 年的标准向标准库添加了以下功能：

+   `std::shared_lock`

+   `std::shared_timed_mutex`

这两者都在`<shared_mutex>`STL 头文件中定义。由于锁是基于互斥锁的，因此共享锁依赖于共享互斥锁。

# 线程类

`thread`类是整个线程 API 的核心；它包装了底层操作系统的线程，并提供了我们启动和停止线程所需的功能。

通过包含`<thread>`头文件，可以访问此功能。

# 基本用法

创建线程后立即启动：

```cpp
#include <thread> 

void worker() { 
   // Business logic. 
} 

int main () { 
   std::thread t(worker);
   return 0; 
} 
```

上述代码将启动线程，然后立即终止应用程序，因为我们没有等待新线程执行完毕。

为了正确执行这个操作，我们需要等待线程完成，或者重新加入如下：

```cpp
#include <thread> 

void worker() { 
   // Business logic. 
} 

int main () { 
   std::thread t(worker); 
   t.join(); 
   return 0; 
} 
```

这段代码将执行，等待新线程完成，然后返回。

# 传递参数

也可以向新线程传递参数。这些参数值必须是可移动构造的，这意味着它是一个具有移动或复制构造函数（用于右值引用）的类型。实际上，对于所有基本类型和大多数（用户定义的）类来说，这是成立的：

```cpp
#include <thread> 
#include <string> 

void worker(int n, std::string t) { 
   // Business logic. 
} 

int main () { 
   std::string s = "Test"; 
   int i = 1; 
   std::thread t(worker, i, s); 
   t.join(); 
   return 0; 
} 
```

在上述代码中，我们将一个整数和一个字符串传递给`thread`函数。该函数将接收这两个变量的副本。当传递引用或指针时，生命周期问题、数据竞争等会变得更加复杂，可能会成为一个问题。

# 返回值

传递给`thread`类构造函数的函数返回的任何值都将被忽略。要将信息返回给创建新线程的线程，必须使用线程间同步机制（如互斥锁）和某种共享变量。

# 移动线程

2011 年的标准在`<utility>`头文件中添加了`std::move`。使用这个模板方法，可以在对象之间移动资源。这意味着它也可以移动线程实例：

```cpp
#include <thread> 
#include <string> 
#include <utility> 

void worker(int n, string t) { 
   // Business logic. 
} 

int main () { 
   std::string s = "Test"; 
   std::thread t0(worker, 1, s); 
   std::thread t1(std::move(t0)); 
   t1.join(); 
   return 0; 
} 
```

在这个版本的代码中，我们在将线程移动到另一个线程之前创建了一个线程。因此线程 0 停止存在（因为它立即完成），并且`thread`函数的执行在我们创建的新线程中恢复。

因此，我们不必等待第一个线程重新加入，只需要等待第二个线程。

# 线程 ID

每个线程都有一个与之关联的标识符。这个 ID 或句柄是 STL 实现提供的唯一标识符。可以通过调用`thread`类实例的`get_id()`函数或调用`std::this_thread::get_id()`来获取调用该函数的线程的 ID：

```cpp
#include <iostream>
 #include <thread>
 #include <chrono>
 #include <mutex>

 std::mutex display_mutex;

 void worker() {
     std::thread::id this_id = std::this_thread::get_id();

     display_mutex.lock();
     std::cout << "thread " << this_id << " sleeping...n";
     display_mutex.unlock();

     std::this_thread::sleep_for(std::chrono::seconds(1));
 }

 int main() {
    std::thread t1(worker);
    std::thread::id t1_id = t1.get_id();

    std::thread t2(worker);
    std::thread::id t2_id = t2.get_id();

    display_mutex.lock();
    std::cout << "t1's id: " << t1_id << "n";
    std::cout << "t2's id: " << t2_id << "n";
    display_mutex.unlock();

    t1.join();
    t2.join();

    return 0;
 } 

```

这段代码将产生类似于以下的输出：

```cpp
t1's id: 2
t2's id: 3
thread 2 sleeping...
thread 3 sleeping...
```

在这里，可以看到内部线程 ID 是一个整数（`std::thread::id`类型），相对于初始线程（ID 为 1）。这类似于大多数本机线程 ID，比如 POSIX 的线程 ID。这些也可以使用`native_handle()`获得。该函数将返回底层的本机线程句柄。当希望使用 STL 实现中不可用的特定 PThread 或 Win32 线程功能时，这是特别有用的。

# 休眠

可以使用两种方法延迟执行线程（休眠）。一种是`sleep_for()`，它至少延迟指定的持续时间，但可能更长：

```cpp
#include <iostream> 
#include <chrono> 
#include <thread> 
        using namespace std::chrono_literals;

        typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint; 
int main() { 
         std::cout << "Starting sleep.n"; 

         timepoint start = std::chrono::high_resolution_clock::now(); 

         std::this_thread::sleep_for(2s); 

         timepoint end = std::chrono::high_resolution_clock::now(); 
         std::chrono::duration<double, std::milli> elapsed = end - 
         start; 
         std::cout << "Slept for: " << elapsed.count() << " msn"; 
} 
```

上述代码展示了如何休眠大约 2 秒，使用具有当前操作系统上可能的最高精度的计数器来测量确切的持续时间。

请注意，我们可以直接指定秒数，使用秒后缀。这是 C++14 添加到`<chrono>`头文件的功能。对于 C++11 版本，需要创建一个 std::chrono::seconds 的实例并将其传递给`sleep_for()`函数。

另一种方法是`sleep_until()`，它接受一个类型为`std::chrono::time_point<Clock, Duration>`的单个参数。使用这个函数，可以设置线程休眠，直到达到指定的时间点。由于操作系统的调度优先级，这个唤醒时间可能不是指定的确切时间。

# 屈服

可以告诉操作系统当前线程可以重新调度，以便其他线程可以运行。为此，可以使用`std::this_thread::yield()`函数。此函数的确切结果取决于底层操作系统实现及其调度程序。在 FIFO 调度程序的情况下，调用线程可能会被放在队列的末尾。

这是一个高度专业化的函数，具有特殊的用例。在未验证其对应用程序性能的影响之前，不应使用它。

# 分离

启动线程后，可以在线程对象上调用`detach()`。这实际上将新线程与调用线程分离，这意味着前者将在调用线程退出后继续执行。

# 交换

使用`swap()`，可以作为独立方法或作为线程实例的函数，可以交换线程对象的基础线程句柄：

```cpp
#include <iostream> 
#include <thread> 
#include <chrono> 

void worker() { 
   std::this_thread::sleep_for(std::chrono::seconds(1)); 
} 

int main() { 
         std::thread t1(worker); 
         std::thread t2(worker); 

         std::cout << "thread 1 id: " << t1.get_id() << "n"; 
         std::cout << "thread 2 id: " << t2.get_id() << "n"; 

         std::swap(t1, t2); 

         std::cout << "Swapping threads..." << "n"; 

         std::cout << "thread 1 id: " << t1.get_id() << "n"; 
         std::cout << "thread 2 id: " << t2.get_id() << "n"; 

         t1.swap(t2); 

         std::cout << "Swapping threads..." << "n"; 

         std::cout << "thread 1 id: " << t1.get_id() << "n"; 
         std::cout << "thread 2 id: " << t2.get_id() << "n"; 

         t1.join(); 
         t2.join(); 
} 
```

此代码的可能输出如下：

```cpp
thread 1 id: 2
thread 2 id: 3
Swapping threads...
thread 1 id: 3
thread 2 id: 2
Swapping threads...
thread 1 id: 2
thread 2 id: 3
```

其效果是每个线程的状态与另一个线程的状态交换，实质上交换了它们的身份。

# 互斥锁

`<mutex>`头文件包含多种类型的互斥锁和锁。互斥锁类型是最常用的类型，提供基本的锁定/解锁功能，没有更多的复杂性。

# 基本用法

在本质上，互斥锁的目标是排除同时访问的可能性，以防止数据损坏，并防止由于使用非线程安全例程而导致崩溃。

一个需要使用互斥锁的示例代码如下：

```cpp
#include <iostream> 
#include <thread> 

void worker(int i) { 
         std::cout << "Outputting this from thread number: " << i << "n"; 
} 

int main() { 
         std::thread t1(worker, 1);
         std::thread t2(worker, 2); 

         t1.join(); 
   t2.join(); 

   return 0; 
} 
```

如果一个人尝试直接运行上述代码，就会注意到两个线程的文本输出会被混在一起，而不是依次输出。原因是标准输出（无论是 C 还是 C++风格）不是线程安全的。虽然应用程序不会崩溃，但输出会是一团糟。

对此的修复很简单，如下所示：

```cpp
#include <iostream> 
#include <thread> 
#include <mutex> 

std::mutex globalMutex; 

void worker(int i) { 
   globalMutex.lock(); 
         std::cout << "Outputting this from thread number: " << i << "n"; 
   globalMutex.unlock(); 
} 

int main() { 
         std::thread t1(worker, 1);
         std::thread t2(worker, 2); 

         t1.join(); 
   t2.join(); 

   return 0; 
} 
```

在这种情况下，每个线程首先需要获取`mutex`对象的访问权。由于只有一个线程可以访问`mutex`对象，另一个线程将等待第一个线程完成对标准输出的写入，两个字符串将按预期依次出现。

# 非阻塞锁定

可能不希望线程阻塞并等待`mutex`对象可用：例如，当一个人只想知道是否另一个线程已经处理了请求，并且没有必要等待其完成时。

为此，互斥锁带有`try_lock()`函数，可以做到这一点。

在下面的示例中，我们可以看到两个线程尝试递增相同的计数器，但是当一个线程无法立即访问共享计数器时，它会递增自己的计数器：

```cpp
#include <chrono> 
#include <mutex> 
#include <thread> 
#include <iostream> 

std::chrono::milliseconds interval(50); 

std::mutex mutex; 
int shared_counter = 0;
int exclusive_counter = 0; 

void worker0() { 
   std::this_thread::sleep_for(interval);

         while (true) { 
               if (mutex.try_lock()) { 
                     std::cout << "Shared (" << job_shared << ")n"; 
                     mutex.unlock(); 
                     return; 
               } 
         else { 
                     ++exclusive_counter; 
                           std::cout << "Exclusive (" << exclusive_counter << ")n"; 
                           std::this_thread::sleep_for(interval); 
               } 
         } 
} 

void worker1() { 
   mutex.lock(); 
         std::this_thread::sleep_for(10 * interval); 
         ++shared_counter; 
         mutex.unlock(); 
} 

int main() { 
         std::thread t1(worker0); 
         std::thread t2(worker1); 

         t1.join(); 
         t2.join(); 
}
```

在上述示例中，两个线程运行不同的`worker`函数，但它们都有一个共同点，即它们都会在一段时间内休眠，并在醒来时尝试获取共享计数器的互斥锁。如果成功，它们将增加计数器，但只有第一个工作线程会输出这个事实。

第一个工作线程还会记录当它没有获得共享计数器时，但只增加了它自己的独立计数器。结果输出可能看起来像这样：

```cpp
Exclusive (1)
Exclusive (2)
Exclusive (3)
Shared (1)
Exclusive (4)
```

# 定时互斥锁

定时互斥锁是常规互斥锁类型，但具有一些额外的函数，可以控制在尝试获取锁期间的时间段，即`try_lock_for`和`try_lock_until`。

前者在指定的时间段（`std::chrono`对象）内尝试获取锁，然后返回结果（true 或 false）。后者将等待直到将来的特定时间点，然后返回结果。

这些功能的使用主要在于提供常规互斥锁的阻塞（`lock`）和非阻塞（`try_lock`）方法之间的中间路径。一个人可能希望使用单个线程等待一些任务，而不知道何时任务将变为可用，或者任务可能在某个特定时间点过期，此时等待它就不再有意义了。

# 锁卫

锁卫是一个简单的互斥锁包装器，它处理对`mutex`对象的锁定以及在锁卫超出范围时的释放。这是一个有用的机制，可以确保不会忘记释放互斥锁，并且在必须在多个位置释放相同的互斥锁时，可以帮助减少代码的混乱。

尽管重构，例如大的 if/else 块可以减少需要释放互斥锁的情况，但最好还是使用这个锁卫包装器，不用担心这些细节：

```cpp
#include <thread> 
#include <mutex> 
#include <iostream> 

int counter = 0; 
std::mutex counter_mutex; 

void worker() { 
         std::lock_guard<std::mutex> lock(counter_mutex); 
   if (counter == 1) { counter += 10; } 
   else if (counter >= 10) { counter += 15; } 
   else if (counter >= 50) { return; } 
         else { ++counter; } 

   std::cout << std::this_thread::get_id() << ": " << counter << 'n'; 
} 

int main() { 
    std::cout << __func__ << ": " << counter << 'n'; 

    std::thread t1(worker); 
    std::thread t2(worker); 

    t1.join(); 
    t2.join(); 

    std::cout << __func__ << ": " << counter << 'n'; 
} 
```

在前面的例子中，我们看到一个小的 if/else 块，其中一个条件导致`worker`函数立即返回。如果没有锁卫，我们必须确保在从函数返回之前在此条件下也解锁互斥锁。

然而，有了锁卫，我们就不必担心这些细节，这使我们可以专注于业务逻辑，而不是担心互斥锁管理。

# 唯一锁

唯一锁是一个通用的互斥锁包装器。它类似于定时互斥锁，但具有附加功能，主要是所有权的概念。与其他锁类型不同，唯一锁不一定拥有它包装的互斥锁，如果有的话。互斥锁可以在唯一锁实例之间以及使用`swap()`函数转移这些互斥锁的所有权。

唯一锁实例是否拥有其互斥锁的所有权，以及它是否被锁定或未锁定，是在创建锁时首先确定的，可以从其构造函数中看到。例如：

```cpp
std::mutex m1, m2, m3; 
std::unique_lock<std::mutex> lock1(m1, std::defer_lock); 
std::unique_lock<std::mutex> lock2(m2, std::try_lock); 
std::unique_lock<std::mutex> lock3(m3, std::adopt_lock); 
```

最后一个代码中的第一个构造函数不锁定分配的互斥锁（延迟）。第二个尝试使用`try_lock()`锁定互斥锁。最后，第三个构造函数假定它已经拥有提供的互斥锁。

除此之外，其他构造函数允许定时互斥锁的功能。也就是说，它将等待一段时间，直到达到某个时间点，或者直到获得锁。

最后，使用`release()`函数可以断开锁与互斥锁之间的关联，并返回`mutex`对象的指针。然后调用者负责释放互斥锁上的任何剩余锁，并进一步处理它。

这种类型的锁通常不会单独使用，因为它非常通用。大多数其他类型的互斥锁和锁都要简单得多，并且可能在 99%的情况下满足所有需求。唯一锁的复杂性因此既是优点也是风险。

然而，它通常被 C++11 线程 API 的其他部分使用，例如我们马上就会看到的条件变量。

唯一锁可能有用的一个领域是作为作用域锁，允许使用作用域锁而不必依赖 C++17 标准中的原生作用域锁。看这个例子：

```cpp
#include <mutex>
std::mutex my_mutex
int count = 0;
int function() {
         std::unique_lock<mutex> lock(my_mutex);
   count++;
}  
```

当我们进入函数时，我们使用全局互斥锁实例创建一个新的 unique_lock。在这一点上，互斥锁被锁定，之后我们可以执行任何关键操作。

当函数作用域结束时，唯一锁的析构函数被调用，这导致互斥锁再次被解锁。

# 作用域锁

作用域锁是在 2017 年标准中首次引入的，它是一个互斥锁包装器，用于获取（锁定）提供的互斥锁，并确保在作用域锁超出范围时解锁。它与锁卫的不同之处在于它是多个互斥锁的包装器，而不是一个。

当在单个作用域中处理多个互斥时，这可能是有用的。使用作用域锁的一个原因是为了避免意外引入死锁和其他不愉快的复杂情况，例如一个互斥被作用域锁锁定，另一个锁仍在等待，另一个线程实例具有完全相反的情况。

作用域锁的一个特性是，它试图避免这种情况，从理论上讲，使得这种类型的锁具有死锁安全性。

# 递归互斥

递归互斥是互斥的另一种子类型。尽管它具有与常规互斥完全相同的功能，但它允许最初锁定互斥的调用线程重复锁定同一互斥。通过这样做，互斥在拥有线程解锁它的次数与锁定它的次数相同之前，不会对其他线程可用。

使用递归互斥的一个很好的理由是，例如在使用递归函数时。使用常规互斥时，需要发明某种进入点，在进入递归函数之前锁定互斥。

使用递归互斥时，递归函数的每次迭代都会再次锁定递归互斥，并在完成一次迭代后解锁互斥。结果是互斥锁定和解锁的次数相同。

因此，这里可能存在的一个复杂情况是，递归互斥可以被锁定的最大次数在标准中没有定义。当达到实现的限制时，如果尝试锁定它，将抛出`std::system_error`，或者在使用非阻塞的`try_lock`函数时返回 false。

# 递归定时互斥

递归定时互斥是，正如其名称所示，定时互斥和递归互斥功能的融合。因此，它允许使用定时条件函数递归锁定互斥。

尽管这增加了确保互斥锁定的次数与线程锁定次数相同的挑战，但它仍然为更复杂的算法提供了可能性，比如前面提到的任务处理程序。

# 共享互斥

`<shared_mutex>`头文件是在 2014 年标准中首次添加的，通过添加`shared_timed_mutex`类。在 2017 年标准中，还添加了`shared_mutex`类。

自 C++17 以来，共享互斥头文件一直存在。除了通常的互斥访问之外，这个`mutex`类还增加了提供互斥访问的能力。这允许多个线程对资源进行读访问，而写线程仍然可以获得独占访问。这类似于 Pthreads 的读写锁。

添加到这种互斥类型的函数如下：

+   `lock_shared()`

+   `try_lock_shared()`

+   `unlock_shared()`

这种互斥的共享功能的使用应该是相当不言自明的。理论上，无限数量的读者可以获得对互斥的读访问，同时确保只有一个线程可以随时写入资源。

# 共享定时互斥

这个头文件自 C++14 以来一直存在。它通过这些函数向定时互斥添加了共享锁定功能：

+   `lock_shared()`

+   `try_lock_shared()`

+   `try_lock_shared_for()`

+   `try_lock_shared_until()`

+   `unlock_shared()`

这个类本质上是共享互斥和定时互斥的融合，正如其名称所示。这里有趣的是，它在更基本的共享互斥之前被添加到了标准中。

# 条件变量

从本质上讲，条件变量提供了一种机制，通过这种机制，一个线程的执行可以被另一个线程控制。这是通过一个共享变量来实现的，一个线程会等待这个变量，直到被另一个线程发出信号。这是我们在第十一章中看到的调度器实现的一个基本部分，*线程同步和通信*。

对于 C++11 API，条件变量及其相关功能在`<condition_variable>`头文件中定义。

条件变量的基本用法可以从第十一章的调度器代码中总结出来，*线程同步和通信*。

```cpp
 #include "abstract_request.h"

 #include <condition_variable>
 #include <mutex> 

using namespace std;

 class Worker {
    condition_variable cv;
    mutex mtx;
    unique_lock<mutex> ulock;
    AbstractRequest* request;
    bool running;
    bool ready;
    public:
    Worker() { running = true; ready = false; ulock = unique_lock<mutex>(mtx); }
    void run();
    void stop() { running = false; }
    void setRequest(AbstractRequest* request) { this->request = request; ready = true; }
    void getCondition(condition_variable* &cv);
 }; 
```

在前面的`Worker`类声明中定义的构造函数中，我们看到了 C++11 API 中条件变量的初始化方式。步骤如下：

1.  创建`condition_variable`和`mutex`实例。

1.  将互斥锁分配给一个新的`unique_lock`实例。使用我们在这里用于锁的构造函数，分配的互斥锁也在分配时被锁定。

1.  条件变量现在可以使用了：

```cpp
#include <chrono>
using namespace std;
void Worker::run() {
    while (running) {
        if (ready) {
            ready = false;
            request->process();
            request->finish();
        }
        if (Dispatcher::addWorker(this)) {
            while (!ready && running) {
                if (cv.wait_for(ulock, chrono::seconds(1)) == 
                cv_status::timeout) {
                    // We timed out, but we keep waiting unless the 
                    worker is
                    // stopped by the dispatcher.
                }
            }
        }
    }
} 
```

在这里，我们使用条件变量的`wait_for()`函数，并传递我们之前创建的唯一锁实例和我们想要等待的时间。这里我们等待 1 秒。如果我们在这个等待中超时，我们可以自由地重新进入等待（就像这里做的那样）在一个连续的循环中，或者继续执行。

还可以使用简单的`wait()`函数执行阻塞等待，或者使用`wait_for()`等待到某个特定的时间点。

正如我们之前看到的，这个工作线程的代码使用`ready`布尔变量的原因是为了检查是否真的是另一个线程发出了条件变量的信号，而不仅仅是一个虚假的唤醒。这是大多数条件变量实现（包括 C++11）都容易受到的不幸的复杂性。

由于这些随机唤醒事件，有必要确保我们确实是有意醒来的。在调度器代码中，这是通过唤醒工作线程的线程也设置一个`Boolean`值来完成的，工作线程可以唤醒。

我们是否超时，或者被通知，或者遭受虚假唤醒，都可以通过`cv_status`枚举来检查。这个枚举知道这两种可能的情况：

+   `timeout`

+   `no_timeout`

信号或通知本身非常简单：

```cpp
void Dispatcher::addRequest(AbstractRequest* request) {
    workersMutex.lock();
    if (!workers.empty()) {
          Worker* worker = workers.front();
          worker->setRequest(request);
          condition_variable* cv;
          worker->getCondition(cv);
          cv->notify_one();
          workers.pop();
          workersMutex.unlock();
    }
    else {
          workersMutex.unlock();
          requestsMutex.lock();
          requests.push(request);
          requestsMutex.unlock();
    }
          } 
```

在`Dispatcher`类的前面的函数中，我们尝试获取一个可用的工作线程实例。如果找到，我们按如下方式获取对工作线程条件变量的引用：

```cpp
void Worker::getCondition(condition_variable* &cv) {
    cv = &(this)->cv;
 } 
```

设置工作线程上的新请求也会将`ready`变量的值更改为 true，从而允许工作线程检查它确实被允许继续。

最后，条件变量被通知，任何等待它的线程现在可以继续使用`notify_one()`。这个特定的函数将信号传递给条件变量中 FIFO 队列中的第一个线程。在这里，只有一个线程会被通知，但如果有多个线程在等待相同的条件变量，调用`notify_all()`将允许 FIFO 队列中的所有线程继续。

# Condition_variable_any

`condition_variable_any`类是`condition_variable`类的泛化。它与后者的不同之处在于它允许使用除`unique_lock<mutex>`之外的其他互斥机制。唯一的要求是所使用的锁符合`BasicLockable`的要求，这意味着它提供了`lock()`和`unlock()`函数。

# 在线程退出时通知所有

`std::notify_all_at_thread_exit()`函数允许（分离的）线程通知其他线程它已经完全完成，并且正在销毁其范围内的所有对象（线程本地）。它的功能是在发出提供的条件变量信号之前将提供的锁移动到内部存储中。

结果就像锁被解锁并且在条件变量上调用了`notify_all()`一样。

可以给出一个基本（非功能性）示例如下：

```cpp
#include <mutex> 
#include <thread> 
#include <condition_variable> 
using namespace std; 

mutex m; 
condition_variable cv;
bool ready = false; 
ThreadLocal result;

void worker() { 
   unique_lock<mutex> ulock(m); 
   result = thread_local_method(); 
         ready = true; 
         std::notify_all_at_thread_exit(cv, std::move(ulock)); 
} 

int main() { 
         thread t(worker); 
         t.detach(); 

         // Do work here. 

         unique_lock<std::mutex> ulock(m); 
         while(!ready) { 
               cv.wait(ulock); 
         } 

         // Process result 
} 
```

在这里，工作线程执行一个创建线程本地对象的方法。因此，主线程必须首先等待分离的工作线程完成。如果主线程完成任务时后者尚未完成，它将使用全局条件变量进入等待。在工作线程中，设置`ready`布尔值后，调用`std::notify_all_at_thread_exit()`。

这样做有两个目的。在调用函数后，不允许更多的线程等待条件变量。它还允许主线程等待分离的工作线程的结果变得可用。

# Future

C++11 线程支持 API 的最后一部分在`<future>`中定义。它提供了一系列类，实现了更高级的多线程概念，旨在更容易地进行异步处理，而不是实现多线程架构。

在这里，我们必须区分两个概念：`future`和`promise`。前者是最终结果（未来的产品），将被读取者/消费者使用。后者是写入者/生产者使用的。

`future`的一个基本示例是：

```cpp
#include <iostream>
#include <future>
#include <chrono>

bool is_prime (int x) {
  for (int i = 2; i < x; ++i) if (x%i==0) return false;
  return true;
}

int main () {
  std::future<bool> fut = std::async (is_prime, 444444443);
  std::cout << "Checking, please wait";
  std::chrono::milliseconds span(100);
  while (fut.wait_for(span) == std::future_status::timeout) {               std::cout << '.' << std::flush;
   }

  bool x = fut.get();
  std::cout << "n444444443 " << (x?"is":"is not") << " prime.n";
  return 0;
}
```

这段代码异步调用一个函数，传递一个参数（可能是质数）。然后它进入一个活动循环，同时等待异步函数调用返回的`future`完成。它在等待函数上设置了 100 毫秒的超时。

一旦`future`完成（在等待函数上没有超时），我们就可以获得结果值，本例中告诉我们提供给函数的值实际上是一个质数。

在本章的*async*部分，我们将更详细地看一下异步函数调用。

# Promise

`promise`允许在线程之间传输状态。例如：

```cpp
#include <iostream> 
#include <functional>
#include <thread> 
#include <future> 

void print_int (std::future<int>& fut) {
  int x = fut.get();
  std::cout << "value: " << x << 'n';
}

int main () {
  std::promise<int> prom;
  std::future<int> fut = prom.get_future();
  std::thread th1 (print_int, std::ref(fut));
  prom.set_value (10);                            
  th1.join();
  return 0;
```

上面的代码使用了传递给工作线程的`promise`实例，以将一个值传输到另一个线程，本例中是一个整数。新线程等待我们从`promise`创建的`future`完成，这个`future`是从主线程接收到的。

当我们在`promise`上设置值时，`promise`就完成了。这完成了`future`并结束了工作线程。

在这个特定的例子中，我们对`future`对象进行了阻塞等待，但也可以使用`wait_for()`和`wait_until()`，分别等待一段时间或一个时间点，就像我们在上一个例子中对`future`进行的操作一样。

# 共享 future

`shared_future`就像一个普通的`future`对象一样，但可以被复制，这允许多个线程读取其结果。

创建一个`shared_future`与创建一个普通的`future`类似。

```cpp
std::promise<void> promise1; 
std::shared_future<void> sFuture(promise1.get_future()); 
```

最大的区别是普通的`future`被传递给它的构造函数。

之后，所有可以访问`future`对象的线程都可以等待它，并获取其值。这也可以用于类似条件变量的方式来通知线程。

# 包装任务

`packaged_task`是任何可调用目标（函数、绑定、lambda 或其他函数对象）的包装器。它允许异步执行，并将结果可用于`future`对象。它类似于`std::function`，但自动将其结果传输到`future`对象。

例如：

```cpp
#include <iostream> 
#include <future> 
#include <chrono>
#include <thread>

using namespace std; 

int countdown (int from, int to) { 
   for (int i = from; i != to; --i) { 
         cout << i << 'n'; 
         this_thread::sleep_for(chrono::seconds(1)); 
   } 

   cout << "Finished countdown.n"; 
   return from - to; 
} 

int main () { 
   packaged_task<int(int, int)> task(countdown);
   future<int> result = task.get_future();
   thread t (std::move(task), 10, 0);

   //  Other logic. 

   int value = result.get(); 

   cout << "The countdown lasted for " << value << " seconds.n"; 

   t.join(); 
   return 0; 
} 
```

上面的代码实现了一个简单的倒计时功能，从 10 倒数到 0。创建任务并获取其`future`对象的引用后，我们将其推送到一个线程，同时传递`worker`函数的参数。

倒计时工作线程的结果在完成后立即可用。我们可以使用`future`对象的等待函数，方式与`promise`一样。 

# Async

`promise`和`packaged_task`的更简单的版本可以在`std::async()`中找到。这是一个简单的函数，它接受一个可调用对象（函数、绑定、lambda 等）以及它的任何参数，并返回一个`future`对象。

以下是`async()`函数的一个基本示例：

```cpp
#include <iostream>
#include <future>

using namespace std; 

bool is_prime (int x) { 
   cout << "Calculating prime...n"; 
   for (int i = 2; i < x; ++i) { 
         if (x % i == 0) { 
               return false; 
         } 
   } 

   return true; 
} 

int main () { 
   future<bool> pFuture = std::async (is_prime, 343321); 

   cout << "Checking whether 343321 is a prime number.n"; 

   // Wait for future object to be ready. 

   bool result = pFuture.get(); 
   if (result) {
         cout << "Prime found.n"; 
   } 
   else { 
         cout << "No prime found.n"; 
   } 

   return 0; 
} 
```

前面代码中的`worker`函数确定提供的整数是否为质数。正如我们所看到的，结果代码比使用`packaged_task`或`promise`要简单得多。

# 启动策略

除了`std::async()`的基本版本之外，还有第二个版本，允许将启动策略作为其第一个参数进行指定。这是一个`std::launch`类型的位掩码值，可能的取值如下：

```cpp
* launch::async 
* launch::deferred 
```

`async`标志意味着立即为`worker`函数创建一个新线程和执行上下文。`deferred`标志意味着这将被推迟，直到在`future`对象上调用`wait()`或`get()`。指定两个标志会导致函数根据当前系统情况自动选择方法。

未明确指定位掩码值的`std::async()`版本默认为后者，即自动方法。

# 原子操作

在多线程中，原子操作的使用也非常重要。C++11 STL 出于这个原因提供了一个`<atomic>`头文件。这个主题在第十五章中得到了广泛覆盖，即*原子操作-与硬件交互*。

# 总结

在本章中，我们探讨了 C++11 API 中的整个多线程支持，以及 C++14 和 C++17 中添加的特性。

我们看到了如何使用描述和示例代码来使用每个特性。现在我们可以使用本机 C++多线程 API 来实现多线程、线程安全的代码，以及使用异步执行特性来加速并并行执行函数。

在下一章中，我们将看一下多线程代码实现中不可避免的下一步：调试和验证所得应用程序。
