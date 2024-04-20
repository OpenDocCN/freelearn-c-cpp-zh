# 第十六章：最佳实践

和大多数事情一样，最好是避免犯错，而不是事后纠正。本章将介绍多线程应用程序中的一些常见错误和设计问题，并展示避免常见和不太常见问题的方法。

本章的主题包括：

+   常见的多线程问题，如死锁和数据竞争。

+   互斥锁、锁的正确使用和陷阱。

+   静态初始化时可能出现的潜在问题。

# 正确的多线程

在前面的章节中，我们已经看到了编写多线程代码时可能出现的各种潜在问题。这些问题从明显的问题，比如两个线程无法同时写入同一位置，到更微妙的问题，比如互斥锁的不正确使用。

还有许多与多线程代码直接相关的问题，但它们仍然可能导致看似随机的崩溃和其他令人沮丧的问题。其中一个例子是变量的静态初始化。在接下来的章节中，我们将看到所有这些问题以及更多问题，并介绍避免不得不处理这些问题的方法。

和生活中的许多事情一样，它们是有趣的经历，但通常你不想重复它们。

# 错误的期望 - 死锁

死锁的描述已经相当简洁了。当两个或更多进程试图访问另一个进程持有的资源，而另一个线程同时正在等待访问它持有的资源时，就会发生死锁。

例如：

1.  线程 1 获得对资源 A 的访问

1.  线程 1 和 2 都想获得对资源 B 的访问

1.  线程 2 获胜，现在拥有 B，而线程 1 仍在等待 B

1.  线程 2 现在想要使用 A，并等待访问。

1.  线程 1 和 2 都永远等待资源

在这种情况下，我们假设线程最终能够访问每个资源，而事实上却相反，因为每个线程都持有另一个线程需要的资源。

可视化，这个死锁过程会像这样：

![](img/6bda8592-427c-467e-bfc6-e9a87991853a.png)

这清楚地表明了在防止死锁时的两个基本规则：

+   尽量不要同时持有多个锁。

+   尽快释放任何持有的锁。

我们在第十一章中看到了一个现实生活中的例子，*线程同步和通信*，当我们查看调度程序演示代码时。这段代码涉及两个互斥锁，以保护对两个数据结构的访问：

```cpp
void Dispatcher::addRequest(AbstractRequest* request) {
    workersMutex.lock();
    if (!workers.empty()) {
          Worker* worker = workers.front();
          worker->setRequest(request);
          condition_variable* cv;
          mutex* mtx;
          worker->getCondition(cv);
          worker->getMutex(mtx);
          unique_lock<mutex> lock(*mtx);
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

这里的互斥锁是`workersMutex`和`requestsMutex`变量。我们可以清楚地看到，在任何时候我们都没有在尝试获取另一个互斥锁之前持有一个互斥锁。我们明确地在方法的开始处锁定`workersMutex`，这样我们就可以安全地检查工作数据结构是否为空。

如果不为空，我们将新请求交给一个工作线程。然后，当我们完成了对工作数据结构的操作后，我们释放互斥锁。此时，我们不再持有任何互斥锁。这里没有太复杂的东西，因为我们只使用了一个互斥锁。

有趣的是在 else 语句中，当没有等待的工作线程并且我们需要获取第二个互斥锁时。当我们进入这个范围时，我们保留一个互斥锁。我们可以尝试获取`requestsMutex`并假设它会起作用，但这可能会导致死锁，原因很简单：

```cpp
bool Dispatcher::addWorker(Worker* worker) {
    bool wait = true;
    requestsMutex.lock();
    if (!requests.empty()) {
          AbstractRequest* request = requests.front();
          worker->setRequest(request);
          requests.pop();
          wait = false;
          requestsMutex.unlock();
    }
    else {
          requestsMutex.unlock();
          workersMutex.lock();
          workers.push(worker);
          workersMutex.unlock();
    }
          return wait;
 } 
```

与前面的函数相配套的函数也使用了这两个互斥锁。更糟糕的是，这个函数在一个单独的线程中运行。结果，当第一个函数持有`workersMutex`并尝试获取`requestsMutex`时，第二个函数同时持有后者，并尝试获取前者时，我们就陷入了死锁。

然而，在这里我们看到的函数中，这两条规则都已成功实施；我们从不同时持有多个锁，并且尽快释放我们持有的任何锁。这可以在两个 else 情况中看到，当我们进入它们时，我们首先释放不再需要的任何锁。

在任一情况下，我们都不需要再分别检查工作线程或请求数据结构；在做其他事情之前，我们可以释放相关的锁。这导致以下可视化效果：

![](img/75aa2220-ea54-4fa5-bf1f-bc61cf4d3a68.png)

当然，我们可能需要使用两个或更多数据结构或变量中包含的数据；这些数据同时被其他线程使用。很难确保在生成的代码中没有死锁的可能性。

在这里，人们可能希望考虑使用临时变量或类似方法。通过锁定互斥量，复制相关数据，并立即释放锁，就不会出现死锁的可能性。即使必须将结果写回数据结构，也可以在单独的操作中完成。

这增加了防止死锁的两条规则：

+   尽量不要同时持有多个锁。

+   尽快释放任何持有的锁。

+   永远不要持有锁的时间超过绝对必要的时间。

+   持有多个锁时，要注意它们的顺序。

# 粗心大意 - 数据竞争

数据竞争，也称为竞争条件，发生在两个或更多线程同时尝试写入同一共享内存时。因此，每个线程执行的指令序列期间和结束时的共享内存状态在定义上是不确定的。

正如我们在第十三章中看到的，“调试多线程代码”，调试多线程应用程序的工具经常报告数据竞争。例如：

```cpp
    ==6984== Possible data race during write of size 1 at 0x5CD9260 by thread #1
 ==6984== Locks held: none
 ==6984==    at 0x40362C: Worker::stop() (worker.h:37)
 ==6984==    by 0x403184: Dispatcher::stop() (dispatcher.cpp:50)
 ==6984==    by 0x409163: main (main.cpp:70)
 ==6984== 
 ==6984== This conflicts with a previous read of size 1 by thread #2
 ==6984== Locks held: none
 ==6984==    at 0x401E0E: Worker::run() (worker.cpp:51)
 ==6984==    by 0x408FA4: void std::_Mem_fn_base<void (Worker::*)(), true>::operator()<, void>(Worker*) const (in /media/sf_Projects/Cerflet/dispatcher/dispatcher_demo)
 ==6984==    by 0x408F38: void std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::_M_invoke<0ul>(std::_Index_tuple<0ul>) (functional:1531)
 ==6984==    by 0x408E3F: std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)>::operator()() (functional:1520)
 ==6984==    by 0x408D47: std::thread::_Impl<std::_Bind_simple<std::_Mem_fn<void (Worker::*)()> (Worker*)> >::_M_run() (thread:115)
 ==6984==    by 0x4EF8C7F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
 ==6984==    by 0x4C34DB6: ??? (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
 ==6984==    by 0x53DF6B9: start_thread (pthread_create.c:333)
 ==6984==  Address 0x5cd9260 is 96 bytes inside a block of size 104 alloc'd
 ==6984==    at 0x4C2F50F: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_helgrind-amd64-linux.so)
 ==6984==    by 0x40308F: Dispatcher::init(int) (dispatcher.cpp:38)
 ==6984==    by 0x4090A0: main (main.cpp:51)
 ==6984==  Block was alloc'd by thread #1

```

生成上述警告的代码如下：

```cpp
bool Dispatcher::stop() {
    for (int i = 0; i < allWorkers.size(); ++i) {
          allWorkers[i]->stop();
    }
          cout << "Stopped workers.n";
          for (int j = 0; j < threads.size(); ++j) {
          threads[j]->join();
                      cout << "Joined threads.n";
    }
 } 
```

考虑在`Worker`实例中的这段代码：

```cpp
   void stop() { running = false; } 
```

我们还有：

```cpp
void Worker::run() {
    while (running) {
          if (ready) {
                ready = false;
                request->process();
                request->finish();
          }
                      if (Dispatcher::addWorker(this)) {
                while (!ready && running) {
                      unique_lock<mutex> ulock(mtx);
                      if (cv.wait_for(ulock, chrono::seconds(1)) == cv_status::timeout) {
                      }
                }
          }
    }
 } 
```

在这里，`running`是一个布尔变量，被设置为`false`（从一个线程写入），表示工作线程应该终止其等待循环，而读取布尔变量是从不同的进程进行的，主线程与工作线程：

![](img/c5cacd65-8eda-4ec0-b186-915b29ab3acc.png)

这个特定示例的警告是由于一个布尔变量同时被写入和读取。当然，这种特定情况之所以安全，与原子操作有关，详细解释在第八章“原子操作 - 与硬件交互”中。

即使像这样的操作潜在风险很大的原因是，读取操作可能发生在变量仍在更新过程中。例如，对于 32 位整数，根据硬件架构，更新此变量可能是一次完成，或者多次完成。在后一种情况下，读取操作可能读取一个中间值，导致结果不确定：

![](img/1ce79d19-50bf-4450-b4b5-0299d486910a.png)

更有趣的情况是，当多个线程写入一个标准输出时，例如，不使用`cout`。由于这个流不是线程安全的，结果输出流将包含输入流的片段，每当任一线程有机会写入时：

![](img/4bc9344c-37a8-4666-998a-ed75bac419fb.png)

因此，防止数据竞争的基本规则是：

+   永远不要向未锁定的、非原子的共享资源中写入

+   永远不要从未锁定的、非原子的共享资源中读取

这基本上意味着任何写入或读取都必须是线程安全的。如果一个线程写入共享内存，那么其他线程就不应该能够同时写入它。同样，当我们从共享资源中读取时，我们需要确保最多只有其他线程也在读取共享资源。

这种级别的互斥自然是由互斥锁实现的，正如我们在前面的章节中所看到的，读写锁提供了一种改进，允许同时进行读取，同时将写入作为完全互斥的事件。

当然，互斥锁也有一些陷阱，我们将在下一节中看到。

# 互斥锁并不是魔术

互斥锁构成了几乎所有形式的互斥 API 的基础。在它们的核心，它们似乎非常简单，只有一个线程可以拥有一个互斥锁，其他线程则整齐地等待在队列中，直到它们可以获得互斥锁上的锁。

甚至可以将这个过程想象成如下：

![](img/697b3b4a-2072-498c-a97c-64ab09b5f9a5.png)

现实当然没有那么美好，主要是由于硬件对我们施加的实际限制。一个明显的限制是同步原语并不是免费的。即使它们是在硬件中实现的，也需要多次调用才能使它们工作。

在硬件中实现互斥锁的两种最常见的方法是使用**测试和设置**（TAS）或**比较和交换**（CAS）CPU 特性。

测试和设置通常被实现为两个汇编级指令，这些指令是自主执行的，意味着它们不能被中断。第一条指令测试某个内存区域是否设置为 1 或零。第二条指令只有在值为零（`false`）时才执行。这意味着互斥锁尚未被锁定。因此，第二条指令将内存区域设置为 1，锁定互斥锁。

在伪代码中，这将如下所示：

```cpp
bool TAS(bool lock) { 
   if (lock) { 
         return true; 
   } 
   else { 
         lock = true; 
         return false; 
   } 
} 
```

比较和交换是一个较少使用的变体，它对内存位置和给定值执行比较操作，只有在前两者匹配时才替换该内存位置的内容：

```cpp
bool CAS(int* p, int old, int new) { 
   if (*p != old) { 
               return false; 
         } 

   *p = new; 
         return true; 
} 
```

在任何一种情况下，都需要积极重复任一函数，直到返回一个正值：

```cpp
volatile bool lock = false; 

 void critical() { 
     while (TAS(&lock) == false); 
     // Critical section 
     lock = 0; 
 } 
```

在这里，使用一个简单的 while 循环来不断轮询内存区域（标记为 volatile 以防止可能有问题的编译器优化）。通常，使用一个算法来慢慢减少轮询的频率。这是为了减少对处理器和内存系统的压力。

这清楚地表明使用互斥锁并不是免费的，而每个等待互斥锁的线程都会积极地使用资源。因此，这里的一般规则是：

+   确保线程尽可能短暂地等待互斥锁和类似的锁。

+   对于较长的等待期间，使用条件变量或定时器。

# 锁是一种高级的互斥锁

正如我们在互斥锁部分中所看到的，使用互斥锁时需要牢记一些问题。当然，当使用基于互斥锁的锁和其他机制时，这些问题也同样适用，即使其中一些问题被这些 API 平滑地解决了。

当首次使用多线程 API 时，人们可能会对不同的同步类型之间的实际区别感到困惑。正如我们在本章前面所介绍的，互斥锁是几乎所有同步机制的基础，只是在它们使用互斥锁来实现所提供的功能的方式上有所不同。

这里的重要一点是它们不是不同的同步机制，而只是基本互斥类型的特殊化。无论是使用常规互斥锁、读/写锁、信号量，甚至像可重入（递归）互斥锁或锁这样奇特的东西，完全取决于试图解决的特定问题。

对于调度器，我们首先在第十一章中遇到，*线程同步和通信*，我们使用常规互斥锁来保护包含排队工作线程和请求的数据结构。由于任何对任一数据结构的访问可能不仅涉及读取操作，还可能涉及结构的操作，因此在那里使用读/写锁是没有意义的。同样，递归锁也不会对谦虚的互斥锁有任何作用。

对于每个同步问题，因此必须问以下问题：

+   我有哪些要求？

+   哪种同步机制最适合这些要求？

因此，选择复杂类型是有吸引力的，但通常最好坚持满足所有要求的更简单的类型。当涉及调试自己的实现时，与使用更直接和低级的 API 相比，可以节省宝贵的时间。

# 线程与未来

最近，有人开始建议不要使用线程，而是倡导使用其他异步处理机制，比如`promise`。背后的原因是使用线程和涉及的同步是复杂且容易出错的。通常，人们只想并行运行一个任务，而不用担心结果是如何获得的。

对于只运行短暂的简单任务，这当然是有意义的。基于线程的实现的主要优势始终是可以完全定制其行为。使用`promise`，可以发送一个要运行的任务，并在最后，从`future`实例中获取结果。这对于简单的任务很方便，但显然并不涵盖很多情况。

在这里最好的方法是首先充分了解线程和同步机制，以及它们的限制。只有在那之后才真正有意义地考虑是否希望使用`promise`、`packaged_task`或完整的线程。

另一个重要考虑因素是，这些更复杂的、基于未来的 API 通常是基于模板的，这可能会使调试和解决可能发生的任何问题变得更加困难，而不像使用更直接和低级的 API 那样容易。

# 静态初始化顺序

静态变量是只声明一次的变量，基本上存在于全局范围内，尽管可能只在特定类的实例之间共享。也可能有完全静态的类：

```cpp
class Foo { 
   static std::map<int, std::string> strings; 
   static std::string oneString; 

public: 
   static void init(int a, std::string b, std::string c) { 
         strings.insert(std::pair<int, std::string>(a, b)); 
         oneString = c; 
   } 
}; 

std::map<int, std::string> Foo::strings; 
std::string Foo::oneString; 
```

正如我们在这里所看到的，静态变量和静态函数似乎是一个非常简单但强大的概念。虽然从本质上讲这是正确的，但在静态变量和类的初始化方面存在一个主要问题，这将会让不注意的人掉入陷阱。这个问题就是初始化顺序。

想象一下，如果我们希望在另一个类的静态初始化中使用前面的类，就像这样：

```cpp
class Bar { 
   static std::string name; 
   static std::string initName(); 

public: 
   void init(); 
}; 

// Static initializations. 
std::string Bar::name = Bar::initName(); 

std::string Bar::initName() { 
   Foo::init(1, "A", "B"); 
   return "Bar"; 
} 
```

虽然这似乎会很好地工作，将第一个字符串添加到类的映射结构中，整数作为键，但这段代码很有可能会崩溃。原因很简单，没有保证在调用`Foo::init()`时`Foo::string`已经初始化。因此，尝试使用未初始化的映射结构将导致异常。

简而言之，静态变量的初始化顺序基本上是随机的，如果不考虑这一点，就会导致非确定性行为。

这个问题的解决方案非常简单。基本上，目标是使更复杂的静态变量的初始化显式，而不是像前面的例子中那样隐式。为此，我们修改了 Foo 类：

```cpp
class Foo { 
   static std::map<int, std::string>& strings(); 
   static std::string oneString; 

public: 
   static void init(int a, std::string b, std::string c) { 
         static std::map<int, std::string> stringsStatic = Foo::strings(); 
         stringsStatic.insert(std::pair<int, std::string>(a, b)); 
         oneString = c; 
   } 
}; 

std::string Foo::oneString; 

std::map<int, std::string>& Foo::strings() { 
   static std::map<int, std::string>* stringsStatic = new std::map<int, std::string>(); 
   return *stringsStatic; 
} 
```

从顶部开始，我们看到我们不再直接定义静态映射。相反，我们有一个同名的私有函数。这个函数的实现在这个示例代码的底部找到。在其中，我们有一个指向具有熟悉映射定义的静态指针。

当调用此函数时，如果尚未存在实例，则会创建一个新的映射，因为它是一个静态变量。在修改后的`init()`函数中，我们看到我们调用`strings()`函数来获取对此实例的引用。这是显式初始化的部分，因为调用该函数将始终确保在使用之前初始化映射结构，解决了我们先前遇到的问题。

我们还可以看到一个小优化：我们创建的`stringsStatic`变量也是静态的，这意味着我们只会调用`strings()`函数一次。这样就不需要重复调用函数，恢复了我们在先前简单但不稳定的实现中所具有的速度。

静态变量初始化的基本规则是，对于非平凡的静态变量，始终使用显式初始化。

# 摘要

在本章中，我们看了一些编写多线程代码时需要牢记的良好实践和规则，以及一些建议。到这一点，人们应该能够避免一些编写此类代码时的较大陷阱和主要混淆源。

在下一章中，我们将看看如何利用底层硬件来实现原子操作，以及在 C++11 中引入的`<atomics>`头文件。
