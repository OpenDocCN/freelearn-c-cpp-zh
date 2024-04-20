# C++中的语言级并发和并行

自 C++ 11 语言标准发布以来，C++一直对并发编程提供了出色的支持。在那之前，线程是由特定于平台的库处理的事务。微软公司有自己的线程库，其他平台（GNU Linux/macOS X）支持 POSIX 线程模型。作为语言的一部分的线程机制帮助 C++程序员编写可在多个平台上运行的可移植代码。

最初的 C++标准于 1998 年发布，语言设计委员会坚信线程、文件系统、GUI 库等最好留给特定平台的库。Herb Sutter 在《Dr. Dobbs Journal》上发表了一篇有影响力的文章，题为《免费午餐结束了》，他在文章中提倡利用多核处理器中的多个核心的编程技术。在编写并行代码时，函数式编程模型非常适合这项任务。线程、Lambda 函数和表达式、移动语义和内存保证等特性帮助人们轻松地编写并发或并行代码。本章旨在使开发人员能够利用线程库及其最佳实践。

在本章中，我们将涵盖以下主题：

+   什么是并发？

+   使用多个线程的特征 Hello World 程序

+   如何管理线程的生命周期和资源

+   在线程之间共享数据

+   如何编写线程安全的数据结构

# 什么是并发？

在基本层面上，并发代表着多个活动同时发生。我们可以将并发与我们的许多现实生活情况联系起来，比如我们一边吃爆米花一边看电影，或者同时用两只手进行不同的功能，等等。那么，在计算机中，并发是什么呢？

几十年前，计算机系统已经能够进行任务切换，多任务操作系统也存在了很长时间。为什么计算领域突然对并发产生了新的兴趣？微处理器制造商通过将更多的硅片塞入处理器来增加计算能力。在这个过程的某个阶段，由于达到了基本的物理极限，他们无法再将更多的东西塞入相同的区域。那个时代的 CPU 一次只能执行一条执行路径，并通过切换任务（指令流）来运行多条指令路径。在 CPU 级别上，只有一个指令流在执行，由于事情发生得非常快（与人类感知相比），用户感觉动作是同时发生的。

大约在 2005 年，英特尔宣布了他们的新多核处理器（支持硬件级别的多条执行路径），这是一个改变游戏规则的事件。多核处理器不再是通过在任务之间切换来执行每个任务的处理器，而是作为一个解决方案来实际并行执行它们。但这给程序员带来了另一个挑战，即编写他们的代码以利用硬件级别的并发性。此外，实际硬件并发行为与任务切换所创建的幻觉之间存在差异的问题也出现了。直到多核处理器出现之前，芯片制造商一直在竞相增加他们的计算能力，期望在 21 世纪初达到 10 GHz。正如 Herb Sutter 在《免费午餐结束了》中所说的：“如果软件要利用这种增加的计算能力，它必须设计成能够同时运行多个任务”。Herb 警告程序员，那些忽视并发性的人在编写程序时也必须考虑这一点。

现代 C++标准库提供了一套机制来支持并发和并行。首先，`std::thread`以及同步对象（如`std::mutex`、`std::lock_guards`、`std::unique_lock`、`std::condition_variables`等）使程序员能够使用标准 C++编写并发的多线程代码。其次，为了使用基于任务的并行（如.NET 和 Java），C++引入了`std::future`和`std::promise`类，它们配对工作以分离函数调用和等待结果。

最后，为了避免管理线程的额外开销，C++引入了一个名为`std::async`的类，它将在接下来的章节中详细介绍，讨论重点将是编写无锁并发程序（至少在可能的情况下最小化锁定）。

并发是指两个或更多个线程或执行路径可以在重叠的时间段内启动、运行和完成（以某种交错的执行方式）。并行意味着两个任务可以同时运行（就像在多核 CPU 上看到的那样）。并发是关于响应时间，而并行主要是利用可用资源。

# 并发的 Hello World（使用 std::thread）

现在，让我们开始使用`std::thread`库编写我们的第一个程序。我们期望您有 C++ 11 或更高版本来编译我们将在本章讨论的程序。在深入讨论多线程的 Hello World 之前，让我们以一个简单的经典的 Hello World 示例作为参考：

```cpp
//---- Thanks to Dennis Ritchie and Brian Kernighan, this is a norm for all languages
#include <iostream> 
int main() 
{ 
   std::cout << "Hello World\n"; 
} 
```

这个程序简单地将 Hello World 写入标准输出流（主要是控制台）。现在，让我们看另一个例子，它做同样的事情，但是使用一个后台线程（通常称为工作线程）：

```cpp
#include <iostream> 
#include <thread> 
#include <string> 
//---- The following function will be invoked by the thread library 
void thread_proc(std::string msg) 
{ 
   std::cout << "ThreadProc msg:" << msg; 
}  
int main() 
{ 
   // creates a new thread and execute thread_proc on it. 
   std::thread t(thread_proc, "Hello World\n");  
   // Waiting for the thread_proc to complete its execution 
   // before exiting from the program 
   t.join(); 
} 
```

与传统代码的第一个区别是包含了`<thread>`标准头文件。所有的多线程支持函数和类都声明在这个新头文件中。但是为了实现同步和共享数据保护，支持类是在其他头文件中可用的。如果您熟悉 Windows 或 POSIX 系统中的平台级线程，所有线程都需要一个初始函数。标准库也遵循相同的概念。在这个例子中，`thread_proc`函数是在主函数中声明的线程的初始函数。初始函数（通过函数指针）在`std::thread`对象`t`的构造函数中指定，并且构造开始执行线程。

最显著的区别是现在应用程序从一个新线程（后台线程）向标准输出流写入消息，这导致在此应用程序中有两个线程或执行路径。一旦新线程启动，主线程就会继续执行。如果主线程不等待新启动的线程完成，`main()`函数将结束，这样应用程序就会结束——甚至在新线程有机会完成执行之前。这就是在主线程完成之前调用`join()`的原因，以等待新线程`t`的结束。

# 管理线程

在运行时，执行从用户入口点`main()`开始（在启动代码执行之后），并且将在已创建的默认线程中执行。因此，每个程序都至少有一个执行线程。在程序执行期间，可以通过标准库或特定于平台的库创建任意数量的线程。如果 CPU 核心可用于执行它们，这些线程可以并行运行。如果线程数多于 CPU 核心数，即使存在并行性，我们也无法同时运行所有线程。因此，线程切换也在这里发生。程序可以从主线程启动任意数量的线程，并且这些线程在初始线程上同时运行。正如我们所看到的，程序线程的初始函数是`main()`，并且当主线程从其执行返回时程序结束。这将终止所有并行线程。因此，主线程需要等待直到所有子线程完成执行。因此，让我们看看线程的启动和加入是如何发生的。

# 线程启动

在前面的示例中，我们看到初始化函数作为参数传递给`std::thread`构造函数，并且线程被启动。此函数在自己的线程上运行。线程启动发生在线程对象的构造期间，但初始化函数也可以有其他替代方案。函数对象是线程类的另一个可能参数。C++标准库确保`std::thread`与任何可调用类型一起工作。

现代 C++标准支持通过以下方式初始化线程：

+   函数指针（如前一节中）

+   实现调用运算符的对象

+   Lambda

任何可调用实体都可以用于初始化线程。这使得`std::thread`能够接受具有重载函数调用运算符的类对象：

```cpp
class parallel_job 
{ 
public: 
void operator() () 
{ 
    some_implementation(); 
} 
};  
parallel_job job; 
std::thread t(job); 
```

在这里，新创建的线程将对象复制到其存储中，因此必须确保复制行为。在这里，我们还可以使用`std::move`来避免与复制相关的问题：

```cpp
std::thread t(std::move(job)); 
```

如果传递临时对象（rvalue）而不是函数对象，则语法如下：

```cpp
std::thread t(parallel_job()); 
```

编译器可以将此代码解释为接受函数指针并返回`std::thread`对象的函数声明。但是，我们可以通过使用新的统一初始化语法来避免这种情况，如下所示：

```cpp
std::thread t{ parallel_job() };
```

在以下代码片段中给出的额外一组括号也可以避免将`std::thread`对象声明解释为函数声明：

```cpp
std::thread t((parallel_job()));
```

启动线程的另一个有趣的方法是通过将 C++ Lambda 作为参数传递给`std::thread`构造函数。Lambda 可以捕获局部变量，从而避免不必要地使用任何参数。当涉及编写匿名函数时，Lambda 非常有用，但这并不意味着它们应该随处使用。

Lambda 函数可以与线程声明一起使用，如下所示：

```cpp
std::thread t([]{ 
    some_implementation(); 
}); 
```

# 线程加入

在 Hello World 示例中，您可能已经注意到在`main()`结束之前使用了`t.join()`。在函数离开之前，对关联线程实例的`join()`调用确保启动的函数将等待直到后台线程完成执行。如果没有 join，线程将在线程开始之前终止，直到当前上下文完成（它们的子线程也将被终止）。

`join()`是一个直接的函数，可以等待线程完成，也可以不等待。为了更好地控制线程，我们还有其他机制，比如互斥锁、条件变量和期物，它们将在本章和下一章的后面部分进行讨论。调用`join()`会清理与线程相关联的存储，因此确保对象不再与启动的线程相关联。这意味着`join()`函数只能每个线程调用一次；在调用`join()`后，调用`joinable()`将始终返回 false。前面的使用函数对象的示例可以修改如下以理解`join()`：

```cpp
class parallel_job 
{ 
   int& _iterations; 

public: 
    parallel_job(int& input): _iterations(input) 
    {} 

    void operator() () 
    { 
        for (int i = 0; i < _iterations; ++i) 
        { 
            some_implementation(i); 
        } 
    } 
}; 
void func() 
{ 
    int local_Val = 10000; 
    parallel_job job(local_Val); 
    std::thread t(job); 

    if(t.joinable()) 
        t.join(); 
} 
```

在这种情况下，在`func()`函数结束时，验证线程对象以确认线程是否仍在执行。在放置 join 调用之前，我们调用`joinable()`来查看其返回值。

为了防止在`func()`上等待，标准引入了一种机制，即使父函数完成执行，也可以继续执行。这可以通过另一个标准函数`detach()`来实现：

```cpp
if(t.joinable()) 
         t.detach(); 
```

在分离线程之前，我们需要考虑几件事情；当`func()`退出时，线程`t`可能仍在运行。根据前面示例中给出的实现，线程使用了在`func()`中创建的局部变量的引用，这不是一个好主意，因为在大多数架构上，旧的堆栈变量随时可能被覆盖。在编写代码时，必须始终解决这些情况。处理这种情况的最常见方法是使线程自包含，并将数据复制到线程中，而不是共享它。

# 将参数传递给线程

因此，我们已经找出了如何启动和等待线程。现在，让我们看看如何将参数传递给线程初始化函数。让我们看一个计算阶乘的示例：

```cpp
class Factorial 
{ 
private: 
    long long myFact; 

public: 
    Factorial() : myFact(1) 
    { 
    } 

    void operator() (int number) 
    { 
        myFact = 1; 
        for (int i = 1; i <= number; ++i) 
        { 
            myFact *= i; 
        } 
        std::cout << "Factorial of " << number << " is " << myFact; 
    } 
}; 

int main() 
{ 
    Factorial fact; 

    std::thread t1(fact, 10); 

    t1.join(); 
} 

```

从这个例子中，可以清楚地看出，通过向`std::thread()`声明中传递额外的参数，可以实现将参数传递给线程函数或线程可调用对象。我们必须记住一件事；*传递的参数被复制到线程的内部存储以供进一步执行*。对于线程的执行来说，拥有自己的参数副本是很重要的，因为我们已经看到了与局部变量作用域结束相关的问题。要进一步讨论将参数传递给线程，让我们回到本章的第一个 Hello World 示例：

```cpp
void thread_proc(std::string msg); 

std::thread t(thread_proc, "Hello World\n"); 
```

在这种情况下，`thread_proc()`函数以`std::string`作为参数，但我们将`const char*`作为参数传递给线程函数。只有在线程的情况下，参数才会被传递、转换并复制到线程的内部存储中。在这里，`const char*`将被转换为`std::string`。必须在选择线程提供的参数类型时考虑到这一点。让我们看看如果将指针作为参数提供给线程会发生什么：

```cpp
void thread_proc(std::string msg); 
void func() 
{ 
   char buf[512]; 
   const char* hello = "Hello World\n"; 
   std::strcpy(buf, hello); 

   std::thread t(thread_proc, buf); 
   t.detach(); 
} 
```

在前面的代码中，提供给线程的参数是指向局部变量`buf`的指针。`func()`函数在线程上发生`buf`转换为`std::string`之前可能会退出。这可能导致未定义的行为。可以通过在声明中将`buf`变量转换为`std::string`来解决这个问题，如下所示：

```cpp
std::thread t(thread_proc, std::string(buf)); 
```

现在，让我们看看当您希望在线程中更新引用时的情况。在典型情况下，线程会复制传递给线程的值，以确保安全执行，但标准库还提供了一种通过引用传递参数给线程的方法。在许多实际系统中，您可能已经看到在线程内部更新共享数据结构。以下示例展示了如何在线程中实现按引用传递：

```cpp
void update_data(shared_data& data);

void another_func() 
{ 
   shared_data data; 
   std::thread t(update_data, std::ref(data)); 
   t.join(); 
   do_something_else(data); 
} 
```

在前面的代码中，使用`std::ref`将传递给`std::thread`构造函数的参数包装起来，确保线程内部使用的变量是实际参数的引用。您可能已经注意到，线程初始化函数的函数原型接受了对`shared_data`对象的引用，但为什么在线程调用中仍然需要`std::ref()`包装呢？考虑以下线程调用的代码：

```cpp
std::thread t(update_data, data);
```

在这种情况下，`update_data()`函数期望`shared_data`参数被视为实际参数的引用。但当用作线程初始化函数时，参数会在内部被简单地复制。当调用`update_data()`时，它将传递给参数的内部副本的引用，而不是实际参数的引用。

# 使用 Lambda

现在，让我们看一下 Lambda 表达式在多线程中的用处。在以下代码中，我们将创建五个线程，并将它们放入一个向量容器中。每个线程将使用 Lambda 函数作为初始化函数。在以下代码中初始化的线程通过值捕获循环索引：

```cpp
int main() 
{ 
    std::vector<std::thread> threads; 

    for (int i = 0; i < 5; ++i) 
    { 
        threads.push_back(std::thread( [i]() { 
            std::cout << "Thread #" << i << std::endl; 
        })); 
    } 

    std::cout << "nMain function"; 

    std::for_each(threads.begin(), threads.end(), [](std::thread &t) { 
        t.join(); 
    }); 
} 
```

向量容器线程存储了在循环内创建的五个线程。一旦执行结束，它们将在`main()`函数的末尾被连接。前面代码的输出可能如下所示：

```cpp
Thread # Thread # Thread # Thread # Thread #
Main function
0
4
1
3
2
```

程序的输出可能在每次运行时都不同。这个程序是一个很好的例子，展示了并发编程中的不确定性。在接下来的部分中，我们将讨论`std::thread`对象的移动属性。

# 所有权管理

从本章迄今讨论的示例中，您可能已经注意到启动线程的函数必须使用`join()`函数等待线程完成执行，否则它将以程序失去对线程的控制为代价调用`detach()`。在现代 C++中，许多标准类型是可移动的，但不能被复制；`std::thread`就是其中之一。这意味着线程执行的所有权可以在`std::thread`实例之间通过移动语义移动。

有许多情况下，我们希望将所有权移交给另一个线程，例如，如果我们希望线程在创建线程的函数上后台运行而不等待它。这可以通过将线程所有权传递给调用函数来实现，而不是在创建的函数中等待它完成。在另一种情况下，将所有权传递给另一个函数，该函数将等待线程完成其执行。这两种情况都可以通过将一个线程实例的所有权传递给另一个线程实例来实现。

为了进一步解释，让我们定义两个函数来用作线程函数：

```cpp
void function1() 
{ 
    std::cout << "function1()n"; 
} 

void function2() 
{ 
    std::cout << "function2()n"; 
} 
```

让我们来看一下从先前声明的函数中生成线程的主要函数：

```cpp
int main() 
{ 
    std::thread t1(function1); 

    // Ownership of t1 is transferred to t2 
    std::thread t2 = std::move(t1);
```

在前面的代码中，`main()`的第一行启动了一个新的线程`t1`。然后，使用`std::move()`函数将所有权转移到`t2`，该函数调用了与`t2`关联的`std::thread`的移动构造函数。现在，t1 实例没有关联的线程执行。初始化函数`function1()`现在与`t2`关联：

```cpp
    t1 = std::thread(function2); 
```

然后，使用 rvalue 启动了一个新的线程，这将调用与`t1`关联的`std::thread`的移动赋值运算符。由于我们使用了 rvalue，因此不需要显式调用`std::move()`：

```cpp
    // thread instance Created without any associated thread execution 
    std::thread t3; 

    // Ownership of t2 is transferred to t3 
    t3 = std::move(t2); 
```

`t3`是在没有任何线程执行的情况下实例化的，这意味着它正在调用默认构造函数。然后，通过显式调用`std::move()`函数，通过移动赋值运算符将当前与`t2`关联的所有权转移到`t3`：

```cpp
    // No need to join t1, no longer has any associated thread of execution 
    if (t1.joinable())  t1.join(); 
    if (t3.joinable())  t3.join(); 

    return 0; 
} 
```

最后，与关联执行线程的`std::thread`实例在程序退出之前被连接。在这里，`t1`和`t3`是与关联执行线程的实例。

现在，让我们假设在前面示例中的线程`join()`之前存在以下代码：

```cpp
t1 = std::move(t3); 
```

在这里，实例`t1`已经与正在运行的函数(`function2`)相关联。当`std::move()`试图将`function1`的所有权转移回`t1`时，将调用`std::terminate()`来终止程序。这保证了`std::thread`析构函数的一致性。

`std::thread`中的移动支持有助于将线程的所有权从函数中转移出来。以下示例演示了这样的情况：

```cpp
void func() 
{ 
    std::cout << "func()n"; 
} 

std::thread thread_creator() 
{ 
    return std::thread(func); 
} 

void thread_wait_func() 
{ 
    std::thread t = thread_creator(); 

    t.join(); 
} 
```

在这里，`thread_creator()`函数返回与`func()`函数相关联的`std::thread`。`thread_wait_func()`函数调用`thread_creator()`，然后返回线程对象，这是一个 rvalue，分配给了一个`std::thread`对象。这将线程的所有权转移到`std::thread`对象`t`中，对象`t`正在等待转移函数中线程执行的完成。

# 在线程之间共享数据

我们已经看到了如何启动线程和管理它们的不同方法。现在，让我们讨论如何在线程之间共享数据。并发的一个关键特性是它能够在活动的线程之间共享数据。首先，让我们看看线程访问共同（共享）数据所带来的问题。

如果在线程之间共享的数据是不可变的（只读），那么就不会有问题，因为一个线程读取的数据不受其他线程是否读取相同数据的影响。当线程开始修改共享数据时，问题就开始出现了。

例如，如果线程正在访问一个共同的数据结构，如果正在进行更新，与数据结构相关的不变量将被破坏。在这种情况下，数据结构中存储了元素的数量，通常需要修改多个值。考虑自平衡树或双向链表的删除操作。如果不采取任何特殊措施来确保否则，如果一个线程正在读取数据结构，而另一个正在删除一个节点，很可能会导致读取线程看到具有部分删除节点的数据结构，因此不变量被破坏。这可能最终会永久损坏数据结构，并可能导致程序崩溃。

不变量是一组在程序执行或对象生命周期中始终为真的断言。在代码中放置适当的断言来查看不变量是否被违反将产生健壮的代码。这是一种很好的记录软件的方式，也是防止回归错误的良好机制。关于这一点可以在以下维基百科文章中阅读更多：[`en.wikipedia.org/wiki/Invariant_(computer_science)`](https://en.wikipedia.org/wiki/Invariant_(computer_science))。

这经常导致一种称为*竞争条件*的情况，这是并发程序中最常见的错误原因。在多线程中，竞争条件意味着线程竞争执行各自的操作。在这里，结果取决于两个或更多线程中操作的执行相对顺序。通常，竞争条件一词指的是问题性的竞争条件；正常的竞争条件不会导致任何错误。问题性的竞争条件通常发生在完成操作需要修改两个或更多位数据的情况下，例如在树数据结构或双向链表中删除节点。因为修改必须访问不同的数据片段，当另一个线程尝试访问数据结构时，这些数据必须在单独的指令中进行修改。这发生在先前修改的一半已经完成时。

竞争条件通常很难找到，也很难复制，因为它们发生在非常短的执行窗口内。对于使用并发的软件，实现的主要复杂性来自于避免问题性的竞争条件。

有许多方法可以处理问题性的竞争条件。常见且最简单的选择是使用*同步原语*，这是基于锁的保护机制。它通过使用一些锁定机制来包装数据结构，以防止其他线程在其执行期间访问。我们将在本章中详细讨论可用的同步原语及其用途。

另一个选择是修改数据结构及其不变量的设计，以确保修改可以保证代码的顺序一致性，即使跨多个线程。这是一种编写程序的困难方式，通常被称为*无锁编程*。无锁编程和 C++内存模型将在第四章中进行介绍，《C++中的异步和无锁编程》。

然后，还有其他机制，比如将对数据结构的更新视为事务，就像对数据库的更新是在事务中完成的一样。目前，这个主题不在本书的范围内，因此不会涉及。

现在，让我们考虑 C++标准中用于保护共享数据的最基本机制，即*互斥锁*。

# 互斥锁

互斥锁是用于并发控制的机制，用于防止竞争条件。互斥锁的功能是防止执行线程在另一个并发线程进入其自己的临界区时进入其*临界区*。它是一个可锁定的对象，设计用于在代码的临界区需要独占访问时发出信号，从而限制其他并发线程在执行和内存访问方面具有相同的保护。C++ 11 标准引入了`std::mutex`类到标准库中，以实现跨并发线程的数据保护。

`std::mutex`类包括`lock()`和`unlock()`函数，用于在代码中创建临界区。在使用成员函数创建临界区时要记住的一件事是，永远不要跳过与锁定函数相关联的解锁函数，以标记代码中的临界区。

现在，让我们讨论与线程一起使用 Lambda 时所使用的相同代码。在那里，我们观察到程序的输出由于与共享资源`std::cout`和`std::ostream`操作符的竞争条件而混乱。现在，该代码正在使用`std::mutex`进行重写，以打印线程索引：

```cpp
#include <iostream> 
#include <thread> 
#include <mutex> 
#include <vector>  
std::mutex m; 
int main() 
{ 
    std::vector<std::thread> threads; 

    for (int i = 1; i < 10; ++i) 
    { 
        threads.push_back(std::thread( [i]() { 
            m.lock(); 
            std::cout << "Thread #" << i << std::endl; 
            m.unlock();
        })); 
    }      
    std::for_each(threads.begin(), threads.end(), [](std::thread &t) { 
        t.join(); 
    }); 
} 
```

前面代码的输出可能如下所示：

```cpp
Thread #1 
Thread #2 
Thread #3 
Thread #4 
Thread #5 
Thread #6 
Thread #7 
Thread #8 
Thread #9 
```

在前面的代码中，互斥锁用于保护共享资源，即`std::cout`和级联的`std::ostream`操作符。与旧示例不同，现在代码中添加了互斥锁，避免了混乱的输出，但输出将以随机顺序出现。在`std::mutex`类中使用`lock()`和`unlock()`函数可以保证输出不会混乱。然而，直接调用成员函数的做法并不推荐，因为你需要在函数的每个代码路径上调用解锁，包括异常情况。相反，C++标准引入了一个新的模板类`std::lock_guard`，它为互斥锁实现了**资源获取即初始化**（**RAII**）习惯用法。它在构造函数中锁定提供的互斥锁，并在析构函数中解锁。这个模板类的实现在`<mutex>`标准头文件库中可用。前面的示例可以使用`std::lock_guard`进行重写，如下所示：

```cpp
std::mutex m; 
int main() 
{ 
    std::vector<std::thread> threads;  
    for (int i = 1; i < 10; ++i) 
    { 
        threads.push_back(std::thread( [i]() { 
            std::lock_guard<std::mutex> local_lock(m); 
            std::cout << "Thread #" << i << std::endl; 
        })); 
    }      
    std::for_each(threads.begin(), threads.end(), [](std::thread &t) { 
        t.join(); 
    }); 
}
```

在前面的代码中，保护临界区的互斥锁位于全局范围，而`std::lock_guard`对象在每次线程执行时都是局部的 Lambda。这样，一旦对象被构造，互斥锁就会获得锁。当 Lambda 执行结束时，调用析构函数解锁互斥锁。

RAII 是 C++的一种习惯用法，其中诸如数据库/文件句柄、套接字句柄、互斥锁、堆上动态分配的内存等实体的生命周期都与持有它的对象的生命周期绑定。你可以在以下维基百科页面上阅读更多关于 RAII 的内容：[`en.wikipedia.org/wiki/Resource_acquisition_is_initialization`](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)。

# 避免死锁

在处理互斥锁时，可能出现的最大问题就是死锁。要理解死锁是什么，想象一下 iPod。为了实现 iPod 的目的，它需要 iPod 和耳机。如果两个兄弟共享一个 iPod，有时候两个人都想同时听音乐。想象一个人拿到了 iPod，另一个拿到了耳机，他们都不愿意分享自己拥有的物品。现在他们陷入僵局，除非其中一个人试图友好一点，让另一个人听音乐。

在这里，兄弟们在争夺 iPod 和耳机，但回到我们的情况，线程在争夺互斥锁上的锁。在这里，每个线程都有一个互斥锁，并且正在等待另一个线程。没有互斥锁可以继续进行，因为每个线程都在等待另一个线程释放其互斥锁。这种情况被称为**死锁**。

避免死锁有时候相当简单，因为不同的互斥锁用于不同的目的，但也有一些情况处理起来并不那么明显。我能给你的最好建议是，为了避免死锁，始终以相同的顺序锁定多个互斥锁。这样，你就永远不会遇到死锁情况。

考虑一个具有两个线程的程序的例子；每个线程都打算单独打印奇数和偶数。由于两个线程的意图不同，程序使用两个互斥锁来控制每个线程。两个线程之间的共享资源是`std::cout`。让我们看一个具有死锁情况的以下程序：

```cpp
// Global mutexes 
std::mutex evenMutex; 
std::mutex oddMutex;  
// Function to print even numbers 
void printEven(int max) 
{ 
    for (int i = 0; i <= max; i +=2) 
    { 
        oddMutex.lock(); 
        std::cout << i << ","; 
        evenMutex.lock(); 
        oddMutex.unlock(); 
        evenMutex.unlock(); 
    } 
} 
```

`printEven()`函数被定义为将所有小于`max`值的正偶数打印到标准控制台中。同样，让我们定义一个`printOdd()`函数，以打印小于`max`的所有正奇数，如下所示：

```cpp
// Function to print odd numbers 
void printOdd(int max) 
{ 
    for (int i = 1; i <= max; i +=2) 
    { 
        evenMutex.lock(); 
        std::cout << i << ","; 
        oddMutex.lock(); 
        evenMutex.unlock(); 
        oddMutex.unlock(); 

    } 
} 
```

现在，让我们编写`main`函数，生成两个独立的线程，使用先前定义的函数作为每个操作的线程函数来打印奇数和偶数：

```cpp
int main() 
{ 
    auto max = 100; 

    std::thread t1(printEven, max); 
    std::thread t2(printOdd, max); 

    if (t1.joinable()) 
        t1.join(); 
    if (t2.joinable()) 
        t2.join(); 
} 
```

在这个例子中，`std::cout`受到两个互斥锁`printEven`和`printOdd`的保护，它们以不同的顺序进行锁定。使用这段代码，我们总是陷入死锁，因为每个线程明显都在等待另一个线程锁定的互斥锁。运行这段代码将导致程序挂起。如前所述，可以通过以相同的顺序锁定它们来避免死锁，如下所示：

```cpp
void printEven(int max) 
{ 
    for (int i = 0; i <= max; i +=2) 
    { 
        evenMutex.lock(); 
        std::cout << i << ","; 
        oddMutex.lock(); 
        evenMutex.unlock(); 
        oddMutex.unlock(); 
    } 
}  
void printOdd(int max) 
{ 
    for (int i = 1; i <= max; i +=2) 
    { 
        evenMutex.lock(); 
        std::cout << i << ","; 
        oddMutex.lock(); 
        evenMutex.unlock(); 
        oddMutex.unlock(); 

    } 
} 
```

但是这段代码显然不够干净。你已经知道使用 RAII 习惯用法的互斥锁可以使代码更清晰、更安全，但为了确保锁定的顺序，C++标准库引入了一个新函数`std::lock`——一个可以一次锁定两个或更多互斥锁而不会出现死锁风险的函数。以下示例展示了如何在先前的奇偶程序中使用这个函数：

```cpp
void printEven(int max) 
{ 
    for (int i = 0; i <= max; i +=2) 
    { 
        std::lock(evenMutex, oddMutex); 
        std::lock_guard<std::mutex> lk_even(evenMutex, std::adopt_lock); 
        std::lock_guard<std::mutex> lk_odd(oddMutex, std::adopt_lock); 
        std::cout << i << ","; 
    } 
}  
void printOdd(int max) 
{ 
    for (int i = 1; i <= max; i +=2) 
    { 
        std::lock(evenMutex, oddMutex); 
        std::lock_guard<std::mutex> lk_even(evenMutex, std::adopt_lock); 
        std::lock_guard<std::mutex> lk_odd(oddMutex, std::adopt_lock); 

        std::cout << i << ","; 

    } 
} 
```

在这种情况下，一旦线程执行进入循环，对`std::lock`的调用会锁定两个互斥锁。为每个互斥锁构造了两个`std::lock_guard`实例。除了互斥锁实例之外，还提供了`std::adopt_lock`参数给`std::lock_guard`，以指示互斥锁已经被锁定，它们应该只是接管现有锁的所有权，而不是尝试在构造函数中锁定互斥锁。这保证了安全的解锁，即使在异常情况下也是如此。

然而，`std::lock`可以帮助您避免死锁，因为程序要求同时锁定两个或多个互斥锁时，它并不会帮助您解决问题。死锁是多线程程序中可能发生的最困难的问题之一。它最终依赖于程序员的纪律，不要陷入任何死锁情况。

# 使用 std::unique_lock 进行锁定

与`std::lock_guard`相比，`std::unique_lock`在操作上提供了更多的灵活性。`std::unique_lock`实例并不总是拥有与之关联的互斥锁。首先，您可以将`std::adopt_lock`作为第二个参数传递给构造函数，以管理与`std::lock_guard`类似的互斥锁上的锁。其次，通过将`std::defer_lock`作为第二个参数传递给构造函数，在构造期间互斥锁可以保持未锁定状态。因此，稍后在代码中，可以通过在同一`std::unique_lock`对象上调用`lock()`来获取锁。但是，`std::unique_lock`提供的灵活性是有代价的；它在存储额外信息方面比`lock_guard`慢一些，并且需要更新。因此，建议除非确实需要`std::unique_lock`提供的灵活性，否则使用`lock_guard`。

关于`std::unique_lock`的另一个有趣特性是其所有权转移的能力。由于`std::unique_lock`必须拥有其关联的互斥锁，这导致互斥锁的所有权转移。与`std::thread`类似，`std::unique_lock`类也是一种只能移动的类型。C++标准库中提供的所有移动语义语言细微差别和右值引用处理都适用于`std::unique_lock`。

与`std::mutex`类似，具有`lock()`和`unlock()`等成员函数的可用性增加了它在代码中的灵活性，相对于`std::lock_guard`。在`std::unique_lock`实例被销毁之前释放锁的能力意味着，如果明显不再需要锁，可以在代码的任何地方选择性地释放它。不必要地持有锁会严重降低应用程序的性能，因为等待锁的线程会被阻止执行比必要时间更长的时间。因此，`std::unique_lock`是 C++标准库引入的非常方便的功能，支持 RAII 习惯用法，并且可以有效地最小化适用代码的关键部分的大小：

```cpp
void retrieve_and_process_data(data_params param) 
{ 
   std::unique_lock<std::mutex> local_lock(global_mutex, std::defer_lock); 
   prepare_data(param); 

   local_lock.lock(); 
   data_class data = get_data_to_process(); 
   local_lock.unlock(); 

   result_class result = process_data(data); 

   local_lock.lock(); 
   strore_result(result); 
} 
```

在前面的代码中，您可以看到通过利用`std::unique_lock`的灵活性实现的细粒度锁定。当函数开始执行时，使用`global_mutex`构造了一个处于未锁定状态的`std::unique_lock`对象。立即准备了不需要独占访问的参数，它可以自由执行。在检索准备好的数据之前，`local_lock`使用`std::unique_lock`中的 lock 成员函数标记了关键部分的开始。一旦数据检索完成，锁将被释放，标志着关键部分的结束。在此之后，调用`process_data()`函数，再次不需要独占访问，可以自由执行。最后，在执行`store_result()`函数之前，锁定互斥锁以保护更新处理结果的写操作。在退出函数时，当`std::unique_lock`的局部实例被销毁时，锁将被释放。

# 条件变量

我们已经知道互斥锁可以用于共享公共资源并在线程之间同步操作。但是，如果不小心使用互斥锁进行同步，会变得有点复杂并容易发生死锁。在本节中，我们将讨论如何使用条件变量等待事件，以及如何以更简单的方式在同步中使用它们。

当涉及使用互斥锁进行同步时，如果等待的线程已经获得了对互斥锁的锁定，那么任何其他线程都无法锁定它。此外，通过定期检查由互斥锁保护的状态标志来等待一个线程完成执行是一种浪费 CPU 资源。这是因为这些资源可以被系统中的其他线程有效利用，而不必等待更长的时间。

为了解决这些问题，C++标准库提供了两种条件变量的实现：`std::condition_variable`和`std::condition_variable_any`。两者都声明在`<condition_variable>`库头文件中，两种实现都需要与互斥锁一起工作以同步线程。`std::condition_variable`的实现仅限于与`std::mutex`一起工作。另一方面，`std::condition_variable_any`可以与满足类似互斥锁标准的任何东西一起工作，因此带有`_any`后缀。由于其通用行为，`std::condition_variable_any`最终会消耗更多内存并降低性能。除非有真正的、定制的需求，否则不建议使用它。

以下程序是我们在讨论互斥锁时讨论过的奇偶线程的实现，现在正在使用条件变量进行重新实现。

```cpp
std::mutex numMutex; 
std::condition_variable syncCond; 
auto bEvenReady = false; 
auto bOddReady  = false; 
void printEven(int max) 
{ 
    for (int i = 0; i <= max; i +=2) 
    { 
        std::unique_lock<std::mutex> lk(numMutex); 
        syncCond.wait(lk, []{return bEvenReady;}); 

        std::cout << i << ","; 

        bEvenReady = false; 
        bOddReady  = true; 
        syncCond.notify_one(); 
    } 
}
```

程序从全局声明一个互斥锁、一个条件变量和两个布尔标志开始，以便在两个线程之间进行同步。`printEven`函数在一个工作线程中执行，并且只打印从 0 开始的偶数。在这里，当它进入循环时，互斥锁受到`std::unique_lock`的保护，而不是`std::lock_guard`；我们马上就会看到原因。然后线程调用`std::condition_variable`中的`wait()`函数，传递锁对象和一个 Lambda 谓词函数，表达了正在等待的条件。这可以用任何返回 bool 的可调用对象替换。在这个函数中，谓词函数返回`bEvenReady`标志，以便在它变为 true 时函数继续执行。如果谓词返回 false，`wait()`函数将解锁互斥锁并等待另一个线程通知它，因此`std::unique_lock`对象在这里非常方便，提供了锁定和解锁的灵活性。

一旦`std::cout`打印循环索引，`bEvenReady`标志就会被设置为 false，`bOddReady`标志则会被设置为 true。然后，与`syncCond`相关联的`notify_one()`函数的调用会向等待的奇数线程发出信号，要求其将奇数写入标准输出流：

```cpp
void printOdd(int max) 
{ 
    for (int i = 1; i <= max; i +=2) 
    { 
        std::unique_lock<std::mutex> lk(numMutex); 
        syncCond.wait(lk, []{return bOddReady;}); 

        std::cout << i << ","; 

        bEvenReady = true; 
        bOddReady  = false; 
        syncCond.notify_one(); 
    } 
} 
```

`printOdd`函数在另一个工作线程中执行，并且只打印从`1`开始的奇数。与`printEven`函数不同，循环迭代并打印由全局声明的条件变量和互斥锁保护的索引。在`std::condition_variable`的`wait()`函数中使用的谓词返回`bOddReady`，`bEvenReady`标志被设置为`true`，`bOddReady`标志被设置为`false`。随后，调用与`syncCond`相关联的`notify_one()`函数会向等待的偶数线程发出信号，要求其将偶数写入标准输出流。这种奇偶数交替打印将持续到最大值：

```cpp
int main() 
{ 
    auto max = 10; 
    bEvenReady = true; 

    std::thread t1(printEven, max); 
    std::thread t2(printOdd, max); 

    if (t1.joinable()) 
        t1.join(); 
    if (t2.joinable()) 
        t2.join(); 

} 
```

主函数启动两个后台线程，`t1`与`printEven`函数相关联，`t2`与`printOdd`函数相关联。输出在确认偶数奇数性之前开始，通过将`bEvenReady`标志设置为 true。

# 线程安全的堆栈数据结构

到目前为止，我们已经讨论了如何启动和管理线程，以及如何在并发线程之间同步操作。但是，当涉及到实际系统时，数据以数据结构的形式表示，必须根据情况选择适当的数据结构，以确保程序的性能。在本节中，我们将讨论如何使用条件变量和互斥量设计并发栈。以下程序是 `std::stack` 的包装器，声明在库头文件 `<stack>` 下，并且栈包装器将提供不同的 pop 和 push 功能的重载（这样做是为了保持清单的简洁，并且还演示了如何将顺序数据结构调整为在并发上下文中工作）：

```cpp
template <typename T> 
class Stack 
{ 
private: 
    std::stack<T> myData; 
    mutable std::mutex myMutex; 
    std::condition_variable myCond; 

public: 
    Stack() = default; 
    ~Stack() = default; 
    Stack& operator=(const Stack&) = delete; 

    Stack(const Stack& that) 
    { 
        std::lock_guard<std::mutex> lock(that.myMutex); 
        myData = that.myData; 
    }
```

`Stack` 类包含模板类 `std::stack` 的对象，以及 `std::mutex` 和 `std::condition_variable` 的成员变量。类的构造函数和析构函数标记为默认，让编译器为其生成默认实现，并且复制赋值运算符标记为删除，以防止在编译时调用此类的赋值运算符。定义了复制构造函数，它通过调用自己的复制赋值运算符来复制 `std::stack` 成员对象 `myData`，该操作受到右侧对象的互斥量保护：

```cpp
      void push(T new_value) 
      { 
          std::lock_guard<std::mutex> local_lock(myMutex); 
          myData.push(new_value); 
          myCond.notify_one(); 
      } 
```

成员函数 `push()` 包装了 `std::stack` 容器的 `push` 函数。可以看到，互斥量成员变量 `myMutex` 被 `std::lock_guard` 对象锁定，以保护接下来的 `push` 操作。随后，使用成员 `std::condition_variable` 对象调用 `notify_one()` 函数，以通过相同的条件变量引发事件来通知等待的线程。在以下代码清单中，您将看到 `pop` 操作的两个重载，它们等待在此条件变量上得到信号：

```cpp
    bool try_pop(T& return_value) 
    { 
        std::lock_guard<std::mutex> local_lock(myMutex); 
        if (myData.empty()) return false; 
        return_value = myData.top(); 
        myData.pop(); 
        return true; 
    }
```

`try_pop()` 函数以模板参数作为引用。由于实现从不等待栈至少填充一个元素，因此使用 `std::lock_guard` 对象来保护线程。如果栈为空，函数返回 `false`，否则返回 `true`。在这里，输出通过调用 `std::stack` 的 `top()` 函数分配给输入引用参数，该函数返回栈中的顶部元素，然后调用 `pop()` 函数来清除栈中的顶部元素。所有 `pop` 函数的重载都调用 `top()` 函数，然后调用 `std::stack` 的 `pop()` 函数：

```cpp
    std::shared_ptr<T> try_pop() 
    { 
        std::lock_guard<std::mutex> local_lock(myMutex); 
        if (myData.empty()) return std::shared_ptr<T>(); 

        std::shared_ptr<T> return_value(std::make_shared<T>(myData.top())); 
        myData.pop(); 

        return return_value;
    } 
```

这是 `try_pop()` 函数的另一个重载，它返回模板类型的 `std::shared_ptr`（智能指针）的实例。正如您已经看到的，`try_pop` 函数有多个重载，并且从不等待栈至少填充一个元素；因此，此实现使用 `std::lock_guard`。如果内部栈为空，函数返回 `std::shared_ptr` 的实例，并且不包含栈的任何元素。否则，返回包含栈顶元素的 `std::shared_ptr` 实例：

```cpp
    void wait_n_pop(T& return_value) 
    { 
        std::unique_lock<std::mutex> local_lock(myMutex); 
        myCond.wait(local_lock, [this]{ return !myData.empty(); }); 
        return_value = myData.top(); 
        myData.pop(); 
    }      
    std::shared_ptr<T> wait_n_pop() 
    { 
        std::unique_lock<std::mutex> local_lock(myMutex); 
        myCond.wait(local_lock, [this]{ return !myData.empty(); }); 
        std::shared_ptr<T> return_value(std::make_shared<T>(myData.top())); 
        return return_value; 
    }   
}; 
```

到目前为止，`pop`函数的重载不会等待堆栈至少填充一个元素，如果它是空的。为了实现这一点，添加了`pop`函数的另外两个重载，它们使用与`std::condition_variable`相关的等待函数。第一个实现将模板值作为输出参数返回，第二个实现返回一个`std::shared_ptr`实例。这两个函数都使用`std::unique_lock`来控制互斥锁，以便提供`std::condition_variable`的`wait()`函数。在`wait`函数中，`predicate`函数正在检查堆栈是否为空。如果堆栈为空，那么`wait()`函数会解锁互斥锁，并继续等待，直到从`push()`函数接收到通知。一旦调用了 push，predicate 将返回 true，`wait_n_pop`继续执行。函数重载接受模板引用，并将顶部元素分配给输入参数，后一个实现返回一个包含顶部元素的`std::shared_ptr`实例。

# 总结

在本章中，我们讨论了 C++标准库中可用的线程库。我们看到了如何启动和管理线程，并讨论了线程库的不同方面，比如如何将参数传递给线程，线程对象的所有权管理，线程之间数据的共享等等。C++标准线程库可以执行大多数可调用对象作为线程！我们看到了所有可用的可调用对象与线程的关联的重要性，比如`std::function`，Lambda 和函数对象。我们讨论了 C++标准库中可用的同步原语，从简单的`std::mutex`开始，使用 RAII 习惯用法来保护互斥锁免受未处理的退出情况的影响，以避免显式解锁，并使用诸如`std::lock_guard`和`std::unique_lock`之类的类。我们还讨论了条件变量(`std::condition_variable`)在线程同步的上下文中。本章为现代 C++引入的并发支持奠定了良好的基础，为本书进入功能习惯打下了基础。

在接下来的章节中，我们将涵盖 C++中更多的并发库特性，比如基于任务的并行性和无锁编程。
