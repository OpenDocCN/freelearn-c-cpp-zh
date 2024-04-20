# 并行性和并发性

在本章中，我们将涵盖以下内容：

+   自动并行化使用标准算法的代码

+   让程序在特定时间内休眠

+   启动和停止线程

+   使用`std::unique_lock`和`std::shared_lock`执行异常安全的共享锁定

+   使用`std::scoped_lock`避免死锁

+   同步并发的`std::cout`使用

+   使用`std::call_once`安全地延迟初始化

+   使用`std::async`将任务执行推入后台

+   使用`std::condition_variable`实现生产者/消费者模式

+   使用`std::condition_variable`实现多个生产者/消费者模式

+   使用`std::async`并行化 ASCII Mandelbrot 渲染器

+   使用`std::future`实现一个小型自动并行化库

# 介绍

在 C++11 之前，C++对并行化的支持并不多。这并不意味着启动、控制、停止和同步线程是不可能的，但是必须使用特定于操作系统的库，因为线程本质上与操作系统相关。

使用 C++11，我们得到了`std::thread`，它可以在所有操作系统上进行基本的可移植线程控制。为了同步线程，C++11 还引入了互斥类和舒适的 RAII 风格的锁包装器。除此之外，`std::condition_variable`允许线程之间灵活地进行事件通知。

一些其他非常有趣的添加是`std::async`和`std::future`--我们现在可以将任意普通函数包装成`std::async`调用，以便在后台异步执行它们。这样包装的函数返回`std::future`对象，承诺稍后包含函数结果，因此我们可以在等待其到达之前做其他事情。

STL 的另一个实际上巨大的改进是*执行策略*，可以添加到 69 个已经*存在*的算法中。这个添加意味着我们可以在旧程序中的现有标准算法调用中添加一个单一的执行策略参数，从而实现并行化，而无需进行复杂的重写。

在本章中，我们将逐个介绍所有这些添加内容，以便了解其中最重要的内容。之后，我们将对 C++17 STL 中的并行化支持有足够的概览。我们不涵盖所有细节，但是最重要的部分。从本书中获得的概览有助于快速理解 C++ 17 STL 在线文档中的其余并行编程机制。

最后，本章包含两个额外的示例。在一个示例中，我们将使用最小的更改来并行化第二十三章中的 Mandelbrot ASCII 渲染器，*STL 算法的高级使用*。在最后一个示例中，我们将实现一个小型库，以隐式和自动地帮助并行化复杂任务。

# 自动并行化使用标准算法的代码

C++17 带来了一个非常重要的并行扩展：标准算法的*执行策略*。六十九个算法被扩展以接受执行策略，以便在多个核心上并行运行，甚至启用矢量化。

对于用户来说，这意味着如果我们已经在所有地方使用 STL 算法，我们可以免费获得一个不错的并行化奖励。我们可以通过简单地向现有的 STL 算法调用中添加一个单一的执行策略参数，轻松地为我们的应用程序提供后续的并行化。

在这个示例中，我们将实现一个简单的程序（具有不太严肃的用例场景），排列多个 STL 算法调用。在使用这些算法时，我们将看到使用 C++17 执行策略是多么容易，以便让它们多线程运行。在本节的最后几个小节中，我们将更仔细地研究不同的执行策略。

# 如何做...

在这一节中，我们将编写一个使用一些标准算法的程序。程序本身更多地是一个示例，展示了现实生活中的情景可能是什么样子，而不是真正的实际工作情况。在使用这些标准算法时，我们嵌入了执行策略以加快代码速度：

1.  首先，我们需要包含一些头文件，并声明我们使用`std`命名空间。`execution`头文件是一个新的头文件；它是 C++17 中新增的：

```cpp
      #include <iostream>
      #include <vector>
      #include <random>
      #include <algorithm>
      #include <execution>      

      using namespace std;
```

1.  仅仅为了示例，我们将声明一个谓词函数，告诉一个数字是否是奇数。我们稍后会用到它：

```cpp
      static bool odd(int n) { return n % 2; }
```

1.  让我们首先在主函数中定义一个大向量。我们将用大量数据填充它，以便对其进行计算需要一些时间。这段代码的执行速度将会*很大*地变化，取决于执行这段代码的计算机。在不同的计算机上，较小/较大的向量大小可能更好：

```cpp
      int main()
      {
          vector<int> d (50000000);
```

1.  为了获得大量的随机数据用于向量，让我们实例化一个随机数生成器和一个分布，并将它们打包在一个可调用对象中。如果这对你来说看起来很奇怪，请先看一下处理随机数生成器和分布的示例，第二十五章，*实用类*：

```cpp
          mt19937 gen;
          uniform_int_distribution<int> dis(0, 100000);

          auto rand_num ([=] () mutable { return dis(gen); });
```

1.  现在，让我们使用`std::generate`算法来填充向量的随机数据。这个算法有一个新的 C++17 版本，可以接受一种新类型的参数：执行策略。我们在这里放入了`std::par`，它允许自动并行化这段代码。通过这样做，我们允许多个线程同时开始填充向量，这样可以减少执行时间，如果计算机有多个 CPU 的话，这通常是现代计算机的情况：

```cpp
          generate(execution::par, begin(d), end(d), rand_num);
```

1.  `std::sort` 方法也应该已经很熟悉了。C++17 版本也支持一个额外的参数来定义执行策略：

```cpp
          sort(execution::par, begin(d), end(d));
```

1.  对`std::reverse`也是一样的：

```cpp
          reverse(execution::par, begin(d), end(d));
```

1.  然后我们使用`std::count_if`来计算向量中所有奇数的个数。我们甚至可以通过只添加一个执行策略来并行化它！

```cpp
          auto odds (count_if(execution::par, begin(d), end(d), odd));
```

1.  整个程序并没有做任何*真正*的科学工作，因为我们只是要看一下如何并行化标准算法，但最后让我们打印一些东西：

```cpp
          cout << (100.0 * odds / d.size()) 
               << "% of the numbers are odd.n";
      }
```

1.  编译和运行程序会给我们以下输出。在这一点上，有趣的是看到在使用算法时，不带执行策略与所有其他执行策略相比，执行速度有何不同。这留给读者作为一个练习。试一试；可用的执行策略有`seq`、`par`和`par_vec`。我们应该得到每个执行策略的不同执行时间：

```cpp
      $ ./auto_parallel
      50.4% of the numbers are odd.
```

# 它是如何工作的...

特别是因为这个示例没有让我们分心于任何复杂的现实问题解决方案，我们能够完全集中精力在标准库函数调用上。很明显，它们的并行化版本与经典的顺序版本几乎没有区别。它们只是多了一个参数，即*执行策略*。

让我们看一下调用并回答三个核心问题：

```cpp
generate(execution::par, begin(d), end(d), rand_num);
sort(    execution::par, begin(d), end(d));
reverse( execution::par, begin(d), end(d));

auto odds (count_if(execution::par, begin(d), end(d), odd));
```

# 我们可以用这种方式并行化哪些 STL 算法？

在 C++17 标准中，现有的 69 个 STL 算法升级为支持并行处理，还有七个新算法也支持并行处理。虽然这样的升级对于实现来说可能相当具有侵入性，但在接口方面并没有太多改变--它们都增加了一个额外的`ExecutionPolicy&& policy`参数，就是这样。这*不*意味着我们*总是*必须提供执行策略参数。只是它们*另外*支持接受执行策略作为它们的第一个参数。

这些是升级的 69 个标准算法。还有七个新的算法从一开始就支持执行策略（用*粗体*标出）：

| `std::adjacent_difference` `std::adjacent_find`

`std::all_of`

`std::any_of`

`std::copy`

`std::copy_if`

`std::copy_n`

`std::count`

`std::count_if`

`std::equal` `**std::exclusive_scan**`

`std::fill`

`std::fill_n`

`std::find`

`std::find_end`

`std::find_first_of`

`std::find_if`

`std::find_if_not` `**std::for_each**`

`**std::for_each_n**`

`std::generate`

`std::generate_n`

`std::includes` `**std::inclusive_scan**`

`std::inner_product` | `std::inplace_merge` `std::is_heap` `std::is_heap_until` `std::is_partitioned`

`std::is_sorted`

`std::is_sorted_until`

`std::lexicographical_compare`

`std::max_element`

`std::merge`

`std::min_element`

`std::minmax_element`

`std::mismatch`

`std::move`

`std::none_of`

`std::nth_element`

`std::partial_sort`

`std::partial_sort_copy`

`std::partition`

`std::partition_copy`

`std::remove`

`std::remove_copy`

`std::remove_copy_if`

`std::remove_if`

`std::replace`

`std::replace_copy`

`std::replace_copy_if` | `std::replace_if` `std::reverse`

`std::reverse_copy`

`std::rotate`

`std::rotate_copy`

`std::search`

`std::search_n`

`std::set_difference`

`std::set_intersection`

`std::set_symmetric_difference`

`std::set_union`

`std::sort`

`std::stable_partition`

`std::stable_sort`

`std::swap_ranges`

`std::transform`

`**std::transform_exclusive_scan**` `**std::transform_inclusive_scan**` `**std::transform_reduce**`

`std::uninitialized_copy`

`std::uninitialized_copy_n`

`std::uninitialized_fill`

`std::uninitialized_fill_n`

`std::unique`

`std::unique_copy` |

这些算法的升级是个好消息！我们的旧程序越多地利用 STL 算法，我们就越容易事后为它们添加并行性。请注意，这并*不*意味着这些更改会使每个程序自动* N *倍加速，因为多程序设计要复杂得多。

然而，我们现在可以以非常优雅、独立于操作系统的方式并行化标准任务，而不是设计自己复杂的并行算法，使用`std::thread`、`std::async`或包含外部库。

# 这些执行策略是如何工作的？

执行策略告诉我们允许使用哪种策略来自动并行化我们的标准算法调用。

`std::execution`命名空间中存在以下三种策略类型：

| **策略** | **含义** |
| --- | --- |
| `sequenced_policy` | 该算法必须以类似于原始算法的顺序形式执行，而不使用执行策略。全局可用的实例名为`std::execution::seq`。 |
| `parallel_policy` | 该算法可以以多线程方式执行，共享工作以并行方式进行。全局可用的实例名为`std::execution::par`。 |
| `parallel_unsequenced_policy` | 该算法可以以多线程共享工作的方式执行。除此之外，允许对代码进行向量化。在这种情况下，容器访问可以在线程之间交错，也可以在同一线程内由于向量化而交错。全局可用的实例名为`std::execution::par_unseq`。 |

执行策略对我们有特定的约束。约束越严格，我们可以允许的并行化策略措施就越多：

+   并行化算法使用的所有元素访问函数*不能*引起*死锁*或*数据竞争*

+   在并行和向量化的情况下，所有访问函数*必须不*使用任何形式的阻塞同步

只要我们遵守这些规则，我们就应该免受使用 STL 算法的并行版本引入的错误的影响。

请注意，正确使用并行 STL 算法并不总是会导致保证的加速。根据我们尝试解决的问题、问题规模以及我们的数据结构和其他访问方法的效率，可测量的加速将会变化很大，或者根本不会发生。*多程序设计仍然很困难。*

# 向量化是什么意思？

向量化是 CPU 和编译器都需要支持的功能。让我们简要了解一下向量化是什么以及它是如何工作的。想象一下，我们想要对一个非常大的向量中的数字进行求和。这个任务的简单实现可能如下所示：

```cpp
std::vector<int> v {1, 2, 3, 4, 5, 6, 7 /*...*/};

int sum {std::accumulate(v.begin(), v.end(), 0)};
```

编译器最终将从`accumulate`调用生成一个循环，可能如下所示：

```cpp
int sum {0};
for (size_t i {0}; i < v.size(); ++i) {
    sum += v[i];
}
```

从这一点出发，允许并启用向量化，编译器可以生成以下代码。循环在一个循环步骤中执行四个累加步骤，也减少了四倍的迭代次数。为简单起见，示例未处理向量不包含`N * 4`元素的余数：

```cpp
int sum {0};
for (size_t i {0}; i < v.size() / 4; i += 4) {
    sum += v[i] + v[i+1] + v[i + 2] + v[i + 3];
}
// if v.size() / 4 has a remainder, 
// real code has to deal with that also.
```

为什么要这样做？许多 CPU 提供的指令可以在*一步*中执行数学运算，例如`sum += v[i] + v[i+1] + v[i + 2] + v[i + 3];`。尽可能多地将*许多*数学运算压缩到*尽可能少*的指令中是目标，因为这会加快程序速度。

自动向量化很难，因为编译器需要在一定程度上理解我们的程序，以使我们的程序更快，但又不影响其*正确性*。至少，我们可以通过尽可能经常使用标准算法来帮助编译器，因为这些对编译器来说比复杂的手工循环和复杂的数据流依赖更容易理解。

# 将程序休眠特定时间

C++11 引入了一种简单的控制线程的可能性。它引入了`this_thread`命名空间，其中包括只影响调用线程的函数。它包含两个不同的函数，允许将线程休眠一段时间，因此我们不再需要为此类任务使用任何外部或操作系统相关的库。

在这个示例中，我们专注于如何暂停线程一段时间，或者如何将它们置于*休眠*状态。

# 如何做...

我们将编写一个简短的程序，只是将主线程休眠一段时间：

1.  让我们首先包含所有需要的头文件，并声明我们将使用`std`和`chrono_literals`命名空间。`chrono_literals`命名空间包含用于创建时间跨度值的方便缩写：

```cpp
      #include <iostream>
      #include <chrono>
      #include <thread>      

      using namespace std;
      using namespace chrono_literals;
```

1.  让我们立即将主线程休眠 5 秒和 300 毫秒。由于`chrono_literals`，我们可以以非常易读的格式表达这一点：

```cpp
      int main()
      {
          cout << "Going to sleep for 5 seconds"
                  " and 300 milli seconds.n";

          this_thread::sleep_for(5s + 300ms);
```

1.  最后一个休眠语句是`relative`。我们也可以表达`absolute`的休眠请求。让我们休眠到*现在*加上`3`秒的时间点：

```cpp
          cout << "Going to sleep for another 3 seconds.n";

          this_thread::sleep_until(
              chrono::high_resolution_clock::now() + 3s);
```

1.  在退出程序之前，让我们打印一些其他内容，以示第二个休眠期结束：

```cpp
          cout << "That's it.n";
      }
```

1.  编译和运行程序产生以下结果。Linux、Mac 和其他类 UNIX 操作系统提供`time`命令，该命令接受另一个命令以执行它并停止所需的时间。使用`time`运行我们的程序显示它运行了`8.32`秒，大约是我们让程序休眠的`5.3`秒和`3`秒。运行程序时，可以计算在终端上打印行到达之间的时间。

```cpp
      $ time ./sleep 
      Going to sleep for 5 seconds and 300 milli seconds.
      Going to sleep for another 3 seconds.
      That's it.

      real 0m8.320s
      user 0m0.005s
      sys  0m0.003s
```

# 它是如何工作的...

`sleep_for`和`sleep_until`函数已添加到 C++11 中，并驻留在`std::this_thread`命名空间中。它们阻塞当前线程（而不是整个进程或程序）一段特定的时间。线程在被阻塞时不会消耗 CPU 时间。它只是被操作系统置于非活动状态。当然，操作系统会提醒自己再次唤醒线程。最好的是，我们不需要关心我们的程序运行在哪个操作系统上，因为 STL 将这个细节抽象化了。

`this_thread::sleep_for`函数接受`chrono::duration`值。在最简单的情况下，这只是`1s`或`5s + 300ms`，就像我们的示例代码中一样。为了获得这样漂亮的时间跨度文字，我们需要声明`using namespace std::chrono_literals;`。

`this_thread::sleep_until`函数接受`chrono::time_point`而不是时间跨度。如果我们希望将线程休眠直到特定的挂钟时间，这是很方便的。

唤醒的时间只有操作系统允许的那么准确。这将在大多数操作系统中通常足够准确，但如果某些应用程序需要纳秒级精度，可能会变得困难。

将线程休眠一小段时间的另一种可能性是`this_thread::yield`。它不接受*任何*参数，这意味着我们无法知道线程的执行被放置回去多长时间。原因是这个函数并没有真正实现睡眠或停放线程的概念。它只是以一种合作的方式告诉操作系统可以重新安排任何其他进程的任何其他线程。如果没有，那么线程将立即再次执行。因此，`yield`通常比仅仅睡眠一段最小但指定的时间不太有用。

# 启动和停止线程

C++11 带来的另一个新增功能是`std::thread`类。它提供了一种干净简单的方法来启动和停止线程，而无需外部库或了解操作系统如何实现这一点。这一切都包含在 STL 中。

在这个示例中，我们将实现一个启动和停止线程的程序。一旦线程启动，就需要了解如何处理线程的一些细节，所以我们也会详细介绍这些内容。

# 如何做...

我们将启动多个线程，并查看当我们释放多个处理器核心同时执行其代码的部分时，我们的程序的行为如何：

1.  首先，我们需要包括只有两个标题，然后我们声明使用`std`和`chrono_literals`命名空间：

```cpp
      #include <iostream>
      #include <thread>      

      using namespace std;
      using namespace chrono_literals;
```

1.  为了启动一个线程，我们需要能够告诉它应该执行什么代码。因此，让我们定义一个可以执行的函数。函数是线程的自然潜在入口点。示例函数接受一个参数`i`，它充当线程 ID。这样我们可以稍后知道哪个打印行来自哪个线程。此外，我们使用线程 ID 让所有线程等待不同的时间，这样我们可以确保它们不会在完全相同的时间使用`cout`。如果是这样，输出将会混乱。本章的另一个示例专门处理了这个问题：

```cpp
      static void thread_with_param(int i)
      {
          this_thread::sleep_for(1ms * i);

          cout << "Hello from thread " << i << 'n';

          this_thread::sleep_for(1s * i);

          cout << "Bye from thread " << i << 'n';
      }
```

1.  在主函数中，我们可以出于好奇，使用`std::thread::hardware_concurrency`打印可以同时运行多少个线程。这取决于机器实际上有多少个核心，以及 STL 实现支持多少个核心。这意味着在每台计算机上这个数字可能是不同的：

```cpp
      int main()
      {
          cout << thread::hardware_concurrency()
               << " concurrent threads are supported.n";
```

1.  现在让我们最终启动线程。对于每个线程，我们使用不同的 ID 启动三个线程。当使用`thread t {f, x}`这样的表达式实例化线程时，这将导致新线程调用`f(x)`。这样我们可以为每个线程的`thread_with_param`函数提供不同的参数：

```cpp
          thread t1 {thread_with_param, 1};
          thread t2 {thread_with_param, 2};
          thread t3 {thread_with_param, 3};
```

1.  由于这些线程是自由运行的，当它们完成工作时，我们需要再次停止它们。我们使用`join`函数来做到这一点。它将*阻塞*调用线程，直到我们尝试加入的线程返回：

```cpp
          t1.join();
          t2.join();
```

1.  与加入相对应的是*分离*。如果我们不调用`join`或分离，整个应用程序将在`thread`对象的析构函数执行时立即终止。通过调用`detach`，我们告诉`thread`，我们真的希望让线程 3 继续运行，即使它的`thread`实例被销毁：

```cpp
          t3.detach();
```

1.  在退出主函数和整个程序之前，我们打印另一条消息：

```cpp
          cout << "Threads joined.n";
      }
```

1.  编译和运行代码显示了以下输出。我们可以看到我的机器有八个 CPU 核心。然后，我们看到了所有线程的*hello*消息，但只有两个我们实际加入的线程的*bye*消息。线程 3 仍然处于等待 3 秒的期间，但整个程序在第二个线程等待 2 秒后就已经终止了。这样，我们无法看到线程 3 的 bye 消息，因为它被简单地杀死了，没有任何完成的机会（也没有噪音）：

```cpp
      $ ./threads 
      8 concurrent threads are supported.
      Hello from thread 1
      Hello from thread 2
      Hello from thread 3
      Bye from thread 1
      Bye from thread 2
      Threads joined.
```

# 它是如何工作的...

启动和停止线程是一件非常简单的事情。当线程需要共同工作（共享资源，等待彼此等）时，多道程序设计开始变得复杂。

为了启动一个线程，我们首先需要一个将由它执行的函数。这个函数不需要特殊，因为线程可以执行几乎每个函数。让我们确定一个最小的示例程序，启动一个线程并等待它的完成：

```cpp
void f(int i) { cout << i << 'n'; }

int main()
{
    thread t {f, 123};
    t.join();
}
```

`std::thread`的构造函数调用接受一个函数指针或可调用对象，后面跟着应该与函数调用一起使用的参数。当然，也可以启动一个不接受任何参数的函数的线程。

如果系统有多个 CPU 核心，那么线程可以并行和同时运行。并行和同时运行有什么区别？如果计算机只有一个 CPU 核心，那么可以有很多线程并行运行，但从来不会同时运行，因为一个 CPU 核心一次只能运行一个线程。然后线程以交错的方式运行，每个线程执行一部分时间，然后暂停，然后下一个线程获得时间片（对于人类用户来说，这看起来像它们同时运行）。如果它们不需要共享 CPU 核心，那么它们可以同时运行，就像*真正同时*一样。

在这一点上，我们绝对*无法控制*以下细节：

+   当共享一个 CPU 核心时，线程交错的顺序。

+   线程的*优先级*，或者哪一个比其他更重要。

+   线程真正*分布*在所有 CPU 核心之间，或者操作系统只是将它们固定在同一个核心上。事实上，我们的所有线程可能只在一个核心上运行，尽管机器有 100 多个核心。

大多数操作系统提供了控制多道程序设计这些方面的可能性，但这些功能在 STL 中*不*包括在内。

然而，我们可以启动和停止线程，并告诉它们在什么时候工作，什么时候暂停。这对于大多数应用程序来说应该足够了。在本节中，我们启动了三个额外的线程。之后，我们*加入*了大部分线程，并*分离*了最后一个线程。让我们用一个简单的图表总结一下发生了什么：

![](img/6b848126-e59e-4cc1-baa0-6f31962879a3.png)

从上到下阅读图表，它显示了程序工作流程在某一时刻分成了总共四个线程。我们启动了三个额外的线程，它们做了一些事情（即等待和打印），但在启动线程之后，执行主函数的主线程仍然没有工作。

每当一个线程执行完它启动的函数时，它将从这个函数返回。标准库然后做一些整理工作，导致线程从操作系统的调度中移除，也许会被销毁，但我们不需要担心这些。

我们唯一需要担心的是*加入*。当一个线程在另一个`thread`对象上调用函数`x.join()`时，它会被放到睡眠状态，直到线程`x`返回。请注意，如果线程被困在无限循环中，我们就没那么幸运了！如果我们希望线程继续存在，直到它决定终止自己，我们可以调用`x.detach()`。这样做后，我们就不再对线程有外部控制。无论我们做出什么决定，我们*必须*始终*加入*或*分离*线程。如果我们不做这两者之一，`thread`对象的析构函数将调用`std::terminate()`，这将导致应用程序突然关闭。

当我们的主函数返回时，整个应用程序当然被终止。但与此同时，我们分离的线程`t3`仍在睡眠，然后将其*再见*消息打印到终端。操作系统并不在乎，它只是在不等待该线程完成的情况下终止了整个程序。这是我们需要考虑的事情。如果该附加线程必须完成一些重要工作，我们必须让主函数*等待*它。

# 使用 std::unique_lock 和 std::shared_lock 执行异常安全的共享锁定

由于线程的操作是与操作系统支持密切相关的事情，STL 为此提供了良好的操作系统无关接口，因此为线程之间的*同步*提供 STL 支持也是明智的。这样，我们不仅可以在没有外部库的情况下启动和停止线程，还可以使用来自单一统一库的抽象来同步它们：STL。

在这个示例中，我们将看看 STL 互斥锁类和 RAII 锁抽象。虽然我们在具体的示例实现中玩弄了其中一些，但我们也将概述 STL 提供的更多同步助手。

# 如何做...

我们将编写一个程序，该程序在其*独占*和*共享*模式下使用`std::shared_mutex`实例，并查看这意味着什么。此外，我们不调用锁定和解锁函数，而是使用 RAII 助手进行自动解锁：

1.  首先，我们需要包括所有必要的头文件。因为我们一直与时间文字一起使用 STL 函数和数据结构，所以我们声明我们使用`std`和`chrono_literal`命名空间：

```cpp
      #include <iostream>
      #include <shared_mutex>
      #include <thread>
      #include <vector>      

      using namespace std;
      using namespace chrono_literals;
```

1.  整个程序围绕一个共享互斥锁展开，因此为了简单起见，让我们定义一个全局实例：

```cpp
      shared_mutex shared_mut;
```

1.  我们将使用`std::shared_lock`和`std::unique_lock`的 RAII 助手。为了使它们的名称看起来不那么笨拙，我们为它们定义了短类型别名：

```cpp
      using shrd_lck = shared_lock<shared_mutex>;
      using uniq_lck = unique_lock<shared_mutex>;
```

1.  在开始主函数之前，我们定义了两个辅助函数，它们都尝试以*独占*模式锁定互斥锁。这个函数在共享互斥锁上实例化一个`unique_lock`实例。第二个构造函数参数`defer_lock`告诉对象保持锁定。否则，它的构造函数将尝试锁定互斥锁，然后阻塞直到成功。然后我们在`exclusive_lock`对象上调用`try_lock`。这个调用将立即返回，其布尔返回值告诉我们它是否获得了锁，还是互斥锁已经在其他地方被锁定：

```cpp
      static void print_exclusive()
      {
          uniq_lck l {shared_mut, defer_lock};

          if (l.try_lock()) {
              cout << "Got exclusive lock.n";
          } else {
              cout << "Unable to lock exclusively.n";
          }
      }
```

1.  另一个辅助函数也尝试以独占模式锁定互斥锁。它会阻塞直到获得锁。然后我们通过抛出异常来模拟一些错误情况（它只携带一个普通整数而不是更复杂的异常对象）。尽管这会导致我们持有锁定互斥锁的上下文立即退出，但互斥锁将会被设计上的`unique_lock`析构函数在任何情况下都会释放锁定：

```cpp
      static void exclusive_throw()
      {
          uniq_lck l {shared_mut};
          throw 123;
      }
```

1.  现在到主要功能。首先，我们打开另一个范围并实例化一个`shared_lock`实例。它的构造函数立即以`shared`模式锁定互斥锁。我们将在接下来的步骤中看到这意味着什么：

```cpp
      int main()
      {
          {
              shrd_lck sl1 {shared_mut};

              cout << "shared lock once.n";
```

1.  现在我们打开另一个作用域，并在同一个互斥体上实例化第二个`shared_lock`实例。现在我们有两个`shared_lock`实例，它们都持有互斥体的共享锁。实际上，我们可以在同一个互斥体上实例化任意多个`shared_lock`实例。然后我们调用`print_exclusive`，它试图以*独占*模式锁定互斥体。这不会成功，因为它已经以*共享*模式锁定了：

```cpp
              {
                  shrd_lck sl2 {shared_mut};

                  cout << "shared lock twice.n";

                  print_exclusive();
              }
```

1.  在离开最新的作用域后，`shared_lock` `sl2`的析构函数释放了它对互斥体的共享锁。`print_exclusive`函数将再次失败，因为互斥体仍处于共享锁定模式：

```cpp
              cout << "shared lock once again.n";

              print_exclusive();

          }
          cout << "lock is free.n";
```

1.  在离开另一个作用域后，所有`shared_lock`对象都被销毁，互斥体再次处于未锁定状态。*现在*我们终于可以以独占模式锁定互斥体。让我们通过调用`exclusive_throw`然后`print_exclusive`来做到这一点。请记住，我们在`exclusive_throw`中抛出异常。但是因为`unique_lock`是一个 RAII 对象，它给我们提供了异常安全性，无论我们如何从`exclusive_throw`返回，互斥体都将再次被解锁。这样`print_exclusive`将不会在错误地仍然锁定的互斥体上阻塞：

```cpp
          try {
              exclusive_throw();
          } catch (int e) {
              cout << "Got exception " << e << 'n';
          }

          print_exclusive();
      }
```

1.  编译和运行代码产生以下输出。前两行显示我们得到了两个共享锁实例。然后`print_exclusive`函数无法以独占模式锁定互斥体。在离开内部作用域并解锁第二个共享锁后，`print_exclusive`函数仍然失败。在离开另一个作用域后，最终释放了互斥体，`exclusive_throw`和`print_exclusive`最终能够锁定互斥体：

```cpp
      $ ./shared_lock 
      shared lock once.
      shared lock twice.
      Unable to lock exclusively.
      shared lock once again.
      Unable to lock exclusively.
      lock is free.
      Got exception 123
      Got exclusive lock.
```

# 工作原理...

当查看 C++文档时，首先会让人感到困惑的是有不同的互斥类和 RAII 锁辅助工具。在查看我们的具体代码示例之前，让我们总结一下 STL 为我们提供了什么。

# 互斥类

术语互斥体代表**mut**ual **ex**clusion。为了防止并发运行的线程以非协调的方式更改相同的对象，可能导致数据损坏，我们可以使用互斥对象。STL 提供了不同的互斥类，具有不同的特性。它们都有一个`lock`和一个`unlock`方法。

每当一个线程是第一个在之前未锁定的互斥体上调用`lock()`的线程时，它就拥有了互斥体。在这一点上，其他线程将在它们的`lock`调用上阻塞，直到第一个线程再次调用`unlock`。`std::mutex`正好可以做到这一点。

STL 中有许多不同的互斥类：

| **类型名称** | **描述** |
| --- | --- |
| `mutex` | 具有`lock`和`unlock`方法的标准互斥体。提供额外的非阻塞`try_lock`方法。 |
| `timed_mutex` | 与互斥体相同，但提供了额外的`try_lock_for`和`try_lock_until`方法，允许*超时*而不是永远阻塞。 |
| `recursive_mutex` | 与`mutex`相同，但如果一个线程已经锁定了它的实例，它可以在同一个互斥对象上多次调用`lock`而不会阻塞。在拥有线程调用`unlock`与调用`lock`的次数一样多后，它将被释放。 |
| `recursive_timed_mutex` | 提供了`timed_mutex`和`recursive_mutex`的特性。 |
| `shared_mutex` | 这个互斥体在这方面很特别，它可以以*独占*模式和*共享*模式锁定。在独占模式下，它显示与标准互斥体类相同的行为。如果一个线程以共享模式锁定它，其他线程也可以以共享模式锁定它。只要最后一个共享模式锁定所有者释放它，它就会被解锁。在共享模式锁定时，不可能获得独占所有权。这与`shared_ptr`的行为非常相似，只是它不管理内存，而是锁的所有权。 |
| `shared_timed_mutex` | 结合了`shared_mutex`和`timed_mutex`的特性，既可以进行独占模式也可以进行共享模式。 |

# 锁类

只要线程只锁定互斥锁，访问一些并发保护对象，然后再次解锁互斥锁，一切都很顺利。一旦一个健忘的程序员在某处忘记解锁互斥锁，或者在互斥锁仍然被锁定时抛出异常，事情就会很快变得混乱。在最好的情况下，程序会立即挂起，并且很快就会发现缺少的解锁调用。然而，这些错误与内存泄漏非常相似，当缺少显式的`delete`调用时也会发生内存泄漏。

在考虑内存管理时，我们有`unique_ptr`、`shared_ptr`和`weak_ptr`。这些辅助程序提供了非常方便的方法来避免内存泄漏。互斥锁也有这样的辅助程序。最简单的是`std::lock_guard`。可以按照以下方式使用：

```cpp
void critical_function()
{
    lock_guard<mutex> l {some_mutex};

    // critical section
}
```

`lock_guard`元素的构造函数接受一个互斥锁，在该互斥锁上立即调用`lock`。整个构造函数调用将阻塞，直到它获得互斥锁上的锁。在销毁时，它再次解锁互斥锁。这样做是为了防止`lock`/`unlock`循环出错，因为它会自动发生。

C++17 STL 提供了以下不同的 RAII 锁辅助程序。它们都接受一个模板参数，该参数应与互斥锁的类型相同（尽管自 C++17 以来，编译器可以自行推断该类型）：

| **名称** | **描述** |
| --- | --- |
| `lock_guard` | 该类除了构造函数和析构函数外，没有提供其他内容，它们`lock`和`unlock`一个互斥锁。 |
| `scoped_lock` | 类似于`lock_guard`，但在其构造函数中支持任意数量的互斥锁。在其析构函数中以相反的顺序释放它们。 |
| `unique_lock` | 以独占模式锁定互斥锁。构造函数还接受参数，指示它在锁定调用时超时而不是永远阻塞。还可以选择根本不锁定互斥锁，或者假定它已经被锁定，或者仅*尝试*锁定互斥锁。额外的方法允许在`unique_lock`锁的生命周期内锁定和解锁互斥锁。 |
| `shared_lock` | 与`unique_lock`相同，但所有操作都以共享模式应用于互斥锁。 |

虽然`lock_guard`和`scoped_lock`具有非常简单的接口，只包括构造函数和析构函数，但`unique_lock`和`shared_lock`更复杂，但也更灵活。在本章的后续配方中，我们将看到它们可以如何被用于不仅仅是简单的锁定区域。

现在让我们回到配方代码。虽然我们只在单线程上运行了代码，但我们已经看到了如何使用锁辅助程序。`shrd_lck`类型别名代表`shared_lock<shared_mutex>`，允许我们以共享模式多次锁定实例。只要`sl1`和`sl2`存在，就无法通过`print_exclusive`调用以独占模式锁定互斥锁。这仍然很简单。

现在让我们来看一下稍后在主函数中出现的独占锁定函数：

```cpp
int main()
{
    {
        shrd_lck sl1 {shared_mut};
        {
            shrd_lck sl2 {shared_mut};

            print_exclusive();
        }
        print_exclusive();
    }

    try {
        exclusive_throw();
    } catch (int e) {
        cout << "Got exception " << e << 'n';
    }
    print_exclusive();
}
```

一个重要的细节是，在从`exclusive_throw`返回后，`print_exclusive`函数能够再次锁定互斥锁，尽管`exclusive_throw`由于抛出的异常而没有干净地退出。

让我们再看一下`print_exclusive`，因为它使用了一个奇怪的构造函数调用：

```cpp
void print_exclusive()
{
    uniq_lck l {shared_mut, defer_lock};

    if (l.try_lock()) {
        // ...
    }
}
```

在这个过程中，我们不仅提供了`shared_mut`，还提供了`defer_lock`作为`unique_lock`的构造函数参数。`defer_lock`是一个空的全局对象，可以用来选择`unique_lock`的不锁定互斥锁的不同构造函数。通过这样做，我们能够稍后调用`l.try_lock()`，它不会阻塞。如果互斥锁已经被锁定，我们可以做其他事情。如果确实可能获得锁，我们仍然有析构函数在我们之后整理。

# 使用 std::scoped_lock 避免死锁

如果死锁发生在道路交通中，它们看起来会像以下情况：

![](img/e02fbc8d-bc97-496b-b388-f0ecbea0e5e5.png)

为了让交通流量再次畅通，我们要么需要一台大型起重机，随机从街道交叉口中心挑选一辆汽车并将其移走。如果这不可能，那么我们需要足够多的司机合作。死锁可以通过一个方向的所有司机向后倒车几米，为其他司机继续行驶腾出空间来解决。

在多线程程序中，当然需要程序员严格避免这种情况。然而，当程序真正复杂时，很容易在这方面失败。

在这个示例中，我们将编写故意引发死锁情况的代码。然后我们将看到如何编写代码，以获取导致其他代码陷入死锁的相同资源，但使用新的 C++17 中引入的 STL 锁类`std::scoped_lock`来避免这个错误。

# 如何做...

本节的代码包含两对应该由并发线程执行的函数，它们以互斥量的形式获取两个资源。一对引发死锁，另一对避免了死锁。在主函数中，我们将尝试它们：

1.  让我们首先包含所有需要的头文件，并声明我们使用`std`和`chrono_literals`命名空间：

```cpp
      #include <iostream>
      #include <thread>
      #include <mutex>      

      using namespace std;
      using namespace chrono_literals;
```

1.  然后我们实例化两个互斥对象，这是为了陷入死锁所必需的：

```cpp
      mutex mut_a;
      mutex mut_b;
```

1.  为了通过两个资源引发死锁，我们需要两个函数。一个函数尝试锁定互斥量 A，然后锁定互斥量 B，而另一个函数将以相反的顺序执行。通过让两个函数在锁定之间稍微休眠一会儿，我们可以确保这段代码永远在死锁上阻塞。（这是为了演示目的。如果我们重复启动程序，没有一些休眠行可能会成功运行而不会发生死锁。）

请注意，我们不使用`'n'`字符来打印换行，而是使用`endl`。`endl`不仅执行换行，还刷新了`cout`的流缓冲区，因此我们可以确保打印不会被堆积和延迟：

```cpp
      static void deadlock_func_1()
      {
          cout << "bad f1 acquiring mutex A..." << endl;
          lock_guard<mutex> la {mut_a};

          this_thread::sleep_for(100ms);

          cout << "bad f1 acquiring mutex B..." << endl;
          lock_guard<mutex> lb {mut_b};

          cout << "bad f1 got both mutexes." << endl;
      }
```

1.  如上一步所承诺的，`deadlock_func_2`看起来与`deadlock_func_1`完全相同，但是以相反的顺序锁定了互斥量 A 和 B：

```cpp
      static void deadlock_func_2()
      {
          cout << "bad f2 acquiring mutex B..." << endl;
          lock_guard<mutex> lb {mut_b};

          this_thread::sleep_for(100ms);

          cout << "bad f2 acquiring mutex A..." << endl;
          lock_guard<mutex> la {mut_a};

          cout << "bad f2 got both mutexes." << endl;
      }
```

1.  现在我们编写这两个函数的无死锁变体。它们使用`scoped_lock`类，该类锁定我们提供为构造函数参数的所有互斥量。它的析构函数再次解锁它们。在锁定互斥量时，它内部为我们应用了死锁避免策略。请注意，这两个函数仍然以相反的顺序使用互斥量 A 和 B：

```cpp
      static void sane_func_1()
      {
          scoped_lock l {mut_a, mut_b};

          cout << "sane f1 got both mutexes." << endl;
      }

      static void sane_func_2()
      {
          scoped_lock l {mut_b, mut_a};

          cout << "sane f2 got both mutexes." << endl;
      }
```

1.  在主函数中，我们将通过两种情况。首先，我们在多线程环境中使用*正常*函数：

```cpp
      int main()
      {
          {
              thread t1 {sane_func_1};
              thread t2 {sane_func_2};

              t1.join();
              t2.join();
          }
```

1.  然后我们使用不使用任何死锁避免策略的引发死锁的函数：

```cpp
          {
              thread t1 {deadlock_func_1};
              thread t2 {deadlock_func_2};

              t1.join();
              t2.join();
          }
      }
```

1.  编译和运行程序产生以下输出。前两行显示*正常*锁定函数场景有效，并且两个函数都能够返回而不会永远阻塞。另外两个函数陷入了死锁。我们可以看到这是一个死锁，因为我们看到打印行告诉我们各个线程尝试锁定互斥量 A 和 B，然后永远等待。两者都没有达到成功锁定两个互斥量的地方。我们可以让这个程序运行数小时、数天和数年，*什么*都不会发生。

这个应用程序需要从外部终止，例如通过按下*Ctrl* + *C*键：

```cpp
      $ ./avoid_deadlock 
      sane f1 got both mutexes
      sane f2 got both mutexes
      bad f2 acquiring mutex B...
      bad f1 acquiring mutex A...
      bad f1 acquiring mutex B...
      bad f2 acquiring mutex A...
```

# 它是如何工作的...

通过实现故意引发死锁的代码，我们看到了这种不希望的情况会发生得多么迅速。在一个大型项目中，多个程序员编写需要共享一组互斥保护资源的代码时，所有程序员都需要遵守*相同的顺序*来锁定和解锁互斥量。虽然这些策略或规则确实很容易遵循，但也很容易忘记。这个问题的另一个术语是*锁定顺序倒置*。

`scoped_lock`在这种情况下真的很有帮助。它是 C++17 中的新功能，工作方式与`lock_guard`和`unique_lock`相同：它的构造函数执行锁定，其析构函数执行互斥量的解锁。`scoped_lock`的特点是它可以使用*多个*互斥量来执行这个操作。

`scoped_lock`使用`std::lock`函数，该函数应用一种特殊的算法，对提供的所有互斥量执行一系列`try_lock`调用，以防止死锁。因此，可以完全安全地使用`scoped_lock`或在不同顺序下调用`std::lock`相同的一组锁。

# 同步并发使用 std::cout

在多线程程序中的一个不便之处是，我们必须实际上保护*每一个*它们修改的数据结构，使用互斥量或其他措施来防止不受控制的并发修改。

通常用于打印的一个数据结构是`std::cout`。如果多个线程同时访问`cout`，那么输出将以有趣的混合模式出现在终端上。为了防止这种情况，我们需要编写自己的函数，以并发安全的方式进行打印。

我们将学习如何提供一个由最少代码组成且与`cout`一样方便使用的`cout`包装器。

# 如何做...

在本节中，我们将实现一个程序，它可以从许多线程并发地打印到终端。为了防止消息由于并发而混乱，我们实现了一个小的辅助类，它在线程之间同步打印：

1.  和往常一样，首先是包含：

```cpp
      #include <iostream>
      #include <thread>
      #include <mutex>
      #include <sstream>
      #include <vector>      

      using namespace std;
```

1.  然后我们实现我们的辅助类，我们称之为`pcout`。`p`代表*parallel*，因为它可以在并行上下文中同步工作。`pcout`公开继承自`stringstream`。这样我们就可以在它的实例上使用`operator<<`。一旦`pcout`实例被销毁，它的析构函数会锁定一个互斥量，然后打印`stringstream`缓冲区的内容。我们将在下一步中看到如何使用它：

```cpp
      struct pcout : public stringstream {
          static inline mutex cout_mutex;

          ~pcout() {
              lock_guard<mutex> l {cout_mutex};
              cout << rdbuf();
              cout.flush();
          }
      };
```

1.  现在让我们编写两个可以由额外线程执行的函数。两者都接受线程 ID 作为参数。它们唯一的区别是第一个简单地使用`cout`进行打印。另一个看起来几乎相同，但是它不直接使用`cout`，而是实例化`pcout`。这个实例是一个临时对象，只在这行代码中存在。在所有`operator<<`调用执行完毕后，内部字符串流被填充了我们想要打印的内容。然后调用`pcout`实例的析构函数。我们已经看到析构函数的作用：它锁定所有`pcout`实例共享的特定互斥量并进行打印：

```cpp
      static void print_cout(int id)
      {
          cout << "cout hello from " << id << 'n';
      }

      static void print_pcout(int id)
      {
           pcout{} << "pcout hello from " << id << 'n';
      }
```

1.  让我们试一下。首先，我们将使用`print_cout`，它只使用`cout`进行打印。我们启动 10 个线程，它们同时打印它们的字符串并等待直到它们完成：

```cpp
      int main()
      {
          vector<thread> v;

          for (size_t i {0}; i < 10; ++i) {
              v.emplace_back(print_cout, i);
          }

          for (auto &t : v) { t.join(); }
```

1.  然后我们用`print_pcout`函数做同样的事情：

```cpp
          cout << "=====================n";

          v.clear();
          for (size_t i {0}; i < 10; ++i) {
              v.emplace_back(print_pcout, i);
          }

          for (auto &t : v) { t.join(); }
      }
```

1.  编译和运行程序产生以下结果。正如我们所看到的，前 10 个打印完全是乱码。这就是在没有锁定的情况下并发使用`cout`时的情况。程序的最后 10 行是`print_pcout`行，没有显示任何乱码的迹象。我们可以看到它们是从不同的线程打印出来的，因为它们的顺序在每次运行程序时都是随机的：

![](img/c9208c3b-6c0c-4187-aaba-dd51820c5c5e.png)

# 它的工作原理...

好的，我们已经构建了这个*"cout 包装器"*，它可以自动序列化并发打印尝试。它是如何工作的呢？

让我们以手动方式执行我们的`pcout`辅助程序所做的相同步骤，而不使用任何魔法。首先，它实例化一个字符串流并接受我们输入的内容：

```cpp
stringstream ss;
ss << "This is some printed line " << 123 << 'n';
```

然后它锁定一个全局可用的互斥量：

```cpp
{
    lock_guard<mutex> l {cout_mutex};
```

在这个锁定的范围内，它访问字符串流`ss`的内容，打印它，然后通过离开范围释放互斥锁。`cout.flush()`行告诉流对象立即打印到终端。没有这一行，程序可能会运行得更快，因为多个打印行可以被捆绑在一起，并在稍后一次运行中打印。在我们的示例中，我们希望立即看到所有输出行，所以我们使用`flush`方法：

```cpp
    cout << ss.rdbuf();
    cout.flush();
}
```

好的，这很简单，但如果我们不得不一遍又一遍地做同样的事情，那就太繁琐了。我们可以将`stringstream`的实例化缩短如下：

```cpp
stringstream{} << "This is some printed line " << 123 << 'n';
```

这实例化了一个字符串流对象，将我们想要打印的所有内容输入其中，然后再次销毁它。字符串流的生命周期仅缩短到这一行。之后，我们无法再打印它，因为我们无法访问它。哪段代码是最后能够访问流内容的？它是`stringstream`的析构函数。

我们不能修改`stringstream`实例的成员方法，但是我们可以通过继承将自己的类型包装在它周围来扩展它们：

```cpp
struct pcout : public stringstream {
    ~pcout() {
        lock_guard<mutex> l {cout_mutex};
        cout << rdbuf();
        cout.flush();
    }
};
```

这个类*仍然*是一个字符串流，我们可以像任何其他字符串流一样使用它。唯一的区别是它将锁定一个互斥锁，并使用`cout`打印自己的缓冲区。

我们还将`cout_mutex`对象移入结构`pcout`中作为静态实例，这样我们就可以在一个地方将它们捆绑在一起。

# 使用 std::call_once 安全地延迟初始化

有时我们有特定的代码部分，可以由多个线程在并行上下文中运行，但必须在执行实际函数之前执行一些*设置代码*。一个简单的解决方案是在程序进入可以不时执行并行代码的状态之前，只需执行现有的设置函数。

这种方法的缺点如下：

+   如果并行函数来自库，用户不能忘记调用设置函数。这并不会使库更容易使用。

+   如果设置函数在某种方式上是昂贵的，并且甚至可能不需要在并行函数不总是被使用的情况下执行，那么我们需要的是决定何时/是否运行它的代码。

在这个示例中，我们将看看`std::call_once`，它是一个帮助函数，以一种简单易用和优雅的隐式方式解决了这个问题。

# 如何做...

我们将编写一个程序，使用完全相同的代码启动多个线程。尽管它们被编程为执行完全相同的代码，但我们的示例设置函数只会被调用一次：

1.  首先，我们需要包括所有必要的头文件：

```cpp
      #include <iostream>
      #include <thread>
      #include <mutex>
      #include <vector>     

      using namespace std;
```

1.  我们稍后将使用`std::call_once`。为了使用它，我们需要在某个地方有一个`once_flag`的实例。它用于同步所有使用`call_once`的线程在特定函数上：

```cpp
      once_flag callflag;
```

1.  必须只执行一次的函数如下。它只打印一个感叹号：

```cpp
      static void once_print()
      {
          cout << '!';
      }
```

1.  所有线程将执行打印函数。我们首先通过`std::call_once`函数调用函数`once_print`。`call_once`需要我们之前定义的变量`callflag`。它将用它来编排线程：

```cpp
      static void print(size_t x)
      {
          std::call_once(callflag, once_print);
          cout << x;
      }
```

1.  好的，现在让我们启动 10 个使用`print`函数的线程：

```cpp
      int main()
      {
          vector<thread> v;

          for (size_t i {0}; i < 10; ++i) {
              v.emplace_back(print, i);
          }

          for (auto &t : v) { t.join(); }
          cout << 'n';
      }
```

1.  编译和运行产生以下输出。首先，我们看到`once_print`函数的感叹号。然后我们看到所有线程 ID。`call_once`不仅确保`once_print`只被调用一次。此外，它还同步了所有线程，以便在执行`once_print`之前不会打印任何 ID：

```cpp
      $ ./call_once
      !1239406758
```

# 工作原理...

`std:call_once`的工作原理类似于屏障。它维护对函数（或可调用对象）的访问。第一个到达它的线程将执行该函数。直到它完成，任何到达`call_once`行的其他线程都将被阻塞。在第一个线程从函数返回后，所有其他线程也将被释放。

为了组织这个小舞蹈，需要一个变量，其他线程可以从中确定它们是否必须等待，以及它们何时被释放。这就是我们的变量`once_flag callflag;`的作用。每个`call_once`行也需要一个`once_flag`实例作为参数，该参数在调用一次的函数之前。

另一个好处是：如果发生了这种情况，即所选用来执行`call_once`函数的线程*失败*，因为抛出了一些*异常*，那么下一个线程就可以再次执行该函数。希望下一次不会再抛出异常。

# 使用 std::async 将任务推送到后台执行

每当我们希望某些代码在后台执行时，我们可以简单地启动一个新线程来执行这些代码。在此期间，我们可以做其他事情，然后等待结果。这很简单：

```cpp
std::thread t {my_function, arg1, arg2, ...};
// do something else
t.join(); // wait for thread to finish
```

但接下来就会出现不便：`t.join()`不会给我们`my_function`的返回值。为了得到它，我们需要编写一个调用`my_function`并将其返回值存储在某个变量中的函数，该变量也可以被我们启动新线程的第一个线程访问。如果这种情况反复发生，那么我们就需要一遍又一遍地编写大量样板代码。

自 C++11 以来，我们有`std::async`可以为我们做这个工作，不仅如此。在这个示例中，我们将编写一个简单的程序，使用异步函数调用同时执行多个任务。由于`std::async`比单独使用更强大，我们将更仔细地研究它的不同方面。

# 如何做...

我们将实现一个程序，它可以同时执行多个不同的任务，但我们不是显式地启动线程，而是使用`std::async`和`std::future`：

1.  首先，我们包括所有必要的头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <map>
      #include <string>
      #include <algorithm>
      #include <iterator>
      #include <future>      

      using namespace std;
```

1.  我们实现了三个函数，它们与并行无关，但执行有趣的任务。第一个函数接受一个字符串，并创建该字符串中出现的所有字符的直方图：

```cpp
      static map<char, size_t> histogram(const string &s)
      {
          map<char, size_t> m;

          for (char c : s) { m[c] += 1; }

          return m;
      }
```

1.  第二个函数也接受一个字符串，并返回它的排序副本：

```cpp
      static string sorted(string s)
      {
          sort(begin(s), end(s));
          return s;
      }
```

1.  第三个函数计算它接受的字符串中存在多少元音字母：

```cpp
      static bool is_vowel(char c)
      {
          char vowels[] {"aeiou"};
          return end(vowels) != 
                 find(begin(vowels), end(vowels), c);
      }

      static size_t vowels(const string &s)
      {
          return count_if(begin(s), end(s), is_vowel);
      }
```

1.  在主函数中，我们将整个标准输入读入一个字符串。为了不将输入分割成单词，我们取消了`ios::skipws`。这样我们就得到了一个大字符串，无论输入包含多少空白。之后我们对结果字符串使用`pop_back`，因为这样我们得到了一个多余的终止`''`字符：

```cpp
      int main()
      {
          cin.unsetf(ios::skipws);
          string input {istream_iterator<char>{cin}, {}};
          input.pop_back();
```

1.  现在让我们从之前实现的所有函数中获取返回值。为了加快非常长输入的执行速度，我们*异步*启动它们。`std::async`函数接受一个策略、一个函数和该函数的参数。我们使用`launch::async`作为策略调用`histogram`、`sorted`和`vowels`（稍后我们将看到这意味着什么）。所有函数都以相同的输入字符串作为参数：

```cpp
          auto hist        (async(launch::async, 
                                  histogram, input));
          auto sorted_str  (async(launch::async, 
                                  sorted,    input));
          auto vowel_count (async(launch::async, 
                                  vowels,    input));
```

1.  `async`调用会立即返回，因为它们实际上并不执行我们的函数。相反，它们设置了同步结构，稍后将获取函数调用的结果。结果现在正在由额外的线程并发计算。与此同时，我们可以自由地做任何我们想做的事情，因为我们可以稍后获取这些值。返回值`hist`、`sorted_str`和`vowel_count`是函数`histogram`、`sorted`和`vowels`的返回类型，但它们被`std::async`包装在`future`类型中。这种类型的对象表明它们将在某个时间点包含它们的值。通过对它们所有使用`.get()`，我们可以使主函数阻塞，直到值到达，然后用它们进行打印：

```cpp
          for (const auto &[c, count] : hist.get()) {
              cout << c << ": " << count << 'n';
          }

          cout << "Sorted string: " 
               << quoted(sorted_str.get()) << 'n'
               << "Total vowels: "  
               << vowel_count.get()        << 'n';
      }
```

1.  编译和运行代码如下。我们使用一个简短的示例字符串，这并不值得并行化，但为了这个例子，代码仍然是并发执行的。此外，程序的整体结构与其天真的顺序版本相比并没有太大变化：

```cpp
      $ echo "foo bar baz foobazinga" | ./async 
       : 3
      a: 4
      b: 3
      f: 2
      g: 1
      i: 1
      n: 1
      o: 4
      r: 1
      z: 2
      Sorted string: "   aaaabbbffginoooorzz"
      Total vowels: 9
```

# 它是如何工作的...

如果我们没有使用`std::async`，串行未并行化的代码可能看起来就像这样简单：

```cpp
auto hist        (histogram(input));
auto sorted_str  (sorted(   input));
auto vowel_count (vowels(   input));

for (const auto &[c, count] : hist) {
    cout << c << ": " << count << 'n';
}
cout << "Sorted string: " << quoted(sorted_str) << 'n';
cout << "Total vowels: "  << vowel_count        << 'n';
```

我们为了并行化代码所做的唯一事情是将这三个函数调用包装在`async(launch::async, ...)`调用中。这样这三个函数就不会由我们当前运行的主线程执行。相反，`async`启动新线程并让它们并发执行函数。这样我们只执行启动另一个线程的开销，并且可以继续下一行代码，而所有的工作都在后台进行：

```cpp
auto hist        (async(launch::async, histogram, input));
auto sorted_str  (async(launch::async, sorted,    input));
auto vowel_count (async(launch::async, vowels,    input));

for (const auto &[c, count] : hist.get()) {
    cout << c << ": " << count << 'n';
}
cout << "Sorted string: " 
     << quoted(sorted_str.get()) << 'n'
     << "Total vowels: "  
     << vowel_count.get()        << 'n';
```

例如，`histogram`返回给我们一个 map 实例，而`async(..., histogram, ...)`返回给我们一个被包装在`future`对象中的 map。这个`future`对象在执行`histogram`函数返回之前是一种空的*占位符*。然后将生成的 map 放入`future`对象中，这样我们最终可以访问它。然后`get`函数给我们访问封装结果的权限。

让我们看另一个最小的例子。考虑以下代码片段：

```cpp
auto x (f(1, 2, 3));
cout << x;
```

与编写前面的代码相比，我们也可以这样做：

```cpp
auto x (async(launch::async, f, 1, 2, 3));
cout << x.get();
```

基本上就是这样。在标准 C++中执行后台任务可能从未如此简单。还有一件事需要解决：`launch::async`是什么意思？`launch::async`是一个标志，定义了启动策略。有两个策略标志，允许三种情况：

| **策略选择** | **含义** |
| --- | --- |
| `launch::async` | 保证函数由另一个线程执行。 |
| `launch::deferred` | 函数由同一个线程执行，但稍后执行（*延迟评估*）。当在 future 上调用`get`或`wait`时才会执行。如果*两者都没有*发生，函数根本不会被调用。 |
| `launch::async &#124; launch::deferred` | 两个标志都设置时，STL 的`async`实现可以自由选择要遵循哪种策略。如果没有提供策略，则这是默认选择。 |

通过只调用`async(f, 1, 2, 3)`而不带有策略参数，我们自动选择*两种*策略。然后，`async`的实现可以自由选择要使用哪种策略。这意味着我们无法*确定*是否启动了另一个线程，或者执行是否只是在当前线程中延迟进行。

# 还有更多...

确实还有最后一件事我们应该知道。假设我们编写如下代码：

```cpp
async(launch::async, f);
async(launch::async, g);
```

这可能是为了在并发线程中执行函数`f`和`g`（在这个例子中我们不关心它们的返回值），然后同时做不同的事情。在运行这样的代码时，我们会注意到代码在这些调用上*阻塞*，这很可能不是我们想要的。

那么为什么会阻塞呢？`async`不是关于非阻塞异步调用的吗？是的，但有一个特殊的特点：如果从带有`launch::async`策略的 async 调用中获得了一个 future，那么它的析构函数会执行*阻塞等待*。

这意味着这个简短示例中的两个 async 调用都是阻塞的，因为它们返回的 futures 的生命周期在同一行结束！我们可以通过将它们的返回值捕获到具有更长生命周期的变量中来解决这个问题。

# 使用 std::condition_variable 实现生产者/消费者习语

在这个配方中，我们将实现一个具有多个线程的典型生产者/消费者程序。总体思路是有一个线程生产项目并将它们放入队列。然后有另一个线程消费这些项目。如果没有东西可以生产，生产者线程就会休眠。如果队列中没有要消费的项目，消费者就会休眠。

由于两个线程都可以访问的队列在每次生产或消费项目时都会被两者修改，因此需要通过互斥锁来保护。

另一个需要考虑的事情是：如果队列中没有项目，消费者该怎么办？它是否每秒轮询队列，直到看到新项目？这是不必要的，因为我们可以让消费者等待由生产者触发的唤醒*事件*，每当有新项目时。

C++11 提供了一种称为`std::condition_variable`的很好的数据结构，用于这种类型的事件。在这个配方中，我们将实现一个简单的生产者/消费者应用程序，利用这一点。

# 如何做...

我们将实现一个简单的生产者/消费者程序，其中一个线程中有一个单一的值生产者，另一个线程中有一个单一的消费者线程：

1.  首先，我们需要执行所有需要的包含：

```cpp
      #include <iostream>
      #include <queue>
      #include <tuple>
      #include <condition_variable>
      #include <thread>      

      using namespace std;
      using namespace chrono_literals;
```

1.  我们实例化一个简单数值值队列，并称其为`q`。生产者将向其中推送值，消费者将从中取出值。为了同步两者，我们需要一个互斥锁。除此之外，我们实例化一个`condition_variable` `cv`。变量`finished`将是生产者告诉消费者不会再有更多值的方式：

```cpp
      queue<size_t>      q;
      mutex              mut;
      condition_variable cv;
      bool               finished {false};
```

1.  让我们首先实现生产者函数。它接受一个名为`items`的参数，限制了生产的最大项目数。在一个简单的循环中，它将为每个项目休眠 100 毫秒，这模拟了一些计算*复杂性*。然后我们锁定同步访问队列的互斥锁。在成功生产并插入队列后，我们调用`cv.notify_all()`。这个函数唤醒消费者。我们稍后会在消费者端看到这是如何工作的：

```cpp
      static void producer(size_t items) {
          for (size_t i {0}; i < items; ++i) {
              this_thread::sleep_for(100ms);
              {
                  lock_guard<mutex> lk {mut};
                  q.push(i);
              }
              cv.notify_all();
          }
```

1.  在生产完所有项目后，我们再次锁定互斥锁，因为我们将要更改设置`finished`位。然后我们再次调用`cv.notify_all()`：

```cpp
          {
              lock_guard<mutex> lk {mut};
              finished = true;
          }
          cv.notify_all();
      }
```

1.  现在我们可以实现消费者函数。它不带参数，因为它会盲目地消费，直到队列为空为止。在一个循环中，只要`finished`没有设置，它首先会锁定保护队列和`finished`标志的互斥锁。一旦获得锁，它就会调用`cv.wait`，并将锁和 lambda 表达式作为参数。lambda 表达式是一个断言，告诉生产者线程是否仍然存活，以及队列中是否有任何东西可以消费。

```cpp
      static void consumer() {
          while (!finished) {
              unique_lock<mutex> l {mut};

              cv.wait(l, [] { return !q.empty() || finished; });
```

1.  `cv.wait`调用解锁锁，并等待由断言函数描述的条件成立。然后，它再次锁定互斥锁，并从队列中消费所有东西，直到它为空。如果生产者仍然存活，它将再次迭代循环。否则，它将终止，因为`finished`被设置，这是生产者告知不再生产更多项目的方式：

```cpp
              while (!q.empty()) {
                  cout << "Got " << q.front() 
                       << " from queue.n";
                  q.pop();
              }
          }
      }
```

1.  在主函数中，我们启动一个生产者线程，它会生产 10 个项目，以及一个消费者线程。然后我们等待它们完成并终止程序：

```cpp
      int main() {
          thread t1 {producer, 10};
          thread t2 {consumer};
          t1.join();
          t2.join();
          cout << "finished!n";
      }
```

1.  编译和运行程序产生以下输出。当程序执行时，我们可以看到每行之间有一些时间（100 毫秒），因为生产项目需要一些时间：

```cpp
      $ ./producer_consumer
      Got 0 from queue.
      Got 1 from queue.
      Got 2 from queue.
      Got 3 from queue.
      Got 4 from queue.
      Got 5 from queue.
      Got 6 from queue.
      Got 7 from queue.
      Got 8 from queue.
      Got 9 from queue.
      finished!
```

# 它是如何工作的...

在这个配方中，我们简单地启动了两个线程。第一个线程生产项目并将它们放入队列。另一个从队列中取出项目。每当这些线程中的一个以任何方式触及队列时，它都会锁定共同的互斥锁`mut`，这对两者都是可访问的。通过这种方式，我们确保不会发生两个线程同时操纵队列状态的情况。

除了队列和互斥锁，我们通常声明了四个与生产者-消费者相关的变量：

```cpp
queue<size_t>      q;
mutex              mut;
condition_variable cv;
bool               finished {false};
```

变量`finished`很容易解释。当生产者完成生产固定数量的物品时，它被设置为`true`。当消费者看到这个变量为`true`时，它会消耗队列中的最后物品并停止消耗。但`condition_variable` `cv`是用来做什么的？我们在两个不同的上下文中使用了`cv`。一个上下文是*等待特定条件*，另一个是*发出该条件的信号*。

等待特定条件的消费者端看起来像这样。消费者线程循环执行一个块，首先在`unique_lock`中锁定互斥锁`mut`。然后调用`cv.wait`：

```cpp
while (!finished) {
    unique_lock<mutex> l {mut};

 cv.wait(l, [] { return !q.empty() || finished; });

    while (!q.empty()) {
        // consume
    }
}
```

这段代码*在某种程度上*等同于以下替代代码。我们很快会详细说明为什么它实际上并不相同：

```cpp
while (!finished) {
    unique_lock<mutex> l {mut};

 while (q.empty() && !finished) {
 l.unlock();
 l.lock();
 }

    while (!q.empty()) {
        // consume
    }
}
```

这意味着通常我们首先获取锁，然后检查我们的情况是什么：

1.  有可消耗的物品吗？那么保持锁定，消耗，释放锁定，然后重新开始。

1.  否则，如果没有可消耗的物品，但生产者仍然*活着*，则释放互斥锁以使生产者有机会向队列中添加物品。然后，尝试再次锁定它，希望情况会改变，我们会看到情况 1。

`cv.wait`行不等同于`while (q.empty() && ... )`构造的真正原因是，我们不能简单地循环执行`l.unlock(); l.lock();`。如果生产者线程在某段时间内处于非活动状态，那么这将导致互斥锁的持续锁定和解锁，这是没有意义的，因为这会不必要地消耗 CPU 周期。

像`cv.wait(lock, predicate)`这样的表达式将等待，直到`predicate()`返回`true`。但它并不是通过不断解锁和锁定`lock`来做到这一点的。为了唤醒一个在`condition_variable`对象的`wait`调用上阻塞的线程，另一个线程必须在同一对象上调用`notify_one()`或`notify_all()`方法。只有这样，等待的线程才会被唤醒，以检查`predicate()`是否成立。

`wait`调用检查谓词的好处是，如果有*虚假*唤醒调用，线程将立即再次进入睡眠状态。这意味着如果我们有太多的通知调用，它并不会真正影响程序流程（但可能会影响性能）。

在生产者端，我们在生产者将物品插入队列后和生产者生产最后一个物品并将`finished`标志设置为`true`后，只需调用`cv.notify_all()`。这足以引导消费者。

# 使用 std::condition_variable 实现多个生产者/消费者习语

让我们从上一个示例中解决生产者/消费者问题，并使其变得更加复杂：我们让*多个*生产者生产物品，*多个*消费者消耗它们。除此之外，我们定义队列不得超过最大大小。

这种方式不仅消费者必须不时休眠，如果队列中没有物品，生产者也必须不时休眠，当队列中有足够的物品时。

我们将看到如何使用多个`std::condition_variable`对象解决此问题，并且还将以与上一个示例略有不同的方式使用它们。

# 如何做...

在本节中，我们将实现一个程序，就像在上一个示例中一样，但这次有多个生产者和多个消费者：

1.  首先，我们需要包含所有需要的标头，并声明我们使用`std`和`chrono_literals`命名空间：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <sstream>
      #include <vector>
      #include <queue>
      #include <thread>
      #include <mutex>
      #include <condition_variable>
      #include <chrono>     

      using namespace std;
      using namespace chrono_literals;
```

1.  然后我们在本章的另一个示例中实现了同步打印助手，因为我们将进行大量并发打印：

```cpp
      struct pcout : public stringstream {
          static inline mutex cout_mutex;
          ~pcout() {
              lock_guard<mutex> l {cout_mutex};
              cout << rdbuf();
          }
      };
```

1.  所有生产者将值写入同一个队列，所有消费者也将从该队列中取出值。除了该队列，我们还需要一个保护队列和一个标志的互斥锁，该标志可以告诉我们生产是否在某个时刻停止：

```cpp
      queue<size_t> q;
      mutex         q_mutex; 
      bool          production_stopped {false};
```

1.  在这个程序中，我们将使用两个不同的`condition_variables`。在单个生产者/消费者配方中，我们有一个`condition_variable`告诉队列中有新物品。在这种情况下，我们要复杂一点。我们希望生产者生产，直到队列包含一定数量的*库存物品*。如果达到了库存量，它们将*休眠*。这样`go_consume`变量可以用来唤醒消费者，然后消费者可以再次用`go_produce`变量唤醒生产者：

```cpp
      condition_variable go_produce;
      condition_variable go_consume;
```

1.  生产者函数接受生产者 ID 号、要生产的物品总数和库存限制作为参数。然后它进入自己的生产循环。在那里，它首先锁定队列的互斥量，并在`go_produce.wait`调用中再次解锁。它等待队列大小低于`stock`阈值的条件：

```cpp
      static void producer(size_t id, size_t items, size_t stock)
      {
          for (size_t i = 0; i < items; ++i) {
              unique_lock<mutex> lock(q_mutex);
              go_produce.wait(lock, 
                  [&] { return q.size() < stock; });
```

1.  生产者被唤醒后，它生产一个物品并将其推入队列。队列值是从表达式`id * 100 + i`计算出来的。这样我们以后可以看到哪个生产者生产了它，因为数字中的百位数是生产者 ID。我们还将生产事件打印到终端。打印的格式可能看起来奇怪，但它将与终端中的消费者输出很好地对齐：

```cpp
              q.push(id * 100 + i);

              pcout{} << "   Producer " << id << " --> item "
                      << setw(3) << q.back() << 'n';
```

1.  生产后，我们可以唤醒正在睡眠的消费者。90 毫秒的睡眠时间模拟了生产物品需要一些时间：

```cpp
              go_consume.notify_all();
              this_thread::sleep_for(90ms);
           }

           pcout{} << "EXIT: Producer " << id << 'n';
      }
```

1.  现在是消费者函数，它只接受一个消费者 ID 作为参数。如果生产还没有停止，或者队列不为空，它将继续等待物品。如果队列为空，但生产还没有停止，那么可能很快会有新的物品：

```cpp
      static void consumer(size_t id)
      {
           while (!production_stopped || !q.empty()) {
               unique_lock<mutex> lock(q_mutex);
```

1.  在锁定队列互斥量后，我们再次解锁它，以便在`go_consume`事件变量上等待。lambda 表达式参数描述了我们希望在队列包含物品时从等待调用中返回。第二个参数`1s`表示我们不想永远等待。如果超过 1 秒，我们希望退出等待函数。我们可以区分`wait_for`函数返回的原因，因为谓词条件成立，或者因为超时而退出，因为在超时的情况下它将返回`false`。如果队列中有新物品，我们会消耗它们并将此事件打印到终端：

```cpp
               if (go_consume.wait_for(lock, 1s, 
                       [] { return !q.empty(); })) {
                   pcout{} << "                  item "
                           << setw(3) << q.front() 
                           << " --> Consumer "
                           << id << 'n';
                   q.pop();
```

1.  在物品消耗后，我们通知生产者并睡眠 130 毫秒，以模拟消耗物品也需要时间：

```cpp
                  go_produce.notify_all();
                  this_thread::sleep_for(130ms);
              }
          }

          pcout{} << "EXIT: Producer " << id << 'n';
      }
```

1.  在主函数中，我们为工作线程实例化一个向量，另一个为消费者线程：

```cpp
      int main()
      {
          vector<thread> workers;
          vector<thread> consumers;
```

1.  然后我们生成三个生产者线程和五个消费者线程：

```cpp
          for (size_t i = 0; i < 3; ++i) {
              workers.emplace_back(producer, i, 15, 5);
          }

          for (size_t i = 0; i < 5; ++i) {
              consumers.emplace_back(consumer, i);
          }
```

1.  首先让生产者线程完成。一旦它们全部返回，我们设置`production_stopped`标志，这将导致消费者也完成。我们需要收集它们，然后我们可以退出程序：

```cpp
          for (auto &t : workers)   { t.join(); }
          production_stopped = true;
          for (auto &t : consumers) { t.join(); }
      }
```

1.  编译和运行程序会产生以下输出。输出非常长，这就是为什么在这里进行截断。我们可以看到生产者不时进入睡眠状态，让消费者吃掉一些物品，直到它们最终再次生产。改变生产者/消费者的等待时间以及操纵生产者/消费者和库存物品的数量是很有趣的，因为这完全改变了输出模式：

```cpp
      $ ./multi_producer_consumer
         Producer 0 --> item   0
         Producer 1 --> item 100
                        item   0 --> Consumer 0
         Producer 2 --> item 200
                        item 100 --> Consumer 1
                        item 200 --> Consumer 2
         Producer 0 --> item   1
         Producer 1 --> item 101
                        item   1 --> Consumer 0
      ...
         Producer 0 --> item  14
      EXIT: Producer 0
         Producer 1 --> item 114
      EXIT: Producer 1
                        item  14 --> Consumer 0
         Producer 2 --> item 214
      EXIT: Producer 2
                        item 114 --> Consumer 1
                        item 214 --> Consumer 2
      EXIT: Consumer 2
      EXIT: Consumer 3
      EXIT: Consumer 4
      EXIT: Consumer 0
      EXIT: Consumer 1
```

# 工作原理...

这个配方是前一个配方的扩展。我们不仅同步一个生产者和一个消费者，而是实现了一个同步`M`个生产者和`N`个消费者的程序。除此之外，如果没有物品留给消费者，不仅消费者会进入睡眠状态，一旦物品队列变得*太长*，生产者也会进入睡眠状态。

当多个消费者等待相同的队列填满时，这通常也适用于一个生产者/一个消费者场景中的消费者代码。只要只有一个线程锁定保护队列的互斥锁，然后从中取出项目，代码就是安全的。无论有多少线程同时等待锁定，都无关紧要。生产者也是如此，因为在这两种情况下，唯一重要的是队列永远不会被多个线程同时访问。

这个程序之所以比只有一个生产者/一个消费者的例子更复杂，是因为我们让生产者线程在项目队列长度达到一定阈值时停止。为了满足这一要求，我们实现了两个不同的信号，它们各自拥有自己的`condition_variable`：

1.  `go_produce`信号着事件，队列没有完全填满到最大，生产者可以再次填满它。

1.  `go_consume`信号着事件，队列达到最大长度，消费者可以再次消费项目。

这样，生产者将项目填入队列，并向消费者线程发出`go_consume`事件的信号，消费者线程在以下行上等待：

```cpp
if (go_consume.wait_for(lock, 1s, [] { return !q.empty(); })) {
    // got the event without timeout
}
```

另一方面，生产者会在以下行上等待，直到被允许再次生产：

```cpp
go_produce.wait(lock, [&] { return q.size() < stock; });
```

一个有趣的细节是，我们不让消费者永远等待。在`go_consume.wait_for`调用中，我们额外添加了一个 1 秒的超时参数。这是消费者的退出机制：如果队列空闲超过一秒，可能没有活跃的生产者了。

为了简化起见，代码试图始终将队列长度保持在最大值。更复杂的程序可以让消费者线程在队列只有其最大长度的一半时推送唤醒通知，*只有*在队列中的项目数量仍然足够时，生产者才会被提前唤醒。这样，生产者就不会在队列中仍有足够的项目时被不必要地提前唤醒。

`condition_variable`为我们优雅地解决了以下情况：如果消费者触发了`go_produce`通知，可能会有一大群生产者竞相生产下一个项目。如果只缺一个项目，那么只会有一个生产者生产它。如果所有生产者在`go_produce`事件触发后总是立即生产一个项目，我们经常会看到队列填满到允许的最大值以上的情况。

假设我们的队列中有`(max - 1)`个项目，并且希望生产一个新项目，以便再次填满队列。无论消费者线程调用`go_produce.notify_one()`（只唤醒一个等待线程）还是`go_produce.notify_all()`（唤醒*所有*等待线程），我们保证只有一个生产者线程会退出`go_produce.wait`调用，因为对于所有其他生产者线程来说，一旦它们在被唤醒后获得互斥锁，`q.size() < stock`等待条件就不再成立。

# 使用 std::async 并行化 ASCII Mandelbrot 渲染器

还记得第二十三章中的*ASCII Mandelbrot 渲染器*吗，*STL 算法的高级用法*？在这个配方中，我们将使用线程来加速其计算时间。

首先，我们将修改原始程序中限制每个选定坐标迭代次数的行。这将使程序*变慢*，其结果*比我们实际上可以在终端上显示的更准确*，但这样我们就有了一个很好的并行化目标。

然后，我们将对程序进行轻微修改，看整个程序如何运行得更快。在这些修改之后，程序将使用`std::async`和`std::future`运行。为了充分理解这个配方，理解原始程序是至关重要的。

# 如何做...

在这一部分，我们将采用我们在第二十三章中实现的 ASCII Mandelbrot 分形渲染器，*STL 算法的高级用法*。首先，我们将通过增加计算限制来使计算时间更长。然后，我们通过对程序进行四处小改动来实现加速，以便并行化：

1.  为了跟随这些步骤，最好是直接从其他的配方中复制整个程序。然后按照以下步骤的说明进行所有必要的调整。所有与原始程序的不同之处都用*加粗*标出。

第一个变化是一个额外的头文件，`<future>`:

```cpp
      #include <iostream>
      #include <algorithm>
      #include <iterator>
      #include <complex>
      #include <numeric>
      #include <vector>
 #include <future>      

      using namespace std;
```

1.  `scaler`和`scaled_cmplx`函数不需要任何更改：

```cpp
      using cmplx = complex<double>;

      static auto scaler(int min_from, int max_from,
                         double min_to, double max_to)
      {
          const int w_from {max_from - min_from};
          const double w_to {max_to - min_to};
          const int mid_from {(max_from - min_from) / 2 + min_from};
          const double mid_to {(max_to - min_to) / 2.0 + min_to};

          return [=] (int from) {
              return double(from - mid_from) / w_from * w_to + mid_to;
          };
      }

      template <typename A, typename B>
      static auto scaled_cmplx(A scaler_x, B scaler_y)
      {
          return = {
              return cmplx{scaler_x(x), scaler_y(y)};
          };
      }
```

1.  在函数`mandelbrot_iterations`中，我们只是增加迭代次数，以使程序更加计算密集：

```cpp
      static auto mandelbrot_iterations(cmplx c)
      {
          cmplx z {};
          size_t iterations {0};
          const size_t max_iterations {100000};
          while (abs(z) < 2 && iterations < max_iterations) {
              ++iterations;
              z = pow(z, 2) + c;
          }
          return iterations;
      }
```

1.  然后我们有一个主函数的一部分，它再次不需要任何更改：

```cpp
      int main()
      {
          const size_t w {100};
          const size_t h {40};

          auto scale (scaled_cmplx(
              scaler(0, w, -2.0, 1.0),
              scaler(0, h, -1.0, 1.0)
          ));

          auto i_to_xy (= { 
              return scale(x % w, x / w); 
          });
```

1.  在`to_iteration_count`函数中，我们不再直接调用`mandelbrot_iterations(x_to_xy(x))`，而是使用`std::async`异步调用：

```cpp
          auto to_iteration_count (= {
              return async(launch::async,
 mandelbrot_iterations, i_to_xy(x));
          });
```

1.  在最后一个变化之前，函数`to_iteration_count`返回了特定坐标需要 Mandelbrot 算法收敛的迭代次数。现在它返回一个将来会包含相同值的`future`变量，因为它是异步计算的。因此，我们需要一个保存所有未来值的向量，所以让我们添加一个。我们提供给`transform`的输出迭代器作为新输出向量`r`的起始迭代器：

```cpp
          vector<int> v (w * h);
 vector<future<size_t>> r (w * h);
          iota(begin(v), end(v), 0);
          transform(begin(v), end(v), begin(r), 
                    to_iteration_count);
```

1.  `accumulate`调用不再接受`size_t`值作为第二个参数，而是`future<size_t>`值。我们需要调整为这种类型（如果一开始就使用`auto&`作为它的类型，那么这甚至是不必要的），然后我们需要调用`x.get()`来等待值的到来，而不是之前直接访问`x`：

```cpp
          auto binfunc ([w, n{0}] (auto output_it, future<size_t> &x) 
                  mutable {
              *++output_it = (x.get() > 50 ? '*' : ' ');
              if (++n % w == 0) { ++output_it = 'n'; }
              return output_it;
          });

          accumulate(begin(r), end(r), 
                     ostream_iterator<char>{cout}, binfunc);
      }
```

1.  编译和运行给我们与以前相同的输出。唯一有趣的区别是执行速度。如果我们也增加原始版本程序的迭代次数，那么并行化版本应该计算得更快。在我的计算机上，有四个 CPU 核心和超线程（导致 8 个虚拟核心），我用 GCC 和 clang 得到了不同的结果。最佳加速比为`5.3`，最差为`3.8`。当然，结果在不同的机器上也会有所不同。

# 工作原理...

首先要理解整个程序，因为这样就清楚了所有的 CPU 密集型工作都发生在主函数的一行代码中：

```cpp
transform(begin(v), end(v), begin(r), to_iteration_count);
```

向量`v`包含了所有映射到复坐标的索引，然后用 Mandelbrot 算法迭代。每次迭代的结果都保存在向量`r`中。

在原始程序中，这是消耗所有处理时间来计算分形图像的单行代码。它之前的所有代码都是设置工作，之后的所有代码都是用于打印。这意味着并行化这一行是提高性能的关键。

并行化这个问题的一种可能方法是将从`begin(v)`到`end(v)`的整个线性范围分成相同大小的块，并均匀分配到所有核心上。这样所有核心将共享工作量。如果我们使用带有并行执行策略的`std::transform`的并行版本，这将正好是这种情况。不幸的是，这不是*这个*问题的正确策略，因为 Mandelbrot 集中的每个点都显示出非常独特的迭代次数。

我们的方法是使每个单独的向量项（代表终端上的单个打印字符单元）成为一个异步计算的`future`值。由于源向量和目标向量都有`w * h`个项，也就是在我们的情况下是`100 * 40`，我们有一个包含 4000 个异步计算值的向量。如果我们的系统有 4000 个 CPU 核心，那么这意味着我们启动了 4000 个线程，这些线程真正同时进行 Mandelbrot 迭代。在具有较少核心的普通系统上，CPU 将依次处理每个核心的一个异步项。

`transform`调用与`to_iteration_count`的异步版本本身*不进行计算*，而是设置线程和 future 对象，它几乎立即返回。程序的原始版本在这一点上被阻塞，因为迭代时间太长。

程序的并行化版本当然也会在*某个地方*阻塞。在终端上打印所有值的函数必须访问 future 中的结果。为了做到这一点，它对所有值调用`x.get()`。这就是诀窍：当它等待第一个值被打印时，很多其他值同时被计算。因此，如果第一个 future 的`get()`调用返回，下一个 future 可能已经准备好打印了！

如果`w * h`的结果是非常大的数字，那么创建和同步所有这些 future 将会有一些可测量的开销。在这种情况下，开销并不太显著。在我的笔记本电脑上，配备了一颗 Intel i7 处理器，有 4 个*超线程*核心（结果是八个虚拟核心），与原始程序相比，这个程序的并行版本运行速度快了 3-5 倍。理想的并行化会使其快 8 倍。当然，这种加速会因不同的计算机而异，因为它取决于很多因素。

# 使用 std::future 实现一个微小的自动并行化库

大多数复杂的任务可以分解成子任务。从所有子任务中，我们可以绘制一个描述哪个子任务依赖于其他子任务以完成更高级任务的**有向无环图**（**DAG**）。例如，假设我们想要生成字符串`"foo bar foo bar this that "`，我们只能通过创建单词并将其与其他单词或自身连接来实现。假设这个功能由三个基本函数`create`、`concat`和`twice`提供。

考虑到这一点，我们可以绘制以下 DAG，以可视化它们之间的依赖关系，以便获得最终结果：

![](img/0648ebd7-03e5-4c9c-8ee2-8ed58cb25773.png)

在代码中实现这一点时，很明显一切都可以在一个 CPU 核心上以串行方式实现。或者，所有不依赖其他子任务或其他已经完成的子任务的子任务可以在多个 CPU 核心上*并发*执行。

编写这样的代码可能看起来有点乏味，即使使用`std::async`，因为子任务之间的依赖关系需要被建模。在本配方中，我们将实现两个小型库辅助函数，帮助将普通函数`create`、`concat`和`twice`转换为异步工作的函数。有了这些，我们将找到一种真正优雅的方式来设置依赖图。在执行过程中，图将以一种*看似智能*的方式并行化，以尽可能快地计算结果。

# 如何做...

在本节中，我们将实现一些函数，模拟相互依赖的计算密集型任务，并让它们尽可能并行运行：

1.  让我们首先包含所有必要的头文件：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <thread>
      #include <string>
      #include <sstream>
      #include <future>      

      using namespace std;
      using namespace chrono_literals;
```

1.  我们需要同步对`cout`的并发访问，因此让我们使用本章其他配方中的同步助手：

```cpp
      struct pcout : public stringstream {
          static inline mutex cout_mutex;

          ~pcout() {
              lock_guard<mutex> l {cout_mutex};
              cout << rdbuf();
              cout.flush();
          }
      };
```

1.  现在让我们实现三个转换字符串的函数。第一个函数将从 C 字符串创建一个`std::string`对象。我们让它休眠 3 秒来模拟字符串创建是计算密集型的：

```cpp
      static string create(const char *s)
      {
          pcout{} << "3s CREATE " << quoted(s) << 'n';
          this_thread::sleep_for(3s);
          return {s};
      }
```

1.  下一个函数接受两个字符串对象作为参数并返回它们的连接。我们给它 5 秒的等待时间来模拟这是一个耗时的任务：

```cpp
      static string concat(const string &a, const string &b)
      {
          pcout{} << "5s CONCAT "
                  << quoted(a) << " "
                  << quoted(b) << 'n';
          this_thread::sleep_for(5s);
          return a + b;
      }
```

1.  最后一个计算密集型的函数接受一个字符串并将其与自身连接。这将花费 3 秒的时间：

```cpp
      static string twice(const string &s)
      {
          pcout{} << "3s TWICE " << quoted(s) << 'n';
          this_thread::sleep_for(3s);
          return s + s;
      }
```

1.  现在我们已经可以在串行程序中使用这些函数了，但我们想要实现一些优雅的自动并行化。所以让我们为此实现一些辅助函数。*请注意*，接下来的三个函数看起来非常复杂。`asynchronize`接受一个函数`f`并返回一个可调用对象来捕获它。我们可以用任意数量的参数调用这个可调用对象，然后它将这些参数与`f`一起捕获在另一个可调用对象中返回给我们。然后可以无需参数调用这个最后的可调用对象。它会异步地使用捕获的所有参数调用`f`：

```cpp
      template <typename F>
      static auto asynchronize(F f)
      {
          return f {
              return [=] () {
                  return async(launch::async, f, xs...);
              };
          };
      }
```

1.  下一个函数将被我们在下一步中声明的函数使用。它接受一个函数`f`，并将其捕获在一个可调用对象中返回。该对象可以用一些 future 对象调用。然后它将在所有的 future 上调用`.get()`，对它们应用`f`，并返回其结果：

```cpp
      template <typename F>
      static auto fut_unwrap(F f)
      {
          return f {
              return f(xs.get()...);
          };
      }
```

1.  最后一个辅助函数也接受一个函数`f`。它返回一个可调用对象来捕获`f`。这个可调用对象可以用任意数量的可调用对象作为参数调用，然后它将这些与`f`一起捕获在另一个可调用对象中返回。然后可以无需参数调用这个最后的可调用对象。它会异步地调用捕获在`xs...`包中的所有可调用对象。这些将返回 future，需要使用`fut_unwrap`来解开。未来的解开和对来自未来的实际值应用真实函数`f`的实际应用再次使用`std::async`异步进行：

```cpp
      template <typename F>
      static auto async_adapter(F f)
      {
          return f {
              return [=] () {
                  return async(launch::async, 
                               fut_unwrap(f), xs()...);
              };
          };
      }
```

1.  好的，这可能有点疯狂，有点像电影《盗梦空间》，因为它使用了返回 lambda 表达式的 lambda 表达式。我们稍后会对这段代码进行非常详细的解释。现在让我们将函数`create`、`concat`和`twice`改为异步的。函数`async_adapter`使一个完全正常的函数等待未来的参数并返回未来的结果。它是一种从同步到异步世界的翻译包装。我们将其应用于`concat`和`twice`。我们必须在`create`上使用`asynchronize`，因为它应该返回一个 future，但我们将用实际值而不是 future 来提供它。任务依赖链必须从`create`调用开始：

```cpp
      int main()
      {
          auto pcreate (asynchronize(create));
          auto pconcat (async_adapter(concat));
          auto ptwice  (async_adapter(twice));
```

1.  现在我们有了自动并行化的函数，它们的名称与原来的同步函数相同，但带有`p`前缀。现在让我们建立一个复杂的依赖树示例。首先，我们创建字符串`"foo "`和`"bar "`，然后立即将它们连接成`"foo bar "`。然后使用`twice`将这个字符串与自身连接。然后我们创建字符串`"this "`和`"that "`，将它们连接成`"this that "`。最后，我们将结果连接成`"foo bar foo bar this that "`。结果将保存在变量`callable`中。然后最后调用`callable().get()`来开始计算并等待其返回值，以便打印出来。在调用`callable()`之前不会进行任何计算，而在此调用之后，所有的魔法就开始了：

```cpp
          auto result (
              pconcat(
                  ptwice(
                      pconcat(
                          pcreate("foo "),
                          pcreate("bar "))),
                  pconcat(
                      pcreate("this "),
                      pcreate("that "))));

          cout << "Setup done. Nothing executed yet.n";

          cout << result().get() << 'n';
      }
```

1.  编译和运行程序显示所有`create`调用同时执行，然后执行其他调用。看起来它们被智能地调度了。整个程序运行了 16 秒。如果步骤不是并行执行的，完成需要 30 秒。请注意，我们需要至少四个 CPU 核心的系统才能同时执行所有`create`调用。如果系统的 CPU 核心较少，那么一些调用将不得不共享 CPU，这当然会消耗更多时间。

```cpp
      $ ./chains 
      Setup done. Nothing executed yet.
      3s CREATE "foo "
      3s CREATE "bar "
      3s CREATE "this "
      3s CREATE "that "
      5s CONCAT "this " "that "
      5s CONCAT "foo " "bar "
      3s TWICE  "foo bar "
      5s CONCAT "foo bar foo bar " "this that "
      foo bar foo bar this that
```

# 它是如何工作的...

这个程序的普通串行版本，没有任何`async`和`future`魔法，看起来像这样：

```cpp
int main()
{
    string result {
        concat(
            twice(
                concat(
                    create("foo "),
                    create("bar "))),
            concat(
                create("this "),
                create("that "))) };

    cout << result << 'n';
}
```

在这个示例中，我们编写了辅助函数`async_adapter`和`asynchronize`，帮助我们从`create`、`concat`和`twice`创建新函数。我们称这些新的异步函数为`pcreate`、`pconcat`和`ptwice`。

让我们首先忽略`async_adapter`和`asynchronize`的实现复杂性，先看看这给我们带来了什么。

串行版本看起来类似于这段代码：

```cpp
string result {concat( ... )};
cout << result << 'n';
```

并行化版本看起来类似于以下内容：

```cpp
auto result (pconcat( ... ));
cout << result().get() << 'n';
```

好的，现在我们来到了复杂的部分。并行化结果的类型不是`string`，而是一个返回`future<string>`的可调用对象，我们可以在其上调用`get()`。这一开始可能看起来很疯狂。

那么，我们到底是如何以及*为什么*最终得到了返回 future 的可调用对象？我们的`create`、`concat`和`twice`方法的问题在于它们很*慢*。（好吧，我们人为地让它们变慢，因为我们试图模拟消耗大量 CPU 时间的真实任务）。但我们发现描述数据流的依赖树有独立的部分可以并行执行。让我们看两个示例调度：

![](img/8745f882-ba89-481e-bfcd-06203d12370f.png)

在左侧，我们看到一个*单核*调度。所有函数调用必须一个接一个地执行，因为我们只有一个 CPU。这意味着，当`create`花费 3 秒，`concat`花费 5 秒，`twice`花费 3 秒时，获取最终结果需要 30 秒。

在右侧，我们看到一个*并行调度*，在这个调度中尽可能多地并行执行函数调用之间的依赖关系。在一个理想的拥有四个核心的世界中，我们可以同时创建所有子字符串，然后连接它们等等。以最佳并行调度获得结果的最短时间是 16 秒。如果函数调用本身不能更快，我们就无法更快地进行。只有四个 CPU 核心，我们就可以实现这个执行时间。我们可以明显地实现了最佳调度。这是如何实现的？

我们可以天真地写下以下代码：

```cpp
auto a (async(launch::async, create, "foo "));
auto b (async(launch::async, create, "bar "));
auto c (async(launch::async, create, "this "));
auto d (async(launch::async, create, "that "));
auto e (async(launch::async, concat, a.get(), b.get()));
auto f (async(launch::async, concat, c.get(), d.get()));
auto g (async(launch::async, twice, e.get()));
auto h (async(launch::async, concat, g.get(), f.get()));
```

这是`a`、`b`、`c`和`d`的良好起点，它们代表了最初的四个子字符串。这些都是在后台异步创建的。不幸的是，这段代码在初始化`e`的地方阻塞了。为了连接`a`和`b`，我们需要在它们两个上调用`get()`，这会*阻塞*直到这些值准备好。显然，这不是一个好主意，因为并行化在第一次`get()`调用时就停止了。我们需要一个更好的策略。

好的，让我们先展开我们编写的复杂辅助函数。第一个是`asynchronize`：

```cpp
template <typename F>
static auto asynchronize(F f)
{
    return f {
        return [=] () {
            return async(launch::async, f, xs...);
        };
    };
}
```

当我们有一个函数`int f(int, int)`时，我们可以这样做：

```cpp
auto f2 ( asynchronize(f) );
auto f3 ( f2(1, 2) );
auto f4 ( f3() );
int result { f4.get() };
```

`f2`是我们的`f`的异步版本。它可以用与`f`相同的参数调用，因为它*模仿*了`f`。然后它返回一个可调用对象，我们将其保存在`f3`中。`f3`现在捕获了`f`和参数`1, 2`，但它还没有调用任何东西。这只是关于捕获的。 

当我们现在调用`f3()`时，最终我们得到一个 future，因为`f3()`执行了`async(launch::async, **f, 1, 2**);`调用！从这个意义上说，`f3`的语义意思是“*取得捕获的函数和参数，然后将它们一起抛到`std::async`中*”。

不接受任何参数的内部 lambda 表达式给了我们一个间接引用。有了它，我们可以为并行调度设置工作，但不必调用任何阻塞的东西，*至少*目前还不用。我们在更复杂的函数`async_adapter`中也遵循相同的原则：

```cpp
template <typename F>
static auto async_adapter(F f)
{
    return f {
        return [=] () {
            return async(launch::async, fut_unwrap(f), xs()...);
        };
    };
}
```

这个函数首先返回一个模仿`f`的函数，因为它接受相同的参数。然后该函数返回一个可调用对象，再次不接受任何参数。然后这个可调用对象最终与其他辅助函数不同。

`async(launch::async, fut_unwrap(f), xs()...);`这一行是什么意思？`xs()...`部分意味着，保存在包`xs`中的所有参数都被假定为可调用对象（就像我们一直在创建的那些对象！），因此它们都被调用而不带参数。我们一直在产生的这些可调用对象本身产生未来值，我们可以在其上调用`get()`。这就是`fut_unwrap`发挥作用的地方：

```cpp
template <typename F>
static auto fut_unwrap(F f)
{
    return f {
        return f(xs.get()...);
    };
}
```

`fut_unwrap`只是将函数`f`转换为一个接受一系列参数的函数对象。然后这个函数对象调用`.get()`，最后将它们转发给`f`。

慢慢消化这一切。当我们在主函数中使用时，`auto result (pconcat(...));`调用链只是构造了一个包含所有函数和所有参数的大型可调用对象。此时还没有进行任何`async`调用。然后，当我们调用`result()`时，我们*释放了一小波*`async`和`.get()`调用，它们恰好按照正确的顺序来避免相互阻塞。事实上，在所有`async`调用都已经分派之前，没有`get()`调用会发生。

最后，我们终于可以在`result()`返回的未来值上调用`.get()`，然后我们就得到了我们的最终字符串。
