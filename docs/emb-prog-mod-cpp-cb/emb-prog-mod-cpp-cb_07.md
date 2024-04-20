# 多线程和同步

嵌入式平台涵盖了广阔的计算能力领域。有些微控制器只有几千字节的内存；有些功能强大的**系统级芯片**（**SoCs**）有几千兆字节的内存；还有一些多核 CPU 能够同时运行许多应用程序。

随着嵌入式开发人员可用的计算资源增加，以及他们可以构建的更复杂的应用程序，多线程支持变得非常重要。开发人员需要知道如何并行化他们的应用程序，以有效地利用所有 CPU 核心。我们将学习如何编写能够以高效和安全的方式利用所有可用 CPU 核心的应用程序。

在本章中，我们将涵盖以下主题：

+   探索 C++中的线程支持

+   探索数据同步

+   使用条件变量

+   使用原子变量

+   使用 C++内存模型

+   探索无锁同步

+   在共享内存中使用原子变量

+   探索异步函数和期货

这些示例可以用作构建自己的高效多线程和多进程同步代码的示例。

# 探索 C++中的线程支持

在 C++11 之前，线程完全超出了 C++作为一种语言的范围。开发人员可以使用特定于平台的库，如 pthread 或 Win32 **应用程序编程接口**（**API**）。由于每个库都有自己的行为，将应用程序移植到另一个平台需要大量的开发和测试工作。

C++11 引入了线程作为 C++标准的一部分，并在其标准库中定义了一组类来创建多线程应用程序。

在这个示例中，我们将学习如何使用 C++在单个应用程序中生成多个并发线程。

# 如何做...

在这个示例中，我们将学习如何创建两个并发运行的工作线程。

1.  在您的`〜/test`工作目录中，创建一个名为`threads`的子目录。

1.  使用您喜欢的文本编辑器在`threads`子目录中创建一个名为`threads.cpp`的文件。将代码片段复制到`threads.cpp`文件中：

```cpp
#include <chrono>
#include <iostream>
#include <thread>

void worker(int index) {
  for (int i = 0; i < 10; i++) {
    std::cout << "Worker " << index << " begins" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "Worker " << index << " ends" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int main() {
  std::thread worker1(worker, 1);
  std::thread worker2(worker, 2);
  worker1.join();
  worker2.join();
  std::cout << "Done" << std::endl;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(threads)
add_executable(threads threads.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(threads pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 它是如何工作的...

在这个应用程序中，我们定义了一个名为`worker`的函数。为了保持代码简单，它并没有做太多有用的工作，只是打印`Worker X`开始和`Worker X`结束 10 次，消息之间有 50 毫秒的延迟。

在`main`函数中，我们创建了两个工作线程，`worker1`和`worker2`：

```cpp
 std::thread worker1(worker, 1);
 std::thread worker2(worker, 2);
```

我们向线程构造函数传递了两个参数：

+   在线程中运行的函数。

+   函数的参数。由于我们将先前定义的`worker`函数作为线程函数传递，参数应该与其类型匹配——在我们的例子中，它是`int`。

这样，我们定义了两个工作线程，它们执行相同的工作，但具有不同的索引——`1`和`2`。

线程一旦创建就立即开始运行；不需要调用任何额外的方法来启动它们。它们完全并行执行，正如我们从程序输出中看到的那样：

![](img/5772496b-0c8a-4c02-96c1-b9494da7fe2c.png)

我们的工作线程的输出是混合的，有时会混乱，比如`Worker Worker 1 ends2 ends`。这是因为终端的输出也是并行工作的。

由于工作线程是独立执行的，主线程在创建工作线程后没有任何事情可做。但是，如果主线程的执行达到`main`函数的末尾，程序将终止。为了避免这种情况，我们为每个工作线程添加了`join`方法的调用。这种方法会阻塞，直到线程终止。这样，我们只有在两个工作线程完成工作后才退出主程序。

# 探索数据同步

数据同步是处理多个执行线程的任何应用程序的重要方面。不同的线程经常需要访问相同的变量或内存区域。两个或更多独立线程同时写入同一内存可能导致数据损坏。即使在另一个线程更新变量时同时读取该变量也是危险的，因为在读取时它可能只被部分更新。

为了避免这些问题，并发线程可以使用所谓的同步原语，这是使对共享内存的访问变得确定和可预测的 API。

与线程支持的情况类似，C++语言在 C++11 标准之前没有提供任何同步原语。从 C++11 开始，一些同步原语被添加到 C++标准库中作为标准的一部分。

在这个配方中，我们将学习如何使用互斥锁和锁保护来同步对变量的访问。

# 如何做...

在前面的配方中，我们学习了如何完全并发地运行两个工作线程，并注意到这可能导致终端输出混乱。我们将修改前面配方中的代码，添加同步，使用互斥锁和锁保护，并查看区别。

1.  在您的`~/test`工作目录中，创建一个名为`mutex`的子目录。

1.  使用您喜欢的文本编辑器在`mutex`子目录中创建一个`mutex.cpp`文件。将代码片段复制到`mutex.cpp`文件中：

```cpp
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex m;

void worker(int index) {
  for (int i = 0; i < 10; i++) {
    {
 std::lock_guard<std::mutex> g(m);
 std::cout << "Worker " << index << " begins" << std::endl;
 std::this_thread::sleep_for(std::chrono::milliseconds(50));
 std::cout << "Worker " << index << " ends" << std::endl;
 }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int main() {
  std::thread worker1(worker, 1);
  std::thread worker2(worker, 2);
  worker1.join();
  worker2.join();
  std::cout << "Done" << std::endl;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(mutex)
add_executable(mutex mutex.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(mutex pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 工作原理...

构建并运行应用程序后，我们可以看到其输出与线程应用程序的输出类似。但也有明显的区别：

![](img/36c850c0-19b0-49b4-a851-d1878279476c.png)

首先，输出不会混乱。其次，我们可以看到一个清晰的顺序——没有一个工作线程被另一个工作线程中断，每个开始都后跟相应的结束。区别在于源代码的突出部分。我们创建一个全局的`mutex m`：

```cpp
std::mutex m;
```

然后，我们使用`lock_guard`来保护我们的关键代码部分，从打印`Worker X begins`的行开始，到打印`Worker X ends`的行结束。

`lock_guard`是互斥锁的包装器，它使用**RAII**（**资源获取即初始化**的缩写）技术，在构造函数中自动锁定相应的互斥锁，当定义锁对象时，它在析构函数中解锁，在其作用域结束时。这就是为什么我们添加额外的花括号来定义我们关键部分的作用域：

```cpp
    {
      std::lock_guard<std::mutex> g(m);
      std::cout << "Worker " << index << " begins" << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::cout << "Worker " << index << " ends" << std::endl;
    }
```

虽然可以通过调用其 lock 和 unlock 方法显式锁定和解锁互斥锁，但不建议这样做。忘记解锁已锁定的互斥锁会导致难以检测和难以调试的多线程同步问题。RAII 方法会自动解锁互斥锁，使代码更安全、更易读和更易理解。

# 还有更多...

正确实现线程同步需要非常注意细节和彻底分析。多线程应用程序中一个非常常见的问题是死锁。这是一种情况，其中一个线程被阻塞，因为它正在等待另一个线程，而另一个线程又被阻塞，因为它正在等待第一个线程。因此，两个线程被无限期地阻塞。

如果需要两个或更多个互斥锁进行同步，则会发生死锁。C++17 引入了*std::scoped_lock*，可在[`en.cppreference.com/w/cpp/thread/scoped_lock`](https://en.cppreference.com/w/cpp/thread/scoped_lock)上找到，这是一个多个互斥锁的 RAII 包装器，有助于避免死锁。

# 使用条件变量

我们学会了如何同步两个或多个线程对同一变量的同时访问。线程访问变量的特定顺序并不重要；我们只是防止了对变量的同时读写。

一个线程等待另一个线程开始处理数据是一个常见的情况。在这种情况下，当数据可用时，第二个线程应该由第一个线程通知。这可以使用条件变量来完成，C++从 C++11 标准开始支持。

在这个配方中，我们将学习如何使用条件变量在数据可用时立即激活数据处理的单独线程。

# 如何做...

我们将实现一个具有两个工作线程的应用程序，类似于我们在*探索数据同步*配方中创建的应用程序。

1.  在您的`~/test`工作目录中，创建一个名为`condvar`的子目录。

1.  使用您喜欢的文本编辑器在`condvar`子目录中创建一个名为`condv.cpp`的文件。

1.  现在，在`condvar.cpp`中放置所需的头文件并定义全局变量：

```cpp
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex m;
std::condition_variable cv;
std::vector<int> result;
int next = 0;
```

1.  在定义全局变量之后，我们添加了我们的`worker`函数，它与前面的配方中的`worker`函数类似：

```cpp
void worker(int index) {
  for (int i = 0; i < 10; i++) {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [=]{return next == index; });
    std::cout << "worker " << index << "\n";
    result.push_back(index);
    next = next + 1;
    if (next > 2) { next = 1; };
    cv.notify_all();
  }
}
```

1.  最后，我们定义我们的入口点——`main`函数：

```cpp
int main() {
  std::thread worker1(worker, 1);
  std::thread worker2(worker, 2);
  {
    std::lock_guard<std::mutex> l(m);
    next = 1;
  }
  std::cout << "Start\n";
  cv.notify_all();
  worker1.join();
  worker2.join();
  for (int e : result) {
    std::cout << e << ' ';
  }
  std::cout << std::endl;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
cmake_minimum_required(VERSION 3.5.1)
project(condvar)
add_executable(condvar condvar.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(condvar pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 工作原理...

与我们在*探索数据同步*配方中创建的应用程序类似，我们创建了两个工作线程`worker1`和`worker2`，它们使用相同的`worker`函数线程，只是`index`参数不同。

除了向控制台打印消息外，工作线程还更新了一个全局向量 result。每个工作线程只是在其循环中将其索引添加到`result`变量中，如下命令所示：

```cpp
std::vector<int> result;
```

我们希望每个工作线程只在轮到它时将其索引添加到结果中——`worker 1`，然后`worker 2`，然后再次`worker 1`，依此类推。没有同步是不可能做到这一点的；然而，简单的互斥同步是不够的。它可以保证两个并发线程不会同时访问代码的同一关键部分，但不能保证顺序。可能是`worker 1`在`worker 2`锁定之前再次锁定互斥锁。

为了解决排序问题，我们定义了一个`cv`条件变量和一个`next`整数变量：

```cpp
std::condition_variable cv;
int next = 0;
```

`next`变量包含一个工作线程的索引。它初始化为`0`，并在`main`函数中设置为特定的工作线程索引。由于这个变量被多个线程访问，我们在锁保护下进行操作：

```cpp
  {
    std::lock_guard<std::mutex> l(m);
    next = 1;
  }
```

尽管工作线程在创建后开始执行，但它们两者立即被条件变量阻塞，等待`next`变量的值与它们的索引匹配。条件变量需要`std::unique_lock`进行等待。我们在调用`wait`方法之前创建它：

```cpp
std::unique_lock<std::mutex> l(m);
cv.wait(l, [=]{return next == index; });
```

虽然条件变量`cv`在`main`函数中设置为`1`，但这还不够。我们需要显式通知等待条件变量的线程。我们使用`notify_all`方法来做到这一点：

```cpp
cv.notify_all();
```

这将唤醒所有等待的线程，它们将自己的索引与`next`变量进行比较。匹配的线程解除阻塞，而所有其他线程再次进入睡眠状态。

活动线程向控制台写入消息并更新`result`变量。然后，它更新`next`变量以选择下一个要激活的线程。我们递增索引直到达到最大值，然后将其重置为`1`：

```cpp
next = next + 1;
if (next > 2) { next = 1; };
```

与`main`函数中的代码情况类似，在决定`next`线程的索引后，我们需要调用`notify_all`来唤醒所有线程，并让它们决定轮到谁工作：

```cpp
cv.notify_all();
```

在工作线程工作时，`main`函数等待它们的完成：

```cpp
 worker1.join();
 worker2.join();
```

当所有工作线程完成时，将打印`result`变量的值：

```cpp
  for (int e : result) {
    std::cout << e << ' ';
  }
```

构建并运行程序后，我们得到以下输出：

![](img/e9547f19-9f61-4307-bed0-e7fa66406e5a.png)

正如我们所看到的，所有线程都按预期顺序激活了。

# 还有更多...

在这个示例中，我们只使用了条件变量对象提供的一些方法。除了简单的`wait`函数外，还有一些等待特定时间或等待直到达到指定时间点的函数。在[`en.cppreference.com/w/cpp/thread/condition_variable`](https://en.cppreference.com/w/cpp/thread/condition_variable)上了解更多关于*C++条件变量类*的信息。

# 使用原子变量

原子变量之所以被命名为原子变量，是因为它们不能被部分读取或写入。例如，比较`Point`和`int`数据类型：

```cpp
struct Point {
  int x, y;
};

Point p{0, 0};
int b = 0;

p = {10, 10};
b = 10;
```

在这个例子中，修改`p`变量相当于两次赋值：

```cpp
p.x = 10;
p.y = 10;
```

这意味着任何并发线程读取`p`变量时可能会得到部分修改的数据，比如`x=10`，`y=0`，这可能导致难以检测和难以重现的错误计算。这就是为什么对这种数据类型的访问应该是同步的。

那么`b`变量呢？它能被部分修改吗？答案是：取决于平台。然而，C++提供了一组数据类型和模板，以确保变量作为一个整体原子地一次性改变。

在这个示例中，我们将学习如何使用原子变量来同步多个线程。由于原子变量不能被部分修改，因此不需要使用互斥锁或其他昂贵的同步原语。

# 如何做...

我们将创建一个应用程序，生成两个工作线程来并发更新一个数据数组。我们将使用原子变量而不是互斥锁，以确保并发更新是安全的。

1.  在你的`~/test`工作目录中，创建一个名为`atomic`的子目录。

1.  使用你喜欢的文本编辑器在`atomic`子目录中创建一个名为`atomic.cpp`的文件。

1.  现在，我们放置所需的头文件，并在`atomic.cpp`中定义全局变量：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<size_t> shared_index{0};
std::vector<int> data;
```

1.  在定义全局变量之后，我们添加我们的`worker`函数。它类似于之前示例中的`worker`函数，但除了一个`index`之外，它还有一个额外的参数`timeout`：

```cpp
void worker(int index, int timeout) {
  while(true) {
  size_t worker_index = shared_index.fetch_add(1);
  if (worker_index >= data.size()) {
      break;
  }
  std::cout << "Worker " << index << " handles "
              << worker_index << std::endl;
  data[worker_index] = data[worker_index] * 2;
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
  }
  }
```

1.  最后，我们定义我们的入口点——`main`函数：

```cpp
int main() {
  for (int i = 0; i < 10; i++) {
    data.emplace_back(i);
  }
  std::thread worker1(worker, 1, 50);
  std::thread worker2(worker, 2, 20);
  worker1.join();
  worker2.join();
  std::cout << "Result: ";
  for (auto& v : data) {
    std::cout << v << ' ';
  }
  std::cout << std::endl;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(atomic)
add_executable(atomic atomic.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(atomic pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

你可以构建并运行应用程序。

# 工作原理...

我们正在创建一个应用程序，使用多个工作线程更新数组的所有元素。对于昂贵的更新操作，这种方法可以在多核平台上实现显著的性能提升。

困难在于在多个工作线程之间共享工作，因为它们每个可能需要不同的时间来处理数据元素。

我们使用一个`shared_index`原子变量来存储尚未被任何工作线程声明的下一个元素的索引。这个变量，以及要处理的数组，被声明为全局变量：

```cpp
std::atomic<size_t> shared_index{0};
std::vector<int> data;
```

我们的`worker`函数类似于之前的示例中的`worker`函数，但有重要的区别。首先，它有一个额外的参数`timeout`。这用于模拟处理每个元素所需的时间差异。

其次，我们的工作线程不是在固定次数的迭代中运行，而是在一个循环中运行，直到`shared_index`变量达到最大值。这表示所有元素都已被处理，工作线程可以终止。

在每次迭代中，一个工作线程读取`shared_index`的值。如果有要处理的元素，它将`shared_index`变量的值存储在一个本地的`worker_index`变量中，并同时增加`shared_index`变量。

虽然可以像使用常规变量一样使用原子变量——首先获取其当前值，然后增加变量——但这可能导致竞争条件。两个工作线程几乎同时读取变量。在这种情况下，它们都获得相同的值，然后开始处理相同的元素，相互干扰。这就是为什么我们使用特殊的`fetch_add`方法，它增加变量并返回增加之前的值作为单个、不可中断的操作：

```cpp
size_t worker_index = shared_index.fetch_add(1);
```

如果`worker_index`变量达到数组的大小，这意味着所有元素都已经处理完毕，工作线程可以终止：

```cpp
if (worker_index >= data.size()) {
      break;
}
```

如果`worker_index`变量有效，则工作线程将使用它来更新数组元素的值。在我们的情况下，我们只是将它乘以`2`：

```cpp
data[worker_index] = data[worker_index] * 2;
```

为了模拟昂贵的数据操作，我们使用自定义延迟。延迟的持续时间由`timeout`参数确定：

```cpp
std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
```

在`main`函数中，我们向数据向量中添加要处理的元素。我们使用循环将向量填充为从零到九的数字：

```cpp
for (int i = 0; i < 10; i++) {
    data.emplace_back(i);
}
```

初始数据集准备好后，我们创建两个工作线程，提供`index`和`timeout`参数。使用工作线程的不同超时来模拟不同的性能：

```cpp
 std::thread worker1(worker, 1, 50);
 std::thread worker2(worker, 2, 20);
```

然后，我们等待两个工作线程完成它们的工作，并将结果打印到控制台。当我们构建和运行我们的应用程序时，我们会得到以下输出：

![](img/c34579d8-b62e-4c9e-bcbe-a4441c2d5e89.png)

正如我们所看到的，`Worker 2`处理的元素比`Worker 1`多，因为它的超时是 20 毫秒，而`Worker 1`是 50 毫秒。此外，所有元素都按预期进行处理，没有遗漏和重复。

# 还有更多...

我们学会了如何处理整数原子变量。虽然这种类型的原子变量是最常用的，但 C++也允许定义其他类型的原子变量，包括非整数类型，只要它们是平凡可复制的、可复制构造的和可复制赋值的。

除了我们在示例中使用的`fetch_add`方法，原子变量还有其他类似的方法，可以帮助开发人员在单个操作中查询值和修改变量。考虑使用这些方法来避免竞争条件或使用互斥锁进行昂贵的同步。

在 C++20 中，原子变量获得了`wait`、`notify_all`和`notify_one`方法，类似于条件变量的方法。它们允许使用更高效、轻量级的原子变量来实现以前需要条件变量的逻辑。

有关原子变量的更多信息，请访问[`en.cppreference.com/w/cpp/atomic/atomic`](https://en.cppreference.com/w/cpp/atomic/atomic)。

# 使用 C++内存模型

从 C++11 标准开始，C++定义了线程和同步的 API 和原语作为语言的一部分。在具有多个处理器核心的系统中进行内存同步是复杂的，因为现代处理器可以通过重新排序指令来优化代码执行。即使使用原子变量，也不能保证数据按预期顺序修改或访问，因为编译器可以改变顺序。

为了避免歧义，C++11 引入了内存模型，定义了对内存区域的并发访问行为。作为内存模型的一部分，C++定义了`std::memory_order`枚举，它向编译器提供有关预期访问模型的提示。这有助于编译器以不干扰预期代码行为的方式优化代码。

在这个示例中，我们将学习如何使用最简单的`std::memory_order`枚举来实现一个共享计数器变量。

# 如何做...

我们正在实现一个应用程序，其中有一个共享计数器，由两个并发的工作线程递增。

1.  在您的`~/test`工作目录中，创建一个名为`memorder`的子目录。

1.  使用您喜欢的文本编辑器在`atomic`子目录中创建一个`memorder.cpp`文件。

1.  现在，我们在`memorder.cpp`中放置所需的头文件并定义全局变量：

```cpp
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<bool> running{true};
std::atomic<int> counter{0};
```

1.  全局变量定义后，我们添加我们的`worker`函数。该函数只是递增一个计数器，然后休眠一段特定的时间间隔：

```cpp
void worker() {
 while(running) {
 counter.fetch_add(1, std::memory_order_relaxed);
 }
 }
```

1.  然后，我们定义我们的`main`函数：

```cpp
int main() {
  std::thread worker1(worker);
  std::thread worker2(worker);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  running = false;
  worker1.join();
  worker2.join();
  std::cout << "Counter: " << counter << std::endl;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(memorder)
add_executable(memorder memorder.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(memorder pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建和运行应用程序。

# 工作原理...

在我们的应用程序中，我们将创建两个工作线程，它们将递增一个共享计数器，并让它们运行一段特定的时间。

首先，我们定义两个全局原子变量`running`和`counter`：

```cpp
std::atomic<bool> running{true};
std::atomic<int> counter{0};
```

`running`变量是一个二进制标志。当它设置为`true`时，工作线程应该继续运行。在它变为`false`后，工作线程应该终止。

`counter`变量是我们的共享计数器。工作线程将同时递增它。我们使用了在*使用原子变量*示例中已经使用过的`fetch_add`方法。它用于原子地递增一个变量。在这个示例中，我们将额外的参数`std::memory_order_relaxed`传递给这个方法：

```cpp
counter.fetch_add(1, std::memory_order_relaxed);
```

这个参数是一个提示。虽然原子性和修改的一致性对于计数器的实现很重要并且应该得到保证，但并发内存访问之间的顺序并不那么重要。`std::memory_order_relaxed`为原子变量定义了这种内存访问。将其传递给`fetch_add`方法允许我们为特定目标平台进行微调，以避免不必要的同步延迟，从而影响性能。

在`main`函数中，我们创建两个工作线程：

```cpp
std::thread worker1(worker);
std::thread worker2(worker);
```

然后，主线程暂停 1 秒。暂停后，主线程将`running`变量的值设置为`false`，表示工作线程应该终止。

```cpp
running = false;
```

工作线程终止后，我们打印计数器的值：

![](img/a33e028f-a5f8-4fe4-b857-23fd84788a3a.png)

生成的计数器值由传递给`worker`函数的超时间隔确定。在我们的示例中，更改`fetch_add`方法中的内存顺序类型不会导致结果值的明显变化。但是，它可以提高使用原子变量的高并发应用程序的性能，因为编译器可以重新排序并发线程中的操作而不会破坏应用程序逻辑。这种优化高度依赖于开发人员的意图，并且不能在没有开发人员提示的情况下自动推断。

# 还有更多...

C++内存模型和内存排序类型是复杂的主题，需要深入了解现代 CPU 如何访问内存并优化其代码执行。*C++内存模型参考*，[`en.cppreference.com/w/cpp/language/memory_model`](https://en.cppreference.com/w/cpp/language/memory_model)提供了大量信息，是学习多线程应用程序优化的高级技术的良好起点。

# 探索无锁同步

在前面的示例中，我们学习了如何使用互斥锁和锁同步多个线程对共享数据的访问。如果多个线程尝试运行由锁保护的代码的关键部分，只有一个线程可以一次执行。所有其他线程都必须等待，直到该线程离开关键部分。

然而，在某些情况下，可以在没有互斥锁和显式锁的情况下同步对共享数据的访问。其思想是使用数据的本地副本进行修改，然后在单个、不可中断和不可分割的操作中更新共享副本。

这种类型的同步取决于硬件。目标处理器应该提供某种形式的**比较和交换**（**CAS**）指令。这检查内存位置中的值是否与给定值匹配，并且仅当它们匹配时才用新给定值替换它。由于它是单处理器指令，它不会被上下文切换中断。这使它成为更复杂的原子操作的基本构建块。

在本教程中，我们将学习如何检查原子变量是否是无锁的，或者是使用互斥体或其他锁定操作实现的。我们还将根据 C++11 中的原子比较交换函数的示例实现一个无锁推送操作，该示例可在[`en.cppreference.com/w/cpp/atomic/atomic_compare_exchange`](https://en.cppreference.com/w/cpp/atomic/atomic_compare_exchange)上找到。

# 如何做...

我们正在实现一个简单的`Stack`类，它提供了一个构造函数和一个名为`Push`的函数。

1.  在您的`~/test`工作目录中，创建一个名为`lockfree`的子目录。

1.  使用您喜欢的文本编辑器在`lockfree`子目录中创建一个名为`lockfree.cpp`的文件。

1.  现在，我们放入所需的头文件，并在`lockfree.cpp`文件中定义一个`Node`辅助数据类型：

```cpp
#include <atomic>
#include <iostream>

struct Node {
  int data;
  Node* next;
};
```

1.  接下来，我们定义一个简单的`Stack`类。这使用`Node`数据类型来组织数据存储：

```cpp
class Stack {
  std::atomic<Node*> head;

  public:
    Stack() {
    std::cout << "Stack is " <<
    (head.is_lock_free() ? "" : "not ")
    << "lock-free" << std::endl;
    }

   void Push(int data) {
      Node* new_node = new Node{data, nullptr};
      new_node->next = head.load();
      while(!std::atomic_compare_exchange_weak(
                &head,
                &new_node->next,
                new_node));
    }
    };
```

1.  最后，我们定义一个简单的`main`函数，创建一个`Stack`实例并将一个元素推入其中：

```cpp
int main() {
  Stack s;
  s.Push(1);
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(lockfree)
add_executable(lockfree lockfree.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(lockfree pthread)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 工作原理...

我们创建了一个简单的应用程序，实现了一个整数值的简单堆栈。我们将堆栈的元素存储在动态内存中，对于每个元素，我们应该能够确定其后面的元素。

为此，我们定义了一个`Node`辅助结构，它有两个数据字段。`data`字段存储元素的实际值，而`next`字段是堆栈中下一个元素的指针：

```cpp
int data;
Node* next;
```

然后，我们定义`Stack`类。通常，堆栈意味着两个操作：

+   `Push`：将一个元素放在堆栈顶部

+   `Pull`：从堆栈顶部获取一个元素

为了跟踪堆栈的顶部，我们创建一个`top`变量，它保存指向`Node`对象的指针。它将是我们堆栈的顶部：

```cpp
std::atomic<Node*> head;
```

我们还定义了一个简单的构造函数，它初始化了我们的`top`变量的值，并检查它是否是无锁的。在 C++中，原子变量可以使用原子**一致性、可用性和分区容错性**（**CAP**）操作或使用常规互斥体来实现。这取决于目标 CPU：

```cpp
(head.is_lock_free() ? "" : "not ")
```

在我们的应用程序中，我们只实现了`Push`方法，以演示如何以无锁的方式实现它。

`Push`方法接受要放在堆栈顶部的值。为此，我们创建一个新的`Node`对象的实例：

```cpp
 Node* new_node = new Node{data, nullptr};
```

由于我们将元素放在堆栈的顶部，新创建的实例的指针应该分配给`top`变量，并且应该将`top`变量的旧值分配给我们的新`Node`对象的`next`指针。

然而，直接这样做是不安全的。两个或更多线程可以同时修改`top`变量，导致数据损坏。我们需要某种数据同步。我们可以使用锁和互斥体来做到这一点，但也可以以无锁的方式来实现。

这就是为什么我们最初只更新下一个指针。由于我们的新`Node`对象还不是堆栈的一部分，所以我们可以在没有同步的情况下执行，因为其他线程无法访问它：

```cpp
new_node->next = head.load();
```

现在，我们需要将其添加为堆栈的新`top`变量。我们使用`std::atomic_compare_exchange_weak`函数进行循环：

```cpp
      while(!std::atomic_compare_exchange_weak(
                &head,
                &new_node->next,
                new_node));
```

此函数将`top`变量的值与新元素的`next`指针中存储的值进行比较。如果它们匹配，则将`top`变量的值替换为新节点的指针并返回`true`。否则，它将`top`变量的值写入新元素的`next`指针并返回`false`。由于我们在下一步中更新了`next`指针以匹配`top`变量，这只能发生在另一个线程在调用`std::atomic_compare_exchange_weak`函数之前修改了它。最终，该函数将返回`true`，表示`top`头部已更新为指向我们的元素的指针。

`main`函数创建一个堆栈的实例，并将一个元素推入其中。在输出中，我们可以看到底层实现是否是无锁的：

![](img/1c7151c3-b9d3-44d2-afb2-8a5caa5119f2.png)

对于我们的目标，实现是无锁的。

# 还有更多...

无锁同步是一个非常复杂的话题。开发无锁数据结构和算法需要大量的工作。即使是使用无锁操作实现简单的`Push`逻辑也不容易理解。对于代码的适当分析和调试需要更大的努力。通常，这可能导致难以注意和难以实现的微妙问题。

尽管无锁算法的实现可以提高应用程序的性能，但考虑使用现有的无锁数据结构库之一，而不是编写自己的库。例如，[Boost.Lockfree](https://www.boost.org/doc/libs/1_66_0/doc/html/lockfree.html)提供了一系列无锁数据类型供您使用。

# 在共享内存中使用原子变量

我们学会了如何使用原子变量来同步多线程应用程序中的两个或多个线程。但是，原子变量也可以用于同步作为独立进程运行的独立应用程序。

我们已经知道如何在两个应用程序之间交换数据使用共享内存。现在，我们可以结合这两种技术——共享内存和原子变量——来实现两个独立应用程序的数据交换和同步。

# 如何做...

在这个示例中，我们将修改我们在第六章中创建的应用程序，*内存管理*，用于在两个处理器之间使用共享内存区域交换数据。

1.  在您的`~/test`工作目录中，创建一个名为`shmatomic`的子目录。

1.  使用您喜欢的文本编辑器在`shmatomic`子目录中创建一个名为`shmatomic.cpp`的文件。

1.  我们重用了我们在`shmem`应用程序中创建的共享内存数据结构。将公共头文件和常量放入`shmatomic.cpp`文件中：

```cpp
#include <atomic>
#include <iostream>
#include <chrono>
#include <thread>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

const char* kSharedMemPath = "/sample_point";
```

1.  接下来，开始定义模板化的`SharedMem`类：

```cpp
template<class T>
class SharedMem {
  int fd;
  T* ptr;
  const char* name;

  public:
```

1.  该类将有一个构造函数，一个析构函数和一个 getter 方法。让我们添加构造函数：

```cpp
    SharedMem(const char* name, bool owner=false) {
      fd = shm_open(name, O_RDWR | O_CREAT, 0600);
      if (fd == -1) {
        throw std::runtime_error("Failed to open a shared
        memory region");
      }
      if (ftruncate(fd, sizeof(T)) < 0) {
        close(fd);
        throw std::runtime_error("Failed to set size of a shared
        memory region");
      };
      ptr = (T*)mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, 
      MAP_SHARED, fd, 0);
      if (!ptr) {
        close(fd);
        throw std::runtime_error("Failed to mmap a shared memory
        region");
      }
      this->name = owner ? name : nullptr;
      }
```

1.  接下来是简单的析构函数和 getter：

```cpp
~SharedMem() {
munmap(ptr, sizeof(T));
close(fd);
if (name) {
std::cout << "Remove shared mem instance " << name << std::endl;
shm_unlink(name);
}
}

T& get() const {
return *ptr;
}
};
```

1.  现在，我们定义要用于数据交换和同步的数据类型：

```cpp
struct Payload {
std::atomic_bool data_ready;
std::atomic_bool data_processed;
int index;
};
```

1.  接下来，我们定义一个将生成数据的函数：

```cpp
void producer() {
  SharedMem<Payload> writer(kSharedMemPath);
  Payload& pw = writer.get();
if (!pw.data_ready.is_lock_free()) {
throw std::runtime_error("Flag is not lock-free");
  }
for (int i = 0; i < 10; i++) {
pw.data_processed.store(false);
pw.index = i;
    pw.data_ready.store(true);
while(!pw.data_processed.load());
}
}
```

1.  接下来是消耗数据的函数：

```cpp
void consumer() {
SharedMem<Payload> point_reader(kSharedMemPath, true);
Payload& pr = point_reader.get();
if (!pr.data_ready.is_lock_free()) {
throw std::runtime_error("Flag is not lock-free");
}
for (int i = 0; i < 10; i++) {
 while(!pr.data_ready.load());
    pr.data_ready.store(false);
std::cout << "Processing data chunk " << pr.index << std::endl;
    pr.data_processed.store(true);
}
}
```

1.  最后，我们添加我们的`main`函数，将所有内容联系在一起：

```cpp
int main() {

if (fork()) {
    consumer();
} else {
    producer();
}
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(shmatomic)
add_executable(shmatomic shmatomic.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
target_link_libraries(shmatomic pthread rt)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 工作原理...

在我们的应用程序中，我们重用了我们在第六章中介绍的模板化的`SharedMem`类，*内存管理*。该类用于在共享内存区域中存储特定类型的元素。让我们快速回顾一下它的工作原理。

`SharedMem`类是**可移植操作系统接口**（**POSIX**）共享内存 API 的包装器。它定义了三个私有数据字段来保存特定于系统的处理程序和指针，并公开由两个函数组成的公共接口：

+   一个接受共享区域名称和所有权标志的构造函数

+   一个`get`方法，返回存储在共享内存中的对象的引用

该类还定义了一个析构函数，执行所有必要的操作以正确关闭共享对象。因此，`SharedMem`类可以用于使用 C++ RAII 习语进行安全资源管理。

`SharedMem`类是一个模板类。它由我们想要存储在共享内存中的数据类型参数化。为此，我们定义了一个名为`Payload`的结构：

```cpp
struct Payload {
  std::atomic_bool data_ready;
  std::atomic_bool data_processed;
  int index;
};
```

它有一个`index`整数变量，我们将使用它作为数据交换字段，并且有两个原子布尔标志，`data_ready`和`data_processed`，用于数据同步。

我们还定义了两个函数，`producer`和`consumer`，它们将在单独的进程中工作，并使用共享内存区域相互交换数据。

`producer`函数正在生成数据块。首先，它创建了`SharedMem`类的一个实例，由`Payload`数据类型参数化。它将共享内存区域的路径传递给`SharedMem`构造函数：

```cpp
SharedMem<Payload> writer(kSharedMemPath);
```

创建共享内存实例后，它获取对存储在其中的有效负载数据的引用，并检查我们在`Payload`数据类型中定义的任何原子标志是否是无锁定的：

```cpp
if (!pw.data_ready.is_lock_free()) {
    throw std::runtime_error("Flag is not lock-free");
}
```

该函数在循环中生成 10 个数据块。数据块的索引被放入有效负载的`index`字段中：

```cpp
pw.index = i;
```

但是，除了将数据放入共享内存中，我们还需要同步对这些数据的访问。这就是我们使用原子标志的时候。

对于每次迭代，在更新`index`字段之前，我们重置`data_processed`标志。更新索引后，我们设置`data ready`标志，这是向消费者指示新的数据块已准备就绪的指示器，并等待数据被消费者处理。我们循环直到`data_processed`标志变为`true`，然后进入下一个迭代：

```cpp
pw.data_ready.store(true);
while(!pw.data_processed.load());
```

`consumer`函数的工作方式类似。由于它在一个单独的进程中工作，它通过使用相同的路径创建`SharedMem`类的实例来打开相同的共享内存区域。我们还使`consumer`函数成为共享内存实例的所有者。这意味着它负责在`SharedMem`实例被销毁后删除共享内存区域：

```cpp
SharedMem<Payload> point_reader(kSharedMemPath, true);
```

与`producer`函数类似，`consumer`函数检查原子标志是否是无锁定的，并进入数据消耗的循环。

对于每次迭代，它在一个紧密的循环中等待直到数据准备就绪：

```cpp
while(!pr.data_ready.load());
```

在`producer`函数将`data_ready`标志设置为`true`后，`consumer`函数可以安全地读取和处理数据。在我们的实现中，它只将`index`字段打印到控制台。处理完数据后，`consumer`函数通过将`data_processed`标志设置为`true`来指示这一点：

```cpp
pr.data_processed.store(true);
```

这触发了`producer`函数端的数据生产的下一个迭代：

![](img/96155edc-9e5c-42dc-b8d6-46969182a299.png)

结果，我们可以看到处理的数据块的确定性输出，没有遗漏或重复；这在数据访问不同步的情况下很常见。

# 探索异步函数和期货

在多线程应用程序中处理数据同步是困难的，容易出错，并且需要开发人员编写大量代码来正确对齐数据交换和数据通知。为了简化开发，C++11 引入了一种标准 API，以一种类似于常规同步函数调用的方式编写异步代码，并在底层隐藏了许多同步复杂性。

在这个示例中，我们将学习如何使用异步函数调用和期货在多个线程中运行我们的代码，几乎不需要额外的工作来进行数据同步。

# 如何做到这一点...

我们将实现一个简单的应用程序，调用一个长时间运行的函数，并等待其结果。在函数运行时，应用程序可以继续进行其他计算。

1.  在您的`~/test`工作目录中，创建一个名为`async`的子目录。

1.  使用您喜欢的文本编辑器在`async`子目录中创建一个名为`async.cpp`的文件。

1.  将我们的应用程序代码放入`async.cpp`文件中，从公共头文件和我们的长时间运行的函数开始：

```cpp
#include <chrono>
#include <future>
#include <iostream>

int calculate (int x) {
  auto start = std::chrono::system_clock::now();
  std::cout << "Start calculation\n";
  std::this_thread::sleep_for(std::chrono::seconds(1));
  auto delta = std::chrono::system_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
  std::cout << "Done in " << ms.count() << " ms\n";
  return x*x;
}
```

1.  接下来，添加`test`函数，调用长时间运行的函数：

```cpp
void test(int value, int worktime) {
  std::cout << "Request result of calculations for " << value << std::endl;
  std::future<int> fut = std::async (calculate, value);
  std::cout << "Keep working for " << worktime << " ms" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(worktime));
  auto start = std::chrono::system_clock::now();
  std::cout << "Waiting for result" << std::endl;
  int result = fut.get();
  auto delta = std::chrono::system_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);

  std::cout << "Result is " << result
            << ", waited for " << ms.count() << " ms"
            << std::endl << std::endl;
}

```

1.  最后，添加一个最简单的`main`函数：

```cpp
int main ()
{
  test(5, 400);
  test(8, 1200);
  return 0;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(async)
add_executable(async async.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")
target_link_libraries(async pthread -static-libstdc++)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

您可以构建并运行应用程序。

# 工作原理...

在我们的应用程序中，我们定义了一个`calculate`函数，应该需要很长时间才能运行。从技术上讲，我们的函数计算整数参数的平方，但我们添加了人为的延迟，使其运行 1 秒钟。我们使用`sleep_for`标准库函数来为应用程序添加延迟：

```cpp
std::this_thread::sleep_for(std::chrono::seconds(1));
```

除了计算，该函数还在控制台记录了开始工作时的时间，完成时的时间以及花费的时间。

接下来，我们定义了一个`test`函数，调用`calculate`函数，以演示异步调用的工作原理。

该函数有两个参数。第一个参数是传递给`calculate`函数的值。第二个参数是在运行`calculate`函数后并在请求结果之前，`test`函数将花费的时间。这样，我们模拟了函数可以在并行计算中执行的有用工作。

`test`函数通过异步模式运行`calculate`函数，并传递第一个参数`value`：

```cpp
std::future<int> fut = std::async (calculate, value);
```

`async`函数隐式地生成一个线程，并开始执行`calculate`函数。

由于我们异步运行函数，结果还没有准备好。相反，`async`函数返回一个`std::future`的实例，一个在结果可用时将保存结果的对象。

接下来，我们模拟有用的工作。在我们的情况下，这是指定时间间隔的暂停。在可以并行完成的工作完成后，我们需要获取`calculate`函数的结果才能继续。为了请求结果，我们使用`std::future`对象的`get`方法，如下所示：

```cpp
int result = fut.get();
```

`get`方法会阻塞，直到结果可用。然后，我们可以计算等待结果的时间，并将结果以及等待时间输出到控制台。

在`main`函数中，我们运行`test`函数来评估两种情况：

+   有用的工作所花费的时间比计算结果的时间更短。

+   有用的工作所花费的时间比计算结果的时间更长。

运行应用程序会产生以下输出。

在第一种情况下，我们可以看到我们开始计算，然后在计算完成之前开始等待结果。结果，`get`方法阻塞了 600 毫秒，直到结果准备就绪：

![](img/46a609c8-dcd7-4286-b46c-dc2341addc93.png)

在第二种情况下，有用的工作花费了`1200`毫秒。正如我们所看到的，计算在结果被请求之前就已经完成了，因此`get`方法没有阻塞，立即返回了结果。

# 还有更多...

期货和异步函数提供了一个强大的机制来编写并行和易懂的代码。异步函数是灵活的，支持不同的执行策略。Promise 是另一种机制，使开发人员能够克服异步编程的复杂性。更多信息可以在`std::future`的参考页面找到[[`en.cppreference.com/w/cpp/thread/future`](https://en.cppreference.com/w/cpp/thread/future)], `std::promise`的参考页面[[`en.cppreference.com/w/cpp/thread/promise`](https://en.cppreference.com/w/cpp/thread/promise)], 以及`std::async`的参考页面[[`en.cppreference.com/w/cpp/thread/async`](https://en.cppreference.com/w/cpp/thread/async)]。
