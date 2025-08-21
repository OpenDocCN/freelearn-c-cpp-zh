# 第十一章：使用 Boost Asio 进行网络编程

在今天的网络世界中，处理每秒数千个请求的互联网服务器有一个艰巨的任务要完成——保持响应性，并且即使请求量增加也不会减慢。构建可靠的进程，有效地处理网络 I/O 并随着连接数量的增加而扩展，是具有挑战性的，因为它通常需要应用程序员理解底层协议栈并以巧妙的方式利用它。增加挑战的是跨平台的网络编程接口和模型的差异，以及使用低级 API 的固有困难。

Boost Asio（发音为 ay-see-oh）是一个可移植的库，用于使用一致的编程模型执行高效的网络 I/O。重点是执行异步 I/O（因此称为 Asio），其中程序启动 I/O 操作并继续执行其他任务，而不会阻塞等待操作系统返回操作结果。当底层操作系统完成操作时，Asio 库会通知程序并采取适当的操作。Asio 帮助解决的问题以及它使用的一致、可移植接口使其非常有用。但是，交互的异步性质也使其更加复杂和不那么直观。这就是为什么我们将分两部分学习 Asio 的原因：首先理解其交互模型，然后使用它执行网络 I/O：

+   使用 Asio 进行任务执行

+   使用 Asio 进行网络编程

Asio 提供了一个工具包，用于执行和管理任意任务，本章的第一部分重点是理解这个工具包。我们在本章的第二部分应用这种理解，当我们具体看一下 Asio 如何帮助编写使用互联网协议（IP）套件的程序与其他程序进行网络通信时。

# 使用 Asio 进行任务执行

在其核心，Boost Asio 提供了一个任务执行框架，您可以使用它来执行任何类型的操作。您将您的任务创建为函数对象，并将它们发布到 Boost Asio 维护的任务队列中。您可以注册一个或多个线程来选择这些任务（函数对象）并调用它们。线程不断地选择任务，直到任务队列为空，此时线程不会阻塞，而是退出。

## IO 服务、队列和处理程序

Asio 的核心是类型`boost::asio::io_service`。程序使用`io_service`接口执行网络 I/O 和管理任务。任何想要使用 Asio 库的程序都会创建至少一个`io_service`实例，有时甚至会创建多个。在本节中，我们将探索`io_service`的任务管理能力，并将网络 I/O 的讨论推迟到本章的后半部分。

以下是 IO 服务在使用强制性的“hello world”示例：

**清单 11.1：Asio Hello World**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4
 5 int main() {
 6   asio::io_service service;
 7
 8   service.post(
 9     [] {
10       std::cout << "Hello, world!" << '\n';
11     });
12
13   std::cout << "Greetings: \n";
14   service.run();
15 }
```

我们包括方便的头文件`boost/asio.hpp`，其中包括本章示例中需要的大部分 Asio 库（第 1 行）。Asio 库的所有部分都在命名空间`boost::asio`下，因此我们为此使用一个更短的别名（第 3 行）。程序本身只是在控制台上打印`Hello, world!`，但是通过一个任务来实现。

程序首先创建了一个`io_service`的实例（第 6 行），并使用`io_service`的`post`成员函数将一个函数对象*发布*到其中。在这种情况下，使用 lambda 表达式定义的函数对象被称为**处理程序**。对`post`的调用将处理程序添加到`io_service`内部的队列中；一些线程（包括发布处理程序的线程）必须*分派*它们，即，将它们从队列中移除并调用它们。对`io_service`的`run`成员函数的调用（第 14 行）正是这样做的。它循环遍历`io_service`内部的处理程序，移除并调用每个处理程序。实际上，我们可以在调用`run`之前向`io_service`发布更多的处理程序，并且它会调用所有发布的处理程序。如果我们没有调用`run`，则不会分派任何处理程序。`run`函数会阻塞，直到队列中的所有处理程序都被分派，并且只有在队列为空时才会返回。单独地，处理程序可以被视为独立的、打包的任务，并且 Boost Asio 提供了一个很好的机制来分派任意任务作为处理程序。请注意，处理程序必须是无参数的函数对象，也就是说，它们不应该带有参数。

### 注意

默认情况下，Asio 是一个仅包含头文件的库，但使用 Asio 的程序需要至少链接`boost_system`。在 Linux 上，我们可以使用以下命令行构建这个示例：

```cpp
$ g++ -g listing11_1.cpp -o listing11_1 -lboost_system -std=c++11

```

本章中的大多数示例都需要您链接到其他库。您可以使用以下命令行构建本章中的所有示例：

```cpp
$ g++ -g listing11_25.cpp -o listing11_25 -lboost_system -lboost_coroutine -lboost_date_time -std=c++11

```

如果您没有从本机包安装 Boost，并且需要在 Windows 上安装，请参考第一章*介绍 Boost*。

运行此程序会打印以下内容：

```cpp
Greetings: Hello, World!
```

请注意，在调用`run`（第 14 行）之前，`Greetings：`是从主函数（第 13 行）打印出来的。调用`run`最终会分派队列中的唯一处理程序，打印出`Hello, World!`。多个线程也可以调用相同的 I/O 对象上的`run`并发地分派处理程序。我们将在下一节中看到这如何有用。

### 处理程序状态 - run_one、poll 和 poll_one

虽然`run`函数会阻塞，直到队列中没有更多的处理程序，但`io_service`还有其他成员函数，让您以更大的灵活性处理处理程序。但在我们查看这个函数之前，我们需要区分挂起和准备好的处理程序。

我们发布到`io_service`的处理程序都准备立即运行，并在它们在队列上轮到时立即被调用。一般来说，处理程序与在底层操作系统中运行的后台任务相关联，例如网络 I/O 任务。这样的处理程序只有在关联的任务完成后才会被调用，这就是为什么在这种情况下，它们被称为**完成处理程序**。这些处理程序被称为**挂起**，直到关联的任务等待完成，一旦关联的任务完成，它们就被称为**准备**。

与`run`不同，`poll`成员函数会分派所有准备好的处理程序，但不会等待任何挂起的处理程序准备就绪。因此，如果没有准备好的处理程序，它会立即返回，即使有挂起的处理程序。`poll_one`成员函数如果有一个准备好的处理程序，会分派一个，但不会阻塞等待挂起的处理程序准备就绪。

`run_one`成员函数会在非空队列上阻塞，等待处理程序准备就绪。如果在空队列上调用它，它会返回，并且在找到并分派一个准备好的处理程序后立即返回。

### 发布与分派

对`post`成员函数的调用会将处理程序添加到任务队列并立即返回。稍后对`run`的调用负责调度处理程序。还有另一个名为`dispatch`的成员函数，可以用来请求`io_service`立即调用处理程序。如果在已经调用了`run`、`poll`、`run_one`或`poll_one`的线程中调用了`dispatch`，那么处理程序将立即被调用。如果没有这样的线程可用，`dispatch`会将处理程序添加到队列并像`post`一样立即返回。在以下示例中，我们从`main`函数和另一个处理程序中调用`dispatch`：

**清单 11.2：post 与 dispatch**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4
 5 int main() {
 6   asio::io_service service;
 7   // Hello Handler – dispatch behaves like post
 8   service.dispatch([]() { std::cout << "Hello\n"; });
 9
10   service.post(
11     [&service] { // English Handler
12       std::cout << "Hello, world!\n";
13       service.dispatch([] {  // Spanish Handler, immediate
14                          std::cout << "Hola, mundo!\n";
15                        });
16     });
17   // German Handler
18   service.post([&service] {std::cout << "Hallo, Welt!\n"; });
19   service.run();
20 }
```

运行此代码会产生以下输出：

```cpp
Hello
Hello, world!
Hola, mundo!
Hallo, Welt!
```

对`dispatch`的第一次调用（第 8 行）将处理程序添加到队列中而不调用它，因为`io_service`上尚未调用`run`。我们称这个为 Hello 处理程序，因为它打印`Hello`。然后是两次对`post`的调用（第 10 行，第 18 行），它们分别添加了两个处理程序。这两个处理程序中的第一个打印`Hello, world!`（第 12 行），然后调用`dispatch`（第 13 行）添加另一个打印西班牙问候语`Hola, mundo!`（第 14 行）的处理程序。这两个处理程序中的第二个打印德国问候语`Hallo, Welt`（第 18 行）。为了方便起见，让我们称它们为英文、西班牙文和德文处理程序。这在队列中创建了以下条目：

```cpp
Hello Handler
English Handler
German Handler
```

现在，当我们在`io_service`上调用`run`（第 19 行）时，首先调度 Hello 处理程序并打印`Hello`。然后是英文处理程序，它打印`Hello, World!`并在`io_service`上调用`dispatch`，传递西班牙处理程序。由于这在已经调用`run`的线程的上下文中执行，对`dispatch`的调用会调用西班牙处理程序，打印`Hola, mundo!`。随后，德国处理程序被调度打印`Hallo, Welt!`，在`run`返回之前。

如果英文处理程序调用`post`而不是`dispatch`（第 13 行），那么西班牙处理程序将不会立即被调用，而是在德国处理程序之后排队。德国问候语`Hallo, Welt!`将在西班牙问候语`Hola, mundo!`之前出现。输出将如下所示：

```cpp
Hello
Hello, world!
Hallo, Welt!
Hola, mundo!
```

## 通过线程池并发执行

`io_service`对象是线程安全的，多个线程可以同时在其上调用`run`。如果队列中有多个处理程序，它们可以被这些线程同时处理。实际上，调用`run`的一组线程在给定的`io_service`上形成一个**线程池**。后续的处理程序可以由池中的不同线程处理。哪个线程调度给定的处理程序是不确定的，因此处理程序代码不应该做出任何这样的假设。在以下示例中，我们将一堆处理程序发布到`io_service`，然后启动四个线程，它们都在其上调用`run`：

**清单 11.3：简单线程池**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/thread.hpp>
 3 #include <boost/date_time.hpp>
 4 #include <iostream>
 5 namespace asio = boost::asio;
 6
 7 #define PRINT_ARGS(msg) do {\
 8   boost::lock_guard<boost::mutex> lg(mtx); \
 9   std::cout << '[' << boost::this_thread::get_id() \
10             << "] " << msg << std::endl; \
11 } while (0)
12
13 int main() {
14   asio::io_service service;
15   boost::mutex mtx;
16
17   for (int i = 0; i < 20; ++i) {
18     service.post([i, &mtx]() { 
19                          PRINT_ARGS("Handler[" << i << "]");
20                          boost::this_thread::sleep(
21                               boost::posix_time::seconds(1));
22                        });
23   }
24
25   boost::thread_group pool;
26   for (int i = 0; i < 4; ++i) {
27     pool.create_thread([&service]() { service.run(); });
28   }
29
30   pool.join_all();
31 }
```

我们在循环中发布了 20 个处理程序（第 18 行）。每个处理程序打印其标识符（第 19 行），然后休眠一秒钟（第 19-20 行）。为了运行处理程序，我们创建了一个包含四个线程的组，每个线程在`io_service`上调用 run（第 21 行），并等待所有线程完成（第 24 行）。我们定义了宏`PRINT_ARGS`，它以线程安全的方式将输出写入控制台，并标记当前线程 ID（第 7-10 行）。我们以后也会在其他示例中使用这个宏。

要构建此示例，您还必须链接`libboost_thread`、`libboost_date_time`，在 Posix 环境中还必须链接`libpthread`。

```cpp
$ g++ -g listing9_3.cpp -o listing9_3 -lboost_system -lboost_thread -lboost_date_time -pthread -std=c++11

```

在我的笔记本电脑上运行此程序的一个特定运行产生了以下输出（有些行被剪掉）：

```cpp
[b5c15b40] Handler[0]
[b6416b40] Handler[1]
[b6c17b40] Handler[2]
[b7418b40] Handler[3]
[b5c15b40] Handler[4]
[b6416b40] Handler[5]
…
[b6c17b40] Handler[13]
[b7418b40] Handler[14]
[b6416b40] Handler[15]
[b5c15b40] Handler[16]
[b6c17b40] Handler[17]
[b7418b40] Handler[18]
[b6416b40] Handler[19]
```

您可以看到不同的处理程序由不同的线程执行（每个线程 ID 标记不同）。

### 提示

如果任何处理程序抛出异常，它将传播到执行处理程序的线程上对`run`函数的调用。

### io_service::work

有时，即使没有处理程序要调度，保持线程池启动也是有用的。`run`和`run_one`都不会在空队列上阻塞。因此，为了让它们阻塞等待任务，我们必须以某种方式指示有未完成的工作要执行。我们通过创建`io_service::work`的实例来实现这一点，如下例所示：

**11.4 节：使用 io_service::work 保持线程忙碌**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <memory>
 3 #include <boost/thread.hpp>
 4 #include <iostream>
 5 namespace asio = boost::asio;
 6
 7 typedef std::unique_ptr<asio::io_service::work> work_ptr;
 8
 9 #define PRINT_ARGS(msg) do {\ … 
...
14
15 int main() {
16   asio::io_service service;
17   // keep the workers occupied
18   work_ptr work(new asio::io_service::work(service));
19   boost::mutex mtx;
20
21   // set up the worker threads in a thread group
22   boost::thread_group workers;
23   for (int i = 0; i < 3; ++i) {
24     workers.create_thread([&service, &mtx]() {
25                          PRINT_ARGS("Starting worker thread ");
26                          service.run();
27                          PRINT_ARGS("Worker thread done");
28                        });
29   }
30
31   // Post work
32   for (int i = 0; i < 20; ++i) {
33     service.post(
34       [&service, &mtx]() {
35         PRINT_ARGS("Hello, world!");
36         service.post([&mtx]() {
37                            PRINT_ARGS("Hola, mundo!");
38                          });
39       });
40   }
41
42   work.reset(); // destroy work object: signals end of work
43   workers.join_all(); // wait for all worker threads to finish
44 }
```

在这个例子中，我们创建了一个包装在`unique_ptr`中的`io_service::work`对象（第 18 行）。我们通过将`io_service`对象的引用传递给`work`构造函数，将其与`io_service`对象关联起来。请注意，与 11.3 节不同，我们首先创建了工作线程（第 24-27 行），然后发布了处理程序（第 33-39 行）。然而，由于调用`run`阻塞（第 26 行），工作线程会一直等待处理程序。这是因为我们创建的`io_service::work`对象指示`io_service`队列中有未完成的工作。因此，即使所有处理程序都被调度，线程也不会退出。通过在包装`work`对象的`unique_ptr`上调用`reset`，其析构函数被调用，通知`io_service`所有未完成的工作已完成（第 42 行）。线程中的`run`调用返回，一旦所有线程都加入，程序就会退出（第 43 行）。我们将`work`对象包装在`unique_ptr`中，以便在程序的适当位置以异常安全的方式销毁它。 

我们在这里省略了`PRINT_ARGS`的定义，请参考 11.3 节。

## 通过 strands 进行序列化和有序执行

线程池允许处理程序并发运行。这意味着访问共享资源的处理程序需要同步访问这些资源。我们在 11.3 和 11.4 节中已经看到了这方面的例子，当我们同步访问全局对象`std::cout`时。作为在处理程序中编写同步代码的替代方案，我们可以使用**strands**。

### 提示

将 strand 视为任务队列的子序列，其中没有两个来自同一 strand 的处理程序会同时运行。

队列中的其他处理程序的调度不受 strand 的影响。让我们看一个使用 strands 的例子：

**11.5 节：使用 strands**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/thread.hpp>
 3 #include <boost/date_time.hpp>
 4 #include <cstdlib>
 5 #include <iostream>
 6 #include <ctime>
 7 namespace asio = boost::asio;
 8 #define PRINT_ARGS(msg) do {\
...
13
14 int main() {
15   std::srand(std::time(0));
16   asio::io_service service;
17   asio::io_service::strand strand(service);
18   boost::mutex mtx;
19   size_t regular = 0, on_strand = 0;
20 
21  auto workFuncStrand = [&mtx, &on_strand] {
22           ++on_strand;
23           PRINT_ARGS(on_strand << ". Hello, from strand!");
24           boost::this_thread::sleep(
25                       boost::posix_time::seconds(2));
26         };
27
28   auto workFunc = [&mtx, &regular] {
29                   PRINT_ARGS(++regular << ". Hello, world!");
30                   boost::this_thread::sleep(
31                         boost::posix_time::seconds(2));
32                 };
33   // Post work
34   for (int i = 0; i < 15; ++i) {
35     if (rand() % 2 == 0) {
36       service.post(strand.wrap(workFuncStrand));
37     } else {
38       service.post(workFunc);
39     }
40   }
41
42   // set up the worker threads in a thread group
43   boost::thread_group workers;
44   for (int i = 0; i < 3; ++i) {
45     workers.create_thread([&service, &mtx]() {
46                        PRINT_ARGS("Starting worker thread ");
47                       service.run();
48                        PRINT_ARGS("Worker thread done");
49                     });
50   }
51
52   workers.join_all(); // wait for all worker threads to finish
53 }
```

在这个例子中，我们创建了两个处理程序函数：`workFuncStrand`（第 21 行）和`workFunc`（第 28 行）。lambda `workFuncStrand`捕获一个计数器`on_strand`，递增它，并打印一个带有计数器值前缀的消息`Hello, from strand!`。函数`workFunc`捕获另一个计数器`regular`，递增它，并打印带有计数器前缀的消息`Hello, World!`。两者在返回前暂停 2 秒。

要定义和使用 strand，我们首先创建一个与`io_service`实例关联的`io_service::strand`对象（第 17 行）。然后，我们通过使用`strand`的`wrap`成员函数（第 36 行）将所有要成为该 strand 一部分的处理程序发布。或者，我们可以直接使用 strand 的`post`或`dispatch`成员函数发布处理程序到 strand，如下面的代码片段所示：

```cpp
33   for (int i = 0; i < 15; ++i) {
34     if (rand() % 2 == 0) {
35       strand.post(workFuncStrand);
37     } else {
...
```

strand 的`wrap`成员函数返回一个函数对象，该函数对象调用 strand 上的`dispatch`来调用原始处理程序。最初，添加到队列中的是这个函数对象，而不是我们的原始处理程序。当得到适当的调度时，这将调用原始处理程序。对于这些包装处理程序的调度顺序没有约束，因此，原始处理程序被调用的实际顺序可能与它们被包装和发布的顺序不同。

另一方面，在线程上直接调用`post`或`dispatch`可以避免中间处理程序。直接向线程发布也可以保证处理程序将按照发布的顺序进行分发，实现线程中处理程序的确定性排序。`strand`的`dispatch`成员会阻塞，直到处理程序被分发。`post`成员只是将其添加到线程并返回。

请注意，`workFuncStrand`在没有同步的情况下递增`on_strand`（第 22 行），而`workFunc`在`PRINT_ARGS`宏（第 29 行）中递增计数器`regular`，这确保递增发生在临界区内。`workFuncStrand`处理程序被发布到一个线程中，因此可以保证被序列化；因此不需要显式同步。另一方面，整个函数通过线程串行化，无法同步较小的代码块。在线程上运行的处理程序和其他处理程序之间没有串行化；因此，对全局对象的访问，如`std::cout`，仍然必须同步。

运行上述代码的示例输出如下：

```cpp
[b73b6b40] Starting worker thread 
[b73b6b40] 0\. Hello, world from strand!
[b6bb5b40] Starting worker thread 
[b6bb5b40] 1\. Hello, world!
[b63b4b40] Starting worker thread 
[b63b4b40] 2\. Hello, world!
[b73b6b40] 3\. Hello, world from strand!
[b6bb5b40] 5\. Hello, world!
[b63b4b40] 6\. Hello, world!
…
[b6bb5b40] 14\. Hello, world!
[b63b4b40] 4\. Hello, world from strand!
[b63b4b40] 8\. Hello, world from strand!
[b63b4b40] 10\. Hello, world from strand!
[b63b4b40] 13\. Hello, world from strand!
[b6bb5b40] Worker thread done
[b73b6b40] Worker thread done
[b63b4b40] Worker thread done
```

线程池中有三个不同的线程，并且线程中的处理程序由这三个线程中的两个选择：最初由线程 ID`b73b6b40`选择，后来由线程 ID`b63b4b40`选择。这也消除了一个常见的误解，即所有线程中的处理程序都由同一个线程分发，这显然不是这样。

### 提示

同一线程中的不同处理程序可能由不同的线程分发，但永远不会同时运行。

# 使用 Asio 进行网络 I/O

我们希望使用 Asio 构建可扩展的网络服务，执行网络 I/O。这些服务接收来自远程机器上运行的客户端的请求，并通过网络向它们发送信息。跨机器边界的进程之间的数据传输，通过网络进行，使用某些网络通信协议。其中最普遍的协议是 IP 或**Internet Protocol**及其上层的**一套协议**。Boost Asio 支持 TCP、UDP 和 ICMP，这三种流行的 IP 协议套件中的协议。本书不涵盖 ICMP。

## UDP 和 TCP

**用户数据报协议**或 UDP 用于在 IP 网络上从一个主机向另一个主机传输**数据报**或消息单元。UDP 是一个基于 IP 的非常基本的协议，它是无状态的，即在多个网络 I/O 操作之间不维护任何上下文。使用 UDP 进行数据传输的可靠性取决于底层网络的可靠性，UDP 传输具有以下注意事项：

+   UDP 数据报可能根本不会被传递

+   给定的数据报可能会被传递多次

+   两个数据报可能不会按照从源发送到目的地的顺序被传递

+   UDP 将检测数据报的任何数据损坏，并丢弃这样的消息，没有任何恢复的手段

因此，UDP 被认为是一种不可靠的协议。

如果应用程序需要协议提供更强的保证，我们选择**传输控制协议**或 TCP。TCP 使用字节流而不是消息进行处理。它在网络通信的两个端点之间使用握手机制建立持久的**连接**，并在连接的生命周期内维护状态。两个端点之间的所有通信都发生在这样的连接上。以比 UDP 略高的延迟为代价，TCP 提供以下保证：

+   在给定的连接上，接收应用程序按照发送顺序接收发送方发送的字节流

+   在传输过程中丢失或损坏的数据可以重新传输，大大提高了交付的可靠性

可以自行处理不可靠性和数据丢失的实时应用通常使用 UDP。此外，许多高级协议都是在 UDP 之上运行的。TCP 更常用，其中正确性关注超过实时性能，例如电子邮件和文件传输协议，HTTP 等。

## IP 地址

IP 地址是用于唯一标识连接到 IP 网络的接口的数字标识符。较旧的 IPv4 协议在 4 十亿（2³²）地址的地址空间中使用 32 位 IP 地址。新兴的 IPv6 协议在 3.4 × 10³⁸（2¹²⁸）个唯一地址的地址空间中使用 128 位 IP 地址，这几乎是不可枯竭的。您可以使用类`boost::asio::ip::address`表示两种类型的 IP 地址，而特定版本的地址可以使用`boost::asio::ip::address_v4`和`boost::asio::ip::address_v6`表示。

### IPv4 地址

熟悉的 IPv4 地址，例如 212.54.84.93，是以*点分四进制表示法*表示的 32 位无符号整数；四个 8 位无符号整数或*八位字节*表示地址中的四个字节，从左到右依次是最重要的，以点（句号）分隔。每个八位字节的范围是从 0 到 255。IP 地址通常以网络字节顺序解释，即大端序。

#### 子网

较大的计算机网络通常被划分为称为**子网**的逻辑部分。子网由一组可以使用广播消息相互通信的节点组成。子网有一个关联的 IP 地址池，具有一个共同的前缀，通常称为*路由前缀*或*网络地址*。IP 地址字段的剩余部分称为*主机部分*。

给定 IP 地址*和*前缀长度，我们可以使用**子网掩码**计算前缀。子网的子网掩码是一个 4 字节的位掩码，与子网中的 IP 地址进行按位与运算得到路由前缀。对于具有长度为 N 的路由前缀的子网，子网掩码的最高有效 N 位设置，剩余的 32-N 位未设置。子网掩码通常以点分四进制表示法表示。例如，如果地址 172.31.198.12 具有长度为 16 位的路由前缀，则其子网掩码将为 255.255.0.0，路由前缀将为 172.31.0.0。

一般来说，路由前缀的长度必须明确指定。**无类域间路由**（**CIDR）表示法**使用点分四进制表示法，并在末尾加上一个斜杠和一个介于 0 和 32 之间的数字，表示前缀长度。因此，10.209.72.221/22 表示具有前缀长度为 22 的 IP 地址。一个旧的分类方案，称为*有类方案*，将 IPv4 地址空间划分为范围，并为每个范围分配一个*类*（见下表）。属于每个范围的地址被认为是相应类的地址，并且路由前缀的长度是基于类确定的，而不是使用 CIDR 表示法指定。

| 类 | 地址范围 | 前缀长度 | 子网掩码 | 备注 |
| --- | --- | --- | --- | --- |
| 类 A | 0.0.0.0 – 127.255.255.255 | 8 | 255.0.0.0 |   |
| 类 B | 128.0.0.0 – 191.255.255.255 | 16 | 255.255.0.0 |   |
| 类 C | 192.0.0.0 – 223.255.255.255 | 24 | 255.255.255.0 |   |
| 类 D | 224.0.0.0 – 239.255.255.255 | 未指定 | 未指定 | 多播 |
| 类 E | 240.0.0.0 – 255.255.255.255 | 未指定 | 未指定 | 保留 |

#### 特殊地址

一些 IPv4 地址具有特殊含义。例如，主机部分中所有位设置的 IP 地址被称为子网的**广播地址**，用于向子网中的所有主机广播消息。例如，网络 172.31.0.0/16 中的广播地址为 172.31.255.255。

监听传入请求的应用程序使用**未指定地址** 0.0.0.0（`INADDR_ANY`）来监听所有可用的网络接口，而无需知道系统上的地址。

**回环地址** 127.0.0.1 通常与一个虚拟网络接口相关联，该接口不与任何硬件相关，并且不需要主机连接到网络。通过回环接口发送的数据立即显示为发送方主机上的接收数据。经常用于在一个盒子内测试网络应用程序，您可以配置额外的回环接口，并将回环地址从 127.0.0.0 到 127.255.255.255 的范围关联起来。

#### 使用 Boost 处理 IPv4 地址

现在让我们看一个构造 IPv4 地址并从中获取有用信息的代码示例，使用类型`boost::asio::ip::address_v4`：

**清单 11.6：处理 IPv4 地址**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 #include <vector>
 5 namespace asio = boost::asio;
 6 namespace sys = boost::system;
 7 using namespace asio::ip;
 8
 9 void printAddrProperties(const address& addr) {
10   std::cout << "\n\n" << addr << ": ";
11
12   if (addr.is_v4()) {
13     std::cout << "netmask=" << address_v4::netmask(addr.to_v4());
14   } else if (addr.is_v6()) { /* ... */ }
15
16   if (addr.is_unspecified()) { std::cout << "is unspecified, "; }
17   if (addr.is_loopback()) { std::cout << "is loopback, "; }
18   if (addr.is_multicast()) { std::cout << "is multicast, "; }
19 }
20
21 int main() {
22   sys::error_code ec;
23   std::vector<address> addresses;
24   std::vector<const char*> addr_strings{"127.0.0.1", 
25            "10.28.25.62", "137.2.33.19", "223.21.201.30",
26            "232.28.25.62", "140.28.25.62/22"};
27
28   addresses.push_back(address_v4());       // default: 0.0.0.0
29   addresses.push_back(address_v4::any());  // INADDR_ANY
30
31   for (const auto& v4str : addr_strings) {
32     address_v4 addr = address_v4::from_string(v4str, ec);
33     if (!ec) {
34       addresses.push_back(addr);
35     }
36   }
37
38   for (const address& addr1: addresses) {
39     printAddrProperties(addr1);
40   }
41 }
```

这个例子突出了 IPv4 地址的一些基本操作。我们创建了一个`boost::asio::ip::address`对象的向量（不仅仅是`address_v4`），并从它们的字符串表示中构造 IPv4 地址，使用`address_v4::from_string`静态函数（第 32 行）。我们使用`from_string`的两个参数重载，它接受地址字符串和一个非 const 引用到`error_code`对象，如果无法解析地址字符串，则设置该对象。存在一个单参数重载，如果有错误则抛出。请注意，您可以隐式转换或分配`address_v4`实例到`address`实例。默认构造的`address_v4`实例等同于未指定地址 0.0.0.0（第 28 行），也可以由`address_v4::any()`（第 29 行）返回。

为了打印地址的属性，我们编写了`printAddrProperties`函数（第 9 行）。我们通过将 IP 地址流式传输到`std::cout`（第 10 行）来打印 IP 地址。我们使用`is_v4`和`is_v6`成员函数（第 12、14 行）来检查地址是 IPv4 还是 IPv6 地址，使用`address_v4::netmask`静态函数（第 13 行）打印 IPv4 地址的网络掩码，并使用适当的成员谓词（第 16-18 行）检查地址是否为未指定地址、回环地址或 IPv4 多播地址（类 D）。请注意，`address_v4::from_string`函数不识别 CIDR 格式（截至 Boost 版本 1.57），并且网络掩码是基于类别的方案计算的。

在下一节中，我们将在简要概述 IPv6 地址之后，增强`printAddrProperties`（第 14 行）函数，以打印 IPv6 特定属性。

### IPv6 地址

在其最一般的形式中，IPv6 地址被表示为由冒号分隔的八个 2 字节无符号十六进制整数序列。按照惯例，十六进制整数中的数字`a`到`f`以小写字母写入，并且每个 16 位数字中的前导零被省略。以下是以这种表示法的 IPv6 地址的一个例子：

2001:0c2f:003a:01e0:0000:0000:0000:002a

两个或多个零项的序列可以完全折叠。因此，前面的地址可以写成 2001:c2f:3a:1e0::2a。所有前导零已被移除，并且在字节 16 和 63 之间的连续零项已被折叠，留下了冒号对(::)。如果有多个零项序列，则折叠最长的序列，如果有平局，则折叠最左边的序列。因此，我们可以将 2001:0000:0000:01e0:0000:0000:001a:002a 缩写为 2001::1e0:0:0:1a:2a。请注意，最左边的两个零项序列被折叠，而 32 到 63 位之间的其他零项未被折叠。

在从 IPv4 过渡到 IPv6 的环境中，软件通常同时支持 IPv4 和 IPv6。*IPv4 映射的 IPv6 地址*用于在 IPv6 和 IPv4 接口之间进行通信。IPv4 地址被映射到具有::ffff:0:0/96 前缀和最后 32 位与 IPv4 地址相同的 IPv6 地址。例如，172.31.201.43 将表示为::ffff:172.31.201.43/96。

#### 地址类、范围和子网

IPv6 地址有三类：

+   **单播地址**：这些地址标识单个网络接口

+   **多播地址**：这些地址标识一组网络接口，并用于向组中的所有接口发送数据

+   **任播地址**：这些地址标识一组网络接口，但发送到**任播**地址的数据将传递给距离发送者拓扑最近的一个或多个接口，而不是传递给组中的所有接口

在单播和任播地址中，地址的最低有效 64 位表示主机 ID。一般来说，高阶 64 位表示网络前缀。

每个 IPv6 地址也有一个**范围**，用于标识其有效的网络段：

+   **节点本地**地址，包括环回地址，用于节点内通信。

+   **全局**地址是可通过网络到达的可路由地址。

+   **链路本地**地址会自动分配给每个启用 IPv6 的接口，并且只能在网络内访问，也就是说，路由器不会路由到链路本地地址的流量。即使具有可路由地址，链路本地地址也会分配给接口。链路本地地址的前缀为 fe80::/64。

#### 特殊地址

IPv6 的**环回地址**类似于 IPv4 中的 127.0.0.1，为::1。在 IPv6 中，**未指定地址**（全零）写为::（`in6addr_any`）。IPv6 中没有广播地址，多播地址用于定义接收方接口的组，这超出了本书的范围。

#### 使用 Boost 处理 IPv6 地址

在下面的例子中，我们构造 IPv6 地址，并使用`boost::asio::ip::address_v6`类查询这些地址的属性：

**列表 11.7：处理 IPv6 地址**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 #include <vector>
 4 namespace asio = boost::asio;
 5 namespace sys = boost::system;
 6 using namespace asio::ip;
 7
 8 void printAddr6Properties(const address_v6& addr) {
 9   if (addr.is_v4_mapped()) { std::cout << "is v4-mapped, "; }
10   else {  
11     if (addr.is_link_local()) { std::cout << "is link local";}
12   }  
13 }
14
15 void printAddrProperties(const address& addr) { ... }
16
17 int main() {
18   sys::error_code ec;
19   std::vector<address> addresses;
20   std::vector<const char*> addr_strings{"::1", "::",
21     "fe80::20", "::ffff:223.18.221.9", "2001::1e0:0:0:1a:2a"};
22
23   for (const auto& v6str: addr_strings) {
24     address addr = address_v6::from_string(v6str, ec);
25     if (!ec) { addresses.push_back(addr); }
26   }
27
28   for (const auto& addr : addresses) {
29     printAddrProperties(addr);
30   }
31 }
```

这个例子通过 IPv6 特定的检查增强了列表 11.6。函数`printAddrProperties`（第 15 行）与列表 11.6 中的相同，因此不再完整重复。`printAddr6Properties`函数（第 8 行）检查地址是否为 IPv4 映射的 IPv6 地址（第 9 行），以及它是否为链路本地地址（第 11 行）。其他相关检查已经通过`printAddrProperties`中的与版本无关的`address`成员执行（参见列表 11.6）。

我们创建一个`boost::asio::ip::address`对象的向量（不仅仅是`address_v6`），并推送由它们的字符串表示构造的 IPv6 地址，使用`address_v6::from_string`静态函数（第 24 行），它返回`address_v6`对象，可以隐式转换为`address`。请注意，我们有环回地址、未指定地址、IPv4 映射地址、常规 IPv6 单播地址和链路本地地址（第 20-21 行）。

## 端点、套接字和名称解析

应用程序在提供网络服务时绑定到 IP 地址，多个应用程序从 IP 地址开始发起对其他应用程序的出站通信。多个应用程序可以使用不同的**端口**绑定到同一个 IP 地址。端口是一个无符号的 16 位整数，与 IP 地址和协议（TCP、UDP 等）一起，唯一标识一个通信**端点**。数据通信发生在两个这样的端点之间。Boost Asio 为 UDP 和 TCP 提供了不同的端点类型，即`boost::asio::ip::udp::endpoint`和`boost::asio::ip::tcp::endpoint`。

### 端口

许多标准和广泛使用的网络服务使用固定的众所周知的端口。端口 0 到 1023 分配给众所周知的系统服务，包括 FTP、SSH、telnet、SMTP、DNS、HTTP 和 HTTPS 等。广泛使用的应用程序可以在 1024 到 49151 之间注册标准端口，由**互联网编号分配机构**（**IANA**）负责。49151 以上的端口可以被任何应用程序使用，无需注册。通常将众所周知的端口映射到服务的映射通常保存在磁盘文件中，例如在 POSIX 系统上是`/etc/services`，在 Windows 上是`%SYSTEMROOT%\system32\drivers\etc\services`。

### 套接字

**套接字**表示用于网络通信的端点。它表示通信通道的一端，并提供执行所有数据通信的接口。Boost Asio 为 UDP 和 TCP 提供了不同的套接字类型，即`boost::asio::ip::udp::socket`和`boost::asio::ip::tcp::socket`。套接字始终与相应的本地端点对象相关联。所有现代操作系统上的本机网络编程接口都使用某种伯克利套接字 API 的衍生版本，这是用于执行网络通信的 C API。Boost Asio 库提供了围绕这个核心 API 构建的类型安全抽象。

套接字是**I/O 对象**的一个例子。在 Asio 中，I/O 对象是用于启动 I/O 操作的对象类。这些操作由底层操作系统的**I/O 服务**对象分派，该对象是`boost::asio::io_service`的实例。在本章的前面，我们看到了 I/O 服务对象作为任务管理器的实例。但是它们的主要作用是作为底层操作系统上操作的接口。每个 I/O 对象都是使用关联的 I/O 服务实例构造的。通过这种方式，高级 I/O 操作在 I/O 对象上启动，但是 I/O 对象和 I/O 服务之间的交互保持封装。在接下来的章节中，我们将看到使用 UDP 和 TCP 套接字进行网络通信的示例。

### 主机名和域名

通过名称而不是数字地址来识别网络中的主机通常更方便。域名系统（DNS）提供了一个分层命名系统，其中网络中的主机通过带有唯一名称的主机名来标识，该名称标识了网络，称为**完全限定域名**或简称**域名**。例如，假想的域名`elan.taliesyn.org`可以映射到 IP 地址 140.82.168.29。在这里，`elan`将标识特定主机，`taliesyn.org`将标识主机所属的域。在同一网络中，不同组的计算机可能报告给不同的域，甚至某台计算机可能属于多个域。

#### 名称解析

全球范围内的 DNS 服务器层次结构以及私人网络内的 DNS 服务器维护名称到地址的映射。应用程序询问配置的 DNS 服务器以解析完全限定域名到地址。如果有的话，DNS 服务器将请求解析为 IP 地址，否则将其转发到层次结构更高的另一个 DNS 服务器。如果直到层次结构的根部都没有答案，解析将失败。发起这种名称解析请求的专门程序或库称为**解析器**。Boost Asio 提供了特定协议的解析器：`boost::asio::ip::tcp::resolver`和`boost::asio::ip::udp::resolver`用于执行此类名称解析。我们查询主机名上的服务，并获取该服务的一个或多个端点。以下示例显示了如何做到这一点，给定一个主机名，以及可选的服务名或端口：

**清单 11.8：查找主机的 IP 地址**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4
 5 int main(int argc, char *argv[]) {
 6   if (argc < 2) {
 7     std::cout << "Usage: " << argv[0] << " host [service]\n";
 8     exit(1);
 9   }
10   const char *host = argv[1];
11   const char *svc = (argc > 2) ? argv[2] : "";
12
13   try {
14     asio::io_service service;
15     asio::ip::tcp::resolver resolver(service);
16     asio::ip::tcp::resolver::query query(host, svc);
17     asio::ip::tcp::resolver::iterator end,
18                             iter = resolver.resolve(query);
19     while (iter != end) {
20       asio::ip::tcp::endpoint endpoint = iter->endpoint();
21       std::cout << "Address: " << endpoint.address()
22                 << ", Port: " << endpoint.port() << '\n';
23       ++iter;
24     }
25   } catch (std::exception& e) {
26     std::cout << e.what() << '\n';
27   }
28 }
```

您可以通过在命令行上传递主机名和可选的服务名来运行此程序。该程序将这些解析为 IP 地址和端口，并将它们打印到标准输出（第 21-22 行）。程序创建了一个`io_service`实例（第 14 行），它将成为底层操作系统操作的通道，以及一个`boost::asio::ip::tcp::resolver`实例（第 15 行），它提供了请求名称解析的接口。我们根据主机名和服务名创建一个名称查找请求，封装在一个`query`对象中（第 16 行），并调用`resolver`的`resolve`成员函数，将`query`对象作为参数传递（第 18 行）。`resolve`函数返回一个**endpoint iterator**，指向查询解析的一系列`endpoint`对象。我们遍历这个序列，打印每个端点的地址和端口号。如果有的话，这将打印 IPv4 和 IPv6 地址。如果我们想要特定于 IP 版本的 IP 地址，我们需要使用`query`的三参数构造函数，并在第一个参数中指定协议。例如，要仅查找 IPv6 地址，我们可以使用这个：

```cpp
asio::ip::tcp::resolver::query query(asio::ip::tcp::v6(), 
 host, svc);

```

在查找失败时，`resolve`函数会抛出异常，除非我们使用接受非 const 引用`error_code`的两参数版本，并在错误时设置它。在下面的例子中，我们执行反向查找。给定一个 IP 地址和一个端口，我们查找关联的主机名和服务名：

**清单 11.9：查找主机和服务名称**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4
 5 int main(int argc, char *argv[]) {
 6   if (argc < 2) {
 7     std::cout << "Usage: " << argv[0] << " ip [port]\n";
 8     exit(1);
 9   }
10
11   const char *addr = argv[1];
12   unsigned short port = (argc > 2) ? atoi(argv[2]) : 0;
13
14   try {
15     asio::io_service service;
16     asio::ip::tcp::endpoint ep(
17               asio::ip::address::from_string(addr), port);
18     asio::ip::tcp::resolver resolver(service);
19     asio::ip::tcp::resolver::iterator iter = 
20                              resolver.resolve(ep), end;
21     while (iter != end) {
22       std::cout << iter->host_name() << " "
23                 << iter->service_name() << '\n';
24       iter++;
25     }
26   } catch (std::exception& ex) {
27     std::cerr << ex.what() << '\n';
28   }
29 }
```

我们从命令行传递 IP 地址和端口号给程序，然后使用它们构造`endpoint`（第 16-17 行）。然后我们将`endpoint`传递给`resolver`的`resolve`成员函数（第 19 行），并遍历结果。在这种情况下，迭代器指向`boost::asio::ip::tcp::query`对象，我们使用适当的成员函数打印每个对象的主机和服务名称（第 22-23 行）。

## 缓冲区

数据作为字节流通过网络发送或接收。一个连续的字节流可以用一对值来表示：序列的起始地址和序列中的字节数。Boost Asio 提供了两种用于这种序列的抽象，`boost::asio::const_buffer`和`boost::asio::mutable_buffer`。`const_buffer`类型表示一个只读序列，通常用作发送数据时的数据源。`mutable_buffer`表示一个读写序列，当您需要在缓冲区中添加或更新数据时使用，例如当您从远程主机接收数据时：

**清单 11.10：使用 const_buffer 和 mutable_buffer**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 namespace asio = boost::asio;
 5
 6 int main() {
 7   char buf[10];
 8   asio::mutable_buffer mbuf(buf, sizeof(buf));
 9   asio::const_buffer cbuf(buf, 5);
10
11   std::cout << buffer_size(mbuf) << '\n';
12   std::cout << buffer_size(cbuf) << '\n';
13
14   char *mptr = asio::buffer_cast<char*>(mbuf);
15   const char *cptr = asio::buffer_cast<const char*>(cbuf);
16   assert(mptr == cptr && cptr == buf);
17   
18   size_t offset = 5;
19   asio::mutable_buffer mbuf2 = mbuf + offset;
20   assert(asio::buffer_cast<char*>(mbuf2)
21         - asio::buffer_cast<char*>(mbuf) == offset);
22   assert(buffer_size(mbuf2) == buffer_size(mbuf) - offset);
23 }
```

在这个例子中，我们展示了如何将 char 数组包装在`mutable_buffer`和`const_buffer`中（第 8-9 行）。在构造缓冲区时，您需要指定内存区域的起始地址和区域的字节数。`const char`数组只能被包装在`const_buffer`中，而不能被包装在`mutable_buffer`中。这些缓冲区包装器*不*分配存储空间，不管理任何堆分配的内存，也不执行任何数据复制。

函数`boost::asio::buffer_size`返回缓冲区的字节长度（第 11-12 行）。这是您在构造缓冲区时传递的长度，它不依赖于缓冲区中的数据。默认初始化的缓冲区长度为零。

函数模板`boost::asio::buffer_cast<>`用于获取缓冲区的基础字节数组的指针（第 14-15 行）。请注意，如果我们尝试使用`buffer_cast`从`const_buffer`获取可变数组，将会得到编译错误：

```cpp
asio::const_buffer cbuf(addr, length);
char *buf = asio::buffer_cast<char*>(cbuf); // fails to compile

```

最后，您可以使用`operator+`从另一个缓冲区的偏移量创建一个缓冲区（第 19 行）。结果缓冲区的长度将比原始缓冲区的长度少偏移量的长度（第 22 行）。

### 向量 I/O 的缓冲区序列

有时，从一系列缓冲区发送数据或将接收到的数据分割到一系列缓冲区中是很方便的。每个序列调用一次网络 I/O 函数会很低效，因为这些调用最终会转换为系统调用，并且每次调用都会有开销。另一种选择是使用可以处理作为参数传递给它的缓冲区序列的网络 I/O 函数。这通常被称为**向量 I/O**或**聚集-分散 I/O**。Boost Asio 的所有 I/O 函数都处理缓冲区序列，因此必须传递缓冲区序列而不是单个缓冲区。用于 Asio I/O 函数的有效缓冲区序列满足以下条件：

+   有一个返回双向迭代器的成员函数`begin`，该迭代器指向`mutable_buffer`或`const_buffer`

+   有一个返回指向序列末尾的迭代器的成员函数`end`

+   可复制

要使缓冲区序列有用，它必须是`const_buffer`序列或`mutable_buffer`序列。形式上，这些要求总结在**ConstBufferSequence**和**MutableBufferSequence**概念中。这是一组稍微简化的条件，但对我们的目的来说已经足够了。我们可以使用标准库容器（如`std::vector`、`std::list`等）以及 Boost 容器来创建这样的序列。然而，由于我们经常只处理单个缓冲区，Boost 提供了`boost::asio::buffer`函数，可以轻松地将单个缓冲区适配为长度为 1 的缓冲区序列。以下是一个简短的示例，说明了这些想法：

**清单 11.11：使用缓冲区**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <vector>
 3 #include <string>
 4 #include <iostream>
 5 #include <cstdlib>
 6 #include <ctime>
 7 namespace asio = boost::asio;
 8
 9 int main() {
10   std::srand(std::time(nullptr));
11
12   std::vector<char> v1(10);
13   char a2[10];
14   std::vector<asio::mutable_buffer> bufseq(2);
15
16   bufseq.push_back(asio::mutable_buffer(v1.data(), 
17                                         v1.capacity()));
18   bufseq.push_back(asio::mutable_buffer(a2, sizeof(a2)));
19
20   for (auto cur = asio::buffers_begin(bufseq),
21        end = asio::buffers_end(bufseq); cur != end; cur++) {
22     *cur = 'a' + rand() % 26;
23   }
24
25   std::cout << "Size: " << asio::buffer_size(bufseq) << '\n';
26
27   std::string s1(v1.begin(), v1.end());
28   std::string s2(a2, a2 + sizeof(a2));
29
30   std::cout << s1 << '\n' << s2 << '\n';
31 }
```

在这个示例中，我们创建一个`vector`的两个`mutable_buffer`的可变缓冲区序列（第 14 行）。这两个可变缓冲区包装了一个`char`的`vector`（第 16-17 行）和一个`char`的数组（第 18 行）。使用`buffers_begin`函数（第 20 行）和`buffers_end`函数（第 21 行），我们确定了缓冲区序列`bufseq`所封装的字节的整个范围，并遍历它，将每个字节设置为随机字符（第 22 行）。当这些写入底层的 vector 或数组时，我们使用底层的 vector 或数组构造字符串并打印它们的内容（第 27-28 行）。

## 同步和异步通信

在接下来的几节中，我们将整合我们迄今为止学到的 IP 地址、端点、套接字、缓冲区和其他 Asio 基础设施的理解，来编写网络客户端和服务器程序。我们的示例使用**客户端-服务器模型**进行交互，其中**服务器**程序服务于传入的请求，而**客户端**程序发起这些请求。这样的客户端被称为**主动端点**，而这样的服务器被称为**被动端点**。

客户端和服务器可以进行**同步**通信，即在每个网络 I/O 操作上阻塞，直到请求被底层操作系统处理，然后才继续下一步。或者，它们可以使用**异步 I/O**，在不等待完成的情况下启动网络 I/O，并在稍后被通知其完成。与同步情况不同，使用异步 I/O 时，程序不会在需要执行 I/O 操作时空闲等待。因此，异步 I/O 在具有更多对等方和更大数据量时具有更好的扩展性。我们将研究通信的同步和异步模型。虽然异步交互的编程模型是事件驱动的且更复杂，但使用 Boost Asio 协程可以使其非常易于管理。在编写 UDP 和 TCP 服务器之前，我们将看一下 Asio 截止时间定时器，以了解如何使用 Asio 编写同步和异步逻辑。

## Asio 截止时间定时器

Asio 提供了`basic_deadline_timer`模板，使用它可以等待特定持续时间的过去或绝对时间点。特化的`deadline_timer`定义如下：

```cpp
typedef basic_deadline_timer<boost::posix_time::ptime> 
                                             deadline_timer;
```

它使用`boost::posix_time::ptime`和`boost::posix_time::time_duration`作为时间点和持续时间类型。下面的例子演示了一个应用程序如何使用`deadline_timer`等待一段时间：

**清单 11.12：同步等待**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/date_time.hpp>
 3 #include <iostream>
 4
 5 int main() {
 6   boost::asio::io_service service;
 7   boost::asio::deadline_timer timer(service);
 8
 9   long secs = 5;
10   std::cout << "Waiting for " << secs << " seconds ..." 
11             << std::flush;
12   timer.expires_from_now(boost::posix_time::seconds(secs));
13
14   timer.wait();
15 
16   std::cout << " done\n";
17 }
```

我们创建了一个`io_service`对象（第 6 行），它作为底层操作的通道。我们创建了一个与`io_service`相关联的`deadline_timer`实例（第 7 行）。我们使用`deadline_timer`的`expires_from_now`成员函数指定了一个 5 秒的等待时间（第 12 行）。然后我们调用`wait`成员函数来阻塞直到时间到期。注意我们不需要在`io_service`实例上调用`run`。我们可以使用`expires_at`成员函数来等待到特定的时间点，就像这样：

```cpp
using namespace boost::gregorian;
using namespace boost::posix_time;

timer.expires_at(day_clock::local_day(), 
                 hours(16) + minutes(12) + seconds(58));
```

有时，程序不想阻塞等待定时器触发，或者一般来说，不想阻塞等待它感兴趣的任何未来事件。与此同时，它可以完成其他有价值的工作，因此比起阻塞等待事件，它可以更加响应。我们不想在事件上阻塞，只是想告诉定时器在触发时通知我们，并且同时进行其他工作。为此，我们调用`async_wait`成员函数，并传递一个*完成处理程序*。完成处理程序是我们使用`async_wait`注册的函数对象，一旦定时器到期就会被调用：

**清单 11.13：异步等待**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/date_time.hpp>
 3 #include <iostream>
 4
 5 void on_timer_expiry(const boost::system::error_code& ec)
 6 {
 7   if (ec) {
 8     std::cout << "Error occurred while waiting\n";
 9   } else {
10     std::cout << "Timer expired\n";
11   }
12 }
13
14 int main()
15 {
16   boost::asio::io_service service;
17   boost::asio::deadline_timer timer(service);
18
19
20   long secs = 5;
21   timer.expires_from_now(boost::posix_time::seconds(secs));
22
23   std::cout << "Before calling deadline_timer::async_wait\n";
24   timer.async_wait(on_timer_expiry);
25   std::cout << "After calling deadline_timer::async_wait\n";
26
27   service.run();
28 }
```

与清单 11.12 相比，清单 11.13 有两个关键的变化。我们调用`deadline_timer`的`async_wait`成员函数而不是`wait`，并传递一个指向完成处理程序函数`on_timer_expiry`的指针。然后在`io_service`对象上调用`run`。当我们运行这个程序时，它会打印以下内容：

```cpp
Before calling deadline_timer::async_wait
After calling deadline_timer::async_wait
Timer expired
```

调用`async_wait`不会阻塞（第 24 行），因此前两行消息会快速连续打印出来。随后，调用`run`（第 27 行）会阻塞直到定时器到期，并且定时器的完成处理程序被调度。除非发生了错误，否则完成处理程序会打印`Timer expired`。因此，第一和第二条消息出现与第三条消息之间存在时间差，第三条消息是完成处理程序的输出。

## 使用 Asio 协程的异步逻辑

`deadline_timer`的`async_wait`成员函数启动了一个异步操作。这样的函数在启动的操作完成之前就返回了。它注册了一个完成处理程序，并且通过调用这个处理程序来通知程序异步事件的完成。如果我们需要按顺序运行这样的异步操作，控制流就会变得复杂。例如，假设我们想等待 5 秒，打印`Hello`，然后再等待 10 秒，最后打印`world`。使用同步的`wait`，就像下面的代码片段一样简单：

```cpp
boost::asio::deadline_timer timer;
timer.expires_from_now(boost::posix_time::seconds(5));
timer.wait();
std::cout << "Hello, ";
timer.expires_from_now(boost::posix_time::seconds(10));
timer.wait();
std::cout << "world!\n";
```

在许多现实场景中，特别是在网络 I/O 中，阻塞同步操作根本不是一个选择。在这种情况下，代码变得更加复杂。使用`async_wait`作为模型异步操作，下面的例子演示了异步代码的复杂性：

**清单 11.14：异步操作**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/bind.hpp>
 3 #include <boost/date_time.hpp>
 4 #include <iostream>
 5
 6 void print_world(const boost::system::error_code& ec) {
 7   std::cout << "world!\n";
 8 }
 9
10 void print_hello(boost::asio::deadline_timer& timer,
11                  const boost::system::error_code& ec) {
12   std::cout << "Hello, " << std::flush;
13
14   timer.expires_from_now(boost::posix_time::seconds(10));
15   timer.async_wait(print_world);
16 }
17
18 int main()
19 {
20   boost::asio::io_service service;
21   boost::asio::deadline_timer timer(service);
22   timer.expires_from_now(boost::posix_time::seconds(5));
23
24   timer.async_wait(boost::bind(print_hello, boost::ref(timer),
25                                            ::_1));
26
27   service.run();
28 }
```

将相同功能从同步逻辑转换为异步逻辑，代码行数超过两倍，控制流也变得复杂。我们将函数`print_hello`（第 10 行）注册为第一个 5 秒等待的完成处理程序（第 22、24 行）。`print_hello`又使用同一个定时器开始了一个 10 秒的等待，并将函数`print_world`（第 6 行）注册为这个等待的完成处理程序（第 14-15 行）。

请注意，我们使用`boost::bind`为第一个 5 秒等待生成完成处理程序，将`timer`从`main`函数传递给`print_hello`函数。因此，`print_hello`函数使用相同的计时器。为什么我们需要这样做呢？首先，`print_hello`需要使用相同的`io_service`实例来启动 10 秒等待操作和之前的 5 秒等待。`timer`实例引用了这个`io_service`实例，并且被两个完成处理程序使用。此外，在`print_hello`中创建一个本地的`deadline_timer`实例会有问题，因为`print_hello`会在计时器响起之前返回，并且本地计时器对象会被销毁，所以它永远不会响起。

示例 11.14 说明了*控制流反转*的问题，在异步编程模型中是一个重要的复杂性来源。我们不能再将一系列语句串在一起，并假设每个语句只有在前面的操作完成后才会启动一个操作——这对于同步模型是一个安全的假设。相反，我们依赖于`io_service`的通知来确定运行下一个操作的正确时间。逻辑在函数之间分散，需要更多的努力来管理需要在这些函数之间共享的任何数据。

Asio 使用 Boost Coroutine 库的薄包装简化了异步编程。与 Boost Coroutine 一样，可以使用有栈和无栈协程。在本书中，我们只研究有栈协程。

使用`boost::asio::spawn`函数模板，我们可以启动任务作为协程。如果一个协程被调度并调用了一个异步函数，那么协程会被暂停。与此同时，`io_service`会调度其他任务，包括其他协程。一旦异步操作完成，启动它的协程会恢复，并继续下一步。在下面的清单中，我们使用协程重写清单 11.14：

**清单 11.15：使用协程进行异步编程**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/asio/spawn.hpp>
 3 #include <boost/bind.hpp>
 4 #include <boost/date_time.hpp>
 5 #include <iostream>
 6
 7 void wait_and_print(boost::asio::yield_context yield,
 8                     boost::asio::io_service& service)
 9 {
10   boost::asio::deadline_timer timer(service);
11
12   timer.expires_from_now(boost::posix_time::seconds(5));
13   timer.async_wait(yield);
14   std::cout << "Hello, " << std::flush;
15 
16   timer.expires_from_now(boost::posix_time::seconds(10));
17   timer.async_wait(yield);
18   std::cout << "world!\n";
19 }
20
21 int main()
22 {
23   boost::asio::io_service service;
24   boost::asio::spawn(service,
25           boost::bind(wait_and_print, ::_1, 
26                                       boost::ref(service)));
27   service.run();
28 }
```

`wait_and_print`函数是协程，接受两个参数：一个`boost::asio::yield_context`类型的对象和一个`io_service`实例的引用（第 7 行）。`yield_context`是 Boost Coroutine 的薄包装。我们必须使用`boost::asio::spawn`来调度一个协程，这样一个协程必须具有`void (boost::asio::yield_context)`的签名。因此，我们使用`boost::bind`来使`wait_and_print`函数与`spawn`期望的协程签名兼容。我们将第二个参数绑定到`io_service`实例的引用（第 24-26 行）。

`wait_and_print`协程在堆栈上创建一个`deadline_timer`实例，并开始一个 5 秒的异步等待，将其`yield_context`传递给`async_wait`函数，而不是完成处理程序。这会暂停`wait_and_print`协程，只有在等待完成后才会恢复。与此同时，如果有其他任务，可以从`io_service`队列中处理。等待结束并且`wait_and_print`恢复后，它打印`Hello`并开始等待 10 秒。协程再次暂停，只有在 10 秒后才会恢复，然后打印`world`。协程使异步逻辑与同步逻辑一样简单易读，开销很小。在接下来的章节中，我们将使用协程来编写 TCP 和 UDP 服务器。

## UDP

UDP I/O 模型相对简单，客户端和服务器之间的区别模糊不清。对于使用 UDP 的网络 I/O，我们创建一个 UDP 套接字，并使用`send_to`和`receive_from`函数将数据报发送到特定的端点。

### 同步 UDP 客户端和服务器

在本节中，我们编写了一个 UDP 客户端（清单 11.16）和一个同步 UDP 服务器（清单 11.17）。UDP 客户端尝试向给定端点的 UDP 服务器发送一些数据。UDP 服务器阻塞等待从一个或多个 UDP 客户端接收数据。发送数据后，UDP 客户端阻塞等待从服务器接收响应。服务器在接收数据后，在继续处理更多传入消息之前发送一些响应。

**清单 11.16：同步 UDP 客户端**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 #include <exception>
 4 namespace asio = boost::asio;
 5
 6 int main(int argc, char *argv[]) {
 7   if (argc < 3) {
 8     std::cerr << "Usage: " << argv[0] << " host port\n";
 9     return 1;
10   }
11
12   asio::io_service service;
13   try {
14     asio::ip::udp::resolver::query query(asio::ip::udp::v4(),
15                                        argv[1], argv[2]);
16     asio::ip::udp::resolver resolver(service);
17     auto iter = resolver.resolve(query);
18     asio::ip::udp::endpoint endpoint = iter->endpoint();
19   
20     asio::ip::udp::socket socket(service, 
21                                  asio::ip::udp::v4());
22     const char *msg = "Hello from client";
23     socket.send_to(asio::buffer(msg, strlen(msg)), endpoint);
24     char buffer[256];
25     size_t recvd = socket.receive_from(asio::buffer(buffer,
26                                  sizeof(buffer)), endpoint);
27     buffer[recvd] = 0;
28     std::cout << "Received " << buffer << " from " 
29        << endpoint.address() << ':' << endpoint.port() << '\n';
30   } catch (std::exception& e) {
31     std::cerr << e.what() << '\n';
32   }
33 }
```

我们通过命令行向客户端传递服务器主机名和要连接的服务（或端口）。它们会解析为 UDP 的端点（IP 地址和端口号）（第 13-17 行），为 IPv4 创建一个 UDP 套接字（第 18 行），并在其上调用`send_to`成员函数。我们传递给`send_to`一个包含要发送的数据和目标端点的`const_buffer`（第 23 行）。

每个使用 Asio 执行网络 I/O 的程序都使用*I/O 服务*，它是类型`boost::asio::io_service`的实例。我们已经看到`io_service`作为任务管理器的作用。但是 I/O 服务的主要作用是作为底层操作的接口。Asio 程序使用负责启动 I/O 操作的*I/O 对象*。例如，套接字就是 I/O 对象。

我们调用 UDP 套接字的`send_to`成员函数，向服务器发送预定义的消息字符串（第 23 行）。请注意，我们将消息数组包装在长度为 1 的缓冲区序列中，该序列使用`boost::asio::buffer`函数构造，如本章前面所示，在缓冲区部分。一旦`send_to`完成，客户端在同一套接字上调用`recv_from`，传递一个可变的缓冲区序列，该序列由可写字符数组使用`boost::asio::buffer`构造（第 25-26 行）。`receive_from`的第二个参数是对`boost::asio::ip::udp::endpoint`对象的非 const 引用。当`receive_from`返回时，该对象包含发送消息的远程端点的地址和端口号（第 28-29 行）。

调用`send_to`和`receive_from`的是**阻塞调用**。调用`send_to`不会返回，直到传递给它的缓冲区已经被写入系统中的底层 UDP 缓冲区。将 UDP 缓冲区通过网络发送到服务器可能会在稍后发生。调用`receive_from`不会返回，直到接收到一些数据为止。

我们可以使用单个 UDP 套接字向多个其他端点发送数据，并且可以在单个套接字上从多个其他端点接收数据。因此，每次调用`send_to`都将目标端点作为输入。同样，每次调用`receive_from`都会使用非 const 引用传递一个端点，并在返回时将其设置为发送方的端点。现在我们将使用 Asio 编写相应的 UDP 服务器：

**清单 11.17：同步 UDP 服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <exception>
 4 #include <iostream>
 5 namespace asio = boost::asio;
 6
 8 int main() 
 9 {
10   const unsigned short port = 55000;
11   const std::string greet("Hello, world!");
12
13   asio::io_service service;
14   asio::ip::udp::endpoint endpoint(asio::ip::udp::v4(), port);
15   asio::ip::udp::socket socket(service, endpoint);
16   asio::ip::udp::endpoint ep;
17
18   while (true) try {	
19     char msg[256];
20     auto recvd = socket.receive_from(asio::buffer(msg, 
21                                             sizeof(msg)), ep);
22     msg[recvd] = 0;
23     std::cout << "Received: [" << msg << "] from [" 
24               << ep << "]\n";
25
26     socket.send_to(asio::buffer(greet.c_str(), greet.size()),
27                    ep);
27     socket.send_to(asio::buffer(msg, strlen(msg)), ep);
28   } catch (std::exception& e) {
29     std::cout << e.what() << '\n';
30   }
31 }
```

同步 UDP 服务器在端口 55000 上创建一个`boost::asio::ip::udp::endpoint`类型的单个 UDP 端点，保持地址未指定（第 14 行）。请注意，我们使用了一个两参数的`endpoint`构造函数，它将*协议*和端口作为参数。服务器为此端点创建一个`boost::asio::ip::udp::socket`类型的单个 UDP 套接字（第 15 行），并在循环中旋转，每次迭代调用套接字上的`receive_from`，等待直到客户端发送一些数据。数据以一个名为`msg`的`char`数组接收，该数组被包装在长度为一的可变缓冲序列中传递给`receive_from`。`receive_from`的调用返回接收到的字节数，用于在`msg`中添加一个终止空字符，以便它可以像 C 风格的字符串一样使用（第 22 行）。一般来说，UDP 将传入的数据呈现为包含一系列字节的消息，其解释留给应用程序。每当服务器从客户端接收数据时，它会将发送的数据回显，先前由一个固定的问候字符串。它通过在套接字上两次调用`send_to`成员函数来实现，传递要发送的缓冲区和接收方的端点（第 26-27 行，28 行）。

对`send_to`和`receive_from`的调用是同步的，只有当数据完全传递给操作系统（`send_to`）或应用程序完全接收到数据（`receive_from`）时才会返回。如果许多客户端实例同时向服务器发送消息，服务器仍然只能一次处理一条消息，因此客户端排队等待响应。当然，如果客户端不等待响应，它们可以全部发送消息并退出，但消息仍然会按顺序被服务器接收。

### 异步 UDP 服务器

UDP 服务器的异步版本可以显著提高服务器的响应性。传统的异步模型可能涉及更复杂的编程模型，但协程可以显著改善情况。

#### 使用完成处理程序链的异步 UDP 服务器

对于异步通信，我们使用`socket`的`async_receive_from`和`async_send_to`成员函数。这些函数不会等待 I/O 请求被操作系统处理，而是立即返回。它们被传递一个函数对象，当底层操作完成时将被调用。这个函数对象被排队在`io_service`的任务队列中，在操作系统上的实际操作返回时被调度。

```cpp
template <typename MutableBufSeq, typename ReadHandler>
deduced async_receive_from(
    const MutableBufSeq& buffers,
    endpoint_type& sender_ep,
 ReadHandler handler);

template <typename ConstBufSeq, typename WriteHandler>
deduced async_send_to(
    const ConstBufSeq& buffers,
    endpoint_type& sender_ep,
 WriteHandler handler);

```

传递给`async_receive_from`的读处理程序和传递给`async_send_to`的写处理程序的签名如下：

```cpp
void(const boost::system::error_code&, size_t)
```

处理程序期望传递一个非 const 引用给`error_code`对象，指示已完成操作的状态和读取或写入的字节数。处理程序可以调用其他异步 I/O 操作并注册其他处理程序。因此，整个 I/O 操作是以一系列处理程序的链条来定义的。现在我们来看一个异步 UDP 服务器的程序：

**清单 11.18：异步 UDP 服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4 namespace sys = boost::system;
 5
 6 const size_t MAXBUF = 256;
 7
 8 class UDPAsyncServer {
 9 public:
10   UDPAsyncServer(asio::io_service& service, 
11                  unsigned short port) 
12      : socket(service, 
13           asio::ip::udp::endpoint(asio::ip::udp::v4(), port))
14   {  waitForReceive();  }
15
16   void waitForReceive() {
17     socket.async_receive_from(asio::buffer(buffer, MAXBUF),
18           remote_peer,
19           [this] (const sys::error_code& ec,
20                   size_t sz) {
21             const char *msg = "hello from server";
22             std::cout << "Received: [" << buffer << "] "
23                       << remote_peer << '\n';
24             waitForReceive();
25
26             socket.async_send_to(
27                 asio::buffer(msg, strlen(msg)),
28                 remote_peer,
29                 this {});
31           });
32   }
33
34 private:
35   asio::ip::udp::socket socket;
36   asio::ip::udp::endpoint remote_peer;
37   char buffer[MAXBUF];
38 };
39
40 int main() {
41   asio::io_service service;
42   UDPAsyncServer server(service, 55000);
43   service.run();
44 }
```

UDP 服务器封装在`UDPAsyncServer`类中（第 8 行）。要启动服务器，我们首先创建必需的`io_service`对象（第 42 行），然后创建一个`UDPAsyncServer`实例（第 43 行），该实例传递了`io_service`实例和应该使用的端口号。最后，调用`io_service`的`run`成员函数开始处理传入的请求（第 44 行）。那么`UDPAsyncServer`是如何工作的呢？

`UDPAsyncServer`的构造函数使用本地端点初始化了 UDP `socket`成员（第 12-13 行）。然后调用成员函数`waitForReceive`（第 14 行），该函数又在套接字上调用`async_receive_from`（第 18 行），开始等待任何传入的消息。我们调用`async_receive_from`，传递了从`buffer`成员变量制作的可变缓冲区（第 17 行），对`remote_peer`成员变量的非 const 引用（第 18 行），以及一个定义接收操作完成处理程序的 lambda 表达式（第 19-31 行）。`async_receive_from`启动了一个 I/O 操作，将处理程序添加到`io_service`的任务队列中，然后返回。对`io_service`的`run`调用（第 43 行）会阻塞，直到队列中有 I/O 任务。当 UDP 消息到来时，数据被操作系统接收，并调用处理程序来采取进一步的操作。要理解 UDP 服务器如何无限处理更多消息，我们需要了解处理程序的作用。

接收处理程序在服务器接收到消息时被调用。它打印接收到的消息和远程发送者的详细信息（第 22-23 行），然后发出对`waitForReceive`的调用，从而重新启动接收操作。然后它发送一条消息`hello from server`（第 21 行）回到由`remote_peer`成员变量标识的发送者。它通过调用 UDP `socket`的`async_send_to`成员函数来实现这一点，传递消息缓冲区（第 27 行），目标端点（第 28 行），以及另一个以 lambda 形式的处理程序（第 29-32 行），该处理程序什么也不做。

请注意，我们在 lambda 中捕获了`this`指针，以便能够从周围范围访问成员变量（第 20 行，29 行）。另外，处理程序都没有使用`error_code`参数进行错误检查，这在现实世界的软件中是必须的。

#### 使用协程的异步 UDP 服务器

处理程序链接将逻辑分散到一组处理程序中，并且在处理程序之间共享状态变得特别复杂。这是为了更好的性能，但我们可以避免这个代价，就像我们在列表 11.15 中使用 Asio 协程处理`boost::asio::deadline_timer`上的异步等待一样。现在我们将使用 Asio 协程来编写一个异步 UDP 服务器：

**列表 11.19：使用 Asio 协程的异步 UDP 服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/asio/spawn.hpp>
 3 #include <boost/bind.hpp>
 4 #include <boost/shared_ptr.hpp>
 5 #include <boost/make_shared.hpp>
 6 #include <iostream>
 7 namespace asio = boost::asio;
 8 namespace sys = boost::system;
 9
10 const size_t MAXBUF = 256;
11 typedef boost::shared_ptr<asio::ip::udp::socket>
12                                   shared_udp_socket;
13
14 void udp_send_to(boost::asio::yield_context yield,
15                  shared_udp_socket socket,
16                  asio::ip::udp::endpoint peer)
17 {
18     const char *msg = "hello from server";
19     socket->async_send_to(asio::buffer(msg, std::strlen(msg)),
20                          peer, yield);
21 }
22
23 void udp_server(boost::asio::yield_context yield,
24                 asio::io_service& service,
25                 unsigned short port)
26 {
27   shared_udp_socket socket =
28       boost::make_shared<asio::ip::udp::socket>(service,
29           asio::ip::udp::endpoint(asio::ip::udp::v4(), port));
30
31   char buffer[MAXBUF];
32   asio::ip::udp::endpoint remote_peer;
33   boost::system::error_code ec;
34
35   while (true) {
36     socket->async_receive_from(asio::buffer(buffer, MAXBUF),
37                 remote_peer, yield[ec]);
38
39     if (!ec) {
40       spawn(socket->get_io_service(), 
41         boost::bind(udp_send_to, ::_1, socket,
42                                  remote_peer));
43     }
44   }
45 }
46
47 int main() {
48   asio::io_service service;
49   spawn(service, boost::bind(udp_server, ::_1,
50                      boost::ref(service), 55000));
51   service.run();                               
52 }
```

通过使用协程，异步 UDP 服务器的结构与列表 11.18 有了相当大的变化，并且更接近列表 11.17 的同步模型。函数`udp_server`包含了 UDP 服务器的核心逻辑（第 23 行）。它被设计为协程使用，因此它的一个参数是`boost::asio::yield_context`类型（第 23 行）。它还接受两个额外的参数：对`io_service`实例的引用（第 24 行）和 UDP 服务器端口（第 25 行）。

在主函数中，我们创建了一个`io_service`实例（第 48 行），然后添加一个任务以将`udp_server`作为协程运行，使用`boost::asio::spawn`函数模板（第 49-50 行）。我们适当地绑定了`udp_server`的服务和端口参数。然后我们调用`io_service`实例上的`run`来开始处理 I/O 操作。对`run`的调用会派发`udp_server`协程（第 51 行）。

`udp_server` 协程创建一个与未指定的 IPv4 地址（0.0.0.0）和作为参数传递的特定端口相关联的 UDP 套接字（第 27-29 行）。 套接字被包装在 `shared_ptr` 中，稍后将清楚其原因。 协程堆栈上有额外的变量来保存从客户端接收的数据（第 31 行）并标识客户端端点（第 32 行）。 `udp_server` 函数然后在循环中旋转，调用套接字上的 `async_receive_from`，传递接收处理程序的 `yield_context`（第 36-37 行）。 这会暂停 `udp_server` 协程的执行，直到 `async_receive_from` 完成。 与此同时，对 `run` 的调用会恢复并处理其他任务（如果有）。 一旦调用 `async_receive_from` 函数完成，`udp_server` 协程将恢复执行并继续进行其循环的下一次迭代。

对于每个完成的接收操作，`udp_server` 都会发送一个固定的问候字符串（“来自服务器的问候”）作为对客户端的响应。 发送这个问候的任务也封装在一个协程中，即 `udp_send_to`（第 14 行），`udp_server` 协程使用 `spawn`（第 40 行）将其添加到任务队列中。 我们将 UDP 套接字和标识客户端的端点作为参数传递给这个协程。 请注意，称为 `remote_peer` 的局部变量被按值传递给 `udp_send_to` 协程（第 42 行）。 这在 `udp_send_to` 中被使用，作为 `async_send_to` 的参数，用于指定响应的接收者（第 19-20 行）。 我们传递副本而不是引用给 `remote_peer`，因为当发出对 `async_send_to` 的调用时，另一个对 `async_receive_from` 的调用可能是活动的，并且可能在 `async_send_to` 使用之前覆盖 `remote_peer` 对象。 我们还传递了包装在 `shared_ptr` 中的套接字。 套接字不可复制，不像端点。 如果套接字对象在 `udp_server` 函数中的自动存储中，并且在仍有待处理的 `udp_send_to` 任务时 `udp_server` 退出，那么 `udp_send_to` 中的套接字引用将无效，并可能导致崩溃。 出于这个原因，`shared_ptr` 包装器是正确的选择。

如果您注意到，对 `async_receive_from` 的处理程序写为 `yield[ec]`（第 37 行）。 `yield_context` 类具有重载的下标运算符，我们可以使用它来指定对 `error_code` 类型的变量的可变引用。 当异步操作完成时，作为下标运算符参数传递的变量将设置为错误代码（如果有）。

### 提示

在编写异步服务器时，更倾向于使用协程而不是处理程序链。 协程使代码更简单，控制流更直观。

### 性能和并发

我们声称异步通信模式提高了服务器的响应性。 让我们确切地了解哪些因素导致了这种改进。 在列表 11.17 的同步模型中，除非 `send_to` 函数返回，否则无法发出对 `receive_from` 的调用。 在列表 11.18 的异步代码中，一旦接收并消耗了消息，就会立即调用 `waitForReceive`（第 23-25 行），它不会等待 `async_send_to` 完成。 同样，在列表 11.19 中，它展示了在异步模型中使用协程，协程帮助暂停等待异步 I/O 操作完成的函数，并同时继续处理队列中的其他任务。 这是异步服务器响应性改进的主要来源。

值得注意的是，在列表 11.18 中，所有 I/O 都在单个线程上进行。这意味着在任何给定时间点，我们的程序只处理一个传入的 UDP 消息。这使我们能够重用`buffer`和`remote_peer`成员变量，而不必担心同步。我们仍然必须确保在再次调用`waitForReceive`之前打印接收到的缓冲区（第 22-23 行）。如果我们颠倒了顺序，缓冲区可能会在打印之前被新的传入消息覆盖。

考虑一下，如果我们像这样在接收处理程序中调用`waitForReceive`而不是发送处理程序中：

```cpp
18     socket.async_receive_from(asio::buffer(buffer, MAXBUF),
19           remote_peer,
20           [this] (const sys::error_code& ec,
21                   size_t sz) {
...            ...
26             socket.async_send_to(
27                 asio::buffer(msg, strlen(msg)),
28                 remote_peer,
29                 this {
31                   waitForReceive();
32                 });
33           });
```

在这种情况下，接收将仅在发送完成后开始；因此，即使使用异步调用，它也不会比列表 11.17 中的同步示例更好。

在列表 11.18 中，我们在发送内容回来时不需要来自远程对等方的缓冲区，因此我们不需要在发送完成之前保留该缓冲区。这使我们能够在不等待发送完成的情况下开始异步接收（第 24 行）。接收可能会首先完成并覆盖缓冲区，但只要发送操作不使用缓冲区，一切都没问题。在现实世界中，这种情况经常发生，因此让我们看看如何在不延迟接收直到发送之后的情况下解决这个问题。以下是处理程序的修改实现：

```cpp
  17 void waitForReceive() {
 18   boost::shared_array<char> recvbuf(new char[MAXBUF]);
 19   auto epPtr(boost::make_shared<asio::ip::udp::endpoint>());
 20   socket.async_receive_from(
 21         asio::buffer(recvbuf.get(), MAXBUF),
  22         *epPtr,
 23         [this, recvbuf, epPtr] (const sys::error_code& ec,
  24                 size_t sz) {
 25           waitForReceive();
  26
  27           recvbuf[sz] = 0;
  28           std::ostringstream sout;
  29           sout << '[' << boost::this_thread::get_id()
  30                << "] Received: " << recvbuf.get()
  31                << " from client: " << *epPtr << '\n';
  32           std::cout << sout.str() << '\n';
  33           socket.async_send_to(
 34               asio::buffer(recvbuf.get(), sz),
  35               *epPtr,
 36               this, recvbuf, epPtr {
  38               });
  39        });
  40 }
```

现在，我们不再依赖于一个共享的成员变量作为缓冲区，而是为每个新消息分配一个接收缓冲区（第 18 行）。这消除了列表 11.18 中`buffer`成员变量的需要。我们使用`boost::shared_array`包装器，因为这个缓冲区需要从`waitForReceive`调用传递到接收处理程序，而且只有在最后一个引用消失时才应该释放它。同样，我们移除了代表远程端点的`remote_peer`成员变量，并为每个新请求使用了一个`shared_ptr`包装的端点。

我们将底层数组传递给`async_receive_from`（第 21 行），并通过在`async_receive_from`的完成处理程序中捕获其`shared_array`包装器（第 23 行）来确保它存活足够长的时间。出于同样的原因，我们还捕获端点包装器`epPtr`。接收处理程序调用`waitForReceive`（第 25 行），然后打印从客户端接收到的消息，并在当前线程的线程 ID 前加上前缀（考虑未来）。然后它调用`async_send_to`，传递接收到的缓冲区而不是一些固定的消息（第 34 行）。再一次，我们需要确保缓冲区和远程端点在发送完成之前存活；因此，我们在发送完成处理程序中捕获了缓冲区的`shared_array`包装器和远程端点的`shared_ptr`包装器（第 36 行）。

基于协程的异步 UDP 服务器的更改（列表 11.19）也是在同样的基础上进行的。

```cpp
 1 #include <boost/shared_array.hpp>
...
14 void udp_send_to(boost::asio::yield_context yield,
15               shared_udp_socket socket,
16               asio::ip::udp::endpoint peer,
17               boost::shared_array<char> buffer, size_t size)
18 {
19     const char *msg = "hello from server";
20     socket->async_send_to(asio::buffer(msg, std::strlen(msg)),
21                          peer, yield);
22     socket->async_send_to(asio::buffer(buffer.get(), size),
23                           peer, yield);
24 }
25
26 void udp_server(boost::asio::yield_context yield,
27                 asio::io_service& service,
28                 unsigned short port)
29 {
30   shared_udp_socket socket =
31       boost::make_shared<asio::ip::udp::socket>(service,
32           asio::ip::udp::endpoint(asio::ip::udp::v4(), port));
33
34   asio::ip::udp::endpoint remote_peer;
35   boost::system::error_code ec;
36
38   while (true) {
39     boost::shared_array<char> buffer(new char[MAXBUF]);
40     size_t size = socket->async_receive_from(
41                       asio::buffer(buffer.get(), MAXBUF),
42                       remote_peer, yield[ec]);
43
44     if (!ec) {
45       spawn(socket->get_io_service(), 
46         boost::bind(udp_send_to, ::_1, socket, remote_peer,
47                                  buffer, size));
43     }
44   }
45 }
```

由于需要将从客户端接收的数据回显回去，`udp_send_to`协程必须访问它。因此，它将包含接收到的数据的缓冲区和读取的字节数作为参数（第 17 行）。为了确保这些数据不会被后续接收覆盖，我们必须在`udp_server`循环的每次迭代中为接收数据分配缓冲区（第 39 行）。我们将这个缓冲区，以及`async_receive_from`返回的读取的字节数（第 40 行），传递给`udp_send_to`（第 47 行）。通过这些更改，我们的异步 UDP 服务器现在可以在响应对等方之前保持每个传入请求的上下文，而无需延迟处理新请求的需要。

这些更改还使处理程序线程安全，因为实质上，我们删除了处理程序之间的任何共享数据。虽然`io_service`仍然是共享的，但它是一个线程安全的对象。我们可以很容易地将 UDP 服务器转换为多线程服务器。下面是我们如何做到这一点：

```cpp
46 int main() {
47   asio::io_service service;
48   UDPAsyncServer server(service, 55000);
49
50   boost::thread_group pool;
51   pool.create_thread([&service] { service.run(); });
52   pool.create_thread([&service] { service.run(); });
53   pool.create_thread([&service] { service.run(); });
54   pool.create_thread([&service] { service.run(); });
55   pool.join_all();
56 }
```

这将创建四个处理传入 UDP 消息的工作线程。使用协程也可以实现相同的功能。

## TCP

在网络 I/O 方面，UDP 的编程模型非常简单——你要么发送消息，要么接收消息，要么两者都做。相比之下，TCP 是一个相当复杂的东西，它的交互模型有一些额外的细节需要理解。

除了可靠性保证外，TCP 还实现了几个巧妙的算法，以确保过于热心的发送方不会用大量数据淹没相对较慢的接收方（**流量控制**），并且所有发送方都能公平地分享网络带宽（**拥塞控制**）。TCP 层需要进行相当多的计算来实现这一切，并且需要维护一些状态信息来执行这些计算。为此，TCP 使用端点之间的**连接**。

### 建立 TCP 连接

**TCP 连接**由一对 TCP 套接字组成，可能位于不同主机上，通过 IP 网络连接，并带有一些相关的状态数据。相关的连接状态信息在连接的每一端都得到维护。**TCP 服务器**通常开始*监听传入连接*，被称为连接的**被动端**。**TCP 客户端**发起连接到 TCP 服务器的请求，并被称为*主动端*。一个被称为**TCP 三次握手**的明确定义的机制用于建立 TCP 连接。类似的机制也存在于协调连接终止。连接也可以被单方面重置或终止，比如在应用程序或主机因各种原因关闭或发生不可恢复的错误的情况下。

#### 客户端和服务器端的调用

要建立 TCP 连接，服务器进程必须在一个端点上监听，并且客户端进程必须主动发起到该端点的连接。服务器执行以下步骤：

1.  创建一个 TCP 监听套接字。

1.  为监听传入连接创建一个本地端点，并将 TCP 监听套接字绑定到该端点。

1.  开始在监听器上监听传入的连接。

1.  接受任何传入的连接，并打开一个服务器端点（与监听端点不同）来服务该连接。

1.  在该连接上进行通信。

1.  处理连接的终止。

1.  继续监听其他传入的连接。

客户端依次执行以下步骤：

1.  创建一个 TCP 套接字，并可选地将其绑定到本地端点。

1.  连接到由 TCP 服务器提供服务的远程端点。

1.  一旦连接建立，就在该连接上进行通信。

1.  处理连接的终止。

### 同步 TCP 客户端和服务器

我们现在将编写一个 TCP 客户端，它连接到指定主机和端口上的 TCP 服务器，向服务器发送一些文本，然后从服务器接收一些消息：

**清单 11.20：同步 TCP 客户端**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <iostream>
 3 namespace asio = boost::asio;
 4
 5 int main(int argc, char* argv[]) {
 6   if (argc < 3) {
 7     std::cerr << "Usage: " << argv[0] << " host port\n";
 8     exit(1);
 9   }
10
11   const char *host = argv[1], *port = argv[2];
12
13   asio::io_service service;
14   asio::ip::tcp::resolver resolver(service);
15   try {
16     asio::ip::tcp::resolver::query query(asio::ip::tcp::v4(),
17                                        host, port);
18     asio::ip::tcp::resolver::iterator end, 
19                        iter = resolver.resolve(query);
20
21     asio::ip::tcp::endpoint server(iter->endpoint());
22     std::cout << "Connecting to " << server << '\n';
23     asio::ip::tcp::socket socket(service, 
24                                  asio::ip::tcp::v4());
25     socket.connect(server);
26     std::string message = "Hello from client";
27     asio::write(socket, asio::buffer(message.c_str(),
28                                    message.size()));
29     socket.shutdown(asio::ip::tcp::socket::shutdown_send);
30 
31     char msg[BUFSIZ];
32     boost::system::error_code ec;
33     size_t sz = asio::read(socket, 
34                          asio::buffer(msg, BUFSIZ), ec);
35     if (!ec || ec == asio::error::eof) {
36       msg[sz] = 0;
37       std::cout << "Received: " << msg << '\n';
38     } else {
39       std::cerr << "Error reading response from server: "
40                 << ec.message() << '\n';
41     }
34   } catch (std::exception& e) {
35     std::cerr << e.what() << '\n';
36   }
37 }
```

TCP 客户端解析传递给它的主机和端口（或服务名称）（第 16-19 行），并创建一个表示要连接的服务器的端点（第 21 行）。它创建一个 IPv4 套接字（第 23 行），并调用`connect`成员函数来启动与远程服务器的连接（第 25 行）。`connect`调用会阻塞，直到建立连接，或者如果连接尝试失败则抛出异常。连接成功后，我们使用`boost::asio::write`函数将文本`Hello from client`发送到服务器（第 27-28 行）。我们调用套接字的`shutdown`成员函数，参数为`shutdown_send`（第 29 行），关闭与服务器的写通道。这在服务器端显示为 EOF。然后我们使用`read`函数接收服务器发送的任何消息（第 33-34 行）。`boost::asio::write`和`boost::asio::read`都是阻塞调用。对于失败的`write`调用会抛出异常，例如，如果连接被重置或由于服务器繁忙而发送超时。我们调用`read`的非抛出重载，在失败时，它会将我们传递给它的错误代码设置为非 const 引用。

函数`boost::asio::read`尝试读取尽可能多的字节以填充传递的缓冲区，并阻塞，直到所有数据到达或接收到文件结束符。虽然文件结束符被`read`标记为错误条件，但它可能只是表示服务器已经完成发送数据，我们对接收到的任何数据感兴趣。因此，我们特别使用`read`的非抛出重载，并在`error_code`引用中设置错误时，区分文件结束符和其他错误（第 35 行）。出于同样的原因，我们调用`shutdown`关闭此连接的写通道（第 29 行），以便服务器不等待更多输入。

### 提示

与 UDP 不同，TCP 是面向流的，并且不定义消息边界。应用程序必须定义自己的机制来识别消息边界。一些策略包括在消息前面加上消息的长度，使用字符序列作为消息结束标记，或者使用固定长度的消息。在本书的示例中，我们使用`tcp::socket`的`shutdown`成员函数，这会导致接收方读取文件结束符，表示我们已经完成发送消息。这使示例保持简单，但实际上，这不是最灵活的策略。

现在让我们编写 TCP 服务器，它将处理来自此客户端的请求：

**清单 11.21：同步 TCP 服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/thread.hpp>
 3 #include <boost/shared_ptr.hpp>
 4 #include <boost/array.hpp>
 5 #include <iostream>
 6 namespace asio = boost::asio;
 7
 8 typedef boost::shared_ptr<asio::ip::tcp::socket> socket_ptr;
 9
10 int main() {
11   const unsigned short port = 56000;
12   asio::io_service service;
13   asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port);
14   asio::ip::tcp::acceptor acceptor(service, endpoint);
15
16   while (true) {
17     socket_ptr socket(new asio::ip::tcp::socket(service));
18     acceptor.accept(*socket);
19     boost::thread([socket]() {
20       std::cout << "Service request from "
21                 << socket->remote_endpoint() << '\n';
22       boost::array<asio::const_buffer, 2> bufseq;
23       const char *msg = "Hello, world!";
24       const char *msg2 = "What's up?";
25       bufseq[0] = asio::const_buffer(msg, strlen(msg));
26       bufseq[1] = asio::const_buffer(msg2, strlen(msg2));
27 
28       try {
29         boost::system::error_code ec;
30         char recvbuf[BUFSIZ];
31         auto sz = read(*socket, asio::buffer(recvbuf,
32                                             BUFSIZ), ec);
33         if (!ec || ec == asio::error::eof) {
34           recvbuf[sz] = 0;
35           std::cout << "Received: " << recvbuf << " from "
36                     << socket->remote_endpoint() << '\n';
37           write(*socket, bufseq);
38           socket->close();
39         }
40       } catch (std::exception& e) {
41         std::cout << "Error encountered: " << e.what() << '\n';
42       }
43     });
44   }
45 }
```

TCP 服务器的第一件事是创建一个监听套接字并将其绑定到本地端点。使用 Boost Asio，您可以通过创建`asio::ip::tcp::acceptor`的实例并将其传递给要绑定的端点来实现这一点（第 14 行）。我们创建一个 IPv4 端点，只指定端口而不指定地址，以便使用未指定地址 0.0.0.0（第 13 行）。我们通过将其传递给`acceptor`的构造函数将端点绑定到监听器（第 14 行）。然后我们在循环中等待传入的连接（第 16 行）。我们需要一个独立的套接字来作为每个新连接的服务器端点，因此我们创建一个新的套接字（第 17 行）。然后我们在接受器上调用`accept`成员函数（第 18 行），将新套接字传递给它。`accept`调用会阻塞，直到建立新连接。当`accept`返回时，传递给它的套接字表示建立的连接的服务器端点。

我们创建一个新线程来为每个建立的新连接提供服务（第 19 行）。我们使用 lambda（第 19-44 行）生成此线程的初始函数，捕获此连接的`shared_ptr`包装的服务器端`socket`（第 19 行）。在线程内部，我们调用`read`函数来读取客户端发送的数据（第 31-32 行），然后使用`write`写回数据（第 37 行）。为了展示如何做到这一点，我们从两个字符字符串设置的多缓冲序列中发送数据（第 22-26 行）。此线程中的网络 I/O 在 try 块内完成，以确保没有异常逃逸出线程。请注意，在`write`返回后我们在 socket 上调用`close`（第 38 行）。这关闭了服务器端的连接，客户端在接收流中读取到文件结束符。

#### 并发和性能

TCP 服务器独立处理每个连接。但是为每个新连接创建一个新线程的扩展性很差，如果在非常短的时间内有大量连接到达服务器，服务器的资源可能会耗尽。处理这种情况的一种方法是限制线程数量。之前，我们修改了清单 11.18 中的 UDP 服务器示例，使用了线程池并限制了总线程数量。我们可以对清单 11.21 中的 TCP 服务器做同样的事情。以下是如何实现的概述：

```cpp
12 asio::io_service service;
13 boost::unique_ptr<asio::io_service::work> workptr(
14                                    new dummyWork(service));
15 auto threadFunc = [&service] { service.run(); };
16 
17 boost::thread_group workers;
18 for (int i = 0; i < max_threads; ++i) { //max_threads
19   workers.create_thread(threadFunc);
20 }
21
22 asio::ip::tcp::endpoint ep(asio::ip::tcp::v4(), port);
23 asio::ip::tcp::acceptor acceptor(service, ep);24 while (true) {
25   socket_ptr socket(new asio::ip::tcp::socket(service));
26   acceptor.accept(*socket);
27
28   service.post([socket] { /* do I/O on the connection */ });
29 }
30
31 workers.join_all();
32 workptr.reset(); // we don't reach here
```

首先，我们创建了一个固定数量线程的线程池（第 15-20 行），并通过向`io_service`的任务队列发布一个虚拟工作（第 13-14 行）来确保它们不会退出。我们不是为每个新连接创建一个线程，而是将连接的处理程序发布到`io_service`的任务队列中（第 28 行）。这个处理程序可以与清单 11.21 中每个连接线程的初始函数完全相同。然后线程池中的线程按照自己的时间表分派处理程序。`max_threads`表示的线程数量可以根据系统中的处理器数量轻松调整。

虽然使用线程池限制了线程数量，但对于服务器的响应性几乎没有改善。在大量新连接涌入时，新连接的处理程序会在队列中形成一个大的积压，这些客户端将被保持等待，而服务器则服务于先前的连接。我们已经通过使用异步 I/O 在 UDP 服务器中解决了类似的问题。在下一节中，我们将使用相同的策略来更好地扩展我们的 TCP 服务器。

### 异步 TCP 服务器

同步 TCP 服务器效率低下主要是因为套接字上的读写操作会阻塞一段有限的时间，等待操作完成。在此期间，即使有线程池，服务连接的线程也只是空闲地等待 I/O 操作完成，然后才能处理下一个可用连接。

我们可以使用异步 I/O 来消除这些空闲等待。就像我们在异步 UDP 服务器中看到的那样，我们可以使用处理程序链或协程来编写异步 TCP 服务器。虽然处理程序链使代码复杂，因此容易出错，但协程使代码更易读和直观。我们将首先使用协程编写一个异步 TCP 服务器，然后再使用更传统的处理程序链，以便更好地理解这两种方法之间的差异。在第一次阅读时，您可以跳过处理程序链的实现。

#### 使用协程的异步 TCP 服务器

以下是使用协程进行异步 I/O 的 TCP 服务器的完整代码：

**清单 11.22：使用协程的异步 TCP 服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/asio/spawn.hpp>
 3 #include <boost/thread.hpp>
 4 #include <boost/shared_ptr.hpp>
 5 #include <boost/make_shared.hpp>
 6 #include <boost/bind.hpp>
 7 #include <boost/array.hpp>
 8 #include <iostream>
 9 #include <cstring>
10
11 namespace asio = boost::asio;
12 typedef boost::shared_ptr<asio::ip::tcp::socket> socketptr;
13
14 void handle_connection(asio::yield_context yield,
15                        socketptr socket)
16 {
17   asio::io_service& service = socket->get_io_service();
18   char msg[BUFSIZ];
19   msg[0] = '\0';
20   boost::system::error_code ec;
21   const char *resp = "Hello from server";
22
23   size_t size = asio::async_read(*socket, 
24                      asio::buffer(msg, BUFSIZ), yield[ec]);
25
26   if (!ec || ec == asio::error::eof) {
27     msg[size] = '\0';
28     boost::array<asio::const_buffer, 2> bufseq;
29     bufseq[0] = asio::const_buffer(resp, ::strlen(resp));
30     bufseq[1] = asio::const_buffer(msg, size);
31
32     asio::async_write(*socket, bufseq, yield[ec]);
33     if (ec) {
34       std::cerr << "Error sending response to client: "
35                 << ec.message() << '\n';
36     }
37   } else {
38     std::cout << ec.message() << '\n';
39   }
40 }
41
42 void accept_connections(asio::yield_context yield,
43                         asio::io_service& service,
44                         unsigned short port)
45 {
46   asio::ip::tcp::endpoint server_endpoint(asio::ip::tcp::v4(),
47                                           port);
48   asio::ip::tcp::acceptor acceptor(service, server_endpoint);
49
50   while (true) {
51     auto socket = 
52         boost::make_shared<asio::ip::tcp::socket>(service);
53     acceptor.async_accept(*socket, yield);
54
55     std::cout << "Handling request from client\n";
56     spawn(service, boost::bind(handle_connection, ::_1, 
57                                socket));
58   }
59 }
60
61 int main() {
62   asio::io_service service;
63   spawn(service, boost::bind(accept_connections, ::_1,
64                              boost::ref(service), 56000));
65   service.run();
66 }
```

我们使用了两个协程：`accept_connections`处理传入的连接请求（第 42 行），而`handle_connection`在每个新连接上执行 I/O（第 14 行）。`main`函数调用`spawn`函数模板将`accept_connections`任务添加到`io_service`队列中，以作为协程运行（第 63 行）。`spawn`函数模板可通过头文件`boost/asio/spawn.hpp`（第 2 行）获得。调用`io_service`的`run`成员函数会调用`accept_connections`协程，该协程在一个循环中等待新的连接请求（第 65 行）。

`accept_connections`函数除了强制的`yield_context`之外，还接受两个参数。这些是对`io_service`实例的引用，以及用于监听新连接的端口——在`main`函数在生成此协程时绑定的值（第 63-64 行）。`accept_connections`函数为未指定的 IPv4 地址和传递的特定端口创建一个端点（第 46-47 行），并为该端点创建一个接受者（第 48 行）。然后，在循环的每次迭代中调用接受者的`async_accept`成员函数，传递一个 TCP 套接字的引用，并将本地的`yield_context`作为完成处理程序（第 53 行）。这会暂停`accept_connections`协程，直到接受到新的连接。一旦接收到新的连接请求，`async_accept`接受它，将传递给它的套接字引用设置为新连接的服务器端套接字，并恢复`accept_connections`协程。`accept_connections`协程将`handle_connection`协程添加到`io_service`队列中，用于处理此特定连接的 I/O（第 56-57 行）。在下一次循环迭代中，它再次等待新的传入连接。

`handle_connection`协程除了`yield_context`之外，还接受一个包装在`shared_ptr`中的 TCP 套接字作为参数。`accept_connections`协程创建此套接字，并将其包装在`shared_ptr`中传递给`handle_connection`。`handle_connection`函数使用`async_read`接收客户端发送的任何数据（第 23-24 行）。如果接收成功，它会发送一个响应字符串`Hello from server`，然后使用长度为 2 的缓冲区序列回显接收到的数据（第 28-30 行）。

#### 没有协程的异步 TCP 服务器

现在我们来看如何编写一个没有协程的异步 TCP 服务器。这涉及处理程序之间更复杂的握手，因此，我们希望将代码拆分成适当的类。我们在两个单独的头文件中定义了两个类。`TCPAsyncServer`类（清单 11.23）表示监听传入连接的服务器实例。它放在`asyncsvr.hpp`头文件中。`TCPAsyncConnection`类（清单 11.25）表示单个连接的处理上下文。它放在`asynconn.hpp`头文件中。

`TCPAsyncServer`为每个新的传入连接创建一个新的`TCPAsyncConnection`实例。`TCPAsyncConnection`实例从客户端读取传入数据，并向客户端发送消息，直到客户端关闭与服务器的连接。

要启动服务器，您需要创建一个`TCPAsyncServer`实例，传递`io_service`实例和端口号，然后调用`io_service`的`run`成员函数来开始处理新连接：

**清单 11.23：异步 TCP 服务器（asyncsvr.hpp）**

```cpp
 1 #ifndef ASYNCSVR_HPP
 2 #define ASYNCSVR_HPP
 3 #include <boost/asio.hpp>
 4 #include <boost/shared_ptr.hpp>
 5 #include <boost/make_shared.hpp>
 6 #include <iostream>
 7 #include "asynconn.hpp"
 8
 9 namespace asio = boost::asio;
10 namespace sys = boost::system;
11 typedef boost::shared_ptr<TCPAsyncConnection>
12               TCPAsyncConnectionPtr;
13
14 class TCPAsyncServer {
15 public:
16   TCPAsyncServer(asio::io_service& service, unsigned short p)
17           : acceptor(service,
18                     asio::ip::tcp::endpoint(
19                           asio::ip::tcp::v4(), p)) {
20     waitForConnection();
21   }
22
23   void waitForConnection() {
24     TCPAsyncConnectionPtr connectionPtr = boost::make_shared
25           <TCPAsyncConnection>(acceptor.get_io_service());
26     acceptor.async_accept(connectionPtr->getSocket(),
27           this, connectionPtr {
28             if (ec) {
29               std::cerr << "Failed to accept connection: "
30                         << ec.message() << "\n";
31             } else {
32               connectionPtr->waitForReceive();
33               waitForConnection();
34             }
35           });
36   }
37
38 private:
39   asio::ip::tcp::acceptor acceptor;
40 };
41
42 #endif /* ASYNCSVR_HPP */
```

`TCPAsyncServer`类具有一个`boost::asio::ip::tcp::acceptor`类型的接受者成员变量，用于监听和接受传入连接（第 39 行）。构造函数使用未指定的 IPv4 地址和特定端口初始化接受者（第 17-19 行），然后调用`waitForConnection`成员函数（第 20 行）。

`waitForConnection`函数创建了一个新的`TCPAsyncConnection`实例，将其包装在名为`connectionPtr`的`shared_ptr`中（第 24-25 行），以处理来自客户端的每个新连接。我们已经包含了我们自己的头文件`asynconn.hpp`来访问`TCPAsyncConnection`的定义（第 7 行），我们很快就会看到。然后调用 acceptor 的`async_accept`成员函数来监听新的传入连接并接受它们（第 26-27 行）。我们传递给`async_accept`一个对`TCPAsyncConnection`的`tcp::socket`对象的非 const 引用，以及一个在每次建立新连接时调用的完成处理程序（第 27-35 行）。这是一个异步调用，会立即返回。但每次建立新连接时，套接字引用都会设置为用于服务该连接的服务器端套接字，并调用完成处理程序。

`async_accept`的完成处理程序被编写为 lambda，并捕获指向`TCPAsyncServer`实例的`this`指针和`connectionPtr`（第 27 行）。这允许 lambda 在`TCPAsyncServer`实例和为该特定连接提供服务的`TCPAsyncConnection`实例上调用成员函数。

### 提示

lambda 表达式生成一个函数对象，并将捕获的`connectionPtr`复制到其中的一个成员。由于`connectionPtr`是一个`shared_ptr`，在此过程中它的引用计数会增加。`async_accept`函数将此函数对象推送到`io_service`的任务处理程序队列中，因此`TCPAsyncConnection`的底层实例会在`waitForConnection`返回后继续存在。

在连接建立时，当调用完成处理程序时，它会执行两件事。如果没有错误，它会通过在`TCPAsyncConnection`对象上调用`waitForReceive`函数（第 32 行）来启动新连接上的 I/O。然后通过调用`TCPAsyncServer`对象上的`waitForConnection`（通过捕获的`this`指针）来重新等待下一个连接（第 33 行）。如果出现错误，它会打印一条消息（第 29-30 行）。`waitForConnection`调用是异步的，我们很快就会发现`waitForReceive`调用也是异步的，因为两者都调用了异步 Asio 函数。处理程序返回后，服务器将继续处理现有连接上的 I/O 或接受新连接：

**清单 11.24：运行异步服务器**

```cpp
 1 #include <boost/asio.hpp>
 2 #include <boost/thread.hpp>
 3 #include <boost/shared_ptr.hpp>
 4 #include <iostream>
 5 #include "asyncsvr.hpp"
 6 #define MAXBUF 1024
 7 namespace asio = boost::asio;
 8
 9 int main() {
10   try {
11     asio::io_service service;
12     TCPAsyncServer server(service, 56000);
13     service.run();
14   } catch (std::exception& e) {
15     std::cout << e.what() << '\n';
16   }
17 }
```

要运行服务器，我们只需用`io_service`和端口号实例化它（第 12 行），然后在`io_service`上调用`run`方法（第 13 行）。我们正在构建的服务器将是线程安全的，因此我们也可以从线程池中的每个线程调用`run`，以在处理传入连接时引入一些并发。现在我们将看到如何处理每个连接上的 I/O：

**清单 11.25：每个连接的 I/O 处理程序类（asynconn.hpp）**

```cpp
 1 #ifndef ASYNCONN_HPP
 2 #define ASYNCONN_HPP
 3
 4 #include <boost/asio.hpp>
 5 #include <boost/thread.hpp>
 6 #include <boost/shared_ptr.hpp>
 7 #include <iostream>
 8 #define MAXBUF 1024
 9
10 namespace asio = boost::asio;
11 namespace sys = boost::system;
12
13 class TCPAsyncConnection
14   : public boost::enable_shared_from_this<TCPAsyncConnection> {
15 public:
16   TCPAsyncConnection(asio::io_service& service) :
17       socket(service) {}
18
19   asio::ip::tcp::socket& getSocket() {
20     return socket;
21   }
22
23   void waitForReceive() {
24     auto thisPtr = shared_from_this();
25     async_read(socket, asio::buffer(buf, sizeof(buf)),
26         thisPtr {
27           if (!ec || ec == asio::error::eof) {
28             thisPtr->startSend();
29             thisPtr->buf[sz] = '\0'; 
30             std::cout << thisPtr->buf << '\n';
31             
32             if (!ec) { thisPtr->waitForReceive(); }
33           } else {
34             std::cerr << "Error receiving data from "
35                     "client: " << ec.message() << "\n";
36           }
37         });
38   }
39
40   void startSend() {
41     const char *msg = "Hello from server";
42     auto thisPtr = shared_from_this();
43     async_write(socket, asio::buffer(msg, strlen(msg)),
44         thisPtr {
45           if (ec) {
46             if (ec == asio::error::eof) {
47                thisPtr->socket.close();
48             }
49             std::cerr << "Failed to send response to "
50                     "client: " << ec.message() << '\n';
51           }
52         });
53   }
54
55 private:
56   asio::ip::tcp::socket socket;
57   char buf[MAXBUF];
58 };
59
60 #endif /* ASYNCONN_HPP */
```

我们在 11.23 清单中看到了如何创建`TCPAsyncConnection`的实例，并将其包装在`shared_ptr`中，以处理每个新连接，并通过调用`waitForReceive`成员函数来启动 I/O。现在让我们来了解它的实现。`TCPAsyncConnection`有两个用于在连接上执行异步 I/O 的公共成员：`waitForReceive`用于执行异步接收（第 23 行），`startSend`用于执行异步发送（第 40 行）。

`waitForReceive`函数通过在套接字上调用`async_read`函数（第 25 行）来启动接收。数据被接收到`buf`成员中（第 57 行）。此调用的完成处理程序（第 26-37 行）在数据完全接收时被调用。如果没有错误，它调用`startSend`，它异步地向客户端发送一条消息（第 28 行），然后再次调用`waitForReceive`，前提是之前的接收没有遇到文件结尾（第 32 行）。因此，只要没有读取错误，服务器就会继续等待在连接上读取更多数据。如果出现错误，它会打印诊断消息（第 34-35 行）。

`startSend`函数使用`async_write`函数向客户端发送文本`Hello from server`。它的处理程序在成功时不执行任何操作，但在失败时打印诊断消息（第 49-50 行）。对于 EOF 写入错误，它关闭套接字（第 47 行）。

#### TCPAsyncConnection 的生命周期

每个`TCPAsyncConnection`实例需要在客户端保持连接到服务器的时间内存活。这使得将这个对象的范围绑定到服务器中的任何函数变得困难。这就是我们在`shared_ptr`中创建`TCPAsyncConnection`对象的原因，然后在处理程序 lambda 中捕获它。`TCPAsyncConnection`用于在连接上执行 I/O 的成员函数`waitForReceive`和`startSend`都是异步的。因此，它们在返回之前将处理程序推入`io_service`的任务队列。这些处理程序捕获了`TCPAsyncConnection`的`shared_ptr`包装实例，以保持实例在调用之间的存活状态。

为了使处理程序能够从`waitForReceive`和`startSend`中访问`TCPAsyncConnection`对象的`shared_ptr`包装实例，需要这些`TCPAsyncConnection`的成员函数能够访问它们被调用的`shared_ptr`包装实例。我们在第三章中学到的*enable shared from this*习惯用法，*内存管理和异常安全*，是为这种目的量身定制的。这就是我们将`TCPAsyncConnection`从`enable_shared_from_this<TCPAsyncConnection>`派生的原因。由于这个原因，`TCPAsyncConnection`继承了`shared_from_this`成员函数，它返回我们需要的`shared_ptr`包装实例。这意味着`TCPAsyncConnection`应该始终动态分配，并用`shared_ptr`包装，否则会导致未定义的行为。

这就是我们在`waitForReceive`（第 24 行）和`startSend`（第 42 行）中都调用`shared_from_this`的原因，它被各自的处理程序捕获（第 26 行，44 行）。只要`waitForReceive`成员函数从`async_read`（第 32 行）的完成处理程序中被调用，`TCPAsyncConnection`实例就会存活。如果在接收中遇到错误，要么是因为远程端点关闭了连接，要么是因为其他原因，那么这个循环就会中断。包装`TCPAsyncConnection`对象的`shared_ptr`不再被任何处理程序捕获，并且在作用域结束时被销毁，关闭连接。

#### 性能和并发性

请注意，TCP 异步服务器的两种实现，使用和不使用协程，都是单线程的。然而，在任何实现中都没有线程安全问题，因此我们也可以使用线程池，每个线程都会在`io_service`上调用`run`。

#### 控制流的反转

编写异步系统的最大困难在于控制流的反转。要编写同步服务器的代码，我们知道必须按以下顺序调用操作：

1.  在接收器上调用`accept`。

1.  在套接字上调用`read`。

1.  在套接字上调用`write`。

我们知道`accept`仅在连接建立后才返回，因此可以安全地调用`read`。此外，`read`仅在读取所请求的字节数或遇到文件结束后才返回。因此，可以安全地调用`write`。与异步模型相比，这使得编写代码变得非常容易，但引入了等待，影响了我们处理其他等待连接的能力，同时我们的请求正在被处理。

我们通过异步 I/O 消除了等待，但在使用处理程序链接时失去了模型的简单性。由于我们无法确定地告诉异步 I/O 操作何时完成，因此我们要求`io_service`在我们的请求完成时运行特定的处理程序。我们仍然知道在之后执行哪个操作，但不再知道何时。因此，我们告诉`io_service`要运行*什么*，它使用来自操作系统的适当通知来知道*何时*运行它们。这种模型的最大挑战是在处理程序之间维护对象状态和管理对象生命周期。

通过允许将异步 I/O 操作的序列写入单个协程来消除这种*控制流反转*，该协程被*挂起*而不是等待异步操作完成，并在操作完成时*恢复*。这允许无等待逻辑，而不会引入处理程序链接的固有复杂性。

### 提示

在编写异步服务器时，始终优先使用协程而不是处理程序链接。

# 自测问题

对于多项选择题，选择所有适用的选项：

1.  `io_service::dispatch`和`io_service::post`之间的区别是什么？

a. `dispatch`立即返回，而`post`在返回之前运行处理程序

b. `post`立即返回，而`dispatch`如果可以在当前线程上运行处理程序，或者它的行为类似于 post

c. `post`是线程安全的，而`dispatch`不是

d. `post`立即返回，而`dispatch`运行处理程序

1.  当处理程序在分派时抛出异常会发生什么？

a. 这是未定义行为

b. 它通过调用`std::terminate`终止程序

c. 在分派处理程序的`io_service`上调用 run 将抛出异常。

d. `io_service`被停止

1.  未指定地址 0.0.0.0（IPv4）或::/1（IPv6）的作用是什么？

a. 它用于与系统上的本地服务通信

b. 发送到此地址的数据包将被回显到发送方

c. 它用于向网络中的所有连接的主机进行广播

d. 它用于绑定到所有可用接口，而无需知道地址

1.  以下关于 TCP 的哪些陈述是正确的？

a. TCP 比 UDP 更快

b. TCP 检测数据损坏但不检测数据丢失

c. TCP 比 UDP 更可靠

d. TCP 重新传输丢失或损坏的数据

1.  当我们说特定函数，例如`async_read`是异步时，我们是什么意思？

a. 在请求的操作完成之前，该函数会返回

b. 该函数在不同的线程上启动操作，并立即返回

c. 请求的操作被排队等待由同一线程或另一个线程处理

d. 如果可以立即执行操作，则该函数执行该操作，否则返回错误

1.  我们如何确保在调用异步函数之前创建的对象仍然可以在处理程序中使用？

a. 将对象设为全局。

b. 在处理程序中复制/捕获包装在`shared_ptr`中的对象。

c. 动态分配对象并将其包装在`shared_ptr`中。

d. 将对象设为类的成员。

# 总结

Asio 是一个设计良好的库，可用于编写快速、灵活的网络服务器，利用系统上可用的最佳异步 I/O 机制。它是一个不断发展的库，是提议在未来的 C++标准修订版中添加网络库的技术规范的基础。

在这一章中，我们学习了如何使用 Boost Asio 库作为任务队列管理器，并利用 Asio 的 TCP 和 UDP 接口编写可以在网络上通信的程序。使用 Boost Asio，我们能够突出显示网络编程的一些一般性问题，针对大量并发连接的扩展挑战，以及异步 I/O 的优势和复杂性。特别是，我们看到使用 stackful 协程相对于旧模型的处理程序链，使得编写异步服务器变得轻而易举。虽然我们没有涵盖 stackless 协程、ICMP 协议和串口通信等内容，但本章涵盖的主题应该为您提供了理解这些领域的坚实基础。

# 参考资料

+   *Thinking Asynchronously in C++* (博客), *Christopher Kohlhoff*: [`blog.think-async.com/`](http://blog.think-async.com/)

+   *Networking Library Proposal*, *Christopher Kohlhoff*: [`www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4332.html`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4332.html)
