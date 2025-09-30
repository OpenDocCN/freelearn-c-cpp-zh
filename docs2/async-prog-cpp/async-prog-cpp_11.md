# 11

# 异步软件的日志和调试

没有办法确保软件产品完全没有错误，所以有时会出现错误。这时，日志和调试是必不可少的。

日志和调试对于识别和诊断软件系统中的问题至关重要。它们提供了对代码运行时行为的可见性，帮助开发者追踪错误、监控性能以及理解执行流程。通过有效地使用日志和调试，开发者可以检测到错误、解决意外行为，并提高整体系统的稳定性和可维护性。

在编写本章时，我们假设你已经熟悉使用调试器调试 C++程序，并了解一些基本的调试器命令和术语，例如断点、监视器、帧或堆栈跟踪。为了复习这些知识，你可以参考章节末尾的*进一步阅读*部分提供的参考资料。

在本章中，我们将涵盖以下主要主题：

+   如何使用日志来查找错误

+   如何调试异步软件

# 技术要求

对于本章，我们需要安装第三方库来编译示例。

要编译日志部分中的示例，需要安装**spdlog**和**{fmt}**库。请检查它们的文档（**spdlog**的文档可在[`github.com/gabime/spdlog`](https://github.com/gabime/spdlog)找到，**{fmt}**的文档可在[`github.com/fmtlib/fmt`](https://github.com/fmtlib/fmt)找到），并按照适合您平台的安装步骤进行操作。

一些示例需要支持 C++20 的编译器。因此，请检查*第三章*中的技术要求部分，其中提供了一些关于如何安装 GCC 13 和 Clang 8 编译器的指导。

你可以在以下 GitHub 仓库中找到所有完整的代码：

[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)

本章的示例位于**Chapter_11**文件夹下。所有源代码文件都可以使用以下命令使用 CMake 编译：

```cpp
$ cmake . && cmake —build .
```

可执行二进制文件将在**bin**目录下生成。

# 如何使用日志来查找错误

让我们从理解软件程序在执行时做了什么的简单但有用方法开始——日志。

**日志**是记录程序中发生的事件的过程，通过使用消息记录程序如何执行，跟踪其流程，并帮助识别问题和错误。

大多数基于 Unix 的日志系统使用由 Eric Altman 在 1980 年作为 Sendmail 项目一部分创建的标准协议**syslog**。这个标准协议定义了生成日志消息的软件、存储它们的系统和报告和分析这些日志事件的软件之间的边界。

每条日志消息都包含一个设施代码和严重级别。设施代码标识了产生特定日志消息的系统类型（用户级、内核、系统、网络等），严重级别描述了系统的状态，表明处理特定问题的紧迫性，严重级别包括紧急、警报、关键、错误、警告、通知、信息和调试。

大多数日志系统或日志记录器都提供了各种日志消息的目的地或接收器：控制台、可以稍后打开和分析的文件、远程 syslog 服务器或中继，以及其他目的地。

在调试器无法使用的地方，日志非常有用，正如我们稍后将看到的，特别是在分布式、多线程、实时、科学或以事件为中心的应用程序中，使用调试器检查数据或跟踪程序流程可能变得是一项繁琐的任务。

日志库通常还提供一个线程安全的单例类，允许多线程和异步写入日志文件，有助于日志轮转，通过动态创建新文件而不丢失日志事件来避免大型日志文件，并添加时间戳，以便更好地跟踪日志事件发生的时间。

而不是实现我们自己的多线程日志系统，更好的方法是使用一些经过良好测试和文档化的生产就绪库。

## 如何选择第三方库

在将日志库（或任何其他库）集成到我们的软件之前，我们需要调查以下问题，以避免未来出现的问题：

+   **支持**：库是否定期更新和升级？是否有社区或活跃的生态系统围绕该库，可以帮助解决可能出现的任何问题？社区是否对使用该库感到满意？

+   **质量**：是否存在公开的缺陷报告系统？缺陷报告是否得到及时处理，提供解决方案并修复库中的缺陷？它是否支持最近的编译器版本并支持最新的 C++特性？

+   **安全性**：库或其任何依赖库是否有已报告的漏洞？

+   **许可证**：库的许可证是否与我们的开发和产品需求一致？成本是否可承受？

对于复杂系统，考虑集中式系统来收集和生成日志报告或仪表板可能是值得的，例如 **Sentry** ([`sentry.io`](https://sentry.io)) 或 **Logstash** ([`www.elastic.co/logstash`](https://www.elastic.co/logstash))，它们可以收集、解析和转换日志，并且可以与其他工具集成，如 **Graylog** ([`graylog.org`](https://graylog.org))、**Grafana** ([`grafana.com`](https://grafana.com)) 或 **Kibana** ([`www.elastic.co/kibana`](https://www.elastic.co/kibana))。

下一个部分将描述一些有趣的日志库。

## 一些相关的日志库

市场上有许多日志库，每个库都覆盖了一些特定的软件需求。根据程序约束和需求，以下库中的一些可能比其他库更适合。

在 *第九章* 中，我们探讨了 **Boost.Asio**。Boost 还提供了另一个库，**Boost.Log**（[`github.com/boostorg/log`](https://github.com/boostorg/log)），这是一个强大且可配置的日志库。

Google 也提供了许多开源库，包括 **glog**，Google 日志库（[`github.com/google/glog`](https://github.com/google/glog)），这是一个 C++14 库，提供了 C++ 风格的流 API 和辅助宏。

如果开发者熟悉 Java，一个不错的选择可能是基于 **Log4j**（[`logging.apache.org/log4j`](https://logging.apache.org/log4j)）的 Apache **Log4cxx**（[`logging.apache.org/log4cxx`](https://logging.apache.org/log4cxx)），这是一个多才多艺、工业级、Java 日志框架。

值得考虑的其他日志库如下：

+   **spdlog**（[`github.com/gabime/spdlog`](https://github.com/gabime/spdlog)）是一个有趣的日志库，我们可以与 **{fmt}** 库一起使用。此外，程序可以从启动时开始记录消息并将它们排队，甚至在指定日志输出文件名之前。

+   **Quill**（[`github.com/odygrd/quill`](https://github.com/odygrd/quill)）是一个异步低延迟的 C++ 日志库。

+   **NanoLog**（[`github.com/PlatformLab/NanoLog`](https://github.com/PlatformLab/NanoLog)）是一个具有类似 **printf** API 的纳秒级日志系统。

+   **lwlog**（[`github.com/ChristianPanov/lwlog`](https://github.com/ChristianPanov/lwlog)）是一个惊人的快速异步 C++17 日志库。

+   **XTR**（[`github.com/choll/xtr`](https://github.com/choll/xtr)）是一个适用于低延迟和实时环境的快速便捷的 C++ 日志库。

+   **Reckless**（[`github.com/mattiasflodin/reckless`](https://github.com/mattiasflodin/reckless)）是一个低延迟和高吞吐量的日志库。

+   **uberlog**（[`github.com/IMQS/uberlog`](https://github.com/IMQS/uberlog)）是一个跨平台和多进程的 C++ 日志系统。

+   **Easylogging++**（[`github.com/abumq/easyloggingpp`](https://github.com/abumq/easyloggingpp)）是一个单头文件 C++ 日志库，具有编写自定义存储和跟踪性能的能力。

+   **tracetool**（[`github.com/froglogic/tracetool`](https://github.com/froglogic/tracetool)）是一个日志和跟踪共享库。

作为指导，根据要开发的系统，我们可能会选择以下库之一：

+   **对于低延迟或实时系统**：Quill、XTR 或 Reckless

+   **对于纳秒级性能的日志**：NanoLog

+   **对于异步日志**：Quill 或 **lwlog**

+   **对于跨平台、多进程 **应用程序**：**uberlog**

+   **对于简单灵活的日志**：Easylogging++ 或 glog

+   **对于熟悉 Java **日志**：Log4cxx

所有库都有优点，但也存在需要在使用前调查的缺点。以下表格总结了这些要点：

| **库** | **优点** | **缺点** |
| --- | --- | --- |
| **spdlog** | 简单集成，性能导向，可定制 | 缺乏一些针对极低延迟需求的高级功能 |
| Quill | 在低延迟系统中性能高 | 相比于更简单的同步日志记录器，设置更复杂 |
| NanoLog | 在速度上表现最佳，针对性能优化 | 功能有限；适用于专用用例 |
| **lwlog** | 轻量级，适合快速集成 | 相比于其他替代方案，成熟度和功能较少 |
| XTR | 非常高效，用户界面友好 | 更适合特定的实时应用 |
| Reckless | 高度优化吞吐量和低延迟 | 相比于更通用的日志记录器，灵活性有限 |
| **uberlog** | 适用于多进程和分布式系统 | 不如专门的低延迟日志记录器快 |
| Easylogging++ | 使用简单，可自定义输出目标 | 性能优化不如一些其他库 |
| **tracetool** | 将日志和跟踪结合在一个库中 | 不专注于低延迟或高吞吐量 |
| Boost.Log | 通用性强，与 Boost 库集成良好 | 复杂度较高；对于简单的日志需求可能过于复杂 |
| glog | 使用简单，适合需要简单 API 的项目 | 对于高级定制功能不如其他库丰富 |
| Log4cxx | 稳定，经过时间考验，工业级日志 | 设置较为复杂，特别是对于小型项目 |

表 11.1：各种库的优点和缺点

请访问日志库的网站以更好地了解它们提供的功能，并比较它们之间的性能。

由于 **spdlog** 是 GitHub 上被分叉和星标最多的 C++ 日志库仓库，在下一节中，我们将实现一个使用此库来捕获竞态条件的示例。

## 记录死锁 - 示例

在实现此示例之前，我们需要安装 **spdlog** 和 **{fmt}** 库。**{fmt}** (https://github.com/fmtlib/fmt) 是一个开源格式化库，提供了一种快速且安全的 C++ IOStreams 替代方案。

请检查它们的文档，并根据您的平台遵循安装步骤。

让我们实现一个发生死锁的示例。正如我们在 *第四章* 中所学，当两个或更多线程需要获取多个互斥锁以执行其工作时会发生死锁。如果互斥锁不是以相同的顺序获取，一个线程可以获取一个互斥锁并永远等待另一个线程获取的互斥锁。

在这个例子中，两个线程需要获取两个互斥锁，**mtx1** 和 **mtx2**，以增加 **counter1** 和 **counter2** 计数器的值并交换它们。由于线程以不同的顺序获取互斥锁，可能会发生死锁。

让我们先包含所需的库：

```cpp
#include <fmt/core.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
using namespace std::chrono_literals;
```

在 **main()** 函数中，我们定义了计数器和互斥锁：

```cpp
uint32_t counter1{};
std::mutex mtx1;
uint32_t counter2{};
std::mutex mtx2;
```

在生成线程之前，让我们设置一个**多目标记录器**，这是一种可以将日志消息写入控制台和日志文件的记录器。我们还将设置其日志级别为调试，使记录器发布所有严重性级别大于调试的日志消息，每行日志的格式包括时间戳、线程标识符、日志级别和日志消息：

```cpp
auto console_sink = std::make_shared<
         spdlog::sinks::stdout_color_sink_mt>();
console_sink->set_level(spdlog::level::debug);
auto file_sink = std::make_shared<
         spdlog::sinks::basic_file_sink_mt>("logging.log",
                                            true);
file_sink->set_level(spdlog::level::info);
spdlog::logger logger("multi_sink",
         {console_sink, file_sink});
logger.set_pattern(
         "%Y-%m-%d %H:%M:%S.%f - Thread %t [%l] : %v");
logger.set_level(spdlog::level::debug);
```

我们还声明了一个**increase_and_swap** lambda 函数，该函数增加两个计数器的值并交换它们：

```cpp
auto increase_and_swap = [&]() {
    logger.info("Incrementing both counters...");
    counter1++;
    counter2++;
    logger.info("Swapping counters...");
    std::swap(counter1, counter2);
};
```

两个工作 lambda 函数**worker1**和**worker2**获取两个互斥锁，并在退出前调用**increase_and_swap()**。由于使用了锁保护（**std::lock_guard**）对象，因此在销毁工作 lambda 函数时释放互斥锁：

```cpp
auto worker1 = [&]() {
    logger.debug("Entering worker1");
    logger.info("Locking mtx1...");
    std::lock_guard<std::mutex> lock1(mtx1);
    logger.info("Mutex mtx1 locked");
    std::this_thread::sleep_for(100ms);
    logger.info("Locking mtx2...");
    std::lock_guard<std::mutex> lock2(mtx2);
    logger.info("Mutex mtx2 locked");
    increase_and_swap();
    logger.debug("Leaving worker1");
};
auto worker2 = [&]() {
    logger.debug("Entering worker2");
    logger.info("Locking mtx2...");
    std::lock_guard<std::mutex> lock2(mtx2);
    logger.info("Mutex mtx2 locked");
    std::this_thread::sleep_for(100ms);
    logger.info("Locking mtx1...");
    std::lock_guard<std::mutex> lock1(mtx1);
    logger.info("Mutex mtx1 locked");
    increase_and_swap();
    logger.debug("Leaving worker2");
};
logger.debug("Starting main function...");
std::thread t1(worker1);
std::thread t2(worker2);
t1.join();
t2.join();
```

两个工作 lambda 函数**worker1**和**worker2**相似，但有一个小差异：**worker1**先获取**mutex1**然后获取**mutex2**，而**worker2**则相反，先获取**mutex2**然后获取**mutex1**。在获取两个互斥锁之间有一个睡眠期，以便其他线程获取其互斥锁，因此，这会导致死锁，因为**worker1**将获取**mutex1**而**worker2**将获取**mutex2**。

然后，在睡眠之后，**worker1**将尝试获取**mutex2**，而**worker2**将尝试获取**mutex1**，但它们都不会成功，永远在死锁中阻塞。

以下是在运行此代码时的输出：

```cpp
2024-09-04 23:39:54.484005 - Thread 38984 [debug] : Starting main function...
2024-09-04 23:39:54.484106 - Thread 38985 [debug] : Entering worker1
2024-09-04 23:39:54.484116 - Thread 38985 [info] : Locking mtx1...
2024-09-04 23:39:54.484136 - Thread 38986 [debug] : Entering worker2
2024-09-04 23:39:54.484151 - Thread 38986 [info] : Locking mtx2...
2024-09-04 23:39:54.484160 - Thread 38986 [info] : Mutex mtx2 locked
2024-09-04 23:39:54.484146 - Thread 38985 [info] : Mutex mtx1 locked
2024-09-04 23:39:54.584250 - Thread 38986 [info] : Locking mtx1...
2024-09-04 23:39:54.584255 - Thread 38985 [info] : Locking mtx2...
```

在检查日志时，首先要注意的症状是程序从未完成，因此很可能处于死锁状态。

从记录器输出中，我们可以看到**t1**（线程**38985**）正在运行**worker1**，而**t2**（线程**38986**）正在运行**worker2**。一旦**t1**进入**worker1**，它就获取**mtx1**。然而，**mtx2**互斥锁是由**t2**获取的，因为**worker2**一启动就获取了。然后，两个线程等待 100 毫秒并尝试获取另一个互斥锁，但都没有成功，程序保持阻塞。

记录在生产系统中是必不可少的，但如果过度使用，则会对性能造成一些惩罚，并且大多数时候需要人工干预来调查问题。作为日志详细程度和性能惩罚之间的折衷方案，一个人可能会选择实现不同的日志级别，在正常操作期间仅记录主要事件，同时仍然保留在需要时提供极其详细日志的能力。在开发周期早期自动检测代码中的错误的一种更自动化的方法是使用测试和代码清理器，我们将在下一章中学习这些内容。

并非所有错误都可以检测到，因此通常使用调试器是跟踪和修复软件中的错误的方法。让我们接下来学习如何调试多线程和异步代码。

# 如何调试异步软件

**调试**是查找和修复计算机程序中的错误的过程。

在本节中，我们将探讨几种调试多线程和异步软件的技术。您必须具备一些使用调试器的先验知识，例如 **GDB**（GNU 项目调试器）或 **LLDB**（LLVM 低位调试器），以及调试过程的术语，如断点、观察者、回溯、帧和崩溃报告。

GDB 和 LLDB 都是优秀的调试器，它们的大多数命令都是相同的，只有少数命令不同。如果程序是在 macOS 上调试或针对大型代码库，LLDB 可能更受欢迎。另一方面，GDB 拥有稳定的传统，许多开发者都熟悉它，并支持更广泛的架构和平台。在本节中，我们将使用 GDB 15.1，因为它属于 GNU 框架，并且被设计为与 **g++** 编译器协同工作，但随后显示的大多数命令也可以在用 **clang++** 编译的程序上使用 LLDB 进行调试。

由于一些处理多线程和异步代码的调试器功能仍在开发中，请始终更新调试器到最新版本，以包括最新的功能和修复。

## 一些有用的 GDB 命令

让我们从一些在调试任何类型程序时都很有用的 GDB 命令开始，并为下一节打下基础。

在调试程序时，我们可以启动调试器并将程序作为参数传递。程序可能需要的额外参数可以通过 **--** **args** 选项传递：

```cpp
$ gdb <program> --args <args>
```

或者，我们可以通过使用其 **进程** **标识符**（**PID**）来将调试器附加到正在运行的程序：

```cpp
$ gdb –p <PID>
```

一旦进入调试器，我们可以运行程序（使用 **run** 命令）或启动它（使用 **start** 命令）。运行意味着程序执行直到达到断点或完成。**start** 仅在 **main()** 函数的开始处放置一个临时断点并运行程序，在程序开始处停止执行。

例如，如果我们想调试已经崩溃的程序，我们可以使用由崩溃生成的 core dump 文件，该文件可能存储在系统中的特定位置（通常在 Linux 系统上是 **/var/lib/apport/coredump/**，但请通过访问官方文档来检查您系统中的确切位置）。此外，请注意，通常 core dump 默认是禁用的，需要运行 **ulimit -c unlimited** 命令，在程序崩溃之前和同一 shell 中执行。如果处理的是特别大的程序或系统磁盘空间不足，可以将 **unlimited** 参数更改为某个任意限制。

在生成 **coredump** 文件后，只需将其复制到程序二进制文件所在的目录，并使用以下命令：

```cpp
$ gdb <program> <coredump>
```

注意，所有二进制文件都必须有调试符号，因此必须使用**–g**选项编译。在生产系统中，发布二进制文件通常移除了符号并存储在单独的文件中。有 GDB 命令可以包含这些符号，以及命令行工具可以检查它们，但这个主题超出了本书的范围。

一旦调试器开始运行，我们可以使用 GDB 命令在代码中导航或检查变量。一些有用的命令如下：

+   **info args**：这会显示用于调用当前函数的参数信息。

+   **info locals**：这会显示当前作用域中的局部变量。

+   **whatis**：这会显示给定变量或表达式的类型。

+   **return**：这会从当前函数返回，而不执行其余的指令。可以指定返回值。

+   **backtrace**：这会列出当前调用栈中的所有栈帧。

+   **frame**：这允许你切换到特定的栈帧。

+   **up**，**down**：这会在调用栈中移动，向当前函数的调用者（**up**）或被调用者（**down**）移动。

+   **print**：这会评估并显示一个表达式的值，该表达式可以是变量名、类成员、指向内存区域的指针或直接是内存地址。我们还可以定义漂亮的打印器来显示我们自己的类。

让我们以调试程序最基本但也是最常用的技术之一来结束本节。这种技术被称为**printf**。每个开发者都使用过**printf**或替代命令来打印变量内容，以便在代码路径上的战略位置显示其内容。在 GDB 中，**dprintf**命令有助于设置在遇到断点时打印信息的**printf**样式断点，而不会停止程序执行。这样，我们可以在调试程序时使用打印语句，而无需修改代码、重新编译和重启程序。

其语法如下：

```cpp
$ dprintf <location>, <format>, <args>
```

例如，如果我们想在第 25 行设置一个**printf**语句来打印**x**变量的内容，但只有当其值大于**5**时，这是命令：

```cpp
$ dprintf 25, "x = %d\n", x if x > 5
```

现在我们已经建立了一些基础，让我们从调试一个多线程程序开始。

## 调试多线程程序

这里显示的示例永远不会结束，因为会发生死锁，因为不同的线程以不同的顺序锁定两个互斥锁，正如在本章介绍日志时已经解释过的：

```cpp
#include <chrono>
#include <mutex>
#include <thread>
using namespace std::chrono_literals;
int main() {
    std::mutex mtx1, mtx2;
    std::thread t1([&]() {
        std::lock_guard lock1(mtx1);
        std::this_thread::sleep_for(100ms);
        std::lock_guard lock2(mtx2);
    });
    std::thread t2([&]() {
        std::lock_guard lock2(mtx2);
        std::this_thread::sleep_for(100ms);
        std::lock_guard lock1(mtx1);
    });
    t1.join();
    t2.join();
    return 0;
}
```

首先，让我们使用**g++**编译这个示例，并添加调试符号（**–g**选项）以及不允许代码优化（**–O0**选项），防止编译器重构二进制代码，使调试器更难通过使用**--fno-omit-frame-pointer**选项找到并显示相关信息。

以下命令编译**test.cpp**源文件并生成**test**二进制文件。我们还可以使用**clang++**以相同的选项：

```cpp
$ g++ -o test –g -O0 --fno-omit-frame-pointer test.cpp
```

如果我们运行生成的程序，它将永远不会结束：

```cpp
$ ./test
```

要调试一个正在运行的程序，我们首先使用 **ps** Unix 命令检索其 PID：

```cpp
$ ps aux | grep test
```

然后，通过提供 **pid** 来附加调试器并开始调试程序：

```cpp
$ gdb –p <pid>
```

假设调试器以以下消息开始：

```cpp
ptrace: Operation not permitted.
```

然后，只需运行以下命令：

```cpp
$ sudo sysctl -w kernel.yama.ptrace_scope=0
```

一旦 GDB 正确启动，你将能够在其提示符中输入命令。

我们可以执行的第一条命令是下一个，以检查正在运行的线程：

```cpp
(gdb) info threads
  Id   Target Id                                Frame
* 1    Thread 0x79d1f3883740 (LWP 14428) "test" 0x000079d1f3298d61 in __futex_abstimed_wait_common64 (private=128, cancel=true, abstime=0x0, op=265, expected=14429, futex_word=0x79d1f3000990)
    at ./nptl/futex-internal.c:57
  2    Thread 0x79d1f26006c0 (LWP 14430) "test" futex_wait (private=0, expected=2, futex_word=0x7fff5e406b00) at ../sysdeps/nptl/futex-internal.h:146
  3    Thread 0x79d1f30006c0 (LWP 14429) "test" futex_wait (private=0, expected=2, futex_word=0x7fff5e406b30) at ../sysdeps/nptl/futex-internal.h:146
```

输出显示，具有 GDB 标识符 **1** 的 **0x79d1f3883740** 线程是当前线程。如果有许多线程，而我们只对特定的子集感兴趣，比如说线程 1 和 3，我们可以使用以下命令仅显示那些线程的信息：

```cpp
(gdb) info thread 1 3
```

运行一个 GDB 命令将影响当前线程。例如，运行 **bt** 命令将显示线程 1 的回溯（输出已简化）：

```cpp
(gdb) bt
#0  0x000079d1f3298d61 in __futex_abstimed_wait_common64 (private=128, cancel=true, abstime=0x0, op=265, expected=14429, futex_word=0x79d1f3000990) at ./nptl/futex-internal.c:57
#5  0x000061cbaf1174fd in main () at 11x18-debug_deadlock.cpp:22
```

要切换到另一个线程，例如线程 2，我们可以使用 **thread** 命令：

```cpp
(gdb) thread 2
[Switching to thread 2 (Thread 0x79d1f26006c0 (LWP 14430))]
```

现在，**bt** 命令将显示线程 2 的回溯（输出已简化）：

```cpp
(gdb) bt
#0  futex_wait (private=0, expected=2, futex_word=0x7fff5e406b00) at ../sysdeps/nptl/futex-internal.h:146
#2  0x000079d1f32a00f1 in lll_mutex_lock_optimized (mutex=0x7fff5e406b00) at ./nptl/pthread_mutex_lock.c:48
#7  0x000061cbaf1173fa in operator() (__closure=0x61cbafd64418) at 11x18-debug_deadlock.cpp:19
```

要在不同的线程中执行命令，只需使用 **thread apply** 命令，在这种情况下，在线程 1 和 3 上执行 **bt** 命令：

```cpp
(gdb) thread apply 1 3 bt
```

要在所有线程中执行命令，只需使用 **thread apply** **all <command>** 。

注意，当多线程程序中的断点被达到时，所有执行线程都会停止运行，从而允许检查程序的整体状态。当通过 **continue**、**step** 或 **next** 等命令重新启动执行时，所有线程将恢复。当前线程将向前移动一个语句，但其他线程向前移动几个语句或甚至在语句中间停止是不确定的。

当执行停止时，调试器将跳转并显示当前线程的执行上下文。为了避免调试器通过锁定调度器在线程之间跳转，我们可以使用以下命令：

```cpp
(gdb) set scheduler-locking <on/off>
```

我们还可以使用以下命令来检查调度器锁定状态：

```cpp
(gdb) show scheduler-locking
```

现在我们已经学习了一些用于多线程调试的新命令，让我们检查一下我们附加到调试器中的应用程序发生了什么。

如果我们检索线程 2 和 3 的回溯，我们可以看到以下内容（仅输出简化版，仅显示相关部分）：

```cpp
(gdb) thread apply all bt
Thread 3 (Thread 0x79d1f30006c0 (LWP 14429) "test"):
#0  futex_wait (private=0, expected=2, futex_word=0x7fff5e406b30) at ../sysdeps/nptl/futex-internal.h:146
#5  0x000061cbaf117e20 in std::mutex::lock (this=0x7fff5e406b30) at /usr/include/c++/14/bits/std_mutex.h:113
#7  0x000061cbaf117334 in operator() (__closure=0x61cbafd642b8) at 11x18-debug_deadlock.cpp:13
Thread 2 (Thread 0x79d1f26006c0 (LWP 14430) "test"):
#0  futex_wait (private=0, expected=2, futex_word=0x7fff5e406b00) at ../sysdeps/nptl/futex-internal.h:146
#5  0x000061cbaf117e20 in std::mutex::lock (this=0x7fff5e406b00) at /usr/include/c++/14/bits/std_mutex.h:113
#7  0x000061cbaf1173fa in operator() (__closure=0x61cbafd64418) at 11x18-debug_deadlock.cpp:19
```

注意，在运行 **std::mutex::lock()** 之后，两个线程都在第 13 行等待线程 3，在第 19 行等待线程 2，这与 **std::thread** **t1** 中的 **std::lock_guard** **lock2** 和 **std::thread** **t2** 中的 **std::lock_guard** **lock1** 相匹配。

因此，我们在这些代码位置检测到了这些线程中发生的死锁。

现在我们来学习更多关于通过捕获竞态条件来调试多线程软件的知识。

## 调试竞态条件

竞态条件是最难检测和调试的 bug 之一，因为它们通常以间歇性的方式发生，每次发生时都有不同的效果，有时在程序达到失败点之前会发生一些昂贵的计算。

这种不稳定的行为不仅由竞态条件引起。与不正确的内存分配相关的其他问题也可能导致类似症状，因此，在调查并达到根本原因诊断之前，无法将 bug 分类为竞态条件。

调试竞态条件的一种方法是通过 watchpoints 手动检查变量是否在没有当前线程中执行的任何语句修改它的情况下更改其值，或者放置在特定线程触发的策略位置上的断点，如下所示：

```cpp
(gdb) break <linespec> thread <id> if <condition>
```

例如，参见以下内容：

```cpp
(gdb) break test.cpp:11 thread 2
```

或者，甚至可以使用断言并检查任何由不同线程访问的变量的当前值是否具有预期的值。这种方法在下一个示例中得到了应用：

```cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
using namespace std::chrono_literals;
static int g_value = 0;
static std::mutex g_mutex;
void func1() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < 10; ++i) {
        int old_value = g_value;
        int incr = (rand() % 10);
        g_value += incr;
        assert(g_value == old_value + incr);
        std::this_thread::sleep_for(10ms);
    }
}
void func2() {
    for (int i = 0; i < 10; ++i) {
        int old_value = g_value;
        int incr = (rand() % 10);
        g_value += (rand() % 10);
        assert(g_value == old_value + incr);
        std::this_thread::sleep_for(10ms);
    }
}
int main() {
    std::thread t1(func1);
    std::thread t2(func2);
    t1.join();
    t2.join();
    return 0;
}
```

在这里，两个线程**t1**和**t2**正在运行增加**g_value**全局变量随机值的函数。每次增加时，都会将**g_value**与预期值进行比较，如果不相等，断言指令将停止程序。

按照以下方式编译此程序并运行调试器：

```cpp
$ g++ -o test -g -O0 test
$ gdb ./test
```

调试器启动后，使用**运行**命令来运行程序。程序将运行，并在某个时刻由于收到**SIGABRT**信号而终止，表明断言未满足。

```cpp
test: test.cpp:29: void func2(): Assertion `g_value == old_value + incr' failed.
Thread 3 "test" received signal SIGABRT, Aborted.
```

程序停止后，我们可以使用**backtrace**命令检查该点的回溯，并将该点失败处的源代码更改为特定的**帧**或**列表**。

这个例子相当简单，所以通过检查断言输出，可以清楚地看出**g_value**变量出了问题，这很可能是竞态条件。

但是，对于更复杂的程序，手动调试问题的这个过程相当困难，所以让我们关注另一种称为反向调试的技术，它可以帮助我们解决这个问题。

## 反向调试

**反向调试**，也称为**时间旅行调试**，允许调试器在程序失败后停止程序，并回溯到程序执行的记录中，以调查失败的原因。此功能通过记录（记录）正在调试的程序中的每个机器指令以及内存和寄存器值的每次更改来实现，之后，使用这些记录随意回放和重放程序。

在 Linux 上，我们可以使用 GDB（自 7.0 版本起）、**rr**（最初由 Mozilla 开发，[`rr-project.org`](https://rr-project.org)）或**Undo 的时光旅行调试器**（**UDB**）（[`docs.undo.io`](https://docs.undo.io)）。在 Windows 上，我们可以使用**时光旅行调试**（[`learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/time-travel-debugging-overview`](https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/time-travel-debugging-overview)）。

反向调试仅由有限数量的 GDB 目标支持，例如远程目标 Simics、**系统集成和设计**（**SID**）模拟器或原生 Linux 的**进程记录和回放**目标（仅适用于**i386**、**amd64**、**moxie-elf**和**arm**）。在撰写本书时，Clang 的反向调试功能仍在开发中。

因此，由于这些限制，我们决定通过使用**rr**进行一个小型展示。请按照项目网站上的说明构建和安装**rr**调试工具：[`github.com/rr-debugger/rr/wiki/Building-And-Installing`](https://github.com/rr-debugger/rr/wiki/Building-And-Installing)。

安装后，要记录和回放程序，请使用以下命令：

```cpp
$ rr record <program> --args <args>
$ rr replay
```

例如，如果我们有一个名为**test**的程序，命令序列将如下所示：

```cpp
$ rr record test
rr: Saving execution to trace directory `/home/user/.local/share/rr/test-1'.
```

如果显示以下致命错误：

```cpp
[FATAL src/PerfCounters.cc:349:start_counter()] rr needs /proc/sys/kernel/perf_event_paranoid <= 3, but it is 4.
Change it to <= 3.
Consider putting 'kernel.perf_event_paranoid = 3' in /etc/sysctl.d/10-rr.conf.
```

然后，使用以下命令调整内核变量，**kernel.perf_event_paranoid**：

```cpp
$ sudo sysctl kernel.perf_event_paranoid=1
```

一旦有记录可用，请使用**replay**命令开始调试程序：

```cpp
$ rr replay
```

或者，如果程序崩溃并且你只想在记录的末尾开始调试，请使用**–** **e**选项：

```cpp
$ rr replay -e
```

在这一点上，**rr**将使用 GDB 调试器启动程序并加载其调试符号。然后，你可以使用以下任何命令进行反向调试：

+   **reverse-continue**：以反向方式开始执行程序。执行将在达到断点或由于同步异常而停止。

+   **reverse-next**：反向运行到当前栈帧中之前执行的上一行的开始。

+   **reverse-nexti**：这会反向执行一条指令，跳转到内部栈帧。

+   **reverse-step**：运行程序直到控制达到新源行的开始。

+   **reverse-stepi**：反向执行一条机器指令。

+   **reverse-finish**：这会执行到当前函数调用，即当前函数的开始处。

我们也可以通过使用以下命令来反转调试方向，并使用正向调试的常规命令（如**next**、**step**、**continue**等）在相反方向进行：

```cpp
(rr) set exec-direction reverse
```

要将执行方向恢复到正向，请使用以下命令：

```cpp
(rr) set exec-direction forward
```

作为练习，安装**rr**调试器并尝试使用反向调试来调试前面的示例。

现在让我们继续探讨如何调试协程，由于协程的异步特性，这是一个具有挑战性的任务。

## 调试协程

如我们所见，异步代码可以通过在战略位置使用断点、使用观察点检查变量、进入或跳过代码来像同步代码一样进行调试。此外，使用前面描述的技术选择特定线程并锁定调度器有助于在调试时避免不必要的干扰。

如我们所已了解，异步代码中存在复杂性，例如异步代码执行时将使用哪个线程，这使得调试更加困难。对于 C++ 协程，由于它们的挂起/恢复特性，调试甚至更难掌握。

Clang 使用两步编译使用协程的程序：语义分析由 Clang 执行，协程帧在 **LLVM** 中间端构建和优化。由于调试信息是在 Clang 前端生成的，因此在编译过程中较晚生成协程帧时，将会有不足的调试信息。GCC 采用类似的方法。

此外，如果执行在协程内部中断，当前帧将只有一个变量，**frame_ptr**。在协程中，没有指针或函数参数。协程在挂起之前将它们的状态存储在堆中，并且在执行期间只使用栈。**frame_ptr** 用于访问协程正常运行所需的所有必要信息。

让我们调试在 *第九章* 中实现的 **Boost.Asio** 协程示例。在这里，我们只展示相关的指令。请访问 *第九章* 中的 *协程* 部分，以检查完整的源代码：

```cpp
boost::asio::awaitable<void> echo(tcp::socket socket) {
    char data[1024];
    while (true) {
        std::cout << "Reading data from socket...\n";//L12
        std::size_t bytes_read = co_await
            socket.async_read_some(
                boost::asio::buffer(data),
                             boost::asio::use_awaitable);
        /* .... */
        co_await boost::asio::async_write(socket,
                boost::asio::buffer(data, bytes_read),
                boost::asio::use_awaitable);
    }
}
boost::asio::awaitable<void>
listener(boost::asio::io_context& io_context,
         unsigned short port) {
    tcp::acceptor acceptor(io_context,
                           tcp::endpoint(tcp::v4(), port));
    while (true) {
        std::cout << "Accepting connections...\n";  // L45
        tcp::socket socket = co_await
            acceptor.async_accept(
                boost::asio::use_awaitable);
        boost::asio::co_spawn(io_context,
            echo(std::move(socket)),
            boost::asio::detached);
    }
}
/* main function */
```

由于我们使用 Boost，让我们在编译源代码时包含 **Boost.System** 库，以添加更多符号以进行调试：

```cpp
$ g++ --std=c++20 -ggdb -O0 --fno-omit-frame-pointer -lboost_system  test.cpp -o test
```

然后，我们使用生成的程序启动调试器，并在第 12 行和第 45 行设置断点，这些是每个协程中 while 循环内第一条指令的位置：

```cpp
$ gdb –q ./test
(gdb) b 12
(gdb) b 45
```

我们还启用了 GDB 内置的格式化打印器，以显示标准模板库容器的可读输出：

```cpp
(gdb) set print pretty on
```

如果现在运行程序（**运行**命令），它将在接受连接之前到达协程监听器内的第 42 行的断点。使用 **info locals** 命令，我们可以检查局部变量。

协程创建了一个具有多个内部字段的状态机，例如带有线程的承诺对象、调用对象的地址、挂起的异常等。它们还存储 **resume** 和 **destroy** 回调。这些结构是编译器依赖的，与编译器的实现相关联，并且如果我们使用 Clang，可以通过 **frame_ptr** 访问。

如果我们继续运行程序（使用**继续**命令），服务器将等待客户端连接。要退出等待状态，我们使用**telnet**，如*第九章*所示，将客户端连接到服务器。此时，执行将停止，因为达到**echo()**协程内部第 12 行的断点，并且**info locals**显示了每个**echo**连接使用的变量。

使用**回溯**命令将显示一个调用栈，由于协程的挂起特性，可能存在一些复杂性。

在纯 C++例程中，如*第八章*所述，有两个设置断点可能有趣的表达式：

+   **co_await**：执行将在等待的操作完成后挂起。可以通过检查底层的**await_suspend**、**await_resume**或自定义可等待代码来在协程恢复的点设置断点。

+   **co_yield**：挂起执行并返回一个值。在调试期间，进入**co_yield**以观察控制流如何在协程及其调用函数之间进行。

由于协程在 C++世界中相当新颖，并且编译器持续发展，我们希望不久的将来调试协程将更加直接。

一旦我们找到并调试了一些错误，并且可以重现导致这些特定错误的场景，设计一些涵盖这些情况的测试将很方便，以避免未来代码更改可能导致类似问题或事件。让我们在下一章学习如何测试多线程和异步代码。

# 摘要

在本章中，我们学习了如何使用日志和调试异步程序。

我们从使用日志来发现运行软件中的问题开始，展示了使用**spdlog**日志库检测死锁的有用性。还讨论了许多其他库，描述了它们可能适合特定场景的相关功能。

然而，并非所有错误都可以通过使用日志来发现，有些错误可能只能在软件开发生命周期后期，当生产中出现问题时才会被发现，即使在处理程序崩溃和事件时也是如此。调试器是检查运行或崩溃程序的有用工具，了解其代码路径，并找到错误。介绍了几个示例和调试器命令来处理通用代码，但也特别针对多线程和异步软件、竞态条件和协程。还介绍了**rr**调试器，展示了将反向调试纳入我们的开发者工具箱的潜力。

在下一章中，我们将学习使用 sanitizers 和测试技术来性能和优化技术，这些技术可以用来改善异步程序的运行时间和资源使用。

# 进一步阅读

+   *日志*：https://en.wikipedia.org/wiki/Logging_(computing)

+   *Syslog*：https://en.wikipedia.org/wiki/Syslog

+   *Google 日志库* : https://github.com/google/glog

+   *Apache Log4cxx* : https://logging.apache.org/log4cxx

+   *spdlog* : https://github.com/gabime/spdlog

+   *Quill* : https://github.com/odygrd/quill

+   *xtr* : https://github.com/choll/xtr

+   *lwlog* : https://github.com/ChristianPanov/lwlog

+   *uberlog* : https://github.com/IMQS/uberlog

+   *Easylogging++* : https://github.com/abumq/easyloggingpp

+   *NanoLog* : https://github.com/PlatformLab/NanoLog

+   *Reckless 日志库* : https://github.com/mattiasflodin/reckless

+   *tracetool* : [`github.com/froglogic/tracetool`](https://github.com/froglogic/tracetool)

+   *Logback 项目* : [`logback.qos.ch`](https://logback.qos.ch)

+   *Sentry* : [`sentry.io`](https://sentry.io)

+   *Graylog* : [`graylog.org`](https://graylog.org)

+   *Logstash* : [`www.elastic.co/logstash`](https://www.elastic.co/logstash)

+   *使用 GDB 调试* : [`sourceware.org/gdb/current/onlinedocs/gdb.html`](https://sourceware.org/gdb/current/onlinedocs/gdb.html)

+   *LLDB 教程* : [`lldb.llvm.org/use/tutorial.html`](https://lldb.llvm.org/use/tutorial.html)

+   *Clang 编译器用户手册* : [`clang.llvm.org/docs/UsersManual.html`](https://clang.llvm.org/docs/UsersManual.html)

+   *GDB* : *运行程序反向执行* : [`www.zeuthen.desy.de/dv/documentation/unixguide/infohtml/gdb/Reverse-Execution.html#Reverse-Execution`](https://www.zeuthen.desy.de/dv/documentation/unixguide/infohtml/gdb/Reverse-Execution.html#Reverse-Execution)

+   *使用 GDB 进行反向调试* : [`sourceware.org/gdb/wiki/ReverseDebug`](https://sourceware.org/gdb/wiki/ReverseDebug)

+   *调试 C++ 协程* : [`clang.llvm.org/docs/DebuggingCoroutines.html`](https://clang.llvm.org/docs/DebuggingCoroutines.html)

+   *SID 模拟器用户手册* : [`sourceware.org/sid/sid-guide/book1.html`](https://sourceware.org/sid/sid-guide/book1.html)

+   *Intel Simics 模拟器用于 Intel FPGAs: 用户手册* : [`www.intel.com/content/www/us/en/docs/programmable/784383/24-1/about-this-document.html`](https://www.intel.com/content/www/us/en/docs/programmable/784383/24-1/about-this-document.html)

+   *IBM 支持* : *如何启用核心转储* : [`www.ibm.com/support/pages/how-do-i-enable-core-dumps`](https://www.ibm.com/support/pages/how-do-i-enable-core-dumps)

+   *核心转储 – 如何启用它们？* : `medium.com/@sourabhedake/core-dumps-how-to-enable-them-73856a437711`
