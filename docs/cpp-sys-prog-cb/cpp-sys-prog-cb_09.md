# 第九章：处理时间接口

时间在操作系统和应用程序中以多种形式使用。通常，应用程序需要处理以下**时间类别**：

+   **时钟**：实际的时间和日期，就像您手表上读到的那样

+   **时间点**：用于对应用程序的使用情况（例如处理器或资源）进行分析、监视和故障排除所花费的处理时间

+   **持续时间**：单调时间，即某个事件的经过时间

在这一章中，我们将从 C++和 POSIX 的角度处理所有这些方面，以便您在工具箱中有更多可用的工具。本章的示例将教您如何使用时间点来测量事件，以及为什么应该使用稳定的时钟，以及时间超出限制的情况以及如何减轻它。您将学习如何使用 POSIX 和 C++ `std::chrono`来实现这些概念。

本章将涵盖以下示例：

+   学习 C++时间接口

+   使用 C++20 日历和时区

+   学习 Linux 时间

+   处理时间休眠和超出限制

# 技术要求

要立即尝试本章中的程序，我们已经设置了一个包含本书所需的所有工具和库的 Docker 镜像。它基于 Ubuntu 19.04。

为了设置它，按照以下步骤进行：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。输入以下命令查看镜像：`docker images`。

1.  您应该有以下镜像：`kasperondocker/system_programming_cookbook`。

1.  使用`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`命令以交互式 shell 运行 Docker 镜像。

1.  正在运行的容器上的 shell 现在可用。转到`root@39a5a8934370/# cd /BOOK/`以获取本书中将开发的所有程序。

需要`--cap-add sys_ptrace`参数以允许**GDB**（GNU 项目调试器的缩写）设置断点，Docker 默认情况下不允许。

**免责声明**：C++20 标准已经在二月底的布拉格的 WG21 会议上获得批准（即技术上已经最终确定）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 示例代码。GCC 将最新功能的开发保留在分支中（您必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励您自行尝试。因此，请克隆并探索 GCC 合同和模块分支，并尽情享受。

# 学习 C++时间接口

C++11 标准确实标志着时间方面的重要进展。在此之前（C++标准 98 及之前），系统和应用程序开发人员必须依赖于特定于实现的 API（即 POSIX）或外部库（例如`boost`）来操作**时间**，这意味着代码的可移植性较差。本示例将教您如何使用标准时间操作库编写 C++代码。

# 如何做...

让我们编写一个程序来学习 C++标准中支持的**时钟**、**时间点**和**持续时间**的概念：

1.  创建一个新文件并将其命名为`chrono_01.cpp`。首先我们需要一些包含：

```cpp
#include <iostream>
#include <vector>
#include <chrono>
```

1.  在`main`部分，我们需要一些东西来测量，所以让我们用一些整数填充一个`std::vector`：

```cpp
int main ()
{
    std::cout << "Starting ... " << std::endl;
    std::vector <int> elements;
    auto start = std::chrono::system_clock::now();

    for (auto i = 0; i < 100'000'000; ++i)
        elements.push_back(i);

    auto end = std::chrono::system_clock::now();
```

1.  现在我们有了两个时间点`start`和`end`，让我们计算差异（即持续时间）并打印出来看看花了多长时间：

```cpp
    // default seconds
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "Time Spent for populating a vector with     
        100M of integer ..." 
              << diff.count() << "msec" << std::endl;
```

1.  现在，我们想以另一种格式打印`start`变量；例如，以`ctime`的日历本地时间格式：

```cpp
    auto tpStart = std::chrono::system_clock::to_time_t(start);
    std::cout << "Start: " << std::ctime(&tpStart) << std::endl;

    auto tpEnd = std::chrono::system_clock::to_time_t(end);
    std::cout << "End: " << std::ctime(&tpEnd) << std::endl;
    std::cout << "Ended ... " << std::endl;
}
```

这个程序使用了一些`std::chrono`的特性，比如标准库中可用的`system_clock`、`time_point`和持续时间，并且自 C++标准的第 11 版以来一直在使用。

# 它是如何工作的...

*步骤 1*负责包含我们稍后需要的头文件：`<iostream>`用于标准输出，`<vector>`和`<chrono>`用于时间。

*步骤 2*定义了一个名为`elements`的**int 类型的向量**。由于这个，我们可以在`chrono`命名空间中的`system_clock`类上调用`now()`方法来获取当前时间。虽然我们使用了`auto`，这个方法返回一个表示时间点的`time_point`对象。然后，我们循环了 1 亿次来填充`elements`数组，以突出我们使用了新的 C++14 特性来表示*100,000,000*，这提高了代码的可读性。最后，我们通过调用`now()`方法并将`time_point`对象存储在`end`变量中来获取另一个时间点。

在*步骤 3*中，我们看了执行循环需要多长时间。为了计算这个时间，我们实例化了一个`duration`对象，它是一个需要两个参数的模板类：

+   **表示**：表示滴答数的类型。

+   **周期**：这可以是（等等）`std::nano`、`std:micro`、`std::milli`等。

周期的默认值是`std::seconds`。然后，我们只需在标准输出上写`diff.cout()`，它表示`start`和`end`之间的毫秒数。计算这种差异的另一种方法是使用`duration_cast`；例如，`std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count()`。

在*步骤 4*中，我们以日历`localtime`表示打印`start`和`end`的`time_point`变量（注意，容器时间可能与主机容器不同步）。为了做到这一点，我们需要通过使用`system_clock`类的`to_time_t()`静态变量将它们转换为`time_t`，然后将它们传递给`std::ctime`方法。

现在，让我们构建并运行这个：

![](img/b733171a-695f-4db6-b3ae-eab79b55b5d8.png)

我们将在下一节中更多地了解这个示例。

# 还有更多...

我们开发的程序使用了`system_clock`类。在`chrono`命名空间中有三个时钟类：

+   `system_clock`：这代表了所谓的**挂钟时间**。它可以在任何时刻被调整，比如当通过闰秒引入额外的不精确性或用户刚刚设置它时。在大多数实现中，它的纪元（即其起点）使用 UNIX 时间，这意味着起点从 1970 年 1 月 1 日开始计数。

+   `steady_clock`：这代表了所谓的**单调时钟**。它永远不会被调整。它保持稳定。在大多数实现中，它的起点是机器启动时的时间。为了计算某个事件的经过时间，你应该考虑使用这种类型的时钟。

+   `high_resolution_clock`：这是可用最短滴答的时钟。它可能只是`system_clock`或`steady_clock`的别名，或者是一个完全不同的实现。这是由实现定义的。

另一个需要记住的方面是，C++20 标准包括了`time_of_day`、日历和时区。

# 另请参阅

+   *学习 Linux 时间*的简要比较

+   *Bjarne Stroustrup 的《C++之旅，第二版》*

# 使用 C++20 日历和时区

C++20 标准丰富了`std::chrono`命名空间的日历功能。它们包括你所期望的所有典型功能，以及一种更成语化和直观的玩法。这个示例将教你一些最重要的功能，以及如何与`std::chrono`命名空间的日历部分交互是多么简单。

# 如何做...

让我们看一些代码：

1.  创建一个新文件，确保你包含了`<chrono>`和`<iostream>`。我们有一个日期，我们想知道`bday`会在星期几。

```cpp
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main ()
{
    auto bday = January/30/2021;
    cout << weekday(bday) << endl;

    auto anotherDay = December/25/2020;
    if (bday == anotherDay)
        cout << "the two date represent the same day" << endl;
    else
        cout << "the two dates represent two different days"    
            << endl;
}
```

1.  有一整套类可以让您玩转日历。让我们来看看其中一些：

```cpp
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main ()
{
    auto today = year_month_day{ floor<days>(system_clock::now()) };
    auto ymdl = year_month_day_last(today.year(), month*day* last{ month{ 2 } });
    auto last_day_feb = year_month_day{ ymdl };
    std::cout << "last day of Feb is: " << last_day_feb
        << std::endl;

    return 0;
}
```

1.  让我们玩玩时区，并打印不同时区的时间列表：

```cpp
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main()
{
    auto zone_names = {
       "Asia/Tokyo",
       "Europe/Berlin",
       "Europe/London",
       "America/New_York",
    };

    auto localtime = zoned_time<milliseconds>(date::current_zone(),
                                              system_clock::now());
    for(auto const& name : zone_names)
        cout << name
             << zoned_time<milliseconds>(name, localtime)
             << std::endl;

    return 0;
}
```

1.  一个经常使用的功能是用于找到两个时区之间的差异：

```cpp
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main()
{
    auto current = system_clock::now();
    auto lon = zoned_time{"Europe/London", current_time};
    auto newYork = zoned_time{"America/New_York", current_time};
    cout <<"Time Difference between London and New York:" 
         << (lon.get_local_time() - newYork.get_local_time())
             << endl;

    return 0;
}
```

让我们深入了解`std::chrono`日历部分，以了解更多关于这个示例的内容。

# 它是如何工作的...

在新的 C++20 标准中有许多日历和时区辅助函数可用。这个示例只是触及了表面，但仍然让我们了解了处理时间是多么容易。`std::chrono`日历和时区功能的参考可以在[`en.cppreference.com/w/cpp/chrono`](https://en.cppreference.com/w/cpp/chrono)上找到。

*步骤 1*使用`weekday`方法来获取一周的日期（使用公历）。在调用`weekday`方法之前，我们需要获取一个特定的日期，使用 C++20，我们可以直接设置`auto bday = January/30/2021`，这代表一个日期。现在，我们可以将其传递给`weekday`方法来获取特定的一周日期，在我们的例子中是星期六。一个有用的属性是我们可以比较日期，就像我们可以在`bday`和`anotherDay`变量之间进行比较。`weekday`以及所有其他`std::chrono`日历方法都处理闰秒。

*步骤 2*展示了`year_month_day`和`year_month_day_last`方法的使用。该库包含了一整套类似于这两个方法的类，例如`month_day`和`month_day_lat`等等。它们显然有不同的范围，但原则仍然相同。在这一步中，我们对二月的最后一天感兴趣。我们使用`year_month_day{ floor<days>(system_clock::now()) }`将当前日期设置在`today`变量中，然后将`today`传递给`year_month_day_last`方法，它将返回类似`2020/02/last`的内容，我们将其存储在`ymdl`变量中。我们可以再次使用`year_month_day`方法来获取二月的最后一天。我们可以跳过一些步骤，直接调用`year_month_day_last`方法。我们进行这一步是为了教育目的。

*步骤 3*进入时区范围。此步骤中的代码片段通过迭代`zone_names`数组打印出一个时区列表。在这里，我们首先通过循环遍历每个由字符串标识的时区来获取`localtime`。然后，我们使用`zoned_time`方法将`localtime`转换为由`name`变量标识的时区。

在*步骤 4*中，我们涵盖了一个有趣且经常发生的问题：找到两个时区之间的时间差。原则没有改变；我们仍然使用`zoned_time`方法来获取两个时区的本地时间，这些时区在这种情况下是`"America/New_York"`和`"Europe/London"`。然后，我们减去两个本地时间以获取差异。

# 还有更多...

`std::chrono`日历提供了各种各样的方法，欢迎您去探索。完整的列表可以在[`en.cppreference.com/w/cpp/chrono`](https://en.cppreference.com/w/cpp/chrono)上找到。

# 另请参阅

+   《C++之旅，第二版》，作者 Bjarne Stroustrup，第 13.7 章，时间

# 学习 Linux 时间。

在 C++11 之前，标准库没有包含任何直接的时间管理支持，因此系统开发人员必须使用*外部*来源。所谓外部，指的是外部库（例如 Boost ([`www.boost.org/`](https://www.boost.org/)））或特定于操作系统的 API。我们认为系统开发人员有必要了解 Linux 中的时间概念。这个示例将帮助您掌握**时钟**、**时间点**和**持续时间**等概念，使用 POSIX 标准。

# 如何做...

在这个示例中，我们将编写一个程序，以便我们可以学习关于 Linux 中**时钟**、**时间点**和**持续时间**的概念。让我们开始吧：

1.  在 shell 中，创建一个名为`linux_time_01.cpp`的新文件，并添加以下包含和函数原型：

```cpp
#include <iostream>
#include <time.h>
#include <vector>

void timespec_diff(struct timespec* start, struct timespec* stop, struct timespec* result);
```

1.  现在，我们想要看到`clock_gettime`调用中`CLOCK_REALTIME`和`CLOCK_MONOTONIC`之间的差异。我们需要定义两个`struct timespec`变量：

```cpp
int main ()
{
    std::cout << "Starting ..." << std::endl;
    struct timespec tsRealTime, tsMonotonicStart;
    clock_gettime(CLOCK_REALTIME, &tsRealTime);
    clock_gettime(CLOCK_MONOTONIC, &tsMonotonicStart);
```

1.  接下来，我们需要打印`tsRealTime`和`tsMonoliticStart`变量的内容以查看它们之间的差异：

```cpp
    std::cout << "Real Time clock (i.e.: wall clock):"
        << std::endl;
    std::cout << " sec :" << tsRealTime.tv_sec << std::endl;
    std::cout << " nanosec :" << tsRealTime.tv_nsec << std::endl;

    std::cout << "Monotonic clock:" << std::endl;
    std::cout << " sec :" << tsMonotonicStart.tv_sec << std::endl;
    std::cout << " nanosec :" << tsMonotonicStart.tv_nsec+
        << std::endl;
```

1.  我们需要一个任务来监视，所以我们将使用`for`循环来填充一个`std::vector`。之后，我们立即在`tsMonotonicEnd`变量中获取一个时间点：

```cpp
    std::vector <int> elements;
    for (int i = 0; i < 100'000'000; ++i)
        elements.push_back(i);

    struct timespec tsMonotonicEnd;
    clock_gettime(CLOCK_MONOTONIC, &tsMonotonicEnd);
```

1.  现在，我们想要打印任务的持续时间。为此，我们调用`timespec_diff`（辅助方法）来计算`tsMonotonicEnd`和`tsMonotonicStart`之间的差异：

```cpp
    struct timespec duration;
    timespec_diff (&tsMonotonicStart, &tsMonotonicEnd, &duration);

    std::cout << "Time elapsed to populate a vector with
        100M elements:" << std::endl;
    std::cout << " sec :" << duration.tv_sec << std::endl;
    std::cout << " nanosec :" << duration.tv_nsec << std::endl;
    std::cout << "Finished ..." << std::endl;
}
```

1.  最后，我们需要实现一个辅助方法来计算`start`和`stop`变量表示的时间之间的时间差（即持续时间）：

```cpp
// helper method
void timespec_diff(struct timespec* start, struct timespec* stop, struct timespec* result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) 
    {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec
          + 100'000'0000;
    } 
    else 
    {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }
    return;
}
```

上述程序展示了如何收集时间点以计算事件的持续时间。现在，让我们深入了解该程序的细节。

# 工作原理...

首先，让我们编译并执行程序：

![](img/f4a8718b-5a8b-46ac-a1d0-cd3365395fdf.png)

我们可以立即注意到，实时时钟（秒）远远大于单调时钟（秒）。通过一些数学运算，您会注意到第一个大约是 49 年，而后者大约是 12 小时。为什么会这样？第二个观察是我们的代码花费了`1 秒`和`644348500`纳秒来填充 1 亿个项目的向量。让我们收集一些见解来解释这一点。

*步骤 1*只是添加了一些包含和我们编写的原型，用于计算时间差。

*步骤 2*定义了两个变量，`struct timespec tsRealTime`和`struct timespec tsMonotonicStart`，它们将用于存储两个时间点。然后，我们两次调用`clock_gettime()`方法，一次传递`CLOCK_REALTIME`和`tsRealTime`变量。我们再次传递`CLOCK_MONOTONIC`和`tsMonotonicStart`变量。`CLOCK_REALTIME`和`CLOCK_MONOTONIC`都是`clockid_t`类型。当使用`CLOCK_REALTIME`调用`clock_gettime()`时，我们得到的时间将是`挂钟`时间（或实时时间）。

这个时间点有与我们在*学习 C++时间接口*中看到的`std::chrono::SYSTEM_CLOCK`相同的问题。它可以被调整（例如，如果系统时钟与 NTP 同步），因此不适合计算事件的经过时间（或持续时间）。当使用`CLOCK_MONOTONIC`参数调用`clock_gettime()`时，时间不会调整，大多数实现会从系统启动开始计时（即从机器启动开始计算时钟滴答）。这非常适合事件持续时间的计算。

*步骤 3*只是打印时间点的结果，即`tsRealTime`和`tsMonotonicStart`。我们可以看到第一个包含自 1970 年 1 月 1 日以来的秒数（大约 49 年），而后者包含自我的机器启动以来的秒数（大约 12 小时）。

*步骤 4*只是在`std::vector`中添加了 1 亿个项目，然后在`tsMonotonicEnd`中获取了另一个时间点，这将用于计算此事件的持续时间。

*步骤 5*计算了`tsMonotonicStart`和`tsMonotonicEnd`之间的差异，并通过调用`timespec_diff()`辅助方法将结果存储在`duration`变量中。

*步骤 6*实现了`timespec_diff()`方法，逻辑上计算(`tsMonotonicEnd - tsMonotonicStart`)。

# 还有更多...

对于`clock_gettime()`方法，我们使用 POSIX 作为对应的设置方法：`clock_settime()`。对于`gettimeofday()`也是如此：`settimeofday()`。

值得强调的是，`gettimeofday()`是`time()`的扩展，返回一个`struct timeval`（即秒和微秒）。这种方法的问题在于它可以被调整。这是什么意思？让我们想象一下，您使用`usegettimeofday()`在事件之前获取一个时间点来测量，然后在事件之后获取另一个时间点来测量。在这里，您会计算两个时间点之间的差异，认为一切都很好。这里可能会出现什么问题？想象一下，在您获取的两个时间点之间，**网络时间协议**（**NTP**）服务器要求本地机器调整本地时钟以使其与时间服务器同步。由于受到 NTP 同步的影响，计算出的持续时间将不准确。NTP 只是一个例子。本地时钟也可以以其他方式进行调整。

# 另请参阅

+   用于与 C++时间接口进行比较的*了解 C++时间接口*配方

+   *Linux 系统编程，第二版*，作者*Robert Love

# 处理时间休眠和超时

在系统编程的上下文中，时间不仅涉及测量事件持续时间或读取时钟的行为。还可以将进程置于休眠状态一段时间。这个配方将教你如何使用基于秒的 API、基于微秒的 API 和具有纳秒分辨率的`clock_nanosleep()`方法来使进程进入休眠状态。此外，我们将看到时间超时是什么，以及如何最小化它们。

# 如何做...

在这一部分，我们将编写一个程序，学习如何使用不同的 POSIX API 来使程序进入休眠状态。我们还将看看 C++的替代方法：

1.  打开一个 shell 并创建一个名为`sleep.cpp`的新文件。我们需要添加一些稍后需要的头文件：

```cpp
#include <iostream>
#include <chrono>
#include <thread>    // sleep_for
#include <unistd.h>  // for sleep
#include <time.h>    // for nanosleep and clock_nanosleep
```

1.  我们将使用`sleep()`方法和`std::chrono::steady_clock`类作为时间点，将程序置于休眠状态`1`秒，以计算持续时间：

```cpp
int main ()
{
    std::cout << "Starting ... " << std::endl;

    auto start = std::chrono::steady_clock::now();
    sleep (1);
    auto end = std::chrono::steady_clock::now();
    std::cout << "sleep() call cause me to sleep for: " 
              << std::chrono::duration_cast<std::chrono::
                  milliseconds> (end-start).count() 
              << " millisec" <<     std::endl;
```

1.  让我们看看`nanosleep()`是如何工作的。我们仍然使用`std::chrono::steady_clock`来计算持续时间，但我们需要一个`struct timespec`。我们将使进程休眠约`100`毫秒：

```cpp
    struct timespec reqSleep = {.tv_sec = 0, .tv_nsec = 99999999};
    start = std::chrono::steady_clock::now();
    int ret = nanosleep (&reqSleep, NULL);
    if (ret)
         std::cerr << "nanosleep issue" << std::endl;
    end = std::chrono::steady_clock::now();
    std::cout << "nanosleep() call cause me to sleep for: " 
              << std::chrono::duration_cast<std::
                  chrono::milliseconds> (end-start).count() 
              << " millisec" << std::endl;
```

1.  将进程置于休眠状态的更高级方法是使用`clock_nanosleep()`，它允许我们指定一些有趣的参数（更多细节请参见下一节）：

```cpp
    struct timespec reqClockSleep = {.tv_sec = 1, 
        .tv_nsec = 99999999};
    start = std::chrono::steady_clock::now();
    ret = clock_nanosleep (CLOCK_MONOTONIC, 0,
        &reqClockSleep, NULL);
    if (ret)
        std::cerr << "clock_nanosleep issue" << std::endl;
    end = std::chrono::steady_clock::now();
    std::cout << "clock_nanosleep() call cause me to sleep for: " 
              << std::chrono::duration_cast<std::chrono::
                  milliseconds> (end-start).count() 
              << " millisec" << std::endl;
```

1.  现在，让我们看看如何使用 C++标准库（通过`std::this_thread::sleep_for`模板方法）将当前线程置于休眠状态：

```cpp
    start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    end = std::chrono::steady_clock::now();
    std::cout << "std::this_thread::sleep_for() call
      cause me to sleep for: " 
              << std::chrono::duration_cast<std::chrono::
                  milliseconds> (end-start).count() 
              << " millisec" << std::endl;
    std::cout << "End ... " << std::endl;
}
```

现在，让我们更详细地了解这些步骤。

# 它是如何工作的...

程序将以四种不同的方式进入休眠状态。让我们来看看运行时间：

![](img/a3c3be5d-eca0-4e1b-97a5-1d4492bd48a0.png)

*步骤 1*只包含我们需要的头文件：`<iostream>`用于标准输出和标准错误（`cout`和`cerr`），`<chrono>`用于将用于测量实际休眠的时间点，`<thread>`用于`sleep_for`方法，`<unistd>`用于`sleep()`，`<time.h>`用于`nanosleep()`和`clock_nanosleep()`。

*步骤 2*使用`sleep()`方法使进程休眠`1`秒。我们使用`steady_clock::now()`来获取时间点，使用`duration_cast`来转换差异并获取实际持续时间。要精确，`sleep()`返回`0`，如果进程成功休眠至少指定时间量，但它可以返回一个介于 0 和指定秒数之间的值，这代表了**未**休眠的时间。

*步骤 3*展示了如何使用`nanosleep()`使进程进入睡眠状态。我们决定使用这种方法，因为在 Linux 上已经弃用了`usleep()`。`nanosleep()`比`sleep()`更有优势，因为它具有纳秒分辨率，并且`POSIX.1b`是标准化的。`nanosleep()`在成功时返回`0`，在错误时返回`-1`。它通过将`errno`全局变量设置为发生的特定错误来实现这一点。`struct timespec`变量包含`tv_sec`和`tv_nsec`（秒和纳秒）。

*步骤 4*使用了一个更复杂的`clock_nanosleep()`。这种方法包含了我们尚未看到的两个参数。第一个参数是`clock_id`，接受，除其他外，`CLOCK_REALTIME`和`CLOCK_MONOTONIC`，我们在前面的配方中已经看过了。作为一个经验法则，如果你要睡到绝对时间（挂钟时间），你应该使用第一个，如果你要睡到相对时间值，你应该使用第二个。根据我们在前面的配方中看到的，这是有道理的。

第二个参数是一个标志；它可以是`TIME_ABSTIME`或`0`。如果传递第一个，那么`reqClockSleep`变量将被视为绝对时间，但如果传递`0`，那么它将被视为相对时间。为了进一步澄清绝对时间的概念，它可能来自前一次调用`clock_gettime()`，它将绝对时间点存储在一个变量中，比如`ts`。通过向其添加`2`秒，我们可以将`&ts`（即变量`ts`的地址）传递给`clock_nanosleep()`，它将等待到那个特定的绝对时间。

*步骤 5*让当前线程的进程进入睡眠状态（在这种情况下，当前线程是主线程，所以整个进程将进入睡眠状态）1.5 秒（1,500 毫秒=1.5 秒）。`std::this_thread::sleep_for`简单而有效。它是一个模板方法，接受一个参数作为输入；也就是说，`duration`，它需要表示类型和周期（`_Rep`和`_Period`），正如我们在*学习 C++时间接口*配方中看到的。在这种情况下，我们只传递了毫秒的周期，并将表示保留在其默认状态。

这里有一个问题我们应该注意：**时间超出**。我们在这个配方中使用的所有接口都保证进程将至少睡眠*所请求的时间*。否则它们会返回错误。它们可能会因为不同的原因而睡眠时间略长于我们请求的时间。一个原因可能是由于选择了不同的任务来运行的调度程序。当计时器的粒度大于所请求的时间时，就会出现这个问题。例如，考虑一下计时器显示的时间（`10msec`）和睡眠时间为`5msec`。我们可能会遇到一个情况，进程必须等待比预期多`5`毫秒，这是 100%的增加。时间超出可以通过使用支持高精度时间源的方法来减轻，例如`clock_nanosleep()`、`nanosleep()`和`std::this_thread::sleep_for()`。

# 还有更多...

我们没有明确提到`nanosleep()`和`clock_nanosleep()`的线程影响。这两种方法都会导致当前线程进入睡眠状态。在 Linux 上，睡眠意味着线程（或者如果是单线程应用程序，则是进程）将进入**不可运行**状态，以便 CPU 可以继续执行其他任务（请记住，Linux 不区分线程和进程）。

# 另请参阅

+   *学习 C++时间接口*的一篇评论，审查`std::chrono::duration<>`模板类

+   *学习 Linux 时间*的一篇评论，审查**REALTIME**和**MONOTONIC**的概念
