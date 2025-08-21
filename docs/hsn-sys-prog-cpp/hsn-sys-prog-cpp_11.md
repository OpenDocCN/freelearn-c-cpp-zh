# 第十一章：Unix 中的时间接口

在本章中，读者将学习如何使用 C++17 编程 POSIX 和 C++时间接口。首先，本章将介绍 UNIX 纪元和 POSIX `time.h` API 以及如何使用它们。接下来，将简要解释 C++ Chrono API，它们与`time.h`的关系，并提供一些示例。最后，本章将以两个简单的示例结束，演示如何使用时间接口。第一个示例将演示如何读取系统时钟并在间隔上将结果输出到控制台，第二个示例将演示如何使用 C++高分辨率计时器对软件进行基准测试。

在本章中，我们将涵盖以下主题：

+   学习 POSIX `time.h` API

+   C++ Chrono API

+   通过示例了解读取系统时钟

+   涉及高分辨率计时器的示例

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请访问以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter11`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter11)。

# 学习 POSIX `time.h` API

我们将从讨论 POSIX `time.h` API 开始，该 API 提供了用于读取各种时钟并对这些时钟时间进行计算的 API。尽管这些 API 特定于标准 C，但如下一节所示，当使用 C++时仍然需要 C 时间接口，这是 C++20 正在解决的问题。

# 学习有关 API 类型

UNIX 纪元定义了从 1970 年 1 月 1 日起的秒数。本章描述的接口利用 UNIX 纪元来定义时间的概念。本章中描述的 POSIX `time.h` API 定义了三种不同的不透明类型：

+   `tm`：一个不透明的结构，保存日期和时间。

+   `time_t`：一个`typedef`，通常使用存储从 UNIX 纪元起的秒数的整数来实现。

+   `clock_t`：一个`typedef`，用于存储应用程序执行的处理器时间量。

这些 API 提供了各种函数来创建这些类型并对其进行操作。应该注意，有不同类型的时钟：

+   **系统时钟**：系统时钟读取操作系统维护的时钟，并存储向用户呈现的日期和时间（例如，任务栏上显示的时钟）。这个时钟可以在任何时间改变，因此通常不建议在应用程序中使用它进行计时，因为所使用的时钟可能以意想不到的方式向后/向前移动。

+   **稳定时钟**：稳定时钟是程序执行时会滴答作响的时钟。程序执行得越多，这个时钟就会变得越大。应该注意，这个时钟不会与系统时钟的结果匹配，通常只有两个这些时钟之间的差异才有真正的价值。

+   **高分辨率时钟**：这与稳定时钟相同，唯一的区别是返回的结果具有更高的分辨率。这些类型的时钟通常用于基准测试。

# `time()` API

`time()` API 返回当前系统时钟，并采用以下形式：

```cpp
time_t time(time_t *arg);
```

您可以使用预先定义的`time_t`变量提供`time()`函数，或者它将为您返回一个（如果您将`nullptr`作为参数传递），如下所示：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);
    std::cout << "time: " << t << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 1531603643
```

在前面的例子中，我们使用`time()` API 创建一个名为`t`的变量，以获取从 UNIX 纪元开始的当前秒数。然后将这个值输出到`stdout`。应该注意，`time_t` typedef 通常使用整数值实现，这就是为什么我们可以直接将其值输出到`stdout`的原因，就像前面的例子中所示的那样。

正如所述，也可以像下面这样使用`time()`提供自己之前定义的变量：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    time_t t;
    time(&t);
    std::cout << "time: " << t << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 1531603652
```

前面的例子与第一个例子相同，但是不是存储`time()`的返回值，而是将我们的`time_t`变量作为参数传递给函数。虽然这种语法是支持的，但前者更受青睐。`time()`在出现错误时会返回`-1`，可以根据需要进行检查和处理。

# `ctime()` typedef

`time_t` typedef 是特定于实现的，尽管它通常使用存储从 Unix 纪元开始的秒数的整数实现，但不能保证这种情况，这意味着前面的例子可能不会编译。相反，要以支持的方式输出`time_t`变量的值，使用`ctime()` API，形式如下：

```cpp
char* ctime(const time_t* time);
```

`ctime()` API 接受一个指向`time_t`变量的指针，并输出一个标准的 C 字符串。返回的字符串所使用的内存由`time.h` API 维护（因此不需要被释放），因此不是线程安全的。可以如下使用这个 API：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);
    std::cout << "time: " << ctime(&t);
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 15:27:44 2018
```

从前面的例子可以看出，返回的不是从 Unix 纪元开始的秒数，而是当前时间和日期的可读版本。还应该注意的是，除了`ctime()`函数不是线程安全的之外，它也没有提供调整输出格式的机制。因此，通常不鼓励使用这个函数，而是使用其他`time.h`函数。

# `localtime()`和`gmtime()` API

`time()` API 返回一个存储从 Unix 纪元开始的秒数的`time_t`值，正如前面所述。这个值可以进一步处理以暴露日期和时间信息，使我们能够将日期和时间转换为本地时间或**格林尼治标准时间**（**GMT**）。为此，POSIX API 提供了`localtime()`和`gmtime()`函数，形式如下：

```cpp
struct tm *localtime( const time_t *time );
struct tm *gmtime( const time_t *time );
```

这两个函数都接受一个指向`time_t`变量的指针，并返回一个指向`tm`不透明结构的指针。应该注意，返回值指向的结构像`ctime()`一样由`time.h`实现管理，因此不会被用户释放，这意味着这个函数的结果不是线程安全的。

# `asctime()`函数

要将不透明的`tm`结构输出到`stdout`（或者一般来说，只是将结构转换为标准的 C 字符串），POSIX API 提供了`asctime()`函数，形式如下：

```cpp
char* asctime( const struct tm* time_ptr );
```

`asctime()`函数的形式与`ctime()`相同，唯一的区别是主要参数是指向`tm`结构的指针，而不是`time_t`变量，如下所示：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);
    std::cout << "time: " << asctime(localtime(&t));
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 15:28:59 2018
```

如前面的例子所示，`ctime()`和`asctime(localtime())`的输出没有区别。要输出 GMT 时间而不是本地时间，使用以下方式：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);
    std::cout << "time: " << asctime(gmtime(&t));
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 21:46:12 2018
```

如前面的例子所示，`gmtime()`和`localtime()`执行相同，唯一的区别是时区的改变。

# `strftime()`函数

到目前为止，`ctime()`和`asctime()`的输出是由 POSIX API 预先确定的。也就是说，没有办法控制输出格式。此外，这些函数返回内部内存，阻止了它们的线程安全性。为了解决这些问题，POSIX API 添加了`strftime()`函数，这是将不透明的`tm`结构转换为字符串的推荐 API，形式如下：

```cpp
size_t strftime(char * str, size_t count, const char *format, const struct tm *time);
```

`str`参数接受预分配的标准 C 字符串，而`count`参数定义第一个参数的大小。`format`参数接受一个以空字符结尾的标准 C 字符串，定义要将日期和时间转换为的格式，而最终的`time`参数接受不透明的`tm`结构以转换为字符串。提供给此函数的格式字符串类似于提供给其他 POSIX 函数的格式字符串，例如`printf()`。接下来的几个示例将演示一些这些格式说明符。

为了演示`strftime()`函数，以下将当前日期输出到`stdout`：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);

    char buf[256]{};
    strftime(buf, sizeof(buf), "%m/%d/%Y", localtime(&t));

    std::cout << "time: " << buf << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 07/14/2018
```

如前面的例子所示，`time()` API 用于获取当前日期和时间。`localtime()`函数用于将`time()`的结果（即`time_t`）转换为表示本地日期和时间的不透明`tm`结构。得到的`tm`结构传递给`strftime()`，格式字符串为`"%m/%d/%Y"`，将*月/日/年*输出到提供的标准 C 字符串。最后，将此字符串输出到`stdout`，结果为`07/14/2018`。

同样，此函数可用于输出当前时间：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);

    char buf[256]{};
    strftime(buf, sizeof buf, "%H:%M", localtime(&t));

    std::cout << "time: " << buf << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 15:41
```

前面的例子与上一个例子相同，唯一的区别是格式说明符是`％H：％M`，表示`小时：分钟`，结果为`15:41`。

最后，要输出与`ctime()`和`asctime()`相同的字符串，请使用以下示例：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t = time(nullptr);

    char buf[256]{};
    strftime(buf, sizeof buf, "%a %b %d %H:%M:%S %Y", localtime(&t));

    std::cout << "time: " << buf << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 15:44:57 2018
```

前面的例子与前两个例子相同，唯一的区别是格式说明符为`"%a %b %d %H:%M:%S %Y"`，输出与`ctime()`和`asctime()`相同的结果。

# difftime()函数

从技术上讲，`time_t` typedef 被认为是不透明的（尽管在 Unix 系统上几乎总是一个带符号的 32 位整数）。因此，为了确定两个`time_t`值之间的差异，提供了`difftime()`函数，如下所示：

```cpp
double difftime(time_t time_end, time_t time_beg);
```

`difftime()`函数接受两个`time_t`值，并将差异作为双精度返回（因为非 POSIX 函数可能支持分数时间）：

```cpp
#include <ctime>
#include <iostream>

#include <unistd.h>

int main()
{
    auto t1 = time(nullptr);
    sleep(2);
    auto t2 = time(nullptr);

    std::cout << "diff: " << difftime(t2, t1) << '\n';
    std::cout << "diff: " << t2 - t1 << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// diff: 2
```

如前面的例子所示，`difftime()`函数返回两个时间之间的差异。应该注意的是，尽管前面的代码在大多数系统上都可以编译，但应该使用`difftime()`而不是直接减去两个值的第二个示例。

# mktime()函数

如果您有两个不透明的`tm`结构，并希望计算它们的差异怎么办？问题在于`difftime()`函数只接受`time_t`而不是`tm`结构。为了支持`localtime()`和`gmtime()`函数的反向操作，它们将`time_t`转换为`tm`结构，`mktime()`函数将`tm`结构转换回`time_t`值，如下所示：

```cpp
time_t mktime(struct tm *time);
```

`mktime()`函数接受一个参数，即您希望将其转换为`time_t`值的不透明`tm`结构：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    auto t1 = time(nullptr);
    auto lt = localtime(&t1);
    auto t2 = mktime(lt);

    std::cout << "time: " << ctime(&t2);
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 16:00:13 2018
```

前面的例子使用`time()` API 获取当前时间和日期，并使用`localtime()` API 将结果转换为`tm`结构。然后将得到的`tm`结构转换回`time_t`值，使用`mktime()`输出结果到`stdout`使用`ctime()`。

# clock()函数

到目前为止，`time()`已用于获取当前系统日期和时间。这种类型的时钟的问题在于它返回操作系统管理的与当前日期和时间相关的值，这可以在任何时间点发生变化（例如，用户可能在不同时区之间飞行）。例如，如果您使用时间 API 来跟踪某个操作执行了多长时间，这可能是一个问题。在这种情况下，当时区发生变化时，使用`time()`的应用程序可能会记录经过的时间为负数。

为了解决这个问题，POSIX 提供了`clock()`函数，如下所示：

```cpp
clock_t clock(void);
```

`clock()` API 返回一个`clock_t`值，它类似于`time_t`值。`time()`和`clock()`之间的区别在于，`time()`返回当前系统时间，而`clock()`返回一个代表自应用程序启动以来经过的总时间的值，例如：

```cpp
#include <ctime>
#include <iostream>

int main()
{
    std::cout << "clock: " << clock() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// clock: 2002
```

在上面的例子中，`clock()`的结果输出到`stdout`。如图所示，该值是特定于实现的，只有两个`clock_t`值之间的差异才有意义。要将`clock_t`转换为秒，POSIX 提供了`CLOCKS_PER_SEC`宏，它提供了必要的转换，如下例所示：

```cpp
#include <ctime>
#include <iostream>

#include <unistd.h>

int main()
{
    auto c1 = clock();
    sleep(2);
    auto c2 = clock();

    std::cout << "clock: " <<
        static_cast<double>(c2 - c1) / CLOCKS_PER_SEC << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// clock: 3.2e-05
```

在上面的例子中，使用`clock()`API 获取第一个时钟值，然后应用程序睡眠两秒。一旦操作系统再次执行应用程序，就会再次读取时钟值，并将差异转换为毫秒，使用`CLOCKS_PER_SEC`（然后乘以 1,000）。请注意，该值不等于 2,000 毫秒。这是因为应用程序在睡眠时不记录执行时间，因此`clock()`只能看到应用程序的执行时间。

为了更好地展示时间的差异，以下示例演示了`clock()`和`time()`的一对一比较：

```cpp
#include <ctime>
#include <iostream>

#include <unistd.h>

int main()
{
    auto c1 = clock();

    auto t1 = time(nullptr);
    while(time(nullptr) - t1 <= 2);

    auto c2 = clock();

    std::cout << "clock: " <<
        static_cast<double>(c2 - c1) / CLOCKS_PER_SEC << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// clock: 2.05336
```

上面的例子与前面的例子相同，唯一的区别是我们使用`time()`旋转两秒，而不是睡眠两秒，导致`clock()`返回两秒。

# 探索 C++ Chrono API

C++包括 Chrono API，大多数情况下提供了对 POSIX `time.h` API 的 C++包装。因此，仍然需要一些 time.h 函数来提供完整的功能，包括转换为标准 C 字符串。值得注意的是，尽管在 C++17 中进行了一些添加（特别是`floor()`、`ceil()`和`round()`），但随着 C++20 的引入，Chrono API 预计会进行相当大的改进，这超出了本书的范围。因此，本节简要解释了 C++ Chrono API，以提供当前 API 的概述。

# system_clock() API

`std::chrono::system_clock{}` API 类似于`time()`，它能够获取系统时钟。`system_clock{}`也是唯一能够转换为`time_t`的时钟（因为它很可能是使用`time()`实现的），如下例所示：

```cpp
#include <chrono>
#include <iostream>

int main()
{
    auto t = std::chrono::system_clock::now();
    std::cout << "time: " << std::chrono::system_clock::to_time_t(t) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 1531606644
```

在上面的例子中，使用`system_clock::now()`API 读取当前系统时钟，并使用`system_clock::to_time_t()`API 将结果转换为`time_t`值。与前面的例子一样，结果是从 Unix 纪元开始的秒数。

# time_point API

`system_clock::now()` API 的结果是一个`time_point{}`。C++没有提供将`time_point{}`转换为字符串的函数（直到 C++20 才会提供），因此仍然需要使用前面讨论过的 POSIX 函数来执行这种转换，如下所示：

```cpp
#include <chrono>
#include <iostream>

template<typename C, typename D>
std::ostream &
operator<<(std::ostream &os, std::chrono::time_point<C,D> &obj)
{
    auto t = std::chrono::system_clock::to_time_t(obj);
    return os << ctime(&t);
}

int main()
{
    auto now = std::chrono::system_clock::now();
    std::cout << "time: " << now;
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: Sat Jul 14 19:01:55 2018
```

在上面的例子中，我们首先为`std::chrono::system_clock::now()`API 的结果`time_point{}`定义了一个用户定义的重载。这个用户定义的重载使用 C++的`std::chrono::system_clock::to_time_t()`API 将`time_point{}`转换为`time_t`值，然后使用`ctime()`将`time_t`转换为标准 C 字符串，并将结果流式输出到`stdout`。

与 POSIX `time.h` API 不同，Chrono 库提供了各种函数来使用 C++运算符重载对`time_point{}`进行递增、递减和比较，如下所示：

```cpp
#include <chrono>
#include <iostream>

template<typename C, typename D>
std::ostream &
operator<<(std::ostream &os, const std::chrono::time_point<C,D> &obj)
{
    auto t = std::chrono::system_clock::to_time_t(obj);
    return os << ctime(&t);
}

int main()
{
    using namespace std::chrono;

    auto now = std::chrono::system_clock::now();

    std::cout << "time: " << now;

    now += 1h;
    std::cout << "time: " << now;

    now -= 1h;
    std::cout << "time: " << now;
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 1531606644
```

在上面的例子中，提供了`time_point{}`的用户定义重载，与前面的例子一样。使用`std::chrono::system_clock::now()`读取当前日期和时间，并将结果输出到`stdout`。最后，将得到的`time_point{}`增加一个小时，然后减少一个小时（使用小时字面量），并将结果也输出到`stdout`。

此外，还支持算术比较，如下所示：

```cpp
#include <chrono>
#include <iostream>

int main()
{
    auto now1 = std::chrono::system_clock::now();
    auto now2 = std::chrono::system_clock::now();

    std::cout << std::boolalpha;
    std::cout << "compare: " << (now1 < now2) << '\n';
    std::cout << "compare: " << (now1 > now2) << '\n';
    std::cout << "compare: " << (now1 <= now2) << '\n';
    std::cout << "compare: " << (now1 >= now2) << '\n';
    std::cout << "compare: " << (now1 == now2) << '\n';
    std::cout << "compare: " << (now1 != now2) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// compare: true
// compare: false
// compare: true
// compare: false
// compare: false
// compare: true
```

在上面的例子中，系统时钟被读取两次，然后使用支持的比较运算符比较得到的`time_point{}`值。应该注意，这个例子的结果可能因执行代码的系统不同而不同，因为时间的分辨率可能不同。

# 持续时间

`time_point{}`类型提供了增加、减少、执行加法和减法的算术运算。所有这些算术运算都是使用 C++ Chrono `duration{}`完成的，它定义了一段时间。另一种看待`duration{}`的方式是它将是 POSIX `difftime()`调用的结果抽象。事实上，两个`time_point{}`类型的减法结果是一个`duration{}`。

在前面的例子中，`time_point{}`使用*小时*持续时间字面量增加和减少了一个小时。与小时字面量类似，C++还为时间持续时间提供了以下字面量，可用于此类算术运算：

+   **小时**：*h*

+   **分钟**：*min*

+   **秒**：*s*

+   **毫秒**：*ms*

+   **微秒**：*us*

+   **纳秒**：*ns*

持续时间具有相对复杂的模板结构，超出了本书的范围，用于定义它们的分辨率（即持续时间是以秒、毫秒还是小时为单位），并且在技术上可以以几乎任何分辨率进行。尽管存在这种功能，但 C++提供了一些预定义的辅助程序，用于将一种持续时间转换为另一种，从而避免您需要了解`duration{}`的内部工作方式：

+   `std::chrono::nanoseconds`

+   `std::chrono::microseconds`

+   `std::chrono::milliseconds`

+   `std::chrono::seconds`

+   `std::chrono::minutes`

+   `std::chrono::hours `

例如，下面我们将使用这些预定义的辅助程序将系统时钟转换为秒和毫秒：

```cpp
#include <chrono>
#include <iostream>

#include <unistd.h>

int main()
{
    using namespace std::chrono;

    auto now1 = system_clock::now();
    sleep(2);
    auto now2 = system_clock::now();

    std::cout << "time: " <<
        duration_cast<seconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<milliseconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<nanoseconds>(now2 - now1).count() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 2
// time: 2001
// time: 2001415132
```

在上面的例子中，系统时钟被读取两次，每次读取之间间隔两秒的睡眠。然后将得到的`time_point{}`值相减以创建一个`duration{}`，并将得到的`duration{}`转换为秒、毫秒和纳秒，结果使用`count()`成员函数输出到`stdout`，该函数简单地返回`duration{}`的值。

与`time_point{}`一样，持续时间也可以使用算术运算进行操作，如下所示：

```cpp
#include <chrono>
#include <iostream>

int main()
{
    using namespace std::chrono;

    seconds t(42);

    t++;
    std::cout << "time: " << t.count() << '\n';

    t--;
    std::cout << "time: " << t.count() << '\n';

    t += 1s;
    std::cout << "time: " << t.count() << '\n';

    t -= 1s;
    std::cout << "time: " << t.count() << '\n';

    t %= 2s;
    std::cout << "time: " << t.count() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 43
// time: 42
// time: 43
// time: 42
// time: 0
```

在上面的例子中，创建了两个代表一秒的`duration{}`变量，一个值为`0`秒，另一个值为`42`秒。然后对第一个持续时间进行算术运算，并将结果输出到`stdout`。

此外，还支持比较：

```cpp
#include <chrono>
#include <iostream>

int main()
{
    using namespace std::chrono;

    auto t1 = 0s;
    auto t2 = 42s;

    std::cout << std::boolalpha;
    std::cout << "compare: " << (t1 < t2) << '\n';
    std::cout << "compare: " << (t1 > t2) << '\n';
    std::cout << "compare: " << (t1 <= t2) << '\n';
    std::cout << "compare: " << (t1 >= t2) << '\n';
    std::cout << "compare: " << (t1 == t2) << '\n';
    std::cout << "compare: " << (t1 != t2) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// compare: true
// compare: false
// compare: true
// compare: false
// compare: false
// compare: true
```

在上面的例子中，创建了两个分别代表`0`秒和`42`秒的持续时间，并使用比较运算符进行比较。

大多数对 Chrono 库的修改可能会在 C++20 中进行，大量的 API 将被添加以解决现有 API 的明显缺陷。然而，在 C++17 中，`floor()`、`ceil()`、`round()`和`abs()` API 被添加到了 Chrono API 中，它们返回持续时间的 floor、ceil、round 或绝对值，如下例所示（类似的 API 也被添加到了`time_point{}`类型中）：

```cpp
#include <chrono>
#include <iostream>

int main()
{
    using namespace std::chrono;

    auto s1 = -42001ms;

    std::cout << "floor: " << floor<seconds>(s1).count() << '\n';
    std::cout << "ceil: " << ceil<seconds>(s1).count() << '\n';
    std::cout << "round: " << round<seconds>(s1).count() << '\n';
    std::cout << "abs: " << abs(s1).count() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// floor: -43
// ceil: -42
// round: -42
// abs: 42001
```

# 稳定时钟函数

`system_clock{}`类似于`time()`，而`steady_clock{}`类似于`clock()`，并且执行相同的目标——提供一个代表应用程序执行时间的时钟，而不考虑当前系统日期和时间（这可能会根据系统用户而改变）；例如：

```cpp
#include <chrono>
#include <iostream>

#include <unistd.h>

int main()
{
    using namespace std::chrono;

    auto now1 = steady_clock::now();
    sleep(2);
    auto now2 = steady_clock::now();

    std::cout << "time: " <<
        duration_cast<seconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<milliseconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<nanoseconds>(now2 - now1).count() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 2
// time: 2001
// time: 2001447628
```

在上面的示例中，`steady_clock::now()`函数被调用两次，两次调用之间有一个睡眠。然后将得到的值相减，转换为秒、毫秒和纳秒，并将结果输出到`stdout`。需要注意的是，与`clock()`不同，得到的稳定时钟考虑了应用程序休眠的时间。

# 高分辨率时钟函数

在大多数系统上，`high_resolution_clock{}`和`steady_clock{}`是相同的。一般来说，`high_resolution_clock{}`代表最高分辨率的稳定时钟，并且如下例所示，与`stead_clock{}`的结果相同：

```cpp
#include <chrono>
#include <iostream>

#include <unistd.h>

int main()
{
    using namespace std::chrono;

    auto now1 = high_resolution_clock::now();
    sleep(2);
    auto now2 = high_resolution_clock::now();

    std::cout << "time: " <<
        duration_cast<seconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<milliseconds>(now2 - now1).count() << '\n';

    std::cout << "time: " <<
        duration_cast<nanoseconds>(now2 - now1).count() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// time: 2
// time: 2000
// time: 2002297281
```

在上面的示例中，`high_resolution_clock::now()`函数被调用两次，两次调用之间有一个睡眠。然后将得到的值相减，转换为秒、毫秒和纳秒，并将结果输出到`stdout`。

# 研究读取系统时钟的示例

在这个示例中，我们将把本章学到的所有内容融入到一个简单的演示中，该演示按用户指定的间隔读取系统时钟。为了实现这一点，需要以下包含和命名空间：

```cpp
#include <chrono>
#include <iostream>

#include <gsl/gsl>

#include <unistd.h>

using namespace std::chrono;
```

与本章中的其他示例一样，提供了一个用户定义的`std::ostream{}`重载，将`time_point{}`转换为标准 C 字符串，然后将结果流式输出到`stdout`：

```cpp
template<typename C, typename D>
std::ostream &
operator<<(std::ostream &os, std::chrono::time_point<C,D> &obj)
{
    auto t = std::chrono::system_clock::to_time_t(obj);
    return os << ctime(&t);
}
```

在我们的`protected_main()`函数中（这是本书中使用的一种模式），我们按用户提供的间隔输出当前系统时间，如下所示：

```cpp
int
protected_main(int argc, char **argv)
{
    using namespace std::chrono;
    auto args = gsl::make_span(argv, argc);

    if (args.size() != 2) {
        std::cerr << "wrong number of arguments\n";
        ::exit(1);
    }

    gsl::cstring_span<> arg = gsl::ensure_z(args.at(1));

    while(true) {
        auto now = std::chrono::system_clock::now();
        std::cout << "time: " << now;

        sleep(std::stoi(arg.data()));
    }
}
```

在上面的代码中，我们将参数列表转换为`gsl::span{}`，然后确保我们提供了一个参数。如果没有提供参数，我们就退出程序。然后将参数转换为`cstring_span{}`，并启动一个无限循环。在循环中，读取系统时钟并将其输出到`stdout`，然后程序休眠用户提供的时间：

```cpp
int
main(int argc, char **argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

与我们所有的示例一样，`protected_main()`函数由`main()`函数执行，如果发生异常，`main()`函数会捕获异常。

# 编译和测试

要编译这段代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter11/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter11/CMakeLists.txt)。

有了这段代码，我们可以使用以下命令编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行这个示例，运行以下命令：

```cpp
> ./example1 2
time: Sun Jul 15 15:04:41 2018
time: Sun Jul 15 15:04:43 2018
time: Sun Jul 15 15:04:45 2018
time: Sun Jul 15 15:04:47 2018
time: Sun Jul 15 15:04:49 2018
```

如前面的片段所示，示例以两秒的间隔运行，并且应用程序每两秒将系统时钟输出到控制台。

# 研究高分辨率定时器的示例

在这个示例中，我们将使用`high_resolution_clock{}`创建一个简单的基准测试。为了实现这一点，需要以下包含和命名空间：

```cpp
#include <chrono>
#include <iostream>

#include <gsl/gsl>
```

要创建一个`benchmark`函数，我们使用以下内容：

```cpp
template<typename FUNC>
auto benchmark(FUNC func) {
    auto stime = std::chrono::high_resolution_clock::now();
    func();
    auto etime = std::chrono::high_resolution_clock::now();

    return etime - stime;
}
```

这个函数在第八章中已经见过，*学习文件输入/输出编程*，日志示例。这段代码利用函数式编程将一个函数调用（可能是一个 lambda）包装在两次高分辨率时钟调用之间。然后相减并返回结果。正如我们在本章中学到的，`high_resolution_clock{}`返回一个`time_point{}`，它们的差值创建一个`duration{}`。

`protected_main()`函数的实现如下：

```cpp
int
protected_main(int argc, char **argv)
{
    using namespace std::chrono;

    auto args = gsl::make_span(argv, argc);

    if (args.size() != 2) {
        std::cerr << "wrong number of arguments\n";
        ::exit(1);
    }

    gsl::cstring_span<> arg = gsl::ensure_z(args.at(1));

    auto d = benchmark([&arg]{
        for (uint64_t i = 0; i < std::stoi(arg.data()); i++);
    });

    std::cout << "time: " <<
        duration_cast<seconds>(d).count() << '\n';

    std::cout << "time: " <<
        duration_cast<milliseconds>(d).count() << '\n';

    std::cout << "time: " <<
        duration_cast<nanoseconds>(d).count() << '\n';
}
```

在上述代码中，我们将参数列表转换为`gsl::span{}`，然后检查确保我们得到了一个参数。如果没有提供参数，我们就退出程序。然后将参数转换为`cstring_span{}`，并对用户希望运行的时间进行基准测试。基准测试的结果然后转换为秒、毫秒和纳秒，并输出到`stdout`：

```cpp
int
main(int argc, char **argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

与我们所有的示例一样，`protected_main()`函数由`main()`函数执行，如果发生异常，`main()`函数会捕获异常。

# 编译和测试

为了编译这段代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter11/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter11/CMakeLists.txt)。

有了这段代码，我们可以使用以下方法编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行这个示例，运行以下命令：

```cpp
> ./example2 1000000
time: 0
time: 167
time: 167455690
```

如前面的片段所示，示例是通过一个循环运行的，循环次数为`1000000`，并且执行该循环所需的时间被输出到控制台。

# 总结

在本章中，我们学习了如何使用 POSIX 和 C++时间接口来读取系统时钟，以及使用稳定时钟进行更精确的计时。本章以两个示例结束；第一个示例演示了如何读取系统时钟并在用户定义的间隔内将结果输出到控制台，第二个示例演示了如何使用 C++高分辨率计时器对软件进行基准测试。在下一章中，我们将学习如何使用 POSIX 和 C++线程，并且会通过本章所学的知识构建示例。

在下一章中，我们将讨论 C++线程、互斥锁等同步原语，以及如何对它们进行编程。

# 问题

1.  Unix 纪元是什么？

1.  `time_t`通常表示什么类型？

1.  `time()`和`clock()`之间有什么区别？

1.  为什么`difftime()`返回一个 double？

1.  C++ `duration{}`是什么？

1.  `steady_clock{}`和`high_resolution_clock{}`之间有什么区别？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
