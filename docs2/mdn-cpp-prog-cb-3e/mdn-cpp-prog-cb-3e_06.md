

# 通用工具

标准库包含许多通用工具和库，这些工具和库超出了上一章中讨论的容器、算法和迭代器。本章重点介绍三个领域：用于处理日期、时间、日历和时区的 `chrono` 库；类型特性，它提供有关其他类型的元信息；以及标准库中新版本中的实用类型，包括 C++17 中的 `std::any`、`std::optional` 和 `std::variant`，C++20 中的 `std::span` 和 `std::source_location`，以及 C++23 中的 `std::mdspan` 和 `std::expected`。

本章包含的食谱如下：

+   使用 `chrono::duration` 表达时间间隔

+   与日历一起工作

+   在时区之间转换时间

+   使用标准时钟测量函数执行时间

+   为自定义类型生成哈希值

+   使用 `std::any` 存储任何值

+   使用 `std::optional` 存储可选值

+   连接可能或可能不产生值的计算

+   使用 `std::variant` 作为类型安全的联合体

+   访问 `std::variant`

+   使用 `std::expected` 返回值或错误

+   使用 `std::span` 处理对象的连续序列

+   使用 `std::mdspan` 处理对象序列的多维视图

+   注册一个在程序正常退出时调用的函数

+   使用类型特性查询类型的属性

+   编写自己的类型特性

+   使用 `std::conditional` 在类型之间进行选择

+   使用 `source_location` 提供日志细节

+   使用 `stacktrace` 库打印调用序列

本章的第一部分重点介绍 `chrono` 库，它提供了时间和日期工具。

# 使用 `chrono::duration` 表达时间间隔

无论编程语言如何，处理时间和日期都是一项常见操作。C++11 提供了一个灵活的日期和时间库作为标准库的一部分，使我们能够定义时间点和时间间隔。这个名为 `chrono` 的库是一个通用工具库，旨在与不同系统上可能不同的计时器和时钟一起工作，因此是精度中立的。该库在 `<chrono>` 头文件中的 `std::chrono` 命名空间中可用，并定义和实现了以下组件：

+   *持续时间*，表示时间间隔

+   *时间点*，表示自时钟纪元以来的时间长度

+   *时钟*，定义了一个纪元（即时间的开始）和一个滴答

在本食谱中，我们将学习如何处理持续时间。

## 准备工作

本食谱并非 `duration` 类的完整参考。建议您咨询其他资源以获取相关信息（库参考文档可在 [`en.cppreference.com/w/cpp/chrono`](http://en.cppreference.com/w/cpp/chrono) 获取）。

在 `chrono` 库中，时间间隔由 `std::chrono::duration` 类表示。

## 如何做到这一点...

要处理时间间隔，请使用以下方法：

+   `std::chrono::duration` 的小时、分钟、秒、毫秒、微秒和纳秒类型别名：

    ```cpp
    std::chrono::hours        half_day(12);
    std::chrono::minutes      half_hour(30);
    std::chrono::seconds      half_minute(30);
    std::chrono::milliseconds half_second(500);
    std::chrono::microseconds half_millisecond(500);
    std::chrono::nanoseconds  half_microsecond(500); 
    ```

+   使用 C++14 中可用的标准用户定义字面量运算符，在 `std::chrono_literals` 命名空间中创建小时、分钟、秒、毫秒、微秒和纳秒的持续时间：

    ```cpp
    using namespace std::chrono_literals;
    auto half_day         = 12h;
    auto half_hour        = 30min;
    auto half_minute      = 30s;
    auto half_second      = 500ms;
    auto half_millisecond = 500us;
    auto half_microsecond = 500ns; 
    ```

+   使用从低精度持续时间到高精度持续时间的直接转换：

    ```cpp
    std::chrono::hours half_day_in_h(12);
    std::chrono::minutes half_day_in_min(half_day_in_h);
    std::cout << half_day_in_h.count() << "h" << '\n';    //12h
    std::cout << half_day_in_min.count() << "min" << '\n';//720min 
    ```

+   使用 `std::chrono::duration_cast` 将高精度持续时间转换为低精度持续时间：

    ```cpp
    using namespace std::chrono_literals;
    auto total_seconds = 12345s;
    auto hours =
      std::chrono::duration_cast<std::chrono::hours>(total_seconds);
    auto minutes =
      std::chrono::duration_cast<std::chrono::minutes>(total_seconds % 1h);
    auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(total_seconds % 1min);
    std::cout << hours.count()   << ':'
              << minutes.count() << ':'
              << seconds.count() << '\n'; // 3:25:45 
    ```

+   在 C++17 中，当需要四舍五入时，使用 `std::chrono` 命名空间中可用的 `floor()`、`round()` 和 `ceil()` 转换函数（不要与 `<cmath>` 头文件中的 `std::floor()`、`std::round()` 和 `std::ceil()` 函数混淆）：

    ```cpp
    using namespace std::chrono_literals;
    auto total_seconds = 12345s;
    auto m1 = std::chrono::floor<std::chrono::minutes>(total_seconds); 
    // 205 min
    auto m2 = std::chrono::round<std::chrono::minutes>(total_seconds); 
    // 206 min
    auto m3 = std::chrono::ceil<std::chrono::minutes>(total_seconds); 
    // 206 min
    auto sa = std::chrono::abs(total_seconds); 
    ```

+   使用算术运算、复合赋值和比较运算来修改和比较时间间隔：

    ```cpp
    using namespace std::chrono_literals;
    auto d1 = 1h + 23min + 45s; // d1 = 5025s
    auto d2 = 3h + 12min + 50s; // d2 = 11570s
    if (d1 < d2) { /* do something */ } 
    ```

## 它是如何工作的...

`std::chrono::duration` 类定义了时间单位上的多个刻度（两个时间点之间的增量）。默认单位是秒，对于表示其他单位，如分钟或毫秒，我们需要使用一个比率。对于大于秒的单位，比率大于一，例如 `ratio<60>` 用于分钟。对于小于秒的单位，比率小于一，例如 `ratio<1, 1000>` 用于毫秒。刻度的数量可以通过 `count()` 成员函数检索。

标准库为纳秒、微秒、毫秒、秒、分钟和小时的持续时间定义了几个类型别名，我们在上一节的第一例中使用了这些别名。以下代码显示了这些持续时间在 `chrono` 命名空间中的定义：

```cpp
namespace std {
  namespace chrono {
    typedef duration<long long, ratio<1, 1000000000>> nanoseconds;
    typedef duration<long long, ratio<1, 1000000>> microseconds;
    typedef duration<long long, ratio<1, 1000>> milliseconds;
    typedef duration<long long> seconds;
    typedef duration<int, ratio<60> > minutes;
    typedef duration<int, ratio<3600> > hours;
  }
} 
```

然而，有了这种灵活的定义，我们可以表示像 *1.2 六分之一的分钟*（这意味着 12 秒）这样的时间间隔，其中 1.2 是持续时间的刻度数，`ratio<10>`（如 60/6）是时间单位：

```cpp
std::chrono::duration<double, std::ratio<10>> d(1.2); // 12 sec 
```

在 C++14 中，`std::chrono_literals` 命名空间中添加了几个标准用户定义字面量运算符。这使得定义持续时间变得更容易，但您必须将命名空间包含在您想要使用字面量运算符的作用域中。

您应该只将用户定义字面量运算符的作用域包含在您想要使用它们的作用域中，而不是在更大的作用域中，以避免与其他库和命名空间中具有相同名称的运算符冲突。

所有算术运算都适用于 `duration` 类。可以添加和减去持续时间，将它们乘以或除以一个值，或者应用 `modulo` 操作。然而，需要注意的是，当两个不同时间单位的持续时间相加或相减时，结果是这两个时间单位最大公约数的持续时间。这意味着如果您将表示秒的持续时间和表示分钟的持续时间相加，结果是表示秒的持续时间。

从具有较不精确时间单位的持续时间到具有更精确时间单位的持续时间的转换是隐式进行的。另一方面，从更精确的时间单位到较不精确的时间单位的转换需要一个显式的转换。这可以通过非成员函数 `std::chrono::duration_cast()` 来完成。在 *如何做...* 部分中，您看到了一个确定以秒为单位的给定持续时间的小时、分钟和秒数的示例。

C++17 添加了几个更多的非成员转换函数，它们执行带有舍入的持续时间转换：`floor()` 用于向下舍入，`ceil()` 用于向上舍入，`round()` 用于四舍五入到最接近的。此外，C++17 还添加了一个名为 `abs()` 的非成员函数，用于保留持续时间的绝对值。

## 更多内容...

在 C++20 之前，`chrono` 是一个通用库，它缺乏许多有用的功能，例如使用年、月和日部分表达日期，处理时区和日历，以及其他功能。C++20 标准添加了对日历和时区的支持，我们将在下面的食谱中看到。如果您使用不支持这些 C++20 新增功能的编译器，那么第三方库可以实现这些功能，其中一个推荐的是 Howard Hinnant 的 `date` 库，可在 MIT 许可证下在 [`github.com/HowardHinnant/date`](https://github.com/HowardHinnant/date) 找到。这个库是 C++20 `chrono` 新增功能的基础。

## 参见

+   *使用标准时钟测量函数执行时间*，了解您如何确定函数的执行时间

+   *使用日历*，发现 C++20 对 `chrono` 库在处理日期和日历方面的新增功能

+   *在时区之间转换时间*，了解如何在 C++20 中转换不同时区的时间点

# 使用日历

C++11 中可用的 `chrono` 库提供了对时钟、时间点和持续时间的支持，但并没有使表达时间和日期变得容易，尤其是在日历和时区方面。新的 C++20 标准通过扩展现有的 `chrono` 库来纠正这一点，包括：

+   更多时钟，例如 UTC 时钟、国际原子时时钟、GPS 时钟、文件时间时钟以及表示本地时间的伪时钟。

+   白天时间，表示从午夜开始经过的小时、分钟和秒。

+   日历，它使我们能够使用年、月和日部分来表达日期。

+   时区，它使我们能够根据时区表达时间点，并使在不同时区之间转换时间成为可能。

+   对从流中解析 chrono 对象的 I/O 支持。

在本食谱中，我们将学习如何使用日历对象。

## 准备工作

所有新的 chrono 功能都可在 `<chrono>` 头文件中的相同 `std::chrono` 和 `std::chrono_literals` 命名空间中找到。

## 如何做…

您可以使用 C++20 的 chrono 日历功能来：

+   使用 `year_month_day` 类型的实例表示格里高利日历日期。使用标准用户定义的文法、常量和重载的运算符 `/` 来构造这样的对象：

    ```cpp
    // format: year / month /day
    year_month_day d1 = 2024y / 1 / 15;
    year_month_day d2 = 2024y / January / 15;
    // format: day / month / year
    year_month_day d3 = 15d / 1 / 2024;
    year_month_day d4 = 15d / January / 2024
    // format: month / day / year
    year_month_day d5 = 1 / 15d / 2024;
    year_month_day d6 = January / 15 / 2024; 
    ```

+   使用 `year_month_weekday` 类型的实例表示特定年份和月份的第 *n* 个工作日：

    ```cpp
    // format: year / month / weekday
    year_month_weekday d1 = 2024y / January / Monday[1];
    // format: weekday / month / year
    year_month_weekday d2 = Monday[1] / January / 2024;
    // format: month / weekday / year
    year_month_weekday d3 = January / Monday[1] / 2024; 
    ```

+   确定当前日期，并从中计算其他日期，例如明天和昨天的日期：

    ```cpp
    auto today = floor<days>(std::chrono::system_clock::now());
    auto tomorrow = today + days{ 1 };
    auto yesterday = today - days{ 1 }; 
    ```

+   确定特定月份和年份的第一天和最后一天：

    ```cpp
    year_month_day today = floor<days>(std::chrono::system_clock::now());
    year_month_day first_day_this_month = today.year() / today.month() / 1;
    year_month_day last_day_this_month = today.year() / today.month() / last; // std::chrono::last
    year_month_day last_day_feb_2024 = 2024y / February / last;
    year_month_day_last ymdl {today.year(), month_day_last{ month{ 2 } }};
    year_month_day last_day_feb { ymdl }; 
    ```

+   计算两个日期之间的天数：

    ```cpp
    inline int number_of_days(std::chrono::sys_days const& first,
                              std::chrono::sys_days const& last)
    {
      return (last - first).count();
    }
    auto days = number_of_days(2024y / April / 1,
      2024y / December / 25); 
    ```

+   检查一个日期是否有效：

    ```cpp
    auto day = 2024y / January / 33;
    auto is_valid = day.ok(); 
    ```

+   使用 `hh_mm_ss<Duration>` 类模板以小时、分钟和秒表示一天中的时间，其中 `Duration` 决定了分割时间间隔的精度。在下一个示例中，`std::chrono::seconds` 定义了 1 秒的分割精度：

    ```cpp
    chrono::hh_mm_ss<chrono::seconds> td(13h+12min+11s);
    std::cout << td << '\n';  // 13:12:11 
    ```

+   创建包含日期和时间部分的时点：

    ```cpp
    auto tp = chrono::sys_days{ 2024y / April / 1 } + 12h + 30min + 45s;
    std::cout << tp << '\n';  // 2024-04-01 12:30:45 
    ```

+   确定当前一天的时间并使用各种精度表示它：

    ```cpp
    auto tp = std::chrono::system_clock::now();
    auto dp = floor<days>(tp);
    chrono::hh_mm_ss<chrono::milliseconds> time1 {
      chrono::duration_cast<chrono::milliseconds>(tp - dp) };
    std::cout << time1 << '\n';  // 13:12:11.625
    chrono::hh_mm_ss<chrono::minutes> time2 {
      chrono::duration_cast<chrono::minutes>(tp - dp) };
    std::cout << time2 << '\n';  // 13:12 
    ```

## 它是如何工作的…

在这里示例中看到的 `year_month_day` 和 `year_month_weekday` 类型只是添加到 `chrono` 库以支持日历的许多新类型中的一部分。以下表格列出了 `std::chrono` 命名空间中的所有这些类型以及它们所表示的内容：

| **类型** | **表示** |
| --- | --- |
| `day` | 一个月中的某一天 |
| `month` | 年份中的月份 |
| `year` | 格里高利日历中的年份 |
| `weekday` | 格里高利日历中的星期几 |
| `weekday_indexed` | 一个月中的第 *n* 个工作日，其中 *n* 在范围 [1, 5] 内（1 是月份的第一天，5 是月份的第 5 天——如果存在的话） |
| `weekday_last` | 一个月中的最后一个工作日 |
| `month_day` | 特定月份的特定一天 |
| `month_day_last` | 特定月份的最后一天 |
| `month_weekday` | 特定月份的第 *n* 个工作日 |
| `month_weekday_last` | 特定月份的最后一个工作日 |
| `year_month` | 特定年份的特定月份 |
| `year_month_day` | 特定年份、月份和日期 |
| `year_month_day_last` | 特定年份和月份的最后一天 |
| `year_month_weekday` | 特定年份和月份的第 *n* 个工作日 |
| `year_month_weekday_last` | 特定年份和月份的最后一个工作日 |

表 6.1：C++20 用于处理日期的 chrono 类型

表格中列出的所有类型都具有：

+   默认构造函数，该构造函数将成员字段初始化为未初始化状态

+   成员函数用于访问实体的各个部分

+   一个名为 `ok()` 的成员函数，用于检查存储的值是否有效

+   非成员比较运算符，用于比较该类型值的比较

+   一个重载的 `operator<<` 运算符，用于将类型的值输出到流中

+   一个重载的函数模板 `from_stream()`，它根据提供的格式从流中解析值

+   为文本格式化库的 `std::formatter<T, CharT>` 类模板进行特化

此外，这些类型的许多操作符被重载，以便我们能够轻松创建格里高利日历日期。当创建日期（包含年、月和日）时，您可以选择三种不同的格式：

+   **年/月/日**（在中国、日本、韩国、加拿大等国家使用，但还有其他国家，有时与月/日/年格式一起使用）

+   **月/日/年**（在美国使用）

+   **月/日/年**（在世界上大多数地区使用）

在这些情况下，**日**可以是：

+   实际的月份中的某一天（值从 1 到 31）

+   `std::chrono::last`，表示月份的最后一天

+   `weekday[n]`，表示月份的第 *n* 个工作日（其中 *n* 可以取 1 到 5 的值）

+   `weekday[std::chrono::last]`，表示月份的最后一天

为了区分表示日期、月份和年份的整数，库提供了两个用户定义的文法：`""y` 用于构造 `std::chrono::year` 类型的文法，`""d` 用于构造 `std::chrono::day` 类型的文法。

此外，还有一些表示以下内容的常量：

+   `std::chrono::month`，命名为 `January`、`February` 一直到 `December`。

+   `std::chrono::weekday`，命名为 `Sunday`、`Monday`、`Tuesday`、`Wednesday`、`Thursday`、`Friday` 或 `Saturday`。

您可以使用所有这些来构造日期，例如 `2025y/April/1`、`25d/December/2025` 或 `Sunday[last]/May/2025`。

`year_month_day` 类型提供了到 `std::chrono::sys_days` 的隐式转换。这种类型是一个精度为一天的 `std::chrono::time_point`。有一个伴随类型称为 `std::chrono::sys_seconds`，它是一个精度为一秒的 `time_point`。可以使用 `std::chrono::time_point_cast()` 或 `std::chrono::floor()` 来执行 `time_point` 和 `sys_days` / `sys_seconds` 之间的显式转换。

为了表示一天中的某个时刻，我们可以使用 `std::chrono::hh_mm_ss` 类型。这个类表示自午夜以来经过的时间，分解为小时、分钟、秒和毫秒。这个类型主要用作格式化工具。

此外，还有一些实用函数用于在 12 小时/24 小时格式之间转换。这些函数包括：

+   `is_am()` 和 `is_pm()` 函数用于检查以 24 小时格式表示的时间（作为 `std::chrono::hours` 值提供）是上午（中午之前）还是下午（午夜之前）：

    ```cpp
    std::cout << is_am(0h)  << '\n'; // true
    std::cout << is_am(1h)  << '\n'; // true
    std::cout << is_am(12h) << '\n'; // false
    std::cout << is_pm(0h)  << '\n'; // false
    std::cout << is_pm(12h) << '\n'; // true
    std::cout << is_pm(23h) << '\n'; // true
    std::cout << is_pm(24h) << '\n'; // false 
    ```

+   `make12()` 和 `make24()` 函数返回 24 小时格式的 12 小时等效时间，反之亦然。它们都接受输入时间作为 `std::chrono::hours` 值，但 `make24()` 有一个额外的参数，一个布尔值，表示时间是否为下午：

    ```cpp
    for (auto h : { 0h, 1h, 12h, 23h, 24h })
    {
       std::cout << make12(h).count() << '\n';
       // prints 12, 1, 12, 11, 12
    }
    for (auto [h, pm] : { 
       std::pair<hours, bool>{ 0h, false},
       std::pair<hours, bool>{ 1h, false}, 
       std::pair<hours, bool>{ 1h, true}, 
       std::pair<hours, bool>{12h, false}, 
       std::pair<hours, bool>{12h, true}, })
    {
       std::cout << make24(h, pm).count() << '\n';
       // prints 0, 1, 13, 0, 12
    } 
    ```

如您从这些示例中看到的，这四个函数仅适用于小时值，因为只有时间点的小时部分决定了其格式为 12 小时或 24 小时，或者是否为上午或下午时间。

在本书第二版出版时，chrono 的变化尚未完成。`hh_mm_ss`类型被称为`time_of_day`，而`make12()`/`make_24()`函数是其成员。这一版反映了这些变化并利用了标准化的 API。

## 更多内容…

这里描述的日期和时间功能都是基于`std::chrono::system_clock`。自 C++20 起，此时钟被定义为测量 Unix 时间，即自 1970 年 1 月 1 日 00:00:00 UTC 以来的时间。这意味着隐含的时间区域是 UTC。然而，在大多数情况下，你可能对特定时区的地方时间感兴趣。为了帮助解决这个问题，`chrono`库增加了对时区的支持，这是我们将在下一个菜谱中学习的。

## 相关内容

+   *使用 chrono::duration 表示时间间隔*，以熟悉 C++11 `chrono`库的基本原理，并处理持续时间、时间点和时间点

+   *在不同时区之间转换时间*，了解如何在 C++20 中转换不同时区之间的时间点

# 在不同时区之间转换时间

在上一个菜谱中，我们讨论了 C++20 对处理日历以及使用`year_month_day`类型和其他来自`chrono`库的类型在格里高利历中表示日期的支持。

我们还看到了如何使用`hh_mm_ss`类型表示一天中的时间。然而，在所有这些示例中，我们使用系统时钟处理时间点，该时钟测量 Unix 时间，因此默认使用 UTC 作为时区。然而，我们通常对本地时间感兴趣，有时对其他时区的时间感兴趣。这是通过添加到`chrono`库以支持时区的功能实现的。在本菜谱中，你将了解 chrono 时区最重要的功能。

## 准备工作

在继续本菜谱之前，如果你还没有阅读，建议你阅读上一个菜谱，*处理日历*。

## 如何操作…

你可以使用 C++20 的`chrono`库执行以下操作：

+   使用`std::chrono::current_zone()`从时区数据库中检索本地时区。

+   使用`std::chrono::locate_zone()`通过其名称从时区数据库中检索特定时区。

+   使用`std::chrono::zoned_time`类模板在特定时区中表示时间点。

+   检索并显示当前本地时间：

    ```cpp
    auto time = zoned_time{ current_zone(), system_clock::now() };
    std::cout << time << '\n'; // 2024-01-16 22:10:30.9274320 EET 
    ```

+   检索并显示另一个时区的当前时间。在以下示例中，我们使用意大利的时间：

    ```cpp
    auto time = zoned_time{ locate_zone("Europe/Rome"),
                            system_clock::now() };
    std::cout << time << '\n'; // 2024-01-16 21:10:30.9291091 CET 
    ```

+   使用适当的区域设置格式显示当前本地时间。在这个例子中，当前时间是罗马尼亚时间，使用的区域设置是为罗马尼亚设计的：

    ```cpp
    auto time = zoned_time{ current_zone(), system_clock::now() };
    std::cout << std::format(std::locale{"ro_RO"}, "%c", time)
              << '\n'; // 16.01.2024 22:12:57 
    ```

+   在特定时区中表示一个时间点并显示它。在以下示例中，这是纽约的时间：

    ```cpp
    auto time = local_days{ 2024y / June / 1 } + 12h + 30min + 45s + 256ms;
    auto ny_time = zoned_time<std::chrono::milliseconds>{
                      locate_zone("America/New_York"), time};
    std::cout << ny_time << '\n';
    // 2024-06-01 12:30:45.256 EDT 
    ```

+   将特定时区的时间点转换为另一个时区的时间点。在以下示例中，我们将纽约的时间转换为洛杉矶的时间：

    ```cpp
    auto la_time = zoned_time<std::chrono::milliseconds>(
                      locate_zone("America/Los_Angeles"),
                      ny_time);
    std::cout << la_time << '\n'; // 2024-06-01 09:30:45.256 PDT 
    ```

## 工作原理…

系统维护 IANA 时间区域（TZ）数据库的副本（可在[`www.iana.org/time-zones`](https://www.iana.org/time-zones)在线找到）。作为用户，您不能创建或修改数据库，但可以使用 `std::chrono::tzdb()` 或 `std::chrono::get_tzdb_list()` 等函数检索其只读副本。时间区域的信息存储在 `std::chrono::time_zone` 对象中。这个类的实例不能直接创建；它们仅在库初始化时间区域数据库时创建。然而，可以使用两个函数获得对这些实例的常量访问：

+   `std::chrono::current_zone()` 获取表示本地时间区域的 `time_zone` 对象。

+   `std::chrono::locate_zone()` 获取表示指定时间区域的 `time_zone` 对象。

时间区域名称的示例包括 Europe/Berlin、Asia/Dubai 和 America/Los_Angeles。当位置名称包含多个单词时，空格被下划线（`_`）替换，例如在先前的示例中，洛杉矶被写作 Los_Angeles。

所有来自 IANA TZ 数据库的时间区域列表可以在[`en.wikipedia.org/wiki/List_of_tz_database_time_zones`](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)找到。

C++20 的 `chrono` 库中有两套类型来表示时间点：

+   `sys_days` 和 `sys_seconds`（具有天和秒的精度）表示系统时间区域中的时间点，该时间区域是 UTC。这些是 `std::chrono::sys_time` 的类型别名，而 `std::chrono::sys_time` 又是 `std::chrono::time_point` 的别名，它使用 `std::chrono::system_clock`。

+   `local_days` 和 `local_seconds`（也具有天和秒的精度）表示相对于尚未指定的时间区域的时间点。这些是 `std::chrono::local_time` 的类型别名，而 `std::chrono::local_time` 又是使用 `std::chrono::local_t` 伪时钟的 `std::chrono::time_point` 的类型别名。这个时钟的唯一目的是指示尚未指定的时间区域。

`std::chrono::zoned_time` 类模板表示时间区域与时间点的配对。它可以由 `sys_time`、`local_time` 或另一个 `zoned_time` 对象创建。这里展示了所有这些情况的示例：

```cpp
auto zst = zoned_time<std::chrono::seconds>(
  current_zone(),
  sys_days{ 2024y / May / 10 } +14h + 20min + 30s);
std::cout << zst << '\n'; // 2024-05-10 17:20:30 EEST (or GMT+3)
auto zlt = zoned_time<std::chrono::seconds>(
  current_zone(),
  local_days{ 2024y / May / 10 } +14h + 20min + 30s);
std::cout << zlt << '\n'; // 2024-05-10 14:20:30 EEST (or GMT+3)
auto zpt = zoned_time<std::chrono::seconds>(
  locate_zone("Europe/Paris"),
  zlt);
std::cout << zpt << '\n'; //2024-05-10 13:20:30 CEST (or GMT+2) 
```

在此示例代码中，注释中的时间基于罗马尼亚时间区域。请注意，在第一个示例中，时间使用 `sys_days` 表示，它使用 UTC 时间区域。由于罗马尼亚时间在 2024 年 5 月 10 日（因为夏令时）是 UTC+3，所以本地时间是 17:20:30。在第二个示例中，时间使用 `local_days` 指定，它是与时间区域无关的。因此，当与当前时间区域配对时，实际时间是 14:20:30。在第三个和最后一个示例中，将本地罗马尼亚时间转换为巴黎时间，巴黎时间是 13:20:30（因为那天巴黎的时间是 UTC+2）。

## 参见

+   *使用 chrono::duration 表达时间间隔*，以熟悉 C++11 `chrono` 库的基本知识，并处理持续时间、时间点和点

+   *使用日历*，以发现 C++20 对 `chrono` 库中用于处理日期和日历的添加

# 使用标准时钟测量函数执行时间

在前面的菜谱中，我们看到了如何使用 `chrono` 标准库处理时间间隔。然而，我们还需要处理时间点。`chrono` 库提供了一个这样的组件，表示自时钟纪元以来的时间长度（即，时钟定义的时间的起点）。在这个菜谱中，我们将学习如何使用 `chrono` 库和时间点来测量函数的执行时间。

## 准备工作

这个菜谱与前面的一个菜谱紧密相关，*使用 chrono::duration 表达时间间隔*。如果你之前没有完成那个菜谱，你应该在继续这个菜谱之前先完成它。

对于这个菜谱中的示例，我们将考虑以下函数，它什么也不做，只是暂停当前线程的执行给定的时间间隔：

```cpp
void func(int const interval = 1000)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(interval));
} 
```

不言而喻，这个函数仅用于测试目的，并没有做任何有价值的事情。在实际应用中，你将使用这里提供的计数工具来测试你自己的函数。

## 如何做到这一点...

要测量函数的执行时间，你必须执行以下步骤：

1.  使用标准时钟获取当前时间点：

    ```cpp
    auto start = std::chrono::high_resolution_clock::now(); 
    ```

1.  调用你想要测量的函数：

    ```cpp
    func(); 
    ```

1.  再次获取当前时间点；两个时间点之间的差值是函数的执行时间：

    ```cpp
    auto diff = std::chrono::high_resolution_clock::now() - start; 
    ```

1.  将差异（以纳秒表示）转换为你感兴趣的分辨率：

    ```cpp
    std::cout
      << std::chrono::duration<double, std::milli>(diff).count()
      << "ms" << '\n';
    std::cout
      << std::chrono::duration<double, std::nano>(diff).count()
      << "ns" << '\n'; 
    ```

要在可重用组件中实现此模式，请执行以下步骤：

1.  创建一个由分辨率和时钟参数化的类模板。

1.  创建一个静态变长函数模板，它接受一个函数及其参数。

1.  实现之前显示的模式，使用其参数调用函数。

1.  返回一个持续时间，而不是滴答数。

这在以下代码片段中得到了体现：

```cpp
template <typename Time = std::chrono::microseconds,
          typename Clock = std::chrono::high_resolution_clock>
struct perf_timer
{
  template <typename F, typename... Args>
  static Time duration(F&& f, Args... args)
  {
    auto start = Clock::now();
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    auto end = Clock::now();
    return std::chrono::duration_cast<Time>(end - start);
  }
}; 
```

## 它是如何工作的...

时钟是一个定义了两件事的组件：

+   一个称为 *纪元* 的时间起点；关于纪元没有约束，但典型的实现使用 1970 年 1 月 1 日。

+   一个 *滴答率*，它定义了两个时间点之间的增量（例如毫秒或纳秒）。

时间点是从时钟纪元以来的时间长度。有几个特别重要的时间点：

+   当前时间，由时钟的静态成员 `now()` 返回。

+   纪元，或时间的起点；这是由特定时钟的 `time_point` 的默认构造函数创建的时间点。

+   时钟可以表示的最小时间，由 `time_point` 的静态成员 `min()` 返回。

+   时钟可以表示的最大时间，由`time_point`的静态成员`max()`返回。

标准定义了几个时钟：

+   `system_clock`: 这使用当前系统的实时时钟来表示时间点。

+   `high_resolution_clock`: 这代表一个使用当前系统最短可能的滴答周期的时钟。

+   `steady_clock`: 这表示一个永远不会调整的时钟。这意味着，与其它时钟不同，随着时间的推移，两个时间点之间的差异总是正的。

+   `utc_clock`: 这是一个用于协调世界时（UTC）的 C++20 时钟。

+   `tai_clock`: 这是一个用于国际原子时（TAI）的 C++20 时钟。

+   `gps_clock`: 这是一个用于 GPS 时间的 C++20 时钟。

+   `file_clock`: 这是一个用于表示文件时间的 C++20 时钟。

以下示例打印了列表中前三个时钟（C++11 中可用的）的精度，无论它们是否稳定（或单调）：

```cpp
template <typename T>
void print_clock()
{
  std::cout << "precision: "
            << (1000000.0 * double(T::period::num)) / 
               (T::period::den)
            << '\n';
  std::cout << "steady: " << T::is_steady << '\n';
}
print_clock<std::chrono::system_clock>();
print_clock<std::chrono::high_resolution_clock>();
print_clock<std::chrono::steady_clock>(); 
```

可能的输出如下：

```cpp
precision: 0.1
steady: 0
precision: 0.001
steady: 1
precision: 0.001
steady: 1 
```

这意味着`system_clock`的分辨率为 0.1 微秒，并且不是一个单调时钟。另一方面，其它两个时钟`high_resolution_clock`和`steady_clock`都具有 1 纳秒的分辨率，并且是单调时钟。

在测量函数的执行时间时，时钟的稳定性很重要，因为如果在函数运行时调整时钟，结果将不会给出实际的执行时间，甚至可能得到负值。您应该依赖一个稳定的时钟来测量函数执行时间。典型的选择是`high_resolution_clock`，这正是我们在*如何做...*部分示例中使用的时钟。

当我们测量执行时间时，需要在调用之前和调用返回之后检索当前时间。为此，我们使用时钟的`now()`静态方法。结果是`time_point`；当我们从两个时间点中减去时，结果是`duration`，由时钟的持续时间定义。

为了创建一个可重用的组件，可以用来测量任何函数的执行时间，我们定义了一个名为`perf_timer`的类模板。这个类模板以我们感兴趣的分辨率（默认为微秒）和想要使用的时钟（默认为`high_resolution_clock`）作为参数。该类模板有一个名为`duration()`的单个静态成员——一个变参函数模板——它接受一个要执行的功能及其可变数量的参数。实现相对简单：我们检索当前时间，使用`std::invoke`调用函数（以便它处理调用任何可调用对象的不同机制），然后再次检索当前时间。返回值是一个`duration`（具有定义的分辨率）。以下代码片段展示了这个示例：

```cpp
auto t = perf_timer<>::duration(func, 1500);
std::cout << std::chrono::duration<double, std::milli>(t).count()
          << "ms" << '\n';
std::cout << std::chrono::duration<double, std::nano>(t).count()
          << "ns" << '\n'; 
```

重要的是要注意，我们不是从`duration()`函数返回滴答数，而是返回实际的`duration`值。原因是，通过返回滴答数，我们失去了分辨率，也不知道它们实际上代表什么。仅在需要实际滴答数时调用`count()`更好。这里举例说明：

```cpp
auto t1 = perf_timer<std::chrono::nanoseconds>::duration(func, 150);
auto t2 = perf_timer<std::chrono::microseconds>::duration(func, 150);
auto t3 = perf_timer<std::chrono::milliseconds>::duration(func, 150);
std::cout
  << std::chrono::duration<double, std::micro>(t1 + t2 + t3).count()
  << "us" << '\n'; 
```

在本例中，我们使用三种不同的分辨率（纳秒、微秒和毫秒）来测量三个不同函数的执行情况。`t1`、`t2`和`t3`的值代表持续时间。这使得它们可以轻松相加，并将结果转换为微秒。

## 参见

+   *使用 chrono::duration 表达时间间隔*，以便熟悉 C++11 `chrono`库的基本知识以及如何处理持续时间、时间点和点

+   *第三章*，*统一调用任何可调用对象*，学习如何使用`std::invoke()`调用函数和任何可调用对象

# 为自定义类型生成哈希值

标准库提供了几个无序关联容器：`std::unordered_set`、`std::unordered_multiset`、`std::unordered_map`和`std::unordered_map`。这些容器不按特定顺序存储它们的元素；相反，它们被分组在桶中。一个元素所属的桶取决于该元素的哈希值。这些标准容器默认使用`std::hash`类模板来计算哈希值。所有基本类型和一些库类型的特化都是可用的。然而，对于自定义类型，您必须自己特化类模板。本食谱将向您展示如何做到这一点，并解释如何计算一个好的哈希值。一个好的哈希值可以快速计算，并且在值域内均匀分布，因此最小化重复值（冲突）存在的可能性。

## 准备工作

在本食谱的示例中，我们将使用以下类：

```cpp
struct Item
{
  int         id;
  std::string name;
  double      value;
  Item(int const id, std::string const & name, double const value)
    :id(id), name(name), value(value)
  {}
  bool operator==(Item const & other) const
  {
    return id == other.id && name == other.name &&
           value == other.value;
  }
}; 
```

本食谱涵盖了标准库中的哈希功能。您应该熟悉哈希和哈希函数的概念。

## 如何操作...

为了使用自定义类型与无序关联容器一起使用，你必须执行以下步骤：

1.  为您的自定义类型特化`std::hash`类模板；特化必须在`std`命名空间中完成。

1.  定义参数和结果类型的同义词。

1.  实现调用操作符，使其接受对您的类型的常量引用并返回一个哈希值。

为了计算一个好的哈希值，您应该做以下事情：

1.  从一个初始值开始，这个值应该是一个素数（例如，17）。

1.  对于每个用于确定两个类的实例是否相等的字段，根据以下公式调整哈希值：

    ```cpp
    hashValue = hashValue * prime + hashFunc(field); 
    ```

1.  您可以使用相同的素数对所有字段使用前面的公式，但建议使用一个不同于初始值的值（例如，31）。

1.  使用`std::hash`的特化来确定类数据成员的哈希值。

根据这里描述的步骤，`Item`类的`std::hash`特化看起来如下：

```cpp
namespace std
{
  template<>
  struct hash<Item>
  {
    typedef Item argument_type;
    typedef size_t result_type;
    result_type operator()(argument_type const & item) const
 {
      result_type hashValue = 17;
      hashValue = 31 * hashValue + std::hash<int>{}(item.id);
      hashValue = 31 * hashValue + std::hash<std::string>{}(item.name);
      hashValue = 31 * hashValue + std::hash<double>{}(item.value);
      return hashValue;
    }
  };
} 
```

这个专业使您能够使用`Item`类与无序关联容器，例如`std::unordered_set`一起使用。这里提供了一个示例：

```cpp
std::unordered_set<Item> set2
{
  { 1, "one"s, 1.0 },
  { 2, "two"s, 2.0 },
  { 3, "three"s, 3.0 },
}; 
```

## 它是如何工作的...

类模板`std::hash`是一个函数对象模板，其调用操作符定义了一个具有以下属性的哈希函数：

+   接受模板参数类型的参数并返回一个`size_t`值。

+   不会抛出任何异常。

+   对于相等的两个参数，它返回相同的哈希值。

+   对于不相等的两个参数，返回相同值的概率非常小（应接近`1.0/std::numeric_limits<size_t>::max()`）。

标准为所有基本类型提供了特化，例如`bool`、`char`、`int`、`long`、`float`、`double`（以及所有可能的`unsigned`和`long`变体），以及指针类型，但也包括库类型，如`basic_string`和`basic_string_view`类型、`unique_ptr`和`shared_ptr`、`bitset`和`vector<bool>`、`optional`和`variant`（在 C++17 中），以及几种其他类型。然而，对于自定义类型，您必须提供自己的特化。这个特化必须在`std`命名空间中（因为类模板`hash`是在这个命名空间中定义的），并且必须满足前面列举的要求。

标准没有指定如何计算哈希值。只要它为相等的对象返回相同的值，并且对于不相等的对象有非常小的概率返回相同的值，您可以使用任何想要的函数。本食谱中描述的算法在 Joshua Bloch 所著的《Effective Java，第二版》一书中提出。

在计算哈希值时，仅考虑参与确定两个类的实例是否相等的字段（换句话说，用于`operator==`的字段）。但是，您必须使用与`operator==`一起使用的所有这些字段。在我们的例子中，`Item`类的所有三个字段都用于确定两个对象的相等性；因此，我们必须使用它们来计算哈希。初始哈希值不应为零，在我们的例子中，我们选择了素数 17。

重要的是，这些值不应为零；否则，产生哈希值为零的初始字段（即处理顺序中的第一个）将不会改变哈希值（由于`x * 0 + 0 = 0`，哈希值保持为零）。对于用于计算哈希的每个字段，我们通过将其前一个值乘以一个素数并加上当前字段的哈希值来改变当前哈希值。为此，我们使用类模板`std::hash`的特化。

使用素数 31 对于性能优化是有利的，因为 `31 * x` 可以被编译器替换为 `(x << 5) - x`，这更快。同样，你也可以使用 127，因为 `127 * x` 等于 `(x << 7) - x` 或 8191，因为 `8191 * x` 等于 `(x << 13) - x`。

如果你的自定义类型包含一个数组，并且用于确定两个对象的相等性，因此需要用于计算哈希，那么将数组视为其元素是类的数据成员。换句话说，将前面描述的相同算法应用于数组的所有元素。

## 参见

+   *第二章*，*数值类型的限制和其他属性*，了解数值类型的最大值和最小值，以及其他数值类型的属性

# 使用 `std::any` 存储任何值

C++ 没有像其他语言（如 C# 或 Java）那样的层次类型系统，因此它不能像 .NET 和 Java 中的 `Object` 类型或 JavaScript 中的原生类型那样在单个变量中存储多个类型的值。长期以来，开发者一直使用 `void*` 来实现这个目的，但这只能帮助我们存储指向任何东西的指针，并且不是类型安全的。根据最终目标，替代方案可以包括模板或重载函数。然而，C++17 引入了一个标准类型安全的容器，称为 `std::any`，它可以存储任何类型的单个值。

## 准备工作

`std::any` 是基于 `boost::any` 设计的，并在 `<any>` 头文件中可用。如果你熟悉 `boost::any` 并已在代码中使用它，你可以无缝地将它迁移到 `std::any`。

## 如何做...

使用以下操作来处理 `std::any`：

+   要存储值，请使用构造函数或将它们直接赋值给 `std::any` 变量：

    ```cpp
    std::any value(42); // integer 42
    value = 42.0;       // double 42.0
    value = "42"s;      // std::string "42" 
    ```

+   要读取值，请使用非成员函数 `std::any_cast()`：

    ```cpp
    std::any value = 42.0;
    try
    {
      auto d = std::any_cast<double>(value);
      std::cout << d << '\n'; // prints 42
    }
    catch (std::bad_any_cast const & e)
    {
      std::cout << e.what() << '\n';
    } 
    ```

+   要检查存储值的类型，请使用 `type()` 成员函数：

    ```cpp
    inline bool is_integer(std::any const & a)
    {
      return a.type() == typeid(int);
    } 
    ```

+   要检查容器是否存储了值，请使用 `has_value()` 成员函数：

    ```cpp
    auto ltest = [](std::any const & a) {
      if (a.has_value())
        std::cout << "has value" << '\n';
      else
        std::cout << "no value" << '\n';
    };
    std::any value;
    ltest(value); // no value
    value = 42;
    ltest(value); // has value 
    ```

+   要修改存储的值，请使用 `emplace()`、`reset()` 或 `swap()` 成员函数：

    ```cpp
    std::any value = 42;
    ltest(value); // has value
    value.reset();
    ltest(value); // no value 
    ```

## 它是如何工作的...

`std::any` 是一个类型安全的容器，可以存储任何类型（或者更确切地说，其退化类型是可复制的）的值。在容器中存储值非常简单——你可以使用可用的构造函数之一（默认构造函数创建一个不存储任何值的容器）或者赋值运算符。然而，直接读取值是不可能的，你需要使用非成员函数 `std::any_cast()`，该函数将存储的值转换为指定的类型。如果存储的值与你要转换的类型不同，该函数会抛出 `std::bad_any_cast` 异常。在隐式可转换的类型之间进行转换，例如 `int` 和 `long`，也是不可能的。`std::bad_any_cast` 是从 `std::bad_cast` 派生的；因此，你可以捕获这两种异常类型中的任何一种。

可以使用 `type()` 成员函数检查存储值的类型，它返回一个 `type_info` 常量引用。如果容器为空，此函数返回 `typeid(void)`。要检查容器是否存储了值，可以使用成员函数 `has_value()`，如果容器中有值则返回 `true`，如果容器为空则返回 `false`。

以下示例展示了如何检查容器是否有任何值，如何检查存储值的类型，以及如何从容器中读取值：

```cpp
void log(std::any const & value)
{
  if (value.has_value())
  {
    auto const & tv = value.type();
    if (tv == typeid(int))
    {
      std::cout << std::any_cast<int>(value) << '\n';
    }
    else if (tv == typeid(std::string))
    {
      std::cout << std::any_cast<std::string>(value) << '\n';
    }
    else if (tv == typeid(
      std::chrono::time_point<std::chrono::system_clock>))
    {
      auto t = std::any_cast<std::chrono::time_point<
        std::chrono::system_clock>>(value);
      auto now = std::chrono::system_clock::to_time_t(t);
      std::cout << std::put_time(std::localtime(&now), "%F %T")
                << '\n';
    }
    else
    {
      std::cout << "unexpected value type" << '\n';
    }
  }
  else
  {
    std::cout << "(empty)" << '\n';
  }
}
log(std::any{});                       // (empty)
log(42);                               // 42
log("42"s);                            // 42
log(42.0);                             // unexpected value type
log(std::chrono::system_clock::now()); // 2016-10-30 22:42:57 
```

如果你想存储任何类型的多个值，可以使用标准容器，如 `std::vector` 来持有 `std::any` 类型的值。以下是一个示例：

```cpp
std::vector<std::any> values;
values.push_back(std::any{});
values.push_back(42);
values.push_back("42"s);
values.push_back(42.0);
values.push_back(std::chrono::system_clock::now());
for (auto const & v : values)
  log(v); 
values contains elements of the std::any type, which, in turn, contains an int, std::string, double, and std::chrono::time_point value.
```

## 参见

+   *使用 `std::optional` 存储可选值*，了解 C++17 类模板 `std::optional`，它管理可能存在或可能不存在的值

+   *使用 `std::variant` 作为类型安全的联合体*，了解如何使用 C++17 的 `std::variant` 类来表示类型安全的联合体

# 使用 `std::optional` 存储可选值

有时，如果某个特定值不可用，能够存储一个值或一个空指针是有用的。这种情况的一个典型例子是函数的返回值，该函数可能无法生成返回值，但这种失败不是错误。例如，考虑一个通过指定键从字典中查找并返回值的函数。找不到值是一个可能的情况，因此该函数要么返回一个布尔值（或如果需要更多错误代码，则返回整数值），并有一个引用参数来保存返回值，要么返回一个指针（原始指针或智能指针）。在 C++17 中，`std::optional` 是这些解决方案的更好替代。类模板 `std::optional` 是一个用于存储可能存在或不存在值的模板容器。在本食谱中，我们将了解如何使用此容器及其典型用例。

## 准备工作

类模板 `std::optional<T>` 是基于 `boost::optional` 设计的，并在 `<optional>` 头文件中提供。如果你熟悉 `boost::optional` 并已在代码中使用它，你可以无缝地将它迁移到 `std::optional`。

在以下代码片段中，我们将参考以下 `foo` 类：

```cpp
struct foo
{
  int    a;
  double b;
}; 
```

## 如何做到这一点...

使用以下操作来处理 `std::optional`：

+   要存储一个值，使用构造函数或将值直接赋给 `std::optional` 对象：

    ```cpp
    std::optional<int> v1;      // v1 is empty
    std::optional<int> v2(42);  // v2 contains 42
    v1 = 42;                    // v1 contains 42
    std::optional<int> v3 = v2; // v3 contains 42 
    ```

+   要读取存储的值，使用 `operator*` 或 `operator->`：

    ```cpp
    std::optional<int> v1{ 42 };
    std::cout << *v1 << '\n';   // 42
    std::optional<foo> v2{ foo{ 42, 10.5 } };
    std::cout << v2->a << ", "
              << v2->b << '\n'; // 42, 10.5 
    ```

+   或者，使用成员函数 `value()` 和 `value_or()` 来读取存储的值：

    ```cpp
    std::optional<std::string> v1{ "text"s };
    std::cout << v1.value() << '\n'; // text
    std::optional<std::string> v2;
    std::cout << v2.value_or("default"s) << '\n'; // default 
    ```

+   要检查容器是否存储了值，可以使用转换运算符到 `bool` 或成员函数 `has_value()`：

    ```cpp
    std::optional<int> v1{ 42 };
    if (v1) std::cout << *v1 << '\n';
    std::optional<foo> v2{ foo{ 42, 10.5 } };
    if (v2.has_value())
      std::cout << v2->a << ", " << v2->b << '\n'; 
    ```

+   要修改存储的值，使用成员函数 `emplace()`、`reset()` 或 `swap()`：

    ```cpp
    std::optional<int> v{ 42 }; // v contains 42
    v.reset();                  // v is empty 
    ```

使用 `std::optional` 来模拟以下任何一种情况：

+   可能无法生成值的函数的返回值：

    ```cpp
    template <typename K, typename V>
    std::optional<V> find(K const key,
                          std::map<K, V> const & m)
    {
      auto pos = m.find(key);
      if (pos != m.end())
        return pos->second;
      return {};
    }
    std::map<int, std::string> m{
      { 1, "one"s },{ 2, "two"s },{ 3, "three"s } };
    auto value = find(2, m);
    if (value) std::cout << *value << '\n'; // two
    value = find(4, m);
    if (value) std::cout << *value << '\n'; 
    ```

+   可选的函数参数：

    ```cpp
    std::string extract(std::string const & text,
                        std::optional<int> start,
                        std::optional<int> end)
    {
      auto s = start.value_or(0);
      auto e = end.value_or(text.length());
      return text.substr(s, e - s);
    }
    auto v1 = extract("sample"s, {}, {});
    std::cout << v1 << '\n'; // sample
    auto v2 = extract("sample"s, 1, {});
    std::cout << v2 << '\n'; // ample
    auto v3 = extract("sample"s, 1, 4);
    std::cout << v3 << '\n'; // amp 
    ```

+   可选的类数据成员：

    ```cpp
    struct book
    {
      std::string                title;
      std::optional<std::string> subtitle;
      std::vector<std::string>   authors;
      std::string                publisher;
      std::string                isbn;
      std::optional<int>         pages;
      std::optional<int>         year;
    }; 
    ```

## 它是如何工作的...

类模板 `std::optional` 是一个表示可选值容器的类模板。如果容器包含值，则该值作为 `optional` 对象的一部分存储；不涉及堆分配和指针。`std::optional` 类模板的概念性实现如下：

```cpp
template <typename T>
class optional
{
  bool _initialized;
  std::aligned_storage_t<sizeof(t), alignof(T)> _storage;
}; 
```

`std::aligned_storage_t` 别名模板允许我们创建未初始化的内存块，这些内存块可以存储特定类型的对象。类模板 `std::optional` 如果是默认构造的，或者是从另一个空的 `std::optional` 对象或从 `std::nullopt_t` 值复制构造或复制赋值而来，则不包含值。这样的值是 `std::nullopt`，一个用于表示具有未初始化状态的 `std::optional` 对象的 `constexpr` 值。这是一个辅助类型，实现为一个空类，用于指示具有未初始化状态的 `std::optional` 对象。

`optional` 类型（在其他编程语言中称为 *nullable*）的典型用途是从可能失败的功能返回。这种情况的可能解决方案包括以下几种：

+   返回一个 `std::pair<T, bool>`，其中 `T` 是返回值的类型；对的数据是布尔标志，指示第一个元素的有效性。

+   返回一个 `bool`，接受一个额外的类型为 `T&` 的参数，并且仅在函数成功时将值赋给此参数。

+   返回原始指针或智能指针类型，并使用 `nullptr` 来指示失败。

类模板 `std::optional` 是一种更好的方法，因为一方面，它不涉及函数的输出参数（这在 C 和 C++ 之外不是返回值的规范形式），也不需要处理指针，另一方面，它更好地封装了 `std::pair<T, bool>` 的细节。

然而，可选对象也可以用于类的数据成员，并且编译器能够优化内存布局以实现高效的存储。

类模板 `std::optional` 不能用于返回多态类型。例如，如果你编写了一个需要从类型层次结构返回不同类型的工厂方法，你不能依赖于 `std::optional`，需要返回一个指针，最好是 `std::unique_ptr` 或 `std::shared_ptr`（取决于是否需要共享对象的所有权）。

当你使用 `std::optional` 将可选参数传递给函数时，你需要理解它可能会产生复制，如果涉及到大型对象，这可能会成为性能问题。让我们考虑以下具有对 `std::optional` 参数的常量引用的函数的例子：

```cpp
struct bar { /* details */ };
void process(std::optional<bar> const & arg)
{
  /* do something with arg */
}
std::optional<bar> b1{ bar{} };
bar b2{};
process(b1); // no copy
process(b2); // copy construction 
```

第一次调用`process()`不涉及任何额外的对象构造，因为我们传递了一个`std::optional<bar>`对象。然而，第二次调用将涉及`bar`对象的复制构造，因为`b2`是一个`bar`，需要被复制到一个`std::optional<bar>`中；即使`bar`实现了移动语义，也会进行复制。如果`bar`是一个小对象，这不应该引起太大的关注，但对于大对象，这可能会成为一个性能问题。避免这种情况的解决方案取决于上下文，可能包括创建一个接受`bar`常量引用的第二个重载，或者完全避免使用`std::optional`。

## 还有更多…

虽然`std::optional`使得从可能失败的函数中返回值变得更加容易，但将多个此类函数链式连接起来会产生冗长或至少过于重复的代码。为了简化这种情况，在 C++23 中，`std::optional`有多个额外的成员（`transform()`、`and_then()`和`or_else()`），被称为单子操作。我们将在下一道菜谱中了解它们。

## 参见

+   *使用 std::any 存储任何值*，了解如何使用 C++17 类`std::any`，它代表任何类型的单值类型安全容器

+   *使用 std::variant 作为类型安全的联合体*，了解如何使用 C++17 类`std::variant`来表示类型安全的联合体

+   *将可能或可能不产生值的计算链式连接起来*，以了解新的 C++23 单子操作`std::optional`如何简化多个返回`std::optional`的函数依次调用的场景

# 将可能或可能不产生值的计算链式连接起来

在之前的菜谱中，我们看到了如何使用`std::optional`类来存储可能存在或不存在的数据。它的用例包括函数的可选参数和可能无法产生结果的函数的返回值。当需要将多个此类函数链式连接起来时，代码可能会变得冗长且啰嗦。因此，C++23 标准为`std::optional`类添加了几个新方法。它们被称为**单子操作**。这些方法包括`transform()`、`and_then()`和`or_else()`。在本菜谱中，我们将了解它们有什么用。

简而言之，在函数式编程中，一个**单子**是一个封装在其包装值之上的一些功能的容器。例如，C++中的`std::optional`就是一个这样的例子。另一方面，一个**单子操作**是一个从域 *D* 到 *D* 自身的函数。例如，**恒等函数**（返回其参数的函数）就是一个单子操作。新添加的函数`transform()`、`and_then()`和`or_else()`是单子操作，因为它们接受一个`std::optional`并返回一个`std::optional`。

## 准备工作

在以下章节中，我们将参考此处所示的定义：

```cpp
struct booking
{
   int                        id;
   int                        nights;
   double                     rate;
   std::string                description;
   std::vector<std::string>   extras;
};
std::optional<booking> make_booking(std::string_view description, 
 int nights, double rate);
std::optional<booking> add_rental(std::optional<booking> b);
std::optional<booking> add_insurance(std::optional<booking> b);
double calculate_price(std::optional<booking> b);
double apply_discount(std::optional<double> p); 
```

## 如何做到这一点…

根据您的使用情况，您可以使用以下单子操作：

+   如果你有一个`可选`值，并想应用一个函数`f`并返回该调用的值，那么请使用`transform()`:

    ```cpp
    auto b = make_booking("Hotel California", 3, 300);
    auto p = b.transform(calculate_price); 
    ```

+   如果你有一个`可选`值，并想应用一个返回`可选`值的函数`f`，然后返回该调用的值，那么请使用`and_then()`:

    ```cpp
    auto b = make_booking("Hotel California", 3, 300);
         b = b.and_then(add_insurance);
    auto p = b.transform(calculate_price); 
    ```

+   如果你有一个可能为空的`可选`值，在这种情况下，你想调用一个函数来处理这种情况（例如记录日志或抛出异常），并返回另一个`可选`（一个替代值或一个空的`可选`），那么请使用`or_else()`:

    ```cpp
    auto b = make_booking("Hotel California", 3, 300)
             .or_else([]() -> std::optional<booking> {
                std::cout << "creating the booking failed!\n";  
                return std::nullopt; 
             }); 
    ```

下面的片段展示了更大的例子：

```cpp
auto p =
    make_booking("Hotel California", 3, 300)
   .and_then(add_rental)
   .and_then(add_insurance)
   .or_else([]() -> std::optional<booking> {
      std::cout << "creating the booking failed!\n";  
      return std::nullopt; })
   .transform(calculate_price)
   .transform(apply_discount)
   .or_else([]() -> std::optional<double> {
      std::cout << "computing price failed!\n"; return -1; }); 
```

## 它是如何工作的...

`and_then()`和`transform()`成员函数非常相似。它们实际上具有相同数量的重载，具有相同的签名。它们接受一个参数，该参数是一个函数或可调用对象，并且它们都返回一个`可选`。如果`可选`不包含值，那么`and_then()`和`transform()`都返回一个空的`可选`。

否则，如果`可选`确实包含一个值，那么它将使用存储的值调用该函数或可调用对象。这里就是它们的不同之处：

+   传递给`and_then()`的函数/可调用对象必须返回一个`std::optional`类型的值。这将是由`and_then()`返回的值。

+   传递给`transform()`的函数/可调用对象可以返回任何非引用类型的返回类型。然而，它在返回之前将自身包裹在一个`std::optional`中。

为了更好地说明这一点，让我们再次考虑以下函数：

```cpp
double calculate_price(std::optional<booking> b); 
```

之前，我们已经看到了这个片段：

```cpp
auto b = make_booking("Hotel California", 3, 300);
auto p = b.transform(calculate_price); 
```

在这里，`p`具有`std::optional<double>`类型。这是因为`calculate_price()`返回一个`double`，因此`transform()`将返回一个`std::optional<double>`。让我们将`calculate_price()`的签名更改为返回`std::optional<double>`：

```cpp
std::optional<double> calculate_price(std::optional<booking> b); 
```

变量`p`现在将具有`std::optional<std::optional<double>>`的类型。

第三种单子函数`or_else()`是`and_then()`/`transform()`的对立面：如果`可选`对象包含一个值，它将不做任何操作并返回该`可选`。否则，它将调用其单个参数，即一个不带任何参数的函数或可调用对象，并从这次调用返回值。函数/可调用对象的返回类型必须是`std::optional<T>`。

`or_else()`函数通常用于处理预期值缺失时的错误情况。提供的函数可能向日志中添加条目、抛出异常或执行其他操作。除非这个可调用对象抛出异常，否则它必须返回一个值。这可以是一个空的`可选`，或者是一个包含默认值或替代缺失值的`可选`。

## 还有更多...

`std::optional` 的一个最重要的用例是从可能产生也可能不产生值的函数中返回值。然而，当值缺失时，我们可能需要知道失败的原因。使用可选类型时，这并不直接可行，除非存储的类型是一个值和错误的复合体，或者如果我们在函数中使用了额外的参数来检索错误。因此，C++23 标准为这些用例提供了 `std::optional` 的替代方案，即 `std::expected` 类型。

## 参见

+   *使用 `std::expected` 返回值或错误，以了解这种 C++23 类型如何使我们能够从函数中返回值或错误代码*

# 将 `std::variant` 作为类型安全的联合使用

在 C++ 中，联合类型是一种特殊类类型，在任何时刻，它都持有其数据成员中的一个值。与常规类不同，联合不能有基类，也不能被派生，并且不能包含虚拟函数（这本来就没有意义）。联合主要用于定义相同数据的不同表示。然而，联合仅适用于 **纯旧数据** (**POD**) 类型。如果一个联合包含非 POD 类型的值，那么这些成员需要使用带位置的 `new` 进行显式构造和显式销毁，这很麻烦且容易出错。在 C++17 中，类型安全的联合以标准库类模板 `std::variant` 的形式提供。在本食谱中，您将学习如何使用它来建模替代值。

## 准备工作

`std::variant` 类型实现了一个类型安全的 **区分联合**。尽管详细讨论这些内容超出了本食谱的范围，但我们将在这里简要介绍它们。熟悉区分联合将帮助我们更好地理解 `variant` 的设计和其工作方式。

区分联合也称为 **标记联合** 或 **不相交联合**。区分联合是一种能够存储一组类型中的一个值并提供对该值类型安全访问的数据类型。在 C++ 中，这通常如下实现：

```cpp
enum VARTAG {VT_int, VT_double, VT_pint, TP_pdouble /* more */ };
struct variant_t
{
  VARTAG tag;
  union Value 
  {
    int     i;
    int*    pi;
    double  d;
    double* pd;
    /* more */
  } value;
}; 
```

对于 Windows 程序员来说，一个众所周知的区分联合是用于 **组件对象模型** (**COM**) 编程的 `VARIANT` 结构。

类模板 `std::variant` 是基于 `boost::variant` 设计的，并在 `<variant>` 头文件中可用。如果您熟悉 `boost::variant` 并已在代码中使用它，您可以通过少量努力将代码迁移到使用标准的 `variant` 类模板。

## 如何做到这一点...

使用以下操作来处理 `std::variant`：

+   要修改存储的值，请使用成员函数 `emplace()` 或 `swap()`：

    ```cpp
    struct foo
    {
      int value;
      explicit foo(int const i) : value(i) {}
    };
    std::variant<int, std::string, foo> v = 42; // holds int
    v.emplace<foo>(42);                         // holds foo 
    ```

+   要读取存储的值，请使用非成员函数 `std::get` 或 `std::get_if`：

    ```cpp
    std::variant<int, double, std::string> v = 42;
    auto i1 = std::get<int>(v);
    auto i2 = std::get<0>(v);
    try
    {
      auto f = std::get<double>(v);
    }
    catch (std::bad_variant_access const & e)
    {
      std::cout << e.what() << '\n'; // Unexpected index
    } 
    ```

+   要存储一个值，请使用构造函数或将值直接赋给 `variant` 对象：

    ```cpp
    std::variant<int, double, std::string> v;
    v = 42;   // v contains int 42
    v = 42.0; // v contains double 42.0
    v = "42"; // v contains string "42" 
    ```

+   要检查存储的替代项，请使用成员函数 `index()`：

    ```cpp
    std::variant<int, double, std::string> v = 42;
    static_assert(std::variant_size_v<decltype(v)> == 3);
    std::cout << "index = " << v.index() << '\n';
    v = 42.0;
    std::cout << "index = " << v.index() << '\n';
    v = "42";
    std::cout << "index = " << v.index() << '\n'; 
    ```

+   要检查变体是否持有替代方案，请使用非成员函数`std::holds_alternative()`：

    ```cpp
    std::variant<int, double, std::string> v = 42;
    std::cout << "int? " << std::boolalpha
              << std::holds_alternative<int>(v)
              << '\n'; // int? true
    v = "42";
    std::cout << "int? " << std::boolalpha
              << std::holds_alternative<int>(v)
              << '\n'; // int? false 
    ```

+   要定义一个第一个替代方案不是默认可构造的变体，请使用`std::monostate`作为第一个替代方案（在这个例子中，`foo`是我们之前使用的相同类）：

    ```cpp
    std::variant<std::monostate, foo, int> v;
    v = 42;        // v contains int 42
    std::cout << std::get<int>(v) << '\n';
    v = foo{ 42 }; // v contains foo{42}
    std::cout << std::get<foo>(v).value << '\n'; 
    ```

+   要处理变体存储的值并根据替代方案的类型执行某些操作，请使用`std::visit()`：

    ```cpp
    std::variant<int, double, std::string> v = 42;
    std::visit(
      [](auto&& arg) {std::cout << arg << '\n'; },
      v); 
    ```

## 它是如何工作的...

`std::variant`是一个类模板，它模拟了一个类型安全的联合，在任何给定时间持有其可能的替代方案之一。然而，在某些罕见的情况下，变体对象可能不存储任何值。`std::variant`有一个名为`valueless_by_exception()`的成员函数，如果变体不持有值，则返回`true`，这只有在初始化期间发生异常的情况下才可能，因此函数的名称。

`std::variant`对象的大小与其最大的替代方案一样大。变体不存储额外的数据。变体存储的值是在对象的内存表示内部分配的。

变体可以持有相同类型的多个替代方案，并且还可以同时持有不同常量和易失性资格的版本。在这种情况下，您不能分配多个类型的值，而应使用`emplace()`成员函数，如下面的代码片段所示：

```cpp
std::variant<int, int, double> v = 33.0;
v = 42;                               // error
v.emplace<1>(42);                     // OK
std::cout << std::get<1>(v) << '\n';  // prints 42
std::holds_alternative<int>(v);       // error 
```

之前提到的`std::holds_alternative()`函数，它检查变体是否持有替代类型`T`，在此情况下不能使用。您应该避免定义持有相同类型多个替代方案的变体。

另一方面，变体不能持有类型`void`的替代方案，或者数组和引用类型的替代方案。此外，第一个替代方案必须始终是默认可构造的。这是因为，就像区分联合一样，变体使用其第一个替代方案的值进行默认初始化。如果第一个替代方案类型不是默认可构造的，那么变体必须使用`std::monostate`作为第一个替代方案。这是一个空类型，旨在使变体默认可构造。

可以在编译时查询`variant`的大小（即它定义的替代方案的数量）以及通过其零基索引指定的替代方案类型。另一方面，您可以使用成员函数`index()`在运行时查询当前持有的替代方案的索引。

## 更多...

操作变体内容的一种典型方式是通过访问。这基本上是基于变体持有的替代方案执行一个动作。由于这是一个较大的主题，它将在下一个菜谱中单独讨论。

## 参见

+   *使用`std::any`存储任何值*，了解如何使用 C++17 类`std::any`，它代表任何类型的单值类型安全容器

+   *使用 std::optional 存储可选值*，了解 C++17 类模板 `std::optional`，它管理可能存在或不存在的一个值

+   *访问 std::variant*，了解如何执行类型匹配并根据变体替代的类型执行不同的操作

# 访问 std::variant

`std::variant` 是一个新标准容器，它基于 `boost.variant` 库添加到 C++17 中。变体是一个类型安全的联合体，它持有其替代类型之一的值。尽管在之前的食谱中，我们已经看到了各种变体的操作，但我们使用的变体相当简单，主要是 POD 类型，这并不是 `std::variant` 被创建的实际目的。变体旨在用于持有类似非多态和非 POD 类型的替代项。在这个食谱中，我们将看到一个更实际的变体使用示例，并学习如何访问变体。

## 准备工作

对于这个食谱，你应该熟悉 `std::variant` 类型。建议你首先阅读之前的食谱，*使用 std::variant 作为类型安全的联合体*。

为了解释如何进行变体访问，我们将考虑一个用于表示媒体 DVD 的变体。假设我们想要模拟一个商店或图书馆，其中包含可能包含音乐、电影或软件的 DVD。然而，这些选项不是作为具有公共数据和虚拟函数的层次结构来建模，而是作为可能具有类似属性（如标题）的非相关类型。为了简单起见，我们将考虑以下属性：

+   对于电影：标题和长度（以分钟为单位）

+   对于一个专辑：标题、艺术家姓名以及曲目列表（每首曲目都有一个标题和以秒为单位的长度）

+   对于软件：标题和制造商

以下代码展示了这些类型的简单实现，没有包含任何函数，因为这与访问包含这些类型变体的变体无关：

```cpp
enum class Genre { Drama, Action, SF, Comedy };
struct Movie
{
  std::string title;
  std::chrono::minutes length;
  std::vector<Genre> genre;
};
struct Track
{
  std::string title;
  std::chrono::seconds length;
};
struct Music
{
  std::string title;
  std::string artist;
  std::vector<Track> tracks;
};
struct Software
{
  std::string title;
  std::string vendor;
};
using dvd = std::variant<Movie, Music, Software>;
std::vector<dvd> dvds
{
  Movie{ "The Matrix"s, 2h + 16min,{ Genre::Action, Genre::SF } },
  Music{ "The Wall"s, "Pink Floyd"s,
       { { "Mother"s, 5min + 32s },
         { "Another Brick in the Wall"s, 9min + 8s } } },
  Software{ "Windows"s, "Microsoft"s },
}; 
```

另一方面，我们将使用以下函数将文本转换为大写：

```cpp
template <typename CharT>
using tstring = std::basic_string<CharT, std::char_traits<CharT>, 
                                         std::allocator<CharT>>;
template<typename CharT>
inline tstring<CharT> to_upper(tstring<CharT> text)
{
   std::transform(std::begin(text), std::end(text), 
                  std::begin(text), toupper);
   return text;
} 
```

定义好这些之后，让我们开始探讨如何执行访问变体。

## 如何做到这一点...

要访问一个变体，你必须为变体的可能替代提供一个或多个动作。有几种类型的访问者，用于不同的目的：

+   一个不返回任何内容但具有副作用的无返回值访问者。以下示例将每张 DVD 的标题打印到控制台：

    ```cpp
    for (auto const & d : dvds)
    {
      std::visit([](auto&& arg) {
                   std::cout << arg.title << '\n'; },
                 d);
    } 
    ```

+   返回值的访问者；值应该与当前变体的任何替代类型相同，或者本身可以是一个变体。在以下示例中，我们访问一个变体并返回一个具有相同类型的新的变体，其 `title` 属性从任何替代类型转换为大写字母：

    ```cpp
    for (auto const & d : dvds)
    {
      dvd result = std::visit(
        [](auto&& arg) -> dvd
        {
          auto cpy { arg };
          cpy.title = to_upper(cpy.title);
          return cpy;
        },
      d);
      std::visit(
        [](auto&& arg) {
          std::cout << arg.title << '\n'; },
        result);
    } 
    ```

+   通过提供具有为变体的每种替代类型重载的调用操作符的函数对象来实现类型匹配的访问者（这可以是空返回值或返回值的访问者）：

    ```cpp
    struct visitor_functor
    {
      void operator()(Movie const & arg) const
     {
        std::cout << "Movie" << '\n';
        std::cout << " Title: " << arg.title << '\n';
        std::cout << " Length: " << arg.length.count()
                  << "min" << '\n';
      }
      void operator()(Music const & arg) const
     {
        std::cout << "Music" << '\n';
        std::cout << " Title: " << arg.title << '\n';
        std::cout << " Artist: " << arg.artist << '\n';
        for (auto const & t : arg.tracks)
          std::cout << " Track: " << t.title
                    << ", " << t.length.count()
                    << "sec" << '\n';
      }
      void operator()(Software const & arg) const
     {
        std::cout << "Software" << '\n';
        std::cout << " Title: " << arg.title << '\n';
        std::cout << " Vendor: " << arg.vendor << '\n';
      }
    };
    for (auto const & d : dvds)
    {
      std::visit(visitor_functor(), d);
    } 
    ```

+   通过提供执行基于替代类型动作的 lambda 表达式来实现类型匹配的访问者：

    ```cpp
    for (auto const & d : dvds)
    {
      std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Movie>)
        {
          std::cout << "Movie" << '\n';
          std::cout << " Title: " << arg.title << '\n';
          std::cout << " Length: " << arg.length.count()
                    << "min" << '\n';
        }
        else if constexpr (std::is_same_v<T, Music>)
        {
          std::cout << "Music" << '\n';
          std::cout << " Title: " << arg.title << '\n';
          std::cout << " Artist: " << arg.artist << '\n';
          for (auto const & t : arg.tracks)
            std::cout << " Track: " << t.title
                      << ", " << t.length.count()
                      << "sec" << '\n';
        }
        else if constexpr (std::is_same_v<T, Software>)
        {
          std::cout << "Software" << '\n';
          std::cout << " Title: " << arg.title << '\n';
          std::cout << " Vendor: " << arg.vendor << '\n';
        }
      },
      d);
    } 
    ```

## 它是如何工作的...

访问者是一个可调用对象（一个函数、一个 lambda 表达式或一个函数对象），它接受来自变体的所有可能的替代项。通过使用访问者和一个或多个变体对象调用 `std::visit()` 来进行访问。变体不必是相同的类型，但访问者必须能够接受所有被调用的变体的所有可能的替代项。在先前的示例中，我们访问了一个单个的变体对象，但访问多个变体并不意味着比将它们作为参数传递给 `std::visit()` 更多的事情。

当访问一个变体时，可调用对象会使用当前存储在变体中的值来调用。如果访问者不接受变体中存储的类型作为参数，则程序是不规范的。如果访问者是一个函数对象，那么它必须为变体的所有可能的替代类型重载其调用操作符。如果访问者是一个 lambda 表达式，它应该是一个泛型 lambda，这基本上是一个具有调用操作符模板的函数对象，由编译器根据实际调用的类型实例化。

在上一节中展示了两种方法类型的访问者示例。第一个示例中的函数对象很简单，不需要额外的解释。另一方面，泛型 lambda 表达式使用 `constexpr if` 来根据编译时参数的类型选择特定的 `if` 分支。结果是编译器将创建一个具有操作符调用模板和包含 `constexpr if` 语句的体的函数对象；当它实例化该函数模板时，它将为变体的每种可能的替代类型生成一个重载，并且在每个这些重载中，它将只选择与调用操作符参数类型匹配的 `constexpr if` 分支。结果是概念上等同于 `visitor_functor` 类的实现。

## 参见

+   *使用 `std::any` 存储任何值*，了解如何使用 C++17 类 `std::any`，它代表一个类型安全的容器，用于存储任何类型的单个值

+   *使用 `std::optional` 存储可选值*，了解关于 C++17 类模板 `std::optional` 的信息，该模板管理可能存在或不存在的一个值

+   *使用 `std::variant` 作为类型安全的联合*，了解如何使用 C++17 `std::variant` 类来表示类型安全的联合

# 使用 `std::expected` 返回值或错误

我们经常需要编写一个函数，该函数既返回一些数据，又返回成功或失败的指示（对于最简单的情况，可以是 `bool`，对于更复杂的情况，可以是错误代码）。通常，这可以通过返回状态码并使用通过引用传递的参数来返回数据来解决，或者在实际数据返回失败的情况下抛出异常。近年来，`std::optional` 和 `std::variant` 的可用性为解决这个问题提供了新的解决方案。然而，C++23 标准通过 `std::expected` 类型提供了一种新的方法，这是一种两种先前提及类型的组合。这种类型存在于其他编程语言中，如 Rust 中的 `Result` 和 Haskell 中的 `Either`。在本食谱中，我们将学习如何使用这个新的 `std::expected` 类。

## 准备工作

在本食谱中展示的示例中，我们将使用在此处定义的数据类型：

```cpp
enum class Status
{
   Success, InvalidFormat, InvalidLength, FilterError,
};
enum class Filter
{
   Pixelize, Sepia, Blur
};
using Image = std::vector<char>; 
```

## 如何操作…

您可以使用来自新 `<expected>` 头文件的 `std::expected<T, E>` 类型，如下面的示例所示：

+   当从函数返回数据时，返回 `std::unexpected<E>` 以指示错误，或者在一切执行成功时返回数据（`T` 类型的值）：

    ```cpp
    bool IsValidFormat(Image const& img) { return true; }
    bool IsValidLength(Image const& img) { return true; }
    bool Transform(Image& img, Filter const filter)
    { 
       switch(filter)
       {
       case Filter::Pixelize:
          img.push_back('P');
          std::cout << "Applying pixelize\n";
       break;
       case Filter::Sepia:
          img.push_back('S');
          std::cout << "Applying sepia\n";
       break;
       case Filter::Blur:
          img.push_back('B');
          std::cout << "Applying blur\n";
       break;
       }
       return true; 
    }
    std::expected<Image, Status> ApplyFilter(Image img, 
                                             Filter const filter)
    {
       if (!IsValidFormat(img))
          return std::unexpected<Status> {Status::InvalidFormat};
       if (!IsValidLength(img))
          return std::unexpected<Status> {Status::InvalidLength};
       if (!Transform(img, filter))
          return std::unexpected<Status> {Status::FilterError};
       return img;
    }
    std::expected<Image, Status> FlipHorizontally(Image img)
    {
        return Image{img.rbegin(), img.rend()};
    } 
    ```

+   当检查返回 `std::expected<T, E>` 的函数的结果时，使用 `bool` 操作符（或 `has_value()` 方法）来检查对象是否包含预期的值，并使用 `value()` 和 `error()` 方法分别返回预期的值或意外的错误：

    ```cpp
    void ShowImage(Image const& img)
    {
       std::cout << "[img]:";
       for(auto const & e : img) std::cout << e;
       std::cout << '\n';
    }
    void ShowError(Status const status)
    {
       std::cout << "Error code: " 
                 << static_cast<int>(status) << '\n'; 
    }
    int main()
    {
       Image img{'I','M','G'};

       auto result = ApplyFilter(img, Filter::Sepia);
       if (result)
       {
          ShowImage(result.value());
       }
       else
       {
          ShowError(result.error());
       }
    } 
    ```

+   您可以使用返回 `std::expected` 值的函数通过单调操作 `and_then()`、`or_else()`、`transform()` 和 `transform_error()` 组成操作链：

    ```cpp
    int main()
    {
       Image img{'I','M','G'};

       ApplyFilter(img, Filter::Sepia)                
          .and_then([](Image result){
              return ApplyFilter(result, Filter::Pixelize);
          })
          .and_then([](Image result){
              return ApplyFilter(result, Filter::Blur);
          })
          .and_then([](Image result){
              ShowImage(result);
              return std::expected<Image, Status>{result};
          })
          .or_else([](Status status){
              ShowError(status);
              return std::expected<Image, Status>{std::unexpect, 
                                                  status};
          });
    } 
    ```

## 它是如何工作的…

`std::expected<T, E>` 类模板可在新的 C++23 头文件 `<expected>` 中找到。这个类是 `std::variant` 和 `std::optional` 类型（C++17 中引入）的混合体，但旨在从函数返回数据或意外值。由于它要么持有预期类型 `T` 的值，要么持有意外类型（错误）`E` 的值，因此它具有判别联合的逻辑结构。然而，它的接口与 `std::optional` 类非常相似，因为它具有以下成员：

| **函数** | **描述** |
| --- | --- |
| `has_value()` | 返回一个布尔值，指示对象是否包含预期的值。 |
| `operator bool` | 与 `has_value()` 相同。提供用于在 `if` 语句中更简单使用的功能（`if(result)` 相对于 `if(result.has_value())`）。 |
| `value()` | 返回预期的值，除非对象包含意外的值。在这种情况下，它抛出一个包含意外值的 `std::bad_expected_access<E>` 异常。 |
| `value_or()` | 与 `value()` 类似，但如果对象中存储了意外的值，它不会抛出异常，而是返回提供的替代值。 |
| `error()` | 返回意外的值。如果对象包含期望值，则行为未定义。 |
| `operator->` 和 `operator*` | 访问期望值。如果对象包含意外的值，则行为未定义。 |

表 6.2：`std::expected` 最重要成员的列表

虽然之前提到 `std::expected` 类型是两个 `T`（期望）和 `E`（错误）类型的区分联合，但这并不完全正确。它实际持有的类型要么是 `T`，要么是 `std::unexpected<E>`。后者是一个辅助类，用于持有类型为 `E` 的对象。对 `T` 和 `E` 可用类型的有些限制：

+   `T` 可以是 `void` 或可销毁的类型（可以调用析构函数的类型）。不能替换 `T` 的类型是数组和引用类型。如果类型 `T` 是 `void` 类型，则 `value_or()` 方法不可用。

+   `E` 必须是可销毁的类型。数组、引用类型以及 `const` 和 `volatile` 标记的类型不能替换 `E`。

有时你想要对一个值应用多个操作。在我们的例子中，这可能是对图像连续应用不同的过滤器。但这也可能是其他事情，比如调整图像大小、更改格式/类型、在不同方向上翻转等。每个这些操作都可能返回一个 `std::expected` 值。在这种情况下，我们可以编写如下代码：

```cpp
int main()
{
   Image img{'I','M','G'};

   auto result = ApplyFilter(img, Filter::Sepia);
   result = ApplyFilter(result.value(), Filter::Pixelize);
   result = ApplyFilter(result.value(), Filter::Blur);
   result = FlipHorizontally(result.value());
   if (result)
   {
      ShowImage(result.value());
   }
   else
   {
      ShowError(result.error());
   }
} 
```

如果没有发生错误，则运行此程序的结果如下：

```cpp
Applying sepia
Applying pixelize
Applying blur
[img]:BPSGMI 
```

然而，如果在 `ApplyFilter()` 函数中发生错误，后续调用中调用 `value()` 方法将导致 `std::bad_expected_access` 异常。实际上，我们必须在每次操作后检查结果。这可以使用单一操作来改进。

由于 `std::expected` 类型与 `std::optional` 类型非常相似，C++23 中为 `std::optional` 提供的单一操作也适用于 `std::expected`。以下是一些操作：

| **函数** | **描述** |
| --- | --- |
| `and_then()` | 如果 `std::expected` 对象包含期望值（类型为 `T`），则对它应用给定的函数并返回结果。否则，返回 `std::expected` 值。 |
| `or_else()` | 如果 `std::expected` 对象包含意外的值（类型为 `E`），则对意外的值应用给定的函数并返回结果。否则，返回 `std::expected` 值。 |
| `transform()` | 这与 `and_then()` 类似，但返回的值也被封装在一个 `std::expected` 值中。 |
| `transform_error()` | 这与 `or_else()` 类似，但返回的值也被封装在一个 `std::expected` 值中。 |

表 6.3：`std::expected` 的单一操作

我们可以使用单一操作重写上一个代码片段，如下所示：

```cpp
int main()
{
   Image img{'I','M','G'};

   ApplyFilter(img, Filter::Sepia)                
      .and_then([](Image result){
          return ApplyFilter(result, Filter::Pixelize);
      })
      .and_then([](Image result){
          return ApplyFilter(result, Filter::Blur);
      })
      .and_then(FlipHorizontally)       
      .and_then([](Image result){
          ShowImage(result);
          return std::expected<Image, Status>{result};
      })
      .or_else([](Status status){
          ShowError(status);
          return std::expected<Image, Status>{std::unexpect, status};
      });
} 
```

如果没有发生错误，则输出将是我们已经看到的那个。然而，如果发生错误，比如说在应用棕褐色滤镜时，输出将变为以下内容：

```cpp
Applying sepia
Error code: 3 
```

此示例仅展示了可用的两种单子操作，`and_then()` 和 `or_else()`。其他两种，`transform()` 和 `transform_or()`，功能相似，但它们旨在将（正如其名称所暗示的）预期值或意外值转换为另一个值。在以下代码片段（对上一个代码片段的修改）中，我们为预期值和意外值都链式调用了转换操作，在任一情况下都返回一个字符串：

```cpp
int main()
{
   Image img{'I','M','G'};

   auto obj = ApplyFilter(img, Filter::Sepia)                
      .and_then([](Image result){
          return ApplyFilter(result, Filter::Pixelize);
      })
      .and_then([](Image result){
          return ApplyFilter(result, Filter::Blur);
      })
      .and_then(FlipHorizontally)       
      .and_then([](Image result){
          ShowImage(result);
          return std::expected<Image, Status>{result};
      })
      .or_else([](Status status){
          ShowError(status);
          return std::expected<Image, Status>{std::unexpect, status};
      })       
      .transform([](Image result){
          std::stringstream s;
          s << std::quoted(std::string(result.begin(), 
                                       result.end()));
          return s.str();
      })
      .transform_error([](Status status){
          return status == Status::Success ? "success" : "fail";
      });
    if(obj)
       std::cout << obj.value() << '\n';
    else
       std::cout << obj.error() << '\n';
} 
```

如果程序执行过程中没有发生错误，则将打印以下输出：

```cpp
Applying sepia
Applying pixelize
Applying blur
[img]:BPSGMI
"BPSGMI" 
```

然而，如果在执行过程中发生错误，例如在应用棕褐色滤镜时，输出将变为以下内容：

```cpp
Applying sepia
Error code: 3
fail 
```

在上面的 `or_else()` 函数中，你会注意到 `std::unexpected` 的使用。这是一个辅助类，它作为 `std::expected` 构造函数的标签，以指示意外值的构造。因此，参数被完美转发到 `E` 类型（意外类型）的构造函数中。`has_value()` 方法将返回 `false` 对于新创建的 `std::expected` 值，表示它包含一个意外值。

## 参见

+   *使用 `std::optional` 存储可选值*，了解 C++17 类模板 `std::optional`，它管理可能存在或不存在的数据值

+   *使用 `std::variant` 作为类型安全的联合体*，了解如何使用 C++17 `std::variant` 类来表示类型安全的联合体

# 使用 `std::span` 对象的连续序列

在 C++17 中，`std::string_view` 类型被添加到标准库中。这是一个表示对字符连续序列视图的对象。视图通常使用指向序列第一个元素的指针和长度来实现。字符串是任何编程语言中最常用的数据类型之一。它们有一个非拥有视图，不分配内存，避免复制，并且比 `std::string` 执行某些操作更快，这是一个重要的好处。然而，字符串只是一个具有特定于文本操作的字符特殊向量。因此，有一个类型，它是对连续序列对象的视图，无论它们的类型如何，这是有意义的。这就是 C++20 中的 `std::span` 类模板所代表的。我们可以这样说，`std::span` 对于 `std::vector` 和数组类型来说，就像 `std::string_view` 对于 `std::string` 一样。

## 准备工作

`std::span` 类模板在头文件 `<span>` 中可用。

## 如何做到这一点…

使用 `std::span<T>` 而不是指针和大小对，就像在 C 类接口中通常所做的那样。换句话说，替换如下函数：

```cpp
void func(int* buffer, size_t length) { /* ... */ } 
```

如此：

```cpp
void func(std::span<int> buffer) { /* ... */ } 
```

当使用 `std::span` 时，你可以做以下操作：

+   通过指定跨度中的元素数量来创建具有编译时长度（称为*静态范围*）的跨度：

    ```cpp
    int arr[] = {1, 1, 2, 3, 5, 8, 13};
    std::span<int, 7> s {arr}; 
    ```

+   通过不指定跨度中的元素数量来创建具有运行时长度（称为 *动态范围*）的跨度：

    ```cpp
    int arr[] = {1, 1, 2, 3, 5, 8, 13};
    std::span<int> s {arr}; 
    ```

+   您可以使用跨度在基于范围的 for 循环中：

    ```cpp
    void func(std::span<int> buffer)
    {
       for(auto const e : buffer)
          std::cout << e << ' ';
       std::cout << '\n';
    } 
    ```

+   您可以使用 `front()`、`back()`、`data()` 方法和 `operator[]` 来访问跨度的元素：

    ```cpp
    int arr[] = {1, 1, 2, 3, 5, 8, 13};
    std::span<int, 7> s {arr};
    std::cout << s.front() << " == " << s[0] << '\n';    
    // prints 1 == 1
    std::cout << s.back() << " == " << s[s.size() - 1] << '\n'; 
    // prints 13 == 13
    std::cout << *s.data() << '\n';
    // prints 1 
    ```

+   您可以使用 `first()`、`last()` 和 `subspan()` 方法从跨度中获取子跨度：

    ```cpp
    std::span<int> first_3 = s.first(3);
    func(first_3);  // 1 1 2.
    std::span<int> last_3 = s.last(3);
    func(last_3);   // 5 8 13
    std::span<int> mid_3 = s.subspan(2, 3);
    func(mid_3);    // 2 3 5 
    ```

## 它是如何工作的...

`std::span` 类模板不是一个对象的容器，而是一个轻量级包装器，它定义了一个连续对象序列的视图。最初，跨度被称为 `array_view`，有人认为这是一个更好的名称，因为它清楚地表明该类型是序列的非拥有视图，并且它与 `string_view` 的名称保持一致。然而，该类型是在标准库中以 *span* 的名称采用的。

尽管标准没有指定实现细节，跨度通常通过存储指向序列第一个元素的指针和一个长度来实现，该长度表示视图中的元素数量。因此，跨度可以用来定义对（但不限于）`std::vector`、`std::array`、`T[]` 或 `T*` 的非拥有视图。然而，它不能与列表或关联容器（例如，`std::list`、`std::map` 或 `std::set`）一起使用，因为这些不是连续元素序列的容器。

跨度可以具有编译时大小或运行时大小。当跨度的元素数量在编译时指定时，我们有一个具有静态范围（编译时大小）的跨度。如果元素数量未指定但在运行时确定，我们有一个动态范围。

`std::span` 类具有简单的接口，主要由以下成员组成：

| **成员函数** | **描述** |
| --- | --- |
| `begin()`, `end()`, `cbegin()`, `cend()` | 可变和常量迭代器，分别指向序列的第一个元素和最后一个元素之后的元素。 |
| `rbegin()`, `rend()`, `cbegin()`, `crend()` | 可变和常量反向迭代器，分别指向序列的开始和结束。 |
| `front()`, `back()` | 访问序列的第一个和最后一个元素。 |
| `data()` | 返回指向序列元素开头的指针。 |
| `operator[]` | 通过其索引访问序列中的元素。 |
| `size()` | 获取序列中的元素数量。 |
| `size_bytes()` | 获取序列的字节数。 |
| `empty()` | 检查序列是否为空。 |
| `first()` | 获取序列中前 *N* 个元素的子跨度。 |
| `last()` | 获取序列中最后 *N* 个元素的子跨度。 |
| `subspan()` | 从指定的偏移量开始获取具有 *N* 个元素的子跨度。如果未指定计数 *N*，则返回从偏移量到序列末尾的所有元素的跨度。 |

表 6.4：std::span 最重要成员函数列表

Span 不适用于使用一对迭代器（指向范围的开始和结束）的通用算法（如 `sort`、`copy`、`find_if` 等），也不适用于标准容器的替代品。其主要目的是构建比传递指针和大小到函数的 C 类接口更好的接口。用户可能会传递错误的大小值，这可能导致访问序列之外的内存。Span 提供了安全和边界检查。它也是将常量引用作为函数参数传递给 `std::vector<T>` (`std::vector<T> const &`) 的良好替代品。Span 不拥有其元素，足够小，可以按值传递（你不应该通过引用或常量引用传递 Span）。

与不支持更改序列中元素值的 `std::string_view` 不同，`std::span` 定义了一个可变视图，并支持修改其元素。为此，函数如 `front()`、`back()` 和 `operator[]` 返回一个引用。

## 参见

+   *第二章*，*使用 `std::string_view` 而不是常量字符串引用*，了解如何使用 `std::string_view` 在处理字符串时在某些场景中提高性能

+   *使用 `std::mdspan` 对对象序列的多维视图进行操作*，了解 C++23 的多维序列 span 类

# 使用 `std::mdspan` 对对象序列的多维视图

在之前的菜谱中，*使用 `std::span` 对连续对象序列进行操作*，我们学习了 C++20 类 `std::span`，它表示一个连续元素序列的视图（一个非拥有包装器）。这与 C++17 类 `std::string_view` 类似，它执行相同的操作，但针对字符序列。这两个都是一维序列的视图。然而，有时我们需要处理多维序列。这些可以通过多种方式实现，例如 C 类数组 (`int[2][3][4]`)、指针的指针 (`int**` 或 `int***`)、数组数组（或向量向量，如 `vector<vector<vector<int>>>`）。另一种方法是使用一维对象序列，但定义操作将其呈现为逻辑上的多维序列。这正是 C++23 `std::mdspan` 类所做的：它表示一个作为多维序列呈现的连续对象序列的非拥有视图。我们可以这样说，`std::mdspan` 是 `std::span` 类的多维视图扩展。

## 准备工作

在本菜谱中，我们将参考以下二维矩阵（其大小在编译时已知）的简单实现：

```cpp
template <typename T, std::size_t ROWS, std::size_t COLS>
struct matrix
{
   T& 
#ifdef __cpp_multidimensional_subscript
operator[] // C++23
#else
operator() // previously
#endif
   (std::size_t const r, std::size_t const c)
   {
      if (r >= ROWS || c >= COLS)
         throw std::runtime_error("Invalid index");
      return data[r * COLS + c];
   }
   T const & 
#ifdef __cpp_multidimensional_subscript
operator[] // C++23
#else
operator() // previously
#endif
   (std::size_t const r, std::size_t const c) const
   {
      if (r >= ROWS || c >= COLS)
         throw std::runtime_error("Invalid index");
      return data[r * COLS + c];
   }
   std::size_t size() const { return data.size(); }
   std::size_t empty() const { return data.empty(); }
   template <std::size_t dimension>
   std::size_t extent() const
 {
      static_assert(dimension <= 1, 
                    "The matrix only has two dimensions.");
      if constexpr (dimension == 0) return ROWS;
      else if constexpr(dimension == 1) return COLS;
   }
private:
   std::array<T, ROWS* COLS> data;
}; 
```

在 C++23 中，你应该优先使用 `operator[]` 而不是 `operator()` 来访问多维数据结构的元素。

## 如何实现...

更倾向于使用 `std::mdspan` 而不是多维 C 样式的数组、指针的指针或向量-向量/数组-数组实现。换句话说，替换如下函数：

```cpp
void f(int data[2][3]) { /* … */ }
void g(int** data, size_t row, size_t cols) { /* … */ }
void h(std::vector<std::vector<int>> dat, size_t row, size_t cols)
{ /* … */ } 
```

使用以下方法：

```cpp
void f(std::mdspan<int,std::extents<size_t, 2, 3>> data) 
{ /* … */ } 
```

当与 `std::mdspan` 一起工作时，你可以做以下操作：

+   通过指定跨度每个维度的元素数量来创建具有编译时长度（称为**静态范围**）的 `mdspan`：

    ```cpp
    int* data = get_data();
    std::mdspan<int, std::extents<size_t, 2, 3>> m(data); 
    ```

+   通过不在编译时指定跨度维度中元素的数量，而是在运行时提供它来创建具有运行时长度（称为**动态范围**）的 `mdspan`：

    ```cpp
    int* data = get_data();
    std::mdspan<int, std::extents<size_t, 
                                  2, 
                                  std::dynamic_extent>> mv{v.data(), 3}; 
    ```

    或者

    ```cpp
    int* data = get_data();
    std::mdspan<int, std::extents<size_t, 
                                  std::dynamic_extent,
                                  std::dynamic_extent>> 
    m(data, 2, 3); 
    ```

    或者

    ```cpp
    int* data = get_data();
    std::mdspan m(data, 2, 3); 
    ```

+   要控制 `mdspan` 的多维索引到底层（连续）数据序列的一维索引的映射，请使用布局策略，这是第三个模板参数：

    ```cpp
    std::mdspan<int, 
                std::extents<size_t, 2, 3>,
                std::layout_right> mv{ data }; 
    ```

    或者

    ```cpp
    std::mdspan<int, 
                std::extents<size_t, 2, 3>,
                std::layout_left> mv{ data }; 
    ```

    或者

    ```cpp
    std::mdspan<int, 
                std::extents<size_t, 2, 3>,
                std::layout_stride> mv{ data }; 
    ```

## 它是如何工作的……

如其名所示，`mdspan` 是一个多维跨度。这是一个非拥有视图，它将一维值序列投影为逻辑的多维结构。这是我们之前在 *准备就绪* 部分看到的内容，在那里我们定义了一个名为 `matrix` 的类，它表示一个二维矩阵。它定义的操作（如 C++23 中的 `operator()` 和/或 `operator[]`）是特定于 2D 数据结构的。然而，在内部，数据以连续序列的形式排列，在我们的实现中是一个 `std::array`。我们可以如下使用这个类：

```cpp
matrix<int, 2, 3> m;
for (std::size_t r = 0; r < m.extent<0>(); r++)
{
   for (std::size_t c = 0; c < m.extent<1>(); c++)
   {
      m[r, c] = r * m.extent<1>() + c + 1; // m[r,c] in C++23
// m(r, c) previously
   }
} 
```

这个 for-in-for 循环将矩阵元素的值设置为以下：

```cpp
1 2 3
4 5 6 
```

在 C++23 中，我们可以简单地用 `std::mdspan` 类替换整个类：

```cpp
std::array<int, 6> arr;
std::mdspan m{arr.data(), std::extents{2, 3}};
for (std::size_t r = 0; r < m.extent(0); r++)
{
   for (std::size_t c = 0; c < m.extent(1); c++)
   {
      m[r, c] = r * m.extent(1) + c + 1;
   }
} 
```

这里唯一改变的是 `extent()` 方法的使用，它之前是 `matrix` 类的一个函数模板成员。然而，这只是一个细节。实际上，我们可以将 `matrix` 定义为一个别名模板，如下所示：

```cpp
template <typename T, std::size_t ROWS, std::size_t COLS>
using matrix = std::mdspan<T, std::extents<std::size_t, ROWS, COLS>>;
std::array<int, 6> arr;
matrix<int, 2, 3> ma {arr.data()}; 
```

在这个例子中，`mdspan` 是二维的，但它可以定义在任何数量的维度上。`mdspan` 类型的接口包括以下成员：

| **名称** | **描述** |
| --- | --- |
| `operator[]` | 提供对底层数据的访问。 |
| `size()` | 返回元素的数量。 |
| `empty()` | 指示元素数量是否为零。 |
| `stride()` | 返回指定维度的步长。除非明确自定义，否则默认为 1。 |
| `extents()` | 返回指定维度的尺寸（范围）。 |

表 6.5：mdspan 的一些成员函数列表

如果你查看 `std::mdspan` 类的定义，你会看到以下内容：

```cpp
template<class T,
         class Extents,
         class LayoutPolicy = std::layout_right,
         class AccessorPolicy = std::default_accessor<T>>
class mdspan; 
```

前两个模板参数是元素类型和每个维度的范围（大小）。我们在前面的例子中看到了这些。然而，最后两个是定制点：

+   布局策略控制`mdspan`的多维索引如何映射到一维底层数据的偏移量。有几种选项可供选择：`layout_right`（默认）表示最右边的索引提供对底层内存的步长为 1 的访问（这是 C/C++风格）；`layout_left`表示最左边的索引提供对底层内存的步长为 1 的访问（这是 Fortran 和 Matlab 风格）；以及`layout_stride`，它泛化了前两种，并允许在每个范围上自定义步长。拥有布局策略的原因是与其他语言互操作以及在不改变算法循环结构的情况下更改算法的数据访问模式。

+   访问策略定义了底层序列如何存储其元素以及如何使用布局策略的偏移量来获取存储元素的引用。这些主要用于第三方库。对于`std::mdspan`实现访问策略的可能性不大，正如定义`std::vector`的分配器一样不太可能。

让我们举例说明布局策略，以了解它们是如何工作的。默认的是`std::layout_right`。我们可以考虑这个例子，它明确指定了策略：

```cpp
std::vector v {1,2,3,4,5,6,7,8,9};
std::mdspan<int, 
            std::extents<size_t, 2, 3>,
            std::layout_right> mv{v.data()}; 
```

这里定义的二维矩阵具有以下内容：

```cpp
1 2 3
4 5 6 
```

然而，如果我们将布局策略更改为`std::layout_left`，那么内容也会更改为以下：

```cpp
1 3 5
2 4 6 
defines a stride equivalent to the std::layout_right, for the 2x3 matrix we have seen so far:
```

```cpp
std::mdspan<int, 
            std::extents<size_t, 
                         std::dynamic_extent, 
                         std::dynamic_extent>, 
            std::layout_stride> 
mv{ v.data(), { std::dextents<size_t,2>{2, 3}, 
                std::array<std::size_t, 2>{3, 1}}}; 
```

然而，不同的步长提供不同的结果。以下表格中显示了几个示例：

| **步长** | **矩阵** |
| --- | --- |
| {0, 0} | 1 1 11 1 1 |
| {0, 1} | 1 2 31 2 3 |
| {1, 0} | 1 1 12 2 2 |
| {1, 1} | 1 2 32 3 4 |
| {2, 1} | 1 2 33 4 5 |
| {1, 2} | 1 3 52 4 6 |
| {2, 3} | 1 4 73 6 9 |

表 6.6：自定义步长和结果视图的内容示例

让我们讨论最后一个例子，它可能更为通用。第一个范围步长代表行的偏移增量。第一个元素在底层序列中的索引为 0。因此，步长为 2，如本例所示，表示从索引 0、2、4 等读取行。第二个范围步长代表列的偏移增量。第一个元素对应行的索引。在这个例子中，第一行的索引为 0，因此列的步长为 3 意味着第一行的元素将从索引 0、3 和 6 读取。第二行从索引 2 开始。因此，第二行的元素将从索引 2、5 和 8 读取。这是前表中显示的最后一个例子。

## 还有更多...

`mdspan`的原始提案包括一个名为`submdspan()`的免费函数。此函数创建一个`mdspan`的切片，或者说，是`mdspan`子集的视图。为了使`mdspan`能够包含在 C++23 中，此函数被移除并移至 C++26。在撰写本书时，它已经包含在 C++26 中，尽管还没有编译器支持它。

## 参见

+   *使用`std::span`处理连续对象序列*，了解如何使用对连续元素序列的非拥有视图

# 注册在程序正常退出时被调用的函数

程序在退出时通常需要清理代码以释放资源，向日志中写入内容，或者执行其他结束操作。标准库提供了两个实用函数，使我们能够注册在程序正常终止时被调用的函数，无论是通过从`main()`返回还是通过调用`std::exit()`或`std::quick_exit()`。这对于需要在程序终止前执行操作而无需用户显式调用结束函数的库特别有用。在本食谱中，你将学习如何安装退出处理程序以及它们是如何工作的。

## 准备工作

本食谱中讨论的所有函数，`exit()`、`quick_exit()`、`atexit()`和`at_quick_exit()`，都可在`<cstdlib>`头文件中`std`命名空间中找到。

## 如何操作...

要注册在程序终止时被调用的函数，你应该使用以下方法：

+   使用`std::atexit()`注册在从`main()`返回或调用`std::exit()`时被调用的函数：

    ```cpp
    void exit_handler_1()
    {
      std::cout << "exit handler 1" << '\n';
    }
    void exit_handler_2()
    {
      std::cout << "exit handler 2" << '\n';
    }
    std::atexit(exit_handler_1);
    std::atexit(exit_handler_2);
    std::atexit([]() {std::cout << "exit handler 3" << '\n'; }); 
    ```

+   使用`std::at_quick_exit()`注册在调用`std::quick_exit()`时被调用的函数：

    ```cpp
    void quick_exit_handler_1()
    {
      std::cout << "quick exit handler 1" << '\n';
    }
    void quick_exit_handler_2()
    {
      std::cout << "quick exit handler 2" << '\n';
    }
    std::at_quick_exit(quick_exit_handler_1);
    std::at_quick_exit(quick_exit_handler_2);
    std::at_quick_exit([]() {
      std::cout << "quick exit handler 3" << '\n'; }); 
    ```

## 它是如何工作的...

不论使用何种方法注册的退出处理程序，只有在程序正常或快速终止时才会被调用。如果以异常方式终止，通过调用`std::terminate()`或`std::abort()`，则它们都不会被调用。如果任何处理程序通过异常退出，则调用`std::terminate()`。退出处理程序不得有任何参数，并且必须返回`void`。一旦注册，退出处理程序就不能取消注册。

程序可以安装多个处理程序。标准保证每种方法至少可以注册 32 个处理程序，尽管实际实现可以支持任何更高的数字。`std::atexit()`和`std::at_quick_exit()`都是线程安全的，因此可以从不同的线程同时调用，而不会产生竞争条件。

如果注册了多个处理程序，则它们将按照注册的相反顺序被调用。以下表格显示了注册了退出处理程序的程序（如前节所示）在通过`std::exit()`调用和`std::quick_exit()`调用终止时的输出：

| `std::exit(0);` | `std::quick_exit(0);` |
| --- | --- |

|

```cpp
exit handler 3
exit handler 2
exit handler 1 
```

|

```cpp
quick exit handler 3
quick exit handler 2
quick exit handler 1 
```

|

表 6.7：当由于调用 exit()和 quick_exit()而退出时，前一个代码片段的输出

另一方面，在程序正常终止时，具有局部存储期的对象的析构、具有静态存储期的对象的析构以及调用已注册的退出处理程序是并发执行的。然而，可以保证在静态对象的构造之前注册的退出处理程序将在该静态对象析构之后调用，而在静态对象构造之后注册的退出处理程序将在该静态对象析构之前调用。

为了更好地说明这一点，让我们考虑以下类：

```cpp
struct static_foo
{
  ~static_foo() { std::cout << "static foo destroyed!" << '\n'; }
  static static_foo* instance()
 {
    static static_foo obj;
    return &obj;
  }
}; 
```

在这个上下文中，我们将引用以下代码片段：

```cpp
std::atexit(exit_handler_1);
static_foo::instance();
std::atexit(exit_handler_2);
std::atexit([]() {std::cout << "exit handler 3" << '\n'; });
std::exit(42); 
exit_handler_1 is registered before the creation of the static object static_foo. On the other hand, exit_handler_2 and the lambda expression are both registered, in that order, after the static object was constructed. As a result, the order of calls at normal termination is as follows:
```

1.  Lambda 表达式

1.  `exit_handler_2`

1.  `static_foo`的析构函数

1.  `exit_handler_1`

上一程序的输出如下所示：

```cpp
exit handler 3
exit handler 2
static foo destroyed!
exit handler 1 
```

当使用`std::at_quick_exit()`时，在正常程序终止的情况下，不会调用已注册的函数。如果需要在那种情况下调用函数，您必须使用`std::atexit()`来注册它。

## 参见

+   *第三章*，*使用标准算法与 lambda 表达式*，以探索 lambda 表达式的基础知识以及如何利用它们与标准算法

# 使用类型特性查询类型的属性

模板元编程是语言的一个强大功能，它使我们能够编写和重用适用于所有类型的通用代码。在实践中，通常有必要使通用代码对不同的类型工作方式不同，或者根本不工作，无论是出于意图还是为了语义正确性、性能或其他原因。例如，您可能希望通用算法对 POD 和非 POD 类型有不同的实现，或者函数模板仅对整数类型进行实例化。C++11 提供了一套类型特性来帮助解决这个问题。

类型特性基本上是元类型，它们提供了关于其他类型的信息。类型特性库包含了一个用于查询类型属性（例如检查一个类型是否是整数类型或两个类型是否相同）的特性和类型转换（例如移除`const`和`volatile`限定符或向类型添加指针）的长列表。我们已经在本书的几个配方中使用了类型特性；然而，在这个配方中，我们将探讨类型特性是什么以及它们是如何工作的。

## 准备工作

在 C++11 中引入的所有类型特性都在`<type_traits>`头文件中的`std`命名空间中可用。

类型特性可以在许多元编程上下文中使用，并且在这本书中，我们已经看到它们在各种情况下被使用。在这个配方中，我们将总结一些这些用例，并了解类型特性是如何工作的。

在这个配方中，我们将讨论完全和部分模板特化。对这些概念的了解将帮助您更好地理解类型特性的工作方式。

## 如何做...

以下列表显示了使用类型特性实现各种设计目标的各种情况：

+   使用 `enable_if` 来定义函数模板可以实例化的类型的先决条件：

    ```cpp
    template <typename T,
              typename = typename std::enable_if_t<
                    std::is_arithmetic_v<T> > >
    T multiply(T const t1, T const t2)
    {
      return t1 * t2;
    }
    auto v1 = multiply(42.0, 1.5);     // OK
    auto v2 = multiply("42"s, "1.5"s); // error 
    ```

+   使用 `static_assert` 来确保满足不变性：

    ```cpp
    template <typename T>
    struct pod_wrapper
    {
      static_assert(std::is_standard_layout_v<T> &&
                    std::is_trivial_v<T>,
                    "Type is not a POD!");
      T value;
    };
    pod_wrapper<int> i{ 42 };            // OK
    pod_wrapper<std::string> s{ "42"s }; // error 
    ```

+   使用 `std::conditional` 在类型之间进行选择：

    ```cpp
    template <typename T>
    struct const_wrapper
    {
      typedef typename std::conditional_t<
                std::is_const_v<T>,
                T,
                typename std::add_const_t<T>> const_type;
    };
    static_assert(
      std::is_const_v<const_wrapper<int>::const_type>);
    static_assert(
      std::is_const_v<const_wrapper<int const>::const_type>); 
    ```

+   使用 `constexpr if` 来使编译器能够根据模板实例化的类型生成不同的代码：

    ```cpp
    template <typename T>
    auto process(T arg)
    {
      if constexpr (std::is_same_v<T, bool>)
     return !arg;
      else if constexpr (std::is_integral_v<T>)
        return -arg;
      else if constexpr (std::is_floating_point_v<T>)
        return std::abs(arg);
      else
    return arg;
    }
    auto v1 = process(false); // v1 = true
    auto v2 = process(42);    // v2 = -42
    auto v3 = process(-42.0); // v3 = 42.0
    auto v4 = process("42"s); // v4 = "42" 
    ```

## 它是如何工作的...

类型特性是提供关于类型或可以用来修改类型的元信息的类。实际上有两种类型的类型特性：

+   提供有关类型、其属性或其关系信息（如 `is_integer`、`is_arithmetic`、`is_array`、`is_enum`、`is_class`、`is_const`、`is_trivial`、`is_standard_layout`、`is_constructible`、`is_same` 等）的特性。这些特性提供了一个名为 `value` 的 `const bool` 成员。

+   修改类型属性的特性（如 `add_const`、`remove_const`、`add_pointer`、`remove_pointer`、`make_signed`、`make_unsigned` 等）。这些特性提供了一个名为 `type` 的成员 typedef，它表示转换后的类型。

这两类类型已经在 *如何做...* 部分中展示过；示例在其他菜谱中已经详细讨论和解释。为了方便，这里提供了一个简短的总结：

+   在第一个示例中，函数模板 `multiply()` 只允许用算术类型（即整数或浮点数）实例化；当用不同类型的类型实例化时，`enable_if` 不会定义一个名为 `type` 的 typedef 成员，这将产生编译错误。

+   在第二个示例中，`pod_wrapper` 是一个类模板，它应该只使用 POD 类型实例化。如果使用非 POD 类型（它既不是平凡的也不是标准布局），`static_assert` 声明将产生编译错误。

+   在第三个示例中，`const_wrapper` 是一个类模板，它提供了一个名为 `const_type` 的 typedef 成员，它表示一个常量合格类型。

+   在这个示例中，我们使用了 `std::conditional` 在编译时选择两种类型：如果类型参数 `T` 已经是一个 const 类型，那么我们只选择 `T`。否则，我们使用 `add_const` 类型特性用 `const` 说明符修饰类型。

+   在第四个示例中，`process()` 是一个包含一系列 `if constexpr` 分支的函数模板。根据在编译时通过各种类型特性（如 `is_same`、`is_integer`、`is_floating_point`）查询到的类型类别，编译器只会选择一个分支放入生成的代码中，其余的将被丢弃。因此，像 `process(42)` 这样的调用将产生以下函数模板的实例化：

    ```cpp
    int process(int arg)
    {
      return -arg;
    } 
    ```

类型特性是通过提供一个类模板及其部分或完全特化来实现的。以下是一些类型特性的概念实现示例：

+   `is_void()`方法指示一个类型是否为`void`；这使用了完全特化：

    ```cpp
    template <typename T>
    struct is_void
    { static const bool value = false; };
    template <>
    struct is_void<void>
    { static const bool value = true; }; 
    ```

+   `is_pointer()`方法指示一个类型是否是指向对象的指针或指向函数的指针；这使用了部分特化：

    ```cpp
    template <typename T>
    struct is_pointer
    { static const bool value = false; };
    template <typename T>
    struct is_pointer<T*>
    { static const bool value = true; }; 
    ```

+   `enable_if()`类型特质仅在非类型模板参数是一个评估为`true`的表达式时，为其类型模板参数定义一个类型别名：

    ```cpp
    template<bool B, typename T = void>
    struct enable_if {};
    template<typename T>
    struct enable_if<true, T> { using type = T; }; 
    ```

由于查询属性（如`std::is_integer<int>::value`）的特质或修改类型属性的特质（如`std::enable_if<true, T>::type`）中使用的`bool`成员`value`太冗长（且长），C++14 和 C++17 标准引入了一些辅助工具以简化使用：

+   形式为`std::trait_v<T>`的变量模板是`std::trait<T>::value`的别名。一个例子是`std::is_integer_v<T>`，其定义如下：

    ```cpp
    template <typename T>
    inline constexpr bool is_integral_v = is_integral<T>::value; 
    ```

+   `std::trait_t<T>`形式的别名模板是`std::trait<T>::type`的别名。一个例子是`std::enable_if_t<B, T>`，其定义如下：

    ```cpp
    template <bool B, typename T = void>
    using enable_if_t = typename enable_if<B,T>::type; 
    ```

注意，在 C++20 中，POD 类型的概念已被弃用。这还包括`std::is_pod`类型特质的弃用。POD 类型是一种既是*平凡的*（具有编译器提供的或显式默认的特殊成员，并占用连续的内存区域）又具有*标准布局*（不包含与 C 语言不兼容的语言特性，如虚函数，并且所有成员具有相同的访问控制）的类型。因此，从 C++20 开始，更精细的平凡和标准布局类型概念更受欢迎。这也意味着您不应再使用`std::is_pod`，而应使用`std::is_trivial`和`std::is_standard_layout`。

## 更多...

类型特质不仅限于标准库提供的。使用类似的技术，您可以定义自己的类型特质以实现各种目标。在下一道菜谱*编写自己的类型特质*中，我们将学习如何定义和使用自己的类型特质。

## 参见

+   *第四章*，*使用 constexpr if 在编译时选择分支*，了解如何仅使用`constexpr if`语句编译代码的一部分

+   *第四章*，*使用 enable_if 条件编译类和函数*，了解 SFINAE 以及如何使用它为模板指定类型约束

+   *第四章*，*使用 static_assert 进行编译时断言检查*，了解如何定义在编译时验证的断言

+   *编写自己的类型特质*，了解如何定义自己的类型特质

+   *使用`std::conditional`在类型之间进行选择*，了解如何在编译时布尔表达式中执行类型的编译时选择

# 编写自己的类型特质

在前面的菜谱中，我们学习了类型特质是什么，标准提供了哪些特质，以及它们如何用于各种目的。在本菜谱中，我们将更进一步，看看如何定义我们自己的自定义特质。

## 准备工作

在这个菜谱中，我们将学习如何解决以下问题：我们有一些支持序列化的类。不深入细节，假设其中一些提供了一种“纯”序列化到字符串的方式（无论这意味着什么），而其他一些基于指定的编码进行序列化。最终目标是创建一个单一、统一的 API 来序列化任何这些类型的对象。为此，我们将考虑以下两个类：提供简单序列化的 `foo` 类，以及提供带编码序列化的 `bar` 类。

让我们看看代码：

```cpp
struct foo
{
  std::string serialize()
 {
    return "plain"s;
  }
};
struct bar
{
  std::string serialize_with_encoding()
 {
    return "encoded"s;
  }
}; 
```

建议你在继续阅读本菜谱之前，首先阅读前面的 *使用类型特性查询类型的属性* 菜谱。

## 如何实现...

实现以下类和函数模板：

+   一个名为 `is_serializable_with_encoding` 的类模板，其中包含一个设置为 `false` 的 `static const bool` 变量：

    ```cpp
    template <typename T>
    struct is_serializable_with_encoding
    {
      static const bool value = false;
    }; 
    ```

+   `is_serializable_with_encoding` 模板对类 `bar` 的完全特化，其中 `static const bool` 变量设置为 `true`：

    ```cpp
    template <>
    struct is_serializable_with_encoding<bar>
    {
      static const bool value = true;
    }; 
    ```

+   一个名为 `serializer` 的类模板，其中包含一个名为 `serialize` 的静态模板方法，它接受一个模板类型 `T` 的参数，并调用该对象的 `serialize()`：

    ```cpp
    template <bool b>
    struct serializer
    {
      template <typename T>
      static auto serialize(T& v)
     {
        return v.serialize();
      }
    }; 
    ```

+   一个名为 `true` 的完全特化类模板，其 `serialize()` 静态方法为参数调用 `serialize_with_encoding()`：

    ```cpp
    template <>
    struct serializer<true>
    {
      template <typename T>
      static auto serialize(T& v)
     {
        return v.serialize_with_encoding();
      }
    }; 
    ```

+   一个名为 `serialize()` 的函数模板，它使用之前定义的 `serializer` 类模板和 `is_serializable_with_encoding` 类型特性，来选择应该调用实际的哪种序列化方法（纯或带编码）：

    ```cpp
    template <typename T>
    auto serialize(T& v)
    {
      return serializer<is_serializable_with_encoding<T>::value>::
        serialize(v);
    } 
    ```

## 工作原理...

`is_serializable_with_encoding` 是一个类型特性，用于检查类型 `T` 是否可以使用（指定的）编码进行序列化。它提供了一个类型 `bool` 的静态成员，名为 `value`，如果 `T` 支持使用编码进行序列化，则其值等于 `true`，否则为 `false`。它被实现为一个具有单个类型模板参数 `T` 的类模板；这个类模板对支持编码序列化的类型进行了完全特化——在这个特定例子中，对类 `bar` 进行了特化：

```cpp
std::cout << std::boolalpha;
std::cout <<
  is_serializable_with_encoding<foo>::value << '\n';        // false
std::cout <<
  is_serializable_with_encoding<bar>::value << '\n';        // true
std::cout <<
  is_serializable_with_encoding<int>::value << '\n';        // false
std::cout <<
  is_serializable_with_encoding<std::string>::value << '\n';// false
std::cout << std::boolalpha; 
```

`serialize()` 方法是一个函数模板，它代表了一个支持两种序列化方式的对象的通用 API。它接受一个类型模板参数 `T` 的单个参数，并使用辅助类模板 `serializer` 来调用其参数的 `serialize()` 或 `serialize_with_encoding()` 方法。

`serializer`类型是一个只有一个非类型模板参数（类型为`bool`）的类模板。这个类模板包含一个名为`serialize()`的静态函数模板。这个函数模板接受一个类型模板参数`T`的单个参数，对参数调用`serialize()`，并返回该调用返回的值。`serializer`类模板对其非类型模板参数的值`true`有一个完全特化。在这个特化中，函数模板`serialize()`具有未更改的签名，但调用`serialize_with_encoding()`而不是`serialize()`。

在`serialize()`函数模板中使用`is_serializable_with_encoding`类型特性来完成使用泛型或完全特化的类模板之间的选择。类型特性中的静态成员`value`用作`serializer`的非类型模板参数的参数。

在定义了所有这些之后，我们可以编写以下代码：

```cpp
foo f;
bar b;
std::cout << serialize(f) << '\n'; // plain
std::cout << serialize(b) << '\n'; // encoded 
serialize() with a foo argument will return the string *plain*, while calling serialize() with a bar argument will return the string *encoded*.
```

## 参见

+   *使用类型特性查询类型的属性*，探索一种 C++元编程技术，允许我们检查和转换类型的属性

+   *使用 std::conditional 在类型之间进行选择*，了解如何在编译时基于编译时布尔表达式执行类型的编译时选择

# 使用 std::conditional 在类型之间进行选择

在前面的配方中，我们查看了一些类型支持库的功能，特别是类型特性。相关主题在其他部分的本章中有所讨论，例如在*第四章*，*预处理和编译*中使用`std::enable_if`来隐藏函数重载，以及在本章讨论访问变体时使用的`std::decay`来移除`const`和`volatile`限定符。另一个值得更深入讨论的类型转换功能是`std::conditional`，它允许我们根据编译时布尔表达式在编译时选择两种类型。在本配方中，您将通过几个示例了解它是如何工作的以及如何使用它。

## 准备工作

建议您首先阅读本章前面提到的*使用类型特性查询类型的属性*配方。

## 如何做...

以下是一些示例，展示了如何使用在`<type_traits>`头文件中可用的`std::conditional`（以及`std::conditional_t`），在编译时选择两种类型：

+   在类型别名或 typedef 中，根据平台选择 32 位和 64 位整数类型（在 32 位平台上指针大小为 4 字节，在 64 位平台上为 8 字节）：

    ```cpp
    using long_type = std::conditional_t<
        sizeof(void*) <= 4, long, long long>;
    auto n = long_type{ 42 }; 
    ```

+   在别名模板中，根据用户指定（作为一个非类型模板参数）选择 8 位、16 位、32 位或 64 位整数类型：

    ```cpp
    template <int size>
    using number_type =
      typename std::conditional_t<
        size<=1,
        std::int8_t,
        typename std::conditional_t<
          size<=2,
          std::int16_t,
          typename std::conditional_t<
            size<=4,
            std::int32_t,
            std::int64_t
          >
        >
      >;
    auto n = number_type<2>{ 42 };
    static_assert(sizeof(number_type<1>) == 1);
    static_assert(sizeof(number_type<2>) == 2);
    static_assert(sizeof(number_type<3>) == 4);
    static_assert(sizeof(number_type<4>) == 4);
    static_assert(sizeof(number_type<5>) == 8);
    static_assert(sizeof(number_type<6>) == 8);
    static_assert(sizeof(number_type<7>) == 8);
    static_assert(sizeof(number_type<8>) == 8);
    static_assert(sizeof(number_type<9>) == 8); 
    ```

+   在类型模板参数中，根据类型模板参数是整数类型还是实数均匀分布类型来选择，具体取决于类型模板参数是否为整数类型：

    ```cpp
    template <typename T,
              typename D = std::conditional_t<
                             std::is_integral_v<T>,
                             std::uniform_int_distribution<T>,
                             std::uniform_real_distribution<T>>,
              typename = typename std::enable_if_t<
                             std::is_arithmetic_v<T>>>
    std::vector<T> GenerateRandom(T const min, T const max,
                                  size_t const size)
    {
      std::vector<T> v(size);
      std::random_device rd{};
      std::mt19937 mt{ rd() };
      D dist{ min, max };
      std::generate(std::begin(v), std::end(v),
        [&dist, &mt] {return dist(mt); });
      return v;
    }
    auto v1 = GenerateRandom(1, 10, 10);     // integers
    auto v2 = GenerateRandom(1.0, 10.0, 10); // doubles 
    ```

## 它是如何工作的...

`std::conditional`是一个类模板，它定义了一个名为`type`的成员，该成员可以是它的两个类型模板参数之一。这个选择是基于作为非类型模板参数的编译时常量布尔表达式提供的。它的实现如下所示：

```cpp
template<bool Test, class T1, class T2>
struct conditional
{
  typedef T2 type;
};
template<class T1, class T2>
struct conditional<true, T1, T2>
{
  typedef T1 type;
}; 
```

让我们总结一下上一节中的例子：

+   在第一个例子中，如果平台是 32 位的，那么指针类型的大小是 4 字节，因此编译时表达式`sizeof(void*) <= 4`是`true`；因此，`std::conditional`将其成员类型定义为`long`。如果平台是 64 位的，那么条件评估为`false`，因为指针类型的大小是 8 字节；因此，成员类型被定义为`long long`。

+   在第二个例子中，也遇到了类似的情况，其中多次使用`std::conditional`来模拟一系列`if...else`语句以选择合适的数据类型。

+   在第三个例子中，我们使用了别名模板`std::conditional_t`来简化函数模板`GenerateRandom`的声明。在这里，`std::conditional`用于定义表示统计分布的类型模板参数的默认值。根据第一个类型模板参数`T`是整数类型还是浮点类型，默认分布类型将在`std::uniform_int_distribution<T>`和`std::uniform_real_distribution<T>`之间选择。通过使用带有第三个模板参数的`std::enable_if`来禁用其他类型的使用，正如我们在其他菜谱中已经看到的那样。

为了帮助简化`std::conditional`的使用，C++14 提供了一个名为`std::conditional_t`的别名模板，我们在这里已经看到过，其定义如下：

```cpp
template<bool Test, class T1, class T2>
using conditional_t = typename conditional_t<Test,T1,T2>; 
```

使用这个辅助类（以及许多其他类似的标准库中的类）是可选的，但有助于编写更简洁的代码。

## 参见

+   *使用类型特性查询类型的属性*，探索一种 C++元编程技术，该技术允许我们检查和转换类型的属性

+   *编写自己的类型特性*，学习如何定义自己的类型特性

+   *第四章*，*使用 enable_if 条件编译类和函数*，学习 SFINAE 及其如何用于指定模板的类型约束

# 使用 source_location 提供日志细节

调试是软件开发的一个基本部分。无论它多么简单或复杂，没有程序会从第一次尝试就按预期工作。因此，开发者会花费大量时间调试他们的代码，使用从调试器到打印到控制台或文本文件的多种工具和技术。有时，我们希望在日志中提供有关消息来源的详细信息，包括文件、行号和可能的功能名。尽管这可以通过一些标准宏实现，但在 C++20 中，一个新的实用类型 `std::source_location` 允许我们以现代方式完成它。在本食谱中，我们将学习如何实现。

## 如何做…

要记录包括文件名、行号和函数名的信息，请执行以下操作：

+   定义一个带有所有需要提供的信息（如消息、严重性等）参数的日志函数。

+   添加一个类型为 `std::source_location` 的额外参数（您必须包含 `<source_location>` 头文件），默认值为 `std::source_location::current()`。

+   使用成员函数 `file_name()`、`line()`、`column()` 和 `function_name()` 来检索调用源的信息。

这里展示了一个这样的日志函数的例子：

```cpp
void log(std::string_view message, 
         std::source_location const location = std::source_location::current())
{
   std::cout   << location.file_name() << '('
               << location.line() << ':'
               << location.column() << ") '"
               << location.function_name() << "': "
               << message << '\n';
} 
```

## 它是如何工作的…

在 C++20 之前，如源文件、行和函数名之类的日志信息只能通过几个宏来实现：

+   `__FILE__`，它展开为当前文件的名称

+   `__LINE__`，它展开为源文件行号

此外，所有支持的编译器都包括非标准宏，如 `__func__` / `__FUNCTION__`，它们提供当前函数的名称。

使用这些宏，可以编写以下日志函数：

```cpp
void log(std::string_view message, 
         std::string_view file, 
 int line, 
         std::string_view function)
{
   std::cout << file << '('
             << line << ") '"
             << function << "': "
             << message << '\n';
} 
```

然而，必须从函数执行的上下文中使用这些宏，如下面的代码片段所示：

```cpp
int main()
{
   log("This is a log entry!", __FILE__, __LINE__, __FUNCTION__);
} 
```

运行此函数的结果在控制台上看起来如下：

```cpp
[...]\source.cpp(23) 'main': This is a log entry! 
```

C++20 的 `std::source_line` 由于以下几个原因是一个更好的替代方案：

+   您不再需要依赖于宏。

+   它包括关于列的信息，而不仅仅是行。

+   它可以用在日志函数签名中，简化调用过程。

在 *如何做…* 部分定义的 `log()` 函数可以这样调用：

```cpp
int main()
{
   log("This is a log entry!");
} 
```

这将产生以下输出：

```cpp
[...]\source.cpp(23:4) 'int __cdecl main(void)': This is a log entry! 
```

尽管存在默认构造函数，但它使用默认值初始化数据。要获取正确的值，必须调用静态成员函数 `current()`。此函数的工作方式如下：

+   当在函数调用中直接调用时，它使用调用位置的信息初始化数据。

+   当用作默认成员初始化器时，它使用初始化数据成员的构造函数聚合初始化的数据位置信息初始化数据。

+   当在默认参数（如这里所示的示例）中使用时，它使用调用点的位置初始化数据。

+   在其他上下文中使用时，行为是未定义的。

必须注意，预处理器指令 `#line` 会改变源代码的行号和文件名。这会影响宏 `__FILE__` 和 `__LINE__` 返回的值。`std::source_location` 也以相同的方式受到 `#line` 指令的影响。

## 参见

+   *使用堆栈跟踪库打印调用栈*，了解如何遍历或打印当前堆栈跟踪的内容

# 使用堆栈跟踪库打印调用序列

在前面的菜谱中，我们看到了如何使用 C++20 `std::source_location` 为日志记录、测试和调试目的提供源位置信息。另一种调试机制由断言表示，但它们并不总是足够，因为我们经常需要知道导致执行点的调用序列。这被称为堆栈跟踪。C++23 标准包含一个新的诊断实用工具库。这允许我们打印堆栈跟踪。在本菜谱中，您将学习如何使用这些诊断实用工具。

## 如何做到这一点...

您可以使用 C++23 堆栈跟踪库来：

+   打印堆栈跟踪的整个内容：

    ```cpp
    std::cout << std::stacktrace::current() << '\n'; 
    ```

+   遍历堆栈跟踪中的每一帧并打印它：

    ```cpp
    for (auto const & frame : std::stacktrace::current())
    {
       std::cout << frame << '\n';
    } 
    ```

+   遍历堆栈跟踪中的每一帧并检索其信息：

    ```cpp
    for (auto const& frame : std::stacktrace::current())
    {
       std::cout << frame.source_file()
                 << "("<< frame.source_line() << ")"
                 << ": " << frame.description()
                 << '\n';
    } 
    ```

## 它是如何工作的...

新的诊断实用工具包含在一个名为 `<stacktrace>` 的单独头文件中。此头文件包含以下两个类：

+   `std::basic_stacktrace`，这是一个表示堆栈跟踪条目序列的类模板。定义了一个类型别名 `std::stacktrace`，作为 `std::basic_stacktrace<std::allocator<std::stacktrace_entry>>`。

+   `std::stacktrace_entry`，它表示堆栈跟踪中的一个评估。

在讨论调用序列时，有两个术语需要正确理解：**调用栈**和**堆栈跟踪**。调用栈是用于存储运行程序中活动帧（调用）信息的数结构。堆栈跟踪是在某个时间点对调用栈的快照。

虽然 `std::basic_stacktrace` 是一个容器，但它不是由用户实例化并填充堆栈条目的。堆栈跟踪序列中没有用于添加或删除元素的成员函数；然而，有用于元素访问的成员函数（`at()` 和 `operator[]`）以及检查大小（`capacity()`、`size()` 和 `max_size()`）。为了获取调用栈的快照，您必须调用静态成员函数 `current()`：

```cpp
std::stacktrace trace = std::stacktrace::current(); 
```

当前跟踪可以以多种方式打印：

+   使用重载的 `operator<<` 操作符到一个输出流：

    ```cpp
    std::cout << std::stacktrace::current() << '\n'; 
    ```

+   使用成员函数 `to_string()` 将其转换为 `std::string`：

    ```cpp
    std::cout << std::to_string(std::stacktrace::current()) 
              << '\n'; 
    ```

+   使用格式化函数，如 `std::format()`。请注意，不允许使用任何格式化说明符：

    ```cpp
    auto str = std::format("{}\n", std::stacktrace::current());
    std::cout << str; 
    ```

以下代码片段展示了如何将堆栈跟踪打印到标准输出：

```cpp
int plus_one(int n)
{
   std::cout << std::stacktrace::current() << '\n';
   return n + 1;
}
int double_n_plus_one(int n)
{
   return plus_one(2 * n);
}
int main()
{
   std::cout << double_n_plus_one(42) << '\n';
} 
```

运行此程序的结果会根据编译器和目标系统而有所不同，但以下是一个可能的输出示例：

```cpp
0> [...]\main.cpp(24): chapter06!plus_one+0x4F
1> [...]\main.cpp(37): chapter06!double_n_plus_one+0xE
2> [...]\main.cpp(61): chapter06!main+0x5F
3> D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl(78): chapter06!invoke_main+0x33
4> D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl(288): chapter06!__scrt_common_main_seh+0x157
5> D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl(331): chapter06!__scrt_common_main+0xD
6> D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_main.cpp(17): chapter06!mainCRTStartup+0x8
7> KERNEL32+0x17D59
8> ntdll!RtlInitializeExceptionChain+0x6B
9> ntdll!RtlClearBits+0xBF 
```

对于如上所示的跟踪条目，我们可以识别出三个部分：源文件、行号和评估的描述。这些内容如下所示：

```cpp
[...]\main.cpp(24): chapter06!main+0x5F
-------------- --   -------------------
source         line description 
```

这些部分可以独立获取，使用 `std::stacktrace_entry` 的成员函数 `source_file()`、`source_line()` 和 `description()`。可以从 `stacktrace` 容器迭代堆栈跟踪条目的序列，或者使用成员函数 `at()` 和 `operator[]` 访问。

## 参见

+   *使用 `source_location` 提供日志详细信息*，了解如何使用 C++20 的 `source_location` 类来显示有关源文件、行和函数名称的信息

# 在 Discord 上了解更多

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
