# 第八章：日期和时间库

这是一个简短的章节，向您展示如何使用不同的 Boost 库执行基本的日期和时间计算。大多数实际软件都以某种形式使用日期和时间测量。应用程序计算当前日期和时间，以生成应用程序活动的时间日志。专门的程序根据复杂的调度策略计算作业的时间表，并等待特定的时间点或时间间隔过去。有时，应用程序甚至会监视自己的性能和执行速度，并在需要时采取补救措施或发出通知。

在本章中，我们将介绍使用 Boost 库进行日期和时间计算以及测量代码性能。这些主题分为以下几个部分：

+   使用 Boost `Date Time`进行日期和时间计算

+   使用 Boost Chrono 测量时间

+   使用 Boost Timer 测量程序性能

# 使用 Boost Date Time 进行日期和时间计算

日期和时间计算在许多软件应用程序中都很重要，但是 C++03 对于操作日期和执行计算的支持有限。Boost `Date Time`库提供了一组直观的接口，用于表示日期、时间戳、持续时间和时间间隔。通过允许涉及日期、时间戳、持续时间的简单算术运算，并补充一组有用的日期/时间算法，它可以使用很少的代码进行相当复杂的时间和日历计算。

## 公历中的日期

公历，也称为基督教历，由教皇格里高利十三世于 1582 年 2 月引入，并在接下来的几个世纪内取代了儒略历在绝大多数西方世界的使用。`Date_Time`库提供了一组用于表示日期和相关数量的类型：

+   `boost::gregorian::date`：我们使用这种类型来表示公历中的日期。

+   `boost::gregorian::date_duration`：除了日期，我们还需要表示日期间的持续时间——以天为单位的两个给定日期之间的时间长度。为此，我们使用`boost::gregorian::date_duration`类型。它指的是与`boost::gregorian::days`相同的类型。

+   `boost::date_period`：使用`boost::date_period`类型表示日历中从给定日期开始并延续一段特定持续时间的固定日期周期。

### 创建日期对象

我们可以使用日期的组成部分，即年份、月份和日期，创建`boost::gregorian::date`类型的对象。此外，还有许多工厂函数可以解析不同格式的日期字符串，以创建`date`对象。在下面的示例中，我们演示了创建`date`对象的不同方法：

**清单 8.1：使用 boost::gregorian::date**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 namespace greg = boost::gregorian;
 5
 6 int main() {
 7   greg::date d0;  // default constructed, is not a date
 8   assert(d0.is_not_a_date());
 9   // Construct dates from parts
10   greg::date d1(1948, greg::Jan, 30);
11   greg::date d2(1968, greg::Apr, 4);
12
13   // Construct dates from string representations
14   greg::date dw1 = greg::from_uk_string("15/10/1948");
15   greg::date dw2 = greg::from_simple_string("1956-10-29");
16   greg::date dw3 = greg::from_undelimited_string("19670605");
17   greg::date dw4 = greg::from_us_string("10-06-1973");
18
19   // Current date
20   greg::date today = greg::day_clock::local_day();
21   greg::date londonToday = greg::day_clock::universal_day();
22
23   // Take dates apart
24   std::cout << today.day_of_week() << " " << today.day() << ", "
25             << today.month() << ", " << today.year() << '\n';
26 }
```

默认构造的日期表示无效日期（第 7 行）；`is_not_a_date`成员谓词对于这样的日期返回 true（第 8 行）。我们可以从其组成部分构造日期：年、月和日。月份可以使用名为`Jan`、`Feb`、`Mar`、`Apr`、`May`、`Jun`、`Jul`、`Aug`、`Sep`、`Oct`、`Nov`和`Dec`的`enum`值来表示，这些是年份的英文缩写。使用特殊的工厂函数，可以从其他标准表示中构造日期。我们使用`boost::gregorian::from_uk_string`函数从 DD/MM/YYYY 格式的字符串中构造一个`date`对象，这是英国的标准格式（第 14 行）。`boost::gregorian::from_us_string`函数用于从美国使用的 MM/DD/YYYY 格式的字符串中构造一个`date`（第 17 行）。`boost::gregorian::from_simple_string`函数用于从 ISO 8601 YYYY-MM-DD 格式的字符串中构造一个`date`（第 15 行），并且其无分隔形式 YYYYMMDD 可以使用`boost::gregorian::from_undelimited_string`函数转换为`date`对象（第 16 行）。

**时钟**提供了一种在系统上检索当前日期和时间的方法。Boost 为此提供了几个时钟。`day_clock`类型提供了`local_day`（第 20 行）和`universal_day`（第 21 行）函数，它们返回本地和 UTC 时区的当前日期，这两者可能相同，也可能相差一天，这取决于时区和时间。

使用方便的访问器成员函数，如`day`，`month`，`year`和`day_of_week`，我们可以获取`date`的部分（第 24-25 行）。

### 注意

`Date_Time`库不是一个仅包含头文件的库，为了在本节中运行示例，它们必须链接到`libboost_date_time`库。在 Unix 上，使用 g++，您可以使用以下命令行来编译和链接涉及 Boost Date Time 的示例：

```cpp
$ g++ example.cpp -o example -lboost_date_time
```

有关更多详细信息，请参见第一章*介绍 Boost*。

### 处理日期持续时间

两个日期之间的时间持续时间由`boost::gregorian::date_duration`表示。在下面的示例中，我们计算日期之间的时间持续时间，并将持续时间添加到日期或从日期中减去以得到新的日期。

**清单 8.2：基本日期算术**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 namespace greg = boost::gregorian;
 4
 5 int main() {
 6   greg::date d1(1948, greg::Jan, 30);
 7   greg::date d2(1968, greg::Apr, 4);
 8
 9   greg::date_duration day_diff = d2 - d1;
10   std::cout << day_diff.days() 
11             << " days between the two dates\n";
12
13   greg::date six_weeks_post_d1 = d1 + greg::weeks(6);
14   std::cout << six_weeks_post_d1 << '\n';
15
16   greg::date day_before_d2 = d2 - greg::days(1);
17   std::cout << day_before_d2 << '\n';
18 }
```

我们计算持续时间（可以是负数）作为两个日期的差异（第 9 行），并以天为单位打印出来（第 10 行）。`date_duration`对象在内部以天为单位表示持续时间。我们还可以使用类型`boost::gregorian::weeks`，`boost::gregorian::months`和`boost::gregorian::years`来构造以周、月或年为单位的`date_duration`对象。请注意，`boost::gregorian::days`和`boost::gregorian::date_duration`指的是相同的类型。我们通过将持续时间添加到日期或从日期中减去它们（第 13、16 行）来获得新的日期。

### 日期周期

以固定日期开始的周期由类型`boost::gregorian::date_period`表示。在下面的示例中，我们构造了两个日期周期，一个是日历年，一个是美国财政年。我们计算它们的重叠期，然后确定重叠期内每个月的最后一个星期五的日期。

**清单 8.3：日期周期和日历计算**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 namespace greg = boost::gregorian;
 4 namespace dt = boost::date_time;
 5
 6 int main() {
 7   greg::date startCal(2015, greg::Jan, 1);
 8   greg::date endCal(2015, greg::Dec, 31);
 9
10   greg::date startFiscal(2014, greg::Oct, 1);
11   greg::date endFiscal(2015, greg::Sep, 30);
12
13   greg::date_period cal(startCal, endCal);
14   greg::date_period fisc(startFiscal, endFiscal);
15
16   std::cout << "Fiscal year begins " << fisc.begin()
17     << " and ends " << fisc.end() << '\n';
18
19   if (cal.intersects(fisc)) {
20     auto overlap = cal.intersection(fisc);
21     greg::month_iterator miter(overlap.begin());
22
23     while (*miter < overlap.end()) {
24       greg::last_day_of_the_week_in_month 
25                    last_weekday(greg::Friday, miter->month());
26       std::cout << last_weekday.get_date(miter->year())
27                 << '\n';
28       ++miter;
29     }
30   }
31 }
```

我们根据开始日期和结束日期定义日期周期（第 13、14 行）。我们可以使用`date_period`的`intersects`成员函数（第 19 行）检查两个周期是否重叠，并使用`intersection`成员函数（第 20 行）获取重叠期。我们通过在开始日期处创建一个`month_iterator`（第 21 行），并使用预增量运算符（第 28 行）迭代到结束日期（第 23 行）来遍历一个周期。有不同类型的迭代器，具有不同的迭代周期。我们使用`boost::gregorian::month_iterator`来迭代周期内连续的月份。`month_iterator`每次递增时都会将日期提前一个月。您还可以使用其他迭代器，如`year_iterator`，`week_iterator`和`day_iterator`，它们分别以年、周或天为单位递增迭代器。

对于周期中的每个月，我们想要找到该月的最后一个星期五的日期。`Date Time`库具有一些有趣的算法类，用于此类日历计算。我们使用`boost::gregorian::last_day_of_the_week_in_month`算法来执行这样的计算，以确定月份的最后一个星期五的日期。我们构造了一个`last_day_of_the_week_in_month`对象，构造函数参数是星期几（星期五）和月份（第 24、25 行）。然后我们调用它的`get_date`成员函数，将特定年份传递给它（第 26 行）。

## Posix 时间

`Date_Time`库还提供了一组类型，用于表示时间点、持续时间和周期。

+   `boost::posix_time::ptime`：特定的时间点，或者**时间点**，由类型`boost::posix_time::ptime`表示。

+   `boost::posix_time::time_duration`：与日期持续时间一样，两个时间点之间的时间长度称为**时间持续时间**，并由类型`boost::posix_time::time_duration`表示。

+   `boost::posix_time::time_period`：从特定时间点开始的固定间隔，到另一个时间点结束，称为**时间段**，由类型`boost::posix_time::time_period`表示。

这些类型及其上的操作一起定义了一个**时间系统**。Posix Time 使用`boost::gregorian::date`来表示时间点的日期部分。

### 构造时间点和持续时间

我们可以从其组成部分，即日期、小时、分钟、秒等创建`boost::posix_time::ptime`的实例，或者使用解析时间戳字符串的工厂函数。在以下示例中，我们展示了创建`ptime`对象的不同方式：

清单 8.4：使用 boost::posix_time

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 #include <ctime>
 5 namespace greg = boost::gregorian;
 6 namespace pt = boost::posix_time;
 7
 8 int main() {
 9   pt::ptime pt; // default constructed, is not a time
10   assert(pt.is_not_a_date_time());
11
12   // Get current time
13   pt::ptime now1 = pt::second_clock::universal_time();
14   pt::ptime now2 = pt::from_time_t(std::time(0));
15
16   // Construct from strings
17   // Create time points using durations
18   pt::ptime pt1(greg::day_clock::universal_day(),
19           pt::hours(10) + pt::minutes(42)
20           + pt::seconds(20) + pt::microseconds(30));
21   std::cout << pt1 << '\n';
22
23   // Compute durations
24   pt::time_duration dur = now1 - pt1;
25   std::cout << dur << '\n';
26   std::cout << dur.total_microseconds() << '\n';
27
28   pt::ptime pt2(greg::day_clock::universal_day()),
29        pt3 = pt::time_from_string("2015-01-28 10:00:31.83"),
30        pt4 = pt::from_iso_string("20150128T151200");
31
32   std::cout << pt2 << '\n' << to_iso_string(pt3) << '\n'
33             << to_simple_string(pt4) << '\n';
34 }
```

就像日期对象一样，默认构造的`ptime`对象（第 9 行）不是一个有效的时间点（第 10 行）。有时钟可以用来推导一天中的当前时间，例如，`second_clock`和`microsec_clock`，它们分别以秒或微秒单位给出时间。在这些时钟上调用`local_time`和`universal_time`函数（第 13 行）将返回本地和 UTC 时区中的当前日期和时间。

`from_time_t`工厂函数传递 Unix 时间，即自 Unix 纪元（1970 年 1 月 1 日 00:00:00 UTC）以来经过的秒数，并构造一个表示该时间点的`ptime`对象（第 14 行）。当传递 0 时，C 库函数`time`返回 UTC 时区中的当前 Unix 时间。

两个时间点之间的持续时间，可以是负数，是通过计算两个时间点之间的差值来计算的（第 24 行）。它可以被流式传输到输出流中，以默认方式打印持续时间，以小时、分钟、秒和小数秒为单位。使用访问器函数`hours`、`minutes`、`seconds`和`fractional_seconds`，我们可以获取持续时间的相关部分。或者我们可以使用访问器`total_seconds`、`total_milliseconds`、`total_microseconds`和`total_nanoseconds`将整个持续时间转换为秒或亚秒单位（第 26 行）。

我们可以从一个公历日期和一个类型为`boost::posix_time::time_duration`的持续时间对象创建一个`ptime`对象（第 18-20 行）。我们可以在`boost::posix_time`命名空间中使用 shim 类型`hours`、`minutes`、`seconds`、`microseconds`等来生成适当单位的`boost::posix_time::time_duration`类型的持续时间，并使用`operator+`将它们组合起来。

我们可以仅从一个`boost::gregorian::date`对象构造一个`ptime`对象（第 28 行）。这代表了给定日期的午夜时间。我们可以使用工厂函数从不同的字符串表示中创建`ptime`对象（第 29-30 行）。函数`time_from_string`用于从“YYYY-MM-DD hh:mm:ss.xxx…”格式的时间戳字符串构造一个`ptime`实例，在该格式中，日期和时间部分由空格分隔（第 29 行）。函数`from_iso_string`用于从“YYYYMMDDThhmmss.xxx…”格式的无分隔字符串构造一个`ptime`实例，其中大写 T 分隔日期和时间部分（第 30 行）。在这两种情况下，分钟、秒和小数秒是可选的，如果未指定，则被视为零。小数秒可以跟在秒后，用小数点分隔。这些格式是与地区相关的。例如，在几个欧洲地区，使用逗号代替小数点。

我们可以将`ptime`对象流式输出到输出流，比如`std::cout`（第 32 行）。我们还可以使用转换函数，比如`to_simple_string`和`to_iso_string`（第 32-33 行），将`ptime`实例转换为`string`。在英文环境中，`to_simple_string`函数将其转换为"YYYY-MM-DD hh:mm:ss.xxx…"格式。请注意，这是`time_from_string`预期的相同格式，也是在流式输出`ptime`时使用的格式。`to_iso_string`函数将其转换为"YYYYMMDDThhmmss.xxx…"格式，与`from_iso_string`预期的格式相同。

### 分辨率

时间系统可以表示的最小持续时间称为其分辨率。时间在特定系统上表示的精度，因此，有效的小数秒数取决于时间系统的分辨率。Posix 时间使用的默认分辨率是微秒（10^(-6)秒），也就是说，它不能表示比微秒更短的持续时间，因此不能区分相隔不到一微秒的两个时间点。以下示例演示了如何获取和解释时间系统的分辨率：

**清单 8.5：时间刻度和分辨率**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 namespace pt = boost::posix_time;
 4 namespace dt = boost::date_time;
 5 
 6 int main() {
 7   switch (pt::time_duration::resolution()) {
 8   case dt::time_resolutions::sec:
 9     std::cout << " second\n";
10     break;
11   case dt::time_resolutions::tenth:
12     std::cout << " tenth\n";
13     break;
14   case dt::time_resolutions::hundredth:
15     std::cout << " hundredth\n";
16     break;
17   case dt::time_resolutions::milli:
18     std::cout << " milli\n";
19     break;
20   case dt::time_resolutions::ten_thousandth:
21     std::cout << " ten_thousandth\n";
22     break;
23   case dt::time_resolutions::micro:
24     std::cout << " micro\n";
25     break;
26   case dt::time_resolutions::nano:
27     std::cout << " nano\n";
28     break;
29   default:
30     std::cout << " unknown\n";
31     break;
32   }
33   std::cout << pt::time_duration::num_fractional_digits()
34             << '\n';
35   std::cout << pt::time_duration::ticks_per_second() 
36             << '\n';
37 }
```

`time_duration`类的`resolution`静态函数返回一个枚举常量作为分辨率（第 7 行）；我们解释这个`enum`并打印一个字符串来指示分辨率（第 7-32 行）。

`num_fractional_digits`静态函数返回小数秒的有效数字位数（第 33 行）；在具有微秒分辨率的系统上，这将是 6，在具有纳秒分辨率的系统上，这将是 9。`ticks_per_second`静态函数将 1 秒转换为系统上最小可表示的时间单位（第 35 行）；在具有微秒分辨率的系统上，这将是 10⁶，在具有纳秒分辨率的系统上，这将是 10⁹。

### 时间段

与日期一样，我们可以使用`boost::posix_time::time_period`表示固定的时间段。以下是一个简短的示例，演示了如何创建时间段并比较不同的时间段：

**清单 8.6：使用时间段**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 namespace greg = boost::gregorian;
 5 namespace pt = boost::posix_time;
 6
 7 int main()
 8 {
 9   // Get current time
10   pt::ptime now1 = pt::second_clock::local_time();
11   pt::time_period starts_now(now1, pt::hours(2));
12
13   assert(starts_now.length() == pt::hours(2));
14
15   auto later1 = now1 + pt::hours(1);
16   pt::time_period starts_in_1(later1, pt::hours(3));
17
18   assert(starts_in_1.length() == pt::hours(3));
19
20   auto later2 = now1 + pt::hours(3);
21   pt::time_period starts_in_3(later2, pt::hours(1));
22
23   assert(starts_in_3.length() == pt::hours(1));
24
26   std::cout << "starts_in_1 starts at " << starts_in_1.begin()
27             << " and ends at " << starts_in_1.last() << '\n';
28
29   // comparing time periods
30   // non-overlapping
31   assert(starts_now < starts_in_3);
32   assert(!starts_now.intersects(starts_in_3));
33
34   // overlapping
35   assert(starts_now.intersects(starts_in_1));
36
37   assert(starts_in_1.contains(starts_in_3));
38 }
```

我们创建了一个名为`starts_now`的时间段，它从当前时刻开始，持续 2 小时。为此，我们使用了`time_period`的两个参数构造函数，传递了当前时间戳和 2 小时的持续时间（第 11 行）。使用`time_period`的`length`成员函数，我们验证了该时间段的长度确实为 2 小时（第 13 行）。

我们创建了另外两个时间段：`starts_in_1`从 1 小时后开始，持续 3 小时（第 16 行），`starts_in_3`从 3 小时后开始，持续 1 小时（第 20 行）。`time_period`的`begin`和`last`成员函数返回时间段中的第一个和最后一个时间点（第 26-27 行）。

我们使用关系运算符和称为`intersects`和`contains`的两个成员函数来表示三个时间段`starts_now`，`starts_in_1`和`starts_in_3`之间的关系。显然，`starts_in_1`的第一个小时与`starts_now`的最后一个小时重叠，因此我们断言`starts_now`和`starts_in_1`相交（第 35 行）。`starts_in_1`的最后一个小时与整个时间段`starts_in_3`重合，因此我们断言`starts_in_1`包含`starts_in_3`（第 37 行）。但是`starts_now`和`starts_in_3`不重叠；因此，我们断言`starts_now`和`starts_in_3`不相交（第 32 行）。

关系运算符`operator<`被定义为对于两个时间段`tp1`和`tp2`，条件`tp1 < tp2`成立当且仅当`tp1.last() < tp2.begin()`。同样，`operator>`被定义为条件`tp1 > tp2`成立当且仅当`tp1.begin() > tp2.last()`。这些定义意味着`tp1`和`tp2`是不相交的。因此，对于不相交的`time_period` `starts_now`和`starts_in_3`，关系`starts_now < starts_in_3`成立（第 31 行）。这些关系对于重叠的时间段是没有意义的。

### 时间迭代器

我们可以使用`boost::posix_time::time_iterator`来遍历一个时间段，类似于我们使用`boost::gregorian::date_iterator`的方式。下面的例子展示了这一点：

**清单 8.7：遍历一个时间段**

```cpp
 1 #include <boost/date_time.hpp>
 2 #include <iostream>
 3
 4 namespace greg = boost::gregorian;
 5 namespace pt = boost::posix_time;
 6
 7 int main()
 8 {
 9   pt::ptime now = pt::second_clock::local_time();
10   pt::ptime start_of_day(greg::day_clock::local_day());
11
12   for (pt::time_iterator iter(start_of_day, 
13          pt::hours(1)); iter < now; ++iter)
14   {
15     std::cout << *iter << '\n';
16   }
17 }
```

前面的例子打印了当天每个完成的小时的时间戳。我们实例化了一个`time_iterator`（第 12 行），将开始迭代的时间点(`start_of_day`)和迭代器每次增加的持续时间（1 小时）传递给它。我们迭代直到当前时间，通过解引用迭代器获得时间戳（第 15 行）并增加迭代器（第 13 行）。请注意，在表达式`iter < now`中，我们将迭代器与时间点进行比较，以决定何时停止迭代——这是`posix_time::time_iterator`的一个特殊属性，与其他迭代器不同。

# 使用 Chrono 来测量时间

Boost Chrono 是一个用于时间计算的库，与`Date Time`库的 Posix Time 部分有一些重叠的功能。与 Posix Time 一样，Chrono 也使用时间点和持续时间的概念。Chrono 不处理日期。它比`Date Time`库更新，实现了 C++标准委员会工作组(WG21)的一份提案中提出的设施。该提案的部分内容成为了 C++11 标准库的一部分，即`Chrono`库，Boost Chrono 上的许多讨论也适用于 Chrono 标准库(`std::chrono`)。

## 持续时间

持续时间表示一段时间间隔。持续时间具有数值大小，并且必须用时间单位表示。`boost::chrono::duration`模板用于表示任何这样的持续时间，并声明如下：

```cpp
template <typename Representation, typename Period>
class duration;
```

`Representation`类型参数标识用于持续时间大小的基础算术类型。`Period`类型参数标识滴答周期，即用于测量持续时间的一个时间单位的大小。该周期通常表示为 1 秒的比例或分数，使用一个名为`boost::ratio`的模板。

因此，如果我们想要以百分之一秒（centiseconds）表示持续时间，我们可以使用`int64_t`作为基础类型，并且可以使用比例(1/100)来表示滴答周期，因为滴答周期是一百分之一秒。使用`boost::ratio`，我们可以特化`duration`来表示百分之一秒的间隔，如下所示：

```cpp
typedef boost::chrono::duration<int64_t, boost::ratio<1, 100>> 
                                                    centiseconds;
centiseconds cs(1000);  // represents 10 seconds
```

我们创建了一个名为`centiseconds`的`typedef`，并将`1000`作为构造函数参数传递进去，这是持续时间中的百分之一秒的数量。`1000`百分之一秒相当于(1/100)*1000 秒，也就是 10 秒。

`boost::ratio`模板用于构造表示有理数的类型，即两个整数的比例。我们通过将我们的有理数的分子和分母作为两个非类型模板参数来特化`ratio`，按照这个顺序。第二个参数默认为 1；因此，要表示一个整数，比如 100，我们可以简单地写成`boost::ratio<100>`，而不是`boost::ratio<100, 1>`。表达式`boost::ratio<100>`并不代表值 100，而是封装了有理数 100 的类型。

`Chrono`库已经提供了一组预定义的`duration`的特化，用于构造以常用时间单位表示的持续时间。这些包括：

+   `boost::chrono::hours`（滴答周期=`boost::ratio<3600>`）

+   `boost::chrono::minutes`（滴答周期=`boost::ratio<60>`）

+   `boost::chrono::seconds`（滴答周期 = `boost::ratio<1>`）

+   `boost::chrono::milliseconds`（滴答周期 = `boost::ratio<1, 1000>`）

+   `boost::chrono::microseconds`（滴答周期 = `boost::ratio<1, 1000000>`）

+   `boost::chrono::nanoseconds`（滴答周期 = `boost::ratio<1, 1000000000>`）

### 持续时间算术

持续时间可以相加和相减，并且不同单位的持续时间可以组合成其他持续时间。较大单位的持续时间可以隐式转换为较小单位的持续时间。如果使用浮点表示，从较小单位到较大单位的隐式转换是可能的；对于整数表示，这样的转换会导致精度损失。为了处理这个问题，我们必须使用类似于强制转换运算符的函数，进行从较小单位到较大单位的显式转换：

**清单 8.8：使用 chrono 持续时间**

```cpp
 1 #include <boost/chrono/chrono.hpp>
 2 #include <boost/chrono/chrono_io.hpp>
 3 #include <iostream>
 4 #include <cstdint>
 5 namespace chrono = boost::chrono;
 6
 7 int main()
 8 {
 9   chrono::duration<int64_t, boost::ratio<1, 100>> csec(10);
10   std::cout << csec.count() << '\n';
11   std::cout << csec << '\n';
12
13   chrono::seconds sec(10);
14   chrono::milliseconds sum = sec + chrono::milliseconds(20);
15   // chrono::seconds sum1 = sec + chrono::milliseconds(20);
16
17   chrono::milliseconds msec = sec;
18
19   // chrono::seconds sec2 = sum;
20   chrono::seconds sec2 = 
21                  chrono::duration_cast<chrono::seconds>(sum);
22 }
```

这个例子说明了您可以执行的不同操作与持续时间。`boost/chrono/chrono.hpp`头文件包括了我们需要的大部分 Boost Chrono 设施（第 1 行）。我们首先创建一个 10 厘秒的`duration`（第 9 行）。`count`成员函数返回持续时间的滴答计数，即持续时间中所选单位的时间单位数，厘秒（第 10 行）。我们可以直接将持续时间流式传输到输出流（第 11 行），但需要包含额外的头文件`boost/chrono/chrono_io.hpp`来访问这些操作符（第 2 行）。流式传输`csec`打印如下：

```cpp
10 centiseconds
```

Boost Ratio 根据持续时间使用的时间单位提供适当的 SI 单位前缀，并用于智能打印适当的 SI 前缀。这在 C++11 标准库 Chrono 实现中不可用。

我们使用适当的持续时间特化创建秒和毫秒持续时间，并使用重载的`operator+`计算它们的和（第 13、14 行）。秒和毫秒持续时间的和是毫秒持续时间。毫秒持续时间隐式转换为秒单位的持续时间会导致精度损失，因为较大类型的表示是整数类型。因此，不支持这种隐式转换（第 15 行）。例如，10 秒+20 毫秒将计算为 10020 毫秒。`boost:::chrono::seconds` `typedef`使用带符号整数类型表示，要将 10020 毫秒表示为秒，20 毫秒需要被隐式四舍五入。

我们使用`duration_cast`函数模板，类似于 C++转换运算符，执行这种转换（第 20-21 行），使意图明确。`duration_cast`将进行四舍五入。另一方面，秒单位的持续时间总是可以隐式转换为毫秒单位的持续时间，因为没有精度损失（第 17 行）。

### 注意

`Chrono`库是一个单独构建的库，也依赖于 Boost System 库。因此，我们必须将本节中的示例链接到`libboost_system`。在 Unix 上使用 g++，您可以使用以下命令行来编译和链接涉及 Boost Chrono 的示例：

```cpp
$ g++ example.cpp -o example -lboost_system -lboost_chrono
```

对于非标准位置安装的 Boost 库，请参阅第一章*介绍 Boost*。

如果我们将持续时间专门化为使用`double`表示秒，而不是带符号整数，那么情况将会有所不同。以下代码将编译，因为`double`表示将能够容纳小数部分：

```cpp
boost::chrono::milliseconds millies(20);
boost::chrono::duration<double> sec(10);

boost::chrono::duration<double> sec2 = sec + millies;
std::cout << sec2 << '\n';
```

### 注意

我们在本书中没有详细介绍 Boost Ratio，但本章介绍了处理 Boost Chrono 所需的足够细节。此外，您可以访问比率的部分，并将比率打印为有理数或 SI 前缀，如果有意义的话。以下代码说明了这一点：

```cpp
#include <boost/ratio.hpp>
typedef boost::ratio<1000> kilo;
typedef boost::ratio<1, 1000> milli;
typedef boost::ratio<22, 7> not_quite_pi;
std::cout << not_quite_pi::num << "/" 
          << not_quite_pi::den << '\n';
std::cout << boost::ratio_string<kilo, char>::prefix() 
          << '\n';
std::cout << boost::ratio_string<milli, char>::prefix() 
          << '\n';
```

注意我们如何使用`ratio_string`模板及其前缀成员函数来打印 SI 前缀。代码打印如下：

```cpp
22/7
kilo
milli
```

C++11 标准库中的`std::ratio`模板对应于 Boost Ratio，并被`std::chrono`使用。标准库中没有`ratio_string`，因此缺少 SI 前缀打印。

## 时钟和时间点

时间点是时间的固定点，而不是持续时间。给定一个时间点，我们可以从中添加或减去一个持续时间，以得到另一个时间点。时代是某个时间系统中的参考时间点，可以与持续时间结合，以定义其他时间点。最著名的时代是 Unix 或 POSIX 时代，即 1970 年 1 月 1 日 00:00:00 UTC。

Boost Chrono 提供了几种时钟，用于在不同的上下文中测量时间。时钟具有以下关联成员：

+   一个名为`duration`的 typedef，表示使用该时钟可以表示的最小持续时间

+   一个名为`time_point`的 typedef，用于表示该时钟的时间点

+   一个静态成员函数`now`，返回当前时间点

Boost Chrono 定义了几种时钟，其中一些可能在您的系统上可用，也可能不可用：

+   `system_clock`类型表示壁钟或系统时间。

+   `steady_clock`类型表示一个单调时间系统，这意味着如果连续调用`now`函数，第二次调用将始终返回比第一次调用返回的时间点晚的时间点。这对于`system_clock`不能保证。只有在定义了`BOOST_CHRONO_HAS_STEADY_CLOCK`预处理宏时，才可用`steady_clock`类型。

+   如果可用，`high_resolution_clock`类型被定义为`steady_clock`，否则被定义为`system_clock`。

前面的时钟也可以作为`std::chrono`的一部分使用。它们使用一个实现定义的时代，并提供了在`time_point`和 Unix 时间(`std::time_t`)之间转换的函数。以下示例说明了时钟和时间点的使用方式：

**清单 8.9：使用 chrono system_clock**

```cpp
 1 #include <iostream>
 2 #include <boost/chrono.hpp>
 3
 4 namespace chrono = boost::chrono;
 5
 6 int main()
 7 {
 8   typedef chrono::system_clock::period tick_period;
 9   std::cout
10      << boost::ratio_string<tick_period, char>::prefix() 
11      << " seconds\n";
12   chrono::system_clock::time_point epoch;
13   chrono::system_clock::time_point now = 
14                             chrono::system_clock::now();
15
16   std::cout << epoch << '\n';
17   std::cout << chrono::time_point_cast<chrono::hours>(now) 
18             << '\n';
19 }
```

在这个例子中，我们首先打印与`system_clock`关联的持续时间的滴答周期。`system_clock::period`是`system_clock::duration::period`的一个 typedef，表示与`system_clock`关联的持续时间的滴答周期（第 8 行）。我们将其传递给`boost::ratio_string`，并使用`prefix`成员函数打印正确的 SI 前缀（第 9-10 行）。

它构造了两个时间点：一个用于`system_clock`的默认构造时间点，表示时钟的时代（第 12 行），以及由`system_clock`提供的`now`函数返回的当前时间（第 13-14 行）。然后我们打印时代（第 16 行），然后是当前时间（第 17 行）。时间点被打印为自时代以来的时间单位数。请注意，我们使用`time_point_cast`函数将当前时间转换为自时代以来的小时数。前面的代码在我的系统上打印如下：

```cpp
nanoseconds
0 nanoseconds since Jan 1, 1970
395219 hours since Jan 1, 1970
```

Boost Chrono 还提供了以下时钟，这些时钟都不作为 C++标准库 Chrono 的一部分：

+   `process_real_cpu_clock`类型用于测量程序启动以来的总时间。

+   `process_user_cpu_clock`类型用于测量程序在用户空间运行的时间。

+   `process_system_cpu`类型用于测量内核代表程序运行某些代码的时间。

+   `thread_clock`类型用于测量特定线程调度的总时间。只有在定义了`BOOST_CHRONO_HAS_THREAD_CLOCK`预处理宏时才可用此时钟。

只有在定义了`BOOST_CHRONO_HAS_PROCESS_CLOCKS`预处理宏时，才可用处理时钟。这些时钟可以类似于系统时钟使用，但它们的时代是 CPU 时钟的程序启动时，或者线程时钟的线程启动时。

# 使用 Boost Timer 测量程序性能

作为程序员，我们经常需要测量代码段的性能。虽然有几种出色的性能分析工具可用于此目的，但有时，能够对我们自己的代码进行仪器化既简单又更精确。Boost Timer 库提供了一个易于使用的、可移植的接口，用于通过仪器化代码来测量执行时间并报告它们。它是一个单独编译的库，不是仅头文件，并且在内部使用 Boost Chrono。

## cpu_timer

`boost::timer::cpu_timer`类用于测量代码段的执行时间。在下面的示例中，我们编写一个函数，该函数读取文件的内容并将其包装在`unique_ptr`中返回（参见第三章*内存管理和异常安全*）。它还使用`cpu_timer`计算并打印读取文件所用的时间。

**清单 8.10：使用 cpu_timer**

```cpp
 1 #include <fstream>
 2 #include <memory>
 3 #include <boost/timer/timer.hpp>
 4 #include <string>
 5 #include <boost/filesystem.hpp>
 6 using std::ios;
 7
 8 std::unique_ptr<char[]> readFile(const std::string& file_name,
 9                                  std::streampos& size)
10 {
11   std::unique_ptr<char[]> buffer;
12   std::ifstream file(file_name, ios::binary);
13
14   if (file) {
15     size = boost::filesystem::file_size(file_name);
16
17     if (size > 0) {
18       buffer.reset(new char[size]);
19
20       boost::timer::cpu_timer timer;
21       file.read(buffer.get(), size);
22       timer.stop();
23
24       std::cerr << "file size = " << size
25                 << ": time = " << timer.format();
26     }
27   }
28
29   return buffer;
30 }
```

我们在代码段的开始处创建一个`cpu_timer`实例（第 20 行），它启动计时器。在代码段结束时，我们在`cpu_timer`对象上调用`stop`成员函数（第 22 行），它停止计时器。我们调用`format`成员函数以获得可读的经过时间表示，并将其打印到标准错误（第 25 行）。使用文件名调用此函数，将以下内容打印到标准输入：

```cpp
file size = 1697199:  0.111945s wall, 0.000000s user + 0.060000s system = 0.060000s CPU (53.6%)
```

这表明对`fstream`的`read`成员函数的调用（第 21 行）被阻塞了 0.111945 秒。这是挂钟时间，即计时器测量的总经过时间。CPU 在用户模式下花费了 0.000000 秒，在内核模式下花费了 0.060000 秒（即在系统调用中）。请注意，读取完全在内核模式下进行，这是预期的，因为它涉及调用系统调用（例如在 Unix 上的读取）来从磁盘中读取文件的内容。CPU 在执行此代码时花费的经过时间的百分比为 53.6。它是作为在用户模式和内核模式中花费的持续时间之和除以总经过时间计算的，即（0.0 + 0.06）/ 0.111945，约为 0.536。

### 注意

使用 Boost Timer 的代码必须链接`libboost_timer`和`libboost_system`。要在 POSIX 系统上使用 g++构建涉及 Boost Timer 的示例，使用以下命令行：

```cpp
$ g++ source.cpp -o executable -std=c++11 -lboost_system -lboost_timer
```

对于安装在非标准位置的 Boost 库，请参阅第一章*介绍 Boost*。

如果我们想要测量打开文件、从文件中读取并关闭文件所花费的累积时间，那么我们可以使用单个计时器来测量多个部分的执行时间，根据需要停止和恢复计时器。

以下代码片段说明了这一点：

```cpp
12   boost::timer::cpu_timer timer;
13   file.open(file_name, ios::in|ios::binary|ios::ate);
14
15   if (file) {
16     size = file.tellg();
17
18     if (size > 0) {
19       timer.stop();
20       buffer.reset(new char[size]);
21
22       timer.resume();
23       file.seekg(0, ios::beg);
24       file.read(buffer.get(), size);
25     }
26
27     file.close();
28   }
29
30   timer.stop();
31 
```

在停止的计时器上调用`resume`成员函数会重新启动计时器，并添加到任何先前的测量中。在前面的代码片段中，我们在分配堆内存之前停止计时器（第 19 行），然后立即恢复计时器（第 22 行）。

还有一个`start`成员函数，它在`cpu_timer`构造函数内部调用以开始测量。在停止的计时器上调用`start`而不是`resume`会清除任何先前的测量，并有效地重置计时器。您还可以使用`is_stopped`成员函数检查计时器是否已停止，如果计时器已停止，则返回`true`，否则返回`false`。

我们可以通过调用`cpu_timer`的`elapsed`成员函数获取经过的时间（挂钟时间）、在用户模式下花费的 CPU 时间和在内核模式下花费的 CPU 时间（以纳秒为单位）：

```cpp
20       file.seekg(0, ios::beg);
21       boost::timer::cpu_timer timer;
22       file.read(buffer.get(), size);
23       timer.stop();
24
25       boost::timer::cpu_times times = timer.elapsed();
26       std::cout << std::fixed << std::setprecision(8)
27                 << times.wall / 1.0e9 << "s wall, "
28                 << times.user / 1.0e9 << "s user + "
29                 << times.system / 1.0e9 << "s system. "
30                 << (double)100*(timer.user + timer.system) 
31                       / timer.wall << "% CPU\n";
```

`elapsed`成员函数返回一个`cpu_times`类型的对象（第 25 行），其中包含三个字段，分别称为`wall`、`user`和`system`，它们以纳秒（10^(-9)秒）为单位包含适当的持续时间。

## 自动 CPU 计时器

`boost::timer::auto_cpu_timer`是`cpu_timer`的子类，它会在其封闭作用域结束时自动停止计数器，并将测量的执行时间写入标准输出或用户提供的另一个输出流。您无法停止和恢复它。当您需要测量代码段的执行时间直到作用域结束时，您可以使用`auto_cpu_timer`，只需使用一行代码，如下面从列表 8.10 调整的片段所示：

```cpp
17     if (size > 0) {
18       buffer.reset(new char[size]);
19
20       file.seekg(0, ios::beg);
21
22       boost::timer::auto_cpu_timer timer;
23       file.read(buffer.get(), size);
24     }
```

这将以熟悉的格式将测量的执行时间打印到标准输出：

```cpp
0.102563s wall, 0.000000s user + 0.040000s system = 0.040000s CPU (39.0%)
```

要将其打印到不同的输出流，我们需要将流作为构造函数参数传递给`timer`。

要测量读取文件所需的时间，我们只需在调用`read`之前声明`auto_cpu_timer`实例（第 22 行）。如果调用 read 不是作用域中的最后一条语句，并且我们不想测量后续内容的执行时间，那么这将不起作用。然后，我们可以使用`cpu_timer`而不是`auto_cpu_timer`，或者只将我们感兴趣的语句放在一个嵌套作用域中，并在开始时创建一个`auto_cpu_timer`实例：

```cpp
17     if (size > 0) {
18       buffer.reset(new char[size]);
19
20       file.seekg(0, ios::beg);
21
22       {
23         boost::timer::auto_cpu_timer timer(std::cerr);
24         file.read(buffer.get(), size);
25       }
26       // remaining statements in scope
27     }
```

在上面的例子中，我们创建了一个新的作用域（第 22-25 行），使用`auto_cpu_timer`来隔离要测量的代码部分。

# 自测问题

对于多项选择题，选择所有适用的选项：

1.  以下代码行哪个/哪些是不正确的？假设符号来自`boost::chrono`命名空间。

a. `milliseconds ms = milliseconds(5) + microseconds(10);`

b. `nanoseconds ns = milliseconds(5) + microseconds(10);`

c. `microseconds us = milliseconds(5) + microseconds(10);`

d. `seconds s = minutes(5) + microseconds(10);`

1.  `boost::chrono::duration<std::intmax_t, boost::ratio<1, 1000000>>`代表什么类型？

a. 以整数表示的毫秒持续时间

b. 以整数表示的微秒持续时间

c. 以浮点表示的毫秒持续时间

d. 以整数表示的纳秒持续时间

1.  `boost::timer::cpu_timer`和`boost::timer::auto_cpu_timer`之间有什么区别？

a. `auto_cpu_timer`在构造函数中调用`start`，`cpu_timer`不会

b. `auto_cpu_timer` 无法停止和恢复

c. `auto_cpu_timer`在作用域结束时写入输出流，`cpu_timer`不会

d. 你可以从`cpu_timer`中提取墙壁时间、用户时间和系统时间，但不能从`auto_cpu_timer`中提取

# 总结

本章介绍了用于测量时间和计算日期的库。本章让您快速了解了日期和时间计算的基础知识，而不涉及复杂的日历计算、时区意识和自定义和特定区域设置的格式。Boost 在线文档是这些细节的绝佳来源。

# 参考

+   《C++标准库：教程和参考指南（第二版）》，《Nicolai M. Josuttis》，《Addison Wesley Professional》

+   *睡眠的基础*：*Howard E. Hinnant*，*Walter E. Brown*，*Jeff Garland*和*Marc Paterno*（[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2661.htm`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2661.htm)）
