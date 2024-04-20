# 第十六章：日期和时间

# 问题

本章的问题解决部分如下。

# 39.测量函数执行时间

编写一个函数，可以测量函数（带任意数量的参数）在任何所需持续时间（如秒、毫秒、微秒等）内的执行时间。

# 40.两个日期之间的天数

编写一个函数，给定两个日期，返回两个日期之间的天数。该函数应该能够在输入日期的顺序不同的情况下工作。

# 41.星期几

编写一个函数，给定一个日期，确定星期几。此函数应返回 1（星期一）到 7（星期日）之间的值。

# 42.年的日子和星期

编写一个函数，给定一个日期，返回一年中的日子（从 1 到 365 或闰年的 366），并且另一个函数，对于相同的输入，返回一年中的日历周。

# 43.多个时区的会议时间

编写一个函数，给定会议参与者和他们的时区列表，显示每个参与者的本地会议时间。

# 44.每月日历

编写一个函数，给定一年和一个月，将月历打印到控制台。预期的输出格式如下（示例是 2017 年 12 月）：

```cpp
Mon Tue Wed Thu Fri Sat Sun
                  1   2   3
  4   5   6   7   8   9  10
 11  12  13  14  15  16  17
 18  19  20  21  22  23  24
 25  26  27  28  29  30  31
```

# 解决方案

以下是上述问题解决部分的解决方案。

# 39.测量函数执行时间

要测量函数的执行时间，您应该在函数执行之前检索当前时间，执行函数，然后再次检索当前时间，并确定两个时间点之间经过了多少时间。为了方便起见，所有这些都可以放在一个“可变参数”函数模板中，该模板将函数执行及其参数作为参数，并且：

+   默认使用`std::high_resolution_clock`来确定当前时间。

+   使用`std::invoke()`来执行要测量的函数及其指定的参数。

+   返回一个持续时间而不是特定持续时间的滴答数。这很重要，这样您就不会丢失分辨率。它使您能够添加各种分辨率的执行时间持续时间，例如秒和毫秒，这是通过返回滴答数是不可能的：

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

这个函数模板可以按照以下方式使用：

```cpp
void f() 
{ 
   // simulate work
   std::this_thread::sleep_for(2s); 
}

void g(int const a, int const b) 
{ 
   // simulate work
   std::this_thread::sleep_for(1s); 
}

int main()
{
   auto t1 = perf_timer<std::chrono::microseconds>::duration(f);
   auto t2 = perf_timer<std::chrono::milliseconds>::duration(g, 1, 2);

   auto total = std::chrono::duration<double, std::nano>(t1 + t2).count();
}
```

# 40.两个日期之间的天数

截至 C++17，`chrono`标准库不支持处理日期、周、日历、时区和其他有用的相关功能。这将在 C++20 中改变，因为时区和日历支持已经在 2018 年 3 月的杰克逊维尔会议上被添加到了标准中。新的添加是基于一个名为`date`的开源库，它是在`chrono`之上构建的，由 Howard Hinnant 开发，并在 GitHub 上可用[`github.com/HowardHinnant/date`](https://github.com/HowardHinnant/date)。我们将使用这个库来解决本章的几个问题。尽管在这个实现中命名空间是`date`，但在 C++20 中它将成为`std::chrono`的一部分。但是，您应该能够简单地替换命名空间而不需要进行任何其他代码更改。

要解决这个任务，您可以使用`date.h`头文件中提供的`date::sys_days`类。它表示自`std::system_clock`纪元以来的天数。这是一个具有一天分辨率的`time_point`，可以隐式转换为`std::system_clock::time_point`。基本上，您必须构造两个这种类型的对象并对它们进行减法。结果恰好是两个日期之间的天数。以下是这样一个函数的简单实现：

```cpp
inline int number_of_days(
   int const y1, unsigned int const m1, unsigned int const d1,
   int const y2, unsigned int const m2, unsigned int const d2)
{
   using namespace date;

   return (sys_days{ year{ y1 } / month{ m1 } / day{ d1 } } -
           sys_days{ year{ y2 } / month{ m2 } / day{ d2 } }).count();
}

inline int number_of_days(date::sys_days const & first,
                          date::sys_days const & last)
{
   return (last - first).count();
}
```

以下是这些重载函数如何使用的一些示例：

```cpp
int main()
{
   auto diff1 = number_of_days(2016, 9, 23, 2017, 5, 15);

   using namespace date::literals;
   auto diff2 = number_of_days(2016_y/sep/23, 15_d/may/2017);
}
```

# 41.星期几

如果使用`date`库，解决这个问题也是相对简单的。但是，这一次，您必须使用以下类型：

+   `date::year_month_day`，一个表示具有年、月（1 到 12）和日（1 到 31）字段的日期的结构。

+   `date::iso_week::year_weeknum_weekday`，来自`iso_week.h`头文件，是一个结构，它有年份、一年中的周数和一周中的天数（1 到 7）的字段。这个类可以隐式转换为`date::sys_days`，这使得它可以显式转换为任何其他日历系统，只要它可以隐式转换为`date::sys_days`，比如`date::year_month_day`。

说到这里，问题就变成了创建一个`year_month_day`对象来表示所需的日期，然后从中创建一个`year_weeknum_weekday`对象，并用`weekday()`检索星期几：

```cpp
unsigned int week_day(int const y, unsigned int const m, 
                      unsigned int const d)
{
   using namespace date;

   if(m < 1 || m > 12 || d < 1 || d > 31) return 0;

   auto const dt = date::year_month_day{year{ y }, month{ m }, day{ d }};
   auto const tiso = iso_week::year_weeknum_weekday{ dt };

   return (unsigned int)tiso.weekday();
}

int main()
{
   auto wday = week_day(2018, 5, 9);
}
```

# 42\. 一年中的日和周

这个两部分问题的解决方案应该是基于前两部分的。

+   要计算一年中的天数，您需要减去两个`date::sys_days`对象，一个代表给定的日期，另一个代表同一年的 1 月 0 日。或者，您可以从 1 月 1 日开始，然后将结果加 1。

+   要确定一年中的周数，构造一个`year_weeknum_weekday`对象，就像在前面的问题中一样，并检索`weeknum()`的值：

```cpp
int day_of_year(int const y, unsigned int const m, 
                unsigned int const d)
{
   using namespace date;

   if(m < 1 || m > 12 || d < 1 || d > 31) return 0;

   return (sys_days{ year{ y } / month{ m } / day{ d } } -
           sys_days{ year{ y } / jan / 0 }).count();
}

unsigned int calendar_week(int const y, unsigned int const m, 
                           unsigned int const d)
{
   using namespace date;

   if(m < 1 || m > 12 || d < 1 || d > 31) return 0;

   auto const dt = date::year_month_day{year{ y }, month{ m }, day{ d }};
   auto const tiso = iso_week::year_weeknum_weekday{ dt };

   return (unsigned int)tiso.weeknum();
}
```

这些函数可以如下使用：

```cpp
int main()
{
   int y = 0;
   unsigned int m = 0, d = 0;
   std::cout << "Year:"; std::cin >> y;
   std::cout << "Month:"; std::cin >> m;
   std::cout << "Day:"; std::cin >> d;

   std::cout << "Calendar week:" << calendar_week(y, m, d) << std::endl;
   std::cout << "Day of year:" << day_of_year(y, m, d) << std::endl;
}
```

# 43\. 多个时区的会议时间

要使用时区，您必须使用`date`库的`tz.h`头文件。然而，这需要在您的机器上下载并解压*IANA 时区数据库*。

这是如何为日期库准备时区数据库的：

+   从[`www.iana.org/time-zones`](https://www.iana.org/time-zones)下载数据库的最新版本。目前，最新版本被称为`tzdata2017c.tar.gz`。

+   将其解压缩到机器上的任何位置，在一个名为`tzdata`的子目录中。假设父目录是`c:\work\challenges\libs\date`（在 Windows 机器上）；这将有一个名为`tzdata`的子目录。

+   对于 Windows，您需要下载一个名为`windowsZones.xml`的文件，其中包含 Windows 时区到 IANA 时区的映射。这可以在[`unicode.org/repos/cldr/trunk/common/supplemental/windowsZones.xml`](https://unicode.org/repos/cldr/trunk/common/supplemental/windowsZones.xml)找到。该文件必须存储在之前创建的`tzdata`子目录中。

+   在项目设置中，定义一个名为`INSTALL`的预处理器宏，指示`tzdata`子目录的父目录。对于这里给出的示例，您应该有`INSTALL=c:\\work\\challenges\\libs\\date`。（请注意，双反斜杠是必需的，因为该宏用于使用字符串化和连接创建文件路径，否则会导致不正确的路径。）

为了解决这个问题，我们将考虑一个具有最少信息的用户结构，比如姓名和时区。时区是使用`date::locate_zone()`函数创建的：

```cpp
struct user
{
   std::string Name;
   date::time_zone const * Zone;

   explicit user(std::string_view name, std::string_view zone)
      : Name{name.data()}, Zone(date::locate_zone(zone.data()))
   {}
};
```

一个显示用户列表和他们当地时间的函数应该将给定的时间从一个参考时区转换为他们自己时区的时间。为了做到这一点，我们可以使用`date::zoned_time`类的转换构造函数：

```cpp
template <class Duration, class TimeZonePtr>
void print_meeting_times(
   date::zoned_time<Duration, TimeZonePtr> const & time,
   std::vector<user> const & users)
{
   std::cout 
      << std::left << std::setw(15) << std::setfill(' ')
      << "Local time: " 
      << time << std::endl;

   for (auto const & user : users)
   {
      std::cout
         << std::left << std::setw(15) << std::setfill(' ')
         << user.Name
         << date::zoned_time<Duration, TimeZonePtr>(user.Zone, time) 
         << std::endl;
   }
}
```

这个函数可以如下使用，给定的时间（小时和分钟）在当前时区中表示：

```cpp
int main()
{
   std::vector<user> users{
      user{ "Ildiko", "Europe/Budapest" },
      user{ "Jens", "Europe/Berlin" },
      user{ "Jane", "America/New_York" }
   };

   unsigned int h, m;
   std::cout << "Hour:"; std::cin >> h;
   std::cout << "Minutes:"; std::cin >> m;

   date::year_month_day today = 
      date::floor<date::days>(ch::system_clock::now());

   auto localtime = date::zoned_time<std::chrono::minutes>(
      date::current_zone(), 
      static_cast<date::local_days>(today)+ch::hours{h}+ch::minutes{m});

   print_meeting_times(localtime, users);
}
```

# 44\. 月度日历

解决这个任务实际上部分地基于前面的任务。为了按照问题中指示的方式打印月份的天数，您应该知道：

+   月初的第一天是星期几。这可以使用为前一个问题创建的`week_day()`函数来确定。

+   月份中的天数。这可以使用`date::year_month_day_last`结构来确定，并检索`day()`的值。

有了这些信息确定后，您应该：

+   在第一个工作日之前的第一周打印空值

+   以从 1 到月底的适当格式打印日期

+   在每七天后换行（从第一周的第一天开始计算，即使它可能属于上个月）

所有这些的实现如下所示：

```cpp
unsigned int week_day(int const y, unsigned int const m, 
                      unsigned int const d)
{
   using namespace date;

   if(m < 1 || m > 12 || d < 1 || d > 31) return 0;

   auto const dt = date::year_month_day{year{ y }, month{ m }, day{ d }};
   auto const tiso = iso_week::year_weeknum_weekday{ dt };

   return (unsigned int)tiso.weekday();
}

void print_month_calendar(int const y, unsigned int m)
{
   using namespace date;
   std::cout << "Mon Tue Wed Thu Fri Sat Sun" << std::endl;

   auto first_day_weekday = week_day(y, m, 1);
   auto last_day = (unsigned int)year_month_day_last(
      year{ y }, month_day_last{ month{ m } }).day();

   unsigned int index = 1;
   for (unsigned int day = 1; day < first_day_weekday; ++day, ++index)
   {
      std::cout << " ";
   }

   for (unsigned int day = 1; day <= last_day; ++day)
   {
      std::cout << std::right << std::setfill(' ') << std::setw(3)
                << day << ' ';
      if (index++ % 7 == 0) std::cout << std::endl;
   }

   std::cout << std::endl;
}

int main()
{
   print_month_calendar(2017, 12);
}
```
