# 流和文件系统

# 问题

这是本章的问题解决部分。

# 32\. 帕斯卡三角形

编写一个函数，将帕斯卡三角形的最多 10 行打印到控制台。

# 33\. 列出进程列表

假设您有系统中所有进程列表的快照。每个进程的信息包括名称、标识符、状态（可以是*运行*或*挂起*）、帐户名称（进程运行的帐户）、以字节为单位的内存大小和平台（可以是 32 位或 64 位）。您的任务是编写一个函数，该函数接受这样一个进程列表，并以表格格式按字母顺序将它们打印到控制台。所有列必须左对齐，除了内存列必须右对齐。内存大小的值必须以 KB 显示。以下是此函数的输出示例：

```cpp
chrome.exe      1044   Running    marius.bancila    25180  32-bit
chrome.exe      10100  Running    marius.bancila   227756  32-bit
cmd.exe         512    Running    SYSTEM               48  64-bit
explorer.exe    7108   Running    marius.bancila    29529  64-bit
skype.exe       22456  Suspended  marius.bancila      656  64-bit
```

# 34\. 从文本文件中删除空行

编写一个程序，给定文本文件的路径，通过删除所有空行来修改文件。只包含空格的行被视为空行。

# 35\. 计算目录的大小

编写一个函数，递归计算目录的大小（以字节为单位）。应该可以指示是否应该跟随符号链接。

# 36\. 删除早于给定日期的文件

编写一个函数，给定目录的路径和持续时间，以递归方式删除所有早于指定持续时间的条目（文件或子目录）。持续时间可以表示任何内容，例如天、小时、分钟、秒等，或这些的组合，例如一小时二十分钟。如果指定的目录本身早于给定的持续时间，则应完全删除它。

# 37\. 查找与正则表达式匹配的目录中的文件

编写一个函数，给定目录的路径和正则表达式，返回所有目录条目的列表，其名称与正则表达式匹配。

# 38\. 临时日志文件

创建一个日志类，将文本消息写入可丢弃的文本文件。文本文件应具有唯一名称，并且必须位于临时目录中。除非另有说明，否则当类的实例被销毁时，应删除此日志文件。但是，可以通过将其移动到永久位置来保留日志文件。

# 解决方案

以下是上述问题解决部分的解决方案。

# 32\. 帕斯卡三角形

帕斯卡三角形是表示二项式系数的构造。三角形以一个具有单个值 1 的行开始。每行的元素是通过将上面、左边和右边的数字相加，并将空白条目视为 0 来构造的。以下是一个具有五行的三角形的示例：

```cpp
 1
 1   1
 1   2   1
 1   3   3   1
1   4   6   4   1
```

要打印三角形，我们必须：

+   将输出位置向右移动适当数量的空格，以便顶部投影在三角形底部的中间。

+   通过对上述左值和右值求和来计算每个值。一个更简单的公式是，对于第`i`行和第`j`列，每个新值`x`等于前一个值`x`乘以`(i - j) / (j + 1)`，其中`x`从 1 开始。

以下是一个可能的打印三角形的函数实现：

```cpp
unsigned int number_of_digits(unsigned int const i)
{
   return i > 0 ? (int)log10((double)i) + 1 : 1;
}

void print_pascal_triangle(int const n)
{
   for (int i = 0; i < n; i++) 
   {
      auto x = 1;
      std::cout << std::string((n - i - 1)*(n / 2), ' ');
      for (int j = 0; j <= i; j++) 
      {
         auto y = x;
         x = x * (i - j) / (j + 1);
         auto maxlen = number_of_digits(x) - 1;
         std::cout << y << std::string(n - 1 - maxlen - n%2, ' ');
      }
      std::cout << std::endl;
   }
}
```

以下程序要求用户输入级别的数量，并将三角形打印到控制台：

```cpp
int main()
{
   int n = 0;
   std::cout << "Levels (up to 10): ";
   std::cin >> n;
   if (n > 10)
      std::cout << "Value too large" << std::endl;
   else
      print_pascal_triangle(n);
}
```

# 33\. 列出进程列表

为了解决这个问题，我们将考虑以下表示有关进程信息的类：

```cpp
enum class procstatus {suspended, running};
enum class platforms {p32bit, p64bit};

struct procinfo
{
   int         id;
   std::string name;
   procstatus  status;
   std::string account;
   size_t      memory;
   platforms   platform;
};
```

为了将状态和平台以文本形式而不是数值形式打印出来，我们需要从枚举到`std::string`的转换函数：

```cpp
std::string status_to_string(procstatus const status)
{
   if (status == procstatus::suspended) return "suspended";
   else return "running";
}

std::string platform_to_string(platforms const platform)
{
   if (platform == platforms::p32bit) return "32-bit";
   else return "64-bit";
}
```

需要按进程名称按字母顺序排序进程。因此，第一步是对进程的输入范围进行排序。对于打印本身，我们应该使用 I/O 操纵符：

```cpp
void print_processes(std::vector<procinfo> processes)
{
   std::sort(
      std::begin(processes), std::end(processes),
      [](procinfo const & p1, procinfo const & p2) {
         return p1.name < p2.name; });

   for (auto const & pi : processes)
   {
      std::cout << std::left << std::setw(25) << std::setfill(' ')
                << pi.name;
      std::cout << std::left << std::setw(8) << std::setfill(' ')
                << pi.id;
      std::cout << std::left << std::setw(12) << std::setfill(' ')
                << status_to_string(pi.status);
      std::cout << std::left << std::setw(15) << std::setfill(' ')
                << pi.account;
      std::cout << std::right << std::setw(10) << std::setfill(' ')
                << (int)(pi.memory/1024);
      std::cout << std::left << ' ' << platform_to_string(pi.platform);
      std::cout << std::endl;
   }
}
```

以下程序定义了一个进程列表（实际上可以使用特定于操作系统的 API 检索运行中的进程列表），并以请求的格式打印到控制台：

```cpp
int main()
{
   using namespace std::string_literals;

   std::vector<procinfo> processes
   {
      {512, "cmd.exe"s, procstatus::running, "SYSTEM"s, 
            148293, platforms::p64bit },
      {1044, "chrome.exe"s, procstatus::running, "marius.bancila"s, 
            25180454, platforms::p32bit},
      {7108, "explorer.exe"s, procstatus::running, "marius.bancila"s,  
            2952943, platforms::p64bit },
      {10100, "chrome.exe"s, procstatus::running, "marius.bancila"s, 
            227756123, platforms::p32bit},
      {22456, "skype.exe"s, procstatus::suspended, "marius.bancila"s, 
            16870123, platforms::p64bit }, 
   };

   print_processes(processes);
}
```

# 34. 从文本文件中删除空行

解决此任务的一种可能方法是执行以下操作：

1.  创建一个临时文件，其中只包含要保留的原始文件的文本

1.  从输入文件逐行读取并将不为空的行复制到临时文件中

1.  在处理完原始文件后删除它

1.  将临时文件移动到原始文件的路径

另一种方法是移动临时文件并覆盖原始文件。以下实现遵循列出的步骤。临时文件是在`filesystem::temp_directory_path()`返回的临时目录中创建的：

```cpp
namespace fs = std::experimental::filesystem;

void remove_empty_lines(fs::path filepath)
{
   std::ifstream filein(filepath.native(), std::ios::in);
   if (!filein.is_open())
      throw std::runtime_error("cannot open input file");

   auto temppath = fs::temp_directory_path() / "temp.txt";
   std::ofstream fileout(temppath.native(), 
   std::ios::out | std::ios::trunc);
   if (!fileout.is_open())
      throw std::runtime_error("cannot create temporary file");

   std::string line;
   while (std::getline(filein, line))
   {
      if (line.length() > 0 &&
      line.find_first_not_of(' ') != line.npos)
      {
         fileout << line << '\n';
      }
   }
   filein.close();
   fileout.close();

   fs::remove(filepath);
   fs::rename(temppath, filepath);
}
```

# 35. 计算目录的大小

要计算目录的大小，我们必须遍历所有文件并计算各个文件的大小之和。

`filesystem::recursive_directory_iterator`是`filesystem`库中的一个迭代器，允许以递归方式遍历目录的所有条目。它有各种构造函数，其中一些采用`filesystem::directory_options`类型的值，指示是否应该跟随符号链接。通用的`std::accumulate()`算法可以用于将文件大小总和在一起。由于目录的总大小可能超过 2GB，因此不应使用`int`或`long`，而应使用`unsigned long long`作为总和类型。以下函数显示了所需任务的可能实现：

```cpp
namespace fs = std::experimental::filesystem;

std::uintmax_t get_directory_size(fs::path const & dir,
                                  bool const follow_symlinks = false)
{
   auto iterator = fs::recursive_directory_iterator(
      dir,
      follow_symlinks ? fs::directory_options::follow_directory_symlink : 
                        fs::directory_options::none);

   return std::accumulate(
      fs::begin(iterator), fs::end(iterator),
      0ull,
      [](std::uintmax_t const total,
         fs::directory_entry const & entry) {
             return total + (fs::is_regular_file(entry) ?
                    fs::file_size(entry.path()) : 0);
   });
}

int main()
{
   std::string path;
   std::cout << "Path: ";
   std::cin >> path;
   std::cout << "Size: " << get_directory_size(path) << std::endl;
}
```

# 36. 删除早于指定日期的文件

要执行文件系统操作，应该使用`filesystem`库。对于处理时间和持续时间，应该使用`chrono`库。实现请求功能的函数必须执行以下操作：

1.  检查目标路径指示的条目是否存在且是否比给定持续时间旧，如果是，则删除它

1.  如果不是旧的，并且它是一个目录，则遍历其所有条目并递归调用该函数：

```cpp
namespace fs = std::experimental::filesystem;
namespace ch = std::chrono;

template <typename Duration>
bool is_older_than(fs::path const & path, Duration const duration)
{
   auto ftimeduration = fs::last_write_time(path).time_since_epoch();
   auto nowduration = (ch::system_clock::now() - duration)
                      .time_since_epoch();
   return ch::duration_cast<Duration>(nowduration - ftimeduration)
                      .count() > 0;
}

template <typename Duration>
void remove_files_older_than(fs::path const & path, 
                             Duration const duration)
{
   try
   {
      if (fs::exists(path))
      {
         if (is_older_than(path, duration))
         {
            fs::remove(path);
         }
         else if(fs::is_directory(path))
         {
            for (auto const & entry : fs::directory_iterator(path))
            {
               remove_files_older_than(entry.path(), duration);
            }
         }
      }
   }
   catch (std::exception const & ex)
   {
      std::cerr << ex.what() << std::endl;
   }
}
```

除了使用`directory_iterator`和递归调用`remove_files_older_than()`之外，另一种方法是使用`recursive_directory_iterator`，并且如果超过给定持续时间，则简单地删除条目。然而，这种方法会使用未定义的行为，因为如果在创建递归目录迭代器后删除或添加文件或目录到目录树中，则不指定是否通过迭代器观察到更改。因此，应避免使用此方法。

`is_older_than()`函数模板确定了自系统时钟纪元以来当前时刻和最后一次文件写入操作之间经过的时间，并检查两者之间的差异是否大于指定的持续时间。

`remove_files_older_than()`函数可以如下使用：

```cpp
int main()
{
   using namespace std::chrono_literals;

#ifdef _WIN32
   auto path = R"(..\Test\)";
#else
   auto path = R"(../Test/)";
#endif

   remove_files_older_than(path, 1h + 20min);
}
```

# 37. 在目录中查找与正则表达式匹配的文件

实现指定的功能应该很简单：递归遍历指定目录的所有条目，并保留所有正则文件名匹配的条目。为此，您应该使用以下方法：

+   `filesystem::recursive_directory_iterator`用于遍历目录条目

+   `regex`和`regex_match()`来检查文件名是否与正则表达式匹配

+   `copy_if()`和`back_inserter`来复制符合特定条件的目录条目到`vector`的末尾。

这样的函数可能如下所示：

```cpp
namespace fs = std::experimental::filesystem;

std::vector<fs::directory_entry> find_files(
   fs::path const & path,
   std::string_view regex)
{
   std::vector<fs::directory_entry> result;
   std::regex rx(regex.data());

   std::copy_if(
      fs::recursive_directory_iterator(path),
      fs::recursive_directory_iterator(),
      std::back_inserter(result),
      &rx {
         return fs::is_regular_file(entry.path()) &&
                std::regex_match(entry.path().filename().string(), rx);
   });

   return result;
}
```

有了这个，我们可以编写以下代码：

```cpp
int main()
{
   auto dir = fs::temp_directory_path();
   auto pattern = R"(wct[0-9a-zA-Z]{3}\.tmp)";
   auto result = find_files(dir, pattern);

   for (auto const & entry : result)
   {
      std::cout << entry.path().string() << std::endl;
   }
}
```

# 38. 临时日志文件

您必须为此任务实现的日志类应该：

+   有一个构造函数，在临时目录中创建一个文本文件并打开它进行写入

+   在销毁期间，如果文件仍然存在，则关闭并删除它

+   有一个关闭文件并将其移动到永久路径的方法

+   重载`operator<<`以将文本消息写入输出文件

为了为文件创建唯一的名称，可以使用 UUID（也称为 GUID）。C++标准不支持与此相关的任何功能，但有第三方库，如`boost::uuid`、*CrossGuid*或`stduuid`，实际上是我创建的一个库。对于这个实现，我将使用最后一个。你可以在[`github.com/mariusbancila/stduuid`](https://github.com/mariusbancila/stduuid)找到它。

```cpp
namespace fs = std::experimental::filesystem;

class logger
{
   fs::path logpath;
   std::ofstream logfile;
public:
   logger()
   {
      auto name = uuids::to_string(uuids::uuid_random_generator{}());
      logpath = fs::temp_directory_path() / (name + ".tmp");
      logfile.open(logpath.c_str(), std::ios::out|std::ios::trunc);
   }

   ~logger() noexcept
   {
      try {
         if(logfile.is_open()) logfile.close();
         if (!logpath.empty()) fs::remove(logpath);
      }
      catch (...) {}
   }

   void persist(fs::path const & path)
   {
      logfile.close();
      fs::rename(logpath, path);
      logpath.clear();
   }

   logger& operator<<(std::string_view message)
   {
      logfile << message.data() << '\n';
      return *this;
   }
};
```

使用这个类的一个例子如下：

```cpp
int main()
{
   logger log;
   try 
   {
      log << "this is a line" << "and this is another one";
      throw std::runtime_error("error");
   }
   catch (...) 
   {
      log.persist(R"(lastlog.txt)");
   }
}
```
