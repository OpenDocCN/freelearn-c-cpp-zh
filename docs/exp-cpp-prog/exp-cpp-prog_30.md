# 文件系统

在本章中，我们将涵盖以下内容：

+   实施路径标准化

+   从相对路径获取规范文件路径

+   列出目录中的所有文件

+   实施类似 grep 的文本搜索工具

+   实施自动文件重命名工具

+   实施磁盘使用量计数器

+   计算文件类型的统计信息

+   实施通过用符号链接替换重复项来减小文件夹大小的工具

# 介绍

如果没有一个帮助我们的库，处理文件系统路径总是很繁琐，因为有许多条件需要我们处理。

有些路径是*绝对*的，有些是*相对*的，也许它们甚至不是直接的，因为它们还包含`.`（当前目录）和`..`（父目录）的间接。同时，不同的操作系统使用斜杠`/`来分隔目录（Linux、MacOS 和不同的 UNIX 衍生版本），或者反斜杠（Windows）。当然还有不同类型的文件。

由于处理与文件系统相关的其他程序都需要这样的功能，因此在 C++17 STL 中拥有新的文件系统库是非常好的。最好的一点是，它对不同的操作系统都是以相同的方式工作，因此我们不必为支持不同操作系统的程序版本编写不同的代码。

在本章中，我们将首先看到`path`类的工作原理，因为它对于这个库中的任何其他内容都是最核心的。然后，我们将看到`directory_iterator`和`recursive_directory_iterator`类是多么强大但又简单易用，同时我们会对文件进行一些有用的操作。最后，我们将使用一些小而简单的示例工具，执行一些与文件系统相关的真实任务。从这一点开始，构建更复杂的工具将变得容易。

# 实施路径标准化

我们将本章以围绕`std::filesystem::path`类和一个智能规范化文件系统路径的辅助函数的非常简单的示例开始。

这个示例的结果是一个小应用程序，它接受任何文件系统路径，并以规范化形式返回相同的路径。规范化意味着我们得到一个不包含`.`或`..`路径间接的绝对路径。

在实施这一点的同时，我们还将看到在处理文件系统库的这个基本部分时需要注意哪些细节。

# 如何做...

在本节中，我们将实现一个程序，它只接受文件系统路径作为命令行参数，然后以规范化形式打印出来。

1.  首先是包含，然后我们声明使用`std`和`filesystem`命名空间。

```cpp
      #include <iostream>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  在主函数中，我们检查用户是否提供了命令行参数。如果没有，我们就会报错并打印如何使用程序。如果提供了路径，我们就会从中实例化一个`filesystem::path`对象。

```cpp
      int main(int argc, char *argv[])
      {
          if (argc != 2) {
              cout << "Usage: " << argv[0] << " <path>n";
              return 1;
          }

          const path dir {argv[1]};
```

1.  由于我们可以从任何字符串实例化`path`对象，我们不能确定路径是否真的存在于计算机的文件系统中。为了做到这一点，我们可以使用`filesystem::exists`函数。如果不存在，我们就会再次报错。

```cpp
          if (!exists(dir)) {
              cout << "Path " << dir << " does not exist.n";
              return 1;
          }
```

1.  好的，在这一点上，我们非常确定用户提供了一条*现有*路径，知道我们可以要求其规范化版本，然后我们打印出来。`filesystem::canonical`会返回另一个`path`对象。我们可以直接打印它，但`path`类型重载的`<<`运算符会用引号括起路径。为了避免这种情况，我们可以通过其`.c_str()`或`.string()`方法打印路径。

```cpp
          cout << canonical(dir).c_str() << 'n';
      }
```

1.  让我们编译程序并与之交互。当我们在我的家目录中执行它，使用相对路径`"src"`，它将打印出完整的绝对路径。

```cpp
      $ ./normalizer src
      /Users/tfc/src
```

1.  当我们再次在我的家目录中运行程序，但给它一个古怪的相对路径描述，首先进入我的`Desktop`文件夹，然后再次使用`..`退出它，然后进入`Documents`文件夹并再次退出，最后进入`src`目录，程序打印出*相同*的路径！

```cpp
      $ ./normalizer Desktop/../Documents/../src
      /Users/tfc/src
```

# 它是如何工作的...

作为`std::filesystem`的入门，这个示例仍然相当简短和直接。我们从包含文件系统路径描述的字符串初始化了一个`path`对象。`std::filesystem::path`类在我们使用文件系统库时扮演着非常重要的角色，因为大多数函数和类都与它相关。

使用`filesystem::exists`函数，我们能够检查路径是否真的存在。在那之前，我们不能确定，因为确实可能创建与现有文件系统对象无关的`path`对象。`exists`只接受一个`path`实例，并在它真的存在时返回`true`。该函数已经能够自行确定我们给它一个绝对路径还是相对路径，这使得它非常方便使用。

最后，我们使用`filesystem::canonical`在目录上，以便以规范化的形式打印它。

```cpp
path canonical(const path& p, const path& base = current_path());
```

`canonical`接受一个路径，并作为可选的第二个参数，它接受另一个路径。如果`p`是一个相对路径，第二个路径`base`将被添加到路径`p`之前。在这样做之后，`canonical`会尝试移除任何`.`和`..`路径指示。

在打印时，我们在规范化的路径上使用了`.c_str()`方法。这样做的原因是，对于输出流的`operator<<`的重载会用引号括起路径，而我们并不总是想要这样。

# 还有更多...

如果我们要规范化的路径不存在，`canonical`会抛出一个`filesystem_error`类型的异常。为了防止这种情况，我们用`exists`检查了我们的文件系统路径。但是那个检查真的足以避免出现未处理的异常吗？不是。

`exists`和`canonical`都可以抛出`bad_alloc`异常。如果遇到这些异常，有人可能会认为程序无论如何都要失败。一个更为严重，也更为可能的问题是，当我们检查文件是否存在并对其进行规范化之间，其他人重命名或删除了底层文件！在这种情况下，`canonical`会抛出一个`filesystem_error`，尽管我们之前检查了文件的存在。

大多数文件系统函数都有一个额外的重载，它接受相同的参数，但还有一个`std::error_code`引用。

```cpp
path canonical(const path& p, const path& base = current_path());
path canonical(const path& p, error_code& ec);
path canonical(const std::filesystem::path& p,
               const std::filesystem::path& base,
               std::error_code& ec );

```

这样我们可以选择是否用`try`-`catch`结构包围我们的文件系统函数调用，或者手动检查错误。请注意，这只会改变*与文件系统相关的*错误的行为！有了`ec`参数和没有`ec`参数，更基本的异常，例如`bad_alloc`，如果系统内存不足，仍然可能被抛出。

# 从相对路径获取规范化的文件路径

在上一个示例中，我们已经规范化/标准化了路径。`filesystem::path`类当然能够做更多的事情，而不仅仅是保存和检查路径。它还帮助我们轻松地从字符串中组合路径，并再次分解它们。

在这一点上，`path`已经将操作系统的细节抽象化了，但也有一些情况下我们仍然需要记住这些细节。

我们将通过玩弄绝对路径和相对路径来看如何处理路径及其组合/分解。

# 如何做...

在这一部分，我们将尝试使用绝对路径和相对路径，以便看到`path`类及其周围的辅助函数的优势。

1.  首先，我们包含了所有必要的头文件，并声明我们使用`std`和`sfilesystem`命名空间。

```cpp
      #include <iostream>
      #include <filesystem>     

      using namespace std;
      using namespace filesystem;
```

1.  然后，我们声明一个示例路径。在这一点上，它指的文本文件是否真的存在并不重要。然而，如果底层文件不存在，有一些函数会抛出异常。

```cpp
      int main()
      {
          path p {"testdir/foobar.txt"};
```

1.  现在我们将看看四个不同的文件系统库函数。`current_path`返回程序当前执行的路径，即*工作目录*。`absolute`接受一个相对路径，比如我们的路径`p`，并返回整个文件系统中的绝对、非歧义路径。`system_complete`在 Linux、MacOS 或类 UNIX 操作系统上实际上与`absolute`做的事情几乎一样。在 Windows 上，我们会得到绝对路径，另外还会加上磁盘卷标（例如`"C:"`）。`canonical`再次做的事情与`absolute`一样，但然后又移除了任何`"."`（代表*当前目录*）或`".."`（代表*上一级目录*）的间接。我们将在以下步骤中玩弄这样的间接：

```cpp
          cout << "current_path      : " << current_path()
               << "nabsolute_path   : " << absolute(p)
               << "nsystem_complete : " << system_complete(p)
               << "ncanonical(p)    : " << canonical(p)
               << 'n';
```

1.  `path`类的另一个好处是它重载了`/`运算符。这样我们就可以使用`/`连接文件夹名称和文件名，并从中组合路径。让我们试一试，并打印一个组合的路径。

```cpp
          cout << path{"testdir"} / "foobar.txt" << 'n';
```

1.  让我们来玩玩`canonical`和组合路径。通过给`canonical`一个相对路径，比如`"foobar.txt"`，和一个组合的绝对路径`current_path() / "testdir"`，它应该返回我们现有的绝对路径。在另一个调用中，我们给它我们的路径`p`（即`"testdir/foobar.txt"`），并提供一个绝对路径`current_path()`，这将引导我们进入`"testdir"`，然后再次返回。这应该与`current_path()`相同，因为有间接。在这两个调用中，`canonical`应该返回相同的绝对路径。

```cpp
          cout << "canonical testdir     : "
               << canonical("foobar.txt", 
                            current_path() / "testdir")
               << "ncanonical testdir 2 : "
               << canonical(p, current_path() / "testdir/..") 
               << 'n';
```

1.  我们还可以测试两个非规范路径的等价性。`equivalence`将接受的路径规范化，并在最终描述相同路径时返回`true`。对于这个测试，路径必须真的*存在*，否则会抛出异常。

```cpp
          cout << "equivalence: "
               << equivalent("testdir/foobar.txt",
                            "testdir/../testdir/foobar.txt") 
               << 'n';
      }
```

1.  编译和运行程序会产生以下输出。`current_path()`返回我笔记本电脑上的主文件夹，因为我是从那里执行应用程序的。我们的相对路径`p`已经被`absolute_path`、`system_complete`和`canonical`添加了这个目录。我们看到`absolute_path`和`system_complete`在我的系统上返回完全相同的路径，因为我用的是 Mac（在 Linux 上也是一样的）。在 Windows 机器上，`system_complete`会添加`"C:"`，或者工作目录所在的任何驱动器。

```cpp
      $ ./canonical_filepath
      current_path    : "/Users/tfc"
      absolute_path   : "/Users/tfc/testdir/foobar.txt"
      system_complete : "/Users/tfc/testdir/foobar.txt"
      canonical(p)    : "/Users/tfc/testdir/foobar.txt"
      "testdir/foobar.txt"
      canonical testdir   : "/Users/tfc/testdir/foobar.txt"
      canonical testdir 2 : "/Users/tfc/testdir/foobar.txt"
      equivalence: 1
```

1.  我们的简短程序中没有处理任何异常。如果我们删除`testdir`目录中的`foobar.txt`文件，那么程序会因为异常而中止执行。`canonical`函数要求路径存在。还有一个`weakly_canonical`函数，它不具备这个要求。

```cpp
      $ ./canonial_filepath 
      current_path    : "/Users/tfc"
      absolute_path   : "/Users/tfc/testdir/foobar.txt"
      system_complete : "/Users/tfc/testdir/foobar.txt"
 terminate called after throwing an instance of 
      'std::filesystem::v1::__cxx11::filesystem_error'
        what():  filesystem error: cannot canonicalize: 
        No such file or directory [testdir/foobar.txt] [/Users/tfc]
```

# 工作原理...

这个食谱的目标是看看动态组合新路径有多容易。这主要是因为`path`类对`/`运算符有一个方便的重载。除此之外，文件系统函数可以很好地处理相对路径、绝对路径，以及包含`.`和`..`间接的路径。

`path`实例的函数有很多，有些带有转换，有些没有。我们不会在这里列出所有的函数，因为简单地查看 C++参考文献是获得概述的最佳方式。

`path`类的成员函数可能值得更仔细地研究。让我们看看`path`的成员函数返回路径的哪一部分。下面的图表还显示了 Windows 路径与 UNIX/Linux 路径稍有不同。

![](img/9c9ab3d9-e0c4-41d0-b90c-2de3c0075dd2.png)

你可以看到图表显示了`path`的成员函数对*绝对*路径返回的内容。对于*相对*路径，`root_path`、`root_name`和`root_directory`是空的。然后，如果路径已经是相对的，`relative_path`就只返回路径。

# 列出目录中的所有文件

当然，每个提供文件系统支持的操作系统也都配备了某种在文件系统中仅*列出*目录中所有文件的实用程序。最简单的例子是 Linux、MacOS 和其他 UNIX 相关操作系统上的`ls`命令。在 DOS 和 Windows 中，有`dir`命令。两者都列出目录中的所有文件，并提供文件大小、权限等补充信息。

重新实现这样的工具也是一个很好的标准任务，可以开始进行目录和文件遍历。所以，让我们来做吧！

我们自己的`ls`/`dir`实用程序将能够按名称列出目录中的所有项目，指示有哪些项目，列出它们的访问权限标志，并显示它们在文件系统上占用的字节数。

# 如何做...

在本节中，我们将实现一个小工具，列出用户提供的任何目录中的所有文件。它不仅会列出文件名，还会列出它们的类型、大小和访问权限。

1.  首先，我们需要包含一些头文件，并声明我们默认使用`std`和`filesystem`命名空间。

```cpp
      #include <iostream>
      #include <sstream>
      #include <iomanip>
      #include <numeric>
      #include <algorithm>
      #include <vector>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  我们将需要的另一个辅助函数是`file_info`。它接受一个`directory_entry`对象引用，并从中提取路径，以及一个`file_status`对象（使用`status`函数），其中包含文件类型和权限信息。最后，如果是常规文件，它还提取条目的大小。对于目录或其他特殊文件，我们简单地返回大小为`0`。所有这些信息都被捆绑成一个元组。

```cpp
static tuple<path, file_status, size_t> 
      file_info(const directory_entry &entry)
      {
          const auto fs (status(entry));
          return {entry.path(),
                  fs,
                  is_regular_file(fs) ? file_size(entry.path()) : 0u};
      }
```

1.  我们需要的另一个辅助函数是`type_char`。路径不仅可以表示目录和简单的文本/二进制文件。操作系统提供了许多其他类型，用于抽象其他内容，例如硬件设备接口，以所谓的字符/块文件的形式。STL 文件系统库为它们提供了许多谓词函数。这样，我们可以为目录返回字母'd'，对于常规文件返回字母'f'，依此类推。

```cpp
      static char type_char(file_status fs)
      {
          if      (is_directory(fs))      { return 'd'; }
          else if (is_symlink(fs))        { return 'l'; }
          else if (is_character_file(fs)) { return 'c'; }
          else if (is_block_file(fs))     { return 'b'; }
          else if (is_fifo(fs))           { return 'p'; }
          else if (is_socket(fs))         { return 's'; }
          else if (is_other(fs))          { return 'o'; }
          else if (is_regular_file(fs))   { return 'f'; }

          return '?';
      }
```

1.  我们还需要的另一个辅助函数是`rwx`函数。它接受一个`perms`变量（它只是文件系统库中的一个`enum`类类型）并返回一个字符串，例如`"rwxrwxrwx"`，描述文件的权限设置。第一组`"rwx"`字符描述了文件所有者的***读、写和执行***权限。下一组描述了属于文件所属的*用户组*的所有用户的相同权限。最后一组字符描述了其他所有人对访问文件的权限。例如`"rwxrwxrwx"`表示每个人都可以以任何方式访问对象。`"rw-r--r--"`表示只有所有者可以读取和修改文件，而其他人只能读取。

我们只需从这些读/写/执行字符值中组合一个字符串，逐个权限位检查`perms`变量`p`是否包含特定的所有者位，然后返回'-'或正确的字符。

```cpp
      static string rwx(perms p)
      {
          auto check (p {
              return (p & bit) == perms::none ? '-' : c; 
          });

          return {check(perms::owner_read,   'r'),
                  check(perms::owner_write,  'w'),
                  check(perms::owner_exec,   'x'),
                  check(perms::group_read,   'r'),
                  check(perms::group_write,  'w'),
                  check(perms::group_exec,   'x'),
                  check(perms::others_read,  'r'),
                  check(perms::others_write, 'w'),
                  check(perms::others_exec,  'x')};
      }
```

1.  最后，最后一个辅助函数接受一个整数文件大小，并将其转换为更易读的形式。我们在除法时忽略小数点，并将其向下取整到最近的千、兆或吉边界。

```cpp
      static string size_string(size_t size)
      {
          stringstream ss;
          if        (size >= 1000000000) { 
              ss << (size / 1000000000) << 'G'; 
          } else if (size >= 1000000)    { 
              ss << (size / 1000000) << 'M';
          } else if (size >= 1000)       { 
              ss << (size / 1000) << 'K'; 
          } else { ss << size << 'B'; }

          return ss.str();
      }
```

1.  现在我们终于可以实现主函数了。我们首先检查用户是否在命令行中提供了路径。如果没有，我们就使用当前目录"。"。然后，我们检查目录是否存在。如果不存在，我们就无法列出任何文件。

```cpp
      int main(int argc, char *argv[])
      {
          path dir {argc > 1 ? argv[1] : "."};

          if (!exists(dir)) {
              cout << "Path " << dir << " does not exist.n";
              return 1;
          }
```

1.  现在，我们将用文件信息元组填充一个`vector`，就像我们的第一个辅助函数`file_info`从`directory_entry`对象返回的那样。我们实例化一个`directory_iterator`，并将其构造函数给予我们在上一步中创建的`path`对象。在使用目录迭代器进行迭代时，我们将`directory_entry`对象转换为文件信息元组，并将其插入向量中。

```cpp
          vector<tuple<path, file_status, size_t>> items;

          transform(directory_iterator{dir}, {},
              back_inserter(items), file_info);
```

1.  现在我们已经将所有信息保存在向量项中，可以使用我们编写的所有辅助函数简单地打印它。

```cpp
          for (const auto &[path, status, size] : items) {
              cout << type_char(status) 
                   << rwx(status.permissions()) << " "
                   << setw(4) << right << size_string(size) 
                   << " " << path.filename().c_str() 
                   << 'n';
          }
      }
```

1.  在离线版本的 C++文档中使用文件路径编译和运行项目会产生以下输出。我们看到该文件夹只包含目录和普通文件，因为所有输出行的第一个字符只有'd'和'f'。这些文件具有不同的访问权限，当然也有不同的大小。请注意，文件按其名称的字母顺序出现，但我们不能真正依赖它，因为字母顺序不是 C++17 标准要求的。

```cpp
      $ ./list ~/Documents/cpp_reference/en/cpp
      drwxrwxr-x    0B  algorithm
      frw-r--r--   88K  algorithm.html
      drwxrwxr-x    0B  atomic
      frw-r--r--   35K  atomic.html
      drwxrwxr-x    0B  chrono
      frw-r--r--   34K  chrono.html
      frw-r--r--   21K  comment.html
      frw-r--r--   21K  comments.html
      frw-r--r--  220K  compiler_support.html
      drwxrwxr-x    0B  concept
      frw-r--r--   67K  concept.html
      drwxr-xr-x    0B  container
      frw-r--r--  285K  container.html
      drwxrwxr-x    0B  error
      frw-r--r--   52K  error.html
```

# 它是如何工作的...

在这个示例中，我们遍历了文件，并对每个文件检查了其状态和大小。虽然我们的每个文件操作都相当简单直接，但我们的实际目录遍历看起来有点神奇。

为了遍历我们的目录，我们只需实例化一个`directory_iterator`，然后对其进行迭代。使用文件系统库遍历目录非常简单。

```cpp
for (const directory_entry &e : directory_iterator{dir}) {
    // do something
}
```

关于这个类，除了以下几点外，没有更多要说的：

+   它访问目录的每个元素一次

+   目录元素的迭代顺序是未指定的

+   目录元素`.`和`..`已经被过滤掉

然而，值得注意的是，`directory_iterator`似乎既是*迭代器*，又是*可迭代范围*。为什么？在我们刚刚看到的最小`for`循环示例中，它被用作可迭代范围。在实际的代码中，我们将它用作迭代器：

```cpp
transform(directory_iterator{dir}, {},
          back_inserter(items), file_info);
```

事实上，它只是一个迭代器类类型，但`std::begin`和`std::end`函数为这种类型提供了重载。这样我们就可以在这种迭代器上调用`begin`和`end`函数，它们会再次返回给我们迭代器。乍一看可能会觉得奇怪，但这样可以使这个类更有用。

# 实现类似 grep 的文本搜索工具

大多数操作系统都配备了某种本地搜索引擎。用户可以通过一些键盘快捷键启动它，然后输入他们要查找的本地文件。

在这些功能出现之前，命令行用户已经使用诸如`grep`或`awk`之类的工具搜索文件。用户可以简单地输入"`grep -r foobar .`"，该工具将在当前目录中递归搜索，并找到包含`"foobar"`字符串的任何文件。

在这个示例中，我们将实现一个这样的应用程序。我们的小型 grep 克隆将从命令行接受一个模式，然后递归地搜索我们在应用程序启动时所在的目录。然后，它将打印出每个与我们的模式匹配的文件的名称。模式匹配将逐行应用，因此我们还可以打印出文件匹配模式的确切行号。

# 如何做...

我们将实现一个小工具，用于在文件中搜索用户提供的文本模式。该工具类似于 UNIX 工具`grep`，但为了简单起见，它不会像`grep`那样成熟和强大。

1.  首先，我们需要包括所有必要的头文件，并声明我们使用`std`和`filesystem`命名空间。

```cpp
      #include <iostream>
      #include <fstream>
      #include <regex>
      #include <vector>
      #include <string>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  我们实现了一个辅助函数。它接受一个文件路径和一个描述我们正在寻找的模式的正则表达式对象。然后，我们实例化一个`vector`，其中包含匹配行号和它们的内容。我们还实例化了一个输入文件流对象，从中我们将逐行读取和匹配内容。

```cpp
      static vector<pair<size_t, string>> 
      matches(const path &p, const regex &re)
      {
          vector<pair<size_t, string>> d;
          ifstream is {p.c_str()};
```

1.  我们使用`getline`函数逐行遍历文件。如果`regex_search`返回`true`，则表示字符串包含我们的模式。如果是这种情况，我们将行号和字符串放入向量中。最后，我们返回所有收集到的匹配项。

```cpp
          string s;
          for (size_t line {1}; getline(is, s); ++line) {
              if (regex_search(begin(s), end(s), re)) {
                  d.emplace_back(line, move(s));
              }
          }

          return d;
      }
```

1.  在主函数中，我们首先检查用户是否提供了可以用作模式的命令行参数。如果没有，我们会报错。

```cpp
      int main(int argc, char *argv[])
      {
          if (argc != 2) {
              cout << "Usage: " << argv[0] << " <pattern>n";
              return 1;
          }
```

1.  接下来，我们从输入模式构造一个正则表达式对象。如果模式不是有效的正则表达式，这将导致异常。如果发生这样的异常，我们会捕获它并报错。

```cpp
          regex pattern;

          try { pattern = regex{argv[1]}; }
          catch (const regex_error &e) {
              cout << "Invalid regular expression provided.n";
              return 1;
          }
```

1.  现在，我们终于可以遍历文件系统并寻找模式匹配了。我们使用`recursive_directory_iterator`来遍历工作目录中的所有文件。它的工作方式与上一个教程中的`directory_iterator`完全相同，但它还会进入子目录。这样我们就不必管理递归。在每个条目上，我们调用我们的辅助函数`matches`。

```cpp
          for (const auto &entry :
                recursive_directory_iterator{current_path()}) {
              auto ms (matches(entry.path(), pattern));
```

1.  对于每个匹配（如果有的话），我们打印文件路径、行号和匹配行的完整内容。

```cpp
              for (const auto &[number, content] : ms) {
                  cout << entry.path().c_str() << ":" << number
                       << " - " << content << 'n';
              }
          }
      }
```

1.  让我们准备一个名为`"foobar.txt"`的文件，其中包含一些我们可以搜索的测试行。

```cpp
      foo
      bar
      baz
```

1.  编译和运行产生以下输出。我在我的笔记本电脑的`/Users/tfc/testdir`文件夹中启动了应用程序，首先使用模式`"bar"`。在该目录中，它找到了我们的`foobar.txt`文件的第二行和另一个文件`"text1.txt"`，它位于`testdir/dir1`中。

```cpp
      $ ./grepper bar
      /Users/tfc/testdir/dir1/text1.txt:1 - foo bar bla blubb
      /Users/tfc/testdir/foobar.txt:2 - bar

```

1.  再次启动应用程序，但这次使用模式`"baz"`，它找到了我们示例文本文件的第三行。

```cpp
      $ ./grepper baz
      /Users/tfc/testdir/foobar.txt:3 - baz
```

# 它的工作原理...

设置和使用正则表达式来过滤文件内容肯定是这个教程的主要任务。然而，让我们集中在`recursive_directory_iterator`上，因为递归地过滤迭代的文件只是我们在这个教程中使用这个特殊迭代器类的动机。

就像`directory_iterator`一样，`recursive_directory_iterator`遍历目录的元素。它的特点是递归地执行这个操作，正如它的名字所示。每当它遇到一个文件系统元素是*目录*时，它将向这个路径产生一个`directory_entry`实例，然后还会进入其中以遍历它的子元素。

`recursive_directory_iterator`有一些有趣的成员函数：

+   `depth（）`：这告诉我们迭代器当前已经进入子目录的级数。

+   `recursion_pending（）`：这告诉我们迭代器当前指向的元素之后是否会进入递归。

+   禁用递归挂起（）：如果当前指向的是一个目录，可以调用此方法来阻止迭代器进入下一个子目录，如果它当前指向的是一个目录，那么调用此方法将不起作用，因为我们调用它*太早*。

+   `pop（）`：中止当前递归级别，并在目录层次结构中向上移动一级以从那里继续。

# 还有更多...

还要了解的一件事是`directory_options`枚举类。`recursive_directory_iterator`的构造函数确实接受这种类型的值作为第二个参数。我们一直在隐式使用的默认值是`directory_options::none`。其他值包括：

+   `follow_directory_symlink`：这允许递归迭代器跟随符号链接到目录

+   `skip_permission_denied`：这告诉迭代器跳过否则会因为文件系统拒绝访问权限而导致错误的目录

这些选项可以与`|`运算符结合使用。

# 实现自动文件重命名器

这个教程的动机是我经常发现自己处于这样的情况。例如，从不同的朋友和不同的照片设备收集假期的图片文件放在一个文件夹中，文件扩展名经常看起来不同。一些 JPEG 文件有`.jpg`扩展名，一些有`.jpeg`，还有一些甚至有`.JPEG`。

有些人可能更喜欢使所有扩展名统一。使用单个命令重命名所有文件将是有用的。同时，我们可以删除空格`' '`并用下划线`'_'`替换它们，例如。

在这个教程中，我们将实现这样一个工具，并将其称为`renamer`。它将接受一系列输入模式及其替代品，如下所示：

```cpp
$ renamer jpeg jpg JPEG jpg
```

在这种情况下，重命名器将递归地遍历当前目录，并在所有文件名中搜索模式`jpeg`和`JPEG`。它将用`jpg`替换两者。

# 如何做...

我们将实现一个工具，递归扫描目录中的所有文件，并将它们的文件名与模式进行匹配。所有匹配项都将替换为用户提供的标记，并相应地重命名受影响的文件。

1.  首先，我们需要包括一些头文件，并声明我们使用命名空间`std`和`filesystem`。

```cpp
      #include <iostream>
      #include <regex>
      #include <vector>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  我们实现了一个简短的辅助函数，它接受一个字符串形式的输入文件路径和一系列替换对。每个替换对包括一个模式和其替换。在循环遍历替换范围时，我们使用`regex_replace`将其提供给输入字符串，并让其返回转换后的字符串。然后，我们返回结果字符串。

```cpp
      template <typename T>
      static string replace(string s, const T &replacements)
      {
          for (const auto &[pattern, repl] : replacements) {
              s = regex_replace(s, pattern, repl);
          }

          return s;
      }
```

1.  在主函数中，我们首先验证命令行。我们接受*成对*的命令行参数，因为我们希望模式与它们的替换一起。`argv`的第一个元素始终是可执行文件名。这意味着如果用户提供了至少一对或更多对，那么`argc`必须是*奇数*，且不小于`3`。

```cpp
      int main(int argc, char *argv[])
      {
          if (argc < 3 || argc % 2 != 1) {
              cout << "Usage: " << argv[0] 
                   << " <pattern> <replacement> ...n";
              return 1;
          }
```

1.  一旦我们检查到有输入对，我们将用这些对填充一个向量。

```cpp
          vector<pair<regex, string>> patterns;

          for (int i {1}; i < argc; i += 2) {
              patterns.emplace_back(argv[i], argv[i + 1]);
          }
```

1.  现在我们可以遍历文件系统。为了简单起见，我们只需将应用程序的当前路径定义为要遍历的目录。

对于每个目录条目，我们提取其原始路径到`opath`变量中。然后，我们只取文件名而不是其余路径，并根据之前收集的模式和替换列表进行转换。我们复制`opath`，称其为`rpath`，并用新文件名替换其文件名部分。

```cpp
          for (const auto &entry :
                recursive_directory_iterator{current_path()}) {
              path opath {entry.path()};
              string rname {replace(opath.filename().string(),
                                    patterns)};

              path rpath {opath};
              rpath.replace_filename(rname);
```

1.  对于受我们模式影响的所有文件，我们打印出我们重命名它们。如果替换模式后的文件名已经存在，我们无法继续。让我们跳过这样的文件。当然，我们当然也可以只向路径追加一些数字或其他内容来解决名称冲突。

```cpp
              if (opath != rpath) {
                  cout << opath.c_str() << " --> " 
                       << rpath.filename().c_str() << 'n';
                  if (exists(rpath)) {
                      cout << "Error: Can't rename."
                              " Destination file exists.n";
                  } else {
                      rename(opath, rpath);
                  }
              }
          }
      }
```

1.  在示例目录中编译和运行程序会产生以下输出。我把一些 JPEG 图片放入了目录中，但给它们不同的名称结尾`jpg`，`jpeg`和`JPEG`。然后，我用模式`jpeg`和`JPEG`执行了程序，并选择了`jpg`作为两者的替换。结果是一个具有同质文件扩展名的文件夹。

```cpp
      $ ls
      birthday_party.jpeg   holiday_in_dubai.jpg  holiday_in_spain.jpg 
      trip_to_new_york.JPEG
      $ ../renamer jpeg jpg JPEG jpg
      /Users/tfc/pictures/birthday_party.jpeg --> birthday_party.jpg
      /Users/tfc/pictures/trip_to_new_york.JPEG --> trip_to_new_york.jpg
      $ ls
      birthday_party.jpg   holiday_in_dubai.jpg holiday_in_spain.jpg
      trip_to_new_york.jpg
```

# 实现磁盘使用计数器

我们已经实现了一个类似于 Linux/MacOS 上的`ls`或 Windows 上的`dir`的工具，但与这些工具一样，它不会打印*目录*的文件大小。

为了获得目录的大小等价值，我们需要进入其中并计算其中包含的所有文件的大小。

在这个示例中，我们将实现一个工具来做到这一点。该工具可以在任何文件夹上运行，并汇总所有目录条目的累积大小。

# 如何做...

在这一部分，我们将实现一个应用程序，它遍历目录并列出每个条目的文件大小。对于常规文件来说很简单，但如果我们看到的目录条目本身是一个目录，那么我们必须查看它并总结其包含的所有文件的大小。

1.  首先，我们需要包括所有必要的头文件，并声明我们使用命名空间`std`和`filesystem`。

```cpp
      #include <iostream>
      #include <sstream>
      #include <iomanip>
      #include <numeric>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  然后我们实现一个辅助函数，它接受一个`directory_entry`作为参数，并返回其在文件系统中的大小。如果不是目录，我们只需返回由`file_size`计算的文件大小。

```cpp
      static size_t entry_size(const directory_entry &entry)
      {
          if (!is_directory(entry)) { return file_size(entry); }
```

1.  如果是目录，我们需要遍历其所有条目并计算它们的大小。如果我们再次遇到子目录，我们最终会递归调用我们自己的`entry_size`辅助函数。

```cpp
          return accumulate(directory_iterator{entry}, {}, 0u,
              [](size_t accum, const directory_entry &e) {
                  return accum + entry_size(e);
              });
      }
```

1.  为了更好地可读性，我们在本章的其他示例中使用相同的`size_string`函数。它只是将大文件大小分成更短、更美观的字符串，以便读取带有 kilo、mega 或 giga 后缀的字符串。

```cpp
      static string size_string(size_t size)
      {
          stringstream ss;
          if        (size >= 1000000000) { 
              ss << (size / 1000000000) << 'G'; 
          } else if (size >= 1000000)    { 
              ss << (size / 1000000) << 'M'; 
          } else if (size >= 1000)       { 
              ss << (size / 1000) << 'K'; 
          } else { ss << size << 'B'; }

          return ss.str();
      }  
```

1.  在主函数中，我们需要做的第一件事是检查用户是否在命令行上提供了文件系统路径。如果不是这种情况，我们就取当前文件夹。在继续之前，我们要检查它是否存在。

```cpp
      int main(int argc, char *argv[])
      {
          path dir {argc > 1 ? argv[1] : "."};

          if (!exists(dir)) {
              cout << "Path " << dir << " does not exist.n";
              return 1;
          }
```

1.  现在，我们可以遍历所有目录条目并打印它们的大小和名称。

```cpp
          for (const auto &entry : directory_iterator{dir}) {
              cout << setw(5) << right 
                   << size_string(entry_size(entry))
                   << " " << entry.path().filename().c_str() 
                   << 'n';
          }
      }
```

1.  编译和运行程序产生以下结果。我在 C++离线参考手册的一个文件夹中启动了它。由于它也包含子文件夹，我们的递归文件大小摘要助手立即就派上了用场。

```cpp
      $ ./file_size ~/Documents/cpp_reference/en/
        19M c
        12K c.html
       147M cpp
        17K cpp.html
        22K index.html
        22K Main_Page.html
```

# 它的工作原理...

整个程序围绕着在常规文件上使用`file_size`。如果程序看到一个目录，它会递归进入其中，并对所有条目调用`file_size`。

我们用来区分是否直接调用`file_size`还是需要递归策略的唯一方法是询问`is_directory`谓词。这对于只包含常规文件和目录的目录非常有效。

尽管我们的示例程序很简单，但在以下情况下会崩溃，因为没有处理异常：

+   `file_size`只对常规文件和符号链接有效。在其他情况下会抛出异常。

+   尽管`file_size`对符号链接有效，但如果我们在*损坏的*符号链接上调用它，它仍然会抛出异常。

为了使这个示例程序更加成熟，我们需要更多的防御性编程来处理错误类型的文件和异常处理。

# 计算文件类型的统计信息

在上一个示例中，我们实现了一个工具，列出任何目录中所有成员的大小。

在这个示例中，我们也将递归计算大小，但这次我们将每个文件的大小累积到它们的文件名*扩展名*中。这样我们就可以向用户打印一个表，列出我们有多少个每种文件类型的文件，以及这些文件类型的平均大小。

# 如何做...

在本节中，我们将实现一个小工具，它会递归地遍历给定目录。在这样做的过程中，它会计算所有文件的数量和大小，按其扩展名分组。最后，它会打印出该目录中存在的文件名扩展名，每个扩展名的数量以及它们的平均文件大小。

1.  我们需要包括必要的头文件，并声明我们使用`std`和`filesystem`命名空间。

```cpp
      #include <iostream>
      #include <sstream>
      #include <iomanip>
      #include <map>
      #include <filesystem>     

      using namespace std;
      using namespace filesystem;
```

1.  `size_string`函数在其他示例中已经很有用了。它将文件大小转换为人类可读的字符串。

```cpp
      static string size_string(size_t size)
      {
          stringstream ss;
          if        (size >= 1000000000) { 
              ss << (size / 1000000000) << 'G'; 
          } else if (size >= 1000000)    { 
              ss << (size / 1000000) << 'M'; 
          } else if (size >= 1000)       { 
              ss << (size / 1000) << 'K';
          } else { ss << size << 'B'; }

          return ss.str();
      }
```

1.  然后，我们实现一个辅助函数，它接受一个`path`对象作为参数，并遍历该路径下的所有文件。在此过程中，它将所有信息收集到一个映射中，该映射将文件名扩展名映射到包含具有相同扩展名的所有文件的总数和累积大小的对中。

```cpp
      static map<string, pair<size_t, size_t>> ext_stats(const path &dir)
      {
          map<string, pair<size_t, size_t>> m;

          for (const auto &entry :
                recursive_directory_iterator{dir}) {
```

1.  如果目录条目本身是一个目录，我们就跳过它。此时跳过它并不意味着我们不会递归进入其中。`recursive_directory_iterator`仍然会这样做，但我们不想查看目录条目本身。

```cpp
              const path        p  {entry.path()};
              const file_status fs {status(p)};

              if (is_directory(fs)) { continue; }
```

1.  接下来，我们提取目录条目字符串的扩展部分。如果没有扩展名，我们就简单地跳过它。

```cpp
              const string ext {p.extension().string()};

              if (ext.length() == 0) { continue; }
```

1.  接下来，我们计算我们正在查看的文件的大小。然后，我们在地图中查找这个扩展名的聚合对象。如果此时还没有，它会被隐式创建。我们只是增加文件计数并将文件大小添加到大小累加器中。

```cpp
              const size_t size {file_size(p)};

              auto &[size_accum, count] = m[ext];

              size_accum += size;
              count      += 1;
          }
```

1.  之后，我们返回地图。

```cpp
          return m;
      }
```

1.  在主函数中，我们从命令行中获取用户提供的路径或当前目录。当然，我们需要检查它是否存在，否则继续下去就没有意义。

```cpp
      int main(int argc, char *argv[])
      {
          path dir {argc > 1 ? argv[1] : "."};

          if (!exists(dir)) {
              cout << "Path " << dir << " does not exist.n";
              return 1;
          }
```

1.  我们可以立即遍历`ext_stats`给我们的映射。因为映射中的`accum_size`项包含相同扩展名的所有文件的总和，所以在打印之前，我们将这个总和除以这些文件的总数。

```cpp
          for (const auto &[ext, stats] : ext_stats(dir)) {
              const auto &[accum_size, count] = stats;

              cout << setw(15) << left << ext << ": "
                   << setw(4) << right << count 
                   << " items, avg size "
                   << setw(4) << size_string(accum_size / count)
                   << 'n';
          }
      }
```

1.  编译和运行程序产生以下输出。我将离线 C++参考手册中的一个文件夹作为命令行参数。

```cpp
      $ ./file_type ~/Documents/cpp_reference/
      .css           :    2 items, avg size  41K
      .gif           :    7 items, avg size 902B
      .html          : 4355 items, avg size  38K
      .js            :    3 items, avg size   4K
      .php           :    1 items, avg size 739B
      .png           :   34 items, avg size   2K
      .svg           :   53 items, avg size   6K
      .ttf           :    2 items, avg size 421K
```

# 实现一个通过用符号链接替换重复项来减小文件夹大小的工具

有很多工具以各种方式压缩数据。文件打包算法/格式的最著名的例子是 ZIP 和 RAR。这些工具试图通过减少内部冗余来减小文件的大小。

在将文件压缩到存档文件之前，减少磁盘使用的一个非常简单的方法就是*删除* *重复*文件。在这个示例中，我们将实现一个小工具，它会递归地遍历一个目录。在遍历过程中，它将寻找具有相同内容的文件。如果找到这样的文件，它将删除所有重复项，只保留一个。所有删除的文件将被替换为指向现在唯一文件的符号链接。这样可以节省空间，而不需要任何压缩，同时保留所有数据。

# 如何做...

在这一部分，我们将实现一个小工具，找出目录中彼此重复的文件。有了这个知识，它将删除所有重复的文件，只保留一个，并用符号链接替换它们，从而减小文件夹的大小。

确保系统数据有一个*备份*。我们将使用 STL 函数删除文件。在这样一个程序中，一个简单的*拼写错误*路径可能导致程序以不希望的方式贪婪地删除太多文件。

1.  首先，我们需要包含必要的头文件，然后我们声明我们默认使用`std`和`filesystem`命名空间。

```cpp
      #include <iostream>
      #include <fstream>
      #include <unordered_map>
      #include <filesystem>      

      using namespace std;
      using namespace filesystem;
```

1.  为了找出哪些文件是彼此的重复项，我们将构建一个哈希映射，将文件内容的哈希映射到生成该哈希的第一个文件的路径。对于文件，使用生产哈希算法如 MD5 或 SHA 变体会是一个更好的主意。为了保持清晰和简单，我们只是将整个文件读入一个字符串，然后使用`unordered_map`已经用于字符串的相同哈希函数对象来计算哈希。

```cpp
      static size_t hash_from_path(const path &p)
      {
          ifstream is {p.c_str(), 
                       ios::in | ios::binary};
          if (!is) { throw errno; }

          string s;

          is.seekg(0, ios::end);
          s.reserve(is.tellg());
          is.seekg(0, ios::beg);

          s.assign(istreambuf_iterator<char>{is}, {});

          return hash<string>{}(s);
      }
```

1.  然后我们实现构建这样一个哈希映射并删除重复项的函数。它通过目录及其子目录进行递归迭代。

```cpp
      static size_t reduce_dupes(const path &dir)
      {
          unordered_map<size_t, path> m;
          size_t count {0};

          for (const auto &entry :
                recursive_directory_iterator{dir}) {
```

1.  对于每个目录条目，它都会检查它是否是一个目录本身。所有目录项都将被跳过。对于每个文件，我们生成其哈希值并尝试将其插入哈希映射中。如果哈希映射已经包含相同的哈希，则这意味着我们已经插入了具有相同哈希的文件。这意味着我们刚刚找到了一个重复项！在插入过程中发生冲突时，`try_emplace`返回的对中的第二个值为`false`。

```cpp
              const path p {entry.path()};

              if (is_directory(p)) { continue; }

              const auto &[it, success] =
                  m.try_emplace(hash_from_path(p), p);
```

1.  使用`try_emplace`的返回值，我们可以告诉用户我们刚刚插入了一个文件，因为我们第一次看到了它的哈希。如果我们找到了重复项，我们会告诉用户它是哪个其他文件的重复项，并将其删除。删除后，我们创建一个替换重复项的符号链接。

```cpp
              if (!success) {
                  cout << "Removed " << p.c_str()
                       << " because it is a duplicate of "
                       << it->second.c_str() << 'n';

                  remove(p);
                  create_symlink(absolute(it->second), p);
                  ++count;
              }
```

1.  在文件系统迭代之后，我们返回删除的文件数，并用符号链接替换。

```cpp
          }

          return count;
      }
```

1.  在主函数中，我们确保用户在命令行上提供了一个目录，并且该目录存在。

```cpp
      int main(int argc, char *argv[])
      {
          if (argc != 2) {
              cout << "Usage: " << argv[0] << " <path>n";
              return 1;
          }

          path dir {argv[1]};

          if (!exists(dir)) {
              cout << "Path " << dir << " does not exist.n";
              return 1;
          }
```

1.  现在我们唯一需要做的就是在这个目录上调用`reduce_dupes`，并打印它删除了多少文件。

```cpp
          const size_t dupes {reduce_dupes(dir)};

          cout << "Removed " << dupes << " duplicates.n";
      }
```

1.  在包含一些重复文件的示例目录上编译和运行程序如下。我使用`du`工具在启动我们的程序之前和之后检查文件夹大小，以演示这种方法的工作原理。

```cpp
      $ du -sh dupe_dir
      1.1M dupe_dir

      $ ./dupe_compress dupe_dir
      Removed dupe_dir/dir2/bar.jpg because it is a duplicate of 
      dupe_dir/dir1/bar.jpg
      Removed dupe_dir/dir2/base10.png because it is a duplicate of 
      dupe_dir/dir1/base10.png
      Removed dupe_dir/dir2/baz.jpeg because it is a duplicate of 
      dupe_dir/dir1/baz.jpeg
      Removed dupe_dir/dir2/feed_fish.jpg because it is a duplicate of 
      dupe_dir/dir1/feed_fish.jpg
      Removed dupe_dir/dir2/foo.jpg because it is a duplicate of 
      dupe_dir/dir1/foo.jpg
      Removed dupe_dir/dir2/fox.jpg because it is a duplicate of 
      dupe_dir/dir1/fox.jpg
      Removed 6 duplicates.

      $ du -sh dupe_dir
      584K dupe_dir
```

# 它是如何工作的...

我们使用`create_symlink`函数来使文件系统中的一个文件指向文件系统中的另一个文件。这样我们就可以避免重复的文件。我们也可以使用`create_hard_link`来设置硬链接。从语义上讲，这是相似的，但是硬链接有其他技术含义。不同的文件系统格式可能根本不支持硬链接，或者只支持指向同一文件的一定数量的硬链接。另一个问题是硬链接不能从一个文件系统链接到另一个文件系统。

然而，除了实现细节之外，在使用`create_symlink`或`create_hard_link`时有一个*明显的错误*源。以下行包含一个错误。你能立刻发现它吗？

```cpp
path a {"some_dir/some_file.txt"};
path b {"other_dir/other_file.txt"};
remove(b);
create_symlink(a, b);
```

执行此程序时不会发生任何不良情况，但符号链接将会*损坏*。符号链接指向`"some_dir/some_file.txt"`，这是错误的。问题在于它实际上应该指向`"/absolute/path/some_dir/some_file.txt"`，或者`"../some_dir/some_file.txt"`。如果我们将`create_symlink`调用写成以下形式，则使用了正确的绝对路径：

```cpp
create_symlink(absolute(a), b);
```

`create_symlink`不检查我们要链接的路径是否*正确*。

# 还有更多...

我们已经注意到我们的哈希函数太简单了。为了使这个方法简单并且没有外部依赖，我们选择了这种方式。

我们的哈希函数有什么问题？实际上有两个问题：

+   我们将整个文件读入一个字符串。这对于大于我们系统内存的文件是灾难性的。

+   C++哈希函数特性`hash<string>`很可能不是为这样的哈希设计的。

如果我们正在寻找更好的哈希函数，我们应该选择一个快速、内存友好的函数，并确保没有两个真正大但不同的文件得到相同的哈希值。后一个要求可能是最重要的。如果我们决定一个文件是另一个文件的副本，尽管它们不包含相同的数据，那么在删除后我们肯定会有一些*数据丢失*。

更好的哈希算法例如 MD5 或 SHA 变体之一。为了在我们的程序中访问这样的函数，我们可以使用 OpenSSL 密码 API。
