# 第十章：*第十章*：使用文件系统

STL `filesystem`库的目的是在各个平台上标准化文件系统操作。`filesystem`库旨在标准化操作，弥合 POSIX/Unix、Windows 和其他文件系统之间的不规则性。

`filesystem`库是从相应的*Boost*库中采纳的，并在 C++17 中纳入了 STL。在撰写本文时，一些系统上其实现仍存在空白，但本章中的菜谱已在 Linux、Windows 和 macOS 文件系统上进行了测试，并分别使用最新的 GCC、MSVC 和 Clang 编译器编译。

该库使用`<filesystem>`头文件，并且`std::filesystem`命名空间通常被别名为`fs`：

```cpp
namespace fs = std::filesystem;
```

`fs::path`类是`filesystem`库的核心。它为不同的环境提供了标准化的文件名和目录路径表示。一个`path`对象可以表示一个文件、一个目录，甚至是一个，即使是一个不存在或不可能的对象。

在下面的菜谱中，我们将介绍使用`filesystem`库处理文件和目录的工具：

+   为`path`类特化`std::formatter`

+   使用`path`的操纵函数

+   列出目录中的文件

+   使用`grep`实用工具搜索目录和文件

+   使用`regex`和`directory_iterator`重命名文件

+   创建磁盘使用计数器

# 技术要求

您可以在 GitHub 上找到本章的代码文件，链接为[`github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap10`](https://github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap10)。

# 为 path 类特化 std::formatter

在`filesystem`库中，`path`类被用来表示文件或目录路径。在符合 POSIX 的系统上，例如 macOS 和 Linux，`path`对象使用`char`类型来表示文件名。在 Windows 上，`path`使用`wchar_t`。在 Windows 上，`cout`和`format()`将不会显示`wchar_t`字符的原始字符串。这意味着没有简单的方法来编写既使用`filesystem`库又在 POSIX 和 Windows 之间可移植的代码。

我们可以使用预处理器指令为 Windows 编写特定版本的代码。这可能是一些代码库的合理解决方案，但对于这本书来说，它会变得混乱，并且不符合简单、可移植、可重用的菜谱的目的。

精美的解决方案是编写一个 C++20 的`formatter`特化，用于`path`类。这允许我们简单地、可移植地显示`path`对象。

## 如何做到这一点...

在这个菜谱中，我们编写了一个`formatter`特化，用于与`fs::path`类一起使用：

+   为了方便，我们首先定义一个命名空间别名。所有的`filesystem`名称都在`std::filesystem`命名空间中：

    ```cpp
    namespace fs = std::filesystem;
    ```

+   我们为`path`类提供的`formatter`特化简单而简洁：

    ```cpp
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

在这里，我们正在为`fs::path`类型特殊化`formatter`，使用其`string()`方法获取可打印的表示形式。我们无法使用`c_str()`方法，因为它在 Windows 上的`wchar_t`字符上不起作用。

本书中的*第一章*，“新 C++20 特性”，对`formatter`特殊化的解释更为完整。

+   在`main()`函数中，我们使用命令行传递一个文件名或路径：

    ```cpp
    int main(const int argc, const char** argv) {
        if(argc != 2) {
            fs::path fn{ argv[0] };
            cout << format("usage: {} <path>\n", 
              fn.filename());
            return 0;
        }
        fs::path dir{ argv[1] };
        if(!fs::exists(dir)) {
            cout << format("path: {} does not exist\n", 
              dir);
            return 1;
        }
        cout << format("path: {}\n", dir);
        cout << format("filename: {}\n", dir.filename());
        cout << format("cannonical: {}\n", 
          fs::canonical(dir));
    }
    ```

`argc`和`argv`参数是标准的命令行参数。

`argv[0]`始终是可执行文件的完整目录路径和文件名。如果我们没有正确的参数数量，我们将显示`argv[0]`中的文件名部分作为我们的*用法*消息的一部分。

我们在这个例子中使用了一些`filesystem`函数：

+   `fs::exists()`函数检查目录或文件是否存在。

+   `dir`是一个`path`对象。我们现在可以直接将其传递给`format()`，使用我们的特殊化来显示路径的字符串表示形式。

+   `filename()`方法返回一个新的`path`对象，我们将其直接传递给`format()`，使用我们的特殊化。

+   `fs::cannonical()`函数接受一个`path`对象，并返回一个新的`path`对象，其中包含规范绝对目录路径。我们直接将此`path`对象传递给`format()`，并显示从`cannonical()`返回的目录路径。

输出：

```cpp
$ ./formatter ./formatter.cpp
path: ./formatter.cpp
filename: formatter.cpp
cannonical: /home/billw/working/chap10/formatter.cpp
```

## 它是如何工作的…

`fs::path`类在`filesystem`库中用于表示目录路径和文件名。通过提供`formatter`特殊化，我们可以轻松地在各个平台上一致地显示`path`对象。

`path`类提供了一些有用的方法。我们可以遍历路径以查看其组成部分：

```cpp
fs::path p{ "~/include/bwprint.h" };
cout << format("{}\n", p);
for(auto& x : p) cout << format("[{}] ", x);
cout << '\n';
```

输出：

```cpp
~/include/bwprint.h
[~] [include] [bwprint.h]
```

迭代器为路径的每个元素返回一个`path`对象。

我们也可以获取路径的不同部分：

```cpp
fs::path p{ "~/include/bwprint.h" };
cout << format("{}\n", p);
cout << format("{}\n", p.stem());
cout << format("{}\n", p.extension());
cout << format("{}\n", p.filename());
cout << format("{}\n", p.parent_path());
```

输出：

```cpp
~/include/bwprint.h
bwprint
.h
bwprint.h
~/include
```

我们将在本章中继续使用这个`formatter`特殊化来处理`path`类。

# 使用路径操作函数

`filesystem`库包括用于操作`path`对象内容的函数。在本例中，我们将考虑这些工具中的几个。

## 如何做到这一点…

在这个菜谱中，我们检查了一些操作`path`对象内容的函数：

+   我们从`namespace`指令和我们的`formatter`特殊化开始。我们在本章的每个菜谱中都这样做：

    ```cpp
    namespace fs = std::filesystem;
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

+   我们可以使用`current_path()`函数获取当前工作目录，该函数返回一个`path`对象：

    ```cpp
    cout << format("current_path: {}\n", fs::current_path());
    ```

输出：

```cpp
current_path: /home/billw/chap10
```

+   `absolute()`函数从相对路径返回绝对路径：

    ```cpp
    cout << format("absolute(p): {}\n", fs::absolute(p));
    ```

输出：

```cpp
absolute(p): /home/billw/chap10/testdir/foo.txt
```

`absolute()`也会取消符号链接的引用。

+   `+=`运算符将字符串连接到`path`字符串的末尾：

    ```cpp
    cout << format("concatenate: {}\n",
        fs::path{ "testdir" } += "foo.txt");
    ```

输出：

```cpp
concatenate: testdirfoo.txt
```

+   `/=`运算符将字符串附加到`path`字符串的末尾并返回一个新的`path`对象：

    ```cpp
    cout << format("append: {}\n",
        fs::path{ "testdir" } /= "foo.txt");
    ```

输出：

```cpp
append: testdir/foo.txt
```

+   `canonical()`函数返回完整的规范目录路径：

    ```cpp
    cout << format("canonical: {}\n",
        fs::canonical(fs::path{ "." } /= "testdir"));
    ```

输出：

```cpp
canonical: /home/billw/chap10/testdir
```

+   `equivalent()`函数测试两个相对路径是否解析到相同的文件系统实体：

    ```cpp
    cout << format("equivalent: {}\n", 
        fs::equivalent("testdir/foo.txt", 
            "testdir/../testdir/foo.txt"));
    ```

输出：

```cpp
equivalent: true
```

+   `filesystem` 库包含了用于异常处理的 `filesystem_error` 类：

    ```cpp
    try {
        fs::path p{ fp };
        cout << format("p: {}\n", p);
        ...
        cout << format("equivalent: {}\n", 
            fs::equivalent("testdir/foo.txt", 
                "testdir/../testdir/foo.txt"));
    } catch (const fs::filesystem_error& e) {
        cout << format("{}\n", e.what());
        cout << format("path1: {}\n", e.path1());
        cout << format("path2: {}\n", e.path2());
    }
    ```

`filesystem_error` 类包含了显示错误信息和获取涉及错误路径的方法。

如果我们在 `equivalent()` 调用中引入错误，我们可以看到 `filesystem_error` 类的结果：

```cpp
cout << format("equivalent: {}\n", 
    fs::equivalent("testdir/foo.txt/x", 
        "testdir/../testdir/foo.txt/y"));
```

输出：

```cpp
filesystem error: cannot check file equivalence: No such file or directory [testdir/foo.txt/x] [testdir/../testdir/foo.txt/y]
path1: testdir/foo.txt/x
path2: testdir/../testdir/foo.txt/y
```

这是 Debian 上使用 GCC 的输出。

`filesystem_error` 类通过其 `path1()` 和 `path2()` 方法提供了额外的详细信息。这些方法返回 `path` 对象。

+   你也可以使用 `std::error_code` 与一些 `filesystem` 函数一起使用：

    ```cpp
    fs::path p{ fp };
    std::error_code e;
    cout << format("canonical: {}\n", 
        fs::canonical(p /= "foo", e));
    cout << format("error: {}\n", e.message());
    ```

输出：

```cpp
canonical:
error: Not a directory
```

+   尽管 Windows 使用一个非常不同的文件系统，但此代码仍然按预期工作，使用 Windows 文件命名约定：

    ```cpp
    p: testdir/foo.txt
    current_path: C:\Users\billw\chap10
    absolute(p): C:\Users\billw\chap10\testdir\foo.txt
    concatenate: testdirfoo.txt
    append: testdir\foo.txt
    canonical: C:\Users\billw\chap10\testdir
    equivalent: true
    ```

## 它是如何工作的...

大多数这些函数接受一个 `path` 对象、一个可选的 `std::error_code` 对象，并返回一个 `path` 对象：

```cpp
path absolute(const path& p);
path absolute(const path& p, std::error_code& ec);
```

`equivalent()` 函数接受两个 `path` 对象并返回一个 `bool`：

```cpp
bool equivalent( const path& p1, const path& p2 );
bool equivalent( const path& p1, const path& p2,
    std::error_code& ec );
```

`path` 类有用于连接和追加的运算符。这两个运算符都是破坏性的。它们修改运算符左侧的 `path`：

```cpp
p1 += source; // concatenate
p1 /= source; // append
```

对于右侧，这些运算符可以接受一个 `path` 对象、一个 `string`、一个 `string_view`、一个 C 字符串或一对迭代器。

连接运算符将运算符右侧的字符串添加到 `p1` `path` 字符串的末尾。

追加运算符添加一个分隔符（例如，`/` 或 `\`），然后是运算符右侧的字符串到 `p1` `path` 字符串的末尾。

# 列出目录中的文件

`filesystem` 库提供了一个包含给定 `path` 的目录相关信息的 `directory_entry` 类。我们可以使用它来创建有用的目录列表。

## 如何做到这一点...

在这个菜谱中，我们使用 `directory_entry` 类中的信息创建一个目录列表实用工具：

+   我们从命名空间别名和用于显示 `path` 对象的 `formatter` 特化开始：

    ```cpp
    namespace fs = std::filesystem;
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

+   `directory_iterator` 类使得列出目录变得容易：

    ```cpp
    int main() {
        constexpr const char* fn{ "." };
        const fs::path fp{fn};
        for(const auto& de : fs::directory_iterator{fp}) {
            cout << format("{} ", de.path().filename());
        }
        cout << '\n';
    }
    ```

输出：

```cpp
chrono Makefile include chrono.cpp working formatter testdir formatter.cpp working.cpp
```

+   我们可以添加命令行选项来使它工作，就像 Unix `ls`：

    ```cpp
    int main(const int argc, const char** argv) {
        fs::path fp{ argc > 1 ? argv[1] : "." };
        if(!fs::exists(fp)) {
            const auto cmdname { 
              fs::path{argv[0]}.filename() };
            cout << format("{}: {} does not exist\n",
                cmdname, fp);
            return 1;
        }
        if(is_directory(fp)) {
            for(const auto& de : 
              fs::directory_iterator{fp}) {
                cout << format("{} ", 
                  de.path().filename());
            }
        } else {
            cout << format("{} ", fp.filename());
        }
        cout << '\n';
    }
    ```

如果有命令行参数，我们使用它来创建一个 `path` 对象。否则，我们使用 `"."` 表示当前目录。

我们使用 `if_exists()` 检查路径是否存在。如果不存在，我们打印错误消息并退出。错误消息包括 `argv[0]` 中的 `cmdname`。

接下来，我们检查 `is_directory()`。如果我们有一个目录，我们通过 `directory_iterator` 对每个条目进行循环。`directory_iterator` 遍历 `directory_entry` 对象。`de.path().filename()` 从每个 `directory_entry` 对象中获取 `path` 和 `filename`。

输出：

```cpp
$ ./working
chrono Makefile include chrono.cpp working formatter testdir formatter.cpp working.cpp
$ ./working working.cpp
working.cpp
$ ./working foo.bar
working: foo.bar does not exist
```

+   如果我们希望输出排序，我们可以将我们的 `directory_entry` 对象存储在可排序的容器中。

让我们为 `fs::directory_entry` 创建一个别名。我们会经常使用它。这个别名放在文件的顶部：

```cpp
using de = fs::directory_entry;
```

在 `main()` 函数的顶部，我们声明了一个 `de` 对象的 `vector`：

```cpp
vector<de> entries{};
```

在 `is_directory()` 块内部，我们加载 `vector`，对其进行排序，然后显示它：

```cpp
if(is_directory(fp)) {
    for(const auto& de : fs::directory_iterator{fp}) {
        entries.emplace_back(de);
    }
    std::sort(entries.begin(), entries.end());
    for(const auto& e : entries) {
        cout << format("{} ", e.path().filename());
    }
} else { ...
```

现在输出已经排序：

```cpp
Makefile chrono chrono.cpp formatter formatter.cpp include testdir working working.cpp
```

注意到 `Makefile` 是首先排序的，看起来顺序不对。这是因为大写字母在 ASCII 排序中排在小写字母之前。

+   如果我们想要不区分大小写的排序，我们需要一个忽略大小写的比较函数。首先，我们需要一个函数来返回小写的 `string`：

    ```cpp
    string strlower(string s) {
        auto char_lower = [](const char& c) -> char {
            if(c >= 'A' && c <= 'Z') return c + ('a' - 'A');
            else return c;
        };
        std::transform(s.begin(), s.end(), s.begin(),
            char_lower);
        return s;
    }
    ```

现在我们需要一个函数来比较两个 `directory_entry` 对象，使用 `strlower()`：

```cpp
bool dircmp_lc(const de& lhs, const de& rhs) {
    const auto lhstr{ lhs.path().string() };
    const auto rhstr{ rhs.path().string() };
    return strlower(lhstr) < strlower(rhstr);
}
```

现在，我们可以在排序中使用 `dircmp_lc()`：

```cpp
std::sort(entries.begin(), entries.end(), dircmp_lc);
```

我们现在忽略大小写排序的输出：

```cpp
chrono chrono.cpp formatter formatter.cpp include Makefile testdir working working.cpp
```

+   到目前为止，我们有一个简单的目录列表工具。

`filesystem` 库中还有更多可用的信息。让我们创建一个 `print_dir()` 函数来收集更多信息，并以 Unix `ls` 的样式格式化显示：

```cpp
void print_dir(const de& dir) {
    using fs::perms;
    const auto fpath{ dir.path() };
    const auto fstat{ dir.symlink_status() };
    const auto fperm{ fstat.permissions() };
    const uintmax_t fsize{ 
        is_regular_file(fstat) ? file_size(fpath) : 0 };
    const auto fn{ fpath.filename() };
    string suffix{};
    if(is_directory(fstat)) suffix = "/";
    else if((fperm & perms::owner_exec) != perms::none) {
        suffix = "*";
    }
    cout << format("{}{}\n", fn, suffix);
}
```

`print_dir()` 函数接受一个 `directory_entry` 参数。然后我们从 `directory_entry` 对象中检索一些有用的对象：

+   `dir.path()` 返回一个 `path` 对象。

+   `dir.symlink_status()` 返回一个 `file_status` 对象，不跟随符号链接。

+   `fstat.permissions()` 返回一个 `perms` 对象。

+   `fsize` 是文件的大小，`fn` 是文件名 `string`。我们将在使用它们时更详细地查看这些。

Unix `ls` 使用文件名之后的尾随字符来指示目录或可执行文件。我们用 `is_directory()` 测试 `fstat` 对象以查看文件是否是目录，并将尾随的 `/` 添加到文件名。同样，我们可以用 `fperm` 对象测试文件是否可执行。

在 `sort()` 之后，我们在 `for` 循环中调用 `main()` 中的 `print_dir()`：

```cpp
std::sort(entries.begin(), entries.end(), dircmp_lc);
for(const auto& e : entries) {
    print_dir(e);
}
```

我们现在的输出看起来像这样：

```cpp
chrono*
chrono.cpp
formatter*
formatter.cpp
include*
Makefile
testdir/
working*
working.cpp
```

+   注意到 `include*` 条目。实际上这是一个符号链接。让我们通过跟随链接来正确地标记目标路径：

    ```cpp
    string suffix{};
    if(is_symlink(fstat)) {
        suffix = " -> ";
        suffix += fs::read_symlink(fpath).string();
    }
    else if(is_directory(fstat)) suffix = "/";
    else if((fperm & perms::owner_exec) != perms::none) suffix = "*";
    ```

`read_symlink()` 函数返回一个 `path` 对象。我们取返回的 `path` 对象的 `string()` 表示形式，并将其添加到这个输出的后缀：

```cpp
chrono*
chrono.cpp
formatter*
formatter.cpp
include -> /Users/billw/include
Makefile
testdir/
working*
working.cpp
```

+   Unix `ls` 命令还包括一个字符序列来指示文件的权限位。它看起来像这样：`drwxr-xr-x`。

第一个字符表示文件的类型，例如：`d` 表示目录，`l` 表示符号链接，`-` 表示常规文件。

`type_char()` 函数返回适当的字符：

```cpp
char type_char(const fs::file_status& fstat) {
         if(is_symlink(fstat))        return 'l';
    else if(is_directory(fstat))      return 'd';
    else if(is_character_file(fstat)) return 'c';
    else if(is_block_file(fstat))     return 'b';
    else if(is_fifo(fstat))           return 'p';
    else if(is_socket(fstat))         return 's';
    else if(is_other(fstat))          return 'o';
    else if(is_regular_file(fstat))   return '-';
    return '?';
}
```

字符串的其余部分分为三个三元组。每个三元组包括读取、写入和执行权限位的位，形式为 `rwx`。如果位未设置，则其字符被替换为 `-`。有三组权限位，分别对应所有者、组和其它。

```cpp
string rwx(const fs::perms& p) {
    using fs::perms;
    auto bit2char = &p {
        return (p & bit) == perms::none ? '-' : c;
    };
    return { bit2char(perms::owner_read,   'r'),
             bit2char(perms::owner_write,  'w'),
             bit2char(perms::owner_exec,   'x'),
             bit2char(perms::group_read,   'r'),
             bit2char(perms::group_write,  'w'),
             bit2char(perms::group_exec,   'x'),
             bit2char(perms::others_read,  'r'),
             bit2char(perms::others_write, 'w'),
             bit2char(perms::others_exec,  'x') };
}
```

`perms` 对象代表 POSIX 权限位图，但并不一定以位的形式实现。每个条目都必须与 `perms::none` 值进行比较。我们的 lambda 函数满足这一要求。

我们将这个定义添加到 `print_dir()` 函数的顶部：

```cpp
const auto permstr{ type_char(fstat) + rwx(fperm) };
```

我们更新我们的 `format()` 字符串：

```cpp
cout << format("{} {}{}\n", permstr, fn, suffix);
```

我们得到以下输出：

```cpp
-rwxr-xr-x chrono*
-rw-r--r-- chrono.cpp
-rwxr-xr-x formatter*
-rw-r--r-- formatter.cpp
lrwxr-xr-x include -> /Users/billw/include
-rw-r--r-- Makefile
drwxr-xr-x testdir/
-rwxr-xr-x working*
-rw-r--r-- working.cpp
```

+   现在，让我们添加一个大小字符串。`fsize`值来自`file_size()`函数，它返回一个`std::uintmax_t`类型。这代表目标系统上的最大自然整数。`uintmax_t`并不总是与`size_t`相同，并且并不总是容易转换。值得注意的是，在 Windows 上`uintmax_t`是 32 位，而`size_t`是 64 位：

    ```cpp
    string size_string(const uintmax_t fsize) {
        constexpr const uintmax_t kilo{ 1024 };
        constexpr const uintmax_t mega{ kilo * kilo };
        constexpr const uintmax_t giga{ mega * kilo };
        string s;
        if(fsize >= giga ) return
            format("{}{}", (fsize + giga / 2) / giga, 'G');
        else if (fsize >= mega) return
            format("{}{}", (fsize + mega / 2) / mega, 'M');
        else if (fsize >= kilo) return
            format("{}{}", (fsize + kilo / 2) / kilo, 'K');
        else return format("{}B", fsize);
    }
    ```

我选择在这个函数中使用 1,024 作为 1K，因为这看起来是 Linux 和 BSD Unix 的默认设置。在生产环境中，这可以是一个命令行选项。

我们在`main()`中更新我们的`format()`字符串：

```cpp
cout << format("{} {:>6} {}{}\n",
    permstr, size_string(fsize), fn, suffix);
```

现在，我们得到这个输出：

```cpp
-rwxr-xr-x   284K chrono*
-rw-r--r--     2K chrono.cpp
-rwxr-xr-x   178K formatter*
-rw-r--r--   906B formatter.cpp
lrwxr-xr-x     0B include -> /Users/billw/include
-rw-r--r--   642B Makefile
drwxr-xr-x     0B testdir/
-rwxr-xr-x   197K working*
-rw-r--r--     5K working.cpp
```

注意

这个实用程序是为 POSIX 系统设计的，例如 Linux 和 macOS。它在 Windows 系统上也能工作，但 Windows 的权限系统与 POSIX 系统不同。在 Windows 上，权限位总是完全设置的。

## 它是如何工作的...

`filesystem`库通过其`directory_entry`和相关类携带丰富的信息。我们在本菜谱中使用的的主要类包括：

+   `path`类表示一个文件系统路径，根据目标系统的规则。一个`path`对象可以从一个字符串或另一个路径构造而成。它不需要表示一个现有路径，甚至不是一个可能的路径。路径字符串被解析成组件部分，包括根名称、根目录以及一系列可选的文件名和目录分隔符。

+   `directory_entry`类携带一个`path`对象作为成员，并且也可能存储额外的属性，包括硬链接计数、状态、符号链接、文件大小和最后写入时间。

+   `file_status`类携带有关文件类型和权限的信息。`perms`对象可能是`file_status`的一个成员，表示文件的权限结构。

有两个函数可以从`file_status`检索`perms`对象。`status()`函数和`symlink_status()`函数都返回一个`perms`对象。区别在于它们处理符号链接的方式。`status()`函数会跟随符号链接并返回目标文件的`perms`。`symlink_status()`将返回符号链接本身的`perms`。

## 更多...

我原本打算在目录列表中包含每个文件的最后写入时间。

`directory_entry`类有一个成员函数`last_write_time()`，它返回一个表示文件最后一次写入时间戳的`file_time_type`对象。

不幸的是，在写作的时候，可用的实现缺乏一种可移植的方式来将`file_time_type`对象转换为标准的`chrono::sys_time`，适用于与`cout`或`format()`一起使用。

目前，这里有一个与 GCC 兼容的解决方案：

```cpp
string time_string(const fs::directory_entry& dir) {
    using std::chrono::file_clock;
    auto file_time{ dir.last_write_time() };
    return format("{:%F %T}", 
        file_clock::to_sys(dir.last_write_time()));
}
```

建议用户代码使用`std::chrono::clock_cast`而不是`file::clock::to_sys`来在时钟之间转换时间点。不幸的是，目前可用的实现中没有任何一个为这个目的工作的`std::chrono::clock_cast`特化。

使用这个`time_string()`函数，我们可以在`print_dir()`中添加：

```cpp
const string timestr{ time_string(dir) };
```

然后，我们可以更改 `format()` 字符串：

```cpp
cout << format("{} {:>6} {} {}{}\n",
    permstr, sizestr, timestr, fn, suffix);
```

我们得到以下输出：

```cpp
-rwxr-xr-x   248K 2022-03-09 09:39:49 chrono*
-rw-r--r--     2K 2022-03-09 09:33:56 chrono.cpp
-rwxr-xr-x   178K 2022-03-09 09:39:49 formatter*
-rw-r--r--   906B 2022-03-09 09:33:56 formatter.cpp
lrwxrwxrwx     0B 2022-02-04 11:39:53 include -> /home/billw/include
-rw-r--r--   642B 2022-03-09 14:08:37 Makefile
drwxr-xr-x     0B 2022-03-09 10:38:39 testdir/
-rwxr-xr-x   197K 2022-03-12 17:13:46 working*
-rw-r--r--     5K 2022-03-12 17:13:40 working.cpp
```

这在 Debian 系统上使用 GCC-11 是可行的。不要期望它在任何其他系统上无需修改就能工作。

# 使用 grep 工具搜索目录和文件

为了演示遍历和搜索目录结构，我们创建了一个类似于 Unix *grep* 的简单工具。这个工具使用 `recursive_directory_iterator` 来遍历嵌套目录，并搜索与正则表达式匹配的文件。

## 如何做到这一点...

在这个菜谱中，我们编写了一个简单的 *grep* 工具，它遍历目录以搜索使用正则表达式的文件：

+   我们从一些便利的别名开始：

    ```cpp
    namespace fs = std::filesystem;
    using de = fs::directory_entry;
    using rdit = fs::recursive_directory_iterator;
    using match_v = vector<std::pair<size_t, std::string>>;
    ```

`match_v` 是正则表达式匹配结果的一个 `vector`。

+   我们继续使用 `formatter` 特化来处理 `path` 对象：

    ```cpp
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

+   我们有一个简单的函数用于从文件中获取正则表达式匹配：

    ```cpp
    match_v matches(const fs::path& fpath, const regex& re) {
        match_v matches{};
        std::ifstream instrm(fpath.string(),
            std::ios_base::in);
        string s;
        for(size_t lineno{1}; getline(instrm, s); ++lineno) {
            if(std::regex_search(s.begin(), s.end(), re)) {
                matches.emplace_back(lineno, move(s));
            }
        }
        return matches;
    }
    ```

在这个函数中，我们使用 `ifstream` 打开文件，使用 `getline()` 从文件中读取行，并使用 `regex_search()` 匹配正则表达式。结果收集在 `vector` 中并返回。

+   我们现在可以从 `main()` 中调用这个函数：

    ```cpp
    int main() {
        constexpr const char * fn{ "working.cpp" };
        constexpr const char * pattern{ "path" };
        fs::path fpath{ fn };
        regex re{ pattern };
        auto regmatches{ matches(fpath, re) };
        for(const auto& [lineno, line] : regmatches) {
            cout << format("{}: {}\n", lineno, line);
        }
        cout << format("found {} matches\n", regmatches.size());
    }
    ```

在这个例子中，我们使用常量来表示文件名和正则表达式模式。我们创建 `path` 和 `regex` 对象，调用 `matches()` 函数，并打印结果。

我们输出带有行号和匹配行的字符串：

```cpp
25: struct std::formatter<fs::path>: std::formatter<std::string> {
27:     auto format(const fs::path& p, FormatContext& ctx) {
32: match_v matches(const fs::path& fpath, const regex& re) {
34:     std::ifstream instrm(fpath.string(), std::ios_base::in);
62:     constexpr const char * pattern{ "path" };
64:     fs::path fpath{ fn };
66:     auto regmatches{ matches(fpath, re) };
```

+   我们的工具需要接受命令行参数作为 `regex` 模式和文件名。它应该能够遍历目录或接受文件名列表（这可能是命令行通配符扩展的结果）。这需要在 `main()` 函数中添加一些逻辑。

首先，我们需要一个额外的辅助函数：

```cpp
size_t pmatches(const regex& re, const fs::path& epath,
        const fs::path& search_path) {
    fs::path target{epath};
    auto regmatches{ matches(epath, re) };
    auto matchcount{ regmatches.size() };
    if(!matchcount) return 0;
    if(!(search_path == epath)) {
        target = 
          epath.lexically_relative(search_path);
    }
    for (const auto& [lineno, line] : regmatches) {
        cout << format("{} {}: {}\n", target, lineno, 
          line);
    }
    return regmatches.size();
}
```

这个函数调用我们的 `matches()` 函数并打印结果。它接受一个 `regex` 对象和两个 `path` 对象。`epath` 是目录搜索的结果，而 `search_path` 是搜索的目录本身。我们将在 `main()` 中设置这些。

+   在 `main()` 中，我们使用 `argc` 和 `argv` 命令行参数，并声明了一些变量：

    ```cpp
    int main(const int argc, const char** argv) {
        const char * arg_pat{};
        regex re{};
        fs::path search_path{};
        size_t matchcount{};
        ...
    ```

这里声明的变量有：

+   `arg_pat` 是用于命令行中的正则表达式模式。

+   `re` 是 `regex` 对象。

+   `search_path` 是命令行搜索路径参数。

+   `matchcount` 用于计算匹配的行数。

+   继续在 `main()` 中，如果没有参数，则打印一个简短的用法字符串：

    ```cpp
    if(argc < 2) {
        auto cmdname{ fs::path(argv[0]).filename() };
        cout << format("usage: {} pattern [path/file]\n", 
            cmdname);
        return 1;
    }
    ```

`argv[1]` 总是命令行中的调用命令。`cmdname` 使用 `filename()` 方法返回一个只包含调用命令路径文件名的 `path`。

+   接下来，我们解析正则表达式。我们使用 `try-catch` 块来捕获 `regex` 解析器可能产生的任何错误：

    ```cpp
    arg_pat = argv[1];
    try {
        re = regex(arg_pat, std::regex_constants::icase);
    } catch(const std::regex_error& e) {
        cout << format("{}: {}\n", e.what(), arg_pat);
        return 1;
    }
    ```

我们使用 `icase` 标志来告诉 `regex` 解析器忽略大小写。

+   如果 `argc == 2`，我们只有一个参数，我们将其视为正则表达式模式，并使用当前目录作为搜索路径：

    ```cpp
    if(argc == 2) {
        search_path = ".";
            for (const auto& entry : rdit{ search_path }) {
            const auto epath{ entry.path() };
            matchcount += pmatches(re, epath, 
              search_path);
        }
    }
    ```

`rdit`是`recursive_directory_iterator`类的别名，它从起始路径遍历目录树，为遇到的每个文件返回一个`directory_entry`对象。然后我们创建一个`path`对象并调用`pmatches()`来遍历文件并打印任何正则表达式匹配。

+   在`main()`的这个点上，我们知道`argc`是`>=2`。现在，我们处理命令行上有一个或多个文件路径的情况：

    ```cpp
    int count{ argc - 2 };
    while(count-- > 0) {
        fs::path p{ argv[count + 2] };
        if(!exists(p)) {
            cout << format("not found: {}\n", p);
            continue;
        }
        if(is_directory(p)) {
            for (const auto& entry : rdit{ p }) {
                const auto epath{ entry.path() };
                matchcount += pmatches(re, epath, p);
            }
        } else {
            matchcount += pmatches(re, p, p);
        }
    }
    ```

`while`循环处理命令行上的搜索模式之后的一个或多个参数。它检查每个文件名以确保它存在。然后，如果它是一个目录，它使用`rdit`别名（`recursive_directory_iterator`类）来遍历目录并调用`pmatches()`来打印文件中的任何模式匹配。

如果是单个文件，它会在该文件上调用`pmatches()`。

+   我们可以用一个搜索模式作为参数运行我们的`grep`克隆：

    ```cpp
    $ ./bwgrep using
    dir.cpp 12: using std::format;
    dir.cpp 13: using std::cout;
    dir.cpp 14: using std::string;
    ...
    formatter.cpp 10: using std::cout;
    formatter.cpp 11: using std::string;
    formatter.cpp 13: using namespace std::filesystem;
    found 33 matches
    ```

我们可以用第二个参数作为搜索目录来运行它：

```cpp
$ ./bwgrep using ..
chap04/iterator-adapters.cpp 12: using std::format;
chap04/iterator-adapters.cpp 13: using std::cout;
chap04/iterator-adapters.cpp 14: using std::cin;
...
chap01/hello-version.cpp 24: using std::print;
chap01/chrono.cpp 8: using namespace std::chrono_literals;
chap01/working.cpp 15: using std::cout;
chap01/working.cpp 34:     using std::vector;
found 529 matches
```

注意，它*遍历目录树*来查找子目录中的文件。

或者我们可以用一个单个文件参数来运行它：

```cpp
$ ./bwgrep using bwgrep.cpp
bwgrep.cpp 13: using std::format;
bwgrep.cpp 14: using std::cout;
bwgrep.cpp 15: using std::string;
...
bwgrep.cpp 22: using rdit = fs::recursive_directory_iterator;
bwgrep.cpp 23: using match_v = vector<std::pair<size_t, std::string>>;
found 9 matches
```

## 它是如何工作的…

虽然这个实用程序的主要任务是正则表达式匹配，但我们专注于递归处理文件目录的技术。

`recursive_directory_iterator`对象与`directory_iterator`可互换，除了`recursive_directory_iterator`递归地遍历每个子目录的所有条目。

## 参见…

更多关于正则表达式的信息，请参阅*第七章*中的配方*使用正则表达式解析字符串*，*字符串、流和格式化*。

# 使用正则表达式和目录迭代器重命名文件

这是一个简单的实用程序，使用正则表达式重命名文件。它使用`directory_iterator`在目录中查找文件，并使用`fs::rename()`来重命名它们。

## 如何做到这一点…

在这个配方中，我们创建了一个使用正则表达式的文件重命名实用程序：

+   我们首先定义一些便利别名：

    ```cpp
    namespace fs = std::filesystem;
    using dit = fs::directory_iterator;
    using pat_v = vector<std::pair<regex, string>>;
    ```

`pat_v`别名是一个用于正则表达式的`vector`。

+   我们还继续使用`path`对象的`formatter`特化：

    ```cpp
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

+   我们有一个函数用于将正则表达式替换应用到文件名字符串：

    ```cpp
    string replace_str(string s, const pat_v& replacements) {
        for(const auto& [pattern, repl] : replacements) {
            s = regex_replace(s, pattern, repl);
        }
        return s;
    }
    ```

注意，我们遍历一个包含模式/替换对的`vector`，依次应用正则表达式。这允许我们堆叠我们的替换。

+   在`main()`中，我们首先检查命令行参数：

    ```cpp
    int main(const int argc, const char** argv) {
        pat_v patterns{};
        if(argc < 3 || argc % 2 != 1) {
            fs::path cmdname{ fs::path{argv[0]}.filename() };
            cout << format(
                "usage: {} [regex replacement] ...\n", 
                cmdname);
            return 1;
        }
    ```

命令行接受一个或多个字符串对。每个字符串对包括一个正则表达式（正则表达式）后跟一个替换。

+   现在我们用`regex`和`string`对象填充`vector`：

    ```cpp
    for(int i{ 1 }; i < argc; i += 2) {
        patterns.emplace_back(argv[i], argv[i + 1]);
    }
    ```

`pair`构造函数在原地构造`regex`和`string`对象，从命令行传递的 C-字符串。这些通过`emplace_back()`方法添加到`vector`中。

+   我们使用`directory_iterator`对象在当前目录中搜索：

    ```cpp
    for(const auto& entry : dit{fs::current_path()}) {
        fs::path fpath{ entry.path() };
        string rname{
            replace_str(fpath.filename().string(), 
    patterns) };
        if(fpath.filename().string() != rname) {
            fs::path rpath{ fpath };
            rpath.replace_filename(rname);
            if(exists(rpath)) {
                cout << "Error: cannot rename - destination file exists.\n";
            } else {
                fs::rename(fpath, rpath);
                cout << format(
                    "{} -> {}\n", 
                    fpath.filename(), 
                    rpath.filename());
            }
        }
    }
    ```

在这个 `for` 循环中，我们调用 `replace_str()` 来获取替换后的文件名，然后检查新名称不是目录中文件的重复项。我们在 `path` 对象上使用 `replace_filename()` 方法来创建具有新文件名的 `path`，并使用 `fs::rename()` 来重命名文件。

+   为了测试这个实用程序，我创建了一个包含一些文件的目录以进行重命名：

    ```cpp
    $ ls
    bwfoo.txt bwgrep.cpp chrono.cpp dir.cpp formatter.cpp path-ops.cpp working.cpp
    ```

+   我们可以做一些简单的事情，比如将 `.cpp` 改为 `.Cpp`：

    ```cpp
    $ ../rerename .cpp .Cpp
    dir.cpp -> dir.Cpp
    path-ops.cpp -> path-ops.Cpp
    bwgrep.cpp -> bwgrep.Cpp
    working.cpp -> working.Cpp
    formatter.cpp -> formatter.Cpp
    ```

让我们再次更改它们：

```cpp
$ ../rerename .Cpp .cpp
formatter.Cpp -> formatter.cpp
bwgrep.Cpp -> bwgrep.cpp
dir.Cpp -> dir.cpp
working.Cpp -> working.cpp
path-ops.Cpp -> path-ops.cpp
```

+   使用标准的正则表达式语法，我可以将 "`bw`" 添加到每个文件名的开头：

    ```cpp
    $ ../rerename '^' bw
    bwgrep.cpp -> bwbwgrep.cpp
    chrono.cpp -> bwchrono.cpp
    formatter.cpp -> bwformatter.cpp
    bwfoo.txt -> bwbwfoo.txt
    working.cpp -> bwworking.cpp
    ```

注意，它甚至重命名了那些已经以 "`bw`" 开头的文件。让我们让它不要这样做。首先，我们恢复文件名：

```cpp
$ ../rerename '^bw' ''
bwbwgrep.cpp -> bwgrep.cpp
bwworking.cpp -> working.cpp
bwformatter.cpp -> formatter.cpp
bwchrono.cpp -> chrono.cpp
bwbwfoo.txt -> bwfoo.txt
```

现在我们使用一个检查文件名是否以 "`bw`" 开头的正则表达式：

```cpp
$ ../rerename '^(?!bw)' bw
chrono.cpp -> bwchrono.cpp
formatter.cpp -> bwformatter.cpp
working.cpp -> bwworking.cpp
```

因为我们使用正则表达式/替换字符串的 `vector`，所以我们可以堆叠多个替换：

```cpp
$ ../rerename foo bar '\.cpp$' '.xpp' grep grok
bwgrep.cpp -> bwgrok.xpp
bwworking.cpp -> bwworking.xpp
bwformatter.cpp -> bwformatter.xpp
bwchrono.cpp -> bwchrono.xpp
bwfoo.txt -> bwbar.txt
```

## 它是如何工作的…

此食谱中的 `filesystem` 部分使用 `directory_iterator` 返回当前目录中每个文件的 `directory_entry` 对象：

```cpp
for(const auto& entry : dit{fs::current_path()}) {
    fs::path fpath{ entry.path() };
    ...
}
```

然后，我们从 `directory_entry` 对象构造一个 `path` 对象来处理文件。

我们在 `path` 对象上使用 `replace_filename()` 方法来创建重命名操作的目标：

```cpp
fs::path rpath{ fpath };
rpath.replace_filename(rname);
```

在这里，我们创建一个副本并更改其名称，这样我们就有两个用于重命名操作的版本：

```cpp
fs::rename(fpath, rpath);
```

在正则表达式的这一侧，我们使用 `regex_replace()`，它使用正则表达式语法在字符串中执行替换：

```cpp
s = regex_replace(s, pattern, repl);
```

正则表达式语法非常强大。它甚至允许替换包括搜索字符串的部分：

```cpp
$ ../rerename '(bw)(.*\.)(.*)$' '$3$2$1'
bwgrep.cpp -> cppgrep.bw
bwfoo.txt -> txtfoo.bw
```

通过在搜索模式中使用括号，我可以轻松地重新排列文件名的一部分。

## 参见…

更多关于正则表达式的信息，请参阅 *第七章* 中的食谱 *使用正则表达式解析字符串*，*字符串、流和格式化*。

# 创建磁盘使用计数器

这是一个简单的实用程序，它计算目录及其子目录中每个文件的总大小。它可以在 POSIX/Unix 和 Windows 文件系统上运行。

## 如何做到这一点…

这个食谱是一个实用程序，用于报告目录及其子目录中每个文件的大小，以及总大小。我们将重用本章其他地方使用的一些函数：

+   我们从一些便利别名开始：

    ```cpp
    namespace fs = std::filesystem;
    using dit = fs::directory_iterator;
    using de = fs::directory_entry;
    ```

+   我们还使用了我们的 `format` 特化 `fs::path` 对象：

    ```cpp
    template<>
    struct std::formatter<fs::path>: std::formatter<std::string> {
        template<typename FormatContext>
        auto format(const fs::path& p, FormatContext& ctx) {
            return format_to(ctx.out(), "{}", p.string());
        }
    };
    ```

+   为了报告目录的大小，我们将使用这个 `make_commas()` 函数：

    ```cpp
    string make_commas(const uintmax_t& num) {
        string s{ std::to_string(num) };
        for(long l = s.length() - 3; l > 0; l -= 3) {
            s.insert(l, ",");
        }
        return s;
    }
    ```

我们之前已经使用过这个。它在每个第三个字符之前插入一个逗号。

+   为了对目录进行排序，我们需要一个将字符串转换为小写的函数：

    ```cpp
    string strlower(string s) {
        auto char_lower = [](const char& c) -> char {
            if(c >= 'A' && c <= 'Z') return c + ('a' – 
               'A');
            else return c;
        };
        std::transform(s.begin(), s.end(), s.begin(), 
          char_lower);
        return s;
    }
    ```

+   我们需要一个比较谓词来按 `path` 名称的小写对 `directory_entry` 对象进行排序：

    ```cpp
    bool dircmp_lc(const de& lhs, const de& rhs) {
        const auto lhstr{ lhs.path().string() };
        const auto rhstr{ rhs.path().string() };
        return strlower(lhstr) < strlower(rhstr);
    }
    ```

+   `size_string()` 返回用于报告文件大小的缩写值，单位为千兆字节、兆字节、千字节或字节：

    ```cpp
    string size_string(const uintmax_t fsize) {
        constexpr const uintmax_t kilo{ 1024 };
        constexpr const uintmax_t mega{ kilo * kilo };
        constexpr const uintmax_t giga{ mega * kilo };
        if(fsize >= giga ) return format("{}{}",
            (fsize + giga / 2) / giga, 'G');
        else if (fsize >= mega) return format("{}{}",
            (fsize + mega / 2) / mega, 'M');
        else if (fsize >= kilo) return format("{}{}",
            (fsize + kilo / 2) / kilo, 'K');
        else return format("{}B", fsize);
    }
    ```

+   `entry_size()` 返回文件的大小，如果是目录，则返回目录的递归大小：

    ```cpp
    uintmax_t entry_size(const fs::path& p) {
        if(fs::is_regular_file(p)) return 
           fs::file_size(p);
        uintmax_t accum{};
        if(fs::is_directory(p) && ! fs::is_symlink(p)) {
            for(auto& e : dit{ p }) {
                accum += entry_size(e.path());
            }
        }
        return accum;
    }
    ```

+   在`main()`中，我们开始声明并测试是否有有效的目录要搜索：

    ```cpp
    int main(const int argc, const char** argv) {
        auto dir{ argc > 1 ? 
            fs::path(argv[1]) : fs::current_path() };
        vector<de> entries{};
        uintmax_t accum{};
        if (!exists(dir)) {
            cout << format("path {} does not exist\n", 
              dir);
            return 1;
        }
        if(!is_directory(dir)) {
            cout << format("{} is not a directory\n", 
              dir);
            return 1;
        }
        cout << format("{}:\n", absolute(dir));
    ```

对于我们的目录路径`dir`，如果我们有一个参数，我们使用`argv[1]`；否则，我们使用`current_path()`表示当前目录。然后我们为我们的使用计数器设置一个环境：

+   `directory_entry`对象的`vector`用于对响应进行排序。

+   `accum`用于累计我们最终的总大小值。

+   在检查目录之前，我们确保`dir`存在并且是一个目录。

+   接下来，一个简单的循环来填充`vector`。一旦填充完成，我们使用我们的`dircmp_lc()`函数作为比较谓词对`entries`进行排序：

    ```cpp
    for (const auto& e : dit{ dir }) {
        entries.emplace_back(e.path());
    }
    std::sort(entries.begin(), entries.end(), dircmp_lc);
    ```

+   现在一切都已经设置好了，我们可以从排序的`directory_entry`对象的`vector`中累计结果：

    ```cpp
    for (const auto& e : entries) {
        fs::path p{ e };
        uintmax_t esize{ entry_size(p) };
        string dir_flag{};
        accum += esize;
        if(is_directory(p) && !is_symlink(p)) dir_flag = 
           " ![](img/6.png)";
        cout << format("{:>5} {}{}\n",
            size_string(esize), p.filename(), dir_flag);
    }
    cout << format("{:->25}\n", "");
    cout << format("total bytes: {} ({})\n",
        make_commas(accum), size_string(accum));
    ```

`entry_size()`的调用返回由`directory_entry`对象表示的文件或目录的大小。

如果当前条目是一个目录（而不是一个*符号链接*），我们添加一个符号来表示它是一个目录。我选择了一个倒三角形。你可以在这里使用任何东西。

循环完成后，我们以字节为单位显示累计的大小，并用逗号分隔，以及来自`size_string()`的缩写表示法。

我们输出：

```cpp
/home/billw/working/cpp-stl-wkbk/chap10:
 327K bwgrep
   3K bwgrep.cpp
 199K dir
   4K dir.cpp
 176K formatter
 905B formatter.cpp
   0B include
   1K Makefile
 181K path-ops
   1K path-ops.cpp
 327K rerename
   2K rerename.cpp
11K testdir ![](img/6.png)
11K testdir-backup ![](img/6.png)
 203K working
   3K working.cpp
-------------------------
total bytes: 1,484,398 (1M)
```

## 它是如何工作的…

`fs::file_size()`函数返回一个`uintmax_t`值，它表示文件的大小，作为给定平台上的最大自然无符号整数。虽然这通常在大多数 64 位系统上是一个 64 位整数，但有一个值得注意的例外是 Windows，它使用 32 位整数。这意味着虽然在某些系统上`size_t`可能适用于此值，但在 Windows 上它无法编译，因为它可能尝试将 64 位值提升为 32 位值。

`entry_size()`函数接受一个`path`对象并返回一个`uintmax_t`值：

```cpp
uintmax_t entry_size(const fs::path& p) {
    if(fs::is_regular_file(p)) return fs::file_size(p);
    uintmax_t accum{};
    if(fs::is_directory(p) && !fs::is_symlink(p)) {
        for(auto& e : dit{ p }) {
            accum += entry_size(e.path());
        }
    }
    return accum;
}
```

函数检查是否为常规文件，并返回文件的大小。否则，它检查是否为既不是符号链接的目录。我们只想获取目录中文件的大小，所以不想跟随符号链接。（符号链接也可能导致引用循环，导致失控状态。）

如果我们找到一个目录，我们将遍历它，为遇到的每个文件调用`entry_size()`。这是一个递归循环，所以我们最终得到目录的大小。
