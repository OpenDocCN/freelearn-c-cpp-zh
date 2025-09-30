# *第七章*：字符串、流和格式化

STL 的 `string` 类是存储、操作和显示基于字符数据的一个强大、功能齐全的工具。它具有您在高级脚本语言中会发现的大部分便利性，同时仍然像您期望的那样快速敏捷。

`string` 类基于 `basic_string`，这是一个连续容器类，可以用任何字符类型实例化。其类签名如下：

```cpp
template<
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
> class basic_string;
```

`Traits` 和 `Allocator` 模板参数通常保留为默认值。

`basic_string` 的底层存储是一个连续的 `CharT` 序列，可以通过 `data()` 成员函数访问：

```cpp
const std::basic_string<char> s{"hello"};
const char * sdata = s.data();
for(size_t i{0}; i < s.size(); ++i) {
    cout << sdata[i] << ' ';
}
cout << '\n';
```

输出：

```cpp
h e l l o
```

`data()` 成员函数返回一个指向字符底层数组的 `CharT*`。自 C++11 以来，`data()` 返回的数组是空终止的，这使得 `data()` 等同于 `c_str()`。

`basic_string` 类包含了许多在其他连续存储类中可以找到的方法，包括 `insert()`、`erase()`、`push_back()`、`pop_back()` 以及其他方法。这些方法在 `CharT` 的底层数组上操作。

`std::string` 是 `std::basic_string<char>` 的类型别名：

```cpp
using std::string = std::basic_string<char>;
```

对于大多数用途，您将使用 `std::string`。

# 字符串格式化

字符串格式化一直是 STL 的弱点。直到最近，我们只能在不完美的选择之间做出选择，要么是笨拙的 STL `iostreams`，要么是过时的遗产 `printf()`。从 C++20 和 `format` 库开始，STL 字符串格式化终于成熟起来。新的 `format` 库紧密基于 Python 的 `str.format()` 方法，快速灵活，提供了 `iostreams` 和 `printf()` 的许多优点，以及良好的内存管理和类型安全。

更多关于 `format` 库的信息，请参阅 *第一章* 中的 *使用新的格式化库格式化文本* 菜谱，*新 C++20 功能*。

虽然我们不再需要使用 `iostreams` 进行字符串格式化，但它仍然在其他用途中非常有用，包括文件和流 I/O 以及一些类型转换。

在本章中，我们将涵盖以下主题以及更多内容：

+   将 `string_view` 用作轻量级字符串对象

+   连接字符串

+   转换字符串

+   使用 C++20 的 `format` 库格式化文本

+   从字符串中删除空白字符

+   从用户输入读取字符串

+   在文件中计算单词数

+   从文件输入初始化复杂结构

+   使用 `char_traits` 自定义字符串类

+   使用正则表达式解析字符串

# 技术要求

您可以在 GitHub 上找到本章的代码文件，地址为 [`github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap07`](https://github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap07)。

# 将 `string_view` 用作轻量级字符串对象

`string_view` 类为 `string` 类提供了一个轻量级的替代方案。它不是维护自己的数据存储，而是对 C 字符串的 *视图* 进行操作。这使得 `string_view` 比起 `std::string` 更小、更高效。在需要字符串对象但不需要 `std::string` 的更多内存和计算密集型功能的情况下，它非常有用。

## 如何做到这一点…

`string_view` 类看起来与 STL 的 `string` 类非常相似，但工作方式略有不同。让我们考虑一些例子：

+   这里是一个从 C 字符串（`char` 数组）初始化的 STL `string`：

    ```cpp
    char text[]{ "hello" };
    string greeting{ text };
    text[0] = 'J';
    cout << text << ' ' << greeting << '\n';
    ```

输出：

```cpp
Jello hello
```

注意，当我们修改数组时，`string` 并没有改变。这是因为 `string` 构造函数创建了底层数据的副本。

+   当我们用 `string_view` 做同样的事情时，我们得到不同的结果：

    ```cpp
    char text[]{ "hello" };
    string_view greeting{ text };
    text[0] = 'J';
    cout << text << ' ' << greeting << '\n';
    ```

输出：

```cpp
Jello Jello
```

`string_view` 构造函数创建底层数据的 *视图*。它不会创建自己的副本。这导致显著的效率，但也允许副作用。

+   由于 `string_view` 不会复制底层数据，源数据必须在 `string_view` 对象持续的时间内保持作用域。因此，这行不通：

    ```cpp
    string_view sv() {
        const char text[]{ "hello" };  // temporary storage
        string_view greeting{ text };
        return greeting;
    }
    int main() {
        string_view greeting = sv();  // data out of scope
        cout << greeting << '\n';  // output undefined
    }
    ```

由于底层数据在 `sv()` 函数返回后超出作用域，所以在使用它的时候，`main()` 中的 `greeting` 对象就不再有效了。

+   `string_view` 类具有适合底层数据的构造函数。这包括字符数组（`const char*`）、连续 *范围*（包括 `std::string`）和其他 `string_view` 对象。此示例使用 *范围* 构造函数：

    ```cpp
    string str{ "hello" };
    string_view greeting{ str };
    cout << greeting << '\n';
    ```

输出：

```cpp
hello
```

+   此外，还有一个 `string_view` 文字操作符 `sv`，它在 `std::literals` 命名空间中定义：

    ```cpp
    using namespace std::literals;
    cout << "hello"sv.substr(1, 4) << '\n';
    ```

这构建了一个 `constexpr string_view` 对象，并调用其方法 `substr()` 来获取从索引 `1` 开始的 `4` 个值。

输出：

```cpp
ello
```

## 它是如何工作的…

`string_view` 类实际上是一个连续字符序列的 *迭代器适配器*。其实现通常有两个成员：一个 `const CharT *` 和一个 `size_t`。它通过在源数据周围包装一个 `contiguous_iterator` 来工作。

这意味着你可以像 `std::string` 一样用于许多目的，但有一些重要的区别：

+   复制构造函数不会复制数据。这意味着当你复制一个 `string_view` 时，每个副本都操作相同的底层数据：

    ```cpp
    char text[]{ "hello" };
    string_view sv1{ text };
    string_view sv2{ sv1 };
    string_view sv3{ sv2 };
    string_view sv4{ sv3 };
    cout << format("{} {} {} {}\n", sv1, sv2, sv3, sv4);
    text[0] = 'J';
    cout << format("{} {} {} {}\n", sv1, sv2, sv3, sv4);
    ```

输出：

```cpp
hello hello hello hello
Jello Jello Jello Jello
```

+   请记住，当你将一个 `string_view` 传递给一个函数时，它使用复制构造函数：

    ```cpp
    void f(string_view sv) {
        if(sv.size()) {
            char* x = (char*)sv.data();  // dangerous
            x[0] = 'J';  // modifies the source
        }
        cout << format("f(sv): {} {}\n", (void*)sv.data(),      sv);
    }
    int main() {
        char text[]{ "hello" };
        string_view sv1{ text };
        cout << format("sv1: {} {}\n", (void*)sv1.data(),       sv1);
        f(sv1);
        cout << format("sv1: {} {}\n", (void*)sv1.data(),       sv1);
    }
    ```

输出：

```cpp
sv1: 0x7ffd80fa7b2a hello
f(sv): 0x7ffd80fa7b2a Jello
sv1: 0x7ffd80fa7b2a Jello
```

注意，底层数据的地址（由 `data()` 成员函数返回）对于所有 `string_view` 实例都是相同的。这是因为复制构造函数不会复制底层数据。尽管 `string_view` 成员指针是 `const`-修饰的，但仍然可以取消 `const` 修饰符，尽管这 *不推荐* 因为它可能会引起意外的副作用。但值得注意的是，数据永远不会被复制。

+   `string_view`类缺少直接操作底层字符串的方法。例如`append()`、`operator+()`、`push_back()`、`pop_back()`、`replace()`和`resize()`等，这些在`string`中支持的方法在`string_view`中不支持。

如果你需要使用`+`运算符连接字符串，你需要一个`std::string`。例如，这不能与`string_view`一起使用：

```cpp
sv1 = sv2 + sv3 + sv4; // does not work
```

你需要使用`string`：

```cpp
string str1{ text };
string str2{ str1 };
string str3{ str2 };
string str4{ str3 };
str1 = str2 + str3 + str4; // works
cout << str1 << '\n';
```

输出：

```cpp
JelloJelloJello
```

# 连接字符串

在 C++中连接字符串有几种方法。在这个菜谱中，我们将查看三种最常见的方法：`string`类的`operator+()`运算符、`string`类的`append()`函数和`ostringstream`类的`operator<<()`运算符。C++20 新引入的`format()`函数。每个都有其优点、缺点和使用场景。

## 如何做到这一点...

在这个菜谱中，我们将检查连接字符串的方法。然后我们将进行一些基准测试并考虑不同的使用场景。

+   我们将从一个`std::string`对象开始：

    ```cpp
    string a{ "a" };
    string b{ "b" };
    ```

`string`对象是由字面量 C 字符串构造的。

C 字符串构造函数复制字面量字符串，并使用本地副本作为`string`对象的底层数据。

+   现在，让我们构造一个新的空字符串对象，并使用分隔符和换行符将`a`和`b`连接起来：

    ```cpp
    string x{};
    x += a + ", " + b + "\n";
    cout << x;
    ```

在这里，我们使用了`string`对象的`+=`和`+`运算符来连接`a`和`b`字符串，以及字面量字符串`", "`和`"\n"`。结果字符串将元素连接在一起：

```cpp
a, b
```

+   我们可以使用`string`对象的`append()`成员函数：

    ```cpp
    string x{};
    x.append(a);
    x.append(", ");
    x.append(b);
    x.append("\n");
    cout << x;
    ```

这给我们带来了相同的结果：

```cpp
a, b
```

+   或者，我们可以构造一个`ostringstream`对象，它使用流接口：

    ```cpp
    ostringstream x{};
    x << a << ", " << b << "\n";
    cout << x.str();
    ```

我们得到相同的结果：

```cpp
a, b
```

+   我们还可以使用 C++20 的`format()`函数：

    ```cpp
    string x{};
    x = format("{}, {}\n", a, b);
    cout << x;
    ```

再次，我们得到相同的结果：

```cpp
a, b
```

## 它是如何工作的...

`string`对象有两种不同的方法来连接字符串，即`+`运算符和`append()`成员函数。

`append()`成员函数将数据添加到`string`对象数据的末尾。它必须分配和管理内存以完成此操作。

`+`运算符使用`operator+()`重载来构造一个新的`string`对象，该对象包含旧数据和新的数据，并返回新对象。

`ostringstream`对象的工作方式类似于`ostream`，但存储其输出以用作字符串。

C++20 的`format()`函数使用格式字符串和可变参数，并返回一个新构造的`string`对象。

## 还有更多...

你如何决定哪种连接策略适合你的代码？我们可以从一些基准测试开始。

### 基准测试

我使用 GCC 11 在 Debian Linux 上执行了这些测试：

+   首先，我们将使用`<chrono>`库创建一个`timer`函数：

    ```cpp
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    void timer(string(*f)()) {
        auto t1 = high_resolution_clock::now();
        string s{ f() };
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;
        cout << s;
        cout << format("duration: {} ms\n", ms.count());
    }
    ```

`timer`函数调用传递给它的函数，标记函数调用前后的时间。然后它使用`cout`显示持续时间。

+   现在，我们创建一个使用`append()`成员函数连接字符串的函数：

    ```cpp
    string append_string() {
        cout << "append_string\n";
        string a{ "a" };
        string b{ "b" };
        long n{0};
        while(++n) {
            string x{};
            x.append(a);
            x.append(", ");
            x.append(b);
            x.append("\n");
            if(n >= 10000000) return x;
        }
        return "error\n";
    }
    ```

为了基准测试的目的，这个函数重复进行了 1000 万次的连接操作。我们从`main()`函数中调用这个函数并使用`timer()`：

```cpp
int main() {
    timer(append_string);
}
```

我们得到以下输出：

```cpp
append_string
a, b
duration: 425.361643 ms
```

因此，在这个系统上，我们的连接操作进行了 1000 万次迭代，大约耗时 425 毫秒。

+   现在，让我们用`+`运算符重载创建相同的函数：

    ```cpp
    string concat_string() {
        cout << "concat_string\n";
        string a{ "a" };
        string b{ "b" };
        long n{0};
        while(++n) {
            string x{};
            x += a + ", " + b + "\n";
            if(n >= 10000000) return x;
        }
        return "error\n";
    }
    ```

我们的基准输出：

```cpp
concat_string
a, b
duration: 659.957702 ms
```

这个版本进行了 1000 万次迭代，大约耗时 660 毫秒。

+   现在，让我们用`ostringstream`来试一试：

    ```cpp
    string concat_ostringstream() {
        cout << "ostringstream\n";
        string a { "a" };
        string b { "b" };
        long n{0};
        while(++n) {
            ostringstream x{};
            x << a << ", " << b << "\n";
            if(n >= 10000000) return x.str();
        }
        return "error\n";
    }
    ```

我们的基准输出：

```cpp
ostringstream
a, b
duration: 3462.020587 ms
```

这个版本进行了 1000 万次迭代，大约耗时 3.5 秒。

+   这里是`format()`版本的示例：

    ```cpp
    string concat_format() {
        cout << "append_format\n";
        string a{ "a" };
        string b{ "b" };
        long n{0};
        while(++n) {
            string x{};
            x = format("{}, {}\n", a, b);
            if(n >= 10000000) return x;
        }
        return "error\n";
    }
    ```

我们的基准输出：

```cpp
append_format
a, b
duration: 782.800547 ms
```

`format()`版本进行了 1000 万次迭代，大约耗时 783 毫秒。

+   结果总结：

![连接性能比较](img/B18267_table_7.1.jpg)

连接性能比较

### 性能差异的原因是什么？

从这些基准测试中我们可以看出，`ostringstream`版本比基于`string`的版本慢很多倍。

`append()`方法比`+`运算符略快。它需要分配内存但不构造新对象。由于重复，可能存在一些优化。

`+`运算符重载可能调用`append()`方法。额外的函数调用可能会使其比`append()`方法逐渐慢。

`format()`版本创建了一个新的`string`对象，但没有`iostream`系统的开销。

`ostringstream`的`<<`运算符重载为每个操作创建一个新的`ostream`对象。考虑到流对象的复杂性以及管理流状态，这使得它比基于`string`的任何版本都要慢得多。

## 为什么我会选择其中一个而不是另一个？

个人的偏好将涉及一些度量。运算符重载（`+`或`<<`）可能是方便的。性能可能对你来说是一个问题，也可能不是。

`ostringstream`类相对于`string`方法有一个独特的优势：它为每种不同类型专门化了`<<`运算符，因此能够在可能存在不同类型调用相同代码的情况下操作。

`format()`函数提供了相同类型安全和定制选项，并且比`ostringstream`类快得多。

`string`对象的`+`运算符重载速度快，使用方便，易于阅读，但比`append()`方法逐渐慢。

`append()`版本最快，但需要为每个项目调用一个单独的函数。

对于我的目的，我更喜欢`format()`函数或`string`对象的`+`运算符，在大多数情况下。如果每个速度的比特都很重要，我会使用`append()`。如果需要`ostringstream`的独特功能和性能不是问题，我会使用它。

# 转换字符串

`std::string`类是一个*连续容器*，类似于`vector`或`array`。它支持`contiguous_iterator`概念和所有相应的算法。

`string`类是`basic_string`的一个特化，其类型为`char`。这意味着容器的元素是`char`类型。其他特化也可用，但`string`是最常见的。

因为它本质上是一个连续的`char`元素容器，所以`string`可以使用`transform()`算法，或者任何使用`contiguous_iterator`概念的技巧。

## 如何做到这一点…

根据应用的不同，有多种方式进行转换。本食谱将探讨其中的一些。

+   我们将从几个谓词函数开始。谓词函数接受一个转换元素并返回一个相关元素。例如，这里有一个简单的谓词，它返回一个大写字母：

    ```cpp
    char char_upper(const char& c) {
        return static_cast<char>(std::toupper(c));
    }
    ```

这个函数是`std::toupper()`的包装器。因为`toupper()`函数返回一个`int`，而`string`元素是`char`类型，所以我们不能直接在转换中使用`toupper()`函数。

这里是相应的`char_lower()`函数：

```cpp
char char_lower(const char& c) {
    return static_cast<char>(std::tolower(c));
}
```

+   `rot13()`函数是一个用于演示目的的有趣转换谓词。它是一个简单的替换密码，**不适用于加密**，但常用于**混淆**：

    ```cpp
    char rot13(const char& x) {
        auto rot13a = [](char x, char a)->char { 
            return a + (x - a + 13) % 26; 
        };
        if (x >= 'A' && x <= 'Z') return rot13a(x, 'A');
        if (x >= 'a' && x <= 'z') return rot13a(x, 'a');
        return x;
    }
    ```

+   我们可以使用这些谓词与`transform()`算法一起使用：

    ```cpp
    main() {
        string s{ "hello jimi\n" };
        cout << s;
        std::transform(s.begin(), s.end(), s.begin(), 
          char_upper);
        cout << s;
        ...
    ```

`transform()`函数对`s`的每个元素调用`char_upper()`，将结果放回`s`中，并将所有字符转换为大写：

输出：

```cpp
hello jimi
HELLO JIMI
```

+   除了`transform()`，我们还可以使用一个简单的**带有谓词函数的**`for`循环：

    ```cpp
    for(auto& c : s) c = rot13(c);
    cout << s;
    ```

从我们的大写字符串对象开始，结果是：

```cpp
URYYB WVZV
```

+   `rot13`密码的有趣之处在于它可以自己解密。因为 ASCII 字母表中有 26 个字母，旋转 13 次然后再旋转 13 次会得到原始字符串。让我们将字符串转换为小写并再次进行`rot13`转换以恢复我们的字符串：

    ```cpp
    for(auto& c : s) c = rot13(char_lower(c));
    cout << s;
    ```

输出：

```cpp
hello jimi
```

由于它们的接口统一，谓词函数可以作为彼此的参数进行**链式**调用。我们也可以使用`char_lower(rot13(c))`得到相同的结果。

+   如果你的需求对于简单的字符转换过于复杂，你可以像使用任何连续容器一样使用`string`迭代器。以下是一个简单的函数，它通过将第一个字符和每个跟在空格后面的字符大写，将小写字符串转换为**标题大小写**：

    ```cpp
    string& title_case(string& s) {
        auto begin = s.begin();
        auto end = s.end();
        *begin++ = char_upper(*begin);  // first element
        bool space_flag{ false };
        for(auto it{ begin }; it != end; ++it) {
            if(*it == ' ') {
                space_flag = true;
            } else {
                if(space_flag) *it = char_upper(*it);
                space_flag = false;
            }
        }
        return s;
    }
    ```

因为它返回转换后字符串的引用，我们可以像这样用`cout`调用它：

```cpp
cout << title_case(s);
```

输出：

```cpp
Hello Jimi
```

## 它是如何工作的…

`std::basic_string`类及其特化（包括`string`），都由完全符合`contiguous_iterator`的迭代器支持。这意味着任何适用于任何连续容器的技巧也适用于`string`。

注意

这些转换不会与`string_view`对象一起工作，因为底层数据是`const`修饰的。

# 使用 C++20 的格式库格式化文本

C++20 引入了新的 `format()` 函数，该函数返回其参数的格式化字符串表示。`format()` 使用 Python 风格的格式化字符串，具有简洁的语法、类型安全和优秀的性能。

`format()` 函数接受一个格式字符串和一个模板，即 *参数包* 作为其参数：

```cpp
template< class... Args >
string format(const string_view fmt, Args&&... args );
```

格式字符串使用花括号 `{}` 作为格式化参数的占位符：

```cpp
const int a{47};
format("a is {}\n", a);
```

输出：

```cpp
a is 47
```

它也使用花括号作为格式说明符，例如：

```cpp
format("Hex: {:x} Octal: {:o} Decimal {:d} \n", a, a, a);
```

输出：

```cpp
Hex: 2f Octal: 57 Decimal 47
```

这个配方将向您展示如何使用 `format()` 函数来实现一些常见的字符串格式化解决方案。

注意

这章是在 Windows 10 上使用 Microsoft Visual C++ 编译器的预览版开发的。在撰写本文时，这是唯一完全支持 C++20 `<format>` 库的编译器。最终实现可能在某些细节上有所不同。

## 如何做到这一点…

让我们考虑一些使用 `format()` 函数的常见格式化解决方案：

+   我们将从一些需要格式化的变量开始：

    ```cpp
    const int inta{ 47 };
    const char * human{ "earthlings" };
    const string_view alien{ "vulcans" };
    const double df_pi{ pi };
    ```

`pi` 常量在 `<numbers>` 头文件和 `std::numbers` 命名空间中。

+   我们可以使用 `cout` 来显示变量：

    ```cpp
    cout << "inta is " << inta << '\n'
        << "hello, " << human << '\n'
        << "All " << alien << " are welcome here\n"
        << "π is " << df_pi << '\n';
    ```

我们得到以下输出：

```cpp
a is 47
hello, earthlings
All vulcans are welcome here
π is 3.14159
```

+   现在，让我们用 `format()` 来查看这些内容，从 C-string 的 `human` 开始：

    ```cpp
    cout << format("Hello {}\n", human);
    ```

这是 `format()` 函数的最简单形式。格式字符串有一个占位符 `{}` 和一个相应的变量 `human`。输出如下：

```cpp
Hello earthlings
```

+   `format()` 函数返回一个字符串，我们使用 `cout <<` 来显示这个字符串。

`format()` 库的原版提案包括一个 `print()` 函数，它使用与 `format()` 相同的参数。这将允许我们一步打印我们的格式化字符串：

```cpp
print("Hello {}\n", cstr);
```

不幸的是，`print()` 没有被纳入 C++20 标准，尽管它预计将在 C++23 中被包含。

我们可以使用一个简单的函数，通过 `vformat()` 来提供相同的功能：

```cpp
template<typename... Args>
constexpr void print(const string_view str_fmt, 
                     Args&&... args) {
    fputs(std::vformat(str_fmt, 
          std::make_format_args(args...)).c_str(), 
          stdout);
}
```

这个简单的单行函数为我们提供了一个可用的 `print()` 函数。我们可以用它来代替 `cout << format()` 组合：

```cpp
print("Hello {}\n", human);
```

输出：

```cpp
Hello earthlings
```

在示例文件的 `include` 目录中可以找到这个函数的更完整版本。

+   格式字符串还提供了位置选项：

    ```cpp
    print("Hello {} we are {}\n", human, alien);
    ```

输出：

```cpp
Hello earthlings we are vulcans
```

我们可以通过在格式字符串中使用位置选项来改变参数的顺序：

```cpp
print("Hello {1} we are {0}\n", human, alien);
```

现在，我们得到以下输出：

```cpp
Hello vulcans we are earthlings
```

注意，参数保持不变。只有花括号中的位置值发生了变化。位置索引是从零开始的，就像 `[]` 操作符一样。

这个特性对于国际化很有用，因为不同的语言在句子中不同词性的顺序不同。

+   数字有许多格式化选项：

    ```cpp
    print("π is {}\n", df_pi);
    ```

输出：

```cpp
π is 3.141592653589793
```

我们可以指定精度的位数：

```cpp
print("π is {:.5}\n", df_pi);
```

输出：

```cpp
π is 3.1416
```

冒号字符 `:` 用于分隔位置索引和格式化参数：

```cpp
print("inta is {1:}, π is {0:.5}\n", df_pi, inta);
```

输出：

```cpp
inta is 47, π is 3.1416
```

+   如果我们想让一个值占据一定数量的空间，我们可以指定字符数，如下所示：

    ```cpp
    print("inta is [{:10}]\n", inta);
    ```

输出：

```cpp
inta is [        47]
```

我们可以将其左对齐或右对齐：

```cpp
print("inta is [{:<10}]\n", inta);
print("inta is [{:>10}]\n", inta);
```

输出：

```cpp
inta is [47        ]
inta is [        47]
```

默认情况下，它用空格字符填充，但我们可以更改它：

```cpp
print("inta is [{:*<10}]\n", inta);
print("inta is [{:0>10}]\n", inta);
```

输出：

```cpp
inta is [47********]
inta is [0000000047]
```

我们还可以居中一个值：

```cpp
print("inta is [{:¹⁰}]\n", inta);
print("inta is [{:_¹⁰}]\n", inta);
```

输出：

```cpp
inta is [    47    ]
inta is [____47____]
```

+   我们可以将整数格式化为十六进制、八进制或默认的十进制表示：

    ```cpp
    print("{:>8}: [{:04x}]\n", "Hex", inta);
    print("{:>8}: [{:4o}]\n", "Octal", inta);
    print("{:>8}: [{:4d}]\n", "Decimal", inta);
    ```

输出：

```cpp
     Hex: [002f]
   Octal: [  57]
 Decimal: [  47]
```

注意，我使用了右对齐来对齐标签。

使用大写 `X` 表示大写十六进制：

```cpp
print("{:>8}: [{:04X}]\n", "Hex", inta);
```

输出：

```cpp
     Hex: [002F]
```

小贴士

默认情况下，Windows 使用不常见的字符编码。最新版本可能默认为 UTF-16 或 UTF-8 BOM。较旧版本可能默认为 "代码页" 1252，它是 ISO 8859-1 ASCII 标准的超集。没有 Windows 系统默认使用更常见的 UTF-8（无 BOM）。

默认情况下，Windows 不会显示标准的 UTF-8 `π` 字符。为了使 Windows 与 UTF-8 编码（以及世界上的其他部分）兼容，使用编译器开关 `/utf-8` 并在命令行上执行 `chcp 65001` 命令进行测试。现在，你可以拥有你的 `π` 并享用它。

## 它是如何工作的…

`<format>` 库使用模板 *参数包* 将参数传递给格式化器。这允许单独检查参数的类和类型。库函数 `make_format_args()` 接收一个参数包并返回一个 `format_args` 对象，该对象提供了一个要格式化的 *类型擦除* 参数列表。

我们可以在 `print()` 函数中看到这一点：

```cpp
template<typename... Args>
constexpr void print(const string_view str_fmt, Args&&... args) {
    fputs(vformat(str_fmt, 
      make_format_args(args...)).c_str(), 
          stdout);
}
```

`make_format_args()` 函数接收一个参数包并返回一个 `format_args` 对象。`vformat()` 函数接收一个格式字符串和 `format_args` 对象，并返回一个 `std::string`。我们使用 `c_str()` 方法获取用于 `fputs()` 的 C 字符串。

## 还有更多…

对于自定义类，通常的做法是重载 `ostream` 的 `<<` 操作符。例如，给定一个包含分数值的 `Frac` 类：

```cpp
template<typename T>
struct Frac {
    T n;
    T d;
};
...
Frac<long> n{ 3, 5 };
cout << "Frac: " << n << '\n';
```

我们希望将对象打印成分数形式，例如 `3/5`。因此，我们会编写一个简单的 `operator<<` 特化如下：

```cpp
template <typename T>
std::ostream& operator<<(std::ostream& os, const Frac<T>& f) {
    os << f.n << '/' << f.d;
    return os;
}
```

现在的输出是：

```cpp
Frac: 3/5
```

为了为我们自定义的类提供 `format()` 支持，我们需要创建一个 `formatter` 对象特化，如下所示：

```cpp
template <typename T>
struct std::formatter<Frac<T>> : std::formatter<unsigned> {
    template <typename Context>
    auto format(const Frac<T>& f, Context& ctx) const {
        return format_to(ctx.out(), "{}/{}", f.n, f.d);
    }
};
```

`std::formatter` 类的特化重载了其 `format()` 方法。为了简单起见，我们继承自 `formatter<unsigned>` 特化。`format()` 方法使用一个 `Context` 对象调用，该对象提供了格式化字符串的输出上下文。对于返回值，我们使用 `format_to()` 函数与 `ctx.out`、一个普通格式字符串和参数。

现在，我们可以使用 `print()` 函数和 `Frac` 类：

```cpp
print("Frac: {}\n", n);
```

格式化器现在识别我们的类并提供了我们期望的输出：

```cpp
Frac: 3/5
```

# 从字符串中修剪空白

用户输入通常会在字符串的一端或两端包含多余的空白。这可能会引起问题，因此我们通常需要删除它。在这个菜谱中，我们将使用 `string` 类的 `find_first_not_of()` 和 `find_last_not_of()` 方法来修剪字符串的端部空白。

## 如何做到这一点…

`string` 类包含用于查找是否包含在字符列表中的元素的方法。我们将使用这些方法来修剪 `string`：

+   我们首先使用来自一个假设的多指用户的输入来定义 `string`：

    ```cpp
    int main() {
        string s{" \t  ten-thumbed input   \t   \n \t "};
        cout << format("[{}]\n", s);
        ...
    ```

我们的内容前后有一些额外的制表符 `\t` 和换行符 `\n` 字符。我们用括号包围它来显示空白：

```cpp
[       ten-thumbed input
      ]
```

+   这里有一个 `trimstr()` 函数，用于从 `string` 的两端删除所有空白字符：

    ```cpp
    string trimstr(const string& s) {
        constexpr const char * whitespace{ " \t\r\n\v\f" };
        if(s.empty()) return s;
        const auto first{ s.find_first_not_of(whitespace) };
        if(first == string::npos) return {};
        const auto last{ s.find_last_not_of(whitespace) };
        return s.substr(first, (last - first + 1));
    }
    ```

我们定义了我们的一组空白字符为 *空格*、*制表符*、*回车*、*换行符*、*垂直制表符* 和 *换页符*。其中一些比其他更常见，但这是规范集合。

此函数使用 `string` 类的 `find_first_not_of()` 和 `find_last_not_of()` 方法来查找第一个/最后一个不是集合成员的元素。

+   现在，我们可以调用该函数来去除所有那些不请自来的空白字符：

    ```cpp
    cout << format("[{}]\n", trimstr(s));
    ```

输出：

```cpp
[ten-thumbed input]
```

## 它是如何工作的…

`string` 类的各个 `find...()` 成员函数返回一个 `size_t` 类型的位置：

```cpp
size_t find_first_not_of( const CharT* s, size_type pos = 0 );
size_t find_last_not_of( const CharT* s, size_type pos = 0 );
```

返回值是第一个匹配字符（不在 `s` 字符列表中）的零基于位置，或者如果没有找到，则返回特殊值，`string::npos`。`npos` 是一个静态成员常量，表示一个无效的位置。

我们测试 `(first == string::npos)`，如果没有匹配，则返回空字符串 `{}`。否则，我们使用 `first` 和 `last` 位置与 `s.substr()` 方法一起返回没有空白的字符串。

# 从用户输入读取字符串

STL 使用 `std::cin` 对象从标准输入流提供基于字符的输入。`cin` 对象是一个全局 *单例*，它将输入从控制台作为 `istream` 输入流读取。

默认情况下，`cin` 一次读取 *一个单词*，直到达到流的末尾：

```cpp
string word{};
cout << "Enter words: ";
while(cin >> word) {
    cout << format("[{}] ", word);
}
cout << '\n';
```

输出：

```cpp
$ ./working
Enter words: big light in sky
[big] [light] [in] [sky]
```

这有限的使用价值，并且可能会导致一些人将 `cin` 视为功能最小化。

虽然 `cin` 确实有其怪癖，但它可以轻松地被整理成提供面向行的输入。

## 如何做到这一点…

要从 `cin` 获取基本的面向行的功能，需要了解两个重要的行为。一个是能够一次获取一行，而不是一次一个单词。另一个是在错误条件下重置流的能力。让我们详细看看这些：

+   首先，我们需要提示用户输入。这里有一个简单的 `prompt` 函数：

    ```cpp
    bool prompt(const string_view s, const string_view s2 = "") {
        if(s2.size()) cout << format("{} ({}): ", s, s2);
        else cout << format("{}: ", s);
        cout.flush();
        return true;
    }
    ```

`cout.flush()` 函数调用确保输出立即显示。有时，当输出不包含换行符时，输出流可能不会自动刷新。

+   `cin` 类有一个 `getline()` 方法，可以从输入流获取一行文本并将其放入 C-string 数组中：

    ```cpp
    constexpr size_t MAXLINE{1024 * 10};
    char s[MAXLINE]{};
    const char * p1{ "Words here" };
    prompt(p1);
    cin.getline(s, MAXLINE, '\n');
    cout << s << '\n';
    ```

输出：

```cpp
Words here: big light in sky![](img/1.png)
big light in sky
```

`cin.getline()` 方法接受三个参数：

```cpp
getline(char* s, size_t count, char delim );
```

第一个参数是目标 C-string 数组。第二个是数组的大小。第三个是行结束的分隔符。

函数将不会在数组中放置超过 `count`-1 个字符，为 *空字符* 终止符留出空间。

分隔符默认为换行符 `'\n'`。

+   STL 还提供了一个独立的 `getline()` 函数，它可以与 STL `string` 对象一起使用：

    ```cpp
    string line{};
    const char * p1a{ "More words here" };
    prompt(p1a, "p1a");
    getline(cin, line, '\n');
    cout << line << '\n';
    ```

输出：

```cpp
$ ./working
More words here (p1a): slated to appear in east![](img/1.png)
slated to appear in east
```

独立的 `std::getline()` 函数接受三个参数：

```cpp
getline(basic_istream&& in, string& str, char delim );
```

第一个参数是输出流，第二个参数是 `string` 对象的引用，第三个是行结束符。

如果未指定，分隔符默认为换行符 `'\n'`。

我发现独立的 `getline()` 比使用 `cin.getline()` 方法更方便。

+   我们可以使用 `cin` 从输入流中获取特定类型。为了做到这一点，我们必须能够处理错误条件。

当 `cin` 遇到错误时，它会将流设置为错误条件并停止接受输入。为了在错误后重试输入，我们必须重置流的状态。这里有一个在错误后重置输入流的函数：

```cpp
void clearistream() {
    string s{};
    cin.clear();
    getline(cin, s);
}
```

`cin.clear()` 函数重置输入流的错误标志，但留下缓冲区中的文本。然后我们通过读取一行并丢弃它来清除缓冲区。

+   我们可以通过使用 `cin` 和数值类型变量来接受数值输入：

    ```cpp
    double a{};
    double b{};
    const char * p2{ "Please enter two numbers" };
    for(prompt(p2); !(cin >> a >> b); prompt(p2)) {
        cout << "not numeric\n";
        clearistream();
    }
    cout << format("You entered {} and {}\n", a, b);
    ```

输出：

```cpp
$ ./working
Please enter two numbers: a b![](img/1.png)
not numeric
Please enter two numbers: 47 73![](img/1.png)
You entered 47 and 73
```

`cin >> a >> b` 表达式从控制台接受输入，并尝试将前两个单词转换为与 `a` 和 `b` (`double`) 兼容的类型。如果失败，我们调用 `clearistream()` 并再次尝试。

+   我们可以使用 `getline()` 的分隔符参数来获取逗号分隔的输入：

    ```cpp
    line.clear();
    prompt(p3);
    while(line.empty()) getline(cin, line);
    stringstream ss(line);
    while(getline(ss, word, ',')) {
        if(word.empty()) continue;
        cout << format("word: [{}]\n", trimstr(word));
    }
    ```

输出：

```cpp
$ ./working
Comma-separated words: this, that, other
word: [this]
word: [that]
word: [other]
```

因为这段代码在数字代码之后运行，并且因为 `cin` 输入混乱，缓冲区中可能仍然存在一个行结束符。`while(line.empty())` 循环将可选地吃掉任何空行。

我们使用 `stringstream` 对象来处理单词，因此我们不必使用 `cin` 来做。这允许我们使用 `getline()` 获取一行，而无需等待文件结束状态。

然后，我们在 `stringstream` 对象上调用 `getline()` 来解析出由逗号分隔的单词。这给我们单词，但带有前导空白。我们使用本章中 *从字符串中删除空白* 食谱中的 `trimstr()` 函数来删除空白。

## 它是如何工作的…

`std::cin` 对象比它看起来更有用，但使用它可能是一个挑战。它倾向于在流中留下行结束符，并且在错误的情况下，它可能会忽略输入。

解决方案是使用 `getline()`，并在必要时将行放入 `stringstream` 中以便方便解析。

# 在文件中计数单词

默认情况下，`basic_istream` 类一次读取一个单词。我们可以利用这个特性来使用 `istream_iterator` 来计数单词。

## 如何做到这一点…

这是一个简单的使用 `istream_iterator` 来计数单词的食谱：

+   我们将从使用 `istream_iterator` 对象来计数单词的简单函数开始：

    ```cpp
    size_t wordcount(auto& is) {
        using it_t = istream_iterator<string>;
        return distance(it_t{is}, it_t{});
    }
    ```

`distance()` 函数接受两个迭代器并返回它们之间的步骤数。`using` 语句为具有 `string` 特化的 `istream_iterator` 类创建了一个别名 `it_t`。然后我们使用一个初始化为输入流 `it_t{is}` 的迭代器和另一个使用默认构造函数的迭代器调用 `distance()`，后者给出了流结束的哨兵。

+   我们在 `main()` 函数中调用 `wordcount()`：

    ```cpp
    int main() {
        const char * fn{ "the-raven.txt" };
        std::ifstream infile{fn, std::ios_base::in};
        size_t wc{ wordcount(infile) };
        cout << format("There are {} words in the 
          file.\n", wc);
    }
    ```

这调用 `wordcount()` 并打印文件中的单词数。当我用埃德加·爱伦·坡的 *The Raven* 的文本调用它时，我们得到以下输出：

```cpp
There are 1068 words in the file.
```

## 它是如何工作的…

由于 `basic_istream` 默认按单词输入，文件中的步骤数将是单词数。`distance()` 函数将测量两个迭代器之间的步骤数，因此使用起始迭代器和兼容对象的哨兵调用它将计算文件中的单词数。

# 从文件输入初始化复杂结构

*输入流* 的一项优点是它能够从文本文件中解析不同类型的数据并将它们转换为相应的基本类型。这里有一个简单的技术，使用输入流将数据导入结构体的容器中。

## 如何做到这一点…

在这个菜谱中，我们将从一个数据文件中导入其不同的字段到 `struct` 对象的 `vector` 中。数据文件表示城市及其人口和地图坐标：

+   这是 `cities.txt`，我们将要读取的数据文件：

    ```cpp
    Las Vegas
    661903 36.1699 -115.1398
    New York City
    8850000 40.7128 -74.0060
    Berlin
    3571000 52.5200 13.4050
    Mexico City
    21900000 19.4326 -99.1332
    Sydney
    5312000 -33.8688 151.2093
    ```

城市名称独占一行。第二行是人口，后面跟着经度和纬度。这种模式为五个城市中的每一个重复。

+   我们将在一个常量中定义我们的文件名，这样我们就可以稍后打开它：

    ```cpp
    constexpr const char * fn{ "cities.txt" };
    ```

+   这是一个用于存储数据的 `City` 结构体：

    ```cpp
    struct City {
        string name;
        unsigned long population;
        double latitude;
        double longitude;
    };
    ```

+   我们希望读取文件并将 `City` 对象的 `vector` 填充：

    ```cpp
    vector<City> cities;
    ```

+   这就是输入流使这变得简单的地方。我们可以简单地像这样为我们的 `City` 类特化 `operator>>`：

    ```cpp
    std::istream& operator>>(std::istream& in, City& c) {
        in >> std::ws;
        std::getline(in, c.name);
        in >> c.population >> c.latitude >> c.longitude;
        return in;
    }
    ```

`std::ws` 输入操纵符会从输入流中丢弃前导空白字符。

我们使用 `getline()` 读取城市名称，因为它可能是一个或多个单词。

这利用了 `>>` 操作符为 `population`（`unsigned long`）、`latitude` 和 `longitude`（都是 `double`）元素填充正确的类型。

+   现在，我们可以打开文件并使用 `>>` 操作符直接将文件读取到 `City` 对象的 `vector` 中：

    ```cpp
    ifstream infile(fn, std::ios_base::in);
    if(!infile.is_open()) {
        cout << format("failed to open file {}\n", fn);
        return 1;
    }
    for(City c{}; infile >> c;) cities.emplace_back(c);
    ```

+   我们可以使用 `format()` 显示这个向量：

    ```cpp
    for (const auto& [name, pop, lat, lon] : cities) {
        cout << format("{:.<15} pop {:<10} coords {}, {}\n", 
            name, make_commas(pop), lat, lon);
    }
    ```

输出：

```cpp
$ ./initialize_container < cities.txt
Las Vegas...... pop 661,903    coords 36.1699, -115.1398
New York City.. pop 8,850,000  coords 40.7128, -74.006
Berlin......... pop 3,571,000  coords 52.52, 13.405
Mexico City.... pop 21,900,000 coords 19.4326, -99.1332
Sydney......... pop 5,312,000  coords -33.8688, 151.2093
```

+   `make_commas()` 函数也用于 *使用结构化绑定返回多个值* 菜谱中的 *第二章*，*通用 STL 功能*。它接受一个数值并返回一个 `string` 对象，其中添加了逗号以提高可读性：

    ```cpp
    string make_commas(const unsigned long num) {
        string s{ std::to_string(num) };
        for(int l = s.length() - 3; l > 0; l -= 3) {
            s.insert(l, ",");
        }
        return s;
    }
    ```

## 它是如何工作的…

这个菜谱的核心是 `istream` 类的 `operator>>` 重载：

```cpp
std::istream& operator>>(std::istream& in, City& c) {
    in >> std::ws;
    std::getline(in, c.name);
    in >> c.population >> c.latitude >> c.longitude;
    return in;
}
```

通过在函数头中指定我们的 `City` 类，每当一个 `City` 对象出现在输入流 `>>` 操作符的右侧时，这个函数就会被调用：

```cpp
City c{};
infile >> c;
```

这允许我们精确指定输入流如何将数据读入 `City` 对象。

## 更多...

当你在 Windows 系统上运行此代码时，你会注意到第一行的第一个单词被破坏。这是因为 Windows 总是在任何 UTF-8 文件的开头包含一个 **字节顺序标记**（**BOM**）。所以，当你读取 Windows 上的文件时，BOM 将包含在你读取的第一个对象中。BOM 是过时的，但在写作的时候，没有方法可以阻止 Windows 使用它。

解决方案是调用一个函数来检查文件的前三个字节是否为 BOM。UTF-8 的 BOM 是 `EF BB BF`。以下是一个搜索并跳过 UTF-8 BOM 的函数：

```cpp
// skip BOM for UTF-8 on Windows
void skip_bom(auto& fs) {
    const unsigned char boms[]{ 0xef, 0xbb, 0xbf };
    bool have_bom{ true };
    for(const auto& c : boms) {
        if((unsigned char)fs.get() != c) have_bom = false; 
    }
    if(!have_bom) fs.seekg(0);
    return;
}
```

这个函数读取文件的前三个字节并检查它们是否为 UTF-8 BOM 签名。如果三个字节中的任何一个不匹配，它将输入流重置为文件的开头。如果文件没有 BOM，则不会造成任何损害。

你只需在开始读取文件之前调用此函数：

```cpp
int main() {
    ...
    ifstream infile(fn, std::ios_base::in);
    if(!infile.is_open()) {
        cout << format("failed to open file {}\n", fn);
        return 1;
    }
    skip_bom(infile);
    for(City c{}; infile >> c;) cities.emplace_back(c);
    ...
}
```

这将确保 BOM 不会包含在文件的第一行字符串中。

注意

因为 `cin` 输入流不可定位，所以 `skip_bom()` 函数在 `cin` 流上不会工作。它只能与可定位的文本文件一起工作。

# 使用 char_traits 自定义字符串类

`string` 类是 `basic_string` 类的别名，其签名为：

```cpp
class basic_string<char, std::char_traits<char>>;
```

第一个模板参数提供了字符类型。第二个模板参数提供了一个字符 traits 类，它为指定的字符类型提供基本的字符和字符串操作。我们通常使用默认的 `char_traits<char>` 类。

我们可以通过提供我们自己的自定义字符 traits 类来修改字符串的行为。

## 如何实现...

在这个菜谱中，我们将创建一个用于 `basic_string` 的 *字符 traits 类*，该类在比较时将忽略大小写：

+   首先，我们需要一个函数将字符转换为通用的大小写。这里我们将使用小写，但这是一个任意的选择。大写也可以工作：

    ```cpp
    constexpr char char_lower(const char& c) {
        if(c >= 'A' && c <= 'Z') return c + ('a' - 'A');
        else return c;
    }
    ```

这个函数必须是 `constexpr`（对于 C++20 及以后的版本），所以现有的 `std::tolower()` 函数在这里不会工作。幸运的是，这是一个简单问题的简单解决方案。

+   我们的 traits 类称为 `ci_traits`（*ci* 代表不区分大小写）。它继承自 `std::char_traits<char>`：

    ```cpp
    class ci_traits : public std::char_traits<char> {
    public:
        ...
    };
    ```

继承允许我们仅覆盖我们需要的函数。

+   比较函数分别称为 `lt()`（小于）和 `eq()`（等于）：

    ```cpp
    static constexpr bool lt(char_type a, char_type b) noexcept {
        return char_lower(a) < char_lower(b);
    }
    static constexpr bool eq(char_type a, char_type b) noexcept {
        return char_lower(a) == char_lower(b);
    }
    ```

注意到我们比较的是字符的小写版本。

+   还有一个 `compare()` 函数，它比较两个 C-字符串。它返回 `+1` 表示大于，`-1` 表示小于，`0` 表示等于。我们可以使用 spaceship `<=>` 运算符来完成这个操作：

    ```cpp
    static constexpr int compare(const char_type* s1, 
            const char_type* s2, size_t count) {
        for(size_t i{0}; i < count; ++i) {
            auto diff{ char_lower(s1[i]) <=> 
              char_lower(s2[i]) };
            if(diff > 0) return 1;
            if(diff < 0) return -1;
            }
        return 0;
    }
    ```

+   最后，我们需要实现一个 `find()` 函数。它返回找到的第一个字符实例的指针，如果没有找到则返回 `nullptr`：

    ```cpp
    static constexpr const char_type* find(const char_type* p, 
            size_t count, const char_type& ch) {
        const char_type find_c{ char_lower(ch) };
        for(size_t i{0}; i < count; ++i) {
            if(find_c == char_lower(p[i])) return p + i;
        }
        return nullptr;
    }
    ```

+   现在我们有了 `ci_traits` 类，我们可以为我们的 `string` 类定义一个别名：

    ```cpp
    using ci_string = std::basic_string<char, ci_traits>;
    ```

+   在我们的 `main()` 函数中，我们定义了一个 `string` 和一个 `ci_string`：

    ```cpp
    int main() {
        string s{"Foo Bar Baz"};
        ci_string ci_s{"Foo Bar Baz"};
        ...
    ```

+   我们想使用 `cout` 打印它们，但这不会工作：

    ```cpp
    cout << "string: " << s << '\n';
    cout << "ci_string: " << ci_s << '\n';
    ```

首先，我们需要为 `operator<<` 重载一个操作符：

```cpp
std::ostream& operator<<(std::ostream& os, 
        const ci_string& str) {
    return os << str.c_str();
}
```

现在，我们得到以下输出：

```cpp
string: Foo Bar Baz
ci_string: Foo Bar Baz
```

+   让我们比较两个不同大小写的 `ci_string` 对象：

    ```cpp
    ci_string compare1{"CoMpArE StRiNg"};
    ci_string compare2{"compare string"};
    if (compare1 == compare2) {
        cout << format("Match! {} == {}\n", compare1, 
          compare2);
    } else {
        cout << format("no match {} != {}\n", compare1, 
          compare2);
    }
    ```

输出：

```cpp
Match! CoMpArE StRiNg == compare string
```

比较按预期工作。

+   在 `ci_s` 对象上使用 `find()` 函数，我们搜索小写的 `b` 并找到一个大写的 `B`：

    ```cpp
    size_t found = ci_s.find('b');
    cout << format("found: pos {} char {}\n", found, ci_s[found]);
    ```

输出：

```cpp
found: pos 4 char B
```

注意

注意，`format()` 函数不需要特化。这已经在 `fmt.dev` 参考实现中进行了测试。即使在特化的情况下，它也没有在 MSVC 的预览版 `format()` 中工作。希望这将在未来的版本中得到修复。

## 它是如何工作的…

这个配方通过在 `string` 类的模板特化中用我们自己的 `ci_traits` 类替换 `std::char_traits` 类来实现。`basic_string` 类使用特性类为其基本字符特定功能，如比较和搜索。当我们用我们自己的类替换它时，我们可以改变这些基本行为。

## 还有更多…

我们还可以重写 `assign()` 和 `copy()` 成员函数来创建一个存储小写字符的类：

```cpp
class lc_traits : public std::char_traits<char> {
public:
    static constexpr void assign( char_type& r, const
      char_type& a )
            noexcept {
        r = char_lower(a);
    }
    static constexpr char_type* assign( char_type* p,
            std::size_t count, char_type a ) {
        for(size_t i{}; i < count; ++i) p[i] = 
          char_lower(a);
        return p;
    }
    static constexpr char_type* copy(char_type* dest, 
            const char_type* src, size_t count) {
        for(size_t i{0}; i < count; ++i) {
            dest[i] = char_lower(src[i]);
        }
        return dest;
    }
};
```

现在，我们可以创建一个 `lc_string` 别名，并且对象存储小写字符：

```cpp
using lc_string = std::basic_string<char, lc_traits>;
...
lc_string lc_s{"Foo Bar Baz"};
cout << "lc_string: " << lc_s << '\n';
```

输出：

```cpp
lc_string: foo bar baz
```

注意

这些技术在 GCC 和 Clang 上按预期工作，但在 MSVC 的预览版上不起作用。我预计这将在未来的版本中得到修复。

# 使用正则表达式解析字符串

*正则表达式*（通常缩写为 *regex*）常用于文本流中的词法分析和模式匹配。它们在 Unix 文本处理工具中很常见，如 `grep`、`awk` 和 `sed`，并且是 *Perl* 语言的一个组成部分。在语法中存在一些常见的变体。1992 年批准了一个 POSIX 标准，而其他常见的变体包括 *Perl* 和 *ECMAScript*（JavaScript）方言。C++ 的 `regex` 库默认使用 ECMAScript 方言。

`regex` 库首次在 C++11 中引入到 STL 中。它对于在文本文件中查找模式非常有用。

要了解更多关于正则表达式语法和用法的信息，我推荐阅读 Jeffrey Friedl 的书籍，*Mastering Regular Expressions*。

## 如何做到这一点…

对于这个配方，我们将从 HTML 文件中提取超链接。超链接在 HTML 中的编码如下：

```cpp
<a href="http://example.com/file.html">Text goes here</a>
```

我们将使用一个 `regex` 对象来提取链接和文本，作为两个单独的字符串。

+   我们的示例文件名为 `the-end.html`。它来自我的网站 ([`bw.org/end/`](https://bw.org/end/))，并包含在 GitHub 仓库中：

    ```cpp
    const char * fn{ "the-end.html" };
    ```

+   现在，我们定义我们的 `regex` 对象，并使用正则表达式字符串：

    ```cpp
    const std::regex 
        link_re{ "<a href=\"([^\"]*)\"[^<]*>([^<]*)</a>" };
    ```

正则表达式一开始可能看起来很吓人，但实际上相当简单。

这被解析如下：

1.  匹配整个字符串。

1.  找到子串 `<a href="`.

1.  将直到下一个 `"` 的所有内容存储为子匹配 `1`。

1.  跳过 `>` 字符。

1.  将直到字符串 `</a>` 的所有内容存储为子匹配 `2`。

+   现在，我们将整个文件读入一个字符串中：

    ```cpp
    string in{};
    std::ifstream infile(fn, std::ios_base::in);
    for(string line{}; getline(infile, line);) in += line;
    ```

这将打开 HTML 文件，逐行读取它，并将每一行追加到 `string` 对象 `in` 中。

+   为了提取链接字符串，我们设置一个 `sregex_token_iterator` 对象来遍历文件并提取每个匹配的元素：

    ```cpp
    std::sregex_token_iterator it{ in.begin(), in.end(),
        link_re, {1, 2} };
    ```

`1` 和 `2` 对应于正则表达式中的子匹配。

+   我们有一个相应的函数来使用迭代器遍历结果：

    ```cpp
    template<typename It>
    void get_links(It it) {
        for(It end_it{}; it != end_it; ) {
            const string link{ *it++ };
            if(it == end_it) break;
            const string desc{ *it++ };
            cout << format("{:.<24} {}\n", desc, link);
        }
    }
    ```

我们用 `regex` 迭代器调用该函数：

```cpp
get_links(it);
```

我们用描述和链接得到这个结果：

```cpp
Bill Weinman............ https://bw.org/
courses................. https://bw.org/courses/
music................... https://bw.org/music/
books................... https://packt.com/
back to the internet.... https://duckduckgo.com/
```

## 它是如何工作的…

STL 的 `regex` 引擎作为一个 *生成器* 运行，每次评估并产生一个结果。我们使用 `sregex_iterator` 或 `sregex_token_iterator` 设置迭代器。虽然 `sregex_token_iterator` 支持子匹配，但 `sregex_iterator` 不支持。

我们正则表达式中的括号作为 *子匹配*，分别编号为 `1` 和 `2`：

```cpp
const regex link_re{ "<a href=\"([^\"]*)\"[^<]*>([^<]*)</a>" };
```

这里展示了 `regex` 匹配的每一部分：

![图 7.1 – 带有子匹配的正则表达式![img/B18267_07_01.jpg](img/B18267_07_01.jpg)

图 7.1 – 带有子匹配的正则表达式

这允许我们匹配一个字符串，并使用该字符串的某些部分作为我们的结果：

```cpp
sregex_token_iterator it{ in.begin(), in.end(), link_re, {1, 2} };
```

子匹配是编号的，从 `1` 开始。子匹配 `0` 是一个特殊值，代表整个匹配。

一旦我们有了迭代器，我们就像使用任何其他迭代器一样使用它：

```cpp
for(It end_it{}; it != end_it; ) {
    const string link{ *it++ };
    if(it == end_it) break;
    const string desc{ *it++ };
    cout << format("{:.<24} {}\n", desc, link);
}
```

这只是简单地通过 `regex` 迭代器遍历我们的结果，从而给出格式化的输出：

```cpp
Bill Weinman............ https://bw.org/
courses................. https://bw.org/courses/
music................... https://bw.org/music/
books................... https://packt.com/
back to the internet.... https://duckduckgo.com/
```
