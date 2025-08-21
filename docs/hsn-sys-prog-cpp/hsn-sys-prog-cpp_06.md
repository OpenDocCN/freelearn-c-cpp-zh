# 第六章：学习编程控制台输入/输出

控制台 IO 对于任何程序都是必不可少的。它可以用于获取用户输入，提供输出，并支持调试和诊断。程序不稳定的常见原因通常源于 IO 编写不佳，这只会加剧标准 C `printf()`/`scanf()` IO 函数的滥用。在本章中，我们将讨论使用 C++ IO 的利弊，通常称为基于流的 IO，与标准 C 风格的替代方法相比。此外，我们将提供一个关于 C++操作器的高级介绍，以及它们如何可以用来替代标准 C 风格的格式字符串。我们将以一组示例结束本章，旨在引导读者如何使用`std::cout`和`std::cin`。

本章有以下目标：

+   学习基于流的 IO

+   用户定义的类型操作器

+   回声的例子

+   串行回声服务器示例

# 技术要求

为了编译和执行本章中的示例，读者必须具备：

+   能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请参阅以下 GitHub 链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter06`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter06)。

# 学习基于流的 IO

在本节中，我们将学习基于流的 IO 的基础知识以及一些优缺点。

# 流的基础知识

与 C 风格的`printf()`和`scanf()`函数不同，C++ IO 使用流（`std::ostream`用于输出，`std::istream`用于输入），利用`<<`和`>>`操作符。例如，以下代码使用`basic_ostream`的非成员`<<`重载将`Hello World`输出到`stdout`：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello World\n";
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
Hello World
```

默认情况下，`std::cout`和`std::wcout`对象是`std::ostream`的实例，将数据输出到标准 C `stdout`，唯一的区别是`std::wcout`支持 Unicode，而`std::cout`支持 ASCII。除了几个非成员重载外，C++还提供了以下算术风格的成员重载：

```cpp
basic_ostream &operator<<(short value);
basic_ostream &operator<<(unsigned short value);
basic_ostream &operator<<(int value);
basic_ostream &operator<<(unsigned int value);
basic_ostream &operator<<(long value);
basic_ostream &operator<<(unsigned long value);
basic_ostream &operator<<(long long value);
basic_ostream &operator<<(unsigned long long value);
basic_ostream &operator<<(float value);
basic_ostream &operator<<(double value);
basic_ostream &operator<<(long double value);
basic_ostream &operator<<(bool value);
basic_ostream &operator<<(const void* value);
```

这些重载可以用于将各种类型的数字流到`stdout`或`stderr`。考虑以下例子：

```cpp
#include <iostream>

int main()
{
    std::cout << "The answer is: " << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

默认情况下，使用`stdin`进行输入，通过`std::cin`和`std::wcin`执行输入。与`std::cout`不同，`std::cin`使用`>>`流操作符，而不是`<<`流操作符。以下接受来自`stdin`的输入并将结果输出到`stdout`：

```cpp
#include <iostream>

int main()
{
    auto n = 0;

    std::cin >> n; 
    std::cout << "input: " << n << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
42 ↵
input: 42
```

# C++基于流的 IO 的优缺点

使用 C++进行 IO 而不是标准 C 函数有许多优缺点。

# C++基于流的 IO 的优点

通常情况下，C++流优于使用格式说明符的标准 C 函数，因为 C++流具有以下特点：

+   能够处理用户定义的类型，提供更清晰、类型安全的 IO

+   更安全，可以防止更多的意外缓冲区溢出漏洞，因为并非所有格式说明符错误都可以被编译器检测到或使用 C11 添加的`_s` C 函数变体来预防

+   能够提供隐式内存管理，不需要可变参数函数

因此，C++核心指南不鼓励使用格式说明符，包括`printf()`、`scanf()`等函数。尽管使用 C++流有许多优点，但也有一些缺点。

# C++基于流的 IO 的缺点

关于 C++流的两个最常见的抱怨如下：

+   标准 C 函数（特别是`printf()`）通常优于 C++流（这在很大程度上取决于您的操作系统和 C++实现）

+   格式说明符通常比`#include <iomanip>`更灵活

尽管这些通常是有效的抱怨，但有方法可以解决这些问题，而不必牺牲 C++流的优势，我们将在接下来的部分中解释。

# 从用户定义的类型开始

C++流提供了为用户定义的类型重载`<<`和`>>`运算符的能力。这提供了为任何数据类型创建自定义、类型安全的 IO 的能力，包括系统级数据类型、结构，甚至更复杂的类型，如类。例如，以下提供了对`<<`流运算符的重载，以打印由 POSIX 风格函数提供的错误代码：

```cpp
#include <fcntl.h>
#include <string.h>
#include <iostream>

class custom_errno
{ };

std::ostream &operator<<(std::ostream &os, const custom_errno &e)
{ return os << strerror(errno); }

int main()
{
    if (open("filename.txt", O_RDWR) == -1) {
        std::cout << custom_errno{} << '\n';
    }
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
No such file or directory
```

在这个例子中，我们创建了一个空类，为我们提供了一个自定义类型，并重载了这个自定义类型的`<<`运算符。然后我们使用`strerror()`来输出`errno`的错误字符串到提供的输出流。虽然可以通过直接将`strerror()`的结果输出到流中来实现这一点，但它演示了如何创建并使用流的用户定义类型。

除了更复杂的类型，用户定义的类型也可以通过输入流进行利用。考虑以下例子：

```cpp
#include <iostream>

struct object_t
{
    int data1;
    int data2;
};

std::ostream &operator<<(std::ostream &os, const object_t &obj)
{
    os << "data1: " << obj.data1 << '\n';
    os << "data2: " << obj.data2 << '\n';
    return os;
}

std::istream &operator>>(std::istream &is, object_t &obj)
{
    is >> obj.data1;
    is >> obj.data2;
    return is;
}

int main()
{
    object_t obj;

    std::cin >> obj;
    std::cout << obj;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
42 ↵
43 ↵
data1: 42
data2: 43
```

在这个例子中，我们创建了一个存储两个整数的结构。然后我们为这个用户定义的类型重载了`<<`和`>>`运算符，通过读取数据到我们类型的实例来练习这些重载，然后输出结果。通过我们的重载，我们已经指示了`std::cin`和`std::cout`如何处理我们用户定义的类型的输入和输出。

# 安全和隐式内存管理

尽管 C++流仍然可能存在漏洞，但与它们的标准 C 对应物相比，这种可能性较小。使用标准 C 的`scanf()`函数进行缓冲区溢出的经典示例如下：

```cpp
#include <stdio.h>

int main()
{
    char buf[2];
    scanf("%s", buf);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is 42 ↵
*** stack smashing detected ***: <unknown> terminated
Aborted (core dumped)
```

用户输入的缓冲区大于为该缓冲区分配的空间，导致缓冲区溢出的情况。在这个例子中增加`buf`的大小不会解决问题，因为用户总是可以输入一个比提供的缓冲区更大的字符串。这个问题可以通过在`scanf()`上指定长度限制来解决：

```cpp
#include <stdio.h>

int main()
{
    char buf[2];
    scanf("%2s", buf);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is 42 ↵
```

在这里，我们向`scanf()`函数提供了`buf`的大小，防止了缓冲区溢出。这种方法的问题是`buf`的大小声明了两次。如果这两者中的一个改变了，就可能重新引入缓冲区溢出。可以使用 C 风格的宏来解决这个问题，但缓冲区和其大小的解耦仍然存在。

虽然还有其他方法可以用 C 来解决这个问题，但解决 C++中前面提到的问题的一种方法如下：

```cpp
#include <iomanip>
#include <iostream>

template<std::size_t N>
class buf_t
{
    char m_buf[N];

public:

    constexpr auto size()
    { return N; }

    constexpr auto data()
    { return m_buf; }
};

template<std::size_t N>
std::istream &operator>>(std::istream &is, buf_t<N> &b)
{
    is >> std::setw(b.size()) >> b.data();
    return is;
}

int main()
{
    buf_t<2> buf;
    std::cin >> buf;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is 42 ↵
```

我们不是使用`*` char，而是创建一个封装`*` char 及其长度的用户定义类型。缓冲区的总大小与缓冲区本身耦合在一起，防止意外的缓冲区溢出。然而，如果允许内存分配（在编程系统中并非总是如此），我们可以做得更好：

```cpp
#include <string>
#include <iostream>

int main()
{
    std::string buf;
    std::cin >> buf;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is 42 ↵
```

在这个例子中，我们使用`std::string`来存储从`std::cin`输入的内容。这里的区别在于`std::string`根据需要动态分配内存来存储输入，防止可能的缓冲区溢出。如果需要更多的内存，就分配更多的内存，或者抛出`std::bad_alloc`并中止程序。C++流的用户定义类型提供了更安全的处理 IO 的机制。

# 常见的调试模式

在编程系统中，控制台输出的主要用途之一是调试。C++流提供了两个不同的全局对象——`std::cout`和`std::cerr`。第一个选项`std::cout`通常是缓冲的，发送到`stdout`，并且只有在发送到流的`std::flush`或`std::endl`时才会刷新。第二个选项`std::cerr`提供了与`std::cout`相同的功能，但是发送到`stderr`而不是`stdout`，并且在每次调用全局对象时都会刷新。看一下以下的例子：

```cpp
#include <iostream>

int main()
{
    std::cout << "buffered" << '\n';
    std::cout << "buffer flushed" << std::endl;
    std::cerr << "buffer flushed" << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
buffer
buffer flushed
buffer flushed
```

因此，错误逻辑通常使用`std::cerr`发送到`stderr`，以确保在发生灾难性问题时接收所有错误控制台输出。同样，一般输出，包括调试逻辑，使用`std::cout`发送到`stdout`，以利用缓冲加快控制台输出速度，并且使用`'\n'`发送换行而不是`std::endl`，除非需要显式刷新。

以下是 C 语言中调试的典型模式：

```cpp
#include <iostream>

#ifndef NDEBUG
#define DEBUG(...) fprintf(stdout, __VA_ARGS__);
#else
#define DEBUG(...)
#endif

int main()
{
    DEBUG("The answer is: %d\n", 42);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

如果启用了调试，通常意味着定义了`NDEBUG`，则可以使用`DEBUG`宏将调试语句发送到控制台。使用`NDEBUG`是因为这是大多数编译器设置为发布模式时定义的宏，禁用了标准 C 中的`assert()`。另一个常见的调试模式是为调试宏提供调试级别，允许开发人员在调试时调整程序的详细程度：

```cpp
#include <iostream>

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef NDEBUG
#define DEBUG(level,...) \
    if(level <= DEBUG_LEVEL) fprintf(stdout, __VA_ARGS__);
#else
#define DEBUG(...)
#endif

int main()
{
    DEBUG(0, "The answer is: %d\n", 42);
    DEBUG(1, "The answer no is: %d\n", 43);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

这种逻辑的问题在于过度使用宏来实现调试，这是 C++核心指南不赞成的模式。使用 C++17 进行调试的简单方法如下：

```cpp
#include <iostream>

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif

int main()
{
    if constexpr (!g_ndebug) {
        std::cout << "The answer is: " << 42 << '\n';
    }
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

即使使用了 C++17，仍然需要一些宏逻辑来处理启用调试时编译器提供的`NDEBUG`宏。在这个例子中，`NDEBUG`宏被转换为`constexpr`，然后在源代码中用于处理调试。调试级别也可以使用以下方式实现：

```cpp
#include <iostream>

#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif

int main()
{
    if constexpr (!g_ndebug && (0 <= g_debug_level)) {
        std::cout << "The answer is: " << 42 << '\n';
    }

    if constexpr (!g_ndebug && (1 <= g_debug_level)) {
        std::cout << "The answer is not: " << 43 << '\n';
    }
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

由于在这个例子中调试级别是一个编译时特性，它将使用`-DDEBUG_LEVEL=xxx`传递给编译器，因此仍然需要宏逻辑将 C 宏转换为 C++的`constexpr`。正如在这个例子中所看到的，C++实现比利用`fprintf()`和其他函数的简单`DEBUG`宏要复杂得多。为了克服这种复杂性，我们将利用封装，而不会牺牲编译时优化：

```cpp
#include <iostream>

#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif

template <std::size_t LEVEL>
constexpr void debug(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        func();
    };
}

int main()
{
    debug<0>([] {
        std::cout << "The answer is: " << 42 << '\n';
    });

    debug<1>([] {
        std::cout << "The answer is not: " << 43 << '\n';
    });
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

在这个例子中，调试逻辑被封装为一个接受 Lambda 的`constexpr`函数。调试级别使用模板参数来保持常数。与典型的标准 C 调试模式不同，这个实现将接受任何适合于`void(*func)()`函数或 lambda 的调试逻辑，并且与标准 C 版本一样，在编译器设置为发布模式时将被编译和移除（即定义了`NDEBUG`并且通常启用了优化）。为了证明这一点，当启用发布模式时，GCC 7.3 输出如下内容：

```cpp
> g++ -std=c++17 -O3 -DNDEBUG scratchpad.cpp; ./a.out
> ls -al a.out
-rwxr-xr-x 1 user users 8600 Apr 13 18:23 a.out

> readelf -s a.out | grep cout
```

当在源代码中添加`#undef NDEBUG`时，GCC 7.3 输出如下内容（确保唯一的区别是调试逻辑被禁用，但编译标志保持不变）：

```cpp
> g++ -std=c++17 scratchpad.cpp; ./a.out
> ls -al a.out
-rwxr-xr-x 1 user users 8888 Apr 13 18:24 a.out

> readelf -s a.out | grep cout
    23: 0000000000201060 272 OBJECT GLOBAL DEFAULT 24 _ZSt4cout@GLIBCXX_3.4 (5)
    59: 0000000000201060 272 OBJECT GLOBAL DEFAULT 24 _ZSt4cout@@GLIBCXX_3.4
```

额外的 288 字节来自于调试逻辑，这些逻辑完全被编译器移除，这要归功于 C++17 中`constexpr`的常数性，提供了一种更清晰的调试方法，而无需大量使用宏。

另一个常见的调试模式是在调试语句中包含当前行号和文件名，以提供额外的上下文。`__LINE__`和`__FILE__`宏用于提供这些信息。遗憾的是，在 C++17 中没有包含源位置 TS，因此没有办法在没有这些宏的情况下提供这些信息，也没有类似以下模式的包含：

```cpp
#include <iostream>

#ifndef NDEBUG
#define DEBUG(fmt, args...) \
    fprintf(stdout, "%s [%d]: " fmt, __FILE__, __LINE__, args);
#else
#define DEBUG(...)
#endif

int main()
{
    DEBUG("The answer is: %d\n", 42);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
scratchpad.cpp [11]: The answer is: 42
```

在这个例子中，`DEBUG`宏会自动将文件名和行号插入标准 C 风格的`fprintf()`函数中。这是因为无论编译器在哪里看到`DEBUG`宏，它都会插入`fprintf(stdout, "%s [%d]: " fmt, __FILE__, __LINE__, args);`，然后必须评估行和文件宏，从而产生预期的输出。将这种模式转换为我们现有的 C++示例的一个例子如下：

```cpp
#include <iostream>

#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif

#define console std::cout << __FILE__ << " [" << __LINE__ << "]: "

template <std::size_t LEVEL>
constexpr void debug(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        func();
    };
}

int main()
{
    debug<0>([] {
        console << "The answer is: " << 42 << '\n';
    });
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
scratchpad.cpp [27]: The answer is: 42
```

在我们的调试 lambda 中，我们不再使用`std::cout`，而是添加一个使用`std::cout`的控制台宏，但也将文件名和行号添加到调试语句中，以提供与标准 C 版本相同的功能。与标准 C 版本不同的是，不需要额外的 C 宏函数，因为控制台宏将正确提供使用的文件名和行号。

最后，为了完成我们的 C++17 调试模式，我们添加了一个带颜色的调试、警告和致命错误版本的前面示例，并为`fatal`函数添加了一个默认退出错误码为`-1`的重载版本。

首先，我们利用与前面代码片段中相同的标准 C 宏：

```cpp
#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif
```

这些宏将标准 C 风格的宏转换为 C++风格的常量表达式，这些宏在命令行兼容性中是必需的。接下来，我们创建一个名为`debug`的模板函数，能够接受一个 lambda 函数。这个`debug`函数首先将绿色的`debug`输出到`stdout`，然后执行 lambda 函数，只有在调试被启用并且调试级别与提供给`debug`函数本身的级别匹配时才执行。如果调试未启用，`debug`函数将在不影响程序大小或性能的情况下编译。

```cpp
template <std::size_t LEVEL>
constexpr void debug(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::cout << "\033[1;32mDEBUG\033[0m ";
        func();
    };
}
```

这个相同的`debug`函数被重复使用来提供警告和致命错误版本的函数，唯一的区别是颜色（这是特定于平台的，在这种情况下是为 UNIX 操作系统设计的），而`fatal`函数在执行 lambda 函数后退出程序，退出时使用用户定义的错误码或`-1`。

```cpp
template <std::size_t LEVEL>
constexpr void warning(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::cout << "\033[1;33mWARNING\033[0m ";
        func();
    };
}

template <std::size_t LEVEL>
constexpr void fatal(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::cout << "\033[1;31mFATAL ERROR\033[0m ";
        func();
        ::exit(-1);
    };
}

template <std::size_t LEVEL>
constexpr void fatal(int error_code, void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::cout << "\033[1;31mFATAL ERROR\033[0m ";
        func();
        ::exit(error_code);
    };
}
```

最后，这些调试模式在`main()`函数中得到了应用，以演示它们的使用方法。

```cpp
int main()
{
    debug<0>([] {
        console << "The answer is: " << 42 << '\n';
    });

    warning<0>([] {
        console << "The answer might be: " << 42 << '\n';
    });

    fatal<0>([] {
        console << "The answer was not: " << 42 << '\n';
    });
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
DEBUG scratchpad.cpp [54]: The answer is: 42
WARNING scratchpad.cpp [58]: The answer might be: 42
FATAL ERROR scratchpad.cpp [62]: The answer was not: 42
```

# C++流的性能

关于 C++流的一个常见抱怨是性能问题，这个问题在多年来已经得到了很大的缓解。为了确保 C++流的最佳性能，可以应用一些优化：

+   **禁用 std::ios::sync_with_stdio**：C++流默认会与标准 C 函数（如`printf()`等）同步。如果不使用这些函数，应该禁用这个同步功能，因为这将显著提高性能。

+   **避免刷新**：在可能的情况下，避免刷新 C++流，让`libc++`和操作系统来处理刷新。这包括不使用`std::flush`，而是使用`'\n'`代替`std::endl`，后者在输出换行后会刷新。避免刷新时，所有输出都会被缓冲，减少了向操作系统传递输出的次数。

+   **使用 std::cout 和 std::clog 而不是 std::cerr**：出于同样的原因，`std::cerr`在销毁时会刷新，增加了操作系统传递输出的次数。在可能的情况下，应该使用`std::cout`，只有在出现致命错误后才使用`std::cerr`，例如退出、异常、断言和可能的崩溃。

对于问题“*哪个更快*，`printf()` *还是* `std::cout`*？*”，不可能提供一个一般性的答案。但实际上，如果使用了前面的优化，`std::cout`通常可以优于标准 C 的`printf()`，但这高度依赖于您的环境和用例。

除了前面的例子，避免不必要的刷新以提高性能的一种方法是使用`std::stringstream`而不是`std::cout`。

```cpp
#include <sstream>
#include <iostream>

int main()
{
    std::stringstream stream;
    stream << "The answer is: " << 42 << '\n';

    std::cout << stream.str() << std::flush;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

通过使用`std::stringstream`，所有输出都被定向到您控制的缓冲区，直到您准备通过`std::cout`和手动刷新将输出发送到操作系统。这也可以用于缓冲输出到`std::cerr`，减少总刷新次数。避免刷新的另一种方法是使用`std::clog`：

```cpp
#include <iostream>

int main()
{
    std::clog << "The answer is: " << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
```

`std::clog`的操作方式类似于`std::cout`，但是不是将输出发送到`stdout`，而是将输出发送到`stderr`。

# 学习操纵器

C++流有几种不同的操纵器，可以用来控制输入和输出，其中一些已经讨论过。最常见的操纵器是`std::endl`，它输出一个换行符，然后刷新输出流：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello World" << std::endl;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
Hello World
```

编写相同逻辑的另一种方法是使用`std::flush`操纵器：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello World\n" << std::flush;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
Hello World
```

两者是相同的，尽管除非明确需要刷新，否则应始终使用`'\n'`。例如，如果需要多行，应首选以下方式：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello World\n";
    std::cout << "Hello World\n";
    std::cout << "Hello World\n";
    std::cout << "Hello World" << std::endl;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
Hello World
Hello World
Hello World
Hello World
```

与前面的代码相比，以下代码不是首选：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello World" << std::endl;
    std::cout << "Hello World" << std::endl;
    std::cout << "Hello World" << std::endl;
    std::cout << "Hello World" << std::endl;
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
Hello World
Hello World
Hello World
Hello World
```

应该注意，不需要尾随刷新，因为`::exit()`在`main`完成时会为您刷新`stdout`。

在任何程序开始时设置的常见操纵器是`std::boolalpha`，它导致布尔值输出为`true`或`false`，而不是`1`或`0`（`std::noboolalpha`提供相反的效果，这也是默认值）：

```cpp
#include <iostream>

int main()
{
    std::cout << std::boolalpha;
    std::cout << "The answer is: " << true << '\n';
    std::cout << "The answer is: " << false << '\n';

    std::cout << std::noboolalpha;
    std::cout << "The answer is: " << true << '\n';
    std::cout << "The answer is: " << false << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: true
The answer is: false
The answer is: 1
The answer is: 0
```

另一组常见的操纵器是数字基操纵器——`std::hex`、`std::dec`和`std::oct`。这些操纵器类似于标准 C 格式说明符（例如`printf()`使用的`%d`、`%x`和`%o`）。与标准 C 版本不同，这些操纵器是全局的，因此在库中使用时应谨慎使用。要使用这些操纵器，只需在添加所需基数的数字之前将它们添加到流中：

```cpp
#include <iostream>

int main()
{
    std::cout << "The answer is: " << 42 << '\n' << std::hex 
              << "The answer is: " << 42 << '\n';
    std::cout << "The answer is: " << 42 << '\n' << std::dec 
              << "The answer is: " << 42 << '\n';
    std::cout << "The answer is: " << 42 << '\n' << std::oct 
              << "The answer is: " << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 42
The answer is: 2a
The answer is: 2a
The answer is: 42
The answer is: 42
The answer is: 52
```

第一个数字`42`打印为`42`，因为尚未使用任何数字基操纵器。第二个数字打印为`2a`，因为使用了`std::hex`操纵器，导致`2a`是`42`的十六进制值。打印的第三个数字也是`2a`，因为数字基操纵器是全局的，因此，即使第二次调用`std::cout`，流仍然被告知使用十六进制值而不是十进制值。这种模式对于`std::dec`（例如，十进制数）和`std::oct`（例如，八进制数）都是一样的，结果是`42`、`2a`、`2a`、`42`、`42`，最后是`52`。

还可以使用`std::hex`的大写版本，而不是前面示例中看到的默认小写版本。要实现这一点，使用`std::uppercase`和`std::nouppercase`（`std::uppercase`显示大写字母数字字符，而`std::nouppercase`不显示，这是默认值）：

```cpp
#include <iostream>

int main()
{
    std::cout << std::hex << std::uppercase << "The answer is: " 
              << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 2A
```

在这个例子中，`42`不再输出为`2a`，而是输出为`2A`，其中字母数字字符是大写的。

通常，特别是在编程系统方面，十六进制和八进制数以它们的基数标识符（例如`0x`和`0`）打印。要实现这一点，使用`std::showbase`和`std::noshowbase`操纵器（`std::showbase`显示基数，`std::noshowbase`不显示，这是默认值）：

```cpp
#include <iostream>

int main()
{
    std::cout << std::showbase;
    std::cout << std::hex << "The answer is: " << 42 << '\n';
    std::cout << std::dec << "The answer is: " << 42 << '\n';
    std::cout << std::oct << "The answer is: " << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 0x2a
The answer is: 42
The answer is: 052
```

从这个例子中可以看出，`std::hex`现在输出`0x2a`，而不是`2a`，`std::oct`输出`052`，而不是`52`，而`std::dec`继续按预期输出`42`（因为十进制数没有基数标识符）。与数字不同，指针始终以十六进制、小写形式输出，并显示它们的基数，`std::uppercase`、`std::noshowbase`、`std::dec`和`std::oct`不会影响输出。解决这个问题的一个方法是将指针转换为数字，然后可以使用前面的操纵器，如下例所示，但是 C++核心指南不鼓励这种逻辑，因为需要使用`reinterpret_cast`：

```cpp
#include <iostream>

int main()
{
    int i = 0;
    std::cout << &i << '\n';
    std::cout << std::hex << std::showbase << std::uppercase 
              << reinterpret_cast<uintptr_t>(&i) << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
0x7fff51d370b4
0X7FFF51D370B4
```

输出指针的一个问题是它们的总长度（即字符的总数）会从一个指针变化到另一个指针。当同时输出多个指针时，这通常会分散注意力，因为它们的基本修改器可能不匹配。为了克服这一点，可以使用`std::setw`和`std::setfill`。`std::setw`设置下一个输出的总宽度（即字符的总数）。如果下一个输出的大小不至少是传递给`std::setw`的值的大小，流将自动向流中添加空格：

```cpp
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "The answer is: " << std::setw(18) << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is:                 42
```

在这个例子中，宽度设置为`18`。由于流的下一个添加是两个字符（来自数字`42`），在将`42`添加到流之前添加了`16`个空格。要更改由`std::setw`添加到流中的字符，请使用`std::setfill`：

```cpp
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "The answer is: " << std::setw(18) << std::setfill('0') 
              << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 000000000000000042
```

可以看到，流中添加的不是空格（默认情况下），而是添加到流中的'0'字符。可以使用`std::left`，`std::right`和`std::internal`来控制添加到流中的字符的方向：

```cpp
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "The answer is: "
              << std::setw(18) << std::left << std::setfill('0')
              << 42 << '\n';

    std::cout << "The answer is: "
              << std::setw(18) << std::right << std::setfill('0')
              << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 420000000000000000
The answer is: 000000000000000042
```

`std::left`首先输出到流中，然后用剩余的字符填充流，而`std::right`用未使用的字符填充流，然后输出到流中。`std::internal`特定于使用基本标识符（如`std::hex`和`std::oct`）的文本以及使用`std::showbase`或自动显示基本标识符的指针。

```cpp
#include <iomanip>
#include <iostream>

int main()
{
    int i = 0;

    std::cout << std::hex
              << std::showbase;

    std::cout << "The answer is: "
              << std::setw(18) << std::internal << std::setfill('0')
              << 42 << '\n';

    std::cout << "The answer is: "
              << std::setw(18) << std::internal << std::setfill('0')
              << &i << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 0x000000000000002a
The answer is: 0x00007ffc074c9be4
```

通常，特别是在库中，设置一些操纵器然后将流恢复到其原始状态是有用的。例如，如果您正在编写一个库，并且想要以`hex`输出一个数字，您需要使用`std::hex`操纵器，但这样做会导致从那时起用户输出的所有数字也以`hex`输出。问题是，您不能简单地使用`std::dec`将流设置回十进制，因为用户可能实际上是首先使用`std::hex`。解决这个问题的一种方法是使用`std::cout.flags()`函数，它允许您获取和设置流的内部标志：

```cpp
#include <iostream>

int main()
{
    auto flags = std::cout.flags();
    std::cout.flags(flags);
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
```

总的来说，所有已经讨论过的操纵器以及其他一些操纵器都可以使用`std::cout.flags()`函数启用/禁用，所讨论的操纵器只是这个函数的包装器，以减少冗长。虽然这个函数可以用来配置操纵器（应该避免），`std::cout.flags()`函数是在流被更改后恢复操纵器的便捷方法。还应该注意，前面的方法适用于所有流，而不仅仅是`std::cout`。简化恢复操纵器的一种方法是使用一些函数式编程，并用保存/恢复逻辑包装用户函数，如下所示：

```cpp
#include <iomanip>
#include <iostream>

template<typename FUNC>
void cout_transaction(FUNC f)
{
    auto flags = std::cout.flags();
    f();
    std::cout.flags(flags);
}

int main()
{
    cout_transaction([]{
        std::cout << std::hex << std::showbase;
        std::cout << "The answer is: " << 42 << '\n';
    });

    std::cout << "The answer is: " << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 0x2a
The answer is: 42
```

在这个例子中，我们将`std::cout`的使用包装在`cout_transation`中。这个包装器存储操纵器的当前状态，调用用户提供的函数（改变操纵器），然后在完成之前恢复操纵器。结果是，事务完成后操纵器不受影响，这意味着这个例子中的第二个`std::cout`输出`42`而不是`0x2a`。

最后，为了简化操纵器的使用，有时创建自定义的用户定义操纵器可以封装自定义逻辑是很有用的：

```cpp
#include <iomanip>
#include <iostream>

namespace usr
{
    class hex_t { } hex;
}

std::ostream &
operator<<(std::ostream &os, const usr::hex_t &obj)
{
    os << std::hex << std::showbase << std::internal
        << std::setfill('0') << std::setw(18);

    return os;
}

int main()
{
    std::cout << "The answer is: " << usr::hex << 42 << '\n';
}

> g++ -std=c++17 scratchpad.cpp; ./a.out
The answer is: 0x000000000000002a
```

从这个例子可以看出，只需使用`usr::hex`而不是`std::hex`，就可以使用`std::hex`，`std::showbase`，`std::internal`，`std::setfill('0')`和`std::setw(18)`输出`42`，减少冗长并简化对相同逻辑的多次使用。

# 重新创建 echo 程序

在这个实际例子中，我们将重新创建几乎所有`POSIX`系统上都可以找到的流行的 echo 程序。echo 程序接受程序提供的所有输入并将其回显到`stdout`。这个程序非常简单，具有以下程序选项：

+   -n：防止 echo 在退出时输出换行符

+   `--help`：打印帮助菜单

+   `--version`：打印一些版本信息

还有两个选项，`-e`和`-E`；我们在这里省略了它们，以保持简单，但如果需要，可以作为读者的一个独特练习。

要查看此示例的完整源代码，请参见以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter06/example1.cpp`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter06/example1.cpp)。

此处呈现的`main`函数是一个有用的模式，与原始的 echo 程序略有不同，因为异常（在本例中极不可能）可能会生成原始 echo 程序中看不到的错误消息；但是，它仍然很有用：

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

此逻辑的目标是在程序退出之前捕获任何异常，并在退出之前将异常描述输出到`stderr`。

考虑以下示例：

```cpp
catch (const std::exception &e) {
    std::cerr << "Caught unhandled exception:\n";
    std::cerr << " - what(): " << e.what() << '\n';
}
```

前面的代码捕获所有`std::exceptions`并将捕获的异常描述（即`e.what()`）输出到`stderr`。请注意，这里使用的是`std::cerr`（而不是`std::clog`），以防异常的使用会导致不稳定性，确保发生刷新。在使用错误处理逻辑时，最好始终保持谨慎，并确保所有调试输出都以性能为次要考虑因素进行。

考虑以下示例：

```cpp
catch (...) {
    std::cerr << "Caught unknown exception\n";
}
```

前面的代码捕获所有未知异常，在本程序中几乎肯定永远不会发生，并且纯粹是为了完整性而添加：

```cpp
try {
    return protected_main(argc, argv);
}
```

`try`块尝试执行`protected_main()`函数，如果出现异常，则执行先前描述的`catch`块；否则，从`main`函数返回，最终退出程序。

`protected_main()`函数的目标是解析程序提供的参数，并按预期处理每个参数：

```cpp
int
protected_main(int argc, char **argv)
{
    using namespace gsl;

    auto endl = true;
    auto args = make_span(argv, argc);

    for (int i = 1, num = 0; i < argc; i++) {
        cstring_span<> span_arg = ensure_z(args.at(i));

        if (span_arg == "-n") {
            endl = false;
            continue;
        }

        if (span_arg == "--help") {
            handle_help();
        }

        if (span_arg == "--version") {
            handle_version();
        }

        if (num++ > 0) {
            std::cout << " ";
        }

        std::cout << span_arg.data();
    }

    if (endl) {
        std::cout << '\n';
    }

    return EXIT_SUCCESS;
}
```

以下是第一行：

```cpp
auto endl = true;
```

它用于控制是否在退出时向`stdout`添加换行符，就像原始的 echo 程序一样，并由`-n`程序参数控制。以下是下一行：

```cpp
auto args = make_span(argv, argc);
```

前面的代码将标准 C `argv`和`argc`参数转换为 C++ GSL span，使我们能够以符合 C++核心指南的方式安全地处理程序参数。该 span 只不过是一个列表（具体来说，它与`std::array`非常相似），每次访问列表时都会检查此列表的边界（不像`std::array`）。如果我们的代码尝试访问不存在的参数，将抛出异常，并且程序将以错误代码安全退出，通过`stderr`告诉我们尝试访问不存在的列表元素（通过`main`函数中的`try`/`catch`逻辑）。

以下是下一部分：

```cpp
for (int i = 1, num = 0; i < argc; i++) {
    cstring_span<> span_arg = ensure_z(args.at(i));
```

它循环遍历列表中的每个参数。通常，我们会使用范围`for`语法循环遍历列表中的每个元素：

```cpp
for (const auto &arg : args) {
    ...
}
```

但是，不能使用此语法，因为参数列表中的第一个参数始终是程序名称，在我们的情况下应该被忽略。因此，我们从`1`开始（而不是`0`），如前所述，然后循环遍历列表中的其余元素。此片段中的第二行从列表中的每个程序参数创建`cstring_span{}`。`cstring_span{}`只不过是一个标准的 C 风格字符串，包装在 GSL span 中，以保护对字符串的任何访问，使 C 风格字符串访问符合 C++核心指南。稍后将使用此包装器来比较字符串，以安全和符合规范的方式查找我们的程序选项，例如`-n`，`--help`和`--version`。`ensure_z()`函数确保字符串完整，防止可能的意外损坏。

下一步是将每个参数与我们计划支持的参数列表进行比较：

```cpp
if (span_arg == "-n") {
    endl = false;
    continue;
}
```

由于我们使用`cstring_span{}`而不是标准的 C 风格字符串，我们可以安全地直接将参数与`"-n"`字面字符串进行比较，而无需使用不安全的函数（如`strcmp()`）或直接字符比较，这是原始 echo 实现所做的（由于我们只支持一个单个字符选项，性能是相同的）。如果参数是`-n`，我们指示我们的实现在程序退出时不应向`stdout`添加换行符，通过将`endl`设置为`false`，然后我们继续循环处理参数，直到它们全部被处理。

以下是接下来的两个代码块：

```cpp
if (span_arg == "--help") {
    handle_help();
}

if (span_arg == "--version") {
    handle_version();
}
```

它们检查参数是否为`--help`或`--version`。如果用户提供了其中任何一个，将执行特殊的`handle_help()`或`handle_version()`函数。需要注意的是，`handle_xxx()`函数在完成时退出程序，因此不需要进一步的逻辑，并且应该假定这些函数永远不会返回（因为程序退出）。

此时，所有可选参数都已处理。所有其他参数应该像原始的 echo 程序一样输出到`stdout`。问题在于用户可能提供多个希望输出到`stdout`的参数。考虑以下例子：

```cpp
> echo Hello World
Hello World
```

在这个例子中，用户提供了两个参数——`Hello`和`World`。预期输出是`Hello World`（有一个空格），而不是`HelloWorld`（没有空格），需要一些额外的逻辑来确保根据需要将空格输出到`stdout`。

以下是下一个代码块：

```cpp
if (num++ > 0) {
    std::cout << " ";
}
```

这在第一个参数已经输出后向`stdout`输出一个空格，但在下一个参数即将输出之前（以及所有剩余的参数）。这是因为`num`开始为`0`（`0`等于`0`，而不是大于`0`，因此在第一个参数上不会输出空格），然后`num`被递增。当处理下一个参数时，`num`为`1`（或更大），大于`0`，因此空格被添加到`stdout`。

最后，通过向`std::cout`提供参数的数据，将参数添加到`stdout`，这只是`std::cout`可以安全处理的参数的不安全的标准 C 版本：

```cpp
std::cout << span_arg.data();
```

`protected_main()`函数中的最后一个代码块是：

```cpp
if (endl) {
    std::cout << '\n';
}

return EXIT_SUCCESS;
```

默认情况下，`endl`是`true`，因此在程序退出之前会向`stdout`添加一个换行符。然而，如果用户提供了`-n`，那么`endl`将被设置为`false`。

```cpp
if (span_arg == "-n") {
    endl = false;
    continue;
}
```

在上述代码中，如果用户提供了`--help`，则会执行`handle_help()`函数如下：

```cpp
void
handle_help()
{
    std::cout
            << "Usage: echo [SHORT-OPTION]... [STRING]...\n"
            << " or: echo LONG-OPTION\n"
            << "Echo the STRING(s) to standard output.\n"
            << "\n"
            << " -n do not output the trailing newline\n"
            << " --help display this help and exit\n"
            << " --version output version information and exit\n";

    ::exit(EXIT_SUCCESS);
}
```

该函数使用`std::cout`将帮助菜单输出到`stdout`，然后成功退出程序。如果用户提供了`--version`，`handle_version()`函数也会执行相同的操作：

```cpp
void
handle_version()
{
    std::cout
            << "echo (example) 1.0\n"
            << "Copyright (C) ???\n"
            << "\n"
            << "Written by Rian Quinn.\n";

    ::exit(EXIT_SUCCESS);
}
```

要编译这个例子，我们使用 CMake：

```cpp
# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.6)
project(chapter6)

include(ExternalProject)
find_package(Git REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Guideline Support Library
# ------------------------------------------------------------------------------

list(APPEND GSL_CMAKE_ARGS
    -DGSL_TEST=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
)

ExternalProject_Add(
    gsl
    GIT_REPOSITORY https://github.com/Microsoft/GSL.git
    GIT_SHALLOW 1
    CMAKE_ARGS ${GSL_CMAKE_ARGS}
    PREFIX ${CMAKE_BINARY_DIR}/external/gsl/prefix
    TMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/tmp
    STAMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/stamp
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/gsl/download
    SOURCE_DIR ${CMAKE_BINARY_DIR}/external/gsl/src
    BINARY_DIR ${CMAKE_BINARY_DIR}/external/gsl/build
)

# ------------------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------------------
```

```cpp

include_directories(${CMAKE_BINARY_DIR}/include)
add_executable(example1 example1.cpp)
add_dependencies(example1 gsl)
```

这是`CMakeLists.txt`文件的头部分：

```cpp
cmake_minimum_required(VERSION 3.6)
project(chapter6)

include(ExternalProject)
find_package(Git REQUIRED)

set(CMAKE_CXX_STANDARD 17)
```

这设置了 CMake 要求版本为 3.6（因为我们使用`GIT_SHALLOW`），为项目命名，包括`ExternalProject`模块（提供了`ExternalProject_Add`），并将 C++标准设置为 C++17。

以下是下一部分：

```cpp
# ------------------------------------------------------------------------------
# Guideline Support Library
# ------------------------------------------------------------------------------

list(APPEND GSL_CMAKE_ARGS
    -DGSL_TEST=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
)

ExternalProject_Add(
    gsl
    GIT_REPOSITORY https://github.com/Microsoft/GSL.git
    GIT_SHALLOW 1
    CMAKE_ARGS ${GSL_CMAKE_ARGS}
    PREFIX ${CMAKE_BINARY_DIR}/external/gsl/prefix
    TMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/tmp
    STAMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/stamp
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/gsl/download
    SOURCE_DIR ${CMAKE_BINARY_DIR}/external/gsl/src
    BINARY_DIR ${CMAKE_BINARY_DIR}/external/gsl/build
)
```

它使用 CMake 的`ExternalProject_Add`从 GitHub 上的 Git 存储库下载并安装 GSL，使用深度为 1（即`GIT_SHALLOW 1`）来加快下载过程。提供给`ExternalProject_Add`的参数（即`GSL_CMAKE_ARGS`）告诉 GSL 的构建系统关闭单元测试（我们的项目不需要）并将生成的头文件安装到我们的构建目录中（将它们放在我们的`build`目录中的`include`文件夹中）。提供给`ExternalProject_Add`的其余参数是可选的，只是用来清理`ExternalProject_Add`的输出，并且可以被忽略，甚至在需要时删除。

最后，这是最后一个代码块：

```cpp
include_directories(${CMAKE_BINARY_DIR}/include)
add_executable(example1 example1.cpp)
```

它告诉构建系统在哪里找到我们新安装的 GSL 头文件，然后从`example1.cpp`源代码创建一个名为`example1`的可执行文件。要编译和运行此示例，只需执行：

```cpp
> mkdir build; cd build
> cmake ..; make
...
> ./example1 Hello World
Hello World
```

# 理解串行回显服务器示例

在这个实际示例中，我们将创建一个基于串行的回显服务器。回显服务器（无论类型如何）都会接收输入并将输入回显到程序的输出（类似于第一个示例，但在这种情况下使用串行端口上的服务器式应用程序）。

要查看此示例的完整源代码，请参阅以下内容：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter06/example2.cpp`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter06/example2.cpp)。

```cpp
#include <fstream>
#include <iostream>

#include <gsl/gsl>
using namespace gsl;

void
redirect_output(
    const std::ifstream &is,
    const std::ofstream &os,
    std::function<void()> f)
{
    auto cinrdbuf = std::cin.rdbuf();
    auto coutrdbuf = std::cout.rdbuf();

    std::cin.rdbuf(is.rdbuf());
    std::cout.rdbuf(os.rdbuf());

    f();

    std::cin.rdbuf(cinrdbuf);
    std::cout.rdbuf(coutrdbuf);
}

auto
open_streams(cstring_span<> port)
{
    std::ifstream is(port.data());
    std::ofstream os(port.data());

    if (!is || !os) {
        std::clog << "ERROR: unable to open serial port:" << port.data() << '\n';
        ::exit(EXIT_FAILURE);
    }

    return std::make_pair(std::move(is), std::move(os));
}

int
protected_main(int argc, char** argv)
{
    auto args = make_span(argv, argc);

    if (argc != 2) {
        std::clog << "ERROR: unsupported number of arguments\n";
        ::exit(EXIT_FAILURE);
    }

    auto [is, os] = open_streams(
        ensure_z(args.at(1))
    );

    redirect_output(is, os, []{
        std::string buf;

        std::cin >> buf;
        std::cout << buf << std::flush;
    });

    return EXIT_SUCCESS;
}
```

`main`函数与第一个示例相同。它的唯一目的是捕获可能触发的任何异常，将异常的描述输出到`stderr`，并以失败状态安全地退出程序。有关其工作原理的更多信息，请参见第一个示例。`protected_main()`函数的目的是打开串行端口，读取输入，并将输入回显到输出：

```cpp
int
protected_main(int argc, char** argv)
{
    auto args = make_span(argv, argc);

    if (argc != 2) {
        std::clog << "ERROR: unsupported number of arguments\n";
        ::exit(EXIT_FAILURE);
    }

    auto [is, os] = open_streams(
        ensure_z(args.at(1))
    );

    redirect_output(is, os, []{
        std::string buf;

        std::cin >> buf;
        std::cout << buf << std::flush;
    });

    return EXIT_SUCCESS;
}
```

这是第一行：

```cpp
auto args = make_span(argv, argc);
```

它做的事情和第一个示例一样，将`argc`和`argv`参数参数包装在 GSL span 中，为解析用户提供的参数提供了安全机制。

这是第二个块：

```cpp
if (argc != 2) {
    std::clog << "ERROR: unsupported number of arguments\n";
    ::exit(EXIT_FAILURE);
}
```

它检查确保用户提供了一个且仅一个参数。`argc`的总数为`2`而不是`1`的原因是因为第一个参数总是程序的名称，在这种情况下应该被忽略，因此用户提供的`1`个参数实际上等于`argc`的`2`。此外，我们使用`std::clog`而不是`std::cerr`，因为在这种情况下不太可能不稳定，并且当调用`::exit()`时，`libc`将为我们执行刷新。

这是第二个块：

```cpp
auto [is, os] = open_streams(
    ensure_z(args.at(1))
);
```

它打开串行端口并返回输入和输出流，`std::cout`和`std::cin`可以使用串行端口而不是`stdout`和`stdin`。为此，使用了`open_streams()`函数：

```cpp
auto
open_streams(cstring_span<> port)
{
    std::ifstream is(port.data());
    std::ofstream os(port.data());

    if (!is || !os) {
        std::clog << "ERROR: unable to open serial port:" << port.data() << '\n';
        ::exit(EXIT_FAILURE);
    }

    return std::make_pair(std::move(is), std::move(os));
}
```

此函数接受一个`cstring_span{}`，用于存储要打开的串行端口（例如`/dev/ttyS0`）。

接下来我们转到以下流：

```cpp
std::ifstream is(port.data());
std::ofstream os(port.data());
```

前面的代码打开了一个输入和输出流，绑定到这个串行端口。`ifstream{}`和`ofstream{}`都是文件流，超出了本章的范围（它们将在以后的章节中解释），但简而言之，这些打开了串行设备并提供了一个流对象，`std::cout`和`std::cin`可以使用它们，就好像它们在使用`stdout`和`stdin`（这在`POSIX`系统上也是技术上的文件流）。

这是下一个块：

```cpp
if (!is || !os) {
    std::clog << "ERROR: unable to open serial port:" << port.data() << '\n';
    ::exit(EXIT_FAILURE);
}
```

它验证了输入流和输出流是否成功打开，这很重要，因为这种类型的错误可能发生（例如，提供了无效的串行端口，或者用户无法访问串行端口）。如果发生错误，用户将通过输出到`std::clog`的消息得到通知，并且程序以失败状态退出。

最后，如果输入流和输出流成功打开，它们将作为一对返回，`protected_main()`函数将使用结构化绑定语法（C++17 中添加的功能）读取它们。

这是`protected_main()`函数中的下一个块：

```cpp
redirect_output(is, os, []{
    std::string buf;

    std::cin >> buf;
    std::cout << buf << std::flush;
});
```

它将`std::cout`和`std::cin`重定向到串行端口，然后将输入回显到程序的输出，实际上回显了用户提供的串行端口。为了执行重定向，使用了`redirect_output()`函数：

```cpp
void
redirect_output(
    const std::ifstream &is,
    const std::ofstream &os,
    std::function<void()> f)
{
    auto cinrdbuf = std::cin.rdbuf();
    auto coutrdbuf = std::cout.rdbuf();

    std::cin.rdbuf(is.rdbuf());
    std::cout.rdbuf(os.rdbuf());

    f();

    std::cin.rdbuf(cinrdbuf);
    std::cout.rdbuf(coutrdbuf);
}
```

`redirect_output()`函数将输入和输出流作为参数，以及要执行的函数和最终参数。`redirect_function()`的第一件事是保存`std::cin`和`std::cout`的当前缓冲区：

```cpp
auto cinrdbuf = std::cin.rdbuf();
auto coutrdbuf = std::cout.rdbuf();
```

接下来我们看到：

```cpp
std::cin.rdbuf(is.rdbuf());
std::cout.rdbuf(os.rdbuf());
```

`std::cin`和`std::cout`都重定向到提供的输入和输出流。完成此操作后，将执行提供的函数。任何对`std::cin`和`std::cout`的使用都将重定向到提供的串行端口，而不是标准的`stdout`和`stdin`。当`f()`函数完成时，`std::cin`和`std::cout`将恢复到它们的原始缓冲区，将它们重定向回`stdout`和`stdin`：

```cpp
std::cin.rdbuf(cinrdbuf);
std::cout.rdbuf(coutrdbuf);
```

最后，程序成功退出。要编译此示例，我们使用 CMake：

```cpp
# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.6)
project(chapter6)

include(ExternalProject)
find_package(Git REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Guideline Support Library
# ------------------------------------------------------------------------------

list(APPEND GSL_CMAKE_ARGS
    -DGSL_TEST=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
)

ExternalProject_Add(
    gsl
    GIT_REPOSITORY https://github.com/Microsoft/GSL.git
    GIT_SHALLOW 1
    CMAKE_ARGS ${GSL_CMAKE_ARGS}
    PREFIX ${CMAKE_BINARY_DIR}/external/gsl/prefix
    TMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/tmp
    STAMP_DIR ${CMAKE_BINARY_DIR}/external/gsl/stamp
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/gsl/download
    SOURCE_DIR ${CMAKE_BINARY_DIR}/external/gsl/src
    BINARY_DIR ${CMAKE_BINARY_DIR}/external/gsl/build
)

# ------------------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------------------

include_directories(${CMAKE_BINARY_DIR}/include)
add_executable(example2 example2.cpp)
add_dependencies(example2 gsl)
```

这个`CMakeLists.txt`与第一个例子中的`CMakeLists.txt`相同（减去了使用`example1`而不是`example2`）。有关此操作原理的完整解释，请参阅本章中的第一个例子。

要编译和使用此示例，需要两台计算机，一台用作 echo 服务器，另一台用作客户端，两台计算机的串行端口连接在一起。在 echo 服务器计算机上，使用以下命令：

```cpp
> mkdir build; cd build
> cmake ..; make
...
> ./example2 /dev/ttyS0
```

请注意，您的串行端口设备可能不同。在客户计算机上，打开两个终端。在第一个终端中，运行以下命令：

```cpp
> cat < /dev/ttyS0
```

这段代码等待串行设备输出数据。在第二个终端中运行：

```cpp
> echo "Hello World" > /dev/ttyS0
```

这将通过串行端口将数据发送到 echo 服务器。当您按下*Enter*时，您将看到我们在 echo 服务器上成功关闭的`example2`程序，并且客户端的第一个终端将显示`Hello World`：

```cpp
> cat < /dev/ttyS0
Hello World
```

# 摘要

在本章中，我们学习了如何使用 C++17 执行基于控制台的 IO，这是一种常见的系统编程需求。与`printf()`和`scanf()`等标准 C 风格的 IO 函数不同，C++使用基于流的 IO 函数，如`std::cout`和`std::cin`。使用基于流的 IO 有许多优点和一些缺点。例如，基于流的 IO 提供了一种类型安全的机制来执行 IO，而原始的 POSIX 风格的`write()`函数通常由于不调用`malloc()`和`free()`而能够优于基于流的 IO。

此外，我们还研究了基于流的操作符，它们为基于流的 IO 提供了与标准 C 风格格式字符串类似的功能集，但没有 C 等效项中常见的不稳定性问题。除了操纵数字和布尔值的格式之外，我们还探讨了字段属性，包括宽度和对齐。

最后，我们用两个不同的例子结束了本章。第一个例子展示了如何在 C++中实现流行的 POSIX *echo*程序，而不是在 C 中。第二个例子创建了一个*echo*服务器，用于串行端口，它使用`std::cin`从串行端口接收输入，并使用`std::cout`将该输入作为输出发送回串行端口。

在下一章中，我们将全面介绍 C、C++和 POSIX 提供的内存管理设施，包括对齐内存和 C++智能指针。

# 问题

1.  相比标准 C 的`scanf`，`std::cin`如何帮助防止缓冲区溢出？

1.  至少列举一个使用 C++流相对于标准 C 风格的`printf`/`scanf`的优点。

1.  至少列举一个使用 C++流相对于标准 C 风格的`printf`/`scanf`的缺点。

1.  何时应该使用`std::endl`而不是`\n`？

1.  `std::cerr`和`std::clog`之间有什么区别，何时应该使用`std::cerr`？

1.  如何在基数标识符和十六进制值之间输出额外字符？

1.  如何输出八进制和大写字母？

1.  如何使用 C++和 GSL 安全地解析标准 C 风格的程序参数？

1.  如何保存/恢复`std::cin`的读取缓冲区？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
