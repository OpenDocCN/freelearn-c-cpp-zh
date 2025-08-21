# 第四章：C++，RAII 和 GSL 复习

在本章中，我们将概述本书中利用的 C++的一些最新进展。我们将首先概述 C++17 规范中对 C++所做的更改。然后我们将简要介绍一种名为**资源获取即初始化**（**RAII**）的 C++设计模式，以及它在 C++中的使用方式以及为什么它对 C++以及许多其他利用相同设计模式的语言如此重要。本章将以介绍**指导支持库**（**GSL**）并讨论它如何通过帮助遵守 C++核心指南来增加系统编程的可靠性和稳定性而结束。

在本章中，我们将涵盖以下主题：

+   讨论 C++17 中的进展

+   概述 RAII

+   介绍 GSL

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请转到以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter04`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter04)。

# C++17 的简要概述

本节的目标是简要概述 C++17 和添加到 C++的功能。要了解更全面和深入的 C++17，请参阅本章的*进一步阅读*部分，其中列出了 Packt Publishing 关于该主题的其他书籍。

# 语言变化

C++17 语言和语法进行了几处更改。以下是一些示例。

# if/switch 语句中的初始化器

在 C++17 中，现在可以在`if`和`switch`语句的定义中定义变量并初始化，如下所示：

```cpp
#include <iostream>

int main(void)
{
    if (auto i = 42; i > 0) {
        std::cout << "Hello World\n";
    }
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的示例中，`i`变量在`if`语句内部使用分号（`;`）进行定义和初始化。这对于返回错误代码的 C 和 POSIX 风格函数特别有用，因为存储错误代码的变量可以在适当的上下文中定义。

这个特性如此重要和有用的原因在于只有在条件满足时才定义变量。也就是说，在前面的示例中，只有当`i`大于`0`时，`i`才存在。

这对确保变量在有效时可用非常有帮助，有助于减少使用无效变量的可能性。

`switch`语句可以发生相同类型的初始化，如下所示：

```cpp
#include <iostream>

int main(void)
{
    switch(auto i = 42) {
        case 42:
            std::cout << "Hello World\n";
            break;

        default:
            break;
    }
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的示例中，`i`变量仅在`switch`语句的上下文中创建。与`if`语句不同，`i`变量存在于所有情况下，这意味着`i`变量在`default`状态中可用，这可能代表无效状态。

# 增加编译时设施

在 C++11 中，`constexpr`被添加为一种声明，告诉编译器变量、函数等可以在编译时进行评估和优化，从而减少运行时代码的复杂性并提高整体性能。在某些情况下，编译器足够聪明，可以将`constexpr`语句扩展到其他组件，包括分支语句，例如：

```cpp
#include <iostream>

constexpr const auto val = true;

int main(void)
{
    if (val) {
        std::cout << "Hello World\n";
    }
}
```

在这个例子中，我们创建了一个`constexpr`变量，并且只有在`constexpr`为`true`时才将`Hello World`输出到`stdout`。由于在这个例子中它总是为真，编译器将完全从代码中删除该分支，如下所示：

```cpp
push %rbp
mov %rsp,%rbp
lea 0x100(%rip),%rsi
lea 0x200814(%rip),%rdi
callq 6c0 <...cout...>
mov $0x0,%eax
pop %rbp
retq
```

正如你所看到的，代码加载了一些寄存器并调用`std::cout`，而没有检查`val`是否为真，因为编译器完全从生成的二进制代码中删除了该代码。C++11 的问题在于作者可能会假设这种类型的优化正在进行，而实际上可能并没有。

为了防止这种类型的错误，C++17 添加了`constexpr` `if`语句，告诉编译器在编译时特别优化分支。如果编译器无法优化`if`语句，将会发生显式的编译时错误，告诉用户无法进行优化，为用户提供修复问题的机会（而不是假设优化正在进行，实际上可能并没有进行），例如：

```cpp
#include <iostream>

int main(void)
{
    if constexpr (constexpr const auto i = 42; i > 0) {
        std::cout << "Hello World\n";
    }
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的例子中，我们有一个更复杂的`if`语句，它利用了编译时的`constexpr`优化以及`if`语句的初始化器。生成的二进制代码如下：

```cpp
push %rbp
mov %rsp,%rbp
sub $0x10,%rsp
movl $0x2a,-0x4(%rbp)
lea 0x104(%rip),%rsi 
lea 0x200809(%rip),%rdi 
callq 6c0 <...cout...>
mov $0x0,%eax
leaveq
retq
```

可以看到，结果的二进制代码中已经移除了分支，更具体地说，如果表达式不是常量，编译器会抛出一个错误，说明这段代码无法按照所述进行编译。

应该注意到，这个结果并不是之前的相同二进制代码，可能会有人期望的那样。似乎 GCC 7.3 在其优化引擎中还有一些额外的改进，因为在这段代码中定义和初始化的`constexpr` `i`变量没有被移除（当代码中并不需要为`i`分配栈空间时）。

另一个编译时的变化是`static_assert`编译时函数的不同版本。在 C++11 中，添加了以下内容：

```cpp
#include <iostream>

int main(void)
{
    static_assert(42 == 42, "the answer");
}

// > g++ scratchpad.cpp; ./a.out
// 
```

`static_assert`函数的目标是确保某些编译时的假设是正确的。当编写系统时，这是特别有帮助的，比如确保一个结构体的大小是特定的字节数，或者根据你正在编译的系统来确保某个代码路径被执行。这个断言的问题在于它需要添加一个在编译时输出的描述，这个描述可能只是用英语描述了断言而没有提供任何额外的信息。在 C++17 中，添加了另一个版本的这个断言，它去掉了对描述的需求，如下所示：

```cpp
#include <iostream>

int main(void)
{
    static_assert(42 == 42);
}

// > g++ scratchpad.cpp; ./a.out
//
```

# 命名空间

C++17 中一个受欢迎的变化是添加了嵌套命名空间。在 C++17 之前，嵌套命名空间必须在不同的行上定义，如下所示：

```cpp
#include <iostream>

namespace X 
{
namespace Y
{
namespace Z 
{
    auto msg = "Hello World\n";
}
}
}

int main(void)
{
    std::cout << X::Y::Z::msg;
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的例子中，我们定义了一个在嵌套命名空间中输出到`stdout`的消息。这种语法的问题是显而易见的——它占用了大量的空间。在 C++17 中，通过在同一行上声明嵌套命名空间来消除了这个限制，如下所示：

```cpp
#include <iostream>

namespace X::Y::Z 
{
    auto msg = "Hello World\n";
}

int main(void)
{
    std::cout << X::Y::Z::msg;
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的例子中，我们能够定义一个嵌套的命名空间，而不需要单独的行。

# 结构化绑定

我对 C++17 的一个最喜欢的新增功能是**结构化绑定**。在 C++17 之前，复杂的结构，比如结构体或`std::pair`，可以用来作为函数输出的多个值，但语法很繁琐，例如：

```cpp
#include <utility>
#include <iostream>

std::pair<const char *, int>
give_me_a_pair()
{
    return {"The answer is: ", 42};
}

int main(void)
{
    auto p = give_me_a_pair();
    std::cout << std::get<0>(p) << std::get<1>(p) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// The answer is: 42
```

在前面的例子中，`give_me_a_pair()`函数返回一个带有`The answer is:`字符串和整数`42`的`std::pair`。这个函数的结果存储在`main`函数中的一个名为`p`的变量中，需要使用`std::get()`来获取`std::pair`的第一部分和第二部分。这段代码在没有进行积极的优化时既笨拙又低效，因为需要额外的函数调用来获取`give_me_a_pair()`的结果。

在 C++17 中，结构化绑定为我们提供了一种检索结构体或`std::pair`的各个字段的方法，如下所示：

```cpp
#include <iostream>

std::pair<const char *, int>
give_me_a_pair()
{
    return {"The answer is: ", 42};
}

int main(void)
{
    auto [msg, answer] = give_me_a_pair();
    std::cout << msg << answer << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// The answer is: 42
```

在前面的例子中，`give_me_a_pair()`函数返回与之前相同的`std::pair`，但这次我们使用了结构化绑定来获取`give_me_a_pair()`的结果。`msg`和`answer`变量被初始化为`std::pair`的结果，为我们提供了直接访问结果的方式，而不需要使用`std::get()`。

同样的也适用于结构体，如下所示：

```cpp
#include <iostream>

struct mystruct
{
    const char *msg;
    int answer;
};

mystruct
give_me_a_struct()
{
    return {"The answer is: ", 42};
}

int main(void)
{
    auto [msg, answer] = give_me_a_struct();
    std::cout << msg << answer << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// The answer is: 42
```

在前面的示例中，我们创建了一个由`give_me_a_struct()`返回的结构。使用结构化绑定获取此函数的结果，而不是使用`std::get()`。

# 内联变量

C++17 中更具争议的一个新增功能是内联变量的包含。随着时间的推移，越来越多的仅头文件库由 C++社区的各个成员开发。这些库提供了在 C++中提供复杂功能的能力，而无需安装和链接到库（只需包含库即可）。这些类型的库的问题在于它们必须在库本身中使用花哨的技巧来包含全局变量。

内联变量解决了这个问题，如下所示：

```cpp
#include <iostream>

inline auto msg = "Hello World\n";

int main(void)
{
    std::cout << msg;
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的示例中，`msg`变量被声明为`inline`。这种类型的变量可以在头文件（即`.h`文件）中定义，并且可以多次包含而不会在链接期间定义多个定义。值得注意的是，内联变量还消除了对以下内容的需求：

```cpp
extern const char *msg;
```

通常，多个源文件需要一个全局变量，并且使用前述模式将变量暴露给所有这些源文件。前面的代码添加到一个由所有源文件包含的头文件中，然后一个源文件实际上定义变量，例如：

```cpp
const char *msg = "Hello World\n";
```

尽管这种方法有效，但它很麻烦，而且并不总是清楚哪个源文件实际上应该定义变量。使用内联变量可以解决这个问题，因为头文件既定义了变量，又将符号暴露给所有需要它的源文件，消除了歧义。

# 库的更改

除了对语言语法的更改，还对库进行了一些更改。以下是一些显著的更改。

# 字符串视图

正如本章的*GSL*部分将讨论的那样，C++社区内部正在推动消除对指针和数组的直接访问。在应用程序中发现的大多数段错误和漏洞都可以归因于对指针和数组的处理不当。随着程序变得越来越复杂，并由多人修改而没有完整了解应用程序及其如何使用每个指针和/或数组的情况，引入错误的可能性也会增加。

为了解决这个问题，C++社区已经采纳了 C++核心指南：[`github.com/isocpp/CppCoreGuidelines`](https://github.com/isocpp/CppCoreGuidelines)。

C++核心指南的目标是定义一组最佳实践，以帮助防止在使用 C++编程时出现的常见错误，以限制引入程序的总错误数量。 C++已经存在多年了，尽管它有很多设施来防止错误，但它仍然保持向后兼容性，允许旧程序与新程序共存。 C++核心指南帮助新用户和专家用户浏览可用的许多功能，以帮助创建更安全和更健壮的应用程序。

C++17 中为支持这一努力添加的一个功能是`std::string_view{}`类。`std::string_view`是字符数组的包装器，类似于`std::array`，有助于使使用基本 C 字符串更安全和更容易，例如：

```cpp
#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World\n");
    std::cout << str;
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

在前面的示例中，我们创建了`std::string_view{}`并将其初始化为 ASCII C 字符串。然后使用`std::cout`将字符串输出到`stdout`。与`std::array`一样，`std::string_view{}`提供了对基础数组的访问器，如下所示：

```cpp
#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World");

    std::cout << str.front() << '\n';
    std::cout << str.back() << '\n';
    std::cout << str.at(1) << '\n';
    std::cout << str.data() << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// H
// d
// e
// Hello World
```

在上面的例子中，`front()`和`back()`函数可用于获取字符串中的第一个和最后一个字符，而`at()`函数可用于获取字符串中的任何字符；如果索引超出范围（即，提供给`at()`的索引比字符串本身还长），则会抛出`std::out_of_range{}`异常。最后，`data()`函数可用于直接访问底层数组。不过，应谨慎使用此函数，因为其使用会抵消`std::string_view{}`的安全性好处。

除了访问器之外，`std::string_view{}`类还提供了有关字符串大小的信息：

```cpp
#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World");

    std::cout << str.size() << '\n';
    std::cout << str.max_size() << '\n';
    std::cout << str.empty() << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 11
// 4611686018427387899
// 0
```

在上面的例子中，`size()`函数返回字符串中的字符总数，而`empty()`函数在`size() == 0`时返回`true`，否则返回`false`。`max_size()`函数定义了`std::string_view{}`可以容纳的最大大小，在大多数情况下是无法实现或现实的。在上面的例子中，最大字符串大小超过一百万兆字节。

与`std::array`不同，`std::string_view{}`提供了通过从字符串的前面或后面删除字符来减小字符串视图的能力，如下所示：

```cpp
#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World");

    str.remove_prefix(1);
    str.remove_suffix(1);
    std::cout << str << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// ello Worl
```

在上面的例子中，`remove_prefix()`和`remove_suffix()`函数用于从字符串的前面和后面各删除一个字符，结果是将`ello Worl`输出到`stdout`。需要注意的是，这只是改变了起始字符并重新定位了结束的空字符指针，而无需重新分配内存。对于更高级的功能，应该使用`std::string{}`，但这会带来额外的内存分配性能损失。

也可以按如下方式访问子字符串：

```cpp
#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World");
    std::cout << str.substr(0, 5) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// Hello
```

在上面的例子中，我们使用`substr()`函数访问`Hello`子字符串。

也可以比较字符串：

```cpp
#if SNIPPET13

#include <iostream>
#include <string_view>

int main(void)
{
    std::string_view str("Hello World");

    if (str.compare("Hello World") == 0) {
        std::cout << "Hello World\n";
    }

    std::cout << str.compare("Hello") << '\n';
    std::cout << str.compare("World") << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
// 6
// -1
```

与`strcmp()`函数类似，比较函数在比较两个字符串时返回`0`，而它们不同时返回差异。

最后，搜索函数如下所示：

```cpp
#include <iostream>

int main(void)
{
    std::string_view str("Hello this is a test of Hello World");

    std::cout << str.find("Hello") << '\n';
    std::cout << str.rfind("Hello") << '\n';
    std::cout << str.find_first_of("Hello") << '\n';
    std::cout << str.find_last_of("Hello") << '\n';
    std::cout << str.find_first_not_of("Hello") << '\n';
    std::cout << str.find_last_not_of("Hello") << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 0
// 24
// 0
// 33
// 5
// 34
```

这个例子的结果如下：

+   `find()`函数返回字符串中第一次出现`Hello`的位置，这种情况下是`0`。

+   `rfind()`返回提供的字符串的最后出现位置，在这种情况下是`24`。

+   `find_first_of()`和`find_last_of()`找到提供的任何字符的第一个和最后一个出现位置（而不是整个字符串）。在这种情况下，`H`在提供的字符串中，而`H`是`msg`中的第一个字符，这意味着`find_first_of()`返回`0`，因为`0`是字符串中的第一个索引。

+   在`find_last_of()`中，`l`是最后出现的字母，位置在`33`。

+   `find_first_not_of()`和`find_last_not_of()`是`find_first_of()`和`find_last_of()`的相反，返回提供的字符串中任何字符的第一个和最后一个出现位置。

# std::any，std::variant 和 std::optional

C++17 中的其他受欢迎的新增功能是`std::any{}`，`std::variant{}`和`std::optional{}`类。`std::any{}`能够随时存储任何值。需要特殊的访问器来检索`std::any{}`中的数据，但它们能够以类型安全的方式保存任何值。为了实现这一点，`std::any{}`利用了内部指针，并且每次更改类型时都必须分配内存，例如：

```cpp
#include <iostream>
#include <any>

struct mystruct {
    int data;
};

int main(void)
{
    auto myany = std::make_any<int>(42);
    std::cout << std::any_cast<int>(myany) << '\n';

    myany = 4.2;
    std::cout << std::any_cast<double>(myany) << '\n';

    myany = mystruct{42};
    std::cout << std::any_cast<mystruct>(myany).data << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 42
// 4.2
// 42
```

在上面的例子中，我们创建了`std::any{}`并将其设置为具有值`42`的`int`，具有值`4.2`的`double`，以及具有值`42`的`struct`。

`std::variant`更像是一个类型安全的联合。联合在编译时为联合中存储的所有类型保留存储空间（因此不需要分配，但是所有可能的类型必须在编译时已知）。标准 C 联合的问题在于无法知道任何给定时间存储的是什么类型。同时存储 int 和`double`是有问题的，因为同时使用两者会导致损坏。使用`std::variant`可以避免这种问题，因为`std::variant`知道它当前存储的是什么类型，并且不允许尝试以不同类型访问数据（因此，`std::variant`是类型安全的），例如：

```cpp
#include <iostream>
#include <variant>

int main(void)
{
    std::variant<int, double> v = 42;
    std::cout << std::get<int>(v) << '\n';

    v = 4.2;
    std::cout << std::get<double>(v) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 42
// 4.2
```

在前面的例子中，`std::variant`被用来存储`integer`和`double`，我们可以安全地从`std::variant`中检索数据而不会损坏。

`std::optional`是一个可空的值类型。指针是一个可空的引用类型，其中指针要么无效，要么有效并存储一个值。要创建一个指针值，必须分配内存（或者至少指向内存）。`std::optional`是一个值类型，这意味着不需要为`std::optional`分配内存，并且在底层，只有在可选项有效时才执行构造，消除了在实际未设置时构造默认值类型的开销。对于复杂对象，这不仅提供了确定对象是否有效的能力，还允许我们在无效情况下跳过构造，从而提高性能，例如：

```cpp
#include <iostream>
#include <optional>

class myclass
{
public:
    int val;

    myclass(int v) :
        val{v}
    {
        std::cout << "constructed\n";
    }
};

int main(void)
{
    std::optional<myclass> o;
    std::cout << "created, but not constructed\n";

    if (o) {
        std::cout << "Attempt #1: " << o->val << '\n';
    }

    o = myclass{42};

    if (o) {
        std::cout << "Attempt #2: " << o->val << '\n';
    }
}

// > g++ scratchpad.cpp; ./a.out
// created, but not constructed
// constructed
// Attempt #2: 42
```

在前面的例子中，我们创建了一个简单的类，用于存储一个`integer`。在这个类中，当类被构造时，我们向 stdout 输出一个字符串。然后我们使用`std::optional`创建了这个类的一个实例。我们尝试在实际设置类为有效值之前和之后访问这个`std::optional`。如所示，只有在我们实际设置类为有效值之后，类才被构造。由于`sts::unique_ptr`曾经是创建 optionals 的常用方法，因此`std::optional`共享一个常用的接口并不奇怪。

# 资源获取即初始化（RAII）

RAII 可以说是 C 和 C++之间最显著的区别之一。RAII 为整个 C++库奠定了基础和设计模式，并且已经成为无数其他语言的灵感之源。这个简单的概念为 C++提供了无与伦比的安全性，与 C 相比，这个概念将在本书中被充分利用，当 C 和 POSIX 必须用于替代 C++时（例如，当 C++的替代方案要么不存在，要么不完整时）。

RAII 的理念很简单。如果分配了资源，它是在对象构造期间分配的，当对象被销毁时，资源被释放。为了实现这一点，RAII 利用了 C++的构造和销毁特性，例如：

```cpp
#include <iostream>

class myclass
{
public:
    myclass()
    {
        std::cout << "Hello from constructor\n";
    }

    ~myclass()
    {
        std::cout << "Hello from destructor\n";
    }
};

int main(void)
{
    myclass c;
}

// > g++ scratchpad.cpp; ./a.out
// Hello from constructor
// Hello from destructor
```

在前面的例子中，我们创建了一个在构造和销毁时向`stdout`输出的类。如所示，当类被实例化时，类被构造，当类失去焦点时，类被销毁。

这个简单的概念可以用来保护资源，如下所示：

```cpp
#include <iostream>

class myclass
{
    int *ptr;

public:
    myclass() :
        ptr{new int(42)}
    { }

    ~myclass()
    {
        delete ptr;
    }

    int get()
    {
        return *ptr;
    }
};

int main(void)
{
    myclass c;
    std::cout << "The answer is: " << c.get() << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// The answer is: 42
```

在前面的例子中，当`myclass{}`被构造时，分配了一个指针，并且当`myclass{}`被销毁时，指针被释放。这种模式提供了许多优势：

+   只要`myclass{}`的实例可见（即可访问），指针就是有效的。因此，任何尝试访问类中的内存都是安全的，因为只有在类的范围丢失时才会释放内存，这将导致无法访问类（假设没有使用指向类的指针和引用）。

+   不会发生内存泄漏。如果类可见，类分配的内存将是有效的。一旦类不再可见（即失去范围），内存就会被释放，不会发生泄漏。

具体来说，RAII 确保在对象初始化时获取资源，并在不再需要对象时释放资源。正如稍后将在第七章中展示的那样，`std::unique_ptr[]`和`std::shared_ptr{}`利用了这种精确的设计模式（尽管，这些类不仅仅是上面的例子，还要求在获取资源的同时确保所有权）。

RAII 不仅适用于指针；它可以用于必须获取然后释放的任何资源，例如：

```cpp
#include <iostream>

class myclass
{
    FILE *m_file;

public:
    myclass(const char *filename) :
        m_file{fopen(filename, "rb")}
    {
        if (m_file == 0) {
            throw std::runtime_error("unable to open file");
        }
    }

    ~myclass()
    {
        fclose(m_file);
        std::clog << "Hello from destructor\n";
    }
};

int main(void)
{
    myclass c1("test.txt");

    try {
        myclass c2("does_not_exist.txt");
    }
    catch(const std::exception &e) {
        std::cout << "exception: " << e.what() << '\n';
    }
}

// > g++ scratchpad.cpp; touch test.txt; ./a.out
// exception: unable to open file
// Hello from destructor
```

在前面的例子中，我们创建了一个在构造时打开文件并存储其句柄，然后在销毁时关闭文件并释放句柄的类。在主函数中，我们创建了一个类的实例，它既被构造又被正常销毁，利用 RAII 来防止文件泄漏。

除了正常情况外，我们创建了第二个类，试图打开一个不存在的文件。在这种情况下，会抛出异常。这里需要注意的重要一点是，对于这个第二个实例，析构函数不会被调用。这是因为构造失败并抛出了异常。因此，没有获取资源，因此也不需要销毁。也就是说，资源的获取直接与类本身的初始化相关联，而安全地构造类可以防止销毁从未分配的资源。

RAII 是 C++的一个简单而强大的特性，在 C++中被广泛利用，这种设计模式将在本书中进行扩展。

# 指导支持库（GSL）

如前所述，C++核心指南的目标是提供与 C++编程相关的最佳实践。GSL 是一个旨在帮助遵守这些指南的库。总的来说，GSL 有一些整体主题：

+   **指针所有权**：定义谁拥有指针是防止内存泄漏和指针损坏的简单方法。一般来说，定义所有权的最佳方法是通过使用`std::unique_ptr{}`和`std::shared_ptr{}`，这将在第七章中深入解释，但在某些情况下，这些不能使用，GSL 有助于处理这些边缘情况。

+   **期望管理**：GSL 还有助于定义函数对输入的期望和对输出的保证，目标是将这些概念转换为 C++合同。

+   **没有指针算术**：指针算术是程序不稳定和易受攻击的主要原因之一。消除指针算术（或者至少将指针算术限制在经过充分测试的支持库中）是消除这些问题的简单方法。

# 指针所有权

经典的 C++不区分谁拥有指针（即负责释放与指针关联的内存的代码或对象）和谁只是使用指针访问内存，例如：

```cpp
void init(int *p)
{
    *p = 0;
}

int main(void)
{
    auto p = new int;
    init(p);
    delete p;
}

// > g++ scratchpad.cpp; ./a.out
//
```

在前面的例子中，我们分配了一个指向整数的指针，然后将该指针传递给一个名为`init()`的函数，该函数初始化指针。最后，在`init()`函数使用完指针后，我们删除了指针。如果`init()`函数位于另一个文件中，就不清楚`init()`函数是否应该删除指针。尽管在这个简单的例子中，这可能是显而易见的，但在有大量代码的复杂项目中，这种意图可能会丢失。对这样的代码进行未来修改可能会导致使用未定义所有权的指针。

为了克服这一点，GSL 提供了一个`gsl::owner<>`修饰，用于记录给定变量是否是指针的所有者，例如：

```cpp
#include <gsl/gsl>

void init(int *p)
{
    *p = 0;
}

int main(void)
{
    gsl::owner<int *> p = new int;
    init(p);
    delete p;
}

// > g++ scratchpad.cpp; ./a.out
//
```

在前面的例子中，我们记录了`main`函数中的`p`是指针的所有者，这意味着一旦`p`不再需要，指针应该被释放。前面例子中的另一个问题是`init()`函数期望指针不为空。如果指针为空，将发生空指针解引用。

有两种常见的方法可以克服空指针解引用的可能性。第一种选择是检查`nullptr`并抛出异常。这种方法的问题在于你必须在每个函数上执行这个空指针检查。这些类型的检查成本高，而且会使代码混乱。另一个选择是使用`gsl::not_null<>{}`类。像`gsl::owner<>{}`一样，`gsl::not_null<>{}`是一个装饰，可以在不使用调试时从代码中编译出来。然而，如果启用了调试，`gsl::not_null<>{}`将抛出异常，`abort()`，或者在某些情况下，如果变量设置为 null，拒绝编译。使用`gsl::not_null<>{}`，函数可以明确说明是否允许和安全处理空指针，例如：

```cpp
#include <gsl/gsl>

gsl::not_null<int *>
test(gsl::not_null<int *> p)
{
    return p;
}

int main(void)
{
    auto p1 = std::make_unique<int>();
    auto p2 = test(gsl::not_null(p1.get()));
}

// > g++ scratchpad.cpp; ./a.out
//
```

在前面的例子中，我们使用`std::unique_ptr{}`创建了一个指针，然后将得到的指针传递给一个名为`test()`的函数。`test()`函数不支持空指针，因此使用`gsl::not_null<>{}`来表示这一点。反过来，`test()`函数返回`gsl::not_null<>{}`，告诉用户`test()`函数确保函数的结果不为空（这也是为什么`test`函数一开始不支持空指针的原因）。

# 指针算术

**指针算术**是导致不稳定和易受攻击的常见错误源。因此，C++核心指南不鼓励使用这种类型的算术。以下是一些指针算术的例子：

```cpp
int array[10];

auto r1 = array + 1;
auto r2 = *(array + 1);
auto r3 = array[1];
```

最后一个例子可能是最令人惊讶的。下标运算符实际上是指针算术，其使用可能导致越界错误。为了克服这一点，GSL 提供了`gsl::span{}`类，为我们提供了一个安全的接口，用于处理指针，包括数组，例如：

```cpp
#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl>
#include <iostream>

int main(void)
{
    int array[5] = {1, 2, 3, 4, 5};
    auto span = gsl::span(array);

    for (const auto &elem : span) {
        std::clog << elem << '\n';
    }

    for (auto i = 0; i < 5; i++) {
        std::clog << span[i] << '\n';
    }

    try {
        std::clog << span[5] << '\n';
    }
    catch(const gsl::fail_fast &e) {
        std::cout << "exception: " << e.what() << '\n';
    }
}

// > g++ scratchpad.cpp; ./a.out
// 1
// 2
// 3
// 4
// 5
// 1
// 2
// 3
// 4
// 5
// exception: GSL: Precondition failure at ...
```

让我们看看前面的例子是如何工作的：

1.  我们创建一个数组，并用一组整数初始化它。

1.  我们创建一个 span，以便可以安全地与数组交互。我们使用基于范围的`for`循环（因为 span 包括一个迭代器接口）将数组输出到`stdout`。

1.  我们使用传统的索引和下标运算符（即`[]`运算符）将数组第二次输出到`stdout`。这个下标运算符的不同之处在于每个数组访问都会检查是否越界。为了证明这一点，我们尝试访问数组越界，`gsl::span{}`抛出了一个`gsl::fail_fast{}`异常。应该注意的是，`GSL_THROW_ON_CONTRACT_VIOLATION`用于告诉 GSL 抛出异常，而不是执行`std::terminate`或完全忽略边界检查。

除了`gsl::span{}`之外，GSL 还包含`gsl::span{}`的特殊化，这些特殊化在处理常见类型的数组时对我们有所帮助。例如，GSL 提供了`gsl::cstring_span{}`，如下所示：

```cpp
#include <gsl/gsl>
#include <iostream>

int main(void)
{
    gsl::cstring_span<> str = gsl::ensure_z("Hello World\n");
    std::cout << str.data();

    for (const auto &elem : str) {
        std::clog << elem;
    }
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
// Hello World
```

`gsl::cstring_span{}`是一个包含标准 C 风格字符串的`gsl::span{}`。在前面的例子中，我们使用`gsl::ensure_z()`函数将`gsl::cstring_span{}`加载到标准 C 风格字符串中，以确保字符串在继续之前以空字符结尾。然后我们使用常规的`std::cout`调用和使用基于范围的循环输出标准 C 风格字符串。

# 合同

C++合同为用户提供了一种说明函数期望的输入以及函数确保的输出的方法。具体来说，C++合同记录了 API 的作者和 API 的用户之间的合同，并提供了对该合同的编译时和运行时验证。

未来的 C++版本将内置支持合同，但在此之前，GSL 通过提供`expects()`和`ensures()`宏的库实现了 C++合同，例如：

```cpp
#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl>
#include <iostream>

int main(void)
{
    try {
        Expects(false);
    }
    catch(const gsl::fail_fast &e) {
        std::cout << "exception: " << e.what() << '\n';
    }
}

// > g++ scratchpad.cpp; ./a.out
// exception: GSL: Precondition failure at ...
```

在前面的例子中，我们使用`Expects()`宏并将其传递为`false`。与标准 C 库提供的`assert()`函数不同，`Expects()`宏在`false`时失败。与`assert()`不同，即使在禁用调试时，如果传递给`Expects()`的表达式求值为`false`，`Expects()`也将执行`std::terminate()`。在前面的例子中，我们声明`Expects()`应该抛出`gsl::fail_fast{}`异常，而不是执行`std::terminate()`。

`Ensures()`宏与`Expects()`相同，唯一的区别是名称，用于记录合同的输出而不是输入，例如：

```cpp
#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl>
#include <iostream>

int
test(int i)
{
    Expects(i >= 0 && i < 41);
    i++;

    Ensures(i < 42);
    return i;
}

int main(void)
{
    test(0);

    try {
        test(42);
    }
    catch(const gsl::fail_fast &e) {
        std::cout << "exception: " << e.what() << '\n';
    }
}

// > g++ scratchpad.cpp; ./a.out
// exception: GSL: Precondition failure at ...
```

在前面的例子中，我们创建了一个函数，该函数期望输入大于或等于`0`且小于`41`。然后函数对输入进行操作，并确保结果输出始终小于`42`。一个正确编写的函数将定义其期望，以便`Ensures()`宏永远不会触发。相反，如果输入导致输出违反合同，则`Expects()`检查可能会触发。

# 实用程序

GSL 还提供了一些有用的辅助工具，有助于创建更可靠和可读的代码。其中一个例子是`gsl::finally{}`API，如下：

```cpp
#define concat1(a,b) a ## b
#define concat2(a,b) concat1(a,b)
#define ___ concat2(dont_care, __COUNTER__)

#include <gsl/gsl>
#include <iostream>

int main(void)
{
    auto ___ = gsl::finally([]{
        std::cout << "Hello World\n";
    });
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

`gsl::finally{}`提供了一种简单的方法，在函数退出之前执行代码，利用 C++析构函数。当函数必须在退出之前执行清理时，这是有帮助的。应该注意，`gsl::finally{}`在存在异常时最有用。通常，当触发异常时，清理代码被遗忘，导致清理逻辑永远不会执行。`gsl::finally{}` API 将始终执行，即使发生异常，只要它在执行可能生成异常的操作之前定义。

在前面的代码中，我们还包括了一个有用的宏，允许使用`___`来定义要使用的`gsl::finally{}`的名称。具体来说，`gsl::finally{}`的用户必须存储`gsl::finally{}`对象的实例，以便在退出函数时销毁该对象，但是必须命名`gsl::finally{}`对象是繁琐且无意义的，因为没有 API 与`gsl::finally{}`对象交互（它的唯一目的是在`exit`时执行）。这个宏提供了一种简单的方式来表达，“我不在乎变量的名称是什么”。

GSL 提供的其他实用程序包括`gsl::narrow<>()`和`gsl::narrow_cast<>()`，例如：

```cpp
#include <gsl/gsl>
#include <iostream>

int main(void)
{
    uint64_t val = 42;

    auto val1 = gsl::narrow<uint32_t>(val);
    auto val2 = gsl::narrow_cast<uint32_t>(val);
}

// > g++ scratchpad.cpp; ./a.out
//
```

这两个 API 与常规的`static_cast<>()`相同，唯一的区别是`gsl::narrow<>()`执行溢出检查，而`gsl::narrow_cast<>()`只是`static_cast<>()`的同义词，用于记录整数的缩小（即将具有更多位的整数转换为具有较少位的整数）。

```cpp
#endif

#if SNIPPET30

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl>
#include <iostream>

int main(void)
{
    uint64_t val = 0xFFFFFFFFFFFFFFFF;

    try {
        gsl::narrow<uint32_t>(val);
    }
    catch(...) {
        std::cout << "narrow failed\n";
    }
}

// > g++ scratchpad.cpp; ./a.out
// narrow failed
```

在前面的例子中，我们尝试使用`gsl::narrow<>()`函数将 64 位整数转换为 32 位整数，该函数执行溢出检查。由于发生了溢出，抛出了异常。

# 总结

在本章中，我们概述了本书中使用的 C++的一些最新进展。我们从 C++17 规范中对 C++所做的更改开始。然后我们简要介绍了一个称为 RAII 的 C++设计模式，以及它如何被 C++使用。最后，我们介绍了 GSL 以及它如何通过帮助遵守 C++核心指南来增加系统编程的可靠性和稳定性。

在下一章中，我们将介绍 UNIX 特定的主题，如 UNIX 进程和信号，以及 System V 规范的全面概述，该规范用于定义如何在 Intel CPU 上为 UNIX 编写程序。

# 问题

1.  什么是结构化绑定？

1.  C++17 对嵌套命名空间做了哪些改变？

1.  C++17 对`static_assert()`函数做了哪些改变？

1.  什么是`if`语句的初始化器？

1.  RAII 代表什么？

1.  RAII 用于什么？

1.  `gsl::owner<>{}`有什么作用？

1.  `Expects()`和`Ensures()`的目的是什么？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
