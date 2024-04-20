# 第十章：收集平台和编译器信息

在本章中，我们将涵盖：

+   检测操作系统和编译器

+   检测 int128 的支持

+   检测和绕过禁用的 RTTI

+   使用更简单的方法编写元函数

+   减少代码大小并提高 C++11 中用户定义类型（UDTs）的性能

+   导出和导入函数和类的可移植方式

+   检测 Boost 版本并获取最新功能

# 介绍

不同的项目和公司有不同的编码要求。有些禁止异常或 RTTI，而有些禁止 C++11。如果您愿意编写可供广泛项目使用的可移植代码，那么这一章就是为您准备的。

想要尽可能快地编写代码并使用最新的 C++功能吗？您肯定需要一个工具来检测编译器功能。

一些编译器具有独特的功能，可以极大地简化您的生活。如果您只针对一个编译器，可以节省很多时间并使用这些功能。无需从头开始实现它们的类似物！

本章专门介绍了用于检测编译器、平台和 Boost 功能的不同辅助宏。这些宏广泛用于 Boost 库，并且对于编写能够使用任何编译器标志的可移植代码至关重要。

# 检测操作系统和编译器

我猜你可能见过很多丑陋的宏来检测代码编译的编译器。像这样的东西在 C 语言中是一种典型的做法：

```cpp
#include <something_that_defines_macros>
#if !defined(__clang__) \
    && !defined(__ICC) \
    && !defined(__INTEL_COMPILER) \
    && (defined(__GNUC__) || defined(__GNUG__))

// GCC specific

#endif
```

现在，试着想出一个好的宏来检测 GCC 编译器。尽量使宏的使用尽可能简短。

看一下以下的步骤来验证你的猜测。

# 准备工作

只需要基本的 C++知识。

# 如何做...

这个步骤很简单，只包括一个头文件和一个宏。

1.  头文件：

```cpp
#include <boost/predef/compiler.h>
```

1.  宏：

```cpp
#if BOOST_COMP_GNUC

// GCC specific

#endif
```

# 它是如何工作的...

头文件`<boost/predef/compiler.h>`知道所有可能的编译器，并为每个编译器都有一个宏。因此，如果当前编译器是 GCC，那么宏`BOOST_COMP_GNUC`被定义为`1`，而其他编译器的所有其他宏都被定义为`0`。如果我们不在 GCC 编译器上，那么`BOOST_COMP_GNUC`宏被定义为`0`。

通过这种方法，您无需检查宏本身是否已定义：

```cpp
#if defined(BOOST_COMP_GNUC) // Wrong!

// GCC specific

#endif
```

`Boost.Predef`库的宏总是被定义的，这样就不需要在`#ifdef`中输入`defined()`或`def`。

# 还有更多...

`Boost.Predef`库还有用于检测操作系统、架构、标准库实现和一些硬件能力的宏。使用总是被定义的宏的方法；这使您能够更简洁地编写复杂的表达式：

```cpp
#include <boost/predef/os.h>
#include <boost/predef/compiler.h>

#if BOOST_COMP_GNUC && BOOST_OS_LINUX && !BOOST_OS_ANDROID

// Do something for non Android Linux.

#endif
```

现在，最好的部分。`Boost.Predef`库可用于 C、C++和 Objective-C 编译器。如果您喜欢它，可以在非 C++项目中使用它。

C++17 没有`Boost.Predef`库的功能。

# 另请参阅

+   阅读`Boost.Predef`的官方文档，了解更多关于其在[`boost.org/libs/predef`](http://boost.org/libs/predef)的能力。

+   下一个步骤将向您介绍`Boost.Config`库，它的顺序更多，稍微不那么美观，但功能更加强大

# 检测 int128 的支持

一些编译器支持扩展算术类型，如 128 位浮点数或整数。让我们快速了解如何使用 Boost 来使用它们。

我们将创建一个接受三个参数并返回这些方法的乘积值的方法。如果编译器支持 128 位整数，那么我们就使用它们。如果编译器支持`long long`，那么我们就使用它；否则，我们需要发出编译时错误。

# 准备工作

只需要基本的 C++知识。

# 如何做...

我们需要什么来处理 128 位整数？显示它们可用的宏和一些`typedefs`以在各个平台上拥有可移植的类型名称。

1.  包括一个头文件：

```cpp
#include <boost/config.hpp>
```

1.  现在，我们需要检测 int128 的支持：

```cpp
#ifdef BOOST_HAS_INT128
```

1.  添加一些`typedefs`并按以下方式实现该方法：

```cpp
typedef boost::int128_type int_t;
typedef boost::uint128_type uint_t;

inline int_t mul(int_t v1, int_t v2, int_t v3) {
    return v1 * v2 * v3;
}
```

1.  对于不支持 int128 类型且没有`long long`的编译器，我们可能会产生编译时错误：

```cpp
#else // #ifdef BOOST_HAS_INT128

#ifdef BOOST_NO_LONG_LONG
#error "This code requires at least int64_t support"
#endif
```

1.  现在，我们需要为不支持 int128 的编译器使用`int64`提供一些实现：

```cpp
struct int_t { boost::long_long_type hi, lo; };
struct uint_t { boost::ulong_long_type hi, lo; };

inline int_t mul(int_t v1, int_t v2, int_t v3) {
    // Some hand written math.
    // ...
}

#endif // #ifdef BOOST_HAS_INT128
```

# 工作原理...

头文件`<boost/config.hpp>`包含许多宏来描述编译器和平台特性。在此示例中，我们使用`BOOST_HAS_INT128`来检测对 128 位整数的支持，使用`BOOST_NO_LONG_LONG`来检测对 64 位整数的支持。

正如我们从示例中看到的，Boost 具有 64 位有符号和无符号整数的`typedefs`：

```cpp
boost::long_long_type  
boost::ulong_long_type  
```

它还有 128 位有符号和无符号整数的`typedefs`：

```cpp
boost::int128_type 
boost::uint128_type 
```

# 还有更多...

C++11 通过`long long int`和`unsigned long long int`内置类型支持 64 位类型。不幸的是，并非所有编译器都支持 C++11，因此`BOOST_NO_LONG_LONG`可能对您有用。

128 位整数不是 C++17 的一部分，因此 Boost 中的`typedefs`和宏是编写可移植代码的一种方式。

C++标准化委员会正在进行工作，以添加编译时指定宽度的整数。当这项工作完成时，您将能够创建 128 位、512 位甚至 8388608 位（1 MB 大）的整数。

# 另请参阅

+   阅读有关“检测和绕过禁用的 RTTI”更多关于`Boost.Config`的信息。

+   阅读官方文档[`boost.org/libs/config`](http://boost.org/libs/config)以获取有关其功能的更多信息。

+   Boost 中有一个允许构造无限精度类型的库。点击链接[`boost.org/libs/multiprecision`](http://boost.org/libs/multiprecision)并查看`Boost.Multiprecision`库。

# 检测和绕过禁用的 RTTI

一些公司和库对其 C++代码有特定要求，例如成功编译而无需 RTTI。

在这个小配方中，我们不仅会检测禁用的 RTTI，还会从头开始编写一个类似 Boost 的库，用于存储类型信息，并在运行时比较类型，即使没有`typeid`。

# 准备工作

对于这个配方，需要基本的 C++ RTTI 使用知识。

# 如何做到...

检测禁用的 RTTI，存储类型信息，并在运行时比较类型是 Boost 库中广泛使用的技巧。

1.  为此，我们首先需要包含以下头文件：

```cpp
#include <boost/config.hpp> 
```

1.  让我们首先看一下启用了 RTTI 并且 C++11 的`std::type_index`类可用的情况：

```cpp
#if !defined(BOOST_NO_RTTI) \
    && !defined(BOOST_NO_CXX11_HDR_TYPEINDEX)

#include <typeindex>
using std::type_index;

template <class T>
type_index type_id() {
    return typeid(T);
}
```

1.  否则，我们需要构造自己的`type_index`类：

```cpp
#else

#include <cstring>
#include <iosfwd> // std::basic_ostream
#include <boost/current_function.hpp>

struct type_index {
    const char * name_;

    explicit type_index(const char* name)
        : name_(name)
    {}

    const char* name() const { return name_; }
};

inline bool operator == (type_index v1, type_index v2) {
    return !std::strcmp(v1.name_, v2.name_);
}

inline bool operator != (type_index v1, type_index v2) {
    return !(v1 == v2);
}
```

1.  最后一步是定义`type_id`函数：

```cpp
template <class T>
inline type_index type_id() {
    return type_index(BOOST_CURRENT_FUNCTION);
}

#endif
```

1.  现在，我们可以比较类型：

```cpp
#include <cassert>

int main() {
    assert(type_id<unsigned int>() == type_id<unsigned>());
    assert(type_id<double>() != type_id<long double>());
}
```

# 工作原理...

如果禁用了 RTTI，则宏`BOOST_NO_RTTI`将被定义，如果编译器没有`<typeindex>`头文件和没有`std::type_index`类，则宏`BOOST_NO_CXX11_HDR_TYPEINDEX`将被定义。

上一节*步骤 3*中手写的`type_index`结构只保存指向某个字符串的指针；这里没有什么真正有趣的东西。

看一下`BOOST_CURRENT_FUNCTION`宏。它返回当前函数的完整名称，包括模板参数、参数和返回类型。

例如，`type_id<double>()`表示如下：

```cpp
 type_index type_id() [with T = double]
```

因此，对于任何其他类型，`BOOST_CURRENT_FUNCTION`返回不同的字符串，这就是为什么示例中的`type_index`变量不等于它的原因。

恭喜！我们刚刚重新发明了大部分`Boost.TypeIndex`库的功能。删除*步骤 1 到 4*中的所有代码，并稍微更改*步骤 5*中的代码以使用`Boost.TypeIndex`库：

```cpp
#include <boost/type_index.hpp>

void test() {
    using boost::typeindex::type_id;

    assert(type_id<unsigned int>() == type_id<unsigned>());
    assert(type_id<double>() != type_id<long double>());
}
```

# 还有更多...

当然，`Boost.TypeIndex`略微超出了这个范围；它允许您以与平台无关的方式获取可读的类型名称，解决与平台相关的问题，允许发明自己的 RTTI 实现，拥有 constexpr RTTI 等等。

不同的编译器有不同的宏用于获取完整的函数名。使用 Boost 的宏是最通用的解决方案。`BOOST_CURRENT_FUNCTION`宏在编译时返回名称，因此它意味着最小的运行时惩罚。

C++11 有一个`__func__`魔术标识符，它被评估为当前函数的名称。然而，`__func__`的结果只是函数名，而`BOOST_CURRENT_FUNCTION`则努力显示函数参数，包括模板参数。

# 另请参阅

+   阅读即将发布的食谱，了解更多关于`Boost.Config`的信息

+   浏览[`github.com/boostorg/type_index`](http://github.com/boostorg/type_index)以查看`Boost.TypeIndex`库的源代码

+   阅读[`boost.org/libs/config`](http://boost.org/libs/config)上的`Boost.Config`的官方文档

+   阅读[`boost.org/libs/type_index`](http://boost.org/libs/type_index)上的`Boost.TypeIndex`库的官方文档

+   第一章的食谱*获取可读的类型名称*，*开始编写您的应用程序*将向您介绍`Boost.TypeIndex`的其他功能

# 使用更简单的方法编写元函数

第四章，*编译时技巧*，和第八章，*元编程*，都致力于元编程。如果您尝试使用这些章节中的技术，您可能已经注意到编写元函数可能需要很长时间。因此，在编写可移植实现之前，使用更用户友好的方法，如 C++11 的`constexpr`，进行元函数的实验可能是一个好主意。

在这个食谱中，我们将看看如何检测`constexpr`的支持。

# 准备就绪

`constexpr`函数是可以在编译时评估的函数。这就是我们需要了解的全部内容。

# 如何做...

让我们看看如何检测编译器对`constexpr`功能的支持：

1.  就像本章的其他食谱一样，我们从以下头文件开始：

```cpp
#include <boost/config.hpp> 
```

1.  编写`constexpr`函数：

```cpp
#if !defined(BOOST_NO_CXX11_CONSTEXPR) \
    && !defined(BOOST_NO_CXX11_HDR_ARRAY)

template <class T>
constexpr int get_size(const T& val) {
    return val.size() * sizeof(typename T::value_type);
}
```

1.  如果缺少 C++11 功能，则打印错误：

```cpp
#else
#error "This code requires C++11 constexpr and std::array"
#endif
```

1.  就是这样。现在，我们可以自由地编写以下代码：

```cpp
#include <array>

int main() {
    std::array<short, 5> arr;
    static_assert(get_size(arr) == 5 * sizeof(short), "");

    unsigned char data[get_size(arr)];
}
```

# 它是如何工作的...

当 C++11 的`constexpr`可用时，定义了`BOOST_NO_CXX11_CONSTEXPR`宏。

`constexpr`关键字告诉编译器，如果该函数的所有输入都是编译时常量，那么该函数可以在编译时评估。C++11 对`constexpr`函数的功能施加了许多限制。C++14 取消了一些限制。

当 C++11 的`std::array`类和`<array>`头文件可用时，定义了`BOOST_NO_CXX11_HDR_ARRAY`宏。

# 还有更多...

然而，对于`constexpr`，还有其他可用和有趣的宏，如下所示：

+   `BOOST_CONSTEXPR`宏扩展为`constexpr`或不扩展

+   `BOOST_CONSTEXPR_OR_CONST`宏扩展为`constexpr`或`const`

+   `BOOST_STATIC_CONSTEXPR`宏与`static BOOST_CONSTEXPR_OR_CONST`相同

使用这些宏，如果可用的话，可以编写利用 C++11 常量表达式特性的代码：

```cpp
template <class T, T Value> 
struct integral_constant { 
    BOOST_STATIC_CONSTEXPR T value = Value; 

    BOOST_CONSTEXPR operator T() const { 
        return this->value; 
    } 
}; 
```

现在，我们可以像下面的代码中所示使用`integral_constant`：

```cpp
char array[integral_constant<int, 10>()]; 
```

在示例中，调用`BOOST_CONSTEXPR operator T()`来获取数组大小。

C++11 的常量表达式可以提高编译速度和错误诊断信息。这是一个很好的功能。如果您的函数需要来自 C++14 的**relaxed constexpr**，那么您可以使用`BOOST_CXX14_CONSTEXPR`宏。如果放松的 constexpr 可用，则它扩展为`constexpr`，否则不扩展。

# 另请参阅

+   有关`constexpr`用法的更多信息，请阅读[`en.cppreference.com/w/cpp/language/constexpr`](http://en.cppreference.com/w/cpp/language/constexpr)

+   阅读官方文档`Boost.Config`，了解有关宏的更多信息[`boost.org/libs/config`](http://boost.org/libs/config)

# 减小 C++11 中用户定义类型（UDTs）的代码大小并提高性能

当标准库容器中使用**用户定义类型**（UDTs）时，C++11 具有非常具体的逻辑。一些容器仅在移动构造函数不抛出异常或不存在复制构造函数时才使用移动赋值和移动构造。

让我们看看如何确保编译器知道`move_nothrow`类具有不抛出异常的`move`赋值运算符和不抛出异常的`move`构造函数。

# 准备工作

本教程需要基本的 C++11 右值引用知识。对标准库容器的了解也会对你有所帮助。

# 如何做...

让我们看看如何使用 Boost 改进我们的 C++类。

1.  我们只需要使用`BOOST_NOEXCEPT`宏标记`move_nothrow`赋值运算符和`move_nothrow`构造函数：

```cpp
#include <boost/config.hpp>

class move_nothrow {
    // Some class class members go here.
    // ...

public:
    move_nothrow() BOOST_NOEXCEPT;
    move_nothrow(move_nothrow&&) BOOST_NOEXCEPT
        // Members initialization goes here.
        // ...
    {}

    move_nothrow& operator=(move_nothrow&&) BOOST_NOEXCEPT {
        // Implementation goes here.
        // ...
        return *this;
    }

    move_nothrow(const move_nothrow&);
    move_nothrow& operator=(const move_nothrow&);
};
```

1.  现在，我们可以在 C++11 中使用`std::vector`类而无需进行任何修改：

```cpp
#include <vector>

int main() {
    std::vector<move_nothrow> v(10);
    v.push_back(move_nothrow());
}
```

1.  如果我们从`move`构造函数中移除`BOOST_NOEXCEPT`，我们将收到以下错误，因为我们没有为复制构造函数提供定义：

```cpp
 undefined reference to `move_nothrow::move_nothrow(move_nothrow 
    const&)  
```

# 工作原理...

`BOOST_NOEXCEPT`宏在支持它的编译器上扩展为`noexcept`。标准库容器使用类型特征来检测构造函数是否抛出异常。类型特征主要基于`noexcept`说明符做出决定。

为什么没有`BOOST_NOEXCEPT`会出错？编译器的类型特征返回`move_nothrow`会抛出异常，因此`std::vector`尝试使用`move_nothrow`的复制构造函数，但该构造函数未定义。

# 还有更多...

`BOOST_NOEXCEPT`宏还可以减小二进制大小，无论`noexcept`函数或方法的定义是否在单独的源文件中。

```cpp
// In header file.
int foo() BOOST_NOEXCEPT; 

// In source file.
int foo() BOOST_NOEXCEPT { 
    return 0; 
} 
```

这是因为在后一种情况下，编译器知道函数不会抛出异常，因此无需生成处理异常的代码。

如果标记为`noexcept`的函数确实抛出异常，您的程序将在不调用已构造对象的析构函数的情况下终止。

# 另请参阅

+   有关`move`构造函数允许抛出异常以及容器必须移动对象的文档可在[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html)上找到。

+   阅读`Boost.Config`的官方文档，了解更多`BOOST_NOEXCEPT`的示例，例如 Boost 中存在的宏[`boost.org/libs/config`](http://boost.org/libs/config)

# 导出和导入函数和类的可移植方式

几乎所有现代语言都有制作库的能力，即一组具有明确定义接口的类和方法。C++也不例外。我们有两种类型的库：运行时（也称为共享或动态）和静态。但是，在 C++中编写库并不是一项简单的任务。不同的平台有不同的方法来描述必须从共享库中导出哪些符号。

让我们看看如何使用 Boost 以一种可移植的方式管理符号可见性。

# 准备工作

在本教程中，创建动态和静态库的经验可能会有所帮助。

# 如何做...

本教程的代码由两部分组成。第一部分是库本身。第二部分是使用该库的代码。这两部分都使用相同的头文件，在其中声明了库方法。使用 Boost 以一种可移植的方式管理符号可见性很简单，可以通过以下步骤完成：

1.  在头文件中，我们需要以下头文件的定义：

```cpp
#include <boost/config.hpp> 
```

1.  以下代码也必须添加到头文件中：

```cpp
#if defined(MY_LIBRARY_LINK_DYNAMIC)
#   if defined(MY_LIBRARY_COMPILATION)
#       define MY_LIBRARY_API BOOST_SYMBOL_EXPORT
#   else
#       define MY_LIBRARY_API BOOST_SYMBOL_IMPORT
#   endif
#else
#   define MY_LIBRARY_API
#endif
```

1.  现在，所有声明都必须使用`MY_LIBRARY_API`宏：

```cpp
int MY_LIBRARY_API foo();

class MY_LIBRARY_API bar {
public:
    /* ... */ 
    int meow() const;
};
```

1.  异常必须使用`BOOST_SYMBOL_VISIBLE`声明；否则，它们只能在使用库的代码中使用`catch(...)`捕获：

```cpp
#include <stdexcept>

struct BOOST_SYMBOL_VISIBLE bar_exception
    : public std::exception 
{};
```

1.  库源文件必须包括头文件：

```cpp
#define MY_LIBRARY_COMPILATION
#include "my_library.hpp"
```

1.  方法的定义也必须在库的源文件中：

```cpp
int MY_LIBRARY_API foo() {
    // Implementation goes here.
    // ...
    return 0;
}

int bar::meow() const {
    throw bar_exception();
}
```

1.  现在，我们可以像下面的代码一样使用库：

```cpp
#include "../06_A_my_library/my_library.hpp"
#include <cassert>

int main() {
    assert(foo() == 0);
    bar b;
    try {
        b.meow();
        assert(false);
    } catch (const bar_exception&) {}
}
```

# 它是如何工作的...

所有工作都在*步骤 2*中完成。在那里，我们定义了宏`MY_LIBRARY_API`，并将其应用于我们希望从库中导出的类和方法。在*步骤 2*中，我们检查了`MY_LIBRARY_LINK_DYNAMIC`。如果未定义，我们正在构建一个静态库，不需要定义`MY_LIBRARY_API`。

开发人员必须注意`MY_LIBRARY_LINK_DYNAMIC`！它不会自己定义。如果我们正在制作一个动态库，我们需要让我们的构建系统来定义它，

如果定义了`MY_LIBRARY_LINK_DYNAMIC`，我们正在构建一个运行时库，这就是解决方法的开始。作为开发人员，您必须告诉编译器我们现在正在向用户导出函数。用户必须告诉编译器他/她正在从库中导入方法。为了拥有一个单一的头文件，既可以用于导入也可以用于导出库，我们使用以下代码：

```cpp
#if defined(MY_LIBRARY_COMPILATION) 
#    define MY_LIBRARY_API BOOST_SYMBOL_EXPORT 
#else 
#    define MY_LIBRARY_API BOOST_SYMBOL_IMPORT 
#endif 
```

在导出库（或者说编译库）时，我们必须定义`MY_LIBRARY_COMPILATION`。这会导致`MY_LIBRARY_API`被定义为`BOOST_SYMBOL_EXPORT`。例如，参见*步骤 5*，在包含`my_library.hpp`之前我们定义了`MY_LIBRARY_COMPILATION`。如果未定义`MY_LIBRARY_COMPILATION`，则用户包含了头文件，而用户对该宏一无所知。如果用户包含了头文件，则必须从库中导入符号。

`BOOST_SYMBOL_VISIBLE`宏只能用于那些未导出但被 RTTI 使用的类。这类示例包括异常和使用`dynamic_cast`进行转换的类。

# 还有更多...

一些编译器默认导出所有符号，但提供了禁用此行为的标志。例如，Linux 上的 GCC 和 Clang 提供了`-fvisibility=hidden`。强烈建议使用这些标志，因为它可以导致更小的二进制文件大小，更快的动态库加载，以及更好的二进制逻辑结构。一些程序间优化在导出较少符号时可以表现更好。C++17 没有描述可见性的标准方式。希望有一天，C++中会出现一种可移植的可见性处理方式，但在那之前，我们必须使用 Boost 中的宏。

# 另请参阅

+   从头开始阅读本章，以获取更多关于`Boost.Config`使用的示例

+   请阅读`Boost.Config`的官方文档，以获取完整的`Boost.Config`宏列表及其描述，网址为[`boost.org/libs/config`](http://boost.org/libs/config)。

# 检测 Boost 版本并获取最新功能

Boost 正在积极开发，因此每个版本都包含新功能和库。一些人希望有针对不同版本的 Boost 编译的库，并且还想使用新版本的一些功能。

让我们看一下`boost::lexical_cast`的变更日志。根据它，Boost 1.53 有一个`lexical_cast(const CharType* chars, std::size_t count)`函数重载。我们这个示例的任务是为新版本的 Boost 使用该函数重载，并为旧版本解决缺少的函数重载。

# 准备工作

只需要基本的 C++和`Boost.LexicalCast`库的知识。

# 如何做...

好吧，我们需要做的就是获取有关 Boost 版本的信息，并使用它来编写最佳代码。这可以按以下步骤完成：

1.  我们需要包含包含 Boost 版本和`boost::lexical_cast`的头文件：

```cpp
#include <boost/version.hpp>
#include <boost/lexical_cast.hpp>
```

1.  如果可用，我们使用`Boost.LexicalCast`的新功能：

```cpp
#if (BOOST_VERSION >= 105200)

int to_int(const char* str, std::size_t length) {
    return boost::lexical_cast<int>(str, length);
}
```

1.  否则，我们需要先将数据复制到`std::string`中：

```cpp
#else

int to_int(const char* str, std::size_t length) {
    return boost::lexical_cast<int>(
        std::string(str, length)
    );
}

#endif
```

1.  现在，我们可以像这里展示的代码一样使用：

```cpp
#include <cassert>

int main() {
    assert(to_int("10000000", 3) == 100);
}
```

# 它是如何工作的...

`BOOST_VERSION`宏包含 Boost 版本，格式如下：主版本号为一个数字，次版本号为三个数字，修订级别为两个数字。例如，Boost 1.73.1 将在`BOOST_VERSION`宏中包含`107301`数字。

因此，在*步骤 2*中，我们检查 Boost 版本，并根据`Boost.LexicalCast`的能力选择`to_int`函数的正确实现。

# 还有更多...

拥有版本宏是大型库的常见做法。一些 Boost 库允许您指定要使用的库的版本；请参阅`Boost.Thread`及其`BOOST_THREAD_VERSION`宏以获取示例。

顺便说一句，C++也有一个版本宏。`__cplusplus`宏的值允许您区分 C++11 之前的版本和 C++11，C++11 和 C++14，或 C++17。目前，它可以定义为以下值之一：`199711L`，`201103L`，`201402L`或`201703L`。宏值代表委员会批准标准的年份和月份。

# 另请参阅

+   阅读第五章中的*创建执行线程*配方，了解有关`BOOST_THREAD_VERSION`及其对`Boost.Thread`库的影响的更多信息，或阅读[`boost.org/libs/thread`](http://boost.org/libs/thread)的文档。

+   从头开始阅读本章，或考虑阅读[Boost.Config](http://boost.org/libs/config)的官方文档
