# 第十章. 收集平台和编译器信息

在本章中，我们将涵盖：

+   检测 int128 支持

+   检测 RTTI 支持

+   使用 C++11 extern 模板加速编译

+   使用更简单的方法编写元函数

+   在 C++11 中减少用户定义类型（UDT）的代码大小并提高性能

+   导出和导入函数和类的可移植方式

+   检测 Boost 版本和获取最新功能

# 简介

不同的项目和公司有不同的编码要求。其中一些禁止异常或 RTTI，而另一些禁止 C++11。如果您愿意编写可移植的代码，这些代码可以用于广泛的工程，那么这一章就是为您准备的。

想要使您的代码尽可能快并使用最新的 C++功能？您肯定会需要一个用于检测编译器功能的工具。

一些编译器具有独特的功能，这些功能可能会极大地简化您的生活。如果您针对单个编译器，您可以节省许多小时并使用这些功能。无需从头开始实现它们的类似功能！

本章致力于不同类型的辅助宏，用于检测编译器、平台和 Boost 功能。这些宏在 Boost 库中广泛使用，对于编写能够与任何编译器标志一起工作的可移植代码至关重要。

# 检测 int128 支持

一些编译器支持扩展算术类型，例如 128 位浮点数或整数。让我们快速了解一下如何使用 Boost 来使用它们。我们将创建一个接受三个参数并返回这些方法乘积的方法。

## 准备工作

只需要具备基本的 C++知识。

## 如何做到这一点...

我们需要什么来处理 128 位整数？显示它们可用的宏以及一些跨平台的 typedef 来具有可移植的类型名称。

1.  我们只需要一个头文件：

    ```cpp
    #include <boost/config.hpp>
    ```

1.  现在我们需要检测 int128 支持：

    ```cpp
    #ifdef BOOST_HAS_INT128
    ```

1.  添加一些 typedef 并按以下方式实现方法：

    ```cpp
    typedef boost::int128_type int_t;
    typedef boost::uint128_type uint_t;

    inline int_t mul(int_t v1, int_t v2, int_t v3) {
        return v1 * v2 * v3;
    }
    ```

1.  对于不支持 int128 类型的编译器，我们可能需要支持 int64 类型：

    ```cpp
    #else // BOOST_NO_LONG_LONG

    #ifdef BOOST_NO_LONG_LONG
    #error "This code requires at least int64_t support"
    #endif
    ```

1.  现在我们需要为不支持 int128 的编译器提供使用 int64 的实现：

    ```cpp
    struct int_t { boost::long_long_type hi, lo; };
    struct uint_t { boost::ulong_long_type hi, lo; };

    inline int_t mul(int_t v1, int_t v2, int_t v3) {
        // Some hand written math
        // ...
    }

    #endif // BOOST_NO_LONG_LONG
    ```

## 它是如何工作的...

头文件 `<boost/config.hpp>` 包含了许多宏来描述编译器和平台功能。在这个例子中，我们使用了 `BOOST_HAS_INT128` 来检测 128 位整数的支持，以及 `BOOST_NO_LONG_LONG` 来检测 64 位整数的支持。

如我们从示例中看到的那样，Boost 为 64 位有符号和无符号整数提供了 typedef：

```cpp
boost::long_long_type 
boost::ulong_long_type 
```

它也提供了 128 位有符号和无符号整数的 typedef：

```cpp
boost::int128_type
boost::uint128_type
```

## 更多...

C++11 通过 `long long int` 和 `unsigned long long int` 内置类型支持 64 位类型。不幸的是，并非所有编译器都支持 C++11，所以 `BOOST_NO_LONG_LONG` 对您将很有用。128 位整数不是 C++11 的一部分，因此 Boost 的 typedef 和宏是编写可移植代码的唯一方法。

## 参见

+   有关 `Boost.Config` 的更多信息，请参阅食谱 *检测 RTTI 支持*。

+   有关其能力的更多信息，请阅读`Boost.Config`的官方文档，链接为[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)。

+   Boost 中有一个库允许构建无限精度的类型。请查看`Boost.Multiprecision`库，链接为[`www.boost.org/doc/libs/1_53_0/libs/multiprecision/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/multiprecision/doc/html/index.html)。

# 检测 RTTI 支持

一些公司和库对他们的 C++代码有特定的要求，例如在**运行时类型信息**（**RTTI**）禁用的情况下成功编译。在这个小食谱中，我们将看看我们如何检测禁用的 RTTI，如何存储类型信息，以及如何在运行时比较类型，即使没有`typeid`。

## 准备工作

需要基本了解 C++ RTTI 的使用才能完成这个食谱。

## 如何做...

检测禁用的 RTTI、存储类型信息以及在运行时比较类型是 Boost 库中广泛使用的技巧。例如，`Boost.Exception`和`Boost.Function`。

1.  要做到这一点，我们首先需要包含以下头文件：

    ```cpp
    #include <boost/config.hpp>
    ```

1.  让我们首先看看 RTTI 已启用且 C++11 `std::type_index`类可用的情况：

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

1.  否则，我们需要自己构建自己的`type_index`类：

    ```cpp
    #else

    #include <cstring>

    struct type_index {
        const char* name_;

        explicit type_index(const char* name)
            : name_(name)
        {}
    };

    inline bool operator == (const type_index& v1, 
        const type_index& v2) 
    {
        return !std::strcmp(v1.name_, v2.name_);
    }

    inline bool operator != (const type_index& v1, 
        const type_index& v2) 
    {
        // '!!' to supress warnings
        return !!std::strcmp(v1.name_, v2.name_);
    }
    ```

1.  最后一步是定义`type_id`函数：

    ```cpp
    #include <boost/current_function.hpp>

    template <class T>
    inline type_index type_id() {
        return type_index(BOOST_CURRENT_FUNCTION);
    }
    #endif
    ```

1.  现在我们可以比较类型了：

    ```cpp
        assert(type_id<unsigned int>() == type_id<unsigned>());
        assert(type_id<double>() != type_id<long double>());
    ```

## 它是如何工作的...

如果 RTTI 被禁用，则将定义`BOOST_NO_RTTI`宏，如果编译器没有`<typeindex>`头文件和`std::type_index`类，则将定义`BOOST_NO_CXX11_HDR_TYPEINDEX`宏。

上一个部分步骤 3 中手写的`type_index`结构只包含指向某些字符串的指针；这里没有什么真正有趣的内容。

看一下`BOOST_CURRENT_FUNCTION`宏。它返回当前函数的完整名称，包括模板参数、参数和返回类型。例如，`type_id<double>()`将表示如下：

```cpp
type_index type_id() [with T = double]

```

因此，对于任何其他类型，`BOOST_CURRENT_FUNCTION`将返回不同的字符串，这就是为什么示例中的`type_index`变量不会与它相等。

## 更多...

不同的编译器有不同的宏来获取完整的函数名和 RTTI。使用 Boost 的宏是最便携的解决方案。`BOOST_CURRENT_FUNCTION`宏在编译时返回名称，因此它意味着最小的运行时惩罚。

## 参见

+   阅读即将到来的食谱，了解更多关于`Boost.Config`的信息

+   浏览到[`github.com/apolukhin/type_index`](https://github.com/apolukhin/type_index)并参考那里的库，该库使用本食谱中的所有技巧来实现`type_index`

+   在[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)阅读`Boost.Config`的官方文档

# 使用 C++11 外部模板加速编译

记得你曾经使用过的一些在头文件中声明的复杂模板类的情况吗？这样的类的例子包括`boost::variant`、来自`Boost.Container`的容器或`Boost.Spirit`解析器。当我们使用这样的类或方法时，它们通常在每个使用它们的源文件中单独编译（实例化），并且在链接过程中会丢弃重复项。在某些编译器上，这可能会导致编译速度变慢。

如果有一种方法可以告诉编译器在哪个源文件中实例化它就好了！

## 准备工作

需要具备模板的基本知识才能完成此食谱。

## 如何做...

这种方法在现代 C++标准库中广泛用于支持它的编译器。例如，与 GCC 一起提供的 STL 库使用这种技术实例化`std::basic_string<char>`和`std::basic_fstream<char>`。

1.  要自行完成，我们需要包含以下头文件：

    ```cpp
    #include <boost/config.hpp>
    ```

1.  我们还需要包含一个包含我们希望减少实例化计数的模板类的头文件：

    ```cpp
    #include <boost/variant.hpp>
    #include <boost/blank.hpp>
    #include <string>
    ```

1.  以下是为支持 C++11 外部模板的编译器提供的代码：

    ```cpp
    #ifndef BOOST_NO_CXX11_EXTERN_TEMPLATE

    extern template class boost::variant<
        boost::blank,
        int,
        std::string,
        double
    >;

    #endif
    ```

1.  现在，我们需要将以下代码添加到我们希望模板实例化的源文件中：

    ```cpp
    // Header with 'extern template'
    #include "header.hpp"

    #ifndef BOOST_NO_CXX11_EXTERN_TEMPLATE
    template class boost::variant<
        boost::blank,
        int,
        std::string,
        double
    >;
    #endif
    ```

## 它是如何工作的...

C++11 关键字`extern template`只是告诉编译器不要在没有显式请求的情况下实例化模板。

第 4 步中的代码是显式请求在此源文件中实例化模板。

当编译器支持 C++11 外部模板时，定义了`BOOST_NO_CXX11_EXTERN_TEMPLATE`宏。

## 还有更多...

外部模板不会影响你程序的运行时性能，但可以显著减少某些模板类的编译时间。不要过度使用它们；对于小型模板类来说，它们几乎毫无用处。

## 参见

+   阅读本章的其他食谱，以获取有关`Boost.Config`的更多信息。

+   有关本章未涵盖的宏的官方文档，请参阅`Boost.Config`的文档，网址为[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)

# 使用更简单的方法编写元函数

第四章，“编译时技巧”，和第八章，“元编程”，都是关于元编程的。如果你试图使用那些章节中的技术，你可能已经注意到编写元函数可能需要花费很多时间。因此，在编写可移植实现之前，使用更用户友好的方法，如 C++11 `constexpr`进行元函数实验可能是一个好主意。

在这个食谱中，我们将探讨如何检测`constexpr`支持。

## 准备工作

`constexpr`函数是可以编译时评估的函数。这就是我们为此食谱需要了解的所有内容。

## 如何做...

目前，很少有编译器支持 `constexpr` 功能，因此可能需要一个良好的新编译器来进行实验。让我们看看如何检测编译器对 `constexpr` 功能的支持：

1.  就像本章其他食谱一样，我们从一个以下头文件开始：

    ```cpp
    #include <boost/config.hpp>
    ```

1.  现在我们将使用 `constexpr`：

    ```cpp
    #if !defined(BOOST_NO_CXX11_CONSTEXPR) \
        && !defined(BOOST_NO_CXX11_HDR_ARRAY)

    template <class T>
    constexpr int get_size(const T& val) {
        return val.size() * sizeof(typename T::value_type);
    }
    ```

1.  如果缺少 C++11 功能，让我们打印一个错误：

    ```cpp
    #else
    #error "This code requires C++11 constexpr and std::array"
    #endif
    ```

1.  就这样；现在我们可以自由地编写如下代码：

    ```cpp
    std::array<short, 5> arr;
    assert(get_size(arr) == 5 * sizeof(short));

    unsigned char data[get_size(arr)];
    ```

## 它是如何工作的...

当 C++11 的 `constexpr` 可用时，定义了 `BOOST_NO_CXX11_CONSTEXPR` 宏。

`constexpr` 关键字告诉编译器，如果该函数的所有输入都是编译时常量，则可以在编译时评估该函数。C++11 对 `constexpr` 函数能做什么有很多限制。C++14 将移除一些限制。

当 C++11 的 `std::array` 类和 `<array>` 头文件可用时，定义了 `BOOST_NO_CXX11_HDR_ARRAY` 宏。

## 还有更多...

然而，还有其他可用的和有趣的宏用于 `constexpr`，如下所示：

+   `BOOST_CONSTEXPR` 宏展开为 `constexpr` 或不展开

+   `BOOST_CONSTEXPR_OR_CONST` 宏展开为 `constexpr` 或 `const`

+   `BOOST_STATIC_CONSTEXPR` 宏与 `static BOOST_CONSTEXPR_OR_CONST` 相同

使用这些宏，如果可用，可以编写利用 C++11 常量表达式特性的代码：

```cpp
template <class T, T Value>
struct integral_constant {
    BOOST_STATIC_CONSTEXPR T value = Value;

    BOOST_CONSTEXPR operator T() const {
        return this->value;
    }
};
```

现在，我们可以像以下代码所示使用 `integral_constant`：

```cpp
char array[integral_constant<int, 10>()];
```

在示例中，`BOOST_CONSTEXPR operator T()` 将被调用以获取数组大小。

C++11 的常量表达式可以在出错情况下提高编译速度和诊断信息。这是一个很好的特性来使用。

## 参见

+   有关 `constexpr` 用法的更多信息，请参阅 [`en.cppreference.com/w/cpp/language/constexpr`](http://en.cppreference.com/w/cpp/language/constexpr)

+   有关宏的更多信息，请阅读 `Boost.Config` 的官方文档：[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)

# 在 C++11 中减少用户定义类型（UDTs）的代码大小并提高性能

当在 STL 容器中使用用户定义类型（UDTs）时，C++11 有非常具体的逻辑。如果移动构造函数不抛出异常或者没有复制构造函数，容器将仅使用移动赋值和移动构造。

让我们看看如何确保我们的类型的 `move_nothrow` 赋值运算符和 `move_nothrow` 构造函数不会抛出异常。

## 准备工作

需要具备 C++11 rvalue references 的基本知识才能完成此食谱。了解 STL 容器也将对你大有裨益。

## 如何做到...

让我们看看如何使用 Boost 来改进我们的 C++ 类。

1.  我们需要做的只是用 `BOOST_NOEXCEPT` 宏标记 `move_nothrow` 赋值运算符和 `move_nothrow` 构造函数：

    ```cpp
    #include <boost/config.hpp>
    class move_nothrow {
        // Some class class members go here
        // ...
    public:
        move_nothrow() BOOST_NOEXCEPT {}
        move_nothrow(move_nothrow&&) BOOST_NOEXCEPT
            // : members initialization
            // ...
        {}

        move_nothrow& operator=(move_nothrow&&) BOOST_NOEXCEPT {
            // Implementation
            // ...
            return *this;
        }

        move_nothrow(const move_nothrow&);
        move_nothrow& operator=(const move_nothrow&);
    };
    ```

1.  现在我们可以直接在 C++11 中使用 `std::vector` 类，无需任何修改：

    ```cpp
        std::vector<move_nothrow> v(10);
        v.push_back(move_nothrow());
    ```

1.  如果我们从移动构造函数中移除 `BOOST_NOEXCEPT`，对于 GCC-4.7 及以后的编译器，我们将得到以下错误：

    ```cpp
    /usr/include/c++/4.7/bits/stl_construct.h:77: undefined reference to `move_nothrow::move_nothrow(move_nothrow const&)'
    ```

## 它是如何工作的...

在支持它的编译器上，`BOOST_NOEXCEPT` 宏展开为 `noexcept`。STL 容器使用类型特性来检测构造函数是否抛出异常。类型特性主要基于 `noexcept` 说明符做出决定。

为什么没有 `BOOST_NOEXCEPT` 会出错？GCC 的类型特性返回 `move_nothrow` 抛出的移动构造函数，因此 `std::vector` 将尝试使用 `move_nothrow` 的复制构造函数，而这个复制构造函数并未定义。

## 还有更多...

`BOOST_NOEXCEPT` 宏无论 `noexcept` 函数或方法的定义是在单独的源文件中还是不在，都会减少二进制文件的大小。

```cpp
// In header file
int foo() BOOST_NOEXCEPT;

// In source file
int foo() BOOST_NOEXCEPT {
    return 0;
}
```

这是因为在后一种情况下，编译器知道该函数不会抛出异常，因此不需要生成处理它们的代码。

### 注意

如果标记为 `noexcept` 的函数抛出异常，则程序将在不调用构造对象的析构函数的情况下终止。

## 参考资料还有

+   一份描述为什么移动构造函数允许抛出异常以及容器如何移动对象的文档可在[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html)找到。

+   有关 `Boost.Config` 的官方文档中提供了更多 `noexcept` 宏在 Boost 中的示例，请参阅[`www.boost.org/doc/libs/1_53_0/libs/conf`](http://www.boost.org/doc/libs/1_53_0/libs/conf)[ig/doc/html/index.html](http://ig/doc/html/index.html)。

# 可移植地导出和导入函数和类的方法

几乎所有现代语言都有创建库的能力，库是一组具有良好定义接口的类和方法。C++ 也不例外。我们有两种类型的库：运行时库（也称为共享或动态加载）和静态库。但在 C++ 中编写库并不是一个简单任务。不同的平台有不同的方法来描述必须从共享库中导出的符号。

让我们看看如何使用 Boost 以可移植的方式管理符号可见性。

## 准备工作

在此配方中，创建动态和静态库的经验将很有用。

## 如何操作...

此配方的代码由两部分组成。第一部分是库本身。第二部分是使用该库的代码。这两部分都使用相同的头文件，在该头文件中声明了库方法。使用 Boost 以可移植的方式管理符号可见性简单且可以通过以下步骤完成：

1.  在头文件中，我们需要从以下 `include` 头文件中获取定义：

    ```cpp
    #include <boost/config.hpp>
    ```

1.  以下代码也必须添加到头文件中：

    ```cpp
    #if defined(MY_LIBRARY_LINK_DYNAMIC)
    # if defined(MY_LIBRARY_COMPILATION)
    #   define MY_LIBRARY_API BOOST_SYMBOL_EXPORT
    # else
    #   define MY_LIBRARY_API BOOST_SYMBOL_IMPORT
    # endif
    #else
    # define MY_LIBRARY_API
    #endif
    ```

1.  现在所有声明都必须使用 `MY_LIBRARY_API` 宏：

    ```cpp
    int MY_LIBRARY_API foo();
    class MY_LIBRARY_API bar { 
    public:
        /* ... */ 
        int meow() const;
    };
    ```

1.  异常必须使用 `BOOST_SYMBOL_VISIBLE` 声明，否则只能在将使用库的代码中使用 `catch(...)` 来捕获：

    ```cpp
    #include <stdexcept>
    struct BOOST_SYMBOL_VISIBLE bar_exception
        : public std::exception 
    {};
    ```

1.  库源文件必须包含头文件：

    ```cpp
    #define MY_LIBRARY_COMPILATION
    #include "my_library.hpp"
    ```

1.  方法的定义也必须在库的源文件中：

    ```cpp
    int MY_LIBRARY_API foo() {
        // Implementation
        // ...
        return 0;
    }
    int bar::meow() const {
        throw bar_exception();
    }
    ```

1.  现在，我们可以像以下代码所示使用库：

    ```cpp
    #include "../my_library/my_library.hpp"
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

## 它是如何工作的...

所有工作都在第 2 步完成。在那里我们定义了宏`MY_LIBRARY_API`，它将被应用于我们希望从库中导出的类和方法。在第 2 步中，我们检查`MY_LIBRARY_LINK_DYNAMIC`是否已定义；如果没有定义，我们正在构建一个静态库，因此不需要定义`MY_LIBRARY_API`。

### 注意

开发者必须注意`MY_LIBRARY_LINK_DYNAMIC`！它不会自动定义。因此，如果我们正在构建动态库，我们需要让我们的构建系统来定义它。

如果定义了`MY_LIBRARY_LINK_DYNAMIC`，我们正在构建一个运行时库，这就是解决方案开始的地方。作为开发者，你必须告诉编译器我们现在正在将这些方法导出给用户。用户必须告诉编译器他/她正在从库中导入方法。为了有一个用于库导入和导出的单个头文件，我们使用以下代码：

```cpp
# if defined(MY_LIBRARY_COMPILATION)
#   define MY_LIBRARY_API BOOST_SYMBOL_EXPORT
# else
#   define MY_LIBRARY_API BOOST_SYMBOL_IMPORT
# endif
```

当导出库（或者说，换句话说，编译它）时，我们必须定义`MY_LIBRARY_COMPILATION`。这导致`MY_LIBRARY_API`被定义为`BOOST_SYMBOL_EXPORT`。例如，参见第 5 步，我们在包含`my_library.hpp`之前定义了`MY_LIBRARY_COMPILATION`。如果未定义`MY_LIBRARY_COMPILATION`，则由用户包含头文件，而用户对此宏一无所知。而且，如果头文件由用户包含，则必须从库中导入符号。

必须仅使用`BOOST_SYMBOL_VISIBLE`宏来处理那些未导出且用于 RTTI 的类。此类类的例子包括异常和被`dynamic_cast`转换的类。

## 还有更多...

一些编译器默认导出所有符号，但提供标志来禁用此行为。例如，GCC 提供`-fvisibility=hidden`标志。强烈建议使用这些标志，因为它会导致二进制文件大小减小，动态库加载更快，以及二进制输入的逻辑结构更好。当导出的符号较少时，一些过程间优化可以表现得更好。

C++11 已经推广了属性，将来可能会被用来提供一种可移植的方式来处理可见性，但在此之前，我们必须使用 Boost 的宏。

## 参见

+   从头开始阅读本章，以获取更多`Boost.Config`使用的示例

+   考虑阅读`Boost.Config`的官方文档，以获取`Boost.Config`宏及其描述的完整列表，请参阅[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.htm)

# 检测 Boost 版本和获取最新功能

Boost 正在积极开发中，因此每个版本都包含新的特性和库。有些人希望有针对不同 Boost 版本的库，并且还想使用新版本的一些特性。

让我们来看看`boost::lexical_cast`的变更日志。根据它，Boost 1.53 有一个`lexical_cast(const CharType* chars, std::size_t count)`函数重载。我们这个菜谱的任务将是使用该函数重载来处理 Boost 的新版本，并为旧版本处理缺失的函数重载。

## 准备工作

只需要具备基本的 C++知识和`Boost.Lexical`库知识。

## 如何做...

好吧，我们所需做的只是获取一个 Boost 版本并使用它来编写最优代码。这可以通过以下步骤完成：

1.  我们需要包含包含 Boost 版本和`boost::lexical_cast`的头文件：

    ```cpp
    #include <boost/version.hpp>
    #include <boost/lexical_cast.hpp>
    ```

1.  如果可用，我们将使用`Boost.LexicalCast`的新特性：

    ```cpp
    #if (BOOST_VERSION >= 105200)
    int to_int(const char* str, std::size_t length) {
        return boost::lexical_cast<int>(str, length);
    }
    ```

1.  否则，我们首先需要将数据复制到`std::string`中：

    ```cpp
    #else
    int to_int(const char* str, std::size_t length) {
        return boost::lexical_cast<int>(
            std::string(str, length)
        );
    }
    #endif
    ```

1.  现在，我们可以使用以下代码：

    ```cpp
    assert(to_int("10000000", 3) == 100);
    ```

## 它是如何工作的...

`BOOST_VERSION`宏包含以以下格式编写的 Boost 版本：一个用于主版本的数字，后面跟着三个用于次版本的数字，然后是两个用于修补级别的数字。例如，Boost 1.46.1 将在`BOOST_VERSION`宏中包含`104601`这个数字。

因此，在第二步中，我们将检查 Boost 版本，并根据`Boost.LexicalCast`的能力选择正确的`to_int`函数实现。

## 更多...

对于大型库来说，拥有一个版本宏是一种常见做法。一些 Boost 库允许你指定要使用的库版本；例如，参见`Boost.Thread`及其`BOOST_THREAD_VERSION`宏。

## 参见

+   有关`BOOST_THREAD_VERSION`及其如何影响`Boost.Thread`库的更多信息，请参阅第五章中的菜谱*创建执行线程*，或者阅读[`www.boost.org/doc/libs/1_53_0/doc/html/thread/changes.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/thread/changes.html)上的文档。

+   从本章开始阅读，或者考虑阅读[`www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)上的官方`Boost.Config`文档。
