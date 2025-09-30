

# 第四章：预处理和编译

在 C++中，编译是将源代码转换为机器代码并组织成对象文件的过程，这些对象文件随后被链接在一起以生成可执行文件。编译器实际上一次只处理一个文件（称为翻译单元），该文件由预处理程序（处理预处理指令的编译器部分）从单个源文件及其包含的所有头文件生成。然而，这只是一个简化的编译代码时发生的事情的描述。本章讨论与预处理和编译相关的话题，重点关注执行条件编译的各种方法，同时也涉及其他现代主题，例如使用属性提供实现定义的语言扩展。

本章包含的食谱如下：

+   条件编译您的源代码

+   使用间接模式进行预处理程序字符串化和连接

+   使用`static_assert`执行编译时断言检查

+   使用`enable_if`条件编译类和函数

+   使用`constexpr if`在编译时选择分支

+   使用属性向编译器提供元数据

我们将在本章开始时讨论的食谱解决的是开发者面临的一个非常普遍的问题，即根据各种条件仅编译代码库的一部分。

# 条件编译您的源代码

条件编译是一种简单的机制，它使开发者能够维护单个代码库，但只考虑代码的一部分进行编译以生成不同的可执行文件，通常是为了在不同的平台或硬件上运行，或者依赖于不同的库或库版本。常见的例子包括根据编译器、平台（x86、x64、ARM 等）、配置（调试或发布）或任何用户定义的特定条件使用或忽略代码。在本食谱中，我们将探讨条件编译是如何工作的。

## 准备工作

条件编译是一种广泛用于许多目的的技术。在本食谱中，我们将查看几个示例并解释它们是如何工作的。这种技术并不局限于这些示例。对于本食谱的范围，我们只考虑三个主要的编译器：GCC、Clang 和 VC++。

## 如何做到这一点...

要条件编译代码的部分，请使用`#if`、`#ifdef`和`#ifndef`指令（以及`#elif`、`#else`和`#endif`指令）。条件编译的一般形式如下：

```cpp
#if condition1
  text1
#elif condition2
  text2
#elif condition3
  text3
#else
  text4
#endif 
```

因为这里的条件通常意味着使用`defined identifier`或`defined (identifier)`语法检查宏是否已定义，因此也可以使用以下形式：

```cpp
#ifdef identifier1
  text1
#elifdef identifier2
  text2
#endif
#ifndef identifier1
  text1
#elifndef identifier2
  text2
#endif 
```

`#elifdef`和`#elifndef`指令是在 C++23 中引入的。

要为条件编译定义宏，您可以使用以下两种方法之一：

+   在您的源代码中的`#define`指令：

    ```cpp
    #define VERBOSE_PRINTS
    #define VERBOSITY_LEVEL 5 
    ```

+   每个编译器特有的编译器命令行选项。以下是最广泛使用的编译器的示例：

    +   对于 Visual C++，使用`/Dname`或`/Dname=value`（其中`/Dname`等同于`/Dname=1`），例如，`cl /DVERBOSITY_LEVEL=5`。

    +   对于 GCC 和 Clang，使用`-D name`或`-D name=value`（其中`-D name`等同于`-D name=1`），例如，`gcc -D VERBOSITY_LEVEL=5`。

以下是一些典型的条件编译示例：

+   头文件保护以避免重复定义（由于在同一翻译单元中多次包含相同的头文件）：

    ```cpp
    #ifndef UNIQUE_NAME
    #define UNIQUE_NAME
    class widget { };
    #endif 
    ```

+   针对跨平台应用的编译器特定代码。以下是一个向控制台打印带有编译器名称的消息的示例：

    ```cpp
    void show_compiler()
    {
      #if defined _MSC_VER
        std::cout << "Visual C++\n";
      #elif defined __clang__
        std::cout << "Clang\n";
      #elif defined __GNUG__
        std::cout << "GCC\n";
      #else
        std::cout << "Unknown compiler\n";
      #endif
    } 
    ```

+   针对多个架构的特定代码，例如，为多个编译器和架构条件编译代码：

    ```cpp
    void show_architecture()
    {
    #if defined _MSC_VER
    #if defined _M_X64
      std::cout << "AMD64\n";
    #elif defined _M_IX86
      std::cout << "INTEL x86\n";
    #elif defined _M_ARM
      std::cout << "ARM\n";
    #else
      std::cout << "unknown\n";
    #endif
    #elif defined __clang__ || __GNUG__
    #if defined __amd64__
      std::cout << "AMD64\n";
    #elif defined __i386__
      std::cout << "INTEL x86\n";
    #elif defined __arm__
      std::cout << "ARM\n";
    #else
      std::cout << "unknown\n";
    #endif
    #else
    #error Unknown compiler
    #endif
    } 
    ```

+   针对特定配置的代码，例如，为调试和发布构建条件编译代码：

    ```cpp
    void show_configuration()
    {
    #ifdef _DEBUG
      std::cout << "debug\n";
    #else
      std::cout << "release\n";
    #endif
    } 
    ```

+   要检查语言或库功能是否可用，请使用预定义的宏`__cpp_xxx`（例如`__cpp_constexpr`、`__cpp_constinit`或`__cpp_modules`）用于语言功能，以及`__cpp_lib_xxx`（例如`__cpp_lib_concepts`、`__cpp_lib_expected`或`__cpp_lib_jthread`）用于库功能。库功能宏是在 C++20 中引入的，并在`<version>`头文件中可用：

    ```cpp
    #ifdef __cpp_consteval
    #define CONSTEVAL consteval
    #else
    #define CONSTEVAL constexpr
    #endif
    CONSTEVAL int twice(int const n)
    {
        return n + n;
    }
    int main()
    {
      twice(42);
    } 
    ```

+   要检查头文件或源文件是否可用于包含，请使用`__has_include`指令，该指令在 C++17 中可用。以下示例检查`<optional>`头文件是否存在：

    ```cpp
    #if __has_include(<optional>)
    #include <optional>
    template<class T> using optional_t = std::optional<T>;
    #elif
    #include "myoptional.h"
    template<class T> using optional_t = my::optional<T>;
    #endif 
    ```

+   要检查属性是否受支持（以及从哪个版本开始），请使用`__has_cpp_attribute`指令，该指令在 C++20 中可用：

    ```cpp
    #if defined(__has_cpp_attribute) 
    #if __has_cpp_attribute(deprecated)
    #define DEPRECATED(msg) [[deprecated(msg)]]
    #endif
    #endif
    DEPRECATED("This function is deprecated.")
    void func() {} 
    ```

## 它是如何工作的...

在讨论编译之前，我们首先应该明确一个我们经常会遇到的术语：**翻译单元**。在 C++中，这是编译的基本单位。它是将源文件（一个`.cpp`文件）的内容和所有直接或间接包含的头文件的全部图（不包括由条件预处理语句排除的文本）组合起来的结果，正如本食谱中所述。

当您使用预处理指令`#if`、`#ifndef`、`#ifdef`、`#elif`、`#else`和`#endif`时，编译器将选择最多一个分支，其主体将被包含在翻译单元中进行编译。这些指令的主体可以是任何文本，包括其他预处理指令。以下规则适用：

+   `#if`、`#ifdef`和`#ifndef`必须与`#endif`匹配。

+   `#if`指令可以有多个`#elif`指令，但只能有一个`#else`，它也必须是`#endif`之前的最后一个。

+   `#if`、`#ifdef`、`#ifndef`、`#elif`、`#else`和`#endif`可以嵌套。

+   `#if`指令需要一个常量表达式，而`#ifdef`和`#ifndef`需要一个标识符。

+   `defined`运算符可用于预处理器常量表达式，但仅限于`#if`和`#elif`指令。

+   如果 `identifier` 被定义，则 `defined(identifier)` 被认为是 `true`；否则，被认为是 `false`。

+   被定义为空文本的标识符被认为是已定义的。

+   `#ifdef identifier` 等同于 `#if defined(identifier)`。

+   `#ifndef identifier` 等同于 `#if !defined(identifier)`。

+   `defined(identifier)` 和 `defined identifier` 是等效的。

头文件保护是条件编译中最常见的形式之一。这种技术用于防止头文件的内容在同一个翻译单元中被多次包含（尽管每次都会扫描头文件以检测应该包含的内容）。因此，头文件中的代码以示例中所示的方式进行了保护，以防止多次包含。考虑到给定的示例，如果 `UNIQUE_NAME` 宏（这是上一节中的通用名称）未定义，则 `#if` 指令之后的代码，直到 `#endif`，将被包含在翻译单元中并编译。当这种情况发生时，使用 `#define` 指令定义 `UNIQUE_NAME` 宏。下次将头文件包含在（相同的）翻译单元中时，`UNIQUE_NAME` 宏已被定义，因此 `#if` 指令体中的代码不会被包含在翻译单元中，从而避免了重复。

注意，宏的名称在整个应用程序中必须是唯一的；否则，只有使用该宏的第一个头文件中的代码将被编译。使用相同名称的其他头文件中的代码将被忽略。通常，宏的名称基于定义它的头文件名称。

条件编译的另一个重要例子是跨平台代码，它需要考虑不同的编译器和架构，通常是 Intel x86、AMD64 或 ARM。然而，编译器为可能的平台定义了自己的宏。*如何做...* 部分的示例展示了如何为多个编译器和架构条件编译代码。

注意，在上述示例中，我们只考虑了几个架构。在实际应用中，存在多个宏可以用来识别相同的架构。在使用这些类型的宏之前，请确保您已经阅读了每个编译器的文档。

特定配置的代码也使用宏和条件编译来处理。例如，GCC 和 Clang 编译器在调试配置（当使用 `-g` 标志时）中不定义任何特殊的宏。Visual C++ 为调试配置定义了 `_DEBUG`，这在上一节的 *如何做...* 部分中已展示。对于其他编译器，您必须显式定义一个宏来识别此类调试配置。

功能测试是条件编译的重要用例，特别是在为多个平台（Windows、Linux 等）和编译器版本（C++11、C++14、C++17 等）提供支持的库中。库实现者通常需要检查特定语言功能或语言属性是否可用。这可以通过一组预定义的宏来实现，包括以下内容：

+   `__cplusplus`: 表示正在使用的 C++ 标准版本。它扩展为以下值之一：`199711L` 用于 C++11 之前的版本，`201103L` 用于 C++11，`201402L` 用于 C++14，`201703L` 用于 C++17，以及 `202002L` 用于 C++20。在撰写本书时，C++23 的值尚未定义。

+   `__cpp_xxx` 宏，用于确定语言功能是否受支持。例如包括 `__cpp_concepts`、`__cpp_consteval`、`__cpp_modules` 等。

+   `__cpp_lib_xxx` 宏，用于确定库功能是否受支持。例如包括 `__cpp_lib_any`、`__cpp_lib_optional`、`__cpp_lib_constexpr_string` 等。这些宏定义在 C++20 中引入的 `<version>` 头文件中。

随着 C++ 中新功能的添加，`__cpp_xxx` 语言功能宏和 `__cpp_lib_xxx` 库功能宏正在通过新宏进行扩展。宏的完整列表太长，无法在此处展示，但可以在 [`en.cppreference.com/w/cpp/feature_test`](https://en.cppreference.com/w/cpp/feature_test) 查询。

除了这些宏之外，还有两个指令，`__has_include` 和 `__has_cpp_attribute`，可以在 `#if`/`#elif` 表达式中使用，以确定头文件或源文件是否存在，或者编译器是否支持某个属性。所有这些宏和指令都是确定特定功能是否存在的有用工具。它们使我们能够编写跨平台和编译器版本的代码。

## 更多内容...

有时，在执行条件编译时，你可能希望显示一个警告或完全停止编译。这可以通过两个诊断宏来实现：

+   `#error` 向控制台显示消息并停止程序的编译。

+   `#warning` 自 C++23 起可用，向控制台显示消息而不停止程序的编译。

以下代码片段展示了使用这些指令的示例：

```cpp
#ifdef _WIN64
#error "64-bit not supported"
#endif
#if __cplusplus < 201703L
#warning "Consider upgrading to a C++17 compiler"
#endif 
```

虽然仅从 C++23 开始提供 `#warning`，但许多编译器提供对该指令的支持作为扩展。

## 参见

+   *使用间接模式进行预处理器字符串化和连接*，了解如何将标识符转换为字符串并在预处理期间连接标识符

# 使用间接模式进行预处理器字符串化和连接

C++ 预处理器提供了两个操作符，用于将标识符转换为字符串并将标识符连接在一起。第一个操作符，操作符 `#`，被称为 **字符串化操作符**，而第二个操作符，操作符 `##`，被称为 **标记粘贴**、**合并**或**连接操作符**。尽管它们的使用仅限于某些特定情况，但理解它们的工作原理是很重要的。

## 准备工作

对于这个配方，你需要知道如何使用预处理指令 `#define` 来定义宏。

## 如何做到这一点...

要使用预处理操作符 `#` 从标识符创建字符串，请使用以下模式：

1.  定义一个辅助宏，它接受一个参数，该参数展开为 `#`，后跟参数：

    ```cpp
    #define MAKE_STR2(x) #x 
    ```

1.  定义你想要使用的宏，它接受一个参数，该参数展开为辅助宏：

    ```cpp
    #define MAKE_STR(x) MAKE_STR2(x) 
    ```

要使用预处理操作符 `##` 将标识符连接在一起，请使用以下模式：

1.  定义一个辅助宏，它有一个或多个参数，这些参数使用标记粘贴操作符 `##` 来连接参数：

    ```cpp
    #define MERGE2(x, y)    x##y 
    ```

1.  使用辅助宏定义你想要使用的宏：

    ```cpp
    #define MERGE(x, y)     MERGE2(x, y) 
    ```

## 它是如何工作的...

要理解这些是如何工作的，让我们考虑之前定义的 `MAKE_STR` 和 `MAKE_STR2` 宏。当与任何文本一起使用时，它们将生成包含该文本的字符串。以下示例展示了这两个宏如何被用来定义包含文本 `"sample"` 的字符串：

```cpp
std::string s1 { MAKE_STR(sample) };  // s1 = "sample"
std::string s2 { MAKE_STR2(sample) }; // s2 = "sample" 
```

另一方面，当宏作为参数传递时，结果会有所不同。在以下示例中，`NUMBER` 是一个展开为整数的宏，`42`。当它作为 `MAKE_STR` 的参数使用时，确实生成了字符串 `"42"`；然而，当它作为 `MAKE_STR2` 的参数使用时，生成了字符串 `"NUMBER"`：

```cpp
#define NUMBER 42
std::string s3 { MAKE_STR(NUMBER) };    // s3 = "42"
std::string s4 { MAKE_STR2(NUMBER) };   // s4 = "NUMBER" 
```

C++ 标准定义了以下规则，用于函数式宏中的参数替换（来自 C++ 标准文档编号 N4917 的第 15.6.2 段）：

> 在识别了函数式宏调用的参数之后，就会进行参数替换。替换列表中的参数，除非它前面有一个 # 或 ## 预处理令牌，或者后面有一个 ## 预处理令牌（见下文），否则在展开其中包含的所有宏之后，会被相应的参数替换。在替换之前，每个参数的预处理令牌会被完全宏替换，就像它们构成了预处理文件的其余部分一样；没有其他预处理令牌可用。

这意味着在将宏参数替换到宏体之前，会先展开这些参数，除了当操作符 `#` 或 `##` 位于宏体中的参数之前或之后的情况。因此，以下情况会发生：

+   对于 `MAKE_STR2(NUMBER)`，替换列表中的 `NUMBER` 参数前面有一个 `#`，因此，在将参数替换到宏体之前不会展开；因此，在替换之后，我们得到 `#NUMBER`，它变成了 `"NUMBER"`。

+   对于 `MAKE_STR(NUMBER)`，替换列表是 `MAKE_STR2(NUMBER)`，它没有 `#` 或 `##`；因此，`NUMBER` 参数在替换之前被替换为其相应的参数，`42`。结果是 `MAKE_STR2(42)`，然后再次扫描，并在展开后变为 `"42"`。

相同的处理规则适用于使用标记粘贴运算符的宏。因此，为了确保您的字符串化和连接宏适用于所有情况，始终应用本食谱中描述的间接模式。

标记粘贴运算符通常用于考虑重复代码的宏中，以避免反复明确地编写相同的内容。以下简单的示例展示了标记粘贴运算符的实际应用；给定一组类，我们希望提供创建每个类实例的工厂方法：

```cpp
#define DECL_MAKE(x)    DECL_MAKE2(x)
#define DECL_MAKE2(x)   x* make##_##x() { return new x(); }
struct bar {};
struct foo {};
DECL_MAKE(foo)
DECL_MAKE(bar)
auto f = make_foo(); // f is a foo*
auto b = make_bar(); // b is a bar* 
```

熟悉 Windows 平台的人可能已经使用过 `_T`（或 `_TEXT`）宏来声明字符串字面量，这些字符串字面量可以是转换为 Unicode 或 ANSI 字符串（单字节和多字节字符字符串）：

```cpp
auto text{ _T("sample") }; // text is either "sample" or L"sample" 
```

Windows SDK 定义 `_T` 宏如下。注意，当 `_UNICODE` 被定义时，标记粘贴运算符被定义为将 `L` 前缀和实际传递给宏的字符串连接起来：

```cpp
#ifdef _UNICODE
#define __T(x)   L ## x
#else
#define __T(x)   x
#endif
#define _T(x)    __T(x)
#define _TEXT(x) __T(x) 
```

乍一看，似乎没有必要有一个宏调用另一个宏，但这种间接级别对于使 `#` 和 `##` 运算符与其他宏一起工作至关重要，正如我们在本食谱中看到的。

## 参见

+   *有条件地编译源代码*，了解如何根据各种条件编译代码的某些部分

# 使用 `static_assert` 执行编译时断言检查

在 C++ 中，可以执行运行时和编译时断言检查，以确保代码中的特定条件为真。运行时断言的缺点是它们在程序运行时较晚被验证，并且只有当控制流到达它们时才会验证。当条件依赖于运行时数据时没有替代方案；然而，当这种情况不成立时，应优先考虑编译时断言检查。使用编译时断言，编译器能够在开发早期通过错误通知您特定条件尚未满足。然而，这些只能在条件可以在编译时评估的情况下使用。在 C++11 中，编译时断言使用 `static_assert` 执行。

## 准备工作

静态断言检查最常见的用途是与模板元编程一起使用，其中它们可以用来验证模板类型的前置条件是否得到满足（示例可以包括类型是否是 POD 类型、可复制构造、引用类型等）。另一个典型示例是确保类型（或对象）具有预期的尺寸。

## 如何操作...

使用 `static_assert` 声明来确保不同作用域中的条件得到满足：

+   **命名空间**：在这个例子中，我们验证类 `item` 的大小始终为 16：

    ```cpp
    struct alignas(8) item
    {
      int      id;
      bool     active;
      double   value;
    };
    static_assert(sizeof(item) == 16, "size of item must be 16 bytes"); 
    ```

+   **类**：在这个例子中，我们验证 `pod_wrapper` 只能用于 POD 类型：

    ```cpp
    template <typename T>
    class pod_wrapper
    {
      static_assert(std::is_standard_layout_v<T>, "POD type expected!");
      T value;
    };
    struct point
    {
      int x;
      int y;
    };
    pod_wrapper<int>         w1; // OK
    pod_wrapper<point>       w2; // OK
    pod_wrapper<std::string> w3; // error: POD type expected 
    ```

+   **块（函数）**：在这个例子中，我们验证一个函数模板只接受整型类型的参数：

    ```cpp
    template<typename T>
    auto mul(T const a, T const b)
    {
      static_assert(std::is_integral_v<T>, "Integral type expected");
      return a * b;
    }
    auto v1 = mul(1, 2);       // OK
    auto v2 = mul(12.0, 42.5); // error: Integral type expected 
    ```

## 它是如何工作的...

`static_assert` 实际上是一个声明，但它不会引入新的名称。这些声明具有以下形式：

```cpp
static_assert(condition, message); 
```

条件必须在编译时可转换为布尔值，并且消息必须是一个字符串字面量。从 C++17 开始，消息是可选的。

当 `static_assert` 声明中的条件评估为 `true` 时，不会发生任何事情。当条件评估为 `false` 时，编译器生成包含指定消息（如果有）的错误。

消息参数必须是一个字符串字面量。然而，从 C++26 开始，它可以是产生字符序列的任意常量表达式。这有助于为用户提供更好的诊断信息。例如，假设会有一个 `constexpr` `std::format()` 函数，可以编写以下内容：

```cpp
static_assert(
   sizeof(item) == 16,
   std::format("size of item must be 16 bytes but got {}", sizeof(item))); 
```

## 参见

+   *使用 enable_if 条件编译类和函数*，了解 SFINAE 以及如何使用它来为模板指定类型约束

+   *第十二章*，*使用概念指定模板参数的要求*，了解 C++20 概念的基本原理以及如何使用它们来指定模板类型的约束

+   *在编译时使用 constexpr if 选择分支*，了解如何仅使用 *constexpr if* 语句编译代码的一部分

# 使用 enable_if 条件编译类和函数

模板元编程是 C++ 的一个强大功能，它使我们能够编写通用的类和函数，它们可以与任何类型一起工作。这有时是一个问题，因为语言没有定义任何机制来指定可以替换模板参数的类型约束。然而，我们仍然可以通过元编程技巧和利用一个称为 **替换失败不是错误** 的规则来实现这一点，也称为 **SFINAE**。该规则确定当在替换模板参数时，如果显式指定的或推导出的类型替换失败，则编译器是否从重载集中丢弃特定化，而不是生成错误。本食谱将专注于实现模板的类型约束。

## 准备工作

开发者多年来一直使用与 SFINAE 结合的类模板 `enable_if` 来对模板类型实施约束。`enable_if` 模板系列已成为 C++11 标准的一部分，并如下实现：

```cpp
template<bool Test, class T = void>
struct enable_if
{};
template<class T>
struct enable_if<true, T>
{
  typedef T type;
}; 
```

要使用 `std::enable_if`，你必须包含 `<type_traits>` 头文件。

## 如何做到这一点...

`std::enable_if` 可以在多个作用域中使用以实现不同的目的；考虑以下示例：

+   在类模板参数上启用类模板，仅对满足指定条件的类型：

    ```cpp
    template <typename T,
              typename = typename
              std::enable_if_t<std::is_standard_layout_v<T>, T>>
    class pod_wrapper
    {
      T value;
    };
    struct point
    {
      int x;
      int y;
    };
    struct foo
    {
       virtual int f() const
       {
          return 42;
       }
    };
    pod_wrapper<int>         w1; // OK
    pod_wrapper<point>       w2; // OK
    pod_wrapper<std::string> w3; // OK with Clang and GCC
                                 // error with MSVC
                                 // too few template arguments
    pod_wrapper<foo>         w4; // error 
    ```

+   在函数模板参数、函数参数或函数返回类型上启用函数模板，仅对满足指定条件的类型：

    ```cpp
    template<typename T,
             typename = typename std::enable_if_t<std::is_integral_v<T>, T>>
    auto mul(T const a, T const b)
    {
      return a * b;
    }
    auto v1 = mul(1, 2);     // OK
    auto v2 = mul(1.0, 2.0); // error: no matching overloaded function found 
    ```

为了简化我们使用`std::enable_if`时最终编写的杂乱代码，我们可以利用别名模板并定义两个别名，分别称为`EnableIf`和`DisableIf`：

```cpp
template <typename Test, typename T = void>
using EnableIf = typename std::enable_if_t<Test::value, T>;
template <typename Test, typename T = void>
using DisableIf = typename std::enable_if_t<!Test::value, T>; 
```

基于这些别名模板，以下定义与前面的定义等效：

```cpp
template <typename T, typename = EnableIf<std::is_standard_layout<T>>>
class pod_wrapper
{
  T value;
};
template<typename T, typename = EnableIf<std::is_integral<T>>>
auto mul(T const a, T const b)
{
  return a * b;
} 
```

## 它是如何工作的...

`std::enable_if`之所以有效，是因为编译器在执行重载解析时应用了 SFINAE 规则。在我们能够解释`std::enable_if`是如何工作的之前，我们应该快速了解一下 SFINAE 是什么。

当编译器遇到函数调用时，它需要构建一组可能的重载并基于函数调用的参数选择最佳匹配。在构建这个重载集时，编译器也会评估函数模板，并必须对模板参数中指定的或推导出的类型进行替换。根据 SFINAE（Substitution Failure Is Not An Error），当替换失败时，编译器不应产生错误，而应仅从重载集中移除函数模板并继续。

标准指定了一个类型和表达式错误列表，这些也是 SFINAE 错误。这包括尝试创建`void`数组或大小为零的数组，尝试创建对`void`的引用，尝试创建具有`void`类型参数的函数类型，以及尝试在模板参数表达式中或在函数声明中使用的表达式中执行无效转换。有关异常的完整列表，请参阅 C++标准或其他资源。

让我们考虑一个名为`func()`的函数的两个重载。第一个重载是一个只有一个`T::value_type`类型参数的函数模板；这意味着它只能用具有名为`value_type`的内部类型的类型实例化。第二个重载是一个只有一个`int`类型参数的函数：

```cpp
template <typename T>
void func(typename T::value_type const a)
{ std::cout << "func<>" << '\n'; }
void func(int const a)
{ std::cout << "func" << '\n'; }
template <typename T>
struct some_type
{
  using value_type = T;
}; 
```

如果编译器遇到`func(42)`这样的调用，它必须找到一个可以接受`int`参数的重载。当它构建重载集并用提供的模板参数替换模板参数时，结果`void func(int::value_type const)`是无效的，因为`int`没有`value_type`成员。由于 SFINAE，编译器不会发出错误并停止，而只是忽略该重载并继续。然后它找到`void func(int const)`，这将是最合适（也是唯一）的匹配，它将调用。

如果编译器遇到`func<some_type<int>>(42)`这样的调用，它将构建一个包含`void func(some_type<int>::value_type const)`和`void func(int const)`的重载集，在这种情况下，最佳匹配是第一个重载；这次没有涉及 SFINAE。

另一方面，如果编译器遇到 `func("string"s)` 这样的调用，它再次依赖于 SFINAE 来忽略函数模板，因为 `std::basic_string` 也没有 `value_type` 成员。然而，这次重载集合中不包含任何与字符串参数匹配的项；因此，程序是无效的，编译器发出错误并停止。

`enable_if<bool, T>` 类模板没有任何成员，但它的部分特化 `enable_if<true, T>` 有一个内部类型称为 `type`，它是 `T` 的同义词。当将 `enable_if` 的第一个参数作为编译时表达式评估为 `true` 时，内部成员 `type` 是可用的；否则，它不可用。

考虑到 *如何做到...* 部分的 `mul()` 函数的最后定义，当编译器遇到 `mul(1, 2)` 这样的调用时，它试图用 `int` 替换模板参数 `T`；由于 `int` 是一个整型，`std::is_integral<T>` 评估为 `true`，因此，定义了一个内部类型 `type` 的 `enable_if` 特化被实例化。结果，别名模板 `EnableIf` 成为此类型的同义词，即 `void`（来自表达式 `typename T = void`）。结果是，可以带有提供的参数调用的函数模板 `int mul<int, void>(int a, int b)`。

另一方面，当编译器遇到 `mul(1.0, 2.0)` 这样的调用时，它试图用 `double` 替换模板参数 `T`。然而，这并不是一个整型；因此，`std::enable_if` 中的条件评估为 `false`，类模板没有定义内部成员 `type`。这导致替换错误，但根据 SFINAE，编译器不会发出错误，而是继续进行。然而，由于没有找到其他重载，将没有可以调用的 `mul()` 函数。因此，程序被认为是无效的，编译器停止并报错。

类模板 `pod_wrapper` 遇到类似的情况。它有两个模板类型参数：第一个是被包装的实际 POD 类型，而第二个是 `enable_if` 和 `is_standard_layout` 替换的结果。如果类型是 POD 类型（如 `pod_wrapper<int>`），则 `enable_if` 的内部成员 `type` 存在，并替换第二个模板类型参数。然而，如果内部成员 `type` 不是一个 POD 类型（如 `pod_wrapper<std::string>`），则内部成员 `type` 未定义，替换失败，产生如 *模板参数太少* 这样的错误。

## 还有更多...

`static_assert` 和 `std::enable_if` 可以用来实现相同的目标。实际上，在前面的配方中，*使用 `static_assert` 进行编译时断言检查*，我们定义了相同的类模板 `pod_wrapper` 和函数模板 `mul()`。对于这些示例，`static_assert` 似乎是一个更好的解决方案，因为编译器会发出更好的错误信息（前提是在 `static_assert` 声明中指定了相关的消息）。然而，这两个函数的工作方式相当不同，并不打算作为替代品。

`static_assert` 不依赖于 SFINAE，并且在重载解析完成后应用。失败的断言会导致编译器错误。另一方面，`std::enable_if` 用于从重载集中移除候选者，并且不会触发编译器错误（假设标准中指定的 SFINAE 异常没有发生）。SFINAE 后可能发生的实际错误是一个空的重载集，这会使程序无效。这是因为特定的函数调用无法执行。

要了解 `static_assert` 和 `std::enable_if` 与 SFINAE 之间的区别，让我们考虑一个我们想要有两个函数重载的情况：一个用于整型类型的参数，另一个用于除整型类型之外的所有类型的参数。使用 `static_assert`，我们可以编写以下内容（注意，第二个重载上的虚拟第二个类型参数是必要的，以便定义两个不同的重载；否则，我们只会有两个相同函数的定义）：

```cpp
template <typename T>
auto compute(T const a, T const b)
{
  static_assert(std::is_integral_v<T>, "An integral type expected");
  return a + b;
}
template <typename T, typename = void>
auto compute(T const a, T const b)
{
  static_assert(!std::is_integral_v<T>, "A non-integral type expected");
  return a * b;
}
auto v1 = compute(1, 2);
// error: ambiguous call to overloaded function
auto v2 = compute(1.0, 2.0);
// error: ambiguous call to overloaded function 
```

无论我们如何尝试调用此函数，最终都会出错，因为编译器找到了两个可能调用的重载。这是因为 `static_assert` 仅在重载解析完成后才被考虑，在这种情况下，构建了一个包含两个可能候选者的集合。

解决这个问题的方法是 `std::enable_if` 和 SFINAE。我们通过之前定义的别名模板 `EnableIf` 和 `DisableIf` 在模板参数上使用 `std::enable_if`（尽管我们仍然在第二个重载上使用虚拟模板参数以引入两个不同的定义）。以下示例显示了重载的重新编写。第一个重载仅对整型类型启用，而第二个对整型类型禁用：

```cpp
template <typename T, typename = EnableIf<std::is_integral<T>>>
auto compute(T const a, T const b)
{
  return a * b;
}
template <typename T, typename = DisableIf<std::is_integral<T>>,
          typename = void>
auto compute(T const a, T const b)
{
  return a + b;
}
auto v1 = compute(1, 2);     // OK; v1 = 2
auto v2 = compute(1.0, 2.0); // OK; v2 = 3.0 
```

在 SFINAE 作用下，当编译器为 `compute(1, 2)` 或 `compute(1.0, 2.0)` 构建重载集时，它将简单地丢弃产生替换失败的过载，并继续进行，在每种情况下，我们最终都会得到一个只包含单个候选者的重载集。

## 参见

+   *使用 `static_assert` 进行编译时断言检查*，了解如何定义在编译时验证的断言

+   *第一章*，*创建类型别名和别名模板*，了解类型别名

# 使用 `constexpr if` 在编译时选择分支

在之前的菜谱中，我们看到了如何使用 `static_assert` 和 `std::enable_if` 对类型和函数施加限制，以及这两个是如何不同的。当我们使用 SFINAE 和 `std::enable_if` 来定义函数重载或编写变长模板函数时，模板元编程可能会变得复杂和杂乱。C++17 的一个新特性旨在简化此类代码；它被称为 *constexpr if*，它定义了一个在编译时评估条件的 `if` 语句，从而使得编译器选择翻译单元中某个分支或另一个分支的主体。*constexpr if* 的典型用法是简化变长模板和基于 `std::enable_if` 的代码。

## 准备工作

在这个菜谱中，我们将参考并简化在前两个菜谱中编写的代码。在继续这个菜谱之前，你应该花点时间回顾一下我们在之前的菜谱中编写的代码，如下所示：

+   来自 *使用 enable_if 条件编译类和函数* 菜谱的整型和非整型的 `compute()` 重载。

+   来自 *第二章*，*处理数字和字符串* 的 *创建原始用户定义字面量* 菜谱的用户定义的 8 位、16 位和 32 位二进制字面量。

这些实现有几个问题：

+   它们很难阅读。有很多关注模板声明，而函数的主体却非常简单，例如。然而，最大的问题是它需要开发者更加注意，因为它充满了复杂的声明，如 `typename = std::enable_if<std::is_integral<T>::value, T>::type`。

+   代码太多。第一个示例的最终目的是拥有一个对不同的类型表现不同的泛型函数，但我们不得不为该函数编写两个重载；此外，为了区分这两个重载，我们不得不使用一个额外的、未使用的模板参数。在第二个示例中，目的是从字符 `'0'` 和 `'1'` 构建一个整数值，但我们不得不编写一个类模板和三个特化来实现这一点。

+   它需要高级模板元编程技能，而这对于做这样简单的事情是不必要的。

*constexpr if* 的语法与常规 `if` 语句非常相似，需要在条件之前使用 `constexpr` 关键字。一般形式如下：

```cpp
if constexpr (init-statement condition) statement-true
else statement-false 
```

注意，在这个形式中，`init-statement` 是可选的。

在以下部分，我们将探讨使用 *constexpr if* 进行条件编译的几个用例。

## 如何做...

使用 *constexpr if* 语句来完成以下操作：

+   为了避免使用 `std::enable_if` 并依赖于 SFINAE 对函数模板类型施加限制以及条件编译代码：

    ```cpp
    template <typename T>
    auto value_of(T value)
    {
      if constexpr (std::is_pointer_v<T>)
     return *value;
      else
    return value;
    } 
    ```

+   为了简化编写变长模板并实现元编程编译时递归：

    ```cpp
    namespace binary
    {
      using byte8 = unsigned char;
      namespace binary_literals
      {
        namespace binary_literals_internals
        {
          template <typename CharT, char d, char... bits>
          constexpr CharT binary_eval()
     {
            if constexpr(sizeof...(bits) == 0)
     return static_cast<CharT>(d-'0');
            else if constexpr(d == '0')
              return binary_eval<CharT, bits...>();
            else if constexpr(d == '1')
              return static_cast<CharT>(
                (1 << sizeof...(bits)) |
                binary_eval<CharT, bits...>());
          }
        }
        template<char... bits>
        constexpr byte8 operator""_b8()
        {
          static_assert(
            sizeof...(bits) <= 8,
            "binary literal b8 must be up to 8 digits long");
          return binary_literals_internals::
                     binary_eval<byte8, bits...>();
        }
      }
    } 
    ```

## 它是如何工作的...

`constexpr if` 的工作方式相对简单：`if` 语句中的条件必须是一个编译时表达式，该表达式可以评估或转换为布尔值。如果条件为 `true`，则选择 `if` 语句的主体，这意味着它最终会进入编译单元进行编译。如果条件为 `false`，则评估（如果已定义）`else` 分支。丢弃的 `constexpr if` 分支中的返回语句不会对函数返回类型推导做出贡献。

在 *How to do it...* 部分的第一个示例中，`value_of()` 函数模板有一个干净的签名。其主体也非常简单；如果用于模板参数的类型是指针类型，编译器将选择第一个分支（即 `return *value;`）进行代码生成并丢弃 `else` 分支。对于非指针类型，因为条件评估为 `false`，编译器将选择 `else` 分支（即 `return value;`）进行代码生成并丢弃其余部分。此函数可以使用如下方式：

```cpp
auto v1 = value_of(42);
auto p = std::make_unique<int>(42);
auto v2 = value_of(p.get()); 
```

然而，没有 `constexpr if` 的帮助，我们只能使用 `std::enable_if` 来实现这一点。以下是一个更杂乱的替代实现：

```cpp
template <typename T,
          typename = typename std::enable_if_t<std::is_pointer_v<T>, T>>
auto value_of(T value)
{
  return *value;
}
template <typename T,
          typename = typename std::enable_if_t<!std::is_pointer_v<T>, T>>
T value_of(T value)
{
  return value;
} 
```

如您所见，`constexpr if` 变体不仅更短，而且更具表达性，更容易阅读和理解。

在 *How to do it...* 部分的第二个示例中，内部的 `binary_eval()` 辅助函数是一个没有任何参数的变长模板函数；它只有模板参数。该函数评估第一个参数，然后以递归方式处理剩余的参数（但请记住，这并不是运行时递归）。当只剩下一个字符并且剩余的包的大小为 `0` 时，我们返回由字符表示的十进制值（`'0'` 为 `0`，`'1'` 为 `1`）。如果当前第一个元素是 `'0'`，我们通过评估剩余的参数包来确定值，这涉及到递归调用。如果当前第一个元素是 `'1'`，我们通过将 `1` 左移由剩余包的大小或确定的值指定的位数来返回值。我们通过评估剩余的参数包来完成这项工作，这又涉及到递归调用。

## 参见

+   使用 `enable_if` 条件编译类和函数，了解 SFINAE 以及如何使用它来为模板指定类型约束

# 使用属性向编译器提供元数据

C++在提供数据类型反射或内省功能以及定义语言扩展的标准机制方面一直存在很大缺陷。正因为如此，编译器为这个目的定义了自己的特定扩展。例如，VC++的`__declspec()`指定符和 GCC 的`__attribute__((...))`。然而，C++11 引入了属性的概念，这使得编译器能够以标准方式或甚至嵌入特定领域的语言来实现扩展。新的 C++标准定义了所有编译器都应该实现的几个属性，这将是本菜谱的主题。

## 如何操作...

使用标准属性为编译器提供有关各种设计目标的提示，例如在以下场景中，但不仅限于此：

+   为了确保函数的返回值不能被忽略，使用`[[nodiscard]]`属性声明函数。在 C++20 中，你可以指定一个字符串字面量，形式为`[[nodiscard(text)]]`，来解释为什么结果不应该被丢弃：

    ```cpp
    [[nodiscard]] int get_value1()
    {
      return 42;
    }
    get_value1();
    // warning: ignoring return value of function
    //          declared with 'nodiscard' attribute get_value1(); 
    ```

+   或者，你可以使用`[[nodiscard]]`属性声明用作函数返回类型的枚举和类；在这种情况下，任何返回此类类型的函数的返回值都不能被忽略：

    ```cpp
    enum class[[nodiscard]] ReturnCodes{ OK, NoData, Error };
    ReturnCodes get_value2()
    {
      return ReturnCodes::OK;
    }
    struct[[nodiscard]] Item{};
    Item get_value3()
    {
      return Item{};
    }
    // warning: ignoring return value of function
    //          declared with 'nodiscard' attribute
    get_value2();
    get_value3(); 
    ```

+   为了确保被认为已过时的函数或类型的用法被编译器标记为警告，使用`[[deprecated]]`属性声明它们：

    ```cpp
    [[deprecated("Use func2()")]] void func()
    {
    }
    // warning: 'func' is deprecated : Use func2()
    func();
    class [[deprecated]] foo
    {
    };
    // warning: 'foo' is deprecated
    foo f; 
    ```

+   为了确保编译器不对未使用的变量发出警告，使用`[[maybe_unused]]`属性：

    ```cpp
    double run([[maybe_unused]] int a, double b)
    {
      return 2 * b;
    }
    [[maybe_unused]] auto i = get_value1(); 
    ```

+   为了确保`switch`语句中的有意跳过的情况标签不会被编译器标记为警告，使用`[[fallthrough]]`属性：

    ```cpp
    void option1() {}
    void option2() {}
    int alternative = get_value1();
    switch (alternative)
    {
      case 1:
        option1();
        [[fallthrough]]; // this is intentional
    case 2:
        option2();
    } 
    ```

+   为了帮助编译器优化执行路径，使用 C++20 的`[[likely]]`和`[[unlikely]]`属性：

    ```cpp
    void execute_command(char cmd)
    {
      switch(cmd)
      {
        [[likely]]
        case 'a': /* add */ break;
        [[unlikely]]
        case 'd': /* delete */ break;
        case 'p': /* print */ break;
        default:  /* do something else */ break;
      }
    } 
    ```

+   为了帮助编译器根据用户提供的假设优化代码，使用 C++23 的`[[assume]]`属性：

    ```cpp
    void process(int* data, size_t len)
    {
       [[assume(len > 0)]];
       for(size_t i = 0; i < len; ++i)
       {
         // do something with data[i]
       }
    } 
    ```

## 它是如何工作的...

属性是 C++的一个非常灵活的特性；它们几乎可以在任何地方使用，但实际使用是针对每个特定属性具体定义的。它们可以用在类型、函数、变量、名称、代码块或整个翻译单元上。

属性指定在双方括号之间（例如，`[[attr1]]`），并且在声明中可以指定多个属性（例如，`[[attr1, attr2, attr3]]`）。

属性可以有参数，例如`[[mode(greedy)]]`，并且可以是完全限定的，例如`[[sys::hidden]]`或`[[using sys: visibility(hidden), debug]]`。

属性可以出现在它们所应用的实体名称之前或之后，或者两者都出现，在这种情况下它们会被组合。以下是一些示例，说明了这一点：

```cpp
// attr1 applies to a, attr2 applies to b
int a [[attr1]], b [[attr2]];
// attr1 applies to a and b
int [[attr1]] a, b;
// attr1 applies to a and b, attr2 applies to a
int [[attr1]] a [[attr2]], b; 
```

属性不能出现在命名空间声明中，但可以作为单行声明出现在命名空间中的任何位置。在这种情况下，是否应用于后续声明、命名空间或翻译单元取决于每个属性：

```cpp
namespace test
{
  [[debug]];
} 
```

标准确实定义了所有编译器都必须实现的几个属性，使用它们可以帮助你编写更好的代码。我们在上一节给出的示例中看到了一些。这些属性已在标准的不同版本中定义：

+   在 C++11 中：

    +   `[[noreturn]]` 属性表示函数不会返回。

    +   `[[carries_dependency]]` 属性表示在发布-消费 `std::memory_order` 中的依赖链在函数中传播进出，这允许编译器跳过不必要的内存栅栏指令。

+   在 C++14 中：

    +   `[[deprecated]]` 和 `[[deprecated("reason")]]` 属性表示使用这些属性声明的实体被认为是过时的，不应使用。这些属性可以与类、非静态数据成员、typedefs、函数、枚举和模板特化一起使用。`"reason"` 字符串是一个可选参数。

+   在 C++17 中：

    +   `[[fallthrough]]` 属性表示在 `switch` 语句中的标签之间的穿透是故意的。该属性必须单独一行，紧接在 `case` 标签之前。

    +   `[[nodiscard]]` 属性表示函数的返回值不能被忽略。

    +   `[[maybe_unused]]` 属性表示实体可能未使用，但编译器不应发出有关该问题的警告。此属性可以应用于变量、类、非静态数据成员、枚举、枚举符和 typedefs。

+   在 C++20 中：

    +   `[[nodiscard(text)]]` 属性是 C++17 的 `[[nodiscard]]` 属性的扩展，并提供描述结果不应被丢弃原因的文本。

    +   `[[likely]]` 和 `[[unlikely]]` 属性为编译器提供提示，表明执行路径更有可能或不太可能执行，因此允许它相应地进行优化。它们可以应用于语句（但不能是声明）和标签，但只能使用其中一个，因为它们是互斥的。

    +   `[[no_unique_address]]` 属性可以应用于非静态数据成员（排除位域），并告知编译器该成员不必具有唯一的地址。当应用于具有空类型的成员时，编译器可以将其优化为不占用空间，例如，当它是一个空基类时。另一方面，如果成员的类型不为空，编译器可能会重用任何后续的填充来存储其他数据成员。

+   在 C++23 中：

    +   `[[assume(expr)]]` 表示一个表达式将始终评估为 `true`。它的目的是让编译器执行代码优化，而不是记录函数的先决条件。然而，表达式永远不会被评估。具有未定义行为或抛出异常的表达式将被评估为 `false`。不成立的假设会导致未定义行为；因此，假设应该谨慎使用。另一方面，编译器可能根本不会使用假设。

在现代 C++ 编程的书籍和教程中，属性通常被忽略或简略提及，这其中的原因可能是因为开发者实际上无法编写属性，因为这个语言特性是为编译器实现而设计的。然而，对于某些编译器来说，可能可以定义用户提供的属性；GCC 就是这样一种编译器，它支持插件，这些插件可以为编译器添加额外功能，也可以用来定义新的属性。

## 参见

+   *第九章*，*使用 noexcept 处理不抛出异常的函数*，了解如何通知编译器一个函数不应该抛出异常

# 在 Discord 上了解更多

加入我们的 Discord 社区空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
