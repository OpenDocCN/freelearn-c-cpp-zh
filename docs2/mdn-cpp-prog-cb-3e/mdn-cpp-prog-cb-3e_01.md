# 1

# 学习现代核心语言特性

C++语言在过去几十年中经历了重大变革，随着 C++11 的发布以及随后更新的版本：C++14、C++17、C++20 和 C++23，这些新标准引入了新的概念，简化并扩展了现有的语法和语义，并彻底改变了我们编写代码的方式。与之前所知相比，C++11 看起来和感觉上像是一种全新的语言，使用这些新标准编写的代码被称为现代 C++代码。本入门章节将涉及一些引入的语言特性，从 C++11 开始，这些特性有助于你处理许多编码常规。然而，语言的核心内容远远超出了本章所讨论的主题，书中其他章节还讨论了许多其他特性。

本章包含的食谱如下：

+   尽可能使用`auto`

+   创建类型别名和别名模板

+   理解统一初始化

+   理解非静态成员初始化的各种形式

+   控制和查询对象对齐

+   使用范围枚举

+   使用`override`和`final`关键字为虚方法

+   使用基于范围的 for 循环遍历范围

+   为自定义类型启用基于范围的 for 循环

+   使用显式构造函数和转换运算符来避免隐式转换

+   使用无名命名空间而不是静态全局变量

+   使用内联命名空间进行符号版本控制

+   使用结构化绑定来处理多返回值

+   使用类模板参数推导简化代码

+   使用下标运算符访问集合中的元素

让我们从学习自动类型推导开始。

# 尽可能使用`auto`

自动类型推导是现代 C++ 中最重要且最广泛使用的特性之一。新的 C++ 标准使得在多种上下文中使用 `auto` 作为类型的占位符成为可能，让编译器推导出实际类型。在 C++11 中，`auto` 可以用来声明局部变量以及具有尾随返回类型的函数的返回类型。在 C++14 中，`auto` 可以用来声明没有指定尾随类型的函数返回类型以及 lambda 表达式中的参数声明。在 C++17 中，它可以用来声明结构化绑定，这在章节末尾有讨论。在 C++20 中，它可以用来简化函数模板语法，所谓的缩写函数模板。在 C++23 中，它可以用来执行对 prvalue 复制的显式转换。未来的标准版本可能会将 `auto` 的使用扩展到更多的情况。C++11 和 C++14 中引入的 `auto` 的使用有几个重要的好处，所有这些都会在 *它是如何工作的...* 部分讨论。开发者应该意识到它们，并尽可能使用 `auto`。安德烈·亚历山德鲁斯库（Andrei Alexandrescu）提出了一个实际术语，并由 Herb Sutter 推广——**几乎总是 auto** （**AAA**） ([`herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/`](https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/))。

## 如何做到...

在以下情况下考虑使用 `auto` 作为实际类型的占位符：

+   使用 `auto name = expression` 形式声明局部变量，当你不想承诺一个特定类型时：

    ```cpp
    auto i = 42;          // int
    auto d = 42.5;        // double
    auto s = "text";      // char const *
    auto v = { 1, 2, 3 }; // std::initializer_list<int> 
    ```

+   使用 `auto name = type-id { expression }` 形式声明局部变量，当你需要承诺一个特定类型时：

    ```cpp
    auto b  = new char[10]{ 0 };            // char*
    auto s1 = std::string {"text"};         // std::string
    auto v1 = std::vector<int> { 1, 2, 3 }; // std::vector<int>
    auto p  = std::make_shared<int>(42);    // std::shared_ptr<int> 
    ```

+   使用 `auto name = lambda-expression` 形式声明命名 lambda 函数，除非 lambda 需要传递或返回给函数：

    ```cpp
    auto upper = [](char const c) {return toupper(c); }; 
    ```

+   声明 lambda 参数和返回值：

    ```cpp
    auto add = [](auto const a, auto const b) {return a + b;}; 
    ```

+   在不想承诺一个特定类型时声明函数返回类型：

    ```cpp
    template <typename F, typename T>
    auto apply(F&& f, T value)
    {
      return f(value);
    } 
    ```

## 它是如何工作的...

`auto` 说明符基本上是一个实际类型的占位符。当使用 `auto` 时，编译器从以下实例推导出实际类型：

+   从初始化变量的表达式类型，当使用 `auto` 声明变量时。

+   从函数的尾随返回类型或返回表达式类型，当 `auto` 用作函数返回类型的占位符时。

在某些情况下，有必要承诺一个特定的类型。例如，在第一个例子中，编译器推断出`s`的类型为`char const *`。如果意图是使用`std::string`，则必须显式指定类型。同样，`v`的类型被推断为`std::initializer_list<int>`，因为它绑定到`auto`而不是特定类型；在这种情况下，规则说明推断的类型是`std::initializer_list<T>`，其中`T`在我们的例子中是`int`。然而，意图可能是拥有一个`std::vector<int>`。在这种情况下，必须在赋值右侧显式指定类型。

使用`auto`指定符而不是实际类型有一些重要的好处；以下是一些可能最重要的好处：

+   无法留下未初始化的变量。这是开发者在声明变量并指定实际类型时常见的错误。然而，使用`auto`是不可能的，因为它需要初始化变量以推断类型。使用定义的值初始化变量很重要，因为未初始化的变量会导致未定义的行为。

+   使用`auto`确保你始终使用预期的类型，并且不会发生隐式转换。考虑以下示例，其中我们检索局部变量的向量大小。在第一种情况下，变量的类型是`int`，尽管`size()`方法返回`size_t`。这意味着将从`size_t`到`int`发生隐式转换。然而，使用`auto`来推断类型将得出正确的类型——即`size_t`：

    ```cpp
    auto v = std::vector<int>{ 1, 2, 3 };
    // implicit conversion, possible loss of data
    int size1 = v.size();
    // OK
    auto size2 = v.size();
    // ill-formed (warning in gcc, error in clang & VC++)
    auto size3 = int{ v.size() }; 
    ```

+   使用`auto`可以促进良好的面向对象实践，例如优先选择接口而非实现。这在**面向对象编程**（**OOP**）中非常重要，因为它提供了在不同实现之间进行更改的灵活性，代码的模块化，以及更好的可测试性，因为模拟对象很容易。指定的类型数量越少，代码越通用，对未来变化的开放性也越大，这是面向对象编程的基本原则。

+   它意味着（通常）更少的输入和更少的对实际类型（我们实际上并不关心的类型）的关注。我们经常遇到的情况是，即使我们明确指定了类型，我们实际上并不关心它。一个非常常见的例子是与迭代器相关，但还有更多。当你想要遍历一个范围时，你并不关心迭代器的实际类型。你只对迭代器本身感兴趣；因此使用 `auto` 可以节省输入（可能很长的）名称的时间，并帮助你专注于实际的代码而不是类型名称。在下面的例子中，在第一个 `for` 循环中，我们明确使用了迭代器的类型。这需要输入很多文本；长语句实际上可能使代码更难以阅读，而且你还需要知道类型名称，而你实际上并不关心。带有 `auto` 指示符的第二个循环看起来更简单，可以节省你输入和关注实际类型的时间：

    ```cpp
    std::map<int, std::string> m;
    for (std::map<int, std::string>::const_iterator
      it = m.cbegin();
      it != m.cend(); ++it)
    { /*...*/ }
    for (auto it = m.cbegin(); it != m.cend(); ++it)
    { /*...*/ } 
    ```

+   使用 `auto` 声明变量提供了一种一致的编码风格，类型始终位于右侧。如果你动态分配对象，你需要在赋值语句的左右两侧都写上类型，例如，`int* p = new int(42)`。使用 `auto`，类型仅在右侧指定一次。

然而，在使用 `auto` 时也有一些需要注意的问题：

+   `auto` 指示符仅是类型的占位符，而不是 `const`/`volatile` 和引用指定符。如果你需要一个 `const`/`volatile` 和/或引用类型，那么你需要明确指定它们。在下面的例子中，`foo` 的 `get()` 成员函数返回 `int` 的引用；当变量 `x` 从返回值初始化时，编译器推断出的类型是 `int`，而不是 `int&`。因此，对 `x` 的任何更改都不会传播到 `foo.x_`。为了做到这一点，我们应该使用 `auto&`：

    ```cpp
    class foo {
      int x;
    public:
      foo(int const value = 0) :x{ value } {}
      int& get() { return x; }
    };
    foo f(42);
    auto x = f.get();
    x = 100;
    std::cout << f.get() << '\n'; // prints 42 
    ```

+   对于不可移动的类型，无法使用 `auto`。

    ```cpp
    auto ai = std::atomic<int>(42); // error 
    ```

+   对于多词类型，如 `long long`、`long double` 或 `struct foo`，无法使用 `auto`。然而，在第一种情况下，可能的解决方案是使用字面量或类型别名；此外，在 Clang 和 GCC（但不是 MSVC）中，可以将类型名称放在括号中，例如 `(long long){ 42 }`。至于第二种情况，以那种形式使用 `struct`/`class` 只在 C++ 中支持与 C 的兼容性，并且应该避免使用：

    ```cpp
    auto l1 = long long{ 42 }; // error
    using llong = long long;
    auto l2 = llong{ 42 };     // OK
    auto l3 = 42LL;            // OK
    auto l4 = (long long){ 42 }; // OK with gcc/clang 
    ```

+   如果你使用了 `auto` 指示符但仍然需要知道类型，你可以在大多数 IDE 中通过将光标放在变量上来实现，例如。然而，如果你离开 IDE，那就不再可能了，唯一知道实际类型的方法是自己从初始化表达式中推断出来，这可能意味着在代码中搜索函数返回类型。

`auto`可以用来指定函数的返回类型。在 C++11 中，这需要在函数声明中有一个尾随返回类型。在 C++14 中，这已经放宽，编译器会从`return`表达式推断返回值的类型。如果有多个返回值，它们应该具有相同的类型：

```cpp
// C++11
auto func1(int const i) -> int
{ return 2*i; }
// C++14
auto func2(int const i)
{ return 2*i; } 
```

如前所述，`auto`不会保留`const`/`volatile`和引用限定符。这导致`auto`作为函数返回类型占位符时出现问题。为了解释这一点，让我们考虑前面的例子`foo.get()`。这次，我们有一个名为`proxy_get()`的包装函数，它接受一个`foo`的引用，调用`get()`，并返回`get()`返回的值，该值是一个`int&`。然而，编译器将推断`proxy_get()`的返回类型为`int`，而不是`int&`。

尝试将那个值赋给一个`int&`会失败并报错：

```cpp
class foo
{
  int x_;
public:
  foo(int const x = 0) :x_{ x } {}
  int& get() { return x_; }
};
auto proxy_get(foo& f) { return f.get(); }
auto f = foo{ 42 };
auto& x = proxy_get(f); // cannot convert from 'int' to 'int &' 
```

为了解决这个问题，我们需要实际返回`auto&`。然而，模板和完美前向传递返回类型时不知道它是值还是引用存在一个问题。C++14 中解决这个问题的方法是`decltype(auto)`，它将正确推断类型：

```cpp
decltype(auto) proxy_get(foo& f) 
{ return f.get(); }
auto f = foo{ 42 };
decltype(auto) x = proxy_get(f); 
```

`decltype`说明符用于检查实体或表达式的声明类型。它主要用于在声明类型比较繁琐或根本无法使用标准符号声明时。这类例子包括声明 lambda 类型以及依赖于模板参数的类型。

`auto`可以使用的最后一个重要情况是与 lambda 一起。截至 C++14，lambda 返回类型和 lambda 参数类型都可以是`auto`。这样的 lambda 被称为*泛型 lambda*，因为 lambda 定义的闭包类型有一个模板调用操作符。以下是一个泛型 lambda 的示例，它接受两个`auto`参数，并返回将`operator+`应用于实际类型的操作结果：

```cpp
auto ladd = [] (auto const a, auto const b) { return a + b; }; 
```

编译器生成的函数对象具有以下形式，其中调用操作符是一个函数模板：

```cpp
struct
{
  template<typename T, typename U>
 auto operator () (T const a, U const b) const { return a+b; }
} L; 
```

这个 lambda 可以用于添加任何定义了`operator+`的操作数的值，如下面的代码片段所示：

```cpp
auto i = ladd(40, 2);            // 42
auto s = ladd("forty"s, "two"s); // "fortytwo"s 
```

在这个例子中，我们使用了`ladd` lambda 函数来添加两个整数并将它们连接到`std::string`对象（使用 C++14 的用户定义字面量`operator ""s`）。

## 参见

+   *创建类型别名和别名模板*，了解类型别名的使用

+   *理解统一初始化*，了解花括号初始化是如何工作的

# 创建类型别名和别名模板

在 C++中，可以创建同义词，这些同义词可以用来代替类型名称。这是通过创建 typedef 声明来实现的。这在几种情况下很有用，例如为类型创建更短或更有意义的名称，或者为函数指针创建名称。然而，typedef 声明不能与模板一起使用来创建模板类型别名。例如，`std::vector<T>`不是一个类型（`std::vector<int>`是一个类型），而是一系列类型，当类型占位符`T`被实际类型替换时可以创建。

在 C++11 中，类型别名是已声明类型的名称，别名模板是已声明模板的名称。这两种类型的别名都是通过新的 using 语法引入的。

## 如何实现...

+   创建具有形式`using identifier = type-id`的类型别名，如下面的示例所示：

    ```cpp
    using byte     = unsigned char;
    using byte_ptr = unsigned char *;
    using array_t  = int[10];
    using fn       = void(byte, double);
    void func(byte b, double d) { /*...*/ }
    byte b{42};
    byte_ptr pb = new byte[10] {0};
    array_t a{0,1,2,3,4,5,6,7,8,9};
    fn* f = func; 
    ```

+   创建具有形式`template<template-params-list> identifier = type-id`的别名模板，如下面的示例所示：

    ```cpp
    template <class T>
    class custom_allocator { /* ... */ };
    template <typename T>
    using vec_t = std::vector<T, custom_allocator<T>>;
    vec_t<int>           vi;
    vec_t<std::string>   vs; 
    ```

为了保持一致性和可读性，你应该做以下事情：

+   在创建别名时不要混合使用 typedef 和 using 声明

+   在创建函数指针类型的名称时，优先使用 using 语法

## 它是如何工作的...

typedef 声明引入了一个类型的同义词（换句话说，别名）。它不会引入另一个类型（如`class`、`struct`、`union`或`enum`声明）。使用 typedef 声明引入的类型名称遵循与标识符名称相同的隐藏规则。它们也可以重新声明，但只能引用相同的类型（因此，你可以在翻译单元中拥有有效的多个 typedef 声明，引入相同的类型名称同义词，只要它是相同类型的同义词）。以下是一些典型的 typedef 声明示例：

```cpp
typedef unsigned char   byte;
typedef unsigned char * byte_ptr;
typedef int array_t[10];
typedef void(*fn)(byte, double);
template<typename T>
class foo {
  typedef T value_type;
};
typedef std::vector<int> vint_t;
typedef int INTEGER;
INTEGER x = 10;
typedef int INTEGER; // redeclaration of same type
INTEGER y = 20; 
```

类型别名声明与 typedef 声明等价。它可以出现在块作用域、类作用域或命名空间作用域中。根据 C++11 标准（第 9.2.4 段，文档版本 N4917）：

> 可以通过别名声明来引入 typedef 名称。使用 using 关键字之后的标识符成为 typedef 名称，标识符之后的可选属性说明符序列属于该 typedef 名称。它具有与使用 typedef 说明符引入相同的语义。特别是，它不会定义一个新的类型，并且它不应出现在类型标识符中。

然而，当创建数组类型和函数指针类型的别名时，别名声明在可读性和对实际别名的清晰度方面更为出色。在*如何实现...*部分的示例中，很容易理解`array_t`是 10 个整数的数组类型的名称，而`fn`是接受两个类型为`byte`和`double`的参数并返回`void`的函数类型的名称。这也与声明`std::function`对象的语法一致（例如，`std::function<void(byte, double)> f`）。

以下事项非常重要：

+   别名模板不能部分或显式地特化。

+   别名模板在推导模板参数时永远不会通过模板参数推导进行推导。

+   当特化别名模板时产生的类型不允许直接或间接地使用其自身的类型。

新语法的驱动目的是定义别名模板。这些模板在特化时，等价于将别名模板的模板参数替换为类型-id 中的模板参数的结果。

## 参见

+   *使用类模板参数推导简化代码*，学习如何在不显式指定模板参数的情况下使用类模板

# 理解统一初始化

大括号初始化是 C++11 中初始化数据的统一方法。因此，它也被称为 *统一初始化*。这可能是 C++11 中开发者应该理解和使用的最重要的特性之一。它消除了初始化基本类型、聚合和非聚合类型、数组和标准容器之间的区别。

## 准备工作

要继续这个配方，你需要熟悉直接初始化，即从一组显式的构造函数参数初始化对象，以及复制初始化，即从一个对象初始化另一个对象。以下是对这两种初始化的简单示例：

```cpp
std::string s1("test");   // direct initialization
std::string s2 = "test";  // copy initialization 
```

在这些考虑的基础上，让我们探索如何执行统一初始化。

## 如何做...

要统一初始化对象，无论其类型如何，都使用大括号初始化形式 `{}`，它可以用于直接初始化和复制初始化。当与大括号初始化一起使用时，这些被称为直接列表和复制列表初始化：

```cpp
T object {other};   // direct-list-initialization
T object = {other}; // copy-list-initialization 
```

统一初始化的例子如下：

+   标准容器：

    ```cpp
    std::vector<int> v { 1, 2, 3 };
    std::map<int, std::string> m { {1, "one"}, { 2, "two" }}; 
    ```

+   动态分配的数组：

    ```cpp
    int* arr2 = new int[3]{ 1, 2, 3 }; 
    ```

+   数组：

    ```cpp
    int arr1[3] { 1, 2, 3 }; 
    ```

+   内置类型：

    ```cpp
    int i { 42 };
    double d { 1.2 }; 
    ```

+   用户定义的类型：

    ```cpp
    class foo
    {
      int a_;
      double b_;
    public:
      foo():a_(0), b_(0) {}
      foo(int a, double b = 0.0):a_(a), b_(b) {}
    };
    foo f1{};
    foo f2{ 42, 1.2 };
    foo f3{ 42 }; 
    ```

+   用户定义的 **纯旧数据** （**POD**）类型：

    ```cpp
    struct bar { int a_; double b_;};
    bar b{ 42, 1.2 }; 
    ```

## 它是如何工作的...

在 C++11 之前，对象根据其类型需要不同类型的初始化：

+   基本类型可以使用赋值进行初始化：

    ```cpp
    int a = 42;
    double b = 1.2; 
    ```

+   如果类对象有一个转换构造函数（在 C++11 之前，只有一个参数的构造函数被称为 *转换构造函数*），它们也可以使用单个值的赋值进行初始化：

    ```cpp
    class foo
    {
      int a_;
    public:
      foo(int a):a_(a) {}
    };
    foo f1 = 42; 
    ```

+   非聚合类可以在提供参数时使用括号（函数形式）进行初始化，而在执行默认初始化（调用默认构造函数）时则无需任何括号。在下一个例子中，`foo` 是在 *如何做...* 部分定义的结构：

    ```cpp
    foo f1;           // default initialization
    foo f2(42, 1.2);
    foo f3(42);
    foo f4();         // function declaration 
    ```

+   聚合和 POD 类型可以使用大括号初始化。在以下示例中，`bar` 是在 *如何做...* 部分定义的结构：

    ```cpp
    bar b = {42, 1.2};
    int a[] = {1, 2, 3, 4, 5}; 
    ```

**纯数据**（**POD**）类型是一种既是平凡（具有编译器提供的或显式默认的特殊成员，并占用连续的内存区域）又具有标准布局（不包含与 C 语言不兼容的语言功能，如虚函数，并且所有成员具有相同的访问控制）的类型。POD 类型的概念在 C++20 中已被弃用，以支持平凡和标准布局类型。

除了不同的数据初始化方法外，还有一些限制。例如，初始化标准容器（除了复制构造之外）的唯一方法是在其中声明一个对象，然后向其中插入元素；`std::vector` 是一个例外，因为它可以从可以预先使用聚合初始化的数组中分配值。然而，另一方面，动态分配的聚合不能直接初始化。

*“如何做...”* 部分中的所有示例都使用直接初始化，但也可以使用花括号初始化进行复制初始化。这两种形式，直接和复制初始化，在大多数情况下可能是等效的，但复制初始化的限制较少，因为它在其隐式转换序列中不考虑显式构造函数，而必须直接从初始化器生成对象，而直接初始化则期望从初始化器到构造函数参数的隐式转换。动态分配的数组只能使用直接初始化。

在前面示例中显示的类中，`foo` 是唯一一个既有默认构造函数又有参数构造函数的类。要使用默认构造函数进行默认初始化，我们需要使用空花括号——即 `{}`。要使用参数构造函数，我们需要在花括号 `{}` 中提供所有参数的值。与非聚合类型不同，其中默认初始化意味着调用默认构造函数，对于聚合类型，默认初始化意味着使用零进行初始化。

如前所述，标准容器（如向量 map）的初始化也是可能的，因为所有标准容器在 C++11 中都有一个额外的构造函数，它接受类型为 `std::initializer_list<T>` 的参数。这基本上是一个轻量级代理，覆盖了类型 `T const` 的元素数组。然后这些构造函数从初始化列表中的值初始化内部数据。

使用 `std::initializer_list` 进行初始化的方式如下：

+   编译器解析初始化列表中元素的类型（所有元素必须具有相同的类型）。

+   编译器创建一个包含初始化列表中元素的数组。

+   编译器创建一个 `std::initializer_list<T>` 对象来包装之前创建的数组。

+   `std::initializer_list<T>` 对象作为参数传递给构造函数。

初始化器列表始终优先于使用花括号初始化的其他构造函数。如果此类存在此类构造函数，则在执行花括号初始化时将被调用：

```cpp
class foo
{
  int a_;
  int b_;
public:
  foo() :a_(0), b_(0) {}
  foo(int a, int b = 0) :a_(a), b_(b) {}
  foo(std::initializer_list<int> l) {}
};
foo f{ 1, 2 }; // calls constructor with initializer_list<int> 
```

优先级规则适用于任何函数，而不仅仅是构造函数。在以下示例中，存在同一函数的两个重载。使用初始化器列表调用函数将解析为具有`std::initializer_list`的重载：

```cpp
void func(int const a, int const b, int const c)
{
  std::cout << a << b << c << '\n';
}
void func(std::initializer_list<int> const list)
{
  for (auto const & e : list)
    std::cout << e << '\n';
}
func({ 1,2,3 }); // calls second overload 
```

然而，这可能导致错误。以`std::vector`类型为例。在向量的构造函数中，有一个只有一个参数的构造函数，表示要分配的初始元素数量，还有一个参数为`std::initializer_list`的构造函数。如果目的是创建具有预分配大小的向量，使用花括号初始化将不起作用，因为具有`std::initializer_list`的构造函数将是最佳重载以被调用：

```cpp
std::vector<int> v {5}; 
```

前面的代码并没有创建一个包含五个元素的向量，而是一个包含一个值为`5`的元素的向量。要实际创建一个包含五个元素的向量，必须使用括号形式进行初始化：

```cpp
std::vector<int> v (5); 
```

另一点需要注意的是，花括号初始化不允许缩窄转换。根据 C++标准（参考标准第 9.4.5 段，文档版本 N4917），缩窄转换是一种隐式转换：

> 从浮点类型到整数类型。
> 
> 从`long double`到`double`或`float`，或从`double`到`float`，除非源是常量表达式，并且转换后的实际值在可以表示的值范围内（即使不能精确表示）。
> 
> 从整数类型或无范围枚举类型到浮点类型，除非源是常量表达式，并且转换后的实际值可以适合目标类型，并且在转换回原始类型时将产生原始值。
> 
> 从整数类型或无范围枚举类型到无法表示原始类型所有值的整数类型，除非源是常量表达式，并且转换后的实际值可以适合目标类型，并且在转换回原始类型时将产生原始值。

以下声明会触发编译器错误，因为它们需要缩窄转换：

```cpp
int i{ 1.2 };           // error
double d = 47 / 13;
float f1{ d };          // error, only warning in gcc 
```

为了修复此错误，必须进行显式转换：

```cpp
int i{ static_cast<int>(1.2) };
double d = 47 / 13;
float f1{ static_cast<float>(d) }; 
```

花括号初始化列表不是一个表达式，也没有类型。因此，不能在花括号初始化列表上使用`decltype`，模板类型推导也不能推导出与花括号初始化列表匹配的类型。

让我们再考虑一个例子：

```cpp
float f2{47/13};        // OK, f2=3 
```

尽管如此，前面的声明是正确的，因为存在从`int`到`float`的隐式转换。表达式`47/13`首先被评估为整数值`3`，然后将其赋值给类型为`float`的变量`f2`。

## 还有更多...

以下示例展示了直接列表初始化和复制列表初始化的几个例子。在 C++11 中，所有这些表达式的推导类型是 `std::initializer_list<int>`：

```cpp
auto a = {42};   // std::initializer_list<int>
auto b {42};     // std::initializer_list<int>
auto c = {4, 2}; // std::initializer_list<int>
auto d {4, 2};   // std::initializer_list<int> 
```

C++17 改变了列表初始化的规则，区分了直接列表初始化和复制列表初始化。类型推导的新规则如下：

+   对于复制列表初始化，如果列表中的所有元素都具有相同的类型，则自动推导将推导出 `std::initializer_list<T>`；否则是不合法的。

+   对于直接列表初始化，如果列表只有一个元素，则自动推导将推导出 `T`；如果有多个元素，则是不合法的。

根据这些新规则，前面的例子将如下改变（推导类型在注释中提及）：

```cpp
auto a = {42};   // std::initializer_list<int>
auto b {42};     // int
auto c = {4, 2}; // std::initializer_list<int>
auto d {4, 2};   // error, too many 
```

在这种情况下，`a` 和 `c` 被推导为 `std::initializer_list<int>`，`b` 被推导为 `int`，而 `d` 使用直接初始化并且花括号初始化列表中有多个值，这会触发编译器错误。

## 参见

+   *尽可能使用 auto*，以了解 C++ 中自动类型推导的工作原理

+   *理解非静态成员的多种初始化形式*，以了解如何最佳地执行类成员的初始化

# 理解非静态成员的多种初始化形式

构造函数是执行非静态类成员初始化的地方。许多开发者更喜欢在构造函数体中使用赋值。除了实际需要的那几个例外情况，非静态成员的初始化应该在构造函数的初始化列表中完成，或者从 C++11 开始，当它们在类中声明时，可以使用默认成员初始化。在 C++11 之前，类的常量和非常量非静态数据成员必须在构造函数中初始化。在类中声明初始化仅适用于静态常量。正如我们将在这里看到的，这种限制在 C++11 被移除，允许在类声明中初始化非静态成员。这种初始化被称为 *默认成员初始化*，将在以下章节中解释。

这个配方将探讨非静态成员初始化应该如何进行。为每个成员使用适当的初始化方法不仅会使代码更高效，而且会使代码组织得更好，更易于阅读。

## 如何做到这一点...

要初始化类的非静态成员，你应该：

+   使用默认成员初始化为常量，包括静态和非静态（参见以下代码中的 `[1]` 和 `[2]`）。

+   使用默认成员初始化为具有多个构造函数的类成员提供默认值（参见以下代码中的 `[3]` 和 `[4]`）。

+   使用构造函数初始化列表来初始化没有默认值但依赖于构造函数参数的成员（参见以下代码中的 `[5]` 和 `[6]`）。

+   当其他选项不可用时，请在构造函数中使用赋值操作（例如，使用指针 `this` 初始化数据成员，检查构造函数参数值，以及在用这些值或两个非静态数据成员的自我引用初始化成员之前抛出异常）。

以下示例显示了这些初始化形式：

```cpp
struct Control
{
  const int DefaultHeight = 14;                                // [1]
const int DefaultWidth  = 80;                                // [2]
  std::string text;
  TextVerticalAlignment valign = TextVerticalAlignment::Middle;   // [3]
  TextHorizontalAlignment halign = TextHorizontalAlignment::Left; // [4]
Control(std::string const & t) : text(t)      // [5]
  {}
  Control(std::string const & t,
    TextVerticalAlignment const va,
    TextHorizontalAlignment const ha):
    text(t), valign(va), halign(ha)             // [6]
  {}
}; 
```

## 它是如何工作的...

非静态数据成员应该在构造函数的初始化列表中进行初始化，如下面的示例所示：

```cpp
struct Point
{
  double x, y;
  Point(double const x = 0.0, double const y = 0.0) : x(x), y(y)  {}
}; 
```

然而，许多开发者并不使用初始化列表，而是偏好构造函数体中的赋值操作，甚至混合使用赋值和初始化列表。这可能由几个原因造成——对于具有许多成员的大类，构造函数中的赋值可能比长初始化列表更容易阅读，也许分散在多行中，或者可能是因为这些开发者熟悉没有初始化列表的其他编程语言。

重要的一点是，非静态数据成员的初始化顺序是它们在类定义中声明的顺序，而不是在构造函数初始化列表中初始化的顺序。相反，非静态数据成员的销毁顺序是构造顺序的反向。

在构造函数中使用赋值操作不是高效的，因为这可能会创建后来被丢弃的临时对象。如果不在初始化列表中初始化，非静态成员将通过它们的默认构造函数进行初始化，然后在构造函数体中分配值时，将调用赋值运算符。如果默认构造函数分配了资源（如内存或文件），并且需要在赋值运算符中重新分配和释放，这可能会导致低效的工作。这在下述代码片段中得到了体现：

```cpp
struct foo
{
  foo()
  { std::cout << "default constructor\n"; }
  foo(std::string const & text)
  { std::cout << "constructor '" << text << "\n"; }
  foo(foo const & other)
  { std::cout << "copy constructor\n"; }
  foo(foo&& other)
  { std::cout << "move constructor\n"; };
  foo& operator=(foo const & other)
  { std::cout << "assignment\n"; return *this; }
  foo& operator=(foo&& other)
  { std::cout << "move assignment\n"; return *this;}
  ~foo()
  { std::cout << "destructor\n"; }
};
struct bar
{
  foo f;
  bar(foo const & value)
  {
    f = value;
  }
};
foo f;
bar b(f); 
```

上述代码产生以下输出，显示了数据成员 `f` 首先通过默认初始化，然后被分配了一个新值：

```cpp
default constructor
default constructor
assignment
destructor
destructor 
```

如果你想跟踪哪个对象被创建和销毁，你可以稍微修改上面的 `foo` 类，并打印每个特殊成员函数的 `this` 指针的值。你可以将此作为后续练习来完成。

将构造函数体中的赋值操作更改为初始化列表，将替换对默认构造函数和赋值运算符的调用，改为调用拷贝构造函数：

```cpp
bar(foo const & value) : f(value) { } 
```

添加上述代码行会产生以下输出：

```cpp
default constructor
copy constructor
destructor
destructor 
```

由于这些原因，至少对于除内置类型（如 `bool`、`char`、`int`、`float`、`double` 或指针）之外的类型，你应该优先选择构造函数的初始化列表。然而，为了保持初始化风格的一致性，在可能的情况下，你应该始终优先选择构造函数的初始化列表。存在一些情况下使用初始化列表是不可能的；以下是一些情况（但列表可以扩展到其他情况）：

+   如果一个成员必须使用指向包含它的对象的指针或引用进行初始化，在初始化列表中使用 `this` 指针可能会在某些编译器上触发警告，表明它应该在对象构造之前使用。

+   如果你有两个数据成员必须相互包含对方的引用。

+   如果你想在初始化非静态数据成员之前测试输入参数并抛出异常。

从 C++11 开始，非静态数据成员可以在类中声明时进行初始化。这被称为 *默认成员初始化*，因为它表示使用默认值进行初始化。默认成员初始化旨在用于常量和那些不基于构造函数参数初始化的成员（换句话说，成员的值不依赖于对象的构造方式）：

```cpp
enum class TextFlow { LeftToRight, RightToLeft };
struct Control
{
  const int DefaultHeight = 20;
  const int DefaultWidth = 100;
  TextFlow textFlow = TextFlow::LeftToRight;
  std::string text;
  Control(std::string const & t) : text(t)
  {}
}; 
```

在前面的例子中，`DefaultHeight` 和 `DefaultWidth` 都是常量；因此，它们的值不依赖于对象的构造方式，所以它们在声明时进行初始化。`textFlow` 对象是一个非常量、非静态数据成员，其值也不依赖于对象的初始化方式（它可以通过另一个成员函数进行更改）；因此，它在声明时也使用默认成员初始化进行初始化。相反，`text` 也是一个非常量、非静态数据成员，但它的初始值依赖于对象的构造方式。

因此，它使用传递给构造函数的参数值在构造函数的初始化列表中进行初始化。

如果一个数据成员同时使用默认成员初始化和构造函数初始化列表进行初始化，后者具有优先级，并且默认值会被丢弃。为了说明这一点，让我们再次考虑前面提到的 `foo` 类和下面的 `bar` 类，它使用了 `foo` 类：

```cpp
struct bar
{
  foo f{"default value"};
  bar() : f{"constructor initializer"}
  {
  }
};
bar b; 
```

在这种情况下，输出如下不同：

```cpp
constructor 'constructor initializer'
destructor 
```

不同行为的原因是默认初始化列表中的值被丢弃，对象不会被初始化两次。

## 参见

+   *理解统一初始化*，了解花括号初始化是如何工作的

# 控制和查询对象对齐

C++11 提供了指定和查询类型对齐要求的标准方法（这之前只能通过编译器特定的方法实现）。控制对齐对于提高不同处理器的性能和允许使用一些仅在特定对齐上工作的指令非常重要。

例如，Intel **流式单指令多数据扩展**（**SSE**）和 Intel SSE2，它们是一组处理器指令，当要对多个数据对象应用相同的操作时，可以大大提高性能，需要数据对齐 16 字节。相反，对于 **Intel 高级向量扩展**（或 **Intel AVX**），它将大多数整数处理器指令扩展到 256 位，强烈建议使用 32 字节对齐。本食谱探讨了用于控制对齐要求的 `alignas` 指定符和用于检索类型对齐要求的 `alignof` 操作符。

## 准备工作

您应该熟悉数据对齐是什么以及编译器如何执行默认数据对齐。然而，有关后者的基本信息在 *它是如何工作的...* 部分提供。

## 如何做到这一点...

+   要控制类型（在类级别或数据成员级别）或对象的对齐，请使用 `alignas` 指定符：

    ```cpp
    struct alignas(4) foo
    {
      char a;
      char b;
    };
    struct bar
    {
      alignas(2) char a;
      alignas(8) int  b;
    };
    alignas(8)   int a;
    alignas(256) long b[4]; 
    ```

+   要查询类型的对齐，请使用 `alignof` 操作符：

    ```cpp
    auto align = alignof(foo); 
    ```

## 它是如何工作的...

处理器不是逐字节访问内存，而是以 2 的幂次方（2、4、8、16、32 等）的更大块来访问。因此，编译器在内存中对齐数据以便处理器可以轻松访问是很重要的。如果数据未对齐，编译器必须做额外的工作来访问数据；它必须读取多个数据块，移位并丢弃不必要的字节，然后组合剩余的部分。

C++ 编译器根据数据类型的大小来对变量进行对齐。标准仅指定了 `char`、`signed char`、`unsigned char`、`char8_t`（在 C++20 中引入）和 `std::byte`（在 C++17 中引入）的大小，这些大小必须是 1。它还要求 `short` 的大小至少为 16 位，`long` 的大小至少为 32 位，`long long` 的大小至少为 64 位。它还要求 1 == `sizeof(char)` <= `sizeof(short)` <= `sizeof(int)` <= `sizeof(long)` <= `sizeof(long long)`。因此，大多数类型的大小是编译器特定的，并且可能取决于平台。通常，这些大小是 `bool` 和 `char` 为 1 字节，`short` 为 2 字节，`int`、`long` 和 `float` 为 4 字节，`double` 和 `long long` 为 8 字节，等等。当涉及到结构体或联合体时，对齐必须与最大成员的大小相匹配，以避免性能问题。为了举例说明，让我们考虑以下数据结构：

```cpp
struct foo1 // size = 1, alignment = 1
{              // foo1:    +-+
char a;      // members: |a|
};
struct foo2 // size = 2, alignment = 1
{              // foo2:    +-+-+
char a;      // members  |a|b|
char b;
};
struct foo3 // size = 8, alignment = 4
{              // foo3:    +----+----+
char a;      // members: |a...|bbbb|
int  b;      // . represents a byte of padding
}; 
```

`foo1` 和 `foo2` 的大小不同，但它们的对齐方式相同——即 1——因为所有数据成员都是 `char` 类型，其大小为 1 字节。在结构 `foo3` 中，第二个成员是一个整数，其大小为 4。因此，该结构的成员对齐是在地址为 4 的倍数的地方进行的。为了实现这一点，编译器引入了填充字节。

结构 `foo3` 实际上被转换成以下形式：

```cpp
struct foo3_
{
  char a;        // 1 byte
char _pad0[3]; // 3 bytes padding to put b on a 4-byte boundary
int  b;        // 4 bytes
}; 
```

类似地，以下结构的大小为 32 字节，对齐为 8；这是因为最大的成员是一个大小为 8 的`double`。然而，这个结构需要在几个地方进行填充，以确保所有成员都可以在地址为 8 的倍数的位置访问：

```cpp
struct foo4 // size = 24, alignment = 8
{               // foo4:    +--------+--------+--------+--------+
int a;        // members: |aaaab...|cccc....|dddddddd|e.......|
char b;       // . represents a byte of padding
float c;
  double d;
  bool e;
}; 
```

编译器创建的等效结构如下：

```cpp
struct foo4_
{
  int a;         // 4 bytes
char b;        // 1 byte
char _pad0[3]; // 3 bytes padding to put c on a 8-byte boundary
float c;       // 4 bytes
char _pad1[4]; // 4 bytes padding to put d on a 8-byte boundary
double d;      // 8 bytes
bool e;        // 1 byte
char _pad2[7]; // 7 bytes padding to make sizeof struct multiple of 8
}; 
```

在 C++11 中，指定对象或类型的对齐是通过使用`alignas`指定符来完成的。这可以是一个表达式（一个求值为`0`或对齐有效值的整型常量表达式）、一个类型标识符或参数包。`alignas`指定符可以应用于变量或类的数据成员的声明，这些成员不表示位字段，或者可以应用于类、联合或枚举的声明。

在声明中使用的所有`alignas`指定符中，应用于类型或对象的`alignas`指定将对齐要求等于所有`alignas`指定符中最大的、大于零的表达式。

使用`alignas`指定符时有一些限制：

+   只有 2 的幂（1、2、4、8、16、32 等等）是有效的对齐方式。任何其他值都是非法的，程序被认为是无效的；这并不一定必须产生错误，因为编译器可以选择忽略该指定。

+   0 的对齐始终被忽略。

+   如果声明中最大的`alignas`值小于没有任何`alignas`指定符的自然对齐，则程序也被认为是无效的。

在下面的示例中，`alignas`指定符已被应用于类声明。如果没有`alignas`指定符的自然对齐将是 1，但使用`alignas(4)`后变为 4：

```cpp
struct alignas(4) foo
{
  char a;
  char b;
}; 
```

换句话说，编译器将前面的类转换为以下形式：

```cpp
struct foo
{
  char a;
  char b;
  char _pad0[2];
}; 
```

`alignas`指定符可以同时应用于类声明和成员数据声明。在这种情况下，最严格的（即，最大的）值获胜。在下面的示例中，成员`a`的自然大小为 1，需要 2 的对齐；成员`b`的自然大小为 4，需要 8 的对齐，因此最严格的对齐将是 8。整个类的对齐要求是 4，这比最严格要求的对齐更弱（即，更小），因此它将被忽略，尽管编译器将生成一个警告：

```cpp
struct alignas(4) foo
{
  alignas(2) char a;
  alignas(8) int  b;
}; 
```

结果是一个看起来像这样的结构：

```cpp
struct foo
{
  char a;
  char _pad0[7];
  int b;
  char _pad1[4];
}; 
```

`alignas`指定符也可以应用于变量。在下面的示例中，整数变量`a`必须放置在内存的 8 的倍数位置。下一个变量，即 4 个长整型的数组，必须放置在内存的 256 的倍数位置。因此，编译器将在两个变量之间引入多达 244 字节的填充（取决于内存中的位置，在地址为 8 的倍数的位置，变量`a`被放置）：

```cpp
alignas(8)   int a;
alignas(256) long b[4];
printf("%p\n", &a); // eg. 0000006C0D9EF908
printf("%p\n", &b); // eg. 0000006C0D9EFA00 
```

通过查看地址，我们可以看到 `a` 的地址确实是 8 的倍数，而 `b` 的地址是 256（十六进制 100）的倍数。

要查询类型的对齐，我们使用 `alignof` 操作符。与 `sizeof` 不同，此操作符只能应用于类型，不能应用于变量或类数据成员。它可以应用于完整类型、数组类型或引用类型。对于数组，返回的值是元素类型的对齐方式；对于引用，返回的值是引用类型的对齐方式。以下是一些示例：

| **表达式** | **评估** |
| --- | --- |
| `alignof(char)` | 1，因为 `char` 的自然对齐方式为 1 |
| `alignof(int)` | 4，因为 `int` 的自然对齐方式为 4 |
| `alignof(int*)` | 32 位系统上的对齐方式为 4，64 位系统上的对齐方式为 8，这是指针的对齐方式 |
| `alignof(int[4])` | 4，因为元素类型的自然对齐方式为 4 |
| `alignof(foo&)` | 8，因为类 `foo` 的指定对齐方式为 8，这是一个引用类型（如前一个示例所示） |

表 1.1：alignof 表达式的示例及其评估值

如果你想强制对数据类型进行对齐（考虑前面提到的限制），`alignas` 指示符非常有用，以便可以有效地访问和复制该类型的变量。这意味着优化 CPU 读取和写入，并避免缓存行不必要的无效化。

在某些类别中的应用中，性能至关重要，例如游戏或交易应用，这可以非常重要。相反，`alignof` 操作符尝试指定类型的最低对齐要求。

## 参见

+   *创建类型别名和别名模板*，以了解类型别名

# 使用范围枚举

枚举是 C++ 中的一个基本类型，它定义了一组值，这些值始终具有一个整型基础类型。它们的命名值，这些值是常量，被称为枚举符。使用关键字 `enum` 声明的枚举称为 *无范围枚举*，而使用 `enum class` 或 `enum struct` 声明的枚举称为 *范围枚举*。后者是在 C++11 中引入的，旨在解决无范围枚举的几个问题，这些问题在本食谱中进行了说明。

## 如何做到这一点...

当处理枚举时，你应该：

+   更倾向于使用范围枚举而不是无范围枚举

+   使用 `enum class` 或 `enum struct` 声明范围枚举：

    ```cpp
    enum class Status { Unknown, Created, Connected };
    Status s = Status::Created; 
    ```

`enum class` 和 `enum struct` 声明是等效的，在这份食谱和本书的其余部分，我们将使用 `enum class`。

由于范围枚举是受限命名空间，C++20 标准允许我们使用 `using` 指令将它们关联起来。你可以这样做：

+   使用 `using` 指令在局部作用域中引入范围枚举标识符，如下所示：

    ```cpp
    int main()
    {
      using Status::Unknown;
      Status s = Unknown;
    } 
    ```

+   使用 `using` 指令在局部作用域中引入范围枚举的所有标识符，如下所示：

    ```cpp
    struct foo
    {
      enum class Status { Unknown, Created, Connected };
      using enum Status;
    };
    foo::Status s = foo::Created; // instead of
    // foo::Status::Created 
    ```

+   使用`using enum`指令在`switch`语句中引入枚举标识符，以简化你的代码：

    ```cpp
    void process(Status const s)
    {
      switch (s)
      {
        using enum Status;
        case Unknown:   /*…*/ break;
        case Created:   /*...*/ break;
        case Connected: /*...*/ break;
      }
    } 
    ```

在使用旧式 API（这些 API 接受整数作为参数）的上下文中，有时需要将范围枚举转换为它的基础类型。在 C++23 中，你可以通过使用`std::to_underlying()`实用函数将范围枚举转换为它的基础类型：

```cpp
void old_api(unsigned flag);
enum class user_rights : unsigned
{
    None, Read = 1, Write = 2, Delete = 4
};
old_api(std::to_underlying(user_rights::Read)); 
```

## 它是如何工作的...

无范围枚举存在一些问题，这些问题会给开发者带来麻烦：

+   它们将枚举符导出到周围作用域（这就是为什么它们被称为无范围枚举），这有两个缺点：

    +   如果同一命名空间中的两个枚举具有相同名称的枚举符，可能会导致名称冲突

    +   使用完全限定名称使用枚举符是不可能的：

        ```cpp
        enum Status {Unknown, Created, Connected};
        enum Codes {OK, Failure, Unknown};   // error
        auto status = Status::Created;       // error 
        ```

+   在 C++11 之前，它们不能指定基础类型，基础类型必须是整型。除非枚举值无法适应有符号或无符号整数，否则此类型不得大于`int`。因此，枚举的前向声明是不可能的。原因在于枚举的大小是未知的。这是因为直到枚举符的值被定义，基础类型才未知，以便编译器选择适当的整型类型。这在 C++11 中得到了修复。

+   枚举符的值隐式转换为`int`。这意味着你可以故意或意外地将具有特定意义的枚举和整数（可能甚至与枚举的意义无关）混合，编译器将无法警告你：

    ```cpp
    enum Codes { OK, Failure };
    void include_offset(int pixels) {/*...*/}
    include_offset(Failure); 
    ```

范围枚举基本上是强类型枚举，其行为与无范围枚举不同：

+   它们不会将枚举符导出到周围作用域。前面显示的两个枚举将变为以下内容，不再生成名称冲突，并使得完全限定枚举符的名称成为可能：

    ```cpp
    enum class Status { Unknown, Created, Connected };
    enum class Codes { OK, Failure, Unknown }; // OK
    Codes code = Codes::Unknown;               // OK 
    ```

+   你可以指定基础类型。无范围枚举的基础类型的相同规则也适用于范围枚举，除了用户可以显式指定基础类型。这也解决了关于前向声明的問題，因为基础类型可以在定义可用之前就已知：

    ```cpp
    enum class Codes : unsigned int;
    void print_code(Codes const code) {}
    enum class Codes : unsigned int
    {
      OK = 0,
      Failure = 1,
      Unknown = 0xFFFF0000U
    }; 
    ```

+   范围枚举的值不再隐式转换为`int`。将`enum class`的值赋给整数变量将触发编译器错误，除非指定了显式转换：

    ```cpp
    Codes c1 = Codes::OK;                       // OK
    int c2 = Codes::Failure;                    // error
    int c3 = static_cast<int>(Codes::Failure);  // OK 
    ```

然而，范围枚举有一个缺点：它们是受限命名空间。它们不会导出外部作用域中的标识符，这在某些情况下可能不方便，例如，当你编写一个`switch`语句并且需要为每个情况标签重复枚举名称时，如下面的示例所示：

```cpp
std::string_view to_string(Status const s)
{
  switch (s)
  {
    case Status::Unknown:   return "Unknown";
    case Status::Created:   return "Created";
    case Status::Connected: return "Connected";
  }
} 
```

在 C++20 中，可以通过使用具有范围枚举名称的`using`指令来简化这一点。前面的代码可以简化如下：

```cpp
std::string_view to_string(Status const s)
{
  switch (s)
  {
    using enum Status;
    case Unknown:   return "Unknown";
    case Created:   return "Created";
    case Connected: return "Connected";
  }
} 
```

此 `using` 指令的效果是，所有枚举标识符都引入到局部作用域中，使得可以使用未限定形式引用它们。也可以使用具有限定标识符名称的 `using` 指令仅将特定的枚举标识符引入局部作用域，例如 `using Status::Connected`。

C++23 标准版本添加了一些用于处理作用域枚举的实用函数。其中第一个是 `std::to_underlying()`，可在 `<utility>` 头文件中找到。它的作用是将枚举转换为它的底层类型。

它的目的是与不使用作用域枚举的 API（无论是遗留的还是新的）一起工作。让我们看看以下函数 `old_api()` 的例子，它接受一个整数参数，将其解释为控制用户权限的系统标志：

```cpp
void old_api(unsigned flag)
{
    if ((flag & 0x01) == 0x01) { /* can read */ }
    if ((flag & 0x02) == 0x02) { /* can write */ }
    if ((flag & 0x04) == 0x04) { /* can delete */ }
} 
```

此函数可以按以下方式调用：

```cpp
old_api(1); // read only
old_api(3); // read & write 
```

相反，系统的较新部分为用户权限定义了以下作用域枚举：

```cpp
enum class user_rights : unsigned
{
    None,
    Read = 1,
    Write = 2,
    Delete = 4
}; 
```

然而，使用来自 `user_rights` 的枚举调用 `old_api()` 函数是不可能的，必须使用 `static_cast`：

```cpp
old_api(static_cast<int>(user_rights::Read)); // read only
old_api(static_cast<int>(user_rights::Read) | 
        static_cast<int>(user_rights::Write)); // read & write 
```

为了避免这些静态转换，C++23 提供了函数 `std::to_underlying()`，可以使用如下方式：

```cpp
old_api(std::to_underlying(user_rights::Read));
old_api(std::to_underlying(user_rights::Read) | 
        std::to_underlying(user_rights::Write)); 
```

C++23 中引入的其他实用工具是一个名为 `is_scoped_enum<T>` 的类型特质，可在 `<type_traits>` 头文件中找到。它包含一个名为 `value` 的成员常量，如果模板类型参数 `T` 是作用域枚举类型，则等于 `true`，否则为 `false`。还有一个辅助变量模板，`is_scoped_enum_v<T>`。

此类型特质的目的是确定枚举是否具有作用域，以便根据枚举的类型应用不同的行为。以下是一个简单的示例：

```cpp
enum A {};
enum class B {};
int main()
{
   std::cout << std::is_scoped_enum_v<A> << '\n';
   std::cout << std::is_scoped_enum_v<B> << '\n';
} 
```

第一行将打印 0，因为 `A` 是无作用域枚举，而第二行将打印 `1`，因为 `B` 是作用域枚举。

## 参见

+   *第九章*，*创建编译时常量表达式*，了解如何处理编译时常量

# 使用 `override` 和 `final` 为虚方法

与其他类似的编程语言不同，C++ 没有用于声明接口（基本上是只有纯虚方法的类）的特定语法，并且还有一些与如何声明虚方法相关的缺陷。在 C++ 中，虚方法是通过 `virtual` 关键字引入的。然而，对于派生类中的重写声明，`virtual` 关键字是可选的，这可能导致在处理大型类或层次结构时产生混淆。您可能需要在整个层次结构中导航到基类，以确定一个函数是否是虚的。相反，有时确保虚函数或派生类不能再被重写或进一步派生是有用的。在这个菜谱中，我们将看到如何使用 C++11 特殊标识符 `override` 和 `final` 来声明虚函数或类。

## 准备就绪

您应该熟悉 C++中的继承和多态，以及抽象类、纯指定符、虚拟和覆盖方法等概念。

## 如何做到...

为了确保在基类和派生类中正确声明虚拟方法，同时确保提高可读性，请执行以下操作：

+   在派生类中声明虚拟函数时，旨在使用`virtual`关键字，这些虚拟函数应该覆盖基类中的虚拟函数。

+   在虚拟函数的声明或定义的声明部分之后始终使用`override`特殊标识符：

    ```cpp
    class Base
    {
      virtual void foo() = 0;
      virtual void bar() {}
      virtual void foobar() = 0;
    };
    void Base::foobar() {}
    class Derived1 : public Base
    {
      virtual void foo() override = 0;
      virtual void bar() override {}
      virtual void foobar() override {}
    };
    class Derived2 : public Derived1
    {
      virtual void foo() override {}
    }; 
    ```

声明符是函数类型的一部分，不包括返回类型。

为了确保函数不能进一步覆盖或类不能再派生，使用`final`特殊标识符，如下所示：

+   在虚拟函数声明或定义的声明部分之后，以防止在派生类中进一步覆盖：

    ```cpp
    class Derived2 : public Derived1
    {
      virtual void foo() final {}
    }; 
    ```

+   在类声明的类名之后，以防止进一步派生该类：

    ```cpp
    class Derived4 final : public Derived1
    {
      virtual void foo() override {}
    }; 
    ```

## 它是如何工作的...

`override`的工作方式非常简单；在虚拟函数的声明或定义中，它确保函数实际上覆盖了基类函数；否则，编译器将触发错误。

应该注意，`override`和`final`特殊标识符都是仅在成员函数声明或定义中有意义的特殊标识符。它们不是保留关键字，并且仍然可以在程序的其他地方作为用户定义的标识符使用。

使用`override`特殊标识符有助于编译器检测虚拟方法没有覆盖另一个方法的情况，如下面的示例所示：

```cpp
class Base
{
public:
  virtual void foo() {}
  virtual void bar() {}
};
class Derived1 : public Base
{
public:
  void foo() override {}
  // for readability use the virtual keyword
virtual void bar(char const c) override {}
  // error, no Base::bar(char const)
}; 
```

如果没有`override`指定符的存在，`Derived1`类的虚拟`bar(char const)`方法将不会是一个覆盖方法，而是一个从`Base`类重载的`bar()`方法。

另一个特殊标识符`final`用于成员函数的声明或定义中，以指示该函数是虚拟的，并且在派生类中不能被覆盖。如果派生类尝试覆盖虚拟函数，编译器将触发错误：

```cpp
class Derived2 : public Derived1
{
  virtual void foo() final {}
};
class Derived3 : public Derived2
{
  virtual void foo() override {} // error
}; 
```

`final`指定符也可以在类声明中使用，以指示它不能被派生：

```cpp
class Derived4 final : public Derived1
{
  virtual void foo() override {}
};
class Derived5 : public Derived4 // error
{
}; 
```

由于`override`和`final`在定义的上下文中具有这种特殊含义，并且实际上不是保留关键字，因此您仍然可以在 C++代码的任何其他地方使用它们。这确保了在 C++11 之前编写的现有代码不会因为使用这些名称作为标识符而中断：

```cpp
class foo
{
  int final = 0;
  void override() {}
}; 
```

尽管之前给出的建议建议在重写虚拟方法的声明中使用`virtual`和`override`，但`virtual`关键字是可选的，可以省略以缩短声明。存在`override`指定符应该足以向读者表明该方法虚拟。这更多的是个人偏好的问题，不会影响语义。

## 参见

+   *第十章*，*使用 curiously recurring template pattern 实现静态多态*，了解 CRTP 模式如何帮助在编译时实现多态

# 使用基于范围的 for 循环遍历范围

许多编程语言支持一种名为`for each`的`for`循环变体——即，重复一组语句遍历集合中的元素。C++直到 C++11 之前都没有对这种功能提供核心语言支持。最接近的功能是标准库中的通用算法`std::for_each`，它将一个函数应用于范围中的所有元素。C++11 引入了对`for each`的语言支持，实际上称为基于范围的 for 循环。新的 C++17 标准为原始语言特性提供了几个改进。

## 准备工作

在 C++11 中，基于范围的 for 循环具有以下通用语法：

```cpp
for ( range_declaration : range_expression ) loop_statement 
```

在 C++20 中，初始化语句（必须以分号结束）可以在范围声明之前存在。因此，一般形式变为以下内容：

```cpp
for(init-statement range-declaration : range-expression)
loop-statement 
```

为了说明使用基于范围的 for 循环的各种方式，我们将使用以下函数，它们返回元素序列：

```cpp
std::vector<int> getRates()
{
  return std::vector<int> {1, 1, 2, 3, 5, 8, 13};
}
std::multimap<int, bool> getRates2()
{
  return std::multimap<int, bool> {
    { 1, true },
    { 1, true },
    { 2, false },
    { 3, true },
    { 5, true },
    { 8, false },
    { 13, true }
  };
} 
```

在下一节中，我们将探讨我们可以使用基于范围的 for 循环的各种方式。

## 如何做到这一点...

基于范围的 for 循环可以用各种方式使用：

+   通过为序列的元素指定特定类型：

    ```cpp
    auto rates = getRates();
    for (int rate : rates)
      std::cout << rate << '\n';
    for (int& rate : rates)
      rate *= 2; 
    ```

+   通过不指定类型，让编译器推断它：

    ```cpp
    for (auto&& rate : getRates())
      std::cout << rate << '\n';
    for (auto & rate : rates)
      rate *= 2;
    for (auto const & rate : rates)
      std::cout << rate << '\n'; 
    ```

+   通过在 C++17 中使用结构化绑定和分解声明：

    ```cpp
    for (auto&& [rate, flag] : getRates2())
      std::cout << rate << '\n'; 
    ```

## 它是如何工作的...

在*如何做到这一点...*部分之前显示的基于范围的 for 循环的表达式基本上是语法糖，因为编译器将其转换为其他内容。在 C++17 之前，编译器生成的代码通常是以下内容：

```cpp
{
  auto && __range = range_expression;
  for (auto __begin = begin_expr, __end = end_expr;
  __begin != __end; ++__begin) {
    range_declaration = *__begin;
    loop_statement
  }
} 
```

`begin_expr`和`end_expr`在这个代码中的含义取决于范围类型：

+   对于 C 样式的数组：`__range`和`__range + __bound`（其中`__bound`是数组中元素的数量）。

+   对于具有`begin`和`end`成员的类类型（无论其类型和可访问性）：`__range.begin()`和`__range.end()`。

+   对于其他情况，它是`begin(__range)`和`end(__range)`，这些是通过参数依赖查找确定的。

需要注意的是，如果一个类包含任何名为`begin`或`end`的成员（函数、数据成员或枚举器），无论其类型和可访问性如何，它们将被用于`begin_expr`和`end_expr`。因此，这种类类型不能用于基于范围的 for 循环。

在 C++17 中，编译器生成的代码略有不同：

```cpp
{
  auto && __range = range_expression;
  auto __begin = begin_expr;
  auto __end = end_expr;
  for (; __begin != __end; ++__begin) {
    range_declaration = *__begin;
    loop_statement
  }
} 
```

新标准已经取消了 `begin` 表达式和 `end` 表达式必须是相同类型的约束。`end` 表达式不需要是一个实际的迭代器，但它必须能够与迭代器进行比较。这个好处是范围可以通过谓词来界定。相反，`end` 表达式只计算一次，而不是每次循环迭代时都计算，这可能会提高性能。

如前所述，在 C++20 中，范围声明之前可以有一个初始化语句。这导致编译器为基于范围的 for 循环生成的代码具有以下形式：

```cpp
{
  init-statement
  auto && __range = range_expression;
  auto __begin = begin_expr;
  auto __end = end_expr;
  for (; __begin != __end; ++__begin) {
    range_declaration = *__begin;
    loop_statement
  }
} 
```

初始化语句可以是一个空语句、表达式语句、简单声明，或者从 C++23 开始，是一个别名声明。以下是一个示例：

```cpp
for (auto rates = getRates(); int rate : rates)
{
   std::cout << rate << '\n';
} 
```

在 C++23 之前，这有助于避免范围表达式中的临时变量引起的未定义行为。`range-expression` 返回的临时变量的生命周期被扩展到循环结束。然而，如果它们将在 `range-expression` 结束时被销毁，则不会扩展 `range-expression` 内部临时变量的生命周期。

我们将通过以下代码片段来解释这一点：

```cpp
struct item
{
   std::vector<int> getRates()
 {
      return std::vector<int> {1, 1, 2, 3, 5, 8, 13};
   }
};
item make_item()
{
   return item{};
}
// undefined behavior, until C++23
for (int rate : make_item().getRates())
{
   std::cout << rate << '\n';
} 
```

由于 `make_item()` 通过值返回，我们在 `range-expression` 中有一个临时变量。这引入了未定义的行为，可以通过以下初始化语句避免：

```cpp
for (auto item = make_item(); int rate : item.getRates())
{
   std::cout << rate << '\n';
} 
```

在 C++23 中，这个问题不再出现，因为该版本的规范还扩展了 `range-expression` 中所有临时变量的生命周期，直到循环结束。

## 参见

+   *为自定义类型启用基于范围的 for 循环*，了解如何使用户定义的类型能够与基于范围的 for 循环一起使用

+   *第十二章*，*使用范围库遍历集合*，了解 C++20 范围库的基本知识

+   *第十二章*，*创建自己的范围视图*，了解如何通过用户定义的范围适配器扩展 C++20 范围库的功能

# 为自定义类型启用基于范围的 for 循环

正如我们在前面的配方中看到的，基于范围的 for 循环，在其他编程语言中称为 `for each`，允许您遍历范围中的元素，提供了一种比标准 `for` 循环更简化的语法，并在许多情况下使代码更易于阅读。然而，基于范围的 for 循环并不与任何表示范围的类型直接工作，而是需要存在 `begin()` 和 `end()` 函数（对于非数组类型），无论是作为成员函数还是自由函数。在本配方中，我们将学习如何使自定义类型能够在基于范围的 for 循环中使用。

## 准备工作

如果您需要了解基于范围的 for 循环如何工作，以及编译器为这种循环生成的代码，建议在继续阅读本部分之前先阅读 *使用基于范围的 for 循环遍历范围* 的配方。

为了展示我们如何为表示序列的自定义类型启用基于范围的 for 循环，我们将使用以下简单数组的实现：

```cpp
template <typename T, size_t const Size>
class dummy_array
{
  T data[Size] = {};
public:
  T const & GetAt(size_t const index) const
 {
    if (index < Size) return data[index];
    throw std::out_of_range("index out of range");
  }
  void SetAt(size_t const index, T const & value)
 {
    if (index < Size) data[index] = value;
    else throw std::out_of_range("index out of range");
  }
  size_t GetSize() const { return Size; }
}; 
```

本食谱的目的是使编写如下代码成为可能：

```cpp
dummy_array<int, 3> arr;
arr.SetAt(0, 1);
arr.SetAt(1, 2);
arr.SetAt(2, 3);
for(auto&& e : arr)
{
  std::cout << e << '\n';
} 
```

实现所有这些所需步骤的详细描述将在以下章节中介绍。

## 如何实现...

要使自定义类型能够用于基于范围的 `for` 循环，你需要做以下事情：

+   为该类型创建可变和常量迭代器，这些迭代器必须实现以下运算符：

    +   `operator++`（前缀和后缀版本）用于递增迭代器

    +   `operator*` 用于解引用迭代器并访问迭代器所指向的实际元素

    +   `operator!=` 用于比较迭代器与另一个迭代器以进行不等性比较

+   为该类型提供免费的 `begin()` 和 `end()` 函数。

给定前面的简单范围示例，我们需要提供以下内容：

+   以下是一个迭代器类的最小实现：

    ```cpp
    template <typename T, typename C, size_t const Size>
    class dummy_array_iterator_type
    {
    public:
      dummy_array_iterator_type(C& collection,
                                size_t const index) :
      index(index), collection(collection)
      { }
      bool operator!= (dummy_array_iterator_type const & other) const
      {
        return index != other.index;
      }
      T const & operator* () const
      {
        return collection.GetAt(index);
      }
      dummy_array_iterator_type& operator++()
      {
        ++index;
        return *this;
      }
      dummy_array_iterator_type operator++(int)
      {
        auto temp = *this;
        ++*this;
        return temp;
      }
    private:
      size_t   index;
      C&       collection;
    }; 
    ```

+   可变和常量迭代器的别名模板：

    ```cpp
    template <typename T, size_t const Size>
    using dummy_array_iterator =
      dummy_array_iterator_type<
        T, dummy_array<T, Size>, Size>;
    template <typename T, size_t const Size>
    using dummy_array_const_iterator =
      dummy_array_iterator_type<
        T, dummy_array<T, Size> const, Size>; 
    ```

+   提供免费的 `begin()` 和 `end()` 函数，这些函数返回相应的开始和结束迭代器，并为这两个别名模板提供重载：

    ```cpp
    template <typename T, size_t const Size>
    inline dummy_array_iterator<T, Size> begin(
      dummy_array<T, Size>& collection)
    {
      return dummy_array_iterator<T, Size>(collection, 0);
    }
    template <typename T, size_t const Size>
    inline dummy_array_iterator<T, Size> end(
      dummy_array<T, Size>& collection)
    {
      return dummy_array_iterator<T, Size>(
        collection, collection.GetSize());
    }
    template <typename T, size_t const Size>
    inline dummy_array_const_iterator<T, Size> begin(
      dummy_array<T, Size> const & collection)
    {
      return dummy_array_const_iterator<T, Size>(
        collection, 0);
    }
    template <typename T, size_t const Size>
    inline dummy_array_const_iterator<T, Size> end(
      dummy_array<T, Size> const & collection)
    {
      return dummy_array_const_iterator<T, Size>(
        collection, collection.GetSize());
    } 
    ```

## 如何工作...

在此实现可用的情况下，前面展示的基于范围的 for 循环将按预期编译和执行。在执行参数依赖查找时，编译器将识别我们编写的两个 `begin()` 和 `end()` 函数（它们接受对 `dummy_array` 的引用），因此，它生成的代码是有效的。

在前面的例子中，我们定义了一个迭代器类模板和两个别名模板，分别称为 `dummy_array_iterator` 和 `dummy_array_const_iterator`。`begin()` 和 `end()` 函数都有这两种迭代器类型的两个重载。

这是有必要的，这样我们考虑的容器就可以在基于范围的 for 循环中与常量和非常量实例一起使用：

```cpp
template <typename T, const size_t Size>
void print_dummy_array(dummy_array<T, Size> const & arr)
{
  for (auto && e : arr)
  {
    std::cout << e << '\n';
  }
} 
```

为了使简单范围类能够使用基于范围的 for 循环，我们考虑的一个可能的替代方案是提供成员函数 `begin()` 和 `end()`。一般来说，这只有在你可以拥有并修改源代码的情况下才有意义。相反，本食谱中展示的解决方案适用于所有情况，并且应该优先于其他替代方案。

## 参见

+   *创建类型别名和别名模板*，了解类型别名的知识

+   *第十二章*，*使用 ranges 库遍历集合*，了解 C++20 ranges 库的基本知识

# 使用显式构造函数和转换运算符来避免隐式转换

在 C++11 之前，只有一个参数的构造函数被认为是转换构造函数（因为它接受另一个类型的值并从中创建一个新的类实例）。从 C++11 开始，每个没有`explicit`指定符的构造函数都被认为是转换构造函数。这很重要，因为这样的构造函数定义了从其参数类型或类型到类类型的隐式转换。类还可以定义将类类型转换为另一个指定类型的转换运算符。所有这些在某些情况下都是有用的，但在其他情况下可能会造成问题。在这个食谱中，我们将学习如何使用显式构造函数和转换运算符。

## 准备工作

对于这个食谱，你需要熟悉构造函数和转换运算符的转换。在这个食谱中，你将学习如何编写显式构造函数和转换运算符以避免隐式转换到或从某个类型。显式构造函数和转换运算符（称为*用户定义的转换函数*）的使用使得编译器能够产生错误——在某些情况下，这些错误是编码错误——并允许开发者快速发现这些错误并修复它们。

## 如何操作...

要声明显式构造函数和显式转换运算符（无论它们是函数还是函数模板），在声明中使用`explicit`指定符。

以下示例显示了显式构造函数和显式转换运算符：

```cpp
struct handle_t
{
  explicit handle_t(int const h) : handle(h) {}
  explicit operator bool() const { return handle != 0; };
private:
  int handle;
}; 
```

## 它是如何工作的...

要理解显式构造函数的必要性以及它们是如何工作的，我们首先将查看转换构造函数。以下名为`foo`的类有三个构造函数：一个不带参数的默认构造函数、一个接受`int`的构造函数和一个接受两个参数（一个`int`和一个`double`）的构造函数。它们除了打印一条消息外不做任何事情。截至 C++11，这些都被认为是转换构造函数。该类还有一个转换运算符，它将`foo`类型的值转换为`bool`：

```cpp
struct foo
{
  foo()
  { std::cout << "foo" << '\n'; }
  foo(int const a)
  { std::cout << "foo(a)" << '\n'; }
  foo(int const a, double const b)
  { std::cout << "foo(a, b)" << '\n'; }
  operator bool() const { return true; }
}; 
```

基于此，以下对象的定义是可能的（注意，注释代表控制台的输出）：

```cpp
foo f1;              // foo()
foo f2 {};           // foo()
foo f3(1);           // foo(a)
foo f4 = 1;          // foo(a)
foo f5 { 1 };        // foo(a)
foo f6 = { 1 };      // foo(a)
foo f7(1, 2.0);      // foo(a, b)
foo f8 { 1, 2.0 };   // foo(a, b)
foo f9 = { 1, 2.0 }; // foo(a, b) 
```

变量`f1`和`f2`调用默认构造函数。`f3`、`f4`、`f5`和`f6`调用接受`int`的构造函数。请注意，所有这些对象的定义都是等效的，尽管它们看起来不同（`f3`使用函数形式初始化，`f4`和`f6`是复制初始化，而`f5`直接使用花括号初始化列表初始化）。同样，`f7`、`f8`和`f9`调用具有两个参数的构造函数。

在这种情况下，`f5`和`f6`将`print foo(l)`，而`f8`和`f9`将生成编译器错误（尽管编译器可能有选项忽略一些警告，例如 GCC 的`-Wno-narrowing`），因为初始化列表中的所有元素都应该为整数。

可能需要注意，如果`foo`定义了一个接受`std::initializer_list`的构造函数，那么所有使用`{}`的初始化都将解析为该构造函数：

```cpp
foo(std::initializer_list<int> l)
{ std::cout << "foo(l)" << '\n'; } 
```

这些可能看起来都是正确的，但隐式转换构造函数允许出现隐式转换可能不是我们想要的情况。首先，让我们看看一些正确的例子：

```cpp
void bar(foo const f)
{
}
bar({});             // foo()
bar(1);              // foo(a)
bar({ 1, 2.0 });     // foo(a, b) 
```

`foo`类到`bool`的转换运算符也使我们能够在期望布尔值的地方使用`foo`对象。以下是一个例子：

```cpp
bool flag = f1;                // OK, expect bool conversion
if(f2) { /* do something */ }  // OK, expect bool conversion
std::cout << f3 + f4 << '\n';  // wrong, expect foo addition
if(f5 == f6) { /* do more */ } // wrong, expect comparing foos 
```

前两个例子是`foo`被期望用作布尔值的例子。然而，最后两个，一个用于加法和一个用于测试相等性，可能是不正确的，因为我们最可能期望添加`foo`对象并测试`foo`对象是否相等，而不是它们隐式转换成的布尔值。

可能一个更现实的例子来理解可能出现问题的场景是考虑一个字符串缓冲区实现。这将是一个包含字符内部缓冲区的类。

本类提供了几个转换构造函数：一个默认构造函数，一个接受一个表示预分配缓冲区大小的`size_t`参数的构造函数，以及一个接受`char`指针的构造函数，该指针应用于分配和初始化内部缓冲区。简而言之，我们用于本例的字符串缓冲区实现看起来如下：

```cpp
class string_buffer
{
public:
  string_buffer() {}
  string_buffer(size_t const size) { data.resize(size); }
  string_buffer(char const * const ptr) : data(ptr) {}
  size_t size() const { return data.size(); }
  operator bool() const { return !data.empty(); }
  operator char const * () const { return data.c_str(); }
private:
   std::string data;
}; 
```

根据这个定义，我们可以构造以下对象：

```cpp
std::shared_ptr<char> str;
string_buffer b1;            // calls string_buffer()
string_buffer b2(20);        // calls string_buffer(size_t const)
string_buffer b3(str.get()); // calls string_buffer(char const*) 
```

对象`b1`使用默认构造函数创建，因此具有空缓冲区；`b2`使用单参数构造函数进行初始化，其中参数的值表示内部缓冲区的字符大小；`b3`使用现有的缓冲区进行初始化，该缓冲区用于定义内部缓冲区的大小并将值复制到内部缓冲区。然而，相同的定义也允许以下对象定义：

```cpp
enum ItemSizes {DefaultHeight, Large, MaxSize};
string_buffer b4 = 'a';
string_buffer b5 = MaxSize; 
```

在这种情况下，`b4`使用一个`char`进行初始化。由于存在到`size_t`的隐式转换，将调用单参数的构造函数。这里的意图不一定清楚；也许它应该是`"a"`而不是`'a'`，在这种情况下，将调用第三个构造函数。

然而，`b5`很可能是错误，因为`MaxSize`是一个表示`ItemSizes`的枚举器，应该与字符串缓冲区大小无关。这些错误情况在编译器中没有任何标记。未限定的枚举到`int`的隐式转换是倾向于使用限定的枚举（使用`enum class`声明的）的一个很好的论据，因为它们没有这种隐式转换。如果`ItemSizes`是一个限定的枚举，那么这里描述的情况就不会出现。

当在构造函数的声明中使用 `explicit` 指定时，该构造函数成为显式构造函数，不再允许对 `class` 类型的对象进行隐式构造。为了说明这一点，我们将稍微修改 `string_buffer` 类以声明所有构造函数为 `explicit`:

```cpp
class string_buffer
{
public:
  explicit string_buffer() {}
  explicit string_buffer(size_t const size) { data.resize(size); }
  explicit string_buffer(char const * const ptr) :data(ptr) {}
  size_t size() const { return data.size(); }
  explicit operator bool() const { return !data.empty(); }
  explicit operator char const * () const { return data.c_str(); }
private:
   std::string data;
}; 
```

这里的变化很小，但之前示例中 `b4` 和 `b5` 的定义不再有效且是错误的。这是因为重载解析期间不再可用从 `char` 或 `int` 到 `size_t` 的隐式转换来确定应该调用哪个构造函数。结果是 `b4` 和 `b5` 都会出现编译错误。请注意，即使构造函数是显式的，`b1`、`b2` 和 `b3` 仍然是有效的定义。

在这种情况下，解决问题的唯一方法是从 `char` 或 `int` 显式转换为 `string_buffer`:

```cpp
string_buffer b4 = string_buffer('a');
string_buffer b5 = static_cast<string_buffer>(MaxSize);
string_buffer b6 = string_buffer{ "a" }; 
```

使用显式构造函数，编译器能够立即标记出错误情况，开发者可以相应地做出反应，要么使用正确的值修复初始化，要么提供显式转换。

这仅在用复制初始化进行初始化时才成立，而不是在使用函数式或通用初始化时。

以下定义仍然可能（但错误）使用显式构造函数：

```cpp
string_buffer b7{ 'a' };
string_buffer b8('a'); 
```

与构造函数类似，转换运算符可以被声明为显式（如前所述）。在这种情况下，从对象类型到转换运算符指定的类型的隐式转换不再可能，需要显式转换。考虑到 `b1` 和 `b2`，它们是我们之前定义的 `string_buffer` 对象，以下使用显式 `operator bool` 转换将不再可能：

```cpp
std::cout << b4 + b5 << '\n'; // error
if(b4 == b5) {}               // error 
```

相反，它们需要显式转换为 `bool`:

```cpp
std::cout << static_cast<bool>(b4) + static_cast<bool>(b5);
if(static_cast<bool>(b4) == static_cast<bool>(b5)) {} 
```

两个 `bool` 值相加没有太多意义。前面的示例仅用于说明为了使语句编译，需要显式转换。当没有显式静态转换时，编译器发出的错误可以帮助你确定表达式本身是错误的，可能原本意图是其他内容。

## 参见

+   *理解统一初始化*，以了解花括号初始化是如何工作的

# 使用无名命名空间而不是静态全局变量

程序越大，当你的程序链接到多个翻译单元时遇到名称冲突的可能性就越大。在源文件中声明的函数或变量，目的是在翻译单元内部局部使用，可能与另一个翻译单元中声明的其他类似函数或变量冲突。

这是因为所有未声明为静态的符号都具有外部链接，并且它们的名称必须在整个程序中是唯一的。C 语言解决这个问题的典型方法是将这些符号声明为静态，将它们的链接从外部更改为内部，因此使它们成为翻译单元的本地符号。另一种选择是在名称前加上它们所属的模块或库的名称。在本菜谱中，我们将探讨 C++解决这个问题的方法。

## 准备工作

在这个菜谱中，我们将讨论诸如全局函数和静态函数、变量、命名空间和翻译单元等概念。我们期望你已经对这些概念有基本的了解。除此之外，你还必须理解内部链接和外部链接之间的区别；这对于本菜谱至关重要。

## 如何实现...

当你处于需要将全局符号声明为静态以避免链接问题的情境时，你应该优先使用无名的命名空间：

1.  在你的源文件中声明一个无名的命名空间。

1.  将全局函数或变量的定义放在无名的命名空间中，而不将其声明为 `static`。

以下示例展示了在两个不同的翻译单元中调用名为 `print()` 的两个函数；每个函数都在一个无名的命名空间中定义：

```cpp
// file1.cpp
namespace
{
  void print(std::string const & message)
 {
    std::cout << "[file1] " << message << '\n';
  }
}
void file1_run()
{
  print("run");
}
// file2.cpp
namespace
{
  void print(std::string const & message)
 {
    std::cout << "[file2] " << message << '\n';
  }
}
void file2_run()
{
  print("run");
} 
```

## 它是如何工作的...

当一个函数在翻译单元中声明时，它具有外部链接。这意味着来自两个不同翻译单元的两个具有相同名称的函数将生成链接错误，因为不可能有两个具有相同名称的符号。在 C 语言中解决这个问题，有时在 C++中也是如此，是将函数或变量声明为静态，并将它的链接从外部更改为内部。在这种情况下，它的名称不再导出至翻译单元之外，从而避免了链接问题。

在 C++中，正确的解决方案是使用无名的命名空间。当你定义一个类似于前面展示的命名空间时，编译器将其转换成以下形式：

```cpp
// file1.cpp
namespace _unique_name_ {}
using namespace _unique_name_;
namespace _unique_name_
{
  void print(std::string message)
 {
    std::cout << "[file1] " << message << '\n';
  }
}
void file1_run()
{
  print("run");
} 
```

首先，它声明了一个具有唯一名称的命名空间（名称是什么以及它是如何生成该名称的是编译器实现细节，不应成为关注点）。在这个时候，命名空间是空的，这一行的目的是基本建立命名空间。其次，一个 `using` 指令将 `_unique_name_` 命名空间中的所有内容引入当前命名空间。第三，具有编译器生成的名称的命名空间被定义为它原始源代码中的样子（当它没有名称时）。

通过在无名的命名空间中定义翻译单元本地的 `print()` 函数，它们只有本地可见性，但它们的链接外部性不再产生链接错误，因为它们现在具有外部唯一名称。

无名命名空间在涉及模板的某些更不明显的情况下也有效。在 C++11 之前，模板的非类型参数不能具有内部链接的名称，因此使用静态变量是不可能的。相反，无名命名空间中的符号具有外部链接，可以用作模板参数。尽管模板非类型参数的这种链接限制在 C++11 中被取消，但在最新的 VC++编译器版本中仍然存在。以下示例展示了这个问题：

```cpp
template <int const& Size>
class test {};
static int Size1 = 10;
namespace
{
  int Size2 = 10;
}
test<Size1> t1;
test<Size2> t2; 
t1 variable produces a compiler error because the non-type argument expression, Size1, has internal linkage. Conversely, the declaration of the t2 variable is correct because Size2 has an external linkage. (Note that compiling this snippet with Clang and GCC does not produce an error.)
```

## 参见

+   *使用内联命名空间进行符号版本化*，了解如何使用内联命名空间和条件编译来对源代码进行版本控制

# 使用内联命名空间进行符号版本化

C++11 标准引入了一种新的命名空间类型，称为*内联命名空间*，它基本上是一种机制，使得嵌套命名空间中的声明看起来和表现得像它们是周围命名空间的一部分。内联命名空间使用命名空间声明中的`inline`关键字来声明（无名命名空间也可以内联）。这是一个有助于库版本化的特性，在本食谱中，我们将学习如何使用内联命名空间进行符号版本化。通过本食谱，你将学习如何使用内联命名空间和条件编译来对源代码进行版本控制。

## 准备工作

在本食谱中，我们将讨论命名空间和嵌套命名空间、模板和模板特化，以及使用预处理器宏进行条件编译。为了继续本食谱，对这些概念的了解是必要的。

## 如何做到这一点...

为了提供库的多个版本并让用户决定使用哪个版本，请执行以下操作：

+   在命名空间内定义库的内容。

+   在内部内联命名空间内定义库的每个版本或其部分。

+   使用预处理器宏和`#if`指令来启用库的特定版本。

以下示例展示了一个库有两个版本，客户端可以使用：

```cpp
namespace modernlib
{
  #ifndef LIB_VERSION_2
inline namespace version_1
  {
    template<typename T>
 int test(T value) { return 1; }
  }
  #endif
#ifdef LIB_VERSION_2
inline namespace version_2
  {
    template<typename T>
 int test(T value) { return 2; }
  }
  #endif
} 
```

## 它是如何工作的...

内联命名空间的一个成员被视为周围命名空间的一个成员。这样的成员可以是部分特化的、显式实例化的或显式特化的。这是一个传递属性，这意味着如果命名空间`A`包含一个内联命名空间`B`，而`B`又包含一个内联命名空间`C`，那么`C`的成员将作为`B`和`A`的成员出现，而`B`的成员将作为`A`的成员出现。

为了更好地理解内联命名空间为什么有用，让我们考虑一个案例，即开发一个随着时间的推移从第一个版本到第二个版本（以及更进一步的）演变的库。这个库在其名为`modernlib`的命名空间下定义了所有其类型和函数。在第一个版本中，这个库可能看起来像这样：

```cpp
namespace modernlib
{
  template<typename T>
 int test(T value) { return 1; }
} 
```

库的客户端可以执行以下调用并返回值`1`：

```cpp
auto x = modernlib::test(42); 
```

然而，客户端可能会决定如下特化模板函数 `test()`：

```cpp
struct foo { int a; };
namespace modernlib
{
  template<>
  int test(foo value) { return value.a; }
}
auto y = modernlib::test(foo{ 42 }); 
```

在这种情况下，`y` 的值不再是 `1`，而是 `42`，因为调用了用户特定的函数。

到目前为止，一切正常，但作为库的开发者，你决定创建库的第二个版本，同时仍然提供第一个和第二个版本，并让用户通过宏来控制使用哪个版本。在这个第二个版本中，你提供了一个新的 `test()` 函数实现，它不再返回 `1`，而是返回 `2`。

为了能够提供第一个和第二个实现，你将它们放在名为 `version_1` 和 `version_2` 的嵌套命名空间中，并使用预处理器宏条件编译库：

```cpp
namespace modernlib
{
  namespace version_1
  {
    template<typename T>
 int test(T value) { return 1; }
  }
  #ifndef LIB_VERSION_2
using namespace version_1;
  #endif
namespace version_2
  {
    template<typename T>
 int test(T value) { return 2; }
  }
  #ifdef LIB_VERSION_2
using namespace version_2;
  #endif
} 
```

突然之间，客户端代码崩溃了，无论它使用库的第一个版本还是第二个版本。这是因为测试函数现在位于嵌套命名空间内部，而 `foo` 的特化是在 `modernlib` 命名空间中完成的，而实际上它应该在 `modernlib::version_1` 或 `modernlib::version_2` 中完成。这是因为模板的特化必须在声明模板的同一命名空间中完成。

在这种情况下，客户端需要更改代码，如下所示：

```cpp
#define LIB_VERSION_2
#include "modernlib.h"
struct foo { int a; };
namespace modernlib
{
  namespace version_2
  {
    template<>
    int test(foo value) { return value.a; }
  }
} 
```

这是一个问题，因为库泄露了实现细节，客户端需要了解这些细节才能进行模板特化。这些内部细节在 *如何做...* 部分的示例中以内联命名空间的方式被隐藏起来。根据对 `modernlib` 库的定义，具有在 `modernlib` 命名空间中特化 `test()` 函数的客户端代码不再崩溃，因为 `version_1::test()` 或 `version_2::test()`（取决于客户端实际使用的版本）在模板特化时表现得像它是封装的 `modernlib` 命名空间的一部分。现在，实现细节对客户端是隐藏的，客户端只能看到周围的命名空间 `modernlib`。

然而，你应该记住，命名空间 `std` 是为标准保留的，永远不应该内联。此外，如果一个命名空间在其第一次定义时不是内联的，那么它也不应该内联定义。

## 参见

+   *使用未命名的命名空间而不是静态全局变量*，探索匿名命名空间并了解它们如何帮助

+   *第四章*，*条件编译源代码*，了解执行条件编译的各种选项

# 使用结构化绑定来处理多返回值

从函数中返回多个值是非常常见的，但在 C++ 中没有第一类解决方案可以使其以简单的方式实现。开发者必须在通过函数的引用参数返回多个值、定义一个包含多个值的结构或返回 `std::pair` 或 `std::tuple` 之间进行选择。前两种使用命名变量，这给了它们一个优势，即它们可以清楚地指示返回值的含义，但缺点是它们必须被显式定义。`std::pair` 的成员称为 `first` 和 `second`，而 `std::tuple` 有未命名的成员，只能通过函数调用检索，但可以使用 `std::tie()` 复制到命名变量。这些解决方案都不是理想的。

C++17 将 `std::tie()` 的语义使用扩展为第一类核心语言特性，该特性允许将元组的值解包到命名变量中。这个特性被称为 *结构化绑定*。

## 准备工作

对于这个菜谱，你应该熟悉标准实用类型 `std::pair` 和 `std::tuple` 以及实用函数 `std::tie()`。

## 如何做到...

要使用支持 C++17 的编译器从函数中返回多个值，你应该做以下操作：

1.  使用 `std::tuple` 作为返回类型：

    ```cpp
    std::tuple<int, std::string, double> find()
    {
      return {1, "marius", 1234.5};
    } 
    ```

1.  使用结构化绑定将元组的值解包到命名对象中：

    ```cpp
    auto [id, name, score] = find(); 
    ```

1.  使用结构绑定将返回的值绑定到 `if` 语句或 `switch` 语句内部的变量：

    ```cpp
    if (auto [id, name, score] = find(); score > 1000)
    {
      std::cout << name << '\n';
    } 
    ```

## 它是如何工作的...

结构化绑定（有时被称为 *分解声明*）是一种语言特性，它的工作方式与 `std::tie()` 类似，除了我们不需要为每个需要使用 `std::tie()` 显式解包的值定义命名变量。使用结构绑定，我们使用 `auto` 说明符在单个定义中定义所有命名变量，以便编译器可以推断每个变量的正确类型。

为了举例说明，让我们考虑将项目插入到 `std::map` 的情况。`insert` 方法返回一个 `std::pair`，包含插入元素或阻止插入的元素的迭代器，以及一个布尔值，指示插入是否成功。以下代码非常明确，使用 `second` 或 `first->second` 使得代码更难阅读，因为你需要不断弄清楚它们代表什么：

```cpp
std::map<int, std::string> m;
auto result = m.insert({ 1, "one" });
std::cout << "inserted = " << result.second << '\n'
          << "value = " << result.first->second << '\n'; 
```

之前的代码可以通过使用 `std::tie` 来提高可读性，它将元组解包成单个对象（并且与 `std::pair` 一起工作，因为 `std::tuple` 从 `std::pair` 有转换赋值）：

```cpp
std::map<int, std::string> m;
std::map<int, std::string>::iterator it;
bool inserted;
std::tie(it, inserted) = m.insert({ 1, "one" });
std::cout << "inserted = " << inserted << '\n'
          << "value = " << it->second << '\n';
std::tie(it, inserted) = m.insert({ 1, "two" });
std::cout << "inserted = " << inserted << '\n'
          << "value = " << it->second << '\n'; 
```

代码不一定更简单，因为它需要提前定义对偶解包到的对象。同样，元组包含的元素越多，你需要定义的对象就越多，但使用命名对象可以使代码更容易阅读。

C++17 结构化绑定将解包元组元素到命名对象提升为语言特性的级别；不需要使用 `std::tie()`，并且对象在声明时被初始化：

```cpp
std::map<int, std::string> m;
{
  auto [it, inserted] = m.insert({ 1, "one" });
  std::cout << "inserted = " << inserted << '\n'
            << "value = " << it->second << '\n';
}
{
  auto [it, inserted] = m.insert({ 1, "two" });
  std::cout << "inserted = " << inserted << '\n'
            << "value = " << it->second << '\n';
} 
```

在前面的例子中使用多个块是必要的，因为变量不能在同一个块中重新声明，而结构化绑定意味着使用 `auto` 指示符进行声明。因此，如果您需要多次调用，如前面的示例所示，并使用结构化绑定，您必须使用不同的变量名或多个块。另一个选择是避免使用结构化绑定并使用 `std::tie()`，因为它可以用相同的变量多次调用，因此您只需声明一次。

在 C++17 中，也可以分别以 `if(init; condition)` 和 `switch(init; condition)` 的形式在 `if` 和 `switch` 语句中声明变量。这可以与结构化绑定结合，以生成更简单的代码。让我们来看一个示例：

```cpp
if(auto [it, inserted] = m.insert({ 1, "two" }); inserted)
{ std::cout << it->second << '\n'; } 
it and inserted, defined in the scope of the if statement in the initialization part. Then, the condition of the if statement is evaluated from the value of the inserted variable.
```

## 还有更多...

尽管我们专注于将名称绑定到元组的元素上，但结构化绑定可以在更广泛的范围内使用，因为它们还支持绑定到数组元素或类的数据成员。如果您想绑定到数组的元素上，您必须为每个数组元素提供一个名称；否则，声明是不合法的。以下是一个绑定到数组元素的示例：

```cpp
int arr[] = { 1,2 };
auto [a, b] = arr;
auto& [x, y] = arr;
arr[0] += 10;
arr[1] += 10;
std::cout << arr[0] << ' ' << arr[1] << '\n'; // 11 12
std::cout << a << ' ' << b << '\n';           // 1 2
std::cout << x << ' ' << y << '\n';           // 11 12 
```

在这个例子中，`arr` 是一个包含两个元素的数组。我们首先将 `a` 和 `b` 绑定到其元素上，然后将 `x` 和 `y` 引用绑定到其元素上。对数组元素所做的更改通过变量 `a` 和 `b` 是不可见的，但通过 `x` 和 `y` 引用是可见的，如注释中打印到控制台这些值的示例所示。这是因为当我们进行第一次绑定时，会创建数组的副本，`a` 和 `b` 被绑定到副本的元素上。

正如我们之前提到的，也可以绑定到类的数据成员上。以下有一些限制：

+   绑定仅适用于类的非静态成员。

+   类不能有匿名联合成员。

+   标识符的数量必须与类的非静态成员的数量匹配。

标识符的绑定按照数据成员声明的顺序进行，这可以包括位域。以下是一个示例：

```cpp
struct foo
{
   int         id;
   std::string name;
};
foo f{ 42, "john" };
auto [i, n] = f;
auto& [ri, rn] = f;
f.id = 43;
std::cout << f.id << ' ' << f.name << '\n';   // 43 john
std::cout << i <<'''' << n <<''\'';           // 42 john
std::cout << ri <<'''' << rn <<''\'';         // 43 john 
```

再次，对 `foo` 对象的更改对变量 `i` 和 `n` 是不可见的，但对 `ri` 和 `rn` 是可见的。这是因为结构绑定中的每个标识符都成为指向类数据成员（就像数组一样，它指向数组的元素）的 lvalue 的名称。然而，标识符的引用类型是对应的数据成员（或数组元素）。

新的 C++20 标准引入了一系列对结构化绑定的改进，包括以下内容：

+   在结构绑定声明中包含 `static` 或 `thread_local` 存储类指定符的可能性。

+   使用 `[[maybe_unused]]` 属性声明结构化绑定。一些编译器，如 Clang 和 GCC，已经支持此功能。

+   在 lambda 中捕获结构绑定标识符的可能性。所有标识符，包括绑定到位字段的标识符，都可以按值捕获。相反，除了绑定到位字段的标识符之外的所有标识符也可以按引用捕获。

这些更改使我们能够编写以下内容：

```cpp
foo f{ 42,"john" };
auto [i, n] = f;
auto l1 = [i] {std::cout << i; };
auto l2 = [=] {std::cout << i; };
auto l3 = [&i] {std::cout << i; };
auto l4 = [&] {std::cout << i; }; 
```

这些示例展示了在 C++20 中结构化绑定可以以各种方式在 lambda 中捕获的各种方法。

有时，我们需要绑定我们不使用的变量。在 C++26 中，将可以使用下划线（`_`）而不是名称来忽略一个变量。尽管在撰写本文时没有任何编译器支持此功能，但该功能已被包含在 C++26 中。

```cpp
foo f{ 42,"john" };
auto [_, n] = f; 
```

在这里，`_` 是一个占位符，用于绑定到 `foo` 对象的 `id` 成员。它用于表示此值在此上下文中未使用且将被忽略。

使用 `_` 占位符不仅限于结构化绑定。它可以用作非静态类成员、结构化绑定和 lambda 捕获的标识符。您可以使用下划线重新定义同一作用域中已存在的声明，因此可以忽略多个变量。然而，如果变量名为 `_` 在重新声明之后使用，则程序被认为是格式不正确的。

## 参见

+   *尽可能使用 auto*，了解 C++中自动类型推导的工作原理

+   *第三章*，*使用标准算法中的 lambda*，了解 lambda 如何与标准库通用算法一起使用

+   *第四章*，*使用属性向编译器提供元数据*，了解如何使用标准属性向编译器提供提示

# 使用类模板参数推导简化代码

模板在 C++中无处不在，但总是需要指定模板参数可能会很烦人。有些情况下，编译器实际上可以从上下文中推断模板参数。此功能在 C++17 中可用，称为*类模板参数推导*，它使编译器能够从初始化器的类型推断缺失的模板参数。在本食谱中，我们将学习如何利用此功能。

## 如何做到...

在 C++17 中，您可以在以下情况下省略指定模板参数，让编译器推断它们：

+   当您声明一个变量或变量模板并对其进行初始化时：

    ```cpp
    std::pair   p{ 42, "demo" };  // deduces std::pair<int, char const*>
    std::vector v{ 1, 2 };        // deduces std::vector<int>
    std::less   l;                // deduces std::less<void> 
    ```

+   当您使用 new 表达式创建对象时：

    ```cpp
    template <class T>
    struct foo
    {
       foo(T v) :data(v) {}
    private:
       T data;
    };
    auto f = new foo(42); 
    ```

+   当您执行函数式类型转换表达式时：

    ```cpp
    std::mutex mx;
    // deduces std::lock_guard<std::mutex>
    auto lock = std::lock_guard(mx);
    std::vector<int> v;
    // deduces std::back_insert_iterator<std::vector<int>>
    std::fill_n(std::back_insert_iterator(v), 5, 42); 
    ```

## 它是如何工作的...

在 C++17 之前，您必须在初始化变量时指定所有模板参数，因为所有这些都必须已知才能实例化类模板，例如以下示例：

```cpp
std::pair<int, char const*> p{ 42, "demo" };
std::vector<int>            v{ 1, 2 };
foo<int>                    f{ 42 }; 
```

使用函数模板，例如`std::make_pair()`，可以避免显式指定模板参数的问题，它受益于函数模板参数推导，并允许我们编写如下代码：

```cpp
auto p = std::make_pair(42, "demo"); 
```

在这里展示的`foo`类模板的情况下，我们可以编写以下`make_foo()`函数模板来启用相同的行为：

```cpp
template <typename T>
constexpr foo<T> make_foo(T&& value)
{
   return foo{ value };
}
auto f = make_foo(42); 
```

在 C++17 中，在*如何工作...*部分列出的情况下，这不再必要。以下是一个示例声明：

```cpp
std::pair p{ 42, "demo" }; 
```

在这个上下文中，`std::pair`不是一个类型，但它作为一个类型占位符，激活了类模板参数推导。当编译器在声明带有初始化或函数式转换的变量时遇到它，它将构建一个推导指南集。这些推导指南是假设类类型的虚构构造函数。

作为用户，你可以通过用户定义的推导规则来补充这个集合。这个集合用于执行模板参数推导和重载解析。

在`std::pair`的情况下，编译器将构建一个包含以下虚构函数模板的推导指南集（但不仅限于此）：

```cpp
template <class T1, class T2>
std::pair<T1, T2> F();
template <class T1, class T2>
std::pair<T1, T2> F(T1 const& x, T2 const& y);
template <class T1, class T2, class U1, class U2>
std::pair<T1, T2> F(U1&& x, U2&& y); 
```

这些由编译器生成的推导指南是从类模板的构造函数中创建的，如果没有提供，则创建一个假设默认构造函数的推导指南。此外，在所有情况下，都会创建一个假设复制构造函数的推导指南。

用户定义的推导指南是具有尾随返回类型且不带`auto`关键字的函数签名（因为它们代表没有返回值的假设构造函数）。它们必须在应用于该类模板的命名空间中定义。

为了理解它是如何工作的，让我们考虑与`std::pair`对象相同的示例：

```cpp
std::pair p{ 42, "demo" }; 
```

编译器推导出的类型是`std::pair<int, char const*>`。如果我们想指示编译器推导出`std::string`而不是`char const*`，那么我们需要几个用户定义的推导规则，如下所示：

```cpp
namespace std {
   template <class T>
   pair(T&&, char const*)->pair<T, std::string>;
   template <class T>
   pair(char const*, T&&)->pair<std::string, T>;
   pair(char const*, char const*)->pair<std::string, std::string>;
} 
```

这些将使我们能够执行以下声明，其中字符串`"demo"`的类型始终推导为`std::string`：

```cpp
std::pair  p1{ 42, "demo" };    // std::pair<int, std::string>
std::pair  p2{ "demo", 42 };    // std::pair<std::string, int>
std::pair  p3{ "42", "demo" };  // std::pair<std::string, std::string> 
```

如此示例所示，推导指南不必是函数模板。

重要的一点是，如果存在模板参数列表，则不会发生类模板参数推导，无论指定了多少个参数。以下是一些示例：

```cpp
std::pair<>    p1 { 42, "demo" };
std::pair<int> p2 { 42, "demo" }; 
```

由于这两个声明都指定了模板参数列表，它们是无效的，并产生编译器错误。

有一些已知的情况，其中类模板参数推导不起作用：

+   聚合模板，其中你可以编写用户定义的推导指南来规避这个问题。

    ```cpp
    template<class T>
    struct Point3D { T x; T y; T z; }; 

    Point3D p{1, 2, 2};   // error, requires Point3D<int> 
    ```

+   类型别名，如下面的示例所示（对于 GCC，在编译时使用`-std=c++20`实际上可以工作）：

    ```cpp
    template <typename T>
    using my_vector = std::vector<T>;
    std::vector v{1,2,3}; // OK
    my_vector mv{1,2,3};  // error 
    ```

+   继承构造函数，因为推导指南（无论是隐式还是用户定义的）在继承构造函数时不会被继承：

    ```cpp
    template <typename T> 
    struct box
    {
       box(T&& t) : content(std::forward<T>(t)) {}
       virtual void unwrap()
     { std::cout << "unwrapping " << content << '\n'; }
       T content;
    };
    template <typename T>
    struct magic_box : public box<T>
    {
       using box<T>::box;
       virtual void unwrap() override
     { std::cout << "unwrapping " << box<T>::content << '\n'; }
    };
    int main()
    {
       box b(42);        // OK
       b.unwrap();
       magic_box m(21);  // error, requires magic_box<int>
       m.unwrap();
    } 
    ```

这种限制在 C++23 中被移除，因为在继承构造函数时，推导指南也会被继承。

## 参见

+   *理解统一初始化*，以了解花括号初始化是如何工作的

# 使用下标运算符访问集合中的元素

访问数组元素是 C++以及任何支持数组的编程语言的基本功能。语法在许多编程语言中也是相同的。在 C++中，用于此目的的下标运算符`[]`可以被重载以提供对类中数据的访问。通常，这是对容器进行建模的类的情况。在本食谱中，我们将看到如何利用此运算符以及 C++23 带来了哪些变化。

## 如何做到…

为了提供对容器中元素的随机访问，以下是如何重载下标运算符：

+   对于一维容器，你可以使用一个参数重载下标运算符，无论标准版本如何：

    ```cpp
    template <typename T>
    struct some_buffer
    {
       some_buffer(size_t const size):data(size)
       {}
       size_t size() const { return data.size(); }
       T const& operator[](size_t const index) const
       {
          if(index >= data.size())
             std::runtime_error("invalid index");
          return data[index];
       }
       T & operator[](size_t const index)
       {
          if (index >= data.size())
             std::runtime_error("invalid index");
          return data[index];
       }
    private:
       std::vector<T> data;
    }; 
    ```

+   对于多维容器，在 C++23 中，你可以使用多个参数重载下标运算符：

    ```cpp
    template <typename T, size_t ROWS, size_t COLS>
    struct matrix
    {
       T& operator[](size_t const row, size_t const col)
       {
          if(row >= ROWS || col >= COLS)
             throw std::runtime_error("invalid index");
          return data[row * COLS + col];
       }
       T const & operator[](size_t const row,                         size_t const col) const
       {
          if (row >= ROWS || col >= COLS)
             throw std::runtime_error("invalid index");
          return data[row * COLS + col];
       }
    private:
       std::array<T, ROWS* COLS> data;
    }; 
    ```

## 它是如何工作的…

下标运算符用于访问数组中的元素。然而，它也可以作为类中通常建模容器（或一般集合）的成员函数重载，以访问其元素。标准容器如`std::vector`、`std::set`和`std::map`为此目的提供了下标运算符的重载。因此，你可以编写如下代码：

```cpp
std::vector<int> v {1, 2, 3};
v[2] = v[1] + v[0]; 
```

在上一节中，我们看到了下标运算符可以如何重载。通常有两种重载方式，一种是常量重载，另一种是可变重载。常量重载返回一个指向常量对象的引用，而可变重载返回一个引用。

下标运算符的主要问题是，在 C++23 之前，它只能有一个参数。因此，它不能用来提供对多维容器元素的访问。因此，开发者通常求助于使用调用运算符来达到这个目的。以下是一个示例片段：

```cpp
template <typename T, size_t ROWS, size_t COLS>
struct matrix
{
   T& operator()(size_t const row, size_t const col)
 {
      if(row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
   T const & operator()(size_t const row, size_t const col) const
 {
      if (row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
private:
   std::array<T, ROWS* COLS> data;
};
matrix<int, 2, 3> m;
m(0, 0) = 1; 
```

为了帮助解决这个问题，并允许更一致的方法，C++11 使得可以使用下标运算符的语法`[{expr1, expr2, …}]`。下面展示了一个利用此语法的`matrix`类的修改实现：

```cpp
template <typename T, size_t ROWS, size_t COLS>
struct matrix
{
   T& operator[](std::initializer_list<size_t> index)
   {
      size_t row = *index.begin();
      size_t col = *(index.begin() + 1);
      if (row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
   T const & operator[](std::initializer_list<size_t> index) const
   {
      size_t row = *index.begin();
      size_t col = *(index.begin() + 1);
      if (row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
private:
   std::array<T, ROWS* COLS> data;
};
matrix<int, 2, 3> m;
m[{0, 0}] = 1; 
```

然而，语法相当繁琐，在实践中可能很少使用。因此，C++23 标准使得使用多个参数重载下标运算符成为可能。这里展示了一个修改后的`matrix`类：

```cpp
template <typename T, size_t ROWS, size_t COLS>
struct matrix
{
   T& operator[](size_t const row, size_t const col)
   {
      if(row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
   T const & operator[](size_t const row, size_t const col) const
   {
      if (row >= ROWS || col >= COLS)
         throw std::runtime_error("invalid index");
      return data[row * COLS + col];
   }
private:
   std::array<T, ROWS* COLS> data;
};
matrix<int, 2, 3> m;
m[0, 0] = 1; 
```

这使得调用语法与访问一维容器保持一致。`std::mdspan` 使用它来提供元素访问。这是一个新的 C++23 类，它表示对连续序列（如数组）的非拥有视图，但它将序列重新解释为多维数组。

之前显示的 `matrix` 类实际上可以用数组上的 `mdspan` 视图替换，如下面的代码片段所示：

```cpp
int data[2*3] = {};
auto m = std::mdspan<int, std::extents<2, 3>> (data);
m[0, 0] = 1; 
```

## 参见

+   *第五章*，*编写自己的随机访问迭代器*，了解您如何编写用于访问容器元素的迭代器

+   *第六章*，*使用 std::mdspan 对对象序列进行多维视图*，了解 `std::mdspan` 类的更多信息

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
