

# 探索函数

函数是编程中的基本概念；无论我们讨论什么主题，最终都会谈到函数。试图在一个章节中涵盖关于函数的所有内容不仅困难，而且不太合理。作为语言的基本元素，函数出现在本书的每一道食谱中。然而，这一章涵盖了与函数和可调用对象相关的现代语言特性，重点关注 lambda 表达式、来自函数式语言的概念，如高阶函数和函数模板。

本章包含的食谱如下：

+   默认化和删除函数

+   使用标准算法与 lambda 表达式

+   使用泛型和模板 lambda

+   编写递归 lambda

+   编写函数模板

+   编写具有可变参数数量的函数模板

+   使用折叠表达式简化变长函数模板

+   实现高阶函数 `map` 和 `fold`

+   将函数组合成高阶函数

+   统一调用任何可调用对象

我们将从这个章节开始，学习一个使我们可以更容易地提供特殊类成员函数或防止任何函数（成员或非成员）被调用的特性。

# 默认化和删除函数

在 C++中，类有特殊的成员（构造函数、析构函数和赋值运算符），这些成员可能由编译器默认实现，或者由开发者提供。然而，可以默认实现的规则有点复杂，可能会导致问题。另一方面，开发者有时希望防止对象以特定方式被复制、移动或构造。

这可以通过使用这些特殊成员实现不同的技巧来实现。C++11 标准通过允许函数被删除或默认，简化了其中许多规则，我们将在下一节中看到这些规则。

## 入门

对于这个食谱，你需要熟悉以下概念：

+   特殊成员函数（默认构造函数、析构函数、拷贝构造函数、移动构造函数、拷贝赋值运算符和移动赋值运算符）

+   可拷贝的概念（一个类具有拷贝构造函数和拷贝赋值运算符，使得创建副本成为可能）

+   可移动的概念（一个类具有移动构造函数和移动赋值运算符，使得移动对象成为可能）

考虑到这一点，让我们学习如何定义默认和删除的特殊函数。

## 如何做到这一点...

使用以下语法来指定函数应该如何处理：

+   要将函数默认化，请使用 `=default` 而不是函数体。只有编译器可以提供默认实现的特殊类成员函数才能被默认化：

    ```cpp
    struct foo
    {
      foo() = default;
    }; 
    ```

+   要删除一个函数，请使用 `=delete` 而不是函数体。任何函数，包括非成员函数，都可以被删除：

    ```cpp
    struct foo
    {
      foo(foo const &) = delete;
    };
    void func(int) = delete; 
    ```

使用默认化和删除函数来实现各种设计目标，例如以下示例：

+   要实现一个不可拷贝且隐式不可移动的类，请将拷贝构造函数和拷贝赋值运算符声明为已删除：

    ```cpp
    class foo_not_copyable
    {
    public:
      foo_not_copyable() = default;
      foo_not_copyable(foo_not_copyable const &) = delete;
      foo_not_copyable& operator=(foo_not_copyable const&) = delete;
    }; 
    ```

+   要实现一个不可拷贝但可移动的类，请将拷贝操作声明为已删除，并显式实现移动操作（以及提供所需的任何其他构造函数）：

    ```cpp
    class data_wrapper
    {
      Data* data;
    public:
      data_wrapper(Data* d = nullptr) : data(d) {}
      ~data_wrapper() { delete data; }
      data_wrapper(data_wrapper const&) = delete;
      data_wrapper& operator=(data_wrapper const &) = delete;
      data_wrapper(data_wrapper&& other)  
     :data(std::move(other.data))
      {
        other.data = nullptr;
      }
      data_wrapper& operator=(data_wrapper&& other)
      {
        if (data != other.data))
        {
          delete data;
          data = std::move(other.data);
          other.data = nullptr;
        }
        return *this;
      }
    }; 
    ```

+   要确保函数仅由特定类型的对象调用，并且可能防止类型提升，请为该函数提供已删除的重载（以下示例中的自由函数也可以应用于任何类的成员函数）：

    ```cpp
    template <typename T>
    void run(T val) = delete;
    void run(long val) {} // can only be called with long integers 
    ```

## 它是如何工作的...

一个类有多个可以实现的特殊成员，默认情况下可以由编译器实现。这些是默认构造函数、拷贝构造函数、移动构造函数、拷贝赋值、移动赋值和析构函数（关于移动语义的讨论，请参阅第九章“健壮性和性能”中的*实现移动语义*配方）。如果您不实现它们，则编译器会根据以下规则生成它们。然而，如果您显式提供了一个或多个这些特殊方法，则编译器将不会根据以下规则生成其他方法：

+   如果存在用户定义的构造函数，则默认不生成默认构造函数。

+   如果存在用户定义的虚析构函数，则不生成默认析构函数。

+   如果存在用户定义的移动构造函数或移动赋值运算符，则默认不生成拷贝构造函数和拷贝赋值运算符。

+   如果存在用户定义的拷贝构造函数、移动构造函数、拷贝赋值运算符、移动赋值运算符或析构函数，则默认不生成移动构造函数和移动赋值运算符。

+   如果存在用户定义的拷贝构造函数或析构函数，则默认生成拷贝赋值运算符。

+   如果存在用户定义的拷贝赋值运算符或析构函数，则默认生成拷贝构造函数。

注意，前面列表中的最后两条规则已被弃用，并且可能不再被您的编译器支持。

有时，开发者需要提供这些特殊成员的空实现或隐藏它们，以防止类的实例以特定方式构造。一个典型的例子是一个不应该可拷贝的类。这种情况下，经典的模式是提供一个默认构造函数并隐藏拷贝构造函数和拷贝赋值运算符。虽然这可行，但显式定义的默认构造函数确保该类不再被视为平凡类型，因此是一个**纯旧数据**（**POD**）类型。现代的替代方法是使用已删除的函数，如前节所示。

当编译器在函数的定义中遇到`=default`时，它将提供默认实现。前面提到的特殊成员函数的规则仍然适用。如果函数是内联的，那么只有在类体外部声明函数时才能使用`=default`：

```cpp
class foo
{
public:
  foo() = default;
  inline foo& operator=(foo const &);
};
inline foo& foo::operator=(foo const &) = default; 
```

默认实现有几个好处，包括以下内容：

+   可能比显式实现更高效。

+   非默认实现，即使它们是空的，也被认为是非平凡的，这影响了类型的语义，使得类型变得非平凡（因此，非 POD）。

+   帮助用户不编写显式的默认实现。例如，如果存在用户定义的移动构造函数，那么编译器不会默认提供拷贝构造函数和拷贝赋值运算符。然而，你仍然可以显式地默认它们，并要求编译器提供它们，这样你就不必手动做了。

当编译器在函数的定义中遇到`=delete`时，它将阻止函数的调用。然而，函数在重载解析期间仍然被考虑，只有当删除的函数是最好的匹配时，编译器才会生成错误。例如，通过为之前定义的`run()`函数的重载提供，只有使用长整数的调用是可能的。使用任何其他类型（包括自动提升到`long`的`int`）的参数的调用将确定删除的重载被认为是最佳匹配，因此编译器将生成错误：

```cpp
run(42);  // error, matches a deleted overload
run(42L); // OK, long integer arguments are allowed 
```

注意，之前声明的函数不能被删除，因为`=delete`定义必须是翻译单元中的第一个声明：

```cpp
void forward_declared_function();
// ...
void forward_declared_function() = delete; // error 
```

对于类特殊成员函数的规则（也称为“五规则”）是，如果你明确定义了任何拷贝构造函数、移动构造函数、拷贝赋值运算符、移动赋值运算符或析构函数，那么你必须要么明确定义，要么默认所有这些。

用户定义的析构函数、拷贝构造函数和拷贝赋值运算符是必要的，因为在各种情况下对象都是从副本中构建的（例如将参数传递给函数）。如果它们没有被用户定义，编译器会提供它们，但它们的默认实现可能是不正确的。如果一个类管理资源，那么默认实现执行的是浅拷贝，这意味着它复制了资源句柄的值（例如指向对象的指针）而不是资源本身。在这种情况下，用户定义的实现必须执行深拷贝，即复制资源而不是其句柄。在这种情况下，移动构造函数和移动赋值运算符的存在是可取的，因为它们代表了性能的提升。缺少这两个运算符不是错误，但是一个被错过的优化机会。

一方面与五规则相对立，另一方面与之相补充的是所谓的零规则。该规则指出，除非类处理资源所有权，否则它不应有自定义析构函数、拷贝和移动构造函数，以及相应的拷贝和移动赋值运算符。

在设计类时，你应该遵循以下指南：

+   管理资源的类应该只负责处理该资源的所有权。这样的类必须遵循五规则，并实现自定义析构函数、拷贝/移动构造函数和拷贝/移动赋值运算符。

+   不管理资源的类不应该有自定义析构函数、拷贝/移动构造函数和拷贝/移动赋值运算符（因此遵循零规则）。

## 参见

+   *统一调用任何可调用对象*，了解如何使用`std::invoke()`以提供的参数调用任何可调用对象

# 使用 lambda 表达式与标准算法

C++最现代的特性之一是 lambda 表达式，也称为 lambda 函数或简称为 lambdas。Lambda 表达式使我们能够定义匿名函数对象，这些对象可以捕获作用域内的变量，并作为参数调用或传递给函数。它们避免了定义命名函数或函数对象的必要性。Lambda 表达式在许多用途中都很有用，在这个菜谱中，我们将学习如何使用它们与标准算法一起。

## 准备工作

在这个菜谱中，我们将讨论接受一个函数或谓词作为参数的标准算法，该函数或谓词应用于它迭代的元素。你需要了解一元和二元函数是什么，以及谓词和比较函数是什么。你还应该熟悉函数对象，因为 lambda 表达式是函数对象的语法糖。

## 如何做...

你应该优先使用 lambda 表达式将回调传递给标准算法，而不是函数或函数对象：

+   如果你只需要在调用处定义匿名 lambda 表达式，就使用它：

    ```cpp
    auto numbers =
      std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 };
    auto positives = std::count_if(
      std::begin(numbers), std::end(numbers),
      [](int const n) {return n > 0; }); 
    ```

+   如果你需要在多个地方调用 lambda，定义一个命名的 lambda，即分配给变量的 lambda（通常使用`auto`指定器指定类型）：

    ```cpp
    auto ispositive = [](int const n) {return n > 0; };
    auto positives = std::count_if(
      std::begin(numbers), std::end(numbers), ispositive); 
    ```

+   如果你需要 lambda 表达式仅在参数类型方面有所不同（自 C++14 起可用），请使用泛型 lambda 表达式：

    ```cpp
    auto positives = std::count_if(
      std::begin(numbers), std::end(numbers),
      [](auto const n) {return n > 0; }); 
    ```

## 它是如何工作的...

第二点中显示的非泛型 lambda 表达式接受一个常量整数，如果它大于`0`则返回`true`，否则返回`false`。编译器定义了一个无名的函数对象，具有 lambda 表达式的签名，该签名具有调用操作符：

```cpp
struct __lambda_name__
{
  bool operator()(int const n) const { return n > 0; }
}; 
```

编译器定义未命名函数对象的方式取决于我们定义的可以捕获变量的 lambda 表达式的方式，使用`mutable`指定符或异常指定符，或者有尾随返回类型。前面展示的`__lambda_name__`函数对象实际上是编译器生成的简化版本，因为它还定义了一个默认的拷贝构造函数、默认析构函数和一个删除的赋值运算符。

必须清楚了解 lambda 表达式实际上是一个类。为了调用它，编译器需要实例化类的对象。从 lambda 表达式实例化的对象被称为*lambda 闭包*。

在以下示例中，我们想要计算一个范围中大于或等于 5 且小于或等于 10 的元素数量。在这种情况下，lambda 表达式将看起来像这样：

```cpp
auto numbers = std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 };
auto minimum { 5 };
auto maximum { 10 };
auto inrange = std::count_if(
    std::begin(numbers), std::end(numbers),
    minimum, maximum {
      return minimum <= n && n <= maximum;}); 
```

这个 lambda 通过拷贝（即值）捕获了两个变量，`minimum`和`maximum`。编译器创建的未命名函数对象看起来非常像我们之前定义的。使用前面提到的默认和删除的特殊成员，类看起来像这样：

```cpp
class __lambda_name_2__
{
  int minimum_;
  int maximum_;
public:
  explicit __lambda_name_2__(int const minimum, int const maximum) :
    minimum_( minimum), maximum_( maximum)
  {}
  __lambda_name_2__(const __lambda_name_2__&) = default;
  __lambda_name_2__(__lambda_name_2__&&) = default;
  __lambda_name_2__& operator=(const __lambda_name_2__&)
    = delete;
  ~__lambda_name_2__() = default;
  bool operator() (int const n) const
 {
    return minimum_ <= n && n <= maximum_;
  }
}; 
```

Lambda 表达式可以通过拷贝（或值）或通过引用捕获变量，并且这两种组合的不同组合是可能的。然而，无法多次捕获一个变量，并且捕获列表的开头只能有`&`或`=`。

Lambda 表达式可以访问以下类型的变量：从封装作用域捕获的变量、lambda 参数、在其体内局部声明的变量、当 lambda 在类内部声明且指针被 lambda 捕获时的类数据成员，以及任何具有静态存储期的变量，如全局变量。

Lambda 只能捕获封装函数作用域中的变量。它不能捕获具有静态存储期的变量（即，在命名空间作用域中声明的变量或使用`static`或`external`指定符声明的变量）。

以下表格展示了 lambda 捕获语义的各种组合：

| **Lambda** | **描述** |
| --- | --- |
| `[](){}` | 不捕获任何内容。 |
| `[&](){}` | 通过引用捕获所有内容。 |
| `[=](){}` | 通过拷贝捕获所有内容。在 C++20 中，隐式捕获指针`this`已被弃用。 |
| `[&x](){}` | 仅通过引用捕获`x`。 |
| `[x](){}` | 仅通过拷贝捕获`x`。 |
| `[&x...](){}` | 通过引用捕获 pack 扩展`x`。 |
| `[x...](){}` | 通过拷贝捕获 pack 扩展`x`。 |
| `[&, x](){}` | 通过引用捕获所有内容，除了通过拷贝捕获的`x`。 |
| `[=, &x](){}` | 通过拷贝捕获所有内容，除了通过引用捕获的`x`。 |
| `[&, this](){}` | 通过引用捕获所有内容，除了通过拷贝捕获的指针`this`（`this`总是通过拷贝捕获）。 |
| `[x, x](){}` | 错误；`x`被捕获两次。 |
| `[&, &x](){}` | 错误；所有内容都是通过引用捕获的，我们不能再指定再次通过引用捕获 `x`。 |
| `[=, =x](){}` | 错误；所有内容都是通过复制捕获的，我们不能再指定再次通过复制捕获 `x`。 |
| `[&this](){}` | 错误；指针 `this` 总是通过复制捕获。 |
| `[&, =](){}` | 错误；不能同时通过复制和引用捕获所有内容。 |
| `[x=expr](){}` | `x` 是 lambda 的闭包中的数据成员，由表达式 `expr` 初始化。 |
| `[&x=expr](){}` | `x` 是 lambda 的闭包中的引用数据成员，由表达式 `expr` 初始化。 |

表 3.1：带有解释的 lambda 捕获示例

截至 C++17，lambda 表达式的一般形式如下所示：

```cpp
capture-list mutable constexpr exception attr -> ret
{ body } 
```

此语法中显示的所有部分实际上都是可选的，除了捕获列表，它可以空着，主体也可以空着。如果不需要参数，实际上可以省略参数列表。不需要指定返回类型，因为编译器可以从返回表达式的类型中推断它。`mutable` 说明符（它告诉编译器 lambda 实际上可以修改通过复制捕获的变量，这与通过值捕获不同，因为更改仅在 lambda 内部观察到），`constexpr` 说明符（它告诉编译器生成 `constexpr` 调用操作符），以及异常说明符和属性都是可选的。

最简单的 lambda 表达式是 `[]{}`，尽管它通常写作 `[]()`。

在前面的表格中后两个示例是泛化 lambda 捕获的形式。这些是在 C++14 中引入的，以便我们可以捕获具有移动语义的变量，但它们也可以用于在 lambda 中定义新的任意对象。以下示例显示了如何通过泛化 lambda 捕获以 `move` 的方式捕获变量：

```cpp
auto ptr = std::make_unique<int>(42);
auto l = [lptr = std::move(ptr)](){return ++*lptr;}; 
```

在类方法中编写的 lambda，如果需要捕获类数据成员，可以通过几种方式做到：

+   使用形式 `[x=expr]` 捕获单个数据成员：

    ```cpp
    struct foo
    {
      int         id;
      std::string name;
      auto run()
     {
        return [i=id, n=name] { std::cout << i << ' ' << n << '\n'; };
      }
    }; 
    ```

+   使用形式 `[=]` 捕获整个对象（请注意，通过 `[=]` 隐式捕获指针 `this` 在 C++20 中已被弃用）：

    ```cpp
    struct foo
    {
      int         id;
      std::string name;
      auto run()
     {
        return [=] { std::cout << id << ' ' << name << '\n'; };
      }
    }; 
    ```

+   通过捕获 `this` 指针捕获整个对象。如果需要调用类的其他方法，这是必要的。这可以捕获为 `[this]` 当指针通过值捕获时，或者 `[*this]` 当对象本身通过值捕获时。如果对象在捕获发生之后但在 lambda 调用之前可能超出作用域，这可能会产生重大差异：

    ```cpp
    struct foo
    {
      int         id;
      std::string name;
      auto run()
     {
        return[this]{ std::cout << id << ' ' << name << '\n'; };
      }
    };
    auto l = foo{ 42, "john" }.run();
    l(); // does not print 42 john 
    ```

在此情况下，正确的捕获应该是 `[*this]`，以便对象通过值复制。在这种情况下，调用 lambda 将打印 *42 john*，即使临时变量已经超出作用域。

C++20 标准引入了对捕获指针 `this` 的几个更改：

+   当使用 `[=]` 时，它会弃用隐式捕获 `this`。这将导致编译器发出弃用警告。

+   当你想要使用 `[=, this]` 显式捕获所有内容时，它引入了通过值捕获 `this` 指针。你仍然只能使用 `[this]` 捕获指针 `this`。

有一些情况下，lambda 表达式仅在它们的参数方面有所不同。在这种情况下，lambda 可以以泛型方式编写，就像模板一样，但使用 `auto` 指定类型参数（不涉及模板语法）。这将在下一道菜谱中解决，正如即将到来的 *参见* 部分所注明的。

在 C++23 之前，属性可以指定在可选的异常指定符和可选的尾随返回类型之间的 lambda 表达式中。这些属性将应用于类型，而不是函数调用操作符。然而，如 `[[nodiscard]]` 或 `[[noreturn]]` 这样的属性仅在函数上才有意义，而不是类型。

因此，从 C++23 开始，这个限制已经改变，属性也可以被指定：

+   在 lambda 引入符及其可选捕获之后，或者

+   在模板参数列表及其可选的 requires 子句之后。

在 lambda 声明中的任何这些部分声明的属性应用于函数调用操作符，而不是类型。

让我们考察以下示例：

```cpp
auto linc = [](int a) [[deprecated]] { return a+1; };
linc(42); 
```

`[[deprecated]]` 属性应用于 lambda 的类型，在编译代码片段时不会产生警告。在 C++23 中，我们可以写出以下代码：

```cpp
auto linc = [][[nodiscard,deprecated]](int a) { return a+1; };
linc(42); 
```

通过这个变化，`[[nodiscard]]` 和 `[[deprecated]]` 属性都应用于 lambda 类型的函数调用操作符。这导致发出两个警告：一个是指示正在使用弃用的函数，另一个是指示返回类型被忽略。

## 参见

+   *使用泛型和模板 lambda*，了解如何为 lambda 参数使用 `auto` 并如何在 C++20 中定义模板 lambda

+   *编写递归 lambda*，了解我们可以用来使 lambda 递归调用的技术

+   *第四章*，*使用属性向编译器提供元数据*，了解可用的标准属性以及如何使用它们

# 使用泛型和模板 lambda

在前面的食谱中，我们看到了如何编写 lambda 表达式以及如何与标准算法一起使用它们。在 C++ 中，lambda 表达式基本上是无名函数对象的语法糖，这些对象是实现了 call 操作符的类。然而，就像任何其他函数一样，这可以通过模板进行泛型实现。C++14 利用这一点并引入了不需要为它们的参数指定实际类型的泛型 lambda，而是使用 `auto` 指示符。尽管没有使用这个名字，但泛型 lambda 实际上就是 lambda 模板。当我们需要使用相同的 lambda 但具有不同类型的参数时，它们非常有用。此外，C++20 标准更进一步，支持显式定义模板 lambda。这有助于一些泛型 lambda 令人繁琐的场景。

## 入门

建议你在继续阅读本食谱之前，先阅读前面的食谱，*使用 lambda 与标准算法*，以便熟悉 C++ 中 lambda 的基础知识。

## 如何做到这一点...

自 C++14 以来，我们可以编写泛型 lambda：

+   通过使用 `auto` 指示符而不是实际类型作为 lambda 表达式参数

+   当我们需要使用多个 lambda 表达式，而这些 lambda 表达式仅通过它们的参数类型不同时

以下示例展示了如何使用 `std::accumulate()` 算法使用泛型 lambda，首先使用整数向量，然后使用字符串向量：

```cpp
auto numbers =
  std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};
using namespace std::string_literals;
auto texts =
  std::vector<std::string>{"hello"s, " "s, "world"s, "!"s};
auto lsum = [](auto const s, auto const n) {return s + n;};
auto sum = std::accumulate(
  std::begin(numbers), std::end(numbers), 0, lsum);
  // sum = 22
auto text = std::accumulate(
  std::begin(texts), std::end(texts), ""s, lsum);
  // sum = "hello world!"s 
```

自 C++20 以来，我们可以编写模板 lambda：

+   通过在捕获子句之后使用尖括号中的模板参数列表（例如 `<template T>`）

+   当你想要：

    +   仅对某些类型（如容器或满足概念的类型）限制泛型 lambda 的使用。

    +   确保泛型 lambda 的两个或多个参数实际上具有相同的类型。

    +   获取泛型参数的类型，例如，我们可以创建其实例，调用静态方法或使用其迭代器类型。

    +   在泛型 lambda 中执行完美转发。

以下示例展示了一个只能使用 `std::vector` 调用的模板 lambda：

```cpp
std::vector<int> vi { 1, 1, 2, 3, 5, 8 };
auto tl = []<typename T>(std::vector<T> const& vec)
{
   std::cout << std::size(vec) << '\n';
};
tl(vi); // OK, prints 6
tl(42); // error 
```

## 它是如何工作的...

在上一节的第一例中，我们定义了一个命名的 lambda 表达式——即，将它的闭包分配给变量的 lambda 表达式。然后，这个变量被传递给 `std::accumulate()` 函数作为参数。

这个通用算法接受开始和结束迭代器，这些迭代器定义了一个范围，一个要累加的初始值，以及一个函数，该函数将范围中的每个值累加到总和中。这个函数接受一个表示当前累加值的第一个参数和一个表示要累加到总和中当前值的第二个参数，并返回新的累加值。请注意，我没有使用术语 `add`，因为这不仅可以用于加法，还可以用于计算乘积、连接或其他聚合值的操作。

这个示例中的两次`std::accumulate()`调用几乎相同；只是参数的类型不同：

+   在第一次调用中，我们传递了整数范围（来自`vector<int>`）的迭代器、0 作为初始和，以及一个将两个整数相加并返回它们的和的 lambda。这会产生范围内所有整数的和；对于这个示例，它是`22`。

+   在第二次调用中，我们传递了字符串范围（来自`vector<string>`）的迭代器、一个空字符串作为初始值，以及一个通过将两个字符串相加并返回结果来连接两个字符串的 lambda。这会产生一个包含范围内所有字符串的字符串，一个接一个地放在一起；对于这个示例，结果是`hello world!`。

虽然泛型 lambda 可以在它们被调用的地方匿名定义，但这实际上并没有什么意义，因为泛型 lambda（基本上，如我们之前提到的，是一个 lambda 表达式模板）的主要目的就是为了重用，正如在*如何做...*部分中的示例所示。

在定义这个 lambda 表达式时，当与多个`std::accumulate()`调用一起使用时，我们不是为 lambda 参数指定具体类型（如`int`或`std::string`），而是使用了`auto`指定符，让编译器推断类型。

当遇到一个参数类型具有`auto`指定符的 lambda 表达式时，编译器会生成一个具有调用操作符模板的无名函数对象。对于这个示例中的泛型 lambda 表达式，函数对象看起来是这样的：

```cpp
struct __lambda_name__
{
  template<typename T1, typename T2>
 auto operator()(T1 const s, T2 const n) const { return s + n; }
  __lambda_name__(const __lambda_name__&) = default;
  __lambda_name__(__lambda_name__&&) = default;
  __lambda_name__& operator=(const __lambda_name__&) = delete;
  ~__lambda_name__() = default;
}; 
```

调用操作符是一个模板，它为 lambda 中每个使用`auto`指定的参数有一个类型参数。调用操作符的返回类型也是`auto`，这意味着编译器将从返回值的类型中推断它。这个操作符模板将使用编译器在泛型 lambda 使用的上下文中识别的实际类型进行实例化。

C++20 的模板 lambda 是对 C++14 泛型 lambda 的改进，使得某些场景更容易实现。一个典型的例子是上一节中的第二个示例，其中 lambda 的使用被限制为`std::vector`类型的参数。另一个例子是当你想要确保 lambda 的两个参数具有相同的类型。在 C++20 之前，这很难做到，但有了模板 lambda，这非常简单，如下面的示例所示：

```cpp
auto tl = []<typename T>(T x, T y)
{
  std::cout << x << ' ' << y << '\n';
};
tl(10, 20);   // OK
tl(10, "20"); // error 
```

模板 lambda 的另一个场景是当你需要知道参数的类型，以便你可以创建该类型的实例或调用它的静态成员时。使用泛型 lambda，解决方案如下：

```cpp
struct foo
{
   static void f() { std::cout << "foo\n"; }
};
auto tl = [](auto x)
{
  using T = std::decay_t<decltype(x)>;
  T other;
  T::f();
};
tl(foo{}); 
```

这个解决方案需要使用`std::decay_t`和`decltype`。`decltype`是一个类型指定符，它返回指定表达式的类型，主要用于编写模板。另一方面，`std::decay`是来自`<type_traits>`的一个实用工具，它执行与通过值传递函数参数相同的类型转换。

然而，在 C++20 中，相同的 lambda 可以这样编写：

```cpp
auto tl = []<typename T>(T x)
{
  T other;
  T::f();
}; 
```

当我们需要在泛型 lambda 中进行完美转发时，也会出现类似的情况，这需要使用 `decltype` 来确定参数的类型：

```cpp
template <typename ...T>
void foo(T&& ... args)
{ /* ... */ }
auto tl = [](auto&& ...args)
{
  return foo(std::forward<decltype(args)>(args)...);
};
tl(1, 42.99, "lambda"); 
```

使用模板 lambda，我们可以以更简单的方式重写如下：

```cpp
auto tl = []<typename ...T>(T && ...args)
{
  return foo(std::forward<T>(args)...);
}; 
```

如这些示例所示，模板 lambda 是对泛型 lambda 的改进，使得处理本食谱中提到的场景更加容易。

## 参见

+   *使用 lambda 与标准算法*，以探索 lambda 表达式的基础知识以及如何利用它们与标准算法。

+   *第一章*，*尽可能使用 auto*，以了解 C++ 中自动类型推导的工作原理

# 编写递归 lambda

Lambda 本质上是无名的函数对象，这意味着应该可以递归地调用它们。确实，它们可以递归地调用；然而，执行此操作的机制并不明显，因为它需要将 lambda 分配给函数包装器并通过引用捕获包装器。尽管可以争论递归 lambda 并没有真正意义，并且函数可能是一个更好的设计选择，但在本食谱中，我们将探讨如何编写递归 lambda。

## 准备工作

为了演示如何编写递归 lambda，我们将考虑斐波那契函数的著名示例。这通常在 C++ 中递归实现，如下所示：

```cpp
constexpr int fib(int const n)
{
  return n <= 2 ? 1 : fib(n - 1) + fib(n - 2);
} 
```

以此实现作为起点，让我们看看我们如何使用递归 lambda 重写它。

## 如何做到这一点...

在 C++11 中，为了编写递归 lambda 函数，您必须执行以下操作：

+   在函数作用域中定义 lambda。

+   将 lambda 分配给 `std::function` 包装器。

+   在 lambda 中通过引用捕获 `std::function` 对象，以便递归地调用它。

在 C++14 中，可以使用泛型 lambda 简化上述模式：

+   在函数作用域中定义 lambda。

+   使用 `auto` 占位符声明第一个参数；这用于将 lambda 表达式作为参数传递给自己。

+   通过传递 lambda 本身作为第一个参数来调用 lambda 表达式。

在 C++23 中，此模式可以进一步简化如下：

+   在函数作用域中定义 lambda。

+   声明第一个参数 `this const auto&& self`; 这是为了启用一个新的 C++23 特性，称为 *推导 this* 或 *显式对象参数*。您可以通过 `self` 参数递归调用 lambda 表达式。

+   通过调用它并传递显式参数（如果有）来调用 lambda 表达式，并让编译器推导第一个参数。

以下是一些递归 lambda 的示例：

+   在从定义它的作用域调用的函数的作用域中返回的递归 Fibonacci lambda 表达式：

    ```cpp
    void sample()
    {
      std::function<int(int const)> lfib =
        &lfib
        {
          return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2);
        };
      auto f10 = lfib(10);
    } 
    ```

+   由函数返回的递归 Fibonacci lambda 表达式，可以从任何作用域调用：

    ```cpp
    std::function<int(int const)> fib_create()
    {
      std::function<int(int const)> f = [](int const n)
      {
        std::function<int(int const)> lfib = &lfib
        {
          return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2);
        };
        return lfib(n);
      };
      return f;
    }
    void sample()
    {
      auto lfib = fib_create();
      auto f10 = lfib(10);
    } 
    ```

+   作为类成员的 lambda 表达式，该类被递归调用：

    ```cpp
    struct fibonacci
    {
      std::function<int(int const)> lfib =
        this
        {
          return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2);
        };
    };
    fibonacci f;
    f.lfib(10); 
    ```

+   一个递归的 Fibonacci 泛型 lambda 表达式——C++14 对第一个要点中例子的替代方案：

    ```cpp
    void sample()
    {
       auto lfib = [](auto f, int const n)
       {
          if (n < 2) return 1;
          else return f(f, n - 1) + f(f, n - 2);
       };
       lfib(lfib, 10);
    } 
    ```

+   一个递归的 Fibonacci lambda 表达式，利用了 C++23 中的显式对象参数（或推导此）功能，这是上述方法的进一步简化替代方案：

    ```cpp
    void sample()
    {
      auto lfib = [](this const auto& self, int n) -> int
      {
        return n <= 2 ? 1 : self(n - 1) + self(n - 2);
      };
      lfib(5);
    } 
    ```

## 它是如何工作的...

当在 C++11 中编写递归 lambda 时，你需要考虑的第一件事是 lambda 表达式是一个函数对象，并且为了从 lambda 的主体中递归调用它，lambda 必须捕获其闭包（即 lambda 的实例化）。换句话说，lambda 必须捕获自身，这有几个含义：

+   首先，lambda 必须有一个名称；无名的 lambda 不能被捕获以便再次调用。

+   其次，lambda 只能在函数作用域内定义。这是因为 lambda 只能捕获函数作用域中的变量；它不能捕获任何具有静态存储期的变量。在命名空间作用域中定义的对象或具有静态或外部指定符的对象具有静态存储期。如果 lambda 在命名空间作用域中定义，其闭包将具有静态存储期，因此 lambda 不会捕获它。

+   第三个含义是 lambda 闭包的类型不能保持未指定；也就是说，不能使用`auto`指定符声明。使用`auto`类型指定符声明的变量不能出现在其自己的初始化器中。这是因为当处理初始化器时，变量的类型是未知的。因此，你必须指定 lambda 闭包的类型。我们可以通过使用通用函数包装器`std::function`来实现这一点。

+   最后但同样重要的是，lambda 闭包必须通过引用捕获。如果我们通过复制（或值）捕获，那么将创建函数包装器的副本，但在捕获时包装器未初始化。我们最终得到一个无法调用的对象。即使编译器不会对通过值捕获提出抱怨，当闭包被调用时，会抛出`std::bad_function_call`。

在“如何做……”部分的第一个例子中，递归 lambda 表达式定义在另一个名为`sample()`的函数内部。lambda 表达式的签名和主体与在介绍部分定义的常规递归函数`fib()`的签名和主体相同。lambda 闭包被分配给一个名为`lfib`的函数包装器，然后 lambda 通过引用捕获它，并从其主体中递归调用。由于闭包是通过引用捕获的，因此它将在 lambda 主体需要调用时初始化。

在第二个例子中，我们定义了一个返回 lambda 表达式闭包的函数，该 lambda 表达式反过来定义并调用一个递归 lambda，该递归 lambda 使用它被依次调用的参数。这是一个在需要从函数返回递归 lambda 时必须实现的模式。这是必要的，因为 lambda 闭包必须在递归 lambda 被调用时仍然可用。`fib_create()`方法返回一个函数包装器，当被调用时，创建一个捕获自身的递归 lambda。外部的`f` lambda 没有捕获任何东西，特别是通过引用；因此，我们不会遇到悬垂引用的问题。然而，当被调用时，它创建了一个嵌套 lambda 的闭包，这是我们真正想要调用的 lambda，并返回将递归`lfib` lambda 应用于其参数的结果。

在 C++14 中编写递归 lambda 更简单，如*如何做…*部分的第四个例子所示。不是捕获 lambda 的闭包，而是将其作为参数传递（通常是第一个）。为此，使用`auto`占位符声明了一个参数。让我们回顾一下实现，以便讨论它：

```cpp
auto lfib = [](auto f, int const n)
{
   if (n < 2) return 1;
   else return f(f, n - 1) + f(f, n - 2);
};
lfib(lfib, 10); 
```

lambda 表达式是一个具有函数调用操作符的函数对象。一个泛型 lambda 是一个具有模板函数调用操作符的函数对象。编译器为前面的代码片段生成类似于以下代码的代码：

```cpp
class __lambda_name_3
{
public:
   template<class T1>
 inline int operator()(T1 f, const int n) const
 {
      if (n < 2) {
         return 1;
      }
      else {
         return f(f, n - 1) + f(f, n - 2);
      }
   }
   template<>
   inline int operator()<__lambda_name_3> (__lambda_name_3 f, 
 const int n) const
 {
      if (n < 2) {
         return 1;
      }
      else {
         return f.operator()(__lambda_name_3(f), n - 1) + 
                f.operator()(__lambda_name_3(f), n - 2);
      }
   }
};
__lambda_name_3 lfib = __lambda_name_3{};
lfib.operator()(__lambda_name_3(lfib), 10); 
```

函数调用操作符是一个模板函数。它的第一个参数具有类型模板参数的类型。对于这个主要模板，提供了对类类型的完整显式特化。这使得可以调用 lambda，将自身作为参数传递，从而避免捕获`std::function`对象，这在 C++11 中是必须做的。

如果你的编译器支持 C++23，那么在*显式对象参数*功能（也称为*推导 this*）的帮助下，可以进一步简化这一点。这个功能是为了使编译器能够从函数内部确定它被调用的表达式是一个左值还是右值，或者它是否是*cv-*或*ref-*限定，以及表达式的类型。这个功能使得以下场景成为可能：

+   通过基于重载的*cv-*和*ref-*限定符（例如，没有限定符和具有`const`限定符的相同函数，这是最常见的情况）避免代码重复。

+   通过使用简单的继承来简化**奇特重复模板模式**（**CRTP**），从而从模式中去除重复。

+   简化编写递归 lambda。

对于*如何做…*部分给出的例子，编译器能够推断出第一个参数`self`的类型，这使得不需要显式传递 lambda 闭包作为参数。

注意，在 C++23 的例子中，我们使用尾随返回类型语法定义了一个 lambda 表达式：

```cpp
[](this auto const & self, int n) -> int 
```

没有这个，你会得到如下编译器错误：

```cpp
error: function 'operator()<(lambda)>' with deduced return type cannot be used before it is defined 
```

通过对函数实现进行微小更改，如以下所示，不再需要尾随返回类型，并且推导这个特性再次工作：

```cpp
auto lfib = [](this auto const& self, int n)
{
   if (n <= 2) return 1;
   return self(n - 1) + self(n - 2);
}; 
```

## 参见

+   *使用通用和模板 lambda*，学习如何在 C++20 中使用`auto`作为 lambda 参数以及如何定义模板 lambda

+   *第九章*，*使用怪异重复模板模式进行静态多态*，了解 CRTP 是什么以及它是如何工作的

# 编写函数模板

通用代码是避免编写重复代码的关键。在 C++中，这是通过模板实现的。类、函数和变量都可以进行模板化。尽管模板通常被视为复杂且繁琐，但它们能够创建通用库，例如标准库，并帮助我们编写更少且更好的代码。

模板是 C++语言的一等公民，可能需要整本书来详细说明。实际上，这本书中的多个菜谱都处理了模板的各个方面。在本菜谱中，我们将讨论编写函数模板的基础。

## 如何做到这一点...

要创建函数模板，请执行以下操作：

+   要创建一个函数模板，在函数声明前加上`template`关键字，后跟尖括号中的模板参数列表：

    ```cpp
    template <typename T>
    T minimum(T a, T b)
    {
       return a <= b ? a : b;
    }
    minimum(3, 4);
    minimum(3.99, 4.01); 
    ```

+   要专门化一个函数模板，在函数签名中留空模板参数列表，并用实际类型或值替换模板参数：

    ```cpp
    template <>
    const char* minimum(const char* a, const char* b)
    {
       return std::strcmp(a, b) <= 1 ? a : b;
    }
    minimum("abc", "acxyz"); 
    ```

+   要重载函数模板，提供另一个定义，这可以是模板或非模板：

    ```cpp
    template <typename T>
    std::basic_string<T> minimum(std::basic_string<T> a, 
                                 std::basic_string<T> b) // [1]
    {
       return a.length() <= b.length() ? a : b;
    }
    std::string minimum(std::string a, std::string b) // [2]
    {
       return a.length() <= b.length() ? a : b;
    }
    minimum(std::string("def"), std::string("acxyz")); // calls [2]
    minimum(std::wstring(L"def"), std::wstring(L"acxyz")); // calls [1] 
    ```

+   要确保特定的函数模板或函数模板的专门化不能被调用（从重载集中删除），请将其声明为`deleted`：

    ```cpp
    template <typename T>
    T* minimum(T* a, T* b) = delete;
    int a = 3;
    int b = 4;
    minimum(&a, &b); // error 
    ```

## 它是如何工作的...

至少乍一看，函数模板与其他函数只有细微的差别。它们使用模板语法引入，可以用类型、值甚至其他模板进行参数化。然而，由于模板只是创建实际代码的蓝图，函数模板基本上是一个定义函数族的蓝图。模板仅在源代码中存在，直到它们被使用。

编译器根据其使用情况实例化模板。这个过程称为*模板实例化*。编译器通过替换模板参数来完成此操作。例如，在前面展示的`minimum<T>`函数模板的情况下，当我们以`minimum<int>(1, 2)`的方式调用它时，编译器将`int`类型替换为`T`参数。存在两种实例化的形式：

+   **隐式实例化**发生在编译器根据代码中使用的模板生成代码时。例如，如果您的代码中通过 `int` 和 `double` 值调用 `minimum<T>` 函数，那么将生成两个重载（一个带有整数参数，另一个带有 `double` 参数）。这被称为隐式实例化，如下面的代码片段所示：

    ```cpp
    minimum<int>(1, 2);  // explicit int template argument
    minimum(3.99, 4.50); // deduced double template argument 
    ```

+   **显式实例化**发生在您作为用户请求编译器从模板生成代码，即使该实例化在代码中没有使用时。这种用法的一个例子是在创建库（二进制）文件时，因为未实例化的模板（它们只是蓝图）不会被放入对象文件中。以下是一个为 `char` 类型的 `minimum<T>` 函数显式实例化的示例。请注意，如果显式实例化没有在模板所在的同一命名空间中定义，则必须在显式实例化定义中使用完全限定的名称：

    ```cpp
    template char minimum(char a, char b); 
    ```

如前所述，模板可以有不同的参数类型。这些参数位于 `template` 关键字之后的角度括号中，可以是以下类型：

+   **类型模板参数**，其中参数是类型的占位符。这是前一个章节中看到的所有示例的情况。

+   **非类型模板参数**，其中参数是结构化类型的值。整数类型、浮点类型（自 C++20 起）、指针类型、枚举类型和左值引用类型都是结构化类型。在下面的示例中，`T` 是一个类型模板参数，而 `S` 是一个非类型模板参数：

    ```cpp
    template <typename T, std::size_t S>
    std::array<T, S> make_array()
    {
       return std::array<T, S>{};
    } 
    ```

在 C++17 中，可以使用 `auto` 关键字声明非类型模板参数：

```cpp
template <typename T, auto S>
std::array<T, S> make_array()
{
   return std::array<T, S>{};
} 
```

+   **模板模板参数**，其中参数的类型是另一个类型。在下面的示例中，`trimin` 函数模板有两个模板参数，一个类型模板参数 `T` 和一个模板模板参数 `M`：

    ```cpp
    template <typename T>
    struct Minimum
    {
       T operator()(T a, T b)
     {
          return a <= b ? a : b;
       }
    };
    template <typename T, template <typename> class M>
    T trimin(T a, T b, T c)
    {
       return M<T>{}(a, M<T>{}(b, c));
    }
    trimin<int, Minimum>(5, 2, 7); 
    ```

虽然模板允许我们为许多类型（或更一般地说，模板参数）编写一个实现，但为不同类型提供修改后的实现通常是有用的，或者可能是必要的。为某些模板参数提供替代实现的过程称为特化。正在特化的模板称为*主模板*。有两种可能的形式：

+   **部分特化**是指只为某些模板参数提供不同的实现。

+   **完全特化**是指为模板参数的整个集合提供不同的实现。

函数模板仅支持完全特化。部分特化仅适用于类模板。在*如何做到这一点…*部分提供了一个完全特化的例子，当时我们为`const char*`类型特化了`minimum<T>`函数模板。我们决定不是基于两个参数的字典顺序比较，而是根据它们的长度来决定哪个“更小”。请记住，这只是一个为了理解特化而给出的例子。

函数模板可以像任何其他函数一样重载。请注意，当有多个重载可用，包括模板和非模板时，编译器将优先选择非模板重载。前面已经提供了一个例子。让我们再次看看，只包含函数的声明：

```cpp
template <typename T>
std::basic_string<T> minimum(std::basic_string<T> a, std::basic_string<T> b);
std::string minimum(std::string a, std::string b);
minimum(std::string("def"), std::string("acxyz"));
minimum(std::wstring(L"def"), std::wstring(L"acxyz")); 
```

对`minimum`函数的第一个调用接受`std::string`参数，因此将调用非模板重载。第二个调用接受`std::wstring`参数，由于函数模板是唯一匹配的重载，因此将调用其`std::wstring`实例化。

在调用函数模板时指定模板参数并不总是必要的。以下两个调用是相同的：

```cpp
minimum(1, 2);
minimum<int>(1, 2); 
```

在许多情况下，编译器可以从函数的调用中推导出模板参数。在这个例子中，由于两个函数参数都是整数，它可以推断出模板参数应该是`int`类型。因此，明确指定这一点是不必要的。然而，也存在编译器无法推导类型的情况。在这些情况下，您必须明确提供它们。下面将给出一个例子：

```cpp
minimum(1, 2u); // error, ambiguous template parameter T 
```

两个参数是一个`int`和一个`unsigned int`。因此，编译器不知道`T`类型应该推断为`int`还是`unsigned int`。为了解决这种歧义，您必须明确提供模板参数：

```cpp
minimum<unsigned>(1, 2u); // OK 
```

在推导模板参数时，编译器会在模板参数和用于调用函数的参数之间进行比较。为了使比较成功并让编译器成功推导出所有参数，这些参数必须具有某种结构。然而，对这个过程的详细探讨超出了本菜谱的范围。您可以查阅其他资源，包括我的书籍《使用 C++的模板元编程》，其中在第四章详细讨论了这一点，包括函数模板和类模板。

如介绍中所述，模板是一个广泛的主题，无法在一道菜谱中涵盖。我们将在整本书中学习更多关于模板的内容，包括在接下来的两个菜谱中，我们将讨论具有可变数量参数的函数模板。

## 参见

+   *编写具有可变数量参数的函数模板*，以了解如何编写接受可变数量参数的函数

+   *第一章*，*使用类模板参数推导简化代码*，以了解模板参数推导对类模板的工作方式。

# 编写具有可变数量参数的函数模板

有时编写具有可变数量参数的函数或具有可变数量成员的类是有用的。典型的例子包括像`printf`这样的函数，它接受一个格式和可变数量的参数，或者像`tuple`这样的类。在 C++11 之前，前者只能通过使用可变宏（它只能编写不安全的函数）来实现，而后者根本不可能。C++11 引入了可变模板，这些是具有可变数量参数的模板，使得可以编写具有可变数量参数的类型安全函数模板，以及具有可变数量成员的类模板。在这个菜谱中，我们将探讨编写函数模板。

## 准备工作

具有可变数量参数的函数被称为*可变参数函数*。具有可变数量参数的函数模板被称为*可变参数函数模板*。了解 C++可变参数宏（`va_start`、`va_end`、`va_arg`、`va_copy`和`va_list`）对于学习如何编写可变参数函数模板不是必需的，但它是一个很好的起点。

我们已经在之前的菜谱中使用了可变模板，但这个将提供详细的解释。

## 如何做到这一点...

为了编写可变参数模板函数，你必须执行以下步骤：

1.  定义一个具有固定数量参数的重载，如果可变参数模板函数的语义需要结束编译时递归（参见以下代码中的[1]）。

1.  定义一个模板参数包，它是一个可以存储任意数量参数的模板参数，包括零；这些参数可以是类型、非类型或模板（参见[2]）。

1.  定义一个函数参数包以存储任意数量的函数参数，包括零；模板参数包的大小和相应的函数参数包的大小相同。这个大小可以用`sizeof...`运算符确定（参见[3]），并参考*如何工作...*部分的结尾以获取有关此运算符的信息）。

1.  扩展参数包以便用提供的实际参数替换它（参见[4]）。

以下示例，它说明了所有前面的点，是一个使用`operator+`添加可变数量参数的可变参数函数模板：

```cpp
template <typename T>                 // [1] overload with fixed
T add(T value) //     number of arguments
{
  return value;
}
template <typename T, typename... Ts> // [2] typename... Ts
T add(T head, Ts... rest) // [3] Ts... rest
{
  return head + add(rest...);         // [4] rest...
} 
```

## 如何工作...

乍一看，前面的实现看起来像是递归，因为 `add()` 函数调用了自身，从某种意义上说确实是，但它是一种编译时递归，不会产生任何运行时递归和开销。实际上，编译器根据变长函数模板的使用生成具有不同参数数量的几个函数，因此只涉及函数重载，而不是递归。然而，实现时似乎参数是以递归方式处理的，有一个结束条件。

在前面的代码中，我们可以识别出以下关键部分：

+   `Typename... Ts` 是一个模板参数包，表示可变数量的模板类型参数。

+   `Ts... rest` 是一个函数参数包，表示可变数量的函数参数。

+   `rest...` 是函数参数包的展开。

省略号的位置在语法上并不重要。`typename... Ts`、`typename ... Ts` 和 `typename ...Ts` 都是等效的。

在 `add(T head, Ts... rest)` 参数中，`head` 是参数列表中的第一个元素，而 `...rest` 是包含列表中其余参数的包（这可以是零个或多个）。在函数体中，`rest...` 是函数参数包的展开。这意味着编译器将参数包及其元素按顺序替换。在 `add()` 函数中，我们基本上将第一个参数添加到剩余参数的总和中，这给人一种递归处理的印象。这种递归在只剩下一个参数时结束，此时调用第一个 `add()` 重载（单个参数）并返回其参数的值。

这种函数模板 `add()` 的实现使我们能够编写如下所示的代码：

```cpp
auto s1 = add(1, 2, 3, 4, 5); // s1 = 15
auto s2 = add("hello"s, " "s, "world"s, "!"s); // s2 = "hello world!" 
```

当编译器遇到 `add(1, 2, 3, 4, 5)` 时，它会生成以下函数（注意 `arg1`、`arg2` 等不是编译器实际生成的名称），这表明这个过程实际上只是一系列对重载函数的调用，而不是递归：

```cpp
int add(int head, int arg1, int arg2, int arg3, int arg4)
{return head + add(arg1, arg2, arg3, arg4);}
int add(int head, int arg1, int arg2, int arg3)
{return head + add(arg1, arg2, arg3);}
int add(int head, int arg1, int arg2)
{return head + add(arg1, arg2);}
int add(int head, int arg1)
{return head + add(arg1);}
int add(int value)
{return value;} 
```

使用 GCC 和 Clang，你可以使用 `__PRETTY_FUNCTION__` 宏来打印函数的名称和签名。

通过添加 `std::cout << __PRETTY_FUNCTION__ << std::endl`，当使用 GCC 或 Clang 时，在两个我们编写的函数的开始处，运行代码时会得到以下结果：

+   使用 GCC：

    ```cpp
    T add(T, Ts ...) [with T = int; Ts = {int, int, int, int}]
    T add(T, Ts ...) [with T = int; Ts = {int, int, int}]
    T add(T, Ts ...) [with T = int; Ts = {int, int}]
    T add(T, Ts ...) [with T = int; Ts = {int}]
    T add(T) [with T = int] 
    ```

+   使用 Clang：

    ```cpp
    T add(T, Ts...) [T = int, Ts = <int, int, int, int>]
    T add(T, Ts...) [T = int, Ts = <int, int, int>]
    T add(T, Ts...) [T = int, Ts = <int, int>]
    T add(T, Ts...) [T = int, Ts = <int>]
    T add(T) [T = int] 
    ```

由于这是一个函数模板，它可以与支持 `operator+` 操作符的任何类型一起使用。另一个例子，`add("hello"s, " "s, "world"s, "!"s)` 产生 *hello world!* 字符串。然而，`std::basic_string` 类型对 `operator+` 有不同的重载，包括一个可以将字符串连接到字符的重载，因此我们应该能够编写以下代码：

```cpp
auto s3 = add("hello"s, ' ', "world"s, '!'); // s3 = "hello world!" 
```

然而，这将生成编译器错误，如下所示（注意我实际上用字符串 *hello world!* 替换了 `std::basic_string<char, std::char_traits<char>, std::allocator<char> >` 以简化问题）：

```cpp
In instantiation of 'T add(T, Ts ...) [with T = char; Ts = {string, char}]':
16:29:   required from 'T add(T, Ts ...) [with T = string; Ts = {char, string, char}]'
22:46:   required from here
16:29: error: cannot convert 'string' to 'char' in return
 In function 'T add(T, Ts ...) [with T = char; Ts = {string, char}]':
17:1: warning: control reaches end of non-void function [-Wreturn-type] 
```

发生的情况是编译器生成了这里显示的代码，其中返回类型与第一个参数的类型相同。然而，第一个参数要么是`std::string`要么是`char`（为了简单起见，`std::basic_string<char, std::char_traits<char>, std::allocator<char> >`被替换为`string`）。在第一个参数的类型是`char`的情况下，返回值`head+add` `(...)`的类型，它是一个`std::string`，与函数返回类型不匹配，并且没有到它的隐式转换：

```cpp
string add(string head, char arg1, string arg2, char arg3)
{return head + add(arg1, arg2, arg3);}
char add(char head, string arg1, char arg2)
{return head + add(arg1, arg2);}
string add(string head, char arg1)
{return head + add(arg1);}
char add(char value)
{return value;} 
```

我们可以通过修改变长函数模板，使其返回类型为`auto`而不是`T`来解决这个问题。在这种情况下，返回类型总是从返回表达式中推断出来的，在我们的例子中，在所有情况下都将是`std::string`：

```cpp
template <typename T, typename... Ts>
auto add(T head, Ts... rest)
{
  return head + add(rest...);
} 
```

应进一步说明参数包可以出现在花括号初始化中，并且可以使用`sizeof...`运算符确定其大小。此外，变长函数模板不一定意味着编译时递归，正如我们在本食谱中所展示的。所有这些都在以下示例中展示：

```cpp
template<typename... T>
auto make_even_tuple(T... a)
{
  static_assert(sizeof...(a) % 2 == 0,
                "expected an even number of arguments");
  std::tuple<T...> t { a... };
  return t;
}
auto t1 = make_even_tuple(1, 2, 3, 4); // OK
// error: expected an even number of arguments
auto t2 = make_even_tuple(1, 2, 3); 
sizeof...(a) to make sure that we have an even number of arguments and assert by generating a compiler error otherwise. The sizeof... operator can be used with both template parameter packs and function parameter packs. sizeof...(a) and sizeof...(T) would produce the same value. Then, we create and return a tuple. 
```

模板参数包`T`被展开（使用`T...`）为`std::tuple`类模板的类型参数，函数参数包`a`被展开（使用`a...`）为元组成员的值，使用花括号初始化。

## 参见

+   *使用折叠表达式简化变长函数模板*，了解如何在创建具有可变数量参数的函数模板时编写更简单、更清晰的代码

+   *第二章*，*创建原始用户定义字面量*，了解如何提供对输入序列的定制解释，以便改变编译器的正常行为

# 使用折叠表达式简化变长函数模板

在本章中，我们已经多次讨论了折叠；这是一个将二元函数应用于值范围以产生单个值的操作。我们在讨论变长函数模板时看到了这一点，我们还将再次在高阶函数中看到它。结果证明，在变长函数模板中参数包的展开基本上是一个折叠操作的情况有很多。为了简化编写这样的变长函数模板，C++17 引入了折叠表达式，它将参数包的展开折叠到二元运算符上。在本食谱中，我们将学习如何使用折叠表达式来简化编写变长函数模板。

## 准备工作

本食谱中的示例基于我们在上一食谱中编写的变长函数模板`add` `()`，即*编写具有可变数量参数的函数模板*。该实现是一个左折叠操作。为了简单起见，我们将再次展示该函数：

```cpp
template <typename T>
T add(T value)
{
  return value;
}
template <typename T, typename... Ts>
T add(T head, Ts... rest)
{
  return head + add(rest...);
} 
```

在下一节中，我们将学习如何简化这种特定的实现，以及其他使用折叠表达式的示例。

## 如何操作...

要在二元运算符上折叠参数包，可以使用以下形式之一：

+   使用一元形式 `(... op pack)` 的左折叠：

    ```cpp
    template <typename... Ts>
    auto add(Ts... args)
    {
      return (... + args);
    } 
    ```

+   使用二元形式 `(init op ... op pack)` 的左折叠：

    ```cpp
    template <typename... Ts>
    auto add_to_one(Ts... args)
    {
      return (1 + ... + args);
    } 
    ```

+   使用一元形式 `(pack op ...)` 的右折叠：

    ```cpp
    template <typename... Ts>
    auto add(Ts... args)
    {
      return (args + ...);
    } 
    ```

+   使用二元形式 `(pack op ... op init)` 的右折叠：

    ```cpp
    template <typename... Ts>
    auto add_to_one(Ts... args)
    {
      return (args + ... + 1);
    } 
    ```

    这里显示的括号是折叠表达式的一部分，不能省略。

## 它是如何工作的...

当编译器遇到折叠表达式时，它会将其展开为以下表达式之一：

| **表达式** | **展开** |
| --- | --- |
| `(... op pack)` | `((pack$1 op pack$2) op ...) op pack$n` |
| `(init op ... op pack)` | `(((init op pack$1) op pack$2) op ...) op pack$n` |
| `(pack op ...)` | `pack$1 op (... op (pack$n-1 op pack$n))` |
| `(pack op ... op init)` | `pack$1 op (... op (pack$n-1 op (pack$n op init)))` |

表 3.2：折叠表达式的可能形式

当使用二元形式时，省略号左右两边的运算符必须相同，并且初始化值不能包含未展开的参数包。

支持以下二元运算符与折叠表达式一起使用：

| `+` | `-` | `*` | `/` | `%` | `^` | `&` | `&#124;` | `=` | `<` | `>` | `<<` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `>>` | `+=` | `-=` | `*=` | `/=` | `%=` | `^=` | `&=` | `&#124;=` | `<<=` | `>>=` | `==` |
| `!=` | `<=` | `>=` | `&&` | `&#124;&#124;` | `,` | `.*` | `->*.` |  |  |  |  |

表 3.3：与折叠表达式一起支持的二元运算符

当使用一元形式时，只有 `*`、`+`、`&`、`|`、`&&`、`||` 和 `,`（逗号）这样的运算符可以与空参数包一起使用。在这种情况下，空包的值如下：

| **运算符** | **空包值** |
| --- | --- |
| `+` | `0` |
| `*` | `1` |
| `&` | `-1` |
| `&#124;` | `0` |
| `&&` | `true` |
| `&#124;&#124;` | `false` |
| `,` | `void()` |

表 3.4：可以使用空参数包的运算符

现在我们有了之前实现的功能模板（让我们考虑左折叠版本），我们可以编写以下代码：

```cpp
auto sum = add(1, 2, 3, 4, 5);         // sum = 15
auto sum1 = add_to_one(1, 2, 3, 4, 5); // sum = 16 
```

考虑 `add(1, 2, 3, 4, 5)` 调用，它将产生以下函数：

```cpp
int add(int arg1, int arg2, int arg3, int arg4, int arg5)
{
  return ((((arg1 + arg2) + arg3) + arg4) + arg5);
} 
```

值得注意的是，由于现代编译器在优化方面的积极方式，这个函数可以被内联，最终我们可能会得到一个如 `auto sum = 1 + 2 + 3 + 4 + 5` 的表达式。

## 还有更多...

折叠表达式与支持的所有二元运算符的重载一起工作，但不与任意二元函数一起工作。可以通过提供一个将包含值和该包装器类型的重载运算符的包装器类型来实现一个解决方案：

```cpp
template <typename T>
struct wrapper
{
  T const & value;
};
template <typename T>
constexpr auto operator<(wrapper<T> const & lhs, wrapper<T> const & rhs)
{
  return wrapper<T> {lhs.value < rhs.value ? lhs.value : rhs.value};
} 
```

在前面的代码中，`wrapper` 是一个简单的类模板，它持有类型 `T` 的值的常量引用。为此类模板提供了一个重载的 `operator<`；这个重载不返回布尔值来指示第一个参数小于第二个参数，而是实际上返回一个 `wrapper` 类型的实例来保存两个参数中的最小值。这里显示的变长函数模板 `min()` 使用这个重载的 `operator<` 来折叠展开为 `wrapper` 类模板实例的参数包：

```cpp
template <typename... Ts>
constexpr auto min(Ts&&... args)
{
  return (wrapper<Ts>{args} < ...).value;
}
auto m = min(3, 1, 2); // m = 1 
```

此 `min()` 函数被编译器扩展为类似以下内容：

```cpp
template<>
inline constexpr int min<int, int, int>(int && __args0,
                                        int && __args1,
                                        int && __args2)
{
  return
operator<(wrapper_min<int>{__args0},
      operator<(wrapper_min<int>{__args1},
                wrapper_min<int>{__args2})).value;
} 
```

我们在这里可以看到的是对二进制 `operator <` 的级联调用，返回 `Wrapper<int>` 值。没有这个，使用折叠表达式实现的 `min()` 函数的实现将是不可能的。以下实现不起作用：

```cpp
template <typename... Ts>
constexpr auto minimum(Ts&&... args)
{
  return (args < ...);
} 
```

根据调用 `min(3, 1, 2)`，编译器将将其转换为以下类似的内容：

```cpp
template<>
inline constexpr bool minimum<int, int, int>(int && __args0,
                                             int && __args1,
                                             int && __args2)
{
  return __args0 < (static_cast<int>(__args1 < __args2));
} 
```

结果是一个返回布尔值的函数，而不是实际整数值，这是提供的参数之间的最小值。

## 参见

+   *实现高阶函数 map 和 fold*，了解函数式编程中的高阶函数以及如何实现广泛使用的 `map` 和 `fold`（或 `reduce`）函数

# 实现高阶函数 map 和 fold

在本书前面的食谱中，我们在几个示例中使用了通用算法 `std::transform()` 和 `std::accumulate()`，例如用于实现字符串实用工具以创建字符串的大写或小写副本，或用于计算范围值的总和。

这些基本上是高阶函数 `map` 和 `fold` 的实现。高阶函数是一种接受一个或多个其他函数作为参数并将它们应用于范围（列表、向量、映射、树等）的函数，从而产生一个新的范围或一个值。在本食谱中，我们将学习如何实现 `map` 和 `fold` 函数，以便它们能够与 C++ 标准容器一起工作。

## 准备工作

`map` 是一种高阶函数，它将函数应用于范围中的元素并返回一个新范围，顺序相同。

`fold` 是一种高阶函数，它将组合函数应用于范围中的元素以产生单个结果。由于处理顺序可能很重要，通常有两个版本的此函数。一个是 `fold_left`，它从左到右处理元素，而另一个是 `fold_right`，它从右到左组合元素。

大多数关于函数 map 的描述表明它应用于列表，但这是一个通用术语，可以指代不同的顺序类型，如列表、向量、数组，以及字典（即映射）、队列等。因此，我更喜欢在描述这些高阶函数时使用术语范围。

例如，映射操作可以将字符串范围转换为表示每个字符串长度的整数范围。然后，折叠操作可以将这些长度相加，以确定所有字符串的总长度。

## 如何做到这一点...

要实现 `map` 函数，你应该：

+   在支持迭代和元素赋值的容器上使用 `std::transform`，例如 `std::vector` 或 `std::list`：

    ```cpp
    template <typename F, typename R>
    R mapf(F&& func, R range)
    {
      std::transform(
        std::begin(range), std::end(range), std::begin(range),
        std::forward<F>(func));
      return range;
    } 
    ```

+   对于不支持对元素进行赋值的容器，例如 `std::map` 和 `std::queue`，使用其他方法，如显式迭代和插入：

    ```cpp
    template<typename F, typename T, typename U>
    std::map<T, U> mapf(F&& func, std::map<T, U> const & m)
    {
      std::map<T, U> r;
      for (auto const kvp : m)
        r.insert(func(kvp));
      return r;
    }
    template<typename F, typename T>
    std::queue<T> mapf(F&& func, std::queue<T> q)
    {
      std::queue<T> r;
      while (!q.empty())
      {
        r.push(func(q.front()));
        q.pop();
      }
      return r;
    } 
    ```

要实现 `fold` 函数，你应该：

+   在支持迭代的容器上使用 `std::accumulate()`：

    ```cpp
    template <typename F, typename R, typename T>
    constexpr T fold_left(F&& func, R&& range, T init)
    {
      return std::accumulate(
        std::begin(range), std::end(range),
        std::move(init),
        std::forward<F>(func));
    }
    template <typename F, typename R, typename T>
    constexpr T fold_right(F&& func, R&& range, T init)
    {
      return std::accumulate(
        std::rbegin(range), std::rend(range),
        std::move(init),
        std::forward<F>(func));
    } 
    ```

+   对于不支持迭代的容器，例如 `std::queue`，使用其他方法显式处理：

    ```cpp
    template <typename F, typename T>
    constexpr T fold_left(F&& func, std::queue<T> q, T init)
    {
      while (!q.empty())
      {
        init = func(init, q.front());
        q.pop();
      }
      return init;
    } 
    ```

## 它是如何工作的...

在前面的示例中，我们以函数式的方式实现了 `map` 高阶函数，没有副作用。这意味着它保留了原始范围并返回一个新的范围。函数的参数是应用函数和范围。为了避免与 `std::map` 容器混淆，我们称此函数为 `mapf`。`mapf` 有几个重载，如前所述：

+   第一个重载适用于支持迭代和对其元素进行赋值的容器；这包括 `std::vector`、`std::list`、`std::array`，以及 C 类型的数组。函数接受一个函数的右值引用和一个定义了 `std::begin()` 和 `std::end()` 的范围。范围按值传递，以便修改局部副本不会影响原始范围。范围通过使用标准算法 `std::transform()` 对每个元素应用给定函数进行转换；然后返回转换后的范围。

+   第二个重载专门针对 `std::map`，它不支持直接对其元素进行赋值（`std::pair<T, U>`）。因此，此重载创建一个新的映射，然后使用基于范围的 `for` 循环遍历其元素，并将将输入函数应用于原始映射的每个元素的结果插入到新映射中。

+   第三个重载专门针对 `std::queue`，它是一个不支持迭代的容器。可以争辩说队列不是映射的典型结构，但为了演示不同的可能实现，我们正在考虑它。为了遍历队列的元素，队列必须被修改——你需要从前面弹出元素直到列表为空。这就是第三个重载所做的事情——它处理输入队列（按值传递）的每个元素，并将给定函数应用于剩余队列的前端元素的结果推送到队列的前端。

现在我们已经实现了这些重载，我们可以将它们应用到许多容器中，如下面的示例所示：

+   保留向量中的绝对值。在这个例子中，向量包含正负值。应用映射后，结果是只包含正值的新的向量：

    ```cpp
    auto vnums =
      std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};
    auto r =mapf([](int const i) { return std::abs(i); }, vnums);
    // r = {0, 2, 3, 5, 1, 6, 8, 4, 9} 
    ```

+   平方列表中的数值。在这个例子中，列表包含整数。应用映射后，结果是包含初始值平方的列表：

    ```cpp
    auto lnums = std::list<int>{1, 2, 3, 4, 5};
    auto l = mapf([](int const i) { return i*i; }, lnums);
    // l = {1, 4, 9, 16, 25} 
    ```

+   浮点数的四舍五入。对于这个例子，我们需要使用`std::round()`；然而，它对所有浮点类型都有重载，这使得编译器无法选择正确的类型。因此，我们要么编写一个 lambda，该 lambda 接受特定浮点类型的参数并返回应用于该值的`std::round()`的值，要么创建一个函数对象模板，该模板包装`std::round()`并仅允许浮点类型调用其调用操作符。这种技术在下例中使用：

    ```cpp
    template<class T = double>
    struct fround
    {
      typename std::enable_if_t<std::is_floating_point_v<T>, T>
      operator()(const T& value) const
      {
        return std::round(value);
      }
    };
    auto amounts =
      std::array<double, 5> {10.42, 2.50, 100.0, 23.75, 12.99};
    auto a = mapf(fround<>(), amounts);
    // a = {10.0, 3.0, 100.0, 24.0, 13.0} 
    ```

+   将单词映射的字符串键转换为大写（其中键是单词，值是文本中的出现次数）。请注意，创建字符串的大写副本本身就是一个映射操作。因此，在这个例子中，我们使用`mapf`将`toupper()`应用于表示键的字符串元素，以生成大写副本：

    ```cpp
    auto words = std::map<std::string, int>{
      {"one", 1}, {"two", 2}, {"three", 3}
    };
    auto m = mapf(
      [](std::pair<std::string, int> const kvp) {
        return std::make_pair(
          funclib::mapf(toupper, kvp.first),
          kvp.second);
        },
        words);
    // m = {{"ONE", 1}, {"TWO", 2}, {"THREE", 3}} 
    ```

+   标准化优先级队列中的值；最初，这些值从 1 到 100，但我们希望将它们标准化为两个值，1=高，2=正常。所有初始优先级值在 30 及以下的都获得高优先级；其余的获得正常优先级：

    ```cpp
    auto priorities = std::queue<int>();
    priorities.push(10);
    priorities.push(20);
    priorities.push(30);
    priorities.push(40);
    priorities.push(50);
    auto p = mapf(
      [](int const i) { return i > 30 ? 2 : 1; },
      priorities);
    // p = {1, 1, 1, 2, 2} 
    ```

要实现`fold`，我们实际上必须考虑两种可能的折叠方式——即从左到右和从右到左。因此，我们提供了两个函数，称为`fold_left`（用于左折叠）和`fold_right`（用于右折叠）。前一个章节中展示的实现非常相似：它们都接受一个函数、一个范围和一个初始值，并调用`std::accumulate()`将范围的值折叠成一个单一值。然而，`fold_left`使用直接迭代器，而`fold_right`使用反向迭代器遍历和处理范围。第二个重载是一个针对类型`std::queue`的特殊化，因为`std::queue`没有迭代器。

基于这些折叠实现，我们可以实现以下示例：

+   添加整数向量的值。在这种情况下，左折叠和右折叠将产生相同的结果。在以下示例中，我们传递一个 lambda，该 lambda 接受一个总和和一个数字，并返回一个新的总和，或者传递标准库中的函数对象`std::plus<>`，该对象将`operator+`应用于相同类型的两个操作数（基本上类似于 lambda 的闭包）：

    ```cpp
    auto vnums =
      std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};
    auto s1 = fold_left(
      [](const int s, const int n) {return s + n; },
      vnums, 0);                // s1 = 22
    auto s2 = fold_left(
      std::plus<>(), vnums, 0); // s2 = 22
    auto s3 = fold_right(
      [](const int s, const int n) {return s + n; },
      vnums, 0);                // s3 = 22
    auto s4 = fold_right(
      std::plus<>(), vnums, 0); // s4 = 22 
    ```

+   将向量中的字符串连接成一个单一字符串：

    ```cpp
    auto texts =
      std::vector<std::string>{"hello"s, " "s, "world"s, "!"s};
    auto txt1 = fold_left(
      [](std::string const & s, std::string const & n) {
      return s + n;},
      texts, ""s);    // txt1 = "hello world!"
    auto txt2 = fold_right(
      [](std::string const & s, std::string const & n) {
      return s + n; },
      texts, ""s);    // txt2 = "!world hello" 
    ```

+   将字符数组连接成一个字符串：

    ```cpp
    char chars[] = {'c','i','v','i','c'};
    auto str1 = fold_left(std::plus<>(), chars, ""s);
    // str1 = "civic"
    Auto str2 = fold_right(std::plus<>(), chars, ""s);
    // str2 = "civic" 
    ```

+   根据已计算的词频统计文本中的单词数量，这些词频存储在`map<string, int>`中：

    ```cpp
    auto words = std::map<std::string, int>{
      {"one", 1}, {"two", 2}, {"three", 3} };
    auto count = fold_left(
      [](int const s, std::pair<std::string, int> const kvp) {
        return s + kvp.second; },
      words, 0); // count = 6 
    ```

## 更多...

这些函数可以被管道化——也就是说，它们可以用另一个函数的结果调用一个函数。以下示例通过将 `std::abs()` 函数应用于其元素，将一系列整数映射到一系列正整数。然后将结果映射到另一个平方数的范围。这些数通过在范围上应用左折叠而相加：

```cpp
auto vnums = std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 };
auto s = fold_left(
  std::plus<>(),
  mapf(
    [](int const i) {return i*I; },
    mapf(
      [](int const i) {return std::abs(i); },
      vnums)),
  0); // s = 236 
```

作为练习，我们可以将 `fold` 函数实现为一个变长函数模板，就像之前看到的那样。执行实际折叠的函数作为参数提供：

```cpp
template <typename F, typename T1, typename T2>
auto fold_left(F&&f, T1 arg1, T2 arg2)
{
  return f(arg1, arg2);
}
template <typename F, typename T, typename... Ts>
auto fold_left(F&& f, T head, Ts… rest)
{
  return f(head, fold_left(std::forward<F>(f), rest...));
} 
```

当我们将它与我们在 *编写带有可变数量参数的函数模板* 菜谱中编写的 `add()` 函数模板进行比较时，我们可以注意到几个差异：

+   第一个参数是一个函数，在递归调用 `fold_left` 时会被完美转发。

+   末尾的情况是一个需要两个参数的函数，因为我们使用的折叠函数是二元的（接受两个参数）。

+   我们编写的两个函数的返回类型被声明为 `auto`，因为它们必须匹配提供的二元函数 `f` 的返回类型，而 `f` 的返回类型在我们调用 `fold_left` 之前是未知的。

`fold_left()` 函数可以使用如下方式：

```cpp
auto s1 = fold_left(std::plus<>(), 1, 2, 3, 4, 5);
// s1 = 15
auto s2 = fold_left(std::plus<>(), "hello"s, ' ', "world"s, '!');
// s2 = "hello world!"
auto s3 = fold_left(std::plus<>(), 1); // error, too few arguments 
```

注意到最后一次调用会产生编译器错误，因为变长函数模板 `fold_left()` 至少需要传入两个参数才能调用提供的二元函数。

## 参见

+   *第二章*，*创建字符串辅助库*，了解如何创建有用的文本实用工具，这些工具在标准库中并不直接可用

+   *编写带有可变数量参数的函数模板*，了解变长模板如何使我们能够编写可以接受任意数量参数的函数

+   *将函数组合成高阶函数*，学习从一个或多个其他函数创建新函数的函数式编程技术

# 将函数组合成高阶函数

在前面的菜谱中，我们实现了两个高阶函数，`map` 和 `fold`，并看到了它们的各种用法示例。在菜谱的结尾，我们看到了它们如何通过几个原始数据的转换来生成最终值。管道化是一种组合形式，这意味着从两个或更多给定的函数中创建一个新的函数。在提到的例子中，我们实际上并没有组合函数；我们只是用一个函数的结果调用另一个函数，但在这个菜谱中，我们将学习如何将函数实际组合成一个新的函数。为了简单起见，我们只考虑一元函数（只接受一个参数的函数）。

## 准备工作

在你继续之前，建议你阅读之前的菜谱，*实现高阶函数 map 和 fold*。这并不是理解这个菜谱的强制要求，但我们将参考在那里实现的 `map` 和 `fold` 函数。

## 如何实现...

要将一元函数组合成高阶函数，你应该这样做：

+   要组合两个函数，提供一个函数，该函数接受两个函数`f`和`g`作为参数，并返回一个新的函数（一个 lambda），该函数返回`f(g(x))`，其中`x`是组合函数的参数：

    ```cpp
    template <typename F, typename G>
    auto compose(F&& f, G&& g)
    {
      return = { return f(g(x)); };
    }
    auto v = compose(
      [](int const n) {return std::to_string(n); },
      [](int const n) {return n * n; })(-3); // v = "9" 
    ```

+   要组合可变数量的函数，提供之前描述的函数的可变模板重载：

    ```cpp
    template <typename F, typename... R>
    auto compose(F&& f, R&&... r)
    {
      return = { return f(compose(r...)(x)); };
    }
    auto n = compose(
      [](int const n) {return std::to_string(n); },
      [](int const n) {return n * n; },
      [](int const n) {return n + n; },
      [](int const n) {return std::abs(n); })(-3); // n = "36" 
    ```

## 它是如何工作的...

将两个一元函数组合成一个新的函数相对简单。创建一个模板函数，我们在前面的例子中将其称为`compose()`，它有两个参数——`f`和`g`——代表函数，并返回一个接受一个参数`x`的函数，并返回`f(g(x))`。重要的是，`g`函数返回的值的类型与`f`函数的参数类型相同。组合函数返回的值是一个闭包——也就是说，它是 lambda 的一个实例化。

在实践中，能够组合不仅仅是两个函数是非常有用的。这可以通过编写`compose()`函数的可变模板版本来实现。可变模板在*编写具有可变数量参数的函数模板*配方中有更详细的解释。

可变模板通过展开参数包来暗示编译时递归。这个实现与`compose()`的第一个版本非常相似，除了以下几点：

+   它接受可变数量的函数作为参数。

+   返回的闭包递归地调用`compose()`与展开的参数包，递归在只剩两个函数时结束，在这种情况下调用之前实现的重载。

即使代码看起来像是在发生递归，这并不是真正的递归。这可以称为编译时递归，但每次展开都会调用另一个具有相同名称但参数数量不同的方法，这并不代表递归。

现在我们已经实现了这些可变模板重载，我们可以重写之前配方中的最后一个例子，*实现高阶函数 map 和 fold*。参考以下片段：

```cpp
auto s = compose(
  [](std::vector<int> const & v) {
    return fold_left(std::plus<>(), v, 0); },
  [](std::vector<int> const & v) {
    return mapf([](int const i) {return i + i; }, v); },
  [](std::vector<int> const & v) {
    return mapf([](int const i) {return std::abs(i); }, v); })(vnums); 
```

有一个初始整数向量，我们通过将每个元素应用`std::abs()`映射到一个只包含正值的新的向量。然后将结果映射到一个新的向量，通过将每个元素的值加倍。最后，将结果向量中的值通过将它们加到初始值`0`上折叠在一起。

## 还有更多...

组合通常用点（`.`）或星号（`*`）表示，例如`f . g`或`f * g`。实际上，我们可以在 C++中通过重载`operator*`（尝试重载操作符点几乎没有意义）做类似的事情。与`compose()`函数类似，`operator*`应该与任何数量的参数一起工作；因此，我们将有两个重载，就像在`compose()`的情况下一样：

+   第一个重载接受两个参数并调用`compose()`来返回一个新的函数。

+   第二个重载是一个变长模板函数，它再次通过展开参数包来调用`operator*`。

基于这些考虑，我们可以如下实现`operator*`：

```cpp
template <typename F, typename G>
auto operator*(F&& f, G&& g)
{
  return compose(std::forward<F>(f), std::forward<G>(g));
}
template <typename F, typename... R>
auto operator*(F&& f, R&&... r)
{
  return operator*(std::forward<F>(f), r...);
} 
```

现在我们可以通过应用`operator*`而不是更冗长的`compose()`调用来简化函数的实际组合：

```cpp
auto n =
  ([](int const n) {return std::to_string(n); } *
   [](int const n) {return n * n; } *
   [](int const n) {return n + n; } *
   [](int const n) {return std::abs(n); })(-3); // n = "36"
auto c =
  [](std::vector<int> const & v) {
    return fold_left(std::plus<>(), v, 0); } *
  [](std::vector<int> const & v) {
    return mapf([](int const i) {return i + i; }, v); } *
  [](std::vector<int> const & v) {
    return mapf([](int const i) {return std::abs(i); }, v); };
auto vnums = std::vector<int>{ 2, -3, 5 };
auto s = c(vnums); // s = 20 
```

虽然乍一看可能不太直观，但函数是按相反的顺序应用的，而不是文本中显示的顺序。例如，在第一个例子中，参数的绝对值被保留。然后，结果被加倍，然后该操作的结果再乘以自身。最后，结果被转换为字符串。对于提供的参数`-3`，最终结果是字符串`"36"`。

## 参见

+   *编写一个带有可变数量参数的函数模板*，以了解变长模板如何使我们能够编写可以接受任意数量参数的函数

# 统一调用任何可调用对象

开发者，尤其是那些实现库的开发者，有时需要以统一的方式调用可调用对象。这可能是一个函数、一个函数指针、一个成员函数指针或一个函数对象。此类情况的例子包括`std::bind`、`std::function`、`std::mem_fn`和`std::thread::thread`。C++17 定义了一个标准函数`std::invoke()`，它可以调用任何可调用对象并传递提供的参数。这并不是要取代对函数或函数对象的直接调用，但在模板元编程中实现各种库函数时非常有用。

## 准备工作

对于这个配方，你应该熟悉如何定义和使用函数指针。

为了说明`std::invoke()`如何在不同的上下文中使用，我们将使用以下函数和类：

```cpp
int add(int const a, int const b)
{
  return a + b;
}
struct foo
{
  int x = 0;
  void increment_by(int const n) { x += n; }
}; 
```

在下一节中，我们将探讨`std::invoke()`函数的可能用例。

## 如何实现...

`std::invoke()`函数是一个变长函数模板，它接受可调用对象作为第一个参数，以及一个可变数量的参数列表，这些参数被传递给调用。`std::invoke()`可以用来调用以下内容：

+   自由函数：

    ```cpp
    auto a1 = std::invoke(add, 1, 2);   // a1 = 3 
    ```

+   通过函数指针实现的自由函数：

    ```cpp
    auto a2 = std::invoke(&add, 1, 2);  // a2 = 3
    int(*fadd)(int const, int const) = &add;
    auto a3 = std::invoke(fadd, 1, 2);  // a3 = 3 
    ```

+   通过成员函数指针实现的成员函数：

    ```cpp
    foo f;
    std::invoke(&foo::increment_by, f, 10); 
    ```

+   数据成员：

    ```cpp
    foo f;
    auto x1 = std::invoke(&foo::x, f);  // x1 = 0 
    ```

+   函数对象：

    ```cpp
    foo f;
    auto x3 = std::invoke(std::plus<>(),
      std::invoke(&foo::x, f), 3); // x3 = 3 
    ```

+   Lambda 表达式：

    ```cpp
    auto l = [](auto a, auto b) {return a + b; };
    auto a = std::invoke(l, 1, 2); // a = 3 
    ```

在实践中，`std:invoke()`应该在模板元编程中用于调用具有任意数量参数的函数。为了说明这种情况，我们将展示`std::apply()`函数的可能实现，以及 C++17 标准库的一部分，它通过将元组的成员解包到函数的参数中调用函数：

```cpp
namespace details
{
  template <class F, class T, std::size_t... I>
  auto apply(F&& f, T&& t, std::index_sequence<I...>)
 {
    return std::invoke(
      std::forward<F>(f),
      std::get<I>(std::forward<T>(t))...);
  }
}
template <class F, class T>
auto apply(F&& f, T&& t)
{
  return details::apply(
    std::forward<F>(f),
    std::forward<T>(t),
    std::make_index_sequence<
      std::tuple_size_v<std::decay_t<T>>> {}); 
} 
```

## 它是如何工作的...

在我们了解`std::invoke()`如何工作之前，让我们快速看一下如何调用不同的可调用对象。给定一个函数，显然，调用它的通用方式是直接传递必要的参数。然而，我们也可以使用函数指针来调用函数。函数指针的问题在于定义指针类型可能会很繁琐。使用`auto`可以简化事情（如下面的代码所示），但在实践中，你通常需要首先定义函数指针的类型，然后定义一个对象并用正确的函数地址初始化它。以下是一些示例：

```cpp
// direct call
auto a1 = add(1, 2);    // a1 = 3
// call through function pointer
int(*fadd)(int const, int const) = &add;
auto a2 = fadd(1, 2);   // a2 = 3
auto fadd2 = &add;
auto a3 = fadd2(1, 2);  // a3 = 3 
```

当你需要通过类的实例调用类函数时，通过函数指针调用会变得更为繁琐。定义成员函数指针和调用它的语法并不简单：

```cpp
foo f;
f.increment_by(3);
auto x1 = f.x;    // x1 = 3
void(foo::*finc)(int const) = &foo::increment_by;
(f.*finc)(3);
auto x2 = f.x;    // x2 = 6
auto finc2 = &foo::increment_by;
(f.*finc2)(3);
auto x3 = f.x;    // x3 = 9 
```

无论这种调用看起来多么繁琐，实际的问题是编写能够以统一方式调用这些类型可调用对象的库组件（函数或类）。这正是从标准函数，如`std::invoke()`中实际受益的地方。

`std::invoke()`的实现细节很复杂，但可以用简单的话来解释它的工作方式。假设调用形式为`invoke(f, arg1, arg2, ..., argN)`，那么考虑以下：

+   如果`f`是指向`T`类成员函数的指针，那么调用等同于以下两种情况之一：

    +   如果`arg1`是`T`的实例，则为`(arg1.*f)(arg2, ..., argN)`

    +   如果`arg1`是`reference_wrapper`的特化，则为`(arg1.get().*f)(arg2, ..., argN)`

    +   `((*arg1).*f)(arg2, ..., argN)`，如果它不是其他情况

+   如果`f`是指向`T`类数据成员的指针，并且有一个单独的参数——换句话说，调用形式为`invoke(f, arg1)`——那么调用等同于以下两种情况之一：

    +   如果`arg1`是`T`的实例，则为`arg1.*f`

    +   如果`arg1`是`reference_wrapper`的特化，则为`arg1.get().*f`

    +   `(*arg1).*f`，如果它不是其他情况

+   如果`f`是一个函数对象，那么调用等同于`f(arg1, arg2, ..., argN)`

标准库还提供了一系列相关的类型特性：一方面是`std::is_invocable`和`std::is_nothrow_invocable`，另一方面是`std::is_invocable_r`和`std::is_nothrow_invocable_r`。第一组确定一个函数是否可以用提供的参数调用，而第二组确定它是否可以用提供的参数调用并产生可以隐式转换为指定类型的结果。这些类型特性的*nothrow*版本验证调用可以在不抛出任何异常的情况下完成。

截至 C++20，`std::invoke`函数是`constexpr`，这意味着它可以在编译时调用可调用对象。

在 C++23 中，已添加了一个类似的实用工具 `std::invoke_r`。它有一个额外的模板参数（第一个），它是一个表示返回值类型的类型模板参数（除非它是 `void`），或者是一个可以将返回值隐式转换为的类型。

## 参见

+   *编写一个带有可变数量参数的函数模板*，以了解变长模板如何使我们能够编写可以接受任意数量参数的函数

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
