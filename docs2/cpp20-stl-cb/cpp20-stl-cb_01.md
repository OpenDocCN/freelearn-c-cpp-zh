# *第一章*: 新的 C++20 特性

本章主要集中介绍 C++20 为 STL 添加的一些更具吸引力的特性。其中一些特性您可以立即使用。其他特性可能需要等待您喜欢的编译器实现。但从长远来看，我预计您会想了解这些特性中的大多数。

C++20 标准新增了很多内容，远远超出了我们在这里所能涵盖的范围。以下是一些我认为将产生长期影响的特性。

在本章中，我们将介绍以下食谱：

+   使用新的`format`库格式化文本

+   使用`constexpr`编译时向量和字符串

+   安全比较不同类型的整数

+   使用“飞船”运算符`<=>`进行三路比较

+   使用`<version>`头文件轻松找到特性测试宏

+   使用概念和约束创建更安全的模板

+   使用模块避免重新编译模板库

+   使用范围创建容器视图

本章旨在使您熟悉 C++20 中的这些新特性，以便您可以在自己的项目中使用它们，并在遇到它们时理解它们。

# 技术要求

本章的代码文件可以在 GitHub 上找到：[`github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap01`](https://github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap01)。

# 使用新的格式化库格式化文本

到目前为止，如果您想格式化文本，可以使用传统的`printf`函数或 STL 的`iostream`库。两者都有其优点和缺点。

基于`printf`的函数是从 C 继承而来的，并且经过 50 多年的证明，它们既高效、灵活又方便。格式化语法看起来可能有点晦涩，但一旦习惯了，就足够简单。

```cpp
printf("Hello, %s\n", c_string);
```

`printf`的主要弱点是其缺乏类型安全。常见的`printf()`函数（及其相关函数）使用 C 的*可变参数*模型将参数传递给格式化器。当它起作用时，效果很好，但当参数类型与其对应的格式说明符不匹配时，可能会引起严重问题。现代编译器尽可能多地执行类型检查，但该模型本身有缺陷，保护作用有限。

STL 的`iostream`库以牺牲可读性和运行时性能为代价，带来了类型安全。`iostream`的语法不寻常，但熟悉。它重载了*位左移运算符*（`<<`），允许一系列对象、操作数和*格式化操作符*，从而生成格式化输出。

```cpp
cout << "Hello, " << str << endl;
```

`iostream`的弱点在于其复杂度，无论是语法还是实现。构建格式化字符串可能既冗长又晦涩。许多格式化操作符在使用后必须重置，否则会创建级联的格式化错误，这可能导致难以调试。该库本身庞大而复杂，导致代码比其`printf`等效版本大得多且运行速度慢。

这种令人沮丧的情况让 C++程序员别无选择，只能在这两个有缺陷的系统之间做出选择，直到现在。

## 如何做到这一点...

新的 `format` 库位于 `<format>` 头文件中。截至本文写作时，`format` 仅在 *MSVC*（微软）编译器中实现。到你阅读本文时，它应该可以在更多系统上使用。否则，你可以从 `fmt.dev`（[j.bw.org/fmt](http://j.bw.org/fmt)）作为第三方库使用其参考实现。

`format` 库是基于 Python 3 中的 `str.format()` 方法构建的。*格式化字符串*与 Python 中的格式化字符串基本相同，并且在大多数情况下可以互换。让我们看看一些简单的例子：

+   在其最简单形式中，`format()` 函数接受一个 `string_view` 格式字符串和一个 *可变参数包* 的参数。它返回一个 `string`。其函数签名看起来像这样：

    ```cpp
    template<typename... Args>
    string format(string_view fmt, const Args&... args);
    ```

+   `format()` 函数返回几乎任何类型或值的 `string` 表示形式。例如：

    ```cpp
    string who{ "everyone" };
    int ival{ 42 };
    double pi{ std::numbers::pi };
    format("Hello, {}!\n ", who);   // Hello, everyone!
    format("Integer: {}\n ", ival); // Integer: 42
    format("π: {}\n", pi);          // π: 3.141592653589793
    ```

*格式化字符串* 使用花括号 `{}` 作为占位符。如果没有 *格式说明符*，花括号实际上是一个类型安全的 *占位符*，它将任何兼容类型的值转换为合理的字符串表示形式。

+   你可以在格式化字符串中包含多个占位符，如下所示：

    ```cpp
    format("Hello {} {}", ival, who);  // Hello 42 
                                       // everyone
    ```

+   你可以指定替换值的顺序。这可能对国际化很有用：

    ```cpp
    format("Hello {1} {0}", ival, who); // Hello everyone 42
    format("Hola {0} {1}", ival, who);  // Hola 42 everyone
    ```

+   你可以左对齐（`<`）、右对齐（`>`）或居中对齐（`^`）值，可以带或不带填充字符：

    ```cpp
    format("{:.<10}", ival);  // 42........
    format("{:.>10}", ival);  // ........42
    format("{:.¹⁰}", ival);  // ....42....
    ```

+   你可以设置值的十进制精度：

    ```cpp
    format("π: {:.5}", pi);  // π: 3.1416
    ```

+   以及更多更多。

这是一个丰富且完整的格式化规范，它提供了 `iostream` 的类型安全，以及 `printf` 的性能和简单性，实现了两者的最佳结合。

## 它是如何工作的……

`format` 库尚未包含 `print()` 函数，该函数计划在 *C++23* 中实现。`format()` 函数本身返回一个 `string` 对象。因此，如果你想打印字符串，你需要使用 `iostream` 或 `cstdio`。 (悲伤的表情。)

你可以使用 `iostream` 打印字符串：

```cpp
cout << format("Hello, {}", who) << "\n";
```

或者你也可以使用 `cstdio`：

```cpp
puts(format("Hello, {}", who).c_str());
```

这两种方法都不理想，但编写一个简单的 `print()` 函数并不困难。我们可以通过这个过程了解 `format` 库的一些内部工作原理。

这里是使用 `format` 库实现的 `print()` 函数的一个简单示例：

```cpp
#include <format>
#include <string_view>
#include <cstdio>
template<typename... Args>
void print(const string_view fmt_str, Args&&... args) {
    auto fmt_args{ make_format_args(args...) };
    string outstr{ vformat(fmt_str, fmt_args) };
    fputs(outstr.c_str(), stdout);
} 
```

这使用与 `format()` 函数相同的参数。第一个参数是格式字符串的 `string_view` 对象。随后是一个参数的可变参数包。

`make_format_args()` 函数接受参数包并返回一个包含 *类型擦除值* 的对象，这些值适合格式化。然后，该对象被传递给 `vformat()`，它返回一个适合打印的 `string`。我们使用 `fputs()` 将值打印到控制台，因为它比 `cout` 效率要高得多。

我们现在可以使用这个 `print()` 函数代替 `cout << format()` 组合：

```cpp
print("Hello, {}!\n", who);
print("π: {}\n", pi);
print("Hello {1} {0}\n", ival, who);
print("{:.¹⁰}\n", ival);
print("{:.5}\n", pi);
```

输出：

```cpp
Hello, everyone!
π: 3.141592653589793
Hello everyone 42
....42....
3.1416
```

当您最终获得一个支持`print()`的 C++23 编译器时，您应该能够简单地用`using std::print;`替换上面的`print()`模板函数定义，并且所有的`print()`调用都应该继续工作。

## 还有更多……

能够格式化字符串和原始数据很有用，但为了让`format`库完全功能，它需要自定义以与您自己的类一起工作。

例如，这里有一个简单的`struct`结构，包含两个成员：一个**分子**和一个**分母**。我们希望它以分数的形式打印出来：

```cpp
struct Frac {
    long n;
    long d;
};
int main() {
    Frac f{ 5, 3 };
    print("Frac: {}\n", f);    
}
```

当我编译这段代码时，会出现一系列错误，效果类似于“没有用户定义的转换操作符……”。不错。那么，让我们来修复它！

当`format`系统遇到一个需要进行**转换**的对象时，它会寻找与相应类型对应的`formatter`对象的**特化**。标准特化包括字符串和数字等常见对象。

为我们的`Frac`类型创建一个特化相当简单：

```cpp
template<>
struct std::formatter<Frac>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }
    template<typename FormatContext>
    auto format(const Frac& f, FormatContext& ctx) {
        return format_to(ctx.out(), "{0:d}/{1:d}", 
            f.n, f.d);
    }
};
```

这个`formatter`特化是一个包含两个短模板函数的类：

+   `parse()`函数解析冒号（或如果没有冒号，则在开括号之后）之后的**格式字符串**，直到但不包括闭括号。（换句话说，指定对象类型的部分。）它接受一个`ParseContext`对象并返回一个迭代器。对于我们的目的，我们只需返回`begin()`迭代器，因为我们不需要为我们的**类型**添加任何新语法。您很少需要在这里放置其他内容。

+   `format()`函数接受一个`Frac`对象和一个`FormatContext`对象。它返回一个**结束迭代器**。`format_to()`函数使这变得简单。它接受一个迭代器、一个格式字符串和一个参数包。在这种情况下，参数包是我们`Frac`类的两个属性，即分子和分母。

在这里，我们只需要提供一个简单的格式字符串`"{0}/{1}"`以及分子和分母的值。（`0`和`1`表示参数的位置。它们不是必需的，但将来可能会有用。）

现在我们为`Frac`创建了一个特化，我们可以将我们的对象传递给`print()`以获得可读的结果：

```cpp
int main() {
    Frac f{ 5, 3 };
    print("Frac: {}\n", f);    
}
```

输出：

```cpp
Frac: 5/3
```

C++20 的`format`库通过提供一个既高效又方便的类型安全文本格式化库来解决了一个长期存在的问题。

# 使用`constexpr`编译时向量和字符串

C++20 允许在多个新的上下文中使用`constexpr`。这提供了改进的效率，因为这些事情可以在编译时而不是运行时进行评估。

## 如何做到这一点……

该规范包括在`constexpr`上下文中使用`string`和`vector`对象的能力。重要的是要注意，这些对象本身可能不是`constexpr`声明的，但它们可以在编译时上下文中使用：

```cpp
constexpr auto use_string() {
    string str{"string"};
    return str.size();
}
```

您还可以在`constexpr`上下文中使用算法：

```cpp
constexpr auto use_vector() {
    vector<int> vec{ 1, 2, 3, 4, 5};
    return accumulate(begin(vec), end(vec), 0);
}
```

`accumulate`算法的结果在编译时和`constexpr`上下文中都是可用的。

## 它是如何工作的……

`constexpr` 修饰符声明了一个可能在编译时评估的变量或函数。在 C++20 之前，这仅限于使用字面值初始化的对象，或者是在有限约束内的函数。C++17 允许有某种程度的扩展使用，而 C++20 进一步扩展了它。

截至 C++20，STL 的 `string` 和 `vector` 类现在有了 `constexpr` 修饰的构造函数和析构函数，这使得它们可以在编译时调用。这也意味着为 `string` 或 `vector` 对象分配的内存 *必须在编译时释放*。

例如，这个返回`vector`的`constexpr`函数将无错误编译：

```cpp
constexpr auto use_vector() {
    vector<int> vec{ 1, 2, 3, 4, 5};
    return vec;
}
```

但如果在运行时环境中尝试使用该结果，你将得到一个关于在常量评估期间分配的内存的错误：

```cpp
int main() {
    constexpr auto vec = use_vector();
    return vec[0];
}
```

这是因为`vector`对象是在编译期间分配和释放的。因此，该对象在运行时不再可用。

另一方面，你可以在运行时使用`vector`对象的某些`constexpr`修饰的方法，例如`size()`：

```cpp
int main() {
    constexpr auto value = use_vector().size();
    return value;
}
```

因为`size()`方法是`constexpr`修饰的，表达式可以在编译时评估。

# 安全地比较不同类型的整数

比较不同类型的整数可能不会总是产生预期的结果。例如：

```cpp
int x{ -3 };
unsigned y{ 7 };
if(x < y) puts("true");
else puts("false");
```

你可能期望这段代码打印 `true`，这是可以理解的。-3 通常小于 7。但它会打印 `false`。

问题在于`x`是有符号的，而`y`是无符号的。标准化的行为是将有符号类型转换为无符号类型进行比较。这似乎有些反直觉，不是吗？确实，你不能可靠地将无符号值转换为相同大小的有符号值，因为有符号整数使用二进制补码表示（它使用最高位作为符号）。对于相同大小的整数，最大有符号值是无符号值的一半。使用这个例子，如果你的整数是 32 位，-3（有符号）变为`FFFF FFFD`（十六进制），或 4,294,967,293（无符号十进制），这并不是小于 7。

一些编译器在尝试比较有符号和无符号整数值时可能会发出警告，但大多数不会。

C++20 标准在 `<utility>` 头文件中包含了一组整数安全的比较函数。

## 如何做到这一点...

新的整数比较函数可以在 `<utility>` 头文件中找到。它们各自接受两个参数，对应于运算符的左右两侧。

```cpp
#include <utility>
int main() {
    int x{ -3 };
    unsigned y{ 7 };
    if(cmp_less(x, y)) puts("true");
    else puts("false");
}
```

`cmp_less()` 函数给出了我们期望的结果。-3 小于 7，程序现在打印 `true`。

`<utility>` 头文件提供了完整的整数比较函数。假设我们的 `x` 和 `y` 的值，我们得到以下比较：

```cpp
cmp_equal(x, y)          // x == y is false
cmp_not_equal(x, y)      // x != y is true
cmp_less(x, y)           // x < y is true
cmp_less_equal(x, y)     // x <= y is true
cmp_greater(x, y)        // x > y is false
cmp_greater_equal(x, y)  // x >= y is false
```

## 它是如何工作的...

这里是 C++20 标准中`cmp_less()`函数的示例实现，以给你一个更完整的关于其工作方式的了解：

```cpp
template< class T, class U >
constexpr bool cmp_less( T t, U u ) noexcept
{
    using UT = make_unsigned_t<T>;
    using UU = make_unsigned_t<U>;
    if constexpr (is_signed_v<T> == is_signed_v<U>)
        return t < u;
    else if constexpr (is_signed_v<T>)
        return t < 0 ? true : UT(t) < u;
    else
        return u < 0 ? false : t < UU(u);
}
```

`UT` 和 `UU` 别名被声明为 `make_unsigned_t`，这是一个在 C++17 中引入的有用的辅助类型。这允许安全地将有符号类型转换为无符号类型。

函数首先测试两个参数是否都是有符号或无符号。如果是这样，它返回一个简单的比较。

然后它测试任一边是否为有符号。如果该有符号值小于零，它可以不执行比较就返回 `true` 或 `false`。否则，它将有符号值转换为无符号并返回比较结果。

类似的逻辑应用于每个其他比较函数。

# 使用“飞船”运算符 `<=>` 进行三向比较

**三向比较**运算符 (`<=>`)，通常称为**飞船**运算符，因为从侧面看它像一只飞碟，是 C++20 中的新特性。你可能想知道，现有的六个比较运算符有什么问题？根本没问题，你将继续使用它们。飞船的目的在于为对象提供一个统一的比较运算符。

常用的双向比较运算符根据比较结果返回两种状态之一，`true` 或 `false`。例如：

```cpp
const int a = 7;
const int b = 42;
static_assert(a < b);
```

`a < b` 表达式使用**小于比较**运算符 (`<`) 来测试 `a` 是否小于 `b`。如果条件满足，比较运算符返回 `true`，如果不满足，则返回 `false`。在这种情况下，它返回 `true`，因为 7 小于 42。

三向比较的工作方式不同。它返回三种状态之一。如果操作数相等，飞船运算符将返回等于 `0` 的值；如果左操作数小于右操作数，则返回**负值**；如果左操作数大于右操作数，则返回**正值**。

```cpp
const int a = 7;
const int b = 42;
static_assert((a <=> b) < 0);
```

返回值**不是一个整数**。它是一个来自 `<compare>` 头文件的比较对象，与 `0` 进行比较。

如果操作数具有整型，运算符返回 `<compare>` 库中的 `strong_ordering` 对象。

```cpp
strong_ordering::equal    // operands are equal
strong_ordering::less     // lhs is less than rhs
strong_ordering::greater  // lhs is greater than rhs
```

如果操作数具有浮点类型，运算符返回一个 `partial_ordering` 对象：

```cpp
partial_ordering::equivalent  // operands are equivelant
partial_ordering::less        // lhs is less than rhs
partial_ordering::greater     // lhs is greater than rhs
partial_ordering::unordered   // if an operand is unordered
```

这些对象被设计成使用传统的比较运算符（例如，`(a <=> b) < 0`）与字面量零 (`0`) 进行比较。这使得三向比较的结果比传统比较更精确。

如果所有这些都显得有些复杂，那没关系。对于大多数应用，你永远不会直接使用飞船运算符。它的真正力量在于它作为对象统一比较运算符的应用。让我们深入探讨一下。

## 如何做到这一点...

让我们看看一个简单的类，它封装了一个整数并提供比较运算符：

```cpp
struct Num {
    int a;
    constexpr bool operator==(const Num& rhs) const 
        { return a == rhs.a; }
    constexpr bool operator!=(const Num& rhs) const
        { return !(a == rhs.a); }
    constexpr bool operator<(const Num& rhs) const
        { return a < rhs.a; }
    constexpr bool operator>(const Num& rhs) const
        { return rhs.a < a; }
    constexpr bool operator<=(const Num& rhs) const
        { return !(rhs.a < a); }
    constexpr bool operator>=(const Num& rhs) const
        { return !(a < rhs.a); }
};
```

看到这样的比较运算符重载列表并不罕见。实际上，它应该与**非成员友元**更复杂，这些友元与运算符两边的对象一起工作。

使用新的飞船运算符，所有这些都可以通过一个重载来完成：

```cpp
#include <compare>
struct Num {
    int a;
    constexpr Num(int a) : a{a} {}
    auto operator<=>(const Num&) const = default;
};
```

注意，我们需要包含 `<compare>` 头文件以支持三路运算符的返回类型。现在我们可以声明一些变量并通过比较来测试它们：

```cpp
constexpr Num a{ 7 };
constexpr Num b{ 7 };
constexpr Num c{ 42 };
int main() {
    static_assert(a < c);
    static_assert(c > a);
    static_assert(a == b);
    static_assert(a <= b);
    static_assert(a <= c);
    static_assert(c >= a);
    static_assert(a != c);
    puts("done.");
}
```

编译器将自动优先选择 `<=>` 运算符进行每个比较。

因为默认的 `<=>` 运算符已经是 `constexpr` 安全的，所以我们不需要在我们的成员函数中声明它为 `constexpr`。

## 它是如何工作的…

`operator<=>` 重载利用了 C++20 的新概念，*重写表达式*。在重载解析过程中，编译器根据一组规则重写表达式。例如，如果我们写 `a < b`，编译器将重写它为 `(a <=> b < 0)`，以便与我们的成员运算符一起工作。编译器将重写 `<=>` 运算符的每个相关比较表达式，其中我们没有包含更具体的运算符。

事实上，我们不再需要一个非成员函数来处理与左侧兼容类型的比较。编译器将 *合成* 一个与成员运算符一起工作的表达式。例如，如果我们写 `42 > a`，编译器将合成一个反转运算符的表达式 `(a <=> 42 < 0)`，以便与我们的成员运算符一起工作。

注意

`<=>` 运算符的优先级高于其他比较运算符，因此它总是首先评估。所有比较运算符都是从左到右评估的。

## 还有更多…

默认运算符可以与各种类一起正常工作，包括具有多个不同类型数值成员的类：

```cpp
struct Nums {
  int i;
  char c;
  float f;
  double d;
  auto operator<=>(const Nums&) const = default;
};
```

但如果你有一个更复杂的数据类型呢？这里有一个简单的分数类的例子：

```cpp
struct Frac {
    long n;
    long d;
    constexpr Frac(int a, int b) : n{a}, d{b} {}
    constexpr double dbl() const {
        return static_cast<double>(n) / 
          static_cast<double>(d);
    }
    constexpr auto operator<=>(const Frac& rhs) const {
        return dbl() <=> rhs.dbl();
    };
    constexpr auto operator==(const Frac& rhs) const {
        return dbl() <=> rhs.dbl() == 0;
    };
};
```

在这种情况下，我们需要定义 `operator<=>` 重载，因为我们的数据成员不是独立的标量值。这仍然相当简单，并且效果很好。

注意，我们还需要一个 `operator==` 重载。这是因为表达式重写规则不会重写带有自定义 `operator<=>` 重载的 `==` 和 `!=`。你只需要定义 `operator==`。编译器将根据需要重写 `!=` 表达式。

现在，我们可以定义一些对象：

```cpp
constexpr Frac a(10,15);  // compares equal with 2/3
constexpr Frac b(2,3);
constexpr Frac c(5,3);
```

我们可以用正常的比较运算符来测试它们，正如预期的那样：

```cpp
int main() {
    static_assert(a < c);
    static_assert(c > a);
    static_assert(a == b);
    static_assert(a <= b);
    static_assert(a <= c);
    static_assert(c >= a);
    static_assert(a != c);
}
```

空间船运算符的力量在于其简化类中比较重载的能力。与独立重载每个运算符相比，它提高了简单性和效率。

# 使用 `<version>` 头文件轻松找到特性测试宏

C++ 在添加新功能的同时，一直提供了一些形式的特性测试宏。从 C++20 开始，这个过程被标准化，所有 *库特性* 测试宏都已添加到 `<version>` 头文件中。这将使测试代码中的新功能变得更加容易。

这是一个有用的特性，并且使用起来非常简单。

## 如何做到这一点…

所有功能测试宏都以前缀 `__cpp_` 开头。库功能以 `__cpp_lib_` 开头。语言功能测试宏通常由编译器定义。库功能测试宏在新 `<version>` 头文件中定义。你可以像使用任何其他预处理器宏一样使用它们：

```cpp
#include <version>
#ifdef __cpp_lib_three_way_comparison
#   include <compare>
#else
#   error Spaceship has not yet landed
#endif
```

在某些情况下，你可以使用 `__has_include` 预处理器运算符（自 C++17 引入）来测试包含文件的存在。

```cpp
#if __has_include(<compare>)
#   include <compare>
#else
#   error Spaceship has not yet landed
#endif
```

你可以使用 `__has_include` 来测试任何头文件的存在。因为它是一个预处理器指令，所以它不需要自己的头文件来工作。

## 它是如何工作的…

通常，你可以通过使用 `#ifdef` 或 `#if defined` 测试非零值来使用功能测试宏。每个功能测试宏都有一个非零值，对应于它被标准委员会接受的那一年和一个月。例如，`__cpp_lib_three_way_comparison` 宏的值为 `201907`。这意味着它在 2019 年 7 月被接受。

```cpp
#include <version>
#ifdef __cpp_lib_three_way_comparison
    cout << "value is " << __cpp_lib_three_way_comparison 
        << "\n"
#endif
```

输出：

```cpp
$ ./working
value is 201907
```

宏的值在某些不常见的情况下可能很有用，在这些情况下，功能已更改，而你依赖于这些更改。对于大多数目的，你可以安全地忽略值，只需使用 `#ifdef` 测试非零值即可。

几个网站维护了一个功能测试宏的完整列表。我倾向于使用 *cppreference* ([`j.bw.org/cppfeature`](https://j.bw.org/cppfeature))，但还有其他网站。

# 使用概念和约束创建更安全的模板

模板非常适合编写与不同类型一起工作的代码。例如，这个函数将适用于任何数值类型：

```cpp
template <typename T>
T arg42(const T & arg) {
    return arg + 42;
}
```

但当你尝试用非数值类型调用它时会发生什么呢？

```cpp
const char * n = "7";
cout << "result is " << arg42(n) << "\n";
```

输出：

```cpp
Result is ion
```

这个程序编译和运行没有错误，但结果不可预测。实际上，这个调用是危险的，它很容易崩溃或成为漏洞。我更希望编译器生成错误信息，这样我就可以修复代码。

现在，有了概念，我可以这样写：

```cpp
template <typename T>
requires Numeric<T>
T arg42(const T & arg) {
    return arg + 42;
}
```

`requires` 关键字是 C++20 中的新特性。它将约束应用于模板。`Numeric` 是一个只接受整数和浮点类型的 *概念* 的名称。现在，当我用非数值参数编译这段代码时，我得到了一个合理的编译器错误：

```cpp
error: 'arg42': no matching overloaded function found
error: 'arg42': the associated constraints are not satisfied
```

这样的错误信息比大多数编译器错误更有用。

让我们更详细地看看如何在代码中使用概念和约束。

## 如何做到这一点…

概念只是一个命名的约束。上面的 `Numeric` 概念看起来像这样：

```cpp
#include <concepts>
template <typename T>
concept Numeric = integral<T> || floating_point<T>;
```

这个 *概念* 需要一个满足 `std::integral` 或 `std::floating_point` 预定义概念的类型 `T`。这些概念包含在 `<concepts>` 头文件中。

概念和约束可用于类模板、函数模板或变量模板。我们已经看到了一个约束函数模板的例子，现在这里有一个简单的约束类模板示例：

```cpp
template<typename T>
requires Numeric<T>
struct Num {
    T n;
    Num(T n) : n{n} {}
};
```

这里还有一个简单的变量模板示例：

```cpp
template<typename T>
requires floating_point<T>
T pi{3.1415926535897932385L};
```

你可以在任何模板上使用概念和约束。让我们考虑一些进一步的例子。为了简单起见，我们将使用函数模板。

+   约束可以使用概念或 *类型特性* 来评估类型的特征。你可以使用 `<type_traits>` 头文件中找到的任何类型特性，只要它返回一个 `bool`。

例如：

```cpp
template<typename T>
requires is_integral<T>::value  // value is bool
constexpr double avg(vector<T> const& vec) {
    double sum{ accumulate(vec.begin(), vec.end(), 
      0.0)
    };
    return sum / vec.size();
}
```

+   `requires` 关键字是 C++20 中新增的。它为模板参数引入了一个约束。在这个例子中，约束表达式测试模板参数是否满足类型特性 `is_integral`。

+   你可以使用 `<type_traits>` 头文件中找到的预定义特性，或者你可以定义自己的，就像定义一个模板变量一样。用于约束的变量必须返回 `constexpr bool`。例如：

    ```cpp
    template<typename T>
    constexpr bool is_gt_byte{ sizeof(T) > 1 };
    ```

这定义了一个名为 `is_gt_byte` 的类型特性。这个特性使用 `sizeof` 运算符来测试类型 `T` 是否大于 1 字节。

+   一个 *概念* 简单地是一个命名的约束集合。例如：

    ```cpp
    template<typename T>
    concept Numeric = is_gt_byte<T> &&
        (integral<T> || floating_point<T>);
    ```

这定义了一个名为 `Numeric` 的概念。它使用我们的 `is_gt_byte` 约束，以及 `<concepts>` 头文件中的 `floating_point` 和 `integral` 概念。我们可以用它来约束模板，使其只接受大于 1 字节大小的数值类型。

```cpp
template<Numeric T>
T arg42(const T & arg) {
    return arg + 42;
}
```

你会注意到，我在模板声明中应用了约束，而不是在 `requires` 表达式的单独一行上。有几种方法可以应用一个概念。让我们看看这是如何工作的。

## 它是如何工作的…

你可以通过几种不同的方式应用一个概念或约束：

+   你可以使用 `requires` 关键字应用一个概念或约束：

    ```cpp
    template<typename T>
    requires Numeric<T>
    T arg42(const T & arg) {
        return arg + 42;
    }
    ```

+   你可以在模板声明中应用一个概念：

    ```cpp
    template<Numeric T>
    T arg42(const T & arg) {
        return arg + 42;
    }
    ```

+   你可以在函数签名中使用 `requires` 关键字：

    ```cpp
    template<typename T>
    T arg42(const T & arg) requires Numeric<T> {
        return arg + 42;
    }
    ```

+   或者，你可以在函数模板的参数列表中使用一个概念以缩写形式：

    ```cpp
    auto arg42(Numeric auto & arg) {
        return arg + 42;
    }
    ```

对于许多目的，选择这些策略之一可能只是风格问题。而且，在某些情况下，一个可能比另一个更好。

## 还有更多…

标准使用术语 *合取*、*析取* 和 *原子* 来描述可以用来构造约束的表达式类型。让我们定义这些术语。

你可以使用 `&&` 和 `||` 运算符组合概念和约束。这些组合分别称为 *合取* 和 *析取*。你可以把它们看作逻辑的 *AND* 和 *OR*。

通过使用 `&&` 运算符和两个约束形成了一个 *约束合取*。

```cpp
Template <typename T>
concept Integral_s = Integral<T> && is_signed<T>::value;
```

逻辑与（`&&`）运算符仅在两侧都满足时才成立。它的计算顺序是从左到右。逻辑与的运算数是短路操作，也就是说，如果左侧的约束不满足，则不会评估右侧。

通过使用 `||` 运算符和两个约束形成了一个 *约束析取*。

```cpp
Template <typename T>
concept Numeric = integral<T> || floating_point<T>;
```

如果`||`运算符的任一侧被满足，则析取成立。它是从左到右评估的。合取的运算符是短路，也就是说，如果左侧约束成立，则不会评估右侧。

*原子约束*是一个返回`bool`类型、不能进一步分解的表达式。换句话说，它不是合取或析取。

```cpp
template<typename T>
concept is_gt_byte = sizeof(T) > 1;
```

你还可以在原子约束中使用逻辑`!`（*非*）运算符。

```cpp
template<typename T>
concept is_byte = !is_gt_byte<T>;
```

如预期的那样，`!`运算符将右侧的`bool`表达式的值取反。

当然，我们可以将这些表达式类型组合成更大的表达式。在以下示例中，我们可以看到每种约束表达式都有示例。

```cpp
template<typename T>
concept Numeric = is_gt_byte<T> && 
    (integral<T> || floating_point<T>);
```

让我们分解一下。子表达式`(integral<T>` `floating_point<T>)`是一个*析取*。子表达式`is_gt_byte<T>` `(`…`)`是一个*合取*。而每个子表达式`integral<T>`、`floating_point<T>`和`is_gt_byte<T>`都是*原子*的。

这些区别主要是为了描述目的。虽然了解细节是好的，但在编写代码时，可以安全地将它们视为简单的逻辑`||`、`&&`和`!`运算符。

概念和约束是 C++标准的受欢迎的补充，我期待在未来的项目中使用它们。

# 避免使用模块重新编译模板库

头文件自从 C 语言一开始就存在了。最初，它们主要用于*文本替换宏*和在不同翻译单元之间链接*外部符号*。随着模板的引入，C++利用头文件来携带实际代码。因为模板需要为特化的变化重新编译，所以我们已经很多年都在头文件中携带它们。随着 STL 在多年来的持续增长，这些头文件也相应地增长。这种状况已经变得难以管理，并且不再适合未来的扩展。

头文件通常包含比模板多得多的内容。它们通常包含配置宏和其他用于系统目的但不对应用户的符号。随着头文件数量的增加，符号冲突的机会也增加了。考虑到宏的丰富性，这是一个更大的问题，因为宏不受命名空间限制，也不受任何形式的安全类型的影响。

C++20 通过*模块*解决了这个问题。

## 如何做到这一点...

你可能习惯于创建如下所示的头文件：

```cpp
#ifndef BW_MATH
#define BW_MATH
namespace bw {
    template<typename T>
    T add(T lhs, T rhs) {
        return lhs + rhs;
    }
}
#endif // BW_MATH
```

这个最小化示例说明了模块解决的一些问题。`BW_MATH`符号被用作*包含保护器*。它的唯一目的是防止头文件被多次包含，但它的符号在整个翻译单元中都被携带。当你将这个头文件包含到源文件中时，它可能看起来像这样：

```cpp
#include "bw-math.h"
#include <format>
#include <string>
#include <iostream>
```

现在 `BW_MATH` 符号对包含的每个其他头文件以及由其他头文件包含的每个头文件都可用。这有很多机会发生冲突。而且记住，编译器无法检查这些冲突。它们是宏。这意味着在编译器有机会看到它们之前，它们就被预处理器翻译了。

现在我们来到了头文件的实际重点，即模板函数：

```cpp
template<typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}
```

因为它是一个模板，每次使用 `add()` 函数时，编译器都必须创建一个单独的特化。这意味着模板函数必须在每次调用时都进行解析和特化。这就是为什么模板放在头文件中的原因；源代码必须在编译时可用。随着 STL 的增长和演变，以及其许多大型模板类和函数，这成为一个显著的扩展性问题。

*模块* 解决了这些问题以及更多。

作为模块，`bw-math.h` 变为 `bw-math.ixx`（在 MSVC 命名约定中）并且看起来是这样的：

```cpp
export module bw_math;
export template<typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}
```

注意，唯一导出的符号是模块的名称 `bw_math` 和函数的名称 `add()`。这保持了命名空间整洁。

使用它时更干净。当我们将其用于 `module-test.cpp` 时，它看起来像这样：

```cpp
import bw_math;
import std.core;
int main() {
    double f = add(1.23, 4.56);
    int i = add(7, 42);
    string s = add<string>("one ", "two");
    cout << 
        "double: " << f << "\n" <<
        "int: " << i << "\n" <<
        "string: " << s << "\n";
}
```

`import` 声明用于我们可能使用 `#include` 预处理器指令的地方。这些导入模块的符号表以进行链接。

我们示例的输出看起来像这样：

```cpp
$ ./module-test
double: 5.79
int: 49
string: one two
```

模块版本的工作方式与在头文件中完全一样，只是更干净、更高效。

注意

编译的模块包括一个单独的 *元数据文件*（在 MSVC 命名约定中为 `*module-name*`.ifc`），它描述了模块接口。这允许模块支持模板。元数据包括足够的信息，使编译器能够创建模板特化。

## 它是如何工作的…

`import` 和 `export` 声明是 *模块* 实现的核心。让我们再次看看 `bw-math.ixx` 模块：

```cpp
export module bw_math;
export template<typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}
```

注意两个 `export` 声明。第一个使用 `export module bw_math` 导出模块本身，这声明了翻译单元为模块。每个模块文件顶部必须有一个模块声明，并且在任何其他语句之前。第二个 `export` 使函数名称 `add()` 可用于 *模块消费者*。

如果你的模块需要 `#include` 指令或其他全局片段，你将需要首先使用如下简单的模块声明来声明你的模块：

```cpp
module;
#define SOME_MACRO 42
#include <stdlib.h>
export module bw_math;
...
```

文件顶部单独一行上的 `module;` 声明引入了一个 *全局模块片段*。全局模块片段中只能出现预处理器指令。这必须立即后接一个标准模块声明（`export module bw_math;`）以及模块内容的其余部分。让我们更仔细地看看它是如何工作的：

+   `export` 声明使符号对 *模块消费者* 可见，即导入模块的代码。符号默认为私有。

    ```cpp
    export int a{7};  // visible to consumer
    int b{42};        // not visible
    ```

+   你可以导出一个块，如下所示：

    ```cpp
    export {
        int a() { return 7; };     // visible 
        int b() { return 42; };    // also visible
    }
    ```

+   你可以导出一个命名空间：

    ```cpp
    export namespace bw {  // all of the bw namespace is visible
        template<typename T>
        T add(T lhs, T rhs) {  // visible as bw::add()
            return lhs + rhs;
        }
    }
    ```

+   或者，你可以从命名空间中导出单个符号：

    ```cpp
    namespace bw {  // all of the bw namespace is visible
        export template<typename T>
        T add(T lhs, T rhs) {  // visible as bw::add()
            return lhs + rhs;
        }
    }
    ```

+   一个 `import` 声明将模块导入到 *消费者* 中：

    ```cpp
    import bw_math;
    int main() {
        double f = bw::add(1.23, 4.56);
        int i = bw::add(7, 42);
        string s = bw::add<string>("one ", "two");
    }
    ```

+   你甚至可以导入一个模块并将其导出给消费者以传递：

    ```cpp
    export module bw_math;
    export import std.core;
    ```

`export` 关键字必须位于 `import` 关键字之前。

`std.core` 模块现在可供消费者使用：

```cpp
import bw_math;
using std::cout, std::string, std::format;
int main() {
    double f = bw::add(1.23, 4.56);
    int i = bw::add(7, 42);
    string s = bw::add<string>("one ", "two");
    cout << 
        format("double {} \n", f) <<
        format("int {} \n", i) <<
        format("string {} \n", s);
}
```

正如你所见，模块是相对于头文件的一个简单、直接的选择。我知道我们中的许多人都在期待模块的广泛可用性。我认为这将大大减少我们对头文件的依赖。

注意

在撰写本文时，模块的唯一完整实现是在 MSVC 的 *预览发布* 中。模块文件扩展名（`.ixx`）可能因其他编译器而异。此外，合并的 `std.core` 模块是 MSVC 在此版本中实现 STL 作为模块的一部分。其他编译器可能不会使用此约定。当完全符合的实现发布时，一些细节可能会发生变化。

在示例文件中，我包含了基于 `format` 的 `print()` 函数的模块版本。这适用于当前 MSVC 的预览发布版。一旦其他系统支持足够的模块规范，可能需要一些小的修改才能在其他系统上运行。

# 使用范围创建容器中的视图

新的 `ranges` 库是 C++20 中更重要的添加之一。它为过滤和处理容器提供了一种新的范式。范围提供了干净、直观的构建块，以实现更有效和可读的代码。

让我们先定义一些术语：

+   一个 `begin()` 和 `end()` 迭代器是一个范围。这包括大多数 STL 容器。

+   一个 **视图** 是一个转换另一个底层范围的范围。视图是惰性的，意味着它们只在迭代时操作。视图从底层范围返回数据，并不拥有任何数据。视图以 *O(1)* 常数时间操作。

+   一个 `|` 操作符。

    注意

    `<ranges>` 库使用 `std::ranges` 和 `std::ranges::view` 命名空间。认识到这很繁琐，标准包括了一个对 `std::ranges::view` 的别名，简单地称为 `std::view`。我仍然觉得这很繁琐。对于这个配方，我将使用以下别名，以节省空间，因为我认为它更优雅：

    `namespace ranges = std::ranges;  // 省去手指的麻烦！`

    `namespace views = std::ranges::views;  `

    这适用于本配方中的所有代码。

## 如何实现...

`ranges` 和 `views` 类位于 `<ranges>` 头文件中。让我们看看如何使用它们：

+   一个 *视图* 应用到一个 *范围* 上，如下所示：

    ```cpp
    const vector<int> nums{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    auto result = ranges::take_view(nums, 5);
    for (auto v: result) cout << v << " ";
    ```

输出：

```cpp
1 2 3 4 5 
```

`ranges::take_view(range, n)` 是一个返回前 *n* 个元素的视图。

你也可以使用 `take_view()` 的 *视图适配器* 版本：

```cpp
auto result = nums | views::take(5);
for (auto v: result) cout << v << " ";
```

输出：

```cpp
1 2 3 4 5 
```

*视图适配器* 位于 `std::ranges::views` 命名空间中。一个 *视图适配器* 从 `|` 操作符的左侧获取 *范围操作数*，就像 `iostreams` 使用 `<<` 操作符的 `iostreams` 一样。`|` 操作数从左到右评估。

+   因为视图适配器是 *可迭代的*，它也符合 *范围* 的资格。这使得它们可以串联使用，如下所示：

    ```cpp
    const vector<int> nums{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    auto result = nums | views::take(5) | 
       views::reverse;
    ```

输出：

```cpp
5 4 3 2 1
```

+   `filter()` 视图使用谓词函数：

    ```cpp
    auto result = nums | 
        views::filter([](int i){ return 0 == i % 2; });
    ```

输出：

```cpp
2 4 6 8 10
```

+   `transform()` 视图使用转换函数：

    ```cpp
    auto result = nums | 
        views::transform([](int i){ return i * i; });
    ```

输出：

```cpp
1 4 9 16 25 36 49 64 81 100
```

+   当然，这些视图和适配器适用于任何类型的范围：

    ```cpp
    cosnt vector<string>
    words{ "one", "two", "three", "four", "five" };
    auto result = words | views::reverse;
    ```

输出：

```cpp
five four three two one
```

+   `ranges` 库还包括一些 *范围生成器*。`iota` 生成器将生成一个递增的值序列：

    ```cpp
    auto rnums = views::iota(1, 10);
    ```

输出：

```cpp
1 2 3 4 5 6 7 8 9
```

`iota(value, bound)` 函数生成一个从 `value` 开始，在 *bound* 之前结束的序列。如果省略 `bound`，则序列是无限的：

```cpp
auto rnums = views::iota(1) | views::take(200);
```

输出：

```cpp
1 2 3 4 5 6 7 8 9 10 11 12 […] 196 197 198 199 200
```

*范围*、*视图* 和 *视图适配器* 非常灵活且有用。让我们更深入地了解它们，以便更好地理解。

## 它是如何工作的…

为了满足 *范围* 的基本要求，一个对象必须至少有两个迭代器，`begin()` 和 `end()`，其中 `end()` 迭代器是一个哨兵，用于确定范围的终点。大多数 STL 容器都符合 *范围* 的资格，包括 `string`、`vector`、`array`、`map` 等，但容器适配器（如 `stack` 和 `queue`）除外，它们没有 `begin` 和 `end` 迭代器。

*视图* 是一个操作范围并返回修改后范围的对象。视图是惰性操作的，不包含自己的数据。它不是保留底层数据的副本，而是根据需要简单地返回指向底层元素的迭代器。让我们看看这个代码片段：

```cpp
vector<int> vi { 0, 1, 2, 3, 4, 5 };
ranges::take_view tv{vi, 2};
for(int i : tv) {
    cout << i << " ";
}
cout << "\n";
```

输出：

```cpp
0 1
```

在这个例子中，`take_view` 对象接受两个参数，一个 *范围*（在这种情况下，一个 `vector<int>` 对象），和一个 *计数*。结果是包含 `vector` 中前 *count* 个对象的 *视图*。在评估时间，在 `for` 循环迭代期间，`take_view` 对象只需返回指向 `vector` 对象元素的迭代器，按需返回。在这个过程中，`vector` 对象不会被修改。

`ranges` 命名空间中的许多视图都有 `views` 命名空间中的相应的 *范围适配器*。这些适配器可以用作 *按位或* (`|`) 操作符，就像管道一样，如下所示：

```cpp
vector<int> vi { 0, 1, 2, 3, 4, 5 };
auto tview = vi | views::take(2);
for(int i : tview) {
    cout << i << " ";
}
cout << "\n";
```

输出：

```cpp
0 1
```

如预期，`|` 操作符从左到右评估。因为范围适配器的结果是另一个范围，所以这些适配器表达式可以链式使用：

```cpp
vector<int> vi { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
auto tview = vi | views::reverse | views::take(5);
for(int i : tview) {
    cout << i << " ";
}
cout << "\n";
```

输出：

```cpp
9 8 7 6 5
```

该库包括一个用于与 *谓词* 一起使用的 `filter` 视图，用于定义简单的过滤器：

```cpp
vector<int> vi { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
auto even = [](long i) { return 0 == i % 2; };
auto tview = vi | views::filter(even);
```

输出：

```cpp
0 2 4 6 8
```

还包括一个用于与 *转换函数* 一起使用的 `transform` 视图：

```cpp
vector<int> vi { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
auto even = [](int i) { return 0 == i % 2; };
auto x2 = [](auto i) { return i * 2; };
auto tview = vi | views::filter(even) | views::transform(x2);
```

输出：

```cpp
0 4 8 12 16
```

该库中有许多有用的视图和视图适配器。请查看您喜欢的参考网站，或 ([`j.bw.org/ranges`](https://j.bw.org/ranges)) 以获取完整列表。

## 还有更多…

从 C++20 开始，`<algorithm>` 头文件中的大多数算法都包括用于与 `ranges` 一起使用的版本。这些版本仍然在 `<algorithm>` 头文件中，但在 `std::ranges` 命名空间中。这使它们与旧算法区分开来。

这意味着，你不需要调用一个算法并传递两个迭代器：

```cpp
sort(v.begin(), v.end());
```

你现在可以用一个范围来调用它，就像这样：

```cpp
ranges::sort(v);
```

这确实更方便，但它实际上是如何帮助我们的呢？

考虑这样一个情况，你想要对一个向量的部分进行排序，你可以像这样使用旧的方法：

```cpp
sort(v.begin() + 5, v.end());
```

这将按顺序对向量中第一个 5 个元素之后的元素进行排序。使用 `ranges` 版本，你可以使用一个视图来跳过前 5 个元素：

```cpp
ranges::sort(views::drop(v, 5));
```

你甚至可以将视图组合起来：

```cpp
ranges::sort(views::drop(views::reverse(v), 5));
```

事实上，你甚至可以使用范围适配器作为 `ranges::sort` 的参数：

```cpp
ranges::sort(v | views::reverse | views::drop(5));
```

与此相反，如果你想要使用传统的 `sort` 算法和向量迭代器来完成这个任务，它看起来可能像这样：

```cpp
sort(v.rbegin() + 5, v.rend());
```

虽然这确实更短，并且并不难理解，但我发现范围适配器版本更加直观。

你可以在 *cppreference* 网站上找到一个完整的算法列表，这些算法被限制只能与范围一起工作（https://j.bw.org/algoranges）。

在这个菜谱中，我们仅仅只是触及了 *Ranges* 和 *Views* 的表面。这个特性是许多不同团队超过十年工作的结晶，我预计它将从根本上改变我们在 STL 中使用容器的方式。
