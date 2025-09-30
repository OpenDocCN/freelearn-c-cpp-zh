# *第五章*: Lambda 表达式

C++11 标准引入了 *lambda 表达式*（有时称为 *lambda 函数*，或简称 *lambda*）。这个特性允许在表达式的上下文中使用匿名函数。Lambda 可以在函数调用、容器、变量和其他表达式上下文中使用。这可能听起来无害，但它非常有用。

让我们从 lambda 表达式的简要回顾开始。

# Lambda 表达式

Lambda 实质上是一个字面量表达式作为匿名函数：

```cpp
auto la = []{ return "Hello\n"; };
```

变量 `la` 现在可以像函数一样使用：

```cpp
cout << la();
```

它可以被传递给另一个函数：

```cpp
f(la);
```

它可以被传递给另一个 lambda：

```cpp
const auto la = []{ return "Hello\n"; };
const auto lb = [](auto a){ return a(); };
cout << lb(la);
```

输出：

```cpp
Hello
```

或者它可以匿名传递（作为字面量）：

```cpp
const auto lb = [](auto a){ return a(); };
cout << lb([]{ return "Hello\n"; });
```

## 闭包

术语 *闭包* 通常应用于任何匿名函数。严格来说，闭包是一个允许在自身词法作用域之外使用符号的函数。

你可能已经注意到了 lambda 定义中的方括号：

```cpp
auto la = []{ return "Hello\n"; };
```

方括号用于指定 *捕获* 列表。捕获是在 lambda 体作用域内可访问的外部变量。如果我没有将外部变量列为捕获，我将得到编译错误：

```cpp
const char * greeting{ "Hello\n" };
const auto la = []{ return greeting; };
cout << la();
```

当我尝试使用 GCC 编译这个程序时，我得到了以下错误：

```cpp
In lambda function:
error: 'greeting' is not captured
```

这是因为 lambda 的主体有其自己的词法作用域，而 `greeting` 变量在该作用域之外。

我可以在捕获中指定 `greeting` 变量。这允许变量进入 lambda 的作用域：

```cpp
const char * greeting{ "Hello\n" };
const auto la = [greeting]{ return greeting; };
cout << la();
```

现在它按预期编译并运行：

```cpp
$ ./working
Hello
```

这种在其自身作用域之外捕获变量的能力使得 lambda 成为一个 *闭包*。人们以不同的方式使用这个术语，这没关系，只要我们能互相理解。然而，了解这个术语的含义是很好的。

Lambda 表达式使我们能够编写良好、干净的泛型代码。它们允许使用 *函数式编程* 模式，其中我们可以将 lambda 作为函数参数传递给算法，甚至传递给其他 lambda。

在本章中，我们将介绍使用 lambda 与 STL 的方法，以下是一些食谱：

+   使用 lambda 进行作用域可重用代码

+   使用 lambda 作为算法库中的谓词

+   使用 `std::function` 作为多态包装器

+   使用递归连接 lambda

+   使用逻辑合取结合谓词

+   使用相同的输入调用多个 lambda

+   使用映射 lambda 作为跳转表

# 技术要求

你可以在 GitHub 上找到本章的代码：[`github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap05`](https://github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap05)。

# 使用 lambda 进行作用域可重用代码

Lambda 表达式可以被定义并存储以供以后使用。它们可以作为参数传递，存储在数据结构中，并在不同的上下文中使用不同的参数进行调用。它们与函数一样灵活，但具有数据的移动性。

## 如何做到这一点...

让我们从一个小程序开始，我们将使用它来测试 lambda 表达式的各种配置：

+   我们首先定义一个 `main()` 函数，并使用它来实验 lambda：

    ```cpp
    int main() {
        ... // code goes here
    }
    ```

+   在 `main()` 函数内部，我们将声明几个 lambda。lambda 的基本定义需要一个对齐的方括号和花括号中的代码块：

    ```cpp
    auto one = [](){ return "one"; };
    auto two = []{ return "two"; };
    ```

注意，第一个示例 `one` 在方括号后包含括号，而第二个示例 `two` 则没有。空参数括号通常包含在内，但并非总是必需的。返回类型由编译器推断。

+   我可以用 `cout` 调用这些函数，或者用 `format`，或者在任何接受 C-字符串的上下文中：

    ```cpp
    cout << one() << '\n';
    cout << format("{}\n", two());
    ```

+   在许多情况下，编译器可以从 *自动类型推导* 中确定返回类型。否则，您可以使用 `->` 运算符指定返回类型：

    ```cpp
    auto one = []() -> const char * { return "one"; };
    auto two = []() -> auto { return "two"; };
    ```

Lambdas 使用 *尾随返回类型* 语法。这由 `->` 运算符后跟类型指定组成。如果没有指定返回类型，则被认为是 `auto`。如果您使用尾随返回类型，*则必须包含参数括号*。

+   让我们定义一个 lambda 来打印出其他 lambda 的值：

    ```cpp
    auto p = [](auto v) { cout << v() << '\n'; };
    ```

`p()` lambda 期望一个 lambda（或函数）作为其参数 `v`，并在其函数体中调用它。

`auto` 类型参数使这个 lambda 成为 *缩写模板*。在 C++20 之前，这是模板化 lambda 的唯一方法。从 C++20 开始，您可以在捕获括号之后指定模板参数（无需 `template` 关键字）。这与模板参数等价：

```cpp
auto p = []<template T>(T v) { cout << v() << '\n'; };
```

简化的 `auto` 版本更简单且更常见。它适用于大多数目的。

+   现在，我们可以在函数调用中传递一个匿名 lambda：

    ```cpp
    p([]{ return "lambda call lambda"; });
    ```

输出如下：

```cpp
lambda call lambda
```

+   如果我们需要向匿名 lambda 传递参数，我们可以在 lambda 表达式之后放置括号：

    ```cpp
    << [](auto l, auto r){ return l + r; }(47, 73)
        << '\n';
    ```

函数参数 `47` 和 `73` 被传递到函数体后面的括号中的匿名 lambda。

+   您可以通过将它们作为 *捕获* 包含在方括号中来访问 lambda 外部的变量：

    ```cpp
    int num{1};
    p([num]{ return num; });
    ```

+   或者您可以通过引用捕获它们：

    ```cpp
    int num{0};
    auto inc = [&num]{ num++; };
    for (size_t i{0}; i < 5; ++i) {
        inc();
    }
    cout << num << '\n';
    ```

输出如下：

```cpp
5
```

这允许您修改捕获的变量。

+   您还可以定义一个局部捕获变量以保持其状态：

    ```cpp
    auto counter = [n = 0]() mutable { return ++n; };
    for (size_t i{0}; i < 5; ++i) {
        cout << format("{}, ", counter());
    }
    cout << '\n';
    ```

输出：

```cpp
1, 2, 3, 4, 5,
```

`mutable` 指定符允许 lambda 修改其捕获。lambda 默认为 `const`-qualified。

与尾随返回类型一样，任何 *指定符* 都需要参数括号。

+   lambda 支持两种类型的 *默认捕获*：

    ```cpp
    int a = 47;
    int b = 73;
    auto l1 = []{ return a + b; };
    ```

如果我尝试编译此代码，我会得到一个包含以下错误的信息：

```cpp
note: the lambda has no capture-default
```

一种默认捕获类型由等号表示：

```cpp
auto l1 = [=]{ return a + b; };
```

这将捕获 lambda 范围内的所有符号。等号执行 *复制捕获*。它将捕获对象的副本，就像使用赋值运算符复制一样。

另一个默认捕获使用 & 符号进行 *引用捕获*：

```cpp
auto l1 = [&]{ return a + b; };
```

这是一个默认捕获，通过引用捕获。

默认情况下，捕获只使用符号在它们被引用时，所以它们并不像看起来那么混乱。话虽如此，我建议尽可能使用显式捕获，因为它们通常可以提高可读性。

## 它是如何工作的…

lambda 表达式的语法如下：

![图 5.1 – lambda 表达式的语法![img/B18267_05_01.jpg](img/B18267_05_01.jpg)

图 5.1 – lambda 表达式的语法

lambda 表达式的唯一必需部分是捕获列表和主体，主体可以是空的：

```cpp
[]{}
```

这是最小的 lambda 表达式。它不捕获任何内容也不做任何事情。

让我们考虑每个部分。

### 捕获列表

*捕获列表*指定了我们捕获的内容，如果有的话。它不能被省略，但它可以是空的。我们可以在 lambda 的作用域内使用 `[=]` 来捕获所有变量 *通过复制* 或 `[&]` 来捕获所有变量 *通过引用*。

您可以通过在括号中列出它们来捕获单个变量：

```cpp
[a, b]{ return a + b; }
```

指定的捕获默认为复制。您可以使用引用运算符来通过引用捕获：

```cpp
[&a, &b]{ return a + b; }
```

当您通过引用捕获时，您可以修改引用的变量。

注意

您不能直接捕获对象成员。您可以通过列出它们来捕获单个变量 `this` 或 `*this` 以解引用类成员。

### 参数

与函数一样，参数在括号中指定：

```cpp
[](int a, int b){ return a + b };
```

如果没有参数、指定符或尾随返回类型，则括号是可选的。指定符或尾随返回类型使括号成为必需：

```cpp
[]() -> int { return 47 + 73 };
```

### `mutable` 修饰符（可选）

lambda 表达式默认为 `const`-qualified，除非您指定 `mutable` 修饰符。这允许它在 `const` 上下文中使用，但也意味着它不能修改任何通过复制捕获的变量。例如：

```cpp
[a]{ return ++a; };
```

这将无法编译，并显示如下错误信息：

```cpp
In lambda function:
error: increment of read-only variable 'a'
```

使用 `mutable` 修饰符后，lambda 就不再具有 `const`-qualified，捕获的变量可以被更改：

```cpp
[a]() mutable { return ++a; };
```

### `constexpr` 指定符（可选）

您可以使用 `constexpr` 显式指定您希望 lambda 被视为 *常量表达式*。这意味着它可以在编译时评估。如果 lambda 满足要求，即使没有指定符，它也可以被视为 `constexpr`。

### 异常属性（可选）

您可以使用 `noexcept` 指定符来声明您的 lambda 不会抛出任何异常。

### 尾随返回类型（可选）

默认情况下，lambda 返回类型是从 `return` 语句推断的，就像它是 `auto` 返回类型一样。您可以使用 `->` 运算符可选地指定尾随返回类型：

```cpp
[](int a, int b) -> long { return a + b; };
```

如果您使用任何可选指定符或尾随返回类型，则参数括号是必需的。

注意

一些编译器，包括 GCC，允许省略空参数括号，即使存在指定符或尾随返回类型。这是不正确的。根据规范，参数、指定符和尾随返回类型都是`lambda-declarator`的一部分，并且当包含任何部分时都需要括号。这可能在 C++的未来的版本中发生变化。

# 使用算法库中的 lambda 作为谓词

`algorithm`库中的某些函数需要使用**谓词**函数。谓词是一个函数（或仿函数或 lambda），它测试一个条件并返回布尔`true`/`false`响应。

## 如何实现...

对于这个配方，我们将通过使用不同类型的谓词来实验`count_if()`算法：

+   首先，让我们创建一个用作谓词的函数。谓词接受一定数量的参数并返回一个`bool`。`count_if()`的谓词接受一个参数：

    ```cpp
    bool is_div4(int i) {
        return i % 4 == 0;
    }
    ```

这个谓词检查一个`int`值是否可以被 4 整除。

+   在`main()`函数中，我们将定义一个`int`值的向量，并使用它通过`count_if()`测试我们的谓词函数：

    ```cpp
    int main() {
        const vector<int> v{ 1, 7, 4, 9, 4, 8, 12, 10, 20 };
        int count = count_if(v.begin(), v.end(), is_div4);
        cout << format("numbers divisible by 4: {}\n", 
          count);
    }
    ```

输出如下：

```cpp
numbers divisible by 4: 5
```

（可被 5 整除的数字有：4，4，8，12 和 20。）

`count_if()`算法使用谓词函数来确定要计数的序列中的哪些元素。它将每个元素作为参数调用谓词，并且只有当谓词返回`true`时才计数元素。

在这种情况下，我们使用了一个函数作为谓词。

+   我们也可以使用**仿函数**作为谓词：

    ```cpp
    struct is_div4 {
        bool operator()(int i) {
            return i % 4 == 0;
        }
    };
    ```

这里的唯一变化是我们需要使用类的**一个实例**作为谓词：

```cpp
int count = count_if(v.begin(), v.end(), is_div4());
```

仿函数的优势在于它可以携带上下文并访问类和实例变量。这是在 C++11 引入 lambda 表达式之前使用谓词的常见方式。

+   使用 lambda 表达式，我们拥有了两种世界的最佳之处：函数的简洁性和仿函数的强大功能。我们可以将 lambda 用作变量：

    ```cpp
    auto is_div4 = [](int i){ return i % 4 == 0; };
    int count = count_if(v.begin(), v.end(), is_div4);
    ```

或者我们可以使用匿名 lambda：

```cpp
int count = count_if(v.begin(), v.end(), 
    [](int i){ return i % 4 == 0; });
```

+   我们可以利用 lambda 捕获，通过将 lambda 包装在函数中来利用它，并使用该函数上下文产生具有不同参数的相同 lambda：

    ```cpp
    auto is_div_by(int divisor) {
        return divisor{ return i % divisor == 0; };
    }
    ```

这个函数返回一个带有捕获上下文中除数的谓词 lambda。

我们然后可以使用该谓词与`count_if()`一起使用：

```cpp
for( int i : { 3, 4, 5 } ) {
    auto pred = is_div_by(i);
    int count = count_if(v.begin(), v.end(), pred);
    cout << format("numbers divisible by {}: {}\n", i,
      count);
}
```

每次调用`is_div_by()`都会返回一个带有从`i`的不同除数的谓词。现在我们得到以下输出：

```cpp
numbers divisible by 3: 2
numbers divisible by 4: 5
numbers divisible by 5: 2
```

## 它是如何工作的...

函数指针的类型表示为一个指针后跟函数调用`()`运算符：

```cpp
void (*)()
```

你可以声明一个函数指针并用现有函数的名称初始化它：

```cpp
void (*fp)() = func;
```

一旦声明，函数指针可以被解引用并像函数本身一样使用：

```cpp
func();  // do the func thing
```

lambda 表达式与函数指针具有相同的类型：

```cpp
void (*fp)() = []{ cout << "foo\n"; };
```

这意味着无论你在哪里使用具有特定签名的函数指针，你也可以使用具有相同签名的 lambda 表达式。这允许函数指针、仿函数和 lambda 表达式可以互换使用：

```cpp
bool (*fp)(int) = is_div4;
bool (*fp)(int) = [](int i){ return i % 4 == 0; };
```

由于这种可互换性，像 `count_if()` 这样的算法接受一个函数、仿函数或 lambda，其中它期望一个具有特定函数签名的谓词。

这适用于任何使用谓词的算法。

# 使用 std::function 作为多态包装器

类模板 `std::function` 是函数的一个薄薄的多态包装器。它可以存储、复制和调用任何函数、lambda 表达式或其他函数对象。在您希望存储函数或 lambda 引用的地方，它可能很有用。使用 `std::function` 允许您在同一个容器中存储具有不同签名的函数和 lambda，并保持 lambda 捕获的上下文。

## 如何做到这一点…

这个配方使用 `std::function` 类将 lambda 的不同特化存储在 `vector` 中：

+   这个配方包含在 `main()` 函数中，我们首先声明三个不同类型的容器：

    ```cpp
    int main() {
        deque<int> d;
        list<int> l;
        vector<int> v;
    ```

这些容器，`deque`、`list` 和 `vector`，将被模板 lambda 引用。

+   我们将声明一个简单的 `print_c` lambda 函数来打印容器：

    ```cpp
    auto print_c = [](auto& c) {
        for(auto i : c) cout << format("{} ", i);
        cout << '\n';
    };
    ```

+   现在我们声明一个返回匿名 lambda 的 lambda：

    ```cpp
    auto push_c = [](auto& container) {
        return &container {
            container.push_back(value);
        };
    };
    ```

`push_c` lambda 接收一个容器的引用，该容器被匿名 lambda 所捕获。匿名 lambda 调用捕获容器的 `push_back()` 成员函数。`push_c` 的返回值是匿名 lambda。

+   现在我们声明一个 `std::function` 元素的 `vector`，并用三个 `push_c()` 实例填充它：

    ```cpp
    const vector<std::function<void(int)>> 
        consumers { push_c(d), push_c(l), push_c(v) };
    ```

初始化列表中的每个元素都是对 `push_c` lambda 的函数调用。`push_c` 返回匿名 lambda 的一个实例，该实例通过 `function` 包装器存储在 `vector` 中。`push_c` lambda 使用三个容器 `d`、`l` 和 `v` 被调用。容器作为捕获传递给匿名 lambda。

+   现在我们遍历 `consumers` 向量，并对每个 lambda 元素调用 10 次，将整数 0–9 分别填充到每个容器中：

    ```cpp
    for(auto &consume : consumers) {
        for (int i{0}; i < 10; ++i) {
            consume(i);
        }
    }
    ```

+   现在我们的三个容器，`deque`、`list` 和 `vector`，都应该填充了整数。让我们将它们打印出来：

    ```cpp
    print_c(d);
    print_c(l);
    print_c(v);
    ```

我们应该得到的结果是：

```cpp
0 1 2 3 4 5 6 7 8 9
0 1 2 3 4 5 6 7 8 9
0 1 2 3 4 5 6 7 8 9
```

## 它是如何工作的…

Lambda 经常与间接引用一起使用，这个配方是这种用法的一个很好的例子。例如，`push_c` lambda 返回一个匿名 lambda：

```cpp
auto push_c = [](auto& container) {
    return &container {
        container.push_back(value);
    };
};
```

这个匿名 lambda 是存储在 `vector` 中的那个：

```cpp
const vector<std::function<void(int)>> 
    consumers { push_c(d), push_c(l), push_c(v) };
```

这是 `consumers` 容器的定义。它初始化了三个元素，其中每个元素都是通过调用 `push_c` 来初始化的，它返回一个匿名 lambda。存储在向量中的是匿名 lambda，而不是 `push_c` lambda。

`vector` 定义使用 `std::function` 类作为元素的类型。`function` 构造函数接受任何可调用对象并将其引用存储为 `function` 目标：

```cpp
template< class F >
function( F&& f );
```

当其函数调用 `()` 操作符被调用时，`function` 对象会使用预期的参数调用目标函数：

```cpp
for(auto &c : consumers) {
    for (int i{0}; i < 10; ++i) {
        c(i);
    }
}
```

这会调用存储在`consumers`容器中的每个*匿名 lambda* 10 次，从而填充`d`、`l`和`v`容器。

## 还有更多...

`std::function`类的本质使其在许多方面都很有用。你可以把它想象成一个多态函数容器。它可以存储一个独立的函数：

```cpp
void hello() {
    cout << "hello\n";
}
int main() {
    function<void(void)> h = hello;
    h();
}
```

它可以存储一个成员函数，使用`std::bind`来绑定函数参数：

```cpp
struct hello {
    void greeting() const { cout << "Hello Bob\n"; }
};
int main() {
    hello bob{};
    const function<void(void)> h = 
        std::bind(&hello::greeting, &bob);
    h();
}
```

或者它可以存储任何可执行对象：

```cpp
struct hello {
    void operator()() const { cout << "Hello Bob\n"; }
};
int main() {
    const function<void(void)> h = hello();
    h();
}
```

输出如下：

```cpp
Hello Bob
```

# 使用递归连接 lambda

你可以将 lambda 堆叠起来，使得一个的输出是下一个的输入，使用一个简单的递归函数。这创建了一种简单的方法来构建一个函数在另一个函数之上。

## 如何实现...

这是一个简短且简单的配方，使用一个递归函数来完成大部分工作：

+   我们首先定义连接函数`concat()`：

    ```cpp
    template <typename T, typename ...Ts>
    auto concat(T t, Ts ...ts) {
        if constexpr (sizeof...(ts) > 0) {
            return & {
    return t(concat(ts...)(parameters...)); 
            };
        } else  {
            return t;
        }
    }
    ```

这个函数返回一个匿名 lambda，它反过来再次调用函数，直到参数包耗尽。

+   在`main()`函数中，我们创建了一些 lambda 并使用它们调用`concat()`函数：

    ```cpp
    int main() {
        auto twice = [](auto i) { return i * 2; };
        auto thrice = [](auto i) { return i * 3; };
        auto combined = concat(thrice, twice, 
          std::plus<int>{});
        std::cout << format("{}\n", combined(2, 3));
    }
    ```

`concat()`函数使用三个参数被调用：两个 lambda 和`std::plus()`函数。

当递归展开时，函数从右到左被调用，从`plus()`开始。`plus()`函数接受两个参数并返回总和。从`plus()`返回的值传递给`twice()`，然后将其返回值传递给`thrice()`。然后使用`format()`将结果打印到控制台：

```cpp
30
```

## 它是如何工作的...

`concat()`函数很简单，但由于返回 lambda 的*递归*和*间接引用*可能令人困惑：

```cpp
template <typename T, typename ...Ts>
auto concat(T t, Ts ...ts) {
    if constexpr (sizeof...(ts) > 0) {
        return & {
            return t(concat(ts...)(parameters...)); 
        };
    } else  {
        return t;
    }
}
```

`concat()`函数使用参数包被调用。使用省略号，`sizeof...`运算符返回参数包中的元素数量。这用于测试递归的结束。

`concat()`函数返回一个 lambda。这个 lambda 递归地调用`concat()`函数。因为`concat()`的第一个参数不是参数包的一部分，所以每次递归调用都会剥去包的第一个元素。

外部的`return`语句返回 lambda。内部的`return`来自 lambda。lambda 调用传递给`concat()`的函数并返回其值。

随意拆解并研究它。这个技术很有价值。

# 使用逻辑合取连接谓词

这个例子将 lambda 包装在一个函数中，以创建用于算法谓词的自定义合取。

## 如何实现...

`copy_if()`算法需要一个接受一个参数的谓词。在这个配方中，我们将从三个其他 lambda 中创建一个谓词 lambda：

+   首先，我们将编写`combine()`函数。这个函数返回一个用于与`copy_if()`算法一起使用的 lambda 表达式：

    ```cpp
    template <typename F, typename A, typename B>
    auto combine(F binary_func, A a, B b) {
        return = {
            return binary_func(a(param), b(param));
        };
    }
    ```

`combine()`函数接受三个函数参数——一个二元合取和两个谓词——并返回一个调用合取与两个谓词的 lambda。

+   在`main()`函数中，我们创建用于与`combine()`一起使用的 lambda 表达式：

    ```cpp
    int main() {
        auto begins_with = [](const string &s){
            return s.find("a") == 0;
        };
        auto ends_with = [](const string &s){
            return s.rfind("b") == s.length() - 1;
        };
        auto bool_and = [](const auto& l, const auto& r){
            return l && r;
        };
    ```

`begins_with` 和 `ends_with` lambda 是简单的过滤器谓词，分别用于查找以 `'a'` 开头和以 `'b'` 结尾的字符串。`bool_and` lambda 是合取。

+   现在我们可以使用 `combine()` 调用 `copy_if` 算法：

    ```cpp
    std::copy_if(istream_iterator<string>{cin}, {},
                 ostream_iterator<string>{cout, " "},
                 combine(bool_and, begins_with, 
    ends_with));
    cout << '\n';
    ```

`combine()` 函数返回一个 lambda，该 lambda 通过合取将两个谓词结合起来。

输出看起来如下：

```cpp
$ echo aabb bbaa foo bar abazb | ./conjunction
aabb abazb
```

## 它是如何工作的…

`std::copy_if()` 算法需要一个接受一个参数的谓词函数，但我们的合取需要两个参数，每个参数都需要一个参数。我们通过返回一个特定于该上下文的 lambda 的函数来解决这个问题：

```cpp
template <typename F, typename A, typename B>
auto combine(F binary_func, A a, B b) {
    return = {
        return binary_func(a(param), b(param));
    };
}
```

`combine()` 函数从一个函数参数创建一个 lambda，每个参数都是一个函数。返回的 lambda 接受谓词函数所需的单个参数。现在我们可以使用 `combine()` 函数调用 `copy_if()`：

```cpp
std::copy_if(istream_iterator<string>{cin}, {},
             ostream_iterator<string>{cout, " "},
             combine(bool_and, begins_with, ends_with));
```

这将组合 lambda 传递给算法，以便它可以在该上下文中操作。

# 使用相同的输入调用多个 lambda

你可以通过将 lambda 包装在函数中来轻松创建具有不同捕获值的 lambda 的多个实例。这允许你使用相同的输入调用 lambda 的不同版本。

## 如何做到这一点…

这是一个简单的例子，展示了如何使用不同类型的括号包装一个值：

+   我们首先创建包装函数 `braces()`：

    ```cpp
    auto braces (const char a, const char b) {
        return a, b {
            cout << format("{}{}{} ", a, v, b);
        };
    }
    ```

`braces()` 函数包装一个返回三个值字符串的 lambda，其中第一个和最后一个值是传递给 lambda 作为捕获的字符，中间的值作为参数传递。

+   在 `main()` 函数中，我们使用 `braces()` 创建四个 lambda，使用四组不同的括号：

    ```cpp
    auto a = braces('(', ')');
    auto b = braces('[', ']');
    auto c = braces('{', '}');
    auto d = braces('|', '|');
    ```

+   现在我们可以从简单的 `for()` 循环中调用我们的 lambda：

    ```cpp
    for( int i : { 1, 2, 3, 4, 5 } ) {
        for( auto x : { a, b, c, d } ) x(i);
        cout << '\n';
    }
    ```

这是一个嵌套的 `for()` 循环。外层循环简单地从 1 计数到 5，将整数传递给内层循环。内层循环调用带有括号的 lambda。

两个循环都使用一个 *初始化列表* 作为基于范围的 `for()` 循环中的容器。这是一种方便的技术，用于遍历一组小的值。

+   我们程序的输出如下：

    ```cpp
    (1) [1] {1} |1|
    (2) [2] {2} |2|
    (3) [3] {3} |3|
    (4) [4] {4} |4|
    (5) [5] {5} |5|
    ```

输出显示了每个整数，以及每个括号组合。

## 它是如何工作的…

这是一个如何使用 lambda 包装器的简单例子。`braces()` 函数使用传递给它的括号构建一个 lambda：

```cpp
auto braces (const char a, const char b) {
    return a, b {
        cout << format("{}{}{} ", a, v, b);
    };
}
```

通过将 `braces()` 函数的参数传递给 lambda，它可以返回一个具有该上下文的 lambda。因此，主函数中的每个赋值都携带这些参数：

```cpp
auto a = braces('(', ')');
auto b = braces('[', ']');
auto c = braces('{', '}');
auto d = braces('|', '|');
```

当这些 lambda 用数字调用时，它们将返回一个包含相应括号中该数字的字符串。

# 使用映射 lambda 作为跳转表

当你想从用户或其他输入中选择一个动作时，跳转表是一个有用的模式。跳转表通常在 `if`/`else` 或 `switch` 结构中实现。在这个菜谱中，我们将使用 STL `map` 和匿名 lambda 仅构建一个简洁的跳转表。

## 如何做到这一点…

从`map`和 lambda 构建简单的跳转表很容易。`map`提供了简单的索引导航，lambda 可以作为负载存储。下面是如何做到这一点：

+   首先，我们将创建一个简单的`prompt()`函数来从控制台获取输入：

    ```cpp
    const char prompt(const char * p) {
        std::string r;
        cout << format("{} > ", p);
        std::getline(cin, r, '\n');
        if(r.size() < 1) return '\0';
        if(r.size() > 1) {
            cout << "Response too long\n";
            return '\0';
        }
        return toupper(r[0]);
    }
    ```

C 字符串参数用作提示。调用`std::getline()`从用户那里获取输入。响应存储在`r`中，检查长度，然后如果长度为单个字符，则将其转换为大写并返回。

+   在`main()`函数中，我们声明并初始化一个 lambda 的`map`：

    ```cpp
    using jumpfunc = void(*)();
    map<const char, jumpfunc> jumpmap {
        { 'A', []{ cout << "func A\n"; } },
        { 'B', []{ cout << "func B\n"; } },
        { 'C', []{ cout << "func C\n"; } },
        { 'D', []{ cout << "func D\n"; } },
        { 'X', []{ cout << "Bye!\n"; } }
    };
    ```

`map`容器加载了用于跳转表的匿名 lambda。这些 lambda 可以轻松调用其他函数或执行简单任务。

`using`别名是为了方便。我们使用函数指针类型`void(*)()`作为 lambda 的负载。如果你更喜欢，你可以使用`std::function()`，如果你需要更多的灵活性或者觉得它更易读。它的开销非常小：

```cpp
using jumpfunc = std::function<void()>;
```

+   现在我们可以提示用户输入并从`map`中选择一个动作：

    ```cpp
    char select{};
    while(select != 'X') {
        if((select = prompt("select A/B/C/D/X"))) {
            auto it = jumpmap.find(select);
            if(it != jumpmap.end()) it->second();
            else cout << "Invalid response\n";
        }
    }
    ```

这是我们如何使用基于`map`的跳转表。我们循环直到选择`'X'`以退出。我们使用提示字符串调用`prompt()`，在`map`对象上调用`find()`，然后调用 lambda 的`it->second()`。

## 它是如何工作的…

`map`容器是一个出色的跳转表。它简洁且易于导航：

```cpp
using jumpfunc = void(*)();
map<const char, jumpfunc> jumpmap {
    { 'A', []{ cout << "func A\n"; } },
    { 'B', []{ cout << "func B\n"; } },
    { 'C', []{ cout << "func C\n"; } },
    { 'D', []{ cout << "func D\n"; } },
    { 'X', []{ cout << "Bye!\n"; } }
};
```

匿名 lambda 存储在`map`容器中作为负载。键是来自动作菜单的字符响应。

你可以在一个动作中测试键的有效性并选择一个 lambda：

```cpp
auto it = jumpmap.find(select);
if(it != jumpmap.end()) it->second();
else cout << "Invalid response\n";
```

这是一个简单、优雅的解决方案，否则我们可能会使用尴尬的分支代码。
