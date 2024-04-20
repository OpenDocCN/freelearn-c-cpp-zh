# 第二十章：标准语言支持和提案

在本书中，我们已经涉及了许多主题，现在是时候将它们全部归纳到一个方便的章节中，以帮助您记住我们涵盖的函数式编程技术的使用方法。我们将利用这个机会来看看 C++ 20 标准，并提及我们如何在我们的代码中使用这些新功能。

本章将涵盖以下主题：

+   C++中编写纯函数的支持方式和未来提案

+   C++中编写 lambda 的支持方式和未来提案

+   C++中柯里化的支持方式和未来提案

+   C++中函数组合的支持方式和未来提案

# 技术要求

您将需要一个支持 C++ 17 的编译器；我使用的是 GCC 7.4.0c。

代码在 GitHub 上的[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)的`Chapter16`文件夹中。它包括并使用`doctest`，这是一个单头开源单元测试库。您可以在 GitHub 存储库中找到它：[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# 标准语言支持和提案

到目前为止，我们已经探讨了在 C++中以函数式风格编写代码的几种方式。现在，我们将看看 C++ 17 标准允许的一些额外选项，以及 C++ 20 允许的一些选项。因此，让我们开始编写纯函数。

# 纯函数

纯函数是在接收相同输入时返回相同输出的函数。它们的可预测性使它们对于理解编写的代码与其运行时性能的相关性非常有用。

我们在第二章中发现，要在 C++中编写纯函数，需要结合`const`和`static`，具体取决于函数是类的一部分还是自由函数，并且取决于我们如何将参数传递给函数。为了方便起见，我将在此重述我们在纯函数语法上的结论：

+   类函数，按值传递：

+   `static int increment(const int value)`

+   `int increment(const int value) const`

+   类函数，按引用传递：

+   `static int increment(const int& value)`

+   `int increment(const int&value) const`

+   类函数，按值传递指针：

+   `static const int* increment(const int* value)`

+   `const int* increment(const int* value) const`

+   类函数，按引用传递指针：

+   `static const int* increment(const int* const& value)`

+   `const int* increment(const int* const& value) const`

+   独立函数，按值传递`int increment(const int value)`

+   独立函数，按引用传递`int increment(const int& value)`

+   独立函数，按指针传递值`const int* increment(const int* value)`

+   独立函数，按引用传递指针`const int* increment(const int* const& value)`

我们还发现，虽然编译器有助于减少副作用，但它并不总是告诉我们一个函数是纯的还是不纯的。在编写纯函数时，我们始终需要记住使用这三个标准，并小心应用它们：

+   它总是为相同的输入值返回相同的输出值。

+   它没有副作用。

+   它不会改变其参数值。

# Lambda 表达式

Lambda 是函数式编程的基本部分，允许我们对函数进行操作。C++自 C++11 以来就有 lambda，但最近对语法进行了一些添加。此外，我们将探讨一些 lambda 功能，在本书中我们还没有使用过，但对您自己的代码可能会有用。

让我们从一个简单的 lambda 开始——`increment`有一个输入并返回增加后的值：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](auto value) { return value + 1;};

    CHECK_EQ(2, increment(1));
}
```

方括号（`[]`）指定了捕获值的列表，我们将在以下代码中看到。我们可以以与任何函数相同的方式指定参数的类型：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](int value) { return value + 1;};

    CHECK_EQ(2, increment(1));
}
```

我们还可以在参数列表后立即指定返回值，并加上`->`符号：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](int value) -> int { return value + 1;};

    CHECK_EQ(2, increment(1));
}
```

如果没有输入值，参数列表和圆括号`()`可以被忽略：

```cpp
TEST_CASE("One"){
    auto one =  []{ return 1;};

    CHECK_EQ(1, one());
}
```

通过指定名称来捕获一个值，这样它就会被复制：

```cpp
TEST_CASE("Capture value"){
    int value = 5;
    auto addToValue =  value { return value + toAdd;};

    CHECK_EQ(6, addToValue(1));
}
```

或者，我们可以通过引用捕获一个值，使用捕获说明中的`&`运算符：

```cpp
TEST_CASE("Capture value by reference"){
    int value = 5;
    auto addToValue =  &value { return value + toAdd;};

    CHECK_EQ(6, addToValue(1));
}
```

如果我们捕获多个值，我们可以枚举它们，也可以捕获所有值。对于按值捕获，我们使用`=`说明符：

```cpp
TEST_CASE("Capture all values by value"){
    int first = 5;
    int second = 10;
    auto addToValues = = { return first + second + 
        toAdd;};
    CHECK_EQ(16, addToValues(1));
}
```

要通过引用捕获所有值，我们使用`&`说明符而不带任何变量名：

```cpp
TEST_CASE("Capture all values by reference"){
    int first = 5;
    int second = 10;
    auto addToValues = & { return first + second + 
        toAdd;};
    CHECK_EQ(16, addToValues(1));
}
```

虽然不推荐，但我们可以在参数列表后使用`mutable`说明符使 lambda 调用可变：

```cpp
TEST_CASE("Increment mutable - NOT RECOMMENDED"){
    auto increment =  [](int& value) mutable { return ++value;};

    int value = 1;
    CHECK_EQ(2, increment(value));
    CHECK_EQ(2, value);
}

```

此外，从 C++ 20 开始，我们可以指定函数调用为`consteval`，而不是默认的`constexpr`：

```cpp
TEST_CASE("Increment"){
    auto one = []() consteval { return 1;};

    CHECK_EQ(1, one());
}
```

不幸的是，这种用法在 g++8 中尚不受支持。

异常说明也是可能的；也就是说，如果 lambda 没有抛出异常，那么`noexcept`可能会派上用场：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](int value) noexcept { return value + 1;};

    CHECK_EQ(2, increment(1));
}

```

如果 lambda 抛出异常，可以指定为通用或特定：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](int value) throw() { return value + 1;};

    CHECK_EQ(2, increment(1));
}
```

但是，如果您想使用通用类型怎么办？在 C++ 11 中，您可以使用`function<>`类型。从 C++ 20 开始，所有类型约束的好处都可以以一种简洁的语法用于 lambda。

```cpp
TEST_CASE("Increment"){
    auto increment =  [] <typename T>(T value) -> requires 
        NumericType<T> { return value + 1;};

    CHECK_EQ(2, increment(1));
}
```

不幸的是，这在 g++8 中也尚不受支持。

# 部分应用和柯里化

**部分应用**意味着通过在`1`（或更多，但少于*N*）个参数上应用具有*N*个参数的函数来获得一个新函数。

我们可以通过实现一个传递参数的函数或 lambda 来手动实现部分应用。以下是使用`std::plus`函数实现部分应用以获得一个`increment`函数的例子，将其中一个参数设置为`1`：

```cpp
TEST_CASE("Increment"){
    auto increment =  [](const int value) { return plus<int>()(value, 
        1); };

    CHECK_EQ(2, increment(1));
}
```

在本书中，我们主要关注了如何在这些情况下使用 lambda；然而值得一提的是，我们也可以使用纯函数来实现相同的目标。例如，相同的增量函数可以编写为普通的 C++函数：

```cpp
namespace Increment{
    int increment(const int value){
        return plus<int>()(value, 1);
    };
}

TEST_CASE("Increment"){
    CHECK_EQ(2, Increment::increment(1));
}
```

在 C++中可以使用`bind()`函数进行部分应用。`bind()`函数允许我们为函数绑定参数值，从而可以从`plus`派生出`increment`函数，如下所示：

```cpp
TEST_CASE("Increment"){
    auto increment = bind(plus<int>(), _1, 1);

    CHECK_EQ(2, increment(1));
}
```

`bind`接受以下参数：

+   我们想要绑定的函数。

+   要绑定到的参数；这些可以是值或占位符（如`_1`、`_2`等）。占位符允许将参数转发到最终函数。

在纯函数式编程语言中，部分应用与柯里化相关联。**柯里化**是将接受*N*个参数的函数分解为接受一个参数的*N*个函数。在 C++中没有标准的柯里化函数，但我们可以通过使用 lambda 来实现。让我们看一个柯里化`pow`函数的例子：

```cpp
auto curriedPower = [](const int base) {
    return base {
        return pow(base, exponent);
    };
};

TEST_CASE("Power and curried power"){
    CHECK_EQ(16, pow(2, 4));
    CHECK_EQ(16, curriedPower(2)(4));
}
```

如您所见，借助柯里化的帮助，我们可以通过只使用一个参数调用柯里化函数来自然地进行部分应用，而不是两个参数：

```cpp
    auto powerOf2 = curriedPower(2);
    CHECK_EQ(16, powerOf2(4));
```

这种机制在许多纯函数式编程语言中默认启用。然而，在 C++中更难实现。C++中没有标准支持柯里化，但我们可以创建自己的`curry`函数，该函数接受现有函数并返回其柯里化形式。以下是一个具有两个参数的通用`curry`函数的示例：

```cpp
template<typename F>
auto curry2(F f){
    return ={
        return ={
            return f(first, second);
        };
    };
}
```

此外，以下是如何使用它进行柯里化和部分应用：

```cpp
TEST_CASE("Power and curried power"){
    auto power = [](const int base, const int exponent){
        return pow(base, exponent);
    };
    auto curriedPower = curry2(power);
    auto powerOf2 = curriedPower(2);
    CHECK_EQ(16, powerOf2(4));
}
```

现在让我们看看实现函数组合的方法。

# 函数组合

函数组合意味着取两个函数*f*和*g*，并获得一个新函数*h*；对于任何值，*h(x) = f(g(x))*。我们可以手动实现函数组合，无论是在 lambda 中还是在普通函数中。例如，给定两个函数，`powerOf2`计算`2`的幂，`increment`增加一个值，我们将看到以下结果：

```cpp
auto powerOf2 = [](const int exponent){
    return pow(2, exponent);
};

auto increment = [](const int value){
    return value + 1;
};
```

我们可以通过简单地将调用封装到一个名为`incrementPowerOf2`的 lambda 中来组合它们：

```cpp
TEST_CASE("Composition"){
    auto incrementPowerOf2 = [](const int exponent){
        return increment(powerOf2(exponent));
    };

    CHECK_EQ(9, incrementPowerOf2(3));
}
```

或者，我们可以简单地使用一个简单的函数，如下所示：

```cpp
namespace Functions{
    int incrementPowerOf2(const int exponent){
        return increment(powerOf2(exponent));
    };
}

TEST_CASE("Composition"){
    CHECK_EQ(9, Functions::incrementPowerOf2(3));
}
```

然而，一个接受两个函数并返回组合函数的运算符非常方便，在许多编程语言中都有实现。在 C++中最接近函数组合运算符的是`|`管道运算符，它来自于 ranges 库，目前已经包含在 C++ 20 标准中。然而，虽然它实现了组合，但对于一般函数或 lambda 并不适用。幸运的是，C++是一种强大的语言，我们可以编写自己的 compose 函数，正如我们在第四章中发现的，*函数组合的概念*。

```cpp
template <class F, class G>
auto compose(F f, G g){
    return ={return f(g(value));};
}

TEST_CASE("Composition"){
    auto incrementPowerOf2 = compose(increment, powerOf2); 

    CHECK_EQ(9, incrementPowerOf2(3));
}
```

回到 ranges 库和管道运算符，我们可以在 ranges 的上下文中使用这种形式的函数组合。我们在第十四章中对这个主题进行了广泛探讨，*使用 ranges 库进行惰性求值*，这里有一个使用管道运算符计算集合中既是`2`的倍数又是`3`的倍数的所有数字的和的例子：

```cpp
auto isEven = [](const auto number){
    return number % 2 == 0;
};

auto isMultipleOf3 = [](const auto number){
    return number % 3 == 0;
};

auto sumOfMultiplesOf6 = [](const auto& numbers){
    return ranges::accumulate(
            numbers | ranges::view::filter(isEven) | 
                ranges::view::filter(isMultipleOf3), 0);
};

TEST_CASE("Sum of even numbers and of multiples of 6"){
    list<int> numbers{1, 2, 5, 6, 10, 12, 17, 25};

    CHECK_EQ(18, sumOfMultiplesOf6(numbers));
}
```

正如你所看到的，在标准 C++中有多种函数式编程的选项，而且 C++ 20 中还有一些令人兴奋的发展。

# 总结

这就是了！我们已经快速概述了函数式编程中最重要的操作，以及我们如何可以使用 C++ 17 和 C++ 20 来实现它们。我相信你现在掌握了更多工具，包括纯函数、lambda、部分应用、柯里化和函数组合，仅举几例。

从现在开始，你可以自行选择如何使用它们。选择一些，或者组合它们，或者慢慢将你的代码从可变状态转移到不可变状态；掌握这些工具将使你在编写代码的方式上拥有更多选择和灵活性。

无论你选择做什么，我祝你在你的项目和编程生涯中好运。愉快编码！
