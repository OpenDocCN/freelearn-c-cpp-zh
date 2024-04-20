# 使用异常处理错误

在本章中，我们将学习一些高级的 C++异常处理技术。我们在这里假设您已经基本了解如何抛出和捕获 C++异常。本章不是专注于 C++异常的基础知识，而是教会您一些更高级的 C++异常处理技术。这包括正确使用`noexcept`指定符和`noexcept`运算符，以便您可以正确地标记您的 API，要么可能抛出异常，要么明确地不抛出 C++异常，而是在发生无法处理的错误时调用`std::terminate()`。

本章还将解释术语**资源获取即初始化**（**RAII**）是什么，以及它如何补充 C++异常处理。我们还将讨论为什么不应该从类的析构函数中抛出 C++异常以及如何处理这些类型的问题。最后，我们将看看如何创建自己的自定义 C++异常，包括提供一些关于创建自己的异常时要做和不要做的基本准则。

从本章提供的信息中，您将更好地了解 C++异常在底层是如何工作的，以及可以用 C++异常做哪些事情来构建更健壮和可靠的 C++程序。

本章中的配方如下：

+   使用`noexcept`指定符

+   使用`noexcept`运算符

+   使用 RAII

+   学习为什么永远不要在析构函数中抛出异常

+   轻松创建自己的异常类

# 技术要求

要编译和运行本章中的示例，您必须具有对运行 Ubuntu 18.04 的计算机的管理访问权限，并且具有功能正常的互联网连接。在运行这些示例之前，您必须安装以下内容：

```cpp
sudo apt-get install build-essential git cmake
```

如果这是安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

# 使用`noexcept`指定符

`noexcept`指定符用于告诉编译器一个函数是否可能抛出 C++异常。如果一个函数标记有`noexcept`指定符，它是不允许抛出异常的，如果抛出异常，将会调用`std::terminate()`。如果函数没有`noexcept`指定符，异常可以像平常一样被抛出。

在这个配方中，我们将探讨如何在自己的代码中使用`noexcept`指定符。这个指定符很重要，因为它是你正在创建的 API 和 API 的用户之间的一个合同。当使用`noexcept`指定符时，它告诉 API 的用户在使用 API 时不需要考虑异常。它还告诉作者，如果他们将`noexcept`指定符添加到他们的 API 中，他们必须确保不会抛出任何异常，这在某些情况下需要作者捕获所有可能的异常并处理它们，或者在无法处理异常时调用`std::terminate()`。此外，有一些操作，比如`std::move`，在这些操作中不能抛出异常，因为移动操作通常无法安全地被逆转。最后，对于一些编译器，将`noexcept`添加到你的 API 中将减少函数的总体大小，从而使应用程序的总体大小更小。

# 准备工作

开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本配方中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

要尝试这个配方，请执行以下步骤：

1.  从新的终端中运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter02
```

1.  要编译源代码，请运行以下命令：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe01_examples
```

1.  一旦源代码被编译，您可以通过运行以下命令来执行本食谱中的每个示例：

```cpp
> ./recipe01_example01
The answer is: 42

> ./recipe01_example02
terminate called after throwing an instance of 'std::runtime_error'
what(): The answer is: 42
Aborted

> ./recipe01_example03
The answer is: 42

> ./recipe01_example04
terminate called after throwing an instance of 'std::runtime_error'
what(): The answer is: 42
Aborted

> ./recipe01_example05
foo: 18446744069414584320
foo: T is too large
```

在下一节中，我们将逐个介绍这些例子，并解释每个示例程序的作用，以及它与本食谱中所教授的课程的关系。

# 它是如何工作的...

首先，让我们简要回顾一下 C++异常是如何抛出和捕获的。在下面的例子中，我们将从一个函数中抛出一个异常，然后在我们的`main()`函数中捕获异常：

```cpp
#include <iostream>
#include <stdexcept>

void foo()
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    try {
        foo();
    }
    catch(const std::exception &e) {
        std::cout << e.what() << '\n';
    }

    return 0;
}
```

如前面的例子所示，我们创建了一个名为`foo()`的函数，它会抛出一个异常。这个函数在我们的`main()`函数中被调用，位于一个`try`/`catch`块中，用于捕获在`try`块中执行的代码可能抛出的任何异常，这种情况下是`foo()`函数。当`foo()`函数抛出异常时，它被成功捕获并输出到`stdout`。

所有这些都是因为我们没有向`foo()`函数添加`noexcept`说明符。默认情况下，函数允许抛出异常，就像我们在这个例子中所做的那样。然而，在某些情况下，我们不希望允许抛出异常，这取决于我们期望函数执行的方式。具体来说，函数如何处理异常可以定义为以下内容（称为异常安全性）：

+   **无抛出保证**：函数不能抛出异常，如果内部抛出异常，必须捕获和处理异常，包括分配失败。

+   **强异常安全性**：函数可以抛出异常，如果抛出异常，函数修改的任何状态都将被回滚或撤消，没有副作用。

+   **基本异常安全性**：函数可以抛出异常，如果抛出异常，函数修改的任何状态都将被回滚或撤消，但可能会有副作用。应该注意，这些副作用不包括不变量，这意味着程序处于有效的、非损坏的状态。

+   **无异常安全性**：函数可以抛出异常，如果抛出异常，程序可能会进入损坏的状态。

一般来说，如果一个函数具有无抛出保证，它会被标记为`noexcept`；否则，它不会。异常安全性如此重要的一个例子是`std::move`。例如，假设我们有两个`std::vector`实例，我们希望将一个向量移动到另一个向量中。为了执行移动，`std::vector`可能会将向量的每个元素从一个实例移动到另一个实例。如果在移动时允许对象抛出异常，向量可能会在移动过程中出现异常（也就是说，向量中的一半对象被成功移动）。当异常发生时，`std::vector`显然会尝试撤消已经执行的移动，将这些移回原始向量，然后返回异常。问题是，尝试将对象移回将需要`std::move()`，这可能再次抛出异常，导致嵌套异常。实际上，将一个`std::vector`实例移动到另一个实例并不实际执行逐个对象的移动，但调整大小会，而在这个特定问题中，标准库要求使用`std::move_if_noexcept`来处理这种情况以提供异常安全性，当对象的移动构造函数允许抛出时，会退回到复制。

`noexcept`说明符通过明确声明函数不允许抛出异常来解决这些问题。这不仅告诉 API 的用户他们可以安全地使用该函数，而不必担心抛出异常可能会破坏程序的执行，而且还迫使函数的作者安全地处理所有可能的异常或调用`std::terminate()`。尽管`noexcept`根据编译器的不同还提供了通过减少应用程序的整体大小来进行优化，但它的主要用途是说明函数的异常安全性，以便其他函数可以推断函数的执行方式。

在下面的示例中，我们为之前定义的`foo()`函数添加了`noexcept`说明符：

```cpp
#include <iostream>
#include <stdexcept>

void foo() noexcept
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    try {
        foo();
    }
    catch(const std::exception &e) {
        std::cout << e.what() << '\n';
    }

    return 0;
}
```

当编译并执行此示例时，我们得到以下结果：

![](img/f99d2218-74b5-47f1-8108-6a38646732a8.png)

如前面的示例所示，添加了`noexcept`说明符，告诉编译器`foo()`不允许抛出异常。然而，`foo()`函数确实抛出异常，因此在执行时会调用`std::terminate()`。实际上，在这个示例中，`std::terminate()`总是会被调用，这是编译器能够检测并警告的事情。

显然调用`std::terminate()`并不是程序的期望结果。在这种特定情况下，由于作者已经将函数标记为`noexcept`，因此需要作者处理所有可能的异常。可以按照以下方式处理：

```cpp
#include <iostream>
#include <stdexcept>

void foo() noexcept
{
    try {
        throw std::runtime_error("The answer is: 42");
    }
    catch(const std::exception &e) {
        std::cout << e.what() << '\n';
    }
}

int main(void)
{
    foo();
    return 0;
}
```

如前面的示例所示，异常被包裹在`try`/`catch`块中，以确保在`foo()`函数完成执行之前安全地处理异常。此外，在这个示例中，只捕获了源自`std::exception()`的异常。这是作者表明可以安全处理哪些类型的异常的方式。例如，如果抛出的是整数而不是`std::exception()`，由于`foo()`函数添加了`noexcept`，`std::terminate()`仍然会自动执行。换句话说，作为作者，你只需要处理你确实能够安全处理的异常。其余的将被发送到`std::terminate()`；只需理解，这样做会改变函数的异常安全性。如果你打算定义一个不抛出异常的函数，那么该函数就不能抛出异常。

还需注意的是，如果将函数标记为`noexcept`，不仅需要关注自己抛出的异常，还需要关注可能抛出异常的函数。在这种情况下，`foo()`函数内部使用了`std::cout`，这意味着作者要么故意忽略`std::cout`可能抛出的任何异常，导致调用`std::terminate()`（这就是我们这里正在做的），要么作者需要确定`std::cout`可能抛出的异常，并尝试安全地处理它们，包括`std::bad_alloc`等异常。

如果提供的索引超出了向量的边界，`std::vector.at()`函数会抛出`std::out_of_range()`异常。在这种情况下，作者可以捕获这种类型的异常并返回默认值，从而可以安全地将函数标记为`noexcept`。

`noexcept`说明符还可以作为一个函数，接受一个布尔表达式，如下面的示例所示：

```cpp
#include <iostream>
#include <stdexcept>

void foo() noexcept(true)
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    try {
        foo();
    }
    catch(const std::exception &e) {
        std::cout << e.what() << '\n';
    }

    return 0;
}
```

执行时会得到以下结果：

![](img/c75fc0ac-3445-4fe6-a20c-934b783a5d96.png)

如前面的示例所示，`noexcept`说明符被写为`noexcept(true)`。如果表达式求值为 true，则就好像提供了`noexcept`一样。如果表达式求值为 false，则就好像省略了`noexcept`说明符，允许抛出异常。在前面的示例中，表达式求值为 true，这意味着该函数不允许抛出异常，这导致在`foo()`抛出异常时调用`std::terminate()`。

让我们看一个更复杂的示例来演示如何使用它。在下面的示例中，我们将创建一个名为`foo()`的函数，它将一个整数值向左移 32 位并将结果转换为 64 位整数。这个示例将使用模板元编程来编写，允许我们在任何整数类型上使用这个函数：

```cpp
#include <limits>
#include <iostream>
#include <stdexcept>

template<typename T>
uint64_t foo(T val) noexcept(sizeof(T) <= 4)
{
    if constexpr(sizeof(T) <= 4) {
        return static_cast<uint64_t>(val) << 32;
    }

    throw std::runtime_error("T is too large");
}

int main(void)
{
    try {
        uint32_t val1 = std::numeric_limits<uint32_t>::max();
        std::cout << "foo: " << foo(val1) << '\n';

        uint64_t val2 = std::numeric_limits<uint64_t>::max();
        std::cout << "foo: " << foo(val2) << '\n';
    }
    catch(const std::exception &e) {
        std::cout << e.what() << '\n';
    }

    return 0;
}
```

执行时将得到以下结果：

![](img/182cbe31-a769-4160-884a-7f9445e380d2.png)

如前面的示例所示，`foo()`函数的问题在于，如果用户提供了 64 位整数，它无法进行 32 位的移位而不产生溢出。然而，如果提供的整数是 32 位或更少，`foo()`函数就是完全安全的。为了实现`foo()`函数，我们使用了`noexcept`说明符来声明如果提供的整数是 32 位或更少，则该函数不允许抛出异常。如果提供的整数大于 32 位，则允许抛出异常，在这种情况下是一个`std::runtime_error()`异常，说明整数太大无法安全移位。

# 使用 noexcept 运算符

`noexcept`运算符是一个编译时检查，用于询问编译器一个函数是否被标记为`noexcept`。在 C++17 中，这可以与编译时`if`语句配对使用（即在编译时评估的`if`语句，可用于根据函数是否允许抛出异常来改变程序的语义）来改变程序的语义。

在本教程中，我们将探讨如何在自己的代码中使用`noexcept`运算符。这个运算符很重要，因为在某些情况下，你可能无法通过简单地查看函数的定义来确定函数是否能够抛出异常。例如，如果一个函数使用了`noexcept`说明符，你的代码可能无法确定该函数是否会抛出异常，因为你可能无法根据函数的输入来确定`noexcept`说明符将求值为什么。`noexcept`运算符为你提供了处理这些情况的机制，这是至关重要的，特别是在元编程时。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本教程中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试本教程：

1.  从新的终端中，运行以下命令下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter02
```

1.  要编译源代码，请运行以下命令：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe02_examples
```

1.  源代码编译后，可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe02_example01
could foo throw: true

> ./recipe02_example02
could foo throw: true
could foo throw: true
could foo throw: false
could foo throw: false

> ./recipe02_example03
terminate called after throwing an instance of 'std::runtime_error'
what(): The answer is: 42
Aborted

> ./recipe02_example04

> ./recipe02_example05
terminate called after throwing an instance of 'std::runtime_error'
what(): The answer is: 42
Aborted

> ./recipe02_example06
could foo throw: true
could foo throw: true
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程中所教授的课程的关系。

# 它是如何工作的...

`noexcept`运算符用于确定一个函数是否能够抛出异常。让我们从一个简单的示例开始：

```cpp
#include <iostream>
#include <stdexcept>

void foo()
{
    std::cout << "The answer is: 42\n";
}

int main(void)
{
    std::cout << std::boolalpha;
    std::cout << "could foo throw: " << !noexcept(foo()) << '\n';
    return 0;
}
```

这将导致以下结果：

![](img/afafa314-071a-4aa9-8896-0c19d3282f99.png)

如前面的例子所示，我们定义了一个输出到`stdout`的`foo()`函数。我们实际上没有执行`foo()`，而是使用`noexcept`操作符来检查`foo()`函数是否可能抛出异常。如你所见，答案是肯定的；这个函数可能会抛出异常。这是因为我们没有用`noexcept`标记`foo()`函数，正如前面的例子所述，函数默认可以抛出异常。

还应该注意到我们在`noexcept`表达式中添加了`!`。这是因为如果函数被标记为`noexcept`，`noexcept`会返回`true`，这意味着函数不允许抛出异常。然而，在我们的例子中，我们询问的不是函数是否不会抛出异常，而是函数是否可能抛出异常，因此需要逻辑布尔反转。

让我们通过在我们的例子中添加一些函数来扩展这一点。具体来说，在下面的例子中，我们将添加一些会抛出异常的函数以及一些被标记为`noexcept`的函数：

```cpp
#include <iostream>
#include <stdexcept>

void foo1()
{
    std::cout << "The answer is: 42\n";
}

void foo2()
{
    throw std::runtime_error("The answer is: 42");
}

void foo3() noexcept
{
    std::cout << "The answer is: 42\n";
}

void foo4() noexcept
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    std::cout << std::boolalpha;
    std::cout << "could foo throw: " << !noexcept(foo1()) << '\n';
    std::cout << "could foo throw: " << !noexcept(foo2()) << '\n';
    std::cout << "could foo throw: " << !noexcept(foo3()) << '\n';
    std::cout << "could foo throw: " << !noexcept(foo4()) << '\n';
    return 0;
}
```

结果如下：

![](img/6c634422-311e-40ae-a7f8-e20aa940f7a4.png)

在前面的例子中，如果一个函数被标记为`noexcept`，`noexcept`操作符会返回`true`（在我们的例子中输出为`false`）。更重要的是，敏锐的观察者会注意到抛出异常的函数并不会改变`noexcept`操作符的输出。也就是说，如果一个函数*可以*抛出异常，`noexcept`操作符会返回`false`，而不是*会*抛出异常。这一点很重要，因为唯一能知道一个函数*会*抛出异常的方法就是执行它。`noexcept`指定符唯一说明的是函数是否允许抛出异常。它并不说明是否*会*抛出异常。同样，`noexcept`操作符并不能告诉你函数*会*抛出异常与否，而是告诉你函数是否被标记为`noexcept`（更重要的是，`noexcept`指定符的求值结果）。

在我们尝试在更现实的例子中使用`noexcept`指定符之前，让我们看下面的例子：

```cpp
#include <iostream>
#include <stdexcept>

void foo()
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    foo();
}
```

如前面的例子所示，我们定义了一个会抛出异常的`foo()`函数，然后从我们的主函数中调用这个函数，导致调用`std::terminate()`，因为我们在离开程序之前没有处理异常。在更复杂的情况下，我们可能不知道`foo()`是否会抛出异常，因此可能不希望在不需要的情况下添加额外的异常处理开销。为了更好地解释这一点，让我们检查这个例子中`main()`函数的汇编代码：

![](img/8741e7cf-194c-44e7-84c5-b48af8c04011.png)

如你所见，`main`函数很简单，除了调用`foo`函数外没有其他逻辑。具体来说，`main`函数中没有任何捕获逻辑。

现在，让我们在一个更具体的例子中使用`noexcept`操作符：

```cpp
#include <iostream>
#include <stdexcept>

void foo()
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    if constexpr(noexcept(foo())) {
        foo();
    }
    else {
        try {
            foo();
        }
        catch (...)
        { }
    }
}
```

如前面的例子所示，我们在 C++17 中添加的`if`语句中使用了`noexcept`操作符和`constepxr`操作符。这使我们能够询问编译器`foo()`是否允许抛出异常。如果允许，我们在`try`/`catch`块中执行`foo()`函数，以便根据需要处理任何可能的异常。如果我们检查这个函数的汇编代码，如下面的截图所示，我们可以看到一些额外的`catch`逻辑被添加到生成的二进制文件中，以根据需要处理异常：

![](img/49f6bfba-8bae-40ba-8987-e352f7b9625c.png)

现在，让我们进一步说明，使用`noexcept`指定符来声明`foo()`函数不允许抛出异常：

```cpp
#include <iostream>
#include <stdexcept>

void foo() noexcept
{
    throw std::runtime_error("The answer is: 42");
}

int main(void)
{
    if constexpr(noexcept(foo())) {
        foo();
    }
    else {
        try {
            foo();
        }
        catch (...)
        { }
    }
}
```

如前面的示例所示，程序调用了`std::terminate()`，因为`foo()`函数被标记为`noexcept`。此外，如果我们查看生成的汇编代码，我们可以看到`main()`函数不再包含额外的`try`/`catch`逻辑，这意味着我们的优化起作用了：

![](img/2d0478f9-51e3-4438-b303-7d4872bf5a80.png)

最后，如果我们不知道被调用的函数是否会抛出异常，可能无法正确标记自己的函数。让我们看下面的例子来演示这个问题：

```cpp
#include <iostream>
#include <stdexcept>

void foo1()
{
    std::cout << "The answer is: 42\n";
}

void foo2() noexcept(noexcept(foo1()))
{
    foo1();
}

int main(void)
{
    std::cout << std::boolalpha;
    std::cout << "could foo throw: " << !noexcept(foo1()) << '\n';
    std::cout << "could foo throw: " << !noexcept(foo2()) << '\n';
}
```

这将导致以下结果：

![](img/5c98505b-2992-4bc6-a927-e4eb3315fd00.png)

如前面的示例所示，`foo1()`函数没有使用`noexcept`指定符标记，这意味着它允许抛出异常。在`foo2()`中，我们希望确保我们的`noexcept`指定符是正确的，但我们调用了`foo1()`，在这个例子中，我们假设我们不知道`foo1()`是否是`noexcept`。

为了确保`foo2()`被正确标记，我们结合了本示例和上一个示例中学到的知识来正确标记函数。具体来说，我们使用`noexcept`运算符来告诉我们`foo1()`函数是否会抛出异常，然后我们使用`noexcept`指定符的布尔表达式语法来使用`noexcept`运算符的结果来标记`foo2()`是否为`noexcept`。如果`foo1()`被标记为`noexcept`，`noexcept`运算符将返回`true`，导致`foo2()`被标记为`noexcept(true)`，这与简单地声明`noexcept`相同。如果`foo1()`没有被标记为`noexcept`，`noexcept`运算符将返回`false`，在这种情况下，`noexcept`指定符将被标记为`noexcept(false)`，这与不添加`noexcept`指定符相同（即，函数允许抛出异常）。

# 使用 RAII

RAII 是一种编程原则，它规定资源与获取资源的对象的生命周期绑定。RAII 是 C++语言的一个强大特性，它真正有助于将 C++与 C 区分开来，有助于防止资源泄漏和一般不稳定性。

在这个示例中，我们将深入探讨 RAII 的工作原理以及如何使用 RAII 来确保 C++异常不会引入资源泄漏。RAII 对于任何 C++应用程序来说都是至关重要的技术，应该尽可能地使用。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

您需要执行以下步骤来尝试这个示例：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter02
```

1.  要编译源代码，请运行以下命令：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe03_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe03_example01
The answer is: 42

> ./recipe03_example02
The answer is: 42

> ./recipe03_example03
The answer is not: 43

> ./recipe03_example04
The answer is: 42

> ./recipe03_example05
step 1: Collect answers
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它们与本示例中所教授的课程的关系。

# 工作原理...

为了更好地理解 RAII 的工作原理，我们必须首先研究 C++中类的工作原理，因为 C++类用于实现 RAII。让我们看一个简单的例子。C++类提供了对构造函数和析构函数的支持，如下所示：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:
    the_answer()
    {
        std::cout << "The answer is: ";
    }

    ~the_answer()
    {
        std::cout << "42\n";
    }
};

int main(void)
{
    the_answer is;
    return 0;
}
```

这将导致编译和执行时的以下结果：

![](img/1991efef-e0b0-48f0-9c36-1a62bfbec715.png)

在上面的例子中，我们创建了一个既有构造函数又有析构函数的类。当我们创建类的实例时，构造函数被调用，当类的实例失去作用域时，类被销毁。这是一个简单的 C++模式，自从 Bjarne Stroustrup 创建了最初的 C++版本以来一直存在。在底层，编译器在类首次实例化时调用一个构造函数，但更重要的是，编译器必须向程序注入代码，当类的实例失去作用域时执行析构函数。这里需要理解的重要一点是，这个额外的逻辑是由编译器自动为程序员插入的。

在引入类之前，程序员必须手动向程序添加构造和析构逻辑，而构造是一个相当容易做到正确的事情，但析构却不是。在 C 中这种问题的一个经典例子是存储文件句柄。程序员会添加一个调用`open()`函数来打开文件句柄，当文件完成时，会添加一个调用`close()`来关闭文件句柄，忘记在可能出现的所有错误情况下执行`close()`函数。这包括当代码有数百行长，而程序的新成员添加了另一个错误情况，同样忘记根据需要调用`close()`。

RAII 通过确保一旦类失去作用域，所获取的资源就会被释放，解决了这个问题，无论控制流路径是什么。让我们看下面的例子：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:

    int *answer{};

    the_answer() :
        answer{new int}
    {
        *answer = 42;
    }

    ~the_answer()
    {
        std::cout << "The answer is: " << *answer << '\n';
        delete answer;
    }
};

int main(void)
{
    the_answer is;

    if (*is.answer == 42) {
        return 0;
    }

    return 1;
}
```

在这个例子中，我们在类的构造函数中分配一个整数并对其进行初始化。这里需要注意的重要一点是，我们不需要从`new`运算符中检查`nullptr`。这是因为如果内存分配失败，`new`运算符会抛出异常。如果发生这种情况，不仅构造函数的其余部分不会被执行，而且对象本身也不会被构造。这意味着如果构造函数成功执行，你就知道类的实例处于有效状态，并且实际上包含一个在类的实例失去作用域时将被销毁的资源。

然后，类的析构函数输出到`stdout`并删除先前分配的内存。这里需要理解的重要一点是，无论代码采取什么控制路径，当类的实例失去作用域时，这个资源都将被释放。程序员只需要担心类的生命周期。

资源的生命周期与分配资源的对象的生命周期直接相关的这个想法很重要，因为它解决了在 C++异常存在的情况下程序的控制流的一个复杂问题。让我们看下面的例子：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:

    int *answer{};

    the_answer() :
        answer{new int}
    {
        *answer = 43;
    }

    ~the_answer()
    {
        std::cout << "The answer is not: " << *answer << '\n';
        delete answer;
    }
};

void foo()
{
    the_answer is;

    if (*is.answer == 42) {
        return;
    }

    throw std::runtime_error("");
}

int main(void)
{
    try {
        foo();
    }
    catch(...)
    { }

    return 0;
}
```

在这个例子中，我们创建了与上一个例子相同的类，但是在我们的`foo()`函数中，我们抛出了一个异常。然而，`foo()`函数不需要捕获这个异常来确保分配的内存被正确释放。相反，析构函数会为我们处理这个问题。在 C++中，许多函数可能会抛出异常，如果没有 RAII，每个可能抛出异常的函数都需要被包裹在`try`/`catch`块中，以确保任何分配的资源都被正确释放。事实上，在 C 代码中，我们经常看到这种模式，特别是在内核级编程中，使用`goto`语句来确保在函数内部，如果发生错误，函数可以正确地释放之前获取的任何资源。结果就是代码的嵌套，专门用于检查程序中每个函数调用的结果和正确处理错误所需的逻辑。

有了这种类型的编程模型，难怪资源泄漏在 C 中如此普遍。RAII 与 C++异常结合消除了这种容易出错的逻辑，从而使代码不太可能泄漏资源。

在 C++异常存在的情况下如何处理 RAII 超出了本书的范围，因为这需要更深入地了解 C++异常支持是如何实现的。重要的是要记住，C++异常比检查函数的返回值是否有错误更快（因为 C++异常是使用无开销算法实现的），但当实际抛出异常时速度较慢（因为程序必须解开堆栈并根据需要正确执行每个类的析构函数）。因此，出于这个原因以及其他原因，比如可维护性，C++异常不应该用于有效的控制流。

RAII 的另一种用法是`finally`模式，它由 C++ **指导支持库** (**GSL**) 提供。`finally`模式利用了 RAII 的仅析构函数部分，提供了一个简单的机制，在函数的控制流复杂或可能抛出异常时执行非基于资源的清理。考虑以下例子：

```cpp
#include <iostream>
#include <stdexcept>

template<typename FUNC>
class finally
{
    FUNC m_func;

public:
    finally(FUNC func) :
        m_func{func}
    { }

    ~finally()
    {
        m_func();
    }
};

int main(void)
{
    auto execute_on_exit = finally{[]{
        std::cout << "The answer is: 42\n";
    }};
}
```

在前面的例子中，我们创建了一个能够存储在`finally`类实例失去作用域时执行的 lambda 函数的类。在这种特殊情况下，当`finally`类被销毁时，我们输出到`stdout`。尽管这使用了类似于 RAII 的模式，但从技术上讲，这不是 RAII，因为没有获取任何资源。

此外，如果确实需要获取资源，应该使用 RAII 而不是`finally`模式。`finally`模式则在不获取资源但希望在函数返回时执行代码时非常有用（无论程序采取什么控制流路径，条件分支或 C++异常）。

为了证明这一点，让我们看一个更复杂的例子：

```cpp
#include <iostream>
#include <stdexcept>

template<typename FUNC>
class finally
{
    FUNC m_func;

public:
    finally(FUNC func) :
        m_func{func}
    { }

    ~finally()
    {
        m_func();
    }
};

int main(void)
{
    try {
        auto execute_on_exit = finally{[]{
            std::cout << "The answer is: 42\n";
        }};

        std::cout << "step 1: Collect answers\n";
        throw std::runtime_error("???");
        std::cout << "step 3: Profit\n";
    }
    catch (...)
    { }
}
```

执行时，我们得到以下结果：

![](img/974fa02a-a5bd-462b-aa43-3951b03d15dc.png)

在前面的例子中，我们希望无论代码做什么，都能始终输出到`stdout`。在执行过程中，我们抛出了一个异常，尽管抛出了异常，我们的`finally`代码仍然按预期执行。

# 学习为什么永远不要在析构函数中抛出异常

在这个食谱中，我们将讨论 C++异常的问题，特别是在类析构函数中抛出异常的问题，这是应该尽量避免的。这个食谱中学到的经验很重要，因为与其他函数不同，C++类析构函数默认标记为`noexcept`，这意味着如果你在类析构函数中意外地抛出异常，你的程序将调用`std::terminate()`，即使析构函数没有明确标记为`noexcept`。

# 准备工作

在开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本食谱中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

执行以下步骤来尝试这个食谱：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter02
```

1.  要编译源代码，请运行以下命令：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe04_examples
```

1.  源代码编译完成后，您可以通过运行以下命令在本食谱中执行每个示例：

```cpp
> ./recipe04_example01
terminate called after throwing an instance of 'std::runtime_error'
what(): 42
Aborted

> ./recipe04_example02
The answer is: 42

> ./recipe04_example03
terminate called after throwing an instance of 'std::runtime_error'
what(): 42
Aborted

> ./recipe04_example04
# exceptions: 2
The answer is: 42
The answer is: always 42
```

在下一节中，我们将逐步介绍这些示例，并解释每个示例程序的作用以及它与本食谱中教授的课程的关系。

# 它是如何工作的...

在这个食谱中，我们将学习为什么在析构函数中抛出异常是一个*糟糕*的想法，以及为什么类析构函数默认标记为`noexcept`。首先，让我们看一个简单的例子：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:
    ~the_answer()
    {
        throw std::runtime_error("42");
    }
};

int main(void)
{
    try {
        the_answer is;
    }
    catch (const std::exception &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
}
```

当我们执行这个时，我们得到以下结果：

![](img/3d30c668-41a6-430a-8fc4-a95a1da1f660.png)

在这个例子中，我们可以看到，如果我们从类析构函数中抛出异常，将调用`std::terminate()`。这是因为，默认情况下，类析构函数被标记为`noexcept`。

我们可以通过将类的析构函数标记为`noexcept(false)`来明确允许类析构函数抛出异常，就像下一个例子中所示的那样：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:
    ~the_answer() noexcept(false)
    {
        throw std::runtime_error("42");
    }
};

int main(void)
{
    try {
        the_answer is;
 }
    catch (const std::exception &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
}
```

如前面的例子所示，当销毁类时，会抛出异常并得到正确处理。即使这个异常被成功处理了，我们仍然要问自己，在捕获这个异常后程序的状态是什么？析构函数并没有成功完成。如果这个类更复杂，并且有状态/资源需要管理，我们能否得出结论，我们关心的状态/资源是否得到了正确处理/释放？简短的答案是否定的。这就像用锤子摧毁硬盘一样。如果你用锤子猛击硬盘来摧毁它，你真的摧毁了硬盘上的数据吗？没有办法知道，因为当你用锤子猛击硬盘时，你损坏了本来可以用来回答这个问题的电子设备。当你试图销毁硬盘时，你需要一个可靠的过程，确保在任何情况下都不会使销毁硬盘的过程留下可恢复的数据。否则，你无法知道自己处于什么状态，也无法回头。

同样适用于 C++类。销毁 C++类必须是一个必须提供基本异常安全性的操作（即，程序的状态是确定性的，可能会有一些副作用）。否则，唯一的逻辑行为是调用`std::terminate()`，因为你无法确定程序继续执行会发生什么。

除了将程序置于未定义状态之外，从析构函数中抛出异常的另一个问题是，如果已经抛出了异常会发生什么？`try`/`catch`块会捕获什么？让我们看一个这种类型问题的例子：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer
{
public:
    ~the_answer() noexcept(false)
    {
        throw std::runtime_error("42");
    }
};

int main(void)
{
    try {
        the_answer is;
        throw std::runtime_error("first exception");
    }
    catch (const std::exception &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
}
```

在前面的例子中，我们像在前一个例子中一样将析构函数标记为`noexcept(false)`，但是在调用析构函数之前抛出异常，这意味着当调用析构函数时，已经有一个异常正在被处理。现在，当我们尝试抛出异常时，即使析构函数被标记为`noexcept(false)`，也会调用`std::terminate()`：

![](img/afeb2214-83a3-4320-bc7d-abea44492169.png)

这是因为 C++库无法处理这种情况，因为`try`/`catch`块无法处理多个异常。然而，可以有多个待处理的异常；我们只需要一个`try`/`catch`块来处理每个异常。当我们有嵌套异常时，就会出现这种情况，就像这个例子一样：

```cpp
#include <iostream>
#include <stdexcept>

class nested
{
public:
    ~nested()
    {
        std::cout << "# exceptions: " << std::uncaught_exceptions() << '\n';
    }
};

class the_answer
{
public:
    ~the_answer()
    {
        try {
            nested n;
            throw std::runtime_error("42");
        }
        catch (const std::exception &e) {
            std::cout << "The answer is: " << e.what() << '\n';
        }
    }
};
```

在这个例子中，我们将首先创建一个类，输出调用`std::uncaught_exceptions()`的结果，该函数返回当前正在处理的异常总数。然后我们将创建一个第二个类，创建第一个类，然后从其析构函数中抛出异常，重要的是要注意，析构函数中的所有代码都包裹在一个`try`/`catch`块中：

```cpp
int main(void)
{
    try {
        the_answer is;
        throw std::runtime_error("always 42");
    }
    catch (const std::exception &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
}
```

当执行此示例时，我们得到以下结果：

![](img/afcfefea-5caa-45ba-8cb6-817eb3023c2f.png)

最后，我们将创建第二个类，并再次使用另一个`try`/`catch`块抛出异常。与前一个例子不同的是，所有的异常都被正确处理了，实际上，不需要`noexcept(false)`来确保这段代码的正常执行，因为对于每个抛出的异常，我们都有一个`try`/`catch`块。即使在析构函数中抛出了异常，它也被正确处理了，这意味着析构函数安全地执行并保持了`noexcept`的兼容性，即使第二个类在处理两个异常的情况下执行。

# 轻松创建自己的异常类

在本示例中，您将学习如何轻松创建自己的异常类型。这是一个重要的课程，因为尽管 C++异常很容易自己创建，但应遵循一些准则以确保安全地完成这些操作。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例所需的适当工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行示例。

# 如何做到...

按照以下步骤尝试本示例：

1.  从新的终端中运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter02
```

1.  要编译源代码，请运行以下命令：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe05_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe05_example01
The answer is: 42

> ./recipe05_example02
The answer is: 42

> ./recipe05_example03
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

创建自己的 C++异常允许您过滤出您所获得的异常类型。例如，异常是来自您的代码还是 C++库？通过创建自己的 C++异常，您可以在运行时轻松回答这些问题。让我们看下面的例子：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer : public std::exception
{
public:
    the_answer() = default;
    const char *what() const noexcept
    {
        return "The answer is: 42";
    }
};

int main(void)
{
    try {
        throw the_answer{};
    }
    catch (const std::exception &e) {
        std::cout << e.what() << '\n';
    }
}
```

如上例所示，我们通过继承`std::exception`创建了自己的 C++异常。这不是必需的。从技术上讲，任何东西都可以是 C++异常，包括整数。然而，从`std::exception`开始，可以为您提供一个标准接口，包括重写`what()`函数，描述抛出的异常。

在上述示例中，我们在`what()`函数中返回了一个硬编码的字符串。这是理想的异常类型（甚至比 C++库提供的异常更理想）。这是因为这种类型的异常是`nothrow copy-constructable`。具体来说，这意味着异常本身可以被复制，而复制不会引发异常，例如由于`std::bad_alloc`。C++库提供的异常类型支持从`std::string()`构造，这可能会引发`std::bad_alloc`。

上述 C++异常的问题在于，您需要为每种消息类型提供`1`种异常类型。实现安全异常类型的另一种方法是使用以下方法：

```cpp
#include <iostream>
#include <stdexcept>

class the_answer : public std::exception
{
    const char *m_str;
public:

    the_answer(const char *str):
        m_str{str}
    { }

    const char *what() const noexcept
    {
        return m_str;
    }
};

int main(void)
{
    try {
        throw the_answer("42");
    }
    catch (const std::exception &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
}
```

在上述示例中，我们存储了指向`const char*`（即 C 风格字符串）的指针。C 风格字符串作为常量存储在程序中。这种类型的异常满足了所有先前的规则，并且在构造异常期间不会发生任何分配。还应该注意，由于字符串是全局存储的，这种操作是安全的。

使用这种方法可以创建许多类型的异常，包括通过自定义 getter 访问的字符串以外的其他内容（即，无需使用`what()`函数）。然而，如果这些先前的规则对您不是问题，创建自定义 C++异常的最简单方法是简单地对现有的 C++异常进行子类化，例如`std::runtime_error()`，如下例所示：

```cpp
#include <iostream>
#include <stdexcept>
#include <string.h>

class the_answer : public std::runtime_error
{
public:
    explicit the_answer(const char *str) :
        std::runtime_error{str}
    { }
};

int main(void)
{
    try {
        throw the_answer("42");
    }
    catch (const the_answer &e) {
        std::cout << "The answer is: " << e.what() << '\n';
    }
    catch (const std::exception &e) {
        std::cout << "unknown exception: " << e.what() << '\n';
    }
}
```

当执行此示例时，我们会得到以下结果：

![](img/a982b81c-4220-43df-82f3-73b32e54b2ae.png)

在上面的示例中，我们通过对`std::runtime_error()`进行子类化，仅用几行代码就创建了自己的 C++异常。然后，我们可以使用不同的`catch`块来确定抛出了什么类型的异常。只需记住，如果您使用`std::runtime_error()`的`std::string`版本，您可能会在异常本身的构造过程中遇到`std::bad_alloc`的情况。
