# 使用模板进行通用编程

在本章中，我们将学习高级模板编程技术。这些技术包括根据提供的类型来改变模板类的实现方式，如何处理不同类型的参数以及如何正确地转发它们，如何在运行时和编译时优化代码，以及如何使用 C++17 中添加的一些新特性。这很重要，因为它可以更好地理解模板编程的工作原理，以及如何确保模板的性能符合预期。

经常情况下，我们编写模板代码时假设它以某种方式执行，而实际上它以另一种方式执行，可能会生成不可靠的代码、意外的性能损失，或者两者兼而有之。本章将解释如何避免这些问题，并为编写正确的通用程序奠定基础。

本章中的示例如下：

+   实现 SFINAE

+   学习完美转发

+   使用`if constexpr`

+   使用元组处理参数包

+   使用特性来改变模板实现的行为

+   学习如何实现`template<auto>`

+   使用显式模板声明

# 技术要求

要编译和运行本章中的示例，您必须具有管理权限的计算机，运行 Ubuntu 18.04，并具有正常的互联网连接。在运行这些示例之前，安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake
```

如果这安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

# 实现 SFINAE

在这个示例中，我们将学习如何使用**Substitution Failure Is Not An Error**（**SFINAE**）。这个示例很重要，因为我们经常创建模板时没有确保传递给模板的类型是我们期望的。这可能导致意外行为、性能不佳，甚至是错误的、不可靠的代码。

SFINAE 允许我们明确指定我们在模板中期望的类型。它还为我们提供了一种根据我们提供的类型来改变模板行为的方法。对于一些人来说，SFINAE 的问题在于这个概念很难理解。我们在本示例中的目标是揭开 SFINAE 的神秘面纱，并展示您如何在自己的代码中使用它。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例中示例的必要工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

要尝试这个示例，您需要执行以下步骤：

1.  从新的终端中运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  编译源代码后，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe01_example01
The answer is: 23
The answer is: 42

> ./recipe01_example02
The answer is: 42

> ./recipe01_example03
The answer is: 42

> ./recipe01_example04
The answer is: 42

> ./recipe01_example05
The answer is: 42
The answer is: 42
The answer is: 42.12345678
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

在本示例中，您将学习如何在自己的代码中使用 SFINAE。首先，我们必须先了解 SFINAE 是什么，以及标准库如何使用它来实现`type`特性。如果不了解`type`特性是如何实现的，就很难理解如何使用它们。

首先，理解 SFINAE 最重要的事情是理解它的名字，即*substitution failure is not an error*。这意味着当模板类型被替换时，如果发生失败，编译器将*不会*生成错误。例如，我们可以编写以下内容：

```cpp
#include <iostream>

struct the_answer
{
    using type = unsigned;
};

template<typename T>
void foo(typename T::type t)
{
    std::cout << "The answer is not: " << t << '\n';
}

template<typename T>
void foo(T t)
{
    std::cout << "The answer is: " << t << '\n';
}

int main(void)
{
    foo<the_answer>(23);
    foo<int>(42);

    return 0;
}
```

每个示例的输出如下所示：

```cpp
The answer is: 23
The answer is: 42
```

在这个例子中，我们创建了`foo()`函数的两个版本。第一个版本接受具有我们用来创建函数参数的`type`别名的`T`类型。第二个版本只接受`T`类型本身。然后我们使用`foo()`函数的两个版本，一个使用整数，另一个使用定义了`type`别名的结构。

从前面的例子中可以得出的结论是，当我们调用`foo<int>()`版本的`foo()`函数时，编译器在尝试将`int`类型与具有`type`别名的`foo()`函数的版本进行匹配时不会生成错误。这就是 SFINAE。它只是说，当编译器尝试获取给定类型并将其与模板匹配时，如果发生失败，编译器不会生成错误。唯一会发生错误的情况是，如果编译器找不到合适的替换。例如，如果我们注释掉`foo()`的第二个版本会发生什么？让我们看看：

![](img/84d28ad2-c0bd-49a0-879d-ad42f5add912.png)

从前面的错误输出中可以看出，编译器甚至说错误是一个替换错误。我们提供的模板不是基于提供的类型的有效候选。

从这个例子中得出的另一个重要结论是，编译器能够根据提供的类型在两个不同版本的`foo()`函数之间进行选择。我们可以利用这一点。具体来说，这给了我们根据提供的类型做不同事情的能力。我们所需要的只是一种方法来编写我们的`foo()`函数，以便我们可以根据我们提供的类型启用/禁用模板的不同版本。

这就是`std::enable_if`发挥作用的地方。`std::enable_if`将 SFINAE 的思想推向了下一步，允许我们在其参数为 true 时定义一个类型。否则，它将生成一个替换错误，故意迫使编译器选择模板的不同版本。`std::enable_if`的定义如下：

```cpp
template<bool B, class T = void>
struct enable_if {};

template<class T>
struct enable_if<true, T> { typedef T type; };
```

首先定义了一个结构，它接受`bool B`和一个默认为`void`的`T`类型。然后定义了这个`struct`类型的一个特化，当`bool`为 true 时。具体来说，当`bool`值为`true`时，返回提供的类型，这个类型默认为`void`。为了看到这是如何使用的，让我们看一个例子：

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
constexpr auto is_int()
{ 
    return false; 
}

template<>
constexpr auto is_int<int>()
{ 
    return true; 
}

template<
    typename T,
    std::enable_if_t<is_int<T>(), int> = 0
    >
void the_answer(T is)
{
    std::cout << "The answer is: " << is << '\n';
}

int main(void)
{
    the_answer(42);
    return 0;
}
```

输出如下：

![](img/395bc1d6-02a2-4609-be80-8855f53d6acc.png)

在这个例子中，我们创建了一个名为`is_int()`的函数，它总是返回`false`。然后我们为`int`创建了这个函数的模板特化，返回`true`。接下来，我们创建了一个接受任何类型的函数，但我们在使用我们的`is_int()`函数的模板定义中添加了`std::enable_if_t`（添加的`_t`部分是 C++17 中为`::type`添加的简写）。如果提供的`T`类型是`int`，我们的`is_int()`函数将返回`true`。

`std::enable_if`默认情况下什么也不做。但如果它为`true`，它会返回一个`type`别名，在前面的例子中，就是我们作为`std::enable_if`第二个参数传递的`int`类型。这意味着如果`std::enable_if`为`true`，它将返回一个`int`类型。然后我们将这个`int`类型设置为`0`，这是一个有效的操作。这不会产生失败；我们的模板函数成为一个有效的替换，因此被使用。总之，如果`T`是`int`类型，`std::enable_if`会变成一个`int`类型本身，然后我们将其设置为`0`，这样就可以编译而不会出现问题。如果我们的`T`类型不是`int`，`std::enable_if`会变成什么也没有。试图将什么也没有设置为`0`会导致编译错误，但由于这是 SFINAE，编译器错误不会变成更多的替换错误。

让我们看看错误的情况。如果我们将`42`设置为`42.0`，这是一个`double`，而不是`int`，我们会得到以下结果：

![](img/3eadc479-fa00-4c25-9a68-b82cb13ee914.png)

正如您从上面的错误中看到的，编译器说在`enable_if`中没有名为`type`的类型。如果您查看`std::enable_if`的定义，这是预期的，因为如果为 false，`std::enable_if`不会执行任何操作。它只有在为 true 时才创建一个名为`type`的类型。

为了更好地理解这是如何工作的，让我们看另一个例子：

```cpp
#include <iostream>
#include <type_traits>

template<
    typename T,
    std::enable_if_t<std::is_integral_v<T>>* = nullptr
    >
void the_answer(T is)
{
    std::cout << "The answer is: " << is << '\n';
}

int main(void)
{
    the_answer(42);
    return 0;
}
```

输出如下：

![](img/cb55d047-89b4-4e40-815a-273456762831.png)

在上面的示例中，我们使用了`std::is_integral_v`，它与我们的`is_int()`函数做了相同的事情，不同之处在于它是由标准库提供的，并且可以处理 CV 类型。事实上，标准库有一个巨大的不同版本的这些函数列表，包括不同的类型、继承属性、CV 属性等等。如果您需要检查任何类型的`type`属性，很有可能标准库有一个`std:is_xxx`函数可以使用。

上面的例子几乎与我们之前的例子相同，不同之处在于我们在`std::enable_if`方法中不返回`int`。相反，我们使用`* = nullptr`。这是因为`std::enable_if`默认返回`void`。`*`字符将这个 void 转换为一个 void 指针，然后我们将其设置为`nullptr`。

在下一个例子中，我们展示了另一个变化：

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
std::enable_if_t<std::is_integral_v<T>>
the_answer(T is)
{
    std::cout << "The answer is: " << is << '\n';
}

int main(void)
{
    the_answer(42);
    return 0;
}

```

输出如下：

![](img/5ca6a189-e687-45a2-a8cd-422d3e2f274e.png)

在这个例子中，我们的函数的`void`是由`std::enable_if`创建的。如果`T`不是整数，就不会返回`void`，我们会看到这个错误（而不是首先编译和允许我们执行它）：

![](img/d7ec86d0-edec-409d-8dd5-76b3abfa3978.png)

总之，`std::enable_if`将创建一个名为`type`的类型，该类型基于您提供的类型。默认情况下，这是`void`，但您可以传入任何您想要的类型。这种功能不仅可以用于强制执行模板的类型，还可以根据我们提供的类型定义不同的函数，就像在这个示例中所示的那样：

```cpp
#include <iostream>
#include <type_traits>
#include <iomanip>

template<
    typename T,
    std::enable_if_t<std::is_integral_v<T>>* = nullptr
    >
void the_answer(T is)
{
    std::cout << "The answer is: " << is << '\n';
}

template<
    typename T,
    std::enable_if_t<std::is_floating_point_v<T>>* = nullptr
    >
void the_answer(T is)
{
    std::cout << std::setprecision(10);
    std::cout << "The answer is: " << is << '\n';
}

int main(void)
{
    the_answer(42);
    the_answer(42U);
    the_answer(42.12345678);

    return 0;
}

```

上面代码的输出如下：

![](img/1c12f216-9fc9-4b34-9868-9ccf45ae4fb7.png)

就像本教程中的第一个例子一样，我们创建了相同函数的两个不同版本。SFINAE 允许编译器根据提供的类型选择最合适的版本。

# 学习完美转发

在这个教程中，我们将学习如何使用完美转发。这个教程很重要，因为在编写模板时，通常我们将模板参数传递给其他函数。如果我们不使用完美转发，我们可能会无意中将 r 值引用转换为 l 值引用，导致潜在的复制发生，而不是移动，在某些情况下，这可能是不太理想的。完美转发还为编译器提供了一些提示，可以用来改进函数内联和展开。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本教程中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个教程：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe02_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe02_example01
l-value
l-value

> ./recipe02_example02
l-value
r-value

> ./recipe02_example03
l-value: 42
r-value: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程中所教授的课程的关系。

# 工作原理...

在这个示例中，我们将学习如何使用完美转发来确保当我们在模板中传递参数时（也就是转发我们的参数），我们以不会抹去 r-value 特性的方式进行。为了更好地理解这个问题，让我们看下面的例子：

```cpp
#include <iostream>

struct the_answer
{ };

void foo2(const the_answer &is)
{
    std::cout << "l-value\n";
}

void foo2(the_answer &&is)
{
    std::cout << "r-value\n";
}

template<typename T>
void foo1(T &&t)
{
    foo2(t);
}

int main(void)
{
    the_answer is;
    foo1(is);
    foo1(the_answer());

    return 0;
}

```

输出如下：

![](img/eb4f4cb0-e924-4b6e-af1a-617c8e3183b0.png)

在前面的示例中，我们有`foo()`函数的两个不同版本：一个接受 l-value 引用，一个接受 r-value 引用。然后我们从模板函数中调用`foo()`。这个模板函数接受一个转发引用（也称为通用引用），它是一个 r-value 引用，配合`auto`或模板函数。最后，从我们的主函数中，我们调用我们的模板来看哪个`foo()`函数被调用。第一次调用我们的模板时，我们传入一个 l-value。由于我们得到了一个 l-value，通用引用变成了 l-value，并且调用了我们的`foo()`函数的 l-value 版本。问题是，第二次调用我们的模板函数时，我们给它一个 r-value，但它调用了我们的`foo()`函数的 l-value 版本，即使它得到了一个 r-value。

这里的常见错误是，即使模板函数接受一个通用引用，我们也有一个接受 r-value 的`foo()`函数的版本，我们假设会调用这个`foo()`函数。Scott Meyers 在他关于通用引用的许多讲座中很好地解释了这一点。问题在于，一旦使用通用引用，它就变成了 l-value。传递`names`参数的行为，意味着它必须是 l-value。它迫使编译器将其转换为 l-value，因为它看到你在使用它，即使你只是在传递参数。值得注意的是，我们的示例在优化时无法编译，因为编译器可以安全地确定变量没有被使用，从而可以优化掉 l-value。

为了防止这个问题，我们需要告诉编译器我们希望转发参数。通常，我们会使用`std::move()`来实现。问题是，如果我们最初得到的是 l-value，我们不能使用`std::move()`，因为那样会将 l-value 转换为 r-value。这就是标准库有`std::forward()`的原因，它是使用以下方式实现的：

```cpp
static_cast<T&&>(t)
```

`std::forward()`的作用如下：将参数强制转换回其原始引用类型。这告诉编译器明确地将参数视为 r-value，如果它最初是 r-value，就像以下示例中一样：

```cpp
#include <iostream>

struct the_answer
{ };

void foo2(const the_answer &is)
{
    std::cout << "l-value\n";
}

void foo2(the_answer &&is)
{
    std::cout << "r-value\n";
}

template<typename T>
void foo1(T &&t)
{
    foo2(std::forward<T>(t));
}

int main(void)
{
    the_answer is;
    foo1(is);
    foo1(the_answer());

    return 0;
}

```

输出如下：

![](img/c64d9b68-b5d8-4ce2-ba02-17195ee8906d.png)

前面的示例与第一个示例相同，唯一的区别是我们在模板函数中使用`std::forward()`传递参数。这一次，当我们用 r-value 调用我们的模板函数时，它调用我们的`foo()`函数的 r-value 版本。这被称为**完美转发**。它确保我们在传递参数时保持 CV 属性和 l-/r-value 属性。值得注意的是，完美转发只在使用模板函数或`auto`时有效。这意味着完美转发通常只在编写包装器时有用。标准库包装器的一个很好的例子是`std::make_unique()`。

`std::make_unique()`这样的包装器的一个问题是，你可能不知道需要传递多少个参数。也就是说，你可能最终需要在你的包装器中使用可变模板参数。完美转发通过以下方式支持这一点：

```cpp
#include <iostream>

struct the_answer
{ };

void foo2(const the_answer &is, int i)
{
    std::cout << "l-value: " << i << '\n';
}

void foo2(the_answer &&is, int i)
{
    std::cout << "r-value: " << i << '\n';
}

template<typename... Args>
void foo1(Args &&...args)
{
    foo2(std::forward<Args>(args)...);
}

int main(void)
{
    the_answer is;

    foo1(is, 42);
    foo1(the_answer(), 42);

    return 0;
}
```

输出如下：

![](img/6a2956d6-ac8c-4113-b411-eb131555a556.png)

前面的示例之所以有效，是因为传递给我们的`foo()`函数的可变模板参数被替换为逗号分隔的完美转发列表。

# 使用 if constexpr

在这个教程中，我们将学习如何使用 C++17 中的一个新特性`constexpr if`。这个教程很重要，因为它将教会你如何创建在运行时评估的`if`语句。具体来说，这意味着分支逻辑是在编译时选择的，而不是在运行时。这允许您在编译时更改函数的行为，而不会牺牲性能，这是过去只能通过宏来实现的，而在模板编程中并不实用，正如我们将展示的那样。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本教程中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

您需要执行以下步骤来尝试这个教程：

1.  从新的终端运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe03_example01
The answer is: 42

> ./recipe03_example02
The answer is: 42
The answer is: 42.12345678
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它与本教程所教授的课程的关系。

# 工作原理...

有时，我们希望改变程序的行为，但我们创建的代码始终是常量，这意味着编译器能够确定分支本身的值，就像这个示例中所示的那样：

```cpp
if (!NDEBUG) {}
```

这是一个常见的`if`语句，在很多代码中都有，包括标准库。如果启用了调试，这段代码将评估为`true`。我们可以通过向代码添加调试语句来使用它，这些语句可以被关闭。编译器足够聪明，能够看到`NDEBUG`是`true`还是`false`，并且会添加代码或完全删除代码。换句话说，编译器可以进行简单的优化，减小代码的大小，并且在运行时永远不会改变这个`if`语句的值。问题是，这个技巧依赖于编译器的智能。逻辑的移除是隐式信任的，这经常导致对编译器正在做什么的假设。C++17 添加了一个`constexpr if`语句，允许我们明确地进行。它允许我们告诉编译器：我提供的语句应该在编译时而不是在运行时进行评估。这真正强大的地方在于，当这个假设不成立时，我们会在编译时获得编译时错误，这意味着我们以前隐式信任编译器执行的优化，现在可以在编译时进行验证，如果假设是错误的，我们会得到通知，以便我们可以解决问题，就像这个示例中所示的那样：

```cpp
#include <iostream>

constexpr auto answer = 42;

int main(void)
{
    if constexpr (answer == 42) {
        std::cout << "The answer is: " << answer << '\n';
    }
    else {
        std::cout << "The answer is not: " << answer << '\n';
    }

    return 0;
}

```

输出如下：

![](img/cf8bb6b4-07e7-4b3e-a97e-3a558cfc3533.png)

在前面的示例中，我们创建了`constexpr`并在编译时而不是在运行时进行了评估。如果我们将`constexpr`更改为实际变量，`constexpr if`将导致以下错误：

![](img/9474112b-7528-4649-b754-1e6702247c6c.png)

然后我们可以在我们的模板函数中使用它来根据我们给定的类型改变我们的模板函数的行为，就像这个示例中所示的那样：

```cpp
#include <iostream>
#include <iomanip>

template<typename T>
constexpr void foo(T &&t)
{
    if constexpr (std::is_floating_point_v<T>) {
        std::cout << std::setprecision(10);
    }

    std::cout << "The answer is: " << std::forward<T>(t) << '\n';
}

int main(void)
{
    foo(42);
    foo(42.12345678);
    return 0;
}
```

在前面的示例中，我们使用`std::is_floating_point_v`类型特征来确定我们提供的类型是否是浮点类型。如果类型不是浮点类型，这将返回`constexpr false`，编译器可以优化掉。由于我们使用了`constexpr if`，我们可以确保我们的`if`语句实际上是`constexpr`而不是运行时条件。

# 使用元组处理参数包

在本教程中，我们将学习如何使用`std::tuple`处理可变参数列表。这很重要，因为可变参数列表是用于包装函数的，包装器不知道传递给它的参数，而是将这些参数转发给了解这些参数的东西。然而，也有一些用例，你会关心传递的参数，并且必须有一种方法来处理这些参数。本教程将演示如何做到这一点，包括如何处理任意数量的参数。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本教程中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤

您需要执行以下步骤来尝试本教程：

1.  从新的终端中运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  源代码编译完成后，可以通过运行以下命令执行本教程中的每个示例：

```cpp
> ./recipe04_example01

> ./recipe04_example02
the answer is: 42

> ./recipe04_example03
The answer is: 42

> ./recipe04_example04
2
2

> ./recipe04_example05
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程中所教授的内容的关系。

# 工作原理

可变模板使程序员能够定义模板函数，而无需定义所有参数。这些在包装函数中被广泛使用，因为它们防止包装器必须了解函数的参数，如下例所示：

```cpp
#include <iostream>

template<typename... Args>
void foo(Args &&...args)
{ }

int main(void)
{
    foo("The answer is: ", 42);
    return 0;
}
```

如前面的示例所示，我们创建了一个可以接受任意数量参数的`foo`函数。在这个例子中，我们使用了通用引用符号`Args &&...args`，它确保了 CV 限定符和 l-/r-值性得到保留，这意味着我们可以使用`std::forward()`将可变参数列表传递给任何其他函数，尽可能少地降低性能损失。诸如`std::make_unique()`之类的函数大量使用可变参数。

然而，有时您可能希望访问提供的参数列表中的一个参数。为此，我们可以使用`std::tuple`。这是一个接受可变数量参数并提供`std::get()`函数从`std::tuple`获取任何数据的数据结构，如下例所示：

```cpp
#include <tuple>
#include <iostream>

int main(void)
{
    std::tuple t("the answer is: ", 42);
    std::cout << std::get<0>(t) << std::get<1>(t) << '\n';
    return 0;
}
```

输出如下：

![](img/e3147713-d2d1-4d27-b867-d95407e67851.png)

在前面的示例中，我们创建了`std::tuple`，然后使用`std::get()`函数将`std::tuple`的内容输出到`stdout`。如果尝试访问超出范围的数据，编译器将在编译时知道，并给出类似于以下的错误：

![](img/f0d35dc0-05d0-44ae-9202-1dbc5da6503c.png)

使用`std::tuple`，我们可以按以下方式访问可变参数列表中的数据：

```cpp
#include <tuple>
#include <iostream>

template<typename... Args>
void foo(Args &&...args)
{
    std::tuple t(std::forward<Args>(args)...);
    std::cout << std::get<0>(t) << std::get<1>(t) << '\n';
}

int main(void)
{
    foo("The answer is: ", 42);
    return 0;
}
```

输出如下：

![](img/fac54cfd-01a0-4fc8-a4e1-80bb81c1fd5f.png)

在前面的示例中，我们创建了一个带有可变参数列表的函数。然后，我们使用`std::forward()`传递此列表以保留 l-/r-值性到`std::tuple`。最后，我们使用`std::tuple`来访问这些参数。如果我们不使用`std::forward()`，我们将得到传递给`std::tuple`的数据的 l-value 版本。

上面例子的明显问题是，我们在`std::tuple`中硬编码了`0`和`1`索引。可变参数不是运行时的、动态的参数数组。相反，它们是一种说“我不关心我收到的参数”的方式，这就是为什么它们通常被包装器使用的原因。包装器是包装一些关心参数的东西。在`std::make_unique()`的情况下，该函数正在创建`std::unique_ptr`。为此，`std::make_unique()`将为您分配`std::unique_ptr`，使用可变参数列表来初始化新分配的类型，然后将指针提供给`std::unique_ptr`，就像这个例子中所示的那样：

```cpp
template<
    typename T, 
    typename... Args
    >
void make_unique(Args &&...args)
{
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}
```

包装器不关心传递的参数。`T`的构造函数关心。如果你尝试访问可变参数，你就是在说“我关心这些参数”，在这种情况下，如果你关心，你必须对传递的参数的布局有一些想法。

有一些技巧可以让你处理未知数量的参数，然而。尝试这样做的最大问题是处理可变参数的库设施最好在运行时使用，这在大多数情况下并不起作用，就像这个例子中所示的那样：

```cpp
#include <tuple>
#include <iostream>

template<typename... Args>
void foo(Args &&...args)
{
    std::cout << sizeof...(Args) << '\n';
    std::cout << std::tuple_size_v<std::tuple<Args...>> << '\n';
}

int main(void)
{
    foo("The answer is: ", 42);
    return 0;
}

```

输出如下：

![](img/3e437b68-faa5-4a2b-a2e5-1353d8935542.png)

在上面的例子中，我们试图获取可变参数列表中参数的总大小。我们可以使用`sizeof()`函数的可变版本，也可以使用`std::tuple_size`特性来实现这一点。问题是，这并不能在编译时帮助我们，因为我们无法使用这个大小信息来循环遍历参数（因为编译时逻辑没有`for`循环）。

为了克服这一点，我们可以使用一种称为编译时递归的技巧。这个技巧使用模板来创建一个递归模板函数，它将循环遍历可变参数列表中的所有参数。看看这个例子：

```cpp
#include <tuple>
#include <iostream>

template<
    std::size_t I = 0,
    typename ... Args,
    typename FUNCTION
    >
constexpr void
for_each(const std::tuple<Args...> &t, FUNCTION &&func)
{
    if constexpr (I < sizeof...(Args)) {
        func(std::get<I>(t));
        for_each<I + 1>(t, std::forward<FUNCTION>(func));
    }
}
```

我们从一个执行所有魔术的模板函数开始。第一个模板参数是`I`，它是一个从`0`开始的整数。接下来是一个可变模板参数，最后是一个函数类型。我们的模板函数接受我们希望迭代的`std::tuple`（在这种情况下，我们展示了一个常量版本，但我们也可以重载它以提供一个非常量版本），以及我们希望对`std::tuple`中的每个元素调用的函数。换句话说，这个函数将循环遍历`std::tuple`中的每个元素，并对每个迭代的元素调用提供的函数，就像我们在其他语言或 C++库中运行时使用的`for_each()`一样。

在这个函数内部，我们检查是否已经达到了元组的总大小。如果没有，我们获取元组中当前值为`I`的元素，将其传递给提供的函数，然后再次调用我们的`for_each()`函数，传入`I++`。要使用这个`for_each()`函数，我们可以这样做：

```cpp
template<typename... Args>
void foo(Args &&...args)
{
    std::tuple t(std::forward<Args>(args)...);
    for_each(t, [](const auto &arg) {
        std::cout << arg;
    });
}
```

在这里，我们得到了一个可变参数列表，我们希望迭代这个列表并将每个参数输出到`stdout`。为此，我们创建了`std::tuple`，就像以前一样，但这次，我们将`std::tuple`传递给我们的`for_each()`函数：

```cpp
int main(void)
{
    foo("The answer is: ", 42);
    std::cout << '\n';

    return 0;
}
```

输出如下：

![](img/d6d45daa-3d12-43cd-b947-93a9aa3990f1.png)

就像我们在之前的例子中所做的那样，我们调用我们的`foo`函数，并传入一些文本，我们希望将其输出到`stdout`，从而演示如何使用`std:tuple`处理可变函数参数，即使我们不知道将收到的参数的总数。

# 使用类型特征来重载函数和对象

C++11 创建时，C++需要处理的一个问题是如何处理`std::vector`的调整大小，它能够接受任何类型，包括从`std::move()`抛出异常的类型。调整大小时，会创建新的内存，并将旧向量的元素移动到新向量。这很好地工作，因为如果`std::move()`不能抛出异常，那么一旦调整大小函数开始将元素从一个数组移动到另一个数组，就不会发生错误。

然而，如果`std::move()`可能会抛出异常，那么在循环进行到一半时可能会发生错误。然而，`resize()`函数无法将旧内存恢复正常，因为尝试移动到旧内存也可能会引发异常。在这种情况下，`resize()`执行复制而不是移动。复制确保旧内存有每个对象的有效副本；因此，如果抛出异常，原始数组保持不变，并且可以根据需要抛出异常。

在本示例中，我们将探讨如何通过更改模板类的行为来实现这一点。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例的适当工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行示例。

# 如何做...

要尝试此示例，需要执行以下步骤：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe05_examples
```

1.  编译源代码后，可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe05_example01
noexcept: r-value
can throw: l-value

> ./recipe05_example02
move
move
move
move
move
--------------
copy
copy
copy
copy
copy
```

在下一节中，我们将逐步介绍每个示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

C++添加了一个名为`std::move_if_noexcept()`的函数。如果移动构造函数/赋值运算符不能抛出异常，此函数将转换为右值，否则将转换为左值。例如，看一下以下代码：

```cpp
#include <iostream>

struct the_answer_noexcept
{
    the_answer_noexcept() = default;

    the_answer_noexcept(const the_answer_noexcept &is) noexcept
    {
        std::cout << "l-value\n";
    }

    the_answer_noexcept(the_answer_noexcept &&is) noexcept
    {
        std::cout << "r-value\n";
    }
};
```

要尝试这样做，我们将执行以下步骤：

1.  首先，我们将创建一个类，该类具有一个不能抛出异常的移动/复制构造函数：

```cpp
struct the_answer_can_throw
{
    the_answer_can_throw() = default;

    the_answer_can_throw(const the_answer_can_throw &is)
    {
        std::cout << "l-value\n";
    }

    the_answer_can_throw(the_answer_can_throw &&is)
    {
        std::cout << "r-value\n";
    }
};
```

1.  接下来，我们将提供一个具有可能抛出异常的移动/复制构造函数的类。最后，让我们使用`std::move_if_noexcept()`来查看在尝试移动这些先前类的实例时是发生移动还是复制：

```cpp
int main(void)
{
    the_answer_noexcept is1;
    the_answer_can_throw is2;

    std::cout << "noexcept: ";
    auto is3 = std::move_if_noexcept(is1);

    std::cout << "can throw: ";
    auto is4 = std::move_if_noexcept(is2);

    return 0;
}

```

上述代码的输出如下：

![](img/0308a282-75f7-42fd-82e6-4debaf2bd0d2.png)

如前面的示例所示，在一种情况下，调用移动构造函数，在另一种情况下，调用复制构造函数，这取决于类型在执行移动时是否会抛出异常。

1.  现在，让我们创建一个简单的模拟向量，并添加一个调整大小函数，以演示如何使用特性更改我们的`template`类的行为：

```cpp
#include <memory>
#include <iostream>
#include <stdexcept>

template<typename T>
class mock_vector
{
public:
    using size_type = std::size_t;

    mock_vector(size_type s) :
        m_size{s},
        m_buffer{std::make_unique<T[]>(m_size)}
    { }

    void resize(size_type size)
        noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        auto tmp = std::make_unique<T[]>(size);

        for (size_type i = 0; i < m_size; i++) {
            tmp[i] = std::move_if_noexcept(m_buffer[i]);
        }

        m_size = size;
        m_buffer = std::move(tmp);
    }

private:
    size_type m_size{};
    std::unique_ptr<T[]> m_buffer{};
};
```

我们的模拟向量有一个内部缓冲区和一个大小。当创建向量时，我们使用给定的大小分配内部缓冲区。然后我们提供一个`resize()`函数，可以用来调整内部缓冲区的大小。我们首先创建新的内部缓冲区，然后循环遍历每个元素，并将一个缓冲区的元素复制到另一个缓冲区。如果`T`不能抛出异常，在循环执行过程中不会触发任何异常，此时新缓冲区将是有效的。如果`T`可以抛出异常，将会发生复制。如果发生异常，旧缓冲区尚未被新缓冲区替换。相反，新缓冲区将被删除，以及所有被复制的元素。

要使用这个，让我们创建一个在移动构造函数/赋值运算符中可能抛出异常的类：

```cpp
struct suboptimal
{
    suboptimal() = default;

    suboptimal(suboptimal &&other)
    {
        *this = std::move(other);
    }

    suboptimal &operator=(suboptimal &&)
    {
        std::cout << "move\n";
        return *this;
    }

    suboptimal(const suboptimal &other)
    {
        *this = other;
    }

    suboptimal &operator=(const suboptimal &)
    {
        std::cout << "copy\n";
        return *this;
    }
};
```

让我们还添加一个在移动构造函数/赋值运算符中不能抛出异常的类：

```cpp
struct optimal
{
    optimal() = default;

    optimal(optimal &&other) noexcept
    {
        *this = std::move(other);
    }

    optimal &operator=(optimal &&) noexcept
    {
        std::cout << "move\n";
        return *this;
    }

    optimal(const optimal &other)
    {
        *this = other;
    }

    optimal &operator=(const optimal &)
    {
        std::cout << "copy\n";
        return *this;
    }
};
```

最后，我们将使用这两个类创建一个向量，并尝试调整其大小：

```cpp
int main(void)
{
    mock_vector<optimal> d1(5);
    mock_vector<suboptimal> d2(5);

    d1.resize(10);
    std::cout << "--------------\n";
    d2.resize(10);

    return 0;
}

```

前面的代码的输出如下：

![](img/e9bee1c2-cc4a-4a6b-8b40-d2f08e9c28b8.png)

如前面的示例所示，当我们尝试调整类的大小时，如果移动不能抛出异常，则执行移动操作，否则执行复制操作。换句话说，类的行为取决于`T`类型的特征。

# 学习如何实现 template<auto>

C++很长时间以来就具有创建模板的能力，这使程序员可以根据类型创建类和函数的通用实现。但是，您也可以提供非类型参数。

在 C++17 中，您现在可以使用`auto`来提供通用的非类型模板参数。在本示例中，我们将探讨如何使用此功能。这很重要，因为它允许您在代码中创建更通用的模板。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本示例中的示例。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行示例。

# 操作步骤...

您需要执行以下步骤来尝试此示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe06_examples
```

1.  源代码编译完成后，可以通过运行以下命令来执行本文中的每个示例：

```cpp
> ./recipe06_example01
The answer is: 42
> ./recipe06_example02
The answer is: 42
The answer is: 42
> ./recipe06_example03
The answer is: 42
```

在下一节中，我们将逐个介绍每个示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

在 C++17 之前，您可以在模板中提供非类型模板参数，但是您必须在定义中声明变量类型，就像本示例中所示的那样：

```cpp
#include <iostream>

template<int answer>
void foo()
{
    std::cout << "The answer is: " << answer << '\n';
}

int main(void)
{
    foo<42>();
    return 0;
}

```

输出如下：

![](img/e5ae3433-362c-4d0d-a865-298472d67d5c.png)

在前面的示例中，我们创建了一个`int`类型的模板参数变量，并将此变量的值输出到`stdout`。在 C++17 中，我们现在可以这样做：

```cpp
#include <iostream>

template<auto answer>
void foo()
{
    std::cout << "The answer is: " << answer << '\n';
}

int main(void)
{
    foo<42>();
    return 0;
}
```

以下是输出：

![](img/dee00755-5e4b-4fe9-8067-fe5306327929.png)

如前所示，我们现在可以使用`auto`而不是`int`。这使我们能够创建一个可以接受多个非类型模板参数的函数。我们还可以使用类型特征来确定允许使用哪些非类型参数，就像本示例中所示的那样：

```cpp
#include <iostream>
#include <type_traits>

template<
    auto answer,
 std::enable_if_t<std::is_integral_v<decltype(answer)>, int> = 0
 >
void foo()
{
    std::cout << "The answer is: " << answer << '\n';
}

int main(void)
{
    foo<42>();
    return 0;
}
```

输出如下：

![](img/08ad857b-3e01-41e9-885d-ceea78bc65f1.png)

在前面的示例中，我们的模板非类型参数只能是整数类型。

# 使用显式模板声明

在本示例中，我们将探讨如何通过创建显式模板声明来加快模板类的编译速度。这很重要，因为模板需要编译器根据需要创建类的实例。在某些情况下，显式模板声明可能为程序员提供一种加快编译速度的方法，通过缓存最有可能使用的模板类型，从而避免包含整个模板定义的需要。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本示例中的示例。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行示例。

# 操作步骤...

您需要执行以下步骤来尝试此示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter04
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe07_examples
```

1.  源代码编译完成后，可以通过运行以下命令来执行本文中的每个示例：

```cpp
> ./recipe07_example01 
The answer is: 42
The answer is: 42
The answer is: 42.1
> ./recipe07_example02 
The answer is: 4
```

在下一节中，我们将逐个介绍这些例子，并解释每个例子程序的作用以及它与本教程中所教授的课程的关系。

# 工作原理...

每当编译器看到使用给定类型的模板类时，它会隐式地创建该类型的一个版本。然而，这可能会发生多次，降低编译器的速度。然而，如果预先知道要使用的类型，这个问题可以通过显式模板特化来解决。看看这个例子：

```cpp
#include <iostream>

template<typename T>
class the_answer
{
public:
    the_answer(T t)
    {
        std::cout << "The answer is: " << t << '\n';
    }
};
```

之前，我们创建了一个简单的结构，在构造过程中输出到`stdout`。通常，一旦看到类的第一个特化，编译器就会创建这个类。然而，我们可以执行以下操作：

```cpp
template class the_answer<int>;
template class the_answer<unsigned>;
template class the_answer<double>;
```

这类似于一个类原型，它明确地创建了我们期望使用的特化。这些必须在它们在代码中使用之前声明（这意味着它们通常在模板的定义之后声明）；然而，一旦声明了，它们可以如下使用：

```cpp
int main(void)
{
    the_answer{42};
    the_answer{42U};
    the_answer{42.1};

    return 0;
}
```

代码的输出如下：

![](img/cdc10992-381a-45a6-80a6-aff500c8753f.png)

在前面的示例中，我们可以像平常一样创建模板的实例，但是在这种情况下，我们可以加快编译器在大量使用这个类的情况下的速度。这是因为在源代码中，我们不需要包含模板的实现。为了证明这一点，让我们看另一个更复杂的例子。在一个头文件（名为`recipe07.h`）中，我们将使用以下内容创建我们的模板：

```cpp
template<typename T>
struct the_answer
{
    T m_answer;

    the_answer(T t);
    void print();
};
```

如你所见，我们有一个没有提供函数实现的`template`类。然后，我们将提供这个模板的实现，使用以下内容在它自己的源文件中：

```cpp
#include <iostream>
#include "recipe07.h"

template<typename T>
the_answer<T>::the_answer(T t) :
    m_answer{t}
{ }

template<typename T>
void the_answer<T>::print()
{
    std::cout << "The answer is: " << m_answer << '\n';
}

template class the_answer<int>;
```

正如你在前面的例子中所看到的，我们添加了显式的模板声明。这确保我们生成了我们期望的类的实现。编译器将为我们期望的类显式地创建实例，就像我们通常编写的任何其他源代码一样。不同之处在于，我们可以明确地为任何类型定义这个类。最后，我们将调用这段代码如下：

```cpp
#include "recipe07.h"

int main(void)
{
    the_answer is{42};
    is.print();

    return 0;
}
```

输出如下：

![](img/b30b50ca-8f48-4791-80c2-83b5886b15f3.png)

如你所见，我们可以以与使用显式类型定义的类相同的方式调用我们的类，而不是使用一个小型的头文件，它没有完整的实现，从而使编译器加快速度。
