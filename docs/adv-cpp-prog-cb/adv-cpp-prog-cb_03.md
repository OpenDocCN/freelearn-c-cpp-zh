# 实现移动语义

在本章中，我们将学习一些高级的 C++移动语义。我们将首先讨论大五，这是一种鼓励程序员显式定义类的销毁和移动/复制语义的习语。接下来，我们将学习如何定义移动构造函数和移动赋值运算符；移动语义的不同组合（包括仅移动和不可复制）；不可移动的类；以及如何实现这些类以及它们的重要性。

本章还将讨论一些常见的陷阱，比如为什么`const &&`移动毫无意义，以及如何克服左值与右值引用类型。本章的示例非常重要，因为一旦启用 C++11 或更高版本，移动语义就会启用，这会改变 C++在许多情况下处理类的方式。本章的示例为在 C++中编写高效的代码提供了基础，使其行为符合预期。

本章的示例如下：

+   使用编译器生成的特殊类成员函数和大五

+   使您的类可移动

+   仅移动类型

+   实现`noexcept`移动构造函数

+   学会谨慎使用`const &&`

+   引用限定的成员函数

+   探索无法移动或复制的对象

# 技术要求

要编译和运行本章中的示例，您必须具有管理权限的计算机运行 Ubuntu 18.04，并具有正常的互联网连接。在运行这些示例之前，您必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake 
```

如果这是安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

# 使用编译器生成的特殊类成员函数和大五

在使用 C++11 或更高版本时，如果您没有在类定义中显式提供它们，编译器将为您的 C++类自动生成某些函数。在本示例中，我们将探讨这是如何工作的，编译器将为您创建哪些函数，以及这如何影响您程序的性能和有效性。总的来说，本示例的目标是证明每个类应该至少定义大五，以确保您的类明确地说明了您希望如何管理资源。

# 准备工作

开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本示例中的示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe01_example01
The answer is: 42

> ./recipe01_example02
The answer is: 42

> ./recipe01_example03
The answer is: 42

> ./recipe01_example04
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 它是如何工作的...

在这个示例中，我们将探讨移动和复制之间的区别，以及这与大五的关系，大五是指所有类都应该显式定义的五个函数。首先，让我们先看一个简单的例子，一个在其构造函数中输出整数值的类：

```cpp
class the_answer
{
    int m_answer{42};

public:

    ~the_answer()
    {
        std::cout << "The answer is: " << m_answer << '\n';
    }
};
```

在前面的示例中，当类被销毁时，它将输出到`stdout`。该类还有一个在构造时初始化的整数成员变量。前面示例的问题在于，我们定义了类的析构函数，因此隐式的复制和移动语义被抑制了。

大五是以下函数，每个类都应该定义这些函数中的至少一个（也就是说，如果你定义了一个，你必须定义它们全部）：

```cpp
~the_answer() = default;

the_answer(the_answer &&) noexcept = default;
the_answer &operator=(the_answer &&) noexcept = default;

the_answer(const the_answer &) = default;
the_answer &operator=(const the_answer &) = default;
```

如上所示，Big Five 包括析构函数、移动构造函数、移动赋值运算符、复制构造函数和复制赋值运算符。这些类的作者不需要实现这些函数，而是应该至少*定义*这些函数，明确说明删除、复制和移动应该如何进行（如果有的话）。这确保了如果这些函数中的一个被定义，类的其余移动、复制和销毁语义是正确的，就像这个例子中一样：

```cpp
class the_answer
{
    int m_answer{42};

public:

    the_answer()
    {
        std::cout << "The answer is: " << m_answer << '\n';
    }

public:

    virtual ~the_answer() = default;

    the_answer(the_answer &&) noexcept = default;
    the_answer &operator=(the_answer &&) noexcept = default;

    the_answer(const the_answer &) = default;
    the_answer &operator=(const the_answer &) = default;
};
```

在前面的示例中，通过定义虚拟析构函数（意味着该类能够参与运行时多态），将类标记为`virtual`。不需要实现（通过将析构函数设置为`default`），但定义本身是显式的，告诉编译器我们希望该类支持虚拟函数。这告诉类的用户，可以使用该类的指针来删除从它派生的任何类的实例。它还告诉用户，继承将利用运行时多态而不是组合。该类还声明允许复制和移动。

让我们看另一个例子：

```cpp
class the_answer
{
    int m_answer{42};

public:

    the_answer()
    {
        std::cout << "The answer is: " << m_answer << '\n';
    }

public:

    ~the_answer() = default;

    the_answer(the_answer &&) noexcept = default;
    the_answer &operator=(the_answer &&) noexcept = default;

    the_answer(const the_answer &) = delete;
    the_answer &operator=(const the_answer &) = delete;
};
```

在前面的示例中，复制被明确删除（这与定义移动构造函数但未定义复制语义相同）。这定义了一个仅移动的类，这意味着该类只能被移动；它不能被复制。标准库中的一个这样的类的例子是`std::unique_ptr`。

下一个类实现了相反的功能：

```cpp
class the_answer
{
    int m_answer{42};

public:

    the_answer()
    {
        std::cout << "The answer is: " << m_answer << '\n';
    }

public:

    ~the_answer() = default;

    the_answer(the_answer &&) noexcept = delete;
    the_answer &operator=(the_answer &&) noexcept = delete;

    the_answer(const the_answer &) = default;
    the_answer &operator=(const the_answer &) = default;
};
```

在前面的示例中，我们明确定义了一个仅复制的类。

有许多不同的 Big Five 的组合。这个教程的重点是显示明确定义这五个函数可以确保类的作者对类本身的意图是明确的。这涉及到它应该如何操作以及用户应该如何使用类。明确确保类的作者并不打算获得一种类型的行为，而是因为编译器将根据编译器的实现和 C++规范的定义隐式构造类，而获得另一种类型的行为。

# 使您的类可移动

在 C++11 或更高版本中，对象可以被复制或移动，这可以用来决定对象的资源是如何管理的。复制和移动之间的主要区别很简单：复制会创建对象管理的资源的副本，而移动会将资源从一个对象转移到另一个对象。

在本教程中，我们将解释如何使一个类可移动，包括如何正确添加移动构造函数和移动赋值运算符。我们还将解释可移动类的一些微妙细节以及如何在代码中使用它们。这个教程很重要，因为在很多情况下，移动对象而不是复制对象可以提高程序的性能并减少内存消耗。然而，如果不正确使用可移动对象，可能会引入一些不稳定性。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本教程中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个教程：

1.  从新的终端中，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe02_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe02_example01
The answer is: 42
> ./recipe02_example02
The answer is: 42
The answer is: 42

The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程所教授的课程的关系。

# 工作原理...

在这个示例中，我们将学习如何使一个类可移动。首先，让我们来看一个基本的类定义：

```cpp
#include <iostream>

class the_answer
{
    int m_answer{42};

public:

    the_answer() = default;

public:

    ~the_answer()
    {
        std::cout << "The answer is: " << m_answer << '\n';
    }
};

int main(void)
{
    the_answer is;
    return 0;
}
```

在前面的例子中，我们创建了一个简单的类，它有一个私有的整数成员，被初始化。然后我们定义了一个默认构造函数和一个析构函数，当类的实例被销毁时，它会输出到`stdout`。默认情况下，这个类是可移动的，但移动操作模拟了一个复制（换句话说，这个简单的例子中移动和复制没有区别）。

要真正使这个类可移动，我们需要添加移动构造函数和移动赋值运算符，如下所示：

```cpp
the_answer(the_answer &&other) noexcept;
the_answer &operator=(the_answer &&other) noexcept;
```

一旦我们添加了这两个函数，我们就能够使用以下方法将我们的类从一个实例移动到另一个实例：

```cpp
instance2 = std::move(instance1);
```

为了支持这一点，在前面的类中，我们不仅添加了移动构造函数和赋值运算符，还实现了一个默认构造函数，为我们的示例类提供了一个有效的移动状态，如下所示：

```cpp
#include <iostream>

class the_answer
{
    int m_answer{};

public:

    the_answer() = default;

    explicit the_answer(int answer) :
        m_answer{answer}
    { }
```

如上所示，该类现在有一个默认构造函数和一个显式构造函数，它接受一个整数参数。默认构造函数初始化整数内存变量，表示我们的移动来源或无效状态：

```cpp
public:

    ~the_answer()
    {
        if (m_answer != 0) {
            std::cout << "The answer is: " << m_answer << '\n';
        }
    }
```

如前面的例子所示，当类被销毁时，我们输出整数成员变量的值，但在这种情况下，我们首先检查整数变量是否有效：

```cpp
    the_answer(the_answer &&other) noexcept
    {
        *this = std::move(other);
    }

    the_answer &operator=(the_answer &&other) noexcept
    {
        if (&other == this) {
            return *this;
        }

        m_answer = std::exchange(other.m_answer, 0);        
        return *this;
    }

    the_answer(const the_answer &) = default;
    the_answer &operator=(const the_answer &) = default;
};
```

最后，我们实现了移动构造函数和赋值运算符。移动构造函数简单地调用移动赋值运算符，以防止重复（因为它们执行相同的操作）。移动赋值运算符首先检查我们是否在将自己移动。这是因为这样做会导致损坏，因为用户期望类仍然包含一个有效的整数，但实际上，内部整数会无意中被设置为`0`。

然后我们交换整数值并将原始值设置为`0`。这是因为，再一次强调，移动不是复制。移动将值从一个实例转移到另一个实例。在这种情况下，被移动到的实例开始为`0`，并被赋予一个有效的整数，而被移出的实例开始有一个有效的整数，移动后被设置为`0`，导致只有`1`个实例包含一个有效的整数。

还应该注意，我们必须定义复制构造函数和赋值运算符。这是因为，默认情况下，如果你提供了移动构造函数和赋值运算符，C++会自动删除复制构造函数和赋值运算符，如果它们没有被显式定义的话。

在这个例子中，我们将比较移动和复制，因此我们定义了复制构造函数和赋值运算符，以确保它们不会被隐式删除。一般来说，最好的做法是为你定义的每个类定义析构函数、移动构造函数和赋值运算符，以及复制构造函数和赋值运算符。这确保了你编写的每个类的复制/移动语义都是明确和有意义的：

```cpp
int main(void)
{
    {
        the_answer is;
        the_answer is_42{42};
        is = is_42;
    }

    std::cout << '\n';

    {
        the_answer is{23};
        the_answer is_42{42};
        is = std::move(is_42);
    }

    return 0;
}
```

当执行上述代码时，我们得到了以下结果：

![](img/bc7cc97b-8542-42e5-9ada-0634f0017fbc.png)

在我们的主函数中，我们运行了两个不同的测试：

+   第一个测试创建了我们类的两个实例，并将一个实例的内容复制到另一个实例。

+   第二个测试创建了我们类的两个实例，然后将一个实例的内容移动到另一个实例。

当执行这个例子时，我们看到第一个测试的输出被写了两次。这是因为我们的类的第一个实例得到了第二个实例的一个副本，而第二个实例有一个有效的整数值。第二个测试的输出只被写了一次，因为我们正在将一个实例的有效状态转移到另一个实例，导致在任何给定时刻只有一个实例具有有效状态。

这里有一些值得一提的例子：

+   移动构造函数和赋值运算符不应该抛出异常。具体来说，移动操作将一个类型的实例的有效状态转移到该类型的另一个实例。在任何时候，这个操作都不应该失败，因为没有状态被创建或销毁。它只是被转移。此外，往往很难*撤消*移动操作。因此，这些函数应该始终被标记为`noexcept`（参考[`github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-move-noexcept`](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-move-noexcept)）。

+   移动构造函数和赋值运算符在其函数签名中不包括`const`类型，因为被移动的实例不能是`const`，因为其内部状态正在被转移，这暗示着写操作正在发生。更重要的是，如果将移动构造函数或赋值运算符标记为`const`，则可能会发生复制。

+   除非您打算创建一个副本，否则应该使用移动，特别是对于大型对象。就像将`const T&`作为函数参数传递以防止发生复制一样，当调用函数时，当资源被移动到另一个变量而不是被复制时，应该使用移动代替复制。

+   编译器在可能的情况下会自动生成移动操作而不是复制操作。例如，如果您在函数中创建一个对象，配置该对象，然后返回该对象，编译器将自动执行移动操作。

现在您知道如何使您的类可移动了，在下一个食谱中，我们将学习什么是只可移动类型，以及为什么您可能希望在应用程序中使用它们。

# 只可移动类型

在这个食谱中，我们将学习如何使一个类成为只可移动的。一个很好的例子是`std::unique_ptr`和`std::shared_ptr`之间的区别。

`std::unique_ptr`的目的是强制动态分配类型的单一所有者，而`std::shared_ptr`允许动态分配类型的多个所有者。两者都允许用户将指针类型的内容从一个实例移动到另一个实例，但只有`std::shared_ptr`允许用户复制指针（因为复制指针会创建多个所有者）。

在这个食谱中，我们将使用这两个类来展示如何制作一个只可移动的类，并展示为什么这种类型的类在 C++中被如此广泛地使用（因为大多数时候我们希望移动而不是复制）。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有正确的工具来编译和执行本食谱中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个食谱：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本食谱中的每个示例：

```cpp
> ./recipe03_example01
The answer is: 42

> ./recipe03_example03
count: 2
The answer is: 42
The answer is: 42

count: 1
The answer is: 42
```

在下一节中，我们将逐个介绍每个示例，并解释每个示例程序的作用以及它与本食谱中所教授的课程的关系。

# 工作原理...

只可移动类是一种可以移动但不能复制的类。为了探索这种类型的类，让我们在以下示例中包装`std::unique_ptr`，它本身是一个只可移动的类：

```cpp
class the_answer
{
    std::unique_ptr<int> m_answer;

public:

    explicit the_answer(int answer) :
        m_answer{std::make_unique<int>(answer)}
    { }

    ~the_answer()
    {
        if (m_answer) {
            std::cout << "The answer is: " << *m_answer << '\n';
        }
    }

public:

    the_answer(the_answer &&other) noexcept
    {
        *this = std::move(other);
    }

    the_answer &operator=(the_answer &&other) noexcept
    {
        m_answer = std::move(other.m_answer);
        return *this;
    }
};
```

前面的类将`std::unique_ptr`作为成员变量存储，并在构造时用整数值实例化内存变量。在销毁时，类会检查`std::unique_ptr`是否有效，如果有效，则将值输出到`stdout`。

乍一看，我们可能会想知道为什么我们必须检查 `std::unique_ptr` 的有效性，因为 `std::unique_ptr` 总是被构造。`std::unique_ptr` 可能变得无效的原因是在移动期间。由于我们正在创建一个只能移动的类（而不是一个不可复制、不可移动的类），我们实现了移动构造函数和移动赋值运算符，它们移动 `std::unique_ptr`。`std::unique_ptr` 在移动时将其内部指针的内容从一个类转移到另一个类，导致该类从存储无效指针（即 `nullptr`）移动。换句话说，即使这个类不能被空构造，如果它被移动，它仍然可以存储 `nullptr`，就像下面的例子一样：

```cpp
int main(void)
{
    the_answer is_42{42};
    the_answer is = std::move(is_42);

    return 0;
}
```

正如前面的例子所示，只有一个类输出到 `stdout`，因为只有一个实例是有效的。与 `std::unique_ptr` 一样，只能移动的类确保你总是有一个资源被创建的总数与实际发生的实例化总数之间的 1:1 关系。

需要注意的是，由于我们使用了 `std::unique_ptr`，我们的类无论我们是否喜欢，都变成了一个只能移动的类。例如，尝试添加复制构造函数或复制赋值运算符以启用复制功能将导致编译错误：

```cpp
the_answer(const the_answer &) = default;
the_answer &operator=(const the_answer &) = default;
```

换句话说，每个包含只能移动的类作为成员的类也会成为只能移动的类。尽管这可能看起来不太理想，但你首先必须问自己：你真的需要一个可复制的类吗？很可能答案是否定的。实际上，在大多数情况下，即使在 C++11 之前，我们使用的大多数类（如果不是全部）都应该是只能移动的。当一个类应该被移动而被复制时，会导致资源浪费、损坏等问题，这也是为什么在规范中添加了移动语义的原因之一。移动语义允许我们定义我们希望分配的资源如何处理，并且它为我们提供了一种在编译时强制执行所需语义的方法。

你可能会想知道前面的例子如何转换以允许复制。以下示例利用了 shared pointer 来实现这一点：

```cpp
#include <memory>
#include <iostream>

class the_answer
{
    std::shared_ptr<int> m_answer;

public:

    the_answer() = default;

    explicit the_answer(int answer) :
        m_answer{std::make_shared<int>(answer)}
    { }

    ~the_answer()
    {
        if (m_answer) {
            std::cout << "The answer is: " << *m_answer << '\n';
        }
    }

    auto use_count()
    { return m_answer.use_count(); }
```

前面的类使用了 `std::shared_ptr` 而不是 `std::unique_ptr`。在内部，`std::shared_ptr` 跟踪被创建的副本数量，并且只有在总副本数为 `0` 时才删除它存储的指针。实际上，你可以使用 `use_count()` 函数查询总副本数。

接下来，我们定义移动构造函数，移动赋值运算符，复制构造函数和复制赋值运算符，如下所示：

```cpp
public:

    the_answer(the_answer &&other) noexcept
    {
        *this = std::move(other);
    }

    the_answer &operator=(the_answer &&other) noexcept
    {
        m_answer = std::move(other.m_answer);
        return *this;
    }

    the_answer(const the_answer &other)
    {
        *this = other;
    }

    the_answer &operator=(const the_answer &other)
    {
        m_answer = other.m_answer;
        return *this;
    }
};
```

这些定义也可以使用 `=` 默认语法来编写，因为这些实现是相同的。最后，我们使用以下方式测试这个类：

```cpp
int main(void)
{
    {
        the_answer is_42{42};
        the_answer is = is_42;
        std::cout << "count: " << is.use_count() << '\n';
    }

    std::cout << '\n';

    {
        the_answer is_42{42};
        the_answer is = std::move(is_42);
        std::cout << "count: " << is.use_count() << '\n';
    }

    return 0;
}
```

如果我们执行前面的代码，我们会得到以下结果：

![](img/80128ca4-0b35-4b29-b649-c871a64b025f.png)

在前面的测试中，我们首先创建了一个类的副本，并输出了总副本数，以查看实际上创建了两个副本。第二个测试执行了 `std::move()` 而不是复制，结果只创建了一个预期中的副本。

# 实现 noexcept 移动构造函数

在本示例中，我们将学习如何确保移动构造函数和移动赋值运算符永远不会抛出异常。C++ 规范并不阻止移动构造函数抛出异常（因为确定这样的要求实际上太难以强制执行，即使在标准库中也存在太多合法的例子）。然而，在大多数情况下，确保不会抛出异常应该是可能的。具体来说，移动通常不会创建资源，而是转移资源，因此应该可能提供强异常保证。一个创建资源的好例子是 `std::list`，即使在移动时也必须提供有效的 `end()` 迭代器。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有编译和执行本文示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来尝试这个示例：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本文中每个示例：

```cpp
> ./recipe04_example01
failed to move

The answer is: 42
```

在下一节中，我们将逐个介绍每个示例，并解释每个示例程序的作用以及它与本文所教授的课程的关系。

# 工作原理...

如前所述，移动不应该抛出异常，以确保强异常保证（即，移动对象的行为不会破坏对象），在大多数情况下，这是可能的，因为移动（不像复制）不会创建资源，而是转移资源。确保您的移动构造函数和移动赋值操作符不会抛出异常的最佳方法是只使用`std::move()`来转移成员变量，就像以下示例中所示的那样：

```cpp
m_answer = std::move(other.m_answer);
```

假设您移动的成员变量不会抛出异常，那么您的类也不会。使用这种简单的技术将确保您的移动构造函数和操作符永远不会抛出异常。但如果这个操作不能使用怎么办？让我们通过以下示例来探讨这个问题：

```cpp
#include <vector>
#include <iostream>

class the_answer
{
    std::vector<int> m_answer;

public:

    the_answer() = default;

    explicit the_answer(int answer) :
        m_answer{{answer}}
    { }

    ~the_answer()
    {
        if (!m_answer.empty()) {
            std::cout << "The answer is: " << m_answer.at(0) << '\n';
        }
    }
```

在前面的示例中，我们创建了一个具有向量作为成员变量的类。向量可以通过默认方式初始化为空，或者可以初始化为单个元素。在销毁时，如果向量有值，我们将该值输出到`stdout`。我们实现`move`构造函数和操作符如下：

```cpp
public:

    the_answer(the_answer &&other) noexcept
    {
        *this = std::move(other);
    }

    the_answer &operator=(the_answer &&other) noexcept
    {
        if (&other == this) {
            return *this;
        }

        try {
            m_answer.emplace(m_answer.begin(), other.m_answer.at(0));
            other.m_answer.erase(other.m_answer.begin());
        }
        catch(...) {
            std::cout << "failed to move\n";
        }

        return *this;
    }
};
```

如图所示，移动操作符将单个元素从一个实例转移到另一个实例（这不是实现移动的最佳方式，但这种实现可以演示要点而不会过于复杂）。如果向量为空，这个操作将抛出异常，就像下面的例子一样：

```cpp
int main(void)
{
    {
        the_answer is_42{};
        the_answer is_what{};

        is_what = std::move(is_42);
    }

    std::cout << '\n';

    {
        the_answer is_42{42};
        the_answer is_what{};

        is_what = std::move(is_42);
    }

    return 0;
}
```

最后，我们尝试在两个不同的测试中移动这个类的一个实例。在第一个测试中，两个实例都是默认构造的，这导致空的类，而第二个测试构造了一个带有单个元素的向量，这导致有效的移动。在这种情况下，我们能够防止移动抛出异常，但应该注意的是，结果类实际上并没有执行移动，导致两个对象都不包含所需的状态。这就是为什么移动构造函数不应该抛出异常。即使我们没有捕获异常，也很难断言抛出异常后程序的状态。移动是否发生？每个实例处于什么状态？在大多数情况下，这种类型的错误应该导致调用`std::terminate()`，因为程序进入了一个损坏的状态。

复制不同，因为原始类保持不变。复制是无效的，程序员可以优雅地处理这种情况，因为被复制的实例的原始状态不受影响（因此我们将其标记为`const`）。

然而，由于被移动的实例是可写的，两个实例都处于损坏状态，没有很好的方法来知道如何处理程序的继续运行，因为我们不知道原始实例是否处于可以正确处理的状态。

# 学会谨慎使用 const&&

在这个食谱中，我们将学习为什么移动构造函数或操作符不应标记为`const`（以及为什么复制构造函数/操作符总是标记为`const`）。这很重要，因为它涉及到移动和复制之间的区别。C++中的移动语义是其最强大的特性之一，了解为什么它如此重要以及它实际上在做什么对于编写良好的 C++代码至关重要。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有适当的工具来编译和执行本食谱中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做…

您需要执行以下步骤来尝试这个食谱：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe05_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本食谱中的每个示例：

```cpp
> ./recipe05_example01
copy
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本食谱中所教授的课程的关系。

# 工作原理…

在这个食谱中，我们将学习为什么`const&&`构造函数或操作符没有意义，并将导致意外行为。移动会转移资源，这就是为什么它标记为非`const`。这是因为转移假定两个实例都被写入（一个实例接收资源，而另一个实例被取走资源）。复制会创建资源，这就是为什么它们并不总是标记为`noexcept`（创建资源绝对可能会抛出异常），并且它们被标记为`const`（因为原始实例被复制，而不是修改）。`const&&`构造函数声称是一个不转移的移动，这必须是一个复制（如果您没有写入原始实例，您不是在移动—您在复制），就像这个例子中一样：

```cpp
#include <iostream>

class copy_or_move
{
public:

    copy_or_move() = default;

public:

    copy_or_move(copy_or_move &&other) noexcept
    {
        *this = std::move(other);
    }

    copy_or_move &operator=(copy_or_move &&other) noexcept
    {
        std::cout << "move\n";
        return *this;
    }

    copy_or_move(const copy_or_move &other)
    {
        *this = other;
    }

    copy_or_move &operator=(const copy_or_move &other)
    {
        std::cout << "copy\n";
        return *this;
    }
};

int main(void)
{
    const copy_or_move test1;
    copy_or_move test2;

    test2 = std::move(test1);
    return 0;
}
```

在前面的示例中，我们创建了一个实现默认移动和复制构造函数/操作符的类。唯一的区别是我们向`stdout`添加了输出，告诉我们是执行了复制还是移动。

然后我们创建了两个类的实例，实例被移动，从被标记为`const`。然后我们执行移动，输出的是一个复制。这是因为即使我们要求移动，编译器也使用了复制。我们可以实现一个`const &&`移动构造函数/操作符，但没有办法将移动写成移动，因为我们标记了被移动的对象为`const`，所以我们无法获取它的资源。这样的移动实际上会被实现为一个复制，与编译器自动为我们做的没有区别。

在下一个食谱中，我们将学习如何向我们的成员函数添加限定符。

# 引用限定成员函数

在这个食谱中，我们将学习什么是引用限定的成员函数。尽管 C++语言的这一方面使用和理解较少，但它很重要，因为它为程序员提供了根据类在调用函数时处于 l-value 还是 r-value 状态来处理资源操作的能力。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有适当的工具来编译和执行本食谱中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做…

您需要执行以下步骤来尝试这个食谱：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe06_examples
```

1.  源代码编译后，您可以通过运行以下命令来执行本文中每个示例：

```cpp
> ./recipe06_example01
the answer is: 42
the answer is not: 0
the answer is not: 0
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本文所教授的课程的关系。

# 工作原理...

在这个例子中，我们将看看什么是引用限定的成员函数。为了解释什么是引用限定的成员函数，让我们看下面的例子：

```cpp
#include <iostream>

class the_answer
{
public:

 ~the_answer() = default;

 void foo() &
 {
 std::cout << "the answer is: 42\n";
 }

 void foo() &&
 {
 std::cout << "the answer is not: 0\n";
 }

public:

 the_answer(the_answer &&other) noexcept = default;
 the_answer &operator=(the_answer &&other) noexcept = default;

 the_answer(const the_answer &other) = default;
 the_answer &operator=(const the_answer &other) = default;
};
```

在这个例子中，我们实现了一个 `foo()` 函数，但是我们有两个不同的版本。第一个版本在末尾有 `&`，而第二个版本在末尾有 `&&`。`foo()` 函数的执行取决于实例是 l-value 还是 r-value，就像下面的例子中一样：

```cpp
int main(void)
{
    the_answer is;

    is.foo();
    std::move(is).foo();
    the_answer{}.foo();
}
```

执行时会得到以下结果：

![](img/19571c4b-ebb1-4680-a183-82571ec2416c.png)

如前面的例子所示，`foo()` 的第一次执行是一个 l-value，因为执行了 `foo()` 的 l-value 版本（即末尾带有 `&` 的函数）。`foo()` 的最后两次执行是 r-value，因为执行了 `foo()` 的 r-value 版本。

参考限定成员函数可用于确保函数仅在正确的上下文中调用。使用这些类型的函数的另一个原因是确保只有当存在 l-value 或 r-value 引用时才调用该函数。

例如，您可能不希望允许 `foo()` 作为 r-value 被调用，因为这种类型的调用并不能确保类的实例在调用本身之外实际上具有生命周期，就像前面的例子中所示的那样。

在下一个示例中，我们将学习如何创建一个既不能移动也不能复制的类，并解释为什么要这样做。

# 探索不能移动或复制的对象

在本文中，我们将学习如何创建一个既不能移动也不能复制的对象，以及为什么要创建这样一个类。复制一个类需要能够复制类的内容，在某些情况下可能是不可能的（例如，复制内存池并不简单）。移动一个类假设该类被允许存在于潜在的无效状态（例如，`std::unique_ptr` 移动时会取得一个 `nullptr` 值，这是无效的）。这样的情况也可能是不希望发生的（现在必须检查有效性）。一个既不能移动也不能复制的类可以克服这些问题。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git
```

这将确保您的操作系统具有正确的工具来编译和执行本文中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

您需要执行以下步骤来尝试这个示例：

1.  从新的终端中运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter03
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe07_examples
```

1.  源代码编译后，您可以通过运行以下命令来执行本文中每个示例：

```cpp
> ./recipe07_example01
The answer is: 42
Segmentation fault (core dumped)
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本文所教授的课程的关系。

# 工作原理...

仅移动类可以阻止类被复制，在某些情况下，这可能是性能的提升。仅移动类还确保了创建的资源与分配的资源之间的 1:1 关系，因为副本是不存在的。然而，移动类可能导致类变为无效，就像这个例子中一样：

```cpp
#include <iostream>

class the_answer
{
    std::unique_ptr<int> m_answer;

public:

    explicit the_answer(int answer) :
        m_answer{std::make_unique<int>(answer)}
    { }

    ~the_answer()
    {
        std::cout << "The answer is: " << *m_answer << '\n';
    }

public:

    the_answer(the_answer &&other) noexcept = default;
    the_answer &operator=(the_answer &&other) noexcept = default;
};

int main(void)
{
    the_answer is_42{42};
    the_answer is_what{42};

    is_what = std::move(is_42);
    return 0;
}
```

如果我们运行上述代码，我们会得到以下结果：

![](img/f6a0a4c9-5084-4fae-8a30-69fb5fff3ce5.png)

在上面的例子中，我们创建了一个可以移动的类，它存储了`std::unique_ptr`。在类的析构函数中，我们对类进行了解引用并输出了它的值。我们没有检查`std::unique_ptr`的有效性，因为我们编写了一个强制有效`std::unique_ptr`的构造函数，忘记了移动可能会撤消这种显式的有效性。结果是，当执行移动操作时，我们会得到一个分段错误。

为了克服这一点，我们需要提醒自己做出了以下假设：

```cpp
class the_answer
{
 std::unique_ptr<int> m_answer;

public:

 explicit the_answer(int answer) :
 m_answer{std::make_unique<int>(answer)}
 { }

 ~the_answer()
 {
 std::cout << "The answer is: " << *m_answer << '\n';
 }

public:

 the_answer(the_answer &&other) noexcept = delete;
 the_answer &operator=(the_answer &&other) noexcept = delete;

 the_answer(const the_answer &other) = delete;
 the_answer &operator=(const the_answer &other) = delete;
};
```

前面的类明确删除了复制和移动操作，这是我们期望的意图。现在，如果我们意外地移动这个类，我们会得到以下结果：

```cpp
/home/user/book/chapter03/recipe07.cpp: In function ‘int main()’:
/home/user/book/chapter03/recipe07.cpp:106:30: error: use of deleted function ‘the_answer& the_answer::operator=(the_answer&&)’
is_what = std::move(is_42);
^
/home/user/book/chapter03/recipe07.cpp:95:17: note: declared here
the_answer &operator=(the_answer &&other) noexcept = delete;
^~~~~~~~
```

这个错误告诉我们，假设这个类是有效的，因此不支持移动。我们要么需要正确地支持移动（这意味着我们必须维护对无效的`std::unique_ptr`的支持），要么我们需要删除`move`操作。正如所示，一个不能被移动或复制的类可以确保我们的代码按预期工作，为编译器提供一种机制，当我们对类做了我们不打算做的事情时，它会警告我们。
