# 第十一章：C++中的常见模式

在本章中，您将学习 C++中的各种设计模式。设计模式提供了解决不同类型问题的常见方法，通常在互联网上、会议上以及在工作中的水机前讨论设计模式的优缺点。

本章的目标是向您介绍一些更受欢迎、不太受欢迎甚至有争议的模式，让您了解设计模式试图解决的不同类型问题。这是一个重要的章节，因为它将教会您如何通过教授已经存在的解决方案来解决自己应用程序中遇到的常见问题。学习这些设计模式中的任何一种都将为您打下基础，使您能够在自己的应用程序中遇到问题时自行发现其他设计模式。

本章中的示例如下：

+   学习工厂模式

+   正确使用单例模式

+   使用装饰器模式扩展您的对象

+   使用观察者模式添加通信

+   通过静态多态性提高性能

# 技术要求

要编译和运行本章中的示例，您必须具有管理访问权限，可以访问具有功能互联网连接的运行 Ubuntu 18.04 的计算机。在运行这些示例之前，您必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake 
```

如果这是在 Ubuntu 18.04 之外的任何操作系统上安装的，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter11`](https://github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter11)找到。

# 学习工厂模式

在本示例中，我们将学习工厂模式是什么，如何实现它以及何时使用它。这个示例很重要，特别是在单元测试时，因为工厂模式提供了添加接缝（即，代码中提供机会进行更改的有意义的地方）的能力，能够改变另一个对象分配的对象类型，包括分配虚假对象进行测试的能力。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试工厂模式的代码：

1.  从一个新的终端中，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter11
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本章中的每个示例：

```cpp
> ./recipe01_example01

> ./recipe01_example02

> ./recipe01_example03
correct answer: The answer is: 42

> ./recipe01_example04
wrong answer: Not sure

> ./recipe01_example05
correct answer: The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 它是如何工作的...

工厂模式提供了一个分配资源的对象，可以更改对象分配的类型。为了更好地理解这种模式的工作原理以及它为什么如此有用，让我们看下面的例子：

```cpp
class know_it_all
{
public:
    auto ask_question(const char *question)
    {
        (void) question;
        return answer("The answer is: 42");
    }
};
```

正如前面的代码所示，我们从一个名为`know_it_all`的类开始，当被问及问题时，它会提供一个答案。在这种情况下，无论问什么问题，它总是返回相同的答案。答案定义如下：

```cpp
class answer
{
    std::string m_answer;

public:
    answer(std::string str) :
        m_answer{std::move(str)}
    { }
};
```

如前所示，答案是一个简单的类，它根据一个字符串构造并在内部存储字符串。在这种情况下，重要的是要注意，这个 API 的用户实际上无法提取答案类存储的字符串，这意味着使用这些 API 的方式如下：

```cpp
int main(void)
{
    know_it_all universe;
    auto ___ = universe.ask_question("What is the meaning of life?");
}
```

如上所示，我们可以提问，得到一个结果，但我们不确定实际提供了什么结果。这种问题在面向对象编程中经常存在，测试这种逻辑是为什么整本书都写了。模拟是一个专门设计用来验证测试输出的假对象（不像假对象，它只是提供测试输入的对象）。然而，在上面的例子中，模拟仍然需要一种方式来创建，以便验证函数的输出。这就是工厂模式的作用。

让我们修改`answer`类，如下所示：

```cpp
class answer
{
    std::string m_answer;

public:
    answer(std::string str) :
        m_answer{std::move(str)}
    { }

    static inline auto make_answer(std::string str)
    { return answer(str); }
};
```

如上所示的代码中，我们添加了一个静态函数，允许`answer`类创建自己的实例。我们没有改变`answer`类不提供提取其内部内容的能力，只是改变了`answer`类的创建方式。然后我们可以修改`know_it_all`类，如下所示：

```cpp
template<factory_t factory = answer::make_answer>
class know_it_all
{
public:
    auto ask_question(const char *question)
    {
        (void) question;
        return factory("The answer is: 42");
    }
};
```

如上所示的代码中，唯一的区别是`know_it_all`类接受`factory_t`的模板参数，并使用它来创建`answer`类，而不是直接创建`answer`类。`factory_t`的定义如下：

```cpp
using factory_t = answer(*)(std::string str);
```

这默认使用了我们添加到`answer`类中的静态`make_answer()`函数。在最简单的形式下，上面的例子演示了工厂模式。我们不直接创建对象，而是将对象的创建委托给另一个对象。上述实现并不改变这两个类的使用方式，如下所示：

```cpp
int main(void)
{
    know_it_all universe;
    auto ___ = universe.ask_question("What is the meaning of life?");
}
```

如上所示，`main()`逻辑保持不变，但这种新方法确保`know_it_all`类专注于回答问题，而不必担心如何创建`answer`类本身，将这个任务留给另一个对象。这个微妙变化背后的真正力量是，我们现在可以为`know_it_all`类提供一个不同的工厂，从而返回一个不同的`answer`类。为了演示这一点，让我们创建一个新的`answer`类，如下所示：

```cpp
class expected_answer : public answer
{
public:
    expected_answer(std::string str) :
        answer{str}
    {
        if (str != "The answer is: 42") {
            std::cerr << "wrong answer: " << str << '\n';
            exit(1);
        }

        std::cout << "correct answer: " << str << '\n';
    }

    static inline answer make_answer(std::string str)
    { return expected_answer(str); }
};
```

如上所示，我们创建了一个新的`answer`类，它是原始`answer`类的子类。这个新类在构造时检查给定的值，并根据提供的字符串输出成功或失败。然后我们可以使用这个新的`answer`类，如下所示：

```cpp
int main(void)
{
    know_it_all<expected_answer::make_answer> universe;
    auto ___ = universe.ask_question("What is the meaning of life?");
}
```

以下是结果输出：

![](img/41a225db-9424-4bd1-aa9f-8be824d88b8d.png)

使用上述方法，我们可以询问不同的问题，以查看`know_it_all`类是否提供了正确的答案，而无需修改原始的`answer`类。例如，假设`know_it_all`类是这样实现的：

```cpp
template<factory_t factory = answer::make_answer>
class know_it_all
{
public:
    auto ask_question(const char *question)
    {
        (void) question;
        return factory("Not sure");
    }
};
```

我们测试了这个`know_it_all`类的版本，如下所示：

```cpp
int main(void)
{
    know_it_all<expected_answer::make_answer> universe;
    auto ___ = universe.ask_question("What is the meaning of life?");
}
```

结果将如下所示：

![](img/3626806f-b78e-4c07-81cf-329507c18135.png)

应该注意的是，有几种实现工厂模式的方法。上述方法使用模板参数来改变`know_it_all`类创建答案的方式，但我们也可以使用运行时方法，就像这个例子中一样：

```cpp
class know_it_all
{
    std::function<answer(std::string str)> m_factory;

public:
    know_it_all(answer(*f)(std::string str) = answer::make_answer) :
        m_factory{f}
    { }

    auto ask_question(const char *question)
    {
        (void) question;
        return m_factory("The answer is: 42");
    }
};
```

在上文中，我们首先使用自定义的`know_it_all`构造函数，它存储了一个指向工厂函数的指针，该函数默认为我们的`answer`类，但提供了更改工厂的能力，如下所示：

```cpp
int main(void)
{
    know_it_all universe(expected_answer::make_answer);
    auto ___ = universe.ask_question("What is the meaning of life?");
}
```

如果需要，我们还可以为这个类添加一个 setter 来在运行时更改这个函数指针。

# 正确使用单例模式

在这个教程中，我们将学习如何在 C++11 及以上正确实现单例模式，以及何时适合使用单例模式。这个教程很重要，因为它将教会你何时使用单例模式，它提供了对单个全局资源的清晰定义，确保资源保持全局，而不会出现多个副本的可能性。

# 准备工作

在开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保你的操作系统具有编译和执行本书中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试单例模式：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter11
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  一旦源代码被编译，你可以通过运行以下命令来执行本书中的每个示例：

```cpp
> ./recipe02_example01
memory: 0x4041a0
i1: 0x4041a0
i2: 0x4041a4
i3: 0x4041a8
i4: 0x4041ac

> ./recipe02_example02
memory: 0x4041a0
i1: 0x4041a0
i2: 0x4041a4
i3: 0x4041a0
i4: 0x4041a4

> ./recipe02_example03
memory: 0x4041a0
i1: 0x4041a0
i2: 0x4041a4
i3: 0x4041a8
i4: 0x4041ac

> ./recipe02_example04
memory: 0x4041a0
i1: 0x4041a0
i2: 0x4041a4
i3: 0x4041a8
i4: 0x4041ac
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本书所教授的课程的关系。

# 工作原理...

单例模式在 C++中已经存在了好几年，可以说是 C++中最具争议的模式之一，因为其全局性质会在应用程序中引入耦合（类似于全局变量引入的耦合）。单例模式实现了一个单一的全局资源。具体来说，它创建了一个维持全局范围的对象，同时确保自身没有副本存在。关于是否应该在代码中使用单例模式的争论将不会在本书中得到解答，因为这取决于你的用例，但至少让我们来讨论一下这种模式的一些优缺点。

**优点：**单例模式为只能包含一个实例的全局资源提供了一个明确定义的接口。不管我们喜欢与否，全局资源存在于我们所有的应用程序中（例如，堆内存）。如果需要这样一个全局资源，并且你有一种处理耦合的机制（例如，Hippomocks 这样的模拟引擎），单例模式是确保全局资源得到正确管理的好方法。

**缺点：**以下是缺点：

+   单例模式定义了一个全局资源，就像任何全局资源（例如，全局变量）一样，使用单例对象的任何代码都会与单例对象紧密耦合。在面向对象设计中，耦合应该始终被避免，因为它会阻止你能够伪造代码可能依赖的资源，这会限制测试时的灵活性。

+   单例模式隐藏了依赖关系。当检查一个对象的接口时，无法确定对象的实现是否依赖于全局资源。大多数人认为这可以通过良好的文档来处理。

+   单例模式在应用程序的整个生命周期中保持其状态。这在单元测试时尤其明显（也就是说，缺点是显而易见的），因为单例的状态会从一个单元测试传递到下一个单元测试，这被大多数人认为是对单元测试的违反。

一般来说，全局资源应该始终被避免。为了确保你的代码被正确编写以实施单例设计模式，如果你需要一个单一的全局资源。让我们讨论以下的例子。

假设你正在为一个嵌入式设备编写应用程序，你的嵌入式设备有一个额外的内存池，你可以将其映射到你的应用程序中（例如，用于视频或网络设备的设备内存）。现在，假设你只能有一个这样的额外内存池，并且你需要实现一组 API 来从这个池中分配内存。在我们的例子中，我们将使用以下方式来实现这个内存池：

```cpp
uint8_t memory[0x1000] = {};
```

接下来，我们将实现一个内存管理器类，以从这个池中分配内存，如下所示：

```cpp
class mm
{
    uint8_t *cursor{memory};

public:
    template<typename T>
    T *allocate()
    {
        if (cursor + sizeof(T) > memory + 0x1000) {
            throw std::bad_alloc();
        }

        auto ptr = new (cursor) T;
        cursor += sizeof(T);

        return ptr;
    }
};
```

如前所示的代码，我们创建了一个内存管理器类，它存储指向包含我们单一全局资源的内存缓冲区的指针。然后我们创建一个简单的分配函数，根据需要处理这个内存（没有释放的能力，这使得算法非常简单）。

由于这是一个全局资源，我们可以全局创建这个类，如下所示：

```cpp
mm g_mm;
```

最后，我们可以按照以下方式使用我们的新内存管理器：

```cpp
int main(void)
{
    auto i1 = g_mm.allocate<int>();
    auto i2 = g_mm.allocate<int>();
    auto i3 = g_mm.allocate<int>();
    auto i4 = g_mm.allocate<int>();

    std::cout << "memory: " << (void *)memory << '\n';
    std::cout << "i1: " << (void *)i1 << '\n';
    std::cout << "i2: " << (void *)i2 << '\n';
    std::cout << "i3: " << (void *)i3 << '\n';
    std::cout << "i4: " << (void *)i4 << '\n';
}
```

在上面的例子中，我们分配了四个整数指针，然后输出我们内存块的地址和整数指针的地址，以确保算法按预期工作，结果如下：

![](img/4538b81d-108f-4a76-98a9-f759ccfadcff.png)

如前所示，内存管理器根据需要正确分配内存。

前面实现的问题在于内存管理器只是一个像其他类一样的类，这意味着它可以被创建多次以及被复制。为了更好地说明这是一个问题，让我们看下面的例子。我们不是创建一个内存管理器，而是创建两个：

```cpp
mm g_mm1;
mm g_mm2;
```

接下来，让我们按照以下方式使用这两个内存管理器：

```cpp
int main(void)
{
    auto i1 = g_mm1.allocate<int>();
    auto i2 = g_mm1.allocate<int>();
    auto i3 = g_mm2.allocate<int>();
    auto i4 = g_mm2.allocate<int>();

    std::cout << "memory: " << (void *)memory << '\n';
    std::cout << "i1: " << (void *)i1 << '\n';
    std::cout << "i2: " << (void *)i2 << '\n';
    std::cout << "i3: " << (void *)i3 << '\n';
    std::cout << "i4: " << (void *)i4 << '\n';
}
```

如前所示，唯一的区别是现在我们使用两个内存管理器而不是一个。这导致以下输出：

![](img/8a5fd1a4-19c6-44c5-8250-18a3e61714ec.png)

如前所示，内存已经被双重分配，这可能导致损坏和未定义的行为。发生这种情况的原因是内存缓冲区本身是一个全局资源，这是我们无法改变的。内存管理器本身并没有做任何事情来确保这种情况不会发生，因此，这个 API 的用户可能会意外地创建第二个内存管理器。请注意，在我们的例子中，我们明确地创建了第二个副本，但通过简单地传递内存管理器，可能会意外地创建副本。

为了解决这个问题，我们必须处理两种特定的情况：

+   创建多个内存管理器实例

+   复制内存管理器

为了解决这两个问题，让我们现在展示单例模式：

```cpp
class mm
{
    uint8_t *cursor{memory};
    mm() = default;
```

如前所示，我们从将构造函数标记为`private`开始。将构造函数标记为`private`可以防止内存管理器的使用者创建自己的内存管理器实例。相反，要获得内存管理器的实例，我们将使用以下`public`函数：

```cpp
    static auto &instance()
    {
        static mm s_mm;
        return s_mm;
    }
```

这个前面的函数创建了内存管理器的静态（即全局）实例，然后返回对这个实例的引用。使用这个函数，API 的用户只能从这个函数中获得内存管理器的实例，这个函数总是只返回对全局定义资源的引用。换句话说，没有能力创建额外的类实例，否则编译器会报错。

创建单例类的最后一步是以下：

```cpp
    mm(const mm &) = delete;
    mm &operator=(const mm &) = delete;
    mm(mm &&) = delete;
    mm &operator=(mm &&) = delete;
```

如前所示，复制和移动构造函数/操作符被明确删除。这解决了第二个问题。通过删除复制构造函数和操作符，就没有能力创建全局资源的副本，确保类只存在为单一全局对象。

要使用这个单例类，我们需要做以下操作：

```cpp
int main(void)
{
    auto i1 = mm::instance().allocate<int>();
    auto i2 = mm::instance().allocate<int>();
    auto i3 = mm::instance().allocate<int>();
    auto i4 = mm::instance().allocate<int>();

    std::cout << "memory: " << (void *)memory << '\n';
    std::cout << "i1: " << (void *)i1 << '\n';
    std::cout << "i2: " << (void *)i2 << '\n';
    std::cout << "i3: " << (void *)i3 << '\n';
    std::cout << "i4: " << (void *)i4 << '\n';
}
```

这导致以下输出：

![](img/cde66b36-11d4-4296-b84d-0a76d9e7da36.png)

如果我们尝试自己创建另一个内存管理器实例，我们会得到类似以下的错误：

```cpp
/home/user/book/chapter11/recipe02.cpp:166:4: error: ‘constexpr mm::mm()’ is private within this context
  166 | mm g_mm;
```

最后，由于单例类是一个单一的全局资源，我们可以创建包装器来消除冗长，如下所示：

```cpp
template<typename T>
constexpr T *allocate()
{
    return mm::instance().allocate<T>();
}
```

这个改变可以按照以下方式使用：

```cpp
int main(void)
{
    auto i1 = allocate<int>();
    auto i2 = allocate<int>();
    auto i3 = allocate<int>();
    auto i4 = allocate<int>();

    std::cout << "memory: " << (void *)memory << '\n';
    std::cout << "i1: " << (void *)i1 << '\n';
    std::cout << "i2: " << (void *)i2 << '\n';
    std::cout << "i3: " << (void *)i3 << '\n';
    std::cout << "i4: " << (void *)i4 << '\n';
}
```

如前所示，`constexpr`包装器提供了一种简单的方法来消除我们单例类的冗长，如果内存管理器不是单例的话，这将是很难做到的。

# 使用装饰器模式扩展您的对象

在这个示例中，我们将学习如何实现装饰器模式，该模式提供了在不需要继承的情况下扩展类功能的能力，这是静态性质的设计。这个示例很重要，因为继承不支持在运行时扩展类的能力，这是装饰器模式解决的问题。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

执行以下步骤尝试这个示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter11
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  源代码编译完成后，您可以通过运行以下命令执行本示例中的每个示例：

```cpp
> ./recipe03_example01
button width: 42

> ./recipe03_example02
button1 width: 10
button2 width: 42

> ./recipe03_example03
button width: 74

> ./recipe03_example04
button width: 42
button content width: 4
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 它是如何工作的...

在这个示例中，我们将学习如何实现装饰器模式。首先，让我们看一个简单的例子：假设我们正在编写一个 C++应用程序，将托管一个网站。在我们的网站中，我们需要定义一个用户可以点击的按钮，但我们需要计算给定额外边距的按钮的宽度：

```cpp
class margin
{
public:
    int width()
    {
        return 32;
    }
};
```

如前所示，我们创建了一个名为`margin`的类，返回所讨论边距的宽度（我们只关注宽度以简化我们的示例）。然后我们可以按照以下方式定义我们的按钮：

```cpp
class button : public margin
{
public:
    int width()
    {
        return margin::width() + 10;
    }
};
```

如前所示，我们按钮的总宽度是按钮本身的宽度加上边距的宽度。然后我们可以按照以下方式获取按钮的宽度：

```cpp
int main()
{
    auto b = new button();
    std::cout << "button width: " << b->width() << '\n';
}
```

这将产生以下输出：

![](img/697bea8a-1cdb-45ac-8431-53344285e825.png)

前面示例的问题是按钮必须始终具有边距，因为按钮直接继承了边距类。有方法可以防止这种情况发生（例如，我们的按钮可以有一个配置选项，确定按钮是否返回带有边距的宽度），但在这个示例中，我们将使用装饰器模式来解决这个问题，允许我们创建两个按钮：一个带有边距的按钮，一个没有边距的按钮。让我们试试看：

1.  首先，让我们定义以下纯虚基类如下：

```cpp
class base
{
public:
    virtual int width() = 0;
};
```

如前所示，纯虚基类定义了`width`函数。

1.  然后我们可以按照以下方式实现我们的按钮：

```cpp
class button : public base
{
public:
    int width() override
    {
        return 10;
    }
};
```

如前所示，按钮继承了基类并返回`10`的宽度。使用上述，我们可以开始`button`始终是`10`的宽度，按钮没有边距的概念。

1.  要向按钮添加边距，我们首先必须创建一个装饰器类，如下所示：

```cpp
class decorator : public base
{
    std::unique_ptr<base> m_base;

public:
    decorator(std::unique_ptr<base> b) :
        m_base{std::move(b)}
    { }

    int width()
    {
        return m_base->width();
    }
};
```

装饰器模式从一个私有成员开始，指向一个`base`指针，该指针在装饰器的构造函数中设置。装饰器还定义了`width`函数，但将调用转发给基类。

1.  现在，我们可以创建一个边距类，它是一个装饰器，如下所示：

```cpp
class margin : public decorator
{
public:
    margin(std::unique_ptr<base> b) :
        decorator{std::move(b)}
    { }

    int width()
    {
        return decorator::width() + 32;
    }
};
```

如前所示，边距类返回所装饰对象的宽度，并额外添加`32`。

1.  然后我们可以按照以下方式创建我们的两个按钮：

```cpp
int main()
{
    auto button1 = std::make_unique<button>();
    auto button2 = std::make_unique<margin>(std::make_unique<button>());

    std::cout << "button1 width: " << button1->width() << '\n';
    std::cout << "button2 width: " << button2->width() << '\n';
}
```

这将产生以下输出：

![](img/c40ae265-8e79-4f31-8cea-5a48a0d65d00.png)

装饰器模式的最大优势是它允许我们在运行时扩展一个类。例如，我们可以创建一个带有两个边距的按钮：

```cpp
int main()
{
    auto b =
        std::make_unique<margin>(
            std::make_unique<margin>(
                std::make_unique<button>()
            )
        );

    std::cout << "button width: " << b->width() << '\n';
}
```

否则，我们可以创建另一个装饰器。为了演示这一点，让我们扩展我们的基类如下：

```cpp
class base
{
public:
    virtual int width() = 0;
    virtual int content_width() = 0;
};
```

前面的基类现在定义了一个宽度和一个内容宽度（我们按钮内部可以实际使用的空间）。现在，我们可以按照以下方式创建我们的按钮：

```cpp
class button : public base
{
public:
    int width() override
    {
        return 10;
    }

    int content_width() override
    {
        return width() - 1;
    }
};
```

如前所示，我们的按钮具有静态宽度，内容宽度与宽度本身相同减去 1（为按钮的边框留出空间）。然后我们定义我们的装饰器如下：

```cpp
class decorator : public base
{
    std::unique_ptr<base> m_base;

public:
    decorator(std::unique_ptr<base> b) :
        m_base{std::move(b)}
    { }

    int width() override
    {
        return m_base->width();
    }

    int content_width() override
    {
        return m_base->content_width();
    }
};
```

如前所示，唯一的区别是装饰器现在必须转发宽度和内容宽度函数。我们的边距装饰器如下所示：

```cpp
class margin : public decorator
{
public:
    margin(std::unique_ptr<base> b) :
        decorator{std::move(b)}
    { }

    int width() override
    {
        return decorator::width() + 32;
    }

    int content_width() override
    {
        return decorator::content_width();
    }
};
```

与 Web 编程一样，边距增加了对象的大小。它不会改变对象内部内容的空间，因此边距返回的是内容宽度，没有进行修改。通过前面的更改，我们现在可以按照以下方式添加填充装饰器：

```cpp
class padding : public decorator
{
public:
    padding(std::unique_ptr<base> b) :
        decorator{std::move(b)}
    { }

    int width() override
    {
        return decorator::width();
    }

    int content_width() override
    {
        return decorator::content_width() - 5;
    }
};
```

填充装饰器与边距装饰器相反。它不会改变对象的大小，而是减少了给对象内部内容的总空间。因此，它不会改变宽度，但会减小内容的大小。

使用我们的新装饰器创建一个按钮，我们可以使用以下命令：

```cpp
int main()
{
    auto b =
        std::make_unique<margin>(
            std::make_unique<padding>(
                std::make_unique<button>()
            )
        );

    std::cout << "button width: " << b->width() << '\n';
    std::cout << "button content width: " << b->content_width() << '\n';
}
```

如前所示，我们创建了一个具有额外边距和填充的按钮，结果如下输出：

![](img/2d871265-d736-4585-83b8-30b74a6b04e9.png)

装饰器模式提供了创建不同按钮的能力，而无需编译时继承，这将要求我们为每种可能的按钮类型都有一个不同的按钮定义。然而，需要注意的是，装饰器模式会增加分配和函数调用的重定向成本，因此这种运行时灵活性是有代价的。

# 添加与观察者模式的通信

在这个食谱中，我们将学习如何实现观察者模式。观察者模式提供了一个类注册到另一个类以接收事件发生时的通知的能力。Qt 语言通过使用其信号和槽机制提供了这一功能，同时需要使用 MOC 编译器使其工作。这个食谱很重要，因为我们将学习如何在不需要 Qt 的情况下实现观察者模式，而是使用标准的 C++。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本食谱中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

执行以下步骤来尝试这个食谱：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter11
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本食谱中的每个示例：

```cpp
> ./recipe04_example01 
mom's phone received alarm notification
dad's phone received alarm notification
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它与本食谱中所教授的课程的关系。

# 工作原理...

观察者模式提供了观察者在事件发生时被通知的能力。为了解释这是如何工作的，让我们从以下纯虚基类开始：

```cpp
class observer
{
public:
    virtual void trigger() = 0;
};
```

如前所示，我们定义了`observer`，它必须实现`trigger()`函数。然后我们可以创建两个不同版本的这个纯虚基类，如下所示：

```cpp
class moms_phone : public observer
{
public:
    void trigger() override
    {
        std::cout << "mom's phone received alarm notification\n";
    }
};

class dads_phone : public observer
{
public:
    void trigger() override
    {
        std::cout << "dad's phone received alarm notification\n";
    }
};
```

如前所示的代码，我们创建了两个不同的类，它们都是观察者纯虚类的子类，重写了触发函数。然后我们可以实现一个产生观察者可能感兴趣的事件的类，如下所示：

```cpp
class alarm
{
    std::vector<observer *> m_observers;

public:
    void trigger()
    {
        for (const auto &o : m_observers) {
            o->trigger();
        }
    }

    void add_phone(observer *o)
    {
        m_observers.push_back(o);
    }
};
```

如前面的代码所示，我们首先使用`std::vector`来存储任意数量的观察者。然后我们提供一个触发函数，代表我们的事件。当执行此函数时，我们循环遍历所有观察者，并通过调用它们的`trigger()`函数来通知它们事件。最后，我们提供一个函数，允许观察者订阅相关事件。

以下演示了如何使用这些类：

```cpp
int main(void)
{
    alarm a;
    moms_phone mp;
    dads_phone dp;

    a.add_phone(&mp);
    a.add_phone(&dp);

    a.trigger();
}
```

这将产生以下输出：

![](img/381ac2b5-23b3-46d9-9c09-6eda2174b3b4.png)

如前所示，当触发警报类时，观察者将收到事件通知并根据需要处理通知。

# 使用静态多态性来提高性能

在这个教程中，我们将学习如何创建多态性，而无需虚拟继承。相反，我们将使用编译时继承（称为静态多态性）。这个教程很重要，因为静态多态性不会像运行时虚拟继承那样产生性能和内存使用的惩罚（因为不需要 vTable），但会牺牲可读性和无法利用虚拟子类化的运行时优势。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本教程中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试本教程：

1.  从新的终端中运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter11
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe05_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe05_example01
subclass1 specific
common
subclass2 specific
common
> ./recipe05_example02
subclass1 specific
common
subclass2 specific
common
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它与本教程中所教授的课程的关系。

# 工作原理...

多态性的主要目标之一是它提供了覆盖对象执行特定函数的能力，同时也提供了在一组对象中提供通用逻辑的能力。虚拟继承的问题在于，如果希望使用基类作为接口，覆盖的能力就需要使用 vTable（即虚拟表，这是处理虚拟继承所需的额外内存块）。

例如，考虑以下代码：

```cpp
class base
{
public:
    virtual void foo() = 0;

    void common()
    {
        std::cout << "common\n";
    }
};
```

让我们从之前定义的基类开始。它提供了一个`foo()`函数作为纯虚函数（即，子类必须实现此函数），同时还提供了自己的通用逻辑。然后我们可以创建两个子类，如下所示：

```cpp
class subclass1 : public base
{
public:
    void foo() override
    {
        std::cout << "subclass1 specific\n";
    }
};

class subclass2 : public base
{
public:
    void foo() override
    {
        std::cout << "subclass2 specific\n";
    }
};
```

如前所示，我们对基类进行子类化，并使用子类特定功能重写`foo()`函数。然后我们可以从基类调用子类特定的`foo()`函数，如下所示：

```cpp
int main(void)
{
    subclass1 s1;
    subclass2 s2;

    base *b1 = &s1;
    base *b2 = &s2;

    b1->foo();
    b1->common();

    b2->foo();
    b2->common();
}
```

这将产生以下输出：

![](img/523debdb-a99c-47b7-9ea3-d8aa453ca274.png)

这种类型的运行时多态性需要使用 vTable，这不仅增加了每个对象的内存占用，还会导致性能损失，因为每个函数调用都需要进行 vTable 查找。如果不需要虚拟继承的运行时特性，静态多态性可以提供相同的功能而不会产生这些惩罚。

首先，让我们定义基类如下：

```cpp
template<typename T>
class base
{
public:
    void foo()
    { static_cast<T *>(this)->foo(); }

    void common()
    {
        std::cout << "common\n";
    }
};
```

与我们之前的示例一样，基类不实现`foo()`函数，而是要求子类实现此函数（这就允许静态转换将其转换为类型`T`）。

然后我们可以按以下方式实现我们的子类：

```cpp
class subclass1 : public base<subclass1>
{
public:
    void foo()
    {
        std::cout << "subclass1 specific\n";
    }
};

class subclass2 : public base<subclass2>
{
public:
    void foo()
    {
        std::cout << "subclass2 specific\n";
    }
};
```

与前面的例子一样，子类只是实现了`foo()`函数。不同之处在于，这种情况下继承需要使用模板参数，这消除了`foo()`函数需要覆盖的需要，因为基类从未使用虚函数。

前面的静态多态性允许我们执行来自基类的`foo()`函数如下：

```cpp
template<typename T>
void test(base<T> b)
{
    b.foo();
    b.common();
}
```

如前所示，`test()`函数对每个子类都没有任何信息。它只有关于基类（或接口）的信息。这个`test()`函数可以这样执行：

```cpp
int main(void)
{
    subclass1 c1;
    subclass2 c2;

    test(c1);
    test(c2);
}
```

这再次导致相同的输出：

![](img/18a35eb7-3016-43dc-9871-77f0e57eb78f.png)

如前所示，如果在编译时知道多态类型，可以使用静态多态性来消除对`virtual`的需要，从而消除对 vTable 的需要。这种逻辑在使用模板类时特别有帮助，其中基本类型已知但子类类型不知道（并且已提供），允许模板函数只需要基本接口。
