# 创建和实现自己的容器

在本章中，你将学习如何通过利用 C++标准模板库已经提供的现有容器来创建自己的自定义容器。这一章很重要，因为在很多情况下，你的代码将对标准模板库容器执行常见操作，这些操作在整个代码中都是重复的（比如实现线程安全）。本章的食谱将教你如何将这些重复的代码轻松地封装到一个自定义容器中，而无需从头开始编写自己的容器，也不会在代码中散布难以测试和验证的重复逻辑。

在整个本章中，你将学习实现自定义包装器容器所需的技能，能够确保`std::vector`始终保持排序顺序。第一个食谱将教你如何创建这个包装器的基础知识。第二个食谱将在第一个基础上展开，教你如何根据容器的操作方式重新定义容器的接口。在这种情况下，由于容器始终是有序的，你将学习为什么提供`push_back()`函数是没有意义的，即使我们只是创建一个包装器（包装器的添加改变了容器本身的概念）。在第三个食谱中，你将学习使用迭代器的技能，以及为什么在这个例子中只能支持`const`迭代器。最后，我们将向我们的容器添加几个额外的 API，以提供完整的实现。

本章中的食谱如下：

+   使用简单的 std::vector 包装器

+   添加 std::set API 的相关部分

+   使用迭代器

+   添加 std::vector API 的相关部分

# 技术要求

要编译和运行本章中的示例，读者必须具有对运行 Ubuntu 18.04 的计算机的管理访问权限，并且有一个正常的互联网连接。在运行这些示例之前，读者必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake
```

如果这安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter08`](https://github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter08)找到。

# 使用简单的 std::vector 包装器

在本食谱中，我们将学习如何通过包装现有的标准模板库容器来创建自己的自定义容器，以提供所需的自定义功能。在后续的食谱中，我们将在这个自定义容器的基础上构建，最终创建一个基于`std::vector`的完整容器。

这个食谱很重要，因为经常情况下，利用现有容器的代码伴随着每次使用容器时都会重复的常见逻辑。这个食谱（以及整个章节）将教会你如何将这些重复的逻辑封装到你自己的容器中，以便可以独立测试。

# 准备工作

在开始之前，请确保满足所有的技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本食谱中示例的必要工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试本食谱：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter08
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  一旦源代码编译完成，你可以通过运行以下命令来执行本食谱中的每个示例：

```cpp
> ./recipe01_example01
1
2
3
4
5
6
7
8

> ./recipe01_example02
1
2
3

> ./recipe01_example03
3
elements: 4 42 
3
elements: 4 8 15 42 
3
elements: 4 8 15 16 23 42 
```

在下一节中，我们将逐步介绍每个示例，并解释每个示例的作用以及它与本食谱中所教授的课程的关系。

# 它是如何工作的...

在本教程中，我们将学习如何在`std::vector`周围创建一个简单的包装容器。大多数情况下，**标准模板库**（**STL**）容器足以执行应用程序可能需要的任务，通常应避免创建自己的容器，因为它们很难正确实现。

然而，有时您可能会发现自己在容器上重复执行相同的操作。当发生这种情况时，将这些常见操作封装到一个包装容器中通常是有帮助的，可以独立进行单元测试，以确保容器按预期工作。例如，STL 容器不是线程安全的。如果您需要一个容器在每次访问时都能够与线程安全一起使用，您首先需要确保您对容器有独占访问权限（例如，通过锁定`std::mutex`），然后才能进行容器操作。这种模式将在您的代码中重复出现，增加了进入死锁的机会。通过创建一个容器包装器，为容器的每个公共成员添加一个`std::mutex`，可以避免这个问题。

在本教程中，让我们考虑一个例子，我们创建一个向量（即，在连续内存中有直接访问权限的元素数组），它必须始终保持排序状态。首先，我们需要一些头文件：

```cpp
#include <vector>
#include <algorithm>
#include <iostream>
```

为了实现我们的容器，我们将利用`std::vector`。虽然我们可以从头开始实现自己的容器，但大多数情况下这是不需要的，应该避免，因为这样的任务非常耗时和复杂。我们将需要`algorithm`头文件用于`std::sort`和`iostream`用于测试。因此让我们添加如下内容：

```cpp
template<
    typename T,
    typename Compare = std::less<T>,
    typename Allocator = std::allocator<T>
    >
class container
{
    using vector_type = std::vector<T, Allocator>;
    vector_type m_v;

public:
```

容器的定义将从其模板定义开始，与`std::vector`的定义相同，增加了一个`Compare`类型，用于定义我们希望容器排序的顺序。默认情况下，容器将按升序排序，但可以根据需要进行更改。最后，容器将有一个私有成员变量，即该容器包装的`std::vector`的实例。

为了使容器能够与 C++工具、模板函数甚至一些关键语言特性正常工作，容器需要定义与`std::vector`相同的别名，如下所示：

```cpp
    using value_type = typename vector_type::value_type;
    using allocator_type = typename vector_type::allocator_type;
    using size_type = typename vector_type::size_type;
    using difference_type = typename vector_type::difference_type;
    using const_reference = typename vector_type::const_reference;
    using const_pointer = typename vector_type::const_pointer;
    using compare_type = Compare;
```

如您所见，我们无需手动定义别名。相反，我们可以简单地从`std::vector`本身转发别名的声明。唯一的例外是`compare_type`别名，因为这是我们添加到包装容器中的一个别名，表示模板类用于比较操作的类型，最终将提供给`std::sort`。

我们也不包括引用别名的非 const 版本。原因是我们的容器必须始终保持`std::vector`处于排序状态。如果我们为用户提供对`std::vector`中存储的元素的直接写访问权限，用户可能会使`std::vector`处于无序状态，而我们的自定义容器无法按需重新排序。

接下来，让我们定义我们的构造函数（与`std::vector`提供的相同构造函数相对应）。

# 默认构造函数

以下是我们的默认构造函数的定义：

```cpp
    container() noexcept(noexcept(Allocator()))
    {
        std::cout << "1\n";
    }
```

由于`std::vector`的默认构造函数产生一个空向量，我们不需要添加额外的逻辑，因为空向量默认是排序的。接下来，我们必须定义一个接受自定义分配器的构造函数。

# 自定义分配器构造函数

我们的自定义分配器构造函数定义如下：

```cpp
    explicit container(
        const Allocator &alloc
    ) noexcept :
        m_v(alloc)
    {
        std::cout << "2\n";
    }
```

与前一个构造函数一样，这个构造函数创建一个空向量，但使用已经存在的分配器。

# 计数构造函数

接下来的两个构造函数允许 API 的用户设置向量的最小大小如下：

```cpp
    container(
        size_type count,
        const T &value,
        const Allocator &alloc = Allocator()
    ) :
        m_v(count, value, alloc)
    {
        std::cout << "3\n";
    }

    explicit container(
        size_type count,
        const Allocator &alloc = Allocator()
    ) :
        m_v(count, alloc)
    {
        std::cout << "4\n";
    }
```

第一个构造函数将创建一个包含`count`个元素的向量，所有元素都用`value`的值初始化，而第二个构造函数将使用它们的默认值创建元素（例如，整数向量将被初始化为零）。

# 复制/移动构造函数

为了支持复制和移动容器的能力，我们需要实现一个复制和移动构造函数，如下所示：

```cpp
    container(
        const container &other,
        const Allocator &alloc
    ) :
        m_v(other.m_v, alloc)
    {
        std::cout << "5\n";
    }

    container(
        container &&other
    ) noexcept :
        m_v(std::move(other.m_v))
    {
        std::cout << "6\n";
    }
```

由于我们的自定义包装容器必须始终保持排序顺序，因此将一个容器复制或移动到另一个容器不会改变容器中元素的顺序，这意味着这些构造函数也不需要进行排序操作。然而，我们需要特别注意确保通过复制或移动我们的容器封装的内部`std::vector`来正确进行复制或移动。

为了完整起见，我们还提供了一个移动构造函数，允许我们像`std::vector`一样在提供自定义分配器的同时移动。

```cpp
    container(
        container &&other,
        const Allocator &alloc
    ) :
        m_v(std::move(other.m_v), alloc)
    {
        std::cout << "7\n";
    }
```

接下来，我们将提供一个接受初始化列表的构造函数。

# 初始化列表构造函数

最后，我们还将添加一个接受初始化列表的构造函数，如下所示：

```cpp
    container(
        std::initializer_list<T> init,
        const Allocator &alloc = Allocator()
    ) :
        m_v(init, alloc)
    {
        std::sort(m_v.begin(), m_v.end(), compare_type());
        std::cout << "8\n";
    }
```

如前面的代码所示，初始化列表可以以任何顺序为`std::vector`提供初始元素。因此，我们必须在向量初始化后对列表进行排序。

# 用法

让我们测试这个容器，以确保每个构造函数都按预期工作：

```cpp
int main(void)
{
    auto alloc = std::allocator<int>();

    container<int> c1;
    container<int> c2(alloc);
    container<int> c3(42, 42);
    container<int> c4(42);
    container<int> c5(c1, alloc);
    container<int> c6(std::move(c1));
    container<int> c7(std::move(c2), alloc);
    container<int> c8{4, 42, 15, 8, 23, 16};

    return 0;
}
```

如前面的代码块所示，我们通过调用每个构造函数来测试它们，结果如下：

![](img/d05b7686-5517-4965-80e9-17420fc8564b.png)

如您所见，每个构造函数都成功按预期执行。

# 向容器添加元素

构造函数就位后，我们还需要提供手动向容器添加数据的能力（例如，如果我们最初使用默认构造函数创建了容器）。

首先，让我们专注于`std::vector`提供的`push_back()`函数：

```cpp
    void push_back(const T &value)
    {
        m_v.push_back(value);
        std::sort(m_v.begin(), m_v.end(), compare_type());

        std::cout << "1\n";
    }

    void push_back(T &&value)
    {
        m_v.push_back(std::move(value));
        std::sort(m_v.begin(), m_v.end(), compare_type());

        std::cout << "2\n";
    }
```

如前面的代码片段所示，`push_back()`函数具有与`std::vector`提供的版本相同的函数签名，允许我们简单地将函数调用转发到`std::vector`。问题是，向`std::vector`的末尾添加值可能导致`std::vector`进入无序状态，需要我们在每次推送时重新排序`std::vector`（要求`std::vector`始终保持排序状态的结果）。

解决这个问题的一种方法是向容器包装器添加另一个成员变量，用于跟踪`std::vector`何时被污染。实现这些函数的另一种方法是按排序顺序添加元素（即按照排序顺序遍历向量并将元素放在适当的位置，根据需要移动剩余元素）。如果很少向`std::vector`添加元素，那么这种方法可能比调用`std::sort`更有效。然而，如果向`std::vector`频繁添加元素，那么污染的方法可能表现更好。

创建容器包装器的一个关键优势是，可以实现和测试这些类型的优化，而不必更改依赖于容器本身的代码。可以实现、测试和比较这两种实现（或其他实现），以确定哪种优化最适合您的特定需求，而使用容器的代码永远不会改变。这不仅使代码更清晰，而且这种增加的封装打击了面向对象设计的核心，确保代码中的每个对象只有一个目的。对于容器包装器来说，其目的是封装维护`std::vector`的排序顺序的操作。

为了完整起见，我们还将添加`push_back()`的`emplace_back()`版本，就像`std::vector`一样：

```cpp
    template<typename... Args>
    void emplace_back(Args&&... args)
    {
        m_v.emplace_back(std::forward<Args>(args)...);
        std::sort(m_v.begin(), m_v.end(), compare_type());

        std::cout << "3\n";
    }
```

与`std::vector`等效的`emplace_back()`函数的区别在于，我们的版本不返回对创建的元素的引用。这是因为排序会使引用无效，从而无法返回有效的引用。

# push/emplace 的用法

最后，让我们测试我们的`push_back()`和`emplace`函数，以确保它们被正确调用，如下所示：

```cpp
int main(void)
{
    int i = 42;
    container<int> c;

    c.push_back(i);
    c.push_back(std::move(i));
    c.emplace_back(42);

    return 0;
}
```

如前面的代码片段所示，我们调用了`push_back()`的每个版本以及`emplace_back()`函数，以确保它们被正确调用，结果如下：

![](img/51e13fb0-f3e4-460b-8109-137c99e246ed.png)

我们可以进一步添加更好的测试数据到我们的测试容器，如下所示：

```cpp
int main(void)
{
    int i = 42;
    container<int> c;

    c.emplace_back(4);
    c.push_back(i);
    c.emplace_back(15);
    c.push_back(8);
    c.emplace_back(23);
    c.push_back(std::move(16));

    return 0;
}
```

如前面的代码片段所示，我们向我们的向量添加整数`4`、`42`、`15`、`8`、`23`和`16`。在下一个示例中，我们将从`std::set`中窃取 API，以提供更好的`push`和`emplace`API 给我们的容器，以及一个输出函数，以更好地了解`std::vector`包含的内容以及其包含元素的顺序。

# 向 std::set API 添加相关部分

在本示例中，我们将学习如何从`std::set`中添加 API 到我们在第一个示例中创建的自定义容器。具体来说，我们将学习为什么`std::vector::push_back()`和`std::vector::emplace_back()`在与始终保持内部元素排序顺序的自定义容器一起使用时是没有意义的。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例中的示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

按照以下步骤尝试这个示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter08
```

1.  编译源代码，运行以下命令：

```cpp
> cmake .
> make recipe02_examples
```

1.  源代码编译完成后，可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe02_example01 
elements: 4 
elements: 4 42 
elements: 4 15 42 
elements: 4 8 15 42 
elements: 4 8 15 23 42 
elements: 4 8 15 16 23 42 
```

在下一节中，我们将逐步介绍每个示例，并解释每个示例程序的作用，以及它与本示例中所教授的课程的关系。

# 工作原理...

在本章的第一个示例中，我们创建了一个自定义容器包装器，模拟了`std::vector`，但确保向量中的元素始终保持排序顺序，包括添加`std::vector::push_back()`函数和`std::vector::emplace_back()`函数。在本示例中，我们将向我们的自定义容器添加`std::set::insert()`和`std::set::emplace()`函数。

由于我们的容器包装器始终确保`std::vector`处于排序状态，因此无论将元素添加到向量的前端、后端还是中间，都没有区别。无论将元素添加到向量的哪个位置，都必须在访问向量之前对其进行排序，这意味着无论将元素添加到哪个位置，其添加顺序都可能会发生变化。

对于添加元素的位置，我们不必担心，这与`std::set`类似。`std::set`向集合添加元素，然后根据被测试的元素是否是集合的成员，稍后返回`true`或`false`。`std::set`提供了`insert()`和`emplace()`函数来向集合添加元素。让我们向我们的自定义容器添加这些 API，如下所示：

```cpp
    void insert(const T &value)
    {
        push_back(value);
    }

    void insert(T &&value)
    {
        push_back(std::move(value));
    }

    template<typename... Args>
    void emplace(Args&&... args)
    {
        emplace_back(std::forward<Args>(args)...);
    }
```

如前面的代码片段所示，我们添加了一个`insert()`函数（包括复制和移动），以及一个`emplace()`函数，它们只是调用它们的`push_back()`和`emplace_back()`等效函数，确保正确转发传递给这些函数的参数。这些 API 与我们在上一个教程中添加的 API 之间唯一的区别是函数本身的名称。

尽管这样的改变可能看起来微不足道，但这对于重新定义容器的 API 与用户之间的概念是很重要的。`push_back()`和`emplace_back()`函数表明元素被添加到向量的末尾，但实际上并非如此。相反，它们只是简单地添加到`std::vector`中，并且`std::vector`的顺序会根据添加的元素值而改变。因此，需要`push_back()`和`emplace_back()`函数，但应将它们重命名或标记为私有，以确保用户只使用`insert()`和`emplace()`版本来正确管理期望。在编写自己的容器时（即使是包装器），重要的是要遵循最少惊讶原则，以确保用户使用的 API 将按照 API 可能暗示的方式工作。

# 使用迭代器

在本教程中，我们将学习如何为我们在第一个教程中开始的自定义容器添加迭代器支持，该容器包装了一个`std::vector`，确保其内容始终保持排序顺序。

为了添加迭代器支持，我们将学习如何转发`std::vector`已提供的迭代器（我们不会从头开始实现迭代器，因为这超出了本书的范围，从头开始实现容器非常困难）。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本教程中示例所需的正确工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行我们的示例。

# 操作步骤

要尝试本教程，需要按照以下步骤进行：

1.  从新终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter08
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  源代码编译完成后，可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe03_example01 
elements: 4 8 15 16 23 42 

> ./recipe03_example02 
elements: 4 8 15 16 23 42 
elements: 4 8 15 16 23 42 
elements: 42 23 16 15 8 4 
elements: 1 4 8 15 16 23 42 
elements: 4 8 15 16 23 42 
elements: 
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程中所教授的课程的关系。

# 工作原理

我们的自定义容器包装的`std::vector`已经提供了一个有效的迭代器实现，可以用于处理我们的容器。但是，我们需要转发`std::vector`提供的特定部分 API，以确保迭代器正常工作，包括关键的 C++特性，如基于范围的 for 循环。

首先，让我们向我们的自定义容器添加`std::vector`提供的最后一个剩余构造函数：

```cpp
    template <typename Iter>
    container(
        Iter first,
        Iter last,
        const Allocator &alloc = Allocator()
    ) :
        m_v(first, last, alloc)
    {
        std::sort(m_v.begin(), m_v.end(), compare_type());
    }
```

如前面的代码片段所示，我们得到的迭代器类型未定义。迭代器可以来自我们容器的另一个实例，也可以直接来自`std::vector`，后者不会按排序顺序存储其元素。即使迭代器来自我们自定义容器的一个实例，迭代器存储元素的顺序可能与容器元素的顺序不同。因此，我们必须在初始化后对`std::vector`进行排序。

除了构造之外，我们的自定义容器还必须包括`std::vector`提供的基于迭代器的别名，因为这些别名对于容器与 C++ API 的正确工作是必需的。以下是一个示例代码片段：

```cpp
    using const_iterator = typename vector_type::const_iterator;
    using const_reverse_iterator = typename vector_type::const_reverse_iterator;
```

正如前面的代码片段所示，与第一个示例中定义的别名一样，我们只需要前向声明`std::vector`已经提供的别名，以便我们的自定义容器也可以利用它们。不同之处在于，我们不包括这些迭代器别名的非 const 版本。由于我们的自定义容器必须始终保持有序，我们必须限制用户直接修改迭代器内容的能力，因为这可能导致更改容器元素的顺序，而我们的容器无法根据需要重新排序。相反，对容器的修改应通过使用`insert()`、`emplace()`和`erase()`来进行。

基于 C++模板的函数依赖于这些别名来正确实现它们的功能，这也包括基于范围的 for 循环。

最后，有一系列基于迭代器的成员函数，`std::vector`提供了这些函数，也应该通过我们的自定义容器进行转发。以下代码描述了这一点：

```cpp
    const_iterator begin() const noexcept
    {
        return m_v.begin();
    }

    const_iterator cbegin() const noexcept
    {
        return m_v.cbegin();
    }
```

第一组成员函数是`begin()`函数，它提供表示`std::vector`中第一个元素的迭代器。与别名一样，我们不转发这些成员函数的非 const 版本。此外，出于完整性考虑，我们包括这些函数的`c`版本。在 C++17 中，这些是可选的，如果愿意，可以使用`std::as_const()`代替。接下来的迭代器是`end()`迭代器，它提供表示`std::vector`末尾的迭代器（不要与表示`std::vector`中最后一个元素的迭代器混淆）。以下代码显示了这一点：

```cpp
    const_iterator end() const noexcept
    {
        return m_v.end();
    }

    const_iterator cend() const noexcept
    {
        return m_v.cend();
    }
```

正如前面的代码片段所示，与大多数这些成员函数一样，我们只需要将 API 转发到我们的自定义容器封装的私有`std::vector`。这个过程也可以重复用于`rbegin()`和`rend()`，它们提供与之前相同的 API，但返回一个反向迭代器，以相反的顺序遍历`std::vector`。

接下来，我们实现基于迭代器的`emplace()`函数，如下所示：

```cpp
    template <typename... Args>
    void emplace(const_iterator pos, Args&&... args)
    {
        m_v.emplace(pos, std::forward<Args>(args)...);
        std::sort(m_v.begin(), m_v.end(), compare_type());
    }
```

尽管提供`emplace()` API 提供了更完整的实现，但应该注意的是，只有在进一步优化以利用元素添加到容器的预期位置的方式时，它才会有用。这与更好地排序`std::vector`的方法相结合。

尽管前面的实现是有效的，但它可能与我们在第一个示例中实现的`emplace()`版本表现类似。由于自定义容器始终保持排序顺序，因此将元素插入`std::vector`的位置是无关紧要的，因为`std::vector`的新顺序将改变添加元素的位置。当然，除非位置参数的添加提供了一些额外的支持来更好地优化添加，而我们的实现没有这样做。因此，除非使用`pos`参数进行优化，前面的函数可能是多余且不必要的。

与前面的`emplace()`函数一样，我们不尝试返回表示添加到容器的元素的迭代器，因为在排序后，此迭代器将变为无效，并且关于添加到`std::vector`的内容的信息不足以重新定位迭代器（例如，如果存在重复项，则无法知道实际添加的是哪个元素）。

最后，我们实现了`erase`函数，如下所示：

```cpp
    const_iterator erase(const_iterator pos)
    {
        return m_v.erase(pos);
    }

    const_iterator erase(const_iterator first, const_iterator last)
    {
        return m_v.erase(first, last);
    }
```

与`emplace()`函数不同，从`std::vector`中移除元素不会改变`std::vector`的顺序，因此不需要排序。还应该注意的是，我们的`erase()`函数版本返回`const`版本。再次强调，这是因为我们无法支持迭代器的非 const 版本。

最后，现在我们有能力访问容器中存储的元素，让我们创建一些测试逻辑，以确保我们的容器按预期工作：

```cpp
int main(void)
{
    container<int> c{4, 42, 15, 8, 23, 16};
```

首先，我们将从不带顺序的整数初始化列表创建一个容器。创建完容器后，存储这些元素的`std::vector`应该是有序的。为了证明这一点，让我们循环遍历容器并输出结果：

```cpp
    std::cout << "elements: ";

    for (const auto &elem : c) {
        std::cout << elem << ' ';
    }

    std::cout << '\n';
```

如前面的代码片段所示，我们首先向`stdout`输出一个标签，然后使用范围 for 循环遍历我们的容器，逐个输出每个元素。最后，在所有元素都输出到`stdout`后，我们输出一个新行，导致以下输出：

```cpp
elements: 4 8 15 16 23 42
```

此输出按预期的顺序排序。

需要注意的是，我们的范围 for 循环必须将每个元素定义为`const`。这是因为我们不支持迭代器的非 const 版本。任何尝试使用这些迭代器的非 const 版本都会导致编译错误，如下例所示：

```cpp
    for (auto &elem : c) {
        elem = 42;
    }
```

上述代码将导致以下编译错误（这是预期的）：

```cpp
/home/user/book/chapter08/recipe03.cpp: In function ‘int main()’:
/home/user/book/chapter08/recipe03.cpp:396:14: error: assignment of read-only reference ‘elem’
  396 | elem = 42;
```

发生这种编译错误的原因是因为范围 for 循环也可以写成以下形式：

```cpp
    std::cout << "elements: ";

    for (auto iter = c.begin(); iter != c.end(); iter++) {
        auto &elem = *iter;
        std::cout << elem << ' ';
    }

    std::cout << '\n';
```

如前面的代码片段所示，元素未标记为`const`，因为范围 for 循环使用`begin()`和`end()`成员函数，导致读写迭代器（除非您明确声明为`const`）。

我们还可以为我们的新`emplace()`函数创建一个测试，如下所示：

```cpp
    c.emplace(c.cend(), 1);

    std::cout << "elements: ";
    for (const auto &elem : c) {
        std::cout << elem << ' ';
    }
    std::cout << '\n';
```

这将产生以下输出：

```cpp
elements: 1 4 8 15 16 23 42
```

如前面的输出所示，数字`1`按预期的顺序被添加到我们的容器中，即使我们告诉容器将我们的元素添加到`std::vector`的末尾。

我们还可以反转上述操作并验证我们的`erase()`函数是否正常工作，如下所示：

```cpp
    c.erase(c.cbegin());

    std::cout << "elements: ";
    for (const auto &elem : c) {
        std::cout << elem << ' ';
    }
    std::cout << '\n';
```

这将产生以下输出：

```cpp
elements: 4 8 15 16 23 42
```

如您所见，新添加的`1`已成功被移除。

# 添加 std::vector API 的相关部分

在本文中，我们将通过添加`std::vector`已经提供的剩余 API 来完成我们在本章前三个示例中构建的自定义容器。在此过程中，我们将删除不合理的 API，或者我们无法支持的 API，因为我们的自定义容器必须保持`std::vector`中的元素有序。

本文很重要，因为它将向您展示如何正确创建一个包装容器，该容器可用于封装现有容器的逻辑（例如，线程安全，或者在我们的情况下，元素顺序）。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本文示例所需的适当工具。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤尝试本文：

1.  从新的终端运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter08
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  源代码编译完成后，可以通过运行以下命令来执行本文中的每个示例：

```cpp
> ./recipe04_example01 
elements: 4 8 15 16 23 42 
elements: 4 8 15 16 23 42 
elements: 4 8 15 16 23 42 
elements: 42 
elements: 4 8 15 16 23 42 
elements: 4 8 15 16 23 42 
c1.at(0): 4
c1.front(): 4
c1.back(): 42
c1.data(): 0xc01eb0
c1.empty(): 0
c1.size(): 6
c1.max_size(): 2305843009213693951
c1.capacity(): 42
c1.capacity(): 6
c1.size(): 0
c1.size(): 42
c1.size(): 0
c1.size(): 42
elements: 4 8 15 16 23 
==: 0
!=: 1
 <: 1
<=: 1
 >: 0
>=: 0
```

在接下来的部分中，我们将逐个介绍每个示例，并解释每个示例程序的作用以及它与本文教授的课程的关系。

# 工作原理...

目前，我们的自定义容器能够被构建、添加、迭代和擦除。然而，该容器不支持直接访问容器或支持简单操作，比如`std::move()`或比较。为了解决这些问题，让我们首先添加缺失的`operator=()`重载：

```cpp
    constexpr container &operator=(const container &other)
    {
        m_v = other.m_v;
        return *this;
    }

    constexpr container &operator=(container &&other) noexcept
    {
        m_v = std::move(other.m_v);
        return *this;
    }    
```

第一个`operator=()`重载支持复制赋值，而第二个重载支持移动赋值。由于我们只有一个提供适当复制和移动语义的私有成员变量，我们不需要担心自赋值（或移动），因为`std::vector`函数的复制和移动实现会为我们处理这个问题。

如果您自己的自定义容器有额外的私有元素，可能需要进行自赋值检查。例如，考虑以下代码：

```cpp
    constexpr container &operator=(container &&other) noexcept
    {
        if (&other == this) {
            return *this;
        }

        m_v = std::move(other.m_v);
        m_something = other.m_something;

        return *this;
    }
```

剩下的`operator=()`重载接受一个初始化列表，如下所示：

```cpp
    constexpr container &operator=(std::initializer_list<T> list)
    {
        m_v = list;
        std::sort(m_v.begin(), m_v.end(), compare_type());

        return *this;
    }
```

在上面的代码片段中，与初始化列表构造函数一样，我们必须在赋值后重新排序`std::vector`，因为初始化列表可以以任何顺序提供。

要实现的下一个成员函数是`assign()`函数。以下代码片段显示了这一点：

```cpp
    constexpr void assign(size_type count, const T &value)
    {
        m_v.assign(count, value);
    }

    template <typename Iter>
    constexpr void assign(Iter first, Iter last)
    {
        m_v.assign(first, last);
        std::sort(m_v.begin(), m_v.end(), compare_type());
    }

    constexpr void assign(std::initializer_list<T> list)
    {
        m_v.assign(list);
        std::sort(m_v.begin(), m_v.end(), compare_type());
    }
```

这些函数类似于`operator=()`重载，但不提供返回值或支持其他功能。让我们看看：

+   第一个`assign()`函数用特定的`value`次数填充`std::vector`。由于值永远不会改变，`std::vector`将始终按排序顺序排列，在这种情况下，不需要对列表进行排序。

+   第二个`assign()`函数接受与构造函数版本相似的迭代器范围。与该函数类似，传递给此函数的迭代器可以来自原始`std::vector`或我们自定义容器的另一个实例，但排序顺序不同。因此，我们必须在赋值后对`std::vector`进行排序。

+   最后，`assign()`函数还提供了与我们的`operator=()`重载相同的初始化列表版本。

还应该注意到，我们已经为每个函数添加了`constexpr`。这是因为我们自定义容器中的大多数函数只是将调用从自定义容器转发到`std::vector`，并且在某些情况下调用`std::sort()`。添加`constexpr`告诉编译器将代码视为编译时表达式，使其能够在启用优化时（如果可能）优化掉额外的函数调用，确保我们的自定义包装器具有尽可能小的开销。

过去，这种优化是使用`inline`关键字执行的。在 C++11 中添加的`constexpr`不仅能够向编译器提供`inline`提示，还告诉编译器这个函数可以在编译时而不是运行时使用（这意味着编译器可以在代码编译时执行函数以执行自定义的编译时逻辑）。然而，在我们的例子中，`std::vector`的运行时使用是不可能的，因为需要分配。因此，使用`constexpr`只是为了优化，在大多数编译器上，`inline`关键字也会提供类似的好处。

`std::vector`还支持许多其他函数，例如`get_allocator()`、`empty()`、`size()`和`max_size()`，所有这些都只是直接转发。让我们专注于直到现在为止从我们的自定义容器中缺失的访问器：

```cpp
    constexpr const_reference at(size_type pos) const
    {
        return m_v.at(pos);
    }
```

我们提供的第一个直接访问`std::vector`的函数是`at()`函数。与我们的大多数成员函数一样，这是一个直接转发。但与`std::vector`不同的是，我们没有计划添加`std::vector`提供的`operator[]()`重载。`at()`函数和`operator[]()`重载之间的区别在于，`operator[]()`不会检查提供的索引是否在范围内（也就是说，它不会访问`std::vector`范围之外的元素）。

`operator[]()`重载的设计类似于标准 C 数组。这个运算符（称为下标运算符）的问题在于缺乏边界检查，这为可靠性和安全性错误进入程序打开了大门。因此，C++核心指南不鼓励使用下标运算符或任何其他形式的指针算术（任何试图通过指针计算数据位置而没有显式边界检查的东西）。

为了防止使用`operator[]()`重载，我们不包括它。

像`std::vector`一样，我们也可以添加`front()`和`back()`访问器，如下所示：

```cpp
    constexpr const_reference front() const
    {
        return m_v.front();
    }

    constexpr const_reference back() const
    {
        return m_v.back();
    }
```

前面的额外访问器支持获取我们的`std::vector`中的第一个和最后一个元素。与`at()`函数一样，我们只支持`std::vector`已经提供的这些函数的`const_reference`版本的使用。

现在让我们看一下`data()`函数的代码片段：

```cpp
    constexpr const T* data() const noexcept
    {
        return m_v.data();
    }
```

`data()`函数也是一样的。我们只能支持这些成员函数的`const`版本，因为提供这些函数的非 const 版本将允许用户直接访问`std::vector`，从而使他们能够插入无序数据，而容器无法重新排序。

现在让我们专注于比较运算符。我们首先定义比较运算符的原型，作为我们容器的友元。这是必要的，因为比较运算符通常被实现为非成员函数，因此需要对容器进行私有访问，以比较它们包含的`std::vector`实例。

例如，考虑以下代码片段：

```cpp
    template <typename O, typename Alloc>
    friend constexpr bool operator==(const container<O, Alloc> &lhs,
                                     const container<O, Alloc> &rhs);

    template <typename O, typename Alloc>
    friend constexpr bool operator!=(const container<O, Alloc> &lhs,
                                     const container<O, Alloc> &rhs);

    template <typename O, typename Alloc>
    friend constexpr bool operator<(const container<O, Alloc> &lhs,
                                    const container<O, Alloc> &rhs);

    template <typename O, typename Alloc>
    friend constexpr bool operator<=(const container<O, Alloc> &lhs,
                                     const container<O, Alloc> &rhs);

    template <typename O, typename Alloc>
    friend constexpr bool operator>(const container<O, Alloc> &lhs,
                                    const container<O, Alloc> &rhs);

    template <typename O, typename Alloc>
    friend constexpr bool operator>=(const container<O, Alloc> &lhs,
                                     const container<O, Alloc> &rhs);
```

最后，我们按照以下方式实现比较运算符：

```cpp
template <typename O, typename Alloc>
bool constexpr operator==(const container<O, Alloc> &lhs,
                          const container<O, Alloc> &rhs)
{
    return lhs.m_v == rhs.m_v;
}

template <typename O, typename Alloc>
bool constexpr operator!=(const container<O, Alloc> &lhs,
                          const container<O, Alloc> &rhs)
{
    return lhs.m_v != rhs.m_v;
}
```

与成员函数一样，我们只需要将调用转发到`std::vector`，因为没有必要实现自定义逻辑。剩下的比较运算符也是一样。

例如，我们可以按照以下方式实现`>`、`<`、`>=`和`<=`比较运算符：

```cpp
template <typename O, typename Alloc>
bool constexpr operator<(const container<O, Alloc> &lhs,
                         const container<O, Alloc> &rhs)
{
    return lhs.m_v < rhs.m_v;
}

template <typename O, typename Alloc>
bool constexpr operator<=(const container<O, Alloc> &lhs,
                          const container<O, Alloc> &rhs)
{
    return lhs.m_v <= rhs.m_v;
}

template <typename O, typename Alloc>
bool constexpr operator>(const container<O, Alloc> &lhs,
                         const container<O, Alloc> &rhs)
{
    return lhs.m_v > rhs.m_v;
}

template <typename O, typename Alloc>
bool constexpr operator>=(const container<O, Alloc> &lhs,
                          const container<O, Alloc> &rhs)
{
    return lhs.m_v >= rhs.m_v;
}
```

就是这样！这就是通过利用现有容器来实现自己的容器的方法。

正如我们所看到的，在大多数情况下，除非你需要的容器无法使用 C++标准模板库已经提供的容器来实现，否则没有必要从头开始实现一个容器。

使用这种方法，不仅可以创建自己的容器，更重要的是可以将代码中重复的功能封装到一个单独的容器中，这样可以独立测试和验证。这不仅提高了应用程序的可靠性，而且还使其更易于阅读和维护。

在下一章中，我们将探讨如何在 C++中使用智能指针。
