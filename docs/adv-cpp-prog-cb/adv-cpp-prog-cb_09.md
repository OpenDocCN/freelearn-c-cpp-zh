# 第九章：探索类型擦除

在本章中，您将学习类型擦除（也称为类型擦除）是什么，以及如何在自己的应用程序中使用它。本章很重要，因为类型擦除提供了在不需要对象共享公共基类的情况下使用不同类型对象的能力。

本章从简单解释类型擦除开始，解释了在 C 语言中类型擦除的工作原理，以及如何在 C++中使用继承来执行类型擦除。下一个示例将提供使用 C++模板的不同方法来进行类型擦除，这将教会您如何使用 C++概念来定义类型的规范，而不是类型本身。

接下来，我们将学习经典的 C++类型擦除模式。本示例将教会您擦除类型信息的技能，从而能够创建类型安全的通用代码。最后，我们将通过一个全面的示例来结束，该示例使用类型擦除来实现委托模式，这是一种提供包装任何类型的可调用对象的能力的模式，并且被诸如 ObjC 等语言广泛使用。

本章的示例如下：

+   如何使用继承来擦除类型

+   使用 C++模板编写通用函数

+   学习 C++类型擦除模式

+   实现委托模式

# 技术要求

要编译和运行本章中的示例，您必须具有对运行 Ubuntu 18.04 的计算机的管理访问权限，并且具有正常的互联网连接。在运行这些示例之前，您必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake
```

如果这安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter09`](https://github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter09)找到。

# 如何使用继承来擦除类型

在本示例中，我们将学习如何使用继承来擦除类型。当讨论类型擦除时，通常不考虑继承，但实际上，继承是 C++中最常见的类型擦除形式。本示例很重要，因为它将讨论类型擦除是什么，以及为什么它在日常应用中非常有用，而不仅仅是简单地移除类型信息——这在 C 中很常见。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本示例中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

让我们尝试按照以下步骤进行本示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter09
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe01_example01 
1
0
```

在接下来的部分，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

类型擦除（或类型擦除）简单地是移除、隐藏或减少有关对象、函数等的类型信息。在 C 语言中，类型擦除经常被使用。看看这个例子：

```cpp
int array[10];
memset(array, 0, sizeof(array));
```

在上面的例子中，我们创建了一个包含`10`个元素的数组，然后使用`memset()`函数将数组清零。在 C 中，`memset()`函数看起来像这样：

```cpp
void *memset(void *ptr, int value, size_t num)
{
    size_t i;
    for (i = 0; i < num; i++) {
        ((char *)ptr)[i] = value;    
    }

    return ptr;
}
```

在上面的代码片段中，`memset()`函数的第一个参数是`void*`。然而，在我们之前的例子中，数组是一个整数数组。`memset()`函数实际上并不关心你提供的是什么类型，只要你提供了指向该类型的指针和表示该类型总字节数的大小。然后，`memset()`函数将提供的指针强制转换为表示字节的类型（在 C 中通常是`char`或无符号`char`），然后逐字节设置类型的值。

在 C 中使用`void*`是一种类型擦除的形式。在 C++中，这种类型（双关语）的擦除通常是不鼓励的，因为要恢复类型信息的唯一方法是使用`dynamic_cast()`，这很慢（需要运行时类型信息查找）。尽管有许多种方法可以在 C++中执行类型擦除而不需要`void*`，让我们专注于继承。

继承在大多数文献中通常不被描述为类型擦除，但它很可能是最广泛使用的形式之一。为了更好地探讨这是如何工作的，让我们看一个常见的例子。假设我们正在创建一个游戏，其中用户可以选择多个超级英雄。每个超级英雄在某个时候都必须攻击坏家伙，但超级英雄如何攻击坏家伙因英雄而异。

例如，考虑以下代码片段：

```cpp
class spiderman
{
public:
    bool attack(int x, int) const
    {
        return x == 0 ? true : false;
    }
};
```

如上所示，在我们的第一个英雄中，不关心坏家伙是在地面上还是在空中（也就是说，无论坏家伙的垂直距离如何，英雄都能成功击中坏家伙），但如果坏家伙不在特定的水平位置，英雄就会错过坏家伙。同样，我们可能还有另一个英雄如下：

```cpp
class captain_america
{
public:
    bool attack(int, int y) const
    {
        return y == 0 ? true : false;
    }
};
```

第二个英雄与我们的第一个完全相反。这个英雄可以成功地击中地面上的坏家伙，但如果坏家伙在地面以上的任何地方，他就会错过（英雄可能无法到达他们）。

在下面的例子中，两个超级英雄同时与坏家伙战斗：

```cpp
    for (const auto &h : heroes) {
        std::cout << h->attack(0, 42) << '\n';
    }
```

虽然我们可以在战斗中一个一个地召唤每个超级英雄，但如果我们可以只循环遍历每个英雄并检查哪个英雄击中了坏家伙，哪个英雄错过了坏家伙，那将更加方便。

在上面的例子中，我们有一个假想的英雄数组，我们循环遍历，检查哪个英雄击中了，哪个英雄错过了。在这个例子中，我们不关心英雄的类型（也就是说，我们不关心英雄是否特别是我们的第一个还是第二个英雄），我们只关心每个英雄实际上是一个英雄（而不是一个无生命的物体），并且英雄能够攻击坏家伙。换句话说，我们需要一种方法来擦除每个超级英雄的类型，以便我们可以将两个英雄放入单个数组中（除非每个英雄都是相同的，否则这是不可能的）。

正如你可能已经猜到的那样，在 C++中实现这一点的最常见方法是使用继承（但正如我们将在本章后面展示的那样，这并不是唯一的方法）。首先，我们必须定义一个名为`hero`的基类，每个英雄都将从中继承，如下所示：

```cpp
class hero
{
public:
    virtual ~hero() = default;
    virtual bool attack(int, int) const = 0;
};
```

在我们的例子中，每个英雄之间唯一的共同函数是它们都可以攻击坏家伙，`attack()`函数对所有英雄都是相同的。因此，我们创建了一个纯虚基类，其中包含一个名为`attack()`的单个纯虚函数，每个英雄都必须实现。还应该注意的是，为了使一个类成为纯虚类，所有成员函数必须设置为`0`，并且类的析构函数必须显式标记为`virtual`。

现在我们已经定义了什么是英雄，我们可以修改我们的英雄，使其继承这个纯虚基类，如下所示：

```cpp
class spiderman : public hero
{
public:
    bool attack(int x, int) const override
    {
        return x == 0 ? true : false;
    }
};

class captain_america : public hero
{
public:
    bool attack(int, int y) const override
    {
        return y == 0 ? true : false;
    }
};
```

如上所示，两个英雄都继承了英雄的纯虚定义，并根据需要重写了`attack()`函数。通过这种修改，我们现在可以按以下方式创建我们的英雄列表：

```cpp
int main(void)
{
    std::array<std::unique_ptr<hero>, 2> heros {
        std::make_unique<spiderman>(),
        std::make_unique<captain_america>()
    };

    for (const auto &h : heros) {
        std::cout << h->attack(0, 42) << '\n';
    }

    return 0;
}
```

从上面的代码中，我们观察到以下内容：

+   我们创建了一个`hero`指针数组（使用`std::unique_ptr`来存储英雄的生命周期，这是下一章将讨论的一个主题）。

+   然后，该数组被初始化为包含两个英雄（每个英雄一个）。

+   最后，我们循环遍历每个英雄，看英雄是否成功攻击坏人或者错过。

+   当调用`hero::attack()`函数时，调用会自动路由到正确的`spiderman::attack()`和`captain_america::attack()`函数，通过继承来实现。

该数组以类型安全的方式擦除了每个英雄的类型信息，将每个英雄放入单个容器中。

# 使用 C++模板编写通用函数

在本示例中，我们将学习如何使用 C++模板来擦除（或忽略）类型信息。您将学习如何使用 C++模板来实现 C++概念，以及这种类型擦除在 C++标准库中的使用。这个示例很重要，因为它将教会您如何更好地设计您的 API，使其不依赖于特定类型（或者换句话说，如何编写通用代码）。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本示例中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

让我们按照以下步骤尝试这个示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter09
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe02_examples
```

1.  源代码编译后，可以通过运行以下命令来执行本文中的每个示例：

```cpp
> ./recipe02_example01 
hero won fight
hero lost the fight :(
```

在接下来的部分中，我们将逐个步骤地介绍每个示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

C++最古老和最广泛使用的功能之一是 C++模板。与继承一样，C++模板通常不被描述为一种类型擦除，但它们实际上是。类型擦除只不过是删除或在这种情况下忽略类型信息的行为。

然而，与 C 语言不同，C++中的类型擦除通常试图避免删除类型信息，而是绕过类型的严格定义，同时保留类型安全。实现这一点的一种方法是通过使用 C++模板。为了更好地解释这一点，让我们从一个 C++模板的简单示例开始：

```cpp
template<typename T>
T pow2(T t)
{
    return t * t;
}
```

在上面的示例中，我们创建了一个简单的函数，用于计算任何给定输入的平方。例如，我们可以这样调用这个函数：

```cpp
std::cout << pow2(42U) << '\n'
std::cout << pow2(-1) << '\n'
```

当编译器看到`pow2()`函数的使用时，它会在幕后自动生成以下代码：

```cpp
unsigned pow2(unsigned t)
{
    return t * t;
}

int pow2(int t)
{
    return t * t;
}
```

在上面的代码片段中，编译器创建了`pow2()`函数的两个版本：一个接受无符号值并返回无符号值，另一个接受整数并返回整数。编译器创建了这两个版本，是因为我们第一次使用`pow2()`函数时，我们提供了一个无符号值，而第二次使用`pow2()`函数时，我们提供了`int`。

就我们的代码而言，我们实际上并不关心函数提供的类型是什么，只要提供的类型能够成功执行`operator*()`。换句话说，`pow2()`函数的使用者和`pow2()`函数的作者都安全地忽略（或擦除）了从概念上传递给函数的类型信息。然而，编译器非常清楚正在提供的类型，并且必须根据需要安全地处理每种类型。

这种类型擦除形式在 API 的规范处执行擦除，在 C++中，这种规范被称为概念。与大多数 API 不同，后者规定了输入和输出类型（例如，`sleep()`函数接受一个无符号整数，只接受无符号整数），概念特别忽略类型，而是定义了给定类型必须提供的属性。

例如，前面的`pow2()`函数有以下要求：

+   提供的类型必顺要么是整数类型，要么提供`operator *()`。

+   提供的类型必须是可复制构造或可移动构造的。

如前面的代码片段所示，`pow2()`函数不关心它所接收的类型，只要所提供的类型满足一定的最小要求。让我们来看一个更复杂的例子，以演示 C++模板如何被用作类型擦除的一种形式。假设我们有两个不同的英雄在与一个坏家伙战斗，每个英雄都提供了攻击坏家伙的能力，如下所示：

```cpp
class spiderman
{
public:
    bool attack(int x, int) const
    {
        return x == 0 ? true : false;
    }
};

class captain_america
{
public:
    bool attack(int, int y) const
    {
        return y == 0 ? true : false;
    }
};
```

如前面的代码片段所示，每个英雄都提供了攻击坏家伙的能力，但除了两者都提供具有相同函数签名的`attack()`函数之外，两者没有任何共同之处。我们也无法为每个英雄添加继承（也许我们的设计无法处理继承所增加的额外`vTable`开销，或者英雄定义是由其他人提供的）。

现在假设我们有一个复杂的函数，必须为每个英雄调用`attack()`函数。我们可以为每个英雄编写相同的逻辑（即手动复制逻辑），或者我们可以编写一个 C++模板函数来处理这个问题，如下所示：

```cpp
template<typename T>
auto attack(const T &t, int x, int y)
{
    if (t.attack(x, y)) {
        std::cout << "hero won fight\n";
    }
    else {
        std::cout << "hero lost the fight :(\n";
    }
}
```

如前面的代码片段所示，我们可以利用 C++模板的类型擦除特性，将我们的攻击逻辑封装到一个单一的模板函数中。前面的代码不关心所提供的类型是什么，只要该类型提供了一个接受两个整数类型并返回一个整数类型（最好是`bool`，但任何整数都可以）的`attack()`函数。换句话说，只要所提供的类型符合约定的概念，这个模板函数就会起作用，为编译器提供一种处理类型特定逻辑的方法。

我们可以按照以下方式调用前面的函数：

```cpp
int main(void)
{
    attack(spiderman{}, 0, 42);
    attack(captain_america{}, 0, 42);

    return 0;
}
```

这将产生以下输出：

![](img/70426f57-68a7-48bf-ac42-6ee95388297b.png)

尽管这个示例展示了 C++模板如何被用作类型擦除的一种形式（至少用于创建概念的规范），但是当讨论类型擦除时，有一种特定的模式称为类型擦除模式或者只是类型擦除。在下一个示例中，我们将探讨如何利用我们在前两个示例中学到的知识来擦除类型信息，同时仍然支持诸如容器之类的简单事物。

# 还有更多...

在这个示例中，我们学习了如何使用概念来忽略（或擦除）特定类型的知识，而是要求类型实现一组最小的特性。这些特性可以使用 SFINAE 来强制执行，这是我们在第四章中更详细讨论的一个主题，*使用模板进行通用编程*。

# 另请参阅

在第十三章中，*奖励-使用 C++20 功能*，我们还将讨论如何使用 C++20 新增的功能来执行概念的强制执行。

# 学习 C++类型擦除模式

在本菜谱中，我们将学习 C++中类型擦除模式是什么，以及我们如何利用它来通用地擦除类型信息，而不会牺牲类型安全性或要求我们的类型继承纯虚拟基类。这个菜谱很重要，因为类型擦除模式在 C++标准库中被大量使用，并提供了一种简单的方式来封装不共享任何共同之处的数据类型，除了提供一组类似的 API，同时还支持诸如容器之类的东西。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本菜谱中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

让我们尝试以下步骤来制作这个菜谱：

1.  从一个新的终端中，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter09
```

1.  编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe03_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本菜谱中的每个示例：

```cpp
> ./recipe03_example01 
1
0
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用，以及它与本菜谱中所教授的课程的关系。

# 工作原理...

当我们通常考虑 C++类型擦除时，这就是我们想到的例子。当我们必须将一组对象视为相关对象使用时，可能并不共享一个共同的基类（也就是说，它们要么不使用继承，要么如果使用继承，可能它们不继承自相同的一组类）时，就需要类型擦除模式。

例如，假设我们有以下类：

```cpp
class spiderman
{
public:
    bool attack(int x, int) const
    {
        return x == 0 ? true : false;
    }
};

class captain_america
{
public:
    bool attack(int, int y) const
    {
        return y == 0 ? true : false;
    }
};
```

如前面的代码片段所示，每个类定义了不同类型的英雄。我们想要做的事情如下：

```cpp
for (const auto &h : heros) {
    // something
}
```

问题是，每个类都不继承自相似的基类，所以我们不能只创建每个类的实例并将它们添加到`std::array`中，因为编译器会抱怨这些类不相同。我们可以在`std::array`中存储每个类的原始`void *`指针，但是当使用`void *`时，我们将不得不使用`dynamic_cast()`来将其转换回每种类型以执行任何有用的操作，如下所示：

```cpp
    std::array<void *, 2> heros {
        new spiderman,
        new captain_america
    };

    for (const auto &h : heros) {
        if (ptr = dynamic_cast<spiderman>(ptr)) {
            // something
        }

        if (ptr = dynamic_cast<captain_america>(ptr)) {
            // something
        }
    }
```

使用`void *`是一种类型擦除的形式，但这远非理想，因为使用`dynamic_cast()`很慢，每添加一种新类型都只会增加`if`语句的数量，而且这种实现远非符合 C++核心指南。

然而，还有另一种方法可以解决这个问题。假设我们希望运行`attack()`函数，这个函数在每个英雄类之间是相同的（也就是说，每个英雄类至少遵循一个共享概念）。如果每个类都使用了以下基类，我们可以使用继承，如下所示：

```cpp
class base
{
public:
    virtual ~base() = default;
    virtual bool attack(int, int) const = 0;
};
```

问题是，我们的英雄类没有继承这个基类。因此，让我们创建一个继承它的包装器类，如下所示：

```cpp
template<typename T>
class wrapper :
    public base
{
    T m_t;

public:
    bool attack(int x, int y) const override
    {
        return m_t.attack(x, y);
    }
};
```

如前面的代码片段所示，我们创建了一个模板包装类，它继承自我们的基类。这个包装器存储给定类型的实例，然后覆盖了在纯虚拟基类中定义的`attack()`函数，该函数将调用转发给包装器存储的实例。

现在，我们可以创建我们的数组，如下所示：

```cpp
    std::array<std::unique_ptr<base>, 2> heros {
        std::make_unique<wrapper<spiderman>>(),
        std::make_unique<wrapper<captain_america>>()
    };
```

`std::array`存储了指向我们基类的`std::unique_ptr`，然后我们使用每种需要的类型创建我们的包装器类（它继承自基类），以存储在数组中。编译器为我们需要存储在数组中的每种类型创建了包装器的版本，由于包装器继承了基类，无论我们给包装器什么类型，数组总是可以按需存储结果包装器。

现在，我们可以从这个数组中执行以下操作：

```cpp
    for (const auto &h : heros) {
        std::cout << h->attack(0, 42) << '\n';
    }
```

就是这样：C++中的类型擦除。这种模式利用 C++模板，即使对象本身没有直接使用继承，也可以给对象赋予继承的相同属性。

# 使用类型擦除实现委托

在这个示例中，我们将学习如何实现委托模式，这是一个已经存在多年的模式（并且被一些其他语言，比如 ObjC，广泛使用）。这个示例很重要，因为它将教会你什么是委托，以及如何在你自己的应用程序中利用这种模式，以提供更好的可扩展性，而不需要你的 API 使用继承。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例中的示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

让我们按照以下步骤尝试这个示例：

1.  从一个新的终端中，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter09
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令执行本示例中的每个示例：

```cpp
> ./recipe04_example01
1
0

> ./recipe04_example02
1
0

> ./recipe04_example03
1
0

> ./recipe04_example04
0
1
0
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 它是如何工作的...

如果你曾经读过一本关于 C++的书，你可能已经看过苹果和橙子的例子，它演示了面向对象编程的工作原理。思路如下：

+   苹果是一种水果。

+   橙子是一种水果。

+   苹果不是橙子，但两者都是水果。

这个例子旨在教你如何使用继承将代码组织成逻辑对象。一个苹果和一个橙子共享的逻辑被写入一个叫做`fruit`的对象中，而特定于苹果或橙子的逻辑被写入继承自基类`fruit`的`apple`或`orange`对象中。

这个例子也展示了如何扩展水果的功能。通过对水果进行子类化，我可以创建一个苹果，它能够做比`fruit`基类更多的事情。这种*扩展*类功能的想法在 C++中很常见，通常我们会考虑使用继承来实现它。在这个示例中，我们将探讨如何在不需要苹果或橙子使用继承的情况下实现这一点，而是使用一种称为委托的东西。

假设你正在创建一个游戏，并希望实现一个英雄和坏人在战斗中战斗的战场。在代码的某个地方，战斗中的每个英雄都需要攻击坏人。问题是英雄在战斗中来来去去，因为他们需要时间恢复，所以你真的需要维护一个能够攻击坏人的英雄列表，并且你只需要循环遍历这个动态变化的英雄列表，看看他们的攻击是否成功。

每个英雄都可以存储一个子类化共同基类的英雄列表，然后运行一个`attack()`函数，每个英雄都会重写，但这将需要使用继承，这可能不是期望的。我们也可以使用类型擦除模式来包装每个英雄，然后存储指向我们包装器的基类的指针，但这将特定于我们的`attack()`函数，并且我们相信将需要其他这些类型的扩展的情况。

进入委托模式，这是类型擦除模式的扩展。使用委托模式，我们可以编写如下代码：

```cpp
int main(void)
{
    spiderman s;
    captain_america c;

    std::array<delegate<bool(int, int)>, 3> heros {
        delegate(attack),
        delegate(&s, &spiderman::attack),
        delegate(&c, &captain_america::attack)
    };

    for (auto &h : heros) {
        std::cout << h(0, 42) << '\n';
    }

    return 0;
}
```

如前面的代码片段所示，我们定义了两个不同的类的实例，然后创建了一个存储三个委托的数组。委托的模板参数采用`bool(int, int)`的函数签名，而委托本身似乎是从函数指针以及我们之前创建的类实例的两个成员函数指针创建的。然后我们能够循环遍历每个委托并调用它们，有效地独立调用函数指针和每个成员函数指针。

委托模式提供了将不同的可调用对象封装到一个具有共同类型的单个对象中的能力，该对象能够调用可调用对象，只要它们共享相同的函数签名。更重要的是，委托可以封装函数指针和成员函数指针，为 API 的用户提供了必要时存储私有状态的能力。

为了解释这是如何工作的，我们将从简单的开始，然后逐步构建我们的示例，直到达到最终实现。让我们从一个基类开始：

```cpp
template<
    typename RET,
    typename... ARGS
    >
class base
{
public:
    virtual ~base() = default;
    virtual RET func(ARGS... args) = 0;
};
```

如前面的代码片段所示，我们创建了一个纯虚基类的模板。模板参数是`RET`（定义返回值）和`ARGS...`（定义可变参数列表）。然后我们创建了一个名为`func()`的函数，它接受我们的参数列表并返回模板返回类型。

接下来，让我们定义一个从基类继承的包装器，使用类型擦除模式（如果您还没有阅读之前的示例，请现在阅读）：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
class wrapper :
    public base<RET, ARGS...>
{
    T m_t{};
    RET (T::*m_func)(ARGS...);

public:

    wrapper(RET (T::*func)(ARGS...)) :
        m_func{func}
    { }

    RET func(ARGS... args) override
    {
        return std::invoke(m_func, &m_t, args...);
    }
};
```

就像类型擦除模式一样，我们有一个包装器类，它存储我们的类型的实例，然后提供包装器可以调用的函数。不同之处在于可以调用的函数不是静态定义的，而是由提供的模板参数定义的。此外，我们还存储具有相同函数签名的函数指针，该函数指针由包装器的构造函数初始化，并在`func()`函数中使用`std::invoke`调用。

与典型的类型擦除示例相比，这个额外的逻辑提供了定义我们希望从我们在包装器中存储的对象中调用的任何函数签名的能力，而不是提前定义（意味着我们希望调用的函数可以在运行时而不是编译时确定）。

然后我们可以创建我们的委托类如下：

```cpp
template<
    typename RET,
    typename... ARGS
    >
class delegate
{
    std::unique_ptr<base<RET, ARGS...>> m_wrapper;

public:

    template<typename T>
    delegate(RET (T::*func)(ARGS...)) :
        m_wrapper{
            std::make_unique<wrapper<T, RET, ARGS...>>(func)
        }
    { }

    RET operator()(ARGS... args)
    {
        return m_wrapper->func(args...);
    }
};
```

与类型擦除模式一样，我们将指针存储在包装器中，该包装器是从委托的构造函数中创建的。要注意的重要细节是`T`类型在委托本身中未定义。相反，`T`类型仅在创建委托时才知道，用于创建包装器的实例。这意味着每个委托实例都是相同的，即使委托存储了包装不同类型的包装器。这使我们可以像下面这样使用委托。

假设我们有两个英雄，它们没有共同的基类，但提供了相同签名的`attack()`函数：

```cpp
class spiderman
{
public:
    bool attack(int x, int)
    {
        return x == 0 ? true : false;
    }
};

class captain_america
{
public:
    bool attack(int, int y)
    {
        return y == 0 ? true : false;
    }
};
```

我们可以利用我们的委托类来存储我们的英雄类的实例，并调用它们的攻击函数如下：

```cpp
int main(void)
{
    std::array<delegate<bool, int, int>, 2> heros {
        delegate(&spiderman::attack),
        delegate(&captain_america::attack)
    };

    for (auto &h : heros) {
        std::cout << h(0, 42) << '\n';
    }

    return 0;
}
```

这导致以下输出：

![](img/36666375-3829-4923-ab93-fc4ef67966c3.png)

尽管我们已经在创建我们的委托中取得了重大进展（它至少可以工作），但这个早期实现还存在一些问题：

+   委托的签名是`bool, int, int`，这是误导性的，因为我们真正想要的是一个函数签名，比如`bool(int, int)`，这样代码就是自说明的（委托的类型是单个函数签名，而不是三种不同的类型）。

+   这个委托不能处理标记为`const`的函数。

+   我们必须在包装器内部存储被委托对象的实例，这样我们就无法为同一对象创建多个函数的委托。

+   我们不支持非成员函数。

让我们逐个解决这些问题。

# 向我们的代理添加函数签名

尽管在不需要 C++17 的情况下可以向我们的代理添加函数签名作为模板参数，但是 C++17 中的用户定义类型推导使这个过程变得简单。以下代码片段展示了这一点：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
delegate(RET(T::*)(ARGS...)) -> delegate<RET(ARGS...)>;
```

如前所示的代码片段显示，用户定义的类型推导告诉编译器如何将我们的代理构造函数转换为我们希望使用的模板签名。没有这个用户定义的类型推导指南，`delegate(RET(T::*)(ARGS...))`构造函数将导致代理被推断为`delegate<RET, ARGS...>`，这不是我们想要的。相反，我们希望编译器推断`delegate<RET(ARGS...)>`。我们的代理实现的其他方面都不需要改变。我们只需要告诉编译器如何执行类型推断。

# 向我们的代理添加 const 支持

我们的代理目前无法接受标记为`const`的成员函数，因为我们没有为我们的代理提供能够这样做的包装器。例如，我们英雄的`attack()`函数目前看起来像这样：

```cpp
class spiderman
{
public:
    bool attack(int x, int)
    {
        return x == 0 ? true : false;
    }
};
```

然而，我们希望我们的英雄`attack()`函数看起来像以下这样，因为它们不修改任何私有成员变量：

```cpp
class spiderman
{
public:
    bool attack(int x, int) const
    {
        return x == 0 ? true : false;
    }
};
```

为了支持这个改变，我们必须创建一个支持这一点的包装器，如下所示：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
class wrapper_const :
    public base<RET, ARGS...>
{
    T m_t{};
    RET (T::*m_func)(ARGS...) const;

public:

    wrapper_const(RET (T::*func)(ARGS...) const) :
        m_func{func}
    { }

    RET func(ARGS... args) override
    {
        return std::invoke(m_func, &m_t, args...);
    }
};
```

如前所示，这个包装器与我们之前的包装器相同，不同之处在于我们存储的函数签名具有额外的`const`实例。为了使代理使用这个额外的包装器，我们还必须提供另一个代理构造函数，如下所示：

```cpp
    template<typename T>
    delegate(RET (T::*func)(ARGS...) const) :
        m_wrapper{
            std::make_unique<wrapper_const<T, RET, ARGS...>>(func)
        }
    { }
```

这意味着我们还需要另一个用户定义的类型推导指南，如下所示：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
delegate(RET(T::*)(ARGS...) const) -> delegate<RET(ARGS...)>;
```

通过这些修改，我们现在可以支持标记为`const`的成员函数。

# 向我们的代理添加一对多的支持

目前，我们的包装器存储每种类型的实例。这种方法通常与类型擦除一起使用，但在我们的情况下，它阻止了为同一个对象创建多个代理的能力（即不支持一对多）。为了解决这个问题，我们将在我们的包装器中存储对象的指针，而不是对象本身，如下所示：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
class wrapper :
    public base<RET, ARGS...>
{
    const T *m_t{};
    RET (T::*m_func)(ARGS...);

public:

    wrapper(const T *t, RET (T::*func)(ARGS...)) :
        m_t{t},
        m_func{func}
    { }

    RET func(ARGS... args) override
    {
        return std::invoke(m_func, m_t, args...);
    }
};
```

如前所示，我们所做的唯一改变是我们存储一个指向我们包装的对象的指针，而不是对象本身，这也意味着我们需要在构造函数中初始化这个指针。为了使用这个新的包装器，我们必须修改我们的代理构造函数如下：

```cpp
    template<typename T>
    delegate(const T *t, RET (T::*func)(ARGS...)) :
        m_wrapper{
            std::make_unique<wrapper<T, RET, ARGS...>>(t, func)
        }
    { }
```

这又意味着我们必须更新我们的用户定义类型推导指南，如下所示：

```cpp
template<
    typename T,
    typename RET,
    typename... ARGS
    >
delegate(const T *, RET(T::*)(ARGS...)) -> delegate<RET(ARGS...)>;
```

通过这些修改，我们现在可以创建我们的代理，如下所示：

```cpp
int main(void)
{
    spiderman s;
    captain_america c;

    std::array<delegate<bool(int, int)>, 2> heros {
        delegate(&s, &spiderman::attack),
        delegate(&c, &captain_america::attack)
    };

    for (auto &h : heros) {
        std::cout << h(0, 42) << '\n';
    }

    return 0;
}
```

如前所示，代理接受每个对象的指针，这意味着我们可以创建任意数量的这些代理，包括根据需要创建对其他成员函数指针的代理的能力。

# 向我们的代理添加对非成员函数的支持

最后，我们需要修改代理以支持非成员函数。看看这个例子：

```cpp
bool attack(int x, int y)
{
    return x == 42 && y == 42 ? true : false;
}
```

为了做到这一点，我们只需要添加另一个包装器，如下所示：

```cpp
template<
    typename RET,
    typename... ARGS
    >
class fun_wrapper :
    public base<RET, ARGS...>
{
    RET (*m_func)(ARGS...);

public:

    fun_wrapper(RET (*func)(ARGS...)) :
        m_func{func}
    { }

    RET func(ARGS... args) override
    {
        return m_func(args...);
    }
};
```

如前所示，与我们的原始包装器一样，我们存储我们希望调用的函数的指针，但在这种情况下，我们不需要存储对象的指针，因为没有对象（因为这是一个非成员函数包装器）。为了使用这个新的包装器，我们必须添加另一个代理构造函数，如下所示：

```cpp
    delegate(RET (func)(ARGS...)) :
        m_wrapper{
            std::make_unique<fun_wrapper<RET, ARGS...>>(func)
        }
    { }
```

这意味着我们还必须提供另一个用户定义的类型推导指南，如下所示：

```cpp
template<
    typename RET,
    typename... ARGS
    >
delegate(RET(*)(ARGS...)) -> delegate<RET(ARGS...)>;
```

通过所有这些修改，我们最终能够使用我们在本篇文章开头定义的代理：

```cpp
int main(void)
{
    spiderman s;
    captain_america c;

    std::array<delegate<bool(int, int)>, 3> heros {
        delegate(attack),
        delegate(&s, &spiderman::attack),
        delegate(&c, &captain_america::attack)
    };

    for (auto &h : heros) {
        std::cout << h(0, 42) << '\n';
    }

    return 0;
}
```

当这个被执行时，我们得到以下输出：

![](img/59994462-e91d-48fa-bcaf-1538fa6e4e37.png)

这个委托可以进一步扩展以支持 lambda 函数，方法是添加另一组包装器，并且可以通过使用一个小缓冲区来替换委托中的`std::unique_pointer`，从而避免动态内存分配，这个小缓冲区的大小与成员函数包装器相同（或者换句话说，实现小尺寸优化）。
