# 第五章：关注点的分离 - 软件架构、函数和可变模板

## 学习目标

在本章结束时，您将能够：

+   使用 PIMPL 习惯用法来实现对象级封装

+   使用函数对象、std::function 和 lambda 表达式实现回调系统

+   使用正确的捕获技术来实现 lambda 表达式

+   开发可变模板以实现 C#风格的委托以进行事件处理。

本章将向您展示如何实现 PIMPL 习惯用法，以及如何为您自己的程序开发回调机制。

## 介绍

在上一章中，我们学习了如何实现类来正确管理资源，即使在发生异常时也是如此，使用 RAII。我们还学习了 ADL（**Argument Dependent Lookup**）以及它如何确定要调用的函数。最后，我们谈到了显式关键字如何可以防止编译器进行类型之间的自动转换，即隐式转换。

在本章中，我们将研究依赖关系，包括物理依赖关系和逻辑依赖关系，以及它们如何对构建时间产生不利影响。我们还将学习如何将可见接口类与实现细节分离，以增加构建时间的速度。然后，我们将学习如何捕获函数和上下文，以便以后可以使用“函数对象”、`std::function`和“lambda 表达式”来调用它们。最后，我们将实现可变模板以提供基于事件的回调机制。

### 指向实现的指针（PIMPL）习惯用法

随着 C++实现的项目变得越来越大，构建时间增长的速度可能会超过文件数量的增长速度。这是因为 C++构建模型使用了文本包含模型。这样做是为了让编译器能够确定类的大小和布局，导致了“调用者”和“被调用者”之间的耦合，但也允许进行优化。请记住，一切都必须在使用之前定义。未来的一个特性叫做“模块”承诺解决这个问题，但现在我们需要了解这个问题以及用来解决问题的技术。

### 逻辑和物理依赖关系

当我们希望从另一个类中访问一个类时，我们有一个逻辑依赖关系。一个类在逻辑上依赖于另一个类。如果我们考虑我们在*第 2A 章*“不允许鸭子 - 类型和推导”和*第三章*“能与应该之间的距离 - 对象、指针和继承”中开发的`Graphics`类，`Point3d`和`Matrix3d`，我们有两个逻辑独立的类`Matrix3d`和`Point3d`。然而，由于我们如何在两者之间实现了乘法运算符，我们创建了一个编译时或**物理依赖关系**。

![图 4.1：Matrix3d 和 Point3d 的物理依赖关系](img/C14583_04_01.jpg)

###### 图 4.1：Matrix3d 和 Point3d 的物理依赖关系

正如我们在这些相对简单的类中所看到的，头文件和实现文件之间的物理依赖关系很快就会变得复杂起来。正是这种复杂性导致了大型项目的构建时间增加，因为物理（和逻辑）依赖关系的数量增长到了成千上万。在前面的图表中，我们只显示了 13 个依赖关系，如箭头所示。但实际上还有更多，因为包含标准库头文件通常会引入一系列包含文件的层次结构。这意味着如果修改了一个头文件，那么直接或间接依赖于它的所有文件都需要重新编译以适应变化。如果更改是对用户甚至无法访问的私有类成员定义的，也会触发重新构建。

为了加快编译时间，我们使用了保护技术来防止头文件被多次处理：

```cpp
#if !defined(MY_HEADER_INCLUDED)
#define   MY_HEADER_INCLUDED
// definitions 
#endif // !defined(MY_HEADER_INCLUDED)
```

最近，大多数编译器现在支持`#pragma once`指令，它可以实现相同的结果。

这些实体（文件、类等）之间的关系被称为**耦合**。如果对文件/类的更改导致对其他文件/类的更改，则文件/类与另一个文件/类**高度耦合**。如果对文件/类的更改不会导致对其他文件/类的更改，则文件/类与另一个文件/类**松散耦合**。

高度耦合的代码（文件/类）会给项目带来问题。高度耦合的代码难以更改（不灵活），难以测试和难以理解。另一方面，松散耦合的代码更容易更改（只需修改一个类），更易测试（只需测试正在测试的类）并且更易阅读和理解。耦合反映并与逻辑和物理依赖相关。

### 指向实现（PIMPL）惯用法

解决这种耦合问题的一种方法是使用“Pimpl 惯用法”（即“指向实现的指针惯用法”）。这也被称为不透明指针、编译器防火墙惯用法，甚至是“切尔西猫技术”。考虑 Qt 库，特别是 Qt 平台抽象（QPA）。这是一个隐藏 Qt 应用程序所托管的操作系统和/或平台细节的抽象层。实现这样一层的方法之一是使用 PIMPL 惯用法，其中公共接口暴露给应用程序开发人员，但功能的实现方式是隐藏的。Qt 实际上使用了 PIMPL 的变体，称为 d-pointer。

例如，GUI 的一个特性是使用对话框，它是一个弹出窗口，用于显示信息或提示用户输入。可以在**dialog.hpp**中声明如下：

#### 注

有关 QT 平台抽象（QPA）的更多信息，请访问以下链接：[`doc.qt.io/qt-5/qpa.html#`](https://doc.qt.io/qt-5/qpa.html#)。

```cpp
#pragma once
class Dialog
{
public:
    Dialog();
    ~Dialog();
    void create(const char* message);
    bool show();
private:
    struct DialogImpl;
    DialogImpl* m_pImpl;
};
```

用户可以访问使用`Dialog`所需的所有函数，但不知道它是如何实现的。请注意，我们声明了`DialogImpl`但没有定义它。一般来说，我们对这样的`DialogImpl`类做不了太多事情。但有一件事是允许的，那就是声明一个指向它的指针。C++的这个特性允许我们在实现文件中隐藏实现细节。这意味着在这种简单情况下，我们不需要为这个声明包含任何包含文件。

实现文件**dialogImpl.cpp**可以实现为：

```cpp
#include "dialog.hpp"
#include <iostream>
#include <string>
struct Dialog::DialogImpl
{
    void create(const char* message)
    {
        m_message = message;
        std::cout << "Creating the Dialog\n";
    }
    bool show()
    {
        std::cout << "Showing the message: '" << m_message << "'\n";
        return true;
    }
    std::string m_message;
};
Dialog::Dialog() : m_pImpl(new DialogImpl)
{
}
Dialog::~Dialog()
{
    delete m_pImpl;
}
void Dialog::create(const char* message)
{
    m_pImpl->create(message);
}
bool Dialog::show()
{
    return m_pImpl->show();
}
```

我们从中注意到几件事：

+   在我们定义对话框所需的方法之前，我们先定义实现类`DialogImpl`。这是必要的，因为`Dialog`将需要通过`m_pImpl`来调用这些方法，这意味着它们需要首先被定义。

+   `Dialog`的构造函数和析构函数负责内存管理。

+   我们只在实现文件中包含了实现所需的所有必要头文件。这通过最小化**Dialog.hpp**文件中包含的头文件数量来减少耦合。

该程序可以按以下方式执行：

```cpp
#include <iostream>
#include "dialog.hpp"
int main()
{
    std::cout << "\n\n------ Pimpl ------\n";
    Dialog dialog;
    dialog.create("Hello World");
    if (dialog.show())
    {
        std::cout << "Dialog displayed\n";
    }
    else
    {
        std::cout << "Dialog not displayed\n";
    }
    std::cout << "Complete.\n";
    return 0;
}
```

在执行时，上述程序产生以下输出：

![图 4.2：示例 Pimpl 实现输出](img/C14583_04_02.jpg)

###### 图 4.2：示例 Pimpl 实现输出

### PIMPL 的优缺点

使用 PIMPL 的最大优势是它打破了类的客户端和其实现之间的编译时依赖关系。这样可以加快构建时间，因为 PIMPL 在定义（头）文件中消除了大量的`#include`指令，而只需要在实现文件中才是必要的。

它还将实现与客户端解耦。现在我们可以自由更改 PIMPL 类的实现，只需重新编译该文件。这可以防止编译级联，其中对隐藏成员的更改会触发客户端的重建。这被称为编译防火墙。

PIMPL 惯用法的一些其他优点如下：

+   **数据隐藏** - 实现的内部细节真正地被隔离在实现类中。如果这是库的一部分，那么它可以用来防止信息的泄露，比如知识产权。

+   `DLL`或`.so`文件），并且可以自由更改它而不影响客户端代码。

这些优点是有代价的。缺点如下：

+   **维护工作** - 可见类中有额外的代码将调用转发到实现类。这增加了一定复杂性的间接层。

+   **内存管理** - 现在添加了一个指向实现的指针，我们需要管理内存。它还需要额外的存储空间来保存指针，在内存受限的系统中（例如：物联网设备）这可能是关键的。

### 使用 unique_ptr<>实现 PIMPL

我们当前的 Dialog 实现使用原始指针来持有 PIMPL 实现引用。在*第三章*中，*能与应该之间的距离-对象、指针和继承*中，我们讨论了对象的所有权，并引入了智能指针和 RAII。PIMPL 指针指向的隐藏对象是一个需要管理的资源，应该使用`RAII`和`std::unique_ptr`来执行。正如我们将看到的，使用`std::unique_ptr`实现`PIMPL`有一些注意事项。

让我们将 Dialog 的实现改为使用智能指针。首先，头文件更改以引入`#include <memory>`行，并且可以删除析构函数，因为`unique_ptr`会自动删除实现类。

```cpp
#pragma once
#include <memory>
class Dialog
{
public:
    Dialog();
    void create(const char* message);
    bool show();
private:
    struct DialogImpl;
    std::unique_ptr<DialogImpl> m_pImpl;
};
```

显然，我们从实现文件中删除了析构函数，并修改构造函数以使用`std::make_unique`。

```cpp
Dialog::Dialog() : m_pImpl(std::make_unique<DialogImpl>())
{
}
```

重新编译我们的新版本时，**Dialog.hpp**和**DialogImpl.cpp**文件没有问题，但我们的客户端**main.cpp**报告了以下错误（使用 gcc 编译器），如下所示：

![图 4.3：使用 unique_ptr 失败的 Pimpl 编译](img/C14583_04_03.jpg)

###### 图 4.3：使用 unique_ptr 失败的 Pimpl 编译

当`main()`函数结束时，第一个错误报告了`Dialog`。正如我们在*第 2A 章*中讨论的那样，*不允许鸭子-类型和推断*编译器将为我们生成一个析构函数（因为我们删除了它）。这个生成的析构函数将调用`unique_ptr`的析构函数，这就是错误的原因。如果我们看一下`line 76`，默认`unique_ptr`使用的`deleter`的`operator()`函数（`deleter`是`unique_ptr`在销毁其指向的对象时调用的函数）：

```cpp
void
operator()(_Tp* __ptr) const
{
    static_assert(!is_void<_Tp>::value, "can't delete pointer to incomplete type");
    static_assert(sizeof(_Tp)>0, "can't delete pointer to incomplete type");
    delete __ptr;
}
```

我们的代码在第二个`static_assert()`语句上失败，这会导致编译出错。问题在于编译器试图为`std::unique_ptr<DialogImpl>`和`DialogImpl`生成析构函数，而`DialogImpl`是一个不完整的类型。因此，为了解决问题，我们控制生成析构函数的时机，使`DialogImpl`成为一个完整的类型。

为了做到这一点，我们将析构函数的声明放回类中，并将其实现添加到`DialogImpl.cpp`文件中。

```cpp
Dialog::~Dialog()
{
}
```

当我们编译并运行我们的程序时，它产生的输出与之前完全相同。实际上，如果我们只需要一个空的析构函数，我们可以用以下代码替换上面的代码：

```cpp
Dialog::~Dialog() = default;
```

如果我们编译并运行我们的程序，那么将产生以下输出：

![图 4.4：示例 unique_ptr Pimpl 实现输出](img/C14583_04_04.jpg)

###### 图 4.4：示例 unique_ptr Pimpl 实现输出

### unique_ptr<> PIMPL 特殊函数

由于 PIMPL 通常意味着可见接口类拥有实现类，因此移动语义是一个自然的选择。然而，就像编译器生成的析构函数实现是正确的一样，编译器生成的移动构造函数和移动赋值运算符将产生期望的行为，即对成员`unique_ptr`执行移动。移动操作都可能需要在分配传输值之前执行删除，因此，与不完整类型的析构函数一样，它们也会遇到相同的问题。解决方案与析构函数相同-在头文件中声明该方法，并在类型完成时实现-在实现文件中。因此，我们的头文件看起来像下面这样：

```cpp
class Dialog
{
public:
    Dialog();
    ~Dialog();
    Dialog(Dialog&& rhs);
    Dialog& operator=(Dialog&& rhs);
    void create(const char* message);
    bool show();
private:
    struct DialogImpl;
    std::unique_ptr<DialogImpl> m_pImpl;
};
```

虽然实现看起来像：

```cpp
Dialog::Dialog() : m_pImpl(std::make_unique<DialogImpl>())
{
}
Dialog::~Dialog() = default;
Dialog::Dialog(Dialog&& rhs) = default;
Dialog& Dialog::operator=(Dialog&& rhs) = default;
```

根据我们在实现类中隐藏的数据项，我们可能还希望在我们的 PIMPL 类上具有复制功能。在 Dialog 类内部使用`std::unique_ptr`可以防止自动生成复制构造函数和复制赋值运算符，因为内部成员不支持复制。此外，通过定义移动成员函数，就像我们在*第 2A 章*中看到的那样，它也阻止编译器生成复制版本。此外，如果编译器为我们生成了复制语义，它只会是**浅复制**。但由于 PIMPL 实现，我们需要**深复制**。因此，我们需要编写自己的复制特殊成员函数。同样，定义放在头文件中，实现需要在类型完成的地方完成，即在**DialogImpl.cpp**文件中。

在头文件中，我们添加以下声明：

```cpp
Dialog(const Dialog& rhs);
Dialog& operator=(const Dialog& rhs);
```

实现将如下所示：

```cpp
Dialog::Dialog(const Dialog& rhs) : m_pImpl(nullptr)
{
    if (this == &rhs)   // do nothing on copying self
    return;
    if (rhs.m_pImpl)    // rhs has something -> clone it
        m_pImpl = std::make_unique<DialogImpl>(*rhs.m_pImpl);
}
Dialog& Dialog::operator=(const Dialog& rhs)
{
    if (this == &rhs)   // do nothing on assigning to self
        return *this;
    if (!rhs.m_pImpl)   // rhs is empty -> delete ours
    {
        m_pImpl.reset();
    }
    else if (!m_pImpl)  // ours is empty -> clone rhs
    {
        m_pImpl = std::make_unique<DialogImpl>(*rhs.m_pImpl);
    }
    else // use copy of DialogImpl
    {
        *m_pImpl = *rhs.m_pImpl;
    }
}
```

注意`if(this == &rhs)`子句。这些是为了防止对象不必要地复制自身。还要注意，我们需要检查`unique_ptr`是否为空，并相应地处理复制。

#### 注意

在本章中解决任何实际问题之前，下载 GitHub 存储库[`github.com/TrainingByPackt/Advanced-CPlusPlus`](https://github.com/TrainingByPackt/Advanced-CPlusPlus)并在 Eclipse 中导入 Lesson 4 文件夹，以便您可以查看每个练习和活动的代码。

### 练习 1：使用 unique_ptr<>实现厨房

在这个练习中，我们将通过使用`unique_ptr<>`实现`Pimpl idiom`来隐藏厨房处理订单的细节。按照以下步骤来实现这个练习：

1.  在 Eclipse 中打开**Lesson4**项目，然后在**Project Explorer**中展开**Lesson4**，然后展开**Exercise01**，双击**Exercise1.cpp**以将此练习的文件打开到编辑器中。

1.  由于这是一个基于 CMake 的项目，将当前构建器更改为 CMake Build（便携式）。

1.  单击**Launch Configuration**下拉菜单，然后选择**New Launch Configuration…**。配置**L4Exercise1**以使用名称**Exercise1**运行。

1.  单击**Run**按钮。练习 1 将运行并产生以下输出：![图 4.5：练习 1 程序输出](img/C14583_04_05.jpg)

###### 图 4.5：练习 1 程序输出

1.  打开`Kitchen`。我们将把所有私有成员移到一个实现类中并隐藏细节。

1.  在`#include <memory>`指令中获得对`unique_ptr`的访问。添加析构函数`~Kitchen();`的声明，然后将以下两行添加到私有部分的顶部：

```cpp
struct Impl;
std::unique_ptr<Impl> m_impl;
```

1.  打开`#include`指令：

```cpp
struct Kitchen::Impl
{
};
Kitchen::~Kitchen() = default;
```

1.  单击**Run**按钮重新构建程序。您会看到输出仍然与以前相同。

1.  从`Kitchen`类中的`Kitchen::Impl`声明中删除除两个新成员之外的所有私有成员。`#include <vector>`，`#include "recipe.hpp"`和`#include "dessert.hpp"`：

```cpp
#pragma once
#include <string>
#include <memory>
class Kitchen
{
public:
    Kitchen(std::string chef);
    ~Kitchen();
    std::string processOrder(std::string order);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
```

1.  在`Kitchen::Impl`构造函数中：

```cpp
Kitchen::Impl::Impl(std::string chef) : m_chef{chef}
```

1.  对于原始方法的其余部分，将它们更改为作用域为`Kitchen::Impl`而不是`Kitchen::`。例如，`std::string Kitchen::processOrder(std::string order)`变为`std::string Kitchen::Impl::processOrder(std::string order)`。

1.  在`Kitchen::Impl`中，添加一个带有`std::string`参数和`processOrder()`方法的构造函数。`Kitchen::Impl`声明现在应如下所示：

```cpp
struct Kitchen::Impl
{
    Impl(std::string chef);
    std::string processOrder(std::string order);
    std::string searchForRecipe(std::string order);
    std::string searchForDessert(std::string order);
    std::string cookRecipe(std::string recipe);
    std::string serveDessert(std::string dessert);
    std::vector<Recipe>::iterator getRecipe(std::string recipe);
    std::vector<Dessert>::iterator getDessert(std::string recipe);
    std::string m_chef;
    std::vector<Recipe> m_recipes;
    std::vector<Dessert> m_desserts;
};
```

1.  在`#include <vector>`，`#include "recipe.hpp"`和`#include "dessert.hpp"`添加到文件顶部。

1.  单击`Kitchen::Kitchen`和`Kitchen::processOrder`。

1.  在`Kitchen::Impl`方法定义中，添加以下两个方法：

```cpp
Kitchen::Kitchen(std::string chef) : m_impl(std::make_unique<Kitchen::Impl>(chef))
{
}
std::string Kitchen::processOrder(std::string order)
{
    return m_impl->processOrder(order);
}
```

1.  单击**Run**按钮重新构建程序。程序将再次运行以产生原始输出。

![图 4.6：使用 Pimpl 的厨房程序输出](img/C14583_04_06.jpg)

###### 图 4.6：使用 Pimpl 的厨房程序输出

在这个练习中，我们已经将一个类中的许多细节移到了 PIMPL 类中，以隐藏细节并使用先前描述的技术将接口与实现解耦。

## 函数对象和 Lambda 表达式

在编程中常用的一种模式，特别是在实现基于事件的处理时，如异步输入和输出，是使用**回调**。客户端注册他们希望被通知事件发生的情况（例如：数据可供读取，或数据传输完成）。这种模式称为**观察者模式**或**订阅者发布者模式**。C ++支持各种技术来提供回调机制。

### 函数指针

第一种机制是使用**函数指针**。这是从 C 语言继承的传统功能。以下程序显示了函数指针的示例：

```cpp
#include <iostream>
using FnPtr = void (*)(void);
void function1()
{
    std::cout << "function1 called\n";
}
int main()
{
    std::cout << "\n\n------ Function Pointers ------\n";
    FnPtr fn{function1};
    fn();
    std::cout << "Complete.\n";
    return 0;
}
```

编译和执行此程序时，将产生以下输出：

![图 4.7：函数指针程序输出](img/C14583_04_07.jpg)

###### 图 4.7：函数指针程序输出

严格来说，代码应修改如下：

```cpp
FnPtr fn{&function1};
if(fn != nullptr)
    fn();
```

首先要注意的是应使用地址（`&`）运算符来初始化指针。其次，在调用之前应检查指针是否有效。

```cpp
#include <iostream>
using FnPtr = void (*)(void);
struct foo
{
    void bar() { std::cout << "foo:bar called\n"; }
};
int main()
{
    std::cout << "\n\n------ Function Pointers ------\n";
    foo object;
    FnPtr fn{&object.bar};
    fn();
    std::cout << "Complete.\n";
    return 0;
}
```

当我们尝试编译此程序时，会得到以下错误：

![图 4.8：编译函数指针程序时出现的错误](img/C14583_04_08.jpg)

###### 图 4.8：编译函数指针程序时出现的错误

第一个错误的文本是`this`指针。

通过将上述程序更改为以下内容：

```cpp
#include <iostream>
using FnPtr = void (*)(void);
struct foo
{
    static void bar() { std::cout << "foo:bar called\n"; }
};
int main()
{
    std::cout << "\n\n------ Function Pointers ------\n";
    FnPtr fn{&foo::bar};
    fn();
    std::cout << "Complete.\n";
    return 0;
}
```

它成功编译并运行：

![图 4.9：使用静态成员函数的函数指针程序](img/C14583_04_09.jpg)

###### 图 4.9：使用静态成员函数的函数指针程序

函数指针技术通常用于与使用回调和支持回调的操作系统通知的 C 库进行接口的情况。在这两种情况下，回调通常会接受一个`void *`参数，该参数是用户注册的数据块指针。数据块指针可以是类的`this`指针，然后对其进行解引用，并将回调转发到成员函数。

在其他语言中，如 Python 和 C＃，捕获函数指针也会捕获调用该函数所需的足够数据（例如：`self`或`this`）是语言的一部分。 C ++具有通过函数调用运算符使任何对象可调用的能力，我们将在下面介绍。

### 什么是函数对象？

C ++允许重载函数调用运算符`operator()`。这导致可以使任何对象“可调用”。可调用的对象在以下程序中称为`Scaler`类实现了`functor`。

```cpp
struct Scaler
{
    Scaler(int scale) : m_scale{scale} {};
    int operator()(int value)
    {
        return m_scale * value;
    }
    int m_scale{1};
};
int main()
{
    std::cout << "\n\n------ Functors ------\n";
    Scaler timesTwo{2};
    Scaler timesFour{4};
    std::cout << "3 scaled by 2 = " << timesTwo(3) << "\n";
    std::cout << "3 scaled by 4 = " << timesFour(3) << "\n";
    std::cout << "Complete.\n";
    return 0;
}
```

创建了两个类型为`Scaler`的对象，并且它们在生成输出的行内被用作函数。上述程序产生以下输出：

![图 4.10：functors 程序输出](img/C14583_04_10.jpg)

###### 图 4.10：函数对象程序输出

`functors`相对于函数指针的一个优点是它们可以包含状态，可以是对象或跨所有实例。另一个优点是它们可以传递给期望函数（例如`std::for_each`）或操作符（例如`std::transform`）的 STL 算法。

这样的用法示例可能如下所示：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
struct Scaler
{
    Scaler(int scale) : m_scale{scale} {};
    int operator()(int value)
    {
        return m_scale * value;
    }
    int m_scale{1};
};
void PrintVector(const char* prefix, std::vector<int>& values)
{
    const char* sep = "";
    std::cout << prefix << " = [";
    for(auto n : values)
    {
        std::cout << sep << n;
        sep = ", ";
    }
    std::cout << "]\n";
}
int main()
{
    std::cout << "\n\n------ Functors with STL ------\n";
    std::vector<int> values{1,2,3,4,5};
    PrintVector("Before transform", values);
    std::transform(values.begin(), values.end(), values.begin(), Scaler(3));
    PrintVector("After transform", values);
    std::cout << "Complete.\n";
    return 0;
}
```

如果我们运行这个程序，产生的输出将如下所示：

![图 4.11：显示标量转换向量的程序输出](img/C14583_04_11.jpg)

###### 图 4.11：显示标量转换向量的程序输出

### 练习 2：实现函数对象

在这个练习中，我们将实现两个不同的函数对象，可以与 STL 算法`for_each`一起使用。

1.  在 Eclipse 中打开**Lesson4**项目，然后在**项目资源管理器**中展开**Lesson4**，然后展开**Exercise02**，双击**Exercise2.cpp**以打开此练习的文件到编辑器中。

1.  由于这是一个基于 CMake 的项目，将当前构建器更改为 CMake Build（便携式）。

1.  点击**启动配置**下拉菜单，选择**新启动配置...**。配置**L4Exercise2**以名称**Exercise2**运行。

1.  点击**运行**按钮。练习 2 将运行并产生以下输出：![图 4.12：练习 2 的初始输出](img/C14583_04_12.jpg)

###### 图 4.12：练习 2 的初始输出

我们要做的第一件事是通过引入函数对象来修复输出的格式。

1.  在编辑器中，在`main()`函数定义之前添加以下类定义：

```cpp
struct Printer
{
    void operator()(int n)
    {
        std::cout << m_sep << n;
        m_sep = ", ";
    }
    const char* m_sep = "";
};
```

1.  在**main()**方法中替换以下代码

```cpp
std::cout << "Average of [";
for( auto n : values )
    std::cout << n << ", ";
std::cout << "] = ";
```

**带有**

```cpp
std::cout << "Average of [";
std::for_each(values.begin(), values.end(), Printer());
std::cout << "] = ";
```

1.  点击**运行**按钮。练习将运行并产生以下输出：![图 4.13：改进的输出格式的练习 2](img/C14583_04_13.jpg)

###### 图 4.13：改进的输出格式的练习 2

1.  `Printer`类的内部状态允许我们修复格式。现在，引入一个`aggregator`类，它将允许我们计算`average`。在文件顶部添加以下类定义：

```cpp
struct Averager
{
    void operator()(int n)
    {
        m_sum += n;
        m_count++;
    }
    float operator()() const
    {
        return static_cast<float>(m_sum)/(m_count==0?1:m_count);
    }
    int m_count{0};
    int m_sum{0};
};
```

1.  修改`main()`方法以使用`Averager`类如下：

```cpp
int main(int argc, char**argv)
{
    std::cout << "\n------ Exercise 2 ------\n";
    std::vector<int> values {1,2,3,4,5,6,7,8,9,10};
    Averager averager = std::for_each(values.begin(), values.end(), 
    Averager());
    std::cout << "Average of [";
    std::for_each(values.begin(), values.end(), Printer());
    std::cout << "] = ";
    std::cout << averager() << "\n";
    std::cout << "Complete.\n";
    return 0;
}
```

1.  点击**运行**按钮。练习将运行并产生以下输出：

![图 4.14：带有平均值的练习 2 输出](img/C14583_04_14.jpg)

###### 图 4.14：带有平均值的练习 2 输出

注意，`std::for_each()`返回传递给它的`Averager`的实例。这个实例被复制到变量`averager`中，然后包含了计算平均值所需的数据。在这个练习中，我们实现了两个函数对象或`functor`类：`Averager`和`Printer`，当传递给 STL 算法`for_each`时，我们可以将它们用作函数。

### std::function<>模板

C++11 引入了一个通用的多态函数包装模板，`std::function<>`，使得实现回调和其他与函数相关的功能更容易。`std::function`保存一个可调用对象，称为`std::function`将导致抛出`std::bad_function_call`异常。

函数对象可以存储、复制或调用目标，这些目标可以是以下任何可调用对象：函数、函数对象（定义了`operator()`）、成员函数指针或 lambda 表达式。我们将在主题*什么是 Lambda 表达式？*中更多地介绍它。

在实例化`std::function`对象时，只需要提供函数签名，而不需要初始化它的值，导致一个空实例。实例化如下所示：

![图 4.15：std::function 声明的结构](img/C14583_04_15.jpg)

###### 图 4.15：std::function 声明的结构

模板的参数定义了`variable`存储的目标的`function signature`。签名以返回类型开始（可以是 void），然后在括号内放置函数将被调用的类型列表。

使用自由函数和`std::function`的`functor`非常简单。只要签名与传递给`std::function`模板的参数匹配，我们就可以简单地将自由函数或`functor`等同于实例。

```cpp
void FreeFunc(int value);
struct Functor 
{
    void operator()(int value);
};
std::function<void(int)> func;
Functor functor;
func = FreeFunc;                     // Set target as FreeFunc
func(32);                            // Call FreeFunc with argument 32
func = functor;                      // set target as functor
func(42);                            // Call Functor::operator() with argument 42
```

但是，如果我们想要在对象实例上使用一个方法，那么我们需要使用另一个 STL 辅助模板`std::bind()`。如果我们运行以下程序：

```cpp
#include <iostream>
#include <functional>
struct Binder
{
    void method(int a, int b)
    {
        std::cout << "Binder::method(" << a << ", " << b << ")\n";
    }
};
int main()
{
    std::cout << "\n\n------ Member Functions using bind ------\n";
    Binder binder;
    std::function<void(int,int)> func;
    auto func1 = std::bind(&Binder::method, &binder, 1, 2);
    auto func2 = std::bind(&Binder::method, &binder, std::placeholders::_1, std::placeholders::_2);
    auto func3 = std::bind(&Binder::method, &binder, std::placeholders::_2, std::placeholders::_1);
    func = func1;
    func(34,56);
    func = func2;
    func(34,56);
    func = func3;
    func(34,56);
    std::cout << "Complete.\n";
    return 0;
}
```

然后我们得到以下输出：

![图 4.16：使用 std::bind()和 std::function 的程序输出](img/C14583_04_16.jpg)

###### 图 4.16：使用 std::bind()和 std::function 的程序输出

注意几点：

+   函数`method()`是使用类作为作用域限定符引用的；

+   `Binder`实例的地址作为第二个参数传递给`std::bind()`，这使其成为传递给`method()`的第一个参数。这是必要的，因为所有非静态成员都有一个隐式的`this`指针作为第一个参数传递。

+   使用`std::placeholders`定义，我们可以绑定调用绑定方法时使用的参数，甚至改变传递的顺序（如`func3`所示）。

C++11 引入了一些称为 lambda 表达式的语法糖，使得更容易定义匿名函数，还可以用于绑定方法并将它们分配给`std::function`实例表达式。我们将在*什么是 Lambda 表达式？*主题中更多地涵盖它。

### 练习 3：使用 std::function 实现回调

在这个练习中，我们将利用`std::function<>`模板实现函数回调。按照以下步骤实现这个练习：

1.  在 Eclipse 中打开**Lesson4**项目，然后在**Project Explorer**中展开**Lesson4**，然后展开**Exercise03**，双击**Exercise3.cpp**以将此练习的文件打开到编辑器中。

1.  由于这是一个基于 CMake 的项目，请将当前构建器更改为 CMake Build（便携式）。

1.  单击**启动配置**下拉菜单，然后选择**新启动配置…**。配置**L4Exercise3**以使用名称**Exercise3**运行。

1.  单击**运行**按钮。练习将运行并产生以下输出：![图 4.17：练习 3 输出（调用空的 std::function）](img/C14583_04_17.jpg)

###### 图 4.17：练习 3 输出（调用空的 std::function）

1.  我们要做的第一件事是防止调用空的`TestFunctionTemplate()`行`func(42);`，并用以下代码替换它：

```cpp
if (func)
{
    func(42);
}
else
{
    std::cout << "Not calling an empty func()\n";
}
```

1.  单击**运行**按钮。练习将运行并产生以下输出：![图 4.18：练习 3 输出（防止调用空的 std::function）](img/C14583_04_18.jpg)

###### 图 4.18：练习 3 输出（防止调用空的 std::function）

1.  在函数`TestFunctionTemplate()`之前的文件中添加`FreeFunction()`方法：

```cpp
void FreeFunction(int n)
{
    std::cout << "FreeFunction(" << n << ")\n";
}
```

1.  在`TestFunctionTemplate()`函数中，在`if (func)`之前立即添加以下行：

```cpp
func = FreeFunction;
```

1.  单击**运行**按钮。练习将运行并产生以下输出：![图 4.19：练习 3 输出（FreeMethod）](img/C14583_04_19.jpg)

###### 图 4.19：练习 3 输出（FreeMethod）

1.  在`TestFunctionTemplate()`函数之前添加新的类定义：

```cpp
struct FuncClass
{
    void member(int n)
    {
        std::cout << "FuncClass::member(" << n << ")\n";
    }
    void operator()(int n)
    {
    std::cout << "FuncClass object(" << n << ")\n";
    }
};
```

1.  用以下代码替换行`func = FreeFunction;`：

```cpp
FuncClass funcClass;
func = funcClass;
```

1.  单击**运行**按钮。练习将运行并产生以下输出：![4.20：练习 3 输出（对象函数调用覆盖）](img/C14583_04_20.jpg)

###### 4.20：练习 3 输出（对象函数调用覆盖）

1.  用以下代码替换行`func = funcClass;`：

```cpp
func = std::bind(&FuncClass::member, &funcClass, std::placeholders::_1);
```

1.  单击**运行**按钮。练习将运行并产生以下输出：![图 4.21：练习 3 输出（成员函数）](img/C14583_04_21.jpg)

###### 图 4.21：练习 3 输出（成员函数）

1.  用以下代码替换行`func = std::bind(…);`：

```cpp
func = [](int n) {std::cout << "lambda function(" << n << ")\n";};
```

1.  单击**运行**按钮。练习将运行并产生以下输出：

![图 4.22：练习 3 输出（lambda 函数）](img/C14583_04_22.jpg)

###### 图 4.22：练习 3 输出（lambda 函数）

在这个练习中，我们使用`std::function`模板实现了四种不同类型的函数回调-自由方法，类成员函数，类函数调用方法和 Lambda 函数（我们将在下面讨论）。

### 什么是 Lambda 表达式？

自 C++11 以来，C++支持`匿名函数`，也称为`lambda 表达式`或`lambda`。Lambda 表达式的两种最常见形式是：

![图 4.23：Lambda 表达式的最常见形式](img/C14583_04_23.jpg)

###### 图 4.23：Lambda 表达式的最常见形式

在正常情况下，编译器能够根据**function_body**中的返回语句推断 Lambda 的返回类型（如上图中的形式（1）所示）。然而，如果编译器无法确定返回类型，或者我们希望强制使用不同的类型，那么我们可以使用形式（2）。

[captures]之后的所有内容与普通函数定义相同，只是缺少名称。Lambda 是一种方便的方式，可以在将要使用的位置定义一个简短的方法（只有几行）。Lambda 通常作为参数传递，并且通常不会被重复使用。还应该注意，Lambda 可以分配给一个变量（通常使用 auto 声明）。

我们可以重新编写先前的程序，其中我们使用了`Scaler`类来使用 Lambda 来实现相同的结果：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
void PrintVector(const char* prefix, std::vector<int>& values)
{
    const char* sep = "";
    std::cout << prefix << " = [";
    for(auto n : values)
    {
        std::cout << sep << n;
        sep = ", ";
    }
    std::cout << "]\n";
}
int main()
{
    std::cout << "\n\n------ Lambdas with STL ------\n";
    std::vector<int> values{1,2,3,4,5};
    PrintVector("Before transform", values);
    std::transform(values.begin(), values.end(), values.begin(),
    [] (int n) {return 5*n;}
    );
    PrintVector("After transform", values);
    std::cout << "Complete.\n";
    return 0;
}
```

当此程序运行时，输出显示向量已被缩放了 5 倍：

![图 4.24：使用 lambda 进行缩放的转换](img/C14583_04_24.jpg)

###### 图 4.24：使用 lambda 进行缩放的转换

此程序中的 Lambda 是`[]（int n）{return 5*n;}`，并且具有空的捕获子句`[]`。空的捕获子句意味着 Lambda 函数不访问周围范围内的任何变量。如果没有参数传递给 Lambda，则参数子句`()`是可选的。

### 捕获数据到 Lambda 中

`operator()`。

捕获子句是零个或多个被捕获变量的逗号分隔列表。还有默认捕获的概念-通过引用或通过值。因此，捕获的基本语法是：

+   `[&]` - 通过引用捕获作用域内的所有自动存储期变量

+   `[=]` - 通过值捕获作用域内的所有自动存储期变量（制作副本）

+   `[&x, y]` - 通过引用捕获 x，通过值捕获 y

编译器将此转换为由匿名`functor`类的构造函数初始化的成员变量。在默认捕获（`&`和`=`）的情况下，它们必须首先出现，且仅捕获体中引用的变量。默认捕获可以通过在默认捕获后的捕获子句中放置特定变量来覆盖。例如，`[&，x]`将默认通过引用捕获除`x`之外的所有内容，它将通过值捕获`x`。

然而，默认捕获虽然方便，但并不是首选的捕获方法。这是因为它可能导致悬空引用（通过引用捕获和引用的变量在 Lambda 访问时不再存在）或悬空指针（通过值捕获，特别是 this 指针）。明确捕获变量更清晰，而且编译器能够警告您意外效果（例如尝试捕获全局或静态变量）。

C++14 引入了**init capture**到捕获子句，允许更安全的代码和一些优化。初始化捕获在捕获子句中声明一个变量，并初始化它以在 Lambda 内部使用。例如：

```cpp
int x = 5;
int y = 6;
auto fn = [z=x*x+y, x, y] ()
            {   
                std::cout << x << " * " << x << " + " << y << " = " << z << "\n"; 
            };
fn();
```

在这里，`z`在捕获子句中声明并初始化，以便可以在 Lambda 中使用。如果要在 Lambda 中使用 x 和 y，则它们必须分别捕获。如预期的那样，当调用 Lambda 时，它会产生以下输出：

```cpp
5 * 5 + 6 = 31
```

初始化捕获也可以用于将可移动对象捕获到 Lambda 中，或者如下所示复制类成员：

```cpp
struct LambdaCapture
{
  auto GetTheNameFunc ()
  {
    return [myName = myName] () { return myName.c_str(); };  
  }
  std::string myName;
};
```

这捕获了成员变量的值，并恰好给它相同的名称以在 lambda 内部使用。

默认情况下，lambda 是一个 const 函数，这意味着它不能改变按值捕获的变量的值。在需要修改值的情况下，我们需要使用下面显示的 lambda 表达式的第三种形式。

![图 4.25：另一种 lambda 表达式形式](img/C14583_04_25.jpg)

###### 图 4.25：另一种 lambda 表达式形式

在这种情况下，`specifiers`被`mutable`替换，告诉编译器我们想要修改捕获的值。如果我们不添加 mutable，并尝试修改捕获的值，那么编译器将产生错误。

### 练习 4：实现 Lambda

在这个练习中，我们将实现 lambda 以在 STL 算法的上下文中执行多个操作。按照以下步骤实现这个练习：

1.  在 Eclipse 中打开**Lesson4**项目，然后在**Project Explorer**中展开**Lesson4**，然后双击**Exercise04**，再双击**Exercise4.cpp**以将此练习的文件打开到编辑器中。

1.  由于这是一个基于 CMake 的项目，将当前构建器更改为 CMake Build（便携式）。

1.  单击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。配置**L4Exercise4**以使用名称**Exercise4**运行。

1.  单击**运行**按钮。练习将运行并产生以下输出：![图 4.26：练习 4 的初始输出](img/C14583_04_26.jpg)

###### 图 4.26：练习 4 的初始输出

1.  程序`PrintVector()`和`main()`。`PrintVector()`与我们在*什么是函数对象？*中介绍的版本相同。现在修改它以使用`std::for_each()`库函数和 lambda，而不是范围 for 循环。更新`PrintVector()`如下：

```cpp
void PrintVector(const char* prefix, std::vector<int>& values)
{
    const char* sep = "";
    std::cout << prefix << " = [";
    std::for_each(values.begin(), values.end(),
            [&sep] (int n)
            {
                std::cout << sep << n;
                sep = ", ";
            }
    );
    std::cout << "]\n";
}
```

1.  单击**运行**按钮，我们得到与之前相同的输出。

1.  检查 lambda，我们通过引用捕获了本地变量`sep`。从`sep`中删除`&`，然后单击**运行**按钮。这次编译失败，并显示以下错误：![图 4.27：由于修改只读变量而导致的编译失败](img/C14583_04_27.jpg)

###### 图 4.27：由于修改只读变量而导致的编译失败

1.  更改 lambda 声明以包括`mutable`修饰符：

```cpp
[sep] (int n) mutable
{
    std::cout << sep << n;
    sep = ", ";
}
```

1.  单击**运行**按钮，我们得到与之前相同的输出。

1.  但我们可以再进一步。从函数`PrintVector()`的声明中删除`sep`，并再次更改 lambda 以包括 init 捕获。编写以下代码来实现这一点：

```cpp
[sep = ""] (int n) mutable
{
    std::cout << sep << n;
    sep = ", ";
}
```

1.  单击`PrintVector()`，现在看起来更紧凑：

```cpp
void PrintVector(const char* prefix, std::vector<int>& values)
{
    std::cout << prefix << " = [";
    std::for_each(values.begin(), values.end(), [sep = ""] (int n) mutable
                                  { std::cout << sep << n; sep = ", ";} );
    std::cout << "]\n";
}
```

1.  在`main()`方法中调用`PrintVector()`之后，添加以下行：

```cpp
std::sort(values.begin(), values.end(), [](int a, int b) {return b<a;} );
PrintVector("After sort", values);
```

1.  单击**运行**按钮，现在的输出添加了按降序排序的值列表：![图 4.28：按降序排序 lambda 的程序输出](img/C14583_04_28.jpg)

###### 图 4.28：降序排序 lambda 的程序输出

1.  将 lambda 函数体更改为`{return a<b;}`。单击**运行**按钮，现在的输出显示值按升序排序：![图 4.29：按升序排序 lambda 的程序输出](img/C14583_04_29.jpg)

###### 图 4.29：按升序排序 lambda 的程序输出

1.  在调用`PrintVector()`函数之后，添加以下代码行：

```cpp
int threshold{25};
auto pred = [threshold] (int a) { return a > threshold; };
auto count = std::count_if(values.begin(), values.end(), pred);
std::cout << "There are " << count << " values > " << threshold << "\n";
```

1.  单击`值> 25`：![图 4.30：存储在变量中的 count_if lambda 的输出](img/C14583_04_30.jpg)

###### 图 4.30：存储在变量中的 count_if lambda 的输出

1.  在上述行之后添加以下行，并单击**运行**按钮：

```cpp
threshold = 40;
count = std::count_if(values.begin(), values.end(), pred);
std::cout << "There are " << count << " values > " << threshold << "\n";
```

以下输出将被生成：

![图 4.31：通过重用 pred lambda 产生的错误输出](img/C14583_04_31.jpg)

###### 图 4.31：通过重用 pred lambda 产生的错误输出

1.  程序错误地报告有`七（7）个值> 40`；应该是`三（3）`。问题在于当创建 lambda 并将其存储在变量`pred`中时，它捕获了阈值的当前值，即`25`。将定义`pred`的行更改为以下内容：

```cpp
auto pred = [&threshold] (int a) { return a > threshold; };
```

1.  单击**运行**按钮，现在输出正确报告计数：

![图 4.32：重用 pred lambda 的正确输出](img/C14583_04_32.jpg)

###### 图 4.32：重用 pred lambda 的正确输出

在这个练习中，我们使用 lambda 表达式语法的各种特性实现了几个 lambda，包括 init 捕获和 mutable。

#### 使用 lambda

虽然 lambda 是 C++的一个强大特性，但应该适当使用。目标始终是生成可读的代码。因此，虽然 lambda 可能很简短并且简洁，但有时将功能分解为一个命名良好的方法会更好以便于维护。

## 可变模板

在*第 2B 章*，*不允许鸭子-模板和推导*中，我们介绍了泛型编程和模板。在 C++03 之前的 C++中，模板一直是 C++的一部分。在 C++11 之前，模板的参数数量是有限的。在某些情况下，需要变量数量的参数，需要为所需参数数量的每个变体编写模板。或者，有像`printf()`这样可以接受可变数量参数的可变函数。可变函数的问题在于它们不是类型安全的，因为对参数的访问是通过`va_arg`宏进行类型转换。C++11 通过引入可变模板改变了这一切，其中一个模板可以接受任意数量的参数。C++17 通过引入`constexpr` if 结构改进了可变模板的编写，该结构允许基本情况模板与“递归”模板合并。

最好的方法是实现一个可变模板并解释它是如何工作的。

```cpp
#include <iostream>
#include <string>
template<typename T, typename... Args>
T summer(T first, Args... args) {
    if constexpr(sizeof...(args) > 0)
          return first + summer(args...);
    else
        return first;
}
int main()
{
    std::cout << "\n\n------ Variadic Templates ------\n";
    auto sum = summer(1, 3, 5, 7, 9, 11);
    std::cout << "sum = " << sum << "\n";
    std::string s1{"ab"};
    std::string s2{"cd"};
    std::string s3{"ef"};
    std::string strsum = summer(s1, s2, s3);
    std::cout << "strsum = " << strsum << "\n";
    std::cout << "Complete.\n";
    return 0;
}
```

当我们运行这个程序时，我们得到以下输出：

![图 4.33：可变模板程序输出](img/C14583_04_33.jpg)

###### 图 4.33：可变模板程序输出

那么，可变模板的部分是什么？我们如何阅读它？考虑上面程序中的模板：

```cpp
template<typename T, typename... Args>
T summer(T first, Args... args) {
    if constexpr(sizeof...(args) > 0)
        return first + summer(args...);
    else
        return first;
}
```

在上面的代码中：

+   `typename... Args` 声明 `Args` 为 `模板参数包`。

+   `Args... args`是一个`函数参数包`，其类型由`Args`给出。

+   `sizeof...(args)`返回`args`中包的元素数量。这是一种特殊形式的包扩展。

+   `args...`在对`summer()`的递归调用中展开了包。

或者，您可以将模板视为等效于：

```cpp
template<typename T, typename T1, typename T2, ..., typename Tn>
T summer(T first, T1 t1, T2, ..., Tn tn) {
    if constexpr(sizeof...( t1, t2, ..., tn) > 0)
        return first + summer(t1, t2, ..., tn);
    else
        return first;
}
```

当编译器处理样本程序中的`summer(1, 3, 5, 7, 9, 11)`时，它执行以下操作：

+   它推断`T`是 int，`Args...`是我们的参数包，带有<int, int, int, int, int>。

+   由于包中有多个参数，编译器生成了`first + summer(args...)`，省略号展开了模板参数，将`summer(args...)`转换为`summer(3,5,7,9,11)`。

+   然后编译器生成了`summer(3,5,7,9,11)`的代码。同样，应用了`first + summer(args...)`，其中`summer(5,7,9,11)`。

+   这个过程重复进行，直到编译器必须为`summer(11)`生成代码。在这种情况下，`constexpr` if 语句的 else 子句被触发，它简单地返回`first`。

由于类型由模板的参数确定，因此我们不限于参数具有相同的类型。我们已经在 STL 中遇到了一些可变模板-`std::function`和 std::bind。

还有另一种类型的可变模板，它将其参数转发到另一个函数或模板。这种类型的模板本身并不做太多事情，但提供了一种标准的方法。一个例子是`make_unique`模板，可以实现为：

```cpp
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args)
{
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}
```

`make_unique`必须调用 new 运算符来分配内存，然后调用类型的适当构造函数。调用构造函数所需的参数数量可能会有很大的变化。这种形式的可变模板引入了一些额外的包扩展：

+   `Args&&...`表示我们有一系列转发引用。

+   `std::forward<Args>(args)...`包含要一起展开的参数包，必须具有相同数量的元素-Args 模板参数包和 args 函数参数包。

每当我们需要在可变参数模板中将一个函数调用转发到另一个函数调用时，就会使用这种模式。

### 活动 1：实现多播事件处理程序

1992 年，当 C++处于萌芽阶段时，微软首次引入了`Microsoft Foundation Class`（`MFC`）。这意味着许多围绕这些类的设计选择受到了限制。例如，事件的处理程序通常通过`OnEventXXX()`方法路由。这些通常使用宏配置为从 MFC 类派生的类的一部分。您的团队被要求使用模板来实现更像 C#中可用的委托的多播事件处理程序，这些模板体现了函数对象，并导致可变参数模板以实现可变参数列表。

在 C#中，您可以声明委托如下：

```cpp
delegate int Handler(int parameter);
```

这使得 Handler 成为一个可以分配值并进行调用的类型。这基本上就是 C++中`std::function<>`为我们提供的，除了能够进行多播。您的团队被要求开发一个模板类`Delegate`，可以像 C#委托一样进行操作。

+   Delegate 将接受“可变参数列表”，但只返回`void`

+   `operator+=`将用于向委托添加新的回调

+   它将使用以下语法之一进行调用：`delegate.Notify(…)`或`delegate(…)`

按照以下步骤开发 Delegate 模板：

1.  从**Lesson4/Activity01**文件夹加载准备好的项目，并为项目配置当前构建器为 CMake Build（Portable）。

1.  构建项目，配置启动器并运行单元测试（其中一个虚拟测试失败）。建议为测试运行器使用的名称是**L4delegateTests**。

1.  实现一个`Delegate`类，可以使用所有必需的方法包装单个处理程序，并支持回调的单个 int 参数。

1.  更新模板类以支持多播。

1.  将`Delegate`类转换为模板，可以接受定义回调函数使用的参数类型的单个模板参数。

1.  将`Delegate`模板转换为可变参数模板，可以接受零个或多个定义传递给回调函数的类型的参数。

按照上述步骤后，预期输出如下：

![图 4.34：Delegate 成功实现的输出](img/C14583_04_34.jpg)

###### 图 4.34：Delegate 成功实现的输出

#### 注意

此活动的解决方案可在第 673 页找到。

## 摘要

在本章中，我们实现了一种数据和方法隐藏的设计方法 PIMPL，它具有减少依赖关系和减少构建时间的附加好处。然后，我们直接将函数对象实现为自定义类，然后作为 lambda 函数。然后，我们通过深入研究可变参数模板来扩展我们的模板编程技能，最终实现了一个可用于事件回调处理的模板。在下一章中，我们将学习如何使用 C++的特性来开发具有多个线程的程序，并通过并发构造来管理它们的协作。
