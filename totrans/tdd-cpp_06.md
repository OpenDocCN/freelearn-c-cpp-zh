# 6

# 早期探索改进

我们在测试库方面已经取得了很大的进步，并且一直在使用 TDD 来达到这里。有时，在项目走得太远之前探索新想法是很重要的。在创建任何事物之后，我们将有在开始时没有的见解。并且在与设计一起工作了一段时间之后，我们将对喜欢什么以及可能想要改变什么有一个感觉。我鼓励你在继续之前花时间反思设计。

我们已经有一些正在工作的事物，并且在使用它方面有一些经验，那么我们是否可以改进些什么？

这种方法类似于 TDD 的高级过程，如*第三章*《TDD 过程》中所述。首先，我们确定我们想要如何使用某物，然后构建它，然后进行最小的工作量以使其工作并通过测试，然后增强设计。我们现在有很多事物正在工作，但我们还没有走到一个改变会变得过于困难的地步。我们将探讨如何增强整体设计的方法。

在这一点上，环顾四周看看其他类似解决方案并比较它们也是一个好主意。获取想法。并尝试一些新事物，看看它们是否可能更好。我已经这样做，并希望在本章中探讨两个主题：

+   我们能否使用 C++ 20 的新特性来获取行号，而不是使用`__LINE__`？

+   如果我们使用 lambda 表达式，测试看起来会是什么样子？

到本章结束时，你将了解在项目设计早期探索改进的重要性和涉及的过程。即使你并不总是决定接受新想法并做出改变，但你的项目会因为你有时间考虑替代方案而变得更好。

# 技术要求

本章的代码使用标准 C++，我们将尝试 C++ 20 中引入的特性。代码基于并延续前几章。

你可以在这个 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

# 无宏获取行号

C++ 20 包含了一个新的类，它将帮助我们获取行号。实际上，它包含的信息远不止行号。它包括文件名、函数名，甚至列号。然而，我们只需要行号。请注意，在撰写本书时，这个新类在我编译器的实现中有一个错误。结果是，我不得不将代码放回到本节描述的更改之前的版本。

这个新类被称为`source_location`，一旦它最终正确工作，我们可以将所有现有的`confirm`函数更改为接受`std::source_location`而不是行号的 int。一个现有的`confirm`函数的例子如下：

```cpp
inline void confirm (
    bool expected,
    bool actual,
    int line)
{
    if (actual != expected)
    {
        throw BoolConfirmException(expected, line);
    }
}
```

我们最终可以通过将所有`confirm`函数，包括模板重载，改为类似以下形式来更新确认函数以使用`std::source_location`：

```cpp
inline void confirm (
    bool expected, 
    bool actual,
    const std::source_location location = 
        std::source_location::current())
{
    if (actual != expected)
    {
        throw BoolConfirmException(expected, location.line());
    }
}
```

我们现在不会因为 bug 而进行这些更改。只要项目中只有一个源文件尝试使用`source_location`，代码就能正常工作。一旦多个源文件尝试使用`source_location`，就会产生链接器警告，并且该行方法返回错误数据。这个 bug 最终应该会被修复，我保留这本书中的这一部分，因为它是一个更好的方法。根据你使用的编译器，你现在可能已经开始使用`source_location`了。

不仅最后一个参数的类型和名称发生了变化，当抛出异常时，传递给异常的行号也需要更改。注意新参数包含一个默认值，该值被设置为当前位置。默认参数值意味着我们不再需要传递行号。新位置将获得一个包含当前行号的默认值。

我们需要在`Test.h`的顶部包含`source_location`的头文件，如下所示：

```cpp
#include <ostream>
#include <source_location>
#include <string_view>
#include <vector>
```

调用`confirm`的宏需要更新，不再需要担心行号：

```cpp
#define CONFIRM_FALSE( actual ) \
    MereTDD::confirm(false, actual)
#define CONFIRM_TRUE( actual ) \
    MereTDD::confirm(true, actual)
#define CONFIRM( expected, actual ) \
    MereTDD::confirm(expected, actual)
```

一旦`source_location`正常工作，我们就真的不再需要这些宏了。前两个仍然有用，因为它们消除了指定预期布尔值的需求。此外，所有三个都有点有用，因为它们封装了`MereTDD`命名空间的指定。尽管从技术上讲我们不需要继续使用宏，但我喜欢继续使用它们，因为我认为全部大写字母的名称有助于使确认在测试中更加突出。

这个改进将非常小，仅限于`confirm`函数和宏。那么，即使我们目前还不能使用`source_location`，我们是否仍然应该迁移到 C++ 20 呢？我认为是的。至少，这个 bug 表明标准库总是在不断变化，通常使用最新的编译器和标准库是最好的选择。此外，书中后面将使用到的一些特性只能在 C++20 中找到。例如，我们将使用`std::map`类和 C++20 中添加的一个有用方法来确定映射是否已包含元素。我们将在*第十二章*“创建更好的测试确认”中使用*概念*，这些概念仅在 C++20 中存在。

下一个改进将更加复杂。

# 探索测试中的 lambda 表达式

开发者避免在代码中使用宏变得越来越普遍。我同意现在几乎不再需要宏。从上一节中的`std::source_location`来看，使用宏的最后一个理由已经被消除。

一些公司甚至可能在其代码的任何地方都禁止使用宏。我认为这有点过分，尤其是考虑到`std::source_location`的问题。宏仍然有能力将代码封装起来，以便可以插入而不是使用宏本身。

如前节所示，`CONFIRM_TRUE`、`CONFIRM_FALSE`和`CONFIRM`宏可能不再绝对必要。我仍然喜欢它们。但如果你不想使用它们，那么你不必使用——至少在`std::source_location`在大项目中可靠工作之前。

`TEST`和`TEST_EX`宏仍然需要，因为它们封装了派生测试类的声明，为它们提供了独特的名称，并设置了代码，以便测试体可以跟随。结果看起来就像我们正在声明一个简单的函数。这是我们想要的效果。测试应该简单易写。我们现在拥有的几乎是最简单的了。但是，设计使用了宏。我们能否做些什么来消除对`TEST`和`TEST_EX`宏的需求？

无论我们做出什么改变，我们都应该保持`Creation.cpp`中声明测试的简单性，使其看起来类似于以下内容：

```cpp
TEST("Test can be created")
{
}
```

我们真正需要的是一种能够引入测试、给它命名、让测试注册自己，然后让我们编写测试函数体的东西。`TEST`宏通过隐藏从`TestBase`类派生的类的全局实例的声明来提供这种能力。这个声明被宏留下未完成，因此我们可以在大括号内提供测试函数体的内容。另一个`TEST_EX`宏通过捕获传递给宏的异常来做类似的事情。

在 C++中，还有一种编写函数体而不给函数体命名的方法。那就是声明一个*lambda*。如果我们停止使用`TEST`宏，并用 lambda 代替实现测试函数，测试会是什么样子？目前，让我们只关注那些不期望抛出异常的测试。以下是一个空测试可能的样子：

```cpp
Test test123("Test can be created") = [] ()
{
};
```

通过这个例子，我试图坚持 C++所需的语法。这假设我们有一个名为`Test`的类，我们想要创建其实例。在这个设计中，测试将重用`Test`类而不是定义一个新的类。`Test`类将重写`operator =`方法以接受 lambda。我们需要给实例一个名字，以便示例使用`test123`。为什么是`test123`？好吧，任何创建的对象实例仍然需要一个独特的名称，所以我使用数字来提供一些独特性。如果我们决定使用这个设计，我们可能需要继续使用宏来根据行号生成一个独特的数字。因此，虽然这个设计避免了为每个测试创建一个新的派生类，但它为每个测试创建了一个新的 lambda。

这个想法有一个更大的问题。代码无法编译。可能在函数内部将代码编译成功。但作为一个全局`Test`实例的声明，我们无法调用赋值运算符。我能想到的最好的办法是将 lambda 放在构造函数内部作为新的参数，如下所示：

```cpp
Test test123("Test can be created", [] ()
{
});
```

虽然这对这个测试有效，但当尝试调用`setExpectedFailureReason`方法时，它会在预期的失败测试中引起问题，因为`setExpectedFailureReason`不在 lambda 体内部的作用域内。此外，我们离我们现在简单声明测试的方式越来越远。额外的 lambda 语法以及最后的括号和分号使得正确实现这一点变得更加困难。

我至少看到过另一个使用 lambda 表达式并且似乎避免了声明唯一名称的需求的测试库，从而避免了需要使用类似以下内容的宏：

```cpp
int main ()
{
    Test("Test can be created") = [] ()
    {
    };
    return 0;
};
```

但实际上，这是调用一个名为`Test`的*函数*并将字符串字面量作为参数传递。然后，该函数返回一个临时对象，它覆盖了`operator=`，这是调用 lambda 时调用的。函数只能在其他函数或类方法内部调用。这意味着像这样需要一个解决方案需要在函数内部声明测试，并且测试不能像我们现在这样作为实例全局声明。

通常，这意味着你需要在`main`函数内部声明所有测试。或者，你可以将测试声明为简单的函数，并在`main`函数内部调用这些函数。无论哪种方式，你最终都需要修改`main`来调用每个测试函数。如果你忘记修改`main`，那么你的测试将不会运行。我们将保持`main`简单且不杂乱。在我们的解决方案中，`main`将只执行已注册的测试。

尽管由于增加的复杂性和无法调用测试方法（如`setExpectedFailureReason`）等问题，lambda 表达式对我们不起作用，但我们仍然可以稍微改进当前的设计。`TEST`和特别是`TEST_EX`宏正在执行我们可以从宏中移除的工作。

让我们从修改`Test.h`中的`TestBase`类开始，使其注册自身而不是在宏中使用派生类进行注册。此外，我们需要将`getTests`函数移动到`TestBase`类之前。我们还需要提前声明`TestBase`类，因为`getTests`使用一个指向`TestBase`的指针，如下所示：

```cpp
class TestBase;
inline std::vector<TestBase *> & getTests ()
{
    static std::vector<TestBase *> tests;
    return tests;
}
class TestBase
{
public:
    TestBase (std::string_view name)
    : mName(name), mPassed(true), mConfirmLocation(-1)
    {
        getTests().push_back(this);
    }
```

我们将保持`TestBase`的其余部分不变，因为它处理诸如名称和测试是否通过等属性。我们仍然有派生类，但这个简化的目标是移除`TEST`和`TEST_EX`宏需要执行的所有工作。

`TEST`宏需要做的绝大部分工作是声明一个带有`run`方法的派生类，该方法将被填充。现在，注册测试的需求由`TestBase`处理。可以通过创建另一个名为`TestExBase`的类来进一步简化`TEST_EX`宏，该类将处理预期的异常。在`TestBase`之后立即声明这个新类。它看起来像这样：

```cpp
template <typename ExceptionT>
class TestExBase : public TestBase
{
public:
    TestExBase (std::string_view name,
        std::string_view exceptionName)
    : TestBase(name), mExceptionName(exceptionName)
    { }
    void runEx () override
    {
        try
        {
            run();
        }
        catch (ExceptionT const &)
        {
            return;
        }
        throw MissingException(mExceptionName);
    }
private:
    std::string mExceptionName;
};
```

`TestExBase`类从`TestBase`派生，是一个模板类，旨在捕获预期的异常。此代码目前写入`TEST_EX`中，我们将更改`TEST_EX`以使用这个新的基类。

我们准备好简化`TEST`和`TEST_EX`宏。新的`TEST`宏看起来像这样：

```cpp
#define TEST( testName ) \
namespace { \
class MERETDD_CLASS : public MereTDD::TestBase \
{ \
public: \
    MERETDD_CLASS (std::string_view name) \
    : TestBase(name) \
    { } \
    void run () override; \
}; \
} /* end of unnamed namespace */ \
MERETDD_CLASS MERETDD_INSTANCE(testName); \
void MERETDD_CLASS::run ()
```

与之前相比，这稍微简单一些。构造函数不再需要在主体中包含代码，因为注册是在基类中完成的。

更大的简化在于`TEST_EX`宏，看起来像这样：

```cpp
#define TEST_EX( testName, exceptionType ) \
namespace { \
class MERETDD_CLASS : public MereTDD::TestExBase<exceptionType> \
{ \
public: \
    MERETDD_CLASS (std::string_view name, \
        std::string_view exceptionName) \
    : TestExBase(name, exceptionName) \
    { } \
    void run () override; \
}; \
} /* end of unnamed namespace */ \
MERETDD_CLASS MERETDD_INSTANCE(testName, #exceptionType); \
void MERETDD_CLASS::run ()
```

它比之前简单得多，因为所有的异常处理都在其直接基类中完成。注意，当构造实例时，宏仍然需要使用`#`运算符来指定`exceptionType`。此外，注意当指定从其派生模板类型时，它使用`exceptionType`而不使用`#`运算符。

# 摘要

本章探讨了利用 C++ 20 中的新功能从标准库而不是从预处理器获取行号来改进测试库的方法。尽管新代码现在不起作用，但它最终将使`CONFIRM_TRUE`、`CONFIRM_FALSE`和`CONFIRM`宏成为可选的。您将不再需要使用这些宏。但我仍然喜欢使用它们，因为它们有助于封装容易出错代码。而且，由于它们使用全部大写字母，宏在测试中更容易被发现。

我们还探讨了避免在声明测试时使用宏的趋势，以及如果我们使用 lambda 表达式会是什么样子。这种方法在更复杂的测试声明中几乎可行。然而，额外的复杂性并不重要，因为该设计并不适用于所有测试。

阅读关于提议的更改仍然很有价值。您可以了解其他测试库可能的工作方式，并理解为什么这本书解释了一个采用宏的解决方案。

本章还向您展示了如何在高层次上遵循 TDD（测试驱动开发）流程。在流程中增强测试的步骤可以应用于整体设计。我们能够改进并简化了`TEST`和`TEST_EX`宏，这使得所有测试都变得更好。

下一章将探讨在测试前后添加代码的需求，以帮助为测试做好准备并在测试完成后清理。
