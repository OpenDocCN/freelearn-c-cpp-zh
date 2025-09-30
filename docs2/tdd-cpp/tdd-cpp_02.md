# 2

# 测试结果

到目前为止，我们有一个只能有一个测试的测试库。在本章中，当尝试添加另一个测试时，您将看到会发生什么，您将看到如何增强测试库以支持多个测试。我们将需要使用 C++的一个古老且很少使用的功能，这个功能实际上源于其早期的 C 语言根源，以支持多个测试。

一旦我们有了多个测试，我们需要一种查看结果的方法。这将让您一眼就能看出是否一切顺利。最后，我们将修复结果输出，使其不再假设`std::cout`。

本章我们将涵盖以下主要主题：

+   基于异常报告单个测试结果

+   增强测试库以支持多个测试

+   总结测试结果，以便清楚地看到哪些失败和哪些通过

+   将测试结果重定向，以便输出可以流向任何流。

# 技术要求

本章中所有代码均使用标准 C++，它基于任何现代 C++ 17 或更高版本的编译器和标准库构建。代码基于上一章并继续发展。

您可以在以下 GitHub 仓库找到本章所有代码：[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)。

# 报告单个测试结果

到目前为止，我们的单个测试在运行时只是打印其硬编码的名称。早期有一些想法，我们可能需要一个除了测试名称之外的结果。这实际上是一个向代码中添加不必要或不使用的功能的良好例子。好吧，这是一个小的例子，因为我们需要一些东西来跟踪测试是否通过或失败，但它仍然是一个超越自己的好例子，因为我们实际上从未使用过`mResult`数据成员。我们现在将用一种更好的方式来跟踪测试的运行结果。

我们假设测试成功，除非发生某些导致其失败的情况。可能会发生什么？最终会有很多导致测试失败的方法。现在，我们只考虑异常。这可能是一个测试在检测到有问题时故意抛出的异常，也可能是一个意外抛出的异常。

我们不希望任何异常停止测试的运行。一个测试抛出的异常不应该成为停止运行其他测试的理由。我们仍然只有一个测试，但我们可以确保异常不会停止整个测试过程。

我们想要的是将`run`函数调用包裹在`try`块中，以便任何异常都将被视为失败，如下所示：

```cpp
inline void runTests ()
{
    for (auto * test: getTests())
    {
        try
        {
            test->run();
        }
        catch (...)
        {
            test->setFailed("Unexpected exception thrown.");
        }
    }
}
```

当捕获到异常时，我们想要做两件事。第一是标记测试为失败。第二是设置一个消息，以便可以报告结果。问题是我们在`TestInterface`类上没有名为`setFailed`的方法。实际上，首先编写我们希望它成为的样子是很好的。

事实上，`TestInterface` 的想法是使其成为一组纯虚方法，就像一个接口。我们可以添加一个名为 `setFailed` 的新方法，但实现将需要在派生类中编写。这似乎是测试的一个基本部分，能够保存结果和消息。

因此，让我们重构设计，将 `TestInterface` 改造成一个更基础的类，并改名为 `TestBase`。我们还可以将 `TEST` 宏内部声明的类中的数据成员移动到 `TestBase` 类中：

```cpp
class TestBase
{
public:
    TestBase (std::string_view name)
    : mName(name), mPassed(true)
    { }
    virtual ~TestBase () = default;
    virtual void run () = 0;
    std::string_view name () const
    {
        return mName;
    }
    bool passed () const
    {
        return mPassed;
    }
    std::string_view reason () const
    {
        return mReason;
    }
    void setFailed (std::string_view reason)
    {
        mPassed = false;
        mReason = reason;
    }
private:
    std::string mName;
    bool mPassed;
    std::string mReason;
};
```

使用新的 `setFailed` 方法后，保留 `mResult` 数据成员就不再有意义了。相反，有一个 `mPassed` 成员，以及 `mName` 成员；这两个都来自 `TEST` 宏。添加一些获取方法似乎也是一个好主意，尤其是现在还有一个 `mReason` 数据成员。总的来说，每个测试现在可以存储其名称，记住它是否通过，以及失败的原因（如果失败的话）。

在 `getTests` 函数中，只需要进行细微的更改来引用 `TestBase` 类：

```cpp
inline std::vector<TestBase *> & getTests ()
{
    static std::vector<TestBase *> tests;
    return tests;
}
```

其余的更改简化了 `TEST` 宏，如下所示，以删除现在在基类中的数据成员，并从 `TestBase` 继承：

```cpp
#define TEST \
class Test : public MereTDD::TestBase \
{ \
public: \
    Test (std::string_view name) \
    : TestBase(name) \
    { \
        MereTDD::getTests().push_back(this); \
    } \
    void run () override; \
}; \
Test test("testCanBeCreated"); \
void Test::run ()
```

检查以确保一切构建并再次运行，这表明我们又回到了一个运行程序，其结果与之前相同。你将看到在重构时经常使用这种技术。在重构时，最好将任何功能更改保持在最低限度，并主要关注恢复到之前的行为。

现在，我们可以进行一些将*确实*影响可观察行为的更改。我们想要报告测试运行时发生的事情。目前，我们将输出发送到 `std::cout`。我们将在本章的后面部分更改这一点，以避免假设输出目标。第一个更改是在 `Test.h` 中包含 `iostream`：

```cpp
#define MERETDD_TEST_H
#include <iostream>
#include <string_view>
#include <vector>
```

然后，将 `runTests` 函数修改为报告正在运行的测试进度，如下所示：

```cpp
inline void runTests ()
{
    for (auto * test: getTests())
    {
        std::cout << "---------------\n"
            << test->name()
            << std::endl;
        try
        {
            test->run();
        }
        catch (...)
        {
            test->setFailed("Unexpected exception thrown.");
        }
        if (test->passed())
        {
            std::cout << "Passed"
                << std::endl;
        }
        else
        {
            std::cout << "Failed\n"
                << test->reason()
                << std::endl;
        }
    }
}
```

原始的 `try/catch` 代码保持不变。我们只是打印一些破折号作为分隔符和测试的名称。立即将这一行输出到输出流中可能是一个好主意。如果之后发生某些事情，至少测试的名称将被记录。测试运行后，检查测试是否通过，并显示适当的消息。

我们还将修改 `Creation.cpp` 中的测试，使其抛出异常以确保我们得到失败的结果。我们不再需要包含 `iostream`，因为这通常不是一个好主意，从测试本身显示任何内容。如果你想显示测试的输出，可以这样做，但测试本身中的任何输出往往会弄乱测试结果的报告。当我有时需要从测试内部显示输出时，这通常是临时的。

下面是修改后抛出整数的测试：

```cpp
#include "../Test.h"
TEST
{
    throw 1;
}
```

通常，你会编写抛出除简单`int`值之外内容的代码，但在这个阶段，我们只想展示当确实抛出某些内容时会发生什么。

现在构建和运行它显示了预期的失败，因为出现了意外的异常：

```cpp
---------------
testCanBeCreated
Failed
Unexpected exception thrown.
Program ended with exit code: 0
```

我们可以从测试中移除`throw`语句，使主体完全为空，这样测试现在就会通过了：

```cpp
---------------
testCanBeCreated
Passed
Program ended with exit code: 0
```

我们不希望为不同的场景不断修改测试。是时候添加对多个测试的支持了。

# 增强测试声明以支持多个测试

虽然单个测试可以工作，但尝试添加另一个测试却无法构建。这就是我在`Creation.cpp`中尝试添加另一个测试时所做的。其中一个测试是空的，第二个测试抛出一个整数值。这是我们刚刚试图处理的两种情况：

```cpp
#include "../Test.h"
TEST
{
}
TEST
{
    throw 1;
}
```

失败是由于`Test`类被声明了两次，以及`run`方法。每次使用`TEST`宏时，它都会声明一个新的全局`Test`类实例。每个实例都称为`test`。我们看不到这些类或实例在代码中，因为它们被`TEST`宏隐藏了。

我们需要修改`TEST`宏，使其能够生成唯一的类和实例名称。同时，我们也要修复测试本身的名称。我们不希望所有测试都使用名称`"testCanBeCreated"`，并且由于名称需要来自测试声明，我们还需要修改`TEST`宏以接受一个字符串。以下是新的`Creation.cpp`文件应该看起来像这样：

```cpp
#include "../Test.h"
TEST("Test can be created")
{
}
TEST("Test with throw can be created")
{
    throw 1;
}
```

这让我们可以为每个测试赋予句子名称，而不是像处理单个单词的函数名称那样对待名称。我们仍然需要修改`TEST`宏，但最好先从预期的使用开始，然后再让它工作。

为了生成唯一的类和实例名称，我们本可以直接要求程序员提供一些独特的东西，但类的类型名称和该类的实例名称实际上是一些编写测试的程序员不需要关心的细节。要求提供唯一的名称只会使这些细节变得可见。我们可以改用一个基本名称，并在其中添加测试声明的行号，以使类和实例名称都变得唯一。

宏有获取宏使用源代码文件行号的能力。我们只需要通过在生成的类和实例名称后附加这个行号来修改它们。

如果这很容易就好了。

所有宏都由预处理器处理。实际上，这比那要复杂一些，但以预处理器为思考方式是一种很好的简化。预处理器知道如何进行简单的文本替换和操作。编译器从未看到使用宏编写的原始代码。编译器看到的只是预处理器处理后的最终结果。

我们需要在 `Test.h` 中声明两组宏。一组将生成一个唯一的类名，例如如果第 7 行使用了 `TEST` 宏，则生成 `Test7` 这样的类名。另一组宏将生成一个唯一的实例名，例如 `test7`。

我们需要一组宏，因为从行号到像 `Test7` 这样的连接结果需要多个步骤。如果你第一次看到宏以这种方式使用，发现它们令人困惑是正常的。宏使用简单的文本替换规则，起初可能看起来像是额外的劳动。从行号到唯一名称需要多个文本替换步骤，这些步骤并不明显。宏看起来是这样的：

```cpp
#define MERETDD_CLASS_FINAL( line ) Test ## line
#define MERETDD_CLASS_RELAY( line ) MERETDD_CLASS_FINAL( line )
#define MERETDD_CLASS MERETDD_CLASS_RELAY( __LINE__ )
#define MERETDD_INSTANCE_FINAL( line ) test ## line
#define MERETDD_INSTANCE_RELAY( line ) MERETDD_INSTANCE_FINAL( line )
#define MERETDD_INSTANCE MERETDD_INSTANCE_RELAY( __LINE__ )
```

每组需要三个宏。每组中要使用的宏是最后一个，即 `MERETDD_CLASS` 和 `MERETDD_INSTANCE`。这些都将被 `relay` 宏替换，使用 `__LINE__` 的值。`relay` 宏将看到实际的行号而不是 `__LINE__`，然后 `relay` 宏将被替换为最终的宏和它所给的行号。最终的宏将使用 `##` 操作符来进行连接。我确实警告过，如果这很容易那就好了。我确信这是许多程序员避免使用宏的原因之一。至少你已经通过了这本书中最难使用的宏。

最终结果将是，例如，类名为 `Test7`，实例名为 `test7`。这两组宏之间唯一的真正区别是，类名使用大写 *T* 表示 `Test`，而实例名使用小写 *t* 表示 `test`。

需要将类和实例宏添加到 `Test.h` 中，位于需要使用它们的 `TEST` 宏定义之上。所有这些工作都是因为，尽管 `TEST` 宏看起来像使用了多行源代码，但请记住，每一行都是以反斜杠结尾的。这导致所有内容最终都位于单行代码中。这样，每次使用 `TEST` 宏时，所有行号都将相同，下一次使用时行号将不同。

新的 `TEST` 宏看起来是这样的：

```cpp
#define TEST( testName ) \
class MERETDD_CLASS : public MereTDD::TestBase \
{ \
public: \
    MERETDD_CLASS (std::string_view name) \
    : TestBase(name) \
    { \
        MereTDD::getTests().push_back(this); \
    } \
    void run () override; \
}; \
MERETDD_CLASS MERETDD_INSTANCE(testName); \
void MERETDD_CLASS::run ()
```

`MERETDD_CLASS` 宏用于声明类名，声明构造函数，声明全局实例的类型，并将 `run` 方法声明的作用域限定在类中。这四个宏都将使用相同的行号，因为每个宏的末尾都有反斜杠。

`MERETDD_INSTANCE` 宏仅使用一次来声明全局实例的名称。它也将使用与类名相同的行号。

构建项目并现在运行显示，第一个测试通过，因为它实际上并没有做任何事情，而第二个测试失败，因为它抛出了以下错误：

```cpp
---------------
Test can be created
Passed
---------------
Test with throw can be created
Failed
Unexpected exception thrown.
Program ended with exit code: 0
```

输出结束得有些突然，现在是时候修复这个问题了。我们将添加一个总结。

# 总结结果

总结可以从将要运行的测试数量开始。我曾考虑为每个测试添加一个运行计数，但最终决定不这样做，因为当前的测试没有特定的顺序。我的意思不是每次运行测试应用程序时都会以不同的顺序运行，但如果代码更改并且项目重新构建，它们可能会重新排序。这是因为创建最终应用程序时，链接器在多个`.cpp`编译单元之间没有固定的顺序。当然，我们需要将测试分散在多个文件中，以便看到重新排序，而现在，所有测试都在`Creation.cpp`中。

重点是测试根据全局实例的初始化方式注册自己。在单个`.cpp`源文件中，有一个定义的顺序，但在多个文件之间没有保证的顺序。正因为如此，我决定不在每个测试结果旁边包含一个数字。

我们将跟踪通过和失败的测试数量，并在运行所有测试的`for`循环结束时显示总结。

作为额外的好处，我们还可以将`runTests`函数更改为返回失败的测试数量。这将允许`main`函数也返回失败计数，以便脚本可以测试此值以查看测试是否通过或失败了多少。应用程序退出代码为零表示没有失败。任何非零值都表示失败的运行，并将指示失败的测试数量。

这里是`main.cpp`中的简单更改，以返回失败计数：

```cpp
int main ()
{
    return MereTDD::runTests();
}
```

然后，这是带有总结更改的新`runTests`函数。更改分为三个部分。所有这些都是一个函数。只有描述被分为三个部分。第一部分只是显示将要运行的测试数量：

```cpp
inline int runTests ()
{
    std::cout << "Running "
        << getTests().size()
        << " tests\n";
```

在第二部分中，我们需要跟踪通过和失败的测试数量，如下所示：

```cpp
    int numPassed = 0;
    int numFailed = 0;
    for (auto * test: getTests())
    {
        std::cout << "---------------\n"
            << test->name()
            << std::endl;
        try
        {
            test->run();
        }
        catch (...)
        {
            test->setFailed("Unexpected exception thrown.");
        }
        if (test->passed())
        {
            ++numPassed;
            std::cout << "Passed"
                << std::endl;
        }
        else
        {
            ++numFailed;
            std::cout << "Failed\n"
                << test->reason()
                << std::endl;
        }
    }
```

在第三部分中，在遍历所有测试并计算通过和失败的测试数量之后，我们显示一个带有计数的总结，如下所示：

```cpp
    std::cout << "---------------\n";
    if (numFailed == 0)
    {
        std::cout << "All tests passed."
            << std::endl;
    }
    else
    {
        std::cout << "Tests passed: " << numPassed
            << "\nTests failed: " << numFailed
            << std::endl;
    }
    return numFailed;
}
```

现在运行项目会显示初始计数、单个测试结果和最终总结，你还可以看到由于测试失败，应用程序退出代码是`1`：

```cpp
Running 2 tests
---------------
Test can be created
Passed
---------------
Test with throw can be created
Failed
Unexpected exception thrown.
---------------
Tests passed: 1
Tests failed: 1
Program ended with exit code: 1
```

显示退出代码的最后一行实际上不是测试应用程序的一部分。通常在运行应用程序时不会显示它。它是编写此代码的开发环境的一部分。如果你从脚本（如 Python）中运行测试应用程序作为自动化构建脚本的一部分，你通常会对退出代码感兴趣。

我们还有一项清理工作要做，与结果有关。你看，现在，所有内容都发送到`std::cout`，这个假设应该得到修复，以便结果可以发送到任何输出流。下一节将完成此清理。

# 重定向输出结果

这是一个简单的编辑，不会对到目前为止的应用程序造成任何实际变化。目前，`runTests` 函数在显示结果时直接使用 `std::cout`。我们将改变这一点，让 `main` 函数将 `std::cout` 作为参数传递给 `runTests`。实际上不会有任何变化，因为我们仍然会使用 `std::cout` 来显示结果，但这是一个更好的设计，因为它允许测试应用程序决定将结果发送到何处，而不是测试库。

我所说的测试库是指 `Test.h` 文件。这是其他应用程序包含以创建和运行测试的文件。在我们目前的项目中，它有点不同，因为我们正在编写测试来测试库本身。因此，整个应用程序就是 `Test.h` 文件和包含测试应用程序的 `tests` 文件夹。

我们首先需要将 `main.cpp` 修改为包含 `iostream`，然后将 `std::cout` 传递给 `runTests`，如下所示：

```cpp
#include "../Test.h"
#include <iostream>
int main ()
{
    return MereTDD::runTests(std::cout);
}
```

然后，我们不再需要在 `Test.h` 中包含 `iostream`，因为它实际上不需要任何输入，也不需要直接引用 `std::cout`。它只需要包含 `ostream` 以支持输出流。这可以是标准输出、一个文件或任何其他流：

```cpp
#ifndef MERETDD_TEST_H
#define MERETDD_TEST_H
#include <ostream>
#include <string_view>
#include <vector>
```

大多数更改都是将 `std::cout` 替换为一个新的参数，称为 `output`，就像在 `runTests` 函数中这样：

```cpp
inline int runTests (std::ostream & output)
{
    output << "Running "
        << getTests().size()
        << " tests\n";
    int numPassed = 0;
    int numFailed = 0;
    for (auto * test: getTests())
    {
        output << "---------------\n"
            << test->name()
            << std::endl;
```

之前代码中并没有显示所有的更改。你所需要做的就是将所有使用 `std::cout` 的地方替换为 `output`。

这是一个简单的更改，根本不影响应用程序的输出。实际上，做出这样的独立更改是好事，这样就可以将新结果与之前的结果进行比较，以确保没有发生意外变化。

# 摘要

本章介绍了宏及其根据行号生成代码的能力，作为启用多个测试的一种方式。每个测试都是一个具有自己唯一命名全局对象实例的类。

一旦支持了多个测试，你就看到了如何跟踪和报告每个测试的结果。

下一章将使用本章中的构建失败来展示 TDD 流程的第一步。我们已经遵循了这些流程步骤，但没有特别提及。你将在下一章中了解更多关于 TDD 流程的内容，以及测试库到目前为止是如何开发的，随着你理解这些原因，这些内容应该会变得更加有意义。
