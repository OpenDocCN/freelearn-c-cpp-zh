# 3

# TDD 流程

前两章通过展示涉及步骤向您介绍了 TDD 流程。您在声明多个测试时看到了构建失败。您看到了当我们提前编写尚未需要的代码时可能发生的情况。这是一个带有测试结果的小例子，但它仍然展示了有时代码在没有测试支持的情况下就滑入项目的容易性。您还看到了代码从简单或部分实现开始，先使其工作，然后进行增强。

我们在本章中将涵盖以下主题：

+   构建失败为何先出现，以及应将其视为流程的一部分

+   为什么你应该只编写足够通过测试的代码

+   如何增强测试并获得另一次通过

本章将首先向您介绍 TDD 流程。要获取更详细的代码演示，请参阅*第十章*，*深入探讨 TDD 流程*。

现在，是时候更刻意地学习 TDD 流程了。

# 技术要求

本章中所有代码都使用标准 C++，它基于任何现代 C++ 17 或更高版本编译器和标准库构建。代码基于上一章并继续发展。

您可以在以下 GitHub 仓库找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

# 构建失败先出现

在上一章中，您看到了要使多个测试运行起来的第一步是编写多个测试。这导致了构建失败。在编程时，编写一开始就无法构建的代码是很常见的。这些通常被认为是需要立即修复的错误或错误。随着时间的推移，大多数开发者学会了预测构建错误并避免它们。

在遵循 TDD 的过程中，我想鼓励你们停止避免构建错误，因为避免构建错误的方法通常意味着你在尝试使用新功能或更新后的代码之前，先工作于启用新功能或修改代码。这意味着你在关注细节的同时进行更改，很容易忽略更大的问题，例如使用新功能或更新代码的难易程度。

相反，首先以你认为它应该被使用的方式编写代码。这就是测试所做的那样。我在上一章中向您展示了添加另一个测试的最终结果应该看起来像这样：

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

由于错误，项目被构建但未完成。这让我们知道需要修复什么。但在进行更改之前，我展示了我们真正希望测试看起来像什么：

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

只有当我们对测试应该是什么样子有一个清晰的想法后，才会对测试进行修改。如果我没有采取这种做法，可能就会找到其他方法来命名测试。甚至可能有效。但使用起来会那么方便吗？我们能否像代码所示那样简单地声明第二个`TEST`，并立即为每个测试命名？我不知道。

但我知道，有很多次我没有遵循这个建议，结果得到了一个我不喜欢的解决方案。我不得不回去重新做工作，直到我对结果满意为止。如果一开始我就从想要的结果开始，那么我就会确保编写直接导致那个结果的代码。

所有这些实际上只是关注点的转移。与其深入到你正在编写的代码的详细设计，不如退一步，首先编写测试代码来使用你打算制作的内容。

换句话说，*让测试驱动设计*。这是 TDD 的本质。

你写的代码还不能构建，因为它依赖于其他不存在的代码，但没关系，因为这为你指明了方向，你会对此感到高兴。

从某种意义上说，从用户的角度编写代码为你设定了一个目标，在你甚至开始之前就使这个目标变得真实。不要满足于对想要的东西有一个模糊的想法。先花时间编写你希望它如何使用的代码，构建项目，然后努力修复构建错误。

知道你的项目无法构建时，真的有必要尝试构建它吗？这是一个我有时会采取的捷径，尤其是在构建需要很长时间或构建失败很明显时。我的意思是，如果我现在调用一个还不存在的函数，我通常会先编写这个函数，而不进行构建。我知道它将无法构建，并且我知道需要做什么来修复它。

但有时这种捷径可能会导致问题，比如在处理重载方法或模板方法时。你可能会编写代码来使用一个尚未存在的新的重载版本，并认为代码将无法构建，而实际上编译器会选择现有的某个重载版本来执行调用。模板也是这样。

你可以在*第五章*中找到一个很好的例子，即预期的构建失败实际上构建了，没有任何警告或错误，*添加更多确认类型*。结果并不是想要的，而先进行构建允许立即发现问题。

重点是构建你的项目会让你了解这些情况。如果你预期构建会失败，但编译器仍然能够编译，那么你就知道编译器找到了一种可能你没有预料到的方法来使代码工作。这可能会带来宝贵的见解。因为当你添加预期的新的重载时，现有的代码可能会开始调用你的新方法。总是最好意识到这种情况，而不是被难以找到的 bug 所惊吓。

当你还在努力使测试构建时，你不需要担心通过。事实上，如果最初让测试失败，这会更容易。专注于预期的用法，而不是获得通过测试。

一旦你的代码构建成功，你应该实现多少？这是下一节的主题。主要思想是尽可能少做。

# 只做通过测试所必需的。

在编写代码时，很容易想到一个方法可能被使用的所有可能性，例如，并立即编写代码来处理每种可能性。随着经验的积累，这会变得更容易，通常被认为是一种编写健壮代码的好方法，不会忘记处理不同的用例或错误条件。

我敦促你减少一次性编写所有这些内容的热情。相反，只做通过测试所必需的。然后，当你想到其他用例时，为每个用例编写一个测试，在扩展你的代码来处理它们之前。同样适用于错误情况。当你想到应该添加的一些新错误处理时，在代码中处理之前，先编写一个会导致该错误条件出现的测试。

为了了解这是如何完成的，让我们扩展测试库以允许预期异常。我们目前有两个测试用例：

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

第一个确保可以创建一个测试。它什么都不做并通过。第二个测试抛出一个异常。它实际上只是抛出一个简单的整数值`1`。这导致测试失败。看到你的一个或多个测试失败可能会让你感到泄气。但记住，我们刚刚使测试构建成功，这是你应该感到自豪的成就。

当我们在上一章最初添加第二个测试时，目标是确保可以添加多个测试。我们抛出一个整数是为了确保任何异常都会被视为失败。我们当时还没有准备好完全处理抛出的异常。这正是我们现在要做的。

我们将把现有的抛出异常的代码转换成预期的异常，但我们将遵循这里给出的建议，只做绝对最小的工作来使它工作。这意味着我们不会立即跳入一个尝试抛出多个不同异常的解决方案，我们也不会处理我们认为应该抛出异常但未抛出的情况。

由于我们正在编写测试库本身，我们的关注点有时会集中在测试本身上。在许多方面，测试变得与你要工作的任何特定项目代码相似。因此，虽然现在我们需要小心不要一次性添加大量测试，但稍后你将需要小心不要一次性添加大量尚未测试的额外代码。一旦我们将测试库发展到更完整的版本并开始使用它来创建日志库，你就会看到这种转变。到那时，这些指导原则将适用于日志库，我们希望避免在没有首先为这些场景添加测试的情况下添加处理不同日志场景的额外逻辑。

从最终用途出发，我们需要考虑当存在预期异常时，`TEST` 宏的使用应该是什么样子。我们需要传达的主要信息是我们期望抛出的异常类型。

只需要一种类型的异常。即使测试中的某些代码抛出多个异常类型，我们也不希望在每次测试中列出超过一个异常类型。这是因为，虽然代码检查不同的错误条件并为每个错误抛出不同的异常类型是可以接受的，但每个测试本身应该只测试这些错误条件中的一个。

如果你有一个有时会抛出不同异常的方法，那么你应该为导致每个异常的条件编写一个测试。每个测试都应该具体，并且始终导致单个异常或没有任何异常。如果一个测试期望抛出异常，那么为了使测试被认为是通过的，该异常应该始终被抛出。

在本章的后面部分，我们将讨论一个更复杂的情况，即预期抛出异常但没有捕获到。现在，我们只想做必要的操作。以下是新用法的外观：

```cpp
TEST_EX("Test with throw can be created", int)
{
    throw 1;
}
```

你首先会注意到，我们需要一个新的宏来传递预期抛出的异常类型。我将其命名为 `TEST_EX`，代表测试异常。在测试名称之后，有一个新的宏参数用于指定预期抛出的异常类型。在这种情况下，它是一个 `int`，因为代码抛出了 `1`。

*我们为什么需要一个新宏？*

因为宏并不是真正的函数。它们只是进行简单的文本替换。我们希望能够区分一个不期望抛出任何异常的测试与一个期望抛出异常的测试。宏没有像方法或函数那样重载的能力，每个不同版本都使用不同的参数声明。一个宏需要根据特定的参数数量来编写。

当一个测试不期望抛出任何异常时，传递一个占位符值给异常类型是没有意义的。最好有一个只接受名称的宏，表示不期望任何异常，另一个宏接受名称和异常类型。

这是一个设计需要妥协的真实例子。理想情况下，我们不需要一个新的宏。我们在这里尽我们所能利用语言提供的内容。宏是一种老技术，有自己的规则。

回到 TDD（测试驱动开发）流程，你会发现我们再次以最终用途为出发点。这个解决方案是否可接受？它目前还不存在。但如果它存在，会感觉自然吗？我认为会的。

现在尝试构建并没有真正的意义。这是一个我们会走捷径并跳过实际构建的过程。实际上，在我的编辑器中，`int`类型已经被标记为错误。

它抱怨我们错误地使用了关键字，这对你来说可能看起来也很奇怪。你不能简单地将类型（无论它们是否是关键字）作为方法参数传递。记住，尽管宏不是真正的方法，一旦宏被完全展开，编译器将永远不会看到这种奇怪的`int`使用方式。你可以将类型作为模板参数传递。但是，宏也不支持模板参数。

现在我们有了预期的使用方式，下一步是考虑实现这种使用的解决方案。我们不希望测试作者必须为期望的异常编写`try/catch`块。这正是测试库应该做的。这意味着我们需要在`Test`类中添加一个新的方法，该方法确实包含`try/catch`块。这个方法可以捕获期望的异常并暂时忽略它。我们忽略它是因为我们期望异常，这意味着如果我们捕获它，那么测试应该通过。如果我们让期望的异常在测试之外继续，那么`runTests`函数将捕获它并报告由于意外异常而失败的错误。

我们希望将捕获所有异常的操作放在`runTests`函数中，因为这是我们检测意外异常的方式。对于意外异常，我们不知道要捕获什么类型，因为我们希望准备好捕获任何东西。

在这里，我们知道期望哪种类型的异常，因为它是通过`TEST_EX`宏提供的。我们可以在`Test`类的新方法中捕获期望的异常。让我们把这个新方法叫做`runEx`。`runEx`方法需要做的只是查找期望的异常并忽略它。如果测试抛出了其他东西，`runEx`将不会捕获它。但`runTests`函数一定会捕获它。

让我们看看一些代码来更好地理解。这是`Test.h`中的`TEST_EX`宏：

```cpp
#define TEST_EX( testName, exceptionType ) \
class MERETDD_CLASS : public MereTDD::TestBase \
{ \
public: \
    MERETDD_CLASS (std::string_view name) \
    : TestBase(name) \
    { \
        MereTDD::getTests().push_back(this); \
    } \
    void runEx () override \
    { \
        try \
        { \
            run(); \
        } \
        catch (exceptionType const &) \
        { \
        } \
    } \
    void run () override; \
}; \
MERETDD_CLASS MERETDD_INSTANCE(testName); \
void MERETDD_CLASS::run ()
```

你可以看到，`runEx` 所做的只是在一个捕获了指定 `exceptionType` 的 `try/catch` 块中调用原始的 `run` 方法。在我们的特定情况下，我们将捕获一个整数并忽略它。这仅仅是将 `run` 方法包裹在一个 `try/catch` 块中，这样测试作者就不必这样做。

`runEx` 方法也是一个 *虚拟覆盖*。这是因为 `runTests` 函数需要调用 `runEx` 而不是直接调用 `run`。只有这样，预期的异常才能被捕获。我们不希望 `runTests` 有时为期望异常的测试调用 `runEx`，而为没有期望异常的测试调用 `run`。如果 `runTests` 总是调用 `runEx` 会更好。

这意味着我们需要一个默认的 `runEx` 实现来调用 `run` 而不带 `try/catch` 块。我们可以在 `TestBase` 类中这样做，因为这个类无论如何都需要声明虚拟的 `runEx` 方法。在 `TestBase` 中，`run` 和 `runEx` 方法看起来是这样的：

```cpp
    virtual void runEx ()
    {
        run();
    }
    virtual void run () = 0;
```

期望异常的 `TEST_EX` 宏将覆盖 `runEx` 以捕获异常，而不期望异常的 `TEST` 宏将使用基类 `runEx` 的实现，它直接调用 `run`。

现在，我们需要修改 `runTests` 函数，使其调用 `runEx` 而不是 `run`，如下所示：

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
        try
        {
            test->runEx();
        }
        catch (...)
        {
            test->setFailed("Unexpected exception thrown.");
        }
```

这里只显示了 `runTests` 函数的前半部分。函数的其余部分保持不变。实际上只需要更新 `try` 块中现在调用 `runEx` 的那行代码。

现在，我们可以构建项目并运行它来查看测试的表现。输出如下：

```cpp
Running 2 tests
---------------
Test can be created
Passed
---------------
Test with throw can be created
Passed
---------------
All tests passed.
Program ended with exit code: 0
```

第二个测试之前曾经失败，但现在它通过了，因为异常是预期的。我们还遵循了本节的指导原则，即只做通过测试所需的最少工作。TDD 流程的下一步是增强测试并获得另一个通过。

# 增强测试并获得另一个通过

如果一个期望异常的测试没有看到异常会发生什么？这应该是一个失败，我们将在下一部分处理它。这种情况有点不同，因为下一个 *通过* 实际上将会是一个 *失败*。

当你编写测试并遵循先做最少的工作以获得第一个通过结果，然后增强测试以获得另一个通过时，你将专注于通过。这是好的，因为我们希望所有测试最终都能通过。

任何失败几乎总是失败。在测试中拥有 *预期失败* 通常没有意义。我们接下来要做的事情有点不同寻常，这是因为我们仍在开发测试库本身。我们需要确保预期的缺失异常没有发生时能够被捕获为失败的测试。然后我们希望将这个失败的测试视为通过，因为我们正在测试测试库能够捕获这些失败的能力。

目前，我们在测试库中有一个漏洞，因为添加了一个预期抛出 int 但从未实际抛出 int 的第三个测试被视为通过测试。换句话说，这个集合中的所有测试都通过了：

```cpp
#include "../Test.h"
TEST("Test can be created")
{
}
TEST_EX("Test with throw can be created", int)
{
    throw 1;
}
TEST_EX("Test that never throws can be created", int)
{
}
```

构建这个没有问题，运行它显示所有三个测试都通过：

```cpp
Running 3 tests
---------------
Test can be created
Passed
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Passed
---------------
All tests passed.
Program ended with exit code: 0
```

这不是我们想要的。第三个测试应该失败，因为它预期抛出一个 int，但并没有发生。但这也违反了所有测试都应该通过的目标。没有办法有一个预期的失败。当然，我们可能能够将这个概念添加到测试库中，但这会增加额外的复杂性。

如果我们添加测试失败但仍然被视为通过的能力，那么如果测试由于某些意外原因失败会发生什么？很容易编写一个坏测试，由于多个原因失败，但实际上报告为通过，因为失败是预期的。

在写这篇文档的时候，我最初决定不添加预期失败的能力。我的理由是所有测试都应该通过。但这样我们就陷入了困境，因为否则我们如何验证测试库本身是否能够正确地检测到缺失的预期异常？

我们需要关闭第三次测试暴露的漏洞。

对于这个困境没有好的答案。所以，我将做的是让这个新的测试失败，然后添加将失败视为成功的能力。我不喜欢其他选择，比如在代码中留下测试但将其注释掉，这样它实际上就不会运行，或者完全删除第三个测试。

最终说服我添加对成功失败测试支持的想法是，一切都应该被测试，特别是像确保总是抛出预期异常这样的大型功能。你可能不需要使用标记测试为预期失败的能力，但如果你需要，你将能够做同样的事情。我们处于一个独特的情况，因为我们需要测试关于测试库本身的一些东西。

好吧，让我们让新的测试失败。为此需要的最小代码量是在捕获到预期异常时返回。如果没有捕获到异常，那么我们抛出其他东西。需要更新的代码是`runEx`方法的`TEST_EX`宏重写，如下所示：

```cpp
    void runEx () override \
    { \
        try \
        { \
            run(); \
        } \
        catch (exceptionType const &) \
        { \
            return; \
        } \
        throw 1; \
    } \
```

宏的其他部分没有变化，所以这里只展示了`runEx`重写。当捕获到预期异常时，我们返回，这将导致测试通过。在`try/catch`块之后，我们抛出其他东西，这将导致测试失败。

如果你觉得看到简单的 int 值被抛出很奇怪，请记住，我们的目标是做到这一点。你永远不会想留下这样的代码，我们将在下一版本中修复这个问题。

这很有效，也很好，因为它是我们想要做到的最低限度的需求，但结果看起来很奇怪，具有误导性。以下是测试结果输出：

```cpp
Running 3 tests
---------------
Test can be created
Passed
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Failed
Unexpected exception thrown.
---------------
Tests passed: 2
Tests failed: 1
Program ended with exit code: 1
```

你可以看到我们遇到了失败，但消息显示为`Unexpected exception thrown.`。这个消息几乎是我们不希望看到的。我们希望它显示为“预期的异常没有被抛出”。在我们继续将其转换为预期失败之前，让我们先修复这个问题。

首先，我们需要一种方法让`runTests`函数能够区分意外异常和缺失异常。目前，它只是捕获所有异常，并将任何异常都视为意外的。如果我们抛出一个特殊的异常并首先捕获它，那么这可以成为异常缺失的信号。其他被捕获的任何东西都将被视为意外的。好的，这个特殊的抛出应该是什么？

抛出的最好的东西将是测试库专门为此目的定义的东西。我们可以定义一个新的类来专门处理这个。

让我们称它为`MissingException`，并在`MereTDD`命名空间内定义它，如下所示：

```cpp
class MissingException
{
public:
    MissingException (std::string_view exType)
    : mExType(exType)
    { }
    std::string_view exType () const
    {
        return mExType;
    }
private:
    std::string mExType;
};
```

不仅这个类会表明预期的异常没有被抛出，它还会跟踪应该抛出的异常类型。这个类型在 C++编译器理解类型的意义上，不是一个真正的类型。它将是该类型的文本表示。这实际上与设计非常吻合，因为这正是`TEST_EX`宏接受的，一段文本，当宏展开时，会在代码中替换为实际类型。

在`runEx`方法的`TEST_EX`宏实现中，我们可以将其更改为如下所示：

```cpp
    void runEx () override \
    { \
        try \
        { \
            run(); \
        } \
        catch (exceptionType const &) \
        { \
            return; \
        } \
        throw MereTDD::MissingException(#exceptionType); \
    } \
```

与之前抛出一个整数不同，现在的代码抛出了一个`MissingException`。注意它如何使用宏的另一个特性，即使用`#`运算符将宏参数转换为字符串字面量。通过在`exceptionType`前放置`#`，它将`TEST_EX`宏使用中提供的`int`转换为`"int"`字符串字面量，这样就可以用期望抛出的异常类型的名称来初始化`MissingException`。

我们现在抛出了一个可以识别缺失异常的特殊类型，所以剩下的唯一部分就是捕获这个异常类型并处理它。这发生在`runTests`函数中，如下所示：

```cpp
        try
        {
            test->runEx();
        }
        catch (MissingException const & ex)
        {
            std::string message = "Expected exception type ";
            message += ex.exType();
            message += " was not thrown.";
            test->setFailed(message);
        }
        catch (...)
        {
            test->setFailed("Unexpected exception thrown.");
        }
```

顺序很重要。我们需要首先尝试捕获`MissingException`，然后再捕获其他所有异常。如果我们捕获到`MissingException`，那么代码会更改显示的消息，让我们知道期望抛出但未抛出的异常类型。

现在运行项目会显示一个更适用于失败的更适用的消息，如下所示：

```cpp
Running 3 tests
---------------
Test can be created
Passed
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Failed
Expected exception type int was not thrown.
---------------
Tests passed: 2
Tests failed: 1
Program ended with exit code: 1
```

这清楚地描述了测试失败的原因。我们现在需要将失败转换为通过测试，并且保留失败消息会很好。我们只需将状态从**Failed**更改为**Expected failure**。由于我们保留了失败消息，我有一个想法，可以使将失败的测试标记为通过的功能更安全。

我所说的更安全的功能是什么意思？嗯，这是我添加预期失败能力时最大的担忧之一。一旦我们将测试标记为预期失败，那么测试因其他原因失败就会变得太容易了。那些其他原因应该被视为真正的失败，因为它们不是预期的原因。换句话说，如果我们将任何失败都视为测试通过，那么如果测试因其他原因失败会怎样？这也会被视为通过，这是不好的。我们希望将失败标记为通过，但仅限于预期失败。

在这个特定情况下，如果我们只是将失败视为通过，那么如果测试本应抛出一个整数但反而抛出一个字符串会发生什么？这肯定会引起失败，我们还需要为这种情况添加一个测试用例。我们不妨现在就添加这个测试。我们不希望将抛出不同异常的行为与完全不抛出任何异常的行为同等对待。两者都是失败，但测试应该是具体的。任何其他情况都应导致合法的失败。

让我们从最终用途出发，探讨如何最好地表达新概念。我考虑过在宏中添加一个预期的失败消息，但这将需要一个新的宏。实际上，我们需要为每个现有的宏创建一个新的宏。我们需要扩展`TEST`宏和`TEST_EX`宏，添加两个新的宏，例如`FAILED_TEST`和`FAILED_TEST_EX`。这看起来并不是一个好主意。如果我们相反，给`TestBase`类添加一个新的方法会怎样？当在新测试中使用时，它应该看起来像这样：

```cpp
// This test should fail because it throws an
// unexpected exception.
TEST("Test that throws unexpectedly can be created")
{
    setExpectedFailureReason(
        "Unexpected exception thrown.");
    throw "Unexpected";
}
// This test should fail because it does not throw
// an exception that it is expecting to be thrown.
TEST_EX("Test that never throws can be created", int)
{
    setExpectedFailureReason(
        "Expected exception type int was not thrown.");
}
// This test should fail because it throws an
// exception that does not match the expected type.
TEST_EX("Test that throws wrong type can be created", int)
{
    setExpectedFailureReason(
        "Unexpected exception thrown.");
    throw "Wrong type";
}
```

软件设计完全是关于权衡。我们正在添加将失败测试转换为通过测试的能力。代价是额外的复杂性。用户需要知道需要在测试体内部调用`setExpectedFailureReason`方法来启用此功能。但好处是，我们现在可以以安全的方式测试那些在其他情况下不可能测试的事情。另一件需要考虑的事情是，这种设置预期失败的能力很可能不需要在测试库之外使用。

预期失败的原因也有些难以正确理解。很容易遗漏一些东西，比如失败原因末尾的句号。我发现获取确切原因文本的最佳方式是让测试失败，然后从摘要描述中复制原因。

到目前为止，我们无法有一个专门寻找完全意外异常的测试。现在我们可以了。当我们期望抛出异常时，我们现在可以检查与这种情况相关的两个失败情况，即当期望的类型没有被抛出时，以及当抛出其他类型时。

所有这些都比省略这些测试或注释掉它们的替代方案要好，而且我们可以做到这一切而不需要添加更多的宏。当然，测试现在还无法编译，因为我们还没有创建`setExpectedFailureReason`方法。所以，我们现在就添加它：

```cpp
    std::string_view reason () const
    {
        return mReason;
    }
    std::string_view expectedReason () const
    {
        return mExpectedReason;
    }
    void setFailed (std::string_view reason)
    {
        mPassed = false;
        mReason = reason;
    }
    void setExpectedFailureReason (std::string_view reason)
    {
        mExpectedReason = reason;
    }
private:
    std::string mName;
    bool mPassed;
    std::string mReason;
    std::string mExpectedReason;
};
```

我们需要一个新成员变量来保存预期的原因，它将是一个空字符串，除非在测试体内部设置。我们需要`setExpectedFailureReason`方法来设置预期的失败原因，我们还需要一个`expectedReason`获取方法来检索预期的失败原因。

现在我们有了标记测试为特定预期失败原因的能力，让我们在`runTests`函数中查找预期的失败：

```cpp
        if (test->passed())
        {
            ++numPassed;
            output << "Passed"
                << std::endl;
        }
        else if (not test->expectedReason().empty() &&
            test->expectedReason() == test->reason())
        {
            ++numPassed;
            output << "Expected failure\n"
                << test->reason()
                << std::endl;
        }
        else
        {
            ++numFailed;
            output << "Failed\n"
                << test->reason()
                << std::endl;
        }
```

你可以看到在`else if`块中为未通过测试添加的新测试。我们首先确保预期的原因不是空的，并且它与实际失败原因匹配。如果预期的失败原因与实际失败原因匹配，那么我们因为预期的失败而将这个测试视为通过。

现在构建项目并运行它显示所有五个测试都通过了：

```cpp
Running 5 tests
---------------
Test can be created
Passed
---------------
Test that throws unexpectedly can be created
Expected failure
Unexpected exception thrown.
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Expected failure
Expected exception type int was not thrown.
---------------
Test that throws wrong type can be created
Expected failure
Unexpected exception thrown.
---------------
All tests passed.
Program ended with exit code: 0
```

你可以看到有三个新的预期失败的测试。所有这些都是通过测试，我们现在有了期待测试失败的能力。要明智地使用它。期待测试失败并不正常。

我们还有一个场景需要考虑。我会坦白地说，在我想到这一点之前，我休息了一个小时左右。我们需要确保测试库覆盖了我们能想到的所有内容，因为你会用它来测试你的代码。你需要有很高的信心，即测试库本身尽可能没有错误。

这是我们需要处理的案例。如果有一个测试用例因为某种原因预期会失败，但实际上通过了怎么办？目前，测试库首先检查测试是否通过，如果是的话，它甚至不会查看它是否应该失败。如果通过了，那么它就通过了。

但是，如果你费尽周折设置了预期的失败原因，而测试却通过了，那么结果应该是什么？我们目前遇到的是一个本应被视为通过但实际上通过的失败。这最终应该算作失败吗？一个人可能会因为这些事情而感到头晕。

如果我们将这视为失败，那么我们就回到了起点，有一个我们想要包含但最终会失败的测试用例。这意味着我们不得不面对测试中的失败，忽略这个场景并跳过测试，或者写一个测试然后注释掉，这样它就不会正常运行，或者找到另一种解决方案。

与失败共存不是一种选择。在使用 TDD 时，你需要让你的所有测试都达到通过状态。期待失败没有任何好处。这就是我们费尽周折允许预期失败的测试失败的全部原因。然后，我们可以将这些失败称为通过，因为它们是预期的。

跳过测试也不是一个选项。如果你决定某件事真的不是问题，不需要测试，那么就另当别论。你不想有一堆无用的测试让你的项目变得杂乱。尽管如此，这似乎是一个重要的内容，我们不想跳过。

编写一个测试然后禁用它，让它不运行，也是一个坏主意。很容易忘记测试曾经存在过。

我们需要另一个解决方案。不，这并不是在增加一个层级，在这个层级中，一个本应按预期方式失败的通过测试被当作失败处理，然后我们再以某种方式将其标记为通过。我甚至不确定如何表达这个句子，所以我会让它听起来尽可能的混乱。这条路径会导致一个永无止境的通过-失败-通过-失败-通过思考循环。太复杂了。

我能想到的最好的办法是将这种情况视为**未记录的失败**。这样我们可以测试这个场景，并且总是运行测试，但避免真正的失败，这会导致自动化工具因为发现失败而拒绝构建。

这里是展示上述场景的新测试。它目前没有任何问题地通过：

```cpp
// This test should throw an unexpected exception
// but it doesn't. We need to somehow let the user
// know what happened. This will result in a missed failure.
TEST("Test that should throw unexpectedly can be created")
{
    setExpectedFailureReason(
        "Unexpected exception thrown.");
}
```

运行这个新的测试确实像这样悄无声息地通过：

```cpp
Running 6 tests
---------------
Test can be created
Passed
---------------
Test that throws unexpectedly can be created
Expected failure
Unexpected exception thrown.
---------------
Test that should throw unexpectedly can be created
Passed
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Expected failure
Expected exception type int was not thrown.
---------------
Test that throws wrong type can be created
Expected failure
Unexpected exception thrown.
---------------
All tests passed.
Program ended with exit code: 0
```

我们需要在`runTests`函数中检查通过测试时预期的错误结果是否已设置，如果是，那么就增加一个新的`numMissedFailed`计数而不是通过计数。新的计数也应该在最后总结，但只有当它不是零时。

这里是`runTests`的开始部分，其中声明了新的`numMissedFailed`计数：

```cpp
inline int runTests (std::ostream & output)
{
    output << "Running "
        << getTests().size()
        << " tests\n";
    int numPassed = 0;
    int numMissedFailed = 0;
    int numFailed = 0;
```

这里是`runTests`中检查通过测试的部分。在这里，我们需要寻找一个本应因预期失败而失败但实际通过的通过测试：

```cpp
        if (test->passed())
        {
            if (not test->expectedReason().empty())
            {
                // This test passed but it was supposed
                // to have failed.
                ++numMissedFailed;
                output << "Missed expected failure\n"
                    << "Test passed but was expected to fail."
                    << std::endl;
            }
            else
            {
                ++numPassed;
                output << "Passed"
                    << std::endl;
            }
        }
```

这里是`runTests`函数的结尾，它总结了结果。现在，如果有任何未记录的测试失败，它将显示出来：

```cpp
    output << "---------------\n";
    output << "Tests passed: " << numPassed
           << "\nTests failed: " << numFailed;
    if (numMissedFailed != 0)
    {
        output << "\nTests failures missed: " <<         numMissedFailed;
    }
    output << std::endl;
    return numFailed;
}
```

总结部分开始变得比必要的复杂。所以，现在它总是显示通过和失败的数量，如果有任何失败，只显示失败的次数。现在，对于预期会失败但最终通过的新的测试，我们会得到一个未记录的失败。

未记录的失败是否应该包含在失败计数中？我考虑过这个问题，并决定只返回所有导致这种场景的实际失败数量。记住，你几乎不可能需要编写一个你打算失败并当作通过的测试。所以，你也不应该有未记录的失败。

输出看起来像这样：

```cpp
Running 6 tests
---------------
Test can be created
Passed
---------------
Test that throws unexpectedly can be created
Expected failure
Unexpected exception thrown.
---------------
Test that should throw unexpectedly can be created
Missed expected failure
Test passed but was expected to fail.
---------------
Test with throw can be created
Passed
---------------
Test that never throws can be created
Expected failure
Expected exception type int was not thrown.
---------------
Test that throws wrong type can be created
Expected failure
Unexpected exception thrown.
---------------
Tests passed: 5
Tests failed: 0
Tests failures missed: 1
Program ended with exit code: 0
```

我们现在应该对这部分内容很熟悉了。你具备预期能够抛出异常并依赖测试失败的能力，如果异常没有被抛出，测试库会全面测试所有可能的异常组合。

这一节也多次展示了如何继续增强测试并使它们再次通过。如果你遵循这个过程，你将能够逐步构建测试以覆盖更复杂的场景。

# 摘要

本章已经将我们之前遵循的步骤明确化。

你现在知道首先以你希望代码被使用的方式编写代码，而不是深入细节并从底部向上工作以避免构建失败。从顶部工作，或者从最终用户的角度来看，会更好，这样你将得到一个让你满意的解决方案，而不是一个可构建但难以使用的解决方案。你可以通过编写你希望代码被使用的测试来实现这一点。一旦你对代码的使用方式感到满意，然后构建它并查看构建错误以修复它们。让测试通过还不是目标。这种关注点的微小变化将导致更易于使用和更直观的设计。

一旦你的代码构建完成，下一步就是只做必要的操作以确保测试通过。总有可能某个更改会导致之前通过测试现在失败。这是正常的，也是只做必要操作的另一个好理由。

最后，你可以在编写代码以通过所有测试之前，增强测试或添加更多测试。

尽管测试库远未完善。目前唯一导致测试失败的方法是抛出一个未预期的异常。你可以看到，即使在更高的层面，我们也在遵循只做必要操作、使其工作，然后增强测试以添加更多功能的做法。

下一个增强是让测试程序员检查测试中的条件，以确保一切正常工作。下一章将开始这项工作。
