

# 创建更好的测试确认

本章介绍了*第三部分*，其中我们将 TDD 库扩展以支持日志库不断增长的需求。本书的*第一部分*，*测试 MVP*，开发了一个基本的单元测试库，*第二部分*，*日志库*，开始使用单元测试库构建日志库。现在，我们正在遵循 TDD，它鼓励在基本测试运行良好后进行增强。

嗯，我们成功使基本的单元测试库运行起来，并通过构建日志库证明了其价值。从某种意义上说，日志库就像是单元测试库的系统测试。现在，是时候增强单元测试库了。

本章向单元测试库添加了一种全新的确认类型。首先，我们将查看现有的确认，以了解它们如何改进以及新解决方案将是什么样子。

新的确认将更加直观、灵活和可扩展。并且请记住，不仅要关注本章开发的代码，还要关注过程。这是因为我们将使用 TDD 来编写一些测试，从简单的解决方案开始，然后增强测试以创建更好的解决方案。

本章将涵盖以下主要内容：

+   当前确认存在的问题

+   如何简化字符串确认

+   增强单元测试库以支持 Hamcrest 风格的确认

+   添加更多 Hamcrest 匹配器类型

# 技术要求

本章中所有代码都使用标准 C++，它基于任何现代 C++ 20 或更高版本的编译器和标准库。代码基于并继续增强本书第一部分*测试 MVP*中的测试库。

你可以在以下 GitHub 仓库中找到本章所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

# 当前确认存在的问题

在我们开始进行更改之前，我们应该对为什么要这样做有一个想法。TDD 完全是关于客户体验的。我们如何设计出易于使用且直观的东西？让我们先看看几个现有的测试：

```cpp
TEST("Test string and string literal confirms")
{
    std::string result = "abc";
    CONFIRM("abc", result);
}
TEST("Test float confirms")
{
    float f1 = 0.1f;
    float f2 = 0.2f;
    float sum = f1 + f2;
    float expected = 0.3f;
    CONFIRM(expected, sum);
}
```

这些测试已经很好地服务了，而且很简单，对吧？我们在这里关注的不光是测试本身，而是确认。这种确认风格被称为*经典风格*。

我们该如何大声说出第一个确认？可能如下所示：“*确认 abc 的预期值与结果值匹配*。”

这还不错，但有点尴尬。这不是人们通常说话的方式。不查看任何代码，表达相同内容的一种更自然的方式是：“*确认结果等于 abc*。”

初看之下，我们可能只需要颠倒参数的顺序，将实际值放在预期值之前。但这里缺少了一个部分。我们如何知道一个确认是在检查相等性呢？我们知道，因为现有的`confirm`函数只会检查这一项。这也意味着`CONFIRM`宏也只知道如何检查相等性。

对于布尔值，我们有一个更好的解决方案，因为我们创建了特殊的`CONFIRM_TRUE`和`CONFIRM_FALSE`宏，它们易于使用和理解。而且因为布尔版本只接受一个参数，所以不存在预期值与实际值顺序的问题。

有一个更好的解决方案，与更自然的确认方式相一致。这个更好的解决方案使用了一种称为匹配器的东西，被称为*Hamcrest 风格*。名字“Hamcrest”只是将“matchers”这个词的字母顺序重新排列。以下是一个用 Hamcrest 风格编写的测试示例：

```cpp
TEST("Test can use hamcrest style confirm")
{
    int ten = 10;
    CONFIRM_THAT(ten, Equals(10));
}
```

我们在这本书中并不是真正设计 Hamcrest 风格的。这种风格已经存在，并且在其他测试库中很常见。本书中测试库将预期值放在实际值之前，遵循经典风格的常见做法。

想象一下，如果你要重新发明一个更好的灯开关。我曾在一些尝试过的建筑中待过。灯开关在某些方面可能实际上更好。但如果它不符合正常的预期，那么人们会感到困惑。

这本书开始时我们使用的经典确认也是如此。我本可以将确认设计成将实际值放在前面，也许那样会更好。但对于任何对现有测试库稍有了解的人来说，这将是出乎意料的。

这在创建 TDD 设计时提出了一个值得考虑的要点。有时，当客户期望的是次优解决方案时，次优解决方案反而更好。记住，我们设计的任何东西都应该易于使用且直观。目标不是创造终极和最现代的设计，而是创造用户会满意的东西。

这就是为什么 Hamcrest 匹配器可以工作。设计并不是仅仅交换预期值和实际值的顺序，因为仅仅交换顺序本身只会让用户感到困惑。

Hamcrest 工作得很好，因为还增加了一些其他东西：匹配器。注意确认中的`Equals(10)`部分。`Equals`是一个匹配器，它清楚地说明了确认正在做什么。匹配器与更直观的顺序结合，为解决方案提供了足够的优势，以克服人们转向新做事方式的自然抵触。Hamcrest 风格不仅仅是一个更好的灯开关。Hamcrest 足够不同，提供了足够的价值，避免了稍微好一些但不同的解决方案的困惑。

此外，请注意，宏的名称已从 `CONFIRM` 更改为 `CONFIRM_THAT`。名称更改是避免混淆的另一种方式，并允许用户继续使用较老的经典风格或选择较新的 Hamcrest 风格。

现在我们有一个地方可以指定像 `Equals` 这样的东西，我们也可以使用不同的匹配器，比如 `GreaterThan` 或 `BeginsWith`。想象一下，如果你想要确认某些文本以某些预期的字符开始，你会如何编写这样的测试？你必须在确认之外检查开始文本，然后确认检查的结果。使用 Hamcrest 风格和适当的匹配器，你可以用单行确认来确认文本。而且你得到了一个更易于阅读的确认，这清楚地表明了正在确认的内容。

如果您找不到符合您需求的匹配器，您总是可以编写自己的来做到您需要的精确程度。因此，Hamcrest 是可扩展的。

在深入探讨新的 Hamcrest 设计之前，下一节将稍微偏离一下，解释对现有的经典 `confirm` 模板函数的改进。这个改进将在 Hamcrest 设计中使用，因此首先理解这个改进将有助于我们稍后到达 Hamcrest 代码解释时。

# 简化字符串确认

当我编写本章的代码时，我遇到了一个确认字符串数据类型的问题，这让我想起了我们在*第五章*“添加更多确认类型”中添加对字符串确认支持的情况。第五章的动机因素是为了让代码能够编译，因为我们不能将 `std::string` 传递给 `std::to_string` 函数。我将在下面简要地再次解释这个问题。

我不确定确切的原因，但我想 C++ 标准库的设计者认为没有必要提供接受 `std::string` 的 `std::to_string` 重载，因为没有必要的转换。字符串已经是字符串了！为什么要把某物转换成它已经是的东西呢？

可能这个决定是有意为之，也可能是一个疏忽。但确实，如果有一个将字符串转换为字符串的转换，对于需要将它们的泛型类型转换为字符串的模板函数来说，这会大有帮助。因为没有这个重载，我们不得不采取额外的步骤来避免编译错误。我们需要的是一个可以将任何类型转换为字符串的 `to_string` 函数，即使类型已经是字符串。如果我们总是能够将类型转换为字符串，那么模板就不需要为字符串进行特殊化。

在*第五章*中，我们介绍了这个模板：

```cpp
template <typename T>
void confirm (
    T const & expected,
    T const & actual,
    int line)
{
    if (actual != expected)
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

`confirm`函数接受两个模板参数，称为`expected`和`actual`，它们用于比较相等性。如果不相等，则函数将这两个参数传递给抛出的异常。参数需要根据需要通过`ActualConfirmException`构造函数转换为字符串。

这是我们遇到问题的所在。如果使用字符串调用`confirm`模板函数，那么它将无法编译，因为字符串不能通过调用`std::to_string`转换为字符串。

在第五章中我们采取的解决方案是使用直接接受字符串的非模板版本的`confirm`函数进行重载。我们实际上创建了两个重载，一个用于字符串，一个用于字符串视图。这解决了问题，但留下了以下两个额外的重载：

```cpp
inline void confirm (
    std::string_view expected,
    std::string_view actual,
    int line)
{
    if (actual != expected)
    {
        throw ActualConfirmException(
            expected,
            actual,
            line);
    }
}
inline void confirm (
    std::string const & expected,
    std::string const & actual,
    int line)
{
    confirm(
        std::string_view(expected),
        std::string_view(actual),
        line);
}
```

当使用字符串调用`confirm`时，这些重载会代替模板使用。接受`std::string`类型的版本会调用接受`std::string_view`类型的版本，该版本直接使用`expected`和`actual`参数，而不尝试调用`std::to_string`。

在当时，这并不是一个糟糕的解决方案，因为我们已经有了`confirm`函数的额外重载，用于 bool 类型和各种浮点类型。为字符串添加两个额外的重载是可以接受的。稍后，你将看到一个小小的改动将使我们能够移除这两个字符串重载。

现在我们回到本章将要讨论的新 Hamcrest 设计中的字符串数据类型转换问题。即使对于 bool 或浮点类型，我们也不再需要额外的`confirm`重载。在我开发新解决方案的过程中，我回到了第五章中较早的解决方案，并决定重构现有的经典确认，以便两种解决方案相似。

我们将在本章后面讨论新的设计。但为了避免打断这个解释，我决定现在先绕道解释如何移除经典`confirm`函数的字符串和字符串视图重载的需求。现在通过这个解释应该也会更容易理解新的 Hamcrest 设计，因为你已经熟悉了这个解决方案的这部分。

此外，我想补充一点，TDD 有助于这种重构。因为我们已经有了经典确认的现有测试，我们可以移除`confirm`的字符串重载，并确保所有测试继续通过。我之前在项目中工作过，只有新代码会使用更好的解决方案，我们必须保持现有代码不变，以避免引入错误。这样做只是让代码更难维护，因为现在同一个项目中会有两种不同的解决方案。良好的测试有助于给你所需的信心，以便更改现有代码。

好吧，问题的核心是 C++ 标准库不包括与字符串一起工作的 `to_string` 重载。虽然添加我们自己的 `to_string` 版本到 `std` 命名空间可能很有吸引力，但这是不允许的。它可能工作，我确信很多人已经这样做了。但是，将任何函数添加到 `std` 命名空间是技术上的未定义行为。有一些非常具体的情况，我们被允许将某些内容添加到 `std` 命名空间，不幸的是，这并不是允许的例外之一。

我们将需要我们自己的 `to_string` 版本。我们只是不能将其放入 `std` 命名空间。这是一个问题，因为当我们调用 `to_string` 时，我们目前通过调用 `std::to_string` 来指定命名空间。我们需要做的是简单地调用 `to_string` 而不带任何命名空间，让编译器在 `std` 命名空间中查找与数值类型一起工作的 `to_string` 版本，或者在我们的命名空间中查找我们新的与字符串一起工作的版本。新的 `to_string` 函数和修改后的 `confirm` 模板函数看起来像这样：

```cpp
inline std::string to_string (std::string const & str)
{
    return str;
}
template <typename ExpectedT, typename ActualT>
void confirm (
    ExpectedT const & expected,
    ActualT const & actual,
    int line)
{
    using std::to_string;
    using MereTDD::to_string;
    if (actual != expected)
    {
        throw ActualConfirmException(
            to_string(expected),
            to_string(actual),
            line);
    }
}
```

我们可以移除接受字符串视图和字符串的 `confirm` 的两个重载。现在，`confirm` 模板函数将适用于字符串。

新的接受 `std::string` 的 `to_string` 函数只需要返回相同的字符串。我们实际上不需要另一个与字符串视图一起工作的 `to_string` 函数。

`confirm` 模板函数稍微复杂一些，因为它现在需要两种类型，`ExpectedT` 和 `ActualT`。这两种类型是用于那些我们需要比较字符串字面量和字符串的情况，例如以下测试中所示：

```cpp
TEST("Test string and string literal confirms")
{
    std::string result = "abc";
    CONFIRM("abc", result);
}
```

这个测试之所以在只有一个 `confirm` 模板参数时能够编译，是因为它没有调用模板。编译器将 `"abc"` 字符串字面量转换为字符串，并调用接受两个字符串的重载的 `confirm`。或者，它可能将字符串字面量和字符串都转换为字符串视图，并调用接受两个字符串视图的重载的 `confirm`。无论如何，因为我们有单独的 `confirm` 重载，编译器能够使其工作。

现在我们已经移除了处理字符串的 `confirm` 重载，我们只剩下模板，我们需要让它接受不同的类型以便编译。我知道，我们仍然有处理布尔型和浮点型的重载。我只是在谈论我们可以移除的字符串重载。

在新的模板中，你可以看到我们没有指定任何命名空间就调用了`to_string`。由于模板函数内部有两个 using 语句，编译器能够找到所需的`to_string`版本。第一个 using 语句告诉编译器应该考虑`std`命名空间中所有的`to_string`重载。第二个 using 语句告诉编译器还应考虑在`MereTDD`命名空间中找到的任何`to_string`函数。

当`confirm`函数使用数值类型调用时，编译器现在能够找到一个与数值类型兼容的`to_string`版本。当需要时，编译器也可以找到我们新的与字符串兼容的`to_string`函数。我们不再需要限制编译器只查找`std`命名空间。

现在，我们可以回到新的 Hamcrest 风格设计，我们将在下一节中完成。Hamcrest 设计最终将使用与这里刚刚描述的类似解决方案。

# 增强测试库以支持 Hamcrest 匹配器

一旦基本实现工作正常并通过测试，TDD（测试驱动开发）会引导我们通过创建更多测试并让新测试通过来增强设计。这正是本章的全部内容。我们正在增强经典风格的确认方式以支持 Hamcrest 风格。

让我们从创建一个新文件开始，这个文件叫做`Hamcrest.cpp`，位于`tests`文件夹中。现在，整个项目结构应该看起来像这样：

```cpp
MereTDD project root folder
    Test.h
    tests folder
        main.cpp
        Confirm.cpp
        Creation.cpp
        Hamcrest.cpp
        Setup.cpp
```

如果你一直跟随这本书中的所有代码，记得我们正在回到我们在*第七章*中最后工作的*MereTDD*项目，*测试设置和清理*。这不是*MereMemo*日志项目。

我们需要支持的 Hamcrest 风格测试放在`Hamcrest.cpp`文件中，这样新的文件看起来就像这样：

```cpp
#include "../Test.h"
TEST("Test can use hamcrest style confirm")
{
    int ten = 10;
    CONFIRM_THAT(ten, Equals(10));
}
```

我们不妨从新的`CONFIRM_THAT`宏开始，它位于`Test.h`文件的末尾，紧随其他`CONFIRM`宏之后，如下所示：

```cpp
#define CONFIRM_FALSE( actual ) \
    MereTDD::confirm(false, actual, __LINE__)
#define CONFIRM_TRUE( actual ) \
    MereTDD::confirm(true, actual, __LINE__)
#define CONFIRM( expected, actual ) \
    MereTDD::confirm(expected, actual, __LINE__)
#define CONFIRM_THAT( actual, matcher ) \
    MereTDD::confirm_that(actual, matcher, __LINE__)
```

`CONFIRM_THAT`宏与`CONFIRM`宏类似，除了`actual`参数放在第一位，而不是`expected`参数，我们有一个名为`matcher`的参数。我们还将调用一个新的函数`confirm_that`。这个新函数有助于使保持经典风格的`confirm`重载与 Hamcrest 风格的`confirm_that`函数分开变得更加简单。

我们不需要像`confirm`那样所有的重载。`confirm_that`函数可以用一个单独的模板函数实现。将这个新的模板放在`Test.h`文件中，紧随经典的`confirm`模板函数之后。这两个模板函数应该看起来像这样：

```cpp
template <typename ExpectedT, typename ActualT>
void confirm (
    ExpectedT const & expected,
    ActualT const & actual,
    int line)
{
    using std::to_string;
    using MereTDD::to_string;
    if (actual != expected)
    {
        throw ActualConfirmException(
            to_string(expected),
            to_string(actual),
            line);
    }
}
template <typename ActualT, typename MatcherT>
inline void confirm_that (
    ActualT const & actual,
    MatcherT const & matcher,
    int line)
{
    using std::to_string;
    using MereTDD::to_string;
    if (not matcher.pass(actual))
    {
        throw ActualConfirmException(
            to_string(matcher),
            to_string(actual),
            line);
    }
}
```

我们只添加了`confirm_that`函数。我决定展示两个函数，这样你可以更容易地看到它们之间的差异。注意，现在，`ActualT`类型被放在了第一位。顺序实际上并不重要，但我喜欢将模板参数按照合理的顺序排列。我们不再有`ExpectedT`类型；相反，我们有一个`MatcherT`类型。

新的模板函数的名称也不同，因此由于相似的模板参数而导致的歧义不存在。新的模板函数被称为 `confirm_that`。

当经典 `confirm` 函数直接比较 `actual` 参数和 `expected` 参数时，新的 `confirm_that` 函数会调用 `matcher` 上的 `pass` 方法来执行检查。我们并不真正知道 `matcher` 在 `pass` 方法中会做什么，因为这取决于 `matcher`。而且，由于任何类型之间的比较变化都被封装在 `matcher` 中，我们不需要像经典 `confirm` 函数那样重载 `confirm_that` 函数。我们仍然需要特殊的代码，但差异将由本设计中的 `matcher` 处理。

正是在这里我意识到，需要为将 `matcher` 和 `actual` 参数转换为字符串找到一个不同的解决方案。仅仅为了避免当 `ActualT` 的类型为字符串时调用 `to_string`，而重写 `confirm_that` 看起来毫无意义。因此，我停止调用 `std::to_string(actual)`，而是开始调用 `to_string(actual)`。为了让编译器找到必要的 `to_string` 函数，需要使用 `using` 语句。这正是前一小节中描述的简化字符串比较的解释。

现在我们有了 `confirm_that` 模板，我们可以专注于 `matcher`。我们需要能够调用一个 `pass` 方法并将 `matcher` 转换为字符串。让我们创建一个所有匹配器都可以继承的基类，这样它们都将有一个共同的接口。将这个基类和 `to_string` 函数放在 `Test.h` 中的 `confirm_that` 函数之后，如下所示：

```cpp
class Matcher
{
public:
    virtual ~Matcher () = default;
    Matcher (Matcher const & other) = delete;
    Matcher (Matcher && other) = delete;
    virtual std::string to_string () const = 0;
    Matcher & operator = (Matcher const & rhs) = delete;
    Matcher & operator = (Matcher && rhs) = delete;
protected:
    Matcher () = default;
};
inline std::string to_string (Matcher const & matcher)
{
    return matcher.to_string();
}
```

`to_string` 函数将使我们通过调用 `Matcher` 基类中的虚拟 `to_string` 方法，将匹配器转换为字符串。注意在 `Matcher` 类中并没有 `pass` 方法。

`Matcher` 类本身是一个基类，不需要被复制或赋值。`Matcher` 类定义的唯一公共接口是一个 `to_string` 方法，所有匹配器都将实现这个方法，将自身转换为可以发送到测试运行摘要报告的字符串。

`pass` 方法怎么了？嗯，`pass` 方法需要接受实际类型，该类型将用于确定实际值是否与预期值匹配。预期值本身将保存在派生匹配器类中。实际值将传递给 `pass` 方法。

实际值和预期值接受的类型将完全由派生匹配器类控制。因为类型可以从一个匹配器的使用改变到另一个使用，所以我们不能在`Matcher`基类中定义一个`pass`方法。这是可以接受的，因为`confirm_that`模板不与`Matcher`基类一起工作。`confirm_that`模板将了解实际的匹配器派生类，并且可以直接作为非虚方法调用`pass`方法。

`to_string`方法不同，因为我们想在接受任何`Matcher`引用的`to_string`辅助函数内部调用虚拟的`Matcher::to_string`方法。

因此，当将匹配器转换为字符串时，我们对待所有匹配器都是一样的，并通过虚拟的`to_string`方法进行。而在调用`pass`时，我们直接与真实的匹配器类一起工作，并直接调用`pass`。

让我们看看一个真实的匹配器类将是什么样子。我们正在实现的测试使用了一个名为`Equals`的匹配器。我们可以在`Matcher`类和`to_string`函数之后立即创建派生的`Equals`类，如下所示：

```cpp
template <typename T>
class Equals : public Matcher
{
public:
    Equals (T const & expected)
    : mExpected(expected)
    { }
    bool pass (T const & actual) const
    {
        return actual == mExpected;
    }
    std::string to_string () const override
    {
        using std::to_string;
        using MereTDD::to_string;
        return to_string(mExpected);
    }
private:
    T mExpected;
};
```

`Equals`类是另一个模板，因为它需要持有正确的预期值类型，并且它需要在`pass`方法中使用相同的类型作为`actual`参数。

注意，`to_string`重写方法使用了与我们将要使用相同的解决方案来将`mExpected`数据成员转换为字符串。我们调用`to_string`并让编译器在`std`或`MereTDD`命名空间中找到适当的匹配项。

我们需要做一个小改动才能让一切正常工作。在我们的 Hamcrest 测试中，我们使用`Equals`匹配器而不指定任何命名空间。我们可以将其称为`MereTDD::Equals`。但命名空间指定会分散测试的可读性。让我们在将使用 Hamcrest 匹配器的任何测试文件顶部添加一个`using namespace MereTDD`语句，这样我们就可以直接引用它们，如下所示：

```cpp
#include "../Test.h"
using namespace MereTDD;
TEST("Test can use hamcrest style confirm")
{
    int ten = 10;
    CONFIRM_THAT(ten, Equals(10));
}
```

这是我们支持第一个 Hamcrest 匹配器单元测试所需的一切——构建和运行测试以显示所有测试都通过。那么预期的失败会怎样呢？首先，让我们创建一个像这样的新测试：

```cpp
TEST("Test hamcrest style confirm failure")
{
    int ten = 10;
    CONFIRM_THAT(ten, Equals(9));
}
```

这个测试被设计为失败的，因为`10`不会等于`9`。我们需要构建和运行一次，只是为了从总结报告中获取失败信息。然后，我们可以添加一个调用`setExpectedFailureReason`的语句，并带有精确格式的失败信息。记住，失败信息需要完全匹配，包括所有的空格和标点符号。我知道这可能会很繁琐，但除非你正在测试自己的自定义匹配器以确保自定义匹配器能够格式化正确的错误信息，否则你不需要担心这个测试。

在获取确切的错误信息后，我们可以修改测试，将其转换为预期的失败，如下所示：

```cpp
TEST("Test hamcrest style confirm failure")
{
    std::string reason = "    Expected: 9\n";
    reason += "    Actual  : 10";
    setExpectedFailureReason(reason);
    int ten = 10;
    CONFIRM_THAT(ten, Equals(9));
}
```

再次构建和运行会显示两个 Hamcrest 测试结果，如下所示：

```cpp
------- Test: Test can use hamcrest style confirm
Passed
------- Test: Test hamcrest style confirm failure
Expected failure
    Expected: 9
    Actual  : 10
```

这是一个好的开始。我们还没有开始讨论如何设计自定义匹配器。在我们开始自定义匹配器之前，其他基本类型怎么样？我们只有几个比较整数值的 Hamcrest 测试。下一节将探讨其他基本类型并添加更多测试。

# 添加更多 Hamcrest 类型

你现在应该熟悉使用 TDD 的模式。我们添加一点东西，使其工作，然后添加更多。我们有能力使用 Hamcrest 的 `Equals` 匹配器确认整数值。现在是时候添加更多类型了。其中一些类型可能不需要额外的工作就能工作，这要归功于模板 `confirm_that` 函数。其他类型可能需要更改。我们将通过编写一些测试来找出需要做什么。

第一个测试确保其他整数类型按预期工作。将此测试添加到 `Hamcrest.cpp`：

```cpp
TEST("Test other hamcrest style integer confirms")
{
    char c1 = 'A';
    char c2 = 'A';
    CONFIRM_THAT(c1, Equals(c2));
    CONFIRM_THAT(c1, Equals('A'));
    short s1 = 10;
    short s2 = 10;
    CONFIRM_THAT(s1, Equals(s2));
    CONFIRM_THAT(s1, Equals(10));
    unsigned int ui1 = 3'000'000'000;
    unsigned int ui2 = 3'000'000'000;
    CONFIRM_THAT(ui1, Equals(ui2));
    CONFIRM_THAT(ui1, Equals(3'000'000'000));
    long long ll1 = 5'000'000'000'000LL;
    long long ll2 = 5'000'000'000'000LL;
    CONFIRM_THAT(ll1, Equals(ll2));
    CONFIRM_THAT(ll1, Equals(5'000'000'000'000LL));
}
```

首先，测试声明了一些字符，并以几种不同的方式使用 `Equals` 匹配器。第一种是测试与另一个字符的相等性。第二种使用字符字面量值 `'A'` 进行比较。

第二组确认基于短整数。我们使用 `Equals` 匹配器与另一个短整数以及整数字面量值 `10` 进行比较。

第三组确认基于无符号整数，并且再次尝试使用 `Equals` 匹配器与相同类型的另一个变量以及字面量整数进行比较。

第四组确认确保了长长类型得到支持。

我们没有创建旨在模拟其他正在测试的软件的辅助函数。你已经知道如何根据日志库中的测试在真实项目中使用确认。这就是为什么这个测试使事情简单，并且只专注于确保 `CONFIRM_THAT` 宏（它调用 `confirm_that` 模板函数）能够正常工作。

构建和运行这些测试表明，所有测试都通过，无需任何更改或增强。

关于布尔类型呢？这里有一个测试，它进入 `Hamcrest.cpp` 以测试布尔类型：

```cpp
TEST("Test hamcrest style bool confirms")
{
    bool b1 = true;
    bool b2 = true;
    CONFIRM_THAT(b1, Equals(b2));
    // This works but probably won't be used much.
    CONFIRM_THAT(b1, Equals(true));
    // When checking a bool variable for a known value,
    // the classic style is probably better.
    CONFIRM_TRUE(b1);
}
```

这个测试表明，Hamcrest 风格也适用于布尔类型。当比较一个布尔变量与另一个布尔变量时，Hamcrest 风格比经典风格更好。然而，当比较布尔变量与预期的真或假字面量时，使用经典风格实际上更易于阅读，因为我们已经简化了 `CONFIRM_TRUE` 和 `CONFIRM_FALSE` 宏。

现在，让我们通过这个测试进入 `Hamcrest.cpp` 来处理字符串。请注意，这个测试最初将无法编译，这是可以接受的。测试看起来像这样：

```cpp
TEST("Test hamcrest style string confirms")
{
    std::string s1 = "abc";
    std::string s2 = "abc";
    CONFIRM_THAT(s1, Equals(s2));     // string vs. string
    CONFIRM_THAT(s1, Equals("abc"));  // string vs. literal
    CONFIRM_THAT("abc", Equals(s1));  // literal vs. string
}
```

这个测试中有几个确认，这是可以接受的，因为它们都是相关的。注释有助于阐明每个确认正在测试的内容。

我们在新的测试中总是寻找两件事。第一是测试是否能够编译。第二是测试是否通过。目前，测试将无法编译并出现类似于以下错误的错误：

```cpp
MereTDD/tests/../Test.h: In instantiation of 'MereTDD::Equals<T>::Equals(const T&) [with T = char [4]]':
MereTDD/tests/Hamcrest.cpp:63:5:   required from here
MereTDD/tests/../Test.h:209:7: error: array used as initializer
  209 |     : mExpected(expected)
      |       ^~~~~~~~~~~~~~~~~~~
```

你可能会得到不同的行号，所以我将解释错误所指的是什么。失败发生在 `Equals` 构造函数中，如下所示：

```cpp
    Equals (T const & expected)
    : mExpected(expected)
    { }
```

在 `Hamcrest.cpp` 的第 63 行是以下这一行：

```cpp
    CONFIRM_THAT(s1, Equals("abc"));  // string vs. literal
```

我们正在尝试使用 `"abc"` 字符串字面量构造一个 `Equals` 匹配器，但这无法编译。原因是 `T` 类型是一个需要以不同方式初始化的数组。

我们需要的是一种特殊的 `Equals` 版本，它可以与字符串字面量一起工作。由于字符串字面量是一个常量字符数组，以下模板特化将有效。将这个新模板放在 `Test.h` 中，紧接在现有的 `Equals` 模板之后：

```cpp
template <typename T, std::size_t N> requires (
    std::is_same<char, std::remove_const_t<T>>::value)
class Equals<T[N]> : public Matcher
{
public:
    Equals (char const (& expected)[N])
    {
        memcpy(mExpected, expected, N);
    }
    bool pass (std::string const & actual) const
    {
        return actual == mExpected;
    }
    std::string to_string () const override
    {
        return std::string(mExpected);
    }
private:
    char mExpected[N];
};
```

我们需要在 `Test.h` 中添加一些额外的包含，用于 `cstring` 和 `type_traits`，如下所示：

```cpp
#include <cstring>
#include <map>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <vector>
```

模板特化使用了新的 C++20 功能，称为 *requires*，它帮助我们为模板参数添加约束。`requires` 关键字实际上是 C++20 中更大增强的一部分，称为 *概念*。概念是 C++ 的巨大增强，完整的解释超出了本书的范围。我们使用概念和 `requires` 关键字来简化模板特化，使其仅与字符串一起工作。模板本身接受一个 `T` 类型，就像之前一样，以及一个新的数值 `N`，它将是字符串字面量的大小。`requires` 子句确保 `T` 是一个字符。我们需要从 `T` 中移除 const 限定符，因为字符串字面量实际上是常量。

然后 `Equals` 特化声明它是一个 `T[N]` 的数组。构造函数接受一个 `N` 个字符的数组的引用，并且不再尝试直接使用构造函数的 `expected` 参数初始化 `mExpected`，而是现在调用 `memcpy` 将字符从字面量复制到 `mExpected` 数组中。`char const (& expected)[N]` 的奇怪语法是 C++ 指定不退化成简单指针的数组作为方法参数的方式。

现在 `pass` 方法可以接受一个字符串引用作为其 `actual` 参数类型，因为我们知道我们正在处理字符串。此外，`to_string` 方法可以直接从 `mExpected` 字符数组构造并返回 `std::string`。

`Equals` 模板特化和 `pass` 方法的有趣之处，也许只是理论上的好处，是我们现在可以确认一个字符串字面量等于另一个字符串字面量。我想不出任何地方会有用，但它确实可以工作，所以我们不妨像这样将其添加到测试中：

```cpp
TEST("Test hamcrest style string confirms")
{
    std::string s1 = "abc";
    std::string s2 = "abc";
    CONFIRM_THAT(s1, Equals(s2));       // string vs. string
    CONFIRM_THAT(s1, Equals("abc"));    // string vs. literal
    CONFIRM_THAT("abc", Equals(s1));    // literal vs. string
    // Probably not needed, but this works too.
    CONFIRM_THAT("abc", Equals("abc")); // literal vs. Literal
}
```

字符指针怎么样？它们在模板参数中不如字符数组常见，因为字符数组来源于字符串字面量的工作。字符指针略有不同。我们应该考虑字符指针，因为虽然它们在模板参数中不常见，但字符指针可能比字符数组更常见。以下是一个演示字符指针的测试。请注意，这个测试目前还不能编译。将以下内容添加到 `Hamcrest.cpp` 中：

```cpp
TEST("Test hamcrest style string pointer confirms")
{
    char const * sp1 = "abc";
    std::string s1 = "abc";
    char const * sp2 = s1.c_str();    // avoid sp1 and sp2 being same
    CONFIRM_THAT(sp1, Equals(sp2));   // pointer vs. pointer
    CONFIRM_THAT(sp2, Equals("abc")); // pointer vs. literal
    CONFIRM_THAT("abc", Equals(sp2)); // literal vs. pointer
    CONFIRM_THAT(sp1, Equals(s1));    // pointer vs. string
    CONFIRM_THAT(s1, Equals(sp1));    // string vs. pointer
}
```

我们可以像初始化`std::string`一样，给字符指针初始化一个字符串字面量。但是，虽然`std::string`会将文本复制到自己的内存中以便管理，字符指针只是指向字符串字面量的第一个字符。我一直在说我们在处理字符指针。但为了更具体，我们正在处理常量字符指针。代码需要使用`const`，但我在说话或写作时有时会省略`const`。

新的字符串指针测试确认了需要采取额外步骤以确保`sp1`和`sp2`指向不同的内存地址。

C++中的字符串字面量被合并，所以重复的字面量值都指向相同的内存地址。即使一个字面量如`"abc"`在源代码中可能被多次使用，最终的可执行文件中也只会有一个字符串字面量的副本。测试必须通过额外步骤来确保`sp1`和`sp2`具有不同的指针值，同时保持相同的文本。每当`std::string`用字符串字面量初始化时，字符串字面量的文本就会被复制到`std::string`中以便管理。`std::string`可能会使用动态分配的内存或栈上的局部内存。`std::string`不会仅仅指向初始化时使用的内存地址。如果我们简单地像`sp1`一样初始化`sp2`，那么两个指针都会指向相同的内存地址。但通过将`sp2`初始化为指向`s1`中的字符串，那么`sp2`就指向了与`sp1`不同的内存地址。尽管`sp1`和`sp2`指向不同的内存地址，但每个地址上文本字符的值是相同的。

好的，现在你明白了测试在做什么，它编译了吗？不。在尝试在`confirm_that`模板函数中调用`pass`方法时，构建失败了。

导致构建失败的测试中的那一行是最后的确认。编译器试图将`s1`字符串转换为常量字符指针。但这是有误导性的，因为即使我们注释掉最后的确认，构建成功，但测试在运行时仍然会失败，如下所示：

```cpp
------- Test: Test hamcrest style string pointer confirms
Failed confirm on line 75
    Expected: abc
    Actual  : abc
```

因为你可能得到不同的行号，我会解释第 75 行是测试中的第一个确认：

```cpp
    CONFIRM_THAT(sp1, Equals(sp2));   // pointer vs. pointer
```

看看测试失败的信息。它说`"abc"`不等于`"abc"`！这是怎么回事？

因为我们使用的是原始的`Equals`模板类，它只知道我们正在处理字符指针。当我们调用`pass`时，被比较的是指针值。而且因为我们采取了额外步骤确保`sp1`和`sp2`具有不同的指针值，所以测试失败了。即使两个指针所引用的文本相同，测试也会失败。

为了支持指针，我们需要对`Equals`进行另一个模板特殊化。但我们不能对任何指针类型进行特殊化，就像我们不能对任何数组类型进行特殊化一样。我们确保数组特殊化只适用于 char 数组。因此，我们也应该确保我们的指针特殊化只与 char 指针一起工作。在`Test.h`中的第二个`Equals`类之后添加这个特殊化：

```cpp
template <typename T> requires (
    std::is_same<char, std::remove_const_t<T>>::value)
class Equals<T *> : public Matcher
{
public:
    Equals (char const * expected)
    : mExpected(expected)
    { }
    bool pass (std::string const & actual) const
    {
        return actual == mExpected;
    }
    std::string to_string () const override
    {
        return mExpected;
    }
private:
    std::string mExpected;
};
```

使用这个`Equals`类的第三个版本，我们不仅修复了构建错误，所有的确认都通过了！这个模板为`T *`专门化了`Equals`，并要求`T`是一个 char 类型。

构造函数接受一个指向常量字符的指针，并用该指针初始化`mExpected`。`mExpected`数据成员是`std::string`，它知道如何从指针初始化自己。

`pass`方法也接受`std::string`，这将允许它与实际的字符串或 char 指针进行比较。此外，`to_string`方法可以直接返回`mExpected`，因为它已经是一个字符串。

当我们在*第五章*“添加更多确认类型”中添加更多经典确认时，我们添加了对浮点类型的特殊支持。我们还需要在 Hamcrest 风格中添加对确认浮点类型的特殊支持。Hamcrest 的浮点特殊化将在下一章中介绍，包括如何编写自定义匹配器。

# 摘要

我们在本章中使用了 TDD 来添加 Hamcrest 确认，甚至改进了现有的经典确认代码。没有 TDD，真实项目中的现有代码可能不会得到管理层的批准进行更改。

本章向您展示了拥有单元测试的好处，这些测试可以帮助验证在做出更改后代码的质量。我们能够重构现有的经典确认设计，以处理字符串，使其与新设计相匹配，该设计有类似的需求。这使得经典和 Hamcrest 确认可以共享类似的设计，而不是维护两种不同的设计。所有这些更改都是可能的，因为单元测试验证了一切都按预期继续运行。

本章最重要的变化是添加了 Hamcrest 风格的确认，这些确认比在*第四章*“向项目中添加测试”中开发的经典确认更直观、更灵活。此外，新的 Hamcrest 确认也是可扩展的。

我们遵循 TDD 方法添加了对 Hamcrest 确认的支持，这让我们可以简单地开始。这种简单性是关键的，因为我们很快进入了更高级的模板特殊化，甚至是一个新的 C++20 特性，称为*requires*，它允许我们指定模板应该如何使用。

TDD 使软件设计过程更加流畅——从项目开始或增强初期简单的想法，到像本章开发这样的增强解决方案。尽管我们已经有工作的 Hamcrest 确认，但我们还没有完成。我们将在下一章继续增强确认，确保我们可以确认浮点值和自定义类型值。
