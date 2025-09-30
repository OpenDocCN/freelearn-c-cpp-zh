

# 第十章：深入探讨 TDD 过程

在本章中，我们将向日志库中添加大量代码，虽然这样做很好，但这并不是本章的主要目的。

这是一章关于**测试驱动开发**（**TDD**）过程的章节。第三章不是也关于 TDD 过程吗？是的，但将前面的章节视为一个介绍。本章将详细探讨 TDD 过程，并使用更多的代码。

你将获得编写自己测试的想法，如何确定哪些是重要的，以及如何在不重写测试的情况下重构代码，你还将了解测试过多的情况以及了解许多不同类型的测试。

基本的 TDD 流程如下：

+   首先编写使用软件的自然直观方式的测试

+   即使我们需要提供模拟或存根实现，也要以最小的更改来构建代码

+   为了使基本场景正常工作

+   为了编写更多测试并增强设计

在过程中，我们将向日志库添加日志级别、标签和过滤功能。

具体来说，本章将涵盖以下主要内容：

+   发现测试中的差距

+   添加日志级别

+   添加默认标签值

+   探索过滤选项

+   添加新的标签类型

+   使用 TDD 重构标签设计

+   设计测试以过滤日志消息

+   控制要记录的内容

+   增强相对匹配的过滤功能

+   何时测试过多？

+   测试应该有多侵入性？

+   在 TDD 中，集成或系统测试去哪里？

+   其他类型的测试怎么办？

# 技术要求

本章中所有代码都使用标准 C++，它构建在任何现代 C++ 20 或更高版本的编译器和标准库之上。代码使用本书第一部分*测试 MVP*中的测试库，并继续开发在前一章中开始的日志库。

你可以在以下 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

)

# 发现测试中的差距

我们真的需要更多的测试。目前，我们只有两个日志测试：一个用于简单的日志消息，另一个用于更复杂的日志消息。这两个测试如下所示：

```cpp
TEST("Simple message can be logged")
{
    std::string message = "simple ";
    message += Util::randomString();
    MereMemo::log() << message << " with more text.";
    bool result = Util::isTextInFile(message,     "application.log");
    CONFIRM_TRUE(result);
}
TEST("Complicated message can be logged")
{
    std::string message = "complicated ";
    message += Util::randomString();
    MereMemo::log() << message
        << " double=" << 3.14
        << " quoted=" << std::quoted("in quotes");
    bool result = Util::isTextInFile(message,     "application.log");
    CONFIRM_TRUE(result);
}
```

但有没有一种好的方法来找到更多的测试？让我们看看我们到目前为止所拥有的。我喜欢从简单的测试开始。事情可以构造吗？

那就是为什么我们到目前为止的两个测试都在一个名为`Contruction.cpp`的文件中。当你寻找测试中的差距时，这是一个好起点。你为每个可以构造的东西都有一个简单的测试吗？通常，这些将是类。为你的项目提供的每个类的每个构造函数编写一个测试。

对于日志库，我们还没有任何类。因此，我创建了一个简单的测试，它调用`log`函数。然后，另一个测试以稍微复杂一些的方式调用相同的函数。

有一个论点可以提出，复杂的测试重复了一些简单测试的功能。我认为我们到目前为止做得还不错，但这是你应该注意的事情，以避免有一个测试做了另一个测试所有的事情再加上一点。只要简单测试代表了一个常见的用例，那么即使另一个测试可能做类似的事情，包含它也是有价值的。总的来说，你希望测试能够捕捉到你的代码是如何被使用的。

在寻找测试中的差距时，可以通过寻找对称性来考虑其他事情。如果你有构建测试，也许你应该考虑破坏测试。对于日志库，我们还没有这样的东西——至少，目前还没有——但这是一个需要考虑的事情。本章后面的另一个对称性例子可以找到。我们需要确认某些文本存在于文件中。为什么不包括一个类似的测试，确保某些不同的文本不存在于文件中？

主要功能是测试的好来源。想想你的代码解决了哪些问题，并为每个功能或能力编写测试。对于每个功能，创建一个简单或常见的测试，然后考虑添加一个更复杂的测试，一些探索可能出错情况的错误测试，以及一些探索更有意向的错误使用的测试，以确保你的代码按预期处理所有情况。你甚至会在下一节看到一个例子，其中添加了一个测试只是为了确保它能够编译。

本章将主要探讨缺失功能的测试。我们刚刚开始使用日志库，因此大部分新的测试都将基于新功能。这对于一个新项目来说是常见的，并且是让测试驱动开发的一个很好的方式。

下一个部分将通过首先创建测试来定义新功能，来添加一个新功能。

# 添加日志级别

日志库有一个共同的**日志级别**概念，它允许你在应用程序运行时控制记录多少信息。假设你确定了一个需要记录日志消息的错误条件。这个错误几乎总是应该被记录，但也许代码中的另一个地方你决定记录正在发生的事情可能是有用的。这个其他地方并不总是有趣的，所以避免总是看到这些日志消息会很好。

通过拥有不同的日志级别，你可以决定日志文件变得多么详细。这种方法的几个大问题包括：首先，简单地定义日志级别应该是什么，以及每个级别应该代表什么。常见的日志级别包括错误、警告、一般信息性消息和调试消息。

错误通常很容易识别，除非你还需要将它们分为普通错误和关键错误。什么使一个错误变得关键？你是否真的需要区分它们？为了支持尽可能多的不同客户，许多日志库提供了不同的日志级别，并将决定每个级别含义的任务留给程序员。

日志级别最终主要用于控制记录多少信息，这有助于在应用程序运行时没有问题或投诉的情况下减少日志文件的大小。这是一件好事，但它导致了下一个大问题。当需要进一步调查时，获取更多信息的方法只有更改日志级别，重新运行应用程序，并希望再次捕捉到问题。

对于大型应用程序，将日志级别更改为记录更多信息可能会迅速导致大量额外信息，这使得找到所需信息变得困难。额外的日志消息也可能填满存储驱动器，如果日志文件发送给供应商进行进一步处理，还可能产生额外的财务费用。调试过程通常很匆忙，因此新的日志级别只有效很短的时间。

为了绕过需要更改整个应用程序的日志级别的问题，一种常见的做法是在发现问题时临时更改代码中特定部分使用日志信息时的级别。这需要应用程序重建、部署，然后在问题解决后恢复。

所有关于日志级别的讨论如何帮助我们设计日志库？我们知道我们的目标客户是谁：一个微服务开发者，他们可能会与可以生成大量日志文件的大型应用程序一起工作。考虑什么最能帮助你的客户是一种很好的设计方法。

我们将修复已识别的两个大问题。首先，我们不会在日志库中定义任何日志级别。将不会有错误日志消息与调试日志消息之间的概念。这并不意味着将没有控制记录多少信息的方法，只是使用日志级别的整个想法在根本上是错误的。级别本身太复杂，开启和关闭它们会迅速导致信息过载和匆忙的调试会话。

在日志消息中添加额外信息，如日志级别，的想法是好的。如果我们提出一个通用的解决方案，它不仅可以用于日志级别，还可以用于其他附加信息，那么我们可以让用户添加所需和合理的任何内容。我们可以提供添加日志级别的功能，而无需实际定义这些级别将是什么以及它们代表什么。

因此，解决方案的第一部分将是一个通用 **标签** 系统。这应该避免由库定义的固定日志级别的混淆。我们仍然会提到日志级别的概念，但这仅仅是因为这个概念非常普遍。然而，我们的日志级别将更像日志级别标签，因为不会存在一个日志级别高于或低于另一个日志级别的概念。

第二部分将需要一些新的内容。根据日志级别标签的值来控制消息是否被记录，这只会导致之前同样的问题。开启日志级别最终会在所有地方打开日志，并仍然导致额外的日志消息泛滥。我们需要的是能够精细控制记录的内容，而不是在所有地方打开或关闭额外的日志记录。我们需要的能力是能够根据不仅仅是日志级别来过滤。

让我们一次考虑这两个想法。一个通用的标签系统会是什么样子？让我们编写一个测试来找出答案！我们应该在 `tests` 文件夹中创建一个名为 `Tags.cpp` 的新文件，如下所示：

```cpp
#include "../Log.h"
#include "LogTags.h"
#include "Util.h"
#include <MereTDD/Test.h>
TEST("Message can be tagged in log")
{
    std::string message = "simple tag ";
    message += Util::randomString();
    MereMemo::log(error) << message;
    std::string taggedMessage = " log_level=\"error\" ";
    taggedMessage += message;
    bool result = Util::isTextInFile(taggedMessage,          "application.log");
    CONFIRM_TRUE(result);
}
```

这次测试最重要的部分是 `log` 函数调用。我们希望它易于使用，并能快速传达给阅读代码的任何人，其中涉及一个标签。我们不希望标签被隐藏在消息中。它应该突出显示为不同，同时又不显得使用起来尴尬。

确认部分稍微复杂一些。我们希望日志文件中的输出使用 `key="value"` 格式。这意味着有一些文本后面跟着一个等号，然后是引号内的更多文本。这种格式将使我们能够通过寻找类似以下内容来轻松找到标签：

```cpp
key="value"
```

对于日志级别，我们期望输出看起来像这样：

```cpp
log_level="error"
```

我们还希望避免诸如拼写或大小写差异之类的错误。这就是为什么语法不使用字符串，因为可能会被误输入如下：

```cpp
    MereMemo::log("Eror") << message;
```

通过避免字符串，我们可以让编译器帮助确保标签的一致性。任何错误都应导致编译错误，而不是日志文件中的格式错误的标签。

由于解决方案使用函数参数，我们不需要提供特殊的 `log` 形式，如 `logError`、`logInfo` 或 `logDebug`。我们的一个目标是在库本身中避免定义特定的日志级别，而是想出一些让用户决定日志级别会是什么的东西，就像任何其他标签一样。

这也是为什么额外包含 `LogTags.h` 的原因，它也是一个新文件。这就是我们将定义我们将使用哪些日志级别的地方。我们希望定义尽可能简单，因为日志库不会定义这些。`LogTags.h` 文件应放置在 `tests` 文件夹中，如下所示：

```cpp
#ifndef MEREMEMO_TESTS_LOGTAGS_H
#define MEREMEMO_TESTS_LOGTAGS_H
#include "../Log.h"
inline MereMemo::LogLevel error("error");
inline MereMemo::LogLevel info("info");
inline MereMemo::LogLevel debug("debug");
#endif // MEREMEMO_TESTS_LOGTAGS_H
```

仅因为日志库没有定义自己的日志级别，并不意味着它不能帮助完成这个常见任务。我们可以利用库定义的一个辅助类，称为`LogLevel`。我们包含`Log.h`是为了获取访问`LogLevel`类，以便我们可以定义实例。每个实例都应该有一个名称，例如`error`，这是我们将在日志记录时使用的。构造函数还需要一个用于日志输出的字符串。可能使用与实例名称匹配的字符串是个好主意。所以，例如，错误实例得到一个`"``error"`字符串。

正是这些实例被传递给`log`函数，如下所示：

```cpp
    MereMemo::log(error) << message;
```

有一个需要注意的事项是`LogLevel`实例的命名空间。因为我们正在测试日志库本身，我们将在测试中调用`log`。每个测试体实际上是使用一个`TEST`宏定义的测试类的`run`方法的一部分。测试类本身在一个未命名的命名空间中。我想避免在使用日志级别时需要指定`MereMemo`命名空间，就像这样：

```cpp
    MereMemo::log(MereMemo::error) << message;
```

直接输入`error`而不是`MereMemo::error`要简单得多。因此，目前的解决方案是在`LogTags.h`中全局命名空间内声明日志级别的实例。我建议当你为自己的项目定义自己的标签时，你在项目的命名空间中声明这些标签。例如，可以这样操作：

```cpp
#ifndef YOUR_PROJECT_LOGTAGS_H
#define YOUR_PROJECT_LOGTAGS_H
#include <MereMemo/Log.h>
namespace yourproject
{
inline MereMemo::LogLevel error("error");
inline MereMemo::LogLevel info("info");
inline MereMemo::LogLevel debug("debug");
} // namespace yourproject
#endif // YOUR_PROJECT_LOGTAGS_H
```

然后，当你正在编写你自己的项目中的代码，该项目是你自己的命名空间的一部分时，你可以直接引用像`error`这样的标签，而不需要指定一个命名空间。你可以使用你想要的任何命名空间来代替`yourproject`。你可以在*第十四章*中看到一个很好的例子，*如何* *测试服务*，该项目同时使用了日志库和测试库。

此外，请注意，你应该从你的项目中作为单独的项目引用`Log.h`文件，并使用尖括号。这就像我们在开始日志库的工作时，不得不使用尖括号引用单元测试库包含文件时所做的。

将`MereMemo::LogLevel`的实例传递给`log`函数的一个额外好处是，我们不再需要指定`log`函数的命名空间。编译器知道在尝试解析函数名时在函数参数使用的命名空间中查找。将`error`传递给`log`函数的简单行为让编译器推断出`log`函数是在与`error`实例相同的命名空间中定义的。实际上，我在代码工作并且可以尝试不带命名空间调用`log`之后想到了这个好处。然后我能够向`Tags.cpp`添加一个看起来像这样的测试：

```cpp
TEST("log needs no namespace when used with LogLevel")
{
    log(error) << "no namespace";
}
```

在这里，你可以看到我们可以直接调用`log`而不需要指定`MereMemo`命名空间，我们可以这样做是因为编译器知道被传递的`error`实例本身就是`MereMemo`的一个成员。

如果我们尝试不带任何参数调用`log`，那么我们就需要回退到使用`MereMemo::log`而不是仅仅使用`log`。

此外，注意这个新测试是如何被识别的。它是一种简化代码的替代用法，编写一个测试可以帮助确保我们以后不会做任何会破坏更简单语法的操作。新的测试也没有确认。这是因为测试的存在只是为了确保调用`log`时没有命名空间能够编译。我们已经知道`log`可以发送日志消息到日志文件，因为其他测试已经确认了这一点。这个测试不需要重复确认。如果它能编译，那么它就完成了它的任务。

我们现在唯一需要的是`LogLevel`类的定义。记住，我们真正想要的是一个通用的标记解决方案，日志级别应该只是标记的一种类型。日志级别与其他标签之间不应该有任何特殊之处。我们不妨也定义一个`Tag`类，并让`LogLevel`从`Tag`继承。将这两个新类放在`Log.h`的顶部，就在`MereMemo`命名空间内部，如下所示：

```cpp
class Tag
{
public:
    virtual ~Tag () = default;
    std::string key () const
    {
        return mKey;
    }
    std::string text () const
    {
        return mText;
    }
protected:
    Tag (std::string const & key, std::string const & value)
    : mKey(key), mText(key + "=\"" + value + "\"")
    { }
private:
    std::string mKey;
    std::string const mText;
};
class LogLevel : public Tag
{
public:
    LogLevel (std::string const & text)
    : Tag("log_level", text)
    { }
};
```

确保这两个类都在`MereMemo`命名空间内定义。让我们从`Tag`类开始，这个类不应该被直接使用。`Tag`类应该是一个基类，以便派生类可以指定要使用的键。`Tag`类的真正目的是确保文本输出遵循`key="value"`格式。

`LogLevel`类从`Tag`类继承，并且只需要日志级别的文本。键是硬编码的，总是为`log_level`，这保证了一致性。当我们用特定的字符串声明`LogLevel`的实例并调用`log`时，我们得到了值的一致性。

日志库支持标签，甚至支持日志级别标签，但它本身并不定义任何特定的日志级别。库也不尝试对日志级别进行排序，以便像`error`这样的级别比`debug`高或低。一切只是一个由键和值组成的标签。

现在我们有了`LogLevel`和`Tag`类，它们是如何被`log`函数使用的呢？我们首先需要一个接受`Tag`参数的新重载`log`，如下所示：

```cpp
inline std::fstream log (Tag const & tag)
{
    return log(to_string(tag));
}
```

将这个新的`log`函数放在现有的`log`函数之后，并且仍然在`Log.h`中的`MereMemo`命名空间内。新的`log`函数将标签转换为字符串，并将字符串传递给现有的`log`函数。我们需要定义一个`to_string`函数，可以放在`Tag`类的定义之后，如下所示：

```cpp
inline std::string to_string (Tag const & tag)
{
    return tag.text();
}
```

`to_string`函数只是调用`Tag`类中的`text`方法来获取字符串。我们真的需要一个函数来做这个吗？我们难道不能直接在新的重载`log`函数中调用`text`方法吗？是的，我们可以这样做，但在 C++中提供名为`to_string`的函数，该函数知道如何将类转换为字符串，是一种常见的做法。

所有这些新函数都需要声明在行内，因为我们打算将日志库作为一个单独的包含文件，其他项目可以简单地包含它以开始记录。我们希望避免在 `Log.h` 文件中声明函数，然后在 `Log.cpp` 文件中实现它们，因为这要求用户将 `Log.cpp` 添加到他们的项目中，或者要求将日志库构建为一个库，然后将其链接到项目中。通过将所有内容保持在单个头文件中，我们使其他项目使用日志库变得更加容易。它实际上不是一个库——它只是一个被包含的头文件。尽管如此，我们仍然将其称为日志库。

现有的 `log` 函数需要修改以接受一个字符串。实际上它曾经用于接受一个字符串作为要记录的消息，直到我们移除了这个功能，转而返回一个流，调用者可以使用这个流来指定消息以及任何其他要记录的信息。我们打算将一个字符串参数放回 `log` 函数中，并将其命名为 `preMessage`。`log` 函数仍然会返回一个调用者可以使用的流。`preMessage` 参数将用于传递格式化的标签，`log` 函数将在返回给调用者的流之前输出 `preMessage`。修改后的 `log` 函数看起来像这样：

```cpp
inline std::fstream log (std::string_view preMessage = "")
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const tmNow =          std::chrono::system_clock::to_time_t(now);
    auto const ms = duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::fstream logFile("application.log", std::ios::app);
    logFile << std::endl
        << std::put_time(std::gmtime(&tmNow),            "%Y-%m-%dT%H:%M:%S.")
        << std::setw(3) << std::setfill('0')         << std::to_string(ms.count())
        << " " << preMessage << " ";
    return logFile;
}
```

`preMessage` 参数有一个默认值，这样 `log` 函数仍然可以在没有日志级别标签的情况下被调用。`log` 函数所做的只是发送一个时间戳，然后是 `preMessage` 参数到流中，接着是一个空格，最后让调用者能够访问返回的流。

注意，我们仍然希望日志级别标签与时间戳之间也用空格隔开。如果没有指定日志级别，则输出将有两个空格，这是一个将很快修复的细节。

我们现在拥有所有需要的工具来使用新的测试中使用的日志级别进行记录：

```cpp
    MereMemo::log(error) << message;
```

构建并运行项目显示一切通过：

```cpp
Running 1 test suites
--------------- Suite: Single Tests
------- Test: Message can be tagged in log
Passed
------- Test: log needs no namespace when used with LogLevel
Passed
------- Test: Simple message can be logged
Passed
------- Test: Complicated message can be logged
Passed
-----------------------------------
Tests passed: 4
Tests failed: 0
```

查看新的日志文件可以看到预期的日志级别：

```cpp
2022-06-25T23:52:05.842 log_level="error" simple tag 7529
2022-06-25T23:52:05.844 log_level="error" no namespace
2022-06-25T23:52:05.844  simple 248 with more text.
2022-06-25T23:52:05.844  complicated 637 double=3.14 quoted="in quotes"
```

前两个条目使用了新的日志级别。第二个条目是我们只想确保它能编译的。第三个和第四个日志条目缺少日志级别。这是因为它们从未指定日志级别。我们应该修复这个问题，并允许一些标签有默认值，这样我们就可以在不指定日志级别的情况下添加日志级别，以确保每个日志消息条目都有一个日志级别。第三个和第四个条目还有一个额外的空格，这也会被修复。下一节将添加指定默认标签的能力。

在继续之前，请注意一件事。复杂的日志条目实际上看起来已经使用了标签。这是因为我们使用`key="value"`格式格式化了消息。在文本值周围包含引号并且不在数字周围使用引号是常见的做法。当文本值内部包含空格时，引号有助于定义整个值，而数字不需要空格，因此不需要引号。

此外，请注意，我们不在等号周围添加空格。我们不会记录以下内容：

```cpp
double = 3.14
```

我们不记录这个的原因是额外的空格不是必需的，只会使处理日志输出变得更困难。虽然带有空格可能更容易阅读，但使用脚本自动处理日志文件会更困难。

同样，我们不在标签之间使用逗号。所以，我们不会这样做：

```cpp
double=3.14, quoted="in quotes"
```

在标签之间添加逗号可能会使其更容易阅读，但它们只是代码需要处理日志文件时必须处理的一个额外元素。逗号不是必需的，所以我们不会使用它们。

现在，我们可以继续添加默认标签。

# 添加默认标签值

前一节确定了有时需要在日志消息中添加标签，即使标签没有提供给`log`函数。我们可以利用这一点来添加默认日志级别标签或任何其他任何标签所需的默认值。

使用这个功能，我们开始需要日志库支持配置。我的意思是，我们希望在调用`log`之前告诉日志库如何表现，并且我们希望日志库记住这种行为。

大多数应用程序仅在配置在应用程序开始时设置一次后支持日志记录。这个配置设置通常在`main`函数的开始处完成。因此，让我们专注于添加一些简单的配置，这样我们就可以设置一些默认标签，并在日志记录时使用这些默认标签。如果我们遇到在调用`log`函数期间使用的具有相同键的默认标签和标签，那么我们将使用在`log`函数调用中提供的标签。换句话说，除非在`log`函数调用中覆盖，否则将使用默认标签。

我们将开始讨论设置默认标签值所需的内容。这是一个我们实际上不会在`main`内部设置默认值的测试用例，但我们将有一个测试来确保在`main`中设置的默认值确实出现在测试的日志输出中。我们还可以设计解决方案，以便可以在任何时间设置默认值，而不仅仅是`main`函数内部。这将使我们能够直接测试默认值的设置，而不是依赖于`main`。

即使下面的代码不在测试中，我们仍然可以先修改`main`，以确保解决方案是我们想要的。让我们将`main`修改如下：

```cpp
#include "../Log.h"
#include "LogTags.h"
#include <MereTDD/Test.h>
#include <iostream>
int main ()
{
    MereMemo::addDefaultTag(info);
    MereMemo::addDefaultTag(green);
    return MereTDD::runTests(std::cout);
}
```

我们将包含`Log.h`，以便我们可以获取我们将要编写的`addDefaultTag`函数的定义，并且我们将包含`LogTags.h`以获取对`info`日志级别和一个颜色标签的新标签的访问权限。为什么是颜色标签？因为当我们添加新测试时，我们想要寻找简单和通用的用例。我们已经有了由日志库定义的`LogLevel`标签，我们唯一需要做的是定义具有自己值的特定实例。但我们还没有定义我们自己的标签，这似乎是检查自定义标签是否也工作得好的好地方。使用流程良好，看起来用户想要定义多个默认标签似乎是合理的。

很容易走得太远，添加一大堆需要测试的新功能，但添加相关场景，例如添加两个默认标签`info`和`green`，以使测试更加通用是可行的。至少，这是我可能会一步完成的事情。你可能想要将这两个测试分开。我认为我们可以添加一个单独的测试，确保即使没有提供给`log`函数，两个标签都存在。一个标签类型由日志库提供，另一个是自定义的，这对我来说不足以要求进行单独的测试。如果它们两个都出现在日志输出中，我会很高兴。

现在我们来向`Tags.cpp`添加一个测试，如下所示：

```cpp
TEST("Default tags set in main appear in log")
{
    std::string message = "default tag ";
    message += Util::randomString();
    MereMemo::log() << message;
    std::string logLevelTag = " log_level=\"info\" ";
    std::string colorTag = " color=\"green\" ";
    bool result = Util::isTextInFile(message,          "application.log",
        {logLevelTag, colorTag});
    CONFIRM_TRUE(result);
}
```

结果表明，我很高兴我添加了两个默认标签而不是一个，因为在编写测试时，我开始思考如何验证它们两个都出现在日志文件中，那时我才意识到`isTextInFile`函数对于我们现在需要的来说太僵化了。当我们的兴趣仅限于检查特定字符串是否出现在文件中时，`isTextInFile`函数表现良好，但现在我们正在处理标签，而标签在输出中出现的顺序并未指定。重要的是，我们无法可靠地创建一个始终匹配输出中标签顺序的单个字符串，我们肯定不希望开始检查所有可能的标签顺序。

我们想要的是首先能够识别输出中的特定行。这很重要，因为我们可能有很多具有相同日志级别或相同颜色的日志文件条目，但带有随机数的消息更为具体。一旦我们在文件中找到匹配随机数的单行，我们真正想要做的是检查该行以确保所有标签都存在。行内的顺序并不重要。

因此，我将`isTextInFile`函数更改为接受一个第三个参数，它将是一个字符串集合。这些字符串中的每一个都将是一个要检查的单个标签值。这实际上使测试更容易理解。我们可以保持消息不变，并使用它作为第一个参数来标识我们想要在日志文件中找到的行。假设我们找到了该行，然后我们逐个将格式化的标签以`key="value"`格式作为字符串集合传递，以验证它们是否都存在于已找到的同一行中。

注意，标签字符串以单个空格开始和结束。这确保了标签被正确地用空格分隔，并且我们也不会在标签值末尾有任何逗号。

我们应该修复检查日志级别存在性的其他测试，如下所示：

```cpp
TEST("Message can be tagged in log")
{
    std::string message = "simple tag ";
    message += Util::randomString();
    MereMemo::log(error) << message;
    std::string logLevelTag = " log_level=\"error\" ";
    bool result = Util::isTextInFile(message,          "application.log",
        {logLevelTag});
    CONFIRM_TRUE(result);
}
```

我们不再需要将消息追加到格式化的日志级别标签的末尾。我们只需将单个`logLevelTag`实例作为要检查的附加字符串集合中的单个值传递。现在，在`main`中设置了默认标签值，我们无法保证标签的顺序。因此，我们可能因为颜色标签恰好位于错误标签和消息之间而未能通过此测试。我们检查的只是消息是否出现在输出中，以及错误标签是否也存在于同一日志行条目中。

现在我们来增强`isTextInFile`函数，使其接受一个字符串向量作为第三个参数。如果调用者只想验证文件是否包含一些简单的文本，而不在相同的行上查找其他字符串，则该向量应有一个默认值为空集合。同时，我们添加一个第四个参数，它也将是一个字符串向量。第四个参数将检查确保其字符串不在行中。更新后的函数声明在`Util.h`中看起来如下：

```cpp
#include <string>
#include <string_view>
#include <vector>
struct Util
{
    static std::string randomString ();
    static bool isTextInFile (
        std::string_view text,
        std::string_view fileName,
        std::vector<std::string> const & wantedTags = {},
        std::vector<std::string> const & unwantedTags = {});
};
```

我们需要包含`vector`并确保为额外的参数提供默认空值。`Util.cpp`中的实现如下：

```cpp
bool Util::isTextInFile (
    std::string_view text,
    std::string_view fileName,
    std::vector<std::string> const & wantedTags,
    std::vector<std::string> const & unwantedTags)
{
    std::ifstream logfile(fileName.data());
    std::string line;
    while (getline(logfile, line))
    {
        if (line.find(text) != std::string::npos)
        {
            for (auto const & tag: wantedTags)
            {
                if (line.find(tag) == std::string::npos)
                {
                    return false;
                }
            }
            for (auto const & tag: unwantedTags)
            {
                if (line.find(tag) != std::string::npos)
                {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}
```

此更改在找到由`text`参数指定的行后添加了一个额外的`for`循环。对于提供的所有想要的标签，我们再次搜索该行以确保每个标签都存在。如果任何一个未找到，则函数返回`false`。假设它找到了所有标签，那么函数返回`true`，就像之前一样。

对于不想要的标签，几乎发生相同的事情，只是逻辑相反。如果我们找到一个不想要的标签，那么函数返回`false`。

我们现在需要添加`Color`标签类型的定义，然后添加`green`颜色实例。我们可以将这些添加到`LogTags.h`中，如下所示：

```cpp
inline MereMemo::LogLevel error("error");
inline MereMemo::LogLevel info("info");
inline MereMemo::LogLevel debug("debug");
class Color : public MereMemo::Tag
{
public:
    Color (std::string const & text)
    : Tag("color", text)
    { }
};
inline Color red("red");
inline Color green("green");
inline Color blue("blue");
```

构建项目显示我忘记实现了我们在`main`中开始使用的`addDefaultTag`函数。记得我曾经说过容易分心吗？我开始将函数添加到`Log.h`中，如下所示：

```cpp
inline void addDefaultTag (Tag const & tag)
{
    static std::map<std::string, Tag const *> tags;
    tags[tag.key()] = &tag;
}
```

这是一个很好的例子，说明了先编写使用情况如何有助于实现。我们需要做的是存储传递给 `addDefaultTag` 函数的标签，以便以后可以检索并添加到日志消息中。我们首先需要一个地方来存储标签，这样函数就可以声明一个静态映射。

最初，我想让映射复制标签，但这将需要更改 `Tag` 类，以便它可以直接使用而不是与派生类一起使用。我喜欢派生类如何帮助保持键的一致性，并且不想改变设计的那部分。

因此，我决定使用指针来存储标签集合。使用指针的问题是，对于 `addDefaultTag` 的调用者来说，任何传递给函数的标签的生命周期必须保持有效，直到该标签保留在默认标签集合中。

我们仍然可以创建副本并将副本存储在唯一指针中，但这需要调用者对 `addDefaultTag` 进行额外的工作，或者需要一个知道如何克隆标签的方法。我不想在调用 `addDefaultTag` 的 `main` 代码中添加额外的复杂性，并强迫该代码进行复制。我们已经在 `main` 中编写了代码，我们应该努力保持该代码不变，因为它使用了 TDD 原则，并提供了我们将最满意的解决方案。

为了避免生命周期意外，我们应该在 `Tag` 派生类中添加一个 `clone` 方法。并且因为我们正在 `addDefaultTag` 中使用映射并已确定需要唯一指针，所以我们需要在 `Log.h` 的顶部包含 `map` 和 `memory`，如下所示：

```cpp
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
```

现在，让我们实现正确的 `addDefaultTag` 函数，以便复制传入的标签而不是直接存储调用者的变量指针。这将释放调用者，使得传入的标签不再需要无限期地保持存活。将此代码添加到 `Log.h` 中，紧接在 `LogLevel` 类之后：

```cpp
inline std::map<std::string, std::unique_ptr<Tag>> & getDefaultTags ()
{
    static std::map<std::string, std::unique_ptr<Tag>> tags;
    return tags;
}
inline void addDefaultTag (Tag const & tag)
{
    auto & tags = getDefaultTags();
    tags[tag.key()] = tag.clone();
}
```

我们使用一个辅助函数来存储默认标签的集合。该集合是静态的，因此当第一次请求标签时，它被初始化为空映射。

我们需要在 `Tag` 类中添加一个纯虚 `clone` 方法，该方法将返回一个唯一指针。方法声明可以直接放在 `text` 方法之后，如下所示：

```cpp
    std::string text () const
    {
        return mText;
    }
    virtual std::unique_ptr<Tag> clone () const = 0;
protected:
```

现在，我们需要将 `clone` 方法的实现添加到 `LogLevel` 和 `Color` 类中。第一个看起来像这样：

```cpp
class LogLevel : public Tag
{
public:
    LogLevel (std::string const & text)
    : Tag("log_level", text)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new LogLevel(*this));
    }
};
```

`Color` 类的实现看起来几乎相同：

```cpp
class Color : public MereMemo::Tag
{
public:
    Color (std::string const & text)
    : Tag("color", text)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new Color(*this));
    }
};
```

尽管实现看起来几乎相同，但每个都创建了一个特定类型的实例，该实例作为 `Tag` 的唯一指针返回。这是我开始时希望避免的复杂性，但最好是向派生类添加复杂性，而不是向 `addDefaultTag` 的调用者施加额外的和意外的要求。

现在，我们已经准备好构建和运行测试应用程序。其中一个测试失败了，如下所示：

```cpp
Running 1 test suites
--------------- Suite: Single Tests
------- Test: Message can be tagged in log
Passed
------- Test: log needs no namespace when used with LogLevel
Passed
------- Test: Default tags set in main appear in log
Failed confirm on line 37
    Expected: true
------- Test: Simple message can be logged
Passed
------- Test: Complicated message can be logged
Passed
-----------------------------------
Tests passed: 4
Tests failed: 1
```

失败实际上是一件好事，它是 TDD（测试驱动开发）过程的一部分。我们像在 `main` 中使用它一样编写了代码，并编写了一个测试来验证默认标签是否出现在输出日志文件中。默认标签缺失，这是因为我们需要更改 `log` 函数，使其包含默认标签。

目前，`log` 函数只包括直接提供的标签——或者说，我应该说是直接提供的标签，因为我们还没有一种方法来记录多个标签。我们会达到那个地步。一次只做一件事。

我们的 `log` 函数目前有两个重载版本。一个接受单个 `Tag` 参数并将其转换为传递给另一个函数的字符串。一旦标签被转换为字符串，就很难检测到当前正在使用的标签，我们需要知道这一点，以免最终记录了具有相同键的默认标签和直接指定的标签。

例如，我们不希望日志消息同时包含 `info` 和 `debug` 日志级别，因为日志是用 `debug` 模式创建的，而 `info` 是默认模式。我们只想看到 `debug` 标签出现，因为它应该覆盖默认设置。

我们需要将标签作为 `Tag` 实例传递给执行输出的 `log` 函数，而不是字符串。然而，在调用 `log` 时，让我们允许调用者传递多个标签。我们应该让标签的数量无限吗？可能不是。三个看起来是个不错的数量。如果我们需要超过三个，我们会想出不同的解决方案或增加更多。

我考虑了使用模板编写接受可变数量标签的 `log` 函数的不同方法。虽然这可能可行，但复杂性很快变得难以处理。所以，相反，这里提供了三个重载的 `log` 函数，它们将参数转换为 `Tag` 指针的向量：

```cpp
inline auto log (Tag const & tag1)
{
    return log({&tag1});
}
inline auto log (Tag const & tag1,
    Tag const & tag2)
{
    return log({&tag1, &tag2});
}
inline auto log (Tag const & tag1,
    Tag const & tag2,
    Tag const & tag3)
{
    return log({&tag1, &tag2, &tag3});
}
```

这些函数替换了之前将标签转换为字符串的 `log` 函数。新函数创建了一个 `Tag` 指针的向量。我们最终可能需要调用 `clone` 来创建副本而不是使用指向调用者参数的指针，但就目前而言，这可行，我们不必担心我们之前与默认标签相关的生命周期问题。

我们需要在 `Log.h` 的顶部包含 `vector`，在实现实际执行日志记录的 `log` 函数时，我最终还需要 `algorithm`。新的包含部分如下所示：

```cpp
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
```

现在，让我们看看对执行日志记录的 `log` 函数的更改。它看起来是这样的：

```cpp
inline std::fstream log (std::vector<Tag const *> tags = {})
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const tmNow =          std::chrono::system_clock::to_time_t(now);
    auto const ms = duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::fstream logFile("application.log", std::ios::app);
    logFile << std::endl
        << std::put_time(std::gmtime(&tmNow),            "%Y-%m-%dT%H:%M:%S.")
        << std::setw(3) << std::setfill('0')         << std::to_string(ms.count());
    for (auto const & defaultTag: getDefaultTags())
    {
        if (std::find_if(tags.begin(), tags.end(),
            &defaultTag
            {
                return defaultTag.first == tag->key();
            }) == tags.end())
        {
            logFile << " " << defaultTag.second->text();
        }
    }
    for (auto const & tag: tags)
    {
        logFile << " " << tag->text();
    }
    logFile << " ";
    return logFile;
}
```

现在函数不再接受预格式化的标签字符串，而是接受一个 `Tag` 指针的向量，默认值为空集合。就这个函数而言，可以有无限多个标签。三个标签的限制仅仅是因为重载的 `log` 函数最多接受三个标签。

`tags` 向量的默认值允许调用者继续使用不带参数调用 `log`。

格式化时间戳、打开日志文件和打印时间戳的功能的第一部分保持不变，除了我们不再显示标签的预格式化字符串。

变更从第一个`for`循环开始，该循环检查每个默认标签。我们想要尝试在标签指针的向量中找到相同的标签键。如果我们找到相同的键，则跳过默认标签并尝试下一个。如果我们找不到相同的键，则显示默认标签。

为了进行搜索，我们使用`std::find_if`算法并提供一个知道如何比较键的 lambda 表达式。

在仅显示未被覆盖的默认标签之后，代码通过第二个`for`循环显示所有直接传递的标签。

构建并运行测试应用程序显示所有测试都通过，日志文件现在包含所有条目的默认标签，如下所示：

```cpp
2022-06-26T06:24:26.607 color="green" log_level="error" simple tag 4718
2022-06-26T06:24:26.609 color="green" log_level="error" no namespace
2022-06-26T06:24:26.609 color="green" log_level="info" default tag 8444
2022-06-26T06:24:26.609 color="green" log_level="info" simple 4281 with more text.
2022-06-26T06:24:26.610 color="green" log_level="info" complicated 8368 double=3.14 quoted="in quotes"
```

所有日志消息都将颜色标签设置为`"green"`，并且它们都包含`log_level`标签，该标签的值要么是默认值`"info"`，要么是覆盖值`"error"`。对于覆盖默认值的测试，让我们确保默认值不存在。我们可以利用`isTextInFile`函数中的不受欢迎的标签参数，如下所示：

```cpp
TEST("Message can be tagged in log")
{
    std::string message = "simple tag ";
    message += Util::randomString();
    MereMemo::log(error) << message;
    // Confirm that the error tag value exists and that the
    // default info tag value does not.
    std::string logLevelTag = " log_level=\"error\" ";
    std::string defaultLogLevelTag = " log_level=\"info\" ";
    bool result = Util::isTextInFile(message,          "application.log",
        {logLevelTag}, {defaultLogLevelTag});
    CONFIRM_TRUE(result);
}
```

是否应该将检查默认标签值是否不存在于日志文件中的额外检查添加到单独的测试中？单独测试的好处是它清楚地说明了正在测试的内容。缺点是测试将几乎与这个测试相同。这是一件需要思考的事情。在这种情况下，我认为在现有测试中添加额外检查和注释就足够了。

在继续之前，我们需要为我在多个标签中添加的功能添加一个测试。我真的很应该在增强代码以支持多个标签之前先为这个编写测试，但为了解释代码，一次直接解释多个标签的想法比返回并添加额外的解释要直接得多。

让我们快速在`LogTags.h`中添加一个名为`Size`的新类型`Tag`，并包含几个命名实例，如下所示：

```cpp
class Size : public MereMemo::Tag
{
public:
    Size (std::string const & text)
    : Tag("size", text)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new Size(*this));
    }
};
inline Size small("small");
inline Size medium("medium");
inline Size large("large");
```

现在，这里是一个针对多个标签的测试：

```cpp
TEST("Multiple tags can be used in log")
{
    std::string message = "multi tags ";
    message += Util::randomString();
    MereMemo::log(debug, red, large) << message;
    std::string logLevelTag = " log_level=\"debug\" ";
    std::string colorTag = " color=\"red\" ";
    std::string sizeTag = " size=\"large\" ";
    bool result = Util::isTextInFile(message,          "application.log",
        {logLevelTag, colorTag, sizeTag});
    CONFIRM_TRUE(result);
}
```

日志文件包含包含所有三个标签的条目，如下所示：

```cpp
2022-06-26T07:09:31.192 log_level="debug" color="red" size="large" multi tags 9863
```

我们有使用最多三个直接指定的标签和多个默认标签进行日志记录的能力。我们最终需要使用标签来做的不仅仅是显示日志文件中的信息。我们希望能够根据标签值过滤日志消息，以控制哪些日志消息能够到达日志文件，哪些被忽略。我们还没有准备好进行过滤。下一节将探讨基于标签值的过滤选项。

# 探索过滤选项

过滤日志消息让我们能够编写在代码中的重要位置调用日志信息的代码，但忽略其中的一些日志调用。我们为什么要费尽周折添加用于日志记录的代码，然后又不进行日志记录？

对于代码中的某些事件，例如检测到的错误，始终记录该事件是有意义的。其他地方可能同样重要，即使它们不是错误。通常，这些是代码中创建或删除某些内容的地方。我说的不是创建或删除局部变量的实例。我指的是一些重大的事情，比如创建新的客户账户、完成冒险游戏中的任务，或者删除旧数据文件以释放空间。所有这些都是应该始终记录的重要事件的例子。

其他事件可能有助于开发者了解程序在崩溃前做了什么。这些日志消息就像旅途中的路标。它们不如错误或重大事件那么重要，但它们可以帮助我们了解程序在做什么。这些通常也适合记录，因为没有它们，修复错误可能会很困难。当然——错误日志可能会清楚地显示发生了不好的事情，但没有路标消息，理解导致问题的原因可能会很困难。

有时候，当我们知道导致问题的总体思路时，我们还需要更多细节。这就是我们有时想要关闭日志记录的原因，因为像这样的日志消息有时可能会非常冗长，导致日志文件的大小增加。它们也可能使看到整体情况变得困难。你有没有尝试过眼睛紧紧盯着脚下的地面走路？你可以得到每一步的详细信息，但可能会发现你迷路了。抬头看大致方向会使你难以注意到可能让你绊倒的小石头。

在编写代码时，我们希望将这些类型的日志消息全部放入代码中，因为以后添加额外的日志消息可能会很困难，尤其是在程序在远程客户位置运行时。因此，我们希望代码尝试记录一切。然后，在运行时，我们希望精确控制日志文件中显示的信息量。过滤功能使我们能够通过忽略一些日志请求来控制我们看到多少日志。

我们将根据标签及其值来过滤日志消息，但我们遇到了一个问题。

假设我们想要忽略一个日志消息，除非它具有特定的标签值。我们当前的`log`函数工作方式是立即打开日志文件，开始流式传输时间戳，然后添加标签，最后允许调用者发送所需的其他内容。

确定是否允许日志消息完成输出的唯一方法是在它们最终确定后查看标签。换句话说，我们需要让所有内容都像将被记录一样发送，但实际上不进行任何操作。一旦我们有了完整的消息，我们就可以查看消息，看看它是否符合发送到输出文件的准则。

这意味着我们需要做两件事不同。首先，我们需要立即停止写入日志文件，并收集所有内容，以防我们最终需要写入。其次，我们需要知道何时一个日志消息完成。我们不能简单地返回一个打开的流给调用者，让调用者随意处理流。或者说，我们不能返回一个直接修改输出日志文件的流。让调用者直接与最终的输出日志文件工作，我们无法知道调用者何时完成，以便我们可以完成并决定忽略日志还是让它继续。

我知道三种确定潜在日志消息何时完成的方法。第一种是将所有内容放入一个单独的函数调用中。该函数可以接受可变数量的参数，所以我们不会受到固定数量的限制。但是，因为整个日志消息都捆绑在一个单独的函数调用中，所以我们会知道何时拥有所有内容。它可能看起来像这样：

```cpp
MereMemo::log(info, " count=", 5, " with text");
```

在这个例子中，我使用了一个标签实例、几个字符串字面量和一个整数。字符串字面量可以是字符串变量，或者可能是返回要记录的信息的函数调用。其中一个字符串字面量，连同数字一起，实际上形成了一个`key=value`标签。关键是`log`函数会确切地知道发送了多少信息以供记录，并且我们会知道所有值。我们可以轻松地测试日志消息，看看是否应该允许其继续，或者应该忽略它。

我们已经有了这种解决方案的初步形式，因为我们接受在`log`函数中最多三个标签实例。

确定日志何时完成的第二种方法是使用某种方法来终止我们现在的流。它可能看起来像这样：

```cpp
MereMemo::log(info) << "count=" << 5 << " with text" << MereMemo::endlog;
```

注意，我们不需要在`"count="`字符串字面量内部添加额外的空格，因为`log`函数会在所有标签之后为我们添加一个。

或者，我们甚至可以允许将标签发送到流中，如下所示：

```cpp
MereMemo::log() << info << " count=" << 5 << " with text" << MereMemo::endlog;
```

然后，我们又回到了在`count`字符串字面量之前需要添加前导空格的情况。这在需要调用者管理流元素之间空格的流中很常见。唯一不需要添加空格的地方是在`log`函数之后流出的第一个项目。

流式方法的主要思想是我们需要在末尾添加一些内容，让日志库知道所有信息都已准备好，可以与标准进行比较，以确定是否应该忽略日志。

我更喜欢流式方法。它让我感觉更开放——几乎更自然。而且由于操作符优先级和流操作符的链式操作，我们知道日志行将被评估的顺序。这可能不是非常重要，但它强化了我更喜欢流式方法的感觉。

使用这种第二种方法，调用者从`log`函数获取的流不能是一个直接与日志文件绑定的`std::fstream`实例。直接使用`fstream`将无法忽略日志消息，因为信息已经发送到文件中。也许我们可以返回一个与字符串绑定的流，并让终止的`endlog`元素发送构建的字符串到日志文件或忽略它。

如果忘记了终止的`endlog`元素会发生什么？终止的`endlog`元素需要评估日志并将其向前移动或忽略它。如果忘记了`endlog`，那么日志消息将不会完成。开发者可能直到需要查看日志文件时才会注意到问题，此时期望的日志消息总是被忽略。

第三种方法与第二种类似，但不需要一个可能被遗忘的终止元素。任何设计依赖于人记住做某事的时候，几乎肯定会有遗漏所需部分的情况。通过消除记住添加终止标记的需要，我们得到了一个更好的设计，它不再会因为简单的疏忽而被误用。

我们已经知道不能直接返回一个与日志文件绑定的流。第三种方法更进一步，返回一个自定义流。我们根本不使用标准流，因为我们需要在流析构函数中添加代码来完成日志记录并决定是让消息完成还是忽略它。

这种方法依赖于 C++定义的特定对象生命周期规则。我们需要确切知道析构函数何时运行，因为我们需要析构函数扮演终止的`endlog`元素的角色。其他使用垃圾回收来清理已删除对象的编程语言无法支持这种第三种解决方案，因为流将不会在未来的某个不确定时间被删除。C++非常明确地说明了对象实例何时被删除，我们可以依赖这个顺序。例如，我们可以这样调用`log`：

```cpp
MereMemo::log(info) << "count=" << 5 << " with text";
```

`log`返回的自定义流将在表达式结束的分号处被析构。程序员不会忘记任何事情，流将能够运行与显式的`endlog`元素会触发的相同代码。

也许我们可以结合所有三种方法的最佳之处。第一种函数调用方法不需要终止元素，因为它确切地知道正在传递多少个参数。第二种终止的`endlog`方法更加开放和自然，可以与字符串的标准流一起工作，而自定义流方法也是开放和自然的，并且避免了误用。

我最初想要创建一个能够根据整个消息过滤消息的日志库。虽然根据消息中的任何内容进行过滤似乎是最灵活和强大的解决方案，但它也是最难实现的。我们不想因为一个更容易编码而选择一个设计而不是另一个。我们应该选择一个基于最终使用的设计，这样我们会感到满意并且使用起来自然。有时，复杂的实现可能意味着最终使用也会很复杂。一个可能整体上不那么强大但更容易使用的解决方案会更好，只要我们不取消任何必需的功能。

我们应该能够去除一种过滤复杂性，而不会影响最终使用，那就是只查看通过`Tag`派生类形成的标签。我们应该能够取消根据手动编写的标签内容过滤日志消息的能力。

我们可以做出的另一个简化是只过滤传递给`log`函数的标签。这将结合第一个方法中`log`函数接受多个参数的方面，以及接受一系列直观信息的自定义流式传输方法。所以，看看以下流式传输示例：

```cpp
MereMemo::log(info) << green << " count=" << 5 << " with text";
```

这里总共有三个`key=value`标签。第一个是`info`标签，然后是`green`标签，接着是一个手动形成的带有计数文本和数字的标签。我们不需要尝试根据所有三个标签进行过滤，我们将用于过滤的唯一信息将是`info`标签，因为这是唯一直接传递给`log`函数的标签。我们还应该根据默认标签进行过滤，因为`log`函数也了解默认标签。这使得理解`log`函数的功能变得容易。`log`函数启动日志记录并确定其后的任何内容是否被接受或忽略。

如果我们想在过滤时考虑`green`标签，那么我们只需将其添加到`log`函数中，就像这样：

```cpp
MereMemo::log(info, green) << "count=" << 5 << " with text";
```

这种使用类型需要通过 TDD 进行深思熟虑。结果并不总是最强大的。相反，目标是满足用户的需求，并且易于理解和直观。

由于标签对这个设计变得越来越重要，我们应该增强它们以支持不仅仅是文本值。下一节将添加新的标签类型。

# 添加新的标签类型

由于我们开始用数字而不是文本来引用值，现在添加对不需要围绕值加引号的数字和布尔标签的支持将是一个好时机。

我们将在这里稍微提前一步，添加一些我们没有测试的代码。这仅仅是因为对数字和布尔标签的额外支持与我们已有的非常相似。这个更改在`Log.h`中的`Tag`类中。我们需要在现有的接受字符串的构造函数之后添加四个额外的构造函数，如下所示：

```cpp
protected:
    Tag (std::string const & key, std::string const & value)
    : mKey(key), mText(key + "=\"" + value + "\"")
    { }
    Tag (std::string const & key, int value)
    : mKey(key), mText(key + "=" + std::to_string(value))
    { }
    Tag (std::string const & key, long long value)
    : mKey(key), mText(key + "=" + std::to_string(value))
    { }
    Tag (std::string const & key, double value)
    : mKey(key), mText(key + "=" + std::to_string(value))
    { }
    Tag (std::string const & key, bool value)
    : mKey(key), mText(key + "=" + (value?"true":"false"))
    { }
```

每个构造函数都遵循`key="value"`或`key=value`语法来形成文本。为了测试新的构造函数，我们需要一些新的派生标签类。所有这些类都可以放在`LogTags.h`中。两个整型类看起来是这样的：

```cpp
class Count : public MereMemo::Tag
{
public:
    Count (int value)
    : Tag("count", value)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new Count(*this));
    }
};
class Identity : public MereMemo::Tag
{
public:
    Identity (long long value)
    : Tag("id", value)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new Identity(*this));
    }
};
```

我们不会提供这些标签的命名实例。早期的`Color`和`Size`标签类型都有合理且常见的选项，但即使如此，如果需要记录奇怪的或不同寻常的颜色或尺寸，它们也可以直接使用。新的标签没有这样的常见值。

继续说，双精度标签看起来是这样的：

```cpp
class Scale : public MereMemo::Tag
{
public:
    Scale (double value)
    : Tag("scale", value)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new Scale(*this));
    }
};
```

再次强调，它没有明显的默认值。也许我们可以为 1.0 或其他特定值提供一个命名值，但这些似乎最好由应用程序的领域定义。我们只是在测试一个日志库，并且将没有命名实例地使用这个标签。

布尔标签看起来是这样的：

```cpp
class CacheHit : public MereMemo::Tag
{
public:
    CacheHit (bool value)
    : Tag("cache_hit", value)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new CacheHit(*this));
    }
};
inline CacheHit cacheHit(true);
inline CacheHit cacheMiss(false);
```

对于这个，我们有明显的命名值`true`和`false`可以提供。

所有新的标签类都应该给你一个它们可以用于什么目的的思路。其中许多非常适合大型金融微服务，例如，值可能需要很长时间才能计算出来，并且需要缓存。在确定计算的流程时，记录结果是由于缓存命中还是未命中非常有价值。

我们希望能够将新的标签之一传递给`log`函数返回的流，如下所示：

```cpp
MereMemo::log(info) << Count(1) << " message";
```

要做到这一点，我们需要添加一个知道如何处理`Tag`类的流重载。将此函数添加到`Log.h`中，紧接在`to_string`函数之后：

```cpp
inline std::fstream & operator << (std::fstream && stream, Tag const & tag)
{
    stream << to_string(tag);
    return stream;
}
```

该函数使用对流的右值引用，因为我们正在使用`log`函数返回的临时流。

现在，我们可以创建一个测试，该测试将记录并确认每种新的类型。你可以为每种类型制作单独的测试，或者将它们全部放入一个测试中，如下所示：

```cpp
TEST("Tags can be streamed to log")
{
    std::string messageBase = " 1 type ";
    std::string message = messageBase + Util::randomString();
    MereMemo::log(info) << Count(1) << message;
    std::string countTag = " count=1 ";
    bool result = Util::isTextInFile(message,          "application.log", {countTag});
    CONFIRM_TRUE(result);
    messageBase = " 2 type ";
    message = messageBase + Util::randomString();
    MereMemo::log(info) << Identity(123456789012345)             << message;
    std::string idTag = " id=123456789012345 ";
    result = Util::isTextInFile(message, "application.log",
        {idTag});
    CONFIRM_TRUE(result);
    messageBase = " 3 type ";
    message = messageBase + Util::randomString();
    MereMemo::log(info) << Scale(1.5) << message;
    std::string scaleTag = " scale=1.500000 ";
    result = Util::isTextInFile(message, "application.log",
        {scaleTag});
    CONFIRM_TRUE(result);
    messageBase = " 4 type ";
    message = messageBase + Util::randomString();
    MereMemo::log(info) << cacheMiss << message;
    std::string cacheTag = " cache_hit=false ";
    result = Util::isTextInFile(message, "application.log",
        {cacheTag});
    CONFIRM_TRUE(result);
}
```

我之所以在添加代码以启用测试之前不那么担心创建这个测试，是因为我们在开始之前已经思考了期望的使用方式。

对于双精度值的标签可能需要稍后进行更多工作以控制精度。你可以看到它使用了默认的六位小数精度。新测试的日志条目看起来是这样的：

```cpp
2022-06-27T02:06:43.569 color="green" log_level="info" count=1 1 type 2807
2022-06-27T02:06:43.569 color="green" log_level="info" id=123456789012345 2 type 7727
2022-06-27T02:06:43.570 color="green" log_level="info" scale=1.500000 3 type 5495
2022-06-27T02:06:43.570 color="green" log_level="info" cache_hit=false 4 type 3938
```

注意到为每个`log`调用准备的每条消息是如何通过数字`1`到`4`来确保唯一的。这确保了在极少数情况下，如果生成了重复的随机数，四个日志消息中不会有相同的文本。

我们现在可以记录默认标签、直接传递给`log`函数的标签，以及像任何其他信息一样流出的标签。在我们实现实际过滤之前，下一节将进行一些增强，通过减少每个标签类需要编写的代码量来进一步改进标签类。

# 使用 TDD 重构标签设计

在测试中，我们有一个基类 `Tag` 和几个派生标签类。尽管日志库将只定义日志级别标签，但它仍然应该使开发者能够轻松创建新的派生标签类。目前，创建一个新的派生标签类主要是需要重复多次的样板代码。我们应该能够通过使用模板来提高这种体验。

下面是一个现有的派生标签类的样子：

```cpp
class LogLevel : public Tag
{
public:
    LogLevel (std::string const & text)
    : Tag("log_level", text)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new LogLevel(*this));
    }
};
```

从 `LogLevel` 派生的标签类是日志库将提供的唯一此类类。它定义了日志级别标签，但实际上并没有定义任何特定的日志级别值。更好的说法是，这个类定义了日志级别应该是什么。

我们可以将 `LogLevel` 类与测试中其他派生标签类之一进行比较。让我们选择 `CacheHit` 类，它看起来是这样的：

```cpp
class CacheHit : public MereMemo::Tag
{
public:
    CacheHit (bool value)
    : Tag("cache_hit", value)
    { }
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new CacheHit(*this));
    }
};
```

我们可以对这些类进行哪些改进？它们几乎相同，只有一些可以移动到模板类中的不同之处。这两个类有什么不同？

+   显然是名称。`LogLevel` 与 `CacheHit`。

+   父类命名空间。`LogLevel` 已经在 `MereMemo` 命名空间中。

+   关键字符串。`LogLevel` 使用 `"log_level"`，而 `CacheHit` 使用 `"cache_hit"`。

+   值的类型。`LogLevel` 使用 `std::string` 值，而 `CacheHit` 使用 `bool` 值。

这些就是所有的区别。不应该需要开发者每次需要新的标签类时都重新创建所有这些。而且，我们需要向标签类添加更多代码以支持过滤，所以现在是简化设计的好时机。

我们应该能够在不影响任何现有测试的情况下进行即将到来的过滤更改，但这将需要现在进行设计更改。我们正在重构设计，测试将有助于确保新设计继续像当前设计一样表现。从知道一切仍然正常工作所获得的信心是使用 TDD 的一大好处。

`Tag` 类代表了一个所有标签都支持的接口。我们将保持其原样并保持简单。我们不会更改 `Tag` 类，而是引入一个新的模板类，它可以包含 `clone` 方法实现以及任何即将到来的过滤更改。

将 `Log.h` 中的 `LogLevel` 类更改为使用新的 `TagType` 模板类，该类可以使用不同类型的值，如下所示：

```cpp
template <typename T, typename ValueT>
class TagType : public Tag
{
public:
    std::unique_ptr<Tag> clone () const override
    {
        return std::unique_ptr<Tag>(
            new T(*static_cast<T const *>(this)));
    }
    ValueT value () const
    {
        return mValue;
    }
protected:
    TagType (ValueT const & value)
    : Tag(T::key, value), mValue(value)
    { }
    ValueT mValue;
};
class LogLevel : public TagType<LogLevel, std::string>
{
public:
    static constexpr char key[] = "log_level";
    LogLevel (std::string const & value)
    : TagType(value)
    { }
};
```

我们仍然有一个名为 `LogLevel` 的类，可以像以前一样使用。现在它指定了值的类型，即 `std::string`，在 `TagType` 模板的参数中，而 `key` 字符串现在是一个由每个派生标签类定义的常量字符数组。`LogLevel` 类更简单，因为它不再需要处理克隆。

新的`TagType`模板类做了大部分的艰苦工作。目前，这项工作只是克隆，但我们需要添加更多功能来实现过滤。我们应该能够将这些即将到来的功能放入`TagType`类中，并保持派生标签类不变。

这种设计的工作方式基于某种称为`LogLevel`继承自`TagType`，而`TagType`将`LogLevel`作为其模板参数之一。这使得`TagType`能够在`clone`方法内部回指`LogLevel`以构造一个新的`LogLevel`实例。如果没有 CRTP，那么`TagType`将无法创建新的`LogLevel`实例，因为它不知道要创建什么类型。

并且`TagType`需要再次回指`LogLevel`以获取键名。`TagType`通过再次引用 CRTP 在`T`参数中给出的类型来实现这一点。

`clone`方法稍微复杂一些，因为当我们处于`clone`方法内部时，我们处于`TagType`类中，这意味着`this`指针需要被转换为派生类型。

我们现在可以简化`LogTags.h`中的其他派生标签类型。`Color`和`Size`类型都使用`std::string`作为值类型，就像`LogLevel`一样，它们看起来是这样的：

```cpp
class Color : public MereMemo::TagType<Color, std::string>
{
public:
    static constexpr char key[] = "color";
    Color (std::string const & value)
    : TagType(value)
    { }
};
class Size : public MereMemo::TagType<Size, std::string>
{
public:
    static constexpr char key[] = "size";
    Size (std::string const & value)
    : TagType(value)
    { }
};
```

`Count`和`Identity`类型都使用不同长度的整数值类型，看起来是这样的：

```cpp
class Count : public MereMemo::TagType<Count, int>
{
public:
    static constexpr char key[] = "count";
    Count (int value)
    : TagType(value)
    { }
};
class Identity : public MereMemo::TagType<Identity, long long>
{
public:
    static constexpr char key[] = "id";
    Identity (long long value)
    : TagType(value)
    { }
};
```

`Scale`类型使用`double`值类型，看起来是这样的：

```cpp
class Scale : public MereMemo::TagType<Scale, double>
{
public:
    static constexpr char key[] = "scale";
    Scale (double value)
    : TagType(value)
    { }
};
```

而`CacheHit`类型使用`bool`值类型，看起来是这样的：

```cpp
class CacheHit : public MereMemo::TagType<CacheHit, bool>
{
public:
    static constexpr char key[] = "cache_hit";
    CacheHit (bool value)
    : TagType(value)
    { }
};
```

每个派生标签类型都比以前简单得多，可以专注于每个类型的独特之处：类名、键名和值的类型。

下一节将创建基于逻辑标准的过滤测试，这将允许我们指定应该记录什么，我们还将使用简化的标签类以及`clone`方法。

# 设计过滤日志消息的测试

过滤日志消息将是日志库最重要的功能之一。这就是为什么本章投入了如此多的努力来探索想法并增强设计。大多数日志库都提供了一些过滤支持，但通常仅限于日志级别。日志级别通常也是有序的，当你设置一个日志级别时，你会得到所有等于或高于过滤级别的日志。

这对我来说总是显得很随意。日志级别是上升还是下降？将过滤级别设置为`info`是否意味着你也会得到`debug`，或者只是`info`和`error`日志？

这忽略了更大的问题——信息过载。一旦你弄清楚如何获取调试级别的日志，它们都会被记录下来，日志很快就会填满。我甚至见过日志填得如此快，以至于我感兴趣的日志消息已经被压缩并即将被删除以节省空间，在我能够退出应用程序查看发生了什么之前。

我们的目标客户是日志库的微服务开发者。这意味着正在开发的应用程序可能很大且分布广泛。在单个服务中甚至是在所有地方开启调试日志会导致很多问题。

我们正在构建的日志库将解决这些问题，但我们需要从简单开始。在`Tags.cpp`中的这个测试就是一个好的开始：

```cpp
TEST("Tags can be used to filter messages")
{
    int id = MereMemo::createFilterClause();
    MereMemo::addFilterLiteral(id, error);
    std::string message = "filter ";
    message += Util::randomString();
    MereMemo::log(info) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
    MereMemo::clearFilterClause(id);
    MereMemo::log(info) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_TRUE(result);
}
```

这个测试的想法是首先设置一个过滤器，该过滤器将导致日志消息被忽略。我们确认该消息没有出现在日志文件中。然后，测试清除过滤器，并再次尝试记录相同的消息。这次，它应该出现在日志文件中。

通常，过滤器匹配应允许日志继续，而没有任何匹配应导致消息被忽略。但是，当没有任何过滤器设置时，我们应该让所有内容通过。不设置任何过滤器让所有内容通过，让用户可以选择是否进行过滤。如果正在使用过滤，那么它将控制日志输出，但当没有过滤器时，不让任何内容通过就显得很奇怪。当测试集设置了一个与日志消息不匹配的过滤器时，由于已经启用了过滤，该消息就不会出现在日志文件中。当清除过滤器时，我们假设没有设置其他过滤器，所有日志消息将再次被允许继续。

我们将根据**析取范式**（**DNF**）中的公式来过滤日志。DNF 指定了一个或多个通过 OR 运算组合在一起的子句。每个子句包含通过 AND 运算组合在一起的文字。这里的“文字”不是 C++意义上的文字。在这里，“文字”是一个数学术语。子句中的每个文字可以是直接 AND 运算，也可以先进行 NOT 运算。所有这些都是布尔逻辑，并且能够表示从简单到复杂的任何逻辑条件。解释 DNF 的所有细节不是本书的目的，因此我不会解释 DNF 背后的所有数学。只需知道 DNF 足够强大，可以表示我们所能想到的任何过滤器。

这是一个需要强大解决方案的案例。即便如此，我们仍将专注于最终用途，并尽可能使解决方案易于使用。

测试调用一个`createFilterClause`函数，该函数返回创建的子句的标识符。然后，测试调用`addFilterLiteral`向刚刚创建的子句添加一个`error`标签。测试试图完成的是，只有当`error`标签存在时，才完成日志。如果这个标签不存在，那么日志应该被忽略。并且记住，为了使标签被考虑，它必须存在于默认标签中或直接提供给`log`函数。

然后，测试调用另一个函数`clearFilterClause`，该函数旨在清除刚刚创建的过滤器子句，并再次允许所有内容被记录。

通常，微服务开发者不会运行一个完全空的过滤应用，因为这会让所有日志消息都通过。在任何时候都可能存在一些过滤。只要至少有一个过滤子句是激活的，过滤就会只允许与其中一个子句匹配的消息继续。通过允许多个子句，我们实际上是在让额外的日志消息通过，因为每个额外的子句都有机会匹配更多的日志消息。我们将能够通过一个强大的布尔逻辑系统来调整记录的内容。

一个大型项目可以添加标识不同组件的标签。调试日志可以只为某些组件或其他匹配标准打开。额外的逻辑在调试会话期间增加了更多灵活性，可以在不影响其他区域并保持正常日志级别的同时，增加对有趣区域的日志记录。

如果一个标签存在于默认标签中，但在调用`log`时被直接覆盖，会发生什么？我们应该忽略默认标签，而选择显式的标签吗？我认为是这样，这将是一个很好的测试案例。边缘情况如此类实有助于定义项目并提高使用 TDD 获得的好处。现在让我们添加这个测试，以免忘记。它看起来是这样的：

```cpp
TEST("Overridden default tag not used to filter messages")
{
    int id = MereMemo::createFilterClause();
    MereMemo::addFilterLiteral(id, info);
    std::string message = "override default ";
    message += Util::randomString();
    MereMemo::log(debug) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
    MereMemo::clearFilterClause(id);
}
```

这个测试依赖于`info`标签已经在默认标签中设置。我们可能需要添加测试默认标签的能力，以便如果`info`在默认标签中找不到，测试会失败，并且我们需要确保在测试结束时清除过滤子句，以免影响其他测试。之前的测试也清除了子句，但在测试的特定点上。即便如此，之前的测试应该有一个更强的保证，即测试不会以过滤子句仍然设置的状态结束。我们应该利用测试拆解来确保在任何创建过滤子句的测试结束时始终清除过滤子句。

在继续添加拆解步骤之前，我刚开始解释的测试想法是这样的。在设置一个只允许带有`info`标签的日志的子句后，日志消息应该被允许继续，因为它将通过默认的标签集获得`info`标签。但相反，日志覆盖了`info`标签，使用了`debug`标签。最终结果是，日志消息不应该出现在输出日志文件中。

为了确保即使在测试失败并抛出异常之前测试未到达末尾，我们也能始终清除过滤子句，我们需要在`Tags.cpp`中定义一个设置和拆解类，如下所示：

```cpp
class TempFilterClause
{
public:
    void setup ()
    {
        mId = MereMemo::createFilterClause();
    }
    void teardown ()
    {
        MereMemo::clearFilterClause(mId);
    }
    int id () const
    {
        return mId;
    }
private:
    int mId;
};
```

如果你想了解更多关于设置和拆解类的信息，请参阅*第七章*，*测试设置*和*拆解*。

在适当的时候，测试自己清除过滤器是可以的。添加一个`SetupAndTeardown`实例将确保即使它已经被调用，也会调用`clearFilterClause`函数。本节的第一项测试看起来像这样：

```cpp
TEST("Tags can be used to filter messages")
{
    int id = MereMemo::createFilterClause();
    MereMemo::addFilterLiteral(id, error);
    std::string message = "filter ";
    message += Util::randomString();
    MereMemo::log(info) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
    MereMemo::clearFilterClause(id);
    MereMemo::log(info) << message;
    result = Util::isTextInFile(message, "application.log");
    CONFIRM_TRUE(result);
}
```

测试现在从设置和清理实例中获取条款 ID。ID 用于添加过滤字面量和在正确的时间清除过滤条款。过滤条款将在测试结束时再次清除，但没有效果。

本节中的第二个测试不再需要显式清除过滤器本身，只需要添加`SetupAndTeardown`实例，如下所示：

```cpp
TEST("Overridden default tag not used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(), info);
    std::string message = "override default ";
    message += Util::randomString();
    MereMemo::log(debug) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
}
```

这个测试在结束时调用`clearFilterClause`以将过滤器放回未过滤的状态。测试不再需要直接调用`clearFilterClause`，因为依赖于`SetupAndTeardown`析构函数更可靠。

我们有两个过滤测试调用尚未存在的函数。让我们在`Log.h`文件中`addDefaultTag`函数之后添加以下函数占位符：

```cpp
inline int createFilterClause ()
{
    return 1;
}
inline void addFilterLiteral (int filterId,
    Tag const & tag,
    bool normal = true)
{
}
inline void clearFilterClause (int filterId)
{
}
```

目前`createFilterClause`函数只是返回`1`。它最终需要为每个创建的条款返回不同的标识符。

`addFilterLiteral`函数将给定的标签添加到指定的条款中。`normal`参数将允许我们通过传递`false`来添加 NOT 或反转的字面量。小心处理此类标志的含义。当我第一次写这个时，标志被命名为`invert`，默认值为`false`。我没有注意到这个问题，直到为反转过滤器编写测试，并且传递`true`以获取反转字面量看起来很奇怪。测试突出了反向使用，而初始函数声明让它悄悄溜走，没有被发现。

`clearFilterClause`函数目前没有任何作用。我们稍后需要有一些可以操作的条款集合。

占位符过滤函数让我们构建和运行测试应用程序。我们得到两个测试失败，如下所示：

```cpp
Running 1 test suites
--------------- Suite: Single Tests
------- Test: Message can be tagged in log
Passed
------- Test: log needs no namespace when used with LogLevel
Passed
------- Test: Default tags set in main appear in log
Passed
------- Test: Multiple tags can be used in log
Passed
------- Test: Tags can be streamed to log
Passed
------- Test: Tags can be used to filter messages
Failed confirm on line 123
    Expected: false
------- Test: Overridden default tag not used to filter messages
Failed confirm on line 143
    Expected: false
------- Test: Simple message can be logged
Passed
------- Test: Complicated message can be logged
Passed
-----------------------------------
Tests passed: 7
Tests failed: 2
```

预期结果是使用 TDD（测试驱动开发）。我们只做了必要的最小工作来构建代码，以便我们可以看到失败。我们可以在占位符函数中添加更多的实现。

我提到我们需要一个条款的集合。在`Log.h`文件中，在占位符过滤函数之前添加以下函数：

```cpp
struct FilterClause
{
    std::vector<std::unique_ptr<Tag>> normalLiterals;
    std::vector<std::unique_ptr<Tag>> invertedLiterals;
};
inline std::map<int, FilterClause> & getFilterClauses ()
{
    static std::map<int, FilterClause> clauses;
    return clauses;
}
```

模式与我们为默认标签所做的是相似的。有一个名为`getFilterClauses`的函数，它返回一个静态`FilterClause`对象映射的引用，而`FilterClause`结构体被定义为包含正常和反转字面量的几个向量。字面量是指从克隆中获得的标签的指针。

`createFilterClause`函数可以实施以使用条款集合，如下所示：

```cpp
inline int createFilterClause ()
{
    static int currentId = 0;
    ++currentId;
    auto & clauses = getFilterClauses();
    clauses[currentId] = FilterClause();
    return currentId;
}
```

这个函数通过一个静态变量跟踪当前 id，每次函数被调用时都会递增。需要完成的唯一其他任务是创建一个空的过滤条款记录。id 被返回给调用者，以便稍后可以修改或清除过滤条款。

`addfilterLiteral`函数可以像这样实现：

```cpp
inline void addFilterLiteral (int filterId,
    Tag const & tag,
    bool normal = true)
{
    auto & clauses = getFilterClauses();
    if (clauses.contains(filterId))
    {
        if (normal)
        {
            clauses[filterId].normalLiterals.push_back(
                tag.clone());
        }
        else
        {
            clauses[filterId].invertedLiterals.push_back(
                tag.clone());
        }
    }
}
```

这个函数确保在将克隆指针推入正常或反转向量之前，`clauses`集合包含给定过滤 id 的条目。

而`clearFilterClause`函数是最简单的，因为它只需要获取集合并删除具有给定 id 的任何过滤条款，如下所示：

```cpp
inline void clearFilterClause (int filterId)
{
    auto & clauses = getFilterClauses();
    clauses.erase(filterId);
}
```

我们仍然需要在记录日志时检查过滤条款，这将在下一节中解释。在遵循 TDD 时，当代码构建并且运行时测试失败时，让测试通过是很好的。让我们在下一节中让测试通过！

# 控制要记录的内容

在本章早期，当我们探索过滤选项时，我提到过我们需要一个自定义流类，而不是从`log`函数返回`std::fstream`。我们需要这样做，以便我们不会立即将信息发送到日志文件。我们需要避免直接将日志消息发送到日志文件，因为可能存在过滤规则，这些规则可能导致日志消息被忽略。

我们还决定，我们将完全基于默认标签和直接发送到`log`函数的任何标签来决定是否记录。我们可以让`log`函数做出决定，如果日志消息应该继续，则返回`std::fstream`，如果日志消息应该被忽略，则返回一个假流，但可能更好的做法是始终返回相同的类型。这似乎是最简单、最直接的方法。在流类型之间切换似乎是一个更复杂的解决方案，仍然需要自定义流类型。

使用自定义流类型也将使我们能够解决一个令人烦恼的问题，即我们不得不在每个日志消息之前而不是之后放置换行符。这导致了日志文件的第一行是空的，最后一行突然结束。我们选择了在每次日志消息之前放置换行符的临时解决方案，因为我们当时没有东西可以让我们知道所有信息都已经流过。

好吧，自定义流类将使我们能够解决这个令人烦恼的换行符问题，并给我们一个避免直接将日志消息写入日志文件的方法。让我们从新的流类开始。在`Log.h`中创建这个类，在`log`函数之前，如下所示：

```cpp
class LogStream : public std::fstream
{
public:
    LogStream (std::string const & filename,
        std::ios_base::openmode mode = ios_base::app)
    : std::fstream(filename, mode)
    { }
    LogStream (LogStream const & other) = delete;
    LogStream (LogStream && other)
    : std::fstream(std::move(other))
    { }
    ~LogStream ()
    {
        *this << std::endl;
    }

    LogStream & operator = (LogStream const & rhs) = delete;
    LogStream & operator = (LogStream && rhs) = delete;
};
```

我们将一次解决一个问题。因此，我们将继续重构这个类，直到它完成我们需要的所有事情。现在，它只是从`std::fstream`继承，所以它不会解决直接写入日志文件的问题。构造函数仍然打开日志文件，所有的流能力都是从`fstream`继承的。

这个类所解决的问题就是换行问题。它是通过在类的析构函数中向流发送`std::endl`来解决的。基于提供的名称打开文件和添加换行的析构函数是这个类解决问题的关键部分。类的其余部分是为了使代码能够编译和正常工作。

由于我们添加了一个析构函数，这引发了一系列其他要求。我们现在需要提供一个复制构造函数。实际上，我们需要的是*移动复制构造函数*，因为流在复制时往往会表现得奇怪。复制流不是一个简单的任务，但将流移动到另一个流中要简单得多，并且已经完成了我们需要的所有事情。我们不需要复制流，但我们确实需要从`log`函数返回流，这意味着流要么需要被复制，要么需要被移动。因此，我们显式删除了复制构造函数并实现了移动复制构造函数。

我们还删除了赋值运算符和移动赋值运算符，因为我们不需要对流进行赋值。

我们可以通过修改`log`函数来使用新的`LogStream`类，使其看起来像这样：

```cpp
inline LogStream log (std::vector<Tag const *> tags = {})
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const tmNow =          std::chrono::system_clock::to_time_t(now);
    auto const ms = duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    LogStream ls("application.log");
    ls << std::put_time(std::gmtime(&tmNow),        "%Y-%m-%dT%H:%M:%S.")
        << std::setw(3) << std::setfill('0')         << std::to_string(ms.count());
    for (auto const & defaultTag: getDefaultTags())
    {
        if (std::find_if(tags.begin(), tags.end(),
            &defaultTag
            {
                return defaultTag.first == tag->key();
            }) == tags.end())
        {
            ls << " " << defaultTag.second->text();
        }
    }
    for (auto const & tag: tags)
    {
        ls << " " << tag->text();
    }
    ls << " ";
    return ls;
}
```

现在的`log`函数返回一个`LogStream`实例而不是`std::fstream`。在函数内部，它创建一个`LogStream`实例，就像它是一个`fstream`实例一样。唯一改变的是类型。现在文件打开模式默认为`append`，因此我们不需要指定如何打开文件。流的名字改为`ls`，因为这已经不再是一个日志文件了。

然后，在发送初始时间戳时，我们不再需要发送初始的`std::endl`实例，可以直接开始发送时间戳。

在这些更改之后，测试应用程序运行时唯一的不同之处在于日志文件将不再有空的第一个行，并且所有行都将以换行符结束。

那个小问题已经解决了。那么，直接写入日志文件的大问题怎么办？我们仍然希望写入标准流，因为实现我们自己的流类会增加我们目前并不真正需要的很多复杂性。因此，我们不会从`std::fstream`继承`LogStream`类，而是从`std::stringstream`继承。

我们需要包含`sstream`以获取`stringstream`的定义，我们也可以现在就包含`ostream`。我们需要`ostream`来更改`Log.h`中的流辅助函数，该函数目前使用`std::fstream`，我们将将其改为如下所示：

```cpp
inline std::ostream & operator << (std::ostream && stream, Tag const & tag)
{
    stream << to_string(tag);
    return stream;
}
```

我们可能从一开始就应该实现这个辅助函数来使用`ostream`。这样，我们可以将标签流式传输到任何输出流。由于`fstream`和`stringstream`都基于`ostream`，我们可以使用这个辅助函数将流式传输到两者。

这里是更新后的`Log.h`的包含内容：

```cpp
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
```

从技术上讲，我们不需要包含 `ostream`，因为我们已经通过包含 `fstream` 和 `stringstream` 获得了它。但我喜欢包含我们直接使用的东西的头文件。在查看包含的头文件时，我发现我们包含了 `iostream`。我认为我最初包含 `iostream` 是为了获取 `std::endl` 的定义，但看起来 `endl` 实际上是在 `ostream` 中声明的。所以，根据我包含使用头文件的规则，我们应该从一开始就包含 `ostream` 而不是 `iostream`。

回到 `LogStream`，我们需要将这个类修改为从 `stringstream` 继承，如下所示：

```cpp
class LogStream : public std::stringstream
{
public:
    LogStream (std::string const & filename,
        std::ios_base::openmode mode = ios_base::app)
    : mProceed(true), mFile(filename, mode)
    { }
    LogStream (LogStream const & other) = delete;
    LogStream (LogStream && other)
    : std::stringstream(std::move(other)),
    mProceed(other.mProceed), mFile(std::move(other.mFile))
    { }
    ~LogStream ()
    {
        if (not mProceed)
        {
            return;
        }
        mFile << this->str();
        mFile << std::endl;
    }
    LogStream & operator = (LogStream const & rhs) = delete;
    LogStream & operator = (LogStream && rhs) = delete;
    void ignore ()
    {
        mProceed = false;
    }
private:
    bool mProceed;
    std::fstream mFile;
};
```

有一个新的数据成员叫做 `mProceed`，我们在构造函数中将它设置为 `true`。由于我们不再从 `std::fstream` 继承，我们现在需要一个文件流的数据成员。我们还需要初始化 `mFile` 成员。移动拷贝构造函数需要初始化数据成员，析构函数检查是否应该继续日志记录。如果应该继续日志记录，那么 `stringstream` 的字符串内容将被发送到文件流。

我们还没有实现过滤，但我们已经接近了。这个更改让我们达到了可以控制日志记录的点。除非在析构函数运行之前调用 `ignore`，否则日志记录将继续进行。这个简单的更改将使我们能够构建和测试，以确保我们没有破坏任何东西。

运行测试应用程序显示与过滤相关的相同两个测试失败。主要的事情是其他测试继续通过，这表明当我们直接将流直接写入文件流时，使用 `stringstream` 的更改仍然像以前一样工作。

在进行诸如切换流等关键更改时，确保没有东西被破坏是很重要的。这就是为什么我选择了一个硬编码的选择，总是进行日志记录。我们可以使用我们已有的 TDD 测试来验证在添加过滤之前，流更改是否正常工作。

让我们分两部分来看一下对 `log` 函数的下一个修改。在确定哪些默认标签被覆盖之后，我们需要收集完整的活动标签集合。我们不需要直接将标签发送到流中，而是可以先将其放入一个活动集合中，如下所示：

```cpp
inline LogStream log (std::vector<Tag const *> tags = {})
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const tmNow =          std::chrono::system_clock::to_time_t(now);
    auto const ms = duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    LogStream ls("application.log");
    ls << std::put_time(std::gmtime(&tmNow),        "%Y-%m-%dT%H:%M:%S.")
        << std::setw(3) << std::setfill('0')         << std::to_string(ms.count());
    std::map<std::string, Tag const *> activeTags;
    for (auto const & defaultTag: getDefaultTags())
    {
        activeTags[defaultTag.first] = defaultTag.second.get();
    }
    for (auto const & tag: tags)
    {
        activeTags[tag->key()] = tag;
    }
    for (auto const & activeEntry: activeTags)
    {
        ls << " " << activeEntry.second->text();
    }
    ls << " ";
    // Filtering will go here.
    return ls;
}
```

这样不仅得到了活动集合，而且看起来也更简单。我们让映射首先处理覆盖，将所有默认标签放入映射中，然后将所有提供的标签放入映射中。构建和运行测试应用程序显示，这个修改并没有破坏任何新的东西。因此，我们准备进行下一部分，即比较过滤子句与活动标签。

过滤需要更改 `log` 函数的最后部分，其中注释表明过滤将在这里进行，如下所示：

```cpp
    bool proceed = true;
    for (auto const & clause: getFilterClauses())
    {
        proceed = false;
        bool allLiteralsMatch = true;
        for (auto const & normal: clause.second.normalLiterals)
        {
            // We need to make sure that the tag is
            // present and with the correct value.
            if (not activeTags.contains(normal->key()))
            {
                allLiteralsMatch = false;
                break;
            }
            if (activeTags[normal->key()]->text() !=
                normal->text())
            {
                allLiteralsMatch = false;
                break;
            }
        }
        if (not allLiteralsMatch)
        {
            continue;
        }
        for (auto const & inverted:             clause.second.invertedLiterals)
        {
            // We need to make sure that the tag is either
            // not present or has a mismatched value.
            if (activeTags.contains(inverted->key()))
            {
                if (activeTags[inverted->key()]->text() !=
                    inverted->text())
                {
                    break;
                }
                allLiteralsMatch = false;
                break;
            }
        }
        if (allLiteralsMatch)
        {
            proceed = true;
            break;
        }
    }
    if (not proceed)
    {
        ls.ignore();
    }
    return ls;
```

逻辑有点复杂，这是一个我发现几乎完全实现逻辑比试图将更改分成多个部分更容易的案例。以下是代码做了什么。因为我们使用 DNF 逻辑，我们可以分别处理每个条款。我们开始时假设我们将继续记录日志，以防没有设置任何过滤器。如果有任何过滤器，那么对于每一个，我们开始时假设我们不会继续。但我们还设置了一个新的`bool`变量，它假设所有文字都将匹配，直到证明否则。我们将没有文字的条款视为我们应该继续记录日志的信号。

对于检查文字，我们有两种类型：正常和反转。对于正常文字，标签必须在活动标签中全部存在并且具有匹配的值。如果任何标签缺失或具有错误值，那么我们就没有匹配这个条款的所有文字。我们将继续，因为可能还有另一个条款会匹配。这就是我所说的分别处理每个条款的意思。

假设我们已经匹配了所有正常文字，我们仍然需要检查反转文字。在这里，逻辑被反转了，我们需要确保标签不存在或者它具有错误值。

一旦我们检查完所有条款或找到一个匹配所有文字的条款，代码将进行最后一次检查以确定日志是否应该继续。如果不应该，那么我们将调用`ignore`，这将阻止日志消息被发送到输出日志文件。

这种方法在调用`log`函数时根据默认标签和发送到`log`函数的标签决定是否继续。我们将让调用代码发送所需的所有信息到流中。如果未调用`ignore`，则信息才会完整地到达输出日志文件。

现在一切都可以构建和运行了，我们再次通过了所有测试，如下所示：

```cpp
Running 1 test suites
--------------- Suite: Single Tests
------- Test: Message can be tagged in log
Passed
------- Test: log needs no namespace when used with LogLevel
Passed
------- Test: Default tags set in main appear in log
Passed
------- Test: Multiple tags can be used in log
Passed
------- Test: Tags can be streamed to log
Passed
------- Test: Tags can be used to filter messages
Passed
------- Test: Overridden default tag not used to filter messages
Passed
------- Test: Simple message can be logged
Passed
------- Test: Complicated message can be logged
Passed
-----------------------------------
Tests passed: 9
Tests failed: 0
```

这表明过滤功能正在工作！至少，对于标签的相等性。测试一个标签是否存在并且具有匹配的值是一个好的开始，但我们的微服务开发者将需要比这更多的能力。也许我们只需要在计数标签的值大于 100 或涉及比较指定过滤器值的数值更大或更小的其他比较时记录日志。这就是我说我几乎完全实现了过滤逻辑的意思。我得到了逻辑以及所有循环和中断，用于标签的相等性。我们应该能够在下一节中相对比较中使用相同的基本代码结构。

在我们开始相对比较之前，还有一件事要补充，这是很重要的。每当添加代码，就像我添加 DNF 逻辑那样，没有测试来支持它，我们需要添加一个测试。否则，遗漏的测试可能会被推迟，直到完全忘记。

这个新测试以另一种方式提供了帮助。它捕捉到了`addFilterLiteral`函数初始定义中的一个问题。原始函数定义了一个名为`invert`的`bool`参数，其默认值为`false`。默认值意味着创建一个普通字面量时可以省略该参数并使用默认值。但为了创建一个倒置字面量，该函数要求传递`true`值。这在我看来似乎是反过来的。我意识到，传递`false`给这个参数以获取倒置字面量，而`true`应该创建一个普通字面量，这样会更有意义。因此，我回过头去修改了函数的定义和实现。测试捕捉到了一个最初未被注意到的函数使用问题。

这里是创建倒置过滤器的新测试：

```cpp
TEST("Inverted tag can be used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(), green, false);
    std::string message = "inverted ";
    message += Util::randomString();
    MereMemo::log(info) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
}
```

构建和运行显示新测试通过，并且我们已经确认可以过滤包含匹配标签的日志消息，当过滤器被倒置时。这个测试使用了默认的`green`标签，该标签被添加到日志消息中，并确保由于存在`green`标签，日志消息不会出现在输出日志文件中。

下一节将增强过滤功能，允许根据标签的相对值而不是仅基于精确匹配进行过滤。

# 加强相对匹配的过滤

TDD 鼓励在设计软件时进行增量更改和增强。编写一个测试，让某物工作，然后编写一个更详细的测试来增强设计。我们一直在遵循 TDD 方法来设计日志库，上一节就是一个很好的例子。我们在上一节中实现了过滤功能，但仅限于标签相等。

换句话说，我们现在可以根据标签是否存在来过滤日志消息，这些标签与过滤字面量标签匹配。我们比较标签以查看键和值是否匹配。这是一个很好的第一步，因为即使达到这一步也需要大量的工作。想象一下，如果我们试图做到极致，并支持例如，只有当计数标签的值大于 100 时才进行日志记录。

当使用 TDD（测试驱动开发）设计软件时，在采取下一步之前寻找显而易见的步骤并确认其工作情况非常有帮助。有些步骤可能比其他步骤大，但这没关系，只要你不直接跳到最终实现，因为那样只会导致更长的开发时间和更多的挫败感。确认设计的一些部分按预期工作并拥有确保这些部分继续工作的测试要好的多。这就像建造一座房子，有一个坚实的基础。在建造墙壁之前，确保基础确实牢固要好的多，你希望有测试来确保在添加屋顶时墙壁保持笔直。

我们已经实施了工作测试以确保基本过滤功能正常。我们正在测试正常和倒置的文本。我们通过比较标签的文本来检查匹配的标签，这对于所有值类型都适用。对于像计数大于 100 这样的相对过滤器，我们需要一个能够用数值检查而不是字符串匹配来比较值的解决方案。

我们可以先找出如何表示一个过滤器文本来检查大于或小于的数值。以下是一个可以放入`Tags.cpp`的测试，它基于计数大于 100 设置一个过滤器：

```cpp
TEST("Tag values can be used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(),
        Count(100, MereMemo::TagOperation::GreaterThan));
    std::string message = "values ";
    message += Util::randomString();
    MereMemo::log(Count(1)) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
    MereMemo::log() << Count(101) << message;
    result = Util::isTextInFile(message, "application.log");
    CONFIRM_FALSE(result);
    MereMemo::log(Count(101)) << message;
    result = Util::isTextInFile(message, "application.log");
    CONFIRM_TRUE(result);
}
```

这个测试有什么新内容？主要部分是`Count`标签的创建方式。我们之前在创建标签时只添加了一个值，如下所示：

```cpp
Count(100)
```

由于我们现在需要一种指定是否应该具有相对值的方法，我们需要一个地方来说明相对值的类型以及一个方法来传达要使用的相对值。我认为各种相对比较的枚举应该可以工作。我们可能不需要更高级的相对比较，如`"between"`，因为我们总是可以使用 DNF 来表示更复杂的比较。有关我们如何使用 DNF 的简要概述，请参阅本章的*设计测试以过滤日志消息*部分。

在标签级别，我们真正需要知道的是如何比较一个值与另一个值。因此，在构建标签时指定所需的比较类型是有意义的，如下所示：

```cpp
Count(100, MereMemo::TagOperation::GreaterThan)
```

将具有比较运算符（如`GreaterThan`）的标签视为完全不同的类型可能是有意义的，但我认为我们可以通过单一类型来解决这个问题。在这个解决方案中，任何标签都可以有比较运算符，但只有当标签将用于过滤器时，指定比较运算符才有意义。

如果在过滤器中使用不带比较运算符的常规标签会发生什么？那么，我们应该将其视为精确匹配，因为这是现有测试所期望的。

回到新的测试。它首先创建了一个过滤器，该过滤器只允许具有计数标签且其值大于 100 的消息被记录。它首先尝试记录一个计数为 1 的消息，并验证该消息不存在于日志文件中。

然后，测试创建了一个计数为 101 的计数器，但并没有在`log`函数调用中直接使用计数标签。这也应该不会出现在输出日志文件中，因为我们只想在调用`log`时过滤默认或直接指定的标签。

最后，测试使用计数标签 101 调用`log`，并验证该消息是否出现在日志文件中。

现在我们有了测试，我们将如何让它工作？让我们首先在`Log.h`中定义比较操作，在`TagType`类之前，如下所示：

```cpp
enum class TagOperation
{
    None,
    Equal,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual
};
```

我们将使用`None`操作来表示只想表达值的常规标签。`Equal`操作将像标签之间的现有相等检查一样起作用。真正的变化是支持小于、小于等于、大于和大于等于的比较。

我们需要比较一个标签与另一个标签，而不必担心标签代表什么。一个好的方法是在`Tag`类中声明一个纯虚拟方法，就像我们为克隆所做的那样。新方法被称为`match`，可以紧接在`clone`方法之后，如下所示：

```cpp
    virtual std::unique_ptr<Tag> clone () const = 0;
    virtual bool match (Tag const & other) const = 0;
```

这里事情变得有些困难。我原本想将所有内容都封装在`TagType`类中。想法是首先检查每个被比较的标签的键，并确保标签是相同的。如果它们有相同的键，那么检查值。如果它们没有相同的键，那么它们肯定不匹配。至少，这是一个不错的计划。当我尝试在一个可以比较字符串与字符串、数字与数字以及布尔值与布尔值的地方实现`match`方法时遇到了问题。例如，像`CacheHit`这样的标签有一个`bool`类型的值，唯一有意义的操作是`Equal`比较。基于字符串的标签需要与数字进行比较。如果我们真的想做得更细致，双精度浮点数应该与`int`类型进行比较不同。

每个派生标签类型都可以知道如何比较，但我不想改变派生类型并让它们各自实现`match`方法，尤其是在我们费了很大力气避免派生类型实现`clone`之后。我想出的最佳解决方案是创建一组额外的中间类，这些类从`TagType`派生。每个新类基于值的类型。由于我们只支持五种不同的标签值类型，这不是一个坏解决方案。主要好处是调用者将使用的派生标签类型仅略有影响。这里有一个新的`StringTagType`类，它从`TagType`继承，以便你可以看到我的意思。将这个新类放在`Log.h`中，紧接在`TagType`类之后：

```cpp
template <typename T>
class StringTagType : public TagType<T, std::string>
{
protected:
    StringTagType (std::string const & value,
        TagOperation operation)
    : TagType<T, std::string>(value, operation)
    { }
    bool compareTagTypes (std::string const & value,
        TagOperation operation,
        std::string const & criteria) const override
    {
        int result = value.compare(criteria);
        switch (operation)
        {
        case TagOperation::Equal:
            return result == 0;
        case TagOperation::LessThan:
            return result == -1;
        case TagOperation::LessThanOrEqual:
            return result == 0 || result == -1;
        case TagOperation::GreaterThan:
            return result == 1;
        case TagOperation::GreaterThanOrEqual:
            return result == 0 || result == 1;
        default:
            return false;
        }
    }
};
```

这个类完全是关于比较基于字符串的标签与其他基于字符串的标签。该类实现了一个新虚拟方法，我将在稍后解释，称为`compareTagTypes`。这个方法唯一需要担心的是如何根据操作比较两个字符串。其中一个字符串被称为`value`，另一个被称为`criteria`。重要的是不要混淆`value`和`criteria`字符串，因为例如，虽然`"ABC"`大于`"AAA"`，但反过来并不成立。该方法使用`std::string`类中的`compare`方法来进行比较。

你可以看到`StringTagType`类从`TagType`继承，并传递派生类型`T`，同时为值类型硬编码`std::string`。关于构造函数的一个有趣之处在于，在构造函数初始化列表中构造`TagType`时需要重复模板参数。通常情况下，这不应该需要，但也许有一些我尚未意识到的神秘规则仅适用于此处，即编译器不会查看父类列表中的`TagType`参数来找出模板参数。

在继续到`TagType`的更改之前，让我们看看派生标签类如`LogLevel`将如何使用新的`StringTagType`中间类。将`LogLevel`类修改如下：

```cpp
class LogLevel : public StringTagType<LogLevel>
{
public:
    static constexpr char key[] = "log_level";
    LogLevel (std::string const & value,
        TagOperation operation = TagOperation::None)
    : StringTagType(value, operation)
    { }
};
```

对于`LogLevel`所需的唯一更改是将父类从`TagType`更改为更具体的`StringTagType`。我们不再需要担心指定`std::string`作为模板参数，因为该信息已内置到`StringTagType`类中。我原本想保持派生标签类完全不变，但这种轻微的修改并不糟糕，因为不需要编写任何比较代码。

在`TagType`类中还有更多工作要做。在`TagType`类末尾的保护部分，进行以下更改：

```cpp
protected:
    TagType (ValueT const & value,
        TagOperation operation)
    : Tag(T::key, value), mValue(value), mOperation(operation)
    { }
    virtual bool compareTagTypes (ValueT const & value,
        TagOperation operation,
        ValueT const & criteria) const
    {
        return false;
    }
    ValueT mValue;
    TagOperation mOperation;
};
```

受保护的构造函数需要存储操作，这就是声明虚拟`compareTagTypes`方法并为其提供一个默认实现（返回`false`）的地方。`TagType`类还实现了在`Tag`类中声明的`match`方法，如下所示：

```cpp
    bool match (Tag const & other) const override
    {
        if (key() != other.key())
        {
            return false;
        }
        TagType const & otherCast =                 static_cast<TagType const &>(other);
        if (mOperation == TagOperation::None)
        {
            switch (otherCast.mOperation)
            {
            case TagOperation::None:
                return mValue == otherCast.mValue;
            default:
                return compareTagTypes(mValue,
                    otherCast.mOperation,
                    otherCast.mValue);
            }
        }
        switch (otherCast.mOperation)
        {
        case TagOperation::None:
            return compareTagTypes(otherCast.mValue,
                mOperation,
                mValue);
        default:
            return false;
        }
    }
```

`match`方法首先检查键，看看被比较的两个标签是否具有相同的键。如果键匹配，则假设类型相同，并将另一个标签转换为相同的`TagType`。

我们有几个场景需要确定。至少有一个标签应该是一个没有操作的正常标签，我们将称之为值。另一个标签也可以是一个没有操作的常规标签，在这种情况下，我们只需要比较两个值是否相等。

如果两个标签中有一个是正常的，而另一个有除了`None`之外的比较操作，那么设置了比较运算符的标签被视为标准。记住，知道哪个是值，哪个是标准是很重要的。代码需要处理比较值与标准或比较标准与值的情况。我们调用虚拟的`compareTagTypes`方法来进行实际比较，确保根据哪个是正常标签和哪个是标准，传递`mValue`和`otherCast.mValue`。

最后，如果两个标签的比较运算符都设置为除`None`之外的内容，那么我们将匹配视为`false`，因为比较两个标准标签之间没有意义。

在 `match` 方法中，我想要实现的部分有点复杂性，这个方法只在一个地方实现。这就是为什么我决定保留 `TagType` 类并创建特定于值类型的中间类，如 `StringTagType`。`TagType` 类通过确定正在比较什么以及什么与什么进行比较来实现部分比较，然后依赖于特定类型的类来完成实际的比较。

我们需要添加其他特定类型的中间标签类。所有这些都在 `Log.h` 文件中，紧接在 `StringTagType` 类之后。以下是 `int` 类型的示例：

```cpp
template <typename T>
class IntTagType : public TagType<T, int>
{
protected:
    IntTagType (int const & value,
        TagOperation operation)
    : TagType<T, int>(value, operation)
    { }
    bool compareTagTypes (int const & value,
        TagOperation operation,
        int const & criteria) const override
    {
        switch (operation)
        {
        case TagOperation::Equal:
            return value == criteria;
        case TagOperation::LessThan:
            return value < criteria;
        case TagOperation::LessThanOrEqual:
            return value <= criteria;
        case TagOperation::GreaterThan:
            return value > criteria;
        case TagOperation::GreaterThanOrEqual:
            return value >= criteria;
        default:
            return false;
        }
    }
};
```

这个类几乎与 `StringTagType` 类相同，只是针对 `int` 类型进行了更改，而不是字符串。主要区别在于比较可以使用简单的算术运算符来完成，而不是调用字符串的 `compare` 方法。

我考虑过将这个类用于所有的 `int`、`long long` 和 `double` 算术类型，但这意味着它仍然需要一个模板参数来指定实际类型。那么，问题就变成了一致性。`StringTagType` 类也应该有一个模板参数来指定字符串的类型吗？也许吧。因为存在不同类型的字符串，所以这几乎是有道理的。但关于 `bool` 类型怎么办？我们还需要一个中间类来处理布尔值，当类名中已经包含 `bool` 时，指定一个 `bool` 模板类型似乎很奇怪。因此，为了保持一切的一致性，我决定为所有支持的类型使用单独的中间类。我们将使用 `IntTagType` 类来处理整数，并创建另一个名为 `LongLongTagType` 的类，如下所示：

```cpp
template <typename T>
class LongLongTagType : public TagType<T, long long>
{
protected:
    LongLongTagType (long long const & value,
        TagOperation operation)
    : TagType<T, long long>(value, operation)
    { }
    bool compareTagTypes (long long const & value,
        TagOperation operation,
        long long const & criteria) const override
    {
        switch (operation)
        {
        case TagOperation::Equal:
            return value == criteria;
        case TagOperation::LessThan:
            return value < criteria;
        case TagOperation::LessThanOrEqual:
            return value <= criteria;
        case TagOperation::GreaterThan:
            return value > criteria;
        case TagOperation::GreaterThanOrEqual:
            return value >= criteria;
        default:
            return false;
        }
    }
};
```

我对这个类不是很满意，因为它与整数的实现完全相同。但让我高兴的是它创造了一致性。这意味着所有中间的标签类型类都可以以相同的方式使用。

下一个类是用于 `double` 的，尽管它也有相同的实现，但由于它们不像整型那样比较，所以有潜力以不同的方式比较双精度浮点数。在浮点值之间总是存在一点误差和细微的差异。目前，我们不会对双精度浮点数做任何不同的事情，但这个类将使我们能够在需要时以不同的方式比较它们。这个类的样子如下：

```cpp
template <typename T>
class DoubleTagType : public TagType<T, double>
{
protected:
    DoubleTagType (double const & value,
        TagOperation operation)
    : TagType<T, double>(value, operation)
    { }
    bool compareTagTypes (double const & value,
        TagOperation operation,
        double const & criteria) const override
    {
        switch (operation)
        {
        case TagOperation::Equal:
            return value == criteria;
        case TagOperation::LessThan:
            return value < criteria;
        case TagOperation::LessThanOrEqual:
            return value <= criteria;
        case TagOperation::GreaterThan:
            return value > criteria;
        case TagOperation::GreaterThanOrEqual:
            return value >= criteria;
        default:
            return false;
        }
    }
};
```

最后一个中间标签类型类是用于布尔值的，它确实需要做一些不同的事情。这个类实际上只对相等性感兴趣，其样子如下：

```cpp
template <typename T>
class BoolTagType : public TagType<T, bool>
{
protected:
    BoolTagType (bool const & value,
        TagOperation operation)
    : TagType<T, bool>(value, operation)
    { }
    bool compareTagTypes (bool const & value,
        TagOperation operation,
        bool const & criteria) const override
    {
        switch (operation)
        {
        case TagOperation::Equal:
            return value == criteria;
        default:
            return false;
        }
    }
};
```

现在我们已经解决了所有标签的问题，需要进行比较的地方是在 `log` 函数中，该函数目前使用标签的文本来比较正常和反转的标签。将 `normal` 块更改为如下所示：

```cpp
        for (auto const & normal: clause.second.normalLiterals)
        {
            // We need to make sure that the tag is
            // present and with the correct value.
            if (not activeTags.contains(normal->key()))
            {
                allLiteralsMatch = false;
                break;
            }
            if (not activeTags[normal->key()]->match(*normal))
            {
                allLiteralsMatch = false;
                break;
            }
        }
```

代码仍然遍历标签并检查涉及的键是否存在。一旦发现标签存在并且需要比较，代码现在将调用 `match` 方法，而不是获取每个标签的文本并比较它们是否相等。

反转块需要以类似的方式进行更改，如下所示：

```cpp
        for (auto const & inverted:             clause.second.invertedLiterals)
        {
            // We need to make sure that the tag is either
            // not present or has a mismatched value.
            if (activeTags.contains(inverted->key()))
            {
                if (activeTags[inverted->key()]->match(                   *inverted))
                {
                    allLiteralsMatch = false;
                }
                break;
            }
        }
```

对于反转循环，我能够稍微简化一下代码。真正的变化与正常循环类似，其中调用`match`方法进行比较，而不是直接比较标签文本。

在我们能够构建和尝试新的测试之前，我们需要更新测试应用中其他派生标签类型。就像我们需要更新`LogLevel`标签类以使用新的中间标签类一样，我们需要更改`LogTags.h`中的所有标签类。首先是`Color`类，如下所示：

```cpp
class Color : public MereMemo::StringTagType<Color>
{
public:
    static constexpr char key[] = "color";
    Color (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
```

`Color`类基于字符串值类型，就像`LogLevel`。

`Size`标签类型也使用字符串，现在看起来像这样：

```cpp
class Size : public MereMemo::StringTagType<Size>
{
public:
    static constexpr char key[] = "size";
    Size (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
```

`Count`和`Identity`标签类型分别基于`int`类型和`long long`类型，它们看起来像这样：

```cpp
class Count : public MereMemo::IntTagType<Count>
{
public:
    static constexpr char key[] = "count";
    Count (int value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : IntTagType(value, operation)
    { }
};
class Identity : public MereMemo::LongLongTagType<Identity>
{
public:
    static constexpr char key[] = "id";
    Identity (long long value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : LongLongTagType(value, operation)
    { }
};
```

最后，`Scale`和`CacheHit`标签类型分别基于`double`类型和`bool`类型，它们看起来像这样：

```cpp
class Scale : public MereMemo::DoubleTagType<Scale>
{
public:
    static constexpr char key[] = "scale";
    Scale (double value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : DoubleTagType(value, operation)
    { }
};
class CacheHit : public MereMemo::BoolTagType<CacheHit>
{
public:
    static constexpr char key[] = "cache_hit";
    CacheHit (bool value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : BoolTagType(value, operation)
    { }
};
```

每个标签类型的更改都很小。我认为这是可以接受的，尤其是因为使用标签类型的测试不需要更改。让我们再次看看开始这个部分的测试：

```cpp
TEST("Tag values can be used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(),
        Count(100, MereMemo::TagOperation::GreaterThan));
    std::string message = "values ";
    message += Util::randomString();
    MereMemo::log(Count(1)) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
    MereMemo::log() << Count(101) << message;
    result = Util::isTextInFile(message, "application.log");
    CONFIRM_FALSE(result);
    MereMemo::log(Count(101)) << message;
    result = Util::isTextInFile(message, "application.log");
    CONFIRM_TRUE(result);
}
```

这个测试现在应该更有意义。它创建了一个值为`100`的`Count`标签和一个`TagOperation`标签为`GreaterThan`。操作使得这个标签成为了一个可以与其他`Count`标签实例进行比较的准则标签，以查看其他实例中的计数是否真的大于 100。

然后，测试尝试使用具有值为`1`的正常`Count`标签进行日志记录。我们现在知道这将如何失败匹配，并且日志消息将被忽略。

测试随后尝试使用`101`的`Count`标签进行日志记录，但这次标签位于`log`函数外部，将不会被考虑。第二个日志消息也将被忽略，而不会尝试调用`match`。

测试随后尝试在`log`函数内部使用`101`个计数进行日志记录。这个应该匹配，因为 101 确实大于 100，并且消息应该出现在输出日志文件中。

注意测试的结构。它从几个已知场景开始，这些场景不应该成功，最终过渡到一个应该成功的场景。当你编写测试时，这是一个很好的模式，有助于确认一切按设计工作。

现在过滤功能即使在相对比较的情况下也能完全工作！本章的其余部分将提供见解和建议，以帮助您设计更好的测试。

# 何时测试过多？

我记得曾经听到过一个关于一个孩子在重症监护室的故事，他连接着所有的监测机器，包括一个监测心跳电信号的机器。孩子的状况突然恶化，显示出大脑血流不足的所有迹象。医生们无法理解为什么会这样，因为心跳还在，他们正准备将孩子送去扫描以寻找可能导致中风的血凝块，这时一位医生想到要听一下心跳。没有声音。机器显示心跳还在，但没有声音来确认心跳。医生们能够确定心脏周围的水肿正在对心脏施加压力，阻止它跳动。我不知道他们是怎么做到的，但他们减少了肿胀，孩子的心脏又开始泵血了。

为什么这个故事会浮现在脑海中？因为监测心脏活动的机器正在寻找电信号。在正常情况下，适当的电信号是监测心脏活动的好方法。但它是不直接的。电信号是心脏跳动的*方式*。信号导致心脏跳动，但正如故事所示，它们并不总是意味着*心脏真的在跳动*。

很容易在软件测试中陷入同样的陷阱。我们以为因为我们有很多测试，所以软件一定经过了很好的测试。但测试真的测试了正确的事情吗？换句话说，每个测试都在寻找有形的结果吗？或者有些测试只是在看结果通常是如何获得的？

何时测试过多？我的答案是测试是好的，你添加的每个测试通常都会帮助提高软件的质量。如果测试开始关注错误的事情，它就会变得过多。

并非一个关注错误事情的测试是坏的。坏的部分在于我们依赖于这个测试来预测某些结果。直接确认期望的结果比确认过程中的某个内部步骤要好得多。

例如，看看最近添加了一个`filter`字面量的测试：

```cpp
TEST("Tag values can be used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(),
        Count(100, MereMemo::TagOperation::GreaterThan));
```

我们本可以验证过滤器确实被添加到了收集中。我们可以在测试中调用`getFilterClauses`函数，检查每个子句，寻找刚刚添加的字面量。我们甚至可以确认这个字面量本身的行为符合预期，并且将值`100`分配给了这个字面量。

测试并不这样做。为什么？因为这就是过滤器的工作方式。在收集过程中寻找过滤器就像观察心跳电信号一样。能够调用`getFilterClauses`的能力仅仅是因为我们希望将日志库包含在一个单独的头文件中。这个函数并不是打算由客户调用的。测试反而检查设置过滤器的结果。

一旦设置好过滤器，测试就会尝试记录一些消息，并确保结果符合预期。

如果日志库需要某种自定义集合，那么测试过滤字面量是否正确添加到集合中是否有意义呢？我的回答是，不，至少在这里的过滤测试中不是这样。

如果项目需要自定义集合，那么它需要测试以确保集合能够正常工作。我并不是说因为代码在项目中的支持角色而跳过任何需要编写的代码的测试。我想要表达的是，要使测试集中在它们要测试的内容上。测试试图确认的期望结果是什么？在过滤测试的情况下，期望的结果是某些日志消息将被忽略，而其他消息将出现在输出日志文件中。测试直接设置确认结果所需的条件，执行必要的步骤，并确认结果。在这个过程中，集合和所有匹配的代码也将以间接方式得到测试。

如果涉及到自定义集合，那么间接测试是不够的。但在过滤器测试中进行直接测试也是不合适的。我们需要的是一组设计用来直接测试自定义集合本身的测试。

因此，如果我们需要像自定义集合这样的支持组件，那么这个组件需要单独进行测试。这些测试可以包含在同一个整体测试应用程序中。也许可以将它们放入自己的测试套件中。考虑一下将作为组件客户的代码，并考虑客户的需求。

如果组件足够大或具有更通用的用途，以至于它可能在项目之外也有用，那么给它一个单独的项目是一个好主意。这就是我们在本书中将单元测试库和日志库视为独立项目所做的事情。

关于过度测试的最后一想法将帮助您识别何时处于这种情况，因为很容易滑入过度间接测试。如果您发现重构软件工作方式后需要更改大量测试，那么您可能测试得太多了。

考虑一下这一章是如何添加过滤器并能够几乎完全不变地保留现有测试的。当然——我们不得不通过添加一系列中间标签类型类来更改底层的代码，但我们不需要重写现有的测试。

如果重构导致测试也需要大量工作，那么要么是你测试得太多，要么可能是问题在于你正在改变软件的预期使用方式。注意改变你希望设计如何被使用的方式，因为如果你遵循 TDD，那么最初的使用方式是你想要首先做对的事情之一。一旦你以使软件易于使用和直观的方式设计了软件，那么在重构可能改变测试的情况下要格外小心。

下一个部分解释了与这一部分相关的一个主题。一旦你知道需要测试什么，接下来经常出现的问题是如何设计软件使其易于测试，特别是测试是否需要深入到被测试组件的内部工作原理。

# 测试应该有多侵入性？

设计易于测试的软件是有好处的。对我来说，这始于遵循 TDD（测试驱动开发）并首先编写测试，这些测试利用软件以客户最期望的方式使用。这是最重要的考虑因素。

你不希望你的软件用户质疑为什么需要额外的步骤，或者为什么难以理解如何使用你的软件。在这里，客户或用户指的是任何将使用你的软件的人。客户或用户可能是一个需要使用正在设计的库的软件开发者。测试是用户必须经历的很好的例子。如果用户必须采取的额外步骤对用户没有任何价值，那么应该移除这个步骤，即使这个步骤使得测试代码更容易进行。

额外的步骤可能可以隐藏起来，如果可以的话，那么只要它使测试变得更好，就有可能保留它。任何时候测试依赖于用户不需要或不知道的额外内容，那么测试就是在侵犯软件设计。

我并不是说这是坏事。侵入通常有负面含义。只要你知道这会使你容易陷入前述章节描述的陷阱：过度测试，测试能够深入到组件内部对测试来说可能是好事。

需要理解的主要一点是，任何测试所使用的都应该成为支持接口的一部分。如果一个组件暴露了内部工作原理以便测试可以确认，那么这个内部工作原理应该被视为设计的一部分，而不是任何可以随时更改的内部细节。

本节所描述的内容与上一节之间的区别在于什么被同意支持。当我们试图测试那些应该在其他地方测试或内部细节且应该对测试不可达的事物时，我们会进行过多的测试。如果一个内部细节是稳定的，并且被同意不应该改变，并且如果这个内部细节使测试更可靠，那么测试使用这个细节可能是合理的。

我记得多年前参与的一个项目，该项目通过**可扩展标记语言**（**XML**）暴露了类的内部状态。有时状态可能相当复杂，使用 XML 可以让测试确认状态配置正确。然后，XML 会被传递给其他使用它的类。用户对 XML 并不知情，也不需要使用它，但测试依赖于它来将复杂的场景一分为二。测试的一半可以通过验证 XML 匹配来确保配置正确。然后另一半可以确保在提供已知 XML 输入数据时采取的行动是正确的。

软件并不需要按照这种方式设计才能使用 XML。甚至可以说，测试侵入了设计。XML 成为了设计的一部分。原本可能只是细节的东西变成了更多。但我还会进一步说，在这种情况下使用 XML 从来不是作为一个细节开始的。这是一个有意识的设计决策，是为了特定的原因——使测试更加可靠。

到目前为止，我们只探讨了单元测试。这就是为什么这本书从构建单元测试库开始。在考虑应该测试什么以及测试应该有多侵入性时，下一节将开始解释其他类型的测试。

# 在 TDD 中，集成或系统测试放在哪里？

有时候，创建一个将多个组件组合在一起并确认正在构建的整体系统按预期工作的测试是很有好处的。这些被称为集成测试，因为它们将多个组件集成在一起以确保它们能够良好地协同工作。或者，这些测试也可以被称为系统测试，因为它们测试整个系统。这两个名称在大多数情况下是可以互换的。

对于我们的微服务开发者，他们是日志库的目标客户，他们可能会为单个服务编写单元测试，甚至为服务内部的各种类和函数编写单元测试。特定服务的某些测试甚至可能被称为集成测试，但通常，集成测试将与多个服务一起工作。服务应该协同工作以完成更大的任务。因此，确保整体结果可以实现的测试将有助于提高所有参与服务的可靠性和质量。

如果你不是在构建一组微服务呢？如果你正在构建一个桌面应用程序来管理加密货币钱包呢？你仍然可以使用系统测试。也许你想要一个系统测试，它可以打开一个新的钱包并确保它能够同步到当前区块的区块链数据，或者也许你想要另一个系统测试，它停止同步然后再重新开始。每个这样的测试都将使用许多不同的组件，例如应用程序中的类和函数。系统测试确保能够实现某些高级目标，更重要的是，系统测试使用通过网络下载的真实数据。

系统测试通常需要很长时间才能完成。加上多个系统测试，整个测试集可能需要几个小时才能运行。或者，也许有一些测试会连续使用软件一天或更长时间。

一个特定的测试是否被称为单元测试或系统测试，通常取决于其运行所需的时间以及所需的资源。单元测试通常运行得很快，能够确定某事物是否通过，而无需依赖其他外部因素或组件。如果一个测试需要从另一个服务请求信息，那么这通常是一个很好的迹象，表明这个测试更像是集成测试而不是单元测试。单元测试不应该需要从网络上下载数据。

当谈到 TDD 时，为了使测试真正驱动设计——正如其名称所暗示的——那么测试通常将是单元测试类型。请别误会——系统测试很重要，可以帮助发现单元测试可能错过的奇怪使用模式。但典型的系统测试或集成测试并不是为了确保设计易于使用且直观。相反，系统测试确保能够达到高级目标，并且没有任何东西会破坏最终目标。

如果系统测试和集成测试之间有任何区别，那么在我看来，它归结为集成测试主要是确保多个组件能够良好地协同工作，而系统测试则更多地关注高级目标。集成测试和系统测试都高于单元测试。

TDD 在创建小型组件和函数的初始设计时更多地使用了单元测试。然后，TDD 利用系统测试和集成测试来确保整体解决方案合理且运行正常。

你可以将我们对日志库进行的所有测试都视为单元测试库的系统测试。我们确保单元测试库实际上可以帮助设计另一个项目。

至于系统或集成测试放在哪里，它们通常属于不同的测试项目——可以独立运行的项目。这甚至可以是一个脚本。如果你把它们放在与单元测试相同的测试项目中，那么就需要有一种方法来确保在需要快速响应时只运行单元测试。

除了系统和集成测试之外，还有更多测试你可能想要考虑添加。下一节将描述更多类型的测试。

# 那么，其他类型的测试又如何呢？

还有更多类型的测试需要考虑，例如性能测试、负载测试和渗透测试。你甚至可以涉及到可用性测试、升级测试、认证测试、持续运行测试等等，包括我可能从未听说过的类型。

每种测试类型都有对软件开发有价值的用途。每种类型都有自己的流程和步骤，测试的运行方式以及验证成功的方式。

性能测试可能会选择一个特定的场景，比如加载一个大文件，并确保操作可以在一定时间内完成。如果测试还检查确保操作仅使用一定量的计算机内存或 CPU 时间即可完成，那么在我看来，它开始更像是一个负载测试。而且如果测试确保最终用户不需要等待或被通知延迟，那么它开始更像是一个可用性测试。

测试类型之间的界限有时并不清晰。前一个章节已经解释了系统测试和集成测试通常是同一件事，有一个细微的区别，通常并不重要。其他测试也是如此。例如，一个特定的测试是负载测试还是性能测试，通常取决于意图。测试是否试图确保操作在特定时间内完成？谁决定什么时间足够好？或者，测试是否试图确保操作可以在同时进行其他事情时完成？或者，也许对于加载大文件的测试，一个几兆字节的大文件用于性能测试，因为这是客户可能遇到的一个典型的大文件，而负载测试会尝试加载一个更大的文件。这些只是一些想法。

渗透测试略有不同，因为它们通常作为官方安全审查的一部分创建。整个软件解决方案将被分析，产生大量文档，并创建测试。渗透测试通常试图确保在提供恶意数据或系统被滥用时，软件不会崩溃。

其他渗透测试将检查信息泄露。是否有滥用软件的可能性，使得攻击者获得本应保持机密的知识？

更重要的是渗透测试，它可以捕捉到数据操纵。一个常见的例子是学生试图更改他们的成绩，但这种攻击可以用来窃取金钱或删除关键信息。

提权攻击对于防止渗透测试至关重要，因为它们让攻击者获得可以导致更多攻击的访问权限。当攻击者能够控制远程服务器时，这显然是一种提权，但提权可以用来获得攻击者通常没有的任何额外权限或能力。

可用性测试更加主观，通常涉及客户访谈或试用。

所有不同类型的测试都很重要，我的目标不是列出或描述所有可能的测试类型，而是给你一个关于可用的测试类型以及不同测试可以提供哪些益处的概念。

软件测试不是关于使用哪种测试的问题，而是每种类型在过程中的位置。关于每种测试类型都可以写一本书，而且已经有很多这样的书。这本书之所以如此专注于单元测试，是因为单元测试与 TDD 过程最为接近。

# 摘要

TDD 过程比本章中添加到日志库的功能更为重要。我们添加了日志级别、标签和过滤功能，甚至重构了日志库的设计。虽然所有这些都是有价值的，但最重要的还是要关注涉及的过程。

本章之所以如此详细，是为了让你看到所有设计决策以及测试是如何在整个过程中起到指导作用的。你可以将这种学习应用到自己的项目中。如果你也使用日志库，那么这将是额外的收获。

你学习了理解客户需求的重要性。客户不一定是走进商店买东西的人。客户是正在开发的软件的预期用户。这甚至可以是另一个软件开发者或公司内的另一个团队。理解预期用户的需求将使你能够编写更好的测试来解决这些需求。

编写一个看似合适的函数或设计一个接口非常容易，但后来发现很难使用。先编写测试可以帮助避免使用问题。在本章中，你看到了一个我仍然需要回去更改函数工作方式的地方，因为测试显示它存在逆向问题。

需要支持按值过滤日志消息的广泛更改，而本章展示了如何在保持测试不变的情况下进行更改。

理解 TDD 的最好方法是在项目中使用这个过程。本章为日志库开发了很多新代码，让你能够近距离观察这个过程，并提供了比简单示例所能展示的更多内容。

下一章将探讨依赖关系，并将日志库扩展到向多个日志文件目的地发送日志消息。
