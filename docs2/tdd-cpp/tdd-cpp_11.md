

# 管理依赖关系

识别依赖关系并在依赖关系使用的公共接口周围实现您的代码将有助于您以多种方式。您将能够做到以下事情：

+   避免等待其他团队或甚至自己完成复杂且必要的组件

+   将您的代码隔离并确保其正常工作，即使您使用的其他代码中存在错误

+   通过您的设计实现更大的灵活性，以便您只需更改依赖组件即可更改行为

+   创建接口，清晰地记录并突出关键要求

在本章中，您将了解依赖关系是什么以及如何设计您的代码来使用它们。到本章结束时，您将了解如何更快地完成代码编写并证明其工作，即使项目的其余部分尚未准备好也是如此。

您不需要使用 TDD 来设计和使用依赖关系。但如果您正在使用 TDD，那么整个过程将变得更好，因为您还将能够编写更好的测试，这些测试可以专注于代码的特定区域，而无需担心来自代码外部的额外复杂性和错误。

本章将涵盖以下主要主题：

+   基于依赖进行设计

+   添加多个日志输出

# 技术要求

本章中所有代码都使用基于任何现代 C++ 20 或更高版本编译器和标准库的标准 C++。代码使用了本书*第一部分*，*测试 MVP*中提到的测试库，并继续开发在前面章节中开始的日志库。

您可以在以下 GitHub 仓库中找到本章所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

)

# 基于依赖进行设计

依赖关系并不总是显而易见的。如果一个项目使用库，例如日志项目如何使用单元测试库，那么这是一个容易发现的依赖关系。日志项目依赖于单元测试库以正确运行。或者在这种情况下，只有日志测试依赖于单元测试库。但这已经足够形成一个依赖关系。

另一个容易发现的依赖关系是如果您需要调用另一个服务。即使代码在调用之前检查其他服务是否可用，依赖关系仍然存在。

库和服务是*外部依赖关系*的好例子。您必须做额外的工作才能使项目使用另一个项目的代码或服务，这就是为什么外部依赖关系如此容易被发现。

其他依赖关系更难发现，这些通常是项目内部的*内部依赖关系*。从某种意义上说，项目中的几乎所有代码都依赖于其他代码正确执行其预期功能。因此，让我们细化一下我们对依赖关系的理解。通常，当提到依赖关系时，与代码设计相关，我们指的是可以交换的东西。

这可能通过外部服务依赖的例子最容易理解。该服务在自己的接口上运行。你使用服务定义的接口，根据其位置或地址向服务发出请求。如果第一个服务不可用，你可以为相同的请求调用不同的服务。理想情况下，两个服务会使用相同的接口，这样你代码需要更改的只有地址。

如果两个服务使用不同的接口，那么为每个服务创建一个包装器可能是有意义的，这个包装器知道如何将每个服务期望的内容翻译成你的代码将使用的*通用接口*。有了通用接口，你可以交换一个服务为另一个服务，而无需更改代码。你的代码更多地依赖于服务接口定义，而不是任何特定的服务。

如果我们看看内部设计决策，可能有一个基类和一个派生类。派生类肯定依赖于基类，但这种依赖类型不能在不重写代码以使用不同的基类的情况下更改。

当考虑日志库定义的标签时，我们更接近可以替换的依赖。可以定义新的标签并使用它们，而无需更改现有代码。日志库可以使用任何标签，无需担心每个标签的作用。但我们是真的在替换标签吗？对我来说，标签的设计是为了在日志文件中以一致的方式解决日志键=值元素的问题，而不依赖于值的类型。尽管日志库依赖于标签及其使用的接口，但我不会将标签设计归类为与外部服务相同的依赖类型。

在早期思考日志库时，我提到过我们需要能够将日志信息发送到不同的目的地，或者甚至多个目的地。代码使用`log`函数，并期望它要么被忽略，要么发送到某个地方。将日志消息发送到特定目的地的能力是日志库需要依赖的。日志库应该让执行日志的项目决定目的地。

这就引出了依赖关系的另一个方面。依赖通常是指配置过的某些东西。我的意思是，我们可以这样说，日志库依赖于某些组件来执行将消息发送到目的地的任务。日志库可以被设计成选择自己的目的地，或者日志库可以被告知使用哪个依赖。当我们让其他代码控制依赖时，我们得到的是所谓的*依赖注入*。当你允许调用代码注入依赖时，你会得到一个更灵活的解决方案。

这里有一些我放入 `main` 函数中的初始代码，用于配置一个知道如何将日志消息发送到文件的组件，然后将文件组件注入到日志记录器中，以便日志记录器知道将日志消息发送到何处：

```cpp
int main ()
{
    MereMemo::FileOutput appFile("application.log");
    appFile.maxSize() = 10'000'000;
    appFile.rolloverCount() = 5;
    MereMemo::addLogOutput(appFile);
    MereMemo::addDefaultTag(info);
    MereMemo::addDefaultTag(green);
    return MereTDD::runTests(std::cout);
}
```

策略是创建一个名为 `FileOutput` 的类，并给它提供写入日志消息的文件名。因为我们不希望日志文件变得太大，所以我们应该能够指定最大大小。代码使用 1000 万字节作为最大大小。当日志文件达到最大大小时，我们应该停止向该文件写入，并创建一个新的文件。我们应该能够在开始删除旧文件之前指定要创建的日志文件数量。代码将最大日志文件数设置为五个。

一旦创建并配置好我们想要的 `FileOutput` 实例，就可以通过调用 `addLogOutput` 函数将其注入到日志库中。

这段代码能满足我们的需求吗？它是否直观且易于理解？尽管这不是一个测试，但我们仍然通过在编写实现新功能的代码之前专注于新功能的用法来遵循 TDD。

至于满足我们的需求，这并不是真正需要问的问题。我们需要问的是它是否能够满足我们目标客户的需求。我们正在设计一个日志库，供微服务开发者使用。可能有数百个服务在服务器计算机上运行，我们真的应该将日志文件放在特定的位置。我们需要的第一个更改是让调用者指定日志文件应该创建的路径。路径似乎应该与文件名分开。

对于文件名，我们将如何命名多个日志文件？它们不能都叫 `application.log`。文件应该编号吗？它们都将放在同一个目录中，文件系统唯一的要求是每个文件都有一个唯一的名称。我们需要让调用者提供日志文件名的模式，而不是单个文件名。一个模式将让日志库知道如何使名称唯一，同时仍然遵循开发者想要的总体命名风格。我们可以将初始代码更改为如下：

```cpp
    MereMemo::FileOutput appFile("logs");
    appFile.namePattern() = "application-{}.log";
    appFile.maxSize() = 10'000'000;
    appFile.rolloverCount() = 5;
    MereMemo::addLogOutput(appFile);
```

在设计一个类时，在构造后让类以合理的默认值工作是一个好主意。对于文件输出，我们需要的最基本的是创建日志文件的目录。其他属性很好，但不是必需的。如果没有提供名称模式，我们可以默认为简单的唯一数字。最大大小可以有一个无限默认值，或者至少一个非常大的数字。我们只需要一个日志文件。因此，轮换计数可以是一个告诉我们使用单个文件的值。

我决定使用简单的花括号 `{}` 作为模式中的占位符，其中将放置一个唯一的数字。我们将随机选择一个三位数来使日志文件名唯一。这将给我们提供多达一千个日志文件，这应该足够了。大多数用户可能只想保留少数几个，并删除较旧的文件。

因为输出是一个可以被替换的依赖，甚至可以同时有多个输出，那么不同类型的输出会是什么样子呢？我们将在稍后确定输出依赖组件接口。现在，我们只想探索如何使用不同的输出。以下是输出如何发送到 `std::cout` 控制台的方式：

```cpp
    MereMemo::StreamOutput consoleStream(std::cout);
    MereMemo::addLogOutput(consoleStream);
```

控制台输出是一个 ostream，因此我们应该能够创建一个可以与任何 ostream 一起工作的流输出。这个例子创建了一个名为 `consoleStream` 的输出组件，它可以像文件输出一样添加到日志输出中。

在使用 TDD 时，避免添加可能并非客户真正需要的有趣特性是很重要的。我们不会添加删除输出的功能。一旦输出被添加到日志库中，它将保持不变。为了删除输出，我们可能需要返回某种标识符，以便稍后删除之前添加的相同输出。我们确实添加了删除过滤条件的能力，因为这看起来可能是需要的。对于大多数客户来说，删除输出似乎不太可能。

为了设计一个可以被其他依赖替换的依赖，我们需要一个所有输出都实现的公共接口类。这个类将被命名为 `Output`，并放置在 `Log.h` 文件中，紧挨着 `LogStream` 类之前，如下所示：

```cpp
class Output
{
public:
    virtual ~Output () = default;
    Output (Output const & other) = delete;
    Output (Output && other) = delete;
    virtual std::unique_ptr<Output> clone () const = 0;
    virtual void sendLine (std::string const & line) = 0;
    Output & operator = (Output const & rhs) = delete;
    Output & operator = (Output && rhs) = delete;
protected:
    Output () = default;
};
```

接口中仅包含 `clone` 和 `sendLine` 方法。我们将遵循与标签类似的克隆模式，但不会使用模板。`sendLine` 方法将在需要将一行文本发送到输出时被调用。其他方法确保没有人可以直接构造 `Output` 的实例，或者复制或分配一个 `Output` 实例到另一个实例。`Output` 类被设计为可以被继承的。

我们将通过接下来的两个函数来跟踪所有已添加的输出，这两个函数紧随 `Output` 类之后，如下所示：

```cpp
inline std::vector<std::unique_ptr<Output>> & getOutputs ()
{
    static std::vector<std::unique_ptr<Output>> outputs;
    return outputs;
}
inline void addLogOutput (Output const & output)
{
    auto & outputs = getOutputs();
    outputs.push_back(output.clone());
}
```

`getOutputs` 函数使用一个静态的唯一指针向量，并在请求时返回集合。`addLogOutput` 函数将给定输出的克隆添加到集合中。这都与默认标签的处理方式相似。

你应该知道的一个有趣的依赖关系的使用是它们能够用一个假组件替换一个真实组件的能力。我们正在添加两个真实组件来管理日志输出。一个将输出到文件，另一个到控制台。但是，如果你想在你代码的进展上取得进展，而正在等待另一个团队完成编写所需的组件时，你也可以使用依赖关系。与其等待，不如将组件设置为依赖关系，你可以用更简单的版本替换它。这个更简单的版本不是真实版本，但它应该更快编写，并让你继续取得进展，直到真实版本可用。

一些其他的测试库将这种假依赖关系的能力更进一步，并允许你用几行代码创建可以以各种方式响应的组件，这些方式你可以控制。这让你可以隔离你的代码，并确保它按预期行为，因为你可以依赖假依赖关系始终按指定方式行为，你也不再需要担心真实依赖关系中的错误会影响测试结果。这些假组件的通用术语是*模拟*。

无论你是使用一个通过几行代码为你生成模拟的测试库，还是你自己编写模拟，都没有关系。任何时候，当你有一个模仿另一个类的类时，你就有了一个模拟。

除了将你的代码与错误隔离之外，模拟还可以帮助你加快测试速度，并改善与其他团队的协作。速度的提高是因为真实代码可能需要花费时间请求或计算结果，而模拟可以快速返回，无需进行任何实际工作。与其他团队的协作得到改善，因为每个人都可以同意简单的模拟，这些模拟易于开发，可以用来传达设计变更。

下一节将实现基于通用接口的文件和流输出类。我们将能够简化 `LogStream` 类和 `log` 函数，以使用通用接口，这将记录并使理解发送日志消息到输出的真正需求变得更加容易。

# 添加多个日志输出

验证设计是否适用于多种场景的一个好方法是实现每个场景的解决方案。我们有一个通用的 `Output` 接口类，它定义了两个方法，`clone` 和 `sendLine`，我们需要确保这个接口将适用于将日志消息发送到日志文件和到控制台。

让我们从继承自 `Output` 的一个名为 `FileOutput` 的类开始。新类放在 `Log.h` 中，紧接在 `getOutputs` 和 `addLogOutput` 函数之后，如下所示：

```cpp
class FileOutput : public Output
{
public:
    FileOutput (std::string_view dir)
    : mOutputDir(dir),
    mFileNamePattern("{}"),
    mMaxSize(0),
    mRolloverCount(0)
    { }
    FileOutput (FileOutput const & rhs)
    : mOutputDir(rhs.mOutputDir),
    mFileNamePattern(rhs.mFileNamePattern),
    mMaxSize(rhs.mMaxSize),
    mRolloverCount(rhs.mRolloverCount)
    { }
    FileOutput (FileOutput && rhs)
    : mOutputDir(rhs.mOutputDir),
    mFileNamePattern(rhs.mFileNamePattern),
    mMaxSize(rhs.mMaxSize),
    mRolloverCount(rhs.mRolloverCount),
    mFile(std::move(rhs.mFile))
    { }
    ~FileOutput ()
    {
        mFile.close();
    }
    std::unique_ptr<Output> clone () const override
    {
        return std::unique_ptr<Output>(
            new FileOutput(*this));
    }
    void sendLine (std::string const & line) override
    {
        if (not mFile.is_open())
        {
            mFile.open("application.log", std::ios::app);
        }
        mFile << line << std::endl;
        mFile.flush();
    }
protected:
    std::filesystem::path mOutputDir;
    std::string mFileNamePattern;
    std::size_t mMaxSize;
    unsigned int mRolloverCount;
    std::fstream mFile;
};
```

`FileOutput` 类遵循上一节中确定的用法，如下所示：

```cpp
    MereMemo::FileOutput appFile("logs");
    appFile.namePattern() = "application-{}.log";
    appFile.maxSize() = 10'000'000;
    appFile.rolloverCount() = 5;
    MereMemo::addLogOutput(appFile);
```

我们在构造函数中给`FileOutput`类一个目录，日志文件将保存在那里。该类还支持名称模式、最大日志文件大小和滚动计数。所有数据成员都需要在构造函数中初始化，我们有三个构造函数。

第一个构造函数是一个普通构造函数，它接受目录并为其他数据成员提供默认值。

第二个构造函数是复制构造函数，它根据`FileOutput`的另一个实例中的值初始化数据成员。只有`mFile`数据成员保留在默认状态，因为我们没有复制`fstream`。

第三个构造函数是移动复制构造函数，它看起来几乎与复制构造函数相同。唯一的区别是我们现在将`fstream`移动到正在构建的`FileOutput`类中。

析构函数将关闭输出文件。这实际上是对之前所做工作的重大改进。我们过去每次记录日志消息时都会打开和关闭输出文件。现在我们将打开日志文件并保持打开状态，直到我们稍后需要关闭它。析构函数确保如果日志文件尚未关闭，则将其关闭。

接下来是`clone`方法，它调用复制构造函数来创建一个新的实例，并将其作为基类的唯一指针返回。

`sendLine`方法是最后一个方法，在将行发送到文件之前，它需要检查输出文件是否已经打开。我们将在每行发送到输出文件后添加结束换行符。我们还每行刷新日志文件，这有助于确保在应用程序突然崩溃的情况下，日志文件包含所有写入的内容。

在`FileOutput`类中，我们需要做的最后一件事是定义数据成员。我们不会完全实现所有数据成员。例如，你可以看到我们仍然在打开一个名为`application.log`的文件，而不是遵循命名模式。我们已经有了基本想法，跳过数据成员将使我们能够测试这部分，以确保我们没有破坏任何东西。我们需要在`main`函数中注释掉配置，所以现在看起来是这样的：

```cpp
    MereMemo::FileOutput appFile("logs");
    //appFile.namePattern() = "application-{}.log";
    //appFile.maxSize() = 10'000'000;
    //appFile.rolloverCount() = 5;
    MereMemo::addLogOutput(appFile);
```

一旦我们以基本方式使多个输出工作，我们就可以随时返回配置方法和目录。这遵循了 TDD 实践，即每一步尽可能少做。从某种意义上说，我们正在为最终的`FileOutput`类创建一个模拟。

我差点忘了提，因为我们使用了`filesystem`功能，例如`path`，所以我们需要在`Log.h`的顶部包含`filesystem`，如下所示：

```cpp
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
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

当我们开始将日志文件滚动到新文件而不是每次都打开相同的文件时，我们将更多地使用`filesystem`。

接下来是`StreamOutput`类，它可以直接放在`FileOutput`类后面`Log.h`中，如下所示：

```cpp
class StreamOutput : public Output
{
public:
    StreamOutput (std::ostream & stream)
    : mStream(stream)
    { }
    StreamOutput (StreamOutput const & rhs)
    : mStream(rhs.mStream)
    { }
    std::unique_ptr<Output> clone () const override
    {
        return std::unique_ptr<Output>(
            new StreamOutput(*this));
    }
    void sendLine (std::string const & line) override
    {
        mStream << line << std::endl;
    }
protected:
    std::ostream & mStream;
};
```

`StreamOutput`类比`FileOutput`类简单，因为它具有更少的数据成员。我们只需要跟踪在`main`中的构造函数中传入的 ostream 引用。我们也不需要担心特定的移动复制构造函数，因为我们可以轻松地复制 ostream 引用。`StreamOutput`类已经在`main`中添加，如下所示：

```cpp
    MereMemo::StreamOutput consoleStream(std::cout);
    MereMemo::addLogOutput(consoleStream);
```

`StreamOutput`类将持有`main`传递给它的`std::cout`的引用。

现在我们正在处理输出接口，我们不再需要在`LogStream`类中管理文件。构造函数可以简化，不再需要担心 fstream 数据成员，如下所示：

```cpp
    LogStream ()
    : mProceed(true)
    { }
    LogStream (LogStream const & other) = delete;
    LogStream (LogStream && other)
    : std::stringstream(std::move(other)),
    mProceed(other.mProceed)
    { }
```

`LogStream`类的析构函数是所有工作的发生地。它不再需要直接将消息发送到由类管理的文件。析构函数现在获取*所有*输出，并使用通用接口将消息发送给每个输出，如下所示：

```cpp
    ~LogStream ()
    {
        if (not mProceed)
        {
            return;
        }

        auto & outputs = getOutputs();
        for (auto const & output: outputs)
        {
            output->sendLine(this->str());
        }
    }
```

记住，`LogStream`类从`std::stringstream`继承，并持有要记录的消息。如果我们继续进行，我们可以通过调用`str`方法来获取完整格式的消息。

`LogStream`类的末尾不再需要`mFile`数据成员，只需要`mProceed`标志，如下所示：

```cpp
private:
    bool mProceed;
};
```

由于我们移除了`LogStream`构造函数的文件名和打开模式参数，我们可以简化`log`函数中`LogStream`类的创建方式，如下所示：

```cpp
inline LogStream log (std::vector<Tag const *> tags = {})
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const tmNow =          std::chrono::system_clock::to_time_t(now);
    auto const ms = duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    LogStream ls;
    ls << std::put_time(std::gmtime(&tmNow),        "%Y-%m-%dT%H:%M:%S.")
        << std::setw(3) << std::setfill('0')         << std::to_string(ms.count());
```

我们现在可以不带任何参数构造`ls`实例，它将使用所有已添加的输出。

让我们通过构建和运行项目来检查测试应用程序。控制台输出的内容如下：

```cpp
Running 1 test suites
--------------- Suite: Single Tests
------- Test: Message can be tagged in log
2022-07-24T22:32:13.116 color="green" log_level="error" simple 7809
Passed
------- Test: log needs no namespace when used with LogLevel
2022-07-24T22:32:13.118 color="green" log_level="error" no namespace
Passed
------- Test: Default tags set in main appear in log
2022-07-24T22:32:13.118 color="green" log_level="info" default tag 9055
Passed
------- Test: Multiple tags can be used in log
2022-07-24T22:32:13.118 color="red" log_level="debug" size="large" multi tags 7933
Passed
------- Test: Tags can be streamed to log
2022-07-24T22:32:13.118 color="green" log_level="info" count=1 1 type 3247
2022-07-24T22:32:13.118 color="green" log_level="info" id=123456789012345 2 type 6480
2022-07-24T22:32:13.118 color="green" log_level="info" scale=1.500000 3 type 6881
2022-07-24T22:32:13.119 color="green" log_level="info" cache_hit=false 4 type 778
Passed
------- Test: Tags can be used to filter messages
2022-07-24T22:32:13.119 color="green" log_level="info" filter 1521
Passed
------- Test: Overridden default tag not used to filter messages
Passed
------- Test: Inverted tag can be used to filter messages
Passed
------- Test: Tag values can be used to filter messages
2022-07-24T22:32:13.119 color="green" count=101 log_level="info" values 8461
Passed
------- Test: Simple message can be logged
2022-07-24T22:32:13.120 color="green" log_level="info" simple 9466 with more text.
Passed
------- Test: Complicated message can be logged
2022-07-24T22:32:13.120 color="green" log_level="info" complicated 9198 double=3.14 quoted="in quotes"
Passed
-----------------------------------
Tests passed: 11
Tests failed: 0
```

你可以看到日志消息确实被发送到了控制台窗口。日志消息包含在控制台结果中。那么日志文件呢？它看起来如下所示：

```cpp
2022-07-24T22:32:13.116 color="green" log_level="error" simple 7809
2022-07-24T22:32:13.118 color="green" log_level="error" no namespace
2022-07-24T22:32:13.118 color="green" log_level="info" default tag 9055
2022-07-24T22:32:13.118 color="red" log_level="debug" size="large" multi tags 7933
2022-07-24T22:32:13.118 color="green" log_level="info" count=1 1 type 3247
2022-07-24T22:32:13.118 color="green" log_level="info" id=123456789012345 2 type 6480
2022-07-24T22:32:13.118 color="green" log_level="info" scale=1.500000 3 type 6881
2022-07-24T22:32:13.119 color="green" log_level="info" cache_hit=false 4 type 778
2022-07-24T22:32:13.119 color="green" log_level="info" filter 1521
2022-07-24T22:32:13.119 color="green" count=101 log_level="info" values 8461
2022-07-24T22:32:13.120 color="green" log_level="info" simple 9466 with more text.
2022-07-24T22:32:13.120 color="green" log_level="info" complicated 9198 double=3.14 quoted="in quotes"
```

日志文件只包含日志消息，这些日志消息与发送到控制台窗口的日志消息相同。这表明我们有多处输出！没有很好的方法来验证日志消息是否被发送到控制台窗口，例如，我们可以打开日志文件并搜索特定的行。

但我们可以使用`StreamOutput`类添加另一个输出，该类使用`std::fstream`而不是`std::cout`。我们可以这样做，因为 fstream 实现了 ostream，这正是`StreamOutput`类所需要的。这也是依赖注入，因为`StreamOutput`类依赖于一个 ostream，我们可以给它任何我们想要的 ostream，如下所示：

```cpp
#include <fstream>
#include <iostream>
int main ()
{
    MereMemo::FileOutput appFile("logs");
    //appFile.namePattern() = "application-{}.log";
    //appFile.maxSize() = 10'000'000;
    //appFile.rolloverCount() = 5;
    MereMemo::addLogOutput(appFile);
    MereMemo::StreamOutput consoleStream(std::cout);
    MereMemo::addLogOutput(consoleStream);
    std::fstream streamedFile("stream.log", std::ios::app);
    MereMemo::StreamOutput fileStream(streamedFile);
    MereMemo::addLogOutput(fileStream);
    MereMemo::addDefaultTag(info);
    MereMemo::addDefaultTag(green);
    return MereTDD::runTests(std::cout);
}
```

我们不会进行这个更改。这只是为了演示目的。但它表明你可以打开一个文件并将该文件传递给`StreamOutput`类以代替控制台输出。如果你真的做出这个更改，那么你会看到`stream.log`和`application.log`文件是相同的。

为什么你想考虑使用`StreamOutput`就像使用`FileOutput`一样？而且如果`StreamOutput`也可以写入文件，我们为什么还需要`FileOutput`？

首先，`FileOutput`是针对文件专门化的。它最终将知道如何检查当前文件大小，以确保它不会变得太大，并在当前日志文件接近最大大小时滚动到新的日志文件。需要文件管理，而`StreamOutput`甚至不会意识到这一点。

虽然`StreamOutput`类更简单，因为它根本不需要担心文件。你可能想使用`StreamOutput`将内容写入文件，以防`FileOutput`类创建得太慢。当然，我们创建了一个没有所有文件管理功能的简化`FileOutput`，但另一个团队可能并不愿意给你一个部分实现。你可能发现，在等待完整实现的同时使用模拟解决方案会更好。

能够交换一种实现方式为另一种实现方式是，通过适当管理的依赖关系获得的一大优势。

事实上，这本书将保留当前`FileOutput`的实现方式，因为它现在是这样，因为完成实现将使我们进入与学习 TDD 关系不大的主题。

# 摘要

我们不仅为日志库添加了一个让它能够将日志消息发送到多个目的地的出色新功能，而且还使用接口添加了这种能力。该接口有助于记录和隔离将文本行发送到目的地的概念。这有助于揭示日志库的一个依赖关系。日志库依赖于将文本发送到某处的功能。

目的地可以是日志文件或控制台，或者任何其他地方。在我们确定这个依赖关系之前，日志库在很多地方都做出了假设，认为它只与日志文件一起工作。我们能够简化设计，同时创建一个更灵活的设计。

我们还能够在没有完整的文件记录组件的情况下使文件记录功能正常工作。我们创建了一个文件记录组件的模拟，省略了完整实现所需的所有额外文件管理任务。虽然这些附加功能很有用，但目前并不需要，这个模拟将使我们能够在没有它们的情况下继续前进。

下一章将回到单元测试库，并展示如何将确认提升到一个可扩展且更容易理解的新风格。

# 第三部分：扩展 TDD 库以支持日志库不断增长的需求

本书分为三部分。在这第三部分和最后一部分，我们将增强单元测试确认，以使用一种称为 Hamcrest 确认的新现代风格。你还将学习如何测试服务和如何使用多线程进行测试。这一部分将把你迄今为止所学的一切结合起来，并为你使用 TDD 在自己的项目中做好准备。

本部分涵盖了以下章节：

+   *第十二章*，*创建更好的测试断言*

+   *第十三章*，*如何测试浮点数和自定义值*

+   *第十四章*, *如何测试服务*

+   *第十五章*, *多线程测试*
