# 15

# 如何使用多个线程进行测试

多线程是编写软件中最困难的部分之一。常常被忽视的是我们如何测试多个线程。我们能否使用 TDD 来帮助设计使用多个线程的软件？是的，TDD 可以帮助，你将在本章中找到有用的实用指导，它将向你展示如何使用 TDD 与多个线程一起工作。

本章的主要内容包括以下几项：

+   在测试中使用多个线程

+   使日志库线程安全

+   需要证明多线程的必要性

+   改变服务的返回类型

+   进行多次服务调用

+   如何在不使用睡眠的情况下测试多个线程

+   修复检测到的最后一个日志问题

首先，我们将检查在使用测试中的多个线程时你会遇到什么问题。你将学习如何使用测试库中的一个特殊辅助类来简化测试多个线程时所需的额外步骤。

一旦我们能在测试中使用多个线程，我们就会利用这个能力同时从多个线程调用日志库并观察会发生什么。我会给你一个提示：需要对日志库进行一些修改，以便在从多个线程调用时库能表现得更好。

然后，我们将回到上一章中开发的简单服务，你将学习如何使用 TDD 设计一个使用多个线程的服务，这样就可以支持可靠的测试。

在本章中，我们将依次处理每个项目。首先，我们将使用测试库项目。然后，我们将切换到日志库项目。最后，我们将使用简单服务项目。

# 技术要求

本章中的所有代码都使用标准 C++，它基于任何现代 C++ 20 或更高版本的编译器和标准库。本章中的代码使用了本书中开发的三个项目：来自*第一部分*的测试库*Testing MVP*，来自*第二部分*的日志库*Logging Library*，以及上一章中的简单服务。

你可以在这本书的 GitHub 仓库中找到本章的所有代码：[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)。

# 在测试中使用多个线程

在你的测试中添加多个线程带来的挑战，你需要意识到。我说的不是在多个线程中运行测试本身。测试库注册并运行测试，它将保持单线程。你需要理解的是在测试内部创建多个线程时可能出现的各种问题。

为了理解这些问题，让我们创建一个使用多个线程的测试，这样你就可以确切地看到会发生什么。在本节中，我们将与单元测试库项目一起工作，因此，首先添加一个名为`Thread.cpp`的新测试文件。在你添加了新文件后，项目结构应该看起来像这样：

```cpp
MereTDD project root folder
    Test.h
    tests folder
        main.cpp
        Confirm.cpp
        Creation.cpp
        Hamcrest.cpp
        Setup.cpp
        Thread.cpp
```

在`Thread.cpp`文件中，添加以下代码：

```cpp
#include "../Test.h"
#include <atomic>
#include <thread>
using namespace MereTDD;
TEST("Test can use additional threads")
{
    std::atomic<int> count {0};
    std::thread t1([&count]()
    {
        for (int i = 0; i < 100'000; ++i)
        {
            ++count;
        }
        CONFIRM_THAT(count, NotEquals(100'001));
    });
    std::thread t2([&count]()
    {
        for (int i = 0; i < 100'000; ++i)
        {
            --count;
        }
        CONFIRM_THAT(count, NotEquals(-100'001));
    });
    t1.join();
    t2.join();
    CONFIRM_THAT(count, Equals(0));
}
```

上述代码包括`atomic`，这样我们就可以安全地从多个线程修改`count`变量。我们需要包含`thread`来引入线程类的定义。测试创建了两个线程。第一个线程增加`count`，而第二个线程减少相同的`count`。最终结果应该将`count`返回到零，因为我们增加和减少的次数相同。

如果你构建并运行测试应用程序，一切都会通过。新的测试根本不会造成任何问题。让我们更改第三个`CONFIRM_THAT`宏，以便我们可以尝试在测试结束时确认`count`不等于`0`，如下所示：

```cpp
    t1.join();
    t2.join();
    CONFIRM_THAT(count, NotEquals(0));
```

这次更改导致测试失败，结果如下：

```cpp
------- Test: Test can use additional threads
Failed confirm on line 30
    Expected: not 0
    Actual  : 0
```

到目前为止，我们有一个使用多个线程的测试，它按预期工作。我们添加了一些确认，可以检测并报告当值不匹配预期值时的情况。你可能会想知道当线程似乎到目前为止都在正常工作时，多线程可能会引起什么问题。

这是个快速回答：在测试中创建一个或多个线程根本不会造成任何问题——也就是说，假设线程被正确管理，例如确保在测试结束时它们被连接。确认从主测试线程本身按预期工作。你甚至可以在附加线程中进行确认。当附加线程中的一个确认失败时，会出现一种问题。为了看到这一点，让我们将最终的确认放回`Equals`，并将第一个确认也改为`Equals`，如下所示：

```cpp
        for (int i = 0; i < 100'000; ++i)
        {
            ++count;
        }
        CONFIRM_THAT(count, Equals(100'001));
```

`count`永远不会达到`100'001`，因为我们只增加`100'000`次。在这次更改之前，确认总是通过，这就是为什么它没有引起问题的原因。但是，这次更改后，确认会立即失败。如果这是一个主测试线程中的确认，那么失败会导致测试失败，并带有描述问题的总结消息。但现在我们不在主测试线程中。

记住，失败的确认会抛出异常，并且线程内部未处理的异常会导致应用程序终止。当我们确认计数等于`100'001`时，我们导致抛出异常。主要的测试线程由测试库管理，主线程准备好捕获任何确认异常以便报告。然而，测试 lambda 内部的附加线程没有针对抛出异常的保护。因此，当我们构建和运行测试应用程序时，它会像这样终止：

```cpp
------- Test: Test can use additional threads
terminate called after throwing an instance of 'MereTDD::ActualConfirmException'
Abort trap: 6
```

根据你使用的计算机不同，你可能会得到一条稍微不同的消息。你不会得到的是运行并报告所有测试结果的应用程序。当附加线程中的确认失败并抛出异常时，应用程序很快就会终止。

除了线程内部确认失败并抛出异常之外，在测试中使用多个线程还有其他问题吗？是的。线程需要被正确管理——也就是说，我们需要确保它们在超出作用域之前要么被连接，要么被分离。你不太可能需要在测试中创建的线程上进行分离，所以你只剩下确保在测试结束时所有在测试中创建的线程都被连接。请注意，我们正在使用的测试手动连接了两个线程。

如果测试有其他确认，那么你需要确保失败的确认不会导致测试跳过线程连接。这是因为留下未连接的测试也会导致应用程序终止。让我们通过将第一个确认放回使用`NotEquals`来避免任何问题，这样它就不会引起任何问题。然后，我们将添加一个新的确认，它将在连接之前失败：

```cpp
    CONFIRM_TRUE(false);
    t1.join();
    t2.join();
    CONFIRM_THAT(count, Equals(0));
```

额外线程内的确认不再引起任何问题。然而，新的`CONFIRM_TRUE`确认将导致跳过连接。结果是另一种终止：

```cpp
------- Test: Test can use additional threads
terminate called without an active exception
Abort trap: 6
```

我们不会做任何事情来帮助解决这种第二种类型的终止问题。你需要确保所有创建的线程都被正确连接。你可能想使用 C++20 中的新功能*jthread*，这将确保线程被连接。或者，你可能只需要小心地将确认放在主测试线程中的位置，以确保所有连接都首先发生。

我们现在可以移除`CONFIRM_TRUE`确认，这样我们就可以专注于修复线程内部确认失败的第一个问题。

我们能做些什么来解决这个问题？我们可以在线程中放置一个 try/catch 块，这至少可以停止终止：

```cpp
TEST("Test can use additional threads")
{
    std::atomic<int> count {0};
    std::thread t([&count]()
    {
        try
        {
            for (int i = 0; i < 100'000; ++i)
            {
                ++count;
            }
            CONFIRM_THAT(count, NotEquals(100'001));
        }
        catch (...)
        { }
    });
    t.join();
    CONFIRM_THAT(count, Equals(100'000));
}
```

为了简化代码，我移除了第二个线程。现在测试使用一个额外的线程来增加计数。线程完成后，`count`应该等于`100'000`。在任何时候，`count`都不应该达到`100'001`，这在线程内部得到了确认。假设我们改变线程内的确认，使其失败：

```cpp
            CONFIRM_THAT(count, Equals(100'001));
```

在这里，异常被捕获，测试正常失败并报告结果。或者不是吗？构建和运行此代码显示所有测试都通过了。线程内的确认检测到不匹配的值，但异常没有方法报告回主测试线程。我们无法在 catch 块中抛出任何内容，因为这只会再次终止应用程序。

我们知道，通过捕获确认异常，我们可以避免测试应用程序终止。而且，我们从第一次线程测试中得知，没有抛出异常的确认也是可以的。我们需要解决的大问题是，如何让主测试线程知道任何已创建的附加线程中的确认失败情况。也许我们可以通过传递给线程的变量在捕获块中通知主线程。

我想强调这一点。如果你在测试中创建线程只是为了分割工作并加快测试速度，而且不需要在线程内进行确认，那么你不需要做任何特殊的事情。你所需要管理的只是正常的线程问题，例如确保在测试结束时连接所有线程，并且没有线程有未处理的异常。唯一需要使用以下指导的原因是当你想在附加线程中放置确认时。

在尝试了几个替代方案后，我提出了以下方案：

```cpp
TEST("Test can use additional threads")
{
    ThreadConfirmException threadEx;
    std::atomic<int> count {0};
    std::thread t([&threadEx, &count]()
    {
        try
        {
            for (int i = 0; i < 100'000; ++i)
            {
                ++count;
            }
            CONFIRM_THAT(count, Equals(100'001));
        }
        catch (ConfirmException const & ex)
        {
            threadEx.setFailure(ex.line(), ex.reason());
        }
    });
    t.join();
    threadEx.checkFailure();
    CONFIRM_THAT(count, Equals(100'000));
}
```

这是 TDD 风格。修改测试，直到你对代码满意，然后让它工作。测试假设有一个新的异常类型叫做`ThreadConfirmException`，并创建了一个名为`threadEx`的本地实例。`threadEx`变量通过引用在线程 lambda 中被捕获，以便线程可以访问`threadEx`。

线程可以使用它想要的任何正常确认，只要一切都在一个带有捕获异常`ConfirmException`类型的 try 块中。如果确认失败，它将抛出一个异常，该异常将被捕获。我们可以使用行号和原因在`threadEx`变量中设置一个失败模式。

一旦线程完成并且我们回到了主线程，我们可以调用另一个方法来检查`threadEx`变量中的失败情况。如果设置了失败，那么`checkFailure`方法应该抛出异常，就像常规确认抛出异常一样。因为我们回到了主测试线程，所以任何抛出的确认异常都将被检测并在测试总结报告中报告。

现在，我们需要在`Test.h`中实现`ThreadConfirmException`类，它可以直接放在`ConfirmException`基类之后，如下所示：

```cpp
class ThreadConfirmException : public ConfirmException
{
public:
    ThreadConfirmException ()
    : ConfirmException(0)
    { }
    void setFailure (int line, std::string_view reason)
    {
        mLine = line;
        mReason = reason;
    }
    void checkFailure () const
    {
        if (mLine != 0)
        {
            throw *this;
        }
    }
};
```

如果我们现在构建并运行，那么线程内的确认将检测到`count`不等于`100'001`，失败将在总结结果中报告，如下所示：

```cpp
------- Test: Test can use additional threads
Failed confirm on line 20
    Expected: 100001
    Actual  : 100000
```

现在的问题是，是否有任何方法可以简化测试？当前的测试看起来是这样的：

```cpp
TEST("Test can use additional threads")
{
    ThreadConfirmException threadEx;
    std::atomic<int> count {0};
    std::thread t([&threadEx, &count]()
    {
        try
        {
            for (int i = 0; i < 100'000; ++i)
            {
                ++count;
            }
            CONFIRM_THAT(count, Equals(100'001));
        }
        catch (ConfirmException const & ex)
        {
            threadEx.setFailure(ex.line(), ex.reason());
        }
    });
    t.join();
    threadEx.checkFailure();
    CONFIRM_THAT(count, Equals(100'000));
}
```

这里，我们有一个新的`ThreadConfirmException`类型，这是好的。然而，测试作者仍然需要将此类型的实例传递给线程函数，类似于`threadEx`被 lambda 捕获的方式。线程函数仍然需要一个 try/catch 块，并在捕获到异常时调用`setFailure`。最后，测试需要在回到主测试线程后检查失败。所有这些步骤都在测试中展示。

我们可能可以使用一些宏来隐藏 try/catch 块，但这看起来很脆弱。测试作者可能会有一些不同的需求。例如，让我们回到两个线程，看看多线程的测试会是什么样子。改变测试，使其看起来像这样：

```cpp
TEST("Test can use additional threads")
{
    std::vector<ThreadConfirmException> threadExs(2);
    std::atomic<int> count {0};
    std::vector<std::thread> threads;
    for (int c = 0; c < 2; ++c)
    {
        threads.emplace_back(
            [&threadEx = threadExs[c], &count]()
        {
            try
            {
                for (int i = 0; i < 100'000; ++i)
                {
                    ++count;
                }
                CONFIRM_THAT(count, Equals(200'001));
            }
            catch (ConfirmException const & ex)
            {
                threadEx.setFailure(ex.line(), ex.reason());
            }
        });
    }
    for (auto & t : threads)
    {
        t.join();
    }
    for (auto const & ex: threadExs)
    {
        ex.checkFailure();
    }
    CONFIRM_THAT(count, Equals(200'000));
}
```

这个测试与该节开头原始的两个线程测试不同。我以不同的方式编写了这个测试，以展示编写多线程测试有很多种方法。因为我们线程内部有更多的代码来处理确认异常，所以我让每个线程都相似。不再是其中一个线程增加计数，而另一个线程减少，现在两个线程都增加。此外，不再为每个线程命名 `t1` 和 `t2`，新的测试将线程放入一个向量中。我们还有一个 `ThreadConfirmException` 向量，每个线程都获得对其自己的 `ThreadConfirmException` 的引用。

关于这个解决方案需要注意的一点是，虽然每个线程都会失败其确认，并且两个 `ThreadConfirmationException` 实例都将有一个失败集，但只会报告一个失败。在测试末尾的循环中，通过 `threadExs` 集合，一旦一个 `ThreadConfirmationException` 失败检查，就会抛出异常。我曾考虑扩展测试库以支持多个失败，但最终决定不增加复杂性。

如果你有一个多线程的测试，那么它们可能会使用不同的数据集。如果恰好发生错误导致多个线程在同一测试运行中失败，那么测试应用程序中只会报告一个失败。修复该失败并再次运行可能会报告下一个失败。逐个修复问题虽然有些繁琐，但不太可能需要增加测试库的复杂性。

新的具有两个线程的测试结构突出了创建可以隐藏所有线程确认处理的合理宏的难度。到目前为止，测试的三个版本都不同。似乎没有一种编写多线程测试的通用方法，我们可以将其封装在某个宏中。我认为我们将坚持我们现在所拥有的——一个可以传递给线程的 `ThreadConfirmException` 类型。线程需要捕获 `ConfirmException` 类型并调用 `setFailure`。主测试线程然后可以检查每个 `ThreadConfirmException`，如果设置了失败，它将抛出异常。在我们继续之前，让我们改变线程 lambda 中的确认，使其测试计数不等于 `200'001`，如下所示：

```cpp
                CONFIRM_THAT(count, NotEquals(200'001));
```

`NotEquals` 确认将允许测试再次通过。

通过本节获得的理解，您将能够编写在测试中使用多个线程的测试。您可以继续使用相同的`CONFIRM`和`CONFIRM_THAT`宏来验证结果。下一节将使用多个线程来记录消息，以确保日志库是线程安全的。您还将了解代码线程安全意味着什么。

# 使日志库线程安全

我们不知道使用日志库的项目是会尝试从多个线程或单个线程进行日志记录。在使用应用程序时，我们完全控制，可以选择使用多个线程或不使用。但是，库，尤其是日志库，通常需要是**线程安全的**。这意味着当应用程序从多个线程使用库时，日志库需要表现良好。使代码线程安全会给代码增加一些额外的开销，如果库只会在单个线程中使用，则不需要这样做。

我们需要的是一个同时从多个线程调用`log`的测试。让我们使用我们现在的代码编写一个测试，看看会发生什么。在本节中，我们将使用日志项目，并在`tests`文件夹中添加一个名为`Thread.cpp`的新文件。添加新文件后的项目结构将如下所示：

```cpp
MereMemo project root folder
    MereTDD folder
        Test.h
    MereMemo folder
        Log.h
        tests folder
            main.cpp
            Construction.cpp
            LogTags.h
            Tags.cpp
            Thread.cpp
            Util.cpp
            Util.h
```

在`Thread.cpp`文件内部，让我们添加一个测试，从多个线程调用`log`函数，如下所示：

```cpp
#include "../Log.h"
#include "Util.h"
#include <MereTDD/Test.h>
#include <thread>
TEST("log can be called from multiple threads")
{
    // We'll have 3 threads with 50 messages each.
    std::vector<std::string> messages;
    for (int i = 0; i < 150; ++i)
    {
        std::string message = std::to_string(i);
        message += " thread-safe message ";
        message += Util::randomString();
        messages.push_back(message);
    }
    std::vector<std::thread> threads;
    for (int c = 0; c < 3; ++c)
    {
        threads.emplace_back(
            [c, &messages]()
        {
            int indexStart = c * 50;
            for (int i = 0; i < 50; ++i)
            {
                MereMemo::log() << messages[indexStart + i];
            }
        });
    }
    for (auto & t : threads)
    {
        t.join();
    }
    for (auto const & message: messages)
    {
        bool result = Util::isTextInFile(message,              "application.log");
        CONFIRM_TRUE(result);
    }
}
```

此测试执行三项操作。首先，它创建`150`条消息。我们将在启动线程之前准备好消息，这样线程就可以尽可能快地多次在循环中调用`log`。

一旦消息准备好，测试将启动`3`个线程，每个线程将记录已经格式化的部分消息。第一个线程将记录消息`0`到`49`。第二个线程将记录消息`50`到`99`。最后，第三个线程将记录消息`100`到`149`。我们在线程中不做任何确认。

一旦所有消息都已记录，并且线程已合并，测试将确认所有`150`条消息都出现在日志文件中。

构建和运行此测试几乎肯定会失败。这种类型的测试违反了第八章中解释的良好的测试的一个要点，即*什么是好的测试？*。这种测试不是最好的类型，因为测试不是完全可重复的。每次运行测试应用程序时，您都会得到一个略有不同的结果。您甚至可能会发现这个测试会导致其他测试失败！

尽管我们不是基于随机数来构建测试的行为，但我们使用了线程。线程调度是不可预测的。使这个测试大部分可靠的方法是记录许多消息，就像我们已经在做的那样。测试会尽其所能设置线程以产生冲突。这就是为什么消息是预格式化的。我希望线程立即进入记录消息的循环，而不是花费额外的时间来格式化消息。

当测试失败时，是因为日志文件混乱。我的一个测试运行中日志文件的一部分看起来像这样：

```cpp
2022-08-16T04:54:54.635 100 thread-safe message 4049
2022-08-16T04:54:54.635 100 thread-safe message 4049
2022-08-16T04:54:54.635 0 thread-safe message 8866
2022-08-16T04:54:54.637 101 thread-safe message 8271
2022-08-16T04:54:54.637 1 thread-safe message 3205
2022-08-16T04:54:54.637 102 thread-safe message 7514
2022-08-16T04:54:54.637 51 thread-safe message 7405
2022-08-16T04:54:54.637 2 thread-safe message 5723
2022-08-16T04:54:54.637 52 thread-safe message 4468
2022-08-16T04:54:54.637 52 thread-safe message 4468
```

我移除了`color`和`log_level`标签，以便你能更好地看到消息。你首先会注意到一些消息是重复的。编号`100`出现了两次，而编号`50`似乎完全缺失。

说实话，我本以为日志文件会比现在更混乱。消息组`0-49`和`50-99`以及`100-149`之间的交错是预期的。我们确实有三个线程同时运行。例如，一旦消息编号`51`被记录，我们应该已经看到了编号`50`。

让我们修复日志代码，以便测试通过。这仍然不会是最好的测试，但如果日志库不是线程安全的，它将有很大的机会找到错误。

修复很简单：我们需要一个互斥锁，然后我们需要锁定互斥锁。首先，让我们在`Log.h`的顶部包含`mutex`标准头文件，如下所示：

```cpp
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
```

然后，我们需要一个地方放置全局互斥锁。由于日志库是一个单头文件，我们无法声明全局变量而不产生链接错误。我们可能能够将全局互斥锁声明为内联。这是 C++中的一个新特性，它允许你声明内联变量，就像我们可以声明内联函数一样。我更习惯于使用静态变量的函数。将以下函数添加到`Log.h`的顶部，紧接在`MereMemo`命名空间的开头之后：

```cpp
inline std::mutex & getLoggingMutex ()
{
    static std::mutex m;
    return m;
}
```

现在，我们需要在适当的位置锁定互斥锁。起初，我在`log`函数中添加了锁定，但没有任何效果。这是因为`log`函数在没有实际记录的情况下返回一个`LogStream`。所以，`log`函数在记录发生之前获得了锁并释放了锁。记录是在`LogStream`析构函数中完成的，所以我们需要在那里放置锁：

```cpp
    ~LogStream ()
    {
        if (not mProceed)
        {
            return;
        }
        const std::lock_guard<std::mutex>               lock(getLoggingMutex());
        auto & outputs = getOutputs();
        for (auto const & output: outputs)
        {
            output->sendLine(this->str());
        }
    }
```

锁尝试获取互斥锁，如果另一个线程已经拥有互斥锁，则会阻塞。一次只有一个线程可以在锁定之后进行，并且在将文本发送到所有输出之后释放锁。

如果我们构建并运行，线程问题将会得到解决。然而，当我运行测试应用程序时，有一个测试失败了。起初，我以为线程仍然存在问题，但失败发生在另一个测试上。这是失败的测试：

```cpp
TEST("Overridden default tag not used to filter messages")
{
    MereTDD::SetupAndTeardown<TempFilterClause> filter;
    MereMemo::addFilterLiteral(filter.id(), info);
    std::string message = "message ";
    message += Util::randomString();
    MereMemo::log(debug) << message;
    bool result = Util::isTextInFile(message,          "application.log");
    CONFIRM_FALSE(result);
}
```

这个测试与多线程测试无关。那么，为什么它失败了？嗯，问题在于这个测试正在确认一个特定的消息不会出现在日志文件中。但是这个消息只是单词 `"message"`，后面跟着一个随机数字字符串。我们刚刚添加了额外的 150 条日志消息，这些消息都有相同的文本，后面跟着一个随机数字字符串。

我们自身在测试上遇到了问题。测试有时会因为随机数而失败。在我们只有少量日志消息时，这个问题并没有被发现，但现在我们有更多机会出现重复的随机数，这个问题就更加明显了。

我们可以增加添加到每个日志消息中的随机数字字符串的大小，或者使测试更加具体，以便它们都使用不同的基消息字符串。

到目前为止，你可能想知道为什么我的测试有一个简单的基消息，而自从日志库首次创建以来，我们一直在每个测试中使用独特的消息*第九章*，*使用测试*。那是因为从*第九章*，*使用测试*开始的代码原本就有简单、通用的日志消息。我本可以将这些通用消息保持原样，并等到现在让你回过头来更改它们。然而，我编辑了这些章节，从开始就解决了问题。现在只是为了更改一个字符串而通过所有测试似乎是一种浪费。因此，我在*第九章*，*使用测试*中添加了说明。现在我们不需要更改任何测试消息，因为它们已经被修复了。

好的，回到多线程的话题——新的测试现在通过了，日志文件中的样本看起来好多了：

```cpp
2022-08-16T06:20:36.807 0 thread-safe message 6269
2022-08-16T06:20:36.807 50 thread-safe message 1809
2022-08-16T06:20:36.807 100 thread-safe message 6297
2022-08-16T06:20:36.808 1 thread-safe message 848
2022-08-16T06:20:36.808 51 thread-safe message 4103
2022-08-16T06:20:36.808 101 thread-safe message 5570
2022-08-16T06:20:36.808 2 thread-safe message 6156
2022-08-16T06:20:36.809 102 thread-safe message 4213
2022-08-16T06:20:36.809 3 thread-safe message 6646
```

再次强调，这个样本已经被修改，以删除`color`和`log_level`标签。这个更改使得每一行都更短，这样你可以更好地看到消息。每个线程中的消息是有序的，即使消息在线程之间混合——也就是说，消息编号`0`在某个时候会被消息编号`1`跟随，然后是编号`2`；消息编号`50`稍后被编号`51`跟随，消息编号`100`被编号`101`跟随。每个后续编号的消息可能不会立即跟随前一个消息。这个样本看起来更好，因为没有重复的消息，也没有缺失的消息。

最后一个想法是关于日志库的线程安全性。我们测试了多个线程可以安全地调用`log`而不必担心问题。但我们没有测试多个线程是否可以管理默认标签或过滤，或者添加新的输出。日志库可能需要更多的工作才能完全线程安全。现在它对我们的目的来说已经足够了。

现在日志库基本上是线程安全的，下一节将回到`SimpleService`项目，开始探索如何测试使用多线程的代码。

# 需要证明多线程的必要性

到目前为止，在本章中，你已经学习了如何编写使用多线程的测试，以及如何使用这些额外的线程来测试日志库。日志库本身并不使用多线程，但我们需要确保日志库在使用多线程时是安全的。

本章的剩余部分将提供一些关于如何测试使用多线程的代码的指导。为了测试多线程代码，我们需要一些使用多线程的代码。为此，我们将使用上一章的`SimpleService`项目。

我们需要修改简单的服务，使其使用多个线程。目前，简单服务是一个问候服务的例子，它根据请求问候的用户进行回复。在问候服务中并不需要太多多线程。我们需要一些不同的东西。

这引出了第一条指导原则：在我们尝试添加多个线程之前，我们需要确保存在一个有效的多线程需求。编写多线程代码很困难，如果只需要一个线程，则应避免使用多线程。如果你只需要一个线程，那么确保遵循上一节的建议，并在代码将被多个线程使用时使其线程安全。

你的目标是尽可能多地编写单线程的代码。如果你能找到一个特定的计算结果的方法，它只需要一些输入数据来得到输出，那么尽可能将其作为单线程计算。如果输入数据量很大，并且可以分割成单独计算的部分，那么将输入分割并传递更小的数据块给计算。保持计算单线程，并专注于处理提供的输入。然后，你可以创建多个线程，每个线程被分配一部分输入数据来计算。这将使你的多线程代码与计算分离。

将你的单线程代码隔离出来，将允许你在无需担心线程管理的情况下设计和测试代码。当然，你可能需要确保代码是线程安全的，但当你只需要担心线程安全时，这会更容易。

由于线程调度的随机性，测试多线程更困难。如果可能，尽量避免使用诸如*睡眠*之类的笨拙方法来协调测试。你想要避免将实际的代码线程置于睡眠状态以协调线程之间的顺序。当一个线程进入睡眠状态时，它会停止运行一段时间，具体取决于在睡眠调用中指定的延迟时间。其他未睡眠的线程可以被调度运行。

我们将在本章设计的代码将允许测试控制线程的同步，这样我们就可以去除随机性并使测试可预测。我们不妨从修改后的服务开始，这个服务有使用多个线程的理由。修改后的`handleRequest`方法如下所示：

```cpp
std::string SimpleService::Service::handleRequest (
    std::string const & user,
    std::string const & path,
    std::string const & request)
{
    MereMemo::log(debug, User(user), LogPath(path))
        << "Received: " << Request(request);
    std::string response;
    if (request == "Calculate")
    {
        response = "token";
    }
    if (request == "Status")
    {
        response = "result";
    }
    else
    {
        response = "Unrecognized request.";
    }
    MereMemo::log(debug, User(user), LogPath(path))
        << "Sending: " << Response(response);
    return response;
}
```

在遵循 TDD（测试驱动开发）时，你通常会先从测试开始。那么，为什么我先向你展示一个修改后的服务呢？因为我们的目标是测试多线程代码。在你的项目中，你应该避免在没有充分理由的情况下使用某些技术。我们的理由是需要一个可以学习的例子。因此，我们是从反向需求开始使用多线程的。

我试图想出一个问候服务使用多个线程的好理由，但想不出来。所以，我们将服务改为稍微复杂一些的东西；在我们开始编写测试之前，我想解释这个新想法。

新服务仍然尽可能简单。我们将继续忽略所有网络和消息路由。我们需要将请求和响应类型更改为结构体，并且我们还将继续忽略将数据结构序列化以传输到和从服务中。

新服务将模拟一个难题的计算。创建新线程的一个有效理由是让新线程执行一些工作，而原始线程继续它正在做的事情。新服务的设计理念是`Calculate`请求可能需要很长时间才能完成，我们不希望调用者在等待结果时超时。因此，服务将创建一个新线程来执行计算，并立即向调用者返回一个令牌。调用者可以使用这个令牌以不同的`Status`请求回调到服务，这将检查刚刚开始的计算进度。如果计算尚未完成，则`Status`请求的响应将让调用者知道大约完成了多少。如果计算已完成，则响应将包含答案。

我们现在有理由使用多个线程并可以编写一些测试。让我们处理一个本应已经添加的无关测试。我们想要确保任何使用未识别请求调用服务的人都会收到一个未识别的响应。将以下测试放入`SimpleService`项目的`tests`文件夹中的`Message.cpp`文件中：

```cpp
TEST_SUITE("Unrecognized request is handled properly", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    std::string request = "Hello";
    std::string expectedResponse = "Unrecognized request.";
    std::string response = gService1.service().handleRequest(
        user, path, request);
    CONFIRM_THAT(response, Equals(expectedResponse));
}
```

我将这个测试放在`Message.cpp`的顶部。它所做的只是发送之前的问候请求，但期望得到一个未识别的响应。

让我们还在`SetupTeardown.cpp`中将测试套件的名称更改为`"Calculation Service"`，如下所示：

```cpp
MereTDD::TestSuiteSetupAndTeardown<ServiceSetup>
gService1("Calculation Service", "Service 1");
```

现在，让我们删除问候测试并添加以下简单测试，以确保我们得到除未识别响应之外的其他响应：

```cpp
TEST_SUITE("Calculate request can be sent and recognized", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    std::string request = "Calculate";
    std::string unexpectedResponse = "Unrecognized request.";
    std::string response = gService1.service().handleRequest(
        user, path, request);
    CONFIRM_THAT(response, NotEquals(unexpectedResponse));
}
```

这个测试与未识别测试相反，确保响应不是未识别的。通常，确认结果符合预期发生的事情，而不是确认结果不是你不想发生的事情，会更好。双重否定不仅更难思考，而且可能导致问题，因为不可能捕捉到所有可能出错的方式。通过确认你想要发生的事情，你可以消除所有可能的错误条件，这些条件太多，无法单独捕捉。

这个测试有一点不同。我们并不关心响应。这个测试的目的是确认请求已被识别。确认响应不是未识别的，即使这看起来与刚刚描述的双重否定陷阱相似，也是合适的。

构建和运行此代码显示，未识别的测试通过了，但`Calculate`请求失败了：

```cpp
Running 1 test suites
--------------- Suite: Service 1
------- Setup: Calculation Service
Passed
------- Test: Unrecognized request is handled properly
Passed
------- Test: Calculate request can be sent and recognized
Failed confirm on line 30
    Expected: not Unrecognized request.
    Actual  : Unrecognized request.
------- Teardown: Calculation Service
Passed
-----------------------------------
Tests passed: 3
Tests failed: 1
```

看起来，对于应该有效的请求，我们得到了一个未识别的响应。这就是在项目开始时添加简单测试的价值所在。测试有助于立即捕捉到简单的错误。问题是出在`handleRequest`方法中。我通过复制第一次检查添加了第二次对有效请求的检查，却忘记了将`if`语句更改为`else if`语句。修复方法如下：

```cpp
    if (request == "Calculate")
    {
        response = "token";
    }
    else if (request == "Status")
    {
        response = "result";
    }
    else
    {
        response = "Unrecognized request.";
    }
```

为了进一步进行，我们将发送和接收不仅仅是字符串。当我们发送一个`Calculate`请求时，我们应该得到一个可以传递给`Status`请求的令牌值。然后`Status`响应应该包含答案或进度估计。让我们一步一步来，定义`Calculate`请求和响应结构。将以下两个结构定义添加到`Service.h`文件中的`SimpleService`命名空间顶部：

```cpp
struct CalculateRequest
{
    int mSeed;
};
struct CalculateResponse
{
    std::string mToken;
};
```

这将允许我们传递一些初始值进行计算；作为回报，我们将得到一个可以用来最终获取答案的令牌。但我们有一个问题。如果将`Calculate`请求更改为返回结构体，那么这将破坏现有的测试，因为测试期望得到一个字符串。我们应该改变测试，让它们使用结构体，但这又带来了另一个问题：大多数时候，我们需要返回正确的响应结构体。并且我们需要为错误情况返回错误响应。

我们需要的是可以代表良好响应和错误响应的响应。既然我们将有一个可以服务于多个目的的响应，为什么不让它也处理`Status`响应的结构体呢？这意味着我们将有一个单一的响应类型，它可以是一个错误响应、计算响应或状态响应。既然我们有一个多用途的响应类型，为什么不创建一个多用途的请求类型呢？让我们改变一下测试。

我们将使用 `std::variant` 来存储不同类型的请求和响应。我们可以移除发送了无效请求字符串的测试。我们仍然可能会收到无效请求，但这仅发生在调用者和服务之间的服务版本不匹配。这稍微复杂一些，所以我们暂时忽略服务可能对请求可用性的理解与实际服务知识不一致的情况。如果你正在编写一个真实的服务，那么这是一个需要解决和测试的可能性。你可能还想使用不同于变体的其他东西。一个好的选择可能是类似于谷歌的 *Protocol Buffers*，其中服务将接受 Protocol Buffer 消息。虽然使用 Protocol Buffers 比简单的结构体更好，但其设计也更加复杂，这将使解释变得更加冗长。

在 `Message.cpp` 中，我们将有一个单独的测试，其外观如下：

```cpp
TEST_SUITE("Calculate request can be sent", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    SimpleService::RequestVar request =
        SimpleService::CalculateRequest {
            .mSeed = 5
        };
    std::string emptyResponse = "";
    std::string response = gService1.service().handleRequest(
        user, path, request);
    CONFIRM_THAT(response, NotEquals(emptyResponse));
}
```

此测试首先关注请求类型，并将响应类型留为字符串。我们将逐步进行更改。这对于使用 `std::variant` 尤其是当你不熟悉变体时非常有用。我们将有一个名为 `RequestVar` 的变体类型，它可以被初始化为特定的请求类型。我们使用 `CalculateRequest` 初始化请求，并使用 *指定初始化器* 语法设置 `mSeed` 值。指定初始化器语法在 C++ 中相对较新，它允许我们通过在数据成员名称前放置一个点来根据名称设置数据成员的值。

现在，让我们在 `Service.h` 中定义请求类型：

```cpp
#ifndef SIMPLESERVICE_SERVICE_H
#define SIMPLESERVICE_SERVICE_H
#include <string>
#include <variant>
namespace SimpleService
{
struct CalculateRequest
{
    int mSeed;
};
struct StatusRequest
{
    std::string mToken;
};
using RequestVar = std::variant<
    CalculateRequest,
    StatusRequest
    >;
```

注意，我们需要包含标准 `variant` 头文件。`RequestVar` 类型现在只能是 `CalculateRequest` 或 `StatusRequest` 之一。我们还需要在 `Service.h` 中的 `Service` 类的 `handleRequest` 方法中进行一个额外的更改：

```cpp
class Service
{
public:
    void start ();
    std::string handleRequest (std::string const & user,
        std::string const & path,
        RequestVar const & request);
};
```

需要更改 `Service.cpp` 文件，以便更新 `handleRequest` 方法，如下所示：

```cpp
std::string SimpleService::Service::handleRequest (
    std::string const & user,
    std::string const & path,
    RequestVar const & request)
{
    std::string response;
    if (auto const * req = std::get_       if<CalculateRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Calculate request for: "
            << std::to_string(req->mSeed);
        response = "token";
    }
    else if (auto const * req = std::get_            if<StatusRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Status request for: "
            << req->mToken;
        response = "result";
    }
    else
    {
        response = "Unrecognized request.";
    }
    MereMemo::log(debug, User(user), LogPath(path))
        << "Sending: " << Response(response);
    return response;
}
```

更新的 `handleRequest` 方法继续检查未知请求类型。所有响应都是字符串，需要更改。我们目前还没有查看种子或令牌值，但我们已经有了足够的内容可以构建和测试。

现在单个测试通过后，在下一节中，我们将查看响应并使用结构体而不是响应字符串。

# 更改服务返回类型

我们将在本节中进行类似的更改，以摆脱字符串并使用结构体来处理服务请求。上一节更改了服务请求类型；本节将更改服务返回类型。我们需要进行这些更改，以便将服务提升到能够支持额外线程需求的功能水平。

我们使用的`SimpleService`项目最初是一个问候服务，我无法想出任何理由说明这样一个简单的服务需要另一个线程。我们在上一节中开始将服务调整为计算服务；现在，我们需要修改服务在处理请求时返回的返回类型。

首先，让我们在`Service.h`中定义返回类型结构体，它紧随请求类型之后。将以下代码添加到`Service.h`中：

```cpp
struct ErrorResponse
{
    std::string mReason;
};
struct CalculateResponse
{
    std::string mToken;
};
struct StatusResponse
{
    bool mComplete;
    int mProgress;
    int mResult;
};
using ResponseVar = std::variant<
    ErrorResponse,
    CalculateResponse,
    StatusResponse
    >;
```

这些结构和变体遵循与请求相同的模式。一个小差异是，我们现在有一个`ErrorResponse`类型，它将用于任何错误。我们可以修改`Message.cpp`中的测试，使其看起来像这样：

```cpp
TEST_SUITE("Calculate request can be sent", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    SimpleService::RequestVar request =
        SimpleService::CalculateRequest {
            .mSeed = 5
        };
    auto const responseVar = gService1.service().handleRequest(
        user, path, request);
    auto const response =
        std::get_if<SimpleService::CalculateResponse>(&responseVar);
    CONFIRM_TRUE(response != nullptr);
}
```

这个测试将像之前一样调用服务，使用计算请求；返回的响应将被测试以确认它是否是计算响应。

为了使代码能够编译，我们需要更改`Service.h`中的`handleRequest`声明，使其返回新的类型，如下所示：

```cpp
class Service
{
public:
    void start ();

    ResponseVar handleRequest (std::string const & user,
        std::string const & path,
        RequestVar const & request);
};
```

然后，我们需要更改`Service.cpp`中`handleRequest`的实现：

```cpp
SimpleService::ResponseVar SimpleService::Service::handleRequest (
    std::string const & user,
    std::string const & path,
    RequestVar const & request)
{
    ResponseVar response;
    if (auto const * req = std::get_       if<CalculateRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Calculate request for: "
            << std::to_string(req->mSeed);
        response = SimpleService::CalculateResponse {
            .mToken = "token"
        };
    }
    else if (auto const * req = std::get_            if<StatusRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Status request for: "
            << req->mToken;
        response = SimpleService::StatusResponse {
            .mComplete = false,
            .mProgress = 25,
            .mResult = 0
        };
    }
    else
    {
        response = SimpleService::ErrorResponse {
            .mReason = "Unrecognized request."
        };
    }
    return response;
}
```

代码变得越来越复杂。我在返回前移除了日志记录，之前它是用来记录响应的。我们可以把日志记录放回去，但这需要将`ResponseVar`转换为字符串的能力。或者，我们可以在多个地方记录响应，就像代码中对请求所做的那样。这是一个我们可以跳过的细节。

新的`handleRequest`方法几乎与之前所做的一样，只是现在它初始化一个`ResponseVar`类型而不是返回一个字符串。这允许我们在返回请求和错误时，提供比之前更详细的信息。

要添加一个测试来识别未知的请求，我们需要在`RequestVar`中添加一个新的请求类型，但在`handleRequest`方法内的`if`语句中忽略这个新的请求类型。我们也将跳过这个测试，因为我们真的应该使用除了`std::variant`之外的其他东西。

我们在这个例子中使用`std::variant`的唯一原因是为了避免额外的复杂性。我们试图使代码准备好支持另一个线程。

在下一节中，我们将添加一个使用两种请求类型的测试。第一个请求将开始计算，而第二个请求将在计算完成时检查计算状态并获取结果。

# 进行多次服务调用

如果你正在考虑使用多线程来加速计算，那么我建议你在承担多线程的额外复杂性之前，先使用单线程测试并确保代码能够正常工作。

对于我们正在工作的服务，添加第二个线程的原因不是为了提高任何东西的速度。我们需要避免一个可能需要很长时间的计算超时。我们将添加的额外线程不是为了使计算更快。一旦我们使用一个额外的线程使计算工作，我们就可以考虑添加更多线程来加快计算速度。

在原始线程继续做其他事情的同时创建一个线程来执行一些工作是常见的。这不是应该在以后进行的优化。这是设计的一部分，并且应该从一开始就包含额外的线程。

让我们从向 `Message.cpp` 添加一个新测试开始，这个测试看起来是这样的：

```cpp
TEST_SUITE("Status request generates result", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    SimpleService::RequestVar calcRequest =
        SimpleService::CalculateRequest {
            .mSeed = 5
        };
    auto responseVar = gService1.service().handleRequest(
        user, path, calcRequest);
    auto const calcResponse =
        std::get_if<SimpleService::CalculateResponse>        (&responseVar);
    CONFIRM_TRUE(calcResponse != nullptr);
    SimpleService::RequestVar statusRequest =
        SimpleService::StatusRequest {
            .mToken = calcResponse->mToken
        };
    int result {0};
    for (int i = 0; i < 5; ++i)
    {
        responseVar = gService1.service().handleRequest(
            user, path, statusRequest);
        auto const statusResponse =
            std::get_if<SimpleService::StatusResponse>            (&responseVar);
        CONFIRM_TRUE(statusResponse != nullptr);
        if (statusResponse->mComplete)
        {
            result = statusResponse->mResult;
            break;
        }
    }
    CONFIRM_THAT(result, Equals(50));
}
```

所有代码都已经就绪，以便这个新测试可以编译。现在，我们可以运行测试以查看会发生什么。测试将失败，如下所示：

```cpp
Running 1 test suites
--------------- Suite: Service 1
------- Setup: Calculation Service
Passed
------- Test: Calculate request can be sent
Passed
------- Test: Status request generates result
Failed confirm on line 62
    Expected: 50
    Actual  : 0
------- Teardown: Calculation Service
Passed
-----------------------------------
Tests passed: 3
Tests failed: 1
```

这个测试做什么？首先，它创建一个计算请求 l 并获取一个硬编码的令牌值。在服务开始时还没有进行计算，所以当我们用令牌发出状态请求时，服务会响应一个硬编码的响应，表示计算尚未完成。测试正在寻找一个表示计算已完成的状态响应。测试尝试进行五次状态请求然后放弃，这导致测试结束时的确认失败，因为我们没有得到预期的结果。请注意，即使尝试多次也不是最好的做法。线程是不可预测的，你的电脑可能在服务完成请求之前就尝试了所有五次。如果你的测试继续失败，你可能需要增加尝试的次数，或者等待一段合理的时间。我们的计算最终会将种子乘以 `10`。所以，当我们给出初始种子 `5` 时，我们应该期望最终结果为 `50`。

我们需要在服务中实现计算和状态请求处理，这样我们就可以使用一个线程来使测试通过。我们首先需要做的是在 `Service.cpp` 的顶部包含 `mutex`、`thread` 和 `vector`。我们还需要添加一个无名的命名空间，如下所示：

```cpp
#include "Service.h"
#include "LogTags.h"
#include <MereMemo/Log.h>
#include <mutex>
#include <thread>
#include <vector>
namespace
{
}
```

我们将需要一些锁定机制，这样我们就不在状态被线程更新时尝试读取计算状态。为了进行同步，我们将使用互斥锁和锁，就像我们在日志库中做的那样。你可能还想探索其他设计，例如为不同的计算请求分别锁定数据。我们将采用简单的方法，并为所有内容使用单个锁。在无名的命名空间内添加以下函数：

```cpp
    std::mutex & getCalcMutex ()
    {
        static std::mutex m;
        return m;
    }
```

我们需要某种东西来跟踪每个计算请求的完成状态、进度和结果。我们将在无名的命名空间内创建一个类来保存这些信息，称为 `CalcRecord`，就在 `getCalcMutex` 函数之后，如下所示：

```cpp
    class CalcRecord
    {
    public:
        CalcRecord ()
        { }
        CalcRecord (CalcRecord const & src)
        {
            const std::lock_guard<std::mutex>                   lock(getCalcMutex());
            mComplete = src.mComplete;
            mProgress = src.mProgress;
            mResult = src.mResult;
        }
        void getData (bool & complete, int & progress, int &                      result)
        {
            const std::lock_guard<std::mutex>                   lock(getCalcMutex());
            complete = mComplete;
            progress = mProgress;
            result = mResult;
        }
        void setData (bool complete, int progress, int result)
        {
            const std::lock_guard<std::mutex>                   lock(getCalcMutex());
            mComplete = complete;
            mProgress = progress;
            mResult = result;
        }
        CalcRecord &
        operator = (CalcRecord const & rhs) = delete;
    private:
        bool mComplete {false};
        int mProgress {0};
        int mResult {0};
    };
```

看起来这个类还有很多其他的功能，但它相当简单。默认构造函数不需要做任何事情，因为数据成员已经定义了它们的默认值。我们需要默认构造函数的唯一原因是我们还有一个拷贝构造函数。而我们需要拷贝构造函数的唯一原因是为了在复制数据成员之前锁定互斥锁。

然后，我们有一个方法可以一次性获取所有数据成员，还有一个方法可以设置数据成员。获取器和设置器在继续之前都需要获取锁。

没有必要将一个`CalcRecord`赋值给另一个，因此已经删除了赋值运算符。

在未命名的命名空间中，我们还需要一个`CalcRecord`的向量，如下所示：

```cpp
    std::vector<CalcRecord> calculations;
```

每次有计算请求时，我们都会将一个`CalcRecord`添加到`calculations`集合中。一个真正的服务会希望清理或重用`CalcRecord`条目。

我们需要修改`Service.cpp`中的请求处理，以便每次收到计算请求时都创建一个线程来使用一个新的`CalcRecord`，如下所示：

```cpp
    if (auto const * req = std::get_       if<CalculateRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Calculate request for: "
            << std::to_string(req->mSeed);
        calculations.emplace_back();
        int calcIndex = calculations.size() - 1;
        std::thread calcThread([calcIndex] ()
        {
            calculations[calcIndex].setData(true, 100, 50);
        });
        calcThread.detach();
        response = SimpleService::CalculateResponse {
            .mToken = std::to_string(calcIndex)
        };
    }
```

当我们收到一个计算请求时会发生什么？首先，我们在`calculations`向量的末尾添加一个新的`CalcRecord`。我们将使用`CalcRecord`的索引作为响应中返回的令牌。这是我能够想到的识别计算请求的最简单设计。一个真正的服务会希望使用一个更安全的令牌。然后，请求处理器启动一个线程来进行计算，并从线程中分离出来。

你将要编写的绝大多数线程代码都会创建一个线程然后加入该线程。创建一个线程然后从线程中分离出来并不常见。作为替代，当你想要做一些工作而不必担心加入线程时，你可以使用线程池。分离线程的原因是我想要一个最简单的例子，而不引入线程池。

线程本身非常简单，因为它立即将`CalcRecord`设置为完成，进度为`100`，结果为`50`。

我们现在可以构建并运行测试应用程序了，但我们会得到之前相同的失败。那是因为状态请求处理仍然返回硬编码的响应。我们需要像这样修改请求处理器来处理状态请求：

```cpp
    else if (auto const * req = std::get_            if<StatusRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Status request for: "
            << req->mToken;
        int calcIndex = std::stoi(req->mToken);
        bool complete;
        int progress;
        int result;
        calculations[calcIndex].getData(complete, progress,                                 result);
        response = SimpleService::StatusResponse {
            .mComplete = complete,
            .mProgress = progress,
            .mResult = result
        };
    }
```

通过这个更改，状态请求将令牌转换为它用来查找正确`CalcRecord`的索引。然后，它从`CalcRecord`获取当前数据，并将其作为响应返回。

你可能还想要考虑在尝试五次服务调用请求的测试循环中添加睡眠，以便给服务提供合理的时间。如果所有五次尝试都在服务完成甚至一个简单的计算之前快速完成，当前的测试将会失败。

在构建和运行测试应用程序后，所有测试都通过了。我们现在就完成了吗？还没有。所有这些更改都让服务能够在单独的线程中计算结果，同时继续在主线程上处理请求。添加另一个线程的整个目的是为了避免由于长时间计算导致的超时。但我们的计算非常快。我们需要减慢计算速度，以便我们可以以合理的响应时间测试服务。

我们如何减慢线程的运行？以及计算需要多少时间来完成？这些问题是我们在本章中编写代码来回答的。下一节将解释如何测试使用多个线程的服务。现在我们有一个使用另一个线程进行计算的服务，我们可以探索测试这种情况的最佳方法。

我还想澄清，下一节所做的是与在五个服务调用尝试中添加延迟不同。在测试循环中添加延迟将提高我们目前测试的可靠性。下一节将完全删除循环，并展示如何与其他线程协调测试，以便测试和线程可以一起进行。

# 如何在不使用 sleep 的情况下测试多个线程

在本章的早期，在*需要证明多线程的必要性*部分，我提到你应该尽量使用单线程完成尽可能多的工作。我们现在将遵循这个建议。在当前的计算请求处理中，代码创建了一个执行简单计算的线程，如下所示：

```cpp
        std::thread calcThread([calcIndex] ()
        {
            calculations[calcIndex].setData(true, 100, 50);
        });
```

好吧，也许简单计算不是描述线程所做事情的正确方式。线程将结果设置为硬编码的值。我们知道这是临时代码，我们需要将代码更改为将种子值乘以`10`，这正是测试所期望的。

计算应该在何处进行？在线程 lambda 中进行计算很容易，但这将违反尽量使用单线程完成尽可能多的工作的建议。

我们想要做的是创建一个线程可以调用的计算函数。这将使我们能够单独测试计算函数，而不必担心任何线程问题，并确保计算是正确的。

这里有一个真正有趣的部分：创建一个执行计算的函数将帮助我们测试线程管理！如何？因为我们将创建两个计算函数。

一个函数将是真正的计算函数，可以独立于任何线程进行测试。对于我们的项目，真正的计算仍然简单且快速。我们不会尝试做很多工作来减慢计算，也不会让线程休眠。我们也不会编写大量测试来确保计算正确。这只是一个你可以遵循的项目模式示例。

另一个函数将是一个测试计算函数，它将执行一些旨在匹配真实计算结果的假计算。测试计算函数还将包含一些线程管理代码，用于协调线程的活动。我们将使用测试计算函数中的线程管理代码来减慢线程速度，以便模拟耗时较长的计算。

我们所做的是用代码模拟真实计算，这些代码更关注线程的行为而非计算本身。任何想要测试真实计算的测试都可以使用真实计算函数，而任何想要测试线程定时和协调的测试都可以使用测试计算函数。

首先，我们将在 `Service.h` 中声明这两个函数，位于 `Service` 类之前，如下所示：

```cpp
void normalCalc (int seed, int & progress, int & result);
void testCalc (int seed, int & progress, int & result);
```

您可以在项目中定义自己的计算函数以执行所需的任何操作。您的函数可能不同。需要理解的主要点是它们应该具有相同的签名，以便测试函数可以替换真实函数。

`Service` 类需要修改，以便可以将这些函数之一注入到服务中。我们将在构造函数中设置计算函数，并使用真实函数作为默认值，如下所示：

```cpp
class Service
{
public:
    using CalcFunc = void (*) (int, int &, int &);
    Service (CalcFunc f = normalCalc)
    : mCalc(f)
    { }
    void start ();
    ResponseVar handleRequest (std::string const & user,
        std::string const & path,
        RequestVar const & request);
private:
    CalcFunc mCalc;
};
```

`Service` 类现在有一个成员函数指针，它将指向其中一个计算函数。具体调用哪个函数是在创建 `Service` 类时确定的。

让我们按照如下方式实现这两个函数在 `Service.cpp` 中：

```cpp
void SimpleService::normalCalc (
    int seed, int & progress, int & result)
{
    progress = 100;
    result = seed * 10;
}
void SimpleService::testCalc (
    int seed, int & progress, int & result)
{
    progress = 100;
    result = seed * 10;
}
```

目前，这两个函数是相同的。我们将一步一步来。每个函数只是将 `progress` 设置为 `100`，将 `result` 设置为 `seed` 乘以 `10`。我们将保持真实或正常函数不变。最终，我们将修改测试函数，使其控制线程。

现在，我们可以更改 `Service.cpp` 中的计算请求处理程序，使其使用计算函数，如下所示：

```cpp
    if (auto const * req = std::get_       if<CalculateRequest>(&request))
    {
        MereMemo::log(debug, User(user), LogPath(path))
            << "Received Calculate request for: "
            << std::to_string(req->mSeed);
        calculations.emplace_back();
        int calcIndex = calculations.size() - 1;
        int seed = req->mSeed;
        std::thread calcThread([this, calcIndex, seed] ()
        {
            int progress;
            int result;
            mCalc(seed, progress, result);
            calculations[calcIndex].setData(true, progress,                                     result);
        });
        calcThread.detach();
        response = SimpleService::CalculateResponse {
            .mToken = std::to_string(calcIndex)
        };
    }
```

在线程 lambda 中，我们调用 `mCalc` 而不是将 `progress` 和 `result` 设置为硬编码的值。调用哪个计算函数取决于 `mCalc` 指向哪个函数。

如果我们构建并运行测试应用程序，我们会看到测试通过。但我们在调用 `mCalc` 方式上存在问题。我们希望获取中间进度，以便调用者可以发出状态请求并看到进度增加，直到计算最终完成。通过一次调用 `mCalc`，我们只给函数一次做事情的机会。我们应该在 `progress` 达到 `100` 百分比之前循环调用 `mCalc` 函数。让我们更改 lambda 代码：

```cpp
        std::thread calcThread([this, calcIndex, seed] ()
        {
            int progress {0};
            int result {0};
            while (true)
            {
                mCalc(seed, progress, result);
                if (progress == 100)
                {
                    calculations[calcIndex].setData(true,                     progress, result);
                    break;
                }
                else
                {
                    calculations[calcIndex].setData(false,                     progress, result);
                }
            }
        });
```

此更改不会影响测试，因为当前的`mCalc`函数在第一次调用时将`progress`设置为`100`；因此，while 循环只会运行一次。我们不希望线程在没有与测试同步的情况下运行得太久，因为我们永远不会与线程连接。如果这是一个真实的项目，我们希望使用线程池中的线程，并在停止服务之前等待线程完成。

对测试不产生影响的更改是一种很好的验证更改的方法。采取小步骤，而不是试图在一次巨大的更改集中完成所有事情。

接下来，我们将复制生成结果的测试，但我们将使用复制测试中的测试计算函数。测试需要稍作修改，以便可以使用测试计算函数。但大部分测试应该几乎保持不变。新测试放在`Message.cpp`中，如下所示：

```cpp
TEST_SUITE("Status request to test service generates result", "Service 2")
{
    std::string user = "123";
    std::string path = "";
    SimpleService::RequestVar calcRequest =
        SimpleService::CalculateRequest {
            .mSeed = 5
        };
    auto responseVar = gService2.service().handleRequest(
        user, path, calcRequest);
    auto const calcResponse =
        std::get_if<SimpleService::CalculateResponse>        (&responseVar);
    CONFIRM_TRUE(calcResponse != nullptr);
    SimpleService::RequestVar statusRequest =
        SimpleService::StatusRequest {
            .mToken = calcResponse->mToken
        };
    int result {0};
    for (int i = 0; i < 5; ++i)
    {
        responseVar = gService2.service().handleRequest(
            user, path, statusRequest);
        auto const statusResponse =
            std::get_if<SimpleService::StatusResponse>            (&responseVar);
        CONFIRM_TRUE(statusResponse != nullptr);
        if (statusResponse->mComplete)
        {
            result = statusResponse->mResult;
            break;
        }
    }
    CONFIRM_THAT(result, Equals(40));
}
```

唯一的更改是给测试一个不同的名称，以便它使用一个名为`"Service 2"`的新测试套件，然后使用一个不同的全局服务`gService2`。在这里，我们期望得到略微不同的结果。我们很快就会更改这个测试，使其最终比现在更有价值，并且我们会移除尝试进行五次请求的循环。分步骤进行这些更改将使我们能够验证我们没有破坏任何主要的东西。并且期望得到略微不同的结果将使我们能够验证我们是否使用了不同的计算函数。

要构建项目，我们需要定义`gService2`，它将使用一个新的设置和销毁类。将以下代码添加到`SetupTeardown.h`中：

```cpp
class TestServiceSetup
{
public:
    TestServiceSetup ()
    : mService(SimpleService::testCalc)
    { }
    void setup ()
    {
        mService.start();
    }
    void teardown ()
    {
    }
    SimpleService::Service & service ()
    {
        return mService;
    }
private:
    SimpleService::Service mService;
};
extern MereTDD::TestSuiteSetupAndTeardown<TestServiceSetup>
gService2;
```

`TestServiceSetup`类定义了一个构造函数，该构造函数使用`testCalc`函数初始化`mService`数据成员。`gService2`声明使用`TestServiceSetup`。我们需要在`SetupTeardown.cpp`中对`gService2`进行一些小的更改，如下所示：

```cpp
#include "SetupTeardown.h"
MereTDD::TestSuiteSetupAndTeardown<ServiceSetup>
gService1("Calculation Service", "Service 1");
MereTDD::TestSuiteSetupAndTeardown<TestServiceSetup>
gService2("Calculation Test Service", "Service 2");
```

`SetupTeardown.cpp`文件很短，只需要定义`gService1`和`gService2`的实例。

我们需要修改`testCalc`函数，使其乘以`8`后得到预期的结果`40`而不是`50`。以下是`Service.cpp`中的两个计算函数：

```cpp
void SimpleService::normalCalc (
    int seed, int & progress, int & result)
{
    progress = 100;
    result = seed * 10;
}
void SimpleService::testCalc (
    int seed, int & progress, int & result)
{
    progress = 100;
    result = seed * 8;
}
```

构建和运行测试应用程序显示所有测试都通过了。我们现在有两个测试套件。输出如下：

```cpp
Running 2 test suites
--------------- Suite: Service 1
------- Setup: Calculation Service
Passed
------- Test: Calculate request can be sent
Passed
------- Test: Status request generates result
Passed
------- Teardown: Calculation Service
Passed
--------------- Suite: Service 2
------- Setup: Calculation Test Service
Passed
------- Test: Status request to test service generates result
Passed
------- Teardown: Calculation Test Service
Passed
-----------------------------------
Tests passed: 7
Tests failed: 0
```

在这里，我们引入了一个使用略微不同的计算函数的新服务，并且可以在测试中使用这两个服务。测试通过，且仅进行了最小改动。现在，我们准备进行更多更改以协调线程。这种方法比直接跳入线程管理代码并添加新服务和计算函数要好。

在遵循 TDD（测试驱动开发）时，过程始终相同：让测试通过，对测试进行小改动或添加新测试，然后再次让测试通过。

下一步将完成这一部分。我们将控制`testCalc`函数的工作速度，以便我们可以进行多次状态请求以获得完整的结果。我们将在测试计算函数内部等待，以便测试有时间验证进度确实随着时间的推移而增加，直到进度达到 100%时最终计算出结果。

让我们从测试开始。我们将在测试线程内部向计算线程发送信号，以便计算线程能够与测试同步进行。这就是我不使用睡眠来测试多个线程的意思。在线程内部睡眠不是一个好的解决方案，因为它不可靠。你可能能够通过测试，但后来当时间变化时，同样的测试可能会失败。这里你将学到的解决方案可以应用于你的测试。

你需要做的只是创建你代码的一部分的测试版本，它可以替换真实代码。在我们的例子中，我们有一个`testCalc`函数可以替换`normalCalc`函数。然后，你可以在测试中添加一个或多个*条件变量*，并在你的代码的测试版本中等待这些条件变量。条件变量是 C++中一个标准且受支持的方式，允许一个线程在满足条件之前等待。测试计算函数将等待条件变量。当测试准备好继续计算时，它将通知条件变量。通知条件变量将在正确的时间解除等待的计算线程，以便测试可以验证适当的线程行为。然后，测试将等待计算完成后再继续。我们需要在`Service.h`的顶部包含`condition_variable`，如下所示：

```cpp
#ifndef SIMPLESERVICE_SERVICE_H
#define SIMPLESERVICE_SERVICE_H
#include <condition_variable>
#include <string>
#include <variant>
```

然后，我们需要在`Service.h`中声明一个互斥锁、两个条件变量和两个布尔值，以便它们可以被测试计算函数和测试使用。让我们在测试计算函数之前声明互斥锁、条件变量和布尔值，如下所示：

```cpp
void normalCalc (int seed, int & progress, int & result);
extern std::mutex service2Mutex;
extern std::condition_variable testCalcCV;
extern std::condition_variable testCV;
extern bool testCalcReady;
extern bool testReady;
void testCalc (int seed, int & progress, int & result);
```

这里是修改后的`Message.cpp`测试代码：

```cpp
TEST_SUITE("Status request to test service generates result", "Service 2")
{
    std::string user = "123";
    std::string path = "";
    SimpleService::RequestVar calcRequest =
        SimpleService::CalculateRequest {
            .mSeed = 5
        };
    auto responseVar = gService2.service().handleRequest(
        user, path, calcRequest);
    auto const calcResponse =
        std::get_if<SimpleService::CalculateResponse>        (&responseVar);
    CONFIRM_TRUE(calcResponse != nullptr);
    // Make a status request right away before the service
    // is allowed to do any calculations.
    SimpleService::RequestVar statusRequest =
        SimpleService::StatusRequest {
            .mToken = calcResponse->mToken
        };
    responseVar = gService2.service().handleRequest(
        user, path, statusRequest);
    auto statusResponse =
        std::get_if<SimpleService::StatusResponse>        (&responseVar);
    CONFIRM_TRUE(statusResponse != nullptr);
    CONFIRM_FALSE(statusResponse->mComplete);
    CONFIRM_THAT(statusResponse->mProgress, Equals(0));
    CONFIRM_THAT(statusResponse->mResult, Equals(0));
    // Notify the service that the test has completed the first
    // confirmation so that the service can proceed with the
    // calculation.
    {
        std::lock_guard<std::mutex>              lock(SimpleService::service2Mutex);
        SimpleService::testReady = true;
    }
    SimpleService::testCV.notify_one();
    // Now wait until the service has completed the calculation.
    {
        std::unique_lock<std::mutex>              lock(SimpleService::service2Mutex);
        SimpleService::testCalcCV.wait(lock, []
        {
            return SimpleService::testCalcReady;
        });
    }
    // Make another status request to get the completed result.
    responseVar = gService2.service().handleRequest(
        user, path, statusRequest);
    statusResponse =
        std::get_if<SimpleService::StatusResponse>        (&responseVar);
    CONFIRM_TRUE(statusResponse != nullptr);
    CONFIRM_TRUE(statusResponse->mComplete);
    CONFIRM_THAT(statusResponse->mProgress, Equals(100));
    CONFIRM_THAT(statusResponse->mResult, Equals(40));
}
```

测试比以前要长一些。我们不再在寻找完成响应的同时在循环中发出状态请求。这个测试采取了一种更谨慎的方法，并且确切地知道每个步骤的期望结果。初始的计算请求和计算响应是相同的。测试知道计算将被暂停，因此第一个状态请求将返回一个未完成的响应，进度为零。

在第一次状态请求被确认后，测试会通知计算线程可以继续，然后测试等待。一旦计算完成，计算线程将通知测试可以继续。在所有时候，测试和计算线程都在轮流进行，这样测试可以确认每一步。测试计算线程中存在一个小小的竞争条件，我会在你看到代码后解释。竞争条件是指两个或多个线程可能会相互干扰，导致结果不可完全预测的问题。

现在我们来看另一半——测试计算函数。我们需要声明互斥锁、条件变量以及布尔值。变量和测试计算函数应该看起来像这样：

```cpp
std::mutex SimpleService::service2Mutex;
std::condition_variable SimpleService::testCalcCV;
std::condition_variable SimpleService::testCV;
bool SimpleService::testCalcReady {false};
bool SimpleService::testReady {false};
void SimpleService::testCalc (
    int seed, int & progress, int & result)
{
    // Wait until the test has completed the first status request.
    {
        std::unique_lock<std::mutex> lock(service2Mutex);
        testCV.wait(lock, []
        {
            return testReady;
        });
    }
    progress = 100;
    result = seed * 8;
    // Notify the test that the calculation is ready.
    {
        std::lock_guard<std::mutex> lock(service2Mutex);
        testCalcReady = true;
    }
    testCalcCV.notify_one();
}
```

测试计算函数的第一件事是等待。除非测试有机会确认初始状态，否则不会进行任何计算进度。一旦允许测试计算线程继续，它需要在返回之前通知测试，以便测试可以再次进行状态请求。

理解这个过程的最重要的地方是，测试计算函数应该是唯一与测试交互的代码。你不应该在主服务响应处理器中，甚至是在响应处理器中定义的 lambda 中放置任何等待或通知。只有替换为实际计算函数的测试计算函数应该有测试正在运行的任何意识。换句话说，你应该将所有的等待和条件变量通知放在`testCalc`中。这就是我提到的竞争条件的来源。当`testCalc`函数通知测试线程计算已完成时，这并不完全正确。只有当`setData`完成更新`CalcRecord`时，计算才算完成。然而，我们不想在调用`setData`后发送通知，因为这会将通知放在`testCalc`函数之外。

理想情况下，我们会在计算完成后调用计算函数一次额外的次数。我们可以说这给了计算函数一个清理计算期间使用的任何资源的机会。或者也许我们可以创建另一组用于清理的函数。一个清理函数可以是正常的清理，而另一个函数可以是用于测试清理的替代品。任何一种方法都可以让我们通知测试计算已完成，这将消除竞争条件。

构建和运行这些测试表明所有测试仍然通过。我们几乎完成了。我们将保持竞争条件不变，因为修复它只会给这个解释增加额外的复杂性。唯一剩下的任务是在日志文件中修复我注意到的问题。我将在下一节中解释这个新问题的更多内容。

# 修复最后通过日志检测到的问题

我选择在本书的*第二部分*，*日志库*中构建日志库有一个很大的原因。日志记录在调试已知问题时可以提供巨大的帮助。常常被忽视的是，日志记录在寻找尚未检测到的错误时提供的益处。

我通常会在运行测试后查看日志文件，以确保消息与我预期的相符。在上一节中对测试和测试计算线程之间的线程协调进行了增强后，我在日志文件中注意到了一些奇怪的现象。日志文件看起来是这样的：

```cpp
2022-08-27T05:00:50.409 Service is starting.
2022-08-27T05:00:50.410 user="123" Received Calculate request for: 5
2022-08-27T05:00:50.411 user="123" Received Calculate request for: 5
2022-08-27T05:00:50.411 user="123" Received Status request for: 1
2022-08-27T05:00:50.411 Service is starting.
2022-08-27T05:00:50.411 Service is starting.
2022-08-27T05:00:50.411 user="123" Received Calculate request for: 5
2022-08-27T05:00:50.411 user="123" Received Calculate request for: 5
2022-08-27T05:00:50.411 user="123" Received Status request for: 2
2022-08-27T05:00:50.411 user="123" Received Status request for: 2
2022-08-27T05:00:50.411 user="123" Received Status request for: 2
2022-08-27T05:00:50.411 user="123" Received Status request for: 2
```

我移除了`log_level`和`logpath`标签，只是为了缩短消息，以便你能更好地看到重要部分。我首先注意到的一个奇怪现象是服务启动了三次。我们只有`gService1`和`gService2`，所以服务应该只启动了两次。

日志文件的前四行是有意义的。我们启动`gService1`，然后运行一个简单的测试，请求一个计算并检查响应是否为正确的类型。然后，我们运行另一个测试，在寻找完整响应的同时，最多进行五次状态请求。第一次状态请求找到了完整的响应，因此不需要额外的状态请求。第一次状态请求的令牌是`1`。

日志文件的第 5 行，即第二次启动服务的地方，是日志文件开始看起来奇怪的地方。我们只需要启动第二个服务，进行一次额外的请求，然后进行两次状态请求。看起来日志文件从第 5 行到结尾都在接收重复的消息。

经过一点调试和提示我们正在重复日志消息后，我发现问题所在。当我最初设计该服务时，我在`Service::start`方法中配置了日志记录。我应该将日志配置保留在`main`函数中。一切正常，直到我们需要创建并启动第二个服务，以便第二个服务可以配置为使用测试计算函数。嗯，第二个服务在启动时也配置了日志，并添加了另一个文件输出。第二个文件的输出导致所有日志消息被发送到日志文件两次。解决方案很简单：我们需要像这样在`main`中配置日志：

```cpp
#include <MereMemo/Log.h>
#include <MereTDD/Test.h>
#include <iostream>
int main ()
{
    MereMemo::FileOutput appFile("logs");
    MereMemo::addLogOutput(appFile);
    return MereTDD::runTests(std::cout);
}
```

然后，我们需要从服务的`start`方法中移除日志配置，使其看起来像这样：

```cpp
void SimpleService::Service::start ()
{
    MereMemo::log(info) << "Service is starting.";
}
```

通过这些更改，测试仍然通过，日志文件看起来更好。再次，我移除了一些标签以缩短日志消息行。现在，日志文件的内容如下：

```cpp
2022-08-27T05:35:30.573 Service is starting.
2022-08-27T05:35:30.574 user="123" Received Calculate request for: 5
2022-08-27T05:35:30.574 user="123" Received Calculate request for: 5
2022-08-27T05:35:30.574 user="123" Received Status request for: 1
2022-08-27T05:35:30.574 Service is starting.
2022-08-27T05:35:30.574 user="123" Received Calculate request for: 5
2022-08-27T05:35:30.574 user="123" Received Status request for: 2
2022-08-27T05:35:30.575 user="123" Received Status request for: 2
```

虽然问题最终是日志配置错误导致的，但我想要强调的是，提醒你定期查看日志文件，确保日志消息是有意义的。

# 摘要

这是本书的最后一章，它解释了编写软件中最令人困惑和难以理解的一个方面：如何测试多线程。你会发现很多书籍解释了多线程，但很少会给你提供建议并展示有效测试多线程的方法。

由于本书的目标客户是希望学习如何使用 TDD 来设计更好软件的微服务 C++开发者，因此本章将本书中的所有内容串联起来，解释如何测试多线程服务。

首先，你学会了如何在测试中使用多个线程。你需要确保你处理在启动额外线程的测试中出现的异常。异常很重要，因为测试库使用异常来处理失败的确认。你还学会了如何使用一个特殊的辅助类来报告在额外线程中出现的失败确认。

在编写和使用库时，也必须考虑线程。你看到了如何测试库以确保它是线程安全的。

最后，你学会了如何以快速和可靠的方式测试多线程服务，避免了在尝试协调多个线程的动作时让线程休眠。你学会了如何重构你的代码，以便尽可能地在单线程模式下进行测试，然后如何用特殊的测试感知代码替换正常代码，这些代码与测试一起工作。当你需要测试和多线程代码一起工作时，你可以使用这种技术，以便测试可以采取具体和可靠的步骤，并在过程中确认你的期望。

恭喜你完成了这本书的阅读！本章回顾了我们一直在工作的所有项目。我们增强了单元测试库，帮助你可以在测试中使用多个线程。我们还使日志库线程安全。最后，我们增强了服务，使其能够在服务和测试之间协调多个线程。你现在拥有了将 TDD 应用到你的项目中所需要的所有技能。
