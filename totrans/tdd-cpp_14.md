# 14

# 如何测试服务

我们已经积累到这个阶段，可以使用测试库和日志库在另一个项目中。日志库的客户始终是使用 TDD 设计更好服务的微服务 C++ 开发者。

由于专注于服务，本章将介绍一个模拟微服务的项目。我们不会包括真实服务所需的所有内容。例如，真实的服务需要网络、路由和排队请求的能力以及处理超时。我们的服务将只包含启动服务和处理请求的核心方法。

你将了解测试服务所涉及到的挑战，以及测试服务与测试试图做所有事情的应用程序的不同之处。本章将较少关注服务的设计。我们也不会编写所有需要的测试。实际上，本章只使用了一个测试。还提到了可以添加的其他测试。

我们还将探讨在服务中可以测试的内容，以及一些提示和指导，这将使你能够在调试服务时控制生成的日志量。

服务项目将帮助将测试和日志库结合起来，并展示如何在你的项目中使用这两个库。

本章的主要内容包括以下几项：

+   服务测试挑战

+   在服务中可以测试什么？

+   介绍 SimpleService 项目

# 技术要求

本章中所有代码都使用标准 C++，它基于任何现代 C++ 20 或更高版本的编译器和标准库。代码引入了一个新的服务项目，该项目使用了本书第一部分“*测试 MVP*”中的测试库，并使用了本书第二部分“*日志库*”中的日志库。

你可以在以下 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

)

# 服务测试挑战

在本书中我们一直在思考的客户是一位使用 C++ 编写服务的微服务开发者，他希望更好地理解 TDD 以改进开发过程并提高代码质量。TDD 适用于任何编写代码的人。但为了遵循 TDD，你需要清楚地了解你的客户是谁，这样你才能从客户的角度编写测试。

与测试一个自己完成所有事情的应用程序相比，在测试服务时会有不同的挑战。通常将包含所有内容的程序称为*单体应用程序*。适用于服务的挑战示例包括：

+   服务是否可达？

+   服务是否正在运行？

+   服务是否因其他请求而超载？

+   是否有任何权限或安全检查可能影响你调用服务的能力？

然而，在我们深入探讨之前，我们需要了解什么是服务以及为什么你应该关心。

服务独立运行并接收请求，处理请求，并为每个请求返回某种类型的响应。服务专注于请求和响应，这使得它们更容易编写和调试。你不必担心其他代码以意想不到的方式与你的服务交互，因为请求和响应完全定义了交互。如果你的服务开始收到过多的请求，你总是可以添加更多服务实例来处理额外的负载。当服务专注于处理几个特定的请求时，它们被称为*微服务*。当你能够将工作分解为微服务时，构建大型和复杂解决方案变得更加容易和可靠。

服务也可以向其他服务发出请求以处理请求。这就是微服务如何相互构建以形成更大服务的方式。在每一步，请求和预期响应都是清晰和明确定义的。也许你的整个解决方案完全由服务组成。但更有可能的是，你将有一个客户运行的应用程序，该应用程序接受客户的输入和指示，并向各种服务发出请求以满足客户的需求。也许客户打开一个应用程序窗口，显示基于客户提供的一些日期的信息图表。为了获取显示图表所需的数据，应用程序会将日期作为请求发送到提供数据的服务的服务。该服务甚至可以根据发出请求的具体客户定制数据。

想象一下，如果编写一个试图自己完成所有工作的应用程序会有多困难。开发工作量可能会从难以想象的复杂单体应用程序转变为使用服务时的合理工作量。当任务可以隔离和独立开发、管理时，质量也会提高。

服务通常运行在多台计算机上，因此请求和响应是通过网络进行的。也可能涉及其他路由代码，它接受请求并在将其发送到服务之前将其放入队列。服务可能运行在多台计算机上，路由器将确定哪个服务最能处理请求。

如果你足够幸运，拥有一个庞大且设计良好的服务网络，那么你可能会拥有多个独立的网络，旨在帮助你测试和部署服务。每个网络可以拥有许多不同的计算机，每台计算机可以运行多个不同的服务。这就是路由器变得非常有用的地方。

测试运行在多个网络中的服务通常涉及在为早期测试设计的网络中的一台计算机上部署要测试的服务的新版本。这个网络通常被称为*开发环境*。

如果开发环境中的测试失败，那么你有时间找到错误，进行更改，并测试新版本，直到服务按预期运行。查找错误涉及查看响应以确保它们是正确的，检查日志文件以确保在过程中采取了正确的步骤，并查看任何其他输出，例如在处理请求时可能被修改的数据库条目。根据服务，你可能还有其他需要检查的事项。

一些服务依赖于存储在数据库中的数据来正确响应请求。在开发环境中保持数据库的当前状态可能很困难，这就是为什么通常需要其他环境。如果在开发环境中初始测试通过，那么你可能会将服务更改部署到*测试环境*并再次测试。最终，你将把服务部署到*生产环境*，在那里它将为客户提供服务。

如果你可以控制请求的路由，那么在测试你的更改时运行调试器可能是可能的。这样做的方法是在调试器下启动特定计算机上的服务。通常，这只会发生在开发环境中。然后，你需要确保通过测试用户账户发出的任何请求都路由到运行调试器的计算机。同一服务（没有你的最近更改）可能正在同一环境中的其他计算机上运行，这就是为什么只有当你能确保请求将被路由到你所使用的计算机时，使用调试器进行调试才有效。

如果你没有能力将请求路由到特定的计算机，或者如果你在一个不允许调试器的环境中进行测试，那么你将不得不严重依赖日志消息。有时你事先不知道环境中哪台计算机将处理请求，因此你需要将你的服务部署到该环境中的所有计算机上。

检查日志文件可能很繁琐，因为你需要访问每一台计算机来打开日志文件，看看你的测试请求是否在该计算机上处理，或者在其他计算机上处理。如果你有一个从每台计算机收集日志文件并使日志消息可供搜索的服务，那么你将在具有多台计算机的环境中测试你的服务时遇到许多便利。

当测试不使用服务的单个应用程序时，你不会遇到相同的分布式测试问题。你甚至可以使用自己的计算机进行大部分测试。你可以在调试器下运行你的更改，检查日志文件，并快速直接地运行单元测试。服务需要更多的支持，例如你可能无法在自己的计算机上设置的消息路由基础设施。

每个使用微服务构建解决方案的公司和组织都会有不同的环境和部署步骤。我无法告诉你如何测试你特定的服务。这也不是本节的目标。我只是在解释测试服务时遇到的挑战，这些挑战与测试试图做所有事情的应用程序不同。

即使有所有的额外网络和路由，服务仍然是一个设计大型应用程序的好方法。谁知道呢，路由甚至可能本身就是一个服务。有了所有隔离和独立的服务，我们就可以通过小步骤添加新功能和升级用户体验，而不是发布一个包含所有功能的新版本。

对于小型应用程序，使用服务可能不值得额外的开销。但我看到很多小型应用程序成长为大型应用程序，然后在复杂性变得过高时陷入困境。同样的事情也发生在服务和它们所使用的语言上。我看到一些服务最初非常小，可以用几行 Python 代码编写。开发者可能面临紧迫的截止日期，用 Python 编写小型服务比用 C++编写同样的服务要快。最终，这个小型服务被证明对其他团队有价值，并在使用和功能上增长。它继续增长，直到需要用用 C++编写的服务来替换它。

现在你对测试服务的挑战有了更多了解，下一节将探讨可以测试的内容。

# 在服务中可以测试什么？

服务是由接受的请求和返回的响应定义的。服务还将有一个地址或某种将请求路由到服务的方法。可能有一个版本号，或者版本可能包含在地址中。

当把这些东西放在一起时，首先需要准备一个请求并将请求发送到服务。响应可能一次性到来，也可能分批到来。或者，响应可能像一张可以在以后向同一服务或不同服务出示的票证，以获取实际响应。

所有这些都意味着与服务的交互方式有很多种。唯一保持不变的是请求和响应的基本概念。如果你发出一个请求，不能保证服务会收到这个请求。如果服务回复，也不能保证响应会返回给原始请求者。处理超时始终是与服务一起工作时的一大关注点。

你可能不会想直接测试超时，因为这可能需要从 30 秒到 5 分钟的时间，服务请求才会因为无响应而被终止。但你可能想测试在预期和合理的时间内的响应时间。然而，对于这类测试要小心，因为它们有时会通过，有时会失败，这取决于许多可能变化且不受测试直接控制的因素。超时测试更像是一种压力测试或验收测试，尽管它可能在服务部署后帮助识别不良设计，但最初关注超时通常不是 TDD 的正确选择。

相反，将服务视为你将使用 TDD 进行设计的任何其他软件。明确了解客户是谁以及他们的需求是什么，然后提出一个既合理、易于使用又易于理解的需求和响应。

在测试服务时，可能只需确保响应包含正确的信息就足够了。这可能适用于完全不受你控制的服务。但对于服务来说，保持与任何调用代码完全分离，并通过请求和响应进行交互可能是有用的。

可能你正在调用由不同公司创建的服务，而响应是获取所需信息的唯一方式。如果是这样，那么你为什么要测试这个服务呢？请记住，只测试你的代码。

假设这是你正在设计和测试的服务，并且响应完全包含了所需的信息，那么你可以编写只需要形成请求并检查响应的测试。

有时，一个请求可能会产生一个仅仅确认请求的响应。实际请求的结果可能出现在其他地方。在这种情况下，你需要编写形成请求、验证响应，并在任何地方验证实际结果的测试。比如说，你正在设计一个允许调用者请求删除文件的服务。请求将包含有关文件的信息。响应可能只是确认文件已被删除。然后测试可能需要检查文件曾经所在文件夹，以确保文件不再可用。

通常，要求服务执行某些操作的请求将需要验证该操作是否真的被执行。而要求服务计算某些内容或返回某些内容的请求可能能够在响应中直接确认信息。如果请求的信息真的很大，那么找到其他方式返回信息可能更好。

无论你如何设计你的服务，主要的一点是存在许多选项。在编写测试以创建设计时，你需要考虑你的服务将如何被使用。

你甚至可能需要在测试中调用两个或更多的服务。例如，如果你正在编写一个旨在用具有较慢计算响应时间的老旧服务替换的服务，你可能想调用这两个服务并比较返回的信息，以确保新服务仍然返回与旧服务相同的信息。

服务在请求格式化和路由以及响应解释方面涉及很多开销。测试服务并不像简单地调用一个函数。

然而，在某个内部点上，一个服务将包含一个用于处理或处理请求的函数。这个函数通常不会对服务的用户公开。用户必须通过服务接口进行操作，这涉及到通过网络连接路由请求和响应。

由于 C++ 尚未具备标准网络功能，这些功能可能在 C++23 中出现，我们将跳过所有网络和官方请求和响应定义。我们将创建一个类似真实服务内部结构的简单服务。

我们还将关注那种可以在响应中完全返回信息的请求类型。下一节将介绍该服务。

# 介绍 SimpleService 项目

我们将在本节中开始一个新的项目来构建一个服务。就像日志项目使用测试项目一样，这个服务项目将使用测试项目。服务将更进一步，并使用日志项目。这个服务不会是一个真正的服务，因为一个完整的服务需要大量的非标准 C++ 支持代码，这将使我们进入与学习 TDD 无关的主题。

该服务将被命名为 `SimpleService`，初始文件集将结合本书中已经解释的许多主题。以下是项目结构：

```cpp
SimpleService project root folder
    MereTDD folder
        Test.h
    MereMemo folder
        Log.h
    SimpleService folder
        tests folder
            main.cpp
            Message.cpp
            SetupTeardown.cpp
            SetupTeardown.h
        LogTags.h
        Service.cpp
        Service.h
```

当我开始这个项目时，我不知道需要哪些文件。我知道这个项目将使用 `MereTDD` 和 `MereMemo`，并且会有一个用于服务的独立文件夹。在 `SimpleService` 文件夹内，我知道将有一个包含 `main.cpp` 的 `tests` 文件夹。我猜测还会有 `Service.h` 和 `Service.cpp`。我还为第一个测试添加了一个名为 `Message.cpp` 的文件。第一个测试的想法是发送一个请求并接收一个响应。

因此，让我们从我知道的项目文件开始。`Test.h` 和 `Log.h` 是我们在这本书中迄今为止一直在开发的相同文件，而 `main.cpp` 文件看起来也类似，如下所示：

```cpp
#include <MereTDD/Test.h>
#include <iostream>
int main ()
{
    return MereTDD::runTests(std::cout);
}
```

`main.cpp` 文件实际上比以前简单一些。我们不需要使用任何默认的日志标签，因此不需要包含任何关于日志的内容。我们只需要包含测试库并运行测试。

我编写的第一个测试放在了 `Message.cpp` 中，看起来如下：

```cpp
#include "../Service.h"
#include <MereTDD/Test.h>
using namespace MereTDD;
TEST("Request can be sent and response received")
{
    std::string user = "123";
    std::string path = "";
    std::string request = "Hello";
    std::string expectedResponse = "Hi, " + user;
    SimpleService::Service service;
    service.start();
    std::string response = service.handleRequest(
        user, path, request);
    CONFIRM_THAT(response, Equals(expectedResponse));
}
```

当时我的想法是，会有一个名为 `Service` 的类可以被构造和启动。一旦服务启动，我们可以调用一个名为 `handleRequest` 的方法，该方法需要一个用户 ID、服务路径和请求。`handleRequest` 方法将返回响应，这将是一个字符串。

请求也将是一个字符串，我决定使用一个简单的问候服务。请求将是 `"Hello"` 字符串，响应将是 `"Hi, "` 后跟用户 ID。我在测试中添加了 Hamcrest 风格的响应确认。

我意识到我们最终还需要其他测试，并且其他测试应该使用已经启动的服务。重用已经运行的服务比每次运行测试时创建服务实例并启动服务要好。因此，我将 `Message.cpp` 文件更改为使用具有设置和清理的测试套件，如下所示：

```cpp
#include "../Service.h"
#include "SetupTeardown.h"
#include <MereTDD/Test.h>
using namespace MereTDD;
TEST_SUITE("Request can be sent and response received", "Service 1")
{
    std::string user = "123";
    std::string path = "";
    std::string request = "Hello";
    std::string expectedResponse = "Hi, " + user;
    std::string response = gService1.service().handleRequest(
        user, path, request);
    CONFIRM_THAT(response, Equals(expectedResponse));
}
```

本章我们将要添加到服务中的唯一测试。它将足够发送一个请求并获取一个响应。

我将 `SetupTeardown.h` 和 `SetupTeardown.cpp` 文件添加到 `tests` 文件夹中。头文件看起来是这样的：

```cpp
#ifndef SIMPLESERVICE_TESTS_SUITES_H
#define SIMPLESERVICE_TESTS_SUITES_H
#include "../Service.h"
#include <MereMemo/Log.h>
#include <MereTDD/Test.h>
class ServiceSetup
{
public:
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
extern MereTDD::TestSuiteSetupAndTeardown<ServiceSetup>
gService1;
#endif // SIMPLESERVICE_TESTS_SUITES_H
```

这个文件包含的内容你在这本书中都已经见过。除了我们之前在单个测试 `.cpp` 文件中声明了设置和清理类。这是我们第一次需要在头文件中声明设置和清理，以便以后在其他测试文件中重用。你可以看到 `setup` 方法调用了服务的 `start` 方法。唯一的真正区别是全局实例 `gService1` 需要声明为 `extern`，这样我们就不会在其他使用相同设置和清理代码的测试文件中遇到链接错误。

`SetupTeardown.cpp` 文件看起来是这样的：

```cpp
#include "SetupTeardown.h"
MereTDD::TestSuiteSetupAndTeardown<ServiceSetup>
gService1("Greeting Service", "Service 1");
```

这只是头文件中声明的 `extern` 的 `gService1` 实例。套件名称 `"Service 1"` 需要与 `Message.cpp` 中的 `TEST_SUITE` 宏使用的套件名称相匹配。

接下来，我们看看 `Service.h` 中的 `Service` 类声明，它看起来是这样的：

```cpp
#ifndef SIMPLESERVICE_SERVICE_H
#define SIMPLESERVICE_SERVICE_H
#include <string>
namespace SimpleService
{
class Service
{
public:
    void start ();
    std::string handleRequest (std::string const & user,
        std::string const & path,
        std::string const & request);
};
} // namespace SimpleService
#endif // SIMPLESERVICE_SERVICE_H
```

我将服务代码放在了 `SimpleService` 命名空间中，你可以在原始测试和设置和清理代码中看到。`start` 方法不需要参数，并返回空值。至少现在是这样。我们总是可以在以后增强服务。我觉得从一开始就包括启动服务的想法很重要，即使目前没有太多的事情要做。一个服务已经运行并等待处理请求的想法是定义服务是什么的核心概念。

另一个方法是 `handleRequest` 方法。我们跳过了真实服务的一些细节，例如请求和响应的定义。真实的服务将会有一种文档化的方式来定义请求和响应，几乎就像一种编程语言本身。我们只是将请求和响应都使用字符串。

真实的服务会使用身份验证和授权来验证用户以及每个用户可以使用服务做什么。我们只是简单地将字符串用作`user`身份标识。

并且一些服务有一个称为*服务路径*的概念。路径不是服务的地址。在编程术语中，路径就像调用栈。通常，当应用程序调用服务时，路由器会开始路径。`path`参数充当调用本身的唯一标识符。如果服务需要调用其他服务来处理请求，那么这些附加服务请求的路由器将添加到已经启动的初始`path`中。每次`path`增长时，路由器都会在`path`的末尾添加另一个唯一标识符。`path`可以在服务中用于记录消息。

`path`的整个目的是为了让开发者能够通过关联和排序特定请求的日志消息来理解日志消息。记住，服务一直在处理来自不同用户的请求。调用其他服务会导致这些服务记录它们自己的活动。拥有一个可以识别单个服务请求及其所有相关服务调用的`path`，即使跨越多个日志文件，在调试时也非常有帮助。

服务的实现位于`Service.cpp`中，内容如下：

```cpp
#include "Service.h"
#include "LogTags.h"
#include <MereMemo/Log.h>
void SimpleService::Service::start ()
{
    MereMemo::FileOutput appFile("logs");
    MereMemo::addLogOutput(appFile);
    MereMemo::log(info) << "Service is starting.";
}
std::string SimpleService::Service::handleRequest (
    std::string const & user,
    std::string const & path,
    std::string const & request)
{
    MereMemo::log(debug, User(user), LogPath(path))
        << "Received: " << Request(request);
    std::string response;
    if (request == "Hello")
    {
        response = "Hi, " + user;
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

一些关于 TDD 的书籍和指南会说，对于一个初始测试来说，代码太多，不应该有任何日志记录或检查`request`字符串，实际上，第一个实现应该返回一个空字符串，以便测试失败。

然后，响应应该硬编码为测试期望的确切值。然后应该创建另一个使用不同`user` ID 的测试。只有在这种情况下，`response`才应该通过查看传递给`handleRequest`方法的`user` ID 来构建。

在创建更多通过不同`request`字符串的测试之后，检查`request`与已知值应该放在后面。我相信你明白了这个意思。

虽然我喜欢遵循步骤，但我认为更倾向于编写一些额外的代码，这样 TDD 过程就不会过于繁琐。这个初始服务仍然做得很少。添加日志和一些初始结构到代码有助于为后续内容打下基础。至少这是我的观点。

对于日志记录，你会在`log`调用中注意到一些像`User(user)`这样的内容。这些是自定义日志标签，就像我们在*第十章*中构建的标签一样，*深入理解 TDD 过程*。所有自定义标签都在最后一个项目文件`LogTags.h`中定义，其内容如下：

```cpp
#ifndef SIMPLESERVICE_LOGTAGS_H
#define SIMPLESERVICE_LOGTAGS_H
#include <MereMemo/Log.h>
namespace SimpleService
{
inline MereMemo::LogLevel error("error");
inline MereMemo::LogLevel info("info");
inline MereMemo::LogLevel debug("debug");
class User : public MereMemo::StringTagType<User>
{
public:
    static constexpr char key[] = "user";
    User (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
class LogPath : public MereMemo::StringTagType<LogPath>
{
public:
    static constexpr char key[] = "logpath";
    LogPath (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
class Request : public MereMemo::StringTagType<Request>
{
public:
    static constexpr char key[] = "request";
    Request (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
class Response : public MereMemo::StringTagType<Response>
{
public:
    static constexpr char key[] = "response";
    Response (std::string const & value,
        MereMemo::TagOperation operation =
            MereMemo::TagOperation::None)
    : StringTagType(value, operation)
    { }
};
} // namespace SimpleService
#endif // SIMPLESERVICE_LOGTAGS_H
```

此文件定义了自定义的`User`、`LogPath`、`Request`和`Response`标签。命名的日志级别`error`、`info`和`debug`也被定义。所有标签都放置在`Service`类相同的`SimpleService`命名空间中。

注意，日志项目还包括一个名为`LogTags.h`的文件，它被放在`tests`文件夹中，因为我们正在测试日志本身。对于这个服务项目，`LogTags.h`文件在服务文件夹中，因为标签是服务的一部分。我们不再测试标签是否工作。我们甚至不再测试日志是否工作。标签作为正常服务操作的一部分被记录，因此它们现在是服务项目的一部分。

一切准备就绪后，我们可以构建和运行测试项目，这表明单个测试通过。总结报告实际上显示由于设置和拆卸，有三个测试通过。报告看起来像这样：

```cpp
Running 1 test suites
--------------- Suite: Service 1
------- Setup: Service 1
Passed
------- Test: Message can be sent and received
Passed
------- Teardown: Service 1
Passed
-----------------------------------
Tests passed: 3
Tests failed: 0
```

我们还可以查看包含这些消息的日志文件：

```cpp
2022-08-14T05:58:13.543 log_level="info" Service is starting.
2022-08-14T05:58:13.545 log_level="debug" logpath="" user="123" Received: request="Hello"
2022-08-14T05:58:13.545 log_level="debug" logpath="" user="123" Sending: response="Hi, 123"
```

现在，我们可以看到构成服务的核心结构。服务首先启动并准备好处理请求。当一个请求到达时，请求被记录，进行处理以生成响应，然后记录响应，再将其发送回调用者。

我们使用测试库通过跳过所有网络和路由，直接到服务以启动服务并处理请求来模拟真实服务。

我们现在不会添加更多测试。但对你自己的服务项目来说，那将是你的下一步。如果你的服务支持多种不同的请求，你将为每种请求类型添加一个测试。别忘了添加一个未识别请求的测试。

每种请求类型可能对不同组合的请求参数有多个测试。记住，在真实服务中的一个真实请求将有能力定义丰富和复杂的请求，其中请求可以指定自己的参数集，就像函数可以定义自己的参数一样。

每种请求类型通常都有自己的响应类型。你可能有一个用于错误的通用响应类型。或者，每种响应类型都需要包含错误信息的字段。如果你的响应类型用于成功响应，并且任何错误响应都返回你定义的标准错误响应类型，这可能更容易。

在测试服务时，另一个好主意是为每种请求类型创建一个日志标签。我们只有一个问候请求，但想象一下一个可以处理多种不同请求的服务。如果每个日志消息都标记了请求类型，那么仅启用一种类型的请求的调试日志就变得容易了。

目前，我们正在使用用户 ID 标记日志消息。这是在不使日志文件因过多的日志消息而溢出的情况下启用调试级别日志的另一种绝佳方法。我们可以设置一个过滤器来为特定的测试用户 ID 记录调试日志条目。我们还需要设置一个默认过滤器为`info`。然后我们可以将用户 ID 与请求类型结合起来，以获得更精确的结果。一旦设置好过滤器，正常请求将以 info 级别记录，而测试用户将针对特定请求类型记录所有内容。

# 摘要

提供写作服务需要大量的支持代码和网络连接，而这些是单体应用所不需要的。服务的部署和管理也更加复杂。那么，为什么有人会设计一个使用服务而不是将所有内容都放入单一单体应用中的解决方案呢？因为服务可以帮助简化应用程序的设计，尤其是对于非常大的应用程序。而且因为服务运行在分布式计算机上，你可以扩展解决方案并提高可靠性。使用服务发布更改和新功能也变得更加容易，因为每个服务都可以独立进行测试和更新。你不必测试一个巨大的应用程序并一次性发布所有内容。

本章探讨了服务的一些不同测试挑战以及可以测试的内容。你被介绍了一个简单的服务，它跳过了路由和网络连接，直接进入服务的核心：启动服务并处理请求的能力。

本章开发的一个简单服务将测试和日志库结合起来，这两个库都在服务中使用。在设计你自己的需要使用这两个库的项目时，你可以遵循类似的项目结构。

下一章将探讨在测试中使用多个线程的困难。我们将测试日志库以确保它是线程安全的，了解线程安全意味着什么，并探讨如何使用多个线程测试服务。
