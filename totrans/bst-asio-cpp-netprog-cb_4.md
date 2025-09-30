# 第四章 实现服务器应用程序

在本章中，我们将涵盖以下主题：

+   实现一个同步迭代 TCP 服务器

+   实现一个同步并行 TCP 服务器

+   实现一个异步 TCP 服务器

# 简介

**服务器**是分布式应用的一部分，它提供其他应用部分消费的服务或服务——**客户端**。客户端通过与服务器的通信来消费它提供的服务。

通常，服务器应用程序在客户端-服务器通信过程中扮演被动角色。在启动期间，服务器应用程序连接到主机上的一个特定已知端口（这意味着，它对潜在客户端来说是已知的，或者客户端至少可以在运行时从某个已知注册表中获取它）。之后，它被动地等待来自客户端到达该端口的请求。当请求到达时，服务器通过执行其提供服务的规范来处理它（提供服务）。

根据特定服务器提供的服务，请求处理可能意味着不同的事情。例如，一个 HTTP 服务器通常会读取请求消息中指定的文件内容并将其发送回客户端。一个代理服务器会简单地将客户端的请求重定向到另一个服务器进行实际处理（或者可能是另一轮重定向）。其他更具体的服务器可能提供对客户端在请求中提供的数据进行复杂计算的服务，并将此类计算的结果返回给客户端。

并非所有服务器都扮演被动角色。一些服务器应用程序可能会在没有等待客户端首先发送请求的情况下向客户端发送消息。通常，这样的服务器充当*通知者*，并*通知*客户端一些有趣的事件。在这种情况下，客户端可能根本不需要向服务器发送任何数据。相反，它们被动地等待来自服务器的通知，并在收到通知后相应地做出反应。这种通信模型被称为*推送式通信*。这种模型在现代网络应用中越来越受欢迎，提供了额外的灵活性。

因此，对服务器应用程序的第一种分类方式是按照它们执行的功能（或功能）或提供给客户端的服务（或服务）。

另一个明显的分类维度是服务器用于与客户端通信的传输层协议。

TCP 协议在当今非常流行，许多通用服务器应用程序都使用它进行通信。其他更具体的服务器可能使用 UDP 协议。同时通过 TCP 和 UDP 协议提供服务的混合服务器应用程序属于第三类，被称为**多协议服务器**。在本章中，我们将考虑几种类型的 TCP 服务器。

服务器的一个特点是它服务客户端的方式。一个**迭代服务器**逐个服务客户端，这意味着它不会在完成当前正在服务的客户端之前开始服务下一个客户端。一个**并行服务器**可以并行服务多个客户端。在单处理器计算机上，并行服务器会交错进行与多个客户端的通信的不同阶段，同时在单个处理器上运行它们。例如，在连接到一个客户端并等待其请求消息的同时，服务器可以切换到连接第二个客户端，或者从第三个客户端读取请求；之后，它可以切换回第一个客户端继续为其服务。这种并行性被称为伪并行性，因为处理器只是在几个客户端之间切换，但并不真正地同时服务它们，这在单处理器上是无法实现的。

在多处理器计算机上，当服务器同时使用不同的硬件线程为每个客户端服务多个客户端时，可以实现真正的并行性。

迭代服务器相对简单易实现，可以在请求率足够低，以至于服务器有足够的时间在下一个请求到来之前完成一个请求的处理时使用。很明显，迭代服务器是不可扩展的；向运行此类服务器的计算机添加更多处理器不会增加服务器的吞吐量。另一方面，并行服务器可以处理更高的请求率；如果实现得当，它们是可扩展的。在多处理器计算机上运行的真正并行服务器可以处理比在单处理器计算机上运行的相同服务器更高的请求率。

从实现的角度来看，另一种对服务器应用程序进行分类的方法是按照服务器是同步的还是异步的。一个**同步服务器**使用同步套接字 API 调用，这些调用会阻塞执行线程直到请求的操作完成，或者发生错误。因此，一个典型的同步 TCP 服务器会使用如`asio::ip::tcp::acceptor::accept()`这样的方法来接受客户端连接请求，使用`asio::ip::tcp::socket::read_some()`从客户端接收请求消息，然后使用`asio::ip::tcp::socket::write_some()`将响应消息发送回客户端。这三个方法都是阻塞的。它们会阻塞执行线程直到请求的操作完成，或者发生错误，这使得使用这些操作的服务器是**同步**的。

与同步服务器应用相反，**异步服务器应用**使用异步套接字 API 调用。例如，异步 TCP 服务器可能使用`asio::ip::tcp::acceptor::async_accept()`方法异步接受客户端连接请求，使用`asio::ip::tcp::socket::async_read_some()`方法或`asio::async_read()`自由函数异步接收来自客户端的请求消息，然后使用`asio::ip::tcp::socket::async_write_some()`方法或`asio::async_write()`自由函数异步向客户端发送响应消息。

由于同步服务器应用程序的结构与异步服务器应用程序的结构显著不同，因此应该在服务器应用程序设计阶段早期做出应用哪种方法的决定，并且这个决定应该基于对应用程序要求的仔细分析。此外，还应考虑可能出现的应用演变路径和新需求。

通常，每种方法都有其优点和缺点。当同步方法在某种情况下产生更好的结果时，在另一种情况下可能完全不可接受；在这种情况下，异步方法可能是正确的选择。让我们比较两种方法，以更好地理解每种方法的优点和缺点。

与异步方法相比，同步方法的主要优势在于其**简单性**。同步服务器比功能上等效的异步服务器更容易实现、调试和支持。由于异步操作在代码中完成的位置与它们启动的位置不同，异步服务器更复杂。通常，这需要在空闲内存中分配额外的数据结构以保持请求的上下文，实现回调函数、线程同步以及其他可能使应用程序结构相当复杂且容易出错的功能。在同步服务器中，大多数这些功能都不是必需的。此外，异步方法引入了额外的计算和内存开销，这可能在某些情况下使其不如同步方法高效。

然而，同步方法有一些功能限制，这通常使其不可接受。这些限制包括无法在同步操作开始后取消操作，或为其分配超时，以便在运行时间过长时中断它。与同步操作相反，异步操作可以在操作开始后的任何时刻取消。

同步操作无法取消的事实显著限制了同步服务器应用的领域。公开可用的使用同步操作的服务器容易受到攻击者的攻击。如果这样的服务器是单线程的，一个恶意客户端就足以阻止服务器，不允许其他客户端与其通信。攻击者使用的恶意客户端连接到服务器，但不向其发送任何数据，而后者在同步读取函数或方法中被阻塞，这阻止了它为其他客户端提供服务。

这样的服务器通常用于安全且受保护的私有网络环境，或作为在单个计算机上运行的应用程序内部的一部分，该应用程序使用这样的服务器进行进程间通信。当然，同步服务器的另一个可能的应用领域是实现一次性原型。

除了上述描述的结构复杂性和功能差异之外，这两种方法在处理大量以高频率发送请求的客户端时的效率和可扩展性也存在差异。使用异步操作的服务器比同步服务器更高效和可扩展，尤其是在它们运行在具有原生支持异步网络 I/O 的多处理器计算机上时。

## 样例协议

在本章中，我们将探讨三个配方，描述如何实现同步迭代 TCP 服务器、同步并行 TCP 服务器和异步 TCP 服务器。在所有配方中，假设服务器通过以下故意简化的（为了清晰起见）应用层协议与客户端通信。

服务器应用程序接受表示为 ASCII 字符串的请求消息，其中包含一系列以换行 ASCII 符号结束的符号。服务器忽略换行符号之后的所有符号。

服务器在收到请求后执行一些模拟操作，并以如下恒定消息进行回复：

```cpp
"Response\n"
```

这样的简单协议使我们能够专注于实现*服务器*，而不是它提供的*服务*。

# 实现同步迭代 TCP 服务器

同步迭代 TCP 服务器是满足以下标准的一个分布式应用程序的组成部分：

+   在客户端-服务器通信模型中充当服务器角色

+   通过 TCP 协议与客户端应用程序通信

+   使用 I/O 和控制操作，这些操作会阻塞执行线程，直到相应的操作完成或发生错误。

+   以串行、逐个的方式处理客户端

典型的同步迭代 TCP 服务器按照以下算法工作：

1.  分配一个接受器套接字并将其绑定到特定的 TCP 端口。

1.  运行循环直到服务器停止：

    1.  等待来自客户端的连接请求。

    1.  当连接请求到达时接受客户端的连接请求。

    1.  等待来自客户端的请求消息。

    1.  读取请求消息。

    1.  处理请求。

    1.  向客户端发送响应消息。

    1.  关闭与客户端的连接并释放套接字。

本示例演示了如何使用 Boost.Asio 实现一个同步迭代 TCP 服务器应用程序。

## 如何做到这一点…

我们通过定义一个负责读取请求消息、处理它并发送响应消息的类来开始实现我们的服务器应用程序。此类代表服务器应用程序提供的单个服务，因此我们将它命名为 `Service`：

```cpp
#include <boost/asio.hpp>

#include <thread>
#include <atomic>
#include <memory>
#include <iostream>

using namespace boost;

class Service {
public:
  Service(){}

  void HandleClient(asio::ip::tcp::socket& sock) {
    try {
      asio::streambuf request;
      asio::read_until(sock, request, '\n');

      // Emulate request processing.
      inti = 0;
      while (i != 1000000)
        i++;
        std::this_thread::sleep_for(
std::chrono::milliseconds(500));

      // Sending response.
      std::string response = "Response\n";
      asio::write(sock, asio::buffer(response));
}
    catch (system::system_error&e) {
      std::cout  << "Error occured! Error code = " 
<<e.code() << ". Message: "
          <<e.what();
    }
  }
};
```

为了保持简单，在我们的示例服务器应用程序中，我们实现了一个模拟服务，它仅模拟执行某些操作。请求处理模拟包括执行许多增量操作来模拟消耗 CPU 的操作，然后让控制线程休眠一段时间来模拟读取文件或与外围设备同步通信等操作。

### 注意

`Service` 类相当简单，只包含一个方法。然而，在现实世界应用程序中代表服务的类通常会更为复杂且功能更丰富，尽管主要思想保持不变。

接下来，我们定义另一个代表高级 *接受者* 概念的类（与代表 `asio::ip::tcp::acceptor` 类的低级概念相比）。此类负责接受来自客户端的连接请求，并实例化 `Service` 类的对象，这些对象将为连接的客户端提供服务。让我们相应地命名这个类为 `Acceptor`：

```cpp
class Acceptor {
public:
  Acceptor(asio::io_service&ios, unsigned short port_num) :
    m_ios(ios),
    m_acceptor(m_ios,
        asio::ip::tcp::endpoint(
              asio::ip::address_v4::any(),
              port_num))
  {
    m_acceptor.listen();
  }

  void Accept() {
    asio::ip::tcp::socket sock(m_ios);

    m_acceptor.accept(sock);

    Service svc;
    svc.HandleClient(sock);
  }

private:
  asio::io_service&m_ios;
  asio::ip::tcp::acceptor m_acceptor;
};
```

此类拥有一个名为 `m_acceptor` 的 `asio::ip::tcp::acceptor` 类型的对象，用于同步接受传入的连接请求。

此外，我们还定义了一个代表服务器本身的类。这个类相应地命名为 `Server`：

```cpp
class Server {
public:
  Server() : m_stop(false) {}

  void Start(unsigned short port_num) {
    m_thread.reset(new std::thread([this, port_num]() {
      Run(port_num);
    }));
  }

  void Stop() {
    m_stop.store(true);
    m_thread->join();
  }

private:
  void Run(unsigned short port_num) {
    Acceptor acc(m_ios, port_num);

    while (!m_stop.load()) {
      acc.Accept();
    }
  }

  std::unique_ptr<std::thread>m_thread;
  std::atomic<bool>m_stop;
  asio::io_servicem_ios;
};
```

此类提供了一个由两个方法组成的接口—`Start()` 和 `Stop()`，分别用于启动和停止服务器。循环在由 `Start()` 方法产生的单独线程中运行。`Start()` 方法是非阻塞的，而 `Stop()` 方法会阻塞调用线程，直到服务器停止。

对 `Server` 类的彻底检查揭示了服务器实现的一个严重缺点—在某些情况下，`Stop()` 方法可能永远不会返回。关于这个问题及其解决方法的讨论将在本食谱的后面提供。

最终，我们实现了应用程序入口点函数 `main()`，演示了如何使用 `Server` 类：

```cpp
int main()
{
  unsigned short port_num = 3333;

  try {
    Server srv;
    srv.Start(port_num);

    std::this_thread::sleep_for(std::chrono::seconds(60));

    srv.Stop();
  }
  catch (system::system_error&e) {
        std::cout  << "Error occured! Error code = " 
                   <<e.code() << ". Message: "
                   <<e.what();
  }

  return 0;
}
```

## 它是如何工作的…

示例服务器应用程序由四个组件组成—`Server`、`Acceptor`、`Service` 类和应用程序入口点函数 `main()`。让我们考虑每个组件是如何工作的。

### `Service` 类

`Service`类是整个应用程序中的关键功能组件。虽然其他组件在其目的上是基础设施性的，但这个类实现了服务器提供给客户端的实际功能（或服务）。

这个类很简单，只包含一个`HandleClient()`方法。这个方法接受一个表示连接到客户端的套接字的对象作为其输入参数，并处理该特定客户端。

在我们的示例中，这种处理很简单。首先，从套接字中同步读取请求消息，直到遇到新的换行 ASCII 符号`\n`。然后，处理请求。在我们的情况下，我们通过运行一个模拟循环执行一百万次递增操作，然后让线程休眠半秒钟来模拟处理。之后，准备响应消息，并同步发送回客户端。

Boost.Asio I/O 函数和方法可能抛出的异常在`HandleClient()`方法中被捕获和处理，并且不会传播到方法调用者，这样如果处理一个客户端失败，服务器仍然可以继续工作。

根据特定应用程序的需求，`Service`类可以被扩展并增加提供所需服务的功能。

### 接受者类

`Acceptor`类是服务器应用程序基础设施的一部分。当创建时，它实例化一个接受器套接字对象`m_acceptor`，并调用其`listen()`方法以开始监听来自客户端的连接请求。

这个类公开了一个名为`Accept()`的单个公共方法。当调用此方法时，会实例化一个名为`sock`的`asio::ip::tcp::socket`类对象，代表一个活动套接字，并尝试接受一个连接请求。如果有待处理的连接请求可用，则处理连接请求，并将活动套接字`sock`连接到新的客户端。否则，此方法会阻塞，直到新的连接请求到达。

然后，创建了一个`Service`对象的实例，并调用其`HandleClient()`方法。连接到客户端的`sock`对象被传递到这个方法中。`HandleClient()`方法会阻塞，直到与客户端的通信和请求处理完成，或者发生错误。当`HandleClient()`方法返回时，`Acceptor`类的`Accept()`方法也返回。现在，*接受者*已准备好接受下一个连接请求。

类的`Accept()`方法的一次执行完成了一个客户端的完整处理周期。

### 服务器类

如其名称所示，`Server`类代表一个可以通过类的接口方法`Start()`和`Stop()`进行控制的*服务器*。

`Start()` 方法启动服务器的启动。它产生一个新的线程，从 `Server` 类的 `Run()` 私有方法开始执行并返回。`Run()` 方法接受一个名为 `port_num` 的单个参数，指定接受器套接字应在哪个协议端口上监听传入的连接请求。当调用时，该方法首先实例化一个 `Acceptor` 类的对象，然后启动一个循环，在该循环中调用 `Acceptor` 对象的 `Accept()` 方法。当 `m_stop` 原子变量的值变为 `true` 时，循环终止，这发生在对 `Server` 类的相应实例调用 `Stop()` 方法时。

`Stop()` 方法同步停止服务器。它不会返回，直到在 `Run()` 方法中启动的循环被中断并且由 `Start()` 方法产生的线程完成其执行。为了中断循环，将原子变量 `m_stop` 的值设置为 `true`。之后，`Stop()` 方法在 `Run()` 方法中运行循环的线程表示对象 `m_thread` 上调用 `join()` 方法，等待它退出循环并完成执行。

所提出的实现有一个显著的缺点，即服务器可能无法立即停止。更严重的是，有可能服务器根本不会停止，并且 `Stop()` 方法将永远阻塞其调用者。问题的根本原因在于服务器对客户端行为的强依赖性。

如果在 `Run()` 方法中检查循环终止条件之前，调用 `Stop()` 方法并将原子变量 `m_stop` 的值设置为 `true`，则服务器几乎立即停止，并且不会出现任何问题。然而，如果在服务器线程在 `acc.Accept()` 方法中阻塞等待来自客户端的下一个连接请求，或者在 `Service` 类中的某个同步 I/O 操作中等待来自已连接客户端的请求消息，或者等待客户端接收响应消息时调用 `Stop()` 方法，则服务器无法停止，直到这些阻塞操作完成。因此，例如，如果在调用 `Stop()` 方法时没有挂起的连接请求，则服务器将不会停止，直到新的客户端连接并得到处理，在一般情况下可能永远不会发生，这将导致服务器永远阻塞。

在本节稍后，我们将考虑解决这一缺点的可能方法。

### `main()` 入口点函数

这个函数演示了服务器的使用方法。它创建了一个名为 `srv` 的 `Server` 类实例，并调用其 `Start()` 方法来启动服务器。由于服务器被表示为一个在其自己的控制线程中运行的活跃对象，因此 `Start()` 方法立即返回，而运行方法 `main()` 继续执行。为了让服务器运行一段时间，主线程被置于休眠状态 60 秒。当主线程醒来后，它会在 `srv` 对象上调用 `Stop()` 方法来停止服务器。当 `Stop()` 方法返回时，`main()` 函数也返回，我们的示例应用程序退出。

当然，在实际应用中，服务器会作为对用户输入或任何其他相关事件的反应而停止，而不是在模拟的 60 秒后，或者在服务器启动运行结束后。

### 消除缺点

正如已经提到的，所提出的实现有两个缺点，这显著限制了其适用性。第一个问题是，如果在服务器线程阻塞等待传入连接请求时调用 `Stop()` 方法，可能无法停止服务器，因为没有连接请求到达。第二个问题是，服务器可以被单个恶意（或存在错误的）客户端轻易挂起，使其对其他客户端不可用。为了挂起服务器，客户端应用程序可以简单地连接到服务器，并且永远不向它发送任何请求，这将使服务器应用程序在阻塞输入操作中永远挂起。

这两个问题的根本原因是在服务器中使用阻塞操作（对于同步服务器来说这是自然的）。解决这两个问题的合理且简单的方法是为阻塞操作分配超时时间，这将保证服务器会定期解除阻塞以检查是否已发出停止命令，并且强制丢弃长时间不发送请求的客户。然而，Boost.Asio 不提供取消同步操作或为它们分配超时时间的方法。因此，我们应该尝试找到其他方法来使我们的同步服务器更加响应和稳定。

让我们考虑解决这两个缺点的方法。

#### 在合理的时间内停止服务器

由于在没有任何挂起的连接请求时，使接受者套接字的 `accept()` 同步方法非阻塞的唯一合法方法是向接受者正在监听的端口发送一个模拟连接请求，我们可以使用以下技巧来解决问题。

在`Server`类的`Stop()`方法中，在将`m_stop`原子变量的值设置为`true`之后，我们可以创建一个虚拟活动套接字，将其连接到同一个服务器，并发送一些虚拟请求。这将保证服务器线程将离开接受者的`accept()`方法，并最终检查`m_stop`原子变量的值，发现其值等于`true`，这将导致循环终止并完成`Acceptor::Accept()`方法。

在描述的方法中，假设服务器通过向自己发送消息（实际上是从 I/O 线程发送到工作线程的消息）来停止自己。另一种方法是有特殊客户端（独立应用程序），它会连接并发送特殊服务消息（例如，`stop\n`）到服务器，服务器将解释为停止信号。在这种情况下，服务器将外部控制（来自不同的应用程序），`Server`类不需要有`Stop()`方法。

#### 处理服务器的漏洞

很遗憾，没有分配超时的 I/O 操作阻塞的性质是这样的，它可以很容易地挂起使用此类操作的迭代服务器，使其对其他客户端不可访问。

显然，为了保护服务器免受这种漏洞的影响，我们需要重新设计它，使其永远不会被 I/O 操作阻塞。实现这一目标的一种方法是通过使用非阻塞套接字（这将使我们的服务器变为响应式）或使用异步 I/O 操作。这两种选择都意味着我们的服务器不再同步。我们将在本章的其他菜谱中考虑这些解决方案。

### 分析结果

如上所述，使用 Boost.Asio 实现的同步迭代服务器中固有的漏洞不允许在公共网络上使用，因为存在服务器被恶意者滥用的风险。通常，同步服务器会在封闭和保护的环境中用于客户端被精心设计，以确保它们不会使服务器挂起。

迭代同步服务器的另一个局限性是它们不可扩展，无法利用多处理器硬件。然而，它们的优点——简单性——是为什么这种类型的服务器在许多情况下是一个好的选择的原因。

## 参见

+   第二章，*I/O 操作*，包括提供关于如何执行同步 I/O 的详细讨论的菜谱。

# 实现同步并行 TCP 服务器

同步并行 TCP 服务器是分布式应用的一部分，满足以下标准：

+   在客户端-服务器通信模型中充当服务器

+   通过 TCP 协议与客户端应用程序通信

+   使用 I/O 和控制操作，直到相应的操作完成或发生错误，才会阻塞执行线程

+   可以同时处理多个客户端

一个典型的同步并行 TCP 服务器按照以下算法工作：

1.  分配一个接受者套接字并将其绑定到特定的 TCP 端口。

1.  运行循环直到服务器停止：

    +   等待来自客户端的连接请求

    +   接受客户端的连接请求

    +   在控制线程的上下文中启动一个线程：

        +   等待来自客户端的请求消息

        +   读取请求消息

        +   处理请求

        +   向客户端发送响应消息

        +   关闭与客户端的连接并释放套接字

此配方演示了如何使用 Boost.Asio 实现同步并行 TCP 服务器应用程序。

## 如何做到这一点…

我们开始实现我们的服务器应用程序，通过定义一个处理单个客户端的类，该类通过读取请求消息、处理它并发送响应消息来处理单个客户端。此类代表服务器应用程序提供的一个单一服务，因此我们将它命名为`Service`：

```cpp
#include <boost/asio.hpp>

#include <thread>
#include <atomic>
#include <memory>
#include <iostream>

using namespace boost;

class Service {
public:
   Service(){}

   void StartHandligClient(
         std::shared_ptr<asio::ip::tcp::socket> sock) {

      std::thread th(([this, sock]() {
         HandleClient(sock);
      }));

      th.detach();
   }

private: 
void HandleClient(std::shared_ptr<asio::ip::tcp::socket> sock) {
      try {
         asio::streambuf request;
         asio::read_until(*sock.get(), request, '\n');

         // Emulate request processing.
         int i = 0;
         while (i != 1000000)
            i++;

            std::this_thread::sleep_for(
std::chrono::milliseconds(500));

         // Sending response.
         std::string response = "Response\n";
         asio::write(*sock.get(), asio::buffer(response));
      } 
      catch (system::system_error &e) {
         std::cout    << "Error occured! Error code = " 
<< e.code() << ". Message: "
               << e.what();
      }

      // Clean-up.
      delete this;
   }
};
```

为了保持简单，在我们的示例服务器应用程序中，我们实现了一个模拟服务，它仅模拟执行某些操作。请求处理模拟包括执行许多增量操作来模拟消耗 CPU 的操作，然后让控制线程休眠一段时间来模拟同步的 I/O 操作，如读取文件或与外围设备通信。

### 注意

`Service`类相当简单，只包含一个方法。然而，在现实世界的应用中，代表服务的类通常会更为复杂且功能更丰富，尽管主要思想保持不变。

接下来，我们定义另一个类，它代表一个高级的*接受者*概念（与`asio::ip::tcp::acceptor`类所代表的低级概念相比）。此类负责接受来自客户端的连接请求，并实例化`Service`类的对象，这些对象将为连接的客户端提供服务。让我们称它为`Acceptor`：

```cpp
class Acceptor {
public:
   Acceptor(asio::io_service& ios, unsigned short port_num) :
      m_ios(ios),
      m_acceptor(m_ios,
          asio::ip::tcp::endpoint(
asio::ip::address_v4::any(), 
port_num))
   {
      m_acceptor.listen();
   }

   void Accept() {
      std::shared_ptr<asio::ip::tcp::socket> 
sock(new asio::ip::tcp::socket(m_ios));

      m_acceptor.accept(*sock.get());

      (new Service)->StartHandligClient(sock);
   }

private:
   asio::io_service& m_ios;
   asio::ip::tcp::acceptor m_acceptor;
};
```

此类拥有一个名为`m_acceptor`的`asio::ip::tcp::acceptor`类对象，用于同步接受传入的连接请求。

此外，我们还定义了一个代表服务器的类。类名相应地命名为——`Server`：

```cpp
class Server {
public:
  Server() : m_stop(false) {}

  void Start(unsigned short port_num) {
    m_thread.reset(new std::thread([this, port_num]() {
      Run(port_num);
    }));
  }

  void Stop() {
    m_stop.store(true);
    m_thread->join();
  }

private:
  void Run(unsigned short port_num) {
    Acceptor acc(m_ios, port_num);

    while (!m_stop.load()) {
      acc.Accept();
    }
  }

  std::unique_ptr<std::thread>m_thread;
  std::atomic<bool>m_stop;
  asio::io_servicem_ios;
};
```

此类提供了一个由两个方法组成的接口——`Start()`和`Stop()`，分别用于启动和停止服务器。循环在`Start()`方法启动的单独线程中运行。`Start()`方法是非阻塞的，而`Stop()`方法是阻塞的。它会阻塞调用线程，直到服务器停止。

对`Server`类进行彻底检查揭示了服务器实现的一个严重缺点——`Stop()`方法可能会永远阻塞。下面提供了关于此问题及其解决方法的讨论。

最终，我们实现了应用程序的入口点函数`main()`，演示了如何使用`Server`类：

```cpp
int main()
{
   unsigned short port_num = 3333;

   try {
      Server srv;
      srv.Start(port_num);

      std::this_thread::sleep_for(std::chrono::seconds(60));

      srv.Stop();
   }
   catch (system::system_error &e) {
      std::cout    << "Error occured! Error code = " 
<< e.code() << ". Message: "
            << e.what();
   }

   return 0;
}
```

## 它是如何工作的…

样本服务器应用程序由四个组件组成——`Server`、`Acceptor` 和 `Service` 类以及应用程序入口点函数 `main()`。让我们考虑每个组件是如何工作的。

### `Service` 类

`Service` 类是整个应用程序中的关键功能组件。虽然其他组件构成了服务器的基础设施，但此类实现了服务器提供给客户端的实际功能（或服务）。

此类在其接口中有一个名为 `StartHandlingClient()` 的单一方法。此方法接受一个指向表示连接到客户端的 TCP 套接字的对象的指针作为其输入参数，并开始处理该特定客户端。

此方法启动一个控制线程，从类的 `HandleClient()` 私有方法开始执行，在那里执行实际的同步处理。启动线程后，`StartHandlingClient()` 方法通过将其从代表它的 `std::thread` 对象中分离出来来“释放”线程。之后，`StartHandlingClient()` 方法返回。

如其名称所暗示的，`HandleClient()` 私有方法处理客户端。在我们的示例中，这种处理是微不足道的。首先，从套接字中同步读取请求消息，直到遇到新的换行 ASCII 符号 `\n`。然后，处理请求。在我们的情况下，我们通过运行一个模拟循环执行一百万次递增操作，然后让线程休眠半秒钟来模拟处理。之后，准备响应消息并发送回客户端。

当发送响应消息时，与当前正在运行的 `HandleClient()` 方法关联的 `Service` 类对象被 `delete` 操作符删除。当然，类的设计假设其实例将通过 `new` 操作符在自由内存中分配，而不是在栈上。

根据特定应用程序的需求，`Service` 类可以被扩展并丰富以提供所需的服务功能。

### `Acceptor` 类

`Acceptor` 类是服务器应用程序基础设施的一部分。当构造时，它实例化一个接受器套接字对象 `m_acceptor` 并调用其 `listen()` 方法以开始监听来自客户端的连接请求。

此类公开了一个名为 `Accept()` 的单一公共方法。当调用此方法时，它实例化一个名为 `sock` 的 `asio::ip::tcp::socket` 类对象，代表一个活动套接字，并尝试接受一个连接请求。如果有挂起的连接请求可用，则处理连接请求并将活动套接字 `sock` 连接到新的客户端。否则，此方法会阻塞，直到新的连接请求到达。

然后，在空闲内存中分配了一个`Service`对象的实例，并调用了其`StartHandlingClient()`方法。将`sock`对象作为输入参数传递给此方法。`StartHandlingClient()`方法在客户端将被处理的环境中创建了一个线程，并立即返回。当`StartHandlingClient()`方法返回时，`Acceptor`类的`Accept()`方法也返回了。现在，*接受者*已准备好接受下一个连接请求。

注意，`Acceptor`不拥有`Service`类的对象。相反，当`Service`类完成其工作时，该对象将自行销毁。

### `Server`类

如其名称所示，`Server`类代表一个可以通过类的`Start()`和`Stop()`接口方法进行控制的*服务器*。

`Start()`方法启动服务器的启动过程。它创建了一个新线程，该线程从`Server`类的`Run()`私有方法开始执行并返回。`Run()`方法接受一个名为`port_num`的单个参数，指定了接受者套接字应监听协议端口的编号，以便接收传入的连接请求。当被调用时，该方法首先实例化一个`Acceptor`类的对象，然后启动一个循环，在该循环中调用`Acceptor`对象的`Accept()`方法。当在`Server`类的相应实例上调用`Stop()`方法时，`m_stop`原子变量的值变为`true`，循环终止。

`Stop()`方法同步停止服务器。它不会返回，直到`Run()`方法中启动的循环被中断，并且`Start()`方法创建的线程完成其执行。为了中断循环，将原子变量`m_stop`的值设置为`true`。之后，`Stop()`方法在`Run()`方法中运行的循环对应的`m_thread`对象上调用`join()`方法，以等待其完成执行。

所展示的实现有一个显著的缺点，即服务器可能无法立即停止。更甚者，存在服务器根本无法停止的可能性，并且`Stop()`方法将永远阻塞其调用者。问题的根本原因在于服务器对客户端行为的强依赖。

如果在`Run()`方法中检查循环终止条件之前调用`Stop()`方法并将原子变量`m_stop`的值设置为`true`，则服务器几乎会立即停止，并且不会发生任何问题。然而，如果在服务器线程在`acc.Accept()`方法中阻塞等待来自客户端的下一个连接请求，或者在`Service`类中的某个同步 I/O 操作中等待来自已连接客户端的请求消息或客户端接收响应消息时调用`Stop()`方法，则服务器无法停止，直到这些阻塞操作完成。因此，例如，如果在调用`Stop()`方法时没有挂起的连接请求，服务器将不会停止，直到新的客户端连接并得到处理，在一般情况下可能永远不会发生，可能导致服务器永久阻塞。

在本节稍后，我们将考虑解决此缺点的方法。

### `main()`入口点函数

此功能演示了服务器的使用方法。它创建了一个名为`srv`的`Server`类实例，并调用其`Start()`方法来启动服务器。因为服务器被表示为一个在其自己的控制线程中运行的活跃对象，所以`Start()`方法会立即返回，运行`main()`方法的线程继续执行。为了使服务器运行一段时间，主线程被休眠 60 秒。当主线程醒来后，它会调用`srv`对象的`Stop()`方法来停止服务器。当`Stop()`方法返回时，`main()`函数也会返回，我们的示例应用程序退出。

当然，在实际应用中，服务器会作为对用户输入或任何其他相关事件的反应而停止，而不是在服务器启动运行后的 60 秒后。

### 消除缺点

使用 Boost.Asio 库实现的同步并行服务器应用程序的固有缺点与之前配方中考虑的同步迭代服务器应用程序的缺点相似。请参阅*实现同步迭代 TCP 服务器*配方，以了解缺点的讨论和消除它们的方法。

## 参见

+   配方*实现同步迭代 TCP 服务器*提供了关于同步迭代和同步并行服务器固有的缺点以及消除它们的可能方法的更多详细信息。

+   第二章，*I/O 操作*，包括提供如何执行同步 I/O 的详细讨论的配方

# 实现异步 TCP 服务器

异步 TCP 服务器是满足以下标准的一个分布式应用程序的组成部分：

+   在客户端-服务器通信模型中充当服务器

+   通过 TCP 协议与客户端应用程序通信

+   使用异步 I/O 和控制操作

+   可以同时处理多个客户端

一个典型的异步 TCP 服务器按照以下算法工作：

1.  分配一个接受者套接字并将其绑定到特定的 TCP 端口。

1.  启动异步接受操作。

1.  在 Boost.Asio 事件循环中创建一个或多个控制线程并将它们添加到线程池中。

1.  当异步接受操作完成时，启动一个新的操作以接受下一个连接请求。

1.  启动异步读取操作以从连接的客户端读取请求。

1.  当异步读取操作完成时，处理请求并准备响应消息。

1.  启动异步写入操作以向客户端发送响应消息。

1.  当异步写入操作完成时，关闭连接并释放套接字。

注意，前一个算法中的第四步开始的步骤可能根据具体应用程序中具体异步操作的相对时间顺序以任意顺序执行。由于服务器的异步模型，即使在单处理器计算机上运行服务器时，步骤的执行顺序也可能不成立。

这个配方演示了如何使用 Boost.Asio 实现异步 TCP 服务器应用程序。

## 如何做到这一点...

我们通过定义一个类来开始实现我们的服务器应用程序，该类负责通过读取请求消息、处理它并发送响应消息来处理单个客户端。这个类代表了服务器应用程序提供的一个单一服务。让我们称它为 `Service`：

```cpp
#include <boost/asio.hpp>

#include <thread>
#include <atomic>
#include <memory>
#include <iostream>

using namespace boost;

class Service {
public:
   Service(std::shared_ptr<asio::ip::tcp::socket> sock) :
      m_sock(sock)
   {}

   void StartHandling() {

      asio::async_read_until(*m_sock.get(), 
            m_request, 
            '\n', 
            this 
                        {                  
                              onRequestReceived(ec,
                               bytes_transferred);
               });
   }

private:
   void onRequestReceived(const boost::system::error_code& ec,
                std::size_t bytes_transferred) {
      if (ec != 0) {
         std::cout << "Error occured! Error code = "
            << ec.value()
            << ". Message: " << ec.message();

         onFinish();
                return;
      }

// Process the request.
      m_response = ProcessRequest(m_request);

      // Initiate asynchronous write operation.
      asio::async_write(*m_sock.get(), 
            asio::buffer(m_response),
            this 
                            {
                  onResponseSent(ec,
                                  bytes_transferred);
               });
   }

   void onResponseSent(const boost::system::error_code& ec,
                      std::size_t bytes_transferred) {
      if (ec != 0) {
         std::cout << "Error occured! Error code = "
            << ec.value()
            << ". Message: " << ec.message();
      }

      onFinish();
   }

   // Here we perform the cleanup.
   void onFinish() {
      delete this;
   }

   std::string ProcessRequest(asio::streambuf& request) {

      // In this method we parse the request, process it
      // and prepare the request.

      // Emulate CPU-consuming operations.
      int i = 0;
      while (i != 1000000)
         i++;

      // Emulate operations that block the thread
// (e.g. synch I/O operations).
         std::this_thread::sleep_for(
                      std::chrono::milliseconds(100));

      // Prepare and return the response message. 
      std::string response = "Response\n";
      return response;
   }

private:
   std::shared_ptr<asio::ip::tcp::socket> m_sock;
   std::string m_response;
   asio::streambuf m_request;
};
```

为了保持简单，在我们的示例服务器应用程序中，我们实现了一个模拟服务，它仅模拟执行某些操作。请求处理模拟包括执行许多增量操作来模拟消耗 CPU 的操作，然后让控制线程休眠一段时间来模拟同步的 I/O 操作，例如读取文件或与外围设备通信。

`Service` 类的每个实例都旨在通过读取请求消息、处理它并发送响应消息来处理一个连接的客户端。

接下来，我们定义另一个类，它代表一个高级 *接受者* 概念（与由 `asio::ip::tcp::acceptor` 类代表的低级概念相比）。这个类负责接受来自客户端的连接请求并实例化 `Service` 类的对象，该对象将为连接的客户端提供服务。让我们称它为 `Acceptor`：

```cpp
class Acceptor {
public:
  Acceptor(asio::io_service&ios, unsigned short port_num) :
    m_ios(ios),
    m_acceptor(m_ios,
      asio::ip::tcp::endpoint(
                  asio::ip::address_v4::any(), 
                  port_num)),
    m_isStopped(false)
  {}

  // Start accepting incoming connection requests.
  void Start() {
    m_acceptor.listen();
    InitAccept();
  }

  // Stop accepting incoming connection requests.
  void Stop() {
    m_isStopped.store(true);
  }

private:
  void InitAccept() {
    std::shared_ptr<asio::ip::tcp::socket>
              sock(new asio::ip::tcp::socket(m_ios));

    m_acceptor.async_accept(*sock.get(),
      this, sock 
           {
        onAccept(error, sock);
      });
  }

  void onAccept(const boost::system::error_code&ec,
               std::shared_ptr<asio::ip::tcp::socket> sock) 
  {
    if (ec == 0) {
      (new Service(sock))->StartHandling();
    }
    else {
      std::cout<< "Error occured! Error code = "
        <<ec.value()
        << ". Message: " <<ec.message();
    }

    // Init next async accept operation if
    // acceptor has not been stopped yet.
    if (!m_isStopped.load()) {
      InitAccept();
    }
    else {
      // Stop accepting incoming connections
      // and free allocated resources.
      m_acceptor.close();
    }
  }

private:
  asio::io_service&m_ios;
  asio::ip::tcp::acceptor m_acceptor;
  std::atomic<bool>m_isStopped;
}; 
```

这个类拥有一个名为 `m_acceptor` 的 `asio::ip::tcp::acceptor` 类对象，用于异步接受传入的连接请求。

此外，我们还定义了一个代表服务器本身的类。该类名称相应地命名为 `Server`：

```cpp
class Server {
public:
   Server() {
      m_work.reset(new asio::io_service::work(m_ios));
   }

   // Start the server.
   void Start(unsigned short port_num, 
unsigned int thread_pool_size) {

      assert(thread_pool_size > 0);

      // Create and start Acceptor.
      acc.reset(new Acceptor(m_ios, port_num));
      acc->Start();

      // Create specified number of threads and 
      // add them to the pool.
      for (unsigned int i = 0; i < thread_pool_size; i++) {
         std::unique_ptr<std::thread> th(
                   new std::thread([this]()
                   {
                          m_ios.run();
                   }));

         m_thread_pool.push_back(std::move(th));
      }
   }

   // Stop the server.
   void Stop() {
      acc->Stop();
      m_ios.stop();

      for (auto& th : m_thread_pool) {
         th->join();
      }
   }

private:
   asio::io_servicem_ios;
   std::unique_ptr<asio::io_service::work>m_work;
   std::unique_ptr<Acceptor>acc;
   std::vector<std::unique_ptr<std::thread>>m_thread_pool;
};
```

本类提供了一个包含两个方法——`Start()` 和 `Stop()` 的接口。`Start()` 方法接受服务器应监听传入连接请求的协议端口号以及要添加到池中的线程数作为输入参数，并启动服务器。`Stop()` 方法停止服务器。`Start()` 方法是非阻塞的，而 `Stop()` 方法是阻塞的。它会阻塞调用线程，直到服务器停止并且所有运行事件循环的线程退出。

最后，我们实现了应用程序入口点函数 `main()`，它演示了如何使用 `Server` 类的对象：

```cpp
const unsigned intDEFAULT_THREAD_POOL_SIZE = 2;

int main()
{
  unsigned short port_num = 3333;

  try {
    Server srv;

    unsigned intthread_pool_size =
      std::thread::hardware_concurrency() * 2;

      if (thread_pool_size == 0)
      thread_pool_size = DEFAULT_THREAD_POOL_SIZE;

    srv.Start(port_num, thread_pool_size);

    std::this_thread::sleep_for(std::chrono::seconds(60));

    srv.Stop();
  }
  catch (system::system_error&e) {
    std::cout  << "Error occured! Error code = " 
               <<e.code() << ". Message: "
               <<e.what();
  }

  return 0;
}
```

## 它是如何工作的...

示例服务器应用程序由四个组件组成——`Service`、`Acceptor` 和 `Service` 类以及一个应用程序入口点函数 `main()`。让我们考虑每个组件是如何工作的。

### `Service` 类

`Service` 类是应用程序中的关键功能组件。虽然其他组件构成了服务器的基础设施，但此类实现了服务器提供给客户端的实际功能（或服务）。

本类的单个实例旨在通过读取请求、处理它，然后发送响应消息来处理单个已连接客户端。

类的构造函数接受一个表示连接到特定客户端的套接字的共享指针作为参数，并缓存此指针。此套接字将用于稍后与客户端应用程序通信。

`Service` 类的公共接口由一个名为 `StartHandling()` 的单一方法组成。此方法通过启动异步读取操作来处理客户端，该操作从客户端读取请求消息，并将 `onRequestReceived()` 方法指定为回调。启动异步读取操作后，`StartHandling()` 方法返回。

当请求读取完成或发生错误时，会调用回调方法 `onRequestReceived()`。此方法首先通过测试包含操作完成状态码的 `ec` 参数来检查读取是否成功。如果读取以错误结束，则将相应的消息输出到标准输出流，然后调用 `onFinish()` 方法。之后，`onRequestReceived()` 方法返回，导致客户端处理过程中断。

如果请求消息已成功读取，则调用 `ProcessRequest()` 方法来执行请求的操作并准备响应消息。当 `ProcessRequest()` 方法完成并返回包含响应消息的字符串时，启动异步写入操作以将此响应消息发送回客户端。`onResponseSent()` 方法被指定为回调。

当写入操作完成（或发生错误）时，会调用 `onResponseSent()` 方法。此方法首先检查操作是否成功。如果操作失败，则将相应的消息输出到标准输出流。接下来，调用 `onFinish()` 方法以执行清理。当 `onFinish()` 方法返回时，客户端处理的全周期被认为是完成的。

`ProcessRequest()` 方法是类的核心，因为它实现了服务。在我们的服务器应用程序中，我们有一个模拟服务，它运行一个模拟循环，执行一百万次自增操作，然后使线程休眠 100 毫秒。之后，生成模拟响应消息并返回给调用者。

根据特定应用程序的需求，`Service` 类及其 `ProcessRequest()` 方法可以被扩展和丰富，以提供所需的服务功能。

`Service` 类被设计成当其任务完成时，其对象会自行删除。删除操作在类的 `onFinish()` 私有方法中执行，该方法在客户端处理周期结束时被调用，无论操作是否成功或出错：

```cpp
void onFinish() {
  delete this;
}
```

### 接收者类

`Acceptor` 类是服务器应用程序基础设施的一部分。其构造函数接受一个端口号，它将在此端口号上监听传入的连接请求作为其输入参数。此类的对象包含一个名为 `m_acceptor` 的 `asio::ip::tcp::acceptor` 类的实例，该实例在 `Acceptor` 类的构造函数中构建。

`Acceptor` 类公开了两个公共方法——`Start()` 和 `Stop()`。`Start()` 方法旨在指示 `Acceptor` 类的对象开始监听并接受传入的连接请求。它将 `m_acceptor` 接收器套接字置于监听模式，然后调用类的 `InitAccept()` 私有方法。`InitAccept()` 方法反过来构建一个活动套接字对象，并启动异步接受操作，在接收器套接字对象上调用 `async_accept()` 方法，并将表示活动套接字的对象作为参数传递给它。`Acceptor` 类的 `onAccept()` 方法被指定为回调。

当连接请求被接受或发生错误时，会调用回调方法 `onAccept()`。此方法首先检查在异步操作执行过程中是否发生了任何错误，通过检查其输入参数 `ec` 的值。如果操作成功完成，则创建 `Service` 类的一个实例，并调用其 `StartHandling()` 方法，该方法开始处理已连接的客户端。否则，在出错的情况下，将相应的消息输出到标准输出流。

接下来，检查`m_isStopped`原子变量的值，以查看是否已对`Acceptor`对象发出停止命令。如果是（这意味着已对`Acceptor`对象调用`Stop()`方法），则不会启动新的异步接受操作，并且关闭低级接受器对象。此时，`Acceptor`停止监听并接受来自客户端的传入连接请求。否则，调用`InitAccept()`方法来启动一个新的异步接受操作，以接受下一个传入的连接请求。

正如之前提到的，`Stop()`方法指示`Acceptor`对象在当前运行的异步接受操作完成后不要启动下一个异步接受操作。然而，此方法不会取消当前正在运行的接受操作。

### `Server`类

如其名称所示，`Server`类代表一个*服务器*本身。该类的公共接口由两个方法组成：`Start()`和`Stop()`。

`Start()`方法启动服务器。它接受两个参数。第一个参数名为`port_num`，指定服务器应监听传入连接的协议端口号。第二个参数名为`thread_pool_size`，指定要添加到运行事件循环和传递异步操作完成事件的线程池中的线程数。此参数非常重要，应该谨慎选择，因为它直接影响到服务器的性能。

`Start()`方法首先实例化一个`Acceptor`类的对象，该对象将用于接受传入的连接，然后通过调用其`Start()`方法启动它。之后，它通过调用`asio::io_service`对象的`run()`方法，创建一组工作线程，每个线程都被添加到线程池中。此外，所有`std::thread`对象都被缓存到`m_thread_pool`成员向量中，以便在服务器停止时可以稍后连接这些线程。

`Stop()`方法首先停止`Acceptor`对象`acc`，调用其`Stop()`方法。然后，它调用`asio::io_service`对象`m_ios`上的`stop()`方法，使得之前调用`m_ios.run()`的所有线程尽快加入线程池并退出，丢弃所有挂起的异步操作。之后，`Stop()`方法通过遍历缓存于`m_thread_pool`向量中的所有`std::thread`对象，等待线程池中的所有线程退出。

当所有线程退出时，`Stop()`方法返回。

### `main()`入口点函数

此函数演示了服务器使用方法。首先，它创建了一个名为`srv`的`Server`类对象。因为`Server`类的`Start()`方法需要一个由线程池构成的多个线程传递给它，所以在启动服务器之前，计算池的最优大小。在并行应用程序中通常使用的通用公式是计算机处理器数量的两倍来找到最优的线程数。我们使用`std::thread::hardware_concurrency()`静态方法来获取处理器数量。然而，因为这个方法可能无法完成其任务返回 0，所以我们回退到由常量`DEFAULT_THREAD_POOL_SIZE`表示的默认值，在我们的情况下等于 2。

当线程池大小计算完毕后，调用`Start()`方法来启动服务器。`Start()`方法不会阻塞。当它返回时，运行`main()`方法的线程继续执行。为了允许服务器运行一段时间，主线程被休眠 60 秒。当主线程醒来时，它调用`srv`对象的`Stop()`方法来停止服务器。当`Stop()`方法返回时，`main()`函数也返回，我们的应用程序退出。

当然，在实际应用中，服务器会作为对某些相关事件（如用户输入）的反应而停止，而不是当某个虚拟的时间段过去时。

## 参见

+   第二章，*I/O 操作*，包括提供详细讨论如何执行同步 I/O 的配方。

+   来自第六章的*使用计时器*配方，*其他主题*，展示了如何使用 Boost.Asio 提供的计时器。计时器可以用来实现异步操作超时机制。
