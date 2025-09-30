# 第三章 实现客户端应用

在这一章中，我们将涵盖以下主题：

+   实现同步 TCP 客户端

+   实现同步 UDP 客户端

+   实现异步 TCP 客户端

# 简介

**客户端**是分布式应用的一部分，它通过与其他应用部分（称为**服务器**）通信来消费它提供的服务。另一方面，服务器是分布式应用的一部分，它被动地等待来自客户端的请求。当请求到达时，服务器执行请求的操作，并将操作结果作为响应发送回客户端。

客户端的关键特征是它需要服务器提供的服务，并且它需要与该服务器建立通信会话以消费该服务。服务器的关键特征是它通过提供请求的服务来响应来自客户端的请求。

我们将在下一章中考虑服务器。在这一章中，我们将专注于客户端应用，并将详细考虑几种类型。

## 客户端应用的分类

客户端应用可以根据它们与服务器通信所使用的传输层协议进行分类。如果客户端使用 UDP 协议，则称为**UDP 客户端**。如果它使用 TCP 协议，则相应地称为**TCP 客户端**。当然，还有许多其他传输层协议，客户端应用可能用于通信。此外，还有多协议客户端，可以在多个协议上通信。然而，这些超出了本书的范围。在这一章中，我们将专注于纯 UDP 和 TCP 客户端，因为它们是最受欢迎的，并且在今天的通用软件中是最常用的。

关于在分布式应用的部分之间选择哪种传输层协议进行通信的决定，应在应用设计的早期阶段基于应用规范做出。因为 TCP 和 UDP 协议在概念上不同，所以在应用开发过程的后期阶段从其中一个切换到另一个可能会相当困难。

另一种根据客户端是同步还是异步来分类客户端应用的方法。**同步客户端应用**使用同步套接字 API 调用，这些调用会阻塞执行线程，直到请求的操作完成或发生错误。因此，一个典型的同步 TCP 客户端会使用`asio::ip::tcp::socket::write_some()`方法或`asio::write()`免费函数向服务器发送请求，然后使用`asio::ip::tcp::socket::read_some()`方法或`asio::read()`免费函数接收响应。这些方法和函数是阻塞的，这使得客户端是同步的。

与同步客户端应用相对的是，异步客户端应用使用异步套接字 API 调用。例如，异步 TCP 客户端可能使用`asio::ip::tcp::socket::async_write_some()`方法或`asio::async_write()`免费函数向服务器发送请求，然后使用`asio::ip::tcp::socket::async_read_some()`方法或`asio::async_read()`免费函数异步接收响应。

由于同步客户端的结构与异步客户端的结构显著不同，因此关于应用哪种方法的决策应在应用设计阶段尽早做出，并且这个决策应基于对应用要求的仔细分析。此外，还应考虑可能的应用演变路径和未来可能出现的新要求。

## 同步与异步

通常情况下，每种方法都有其优缺点。当同步方法在某种情况下给出更好的结果时，在另一种情况下可能完全不可接受。在后一种情况下，应使用异步方法。让我们比较两种方法，以更好地理解在什么情况下使用每种方法更有利。

同步方法的主要优势是其*简单性*。与功能上相等的异步客户端相比，同步客户端的开发、调试和支持要容易得多。由于异步客户端使用的异步操作在代码的其他地方（主要是在回调中）完成，而不是在它们开始的地方，因此异步客户端更复杂。通常，这需要在空闲内存中分配额外的数据结构来保持请求和回调函数的上下文，还涉及到线程同步和其他可能使应用程序结构相当复杂且容易出错的额外操作。大多数这些额外操作在同步客户端中都不是必需的。此外，异步方法引入了额外的计算和内存开销，在某些条件下使其不如同步方法高效。

然而，同步方法有一些功能限制，这通常使得这种方法不可接受。这些限制包括在操作开始后无法取消同步操作，或无法为其设置超时，以便在运行时间超过一定时间后中断。与同步操作相反，异步操作可以在操作开始后的任何时刻取消，直到操作完成之前。

想象一个典型的现代网络浏览器。请求取消是一个客户端应用程序的重要功能。在发出加载特定网站的命令后，用户可能会改变主意并决定在页面加载完成之前取消命令。从用户的角度来看，如果不能在页面完全加载之前取消命令，将会非常奇怪。因此，在这种情况下，同步方法不是一个好的选择。

除了上述复杂性和功能上的差异之外，这两种方法在并行运行多个请求时的效率也有所不同。

想象一下我们正在开发一个网络爬虫，这是一个遍历网站页面并处理它们以提取一些有趣信息的应用程序。给定一个包含大量网站（比如说几百万个）的文件，应用程序应该遍历文件中列出的每个网站的页面，然后处理每个页面。自然地，该应用程序的一个关键要求是尽可能快地完成任务。考虑到这些要求，我们应该选择哪种方法，同步还是异步？

在我们回答这个问题之前，让我们从客户端应用程序的角度考虑请求生命周期的各个阶段及其时间。从概念上讲，请求生命周期由以下五个阶段组成：

1.  **准备请求**：此阶段涉及准备请求消息所需的任何操作。这一步骤的持续时间取决于应用程序解决的特定问题。在我们的例子中，这可能是从输入文件中读取下一个网站地址并构建一个符合 HTTP 协议的请求字符串。

1.  **从客户端向服务器发送请求**：此阶段假设请求数据通过网络从客户端传输到服务器。这一步骤的持续时间不依赖于客户端应用程序。它取决于网络的特性和当前状态。

1.  **服务器处理请求**：这一步骤的持续时间取决于服务器的特性和其当前负载。在我们的例子中，服务器应用程序是一个网络服务器，请求处理包括构建请求的网页，这可能涉及 I/O 操作，如读取文件和从数据库加载数据。

1.  **从服务器向客户端发送响应**：与第 2 阶段类似，此阶段也假设通过网络传输数据；然而，这次方向相反——从服务器到客户端。这一阶段的持续时间不依赖于客户端或服务器。它只取决于网络的特性和状态。

1.  **客户端处理响应**：这一阶段的时间取决于客户端应用程序打算执行的具体任务。在我们的例子中，这可能是扫描网页，提取有趣的信息并将其存储到数据库中。

注意，为了简化，我们省略了如连接建立和连接关闭等低级子阶段，这些在使用 TCP 协议时很重要，但对我们请求生命周期概念模型中的实质性价值不大。

正如我们所见，只有在第 1 和第 5 阶段，客户端才会执行与请求相关的有效工作。在第一阶段结束时启动了请求数据的传输后，客户端必须等待请求生命周期的下一个三个阶段（第 2、第 3 和第 4 阶段）才能接收响应并处理它。

现在，让我们带着请求生命周期的各个阶段在心中，看看当我们应用同步和异步方法来实现我们的示例网络爬虫时会发生什么。

如果我们采用同步方法，处理单个请求的执行线程将在请求生命周期的第 2-4 阶段处于休眠状态，只有在第 1 和第 5 阶段，它才会执行有效的工作（为了简化，我们假设第 1 和第 5 阶段不包括会阻塞线程的指令）。这意味着操作系统的资源，即线程，被使用得不够高效，因为在很多情况下，它只是在无所事事，而此时还有很多工作要做——数百万个其他页面需要请求和处理。在这种情况下，异步方法似乎更有效率。采用异步方法，线程在请求生命周期的第 2-4 阶段不会阻塞，它可以有效地用于执行另一个请求的第 1 或第 5 阶段。

因此，我们指导单个线程处理不同请求的不同阶段（这被称为**重叠**），这导致线程使用得更加高效，从而提高了应用程序的整体性能。

然而，异步方法并不总是比同步方法更有效率。正如所提到的，异步操作意味着额外的计算开销，这意味着异步操作的整体持续时间（从开始到完成）略大于等效的同步操作。这意味着，如果第 2-4 阶段的平均总持续时间小于异步方法每个请求的时延开销，那么同步方法就变得更为高效，因此可能被认为是正确的选择。

评估请求生命周期阶段 2-4 的总持续时间以及异步方法的开销通常是通过实验来完成的。持续时间可能会有显著差异，这取决于请求和响应传输的网络属性和状态，以及服务请求的服务器应用程序的属性和负载级别。

## 示例协议

在本章中，我们将考虑三个配方，每个配方都演示了如何实现特定类型的客户端应用程序：同步 UDP 客户端、同步 TCP 客户端和异步 TCP 客户端。在所有配方中，假设客户端应用程序使用以下简单的应用层协议与服务器应用程序通信。

服务器应用程序接受一个表示为 ASCII 字符串的请求。该字符串具有以下格式：

```cpp
EMULATE_LONG_COMP_OP [s]<LF>
```

其中 `[s]` 是一个正整数值，`<LF>` 是 ASCII 换行符。

服务器将此字符串解释为执行持续 `[s]` 秒的虚拟操作请求。例如，请求字符串可能如下所示：

```cpp
"EMULATE_LONG_COMP_OP 10\n"
```

这意味着发送此请求的客户端希望服务器执行持续 `10` 秒的虚拟操作，然后向它发送响应。

与请求一样，服务器返回的响应由一个 ASCII 字符串表示。它可以是 `OK<LF>`，如果操作成功完成，或者 `ERROR<LF>`，如果操作失败。

# 实现同步 TCP 客户端

同步 TCP 客户端是符合以下声明的分布式应用程序的一部分：

+   在客户端-服务器通信模型中充当客户端

+   使用 TCP 协议与服务器应用程序通信

+   使用 I/O 和控制操作（至少是那些与服务器通信相关的 I/O 操作），这些操作会阻塞执行线程，直到相应的操作完成或发生错误

典型的同步 TCP 客户端按照以下算法工作：

1.  获取服务器应用程序的 IP 地址和协议端口号。

1.  分配一个活动套接字。

1.  与服务器应用程序建立连接。

1.  与服务器交换消息。

1.  关闭连接。

1.  释放套接字。

本配方演示了如何使用 Boost.Asio 实现同步 TCP 客户端应用程序。

## 如何操作...

以下代码示例演示了使用 Boost.Asio 实现同步 TCP 客户端应用程序的可能实现。客户端使用本章引言部分中描述的应用层协议：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

class SyncTCPClient {
public:
  SyncTCPClient(const std::string& raw_ip_address,
    unsigned short port_num) :
    m_ep(asio::ip::address::from_string(raw_ip_address),
    port_num),
    m_sock(m_ios) {

    m_sock.open(m_ep.protocol());
  }

  void connect() {
    m_sock.connect(m_ep);
  }

  void close() {
    m_sock.shutdown(
      boost::asio::ip::tcp::socket::shutdown_both);
    m_sock.close();
  }

  std::string emulateLongComputationOp(
    unsigned int duration_sec) {

    std::string request = "EMULATE_LONG_COMP_OP "
      + std::to_string(duration_sec)
      + "\n";

    sendRequest(request);
    return receiveResponse();
  };

private:
  void sendRequest(const std::string& request) {
    asio::write(m_sock, asio::buffer(request));
  }

  std::string receiveResponse() {
    asio::streambuf buf;
    asio::read_until(m_sock, buf, '\n');

    std::istream input(&buf);

    std::string response;
    std::getline(input, response);

    return response;
  }

private:
  asio::io_service m_ios;

  asio::ip::tcp::endpoint m_ep;
  asio::ip::tcp::socket m_sock;
};

int main()
{
  const std::string raw_ip_address = "127.0.0.1";
  const unsigned short port_num = 3333;

  try {
    SyncTCPClient client(raw_ip_address, port_num);

    // Sync connect.
    client.connect();

    std::cout << "Sending request to the server... "
      << std::endl;

    std::string response =
      client.emulateLongComputationOp(10);

    std::cout << "Response received: " << response
      << std::endl;

    // Close the connection and free resources.
    client.close();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的...

示例客户端应用程序由两个主要组件组成——`SyncTCPClient` 类和应用程序入口点函数 `main()`，其中 `SyncTCPClient` 类用于与服务器应用程序通信。让我们分别考虑每个组件。

### SyncTCPClient 类

`SyncTCPClient`类是样本中的关键组件。它实现了并提供对通信功能的访问。

该类有三个私有成员如下：

+   `asio::io_service m_ios`: 这是提供对操作系统通信服务访问的对象，这些服务由套接字对象使用

+   `asio::ip::tcp::endpoint m_ep`: 这是一个指定服务器应用程序的端点

+   `asio::ip::tcp::socket m_sock`: 这是用于通信的套接字

类中的每个对象都旨在与单个服务器应用程序通信；因此，类的构造函数接受服务器 IP 地址和协议端口号作为其参数。这些值用于在构造函数的初始化列表中实例化`m_ep`对象。套接字对象`m_sock`也在构造函数中实例化和打开。

三个公共方法构成了`SyncTCPClient`类的接口。第一个名为`connect()`的方法相当简单；它执行套接字与服务器的连接。`close()`方法关闭连接并关闭套接字，这会导致操作系统中的套接字及其相关资源被释放。

第三个接口方法是`emulateLongComputationOp(unsigned int duration_sec)`。该方法是在其中执行 I/O 操作的地方。它从根据协议准备请求字符串开始。然后，请求被传递到类的私有方法`sendRequest(const std::string& request)`，该方法将其发送到服务器。当请求发送并且`sendRequest()`方法返回时，调用`receiveResponse()`方法从服务器接收响应。当收到响应时，`receiveResponse()`方法返回包含响应的字符串。之后，`emulateLongComputationOp()`方法将响应返回给其调用者。

让我们更详细地看看`sendRequest()`和`receiveResponse()`方法。

`sendRequest()`方法具有以下原型：

```cpp
void sendRequest(const std::string& request)
```

其目的是将作为参数传递给它的字符串发送到服务器。为了将数据发送到服务器，使用了`asio::write()`免费同步函数。函数在请求发送后返回。这就是`sendRequest()`方法的所有内容。基本上，它所做的只是，完全委托其工作给`asio::write()`免费函数。

发送请求后，我们现在想从服务器接收响应。这是`SyncTCPClient`类的`receiveResponse()`方法的目的。为了执行其工作，该方法使用`asio::read_until()`免费函数。根据应用层协议，服务器发送的响应消息的长度可能不同，但必须以`\n`符号结束；因此，我们在调用函数时指定此符号作为分隔符：

```cpp
asio::streambuf buf;
asio::read_until(m_sock, buf, '\n');
```

该函数阻塞执行线程，直到遇到来自服务器的消息中的`\n`符号。当函数返回时，流缓冲区`buf`包含响应。然后将数据从`buf`缓冲区复制到`response`字符串，并将后者返回给调用者。`emulateLongComputationOp()`方法随后将响应返回给其调用者——`main()`函数。

关于`SyncTCPClient`类需要注意的一点是，它不包含与错误处理相关的代码。这是因为该类仅使用那些在失败时抛出异常的 Boost.Asio 函数和对象方法的重载。假设类的用户负责捕获和处理异常。

### `main()`入口点函数

此函数作为`SyncTCPClient`类的用户。在获取服务器 IP 地址和协议端口号（这部分在示例中省略）后，它实例化并使用`SyncTCPClient`类的一个对象与服务器通信，以使用其服务，主要是模拟在服务器上执行 10 秒的虚拟计算操作。此函数的代码简单且易于理解，因此无需额外的注释。

## 参见

+   第二章，*I/O 操作*，包括提供详细讨论如何执行同步 I/O 的配方

# 实现同步 UDP 客户端

同步 UDP 客户端是符合以下声明的分布式应用程序的一部分：

+   在客户端-服务器通信模型中充当客户端

+   使用 UDP 协议与服务器应用程序通信

+   使用 I/O 和控制操作（至少是那些与服务器通信相关的 I/O 操作）阻塞执行线程，直到相应的操作完成或发生错误

典型的同步 UDP 客户端按照以下算法工作：

1.  获取客户端应用程序打算与之通信的每个服务器的 IP 地址和协议端口号。

1.  分配一个 UDP 套接字。

1.  与服务器交换消息。

1.  释放套接字。

此配方演示了如何使用 Boost.Asio 实现同步 UDP 客户端应用程序。

## 如何做到这一点...

以下代码示例演示了使用 Boost.Asio 实现同步 UDP 客户端应用程序的可能方法。假设客户端使用 UDP 协议，底层为 IPv4 协议进行通信：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

class SyncUDPClient {
public:
  SyncUDPClient() :
    m_sock(m_ios) {

    m_sock.open(asio::ip::udp::v4());
  }

  std::string emulateLongComputationOp(
    unsigned int duration_sec,
    const std::string& raw_ip_address,
    unsigned short port_num) {

    std::string request = "EMULATE_LONG_COMP_OP "
      + std::to_string(duration_sec)
      + "\n";

    asio::ip::udp::endpoint ep(
      asio::ip::address::from_string(raw_ip_address),
      port_num);

    sendRequest(ep, request);
    return receiveResponse(ep);
  };

private:
  void sendRequest(const asio::ip::udp::endpoint& ep,
    const std::string& request) {

    m_sock.send_to(asio::buffer(request), ep);
  }

  std::string receiveResponse(asio::ip::udp::endpoint& ep) {
    char response[6];
    std::size_t bytes_recieved =
      m_sock.receive_from(asio::buffer(response), ep);

    m_sock.shutdown(asio::ip::udp::socket::shutdown_both);
    return std::string(response, bytes_recieved);
  }

private:
  asio::io_service m_ios;

  asio::ip::udp::socket m_sock;
};

int main()
{
  const std::string server1_raw_ip_address = "127.0.0.1";
  const unsigned short server1_port_num = 3333;

  const std::string server2_raw_ip_address = "192.168.1.10";
  const unsigned short server2_port_num = 3334;

  try {
    SyncUDPClient client;

    std::cout << "Sending request to the server #1 ... "
      << std::endl;

    std::string response =
      client.emulateLongComputationOp(10,
      server1_raw_ip_address, server1_port_num);

    std::cout << "Response from the server #1 received: "
      << response << std::endl;

    std::cout << "Sending request to the server #2... "
      << std::endl;

    response =
      client.emulateLongComputationOp(10,
      server2_raw_ip_address, server2_port_num);

    std::cout << "Response from the server #2 received: "
      << response << std::endl;
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的...

示例由两个主要组件组成——`SyncUDPClient`类和应用程序入口点函数`main()`，后者使用`SyncUDPClient`类与两个服务器应用程序通信。让我们分别考虑每个组件。

### `SyncUDPClient`类

`SyncUDPClient`类是示例中的关键组件。它实现了服务器通信功能，并为用户提供访问。

该类有两个私有成员如下：

+   `asio::io_service m_ios`：这是提供对操作系统通信服务访问的对象，这些服务由套接字对象使用

+   `asio::ip::udp::socket m_sock`：这是用于通信的 UDP 套接字

`m_sock`套接字对象在类的构造函数中被实例化和打开。由于客户端打算使用 IPv4 协议，我们将`asio::ip::udp::v4()`静态方法返回的对象传递给套接字的`open()`方法，以指定套接字使用 IPv4 协议。

由于`SyncUDPClient`类实现了基于 UDP 协议的通信，UDP 是一种无连接协议，因此该类的一个单独对象可以用来与多个服务器通信。该类的接口由一个单一的方法组成——`emulateLongComputationOp()`。这个方法可以在`SyncUDPClient`类的对象实例化后立即用来与服务器通信。以下是这个方法的原型：

```cpp
std::string emulateLongComputationOp(
         unsigned int duration_sec,
         const std::string& raw_ip_address,
         unsigned short port_num)
```

除了表示请求参数的`duration_sec`参数外，该方法还接受服务器 IP 地址和协议端口号。此方法可以多次调用以与不同的服务器通信。

该方法首先根据应用层协议准备一个请求字符串并创建一个指定目标服务器应用程序的端点对象。然后，请求字符串和端点对象被传递到类的私有方法`sendRequest()`，该方法将请求消息发送到指定的服务器。当请求发送并且`sendRequest()`方法返回时，调用`receiveResponse()`方法从服务器接收响应。

当收到响应时，`receiveResponse()`方法返回包含响应的字符串。然后，`emulateLongComputationOp()`方法将响应返回给其调用者。`sendRequest()`方法使用套接字对象的`send_to()`方法将请求消息发送到特定的服务器。让我们看看这个方法的声明： 

```cpp
  template <typename ConstBufferSequence>
  std::size_t send_to(const ConstBufferSequence& buffers,
      const endpoint_type& destination)
```

该方法接受一个包含请求的缓冲区和指定缓冲区内容应发送到的服务器端点的端点对象作为参数，并阻塞直到整个缓冲区发送完毕，或者发生错误。请注意，如果该方法在没有错误的情况下返回，这仅意味着请求已被发送，*并不*意味着服务器已收到请求。UDP 协议不保证消息的传递，并且不提供检查数据报是否已成功在服务器端接收或在传输过程中丢失的方法。

发送请求后，我们现在想要从服务器接收响应。这是`SyncUDPClient`类的`receiveResponse()`方法的目的。该方法首先分配一个将保存响应消息的缓冲区。我们选择缓冲区的大小，以便它可以容纳服务器根据应用层协议可能发送的最大消息；这条消息是一个由六个 ASCII 符号组成的`ERROR\n`字符串，因此长度为 6 个字节；因此，这是我们的缓冲区大小 - 6 个字节。因为缓冲区足够小，所以我们将其分配在栈上。

要读取从服务器到达的响应数据，我们使用套接字对象的`receive_from()`方法。以下是该方法的原型：

```cpp
  template <typename MutableBufferSequence>
  std::size_t receive_from(const MutableBufferSequence& buffers,
      endpoint_type& sender_endpoint) 
```

此方法将来自由`sender_endpoint`对象指定的服务器的数据报复制到由`buffers`参数指定的缓冲区。

关于套接字对象的`receive_from()`方法有两点需要注意。首先，这个方法是同步的，它会在数据报从指定的服务器到达之前阻塞执行线程。如果数据报永远不会到达（例如，在前往客户端的路上丢失），该方法将永远不会解除阻塞，整个应用程序将会挂起。其次，如果来自服务器的数据报的大小大于提供的缓冲区的大小，该方法将失败。

接收到响应后，创建一个`std::string`对象，用响应字符串初始化，并将其返回给调用者——`emulateLongComputationOp()`方法。然后，它将响应返回给其调用者——`main()`函数。

`SyncUDPClient`类不包含错误处理相关的代码。这是因为它只使用那些在失败时抛出异常的 Boost.Asio 函数和对象方法的那些重载。假设类的用户负责捕获和处理异常。

### `main()`入口点函数

在此函数中，我们使用`SyncUDPClient`类与两个服务器应用程序进行通信。首先，我们获取目标服务器应用程序的 IP 地址和端口号。然后，我们实例化`SyncUDPClient`类的对象，并调用对象的`emulateLongComputationOp()`方法两次，以同步地从两个不同的服务器消费相同的服务。

## 参见

+   第二章，*I/O 操作*，包括提供详细讨论如何执行同步 I/O 的食谱。

# 实现异步 TCP 客户端

如本章引言部分所述，最简单的异步客户端在结构上比等效的同步客户端更复杂。当我们向异步客户端添加如请求取消等特性时，它变得更加复杂。

在这个菜谱中，我们将考虑一个支持异步执行请求和请求取消功能的异步 TCP 客户端应用程序。以下是该应用程序将满足的要求列表：

+   用户输入应该在单独的线程中处理——用户界面线程。这个线程不应该被阻塞在明显的时间段内。

+   用户应该能够向不同的服务器发出多个请求。

+   用户应该在之前发出的请求完成之前能够发出新的请求。

+   用户应该在请求完成之前能够取消之前发出的请求。

## 如何做到这一点...

由于我们的应用程序需要支持请求取消，我们首先指定启用 Windows 上请求取消的设置：

```cpp
#include <boost/predef.h> // Tools to identify the OS.

// We need this to enable cancelling of I/O operations on
// Windows XP, Windows Server 2003 and earlier.
// Refer to "http://www.boost.org/doc/libs/1_58_0/
// doc/html/boost_asio/reference/basic_stream_socket/
// cancel/overload1.html" for details.
#ifdef BOOST_OS_WINDOWS
#define _WIN32_WINNT 0x0501

#if _WIN32_WINNT <= 0x0502 // Windows Server 2003 or earlier.
  #define BOOST_ASIO_DISABLE_IOCP
  #define BOOST_ASIO_ENABLE_CANCELIO  
#endif
#endif
```

然后，我们包含必要的头文件并指定方便的 `using` 指令：

```cpp
#include <boost/asio.hpp>

#include <thread>
#include <mutex>
#include <memory>
#include <iostream>

using namespace boost;
```

我们继续定义一个表示回调函数指针的数据类型。因为我们的客户端应用程序将是异步的，我们需要一个作为请求完成通知机制的概念。稍后，我们将清楚地了解为什么需要它以及它是如何被使用的：

```cpp
// Function pointer type that points to the callback
// function which is called when a request is complete.
typedef void(*Callback) (unsigned int request_id,
  const std::string& response,
  const system::error_code& ec);
```

接下来，我们定义一个数据结构，其目的是在执行过程中保持与特定请求相关的数据。让我们称它为 `Session`：

```cpp
// Structure represents a context of a single request.
struct Session {
  Session(asio::io_service& ios,
  const std::string& raw_ip_address,
  unsigned short port_num,
  const std::string& request,
  unsigned int id,
  Callback callback) :
  m_sock(ios),
  m_ep(asio::ip::address::from_string(raw_ip_address),
  port_num),
  m_request(request),
  m_id(id),
  m_callback(callback),
  m_was_cancelled(false) {}

  asio::ip::tcp::socket m_sock; // Socket used for communication
  asio::ip::tcp::endpoint m_ep; // Remote endpoint.
  std::string m_request;        // Request string.

  // streambuf where the response will be stored.
  asio::streambuf m_response_buf;
  std::string m_response; // Response represented as a string.

  // Contains the description of an error if one occurs during
  // the request life cycle.
  system::error_code m_ec;

  unsigned int m_id; // Unique ID assigned to the request.

  // Pointer to the function to be called when the request
  // completes.
  Callback m_callback;

  bool m_was_cancelled;
  std::mutex m_cancel_guard;
};
```

所有 `Session` 数据结构包含的字段的目的将在我们继续前进时变得清晰。

接下来，我们定义一个提供异步通信功能的类。让我们称它为 `AsyncTCPClient`：

```cpp
class AsyncTCPClient : public boost::noncopyable {
class AsyncTCPClient : public boost::noncopyable {
public:
   AsyncTCPClient(){
      m_work.reset(new boost::asio::io_service::work(m_ios));

      m_thread.reset(new std::thread([this](){
         m_ios.run();
      }));
   }

   void emulateLongComputationOp(
      unsigned int duration_sec,
      const std::string& raw_ip_address,
      unsigned short port_num,
      Callback callback,
      unsigned int request_id) {

      // Preparing the request string.
      std::string request = "EMULATE_LONG_CALC_OP "
         + std::to_string(duration_sec)
         + "\n";

      std::shared_ptr<Session> session =
         std::shared_ptr<Session>(new Session(m_ios,
         raw_ip_address,
         port_num,
         request,
         request_id,
         callback));

      session->m_sock.open(session->m_ep.protocol());

      // Add new session to the list of active sessions so
      // that we can access it if the user decides to cancel
      // the corresponding request before it completes.
      // Because active sessions list can be accessed from 
      // multiple threads, we guard it with a mutex to avoid 
      // data corruption.
      std::unique_lock<std::mutex>
         lock(m_active_sessions_guard);
      m_active_sessions[request_id] = session;
      lock.unlock();

      session->m_sock.async_connect(session->m_ep, 
         this, session 
         {
         if (ec != 0) {
            session->m_ec = ec;
            onRequestComplete(session);
            return;
         }

         std::unique_lock<std::mutex>
            cancel_lock(session->m_cancel_guard);

         if (session->m_was_cancelled) {
            onRequestComplete(session);
            return;
         }

                asio::async_write(session->m_sock, 
                             asio::buffer(session->m_request),
         this, session 
         {
         if (ec != 0) {
            session->m_ec = ec;
            onRequestComplete(session);
            return;
         }

         std::unique_lock<std::mutex>
            cancel_lock(session->m_cancel_guard);

         if (session->m_was_cancelled) {
            onRequestComplete(session);
            return;
         }

                asio::async_read_until(session->m_sock,
                                  session->m_response_buf, 
                                  '\n', 
         this, session 
         {
         if (ec != 0) {
            session->m_ec = ec;
         } else {
            std::istream strm(&session->m_response_buf);
            std::getline(strm, session->m_response);
         }

         onRequestComplete(session);
      });});});
   };

   // Cancels the request.  
   void cancelRequest(unsigned int request_id) {
      std::unique_lock<std::mutex>
         lock(m_active_sessions_guard);

      auto it = m_active_sessions.find(request_id);
      if (it != m_active_sessions.end()) {
         std::unique_lock<std::mutex>
            cancel_lock(it->second->m_cancel_guard);

         it->second->m_was_cancelled = true;
         it->second->m_sock.cancel();
      }
   }

   void close() {
      // Destroy work object. This allows the I/O thread to
      // exits the event loop when there are no more pending
      // asynchronous operations. 
      m_work.reset(NULL);

      // Wait for the I/O thread to exit.
      m_thread->join();
   }

private:
   void onRequestComplete(std::shared_ptr<Session> session) {
      // Shutting down the connection. This method may
      // fail in case socket is not connected. We don’t care 
      // about the error code if this function fails.
      boost::system::error_code ignored_ec;

      session->m_sock.shutdown(
         asio::ip::tcp::socket::shutdown_both,
         ignored_ec);

      // Remove session form the map of active sessions.
      std::unique_lock<std::mutex>
         lock(m_active_sessions_guard);

      auto it = m_active_sessions.find(session->m_id);
      if (it != m_active_sessions.end())
         m_active_sessions.erase(it);

      lock.unlock();

      boost::system::error_code ec;

      if (session->m_ec == 0 && session->m_was_cancelled)
         ec = asio::error::operation_aborted;
      else
         ec = session->m_ec;

      // Call the callback provided by the user.
      session->m_callback(session->m_id, 
         session->m_response, ec);
   };

private:
   asio::io_service m_ios;
   std::map<int, std::shared_ptr<Session>> m_active_sessions;
   std::mutex m_active_sessions_guard;
   std::unique_ptr<boost::asio::io_service::work> m_work;
   std::unique_ptr<std::thread> m_thread;
};
```

这个类是我们示例中的关键组件，提供了应用程序的大部分功能。这些功能通过类的公共接口提供给用户，该接口包含三个公共方法：

+   `void emulateLongComputationOp(unsigned int duration_sec, const std::string& raw_ip_address, unsigned short port_num, Callback callback, unsigned int request_id)`: 这个方法向服务器发起请求

+   `void cancelRequest(unsigned int request_id)`: 这个方法取消由 `request_id` 参数指定的先前启动的请求

+   `void close()`: 这个方法阻塞调用线程，直到所有当前正在运行的任务完成，并初始化客户端。当这个方法返回时，`AsyncTCPClient` 类的相应实例不能再使用。

现在，我们定义一个将作为回调函数使用的函数，我们将将其传递给 `AsyncTCPClient::emulateLongComputationOp()` 方法。在我们的情况下，这个函数相当简单。如果请求成功完成，它将输出请求执行的结果和响应消息到标准输出流：

```cpp
void handler(unsigned int request_id,
         const std::string& response, 
                const system::error_code& ec) 
{
  if (ec == 0) {
    std::cout << "Request #" << request_id
      << " has completed. Response: "
      << response << std::endl;
  } else if (ec == asio::error::operation_aborted) {
    std::cout << "Request #" << request_id
      << " has been cancelled by the user." 
            << std::endl;
  } else {
    std::cout << "Request #" << request_id
      << " failed! Error code = " << ec.value()
      << ". Error message = " << ec.message() 
             << std::endl;
  }

  return;
}
```

`handler()` 函数的签名对应于之前定义的函数指针类型 `Callback`。

现在我们已经拥有了所有必要的组件，我们定义了应用程序的入口点——`main()`函数，该函数展示了如何使用上面定义的组件与服务器进行通信。在我们的示例函数中，`main()`通过初始化三个请求并取消其中一个来模拟人类用户的行为：

```cpp
int main()
{
  try {
    AsyncTCPClient client;

    // Here we emulate the user's behavior.

    // User initiates a request with id 1.
    client.emulateLongComputationOp(10, "127.0.0.1", 3333,
      handler, 1);
    // Then does nothing for 5 seconds.
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // Then initiates another request with id 2.
    client.emulateLongComputationOp(11, "127.0.0.1", 3334,
      handler, 2);
    // Then decides to cancel the request with id 1.
    client.cancelRequest(1);
    // Does nothing for another 6 seconds.
    std::this_thread::sleep_for(std::chrono::seconds(6));
    // Initiates one more request assigning ID3 to it.
    client.emulateLongComputationOp(12, "127.0.0.1", 3335,
      handler, 3);
    // Does nothing for another 15 seconds.
    std::this_thread::sleep_for(std::chrono::seconds(15));
    // Decides to exit the application.
    client.close();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
};
```

## 它是如何工作的…

我们的示例客户端应用程序使用两个执行线程。第一个线程——UI 线程——负责处理用户输入和初始化请求。第二个线程——I/O 线程——负责运行事件循环并调用异步操作的回调例程。这种配置使我们能够使我们的应用程序的用户界面保持响应。

### 启动应用程序 – `main()`入口点函数

`main()`函数在 UI 线程的上下文中被调用。此函数模拟了发起和取消请求的用户的行为。首先，它创建`AsyncTCPClient`类的实例，然后调用其`emulateLongComputationOp()`方法三次以初始化三个异步请求，每次指定不同的目标服务器。第一个请求（分配 ID 为 1 的请求）在请求初始化几秒钟后通过调用`cancelRequest()`方法被取消。

### 请求完成 – `handler()`回调函数

在`main()`函数中初始化的所有三个请求，`handler()`都被指定为回调函数。无论请求完成的原因是什么——无论是成功完成还是出现错误——该函数都会被调用。此外，当请求被用户取消时，该函数也会被调用。该函数接受以下三个参数：

+   `unsigned int request_id`：这包含请求的唯一标识符。这是在请求初始化时分配给请求的相同标识符。

+   `std::string& response`：这包含响应数据。此值仅在请求成功完成且未被取消时才被认为是有效的。

+   `system::error_code& ec`：如果在请求的生命周期中发生错误，此对象包含错误信息。如果请求被取消，它包含`asio::error::operation_aborted`值。

在我们的示例中，`handler()`函数相当简单。根据传递给它的参数的值，它输出有关已完成请求的信息。

### AsyncTCPClient 类 – 初始化

正如已经提到的，所有与服务器应用程序通信的功能都隐藏在`AsyncTCPClient`类中。这个类有一个不接受任何参数的非空构造函数，执行两个操作。首先，它通过将名为`m_ios`的`asio::io_service`类实例传递给其构造函数来实例化`asio::io_service::work`类的对象。然后，它启动一个线程，该线程调用`m_ios`对象的`run()`方法。`asio::io_service::work`类的对象保持线程运行的事件循环，在没有挂起的异步操作时防止事件循环退出。启动的线程在我们的应用程序中扮演 I/O 线程的角色；在这个线程的上下文中，分配给异步操作的回调将被调用。

### `AsyncTCPClient`类 – 启动请求

`emulateLongComputationOp()`方法旨在启动一个异步请求。它接受五个参数。第一个参数名为`duration_sec`，表示根据应用层协议的请求参数。`raw_ip_address`和`port_num`指定了请求应该发送到的服务器。下一个参数是一个指向回调函数的指针，当请求完成时将被调用。我们将在本节稍后讨论回调。最后一个参数`request_id`是请求的唯一标识符。此标识符与请求相关联，并在以后引用它时使用，例如，当需要取消它时。

`emulateLongComputationOp()`方法从准备请求字符串和分配一个`Session`结构实例开始，该实例保留与请求相关的数据，包括用于与服务器通信的套接字对象。

然后，打开套接字并将指向`Session`对象的指针添加到`m_active_sessions`映射中。这个映射包含指向所有活动请求的`Session`对象的指针，即那些已经发起但尚未完成的请求。当请求完成时，在调用相应的回调之前，与该请求关联的`Session`对象的指针将从映射中移除。

`request_id`参数用作添加到映射中的相应`Session`对象的键。我们需要缓存`Session`对象，以便在用户决定取消先前发起的请求时能够访问它们。如果我们不需要支持请求的取消，我们可以避免使用`m_active_sessions`映射。

我们使用`m_active_session_guard`互斥锁来同步对`m_active_sessions`映射的访问。同步是必要的，因为映射可以从多个线程访问。项目在 UI 线程中添加，在调用回调的 I/O 线程中移除，当相应的请求完成时。

现在，当相应的`Session`对象的指针被缓存时，我们需要通过调用 socket 的`async_connect()`方法将 socket 连接到服务器：

```cpp
session->m_sock.async_connect(session->m_ep,
  this, session
  { 
         // ...
  });
```

将指定我们想要连接的服务器的端点对象和当连接完成或发生错误时将被调用的回调函数作为参数传递给此方法。在我们的示例中，我们使用 lambda 函数作为回调函数。`emulateLongComputationOp()`方法中的最后一条语句是对 socket 的`async_connect()`方法的调用。当`async_connect()`返回时，`emulateLongComputationOp()`也会返回，这意味着请求已经被发起。

让我们更仔细地看看我们传递给`async_connect()`作为回调的 lambda 函数。以下是它的代码：

```cpp
this, session
{
  if (ec != 0) {
    session->m_ec = ec;
    onRequestComplete(session);
    return;
  }

  std::unique_lock<std::mutex>
    cancel_lock(session->m_cancel_guard);

  if (session->m_was_cancelled) {
     onRequestComplete(session);
     return;
  }

  asio::async_write(session->m_sock,
  asio::buffer(session->m_request),
        this, session
              {
                    //...
        });
}
```

回调函数首先检查传递给它的`ec`参数的错误代码，其值如果不同于零，则表示相应的异步操作已失败。在失败的情况下，我们将`ec`值存储在相应的`Session`对象中，调用类的`onRequestComplete()`私有方法，并将`Session`对象作为参数传递给它，然后返回。

如果`ec`对象指定成功，我们锁定`m_cancel_guard`互斥锁（请求描述符对象的成员）并检查请求是否尚未被取消。关于取消请求的更多详细信息将在本节后面的`cancelRequest()`方法中提供。

如果我们看到请求尚未被取消，我们将通过调用 Boost.Asio 自由函数`async_write()`来发送请求数据到服务器，从而启动下一个异步操作。再次，我们传递给它一个 lambda 函数作为回调。这个回调与在异步连接操作启动时传递给`anync_connect()`方法的回调非常相似。我们首先检查错误代码，然后如果它指示成功，我们检查请求是否已被取消。如果没有，我们启动下一个异步操作`async_read_until()`，以便从服务器接收响应：

```cpp
this, session{
  if (ec != 0) {
    session->m_ec = ec;
    onRequestComplete(session);
    return;
  }

  std::unique_lock<std::mutex>
    cancel_lock(session->m_cancel_guard);

  if (session->m_was_cancelled) {
    onRequestComplete(session);
    return;
  }

  asio::async_read_until(session->m_sock,
        session->m_response_buf, '\n', 
     this, session 
        {
      // ...
        });
}
```

再次，我们将一个 lambda 函数作为回调参数传递给`async_read_until()`自由函数。这个回调函数相当简单：

```cpp
this, session 
{
  if (ec != 0) {
    session->m_ec = ec;
  } else {
    std::istream strm(&session->m_response_buf);
    std::getline(strm, session->m_response);
  }

  onRequestComplete(session);
}
```

它检查错误代码，在成功的情况下，将接收到的响应数据存储在相应的`Session`对象中。然后，调用`AsyncTCPClient`类的私有方法`onRequestComplete()`，并将`Session`对象作为参数传递给它。

`onRequestComplete()`方法在请求完成时被调用，无论结果如何。它在请求成功完成时被调用，在请求在其生命周期的任何阶段失败时被调用，或者当它被用户取消时被调用。此方法的目的是在执行清理操作后，调用`emulateLongComputationOp()`方法的调用者提供的回调。

`onRequestComplete()` 方法首先关闭套接字。请注意，在这里我们使用套接字的 `shutdown()` 方法的重载版本，它不会抛出异常。我们不在乎连接关闭失败，因为这在我们这个场景中不是一个关键操作。然后，我们从 `m_active_sessions` 映射中删除相应的条目，因为请求已经完成，因此它不再活跃。最后一步，调用用户提供的回调函数。在回调函数返回后，请求生命周期结束。

### AsyncTCPClient 类 – 取消请求

现在，让我们看看 `AsyncTCPClient` 类的 `cancelRequest()` 方法。这个方法接受要取消的请求的标识符作为参数。它首先在 `m_active_sessions` 映射中查找与指定请求对应的 `Session` 对象。如果找到了，它就会在这个 `Session` 对象中存储的套接字对象上调用 `cancel()` 方法。这会导致与该套接字对象关联的当前正在运行的非阻塞操作中断。

然而，有可能在 `cancelRequest()` 方法被调用时，一个异步操作已经完成，而下一个操作尚未开始。例如，想象一下，I/O 线程现在正在运行与特定套接字关联的 `async_connect()` 操作的回调。在这个时候，与该套接字关联的没有正在进行的异步操作（因为下一个异步操作 `async_write()` 尚未启动）；因此，对这个套接字调用 `cancel()` 将没有效果。这就是为什么我们使用一个额外的标志 `Session::m_was_cancelled`，正如其名称所暗示的，它表示请求是否已被取消（或者更准确地说，是否由用户调用了 `cancelRequest()` 方法）。在异步操作的回调中，我们在启动下一个异步操作之前查看这个标志的值。如果我们看到这个标志被设置（这意味着请求已被取消），我们不会启动下一个异步操作，而是中断请求执行并调用 `onRequestComplete()` 方法。

在 `cancelRequest()` 方法和异步操作（如 `async_connect()` 和 `async_write()`）的回调中，我们使用 `Session::m_cancel_guard` 互斥锁来强制执行以下操作顺序：请求可以在回调中测试 `Session::m_was_cancelled` 标志值之前或之后被取消。这种顺序保证了当用户调用 `cancelRequest()` 方法时，请求能够被正确取消。

### AsyncTCPClient 类 – 关闭客户端

在客户端被使用且不再需要后，应该适当地关闭它。`AsyncTCPClient` 类的 `close()` 方法允许我们这样做。首先，此方法销毁 `m_work` 对象，允许 I/O 线程在所有异步操作完成后退出事件消息循环。然后，它连接 I/O 线程以等待其退出。

在 `close()` 方法返回后，`AsyncTCPClient` 类的相应对象不能再使用。

## 还有更多...

在所提供的示例中，`AsyncTCPClient` 类实现了一个异步的 **单线程** TCP 客户端。它使用一个线程来运行事件循环并处理请求。通常，当请求速率较低时，响应的大小不会很大，请求处理器不会执行响应的复杂和耗时处理（请求生命周期的第 5 阶段）；一个线程就足够了。

然而，当我们希望客户端处理数百万个请求并尽可能快地处理它们时，我们可能希望将我们的客户端转换为 **多线程** 的，这样多个线程可以真正同时运行多个请求。当然，这假设运行客户端的计算机是多核或多处理器计算机。如果应用程序运行的线程数量超过计算机中安装的核心或处理器的数量，可能会因为线程切换开销的影响而减慢应用程序的速度。

### 实现多线程 TCP 客户端应用程序

为了将我们的单线程客户端应用程序转换为多线程应用程序，我们需要对其进行一些更改。首先，我们需要将代表单个 I/O 线程的 `AnyncTCPClient` 类的 `m_thread` 成员替换为指向 `std::thread` 对象的指针列表，这些对象将代表一组 I/O 线程：

```cpp
std::list<std::unique_ptr<std::thread>> m_threads;
```

接下来，我们需要更改类的构造函数，使其接受一个表示要创建的 I/O 线程数量的参数。此外，构造函数应该启动指定的 I/O 线程数量，并将它们全部添加到运行事件循环的线程池中：

```cpp
AsyncTCPClient(unsigned char num_of_threads){
  m_work.reset(new boost::asio::io_service::work(m_ios));

  for (unsigned char i = 1; i <= num_of_threads; i++) {
         std::unique_ptr<std::thread> th(
               new std::thread([this](){
        m_ios.run();
      }));

      m_threads.push_back(std::move(th));
    }
  }
```

就像客户端的单线程版本一样，每个线程都会调用 `m_ios` 对象的 `run()` 方法。结果，所有线程都被添加到由 `m_ios` 对象控制的线程池中。池中的所有线程都将用于调用相应的异步操作完成回调。这意味着在多核或多处理器计算机上，多个回调可以在不同的线程中真正同时运行，每个线程在不同的处理器上；而在客户端的单线程版本中，它们将按顺序执行。

在创建每个线程后，将其指针放入 `m_threads` 列表中，以便我们以后可以访问线程对象。

此外，最后的更改是在 `close()` 方法中。在这里，我们需要连接列表中的每个线程。这是更改后的方法看起来像：

```cpp
void close() {
  // Destroy work object. This allows the I/O threads to
  // exit the event loop when there are no more pending
  // asynchronous operations. 
  m_work.reset(NULL);

  // Waiting for the I/O threads to exit.
  for (auto& thread : m_threads) {
    thread->join();
  }
}
```

在销毁了`work`对象之后，我们遍历 I/O 线程列表，并加入每个线程以确保它们都已经退出。

多线程 TCP 客户端应用程序已准备就绪。现在，当我们创建多线程`AsyncTCPClient`类的对象时，应该将指定用于处理请求的线程数量传递给类的构造函数。该类的所有其他使用方面与单线程版本相同。

## 参见

+   第二章，*I/O 操作*，包括提供详细讨论如何使用 TCP 套接字执行异步 I/O 以及如何取消异步操作的配方。

+   来自第六章的*使用计时器*配方，*其他主题*，展示了如何使用 Boost.Asio 提供的计时器。计时器可以用来实现异步操作超时机制。
