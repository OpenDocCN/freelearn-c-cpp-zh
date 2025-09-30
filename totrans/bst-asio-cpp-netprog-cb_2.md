# 第二章。I/O 操作

在本章中，我们将涵盖以下示例：

+   使用固定长度 I/O 缓冲区

+   使用可扩展的流式 I/O 缓冲区

+   同步写入 TCP 套接字

+   同步从 TCP 套接字读取

+   异步写入 TCP 套接字

+   异步从 TCP 套接字读取

+   取消异步操作

+   关闭和断开套接字

# 简介

I/O 操作是任何分布式应用程序网络基础设施中的关键操作。它们直接参与数据交换的过程。输入操作用于从远程应用程序接收数据，而输出操作允许向它们发送数据。

在本章中，我们将看到几个示例，展示如何执行 I/O 操作以及与之相关的其他操作。此外，我们还将了解如何使用 Boost.Asio 提供的一些类，这些类与 I/O 操作一起使用。

以下是对本章讨论的主题的简要总结和介绍。

## I/O 缓冲区

网络编程主要涉及在计算机网络中组织进程间通信。在此上下文中，“通信”意味着在两个或更多进程之间交换数据。从参与此类通信的进程的角度来看，该进程执行 I/O 操作，向其他参与进程发送数据并从它们那里接收数据。

与任何其他类型的 I/O 一样，网络 I/O 涉及使用内存缓冲区，这些缓冲区是在进程的地址空间中分配的连续内存块，用于存储数据。在进行任何类型的输入操作（例如，从文件、管道或通过网络远程计算机读取一些数据）时，数据到达进程，必须在它的地址空间中的某个地方存储，以便它可用于进一步处理。也就是说，当缓冲区派上用场时。在进行输入操作之前，缓冲区被分配，然后在操作期间用作数据目标点。当输入操作完成时，缓冲区包含输入数据，可以被应用程序处理。同样，在进行输出操作之前，数据必须准备并放入输出缓冲区，然后在输出操作中使用，它扮演数据源的角色。

显然，缓冲区是任何执行任何类型 I/O 的应用程序的基本组成部分，包括网络 I/O。这就是为什么对于开发分布式应用程序的开发人员来说，了解如何分配和准备 I/O 缓冲区以在 I/O 操作中使用它们至关重要。

## 同步和异步 I/O 操作

Boost.Asio 支持两种类型的 I/O 操作：同步和异步。同步操作阻塞调用它们的执行线程，并且只有在操作完成时才会解除阻塞。因此，这种类型操作的名称是同步。

第二种类型是异步操作。当异步操作被启动时，它与一个回调函数或函数对象相关联，当操作完成时，由 Boost.Asio 库调用。这些类型的 I/O 操作提供了极大的灵活性，但可能会显著复杂化代码。操作的启动简单，不会阻塞执行线程，这允许我们在异步操作在后台运行的同时使用线程来运行其他任务。

Boost.Asio 库被实现为一个框架，它利用了**控制反转**的方法。在启动一个或多个异步操作之后，应用程序将其执行线程之一交给库，然后库使用此线程来运行事件循环并调用应用程序提供的回调来通知它关于先前启动的异步操作完成的详细信息。异步操作的结果作为参数传递给回调。

## 其他操作

此外，我们还将考虑取消异步操作、关闭和关闭套接字等操作。

取消先前启动的异步操作的能力非常重要。它允许应用程序声明先前启动的操作不再相关，这可能会节省应用程序的资源（CPU 和内存），否则（如果操作继续执行，即使已知没有人再对其感兴趣）将不可避免地浪费。

关闭套接字在需要分布式应用程序的一部分通知另一部分整个消息已发送时很有用，当应用层协议没有提供其他方法来指示消息边界时。

与任何其他操作系统资源一样，当应用程序不再需要套接字时，应将其返回给操作系统。关闭操作允许我们这样做。

# 使用固定长度 I/O 缓冲区

固定长度的 I/O 缓冲区通常用于 I/O 操作，并在已知要发送或接收的消息大小时扮演数据源或目标的角色。例如，这可以是一个在栈上分配的固定长度的字符数组，其中包含要发送到服务器的请求字符串。或者，这可以是一个在空闲内存中分配的可写缓冲区，用作从套接字读取数据的数据目标点。

在这个菜谱中，我们将看到如何表示固定长度缓冲区，以便它们可以与 Boost.Asio I/O 操作一起使用。

## 如何做到这一点...

在 Boost.Asio 中，固定长度的缓冲区由以下两个类之一表示：`asio::mutable_buffer` 或 `asio::const_buffer`。这两个类都表示一个连续的内存块，该内存块由块的第一个字节的地址及其字节大小指定。正如这些类的名称所暗示的，`asio::mutable_buffer` 表示可写缓冲区，而 `asio::const_buffer` 表示只读缓冲区。

然而，在 Boost.Asio 的 I/O 函数和方法中，既不直接使用 `asio::mutable_buffer` 也不使用 `asio::const_buffer` 类。相反，引入了 `MutableBufferSequence` 和 `ConstBufferSequence` 概念。

`MutableBufferSequence` 概念指定了一个表示 `asio::mutable_buffer` 对象集合的对象。相应地，`ConstBufferSequence` 概念指定了一个表示 `asio::const_buffer` 对象集合的对象。Boost.Asio 的函数和方法在执行 I/O 操作时接受满足 `MutableBufferSequence` 或 `ConstBufferSequence` 概念要求的对象作为其参数，以表示缓冲区。

### 注意

`MutableBufferSequence` 和 `ConstBufferSequence` 概念的完整规范可在 Boost.Asio 文档部分找到，该部分可通过以下链接访问：

+   请参阅 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/MutableBufferSequence.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/MutableBufferSequence.html) 了解 `MutableBufferSequence`

+   请参阅 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ConstBufferSequence.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ConstBufferSequence.html) 了解 `ConstBufferSequence`

尽管在大多数使用情况下，单个 I/O 操作只涉及单个缓冲区，但在某些特定情况下（例如，在内存受限的环境中），开发者可能希望使用由多个较小的简单缓冲区组成的复合缓冲区，这些缓冲区分布在进程的地址空间中。Boost.Asio 的 I/O 函数和方法旨在与表示为满足 `MutableBufferSequence` 或 `ConstBufferSequence` 概念要求的缓冲区集合的复合缓冲区一起工作。

例如，`std::vector<asio::mutable_buffer>` 类的对象满足 `MutableBufferSequence` 概念的要求，因此它可以用于在 I/O 相关函数和方法中表示复合缓冲区。

因此，现在我们知道，如果我们有一个表示为 `asio::mutable_buffer` 或 `asio::const_buffer` 类对象的缓冲区，我们仍然不能使用 Boost.Asio 提供的与 I/O 相关的函数或方法。缓冲区必须表示为一个对象，满足 `MutableBufferSequence` 或 `ConstBufferSequence` 概念的要求。为此，例如，我们可以通过实例化 `std::vector<asio::mutable_buffer>` 类的对象并将我们的缓冲区对象放入其中来创建一个由单个缓冲区组成的缓冲区对象集合。现在，缓冲区成为集合的一部分，满足 `MutableBufferSequence` 要求可以在 I/O 操作中使用。

然而，尽管这种方法可以很好地创建由两个或更多简单缓冲区组成的复合缓冲区，但在处理像表示单个简单缓冲区这样的简单任务时，它看起来过于复杂，以便可以使用 Boost.Asio I/O 函数或方法。幸运的是，Boost.Asio 为我们提供了一种简化单个缓冲区与 I/O 相关函数和方法使用的方法。

`asio::buffer()` 自由函数有 28 个重载，接受各种缓冲区表示形式，并返回 `asio::mutable_buffers_1` 或 `asio::const_buffers_1` 类的对象。如果传递给 `asio::buffer()` 函数的缓冲区参数是只读类型，则函数返回 `asio::const_buffers_1` 类的对象；否则，返回 `asio::mutable_buffers_1` 类的对象。

`asio::mutable_buffers_1` 和 `asio::const_buffers_1` 类分别是 `asio::mutable_buffer` 和 `asio::const_buffer` 类的适配器。它们提供了一个满足 `MutableBufferSequence` 和 `ConstBufferSequence` 概念要求的接口和行为，这使得我们可以将这些适配器作为参数传递给 Boost.Asio I/O 函数和方法。

让我们考虑两个算法和相应的代码示例，描述了如何准备一个内存缓冲区，该缓冲区可以用于 Boost.Asio I/O 操作。第一个算法处理用于输出操作的缓冲区，第二个算法用于输入操作。

### 准备缓冲区以进行输出操作

以下算法和相应的代码示例描述了如何准备一个可以用于执行输出操作（如 `asio::ip::tcp::socket::send()` 或 `asio::write()` 自由函数）的 Boost.Asio 套接字方法的缓冲区：

1.  分配一个缓冲区。请注意，此步骤不涉及任何来自 Boost.Asio 的功能或数据类型。

1.  用要作为输出使用的数据填充缓冲区。

1.  将缓冲区表示为一个满足 `ConstBufferSequence` 概念要求的对象。

1.  缓冲区已准备好与 Boost.Asio 输出方法和函数一起使用。

假设我们想向远程应用程序发送字符串 `Hello`。在我们使用 Boost.Asio 发送数据之前，我们需要正确地表示缓冲区。以下是如何在以下代码中做到这一点的示例：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

int main()
{
  std::string buf; // 'buf' is the raw buffer. 
  buf = "Hello";   // Step 1 and 2 in single line.

  // Step 3\. Creating buffer representation that satisfies 
  // ConstBufferSequence concept requirements.
  asio::const_buffers_1 output_buf = asio::buffer(buf);

  // Step 4\. 'output_buf' is the representation of the
  // buffer 'buf' that can be used in Boost.Asio output
  // operations.

  return 0;
}
```

### 准备输入操作的缓冲区

以下算法和相应的代码示例描述了如何准备可以用于执行输入操作（如 `asio::ip::tcp::socket::receive()` 或 `asio::read()` 自由函数）的 Boost.Asio 套接字方法的缓冲区：

1.  分配一个缓冲区。缓冲区的大小必须足够大，以便容纳要接收的数据块。请注意，这一步不涉及任何来自 Boost.Asio 的功能或数据类型。

1.  使用满足 `MutableBufferSequence` 概念要求的对象来表示缓冲区。

1.  缓冲区已准备好，可以与 Boost.Asio 输入方法和函数一起使用。

假设我们想从服务器接收一块数据。为了做到这一点，我们首先需要准备一个缓冲区，数据将存储在其中。以下是如何在以下代码中做到这一点的示例：

```cpp
#include <boost/asio.hpp>
#include <iostream>
#include <memory> // For std::unique_ptr<>

using namespace boost;

int main()
{
  // We expect to receive a block of data no more than 20 bytes 
  // long. 
  const size_t BUF_SIZE_BYTES = 20;

  // Step 1\. Allocating the buffer. 
  std::unique_ptr<char[]> buf(new char[BUF_SIZE_BYTES]);

  // Step 2\. Creating buffer representation that satisfies 
  // MutableBufferSequence concept requirements.
  asio::mutable_buffers_1 input_buf =
    asio::buffer(static_cast<void*>(buf.get()),
     BUF_SIZE_BYTES);

  // Step 3\. 'input_buf' is the representation of the buffer
  // 'buf' that can be used in Boost.Asio input operations.

  return 0;
}
```

## 它是如何工作的……

这两个示例看起来都很简单直接；然而，它们包含一些细微之处，这些细微之处对于正确使用 Boost.Asio 的缓冲区非常重要。在本节中，我们将详细了解每个示例的工作原理。

### 准备输出操作的缓冲区

让我们考虑第一个代码示例，它展示了如何准备一个可以与 Boost.Asio 输出方法和函数一起使用的缓冲区。`main()` 入口函数从实例化 `std::string` 类的对象开始。因为我们想发送一段文本字符串，所以 `std::string` 是存储这类数据的良好选择。在下一行，字符串对象被赋予值 `Hello`。这就是缓冲区分配并填充数据的地方。这一行实现了算法的步骤 1 和 2。

接下来，在缓冲区可以与 Boost.Asio I/O 方法和函数一起使用之前，必须对其进行适当的表示。为了更好地理解为什么需要这样做，让我们看看一个 Boost.Asio 输出函数的例子。以下是代表 TCP 套接字的 Boost.Asio 类的 `send()` 方法的声明：

```cpp
template<typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence & buffers);
```

如我们所见，这是一个模板方法，它接受一个满足 `ConstBufferSequence` 概念要求的对象作为其参数，该参数代表缓冲区。一个合适的对象是一个复合对象，它代表 `asio::const_buffer` 类对象的集合，并提供支持对其元素进行迭代的典型集合接口。例如，`std::vector<asio::const_buffer>` 类的对象适合用作 `send()` 方法的参数，但 `std::string` 或 `asio::const_bufer` 类的对象则不适合。

为了使用我们的 `std::string` 对象与代表 TCP 套接字的类的 `send()` 方法，我们可以这样做：

```cpp
asio::const_buffer asio_buf(buf.c_str(), buf.length());
std::vector<asio::const_buffer> buffers_sequence;
buffers_sequence.push_back(asio_buf);
```

在前面的代码片段中名为 `buffer_sequence` 的对象满足 `ConstBufferSequence` 概念的要求，因此它可以作为套接字对象 `send()` 方法的参数。然而，这种方法非常复杂。相反，我们使用 Boost.Asio 提供的 `asio::buffer()` 函数来获取 *适配器* 对象，我们可以在 I/O 操作中直接使用它们：

```cpp
asio::const_buffers_1 output_buf = asio::buffer(buf);
```

在适配器对象实例化后，它可以与 Boost.Asio 输出操作一起使用，以表示输出缓冲区。

### 为输入操作准备缓冲区

第二个代码示例与第一个非常相似。主要区别在于缓冲区已分配但未填充数据，因为其目的不同。这次，缓冲区旨在在输入操作期间从远程应用程序接收数据。

使用输出缓冲区时，必须正确表示输入缓冲区，以便可以使用 Boost.Asio I/O 方法和函数。然而，在这种情况下，该缓冲区必须表示为一个满足 `MutableBufferSequence` 概念要求的对象。与 `ConstBufferSequence` 相反，这个概念表示可变缓冲区的集合，即可以写入的缓冲区。在这里，我们使用 `buffer()` 函数，它帮助我们创建所需的缓冲区表示。`mutable_buffers_1` 适配器类对象表示单个可变缓冲区，并满足 `MutableBufferSequence` 概念的要求。

在第一步中，分配了缓冲区。在这种情况下，缓冲区是在空闲内存中分配的字符数组。在下一步中，实例化了适配器对象，它可以用于输入和输出操作。

### 注意

**缓冲区所有权**

重要的是要注意，代表缓冲区的类以及我们考虑的由 Boost.Asio 提供的适配器类（即 `asio::mutable_buffer`、`asio::const_buffer`、`asio::mutable_buffers_1` 和 `asio::const_buffers_1`）都不拥有底层原始缓冲区的所有权。这些类仅提供对缓冲区的接口，并不控制其生命周期。

## 参见

+   *向 TCP 套接字同步写入* 配方演示了如何从固定长度缓冲区向套接字写入数据。

+   *从 TCP 套接字同步读取* 配方演示了如何从套接字读取数据到固定长度缓冲区。

+   第六章 中 *使用复合缓冲区进行分散/收集操作* 的配方提供了有关复合缓冲区的更多信息，并演示了如何使用它们。

# 使用可扩展的流式 I/O 缓冲区

可扩展缓冲区是当向其写入新数据时动态增加其大小的缓冲区。它们通常用于从套接字读取数据，当传入消息的大小未知时。

一些应用层协议没有定义消息的确切大小。相反，消息的边界由消息末尾的特定符号序列表示，或者由发送者在完成消息发送后发出的传输协议服务消息**文件结束**（**EOF**）表示。

例如，根据 HTTP 协议，请求和响应消息的头部部分没有固定长度，其边界由四个 ASCII 符号序列表示，即`<CR><LF><CR><LF>`，这是消息的一部分。在这种情况下，动态可扩展的缓冲区和可以与它们一起工作的函数，这些函数由 Boost.Asio 库提供，非常有用。

在这个菜谱中，我们将了解如何实例化可扩展的缓冲区以及如何向这些缓冲区读写数据。要了解这些缓冲区如何与 Boost.Asio 提供的 I/O 相关方法和函数一起使用，请参阅*另请参阅*部分中列出的专门针对 I/O 操作的相应菜谱。

## 如何做到这一点…

可扩展的流式缓冲区在 Boost.Asio 中由`asio::streambuf`类表示，它是`asio::basic_streambuf`的`typedef`：

```cpp
typedef basic_streambuf<> streambuf;
```

`asio::basic_streambuf<>`类是从`std::streambuf`继承的，这意味着它可以作为 STL 流类的流缓冲区使用。除了这一点之外，Boost.Asio 提供的几个 I/O 函数处理表示为该类对象的缓冲区。

我们可以像处理从`std::streambuf`类继承的任何流缓冲区类一样处理`asio::streambuf`类的对象。例如，我们可以将此对象分配给一个流（例如，`std::istream`、`std::ostream`或`std::iostream`，具体取决于我们的需求），然后使用流的`operator<<()`和`operator>>()`运算符向流写入和从流读取数据。

让我们考虑一个示例应用程序，其中实例化了一个`asio::streambuf`对象，向其中写入了一些数据，然后从缓冲区将数据读取回一个`std::string`类对象：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

int main()
{
  asio::streambuf buf;

  std::ostream output(&buf);

  // Writing the message to the stream-based buffer.
  output << "Message1\nMessage2";

  // Now we want to read all data from a streambuf
  // until '\n' delimiter.
  // Instantiate an input stream which uses our 
  // stream buffer.
  std::istream input(&buf);

  // We'll read data into this string.
  std::string message1;

  std::getline(input, message1);

  // Now message1 string contains 'Message1'.

  return 0;
} 
```

注意，这个示例不包含任何网络 I/O 操作，因为它专注于`asio::streambuf`类本身及其操作，而不是如何使用这个类进行 I/O 操作。

## 它是如何工作的…

`main()`应用程序入口点函数从实例化一个名为`buf`的`asio::streambuf`类对象开始。接下来，实例化`std::ostream`类的输出流对象。`buf`对象被用作输出流的*流缓冲区*。

在下一行，将`Message1\nMessage2`样本数据字符串写入输出流对象，该对象随后将数据重定向到`buf`流缓冲区。

通常，在典型的客户端或服务器应用程序中，数据将通过 Boost.Asio 输入函数（如`asio::read()`）写入`buf`流缓冲区，该函数接受一个流缓冲区对象作为参数，并从套接字读取数据到该缓冲区。

现在，我们想要从流缓冲区中读取数据。为此，我们分配一个输入流，并将`buf`对象作为流缓冲区参数传递给其构造函数。之后，我们分配一个名为`message1`的字符串对象，然后使用`std::getline`函数读取`buf`流缓冲区中当前存储的字符串的一部分，直到分隔符符号`\n`。

因此，`string1`对象包含`Message1`字符串，而`buf`流缓冲区包含分隔符符号之后的初始字符串的其余部分，即`Message2`。

## 参见

+   *异步从 TCP 套接字读取*配方演示了如何将数据从套接字读取到可扩展的流式缓冲区

# 同步写入 TCP 套接字

向 TCP 套接字写入是一个输出操作，用于将数据发送到连接到此套接字的远程应用程序。使用 Boost.Asio 提供的套接字进行同步写入是最简单的方式。执行同步写入到套接字的方法会阻塞执行线程，直到数据（至少一些数据）被写入套接字或发生错误才会返回。

在本配方中，我们将了解如何同步地将数据写入 TCP 套接字。

## 如何做到这一点...

使用 Boost.Asio 库提供的套接字的最基本方式是使用`asio::ip::tcp::socket`类的`write_some()`方法。以下是该方法重载之一的声明：

```cpp
template<
typename ConstBufferSequence>
std::size_t write_some(
const ConstBufferSequence & buffers);
```

此方法接受一个表示复合缓冲区的对象作为参数，正如其名称所暗示的，从缓冲区写入*一些*数据到套接字。如果方法成功，返回值指示写入的字节数。这里要强调的是，该方法可能*不会*发送通过`buffers`参数提供的所有数据。该方法仅保证在没有错误发生的情况下至少写入一个字节。这意味着，在一般情况下，为了将缓冲区中的所有数据写入套接字，我们可能需要多次调用此方法。

以下算法描述了在分布式应用程序中同步写入 TCP 套接字所需的步骤：

1.  在客户端应用程序中，分配、打开和连接一个活动 TCP 套接字。在服务器应用程序中，通过使用接受器套接字接受连接请求来获取一个已连接的活动 TCP 套接字。

1.  分配缓冲区并填充要写入套接字的数据。

1.  在循环中，根据需要多次调用套接字的`write_some()`方法，以发送缓冲区中所有可用的数据。

以下代码示例演示了一个客户端应用程序，它根据该算法操作：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

void writeToSocket(asio::ip::tcp::socket& sock) {
  // Step 2\. Allocating and filling the buffer.
  std::string buf = "Hello";

  std::size_t total_bytes_written = 0;

  // Step 3\. Run the loop until all data is written
  // to the socket.
  while (total_bytes_written != buf.length()) {
    total_bytes_written += sock.write_some(
      asio::buffer(buf.c_str() +
      total_bytes_written,
      buf.length() - total_bytes_written));
  }
}

int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

// Step 1\. Allocating and opening the socket.
    asio::ip::tcp::socket sock(ios, ep.protocol());

    sock.connect(ep);

    writeToSocket(sock);
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

尽管在所提供的代码示例中，写入套接字是在充当客户端的应用程序上下文中执行的，但可以使用相同的方法在服务器应用程序中写入套接字。

## 它是如何工作的…

`main()`应用程序入口点函数相当简单。它分配一个套接字，打开，并将其同步连接到远程应用程序。然后，调用`writeToSocket()`函数，并将套接字对象作为参数传递给它。此外，`main()`函数包含一个`try-catch`块，旨在捕获和处理 Boost.Asio 方法和函数可能抛出的异常。

样本中有趣的部分是执行同步写入套接字的`writeToSocket()`函数。它接受一个套接字对象的引用作为参数。它的前提条件是传递给它的套接字已经连接；否则，函数将失败。

函数开始于分配和填充缓冲区。在这个示例中，我们使用 ASCII 字符串作为要写入套接字的数据，因此我们分配了一个`std::string`类的对象，并给它赋值为`Hello`，我们将使用这个作为将要写入套接字的占位符消息。

然后，定义了一个名为`total_bytes_written`的变量，并将其值设置为`0`。这个变量用作计数器，用于存储已经写入套接字的字节数。

接下来，运行一个循环，在该循环中调用套接字的`write_some()`方法。除了缓冲区为空的情况（即`buf.length()`方法返回值为`0`）之外，至少执行一次循环迭代，并且至少调用一次`write_some()`方法。让我们更仔细地看看这个循环：

```cpp
  while (total_bytes_written != buf.length()) {
    total_bytes_written += sock.write_some(
      asio::buffer(buf.c_str() +
      total_bytes_written,
      buf.length() - total_bytes_written));
  }
```

当`total_bytes_written`变量的值等于缓冲区的大小时，终止条件评估为`true`，即当缓冲区中可用的所有字节都已被写入套接字时。在循环的每次迭代中，`total_bytes_written`变量的值都会增加`write_some()`方法返回的值，这个值等于在此方法调用期间写入的字节数。

每次调用`write_some()`方法时，传递给它的参数都会调整。与原始缓冲区相比，缓冲区的起始字节根据`total_bytes_written`的值进行偏移（因为前面的`write_some()`方法调用已经发送了前面的字节），并且缓冲区的大小相应地减少相同的值。

在循环终止后，缓冲区中的所有数据都被写入套接字，并且`writeToSocket()`函数返回。

值得注意的是，在单次调用 `write_some()` 方法期间写入套接字的字节数取决于几个因素。在一般情况下，开发者并不知道这一点；因此，不应将其计入考虑。一个演示的解决方案与此值无关，并且根据需要多次调用 `write_some()` 方法，将缓冲区中所有可用的数据写入套接字。

### 选项 - send() 方法

`asio::ip::tcp::socket` 类包含另一个同步将数据写入套接字的方法，名为 `send()`。此方法有三个重载。其中之一与前面描述的 `write_some()` 方法等效。它具有完全相同的签名，并提供了完全相同的功能。在某种意义上，这些方法是同义词。

第二个重载与 `write_some()` 方法相比接受一个额外的参数。让我们看看它：

```cpp
template<
typename ConstBufferSequence>
std::size_t send(
    const ConstBufferSequence & buffers,
    socket_base::message_flags flags);
```

这个额外的参数被命名为 `flags`。它可以用来指定一个位掩码，表示控制操作的标志。因为这些标志使用得相当少，所以我们不会在本书中考虑它们。有关此主题的更多信息，请参阅 Boost.Asio 文档。

第三个重载与第二个重载等效，但在失败的情况下不会抛出异常。相反，错误信息通过一个额外的 `boost::system::error_code` 类型的方法输出参数返回。

## 还有更多...

使用套接字的 `write_some()` 方法向套接字写入数据对于这样一个简单的操作来说似乎非常复杂。即使我们只想发送由几个字节组成的小消息，我们也必须使用循环、一个变量来跟踪已经写入的字节数，并在循环的每次迭代中正确构造一个缓冲区。这种方法容易出错，并使代码更难以理解。

幸运的是，Boost.Asio 提供了一个免费函数，简化了向套接字写入的过程。这个函数被称为 `asio::write()`。让我们看看它的一种重载形式：

```cpp
template<
    typename SyncWriteStream,
    typename ConstBufferSequence>
std::size_t write(
    SyncWriteStream & s,
    const ConstBufferSequence & buffers);
```

此函数接受两个参数。第一个参数名为 `s` 是一个满足 `SyncWriteStream` 概念要求的对象的引用。关于要求列表的完整信息，请参阅相应的 Boost.Asio 文档部分，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncWriteStream.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncWriteStream.html)。表示 TCP 套接字的 `asio::ip::tcp::socket` 类的对象满足这些要求，因此可以用作函数的第一个参数。第二个参数名为 `buffers` 表示缓冲区（简单或复合），并包含要写入套接字的数据。

与套接字对象的`write_some()`方法不同，后者从缓冲区写入*一些*数据到套接字，`asio::write()`函数将缓冲区中所有可用的数据写入套接字。这简化了套接字的写入操作，并使代码更短更干净。

如果我们使用`asio::write()`函数而不是套接字对象的`write_some()`方法来向套接字写入数据，那么我们之前示例中的`writeToSocket()`函数将看起来是这样的：

```cpp
void writeToSocketEnhanced(asio::ip::tcp::socket& sock) {
  // Allocating and filling the buffer.
  std::string buf = "Hello";

  // Write whole buffer to the socket.
  asio::write(sock, asio::buffer(buf));
}
```

`asio::write()`函数的实现方式与原始的`writeToSocket()`函数通过在循环中对套接字对象的`write_some()`方法进行多次调用而实现的方式相似。

### 注意

注意，`asio::write()`函数在刚刚考虑的函数之上还有七个重载。其中一些可能在特定情况下非常有用。请参阅 Boost.Asio 文档以了解更多关于此函数的信息，请参阅[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/write.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/write.html)。

## 参见

+   在第三章的*实现同步 TCP 客户端*配方中，*实现客户端应用程序*展示了如何实现一个同步 TCP 客户端，该客户端执行同步写入以向服务器发送请求消息

+   在第四章的*实现同步迭代 TCP 服务器*配方中，*实现服务器应用程序*展示了如何实现一个同步 TCP 服务器，该服务器执行同步写入以向客户端发送响应消息

# 从 TCP 套接字中同步读取

从 TCP 套接字中读取是一个输入操作，用于接收连接到此套接字的远程应用程序发送的数据。同步读取是使用 Boost.Asio 提供的套接字接收数据的简单方法。执行同步读取的方法和函数会阻塞执行线程，直到从套接字中读取数据（至少一些数据）或发生错误才会返回。

在这个配方中，我们将看到如何从 TCP 套接字中同步读取数据。

## 如何做到这一点...

使用 Boost.Asio 库提供的套接字读取数据的最基本方式是`asio::ip::tcp::socket`类的`read_some()`方法。让我们看看这个方法的一个重载版本：

```cpp
template<
typename MutableBufferSequence>
std::size_t read_some(
    const MutableBufferSequence & buffers);
```

此方法接受一个表示可写缓冲区（单个或组合）的对象作为参数，正如其名称所暗示的，从套接字读取一定量的数据到缓冲区。如果方法成功，返回值表示读取的字节数。需要注意的是，无法控制方法将读取多少字节。该方法仅保证如果没有发生错误，至少会读取一个字节。这意味着，在一般情况下，为了从套接字读取一定量的数据，我们可能需要多次调用该方法。

以下算法描述了在分布式应用程序中同步从 TCP 套接字读取数据所需的步骤：

1.  在客户端应用程序中，分配、打开并连接一个活动 TCP 套接字。在服务器应用程序中，通过使用接受器套接字接受连接请求来获取一个已连接的活动 TCP 套接字。

1.  分配一个足够大的缓冲区，以便能够容纳要读取的预期消息。

1.  在循环中，根据需要多次调用套接字的`read_some()`方法来读取消息。

以下代码示例演示了一个客户端应用程序，该应用程序按照以下算法操作：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

std::string readFromSocket(asio::ip::tcp::socket& sock) {
  const unsigned char MESSAGE_SIZE = 7;
  char buf[MESSAGE_SIZE];
  std::size_t total_bytes_read = 0;

  while (total_bytes_read != MESSAGE_SIZE) {
    total_bytes_read += sock.read_some(
      asio::buffer(buf + total_bytes_read,
      MESSAGE_SIZE - total_bytes_read));
  }

  return std::string(buf, total_bytes_read);
}

int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

    asio::ip::tcp::socket sock(ios, ep.protocol());

    sock.connect(ep);

    readFromSocket(sock);
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

尽管在所提供的代码示例中，从套接字读取是在充当客户端的应用程序上下文中执行的，但同样的方法也可以用于在服务器应用程序中从套接字读取数据。

## 它是如何工作的...

`main()`应用程序入口点函数相当简单。首先，它分配一个 TCP 套接字，打开并同步将其连接到远程应用程序。然后，调用`readFromSocket()`函数，并将套接字对象作为参数传递给它。此外，`main()`函数包含一个`try-catch`块，旨在捕获和处理 Boost.Asio 方法和函数可能抛出的异常。

样本中的有趣部分是`readFromSocket()`函数，它执行从套接字的同步读取。它接受套接字对象的引用作为输入参数。它的前提是传递给它的作为参数的套接字必须是连接的；否则，函数将失败。

函数开始时分配一个名为`buf`的缓冲区。缓冲区的大小被选择为 7 字节。这是因为在我们这个示例中，我们期望从远程应用程序接收一个正好 7 字节长的消息。

然后，定义一个名为`total_bytes_read`的变量，并将其值设置为`0`。该变量用作计数器，用于记录从套接字读取的总字节数。

接下来，运行循环，在其中调用套接字的`read_some()`方法。让我们更详细地看看这个循环：

```cpp
  while (total_bytes_read != MESSAGE_SIZE) {
    total_bytes_read += sock.read_some(
      asio::buffer(buf + total_bytes_read,
      MESSAGE_SIZE - total_bytes_read));
  }
```

当 `total_bytes_read` 变量的值等于预期消息的大小，即整个消息已从套接字中读取时，终止条件评估为 `true`。在循环的每次迭代中，`total_bytes_read` 变量的值会增加 `read_some()` 方法返回的值，该值等于在此方法调用期间读取的字节数。

每次调用 `read_some()` 方法时，传递给它的输入缓冲区都会进行调整。与原始缓冲区相比，缓冲区的起始字节会根据 `total_bytes_read` 的值进行偏移（因为缓冲区的先前部分已经在前几次调用 `read_some()` 方法时用从套接字读取的数据填充），并且缓冲区的大小相应地减少相同的值。

循环结束后，现在缓冲区中包含了从套接字中预期读取的所有数据。

`readFromSocket()` 函数以从接收到的缓冲区中实例化 `std::string` 类的对象并返回给调用者结束。

值得注意的是，在单次调用 `read_some()` 方法时，从套接字中读取的字节数取决于多个因素。在一般情况下，这并不为开发者所知；因此，不应将其考虑在内。所提出的解决方案与此值无关，并且根据需要多次调用 `read_some()` 方法以从套接字中读取所有数据。

### 替代方案 – `receive()` 方法

`asio::ip::tcp::socket` 类包含另一个从套接字同步读取数据的方法，称为 `receive()`。此方法有三个重载。其中之一与前面描述的 `read_some()` 方法等效。它具有完全相同的签名，并提供了完全相同的功能。在某种意义上，这些方法是同义词。

与 `read_some()` 方法相比，第二个重载方法多接受一个额外的参数。让我们来看看它：

```cpp
template<
    typename MutableBufferSequence>
std::size_t receive(
    const MutableBufferSequence & buffers,
    socket_base::message_flags flags);
```

这个额外的参数被命名为 `flags`。它可以用来指定一个位掩码，表示控制操作的标志。因为这些标志很少使用，所以我们不会在本书中考虑它们。有关此主题的更多信息，请参阅 Boost.Asio 文档。

第三个重载与第二个重载等效，但在失败的情况下不会抛出异常。相反，错误信息通过 `boost::system::error_code` 类型的额外输出参数返回。

## 还有更多...

使用套接字的 `read_some()` 方法从套接字读取数据对于这样一个简单的操作来说似乎非常复杂。这种方法要求我们使用循环、一个变量来跟踪已经读取的字节数，并且为循环的每次迭代正确构造一个缓冲区。这种方法容易出错，并使代码更难以理解和维护。

幸运的是，Boost.Asio 提供了一系列免费函数，这些函数在不同的上下文中简化了从套接字同步读取数据。有三个这样的函数，每个函数都有几个重载版本，提供了丰富的功能，有助于从套接字读取数据。

### The asio::read() function

`asio::read()` 函数是三个函数中最简单的一个。让我们看看其中一个重载的声明：

```cpp
template<
    typename SyncReadStream,
    typename MutableBufferSequence>
std::size_t read(
    SyncReadStream & s,
    const MutableBufferSequence & buffers);
```

此函数接受两个参数。第一个参数名为 `s`，是一个满足 `SyncReadStream` 概念要求的对象的引用。关于要求完整列表，请参阅在 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html) 可用的相应 Boost.Asio 文档部分。表示 TCP 套接字的 `asio::ip::tcp::socket` 类的对象满足这些要求，因此可以用作函数的第一个参数。第二个参数名为 `buffers`，表示一个缓冲区（简单或复合），数据将从套接字读取到该缓冲区。

与从套接字读取到缓冲区的“某些”数据量的 `read_some()` 方法相比，`asio::read()` 函数在单次调用期间从套接字读取数据，直到传递给它的作为参数的缓冲区被填满或发生错误。这简化了从套接字读取的过程，并使代码更短更整洁。

如果我们使用 `asio::read()` 函数而不是套接字对象的 `read_some()` 方法来从套接字读取数据，那么前面的示例中的 `readFromSocket()` 函数将看起来像这样：

```cpp
std::string readFromSocketEnhanced(asio::ip::tcp::socket& sock) {
  const unsigned char MESSAGE_SIZE = 7;
  char buf[MESSAGE_SIZE];

  asio::read(sock, asio::buffer(buf, MESSAGE_SIZE));

  return std::string(buf, MESSAGE_SIZE);
}
```

在前面的示例中，对 `asio::read()` 函数的调用将阻塞执行线程，直到恰好读取了 7 个字节或发生错误。与套接字的 `read_some()` 方法相比，这种方法的优势是显而易见的。

### 注意

`asio::read()` 函数有几个重载版本，在特定上下文中提供了灵活性。有关此函数的更多信息，请参阅 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read.html) 的相应 Boost.Asio 文档部分。

### asio::read_until() 函数

`asio::read_until()` 函数提供了一种从套接字读取数据的方法，直到在数据中遇到指定的模式。该函数有八个重载版本。让我们考虑其中之一：

```cpp
template<
    typename SyncReadStream,
    typename Allocator>
std::size_t read_until(
    SyncReadStream & s,
    boost::asio::basic_streambuf< Allocator > & b,
    char delim);
```

此函数接受三个参数。第一个参数名为 `s` 是一个满足 `SyncReadStream` 概念要求的对象的引用。有关要求列表的完整信息，请参阅相应的 Boost.Asio 文档部分，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html)。`asio::ip::tcp::socket` 类的对象代表一个 TCP 套接字，它满足这些要求，因此可以用作函数的第一个参数。

第二个参数名为 `b` 代表一个面向流的可扩展缓冲区，数据将从中读取。最后一个参数名为 `delim` 指定一个分隔符字符。

`asio::read_until()` 函数将从 `s` 套接字读取数据到缓冲区 `b`，直到遇到由 `delim` 参数指定的字符，该字符位于数据的读取部分。当遇到指定的字符时，函数返回。

需要注意的是，`asio::read_until()` 函数的实现方式是按变量大小的块从套接字读取数据（内部使用套接字的 `read_some()` 方法读取数据）。当函数返回时，缓冲区 `b` 可能包含分隔符符号之后的某些符号。这可能发生在远程应用程序在分隔符符号之后发送更多数据的情况下（例如，它可能连续发送两条消息，每条消息的末尾都有一个分隔符符号）。换句话说，当 `asio::read_until()` 函数成功返回时，可以保证缓冲区 `b` 至少包含一个分隔符符号，但可能包含更多。解析缓冲区中的数据并处理包含分隔符符号之后数据的情形是开发者的责任。

如果我们想要从套接字读取所有数据直到遇到特定符号，我们将这样实现我们的 `readFromSocket()` 函数。假设消息分隔符为换行 ASCII 符号，`\n`：

```cpp
std::string readFromSocketDelim(asio::ip::tcp::socket& sock) {
  asio::streambuf buf;

  // Synchronously read data from the socket until
  // '\n' symbol is encountered.  
  asio::read_until(sock, buf, '\n');

  std::string message;

  // Because buffer 'buf' may contain some other data
  // after '\n' symbol, we have to parse the buffer and
  // extract only symbols before the delimiter. 

  std::istream input_stream(&buf);
  std::getline(input_stream, message);
  return message;
}
```

此示例相当简单直接。因为 `buf` 可能包含分隔符符号之后的更多符号，我们使用 `std::getline()` 函数提取分隔符符号之前感兴趣的消息，并将它们放入 `message` 字符串对象中，然后将其返回给调用者。

### 注意

`read_until()` 函数有几个重载版本，提供了更复杂的方式来指定终止条件，例如字符串分隔符、正则表达式或函数对象。有关此主题的更多信息，请参阅相应的 Boost.Asio 文档部分，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_until.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_until.html)。

### asio::read_at() 函数

`asio::read_at()` 函数提供了一种从套接字读取数据的方法，从特定的偏移量开始。由于此函数很少使用，它超出了本书的范围。有关此函数及其重载的更多详细信息，请参阅相应的 Boost.Asio 文档部分，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_at.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_at.html)。

`asio::read()`、`asio::read_until()` 和 `asio::read_at()` 函数的实现方式与我们的示例中通过在循环中对套接字对象的 `read_some()` 方法进行多次调用直到满足终止条件或发生错误来实现的原始 `readFromSocket()` 函数类似。

## 参见

+   *使用可扩展的流式 I/O 缓冲区* 菜单展示了如何向 `asio::streambuf` 缓冲区写入和读取数据

+   在 第三章 的 *实现客户端应用程序* 菜单中，*实现同步 TCP 客户端* 菜单演示了如何实现一个同步 TCP 客户端，该客户端从套接字同步读取以接收服务器发送的响应消息

+   在 第四章 的 *实现服务器应用程序* 菜单中，*实现同步迭代 TCP 服务器* 菜单演示了如何实现一个同步 TCP 服务器，该服务器执行同步读取以接收客户端发送的请求消息

# 异步写入 TCP 套接字

异步写入是一种灵活且高效地向远程应用程序发送数据的方式。在本菜谱中，我们将看到如何异步写入 TCP 套接字。

## 如何操作…

在 Boost.Asio 库提供的套接字上异步写入数据的最基本工具是 `asio::ip::tcp::socket` 类的 `async_write_some()` 方法。让我们看看该方法的一个重载：

```cpp
template<
    typename ConstBufferSequence,
    typename WriteHandler>
void async_write_some(
    const ConstBufferSequence & buffers,
    WriteHandler handler);
```

此方法启动写入操作并立即返回。它接受一个表示要写入套接字的数据的缓冲区的对象作为其第一个参数。第二个参数是一个回调，当启动的操作完成时，Boost.Asio 将调用它。此参数可以是函数指针、仿函数或满足 `WriteHandler` 概念要求的任何其他对象。完整的要求列表可以在 Boost.Asio 文档的相应部分找到，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/WriteHandler.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/WriteHandler.html)。

回调应具有以下签名：

```cpp
void write_handler(
    const boost::system::error_code& ec,
    std::size_t bytes_transferred);
```

在这里，`ec` 是一个参数，如果发生错误，则表示错误代码，而 `bytes_transferred` 参数表示在相应的异步操作期间写入套接字的字节数。

如`async_write_some()`方法的名字所暗示的，它启动一个操作，目的是从缓冲区向套接字写入*一些*数据。此方法保证在相应的异步操作中如果没有发生错误，至少将写入一个字节。这意味着，在一般情况下，为了将缓冲区中所有可用的数据写入套接字，我们可能需要执行此异步操作多次。

现在我们知道了关键方法是如何工作的，让我们看看如何实现一个执行异步套接字写入的应用程序。

以下算法描述了执行和实现写入 TCP 套接字异步数据的应用程序所需的步骤。请注意，此算法提供了一个*可能*实现此类应用程序的方法。Boost.Asio 非常灵活，允许我们通过以多种不同的方式异步写入套接字数据来组织和结构化应用程序：

1.  定义一个包含指向套接字对象的指针、缓冲区和用作已写入字节数计数器的变量的数据结构。

1.  定义一个回调函数，当异步写入操作完成时将被调用。

1.  在客户端应用程序中，分配并打开一个活跃的 TCP 套接字并将其连接到远程应用程序。在服务器应用程序中，通过接受连接请求来获取一个已连接的活跃 TCP 套接字。

1.  分配一个缓冲区并填充要写入套接字的数据。

1.  通过调用套接字的`async_write_some()`方法来启动异步写入操作。指定在第 2 步中定义的函数作为回调。

1.  在`asio::io_service`类的对象上调用`run()`方法。

1.  在回调中增加已写入字节数的计数器。如果已写入的字节数少于要写入的总字节数，则启动一个新的异步写入操作来写入数据的下一部分。

让我们实现一个示例客户端应用程序，该应用程序根据前面的算法执行异步写入。

我们从添加`include`和`using`指令开始：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;
```

接下来，根据算法的第 1 步，我们定义一个包含指向套接字对象的指针、包含要写入的数据的缓冲区以及包含已写入字节数的计数器变量的数据结构：

```cpp
// Keeps objects we need in a callback to
// identify whether all data has been written
// to the socket and to initiate next async
// writing operation if needed.
struct Session {
  std::shared_ptr<asio::ip::tcp::socket> sock;
  std::string buf;
  std::size_t total_bytes_written;
};
```

在第 2 步中，我们定义了一个回调函数，当异步操作完成时将被调用：

```cpp
// Function used as a callback for 
// asynchronous writing operation.
// Checks if all data from the buffer has
// been written to the socket and initiates
// new asynchronous writing operation if needed.
void callback(const boost::system::error_code& ec,
        std::size_t bytes_transferred,
        std::shared_ptr<Session> s) 
{
  if (ec != 0) {
    std::cout << "Error occured! Error code = " 
    << ec.value()
    << ". Message: " << ec.message();

    return;
  }

  s->total_bytes_written += bytes_transferred;

  if (s->total_bytes_written == s->buf.length()) {
    return;
  }

  s->sock->async_write_some(
  asio::buffer(
  s->buf.c_str() + 
  s->total_bytes_written, 
  s->buf.length() - 
  s->total_bytes_written),
  std::bind(callback, std::placeholders::_1,
  std::placeholders::_2, s));
}
```

现在，我们先跳过第 3 步，并在一个单独的函数中实现第 4 步和第 5 步。让我们把这个函数叫做`writeToSocket()`：

```cpp
void writeToSocket(std::shared_ptr<asio::ip::tcp::socket> sock) {

  std::shared_ptr<Session> s(new Session);

  // Step 4\. Allocating and filling the buffer.
  s->buf = std::string("Hello");
  s->total_bytes_written = 0;
  s->sock = sock;

  // Step 5\. Initiating asynchronous write operation.
  s->sock->async_write_some(
  asio::buffer(s->buf),
  std::bind(callback, 
  std::placeholders::_1,
  std::placeholders::_2, 
  s));
}
```

现在，我们回到第 3 步，并在`main()`应用程序入口点函数中实现它：

```cpp
int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

    // Step 3\. Allocating, opening and connecting a socket.
    std::shared_ptr<asio::ip::tcp::socket> sock(
    new asio::ip::tcp::socket(ios, ep.protocol()));

    sock->connect(ep);

    writeToSocket(sock);

    // Step 6.
    ios.run();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的…

现在，让我们追踪应用程序的执行路径，以便更好地理解它是如何工作的。

应用程序由单个线程运行，在此上下文中调用应用程序的`main()`入口点函数。请注意，Boost.Asio 可能会为某些内部操作创建额外的线程，但它保证不会在这些线程的上下文中执行任何应用程序代码。

`main()`函数分配、打开并同步地将套接字连接到远程应用程序，然后通过传递套接字对象的指针调用`writeToSocket()`函数。此函数启动异步写入操作并返回。我们稍后将考虑此函数。`main()`函数继续调用`asio::io_service`类对象的`run()`方法，其中 Boost.Asio 捕获执行线程，并在异步操作完成时使用它来调用相关的回调函数。

`asio::os_service::run()`方法在至少有一个挂起的异步操作时阻塞。当最后一个挂起的异步操作的最后一个回调完成时，此方法返回。

现在，让我们回到`writeToSocket()`函数并分析其行为。它首先在空闲内存中分配`Session`数据结构的一个实例。然后，它分配并填充缓冲区，以包含要写入套接字的数据。之后，将套接字对象的指针和缓冲区存储在`Session`对象中。由于套接字的`async_write_some()`方法可能不会一次性将所有数据写入套接字，我们可能需要在回调函数中启动另一个异步写入操作。这就是为什么我们需要`Session`对象，并且我们将其分配在空闲内存中而不是栈上；它必须*存活*直到回调函数被调用。

最后，我们启动异步操作，调用套接字对象的`async_write_some()`方法。此方法的调用相对复杂，因此让我们更详细地考虑这一点：

```cpp
s->sock->async_write_some(
  asio::buffer(s->buf),
  std::bind(callback,
     std::placeholders::_1,
std::placeholders::_2, 
s));
```

第一个参数是包含要写入套接字的数据的缓冲区。由于操作是异步的，Boost.Asio 可能在操作启动和回调调用之间的任何时刻访问此缓冲区。这意味着缓冲区必须保持完整，并且必须在回调调用之前可用。我们通过将缓冲区存储在`Session`对象中，而`Session`对象又存储在空闲内存中，来保证这一点。

第二个参数是在异步操作完成后要调用的回调。Boost.Asio 将回调定义为一种*概念*，它可以是一个函数或函数对象，接受两个参数。回调的第一个参数指定在操作执行过程中发生的错误（如果有）。第二个参数指定操作写入的字节数。

因为我们想向回调函数传递一个额外的参数，即指向相应 `Session` 对象的指针，该对象作为操作的上下文，所以我们使用 `std::bind()` 函数构建一个函数对象，并将指向 `Session` 对象的指针作为第三个参数附加到该对象上。然后，将这个函数对象作为回调参数传递给套接字对象的 `async_write_some()` 方法。

由于它是异步的，`async_write_some()` 方法不会阻塞执行线程。它启动写入操作并返回。

实际的写入操作由 Boost.Asio 库和底层操作系统在幕后执行，当操作完成或发生错误时，会调用回调函数。

当被调用时，名为 `callback` 的回调函数（在我们的示例应用程序中直接称为 `callback`）首先检查操作是否成功或发生错误。在后一种情况下，错误信息会被输出到标准输出流，并且函数返回。否则，总写入字节数会增加由操作产生的字节数。然后，我们检查写入套接字的总字节数是否等于缓冲区的大小。如果这些值相等，这意味着所有数据都已写入套接字，没有更多的工作要做。回调函数返回。然而，如果缓冲区中仍有要写入的数据，则会启动一个新的异步写入操作：

```cpp
s->sock->async_write_some(
asio::buffer(
s->buf.c_str() + 
s->total_bytes_written, 
s->buf.length() – 
s->total_bytes_written),
std::bind(callback, std::placeholders::_1,
std::placeholders::_2, s));
```

注意缓冲区开始部分是如何根据已写入的字节数进行偏移的，以及缓冲区大小相应地减少了多少。

作为回调，我们使用 `std::bind()` 函数指定相同的 `callback()` 函数，并附加一个额外的参数——`Session` 对象，就像我们在启动第一个异步操作时做的那样。

异步写入操作的启动和后续回调调用的周期会重复，直到缓冲区中的所有数据都写入套接字或发生错误。

当 `callback` 函数返回而不启动新的异步操作时，在 `main()` 函数中调用的 `asio::io_service::run()` 方法会解除执行线程的阻塞并返回。`main()` 函数也会返回。这时，应用程序退出。

## 还有更多...

尽管前面示例中描述的 `async_write_some()` 方法允许异步地将数据写入套接字，但基于它的解决方案相对复杂且容易出错。幸运的是，Boost.Asio 提供了一种更方便的方法，使用自由函数 `asio::async_write()` 来异步写入套接字数据。让我们考虑它的一个重载版本：

```cpp
template<
    typename AsyncWriteStream,
    typename ConstBufferSequence,
    typename WriteHandler>
void async_write(
    AsyncWriteStream & s,
    const ConstBufferSequence & buffers,
    WriteHandler handler);
```

此函数与套接字的`async_write_some()`方法非常相似。其第一个参数是一个满足`AsyncWriteStream`概念要求的对象。关于要求完整列表，请参阅相应的 Boost.Asio 文档部分，[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncWriteStream.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncWriteStream.html)。`asio::ip::tcp::socket`类的对象满足这些要求，因此可以与该函数一起使用。

`asio::async_write()`函数的第二个和第三个参数与前面示例中描述的 TCP 套接字对象的`async_write_some()`方法的第一个和第二个参数类似。这些参数是包含要写入的数据的缓冲区，以及表示回调的函数或对象，当操作完成时将被调用。

与套接字的`async_write_some()`方法不同，后者启动从缓冲区到套接字的写入操作，写入*一些*数据量，而`asio::async_write()`函数启动的操作则是写入缓冲区中所有可用的数据。在这种情况下，回调函数仅在缓冲区中所有数据都写入套接字或发生错误时被调用。这简化了套接字的写入操作，并使代码更短更整洁。

如果我们将之前的示例修改为使用`asio::async_write()`函数而不是套接字对象的`async_write_some()`方法来异步写入套接字数据，那么我们的应用程序将变得更加简单。

首先，我们不需要跟踪写入套接字字节数，因此，`Session`结构变得更小：

```cpp
struct Session {
  std::shared_ptr<asio::ip::tcp::socket> sock;
  std::string buf;
}; 
```

其次，我们知道当回调函数被调用时，这意味着缓冲区中的所有数据都已写入套接字或发生了错误。这使得回调函数变得更加简单：

```cpp
void callback(const boost::system::error_code& ec,
  std::size_t bytes_transferred,
  std::shared_ptr<Session> s)
{
  if (ec != 0) {
    std::cout << "Error occured! Error code = "
      << ec.value()
      << ". Message: " << ec.message();

    return;
  }

  // Here we know that all the data has
  // been written to the socket.
}
```

`asio::async_write()`函数是通过零个或多个对套接字对象的`async_write_some()`方法的调用实现的。这类似于我们初始示例中的`writeToSocket()`函数的实现。

### 注意

注意，`asio::async_write()`函数有三个额外的重载，提供了额外的功能。在某些特定情况下，其中一些可能非常有用。有关此函数的更多信息，请参阅 Boost.Asio 文档，[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_write.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_write.html)。

## 参见

+   *同步写入 TCP 套接字*配方描述了如何同步地将数据写入 TCP 套接字

+   在第三章*实现客户端应用程序*中的*实现异步 TCP 客户端*配方（ch03.html "第三章。实现客户端应用程序"），展示了如何实现一个异步 TCP 客户端，该客户端执行异步写入 TCP 套接字以向服务器发送请求消息。

+   在第四章*实现服务器应用程序*中的*实现异步 TCP 服务器*配方（ch04.html "第四章。实现服务器应用程序"），展示了如何实现一个异步 TCP 服务器，该服务器执行异步写入 TCP 套接字以向客户端发送响应消息。

# 异步从 TCP 套接字读取

异步读取是一种灵活且高效地从远程应用程序接收数据的方式。在本配方中，我们将了解如何从 TCP 套接字异步读取数据。

## 如何做到这一点...

Boost.Asio 库提供的用于异步从 TCP 套接字读取数据的最基本工具是`asio::ip::tcp::socket`类的`async_read_some()`方法。以下是该方法的一个重载示例：

```cpp
template<
    typename MutableBufferSequence,
    typename ReadHandler>
void async_read_some(
    const MutableBufferSequence & buffers,
    ReadHandler handler);
```

此方法启动一个异步读取操作并立即返回。它接受一个表示可变缓冲区的对象作为其第一个参数，数据将从套接字读取到该对象中。第二个参数是 Boost.Asio 在操作完成时调用的回调。此参数可以是函数指针、仿函数或满足`ReadHandler`概念要求的任何其他对象。完整的要求列表可以在 Boost.Asio 文档的相应部分中找到，请参阅[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ReadHandler.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ReadHandler.html)。

回调应具有以下签名：

```cpp
void read_handler(
    const boost::system::error_code& ec,
    std::size_t bytes_transferred);
```

在这里，`ec`是一个参数，如果发生错误，则通知错误代码，而`bytes_transferred`参数指示在相应的异步操作期间从套接字中读取了多少字节。

如`async_read_some()`方法的名字所暗示的，它启动一个操作，目的是从套接字到缓冲区读取*一些*数据。如果未发生错误，此方法保证在相应的异步操作期间至少读取一个字节。这意味着，在一般情况下，为了从套接字读取所有数据，我们可能需要执行此异步操作多次。

既然我们已经了解了关键方法的工作原理，让我们看看如何实现一个从套接字执行异步读取的应用程序。

以下算法描述了实现一个从套接字异步读取数据的应用程序所需的步骤。请注意，此算法提供了一个*可能*的实现此类应用程序的方法。Boost.Asio 非常灵活，允许我们通过以不同方式从套接字异步读取数据来组织和结构化应用程序：

1.  定义一个包含指向套接字对象的指针、一个缓冲区、一个定义缓冲区大小的变量以及一个用作读取的字节数计数器的变量的数据结构。

1.  定义一个回调函数，当异步读取操作完成时将被调用。

1.  在客户端应用程序中，分配并打开一个活动 TCP 套接字，然后将其连接到远程应用程序。在服务器应用程序中，通过接受连接请求来获取一个已连接的活动 TCP 套接字。

1.  分配一个足够大的缓冲区，以便预期的消息可以容纳。

1.  通过调用套接字的`async_read_some()`方法并指定步骤 2 中定义的函数作为回调来启动异步读取操作。

1.  在`asio::io_service`类的对象上调用`run()`方法。

1.  在回调中，增加读取的字节数计数器。如果读取的字节数少于要读取的总字节数（即预期消息的大小），则启动一个新的异步读取操作以读取下一部分数据。

让我们实现一个示例客户端应用程序，该程序将根据前面的算法执行异步读取。

我们从添加`include`和`using`指令开始：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;
```

接下来，根据步骤 1，我们定义一个包含名为`sock`的套接字对象指针、名为`buf`的缓冲区指针、名为`buf_size`的变量（包含缓冲区大小）以及包含已读取的字节数的`total_bytes_read`变量的数据结构：

```cpp
// Keeps objects we need in a callback to
// identify whether all data has been read
// from the socket and to initiate next async
// reading operation if needed.
struct Session {
  std::shared_ptr<asio::ip::tcp::socket> sock;
  std::unique_ptr<char[]> buf;
  std::size_t total_bytes_read;
  unsigned int buf_size;
};
```

在步骤 2 中，我们定义了一个回调函数，当异步操作完成时将被调用：

```cpp
// Function used as a callback for 
// asynchronous reading operation.
// Checks if all data has been read
// from the socket and initiates
// new reading operation if needed.
void callback(const boost::system::error_code& ec,
  std::size_t bytes_transferred,
  std::shared_ptr<Session> s)
{
  if (ec != 0) {
    std::cout << "Error occured! Error code = "
      << ec.value()
      << ". Message: " << ec.message();

    return;
  }

  s->total_bytes_read += bytes_transferred;

  if (s->total_bytes_read == s->buf_size) {
    return;
  }

  s->sock->async_read_some(
    asio::buffer(
    s->buf.get() +
      s->total_bytes_read,
    s->buf_size -
      s->total_bytes_read),
    std::bind(callback, std::placeholders::_1,
    std::placeholders::_2, s));
} 
```

让我们暂时跳过步骤 3，并在一个单独的函数中实现步骤 4 和 5。让我们把这个函数命名为`readFromSocket()`：

```cpp
void readFromSocket(std::shared_ptr<asio::ip::tcp::socket> sock) {  
  std::shared_ptr<Session> s(new Session);

  // Step 4\. Allocating the buffer.
  const unsigned int MESSAGE_SIZE = 7;

  s->buf.reset(new char[MESSAGE_SIZE]);
  s->total_bytes_read = 0;
  s->sock = sock;
  s->buf_size = MESSAGE_SIZE;

  // Step 5\. Initiating asynchronous reading operation.
  s->sock->async_read_some(
    asio::buffer(s->buf.get(), s->buf_size),
    std::bind(callback,
      std::placeholders::_1,
      std::placeholders::_2,
      s));
}
```

现在，我们回到步骤 3，并在应用程序的`main()`入口点函数中实现它：

```cpp
int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

    // Step 3\. Allocating, opening and connecting a socket.
    std::shared_ptr<asio::ip::tcp::socket> sock(
      new asio::ip::tcp::socket(ios, ep.protocol()));

    sock->connect(ep);

    readFromSocket(sock);

    // Step 6.
    ios.run();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
} 
```

## 它是如何工作的…

现在，让我们跟踪应用程序的执行路径，以便更好地理解它是如何工作的。

应用程序由单个线程运行；在这个上下文中，调用应用程序的`main()`入口点函数。请注意，Boost.Asio 可能会为某些内部操作创建额外的线程，但它保证不会在那些线程的上下文中调用应用程序代码。

`main()`函数开始于分配、打开并将套接字连接到远程应用程序。然后，它调用`readFromSocket()`函数，并将套接字对象的指针作为参数传递。`readFromSocket()`函数启动一个异步读取操作并返回。我们稍后将考虑这个函数。`main()`函数继续调用`asio::io_service`类的对象的`run()`方法，其中 Boost.Asio 捕获执行线程，并在异步操作完成时使用它来调用相关的回调函数。

`asio::io_service::run()` 方法会阻塞，直到至少有一个挂起的异步操作。当最后一个挂起的操作的最后一个回调完成时，此方法返回。

现在，让我们回到 `readFromSocket()` 函数并分析其行为。它首先在空闲内存中分配 `Session` 数据结构的一个实例。然后，它分配一个缓冲区并将指向它的指针存储在先前分配的 `Session` 数据结构实例中。将套接字对象的指针和缓冲区的大小存储在 `Session` 数据结构中。因为套接字的 `async_read_some()` 方法可能不会一次性读取所有数据，我们可能需要在回调函数中启动另一个异步读取操作。这就是为什么我们需要 `Session` 数据结构，以及为什么我们在空闲内存中而不是在栈上分配它的原因。这个结构和其中驻留的所有对象至少必须持续到回调被调用。

最后，我们启动异步操作，调用套接字对象的 `async_read_some()` 方法。这个方法的调用有些复杂；因此，让我们更详细地看看它：

```cpp
s->sock->async_read_some(
  asio::buffer(s->buf.get(), s->buf_size),
  std::bind(callback,
    std::placeholders::_1,
    std::placeholders::_2,
    s));
```

第一个参数是要读取数据的缓冲区。由于操作是异步的，Boost.Asio 可能在任何时刻（从操作启动到回调被调用之间）访问这个缓冲区。这意味着缓冲区必须保持完整并在回调被调用之前可用。我们通过在空闲内存中分配缓冲区并将其存储在 `Session` 数据结构中来保证这一点，而 `Session` 数据结构本身也是在空闲内存中分配的。

第二个参数是一个在异步操作完成后要调用的回调函数。Boost.Asio 将回调定义为一种概念，它可以是一个函数或仿函数，接受两个参数。回调的第一个参数指定在操作执行过程中发生的错误（如果有）。第二个参数指定操作读取的字节数。

因为我们想向我们的回调函数传递一个额外的参数，即指向相应 `Session` 对象的指针，该对象作为操作的上下文——我们使用 `std::bind()` 函数来构造一个函数对象，我们将指向 `Session` 对象的指针作为第三个参数附加到该函数对象上。然后，将这个函数对象作为回调参数传递给套接字对象的 `async_write_some()` 方法。

因为它是异步的，所以 `async_write_some()` 方法不会阻塞执行线程。它启动读取操作然后返回。

实际的读取操作由 Boost.Asio 库和底层操作系统在幕后执行，当操作完成或发生错误时，会调用回调。

当调用时，名为 `callback` 的回调函数（在我们的示例应用程序中直接称为 `callback`）首先检查操作是否成功或发生错误。在后一种情况下，错误信息会被输出到标准输出流，并且函数返回。否则，总读取字节数会增加操作结果读取的字节数。然后，我们检查从套接字读取的总字节数是否等于缓冲区的大小。如果这两个值相等，这意味着缓冲区已满，没有更多工作要做。回调函数返回。然而，如果缓冲区中仍有空间，我们需要继续读取；因此，我们启动一个新的异步读取操作：

```cpp
s->sock->async_read_some(
    asio::buffer(s->buf.get(), s->buf_size),
    std::bind(callback,
      std::placeholders::_1,
      std::placeholders::_2,
      s));
```

注意，缓冲区的开始位置会根据已读取的字节数进行偏移，并且缓冲区的大小会相应减少。

作为回调，我们使用 `std::bind()` 函数指定相同的 `callback` 函数，以附加一个额外的参数——`Session` 对象。

异步读取操作启动和后续回调调用的周期会一直重复，直到缓冲区满或发生错误。

当 `callback` 函数返回而不启动新的异步操作时，在 `main()` 函数中调用的 `asio::io_service::run()` 方法将执行线程解锁并返回。此时，`main()` 函数也会返回。这就是应用程序退出的时刻。

## 还有更多...

尽管前面示例中描述的 `async_read_some()` 方法允许异步从套接字读取数据，但基于它的解决方案相对复杂且容易出错。幸运的是，Boost.Asio 提供了一种更方便的方式异步从套接字读取数据：免费函数 `asio::async_read()`。让我们考虑其重载之一：

```cpp
template<
    typename AsyncReadStream,
    typename MutableBufferSequence,
    typename ReadHandler>
void async_read(
    AsyncReadStream & s,
    const MutableBufferSequence & buffers,
    ReadHandler handler);
```

此函数与套接字的 `async_read_some()` 方法非常相似。它的第一个参数是一个满足 `AsyncReadStream` 概念要求的对象。关于要求的完整列表，请参阅相应的 Boost.Asio 文档部分，链接为 [`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncReadStream.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncReadStream.html)。`asio::ip::tcp::socket` 类的对象满足这些要求，因此可以与该函数一起使用。

`asio::async_read()` 函数的第二个和第三个参数与前面示例中描述的 TCP 套接字对象的 `async_read_some()` 方法的第一个和第二个参数类似。这些参数用作数据目的点的缓冲区以及表示回调的函数或对象，当操作完成时将被调用。

与`async_read_some()`方法不同，后者启动操作，从套接字读取*一些*数据到缓冲区，`asio::async_read()`函数启动的操作是从套接字读取数据，直到作为参数传递给它的缓冲区满为止。在这种情况下，当读取的数据量等于提供的缓冲区大小时，或者当发生错误时，会调用回调函数。这简化了从套接字读取的过程，并使代码更短更整洁。

如果我们将之前的示例修改为使用`asio::async_read()`函数而不是套接字对象的`async_read_some()`方法来异步从套接字读取数据，那么我们的应用程序将变得更加简单。

首先，我们不需要跟踪从套接字读取的字节数；因此，`Session`结构变得更小：

```cpp
struct Session {
  std::shared_ptr<asio::ip::tcp::socket> sock;
  std::unique_ptr<char[]> buf;
  unsigned int buf_size;
}; 
```

其次，我们知道当回调函数被调用时，这意味着要么已从套接字读取了预期数量的数据，要么发生了错误。这使得回调函数变得更加简单：

```cpp
void callback(const boost::system::error_code& ec,
  std::size_t bytes_transferred,
  std::shared_ptr<Session> s)
{
  if (ec != 0) {
    std::cout << "Error occured! Error code = "
      << ec.value()
      << ". Message: " << ec.message();

    return;
  }

  // Here we know that the reading has completed
  // successfully and the buffer is full with
  // data read from the socket.
}
```

`asio::async_read()`函数是通过零个或多个调用套接字对象的`async_read_some()`方法实现的。这与我们初始示例中的`readFromSocket()`函数的实现方式相似。

### 注意

注意，`asio::async_read()`函数有三个额外的重载，提供了额外的功能。在某些特定情况下，其中一些可能非常有用。请参阅 Boost.Asio 文档了解详情，链接为[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_read.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_read.html)。

## 参见

+   *从 TCP 套接字同步读取*配方描述了如何从 TCP 套接字同步读取数据

+   在第三章的*实现异步 TCP 客户端*配方中，*实现客户端应用程序*，展示了如何实现一个异步 TCP 客户端，该客户端从 TCP 套接字异步读取以接收服务器发送的响应消息

+   在第四章的*实现异步 TCP 服务器*配方中，*实现服务器应用程序*，展示了如何实现一个异步 TCP 服务器，该服务器从 TCP 套接字异步读取以接收客户端发送的请求消息

# 取消异步操作

有时，在异步操作已启动但尚未完成时，应用程序中的条件可能会发生变化，使得启动的操作变得无关紧要或过时，没有人对操作完成感兴趣。

此外，如果启动的异步操作是对用户命令的反应，那么在操作执行过程中，用户可能会改变主意。用户可能想要取消之前发出的命令，并可能想要发出不同的命令或决定退出应用程序。

考虑这样一个情况，用户在典型的网络浏览器地址栏中输入一个网站地址并按下*Enter*键。浏览器立即启动 DNS 名称解析操作。当 DNS 名称解析并获取相应的 IP 地址后，它启动连接操作以连接到相应的 Web 服务器。当建立连接后，浏览器启动异步写入操作以向服务器发送请求。最后，当请求发送后，浏览器开始等待响应消息。根据服务器应用程序的响应速度、通过网络传输的数据量、网络状态和其他因素，所有这些操作可能需要相当长的时间。而在等待请求的网页加载时，用户可能会改变主意，在页面加载完成之前，用户可能在地址栏中输入另一个网站地址并按下*Enter*。

另一个（极端）的情况是，客户端应用程序向服务器应用程序发送请求并开始等待响应消息，但服务器应用程序在处理客户端请求时，由于自身中的错误而陷入死锁。在这种情况下，用户将不得不永远等待响应消息，并且永远不会收到它。

在这两种情况下，客户端应用程序的用户将受益于在操作完成之前取消他们启动的操作的能力。一般来说，提供一个用户可以取消可能需要明显时间的操作的能力是一个好的实践。因为网络通信操作可能持续不可预测的长时间，所以在通过网络通信的分布式应用程序中支持操作的取消非常重要。

Boost.Asio 库提供的异步操作的一个好处是它们可以在启动后的任何时刻取消。在这个菜谱中，我们将看到如何取消异步操作。

## 如何实现它...

以下算法提供了使用 Boost.Asio 初始化和取消异步操作的步骤：

1.  如果应用程序旨在在 Windows XP 或 Windows Server 2003 上运行，则定义启用这些 Windows 版本的异步操作取消的标志。

1.  分配并打开一个 TCP 或 UDP 套接字。它可能是客户端或服务器应用程序中的活动套接字或被动（接受者）套接字。

1.  定义一个用于异步操作的回调函数或仿函数。如果需要，在这个回调中实现一段代码分支，用于处理操作被取消的情况。

1.  启动一个或多个异步操作，并指定步骤 4 中定义的函数或对象作为回调。

1.  启动一个额外的线程并使用它来运行 Boost.Asio 事件循环。

1.  在套接字对象上调用`cancel()`方法以取消与此套接字相关联的所有挂起的异步操作。

让我们考虑一个客户端应用程序的实现，该应用程序按照所提出的算法设计，首先启动一个异步*连接*操作，然后取消该操作。

根据步骤 1，为了在 Windows XP 或 Windows Server 2003 上编译和运行我们的代码，我们需要定义一些标志来控制 Boost.Asio 库对底层 OS 机制的使用行为。

默认情况下，当编译为 Windows 版本时，Boost.Asio 使用 I/O 完成端口框架来异步运行操作。在 Windows XP 和 Windows Server 2003 上，该框架在操作取消方面存在一些问题和限制。因此，Boost.Asio 要求开发者明确通知他们希望在目标 Windows 版本的应用程序中启用异步操作取消功能，尽管已知存在这些问题。为此，必须在包含 Boost.Asio 头文件之前定义`BOOST_ASIO_ENABLE_CANCELIO`宏。否则，如果未定义此宏，而应用程序的源代码包含对异步操作、取消方法和函数的调用，则编译将始终失败。

换句话说，当目标 Windows XP 或 Windows Server 2003 时，必须定义`BOOST_ASIO_ENABLE_CANCELIO`宏，并且应用程序需要取消异步操作。

为了消除 Windows XP 和 Windows Server 2003 上使用 I/O 完成端口框架带来的问题和限制，我们可以在包含 Boost.Asio 头文件之前定义另一个名为`BOOST_ASIO_DISABLE_IOCP`的宏。定义此宏后，Boost.Asio 在 Windows 上不使用 I/O 完成端口框架；因此，与异步操作取消相关的问题消失。然而，I/O 完成端口框架的可扩展性和效率优势也随之消失。

注意，与异步操作取消相关的所述问题和限制在 Windows Vista 和 Windows Server 2008 及以后的版本中不存在。因此，当目标这些版本的 Windows 时，取消操作可以正常工作，除非有其他原因需要禁用 I/O 完成端口框架的使用。有关此问题的更多详细信息，请参阅`asio::ip::tcp::cancel()`方法的文档部分，链接为[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/basic_stream_socket/cancel/overload1.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/basic_stream_socket/cancel/overload1.html)。

在我们的示例中，我们将考虑如何构建一个跨平台应用程序，当在编译时针对 Windows，可以从 Windows XP 或 Windows Server 2003 开始运行。因此，我们定义了`BOOST_ASIO_DISABLE_IOCP`和`BOOST_ASIO_ENABLE_CANCELIO`宏。

为了在编译时确定目标操作系统，我们使用`Boost.Predef`库。这个库为我们提供了宏定义，允许我们识别代码编译环境的参数，作为目标操作系统家族及其版本、处理器架构、编译器等。有关此库的更多详细信息，请参阅 Boost.Asio 文档部分，[`www.boost.org/doc/libs/1_58_0/libs/predef/doc/html/index.html`](http://www.boost.org/doc/libs/1_58_0/libs/predef/doc/html/index.html)。

要使用`Boost.Predef`库，我们需要包含以下头文件：

```cpp
#include <boost/predef.h> // Tools to identify the OS.
```

然后，我们检查代码是否正在为 Windows XP 或 Windows Server 2003 编译，如果是，我们定义`BOOST_ASIO_DISABLE_IOCP`和`BOOST_ASIO_ENABLE_CANCELIO`宏：

```cpp
#ifdef BOOST_OS_WINDOWS
#define _WIN32_WINNT 0x0501

#if _WIN32_WINNT <= 0x0502 // Windows Server 2003 or earlier.
#define BOOST_ASIO_DISABLE_IOCP
#define BOOST_ASIO_ENABLE_CANCELIO  
#endif
#endif
```

接下来，我们包含常见的 Boost.Asio 头文件和标准库`<thread>`头文件。我们还需要后者，因为我们在应用程序中会创建额外的线程。此外，我们指定一个`using`指令，使 Boost.Asio 类和函数的名称更短，更方便使用：

```cpp
#include <boost/asio.hpp>
#include <iostream>
#include <thread>

using namespace boost;
```

然后，我们定义应用程序的`main()`入口点函数，它包含应用程序的所有功能：

```cpp
int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

    std::shared_ptr<asio::ip::tcp::socket> sock(
      new asio::ip::tcp::socket(ios, ep.protocol()));

    sock->async_connect(ep,
      sock
    {
      // If asynchronous operation has been
      // cancelled or an error occured during
      // execution, ec contains corresponding
      // error code.
      if (ec != 0) {
        if (ec == asio::error::operation_aborted) {
          std::cout << "Operation cancelled!";
        }
        else {
          std::cout << "Error occured!"
            << " Error code = "
            << ec.value()
            << ". Message: "
            << ec.message();
        }

        return;
      }
      // At this point the socket is connected and
      // can be used for communication with 
      // remote application.
    });

    // Starting a thread, which will be used
    // to call the callback when asynchronous 
    // operation completes.
    std::thread worker_thread([&ios](){
      try {
        ios.run();
      }
      catch (system::system_error &e) {
        std::cout << "Error occured!"
        << " Error code = " << e.code()
        << ". Message: " << e.what();
      }
    });

    // Emulating delay.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Cancelling the initiated operation.
    sock->cancel();

    // Waiting for the worker thread to complete.
    worker_thread.join();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的…

现在，让我们分析应用程序的工作原理。

我们的示例客户端应用程序由一个单一的功能组成，即应用程序的`main()`入口点函数。此函数从根据算法的第 2 步分配和打开 TCP 套接字开始。

接下来，在套接字上启动异步连接操作。提供给方法的回调实现为一个 lambda 函数。这对应于算法的第 3 步和第 4 步。注意，在回调函数中确定操作是否被取消。当异步操作被取消时，回调被调用，其参数指定了错误代码，包含在 Boost.Asio 中定义的与操作系统相关的错误代码`asio::error::operation_aborted`。

然后，我们创建一个名为`worker_thread`的线程，该线程将用于运行 Boost.Asio 事件循环。在这个线程的上下文中，回调函数将由库调用。`worker_thread`线程的入口点函数相当简单。它包含一个`try-catch`块和对`asio::io_service`对象`run()`方法的调用。这对应于算法的第 5 步。

在创建工作线程之后，主线程将休眠 2 秒钟。这是为了让连接操作有更多的时间进行，并模拟实际应用程序中用户发出的两个命令之间的延迟；例如，一个网页浏览器。

根据算法的最后一步 6，我们调用套接字对象的`cancel()`方法来取消已启动的连接操作。此时，如果操作尚未完成，它将被取消，并且相应的回调将使用一个指定包含`asio::error::operation_aborted`值的错误代码的参数来调用，以通知操作已被取消。然而，如果操作已经完成，调用`cancel()`方法将没有效果。

当回调函数返回时，工作线程会退出事件循环，因为没有更多的挂起异步操作需要执行。因此，线程会退出其入口点函数。这导致主线程运行到完成。最终，应用程序退出。

## 更多内容...

在前面的示例中，我们考虑了与活动 TCP 套接字相关联的异步连接操作的取消。然而，任何与 TCP 和 UDP 套接字都相关联的操作都可以以类似的方式取消。在操作启动后，应在相应的套接字对象上调用`cancel()`方法。

此外，`asio::ip::tcp::resolver`或`asio::ip::udp::resolver`类的`async_resolve()`方法，用于异步解析 DNS 名称，可以通过调用解析器对象的`cancel()`方法来取消。

所有由 Boost.Asio 提供的相应免费函数启动的异步操作也可以通过在传递给免费函数的第一个参数的对象上调用`cancel()`方法来取消。此对象可以代表套接字（活动或被动）或解析器。

## 参见

+   在第三章的*实现异步 TCP 客户端*配方中，*实现客户端应用程序*，演示了如何构建一个支持异步操作取消功能的更复杂的客户端应用程序。

+   第一章*基础知识*中的配方演示了如何同步连接套接字和解析 DNS 名称。

# 关闭和关闭套接字

在一些使用 TCP 协议进行通信的分布式应用程序中，需要传输没有固定大小和特定字节序列的消息，并标记其边界。这意味着接收方在从套接字读取消息时，无法通过分析消息本身的大小或内容来确定消息的结束位置。

解决此问题的一种方法是将每条消息结构化为一个逻辑头部分和一个逻辑体部分。头部分具有固定的大小和结构，并指定体部分的大小。这允许接收方首先读取并解析头部分，找出消息体的大小，然后正确读取消息的其余部分。

这种方法相当简单，并且被广泛使用。然而，它带来了一些冗余和额外的计算开销，这在某些情况下可能是不可以接受的。

当一个应用程序为发送给对等方的每条消息使用单独的套接字时，可以采用另一种方法，这是一种相当流行的做法。这种方法的想法是在消息写入套接字后，由消息发送者**关闭**套接字的发送部分。这会导致发送一个特殊的服务消息给接收者，告知接收者消息已结束，发送者将不会使用当前连接发送任何其他内容。

第二种方法比第一种方法提供了更多的好处，并且因为它属于 TCP 协议软件的一部分，所以它对开发者来说很容易使用。

套接字上的另一种操作，即**关闭**，看起来可能类似于关闭，但实际上它与关闭操作非常不同。关闭套接字意味着将套接字及其所有其他相关资源返回给操作系统。就像内存、进程或线程、文件句柄或互斥锁一样，套接字是操作系统的资源。并且像任何其他资源一样，套接字在分配、使用且不再由应用程序需要后，应返回给操作系统。否则，可能会发生资源泄漏，这最终可能导致资源耗尽，并导致应用程序故障或整个操作系统的不稳定。

当套接字未关闭时可能出现的严重问题使得关闭操作变得非常重要。

关闭 TCP 套接字与关闭套接字之间的主要区别在于，如果已经建立了连接，关闭操作会中断连接，并最终释放套接字并将其返回给操作系统，而关闭操作仅禁用套接字的写入、读取或两者操作，并向对等应用程序发送一个服务消息来通知这一事实。关闭套接字永远不会导致释放套接字。

在这个菜谱中，我们将看到如何关闭和关闭 TCP 套接字。

## 如何做到这一点...

在这里，我们将考虑一个由两部分组成的分布式应用程序：客户端和服务器，以便更好地理解如何使用套接字关闭操作来使基于分布式应用程序部分之间基于随机大小的二进制消息的应用层协议更加高效和清晰。

为了简单起见，客户端和服务器应用程序中的所有操作都是同步的。

### 客户端应用程序

客户端应用程序的目的是分配套接字并将其连接到服务器应用程序。在建立连接后，应用程序应准备并发送一个请求消息，通过在消息写入后关闭套接字来通知其边界。

在请求发送后，客户端应用程序应读取响应。响应的大小是未知的；因此，读取应一直进行，直到服务器关闭其套接字以通知响应边界。

我们通过指定`include`和`using`指令开始客户端应用程序：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;
```

接下来，我们定义一个函数，该函数接受一个指向连接到服务器的套接字对象的引用，并使用此套接字与服务器进行通信。让我们称这个函数为`communicate()`：

```cpp
void communicate(asio::ip::tcp::socket& sock) {
  // Allocating and filling the buffer with
  // binary data.
  const char request_buf[] = {0x48, 0x65, 0x0, 0x6c, 0x6c,
 0x6f};

  // Sending the request data.
  asio::write(sock, asio::buffer(request_buf));

  // Shutting down the socket to let the
  // server know that we've sent the whole
  // request.
  sock.shutdown(asio::socket_base::shutdown_send);

  // We use extensible buffer for response
  // because we don't know the size of the
  // response message.
  asio::streambuf response_buf;

  system::error_code ec;
  asio::read(sock, response_buf, ec);

  if (ec == asio::error::eof) {
    // Whole response message has been received.
    // Here we can handle it.
  }
  else {
    throw system::system_error(ec);
  }
}
```

最后，我们定义一个应用程序的`main()`入口点函数。此函数分配和连接套接字，然后调用之前步骤中定义的`communicate()`函数：

```cpp
int main()
{
  std::string raw_ip_address = "127.0.0.1";
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint
      ep(asio::ip::address::from_string(raw_ip_address),
      port_num);

    asio::io_service ios;

    asio::ip::tcp::socket sock(ios, ep.protocol());

    sock.connect(ep);

    communicate(sock);
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

### 服务器应用程序

服务器应用程序旨在分配一个接受器套接字并被动等待连接请求。当连接请求到达时，它应接受该请求并从连接到客户端的套接字读取数据，直到客户端应用程序在其端关闭套接字。在收到请求消息后，服务器应用程序应通过关闭套接字来发送响应消息并通知其边界。

我们通过指定`include`和`using`指令开始客户端应用程序：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;
```

接下来，我们定义一个函数，该函数接受一个指向连接到客户端应用程序的套接字对象的引用，并使用此套接字与客户端进行通信。让我们称这个函数为`processRequest()`：

```cpp
void processRequest(asio::ip::tcp::socket& sock) {
  // We use extensible buffer because we don't
  // know the size of the request message.
  asio::streambuf request_buf;

  system::error_code ec;

  // Receiving the request.
  asio::read(sock, request_buf, ec);

  if (ec != asio::error::eof)
    throw system::system_error(ec);

  // Request received. Sending response.
  // Allocating and filling the buffer with
  // binary data.
  const char response_buf[] = { 0x48, 0x69, 0x21 };

  // Sending the request data.
  asio::write(sock, asio::buffer(response_buf));

  // Shutting down the socket to let the
  // client know that we've sent the whole
  // response.
  sock.shutdown(asio::socket_base::shutdown_send);
}
```

最后，我们定义应用程序的`main()`入口点函数。此函数分配一个接受器套接字并等待传入的连接请求。当连接请求到达时，它获取一个连接到客户端应用程序的活动套接字，并通过传递一个连接套接字对象到其中调用之前步骤中定义的`processRequest()`函数：

```cpp
int main()
{
  unsigned short port_num = 3333;

  try {
    asio::ip::tcp::endpoint ep(asio::ip::address_v4::any(),
      port_num);

    asio::io_service ios;

    asio::ip::tcp::acceptor acceptor(ios, ep);

    asio::ip::tcp::socket sock(ios);

    acceptor.accept(sock);

    processRequest(sock);
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

### 关闭套接字

为了关闭一个分配的套接字，应在`asio::ip::tcp::socket`类的相应对象上调用`close()`方法。然而，通常不需要显式执行此操作，因为如果未显式关闭，套接字对象的析构函数会关闭套接字。

## 它是如何工作的...

服务器应用程序首先启动。在其`main()`入口点函数中，分配了一个接受器套接字，打开它，并将其绑定到端口`3333`，然后开始等待来自客户端的传入连接请求。

然后，启动客户端应用程序。在其`main()`入口点函数中，分配了一个活动套接字，打开它，并将其连接到服务器。在建立连接后，调用`communicate()`函数。在这个函数中，所有有趣的事情都发生了。

客户端应用程序向套接字写入请求消息，然后调用套接字的`shutdown()`方法，并将`asio::socket_base::shutdown_send`常量作为参数传递。这个调用关闭了套接字的发送部分。此时，向套接字写入被禁用，且无法恢复套接字状态使其再次可写：

```cpp
sock.shutdown(asio::socket_base::shutdown_send);
```

在客户端应用程序中关闭套接字被视为服务器应用程序中到达服务器的协议服务消息，通知对等应用程序已关闭套接字。Boost.Asio 通过`asio::read()`函数返回的错误代码将此消息传递给应用程序代码。Boost.Asio 库将此代码定义为`asio::error::eof`。服务器应用程序使用此错误代码来确定客户端何时完成发送请求消息。

当服务器应用程序接收到完整的请求消息时，服务器和客户端交换它们的角色。现在，服务器在其端向套接字写入数据，即响应消息，客户端应用程序在其端读取此消息。当服务器完成将响应消息写入套接字后，它关闭其套接字的发送部分，以表示整个消息已发送到其对等方。

同时，客户端应用程序在`asio::read()`函数中被阻塞，读取服务器发送的响应，直到函数返回错误代码等于`asio::error::eof`，这表示服务器已发送完响应消息。当`asio::read()`函数返回此错误代码时，客户端*知道*它已读取整个响应消息，然后可以开始处理它：

```cpp
system::error_code ec;
asio::read(sock, response_buf, ec);

if (ec == asio::error::eof) {
  // Whole response message has been received.
  // Here we can handle it.
}
```

注意，在客户端关闭其套接字的发送部分后，它仍然可以从套接字读取数据，因为套接字的接收部分独立于发送部分保持打开状态。

## 参见

+   *同步写入 TCP 套接字*配方演示了如何同步地将数据写入 TCP 套接字。

+   *同步从 TCP 套接字读取*配方演示了如何同步地从 TCP 套接字读取数据。

+   第五章中的*实现 HTTP 客户端应用程序*和*实现 HTTP 服务器应用程序*配方演示了在实现 HTTP 协议中如何使用套接字关闭。
