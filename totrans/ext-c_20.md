# 第二十章

# 套接字编程

在前一章中，我们讨论了单主机进程间通信（IPC）并介绍了套接字编程。在这一章中，我们想要完成我们的介绍，并使用一个真实的客户端-服务器示例（计算器项目）深入探讨套接字编程。

本章中主题的顺序可能看起来有些不寻常，但目的是让你更好地理解各种类型的套接字以及它们在实际项目中的行为。作为本章的一部分，我们讨论以下主题：

+   首先，我们回顾一下前一章中我们解释的内容。请注意，这个回顾只是一个简短的总结，你必须阅读前一章关于套接字编程的第二部分。

+   作为回顾的一部分，我们讨论了各种类型的套接字、流和数据报序列，以及对我们继续计算器示例至关重要的其他主题。

+   客户端-服务器示例，即计算器项目，被描述并全面分析。这为我们继续讨论示例中的各种组件和展示 C 代码做好了准备。

+   作为示例的关键组件，我们开发了一个序列化/反序列化库。这个库将代表计算器客户端与其服务器之间使用的主要协议。

+   理解这一点至关重要：计算器客户端和计算器服务器必须能够通过任何类型的套接字进行通信。因此，我们在示例中展示了各种类型的套接字，并以**Unix 域套接字（UDS**）作为起点。

+   在我们的示例中，我们展示了它们如何在单主机设置中建立客户端-服务器连接。

+   为了继续讨论其他类型的套接字，我们讨论网络套接字。我们展示了如何在计算器项目中集成 TCP 和 UDP 套接字。

让我们从总结我们关于套接字和套接字编程的一般知识开始这一章。在深入本章内容之前，强烈建议你熟悉前一章的后半部分，因为我们在这里假设了一些先验知识。

# 套接字编程回顾

在本节中，我们将讨论什么是套接字，它们的各种类型是什么，以及如果我们说我们在进行套接字编程，这通常意味着什么。这将是一个简短的回顾，但这是建立这个基础所必需的，以便我们可以在后续章节中进行更深入的讨论。

如果您还记得前几章的内容，我们有两种 IPC 技术类别，用于两个或更多进程进行通信和共享数据。第一类包含*基于拉取*的技术，这些技术需要一个可访问的*介质*（例如共享内存或常规文件）来存储数据和检索数据。第二类包含*基于推送*的技术。这些技术需要一个*通道*来建立，并且该通道应该对所有进程都是可访问的。这两类技术的主要区别在于，在基于拉取的技术中，数据是从介质中检索的方式，或者在基于推送的技术中，是从通道中检索的方式。

简单来说，在基于拉取的技术中，数据应该从介质中拉取或读取，但在基于推送的技术中，数据会自动推送到或交付给读取进程。在基于拉取的技术中，由于进程从共享介质中拉取数据，如果多个进程可以写入该介质，就容易出现竞态条件。

要更精确地描述基于推送的技术，数据始终被发送到内核中的一个缓冲区，并且该缓冲区可以通过使用描述符（文件或套接字）被接收进程访问。

然后，接收进程可以选择阻塞，直到该描述符上有可用的新数据，或者它可以*轮询*该描述符以查看内核是否在该描述符上接收到了新数据；如果没有，则继续执行其他工作。前者是*阻塞 I/O*，后者是*非阻塞 I/O*或*异步 I/O*。在本章中，所有基于推送的技术都使用阻塞方法。

我们知道，套接字编程是一种特殊的 IPC（进程间通信）类型，属于第二类。因此，所有基于套接字的 IPC 都是基于推送的。但将套接字编程与其他基于推送的 IPC 技术区分开来的主要特征是，在套接字编程中我们使用*套接字*。套接字是类 Unix 操作系统中的一种特殊对象，甚至在非 Unix-like 的 Microsoft Windows 系统中，它代表*双向通道*。

换句话说，单个套接字对象可以用来从同一个通道中读取和写入。这样，位于同一通道两端的两个进程可以实现*双向通信*。

在前一章中，我们了解到套接字由套接字描述符表示，就像文件由文件描述符表示一样。虽然套接字描述符和文件描述符在某些方面相似，例如 I/O 操作和可轮询性，但它们实际上是不同的。单个套接字描述符始终代表一个通道，但文件描述符可以代表一个介质，如常规文件，或者一个通道，如 POSIX 管道。因此，与文件相关的某些操作，如 seek，不支持套接字描述符，甚至当文件描述符代表通道时也不支持。

基于套接字的通信可以是**面向连接**的或**无连接**的。在面向连接的通信中，通道代表两个特定进程之间传输的字节**流**，而在无连接通信中，**数据报**可以沿着通道传输，并且两个进程之间没有特定的连接。多个进程可以使用同一个通道来共享状态或传输数据。

因此，我们有两种类型的通道：**流通道**和**数据报通道**。在程序中，每个流通道都由一个**流套接字**表示，每个数据报通道都由一个**数据报套接字**表示。在设置通道时，我们必须决定它应该是流还是数据报。我们很快就会看到我们的计算器示例可以支持这两种通道。

套接字有多种类型。每种类型的套接字都是为了特定的用途和情况而存在的。通常，我们有两种类型的套接字：Unix 域套接字（UDS）和网络套接字。正如您可能知道的那样，以及我们在上一章中解释的那样，UDS 可以在所有希望参与进程间通信（IPC）的进程都位于同一台机器上时使用。换句话说，UDS 只能在单主机部署中使用。

相比之下，网络套接字几乎可以在任何部署中使用，无论进程如何部署以及它们位于何处。它们可以全部位于同一台机器上，也可以分布在整个网络中。在单主机部署的情况下，UDS 更受欢迎，因为它们更快，并且与网络套接字相比，开销更小。作为我们计算器示例的一部分，我们提供了对 UDS 和网络套接字的支持。

UDS 和网络套接字可以代表流和数据报通道。因此，我们有四种类型：流通道上的 UDS、数据报通道上的 UDS、流通道上的网络套接字，以及最后是数据报通道上的网络套接字。所有这四种变化都在我们的示例中得到了涵盖。

提供流通道的网络套接字通常是 TCP 套接字。这是因为，大多数情况下，我们使用 TCP 作为此类套接字的传输协议。同样，提供数据报通道的网络套接字通常是 UDP 套接字。这是因为，大多数情况下，我们使用 UDP 作为此类套接字的传输协议。请注意，提供流或数据报通道的 UDS 套接字没有特定的名称，因为没有底层传输协议。

为了编写针对不同类型套接字和通道的实际 C 代码，最好是在您处理真实示例时进行。这就是我们采取这种不寻常方法的基本原因。这样，您将注意到不同类型套接字和通道之间的共同部分，我们可以将它们提取为可重用的代码单元。在下一节中，我们将讨论计算器项目及其内部结构。

# 计算器项目

我们将专门用一节来解释计算器项目的目的。这是一个篇幅较长的示例，因此在深入之前有一个坚实的基础将非常有帮助。该项目应帮助你实现以下目标：

+   观察一个具有多个简单且定义明确的功能的完全功能化示例。

+   从各种类型的套接字和通道中提取公共部分，并将它们作为一些可重用的库。这显著减少了我们需要编写的代码量，从学习的角度来看，它展示了不同类型的套接字和通道之间的共同边界。

+   使用定义良好的应用程序协议来维护通信。普通的套接字编程示例缺乏这个非常重要的功能。它们通常处理客户端与其服务器之间非常简单且通常是单次通信场景。

+   在一个示例中工作，这个示例包含了一个完全功能化的客户端-服务器程序所需的所有成分，例如应用程序协议、支持各种类型的通道、具有序列化/反序列化功能等，这为你提供了关于套接字编程的不同视角。

话虽如此，我们将把这个项目作为本章的主要示例来介绍。我们将一步一步地进行，我会引导你通过各种步骤，最终完成一个完整且可工作的项目。

第一步是提出一个相对简单且完整的应用程序协议。这个协议将在客户端和服务器之间使用。正如我们之前所解释的，如果没有一个定义良好的应用程序协议，双方就无法进行通信。他们可以连接并传输数据，因为这是套接字编程提供的功能，但他们无法相互理解。

因此，我们需要花一些时间来理解计算器项目中使用的应用程序协议。在讨论应用程序协议之前，让我们先展示项目代码库中可以看到的源代码层次结构。然后，我们可以在项目代码库中更容易地找到应用程序协议和相关的序列化/反序列化库。

## 源代码层次结构

从程序员的视角来看，POSIX 套接字编程 API 无论关联的套接字对象是 uds 还是网络套接字，都同等对待所有流通道。如果你还记得上一章的内容，对于流通道，我们有监听端和连接端的特定序列，并且这些序列对于不同类型的流套接字来说是相同的。

因此，如果您打算支持各种类型的套接字以及各种类型的通道，最好提取公共部分并一次性编写。这正是我们对待计算器项目的方法，这也是您在源代码中看到的方法。因此，预计在项目中会看到各种库，其中一些包含其他代码部分复用的公共代码。

现在，是时候深入代码库了。首先，项目的源代码可以在这里找到：https://github.com/PacktPublishing/Extreme-C/tree/master/ch20-socket-programming。如果您打开链接并查看代码库，您会看到有多个包含多个源文件的目录。显然，我们无法演示所有这些目录，因为这会花费太多时间，但我们将解释代码的重要部分。我们鼓励您查看代码，并尝试构建和运行它；这将给您一个关于示例是如何开发的思路。

注意，所有与 UDS、UDP 套接字和 TCP 套接字示例相关的代码都已放入一个单独的层次结构中。接下来，我们将解释源层次结构和您在代码库中找到的目录。

如果您进入示例的根目录并使用`tree`命令显示文件和目录，您将找到类似于*Shell Box 20-1*的内容。

下面的 Shell Box 演示了如何克隆本书的 GitHub 仓库以及如何导航到示例的根目录：

```cpp
$ git clone https://github.com/PacktPublishing/Extreme-C
Cloning into 'Extreme-C'...
...
Resolving deltas: 100% (458/458), done.
$ cd Extreme-C/ch20-socket-programming
$ tree
.
├── CMakeLists.txt
├── calcser
...
├── calcsvc
...
├── client
│   ├── CMakeLists.txt
│   ├── clicore
...
│   ├── tcp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   ├── udp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   └── Unix
│       ├── CMakeLists.txt
│       ├── datagram
│       │   ├── CMakeLists.txt
│       │   └── main.c
│       └── stream
│           ├── CMakeLists.txt
│           └── main.c
├── server
│   ├── CMakeLists.txt
│   ├── srvcore
...
│   ├── tcp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   ├── udp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   └── Unix
│       ├── CMakeLists.txt
│       ├── datagram
│       │   ├── CMakeLists.txt
│       │   └── main.c
│       └── stream
│           ├── CMakeLists.txt
│           └── main.c
└── types.h
18 directories, 49 files
$
```

Shell Box 20-1：克隆计算器项目的代码库并列出文件和目录

如您在文件和目录列表中所见，计算器项目由多个部分组成，其中一些是库，每个部分都有自己的专用目录。接下来，我们将解释这些目录：

+   `/calcser`：这是一个序列化/反序列化库。它包含与序列化/反序列化相关的源文件。这个库决定了计算器客户端和计算器服务器之间定义的应用协议。这个库最终被构建成一个名为`libcalcser.a`的静态库文件。

+   `/calcsvc`：这个库包含计算服务器的源代码。*计算服务*与服务器进程不同。这个服务库包含计算器的核心功能，并且与是否位于服务器进程之后无关，它可以作为一个独立的独立 C 库单独使用。这个库最终被构建成一个名为`libcalcsvc.a`的静态库文件。

+   `/server/srvcore`: 此库包含流和数据报服务器进程之间共有的源代码，无论套接字类型如何。因此，所有计算器服务器进程，无论它们是否使用 UDS 或网络套接字，以及无论它们是在流通道还是数据报通道上操作，都可以依赖这个通用部分。此库的最终输出是一个名为 `libsrvcore.a` 的静态库文件。

+   `/server/unix/stream`: 此目录包含使用 UDS 后端流通道的服务器程序的源代码。此目录的最终构建结果是名为 `unix_stream_calc_server` 的可执行文件。这是本项目中可能生成的输出可执行文件之一，我们可以使用它来启动计算器服务器，该服务器监听 UDS 以接收流连接。

+   `/server/unix/datagram`: 此目录包含使用 UDS 后端数据报通道的服务器程序的源代码。此目录的最终构建结果是名为 `unix_datagram_calc_server` 的可执行文件。这是本项目中可能生成的输出可执行文件之一，我们可以使用它来启动计算器服务器，该服务器监听 UDS 以接收数据报消息。

+   `/server/tcp`: 此目录包含使用 TCP 网络套接字后端流通道的服务器程序的源代码。此目录的最终构建结果是名为 `tcp_calc_server` 的可执行文件。这是本项目中可能生成的输出可执行文件之一，我们可以使用它来启动计算器服务器，该服务器监听 TCP 套接字以接收流连接。

+   `/server/udp`: 此目录包含使用 UDP 网络套接字后端数据报通道的服务器程序的源代码。此目录的最终构建结果是名为 `udp_calc_server` 的可执行文件。这是本项目中可能生成的输出可执行文件之一，我们可以使用它来启动计算器服务器，该服务器监听 UDP 套接字以接收数据报消息。

+   `/client/clicore`: 此库包含流和数据报客户端进程之间共有的源代码，无论套接字类型如何。因此，所有计算器客户端进程，无论它们是否使用 UDS 或网络套接字，以及无论它们是在流通道还是数据报通道上操作，都可以依赖这个通用部分。它将被构建成一个名为 `libclicore.a` 的静态库文件。

+   `/client/unix/stream`: 此目录包含使用 UDS 后端流通道的客户端程序的源代码。此目录的最终构建结果是名为 `unix_stream_calc_client` 的可执行文件。这是本项目中可能生成的输出可执行文件之一，我们可以使用它来启动计算器客户端，该客户端连接到 UDS 端点并建立流连接。

+   `/client/unix/datagram`：此目录包含使用 UDS 后端数据报通道的客户端程序源代码。此目录的最终构建结果是名为`unix_datagram_calc_client`的可执行文件。这是本项目可能的输出可执行文件之一，我们可以使用它来启动计算器客户端，该客户端连接到 UDS 端点并发送一些数据报消息。

+   `/client/tcp`：此目录包含使用 TCP 套接字后端流通道的客户端程序源代码。此目录的最终构建结果是名为`tcp_calc_client`的可执行文件。这是本项目可能的输出可执行文件之一，我们可以使用它来启动计算器客户端，该客户端连接到 TCP 套接字端点并建立一个流连接。

+   `/client/udp`：此目录包含使用 UDP 套接字后端数据报通道的客户端程序源代码。此目录的最终构建结果是名为`udp_calc_client`的可执行文件。这是本项目可能的输出可执行文件之一，我们可以使用它来启动计算器客户端，该客户端连接到 UDP 套接字端点并发送一些数据报消息。

## 构建项目

现在我们已经查看了项目中的所有目录，我们需要展示如何构建它。该项目使用 CMake，在构建项目之前，您应该已经安装了它。

为了构建项目，在章节根目录中运行以下命令：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
...
$ make
...
$
```

Shell Box 20-2：构建计算器项目的命令

## 运行项目

没有什么比亲自运行项目来看到它是如何工作的更好的了。因此，在深入研究技术细节之前，我想让您启动一个计算器服务器，然后是一个计算器客户端，最后看看它们是如何互相通信的。

在运行进程之前，您需要有两个独立的终端（或 shell），以便输入两个不同的命令。在第一个终端中，为了运行监听 UDS 的流服务器，请输入以下命令。

注意，在输入以下命令之前，您需要处于`build`目录中。`build`目录是上一节*构建项目*中创建的：

```cpp
$ ./server/unix/stream/unix_stream_calc_server
```

Shell Box 20-3：运行监听 UDS 的流服务器

确保服务器正在运行。在第二个终端中，运行为使用 UDS 构建的流客户端：

```cpp
$ ./client/unix/stream/unix_stream_calc_client
? (type quit to exit) 3++4
The req(0) is sent.
req(0) > status: OK, result: 7.000000
? (type quit to exit) mem
The req(1) is sent.
req(1) > status: OK, result: 7.000000
? (type quit to exit) 5++4
The req(2) is sent.
req(2) > status: OK, result: 16.000000
? (type quit to exit) quit
Bye.
$
```

Shell Box 20-4：运行计算器客户端并发送一些请求

正如您在先前的 Shell Box 中看到的那样，客户端进程有自己的命令行。它从用户那里接收一些命令，根据应用程序协议将它们转换为一些请求，并将它们发送到服务器进行进一步处理。然后，它等待响应，并在收到响应后立即打印结果。请注意，此命令行是所有客户端共同编写的通用代码的一部分，因此，无论客户端使用的是哪种通道类型或套接字类型，您总是看到客户端命令行。

现在，是时候深入了解应用协议，看看请求和响应消息看起来像什么。

## 应用协议

任何想要通信的两个进程都必须遵守一个应用协议。这个协议可以是定制的，比如计算器项目，也可以是众所周知的一些协议，如 HTTP。我们称我们的协议为 *计算器协议*。

计算器协议是一个可变长度的协议。换句话说，每个消息都有自己的长度，每个消息都应该使用分隔符与下一个消息分开。只有一个请求消息类型和一个响应消息类型。该协议也是文本的。这意味着我们只使用字母数字字符以及一些其他字符作为请求和响应消息中的有效字符。换句话说，计算器消息是可读的。

请求消息有四个字段：*请求 ID*、*方法*、*第一个操作数*和*第二个操作数*。每个请求都有一个唯一的 ID，服务器使用这个 ID 将响应与其对应的请求相关联。

方法是计算器服务可以执行的操作。接下来，你可以看到 `calcser/calc_proto_req.h` 头文件。这个文件描述了计算器协议的请求消息：

```cpp
#ifndef CALC_PROTO_REQ_H
#define CALC_PROTO_REQ_H
#include <stdint.h>
typedef enum {
  NONE,
  GETMEM, RESMEM,
  ADD, ADDM,
  SUB, SUBM,
  MUL, MULM,
  DIV
} method_t;
struct calc_proto_req_t {
  int32_t id;
  method_t method;
  double operand1;
  double operand2;
};
method_t str_to_method(const char*);
const char* method_to_str(method_t);
#endif
```

代码框 20-1 [calcser/calc_proto_req.h]：计算器请求对象的定义

正如你所见，我们定义了九种方法作为我们协议的一部分。作为一个好的计算器，我们的计算器有一个内部内存，因此我们有关加法、减法和乘法的内存操作。

例如，`ADD` 方法只是简单地相加两个浮点数，但 `ADDM` 方法是 `ADD` 方法的变体，它将这两个数与内部存储的值相加，并最终更新内存中的值以供进一步使用。这就像你使用台式计算器的内存按钮一样。你可以找到一个标记为 +M 的按钮。

我们还有一个用于读取和重置计算器内部内存的特殊方法。除法方法不能在内部内存上执行，所以我们没有其他变体。

假设客户端想要使用 `ADD` 方法创建一个 ID 为 `1000` 的请求，并且操作数为 `1.5` 和 `5.6`。在 C 语言中，需要从 `calc_proto_req_t` 类型（在前面头文件中作为 *代码框 20-1* 部分声明）创建一个对象，并填充所需的值。接下来，你可以看到如何操作：

```cpp
struct calc_proto_req_t req;
req.id = 1000;
req.method = ADD;
req.operand1 = 1.5;
req.operand2 = 5.6;
```

代码框 20-2：在 C 语言中创建计算器请求对象

正如我们在上一章中解释的，前面代码框中的 `req` 对象在发送到服务器之前需要序列化为请求消息。换句话说，我们需要将前面的 *请求对象* 序列化为等效的 *请求消息*。根据我们的应用协议，计算器项目中的序列化器将 `req` 对象序列化如下：

```cpp
1000#ADD#1.5#5.6$
```

代码框 20-3：与代码框 20-2 中定义的 req 对象等价的序列化消息

如您所见，`#`字符用作*字段分隔符*，而`$`字符用作*消息分隔符*。此外，每个请求消息恰好有四个字段。通道另一端的*反序列化器*对象使用这些事实来解析传入的字节并重新恢复请求对象。

相反，服务器进程在回复请求时需要序列化响应对象。计算器响应对象有三个字段：*请求 ID*、*状态*和*结果*。请求 ID 确定相应的请求。每个请求都有一个唯一的 ID，这样服务器就可以指定它想要响应的请求。

`calcser/calc_proto_resp.h`头文件描述了计算器响应应该是什么样子，您可以在下面的代码框中看到：

```cpp
#ifndef CALC_PROTO_RESP_H
#define CALC_PROTO_RESP_H
#include <stdint.h>
#define STATUS_OK              0
#define STATUS_INVALID_REQUEST 1
#define STATUS_INVALID_METHOD  2
#define STATUS_INVALID_OPERAND 3
#define STATUS_DIV_BY_ZERO     4
#define STATUS_INTERNAL_ERROR  20
typedef int status_t;
struct calc_proto_resp_t {
  int32_t req_id;
  status_t status;
  double result;
};
#endif
```

代码框 20-4 [calcser/calc_proto_resp.h]：计算器响应对象的定义

同样，为了为前面提到的*代码框 20-2*中的`req`请求对象创建一个响应对象，服务器进程应该这样做：

```cpp
struct calc_proto_resp_t resp;
resp.req_id = 1000;
resp.status = STATUS_OK;
resp.result = 7.1;
```

代码框 20-5：为代码框 20-2 中定义的请求对象 req 创建响应对象

前面的响应对象按以下方式序列化：

```cpp
1000#0#7.1$
```

代码框 20-6：与代码框 20-5 中创建的 resp 对象等价的序列化响应消息

再次，我们使用`#`作为字段分隔符，`$`作为消息分隔符。请注意，状态是数值型的，它表示请求的成功或失败。在失败的情况下，它是一个非零数字，其含义在响应头文件中描述，或者更确切地说，在计算器协议中描述。

现在，是时候更详细地谈谈序列化/反序列化库及其内部结构了。

## 序列化/反序列化库

在上一节中，我们描述了请求和响应消息的格式。在本节中，我们将更详细地讨论计算器项目中使用的序列化和反序列化算法。我们将使用`serializer`类，其属性结构为`calc_proto_ser_t`，以提供序列化和反序列化功能。

如前所述，这些功能作为名为`libcalcser.a`的静态库提供给项目的其他部分。在这里，您可以看到`calcser/calc_proto_ser.h`中找到的`serializer`类的公共 API：

```cpp
#ifndef CALC_PROTO_SER_H
#define CALC_PROTO_SER_H
#include <types.h>
#include "calc_proto_req.h"
#include "calc_proto_resp.h"
#define ERROR_INVALID_REQUEST          101
#define ERROR_INVALID_REQUEST_ID       102
#define ERROR_INVALID_REQUEST_METHOD   103
#define ERROR_INVALID_REQUEST_OPERAND1 104
#define ERROR_INVALID_REQUEST_OPERAND2 105
#define ERROR_INVALID_RESPONSE         201
#define ERROR_INVALID_RESPONSE_REQ_ID  202
#define ERROR_INVALID_RESPONSE_STATUS  203
#define ERROR_INVALID_RESPONSE_RESULT  204
#define ERROR_UNKNOWN  220
struct buffer_t {
  char* data;
  int len;
};
struct calc_proto_ser_t;
typedef void (*req_cb_t)(
        void* owner_obj,
        struct calc_proto_req_t);
typedef void (*resp_cb_t)(
        void* owner_obj,
        struct calc_proto_resp_t);
typedef void (*error_cb_t)(
        void* owner_obj,
        const int req_id,
        const int error_code);
struct calc_proto_ser_t* calc_proto_ser_new();
void calc_proto_ser_delete(
        struct calc_proto_ser_t* ser);
void calc_proto_ser_ctor(
        struct calc_proto_ser_t* ser,
        void* owner_obj,
        int ring_buffer_size);
void calc_proto_ser_dtor(
        struct calc_proto_ser_t* ser);
void* calc_proto_ser_get_context(
        struct calc_proto_ser_t* ser);
void calc_proto_ser_set_req_callback(
        struct calc_proto_ser_t* ser,
        req_cb_t cb);
void calc_proto_ser_set_resp_callback(
        struct calc_proto_ser_t* ser,
        resp_cb_t cb);
void calc_proto_ser_set_error_callback(
        struct calc_proto_ser_t* ser,
        error_cb_t cb);
void calc_proto_ser_server_deserialize(
        struct calc_proto_ser_t* ser,
        struct buffer_t buffer,
        bool_t* req_found);
struct buffer_t calc_proto_ser_server_serialize(
        struct calc_proto_ser_t* ser,
        const struct calc_proto_resp_t* resp);
void calc_proto_ser_client_deserialize(
        struct calc_proto_ser_t* ser,
        struct buffer_t buffer,
        bool_t* resp_found);
struct buffer_t calc_proto_ser_client_serialize(
        struct calc_proto_ser_t* ser,
        const struct calc_proto_req_t* req);
#endif
```

代码框 20-7 [calcser/calc_proto_ser.h]：序列化器类的公共接口

除了创建和销毁序列化器对象所需的构造函数和析构函数之外，我们还有一对应由服务器进程使用的函数，以及另一对应由客户端进程使用的函数。

在客户端，我们序列化请求对象，并反序列化响应消息。同时，在服务器端，我们反序列化请求消息，并序列化响应对象。

除了序列化和反序列化函数之外，我们还有三个 *回调函数*：

+   接收从底层通道反序列化的请求对象的回调

+   接收从底层通道反序列化的响应对象的回调

+   接收序列化或反序列化失败时错误的回调

这些回调由客户端和服务器进程用于接收传入的请求和响应，以及序列化和反序列化消息过程中发现的错误。

现在，让我们更深入地看看服务器端的序列化/反序列化函数。

### 服务器端序列化/反序列化函数

我们有两个函数用于服务器进程序列化响应对象和反序列化请求消息。我们首先从响应序列化函数开始。

以下代码框包含响应序列化函数 `calc_proto_ser_server_serialize` 的代码：

```cpp
struct buffer_t calc_proto_ser_server_serialize(
    struct calc_proto_ser_t* ser,
    const struct calc_proto_resp_t* resp) {
  struct buffer_t buff;
  char resp_result_str[64];
  _serialize_double(resp_result_str, resp->result);
  buff.data = (char*)malloc(64 * sizeof(char));
  sprintf(buff.data, "%d%c%d%c%s%c", resp->req_id,
          FIELD_DELIMITER, (int)resp->status, FIELD_DELIMITER,
      resp_result_str, MESSAGE_DELIMITER);
  buff.len = strlen(buff.data);
  return buff;
}
```

代码框 20-8 [calcser/calc_proto_ser.c]：服务器端响应序列化函数

如您所见，`resp` 是一个指向需要序列化的响应对象的指针。此函数返回一个 `buffer_t` 对象，该对象在 `calc_proto_ser.h` 头文件中声明如下：

```cpp
struct buffer_t {
  char* data;
  int len;
};
```

代码框 20-9 [calcser/calc_proto_ser.h]：`buffer_t` 的定义

序列化代码很简单，主要由一个创建响应字符串消息的 `sprintf` 语句组成。现在，让我们看看请求反序列化函数。反序列化通常更难实现，如果您查看代码库并跟踪函数调用，您会看到它可以多么复杂。

*代码框 20-9* 包含请求反序列化函数：

```cpp
void calc_proto_ser_server_deserialize(
    struct calc_proto_ser_t* ser,
    struct buffer_t buff,
    bool_t* req_found) {
  if (req_found) {
    *req_found = FALSE;
  }
  _deserialize(ser, buff, _parse_req_and_notify,
          ERROR_INVALID_REQUEST, req_found);
}
```

代码框 20-9 [calcser/calc_proto_ser.c]：服务器端请求反序列化函数

前一个函数看起来很简单，但实际上它使用了 `_deserialize` 和 `_parse_req_and_notify` 私有函数。这些函数在 `calc_proto_ser.c` 文件中定义，该文件包含 `Serializer` 类的实际实现。

将我们为提到的私有函数编写的代码引入并讨论可能会非常复杂，超出了本书的范围，但为了给您一个概念，尤其是当您想阅读源代码时，反序列化器使用一个固定长度的 *环形缓冲区* 并尝试找到 `$` 作为消息分隔符。

每当它找到`$`时，它就会调用函数指针，在这个例子中，它指向`_parse_req_and_notify`函数（在`_deserialize`函数中传入的第三个参数）。`_parse_req_and_notify`函数试图提取字段并恢复请求对象。然后，它通知已注册的*观察者*，在这种情况下是等待通过回调函数接收请求的服务器对象，以继续处理请求对象。

现在，让我们看看客户端使用的函数。

### 客户端序列化/反序列化函数

就像服务器端一样，客户端也有两个函数。一个用于序列化请求对象，另一个用于反序列化传入的响应。

我们从请求序列化器开始。你可以在*代码框 20-10*中看到其定义：

```cpp
struct buffer_t calc_proto_ser_client_serialize(
    struct calc_proto_ser_t* ser,
    const struct calc_proto_req_t* req) {
  struct buffer_t buff;
  char req_op1_str[64];
  char req_op2_str[64];
  _serialize_double(req_op1_str, req->operand1);
  _serialize_double(req_op2_str, req->operand2);
  buff.data = (char*)malloc(64 * sizeof(char));
  sprintf(buff.data, "%d%c%s%c%s%c%s%c", req->id, FIELD_DELIMITER,
          method_to_str(req->method), FIELD_DELIMITER,
          req_op1_str, FIELD_DELIMITER, req_op2_str,
          MESSAGE_DELIMITER);
  buff.len = strlen(buff.data);
  return buff;
}
```

代码框 20-10 [calcser/calc_proto_ser.c]：客户端请求序列化函数

正如你所见，它接受一个请求对象并返回一个`buffer`对象，与服务器端响应序列化器完全相同。它甚至使用了相同的技巧；使用`sprintf`语句创建请求消息。

*代码框 20-11* 包含响应反序列化函数：

```cpp
void calc_proto_ser_client_deserialize(
    struct calc_proto_ser_t* ser,
    struct buffer_t buff, bool_t* resp_found) {
  if (resp_found) {
    *resp_found = FALSE;
  }
  _deserialize(ser, buff, _parse_resp_and_notify,
          ERROR_INVALID_RESPONSE, resp_found);
}
```

代码框 20-11 [calcser/calc_proto_ser.c]：客户端响应反序列化函数

正如你所见，使用了相同的机制，并且使用了一些类似的私有函数。强烈建议仔细阅读这些源代码，以便更好地理解代码的各个部分是如何组合在一起以实现最大程度的代码复用。

我们不会深入探讨`Serializer`类；深入代码并找出它是如何工作的，这取决于你。

现在我们有了序列化库，我们可以继续编写客户端和服务器程序。拥有一个基于协议序列化对象和反序列化消息的库是编写多进程软件的重要一步。请注意，部署是单主机还是包含多个主机无关紧要；进程应该能够相互理解，并且应该已经定义了适当的应用程序协议。

在跳转到关于套接字编程的代码之前，我们还需要解释一件事：计算器服务。它是服务器进程的核心，并执行实际的计算。

## 计算器服务

计算器服务是我们示例的核心逻辑。请注意，这个逻辑应该独立于底层 IPC 机制工作。下面的代码显示了计算器服务类的声明。

正如你所见，它被设计成即使在非常简单的程序中也可以使用，只需要一个`main`函数，以至于它甚至不做任何 IPC 操作：

```cpp
#ifndef CALC_SERVICE_H
#define CALC_SERVICE_H
#include <types.h>
static const int CALC_SVC_OK = 0;
static const int CALC_SVC_ERROR_DIV_BY_ZERO = -1;
struct calc_service_t;
struct calc_service_t* calc_service_new();
void calc_service_delete(struct calc_service_t*);
void calc_service_ctor(struct calc_service_t*);
void calc_service_dtor(struct calc_service_t*);
void calc_service_reset_mem(struct calc_service_t*);
double calc_service_get_mem(struct calc_service_t*);
double calc_service_add(struct calc_service_t*, double, double b,
    bool_t mem);
double calc_service_sub(struct calc_service_t*, double, double b,
    bool_t mem);
double calc_service_mul(struct calc_service_t*, double, double b,
    bool_t mem);
int calc_service_div(struct calc_service_t*, double,
        double, double*);
#endif
```

代码框 20-12 [calcsvc/calc_service.h]：计算器服务类的公共接口

如您所见，前面的类甚至有自己的错误类型。输入参数是纯 C 类型，并且它完全不依赖于 IPC 相关或序列化相关的类或类型。由于它是作为一个独立的逻辑单元隔离的，我们将其编译成一个名为`libcalcsvc.a`的独立静态库。

每个服务器进程都必须使用计算器服务对象来进行实际的计算。这些对象通常被称为*服务对象*。因此，最终的服务器程序必须与这个库链接。

在我们继续之前的一个重要注意事项：如果对于特定的客户端，计算不需要特定的上下文，那么只需要一个服务对象就足够了。换句话说，如果一个客户端的服务不需要我们记住该客户端之前请求的任何状态，那么我们可以使用一个*单例*服务对象。我们称之为*无状态服务对象*。

相反，如果处理当前请求需要了解之前请求中的某些信息，那么对于每个客户端，我们需要有一个特定的服务对象。这种情况适用于我们的计算器项目。正如您所知，计算器有一个针对每个客户端独特的内部存储。因此，我们不能为两个客户端使用同一个对象。这些对象被称为*有状态的服务对象*。

总结我们上面所说的，对于每个客户端，我们必须创建一个新的服务对象。这样，每个客户端都有自己的计算器，并拥有自己专用的内部存储。计算器服务对象是有状态的，并且需要加载一些状态（内部存储的值）。

现在，我们处于一个很好的位置来继续前进，讨论各种类型的套接字，并在计算器项目的上下文中给出示例。

# Unix 域套接字

从上一章，我们知道如果我们打算在同一台机器上的两个进程之间建立连接，UDS 是最佳选择之一。在这一章中，我们扩展了我们的讨论，并更多地讨论了基于推的 IPC 技术，以及流和数据报通道。现在，是时候将之前和当前章节的知识结合起来，看看 UDS 的实际应用了。

在本节中，我们有四个小节专门讨论在监听器侧或连接器侧的进程，并在流或数据报通道上操作。所有这些进程都在使用 UDS。我们根据上一章讨论的序列，逐步说明它们建立通道的步骤。作为第一个进程，我们从在流通道上操作的监听器进程开始。这将是一个*流服务器*。

## UDS 流服务器

如果您还记得上一章的内容，我们为传输通信中的监听器和连接器端有多个序列。服务器位于监听器的位置。因此，它应该遵循监听器序列。更具体地说，由于我们本节讨论的是流通道，它应该遵循流监听器序列。

作为该序列的一部分，服务器需要首先创建一个 socket 对象。在我们的计算器项目中，愿意通过 UDS 接收连接的流服务器进程必须遵循相同的序列。

以下代码片段位于计算器服务器程序的主函数中，如*代码框 20-13*所示，该过程首先创建了一个`socket`对象：

```cpp
int server_sd = socket(AF_UNIX, SOCK_STREAM, 0);
if (server_sd == -1) {
  fprintf(stderr, "Could not create socket: %s\n", strerror(errno));
  exit(1);
}
```

代码框 20-13 [server/unix/stream/main.c]: 创建流 UDS 对象

如您所见，`socket`函数用于创建一个 socket 对象。此函数包含在`<sys/socket.h>`中，这是一个 POSIX 头文件。请注意，这只是一个 socket 对象，而且尚未确定它将是一个客户端 socket 还是一个服务器 socket。只有后续的函数调用才能确定这一点。

正如我们在上一章中解释的，每个 socket 对象都有三个属性。这些属性由传递给`socket`函数的三个参数确定。这些参数分别指定了在该 socket 对象上使用的地址族、类型和协议。

根据流监听器序列，特别是关于创建 socket 对象之后的 UDS，服务器程序必须将其绑定到一个*socket 文件*。因此，下一步是将 socket 绑定到 socket 文件。计算器项目中使用了*代码框 20-14*来将 socket 对象绑定到由`sock_file`字符数组指定的预定义路径上的文件：

```cpp
struct sockaddr_un addr;
memset(&addr, 0, sizeof(addr));
addr.sun_family = AF_UNIX;
strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
int result = bind(server_sd, (struct sockaddr*)&addr, sizeof(addr));
if (result == -1) {
  close(server_sd);
  fprintf(stderr, "Could not bind the address: %s\n", strerror(errno));
  exit(1);
}
```

代码框 20-14 [server/unix/stream/main.c]: 将流 UDS 对象绑定到由 sock_file 字符数组指定的 socket 文件

前面的代码有两个步骤。第一步是创建一个名为`addr`的`struct sockaddr_un`类型的实例，然后通过将其指向 socket 文件来初始化它。第二步是将`addr`对象传递给`bind`函数，以便让它知道应该将哪个 socket 文件*绑定*到 socket 对象。只有当没有其他 socket 对象绑定到相同的 socket 文件时，`bind`函数调用才会成功。因此，在 UDS 中，两个 socket 对象，可能位于不同的进程中，不能绑定到同一个 socket 文件。

**注意**：

在 Linux 中，UDS 可以绑定到*抽象 socket 地址*。当没有文件系统挂载用于创建 socket 文件时，它们非常有用。一个以空字符`\0`开头的字符串可以用来初始化前面的代码框中的地址结构`addr`，然后提供的名称绑定到内核中的 socket 对象。提供的名称在系统中应该是唯一的，并且不应该有其他 socket 对象绑定到它。

关于套接字文件路径的进一步说明，在大多数 Unix 系统中，路径长度不能超过 104 字节。然而，在 Linux 系统中，这个长度是 108 字节。请注意，用于保存套接字文件路径的字符串变量始终包含一个额外的空字符，作为 C 中的`char`数组。因此，实际上，根据操作系统，可以使用 103 和 107 字节作为套接字文件路径的一部分。

如果`bind`函数返回`0`，则表示绑定成功，您可以继续配置 backlog 的大小；这是绑定端点后流监听序列的下一步。

以下代码展示了如何为监听 UDS 的流计算服务器配置 backlog：

```cpp
result = listen(server_sd, 10);
if (result == -1) {
  close(server_sd);
  fprintf(stderr, "Could not set the backlog: %s\n", strerror(errno));
  exit(1);
}
```

代码框 20-15 [server/unix/stream/main.c]：配置已绑定流套接字的 backlog 大小

`listen`函数配置已绑定套接字的 backlog 大小。正如我们在上一章中解释的，当繁忙的服务器进程无法接受更多传入客户端时，一定数量的这些客户端可以在 backlog 中等待，直到服务器程序可以处理它们。这是在接受客户端之前准备流套接字的一个基本步骤。

根据我们在流监听序列中的内容，在流套接字绑定并配置其 backlog 大小后，我们可以开始接受新客户端。*代码框 20-16*展示了如何接受新客户端：

```cpp
while (1) {
  int client_sd = accept(server_sd, NULL, NULL);
  if (client_sd == -1) {
    close(server_sd);
    fprintf(stderr, "Could not accept the client: %s\n",
        strerror(errno));
    exit(1);
  }
  ...
}
```

代码框 20-16 [server/unix/stream/main.c]：在流监听套接字上接受新客户端

魔法在于`accept`函数，每当接收到新的客户端时，它都会返回一个新的套接字对象。返回的套接字对象指向服务器和已接受客户端之间的底层流通道。请注意，每个客户端都有自己的流通道，因此也有自己的套接字描述符。

注意，如果流监听套接字是阻塞的（默认情况下是阻塞的），`accept`函数将阻塞执行，直到接收到新的客户端。换句话说，如果没有传入客户端，调用`accept`函数的线程将阻塞在其后面。

现在，让我们在一个地方看到上述步骤。以下代码框展示了计算器项目中的流服务器，它监听 UDS：

```cpp
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stream_server_core.h>
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Delete the previously created socket file if it exists.
  unlink(sock_file);
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = bind(server_sd,
 (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not bind the address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Prepare backlog ------------------
  result = listen(server_sd, 10);
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not set the backlog: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 4\. Start accepting clients ---------
  accept_forever(server_sd);
  return 0;
}
```

代码框 20-17 [server/unix/stream/main.c]：监听 UDS 端点的流计算服务的主函数

应该很容易找到执行初始化服务器套接字上述步骤的代码块。唯一缺少的是客户端接受代码。接受新客户端的实际代码放在一个单独的函数中，该函数名为`accept_forever`。请注意，此函数是阻塞的，它会阻塞主线程直到服务器停止。

在下面的代码框中，你可以看到 `accept_forever` 函数的定义。该函数是位于 `srvcore` 目录的服务器通用库的一部分。这个函数应该在那里，因为它的定义对于其他流套接字（如 TCP 套接字）也是相同的。因此，我们可以重用现有的逻辑，而不是再次编写它：

```cpp
void accept_forever(int server_sd) {
  while (1) {
    int client_sd = accept(server_sd, NULL, NULL);
    if (client_sd == -1) {
      close(server_sd);
      fprintf(stderr, "Could not accept the client: %s\n",
              strerror(errno));
      exit(1);
    }
    pthread_t client_handler_thread;
    int* arg = (int *)malloc(sizeof(int));
    *arg = client_sd;
    int result = pthread_create(&client_handler_thread, NULL,
            &client_handler, arg);
    if (result) {
      close(client_sd);
      close(server_sd);
      free(arg);
      fprintf(stderr, "Could not start the client handler thread.\n");
      exit(1);
    }
  }
}
```

代码框 20-18 [server/srvcore/stream_server_core.c]：在监听 UDS 端点的流套接字上接受新客户端的函数

正如你在前面的代码框中所见，在接收新客户端后，我们启动了一个新的线程来处理客户端。这实际上包括从客户端的通道读取字节，将读取的字节传递给反序列化器，并在检测到请求时产生适当的响应。

为每个客户端创建一个新的线程通常是每个在阻塞流通道上操作的服务器进程的通用模式，无论套接字类型如何。因此，在这种情况下，多线程及其相关主题变得极其重要。

**注意**：

关于非阻塞流通道，通常使用一种称为 *事件循环* 的不同方法。

当你拥有客户端的套接字对象时，你可以用它来从客户端读取，也可以向客户端写入。如果我们遵循在 `srvcore` 库中迄今为止所采取的路径，下一步是查看客户端线程的伴随函数；`client_handler`。该函数可以在代码库中 `accept_forever` 旁边找到。接下来，你可以看到包含函数定义的代码框：

```cpp
void* client_handler(void *arg) {
  struct client_context_t context;
  context.addr = (struct client_addr_t*)
      malloc(sizeof(struct client_addr_t));
  context.addr->sd = *((int*)arg);
  free((int*)arg);
 context.ser = calc_proto_ser_new();
  calc_proto_ser_ctor(context.ser, &context, 256);
  calc_proto_ser_set_req_callback(context.ser, request_callback);
  calc_proto_ser_set_error_callback(context.ser, error_callback);
  context.svc = calc_service_new();
  calc_service_ctor(context.svc);
  context.write_resp = &stream_write_resp;
  int ret;
  char buffer[128];
  while (1) {
    int ret = read(context.addr->sd, buffer, 128);
    if (ret == 0 || ret == -1) {
      break;
    }
    struct buffer_t buf;
    buf.data = buffer; buf.len = ret;
    calc_proto_ser_server_deserialize(context.ser, buf, NULL);
  }
  calc_service_dtor(context.svc);
  calc_service_delete(context.svc);
  calc_proto_ser_dtor(context.ser);
  calc_proto_ser_delete(context.ser);
  free(context.addr);
  return NULL;
}
```

代码框 20-19 [server/srvcore/stream_server_core.c]：处理客户端线程的伴随函数

关于前面的代码有很多细节，但有一些重要的细节我想提一下。正如你所见，我们正在使用 `read` 函数从客户端读取数据块。如果你记得，`read` 函数接受一个文件描述符，但在这里我们传递的是一个套接字描述符。这表明，尽管在 I/O 函数方面文件描述符和套接字描述符之间存在差异，我们仍然可以使用相同的 API。

在前面的代码中，我们从输入读取字节数据，并通过调用 `calc_proto_ser_server_deserialize` 函数将它们传递给反序列化器。在完全反序列化一个请求之前，可能需要调用这个函数三到四次。这高度依赖于从输入读取的字块大小以及通过通道传输的消息长度。

进一步来说，每个客户端都有自己的序列化对象。这也适用于计算器服务对象。这些对象作为同一线程的一部分被创建和销毁。

关于前面的代码框的最后一点，我们正在使用一个函数将响应写回客户端。该函数是`stream_write_response`，它旨在在流套接字上使用。这个函数可以在前面的代码框所在的同一文件中找到。接下来，你可以看到这个函数的定义：

```cpp
void stream_write_resp(
        struct client_context_t* context,
        struct calc_proto_resp_t* resp) {
  struct buffer_t buf =
      calc_proto_ser_server_serialize(context->ser, resp);
  if (buf.len == 0) {
    close(context->addr->sd);
    fprintf(stderr, "Internal error while serializing response\n");
    exit(1);
  }
  int ret = write(context->addr->sd, buf.data, buf.len);
  free(buf.data);
  if (ret == -1) {
    fprintf(stderr, "Could not write to client: %s\n",
            strerror(errno));
    close(context->addr->sd);
    exit(1);
  } else if (ret < buf.len) {
    fprintf(stderr, "WARN: Less bytes were written!\n");
    exit(1);
  }
}
```

代码框 20-20 [server/srvcore/stream_server_core.c]: 用于将响应写回客户端的函数

如前所述的代码所示，我们正在使用`write`函数将消息写回客户端。正如我们所知，`write`函数可以接受文件描述符，但似乎套接字描述符也可以使用。所以，这清楚地表明 POSIX I/O API 对文件描述符和套接字描述符都有效。

上述语句也适用于`close`函数。正如你所见，我们已用它来终止一个连接。当我们知道它对文件描述符也有效时，传递套接字描述符就足够了。

现在我们已经了解了 UDS 流服务器的一些最重要的部分，并对它的操作有了大致的了解，是时候继续讨论 UDS 流客户端了。当然，代码中还有很多我们没有讨论的地方，但你应该花时间仔细研究它们。

## UDS 流客户端

与上一节中描述的服务器程序一样，客户端也需要首先创建一个套接字对象。记住，我们现在需要遵循流连接器序列。它使用与服务器完全相同的代码，使用完全相同的参数来指示它需要一个 UDS。之后，它需要通过指定 UDS 端点来连接到服务器进程，就像服务器那样。当流通道建立后，客户端进程可以使用打开的套接字描述符来读取和写入通道。

接下来，你可以看到连接到 UDS 端点的流客户端的`main`函数：

```cpp
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Connect to server ---------------------
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = connect(conn_sd,
 (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could no connect: %s\n", strerror(errno));
    exit(1);
  }
 stream_client_loop(conn_sd);
  return 0;
}
```

代码框 20-21 [client/unix/stream/main.c]: 连接到 UDS 端点的流客户端的主函数

如你所见，代码的第一部分与服务器代码非常相似，但之后，客户端调用`connect`而不是`bind`。请注意，地址准备代码与服务器完全相同。

当`connect`成功返回时，它已经将`conn_sd`套接字描述符关联到打开的通道。因此，从现在开始，`conn_sd`可以用来与服务器通信。我们将其传递给`stream_client_loop`函数，该函数启动客户端的命令行并执行客户端执行的其他操作。它是一个阻塞函数，运行客户端直到它退出。

注意，客户端也使用`read`和`write`函数在服务器之间来回传输消息。*代码框 20-22*包含了`stream_client_loop`函数的定义，这是客户端通用库的一部分，所有流客户端都会使用它，无论套接字类型如何，并且 UDS 和 TCP 套接字之间是共享的。正如你所见，它使用`write`函数向服务器发送一个序列化的请求消息：

```cpp
void stream_client_loop(int conn_sd) {
  struct context_t context;
  context.sd = conn_sd;
  context.ser = calc_proto_ser_new();
  calc_proto_ser_ctor(context.ser, &context, 128);
  calc_proto_ser_set_resp_callback(context.ser, on_response);
  calc_proto_ser_set_error_callback(context.ser, on_error);
  pthread_t reader_thread;
 pthread_create(&reader_thread, NULL,
stream_response_reader, &context);
  char buf[128];
  printf("? (type quit to exit) ");
  while (1) {
    scanf("%s", buf);
    int brk = 0, cnt = 0;
    struct calc_proto_req_t req;
    parse_client_input(buf, &req, &brk, &cnt);
    if (brk) {
      break;
    }
    if (cnt) {
      continue;
    }
    struct buffer_t ser_req =
        calc_proto_ser_client_serialize(context.ser, &req);
    int ret = write(context.sd, ser_req.data, ser_req.len);
    if (ret == -1) {
      fprintf(stderr, "Error while writing! %s\n",
              strerror(errno));
      break;
    }
    if (ret < ser_req.len) {
      fprintf(stderr, "Wrote less than anticipated!\n");
      break;
    }
    printf("The req(%d) is sent.\n", req.id);
  }
  shutdown(conn_sd, SHUT_RD);
  calc_proto_ser_dtor(context.ser);
  calc_proto_ser_delete(context.ser);
  pthread_join(reader_thread, NULL);
  printf("Bye.\n");
}
```

代码框 20-22 [client/clicore/stream_client_core.c]：执行流客户端的函数

正如你在前面的代码中所见，每个客户端进程只有一个序列化对象，这是有道理的。这与服务器进程相反，其中每个客户端都有一个单独的序列化对象。

更重要的是，客户端进程为从服务器端读取响应启动了一个单独的线程。这是因为从服务器进程读取是一个阻塞任务，应该在单独的执行流中完成。

作为主线程的一部分，我们有客户端的命令行，它通过终端接收用户的输入。正如你所见，主线程在退出时加入读取线程，并等待其完成。

关于前面代码的进一步说明，客户端进程使用相同的 I/O API 从流通道读取和写入。正如我们之前所说的，使用了`read`和`write`函数，`write`函数的使用可以在*代码框 20-22*中看到。

在接下来的部分，我们将讨论数据报通道，但仍然使用 UDS 来完成这个目的。我们首先从数据报服务器开始。

## UDS 数据报服务器

如果你记得上一章的内容，数据报进程在传输传输方面有自己的监听器和连接器序列。现在，是时候展示如何基于 UDS 开发数据报服务器了。

根据数据报监听器序列，进程首先需要创建一个套接字对象。以下代码框展示了这一点：

```cpp
int server_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
if (server_sd == -1) {
  fprintf(stderr, "Could not create socket: %s\n",
          strerror(errno));
  exit(1);
}
```

代码框 20-23 [server/unix/datagram/main.c]：创建一个用于数据报通道的 UDS 对象

你可以看到我们使用了`SOCK_DGRAM`而不是`SOCK_STREAM`。这意味着套接字对象将操作在数据报通道上。其他两个参数保持不变。

作为数据报监听器序列的第二步，我们需要将套接字绑定到 UDS 端点。正如我们之前所说的，这是一个套接字文件。这一步与流服务器完全相同，因此我们在这里不展示它，你可以在*代码框 20-14*中看到它。

对于数据报监听器进程，这些步骤是唯一需要执行的，并且与数据报套接字相关的配置没有队列。更重要的是，没有客户端接受阶段，因为我们不能在某些专用的 1-to-1 通道上有流连接。

接下来，你可以看到数据报服务器在 UDS 端点监听的`main`函数，这是计算器项目的一部分：

```cpp
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Delete the previously created socket file if it exists.
  unlink(sock_file);
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = bind(server_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not bind the address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Start serving requests ---------
  serve_forever(server_sd);
  return 0;
}
```

代码框 20-24 [server/unix/datagram/main.c]：监听 UDS 端点的数据报服务器的主函数

如你所知，数据报通道是无连接的，它们的工作方式不像流通道。换句话说，两个进程之间不能有一个专用的 1-to-1 连接。因此，进程只能通过通道传输数据报。客户端进程只能发送一些单独和独立的 数据报，同样，服务器进程只能接收数据报并作为响应发送其他数据报。

因此，数据报通道的关键之处在于请求和响应消息应该适合在一个数据报中。否则，它们不能被分成两个数据报，服务器或客户端也无法处理这些消息。幸运的是，计算器项目中的消息大多数足够短，可以适合在一个数据报中。

数据报的大小高度依赖于底层通道。例如，对于数据报 UDS 来说，这是相当灵活的，因为它通过内核进行，但对于 UDP 套接字，你将受到网络配置的限制。关于 UDS，以下链接可以给你一个更好的想法，了解如何设置正确的大小：[`stackoverflow.com/questions/21856517/whats-the-practical-limit-on-the-size-of-single-packet-transmitted-over-domain`](https://stackoverflow.com/questions/21856517/whats-the-practical-limit-on-the-size-of-single-packet-transmitted-over-domain)。

关于数据报和流套接字，我们可以提到的另一个区别是用于在它们之间传输数据的 I/O API。虽然 `read` 和 `write` 函数仍然可以像流套接字一样用于数据报套接字，但我们使用其他函数从数据报通道读取和发送。通常使用 `recvfrom` 和 `sendto` 函数。

这是因为在流套接字中，通道是专用的，当你向一个通道写入时，两端都是确定的。至于数据报套接字，我们只有一个通道被许多方使用。因此，我们可能会失去对特定数据报的所有权。这些函数可以跟踪并将数据报发送回期望的过程。

接下来，你可以在 `main` 函数的末尾找到 *代码框 20-24* 中使用的 `serve_forever` 函数的定义。这个函数属于服务器通用库，并且专门用于数据报服务器，无论套接字类型如何。你可以清楚地看到 `recvfrom` 函数是如何被使用的：

```cpp
void serve_forever(int server_sd) {
  char buffer[64];
  while (1) {
    struct sockaddr* sockaddr = sockaddr_new();
    socklen_t socklen = sockaddr_sizeof();
    int read_nr_bytes = recvfrom(server_sd, buffer,
 sizeof(buffer), 0, sockaddr, &socklen);
    if (read_nr_bytes == -1) {
      close(server_sd);
      fprintf(stderr, "Could not read from datagram socket: %s\n",
              strerror(errno));
      exit(1);
    }
    struct client_context_t context;
    context.addr = (struct client_addr_t*)
 malloc(sizeof(struct client_addr_t));
    context.addr->server_sd = server_sd;
    context.addr->sockaddr = sockaddr;
    context.addr->socklen = socklen;
    context.ser = calc_proto_ser_new();
    calc_proto_ser_ctor(context.ser, &context, 256);
    calc_proto_ser_set_req_callback(context.ser, request_callback);
    calc_proto_ser_set_error_callback(context.ser, error_callback);
    context.svc = calc_service_new();
    calc_service_ctor(context.svc);
    context.write_resp = &datagram_write_resp;
    bool_t req_found = FALSE;
    struct buffer_t buf;
    buf.data = buffer;
    buf.len = read_nr_bytes;
    calc_proto_ser_server_deserialize(context.ser, buf, &req_found);
    if (!req_found) {
      struct calc_proto_resp_t resp;
      resp.req_id = -1;
      resp.status = ERROR_INVALID_RESPONSE;
      resp.result = 0.0;
      context.write_resp(&context, &resp);
    }
    calc_service_dtor(context.svc);
    calc_service_delete(context.svc);
    calc_proto_ser_dtor(context.ser);
    calc_proto_ser_delete(context.ser);
    free(context.addr->sockaddr);
    free(context.addr);
  }
}
```

代码框 20-25 [server/srvcore/datagram_server_core.c]：处理服务器通用库中找到的数据报的函数，并专门用于数据报服务器

如您在先前的代码框中看到的，数据报服务器是一个单线程程序，并且在其周围没有多线程。不仅如此，它对每个数据报进行单独和独立的操作。它接收一个数据报，反序列化其内容并创建请求对象，通过服务对象处理请求，序列化响应对象并将其放入一个新的数据报中，然后将它发送回拥有原始数据报的进程。对于每个传入的数据报，它都会重复进行相同的周期。

请注意，每个数据报都有自己的序列化对象和自己的服务对象。我们可以设计成只有一个序列化对象和一个服务对象适用于所有数据报。这可能对您思考如何实现以及为什么这可能不适合计算器项目是有趣的。这是一个有争议的讨论，您可能会从不同的人那里得到不同的观点。

注意，在*代码框 20-25*中，我们在接收到数据报时存储了数据报的客户端地址。稍后，我们可以使用这个地址直接向该客户端写入。看看我们如何将数据报写回发送客户端是值得一看的。就像流服务器一样，我们为此使用了一个函数。*代码框 20-26*显示了`datagram_write_resp`函数的定义。该函数位于数据报服务器公共库中，紧邻`serve_forever`函数：

```cpp
void datagram_write_resp(struct client_context_t* context,
        struct calc_proto_resp_t* resp) {
  struct buffer_t buf =
      calc_proto_ser_server_serialize(context->ser, resp);
  if (buf.len == 0) {
    close(context->addr->server_sd);
    fprintf(stderr, "Internal error while serializing object.\n");
    exit(1);
  }
  int ret = sendto(context->addr->server_sd, buf.data, buf.len,
 0, context->addr->sockaddr, context->addr->socklen);
  free(buf.data);
  if (ret == -1) {
    fprintf(stderr, "Could not write to client: %s\n",
            strerror(errno));
    close(context->addr->server_sd);
    exit(1);
  } else if (ret < buf.len) {
    fprintf(stderr, "WARN: Less bytes were written!\n");
    close(context->addr->server_sd);
    exit(1);
  }
}
```

代码框 20-26 [server/srvcore/datagram_server_core.c]：将数据报写回客户端的函数

您可以看到我们使用了排序后的客户端地址，并将其与序列化的响应消息一起传递给`sendto`函数。其余的由操作系统处理，数据报直接发送回发送客户端。

既然我们已经足够了解数据报服务器以及如何使用套接字，让我们来看看数据报客户端，它使用的是相同类型的套接字。

## UDS 数据报客户端

从技术角度来看，流客户端和数据报客户端非常相似。这意味着您应该看到几乎相同的整体结构，但在处理数据报而不是流通道时有一些差异。

但它们之间有一个很大的差异，这是相当独特且专门针对连接到 UDS 端点的数据报客户端的。

差异在于，数据报客户端需要绑定一个套接字文件，就像服务器程序一样，以便接收指向它的数据报。对于使用网络套接字的数据报客户端来说，情况并非如此，您很快就会看到。请注意，客户端应绑定不同的套接字文件，而不是服务器的套接字文件。

这种差异背后的主要原因是服务器程序需要一个地址来发送响应，如果数据报客户端没有绑定套接字文件，则没有端点绑定到客户端套接字文件。但是，对于网络套接字来说，客户端始终有一个对应的套接字描述符，它绑定到一个 IP 地址和一个端口，因此这个问题不会发生。

如果我们忽略这个差异，我们可以看到代码是多么相似。在*代码框 20-26*中，你可以看到数据报计算器客户端的`main`函数：

```cpp
int main(int argc, char** argv) {
  char server_sock_file[] = "/tmp/calc_svc.sock";
  char client_sock_file[] = "/tmp/calc_cli.sock";
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the client socket file ------------
  // Delete the previously created socket file if it exists.
  unlink(client_sock_file);
  // Prepare the client address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, client_sock_file,
          sizeof(addr.sun_path) - 1);
  int result = bind(conn_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could not bind the client address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Connect to server --------------------
  // Prepare the server address
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, server_sock_file,
          sizeof(addr.sun_path) - 1);
  result = connect(conn_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could no connect: %s\n", strerror(errno));
    exit(1);
  }
  datagram_client_loop(conn_sd);
  return 0;
}
```

代码框 20-26 [server/srvcore/datagram_server_core.c]: 将数据报文写回客户端的函数

正如我们之前所解释的，并且可以从代码中看到，客户端需要绑定一个套接字文件。当然，我们不得不在`main`函数的末尾调用一个不同的函数来启动客户端循环。数据报客户端调用`datagram_client_loop`函数。

如果你查看`datagram_client_loop`函数，你仍然会在流客户端和数据报客户端之间看到许多相似之处。尽管存在一些小的差异，但一个大的差异是使用`recvfrom`和`sendto`函数而不是`read`和`write`函数。在上一节中对这些函数的解释，对于数据报客户端仍然适用。

现在是时候讨论网络套接字了。正如你将看到的，客户端和服务器程序中的`main`函数是唯一在从 UDS 转换为网络套接字时发生变化的代码。

# 网络套接字

另一个广泛使用的套接字地址族是`AF_INET`。它简单地指代在网络上建立的所有通道。与没有分配协议名称的 UDS 流和数据报套接字不同，网络套接字之上存在两个知名协议。TCP 套接字在两个进程之间建立流通道，而 UDP 套接字建立的数据报通道可以被多个进程使用。

在接下来的章节中，我们将解释如何使用 TCP 和 UDP 套接字开发程序，并作为计算器项目的一部分展示一些真实示例。

## TCP 服务器

使用 TCP 套接字监听和接受多个客户端的程序，换句话说，是一个 TCP 服务器，它在两个方面的不同之处在于：首先，在调用`socket`函数时指定了不同的地址族，即`AF_INET`而不是`AF_UNIX`；其次，它使用了一个不同的结构来绑定所需的套接字地址。

尽管存在这两个差异，但从 I/O 操作的角度来看，TCP 套接字的其他一切都将与 UDP 套接字相同。我们应该注意，TCP 套接字是一个流套接字，因此为流套接字编写的代码也应该适用于 TCP 套接字。

如果我们回到计算器项目，我们期望看到的主要区别仅在于我们创建套接字对象并将其绑定到端点处的 `main` 函数。除此之外，其余的代码应该保持不变。事实上，这正是我们所看到的。以下代码框包含了 TCP 计算器服务器的 `main` 函数：

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(6666);
  ...
  // ----------- 3\. Prepare backlog ------------------
  ...
  // ----------- 4\. Start accepting clients ---------
  accept_forever(server_sd);
  return 0;
}
```

代码框 20-27 [server/tcp/main.c]：TCP 计算器客户端的 `main` 函数

如果您将前面的代码与 *代码框 20-17* 中看到的 `main` 函数进行比较，您将注意到我们之前解释过的差异。我们不是使用 `sockaddr_un` 结构，而是使用 `sockaddr_in` 结构来为绑定端点地址。`listen` 函数的使用相同，甚至调用了相同的 `accept_forever` 函数来处理传入的连接。

作为最后的说明，关于 TCP 套接字上的 I/O 操作，由于 TCP 套接字是一个流套接字，它继承了流套接字的所有属性；因此，它可以像任何其他流套接字一样使用。换句话说，相同的 `read`、`write` 和 `close` 函数都可以使用。

现在，让我们谈谈 TCP 客户端。

## TCP 客户端

再次强调，一切应该与在 UDS 上运行的流客户端非常相似。上一节中提到的差异对于连接器侧的 TCP 套接字仍然适用。变化再次仅限于 `main` 函数。

接下来，您可以查看 TCP 计算器客户端的 `main` 函数：

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_INET, SOCK_STREAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ------------ 2\. Connect to server-- ------------------
  // Find the IP address behind the hostname
  ...
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr = *((struct in_addr*)host_entry->h_addr);
  addr.sin_port = htons(6666);
  ...
  stream_client_loop(conn_sd);
  return 0;
}
```

代码框 20-27 [server/tcp/main.c]：TCP 计算器服务器的 `main` 函数

变化与我们在 TCP 服务器程序中看到的变化非常相似。使用了不同的地址族和不同的套接字地址结构。除此之外，其余的代码相同，因此我们不需要详细讨论 TCP 客户端。

由于 TCP 套接字是流套接字，我们可以使用相同的通用代码来处理新客户端。您可以通过调用 `stream_client_loop` 函数来看到这一点，该函数是计算器项目中的客户端通用库的一部分。现在，您应该明白为什么我们提取了两个通用库，一个用于客户端程序，一个用于服务器程序，以便编写更少的代码。当我们可以在两种不同的场景中使用相同的代码时，将其提取为库并在场景中重用总是最好的。

让我们来看看 UDP 服务器和客户端程序；我们会发现它们与我们所看到的 TCP 程序大致相似。

## UDP 服务器

UDP 套接字是网络套接字。除此之外，它们是数据报套接字。因此，我们预计将观察到我们在 TCP 服务器代码和数据报服务器代码（在 UDS 上操作）中编写的代码之间的高度相似性。

此外，无论在客户端还是服务器程序中使用，UDP 套接字与 TCP 套接字的主要区别在于 UDP 套接字的套接字类型是 `SOCK_DGRAM`。地址族保持不变，因为它们都是网络套接字。以下代码框包含了计算器 UDP 服务器的主体函数：

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_INET, SOCK_DGRAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(9999);
  ...
  // ----------- 3\. Start serving requests ---------
  serve_forever(server_sd);
  return 0;
}
```

代码框 20-28 [server/udp/main.c]：UDP 计算器服务器的主体函数

注意，UDP 套接字是数据报套接字。因此，为在 UDS 上操作的数据报套接字编写的所有代码仍然适用于它们。例如，我们必须使用 `recvfrom` 和 `sendto` 函数来处理 UDP 套接字。所以，如您所见，我们使用了相同的 `serve_forever` 函数来服务传入的数据报。这个函数是服务器通用库的一部分，旨在包含与数据报相关的代码。

关于 UDP 服务器代码，我们已经说得够多了。让我们看看 UDP 客户端代码的样子。

## UDP 客户端

UDP 客户端代码与 TCP 客户端代码非常相似，但它使用不同的套接字类型，并调用不同的函数来处理传入的消息，这个函数与基于 UDS 的数据报客户端使用的函数相同。您可以看到以下 `main` 函数：

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_INET, SOCK_DGRAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ------------ 2\. Connect to server-- ------------------
  ...
  // Prepare the address
  ...
  datagram_client_loop(conn_sd);
  return 0;
}
```

代码框 20-28 [client/udp/main.c]：UDP 计算器客户端的主体函数

那是本章的最后一个概念。在本章中，我们探讨了各种众所周知的套接字类型，并展示了如何在 C 中实现流和数据报通道的监听器和连接器序列。

计算器项目中有很多我们没有讨论的事情。因此，强烈建议您阅读代码，找到那些地方，并尝试阅读和理解它。一个完全工作的示例可以帮助您在真实应用中检验这些概念。

# 摘要

在本章中，我们讨论了以下主题：

+   我们在回顾 IPC 技术时介绍了各种类型的通信、通道、介质和套接字。

+   我们通过描述其应用协议和所使用的序列化算法来探索了一个计算器项目。

+   我们演示了如何使用 UDS 建立客户端-服务器连接，并展示了它们在计算器项目中的应用。

+   我们分别讨论了使用 Unix 域套接字建立的流和数据报通道。

+   我们演示了如何使用 TCP 和 UDP 套接字来创建客户端-服务器 IPC 通道，并在计算器示例中使用了它们。

下一章将介绍 C 语言与其他编程语言的集成。通过这种方式，我们可以在其他编程语言（如 Java）中加载并使用 C 库。作为下一章的一部分，我们将涵盖与 C++、Java、Python 和 Golang 的集成。
