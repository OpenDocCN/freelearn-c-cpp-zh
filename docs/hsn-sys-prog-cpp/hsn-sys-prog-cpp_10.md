# 第十章：使用 C++编程 POSIX 套接字

在本章中，您将学习如何使用 C++17 编程 POSIX 套接字，包括更常见的 C++范例，如**资源获取即初始化**（**RAII**）。首先，本章将讨论套接字是什么，以及 UDP 和 TCP 之间的区别。在向您介绍五个不同的示例之前，将详细解释 POSIX API。第一个示例将引导您通过使用 POSIX 套接字创建 UDP 回显服务器示例。第二个示例将使用 TCP 而不是 UDP 创建相同的示例，并解释其中的区别。第三个示例将扩展我们在以前章节中创建的现有调试记录器，而第四和第五个示例将解释如何安全地处理数据包。

在本章中，我们将涵盖以下主题：

+   POSIX 套接字

+   利用 C++和 RAII 进行套接字编程

+   TCP vs UDP

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请参见以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter10`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter10)。

# 从 POSIX 套接字开始

不幸的是，C++不包含本地网络库（希望 C++20 能够解决这个问题）。因此，需要使用 POSIX 套接字来执行 C++网络编程。POSIX 套接字 API 定义了使用标准 Unix 文件描述符范式发送和接收网络数据包的 API。在使用套接字进行编程时，必须创建服务器和客户端。服务器负责将特定端口绑定到套接字协议，该协议由套接字库的用户开发。客户端是连接到先前绑定端口的任何其他应用程序。服务器和客户端都有自己的 IP 地址。

在编程套接字时，除了选择地址类型（例如 IPv4 与 IPv6），通常程序员还必须在 UDP 与 TCP 之间进行选择。UDP 是一种无连接协议，不保证可靠发送数据包，其优势在于速度和简单性。UDP 通常用于不需要 100%接收的数据，例如在视频游戏中的位置。另一方面，TCP 是一种基于连接的协议，确保所有数据包按发送顺序接收，并且是其可靠性的典型协议。

# 从 API 开始

以下部分将详细解释不同的套接字 API。

# socket() API

所有 POSIX 套接字编程都始于使用`socket()` API 创建套接字文件描述符，其形式如下：

```cpp
int socket(int domain, int type, int protocol);
```

域定义了创建套接字时使用的地址类型。在大多数情况下，这将是 IPv4 的`AF_INET`或 IPv6 的`AF_INET6`。在本章的示例中，我们将使用`AF_INET`。类型字段通常采用`SOCK_STREAM`用于 TCP 连接或`SOCK_DGRAM`用于 UDP 连接，这两者都将在本章中进行演示。最后，此 API 中的协议字段将在所有示例中设置为`0`，告诉 API 使用指定套接字类型的默认协议。

完成此 API 后，将返回套接字文件描述符，这将是剩余 POSIX API 所需的。如果此 API 失败，则返回`-1`，并将`errno`设置为适当的错误代码。应注意`errno`不是线程安全的，其使用应谨慎处理。处理这些类型的错误的一个很好的方法是立即将`errno`转换为 C++异常，可以使用以下方法完成：

```cpp
if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
    throw std::runtime_error(strerror(errno));
}
```

在前面的示例中，创建了一个 IPv4 TCP 套接字。生成的文件描述符保存在内存变量`m_fd`中。使用 C++17 语法，检查文件描述符的有效性，如果报告错误（即`-1`），则抛出异常。为了提供错误的人类可读版本，`errno`被转换为字符串使用`strerror()`。这不仅提供了`errno`的字符串版本，还确保记录错误的过程不会在过程中更改`errno`，如果使用更复杂的方法可能会发生这种情况。

最后，当套接字不再需要时，应像使用 POSIX`close()`函数关闭任何其他文件描述符一样关闭。应该注意，大多数 POSIX 操作系统在应用程序关闭时仍然打开的套接字将自动关闭。

为了防止可能的描述符泄漏，套接字文件描述符可以封装在一个类中，如下所示：

```cpp
class mytcpsocket
{
public:
    explicit mytcpsocket(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }

    ~mytcpsocket()
    {
        close(m_fd);
    }

    auto descriptor() const
    { return m_fd; }

private:

    int m_fd{};
};
```

在前面的示例中，我们使用先前示例中的逻辑打开了一个 IPv4 TCP 套接字，确保检测到任何错误并正确报告。不同之处在于我们将文件描述符存储为成员变量，并且当`mytcpsocket{}`失去作用域时，我们会自动确保文件描述符被正确释放回操作系统。每当需要文件描述符时，可以使用`descriptor()`访问器。

# bind()和 connect() API

创建套接字文件描述符后，套接字必须绑定或连接，具体取决于套接字是创建连接（服务器）还是连接到现有绑定套接字（客户端）。通过 TCP 或 UDP 进行通信时，绑定套接字会为套接字分配一个端口。端口`0`-`1024`保留用于特定服务，并且通常由操作系统管理（需要特殊权限进行绑定）。其余端口是用户定义的，并且通常可以在没有特权的情况下绑定。确定要使用的端口取决于实现。某些端口预先为特定应用程序确定，或者应用程序可以向操作系统请求一个可用的端口，还可以将这个新分配的端口通知给潜在的客户端应用程序，这增加了通信的复杂性。

`bind()` API 采用以下形式：

```cpp
int bind(int socket, const struct sockaddr *address, socklen_t address_len);
```

`socket`整数参数是先前由`socket()` API 提供的套接字文件描述符。`address`参数告诉操作系统要绑定到哪个端口，并且要接受来自哪个 IP 地址的传入连接，通常是`INADDR_ANY`，告诉操作系统可以接受来自任何 IP 地址的传入连接。最后，`address_len`参数告诉 API 地址结构的总大小是多少。

地址结构需要总大小（以字节为单位），因为根据您使用的套接字类型，支持不同的结构。例如，IPv6 套接字的 IP 地址比 IPv4 套接字大。在本章中，我们将讨论使用`sockaddr_in{}`结构的 IPv4，该结构定义以下字段：

+   `sin_family`：这与套接字域相同，在 IPv4 的情况下是`AF_INET`。

+   `sin_port`：这定义了要绑定到的端口，必须使用`htons()`转换为网络字节顺序。

+   `sin_address`：这定义了要接受传入连接的 IP 地址，也必须使用`htonl()`转换为网络字节顺序。通常，这被设置为`htonl(INADDR_ANY)`，表示可以接受来自任何 IP 地址的连接。

由于地址结构的长度是可变的，`bind()`API 接受一个指向不透明结构类型的指针，并使用长度字段来确保提供了正确的信息。应该注意，C++核心指南不鼓励这种类型的 API，因为没有类型安全的实现方式。事实上，为了使用这个 API，需要使用`reinterpret_cast()`将`sockaddr_in{}`转换为不透明的`sockaddr{}`结构。尽管 C++核心指南不支持使用`reinterpret_cast()`，但没有其他选择，因此如果需要套接字，必须违反这个规则。

服务器使用`bind()`为套接字专用端口，客户端使用`connect()`连接到已绑定的端口。`connect()`API 的形式如下：

```cpp
int connect(int socket, const struct sockaddr *address, socklen_t address_len);
```

应该注意，`connect()`的参数与`bind()`相同。与`bind()`一样，必须提供`socket()`调用返回的文件描述符，并且在 IPv4 的情况下，必须提供指向`sockaddr_in{}`结构的指针以及`sockaddr_in{}`结构的大小。在填写`sockaddr_in{}`结构时，可以使用以下内容：

+   `sin_family`：与套接字域相同，在 IPv4 的情况下为`AF_INET`。

+   `sin_port`：定义要连接的端口，必须使用`htons()`转换为网络字节顺序。

+   `sin_address`：定义要连接的 IP 地址，也必须使用`htonl()`转换为网络字节顺序。对于环回连接，这将设置为`htonl(INADDR_LOOPBACK)`。

最后，`bind()`和`connect()`在成功时返回`0`，失败时返回`-1`，并在发生错误时设置`errno`。

# `listen()`和`accept()`API

对于 TCP 服务器，还存在两个额外的 API，提供了服务器监听和接受传入 TCP 连接的方法——`listen()`和`accept()`。

`listen()`API 的形式如下：

```cpp
int listen(int socket, int backlog); 
```

套接字参数是`socket()`API 返回的文件描述符，backlog 参数限制可以建立的未决连接的总数。在本章的示例中，我们将使用`0`的 backlog，这告诉 API 使用实现特定的值作为 backlog。

如果`listen()`成功，返回`0`，否则返回`-1`，并设置`errno`为适当的错误代码。

一旦应用程序设置好监听传入连接的准备，`accept()`API 可以用来接受连接。`accept()`API 的形式如下：

```cpp
int accept(int socket, struct sockaddr *address, socklen_t *address_len);
```

与其他 API 一样，`socket`参数是`socket()`API 返回的文件描述符和地址，`address_len`参数返回连接的信息。如果不需要连接信息，也可以为地址和`address_len`提供`nullptr`。成功完成`accept()`API 后，将返回客户端连接的套接字文件描述符，可用于与客户端发送和接收数据。

如果 accept 执行失败，返回的不是有效的套接字文件描述符，而是返回`-1`，并且适当地设置了`errno`。

应该注意，`listen()`和`accept()`仅适用于 TCP 连接。对于 TCP 连接，服务器创建两个或多个套接字描述符；第一个用于绑定到端口并监听连接，而第二个是客户端的套接字文件描述符，用于发送和接收数据。另一方面，UDP 是一种无连接的协议，因此用于绑定到端口的套接字也用于与客户端发送和接收数据。

# `send()`、`recv()`、`sendto()`和`recvfrom()`API

在打开套接字后向服务器或客户端发送信息，POSIX 提供了`send()`和`sendto()`API。`send()`API 的形式如下：

```cpp
ssize_t send(int socket, const void *buffer, size_t length, int flags);
```

第一个参数是要发送数据的服务器或客户端的套接字文件描述符。应该注意的是，套接字必须连接到特定的客户端或服务器才能工作（例如，与服务器进行通信，或者使用 TCP 打开的客户端）。`buffer`参数指向要发送的缓冲区，`length`定义了要发送的缓冲区的长度，`flags`提供了各种不同的设置，用于指定发送缓冲区的方式，在大多数情况下只需设置为`0`。还应该注意，当`flags`设置为`0`时，`write()`函数和`send()`函数通常没有区别，两者都可以使用。

如果服务器尝试使用 UDP 与客户端通信，服务器将不知道如何将信息发送给客户端，因为服务器绑定到特定端口，而不是特定客户端。同样，如果使用 UDP 的客户端不连接到特定服务器，它将不知道如何将信息发送给服务器。因此，POSIX 提供了`sendto()`，它添加了`sockaddr{}`结构，用于定义要发送缓冲区的对象和方式。`sendto()`的形式如下：

```cpp
ssize_t sendto(int socket, const void *buffer, size_t length, int flags, const struct sockaddr *dest_addr, socklen_t dest_len);
```

`send()`和`sendto()`之间唯一的区别是`sendto()`还提供了目标`address`和`len`参数，这为用户提供了一种定义缓冲区发送对象的方式。

要从客户端或服务器接收数据，POSIX 提供了`recv()`API，其形式如下：

```cpp
ssize_t recv(int socket, void *buffer, size_t length, int flags);
```

`recv()`API 与`send()`API 具有相同的参数，不同之处在于当接收到数据时，将写入缓冲区（这就是为什么它没有标记为`const`），并且长度字段描述了缓冲区的总大小，而不是接收到的字节数。

同样，POSIX 提供了`recvfrom()`API，类似于`sendto()`API，其形式如下：

```cpp
ssize_t recvfrom(int socket, void *restrict buffer, size_t length, int flags, struct sockaddr *restrict address, socklen_t *restrict address_len);
```

`send()`和`sendto()`函数都返回发送的总字节数，而`recv()`和`recvfrom()`函数返回接收到的总字节数。所有这些函数在发生错误时都返回`-1`并将`errno`设置为适当的值。

# 学习 UDP 回显服务器的示例

在本例中，我们将通过一个简单的 UDP 回显服务器示例来引导您。回显服务器（与我们之前的章节相同）会将任何输入回显到其输出。在这个 UDP 示例中，服务器将从客户端接收到的数据回显回客户端。为了保持示例简单，将回显字符缓冲区。如何正确处理结构化数据包将在接下来的示例中介绍。

# 服务器

首先，我们必须定义从客户端发送到服务器和返回的最大缓冲区大小，并且我们还必须定义要使用的端口：

```cpp
#define PORT 22000
#define MAX_SIZE 0x10
```

应该注意，只要端口号在`1024`以上，任何端口号都可以，以避免需要特权。在本例中，服务器需要以下包括：

```cpp
#include <array>
#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <string.h>

#include <sys/socket.h>
#include <netinet/in.h>
```

服务器将使用一个类来定义，以利用 RAII，提供一个在不再需要时关闭服务器打开的套接字的清理方法。我们还定义了三个私有成员变量。第一个变量将存储服务器在整个示例中将使用的套接字文件描述符。第二个变量存储服务器的地址信息，将提供给`bind()`函数，而第三个参数存储客户端的地址信息，将被`recvfrom()`和`sendto()`函数使用。

```cpp
class myserver
{
    int m_fd{};
    struct sockaddr_in m_addr{};
    struct sockaddr_in m_client{};

public:
```

服务器的构造函数将打开套接字并将提供的端口绑定到套接字，如下所示：

```cpp
    explicit myserver(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_DGRAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (this->bind() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }
```

套接字使用`AF_INET`打开，这告诉套接字 API 需要 IPv4。此外，提供了`SOCK_DGRAM`，这告诉套接字 API 需要 UDP 而不是 TCP。对`::socket()`的调用结果保存在`m_fd`变量中，该变量存储服务器的套接字文件描述符。利用 C++17，如果结果文件描述符为`-1`，则发生错误，我们会抛出错误，稍后会恢复。

接下来，我们填写一个`sockaddr_in{}`结构：

+   `sin_family`被设置为`AF_INET`以匹配套接字，告诉套接字 API 我们希望使用 IPv4。

+   `sin_port`被设置为端口号，`htons`用于将主机字节顺序转换为短网络字节顺序。

+   `sin_addr` 被设置为 `INADDR_ANY`，这告诉套接字 API 服务器将接受来自任何客户端的数据。由于 UDP 是一种无连接的协议，这意味着我们可以从任何客户端接收数据。

最后，调用一个名为`bind()`的成员函数，并检查结果是否有错误。如果发生错误，就会抛出异常。

绑定函数实际上只是`::bind()`套接字 API 的包装器，如下所示：

```cpp
    int bind()
    {
        return ::bind(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }
```

在前面的代码片段中，我们使用在服务器类的构造函数中打开的套接字文件描述符调用`bind`，并在调用此函数之前提供了在构造函数中初始化的端口和地址给`bind` API，这告诉套接字绑定到端口`22000`和任何 IP 地址。

一旦套接字被绑定，服务器就准备好从客户端接收数据。由于我们将套接字绑定到任何 IP 地址，任何客户端都可以向我们发送信息。我们可以使用`recv()` POSIX API 来实现这一点，但这种方法的问题在于一旦我们接收到数据，我们就不知道是谁发送给我们信息。如果我们不需要向该客户端发送任何信息，或者我们将客户端信息嵌入接收到的数据中，这是可以接受的，但在简单的回显服务器的情况下，我们需要知道要将数据回显给谁。为了解决这个问题，我们使用`recvfrom()`而不是`recv()`，如下所示：

```cpp
   ssize_t recv(std::array<char, MAX_SIZE> &buf)
   {
        socklen_t client_len = sizeof(m_client);

        return ::recvfrom(
            m_fd,
            buf.data(),
            buf.size(),
            0,
            (struct sockaddr *) &m_client,
            &client_len
        );
    }
```

第一个参数是在构造过程中创建的套接字文件描述符，而第二个和第三个参数是缓冲区及其最大大小。请注意，我们的`recv()`成员函数使用`std::array`而不是指针和大小，因为使用指针和大小参数不符合 C++核心规范，因为这样做会提供报告数组实际大小的错误机会。最后两个参数是指向`sockaddr_in{}`结构和其大小的指针。

值得注意的是，在我们的示例中，我们向`recvfrom()`提供了一个`sockaddr_in{}`结构，因为我们知道将要连接的客户端将使用 IPv4 地址。如果不是这种情况，`recvfrom()`函数将失败，因为我们提供了一个太小的结构，无法提供例如 IPv6 地址（如果使用）的结构。为了解决这个问题，可以使用`sockaddr_storage{}`而不是`sockaddr_in{}`。`sockaddr_storage{}`结构足够大，可以存储传入的地址类型。要确定收到的地址类型，可以使用所有结构中都需要的`sin_family`字段。

最后，我们返回对`recvfrom()`的调用结果，这可能是接收到的字节数，或者在发生错误时为`-1`。

要将缓冲区发送给连接到 UDP 服务器的客户端，我们使用`sendto()` API，如下所示：

```cpp
    ssize_t send(std::array<char, MAX_SIZE> &buf, ssize_t len)
    {
        if (len >= buf.size()) {
            throw std::out_of_range("len >= buf.size()");
        }

        return ::sendto(
            m_fd,
            buf.data(),
            buf.size(),
            0,
            (struct sockaddr *) &m_client,
            sizeof(m_client)
        );
    }
```

与其他 API 一样，第一个参数是在构造函数中打开的套接字文件描述符。然后提供缓冲区。在这种情况下，“recvfrom（）”和“sendto（）”之间的区别在于提供要发送的字节数，而不是缓冲区的总大小。这不会违反 C++核心指导，因为缓冲区的总大小仍然附加到缓冲区本身，而要发送的字节数是用于确定我们计划寻址数组的位置的第二个值。但是，我们需要确保长度字段不超出范围。这可以使用“Expects（）”调用来完成，如下所示：

```cpp
Expects(len < buf.size())
```

在这个例子中，我们明确检查了是否超出范围的错误，并在发生这种情况时抛出了更详细的错误。任何一种方法都可以。

与“recvfrom（）”调用一样，我们向“sendto（）”API 提供了指向`sockaddr_in{}`结构的指针，告诉套接字要向哪个客户端发送数据。在这种情况下，由于 API 不修改地址结构（因此结构的大小不会改变），因此不需要指向长度字段的指针。

下一步是将所有这些组合在一起，创建回显服务器本身，如下所示：

```cpp
    void echo()
    {
        while(true)
        {
            std::array<char, MAX_SIZE> buf{};

            if (auto len = recv(buf); len != 0) {
                send(buf, len);
            }
            else {
                break;
            }
        }
    }
```

回显服务器设计用于从客户端接收数据缓冲区，将其发送回同一客户端，并重复。首先，我们创建一个无限循环，能够从任何客户端回显数据，直到我们被告知客户端已断开连接。下一步是定义一个缓冲区，该缓冲区将用于向客户端发送和接收数据。然后调用“recv（）”成员函数，并向其提供我们希望接收函数用来填充来自客户端的数据的缓冲区，并检查来自客户端返回的字节数是否大于`0`。如果来自客户端返回的字节数大于`0`，我们使用`send`成员函数将缓冲区发送（或回显）回客户端。如果字节数为`0`，我们假设客户端已断开连接，因此停止无限循环，从而完成回显过程。

客户端信息结构（即`m_client`）提供给“recvfrom（）”和“sendto（）”POSIX API。这是故意的。我们唯一假设的是所有连接的客户端都将使用 IPv4。当从客户端接收到数据时，“recvfrom（）”函数将为我们填充`m_client`结构，告诉我们发送给我们信息的客户端是谁。然后我们将相同的结构提供回“sendto（）”函数，告诉 API 要将数据回显给谁。

如前所述，当服务器类被销毁时，我们关闭套接字，如下所示：

```cpp
    ~myserver()
    {
        close(m_fd);
    }
```

最后，我们通过在“protected_main（）”函数中实例化服务器来完成服务器，并开始回显：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myserver server{PORT};
    server.echo();

    return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

如所示，`main`函数受到可能异常的保护，在“protected_main（）”函数中，我们实例化服务器并调用其“echo（）”成员函数，这将启动用于回显客户端数据的无限循环。

# 客户端逻辑

在这个例子中，客户端需要以下包含：

```cpp
#include <array>
#include <string>
#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <string.h>

#include <sys/socket.h>
#include <netinet/in.h>
```

与服务器一样，客户端是使用类创建的，以利用 RAII：

```cpp
class myclient
{
    int m_fd{};
    struct sockaddr_in m_addr{};

public:
```

除了类定义之外，还定义了两个私有成员变量。第一个，像服务器一样，是客户端将使用的套接字文件描述符。第二个定义了客户端希望与之通信的服务器的地址信息。

客户端的构造函数与服务器的类似，有一些细微的差异：

```cpp
    explicit myclient(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_DGRAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (connect() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }
```

像服务器一样，客户端使用`AF_INET`创建 IPv4 的套接字文件描述符，并且使用`SOCK_DGRAM`将协议类型设置为 UDP。如果`socket()`API 返回错误，则会抛出异常。设置的`sockaddr_in{}`结构与服务器不同。服务器的`sockaddr_in{}`结构定义了服务器将如何绑定套接字，而客户端的`sockaddr_in{}`结构定义了客户端将连接到哪个服务器。在这个例子中，我们将地址设置为`INADDR_LOOPBACK`，因为服务器将在同一台计算机上运行。最后，调用`connect()`成员函数，连接到服务器，如果发生错误，则抛出异常。

连接到服务器，使用以下`connect()`成员函数：

```cpp
    int connect()
    {
        return ::connect(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }
```

应该注意，使用 UDP 连接到服务器是可选的，因为 UDP 是一种无连接的协议。在这种情况下，`connect`函数告诉操作系统您计划与哪个服务器通信，以便在客户端使用`send()`和`recv()`，而不是`sendto()`和`recvfrom()`。像服务器的`bind()`成员函数一样，`connect()`函数利用构造函数填充的`sockaddr_in{}`结构。

要发送数据到服务器进行回显，使用以下`send()`成员变量：

```cpp
    ssize_t send(const std::string &buf)
    {
        return ::send(
            m_fd,
            buf.data(),
            buf.size(),
            0
        );
    }
```

由于我们计划向服务器发送一个字符串，所以我们将`send()`成员函数传递一个字符串引用。然后`send()` POSIX API 被赋予在构造函数中创建的套接字文件描述符，要发送到服务器进行回显的缓冲区以及要发送的缓冲区的总长度。由于我们不使用`flags`字段，`send()`成员函数也可以使用`write()`函数编写如下：

```cpp
    ssize_t send(const std::string &buf)
    {
        return ::write(
            m_fd,
            buf.data(),
            buf.size()
        );
    }
```

要在服务器回显数据后从服务器接收数据，我们使用以下`recv()`成员函数：

```cpp
    ssize_t recv(std::array<char, MAX_SIZE> &buf)
    {
        return ::recv(
            m_fd,
            buf.data(),
            buf.size() - 1,
            0
        );
    }
```

有许多方法可以实现`recv()`成员函数。由于我们知道要发送到服务器的字符串的总大小，并且我们知道服务器将向我们回显相同大小的字符串，我们可以始终创建一个与第一个字符串大小相同的第二个字符串（或者如果您信任回显实际上正在发生，可以简单地重用原始字符串）。在这个例子中，我们创建一个具有特定最大大小的接收缓冲区，以演示更有可能的情况。因此，在这个例子中，我们可以发送任意大小的字符串，但是服务器有自己的内部最大缓冲区大小可以接受。然后服务器将数据回显到客户端。客户端本身有自己的最大接收缓冲区大小，这最终限制了可能被回显的总字节数。由于客户端正在回显字符串，我们必须为尾随的`'\0'`保留一个字节，以便终止由客户端接收到的填满整个接收缓冲区的任何字符串。

要向服务器发送和接收数据，我们创建一个`echo`函数，如下所示：

```cpp
    void echo()
    {
        while(true) {
            std::string sendbuf{};
            std::array<char, MAX_SIZE> recvbuf{};

            std::cin >> sendbuf;
            if (sendbuf == "exit") {
                send({});
                break;
            }

            send(sendbuf);
            recv(recvbuf);

            std::cout << recvbuf.data() << '\n';
        }
    }
```

`echo`函数，就像服务器一样，首先创建一个无限循环，以便可以向服务器发送多个字符串进行回显。在无限循环内，创建了两个缓冲区。第一个是将接收用户输入的字符串。第二个定义了要使用的接收缓冲区。一旦定义了缓冲区，我们使用`std::cin`从用户那里获取要发送到服务器的字符串（最终将被回显）。

如果字符串是单词`exit`，我们向服务器发送 0 字节并退出无限循环。由于 UDP 是一种无连接的协议，服务器无法知道客户端是否已断开连接，因为没有这样的构造存在。因此，如果不向服务器发送停止的信号（在这种情况下我们发送 0 字节），服务器将保持在无限循环中，因为它无法知道何时停止。在这个例子中，这带来了一个有趣的问题，因为如果客户端崩溃或被杀死（例如，使用*Ctrl* + *C*），服务器将永远不会收到 0 字节的信号，因此仍然保持在无限循环中。有许多方法可以解决这个问题（即发送保持活动的信号），但一旦你开始尝试解决这个问题，你很快就会得到一个与 TCP 如此相似的协议，你可能会选择使用 TCP。

最后，用户输入的缓冲区使用`send()`成员函数发送到服务器，服务器回显字符串，然后客户端使用`recv()`成员函数接收字符串。一旦接收到字符串，数据将使用`std::cout`输出到`stdout`。

与服务器一样，当客户端类被销毁时，套接字文件描述符将被关闭，关闭套接字：

```cpp
    ~myclient()
    {
        close(m_fd);
    }
};
```

最后，客户端是使用与服务器和我们先前的示例相同的`protected_main()`函数创建的：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myclient client{PORT};
    client.echo();

    return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

在上面的代码中，客户端是在`protected_main()`函数中实例化的，并调用了`echo`函数，该函数接受用户输入，将输入发送到服务器，并将任何回显的数据输出到`stdout`。

# 编译和测试

要编译此代码，我们利用了我们一直在使用的相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt)。

有了这个代码，我们可以使用以下命令编译这个代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行服务器，请运行以下命令：

```cpp
> ./example1_server
```

要执行客户端，请打开一个新的终端并运行以下命令：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter10/build
> ./example1_client
Hello ↵
Hello
World
World ↵
exit ↵
```

如前面的片段所示，当客户端执行并输入时，输入将回显到终端。完成后，输入单词`exit`，客户端退出。服务器也将在客户端完成时退出。为了演示 UDP 的连接问题，而不是输入`exit`，在客户端上按*Ctrl *+ *C*，客户端将退出，但服务器将继续执行，等待来自客户端的更多输入，因为它不知道客户端已完成。为了解决这个问题，我们的下一个示例将创建相同的回声服务器，但使用 TCP。

# 学习 TCP 回声服务器的示例

在这个例子中，我们将引导读者创建一个回声服务器，但是使用 TCP 而不是 UDP。就像之前的例子一样，回声服务器会将任何输入回显到其输出。与 UDP 示例不同，TCP 是一种基于连接的协议，因此在这个例子中建立连接和发送/接收数据的一些具体细节是不同的。

# 服务器

首先，我们必须定义从客户端发送到服务器和返回的最大缓冲区大小，并且我们还必须定义要使用的端口：

```cpp
#define PORT 22000
#define MAX_SIZE 0x10
```

对于服务器，我们将需要以下包含：

```cpp
#include <array>
#include <iostream>

#include <unistd.h>
#include <string.h>

#include <sys/socket.h>
#include <netinet/in.h>
```

与之前的例子一样，我们将使用一个类来创建服务器，以便利用 RAII：

```cpp
class myserver
{
    int m_fd{};
    int m_client{};
    struct sockaddr_in m_addr{};

public:
```

与 UDP 一样，将使用三个成员变量。第一个成员变量`m_fd`存储与服务器关联的套接字文件描述符。与 UDP 不同，此描述符将不用于与客户端发送/接收数据。相反，`m_client`表示将用于与客户端发送/接收数据的第二个套接字文件描述符。与 UDP 一样，`sockaddr_in{}`结构`m_addr`将填充服务器地址类型，该类型将被绑定。

服务器的构造函数与 UDP 示例类似：

```cpp
    explicit myserver(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (this->bind() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }
```

与 UDP 示例类似，创建了服务器的套接字文件描述符，但是使用的不是`SOCK_DGRAM`，而是使用`SOCK_STREAM`。`sockaddr_in{}`结构与 UDP 示例相同，使用了 IPv4（即`AF_INET`），端口和任何 IP 地址用于表示将接受来自任何 IP 地址的连接。

与 UDP 示例类似，`sockaddr_in{}`结构然后使用以下成员函数进行绑定：

```cpp
    int bind()
    {
        return ::bind(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }
```

前面的`bind()`函数与 UDP 示例中使用的`bind()`函数相同。

与 UDP 不同，创建了第二个特定于客户端的套接字描述符，并为该套接字类型设置了 IP 地址、端口和地址类型，这意味着与客户端通信不需要`sendto()`或`recvfrom()`，因为我们已经有了一个特定的套接字文件描述符，其中已经绑定了这些额外的信息。因此，可以使用`send()`和`recv()`而不是`sendto()`和`recvfrom()`。

要从客户端接收数据，将使用以下成员函数：

```cpp
    ssize_t recv(std::array<char, MAX_SIZE> &buf)
    {
        return ::recv(
            m_client,
            buf.data(),
            buf.size(),
            0
        );
    }
```

UDP 示例和这个示例之间唯一的区别是使用`recv()`而不是`recvfrom()`，这省略了额外的`sockaddr_in{}`结构。如果你还记得之前的 UDP 示例，`m_fd`是与`recvfrom()`一起使用的，而不是`m_client`与`recv()`一起使用的。不同之处在于 UDP 示例中的`m_client`是一个`sockaddr_in{}`结构，用于定义从哪里接收数据。而在 TCP 中，`m_client`实际上是一个套接字描述符，从描述符绑定接收数据，这就是为什么不需要额外的`sockaddr_in{}`结构。

`send()`成员函数也是如此：

```cpp
    ssize_t send(std::array<char, MAX_SIZE> &buf, ssize_t len)
    {
        if (len >= buf.size()) {
            throw std::out_of_range("len >= buf.size()");
        }

        return ::send(
            m_client,
            buf.data(),
            len,
            0
        );
    }
```

与 UDP 示例不同，前面的`send()`函数可能使用`send()` POSIX API 而不是`sendto()`，因为关于如何向客户端发送数据的地址信息已经绑定到描述符上，因此可以省略额外的`sockaddr_in{}`信息。`send()`函数的其余部分与 UDP 示例相同。

`echo`函数与其 UDP 对应函数有很大不同：

```cpp
    void echo()
    {
        if (::listen(m_fd, 0) == -1) {
            throw std::runtime_error(strerror(errno));
        }

        if (m_client = ::accept(m_fd, nullptr, nullptr); m_client == -1) {
            throw std::runtime_error(strerror(errno));
        }

        while(true)
        {
            std::array<char, MAX_SIZE> buf{};

            if (auto len = recv(buf); len != 0) {
                send(buf, len);
            }
            else {
                break;
            }
        }

        close(m_client);
    }
```

由于 TCP 需要连接，服务器`echo`函数的第一步是告诉 POSIX API 您希望开始监听传入连接。在我们的示例中，通过将 backlog 设置为`0`来告诉 API 使用默认连接 backlog，这是特定于实现的。下一步是使用`accept()` POSIX API 等待来自客户端的传入连接。默认情况下，此函数是一个阻塞函数。`accept()`函数返回一个带有地址信息绑定到描述符的套接字文件描述符，因此在`accept()` POSIX API 的地址字段中传递`nullptr`，因为在我们的示例中不需要这些信息（但是如果需要过滤某些传入客户端，可能需要这些信息）。

下一步是等待客户端接收数据，然后使用`send()`成员函数将数据回传给客户端。这个逻辑与 UDP 示例相同。值得注意的是，如果我们从客户端接收到`0`字节，我们将停止处理来自客户端的数据，类似于 UDP。不同之处在于，如将会展示的，客户端端不需要显式地向服务器发送 0 字节以发生这种情况。

`echo`函数中的最后一步是在客户端完成后关闭客户端套接字文件描述符：

```cpp
    ~myserver()
    {
        close(m_fd);
    }
};
```

与其他示例一样，当服务器类被销毁时，关闭服务器的套接字文件描述符。最后，在`protected_main()`函数中实例化服务器，如下所示：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myserver server{PORT};
    server.echo();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

与 UDP 示例类似，实例化了服务器，并执行了`echo()`函数。

# 客户端逻辑

客户端逻辑与 UDP 客户端逻辑类似，有一些细微的例外。需要以下包含：

```cpp
#include <array>
#include <string>
#include <iostream>

#include <unistd.h>
#include <string.h>

#include <sys/socket.h>
#include <netinet/in.h>
```

与 UDP 示例一样，创建了一个客户端类来利用 RAII，并定义了`m_fd`和`m_addr`私有成员变量，用于存储客户端的套接字文件描述符和客户端希望连接到的服务器的地址信息：

```cpp
class myclient
{
    int m_fd{};
    struct sockaddr_in m_addr{};

public:
```

与 UDP 示例不同，但与 TCP 服务器逻辑相同，构造函数创建了一个用于 IPv4 和 TCP 的套接字，使用了`AF_INET`和`SOCK_STREAM`：

```cpp
    explicit myclient(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (connect() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }
```

构造函数的其余部分与 UDP 示例相同，`connect()`、`send()`和`recv()`函数也是如此：

```cpp
     int connect()
    {
        return ::connect(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }

    ssize_t send(const std::string &buf)
    {
        return ::send(
            m_fd,
            buf.data(),
            buf.size(),
            0
        );
    }

    ssize_t recv(std::array<char, MAX_SIZE> &buf)
    {
        return ::recv(
            m_fd,
            buf.data(),
            buf.size() - 1,
            0
        );
    }
```

如前面的代码片段所示，客户端的功能几乎与 UDP 客户端完全相同。UDP 客户端和 TCP 客户端之间的区别，除了使用`SOCK_STREAM`之外，还在于`echo`函数的实现：

```cpp
    void echo()
    {
        while(true) {
            std::string sendbuf{};
            std::array<char, MAX_SIZE> recvbuf{};

            std::cin >> sendbuf;

            send(sendbuf);
            recv(recvbuf);

            std::cout << recvbuf.data() << '\n';
        }
    }
```

与 UDP 示例不同，TCP 客户端不需要检查`exit`字符串。这是因为如果客户端断开连接（例如，使用*Ctrl*+*C*杀死客户端），服务器端会接收到 0 字节，告诉服务器逻辑客户端已断开连接。这是可能的，因为 TCP 是一种基于连接的协议，因此操作系统正在维护一个开放的连接，包括服务器和客户端之间的保持活动信号，以便 API 的用户不必显式地执行此操作。因此，在大多数情况下，这是期望的套接字类型，因为它可以防止许多与连接状态相关的常见问题：

```cpp
    ~myclient()
    {
        close(m_fd);
    }
};
```

如前面的代码所示，与所有其他示例一样，当客户端被销毁时，套接字文件描述符将被关闭，如下所示：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myclient client{PORT};
    client.echo();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

最后，客户端在`protected_main()`函数中实例化，并调用`echo`函数。

# 编译和测试

要编译此代码，我们利用了与本章其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt)。

有了这些代码，我们可以使用以下命令编译此代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行服务器，请运行以下命令：

```cpp
> ./example2_server
```

要执行客户端，请打开一个新的终端并运行以下命令：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter10/build
> ./example2_client
Hello ↵
Hello
World
World ↵
<ctrl+c>
```

如前面的代码片段所示，当客户端被执行并输入时，输入将被回显到终端。完成后，输入*Ctrl*+*C*，客户端退出。如您所见，服务器将在客户端完成时退出。上面的示例演示了 TCP 的易用性及其优于 UDP 的优势。下一个示例将演示如何使用 TCP 进行更有用的操作。

# 探索 TCP 记录器示例

为了演示更有用的功能，以下示例实现了我们在整本书中一直在开发的相同记录器，但作为远程记录设施。

# 服务器

与本章前面的示例一样，此示例也需要相同的宏和包含文件。要启动服务器，我们必须定义日志文件：

```cpp
std::fstream g_log{"server_log.txt", std::ios::out | std::ios::app};
```

由于记录器将在同一台计算机上执行，为了保持示例简单，我们将命名服务器正在记录的文件为`server_log.txt`。

服务器与前面示例中的 TCP 服务器相同，唯一的区别是只需要一个`recv()`成员函数（即不需要`send()`函数，因为服务器只会接收日志数据）：

```cpp
class myserver
{
    int m_fd{};
    int m_client{};
    struct sockaddr_in m_addr{};

public:
    explicit myserver(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (this->bind() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }

    int bind()
    {
        return ::bind(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }

    ssize_t recv(std::array<char, MAX_SIZE> &buf)
    {
        return ::recv(
            m_client, buf.data(), buf.size(), 0
        );
    }
```

前一个 TCP 示例和此示例之间的区别在于使用`log()`函数而不是`echo`函数。这两个函数都类似，它们监听传入的连接，然后无限循环，直到服务器接收到数据：

```cpp
    void log()
    {
        if (::listen(m_fd, 0) == -1) {
            throw std::runtime_error(strerror(errno));
        }

        if (m_client = ::accept(m_fd, nullptr, nullptr); m_client == -1) {
            throw std::runtime_error(strerror(errno));
        }

        while(true)
        {
            std::array<char, MAX_SIZE> buf{};

            if (auto len = recv(buf); len != 0) {
                g_log.write(buf.data(), len);
                std::clog.write(buf.data(), len);
            }
            else {
                break;
            }
        }

        close(m_client);
    }
```

`log`函数的不同之处在于，当客户端接收到数据时，不会将数据回显到服务器，而是将数据输出到`stdout`并写入`server_log.txt`日志文件。

如此所示，服务器逻辑的其余部分与前面的示例相同：

```cpp
    ~myserver()
    {
        close(m_fd);
    }
};

int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myserver server{PORT};
    server.log();

    return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

当服务器对象被销毁时，套接字文件描述符被关闭，在`protected_main()`函数中实例化服务器，然后执行`log()`函数。

# 客户端逻辑

本示例的客户端逻辑是前几章中的调试示例（我们一直在构建）和之前的 TCP 示例的组合。

我们首先定义调试级别并启用宏，与之前的示例一样：

```cpp
#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif
```

客户端类与之前的 TCP 示例中的客户端类相同：

```cpp
class myclient
{
    int m_fd{};
    struct sockaddr_in m_addr{};

public:
    explicit myclient(uint16_t port)
    {
        if (m_fd = ::socket(AF_INET, SOCK_STREAM, 0); m_fd == -1) {
            throw std::runtime_error(strerror(errno));
        }

        m_addr.sin_family = AF_INET;
        m_addr.sin_port = htons(port);
        m_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (connect() == -1) {
            throw std::runtime_error(strerror(errno));
        }
    }

    int connect()
    {
        return ::connect(
            m_fd,
            reinterpret_cast<struct sockaddr *>(&m_addr),
            sizeof(m_addr)
        );
    }

    ssize_t send(const std::string &buf)
    {
        return ::send(
            m_fd,
            buf.data(),
            buf.size(),
            0
        );
    }

    ~myclient()
    {
        close(m_fd);
    }
};
```

本示例中的客户端与上一个示例中的客户端唯一的区别在于，在本示例中不需要`recv()`函数（因为不会从服务器接收数据），也不需要`echo()`函数（或类似的东西），因为客户端将直接用于根据需要向服务器发送数据。

与之前的调试示例一样，需要为客户端创建一个日志文件，在本示例中，我们还将全局实例化客户端，如下所示：

```cpp
myclient g_client{PORT};
std::fstream g_log{"client_log.txt", std::ios::out | std::ios::app};
```

如所示，客户端日志文件将被命名为`client_log.txt`，以防止与服务器日志文件发生冲突，因为两者将在同一台计算机上运行，以简化示例。

`log`函数与第八章中定义的`log`函数相同，*学习编程文件输入/输出*，唯一的区别是除了记录到`stderr`和客户端日志文件外，调试字符串还将记录到服务器上：

```cpp
template <std::size_t LEVEL>
constexpr void log(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::stringstream buf;

        auto g_buf = std::clog.rdbuf();
        std::clog.rdbuf(buf.rdbuf());

        func();

        std::clog.rdbuf(g_buf);

        std::clog << "\033[1;32mDEBUG\033[0m: ";
        std::clog << buf.str();

        g_log << "\033[1;32mDEBUG\033[0m: ";
        g_log << buf.str();

        g_client.send("\033[1;32mDEBUG\033[0m: ");
        g_client.send(buf.str());
    };
}
```

如前面的代码所示，`log`函数封装了对`std::clog`的任何输出，并将结果字符串重定向到`stderr`，日志文件，并且为了本示例的目的，发送字符串到服务器的客户端对象上，以便在服务器端记录。

示例的其余部分与之前的示例相同：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    log<0>([]{
        std::clog << "Hello World\n";
    });

    std::clog << "Hello World\n";

    return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

`protected_main()`函数将`Hello World\n`输出到`stderr`，它被重定向到包括`stderr`，日志文件，并最终发送到服务器。另外调用`std::clog`用于显示只有封装在`log()`函数中的`std:clog`调用才会被重定向。

# 编译和测试

要编译此代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt)。

有了这段代码，我们可以使用以下命令编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行服务器，请运行以下命令：

```cpp
> ./example3_server
```

要执行客户端，请打开一个新的终端并运行以下命令：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter10/build
> ./example3_client
Debug: Hello World
Hello World

> cat client_log.txt
Debug: Hello World

> cat server_log.txt
Debug: Hello World

```

如前面的片段所示，当客户端执行时，客户端和服务器端都将在`stderr`输出`DEBUG: Hello World`。此外，客户端还将`Hello World`输出到`stderr`，因为第二次对`std::clog`的调用没有被重定向。最后，两个日志文件都包含重定向的`DEBUG: Hello World`。

到目前为止，在所有示例中，忽略的一件事是如果多个客户端尝试连接到服务器会发生什么。在本章的示例中，只支持一个客户端。要支持额外的客户端，需要使用线程，这将在第十二章中介绍，*学习编程 POSIC 和 C++线程*，在那里我们将扩展此示例以创建一个能够记录多个应用程序的调试输出的日志服务器。本章的最后两个示例将演示如何使用 TCP 处理非字符串数据包。

# 尝试处理数据包的示例

在本示例中，我们将讨论如何处理从客户端到服务器的以下数据包：

```cpp
struct packet
{
    uint64_t len;
    char buf[MAX_SIZE];

    uint64_t data1;
    uint64_t data2;
};
```

数据包由一些固定宽度的整数数据和一个字符串组成（网络中的字段必须始终是固定宽度，因为您可能无法控制应用程序运行的计算机类型，非固定宽度类型，如`int`和`long`，可能会根据计算机而变化）。

这种类型的数据包在许多程序中很常见，但正如将要演示的那样，这种类型的数据包在安全解析方面存在挑战。

服务器与之前的 TCP 示例相同，减去了`recv_packet()`函数（`recv()`函数处理数据包而不是`std::arrays`）：

```cpp
class myserver
{
...

    void recv_packet()
    {
        if (::listen(m_fd, 0) == -1) {
            throw std::runtime_error(strerror(errno));
        }

        if (m_client = ::accept(m_fd, nullptr, nullptr); m_client == -1) {
            throw std::runtime_error(strerror(errno));
        }

        packet p{};

        if (auto len = recv(p); len != 0) {
            auto msg = std::string(p.buf, p.len);

            std::cout << "data1: " << p.data1 << '\n';
            std::cout << "data2: " << p.data2 << '\n';
            std::cout << "msg: \"" << msg << "\"\n";
            std::cout << "len: " << len << '\n';
        }

        close(m_client);
    }

...
};
```

在`recv_packet()`函数中，我们等待从客户端接收数据。一旦从客户端接收到数据包，我们就解析接收到的数据包。与数据包相关的整数数据被读取并输出到`stdout`而没有问题。然而，字符串数据更加棘手。由于我们不知道接收到的字符串数据的总大小，我们必须考虑整个缓冲区来安全地处理字符串，并在某种程度上保持类型安全。当然，在我们的示例中，为了减小数据包的总大小，我们可以先将整数数据放在数据包中，然后创建一个可变长度的数据包，但这既不安全，也难以在更复杂的情况下控制或实现。大多数解决这个问题的尝试（需要发送和接收比实际需要的更多数据）都会导致长度可变的操作，因此是不安全的。

服务器的其余部分与之前的示例相同：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myserver server{PORT};
    server.recv_packet();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

如前面的代码所示，服务器在`protected_main()`函数中实例化，并调用`recv_packet()`函数。

# 客户端逻辑

客户端的大部分部分也与之前的示例相同：

```cpp
class myclient
{
...

    void send_packet()
    {
        auto msg = std::string("Hello World");

        packet p = {
            42,
            43,
            msg.size(),
            {}
        };

        memcpy(p.buf, msg.data(), msg.size());

        send(p);
    }

...
};
```

`send_packet()`函数是与之前的示例唯一不同的部分（减去`send()`函数发送的是数据包而不是`std::array()`）。在`send_packet()`函数中，我们创建一个不包含`"Hello World"`字符串的数据包。值得注意的是，为了创建这个数据包，我们仍然需要一些处理，包括内存复制。一旦数据包创建完成，我们就将其发送到服务器进行处理。

客户端的其余部分与之前的示例相同：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myclient client{PORT};
    client.send_packet();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

客户端在`proceted_main()`函数中实例化，并执行`send_packet()`函数。

# 编译和测试

要编译此代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt)。

有了这段代码，我们可以使用以下命令编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行服务器，运行以下命令：

```cpp
> ./example4_server
```

要执行客户端，打开一个新的终端并运行以下命令：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter10/build
> ./example4_client
```

在服务器端，以下内容输出到`stdout`：

```cpp
data1: 42
data2: 43
msg: "Hello World"
len: 280
```

如前面的片段所示，客户端发送数据包，服务器接收。服务器接收到的数据包总大小为 280 字节，尽管字符串的总大小要小得多。在下一个示例中，我们将演示如何通过数据包编组安全地减小数据包的总大小，尽管这会增加一些额外的处理（尽管根据您的用例可能是可以忽略的）。

# 处理 JSON 处理的示例

在最后一个示例中，我们将演示如何使用 JSON 对数据包进行编组，以安全地减小网络数据包的大小，尽管这会增加一些额外的处理。为支持此示例，将使用以下 C++ JSON 库：[`github.com/nlohmann/json`](https://github.com/nlohmann/json)。

要将此 JSON 库纳入我们的示例中，需要将以下内容添加到我们的`CMakeLists.txt`中，该文件将下载这个仅包含头文件的库并将其安装到我们的构建文件夹中以供使用：

```cpp
list(APPEND JSON_CMAKE_ARGS
    -DBUILD_TESTING=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
)

ExternalProject_Add(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_SHALLOW 1
    CMAKE_ARGS ${JSON_CMAKE_ARGS}
    PREFIX ${CMAKE_BINARY_DIR}/external/json/prefix
    TMP_DIR ${CMAKE_BINARY_DIR}/external/json/tmp
    STAMP_DIR ${CMAKE_BINARY_DIR}/external/json/stamp
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/json/download
    SOURCE_DIR ${CMAKE_BINARY_DIR}/external/json/src
    BINARY_DIR ${CMAKE_BINARY_DIR}/external/json/build
    UPDATE_DISCONNECTED 1
)
```

# 服务器

服务器包括和宏是一样的，唯一的区别是必须添加 JSON，如下所示：

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;
```

在本示例中，服务器与之前的示例相同，唯一的区别是`recv_packet()`函数：

```cpp
class myserver
{
...

    void recv_packet()
    {
        std::array<char, MAX_SIZE> buf{};

        if (::listen(m_fd, 0) == -1) {
            throw std::runtime_error(strerror(errno));
        }

        if (m_client = ::accept(m_fd, nullptr, nullptr); m_client == -1) {
            throw std::runtime_error(strerror(errno));
        }

        if (auto len = recv(buf); len != 0) {
            auto j = json::parse(buf.data(), buf.data() + len);

            std::cout << "data1: " << j["data1"] << '\n';
            std::cout << "data2: " << j["data2"] << '\n';
            std::cout << "msg: " << j["msg"] << '\n';
            std::cout << "len: " << len << '\n';
        }

        close(m_client);
    }

...
};
```

在`recv_packet()`函数中，我们需要分配一个具有一定最大大小的缓冲区；这个缓冲区不需要完全接收，而是作为我们的 JSON 缓冲区的占位符，其大小可以达到我们的最大值。解析 JSON 数据很简单。整数数据和字符串数据都被安全地解析为它们的整数和`std::string`类型，都遵循 C++核心指南。代码易于阅读和理解，未来可以更改数据包而无需更改任何其他逻辑。

服务器的其余部分是相同的：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myserver server{PORT};
    server.recv_packet();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

服务器在`protected_main()`函数中实例化，然后调用`recv_packet()`函数。

# 客户端逻辑

与服务器一样，客户端也必须包括 JSON 头：

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;
```

与服务器一样，客户端与之前的示例相同，只是没有`send_packet()`函数：

```cpp
class myclient
{
...

    void send_packet()
    {
        json j;

        j["data1"] = 42;
        j["data2"] = 43;
        j["msg"] = "Hello World";

        send(j.dump());
    }

...
};
```

`send_packet()`函数同样简单。构造一个 JSON 数据包并发送到服务器。不同之处在于，在发送之前将数据包编组成 JSON 字符串（使用`dump()`函数）。这将把所有数据转换为一个字符串，其中包含特殊语法来定义每个字段的开始和结束，以防止不安全的解析，以一种经过良好建立和测试的方式。此外，如将很快展示的那样，发送的字节数总量大大减少。

客户端的其余部分是相同的：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    myclient client{PORT};
    client.send_packet();
}

int
main(int argc, char** argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

客户端在`protected_main()`函数中实例化，并调用`send_packet()`函数。

# 编译和测试

要编译这些代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter10/CMakeLists.txt)。

有了这些代码，我们可以使用以下命令编译这些代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter10/
> mkdir build
> cd build

> cmake ..
> make
```

要执行服务器，请运行以下命令：

```cpp
> ./example5_server
```

要执行客户端，请打开一个新的终端并运行以下命令：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter10/build
> ./example5_client
```

在服务器端，将以下内容输出到`stdout`：

```cpp
data1: 42
data2: 43
msg: "Hello World"
len: 43
```

如前面的片段所示，客户端发送数据包，服务器接收数据包。服务器接收的数据包总大小为 43 字节，与之前的示例相比，效率提高了 6.5 倍。除了提供更小的数据包外，创建和解析数据包的逻辑相似，未来更改也更容易阅读和修改。此外，使用 JSON Schema 等内容，甚至可以在处理之前验证数据包，这是本书范围之外的主题。

# 总结

在本章中，我们学习了如何使用 C++17 编程 POSIX 套接字。具体来说，我们学习了与 POSIX 套接字相关的常见 API，并学习了如何使用它们。我们用五个不同的示例结束了本章。第一个示例创建了一个 UDP 回显服务器，而第二个示例创建了一个类似的回显服务器，但使用的是 TCP 而不是 UDP，概述了不同方法之间的区别。第三个示例通过向我们的调试器添加服务器组件来扩展了我们的调试示例。第四和第五个示例演示了如何处理简单的网络数据包，以及使用编组来简化该过程的好处。

在下一章中，我们将讨论可用于获取挂钟时间、测量经过的时间和执行基准测试的 C 和 C++时间接口。

# 问题

1.  UDP 和 TCP 之间的主要区别是什么？

1.  UDP 使用什么协议类型？

1.  TCP 使用什么协议类型？

1.  `AF_INET`代表什么地址类型？

1.  `bind()`和`connect()`之间有什么区别？

1.  `sendto()`和`send()`之间有什么区别？

1.  UDP 服务器如何检测 UDP 客户端何时断开或崩溃？

1.  使用数据包编组的好处是什么？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
