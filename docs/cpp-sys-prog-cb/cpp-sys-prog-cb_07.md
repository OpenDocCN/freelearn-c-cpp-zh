# 网络编程

在第六章中，*管道，先进先出（FIFO），消息队列和共享内存*，我们学习了不同的 IPC 技术，允许在同一台机器上运行的进程相互通信。在本章中（补充了第六章中的内容），你将学习两个在两台不同计算机上运行的进程如何实现相同的结果。这里介绍的主题是当今互联网运行的基础。你将亲自学习连接导向和无连接导向通信之间的区别，定义端点的特征，最后学习两个使用 TCP/IP 和 UDP/IP 的方法。

本章将涵盖以下主题：

+   学习连接导向通信的基础知识

+   学习无连接导向通信的基础知识

+   学习通信端点是什么

+   学习使用 TCP/IP 与另一台机器上的进程进行通信

+   学习使用 UDP/IP 与另一台机器上的进程进行通信

+   处理字节序

# 技术要求

为了让你立即开始使用这些程序，我们设置了一个 Docker 镜像，其中包含了本书中需要的所有工具和库。它基于 Ubuntu 19.04。

为了设置它，按照以下步骤进行：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  使用`docker pull kasperondocker/system_programming_cookbook:latest`从 Docker Hub 拉取镜像。

1.  镜像现在应该可用。输入`docker images`查看镜像。

1.  现在你应该至少有`kasperondocker/system_programming_cookbook`。

1.  使用`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`运行 Docker 镜像与交互式 shell。

1.  正在运行的容器上的 shell 现在可用。使用`root@39a5a8934370/# cd /BOOK/`获取按章节列出的所有程序。

`--cap-add sys_ptrace`参数是为了允许 Docker 容器中的**GNU 项目调试器**（**GDB**）设置断点，Docker 默认情况下不允许。要在同一个容器上启动第二个 shell，运行`docker exec -it container-name bash`命令。你可以从`docker ps`命令中获取容器名称。

免责声明：C++20 标准已经在二月底的布拉格会议上得到了 WG21 的批准（也就是在技术上已经最终确定）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 的代码。GCC 将最新功能的开发保留在分支中（你必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励你自己尝试。所以，克隆并探索 GCC 的合同和模块分支，玩得开心。

# 学习连接导向通信的基础知识

如果你坐在桌前浏览互联网，很可能你正在使用连接导向类型的通信。当你通过 HTTP 或 HTTPS 请求页面时，在实际通信发生之前，你的机器和你试图联系的服务器之间建立了连接。互联网通信的*事实上*标准是**传输控制协议**（**TCP**）。在本章中，你将学习它是什么，为什么它很重要，你还将学习（在命令行上）什么是连接。

# 如何做到这一点...

在本节中，我们将使用命令行来了解当我们与远程机器建立连接时发生了什么。具体来说，我们将学习 TCP/IP 连接的内部方面。让我们按照以下步骤进行：

1.  使用 Docker 镜像运行后，打开一个 shell，输入以下命令，然后按*Enter*键：

```cpp
tcpdump -x tcp port 80
```

1.  打开另一个 shell，输入以下命令，然后按*Enter*：

```cpp
telnet amazon.com 80
```

1.  在第一个 shell 中，您将看到类似以下的输出：

![](img/5a48ff38-c4f1-4ee1-934f-f71da9db0be1.png)

所有这些可能看起来很神秘，但实际上很简单。下一节将详细解释它是如何工作的。

# 它是如何工作的...

基于连接的通信是基于两个实体之间建立连接的假设。在本节中，我们将探讨连接到底是什么。

第一步使用`tcpdump`（`man tcpdump`），这是一个在网络上转储所有流量的命令行工具。在我们的情况下，它将把端口`80`上的所有 TCP 流量写入标准输出，并以十六进制表示形式显示数据。按下*Enter*后，`tcpdump`将切换到监听模式。

第二步使用`telnet`与在`amazon.com`端口`80`上运行的远程服务建立连接。按下*Enter*后，几秒钟后，连接将建立。

在第三步中，我们看到了本地机器通过`telnet`（或`man telnet`，以其全名命名）服务与`amazon.com`（转换为 IP）之间的连接输出。要记住的第一件事是，TCP 中的连接是一个称为**三次握手**的三步过程。客户端发送*SYN*，服务器回复*SYN+ACK*，客户端回复*ACK*。以下图表示了 TCP 头规范：

![](img/a90e0c44-8eec-4e64-b1de-f2cb80cfd1ff.png)

在*SYN* | *SYN+ACK* | *ACK*阶段，客户端和服务器交换了什么数据以成功建立连接？让我们一步一步地来看：

1.  客户端向服务器(`amazon.com`)发送*SYN*：

![](img/40eb9534-c86f-4741-9bba-1c40ba1910ca.png)

让我们从`0xe8f4`和`0x050`开始（以太网头部在此之前，这超出了本章的范围）。从前面的 TCP 头部中可以看到，前两个字节表示源端口（`0xe8f4` = `59636`），接下来的两个字节表示目标端口（`0x0050` = `80`）。在接下来的四个字节中，客户端设置了一个称为序列号的随机数：`0x9bd0 | 0xb114`。在这种情况下，确认号没有设置。为了将此数据包标记为*SYN*，客户端必须将*SYN*位设置为`1`，确实下两个字节的值为`0xa002`，在二进制中为`1010 0000 0000 0010`。我们可以看到倒数第二位设置为 1（将其与前面的屏幕截图中的 TCP 头部进行比较）。

1.  服务器向客户端发送*SYN+ACK*：

![](img/e30cbd29-ea67-47fc-92dd-8dad6943277d.png)

服务器收到来自客户端的*SYN*后，必须以*SYN+ACK*进行响应。忽略前 16 个字节，即以太网头部，我们可以看到以下内容：2 个字节表示源端口（`0x0050` = `80`），第二个 2 个字节表示目标端口（`0xe8f4` = `59636`）。然后我们开始看到一些有趣的东西：服务器在序列号中放入一个随机数，这种情况下是`0x1afe = | 0x5e1e`，在确认号中，是从客户端接收的序列号+1 = `0x9bd0 | 0xb11**5**`。正如我们所学的，服务器必须将标志设置为*SYN+ACK*，根据 TCP 头规范，通过将两个字节设置为`0x7012` = `0111 0000 000**1** 00**1**0`来正确实现。高亮部分分别是*ACK*和*SYN*。然后 TCP 数据包被发送回客户端。

1.  客户端向服务器(`amazon.com`)发送*ACK*：

![](img/626f2c3e-2a7b-4b54-9cb3-a082f5324929.png)

三次握手算法的最后一步是接收客户端发送的 ACK 数据包。消息由两个字节组成，表示源端口（`0xe8f4` = `59636`）和目标端口（`0x050` = `80`）；这次的序列号包含了服务器最初从客户端接收到的值，`0x9bd0 | 0xb115`；确认号包含了服务器接收到的随机值加 1：`0x1afe = | 0x5e1**f**`。最后，通过设置值`0x5010` = `0101 0000 000**1** 0000`来发送*ACK*（被突出显示的部分是*ACK*；与之前的 TCP 头部图片进行比较）。

# 还有更多...

到目前为止，您学到的协议在 RFC 793 中有描述（[`tools.ietf.org/html/rfc793`](https://tools.ietf.org/html/rfc793)）。如果互联网正常工作，那是因为所有网络供应商、设备驱动程序实现和许多程序都完美地实现了这个 RFC（以及其他相关标准）。TCP RFC 定义的远不止我们在这个配方中学到的内容，它严格关注于连接性。它定义了流量控制（通过窗口的概念）和可靠性（通过序列号和其中的*ACK*的概念）。

# 另请参阅

+   *学习使用 TCP/IP 与另一台机器上的进程进行通信*的配方显示了两台机器上的两个进程如何进行通信。连接部分隐藏在系统调用中，我们将看到。

+   第三章，*处理进程和线程*，了解有关进程和线程的内容。

# 学习无连接导向通信的基础知识

在*学习面向连接的通信的基础知识*配方中，我们学到了面向连接的通信与流量控制是可靠的。要使两个进程进行通信，我们必须首先建立连接。这显然会在性能方面产生成本，我们并不总是能够支付——例如，当您观看在线电影时，可用的带宽可能不足以支持 TCP 所带来的所有功能。

在这种情况下，底层通信机制很可能是无连接的。*事实上*的标准无连接通信协议是**用户数据协议**（**UDP**），它与 TCP 处于相同的逻辑级别。在这个配方中，我们将学习命令行上的 UDP 是什么样子。

# 如何做...

在本节中，我们将使用`tcpdump`和`netcast`（`nc`）来分析 UDP 上的无连接链路：

1.  Docker 镜像正在运行时，打开一个 shell，输入以下命令，然后按*Enter*：

```cpp
tcpdump -i lo udp port 45998 -X
```

1.  让我们打开另一个 shell，输入以下命令，然后按*Enter*：

```cpp
echo -n "welcome" | nc -w 1 -u localhost 45998
```

1.  在第一个 shell 中，您将看到类似以下的输出：

![](img/e671d1fb-07cb-4c1a-a09e-5187ebd9b0e9.png)

这似乎也很神秘，但实际上很简单。下一节将详细解释这些步骤。

# 它是如何工作的...

在 UDP 连接中，没有连接的概念。在这种情况下，数据包被发送到接收器。没有流量控制，连接也不可靠。正如您从下图中看到的那样，UDP 头确实非常简单：

![](img/a29b794e-a166-43b2-96de-6adc69398346.png)

*步骤 1*使用`tcpdump`监听端口`45998`，在`loopback`接口上使用`UDP`协议（`-i lo`），通过打印每个数据包的十六进制和 ASCII 数据来查看数据。

*步骤 2*使用`netcast`命令`nc`（`man nc`）发送一个包含字符串`welcome`的 UDP 数据包（`-u`）到本地主机。

*步骤 3* 显示了 UDP 协议的详细信息。我们可以看到源端口（由发送方随机选择）为 `0xdb255` = `56101`，目标端口正确设置为 `0xb3ae` = `459998`。接下来，我们将长度设置为 `0x000f` = `15`，校验和设置为 `0xfe22` = `65058`。长度为 `15` 字节，因为 `7` 字节是接收到的数据长度，`8` 字节是 UDP 标头的长度（源端口 + 目标端口 + 长度 + 校验和）。

没有重传，没有控制流，没有连接。无连接的链接实际上只是发送方发送给接收方的消息，知道可能不会收到它。

# 还有更多...

我们已经讨论了连接，并在 UDP 标头中看到了源端口和目标端口的概念。发送方和接收方的地址存储在其他地方，即在 **IP**（**Internet** **Protocol** 的缩写）层中，逻辑上位于 UDP 层的下方。IP 层具有发送方和接收方地址（IP 地址）的信息，用于将 UDP 数据包从客户端路由到服务器，反之亦然。

UDP 在 RFC 768 中有详细定义，网址为 [`www.ietf.org/rfc/rfc768.txt`](https://www.ietf.org/rfc/rfc768.txt)。

# 另请参阅

+   第一章，*开始系统编程*，回顾命令管道

+   *无连接导向通信基础* 配方，与 TCP 协议进行比较

# 了解通信端点是什么

当两个实体相互通信时，它们本质上是交换信息。为了使这种情况发生，每个实体都必须清楚地知道将信息发送到何处。从程序员的角度来看，参与通信的每个实体都必须有一个清晰的端点。本配方将教你端点是什么，并将在命令行上显示如何识别它们。

# 如何做...

在本节中，我们将使用 `netstat` 命令行实用程序来检查和了解端点是什么：

1.  使用运行 Docker 镜像的 shell，输入以下命令，然后按 *Enter*：

```cpp
b07d3ef41346:/# telnet amazon.com 443
```

1.  打开第二个 shell 并输入以下命令：

```cpp
b07d3ef41346:/# netstat -ntp
```

下一节将解释这两个步骤。

# 工作原理...

在 *步骤 1* 中，我们使用 `telnet` 实用程序连接到本地机器，与 `amazon.com` 远程主机的端口 `443`（HTTP）连接。此命令的输出如下：

![](img/62b15a2f-680e-4b7e-af13-a937e1bc9e0a.png)

它正在等待命令，我们不会发送命令，因为我们真正关心的是连接。

在 *步骤 2* 中，我们想要了解我们在本地机器（`localhost`）和远程主机（`amazon.com` 端口 `443`）之间建立的连接的详细信息。为此，我们执行了 *步骤 2* 中的命令。输出如下：

![](img/027525f4-3f59-4b27-b3c6-6ef58b76f388.png)

我们可以从此命令行的输出中检索到什么信息？嗯，我们可以检索到一些非常有用的信息。让我们看看我们可以从前面的屏幕截图中学到什么，从左到右阅读代码：

+   `tcp` 代表连接的类型。这是一个面向连接的连接，这意味着本地和远程主机经历了我们在 *学习面向连接的通信基础* 配方中看到的三次握手。

+   `Recv-Q` 是一个队列，其中包含本地主机上当前进程要处理的数据。

+   `Send-Q` 是一个队列，其中包含本地主机上当前进程要发送到远程进程的数据。

+   `Local Address` 是 IP 地址和端口号的组合，实际上代表了我们通信的第一个端点，即本地端点。从编程的角度来看，这样的端点通常被称为 `Socket`，它是一个代表 `IP` 和 `PORT` 的整数。在这种情况下，端点是 `172.17.0.2:40850`。

+   `Foreign Address`，就像`Local Address`一样，是`IP`和`PORT`的组合，代表远程端点，在这种情况下是`176.32.98.166:443`。请注意，`443`是一个众所周知的端口，代表`https`服务。

+   `State`代表两个端点之间连接的状态，在这种情况下是`ESTABLISHED`。

+   `PID/Program Name`，或者在我们的例子中，`65`/`telnet`，代表使用两个端点与远程主机通信的本地进程。

当程序员谈论`socket`时，他们是在谈论通信的每个端点的`IP`和`PORT`。正如我们所见，Linux 使得分析通信的两个端点和它们附加的进程变得容易。

一个重要的方面要强调的是，`PORT`代表一个服务。在我们的例子中，本地进程 telnet 使用 IP `176.32.98.166`连接到端口`80`的远程主机，我们知道那里运行着一个 HTTP 守护程序。但是我们如何知道特定服务的端口号？有一个由**IANA**（即**Internet Assigned Numbers Authority**的缩写）维护的众所周知的端口列表（[`www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml`](https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml)）。例如，预期 HTTPS 服务在`PORT 443`上运行，`sftp`（即**Secure File Transfer Protocol**的缩写）在`PORT 22`上运行，依此类推。

# 还有更多...

`port`信息是一个 16 位无符号整数值（即`unsigned int`），由 IANA（[`www.iana.org/`](https://www.iana.org/)）维护，并分为以下范围：

+   0-1023：众所周知的端口。众所周知的端口，例如 HTTP、SFTP 和 HTTPS。

+   1024-49151：注册端口。组织可以要求为其目的注册的端口。

+   49152-65535：动态、私有或临时端口。可自由使用。

# 另请参阅

+   *学习基本的无连接导向通信*的方法来学习无连接通信的工作原理

+   *学习基本的连接导向通信*的方法来学习带有连接的通信工作原理

+   *学习使用 TCP/IP 与另一台机器上的进程通信*的方法来学习如何开发连接导向的程序

+   *学习使用 UDP/IP 与另一台机器上的进程通信*的方法来学习如何开发无连接导向的程序

# 学习使用 TCP/IP 与另一台机器上的进程通信

这个方法将向您展示如何使用连接导向的机制连接两个程序。这个方法将使用 TCP/IP，这是互联网上的*事实*标准。到目前为止，我们已经了解到 TCP/IP 是一种可靠的通信形式，它的连接分为三个阶段。现在是时候编写一个程序来学习如何使两个程序相互通信了。尽管使用的语言将是 C++，但通信部分将使用 Linux 系统调用编写，因为它不受 C++标准库支持。

# 如何做...

我们将开发两个程序，一个客户端和一个服务器。服务器将启动并在准备接受传入连接的特定端口上进行`listen`。客户端将启动并连接到由 IP 和端口号标识的服务器：

1.  使用运行的 Docker 镜像，打开一个 shell 并创建一个新文件`clientTCP.cpp`。让我们添加一些稍后需要的头文件和常量：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <iostream>

constexpr unsigned int SERVER_PORT = 50544;
constexpr unsigned int MAX_BUFFER = 128;
```

1.  让我们现在开始编写`main`方法。我们首先初始化`socket`并获取与服务器相关的信息：

```cpp
int main(int argc, char *argv[])
{
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
    {
        std::cerr << "socket error" << std::endl;
        return 1;
    }
    struct hostent* server = gethostbyname(argv[1]);
    if (server == nullptr) 
    {
        std::cerr << "gethostbyname, no such host" << std::endl;
        return 2;
    }
```

1.  接下来，我们想要连接到服务器，但我们需要正确的信息，即`serv_addr`：

```cpp
    struct sockaddr_in serv_addr;
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
          (char *)&serv_addr.sin_addr.s_addr, 
          server->h_length);
    serv_addr.sin_port = htons(SERVER_PORT);
    if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof
        (serv_addr)) < 0)
    {
        std::cerr << "connect error" << std::endl;
        return 3;
    }
```

1.  服务器将回复连接`ack`，因此我们调用`read`方法：

```cpp
    std::string readBuffer (MAX_BUFFER, 0);
    if (read(sockfd, &readBuffer[0], MAX_BUFFER-1) < 0)
    {
        std::cerr << "read from socket failed" << std::endl;
        return 5;
    }
    std::cout << readBuffer << std::endl;
```

1.  现在我们可以通过调用`write`系统调用将数据发送到服务器：

```cpp
    std::string writeBuffer (MAX_BUFFER, 0);
    std::cout << "What message for the server? : ";
    getline(std::cin, writeBuffer);
    if (write(sockfd, writeBuffer.c_str(), strlen(write
        Buffer.c_str())) < 0) 
    {
        std::cerr << "write to socket" << std::endl;
        return 4;
    }
```

1.  最后，让我们进行清理部分，关闭 socket：

```cpp
    close(sockfd);
    return 0;
}
```

1.  现在让我们开发服务器程序。在第二个 shell 中，我们创建`serverTCP.cpp`文件：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <arpa/inet.h>

constexpr unsigned int SERVER_PORT = 50544;
constexpr unsigned int MAX_BUFFER = 128;
constexpr unsigned int MSG_REPLY_LENGTH = 18;
```

1.  在第二个 shell 中，首先，我们需要一个将标识我们连接的`socket`描述符：

```cpp
int main(int argc, char *argv[])
{
     int sockfd =  socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0)
     {
          std::cerr << "open socket error" << std::endl;
          return 1;
     }

     int optval = 1;
     setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const
       void *)&optval , sizeof(int));

```

1.  我们必须将`socket`绑定到本地机器上的一个端口和`serv_addr`：

```cpp
     struct sockaddr_in serv_addr, cli_addr;
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(SERVER_PORT);
     if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof
        (serv_addr)) < 0)
     {
          std::cerr << "bind error" << std::endl;
          return 2;
     }
```

1.  接下来，我们必须等待并接受任何传入的连接：

```cpp
     listen(sockfd, 5);
     socklen_t clilen = sizeof(cli_addr);
     int newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, 
         &clilen);
     if (newsockfd < 0)
     {
          std::cerr << "accept error" << std::endl;
          return 3;
     }
```

1.  一旦我们建立了连接，我们就会记录谁连接到标准输出（使用他们的 IP 和端口），并发送一个确认*ACK*：

```cpp
     std::cout << "server: got connection from = "
               << inet_ntoa(cli_addr.sin_addr)
               << " and port = " << ntohs(cli_addr.sin_port)
                  << std::endl;
     write(incomingSock, "You are connected!", MSG_REPLY_LENGTH);
```

1.  我们建立了连接（三次握手，记得吗？），所以现在我们可以读取来自客户端的任何数据：

```cpp
     std::string buffer (MAX_BUFFER, 0);
     if (read(incomingSock, &buffer[0], MAX_BUFFER-1) < 0)
     {
          std::cerr << "read from socket error" << std::endl;
          return 4;
     }
     std::cout << "Got the message:" << buffer << std::endl;
```

1.  最后，我们关闭两个套接字：

```cpp
     close(incomingSock);
     close(sockfd);
     return 0;
}
```

我们已经写了相当多的代码，现在是时候解释所有这些是如何工作的了。

# 它是如何工作的...

客户端和服务器都有一个非常常见的算法，我们必须描述它以便你理解和概括这个概念。客户端的算法如下：

```cpp
socket() -> connect() -> send() -> receive()
```

在这里，`connect()`和`receive()`都是阻塞调用（即，调用程序将等待它们的完成）。`connect`短语特别启动了我们在*学习面向连接的通信基础*中详细描述的三次握手。

服务器的算法如下：

```cpp
socket() -> bind() -> listen() -> accept() -> receive() -> send()
```

在这里，`accept`和`receive`都是阻塞调用。现在让我们详细分析客户端和服务器的代码。

客户端代码分析如下：

1.  第一步只包含了在前面客户端算法部分列出的四个 API 的必要包含文件。请注意，常量采用纯 C++风格，不是使用`#define`宏定义，而是使用`constexpr`。区别在于后者由编译器管理，而前者由预处理器管理。作为一个经验法则，你应该总是尽量依赖编译器。

1.  `socket()`系统调用创建了一个套接字描述符，我们将其命名为`sockfd`，它将用于与服务器发送和接收信息。这两个参数表示套接字将是一个 TCP（`SOCK_STREAM`）/IP（`PF_INET`）套接字类型。一旦我们有了一个有效的套接字描述符，并在调用`connect`方法之前，我们需要知道服务器的详细信息；为此，我们使用`gethostbyname()`方法，它会返回一个指向`struct hostent *`的指针，其中包含有关主机的信息，给定一个类似`localhost`的字符串。

1.  我们现在准备调用`connect()`方法，它将负责三次握手过程。通过查看它的原型（`man connect`），我们可以看到它除了套接字外，还需要一个`const struct sockaddr *address`结构，因此我们需要将相应的信息复制到其中，并将其传递给`connect()`；这就是为什么我们使用`utility`方法`bcopy()`（`bzero()`只是在使用之前重置`sockaddr`结构的辅助方法）。

1.  我们现在已经准备好发送和接收数据。一旦建立了连接，服务器将发送一个确认消息（`You are connected!`）。你是否注意到我们正在使用`read()`方法通过套接字从服务器接收信息？这就是在 Linux 环境中编程的美和简单之处。一个方法可以支持多个接口——事实上，我们能够使用相同的方法来读取文件、通过套接字接收数据，以及做许多其他事情。

1.  我们可以向服务器发送消息。使用的方法是，你可能已经猜到了，是`write()`。我们将`socket`传递给它，它标识了连接，我们希望服务器接收的消息，以及消息的长度，这样 Linux 就知道何时停止从缓冲区中读取。

1.  通常情况下，我们需要关闭、清理和释放任何使用的资源。在这种情况下，我们需要通过使用`close()`方法关闭套接字描述符。

服务器代码分析如下：

1.  我们使用了类似于客户端的代码，但包含了一些头文件和三个定义的常量，我们稍后会使用和解释。

1.  我们必须通过调用`socket()` API 来定义套接字描述符。请注意，客户端和服务器之间没有区别。我们只需要一个能够管理 TCP/IP 类型连接的套接字。

1.  我们必须将在上一步中创建的套接字描述符绑定到本地机器上的网络接口和端口。我们使用`bind()`方法来实现这一点，它将地址（作为第二个参数传递的`const struct sockaddr *address`）分配给作为第一个参数传递的套接字描述符。调用`setsockopt()`方法只是为了避免绑定错误，即`地址已在使用`。

1.  通过调用`listen()` API 开始监听任何传入的连接。`listen()`系统调用非常简单：它获取我们正在监听的`socket`描述符以及保持在挂起连接队列中的最大连接数，我们在这种情况下设置为`5`。然后我们在套接字描述符上调用`accept()`。`accept`方法是一个阻塞调用：这意味着它将阻塞，直到有一个新的传入连接可用，然后它将返回一个表示套接字描述符的整数。`cli_addr`结构被填充了连接的信息，我们用它来记录谁连接了（`IP`和`端口`）。

1.  这一步只是步骤 10 的逻辑延续。一旦服务器接受连接，我们就会在标准输出上记录谁连接了（以他们的`IP`和`端口`表示）。我们通过查询`accept`方法填充的`cli_addr`结构中的信息来实现这一点。

1.  在这一步中，我们通过`read()`系统调用从连接的客户端接收信息。我们传入输入，传入连接的套接字描述符，`buffer`（数据将被保存在其中），以及我们想要读取的数据的最大长度（`MAX_BUFFER-1`）。

1.  然后清理和释放任何可能使用和/或分配的资源。在这种情况下，我们必须关闭使用的两个套接字描述符（服务器的`sockfd`和传入连接的`incomingSock`）。

通过按照这个顺序构建和运行服务器和客户端，我们得到以下输出：

+   服务器构建和输出如下：

![](img/6cb2d008-c48a-4572-95b5-c20f08518f1a.png)

+   客户端构建和输出如下：

![](img/56ff6da3-b779-438d-95c5-6821223a16ac.png)

这证明了我们在这个教程中学到的东西。

# 还有更多...

我们如何改进服务器应用程序以管理多个并发的传入连接？我们实现的服务器算法是顺序的；在`listen()`之后，我们只是等待`accept()`，直到最后关闭连接。您应该按照以下步骤进行练习：

1.  无限循环运行`accept()`，以便服务器始终处于准备好为客户端提供服务的状态。

1.  为每个接受的连接启动一个新线程。您可以使用`std::thread`或`std::async`来实现这一点。

另一个重要的实践是注意客户端和服务器之间交换的数据。通常，它们同意使用彼此都知道的协议。它可能是一个 Web 服务器，在这种情况下将涉及客户端和服务器之间的 HTML、文件、资源等的交换。如果是监控和控制系统，可能是由特定标准定义的协议。

# 另请参阅

+   第三章，*处理进程和线程*，以便回顾一下进程和线程是如何工作的，以改进这里描述的服务器解决方案

+   *学习面向连接的通信基础*这个教程来学习 TCP 连接的工作原理

+   *学习通信端点是什么*这个教程来学习端点是什么以及它与套接字的关系

# 学习使用 UDP/IP 与另一台机器上的进程进行通信

当一个进程与另一个进程通信时，可靠性并不总是决定通信机制的主要标准。有时，我们需要的是快速通信，而不需要 TCP 协议实现的连接、流量控制和所有其他控制，以使其可靠。这适用于视频流，**互联网语音**（**VoIP**）通话等情况。在这个示例中，我们将学习如何编写 UDP 代码，使两个（或更多）进程相互通信。

# 如何做到的...

我们将开发两个程序，一个客户端和一个服务器。服务器将启动，将套接字绑定到本地地址，然后只接收来自客户端的数据：

1.  使用运行的 Docker 镜像，打开一个 shell，创建一个新文件`serverUDP.cpp`，并添加一些以后会用到的标头和常量：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <arpa/inet.h>

```

```cpp
constexpr unsigned int SERVER_PORT = 50544;
constexpr unsigned int MAX_BUFFER = 128;
```

1.  在`main`函数中，我们必须实例化`数据报`类型的套接字，并设置选项以在每次重新运行服务器时重用地址：

```cpp
int main(int argc, char *argv[])
{
     int sockfd =  socket(AF_INET, SOCK_DGRAM, 0);
     if (sockfd < 0) 
     {
          std::cerr << "open socket error" << std::endl;
          return 1;
     }
     int optval = 1;
     setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const void 
         *)&optval , sizeof(int));
```

1.  我们必须将创建的套接字与本地地址绑定：

```cpp
     struct sockaddr_in serv_addr, cli_addr;
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;  
     serv_addr.sin_addr.s_addr = INADDR_ANY;  
     serv_addr.sin_port = htons(SERVER_PORT);
     if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof
        (serv_addr)) < 0)
     {
          std::cerr << "bind error" << std::endl;
          return 2;
     }
```

1.  我们现在准备从客户端接收数据包，这次使用`recvfrom` API：

```cpp
     std::string buffer (MAX_BUFFER, 0);
     unsigned int len;
     if (recvfrom(sockfd, &buffer[0], 
                  MAX_BUFFER, 0, 
                  (struct sockaddr*)& cli_addr, &len) < 0)
     {
          std::cerr << "recvfrom failed" << std::endl;
          return 3;
     }
     std::cout << "Got the message:" << buffer << std::endl;
```

1.  我们想用`sendto` API 向客户端发送一个*ACK*消息：

```cpp
     std::string outBuffer ("Message received!");
     if (sendto(sockfd, outBuffer.c_str(), 
                outBuffer.length(), 0, 
                (struct sockaddr*)& cli_addr, len) < 0)
     {
          std::cerr << "sendto failed" << std::endl;
          return 4;
     }
```

1.  最后，我们可以关闭套接字：

```cpp
     close(sockfd);
     return 0; 
}
```

1.  现在让我们创建客户端程序。在另一个 shell 中，创建文件`clientUDP.cpp`：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <iostream>

constexpr unsigned int SERVER_PORT = 50544;
constexpr unsigned int MAX_BUFFER = 128;
```

1.  我们必须实例化`数据报`类型的套接字：

```cpp
int main(int argc, char *argv[])
{
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) 
    {
        std::cerr << "socket error" << std::endl;
        return 1;
    }
```

1.  我们需要获取主机信息，以便能够识别要发送数据包的服务器，我们通过调用`gethostbyname` API 来实现：

```cpp
    struct hostent* server = gethostbyname(argv[1]);
    if (server == NULL) 
    {
        std::cerr << "gethostbyname, no such host" << std::endl;
        return 2;
    }

```

1.  将主机信息复制到`sockaddr_in`结构中以识别服务器：

```cpp
    struct sockaddr_in serv_addr, cli_addr;
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
          (char *)&serv_addr.sin_addr.s_addr, 
          server->h_length);
    serv_addr.sin_port = htons(SERVER_PORT);
```

1.  我们可以使用套接字描述符、用户的消息和服务器地址向服务器发送消息：

```cpp
    std::string outBuffer (MAX_BUFFER, 0);
    std::cout << "What message for the server? : ";
    getline(std::cin, outBuffer);
    unsigned int len = sizeof(serv_addr);
    if (sendto(sockfd, outBuffer.c_str(), MAX_BUFFER, 0, 
               (struct sockaddr *) &serv_addr, len) < 0)
    {
        std::cerr << "sendto failed" << std::endl;
        return 3;
    }
```

1.  我们知道服务器会用*ACK*回复，所以让我们用`recvfrom`方法接收它：

```cpp
    std::string inBuffer (MAX_BUFFER, 0);
    unsigned int len_cli_add;
    if (recvfrom(sockfd, &inBuffer[0], MAX_BUFFER, 0, 
                 (struct sockaddr *) &cli_addr, &len_cli_add) < 0)
    {
        std::cerr << "recvfrom failed" << std::endl;
        return 4;
    }
    std::cout << inBuffer << std::endl;
```

1.  最后，像往常一样，我们要负责关闭和释放所有使用的结构：

```cpp
    close(sockfd);
    return 0;
}
```

让我们深入了解代码，看看所有这些是如何工作的。

# 它是如何工作的...

在*学习使用 TCP/IP 与另一台机器上的进程通信*的示例中，我们学习了客户端和服务器的 TCP 算法。UDP 算法更简单，正如你所看到的，连接部分是缺失的：

**UDP 客户端的算法：**

```cpp
socket() ->  sendto() -> recvfrom()
```

**UDP 服务器的算法：**

```cpp
socket() -> bind() ->  recvfrom() -> sendto()
```

现在看看它们现在简单多了——例如，服务器在这种情况下不会`listen`和`accept`传入的连接。

服务器端的代码分析如下：

1.  我们刚刚定义了一些标头和两个常量，表示服务器将公开服务的端口（`SERVER_PORT`）和数据的最大大小（`MAX_BUFFER`）。

1.  在这一步中，我们定义了套接字（`sockfd`），就像我们在 TCP 代码中所做的那样，但这次我们使用了`SOCK_DGRAM`（UDP）类型。为了避免`Address already in use`的绑定问题，我们设置了选项以允许套接字重用地址。

1.  接下来是`bind`调用。它接受`int socket`、`const struct sockaddr *address`和`socklen_t address_len`这些参数，基本上是套接字、要绑定套接字的地址和地址结构的长度。在`address`变量中，我们指定我们正在监听所有可用的本地网络接口（`INADDR_ANY`），并且我们将使用 Internet 协议版本 4（`AF_INET`）。

1.  我们现在可以通过使用`recvfrom`方法开始接收数据。该方法以套接字描述符（`sockfd`）、用于存储数据的缓冲区（`buffer`）、我们可以存储的数据的最大大小、一个标志（在本例中为`0`）来设置接收消息的特定属性、数据报发送者的地址（`cli_addr`）和地址的长度（`len`）作为输入。最后两个参数将被填充返回，这样我们就知道是谁发送了数据报。

1.  现在我们可以向客户端发送一个*ACK*。我们使用`sendto`方法。由于 UDP 是一种无连接协议，我们没有连接的客户端，所以我们需要以某种方式传递这些信息。我们通过将`cli_addr`和长度(`len`)传递给`sendto`方法来实现这一点，这些信息是由`recvfrom`方法返回的。除此之外，我们还需要传递套接字描述符(`sockfd`)、要发送的缓冲区(`outBuffer`)、缓冲区的长度(`outBuffer.length()`)和标志(`0`)。

1.  然后，我们只需要在程序结束时进行清理。我们必须使用`close()`方法关闭套接字描述符。

客户端代码分析如下：

1.  在这一步中，我们找到了与`serverUDP.cpp`源文件中的`SERVER_PORT`和`MAX_BUFFER`相同的头文件。

1.  我们必须通过调用`socket`方法来定义数据报类型的套接字，再次将`AF_INET`和`SOCK_DGRAM`作为输入。

1.  由于我们需要知道将数据报发送给谁，客户端应用程序在命令行上输入服务器的地址(例如`localhost`)，我们将其作为输入传递给`gethostbyname`，它返回主机地址(`server`)。

1.  我们使用`server`变量填充`serv_addr`结构，用于标识我们要发送数据报的服务器的地址(`serv_addr.sin_addr.s_addr`)、端口(`serv_addr.sin_port`)和协议的族(`AF_INET`)。

1.  然后，我们可以使用`sendto`方法通过传递`sockfd`、`outBuffer`、`MAX_BUFFER`、设置为`0`的标志、服务器的地址`serv_addr`及其长度(`len`)来将用户消息发送到服务器。同样，在这个阶段，客户端不知道消息的接收者是谁，因为它没有连接到任何人，这就是为什么必须正确填写`serv_addr`结构，以便它包含有效的地址。

1.  我们知道服务器会发送一个应用程序*ACK*，所以我们必须接收它。我们调用`recvfrom`方法，将套接字描述符(`sockfd`)作为输入，用于存储返回数据的缓冲区(`buffer`)，我们可以获取的数据的最大大小，以及设置为`0`的标志。`recvfrom`返回消息发送者的地址及其长度，我们分别将其存储在`cli_addr`和`len`中。

让我们先运行服务器，然后再运行客户端。

按照以下方式运行服务器：

![](img/bdbbe7da-c8df-4197-912f-246ee3751e02.png)

按照以下方式运行客户端：

![](img/9a159ad9-61df-452c-91f6-b98de7bbfb2a.png)

这展示了 UDP 的工作原理。

# 还有更多...

另一种使用 UDP 协议的方式是以多播或广播格式发送数据报，作为一种无连接通信类型。多播是一种通信技术，用于将相同的数据报发送到多个主机。代码不会改变；我们只需设置多播组的 IP，以便它知道要发送消息的位置。这是一种方便和高效的*一对多*通信方式，可以节省大量带宽。另一种选择是以广播模式发送数据报。我们必须使用子网掩码设置接收者的 IP，形式为`172.30.255.255`。消息将发送到同一子网中的所有主机。

欢迎您通过以下步骤改进服务器代码：

1.  设置一个无限循环，使用`recvfrom()`，以便您始终有一个准备好为客户端提供服务的服务器。

1.  为每个接受的连接启动一个新线程。您可以使用`std::thread`或`std::async`来实现这一点。

# 另请参阅

+   第三章，*处理进程和线程*，以了解如何处理进程和线程以改进此处描述的服务器解决方案

+   *学习基于无连接的通信的基础知识*，以了解 UDP 连接的工作原理

+   *学习通信端点是什么*，以了解端点是什么，以及它与套接字的关系

# 处理字节序

在系统级编写代码可能意味着处理不同处理器的架构。在这样做时，程序员在 C++20 之前必须自行处理的一件事是**字节序**。字节序指的是数字的二进制表示中字节的顺序。幸运的是，最新的 C++标准帮助我们在编译时输入端口信息。本文将教你如何*意识到*字节序，并编写可以在小端和大端架构上运行的代码。

# 如何做...

我们将开发一个程序，该程序将在编译时查询机器，以便我们可以有意识地决定如何处理以不同格式表示的数字：

1.  我们需要包含`<bit>`头文件；然后我们可以使用`std::endian`枚举：

```cpp
#include <iostream>
#include <bit>

int main()
{ 
    if (std::endian::native == std::endian::big)
        // prepare the program to read/write 
        // in big endian ordering.
        std::cout << "big" << std::endl;
    else if (std::endian::native == std::endian::little)
        // prepare the program to read/write 
        // in little endian ordering.
        std::cout << "little" << std::endl; 

 return 0;
}
```

让我们在下一节更仔细地看看这对我们有什么影响。

# 它是如何工作的...

大端和小端是两种主要的数据表示类型。小端排序格式意味着最不重要的字节（也称为 LSB）放在最高地址，而在大端机器上，最重要的字节（也称为 MSB）放在最低地址。对于十六进制值 0x1234 的表示，示例如下：

|  | **地址** | **地址+1（字节）** |
| --- | --- | --- |
| **大端** | `12` | `34` |
| **小端** | `34` | `12` |

步骤 1 中代码片段的主要目标是回答一个问题：我如何知道我正在处理什么样的机器架构？新的 C++20 枚举`std::endian`完美地帮助我们解决了这个问题。怎么做？首先是从*端口意识*方面。将`std::endian`作为 C++标准库的一部分，帮助程序员随时查询底层机器的端口架构。其次：对于共享资源，两个程序必须就格式达成一致（就像 TCP 协议那样，即以*网络顺序*发送信息），以便读者（或者如果在网络上传输数据，则是接收者）可以进行适当的转换。

另一个问题是：我应该怎么做？有两件事你应该做：一件与应用程序的观点有关，另一件与网络有关。在这两种情况下，如果你的应用程序与另一台具有不同字节序格式的机器交换数据（例如交换文件或共享文件系统等），或者将数据发送到具有不同架构的机器上，则必须确保你的数据能够被理解。为此，你可以使用`hton`、`ntoh`宏等；这可以确保数字从主机转换为网络（对于`hton`）和从网络转换为主机（对于`ntoh`）。我们必须提到，大多数互联网协议使用大端格式，这就是为什么如果你从大端机器调用`hton`，该函数将不执行任何转换的原因。

英特尔 x86 系列和 AMD64 系列处理器都使用小端格式，而 IBM z/Architecture、Freescale 和所有 Motorola 68000 遗产处理器都使用大端格式。还有一些处理器（如 PowerPC）可以切换字节序。

# 还有更多...

理论上，除了小端和大端之外，还存在其他数据表示格式。一个例子是 Honeywell 316 微型计算机使用的中端格式。

# 另请参阅

+   *学习使用 TCP/IP 与另一台机器上的进程通信*配方

+   *学习使用 UDP/IP 与另一台机器上的进程通信*配方
