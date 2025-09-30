# 网络和进程间通信

进程各自独立运行并在各自的地址空间中工作。然而，它们有时需要相互通信以传递信息。为了使进程能够协作，它们需要能够相互通信并同步它们的行为。以下是进程间发生的通信类型：

+   **同步通信**：这种通信不允许进程在通信完成前继续进行任何其他工作

+   **异步通信**：在这种通信中，进程可以继续执行其他任务，因此它支持多任务处理，并导致更高的效率

+   **远程过程调用**（**RPC**）：这是一个使用客户端服务技术进行通信的协议，其中客户端无法执行任何操作，也就是说，它被挂起，直到从服务器收到响应

这些通信可以是单向的或双向的。为了在进程之间启用任何形式的通信，以下常用的**进程间通信**（**IPC**）机制被使用：管道、FIFOs（命名管道）、套接字、消息队列和共享内存。管道和 FIFOs 允许单向通信，而套接字、消息队列和共享内存则允许双向通信。

在本章中，我们将学习如何制作以下食谱，以便我们可以在进程之间建立通信：

+   使用管道在进程间通信

+   使用 FIFO 在进程间通信

+   使用套接字编程在客户端和服务器之间通信

+   使用 UDP 套接字在进程间通信

+   使用消息队列从一个进程向另一个进程传递消息

+   使用共享内存在进程间通信

让我们从第一个食谱开始！

# 使用管道在进程间通信

在这个食谱中，我们将学习如何从其写入端将数据写入管道，然后如何从其读取端读取该数据。这可以通过两种方式发生：

+   一个进程，既从管道中写入又从中读取

+   一个进程写入，另一个进程从管道中读取

在我们开始介绍食谱之前，让我们快速回顾一下在成功的进程间通信中使用的函数、结构和术语。

# 创建和连接进程

用于进程间通信的最常用函数和术语是`pipe`、`mkfifo`、`write`、`read`、`perror`和`fork`。

# pipe()

管道用于连接两个进程。一个进程的输出可以作为另一个进程的输入发送。流是单向的，也就是说，一个进程可以写入管道，另一个进程可以从中读取。写入和读取是在主内存的一个区域进行的，这也可以称为虚拟文件。管道具有**先进先出**（**FIFO**）或队列结构，即先写入的将被先读取。

进程不应该在向管道写入内容之前尝试从管道读取，否则它将挂起，直到向管道写入内容。

这里是其语法：

```cpp
int pipe(int arr[2]);
```

这里，`arr[0]` 是管道读取端的文件描述符，而 `arr[1]` 是管道写入端的文件描述符。

函数在成功时返回 `0`，在出错时返回 `-1`。

# mkfifo()

此函数创建一个新的 FIFO 特殊文件。以下是其语法：

```cpp
int mkfifo(const char *filename, mode_t permission);
```

这里，`filename` 代表文件名及其完整路径，而 `permission` 代表新 FIFO 文件的权限位。默认权限是所有者、组和其他人的读写权限，即 (0666)。

函数在成功完成后返回 `0`；否则，返回 `-1`。

# write()

此函数用于将数据写入指定的文件或管道（其描述符在方法中提供）。以下是其语法：

```cpp
write(int fp, const void *buf, size_t n);
```

它将 *n* 个字节写入由文件指针 `fp` 指向的文件，来自缓冲区 `buf`。

# read()

此函数从指定的文件或管道（其描述符在方法中提供）读取。以下是其语法：

```cpp
read(int fp, void *buf, size_t n);
```

它尝试从由描述符 `fp` 指向的文件读取最多 *n* 个字节。读取的字节随后分配给缓冲区 `buf`。

# perror()

这会显示一个错误消息，指示在调用函数或系统调用时可能发生的错误。错误消息显示在 `stderr`，即标准错误输出流。这基本上是控制台。

这里是其语法：

```cpp
void perror ( const char * str );
```

显示的错误消息可以由代表 `str` 的消息先行。

# fork()

这用于创建新进程。新创建的进程称为子进程，它与父进程并发运行。在执行 `fork` 函数后，程序的执行继续，`fork` 函数之后的指令由父进程和子进程同时执行。如果系统调用成功，它将返回子进程的进程 ID，并将 `0` 返回给新创建的子进程。如果子进程未创建，函数返回负值。

现在，让我们从第一个使用管道实现进程间通信的配方开始。

# 一个进程，既从管道写入又从管道读取

在这里，我们将学习单个进程如何通过管道进行读写操作。

# 如何实现...

1.  定义一个大小为 `2` 的数组，并将其作为参数传递给 `pipe` 函数。

1.  调用 `write` 函数并将选定的字符串写入管道的 `write` 端。为第二条消息重复此过程。

1.  调用 `read` 函数从管道读取第一条消息。再次调用 `read` 函数以读取第二条消息。

`readwritepipe.c` 程序用于写入管道并随后从管道读取，如下所示：

```cpp
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define max 50

int main()
{
    char str[max];
    int pp[2];

    if (pipe(pp) < 0)
        exit(1);
    printf("Enter first message to write into pipe: ");
    gets(str);
    write(pp[1], str, max);
    printf("Enter second message to write into pipe: ");
    gets(str);
    write(pp[1], str, max);
    printf("Messages read from the pipe are as follows:\n");
    read(pp[0], str, max);
    printf("%s\n", str);
    read(pp[0], str, max);
    printf("%s\n", str);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

我们定义了一个大小为 `50` 的宏 `max`，一个大小为 `max` 的字符串 `str`，以及一个大小为 `2` 的数组 `pp`。我们将调用 `pipe` 函数连接两个进程并将 `pp` 数组传递给它。索引位置 `pp[0]` 将获取管道的读取端文件描述符，而 `pp[1]` 将获取管道的写入端文件描述符。如果 `pipe` 函数没有成功执行，程序将退出。

你将被提示输入将要写入管道的第一个消息。你输入的文本将被分配给字符串变量 `str`。调用 `write` 函数，`str` 中的字符串将被写入管道 `pp`。重复此过程以写入第二个消息。你输入的第二个文本也将被写入管道。

显然，第二个文本将在管道中第一个文本之后写入。现在，调用 `read` 函数从管道读取。管道中首先输入的文本将被读取并分配给字符串变量 `str`，然后显示在屏幕上。再次调用 `read` 函数，管道中的第二个文本消息将从其读取端读取并分配给字符串变量 `str`，然后显示在屏幕上。

让我们使用 GCC 编译 `readwritepipe.c` 程序，如下所示：

```cpp
$ gcc readwritepipe.c -o readwritepipe
```

如果你没有错误或警告，这意味着 `readwritepipe.c` 程序已被编译成可执行文件 `readwritepipe.exe`。让我们运行这个可执行文件：

```cpp
$ ./readwritepipe
Enter the first message to write into pipe: This is the first message for the pipe
Enter the second message to write into pipe: Second message for the pipe
Messages read from the pipe are as follows:
This is the first message for the pipe
Second message for the pipe
```

在前面的程序中，主线程负责从管道写入和读取。但如果我们想一个进程向管道写入，另一个进程从管道读取怎么办？让我们看看如何实现这一点。

# 一个进程向管道写入，另一个进程从管道读取

在这个菜谱中，我们将使用 `fork` 系统调用创建一个子进程。然后，我们将使用子进程向管道写入，并通过父进程从管道读取，从而在两个进程之间建立通信。

# 如何做到这一点…

1.  定义一个大小为 `2` 的数组。

1.  调用 `pipe` 函数连接两个进程并将我们之前定义的数组传递给它。

1.  调用 `fork` 函数创建一个新的子进程。

1.  输入将要写入管道的消息。使用新创建的子进程调用 `write` 函数。

1.  父进程调用 `read` 函数读取已写入管道的文本。

通过子进程向管道写入并通过父进程从管道读取的 `pipedemo.c` 程序如下：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define max  50

int main()
{
    char wstr[max];
    char rstr[max];
    int pp[2];
    pid_t p;
    if(pipe(pp) < 0)
    {
        perror("pipe");
    } 
    p = fork();
    if(p >= 0)
    {
        if(p == 0)
        {
            printf ("Enter the string : ");
            gets(wstr);
            write (pp[1] , wstr , strlen(wstr));
            exit(0);
        }
        else
        {
            read (pp[0] , rstr , sizeof(rstr));
            printf("Entered message : %s\n " , rstr);
            exit(0);
        }
    }
    else
    {
        perror("fork");
        exit(2);
    }        
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

定义一个大小为 `50` 的宏 `max` 和两个大小为 `max` 的字符串变量 `wstr` 和 `rstr`。`wstr` 字符串将用于向管道写入，而 `rstr` 将用于从管道读取。定义一个大小为 `2` 的数组 `pp`，它将用于存储管道的读写端文件描述符。定义一个 `pid_t` 数据类型的变量 `p`，它将用于存储进程 ID。

我们将调用 `pipe` 函数来连接两个进程，并将 `pp` 数组传递给它。`pp[0]` 索引位置将获取管道的读取端文件描述符，而 `pp[1]` 将获取管道的写入端文件描述符。如果 `pipe` 函数没有成功执行，程序将退出。

然后，我们将调用 `fork` 函数来创建一个新的子进程。你将被提示输入要写入管道的消息。你输入的文本将被分配给字符串变量 `wstr`。当我们使用新创建的子进程调用 `write` 函数时，`wstr` 变量中的字符串将被写入管道 `pp`。之后，父进程将调用 `read` 函数来读取写入管道的文本。从管道读取的文本将被分配给字符串变量 `rstr`，并随后在屏幕上显示。

让我们使用 GCC 编译 `pipedemo.c` 程序，如下所示：

```cpp
$ gcc pipedemo.c -o pipedemo
```

如果你没有错误或警告，这意味着 `pipedemo.c` 程序已经被编译成一个可执行文件，名为 `pipedemo.exe`。让我们运行这个可执行文件：

```cpp
$ ./pipedemo
Enter the string : This is a message from the pipe
Entered message : This is a message from the pipe
```

哇！我们已经成功使用管道在进程之间进行了通信。现在，让我们继续下一个菜谱！

# 使用 FIFO 在进程之间进行通信

在这个菜谱中，我们将学习两个进程如何使用命名管道（也称为 FIFO）进行通信。这个菜谱分为以下两个部分：

+   展示如何将数据写入 FIFO

+   展示如何从 FIFO 读取数据

我们在先前的菜谱中学到的函数和术语也适用于此处。

# 向 FIFO 写入数据

如其名所示，在这个菜谱中，我们将学习如何将数据写入 FIFO。

# 如何操作...

1.  调用 `mkfifo` 函数来创建一个新的 FIFO 特殊文件。

1.  通过调用 `open` 函数以只写模式打开 FIFO 特殊文件。

1.  输入要写入 FIFO 特殊文件的文本。

1.  关闭 FIFO 特殊文件。

用于向 FIFO 写入数据的 `writefifo.c` 程序如下：

```cpp
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
    int fw;
    char str[255];
    mkfifo("FIFOPipe", 0666);
    fw = open("FIFOPipe", O_WRONLY);
    printf("Enter text: ");
    gets(str);
    write(fw,str, sizeof(str));
    close(fw);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

假设我们已经定义了一个大小为 `255` 的字符串 `str`。我们将调用 `mkfifo` 函数来创建一个新的 FIFO 特殊文件。我们将使用名为 `FIFOPipe` 的名称创建 FIFO 特殊文件，并为所有者、组和其他用户设置读写权限。

我们将通过调用`open`函数以只写模式打开这个 FIFO 特殊文件。然后，我们将打开的 FIFO 特殊文件的文件描述符分配给`fw`变量。你将被提示输入要写入文件的文本。你输入的文本将被分配给`str`变量，然后当调用`write`函数时，它将被写入特殊的 FIFO 文件。最后，关闭 FIFO 特殊文件。

让我们使用 GCC 编译`writefifo.c`程序，如下所示：

```cpp
$ gcc writefifo.c -o writefifo
```

如果你没有错误或警告，这意味着`writefifo.c`程序已编译成可执行文件`writefifo.exe`。让我们运行这个可执行文件：

```cpp
$ ./writefifo
Enter text: This is a named pipe demo example called FIFO
```

如果你的程序没有提示输入字符串，这意味着它正在等待 FIFO 的另一端打开。也就是说，你需要在第二个终端屏幕上运行下一个菜谱，*从 FIFO 读取数据*。请在 Cygwin 上按*Alt+F2*打开下一个终端屏幕。

现在，让我们检查这个菜谱的另一个部分。

# 从 FIFO 读取数据

在这个菜谱中，我们将看到如何从 FIFO 读取数据。

# 如何做到这一点…

1.  通过调用`open`函数以只读模式打开 FIFO 特殊文件。

1.  使用`read`函数从 FIFO 特殊文件中读取文本。

1.  关闭 FIFO 特殊文件。

用于从命名管道（FIFO）读取的`readfifo.c`程序如下：

```cpp
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUFFSIZE 255

int main()
{
    int fr;
    char str[BUFFSIZE];
    fr = open("FIFOPipe", O_RDONLY);
    read(fr, str, BUFFSIZE);
    printf("Read from the FIFO Pipe: %s\n", str);
    close(fr);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

我们首先定义一个名为`BUFFSIZE`的宏，其大小为`255`，以及一个名为`str`的字符串，其大小也是`BUFFSIZE`，即 255 个字符。我们将通过调用`open`函数以只读模式打开名为`FIFOPipe`的 FIFO 特殊文件。打开的 FIFO 特殊文件的文件描述符将被分配给`fr`变量。

使用`read`函数，从 FIFO 特殊文件中读取的文本将被分配到`str`字符串变量。从 FIFO 特殊文件中读取的文本将被显示在屏幕上。最后，关闭 FIFO 特殊文件。

现在，按*Alt + F2*打开第二个终端窗口。在第二个终端窗口中，让我们使用 GCC 编译`readfifo.c`程序，如下所示：

```cpp
$ gcc readfifo.c -o readfifo
```

如果你没有错误或警告，这意味着`readfifo.c`程序已编译成可执行文件`readfifo.exe`。让我们运行这个可执行文件：

```cpp
$ ./readfifo
Read from the FIFO Pipe: This is a named pipe demo example called FIFO
```

当你运行`readfifo.exe`文件时，你会在之前运行`writefifo.c`程序的终端屏幕上发现，会提示你输入一个字符串。当你在这个终端上输入一个字符串并按*Enter*键时，你会得到`readfifo.c`程序的输出。

哇！我们已经成功使用 FIFO 在进程之间进行了通信。现在，让我们继续下一个菜谱！

# 使用套接字编程在客户端和服务器之间进行通信

在这个菜谱中，我们将学习服务器进程的数据是如何发送到客户端进程的。这个菜谱分为以下几部分：

+   向客户端发送数据

+   读取从服务器发送的数据

在我们开始介绍食谱之前，让我们快速回顾一下在成功的客户端-服务器通信中使用的函数、结构和术语。

# 客户端-服务器模型

对于 IPC，使用不同的模型，但最流行的是客户端-服务器模型。在这个模型中，每当客户端需要某些信息时，它会连接到另一个称为服务器的进程。但在建立连接之前，客户端需要知道服务器是否已经存在，并且它应该知道服务器的地址。

另一方面，服务器旨在满足客户端的需求，在建立连接之前不需要知道客户端的地址。为了建立连接，需要一个基本构造，称为套接字，并且连接的进程必须各自建立自己的套接字。客户端和服务器需要遵循某些程序来建立它们的套接字。

在客户端建立套接字时，使用`socket`函数系统调用来创建一个套接字。之后，使用`connect`函数系统调用来将该套接字连接到服务器的地址，然后通过调用`read`函数和`write`函数系统调用来发送和接收数据。

在服务器端建立套接字时，再次使用`socket`函数系统调用来创建一个套接字，然后使用`bind`函数系统调用来将该套接字绑定到一个地址。之后，调用`listen`函数系统调用来监听连接。最后，通过调用`accept`函数系统调用来接受连接。

# `struct sockaddr_in`结构

此结构引用了用于保持地址的套接字元素。以下是该结构的内置成员：

```cpp
struct sockaddr_in {
 short int sin_family;
 unsigned short int sin_port;
 struct in_addr sin_addr;
 unsigned char sin_zero[8];
};
```

这里，我们有以下内容：

+   `sin_family`：表示一个地址族。有效的选项有`AF_INET`、`AF_UNIX`、`AF_NS`和`AF_IMPLINK`。在大多数应用程序中，使用的地址族是`AF_INET`。

+   `sin_port`：表示 16 位服务端口号。

+   `sin_addr`：表示 32 位 IP 地址。

+   `sin_zero`：这个成员不使用，通常设置为`NULL`。

`struct in_addr`包含一个成员，如下所示：

```cpp

struct in_addr {
     unsigned long s_addr; 
};
```

这里，`s_addr`用于表示网络字节序中的地址。

# `socket()`

此函数创建了一个通信端点。为了建立通信，每个进程需要在通信线的末端有一个套接字。此外，两个通信进程必须具有相同的套接字类型，并且它们都应该在同一个域中。以下是创建套接字的语法：

```cpp
int socket(int domain, int type, int protocol);
```

这里，`domain`表示要创建套接字的通信域。基本上，指定了`地址族`或`协议族`，这将用于通信。

一些流行的`地址族`如下所示：

+   `AF_LOCAL`：这用于本地通信。

+   `AF_INET`：这用于 IPv4 互联网协议。

+   `AF_INET6`：这用于 IPv6 互联网协议。

+   `AF_IPX`: 这用于使用标准**IPX**（即**Internetwork Packet Exchange**）套接字地址的协议。

+   `AF_PACKET`: 这用于数据包接口。

+   `type`: 表示要创建的套接字类型。以下是一些流行的套接字类型：

+   `SOCK_STREAM`: 流套接字使用**传输控制协议 (TCP**)作为字符的连续流进行通信。TCP 是一种可靠的面向流的协议。因此，`SOCK_STREAM`类型提供了可靠、双向和基于连接的字节流。

+   `SOCK_DGRAM`: 数据报套接字使用**用户数据报协议 (UDP**)一次性读取整个消息。UDP 是一种不可靠的、无连接的、面向消息的协议。这些消息具有固定的最大长度。

+   `SOCK_SEQPACKET`: 为数据报提供可靠、双向和基于连接的传输路径。

+   `protocol`: 表示与套接字一起使用的协议。指定一个`0`值，以便您可以使用适合请求的套接字类型的默认协议。

您可以将前面列表中的`AF_`前缀替换为`PF_`以表示`协议族`。

在成功执行后，`socket`函数返回一个文件描述符，可以用来管理套接字。

# memset()

这用于使用指定的值填充内存块。以下是它的语法：

```cpp
void *memset(void *ptr, int v, size_t n);
```

在这里，`ptr`指向要填充的内存地址，`v`是要填充到内存块中的值，而`n`是要填充的字节数，从指针的位置开始。

# htons()

这用于将主机无符号短整数转换为网络字节序。

# bind()

使用`socket`函数创建的套接字保持在分配的地址族中。为了使套接字能够接收连接，需要为其分配一个地址。`bind`函数将地址分配给指定的套接字。以下是它的语法：

```cpp
   int bind(int fdsock, const struct sockaddr *structaddr, socklen_t lenaddr);
```

在这里，`fdsock`代表套接字的文件描述符，`structaddr`代表包含要分配给套接字的地址的`sockaddr`结构，而`lenaddr`代表由`structaddr`指向的地址结构的大小。

# listen()

它在套接字上监听连接，以便接受传入的连接请求。以下是它的语法：

```cpp
int listen(int sockfd, int lenque);
```

在这里，`sockfd`代表套接字的文件描述符，而`lenque`代表给定套接字的挂起连接队列的最大长度。如果队列已满，将生成错误。

如果函数成功，它返回零，否则返回`-1`。

# accept()

它接受监听套接字上的新连接，即从挂起的连接队列中选取的第一个连接。实际上，会创建一个新的套接字，其套接字类型协议和地址族与指定的套接字相同，并为该套接字分配一个新的文件描述符。以下是它的语法：

```cpp
int accept(int socket, struct sockaddr *address, socklen_t *len);
```

在这里，我们需要解决以下问题：

+   `socket`：表示等待新连接的套接字的文件描述符。这是当 `socket` 函数通过 `bind` 函数绑定到地址并成功调用 `listen` 函数时创建的套接字。

+   `address`：通过此参数返回连接套接字的地址。它是一个指向 `sockaddr` 结构的指针，通过该结构返回连接套接字的地址。

+   `len`：表示提供的 `sockaddr` 结构的长度。返回时，此参数包含以字节为单位返回的地址长度。

# send()

这用于将指定的消息发送到另一个套接字。在调用此函数之前，套接字需要处于连接状态。以下是其语法：

```cpp
       ssize_t send(int fdsock, const void *buf, size_t length, int flags);
```

在这里，`fdsock` 代表要发送消息的套接字的文件描述符，`buf` 指向包含要发送消息的缓冲区，`length` 代表以字节为单位要发送的消息长度，而 `flags` 指定要发送的消息类型。通常，其值保持为 `0`。

# connect()

这在套接字上初始化一个连接。以下是其语法：

```cpp
int connect(int fdsock, const struct sockaddr *addr,  socklen_t len);
```

在这里，`fdsock` 代表要建立连接的套接字的文件描述符，`addr` 代表包含套接字地址的结构，而 `len` 代表包含地址的结构 `addr` 的大小。

# recv()

这用于从已连接的套接字接收消息。套接字可以是连接模式或无连接模式。以下是其语法：

```cpp
ssize_t recv(int fdsock, void *buf, size_t len, int flags);
```

在这里，`fdsock` 代表必须从中获取消息的套接字的文件描述符，`buf` 代表存储接收到的消息的缓冲区，`len` 指定由 `buf` 参数指向的缓冲区的长度，而 `flags` 指定正在接收的消息类型。通常，其值保持为 `0`。

我们现在可以开始本食谱的第一部分——如何向客户端发送数据。

# 向客户端发送数据

在本部分的食谱中，我们将学习服务器如何将所需数据发送到客户端。

# 如何操作…

1.  定义一个 `sockaddr_in` 类型的变量。

1.  调用 `socket` 函数创建套接字。为套接字指定的端口号是 `2000`。

1.  调用 `bind` 函数为其分配一个 IP 地址。

1.  调用 `listen` 函数。

1.  调用 `accept` 函数。

1.  调用 `send` 函数将用户输入的消息发送到套接字。

1.  客户端端的套接字将接收消息。

发送消息到客户端的服务器程序 `serverprog.c` 如下所示：

```cpp
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

int main(){
    int serverSocket, toSend;
    char str[255];
    struct sockaddr_in server_Address;
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    server_Address.sin_family = AF_INET;
    server_Address.sin_port = htons(2000);
    server_Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    memset(server_Address.sin_zero, '\0', sizeof 
    server_Address.sin_zero); 
    bind(serverSocket, (struct sockaddr *) &server_Address, 
    sizeof(server_Address));
    if(listen(serverSocket,5)==-1)
    {
        printf("Not able to listen\n");
        return -1;
    }
    printf("Enter text to send to the client: ");
    gets(str);
    toSend = accept(serverSocket, (struct sockaddr *) NULL, NULL);
    send(toSend,str, strlen(str),0);
    return 0;
}
```

让我们看看幕后发生了什么。

# 它是如何工作的…

我们将首先定义一个大小为 `255` 的字符串和一个类型为 `sockaddr_in` 的 `server_Address` 变量。这个结构引用了套接字的元素。然后，我们将调用 `socket` 函数以 `serverSocket` 的名称创建套接字。套接字是通信的端点。为套接字提供的地址族是 `AF_INET`，选择的套接字类型是流套接字类型，因为我们想要的通信是字符的连续流。

为套接字指定的地址族是 `AF_INET`，用于 IPv4 互联网协议。为套接字指定的端口号是 `2000`。使用 `htons` 函数，将短整数 `2000` 转换为网络字节序，然后作为端口号应用。`server_Address` 结构的第四个参数 `sin_zero` 通过调用 `memset` 函数设置为 `NULL`。

要使创建的 `serverSocket` 能够接收连接，请调用 `bind` 函数为其分配一个地址。使用 `server_Address` 结构的 `sin_addr` 成员，将一个 32 位 IP 地址应用到套接字上。因为我们是在本地机器上工作，所以将本地主机地址 `127.0.0.1` 分配给套接字。现在，套接字可以接收连接。我们将调用 `listen` 函数使 `serverSocket` 能够接受传入的连接请求。套接字可以有的最大挂起连接数是 5。

你将被提示输入要发送给客户端的文本。你输入的文本将被分配给 `str` 字符串变量。通过调用 `accept` 函数，我们将使 `serverSocket` 能够接受新的连接。

连接套接字的地址将通过类型为 `sockaddr_in` 的结构返回。返回并准备好接受连接的套接字被命名为 `toSend`。我们将调用 `send` 函数发送你输入的消息。客户端的套接字将接收这条消息。

让我们使用 GCC 编译 `serverprog.c` 程序，如下所示：

```cpp
$ gcc serverprog.c -o serverprog
```

如果你没有收到错误或警告，这意味着 `serverprog.c` 程序已编译成可执行文件，名为 `serverprog.exe`。让我们运行这个可执行文件：

```cpp
$ ./serverprog
Enter text to send to the client: thanks and good bye
```

现在，让我们看看这个说明的另一个部分。

# 读取从服务器发送的数据

在本部分的说明中，我们将学习从服务器发送的数据是如何接收并在屏幕上显示的。

# 如何做到这一点…

1.  定义一个类型为 `sockaddr_i` 的变量。

1.  调用 `socket` 函数创建套接字。为套接字指定的端口号是 `2000`。

1.  调用 `connect` 函数初始化与套接字的连接。

1.  因为我们是在本地机器上工作，所以将本地主机地址 `127.0.0.1` 分配给套接字。

1.  调用 `recv` 函数从已连接的套接字接收消息。从套接字读取的消息随后将在屏幕上显示。

客户端程序 `clientprog.c` 用于读取从服务器发送的消息如下：

```cpp
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

int main(){
    int clientSocket;
    char str[255];
    struct sockaddr_in client_Address;
    socklen_t address_size;
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    client _Address.sin_family = AF_INET;
    client _Address.sin_port = htons(2000);
    client _Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    memset(client _Address.sin_zero, '\0', sizeof client_Address.sin_zero); 
    address_size = sizeof server_Address;
    connect(clientSocket, (struct sockaddr *) &client_Address, address_size);
    recv(clientSocket, str, 255, 0);
    printf("Data received from server: %s", str);  
    return 0;
}
```

让我们幕后看看。

# 它是如何工作的...

因此，我们定义了一个大小为 `255` 的字符串和一个名为 `client_Address` 的 `sockaddr_in` 类型的变量。我们将调用 `socket` 函数创建一个名为 `clientSocket` 的套接字。

为套接字提供的地址族是 `AF_INET`，用于 IPv4 互联网协议，所选的套接字类型是流式套接字类型。指定的套接字端口号是 `2000`。通过使用 `htons` 函数，将短整数 `2000` 转换为网络字节序，然后作为端口号应用。

我们将通过调用 `memset` 函数将 `client_Address` 结构的第四个参数 `sin_zero` 设置为 `NULL`。我们将通过调用 `connect` 函数初始化 `clientSocket` 的连接。通过使用 `client_Address` 结构的 `sin_addr` 成员，将一个 32 位 IP 地址应用到套接字上。因为我们是在本地机器上工作，所以将本地主机地址 `127.0.0.1` 分配给套接字。最后，我们将调用 `recv` 函数从已连接的 `clientSocket` 接收消息。从套接字读取的消息将被分配给 `str` 字符串变量，然后显示在屏幕上。

现在，按 *Alt + F2* 打开第二个终端窗口。在这里，我们将使用 GCC 编译 `clientprog.c` 程序，如下所示：

```cpp
$ gcc clientprog.c -o clientprog
```

如果没有错误或警告，这意味着 `clientprog.c` 程序已编译成可执行文件，名为 `clientprog.exe`。让我们运行这个可执行文件：

```cpp
$ ./clientprog
Data received from server: thanks and good bye
```

哇！我们已经成功使用套接字编程在客户端和服务器之间进行了通信。现在，让我们继续下一个菜谱！

# 使用 UDP 套接字进行进程间通信

在本菜谱中，我们将学习如何使用 UDP 套接字在客户端和服务器之间实现双向通信。本菜谱分为以下几部分：

+   等待客户端的消息并使用 UDP 套接字发送回复

+   使用 UDP 套接字向服务器发送消息并从服务器接收回复

在我们开始这些菜谱之前，让我们快速回顾一下在成功使用 UDP 套接字进行进程间通信时使用的函数、结构和术语。

# 使用 UDP 套接字进行服务器-客户端通信

在使用 UDP 进行通信的情况下，客户端不需要与服务器建立连接，而是简单地发送一个数据报。服务器不需要接受连接；它只需等待客户端发送数据报。每个数据报都包含发送者的地址，使服务器能够根据数据报是从哪里发送的来识别客户端。

对于通信，UDP 服务器首先创建一个 UDP 套接字并将其绑定到服务器地址。然后，服务器等待来自客户端的数据报文到达。一旦到达，服务器处理数据报文并向客户端发送回复。这个过程会不断重复。

另一方面，UDP 客户端为了通信，创建一个 UDP 套接字，向服务器发送消息，并等待服务器的响应。如果客户端想要向服务器发送更多消息，则会不断重复此过程，否则套接字描述符将关闭。

# `bzero()`

这将在指定的区域放置*n*个零值字节。其语法如下：

```cpp
void bzero(void *r, size_t n);
```

在这里，`r`是指向`r`的区域，`n`是要放置在由`r`指向的区域中的零值字节的数量。

# `INADDR_ANY`

这是一个在不想将套接字绑定到任何特定 IP 时使用的 IP 地址。基本上，在实现通信时，我们需要将我们的套接字绑定到 IP 地址。当我们不知道我们机器的 IP 地址时，我们可以使用特殊的 IP 地址`INADDR_ANY`。它允许我们的服务器接收被任何接口针对的数据包。

# `sendto()`

这用于在指定的套接字上发送消息。消息可以在连接模式以及无连接模式下发送。在无连接模式下，消息发送到指定的地址。其语法如下：

```cpp
ssize_t sendto(int fdsock, const void *buff, size_t len, int flags, const struct sockaddr *recv_addr, socklen_t recv_len);
```

在这里，我们需要处理以下内容：

+   `fdsock`：指定套接字的文件描述符。

+   `buff`：指向包含要发送消息的缓冲区。

+   `len`：指定消息的字节数。

+   `flags`：指定正在传输的消息类型。通常，其值保持为 0。

+   `recv_addr`：指向包含接收者地址的`sockaddr`结构。地址的长度和格式取决于分配给套接字的地址族。

+   `recv_len`：指定由`recv_addr`参数指向的`sockaddr`结构的长度。

在成功执行后，函数返回发送的字节数，否则返回`-1`。

# `recvfrom()`

这用于从连接模式或无连接模式的套接字接收消息。其语法如下：

```cpp
ssize_t recvfrom(int fdsock, void *buffer, size_t length, int flags, struct sockaddr *address, socklen_t *address_len);
```

在这里，我们需要处理以下内容：

+   `fdsock`：表示套接字的文件描述符。

+   `buffer`：表示存储消息的缓冲区。

+   `length`：表示由`buffer`参数指向的缓冲区中的字节数。

+   `flags`：表示接收到的消息类型。

+   `address`：表示存储发送地址的`sockaddr`结构。地址的长度和格式取决于套接字的地址族。

+   `address_len`：表示由地址参数指向的`sockaddr`结构的长度。

函数返回写入缓冲区的消息长度，该缓冲区由缓冲区参数指向。

现在，我们可以开始这个配方的第一部分：使用 UDP 套接字准备服务器等待并回复客户端的消息。

# 使用 UDP 套接字等待客户端消息并发送回复

在本部分的配方中，我们将学习服务器如何等待客户端的消息，以及当收到客户端的消息时，它如何回复客户端。

# 如何做到这一点...

1.  定义两个类型为`sockaddr_in`的变量。调用`bzero`函数初始化结构体。

1.  调用`socket`函数创建套接字。为套接字提供的地址族是`AF_INET`，选择的套接字类型是数据报类型。

1.  初始化`sockaddr_in`结构体的成员以配置套接字。为套接字指定的端口号是`2000`。使用特殊 IP 地址`INADDR_ANY`为套接字分配 IP 地址。

1.  调用`bind`函数将地址分配给它。

1.  调用`recvfrom`函数从 UDP 套接字接收消息，即从客户端机器接收。在从客户端机器读取的消息中添加一个空字符`\0`，并在屏幕上显示。输入要发送给客户端的回复。

1.  调用`sendto`函数将回复发送给客户端。

等待客户端消息并发送回复的 UDP 套接字服务器程序`udps.c`如下所示：

```cpp
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include<netinet/in.h>
#include <stdlib.h> 

int main()
{   
    char msgReceived[255];
    char msgforclient[255];
    int UDPSocket, len;
    struct sockaddr_in server_Address, client_Address;
    bzero(&server_Address, sizeof(server_Address));
    printf("Waiting for the message from the client\n");
    if ( (UDPSocket = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("Socket could not be created"); 
        exit(1); 
    }      
    server_Address.sin_addr.s_addr = htonl(INADDR_ANY);
    server_Address.sin_port = htons(2000);
    server_Address.sin_family = AF_INET; 
    if ( bind(UDPSocket, (const struct sockaddr *)&server_Address, 
    sizeof(server_Address)) < 0 ) 
    { 
        perror("Binding could not be done"); 
        exit(1); 
    } 
    len = sizeof(client_Address);
    int n = recvfrom(UDPSocket, msgReceived, sizeof(msgReceived),  0, 
    (struct sockaddr*)&client_Address,&len);
    msgReceived[n] = '\0';
    printf("Message received from the client: ");
    puts(msgReceived);
    printf("Enter the reply to be sent to the client: ");
    gets(msgforclient);
    sendto(UDPSocket, msgforclient, 255, 0, (struct 
    sockaddr*)&client_Address, sizeof(client_Address));
    printf("Reply to the client sent \n");
}
```

让我们看看背后的情况。

# 它是如何工作的...

我们首先定义两个名为`msgReceived`和`msgforclient`的字符串，它们的大小都是`255`。这两个字符串将用于接收来自客户端的消息和向客户端发送消息。然后，我们将定义两个类型为`sockaddr_in`的变量，`server_Address`和`client_Address`。这些结构将引用套接字元素并分别存储服务器和客户端的地址。我们将调用`bzero`函数初始化`server_Address`结构体，即`server_Address`结构体的所有成员都将填充零。

服务器如预期的那样等待来自客户端的数据报。因此，屏幕上显示以下文本消息：“等待来自客户端的消息”。我们通过调用名为`UDPSocket`的`socket`函数创建套接字。为套接字提供的地址族是`AF_INET`，选择的套接字类型是数据报。`server_Address`结构体的成员被初始化以配置套接字。

使用`sin_family`成员，指定给套接字的地址族是`AF_INET`，它用于 IPv4 互联网协议。指定给套接字的端口号是`2000`。使用`htons`函数，将短整数`2000`转换为网络字节序，然后作为端口号应用。然后，我们使用一个特殊的 IP 地址`INADDR_ANY`来为套接字分配 IP 地址。使用`htonl`函数，将`INADDR_ANY`转换为网络字节序，然后作为套接字的地址应用。

为了使创建的套接字`UDPSocket`能够接收连接，我们将调用`bind`函数将地址分配给它。我们将调用`recvfrom`函数从 UDP 套接字接收消息，即从客户端机器接收。从客户端机器读取的消息被分配给`msgReceived`字符串，该字符串在`recvfrom`函数中提供。在`msgReceived`字符串中添加一个空字符`\0`，并在屏幕上显示。之后，您将被提示输入要发送给客户端的回复。输入的回复被分配给`msgforclient`。通过调用`sendto`函数，将回复发送给客户端。发送消息后，屏幕上显示以下消息：`Reply to the client sent`。

现在，让我们看看本食谱的另一个部分。

# 使用 UDP 套接字向服务器发送消息并从服务器接收回复

正如名称所暗示的，在本食谱中，我们将向您展示客户端如何通过 UDP 套接字向服务器发送消息，然后从服务器接收回复。

# 如何做到这一点…

1.  执行本食谱前一部分的前三个步骤。将本地主机 IP 地址`127.0.0.1`分配给套接字地址。

1.  输入要发送给服务器的消息。调用`sendto`函数将消息发送到服务器。

1.  调用`recvfrom`函数从服务器获取消息。从服务器接收到的消息随后在屏幕上显示。

1.  关闭套接字的描述符。

客户端程序`udpc.c`用于通过 UDP 套接字向服务器发送消息并接收回复，如下所示：

```cpp
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include<netinet/in.h>
#include<unistd.h>
#include<stdlib.h>

int main()
{   
    char msgReceived[255];
    char msgforserver[255];
    int UDPSocket, n;
    struct sockaddr_in client_Address;    
    printf("Enter the message to send to the server: ");
    gets(msgforserver);
    bzero(&client_Address, sizeof(client_Address));
    client_Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    client_Address.sin_port = htons(2000);
    client_Address.sin_family = AF_INET;     
    if ( (UDPSocket = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("Socket could not be created"); 
        exit(1); 
    } 
    if(connect(UDPSocket, (struct sockaddr *)&client_Address, 
    sizeof(client_Address)) < 0)
    {
        printf("\n Error : Connect Failed \n");
        exit(0);
    } 
    sendto(UDPSocket, msgforserver, 255, 0, (struct sockaddr*)NULL, 
    sizeof(client_Address));
    printf("Message to the server sent. \n");
    recvfrom(UDPSocket, msgReceived, sizeof(msgReceived), 0, (struct 
    sockaddr*)NULL, NULL);
    printf("Received from the server: ");
    puts(msgReceived);
    close(UDPSocket);
}
```

现在，让我们看看幕后。

# 它是如何工作的...

在本食谱的第一部分中，我们已经通过`msgReceived`和`msgforclient`这两个名称定义了两个字符串，它们的大小都是`255`。我们还定义了两个变量`server_Address`和`client_Address`，它们的类型为`sockaddr_in`。

现在，您将被提示输入要发送给服务器的消息。您输入的消息将被分配给`msgforserver`字符串。然后，我们将调用`bzero`函数初始化`client_Address`结构，即`client_Address`结构的所有成员都将填充零。

接下来，我们将初始化`client_Address`结构的成员以配置套接字。使用`sin_family `成员，为套接字指定的地址族是`AF_INET`，用于 IPv4 互联网协议。为套接字指定的端口号是`2000`。通过使用`htons`函数，将短整数`2000`转换为网络字节顺序，然后将其作为端口号应用。然后，我们将本地主机 IP 地址`127.0.0.1`分配给套接字。我们将对本地主机地址调用`inet_addr`函数，将包含地址的标准 IPv4 点分十进制表示法字符串转换为整数值（适合用作互联网地址），然后再将其应用于`client_Address`结构的`sin_addr`成员。

我们将调用`socket`函数以`UDPSocket`为名称创建一个套接字。为套接字提供的地址族是`AF_INET`，选择的套接字类型是数据报。

接下来，我们将调用`sendto`函数将分配给`msgforserver`字符串的消息发送到服务器。同样，我们将调用`recvfrom`函数从服务器获取消息。从服务器接收到的消息分配给`msgReceived`字符串，然后显示在屏幕上。最后，关闭套接字描述符。

让我们使用 GCC 来编译`udps.c`程序，如下所示：

```cpp
$ gcc udps.c -o udps
```

如果没有错误或警告，这意味着`udps.c`程序已编译成可执行文件，`udps.exe`。让我们运行这个可执行文件：

```cpp
$ ./udps
Waiting for the message from the client
```

现在，按*Alt + F2* 打开第二个终端窗口。在这里，让我们再次使用 GCC 来编译`udpc.c`程序，如下所示：

```cpp
$ gcc udpc.c -o udpc
```

如果没有错误或警告，这意味着`udpc.c`程序已编译成可执行文件，`udpc.exe`。让我们运行这个可执行文件：

```cpp
$ ./udpc
Enter the message to send to the server: Will it rain today?
Message to the server sent.
```

服务器上的输出将给出以下输出：

```cpp
Message received from the client: Will it rain today?
Enter the reply to be sent to the client: It might
Reply to the client sent
```

一旦服务器发送回复，在客户端窗口，您将得到以下输出：

```cpp
Received from the server: It might
```

要运行演示使用共享内存和消息队列进行 IPC 的食谱，我们需要运行 Cygserver。如果您在 Linux 上运行这些程序，则可以跳过本节。让我们看看 Cygserver 是如何运行的。

# 运行 Cygserver

在执行运行 Cygwin 服务器命令之前，我们需要配置 Cygserver 并将其安装为服务。为此，您需要在终端上运行`cygserver.conf`脚本。以下是运行脚本后的输出：

```cpp
$ ./bin/cygserver-config
Generating /etc/cygserver.conf file
Warning: The following function requires administrator privileges!
Do you want to install cygserver as service? yes

The service has been installed under LocalSystem account.
To start it, call `net start cygserver' or `cygrunsrv -S cygserver'.

Further configuration options are available by editing the configuration
file /etc/cygserver.conf. Please read the inline information in that
file carefully. The best option for the start is to just leave it alone.

Basic Cygserver configuration finished. Have fun!
```

现在，Cygserver 已经配置并作为服务安装。下一步是运行服务器。要运行 Cygserver，您需要使用以下命令：

```cpp
$ net start cygserver
The CYGWIN cygserver service is starting.
The CYGWIN cygserver service was started successfully.
```

现在，Cygserver 正在运行，我们可以制作一个食谱来演示使用共享内存和消息队列进行 IPC。

# 使用消息队列从一个进程向另一个进程传递消息

在这个食谱中，我们将学习如何使用消息队列在两个进程之间建立通信。这个食谱分为以下几部分：

+   将消息写入消息队列

+   从消息队列中读取消息

在我们开始这些食谱之前，让我们快速回顾一下在成功使用共享内存和消息队列进行进程间通信时使用的函数、结构和术语。

# 在使用共享内存和消息队列进行进程间通信中使用的函数

在使用共享内存和消息队列进行进程间通信中最常用的函数和术语是 `ftok`、`shmget`、`shmat`、`shmdt`、`shmctl`、`msgget`、`msgrcv` 和 `msgsnd`。

# ftok()

这基于提供的文件名和 ID 生成一个 IPC 键。可以提供文件及其完整路径。文件必须引用一个现有文件。以下是它的语法：

```cpp
key_t ftok(const char *filename, int id);
```

如果提供相同的文件名（具有相同的路径）和相同的 ID，则 `ftok` 函数将生成相同的关键值。成功完成后，`ftok` 将返回一个键，否则返回 `-1`。

# shmget()

这分配了一个共享内存段，并返回与键关联的共享内存标识符。以下是它的语法：

```cpp
int shmget(key_t key, size_t size, int shmflg);
```

在这里，我们需要解决以下问题：

+   `key`: 这通常是调用 `ftok` 函数返回的值。如果您不想其他进程访问共享内存，也可以将键的值设置为 `IPC_PRIVATE`。

+   `size`: 表示所需共享内存段的大小。

+   `shmflg`: 这可以是以下任何常量：

    +   `IPC_CREAT`: 如果指定的键不存在共享内存标识符，则此操作将创建一个新的段。如果未使用此标志，则函数返回与键关联的共享内存段。

    +   `IPC_EXCL`: 如果指定的键已经存在段，则使 `shmget` 函数失败。

如果执行成功，该函数以非负整数的格式返回共享内存标识符，否则返回 `-1`。

# shmat()

这用于将共享内存段附加到给定的地址空间。也就是说，通过调用 `shmgt` 函数接收到的共享内存标识符需要与进程的地址空间相关联。以下是它的语法：

```cpp
void *shmat(int shidtfr, const void *addr, int flag);
```

在这里，我们需要解决以下问题：

+   `shidtfr`: 表示共享内存段的内存标识符。

+   `addr`: 表示需要附加段的地址空间。如果 `shmaddr` 是空指针，则段将附加到第一个可用的地址或由系统选择。

+   `flag`: 如果标志为 `SHM_RDONLY`，则将其附加为只读内存；否则，它是可读可写的。

如果成功执行，该函数将附加共享内存段并返回段的起始地址，否则返回 `-1`。

# shmdt()

这将共享内存段分离。以下是它的语法：

```cpp
int shmdt(const void *addr);
```

这里，`addr` 表示共享内存段所在的地址。

# `shmctl()`

这用于在指定的共享内存段上执行某些控制操作。以下是它的语法：

```cpp
int shmctl(int shidtr, int cmd, struct shmid_ds *buf);
```

在这里，我们必须处理以下问题：

+   `shidtr`：表示共享内存段的标识符。

+   `cmd`：可以具有以下常量之一：

    +   `IPC_STAT`：这会将与由`shidtr`表示的共享内存段关联的`shmid_ds`数据结构的内容复制到由`buf`指向的结构中。

    +   `IPC_SET`：这会将由`buf`指向的结构的内容写入与由`shidtr`表示的内存段关联的`shmid_ds`数据结构。

    +   `IPC_RMID`：这将从系统中删除由`shidtr`指定的共享内存标识符，并销毁与其相关的共享内存段和`shmid_ds`数据结构。

+   `buf`：这是指向`shmid_ds`结构的指针。

如果成功执行，函数返回`0`，否则返回`-1`。

# `msgget()`

这用于创建新的消息队列，以及访问与指定键相关联的现有队列。如果执行成功，则函数返回消息队列的标识符：

```cpp
       int msgget(key_t key, int flag);
```

在这里，我们必须处理以下问题：

+   `key`：这是一个由调用`ftok`函数检索的唯一键值。

+   `flag`：可以是以下常量中的任何一个：

    +   `IPC_CREAT`：如果消息队列不存在，则创建它并返回新创建的消息队列的标识符。如果消息队列已存在且提供了相应的键值，则返回其标识符。

    +   `IPC_EXCL`：如果同时指定了`IPC_CREAT`和`IPC_EXCL`，并且消息队列不存在，则创建它。然而，如果它已经存在，则函数将失败。

# `msgrcv()`

这用于从指定的消息队列中读取消息，该队列的标识符由用户提供。以下是它的语法：

```cpp
int msgrcv(int msqid, void *msgstruc, int msgsize, long typemsg, int flag);
```

在这里，我们必须处理以下问题：

+   `msqid`：表示需要从其中读取消息的队列的消息队列标识符。

+   `msgstruc`：这是一个用户定义的结构，用于放置读取的消息。用户定义的结构必须包含两个成员。一个是通常命名为`mtype`的成员，它必须是长整型，用于指定消息的类型，另一个通常称为`mesg`，它应该是`char`类型，用于存储消息。

+   `msgsize`：表示从消息队列中读取的文本大小，以字节为单位。如果读取的消息大于`msgsize`，则它将被截断为`msgsize`字节。

+   `typemsg`：指定需要接收队列上的哪个消息：

    +   如果`typemsg`为`0`，则接收队列上的第一个消息。

    +   如果`typemsg`大于`0`，则接收第一个`mtype`字段等于`typemsg`的消息。

    +   如果`typemsg`小于`0`，则接收`mtype`字段小于或等于`typemsg`的消息。

+   `flag`: 决定了在队列中找不到所需消息时要采取的操作。如果你不想指定`flag`，则保持其值为`0`。`flag`可以具有以下任何值：

    +   `IPC_NOWAIT`: 如果队列中没有所需的消息，则使`msgrcv`函数失败，即它不会使调用者等待队列中的适当消息。如果`flag`未设置为`IPC_NOWAIT`，它将使调用者等待队列中的适当消息而不是使函数失败。

    +   `MSG_NOERROR`: 这允许你接收比在`msgsize`参数中指定的尺寸更大的文本。它简单地截断文本并接收它。如果此`flag`未设置，则在接收较大文本时，函数将不会接收它并使函数失败。

如果函数执行成功，则函数返回实际放置在由`msgstruc`指向的结构体的文本字段中的字节数。在失败的情况下，函数返回`-1`。

# `msgsnd()`

这用于向队列发送或投递消息。以下是它的语法：

```cpp
 int msgsnd ( int msqid, struct msgbuf *msgstruc, int msgsize, int flag );
```

在这里，我们必须解决以下问题：

+   `msqid`: 表示我们想要发送的消息的队列标识符。队列标识符通常通过调用`msgget`函数来获取。

+   `msgstruc`: 这是一个指向用户定义的结构体的指针。它是包含我们想要发送到队列的消息的`mesg`成员。

+   `msgsize`: 表示消息的字节数。

+   `flag`: 决定了对消息采取的操作。如果`flag`值设置为`IPC_NOWAIT`，并且消息队列已满，则消息不会被写入队列，控制权将返回给调用进程。但如果`flag`未设置且消息队列已满，则调用进程将挂起，直到队列中有空间可用。通常，`flag`的值设置为`0`。

如果执行成功，则函数返回`0`，否则返回`-1`。

我们现在将开始本配方的第一部分：将消息写入队列。

# 将消息写入消息队列

在本部分的配方中，我们将学习服务器如何将所需的消息写入消息队列。

# 如何做到这一点...

1.  通过调用`ftok`函数生成一个 IPC 键。在创建 IPC 键时提供文件名和 ID。

1.  调用`msgget`函数创建一个新的消息队列。消息队列与步骤 1 中创建的 IPC 键相关联。

1.  定义一个包含两个成员的结构，`mtype`和`mesg`。将`mtype`成员的值设置为 1。

1.  输入将要添加到消息队列的消息。输入的字符串被分配给我们在步骤 3 中定义的结构体的`mesg`成员。

1.  调用`msgsnd`函数将输入的消息发送到消息队列。

将消息写入消息队列的`messageqsend.c`程序如下：

```cpp
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MSGSIZE     255

struct msgstruc {
    long mtype;
    char mesg[MSGSIZE];
};

int main()
{
    int msqid, msglen;
    key_t key;
    struct msgstruc msgbuf;
    system("touch messagefile");
    if ((key = ftok("messagefile", 'a')) == -1) {
        perror("ftok");
        exit(1);
    } 
    if ((msqid = msgget(key, 0666 | IPC_CREAT)) == -1) {
        perror("msgget");
        exit(1);
    }
    msgbuf.mtype = 1;
    printf("Enter a message to add to message queue : ");
    scanf("%s",msgbuf.mesg);
    msglen = strlen(msgbuf.mesg);
    if (msgsnd(msqid, &msgbuf, msglen, IPC_NOWAIT) < 0)
        perror("msgsnd");
    printf("The message sent is %s\n", msgbuf.mesg);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

我们将首先通过调用`ftok`函数来生成一个 IPC 密钥。在创建 IPC 密钥时提供的文件名和 ID 分别是`messagefile`和`a`。生成的密钥被分配给密钥变量。之后，我们将调用`msgget`函数来创建一个新的消息队列。该消息队列与使用`ftok`函数创建的 IPC 密钥相关联。

接下来，我们将定义一个名为`msgstruc`的结构，包含两个成员，`mtype`和`mesg`。`mtype`成员有助于确定从消息队列发送或接收的消息的序列号。`mesg`成员包含要读取或写入消息队列的消息。我们将定义一个名为`msgbuf`的变量，其类型为`msgstruc`结构。`mtype`成员的值被设置为`1`。

你将被提示输入要添加到消息队列的消息。你输入的字符串被分配给`msgbuf`结构的`mesg`成员。调用`msgsnd`函数将你输入的消息发送到消息队列。一旦消息被写入消息队列，屏幕上就会显示一条文本消息作为确认。

现在，让我们继续本配方的另一部分。

# 从消息队列中读取消息

在本部分的配方中，我们将学习如何读取写入消息队列的消息并将其显示在屏幕上。

# 如何做到这一点...

1.  调用`ftok`函数来生成一个 IPC 密钥。在创建 IPC 密钥时提供的文件名和 ID。这些必须与在消息队列中写入消息时生成密钥时使用的相同。

1.  调用`msgget`函数访问与 IPC 密钥相关联的消息队列。与该密钥相关联的消息队列已经包含了我们通过前面的程序写入的消息。

1.  定义一个包含两个成员的结构，`mtype`和`mesg`。

1.  调用`msgrcv`函数从相关消息队列中读取消息。在步骤 3 中定义的结构被传递给此函数。

1.  读取的消息随后显示在屏幕上。

以下是从消息队列中读取消息的`messageqrecv.c`程序：

```cpp
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdio.h>
#include <stdlib.h>
#define MSGSIZE     255

struct msgstruc {
    long mtype;
    char mesg[MSGSIZE];
};

int main()
{
    int msqid;
    key_t key;
    struct msgstruc rcvbuffer;

    if ((key = ftok("messagefile", 'a')) == -1) {
        perror("ftok");
        exit(1);
    }
    if ((msqid = msgget(key, 0666)) < 0)
    {
        perror("msgget");
        exit(1);
    }
    if (msgrcv(msqid, &rcvbuffer, MSGSIZE, 1, 0) < 0)
    {
        perror("msgrcv");
        exit(1);
    }
    printf("The message received is %s\n", rcvbuffer.mesg);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

首先，我们将调用`ftok`函数来生成一个 IPC 密钥。在创建 IPC 密钥时提供的文件名和 ID 分别是`messagefile`和`a`。这些文件名和 ID 必须与在消息队列中写入消息时生成密钥时使用的相同。生成的密钥被分配给密钥变量。

之后，我们将调用`msgget`函数来访问与 IPC 密钥相关联的消息队列。访问的消息队列的标识符被分配给`msqid`变量。与该密钥相关联的消息队列已经包含了我们之前程序中写入的消息。

然后，我们将定义一个名为`msgstruc`的结构，它有两个成员，`mtype`和`mesg`。`mtype`成员用于确定要从中读取的消息队列的序列号。`mesg`成员将用于存储从消息队列中读取的消息。然后，我们将定义一个名为`rcvbuffer`的变量，其类型为`msgstruc`结构。我们将调用`msgrcv`函数从相关的消息队列中读取消息。

消息标识符`msqid`被传递给函数，以及`rcvbuffer`——其`mesg`成员将存储读取的消息。在`msgrcv`函数成功执行后，`rcvbuffer`中的`mesg`成员将显示在屏幕上，包含来自消息队列的消息。

让我们使用 GCC 编译`messageqsend.c`程序，如下所示：

```cpp
$ gcc messageqsend.c -o messageqsend
```

如果你没有收到任何错误或警告，这意味着`messageqsend.c`程序已编译成可执行文件，名为`messageqsend.exe`。让我们运行这个可执行文件：

```cpp
$ ./messageqsend
Enter a message to add to message queue : GoodBye
The message sent is GoodBye
```

现在，按*Alt + F2*打开第二个终端屏幕。在这个屏幕上，你可以编译和运行从消息队列读取消息的脚本。

让我们使用 GCC 编译`messageqrecv.c`程序，如下所示：

```cpp
$ gcc messageqrecv.c -o messageqrecv
```

如果你没有收到任何错误或警告，这意味着`messageqrecv.c`程序已编译成可执行文件，名为`messageqrecv.exe`。让我们运行这个可执行文件：

```cpp
$ ./messageqrecv
The message received is GoodBye
```

哇！我们已经成功通过消息队列将消息从一个进程传递到另一个进程。让我们继续下一个菜谱！

# 使用共享内存进行进程间通信

在这个菜谱中，我们将学习如何使用共享内存在两个进程之间建立通信。这个菜谱分为以下部分：

+   将消息写入共享内存

+   从共享内存中读取消息

我们将从第一个开始，也就是*将消息写入共享内存*。我们在前面的菜谱中学到的函数也适用于这里。

# 将消息写入共享内存

在这个菜谱的这一部分，我们将学习如何将消息写入共享内存。

# 如何操作…

1.  通过提供文件名和 ID 调用`ftok`函数以生成 IPC 密钥。

1.  调用`shmget`函数分配与步骤 1 中生成的密钥关联的共享内存段。

1.  为所需的内存段指定的尺寸是`1024`。创建一个新的具有读写权限的内存段。

1.  将共享内存段附加到系统中的第一个可用地址。

1.  输入一个字符串，然后将其分配给共享内存段。

1.  附加的内存段将从地址空间中分离。

将数据写入共享内存的`writememory.c`程序如下所示：

```cpp
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    char *str;
    int shmid;

    key_t key = ftok("sharedmem",'a');
    if ((shmid = shmget(key, 1024,0666|IPC_CREAT)) < 0) {
        perror("shmget");
        exit(1);
    }
    if ((str = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }
    printf("Enter the string to be written in memory : ");
    gets(str);
    printf("String written in memory: %s\n",str);
    shmdt(str);
    return 0;
}
```

让我们看看幕后。

# 它是如何工作的...

通过调用`ftok`函数，我们使用文件名`sharedmem`（你可以更改此名称）和 ID 为`a`生成一个 IPC 密钥。生成的密钥被分配给键变量。之后，调用`shmget`函数来分配一个与使用`ftok`函数生成的提供的密钥相关联的共享内存段。

为所需内存段指定的尺寸是`1024`。创建一个新的具有读写权限的内存段，并将共享内存标识符分配给`shmid`变量。然后，将共享内存段连接到系统中的第一个可用地址。

一旦内存段连接到地址空间，段的开头地址就被分配给`str`变量。你将被要求输入一个字符串。你输入的字符串将通过`str`变量分配给共享内存段。最后，连接的内存段从地址空间中分离出来。

让我们继续本配方的下一部分，*从共享内存中读取消息*。

# 读取共享内存中的消息

在本部分的配方中，我们将学习如何读取写入共享内存的消息并将其显示在屏幕上。

# 如何操作...

1.  调用`ftok`函数生成一个 IPC 密钥。提供的文件名和 ID 应与写入共享内存的程序中的相同。

1.  调用`shmget`函数分配一个共享内存段。为分配的内存段指定的尺寸是`1024`，并与步骤 1 中生成的 IPC 密钥相关联。创建具有读写权限的内存段。

1.  将共享内存段连接到系统中的第一个可用地址。

1.  从共享内存段读取内容并在屏幕上显示。

1.  连接的内存段从地址空间中分离出来。

1.  从系统中删除共享内存标识符，然后销毁共享内存段。

以下是从共享内存中读取数据的`readmemory.c`程序：

```cpp
#include <stdio.h> 
#include <sys/ipc.h> 
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int shmid;
    char * str;
    key_t key = ftok("sharedmem",'a');
    if ((shmid = shmget(key, 1024,0666|IPC_CREAT)) < 0) {
        perror("shmget");
        exit(1);
    }
    if ((str = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }
    printf("Data read from memory: %s\n",str);
    shmdt(str);                
    shmctl(shmid,IPC_RMID,NULL);
    return 0;
}
```

让我们深入了解幕后。

# 它是如何工作的...

我们将调用`ftok`函数生成一个 IPC 密钥。用于生成密钥的文件名和 ID 分别是`sharedmem`（可以是任何名称）和`a`。生成的密钥被分配给`key`变量。之后，我们将调用`shmget`函数来分配一个与之前生成的密钥相关联的共享内存段。该分配的内存段尺寸为`1024`。

我们将创建一个新的具有读写权限的内存段，并将获取的共享内存标识符分配给`shmid`变量。然后，将共享内存段连接到系统中的第一个可用地址。这样做是为了我们可以通过先前的程序访问共享内存段中写入的文本。

因此，在内存段附加到地址空间之后，段的首地址被分配给了`str`变量。现在，我们可以在当前程序中通过之前的程序写入共享内存的内容。共享内存段的内容通过`str`字符串读取，并在屏幕上显示。

之后，附加的内存段从地址空间中分离出来。最后，共享内存标识符`shmid`从系统中移除，共享内存段被销毁。

让我们使用 GCC 来编译`writememory.c`程序，具体如下：

```cpp
$ gcc writememory.c -o writememory
```

如果你没有收到任何错误或警告，这意味着`writememory.c`程序已经编译成了一个可执行文件，名为`writememory.exe`。现在我们来运行这个可执行文件：

```cpp
$ ./writememory
Enter the string to be written in memory : Today it might rain
String written in memory: Today it might rain
```

现在，按*Alt + F2*打开第二个终端窗口。在这个窗口中，让我们使用 GCC 来编译`readmemory.c`程序，具体如下：

```cpp
$ gcc readmemory.c -o readmemory
```

如果你没有收到任何错误或警告，这意味着`readmemory.c`程序已经编译成了一个可执行文件，名为`readmemory.exe`。现在我们来运行这个可执行文件：

```cpp
$ ./readmemory
 Data read from memory: Today it might rain
```

哇！我们已经成功使用共享内存在不同进程之间进行了通信。
