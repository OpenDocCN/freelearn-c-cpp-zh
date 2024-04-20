# 第六章：管道、先进先出（FIFO）、消息队列和共享内存

进程之间的通信是软件系统的重要部分，选择适当的通信技术并不是一项简单的任务。开发人员在做出选择时应牢记的一个重要区别是进程是否将在同一台机器上运行。本章重点介绍了第一类，您将学习如何基于管道、**先进先出**（**FIFO**）、消息队列和共享内存开发**进程间通信**（**IPC**）解决方案。它将从第一个配方中概述四种 IPC 的特性和类型之间的区别。然后，每种类型的配方将提供实用信息，以便将它们应用到您的日常工作中。本章不包含任何特定于 C++的解决方案，以便让您熟悉 Linux 本地机制。

本章将涵盖以下主题：

+   学习不同类型的 IPC

+   学习如何使用最古老的 IPC 形式——管道

+   学习如何使用 FIFO

+   学习如何使用消息队列

+   学习如何使用共享内存

# 技术要求

为了让您立即尝试这些程序，我们设置了一个 Docker 镜像，其中包含了本书中将需要的所有工具和库。这是基于 Ubuntu 19.04 的。

为了设置它，请按照以下步骤进行：

1.  从[www.docker.com](http://www.docker.com)下载并安装 Docker Engine。

1.  通过运行以下命令从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。键入以下命令以查看镜像：`docker images`。

1.  您现在应该至少有这个镜像：`kasperondocker/system_programming_cookbook`。

1.  使用以下命令运行 Docker 镜像，获取交互式 shell 的帮助：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。键入 `root@39a5a8934370/# cd /BOOK/` 以获取所有按章节开发的程序。

需要`--cap-add sys_ptrace`参数以允许 Docker 容器中的**GNU 项目调试器**（**GDB**）设置断点，默认情况下 Docker 不允许这样做。

**免责声明**：C++20 标准已经在二月底的布拉格会议上由 WG21 批准（即技术上最终确定）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 配方代码。GCC 将最新功能的开发保留在分支中（您必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励您自行尝试。因此，请克隆并探索 GCC 合同和模块分支，并尽情玩耍。

# 学习不同类型的 IPC

本配方的目标是在同一台机器上运行的进程中提供不同 IPC 解决方案之间的指导。它将从开发人员的角度（您的角度！）提供主要特征的概述，解释它们之间的不同之处。

# 操作步骤...

以下表格显示了 Linux 机器上始终可用的四种 IPC 类型，其中列代表我们认为开发人员在进行设计选择时应考虑的独特因素：

|  | **进程关系需要？** | **需要同步？** | **通信类型** | **范围** | **涉及内核？** |
| --- | --- | --- | --- | --- | --- |
| **管道** | 是 | 通常不 | 半双工 | 同一台机器 | 是 |
| **FIFO** | 否 | 通常不 | 半双工 | 通常是同一台机器 | 是 |
| **消息队列** | 否 | 通常不 | 半双工 | 同一台机器 | 是 |
| **共享内存** | 否 | 是 | 半双工 | 同一台机器 | 是 |

表的列具有以下描述：

+   **进程之间的关系是否需要？**：这表明实现特定 IPC 是否需要进程之间的关系（例如父子关系）。

+   **需要同步？**：这表明您是否需要考虑进程之间的任何形式的同步（例如互斥锁，信号量等；参见第五章，*使用互斥锁、信号量和条件变量*）或不需要。

+   **通信类型**：两个或多个实体之间的通信可以是半双工（最接近的类比是对讲机，只有一个人可以同时说话）或全双工（例如电话，两个人可以同时交谈）。这可能对设计的解决方案产生深远影响。

+   **范围**：这表明解决方案是否可以应用于更广泛的范围，即在不同机器上的进程之间的 IPC。

+   **涉及的内核？**：这警告您有关通信过程中内核的参与。*它是如何工作...*部分将解释为什么这很重要。

在下一节中，我们将逐行分析表中突出显示的单个特征。

# 它是如何工作...

列表中的第一个 IPC 机制是**管道**。管道需要两个进程之间的关系（例如父子关系）才能工作。为了使管道对两个进程都**可见**（与 FIFO 相反），需要这种关系。这就像一个变量必须对一个方法可见才能使用一样。在管道的示例中，我们将看到这是如何在技术上工作的。

通信类型是半双工：数据从进程*A*流向进程*B*，因此不需要同步。为了在两个进程之间实现全双工通信类型，必须使用两个管道。由于两个进程必须有关系才能使用管道，管道不能用作两台不同机器上的进程之间的通信机制。Linux 内核参与通信，因为数据被复制到内核，然后进一步复制到接收进程。

表中的第二个 IPC 机制是**FIFO**（或**命名管道**）。它是命名管道，因为它需要一个路径名来创建，实际上，它是一种特殊类型的文件。这使得 FIFO 可供任何进程使用，即使它们之间没有关系。他们所需要的只是 FIFO 的路径（同样，一个文件名）所有进程都会使用。在这种情况下也不需要同步。但是，我们必须小心，因为有些情况下需要同步，正如`man page`所指定的。

POSIX.1 规定，少于`pipe_BUF`字节的写操作必须是原子的（即，输出数据被作为连续序列写入管道）。超过`pipe_BUF`字节的写操作可能是非原子的（即，内核可能会将数据与其他进程写入的数据交错）。POSIX.1 要求`pipe_BUF`至少为 512 字节（在 Linux 上，`pipe_BUF`为 4,096 字节）。精确的语义取决于文件描述符是否为非阻塞（`O_NONBLOCK`）；管道是否有多个写入者；以及要写入的字节数*n*。

一般规则是，如果你对进程之间应该发生多少数据交换有任何疑问，总是提供一个同步机制（例如互斥锁、信号量和其他许多机制）。FIFO（同样，管道）提供了半双工通信机制，除非为每个进程提供两个 FIFO（每个进程一个读取器和一个写入器）；在这种情况下，它将成为全双工通信。FIFO 通常用于同一台机器上的进程之间的 IPC，但是，由于它基于文件，如果文件对其他机器可见，FIFO 可能潜在地用于不同机器上的进程之间的 IPC。即使在这种情况下，内核也参与了 IPC，数据从内核空间复制到进程的用户空间。

**消息队列**是存储在内核中的消息的链表。这个定义已经包含了一部分信息；这是内核提供的一种通信机制，同样，这意味着数据来回从/到内核进行复制。消息队列不需要进程之间的任何关系；它们必须共享一个键才能访问相同的队列。如果消息小于或等于`pipe_BUF`，Linux 内核保证队列上的操作的原子性。在这种情况下，需要一种同步机制。消息队列不能在机器范围之外使用。

表中的最后一个 IPC 机制是**共享内存**。这是最快的 IPC 形式。这是有代价的，因为使用共享内存的进程应该使用一种同步形式（例如互斥锁或信号量），正如`man page`所建议的那样（`man shm_overview`）。

每当有一个需要保护的临界区时，进程必须使用我们在第五章中看到的机制来同步访问，*使用互斥锁、信号量和条件变量*。

进程必须在同一台机器上运行才能使用相同的共享内存，并且使用一个键进行标识，消息队列也是如此。由于共享内存位于内核空间，数据会从内核空间复制到读取和删除数据的进程中。

# 还有更多...

这四种 IPC 形式最初是在 Unix System V 上开发的，然后在更现代的 POSIX 标准中重新实现，Linux 支持这些标准。有些情况下，进程不在同一台机器上，在这种情况下，我们需要使用其他机制，比如套接字，我们将在下一章中看到。当然，套接字具有更广泛的适用性，因为它可以在网络上的任何位置将进程进行通信。

这种泛化，可以这么说，是有代价的：它们比本食谱中描述的机制慢。因此，作为开发人员，在做设计选择时必须考虑这一因素。

# 另请参阅

+   第五章*，使用互斥锁、信号量和条件变量*：关于你可以使用的同步机制。

+   第七章*，网络编程*：为了补充本章关于套接字（面向连接和无连接）的概念。

# 学习如何使用最古老的 IPC 形式-管道

在上一篇食谱中，你学会了如何根据一些关键因素选择 IPC。现在是时候动手使用四种通信类型了，这篇食谱专注于管道。在这篇食谱中，你将学习如何使用管道通过使用两个管道使两个进程进行全双工通信。我们将不使用任何形式的同步，因为通常情况下是不需要的。在*它是如何工作的...*部分，我们将看到为什么不需要以及何时不需要。

# 如何做...

在本节中，我们将开发一个程序，该程序将创建两个进程，其唯一目标是相互发送消息。正如我们所见，使用管道，数据只能单向流动。为了进行双向通信，并模拟一般情况，我们将使用两个管道：

1.  我们实例化了要发送的两条消息及其大小，稍后我们将需要它们：

```cpp
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

char* msg1 = "Message sent from Child to Parent";
char* msg2 = "Message sent from Parent to Child";
#define MSGSIZE 34
#define IN      0
#define OUT 1
```

1.  接下来，我们进入初始化部分。我们需要为接收到的消息、`childToParent`和`parentToChild`管道以及我们用于跟踪子进程的**进程标识符**（PID）实例化空间：

```cpp
int main()
{
    char inbufToParent[MSGSIZE];
    char inbufToChild[MSGSIZE];
    int childToParent[2], parentToChild[2], pid, nbytes;

    inbufToParent[0] = 0;
    inbufToChild[0] = 0;
    if (pipe(childToParent) < 0)
        return 1;

    if (pipe(parentToChild) < 0)
        return 1;
```

1.  现在，让我们看看子部分。这部分有两个部分：第一个部分是子进程向父进程发送`msg1`消息；第二个部分是子进程从父进程接收`msg2`消息：

```cpp
if ((pid = fork()) > 0)
{
        printf("Created child with PID = %d\n", pid);
        close(childToParent[IN]);
        write(childToParent[OUT], msg1, strlen(msg1));
        close(childToParent[OUT]);

        close (parentToChild[OUT]);

        read(parentToChild[IN], inbufToChild, strlen(msg2));
        printf("%s\n", inbufToChild);
        close (parentToChild[IN]);
        wait(NULL);
}
```

1.  最后，让我们看看父代码。它有两个部分：一个用于从子进程接收消息，另一个用于回复消息：

```cpp
else
{
        close (childToParent[OUT]);
        read(childToParent[IN], inbufToParent, strlen(msg1));
        printf("%s\n", inbufToParent);
        close (childToParent[IN]);

        close (parentToChild[IN]);
        write(parentToChild[OUT], msg2, strlen(msg2));
        close (parentToChild[OUT]);
}
return 0;
```

我们以编程方式实现了我们在第一章中学到的内容，即*开始系统编程*，用于 shell（参见*学习 Linux 基础知识- shell*配方）。这些步骤在下一节中详细介绍。

# 工作原理...

在第一步中，我们只是定义了`msg1`和`msg2`，供两个进程使用，并定义了`MSGSIZE`，用于读取它们所需的消息长度。

第二步基本上定义了两个管道`childToParent`和`parentToChild`，每个都是两个整数的数组。它们由`pipe`系统调用用于创建两个通信缓冲区，进程可以通过`childToParent[0]`和`childToParent[1]`文件描述符访问。消息被写入`childToParent[1]`，并且按照 FIFO 策略从`childToParent[0]`读取。为了避免缓冲区未初始化的情况，此步骤将`inbuf1`和`inbuf2`的指针设置为`0`。

第三步处理子代码。它向`childToParent[1]`写入，然后从`parentToChild[0]`读取。子进程写入`childToParent[1]`的内容可以由父进程在`childToParent[0]`上读取。`read`和`write`系统调用会导致进程进入内核模式，并临时将输入数据保存在内核空间，直到第二个进程读取它。要遵循的一个规则是未使用的管道端点必须关闭。在我们的情况下，我们写入`childToParent[1]`；因此，我们关闭了管道的`read`端`childToParent[0]`，一旦读取完毕，我们关闭了`write`端，因为它不再使用。

第四步，与第三步非常相似，具有与子代码对称的代码。它在`childToParent[0]`管道上读取，并在`parentToChild[1]`上写入，遵循相同的关闭未使用管道端点的规则。

从分析的代码来看，现在应该清楚为什么管道不能被非祖先进程使用了：`childToParent`和`parentToChild`文件描述符必须在运行时对父进程和子进程可见。

如果我们在 Docker 容器的`/BOOK/Chapter06/`文件夹中用`gcc pipe.c`编译代码并运行它，输出将如下所示：

![](img/91aaa497-4f77-4015-8784-f09f8d31dffd.png)

这表明父进程和子进程正确地发送和接收了这两条消息。

# 还有更多...

对于绝大多数用例，管道旨在与少量数据一起使用，但可能存在需要大量数据的情况。我们在本章中遵循的标准 POSIX 规定，`write`少于`pipe_BUF`字节必须是原子的。它进一步规定，`pipe_BUF`必须至少为 512 字节（在 Linux 上为 4KB）；否则，您必须通过使用信号量和互斥锁等机制在用户级别处理同步。

# 另请参阅

+   第一章，*开始系统编程*，从 shell 的角度展示了管道的概念。

+   第五章，*使用互斥锁、信号量和条件变量*具有添加同步所需的工具，以防要发送和接收的数据大于`pipe_BUF`。

# 学习如何使用 FIFO

在上一个配方中看到的管道是临时的，也就是说当没有进程打开它们时，它们就会消失。**FIFO**（也称为**命名管道**）是不同的；它们是特殊的管道，作为文件系统上的特殊文件存在。原则上，任何进程，只要有合适的权限，都可以访问 FIFO。这是 FIFO 的独特特性。使用文件允许我们编程一个更通用的通信机制，以便让进程进行通信，即使它们没有祖先关系；换句话说，我们可以使用 FIFO 让任意两个文件进行通信。在这个配方中，你将学习如何编程 FIFO。

# 如何做...

在本节中，我们将开发一个非常原始的基于 FIFO 的聊天程序，从而产生两个不同的程序，在运行时将允许两个用户进行聊天：

1.  让我们创建一个名为`fifo_chat_user1.c`的文件，并添加我们稍后需要的包含和`MAX_LENGTH`定义，以确定两个用户可以交换的消息的最大长度：

```cpp
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_LENGTH 128
```

1.  接下来，从`main`开始。在这里，我们需要定义`fd`文件描述符以打开文件；我们打算存储文件的路径；我们将用来存储`msgReceived`和`msgToSend`消息的两个字符串；最后，使用`mkfifo`系统调用在定义的路径中创建 FIFO：

```cpp
int main()
{
    char* fifoChat = "/tmp/chat";
    mkfifo(fifoChat, 0600);

    char msgReceived[MAX_LENGTH], msgToSend[MAX_LENGTH];
```

1.  现在我们需要一个无限循环来连续`write`和`read`。我们通过创建两个部分来实现：在`write`部分，我们以写模式打开`fifoChat`文件，使用`fgets`从用户获取消息，并将`msgToSend`写入由`fd`文件描述符表示的文件。在读者部分，我们以读模式打开文件，并使用`read`方法读取文件的内容，打印输出，并关闭`fd`：

```cpp
    while (1)
    {
        int fdUser1 = open(fifoChat, O_WRONLY);
        printf("User1: ");
        fgets(msgToSend, MAX_LENGTH, stdin);
        write(fdUser1, msgToSend, strlen(msgToSend)+1);
        close(fdUser1);

        int fdUser2 = open(fifoChat, O_RDONLY);
        read(fdUser2, msgReceived, sizeof(msgReceived));
        printf("User2: %s\n", msgReceived);
        close(fdUser2);
    }
    return 0;
}
```

1.  第二个程序非常相似。唯一的区别是`while`循环，它是相反的。在这里，我们有`read`部分，然后是`write`部分。你可以将`fifo_chat_user1.c`文件复制到`fifo_chat_user2.c`并进行修改，如下所示：

```cpp
while (1)
{
        int fdUser2 = open(myfifo, O_RDONLY);
        read(fdUser2, msgReceived, sizeof(msgReceived));
        printf("User1: %s\n", msgReceived);
        close(fdUser2);

        int fdUser1 = open(myfifo, O_WRONLY);
        printf("User2: ");
        fgets(msgToSend, MAX_LENGTH, stdin);
        write(fdUser1, msgToSend, strlen(msgToSend)+1);
        close(fdUser1);
}
```

尽管这不是您会在周围找到的最互动的聊天，但它绝对有助于实验 FIFO。在下一节中，我们将分析本节中所见的步骤。

# 它是如何工作的...

让我们首先编译并运行这两个程序。在这种情况下，我们希望为可执行文件提供不同的名称，以便加以区分：

```cpp
gcc fifo_chat_user1.c -o chatUser1

gcc fifo_chat_user2.c -o chatUser2
```

这将创建两个可执行文件：`chatUser1`和`chatUser2`。让我们在两个单独的终端中运行它们，并进行聊天：

![](img/7359f0c3-5c8e-4a74-95fa-74763e739dbf.png)

在*步骤 1*中，我们基本上将`MAX_LENGTH`定义为`128`字节，并添加了我们需要的定义。

在*步骤 2*中，我们创建了`mkfifo`指定路径的 FIFO，该路径指向`/tmp/chat`文件，权限为`6`（用户读写），`0`（用户所属组无读、无写、无执行权限），`0`（其他用户无读、无写、无执行权限）。这些设置可以在调用`mkfifo`后进行检查：

```cpp
root@d73a2ef8d899:/BOOK/chapter6# ls -latr /tmp/chat
prw------- 1 root root 0 Oct 1 23:40 /tmp/chat
```

在*步骤 3*中，我们使用`open`方法打开了 FIFO。值得一提的是，`open`是用于打开常规文件的相同方法，并且在返回的描述符上，我们可以调用`read`和`write`，就像在普通文件上一样。在这一步中，我们创建了一个无限循环，允许用户进行聊天。如您所见，在*步骤 4*中，`read`和`write`部分被交换，以便第二个用户在第一个用户写入时读取，反之亦然。

FIFO 由内核使用 FIFO 策略进行内部管理。每次我们从 FIFO 中`write`或`read`数据时，数据都会从内核传递到内核。您应该记住这一点。消息从`chat1`可执行文件传递，然后在内核空间中，当`chat2`程序调用`read`方法时，再次回到用户空间。

# 还有更多...

到目前为止，应该很清楚 FIFO 是一个特殊的管道。这意味着我们对管道的限制也适用于 FIFO。例如，除非发送的数据量超过了`pipe_BUF`限制，否则不需要同步，标准 POSIX 将其定义为 512 字节，Linux 将其设置为 4 KB。

要强调的另一个方面是，命名管道（FIFO）可以在*N*到*M*通信类型（即多个读取者和多个写入者）中使用。如果满足前述条件，内核将保证操作（`read`和`write`调用）的原子性。

# 另请参阅

+   第三章，*处理进程和线程*

+   第五章，*使用互斥锁、信号量和条件变量*

# 学习如何使用消息队列

POSIX 兼容操作系统（然后是 Linux 内核）直接支持的另一种机制是消息队列。消息队列本质上是存储在内核中的消息的链表，每个队列由一个 ID 标识。在这个配方中，我们将使用消息队列重写聊天程序，突出显示其主要优缺点。

# 如何做...

在本节中，我们将从*学习如何使用 FIFO*的配方中重写聊天程序。这将使您能够亲身体验 FIFO 和消息队列之间的相似之处和不同之处：

1.  创建一个名为`mq_chat_user_1.c`的新文件，并添加以下包含和定义：

```cpp
#include <stdio.h>
#include <string.h>
#include <mqueue.h>

#define MAX_MESSAGES 10
#define MAX_MSG_SIZE 256
```

1.  在`main`方法中，现在让我们定义两个消息队列描述符（`user1Desc`和`user2Desc`），以便稍后存储`mq_open`方法的结果。我们必须定义和初始化`mq_attr`结构以存储我们将创建的消息队列的配置：

```cpp
int main()
{
    mqd_t user1Desc, user2Desc;
    char message[MAX_MSG_SIZE];
    char message2[MAX_MSG_SIZE];

    struct mq_attr attr;
    attr.mq_flags = 0;
    attr.mq_maxmsg = MAX_MESSAGES;
    attr.mq_msgsize = MAX_MSG_SIZE;
    attr.mq_curmsgs = 0;
```

1.  我们可以打开两个`/user1`和`/user2`消息队列：

```cpp
    if ((user1Desc = mq_open ("/user1", O_WRONLY | O_CREAT,
         "0660", &attr)) == -1)
    {
        perror ("User1: mq_open error");
        return (1);
     }
     if ((user2Desc = mq_open ("/user2", O_RDONLY | O_CREAT,
         "0660", &attr)) == -1)
     {
         perror ("User2: mq_open error");
         return (1);
     }
```

1.  程序的核心部分是循环，用于从两个用户那里发送和接收消息。为此，我们必须：

1.  使用`mq_send`方法向用户 2 发送消息，使用`user1Desc`消息队列描述符。

1.  使用`mq_receive`从`user2Desc`消息队列描述符接收用户 2 发送给我们的消息：

```cpp
    while (1)
    {
        printf("USER 1: ");
        fgets(message, MAX_MSG_SIZE, stdin);
        if (mq_send (user1Desc, message, strlen (message)
            + 1, 0) == -1)
        {
            perror ("Not able to send message to User 2");
            continue;
        }
        if (mq_receive (user2Desc, message2, MAX_MSG_SIZE,
             NULL) == -1)
        {
            perror ("tried to receive a message from User 2
                but I've failed!");
            continue;
        }
        printf("USER 2: %s\n", message2);
    }
    return 0;
}
```

1.  我们需要另一个程序来回复给用户 1。这个程序非常相似；唯一的区别是它在`user2Desc`上发送消息（这次以写模式打开），并从`user1Desc`（以读模式打开）读取消息。

现在让我们运行程序。我们需要通过在 shell 中输入以下两个命令来编译`mq_chat_user_1.c`和`mq_chat_user_2.c`程序：

```cpp
gcc mq_chat_user_1.c -o user1 -g -lrt
gcc mq_chat_user_2.c -o user2 -g -lrt
```

我们正在编译和链接程序，并生成`user1`和`user2`可执行文件。我们已经添加了`-lrt`（这是 POSIX.1b 实时扩展库），因为我们需要包含 POSIX 消息队列实现。请记住，使用`-l`时，您正在要求编译器在链接阶段考虑特定的库。在下一节中，我们将看到输出，并分析之前看到的所有步骤。

# 它是如何工作的...

通过运行`./user1`和`./user2`可执行文件，我们将得到以下输出：

![](img/5838d556-e1d8-4538-a817-e7b1fcbe6004.png)

让我们看看以下步骤：

1.  **步骤 1**：我们需要`#include <stdio.h>`进行用户输入/输出，`#include <string.h>`通过`strlen`获取字符串的长度，以及`#include <mqueue.h>`以访问消息队列接口。在这一步中，我们已经定义了队列中的最大消息数（`10`）和队列中消息的最大大小（`256`字节）。

1.  **步骤 2**：在程序的`main`方法中，我们定义了两个消息队列描述符（`user1Desc`和`user2Desc`）来保持对消息队列的引用；两个消息数组（`message`和`message2`）用于在两个用户之间存储要发送和接收的消息；最后，我们定义并初始化了`struct mq_attr`结构，用于初始化我们将在下一步中使用的消息队列。

1.  **步骤 3**：在这一步中，我们已经打开了两个消息队列。它们分别是`/user1`和`/user2`，位于`/dev/mqueue`中：

```cpp
root@1f5b72ed6e7f:/BOOK/chapter6# ll /dev/mqueue/user*
------x--- 1 root root 80 Oct 7 13:11 /dev/mqueue/user1*
------x--- 1 root root 80 Oct 7 13:11 /dev/mqueue/user2*
```

`mq_chat_user_1.c`以只写模式打开`/user1`消息队列，并在不存在时创建它。它还以只读模式打开`/user2`，并在不存在时创建它。应该清楚的是，如果当前进程没有消息队列的访问权限（我们以`660`打开），`mq_open`将失败。

1.  **步骤 4**：这一步包含了程序的主要逻辑。它有一个无限循环，从用户 1 发送消息到用户 2，然后从用户 2 接收到用户 1。发送消息所使用的方法是`mq_send`。它需要消息队列描述符、要发送的消息、消息的长度（`+1`，因为我们需要包括终止符）以及消息的优先级（在这种情况下我们没有使用）。`mq_send`（参见`man mq_send`了解更多信息）如果队列中没有足够的空间，会阻塞直到有足够的空间为止。

发送完毕后，我们调用`mq_receive`方法（参见`man mq_receive`了解更多信息）来从用户 2 获取可能的消息。它需要消息队列描述符、将包含消息的数组、我们可以接收的最大大小以及优先级。请记住，如果队列中没有消息，`mq_receive`会阻塞。

有关更多信息，请参阅`man mq_receive`页面。

由于发送和接收是核心概念，让我们通过一个示意图来更深入地分析它们：

![](img/6f49dbd8-e83c-495b-b7b7-fb2546928205.png)

**(1)** 在这种情况下，用户 1 进程调用`mq_send`。Linux 内核会将要发送的消息从用户空间复制到内核空间。在**(3)**中也是同样的情况。

**(2)** 当用户 2 进程在相同的消息队列（`user1Desc`）上调用`mq_receive`时，Linux 内核会将消息从内核空间复制到用户空间，将数据复制到`message2`缓冲区中。在**(4)**中也是同样的情况。

# 还有更多...

可能会有情况需要根据优先级从队列中获取消息，这在这种情况下我们没有使用。您能修改这个示例程序以包括优先级吗？您需要修改什么？

您可能已经注意到，我们在这个示例中使用了`perror`方法。`perror`方法会在标准输出中打印出最后一个错误（`errno`），以描述性格式出现。开发者的优势在于不必显式地获取`errno`值并将其转换为字符串；这一切都会自动完成。

对于消息队列，我们描述管道和 FIFO 的原子性概念也是适用的。如果消息小于`pipe_BUF`，则消息的传递是保证原子性的。否则，开发者必须提供同步机制。

# 另请参阅

在第三章的示例中，*处理进程和线程*（关于线程）和第五章的示例中，*使用互斥锁、信号量和条件变量*（关于同步）。通常情况下，`man`页面提供了丰富的信息源，建议的起点是`man mq_overview`。 

# 学习如何使用共享内存

在我们迄今为止看到的所有 IPC 机制中，内核在进程之间的通信中起着积极的作用，正如我们所学到的那样。信息确实是从 Linux 内核流向进程，反之亦然。在本示例中，我们将学习最快的进程间通信形式，它不需要内核作为进程之间的中介。尽管 System V API 是广泛可用的，但我们将使用最新的、更简单、设计更好的 POSIX API。我们将使用共享内存重写我们的聊天应用程序，并深入研究它。

# 如何做...

在本节中，我们将重点介绍使用 POSIX 共享内存 API 开发简单的聊天应用程序。由于内核不直接参与通信过程，我们需要提供同步机制来保护关键部分（共享内存）免受两个进程的读写：

1.  让我们首先添加我们需要的包含和定义。我们将有两个共享内存空间（`STORAGE_ID1`和`STORAGE_ID2`）来实现进程之间的双向通信：

```cpp
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define STORAGE_ID1 "/SHM_USER1"
#define STORAGE_ID2 "/SHM_USER2"
#define STORAGE_SIZE 32
```

1.  在`main`方法中，我们需要两个数组来存储发送和接收的消息。此外，我们需要以读写模式打开两个共享内存空间，并且如果不存在则创建，并且标志指示文件所有者的读写权限（分别为`S_IRUSR`和`S_IWUSR`）：

```cpp
int main(int argc, char *argv[])
{
    char message1[STORAGE_SIZE];
    char message2[STORAGE_SIZE];

    int fd1 = shm_open(STORAGE_ID1, O_RDWR | O_CREAT, S_IRUSR | 
        S_IWUSR);
    int fd2 = shm_open(STORAGE_ID2, O_RDWR | O_CREAT, S_IRUSR | 
        S_IWUSR);
    if ((fd1 == -1) || (fd2 == -1))
    {
        perror("open");
        return 10;
    }
```

1.  由于共享内存基于`mmap`（我们实质上将文件映射到内存的一部分），我们需要扩展文件描述符 1（`fd1`）指向的文件到我们需要的大小`STORAGE_SIZE`。然后，我们需要将两个文件描述符映射到共享模式（`MAP_SHARED`）的一部分内存，并且当然，要检查错误：

```cpp
    // extend shared memory object as by default it's initialized 
    //  with size 0
    int res1 = ftruncate(fd1, STORAGE_SIZE);
    if (res1 == -1)
    {
        perror("ftruncate");
        return 20;
    }

    // map shared memory to process address space
    void *addr1 = mmap(NULL, STORAGE_SIZE, PROT_WRITE, MAP_SHARED, 
        fd1, 0);
    void *addr2 = mmap(NULL, STORAGE_SIZE, PROT_WRITE, MAP_SHARED, 
        fd2, 0);
    if ((addr1 == MAP_FAILED) || (addr2 == MAP_FAILED))
    {
        perror("mmap");
        return 30;
    }
```

1.  在`main`循环中，与前两个示例一样，我们在两个共享内存实例中进行`read`和`write`操作：

```cpp
    while (1)
    {
        printf("USER 1: ");
        fgets(message1, STORAGE_SIZE, stdin);
        int len = strlen(message1) + 1;
        memcpy(addr1, message1, len);

        printf("USER 2 (enter to get the message):"); getchar();
        memcpy(message2, addr2, STORAGE_SIZE);
        printf("%s\n", message2);
    }

    return 0;
}
```

1.  第二个程序与此程序相似。您可以在`/BOOK/Chapter06`文件夹中找到它们：`shm_chat_user1.c`（我们描述的那个）和`shm_chat_user2.c`。

让我们通过在 shell 上输入以下两个命令来编译和链接两个`shm_chat_user1.c`和`shm_chat_user2.c`程序：

```cpp
gcc shm_chat_user1.c -o user1 -g -lrt
gcc shm_chat_user2.c -o user2 -g -lrt
```

输出将是两个二进制文件：`user1`和`user2`。在这种情况下，我们也添加了`-lrt`，因为我们需要包含 POSIX 共享内存实现（如果没有它，链接阶段将抛出`undefined reference to 'shm_open'`错误）。在下一节中，我们将分析本节中所见的所有步骤。

# 它是如何工作的...

运行`./user1`和`./user2`程序将产生以下交互：

![](img/4d2095df-4516-4651-bdfa-36f932343e57.png)

让我们按照以下步骤进行：

+   **步骤 1**：第一步只包括我们需要的一些头文件：`stdio.h`用于标准输入/输出（例如`perror`，`printf`等）；`mman.h`用于共享内存 API；`mmap`和`fcntl.h`用于`shm_open`标志（例如`O_CREAT`，`O_RDWR`等）；`unistd.h`用于`ftruncate`方法；`string.h`用于`strlen`和`memcpy`方法。

我们定义了`STORAGE_ID1`和`STORAGE_ID2`来标识两个共享内存对象，它们将在`/dev/shm`文件夹中可用：

```cpp
root@1f5b72ed6e7f:/BOOK/chapter6# ll /dev/shm/SHM_USER*
-rw------- 1 root root 32 Oct 7 23:26 /dev/shm/SHM_USER1
-rw------- 1 root root 0 Oct 7 23:26 /dev/shm/SHM_USER2
```

+   **步骤 2**：在这一步中，我们在堆栈上为两条消息（`message1`和`message2`）分配了空间，我们将使用它们在进程之间发送和接收消息。然后，我们创建并打开了两个新的共享内存对象，并检查是否有任何错误。

+   **步骤 3**：一旦两个共享内存对象可用，我们需要扩展两个文件（通过两个文件描述符`fd1`和`fd2`，每个程序一个）并且非常重要的是将`fd1`和`fd2`映射到当前进程的虚拟地址空间。

+   第 4 步：这一步是程序的核心部分。在这里，有一些有趣的事情需要注意。首先，我们可以看到，与 FIFO、管道和消息队列不同，这里没有数据在用户空间和内核空间之间的移动。我们只是在本地缓冲区（在堆栈上分配）和我们映射的内存之间进行内存复制，反之亦然。第二个因素是，由于我们只处理内存复制，性能将优于其他 IPC 机制。

这一步的机制非常简单：我们要求用户输入一条消息并将其存储在`message1`缓冲区中，然后将缓冲区复制到内存映射地址`addr1`。读取部分（我们从第二个用户那里读取消息的地方）也很简单：我们将消息从内存复制到本地缓冲区`message2`。

# 还有更多...

正如您所看到的，这个配方中两个进程之间没有同步。这是为了让您只关注一个方面：与共享内存的通信。读者再次被邀请改进此代码，通过使用线程使其更加交互，并通过使用同步机制使其更加安全。

自 2.6.19 内核以来，Linux 支持使用访问控制列表（ACL）来控制虚拟文件系统中对象的权限。有关更多信息，请参阅`man acl`。

# 另请参阅

关于线程和同步的配方：

+   第三章，处理进程和线程

+   第五章，使用互斥锁、信号量和条件变量
