# 管理信号

信号是软件中断。它们提供了一种管理异步事件的方式，例如，来自终端的用户键入中断键或另一个进程发送必须被管理的信号。每个信号都有一个以`SIG`开头的名称（例如，`SIGABRT`）。本章将教您如何编写代码来正确管理软件中断，Linux 为每个信号定义的默认操作是什么，以及如何覆盖它们。

本章将涵盖以下配方：

+   学习所有信号及其默认操作

+   学习如何忽略信号

+   学习如何捕获信号

+   学习如何向另一个进程发送信号

# 技术要求

为了让您立即尝试本章中的程序，我们设置了一个 Docker 镜像，其中包含了本书中需要的所有工具和库，它基于 Ubuntu 19.04。

为了设置它，请按照以下步骤进行操作：

1.  从[www.docker.com](http://www.docker.com)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。输入以下命令查看镜像：`docker images`。

1.  现在您应该至少有这个镜像：`kasperondocker/system_programming_cookbook`。

1.  使用以下命令运行带有交互式 shell 的 Docker 镜像：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。使用`root@39a5a8934370/# cd /BOOK/`来获取按章节开发的所有程序。

`--cap-add sys_ptrace`参数是必要的，以允许 Docker 容器中的 GDB 设置断点，默认情况下 Docker 不允许。

# 学习所有信号及其默认操作

本配方将向您展示 Linux 支持的所有信号及相关默认操作。我们还将了解为什么信号是一个重要的概念，以及 Linux 为软件中断做了什么。

# 如何做...

在本节中，我们将列出我们的 Linux 发行版支持的所有信号，以便能够在*工作原理...*部分描述最常见的信号。

在 shell 上，输入以下命令：

```cpp
root@fefe04587d4e:/# kill -l
```

如果在基于 Ubuntu 19.04 发行版的书籍的 Docker 镜像上运行此命令，您将获得以下输出：

```cpp
 1) SIGHUP 2) SIGINT 3) SIGQUIT 4) SIGILL 5) SIGTRAP
 6) SIGABRT 7) SIGBUS 8) SIGFPE 9) SIGKILL 10) SIGUSR1
11) SIGSEGV 12) SIGUSR2 13) SIGPIPE 14) SIGALRM 15) SIGTERM
16) SIGSTKFLT 17) SIGCHLD 18) SIGCONT 19) SIGSTOP 20) SIGTSTP
21) SIGTTIN 22) SIGTTOU 23) SIGURG 24) SIGXCPU 25) SIGXFSZ
26) SIGVTALRM 27) SIGPROF 28) SIGWINCH 29) SIGIO 30) SIGPWR
31) SIGSYS 34) SIGRTMIN 35) SIGRTMIN+1 36) SIGRTMIN+2 37) SIGRTMIN+3
38) SIGRTMIN+4 39) SIGRTMIN+5 40) SIGRTMIN+6 41) SIGRTMIN+7 42) SIGRTMIN+8
43) SIGRTMIN+9 44) SIGRTMIN+10 45) SIGRTMIN+11 46) SIGRTMIN+12 47) SIGRTMIN+13
48) SIGRTMIN+14 49) SIGRTMIN+15 50) SIGRTMAX-14 51) SIGRTMAX-13 52) SIGRTMAX-12
53) SIGRTMAX-11 54) SIGRTMAX-10 55) SIGRTMAX-9 56) SIGRTMAX-8 57) SIGRTMAX-7
58) SIGRTMAX-6 59) SIGRTMAX-5 60) SIGRTMAX-4 61) SIGRTMAX-3 62) SIGRTMAX-2
63) SIGRTMAX-1 64) SIGRTMAX

```

在下一节中，我们将学习进程可以接收的最常见信号的默认操作，以及每个信号的描述，以及 Linux 如何管理这些软件中断。

# 工作原理...

在*步骤 1*中，我们执行了`kill -l`命令来获取当前 Linux 发行版支持的所有信号。以下表格提供了最常见信号的默认操作和描述：

| **信号** | **描述** | **默认操作** |
| --- | --- | --- |
| `SIGHUP` | 控制进程的终端被关闭（例如，用户注销？） | 终止 |
| `SIGABRT` | 由`abort()`发送的信号 | 终止（如果可能，带有核心转储） |
| `SIGSEGV` | 无效的内存引用 | 终止（如果可能，带有核心转储） |
| `SIGSYS` | 错误的系统调用或进程尝试执行无效的系统调用。 | 终止（如果可能，带有核心转储） |
| `SIGINT` | 从键盘生成的中断（例如*Ctrl* + *C*） | 终止 |
| `SIGQUIT` | 从键盘生成的退出（例如：*Ctrl* + */*） | 终止（如果可能，带有核心转储） |
| `SIGPIPE` | 进程尝试向管道写入但没有读取器 | 终止 |
| `SIGILL` | 进程尝试执行非法指令 | 终止（如果可能，带有核心转储） |
| `SIGALRM` | 由`alarm()`发送的信号 | 终止 |
| `SIGSTOP` | 停止进程 | 停止进程 |
| `SIGIO` | 异步 I/O 事件 | 终止 |
| `SIGTRAP` | 断点被捕获 | 终止 |
| `SIGTERM` | 终止信号（可捕获） | 终止 |
| `SIGKILL` | 进程终止（无法捕获） | 终止 |

对于发送到进程的每个信号，Linux 都会应用其默认操作。系统开发人员当然可以通过在进程内实现所需的操作来覆盖此操作，正如我们将在*学习如何捕获信号*食谱中看到的那样。

信号在`<signal.h>`头文件中定义，它们只是带有有意义名称的正整数，始终以`SIG`开头。当信号（即软件中断）被引发时，Linux 会做什么？简单来说，它总是应用相同的顺序生命周期，如下所示：

1.  信号是由另一个进程的用户或 Linux 本身发出的。

1.  信号被存储，直到 Linux 能够传递它。

1.  一旦传递，Linux 将执行这些特定操作之一：

1.  忽略信号：我们已经看到有些信号是不能被忽略的（例如`SIGKILL`）。

1.  执行默认操作：您可以参考前表的第 3 列。

1.  处理注册函数的信号（由系统开发人员实现）。

# 还有更多...

在`<signal.h>`头文件中描述和定义的所有信号都符合 POSIX 标准。这意味着每个标识符、它们的名称和默认操作都是由 POSIX.1-2003 标准定义的，Linux 遵循这一标准。这保证了应用程序中`signals`实现或支持的可移植性。

# 另请参阅

+   *学习如何捕获信号*食谱

+   *学习如何忽略信号*食谱

+   *学习如何向另一个进程发送信号*食谱

+   第三章，*处理进程和线程*，以便重新了解进程和线程。

# 学习如何忽略信号

可能会有一些情况，我们只需要忽略特定的信号。但请放心，有一些信号是不能被忽略的，例如`SIGKILL`（无法捕获）。这个食谱将教你如何忽略一个可捕获的信号。

# 如何做到...

要忽略一个可捕获的信号，请按照以下步骤：

1.  在 shell 上，打开一个名为`signal_ignore.cpp`的新源文件，并开始添加以下代码：

```cpp
#include<stdio.h>
#include<signal.h>
#include <iostream>

int main()
{
    std::cout << "Starting ..." << std::endl;
    signal(SIGTERM, SIG_IGN);
    while (true) ;
    std::cout << "Ending ..." << std::endl;
    return 0;
}
```

1.  在这个第二个程序（`signal_uncatchable.cpp`）中，我们想要看到一个*无法捕获*的信号无法被*忽略*。为了做到这一点，我们将使用在*学习所有信号及其默认操作*食谱中看到的`SIGKILL`信号，这是不可捕获的（也就是说，程序无法忽略它）：

```cpp
#include<stdio.h>
#include<signal.h>
#include <iostream>

int main()
{
    std::cout << "Starting ..." << std::endl;
    signal(SIGKILL, SIG_IGN);
    while (true) ;
    std::cout << "Ending ..." << std::endl;
    return 0;
}
```

下一节将解释前两个程序的细节。

# 它是如何工作的...

*步骤 1*包含忽略`SIGTERM`信号的程序。我们通过调用`signal();`系统调用，将特定信号作为第一个参数（`SIGTERM`）传递，并将要执行的操作作为第二个参数，这种情况下是`SIG_IGN`，即忽略。

*步骤 2*与*步骤 1*具有相同的代码。我们只是使用了`signal();`方法传递了`SIGKILL`参数和`SIG_IGN`。换句话说，我们要求 Linux 忽略此进程的`SIGKILL`信号（一旦构建和执行，`signal_uncatchable.cpp`将成为一个进程）。正如我们在*学习所有信号及其默认操作*食谱中学到的，`SIGKILL`是一个无法捕获的信号。

现在让我们构建和运行这两个程序。我们期望看到的是第一个程序中忽略的`SIGTERM`信号，以及第二个程序中无法忽略的`SIGKILL`信号。第一个程序的输出如下：

![](img/9d9e43c4-0a85-492b-ad1d-5bcd9eddc98d.png)

在这里，我们使用`ps aux`检索了进程的`PID`，并通过运行命令`kill -15 115`发送了`SIGTERM`信号（其中`15`代表`SIGKILL`）。正如你所看到的，该进程通过完全忽略终止信号而继续运行。

第二个程序`signal_uncatchable.cpp`显示，即使我们指定捕获`SIGKILL`信号，Linux 也会忽略这一点并且无论如何杀死我们的进程。我们可以在以下截图中看到这一点：

![](img/2ff0fb88-d0b7-463d-a3b6-c1a69327f748.png)

# 还有更多...

要获得 Linux 机器支持的所有信号的列表，`kill -l`命令非常有帮助，`man signal`包含了您成功将信号集成到程序中所需的所有细节。

# 另请参阅

+   在第一章的*学习 Linux 基础知识- shell*食谱中，了解如何在 shell 上运行程序

+   *学习如何捕获信号*食谱

+   *学习如何向另一个进程发送信号*食谱

+   *学习所有信号及其默认操作*食谱

+   在第三章*，处理进程和线程*，了解进程和线程的相关知识

# 学习如何捕获信号

这个食谱将教你如何在程序中捕获（或捕获）信号。可能需要执行一些特定信号的操作。例如，当应用程序接收到终止信号（`SIGTERM`）时，但我们需要在退出之前清理一些已使用的资源时。

# 如何做...

让我们编写一个应用程序，在其中我们将捕获`SIGTERM`信号，打印一个字符串，并终止应用程序：

1.  在 shell 上，创建一个名为`signal_trap.cpp`的新文件。我们需要包括`<signal.h>`等其他头文件，以便能够处理信号。我们还必须添加所需的原型来管理我们想要捕获的信号。然后，在`main`方法中，我们通过传递我们想要捕获的`SIGTERM`和用于管理它的方法来调用`signal()`系统调用：

```cpp
#include<stdio.h>
#include<signal.h>
#include <iostream>

void handleSigTerm (int sig);

int main()
{
    std::cout << "Starting ..." << std::endl;
    signal(SIGTERM, handleSigTerm);
    while (true);
    std::cout << "Ending ..." << std::endl;
    return 0;
}
```

1.  我们需要定义`handleSigTerm()`方法（可以随意命名）：

```cpp
void handleSigTerm (int sig)
{
    std::cout << "Just got " << sig << " signal" << std::endl;
    std::cout << "cleaning up some used resources ..."
        << std::endl;
    abort();
}
```

下一节将详细描述该程序。

# 工作原理...

*步骤 1*基本上定义了`main`方法。首先，我们需要`<signal.h>`头文件。在`main`方法的定义中，`signal()`系统调用的核心部分是我们传递的`SIGTERM`信号和我们希望 Linux 调用的方法。这是一个值得强调的重要方面。`signal()`系统调用接受（作为第二个参数）一个指向系统开发人员必须定义的函数的指针，就像我们所做的那样。在内核中，当引发软件中断时，Linux 将其发送到特定进程，并将调用该方法（以回调形式）。`signal()`方法的原型如下：

```cpp
void(*signal(int, void (*)(int)))(int);
```

*步骤 2*中有定义将管理我们想要捕获的`SIGTERM`信号的方法。这种方法在其简单性中显示了一些有趣的事情。首先，这个方法是从`signal()`系统调用中调用的回调。其次，我们必须定义其原型为`void (*)(int)`，即返回 void 并接受输入中的整数（表示应用程序实际接收的信号）。与此原型不同的任何内容都将导致编译错误。

让我们现在构建并执行我们在上一节中开发的程序：

![](img/6f6b91f1-1a20-4bfb-b77b-33756494b1f5.png)

我们构建并链接了`signal_trap.cpp`程序，并生成了`a.out`可执行文件。我们运行它；与该进程关联的 PID 为`46`。在右侧 shell 上，我们向 PID 为`46`的进程发送`SIGTERM`信号（带有标识符=`15`）。正如您在标准输出（左侧的 shell）上看到的，该进程捕获了信号并调用了我们定义的方法`handleSigTerm()`。这种方法在标准输出中打印了一些日志，并调用了`abort()`系统调用，该系统调用向正在运行的进程发送`SIGABORT`信号。正如您在*学习所有信号及其默认操作*食谱中看到的，`SIGABORT`的默认操作是终止进程（并生成核心转储）。当然，您可以根据您的要求（例如`exit()`）以另一种更合适的方式玩耍并终止进程。

# 还有更多...

那么，当一个进程分叉（或执行）另一个进程时，信号会发生什么？以下表格将帮助您了解如何处理进程与子进程之间的信号关系：

| **信号行为** | **进程分叉** | **进程执行** |
| --- | --- | --- |
| 默认 | 继承 | 继承 |
| 忽略 | 继承 | 继承 |
| 处理 | 继承 | 未继承 |

在这个阶段，当一个进程分叉另一个进程时，当一个进程执行另一个任务（使用`exec`）时，它继承了父进程的所有行为。当一个进程执行另一个任务（使用`exec`）时，它继承了**默认行为**和**忽略行为**，但没有继承已实现的处理方法。

# 另请参阅

+   *学习如何忽略信号* 配方

+   *学习所有信号及其默认操作* 配方

+   *学习如何向另一个进程发送信号* 配方

+   第三章，*处理进程和线程*，了解进程和线程的相关知识

# 学习如何向另一个进程发送信号

可能存在情况，一个进程需要向其他进程发送信号。这个配方将教你如何使用实际操作来实现这一点。

# 如何做...

我们将编写一个程序，将`SIGTERM`信号发送给正在运行的进程。我们将看到进程按预期终止。在 shell 上，打开一个名为`signal_send.cpp`的新源文件。我们将使用系统调用`kill()`，它向由`pid`指定的进程发送信号`sig`。该程序接受一个输入参数，即要终止的程序的`pid`：

```cpp
#include<stdio.h>
#include<signal.h>
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "Starting ..." << std::endl;
    if (argc <= 1)
    {
       std::cout << "Process pid missing ..." << std::endl;
       return 1;
    }
    int pid = std::atoi(argv[1]);
    kill (pid, SIGTERM);

    std::cout << "Ending ..." << std::endl;
    return 0;
}
```

我们将使用在*学习如何捕获信号* 配方中开发的`signal_trap.cpp`程序作为要终止的进程。下一节将深入介绍此处所见代码的细节。

# 工作原理...

为了看到正确的行为，我们需要运行一个我们打算终止的进程。我们将运行`signal_trap.cpp`程序。让我们构建并运行`signal_send.cpp`程序，如下所示：

![](img/90e6729d-3779-4be7-a592-193d458d00a3.png)

在这里，我们执行了一些事情，如下所示：

1.  我们已经构建了`signal_trap.cpp`程序并生成了`a.out`可执行文件。

1.  运行`./a.out`。

1.  在左侧的 shell 上，我们获取了`a.out`进程的`pid`，为`133`。

1.  我们已经构建了`signal_send.cpp`程序到`terminate`可执行文件。

1.  我们使用`./terminate 133`运行`./terminate`，其中`pid`变量是我们想要终止的`a.out`进程的`pid`。

1.  在右侧的 shell 上，我们可以看到`a.out`进程正确终止。

*步骤 1*有一些事情需要解释。首先，我们从命令行参数中解析了`pid`变量，将其转换为整数，然后将其保存到`pid`变量中。其次，我们通过传递`pid`变量和我们要发送给运行中进程的`SIGTERM`信号来调用`kill()`系统调用。

`man 2 kill`: `int kill(pid_t pid, int sig);`

`kill()`函数将指定的信号`sig`发送给`pid`。

为了与 System V 兼容，如果 PID 为负数（但不是`-1`），则信号将发送到其进程组 ID 等于进程编号绝对值的所有进程。但是，如果`pid`为 0，则`sig`将发送到**调用进程**的进程组中的每个进程。

# 还有更多...

为了向另一个进程（或进程）发送信号，发送进程必须具有适当的特权。简而言之，如果当前用户拥有它，进程可以向另一个进程发送信号。

可能存在情况，一个进程必须向自身发送信号。在这种情况下，系统调用`raise()`可以完成这项工作：

```cpp
int raise (int signo);
```

最后一个非常重要的事情：处理引发的信号的处理程序代码必须是可重入的。其背后的原理是进程可能处于任何处理过程中，因此处理程序在修改任何静态或全局数据时必须非常小心。如果操作的数据分配在堆栈上或者作为输入传递，则函数是**可重入**的。

# 另请参阅

+   *学习如何捕获信号*食谱

+   *学习如何忽略信号*食谱

+   *学习所有信号及其默认操作*食谱
