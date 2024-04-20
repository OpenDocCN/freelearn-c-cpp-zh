# 处理进程和线程

进程和线程是任何计算的基础。一个程序很少只由一个线程或进程组成。在本章中，你将学习处理线程和进程的基本示例。你还将学习处理线程相对于**可移植操作系统接口**（**POSIX**）来说是多么容易和方便。学习这些技能是作为系统开发人员核心技能的重要部分。C++标准库中没有*进程*的概念，因此将使用 Linux 本地实现。

本章将涵盖以下示例：

+   启动一个新进程

+   杀死一个进程

+   创建一个新线程

+   创建一个守护进程

# 技术要求

为了让你立即尝试这些程序，我们已经设置了一个 Docker 镜像，其中包含了本书中需要的所有工具和库。这是基于 Ubuntu 19.04 的。

为了设置它，按照以下步骤：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  通过运行以下命令从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。输入以下命令查看镜像：`docker images`。

1.  现在你应该至少有这个镜像：`kasperondocker/system_programming_cookbook`。

1.  使用以下命令以交互式 shell 运行 Docker 镜像：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。输入`root@39a5a8934370/# cd /BOOK/`以获取所有按章节开发的程序。

需要`--cap-add sys_ptrace`参数来允许 Docker 容器中的**GNU 项目调试器**（**GDB**）设置断点，默认情况下 Docker 不允许。

**免责声明**：C++20 标准已经在二月底的布拉格会议上由 WG21 批准（即技术上完成）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 示例代码。GCC 将最新功能的开发保留在分支中（你必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励你自己尝试。所以，克隆并探索 GCC 合同和模块分支，玩得开心。

# 启动一个新进程

这个示例将展示如何通过程序启动一个新的进程。C++标准不包括对进程的任何支持，因此将使用 Linux 本地实现。能够在程序中管理进程是一项重要的技能，这个示例将教会你进程的基本概念，**进程标识符**（**PID**），父 PID 和所需的系统调用。

# 如何做...

这个示例将展示如何启动一个子进程，以及如何通过使用 Linux 系统调用使父进程等待子进程完成。将展示两种不同的技术：第一种是父进程只 fork 子进程；第二种是子进程使用`execl`系统调用运行一个应用程序。

系统调用的另一种选择是使用外部库（或框架），比如**Boost**库。

1.  首先，在一个名为`process_01.cpp`的新文件中输入程序：

```cpp
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>

int main(void)
{
    pid_t child;
    int status;
    std::cout << "I am the parent, my PID is " << getpid()
        << std::endl;
    std::cout << "My parent's PID is " << getppid() << std::endl;
    std::cout << "I am going to create a new process..."
        << std::endl;
    child = fork();
    if (child == -1)
    {
```

1.  我们必须考虑一个子进程可能没有被 fork 的情况，所以我们需要写这部分：

```cpp
        // fork() returns -1 on failure
        std::cout << "fork() failed." << std::endl;
        return (-1);
    }
    else if (child == 0)
    {
```

1.  这个分支是一个快乐的情况，父进程可以正确地 fork 它的子进程。这里的子进程只是将它的 PID 打印到标准输出：

```cpp
      std::cout << "I am the child, my PID is " << std::endl;
      std::cout << "My parent's PID is " << getppid() << std::endl;
    }
    else
    {
```

1.  现在，我们必须让父进程等待子进程完成：

```cpp
        wait(&status); // wait for the child process to finish...
        std::cout << "I am the parent, my PID is still "
            << getpid() << std::endl;
    }
    return (0);
}
```

现在，让我们开发前一个程序的`fork-exec`版本。

1.  首先，在一个名为`process_02.cpp`的新文件中输入程序：

```cpp
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>

int main(void)
{
    pxid_t child;
    int status;
    std::cout << "I am the parent, my PID is " 
              << getpid() << std::endl;
    std::cout << "My parent's PID is " 
              << getppid() << std::endl;
    std::cout << "I am going to create a new process..." 
              << std::endl;
    child = fork();
    if (child == -1)
    {
        // fork() returns -1 on failure
        std::cout << "fork() failed." << std::endl;
        return 1;
    }
    else if (child == 0)
    {
```

1.  以下代码块显示了使用`execl`*运行`ls -l`的子部分：*

```cpp
        if (execl("/usr/bin/ls", "ls", "-l", NULL) < 0) 
        {
            std::cout << "execl failed!" << std::endl;
            return 2;
        }
        std::cout << "I am the child, my PID is " 
                  << getpid() << std::endl;
        std::cout << "My parent's PID is " 
                  << getppid() << std::endl;
    }
    else
    {
        wait(&status); // wait for the child process to finish...
    }
    return (0);
}
```

下一节将描述两种不同方法（`fork`与`fork-exec`）的详细信息。

# 它是如何工作的...

让我们分析前面的两个例子：

1.  `fork`系统调用：通过编译`g++ process_01.cpp`并运行`./a.out`，输出将如下所示：

![](img/ee5c3fb9-61b9-4ed8-a7c1-ba0ba262c4d8.png)

通过调用`fork`，程序创建了调用进程的副本。这意味着这两个进程具有相同的代码，尽管它们是两个完全不同的进程，但代码库将是相同的。用户必须在`else if (child == 0)`部分中挂接子代码。最终，父进程将不得不等待子进程完成任务，使用`wait(&status);`调用。另一种选择是`waitpid (123, &status, WNOHANG);`调用，它等待特定的 PID（或者如果第一个参数是`-1`，则等待所有子进程）。`WNOHANG`使`waitpid`立即返回，即使子进程的状态不可用。

如果父进程不等待子进程完成会发生什么？也就是说，如果没有`wait(&status);`调用会发生什么？从技术上讲，父进程将完成，而仍在运行的子进程将成为**僵尸**。这在 Linux 内核 2.6 版本之前是一个巨大的问题，因为僵尸进程会一直停留在系统中，直到它们被*等待*。子进程现在由`init`进程（其 PID 为`1`）接管，后者定期等待可能会死亡的子进程。

1.  `fork-exec`系统调用：

![](img/ba51686a-cefb-4c91-9f71-5b4a93a1fc55.png)

创建进程的最常见方法是`fork`/`exec`组合。正如我们所见，`fork`创建一个完全新的进程，具有自己的 PID，但现在，`else if (child == 0)`部分执行一个外部进程，该进程具有不同的代码库。这个例子只是调用`ls -l`命令来列出文件和目录，但开发人员可以在这里放置任何可执行文件。

# 还有更多...

为什么应该使用进程而不是线程是一个重要的方面需要考虑。答案取决于情况，但一般来说，应该考虑以下方面：

+   线程在启动它的进程的相同内存空间中运行。这一方面既有利也有弊。主要的含义是，如果一个线程崩溃，整个应用程序都会崩溃。

+   线程之间的通信比进程间通信要快得多。

+   一个进程可以通过`setrlimit`以较低的权限生成，以限制不受信任的代码可用的资源。

+   在进程中设计的程序比在线程中设计的程序更分离。

在这个步骤中看到的`fork`/`execl`/`wait`调用有许多变体。`man pages`提供了对整个调用系列的全面文档。以下屏幕截图是关于`man execl`的：

![](img/67bf3f79-e515-41b1-9a1a-779d804c909d.png)

# 另请参阅

请参阅第一章，*开始系统编程*，以便了解`man pages`和 Linux 的基础知识。

# 杀死一个进程

在上一个步骤中，我们已经看到了启动新进程的两种方式，其中父进程总是等待子进程完成任务。这并不总是这样。有时，父进程应该能够杀死子进程。在这个步骤中，我们将看到如何做到这一点的一个例子。

# 做好准备

作为先决条件，重要的是要通过*启动新进程*的步骤。

# 如何做...

在这一部分，我们创建一个程序，其中父进程 fork 其子进程，子进程将执行一个无限循环，父进程将杀死它：

1.  让我们开发将被父进程杀死的子程序：

```cpp
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>

int main(void)
{
    std::cout << "Running child ..." << std::endl;
    while (true)
        ;
}
```

1.  接下来，我们必须开发父程序（`/BOOK/Chapter03`文件夹中的`process_03.cpp`）：

```cpp
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
int main(void)
{
    pid_t child;
    int status;
    std::cout << "I am the parent, my PID is " << getpid() 
              << std::endl;
    child = fork();
    std::cout << "Forked a child process with PID = " 
              << child << std::endl;
    if (child == -1)
    {
        std::cout << "fork() failed." << std::endl;
        return 1;
    }
    else if (child == 0)
    {
```

1.  接下来，在父程序的子部分中，我们启动了在上一步中开发的子程序：

```cpp
        std::cout << "About to run the child process with PID = " 
                  << child << std::endl;
        if (execl("./child.out", "child.out", NULL) < 0)
        {
            std::cout << "error in executing child proceess " 
                      << std::endl;
            return 2;
        }
    }
    else
    {
```

1.  在父程序的父节（`else`部分）中，我们必须杀死子进程并检查它是否被正确杀死：

```cpp
        std::cout << "killing the child process with PID = " 
                  << child << std::endl;
        int status = kill (child, 9);
        if (status == 0)
            std::cout << "child process killed ...." << std::endl;
        else
            std::cout << "there was a problem killing
                the process with PID = " 
                      << child << std::endl;
    }
    return (0);
}
```

我们已经看到了父程序和子程序，父程序杀死了子进程。在下一节中，我们将学习这些程序的机制。

# 它是如何工作的...

在这之前，我们需要编译子程序和父程序——`g++ process_03.cpp`和`g++ -o child.out process_04.cpp`。

在编译`process_04.cpp`时，我们必须指定`-o child.out`，这是父进程所需的（进程名为`a.out`）。通过运行它，产生的输出如下：

![](img/93a1d294-77bb-4e04-b466-55e24149172d.png)

执行显示，PID 为 218 的子进程被父进程正确杀死。

这个教程中的代码只是*启动一个新进程*教程的变体。不同之处在于现在，父进程在其编制的一部分中杀死子进程`int status = kill (child, 9);`。`kill`系统调用接受要杀死的进程的 PID 作为第一个参数，作为第二个参数的是要发送给子进程的信号。接受的信号如下：

+   `1` = `HUP`（挂断）

+   `2` = `INT`（中断）

+   `3` = `QUIT`（退出）

+   `6` = `ABRT`（中止）

+   `9` = `KILL`（不可捕获，不可忽略的终止）

+   `14` = `ALRM`（闹钟）

+   `15` = `TERM`（软件终止信号）

`man 2 kill`，`kill`系统调用，向进程发送信号。成功时返回`0`；否则返回`-1`。你需要包含`#include <sys/types.h>`和`#include <signal.h>`来使用它。

# 还有更多...

在第二章的*理解并发性*教程中，我们提供了两种基于`std::thread`和`std::async`的替代解决方案（并且鼓励使用它们），如果可能的话。下一个教程还提供了`std::thread`使用的具体示例。

# 创建一个新线程

进程并不是构建软件系统的唯一方式；一个轻量级的替代方案是使用线程。这个教程展示了如何使用 C++标准库创建和管理线程。我们已经知道使用 C++标准库的主要优势是它的可移植性和不依赖外部库（例如 Boost）。

# 如何做...

我们将编写的代码将是对大整数向量求和的并发版本。向量被分成两部分；每个线程计算其部分的总和，主线程显示结果。

1.  让我们定义一个包含 100,000 个整数的向量，并在`main`方法中生成随机数：

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

void threadFunction (std::vector<int> &speeds, int start, int
    end, int& res);

int main()
{    
    std::vector<int> speeds (100000);
    std::generate(begin(speeds), end(speeds), [] () 
        { return rand() % 10 ; });

```

1.  接下来，启动第一个线程，传递前 50,000 个整数：

```cpp
    int th1Result = 0;
    std::thread t1 (threadFunction, std::ref(speeds), 0, 49999, 
        std::ref(th1Result));

```

1.  然后，启动第二个线程，传递第二个 50,000 个整数：

```cpp
    int th2Result = 0;    
    std::thread t2 (threadFunction, std::ref(speeds), 50000, 99999, 
        std::ref(th2Result));

```

1.  等待两个线程的结果：

```cpp
    t1.join();
    t2.join();
    std::cout << "Result = " << th1Result + th2Result
        << std::endl;
    return 0;
}

void threadFunction (std::vector<int> &speeds, int start, int 
    end, int& res)
{
    std::cout << "starting thread ... " << std::endl;
    for (int i = start; i <= end; ++i)
    res += speeds[i];
    std::cout << "end thread ... " << std::endl;
}
```

下一节解释了动态。

# 它是如何工作的...

通过使用`g++ thread_01.cpp -lpthread`编译程序并执行它，输出如下：

![](img/49847d80-39be-498e-864d-f9653ab3426d.png)

在*步骤 1*中，我们定义了`threadFunction`方法，这是基本的线程单元，负责从`start`到`end`对`speeds`中的元素求和，并将结果保存在`res`输出变量中。

在*步骤 2*和*步骤 3*中，我们启动了两个线程来计算`t1`线程的前 50,000 个项目的计算和第二个`t2`线程的 50,000 个项目。这两个线程并发运行，所以我们需要等待它们完成。在*步骤 4*中，我们等待`th1`和`th2`的结果完成，将两个结果—`th1Results`和`th2Results`—相加，并将它们打印在标准输出（`stdout`）中。

# 还有更多...

*启动一个新进程*食谱展示了如何创建一个进程，以及在哪些情况下进程适合解决方案。一个值得强调的重要方面是，线程在创建它的进程的**相同地址空间**中运行。尽管线程仍然是一种在更独立（可运行）模块中构建系统软件的好方法，但如果线程崩溃（由于段错误，或者如果某种原因调用了**`terminate`**等），整个应用程序都会崩溃。

从积极的一面来看，正如我们在前面的代码中看到的，线程之间的通信非常简单高效。此外，线程彼此之间，以及创建它们的进程，共享**静态**和**堆**内存。

尽管这个食谱中的代码很简单，但它展示了如何并发执行一个任务（大数组的总和）。值得一提的是，如果算法没有设计为并发运行，也就是说，如果线程之间存在依赖关系，那么多线程应用程序就毫无价值。

在这种情况下，重要的是要注意，如果两个线程同时在两个处理器上运行，我们会使用**并行**这个词。在这种情况下，我们没有这个保证。

我们使用了 C++标准库中的`std::thread`，但是同样的例子也可以使用`std::async`来编写。《第二章》《重温 C++》展示了两种方法的例子。您可以尝试使用第二种方法重写这个食谱的代码。

# 另请参阅

在《第二章》《重温 C++》中的*理解并发*食谱中，介绍了一个包括`std::thread`和`std::async`的并发主题的食谱。您还可以阅读 Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》中专门介绍线程的部分。

# 创建守护进程

系统编程实际上是与操作系统资源密切打交道，创建进程、线程、释放资源等等。有些情况下，我们需要一个进程*无限期地*运行；也就是说，一个进程首先提供一些服务或管理资源，然后一直运行下去。在后台*无限期运行*的进程称为**守护进程**。这个食谱将展示如何以编程方式生成一个守护进程。

# 操作步骤如下...

如前所述，守护进程是一个无限期运行的进程。为了被分类为*守护进程*，一个进程必须具有一些明确定义的属性，这将在这个食谱中用一个程序来展示。

1.  输入以下代码通过调用`umask`系统调用重置子进程的初始访问权限：

```cpp
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

int main(void)
{
    pid_t child;
    int status;
    std::cout << "I am the parent, my PID is " << getpid()
        << std::endl;
    std::cout << "I am going to create a new daemon process..."
        << std::endl;

    // 1\. clear file creation mask
    umask(0);

```

1.  输入代码以 fork 一个子进程：

```cpp
    child = fork();
    if (child == -1)
    {
        std::cout << "fork() failed." << std::endl;
        return (-1);
    }
    else if (child == 0) // child (daemon) process
    {

```

1.  在子进程上输入`setsid`命令：

```cpp
        setsid();

```

1.  将工作目录更改为子进程（现在是一个守护进程）：

```cpp
        if (chdir("/") < 0)
            std::cout << "Couldn't change directly" << std::endl;

```

1.  运行守护进程特定的任务——在这种情况下，只需睡眠`10`秒：

```cpp
        // Attach here the daemon specific long running
        // tasks ... sleep for now.
        sleep (10);
    }

```

1.  父进程在`fork`后退出：

```cpp
    return (0);
}
```

下一节将更详细地解释这六点。

# 工作原理...

使用`g++ daemon_01.cpp`（在 Docker 镜像的`/BOOK/Chapter03`文件夹中）编译代码并运行。输出如下：

![](img/d3d2b3b2-859c-41b4-a28e-75a75e9411ee.png)

当我们在 shell 上运行一个进程时，终端会等待子进程完成后才准备好接受另一个命令。我们可以使用`&`符号运行命令（例如，`ls -l &`），shell 会提示终端输入另一个命令。请注意，子进程仍然在与父进程相同的会话中。要使一个进程成为守护进程，应该遵循以下规则（*2*和*3*是强制的；其他是可选的）：

1.  使用参数`0`调用`umask`（`umask(0)`）：当父进程创建子进程时，文件模式创建掩码会被继承（也就是说，子进程将继承父进程的初始访问权限）。我们要确保重置它们。

1.  **在 fork 后使父进程退出**：在前面的代码中，父进程创建了子进程后返回。

1.  **调用** `setsid`。这做了三件事：

+   子进程成为一个新创建会话的领导者。

+   它成为一个新的进程组的领导者。

+   它与其控制终端解除关联。

1.  **更改工作目录**：父进程可能在一个临时（或挂载的）文件夹中运行，这个文件夹可能不会长时间存在。将当前文件夹设置为满足守护进程的长期期望是一个好习惯。

1.  **日志记录**：由于守护服务不再与任何终端设备相关联，将标准输入、输出和错误重定向到`/dev/null`是一个好习惯。

# 还有更多...

到目前为止，一个进程有一个 PID 作为其唯一标识符。它还属于一个具有**进程组 ID**（**PGID**）的组。进程组是一个或多个进程的集合。同一组中的所有进程可以从同一个终端接收信号。每个组都有一个领导者，PGID 的值与领导者的 PID 相同。

一个会话是一个或多个进程组的集合。这个示例表明可以通过调用`setsid`方法创建一个新的会话。

一个会话可以有一个（单一的）控制终端。`ps -efj`命令显示所有使用`PID`、`PPID`和`PGID`以及每个进程的控制终端（`TTY`）信息的进程：

![](img/68f40447-735e-48ba-a16b-4a44ac8be662.png)

输出显示`./a.out`守护进程的`PID = 19`，它是组的领导者（`PGID = 19`），并且它没有连接到任何控制终端（`TTY= ?`）。

# 参见

W.R. Stevens 的*UNIX 环境高级编程*第十三章专门讨论了守护进程。
