# 第十一章：调度

系统编程涉及与底层操作系统的交互。调度程序是每个操作系统的核心组件之一，影响进程在 CPU 上的分配方式。最终，这是最终用户关心的：进程顺利运行，并且具有正确的优先级超过其他进程。本章将教会您与调度程序交互所需的实际技能，方法包括更改进程的策略、其`nice`值、实时优先级、处理器亲和力，以及实时进程如何**让出**处理器。

本章将涵盖以下示例：

+   学习设置和获取调度程序策略

+   学习获取时间片值

+   学习如何设置 nice 值

+   学习如何让出处理器

+   了解处理器亲和力

# 技术要求

为了尝试本章中的程序，我们设置了一个 Docker 镜像，其中包含本书中将需要的所有工具和库。它基于 Ubuntu 19.04。

要设置它，请按照以下步骤进行：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。输入以下命令查看镜像：`docker images`。

1.  您应该有以下镜像：`kasperondocker/system_programming_cookbook`。

1.  使用`docker run -it --cpu-rt-runtime=95000 --ulimit rtprio=99 --cap add=sys_nice kasperondocker/system_programming_cookbook:latest /bin/bash`命令以交互式 shell 运行 Docker 镜像。

1.  正在运行的容器上现在可用 shell。使用`root@39a5a8934370/# cd /BOOK/`获取为本书开发的所有程序。

`--cpu-rt-runtime=95000`、`--ulimit rtprio=99`和`--cap add=sys_nice`参数是为了允许在 Docker 中编写的软件设置调度程序参数。如果主机已正确配置，软件将不会有任何问题。

**免责声明**：C++20 标准已经在二月底的布拉格会议上由 WG21 批准（即在技术上完成）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 示例代码。GCC 将最新功能的开发保留在分支中（您必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励您自己尝试。因此，请克隆并探索 GCC 合同和模块分支，并尽情玩耍。

# 学习设置和获取调度程序策略

在系统编程环境中，有些进程必须与其他进程处理方式不同。不同之处在于进程获取处理器时间或不同优先级的方式。系统程序员必须意识到这一点，并学会如何与调度程序的 API 进行交互。这个示例将向您展示如何更改进程的**策略**以满足不同的调度要求。

# 如何操作...

这个示例将向您展示如何获取和设置进程的*policy*以及可以分配给它的限制。让我们开始吧：

1.  在 shell 上，让我们打开一个名为`schedParameters.cpp`的新源文件。我们需要检查当前（默认）进程策略是什么。为此，我们将使用`sched_getscheduler()`系统调用：

```cpp
#include <sched.h>
#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int main ()
{
    int policy = sched_getscheduler(getpid());
    switch(policy) 
    {
        case SCHED_OTHER: std::cout << "process' policy = 
            SCHED_OTHER" 
                                    << std::endl ; break;
        case SCHED_RR: std::cout << "process' policy = SCHED_RR" 
                                 << std::endl; break;
        case SCHED_FIFO: std::cout << "process' policy = SCHED_FIFO" 
                                   << std::endl; break;
        default: std::cout << "Unknown policy" << std::endl;
    }
```

1.  现在，我们想要分配`SCHED_FIFO`策略和实时（`rt`）优先级。为了使代码可移植，我们从`sched_get_priority_min`和`sched_get_priority_max`API 中获取最小值和最大值：

```cpp
    int fifoMin = sched_get_priority_min(SCHED_FIFO);
    int fifoMax = sched_get_priority_max(SCHED_FIFO);
    std::cout << "MIN Priority for SCHED_FIFO = " << fifoMin
        << std::endl;
    std::cout << "MAX Priority for SCHED_FIFO = " << fifoMax
        << std::endl;

    struct sched_param sched;
    sched.sched_priority = (fifoMax - fifoMin) / 2;
    if (sched_setscheduler(getpid(), SCHED_FIFO, &sched) < 0)
        std::cout << "sched_setscheduler failed = " 
                  << strerror(errno) << std::endl;
    else
        std::cout << "sched_setscheduler has set priority to = "
                  << sched.sched_priority << std::endl;
```

1.  我们应该能够检查已分配的新的`SCHED_FIFO`策略，使用`sched_getscheduler()`函数：

```cpp
    policy = sched_getscheduler(getpid());
    std::cout << "current process' policy = " << policy << std
        ::endl ;
    return 0;
} 
```

下一节将详细描述前面的代码。

# 工作原理...

POSIX 标准定义了以下策略：

+   `SCHED_OTHER`：正常的调度程序策略（即非实时进程）

+   `SCHED_FIFO`：先进先出

+   `SCHED_RR`：轮转

在这里，`SCHED_OTHER`是默认的，而`SCHED_FIFO`和`SCHED_RR`是实时的。实际上，Linux 将`SCHED_NORMAL`、`SCHED_BATCH`和`SCHED_IDLE`定义为其他实时策略。这些定义在`sched.h`头文件中。

*步骤 1*调用`sched_getscheduler()`检查进程的当前策略。预期的默认值是`SCHED_OTHER`。我们将输入传递给`getpid()`函数（`<unistd.h>`），该函数返回当前进程的 PID。`sched_getscheduler()`还接受`0`，在这种情况下表示当前进程。

*步骤 2*的目标是设置实时策略，并使用`sched_setscheduler()`函数为当前进程设置优先级。我们希望该进程比机器上运行的普通进程具有更高的优先级。例如，考虑（软）实时应用程序，其中计算不能被中断，或者收到软件中断并且其处理不能被延迟。这些 Linux 系统通常只运行很少的进程，用于专用目的。为了实现这一点，要设置的策略是`SCHED_FIFO`，我们设置的优先级是当前系统上可以设置的最小值和最大值之间的中间值。建议始终使用`sched_get_priority_max()`和`sched_get_priority_min()`函数检查这些值，以编写可移植的代码。需要强调的一点是，`sched_setscheduler()`函数在内部设置了`struct task_struct`的`rt_priority`字段。

*步骤 3*通过调用`sched_getscheduler()`函数检查了`SCHED_FIFO`是否已正确设置，类似于*步骤 1*中发生的情况。

# 还有更多...

`SCHED_FIFO`和`SCHED_RR`是由 POSIX 定义并在 Linux 上实现的两种策略，它们分配任务给更适合实时软件的处理器。让我们看看它们是如何工作的：

+   `SCHED_FIFO`：当任务由此策略返回时，它会继续运行，直到阻塞（例如，I/O 请求）、让出处理器或更高优先级的任务抢占它。

+   `SCHED_RR`：这与`SCHED_FIFO`的逻辑完全相同，但有一个区别：使用此策略调度的任务分配了时间片，以便任务继续运行，直到时间片到期或更高优先级的任务抢占它或让出处理器。

请注意，当`SCHED_OTHER`（或`SCHED_NORMAL`）实现抢占式的多任务处理时，`SCHED_FIFO`和`SCHED_RR`是协作的（它们不会被抢占）。

Linux 主调度程序函数循环遍历所有策略，并针对每个策略询问下一个要运行的任务。它使用`pick_next_task()`函数执行此操作，该函数由每个策略实现。主调度程序在`kernel/sched.c`中定义，该文件定义了`sched_class`结构。该结构规定必须定义和实现每个策略，以便所有不同的策略都能正常工作。让我们以图形方式查看这一点：

+   `kernel/sched.c`：定义`struct sched_class`并循环遍历以下策略：

+   `kernel/rt.c`（用于`SCHED_FIFO`和`SCHED_RR`）设置了具体实时策略函数的`const struct sched_class rt_sched_class`。

+   `kernel/fair.c`（用于`SCHED_NORMAL`或`SCHED_OTHER`）使用公平调度程序特定函数设置了`const struct sched_class fair_sched_class`。

看待 Linux 调度程序设计的一种方式是：`kernel/sched.c`定义了接口和接口下的具体策略。接口由`struct sched_class`结构表示。以下是`SCHED_OTHER`/`SCHED_NORMAL`（CFS 公平调度程序策略）的接口实现：

```cpp
static const struct sched_class fair_sched_class = {
 .next = &idle_sched_class,
 .enqueue_task = enqueue_task_fair,
 .dequeue_task = dequeue_task_fair,
 .yield_task = yield_task_fair,
 .check_preempt_curr = check_preempt_wakeup,
 .pick_next_task = pick_next_task_fair,
 .put_prev_task = put_prev_task_fair,

#ifdef CONFIG_SMP
 .select_task_rq = select_task_rq_fair,
 .load_balance = load_balance_fair,
 .move_one_task = move_one_task_fair,
 .rq_online = rq_online_fair,
 .rq_offline = rq_offline_fair,
 .task_waking = task_waking_fair,
#endif
 .set_curr_task = set_curr_task_fair,
 .task_tick = task_tick_fair,
 .task_fork = task_fork_fair,
 .prio_changed = prio_changed_fair,
 .switched_to = switched_to_fair,
 .get_rr_interval = get_rr_interval_fair,

#ifdef CONFIG_FAIR_GROUP_SCHED
 .task_move_group = task_move_group_fair,
#endif
};
```

`SCHED_FIFO`和`SCHED_RR`策略的实时优先级范围是`[1, 99]`，而`SCHED_OTHER`优先级（称为`nice`）是`[-20, 10]`。

# 另请参阅

+   *学习如何设置 nice 值*的步骤，以了解实时优先级与 nice 优先级的关系

+   *学习如何让出处理器*这个教程可以学习如何让一个正在运行的实时任务

+   *Linux 内核开发*，第三版，作者 Robert Love

# 学习获取时间片值

Linux 调度程序提供了不同的策略来分配处理器时间给任务。*学习设置和获取调度程序策略*这个教程展示了有哪些策略可用以及如何更改它们。`SCHED_RR`策略，即循环轮询策略，是用于实时任务（使用`SCHED_FIFO`）的策略。`SCHED_RR`策略为每个进程分配一个时间片。这个教程将向您展示如何配置时间片。

# 如何做...

在这个教程中，我们将编写一个小程序，通过使用`sched_rr_get_interval()`函数来获取循环轮询时间片：

1.  在一个新的 shell 中，打开一个名为`schedGetInterval.cpp`的新文件。我们必须包括`<sched.h>`以获取调度程序功能，`<iostream.h>`以记录到标准输出，`<string.h>`以使用`strerror`函数并将`errno`整数转换为可读字符串：

```cpp
#include <sched.h>
#include <iostream>
#include <string.h>

int main ()
{
    std::cout << "Starting ..." << std::endl;
```

1.  要获取循环轮询间隔，我们必须为我们的进程设置调度程序策略：

```cpp
    struct sched_param sched;
    sched.sched_priority = 8;
    if (sched_setscheduler(0, SCHED_RR, &sched) == -1)
        std::cout << "sched_setscheduler failed = "
            << strerror(errno) 
                  << std::endl;
    else
        std::cout << "sched_setscheduler, priority set to = " 
                  << sched.sched_priority << std::endl;
```

1.  现在，我们可以使用`sched_rr_get_interval()`函数获取时间间隔：

```cpp
    struct timespec tp;
    int retCode = sched_rr_get_interval(0, &tp);
    if (retCode == -1)
    {
        std::cout << "sched_rr_get_interval failed = " 
                  << strerror(errno) << std::endl;
        return 1;
    }    

    std::cout << "timespec sec = " << tp.tv_sec 
              << " nanosec = " << tp.tv_nsec << std::endl;
    std::cout << "End ..." << std::endl;
    return 0;
}
```

让我们看看这是如何在底层工作的。

# 它是如何工作的...

当任务使用`SCHED_RR`策略获取处理器时，它优先于`SCHED_OTHER`和`SCHED_NORMAL`任务，并分配一个定义的时间片，直到时间片到期为止。较高优先级的任务运行，直到它们明确让出处理器或阻塞。对于系统程序员来说，了解`SCHED_RR`策略的时间片是一个重要因素。这非常重要。如果时间片太大，其他进程可能要等很长时间才能获得 CPU 时间，而如果时间片太小，系统可能会花费大量时间进行上下文切换。

*步骤 1*显示了程序其余部分所需的包含文件。`<iostream>`用于标准输出，`<sched.h>`用于访问调度程序功能，`<string.h>`用于`strerror()`函数。

*步骤 2*非常重要，因为它为当前进程设置了`SCHED_RR`策略。您可能已经注意到，我们将`0`作为第一个参数传递。这是完全可以的，因为`sched_setscheduler()`函数的手册页上说，*如果 pid 等于零，则将设置调用线程的策略*。

*步骤 3*调用了`sched_rr_get_interval()`函数。它接受两个参数：PID 和`struct timespec`。第一个是输入参数，而后者是一个输出参数，它以`{sec, nanoseconds}`的形式包含时间片。对于第一个参数，我们可以传递`getpid()`函数，该函数返回当前进程的 PID。然后，我们简单地记录标准输出到返回的时间片。

# 还有更多...

`SCHED_RR`时间片来自哪里？正如我们已经知道的，Linux 调度程序有不同的策略。它们都是在不同的模块中实现的：`kernel/sched_fair.c`用于`SCHED_NORMAL`或`SCHED_OTHER`，`kernel/rt.c`用于`SCHED_RR`和`SCHED_FIFO`。通过查看`kernel/rt.c`，我们可以看到`sched_rr_get_interval()`函数返回`sched_rr_timeslice()`变量，该变量在模块顶部定义。我们还可以看到，如果为`SCHED_FIFO`策略调用`sched_rr_timeslice()`，它将返回`0`。

# 另请参阅

+   *学习如何让出处理器*这个教程是停止运行任务的替代方法，而不是等待时间片。

+   *学习设置和获取调度程序策略*这个教程

+   *Linux 内核开发，第三版*，作者 Robert Love

# 学习如何设置一个良好的值

`SCHED_OTHER`/`SCHED_NORMAL`策略实现了所谓的完全公平调度器（`CFS`）。这个配方将向您展示如何为普通进程设置 nice 值以增加它们的优先级。我们将看到 nice 值用于权衡进程的时间片。优先级不应与实时优先级混淆，实时优先级是特定于`SCHED_FIFO`和`SCHED_RR`策略的。

# 如何做...

在这个配方中，我们将实现一个程序，它将增加进程的 nice 值：

1.  在 shell 上，打开一个名为`schedNice.cpp`的新源文件。我们需要添加一些包含并调用`nice()`系统调用，通过传递我们想要为当前进程设置的值：

```cpp
#include <string.h>
#include <iostream>
#include <unistd.h>

int main ()
{
    std::cout << "Starting ..." << std::endl;

    if (nice(5) == -1)
        std::cout << "nice failed = " << strerror(errno)
            << std::endl;
    else
        std::cout << "nice value successfully set = " << std::endl;

    while (1) ;

    std::cout << "End ..." << std::endl;
    return 0;
}
```

在下一节中，我们将看到这个程序是如何工作的，以及`nice`值如何影响任务在处理器上的时间。

# 它是如何工作的...

*步骤 1*基本上调用`nice()`系统调用，它会按给定的数量增加任务的静态优先级。只是为了明确起见，假设一个进程从优先级`0`（这是`SCHED_OTHER`和`SCHED_NORMAL`策略的默认值）开始，连续两次调用`nice(5)`将把它的静态优先级设置为`10`。

让我们构建并运行`schedNice.cpp`程序：

![](img/72ada2c0-3f65-42c4-9324-8ae514069606.png)

在这里，我们可以看到，在左边，我们的进程正在运行，而在右边，我们运行了`ps -el`命令来获取正在运行的进程的`nice`值。我们可以看到`./a.out`进程现在有一个`nice`值为`5`。要给一个任务更高的优先级（然后一个更低的`nice`值），进程需要以 root 身份运行。

# 还有更多...

`struct task_struct`结构有三个值来表示任务优先级：`rt_prio`、`static_prio`和`prio`。我们在*学习如何设置和获取调度策略*配方中讨论了`rt_prio`，并定义了这个字段代表实时任务的优先级。`static_prio`是`struct task_struct`字段，用于存储`nice`值，而`prio`包含实际的任务优先级。`static_prio`越低，任务的`prio`值就越高。

也许有些情况下我们需要在运行时设置进程的`nice`值。在这种情况下，我们应该使用的命令是`renice value -p pid`；例如，`renice 10 -p 186`。

# 参见

+   *学习如何让出处理器*这个配方作为停止运行任务的替代方法，而不是等待时间片

+   *学习如何设置和获取调度策略*配方

# 学习如何让出处理器

当一个任务使用实时调度策略之一（即`SCHED_RR`或`SCHED_FIFO`）进行调度时，您可能需要让出处理器（让出任务意味着放弃 CPU，使其可用于其他任务）。正如我们在*学习如何设置和获取调度策略*配方中所描述的，当一个任务使用`SCHED_FIFO`策略进行调度时，它不会离开处理器，直到发生某个特定事件；也就是说，没有时间片的概念。这个配方将向您展示如何使用`sched_yield()`函数让出一个进程。

# 如何做...

在这个配方中，我们将开发一个程序，它将让出当前进程：

1.  在 shell 上，打开一个名为`schedYield.cpp`的新源文件，并输入以下代码：

```cpp
#include <string.h>
#include <iostream>
#include <sched.h>

int main ()
{
    std::cout << "Starting ..." << std::endl;

    // set policy to SCHED_RR.
    struct sched_param sched;
    sched.sched_priority = 8;
    if (sched_setscheduler(0, SCHED_RR, &sched) == -1)
        std::cout << "sched_setscheduler failed = " 
                  << strerror(errno) 
                  << std::endl;

   for( ;; )
   {
      int counter = 0;
      for(int i = 0 ; i < 10000 ; ++i)
         counter += i;

      if (sched_yield() == -1)
      {
         std::cout << "sched_yield failed = " 
                   << strerror(errno) << std::endl;
         return 1;
      }
   }

   // we should never get here ...
   std::cout << "End ..." << std::endl;
   return 0;
}
```

在下一节中，我们将描述我们的程序和`sched_yield()`的工作原理。

# 它是如何工作的...

当在使用`SCHED_FIFO`或`SCHED_RR`调度的任务上调用`sched_yield()`时，它会被移动到具有相同优先级的队列的末尾，并运行另一个任务。让步会导致上下文切换，因此应谨慎使用，仅在严格需要时使用。

*步骤 1*定义了一个程序，向我们展示了如何使用`sched_yield()`。我们模拟了一种 CPU 密集型的进程，在这种进程中，我们定期检查以便让出处理器。在这之前，我们必须将此进程的策略类型设置为`SCHED_RR`，优先级设置为`8`。如你所见，没有关于要让出的进程（PID）的信息，因此它假定当前任务将被让出。

# 还有更多...

`sched_yield()`是一个可以被用户空间应用程序使用的系统调用。Linux 通常调用`yield()`系统调用，它的优势是保持进程处于`RUNNABLE`状态。

# 另请参阅

+   *学习设置和获取调度程序策略*的教程来回顾如何更改策略的类型

+   *Linux 内核开发*，第三版，作者 Robert Love

# 了解处理器亲和力

在多处理器环境中，调度程序必须处理多个处理器或核心上的任务分配。从 Linux 的角度来看，进程和线程是相同的；两者都由`struct task_struct`内核结构表示。可能需要强制两个或更多任务（即线程或进程）在同一个处理器上运行，以利用例如避免缓存失效的缓存。本教程将教你如何在任务上设置*硬亲和力*。

# 如何做...

在本教程中，我们将开发一个小型软件，强制其在一个 CPU 上运行：

1.  在 shell 中，打开一个名为`schedAffinity.cpp`的新源文件。我们想要检查新创建进程的亲和力掩码。然后，我们需要准备`cpu_set_t`掩码，以在 CPU 上设置亲和力为`3`：

```cpp
#include <iostream>
#include <sched.h>
#include <unistd.h>

void current_affinity();
int main ()
{
    std::cout << "Before sched_setaffinity => ";
    current_affinity();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
int cpu_id = 3;
    CPU_SET(cpu_id, &cpuset);
```

1.  现在，我们准备调用`sched_setaffinity()`方法，并强制将当前任务的硬亲和力设置在 CPU 编号为`3`上。为了检查亲和力是否已正确设置，我们还将打印掩码：

```cpp
    int set_result = sched_setaffinity(getpid(), 
                                       sizeof(cpu_set_t), 
                                       &cpuset);
    if (set_result != 0) 
    {
        std::cerr << "Error on sched_setaffinity" << std::endl;
    }

    std::cout << "After sched_setaffinity => ";
    current_affinity();
    return 0;
}
```

1.  现在，我们需要开发`current_affinity()`方法，它将只打印处理器的掩码：

```cpp
// Helper function
void current_affinity()
{
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) 
    {
        std::cerr << "error on sched_getaffinity";
        return;
    }
    else
    {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        for (int i = 0; i < nproc; i++) 
        {
            std::cout << CPU_ISSET(i, &mask);
        }
        std::cout << std::endl;
    }
}
```

如果我们在一个不存在的 CPU 上设置亲和力（例如`cpu_id = 12`），会发生什么？内核中的哪个位置存储了亲和力掩码信息？我们将在下一节中回答这些和其他问题。

# 它是如何工作的...

*步骤 1*做了两件事。首先，它打印了默认的亲和力掩码。我们可以看到该进程被调度在所有处理器上运行。其次，它准备了`cpu_set_t`，它表示一组 CPU，通过使用`CPU_ZERO`宏进行初始化，并使用`CPU_SET`宏在 CPU`3`上设置亲和力。请注意，`cpu_set_t`对象必须直接操作，但只能通过提供的宏进行操作。完整的宏列表在手册页上有文档：`man cpu_set`。

*步骤 2*调用`sched_setaffinity()`系统调用，在由`getpid()`函数返回的 PID 的进程上设置亲和力（在`mask`变量中指定为`cpu_set_t`）。我们可以传递`0`而不是`getpid()`，表示当前进程。在`setaffinity`函数之后，我们打印 CPU 的掩码以验证正确的新值。

*步骤 3*包含了我们用来将标准输出打印到 CPU 掩码上的辅助函数的定义。请注意，我们通过`sysconf()`系统调用并传递`_SC_NPROCESSORS_ONLN`来获取可用处理器的数量。此函数检查了`/sys/`文件夹中存在的系统信息。然后，我们循环遍历每个处理器，并调用`CPU_ISSET`宏，同时传递`i-th`。`CPU_ISSET`宏将为第`i`个 CPU 设置相应的位。

如果尝试修改`int cpu_id = 3`并传递一个不同的处理器，即一个不存在的处理器（例如`15`），`sched_setaffinity()`函数显然会失败，返回`EINVAL`，并保持亲和力掩码不变。

现在让我们来看一下程序：

![](img/7d5e58de-a8e5-4864-95ee-0d348ada14af.png)

正如我们所看到的，每个处理器的 CPU 掩码都设置为 1。这意味着在这个阶段，进程可以在每个 CPU 上调度。现在，我们设置了掩码，要求调度程序只在 CPU `3`上运行该进程（**硬亲和力**）。当我们调用`sched_getaffinity()`时，掩码会反映这一点。

# 还有更多...

当我们调用`sched_setaffinity()`系统调用时，我们要求调度程序在特定处理器上运行任务。我们称之为硬亲和力。还有软亲和力。这是由调度程序自动管理的。Linux 始终尝试优化资源，并避免缓存失效，以加快整个系统的性能。

当我们通过宏设置亲和性掩码时，基本上是在`task_struct`结构中设置了`cpus_allowed`。这是非常合理的，因为我们正在为一个或多个 CPU 上的进程或线程设置亲和性。

如果要将任务的亲和性设置为多个 CPU，必须为要设置的 CPU 调用`CPU_SET`宏。

# 另请参阅

+   *学习如何让出处理器*食谱

+   *学习获取时间片值*食谱

+   *学习设置和获取调度策略*食谱
