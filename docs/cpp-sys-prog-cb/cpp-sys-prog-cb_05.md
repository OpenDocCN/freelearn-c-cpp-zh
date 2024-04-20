# 使用互斥锁、信号量和条件变量

本章将重点介绍您可以使用的最常见机制，以同步对共享资源的访问。我们将研究的同步机制可以防止临界区域（负责资源的程序段）在两个或多个进程或线程中同时执行。在本章中，您将学习如何使用 POSIX 和 C++标准库同步构建块，如互斥锁、`std::condition_variable`、`std::promise`和`std::future`。

本章将涵盖以下示例：

+   使用 POSIX 互斥锁

+   使用 POSIX 信号量

+   POSIX 信号量的高级用法

+   同步构建块

+   学习使用简单事件进行线程间通信

+   学习使用条件变量进行线程间通信

# 技术要求

为了让您可以立即尝试本章中的所有程序，我们已经设置了一个 Docker 镜像，其中包含本书中将需要的所有工具和库。它基于 Ubuntu 19.04。

为了设置它，按照以下步骤进行：

1.  从[www.docker.com](http://www.docker.com)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该可用。输入`docker images`命令查看镜像。

1.  您应该有以下镜像：`kasperondocker/system_programming_cookbook`。

1.  使用`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`命令以交互式 shell 运行 Docker 镜像。

1.  正在运行的容器上的 shell 现在可用。使用`root@39a5a8934370/# cd /BOOK/`获取本书中将开发的所有程序。 

`--cap-add sys_ptrace`参数是为了允许 GDB 设置断点。Docker 默认情况下不允许这样做。

# 使用 POSIX 互斥锁

这个示例将教你如何使用 POSIX 互斥锁来同步多个线程对资源的访问。我们将通过开发一个包含一个方法（临界区域）的程序来实现这一点，该方法将执行一个不能并发运行的任务。我们将使用`pthread_mutex_lock`、`pthread_mutex_unlock`和`pthread_mutex_init` POSIX 方法来同步线程对其的访问。

# 如何做...

在这个示例中，我们将创建一个多线程程序，只需将一个整数增加到`200000`。为此，我们将开发负责增加计数器的临界区域，必须对其进行保护。然后，我们将开发主要部分，该部分将创建两个线程并管理它们之间的协调。让我们继续：

1.  打开一个名为`posixMutex.cpp`的新文件，并开发其结构和临界区域方法：

```cpp
#include <pthread.h>
#include <iostream>

struct ThreadInfo
{
    pthread_mutex_t lock;
    int counter;
};

void* increment(void *arg)
{
    ThreadInfo* info = static_cast<ThreadInfo*>(arg);
    pthread_mutex_lock(&info->lock);

    std::cout << "Thread Started ... " << std::endl;
    for (int i = 0; i < 100000; ++i)
        info->counter++;
    std::cout << "Thread Finished ... " << std::endl;

    pthread_mutex_unlock(&info->lock);
    return nullptr;
}
```

1.  现在，在`main`部分，添加所需的用于线程同步的锁的`init`方法：

```cpp
int main()
{
    ThreadInfo thInfo;
    thInfo.counter = 0;
    if (pthread_mutex_init(&thInfo.lock, nullptr) != 0)
    {
        std::cout << "pthread_mutex_init failed!" << std::endl;
        return 1;
    }
```

1.  现在我们有了将执行`increment`（即需要保护的临界区域）的方法和将管理线程之间同步的锁，让我们创建线程：

```cpp
    pthread_t t1;
    if (pthread_create(&t1, nullptr, &increment, &thInfo) != 0)
    {
        std::cout << "pthread_create for t1 failed! " << std::endl;
        return 2;
    }

    pthread_t t2;
    if (pthread_create(&t2, nullptr, &increment, &thInfo) != 0)
    {
        std::cout << "pthread_create for t2 failed! " << std::endl;
        return 3;
    }
```

1.  现在，我们需要等待线程完成任务：

```cpp
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    std::cout << "Threads elaboration finished. Counter = " 
              << thInfo.counter << std::endl;
    pthread_mutex_destroy(&thInfo.lock);
    return 0;
```

这个程序（在 Docker 镜像的`/BOOK/Chapter05/`文件夹下可用）向我们展示了如何使用 POSIX 互斥锁接口来同步多个线程对共享资源（在本例中是计数器）的使用。我们将在下一节中详细解释这个过程。

# 工作原理...

在第一步中，我们创建了传递参数给线程所需的`struct`：`struct ThreadInfo`。在这个`struct`中，我们放置了保护资源`counter`所需的锁和计数器本身。然后，我们开发了`increment`功能。`increment`逻辑上需要锁定`pthread_mutex_lock(&info->lock);`资源，增加计数器（或者临界区域需要的其他操作），然后解锁`pthread_mutex_unlock(&info->lock);`资源，以便其他线程执行相同的操作。

在第二步中，我们开始开发`main`方法。我们做的第一件事是使用`pthread_mutex_init`初始化锁互斥锁。在这里，我们需要传递指向本地分配资源的指针。

在第三步中，我们创建了两个线程`th1`和`th2`。它们负责同时运行`increment`方法。这两个线程是使用`pthread_create` POSIX API 创建的，通过传递在*步骤 2*中分配的`thInfo`的地址。如果线程成功创建，它将立即开始处理。

在第四步和最后一步中，我们等待`th1`和`th2`都完成将计数器的值打印到标准输出，我们期望的值是`200000`。通过编译`g++ posixMutex.cpp -lpthread`并运行`./a.out`程序，我们得到以下输出：

![](img/5918fefd-5ad2-4e38-80ca-a057d0b440b7.png)

正如我们所看到的，这两个线程从未重叠执行。因此，关键部分的计数器资源得到了正确管理，输出结果符合我们的预期。

# 还有更多...

在这个示例中，为了完整起见，我们使用了`pthread_create`。完全相同的目标可以通过使用 C++标准库中的`std::thread`和`std::async`来实现。

`pthread_mutex_lock()`函数锁定互斥锁。如果互斥锁已经被锁定，调用线程将被阻塞，直到互斥锁变为可用。`pthread_mutex_unlock`函数如果当前线程持有互斥锁，则解锁互斥锁；否则，将导致未定义的行为。

# 另请参阅

欢迎您修改此程序，并使用`std::thread`或`std::async`与 C++标准库中的`pthread_mutex_lock`和`pthread_mutex_unlock`结合使用。请参阅第二章，*重温 C++*，以便在这个主题上进行刷新。

# 使用 POSIX 信号量

POSIX 互斥锁显然不是您可以用来同步访问共享资源的唯一机制。这个示例将向您展示如何使用另一个 POSIX 工具来实现相同的结果。信号量与互斥锁不同，这个示例将教会您它们的基本用法，而下一个示例将向您展示更高级的用法。信号量是线程和/或进程之间的通知机制。作为一个经验法则，尝试使用互斥锁作为同步机制，使用信号量作为通知机制。在这个示例中，我们将开发一个类似于我们在*使用 POSIX 互斥锁*示例中构建的程序，但这次，我们将使用信号量来保护关键部分。

# 如何做...

在这个示例中，我们将创建一个多线程程序，以增加一个整数直到达到`200000`。同样，负责增量的代码部分必须受到保护，我们将使用 POSIX 信号量。`main`方法将创建两个线程，并确保正确销毁资源。让我们开始吧：

1.  让我们打开一个名为`posixSemaphore.cpp`的新文件，并开发结构和关键部分方法：

```cpp
#include <pthread.h>
#include <semaphore.h>
#include <iostream>

struct ThreadInfo
{
    sem_t sem;
    int counter;
};

void* increment(void *arg)
{
    ThreadInfo* info = static_cast<ThreadInfo*>(arg);
    sem_wait(&info->sem);

    std::cout << "Thread Started ... " << std::endl;
    for (int i = 0; i < 100000; ++i)
        info->counter++;
    std::cout << "Thread Finished ... " << std::endl;

    sem_post(&info->sem);
    return nullptr;
}
```

1.  现在，在`main`部分，添加用于线程之间同步所需的锁的`init`方法：

```cpp
int main()
{
    ThreadInfo thInfo;
    thInfo.counter = 0;
    if (sem_init(&thInfo.sem, 0, 1) != 0)
    {
        std::cout << "sem_init failed!" << std::endl;
        return 1;
    }
```

1.  现在`init`部分已经完成，让我们编写将启动两个线程的代码：

```cpp
pthread_t t1;
if (pthread_create(&t1, nullptr, &increment, &thInfo) != 0)
{
    std::cout << "pthread_create for t1 failed! " << std::endl;
    return 2;
}

pthread_t t2;
if (pthread_create(&t2, nullptr, &increment, &thInfo) != 0)
{
    std::cout << "pthread_create for t2 failed! " << std::endl;
    return 3;
}
```

1.  最后，这是结束部分：

```cpp
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    std::cout << "posixSemaphore:: Threads elaboration
        finished. Counter = " 
              << thInfo.counter << std::endl;
    sem_destroy(&thInfo.sem);
    return 0;
}
```

我们现在使用 POSIX 信号量运行与 POSIX 互斥锁相同的程序。正如您所看到的，程序的设计并没有改变-真正改变的是我们用来保护关键部分的 API。

# 工作原理...

第一部分包含用于与`increment`方法通信的结构以及方法本身的定义。与程序的互斥版本相比，主要区别在于我们现在包括了`#include <semaphore.h>`头文件，以便我们可以使用 POSIX 信号量 API。然后，在结构中，我们使用`sem_t`类型，这是实际将保护临界区的信号量。`increment`方法有两个屏障来保护实际逻辑：`sem_wait(&info->sem);`和`sem_post(&info->sem);`。这两种方法都是原子地分别减少和增加`sem`计数器。`sem_wait(&info->sem);`通过将计数器减少`1`来获取锁。如果计数器的值大于 0，则获取锁，并且线程可以进入临界区。`sem_post(&info->sem);`在退出临界区时只是将计数器增加 1。

在第二步中，我们通过调用`sem_init` API 来初始化信号量。在这里，我们传递了三个参数：

+   要初始化的信号量。

+   `pshared`参数。这表明信号量是在进程的线程之间共享还是在进程之间共享。`0`表示第一个选项。

+   最后一个参数表示信号量的初始值。通过将`1`传递给`sem_init`，我们要求信号量保护一个资源。通过`sem_wait`和`sem_post`，信号量将在内部自动增加和减少该计数器，让每个线程一次进入临界区。

在第三步中，我们创建了使用`increment`方法的两个线程。

在最后一步中，我们等待两个线程完成处理`pthread_join`，并且在本节中最重要的是，我们通过传递到目前为止使用的信号量结构来销毁信号量结构`sem_destroy`。

让我们编译并执行程序：`g++ posixSemaphore.cpp -lpthread`。即使在这种情况下，我们也需要通过将`-lpthread`选项传递给 g++来将程序链接到`libpthread.a`，因为我们使用了`pthreads`。这样做的输出如下：

![](img/6b0aabdc-b066-4e25-b9fc-5300752d41a4.png)

如预期的那样，输出显示计数器为`200000`。它还显示两个线程没有重叠。

# 还有更多...

我们通过向`sem_init`方法传递值`1`，将`sem_t`用作二进制信号量。信号量可以用作*计数信号量*，这意味着将一个大于 1 的值传递给`init`方法。在这种情况下，这意味着临界区将被*N*个线程同时访问。

有关 GNU/Linux man 页面的更多信息，请在 shell 中键入`man sem_init`。

# 另请参阅

您可以在下一个配方中了解有关*计数信号量*的更多信息，那里我们将学习互斥锁和信号量之间的区别。

您可以修改此程序，并使用`pthread_mutex_lock`和`pthread_mutex_unlock`与 C++标准库中的`std::thread`或`std::async`结合使用。

# POSIX 信号量高级用法

*使用 POSIX 信号量*配方向我们展示了如何使用 POSIX 信号量来保护临界区。在这个配方中，您将学习如何将其用作计数信号量和通知机制。我们将通过开发一个经典的发布-订阅程序来实现这一点，其中有一个发布者线程和一个消费者线程。这里的挑战是我们希望将队列中的最大项目数限制为一个定义的值。

# 如何做...

在这个配方中，我们将编写一个代表计数信号量的典型用例的程序 - 一个生产者-消费者问题，我们希望将队列中的项目数限制为某个数字。让我们开始吧：

1.  让我们打开一个名为`producerConsumer.cpp`的新文件，并编写我们在两个线程中需要的结构：

```cpp
#include <pthread.h>
#include <semaphore.h>
#include <iostream>
#include <vector>

constexpr auto MAX_ITEM_IN_QUEUE = 5;

struct QueueInfo
{
    sem_t mutex;
    sem_t full;
    sem_t empty;
    std::vector<int> queue;
};
```

1.  现在，让我们为`producer`编写代码：

```cpp
void* producer(void *arg)
{
    QueueInfo* info = (QueueInfo*)arg;
    std::cout << "Thread Producer Started ... " << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        sem_wait(&info->full);

        sem_wait(&info->mutex);
        info->queue.push_back(i);
        std::cout << "Thread Producer Started ... size = " 
                  << info->queue.size() << std::endl;
        sem_post(&info->mutex);

        sem_post(&info->empty);
    }
    std::cout << "Thread Producer Finished ... " << std::endl;
    return nullptr;
}
```

1.  我们对`consumer`做同样的操作：

```cpp
void* consumer(void *arg)
{
    QueueInfo* info = (QueueInfo*)arg;
    std::cout << "Thread Consumer Started ... " << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        sem_wait(&info->empty);

        sem_wait(&info->mutex);
        if (!info->queue.empty())
        {
            int b = info->queue.back();
            info->queue.pop_back();
        }
        sem_post(&info->mutex);

        sem_post(&info->full);
    }
    std::cout << "Thread Consumer Finished ... " << std::endl;
    return nullptr;
}
```

1.  现在，我们需要编写`main`方法，以便初始化资源（例如信号量）：

```cpp
int main()
{
    QueueInfo thInfo;
    if (sem_init(&thInfo.mutex, 0, 1) != 0 ||
        sem_init(&thInfo.full, 0, MAX_ITEM_IN_QUEUE) != 0 ||
        sem_init(&thInfo.empty, 0, 0) != 0)
    {
        std::cout << "sem_init failed!" << std::endl;
        return 1;
    }

    pthread_t producerPthread;
    if (pthread_create(&producerPthread, nullptr, &producer, 
        &thInfo) != 0)
    {
        std::cout << "pthread_create for producer failed! "
            << std::endl;
        return 2;
    }
    pthread_t consumerPthread;
    if (pthread_create(&consumerPthread, nullptr, &consumer, 
        &thInfo) != 0)
    {
        std::cout << "pthread_create for consumer failed! "
           << std::endl;
        return 3;
    }
```

1.  最后，我们需要编写释放资源的部分：

```cpp
    pthread_join(producerPthread, nullptr);
    pthread_join(consumerPthread, nullptr);

    sem_destroy(&thInfo.mutex);
    sem_destroy(&thInfo.full);
    sem_destroy(&thInfo.empty);
    return 0;
}
```

这个程序是基于信号量的典型消费者-生产者问题的实现，演示了如何将对资源的使用限制为*N*（在我们的例子中为`MAX_ITEM_IN_QUEUE`）。这个概念可以应用于其他问题，包括如何限制对数据库的连接数等。如果我们不是启动一个生产者，而是启动两个生产者线程，会发生什么？

# 它是如何工作的...

在程序的第一步中，我们定义了`struct`，这是让两个线程进行通信所需的。它包含以下内容：

+   一个`full`信号量（计数信号量）：此信号量设置为`MAX_ITEM_IN_QUEUE`。这限制了队列中项目的数量。

+   一个`empty`信号量（计数信号量）：此信号量在队列为空时通知进程。

+   一个`mutex`信号量（二进制信号量）：这是一个使用信号量实现的互斥锁，用于提供对队列访问的互斥排他。

+   队列：使用`std::vector`实现。

在第二步中，我们实现了`producer`方法。该方法的核心部分是`for`循环的实现。生产者的目标是将项目推送到队列中，同时不超过`MAX_ITEM_IN_QUEUE`个项目，因此生产者尝试通过递减`full`信号量（我们在`sem_init`中初始化为`MAX_ITEM_IN_QUEUE`）进入临界区，然后将项目推送到队列并递增空信号量（这允许消费者继续从队列中读取）。我们为什么需要通知消费者可以读取项目？换句话说，为什么我们需要在生产者中调用`sem_post(&info->empty);`？如果我们不这样做，消费者线程将不断读取项目，并且会将`full`信号量增加到大于`MAX_ITEM_IN_QUEUE`的值，导致队列中的项目超过`MAX_ITEM_IN_QUEUE`。

在第三步中，我们实现了`consumer`方法。这与`producer`相似。消费者所做的是等待通知以从队列中读取项目（使用`sem_wait(&info->empty);`），然后从队列中读取，并递增`full`信号量。这最后一步可以理解为：我刚刚从队列中消费了一个项目。

第四步是我们启动了两个线程并初始化了三个信号量。

第五步是结束部分。

如果我们启动更多的生产者，代码仍然可以工作，因为`full`和`empty`信号量将确保我们之前描述的行为，而队列上的`mutex`确保每次只有一个项目写入/读取。

POSIX 互斥锁和信号量都可以在线程和进程之间使用。要使信号量在进程之间工作，我们只需要在`sem_init`方法的第二个参数中传递一个不为 0 的值。对于互斥锁，我们需要在调用`pthread_mutexattr_setpshared`时传递`PTHREAD_PROCESS_SHARED`标志。通过构建和运行程序，我们将得到以下输出：

![](img/df663e70-e7d1-4142-85b6-54910957bcbb.png)

让我们在下一节中了解更多关于这个示例。 

# 还有更多...

值得注意的是，信号量可以初始化为三种可能的值（`sem_init`方法的第三个参数）：

+   对于`1`：在这种情况下，我们将信号量用作互斥锁。

+   对于`N`：在这种情况下，我们将信号量用作*计数信号量*。

+   对于`0`：我们将信号量用作通知机制（参见前面的`empty`信号量示例）。

一般来说，信号量必须被视为线程或进程之间的通知机制。

何时应该使用 POSIX 信号量和 POSIX 互斥锁？尝试使用互斥锁作为同步机制，使用信号量作为通知机制。此外，要考虑到在 Linux 内核中，POSIX 互斥锁通常比 POSIX 信号量更快。

最后一件事：请记住，POSIX 互斥锁和信号量都会使任务进入休眠状态，而自旋锁则不会。实际上，当互斥锁或信号量被锁定时，Linux 调度程序会将任务放入等待队列中。

# 另请参阅

请查看以下列表以获取更多信息：

+   本章中的*使用 POSIX 互斥锁*配方，以了解如何编写 POSIX 互斥锁

+   本章中的*使用 POSIX 信号量*配方，以了解如何编写 POSIX 互斥锁

+   *Linux 内核开发*，作者 Robert Love

# 同步构建模块

从这个配方和接下来的两个配方开始，我们将回到 C++世界。在这个配方中，我们将学习关于 C++同步构建模块。具体来说，我们将学习如何结合**资源获取即初始化**（**RAII**）的概念，使用`std::lock_guard`和`std::unique_lock`，这是一种使代码更健壮和可读的面向对象编程习惯。`std::lock_guard`和`std::unique_lock`将 C++互斥锁的概念封装在两个具有 RAII 概念的类中。`std::lock_guard`是最简单和最小的保护，而`std::unique_lock`在其上添加了一些功能。

# 如何做...

在这个配方中，我们将开发两个程序，以便学习如何使用`std::unique_lock`和`std::lock_guard`。让我们开始吧：

1.  从 shell 中创建一个名为`lock_guard.cpp`的新文件。然后，编写`ThreadInfo`结构和`increment`（线程）方法的代码：

```cpp
#include <iostream>
#include <mutex>
#include <thread>

struct ThreadInfo
{
    std::mutex mutex;
    int counter;
};

void increment(ThreadInfo &info)
{
    std::lock_guard<std::mutex> lock(info.mutex);
    std::cout << "Thread Started ... " << std::endl;

    for (int i = 0; i < 100000; ++i)
        info.counter++;

    std::cout << "Thread Finished ... " << std::endl;
}
```

1.  现在，按照以下方式编写`main`方法的代码：

```cpp
int main()
{
    ThreadInfo thInfo;

    std::thread t1 (increment, std::ref(thInfo));
    std::thread t2 (increment, std::ref(thInfo));

    t1.join();
    t2.join();

    std::cout << "Threads elaboration finished. Counter = " 
              << thInfo.counter << std::endl;
    return 0;
}
```

1.  让我们为`std::unique_lock`编写相同的程序。从 shell 中创建一个名为`unique_lock.cpp`的新文件，并编写`ThreadInfo`结构和`increment`（线程）方法的代码：

```cpp
#include <iostream>
#include <mutex>
#include <thread>
struct ThreadInfo
{
    std::mutex mutex;
    int counter;
};

void increment(ThreadInfo &info)
{
    std::unique_lock<std::mutex> lock(info.mutex);
    std::cout << "Thread Started ... " << std::endl;
    // This is a test so in a real scenario this is not be needed.
    // it is to show that the developer here has the possibility to 
    // unlock the mutex manually.
    // if (info.counter < 0)
    // {
    //    lock.unlock();
    //    return;
    // }
    for (int i = 0; i < 100000; ++i)
        info.counter++;
    std::cout << "unique_lock:: Thread Finished ... " << std::endl;
}
```

1.  关于`main`方法，在这里与我们在*使用 POSIX 互斥锁*配方中看到的没有区别：

```cpp
int main()
{
    ThreadInfo thInfo;

    std::thread t1 (increment, std::ref(thInfo));
    std::thread t2 (increment, std::ref(thInfo));

    t1.join();
    t2.join();

    std::cout << "Unique_lock:: Threads elaboration finished. 
        Counter = " 
              << thInfo.counter << std::endl;
    return 0;
}
```

这两个程序是我们在*使用 POSIX 互斥锁*配方中编写的 C++版本。请注意代码的简洁性。

# 工作原理...

`lock_guard.cpp`程序的*步骤 1*定义了所需的`ThreadInfo`结构和`increment`方法。我们首先看到的是使用`std::mutex`作为关键部分的保护机制。现在，`increment`方法简化了，开发人员的头疼减少了。请注意，我们有`std::lock_guard<std::mutex> lock(info.mutex);`变量定义。正如我们在方法中看到的那样，在最后没有`unlock()`调用-为什么？让我们看看`std::lock_guard`的工作原理：它的构造函数锁定互斥锁。由于`std::lock_guard`是一个类，当对象超出范围时（在这种情况下是在方法的末尾），析构函数被调用。`std::lock_guard`析构函数中调用`std::mutex`对象的解锁。这意味着无论`increment`方法发生什么，构造函数都会被调用，因此不存在死锁的风险，开发人员不必关心`unlock()`。我们在这里描述的是 RAII C++技术，它将`info.mutex`对象的生命周期与`lock`变量的生命周期绑定在一起。

*步骤 2*包含用于管理两个线程的主要代码。在这种情况下，C++具有更清晰和简单的接口。线程是用`std::thread t1 (increment, std::ref(thInfo));`创建的。在这里，`std::thread`接受两个参数：第一个是线程将调用的方法，而第二个是传递给增量方法的`ThreadInfo`。

`unique_lock.cpp`程序是我们迄今为止描述的`lock_guard`的版本。主要区别在于`std::unique_lock`给开发者更多的自由。在这种情况下，我们修改了`increment`方法，以模拟互斥体对`if (info.counter < 0)`情况的解锁需求。使用`std::unique_lock`，我们能够在任何时候手动`unlock()`互斥体并从方法中返回。我们无法在`std::lock_guard`类上做同样的事情。当然，`lock_guard`无论如何都会在作用域结束时解锁，但我们想要强调的是，使用`std::unique_lock`，开发者有自由在任何时候手动解锁互斥体。

通过编译`lock_guard.cpp`：`g++ lock_guard.cpp -lpthread`并运行生成的可执行文件，我们得到以下输出：

![](img/bd008ce4-2587-419a-b206-6f98ee10173c.png)

对于`unique_lock.cpp`也是一样：`g++ unique_lock.cpp -lpthread`，输出如下：

![](img/113fd27c-e7a5-4481-9a8f-49d066070a57.png)

正如预期的那样，两个输出完全相同，使用`lock_guard`的代码更清晰，从开发者的角度来看，肯定更安全。

# 还有更多...

正如我们在这个食谱中看到的，`std::lock_guard`和`std::unique_lock`是我们与`std::mutex`一起使用的模板类。`unique_lock`可以与其他互斥体对象一起定义，例如**`std::timed_mutex`**，它允许我们在特定时间内获取锁：

```cpp
#include <chrono>
using std::chrono::milliseconds;

std::timed_mutex timedMutex;
std::unique_lock<std::timed_mutex> lock {timedMutex, std::defer_lock};
lock.try_lock_for(milliseconds{5});
```

`lock`对象将尝试在`5`毫秒内获取锁。当添加`std::defer_lock`时，我们必须小心，它不会在构造时自动锁定互斥体。这只会在`try_lock_for`成功时发生。

# 另请参阅

这里是您可以参考的参考资料列表：

+   *Linux 内核开发*，作者 Robert Love

+   本章的*使用 POSIX 互斥体*食谱

+   本章的*使用 POSIX 信号量*食谱

+   第二章，*重温 C++*，进行 C++的复习

# 使用简单事件学习线程间通信

到目前为止，我们知道如何使用 POSIX 和 C++标准库机制来同步关键部分。有一些用例不需要显式使用锁；相反，我们可以使用更简单的通信机制。`std::promise`和`std::future`可用于允许两个线程进行通信，而无需同步的麻烦。

# 如何做...

在这个食谱中，我们将编写一个程序，将问题分成两部分：线程 1 将运行一个高强度的计算，并将结果发送给线程 2，线程 2 是结果的消费者。我们将使用`std::promise`和`std::future`来实现这一点。让我们开始吧：

1.  打开一个名为`promiseFuture.cpp`的新文件，并将以下代码输入其中：

```cpp
#include <iostream>
#include <future>

struct Item
{
    int age;
    std::string nameCode;
    std::string surnameCode;
};

void asyncProducer(std::promise<Item> &prom);
void asyncConsumer(std::future<Item> &fut);
```

1.  编写`main`方法：

```cpp
int main()
{
    std::promise<Item> prom;
    std::future<Item> fut = prom.get_future();

    std::async(asyncProducer, std::ref(prom));
    std::async(asyncConsumer, std::ref(fut));

    return 0;
}
```

1.  消费者负责通过`std::future`获取结果并使用它：

```cpp
void asyncConsumer(std::future<Item> &fut)
{
    std::cout << "Consumer ... got the result " << std::endl;
    Item item = fut.get();
    std::cout << "Age = " << item.age << " Name = "
        << item.nameCode
              << " Surname = " << item.surnameCode << std::endl;
}
```

1.  生产者执行处理以获取项目并将其发送给等待的消费者：

```cpp
void asyncProducer(std::promise<Item> &prom)
{
    std::cout << "Producer ... computing " << std::endl;

    Item item;
    item.age = 35;
    item.nameCode = "Jack";
    item.surnameCode = "Sparrow";

    prom.set_value(item);
}
```

这个程序展示了`std::promise`和`std::future`的典型用例，其中不需要互斥体或信号量进行一次性通信。

# 它是如何工作的...

在*步骤 1*中，我们定义了`struct Item`以在生产者和消费者之间使用，并声明了两个方法的原型。

在*步骤 2*中，我们使用`std::async`定义了两个任务，通过传递定义的 promise 和 future。

在*步骤 3*中，`asyncConsumer`方法使用`fut.get()`方法等待处理结果，这是一个阻塞调用。

在*步骤 4*中，我们实现了`asyncProducer`方法。这个方法很简单，只是返回一个预定义的答案。在实际情况下，生产者执行高强度的处理。

这个简单的程序向我们展示了如何简单地将问题从信息的生产者（promise）和信息的消费者中解耦，而不必关心线程之间的同步。这种使用`std::promise`和`std::future`的解决方案只适用于一次性通信（也就是说，我们不能在两个线程中发送和获取项目时进行循环）。

# 还有更多...

`std::promise`和`std::future`只是 C++标准库提供的并发工具。除了`std::future`之外，C++标准库还提供了`std::shared_future`。在这个配方中，我们有一个信息生产者和一个信息消费者，但如果有更多的消费者呢？`std::shared_future`允许多个线程等待相同的信息（来自`std::promise`）。

# 另请参阅

Scott Meyers 的书*Effective Modern C++*和 Bjarne Stroustrup 的书*The C++ Programming Language*详细介绍了这些主题。

您也可以通过 C++核心指南中的*CP:并发和并行*（[`github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#cp-concurrency-and-parallelism`](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#cp-concurrency-and-parallelism)）部分了解更多关于并发的内容。

# 学习使用条件变量进行线程间通信

在这个配方中，您将了解到标准库中提供的另一个 C++工具，它允许多个线程进行通信。我们将使用`std::condition_variable`和`std::mutex`来开发一个生产者-消费者程序。

# 如何做...

这个配方中的程序将使用`std::mutex`来保护队列免受并发访问，并使用`std::condition_variable`来通知消费者队列中已经推送了一个项目。让我们开始吧：

1.  打开一个名为`conditionVariable.cpp`的新文件，并将以下代码输入其中：

```cpp
#include <iostream>
#include <queue>
#include <condition_variable>
#include <thread>

struct Item
{
    int age;
    std::string name;
    std::string surname;
};

std::queue<Item> queue;
std::condition_variable cond;
std::mutex mut;

void producer();
void consumer();
```

1.  现在，让我们编写`main`方法，为消费者和生产者创建线程：

```cpp
int main()
{
    std::thread t1 (producer);
    std::thread t2 (consumer);

    t1.join();
    t2.join();
    return 0;
}
```

1.  让我们定义`consumer`方法：

```cpp
void consumer()
{
    std::cout << "Consumer ... " << std::endl;
    while(true)
    {
        std::unique_lock<std::mutex> lck{mut};
        std::cout << "Consumer ... loop ... START" << std::endl;
        cond.wait(lck);
        // cond.wait(lck, []{ return !queue.empty();});
        auto item = queue.front();
        queue.pop();
        std::cout << "Age = " << item.age << " Name = " 
                  << item.name << " Surname = " << item.surname
                    << std::endl;
        std::cout << "Queue Size = " << queue.size() << std::endl;
        std::cout << "Consumer ... loop ... END" << std::endl;
        lck.unlock();
    }
}
```

1.  最后，让我们定义`producer`方法：

```cpp
void producer()
{
    while(true)
    {
        Item item;
        item.age = 35;
        item.name = "Jack";
        item.surname = "Sparrow";
        std::lock_guard<std::mutex> lock {mut};
        std::cout << "Producer ... loop ... START" << std::endl;
        queue.push(item);
        cond.notify_one();
        std::cout << "Producer ... loop ... END" << std::endl;
    }
}
```

尽管我们开发的程序解决了我们在上一个配方中看到的典型的生产者-消费者问题，但代码更符合惯用法，易于阅读，且更少出错。

# 它是如何工作的...

在第一步中，我们定义了需要从生产者传递给消费者的`struct Item`。这一步中有趣的一点是`std::queue`变量的定义；它使用一个互斥量来同步对队列的访问，并使用`std::condition_variable`来从生产者向消费者通信一个事件。

在第二步中，我们定义了生产者和消费者线程，并调用了`join()`方法。

在第三步中，消费者方法基本上做了四件事：获取锁以从队列中读取项目，等待生产者通过条件变量`cond`发出通知，从队列中弹出一个项目，然后释放锁。有趣的是，条件变量使用`std::unique_lock`而不是`std::lock_guard`，原因很简单：一旦在条件变量上调用`wait()`方法，锁就会（在内部）被释放，以便生产者不被阻塞。当生产者调用`notify_one`方法时，消费者上的`cond`变量会被唤醒并再次锁定互斥量。这使得它可以安全地从队列中弹出一个项目，并在最后再次释放锁`lck.unlock()`。在`cond.wait()`之后（注释掉的代码），还有一种通过传递第二个参数，谓词来调用`wait()`的替代方法，如果第二个参数返回 false，它将继续等待。在我们的情况下，如果队列不为空，消费者将不会等待。

最后一步非常简单：我们创建一个项目，用互斥锁`lock_guard`锁定它，并将其推送到队列中。请注意，通过使用`std::lock_guard`，我们不需要调用 unlock；`lock`变量的析构函数会处理这个问题。在结束当前循环之前，我们需要做的最后一件事是用`notify_one`方法通知消费者。

`g++ conditionVariable.cpp -lpthread`的编译和`./a.out`程序的执行将产生以下输出：

![](img/726a088f-c4c0-4c67-8d11-ddca1550ff4b.png)

请注意，由于`condition_variable`是异步的，生产者比消费者快得多，因此需要支付一定的延迟。正如您可能已经注意到的，生产者和消费者会无限运行，因此您必须手动停止进程（*Ctrl* + *C*）。

# 还有更多...

在这个示例中，我们在生产者中使用了`condition_variable`的`notify_one`方法。另一种方法是使用`notify_all`，它会通知所有等待的线程。

另一个需要强调的重要方面是，当生产者希望通知等待的线程之一发生在计算中的事件，以便消费者可以采取行动时，最好使用条件变量。例如，假设生产者通知消费者已经推送了一个特殊项目，或者生产者通知队列管理器队列已满，因此必须生成另一个消费者。

# 另请参阅

+   在第二章的*创建新线程*一节，*重温 C++*，以了解更多信息或刷新自己关于 C++中的线程。

+   《C++编程语言》，作者 Bjarne Stroustrup，详细介绍了这些主题。
