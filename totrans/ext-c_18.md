# 第十八章

# 进程同步

本章继续上一章的讨论，即*进程执行*，我们的主要焦点将是进程同步。多进程程序中的控制机制与我们在多线程程序中遇到的控制技术不同。不仅仅是内存不同；还有其他因素在多线程程序中找不到，它们存在于多进程环境中。

尽管线程绑定到进程上，但进程可以在任何机器上自由运行，使用任何操作系统，位于互联网大小的网络中的任何位置。正如你可能想象的那样，事情变得复杂。在这样一个分布式系统中同步多个进程将不会容易。

本章专门讨论仅在一台机器上发生的进程同步。换句话说，它主要讨论单主机同步及其相关技术。我们简要讨论了分布式系统中的进程同步，但不会深入探讨。

本章涵盖了以下主题：

+   首先，我们描述了多进程软件，其中所有进程都在同一台机器上运行。我们介绍了单主机环境中的可用技术。我们利用前一章的知识来给出一些示例，以展示这些技术。

+   在我们尝试同步多个进程的第一步中，我们使用了命名 POSIX 信号量。我们解释了它们应该如何使用，然后给出了一个示例，解决了我们在前几章中遇到的竞态条件问题。

+   之后，我们将讨论命名 POSIX 互斥锁，并展示如何使用共享内存区域来创建并使命名互斥锁生效。作为一个例子，我们解决了一个由信号量解决的相同竞态条件问题，这次使用的是命名互斥锁。

+   作为同步多个进程的最后一种技术，我们讨论了命名 POSIX 条件变量。像命名互斥锁一样，它们需要放在共享内存区域中才能被多个进程访问。我们给出了一个关于这一技术的详细示例，展示了如何使用命名 POSIX 条件变量来同步多进程系统。

+   作为本章的最终讨论，我们简要讨论了那些在其网络周围分布有自己进程的多进程系统。我们讨论了它们的特性和与单主机多进程系统相比的问题性差异。

让我们以更多关于单主机并发控制和其中可用的技术为话题开始本章。

# 单主机并发控制

在某些情况下，一个机器上同时运行多个进程，这些进程需要同时访问共享资源是很常见的。由于所有进程都在同一个操作系统下运行，它们可以访问操作系统提供的所有设施。

在本节中，我们展示如何使用这些设施中的某些部分来创建一个同步进程的控制机制。共享内存在这些控制机制中起着关键作用；因此，我们高度依赖我们在上一章中解释的关于共享内存的内容。

以下是一个列表，列出了 POSIX 提供的控制机制，可以在所有进程都在同一 POSIX 兼容机器上运行时使用：

+   **命名 POSIX 信号量**：与我们第十六章“线程同步”中解释的相同 POSIX 信号量，但有一个区别：现在它们有名字，可以在整个系统中全局使用。换句话说，它们不再是*匿名*或*私有*的信号量了。

+   **命名互斥锁**：再次，与第十六章“线程同步”中解释的具有相同属性的相同 POSIX 互斥锁，但现在它们被命名，可以在整个系统中使用。这些互斥锁应放置在共享内存中，以便多个进程可以使用。

+   **命名条件变量**：与我们在第十六章“线程同步”中解释的相同 POSIX 条件变量，但像互斥锁一样，它们应放置在共享内存对象中，以便多个进程可以使用。

在接下来的章节中，我们将讨论所有上述技术，并给出示例以展示它们是如何工作的。在下一节中，我们将讨论命名 POSIX 信号量。

# 命名 POSIX 信号量

正如你在第十六章“线程同步”中看到的，信号量是同步多个并发任务的主要工具。我们在多线程程序中看到了它们，并看到了它们如何帮助克服并发问题。

在本节中，我们将展示它们如何在一些进程之间使用。*示例 18.1*展示了如何使用 POSIX 信号量来解决我们在上一章“进程执行”中给出的*示例 17.6*和*17.7*中遇到的数据竞争问题。该示例与*示例 17.6*非常相似，并且它再次使用共享内存区域来存储共享计数器变量。但它使用命名信号量来同步对共享计数器的访问。

以下代码框显示了我们在访问共享变量时使用命名信号量同步两个进程的方式。以下代码框显示了*示例 18.1*的全局声明：

```cpp
#include <stdio.h>
...
#include <semaphore.h>  // For using semaphores
#define SHARED_MEM_SIZE 4
// Shared file descriptor used to refer to the
// shared memory object
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
// The pointer to the shared semaphore
sem_t* semaphore = NULL;
```

代码框 18-1 [ExtremeC_examples_chapter18_1.c]：示例 18.1 的全局声明

在*代码框 18-1*中，我们声明了一个全局计数器和指向信号量对象的全球指针，该指针稍后将设置。这个指针将由父进程和子进程使用，以同步访问由计数器指针指向的共享计数器。

以下代码显示了预期执行实际进程同步的功能定义。其中一些定义与我们在*示例 17.6*中使用的相同，这些行已从以下代码框中删除：

```cpp
void init_control_mechanism() {
  semaphore = sem_open("/sem0", O_CREAT | O_EXCL, 0600, 1);
  if (semaphore == SEM_FAILED) {
    fprintf(stderr, "ERROR: Opening the semaphore failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void shutdown_control_mechanism() {
  if (sem_close(semaphore) < 0) {
    fprintf(stderr, "ERROR: Closing the semaphore failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (sem_unlink("/sem0") < 0) {
    fprintf(stderr, "ERROR: Unlinking failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void init_shared_resource() {
  ... as in the example 17.6 ...
}
void shutdown_shared_resource() {
  ... as in the example 17.6 ...
}
```

代码框 18-2 [ExtremeC_examples_chapter18_1.c]: 同步函数的定义

与 *示例 17.6* 相比，我们添加了两个新函数：`init_control_mechanism` 和 `shutdown_control_mechanism`。我们还对 `inc_counter` 函数（在 *代码框 18-3* 中显示）进行了修改，以使用信号量并在其中形成一个关键部分。

在 `init_control_mechanism` 和 `shutdown_control_mechanism` 函数内部，我们使用与共享内存 API 类似的 API 来打开、关闭和解除命名信号量的链接。

函数 `sem_open`、`sem_close` 和 `sem_unlink` 可以看作与 `shm_open`、`shm_close` 和 `shm_unlink` 类似。有一个区别，那就是 `sem_open` 函数返回一个信号量指针而不是文件描述符。

注意，在这个示例中用于处理信号量的 API 与我们之前看到的相同，因此其余的代码可以保持不变，就像 *示例 17.6* 一样。在这个示例中，信号量初始化为值 `1`，这使得它成为一个互斥锁。下面的代码框显示了关键部分以及如何使用信号量来同步对共享计数器执行的读写操作：

```cpp
void inc_counter() {
  usleep(1);
  sem_wait(semaphore); // Return value should be checked.
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  sem_post(semaphore); // Return value should be checked.
  usleep(1);
}
```

代码框 18-3 [ExtremeC_examples_chapter18_1.c]: 共享计数器增加的关键部分

与 *示例 17.6* 相比，在 `inc_counter` 函数中，使用 `sem_wait` 和 `sem_post` 函数分别进入和退出关键部分。

在下面的代码框中，你可以看到 `main` 函数。它与 *示例 17.6* 几乎相同，我们只看到初始和最终部分的一些变化，这符合在 *代码框 18-2* 中看到的两个新函数的添加：

```cpp
int main(int argc, char** argv) {
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Parent process needs to initialize the control mechanism
  init_control_mechanism();
  ... as in the example 17.6 ...
  // Only parent process needs to shut down the shared resource
  // and the employed control mechanism
  if (pid) {
    shutdown_shared_resource();
    shutdown_control_mechanism();
  }
  return 0;
}
```

代码框 18-4 [ExtremeC_examples_chapter18_1.c]: 示例 18.1 的主函数

在下面的 Shell 框中，你可以看到 *示例 18.1* 连续运行两次的输出：

```cpp
$ gcc ExtremeC_examples_chapter18_1.c -lrt -lpthread -o ex18_1.out
$ ./ex18_1.out
Shared memory is created with fd: 3
The memory region is truncated.
The child process sees the counter as 1.
The parent process sees the counter as 2.
The child process finished with status 0.
$ ./ex18_1.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 2.
The child process finished with status 0.
$
```

Shell 框 18-1：在 Linux 中构建并连续运行示例 18.1

注意，我们需要将上述代码与 `pthread` 库链接，因为我们正在使用 POSIX 信号量。我们还需要在 Linux 中将其与 `rt` 库链接，以便使用共享内存。

前面的输出是清晰的。有时子进程首先获得 CPU 并增加计数器，有时父进程这样做。它们从未同时进入关键部分，因此它们满足了共享计数器的数据完整性。

注意，使用命名信号量不需要使用 fork API。完全分离的进程，如果不是父子进程，如果它们在同一台机器上运行并在同一操作系统中，仍然可以打开和使用相同的信号量。在 *示例 18.3* 中，我们展示了这是如何实现的。

作为本节的最后一条注意事项，您应该知道在类 Unix 操作系统中，我们有两种类型的命名信号量。一种是**系统 V 信号量**，另一种是**POSIX 信号量**。在本节中，我们解释了 POSIX 信号量，因为它们因其良好的 API 和性能而享有更好的声誉。以下链接是一个 Stack Overflow 问题，它很好地解释了系统 V 信号量和 POSIX 信号量之间的区别：[`stackoverflow.com/questions/368322/differences-between-system-v-and-posix-semaphores`](https://stackoverflow.com/questions/368322/differences-between-system-v-and-posix-semaphores)

**注意：**

在使用信号量的方面，Microsoft Windows 不符合 POSIX 标准，并且它有自己的 API 来创建和管理信号量。

在下一节中，我们将讨论命名互斥锁。简而言之，命名互斥锁是将普通互斥锁对象放入共享内存区域。

# 命名互斥锁

POSIX 互斥锁在多线程程序中工作简单；我们在*第十六章*，*线程同步*中展示了这一点。然而，在多个进程环境中则不是这样。为了使互斥锁在多个进程之间工作，它需要在所有进程都可以访问的地方定义。

对于这样一个共享位置，最佳选择是共享内存区域。因此，为了在多进程环境中使用互斥锁，它应该分布在共享内存区域中。

## 第一个示例

以下示例，*示例 18.2*，是*示例 18.1*的一个克隆，但它使用命名互斥锁而不是命名信号量来解决潜在的竞态条件。它还展示了如何创建共享内存区域并使用它来存储共享互斥锁。

由于每个共享内存对象都有一个全局名称，存储在共享内存区域中的互斥锁可以被认为是**命名的**，并且可以通过系统中的其他进程访问。

以下代码框显示了*示例 18.2*所需的声明。它显示了需要共享互斥锁的内容：

```cpp
#include <stdio.h>
...
#include <pthread.h> // For using pthread_mutex_* functions
#define SHARED_MEM_SIZE 4
// Shared file descriptor used to refer to shared memory object
int shared_fd = -1;
// Shared file descriptor used to refer to the mutex's shared
// memory object
int mutex_shm_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
// The pointer to shared mutex
pthread_mutex_t* mutex = NULL;
```

代码框 18-5 [ExtremeC_examples_chapter18_2.c]：示例 18.2 的全局声明

如您所见，我们已经声明了：

+   一个全局文件描述符，用于指向一个存储共享计数器变量的共享内存区域

+   存储共享互斥锁的共享内存区域的全局文件描述符

+   共享计数器的指针

+   共享互斥锁的指针

这些变量将由即将到来的逻辑相应填充。

以下代码框显示了我们在*示例 18.1*中拥有的所有函数，但如您所见，定义已更新以使用命名互斥锁而不是命名信号量：

```cpp
void init_control_mechanism() {
  // Open the mutex shared memory
  mutex_shm_fd = shm_open("/mutex0", O_CREAT | O_RDWR, 0600);
  if (mutex_shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n"
        , strerror(errno));
    exit(1);
  }
  // Allocate and truncate the mutex's shared memory region
  if (ftruncate(mutex_shm_fd, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Truncation of mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  // Map the mutex's shared memory
  void* map = mmap(0, sizeof(pthread_mutex_t),
          PROT_READ | PROT_WRITE, MAP_SHARED, mutex_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  mutex = (pthread_mutex_t*)map;
  // Initialize the mutex object
  int ret = -1;
  pthread_mutexattr_t attr;
  if ((ret = pthread_mutexattr_init(&attr))) {
    fprintf(stderr, "ERROR: Failed to init mutex attrs: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutexattr_setpshared(&attr,
                  PTHREAD_PROCESS_SHARED))) {
    fprintf(stderr, "ERROR: Failed to set the mutex attr: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutex_init(mutex, &attr))) {
    fprintf(stderr, "ERROR: Initializing the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutexattr_destroy(&attr))) {
    fprintf(stderr, "ERROR: Failed to destroy mutex attrs : %s\n"
        , strerror(ret));
    exit(1);
  }
}
```

代码框 18-6 [ExtremeC_examples_chapter18_2.c]：示例 18.2 中的 init_control_mechanism 函数

作为函数`init_control_mechanism`的一部分，我们创建了一个名为`/mutex0`的新共享内存对象。共享内存区域的大小初始化为`sizeof(pthread_mutex_t)`，这表明我们的意图是在那里共享一个 POSIX 互斥锁对象。

接下来，我们得到共享内存区域的指针。现在我们有一个从共享内存分配的互斥锁，但它仍然需要初始化。因此，下一步是使用函数`pthread_mutex_init`初始化互斥锁对象，并使用属性指示互斥锁对象应该是共享的，并且可以被其他进程访问。这一点尤为重要；否则，即使在共享内存区域内部，互斥锁在多进程环境中也不会工作。正如你在前面的代码框中看到的，以及在函数`init_control_mechanism`中，我们已经设置了属性`PTHREAD_PROCESS_SHARED`来标记互斥锁为共享的。让我们看看下一个函数：

```cpp
void shutdown_control_mechanism() {
  int ret = -1;
  if ((ret = pthread_mutex_destroy(mutex))) {
    fprintf(stderr, "ERROR: Failed to destroy mutex: %s\n",
        strerror(ret));
    exit(1);
  }
  if (munmap(mutex, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (close(mutex_shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (shm_unlink("/mutex0") < 0) {
    fprintf(stderr, "ERROR: Unlinking the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
```

代码框 18-7 [ExtremeC_examples_chapter18_2.c]：示例 18.2 中的函数 destroy_control_mechanism

在函数`destroy_control_mechanism`中，我们销毁了互斥锁对象，然后关闭并解除链接其底层的共享内存区域。这与销毁一个普通共享内存对象的方式相同。让我们继续看示例中的其他代码：

```cpp
void init_shared_resource() {
  ... as in the example 18.1 ...
}
void shutdown_shared_resource() {
  ... as in the example 18.1 ...
}
```

代码框 18-8 [ExtremeC_examples_chapter18_2.c]：这些函数与我们之前在示例 18.1 中看到的是一样的

正如你所见，前面的函数完全没有改变，它们与我们之前在*示例 18.1*中看到的是一样的。让我们看看函数`inc_counter`内部的临界区，现在它使用命名互斥锁而不是命名信号量。

```cpp
void inc_counter() {
  usleep(1);
  pthread_mutex_lock(mutex); // Should check the return value.
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  pthread_mutex_unlock(mutex); // Should check the return value.
  usleep(1);
}
int main(int argc, char** argv) {
  ... as in the example 18.1 ...
}
```

代码框 18-9 [ExtremeC_examples_chapter18_2.c]：现在关键部分使用命名互斥锁来保护共享计数器

通常，正如你在前面的代码框中看到的，只有几个地方与*示例 18.1*不同，我们只对三个函数进行了重大修改。例如，函数`main`完全没有改变，它与*示例 18.1*中的相同。这仅仅是因为我们与*示例 18.1*相比使用了不同的控制机制，而其余的逻辑是相同的。

关于*代码框 18-9*的最后一项说明，在函数`inc_counter`中，我们使用了互斥对象，就像我们在多线程程序中所做的那样。API 是相同的，并且它被设计成可以在多线程和多进程环境中使用相同的 API 来使用互斥锁。这是 POSIX 互斥锁的一个伟大特性，因为它使我们能够在多线程和多进程环境中使用相同的代码来消费这些对象——当然，初始化和销毁可以不同。

上述代码的输出与我们观察到的 *示例 18.1* 非常相似。虽然在这个例子中共享计数器由互斥锁保护，但在上一个例子中它是由信号量保护的。上一个例子中使用的信号量实际上是一个二进制信号量，正如我们在 *第十六章*，*线程同步* 中所解释的，二进制信号量可以模拟互斥锁。因此，除了将二进制信号量替换为互斥锁之外，*示例 18.2* 中并没有太多新内容。

## 第二个示例

命名的共享内存和互斥锁可以被系统中的任何进程使用。不需要有派生的进程才能使用这些对象。下面的例子，*示例 18.3*，试图展示我们如何使用共享互斥锁和共享内存同时终止所有同时运行的进程。我们期望在按下其中一个进程的键组合 `Ctrl` + `C` 后，所有进程都将终止。

注意，代码将分多步提供。与每个步骤相关的注释将紧随其后。让我们先展示第一步。

### 步骤 1 – 全局声明

在这个例子中，我们编写一个可以编译和执行多次的单个源文件，以创建多个进程。这些进程使用一些共享内存区域来同步它们的执行。其中一个进程被选为共享内存区域的拥有者，并管理它们的创建和销毁。其他进程只是使用创建的共享内存。

第一步是声明一些我们在整个代码中需要的全局对象。我们将在代码的后面部分初始化它们。请注意，在以下代码框中定义的全局变量，如 `mutex`，实际上并不是在进程间共享的。它们在自己的内存空间中有这些变量，但每个进程都将自己的全局变量映射到位于各个共享内存区域中的对象或变量：

```cpp
#include <stdio.h>
...
#include <pthread.h> // For using pthread_mutex_* functions
typedef uint16_t bool_t;
#define TRUE 1
#define FALSE 0
#define MUTEX_SHM_NAME "/mutex0"
#define SHM_NAME "/shm0"
// Shared file descriptor used to refer to the shared memory
// object containing the cancel flag
int cancel_flag_shm_fd = -1;
// A flag which indicates whether the current process owns the
// shared memory object
bool_t cancel_flag_shm_owner = FALSE;
// Shared file descriptor used to refer to the mutex's shared
// memory object
int mutex_shm_fd = -1;
// The shared mutex
pthread_mutex_t* mutex = NULL;
// A flag which indicates whether the current process owns the
// shared memory object
bool_t mutex_owner = FALSE;
// The pointer to the cancel flag stored in the shared memory
bool_t* cancel_flag = NULL;
```

代码框 18-10 [ExtremeC_examples_chapter18_3.c]：示例 18.3 中的全局声明

在前面的代码中，我们可以看到代码中使用的全局声明。我们将使用一个共享标志让进程知道取消信号。请注意，在这个例子中，我们将采用忙等待的方法来等待取消标志变为 `true`。

我们有一个专门的共享内存对象用于取消标志，还有一个共享内存对象用于保护标志的互斥锁，就像我们在 *示例 18.2* 中做的那样。请注意，我们可以构造一个单一的结构，并将取消标志和互斥锁对象定义为它的字段，然后使用一个单一的共享内存区域来存储它们。但我们选择使用单独的共享内存区域来实现我们的目的。

在这个例子中，关于共享内存对象的一个重要注意事项是，清理工作应由最初创建和初始化它们的进程执行。由于所有进程都在使用相同的代码，我们 somehow 需要知道哪个进程创建了一个特定的共享内存对象，并使该进程成为该对象的拥有者。然后，在清理对象时，只有拥有者进程可以继续并进行实际的清理。因此，我们不得不为此目的声明了两个布尔变量：`mutex_owner`和`cancel_flag_shm_owner`。

### 第 2 步 - 取消标志的共享内存

以下代码框展示了专门用于取消标志的共享内存区域的初始化：

```cpp
void init_shared_resource() {
  // Open the shared memory object
  cancel_flag_shm_fd = shm_open(SHM_NAME, O_RDWR, 0600);
  if (cancel_flag_shm_fd >= 0) {
    cancel_flag_shm_owner = FALSE;
    fprintf(stdout, "The shared memory object is opened.\n");
  } else if (errno == ENOENT) {
    fprintf(stderr,
            "WARN: The shared memory object doesn't exist.\n");
    fprintf(stdout, "Creating the shared memory object ...\n");
    cancel_flag_shm_fd = shm_open(SHM_NAME,
            O_CREAT | O_EXCL | O_RDWR, 0600);
    if (cancel_flag_shm_fd >= 0) {
      cancel_flag_shm_owner = TRUE;
      fprintf(stdout, "The shared memory object is created.\n");
    } else {
      fprintf(stderr,
          "ERROR: Failed to create shared memory: %s\n",
          strerror(errno));
      exit(1);
    }
  } else {
      fprintf(stderr,
          "ERROR: Failed to create shared memory: %s\n",
          strerror(errno));
    exit(1);
  }
  if (cancel_flag_shm_owner) {
    // Allocate and truncate the shared memory region
    if (ftruncate(cancel_flag_shm_fd, sizeof(bool_t)) < 0) {
      fprintf(stderr, "ERROR: Truncation failed: %s\n",
              strerror(errno));
      exit(1);
    }
    fprintf(stdout, "The memory region is truncated.\n");
  }
  // Map the shared memory and initialize the cancel flag
  void* map = mmap(0, sizeof(bool_t), PROT_WRITE, MAP_SHARED,
      cancel_flag_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  cancel_flag = (bool_t*)map;
  if (cancel_flag_shm_owner) {
    *cancel_flag = FALSE;
  }
}
```

代码框 18-11 [ExtremeC_examples_chapter18_3.c]：取消标志共享内存的初始化

我们采取的方法与我们之前在*示例 18.2*中所做的方法不同。这是因为每当运行一个新的进程时，它都应该检查共享内存对象是否已经被另一个进程创建。请注意，在这个例子中，我们没有使用`fork` API 来创建新进程，用户可以使用他们的 shell 随意启动新进程。

因此，新进程首先尝试仅通过提供标志`O_RDWR`来打开共享内存区域。如果成功，则表明当前进程不是该区域的拥有者，然后它继续映射共享内存区域。如果失败，则表示共享内存区域不存在，这是当前进程应该创建该区域并成为其拥有者的一个指示。因此，它继续尝试以不同的标志打开区域；`O_CREAT`和`O_EXCL`。这些标志在不存在的情况下创建共享内存对象。

如果创建成功，则当前进程是拥有者，并且它继续通过截断和映射共享内存区域。

在前一个场景中，`shm_open`函数的两次连续调用之间，另一个进程可能会创建相同的共享内存区域，因此第二次`shm_open`调用失败。标志`O_EXCL`防止当前进程创建一个已经存在的对象，然后通过显示适当的错误消息退出。如果发生这种情况，这应该非常罕见，我们总是可以尝试再次运行该进程，并且在第二次运行中它不会遇到同样的问题。

以下代码是用于撤销取消标志及其共享内存区域的反向操作：

```cpp
void shutdown_shared_resource() {
  if (munmap(cancel_flag, sizeof(bool_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  if (close(cancel_flag_shm_fd) < 0) {
    fprintf(stderr,
        "ERROR: Closing the shared memory fd filed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (cancel_flag_shm_owner) {
    sleep(1);
    if (shm_unlink(SHM_NAME) < 0) {
      fprintf(stderr,
          "ERROR: Unlinking the shared memory failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
}
```

代码框 18-12 [ExtremeC_examples_chapter18_3.c]：关闭为取消标志共享内存分配的资源

如您在*代码框 18-12*中看到的，书写的逻辑与我们之前在释放共享内存对象的部分示例中看到的内容非常相似。但这里有一个区别，那就是只有所有者进程才能解除共享内存对象的链接。请注意，所有者进程在解除共享内存对象的链接之前会等待 1 秒，以便让其他进程完成资源释放。由于在大多数 POSIX 兼容系统中，共享内存对象会一直保留，直到所有依赖的进程退出，因此这种等待通常是不必要的。

### 第 3 步 – 命名互斥锁的共享内存

以下代码框显示了如何初始化共享互斥锁及其关联的共享内存对象：

```cpp
void init_control_mechanism() {
  // Open the mutex shared memory
  mutex_shm_fd = shm_open(MUTEX_SHM_NAME, O_RDWR, 0600);
  if (mutex_shm_fd >= 0) {
    // The mutex's shared object exists and I'm now the owner.
    mutex_owner = FALSE;
    fprintf(stdout,
            "The mutex's shared memory object is opened.\n");
  } else if (errno == ENOENT) {
    fprintf(stderr,
            "WARN: Mutex's shared memory doesn't exist.\n");
    fprintf(stdout,
            "Creating the mutex's shared memory object ...\n");
    mutex_shm_fd = shm_open(MUTEX_SHM_NAME,
            O_CREAT | O_EXCL | O_RDWR, 0600);
    if (mutex_shm_fd >= 0) {
      mutex_owner = TRUE;
      fprintf(stdout,
              "The mutex's shared memory object is created.\n");
    } else {
      fprintf(stderr,
          "ERROR: Failed to create mutex's shared memory: %s\n",
          strerror(errno));
      exit(1);
    }
  } else {
    fprintf(stderr,
        "ERROR: Failed to create mutex's shared memory: %s\n",
        strerror(errno));
    exit(1);
  }
  if (mutex_owner) {
    // Allocate and truncate the mutex's shared memory region
  }
  if (mutex_owner) {
    // Allocate and truncate the mutex's shared memory region
    if (ftruncate(mutex_shm_fd, sizeof(pthread_mutex_t)) < 0) {
      fprintf(stderr,
          "ERROR: Truncation of the mutex failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
  // Map the mutex's shared memory
  void* map = mmap(0, sizeof(pthread_mutex_t),
          PROT_READ | PROT_WRITE, MAP_SHARED, mutex_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  mutex = (pthread_mutex_t*)map;
  if (mutex_owner) {
    int ret = -1;
    pthread_mutexattr_t attr;
    if ((ret = pthread_mutexattr_init(&attr))) {
      fprintf(stderr,
          "ERROR: Initializing mutex attributes failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_setpshared(&attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR: Setting the mutex attribute failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutex_init(mutex, &attr))) {
      fprintf(stderr,
          "ERROR: Initializing the mutex failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_destroy(&attr))) {
      fprintf(stderr,
          "ERROR: Destruction of mutex attributes failed: %s\n",
          strerror(ret));
      exit(1);
    }
  }
}
```

代码框 18-13 [ExtremeC_examples_chapter18_3.c]：初始化共享互斥锁及其底层共享内存区域

与我们尝试创建与取消标志关联的共享内存区域时所做的操作类似，我们为创建和初始化共享互斥锁下的共享内存区域做了同样的事情。请注意，就像在 *示例 18.2* 中一样，互斥锁已被标记为 `PTHREAD_PROCESS_SHARED`，这允许它被多个进程使用。

以下代码框显示了如何最终化共享互斥锁：

```cpp
void shutdown_control_mechanism() {
  sleep(1);
  if (mutex_owner) {
    int ret = -1;
    if ((ret = pthread_mutex_destroy(mutex))) {
      fprintf(stderr,
          "WARN: Destruction of the mutex failed: %s\n",
          strerror(ret));
    }
  }
  if (munmap(mutex, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (close(mutex_shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (mutex_owner) {
    if (shm_unlink(MUTEX_SHM_NAME) < 0) {
      fprintf(stderr, "ERROR: Unlinking the mutex failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
}
```

代码框 18-14 [ExtremeC_examples_chapter18_3.c]：关闭共享互斥锁及其关联的共享内存区域

再次强调，所有者进程只能解除共享互斥锁的共享内存对象的链接。

### 第 4 步 – 设置取消标志

以下代码框显示了允许进程读取或设置取消标志的函数：

```cpp
bool_t is_canceled() {
  pthread_mutex_lock(mutex); // Should check the return value
  bool_t temp = *cancel_flag;
  pthread_mutex_unlock(mutex); // Should check the return value
  return temp;
}
void cancel() {
  pthread_mutex_lock(mutex); // Should check the return value
  *cancel_flag = TRUE;
  pthread_mutex_unlock(mutex); // Should check the return value
}
```

代码框 18-15 [ExtremeC_examples_chapter18_3.c]：受共享互斥锁保护的读取和设置取消标志的同步函数

前两个函数允许我们对共享取消标志进行同步访问。函数 `is_canceled` 用于检查标志的值，而函数 `cancel` 用于设置标志。如您所见，这两个函数都受到相同的共享互斥锁的保护。

### 第 5 步 – 主函数

最后，以下代码框显示了 `main` 函数和一个我们将简要解释的 *信号处理程序*：

```cpp
void sigint_handler(int signo) {
  fprintf(stdout, "\nHandling INT signal: %d ...\n", signo);
  cancel();
}
int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Parent process needs to initialize the control mechanism
  init_control_mechanism();
  while(!is_canceled()) {
    fprintf(stdout, "Working ...\n");
    sleep(1);
  }
  fprintf(stdout, "Cancel signal is received.\n");
  shutdown_shared_resource();
  shutdown_control_mechanism();
  return 0;
}
```

代码框 18-16 [ExtremeC_examples_chapter18_3.c]：示例 18.3 中的 main 函数和信号处理函数

如您所见，`main` 函数内部的逻辑清晰且直接。它初始化共享标志和互斥锁，然后进入忙等待状态，直到取消标志变为 `true`。最后，它关闭所有共享资源并终止。

这里新的是 `signal` 函数的使用，它将信号处理器分配给特定的 *信号* 集合。信号是所有 POSIX 兼容操作系统提供的一种设施，使用它可以，系统内的进程可以向彼此发送信号。*终端* 是用户与之交互的一个普通进程，它可以用来向其他进程发送信号。按下 `Ctrl` + `C` 是向终端中运行的前台进程发送 `SIGINT` 的一种方便方法。

`SIGINT` 是进程可以接收的 *中断信号*。在前面代码中，我们将函数 `sigint_handler` 分配为 `SIGINT` 信号的处理器。换句话说，每当进程接收到 `SIGINT` 信号时，函数 `sigint_handler` 将被调用。如果未处理 `SIGINT` 信号，则默认操作是终止进程，但可以使用如上所示的信号处理器来覆盖此操作。

向进程发送 `SIGINT` 信号有许多方法，但其中最简单的一种是在键盘上按下 `Ctrl` + `C` 键。进程将立即接收到 `SIGINT` 信号。正如你所见，在信号处理程序中，我们将共享取消标志设置为 `true`，从这一点开始，所有进程开始退出它们的忙等待循环。

以下是如何编译和运行前面代码的演示。让我们构建前面的代码并运行第一个进程：

```cpp
$ gcc ExtremeC_examples_chapter18_3.c -lpthread -lrt -o ex18_3.out
$ ./ex18_3.out
WARN: The shared memory object doesn't exist.
Creating a shared memory object ...
The shared memory object is created.
The memory region is truncated.
WARN: Mutex's shared memory object doesn't exist.
Creating the mutex's shared memory object ...
The mutex's shared memory object is created.
Working ...
Working ...
Working ...
```

Shell Box 18-2：编译示例 18.3 并运行第一个进程

正如你所见，前面的进程是首先运行的，因此它是互斥锁和取消标志的所有者。以下为第二个进程的运行情况：

```cpp
$ ./ex18_3.out
The shared memory object is opened.
The mutex's shared memory object is opened.
Working ...
Working ...
Working ...
```

Shell Box 18-3：运行第二个进程

正如你所见，第二个进程只打开了共享内存对象，它不是所有者。以下输出是在第一个进程上按下 `Ctrl` + `C` 后的输出：

```cpp
...
Working ...
Working ...
^C
Handling INT signal: 2 ...
Cancel signal is received.
$
```

Shell Box 18-4：按下 Ctrl + C 后第一个进程的输出

正如你所见，第一个进程打印出它正在处理编号为 `2` 的信号，这是 `SIGINT` 的标准信号编号。它设置了取消标志，并立即退出。随后，第二个进程退出。以下为第二个进程的输出：

```cpp
...
Working ...
Working ...
Working ...
Cancel signal is received.
$
```

Shell Box 18-5：当第二个进程看到取消标志被设置时的输出

此外，你也可以向第二个进程发送 `SIGINT` 信号，结果将相同；两个进程都会接收到信号并退出。你也可以创建超过两个进程，并且它们都将使用相同的共享内存和互斥锁同步退出。

在下一节中，我们将演示如何使用条件变量。就像命名互斥锁一样，如果你在共享内存区域中放置一个条件变量，它可以通过共享内存的名称被多个进程访问和使用。

# 命名条件变量

如我们之前所解释的，与命名 POSIX 互斥锁类似，为了在多进程系统中使用它，我们需要从共享内存区域分配一个 POSIX 条件变量。以下示例，*示例 18.4*，展示了如何这样做，以便让多个进程按特定顺序计数。正如您从 *第十六章*，*线程同步* 中所知，每个条件变量都应该与一个保护它的伴随互斥对象一起使用。因此，在 *示例 18.4* 中，我们将有三个共享内存区域；一个用于共享计数器，一个用于共享 *命名条件变量*，还有一个用于保护共享条件变量的共享 *命名互斥锁*。

注意，我们也可以使用单个共享内存而不是三个不同的共享内存。这是通过定义一个包含所有所需对象的结构的实现。在这个例子中，我们不会采取这种方法，我们将为每个对象定义一个单独的共享内存区域。

*示例 18.4* 是关于一些进程，它们应该按升序计数。每个进程都会被分配一个数字，从 1 开始，到进程的数量，给定的数字表示该进程在其他进程中的排名。进程必须等待排名（编号）较小的其他进程先计数，然后它才能计数并退出。当然，分配编号 1 的进程将首先计数，即使它是最后创建的进程。

由于我们将有三个不同的共享内存区域，每个区域都需要自己的初始化和终止步骤，如果我们想要采取与之前示例中相同的方法，那么我们将有大量的代码重复。为了减少我们编写的代码量，将重复的部分提取到一些函数中，并使代码组织得更好，我们将根据 *第六章*，*面向对象编程和封装*，*第七章*，*组合和聚合*，以及 *第八章*，*继承和多态* 中讨论的主题和程序，将其做成面向对象的。我们将以面向对象的方式编写 *示例 18.4*，并使用继承来减少重复代码的数量。

我们将为所有需要建立在共享内存区域之上的类定义一个父类。因此，在拥有父共享内存类的同时，我们将定义一个子类用于共享计数器，一个子类用于共享命名互斥锁，另一个子类用于共享命名条件变量。每个类都将有自己的头文件和源文件对，所有这些最终都将用于示例的主函数中。

以下章节将逐个介绍所提到的类。首先，让我们从父类：共享内存开始。

## 第 1 步 – 共享内存类

以下代码框显示了共享内存类的声明：

```cpp
struct shared_mem_t;
typedef int32_t bool_t; 
struct shared_mem_t* shared_mem_new();
void shared_mem_delete(struct shared_mem_t* obj);
void shared_mem_ctor(struct shared_mem_t* obj,
                     const char* name,
                     size_t size);
void shared_mem_dtor(struct shared_mem_t* obj);
char* shared_mem_getptr(struct shared_mem_t* obj);
bool_t shared_mem_isowner(struct shared_mem_t* obj);
void shared_mem_setowner(struct shared_mem_t* obj, bool_t is_owner);
```

代码框 18-17 [ExtremeC_examples_chapter18_4_shared_mem.h]: 共享内存类的公共接口

前面的代码包含了使用共享内存对象所需的所有声明（公共 API）。函数 `shared_mem_getptr`，`shared_mem_isowner` 和 `shared_mem_setowner` 是这个类的行为。

如果这个语法对你来说不熟悉，请阅读 *第六章*，*面向对象编程和封装*，*第七章*，*组合和聚合*，以及 *第八章*，*继承和多态*。

以下代码框展示了作为类公共接口一部分的函数定义，正如在 *代码框 18-17* 中所见：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#define TRUE 1
#define FALSE 0
typedef int32_t bool_t;
bool_t owner_process_set = FALSE;
bool_t owner_process = FALSE;
typedef struct {
  char* name;
  int shm_fd;
  void* map_ptr;
  char* ptr;
  size_t size;
} shared_mem_t;
shared_mem_t* shared_mem_new() {
  return (shared_mem_t*)malloc(sizeof(shared_mem_t));
}
void shared_mem_delete(shared_mem_t* obj) {
  free(obj->name);
  free(obj);
}
void shared_mem_ctor(shared_mem_t* obj, const char* name,
        size_t size) {
  obj->size = size;
  obj->name = (char*)malloc(strlen(name) + 1);
  strcpy(obj->name, name);
  obj->shm_fd = shm_open(obj->name, O_RDWR, 0600);
  if (obj->shm_fd >= 0) {
    if (!owner_process_set) {
      owner_process = FALSE;
      owner_process_set = TRUE;
    }
    printf("The shared memory %s is opened.\n", obj->name);
  } else if (errno == ENOENT) {
    printf("WARN: The shared memory %s does not exist.\n",
            obj->name);
    obj->shm_fd = shm_open(obj->name,
            O_CREAT | O_RDWR, 0600);
    if (obj->shm_fd >= 0) {
      if (!owner_process_set) {
        owner_process = TRUE;
        owner_process_set = TRUE;
      }
      printf("The shared memory %s is created and opened.\n",
              obj->name);
      if (ftruncate(obj->shm_fd, obj->size) < 0) {
        fprintf(stderr, "ERROR(%s): Truncation failed: %s\n",
            obj->name, strerror(errno));
        exit(1);
      }
    } else {
      fprintf(stderr,
          "ERROR(%s): Failed to create shared memory: %s\n",
          obj->name, strerror(errno));
      exit(1);
    }
  } else {
      fprintf(stderr,
          "ERROR(%s): Failed to create shared memory: %s\n",
          obj->name, strerror(errno));
    exit(1);
  }
  obj->map_ptr = mmap(0, obj->size, PROT_READ | PROT_WRITE,
      MAP_SHARED, obj->shm_fd, 0);
  if (obj->map_ptr == MAP_FAILED) {
    fprintf(stderr, "ERROR(%s): Mapping failed: %s\n",
        name, strerror(errno));
    exit(1);
  }
  obj->ptr = (char*)obj->map_ptr;
}
void shared_mem_dtor(shared_mem_t* obj) {
  if (munmap(obj->map_ptr, obj->size) < 0) {
    fprintf(stderr, "ERROR(%s): Unmapping failed: %s\n",
        obj->name, strerror(errno));
    exit(1);
  }
  printf("The shared memory %s is unmapped.\n", obj->name);
  if (close(obj->shm_fd) < 0) {
    fprintf(stderr,
        "ERROR(%s): Closing the shared memory fd failed: %s\n",
        obj->name, strerror(errno));
    exit(1);
  }
  printf("The shared memory %s is closed.\n", obj->name);
  if (owner_process) {
    if (shm_unlink(obj->name) < 0) {
      fprintf(stderr,
          "ERROR(%s): Unlinking the shared memory failed: %s\n",
          obj->name, strerror(errno));
      exit(1);
    }
    printf("The shared memory %s is deleted.\n", obj->name);
  }
}
char* shared_mem_getptr(shared_mem_t* obj) {
  return obj->ptr;
}
bool_t shared_mem_isowner(shared_mem_t* obj) {
  return owner_process;
}
void shared_mem_setowner(shared_mem_t* obj, bool_t is_owner) {
    owner_process = is_owner;
}
```

代码框 18-18 [ExtremeC_examples_chapter18_4_shared_mem.c]: 共享内存类中找到的所有函数的定义

如你所见，我们只是复制了之前示例中为共享内存编写的代码。结构 `shared_mem_t` 封装了我们用来访问 POSIX 共享内存对象所需的所有内容。注意全局布尔变量 `process_owner`。它表示当前进程是否是所有共享内存区域的拥有者。它只设置一次。

## 第 2 步 – 共享 32 位整数计数器类

以下代码框包含共享计数器类的声明，这是一个 32 位整数计数器。这个类从共享内存类继承。正如你可能已经注意到的，我们正在使用 *第八章*，*继承和多态* 中描述的第二种方法来实现继承关系：

```cpp
struct shared_int32_t;
struct shared_int32_t* shared_int32_new();
void shared_int32_delete(struct shared_int32_t* obj);
void shared_int32_ctor(struct shared_int32_t* obj,
                       const char* name);
void shared_int32_dtor(struct shared_int32_t* obj);
void shared_int32_setvalue(struct shared_int32_t* obj,
                           int32_t value);
void shared_int32_setvalue_ifowner(struct shared_int32_t* obj,
                                   int32_t value);
int32_t shared_int32_getvalue(struct shared_int32_t* obj);
```

代码框 18-19 [ExtremeC_examples_chapter18_4_shared_int32.h]: 共享计数器类的公共接口

以下代码框展示了前面声明的函数的实现：

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
typedef struct {
  struct shared_mem_t* shm;
  int32_t* ptr;
} shared_int32_t;
shared_int32_t* shared_int32_new(const char* name) {
  shared_int32_t* obj =
      (shared_int32_t*)malloc(sizeof(shared_int32_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_int32_delete(shared_int32_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_int32_ctor(shared_int32_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(int32_t));
  obj->ptr = (int32_t*)shared_mem_getptr(obj->shm);
}
void shared_int32_dtor(shared_int32_t* obj) {
  shared_mem_dtor(obj->shm);
}
void shared_int32_setvalue(shared_int32_t* obj, int32_t value) {
  *(obj->ptr) = value;
}
void shared_int32_setvalue_ifowner(shared_int32_t* obj,
                                   int32_t value) {
  if (shared_mem_isowner(obj->shm)) {
    *(obj->ptr) = value;
  }
}
int32_t shared_int32_getvalue(shared_int32_t* obj) {
  return *(obj->ptr);
}
```

代码框 18-20 [ExtremeC_examples_chapter18_4_shared_int32.c]: 共享计数器类中找到的所有函数的定义

如你所见，由于继承，我们编写了更少的代码。管理相关共享内存对象所需的所有代码都通过结构 `shared_int32_t` 中的字段 `shm` 带入。

## 第 3 步 – 共享互斥器类

以下代码框包含共享互斥器类的声明：

```cpp
#include <pthread.h>
struct shared_mutex_t;
struct shared_mutex_t* shared_mutex_new();
void shared_mutex_delete(struct shared_mutex_t* obj);
void shared_mutex_ctor(struct shared_mutex_t* obj,
                       const char* name);
void shared_mutex_dtor(struct shared_mutex_t* obj);
pthread_mutex_t* shared_mutex_getptr(struct shared_mutex_t* obj);
void shared_mutex_lock(struct shared_mutex_t* obj);
void shared_mutex_unlock(struct shared_mutex_t* obj);
#if !defined(__APPLE__)
void shared_mutex_make_consistent(struct shared_mutex_t* obj);
#endif
```

代码框 18-21 [ExtremeC_examples_chapter18_4_shared_mutex.h]: 共享互斥器类的公共接口

如你所见，上述类有三种预期的公开行为；`shared_mutex_lock`，`shared_mutex_unlock` 和 `shared_mutex_make_consistent`。但有一个例外，即行为 `shared_mutex_make_consistent` 只在 POSIX 系统中可用，不包括基于 macOS（苹果）的系统。这是因为苹果系统不支持 *健壮互斥锁*。我们将在接下来的段落中讨论什么是健壮互斥锁。请注意，我们使用了宏 `__APPLE__` 来检测我们是否在苹果系统上编译。

以下代码框展示了前面类实现的代码：

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
typedef struct {
  struct shared_mem_t* shm;
  pthread_mutex_t* ptr;
} shared_mutex_t;
shared_mutex_t* shared_mutex_new() {
  shared_mutex_t* obj =
      (shared_mutex_t*)malloc(sizeof(shared_mutex_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_mutex_delete(shared_mutex_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_mutex_ctor(shared_mutex_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(pthread_mutex_t));
  obj->ptr = (pthread_mutex_t*)shared_mem_getptr(obj->shm);
  if (shared_mem_isowner(obj->shm)) {
    pthread_mutexattr_t mutex_attr;
    int ret = -1;
    if ((ret = pthread_mutexattr_init(&mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing mutex attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
#if !defined(__APPLE__)
    if ((ret = pthread_mutexattr_setrobust(&mutex_attr,
                    PTHREAD_MUTEX_ROBUST))) {
      fprintf(stderr,
          "ERROR(%s): Setting the mutex as robust failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
#endif
    if ((ret = pthread_mutexattr_setpshared(&mutex_attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR(%s): Failed to set as process-shared: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutex_init(obj->ptr, &mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing the mutex failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_destroy(&mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Destruction of mutex attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
  }
}
void shared_mutex_dtor(shared_mutex_t* obj) {
  if (shared_mem_isowner(obj->shm)) {
    int ret = -1;
    if ((ret = pthread_mutex_destroy(obj->ptr))) {
      fprintf(stderr,
          "WARN: Destruction of the mutex failed: %s\n",
          strerror(ret));
    }
  }
  shared_mem_dtor(obj->shm);
}
pthread_mutex_t* shared_mutex_getptr(shared_mutex_t* obj) {
  return obj->ptr;
}
#if !defined(__APPLE__)
void shared_mutex_make_consistent(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_consistent(obj->ptr))) {
    fprintf(stderr,
        "ERROR: Making the mutex consistent failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
#endif
void shared_mutex_lock(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_lock(obj->ptr))) {
#if !defined(__APPLE__)
    if (ret == EOWNERDEAD) {
        fprintf(stderr,
                "WARN: The owner of the mutex is dead ...\n");
        shared_mutex_make_consistent(obj);
        fprintf(stdout, "INFO: I'm the new owner!\n");
        shared_mem_setowner(obj->shm, TRUE);
        return;
    }
#endif
    fprintf(stderr, "ERROR: Locking the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
void shared_mutex_unlock(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_unlock(obj->ptr))) {
    fprintf(stderr, "ERROR: Unlocking the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
```

代码框 18-22 [ExtremeC_examples_chapter18_4_shared_mutex.c]：共享命名互斥类中找到的所有函数的定义

在前面的代码中，我们只进行了 POSIX 互斥锁的初始化、终止和暴露一些简单的行为，例如锁定和解锁。与共享内存对象相关的所有其他事情都在共享内存类中处理。这就是使用继承的好处。

注意，在构造函数`shared_mutex_ctor`中，我们将互斥锁设置为*共享进程*互斥锁，使其对所有进程可访问。这对于多进程软件来说是绝对必要的。注意，在非苹果系统上，我们更进一步，将互斥锁配置为*健壮互斥锁*。

对于被进程锁定的普通互斥锁，如果进程突然死亡，则互斥锁进入非一致状态。对于健壮互斥锁，如果发生这种情况，互斥锁可以被放回一致状态。下一个通常等待互斥锁的进程只能通过使其一致来锁定互斥锁。你可以在`shared_mutex_lock`函数中看到如何做到这一点。注意，这种功能在苹果系统中不存在。

## 第 4 步 – 共享条件变量类

以下代码框显示了共享条件变量类的声明：

```cpp
struct shared_cond_t;
struct shared_mutex_t;
struct shared_cond_t* shared_cond_new();
void shared_cond_delete(struct shared_cond_t* obj);
void shared_cond_ctor(struct shared_cond_t* obj,
                      const char* name);
void shared_cond_dtor(struct shared_cond_t* obj);
void shared_cond_wait(struct shared_cond_t* obj,
                      struct shared_mutex_t* mutex);
void shared_cond_timedwait(struct shared_cond_t* obj,
                           struct shared_mutex_t* mutex,
                           long int time_nanosec);
void shared_cond_broadcast(struct shared_cond_t* obj);
```

代码框 18-23 [ExtremeC_examples_chapter18_4_shared_cond.h]：共享条件变量类的公共接口

暴露了三种行为；`shared_cond_wait`、`shared_cond_timedwait`和`shared_cond_broadcast`。如果你还记得第十六章，即*线程同步*，`shared_cond_wait`行为会在条件变量上等待信号。

上面，我们添加了一个新的等待行为版本；`shared_cond_timedwait`。它等待指定时间内的信号，如果条件变量没有收到信号，则超时。另一方面，`shared_cond_wait`只有在收到某种信号时才会存在。我们将在*示例 18.4*中使用等待的定时版本。注意，这两个等待行为函数都接收一个指向伴随共享互斥锁的指针，就像我们在多线程环境中看到的那样。

以下代码框包含了共享条件变量类的实际实现：

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
#include "ExtremeC_examples_chapter18_4_shared_mutex.h"
typedef struct {
  struct shared_mem_t* shm;
  pthread_cond_t* ptr;
} shared_cond_t;
shared_cond_t* shared_cond_new() {
  shared_cond_t* obj =
      (shared_cond_t*)malloc(sizeof(shared_cond_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_cond_delete(shared_cond_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_cond_ctor(shared_cond_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(pthread_cond_t));
  obj->ptr = (pthread_cond_t*)shared_mem_getptr(obj->shm);
  if (shared_mem_isowner(obj->shm)) {
    pthread_condattr_t cond_attr;
    int ret = -1;
    if ((ret = pthread_condattr_init(&cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing cv attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_condattr_setpshared(&cond_attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR(%s): Setting as process shared failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_cond_init(obj->ptr, &cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing the cv failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_condattr_destroy(&cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Destruction of cond attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
  }
}
void shared_cond_dtor(shared_cond_t* obj) {
  if (shared_mem_isowner(obj->shm)) {
    int ret = -1;
    if ((ret = pthread_cond_destroy(obj->ptr))) {
      fprintf(stderr, "WARN: Destruction of the cv failed: %s\n",
          strerror(ret));
    }
  }
  shared_mem_dtor(obj->shm);
}
void shared_cond_wait(shared_cond_t* obj,
                      struct shared_mutex_t* mutex) {
  int ret = -1;
  if ((ret = pthread_cond_wait(obj->ptr,
                  shared_mutex_getptr(mutex)))) {
    fprintf(stderr, "ERROR: Waiting on the cv failed: %s\n",
            strerror(ret));
    exit(1);
  }
}
void shared_cond_timedwait(shared_cond_t* obj,
                           struct shared_mutex_t* mutex,
                           long int time_nanosec) {
  int ret = -1;
  struct timespec ts;
  ts.tv_sec = ts.tv_nsec = 0;
  if ((ret = clock_gettime(CLOCK_REALTIME, &ts))) {
    fprintf(stderr,
            "ERROR: Failed at reading current time: %s\n",
            strerror(errno));
    exit(1);
  }
  ts.tv_sec += (int)(time_nanosec / (1000L * 1000 * 1000));
  ts.tv_nsec += time_nanosec % (1000L * 1000 * 1000);
  if ((ret = pthread_cond_timedwait(obj->ptr,
                  shared_mutex_getptr(mutex), &ts))) {
#if !defined(__APPLE__)
    if (ret == EOWNERDEAD) {
      fprintf(stderr,
              "WARN: The owner of the cv's mutex is dead ...\n");
      shared_mutex_make_consistent(mutex);
      fprintf(stdout, "INFO: I'm the new owner!\n");
      shared_mem_setowner(obj->shm, TRUE);
      return;
    } else if (ret == ETIMEDOUT) {
#else
    if (ret == ETIMEDOUT) {
#endif
      return;
    }
    fprintf(stderr, "ERROR: Waiting on the cv failed: %s\n",
            strerror(ret));
    exit(1);
  }
}
void shared_cond_broadcast(shared_cond_t* obj) {
  int ret = -1;
  if ((ret = pthread_cond_broadcast(obj->ptr))) {
    fprintf(stderr, "ERROR: Broadcasting on the cv failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
```

代码框 18-24 [ExtremeC_examples_chapter18_4_shared_cond.c]：共享条件变量类中找到的所有函数的定义

在我们的共享条件变量类中，我们只暴露了*广播*行为。我们也可以暴露*信号*行为。正如你可能从第十六章，即*线程同步*中记得的，向条件变量发送信号只会唤醒许多等待进程中的一个，而没有能力指定或预测是哪一个。相比之下，广播会唤醒所有等待进程。在*示例 18.4*中，我们只会使用广播，这就是为什么我们只暴露了那个函数。

注意，由于每个条件变量都有一个伴随的互斥锁，共享互斥锁类应该能够使用共享互斥锁类的实例，这就是为什么我们将`shared_mutex_t`声明为前向声明的原因。

## 第 5 步 – 主要逻辑

下面的代码框包含了为我们示例实现的主要逻辑：

```cpp
#include "ExtremeC_examples_chapter18_4_shared_int32.h"
#include "ExtremeC_examples_chapter18_4_shared_mutex.h"
#include "ExtremeC_examples_chapter18_4_shared_cond.h"
int int_received = 0;
struct shared_cond_t* cond = NULL;
struct shared_mutex_t* mutex = NULL;
void sigint_handler(int signo) {
  fprintf(stdout, "\nHandling INT signal: %d ...\n", signo);
  int_received = 1;
}
int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);
  if (argc < 2) {
    fprintf(stderr,
            "ERROR: You have to provide the process number.\n");
    exit(1);
  }
  int my_number = atol(argv[1]);
  printf("My number is %d!\n", my_number);
  struct shared_int32_t* counter = shared_int32_new();
  shared_int32_ctor(counter, "/counter0");
  shared_int32_setvalue_ifowner(counter, 1);
  mutex = shared_mutex_new();
  shared_mutex_ctor(mutex, "/mutex0");
  cond = shared_cond_new();
  shared_cond_ctor(cond, "/cond0");
  shared_mutex_lock(mutex);
  while (shared_int32_getvalue(counter) < my_number) {
    if (int_received) {
        break;
    }
    printf("Waiting for the signal, just for 5 seconds ...\n");
    shared_cond_timedwait(cond, mutex, 5L * 1000 * 1000 * 1000);
    if (int_received) {
        break;
    }
    printf("Checking condition ...\n");
  }
  if (int_received) {
    printf("Exiting ...\n");
    shared_mutex_unlock(mutex);
    goto destroy;
  }
  shared_int32_setvalue(counter, my_number + 1);
  printf("My turn! %d ...\n", my_number);
  shared_mutex_unlock(mutex);
  sleep(1);
  // NOTE: The broadcasting can come after unlocking the mutex.
  shared_cond_broadcast(cond);
destroy:
  shared_cond_dtor(cond);
  shared_cond_delete(cond);
  shared_mutex_dtor(mutex);
  shared_mutex_delete(mutex);
  shared_int32_dtor(counter);
  shared_int32_delete(counter);
  return 0;
}
```

代码框 18-25 [ExtremeC_examples_chapter18_4_main.c]：示例 18.4 的主函数

如您所见，程序接受一个参数来指示其数字。一旦进程得知其数字，它就开始初始化共享计数器、共享互斥锁和共享条件变量。然后它进入由共享互斥锁保护的临界区。

在循环内部，它等待计数器等于其数字。由于它等待 5 秒钟，可能会出现超时，我们可能在 5 秒后离开`shared_cond_timedwait`函数。这基本上意味着在这 5 秒钟内没有通知条件变量。然后进程再次检查条件，并再次休眠 5 秒钟。这个过程会一直持续到进程获得轮次。

当发生这种情况时，进程打印其数字，增加共享计数器，并通过在共享条件变量对象上广播信号，通知其他等待进程它对共享计数器所做的修改。然后它才准备退出。

同时，如果用户按下`Ctrl` + `C`，作为主逻辑一部分定义的信号处理程序将设置局部标志`int_received`，并且一旦进程在主循环中离开`shared_mutex_timedwait`函数，它就会注意到中断信号并退出循环。

下面的 shell box 展示了如何编译*示例 18.4*。我们将在 Linux 上编译它：

```cpp
$ gcc -c ExtremeC_examples_chapter18_4_shared_mem.c -o shared_mem.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_int32.c -o shared_int32.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_mutex.c -o shared_mutex.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_cond.c -o shared_cond.o
$ gcc -c ExtremeC_examples_chapter18_4_main.c -o main.o
$ gcc shared_mem.o shared_int32.o shared_mutex.o shared_cond.o \  main.o -lpthread -lrt -o ex18_4.out
$
```

Shell Box 18-6：编译示例 18.4 的源代码并生成最终的可执行文件

现在我们已经得到了最终的可执行文件`ex18_4.out`，我们可以运行三个进程并观察它们如何按顺序计数，无论您如何分配它们的数字以及它们的运行顺序如何。让我们运行第一个进程。我们通过将数字作为选项传递给可执行文件来给这个进程分配数字 3：

```cpp
$ ./ex18_4.out 3
My number is 3!
WARN: The shared memory /counter0 does not exist.
The shared memory /counter0 is created and opened.
WARN: The shared memory /mutex0 does not exist.
The shared memory /mutex0 is created and opened.
WARN: The shared memory /cond0 does not exist.
The shared memory /cond0 is created and opened.
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
```

Shell Box 18-7：运行第一个进程，该进程取数字 3

正如您在前面的输出中看到的，第一个进程创建了所有必需的共享对象，并成为共享资源的所有者。现在，让我们在另一个终端中运行第二个进程。它取数字 2：

```cpp
$ ./ex18_4.out 2
My number is 2!
The shared memory /counter0 is opened.
The shared memory /mutex0 is opened.
The shared memory /cond0 is opened.
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
```

Shell Box 18-8：运行第二个进程，该进程取数字 2

最后，最后一个进程取数字 1。由于这个进程被分配了数字 1，它立即打印其数字，增加共享计数器，并通知其他进程这一情况：

```cpp
$ ./ex18_4.out 1
My number is 1!
The shared memory /counter0 is opened.
The shared memory /mutex0 is opened.
The shared memory /cond0 is opened.
My turn! 1 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
$
```

Shell Box 18-9：运行第三个进程，该进程取数字 1。由于它具有数字 1，这个进程将立即退出。

现在，如果你回到第二个进程，它打印出其编号，增加共享计数器，并通知第三个进程：

```cpp
...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
My turn! 2 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
$
```

Shell Box 18-10：第二个进程打印其编号并退出

最后，回到第一个进程，它被第二个进程通知，然后打印出其编号并退出。

```cpp
...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
My turn! 3 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /cond0 is deleted.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /mutex0 is deleted.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
The shared memory /counter0 is deleted.
$
```

Shell Box 18-11：第一个进程打印其编号并退出。它还删除了所有共享内存条目。

由于第一个进程是所有共享内存的所有者，它应该在退出时删除它们。在多进程环境中释放分配的资源可能相当复杂，因为一个简单的错误就足以导致所有进程崩溃。当要从系统中移除共享资源时，需要进一步的同步。

假设在前面的例子中，我们用数字 2 运行第一个进程，用数字 3 运行第二个进程。因此，第一个进程应该在第二个进程之前打印其编号。当第一个进程由于它是所有共享资源的创建者而退出时，它删除共享对象，而第二个进程在试图访问它们时立即崩溃。

这只是一个简单的例子，说明了在多进程系统中，终止操作可能会变得复杂且问题重重。为了减轻这种崩溃的风险，需要在进程间引入进一步的同步。

在前面的章节中，我们介绍了可以用于同步同一主机上运行的多个进程的机制。在接下来的章节中，我们将简要讨论分布式并发控制机制及其特性。

# 分布式并发控制

到目前为止，在本章中，我们假设所有进程都存在于同一个操作系统内，也就是说，同一个机器。换句话说，我们一直在谈论单主机软件系统。

但实际的软件系统通常超出了这一点。与单主机软件系统相反，我们有分布式软件系统。这些系统在网络中分布有进程，并且通过网络通信来运行。

关于进程的分布式系统，我们可以看到一些在集中式或单主机系统中不那么明显的挑战。接下来，我们将简要讨论其中的一些：

+   **在分布式软件系统中，你可能会遇到并行性而不是并发性**。由于每个进程运行在单独的机器上，并且每个进程都有自己的特定处理器，我们将观察到并行性而不是并发性。并发通常局限于单个机器的边界。请注意，交错仍然存在，我们可能会遇到与并发系统相同的非确定性。

+   **并非分布式软件系统中的所有流程都是使用单一编程语言编写的**。在分布式软件系统中看到各种编程语言被使用是很常见的。在单主机软件系统的流程中，也常常看到这种多样性。尽管我们对系统内流程的隐含假设是它们都使用 C 语言编写，但我们仍然可以使用任何其他语言来编写流程。不同的语言提供了不同的并发和控制机制。例如，在某些语言中，你可能很难轻松地使用命名互斥锁。在软件系统中使用的各种技术和编程语言，无论是单主机还是分布式，都迫使我们使用足够抽象的并发控制机制，以便在所有这些系统中都可用。这可能会限制我们只能使用在特定技术或编程语言中可用的特定同步技术。

+   **在分布式系统中，你总是有一个网络作为不在同一台机器上的两个进程之间的通信通道**。这与我们对单主机系统的隐含假设相反，在单主机系统中，所有流程都在同一操作系统中运行，并使用可用的消息基础设施相互通信。

+   **中间存在网络意味着你会有延迟**。在单主机系统中也存在轻微的延迟，但它是可以确定和管理的。它也比你在网络中可能遇到的延迟要低得多。延迟简单地说就是，由于许多原因，一个进程可能不会立即收到消息，这些原因根植于网络基础设施。在这些系统中，没有什么应该被认为是即时的。

+   **中间存在网络也会导致安全问题**。当你有一个系统中的所有流程，并且它们都在使用具有极低延迟的机制在同一边界内进行通信时，安全问题就大不相同了。攻击者必须首先访问系统本身才能攻击系统，但在分布式系统中，所有消息传递都是通过网络进行的。你可能会在中间遇到一个监听者来窃听或，更糟糕的是，篡改消息。关于我们在分布式系统中关于同步的讨论，这也适用于旨在同步分布式系统内流程的消息。

+   **除了延迟和安全问题之外，你可能会遇到在单主机多进程系统中远较少发生的交付问题**。消息应该被传递以进行处理。当一个进程向系统内的另一个进程发送消息时，发送进程应确保其消息被另一端接收。*交付保证*机制是可能的，但它们成本高昂，在某些情况下，甚至根本无法使用它们。在这些情况下，会出现一种特殊的消息问题，这通常由著名的*两个将军问题*来建模。

前面的差异和可能的问题足以迫使我们发明新的进程和大型分布式系统各个组件之间的同步方式。通常，有两种方式可以使分布式系统事务性和同步：

+   **集中式进程同步**：这些技术需要一个中心进程（或节点）来管理进程。系统中的所有其他进程都应该与这个中心节点保持持续通信，并且它们需要其批准才能进入它们的临界区。

+   **分布式（或对等）进程同步**：拥有一个没有中心节点的进程同步基础设施并不是一件容易的事情。这实际上是一个活跃的研究领域，并且有一些专门的算法。

在本节中，我们试图对分布式多进程系统中并发控制的复杂性进行一些解释。关于分布式并发控制的进一步讨论将超出本书的范围。

# 摘要

在本章中，我们完成了关于多进程环境的讨论。作为本章的一部分，我们讨论了以下内容：

+   什么是命名信号量以及它是如何被多个进程创建和使用的。

+   什么是命名互斥锁以及它是如何通过共享内存区域使用的。

+   我们给出了一个关于终止编排的例子，其中多个进程正在等待一个终止信号，信号被其中一个进程接收和处理，然后传播给其他进程。我们使用共享互斥锁实现了这个例子。

+   什么是命名条件变量以及它是如何通过共享内存区域实现共享和命名的。

+   我们展示了另一个计数进程的例子。作为这个例子的一部分，我们使用了继承来减少具有相关共享内存区域的互斥锁和条件变量对象的代码重复量。

+   我们简要探讨了分布式系统中存在的差异和挑战。

+   我们简要讨论了可以将并发控制引入分布式软件的方法。

在即将到来的章节中，我们开始讨论**进程间通信**（**IPC**）技术。我们的讨论将涵盖两个章节，我们将涉及许多主题，例如计算机网络、传输协议、套接字编程以及更多有用的主题。
