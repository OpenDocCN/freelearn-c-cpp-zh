# 第十六章

# 线程同步

在上一章中，我们解释了如何创建和管理 POSIX 线程。我们还演示了两种最常见的并发问题：竞态条件和数据竞争。

在本章中，我们将继续讨论使用 POSIX 线程库进行多线程编程，并为您提供控制多个线程所需的技术。

如果您还记得第十四章“同步”，我们展示了与并发相关的问题实际上并不是问题；相反，它们是并发系统基本属性的后果。因此，您很可能在任何并发系统中都会遇到它们。

我们在上一章中展示了我们确实可以使用 POSIX 线程库产生这些问题。上一章的*示例 15.2*和*15.3*演示了竞态条件和数据竞争问题。因此，它们将成为我们使用 pthread 库提供的同步机制来同步多个线程的起点。

在本章中，我们将涵盖以下主题：

+   使用 POSIX 互斥锁来保护访问共享资源的临界区。

+   使用 POSIX 条件变量等待特定条件。

+   使用各种类型的锁与互斥锁和条件变量一起使用。

+   使用 POSIX 障碍（barriers）及其如何帮助同步多个线程。

+   信号量的概念及其在 pthread 库中的对应对象：POSIX 信号量。您将发现互斥锁实际上是二进制信号量。

+   线程的内存结构和这种结构如何影响多核系统中的内存可见性。

我们从本章开始，先对并发控制进行一般性讨论。接下来的几节将为您提供编写良好行为的多线程程序所需的工具和结构。

## POSIX 并发控制

在本节中，我们将探讨 pthread 库提供的可能控制机制。信号量、互斥锁和条件变量以及不同类型的锁以各种组合使用，以使多线程程序具有确定性。首先，我们从 POSIX 互斥锁开始。

### POSIX 互斥锁

pthread 库中引入的互斥锁可以用于同步进程和线程。在本节中，我们将使用它们在多线程 C 程序中同步多个线程。

作为提醒，互斥锁是一种信号量，它一次只允许一个线程进入临界区。通常，信号量有让多个线程进入其临界区的潜力。

**注意**：

互斥锁也被称为*二进制信号量*，因为它们是只接受两种状态的信号量。

我们从解决前一章中作为*示例 15.3*一部分观察到的数据竞争问题开始本节，使用 POSIX 互斥锁。互斥锁一次只允许一个线程进入临界区，并对共享变量执行读写操作。这样，它保证了共享变量的数据完整性。以下代码框包含了解决数据竞争问题的解决方案：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
// The mutex object used to synchronize the access to
// the shared state.
pthread_mutex_t mtx;
void* thread_body_1(void* arg) {
  // Obtain a pointer to the shared variable
  int* shared_var_ptr = (int*)arg;
  // Critical section
  pthread_mutex_lock(&mtx);
  (*shared_var_ptr)++;
  printf("%d\n", *shared_var_ptr);
  pthread_mutex_unlock(&mtx);
  return NULL;
}
void* thread_body_2(void* arg) {
  int* shared_var_ptr = (int*)arg;
  // Critical section
  pthread_mutex_lock(&mtx);
  *shared_var_ptr += 2;
  printf("%d\n", *shared_var_ptr);
  pthread_mutex_unlock(&mtx);
  return NULL;
}
int main(int argc, char** argv) {
  // The shared variable
  int shared_var = 0;
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  // Initialize the mutex and its underlying resources
  pthread_mutex_init(&mtx, NULL);
  // Create new threads
  int result1 = pthread_create(&thread1, NULL,
          thread_body_1, &shared_var);
  int result2 = pthread_create(&thread2, NULL,
          thread_body_2, &shared_var);
  if (result1 || result2) {
    printf("The threads could not be created.\n");
    exit(1);
  }
  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  if (result1 || result2) {
    printf("The threads could not be joined.\n");
    exit(2);
  }
  pthread_mutex_destroy(&mtx);
  return 0;
}
```

代码框 16-1 [ExtremeC_examples_chapter15_3_mutex.c]：使用 POSIX 互斥锁解决前一章中作为示例 15.3 一部分发现的数据竞争问题

如果你编译前面的代码并运行多次，你将只看到输出中的`1 3`或`2 3`。那是因为我们正在使用 POSIX 互斥锁对象来同步前面代码中的临界区。

在文件开头，我们已声明一个全局 POSIX 互斥锁对象作为`mtx`。然后在`main`函数中，我们使用`pthread_mutex_init`函数使用默认属性初始化互斥锁。第二个参数是`NULL`，可以是程序员指定的自定义属性。我们将在接下来的章节中通过一个示例来了解如何设置这些属性。

互斥锁在两个线程中都被用来保护由`pthread_mutex_lock(&mtx)`和`pthread_mutex_unlock(&mtx)`语句包围的临界区。

最后，在离开`main`函数之前，我们销毁互斥锁对象。

在伴随函数`thread_body_1`中的第一对`pthread_mutex_lock(&mtx)`和`pthread_mutex_unlock(&mtx)`语句，构成了第一个线程的临界区。同样，伴随函数`thread_body_2`中的第二对构成了第二个线程的临界区。这两个临界区都由互斥锁保护，并且每次只有一个线程可以在其临界区中，而其他线程应该在临界区外等待，直到忙线程离开。

一旦一个线程进入临界区，它就会锁定互斥锁，而其他线程应该在`pthread_mutex_lock(&mtx)`语句后面等待，直到互斥锁再次解锁。

默认情况下，等待互斥锁解锁的线程会进入睡眠模式，并且不会进行*忙等待*。但如果我们想进行*忙等待*而不是进入睡眠状态呢？那么我们可以使用*自旋锁*。只需要使用以下函数代替所有前面的互斥锁相关函数即可。幸运的是，pthread 在函数命名上使用了一致的约定。

与自旋锁相关的类型和函数如下。

+   `pthread_spin_t`：用于创建自旋锁对象的类型。它类似于`pthread_mutex_t`类型。

+   `pthread_spin_init`：初始化一个自旋锁对象。它类似于`pthread_mutex_init`。

+   `pthread_spin_destroy`：类似于`pthread_mutex_destory`。

+   `pthread_spin_lock`：类似于`pthread_mutex_lock`。

+   `pthread_spin_unlock`：类似于`pthread_mutex_unlock`。

如你所见，只需用自旋锁类型和函数替换前面的互斥锁类型和函数，就可以很容易地实现不同的行为，在等待互斥锁对象释放时进行忙等待。

在本节中，我们介绍了 POSIX 互斥锁及其如何用于解决数据竞争问题。在下一节中，我们将演示如何使用条件变量来等待某个事件的发生。我们将解决在**示例 15.2**中发生的竞争条件，但我们将对原始示例进行一些修改。

### POSIX 条件变量

如果你还记得上一章中的**示例 15.2**，我们遇到了竞争条件。现在，我们想要提出一个新的例子，它与**示例 15.2**非常相似，但在这个例子中，使用条件变量会更简单。*示例 16.1*有两个线程而不是三个（这是**示例 15.2**的情况），它们需要将字符`A`和`B`打印到输出，但我们希望它们始终按照特定的顺序；首先`A`然后是`B`。

我们这个例子中的不变约束是要在输出中**首先看到 A 然后看到 B**（以及所有共享状态的数据完整性，没有坏内存访问，没有悬垂指针，没有崩溃，以及其他明显的约束）。以下代码演示了我们如何使用条件变量来为这个例子提供一个用 C 语言编写的解决方案：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
#define TRUE  1
#define FALSE 0
typedef unsigned int bool_t;
// A structure for keeping all the variables related
// to a shared state
typedef struct {
  // The flag which indicates whether 'A' has been printed or not
  bool_t          done;
  // The mutex object protecting the critical sections
  pthread_mutex_t mtx;
  // The condition variable used to synchronize two threads
  pthread_cond_t  cv;
} shared_state_t;
// Initializes the members of a shared_state_t object
void shared_state_init(shared_state_t *shared_state) {
  shared_state->done = FALSE;
  pthread_mutex_init(&shared_state->mtx, NULL);
  pthread_cond_init(&shared_state->cv, NULL);
}
// Destroy the members of a shared_state_t object
void shared_state_destroy(shared_state_t *shared_state) {
  pthread_mutex_destroy(&shared_state->mtx);
  pthread_cond_destroy(&shared_state->cv);
}
void* thread_body_1(void* arg) {
  shared_state_t* ss = (shared_state_t*)arg;
  pthread_mutex_lock(&ss->mtx);
  printf("A\n");
  ss->done = TRUE;
  // Signal the threads waiting on the condition variable
  pthread_cond_signal(&ss->cv);
  pthread_mutex_unlock(&ss->mtx);
  return NULL;
}
void* thread_body_2(void* arg) {
  shared_state_t* ss = (shared_state_t*)arg;
  pthread_mutex_lock(&ss->mtx);
  // Wait until the flag becomes TRUE
  while (!ss->done) {
    // Wait on the condition variable
    pthread_cond_wait(&ss->cv, &ss->mtx);
  }
  printf("B\n");
  pthread_mutex_unlock(&ss->mtx);
  return NULL;
}
int main(int argc, char** argv) {
  // The shared state
  shared_state_t shared_state;
  // Initialize the shared state
  shared_state_init(&shared_state);
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  // Create new threads
  int result1 =
    pthread_create(&thread1, NULL, thread_body_1, &shared_state);
  int result2 =
    pthread_create(&thread2, NULL, thread_body_2, &shared_state);
  if (result1 || result2) {
    printf("The threads could not be created.\n");
    exit(1);
  }
  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  if (result1 || result2) {
    printf("The threads could not be joined.\n");
    exit(2);
  }
  // Destroy the shared state and release the mutex
  // and condition variable objects
  shared_state_destroy(&shared_state);
  return 0;
}
```

代码框 16-2 [ExtremeC_examples_chapter16_1_cv.c]：使用 POSIX 条件变量来指定两个线程之间的特定顺序

在前面的代码中，使用一个结构体来封装共享互斥锁、共享条件变量和共享标志是一个好主意。请注意，我们只能为每个线程传递一个指针。因此，我们必须将所需的共享变量堆叠到一个单独的结构体变量中。

在示例中的第二个类型定义（在`bool_t`之后），我们定义了一个新的类型，`shared_state_t`，如下所示：

```cpp
typedef struct {
  bool_t          done;
  pthread_mutex_t mtx;
  pthread_cond_t  cv;
} shared_state_t;
```

代码框 16-3：将示例 16.1 所需的所有共享变量放入一个结构体中

在类型定义之后，我们定义了两个函数来初始化和销毁`shared_state_t`实例。它们可以被认为是类型`shared_state_t`的**构造函数**和**析构函数**。要了解更多关于构造函数和析构函数的信息，请参阅*第六章*，*面向对象编程和封装*。

这就是使用条件变量的方法。一个线程可以在条件变量上**等待**（或**睡眠**），然后在将来，它会被通知醒来。不仅如此，一个线程还可以**通知**（或**唤醒**）所有其他在条件变量上等待（或睡眠）的线程。所有这些操作**必须**由互斥锁保护，这就是为什么你应该始终将条件变量与互斥锁一起使用。

我们在前面代码中也做了同样的事情。在我们的共享状态对象中，我们有一个条件变量，以及一个应该保护条件变量的伴随互斥锁。再次强调，条件变量应该只在由其伴随互斥锁保护的临界区中使用。

那么，前面的代码中发生了什么？在应该打印 `A` 的线程中，它尝试使用指向共享状态对象的指针来锁定 `mtx` 互斥锁。当锁被获取后，线程打印 `A`，设置标志 `done`，并最终通过调用 `pthread_cond_signal` 函数通知其他线程，该线程可能正在等待条件变量 `cv`。

另一方面，如果在此时第二个线程变得活跃，而第一个线程还没有打印 `A`，第二个线程将尝试获取 `mtx` 上的锁。如果成功，它会检查标志 `done`，如果它是假的，这仅仅意味着第一个线程还没有进入其临界区（否则标志应该是真的）。因此，第二个线程在条件变量上等待，并通过调用 `pthread_cond_wait` 函数立即释放 CPU。

非常重要的是要注意，在等待条件变量时，相关的互斥锁被释放，其他线程可以继续。同样，在变得活跃并退出等待状态后，应该再次获取相关的互斥锁。对于条件变量的良好实践，你可以查看其他可能的交错情况。

**注意**：

函数 `pthread_cond_signal` 只能用来通知单个线程。如果你要通知所有等待条件变量的线程，你必须使用 `pthread_cond_broadcast` 函数。我们很快就会给出一个例子。

但为什么我们使用 `while` 循环来检查标志 `done`，而不是一个简单的 `if` 语句呢？那是因为第二个线程可以由其他来源而不是仅仅由第一个线程通知。在这些情况下，如果线程在退出等待状态并再次变得活跃时能够获取其互斥锁，它可以检查循环的条件，如果条件尚未满足，它应该再次等待。在循环中等待条件变量是一种可接受的技术，直到其条件匹配我们等待的内容。

前面的解决方案也满足了内存可见性约束。正如我们在前面的章节中解释的，所有锁定和解锁操作都可能触发各个 CPU 核之间的内存一致性；因此，在标志 `done` 的不同缓存版本中看到的值总是最新且相同的。

在例子 15.2 和 16.1 中观察到的竞争条件问题（在没有控制机制的情况下），也可以使用 POSIX 障碍来解决。在下一节中，我们将讨论它们，并使用不同的方法重写 *例子 16.1*。

### POSIX 障碍

POSIX 屏障使用不同的方法来同步多个线程。就像一群人计划并行执行一些任务，并在某些时刻需要会合、重组并继续一样，线程（甚至进程）也可能发生类似的情况。有些线程完成任务更快，而有些则较慢。但是，可以有一个检查点（或会合点），所有线程都必须停止并等待其他线程加入。这些检查点可以通过使用*POSIX 屏障*来模拟。

以下代码使用屏障来解决在*示例 16.1*中看到的问题。作为提醒，在*示例 16.1*中，我们有两个线程。其中一个线程是`打印 A`，另一个线程是`打印 B`，我们希望无论各种交错如何，输出中总是先看到`A`然后是`B`：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
// The barrier object
pthread_barrier_t barrier;
void* thread_body_1(void* arg) {
  printf("A\n");
  // Wait for the other thread to join
  pthread_barrier_wait(&barrier);
  return NULL;
}
void* thread_body_2(void* arg) {
  // Wait for the other thread to join
  pthread_barrier_wait(&barrier);
  printf("B\n");
  return NULL;
}
int main(int argc, char** argv) {
  // Initialize the barrier object
  pthread_barrier_init(&barrier, NULL, 2);
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  // Create new threads
  int result1 = pthread_create(&thread1, NULL,
          thread_body_1, NULL);
  int result2 = pthread_create(&thread2, NULL,
          thread_body_2, NULL);
  if (result1 || result2) {
    printf("The threads could not be created.\n");
    exit(1);
  }
  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  if (result1 || result2) {
    printf("The threads could not be joined.\n");
    exit(2);
  }
  // Destroy the barrier object
  pthread_barrier_destroy(&barrier);
  return 0;
}
```

代码框 16-4 [ExtremeC_examples_chapter16_1_barrier.c]：使用 POSIX 屏障解决示例 16.1 的解决方案

如您所见，前面的代码比我们使用条件变量编写的代码要小得多。使用 POSIX 屏障，在执行过程中的一些特定点同步一些线程会非常容易。

首先，我们声明了一个全局屏障对象，其类型为`pthread_barrier_t`。然后，在`main`函数内部，我们使用`pthread_barrier_init`函数初始化了屏障对象。

第一个参数是屏障对象的指针。第二个参数是屏障对象的自定义属性。由于我们传递了`NULL`，这意味着屏障对象将使用其属性的默认值进行初始化。第三个参数很重要；它是通过调用`pthread_barrier_wait`函数应该等待在同一个屏障对象上的线程数，只有在这之后，它们才会全部释放并被允许继续。

对于前面的例子，我们将其设置为 2。因此，只有当有两个线程在等待屏障对象时，它们才会被解锁并继续执行。其余的代码与前面的例子非常相似，并在上一节中进行了解释。

可以使用互斥锁和条件变量来实现屏障对象，就像我们在上一节中所做的那样。事实上，符合 POSIX 规范的操作系统在其系统调用接口中并不提供屏障这样的东西，而大多数实现都是使用互斥锁和条件变量来完成的。

这基本上是为什么像 macOS 这样的操作系统不提供 POSIX 屏障的实现。前面的代码在 macOS 机器上无法编译，因为 POSIX 屏障函数未定义。前面的代码已在 Linux 和 FreeBSD 上测试，并在两者上都能正常工作。因此，在使用屏障时要小心，因为使用它们会使你的代码的可移植性降低。

macOS 不提供 POSIX 屏障函数的事实仅仅意味着它是部分符合 POSIX 标准的，使用屏障的程序（当然，这是标准）无法在 macOS 机器上编译。这与 C 哲学相悖，即*一次编写，到处编译*。

作为本节的最后一条注释，POSIX 屏障保证了内存可见性。与锁定和解锁操作类似，等待屏障确保在离开屏障点时，各个线程中相同变量的所有缓存版本都同步。

在下一节中，我们将给出一个关于信号量的示例。它们在并发开发中并不常用，但它们有自己的特殊用途。

一种特定的信号量类型，二进制信号量（可以互换地称为互斥锁），经常被使用，你已经在前面的章节中看到了许多相关的例子。

### POSIX 信号量

在大多数情况下，互斥锁（或*二进制信号量*）足以同步访问共享资源的多个线程。这是因为，为了使读写操作顺序进行，一次只能允许一个线程进入临界区。这被称为*互斥*，因此称为“mutex”。

然而，在某些情况下，你可能希望有多个线程进入临界区并操作共享资源。这就是你应该使用*通用信号量*的场景。

在我们进入关于通用信号量的示例之前，让我们先举一个关于二进制信号量（或互斥锁）的例子。在这个例子中，我们不会使用`pthread_mutex_*`函数；相反，我们将使用`sem_*`函数，这些函数旨在公开与信号量相关的功能。

#### 二进制信号量

以下代码是使用信号量解决*示例 15.3*的解决方案。提醒一下，它涉及两个线程；每个线程以不同的值递增共享整数。我们想要保护共享变量的数据完整性。注意，在以下代码中我们不会使用 POSIX 互斥锁：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
// The semaphores are not exposed through pthread.h
#include <semaphore.h>
// The main pointer addressing a semaphore object used
// to synchronize the access to the shared state.
sem_t *semaphore;
void* thread_body_1(void* arg) {
  // Obtain a pointer to the shared variable
  int* shared_var_ptr = (int*)arg;
  // Waiting for the semaphore
  sem_wait(semaphore);
  // Increment the shared variable by 1 by writing directly
  // to its memory address
  (*shared_var_ptr)++;
  printf("%d\n", *shared_var_ptr);
  // Release the semaphore
  sem_post(semaphore);
  return NULL;
}
void* thread_body_2(void* arg) {
  // Obtain a pointer to the shared variable
  int* shared_var_ptr = (int*)arg;
  // Waiting for the semaphore
  sem_wait(semaphore);
  // Increment the shared variable by 1 by writing directly
  // to its memory address
  (*shared_var_ptr) += 2;
  printf("%d\n", *shared_var_ptr);
  // Release the semaphore
  sem_post(semaphore);
  return NULL;
}
int main(int argc, char** argv) {
  // The shared variable
  int shared_var = 0;
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
#ifdef __APPLE__
  // Unnamed semaphores are not supported in OS/X. Therefore
  // we need to initialize the semaphore like a named one using
  // sem_open function.
  semaphore = sem_open("sem0", O_CREAT | O_EXCL, 0644, 1);
#else
  sem_t local_semaphore;
  semaphore = &local_semaphore;
  // Initiliaze the semaphore as a mutex (binary semaphore)
  sem_init(semaphore, 0, 1);
#endif
  // Create new threads
  int result1 = pthread_create(&thread1, NULL,
          thread_body_1, &shared_var);
  int result2 = pthread_create(&thread2, NULL,
          thread_body_2, &shared_var);
  if (result1 || result2) {
    printf("The threads could not be created.\n");
    exit(1);
  }
  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  if (result1 || result2) {
    printf("The threads could not be joined.\n");
    exit(2);
  }
#ifdef __APPLE__
  sem_close(semaphore);
#else
  sem_destroy(semaphore);
#endif
  return 0;
}
```

代码框 16-5 [ExtremeC_examples_chapter15_3_sem.c]：使用 POSIX 信号量解决示例 15.3 的解决方案

你可能会首先注意到前面代码中我们使用的不同信号量函数。在 Apple 操作系统（macOS、OS X 和 iOS）中，*未命名信号量*不受支持。因此，我们无法直接使用`sem_init`和`sem_destroy`函数。未命名信号量没有名称（令人惊讶的是），它们只能在进程内部，由多个线程使用。另一方面，命名信号量是系统范围的，可以在系统中的各个进程中看到和使用。

在 Apple 系统中，创建未命名信号量所需的函数已被标记为已弃用，并且信号量对象不会被`sem_init`初始化。因此，我们不得不使用`sem_open`和`sem_close`函数来定义命名信号量。

命名信号量用于同步进程，我们将在*第十八章*，*进程同步*中解释它们。在其他 POSIX 兼容的操作系统上，特别是 Linux，我们仍然可以使用无名称信号量，并使用`sem_init`和`sem_destroy`函数分别初始化和销毁它们。

在前面的代码中，我们包含了一个额外的头文件，`semaphore.h`。正如我们之前解释的，信号量作为 POSIX 线程库的扩展被添加，因此它们不是作为`pthread.h`头文件的一部分公开。

在头文件包含语句之后，我们声明了一个指向信号量对象的全局指针。这个指针将指向一个适当的地址，该地址指向实际的信号量对象。在这里我们必须使用指针，因为在 Apple 系统中，我们必须使用`sem_open`函数，该函数返回一个指针。

然后，在`main`函数内部，在 Apple 系统中，我们创建了一个命名的信号量`sem0`。在其他 POSIX 兼容的操作系统上，我们使用`sem_init`初始化信号量。请注意，在这种情况下，指针`semaphore`指向在主线程栈上分配的变量`local_semaphore`。由于主线程不会退出并等待线程通过连接它们来完成，因此`semaphore`指针不会成为悬空指针。

注意，我们可以通过使用宏`__APPLE__`来区分 Apple 和非 Apple 系统。这是一个在 Apple 系统中默认由 C 预处理器定义的宏。因此，我们可以通过使用这个宏来排除不应该在 Apple 系统上编译的代码。

让我们看看线程内部的情况。在伴随函数中，关键部分由`sem_wait`和`sem_post`函数保护，这些函数分别对应于 POSIX 互斥锁 API 中的`pthread_mutex_lock`和`pthread_mutex_unlock`函数。请注意，`sem_wait`可能允许多个线程进入关键部分。

允许在关键部分中的最大线程数是在初始化信号量对象时确定的。我们将`1`作为最大线程数传递给`sem_open`和`sem_init`函数的最后一个参数；因此，信号量应该表现得像互斥锁。

为了更好地理解信号量，让我们更深入地探讨一下细节。每个信号量对象都有一个整数值。每当一个线程通过调用`sem_wait`函数等待信号量时，如果信号量的值大于零，则该值减 1，线程被允许进入关键部分。如果信号量的值为 0，线程必须等待直到信号量的值再次变为正数。每当一个线程通过调用`sem_post`函数退出关键部分时，信号量的值增加 1。因此，通过指定初始值`1`，我们最终会得到一个二进制信号量。

我们通过调用`sem_destroy`（或在 Apple 系统中使用`sem_close`）来结束前面的代码，这实际上释放了信号量对象及其所有底层资源。至于命名信号量，由于它们可以在多个进程之间共享，关闭信号量时可能会出现更复杂的场景。我们将在第十八章*进程同步*中讨论这些场景。

#### 通用信号量

现在，是时候给出一个使用通用信号量的经典例子了。其语法与前面的代码非常相似，但允许多个线程进入临界区的场景可能很有趣。

这个经典例子涉及了 50 个水分子的创建。对于 50 个水分子，你需要有 50 个氧原子和 100 个氢原子。如果我们用线程模拟每个原子，我们需要两个氢线程和一个氧线程进入它们的临界区，以便生成一个水分子并对其进行计数。

在下面的代码中，我们首先创建了 50 个氧线程和 100 个氢线程。为了保护氧线程的临界区，我们使用互斥锁，但对于氢线程的临界区，我们使用允许两个线程同时进入临界区的通用信号量。

对于信号量，我们使用 POSIX 屏障，但由于屏障在 Apple 系统中没有实现，我们需要使用互斥锁和条件变量来实现它们。下面的代码框包含了相应的代码：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h> // For errno and strerror function
// The POSIX standard header for using pthread library
#include <pthread.h>
// Semaphores are not exposed through pthread.h
#include <semaphore.h>
#ifdef __APPLE__
// In Apple systems, we have to simulate the barrier functionality.
pthread_mutex_t barrier_mutex;
pthread_cond_t  barrier_cv;
unsigned int    barrier_thread_count;
unsigned int    barrier_round;
unsigned int    barrier_thread_limit;
void barrier_wait() {
  pthread_mutex_lock(&barrier_mutex);
  barrier_thread_count++;
  if (barrier_thread_count >= barrier_thread_limit) {
    barrier_thread_count = 0;
    barrier_round++;
    pthread_cond_broadcast(&barrier_cv);
  } else {
    unsigned int my_round = barrier_round;
    do {
      pthread_cond_wait(&barrier_cv, &barrier_mutex);
    } while (my_round == barrier_round);
  }
  pthread_mutex_unlock(&barrier_mutex);
}
#else
// A barrier to make hydrogen and oxygen threads synchronized
pthread_barrier_t water_barrier;
#endif
// A mutex in order to synchronize oxygen threads
pthread_mutex_t   oxygen_mutex;
// A general semaphore to make hydrogen threads synchronized
sem_t*            hydrogen_sem;
// A shared integer counting the number of made water molecules
unsigned int      num_of_water_molecules;
void* hydrogen_thread_body(void* arg) {
  // Two hydrogen threads can enter this critical section
  sem_wait(hydrogen_sem);
  // Wait for the other hydrogen thread to join
#ifdef __APPLE__
  barrier_wait();
#else
  pthread_barrier_wait(&water_barrier);
#endif
  sem_post(hydrogen_sem);
  return NULL;
}
void* oxygen_thread_body(void* arg) {
  pthread_mutex_lock(&oxygen_mutex);
  // Wait for the hydrogen threads to join
#ifdef __APPLE__
  barrier_wait();
#else
  pthread_barrier_wait(&water_barrier);
#endif
  num_of_water_molecules++;
  pthread_mutex_unlock(&oxygen_mutex);
  return NULL;
}
int main(int argc, char** argv) {
  num_of_water_molecules = 0;
  // Initialize oxygen mutex
  pthread_mutex_init(&oxygen_mutex, NULL);
  // Initialize hydrogen semaphore
#ifdef __APPLE__
  hydrogen_sem = sem_open("hydrogen_sem",
          O_CREAT | O_EXCL, 0644, 2);
#else
  sem_t local_sem;
  hydrogen_sem = &local_sem;
  sem_init(hydrogen_sem, 0, 2);
#endif
  // Initialize water barrier
#ifdef __APPLE__
  pthread_mutex_init(&barrier_mutex, NULL);
  pthread_cond_init(&barrier_cv, NULL);
  barrier_thread_count = 0;
  barrier_thread_limit = 0;
  barrier_round = 0;
#else
  pthread_barrier_init(&water_barrier, NULL, 3);
#endif
  // For creating 50 water molecules, we need 50 oxygen atoms and
  // 100 hydrogen atoms
  pthread_t thread[150];
  // Create oxygen threads
  for (int i = 0; i < 50; i++) {
    if (pthread_create(thread + i, NULL,
                oxygen_thread_body, NULL)) {
      printf("Couldn't create an oxygen thread.\n");
      exit(1);
    }
  }
  // Create hydrogen threads
  for (int i = 50; i < 150; i++) {
    if (pthread_create(thread + i, NULL,
                hydrogen_thread_body, NULL)) {
      printf("Couldn't create an hydrogen thread.\n");
      exit(2);
    }
  }
  printf("Waiting for hydrogen and oxygen atoms to react ...\n");
  // Wait for all threads to finish
  for (int i = 0; i < 150; i++) {
    if (pthread_join(thread[i], NULL)) {
      printf("The thread could not be joined.\n");
      exit(3);
    }
  }
  printf("Number of made water molecules: %d\n",
          num_of_water_molecules);
#ifdef __APPLE__
  sem_close(hydrogen_sem);
#else
  sem_destroy(hydrogen_sem);
#endif
  return 0;
}
```

代码框 16-6 [ExtremeC_examples_chapter16_2.c]：使用通用信号量模拟从 50 个氧原子和 100 个氢原子中创建 50 个水分子的过程

在代码的开始部分，有一些被`#ifdef __APPLE__`和`#endif`包围的行。这些行只在 Apple 系统中编译。这些行主要是模拟 POSIX 屏障行为的实现和变量。在其他除了 Apple 之外的 POSIX 兼容系统中，我们使用普通的 POSIX 屏障。在这里我们不会深入讲解 Apple 系统中屏障实现的细节，但阅读并彻底理解代码是很有价值的。

作为前面代码中定义的多个全局变量的一部分，我们声明了互斥锁`oxygen_mutex`，它应该保护氧线程的临界区。在任何时候，只有一个氧线程（或氧原子）可以进入临界区。

然后在它的临界区中，一个氧线程等待两个其他氢线程加入，然后它继续增加水分子计数器。增加操作发生在氧的临界区内部。

为了更详细地解释在关键部分内部发生的事情，我们需要解释通用信号量的作用。在前面的代码中，我们已声明了通用信号量 `hydrogen_sem`，它应该用来保护氢线程的关键部分。在任何时候，最多只能有两个氢线程进入它们的关键部分，并且它们在氧气和氢线程之间共享的屏障对象上等待。

当等待在共享屏障对象上的线程数量达到两个时，这意味着我们有一个氧原子和两个氢原子，然后 voilà：一个水分子就形成了，所有等待的线程都可以继续。氢线程立即退出，但氧线程只有在增加水分子计数器之后才会退出。

我们以这个最后的笔记结束本节。在 *示例 16.2* 中，我们在为苹果系统实现屏障时使用了 `pthread_cond_broadcast` 函数。它向所有等待屏障条件变量的线程发出信号，这些线程在其他线程加入后应该继续执行。

在下一节中，我们将讨论 POSIX 线程背后的内存模型以及它们如何与它们所属进程的内存交互。我们还将查看使用栈和堆段的示例以及它们如何导致一些严重的内存相关问题。

## POSIX 线程和内存

本节将讨论线程与进程内存之间的交互。正如你所知，进程的内存布局中有多个段。文本段、栈段、数据段和堆段都是这个内存布局的一部分，我们在 *第四章*，*进程内存结构* 中讨论了它们。线程与这些内存段中的每一个都有不同的交互。作为本节的一部分，我们只讨论栈和堆内存区域，因为它们是在编写多线程程序时最常用且最容易出现问题的区域。

此外，我们讨论了线程同步以及真正理解线程背后的内存模型如何帮助我们开发更好的并发程序。这些概念在堆内存方面尤为明显，因为那里的内存管理是手动的，并且在并发系统中，线程负责分配和释放堆块。一个简单的竞态条件可能导致严重的内存问题，因此必须实施适当的同步以避免此类灾难。

在下一小节中，我们将解释不同的线程如何访问栈段以及应该采取哪些预防措施。

### 栈内存

每个线程都有自己的栈区域，这个区域应该是仅对该线程私有的。线程的栈区域是所属进程的栈段的一部分，并且默认情况下，所有线程都应该从栈段分配其栈区域。也有可能线程有一个从堆段分配的栈区域。我们将在未来的例子中展示如何做到这一点，但到目前为止，我们假设线程的栈是进程的栈段的一部分。

由于同一进程中的所有线程都可以读取和修改进程的栈段，因此它们可以有效地读取和修改彼此的栈区域，但它们*不应该*这样做。请注意，与其他线程的栈区域一起工作被认为是一种危险的行为，因为定义在各个栈区域顶部的变量可能随时被释放，尤其是在线程退出或函数返回时。

正是因为这个原因，我们试图假设一个栈区域只能被其所属线程访问，而不能被其他线程访问。因此，*局部变量*（那些在栈顶部声明的变量）被认为是线程私有的，不应该被其他线程访问。

在单线程应用程序中，我们始终只有一个线程，即主线程。因此，我们像使用进程的栈段一样使用其栈区域。这是因为，在单线程程序中，主线程和进程本身之间没有界限。但对于多线程程序来说，情况就不同了。每个线程都有自己的栈区域，这个区域与其他线程的栈区域不同。

在创建新线程时，会为栈区域分配一个内存块。如果程序员在创建时没有指定，栈区域将具有默认的栈大小，并且它将从进程的栈段中分配。默认栈大小是平台相关的，并且因架构而异。您可以使用命令`ulimit -s`在 POSIX 兼容系统中检索默认栈大小。

在我当前的平台上，这是一个基于 Intel 64 位机器的 macOS，默认栈大小是 8 MB：

```cpp
$ ulimit -s
8192
$
```

Shell Box 16-1：读取默认栈大小

POSIX 线程 API 允许你为新的线程设置栈区域。在下面的例子，*示例 16.3*中，我们有两个线程。对于其中一个线程，我们使用默认的栈设置，而对于另一个线程，我们将从堆段分配一个缓冲区并将其设置为该线程的栈区域。请注意，在设置栈区域时，分配的缓冲区应该有一个最小大小；否则它不能用作栈区域：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
void* thread_body_1(void* arg) {
  int local_var = 0;
  printf("Thread1 > Stack Address: %p\n", (void*)&local_var);
  return 0;
}
void* thread_body_2(void* arg) {
  int local_var = 0;
  printf("Thread2 > Stack Address: %p\n", (void*)&local_var);
  return 0;
}
int main(int argc, char** argv) {
  size_t buffer_len = PTHREAD_STACK_MIN + 100;
  // The buffer allocated from heap to be used as
  // the thread's stack region
  char *buffer = (char*)malloc(buffer_len * sizeof(char));
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  // Create a new thread with default attributes
  int result1 = pthread_create(&thread1, NULL,
          thread_body_1, NULL);
  // Create a new thread with a custom stack region
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  // Set the stack address and size
  if (pthread_attr_setstack(&attr, buffer, buffer_len)) {
    printf("Failed while setting the stack attributes.\n");
    exit(1);
  }
  int result2 = pthread_create(&thread2, &attr,
          thread_body_2, NULL);
  if (result1 || result2) {
    printf("The threads could not be created.\n");
    exit(2);
  }
  printf("Main Thread > Heap Address: %p\n", (void*)buffer);
  printf("Main Thread > Stack Address: %p\n", (void*)&buffer_len);
  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  if (result1 || result2) {
    printf("The threads could not be joined.\n");
    exit(3);
  }
  free(buffer);
  return 0;
}
```

Code Box 16-7 [ExtremeC_examples_chapter16_3.c]：将堆块设置为线程的栈区域

要启动程序，我们使用默认的堆栈设置创建第一个线程。因此，其堆栈应该从进程的堆栈段分配。之后，我们通过指定一个缓冲区的内存地址来创建第二个线程，该缓冲区应作为线程的堆栈区域。

注意，指定的尺寸比由`PTHREAD_STACK_MIN`宏指示的已定义最小堆栈大小多`100`字节。这个常量在不同的平台上有不同的值，它包含在头文件`limits.h`中。

如果你构建前面的程序并在 Linux 设备上运行它，你将看到以下类似的内容：

```cpp
$ gcc ExtremeC_examples_chapter16_3.c -o ex16_3.out -lpthread
$ ./ex16_3.out
Main Thread > Heap Address: 0x55a86a251260
Main Thread > Stack Address: 0x7ffcb5794d50
Thread2 > Stack Address: 0x55a86a2541a4
Thread1 > Stack Address: 0x7fa3e9216ee4
$
```

Shell Box 16-2：构建和运行示例 16.3

如*Shell Box 16-2*中看到的输出所示，分配在第二个线程堆栈顶部的局部变量`local_var`的地址属于不同的地址范围（堆空间的范围）。这意味着第二个线程的堆栈区域在堆内。然而，这并不适用于第一个线程。

如输出所示，第一个线程中局部变量的地址位于进程堆栈段的地址范围内。因此，我们可以成功地为新创建的线程分配一个从堆段分配的新堆栈区域。

设置线程堆栈区域的能力在某些用例中可能至关重要。例如，在内存受限的环境中，由于总内存量低，无法拥有大的堆栈，或者在性能要求高的环境中，无法容忍为每个线程分配堆栈的成本，使用一些预分配的缓冲区可能很有用，并且可以使用前面的过程将预分配的缓冲区设置为新创建线程的堆栈区域。

以下示例演示了在某个线程的堆栈中共享一个地址如何导致一些内存问题。当一个线程的地址被共享时，该线程应该保持活动状态，否则所有保持该地址的指针都将悬空。

以下代码不是线程安全的，因此我们预计在连续运行中会不时出现崩溃。线程也有默认的堆栈设置，这意味着它们的堆栈区域是从进程的堆栈段分配的：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
int* shared_int;
void* t1_body(void* arg) {
  int local_var = 100;
  shared_int = &local_var;
  // Wait for the other thread to print the shared integer
  usleep(10);
  return NULL;
}
void* t2_body(void* arg) {
  printf("%d\n", *shared_int);
  return NULL;
}
int main(int argc, char** argv) {
  shared_int = NULL;
  pthread_t t1;
  pthread_t t2;
  pthread_create(&t1, NULL, t1_body, NULL);
  pthread_create(&t2, NULL, t2_body, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
```

Code Box 16-8 [ExtremeC_examples_chapter16_4.c]：尝试读取从另一个线程的堆栈区域分配的变量

在开始时，我们声明了一个全局共享指针。由于它是一个指针，它可以接受任何地址，无论该地址指向进程内存布局中的哪个位置。它可能来自堆栈段、堆段，甚至是数据段。

在前面的代码中，在`t1_body`伴随函数内部，我们将局部变量的地址存储在共享指针中。这个变量属于第一个线程，并且它是在第一个线程的堆栈顶部分配的。

从现在开始，如果第一个线程退出，共享指针就会变成悬垂指针，任何解引用可能都会导致崩溃、逻辑错误或最坏情况下的隐藏内存问题。在某些交错中，这可能会发生，如果你多次运行前面的程序，你可能会时不时地看到崩溃。

作为一个重要的注意事项，如果某个线程打算使用从另一个线程的栈区域分配的变量，应该采用适当的同步技术。由于栈变量的生命周期与其作用域绑定，同步应该旨在保持作用域活跃，直到消费者线程完成对该变量的使用。

注意，为了简单起见，我们没有检查 pthread 函数的结果。始终建议这样做并检查返回值。并非所有 pthread 函数在不同平台上的行为都相同；如果出现问题，通过检查返回值你会意识到这一点。

在本节中，一般来说，我们展示了为什么栈区域所属的地址不应该共享，以及为什么最好不要从栈区域分配共享状态。下一节将讨论堆内存，这是存储共享状态最常见的地方。正如你可能已经猜到的，与堆一起工作也很棘手，你应该小心内存泄漏。

### 堆内存

堆段和数据段对所有线程都是可访问的。与在编译时生成的数据段不同，堆段是动态的，它在运行时形成。线程可以读取和修改堆的内容。此外，堆的内容可以持续到进程的生命周期，并且与单个线程的生命周期独立。此外，大对象可以放入堆中。所有这些因素共同导致堆成为存储将要由一些线程共享的状态的绝佳地方。

当涉及到堆分配时，内存管理变得像噩梦一样，这是因为分配的内存应该在某个时刻由运行中的某个线程释放，否则可能会导致内存泄漏。

关于并发环境，交错很容易产生悬垂指针；因此会出现崩溃。同步的关键作用是将事物置于特定的顺序，这样就不会产生悬垂指针，这是难点。

让我们来看以下示例，*示例 16.5*。在这个例子中有五个线程。第一个线程从堆中分配一个数组。第二个和第三个线程以这种形式填充数组。第二个线程将数组中的偶数索引填充为大写字母，从*Z*开始向后移动到*A*，第三个线程将奇数索引填充为小写字母，从*a*开始向前移动到*z*。第四个线程打印数组。最后，第五个线程释放数组并回收堆内存。

为了防止这些线程在堆空间中表现不当，应使用前几节中描述的所有关于 POSIX 并发控制的技巧。以下代码没有设置任何控制机制，显然，它不是线程安全的。请注意，代码并不完整。带有并发控制机制的完整版本将在下一个代码块中给出：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#define CHECK_RESULT(result) \
if (result) { \
  printf("A pthread error happened.\n"); \
  exit(1); \
}
int TRUE = 1;
int FALSE = 0;
// The pointer to the shared array
char* shared_array;
// The size of the shared array
unsigned int shared_array_len;
void* alloc_thread_body(void* arg) {
  shared_array_len = 20;
  shared_array = (char*)malloc(shared_array_len * sizeof(char*));
  return NULL;
}
void* filler_thread_body(void* arg) {
  int even = *((int*)arg);
  char c = 'a';
  size_t start_index = 1;
  if (even) {
    c = 'Z';
    start_index = 0;
  }
  for (size_t i = start_index; i < shared_array_len; i += 2) {
    shared_array[i] = even ? c-- : c++;
  }
  shared_array[shared_array_len - 1] = '\0';
  return NULL;
}
void* printer_thread_body(void* arg) {
  printf(">> %s\n", shared_array);
  return NULL;
}
void* dealloc_thread_body(void* arg) {
  free(shared_array);
  return NULL;
}
int main(int argc, char** argv) {
  … Create threads ...
}
```

代码框 16-9 [ExtremeC_examples_chapter16_5_raw.c]：没有同步机制的 16.5 示例

很容易看出，前面的代码不是线程安全的，并且由于分配器线程在释放数组时发生干扰，导致严重的崩溃。

当分配器线程获得 CPU 时，它会立即释放堆分配的缓冲区，之后指针 `shared_array` 变得悬空，其他线程开始崩溃。应使用适当的同步技术来确保分配器线程最后运行，并且不同线程的逻辑顺序正确。

在以下代码块中，我们使用 POSIX 并发控制对象装饰前面的代码，使其线程安全：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#define CHECK_RESULT(result) \
if (result) { \
  printf("A pthread error happened.\n"); \
  exit(1); \
}
int TRUE = 1;
int FALSE = 0;
// The pointer to the shared array
char* shared_array;
// The size of the shared array
size_t shared_array_len;
pthread_barrier_t alloc_barrier;
pthread_barrier_t fill_barrier;
pthread_barrier_t done_barrier;
void* alloc_thread_body(void* arg) {
  shared_array_len = 20;
  shared_array = (char*)malloc(shared_array_len * sizeof(char*));
  pthread_barrier_wait(&alloc_barrier);
  return NULL;
}
void* filler_thread_body(void* arg) {
  pthread_barrier_wait(&alloc_barrier);
  int even = *((int*)arg);
  char c = 'a';
  size_t start_index = 1;
  if (even) {
    c = 'Z';
    start_index = 0;
  }
  for (size_t i = start_index; i < shared_array_len; i += 2) {
    shared_array[i] = even ? c-- : c++;
  }
  shared_array[shared_array_len - 1] = '\0';
  pthread_barrier_wait(&fill_barrier);
  return NULL;
}
void* printer_thread_body(void* arg) {
  pthread_barrier_wait(&fill_barrier);
  printf(">> %s\n", shared_array);
  pthread_barrier_wait(&done_barrier);
  return NULL;
}
void* dealloc_thread_body(void* arg) {
  pthread_barrier_wait(&done_barrier);
  free(shared_array);
  pthread_barrier_destroy(&alloc_barrier);
  pthread_barrier_destroy(&fill_barrier);
  pthread_barrier_destroy(&done_barrier);
  return NULL;
}
int main(int argc, char** argv) {
  shared_array = NULL;
  pthread_barrier_init(&alloc_barrier, NULL, 3);
  pthread_barrier_init(&fill_barrier, NULL, 3);
  pthread_barrier_init(&done_barrier, NULL, 2);
  pthread_t alloc_thread;
  pthread_t even_filler_thread;
  pthread_t odd_filler_thread;
  pthread_t printer_thread;
  pthread_t dealloc_thread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  int res = pthread_attr_setdetachstate(&attr,
          PTHREAD_CREATE_DETACHED);
  CHECK_RESULT(res);
  res = pthread_create(&alloc_thread, &attr,
          alloc_thread_body, NULL);
  CHECK_RESULT(res);
  res = pthread_create(&even_filler_thread,
          &attr, filler_thread_body, &TRUE);
  CHECK_RESULT(res);
  res = pthread_create(&odd_filler_thread,
          &attr, filler_thread_body, &FALSE);
  CHECK_RESULT(res);
  res = pthread_create(&printer_thread, &attr,
          printer_thread_body, NULL);
  CHECK_RESULT(res);
  res = pthread_create(&dealloc_thread, &attr,
          dealloc_thread_body, NULL);
  CHECK_RESULT(res);
  pthread_exit(NULL);
  return 0;
}
```

代码框 16-10 [ExtremeC_examples_chapter16_5.c]：带有同步机制的 16.5 示例

为了使 *代码框 16-9* 中的代码线程安全，我们只使用了新的代码中的 POSIX 障碍。这是在多个线程之间形成顺序执行顺序的最简单方法。

如果你比较 *代码框 16-9* 和 *16-10*，你会看到 POSIX 障碍是如何在各个线程之间强加顺序的。唯一的例外是在两个填充线程之间。填充线程可以独立运行而不会相互阻塞，并且由于它们分别改变奇数和偶数索引，不会引发并发问题。请注意，前面的代码不能在苹果系统上编译。你需要在这些系统中使用互斥锁和条件变量来模拟障碍行为（就像我们在 *示例 16.2* 中做的那样）。

以下是对应代码的输出。无论你运行程序多少次，它都不会崩溃。换句话说，前面的代码可以防止各种交错，并且是线程安全的：

```cpp
$ gcc ExtremeC_examples_chapter16_5.c -o ex16_5 -lpthread
$ ./ex16_5
>> ZaYbXcWdVeUfTgShRiQ
$ ./ex16_5
>> ZaYbXcWdVeUfTgShRiQ
$
```

脚本框 16-3：构建和运行 16.5 示例

在本节中，我们给出了使用堆空间作为共享状态占位符的示例。与自动进行内存释放的栈内存不同，堆空间的内存释放应显式执行。否则，内存泄漏将是一个不可避免的副作用。

从程序员最少的内存管理努力的角度来看，最容易且有时也是最佳的可共享状态存储位置是数据段，其中分配和释放都是自动发生的。位于数据段中的变量被认为是全局的，并且具有可能的最长生命周期，从进程诞生的最初时刻到其最后的时刻。但这个长生命周期在某些用例中可能被视为负面因素，尤其是在你打算在数据段中保持一个大对象时。

在下一节中，我们将讨论内存可见性以及 POSIX 函数如何保证这一点。

### 内存可见性

我们在上一章中解释了*内存可见性*和*缓存一致性*，涉及具有多个 CPU 核心的系统。在本节中，我们想看看 pthread 库，看看它是如何保证内存可见性的。

如你所知，CPU 核心之间的缓存一致性协议确保所有 CPU 核心中单个内存地址的所有缓存版本都保持同步并更新，以反映其中一个 CPU 核心所做的最新更改。但这个协议需要以某种方式触发。

系统调用接口中存在 API 来触发缓存一致性协议，并使内存对所有 CPU 核心可见。在 pthread 中，也有许多函数在执行前保证内存可见性。

你可能之前遇到过一些这些函数。下面列出了它们的一些列表：

+   `pthread_barrier_wait`

+   `pthread_cond_broadcast`

+   `pthread_cond_signal`

+   `pthread_cond_timedwait`

+   `pthread_cond_wait`

+   `pthread_create`

+   `pthread_join`

+   `pthread_mutex_lock`

+   `pthread_mutex_timedlock`

+   `pthread_mutex_trylock`

+   `pthread_mutex_unlock`

+   `pthread_spin_lock`

+   `pthread_spin_trylock`

+   `pthread_spin_unlock`

+   `pthread_rwlock_rdlock`

+   `pthread_rwlock_timedrdlock`

+   `pthread_rwlock_timedwrlock`

+   `pthread_rwlock_tryrdlock`

+   `pthread_rwlock_trywrlock`

+   `pthread_rwlock_unlock`

+   `pthread_rwlock_wrlock`

+   `sem_post`

+   `sem_timedwait`

+   `sem_trywait`

+   `sem_wait`

+   `semctl`

+   `semop`

除了 CPU 核心中的本地缓存之外，编译器还可以为常用变量引入缓存机制。为了实现这一点，编译器需要分析代码并以一种方式优化它，即频繁使用的变量被写入和读取到编译器缓存中。这些是由编译器放入最终二进制文件中的软件缓存，以优化和提升程序的执行。

虽然这些缓存可能有益，但它们在编写多线程代码时可能会增加另一个头疼的问题，并引发一些内存可见性问题。因此，有时必须禁用特定变量的这些缓存。

不应该被编译器通过缓存优化的变量可以声明为 *易失性*。请注意，易失性变量仍然可以在 CPU 级别被缓存，但编译器不会通过将其保留在编译器缓存中来进行优化。可以使用关键字 `volatile` 声明一个易失性变量。以下是一个易失性整型变量的声明：

```cpp
volatile int number;
```

代码框 16-11：声明一个易失性整型变量

易失性变量的重要之处在于它们并不能解决多线程系统中的内存可见性问题。为了解决这个问题，你需要正确地使用前面提到的 POSIX 函数，以确保内存可见性。

## 概述

在本章中，我们介绍了 POSIX 线程 API 提供的并发控制机制。我们讨论了：

+   POSIX 互斥锁及其使用方法

+   POSIX 条件变量和屏障及其使用方法

+   POSIX 信号量和它们如何与二进制信号量和通用信号量不同

+   线程如何与栈区域交互

+   如何为线程定义一个新的堆分配的栈区域

+   线程如何与堆空间交互

+   内存可见性和保证内存可见性的 POSIX 函数

+   易失性变量和编译器缓存

在下一章中，我们将继续我们的讨论，我们将讨论在软件系统中实现并发性的另一种方法：多进程。我们将讨论进程的执行方式以及它与线程的不同之处。
