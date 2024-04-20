# C++多线程 API

虽然 C++在标准模板库（STL）中有本地的多线程实现，但基于操作系统和框架的多线程 API 仍然非常常见。这些 API 的示例包括 Windows 和 POSIX（可移植操作系统接口）线程，以及由 Qt、Boost 和 POCO 库提供的线程。

本章将详细介绍每个 API 提供的功能，以及它们之间的相似之处和不同之处。最后，我们将使用示例代码来查看常见的使用场景。

本章涵盖的主题包括以下内容：

+   可用多线程 API 的比较

+   每个 API 的使用示例

# API 概述

在 C++ 2011（C++11）标准之前，开发了许多不同的线程实现，其中许多限于特定的软件平台。其中一些至今仍然相关，例如 Windows 线程。其他已被标准取代，其中 POSIX 线程（Pthreads）已成为类 UNIX 操作系统的事实标准。这包括基于 Linux 和 BSD 的操作系统，以及 OS X（macOS）和 Solaris。

许多库被开发出来，以使跨平台开发更容易。尽管 Pthreads 有助于使类 UNIX 操作系统更或多或少兼容，但要使软件在所有主要操作系统上可移植，需要一个通用的线程 API。这就是为什么创建了 Boost、POCO 和 Qt 等库。应用程序可以使用这些库，并依赖于库来处理平台之间的任何差异。

# POSIX 线程

Pthreads 最初是在 1995 年的 POSIX.1c 标准（线程扩展，IEEE Std 1003.1c-1995）中定义的，作为 POSIX 标准的扩展。当时，UNIX 被选择为制造商中立的接口，POSIX 统一了它们之间的各种 API。

尽管有这种标准化的努力，Pthread 在实现它的操作系统之间仍然存在差异（例如，在 Linux 和 OS X 之间），这是由于不可移植的扩展（在方法名称中标有 _np）。

对于 pthread_setname_np 方法，Linux 实现需要两个参数，允许设置除当前线程以外的线程名称。在 OS X（自 10.6 起），此方法只需要一个参数，允许设置当前线程的名称。如果可移植性是一个问题，就必须注意这样的差异。

1997 年后，POSIX 标准修订由 Austin 联合工作组管理。这些修订将线程扩展合并到主标准中。当前的修订是 7，也称为 POSIX.1-2008 和 IEEE Std 1003.1，2013 版--标准的免费副本可在线获得。

操作系统可以获得符合 POSIX 标准的认证。目前，这些如下表所述：

| **名称** | **开发者** | **自版本** | **架构（当前）** | **备注** |
| --- | --- | --- | --- | --- |
| AIX | IBM | 5L | POWER | 服务器操作系统 |
| HP-UX | Hewlett-Packard | 11i v3 | PA-RISC, IA-64 (Itanium) | 服务器操作系统 |
| IRIX | Silicon Graphics (SGI) | 6 | MIPS | 已停产 |
| Inspur K-UX | Inspur | 2 | X86_64, | 基于 Linux |
| Integrity | Green Hills Software | 5 | ARM, XScale, Blackfin, Freescale Coldfire, MIPS, PowerPC, x86。 | 实时操作系统 |
| OS X/MacOS | Apple | 10.5 (Leopard) | X86_64 | 桌面操作系统 |
| QNX Neutrino | BlackBerry | 1 | Intel 8088, x86, MIPS, PowerPC, SH-4, ARM, StrongARM, XScale | 实时，嵌入式操作系统 |
| Solaris | Sun/Oracle | 2.5 | SPARC, IA-32 (<11), x86_64, PowerPC (2.5.1) | 服务器操作系统 |
| Tru64 | DEC, HP, IBM, Compaq | 5.1B-4 | Alpha | 已停产 |
| UnixWare | Novell, SCO, Xinuos | 7.1.3 | x86 | 服务器操作系统 |

其他操作系统大多是兼容的。以下是相同的示例：

| **名称** | **平台** | **备注** |
| --- | --- | --- |
| Android | ARM, x86, MIPS | 基于 Linux。Bionic C 库。 |
| BeOS (Haiku) | IA-32, ARM, x64_64 | 限于 x86 的 GCC 2.x。 |
| Darwin | PowerPC，x86，ARM | 使用 macOS 基于的开源组件。 |
| FreeBSD | IA-32，x86_64，sparc64，PowerPC，ARM，MIPS 等 | 基本上符合 POSIX。可以依赖已记录的 POSIX 行为。一般来说，比 Linux 更严格地遵守规范。 |
| Linux | Alpha，ARC，ARM，AVR32，Blackfin，H8/300，Itanium，m68k，Microblaze，MIPS，Nios II，OpenRISC，PA-RISC，PowerPC，s390，S+core，SuperH，SPARC，x86，Xtensa 等 | 一些 Linux 发行版（见前表）被认证为符合 POSIX。这并不意味着每个 Linux 发行版都符合 POSIX。一些工具和库可能与标准不同。对于 Pthreads，这可能意味着在 Linux 发行版之间（不同的调度程序等）以及与实现 Pthreads 的其他操作系统之间的行为有时会有所不同。 |
| MINIX 3 | IA-32，ARM | 符合 POSIX 规范标准 3（SUSv3，2004 年）。 |
| NetBSD | Alpha，ARM，PA-RISC，68k，MIPS，PowerPC，SH3，SPARC，RISC-V，VAX，x86 等 | 几乎完全兼容 POSX.1（1990），并且大部分符合 POSIX.2（1992）。 |
| 核心 RTOS | ARM，MIPS，PowerPC，Nios II，MicroBlaze，SuperH 等 | Mentor Graphics 的专有 RTOS，旨在嵌入式应用。 |
| NuttX | ARM，AVR，AVR32，HCS12，SuperH，Z80 等 | 轻量级 RTOS，可在 8 到 32 位系统上扩展，专注于 POSIX 兼容性。 |
| OpenBSD | Alpha，x86_64，ARM，PA-RISC，IA-32，MIPS，PowerPC，SPARC 等 | 1995 年从 NetBSD 分叉出来。类似的 POSIX 支持。 |
| OpenSolaris/illumos | IA-32，x86_64，SPARC，ARM | 与商业 Solaris 发行版兼容认证。 |
| VxWorks | ARM，SH-4，x86，x86_64，MIPS，PowerPC | 符合 POSIX，并获得用户模式执行环境的认证。 |

由此可见，遵循 POSIX 规范并不是一件明显的事情，也不能指望自己的代码在每个平台上都能编译。每个平台还将有其自己的标准扩展，用于标准中省略的但仍然有用的功能。然而，Pthreads 在 Linux、BSD 和类似软件中被广泛使用。

# Windows 支持

也可以使用 POSIX API，例如以下方式：

| **名称** | **兼容性** |
| --- | --- |
| Cygwin | 大部分完整。为 POSIX 应用程序提供完整的运行时环境，可以作为普通的 Windows 应用程序分发。 |
| MinGW | 使用 MinGW-w64（MinGW 的重新开发），Pthreads 支持相当完整，尽管可能会缺少一些功能。 |
| Windows Subsystem for Linux | WSL 是 Windows 10 的一个功能，允许 Ubuntu Linux 14.04（64 位）镜像的工具和实用程序在其上本地运行，尽管不能使用 GUI 功能或缺少内核功能。否则，它提供与 Linux 类似的兼容性。此功能目前要求运行 Windows 10 周年更新，并按照微软提供的说明手动安装 WSL。 |

一般不建议在 Windows 上使用 POSIX。除非有充分的理由使用 POSIX（例如，大量现有的代码库），否则最好使用其中一个跨平台 API（本章后面将介绍），这样可以消除任何平台问题。

在接下来的章节中，我们将看一下 Pthreads API 提供的功能。

# PThreads 线程管理

这些都是以`pthread_`或`pthread_attr_`开头的函数。这些函数都适用于线程本身及其属性对象。

使用 Pthreads 的基本方法如下：

```cpp
#include <pthread.h> 
#include <stdlib.h> 

#define NUM_THREADS     5 
```

主要的 Pthreads 头文件是`pthread.h`。这提供了对除了信号量（稍后在本节中讨论）之外的所有内容的访问。我们还在这里定义了一个希望启动的线程数的常量：

```cpp
void* worker(void* arg) { 
    int value = *((int*) arg); 

    // More business logic. 

    return 0; 
} 
```

我们定义了一个简单的`Worker`函数，稍后将把它传递给新线程。为了演示和调试目的，可以首先添加一个简单的基于`cout`或`printf`的业务逻辑，以打印发送到新线程的值。

接下来，我们定义`main`函数如下：

```cpp
int main(int argc, char** argv) { 
    pthread_t threads[NUM_THREADS]; 
    int thread_args[NUM_THREADS]; 
    int result_code; 

    for (unsigned int i = 0; i < NUM_THREADS; ++i) { 
        thread_args[i] = i; 
        result_code = pthread_create(&threads[i], 0, worker, (void*) &thread_args[i]); 
    } 
```

我们在上述函数中的循环中创建所有线程。每个线程实例在创建时被分配一个线程 ID（第一个参数），并且`pthread_create()`函数返回一个结果代码（成功时为零）。线程 ID 是在将来的调用中引用线程的句柄。

函数的第二个参数是`pthread_attr_t`结构实例，如果没有则为 0。这允许配置新线程的特性，例如初始堆栈大小。当传递零时，将使用默认参数，这些参数因平台和配置而异。

第三个参数是一个指向新线程将启动的函数的指针。此函数指针被定义为一个返回指向 void 数据的指针的函数（即自定义数据），并接受指向 void 数据的指针。在这里，作为参数传递给新线程的数据是线程 ID：

```cpp
    for (int i = 0; i < NUM_THREADS; ++i) { 
        result_code = pthread_join(threads[i], 0); 
    } 

    exit(0); 
} 
```

接下来，我们使用`pthread_join()`函数等待每个工作线程完成。此函数接受两个参数，要等待的线程的 ID，以及`Worker`函数的返回值的缓冲区（或零）。

管理线程的其他函数如下：

+   `void pthread_exit`(`void *value_ptr`)：

此函数终止调用它的线程，使提供的参数值可用于调用`pthread_join()`的任何线程。

+   `int pthread_cancel`(`pthread_t` thread)：

此函数请求取消指定的线程。根据目标线程的状态，这将调用其取消处理程序。

除此之外，还有`pthread_attr_*`函数来操作和获取有关`pthread_attr_t`结构的信息。

# 互斥锁

这些函数的前缀为`pthread_mutex_`或`pthread_mutexattr_`。它们适用于互斥锁及其属性对象。

Pthreads 中的互斥锁可以被初始化、销毁、锁定和解锁。它们还可以使用`pthread_mutexattr_t`结构自定义其行为，该结构具有相应的`pthread_mutexattr_*`函数用于初始化和销毁其属性。

使用静态初始化的 Pthread 互斥锁的基本用法如下：

```cpp
static pthread_mutex_t func_mutex = PTHREAD_MUTEX_INITIALIZER; 

void func() { 
    pthread_mutex_lock(&func_mutex); 

    // Do something that's not thread-safe. 

    pthread_mutex_unlock(&func_mutex); 
} 
```

在这段代码的最后，我们使用了`PTHREAD_MUTEX_INITIALIZER`宏，它为我们初始化了互斥锁，而无需每次都输入代码。与其他 API 相比，人们必须手动初始化和销毁互斥锁，尽管使用宏在某种程度上有所帮助。

之后，我们锁定和解锁互斥锁。还有`pthread_mutex_trylock()`函数，它类似于常规锁定版本，但如果引用的互斥锁已经被锁定，它将立即返回而不是等待它被解锁。

在此示例中，互斥锁没有被显式销毁。然而，这是 Pthreads 应用程序中正常内存管理的一部分。

# 条件变量

这些函数的前缀为`pthread_cond_`或`pthread_condattr_`。它们适用于条件变量及其属性对象。

Pthreads 中的条件变量遵循相同的模式，除了具有相同的`pthread_condattr_t`属性结构管理外，还有初始化和`destroy`函数。

此示例涵盖了 Pthreads 条件变量的基本用法：

```cpp
#include <pthread.h> 
#include <stdlib.h>
#include <unistd.h>

   #define COUNT_TRIGGER 10 
   #define COUNT_LIMIT 12 

   int count = 0; 
   int thread_ids[3] = {0,1,2}; 
   pthread_mutex_t count_mutex; 
   pthread_cond_t count_cv; 
```

在上述代码中，我们获取标准头文件，并定义一个计数触发器和限制，其目的将很快变得清楚。我们还定义了一些全局变量：计数变量，我们希望创建的线程的 ID，以及互斥锁和条件变量：

```cpp
void* add_count(void* t)  { 
    int tid = (long) t; 
    for (int i = 0; i < COUNT_TRIGGER; ++i) { 
        pthread_mutex_lock(&count_mutex); 
        count++; 
        if (count == COUNT_LIMIT) { 
            pthread_cond_signal(&count_cv); 
        } 

        pthread_mutex_unlock(&count_mutex); 
        sleep(1); 
    } 

    pthread_exit(0); 
} 
```

在获取`count_mutex`的独占访问权限后，前面的函数本质上只是将全局计数器变量增加。它还检查计数触发值是否已达到。如果是，它将发出条件变量的信号。

为了让也运行此函数的第二个线程有机会获得互斥锁，我们在循环的每个周期中睡眠 1 秒：

```cpp
void* watch_count(void* t) { 
    int tid = (int) t; 

    pthread_mutex_lock(&count_mutex); 
    if (count < COUNT_LIMIT) { 
        pthread_cond_wait(&count_cv, &count_mutex); 
    } 

    pthread_mutex_unlock(&count_mutex); 
    pthread_exit(0); 
} 
```

在这个第二个函数中，在检查是否已经达到计数限制之前，我们先锁定全局互斥锁。这是我们的保险，以防此函数运行的线程在计数达到限制之前没有被调用。

否则，我们等待条件变量提供条件变量和锁定的互斥锁。一旦发出信号，我们解锁全局互斥锁，并退出线程。

这里需要注意的一点是，此示例未考虑虚假唤醒。Pthreads 条件变量容易受到这种唤醒的影响，这需要使用循环并检查是否已满足某种条件：

```cpp
int main (int argc, char* argv[]) { 
    int tid1 = 1, tid2 = 2, tid3 = 3; 
    pthread_t threads[3]; 
    pthread_attr_t attr; 

    pthread_mutex_init(&count_mutex, 0); 
    pthread_cond_init (&count_cv, 0); 

    pthread_attr_init(&attr); 
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_create(&threads[0], &attr, watch_count, (void *) tid1); 
    pthread_create(&threads[1], &attr, add_count, (void *) tid2); 
    pthread_create(&threads[2], &attr, add_count, (void *) tid3); 

    for (int i = 0; i < 3; ++i) { 
        pthread_join(threads[i], 0); 
    } 

    pthread_attr_destroy(&attr); 
    pthread_mutex_destroy(&count_mutex); 
    pthread_cond_destroy(&count_cv); 
    return 0; 
}  
```

最后，在`main`函数中，我们创建三个线程，其中两个运行将计数器增加的函数，第三个运行等待其条件变量被发出信号的函数。

在这种方法中，我们还初始化全局互斥锁和条件变量。我们创建的线程还明确设置了“可连接”属性。

最后，我们等待每个线程完成，然后进行清理，在退出之前销毁属性结构实例、互斥锁和条件变量。

使用`pthread_cond_broadcast()`函数，还可以向等待条件变量的所有线程发出信号，而不仅仅是队列中的第一个线程。这使得可以更优雅地使用条件变量，例如，当有很多工作线程等待新数据集到达时，无需单独通知每个线程。

# 同步

实现同步的函数以`pthread_rwlock_`或`pthread_barrier_`为前缀。这些实现读/写锁和同步屏障。

**读/写锁**（**rwlock**）与互斥锁非常相似，只是它具有额外的功能，允许无限线程同时读取，而只限制写入访问一个线程。

使用`rwlock`与使用互斥锁非常相似：

```cpp
#include <pthread.h> 
int pthread_rwlock_init(pthread_rwlock_t* rwlock, const pthread_rwlockattr_t* attr); 
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER; 
```

在最后的代码中，我们包含相同的通用头文件，并使用初始化函数或通用宏。有趣的部分是当我们锁定`rwlock`时，可以仅进行只读访问：

```cpp
int pthread_rwlock_rdlock(pthread_rwlock_t* rwlock); 
int pthread_rwlock_tryrdlock(pthread_rwlock_t* rwlock); 
```

这里，如果锁已经被锁定，第二种变体会立即返回。也可以按以下方式锁定它以进行写访问：

```cpp
int pthread_rwlock_wrlock(pthread_rwlock_t* rwlock); 
int pthread_rwlock_trywrlock(pthread_rwlock_t * rwlock); 
```

这些函数基本上是相同的，唯一的区别是在任何给定时间只允许一个写入者，而多个读取者可以获得只读锁定。

屏障是 Pthreads 的另一个概念。这些是类似于一组线程的屏障的同步对象。在这些线程中的所有线程都必须在任何一个线程可以继续执行之前到达屏障。在屏障初始化函数中，指定了线程计数。只有当所有这些线程都使用`pthread_barrier_wait()`函数调用`barrier`对象后，它们才会继续执行。

# 信号量

如前所述，信号量不是原始 Pthreads 扩展的一部分。出于这个原因，它们在`semaphore.h`头文件中声明。

实质上，信号量是简单的整数，通常用作资源计数。为了使它们线程安全，使用原子操作（检查和锁定）。POSIX 信号量支持初始化、销毁、增加和减少信号量以及等待信号量达到非零值的操作。

# 线程本地存储（TLC）

使用 Pthreads，TLS 是通过键和设置线程特定数据的方法来实现的：

```cpp
pthread_key_t global_var_key;

void* worker(void* arg) {
    int *p = new int;
    *p = 1;
    pthread_setspecific(global_var_key, p);
    int* global_spec_var = (int*) pthread_getspecific(global_var_key);
    *global_spec_var += 1;
    pthread_setspecific(global_var_key, 0);
    delete p;
    pthread_exit(0);
}
```

在工作线程中，我们在堆上分配一个新的整数，并将全局密钥设置为其自己的值。将全局变量增加 1 后，其值将为 2，而不管其他线程做什么。我们可以在此线程完成后将全局变量设置为 0，并删除分配的值：

```cpp
int main(void) {
    pthread_t threads[5];

    pthread_key_create(&global_var_key, 0);
    for (int i = 0; i < 5; ++i)
        pthread_create(&threads[i],0,worker,0);
    for (int i = 0; i < 5; ++i) {
        pthread_join(threads[i], 0);
    }
    return 0;
}
```

设置并使用全局密钥来引用 TLS 变量，但我们创建的每个线程都可以为该密钥设置自己的值。

虽然线程可以创建自己的密钥，但与本章中正在查看的其他 API 相比，处理 TLS 的这种方法相当复杂。

# Windows 线程

相对于 Pthreads，Windows 线程仅限于 Windows 操作系统和类似系统（例如 ReactOS 和其他使用 Wine 的操作系统）。这提供了一个相当一致的实现，可以轻松地由支持对应的 Windows 版本来定义。

在 Windows Vista 之前，线程支持缺少诸如条件变量之类的功能，同时具有 Pthreads 中找不到的功能。根据一个人的观点，使用 Windows 头文件定义的无数“类型定义”类型可能也会让人感到烦扰。

# 线程管理

一个使用 Windows 线程的基本示例，从官方 MSDN 文档示例代码中改编而来，看起来像这样：

```cpp
#include <windows.h> 
#include <tchar.h> 
#include <strsafe.h> 

#define MAX_THREADS 3 
#define BUF_SIZE 255  
```

在包含一系列 Windows 特定的头文件（用于线程函数、字符字符串等）之后，我们定义了要创建的线程数以及`Worker`函数中消息缓冲区的大小。

我们还定义了一个结构类型（通过`void pointer: LPVOID`传递），用于包含我们传递给每个工作线程的示例数据：

```cpp
typedef struct MyData { 
 int val1; 
 int val2; 
} MYDATA, *PMYDATA;

DWORD WINAPI worker(LPVOID lpParam) { 
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    if (hStdout == INVALID_HANDLE_VALUE) { 
        return 1; 
    } 

    PMYDATA pDataArray =  (PMYDATA) lpParam; 

    TCHAR msgBuf[BUF_SIZE]; 
    size_t cchStringSize; 
    DWORD dwChars; 
    StringCchPrintf(msgBuf, BUF_SIZE, TEXT("Parameters = %d, %dn"),  
    pDataArray->val1, pDataArray->val2);  
    StringCchLength(msgBuf, BUF_SIZE, &cchStringSize); 
    WriteConsole(hStdout, msgBuf, (DWORD) cchStringSize, &dwChars, NULL); 

    return 0;  
}  
```

在`Worker`函数中，我们将提供的参数转换为我们自定义的结构类型，然后使用它将其值打印到字符串上，然后输出到控制台。

我们还验证是否有活动的标准输出（控制台或类似）。用于打印字符串的函数都是线程安全的。

```cpp
void errorHandler(LPTSTR lpszFunction) { 
    LPVOID lpMsgBuf; 
    LPVOID lpDisplayBuf; 
    DWORD dw = GetLastError();  

    FormatMessage( 
        FORMAT_MESSAGE_ALLOCATE_BUFFER |  
        FORMAT_MESSAGE_FROM_SYSTEM | 
        FORMAT_MESSAGE_IGNORE_INSERTS, 
        NULL, 
        dw, 
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
        (LPTSTR) &lpMsgBuf, 
        0, NULL); 

        lpDisplayBuf = (LPVOID) LocalAlloc(LMEM_ZEROINIT,  
        (lstrlen((LPCTSTR) lpMsgBuf) + lstrlen((LPCTSTR) lpszFunction) + 40) * sizeof(TCHAR));  
        StringCchPrintf((LPTSTR)lpDisplayBuf,  
        LocalSize(lpDisplayBuf) / sizeof(TCHAR), 
        TEXT("%s failed with error %d: %s"),  
        lpszFunction, dw, lpMsgBuf);  
        MessageBox(NULL, (LPCTSTR) lpDisplayBuf, TEXT("Error"), MB_OK);  

        LocalFree(lpMsgBuf); 
        LocalFree(lpDisplayBuf); 
} 
```

在这里，定义了一个错误处理程序函数，该函数获取最后一个错误代码的系统错误消息。获取最后一个错误的代码后，将格式化要输出的错误消息，并显示在消息框中。最后，释放分配的内存缓冲区。

最后，`main`函数如下：

```cpp
int _tmain() {
         PMYDATA pDataArray[MAX_THREADS];
         DWORD dwThreadIdArray[MAX_THREADS];
         HANDLE hThreadArray[MAX_THREADS];
         for (int i = 0; i < MAX_THREADS; ++i) {
               pDataArray[i] = (PMYDATA) HeapAlloc(GetProcessHeap(),
                           HEAP_ZERO_MEMORY, sizeof(MYDATA));                     if (pDataArray[i] == 0) {
                           ExitProcess(2);
             }
             pDataArray[i]->val1 = i;
             pDataArray[i]->val2 = i+100;
             hThreadArray[i] = CreateThread(
                  NULL,          // default security attributes
                  0,             // use default stack size
                  worker,        // thread function name
                  pDataArray[i], // argument to thread function
                  0,             // use default creation flags
                  &dwThreadIdArray[i]);// returns the thread identifier
             if (hThreadArray[i] == 0) {
                         errorHandler(TEXT("CreateThread"));
                         ExitProcess(3);
             }
   }
         WaitForMultipleObjects(MAX_THREADS, hThreadArray, TRUE, INFINITE);
         for (int i = 0; i < MAX_THREADS; ++i) {
               CloseHandle(hThreadArray[i]);
               if (pDataArray[i] != 0) {
                           HeapFree(GetProcessHeap(), 0, pDataArray[i]);
               }
         }
         return 0;
}
```

在`main`函数中，我们在循环中创建我们的线程，为线程数据分配内存，并在启动线程之前为每个线程生成唯一数据。每个线程实例都传递了自己的唯一参数。

之后，我们等待线程完成并重新加入。这本质上与在 Pthreads 上调用`join`函数相同——只是这里，一个函数调用就足够了。

最后，关闭每个线程句柄，并清理之前分配的内存。

# 高级管理

使用 Windows 线程进行高级线程管理包括作业、纤程和线程池。作业基本上允许将多个线程链接在一起成为一个单一单元，从而可以一次性更改所有这些线程的属性和状态。

纤程是轻量级线程，运行在创建它们的线程的上下文中。创建线程预期自己调度这些纤程。纤程还有类似 TLS 的**纤程本地存储**（**FLS**）。

最后，Windows 线程 API 提供了一个线程池 API，允许在应用程序中轻松使用这样的线程池。每个进程也提供了一个默认的线程池。

# 同步

使用 Windows 线程，可以使用临界区、互斥锁、信号量、**轻量级读写器**（**SRW**）锁、屏障和变体来实现互斥和同步。

同步对象包括以下内容：

| **名称** | **描述** |
| --- | --- |
| 事件 | 允许使用命名对象在线程和进程之间进行事件信号传递。 |
| 互斥锁 | 用于线程间和进程同步，协调对共享资源的访问。 |
| 信号量 | 标准信号量计数对象，用于线程间和进程同步。 |
| 可等待定时器 | 可由多个进程使用的定时器对象，具有多种使用模式。 |
| 临界区 | 临界区本质上是互斥锁，限于单个进程，这使得它们比使用互斥锁更快，因为缺少内核空间调用。 |
| 轻量级读写锁 | SRW 类似于 Pthreads 中的读/写锁，允许多个读取者或单个写入者线程访问共享资源。 |
| 交错变量访问 | 允许对一系列变量进行原子访问，否则不能保证原子性。这使得线程可以共享变量，而无需使用互斥锁。 |

# 条件变量

使用 Windows 线程实现条件变量是非常简单的。它使用临界区（`CRITICAL_SECTION`）和条件变量（`CONDITION_VARIABLE`）以及条件变量函数来等待特定的条件变量，或者发出信号。

# 线程本地存储

**线程本地存储**（**TLS**）与 Windows 线程类似于 Pthreads，首先必须创建一个中央键（TLS 索引），然后各个线程可以使用该全局索引来存储和检索本地值。

与 Pthreads 一样，这涉及相似数量的手动内存管理，因为 TLS 值必须手动分配和删除。

# Boost

Boost 线程是 Boost 库集合中相对较小的一部分。然而，它被用作成为 C++11 中多线程实现基础的基础，类似于其他 Boost 库最终完全或部分地成为新的 C++标准。有关多线程 API 的详细信息，请参阅本章中的 C++线程部分。

C++11 标准中缺少的功能，在 Boost 线程中是可用的，包括以下内容：

+   线程组（类似于 Windows 作业）

+   线程中断（取消）

+   带超时的线程加入

+   额外的互斥锁类型（在 C++14 中改进）

除非绝对需要这些功能，或者无法使用支持 C++11 标准（包括 STL 线程）的编译器，否则没有理由使用 Boost 线程而不是 C++11 实现。

由于 Boost 提供了对本机操作系统功能的包装，使用本机 C++线程可能会减少开销，具体取决于 STL 实现的质量。

```cpp
POCO
```

POCO 库是对操作系统功能的相当轻量级的包装。它不需要兼容 C++11 的编译器或任何类型的预编译或元编译。

# 线程类

`Thread`类是对 OS 级别线程的简单包装。它接受从`Runnable`类继承的`Worker`类实例。官方文档提供了一个基本示例，如下所示：

```cpp
#include "Poco/Thread.h" 
#include "Poco/Runnable.h" 
#include <iostream> 

class HelloRunnable: public Poco::Runnable { 
    virtual void run() { 
        std::cout << "Hello, world!" << std::endl; 
    } 
}; 

int main(int argc, char** argv) { 
    HelloRunnable runnable; 
    Poco::Thread thread; 
    thread.start(runnable); 
    thread.join(); 
    return 0; 
} 
```

上述代码是一个非常简单的“Hello world”示例，其中一个工作线程仅通过标准输出输出一个字符串。线程实例分配在堆栈上，并在入口函数的范围内等待工作线程完成，使用`join()`函数。

POCO 的许多线程功能与 Pthreads 非常相似，尽管在配置线程和其他对象等方面有明显的偏差。作为一个 C++库，它使用类方法来设置属性，而不是填充结构并将其作为参数传递。

# 线程池

POCO 提供了一个默认的线程池，有 16 个线程。这个数字可以动态改变。与常规线程一样，线程池需要传递一个从`Runnable`类继承的`Worker`类实例：

```cpp
#include "Poco/ThreadPool.h" 
#include "Poco/Runnable.h" 
#include <iostream> 

class HelloRunnable: public Poco::Runnable { 
    virtual void run() { 
        std::cout << "Hello, world!" << std::endl; 
    } 
}; 

int main(int argc, char** argv) { 
    HelloRunnable runnable; 
    Poco::ThreadPool::defaultPool().start(runnable); 
    Poco::ThreadPool::defaultPool().joinAll(); 
    return 0; 
} 
```

工作线程实例被添加到线程池中，并运行它。当我们添加另一个工作线程实例，更改容量或调用`joinAll()`时，线程池会清理空闲一定时间的线程。结果，单个工作线程将加入，并且没有活动线程，应用程序退出。

# 线程本地存储（TLS）

在 POCO 中，TLS 被实现为一个类模板，允许人们将其用于几乎任何类型。

正如官方文档所述：

```cpp
#include "Poco/Thread.h" 
#include "Poco/Runnable.h" 
#include "Poco/ThreadLocal.h" 
#include <iostream> 

class Counter: public Poco::Runnable { 
    void run() { 
        static Poco::ThreadLocal<int> tls; 
        for (*tls = 0; *tls < 10; ++(*tls)) { 
            std::cout << *tls << std::endl; 
        } 
    } 
}; 

int main(int argc, char** argv) { 
    Counter counter1; 
    Counter counter2; 
    Poco::Thread t1; 
    Poco::Thread t2; 
    t1.start(counter1); 
    t2.start(counter2); 
    t1.join(); 
    t2.join(); 
    return 0; 
} 
```

在上面的 worker 示例中，我们使用`ThreadLocal`类模板创建了一个静态 TLS 变量，并定义它包含一个整数。

因为我们将它定义为静态的，所以每个线程只会创建一次。为了使用我们的 TLS 变量，我们可以使用箭头(`->`)或星号(`*`)运算符来访问它的值。在这个例子中，我们在`for`循环的每个周期增加 TLS 值，直到达到限制为止。

这个例子表明，两个线程将生成自己的一系列 10 个整数，计数相同的数字而互不影响。

# 同步

POCO 提供的同步原语如下：

+   互斥量

+   FastMutex

+   事件

+   条件

+   信号量

+   RWLock

这里需要注意的是`FastMutex`类。这通常是一种非递归的互斥类型，只是在 Windows 上是递归的。这意味着人们通常应该假设任一类型在同一线程中可以多次锁定同一互斥量。

人们还可以使用`ScopedLock`类与互斥量一起使用，确保它封装的互斥量在当前作用域结束时被释放。

事件类似于 Windows 事件，只是它们限于单个进程。它们构成了 POCO 中条件变量的基础。

POCO 条件变量的功能与 Pthreads 等方式基本相同，只是它们不会出现虚假唤醒。通常情况下，条件变量会因为优化原因而出现这些随机唤醒。通过不需要显式检查条件变量等待返回时是否满足条件，减轻了开发者的负担。

# C++线程

C++中的本地多线程支持在第十二章中有详细介绍，*本地 C++线程和原语*。

正如本章中 Boost 部分提到的，C++多线程支持在很大程度上基于 Boost 线程 API，使用几乎相同的头文件和名称。API 本身再次让人联想到 Pthreads，尽管在某些方面有显著的不同，比如条件变量。

接下来的章节将专门使用 C++线程支持进行示例。

# 将它们组合在一起

在本章涵盖的 API 中，只有 Qt 多线程 API 可以被认为是真正高级的。尽管其他 API（包括 C++11）包含一些更高级的概念，包括线程池和异步运行器，不需要直接使用线程，但 Qt 提供了一个完整的信号-槽架构，使得线程间通信异常容易。

正如本章所介绍的，这种便利也伴随着一个代价，即需要开发应用程序以适应 Qt 框架。这可能在项目中是不可接受的。

哪种 API 是正确的取决于个人的需求。然而，可以相对公平地说，当可以使用 C++11 线程、POCO 等 API 时，使用直接的 Pthreads、Windows 线程等并没有太多意义，这些 API 可以在不显著降低性能的情况下轻松地实现跨平台。

所有这些 API 在核心功能上至少在某种程度上是可比较的。

# 总结

在本章中，我们详细介绍了一些较流行的多线程 API 和框架，将它们并列在一起，以了解它们的优势和劣势。我们通过一些示例展示了如何使用这些 API 来实现基本功能。

在下一章中，我们将详细介绍如何同步线程并在它们之间进行通信。
