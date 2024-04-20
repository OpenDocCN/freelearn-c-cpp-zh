# 第三章：C++多线程 API

虽然 C++在**标准模板库**（**STL**）中有本地的多线程实现，但基于操作系统和框架的多线程 API 仍然非常常见。这些 API 的例子包括 Windows 和**POSIX**（**可移植操作系统接口**）线程，以及`Qt`、`Boost`和`POCO`库提供的线程。

本章详细介绍了每个 API 提供的功能，以及它们之间的相似之处和不同之处。最后，我们将使用示例代码来查看常见的使用场景。

本章涵盖的主题包括以下内容：

+   可用多线程 API 的比较

+   每个 API 的用法示例

# API 概述

在**C++ 2011**（**C++11**）标准之前，开发了许多不同的线程实现，其中许多限于特定的软件平台。其中一些至今仍然相关，例如 Windows 线程。其他已被标准取代，其中**POSIX Threads**（**Pthreads**）已成为类 UNIX 操作系统的事实标准。这包括基于 Linux 和基于 BSD 的操作系统，以及 OS X（macOS）和 Solaris。

许多库被开发出来，以使跨平台开发更容易。尽管 Pthreads 有助于使类 UNIX 操作系统更或多或少地兼容，但要使软件在所有主要操作系统上可移植，需要一个通用的线程 API。这就是为什么会创建诸如 Boost、POCO 和 Qt 等库。应用程序可以使用这些库，并依赖于库来处理平台之间的任何差异。

# POSIX 线程

Pthreads 最初是在 1995 年的`POSIX.1c`标准（*Threads extensions*，IEEE Std 1003.1c-1995）中定义的，作为 POSIX 标准的扩展。当时，UNIX 被选择为制造商中立的接口，POSIX 统一了它们之间的各种 API。

尽管有这种标准化的努力，Pthread 在实现它的操作系统之间仍存在差异（例如，在 Linux 和 OS X 之间），这是由于不可移植的扩展（在方法名中标有`_np`）。

对于`pthread_setname_np`方法，Linux 实现需要两个参数，允许设置除当前线程以外的线程的名称。在 OS X（自 10.6 起），此方法只需要一个参数，允许设置当前线程的名称。如果可移植性是一个问题，就必须注意这样的差异。

1997 年后，POSIX 标准的修订由奥斯汀联合工作组负责。这些修订将线程扩展合并到主标准中。当前的修订是第 7 版，也被称为 POSIX.1-2008 和 IEEE Std 1003.1，2013 版--标准的免费副本可在线获得。

操作系统可以获得符合 POSIX 标准的认证。目前，这些如表中所述：

| **名称** | **开发者** | **自版本** | **架构（当前）** | **备注** |
| --- | --- | --- | --- | --- |
| AIX | IBM | 5L | POWER | 服务器操作系统 |
| HP-UX | 惠普 | 11i v3 | PA-RISC, IA-64 (Itanium) | 服务器操作系统 |
| IRIX | Silicon Graphics（SGI） | 6 | MIPS | 已停产 |
| Inspur K-UX | 浪潮 | 2 | X86_64 | 基于 Linux |
| Integrity | Green Hills Software | 5 | ARM, XScale, Blackfin, Freescale Coldfire, MIPS, PowerPC, x86. | 实时操作系统 |
| OS X/MacOS | 苹果 | 10.5（Leopard） | X86_64 | 桌面操作系统 |
| QNX Neutrino | BlackBerry | 1 | Intel 8088, x86, MIPS, PowerPC, SH-4, ARM, StrongARM, XScale | 实时嵌入式操作系统 |
| Solaris | Sun/Oracle | 2.5 | SPARC, IA-32（<11），x86_64，PowerPC（2.5.1） | 服务器操作系统 |
| Tru64 | DEC, HP, IBM, Compaq | 5.1B-4 | Alpha | 已停产 |
| UnixWare | Novell, SCO, Xinuos | 7.1.3 | x86 | 服务器操作系统 |

其他操作系统大多是兼容的。以下是相同的例子：

| **名称** | **平台** | **备注** |
| --- | --- | --- |
| Android | ARM, x86, MIPS | 基于 Linux。Bionic C 库。 |
| BeOS (Haiku) | IA-32, ARM, x64_64 | 仅限于 x86 的 GCC 2.x。 |
| Darwin | PowerPC、x86、ARM | 使用 macOS 基础的开源组件。 |
| FreeBSD | IA-32、x86_64、sparc64、PowerPC、ARM、MIPS 等等 | 基本上符合 POSIX 标准。可以依赖已记录的 POSIX 行为。一般而言，比 Linux 更严格地遵守标准。 |
| Linux | Alpha、ARC、ARM、AVR32、Blackfin、H8/300、Itanium、m68k、Microblaze、MIPS、Nios II、OpenRISC、PA-RISC、PowerPC、s390、S+core、SuperH、SPARC、x86、Xtensa 等等 | 一些 Linux 发行版（见前面的表）被认证为符合 POSIX 标准。这并不意味着每个 Linux 发行版都符合 POSIX 标准。一些工具和库可能与标准不同。对于 Pthreads，这可能意味着在 Linux 发行版之间的行为有时会有所不同（不同的调度程序等），并且与其他实现 Pthreads 的操作系统相比也会有所不同。 |
| MINIX 3 | IA-32、ARM | 符合 POSIX 规范标准 3（SUSv3, 2004）。 |
| NetBSD | Alpha、ARM、PA-RISC、68k、MIPS、PowerPC、SH3、SPARC、RISC-V、VAX、x86 等等 | 几乎完全兼容 POSIX.1（1990），并且大部分符合 POSIX.2（1992）。 |
| Nuclear RTOS | ARM、MIPS、PowerPC、Nios II、MicroBlaze、SuperH 等等 | Mentor Graphics 公司推出的专有 RTOS，面向嵌入式应用。 |
| NuttX | ARM、AVR、AVR32、HCS12、SuperH、Z80 等等 | 轻量级的 RTOS，可在 8 到 32 位系统上扩展，且高度符合 POSIX 标准。 |
| OpenBSD | Alpha、x86_64、ARM、PA-RISC、IA-32、MIPS、PowerPC、SPARC 等等 | 1995 年从 NetBSD 分叉出来。具有类似的 POSIX 支持。 |
| OpenSolaris/illumos | IA-32、x86_64、SPARC、ARM | 与商业 Solaris 发行版兼容认证。 |
| VxWorks | ARM、SH-4、x86、x86_64、MIPS、PowerPC | 符合 POSIX 标准，并获得用户模式执行环境认证。 |

由此可见，遵循 POSIX 规范并不是一件明显的事情，也不能保证代码在每个平台上都能编译。每个平台还会有自己的一套标准扩展，用于标准中省略的但仍然有用的功能。然而，Pthreads 在 Linux、BSD 和类似的软件中被广泛使用。

# Windows 支持

也可以使用 POSIX API，例如以下方式：

| **名称** | **符合度** |
| --- | --- |
| Cygwin | 大部分完整。提供了一个完整的运行时环境，用于将 POSIX 应用程序作为普通的 Windows 应用程序进行分发。 |
| MinGW | 使用 MinGW-w64（MinGW 的重新开发版本），对 Pthreads 的支持相当完整，尽管可能会缺少一些功能。 |
| Windows Subsystem for Linux | WSL 是 Windows 10 的一个功能，允许 Ubuntu Linux 14.04（64 位）镜像的工具和实用程序在其上本地运行，尽管不能运行使用 GUI 功能或缺少内核功能的程序。否则，它提供了与 Linux 类似的兼容性。这个功能目前需要运行 Windows 10 周年更新，并按照微软提供的说明手动安装 WSL。 |

一般不建议在 Windows 上使用 POSIX。除非有充分的理由使用 POSIX（例如，大量现有代码库），否则最好使用跨平台 API（本章后面将介绍），以解决任何平台问题。

在接下来的章节中，我们将看一下 Pthreads API 提供的功能。

# PThreads 线程管理

这些函数都以 `pthread_` 或 `pthread_attr_` 开头。这些函数都适用于线程本身及其属性对象。

使用 Pthreads 的基本线程看起来像下面这样：

```cpp
#include <pthread.h> 
#include <stdlib.h> 

#define NUM_THREADS     5 

```

主要的 Pthreads 头文件是 `pthread.h`。这样可以访问除了信号量（稍后在本节中讨论）之外的所有内容。我们还在这里定义了希望启动的线程数量的常量：

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

我们在前面的函数中使用循环创建所有线程。每个线程实例在创建时都会被分配一个线程 ID（第一个参数），并且`pthread_create()`函数会返回一个结果代码（成功时为零）。线程 ID 是在将来调用中引用线程的句柄。

函数的第二个参数是一个`pthread_attr_t`结构实例，如果没有则为 0。这允许配置新线程的特性，例如初始堆栈大小。当传递零时，将使用默认参数，这些参数因平台和配置而异。

第三个参数是指向新线程将启动的函数的指针。这个函数指针被定义为一个返回指向 void 数据（即自定义数据）的指针的函数，并接受一个指向 void 数据的指针。在这里，作为参数传递给新线程的数据是线程 ID：

```cpp
    for (int i = 0; i < NUM_THREADS; ++i) { 
        result_code = pthread_join(threads[i], 0); 
    } 

    exit(0); 
} 

```

接下来，我们使用`pthread_join()`函数等待每个工作线程完成。此函数接受两个参数，要等待的线程 ID 和`Worker`函数的返回值的缓冲区（或零）。

管理线程的其他函数如下：

+   `void pthread_exit`(`void *value_ptr`):

这个函数终止调用它的线程，使得提供的参数值可以被任何调用`pthread_join()`的线程使用。

+   `int pthread_cancel`(`pthread_t` thread):

这个函数请求取消指定的线程。根据目标线程的状态，这将调用其取消处理程序。

除此之外，还有`pthread_attr_*`函数来操作和获取有关`pthread_attr_t`结构的信息。

# 互斥锁

这些是以`pthread_mutex_`或`pthread_mutexattr_`为前缀的函数。它们适用于互斥锁及其属性对象。

Pthreads 中的互斥锁可以被初始化、销毁、锁定和解锁。它们还可以使用`pthread_mutexattr_t`结构自定义其行为，该结构具有相应的`pthread_mutexattr_*`函数用于初始化和销毁属性。

使用静态初始化的 Pthread 互斥锁的基本用法如下：

```cpp
static pthread_mutex_t func_mutex = PTHREAD_MUTEX_INITIALIZER; 

void func() { 
    pthread_mutex_lock(&func_mutex); 

    // Do something that's not thread-safe. 

    pthread_mutex_unlock(&func_mutex); 
} 

```

在最后一段代码中，我们使用了`PTHREAD_MUTEX_INITIALIZER`宏，它可以为我们初始化互斥锁，而无需每次都输入代码。与其他 API 相比，人们必须手动初始化和销毁互斥锁，尽管宏的使用在一定程度上有所帮助。

之后，我们锁定和解锁互斥锁。还有`pthread_mutex_trylock()`函数，它类似于常规锁定版本，但如果引用的互斥锁已经被锁定，它将立即返回而不是等待它被解锁。

在这个例子中，互斥锁没有被显式销毁。然而，这是 Pthreads 应用程序中正常内存管理的一部分。

# 条件变量

这些函数的前缀要么是`pthread_cond_`，要么是`pthread_condattr_`。它们适用于条件变量及其属性对象。

Pthreads 中的条件变量遵循相同的模式，除了具有初始化和`destroy`函数外，还有用于管理`pthread_condattr_t`属性结构的相同函数。

这个例子涵盖了 Pthreads 条件变量的基本用法：

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

在前面的代码中，我们获取了标准头文件，并定义了一个计数触发器和限制，其目的将在一会儿变得清晰。我们还定义了一些全局变量：一个计数变量，我们希望创建的线程的 ID，以及一个互斥锁和条件变量：

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

前面的函数本质上只是在使用`count_mutex`获得独占访问权后向全局计数变量添加。它还检查计数触发值是否已达到。如果是，它将发出条件变量的信号。

为了给第二个线程，也运行此函数，一个机会获得互斥锁，我们在循环的每个周期中睡眠 1 秒：

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

在这第二个函数中，在检查是否已达到计数限制之前，我们会锁定全局互斥锁。这是我们的保险，以防此函数运行的线程在计数达到限制之前不被调用。

否则，我们在提供条件变量和锁定互斥锁的情况下等待条件变量。一旦收到信号，我们就解锁全局互斥锁，并退出线程。

这里需要注意的一点是，这个示例没有考虑虚假唤醒。Pthreads 条件变量容易受到这种唤醒的影响，这需要使用循环并检查是否已满足某种条件：

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

最后，我们等待每个线程完成，然后在退出之前清理，销毁属性结构实例、互斥锁和条件变量。

使用`pthread_cond_broadcast()`函数，进一步可以向等待条件变量的所有线程发出信号，而不仅仅是队列中的第一个线程。这使得可以更优雅地在某些应用程序中使用条件变量，例如，有很多工作线程在等待新数据集到达，而无需单独通知每个线程。

# 同步

实现同步的函数以`pthread_rwlock_`或`pthread_barrier_`为前缀。这些实现读/写锁和同步屏障。

**读/写锁**（**rwlock**）与互斥锁非常相似，只是它具有额外的功能，允许无限数量的线程同时读取，而只限制写访问一个线程。

使用`rwlock`与使用互斥锁非常相似：

```cpp
#include <pthread.h> 
int pthread_rwlock_init(pthread_rwlock_t* rwlock, const pthread_rwlockattr_t* attr); 
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER; 

```

在最后的代码中，我们包括相同的通用头文件，并使用初始化函数或通用宏。有趣的部分是当我们锁定`rwlock`时，可以仅用于只读访问：

```cpp
int pthread_rwlock_rdlock(pthread_rwlock_t* rwlock); 
int pthread_rwlock_tryrdlock(pthread_rwlock_t* rwlock); 

```

在这里，如果锁已经被锁定，第二种变体会立即返回。也可以按以下方式锁定它以进行写访问：

```cpp
int pthread_rwlock_wrlock(pthread_rwlock_t* rwlock); 
int pthread_rwlock_trywrlock(pthread_rwlock_t * rwlock); 

```

这些函数基本上是相同的，只是在任何给定时间只允许一个写入者，而多个读取者可以获得只读锁。

屏障是 Pthreads 的另一个概念。这些是同步对象，对于一些线程起到屏障的作用。在任何一个线程可以继续执行之前，所有这些线程都必须到达屏障。在屏障初始化函数中，指定了线程计数。只有当所有这些线程都使用`pthread_barrier_wait()`函数调用`barrier`对象后，它们才会继续执行。

# 信号量

如前所述，信号量不是原始 Pthreads 扩展到 POSIX 规范的一部分。出于这个原因，它们在`semaphore.h`头文件中声明。

实质上，信号量是简单的整数，通常用作资源计数。为了使它们线程安全，使用原子操作（检查和锁定）。POSIX 信号量支持初始化、销毁、增加和减少信号量，以及等待信号量达到非零值。

# 线程本地存储（TLC）

使用 Pthreads，TLS 是通过使用键和方法来设置特定于线程的数据来实现的：

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

在工作线程中，我们在堆上分配一个新的整数，并将全局密钥设置为其自己的值。在将全局变量增加 1 之后，其值将为 2，而不管其他线程做什么。我们可以在完成此线程的操作后将全局变量设置为 0，并删除分配的值：

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

设置并使用全局密钥来引用 TLS 变量，然而我们创建的每个线程都可以为此密钥设置自己的值。

虽然线程可以创建自己的密钥，但与本章中正在查看的其他 API 相比，处理 TLS 的这种方法相当复杂。

# Windows 线程

相对于 Pthreads，Windows 线程仅限于 Windows 操作系统和类似系统（例如 ReactOS 和其他使用 Wine 的操作系统）。这提供了一个相当一致的实现，可以轻松地由支持对应的 Windows 版本来定义。

在 Windows Vista 之前，线程支持缺少诸如条件变量之类的功能，同时具有 Pthreads 中找不到的功能。根据一个人的观点，使用 Windows 头文件中定义的无数“类型定义”类型可能也会让人感到烦恼。

# 线程管理

从官方 MSDN 文档示例代码改编的使用 Windows 线程的基本示例如下：

```cpp
#include <windows.h> 
#include <tchar.h> 
#include <strsafe.h> 

#define MAX_THREADS 3 
#define BUF_SIZE 255  

```

在包含一系列 Windows 特定的头文件用于线程函数、字符字符串等之后，我们在`Worker`函数中定义了要创建的线程数以及消息缓冲区的大小。

我们还定义了一个结构类型（通过`void 指针：LPVOID`传递），用于包含我们传递给每个工作线程的示例数据：

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

在`Worker`函数中，我们将提供的参数转换为我们自定义的结构类型，然后使用它将其值打印到字符串上，然后在控制台上输出。

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

在这里，定义了一个错误处理函数，该函数获取最后一个错误代码的系统错误消息。在获取最后一个错误的代码之后，将格式化要输出的错误消息，并显示在消息框中。最后，释放分配的内存缓冲区。

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

在`main`函数中，我们在循环中创建线程，为线程数据分配内存，并在启动线程之前为每个线程生成唯一数据。每个线程实例都传递了自己的唯一参数。

之后，我们等待线程完成并重新加入。这本质上与在 Pthreads 上调用`join`函数的单个线程相同--只是这里，一个函数调用就足够了。

最后，关闭每个线程句柄，并清理之前分配的内存。

# 高级管理

使用 Windows 线程进行高级线程管理包括作业、纤程和线程池。作业基本上允许将多个线程链接到一个单元中，从而可以一次性更改所有这些线程的属性和状态。

纤程是轻量级线程，运行在创建它们的线程的上下文中。创建线程预期自己调度这些纤程。纤程还具有类似 TLS 的**纤程本地存储**（**FLS**）。

最后，Windows 线程 API 提供了一个线程池 API，允许在应用程序中轻松使用这样的线程池。每个进程也都提供了一个默认的线程池。

# 同步

使用 Windows 线程，可以使用临界区、互斥体、信号量、**轻量级读写器**（**SRW**）锁、屏障和变体来实现互斥和同步。

同步对象包括以下内容：

| **名称** | **描述** |
| --- | --- |
| 事件 | 允许使用命名对象在线程和进程之间进行事件信号传递。 |
| 互斥体 | 用于线程间和进程间同步，以协调对共享资源的访问。 |
| 信号量 | 用于线程间和进程同步的标准信号量计数对象。 |
| 可等待定时器 | 可由多个进程使用的定时器对象，具有多种使用模式。 |
| 临界区 | 临界区本质上是限于单个进程的互斥锁，这使得它们比使用互斥锁更快，因为它们不需要内核空间调用。 |
| Slim reader/writer lock | SRW 类似于 Pthreads 中的读/写锁，允许多个读取者或单个写入者线程访问共享资源。 |
| 原子变量访问 | 允许对一系列变量进行原子访问，否则不能保证原子性。这使得线程可以共享变量而无需使用互斥锁。 |

# 条件变量

使用 Windows 线程实现条件变量是相当简单的。它使用临界区（`CRITICAL_SECTION`）和条件变量（`CONDITION_VARIABLE`）以及条件变量函数来等待特定条件变量，或者发出信号。

# 线程本地存储

**线程本地存储**（**TLS**）与 Windows 线程类似于 Pthreads，因为首先必须创建一个中央键（TLS 索引），然后各个线程可以使用该全局索引来存储和检索本地值。

与 Pthreads 一样，这涉及到相似数量的手动内存管理，因为 TLS 值必须手动分配和删除。

# Boost

Boost 线程是 Boost 库集合中相对较小的一部分。然而，它被用作成为 C++11 中多线程实现基础，类似于其他 Boost 库最终完全或部分地成为新的 C++标准。有关多线程 API 的详细信息，请参阅本章中的 C++线程部分。

C++11 标准中缺少的功能，在 Boost 线程中是可用的，包括以下内容：

+   线程组（类似于 Windows 作业）

+   线程中断（取消）

+   带超时的线程加入

+   其他互斥锁类型（C++14 改进）

除非绝对需要这些功能，或者无法使用支持 C++11 标准（包括 STL 线程）的编译器，否则没有理由使用 Boost 线程而不是 C++11 实现。

由于 Boost 提供了对本机操作系统功能的封装，使用本机 C++线程可能会减少开销，具体取决于 STL 实现的质量。

# Qt

Qt 是一个相对高级的框架，这也反映在其多线程 API 中。Qt 的另一个定义特征是，它包装了自己的代码（QApplication 和 QMainWindow），并使用元编译器（`qmake`）来实现其信号-槽架构和框架的其他定义特征。

因此，Qt 的线程支持不能直接添加到现有代码中，而是需要调整代码以适应框架。

# QThread

在 Qt 中，`QThread`类不是一个线程，而是一个围绕线程实例的广泛封装，它添加了信号-槽通信、运行时支持和其他功能。这在 QThread 的基本用法中得到体现，如下面的代码所示：

```cpp
class Worker : public QObject { 
    Q_OBJECT 

    public: 
        Worker(); 
        ~Worker(); 

    public slots: 
        void process(); 

    signals: 
        void finished(); 
        void error(QString err); 

    private: 
}; 

```

上述代码是一个基本的`Worker`类，它将包含我们的业务逻辑。它派生自`QObject`类，这也允许我们使用信号-槽和其他固有的`QObject`特性。信号-槽架构在其核心本质上只是一种方式，允许侦听器注册（连接到）由 QObject 派生类声明的信号，从而实现跨模块、跨线程和异步通信。

它有一个可以调用以开始处理的单一方法，并且有两个信号——一个用于表示完成，一个用于表示错误。

实现如下所示：

```cpp
Worker::Worker() { }  
Worker::~Worker() { } 

void Worker::process() { 
    qDebug("Hello World!"); 
    emit finished(); 
} 

```

构造函数可以扩展以包括参数。任何在`process()`方法中分配的堆分配变量（使用`malloc`或`new`）必须在`process()`方法中分配，而不是在构造函数中，因为`Worker`实例将在其中运行线程上下文中操作，我们马上就会看到。

要创建一个新的 QThread，我们将使用以下设置：

```cpp
QThread* thread = new QThread; 
Worker* worker = new Worker(); 
worker->moveToThread(thread); 
connect(worker, SIGNAL(error(QString)), this, SLOT(errorString(QString))); 
connect(thread, SIGNAL(started()), worker, SLOT(process())); 
connect(worker, SIGNAL(finished()), thread, SLOT(quit())); 
connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater())); 
connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater())); 
thread->start(); 

```

基本过程是在堆上创建一个新的 QThread 实例（这样它就不会超出范围），以及我们的`Worker`类的堆分配实例。然后使用其`moveToThread()`方法将新的工作线程移动到新的线程实例中。

接下来，将连接各种信号到相关的槽，包括我们自己的`finished()`和`error()`信号。线程实例的`started()`信号将连接到我们的工作线程上的槽，以启动它。

最重要的是，必须将工作线程的某种完成信号连接到线程上的`quit()`和`deleteLater()`槽。然后将线程的`finished()`信号连接到工作线程上的`deleteLater()`槽。这将确保在工作线程完成时清理线程和工作线程实例。

# 线程池

Qt 提供线程池。这些需要从`QRunnable`类继承，并实现`run()`函数。然后将此自定义类的实例传递给线程池的`start`方法（全局默认池或新池）。然后线程池会处理此工作线程的生命周期。

# 同步

Qt 提供以下同步对象：

+   `QMutex`

+   `QReadWriteLock`

+   `QSemaphore`

+   `QWaitCondition`（条件变量）

这些应该是相当不言自明的。Qt 的信号-槽架构的另一个好处是，它还允许在线程之间异步通信，而无需关注低级实现细节。

# QtConcurrent

QtConcurrent 命名空间包含针对编写多线程应用程序的高级 API，旨在使编写多线程应用程序成为可能，而无需关注低级细节。

函数包括并发过滤和映射算法，以及允许在单独线程中运行函数的方法。所有这些都返回一个`QFuture`实例，其中包含异步操作的结果。

# 线程本地存储

Qt 通过其`QThreadStorage`类提供 TLS。它处理指针类型值的内存管理。通常，人们会将某种数据结构设置为 TLS 值，以存储每个线程的多个值，例如在`QThreadStorage`类文档中描述的那样：

```cpp
QThreadStorage<QCache<QString, SomeClass> > caches; 

void cacheObject(const QString &key, SomeClass* object) { 
    caches.localData().insert(key, object); 
} 

void removeFromCache(const QString &key) { 
    if (!caches.hasLocalData()) { return; } 

    caches.localData().remove(key); 
} 

```

# POCO

POCO 库是围绕操作系统功能的相当轻量级的包装器。它不需要 C++11 兼容的编译器或任何种类的预编译或元编译。

# 线程类

`Thread`类是围绕操作系统级线程的简单包装器。它接受从`Runnable`类继承的`Worker`类实例。官方文档提供了一个基本示例如下：

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

上述代码是一个非常简单的“Hello world”示例，其中一个工作线程只通过标准输出输出一个字符串。线程实例分配在堆栈上，并在入口函数的范围内等待工作线程使用`join()`函数完成。

在许多线程函数中，POCO 非常类似于 Pthreads，尽管在配置线程和其他对象等方面有明显的偏差。作为 C++库，它使用类方法设置属性，而不是填充结构并将其作为参数传递。

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

工作实例被添加到线程池中，并运行它。当我们添加另一个工作实例、更改容量或调用`joinAll()`时，线程池会清理空闲一定时间的线程。因此，单个工作线程将加入，没有活动线程后，应用程序退出。

# 线程本地存储（TLS）

在 POCO 中，TLS 被实现为一个类模板，允许人们将其用于几乎任何类型。

根据官方文档的详细说明：

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

在前面的工作示例中，我们使用`ThreadLocal`类模板创建了一个静态 TLS 变量，并定义它包含一个整数。

因为我们将其定义为静态的，它将只在每个线程中创建一次。为了使用我们的 TLS 变量，我们可以使用箭头（`->`）或星号（`*`）运算符来访问其值。在这个例子中，我们在`for`循环的每个周期增加 TLS 值，直到达到限制为止。

这个例子演示了两个线程将生成它们自己的一系列 10 个整数，计数相同的数字而互不影响。

# 同步

POCO 提供的同步原语如下：

+   互斥

+   FastMutex

+   事件

+   条件

+   信号量

+   RWLock

这里需要注意的是`FastMutex`类。这通常是一种非递归互斥类型，但在 Windows 上是递归的。这意味着人们通常应该假设任一类型在同一线程中可以被同一线程多次锁定。

人们还可以使用`ScopedLock`类与互斥体一起使用，它确保封装的互斥体在当前作用域结束时被释放。

事件类似于 Windows 事件，不同之处在于它们仅限于单个进程。它们构成了 POCO 中条件变量的基础。

POCO 条件变量的功能与 Pthreads 等方式基本相同，不同之处在于它们不会出现虚假唤醒。通常条件变量会出现这些随机唤醒以进行优化。通过不必须明确检查条件变量等待返回时是否满足其条件，减轻了开发者的负担。

# C++线程

C++中的本地多线程支持在第五章中得到了广泛的覆盖，*本地 C++线程和原语*。

正如本章中 Boost 部分所述，C++多线程支持在很大程度上基于 Boost 线程 API，几乎使用相同的头文件和名称。API 本身再次让人联想到 Pthreads，尽管在某些方面有显著的不同，例如条件变量。

即将发布的章节将专门使用 C++线程支持作为示例。

# 整合

在本章涵盖的 API 中，只有 Qt 多线程 API 可以被认为是真正的高级。尽管其他 API（包括 C++11）具有一些更高级的概念，包括线程池和不需要直接使用线程的异步运行器，但 Qt 提供了一个完整的信号-槽架构，使得线程间通信异常容易。

正如本章所述，这种便利也伴随着成本，即必须开发自己的应用程序以适应 Qt 框架。这可能在项目中是不可接受的。

哪种 API 是正确的取决于人们的需求。然而，可以相对公平地说，当人们可以使用诸如 C++11 线程、POCO 等 API 时，直接使用 Pthreads、Windows 线程等并没有太多意义，这些 API 可以在不显著降低性能的情况下简化开发过程，并在各个平台上获得广泛的可移植性。

所有这些 API 在其核心功能上至少在某种程度上是可比较的。

# 总结

在本章中，我们详细研究了一些较流行的多线程 API 和框架，将它们并列起来，以了解它们的优势和劣势。我们通过一些示例展示了如何使用这些 API 来实现基本功能。

在下一章中，我们将详细讨论如何同步线程并在它们之间进行通信。
