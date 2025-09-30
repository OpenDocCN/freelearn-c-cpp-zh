# 第十五章

# 线程执行

正如我们在上一章中解释的，在 POSIX 兼容系统中，可以使用**多线程**或**多进程**方法中的任何一个来实现并发。由于这两个主题都非常广泛，因此它们被分为四个单独的章节，以便为每个主题提供所需的覆盖范围：

+   **多线程方法**将在本章和第十六章“线程同步”中讨论。

+   **多进程方法**将在第十七章“进程执行”和第十八章“进程同步”中介绍。

在本章中，我们将探讨线程的结构以及可以用来创建和管理线程的 API。在下一章，即第十六章“线程同步”中，我们将研究多线程环境中的并发控制机制，以研究它们应该如何解决与并发相关的问题。

多进程的概念是将并发引入软件，通过将逻辑分解为并发进程来实现，这最终导致多进程软件。由于多线程和多进程之间存在现有差异，我们决定将多进程的讨论移至两个单独的章节。

相比之下，多线程，前两章的重点，局限于单进程系统。这是关于线程的最基本事实，也是我们首先关注它的原因。

在上一章中，我们简要介绍了多线程和多进程之间的差异和相似之处。在本章中，我们将专注于多线程，并探讨它们应该如何使用，以便在单个进程中无缝运行多个执行线程。

本章涵盖了以下主题：

+   我们首先讨论线程。本节中解释了**用户线程**和**内核线程**，并讨论了线程的一些最重要的属性。这些属性有助于我们更好地理解多线程环境。

+   然后我们进入下一节，该节专门介绍使用**POSIX 线程库**进行的基本编程，简称**pthread**库。这个库是主要的标准化库，允许我们在 POSIX 系统上开发并发程序，但这并不意味着不兼容 POSIX 的操作系统不支持并发。对于像 Microsoft Windows 这样的不兼容操作系统，它们仍然能够提供自己的 API 来开发并发程序。POSIX 线程库为线程和进程提供支持。然而，在本章中，我们的重点是线程部分，我们将探讨 pthread 库如何被用来创建线程并进一步管理它。

+   在进一步的研究中，我们还演示了在某些使用 pthread 库的示例 C 代码中产生的竞态条件和数据竞争。这为我们在下一章继续讨论*线程同步*奠定了基础。

**注意**：

为了能够完全理解我们将要讨论的多线程方法，强烈建议你在进入第十六章*线程同步*之前完成本章。这是因为本章引入的主题贯穿于我们下一章将要探讨的线程同步的第二部分。 

在继续之前，请记住，在本章中，我们只将涵盖 POSIX 线程库的基本用法。深入探讨 POSIX 线程库的多个有趣元素超出了本书的范围，因此，建议你花些时间更详细地探索 pthread 库，并通过书面示例获得足够的实践，以便你能够熟悉它。POSIX 线程库的更高级用法将在本书剩余的章节中演示。

然而，现在，让我们深入探讨线程的概念，从概述我们所知道的一切开始。这是我们理解的关键要素，因为我们将在本章剩余的页面上介绍其他关键概念。

# 线程

在上一章中，我们讨论了线程作为多线程方法的一部分，当你在 POSIX 兼容的操作系统上编写并发程序时可以使用这种方法。

在本节中，你将找到关于线程你应该知道的所有内容的回顾。我们还将引入一些与我们将要讨论的主题相关的新信息。请记住，所有这些信息都将作为继续开发多线程程序的基础。

每个线程都是由一个进程初始化的。然后它将永远属于该进程。不可能有一个共享的线程或将线程的所有权转让给另一个进程。每个进程至少有一个线程，即它的*主线程*。在一个 C 程序中，`main`函数作为主线程的一部分被执行。

所有线程共享相同的**进程 ID**（**PID**）。如果你使用`top`或`htop`等工具，可以很容易地看到线程共享相同的进程 ID，并且被分组在其下。不仅如此，所有线程的属性都继承自其所属进程，例如，组 ID、用户 ID、当前工作目录和信号处理器。例如，线程的当前工作目录与其所属进程相同。

每个线程都有一个独特且专用的**线程 ID**（**TID**）。这个 ID 可以用来向该线程传递信号或在进行调试时跟踪它。你将看到在 POSIX 线程中，线程 ID 可以通过`pthread_t`变量访问。此外，每个线程还有一个专用的信号屏蔽，可以用来过滤它可能接收到的信号。

同一进程内的所有线程都可以访问该进程内其他线程打开的所有*文件描述符*。因此，所有线程都可以读取或修改那些文件描述符背后的资源。这也适用于*套接字描述符*和打开的*套接字*。在接下来的章节中，你将了解更多关于文件描述符和套接字的内容。

线程可以使用在第十四章中介绍的进程的所有技术来共享或传递状态。请注意，在共享位置（如数据库）中共享状态与在网络上传输它（例如）是不同的，这导致了两种不同的 IPC 技术类别。我们将在未来的章节中回到这一点。

在这里，你可以找到线程在 POSIX 兼容系统中可以用来共享或传递状态的列表：

+   所属进程的内存（数据、堆栈和堆段）。这种方法*仅*适用于线程，不适用于进程。

+   文件系统。

+   内存映射文件。

+   网络（使用互联网套接字）。

+   线程间的信号传递。

+   共享内存。

+   POSIX 管道。

+   Unix 域套接字。

+   POSIX 消息队列。

+   环境变量。

在处理线程属性时，同一进程内的所有线程可以使用该进程的内存空间来存储和维护共享状态。这是在多个线程之间共享状态最常见的方式。进程的堆段通常用于此目的。

线程的生命周期依赖于其所属进程的生命周期。当一个进程被*杀死*或*终止*时，该进程所属的所有线程也将被终止。

当主线程结束时，进程会立即退出。然而，如果有其他*分离*的线程正在运行，进程会等待它们全部完成后再终止。分离线程将在解释 POSIX 中线程创建时进行说明。

创建线程的进程可以是内核进程。同时，它也可以是在用户空间中启动的用户进程。如果进程是内核，则该线程被称为*内核级线程*或简称为*内核线程*，否则，该线程被称为*用户级线程*。内核线程通常执行重要逻辑，因此它们比用户线程有更高的优先级。例如，设备驱动程序可能使用内核线程来等待硬件信号。

与可以访问相同内存区域的用户线程类似，内核线程也能够访问内核的内存空间，从而能够访问内核中的所有过程和单元。

在整本书中，我们将主要讨论用户线程，而不是内核线程。这是因为与用户线程一起工作的 API 由 POSIX 标准提供。但是，没有标准接口用于创建和管理内核线程，它们仅针对每个内核特定。

创建和管理内核线程超出了本书的范围。因此，从现在开始，当我们使用术语*线程*时，我们指的是用户线程，而不是内核线程。

用户不能直接创建线程。用户需要首先启动一个进程，因为只有进程的主线程才能启动另一个线程。请注意，只有线程可以创建线程。

关于线程的内存布局，每个线程都有自己的栈内存区域，可以视为为该线程专用的私有内存区域。然而，在实践中，当有指针指向它时，其他线程（在同一进程内）也可以访问它。

您应该记住，所有这些栈区域都是同一进程内存空间的一部分，并且可以被同一进程中的任何线程访问。

关于同步技术，用于同步进程的相同控制机制也可以用于同步多个线程。信号量、互斥锁和条件变量是可用于同步线程的工具之一，以及进程。

当其线程同步且没有进一步的数据竞争或竞争条件可以观察到时，程序通常被称为*线程安全*程序。同样，一个库或一组函数，可以轻松地用于多线程程序而不会引入任何新的并发问题，被称为*线程安全库*。作为程序员，我们的目标是生成线程安全的代码。

**注意**:

在以下链接中，您可以找到有关 POSIX 线程及其共享属性的信息。以下链接是关于 POSIX 线程接口的 NTPL 实现。这是针对 Linux 环境的，但其中大部分也适用于其他类 Unix 操作系统。

http://man7.org/linux/man-pages/man7/pthreads.7.html.

在本节中，我们探讨了有关线程的一些基础概念和属性，以便更好地理解即将到来的章节。您将在我们讨论各种多线程示例时看到许多这些属性的实际应用。

下一节将向您介绍如何创建 POSIX 线程的第一个代码示例。这一节将会很简单，因为它只涉及 POSIX 中线程的基本知识。这些基础知识将引导我们进入更高级的主题。

# POSIX 线程

本节专门介绍 POSIX 线程 API，也称为*pthread 库*。这个 API 非常重要，因为它是创建和管理 POSIX 兼容操作系统中的线程的主要 API。

在非 POSIX 兼容的操作系统，例如 Microsoft Windows 中，应该有另一个为这个目的设计的 API，并且可以在该操作系统的文档中找到。例如，在 Microsoft Windows 的情况下，线程 API 作为 Windows API 的一部分提供，称为 Win32 API。这是关于 Microsoft 的[Windows 线程 API](https://docs.microsoft.com/en-us/windows/desktop/procthread/proce)的文档链接：[`docs.microsoft.com/en-us/windows/desktop/procthread/process-and-thread-functions`](https://docs.microsoft.com/en-us/windows/desktop/procthread/process-and-thread-functions)。

然而，作为 C11 的一部分，我们期望有一个统一的 API 来处理线程。换句话说，无论你是在为 POSIX 系统还是非 POSIX 系统编写程序，你都应该能够使用 C11 提供的相同 API。虽然这是非常理想的，但在当前这个时间点，在各种 C 标准实现中，如 glibc，对这种通用 API 的支持并不多。

要继续讨论这个主题，pthread 库简单地说是一组*头文件*和*函数*，可以用来在 POSIX 兼容的操作系统上编写多线程程序。每个操作系统都有自己的 pthread 库实现。这些实现可能与其他 POSIX 兼容操作系统的实现完全不同，但最终，它们都公开了相同的接口（API）。

一个著名的例子是**原生 POSIX 线程库**，简称**NPTL**，它是 Linux 操作系统中 pthread 库的主要实现。

如 pthread API 所述，所有线程功能都通过包含头文件`pthread.h`来提供。还有一些对 pthread 库的扩展，只有当你包含`semaphore.h`时才可用。例如，其中一个扩展涉及特定于信号量的操作，例如创建信号量、初始化它、销毁它等。

POSIX 线程库公开了以下功能。由于我们在前面的章节中已经对它们进行了详细解释，因此你应该很熟悉：

+   线程管理，包括线程创建、线程连接和线程分离

+   互斥锁

+   信号量

+   条件变量

+   各种类型的锁，如自旋锁和递归锁

为了解释前面的功能，我们必须从`pthread_`前缀开始。所有 pthread 函数都以这个前缀开始。这适用于所有情况，除了信号量，它最初不是 POSIX 线程库的一部分，后来作为扩展添加。在这种情况下，函数将以`sem_`前缀开始。

在本章的后续部分，我们将看到如何在编写多线程程序时使用一些前面的功能。首先，我们将学习如何创建 POSIX 线程以与主线程并发运行。在这里，我们将了解 `pthread_create` 和 `pthread_join` 函数，它们分别属于用于 *创建* 和 *连接* 线程的主要 API。

# 创建 POSIX 线程

在前几章中，我们已经学习了诸如交织、锁、互斥锁和条件变量等基本概念，并在本章介绍了 POSIX 线程的概念，现在是时候编写一些代码了。

第一步是创建一个 POSIX 线程。在本节中，我们将演示如何使用 POSIX 线程 API 在进程内创建新线程。接下来的 *示例 15.1* 描述了如何创建一个执行简单任务（如将字符串打印到输出）的线程：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
// This function contains the logic which should be run
// as the body of a separate thread
void* thread_body(void* arg) {
  printf("Hello from first thread!\n");
  return NULL;
}
int main(int argc, char** argv) {
  // The thread handler
  pthread_t thread;
  // Create a new thread
  int result = pthread_create(&thread, NULL, thread_body, NULL);
  // If the thread creation did not succeed
  if (result) {
    printf("Thread could not be created. Error number: %d\n",
            result);
    exit(1);
  }
  // Wait for the created thread to finish
  result = pthread_join(thread, NULL);
  // If joining the thread did not succeed
  if (result) {
    printf("The thread could not be joined. Error number: %d\n",
            result);
    exit(2);
  }
  return 0;
}
```

代码框 15-1 [ExtremeC_examples_chapter15_1.c]：创建一个新的 POSIX 线程

在 *代码框 15-1* 中看到的示例代码创建了一个新的 POSIX 线程。这是本书中第一个包含两个线程的示例。所有之前的示例都是单线程的，代码始终在主线程中运行。

让我们解释一下我们刚刚看到的代码。在顶部，我们包含了一个新的头文件：`pthread.h`。这是一个标准头文件，它公开了所有 pthread 功能。我们需要这个头文件，以便我们可以引入 `pthread_create` 和 `pthread_join` 函数的声明。

在 `main` 函数之前，我们声明了一个新的函数：`thread_body`。这个函数遵循一个特定的签名。它接受一个 `void*` 指针并返回另一个 `void*` 指针。作为一个提醒，`void*` 是一个通用指针类型，可以表示任何其他指针类型，如 `int*` 或 `double*`。

因此，这是 C 函数可以拥有的最一般签名。这是由 POSIX 标准强制的，所有希望成为线程（用作线程逻辑）的 *伴随函数* 的函数都应该遵循这个通用签名。这就是为什么我们定义了 `thread_body` 函数是这样的。

**注意**:

`main` 函数是主线程逻辑的一部分。当主线程被创建时，它作为其逻辑的一部分执行 `main` 函数。这意味着在 `main` 函数之前和之后可能还有其他代码被执行。

回到代码，作为 `main` 函数中的第一条指令，我们声明了一个类型为 `pthread_t` 的变量。这是一个线程句柄变量，在其声明时，它不指向任何特定的线程。换句话说，这个变量还没有包含任何有效的线程 ID。只有成功创建了一个线程之后，这个变量才包含对新创建线程的有效句柄。

创建线程后，线程句柄实际上指的是新创建线程的线程 ID。虽然线程 ID 是操作系统中的线程标识符，但线程句柄是程序中线程的表示。大多数情况下，存储在线程句柄中的值与线程 ID 相同。每个线程都能通过获取一个指向自身的 `pthread_t` 变量来访问其线程 ID。一个线程可以使用 `pthread_self` 函数来获取一个自引用的句柄。我们将在未来的示例中演示这些函数的用法。

线程创建发生在调用 `pthread_create` 函数时。如您所见，我们已经将 `thread` 句柄变量的地址传递给 `pthread_create` 函数，以便将其填充为适当的句柄（或线程 ID），指向新创建的线程。

第二个参数确定线程的属性。每个线程都有一些属性，如 *堆栈大小*、*堆栈地址* 和 *分离状态*，可以在创建线程之前进行配置。

我们将展示更多如何配置这些属性以及它们如何影响线程行为的示例。如果第二个参数传递了 `NULL`，这意味着新线程应该使用其属性的默认值。因此，在先前的代码中，我们创建了一个具有默认属性值的线程。

传递给 `pthread_create` 的第三个参数是一个函数指针。这个指针指向线程的 *伴随函数*，其中包含了线程的逻辑。在先前的代码中，线程的逻辑是在 `thread_body` 函数中定义的。因此，应该传递其地址以便将其绑定到句柄变量 `thread` 上。

第四个也是最后一个参数是线程逻辑的输入参数，在我们的例子中是 `NULL`。这意味着我们不希望向函数传递任何内容。因此，`thread_body` 函数中的参数 `arg` 在线程执行时将是 `NULL`。在下一节提供的示例中，我们将看看如何向这个函数传递一个值而不是 `NULL`。

所有 pthread 函数，包括 `pthread_create`，在成功执行后都应返回零。因此，如果返回了除零以外的任何数字，则意味着函数已失败，并返回了一个 *错误号*。

注意，使用 `pthread_create` 创建线程并不意味着线程的逻辑会立即执行。这是一个调度问题，无法预测新线程何时获得一个 CPU 核心并开始执行。

在创建线程后，我们加入新创建的线程，但这究竟意味着什么呢？正如我们之前解释的，每个进程都以一个线程开始，这个线程是*主线程*。除了主线程，其父进程是拥有进程外，所有其他线程都有一个*父线程*。在默认情况下，如果主线程完成，进程也将完成。当进程被终止时，所有其他正在运行或休眠的线程也将立即被终止。

因此，如果创建了一个新线程，它还没有开始运行（因为它还没有获得 CPU 的使用权），同时，父进程被终止（无论什么原因），线程将在执行第一条指令之前就死亡。因此，主线程需要等待第二个线程通过加入它来执行并完成。

线程只有在它的伴随函数返回时才完成。在前面的例子中，派生的线程在`thread_body`伴随函数返回时完成，这发生在函数返回`NULL`时。当新派生的线程完成时，被`pthread_join`调用阻塞的主线程被释放并可以继续，这最终导致程序成功终止。

如果主线程没有加入新创建的线程，那么新派生的线程根本不可能被执行。正如我们之前解释的，这是因为主线程在派生线程进入执行阶段之前就已经退出。

我们也应该记住，创建一个线程并不足以使其被执行。创建的线程可能需要一段时间才能获得访问 CPU 核心的权限，并通过这种方式最终开始运行。如果在此时，进程被终止，那么新创建的线程将没有机会成功运行。

现在我们已经讨论了代码的设计，*Shell Box 15-1*显示了运行*example 15.1*的输出：

```cpp
$ gcc ExtremeC_examples_chapter15_1.c -o ex15_1.out -lpthread
$ ./ex15_1.out
Hello from first thread!
$
```

Shell Box 15-1：构建和运行示例 15.1

正如你在前面的 shell 框中看到的，我们需要在编译命令中添加`-lpthread`选项。这样做是因为我们需要将我们的程序与现有的 pthread 库实现链接。在某些平台，如 macOS，即使没有`-lpthread`选项，你的程序也可能被链接；然而，强烈建议在链接使用 pthread 库的程序时使用此选项。这条建议的重要性在于确保你的*构建脚本*在任何平台上都能工作，并在构建 C 项目时防止任何跨兼容性问题。

可以被加入的线程被称为*可加入的*。线程默认是可加入的。与可加入线程相反，我们有*分离的*线程。分离的线程不能被加入。

在 *示例 15.1* 中，主线程可以分离新产生的线程而不是连接它。这样，我们就让进程知道，它必须等待分离线程完成才能终止。请注意，在这种情况下，主线程可以退出，而父进程不会被终止。

在本节的最后代码中，我们想要使用分离线程重写前面的示例。而不是连接新创建的线程，主线程将其设置为分离，然后退出。这样，尽管主线程已经退出，但进程仍然会继续运行，直到第二个线程完成：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
// This function contains the logic which should be run
// as the body of a separate thread
void* thread_body(void* arg) {
  printf("Hello from first thread!\n");
  return NULL;
}
int main(int argc, char** argv) {
  // The thread handler
  pthread_t thread;
  // Create a new thread
  int result = pthread_create(&thread, NULL, thread_body, NULL);
  // If the thread creation did not succeed
  if (result) {
    printf("Thread could not be created. Error number: %d\n",
            result);
    exit(1);
  }
  // Detach the thread
  result = pthread_detach(thread);
  // If detaching the thread did not succeed
  if (result) {
    printf("Thread could not be detached. Error number: %d\n",
            result);
    exit(2);
  }
  // Exit the main thread
  pthread_exit(NULL);
  return 0;
}
```

代码框 15-2 [ExtremeC_examples_chapter15_1_2.c]: 示例 15.1 生成分离线程

上述代码的输出与之前使用可连接线程编写的代码完全相同。唯一的区别是我们管理新创建线程的方式。

在新线程创建后，主线程立即将其分离。然后，主线程退出。指令 `pthread_exit(NULL)` 是必要的，以便让进程知道它应该等待其他分离线程完成。如果线程没有被分离，进程会在主线程退出时终止。

**注意**：

*分离状态* 是在创建新线程之前可以设置的一个线程属性，以便使其分离。这是创建新分离线程的另一种方法，而不是在可连接线程上调用 `pthread_detach`。区别在于，这种方式下，新创建的线程从一开始就是分离的。

在下一节中，我们将介绍我们的第一个示例，演示竞态条件。我们将使用本节中介绍的所有函数来编写未来的示例。因此，你将有机会在不同的场景中再次回顾它们。

# 竞态条件示例

对于我们的第二个示例，我们将探讨一个更具挑战性的场景。*示例 15.2*，如 *代码框 15-3* 所示，展示了交织是如何发生的，以及我们在实践中无法可靠地预测示例的最终输出，这主要是因为并发系统的非确定性本质。该示例涉及一个程序，几乎同时创建了三个线程，并且每个线程都打印不同的字符串。

以下代码的最终输出包含三个不同线程打印的字符串，但顺序不可预测。如果以下示例的不可变约束（在前一章中介绍）是要在输出中看到特定的字符串顺序，那么以下代码将无法满足该约束，主要是因为不可预测的交织。让我们看看以下代码框：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
void* thread_body(void* arg) {
  char* str = (char*)arg;
  printf("%s\n", str);
  return NULL;
}
int main(int argc, char** argv) {
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  pthread_t thread3;
  // Create new threads
  int result1 = pthread_create(&thread1, NULL,
          thread_body, "Apple");
  int result2 = pthread_create(&thread2, NULL,
          thread_body, "Orange");
  int result3 = pthread_create(&thread3, NULL,
          thread_body, "Lemon");
  if (result1 || result2 || result3) {
    printf("The threads could not be created.\n");
    exit(1);
  }

  // Wait for the threads to finish
  result1 = pthread_join(thread1, NULL);
  result2 = pthread_join(thread2, NULL);
  result3 = pthread_join(thread3, NULL);
  if (result1 || result2 || result3) {
    printf("The threads could not be joined.\n");
    exit(2);
  }
  return 0;
}
```

代码框 15-3 [ExtremeC_examples_chapter15_2.c]: 示例 15.2 向输出打印三个不同的字符串

我们刚才看到的代码与为 *example 15.1* 编写的代码非常相似，但它创建了三个线程而不是一个。在这个例子中，我们为所有三个线程使用相同的伴随函数。

如前述代码所示，我们向 `pthread_create` 函数传递了第四个参数，而在我们之前的例子 *15.1* 中，它是 `NULL`。这些参数将通过 `thread_body` 伴随函数中的通用指针参数 `arg` 被线程访问。

在 `thread_body` 函数内部，线程将通用指针 `arg` 强制转换为 `char*` 指针，并使用 `printf` 函数从该地址开始打印字符串。这就是我们能够向线程传递参数的方式。同样，它们的大小并不重要，因为我们只传递一个指针。

如果你需要在创建线程时向其发送多个值，你可以使用一个结构来包含这些值，并传递一个指向填充了所需值的结构变量的指针。我们将在下一章的 *线程同步* 中演示如何做到这一点。

**注意**：

我们可以将指针传递给线程的事实意味着新线程应该能够访问主线程可以访问的相同内存区域。然而，访问并不限于拥有进程内存中的特定段或区域，并且所有线程都可以完全访问进程中的栈、堆、文本和数据段。

如果你多次运行 *example 15.2*，你会看到打印的字符串顺序可以变化，因为每次运行都预计会打印相同的字符串，但顺序不同。

*Shell Box 15-2* 展示了在连续三次运行后 *example 15.2* 的编译和输出：

```cpp
$ gcc ExtremeC_examples_chapter15_2.c -o ex15_2.out -lpthread
$ ./ex15_2.out
Apple
Orange
Lemon
$ ./ex15_2.out
Orange
Apple
Lemon
$ ./ex15_2.out
Apple
Orange
Lemon
$
```

Shell Box 15-2：运行示例 15.2 三次以观察现有的竞态条件和各种交织情况

产生第一个和第二个线程在第三个线程之前打印它们的字符串的交织情况很容易，但要产生第三个线程打印其字符串 `Lemon` 作为输出中的第一个或第二个字符串的交织情况就困难得多。然而，这肯定会发生，尽管概率很低。你可能需要多次运行示例才能产生那种交织。这可能需要一些耐心。

上述代码也被认为不是线程安全的。这是一个重要的定义；一个多线程程序只有在没有根据定义的不变约束条件出现竞态条件的情况下才是线程安全的。因此，由于上述代码存在竞态条件，它不是线程安全的。我们的任务就是通过使用将在下一章中介绍的正确控制机制来使上述代码成为线程安全的。

如前一个示例的输出所示，我们并没有在 `Apple` 或 `Orange` 的字符之间看到任何交织。例如，我们没有看到以下输出：

```cpp
$ ./ex15_2.out
AppOrle
Ange
Lemon
$
```

15-3 号 Shell 盒：对于上述示例不会发生的想象中的输出

这表明`printf`函数是*线程安全的*，这仅仅意味着无论交织如何发生，当一个线程正在打印字符串时，其他线程中的`printf`实例不会打印任何内容。

此外，在前面给出的代码中，`thread_body`伴随函数在三个不同的线程的上下文中运行了三次。在之前的章节中，以及在给出多线程示例之前，所有函数都是在主线程的上下文中执行的。从现在起，每个函数调用都发生在特定线程的上下文中（不一定是主线程）。

两个线程无法启动单个函数调用。原因很明显，因为每个函数调用都需要创建一个*栈帧*，这个栈帧应该放在只有一个线程的栈顶上，而两个不同的线程有两个不同的栈区域。因此，函数调用只能由一个线程启动。换句话说，两个线程可以分别调用同一个函数，这会导致两个单独的函数调用，但它们不能共享同一个函数调用。

我们应该注意，传递给线程的指针不应该是一个*悬空指针*。这会导致一些严重的内存问题，难以追踪。作为提醒，悬空指针指向内存中的一个地址，该地址没有分配的变量。更具体地说，这种情况是，在某个时刻，那里可能原本有一个变量或数组，但到了指针即将被使用的时候，它已经被释放了。

在前面的代码中，我们向每个线程传递了三个字面量。由于这些字符串字面量所需的内存是从数据段分配的，而不是从堆或栈段分配的，因此它们的地址永远不会被释放，`arg`指针也不会变成悬空。

很容易将前面的代码写成指针悬空的形式。下面是同样的代码，但使用了悬空指针，你很快就会看到这会导致不良的内存行为：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
void* thread_body(void* arg) {
  char* str = (char*)arg;
  printf("%s\n", str);
  return NULL;
}
int main(int argc, char** argv) {
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
  pthread_t thread3;
  char str1[8], str2[8], str3[8];
  strcpy(str1, "Apple");
  strcpy(str2, "Orange");
  strcpy(str3, "Lemon");
  // Create new threads
  int result1 = pthread_create(&thread1, NULL, thread_body, str1);
  int result2 = pthread_create(&thread2, NULL, thread_body, str2);
  int result3 = pthread_create(&thread3, NULL, thread_body, str3);
  if (result1 || result2 || result3) {
    printf("The threads could not be created.\n");
    exit(1);
  }
  // Detach the threads
  result1 = pthread_detach(thread1);
  result2 = pthread_detach(thread2);
  result3 = pthread_detach(thread3);
  if (result1 || result2 || result3) {
    printf("The threads could not be detached.\n");
    exit(2);
  }
  // Now, the strings become deallocated.
  pthread_exit(NULL);
  return 0;
}
```

代码盒 15-4 [ExtremeC_examples_chapter15_2_1.c]：从主线程的栈区域分配字面量的 15.2 示例

前面的代码几乎与*示例 15.2*中给出的代码相同，但有两点不同。

首先，传递给线程的指针并不是指向数据段中驻留的字符串字面量，而是指向从主线程的栈区域分配的字符数组。作为`main`函数的一部分，这些数组已经被声明，在接下来的几行中，它们被一些字符串字面量填充。

我们需要记住，字符串字面量仍然驻留在数据段中，但声明后的数组在用`strcpy`函数填充后现在具有与字符串字面量相同的值。

第二个区别是关于主线程的行为。在之前的代码中，它加入了线程，但在这段代码中，它解除了线程并立即退出。这将释放主线程栈上声明的数组，在某些交错中，其他线程可能会尝试读取这些已释放的区域。因此，在某些交错中，传递给线程的指针可能会变成悬空。

**注意**：

一些约束，如没有崩溃、没有悬空指针以及通常没有内存相关的问题，都可以被视为程序的不变约束的一部分。因此，在某些交错中产生悬空指针问题的并发系统肯定存在严重的竞态条件。

为了能够检测悬空指针，你需要使用一个*内存分析器*。作为一个更简单的方法，你可以运行程序多次，等待崩溃发生。然而，你并不总是有幸看到这一点，在这个例子中，我们也没有看到崩溃。

为了检测这个例子中的不良内存行为，我们将使用`valgrind`。你还记得我们在*第四章*、*进程内存结构*和*第五章*、*栈和堆*中介绍了这个内存分析器，用于查找*内存泄漏*。回到这个例子，我们想用它来找到发生不良内存访问的地方。

值得记住的是，使用悬空指针并访问其内容，并不一定会导致崩溃。这在之前的代码中尤其如此，其中的字符串被放置在主线程的栈上。

当其他线程运行时，栈段保持与主线程退出时相同，因此即使`str1`、`str2`和`str3`数组在离开`main`函数时被释放，你仍然可以访问这些字符串。换句话说，在 C 或 C++中，运行时环境不会检查指针是否悬空，它只是遵循语句的顺序。

如果一个悬空指针及其底层内存被更改，那么可能会发生像崩溃或逻辑错误这样的坏事，但只要底层内存是*未触及的*，使用悬空指针可能不会导致崩溃，这是非常危险且难以追踪的。

简而言之，仅仅因为你可以通过悬空指针访问一个内存区域，并不意味着你被允许访问该区域。这就是为什么我们需要使用像`valgrind`这样的内存分析器，它会报告这些无效的内存访问。

在下面的 shell 框中，我们编译程序，并使用`valgrind`运行两次。在第一次运行中，没有发生任何坏事，但在第二次运行中，`valgrind`报告了内存访问错误。

*Shell Box 15-4*显示了第一次运行：

```cpp
$ gcc -g ExtremeC_examples_chapter15_2_1.c -o ex15_2_1.out -lpthread
$ valgrind ./ex15_2_1.out
==1842== Memcheck, a memory error detector
==1842== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==1842== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==1842== Command: ./ex15_2_1.out
==1842==
Orange
Apple
Lemon
==1842==
==1842== HEAP SUMMARY:
==1842==     in use at exit: 0 bytes in 0 blocks
==1842==   total heap usage: 9 allocs, 9 frees, 3,534 bytes allocated
==1842==
==1842== All heap blocks were freed -- no leaks are possible
==1842==
==1842== For counts of detected and suppressed errors, rerun with: -v
==1842== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
$
```

Shell Box 15-4：第一次使用 valgrind 运行示例 15.2

在第二次运行中，`valgrind` 报告了一些内存访问问题（注意，当你运行它时，完整的输出将可查看，但为了篇幅考虑，我们已进行了精简）：

```cpp
$ valgrind ./ex15_2_1.out
==1854== Memcheck, a memory error detector
==1854== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==1854== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==1854== Command: ./ex15_2_1.out
==1854==
Apple
Lemon
==1854== Thread 4:
==1854== Conditional jump or move depends on uninitialised value(s)
==1854==    at 0x50E6A65: _IO_file_xsputn@@GLIBC_2.2.5 (fileops.c:1241)
==1854==    by 0x50DBA8E: puts (ioputs.c:40)
==1854==    by 0x1087C9: thread_body (ExtremeC_examples_chapter15_2_1.c:17)
==1854==    by 0x4E436DA: start_thread (pthread_create.c:463)
==1854==    by 0x517C88E: clone (clone.S:95)
==1854==
...
==1854==
==1854== Syscall param write(buf) points to uninitialised byte(s)
==1854==    at 0x516B187: write (write.c:27)
==1854==    by 0x50E61BC: _IO_file_write@@GLIBC_2.2.5 (fileops.c:1203)
==1854==    by 0x50E7F50: new_do_write (fileops.c:457)
==1854==    by 0x50E7F50: _IO_do_write@@GLIBC_2.2.5 (fileops.c:433)
==1854==    by 0x50E8402: _IO_file_overflow@@GLIBC_2.2.5 (fileops.c:798)
==1854==    by 0x50DBB61: puts (ioputs.c:41)
==1854==    by 0x1087C9: thread_body (ExtremeC_examples_chapter15_2_1.c:17)
==1854==    by 0x4E436DA: start_thread (pthread_create.c:463)
==1854==    by 0x517C88E: clone (clone.S:95)
...
==1854==
Orange
==1854==
==1854== HEAP SUMMARY:
==1854==     in use at exit: 272 bytes in 1 blocks
==1854==   total heap usage: 9 allocs, 8 frees, 3,534 bytes allocated
==1854==
==1854== LEAK SUMMARY:
==1854==    definitely lost: 0 bytes in 0 blocks
==1854==    indirectly lost: 0 bytes in 0 blocks
==1854==      possibly lost: 272 bytes in 1 blocks
==1854==    still reachable: 0 bytes in 0 blocks
==1854==         suppressed: 0 bytes in 0 blocks
==1854== Rerun with --leak-check=full to see details of leaked memory
==1854==
==1854== For counts of detected and suppressed errors, rerun with: -v
==1854== Use --track-origins=yes to see where uninitialised values come from
==1854== ERROR SUMMARY: 13 errors from 3 contexts (suppressed: 0 from 0)
$
```

Shell Box 15-5：第二次运行示例 15.2 并使用 valgrind

如你所见，第一次运行顺利，没有内存访问问题，尽管上述竞争条件对我们来说仍然很明显。然而，在第二次运行中，当其中一个线程试图访问由 `str2` 指向的字符串 `Orange` 时，出现了问题。

这意味着传递给第二个线程的指针已经悬空。在前面的输出中，你可以清楚地看到堆栈跟踪指向 `thread_body` 函数内部的行，那里有 `printf` 语句。请注意，堆栈跟踪实际上指的是 `puts` 函数，因为我们的 C 编译器已将 `printf` 语句替换为等效的 `puts` 语句。前面的输出还显示，`write` 系统调用正在使用一个名为 `buf` 的指针，该指针指向一个未初始化或分配的内存区域。

观察前面的例子，`valgrind` 并不结论指针是否悬空。它只是报告无效的内存访问。

在关于不良内存访问的错误信息之前，你可以看到即使读取 `Orange` 的访问是无效的，字符串 `Orange` 仍然被打印出来。这仅仅表明，当我们的代码以并发方式运行时，事情可以多么容易变得复杂。

在本节中，我们向前迈出了重要的一步，了解了编写不安全的代码是多么容易。接下来，我们将演示另一个有趣的例子，该例子会产生数据竞争。在这里，我们将看到对 pthread 库及其各种函数的更复杂使用。

# 数据竞争示例

*示例 15.3* 展示了数据竞争。在先前的例子中，我们没有共享状态，但在这个例子中，我们将有两个线程之间共享的变量。

本例的不变约束是保护共享状态的数据完整性，以及所有其他明显的约束，如没有崩溃、没有不良内存访问等。换句话说，输出看起来如何无关紧要，但线程在共享变量的值被其他线程更改且写入线程不知道最新值时，不得写入新值。这就是我们所说的“数据完整性”：

```cpp
#include <stdio.h>
#include <stdlib.h>
// The POSIX standard header for using pthread library
#include <pthread.h>
void* thread_body_1(void* arg) {
  // Obtain a pointer to the shared variable
  int* shared_var_ptr = (int*)arg;
  // Increment the shared variable by 1 by writing
  // directly to its memory address
  (*shared_var_ptr)++;
  printf("%d\n", *shared_var_ptr);
  return NULL;
}
void* thread_body_2(void* arg) {
  // Obtain a pointer to the shared variable
  int* shared_var_ptr = (int*)arg;
  // Increment the shared variable by 2 by writing
  // directly to its memory address
  *shared_var_ptr += 2;
  printf("%d\n", *shared_var_ptr);
  return NULL;
}
int main(int argc, char** argv) {
  // The shared variable
  int shared_var = 0;
  // The thread handlers
  pthread_t thread1;
  pthread_t thread2;
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
  return 0;
}
```

代码框 15-5 [ExtremeC_examples_chapter15_3.c]：示例 15.3，两个线程操作单个共享变量

共享状态已在 `main` 函数的第一行声明。在这个例子中，我们处理的是主线程堆栈区域分配的单个整型变量，但在实际应用中可能要复杂得多。整型变量的初始值为零，每个线程通过写入其内存位置直接增加其值。

在这个例子中，没有局部变量在每个线程中保留共享变量值的副本。然而，您应该小心线程中的增量操作，因为它们不是*原子*操作，因此可能会经历不同的交错。我们已在上一章中详细解释了这一点。

每个线程都可以通过在其伴随函数中通过参数`arg`接收到的指针来更改共享变量的值。正如您在两次调用`pthread_create`中可以看到的，我们将变量`shared_var`的地址作为第四个参数传递。

值得注意的是，指针在线程中永远不会成为悬垂指针，因为主线程不会退出，它通过连接线程来等待线程完成。

*Shell Box 15-6*展示了前面代码多次运行的输出，以产生不同的交错。请记住，我们希望共享变量`shared_var`的数据完整性得到保留。

因此，根据`thread_body_1`和`thread_body_2`中定义的逻辑，我们只能有`1 3`和`2 3`作为可接受的输出：

```cpp
$ gcc ExtremeC_examples_chapter15_3.c -o ex15_3.out -lpthread
$ ./ex15_3.out
1
3
$
...
...
...
$ ./ex15_3.out
3
1
$
...
...
...
$ ./ex15_3.out
1
2
$
```

Shell Box 15-6：示例 15.3 的多次运行，最终我们看到共享变量的数据完整性没有得到保留

如您所见，最后一次运行表明共享变量的数据完整性条件没有得到满足。

在最后一次运行中，第一个线程，即具有`thread_body_1`作为其伴随函数的线程，读取了共享变量的值，它是`0`。

第二个线程，即具有`thread_body_2`作为其伴随函数的线程，也读取了共享值，它是`0`。在此之后，两个线程都试图增加共享变量的值并立即打印它。这是对数据完整性的违反，因为当一个线程正在操作共享状态时，另一个线程不应该能够写入它。

正如我们之前所解释的，在这个例子中，我们对`shared_var`有明确的数据竞争。

**注意**：

当您自己执行*示例 15.3*时，请耐心等待，以查看`1 2`输出。这可能在运行可执行文件 100 次之后发生！我曾在 macOS 和 Linux 上观察到数据竞争。

为了解决前面的数据竞争，我们需要使用控制机制，例如信号量或互斥锁，来同步对共享变量的访问。在下一章中，我们将向前面的代码引入互斥锁，这将为我们完成这项工作。

# 摘要

本章是我们使用 POSIX 线程库在 C 中编写多线程程序的第一步。作为本章的一部分：

+   我们学习了 POSIX 线程库的基础知识，这是在 POSIX 兼容系统中编写多线程应用程序的主要工具。

+   我们探讨了线程及其内存结构的各种属性。

+   我们对线程可用的通信和共享状态的机制提供了一些见解。

+   我们解释了对于同一进程内所有线程可用的内存区域是共享数据和通信的最佳方式。

+   我们讨论了内核线程和用户级线程，以及它们之间的差异。

+   我们解释了可连接线程和分离线程，以及它们在执行点上的区别。

+   我们演示了如何使用`pthread_create`和`pthread_join`函数以及它们接收的参数。

+   使用实际的 C 代码演示了竞态条件和数据竞争的例子，并展示了如何使用悬垂指针可能导致严重的内存问题，最终可能发生崩溃或逻辑错误。

在接下来的章节中，我们将通过探讨并发相关问题和可用的机制来防止和解决这些问题，继续并发展我们对多线程的讨论。
