# 第十七章

# 进程执行

现在我们已经准备好讨论由多个进程组成的整体架构的软件系统。这些系统通常被称为多进程或多个进程系统。本章以及下一章试图涵盖多进程的概念，并进行利弊分析，以便与我们在第十五章“线程执行”、第十六章“线程同步”中讨论的多线程进行比较。

在本章中，我们的重点是可用的 API 和技术来启动一个新的进程以及进程执行实际上是如何发生的，在下一章中，我们将探讨由多个进程组成的并发环境。我们将解释各种状态如何在多个进程之间共享，以及在多进程环境中访问共享状态的常见方式。

本章的一部分基于比较多进程和多线程环境。此外，我们还简要地讨论了单主机多进程系统和分布式多进程系统。

# 进程执行 API

每个程序都是以进程的形式执行的。在我们拥有进程之前，我们只有一个包含一些内存段和可能大量机器级指令的可执行二进制文件。相反，每个进程都是一个正在执行的程序的独立实例。因此，单个编译程序（或可执行二进制文件）可以通过不同的进程多次执行。事实上，这就是为什么我们关注的是进程，而不是程序本身。

在前两个章节中，我们讨论了单进程软件中的线程，但为了实现本章的目标，我们将讨论具有多个进程的软件。但首先，我们需要了解如何以及通过哪个 API 来生成一个新的进程。

注意，我们主要关注在类 Unix 操作系统中执行进程，因为它们都遵循 Unix 洋葱架构并公开非常知名且相似的 API。其他操作系统可能有它们自己执行进程的方式，但由于它们大多数或多或少遵循 Unix 洋葱架构，我们期望看到类似的过程执行方法。

在类 Unix 操作系统中，在系统调用级别执行进程的方法并不多。如果你还记得*第十一章*，*系统调用 与 内核*中的*内核环*，它是在*硬件环*之后的内部环，它为外部环、*shell*和*用户*提供*系统调用接口*，以便它们执行各种内核特定的功能。其中两个暴露的系统调用是专门用于进程创建和进程执行的；分别是`fork`和`exec`（Linux 中的`execve`）。在*进程创建*中，我们创建一个新进程，但在*进程执行*中，我们使用一个现有进程作为宿主，并用一个新的程序替换它；因此，在进程执行中不会创建新的进程。

由于使用这些系统调用，程序总是以新进程的形式执行，但这个过程并不总是被创建！`fork`系统调用创建一个新进程，而`exec`系统调用则用一个新的进程替换调用者（宿主）进程。我们将在后面讨论`fork`和`exec`系统调用的区别。在那之前，让我们看看这些系统调用是如何暴露给外部环的。

如我们在*第十章*，*Unix – 历史 与 架构*中所述，我们有两个针对类 Unix 操作系统的标准，特别是关于它们应该从其 shell 环中暴露的接口。这些标准是**单一 Unix 规范**（**SUS**）和**POSIX**。有关这些标准的更多信息，包括它们的相似之处和不同之处，请参阅*第十章*，*Unix – 历史 与 架构*。

应从 shell 环中暴露的接口在 POSIX 接口中得到了详细规定，实际上，标准中确实有部分内容涉及进程执行和进程管理。

因此，我们预计在 POSIX 中会找到用于进程创建和进程执行的头部和函数。这些函数确实存在，并且我们可以在提供所需功能的不同头部文件中找到它们。以下是负责进程创建和进程执行的 POSIX 函数列表：

+   可以在`unistd.h`头文件中找到的`fork`函数负责进程创建。

+   可以在`spawn.h`头文件中找到的`posix_spawn`和`posix_spawnp`函数。这些函数负责进程创建。

+   例如，在`unistd.h`头文件中可以找到的`exec*`函数组，如`execl`和`execlp`。这些函数负责进程执行。

注意，前面的函数不应与 `fork` 和 `exec` 系统调用混淆。这些函数是来自 shell 环境暴露的 POSIX 接口的一部分，而系统调用则是来自内核环暴露的。虽然大多数 Unix-like 操作系统都是 POSIX 兼容的，但我们也可以有一个非 Unix-like 系统也是 POSIX 兼容的。那么，前面的函数存在于该系统中，但系统调用级别的进程创建的底层机制可能不同。

一个具体的例子是使用 Cygwin 或 MinGW 使 Microsoft Windows 兼容 POSIX。通过安装这些程序，你可以编写和编译使用 POSIX 接口的标准 C 程序，从而使 Microsoft Windows 部分兼容 POSIX，但在 Microsoft Windows 中没有 `fork` 或 `exec` 系统调用！这实际上既令人困惑又非常重要，你应该知道 shell 环境并不一定暴露与内核环暴露相同的接口。

**注意**：

你可以在 Cygwin 中找到 `fork` 函数的实现细节：https://github.com/openunix/cygwin/blob/master/winsup/cygwin/fork.cc。注意，它并没有调用通常存在于 Unix-like 内核中的 `fork` 系统调用；相反，它包含了 Win32 API 的头文件，并调用了一些关于进程创建和进程管理的知名函数。

根据 POSIX 标准，Unix-like 系统上的 shell 环境暴露出来的不仅仅是 C 标准库。当使用终端时，有一些预先编写的 shell 实用程序被用来提供复杂的 C 标准 API 的使用。关于进程创建，每当用户在终端中输入一个命令时，就会创建一个新的进程。

即使是简单的 `ls` 或 `sed` 命令也会启动一个新的进程，这个进程可能只持续不到一秒钟。你应该知道，这些实用程序大多是用 C 语言编写的，并且它们正在消耗与你在编写自己的程序时所使用的相同的精确 POSIX 接口。

Shell 脚本也是在单独的进程中执行的，但方式略有不同。我们将在未来的章节中讨论如何在 Unix-like 系统中执行进程。

进程创建发生在内核中，尤其是在单核内核中。每当用户进程启动一个新的进程或甚至一个新的线程时，请求会被系统调用接口接收，并传递到内核环。在那里，为传入的请求创建一个新的 *任务*，无论是进程还是线程。

类似于 Linux 或 FreeBSD 这样的单核内核会跟踪内核内的任务（进程和线程），这使得在内核本身创建进程变得合理。

注意，每当内核中创建一个新的任务时，它会被放入 *任务调度单元* 的队列中，并且它可能需要一点时间才能获得 CPU 并开始执行。

为了创建一个新的进程，需要一个父进程。这就是为什么每个进程都有一个父进程。实际上，每个进程只能有一个父进程。父亲和祖父母的链条可以追溯到第一个用户进程，通常称为 *init*，而内核进程是其父进程。

它是 Unix-like 系统中所有其他进程的祖先，存在于系统关闭之前。通常，init 进程成为所有 *孤儿进程* 的父进程，这些进程的父进程已经终止，这样就不会有进程没有父进程。

这种父子关系最终会形成一个大的进程树。这个树可以通过命令工具 *pstree* 来检查。我们将在未来的示例中展示如何使用这个工具。

现在，我们知道了可以执行新进程的 API，我们需要给出一些实际的 C 语言示例来说明这些方法是如何实际工作的。我们首先从 fork API 开始，它最终调用 `fork` 系统调用。

## 进程创建

正如我们在上一节中提到的，fork API 可以用来创建一个新的进程。我们还解释了，新的进程只能作为正在运行进程的子进程来创建。在这里，我们看到了一些示例，展示了进程如何使用 fork API 来创建新的子进程。

为了创建一个新的子进程，父进程需要调用 `fork` 函数。`fork` 函数的声明可以从 `unistd.h` 头文件中包含，它是 POSIX 头文件的一部分。

当调用 `fork` 函数时，会创建调用进程（称为父进程）的一个精确副本，并且两个进程从 `fork` 调用语句之后的下一个指令开始并发运行。请注意，子进程（或被 fork 的进程）从父进程继承了包括所有内存段及其内容在内的大量内容。因此，它有权访问数据、堆栈和堆段中的相同变量，以及文本段中的程序指令。我们将在接下来的段落中讨论其他继承的内容，在讨论示例之后。

由于我们现在有两个不同的进程，`fork` 函数会返回两次；一次在父进程中，另一次在子进程中。此外，`fork` 函数对每个进程返回不同的值。它对子进程返回 0，对父进程返回 forked（或子）进程的 PID。*示例 17.1* 展示了 `fork` 在其最简单用法中的工作方式：

```cpp
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  printf("This is the parent process with process ID: %d\n",
          getpid());
  printf("Before calling fork() ...\n");
  pid_t ret = fork();
  if (ret) {
    printf("The child process is spawned with PID: %d\n", ret);
  } else {
    printf("This is the child process with PID: %d\n", getpid());
  }
  printf("Type CTRL+C to exit ...\n");
  while (1);
  return 0;
}
```

代码框 17-1 [ExtremeC_examples_chapter17_1.c]: 使用 fork API 创建子进程

在前面的代码框中，我们使用了 `printf` 来打印一些日志，以便跟踪进程的活动。正如你所见，我们调用了 `fork` 函数来创建一个新的进程。显然，它不接受任何参数，因此其使用非常简单直接。

在调用`fork`函数后，一个新的进程从调用进程（现在是父进程）中分叉（或克隆）出来，之后，它们作为两个不同的进程继续并发工作。

当然，对`fork`函数的调用将在系统调用级别上引发进一步的调用，然后，内核中的负责逻辑才能创建一个新的分叉进程。

在`return`语句之前，我们使用了一个无限循环来保持两个进程同时运行并防止它们退出。请注意，进程最终应该达到这个无限循环，因为它们在文本段中具有完全相同的指令。

我们有意保持进程运行，以便能够在`pstree`和`top`命令显示的进程列表中看到它们。在此之前，我们需要编译前面的代码，看看新的进程是如何通过*Shell Box 17-1*进行分叉的：

```cpp
$ gcc ExtremeC_examples_chapter17_1.c -o ex17_1.out
$ ./ex17_1.out
This is the parent process with process ID: 10852
Before calling fork() …
The child process is spawned with PID: 10853
This is the child process with PID: 10853
Type CTRL+C to exit ...
$
```

Shell Box 17-1：构建和运行示例 17.1

如你所见，父进程打印其 PID，那是`10852`。请注意，PID 将在每次运行时改变。在分叉子进程后，父进程打印`fork`函数返回的 PID，它是`10853`。

在下一行，子进程打印其 PID，再次是`10853`，这与父进程从`fork`函数接收到的相符。最后，两个进程都进入无限循环，给我们一些时间在探测工具中观察它们。

如你在*Shell Box 17-1*中看到的那样，分叉进程从其父进程继承了相同的`stdout`文件描述符和相同的终端。因此，它可以打印到其父进程写入的相同输出。分叉进程从其父进程继承了在`fork`函数调用时的所有打开文件描述符。

此外，还有其他继承属性，可以在`fork`的手册页中找到。Linux 的`fork`手册页可以在以下链接中找到：http://man7.org/linux/man-pages/man2/fork.2.html。

如果你打开链接并查看属性，你会看到有一些属性是父进程和分叉进程之间共享的，还有一些属性是不同的，并且针对每个进程，例如，PID、父 PID、线程等。

使用像`pstree`这样的实用程序可以很容易地看到进程之间的父子关系。每个进程都有一个父进程，所有进程共同构建了一个大树。请记住，每个进程只有一个父进程，一个进程不能有两个父进程。

尽管前一个例子中的过程陷入了无限循环，但我们可以使用`pstree`实用命令来查看系统中所有进程的列表，这些进程以树状结构显示。以下是在 Linux 机器上使用`pstree`的输出。请注意，`pstree`命令默认安装在 Linux 系统上，但在其他类 Unix 操作系统中可能需要安装：

```cpp
$ pstree -p
systemd(1)─┬─accounts-daemon(877)─┬─{accounts-daemon}(960)
           │                      └─{accounts-daemon}(997)
...
...
...
           ├─systemd-logind(819)
           ├─systemd-network(673)
           ├─systemd-resolve(701)
           ├─systemd-timesyn(500)───{systemd-timesyn}(550)
           ├─systemd-udevd(446)
           └─tmux: server(2083)─┬─bash(2084)───pstree(13559)
                                └─bash(2337)───ex17_1.out(10852)───ex17_1.out(10853)
$
```

Shell 框 17-2：使用`pstree`查找作为示例 17.1 一部分生成的进程

如*Shell 框 17-2*的最后行所示，我们有两个进程，其 PID 分别为`10852`和`10853`，它们处于父子关系。请注意，进程`10852`的父进程 PID 为`2337`，这是一个*bash*进程。

有趣的是，在最后一行之前的一行中，我们可以看到`pstree`进程本身作为 PID 为`2084`的 bash 进程的子进程。这两个 bash 进程都属于同一个 PID 为`2083`的*tmux*终端模拟器。

在 Linux 中，第一个进程是*调度器*进程，它是内核镜像的一部分，其 PID 为 0。下一个进程，通常称为*init*，其 PID 为 1，它是调度器进程创建的第一个用户进程。它从系统启动存在直到系统关闭。所有其他用户进程都是`init`进程的直接或间接子进程。失去父进程的进程成为孤儿进程，它们被`init`进程作为其直接子进程收养。

然而，在几乎所有著名 Linux 发行版的较新版本中，`init`进程已被*systemd 守护进程*取代，这就是为什么您在*Shell 框 17-2*的第一行看到`systemd(1)`的原因。以下链接是一个很好的资源，可以阅读更多关于`init`和`systemd`之间的差异以及为什么 Linux 发行版开发者做出了这样的决定：[`www.tecmint.com/systemd-replaces-init-in-linux`](https://www.tecmint.com/systemd-replaces-init-in-linux)。

当使用`fork`API 时，父进程和派生进程是并发执行的。这意味着我们应该能够检测到并发系统的某些行为。

可以观察到的最著名的行为是一些交错。如果您不熟悉这个术语或者以前没有听说过，强烈建议您阅读*第十三章*，*并发*，和*第十四章*，*同步*。

以下示例，*示例 17.2*，展示了父进程和派生进程可以具有非确定性的交错。我们将打印一些字符串，并观察在两次连续运行中可能发生的各种交错：

```cpp
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  pid_t ret = fork();
  if (ret) {
    for (size_t i = 0; i < 5; i++) {
      printf("AAA\n");
      usleep(1);
    }
  } else {
    for (size_t i = 0; i < 5; i++) {
      printf("BBBBBB\n");
      usleep(1);
    }
  }
  return 0;
}
```

代码框 17-2 [ExtremeC_examples_chapter17_2.c]：两个进程向标准输出打印一些行

上述代码与我们为*示例 17.1*编写的代码非常相似。它创建了一个分支进程，然后父进程和分支进程向标准输出打印一些文本行。父进程打印`AAA`五次，分支进程打印`BBBBBB`五次。以下是对同一编译可执行文件连续两次运行的输出：

```cpp
$ gcc ExtremeC_examples_chapter17_2.c -o ex17_2.out
$ ./ex17_2.out
AAA
AAA
AAA
AAA
AAA
BBBBBB
BBBBBB
BBBBBB
BBBBBB
BBBBBB
$ ./ex17_2.out
AAA
AAA
BBBBBB
AAA
AAA
BBBBBB
BBBBBB
BBBBBB
AAA
BBBBBB
$
```

Shell 框 17-3：示例 17.2 连续两次运行的输出

从前面的输出中很明显，我们有不同的交错。这意味着如果我们根据标准输出的内容定义我们的不变约束，我们可能会在这里遭受潜在的竞争条件。这最终会导致我们在编写多线程代码时遇到的所有问题，我们需要使用类似的方法来克服这些问题。在下一章中，我们将更详细地讨论这些解决方案。

在下一节中，我们将讨论进程执行以及如何使用`exec*`函数实现它。

## 进程执行

执行新进程的另一种方式是使用`exec*`函数家族。与 fork API 相比，这个函数族在执行新进程时采取不同的方法。`exec*`函数背后的哲学是首先创建一个简单的基进程，然后在某个时刻，加载目标可执行文件并将其作为新的*进程映像*替换基进程。进程映像是可执行文件的加载版本，其内存段已分配，并准备好执行。在未来的章节中，我们将讨论加载可执行文件的不同步骤，并更深入地解释进程映像。

因此，在使用`exec*`函数时，不会创建新的进程，而是发生进程替换。这是`fork`和`exec*`函数之间最重要的区别。不是通过分支新的进程，而是将基础进程完全替换为一个新的内存段和代码指令集。

*代码框 17-3*，包含*示例 17.3*，展示了`execvp`函数，它是`exec*`函数家族中的一个函数，是如何用来启动一个 echo 进程的。`execvp`函数是`exec*`函数组中的一个函数，它从父进程继承了环境变量`PATH`，并像父进程一样搜索可执行文件：

```cpp
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
int main(int argc, char** argv) {
  char *args[] = {"echo", "Hello", "World!", 0};
  execvp("echo", args);
  printf("execvp() failed. Error: %s\n", strerror(errno));
  return 0;
}
```

代码框 17-3 [ExtremeC_examples_chapter17_3.c]：展示`execvp`的工作原理

如您在先前的代码框中看到的那样，我们调用了`execvp`函数。正如我们之前解释的那样，`execvp`函数从基础进程继承了环境变量`PATH`，以及它查找现有可执行文件的方式。它接受两个参数；第一个是要加载和执行的可执行文件或脚本的名称，第二个是要传递给可执行文件的参数列表。

注意，我们传递的是`echo`而不是绝对路径。因此，`execvp`应该首先定位`echo`可执行文件。这些可执行文件可以位于类 Unix 操作系统的任何位置，从`/usr/bin`到`/usr/local/bin`或甚至其他地方。可以通过遍历`PATH`环境变量中找到的所有目录路径来找到`echo`的绝对位置。

`exec*`函数可以执行一系列可执行文件。以下是一些可以通过`exec*`函数执行的一些文件格式列表：

+   ELF 可执行文件

+   包含指示脚本*解释器*的*shebang*行的脚本文件

+   传统`a.out`格式的二进制文件

+   ELF FDPIC 可执行文件

找到`echo`可执行文件后，`execvp`完成剩余的工作。它使用一组准备好的参数调用`exec`（在 Linux 中为`execve`）系统调用，然后内核从找到的可执行文件中准备进程映像。当一切准备就绪时，内核用准备好的映像替换当前进程映像，基本进程永远消失。现在，控制权返回到新进程，它从其`main`函数开始执行，就像正常执行一样。

由于这个过程，如果在`execvp`函数调用语句之后，`printf`语句无法被执行，因为现在我们有一个全新的进程，拥有新的内存段和新的指令。如果`execvp`语句没有成功，那么`printf`应该被执行，这是`execvp`函数调用失败的标志。

正如我们之前所说的，我们有一组`exec*`函数，而`execvp`函数只是其中之一。虽然它们的行为相似，但它们之间有一些细微的差别。接下来，你可以找到这些函数的比较：

+   `execl(const char* path, const char* arg0, ..., NULL)`: 接受指向可执行文件的绝对路径以及一系列应该传递给新进程的参数。它们必须以空字符串、`0`或`NULL`结尾。如果我们想使用`execl`重写*示例 17.3*，我们会使用`execl("/usr/bin/echo", "echo", "Hello", "World", NULL)`。

+   `execlp(const char* file, const char* arg0, ..., NULL)`: 接受一个相对路径作为其第一个参数，但由于它可以访问`PATH`环境变量，它可以轻松地定位可执行文件。然后，它接受一系列应该传递给新进程的参数。它们必须以空字符串、`0`或`NULL`结尾。如果我们想使用`execlp`重写*示例 17.3*，我们会使用`execlp("echo", "echo," "Hello," "World," NULL)`。

+   `execle(const char* path, const char* arg0, ..., NULL, const char* env0, ..., NULL)`: 作为其第一个参数接受指向可执行文件的绝对路径。然后，它接受一系列应传递给新进程的参数，后面跟一个空字符串。随后，它接受一系列表示环境变量的字符串。它们也必须以空字符串结尾。如果我们想使用 `execle` 重写 *示例 17.3*，我们将使用 `execle("/usr/bin/echo", "echo", "Hello", "World", NULL, "A=1", "B=2", NULL)`。请注意，在这个调用中，我们向新进程传递了两个新的环境变量，`A` 和 `B`。

+   `execv(const char* path, const char* args[])`: 接受指向可执行文件的绝对路径以及应传递给新进程的参数数组。数组中的最后一个元素必须是一个空字符串，`0` 或 `NULL`。如果我们想使用 `execl` 重写 *示例 17.3*，我们将使用 `execl("/usr/bin/echo", args)`，其中 `args` 的声明如下：`char* args[] = {"echo", "Hello", "World", NULL}`。

+   `execvp(const char* file, const char* args[])`: 它接受一个相对路径作为其第一个参数，但由于它可以访问 `PATH` 环境变量，因此可以轻松地定位可执行文件。然后，它接受一个数组，该数组包含应传递给新进程的参数。数组中的最后一个元素必须是一个空字符串，`0` 或 `NULL`。这是我们在 *示例 17.3* 中使用的函数。

当 `exec*` 函数成功时，之前的进程就消失了，取而代之的是一个新的进程。因此，根本就没有第二个进程。因此，我们无法像对 `fork` API 那样演示交错。在下一节中，我们将比较 `fork` API 和 `exec*` 函数以执行新程序。

## 比较进程创建和进程执行

基于我们之前的讨论和前几节给出的示例，我们可以对用于执行新程序的两个方法进行比较：

+   成功调用 `fork` 函数会导致两个独立进程的结果；一个调用 `fork` 函数的父进程和一个分叉的（或子）进程。但任何 `exec*` 函数的成功调用都会导致调用进程被新的进程映像替换，因此不会创建新的进程。

+   调用 `fork` 函数会复制父进程的所有内存内容，分叉进程会看到相同的内存内容和变量。但调用 `exec*` 函数会破坏基本进程的内存布局，并基于加载的可执行文件创建一个新的布局。

+   一个分叉进程可以访问父进程的某些属性，例如，打开的文件描述符，但使用 `exec*` 函数。新的进程对此一无所知，并且它不会从基本进程继承任何内容。

+   在这两个 API 中，我们最终得到一个只有一个主线程的新进程。父进程中的线程不是使用`fork` API 进行克隆的。

+   可以使用`exec*` API 运行脚本和外部可执行文件，但只能使用`fork` API 创建一个实际上是相同 C 程序的新进程。

在下一节中，我们将讨论大多数内核加载和执行新进程所采取的步骤。这些步骤及其细节因内核而异，但我们尽力涵盖大多数已知内核执行进程所采取的一般步骤。

# 进程执行步骤

要从可执行文件执行进程，大多数操作系统中的用户空间和内核空间需要采取一些通用步骤。正如我们在上一节中提到的，可执行文件大多是可执行对象文件，例如 ELF、Mach 或需要解释器来执行它们的脚本文件。

从用户环的角度来看，应该调用像`exec`这样的系统调用。请注意，我们在这里不解释`fork`系统调用，因为它实际上不是执行。它更多的是当前运行进程的克隆操作。

当用户空间调用`exec`系统调用时，内核内部会创建一个新的执行可执行文件请求。内核试图根据其类型找到指定的可执行文件的处理程序，并根据该处理程序，使用*加载程序*来加载可执行文件的内容。

注意，对于脚本文件，解释程序的可执行二进制文件通常在脚本的第一行的*shebang 行*中指定。为了执行进程，加载程序有以下职责：

+   它检查请求执行的用户的执行上下文和权限。

+   它从主内存为新进程分配内存。

+   它将可执行文件的二进制内容复制到分配的内存中。这主要涉及数据和文本段。

+   它为栈段分配一个内存区域，并准备初始内存映射。

+   创建主线程及其栈内存区域。

+   它将命令行参数作为*栈帧*复制到主线程栈区域的顶部。

+   它初始化执行所需的必要寄存器。

+   它执行程序入口点的第一条指令。

在脚本文件的情况下，脚本文件的路径被复制为解释器进程的命令行参数。大多数内核都采取这些一般步骤，但实现细节可能因内核而异。

要了解更多关于特定操作系统的信息，你需要查看其文档或简单地通过谷歌搜索。以下来自 LWN 的文章是那些寻求更多关于 Linux 进程执行细节的人的绝佳起点：[`lwn.net/Articles/631631/`](https://lwn.net/Articles/631631/) 和 [`lwn.net/Articles/630727/`](https://lwn.net/Articles/630727/)。

在下一节中，我们将开始讨论与并发相关的话题。我们为下一章做准备，下一章将深入探讨多进程特定的同步技术。我们首先从讨论共享状态开始，这些状态可以在多进程软件系统中使用。

# 共享状态

与线程一样，我们可以在进程之间有一些共享状态。唯一的区别是线程能够访问它们所属进程拥有的相同内存空间，但进程没有这样的奢侈。因此，应该采用其他机制来在多个进程之间共享状态。

在本节中，我们将讨论这些技术，作为本章的一部分，我们将重点关注其中一些作为存储功能的技术。在第一部分，我们将讨论不同的技术，并尝试根据它们的性质对它们进行分组。

## 共享技术

如果你看看你可以在两个进程之间共享状态（一个变量或一个数组）的方法，你会发现这可以通过有限的方式完成。理论上，在多个进程之间共享状态主要有两大类，但在实际的计算机系统中，每一类都有一些子类别。

你要么必须将状态放在一个可以被多个进程访问的“地方”，要么你必须将你的状态*发送*或*传输*为消息、信号或事件给其他进程。同样，你要么必须*拉取*或*检索*一个现有的状态从一个“地方”，要么*接收*它作为消息、信号或事件。第一种方法需要存储或一种*介质*，如内存缓冲区或文件系统，而第二种方法要求你在进程之间有一个消息机制或*通道*。

作为第一种方法的例子，我们可以有一个共享内存区域作为介质，其中包含一个数组，多个进程可以访问并修改这个数组。作为第二种方法的例子，我们可以有一个计算机网络作为通道，允许网络中不同主机上的多个进程之间传输一些消息。

我们目前关于如何在一些进程之间共享状态的讨论实际上并不局限于进程；它也可以应用于线程。线程之间也可以进行信号传递以共享状态或传播事件。

在不同的术语中，第一组中发现的、需要像存储这样的**介质**来共享状态的技巧被称为**基于拉取**的技巧。这是因为想要读取状态的进程必须从存储中拉取它们。

第二组中需要**通道**来传输状态的技巧被称为**基于推送**的技巧。这是因为状态是通过通道推送到接收进程的，它不需要从中拉取。从现在开始，我们将使用这些术语来指代这些技巧。

基于推送的技术多样性导致了现代软件行业中各种分布式架构的出现。与基于推送的技术相比，基于拉取的技术被认为是遗留的，你可以在许多企业应用中看到这一点，在这些应用中，单个中央数据库被用来在整个系统中共享各种状态。

然而，基于推送的方法目前正在兴起，并导致了诸如**事件溯源**和其他一些类似的分布式方法的出现，这些方法用于保持大型软件系统的各个部分之间的一致性，而无需将所有数据存储在中央位置。

在讨论的两种方法中，我们特别关注本章中的第一种方法。我们将在第十九章“单主机 IPC 和套接字”，第二十章“套接字编程”中更多地关注第二种方法。在这些章节中，我们将介绍作为**进程间通信（IPC**）技术一部分的用于在进程之间传输消息的各种通道。只有在这种情况下，我们才能探索各种基于推送的技术，并给出一些观察到的并发问题和可以采用的控制机制的实例。

以下是由 POSIX 标准支持的基于拉取的技术列表，可以在所有 POSIX 兼容的操作系统上广泛使用：

+   **共享内存**：这只是一个在主内存中的共享区域，可以被多个进程访问，它们可以像普通内存块一样使用它来存储变量和数组。共享内存对象不是磁盘上的文件，但它确实是内存。即使没有进程使用它，它也可以作为操作系统中的独立对象存在。当不再需要时，共享内存对象可以被进程移除，或者通过重启系统来移除。因此，从重启生存性的角度来看，共享内存对象可以被视为临时对象。

+   **文件系统**：进程可以使用文件来共享状态。这是一种在软件系统中在多个进程之间共享某些状态的最古老的技术之一。最终，同步访问共享文件的问题，以及许多其他有效的原因，导致了**数据库管理系统**（DBMS）的发明，但仍然，在某些用例中仍在使用共享文件。

+   **网络服务**：一旦对所有进程可用，进程可以使用网络存储或网络服务来存储和检索共享状态。在这种情况下，进程并不确切知道幕后发生了什么。他们只是通过一个定义良好的 API 使用网络服务，该 API 允许他们对共享状态执行某些操作。例如，我们可以提到**网络文件系统**（NFS）或数据库管理系统（DBMS）。它们提供网络服务，允许通过定义良好的模型和一系列伴随操作来维护状态。更具体的例子，我们可以提到*关系型数据库管理系统*，它允许您通过使用 SQL 命令在关系模型中存储您的状态。

在以下小节中，我们将讨论作为 POSIX 接口一部分的上述每种方法。我们首先从 POSIX 共享内存开始，展示它如何导致从*第十六章*，*线程同步*中熟悉的数据竞争。

## POSIX 共享内存

由 POSIX 标准支持，共享内存是广泛用于在多个进程之间共享信息的技术之一。与可以访问相同内存空间的线程不同，进程没有这种能力，操作系统禁止进程访问其他进程的内存。因此，我们需要一种机制来在两个进程之间共享内存的一部分，共享内存正是这种技术。

在以下示例中，我们将详细介绍创建和使用共享内存对象的过程，我们的讨论从创建共享内存区域开始。以下代码展示了如何在 POSIX 兼容系统中创建和填充共享内存对象：

```cpp
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#define SH_SIZE 16
int main(int argc, char** argv) {
  int shm_fd = shm_open("/shm0", O_CREAT | O_RDWR, 0600);
  if (shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n",
        strerror(errno));
    return 1;
  }
  fprintf(stdout, "Shared memory is created with fd: %d\n",
          shm_fd);
  if (ftruncate(shm_fd, SH_SIZE * sizeof(char)) < 0) {
    fprintf(stderr, "ERROR: Truncation failed: %s\n",
            strerror(errno));
    return 1;
  }
  fprintf(stdout, "The memory region is truncated.\n");
  void* map = mmap(0, SH_SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  char* ptr = (char*)map;
  ptr[0] = 'A';
  ptr[1] = 'B';
  ptr[2] = 'C';
  ptr[3] = '\n';
  ptr[4] = '\0';
  while(1);
  fprintf(stdout, "Data is written to the shared memory.\n");
  if (munmap(ptr, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory failed: %s\n",
        strerror(errno));
    return 1;
  }
  return 0;
}
```

代码框 17-4 [ExtremeC_examples_chapter17_4.c]：创建和写入 POSIX 共享内存对象

上述代码创建了一个名为`/shm0`的共享内存对象，其中包含 16 个字节。然后它使用字面量`ABC\n`填充共享内存，最后通过*取消映射*共享内存区域来退出。请注意，即使进程退出，共享内存对象仍然保留。未来的进程可以反复打开和读取相同的共享内存对象。共享内存对象要么通过系统重启来销毁，要么通过进程将其*取消链接*（移除）。

**注意**：

在 FreeBSD 中，共享内存对象的名称应该以`/`开头。在 Linux 或 macOS 中这不是强制性的，但我们为了与 FreeBSD 保持兼容，对它们也做了同样的处理。

在前面的代码中，我们首先使用 `shm_open` 函数打开一个共享内存对象。它接受一个名称和共享内存对象应创建的模式。`O_CREAT` 和 `O_RDWR` 表示应创建共享内存，并且它可以用于读取和写入操作。

注意，如果共享内存对象已经存在，创建操作不会失败。最后一个参数表示共享内存对象的权限。`0600` 表示它仅对启动共享内存对象的拥有者进程的读取和写入操作可用。

在接下来的几行中，我们通过使用 `ftruncate` 函数截断共享内存区域的大小来定义共享内存区域的大小。请注意，如果您即将创建一个新的共享内存对象，这是一个必要的步骤。对于前面的共享内存对象，我们已定义了 16 字节进行分配，然后进行了截断。

随着我们的进行，我们使用 `mmap` 函数将共享内存对象映射到进程可访问的区域。因此，我们有一个指向映射内存的指针，可以用来访问后面的共享内存区域。这也是一个必要的步骤，使得共享内存对我们的 C 程序可访问。

函数 `mmap` 通常用于将文件或共享内存区域（最初从内核的内存空间分配）映射到调用进程可访问的地址空间。然后，映射的地址空间可以使用普通指针作为常规内存区域进行访问。

如您所见，该区域被映射为一个可写区域，由 `PROT_WRITE` 指示，并且作为进程间的共享区域，由 `MAP_SHARED` 参数指示。`MAP_SHARED` 简单地意味着对映射区域的任何更改都将对映射相同区域的其它进程可见。

除了 `MAP_SHARED`，我们还可以使用 `MAP_PRIVATE`；这意味着对映射区域的更改不会传播到其他进程，而是对映射进程是私有的。除非您只想在进程内部使用共享内存，否则这种用法并不常见。

在映射共享内存区域后，前面的代码将一个以空字符终止的字符串 `ABC\n` 写入共享内存。注意字符串末尾的新行换行符。作为最后一步，进程通过调用 `munmap` 函数取消映射共享内存区域，然后关闭分配给共享内存对象的文件描述符。

**注意**：

每个操作系统都提供了一种不同的方式来创建一个未命名的或匿名的共享内存对象。在 FreeBSD 中，只需将 `SHM_ANON` 作为共享内存对象的路径传递给 `shm_open` 函数即可。在 Linux 中，可以使用 `memfd_create` 函数创建一个匿名文件，而不是创建共享内存对象，并使用返回的文件描述符创建一个映射区域。匿名共享内存仅对拥有进程是私有的，不能用于在多个进程之间共享状态。

上述代码可以在 macOS、FreeBSD 和 Linux 系统上编译。在 Linux 系统中，共享内存对象可以在目录`/dev/shm`中看到。请注意，这个目录不是一个常规的文件系统，你看到的东西不是磁盘设备上的文件。相反，`/dev/shm`使用`shmfs`文件系统。它的目的是通过挂载的目录来暴露内存中创建的临时对象，并且它仅在 Linux 中可用。

让我们在 Linux 中编译并运行*示例 17.4*，并检查`/dev/shm`目录的内容。在 Linux 中，必须将最终二进制文件与`rt`库链接，才能使用共享内存功能，这就是为什么你在下面的 shell 框中看到`-lrt`选项的原因：

```cpp
$ ls /dev/shm
$ gcc ExtremeC_examples_chapter17_4.c -lrt -o ex17_4.out
$ ./ex17_4.out
Shared memory is created with fd: 3
The memory region is truncated.
Data is written to the shared memory.
$ ls /dev/shm
shm0
$
```

Shell Box 17-4：构建和运行示例 17.4 并检查共享内存对象是否创建

如你在第一行所见，`/dev/shm`目录中没有共享内存对象。在第二行，我们构建了*示例 17.4*，在第三行，我们执行了生成的可执行文件。然后我们检查`/dev/shm`，我们看到那里有一个新的共享内存对象，`shm0`。

程序的输出也确认了共享内存对象的创建。前一个 shell 框中另一个重要的事情是文件描述符`3`，它被分配给了共享内存对象。

对于你打开的每个文件，每个进程都会打开一个新的文件描述符。这个文件不一定在磁盘上，它可以是共享内存对象、标准输出等等。在每个进程中，文件描述符从 0 开始，直到最大允许的数字。

注意，在每个进程中，文件描述符`0`、`1`和`2`分别预分配给了`stdout`、`stdin`和`stderr`流。在`main`函数运行之前，为每个新进程打开了这些文件描述符。这就是为什么前一个例子中的共享内存对象得到`3`作为其文件描述符的原因。

**注意**：

在 macOS 系统上，你可以使用`pics`实用程序检查系统中的活动 IPC 对象。它可以显示活动的消息队列和共享内存。它还显示了活动的信号量。

`/dev/shm`目录还有一个有趣的属性。你可以使用`cat`实用程序查看共享内存对象的内容，但同样，这仅在 Linux 中可用。让我们在我们的创建的`shm0`对象上使用它。如你在下面的 shell 框中看到的，共享内存对象的内容被显示出来。它是字符串`ABC`加上一个换行符`\n`：

```cpp
$ cat /dev/shm/shm0
ABC
$
```

Shell Box 17-5 使用 cat 程序查看作为示例 17.4 部分创建的共享内存对象的内容

正如我们之前解释的，只要至少有一个进程在使用，共享内存对象就会存在。即使其中一个进程已经请求操作系统删除（或*解除链接*）共享内存，它实际上也不会被删除，直到最后一个进程使用它。即使没有进程解除链接共享内存对象，当系统重启时，它也会被删除。共享内存对象无法在重启后存活，进程应该再次创建它们以用于通信。

以下示例展示了进程如何打开并读取已存在的共享内存对象，以及如何最终解除链接它。*Example 17.5*从*example 17.4*中创建的共享内存对象中读取。因此，它可以被视为与我们在*example 17.4*中做的事情的补充：

```cpp
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#define SH_SIZE 16
int main(int argc, char** argv) {
  int shm_fd = shm_open("/shm0", O_RDONLY, 0600);
  if (shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to open shared memory: %s\n",
        strerror(errno));
    return 1;
  }
  fprintf(stdout, "Shared memory is opened with fd: %d\n", shm_fd);
  void* map = mmap(0, SH_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  char* ptr = (char*)map;
  fprintf(stdout, "The contents of shared memory object: %s\n",
          ptr);
  if (munmap(ptr, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory fd filed: %s\n",
        strerror(errno));
    return 1;
  }
  if (shm_unlink("/shm0") < 0) {
    fprintf(stderr, "ERROR: Unlinking shared memory failed: %s\n",
        strerror(errno));
    return 1;
  }
  return 0;
}
```

代码框 17-5 [ExtremeC_examples_chapter17_5.c]：从作为示例 17.4 部分创建的共享内存对象中读取

作为`main`函数中的第一条语句，我们打开了一个名为`/shm0`的现有共享内存对象。如果没有这样的共享内存对象，我们将生成一个错误。如您所见，我们以只读方式打开了共享内存对象，这意味着我们不会向共享内存中写入任何内容。

在接下来的几行中，我们映射了共享内存区域。同样，我们通过传递`PROT_READ`参数表明映射的区域是只读的。之后，我们最终得到了共享内存区域的指针，并使用它来打印其内容。当我们完成共享内存的使用后，我们取消映射该区域。随后，关闭分配的文件描述符，最后通过使用`shm_unlink`函数解除链接共享内存对象。

在这一点之后，当所有使用相同共享内存的其他进程完成使用后，共享内存对象将从系统中删除。请注意，只要有一个进程在使用，共享内存对象就会存在。

以下是在运行前面代码后的输出。注意在运行*example 17.5*前后`/dev/shm`的内容：

```cpp
$ ls /dev/shm
shm0
$ gcc ExtremeC_examples_chapter17_5.c -lrt -o ex17_5.out
$ ./ex17_5.out
Shared memory is opened with fd: 3
The contents of the shared memory object: ABC
$ ls /dev/shm
$
```

Shell 框 17-6：从示例 17.4 中创建的共享内存对象中读取，并最终删除它

### 使用共享内存的数据竞争示例

现在，是时候演示使用 fork API 和共享内存的组合来产生数据竞争了。这可以与*第十五章*中给出的示例类似，以演示多个线程之间的数据竞争。

在*example 17.6*中，我们有一个放置在共享内存区域内的计数器变量。该示例从主运行进程中派生出一个子进程，它们都尝试增加共享计数器。最终的输出显示了共享计数器上的明显数据竞争：

```cpp
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#define SH_SIZE 4
// Shared file descriptor used to refer to the
// shared memory object
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
void init_shared_resource() {
  // Open the shared memory object
  shared_fd = shm_open("/shm0", O_CREAT | O_RDWR, 0600);
  if (shared_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n",
        strerror(errno));
    exit(1);
  }
  fprintf(stdout, "Shared memory is created with fd: %d\n",
          shared_fd);
}
void shutdown_shared_resource() {
  if (shm_unlink("/shm0") < 0) {
    fprintf(stderr, "ERROR: Unlinking shared memory failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void inc_counter() {
  usleep(1);
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  usleep(1);
}
int main(int argc, char** argv) {
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Allocate and truncate the shared memory region
  if (ftruncate(shared_fd, SH_SIZE * sizeof(char)) < 0) {
    fprintf(stderr, "ERROR: Truncation failed: %s\n",
            strerror(errno));
    return 1;
  }
  fprintf(stdout, "The memory region is truncated.\n");
  // Map the shared memory and initialize the counter
  void* map = mmap(0, SH_SIZE, PROT_WRITE,
          MAP_SHARED, shared_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  counter = (int32_t*)map;
  *counter = 0;
  // Fork a new process
  pid_t pid = fork();
  if (pid) { // The parent process
    // Increment the counter
    inc_counter();
    fprintf(stdout, "The parent process sees the counter as %d.\n",
        *counter);
    // Wait for the child process to exit
    int status = -1;
    wait(&status);
    fprintf(stdout, "The child process finished with status %d.\n",
        status);
  } else { // The child process
    // Incrmenet the counter
    inc_counter();
    fprintf(stdout, "The child process sees the counter as %d.\n",
        *counter);
  }
  // Both processes should unmap shared memory region and close
  // its file descriptor
  if (munmap(counter, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shared_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory fd filed: %s\n",
        strerror(errno));
    return 1;
  }
  // Only parent process needs to shutdown the shared resource
  if (pid) {
    shutdown_shared_resource();
  }
  return 0;
}
```

代码框 17-6 [ExtremeC_examples_chapter17_6.c]：使用 POSIX 共享内存和 fork API 演示数据竞争

在前面的代码中，除了 `main` 函数之外，还有三个函数。函数 `init_shared_resource` 创建共享内存对象。我之所以将此函数命名为 `init_shared_resource` 而不是 `init_shared_memory`，是因为在前面的示例中我们可以使用另一种基于拉取的技术，并且为这个函数取一个通用的名字，使得 `main` 函数在未来示例中保持不变。

函数 `shutdown_shared_resource` 销毁共享内存并解除链接。此外，函数 `inc_counter` 通过 1 增加共享计数器。

`main` 函数截断并映射共享内存区域，就像我们在 *example 17.4* 中所做的那样。在将共享内存区域映射后，分叉逻辑开始。通过调用 `fork` 函数，会创建一个新的进程，并且两个进程（分叉进程和分叉进程）都会尝试通过调用 `inc_counter` 函数来增加计数器。

当父进程向共享计数器写入时，它会等待子进程完成，然后才尝试取消映射、关闭和解除链接共享内存对象。请注意，取消映射和文件描述符的关闭在两个进程中都会发生，但只有父进程会解除链接共享内存对象。

正如您在 *Code Box 17-6* 中所看到的那样，我们在 `inc_counter` 函数中使用了某些不寻常的 `usleep` 调用。原因是强制调度器从某个进程收回 CPU 核心并将其分配给另一个进程。如果没有这些 `usleep` 函数调用，CPU 核心通常不会在进程之间转移，而且很难经常看到不同交织的效果。

导致这种效果的一个原因是每个进程中的指令数量较少。如果每个进程的指令数量显著增加，即使没有睡眠调用，也可以看到交织的非确定性行为。例如，在每个进程中有一个循环，计数 10,000 次，并在每次迭代中增加共享计数器，这很可能揭示数据竞争。您可以自己尝试一下。

关于前面代码的最后一个注意事项，父进程在创建和打开共享内存对象并将其分配给文件描述符之前，会进行子进程的创建和分叉。分叉后的进程不会打开共享内存对象，但它可以使用相同的文件描述符。所有打开的文件描述符都是从父进程继承的事实，帮助子进程继续使用文件描述符，并引用相同的共享内存对象。

在 *Shell Box 17-7* 中的以下内容是多次运行 *example 17.6* 的输出。正如您所看到的，我们在共享计数器上存在明显的数据竞争。有时父进程或子进程在未获取最新修改值的情况下更新计数器，这导致两个进程都打印出 `1`：

```cpp
$ gcc ExtremeC_examples_chapter17_6 -o ex17_6.out
$ ./ex17_6.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 2.
The child process finished with status 0.
$ ./ex17_6
...
...
...
$ ./ex17_6.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 1.
The child process finished with status 0.
$
```

Shell Box 17-7：运行示例 17.6 并演示在共享计数器上发生的数据竞争

在本节中，我们展示了如何创建和使用共享内存。我们还演示了一个数据竞争的例子以及并发进程在访问共享内存区域时的行为。在下一节中，我们将讨论文件系统作为另一种广泛使用的基于拉的共享状态方法，在多个进程之间共享状态。

## 文件系统

POSIX 提供了一个类似的 API 来处理文件系统中的文件。只要涉及到文件描述符，并且它们被用来引用各种系统对象，就可以使用与用于共享内存相同的 API。

我们使用文件描述符来引用文件系统中的实际文件，如 **ext4**，以及共享内存、管道等；因此，可以采用相同的语义来打开、读取、写入，将它们映射到本地内存区域等。因此，我们预计会看到与共享内存类似的讨论，也许还有类似的 C 代码。这在 *示例 17.7* 中可以看到。

**注意**：

我们通常映射文件描述符。然而，也有一些特殊情况，其中 *套接字描述符* 可以被映射。套接字描述符类似于文件描述符，但用于网络或 Unix 套接字。这个链接提供了一个有趣的映射 TCP 套接字背后的内核缓冲区的用例，这被称为 *零拷贝接收机制*：https://lwn.net/Articles/752188/。

注意，用于使用文件系统的 API 与我们用于共享内存的 API 非常相似，但这并不意味着它们的实现也相似。事实上，由硬盘支持的文件系统中的文件对象与共享内存对象在本质上是有区别的。让我们简要讨论一些区别：

+   共享内存对象基本上位于内核进程的内存空间中，而文件系统中的文件位于磁盘上。这样的文件最多只有一些用于读写操作的分配缓冲区。

+   写入共享内存的状态在系统重启后会被清除，但写入共享文件的状态，如果它由硬盘或永久存储支持，重启后可以保留。

+   通常情况下，访问共享内存比访问文件系统要快得多。

以下代码是我们在上一节中为共享内存给出的相同数据竞争示例。由于文件系统的 API 与我们用于共享内存的 API 非常相似，我们只需要从 *example 17.6* 中更改两个函数；`init_shared_resource` 和 `shutdown_shared_resource`。其余的将保持不变。这是通过使用相同的 POSIX API 在文件描述符上操作而取得的一项伟大成就。让我们来看看代码：

```cpp
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#define SH_SIZE 4
// The shared file descriptor used to refer to the shared file
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
void init_shared_resource() {
  // Open the file
  shared_fd = open("data.bin", O_CREAT | O_RDWR, 0600);
  if (shared_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create the file: %s\n",
        strerror(errno));
    exit(1);
  }
  fprintf(stdout, "File is created and opened with fd: %d\n",
          shared_fd);
}
void shutdown_shared_resource() {
  if (remove("data.bin") < 0) {
    fprintf(stderr, "ERROR: Removing the file failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void inc_counter() {
  ... As exmaple 17.6 ...
}
int main(int argc, char** argv) {
  ... As exmaple 17.6 ...
}
```

Code Box 17-7 [ExtremeC_examples_chapter17_7.c]：使用常规文件和 fork API 演示数据竞争

如你所见，前面的大部分代码来自 *示例 17.6*。其余的是使用 `open` 和 `remove` 函数代替 `shm_open` 和 `shm_unlink` 函数的替代方案。

注意，文件 `data.bin` 是在当前目录中创建的，因为我们没有给 `open` 函数提供一个绝对路径。运行前面的代码也会产生相同的共享计数器数据竞争。它可以像我们对 *示例 17.6* 的方法一样进行检查。

到目前为止，我们已经看到我们可以使用共享内存和共享文件来存储状态，并从多个进程中并发地访问它。现在，是时候更深入地讨论多线程和多进程，并彻底比较它们了。

# 多线程与多进程

在 *第十四章* 中讨论了多线程和多进程，以及我们在最近几章中介绍的概念之后，我们现在处于一个很好的位置来比较它们，并给出一个高级描述，说明在哪些情况下应该采用每种方法。假设我们正在设计一个旨在并发处理多个输入请求的软件。我们将在三种不同的情况下讨论这个问题。让我们从第一种情况开始。

## 多线程

第一种情况是当你可以编写一个只有单个进程的软件时，所有请求都进入同一个进程。所有逻辑都应该作为同一进程的一部分来编写，结果你得到一个庞大的进程，它执行系统中的一切。由于这是单进程软件，如果你想并发处理许多请求，你需要通过创建线程来以多线程的方式处理多个请求。此外，选择一个具有有限线程数的 *线程池* 可能是一个更好的设计决策。

在并发和同步方面有以下考虑事项需要注意。请注意，我们在这里不讨论使用事件循环或异步 I/O，尽管它仍然可以是多线程的有效替代方案。

如果请求的数量显著增加，线程池中有限的线程数量应该增加以克服需求。这实际上意味着升级运行主进程的机器的硬件和资源。这被称为 *向上扩展* 或 *垂直扩展*。这意味着你升级单台机器上的硬件，以便能够响应更多的请求。除了客户在升级到新硬件期间可能经历的可能的停机时间（尽管可以防止这种情况发生）之外，升级是昂贵的，而且当请求的数量再次增长时，你必须进行另一次扩展。

如果处理请求最终涉及到操作共享状态或数据存储，可以通过知道线程可以访问相同的内存空间这一事实，轻松地实现同步技术。当然，无论它们是否有一个需要维护的共享数据结构，或者它们是否有访问非事务性的远程数据存储，这都是必要的。

所有线程都在同一台机器上运行，因此它们可以使用我们之前解释的用于共享状态的相同技术，这些技术由线程和进程使用。这是一个很棒的功能，并且在处理线程同步时减轻了很多痛苦。

让我们谈谈下一个情况，当我们可以有一个以上的进程，但它们都在同一台机器上。

## 单主机多进程

在这种情况下，我们编写了一个具有多个进程的软件，但所有这些进程都部署在单个机器上。所有这些进程可以是单线程的，或者它们可以在内部有一个线程池，允许每个进程一次处理多个请求。

当请求的数量增加时，可以创建新的进程而不是创建更多的线程。这通常被称为*横向扩展*或*水平扩展*。然而，当你只有一台单机时，你必须向上扩展，换句话说，你必须升级其硬件。这可能会引起我们在前一个子节中提到的多线程程序向上扩展时提到的问题。

当涉及到并发时，进程是在并发环境中执行的。它们只能使用多进程方式共享状态或同步进程。当然，这并不像编写多线程代码那样方便。此外，进程可以使用基于拉或基于推的技术来共享状态。

在单台机器上实现多进程并不十分有效，而且当涉及到编码的劳动强度时，似乎多线程更为方便。

下一个子节将讨论分布式多进程环境，这是创建现代软件的最佳设计。

## 分布式多进程

在最终的情况下，我们编写了一个程序，作为多个进程运行，这些进程运行在多个主机上，所有主机都通过网络相互连接，并且在单个主机上可以运行多个进程。在这种部署中可以看到以下特点。

当面临请求数量的显著增长时，这个系统可以无限扩展。这是一个很棒的功能，使你能够在面对如此高的峰值时使用通用硬件。使用通用硬件的集群而不是强大的服务器是谷歌能够在机器集群上运行其*PageRank*和*Map* *Reduce*算法的其中一个想法。

本章讨论的技术几乎无助于解决问题，因为它们有一个重要的先决条件：即所有进程都在同一台机器上运行。因此，应该采用一组完全不同的算法和技术来使进程同步，并使共享状态对系统中的所有进程可用。应该研究和调整诸如*延迟*、*容错性*、*可用性*、*数据一致性*等多个因素，以适应这样的分布式系统。

不同主机上的进程使用网络套接字以基于推送的方式通信，但同一主机上的进程可能使用本地 IPC 技术，例如消息队列、共享内存、管道等，以传输消息和共享状态。

在本节的最后，在现代软件行业中，我们更倾向于横向扩展而不是纵向扩展。这将引发许多关于数据存储、同步、消息传递等方面的新想法和技术。它甚至可能对硬件设计产生影响，使其适合横向扩展。

# 摘要

在本章中，我们探讨了多进程系统以及可以用于在多个进程之间共享状态的各种技术。本章涵盖了以下主题：

+   我们介绍了用于进程执行的 POSIX API。我们解释了`fork` API 和`exec*`函数的工作原理。

+   我们解释了内核执行进程所采取的步骤。

+   我们讨论了状态如何在多个进程之间共享的方法。

+   我们介绍了基于拉取和基于推送的技术作为所有其他可用技术的两个顶级类别。

+   文件系统上的共享内存和共享文件是常见的基于拉取方式共享状态的技术。

+   我们解释了多线程和多进程部署之间的差异和相似之处，以及分布式软件系统中的垂直和水平扩展的概念。

在下一章中，我们将讨论单主机多进程环境中的并发问题。这包括对并发问题的讨论以及同步多个进程以保护共享资源的方法。这些主题与你在*第十六章*，*线程同步*中遇到的主题非常相似，但它们的焦点在于进程而不是线程。
