# 第十一章

# 系统调用和内核

在上一章中，我们讨论了 Unix 的历史及其洋葱式架构。我们还介绍了 POSIX 和 SUS 标准，这些标准规范了 Unix 中 shell 环的运作，然后在解释 C 标准库如何提供 Unix 兼容系统暴露的常用功能之前。

在本章中，我们将继续讨论 *系统调用接口* 和 Unix *内核*。这将让我们对 Unix 系统的工作方式有一个完整的了解。

在阅读本章之后，你将能够分析程序调用的系统调用，你将能够解释进程如何在 Unix 环境中生存和演变，你还将能够直接或通过 libc 使用系统调用。我们还将讨论 Unix 内核开发，并展示你如何向 Linux 内核添加新的系统调用以及如何从 shell 环中调用它。

在本章的最后部分，我们将讨论 *单一内核* 和 *微内核* 以及它们之间的区别。我们将介绍 Linux 内核作为一个单一内核，并为其编写一个可以动态加载和卸载的 *内核模块*。

让我们以讨论系统调用开始本章。

# 系统调用

在上一章中，我们简要解释了什么是系统调用。在本节中，我们想要更深入地探讨并解释系统调用背后的机制，即从用户进程到内核进程的执行转移机制。

然而，在我们这样做之前，我们需要对内核空间和用户空间进行更多的解释，因为这将有助于我们理解系统调用在幕后是如何工作的。我们还将编写一个简单的系统调用来获得一些关于内核开发的思路。

我们即将要做的事情对于你想要在向内核添加之前不存在的新功能时编写新的系统调用至关重要。这也让你更好地理解内核空间以及它与用户空间的不同，因为实际上，它们是非常不同的。

## 系统调用的显微镜下

正如我们在上一章中讨论的，当从 shell 环移动到内核环时，会发生分离。你会发现位于前两个环中的任何内容，即用户应用程序和 shell，都属于用户空间。同样，出现在内核环或硬件环中的任何内容都属于内核空间。

关于这种分离有一条规则，那就是在两个最内层的环——内核和硬件——中的任何内容都不能被用户空间直接访问。换句话说，用户空间中的任何进程都不能直接访问硬件、内部内核数据结构和算法。相反，它们应该通过系统调用进行访问。

话虽如此，你可能认为这与你对类 Unix 操作系统（如 Linux）所知和所经历的东西似乎有些矛盾。如果你看不到问题，让我为你解释一下。这似乎是一种矛盾，因为例如，当程序从网络套接字读取一些字节时，实际上读取这些字节的不是程序，而是内核读取字节并将它们复制到用户空间，然后程序可以取回并使用它们。

我们可以通过一个例子来明确这一点，即从用户空间到内核空间以及相反方向的所有步骤。当你想从硬盘驱动器读取文件时，你会在用户应用程序环中编写一个程序。你的程序使用一个名为`fread`的 libc I/O 函数（或另一个类似函数），最终作为用户空间中的进程运行。当程序调用`fread`函数时，libc 背后的实现被触发。

到目前为止，一切仍然在用户进程中。然后，`fread`实现最终调用一个系统调用，而`fread`接收一个已经打开的*文件描述符*作为第一个参数，作为第二个参数，是分配在进程内存中的缓冲区的地址，该缓冲区位于用户空间，作为第三个参数，是缓冲区的长度。

当系统调用由 libc 实现触发时，内核代表用户进程控制执行。它从用户空间接收参数并将它们保存在内核空间中。然后，内核通过访问内核内部的文件系统单元来读取文件（如前一章中*图 10-5*所示）。

当内核环中的`read`操作完成时，读取的数据将被复制到由调用`fread`函数时指定的用户空间中的缓冲区，系统调用随后离开并将执行控制权返回给用户进程。同时，用户进程通常会在系统调用忙于操作时等待。在这种情况下，系统调用是阻塞的。

关于这种情况，有一些重要的事情需要注意：

+   我们只有一个内核执行系统调用背后的所有逻辑。

+   如果系统调用是*阻塞的*，当系统调用正在进行时，调用者用户进程必须等待，直到系统调用繁忙并完成。相反，如果系统调用是*非阻塞的*，系统调用会非常快地返回，但用户进程必须进行额外的系统调用以检查结果是否可用。

+   参数以及输入和输出数据将从用户空间复制到/从用户空间。由于实际值被复制，系统调用应该设计成接受小的变量和指针作为输入参数。

+   内核可以完全访问系统的所有资源。因此，应该有一个机制来检查用户进程是否能够执行这样的系统调用。在这种情况下，如果用户不是文件的所有者，`fread` 应该因为缺少所需权限而失败。

+   用户空间和内核空间之间也存在类似的分离。用户进程只能访问用户空间内存。为了完成某个系统调用，可能需要多次传输。

在我们进入下一节之前，我想问你一个问题。系统调用是如何将执行控制权传递给内核的？花一分钟时间思考一下，因为在下一节中，我们将努力找到这个问题的答案。

## 越过标准 C – 直接调用系统调用

在回答提出的问题之前，让我们通过一个绕过标准 C 库并直接调用系统调用的示例。换句话说，程序调用系统调用而不通过 shell 环。正如我们之前所提到的，这被认为是一种反模式，但当某些系统调用没有通过 libc 暴露时，用户应用程序可以直接调用系统调用。

在每个 Unix 系统中，都有一个特定的方法可以直接调用系统调用。例如，在 Linux 中，有一个名为 `syscall` 的函数，位于 `<sys/syscall.h>` 头文件中，可以用于此目的。

以下代码框，*示例 11.1*，是一个不同的 Hello World 示例，它不使用 libc 将内容打印到标准输出。换句话说，该示例不使用作为 shell 环和 POSIX 标准一部分的 `printf` 函数。相反，它直接调用特定的系统调用，因此代码只能在 Linux 机器上编译，不能在其他 Unix 系统上编译。换句话说，代码在各种 Unix 版本之间不可移植：

```cpp
// We need to have this to be able to use non-POSIX stuff
#define _GNU_SOURCE
#include <unistd.h>
// This is not part of POSIX!
#include <sys/syscall.h>
int main(int argc, char** argv) {
  char message[20] = "Hello World!\n";
  // Invokes the 'write' system call that writes
  // some bytes into the standard output.
  syscall(__NR_write, 1, message, 13);
  return 0;
}
```

代码框 11-1 [ExtremeC_examples_chapter11_1.c]：一个不同的 Hello World 示例，它直接调用 write 系统调用

作为前面代码框中的第一个语句，我们必须定义 `_GNU_SOURCE` 以指示我们将使用不属于 POSIX 或 SUS 标准的 **GNU C 库**（**glibc**）的部分。这会破坏程序的可移植性，因此，你可能无法在其他 Unix 机器上编译你的代码。在第二个 `include` 语句中，我们包含了一个 glibc 特定的头文件，该文件在其他使用 glibc 作为主要 libc 核心的 POSIX 系统中不存在。

在 `main` 函数中，我们通过调用 `syscall` 函数来执行系统调用。首先，我们必须通过传递一个数字来指定系统调用。这是一个整数，它指向一个特定的系统调用。每个系统调用在 Linux 中都有一个独特的特定 *系统调用号*。

在示例代码中，`__R_write` 常量被传递而不是系统调用号，我们不知道它的确切数值。在 `unistd.h` 头文件中查找后，显然 64 是 `write` 系统调用的编号。

在传递系统调用号之后，我们应该传递系统调用所需的参数。

注意，尽管前面的代码非常简单，它只包含一个简单的函数调用，但你应该知道 `syscall` 不是一个普通函数。它是一个汇编过程，它填充了一些适当的 CPU 寄存器，并且实际上将执行控制从用户空间转移到内核空间。我们很快就会讨论这一点。

对于 `write`，我们需要传递三个参数：文件描述符，在这里是 `1`，表示标准输出；第二个是用户空间中分配的 *缓冲区指针*；最后是 *应该从缓冲区复制的字节数*。

下面的输出是 *示例 11.1* 的输出，使用 `gcc` 在 Ubuntu 18.04.1 上编译和运行：

```cpp
$ gcc ExtremeC_examples_chapter11_1.c -o ex11_1.out
$ ./ex11_1.out
Hello World!
$
```

Shell Box 11-1：示例 11.1 的输出

现在是时候使用上一章中介绍过的 `strace` 来查看 *示例 11.1* 实际调用的系统调用。以下 `strace` 的输出显示了程序已经调用了所需的系统调用：

```cpp
$ strace ./ex11_1.out
execve("./ex11_1.out", ["./ex11_1.out"], 0x7ffcb94306b0 /* 22 vars */) = 0
brk(NULL)                               = 0x55ebc30fb000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
...
...
arch_prctl(ARCH_SET_FS, 0x7f24aa5624c0) = 0
mprotect(0x7f24aa339000, 16384, PROT_READ) = 0
mprotect(0x55ebc1e04000, 4096, PROT_READ) = 0
mprotect(0x7f24aa56a000, 4096, PROT_READ) = 0
munmap(0x7f24aa563000, 26144)           = 0
write(1, "Hello World!\n", 13Hello World!
)          = 13
exit_group(0)                           = ?
+++ exited with 0 +++
$
```

Shell Box 11-2：运行示例 11.1 时 strace 的输出

正如你在 *Shell Box 11-2* 中的粗体中看到的，系统调用已被 `strace` 记录。看看返回值，它是 `13`。这意味着系统调用已成功将 13 个字节写入给定的文件，在这种情况下是标准输出。

**注意**：

用户应用程序永远不应该尝试直接使用系统调用。在调用系统调用之前和之后通常需要采取一些步骤。Libc 实现这些步骤。当你不打算使用 libc 时，你必须自己执行这些步骤，你必须知道这些步骤在不同的 Unix 系统之间是不同的。

## 在 syscall 函数内部

然而，`syscall` 函数内部发生了什么？请注意，当前的讨论仅适用于 glibc，而不适用于其他 libc 实现。首先，我们需要在 glibc 中找到 `syscall`。这里是 `syscall` [定义的链接：https://github.com/lattera/glibc/blob/master/sysdeps/unix/sysv/linux/x86](https://github.com/lattera/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/syscall.S)_64/syscall.S。

如果你在一个浏览器中打开前面的链接，你会看到这个函数是用汇编语言编写的。

**注意**：

汇编语言可以与 C 语句一起在 C 源文件中使用。事实上，这是 C 的一个重要特性，使其适合编写操作系统。对于 `syscall` 函数，我们有一个用 C 编写的声明，但定义是在汇编中。

这里是作为 `syscall.S` 部分找到的源代码：

```cpp
/* Copyright (C) 2001-2018 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
...
   <http://www.gnu.org/licenses/>.  */
#include <sysdep.h>
/* Please consult the file sysdeps/unix/sysv/linux/x86-64/sysdep.h for
   more information about the value -4095 used below.  */
/* Usage: long syscall (syscall_number, arg1, arg2, arg3, arg4, arg5, arg6)
   We need to do some arg shifting, the syscall_number will be in
   rax.  */
	.text
ENTRY (syscall)
    movq %rdi, %rax            /* Syscall number -> rax.  */
    movq %rsi, %rdi            /* shift arg1 - arg5\.  */
    movq %rdx, %rsi
    movq %rcx, %rdx
    movq %r8, %r10
    movq %r9, %r8
    movq 8(%rsp),%r9           /* arg6 is on the stack.  */
    syscall                    /* Do the system call.  */
    cmpq $-4095, %rax          /* Check %rax for error.  */
    jae SYSCALL_ERROR_LABEL    /* Jump to error handler if error.  */
    ret                        /* Return to caller.  */
PSEUDO_END (syscall)
```

Code Box 11-2：glibc 中 syscall 函数的定义

尽管以这种方式进行系统调用似乎更复杂，但这些指令简短且简单。使用注释解释说，在 glibc 中，每次调用可以提供多达六个参数的系统调用。

这意味着如果底层内核支持具有超过六个参数的系统调用，glibc 无法提供某些内核功能，并且应该修改以支持它们。幸运的是，在大多数情况下，六个参数已经足够了，对于需要超过六个参数的系统调用，我们可以传递在用户空间内存中分配的结构变量的指针。

在前面的代码框中，在`movq`指令之后，汇编代码调用`syscall`子程序。它只是生成一个*中断*，这允许内核中等待此类中断的特定部分唤醒并处理中断。

如您在`syscall`过程的第 一行所看到的，系统调用号被移动到`%rax`寄存器。在接下来的几行中，我们将其他参数复制到不同的寄存器中。当系统调用中断被触发时，内核的中断处理单元接收到调用并收集系统调用号和参数。然后它搜索其*系统调用表*以找到应在内核端调用的适当函数。

一个有趣的观点是，当中断处理程序在 CPU 中执行时，已经离开 CPU 的发起系统调用的用户代码，内核正在执行这项工作。这是系统调用背后的主要机制。当你发起一个系统调用时，CPU 会改变其模式，内核指令被加载到 CPU 中，用户空间应用程序不再被执行。这就是我们说内核代表用户应用程序执行系统调用逻辑背后的逻辑的基本原因。

在下一节中，我们将通过编写一个打印 hello 消息的系统调用来给出一个例子。它可以被认为是*示例 11.1*的渐进版本，它接受一个输入字符串并返回一个问候字符串。

## 向 Linux 添加系统调用

在本节中，我们将向现有类 Unix 内核的系统调用表中添加一个新的系统调用。这可能是有很多读者阅读这本书时第一次编写的应该在内核空间运行的 C 代码。我们之前章节中编写的所有示例，以及我们将在未来章节中编写的几乎所有代码，都是在用户空间运行的。

事实上，我们编写的绝大多数程序都是打算在用户空间运行的。事实上，这就是我们所说的*C 编程*或*C 开发*。然而，如果我们打算编写一个应该在内核空间运行的 C 程序，我们使用一个不同的名称；我们称之为*内核开发*。

我们正在分析下一个示例，*示例 11.2*，但在那之前，我们需要探索内核环境，看看它与用户空间有何不同。

### 内核开发

本节对那些希望成为内核开发者或操作系统领域的安全研究员的你们来说将是有益的。在第一部分，在跳转到系统调用本身之前，我们想要解释内核开发与普通 C 开发之间的差异。

内核的开发与普通 C 程序的开发在许多方面都不同。在探讨这些差异之前，我们应该注意的一点是，C 开发通常发生在用户空间。

在以下列表中，我们提供了内核和用户空间开发过程中六个关键差异：

+   只有一个内核进程在运行一切。这仅仅意味着如果你的代码在内核中导致崩溃，你可能需要重新启动机器并让内核重新初始化。因此，与内核进程相关，开发成本非常高，你不能在不重启机器的情况下尝试各种解决方案，而你可以在处理用户空间程序时轻松地这样做。在内核崩溃时，会生成一个*内核崩溃转储*，可以用来诊断原因。

+   在内核环中没有像 glibc 这样的 C 标准库！换句话说，这是一个 SUS 和 POSIX 标准不再有效的领域。因此，你不能包含任何 libc 头文件，例如 `stdio.h` 或 `string.h`。在这种情况下，你有一组专门用于各种操作的函数。这些函数通常位于 *内核头文件* 中，并且可能因 Unix 版本的不同而不同，因为在这个领域没有标准化。

    例如，如果你在 Linux 上进行内核开发，你可能使用 `printk` 将消息写入内核的 *消息缓冲区*。然而，在 FreeBSD 中，你需要使用 `printf` 函数族，这些函数与 libc 的 `printf` 函数不同。你可以在 FreeBSD 系统的 `<sys/system.h>` 头文件中找到这些 `printf` 函数。在 XNU 内核开发中对应的函数是 `os_log`。请注意，XNU 是 macOS 的内核。

+   你可以在内核中读取或修改文件，但不能使用 libc 函数。每个 Unix 内核都有自己的方法来访问内核环内的文件。这对于通过 libc 暴露的所有功能都是相同的。

+   你可以完全访问内核环中的物理内存和许多其他服务。因此，编写安全可靠代码非常重要。

+   内核中没有系统调用机制。系统调用是用户空间中使用户进程能够与内核环通信的主要机制。因此，一旦你处于内核中，就不再需要它。

+   内核进程是通过将内核镜像复制到物理内存中创建的，由 *引导加载程序* 执行。您不能在不从头创建内核镜像并重新引导系统重新加载的情况下添加新的系统调用。在支持 *内核模块* 的内核中，您可以在内核运行时轻松添加或删除模块，但您不能对系统调用做同样的事情。

如您所看到的，与普通的 C 开发相比，内核开发发生在不同的流程中。测试编写的逻辑不是一件容易的事情，有缺陷的代码可能导致系统崩溃。

在下一节中，我们将通过添加一个新的系统调用来进行我们的第一次内核开发。我们这样做并不是因为当你想在内核中引入新的功能时，添加系统调用是常见的，但我们是想通过尝试来熟悉内核开发。

### 为 Linux 编写一个 Hello World 系统调用

在本节中，我们将为 Linux 编写一个新的系统调用。互联网上有许多优秀的资源解释了如何向现有的 Linux 内核添加系统调用，但以下论坛帖子，*将 Hello World 系统调用添加到 Linux 内核* – 可在 https://medium.com/anubhav-shrimal/adding-a-hello-world-system-call-to-linux-kernel-dad32875872 找到 – 被用作构建我在 Linux 中自己的系统调用的基础。

*示例 11.2* 是 *示例 11.1* 的一个高级版本，它使用了一个不同且定制的系统调用，我们将在本节中编写。新的系统调用接收四个参数。前两个参数用于输入名称，后两个参数用于输出问候字符串。我们的系统调用通过其前两个参数接受一个名称，一个指向用户空间中已分配缓冲区的 `char` 指针和一个表示缓冲区长度的整数，并使用其第二个两个参数返回问候字符串，一个不同于输入缓冲区的指针，并且再次在用户空间中分配，以及一个表示其长度的整数。

**警告**：

请不要在打算用于工作或家庭用途的 Linux 安装上执行此实验。请在实验机器上运行以下命令，强烈建议使用虚拟机。您可以通过使用仿真应用程序（如 VirtualBox 或 VMware）轻松创建虚拟机。

如果不恰当地或以错误的顺序使用以下说明，它们可能会损坏您的系统，并导致您丢失部分甚至全部数据。如果您打算在非实验机器上运行以下命令，请始终考虑一些备份解决方案，以复制您的数据。

首先，我们需要下载 Linux 内核的最新源代码。我们将使用 Linux GitHub 仓库来克隆其源代码，然后我们将选择一个特定的发布版本。版本 5.3 于 2019 年 9 月 15 日发布，因此我们将使用这个版本进行本示例。

**注意**：

Linux 是一个内核。这意味着它只能安装在类 Unix 操作系统的内核环中，但 *Linux 发行版* 是另一回事。Linux 发行版在其内核环中有一个特定的 Linux 内核版本，在其 shell 环中有一个特定的 GNU libc 和 Bash（或 GNU shell）版本。

每个 Linux 发行版通常都附带其外部环中完整的用户应用程序列表。因此，我们可以说 Linux 发行版是一个完整的操作系统。请注意，*Linux 发行版*、*Linux distro* 和 *Linux flavor* 都指的是同一件事。

在这个示例中，我正在使用 64 位机器上的 Ubuntu 18.04.1 Linux 发行版。

在我们开始之前，确保通过运行以下命令安装了先决条件软件包是非常重要的：

```cpp
$ sudo apt-get update
$ sudo apt-get install -y build-essential autoconf libncurses5-dev libssl-dev bison flex libelf-dev git
...
...
$
```

Shell Box 11-3：安装示例 11.2 所需的先决条件软件包

关于前面指令的一些说明：`apt` 是基于 Debian 的 Linux 发行版中的主要软件包管理器，而 `sudo` 是一个我们用来以 *超级用户* 模式运行命令的实用程序。它在几乎每个类 Unix 操作系统中都可用。

下一步是克隆 Linux GitHub 仓库。在克隆仓库之后，我们还需要检出版本 5.3。可以通过使用发布标签名称来检出版本，如下面的命令所示：

```cpp
$ git clone https://github.com/torvalds/linux
$ cd linux
$ git checkout v5.3
$
```

Shell Box 11-4：克隆 Linux 内核并检出版本 5.3

现在，如果你查看根目录中的文件，你会看到很多文件和目录，它们组合起来构成了 Linux 内核代码库：

```cpp
$ ls
total 760K
drwxrwxr-x  33 kamran kamran 4.0K Jan 28  2018 arch
drwxrwxr-x   3 kamran kamran 4.0K Oct 16 22:11 block
drwxrwxr-x   2 kamran kamran 4.0K Oct 16 22:11 certs
...
drwxrwxr-x 125 kamran kamran  12K Oct 16 22:11 Documentation
drwxrwxr-x 132 kamran kamran 4.0K Oct 16 22:11 drivers
-rw-rw-r--   1 kamran kamran 3.4K Oct 16 22:11 dropped.txt
drwxrwxr-x   2 kamran kamran 4.0K Jan 28  2018 firmare
drwxrwxr-x  75 kamraln kamran 4.0K Oct 16 22:11 fs
drwxrwxr-x  27 kamran kamran 4.0K Jan 28  2018 include
...
-rw-rw-r--   1 kamran kamran  287 Jan 28  2018 Kconfig
drwxrwxr-x  17 kamran kamran 4.0K Oct 16 22:11 kernel
drwxrwxr-x  13 kamran kamran  12K Oct 16 22:11 lib
-rw-rw-r--   1 kamran kamran 429K Oct 16 22:11 MAINTAINERS
-rw-rw-r--   1 kamran kamran  61K Oct 16 22:11 Makefile
drwxrwxr-x   3 kamran kamran 4.0K Oct 16 22:11 mm
drwxrwxr-x  69 kamran kamran 4.0K Jan 28  2018 net
-rw-rw-r--   1 kamran kamran  722 Jan 28  2018 README
drwxrwxr-x  28 kamran kamran 4.0K Jan 28  2018 samples
drwxrwxr-x  14 kamran kamran 4.0K Oct 16 22:11 scripts
...
drwxrwxr-x   4 kamran kamran 4.0K Jan 28  2018 virt
drwxrwxr-x   5 kamran kamran 4.0K Oct 16 22:11 zfs
$
```

Shell Box 11-5：Linux 内核代码库的内容

如您所见，有一些目录可能看起来很熟悉：`fs`、`mm`、`net`、`arch` 等。我应该指出，我们不会对每个这些目录的详细信息进行更多说明，因为它们可以从一个内核到另一个内核有很大的不同，但一个共同的特点是所有内核几乎都遵循相同的内部结构。

现在我们已经有了内核源代码，我们应该开始添加我们新的 Hello World 系统调用。然而，在我们这样做之前，我们需要为我们的系统调用选择一个唯一的数值标识符；在这种情况下，我给它命名为 `hello_world`，并选择 `999` 作为它的编号。

首先，我们需要将系统调用函数声明添加到 `include/linux/syscalls.h` 头文件末尾。经过这次修改后，文件应该看起来像这样：

```cpp
/*
 * syscalls.h - Linux syscall interfaces (non-arch-specific)
 *
 * Copyright (c) 2004 Randy Dunlap
 * Copyright (c) 2004 Open Source Development Labs
 *
 * This file is released under the GPLv2.
 * See the file COPYING for more details.
 */
#ifndef _LINUX_SYSCALLS_H
#define _LINUX_SYSCALLS_H
struct epoll_event;
struct iattr;
struct inode;
...
asmlinkage long sys_statx(int dfd, const char __user *path, unsigned flags,
                          unsigned mask, struct statx __user *buffer);
asmlinkage long sys_hello_world(const char __user *str,
 const size_t str_len,
 char __user *buf,
 size_t buf_len);
#endif
```

Code Box 11-3 [include/linux/syscalls.h]：新的 Hello World 系统调用的声明

顶部的描述说明这是一个包含 Linux `syscall` 接口的头文件，这些接口不是 *架构特定的*。这意味着在所有架构上，Linux 都暴露了相同的一组系统调用。

在文件末尾，我们声明了我们的系统调用函数，它接受四个参数。正如我们之前解释的，前两个参数是输入字符串及其长度，后两个参数是输出字符串及其长度。

注意，输入参数是 `const`，但输出参数不是。此外，`__user` 标识符表示指针指向用户空间内的内存地址。正如你所见，每个系统调用都有整数返回值作为其函数签名的一部分，这实际上是它的执行结果。返回值的范围及其含义因系统调用而异。在我们的系统调用中，`0` 表示成功，任何其他数字都表示失败。

现在，我们需要定义我们的系统调用。为此，我们必须首先在根目录下创建一个名为 `hello_world` 的文件夹，我们使用以下命令来完成：

```cpp
$ mkdir hello_world
$ cd hello_world
$
```

Shell 框 11-6：创建 hello_world 目录

接下来，我们在 `hello_world` 目录内创建一个名为 `sys_hello_world.c` 的文件。该文件的 内容应如下所示：

```cpp
#include <linux/kernel.h>   // For printk
#include <linux/string.h>   // For strcpy, strcat, strlen
#include <linux/slab.h>     // For kmalloc, kfree
#include <linux/uaccess.h>  // For copy_from_user, copy_to_user
#include <linux/syscalls.h> // For SYSCALL_DEFINE4
// Definition of the system call
SYSCALL_DEFINE4(hello_world,
          const char __user *, str,    // Input name
          const unsigned int, str_len, // Length of input name
          char __user *, buf,          // Output buffer
          unsigned int, buf_len) {     // Length of output buffer
  // The kernel stack variable supposed to keep the content
  // of the input buffer
  char name[64];
  // The kernel stack variable supposed to keep the final
  // output message.
  char message[96];
  printk("System call fired!\n");
  if (str_len >= 64) {
    printk("Too long input string.\n");
    return -1;
  }
  // Copy data from user space into kernel space
  if (copy_from_user(name, str, str_len)) {
    printk("Copy from user space failed.\n");
    return -2;
  }
  // Build up the final message
  strcpy(message, "Hello ");
  strcat(message, name);
  strcat(message, "!");
  // Check if the final message can be fit into the output binary
  if (strlen(message) >= (buf_len - 1)) {
    printk("Too small output buffer.\n");
    return -3;
  }
  // Copy back the message from the kernel space to the user space
  if (copy_to_user(buf, message, strlen(message) + 1)) {
    printk("Copy to user space failed.\n");
    return -4;
  }
  // Print the sent message into the kernel log
  printk("Message: %s\n", message);
  return 0;
}
```

代码框 11-4：Hello World 系统调用的定义

在 *代码框 11-4* 中，我们使用了 `SYSCALL_DEFINE4` 宏来定义我们的函数定义，其中 `DEFINE4` 后缀仅仅意味着它接受四个参数。

在函数体的开头，我们在内核栈顶部声明了两个字符数组。与普通进程类似，内核进程有一个包含栈的地址空间。在完成这一步之后，我们将用户空间的数据复制到内核空间。随后，我们通过连接一些字符串来创建问候信息。这个字符串仍然在内核内存中。最后，我们将消息复制回用户空间，使其对调用进程可用。

在出现错误的情况下，会返回适当的错误号，以便让调用进程知道系统调用的结果。

使我们的系统调用工作下一步是更新另一个表。x86 和 x64 架构只有一个系统调用表，新添加的系统调用应该添加到这个表中以供暴露。

只有完成这一步后，系统调用才在 x86 和 x64 机器上可用。要将系统调用添加到表中，我们需要添加 `hello_word` 和其函数名 `sys_hello_world`。

要做到这一点，打开 `arch/x86/entry/syscalls/syscall_64.tbl` 文件，并在文件末尾添加以下行：

```cpp
999      64     hello_world             __x64_sys_hello_world
```

代码框 11-5：将新添加的 Hello World 系统调用添加到系统调用表

修改后，文件应如下所示：

```cpp
$ cat arch/x86/entry/syscalls/syscall_64.tbl
...
...
546     x32     preadv2                 __x32_compat_sys_preadv64v2
547     x32     pwritev2                __x32_compat_sys_pwritev64v2
999      64     hello_world             __x64_sys_hello_world
$
```

Shell 框 11-7：Hello World 系统调用添加到系统调用表

注意系统调用名称中的`__x64_`前缀。这是系统调用仅在 x64 系统中公开的指示。

Linux 内核使用 Make 构建系统编译所有源文件并构建最终的内核映像。接下来，您必须在`hello_world`目录中创建一个名为`Makefile`的文件。其内容，即一行文本，应该是以下内容：

```cpp
obj-y := sys_hello_world.o
```

代码框 11-6：Hello World 系统调用的 Makefile

然后，您需要将`hello_world`目录添加到根目录中的主`Makefile`中。切换到内核的根目录，打开`Makefile`文件，找到以下行：

```cpp
core-y  += kernel/certs/mm/fs/ipc/security/crypto/block/
```

代码框 11-7：应在根 Makefile 中修改的目标行

将`hello_world/`添加到该列表中。所有这些目录都是应该作为内核构建部分构建的目录。

我们需要添加 Hello World 系统调用的目录，以便将其包含在构建过程中，并在最终的内核映像中包含它。修改后，该行应类似于以下代码：

```cpp
core-y  += kernel/certs/mm/fs/hello_world/ipc/security/crypto/block/
```

代码框 11-8：修改后的目标行

下一步是构建内核。

### 构建内核

要构建内核，我们首先必须回到内核的根目录，因为在我们开始构建内核之前，您需要提供一个配置。配置包含应作为构建过程一部分构建的功能和单元列表。

以下命令尝试根据当前 Linux 内核的配置创建目标配置。它使用您内核中的现有值，并在我们试图构建的内核中存在较新的配置值时询问您进行确认。如果存在，您只需按`Enter`键即可简单地接受所有较新版本：

```cpp
$ make localmodconfig
...
...
#
# configuration written to .config
#
$
```

Shell 框 11-8：基于当前运行内核创建内核配置

现在，您可以开始构建过程。由于 Linux 内核包含大量源文件，构建可能需要数小时才能完成。因此，我们需要并行运行编译。

如果您正在使用虚拟机，请配置您的机器具有超过一个核心，以便在构建过程中获得有效的提升：

```cpp
$ make -j4
SYSHDR  arch/x86/include/generated/asm/unistd_32_ia32.h
SYSTBL  arch/x86/include/generated/asm/syscalls_32.h
HOSTCC  scripts/basic/bin2c
SYSHDR  arch/x86/include/generated/asm/unistd_64_x32.h
...
...
UPD     include/generated/compile.h
CC      init/main.o
CC      hello_world/sys_hello_world.o
CC      arch/x86/crypto/crc32c-intel_glue.o
...
...
LD [M]  net/netfilter/x_tables.ko
LD [M]  net/netfilter/xt_tcpudp.ko
LD [M]  net/sched/sch_fq_codel.ko
LD [M]  sound/ac97_bus.ko
LD [M]  sound/core/snd-pcm.ko
LD [M]  sound/core/snd.ko
LD [M]  sound/core/snd-timer.ko
LD [M]  sound/pci/ac97/snd-ac97-codec.ko
LD [M]  sound/pci/snd-intel8x0.ko
LD [M]  sound/soundcore.ko
$
```

Shell 框 11-9：内核构建的输出。请注意指示编译 Hello World 系统调用的行

**注意**：

确保您已经安装了本节第一部分介绍的先决条件软件包；否则，您将遇到编译错误。

如您所见，构建过程已经开始，有四个作业正在并行尝试编译 C 文件。您需要等待其完成。完成后，您可以轻松地安装新的内核并重新启动机器：

```cpp
$ sudo make modules_install install
INSTALL arch/x86/crypto/aes-x86_64.ko
INSTALL arch/x86/crypto/aesni-intel.ko
INSTALL arch/x86/crypto/crc32-pclmul.ko
INSTALL arch/x86/crypto/crct10dif-pclmul.ko
...
...
run-parts: executing /et/knel/postinst.d/initam-tools 5.3.0+ /boot/vmlinuz-5.3.0+
update-iniras: Generating /boot/initrd.img-5.3.0+
run-parts: executing /etc/keneostinst.d/unattende-urades 5.3.0+ /boot/vmlinuz-5.3.0+
...
...
Found initrd image: /boot/initrd.img-4.15.0-36-generic
Found linux image: /boot/vmlinuz-4.15.0-29-generic
Found initrd image: /boot/initrd.img-4.15.0-29-generic
done.  
$
```

Shell 框 11-10：创建和安装新的内核映像

如你所见，已经创建并安装了一个版本为 5.3.0 的新内核映像。现在我们准备好重启系统了。如果你不知道当前的内核版本，在重启之前不要忘记检查它。在我的情况下，我的版本是 `4.15.0-36-generic`。我使用了以下命令来找出它：

```cpp
$ uname -r
4.15.0-36-generic $
```

Shell 框 11-11：检查当前安装的内核版本

现在，使用以下命令重启系统：

```cpp
$ sudo reboot
```

Shell 框 11-12：重启系统

当系统启动时，新的内核映像将被选中并使用。请注意，引导加载程序不会选择旧内核；因此，如果你有一个版本高于 5.3 的内核，你需要手动加载构建的内核映像。这个链接可以帮助你：https://askubuntu.com/questions/82140/how-can-i-boot-with-an-older-kernel-version.

当操作系统启动完成时，你应该有新的内核正在运行。检查版本。它必须看起来像这样：

```cpp
$ uname -r
5.3.0+
$
```

Shell 框 11-13：重启后检查内核版本

如果一切顺利，新的内核应该已经就位。现在我们可以继续编写一个调用我们新添加的 Hello World 系统调用的 C 程序。它将非常类似于 *示例 11.1*，它调用了 `write` 系统调用。你可以在下面找到 *示例 11.2*：

```cpp
// We need to have this to be able to use non-POSIX stuff
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
// This is not part of POSIX!
#include <sys/syscall.h>
int main(int argc, char** argv) {
  char str[20] = "Kam";
  char message[64] = "";
  // Call the hello world system call
  int ret_val = syscall(999, str, 4, message, 64);
  if (ret_val < 0) {
    printf("[ERR] Ret val: %d\n", ret_val);
    return 1;
  }
  printf("Message: %s\n", message);
  return 0;
}
```

代码框 11-9 [ExtremeC_examples_chapter11_2.c]：示例 11.2 调用新添加的 Hello World 系统调用

如你所见，我们使用数字 `999` 调用了系统调用。我们传递 `Kam` 作为输入，并期望收到 `Hello Kam!` 作为问候消息。程序等待结果并在内核空间中打印由系统调用填充的消息缓冲区。

在以下代码中，我们构建并运行了示例：

```cpp
$ gcc ExtremeC_examples_chapter11_2.c -o ex11_2.out
$ ./ex11_2.out
Message: Hello Kam!
$
```

Shell 框 11-14：编译和运行示例 11.2

运行示例后，如果你使用 `dmesg` 命令查看内核日志，你会看到使用 `printk` 生成的日志：

```cpp
$ dmesg
...
...
[  112.273783] System call fired!
[  112.273786] Message: Hello Kam!
$
```

Shell 框 11-15：使用 dmesg 查看 Hello World 系统调用生成的日志

如果你使用 `strace` 运行 *示例 11.2*，你可以看到它实际上调用了系统调用 `999`。你可以在以 `syscall_0x3e7(...)` 开头的行中看到它。请注意，`0x3e7` 是 999 的十六进制值：

```cpp
$ strace ./ex11_2.out
...
...
mprotect(0x557266020000, 4096, PROT_READ) = 0
mprotect(0x7f8dd6d2d000, 4096, PROT_READ) = 0
munmap(0x7f8dd6d26000, 27048)           = 0
syscall_0x3e7(0x7fffe7d2af30, 0x4, 0x7fffe7d2af50, 0x40, 0x7f8dd6b01d80, 0x7fffe7d2b088) = 0
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 0), ...}) = 0
brk(NULL)                               = 0x5572674f2000
brk(0x557267513000)
...
...
exit_group(0)                           = ?
+++ exited with 0 +++
$
```

Shell 框 11-16：监控示例 11.2 所做的系统调用

在 *Shell 框 11-16* 中，你可以看到已经调用了 `syscall_0x3e7` 并返回了 `0`。如果你将 *示例 11.2* 中的代码修改为传递一个超过 64 字节的名称，你会收到一个错误。让我们修改示例并再次运行它：

```cpp
int main(int argc, char** argv) {
  char name[84] = "A very very long message! It is really hard to produce a big string!";
  char message[64] = "";
  ...
  return 0;
}
```

代码框 11-10：向我们的 Hello World 系统调用传递一个长消息（超过 64 字节）

让我们再次编译和运行它：

```cpp
$ gcc ExtremeC_examples_chapter11_2.c -o ex11_2.out
$ ./ex11_2.out
[ERR] Ret val: -1
$
```

Shell 框 11-17：修改后编译和运行示例 11.2

如你所见，系统调用根据我们为其编写的逻辑返回 `-1`。使用 `strace` 运行也显示系统调用返回了 `-1`：

```cpp
$ strace ./ex11_2.out
...
...
munmap(0x7f1a900a5000, 27048)           = 0
syscall_0x3e7(0x7ffdf74e10f0, 0x54, 0x7ffdf74e1110, 0x40, 0x7f1a8fe80d80, 0x7ffdf74e1248) = -1 (errno 1)
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 0), ...}) = 0
brk(NULL)                               = 0x5646802e2000
...
...
exit_group(1)                           = ?
+++ exited with 1 +++
$
```

Shell 框 11-18：监控修改后示例 11.2 所做的系统调用

在下一节中，我们将讨论设计内核可以采取的方法。作为我们讨论的一部分，我们介绍了内核模块，并探讨了它们在内核开发中的应用。

# Unix 内核

在本节中，我们将讨论过去 30 年中 Unix 内核所采用的架构。在讨论不同类型的内核之前——实际上种类并不多——我们应该知道，关于内核应该如何设计并没有标准化。

我们获得的最佳实践是基于多年的经验，它们引导我们形成了 Unix 内核内部单元的高级视图，这在上一章的*图 10-5*中有所体现。因此，每个内核与另一个内核相比都有所不同。它们共同的主要特点是它们应该通过系统调用接口来暴露其功能。然而，每个内核都有自己处理系统调用的独特方式。

这种多样性和围绕它的争论使它成为 20 世纪 90 年代最热门的计算机架构相关话题之一，有大量的人参与这些争论——其中*坦能鲍姆-托瓦尔斯*辩论被认为是其中最著名的一次。

我们不会深入这些辩论的细节，但我们要简要谈谈设计 Unix 内核的两种主要主导架构：*单核*和*微核*。仍然存在其他架构，如*混合内核*、*纳米内核*和*外核*，它们都有自己特定的用途。

然而，我们将通过创建一个比较来关注单核内核和微内核，以便我们可以了解它们的特性。

## 单核内核与微内核

在上一章中，我们讨论 Unix 架构时，将内核描述为包含许多单元的单个进程，但实际上我们实际上是在谈论一个单核内核。

单核内核由一个内核进程和一个地址空间组成，该地址空间包含在同一进程内的多个较小的单元。微内核则采取相反的方法。微内核是一个最小的内核进程，它试图将文件系统、设备驱动程序和进程管理等服务推到用户空间，以使内核进程更小、更薄。

这两种架构都有其优缺点，因此它们成为了操作系统历史上最著名的辩论之一。它始于 1992 年，Linux 第一个版本发布之后不久。由**安德鲁·S·坦能鲍姆**撰写的一篇帖子在*Usenet*上引发了一场辩论。这场辩论被称为坦能鲍姆-托瓦尔斯辩论。你可以在 https://en.wikipedia.org/wiki/Tanenbaum–Torvalds_debate 了解更多信息。

那篇帖子是 Linux 创建者**林纳斯·托瓦兹**与谭宁邦以及其他一些爱好者之间引发激烈争论的起点，这些人后来成为了第一批 Linux 开发者。他们正在辩论单核内核和微内核的本质。在这次激烈争论中，讨论了许多内核设计和硬件架构对内核设计的影响的不同方面。

对所描述的辩论和主题的进一步讨论将会很长且复杂，因此超出了本书的范围，但我们想比较这两种方法，并让您熟悉每种方法的优缺点。

以下是比较单核内核和微内核之间差异的列表：

+   单核内核由一个包含内核提供所有服务的单个进程组成。大多数早期的 Unix 内核都是这样开发的，这被认为是一种老方法。微内核与之不同，因为内核提供的每个服务都有单独的进程。

+   单核内核进程位于内核空间，而微内核中的*服务器进程*通常位于用户空间。服务器进程是那些提供内核功能的过程，例如内存管理、文件系统等。微内核与之不同，它们允许服务器进程位于用户空间。这意味着一些操作系统比其他操作系统更类似于微内核。

+   单核内核通常更快。这是因为所有内核服务都在内核进程中执行，但微内核需要在用户空间和内核空间之间进行一些*消息传递*，因此需要更多的系统调用和上下文切换。

+   在单核内核中，所有设备驱动程序都加载到内核中。因此，第三方供应商编写的设备驱动程序将作为内核的一部分运行。任何设备驱动程序或内核内部其他单元的任何缺陷都可能导致内核崩溃。这与微内核的情况不同，因为所有的设备驱动程序和许多其他单元都在用户空间中运行，我们可以假设这就是为什么单核内核没有被用于关键任务项目的原因。

+   在单核内核中，注入一小段恶意代码就足以破坏整个内核，进而破坏整个系统。然而，在微内核中这种情况不太可能发生，因为许多服务器进程位于用户空间，只有最小的一组关键功能集中在内核空间。

+   在单一内核中，即使是内核源代码的简单更改也需要重新编译整个内核，并生成新的内核映像。加载新的映像还需要重新启动机器。但在微内核中，更改可以导致仅编译特定的服务器进程，并且可能在不重新启动系统的情况下加载新的功能。在单一内核中，可以通过内核模块在一定程度上获得类似的功能。

MINIX 是微内核最著名的例子之一。它是由 Andrew S. Tanenbaum 编写的，最初是一个教育操作系统。Linus Torvalds 在 1991 年为 80386 微处理器编写自己的内核 Linux 时，使用了 MINIX 作为他的开发环境。

由于 Linux 几乎是近 30 年来最大的、最成功的单一内核捍卫者，我们将在下一节中更多地讨论 Linux。

## Linux

在本章前面的部分，当我们为它开发一个新的系统调用时，你已经了解了 Linux 内核。在本节中，我们想更多地关注 Linux 是单一内核的事实，以及每个内核功能都在内核内部。

然而，应该有一种方法可以在不重新编译内核的情况下添加新的功能。由于，正如你所看到的，添加一个新的系统调用需要更改许多基本文件，这意味着我们需要重新编译内核以获得新的功能。

新的方法是不同的。在这种技术中，内核模块被编写并动态地插入内核中，我们将在第一部分讨论这一点，然后再继续编写 Linux 内核模块。

## 内核模块

单一内核通常配备另一个设施，使内核开发者能够将新的功能热插拔到正在运行的内核中。这些可插入单元被称为内核模块。这些与微内核中的服务器进程不同。

与微内核中的服务器进程不同，微内核中的服务器进程实际上是使用 IPC 技术相互通信的独立进程，内核模块是已经编译好的内核对象文件，可以动态地加载到内核进程中。这些内核对象文件可以成为内核映像的一部分静态构建，或者当内核正在运行时动态加载。

注意，内核对象文件是 C 开发中产生的普通对象文件的双胞胎概念。

值得再次注意的是，如果内核模块在内核内部做了一些坏事，可能会发生内核崩溃。

与系统调用不同，与内核模块的通信方式不同，不能通过调用函数或使用给定的 API 来使用。通常，在 Linux 和一些类似操作系统中，与内核模块通信有三种方式：

+   **/dev 目录中的设备文件**：内核模块主要是为了被设备驱动程序使用而开发的，这也是为什么设备是与内核模块通信的最常见方式。正如我们在上一章中解释的，设备作为位于 `/dev` 目录中的设备文件是可访问的。你可以从这些文件中读取和写入，并使用它们，你可以向/从模块发送和接收数据。

+   **procfs 中的条目**：`/proc` 目录中的条目可以用来读取特定内核模块的元信息。这些文件也可以用来传递元信息或控制命令给内核模块。我们将在下一示例中简要演示 procfs 的用法，即 *示例 11.3*，作为以下部分的内容。

+   **sysfs 中的条目**：这是 Linux 中的另一个文件系统，允许脚本和用户控制用户进程以及其他与内核相关的单元，例如内核模块。它可以被认为是 procfs 的新版本。

实际上，最好的方法是编写一个内核模块，这正是我们在下一节将要做的，我们将为 Linux 编写一个 Hello World 内核模块。请注意，内核模块不仅限于 Linux；像 FreeBSD 这样的单核内核也受益于内核模块机制。

### 将内核模块添加到 Linux

在本节中，我们将编写一个新的 Linux 内核模块。这是一个 Hello World 内核模块，它在 procfs 中创建一个条目。然后，使用这个条目，我们读取问候字符串。

在本节中，你将熟悉编写内核模块、编译它、将其加载到内核中、从内核中卸载它以及从 procfs 条目中读取数据。本示例的主要目的是让你亲自动手编写内核模块，从而可以自己进行更多开发。

**注意**：

内核模块被编译成可以在运行时直接加载到内核中的内核对象文件。只要内核模块对象文件没有在内核中做任何导致内核崩溃的坏事，就不需要重新启动系统。卸载内核模块也是如此。

第一步是创建一个目录，该目录将包含所有与内核模块相关的文件。我们将其命名为 `ex11_3`，因为这是本章的第三个示例：

```cpp
$ mkdir ex11_3
$ cd ex11_3
$
```

Shell 框 11-19：为示例 11.3 创建根目录

然后，创建一个名为 `hwkm.c` 的文件，它只是由 "Hello World Kernel Module" 的首字母组成的缩写，其内容如下：

```cpp
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
// The structure pointing to the proc file
struct proc_dir_entry *proc_file;
// The read callback function
ssize_t proc_file_read(struct file *file, char __user *ubuf, size_t count, loff_t *ppos) {
  int copied = 0;
  if (*ppos > 0) {
    return 0;
  }
  copied = sprintf(ubuf, "Hello World From Kernel Module!\n");
  *ppos = copied;
  return copied;
}
static const struct file_operations proc_file_fops = {
 .owner = THIS_MODULE,
 .read  = proc_file_read
};
// The module initialization callback
static int __init hwkm_init(void) {
  proc_file = proc_create("hwkm", 0, NULL, &proc_file_fops);
  if (!proc_file) {
    return -ENOMEM;
  }
  printk("Hello World module is loaded.\n");
  return 0;
}
// The module exit callback
static void __exit hkwm_exit(void) {
  proc_remove(proc_file);
  printk("Goodbye World!\n");
}
// Defining module callbacks
module_init(hwkm_init);
module_exit(hkwm_exit);
```

代码框 11-11 [ex11_3/hwkm.c]：Hello World 内核模块

使用 *代码框 11-11* 中的最后两条语句，我们已经注册了模块的初始化和退出回调函数。这些函数分别在模块加载和卸载时被调用。初始化回调是首先执行的代码。

如您在 `hwkm_init` 函数内部所见，它会在 `/proc` 目录下创建一个名为 `hwkm` 的文件。还有一个退出回调。在 `hwkm_exit` 函数内部，它会从 `/proc` 路径中删除 `hwkm` 文件。`/proc/hwkm` 文件是用户空间与内核模块通信的接触点。

`proc_file_read` 函数是读取回调函数。当用户空间尝试读取 `/proc/hwkm` 文件时，会调用此函数。您很快就会看到，我们使用 `cat` 工具程序来读取文件。它简单地将 `Hello World From Kernel Module!` 字符串复制到用户空间。

注意，在这个阶段，内核模块内部编写的代码几乎可以访问内核内部的任何内容，并且它可以向用户空间泄露任何类型的信息。这是一个主要的安全问题，应该进一步阅读有关编写安全内核模块的最佳实践的资料。

要编译前面的代码，我们需要使用适当的编译器，可能还需要将其与适当的库链接。为了使生活更简单，我们创建了一个名为 `Makefile` 的文件，该文件将触发必要的构建工具以构建内核模块。

以下代码框显示了 `Makefile` 的内容：

```cpp
obj-m += hwkm.o
all:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
```

Code Box 11-12：Hello World 内核模块的 Makefile

然后，我们可以运行 `make` 命令。以下 shell 窗口演示了这一点：

```cpp
$ make
make -C /lib/modules/54.318.0+/build M=/home/kamran/extreme_c/ch11/codes/ex11_3 modules
make[1]: Entering directory '/home/kamran/linux'
  CC [M]  /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.o
  Building modules, stage 2.
  MODPOST 1 modules
WARNING: modpost: missing MODULE_LICENSE() in /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.o
see include/linux/module.h for more information
  CC      /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.mod.o
  LD [M]  /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.ko
make[1]: Leaving directory '/home/kamran/linux'
$
```

Shell Box 11-20：构建 Hello World 内核模块

如您所见，编译器编译代码并生成一个对象文件。然后，它继续将对象文件与其他库链接以创建一个 `.ko` 文件。现在，如果您查看生成的文件，您会发现一个名为 `hwkm.ko` 的文件。

注意到 `.ko` 扩展名，它仅仅意味着输出文件是一个内核对象文件。它就像一个可以动态加载到内核并运行的共享库。

请注意，在 *Shell Box 11-20* 中，构建过程生成了一个警告消息。它表示该模块没有与之关联的许可证。在开发和部署内核模块的测试和生产环境中，生成授权模块是一种高度推荐的做法。

以下 shell 窗口显示了构建内核模块后可以找到的文件列表：

```cpp
$ ls -l
total 556
-rw-rw-r-- 1 kamran kamran    154 Oct 19 00:36 Makefile
-rw-rw-r-- 1 kamran kamran      0 Oct 19 08:15 Module.symvers
-rw-rw-r-- 1 kamran kamran   1104 Oct 19 08:05 hwkm.c
-rw-rw-r-- 1 kamran kamran 272280 Oct 19 08:15 hwkm.ko
-rw-rw-r-- 1 kamran kamran    596 Oct 19 08:15 hwkm.mod.c
-rw-rw-r-- 1 kamran kamran 104488 Oct 19 08:15 hwkm.mod.o
-rw-rw-r-- 1 kamran kamran 169272 Oct 19 08:15 hwkm.o
-rw-rw-r-- 1 kamran kamran     54 Oct 19 08:15 modules.order
$
```

Shell Box 11-21：构建 Hello World 内核模块后的现有文件列表

**注意**：

我们使用了 Linux 内核版本 5.3.0 的模块构建工具。如果您使用低于 3.10 的内核版本编译此示例，可能会得到编译错误。

要加载 `hwkm` 内核模块，我们使用 Linux 中的 `insmod` 命令，它简单地加载并安装内核模块，就像我们在以下 shell 窗口中做的那样：

```cpp
$ sudo insmod hwkm.ko
$
```

Shell Box 11-22：加载和安装 Hello World 内核模块

现在，如果您查看内核日志，您将看到由初始化函数产生的行。只需使用 `dmesg` 命令查看最新的内核日志，这是我们接下来要做的：

```cpp
$ dmesg
...
...
[ 7411.519575] Hello World module is loaded.
$
```

Shell Box 11-23：安装内核模块后的内核日志消息检查

现在，模块已经加载，应该已经创建了 `/proc/hwkm` 文件。我们可以通过使用 `cat` 命令来读取它：

```cpp
$ cat /proc/hwkm
Hello World From Kernel Module!
$ cat /proc/hwkm
Hello World From Kernel Module!
$
```

Shell Box 11-24：使用 cat 读取 /proc/hwkm 文件

如您在前面的 shell 窗口中看到的，我们读取了文件两次，两次都返回了相同的 `Hello World From Kernel Module!` 字符串。请注意，该字符串是由内核模块复制到用户空间的，而 `cat` 程序只是将其打印到标准输出。

当涉及到卸载模块时，我们可以使用 Linux 中的 `rmmod` 命令，就像我们接下来要做的：

```cpp
$ sudo rmmod hwkm
$
```

Shell Box 11-25：卸载 Hello World 内核模块

现在模块已经卸载，再次查看内核日志以查看再见信息：

```cpp
$ dmesg
...
...
[ 7411.519575] Hello World module is loaded.
[ 7648.950639] Goodbye World!
$
```

Shell Box 11-26：卸载内核模块后的内核日志消息检查

正如您在前面的示例中看到的，内核模块在编写内核代码时非常方便。

为了完成本章，我相信提供一个关于我们迄今为止所看到的内核模块功能的列表将会有所帮助：

+   内核模块可以在不重新启动机器的情况下加载和卸载。

+   当加载时，它们成为内核的一部分，可以访问内核中的任何单元或结构。这可以被认为是一个漏洞，但 Linux 内核可以保护自己免受安装不受欢迎的模块的影响。

+   在内核模块的情况下，您只需要编译它们的源代码。但对于系统调用，您必须编译整个内核，这可能会占用您一个小时的时间。

最后，当您要开发需要在系统调用背后运行的代码时，内核模块可能很有用。将要使用系统调用暴露的逻辑可以先通过内核模块加载到内核中，经过适当的开发和测试后，它可以放在真正的系统调用后面。

从头开始开发系统调用可能是一项繁琐的工作，因为您不得不无数次地重新启动您的机器。将逻辑首先作为内核模块的一部分编写和测试可以减轻内核开发的痛苦。请注意，如果您的代码试图导致内核崩溃，无论是内核模块还是系统调用之后，都会导致内核崩溃，您必须重新启动您的机器。

在本节中，我们讨论了各种类型的内核。我们还展示了如何在单核内核中通过动态加载和卸载来使用内核模块实现瞬态内核逻辑。

# 概述

我们现在已经完成了关于 Unix 的两章讨论。在本章中，我们学习了以下内容：

+   系统调用是什么以及它是如何暴露特定功能的

+   系统调用调用背后的发生情况

+   如何直接从 C 代码中调用某个系统调用

+   如何向现有的类 Unix 内核（Linux）添加一个新的系统调用以及如何重新编译内核

+   什么是单核内核以及它与微内核的区别

+   内核模块如何在单核内核中工作以及如何为 Linux 编写一个新的内核模块

在接下来的章节中，我们将讨论 C 标准以及最新的 C 标准版本，C18。您将熟悉其中引入的新特性。
