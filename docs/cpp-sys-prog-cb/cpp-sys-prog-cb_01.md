# 开始系统编程

在本章中，你将被介绍整本书的基础知识。你将学习（或者复习）Linux 的设计，还将学习关于 shell、用户和用户组、进程 ID 和线程 ID，以便能够熟练地使用 Linux 系统，并为接下来的章节做好准备。此外，你还将学习如何开发一个简单的`hello world`程序，了解它的 makefile，以及如何执行和调试它。本章的另一个重要方面是学习 Linux 如何处理错误，无论是从 shell 还是源代码的角度。这些基础知识对于理解接下来章节中的其他高级主题非常重要。如果不需要这个复习，你可以安全地跳过本章和下一章。

本章将涵盖以下内容：

+   学习 Linux 基础知识- 架构

+   学习 Linux 基础知识- shell

+   学习 Linux 基础知识- 用户

+   使用 makefile 来编译和链接程序

+   使用 GNU Project Debugger（GDB）调试程序

+   学习 Linux 基础知识- 进程和线程

+   处理 Linux bash 错误

+   处理 Linux 代码错误

# 技术要求

为了让你立即尝试这些程序，我们设置了一个 Docker 镜像，其中包含了整本书中需要的所有工具和库。这是基于 Ubuntu 19.04 的。

为了设置这个，按照以下步骤进行：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  镜像现在应该是可用的。输入以下命令查看镜像：`docker images`。

1.  现在你应该至少有这个镜像：`kasperondocker/system_programming_cookbook`。

1.  使用以下命令在 Docker 镜像上运行交互式 shell：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。运行`root@39a5a8934370/# cd /BOOK/`来获取所有按章节开发的程序。

需要`--cap-add sys_ptrace`参数来允许 Docker 容器中的 GDB 设置断点，默认情况下 Docker 不允许这样做。

# 学习 Linux 基础知识- 架构

Linux 是 Unix 操作系统的一个克隆，由 Linus Torvalds 在 90 年代初开发。它是一个多用户、多任务操作系统，可以运行在各种平台上。Linux 内核采用了单体结构的架构，出于性能原因。这意味着它是一个自包含的二进制文件，所有的服务都在内核空间运行。这在开始时是最具争议的话题之一。阿姆斯特丹自由大学的教授安迪·塔能鲍姆反对其单体系统，他说：“这是对 70 年代的巨大倒退。”他还反对其可移植性，说：“LINUX 与 80 x 86 紧密联系在一起。不是正确的方向。”在 minix 用户组中，仍然有涉及 Torvalds、Tanenbaum 和其他人的完整聊天记录。

以下图表显示了主要的 Linux 构建模块：

![](img/9a7b4405-e9e4-431b-b068-0883d1189150.png)

让我们描述一下图表中看到的层次：

+   在顶层，有用户应用程序、进程、编译器和工具。这一层（在用户空间运行）通过系统调用与 Linux 内核（在内核空间运行）进行通信。

+   系统库：这是一组函数，通过它应用程序可以与内核进行交互。

+   内核：这个组件包含 Linux 系统的核心。除其他功能外，它还有调度程序、网络、内存管理和文件系统。

+   **内核模块**：这些包含仍在内核空间中运行的内核代码片段，但是完全动态（可以在运行系统中加载和卸载）。它们通常包含设备驱动程序、特定于实现协议的特定硬件模块的内核代码等。内核模块的一个巨大优势是用户可以在不重新构建内核的情况下加载它们。

**GNU**是一个递归缩写，代表**GNU 不是 Unix**。GNU 是一个自由软件的操作系统。请注意这里的术语*操作系统*。事实上，单独使用的 GNU 意味着代表操作系统所需的一整套工具、软件和内核部分。GNU 操作系统内核称为**Hurd**。由于 Hurd 尚未达到生产就绪状态，GNU 通常使用 Linux 内核，这种组合被称为**GNU/Linux 操作系统**。

那么，在 GNU/Linux 操作系统上的 GNU 组件是什么？例如**GNU 编译器集合**（**GCC**）、**GNU C 库**、GDB、GNU Bash shell 和**GNU 网络对象模型环境**（**GNOME**）桌面环境等软件包。Richard Stallman 和**自由软件基金会**（**FSF**）——Stallman 是创始人——撰写了**自由软件定义**，以帮助尊重用户的自由。*自由软件*被认为是授予用户以下四种自由（所谓的**基本自由**：[`isocpp.org/std/the-standard`](https://isocpp.org/std/the-standard)）的任何软件包：

1.  自由按照您的意愿运行程序，无论任何目的（自由*0*）。

1.  自由研究程序如何工作并对其进行更改，以便按照您的意愿进行计算（自由*1*）。访问源代码是这一自由的前提条件。

1.  自由重新分发副本，以便您可以帮助他人（自由*2*）。

1.  自由向他人分发您修改版本的副本（自由*3*）。通过这样做，您可以让整个社区有机会从您的更改中受益。访问源代码是这一自由的前提条件。

这些原则的具体实现在 FSF 撰写的 GNU/GPL 许可证中。所有 GNU 软件包都是根据 GNU/GPL 许可证发布的。

# 如何做...

Linux 在各种发行版中有一个相当标准的文件夹结构，因此了解这一点将使您能够轻松地找到程序并将其安装在正确的位置。让我们来看一下：

1.  在 Docker 镜像上打开终端。

1.  键入命令`ls -l /`。

# 它是如何工作的...

命令的输出将包含以下文件夹：

![](img/4a845486-dd7f-40b4-a271-fc851692fe1e.png)

正如您所看到的，这个文件夹结构非常有组织且在所有发行版中保持一致。在 Linux 文件系统底层，它相当模块化和灵活。用户应用程序可以与 GNU C 库（提供诸如 open、read、write 和 close 等接口）或 Linux 系统调用直接交互。在这种情况下，系统调用接口与**虚拟文件系统**（通常称为**VFS**）交谈。VFS 是对具体文件系统实现（例如 ext3、**日志文件系统**（**JFS**）等）的抽象。正如我们可以想象的那样，这种架构提供了高度的灵活性。

# 学习 Linux 基础知识-Shell

Shell 是一个命令解释器，它接收输入中的命令，将其重定向到 GNU/Linux，并返回输出。这是用户和 GNU/Linux 之间最常见的接口。有不同的 shell 程序可用。最常用的是 Bash shell（GNU 项目的一部分）、tcsh shell、ksh shell 和 zsh shell（这基本上是一个扩展的 Bash shell）。

为什么需要 shell？如果用户需要通过**命令行**与操作系统进行交互，则需要 shell。在本食谱中，我们将展示一些最常见的 shell 命令。通常情况下，*shell*和*终端*这两个术语可以互换使用，尽管严格来说它们并不完全相同。

# 如何做……

在本节中，我们将学习在 shell 上运行的基本命令，例如查找文件、在文件中查找`grep`、复制和删除：

1.  打开 shell：根据 GNU/Linux 发行版的不同，打开新 shell 命令有不同的快捷键。在 Ubuntu 上，按*Ctrl* + *Alt* + *T*，或按*Alt* + *F2*，然后输入`gnome-terminal`。

1.  关闭 shell：要关闭终端，只需键入`exit`并按*Enter*。

1.  `find`命令：用于在目录层次结构中搜索文件。在其最简单的形式中，它看起来像这样：

```cpp
find . -name file
```

它也支持通配符：

```cpp
$ find /usr/local "python*"
```

1.  `grep`命令通过匹配模式打印行：

```cpp
 $ grep "text" filename
```

`grep`还支持递归搜索：

```cpp
 $ grep "text" -R /usr/share
```

1.  管道命令：在 shell 上运行的命令可以连接起来，使一个命令的输出成为另一个命令的输入。连接是使用`|`（管道）运算符完成的：

```cpp
$ ls -l | grep filename
```

1.  编辑文件：在 Linux 上编辑文件的最常用工具是`vi`和`emacs`（如果您对编辑文件不感兴趣，`cat filename`将文件打印到标准输出）。前者是 Unix 操作系统的一部分，后者是 GNU 项目的一部分。本书将广泛使用`vi`：

```cpp
 $ vi filename
```

接下来，我们将看一下与文件操作相关的 shell 命令。

1.  这是删除文件的命令：

```cpp
$ rm filename
```

1.  这是删除目录的命令：

```cpp
$ rm -r directoryName
```

1.  这是克隆文件的命令：

```cpp
$ cp file1 file2
```

1.  这是克隆文件夹的命令：

```cpp
$ cp -r folder1 folder2  
```

1.  这是使用相对路径和绝对路径克隆文件夹的命令：

```cpp
$ cp -r /usr/local/folder1 relative/folder2
```

下一节将描述这些命令。

# 它是如何工作的……

让我们详细了解*如何做……*部分中讨论的命令：

1.  第一个命令从当前文件夹搜索（`.`），可以包含绝对路径（例如`/usr/local`）或相对路径（例如`tmp/binaries`）。例如，在这里，`-name`是要搜索的文件。

1.  第二个命令从`/usr/local`文件夹搜索以`python`开头的任何文件或文件夹。`find`命令提供了巨大的灵活性和各种选项。有关更多信息，请通过`man find`命令参考`man page`。

1.  `grep`命令搜索并打印包含`filename`文件中的`text`单词的任何行。

1.  `grep`递归搜索命令搜索并打印任何包含`text`单词的行，从`/usr/share`文件夹递归搜索任何文件。

1.  管道命令（`|`）：第一个命令的输出显示在以下截图中。所有文件和目录的列表作为输入传递给第二个命令（`grep`），将用于`grep`文件名：

![](img/11e480e2-e934-4db1-8d69-fe05a480546d.png)

现在，让我们看一下执行编辑文件、添加/删除文件和目录等操作的命令。

编辑文件：

+   `vi`命令将以编辑模式打开文件名，假设当前用户对其具有写入权限（我们将稍后更详细地讨论权限）。

以下是`vi`中最常用命令的简要总结：

+   *Shift + :*（即*Shift*键+冒号）切换到编辑模式。

+   *Shift + :i*插入。

+   *Shift + :a*追加。

+   *Shift + :q!*退出当前会话而不保存。

+   *Shift + :wq*保存并退出当前会话。

+   *Shift + :set nu*显示文件的行号。

+   *Shift + :23*（*Enter*）转到第 23 行。

+   按下（*Esc*）键切换到命令模式。

+   *.*重复上一个命令。

+   *cw*更改单词，或者通过将光标指向单词的开头来执行此操作。

+   *dd*删除当前行。

+   *yy*复制当前行。如果在*yy*命令之前选择了数字*N*，则将复制*N*行。

+   *p*粘贴使用*yy*命令复制的行。

+   *u*取消。

**添加和删除文件和目录**：

1.  第一个命令删除名为`filename`的文件。

1.  第二个命令递归地删除`directoryName`及其内容。

1.  第三个命令创建了`file2`，它是`file1`的精确副本。

1.  第四个命令创建`folder2`作为`folder1`的克隆：

![](img/1d86e0fb-6dda-477a-b460-51fc4ae8f88a.png)

在本教程中所示的命令执行中存在一个常见模式。它们列举如下：

1.  用户输入命令并按*Enter*。

1.  该命令由 Linux 解释。

1.  Linux 与其不同的部分（内存管理、网络、文件系统等）进行交互以执行命令。这发生在内核空间**。**

1.  结果返回给用户。

# 还有更多...

本教程展示了一些最常见的命令。掌握所有选项，即使只是对于最常见的 shell 命令，也是棘手的，这就是为什么创建了`man pages`。它们为 Linux 用户提供了坚实清晰的参考。

# 另请参阅

第八章，*处理控制台 I/O 和文件*，将更深入地介绍控制台 I/O 和文件管理。

# 学习 Linux 基础知识-用户

Linux 是一个多用户和多任务操作系统，因此基本的用户管理技能是必不可少的。本教程将向您展示文件和目录权限的结构，如何添加和删除用户，如何更改用户的密码以及如何将用户分配给组。

# 如何做...

以下一系列步骤显示了基本用户管理活动的有用命令：

1.  **创建用户**：为每个使用 Linux 的个人配置一个用户不仅是最佳实践，而且也是推荐的。创建用户非常简单：

```cpp
root@90f5b4545a54:~# adduser spacex --ingroup developers
Adding user `spacex' ...
Adding new user `spacex' (1001) with group `developers' ...
Creating home directory `/home/spacex' ...
Copying files from `/etc/skel' ...
New password:
Retype new password:
passwd: password updated successfully
Changing the user information for spacex
Enter the new value, or press ENTER for the default
Full Name []: Onorato
Room Number []:
Work Phone []:
Home Phone []:
Other []:
Is the information correct? [Y/n] Y
```

`spacex`用户已创建并分配给现有的`developers`组。要切换到新创建的用户，请使用新用户的凭据登录：

```cpp
root@90f5b4545a54:~# login spacex
Password:
Welcome to Ubuntu 19.04 (GNU/Linux 4.9.125-linuxkit x86_64)
* Documentation: https://help.ubuntu.com
* Management: https://landscape.canonical.com
* Support: https://ubuntu.com/advantage
This system has been minimized by removing packages and content that are
not required on a system that users do not log into.
To restore this content, you can run the 'unminimize' command.
The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.
Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.
spacex@90f5b4545a54:~$
```

1.  **更新用户密码**：定期更改密码是必要的。以下是执行此操作的命令：

```cpp
spacex@90f5b4545a54:~$ passwd
Changing password for spacex.
 Current password:
 New password:
 Retype new password:
 passwd: password updated successfully
 spacex@90f5b4545a54:~$
```

1.  **将用户分配给组**：如图所示，可以在创建用户时将用户分配给组。或者，可以随时通过运行以下命令将用户分配给组：

```cpp
root@90f5b4545a54:~# usermod -a -G testers spacex
 here spacex is added to the testers group
```

1.  **删除用户**：同样，删除用户非常简单：

```cpp
root@90f5b4545a54:~# userdel -r spacex
userdel: spacex mail spool (/var/mail/spacex) not found
root@90f5b4545a54:~#
```

-r 选项表示删除 spacex 主目录和邮件邮箱。

1.  现在，让我们看看最后一个命令，它显示当前用户（`spacex`）所属的组的列表：

```cpp
spacex@90f5b4545a54:~$ groups
 developers testers
 spacex@90f5b4545a54:~$
```

如您所见，`spacex`用户属于`developers`和`testers`组。

# 工作原理...

在*步骤 1*中，我们使用`adduser`命令添加了`spacex`用户，并在上下文中将用户添加到`developers`组。

*步骤 2*显示了如何更改当前用户的密码。要更改密码，必须提供先前的密码。定期更改密码是一个好习惯。

如果我们想将用户分配给组，可以使用`usermod`命令。在*步骤 3*中，我们已将`spacex`用户添加到`testers`组。`-a`和`-G`参数只是指示新组（`-G`）将被附加到用户的当前组（`-a`）上。也就是说，`spacex`用户将被分配到`testers`组，该组将在上下文中创建。在同一步骤中，`groups`命令显示当前用户属于哪些组。如果您只想创建一个组，那么`groupadd group-name`就是您需要的命令。

*步骤 4*显示了如何使用`userdel`命令删除用户，传递`-r`参数。此参数确保将删除要删除的用户的所有文件。

# 还有更多...

在 Linux 文件系统中，每个文件和目录都有一组信息，定义了谁可以做什么。这个机制既简单又强大。文件（或目录）上允许的操作有读取、写入和执行（`r`、`w`和`x`）。这些操作可以由文件或目录的所有者、一组用户或所有用户执行。Linux 用 Owner: `rwx`；Group: `rwx`；All Users: `rwx`来表示这些信息；或者更简单地表示为`rwx-rwx-rwx`（总共 9 个）。实际上，Linux 在这些标志之上还有一个表示文件类型的标志。它可以是一个文件夹（`d`）、一个符号链接到另一个文件（`l`）、一个常规文件（`-`）、一个命名管道（`p`）、一个套接字（`s`）、一个字符设备文件（`c`）和一个块设备（`b`）。文件的典型权限看起来像这样：

```cpp
root@90f5b4545a54:/# ls -l
 -rwxr-xr-x 1 root root 13 May 8 20:11 conf.json
```

让我们详细看一下：

+   从左边开始阅读，第一个字符`-`告诉我们`conf.json`是一个常规文件。

+   接下来的三个字符是关于当前用户的，`rwx`。用户对文件有完全的**读取**（**r**）、**写入**（**w**）和**执行**（**x**）权限。

+   接下来的三个字符是关于用户所属的组，`r-x`。所有属于该组的用户都可以读取和执行文件，但不能修改它（`w`未被选择，标记为`-`）。

+   最后的三个字符是关于所有其他用户，`r-x`。所有其他用户只能读取和执行文件（`r`和`x`被标记，但`w`没有）。

所有者（或 root 用户）可以更改文件的权限。实现这一点的最简单方法是通过`chmod`命令：

```cpp
 $ chmod g+w conf.json 
```

在这里，我们要求 Linux 内核向组用户类型（`g`）添加写权限（`w`）。用户类型有：`u`（用户）、`o`（其他人）、`a`（所有人）和`g`（组），权限标志可以是`x`、`w`和`r`，如前所述。`chmod`也可以接受一个整数：

```cpp
 $ chmod 751 conf.json 
```

对于每种组类型的权限标志，有一个二进制到十进制的转换，例如：

`wxr`：111 = 7

`w-r`：101 = 5

`--r`：001 = 1

一开始可能有点神秘，但对于日常使用来说非常实用和方便。

# 另请参阅

`man`页面是一个无限的信息资源，应该是你查看的第一件事。像`man groups`、`man userdel`或`man adduser`这样的命令会对此有所帮助。

# 使用`makefile`来编译和链接程序

`makefile`是描述程序源文件之间关系的文件，由`make`实用程序用于构建（编译和链接）目标目标（可执行文件、共享对象等）。`makefile`非常重要，因为它有助于保持源文件的组织和易于维护。要使程序可执行，必须将其编译并链接到其他库中。GCC 是最广泛使用的编译器集合。C 和 C++世界中使用的两个编译器是 GCC 和 g++（分别用于 C 和 C++程序）。本书将使用 g++。

# 如何做...

这一部分将展示如何编写一个`makefile`，来编译和运行一个简单的 C++程序。我们将开发一个简单的程序，并创建它的`makefile`来学习它的规则：

1.  让我们从打开`hello.cpp`文件开始开发程序：

```cpp
$vi hello.cpp
```

1.  输入以下代码（参考*学习 Linux 基础知识- shell*中的`vi`命令）：

```cpp
#include <iostream>
int main()
{
    std::cout << "Hello World!" << std::endl;
    return 0;
}
```

1.  保存并退出：在`vi`中，从命令模式下，输入`:wq`，表示写入并退出。`:x`命令具有相同的效果。

1.  从 shell 中，创建一个名为`Makefile`的新文件：

```cpp
$ vi Makefile
```

1.  输入以下代码：

```cpp
CC = g++
all: hello
hello: hello.o
      ${CC} -o hello hello.o
hello.o: hello.cpp
      ${CC} -c hello.cpp
clean:
      rm hello.o hello
```

尽管这是一个典型的`Hello World!`程序，但它很有用，可以展示一个`makefile`的结构。

# 它是如何工作的...

简单地说，`makefile`由一组规则组成。规则由一个目标、一组先决条件和一个命令组成。

在第一步中，我们打开了文件（`hello.cpp`）并输入了*步骤 2*中列出的程序。同样，我们打开了另一个文件`Makefile`，在`hello.cpp`程序的相同文件夹中，并输入了特定的 makefile 命令。现在让我们深入了解 makefile 的内部。典型的 makefile 具有以下内容：

1.  第一个规则包括一个名为`all`的目标和一个名为`hello`的先决条件。这个规则没有命令。

1.  第二个规则包括一个名为`hello`的目标。它有一个对`hello.o`的先决条件和一个链接命令：`g++`。

1.  第三个规则有一个名为`hello.o`的目标，一个对`hello.cpp`的先决条件和一个编译命令：`g++ -c hello.cpp`。

1.  最后一个规则有一个`clean`目标，带有一个命令来删除所有`hello`和`hello.o`可执行文件。这会强制重新编译文件。

1.  对于任何规则，如果任何源文件发生更改，则执行定义的命令。

现在我们可以使用我们创建的 makefile 来编译程序：

```cpp
$ make
```

我们还可以执行程序，其输出如下：

![](img/2ffd955d-5371-4d04-a52b-13cf17e6eeaf.png)

从源文件生成二进制可执行文件的过程包括编译和链接阶段，这里压缩在一个单独的命令中；在大多数情况下都是这样。一般来说，大型系统代码库依赖于更复杂的机制，但步骤仍然是相同的：源文件编辑、编译和链接。

# 还有更多...

这个简单的例子只是向我们展示了 makefile 及其`make`命令的基本概念。它比这更多。以下是一些例子：

1.  宏的使用：makefile 允许使用宏，它们可以被视为**变量**。这些可以用于组织 makefile 以使其更加模块化，例如：

+   程序中使用的所有动态库的宏：`LIBS = -lxyz -labc`。

+   编译器本身的宏（如果要更改为其他编译器）：`COMPILER = GCC`。

+   在整个 makefile 中引用这些宏：`$(CC)`。这使我们可以在一个地方进行更改。

1.  只需在 shell 上输入`make`，就会运行 makefile 中定义的第一个规则。在我们的情况下，第一个规则是`all`。如果我们通过将**`clean`**作为第一个规则来更改 makefile，运行不带参数的`make`将执行`clean`规则。通常，您总是会传递一些参数，例如`make clean`。

# 使用 GDB 调试程序

调试是从软件系统中识别和消除错误的过程。GNU/Linux 操作系统有一个**标准** *事实上*的工具（即不是任何标准的一部分，但几乎在 Linux 世界中被任何人使用）称为 GDB。安装在本书的 Docker 上的 GDB 版本是 8.2.91。当然，有一些可以在 GDB 下使用的图形工具，但在 Linux 上，GDB 是可靠、简单和快速的选择。在这个示例中，我们将调试我们在上一个示例中编写的软件。

# 如何做...

为了使用一些 GDB 命令，我们需要修改之前的程序并在其中添加一些变量：

1.  打开一个 shell，并通过输入以下代码修改`hello.cpp`文件：

```cpp
 #include <iostream>
 int main()
 {
    int x = 10;
    x += 2;
    std::cout << "Hello World! x = " << x << std::endl;
    return 0;
 }
```

这是一个非常简单的程序：取一个变量，加上`2`，然后打印结果。

1.  通过输入以下命令，确保程序已编译：

```cpp
root@bffd758254f8:~/Chapter1# make
 g++ -c hello.cpp
 g++ -o hello hello.o
```

1.  现在我们有了可执行文件，我们将对其进行调试。从命令行输入`gdb hello`：

```cpp
root@bffd758254f8:~/Chapter1# gdb hello
 GNU gdb (Ubuntu 8.2.91.20190405-0ubuntu3) 8.2.91.20190405-git
 Copyright (C) 2019 Free Software Foundation, Inc.
 License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
 This is free software: you are free to change and redistribute it.
 There is NO WARRANTY, to the extent permitted by law.
 Type "show copying" and "show warranty" for details.
 This GDB was configured as "x86_64-linux-gnu".
 Type "show configuration" for configuration details.
 For bug reporting instructions, please see:
 <http://www.gnu.org/software/gdb/bugs/>.
 Find the GDB manual and other documentation resources online at:
 <http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
 Type "apropos word" to search for commands related to "word"...
 Reading symbols from hello...
 (No debugging symbols found in hello)
 (gdb)
```

1.  正如您所看到的，最后一行说（`hello`中未找到调试符号）。GDB 不需要调试符号来调试程序，因此我们必须告诉编译器在编译过程中包含调试符号。我们必须退出当前会话；要做到这一点，输入`q`（*Enter*）。然后，编辑 makefile，并在`g++`编译器部分的`hello.o`目标中添加`-g`选项：

```cpp
CC = g++
all: hello
hello: hello.o
    ${CC} -o hello hello.o
hello.o: hello.cpp
    $(CC) -c -g hello.cpp
clean:
    rm hello.o hello
```

1.  让我们再次运行它，但首先，我们必须用`make`命令重新构建应用程序：

```cpp
root@bcec6ff72b3c:/BOOK/chapter1# gdb hello
GNU gdb (Ubuntu 8.2.91.20190405-0ubuntu3) 8.2.91.20190405-git
Copyright (C) 2019 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
 <http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from hello...
(No debugging symbols found in hello)
(gdb)
```

我们已经准备好调试了。调试会话通常包括设置断点，观察变量的内容，设置监视点等。下一节将展示最常见的调试命令。

# 它是如何工作的...

在前一节中，我们已经看到了创建程序和 makefile 所需的步骤。在本节中，我们将学习如何调试我们开发的`Hello World!`程序。

让我们从可视化我们要调试的代码开始。我们通过运行`l`命令（缩写）来做到这一点：

```cpp
(gdb) l
 1 #include <iostream>
 2 int main()
 3 {
 4    int x = 10;
 5    x += 2;
 6    std::cout << "Hello World! x = " << x << std::endl;
 7    return 0;
 8 }
```

我们必须设置一个断点。要设置断点，我们运行`b 5`命令。这将在当前模块的代码行号`5`处设置一个断点：

```cpp
(gdb) b 5
 Breakpoint 1 at 0x1169: file hello.cpp, line 5.
 (gdb)
```

现在是运行程序的时候了。要运行程序，我们输入`r`命令。这将运行我们用 GDB 启动的`hello`程序：

```cpp
(gdb) r
 Starting program: /root/Chapter1/hello
```

一旦启动，GDB 将自动停在进程流程命中的任何断点处。在这种情况下，进程运行，然后停在`hello.cpp`文件的第`5`行：

```cpp
Breakpoint 1, main () at hello.cpp:5
 5 x += 2;
```

为了逐步进行，我们在 GDB 上运行`n`命令（即，跳过）。这会执行当前可视化的代码行。类似的命令是`s`（跳入）。如果当前命令是一个函数，它会跳入函数：

```cpp
(gdb) n
6 std::cout << "Hello World! x = " << x << std::endl;
the 'n' command (short for next) execute one line. Now we may want to check the content of the variable x after the increment:
```

如果我们需要知道变量的内容，我们运行`p`命令（缩写），它会打印变量的内容。在这种情况下，预期地，`x = 12`被打印出来：

```cpp
(gdb) p x
$1 = 12
```

现在，让我们运行程序直到结束（或者直到下一个断点，如果设置了）。这是用`c`命令（继续的缩写）完成的：

```cpp
(gdb) c 
 Continuing.
 Hello World! x = 12
 [Inferior 1 (process 101) exited normally]
 (gdb)
```

GDB 实际上充当解释器，让程序员逐行步进程序。这有助于开发人员解决问题，查看运行时变量的内容，更改变量的状态等。

# 还有更多...

GDB 有很多非常有用的命令。在接下来的章节中，将更多地探索 GDB。这里有四个更多的命令要展示：

1.  `s`：跳入的缩写。如果在一个方法上调用，它会跳入其中。

1.  `bt`：回溯的缩写。打印调用堆栈。

1.  `q`：退出的缩写。用于退出 GDB。

1.  `d`：删除的缩写。它删除一个断点。例如，`d 1`删除第一个设置的断点。

GNU GDB 项目的主页可以在这里找到：[`www.gnu.org/software/gdb`](https://www.gnu.org/software/gdb)。更详细的信息可以在`man dbg`的`man pages`和在线上找到。您也可以参考*Using GDB: A Guide to the GNU Source-Level Debugger,* by Richard M. Stallman and Roland H. Pesch*.*

# 学习 Linux 基础知识 - 进程和线程

进程和线程是任何操作系统的执行单元。在这个教程中，您将学习如何在 GNU/Linux 命令行上处理进程和线程。

在 Linux 中，进程由`sched.h`头文件中定义的`task_struct`结构定义。另一方面，线程由`thread_info.h`头文件中的`thread_info`结构定义。线程是主进程的一个可能的执行流。一个进程至少有一个线程（主线程）。进程的所有线程在系统上并发运行。

在 Linux 上需要记住的一点是，它不区分进程和线程。线程就像一个与其他一些进程共享一些资源的进程。因此，在 Linux 中，线程经常被称为**轻量级进程**（**LWP**）。

# 如何做...

在本节中，我们将逐步学习在 GNU/Linux 发行版上控制进程和线程的所有最常见命令：

1.  `ps`命令显示当前系统中的进程、属性和其他参数。

```cpp
root@5fd725701f0f:/# ps u
USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
root 1 0.0 0.1 4184 3396 pts/0 Ss 17:20 0:00 bash
root 18 0.0 0.1 5832 2856 pts/0 R+ 17:22 0:00 ps u
```

1.  获取有关进程（及其线程）的信息的另一种方法是查看`/process/PID`文件夹。该文件夹包含所有进程信息，进程的线程（以**进程标识符**（PID）的形式的子文件夹），内存等等：

```cpp
root@e9ebbdbe3899:/# ps aux
USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
root 1 0.0 0.1 4184 3344 pts/0 Ss 16:24 0:00 bash
root 149 0.0 0.1 4184 3348 pts/1 Ss 17:40 0:00 bash
root 172 85.0 0.0 5832 1708 pts/0 R+ 18:02 0:04 ./hello
root 173 0.0 0.1 5832 2804 pts/1 R+ 18:02 0:00 ps aux
root@e9ebbdbe3899:/# ll /proc/172/
total 0
dr-xr-xr-x 9 root root 0 May 12 18:02 ./
dr-xr-xr-x 200 root root 0 May 12 16:24 ../
dr-xr-xr-x 2 root root 0 May 12 18:02 attr/
-rw-r--r-- 1 root root 0 May 12 18:02 autogroup
-r-------- 1 root root 0 May 12 18:02 auxv
-r--r--r-- 1 root root 0 May 12 18:02 cgroup
--w------- 1 root root 0 May 12 18:02 clear_refs
-r--r--r-- 1 root root 0 May 12 18:02 cmdline
-rw-r--r-- 1 root root 0 May 12 18:02 comm
-rw-r--r-- 1 root root 0 May 12 18:02 coredump_filter
-r--r--r-- 1 root root 0 May 12 18:02 cpuset
lrwxrwxrwx 1 root root 0 May 12 18:02 cwd -> /root/Chapter1/
-r-------- 1 root root 0 May 12 18:02 environ
lrwxrwxrwx 1 root root 0 May 12 18:02 exe -> /root/Chapter1/hello*
dr-x------ 2 root root 0 May 12 18:02 fd/
dr-x------ 2 root root 0 May 12 18:02 fdinfo/
-rw-r--r-- 1 root root 0 May 12 18:02 gid_map
-r-------- 1 root root 0 May 12 18:02 io
-r--r--r-- 1 root root 0 May 12 18:02 limits
... 
```

1.  进程也可以被终止。从技术上讲，终止一个进程意味着停止它的执行：

```cpp
root@5fd725701f0f:/# kill -9 PID
```

该命令向具有 PID 的进程发送`kill`信号（`9`）。其他信号也可以发送给进程，例如`HUP`（挂起）和`INT`（中断）。

# 它是如何工作的...

在*步骤 1*中，对于每个进程，我们可以看到以下内容：

+   进程所属的用户

+   PID

+   特定时刻的 CPU 和内存百分比

+   当进程启动和运行时间

+   用于运行进程的命令

通过`ps aux`命令，我们可以获取`hello`进程的 PID，即`172`。现在我们可以查看`/proc/172`文件夹。

进程和线程是操作系统的构建模块。在本教程中，我们已经看到如何通过命令行与内核交互，以获取有关进程的信息（例如`ps`），并通过查看 Linux 在进程运行时更新的特定文件夹来获取信息。同样，每次我们调用命令（在这种情况下是为了获取有关进程的信息），命令必须进入内核空间以获取有效和更新的信息。

# 还有更多...

`ps`命令有比本教程中所见更多的参数。完整列表可在其 Linux man 页面`man ps`上找到。

作为`ps`的替代方案，一个更高级和交互式的命令是`top`命令，`man top`。

# 处理 Linux bash 错误

我们已经看到，通过 shell 是与 Linux 内核交互的一种方式，通过调用命令。命令可能会失败，正如我们可以想象的那样，而传达失败的一种方式是返回一个非负整数值。在大多数情况下，0 表示成功。本教程将向您展示如何处理 shell 上的错误处理。

# 如何做...

本节将向您展示如何直接从 shell 和通过脚本获取错误，这是脚本开发的一个基本方面：

1.  首先，运行以下命令：

```cpp
root@e9ebbdbe3899:/# cp file file2
 cp: cannot stat 'file': No such file or directory
 root@e9ebbdbe3899:/# echo $?
 1
```

1.  创建一个名为`first_script.sh`的新文件，并输入以下代码：

```cpp
#!/bin/bash
cat does_not_exists.txt
if [ $? -eq 0 ]
then
    echo "All good, does_not_exist.txt exists!"
    exit 0
else
    echo "does_not_exist.txt really DOES NOT exists!!" >&2
    exit 11
fi
```

1.  保存文件并退出（`:wq`或`:x`）。

1.  为`first_script.sh`文件为当前用户授予执行权限（`x`标志）：

```cpp
root@e9ebbdbe3899:~# chmod u+x first_script.sh
```

这些步骤在下一节中详细介绍。

# 它是如何工作的...

在*步骤 1*中，`cp`命令失败了，因为`file`和`file2`不存在。通过查询`echo $?`，我们得到了错误代码；在这种情况下，它是`1`。这在编写 bash 脚本时特别有用，因为我们可能需要检查特定条件。

在*步骤 2*中，脚本只是列出了`does_not_exist.txt`文件，并读取返回的错误代码。如果一切顺利，它会打印一个确认消息并返回`0`。否则，它会返回错误代码`11`。

通过运行脚本，我们得到以下输出：

![](img/5d809462-bb33-4827-9f73-a2cbe6881bbc.png)

在这里，我们注意到了一些事情：

+   我们记录了我们的错误字符串。

+   错误代码是我们在脚本中设置的。

在幕后，每次调用命令时，它都会进入内核空间。命令被执行，并以整数的形式将返回状态发送回用户。考虑这个返回状态非常重要，因为我们可能有一个命令，表面上成功了（没有输出），但最终失败了（返回的代码与`0`不同）。

# 还有更多...

命令的返回状态的一个重要方面是它可以用于（有条件地）运行下一个命令。为此目的使用了两个重要的运算符：`&&`（AND）和`||`（OR）。

在这两个命令中，第二个命令只有在第一个成功时才会运行（`&&`运算符）。如果`file.txt`被复制到项目文件夹中，它将被删除：

```cpp
cp file.txt ~/projects && rm -f file.txt
```

让我们看一个第二个例子：

```cpp
cp file.txt ~/projects || echo 'copy failed!'
```

在前面的示例中，第二个命令仅在第一个失败时运行（`||`运算符）。如果复制失败，则打印`copy failed!`。

在这个示例中，我们只是展示了如何在 shell 脚本中组合命令以创建更复杂的命令，并通过控制错误代码，我们可以控制执行流程。man 页面是一个很好的资源，因为它包含了所有的命令和错误代码（例如，`man cp`和`man cat`）。

# 处理 Linux 代码错误

这个示例代表了错误处理主题中的另一面：源代码级别的错误处理。Linux 通过命令以及编程 API 公开其内核特性。在这个示例中，我们将看到如何通过 C 程序处理错误代码和`errno`，以打开一个文件。

# 如何做...

在本节中，我们将看到如何在 C 程序中从系统调用中获取错误。为此，我们将创建一个程序来打开一个不存在的文件，并显示 Linux 返回的错误的详细信息：

1.  创建一个新文件：`open_file.c`。

1.  编辑新创建的文件中的以下代码：

```cpp
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int fileDesc = open("myFile.txt", O_RDONLY);
    if (fileDesc == -1)
    {
        fprintf(stderr, "Cannot open myFile.txt .. error: %d\n", 
           fileDesc);
        fprintf(stderr, "errno code = %d\n", errno);
        fprintf(stderr, "errno meaningn = %s\n", strerror(errno));
        exit(1);
    }
}
```

1.  保存文件并退出（`:x`）。

1.  编译代码：`gcc open_file.c`。

1.  前面的编译（不带参数）将产生一个名为`a.out`的二进制文件（这是 Linux 和 Unix 操作系统上的默认名称）。

# 工作原理...

列出的程序尝试以读取模式打开文件。错误将通过`fprintf`命令打印在标准错误上。运行后，输出如下：

![](img/ec4464be-3ce9-4b95-b6fe-e888f82b52dd.png)

有一些要点需要强调。该程序是通过严格遵循 open 系统调用的 man 页面（`man 2 open`）开发的：

```cpp
RETURN VALUES
     If successful, open() returns a non-negative integer, termed a 
file descriptor. It 
      returns -1 on failure, and sets errno to indicate the error
```

开发人员（在这种情况下是我们）检查了文件描述符是否为`-1`（通过`fprintf`确认），以打印`errno`（代码为`2`）。`errno 2`是什么意思？`strerror`对于这个目的非常有用，它可以将`errno`（这是晦涩的）翻译成程序员（或用户）能理解的内容。

# 还有更多...

在第二章中，*重新审视 C++*，我们将看到 C++如何通过提供更高级的机制、易于编写和更简洁的代码来帮助程序员。即使我们试图最小化直接与内核 API 的交互，而更倾向于使用 C++11-14-17 更高级的机制，也会有需要检查错误状态的情况。在这些情况下，您被邀请注意错误管理。
