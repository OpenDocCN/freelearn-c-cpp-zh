# 第二章：设置环境

要开始使用嵌入式系统，我们需要设置一个环境。与我们用于桌面开发的环境不同，嵌入式编程的环境需要两个系统：

+   **构建系统**：用于编写代码的系统

+   **目标系统**：您的代码将在其上运行的系统

在本章中，我们将学习如何设置这两个系统并将它们连接在一起。构建系统的配置可能会有很大的差异——可能有不同的操作系统、编译器和集成开发环境。目标系统配置的差异甚至更大，因为每个嵌入式系统都是独特的。此外，虽然您可以使用笔记本电脑或台式机作为构建系统，但您确实需要某种嵌入式板作为目标系统。

不可能涵盖所有可能的构建和目标系统的组合。相反，我们将学习如何使用一个流行的配置：

+   Ubuntu 18.04 作为构建系统

+   树莓派作为目标系统

我们将使用 Docker 在笔记本电脑或台式机上的虚拟环境中运行 Ubuntu。Docker 支持 Windows、macOS 和 Linux，但如果您已经使用 Linux，可以直接使用它，而无需在其上运行容器。

我们将使用**Quick EMUlator**（**QEMU**）来模拟树莓派板。这将教会我们如何在没有真实硬件访问权限的情况下为嵌入式板构建应用程序。在模拟环境中进行开发的初始阶段是常见的，在许多情况下，这是唯一可能的实际解决方案，因为在软件开发开始时，目标硬件可能不可用。

本章将涵盖以下主题：

+   在 Docker 容器中设置构建系统

+   使用模拟器

+   交叉编译

+   连接到嵌入式系统

+   调试嵌入式应用程序

+   使用 gdbserver 进行远程调试

+   使用 CMake 作为构建系统

# 在 Docker 容器中设置构建系统

在这个步骤中，我们将设置一个 Docker 容器，在您的台式机或笔记本电脑上运行 Ubuntu 18.04。无论您的机器上运行什么操作系统，Docker 都支持 Windows、macOS 和 Linux。作为这个步骤的结果，您将在主机操作系统中运行一个统一的、虚拟化的 Ubuntu Linux 构建系统。

如果您的操作系统已经运行 Ubuntu Linux，请随时跳到下一个步骤。

# 操作步骤如下...

我们将在笔记本电脑或台式机上安装 Docker 应用程序，然后使用 Ubuntu 的现成镜像在虚拟环境中运行这个操作系统：

1.  在您的网络浏览器中，打开以下链接并按照说明为您的操作系统设置 Docker：

1.  对于 Windows：[`docs.docker.com/docker-for-windows/install/`](https://docs.docker.com/docker-for-windows/install/)

1.  对于 macOS：[`docs.docker.com/docker-for-mac/install/`](https://docs.docker.com/docker-for-mac/install/)

1.  打开一个终端窗口（Windows 中的命令提示符，macOS 中的终端应用程序）并运行以下命令以检查是否已正确安装：

```cpp
 $ docker --version
```

1.  运行此命令使用 Ubuntu 镜像：

```cpp
$ docker pull ubuntu:bionic
```

1.  创建一个工作目录。在 macOS、Linux shell 或 Windows PowerShell 中运行以下命令：

```cpp
 $ mkdir ~/test 
```

1.  现在，在容器中运行下载的镜像：

```cpp
$ docker run -ti -v $HOME/test:/mnt ubuntu:bionic
```

1.  接下来，运行`uname -a`命令以获取有关系统的信息：

```cpp
# uname -a
```

您现在处于一个虚拟的 Linux 环境中，我们将在本书的后续步骤中使用它。

# 它是如何工作的...

在第一步中，我们安装了 Docker——一个虚拟化环境，允许在 Windows、macOS 或 Linux 上运行一个隔离的 Linux 操作系统。这是一种方便的方式，可以统一地封装所使用的任何操作系统所需的所有库和程序，以便分发和部署容器。

安装 Docker 后，运行一个快速命令来检查是否已正确安装：

![](img/79bd5436-f274-476e-b093-973afff3e233.png)

检查安装后，我们需要从 Docker 存储库中获取现成的 Ubuntu 镜像。Docker 镜像有标签；我们可以使用`bionic`标签来找到 Ubuntu 18.04 版本：

![](img/5d3e8e8d-05b8-4ae5-b232-5a85936edf50.png)

镜像下载需要时间。一旦镜像被获取，我们可以创建一个目录，用于开发。目录内容将在您的操作系统和在 Docker 中运行的 Linux 之间共享。这样，您可以使用您喜欢的文本编辑器来编写代码，但仍然可以使用 Linux 构建工具将代码编译成二进制可执行文件。

然后，我们可以使用第 4 步中获取的 Ubuntu 镜像启动 Docker 容器。`选项-v $HOME/test:/mnt`命令行使第 5 步中创建的文件夹对 Ubuntu 可见，作为`/mnt`目录。这意味着您在`~/test`目录中创建的所有文件都会自动出现在`/mnt`中。`-ti`选项使容器交互，让您访问 Linux shell 环境（bash）：

![](img/4702bc3f-9e14-4604-9fd1-aa4c28db0b03.png)

最后，我们对`.uname`容器进行了快速的健全性检查，它显示了有关 Linux 内核的信息，如下所示：

![](img/d836a658-c6fa-4c21-bfd5-ce05d6bb5e53.png)

尽管您的内核确切版本可能不同，但我们可以看到我们正在运行 Linux，我们的架构是`x86`。这意味着我们已经设置了我们的构建环境，我们将能够以统一的方式编译我们的代码，无论计算机上运行的操作系统是什么。但是，我们仍然无法运行编译后的代码，因为我们的目标架构是**Acorn RISC Machines**（**ARM**），而不是`x86`。我们将在下一个步骤中学习如何设置模拟的 ARM 环境。

# 还有更多...

Docker 是一个功能强大且灵活的系统。此外，其存储库包含许多包含对大多数开发人员有用的工具的现成镜像。

访问[`hub.docker.com/search?q=&type=image`](https://hub.docker.com/search?q=&type=image)并浏览最受欢迎的镜像。您还可以使用关键字搜索镜像，例如*嵌入式*。

# 使用模拟器

并非总是可能或实际使用真实的嵌入式板—硬件尚未准备好，或板的数量有限。模拟器帮助开发人员使用尽可能接近目标系统的环境，但不依赖于硬件可用性。这也是开始学习嵌入式开发的最佳方式。

在本教程中，我们将学习如何设置 QEMU（硬件模拟器）并配置它以模拟运行 Debian Linux 的基于 ARM 的嵌入式系统。

# 如何做...

我们需要一个虚拟环境，与 Docker 不同，它可以模拟具有与计算机架构不同的处理器的处理器：

1.  转到[`www.qemu.org/download/`](https://www.qemu.org/download/)，并单击与您的操作系统匹配的选项卡—Linux、macOS 或 Windows—，然后按照安装说明进行操作。

1.  创建一个测试目录，除非已经存在：

```cpp
 $ mkdir -p $HOME/raspberry
```

1.  下载以下文件并复制到您在上一步中创建的`~/raspberry`目录中：

+   **Raspbian Lite zip 存档**：[`downloads.raspberrypi.org/raspbian_lite/images/raspbian_lite-2019-07-12/2019-07-10-raspbian-buster-lite.zip`](http://downloads.raspberrypi.org/raspbian_lite/images/raspbian_lite-2019-07-12/2019-07-10-raspbian-buster-lite.zip)

+   **内核镜像**：[`github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/kernel-qemu-4.14.79-stretch`](https://github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/kernel-qemu-4.14.79-stretch)

+   **设备树 blob**：[`github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/versatile-pb.dtb`](https://github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/versatile-pb.dtb)

1.  将目录更改为`~/raspberry`并提取在上一步中下载的 Raspbian Lite zip 存档。它包含一个名为`2019-07-10-raspbian-buster-lite.img`的单个文件。

1.  打开一个终端窗口并运行 QEMU。对于 Windows 和 Linux，命令行如下：

```cpp
$ qemu-system-arm -M versatilepb -dtb versatile-pb.dtb -cpu arm1176 -kernel kernel-qemu-4.14.79-stretch -m 256 -drive file=2019-07-10-raspbian-buster-lite.img,format=raw -append "rw console=ttyAMA0 rootfstype=ext4 root=/dev/sda2 loglevel=8" -net user,hostfwd=tcp::22023-:22,hostfwd=tcp::9090-:9090 -net nic -serial stdio
```

1.  应该出现一个新窗口，显示 Linux 引导过程。几秒钟后，将显示登录提示。

1.  使用`pi`作为用户名和`raspberry`作为密码登录。然后，输入以下命令：

```cpp
 # uname -a
```

1.  检查命令的输出。它指示我们的系统架构是`ARM`，而不是`x86`。现在我们可以使用这个环境来测试为 ARM 平台构建的应用程序。

# 它是如何工作的...

在第一步中，我们安装了 QEMU 模拟器。没有可加载的代码映像，这个虚拟机没有太多用处。然后，我们可以获取运行 Linux 操作系统所需的三个映像：

+   **Linux 根文件系统**：包含 Raspbian Linux 的快照，用于树莓派设备

+   **Linux 内核**

+   **设备树 blob**：包含系统的硬件组件描述

一旦所有镜像都被获取并放入`~/raspberry`目录中，我们就运行 QEMU，提供镜像路径作为命令行参数。此外，我们配置虚拟网络，这使我们能够从本机环境连接到虚拟环境中运行的 Linux 系统。

QEMU 启动后，我们可以看到一个带有 Linux 登录提示的窗口：

![](img/1681dd1f-e0a3-43b8-a1aa-dbc92d5a9f73.png)

登录系统后，我们可以通过运行`uname`命令进行快速健全性检查：

![](img/23196296-b663-4e7f-9a88-e4f588196eb3.png)

类似于我们在上一个配方中运行的健全性检查，*在 Docker 容器中设置构建系统*，这表明我们正在运行 Linux 操作系统，但在这种情况下，我们可以看到目标架构是`ARM`。

# 还有更多...

QEMU 是一个强大的处理器模拟器，支持除 x86 和 ARM 之外的其他多种架构，如 PowerPC、SPARC64、SPARC32 和**无锁流水级阶段微处理器**（**MIPS**）。使其如此强大的一个方面是其灵活性，由于其许多配置选项。转到[`qemu.weilnetz.de/doc/qemu-doc.html`](https://qemu.weilnetz.de/doc/qemu-doc.html)根据您的需求配置 QEMU。

微控制器供应商通常也提供模拟器和仿真器。在开始为特定硬件进行开发时，请检查可用的仿真选项，因为这可能会显着影响开发时间和精力。

# 交叉编译

我们已经知道嵌入式开发环境由两个系统组成：构建系统，您在其中编写和构建代码，以及运行代码的主机系统。

我们现在有两个虚拟化环境：

+   在 Docker 容器中的 Ubuntu Linux，这将是我们的构建系统

+   运行 Raspbian Linux 的 QEMU，这将是我们的主机系统

1.  在这个配方中，我们将设置构建 Linux 应用程序所需的交叉编译工具，并构建一个简单的*Hello, world!*应用程序来测试设置。

# 做好准备

要设置交叉编译工具包，我们需要使用我们在*Docker 容器中设置构建系统*配方中设置的 Ubuntu Linux。

我们还需要`~/test`目录来在我们的操作系统和 Ubuntu 容器之间交换我们的源代码。

# 如何做...

让我们首先创建一个简单的 C++程序，我们希望为我们的目标平台进行编译：

1.  在`~/test`目录中创建一个名为`hello.cpp`的文件。

1.  使用您喜欢的文本编辑器将以下代码片段添加到其中：

```cpp
#include <iostream>

int main() {
 std::cout << "Hello, world!" << std::endl;
 return 0;
}
```

1.  现在我们有了`Hello, world!`程序的代码，我们需要编译它。

1.  切换到 Ubuntu（我们的构建系统）控制台。

1.  通过运行以下命令获取可用于安装的软件包的最新列表：

```cpp
# apt update -y
```

1.  从 Ubuntu 服务器获取软件包描述需要一些时间。运行以下命令安装交叉编译工具：

```cpp
 # apt install -y crossbuild-essential-armel
```

1.  您将看到一个要安装的包的长列表。按*Y*确认安装。作为健全性检查，运行一个没有参数的交叉编译器：

```cpp
# arm-linux-gnueabi-g++
```

1.  更改目录到`/mnt`

```cpp
# cd /mnt
```

1.  我们在第 1 步中创建的`hello.cpp`文件位于这里。现在让我们来构建它：

```cpp
 # arm-linux-gnueabi-g++ hello.cpp -o hello
```

1.  这个命令生成一个名为`hello`的可执行文件。您可能想知道为什么它没有任何扩展名。在 Unix 系统中，扩展名是完全可选的，二进制可执行文件通常没有任何扩展名。尝试运行文件。它应该会出现错误。

1.  让我们使用`file`工具生成关于可执行二进制文件的详细信息。

# 它是如何工作的...

在第一步中，我们创建了一个简单的*Hello, World!* C++程序。我们将其放入`~/test`目录中，这样它就可以从运行 Linux 的 Docker 容器中访问。

要构建源代码，我们切换到了 Ubuntu shell。

如果我们尝试运行标准的 Linux g++编译器来构建它，我们将得到一个用于构建平台的可执行文件，即 x86。然而，我们需要一个用于 ARM 平台的可执行文件。为了构建它，我们需要一个可以在 x86 上运行的编译器版本，构建 ARM 代码。

作为预备步骤，我们需要更新 Ubuntu 软件包分发中可用软件包的信息：

![](img/1973b89b-07cc-4125-a9fe-000c0969f05d.png)

我们可以通过运行`apt-get install crossbuild-essential-armel`来安装这个编译器以及一组相关工具：

![](img/dc1bdb2c-8f41-4ec3-9073-ffd6ff23487d.png)

在第 9 步进行的快速健全性检查表明它已正确安装：

![](img/2d99fc73-58e6-4651-8d87-e5bfe30c77e2.png)

现在，我们需要使用交叉编译器构建`hello.cpp`。它为 ARM 平台生成可执行文件，这就是为什么我们在第 12 步中尝试在构建系统中运行它失败的原因。

为了确保它确实是一个 ARM 可执行文件，我们需要运行`file`命令。其输出如下：

![](img/7fbe1b85-47b4-4f1c-aada-898888a43b66.png)

如您所见，该二进制文件是为 ARM 平台构建的，这就是为什么它无法在构建系统上运行的原因。

# 还有更多...

许多交叉编译工具包适用于各种架构。其中一些可以在 Ubuntu 存储库中轻松获得；一些可能需要手动安装。

# 连接到嵌入式系统

在使用交叉编译器在构建系统上构建嵌入式应用程序之后，应将其传输到目标系统。在基于 Linux 的嵌入式系统上，最好的方法是使用网络连接和远程 shell。**安全外壳**（**SSH**）由于其安全性和多功能性而被广泛使用。它不仅允许您在远程主机上运行 shell 命令，还允许您使用加密和基于密钥的身份验证从一台机器复制文件到另一台机器。

在这个教程中，我们将学习如何使用安全拷贝将应用程序二进制文件复制到模拟的 ARM 系统中，使用 SSH 连接到它，并在 SSH 中运行可执行文件。

# 准备就绪

我们将使用我们在*使用模拟器*教程中设置的树莓派模拟器作为目标系统。此外，我们需要我们的 Ubuntu 构建系统和我们在*交叉编译*教程中构建的可执行文件`hello`。

# 如何做...

我们将通过网络访问我们的目标系统。QEMU 为模拟机提供了一个虚拟网络接口，我们可以在不连接到真实网络的情况下使用它。为了这样做，我们需要找出一个要使用的 IP 地址，并确保 SSH 服务器在我们的虚拟环境中运行：

在您的本机操作系统环境中，找出您的机器的 IP 地址。打开一个终端窗口或 PowerShell。在 macOS 或 Linux 上运行`ifconfig`，或在 Windows 上运行`ipconfig`，并检查其输出。

在接下来的步骤中，我们将使用`192.168.1.5`作为模板 IP 地址；您需要用您的实际 IP 地址替换它。

1.  切换到树莓派模拟器并通过运行以下命令启用 SSH 服务：

```cpp
$ sudo systemctl start ssh
```

1.  切换到 Ubuntu 窗口并安装 SSH 客户端：

```cpp
# apt install -y ssh
```

1.  现在，我们可以将`hello`可执行文件复制到目标系统：

```cpp
# scp -P22023 /mnt/hello pi@192.168.1.5:~
```

1.  当要求输入密码时，输入`raspberry`。切换回树莓派模拟器窗口。检查我们刚刚复制的可执行文件是否存在：

```cpp
$ ls hello
hello
```

1.  现在，运行程序：

```cpp
$ ./hello
```

正如我们所看到的，程序现在按预期运行。

# 工作原理...

在这个示例中，我们使用 SSH 在两个虚拟环境——Docker 和 QEMU——之间建立了数据交换。为此，我们需要在目标系统（QEMU）上运行并接受连接的 SSH 服务器，并在构建系统上启动连接的 SSH 客户端。

在第 2 步中，我们在构建系统上设置了 SSH 客户端。我们的目标系统在 QEMU 中运行，已经启动并运行了 SSH 服务器。在*使用模拟器*的步骤中，我们配置了 QEMU 以将主机端口`22023`转发到虚拟机端口`22`，即 SSH。

现在，我们可以使用`scp`通过安全网络连接将文件从构建系统复制到目标系统。我们可以指定我们的系统 IP 地址（在第 1 步中发现）和端口`22023`，作为`scp`连接的参数，以连接到：

![](img/bc75eea7-7ccb-43ea-9e30-f3f3857a007a.png)

在我们复制文件之后，我们可以使用相同的 IP 地址、端口和用户名通过 SSH 登录到目标系统。它会打开一个类似于本地控制台的登录提示，并在授权后，我们会得到与本地终端相同的命令 shell。

我们在上一步中复制的`hello`应用程序应该在`home`目录中可用。我们通过运行`ls`命令在第 5 步中检查了这一点。

最后，我们可以运行应用程序：

![](img/087a7c0c-6ec1-4a7f-bcba-097249ccd47a.png)

当我们尝试在构建系统上运行它时，我们收到了一个错误。现在，输出是`Hello, world!`。这是我们所期望的，因为我们的应用程序是为 ARM 平台构建并在 ARM 平台上运行的。

# 还有更多...

尽管我们运行了连接到模拟系统的示例，但相同的步骤也适用于真实的嵌入式系统。即使目标系统没有显示器，也可以使用串行控制台连接设置 SSH。

在这个示例中，我们只是将文件复制到目标系统。除了复制，通常还会打开一个交互式 SSH 会话到嵌入式系统。通常，这比串行控制台更有效、更方便。它的建立方式与`scp`类似：

```cpp
# ssh pi@192.168.1.5 -p22023
```

SSH 提供各种身份验证机制。一旦启用并设置了公钥身份验证，就无需为每次复制或登录输入密码。这使得开发过程对开发人员来说更快速、更方便。

要了解更多关于 ss 密钥的信息，请访问[`www.ssh.com/ssh/key/`](https://www.ssh.com/ssh/key/)。

# 调试嵌入式应用程序

调试嵌入式应用程序在很大程度上取决于目标嵌入式系统的类型。微控制器制造商通常为他们的**微控制器单元**（**MCU**）提供专门的调试器，以及使用**联合测试动作组**（**JTAG**）协议进行远程调试的硬件支持。它允许开发人员在 MCU 开始执行指令后立即调试微控制器代码。

如果目标板运行 Linux，则调试的最实用方法是使用广泛的调试输出，并使用 GDB 作为交互式调试器。

在这个示例中，我们将学习如何在命令行调试器 GDB 中运行我们的应用程序。

# 准备就绪

我们已经学会了如何将可执行文件传输到目标系统。我们将使用*连接到嵌入式系统*的示例作为学习如何在目标系统上使用调试器的起点。

# 如何做...

我们已经学会了如何将应用程序复制到目标系统并在那里运行。现在，让我们学习如何在目标系统上使用 GDB 开始调试应用程序。在这个配方中，我们只会学习如何调用调试器并在调试器环境中运行应用程序。这将作为以后更高级和实用的调试技术的基础：

1.  切换到`QEMU`窗口。

1.  如果您还没有这样做，请使用`pi`作为用户名和`raspberry`作为密码登录。

1.  运行以下命令：

```cpp
$ gdb ./hello
```

1.  这将打开`gdb`命令行。

1.  输入`run`来运行应用程序：

```cpp
(gdb) run
```

1.  您应该在输出中看到`Hello, world`。

1.  现在，运行`quit`命令，或者只需输入`q`：

```cpp
(gdb) q
```

这将终止调试会话并将我们返回到 Linux shell。

# 工作原理...

我们用于仿真的 Raspberry Pi 映像预先安装了 GNU 调试器，因此我们可以立即使用它。

在`home`用户目录中，我们应该找到`hello`可执行文件，这是作为*连接到嵌入式系统*配方的一部分从我们的构建系统复制过来的。

我们运行`gdb`，将`hello`可执行文件的路径作为参数传递。这个命令打开了`gdb` shell，但并没有运行应用程序本身。要运行它，我们输入`run`命令：

![](img/7730a5e6-f9ba-4ae2-99c8-d775757b5df3.png)

应用程序运行，在屏幕上打印`Hello world!`消息，然后终止。但是，我们仍然在调试器中。要退出调试器，我们输入`quit`命令：

![](img/4c14581e-4665-4e97-9cc1-b58a262cd717.png)

您可以看到命令行提示已经改变。这表明我们不再处于`gdb`环境中。我们已经返回到 Raspberry Pi Linux 的默认 shell 环境，这是我们在运行 GDB 之前使用的环境。

# 还有更多...

在这种情况下，GNU 调试器是预先安装的，但可能不在您的真实目标系统中。如果它是基于 Debian 的，您可以通过运行以下命令来安装它：

```cpp
# apt install gdb gdb-multiarch
```

在其他基于 Linux 的系统中，需要不同的命令来安装 GDB。在许多情况下，您需要从源代码构建并手动安装它，类似于我们在本章的配方中构建和测试的`hello`应用程序。

在这个配方中，我们只学会了如何使用 GDB 运行应用程序，GDB 是一个具有许多命令、技术和最佳实践的复杂工具。我们将在第五章中讨论其中一些。

# 使用 gdbserver 进行远程调试

正如我们所讨论的，嵌入式开发环境通常涉及两个系统 - 构建系统和目标系统（或仿真器）。有时，由于远程通信的高延迟，目标系统上的交互式调试是不切实际的。

在这种情况下，开发人员可以使用 GDB 提供的远程调试支持。在这种设置中，使用**gdbserver**在目标系统上启动嵌入式应用程序。开发人员在构建系统上运行 GDB，并通过网络连接到 gdbserver。

在这个配方中，我们将学习如何使用 GDB 和 gdbserver 开始调试应用程序。

# 准备就绪

在*连接到嵌入式系统*配方中，我们学会了如何使我们的应用程序在目标系统上可用。我们将以此配方为起点，学习远程调试技术。

# 如何做...

我们将安装并运行 gdbserver 应用程序，这将允许我们在构建系统上运行 GDB 并将所有命令转发到目标系统。切换到 Raspberry Pi 仿真器窗口。

1.  以`pi`身份登录，密码为`raspberry`，除非您已经登录。

1.  要安装 gdbserver，请运行以下命令：

```cpp
 # sudo apt-get install gdbserver
```

1.  在`gdbserver`下运行`hello`应用程序：

```cpp
 $ gdbserver 0.0.0.0:9090 ./hello
```

1.  切换到构建系统终端并将目录更改为`/mnt/hello`：

```cpp
 # cd /mnt/hello
```

1.  安装`gdb-multiarch`软件包，它提供了对 ARM 平台的必要支持：

```cpp
 # apt install -y gdb-multiarch
```

1.  接下来，运行`gdb`：

```cpp
 # gdb-multiarch -q ./hello
```

1.  通过在`gdb`命令行中输入以下命令来配置远程连接（确保您用实际 IP 地址替换`192.168.1.5`）：

```cpp
 target remote 192.168.1.5:9090
```

1.  输入以下命令：

```cpp
 continue
```

程序现在将运行。

# 它是如何工作的...

在我们使用的 Raspberry Pi 镜像中，默认情况下未安装`gdbserver`。因此，作为第一步，我们安装`gdbserver`：

![](img/09463e0a-5205-4696-9876-eccdbfae92b5.png)

安装完成后，我们运行`gdbserver`，将需要调试的应用程序的名称、IP 地址和要监听传入连接的端口作为参数传递给它。我们使用`0.0.0.0`作为 IP 地址，表示我们希望接受任何 IP 地址上的连接：

![](img/0dbebcae-bc8b-4787-8b91-684276dde525.png)

然后，我们切换到我们的构建系统并在那里运行`gdb`。但是，我们不直接在 GDB 中运行应用程序，而是指示`gdb`使用提供的 IP 地址和端口启动与远程主机的连接：

![](img/4a3f076e-ed50-4130-8ebf-c13b191c319b.png)

之后，您在`gdb`提示符下键入的所有命令都将传输到 gdbserver 并在那里执行。当我们运行应用程序时，即使我们运行 ARM 可执行文件，我们也将在构建系统的`gdb`控制台中看到生成的输出：

![](img/c6f43a8b-2667-4422-ad15-e67e910f560f.png)

解释很简单——二进制文件在远程 ARM 系统上运行：我们的 Raspberry Pi 模拟器。这是一种方便的调试应用程序的方式，允许您保持在构建系统更舒适的环境中。

# 还有更多...

确保您使用的 GDB 和 gdbserver 的版本匹配，否则它们之间可能会出现通信问题。

# 使用 CMake 作为构建系统

在以前的示例中，我们学习了如何编译由一个 C++文件组成的程序。然而，真实的应用程序通常具有更复杂的结构。它们可以包含多个源文件，依赖于其他库，并被分割成独立的项目。

我们需要一种方便地为任何类型的应用程序定义构建规则的方法。CMake 是最知名和广泛使用的工具之一，它允许开发人员定义高级规则并将它们转换为较低级别的构建系统，如 Unix make。

在本示例中，我们将学习如何设置 CMake 并为我们的*Hello, world!*应用程序创建一个简单的项目定义。

# 准备工作

如前所述，常见的嵌入式开发工作流程包括两个环境：构建系统和目标系统。CMake 是构建系统的一部分。我们将使用 Ubuntu 构建系统作为起点，该系统是作为*在 Docker 容器中设置构建系统*配方的结果创建的。

# 如何做...

1.  我们的构建系统尚未安装 CMake。要安装它，请运行以下命令：

```cpp
 # apt install -y cmake
```

1.  切换回本机操作系统环境。

1.  在`~/test`目录中，创建一个子目录`hello`。使用您喜欢的文本编辑器在`hello`子目录中创建一个名为`CMakeLists.txt`的文件。

1.  输入以下行：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(hello)
add_executable(hello hello.cpp)
```

1.  保存文件并切换到 Ubuntu 控制台。

1.  切换到`hello`目录：

```cpp
# cd /mnt/hello
```

1.  运行 CMake：

```cpp
 # mkdir build && cd build && cmake ..
```

1.  现在，通过运行以下命令构建应用程序：

```cpp
# make
```

1.  使用`file`命令获取有关生成的可执行二进制文件的信息：

```cpp
# file hello
```

1.  如您所见，构建是本地的 x86 平台。我们需要添加交叉编译支持。切换回文本编辑器，打开`CMakeLists.txt`，并添加以下行：

```cpp
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

1.  保存并切换到 Ubuntu 终端。

1.  再次运行`cmake`命令以重新生成构建文件：

```cpp
# cmake ..
```

1.  通过运行`make`来构建代码：

```cpp
# make
```

1.  再次检查生成的输出文件的类型：

```cpp
# file hello
```

现在，我们使用 CMake 为我们的目标系统构建了一个可执行文件。

# 它是如何工作的...

首先，我们将 CMake 安装到我们的构建系统中。安装完成后，我们切换到本机环境创建`CMakeLists.txt`。这个文件包含关于项目组成和属性的高级构建指令。

我们将项目命名为*hello*，它从名为`hello.cpp`的源文件创建一个名为`hello`的可执行文件。此外，我们指定了构建我们的应用程序所需的 CMake 的最低版本。

创建了项目定义之后，我们可以切换回构建系统 shell，并通过运行`make`生成低级构建指令。

创建一个专用的构建目录来保存所有构建产物是一种常见的做法。通过这样做，编译器生成的目标文件或 CMake 生成的文件不会污染源代码目录。

在一个命令行中，我们创建一个`build`目录，切换到新创建的目录，并运行 CMake。

我们将父目录作为参数传递，让 CMake 知道在哪里查找`CMakeListst.txt`：

![](img/14a5bf44-2185-478b-a144-94f6030053c3.png)

默认情况下，CMake 为传统的 Unix `make`实用程序生成`Makefile`文件。我们运行`make`来实际构建应用程序：

![](img/f5983eee-a06d-4d26-9441-8274013cf7be.png)

它可以工作，但会导致为 x86 平台构建的可执行二进制文件，而我们的目标系统是 ARM：

![](img/2f2aedf6-d441-4e80-a11a-4778c32d273e.png)

为了解决这个问题，我们在我们的`CMakeLists.txt`文件中添加了几个选项来配置交叉编译。再次重复构建步骤，我们得到了一个新的`hello`二进制文件，现在是为 ARM 平台而构建的：

![](img/cfdf7699-32f3-41bf-a3fe-54adbf11e341.png)

正如我们在`file`命令的输出中所看到的，我们已经为 ARM 平台构建了可执行文件，而不是 x86，我们用作构建平台。这意味着这个程序将无法在构建机器上运行，但可以成功地复制到我们的目标平台并在那里运行。

# 还有更多...

配置 CMake 进行交叉编译的最佳方法是使用所谓的**工具链**文件。工具链文件定义了特定目标平台的构建规则的所有设置和参数，例如编译器前缀、编译标志以及目标平台上预先构建的库的位置。通过使用不同的工具链文件，可以为不同的目标平台重新构建应用程序。有关更多详细信息，请参阅 CMake 工具链文档[`cmake.org/cmake/help/v3.6/manual/cmake-toolchains.7.html`](https://cmake.org/cmake/help/v3.6/manual/cmake-toolchains.7.html)。
