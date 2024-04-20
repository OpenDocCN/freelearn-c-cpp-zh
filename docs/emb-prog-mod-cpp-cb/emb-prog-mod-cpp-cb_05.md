# 第五章：调试、日志记录和性能分析

调试和性能分析是任何类型应用程序开发工作流程中的重要部分。在嵌入式环境中，这些任务需要开发人员特别注意。嵌入式应用程序在可能与开发人员工作站非常不同的系统上运行，并且通常具有有限的资源和用户界面功能。

开发人员应该提前计划如何在开发阶段调试他们的应用程序，以及如何确定生产环境中问题的根本原因，并加以修复。

通常，解决方案是使用目标设备的仿真器以及嵌入式系统供应商提供的交互式调试器。然而，对于更复杂的系统，完整和准确的仿真几乎是不可行的，远程调试是最可行的解决方案。

在许多情况下，使用交互式调试器是不可能或根本不切实际的。程序在断点停止后几毫秒内硬件状态可能会发生变化，开发人员没有足够的时间来分析它。在这种情况下，开发人员必须使用广泛的日志记录进行根本原因分析。

在本章中，我们将重点介绍基于**SoC**（**片上系统**）和运行 Linux 操作系统的更强大系统的调试方法。我们将涵盖以下主题：

+   在**GDB**（GNU 项目调试器的缩写）中运行您的应用程序

+   使用断点

+   处理核心转储

+   使用 gdbserver 进行调试

+   添加调试日志

+   使用调试和发布版本

这些基本的调试技术将在本书中以及在您处理任何类型嵌入式应用程序的工作中有很大帮助。

# 技术要求

在本章中，我们将学习如何在**ARM**（**Acorn RISC Machines**的缩写）平台仿真器中调试嵌入式应用程序。此时，您应该已经在笔记本电脑或台式电脑上运行的虚拟化 Linux 环境中配置了两个系统：

+   Ubuntu Linux 作为构建系统在 Docker 容器中

+   Debian Linux 作为目标系统在**QEMU**（**快速仿真器**）ARM 仿真器中

要了解交叉编译的理论并设置开发环境，请参考第二章中的示例，*设置环境*。

# 在 GDB 中运行您的应用程序

在这个示例中，我们将学习如何在目标系统上使用调试器运行一个示例应用程序，以及尝试一些基本的调试技术。

GDB 是一个开源且广泛使用的交互式调试器。与大多数作为**集成开发环境**（**IDE**）产品的一部分提供的调试器不同，GDB 是一个独立的命令行调试器。这意味着它不依赖于任何特定的 IDE。正如您在示例中所看到的，您可以使用纯文本编辑器来处理应用程序的代码，同时仍然能够进行交互式调试，使用断点，查看变量和堆栈跟踪的内容，以及更多。

GDB 的用户界面是极简的。您可以像在 Linux 控制台上工作一样运行它——通过输入命令并分析它们的输出。这种简单性使其非常适合嵌入式项目。它可以在没有图形子系统的系统上运行。如果目标系统只能通过串行连接或 ssh shell 访问，它尤其方便。由于它没有花哨的用户界面，它可以在资源有限的系统上运行。

在这个示例中，我们将使用一个人工样本应用程序，它会因异常而崩溃。它不会记录任何有用的信息，异常消息太模糊，无法确定崩溃的根本原因。我们将使用 GDB 来确定问题的根本原因。

# 如何做...

我们现在将创建一个在特定条件下崩溃的简单应用程序：

1.  在您的工作目录`~/test`中，创建一个名为`loop`的子目录。

1.  使用您喜欢的文本编辑器在`loop`子目录中创建一个名为`loop.cpp`的文件。

1.  让我们将一些代码放入`loop.cpp`文件中。我们从包含开始：

```cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
```

1.  现在，我们定义程序将包含的三个函数。第一个是`runner`：

```cpp
void runner(std::chrono::milliseconds limit,
            std::function<void(int)> fn,
            int value) {
  auto start = std::chrono::system_clock::now();
  fn(value);
  auto end = std::chrono::system_clock::now();
  std::chrono::milliseconds delta =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (delta > limit) {
    throw std::runtime_error("Time limit exceeded");
  }
  }
```

1.  第二个函数是`delay_ms`：

```cpp
void delay_ms(int count) {
  for (int i = 0; i < count; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1050));
  }
  }
```

1.  最后，我们添加入口函数`main`：

```cpp
int main() {
  int max_delay = 10;
  for (int i = 0; i < max_delay; i++) {
    runner(std::chrono::milliseconds(max_delay), delay_ms, i);
  }
  return 0;
  }
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(loop)
add_executable(loop loop.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "-g --std=c++11")

set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

1.  现在，切换到构建系统终端，并通过运行以下命令将当前目录更改为`/mnt/loop`。

```cpp
$ cd /mnt/loop
```

1.  按照以下方式构建应用程序：

```cpp
$ cmake . && make
```

1.  切换回您的本机环境，在`loop`子目录中找到`loop`输出文件，并通过 ssh 将其复制到目标系统。使用用户帐户。切换到目标系统终端。根据需要使用用户凭据登录。现在，使用`gdb`运行`loop`可执行二进制文件：

```cpp
$ gdb ./loop
```

1.  调试器已启动，并显示命令行提示（`gdb`）。要运行应用程序，请键入`run`命令：

```cpp
(gdb) run
```

1.  您可以看到应用程序由于运行时异常而异常终止。异常消息`Time limit exceeded`给了我们一个线索，但并没有指出发生异常的具体条件。让我们试着确定这一点。首先，让我们检查崩溃应用程序的堆栈跟踪：

```cpp
(gdb) bt
```

1.  这显示了从顶级函数`main`到库函数`__GI_abort`的七个堆栈帧，后者实际上终止了应用程序。正如我们所看到的，只有帧`7`和`6`属于我们的应用程序，因为只有它们在`loop.cpp`中定义。让我们仔细看一下`frame 6`，因为这是抛出异常的函数：

```cpp
(gdb) frame 6
```

1.  运行`list`命令来查看附近的代码：

```cpp
(gdb) list
```

1.  正如我们所看到的，如果 delta 变量的值超过 limit 变量的值，就会抛出异常。但是这些值是什么？运行`info locals`命令来找出这一点：

```cpp
(gdb) info locals
```

1.  我们无法在这里看到限制变量的值。使用`info args`命令来查看它：

```cpp
(gdb) info args
```

1.  现在，我们可以看到限制是`10`，而 delta 是`11`。当使用`fn`参数设置为`delay_ms`函数，并且`value`参数的值设置为`7`时，崩溃发生。

# 它是如何工作的...

该应用程序是故意创建的，在某些条件下会崩溃，并且没有提供足够的信息来确定这些条件。该应用程序由两个主要函数组成——`runner`和`delay_ms`。

`runner`函数接受三个参数——时间限制、一个参数的函数和函数参数值。它运行作为参数提供的函数，传递值，并测量经过的时间。如果时间超过时间限制，它会抛出异常。

`delay_ms`函数执行延迟。但是，它的实现是错误的，它将每毫秒视为由 1100 微秒而不是 1000 微秒组成。

`main`函数在`loop`目录中运行 runner，提供 10 毫秒作为时间限制的修复值和`delay_ms`作为要运行的函数，但增加`value`参数的值。在某个时候，`delay_ms`函数超过了时间限制，应用程序崩溃了。

首先，我们为 ARM 平台构建应用程序，并将其传输到模拟器上运行：

![](img/52c9bbb6-e17c-4961-a4da-0e45d3a73859.png)

重要的是要向编译器传递`-g`参数。此参数指示编译器向生成的二进制文件添加调试符号。我们将其添加到`CMakeLists.txt`文件中的`CMAKE_CXX_FLAGS`参数中，如下所示：

```cpp
SET(CMAKE_CXX_FLAGS "-g --std=c++11")
```

现在，我们运行调试器，并将应用程序可执行文件名作为其参数：

![](img/e3b7393c-2d4c-4b8d-a1dd-1f5ea2c2ced8.png)

应用程序不会立即运行。我们使用`run` GDB 命令启动它，并观察它在短时间内崩溃：

![](img/66aacaec-d965-40bc-bac7-e0de3e630951.png)

接下来，我们使用`backtrace`命令来查看堆栈跟踪：

![](img/b394f587-a045-48f3-8c60-299f1cbb9fc7.png)

对堆栈跟踪的分析显示`frame 6`应该给我们更多信息来揭示根本原因。通过接下来的步骤，我们切换到`frame 6`并审查相关的代码片段：

![](img/ee18e675-c717-4500-8d75-c88fd036ad38.png)

接下来，我们分析本地变量和函数参数的值，以确定它们与时间限制的关系：

![](img/a37ff8ac-9ac1-4185-9e4c-d641e7c05ff4.png)

我们确定当传递给`delay_ms`的值达到`7`时发生崩溃，而不是预期的`11`，这在正确实现延迟的情况下是预期的。

# 还有更多...

GDB 命令通常接受多个参数来微调它们的行为。使用`help`命令来了解每个命令的更多信息。例如，这是`help bt`命令的输出：

![](img/96d294ad-4432-4294-bd38-b8480c1ccec6.png)

这显示了用于审查和分析堆栈跟踪的`bt`命令的信息。类似地，您可以获取关于 GDB 支持的所有其他命令的信息。

# 使用断点

在这个教程中，我们将学习在使用 GDB 时更高级的调试技术。我们将使用相同的示例应用程序，并使用断点来找到实际延迟与`delay_ms`参数值的依赖关系。

在 GDB 中使用断点与在集成 IDE 中使用断点类似，唯一的区别是开发人员不是使用内置编辑器来导航源代码，而是要学会显式使用行号、文件名或函数名。

这比点击运行调试器不太方便，但是灵活性使开发人员能够创建强大的调试场景。在这个教程中，我们将学习如何在 GDB 中使用断点。

# 如何做到...

在这个教程中，我们将使用与第一个教程相同的环境和相同的测试应用程序。参考第 1 到 9 步的*在 GDB 中运行您的应用程序*教程来构建应用程序并将其复制到目标系统上：

1.  我们想要调试我们的`runner`函数。让我们看一下它的内容。在 gdb shell 中，运行以下程序：

```cpp
(gdb) list runner,delay_ms
```

1.  我们想要看到每次迭代中`delta`的变化。让我们在该行设置一个断点：

```cpp
14 if (delta > limit) {
```

1.  使用`break 14`命令在第 14 行设置一个断点：

```cpp
(gdb) break 14
```

1.  现在运行程序：

```cpp
(gdb) run
```

1.  检查`delta`的值：

```cpp
(gdb) print delta 
$1 = {__r = 0}
```

1.  继续执行程序，输入`continue`或者`c`：

```cpp
(gdb) c
```

1.  再次检查`delta`的值：

```cpp
(gdb) print delta
```

1.  正如我们预期的那样，`delta`的值在每次迭代中都会增加，因为`delay_ms`需要越来越多的时间。

1.  每次运行`print delta`都不方便。让我们使用名为`command`的命令来自动化它：

```cpp
(gdb) command
```

1.  再次运行`c`。现在，每次停止后都会显示`delta`的值：

```cpp
(gdb) c
```

1.  然而，输出太冗长了。让我们通过再次输入`command`并编写以下指令来使 GDB 输出静音。现在，运行`c`或`continue`命令几次以查看差异：

```cpp
(gdb) command
Type commands for breakpoint(s) 1, one per line.
End with a line saying just "end".
>silent
>print delta
>end
(gdb) c
```

1.  我们可以使用`printf`命令使输出更加简洁，如下所示：

```cpp
(gdb) command
Type commands for breakpoint(s) 1, one per line.
End with a line saying just "end".
>silent
>printf "delta=%d, expected=%d\n", delta.__r, value
>end
(gdb) c
```

现在，我们可以看到两个值，计算出的延迟和预期的延迟，并且可以看到它们随时间的变化而发散。

# 它是如何工作的...

在这个教程中，我们想要设置一个断点来调试`runner`函数。由于 GDB 没有内置编辑器，我们需要知道设置断点的行号。虽然我们可以直接从文本编辑器中获取它，但另一种方法是在 GDB 中查看相关代码片段。我们使用带有两个参数的`gdb`命令列表 - 函数名称，以显示`runner`函数的第一行和`delay_ms`函数的第一行之间的代码行。这有效地显示了函数`runner`的内容：

![](img/482a8b75-c389-4e61-b14d-de3f5cd496b9.png)

在*步骤 4*，使用`break 14`命令在第 14 行设置断点，并运行程序。执行将在断点处停止：

![](img/c22a841e-1537-4806-b134-47c6733547bc.png)

我们使用`print`命令检查`delta`变量的值，并使用`continue`命令继续执行程序，由于在循环中调用了`runner`函数，它再次停在相同的断点处：

![](img/3e1b1f31-812a-4201-8a63-1866a3febe7a.png)

接下来，我们尝试更高级的技术。我们定义一组 GDB 命令，以在触发断点时执行。我们从一个简单的`print`命令开始。现在，每次我们继续执行，我们都可以看到`delta`变量的值：

![](img/55443211-11e7-43c4-a6c2-6a135a946c64.png)

接下来，我们使用`silent`命令禁用辅助 GDB 输出，以使输出更加简洁：

![](img/7175eb69-bc52-4946-b777-6277cae952a9.png)

最后，我们使用`printf`命令格式化具有两个最有趣变量的消息：

![](img/7b3fe1fb-b927-4590-9e68-2172dde5d955.png)

正如你所看到的，GDB 为开发人员提供了很多灵活性，使得即使缺乏图形界面，调试也变得更加舒适。

# 还有更多...

重要的是要记住，优化选项`-O2`和`-O3`可能导致编译器完全消除一些代码行。如果将断点设置为这些行，这些断点将永远不会触发。为避免这种情况，关闭调试构建的编译器优化。

# 处理核心转储

在第一个教程中，我们学习了如何使用交互式命令行调试器确定崩溃应用程序的根本原因。但是，在生产环境中，应用程序崩溃时，有时无法或不切实际地在测试系统上重现相同的问题，从 GDB 中运行应用程序。 

Linux 提供了一种机制，可帮助分析崩溃的应用程序，即使它们不是直接从 GDB 中运行的。当应用程序异常终止时，操作系统将其内存映像保存到名为`core`的文件中。在本教程中，我们将学习如何配置 Linux 以生成崩溃应用程序的核心转储，以及如何使用 GDB 进行分析。

# 如何做...

我们将确定一个应用程序崩溃的根本原因，该应用程序未在 GDB 中运行：

1.  在本教程中，我们将使用与第一个教程中相同的环境和相同的测试应用程序。请参阅第一个教程的*步骤 1*至*7*，构建应用程序并将其复制到目标系统。

1.  首先，我们需要启用生成崩溃应用程序的核心转储。在大多数 Linux 发行版中，默认情况下关闭此功能。运行`ulimit -c`命令检查当前状态：

```cpp
$ ulimit -c
```

1.  前一个命令报告的值是要生成的核心转储的最大大小。零表示没有核心转储。要增加限制，我们需要首先获得超级用户权限。运行`su -`命令。提示输入`Password`时，输入`root`：

```cpp
$ su -
Password:
```

1.  运行`ulimit -c unlimited`命令允许任意大小的核心转储：

```cpp
# ulimit -c unlimited
```

1.  现在，通过按*Ctrl* + *D*或运行`logout`命令退出 root shell。

1.  前面的命令仅为超级用户更改了核心转储限制。要将其应用于当前用户，请在用户 shell 中再次运行相同的命令：

```cpp
$ ulimit -c unlimited
```

1.  确保限制已更改：

```cpp
$ ulimit -c
unlimited
```

1.  现在，像往常一样运行应用程序：

```cpp
$ ./loop 
```

1.  它将以异常崩溃。运行`ls`命令检查当前目录中是否创建了核心文件：

```cpp
$ ls -l core
-rw------- 1 dev dev 536576 May 31 00:54 core
```

1.  现在，运行`gdb`，传递可执行文件和`core`文件作为参数：

```cpp
$ gdb ./loop core
```

1.  在 GDB shell 中，运行`bt`命令查看堆栈跟踪：

```cpp
(gdb) bt
```

1.  您可以看到与从`gdb`内部运行的应用程序相同的堆栈跟踪。但是，在这种情况下，我们看到了核心转储的堆栈跟踪。

1.  在这一点上，我们可以使用与第一个教程中相同的调试技术来缩小崩溃原因。

# 它是如何工作的...

核心转储功能是 Linux 和其他类 Unix 操作系统的标准功能。然而，在每种情况下都创建核心文件并不实际。由于核心文件是进程内存的快照，它们可能在文件系统上占用几兆甚至几十几个 G 的空间。在许多情况下，这是不可接受的。

开发人员需要明确指定操作系统允许生成的核心文件的最大大小。这个限制，以及其他限制，可以使用`ulimit`命令来设置。

我们运行`ulimit`两次，首先为超级用户 root 移除限制，然后为普通用户/开发人员移除限制。需要两阶段的过程，因为普通用户的限制不能超过超级用户的限制。

在我们移除了核心文件大小的限制后，我们在没有 GDB 的情况下运行我们的测试应用程序。预期地，它崩溃了。崩溃后，我们可以看到当前目录中创建了一个名为`core`的新文件。

当我们运行我们的应用程序时，它崩溃了。通常情况下，我们无法追踪崩溃的根本原因。然而，由于我们启用了核心转储，操作系统自动为我们创建了一个名为`core`的文件：

![](img/a897ffd1-0aa8-4f4d-b1d3-a9e2941e9e77.png)

核心文件是所有进程内存的二进制转储，但没有额外的工具很难分析它。幸运的是，GDB 提供了必要的支持。

我们运行 GDB 传递两个参数——可执行文件的路径和核心文件的路径。在这种模式下，我们不从 GDB 内部运行应用程序。我们已经在核心转储中冻结了应用程序在崩溃时的状态。GDB 使用可执行文件将`core`文件中的内存地址绑定到函数和变量名：

![](img/8f81cb30-8138-4cd4-8688-2db1c3152d52.png)

因此，即使应用程序未从调试器中运行，您也可以在交互式调试器中分析崩溃的应用程序。当我们调用`bt`命令时，GDB 会显示崩溃时的堆栈跟踪：

![](img/56155e9f-ad93-4de6-b78e-4511160d4840.png)

这样，即使最初没有在调试器中运行，我们也可以找出应用程序崩溃的根本原因。

# 还有更多...

使用 GDB 分析核心转储是嵌入式应用程序的广泛使用和有效实践。然而，要使用 GDB 的全部功能，应用程序应该构建时支持调试符号。

然而，在大多数情况下，嵌入式应用程序会在没有调试符号的情况下部署和运行，以减小二进制文件的大小。在这种情况下，对核心转储的分析变得更加困难，可能需要一些特定架构的汇编语言和数据结构实现的内部知识。

# 使用 gdbserver 进行调试

嵌入式开发的环境通常涉及两个系统——构建系统和目标系统，或者模拟器。尽管 GDB 的命令行界面使其成为低性能嵌入式系统的不错选择，但在许多情况下，由于远程通信的高延迟，目标系统上的交互式调试是不切实际的。

在这种情况下，开发人员可以使用 GDB 提供的远程调试支持。在这种设置中，嵌入式应用程序使用 gdbserver 在目标系统上启动。开发人员在构建系统上运行 GDB，并通过网络连接到 gdbserver。

在这个配方中，我们将学习如何使用 GDB 和 gdbserver 开始调试应用程序。

# 准备就绪...

按照第二章的*连接到嵌入式系统*配方，*设置环境*，在目标系统上有`hello`应用程序可用。

# 如何做...

我们将使用前面的示例中使用的相同应用程序，但现在我们将在不同的环境中运行 GDB 和应用程序：

1.  切换到目标系统窗口，然后输入*Ctrl* + *D*以注销当前用户会话。

1.  以`user`身份登录，使用`user`密码。

1.  在`gdbserver`下运行`hello`应用程序：

```cpp
$ gdbserver 0.0.0.0:9090 ./hello
```

1.  切换到构建系统终端，并将目录更改为`/mnt`：

```cpp
# cd /mnt
```

1.  运行`gdb`，将应用程序二进制文件作为参数传递：

```cpp
# gdb -q hello
```

1.  通过在 GDB 命令行中输入以下命令来配置远程连接：

```cpp
target remote X.X.X.X:9090
```

1.  最后，键入`continue`命令：

```cpp
 continue
```

现在程序正在运行，我们可以看到它的输出并像在本地运行一样对其进行调试。

# 工作原理...

首先，我们以 root 用户身份登录到目标系统并安装 gdbserver，除非它已经安装。安装完成后，我们再次使用用户凭据登录并运行 gdbserver，将要调试的应用程序的名称、IP 地址和要监听的端口作为其参数传递。

然后，我们切换到我们的构建系统并在那里运行 GDB。但是，我们不直接在 GDB 中运行应用程序，而是指示 GDB 使用提供的 IP 地址和端口建立与远程主机的连接。之后，您在 GDB 提示符处键入的所有命令都将传输到 gdbserver 并在那里执行。

# 添加调试日志

日志记录和诊断是任何嵌入式项目的重要方面。在许多情况下，使用交互式调试器是不可能或不切实际的。在程序停在断点后，硬件状态可能在几毫秒内发生变化，开发人员没有足够的时间来分析它。收集详细的日志数据并使用工具进行分析和可视化是高性能、多线程、时间敏感的嵌入式系统的更好方法。

日志记录本身会引入一定的延迟。首先，需要时间来格式化日志消息并将其放入日志流中。其次，日志流应可靠地存储在持久存储器中，例如闪存卡或磁盘驱动器，或者发送到远程系统。

在本教程中，我们将学习如何使用日志记录而不是交互式调试来查找问题的根本原因。我们将使用不同日志级别的系统来最小化日志记录引入的延迟。

# 如何做...

我们将修改我们的应用程序以输出对根本原因分析有用的信息：

1.  转到您的工作目录`~/test`，并复制`loop`项目目录。将副本命名为`loop2`。切换到`loop2`目录。

1.  使用文本编辑器打开`loop.cpp`文件。

1.  添加一个`include`：

```cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

#include <syslog.h>
```

1.  通过在以下代码片段中突出显示的方式修改`runner`函数，添加对`syslog`函数的调用：

```cpp
void runner(std::chrono::milliseconds limit,
            std::function<void(int)> fn,
            int value) {
  auto start = std::chrono::system_clock::now();
  fn(value);
  auto end = std::chrono::system_clock::now();
  std::chrono::milliseconds delta =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
 syslog(LOG_DEBUG, "Delta is %ld",
         static_cast<long int>(delta.count()));
  if (delta > limit) {
 syslog(LOG_ERR, 
 "Execution time %ld ms exceeded %ld ms limit",
 static_cast<long int>(delta.count()),
 static_cast<long int>(limit.count()));
    throw std::runtime_error("Time limit exceeded");
  }
}
```

1.  同样，更新`main`函数以初始化和完成`syslog`：

```cpp
int main() {
 openlog("loop3", LOG_PERROR, LOG_USER);
  int max_delay = 10;
  for (int i = 0; i < max_delay; i++) {
    runner(std::chrono::milliseconds(max_delay), delay_ms, i);
  }
 closelog();
  return 0;
}
```

1.  切换到构建系统终端。转到`/mnt/loop2`目录并运行程序：

```cpp
# cmake && make
```

1.  将生成的`binary`文件复制到目标系统并运行它：

```cpp
$ ./loop 
```

调试输出冗长，并提供更多上下文以找到问题的根本原因。

# 工作原理...

在本教程中，我们使用标准日志记录工具`syslog`添加了日志记录。首先，我们通过调用`openlog`来初始化我们的日志记录：

```cpp
 openlog("loop3", LOG_PERROR, LOG_USER);
```

接下来，我们将日志记录添加到`runner`函数中。有不同的日志记录级别，可以方便地过滤日志消息，从最严重到最不严重。我们使用`LOG_DEBUG`级别记录`delta`值，该值表示`runner`调用的函数实际运行的时间有多长：

```cpp
 syslog(LOG_DEBUG, "Delta is %d", delta);
```

此级别用于记录对应用程序调试有用的详细信息，但在生产环境中运行应用程序时可能会过于冗长。

但是，如果`delta`超过限制，我们将使用`LOG_ERR`级别记录此情况，以指示通常不应发生此情况并且这是一个错误：

```cpp
 syslog(LOG_ERR, 
 "Execution time %ld ms exceeded %ld ms limit",
 static_cast<long int>(delta.count()),
 static_cast<long int>(limit.count()));
```

在从应用程序返回之前，我们关闭日志记录以确保所有日志消息都得到适当保存：

```cpp
 closelog();
```

当我们在目标系统上运行应用程序时，我们可以在屏幕上看到我们的日志消息：

![](img/fee9835c-ae1e-48c7-ac4c-7d5061ab539c.png)

由于我们使用标准的 Linux 日志记录，我们也可以在系统日志中找到消息：

![](img/2aafea69-e7e5-4925-a431-9efec515aca3.png)

如您所见，记录并不难实现，但在调试和正常操作期间，它对于找出应用程序中各种问题的根本原因非常有帮助。

# 还有更多...

有许多日志记录库和框架，可能比标准记录器更适合特定任务；例如，*Boost.Log*，网址为[`theboostcpplibraries.com/boost.log`](https://theboostcpplibraries.com/boost.log)，以及*spdlog*，网址为[`github.com/gabime/spdlog`](https://github.com/gabime/spdlog)。它们提供了比`syslog`的通用 C 接口更方便的 C++接口。在开始项目工作时，请检查现有的日志记录库，并选择最适合您要求的库。

# 使用调试和发布构建

正如我们在前面的食谱中所学到的，记录会带来相关成本。它会延迟格式化日志消息并将其写入持久存储或远程系统。

使用日志级别有助于通过跳过将一些消息写入日志文件来减少开销。但是，在将消息传递给`log`函数之前，消息通常会被格式化。例如，在系统错误的情况下，开发人员希望将系统报告的错误代码添加到日志消息中。尽管字符串格式化通常比将数据写入文件要便宜，但对于负载高的系统或资源有限的系统来说，这可能仍然是一个问题。

编译器添加的调试符号不会增加运行时开销。但是，它们会增加生成二进制文件的大小。此外，编译器进行的性能优化可能会使交互式调试变得困难。

在本食谱中，我们将学习如何通过分离调试和发布构建并使用 C 预处理器宏来避免运行时开销。

# 如何做...

我们将修改我们在前面的食谱中使用的应用程序的构建规则，以拥有两个构建目标——调试和发布：

1.  转到您的工作目录`~/test`，并复制`loop2`项目目录。将副本命名为`loop3`。切换到`loop3`目录。

1.  使用文本编辑器打开`CMakeLists.txt`文件。替换以下行：

```cpp
SET(CMAKE_CXX_FLAGS "-g --std=c++11")
```

1.  前面的行需要替换为以下行：

```cpp
SET(CMAKE_CXX_FLAGS_RELEASE "--std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} -g -DDEBUG")
```

1.  使用文本编辑器打开`loop.cpp`文件。通过添加突出显示的行来修改文件：

```cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <cstdarg>

#ifdef DEBUG
#define LOG_DEBUG(fmt, args...) fprintf(stderr, fmt, args)
#else
#define LOG_DEBUG(fmt, args...)
#endif

void runner(std::chrono::milliseconds limit,
            std::function<void(int)> fn,
            int value) {
  auto start = std::chrono::system_clock::now();
  fn(value);
  auto end = std::chrono::system_clock::now();
  std::chrono::milliseconds delta =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
 LOG_DEBUG("Delay: %ld ms, max: %ld ms\n",
            static_cast<long int>(delta.count()),
            static_cast<long int>(limit.count()));
  if (delta > limit) {
    throw std::runtime_error("Time limit exceeded");
  }
}
```

1.  切换到构建系统终端。转到`/mnt/loop3`目录并运行以下代码：

```cpp
# cmake -DCMAKE_BUILD_TYPE=Release . && make
```

1.  将生成的`loop`二进制文件复制到目标系统并运行它：

```cpp
$ ./loop 
```

1.  如您所见，该应用程序不会生成任何调试输出。现在让我们使用`ls -l`命令检查其大小：

```cpp
$ ls -l loop
-rwxr-xr-x 1 dev dev 24880 Jun 1 00:50 loop
```

1.  生成的二进制文件的大小为 24 KB。现在，让我们构建`Debug`构建并进行如下比较：

```cpp
$ cmake -DCMAKE_BUILD_TYPE=Debug && make clean && make
```

1.  检查可执行文件的大小：

```cpp
$ ls -l ./loop
-rwxr-xr-x 1 dev dev 80008 Jun 1 00:51 ./loop
```

1.  可执行文件的大小现在是 80 KB。它比发布构建大三倍以上。像以前一样运行它：

```cpp
$ ./loop 
```

如您所见，输出现在不同了。

# 它是如何工作的...

我们从用于*添加调试日志*食谱的项目副本开始，并创建两个不同的构建配置：

+   **调试**：具有交互式调试和调试日志支持的配置

+   **发布**：高度优化的配置，在编译时禁用了所有调试支持

为了实现它，我们利用了`CMake`提供的功能。它支持开箱即用的不同构建类型。我们只需要分别为发布和调试构建定义编译选项。

我们为发布构建定义的唯一构建标志是要使用的 C++标准。我们明确要求代码符合 C++11 标准：

```cpp
SET(CMAKE_CXX_FLAGS_RELEASE "--std=c++11")
```

对于调试构建，我们重用与发布构建相同的标志，将其引用为`${CMAKE_CXX_FLAGS_RELEASE}`，并添加两个选项。`-g`指示编译器向目标可执行二进制文件添加调试符号，而`-DDEBUG`定义了一个预处理宏`DEBUG`。

我们在`loop.cpp`的代码中使用`DEBUG`宏来选择`LOG_DEBUG`宏的两种不同实现。

如果定义了`DEBUG`，`LOG_DEBUG`会扩展为调用`fprintf`函数，该函数在标准错误通道中执行实际的日志记录。然而，如果未定义`DEBUG`，`LOG_DEBUG`会扩展为空字符串。这意味着在这种情况下，`LOG_DEBUG`不会产生任何代码，因此不会增加任何运行时开销。

我们在运行函数的主体中使用`LOG_DEBUG`来记录实际延迟和限制的值。请注意，`LOG_DEBUG`周围没有`if` - 格式化和记录数据或不执行任何操作的决定不是由我们的程序在运行时做出的，而是由代码预处理器在构建应用程序时做出的。

要选择构建类型，我们调用`cmake`，将构建类型的名称作为命令行参数传递：

```cpp
cmake -DCMAKE_BUILD_TYPE=Debug
```

`CMake`只生成一个`Make`文件来实际构建我们需要调用`make`的应用程序。我们可以将这两个命令合并成一个单独的命令行：

```cpp
cmake -DCMAKE_BUILD_TYPE=Release && make
```

第一次构建和运行应用程序时，我们选择发布版本。因此，我们看不到任何调试输出：

![](img/55077ea0-cd5e-411c-82f1-286108dc17f0.png)

之后，我们使用调试构建类型重新构建我们的应用程序，并在运行时看到不同的结果：

![](img/719acbab-871a-4922-88e4-07915e394e61.png)

通过调试和发布构建，您可以获得足够的信息进行舒适的调试，但请确保生产构建不会有任何不必要的开销。

# 还有更多...

在复杂项目中切换发布和调试构建时，请确保所有文件都已正确重建。最简单的方法是删除所有先前的构建文件。在使用`make`时，可以通过调用`make clean`命令来完成。

它可以作为命令行的一部分与`cmake`和`make`一起添加：

```cpp
cmake -DCMAKE_BUILD_TYPE=Debug && make clean && make
```

将所有三个命令合并成一行对开发人员更加方便。
