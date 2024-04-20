# 第十章：降低功耗

嵌入式系统有许多应用需要它们以电池供电。从小型**IoT**（**物联网**的缩写）设备收集传感器数据，将其推送到云端进行处理，到自主车辆和机器人 - 这些系统应尽可能节能，以便它们可以在没有稳定外部电源供应的情况下长时间运行。

功率效率意味着智能控制系统的所有部分的功耗，从外围设备到内存和处理器。功率控制的效率在很大程度上取决于硬件组件的选择和系统设计。如果处理器不支持动态电压控制或外围设备在空闲时无法进入节能模式，那么在软件方面就无法做太多。然而，如果硬件组件实现了标准规范，例如**高级配置和电源接口**（**ACPI**），那么很多功耗管理的负担可以转移到操作系统内核。

在本章中，我们将探索现代硬件平台的不同节能模式以及如何利用它们。我们将学习如何管理外部设备的电源状态，并通过编写更高效的软件来减少处理器的功耗。

我们将涵盖以下主题：

+   在 Linux 中探索节能模式

+   使用**RTC**（**实时时钟**的缩写）唤醒

+   控制 USB 设备的自动挂起

+   配置 CPU 频率

+   使用事件等待

+   使用 PowerTOP 对功耗进行分析

本章的配方将帮助您有效利用现代操作系统的节能功能，并编写针对电池供电设备进行优化的代码。

# 技术要求

要在本章中运行代码示例，您需要具有树莓派 PI 盒子修订版 3 或更高版本。

# 在 Linux 中探索节能模式

当系统处于空闲状态且没有工作要做时，可以将其置于睡眠状态以节省电源。类似于人类的睡眠，它在外部事件唤醒之前无法做任何事情，例如闹钟。

Linux 支持多种睡眠模式。选择睡眠模式和它可以节省的功率取决于硬件支持以及进入该模式和从中唤醒所需的时间。

支持的模式如下：

+   **挂起到空闲**（**S2I**）：这是一种轻度睡眠模式，可以纯粹通过软件实现，不需要硬件支持。设备进入低功耗模式，时间保持暂停，以便处理器在节能空闲状态下花费更多时间。系统通过来自任何外围设备的中断唤醒。

+   **待机**：这类似于 S2I，但通过将所有非引导 CPU 脱机来提供更多的节能。某些设备的中断可以唤醒系统。

+   **挂起到 RAM**（**STR**或**S3**）：系统的所有组件（除了内存），包括 CPU，都进入低功耗模式。系统状态保持在内存中，直到被来自有限设备集的中断唤醒。此模式需要硬件支持。

+   **休眠**或**挂起到磁盘**：这提供了最大的节能，因为所有系统组件都可以关闭电源。进入此状态时，会拍摄内存快照并写入持久存储（磁盘或闪存）。之后，系统可以关闭。作为引导过程的一部分，在唤醒时，恢复保存的快照并系统恢复其工作。

在这个配方中，我们将学习如何查询特定系统支持的睡眠模式以及如何切换到其中之一。

# 如何做...

在这个配方中，我们将使用简单的 bash 命令来访问在**QEMU**（**快速仿真器**的缩写）中运行的 Linux 系统支持的睡眠模式。

1.  按照第三章中描述的步骤运行树莓派 QEMU，*使用不同的架构*。

1.  以用户`pi`登录，使用密码`raspberry`。

1.  运行`sudo`以获取 root 访问权限：

```cpp
$ sudo bash
#
```

1.  要获取支持的睡眠模式列表，请运行以下命令：

```cpp
 # cat /sys/power/state
```

1.  现在切换到其中一个支持的模式：

```cpp
 # echo freeze > /sys/power/state
```

1.  系统进入睡眠状态，但我们没有指示它如何唤醒。现在关闭 QEMU 窗口。

# 工作原理...

电源管理是 Linux 内核的一部分；这就是为什么我们不能使用 Docker 容器来处理它。Docker 虚拟化是轻量级的，并使用主机操作系统的内核。

我们也不能使用真正的树莓派板，因为由于硬件限制，它根本不提供任何睡眠模式。然而，QEMU 提供了完整的虚拟化，包括我们用来模拟树莓派的内核中的电源管理。

Linux 通过 sysfs 接口提供对其电源管理功能的访问。应用程序可以读取和写入`/sys/power`目录中的文本文件。对于 root 用户，对电源管理功能的访问是受限的；这就是为什么我们需要在登录系统后获取 root shell：

```cpp
$ sudo bash
```

现在我们可以获取支持的睡眠模式列表。为此，我们读取`/sys/power/state`文件：

```cpp
$ cat /sys/power/state
```

该文件由一行文本组成。每个单词代表一个支持的睡眠模式，模式之间用空格分隔。我们可以看到 QEMU 内核支持两种模式：`freeze`和`mem`：

![](img/e12ba0b1-2558-41d6-83c6-8ad7026751c3.png)

Freeze 代表我们在前一节中讨论的 S2I 状态。`mem`的含义由`/sys/power/mem_sleep`文件的内容定义。在我们的系统中，它只包含`[s2idle]`，代表与`freeze`相同的 S2I 状态。

让我们将我们的模拟器切换到`freeze`模式。我们将单词`freeze`写入`/sys/power/state`，立即 QEMU 窗口变黑并冻结：

![](img/a3f7043f-286b-49d6-acc7-a05c553aa1dd.png)

我们能够让模拟的 Linux 系统进入睡眠状态，但无法唤醒它——没有它能理解的中断源。我们了解了不同的睡眠模式和内核 API 来处理它们。根据嵌入式系统的要求，您可以使用这些模式来降低功耗。

# 还有更多...

有关睡眠模式的更多信息可以在*Linux 内核指南*的相应部分中找到，网址为[`www.kernel.org/doc/html/v4.19/admin-guide/pm/sleep-states.html`](https://www.kernel.org/doc/html/v4.19/admin-guide/pm/sleep-states.html)。

# 使用 RTC 唤醒

在前面的示例中，我们能够让我们的 QEMU 系统进入睡眠状态，但无法唤醒它。我们需要一个设备，当其大部分内部组件关闭电源时，可以向系统发送中断。

**RTC**（**实时时钟**）就是这样的设备之一。它的功能之一是在系统关闭时保持内部时钟运行，并且为此，它有自己的电池。RTC 的功耗类似于电子手表；它使用相同的 3V 电池，并且可以在其自身的电源上工作多年。

RTC 可以作为闹钟工作，在给定时间向 CPU 发送中断。这使得它成为按计划唤醒系统的理想设备。

在这个示例中，我们将学习如何使用内置 RTC 在特定时间唤醒 Linux 系统。

# 如何做...

在这个示例中，我们将提前将系统的唤醒时间设置为 1 分钟，并将系统置于睡眠状态：

1.  登录到任何具有 RTC 时钟的 Linux 系统——任何 Linux 笔记本都可以。不幸的是，树莓派没有内置 RTC，并且没有额外的硬件无法唤醒。

1.  使用`sudo`获取 root 权限：

```cpp
$ sudo bash
#
```

1.  指示 RTC 在`1`分钟后唤醒系统：

```cpp
# date '+%s' -d '+1 minute' > /sys/class/rtc/rtc0/wakealarm
```

1.  将系统置于睡眠状态：

```cpp
# echo freeze > /sys/power/state
```

1.  等待一分钟。您的系统将会唤醒。

# 工作原理...

与 Linux 内核提供的许多其他功能一样，RTC 可以通过 sysfs 接口访问。为了设置一个将向系统发送唤醒中断的闹钟，我们需要向`/sys/class/rtc/rtc0/wakealarm`文件写入一个**POSIX**（**Portable Operating System Interface**的缩写）时间戳。

我们在第十一章中更详细地讨论的 POSIX 时间戳，定义为自纪元以来经过的秒数，即 1970 年 1 月 1 日 00:00。

虽然我们可以编写一个程序，使用`time`函数读取当前时间戳，再加上 60，并将结果写入`wakealarm`文件，但我们可以使用 Unix shell 和`date`命令在一行中完成这个操作，这在任何现代 Unix 系统上都可以实现。

date 实用程序不仅可以使用不同格式格式化当前时间，还可以解释不同格式的日期和时间。

我们指示`date`解释时间字符串`+1 minute`，并使用格式化模式`%s`将其输出为 POSIX 时间戳。我们将其标准输出重定向到`wakealarm`文件，有效地传递给 RTC 驱动程序：

```cpp
date '+%s' -d '+1 minute' > /sys/class/rtc/rtc0/wakealarm
```

现在，知道 60 秒后闹钟会响，我们可以让系统进入睡眠状态。与前一个教程一样，我们将所需的睡眠模式写入`/sys/power/state`文件：

```cpp
# echo freeze > /sys/power/state
```

系统进入睡眠状态。您会注意到屏幕关闭了。如果您使用**Secure Shell**（**SSH**）连接到 Linux 框，命令行会冻结。然而，一分钟后它会醒来，屏幕会亮起，终端会再次响应。

这种技术非常适合定期、不经常地从传感器收集数据，比如每小时或每天。系统大部分时间都处于关闭状态，只有在收集数据并存储或发送到云端时才会唤醒，然后再次进入睡眠状态。

# 还有更多...

设置 RTC 闹钟的另一种方法是使用`rtcwake`实用程序。

# 控制 USB 设备的 autosuspend

关闭外部设备是节省电力的最有效方法之一。然而，并不总是容易理解何时可以安全地关闭设备。外围设备，如网络卡或存储卡，可以执行内部数据处理；否则，在任意时间关闭设备的缓存和电源可能会导致数据丢失。

为了缓解这个问题，许多通过 USB 连接的外部设备在主机请求时可以将自己切换到低功耗模式。这样，它们可以在进入挂起状态之前执行处理内部数据的所有必要步骤。

由于 Linux 只能通过其 API 访问外围设备，它知道设备何时被应用程序和内核服务使用。如果设备在一定时间内没有被使用，Linux 内核中的电源管理系统可以自动指示设备进入省电模式——不需要来自用户空间应用程序的显式请求。这个功能被称为**autosuspend**。然而，内核允许应用程序控制设备的空闲时间，之后 autosuspend 会生效。

在这个教程中，我们将学习如何启用 autosuspend 并修改特定 USB 设备的 autosuspend 间隔。

# 如何做...

我们将启用 autosuspend 并修改连接到 Linux 框的 USB 设备的 autosuspend 时间：

1.  登录到您的 Linux 框（树莓派、Ubuntu 和 Docker 容器不适用）。

1.  切换到 root 账户：

```cpp
$ sudo bash
#
```

1.  获取所有连接的 USB 设备的当前`autosuspend`状态：

```cpp
# for f in /sys/bus/usb/devices/*/power/control; do echo "$f"; cat $f; done
```

1.  为一个设备启用`autosuspend`：

```cpp
# echo auto > /sys/bus/usb/devices/1-1.2/power/control
```

1.  读取设备的`autosuspend`间隔：

```cpp
# cat /sys/bus/usb/devices/1-1.2/power/autosuspend_delay_ms 
```

1.  修改`autosuspend`间隔：

```cpp
# echo 5000 > /sys/bus/usb/devices/1-1.2/power/autosuspend_delay_ms 
```

1.  检查设备的当前电源模式：

```cpp
# cat /sys/bus/usb/devices/1-1.2/power/runtime_status
```

相同的操作可以使用标准文件 API 在 C++中编程。

# 它是如何工作的...

Linux 通过 sysfs 文件系统公开其电源管理 API，这使得可以通过标准文件读写操作读取当前状态并修改任何设备的设置成为可能。因此，我们可以使用支持基本文件操作的任何编程语言来控制 Linux 中的外围设备。

为了简化我们的示例，我们将使用 Unix shell，但在必要时完全相同的逻辑可以用 C++编程。

首先，我们检查所有连接的 USB 设备的`autosuspend`设置。在 Linux 中，每个 USB 设备的参数都作为`/sysfs/bus/usb/devices/`文件夹下的目录公开。每个设备目录又有一组代表设备参数的文件。所有与电源管理相关的参数都分组在`power`子目录中。

要读取`autosuspend`的状态，我们需要读取设备的`power`目录中的`control`文件。使用 Unix shell 通配符替换，我们可以为所有 USB 设备读取此文件：

```cpp
# for f in /sys/bus/usb/devices/*/power/control; do echo "$f"; cat $f; done
```

对于与通配符匹配的每个目录，我们显示控制文件的完整路径及其内容。结果取决于连接的设备，可能如下所示：

![](img/ad39f854-2adc-4c82-93d8-22a61a3718a6.png)

报告的状态可能是 autosuspend 或`on`。如果状态报告为 autosuspend，则自动电源管理已启用；否则，设备始终保持开启。

在我们的情况下，设备`usb1`，`1-1.1`和`1-1.2`是开启的。让我们修改`1-1.2`的配置以使用自动挂起。为此，我们只需向相应的`_control_`文件中写入字符串`_auto_`。

```cpp
# echo auto > /sys/bus/usb/devices/1-1.2/power/control
```

再次运行循环读取所有设备的操作显示，`1-1.2`设备现在处于`autosuspend`模式：

![](img/a67c2bca-1a51-47ae-a018-07121f050716.png)

它将在何时被挂起？我们可以从`power`子目录中的`autosuspend_delay_ms`文件中读取：

```cpp
# cat /sys/bus/usb/devices/1-1.2/power/autosuspend_delay_ms 
```

它显示设备在空闲`2000`毫秒后将被挂起：

![](img/eca24651-a7e7-4029-9c0c-a9e827d52322.png)

让我们将其更改为`5`秒。我们在`autosuspend_delay_ms`文件中写入`5000`：

```cpp
# echo 5000 > /sys/bus/usb/devices/1-1.2/power/autosuspend_delay_ms 
```

再次读取它显示新值已被接受：

![](img/7b6ee3c8-c017-4d68-9df7-b5343e3bf17d.png)

现在让我们检查设备的当前电源状态。我们可以从`runtime_status`文件中读取它：

```cpp
# cat /sys/bus/usb/devices/1-1.2/power/runtime_status
```

状态报告为`active`：

![](img/71e3495c-e054-41b5-bdf4-1265560fe78f.png)

请注意，内核不直接控制设备的电源状态；它只请求它们改变状态。即使请求设备切换到挂起模式，它也可能因为各种原因而拒绝这样做，例如，它可能根本不支持节能模式。

通过 sysfs 接口访问任何设备的电源管理设置是调整运行 Linux OS 的嵌入式系统的功耗的强大方式。

# 还有更多...

没有直接的方法立即关闭 USB 设备；但在许多情况下，可以通过向`autosuspend_delay_ms`文件中写入`0`来实现。内核将零的自动挂起间隔解释为对设备的立即挂起请求。

在 Linux 中，有关 USB 电源管理的更多细节可以在 Linux 内核文档的相应部分中找到，该文档可在[`www.kernel.org/doc/html/v4.13/driver-api/usb/power-management.html`](https://www.kernel.org/doc/html/v4.13/driver-api/usb/power-management.html)上找到。

# 配置 CPU 频率

CPU 频率是系统的重要参数，它决定了系统的性能和功耗。频率越高，CPU 每秒可以执行的指令就越多。但这是有代价的。更高的频率意味着更高的功耗，反过来意味着需要散热更多的热量以避免处理器过热。

现代处理器能够根据负载使用不同的操作频率。对于计算密集型任务，它们使用最大频率以实现最大性能，但当系统大部分空闲时，它们会切换到较低的频率以减少功耗和热量影响。

适当的频率选择由操作系统管理。在这个示例中，我们将学习如何在 Linux 中设置 CPU 频率范围并选择频率管理器，以微调 CPU 频率以满足您的需求。

# 如何做...

我们将使用简单的 shell 命令来调整树莓派盒子上的 CPU 频率参数：

1.  登录到树莓派或另一个非虚拟化的 Linux 系统。

1.  切换到 root 帐户：

```cpp
$ sudo bash
#
```

1.  获取系统中所有 CPU 核心的当前频率：

```cpp
# cat /sys/devices/system/cpu/*/cpufreq/scaling_cur_freq
```

1.  获取 CPU 支持的所有频率：

```cpp
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
```

1.  获取可用的 CPU 频率管理器：

```cpp
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
```

1.  现在让我们检查当前使用的频率管理器是哪个：

```cpp
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 
```

1.  将 CPU 的最小频率调整到最高支持的频率：

```cpp
# echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

1.  再次显示当前频率以了解效果：

```cpp
# cat /sys/devices/system/cpu/*/cpufreq/scaling_cur_freq
```

1.  将最小频率调整到最低支持的频率：

```cpp
# echo 600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_fre
```

1.  现在让我们检查 CPU 频率如何取决于所使用的管理器。选择`performance`管理器并获取当前频率：

```cpp
# echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# cat /sys/devices/system/cpu/*/cpufreq/scaling_cur_freq
```

1.  选择`powersave`管理器并观察结果：

```cpp
# echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# cat /sys/devices/system/cpu/*/cpufreq/scaling_cur_freq
```

您可以使用常规文件 API 在 C++中实现相同的逻辑。

# 它是如何工作的...

与 USB 电源管理类似，CPU 频率管理系统 API 通过 sysfs 公开。我们可以像常规文本文件一样读取和修改其参数。

我们可以在`/sys/devices/system/cpu/`目录下找到与 CPU 核心相关的所有设置。配置参数按 CPU 核心分组在名为每个代码索引的子目录中，如`cpu1`，`cpu2`等。

我们对与 CPU 频率管理相关的几个参数感兴趣，这些参数位于每个核心的`cpufreq`子目录中。让我们读取所有可用核心的当前频率：

```cpp
# cat /sys/devices/system/cpu/*/cpufreq/scaling_cur_freq
```

我们可以看到所有核心的频率都是相同的，为 600 MHz（`cpufreq`子系统使用 KHz 作为频率的测量单位）：

![](img/6d9f305d-d3ca-47eb-8766-1b3fa5718836.png)

接下来，我们弄清楚 CPU 支持的所有频率：

```cpp
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
```

树莓派 3 的 ARM 处理器仅支持两种频率，600 MHz 和 1.2 GHz：

![](img/dd10cdcb-aff0-4f3d-9b17-7416aac38365.png)

我们无法直接设置所需的频率。Linux 通过所谓的**管理器**内部管理 CPU 频率，并且只允许我们调整两个参数：

+   管理器的频率范围

+   管理器的类型

尽管这看起来像是一个限制，但这两个参数足够灵活，可以实现相当复杂的策略。让我们看看如何修改这两个参数如何影响 CPU 频率。

首先，让我们弄清楚支持哪些管理器以及当前使用的是哪个：

![](img/52365c50-07d0-495b-942b-f45dc54dd619.png)

当前的管理器是`ondemand`。*它根据系统负载调整频率。目前，树莓派板卡相当空闲，因此使用最低频率 600 MHz。但是如果我们将最低频率设置为最高频率呢？

```cpp
# echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

在我们更新了一个核心的`scaling_min_freq`参数后，所有核心的频率都被更改为最大值：

![](img/ac1af6c9-2bce-4cfa-a881-8d99f9e5ebad.png)

由于四个核心都属于同一个 CPU，我们无法独立地改变它们的频率；改变一个核心的频率会影响所有核心。但是，我们可以独立地控制不同 CPU 的频率。

现在我们将最小频率恢复到 600 MHz 并更改管理器。我们选择了`performance`管理器，而不是调整频率的`ondemand`管理器，旨在无条件地提供最大性能：

```cpp
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_g;overnor
```

毫不奇怪，它将频率提高到最大支持的频率：

![](img/ac5ded8a-1be2-402d-9f5c-b475828d8ba9.png)

另一方面，`powersave`调度程序旨在尽可能节省电量，因为它始终坚持使用最低支持的频率，而不考虑负载：

![](img/fc0f3d39-c055-4afe-b659-df0401954695.png)

正如您所看到的，调整频率范围和频率调度程序可以灵活地调整频率，以便根据系统的性质减少 CPU 消耗的电量。

# 还有更多...

除了`ondemand`、`performance`和`powersave`之外，还有其他调度程序可以提供更灵活的 CPU 频率调整，供用户空间应用程序使用。您可以在 Linux CPUFreq 的相应部分中找到有关可用调度程序及其属性的更多详细信息[`www.kernel.org/doc/Documentation/cpu-freq/governors.txt`](https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt)

# 使用事件进行等待

等待是软件开发中极为常见的模式。应用程序必须等待用户输入或数据准备好进行处理。嵌入式程序与外围设备通信，需要知道何时可以从设备读取数据以及设备何时准备好接受数据。

通常，开发人员使用轮询技术的变体进行等待。他们在循环中检查设备特定的可用性标志，当设备将其设置为 true 时，他们继续读取或写入数据。

尽管这种方法易于实现，但从能耗的角度来看效率低下。当处理器不断忙于循环检查标志时，操作系统电源管理器无法将其置于更节能的模式中。根据负载，我们之前讨论的 Linux `ondemand`频率调度程序甚至可以决定增加 CPU 频率，尽管这实际上是一种等待。此外，轮询请求可能会阻止目标设备或设备总线保持在节能模式，直到数据准备就绪。

这就是为什么对于关心能效的轮询程序，它应该依赖于操作系统生成的中断和事件。

在本教程中，我们将学习如何使用操作系统事件来等待特定的 USB 设备连接。

# 如何做...

我们将创建一个应用程序，可以监视 USB 设备并等待特定设备出现：

1.  在您的工作`~/test`目录中创建一个名为`udev`的子目录。

1.  使用您喜欢的文本编辑器在`udev`子目录中创建一个名为`udev.cpp`的文件。

1.  将必要的包含和`namespace`定义放入`udev.cpp`文件中：

```cpp
#include <iostream>
#include <functional>

#include <libudev.h>
#include <poll.h>

namespace usb {
```

1.  现在，让我们定义`Device`类：

```cpp
class Device {
  struct udev_device *dev{0};

  public:
    Device(struct udev_device* dev) : dev(dev) {
    }

    Device(const Device& other) : dev(other.dev) {
      udev_device_ref(dev);
    }

    ~Device() {
        udev_device_unref(dev);
    }

    std::string action() const { 
        return udev_device_get_action(dev);
     }

    std::string attr(const char* name) const {
      const char* val = udev_device_get_sysattr_value(dev,
             name);
      return val ? val : "";
    }
};
```

1.  之后，添加`Monitor`类的定义：

```cpp
class Monitor {
  struct udev_monitor *mon;

  public:
    Monitor() {
      struct udev* udev = udev_new();
      mon = udev_monitor_new_from_netlink(udev, "udev");
      udev_monitor_filter_add_match_subsystem_devtype(
           mon, "usb", NULL);
      udev_monitor_enable_receiving(mon);
    }

    Monitor(const Monitor& other) = delete;

    ~Monitor() {
      udev_monitor_unref(mon);
    }

    Device wait(std::function<bool(const Device&)> process) {
      struct pollfd fds[1];
      fds[0].events = POLLIN;
      fds[0].fd = udev_monitor_get_fd(mon);

      while (true) {
          int ret = poll(fds, 1, -1);
          if (ret < 0) {
            throw std::system_error(errno, 
                std::system_category(),
                "Poll failed");
          }
          if (ret) {
            Device d(udev_monitor_receive_device(mon));
            if (process(d)) {
              return d;
            };
          }
      }
    }
};
};
```

1.  在`usb`命名空间中定义了`Device`和`Monitor`之后，添加一个简单的`main`函数，展示如何使用它们：

```cpp
int main() {
  usb::Monitor mon;
  usb::Device d = mon.wait([](auto& d) {
    auto id = d.attr("idVendor") + ":" + 
              d.attr("idProduct");
    auto produce = d.attr("product");
    std::cout << "Check [" << id << "] action: " 
              << d.action() << std::endl;
    return d.action() == "bind" && 
           id == "8086:0808";
  });
  std::cout << d.attr("product")
            << " connected, uses up to "
            << d.attr("bMaxPower") << std::endl;
  return 0;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(udev)
add_executable(usb udev.cpp)
target_link_libraries(usb udev)
```

1.  使用`ssh`将`udev`目录复制到您 Linux 系统上的家目录中。

1.  登录到您的 Linux 系统，将目录切换到`udev`，并使用`cmake`构建程序：

```cpp
$cd ~/udev; cmake. && make
```

现在您可以构建并运行应用程序。

# 它是如何工作的...

为了获取有关 USB 设备事件的系统通知，我们使用了一个名为`libudev`的库。它只提供了一个简单的 C 接口，因此我们创建了简单的 C++包装器来使编码更容易。

对于我们的包装器类，我们声明了一个名为`usb`的`namespace`：

```cpp
namespace usb {
```

它包含两个类。第一个类是`Device`，它为我们提供了一个 C++接口，用于低级`libudev`对象`udev_device`。

我们定义了一个构造函数，从`udev_device`指针创建了一个`Device`实例，并定义了一个析构函数来释放`udev_device`。在内部，`libudev`使用引用计数来管理其对象，因此我们的析构函数调用一个函数来减少`udev_device`的引用计数：

```cpp
    ~Device() {
        udev_device_unref(dev);
    }
    Device(const Device& other) : dev(other.dev) {
      udev_device_ref(dev);
    }
```

这样，我们可以复制`Device`实例而不会出现内存泄漏或文件描述符泄漏。

除了构造函数和析构函数之外，`Device`类只有两个方法：`action`和`attr`。`action`方法返回最近的 USB 设备动作：

```cpp
    std::string action() const { 
        return udev_device_get_action(dev);
     }
```

`attr`方法返回与设备关联的任何 sysfs 属性：

```cpp
    std::string attr(const char* name) const {
      const char* val = udev_device_get_sysattr_value(dev,
             name);
      return val ? val : "";
    }
```

`Monitor`类也有构造函数和析构函数，但我们通过禁用复制构造函数使其不可复制：

```cpp
    Monitor(const Monitor& other) = delete;
```

构造函数使用静态变量初始化`libudev`实例，以确保它只初始化一次：

```cpp
      struct udev* udev = udev_new();
```

它还设置了监视过滤器并启用了监视：

```cpp
      udev_monitor_filter_add_match_subsystem_devtype(
           mon, "usb", NULL);
      udev_monitor_enable_receiving(mon);
```

`wait`方法包含最重要的监视逻辑。它接受类似函数的`process`对象，每次检测到事件时都会调用它：

```cpp
Device wait(std::function<bool(const Device&)> process) {
```

如果事件和它来自的设备是我们需要的，函数应返回`true`；否则，它返回`false`以指示`wait`应继续工作。

在内部，`wait`函数创建一个文件描述符，用于将设备事件传递给程序：

```cpp
      fds[0].fd = udev_monitor_get_fd(mon);
```

然后它设置监视循环。尽管它的名称是`poll`函数，但它并不会不断检查设备的状态；它会等待指定文件描述符上的事件。我们传递`-1`作为超时，表示我们打算永远等待事件：

```cpp
int ret = poll(fds, 1, -1);
```

`poll`函数仅在出现错误或新的 USB 事件时返回。我们通过抛出异常来处理错误情况：

```cpp
          if (ret < 0) {
            throw std::system_error(errno, 
                std::system_category(),
                "Poll failed");
          }
```

对于每个事件，我们创建一个`Device`的新实例，并将其传递给`process`。如果`process`返回`true`，我们退出等待循环，将`Device`的实例返回给调用者：

```cpp
            Device d(udev_monitor_receive_device(mon));
            if (process(d)) {
              return d;
            };
```

让我们看看如何在我们的应用程序中使用这些类。在`main`函数中，我们创建一个`Monitor`实例并调用其`wait`函数。我们使用 lambda 函数来处理每个动作：

```cpp
usb::Device d = mon.wait([](auto& d) {
```

在 lambda 函数中，我们打印有关所有事件的信息：

```cpp
    std::cout << "Check [" << id << "] action: " 
              << d.action() << std::endl;
```

我们还检查特定的动作和设备`id`：

```cpp
    return d.action() == "bind" && 
           id == "8086:0808";
```

一旦找到，我们会显示有关其功能和功率需求的信息：

```cpp
  std::cout << d.attr("product")
            << " connected, uses up to "
            << d.attr("bMaxPower") << std::endl;
```

最初运行此应用程序不会产生任何输出：

![](img/981bab41-7f4d-4a76-9bd0-5e55b1811789.png)

然而，一旦我们插入 USB 设备（在我这里是 USB 麦克风），我们可以看到以下输出：

![](img/d8c72d62-873a-42ca-9486-e57804844157.png)

应用程序可以等待特定的 USB 设备，并在连接后处理它。它可以在不忙碌循环的情况下完成，依靠操作系统提供的信息。因此，应用程序大部分时间都在睡眠，而`poll`调用被操作系统阻塞。

# 还有更多...

有许多`libudev`的 C++包装器。您可以使用其中之一，或者使用本示例中的代码作为起点创建自己的包装器。

# 使用 PowerTOP 进行功耗分析

在像 Linux 这样运行多个用户空间和内核空间服务并同时控制许多外围设备的复杂操作系统中，要找到可能导致过多功耗的组件并不总是容易的。即使找到了效率低下的问题，修复它可能也很困难。

其中一个解决方案是使用功耗分析工具，如 PowerTOP。它可以诊断 Linux 系统中的功耗问题，并允许用户调整可以节省功耗的系统参数。

在这个示例中，我们将学习如何在树莓派系统上安装和使用 PowerTOP。

# 如何做...

在这个示例中，我们将以交互模式运行 PowerTOP 并分析其输出：

1.  以`pi`用户身份登录到您的树莓派系统，使用密码`raspberry`。

1.  运行`sudo`以获得 root 访问权限：

```cpp
$ sudo bash
#
```

1.  从存储库安装 PowerTOP：

```cpp
 # apt-get install powertop
```

1.  保持在 root shell 中，运行 PowerTOP：

```cpp
 # powertop
```

PowerTOP UI 将显示在您的终端中。使用*Tab*键在其屏幕之间导航。

# 工作原理...

PowerTOP 是由英特尔创建的用于诊断 Linux 系统中功耗问题的工具。它是 Raspbian 发行版的一部分，可以使用`apt-get`命令安装：

```cpp
# apt-get install powertop
```

当我们在没有参数的情况下运行它时，它会以交互模式启动，并按其功耗和它们生成事件的频率对所有进程和内核任务进行排序。正如我们在*使用事件进行等待*一节中讨论的那样，程序需要频繁唤醒处理器，它的能效就越低：

![](img/8f7a54ae-1a79-4f1f-91ce-152fbfd006a0.png)

使用*Tab*键，我们可以切换到其他报告模式。例如，设备统计显示设备消耗了多少能量或 CPU 时间：

![](img/2e968743-e8ab-4fe5-a55b-b05a3742586c.png)

另一个有趣的选项卡是 Tunab。PowerTOP 可以检查影响功耗的一些设置，并标记那些不够理想的设置：

![](img/3140c865-d15a-48e7-80de-28bd1c5857a5.png)

如您所见，两个 USB 设备被标记为`Bad`，因为它们没有使用自动挂起。通过按下*Enter*键，PowerTOP 启用了自动挂起，并显示了一个可以从脚本中使用以使其永久化的命令行。启用自动挂起后，可调状态变为`Good`：

![](img/866eb512-32b4-4ea7-a5bd-5f42106a267c.png)

一些系统参数可以调整以节省电力。有时它们是显而易见的，比如在 USB 设备上使用自动挂起。有时它们不是，比如在用于将文件缓存刷新到磁盘的内核上使用超时。使用诊断和优化工具，如 PowerTOP，可以帮助您调整系统以实现最大功耗效率。

# 还有更多...

除了交互模式，PowerTOP 还有其他模式可帮助您优化功耗，如校准、工作负载和自动调整。有关 PowerTOP 功能、使用场景和结果解释的更多信息，请参阅[`01.org/sites/default/files/page/powertop_users_guide_201412.pdf`](https://01.org/sites/default/files/page/powertop_users_guide_201412.pdf)中的*PowerTOP 用户指南*。
