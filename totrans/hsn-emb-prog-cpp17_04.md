# 为嵌入式 Linux 和类似系统开发

现在，基于 SoC 的小型系统无处不在，从智能手机、视频游戏机、智能电视，到汽车和飞机中的信息娱乐系统。依赖于此类系统的消费设备极为常见。

除了消费设备外，它们还作为工业和建筑级控制器系统的一部分存在，在这些系统中，它们监控设备，响应输入，并执行整个传感器和执行器网络的预定任务。与 MCU 相比，SoC 的资源限制较少，通常运行完整的**操作系统**（**OS**），如基于 Linux 的操作系统、VxWorks 或 QNX。

在本章中，我们将涵盖以下主题：

+   如何为基于操作系统的嵌入式系统开发驱动程序

+   集成外围设备的方法

+   如何处理和实现实时性能要求

+   识别和处理资源限制

# 嵌入式操作系统

操作系统通常与嵌入式系统一起使用，当你直接为系统的硬件编写应用程序时，这是一个不切实际的建议。操作系统提供给应用程序的是一系列 API，这些 API 抽象化了使用该硬件实现的硬件和功能，例如网络通信或视频输出。

这里的权衡是在便利性与代码大小和复杂性之间。

与裸机实现理想情况下仅实现所需功能不同，操作系统附带任务调度器，以及运行的应用程序可能永远不会需要的功能。因此，了解何时使用操作系统而不是直接为硬件开发，以及理解随之而来的复杂性是很重要的。

使用操作系统的良好理由是，如果你需要能够同时运行不同的任务（多任务处理或多线程）。从头开始实现自己的调度器通常不值得付出努力。需要运行非固定数量的应用程序，并且能够随意添加和删除它们，使用操作系统也会使这一过程变得容易得多。

最后，当您能够访问操作系统和易于获取的驱动程序以及相关的 API 时，高级图形输出、图形加速（如 OpenGL）、触摸屏和高级网络功能（例如 SSH 和加密）的实现会变得更加容易。

常用的嵌入式操作系统包括以下：

| **名称** | **供应商** | **许可** | **平台** | **详细信息** |
| --- | --- | --- | --- | --- |
| Raspbian | 基于社区的 | 主要为 GPL，类似 | ARM（树莓派） | 基于 Debian Linux 的操作系统 |
| Armbian | 基于社区的 | GPLv2 | ARM（各种板） | 基于 Debian Linux 的操作系统 |
| Android | Google | GPLv2, Apache | ARM, x86, x86_64 | 基于 Linux |
| VxWorks | Wind River (Intel) | 商业版权 | ARM, x86, MIPS, PowerPC, SH-4 | 实时操作系统，单核内核 |
| QNX | BlackBerry | 商业版权 | ARMv7, ARMv8, x86 | 实时操作系统，微内核 |
| Windows IoT | 微软 | 商业版权 | ARM, x86 | 以前称为 Windows Embedded |
| NetBSD | NetBSD 基金会 | 2-clause BSD | ARM, 68k, MIPS, PowerPC, SPARC, RISC-V, x86 及其他 | 最便携的基于 BSD 的操作系统 |

所有这些操作系统共同的特点是它们处理基本功能，如内存和任务管理，同时通过编程接口（API）提供对硬件和操作系统功能的访问。

在本章中，我们将特别关注基于 SoC 和 SBC 的系统，这在前面的操作系统列表中有所体现。这些操作系统都旨在用于至少拥有几兆字节 RAM 的系统，以及从兆字节到千兆字节的存储空间。

如果目标 SoC 或 SBC 还未被现有的 Linux 发行版所针对，或者希望对系统进行大量自定义，可以使用 Yocto 项目（[`www.yoctoproject.org/`](http://www.yoctoproject.org/)）的工具。

基于 Linux 的嵌入式操作系统相当普遍，Android 是一个众所周知的例子。它主要用于智能手机、平板电脑和类似设备，这些设备高度依赖图形用户交互，同时依赖于 Android 应用程序基础设施和相关 API。由于这种高度专业化，它并不适合其他使用场景。

Raspbian 基于非常常见的 Debian Linux 发行版，主要针对 Raspberry Pi 系列的 SBC。Armbian 类似，但覆盖了更广泛的 SBC。这两个都是社区努力的结果。这与 Debian 项目类似，也可以直接用于嵌入式系统。Raspbian、Armbian 和其他类似项目的优势在于它们提供了用于目标 SBC 的现成镜像。

与基于 Linux 的操作系统一样，NetBSD 具有开源的优势，这意味着您可以完全访问源代码，并且可以大幅度自定义操作系统的任何方面，包括对自定义硬件的支持。NetBSD 和类似基于 BSD 的操作系统的一个重大优势是，操作系统是从单个代码库构建的，并由一组开发人员管理。这通常简化了嵌入式项目的开发和维护。

BSD 许可证（三或两条款）为商业项目提供了主要好处，因为此许可证仅要求提供归属，而不是要求制造商在请求时提供操作系统的完整源代码。如果对源代码进行了某些修改，添加了希望保持为闭源代码的代码模块，这可能非常相关。

例如，最近的 PlayStation 游戏机使用了一个修改版的 FreeBSD，这使得索尼能够针对硬件及其作为游戏机的使用进行大量优化，而无需将此代码与操作系统其余部分的源代码一起发布。

还存在一些专有选项，例如 BlackBerry（QNX）和 Microsoft（Windows IoT，以前称为 Windows Embedded，以前称为 Windows CE）提供的解决方案。这些通常需要按设备支付许可费，并且需要制造商的帮助来进行任何定制。

# 实时操作系统

实时操作系统（RTOS）的基本要求是它能够保证任务将在一定时间内执行并完成。这允许在执行时间批次中同一任务的执行时间变化（抖动）不可接受的情况下使用它们进行实时应用。

从这个角度来看，我们可以区分硬实时操作系统和软实时操作系统的基本区别：具有低抖动的操作系统是硬实时操作系统，因为它可以保证给定任务始终以几乎相同的延迟执行。具有较高抖动的操作系统通常但并不总是以相同的延迟执行任务。

在这两个类别中，我们还可以区分事件驱动和时间共享调度程序。前者根据优先级切换任务（优先级调度），而后者使用定时器定期切换任务。哪种设计更好取决于系统用途。

与事件驱动的调度程序相比，时间共享的主要优势在于它还为低优先级任务提供了更多的 CPU 时间，这使得多任务系统看起来运行得更加平滑。

通常情况下，只有当项目需求要求能够保证在严格定义的时间窗口内处理输入时，才会使用 RTOS。对于如机器人技术和工业应用等应用，每次执行动作必须在完全相同的时间范围内，否则会导致生产线中断或产品质量下降。

在本章后面我们将要讨论的示例项目中，我们不使用 RTOS，而是使用基于 Linux 的常规操作系统，因为没有硬时间要求。使用 RTOS 将带来不必要的负担，并可能增加复杂性和成本。

将实时操作系统（RTOS）视为一种方式，即在直接为硬件编程（裸金属）时尽可能接近实时性，同时不必放弃使用完整操作系统带来的所有便利。

# 定制外设和驱动程序

外设被定义为一种辅助设备，它可以为计算机系统添加 I/O 或其他功能。这可以是任何从 I2C、SPI 或 SD 卡控制器到音频或图形设备的设备。大多数这些设备是物理 SoC 的一部分，而其他设备则是通过 SoC 对外界暴露的接口添加的。外部外设的例子包括 RAM（通过 RAM 控制器）和**实时时钟**（**RTC**）。

当使用像 Raspberry Pi、Orange Pi 和无数类似系统这样的较便宜的 SBC 时，人们可能会遇到的一个问题是它们通常缺少 RTC，这意味着当它们断电时，它们就不再跟踪时间。背后的想法通常是这些板将无论如何都会连接到互联网，因此操作系统可以使用在线时间服务（**网络时间协议**，或**NTP**）来同步系统时间，从而节省板空间。

在没有互联网连接的情况下，或者在线时间同步前的延迟无法接受，或者任何其他无数原因的情况下，人们可能会使用 SBC。在这种情况下，人们可能希望将 RTC 外围设备添加到板上，并配置操作系统以使用它。

# 添加 RTC

RTC 可以作为一个现成的模块以低价购买，通常基于 DS1307 芯片。这是一个 5V 模块，通过 I2C 总线连接到 SBC（或 MCU）：

![](img/5881b3bf-15ed-4189-ab38-619ca08aa8c8.png)

这张图片展示了一个基于 DS1307 的小型 RTC 模块。如图所示，它包含 RTC 芯片、晶体和 MCU。最后一个用于与主机系统通信，无论它是基于 SoC 还是 MCU 的板。所需的一切只是提供 RTC 模块运行的所需电压（和电流），以及 I2C 总线。

在将 RTC 模块连接到 SBC 板后，下一个目标就是让操作系统也使用它。为此，我们必须确保 I2C 内核模块已加载，这样我们才能使用 I2C 设备。

SBC 的 Linux 发行版，如 Raspbian 和 Armbian，通常包含许多 RTC 模块的驱动程序。这使得我们能够相对快速地设置 RTC 模块并将其集成到操作系统。对于之前查看的模块，我们需要 I2C 和 DS1307 内核模块。对于第一代 Raspberry Pi SBC 上的 Raspbian 操作系统，这些模块将被称为`i2c-dev`、`2cbcm2708`和`rtc-ds1307`。

首先，你必须启用这些模块，以便它们在系统启动时加载。对于 Raspbian Linux，可以通过编辑`/etc/modules`文件来实现，也可以使用为该平台提供的其他配置工具。重启后，我们应该能够使用 I2C 扫描工具检测到 I2C 总线上的 RTC 设备。

当 RTC 设备工作正常时，我们可以在 Raspbian 上移除 fake-hwclock 包。这是一个简单的模块，它模拟 RTC，但在系统关闭前仅将当前时间存储在文件中，以便在下次启动时，由于从存储的日期和时间恢复，文件系统日期和类似内容将保持一致，而不会出现新创建的文件突然比现有文件*更旧*。

相反，我们将使用 hwclock 实用程序，它将使用任何真实的 RTC 来同步系统时间。这需要修改操作系统启动的方式，将 RTC 模块的位置作为以下形式的引导参数传递：

```cpp
rtc.i2c=ds1307,1,0x68
```

这将在 I2C 总线上初始化一个 RTC (`/dev/rtc0`) 设备，地址为 0x68。

# 自定义驱动程序

驱动程序（内核模块）与操作系统内核的精确格式和集成方式因操作系统而异，因此在这里完全覆盖是不可能的。然而，我们将探讨我们之前使用的 RTC 模块的驱动程序是如何在 Linux 中实现的。

此外，我们将在本章后面，在俱乐部房间监控示例中，探讨如何从用户空间使用 I2C 外设。使用基于用户空间的驱动程序（库）通常是将其实现为内核模块的良好替代方案。

RTC 功能集成到 Linux 内核中，其代码位于 `/drivers/rtc` 文件夹中（在 GitHub 上，见 [`github.com/torvalds/linux/tree/master/drivers/rtc`](https://github.com/torvalds/linux/tree/master/drivers/rtc)）。

`rtc-ds1307.c` 文件包含两个我们需要分别读取和设置 RTC 的函数：`ds1307_get_time()` 和 `ds1307_set_time()`。这些函数的基本功能与我们将在本章后面讨论的俱乐部房间监控示例中将要使用的基本功能非常相似，在那里我们只是将 I2C 设备支持集成到我们的应用程序中。

从用户空间与 I2C、SPI 和其他此类外设通信的主要优势是我们不受操作系统内核支持的编译环境限制。以 Linux 内核为例，它主要用 C 语言编写，其中包含一些汇编代码。它的 API 是 C 风格的 API，因此我们必须使用明显是 C 风格的编码方法来编写我们的内核模块。

显然，这会抵消尝试最初用 C++ 编写这些模块的大部分优势，更不用说这个观点了。当我们把模块代码移动到用户空间并使用它作为应用程序的一部分或作为共享库时，我们就不再有这种限制，可以自由使用任何 C++ 概念和功能。

为了完整性，Linux 内核模块的基本模板如下：

```cpp
#include <linux/module.h>       // Needed by all modules 
#include <linux/kernel.h>       // Needed for KERN_INFO 

int init_module() { 
        printk(KERN_INFO "Hello world.n"); 

        return 0; 
} 

void cleanup_module() { 
        printk(KERN_INFO "Goodbye world.n"); 
} 
```

这是一个必要的 Hello World 示例，用 C++ 风格编写。

在考虑基于内核空间和用户空间的驱动模块时，还有一个需要考虑的最终因素，那就是上下文切换。从效率的角度来看，内核模块更快，延迟更低，因为 CPU 不需要反复在用户空间和内核空间之间切换以与设备通信并将消息从设备传递回与之通信的代码。

对于高带宽设备（如存储和捕获），这可能会在系统平稳运行和严重滞后、难以完成任务之间造成差异。

然而，当考虑本章中的俱乐部房间监控示例及其偶尔使用的 I2C 设备时，很明显，没有实际好处的情况下，使用内核模块将是过度设计。

# 资源限制

尽管 SBC 和 SoC 在近年来对于常见型号的 CPU 性能有了显著提升，但通常仍然建议使用交叉编译器在快速的桌面系统或服务器上为 SBC 生成代码。

由于（永久性安装的）RAM 的容量差异很大，你必须在考虑相对较慢的 CPU 性能之前，先考虑希望在该系统上运行的应用程序的内存需求。

由于 SBC 通常没有，或者只有少量具有高耐用率的存储（这意味着它可以经常写入，而不需要考虑有限的写入周期），它们通常没有交换空间，并将所有内容都保存在可用的 RAM 中。没有交换空间的备用方案，任何内存泄漏和过度使用内存都会迅速导致系统无法正常工作或不断重启。

尽管 SBC 上的 CPU 性能在近年来对于常见型号有了显著提升，但通常仍然建议在快速的桌面系统或服务器上使用交叉编译器为 SBC 生成代码。

更多关于开发问题和解决方案的内容将在第六章，*基于 OS 的应用程序测试*，和附录，*最佳实践*中介绍。

# 示例 - 俱乐部房间监控

在本节中，我们将探讨一个基于 SBC 的解决方案的实际实现，该解决方案为俱乐部房间执行以下功能：

+   监控俱乐部门锁的状态

+   监控俱乐部状态开关

+   通过 MQTT 发送状态变更通知

+   为当前俱乐部状态提供 REST API

+   控制状态灯

+   控制俱乐部房间的电源

在这里的基本用例是我们有一个俱乐部房间，我们希望能够监控其锁的状态，并在俱乐部内部有一个开关来调节俱乐部中非永久性电源插座是否通电。将俱乐部状态开关调至*开启*状态将为这些插座供电。我们还想通过 MQTT 发送通知，以便俱乐部房间或其他地方的设备可以更新其状态。

MQTT 是在 TCP/IP 之上简单二进制发布/订阅协议。它提供了一个轻量级的通信协议，适用于资源受限的应用程序，如传感器网络。每个 MQTT 客户端都与一个中央服务器通信：MQTT 代理。

# 硬件

`clubstatus`系统的框图如下所示：

![](img/bfb888d8-bf9f-4dab-9366-d473d1c7dd7f.png)

对于 SBC 平台，我们使用 Raspberry Pi，无论是 Raspberry Pi B+型号还是 B 系列的新成员，如 Raspberry Pi 3 Model B：

![](img/97073e5a-4311-4381-bf99-025dc76193c3.png)

我们在 SBC 系统中寻找的主要功能是一个以太网连接，当然，还有与 Raspberry Pi 兼容的**通用输入/输出**（**GPIO**）引脚。

使用此板，我们将在μSD 卡上使用标准的 Raspbian OS 安装。除了这个之外不需要特殊配置。选择 B+型号或类似型号的主要原因是因为这些具有标准的安装孔图案。

# 继电器

为了控制房间内的状态灯和非永久电源插座，我们使用多个继电器，在这种情况下是四个继电器：

| **继电器** | **功能** |
| --- | --- |
| 0 | 非永久插座电源状态 |
| 1 | 绿色状态灯 |
| 2 | 黄色状态灯 |
| 3 | 红色状态灯 |

这里的想法是，电源状态继电器连接到一个开关，该开关控制当俱乐部状态关闭时未供电的插座的主电源。状态灯指示当前的俱乐部状态。下一节提供了实现这一概念的具体细节。

为了简化设计，我们将使用一个包含四个继电器的现成继电器板，这些继电器由连接到树莓派单板计算机 I2C 总线的 NXP PCAL9535A I/O 端口芯片（GPIO 扩展器）驱动：

![](img/5853c932-146d-46f6-abc9-8537243bc361.png)

这个特定的板是 Seeed Studio 树莓派继电器板 v1.0：[`wiki.seeedstudio.com/Raspberry_Pi_Relay_Board_v1.0/`](http://wiki.seeedstudio.com/Raspberry_Pi_Relay_Board_v1.0/). 它提供了我们所需的四个继电器，允许我们切换高达 30 VDC（直流）或 250 VAC（交流）的灯光和开关。这使得可以连接几乎任何类型的照明和进一步的继电器等。

通过使用 SBC 的 GPIO 引脚堆叠继电器板，我们与 SBC 连接，这允许我们在继电器板上添加更多板。这使得我们可以将防抖功能添加到系统中，如接线图所示。

# 防抖

防抖板需要防抖开关信号，同时为树莓派板提供电源。机械开关防抖的理论和原因在于，这些开关提供的信号不干净，意味着它们不会立即从开到关切换。它们会在金属触点的弹性导致它们再次打开之前短暂闭合（接触），然后在这两种状态之间快速移动，最终稳定在其最终位置，正如我们可以在以下来自连接到简单开关的示波器的图中看到：

![](img/279f998d-afcf-427d-aeae-3b6f2fcc04e6.png)

这个属性的后果是，到达 SBC GPIO 引脚的信号将在几个毫秒内（或更糟）快速变化。因此，基于这些开关输入变化执行任何类型的操作都会导致巨大的问题，因为无法轻易区分所需的开关变化和在此变化期间开关触点快速弹跳。

可以通过硬件或软件去抖动一个开关。后者解决方案涉及在开关状态首次改变时启动计时器。背后的假设是在一定时间（以毫秒为单位）过后，开关处于稳定状态，可以安全地读取。这种方法的不利之处在于它通过占用一个或多个计时器或暂停程序的执行，给系统增加了额外的负担。此外，在开关输入上使用中断需要在一个计时器运行时禁用中断，这进一步增加了代码的复杂性。

硬件去抖动可以使用离散元件，或者使用 SR 锁存器（由两个与非门组成）。对于这个应用，我们将使用以下电路，它与最常用的 SPST（单刀单掷）开关类型配合得很好：

![](img/78c9ed38-dcb3-4eb3-85a1-733bf19c5bd6.png)

这个电路背后的概念是，当开关闭合时，电容器通过 R1（和 D1）充电，导致反相施密特触发器电路（U1）的输入变高，从而使连接到 U1 输出的 SBC 的 GPIO 引脚读取为低。当开关闭合时，电容器通过 R2 放电到地。

充电和放电都会花费一定的时间，这会在 U1 输入上注册变化之前增加延迟。充电和放电速率由 R1 和 R2 的值决定，其公式如下：

+   充电： ![](img/0ec96986-b5cd-4e73-9ad3-38681f477c37.png)

+   放电： ![](img/6472a5a6-f642-45ee-89c1-8826d4387364.png)

在这里，*V(t)* 是时间 *t*（以秒为单位）时的电压。*V[S]* 是源电压，*t* 是源电压施加后的时间（以秒为单位）。R 是电路电阻（以欧姆为单位），C 是电容（以法拉为单位）。最后，*e* 是一个数学常数，其值为 2.71828（约数），也称为欧拉数。

对于电容器的充电和放电，使用 RC 时间常数 tau (τ)，其定义如下：

![](img/4c70f6d7-152d-44d3-a793-6c3786e82007.png)

这定义了电容器充电到 63.2%（1τ），然后到 86%（2τ）所需的时间。从完全充电状态放电 1τ 将其电荷减少到 37%，2τ 后减少到 13.5%。在这里注意到的一件事是，电容器永远不会完全充电或放电；充电或放电的过程只是减慢到几乎不可察觉的程度。

使用我们为我们的去抖动电路所使用的值，我们得到以下充电时间常数：

![](img/32b180c1-b53b-4bb7-82ed-e324fb5f3094.png)

放电时间如下：

![](img/5f7819ea-e9e2-461d-a593-c4fdd7ba7a39.png)

这对应于 51 和 22 微秒，分别。

任何施密特触发器都有所谓的滞后，这意味着它有两个阈值。这实际上在输出响应的上下方增加了一个死区，输出将不会改变：

![](img/bc437d21-ffee-4425-9d2f-3184fc28df6d.png)

施密特触发器的滞后通常用于通过设置明确的触发电平来从输入信号中去除噪声。即使我们正在使用的 RC 电路应该过滤掉几乎所有噪声，添加施密特触发器也增加了额外的保险，而没有任何负面影响。

当可用时，也可以使用 SBC 的 GPIO 引脚的滞后功能。对于这个项目和所选的消抖电路，我们还想让芯片具有反相特性，以便我们得到预期的开关高/低响应，而不是需要在软件中反转其含义。

# 消抖 HAT

使用上一节中的信息和消抖电路，组装了一个原型板：

![](img/3c655ae5-9b38-479e-8178-6676cf9e0cd9.png)

这个原型实现了两个消抖通道，用于项目所需的两个开关。它还增加了一个螺钉端子，可以将 SBC 电源连接连接起来。这允许用户通过 5V 引脚头而不是使用 Raspberry Pi 的 micro-USB 连接器为 SBC 供电。为了集成目的，通常直接从电源将电线连接到螺钉端子或类似设备比在 micro-USB 插头上进行修补要容易。

当然，这个原型板不符合 Raspberry Pi 基金会规定的 HAT 定义。这些要求以下功能：

+   它连接到 Raspberry Pi SBC 上的`ID_SC`和`ID_SD` I2C 总线引脚的 EEPROM 包含有效的供应商信息、GPIO 映射和设备信息

+   它具有现代的 40 针（雌性）GPIO 连接器，并且将 HAT 与 SBC 之间的间距至少设置为 8 毫米

+   它遵循机械规范

+   如果通过 5V 引脚为 SBC 供电，HAT 必须能够连续提供至少 1.3 安培的电流

在添加所需的 I2C EEPROM（CAT24C32）和其他功能后，我们可以看到使用反相六通道施密特触发器 IC（40106）提供的六个通道的全版本看起来是什么样子：

![](img/f94f1673-7d99-4eb5-b684-35e094620374.png)

该 KiCad 项目的文件可以在作者的 GitHub 账户[`github.com/MayaPosch/DebounceHat`](https://github.com/MayaPosch/DebounceHat)中找到。随着通道数量的增加，将更多的开关、继电器和其他元素集成到系统中相对容易，可能使用各种输出高/低信号的传感器来监控窗户等事物。

# 电源

对于我们的项目，我们需要的电压是 Raspberry Pi 板的 5V 和通过继电器开关的灯的第二电压。我们选择的电源必须能够为 SBC 和灯提供足够的电力。对于前者，1-2 A 应该足够，后者则取决于使用的灯及其功率需求。

# 实现

监控服务将被实现为一个基本的 `systemd` 服务，这意味着操作系统启动时将启动该服务，并且可以使用所有常规的 systemd 工具来监控和重启该服务。

我们将有以下依赖项：

+   POCO

+   WiringPi

+   libmosquittopp（和 libmosquitto）

libmosquitto 依赖项（[`mosquitto.org/man/libmosquitto-3.html`](https://mosquitto.org/man/libmosquitto-3.html)）用于添加 MQTT 支持。libmosquittopp 依赖项是对基于 C 的 API 的封装，以提供基于类的接口，这使得将其集成到 C++ 项目中更加容易。

POCO 框架（[`pocoproject.org/`](https://pocoproject.org/)）是一组高度可移植的 C++ API，它提供了从网络相关函数（包括 HTTP）到所有常见低级函数的一切。在本项目中，将使用其 HTTP 服务器，以及其处理配置文件的支持。

最后，WiringPi ([`wiringpi.com/`](http://wiringpi.com/)) 是访问和使用 Raspberry Pi 和兼容系统上的 GPIO 头功能的既定标准头文件。它实现了与 I2C 设备和 UART 通信的 API，并使用 PWM 和数字引脚。在本项目中，它允许我们与继电器板和消抖板通信。

当前版本的此代码可以在作者的 GitHub 账户中找到：[`github.com/MayaPosch/ClubStatusService`](https://github.com/MayaPosch/ClubStatusService)。

我们将从主文件开始：

```cpp
#include "listener.h"

 #include <iostream>
 #include <string>

 using namespace std;

 #include <Poco/Util/IniFileConfiguration.h>
 #include <Poco/AutoPtr.h>
 #include <Poco/Net/HTTPServer.h>

 using namespace Poco::Util;
 using namespace Poco;
 using namespace Poco::Net;

 #include "httprequestfactory.h"
 #include "club.h"
```

在这里，我们包含了一些基本的 STL 功能，以及来自 POCO 的 HTTP 服务器和 `ini` 文件支持。监听器头文件用于我们的 MQTT 类，而 `httprequestfactory` 和 club 头文件分别用于 HTTP 服务器和主要的监控逻辑：

```cpp
int main(int argc, char* argv[]) {
          Club::log(LOG_INFO, "Starting ClubStatus server...");
          int rc;
          mosqpp::lib_init();

          Club::log(LOG_INFO, "Initialised C++ Mosquitto library.");

          string configFile;
          if (argc > 1) { configFile = argv[1]; }
          else { configFile = "config.ini"; }

          AutoPtr<IniFileConfiguration> config;
          try {
                config = new IniFileConfiguration(configFile);
          }
          catch (Poco::IOException &e) {
                Club::log(LOG_FATAL, "Main: I/O exception when opening configuration file: " + configFile + ". Aborting...");
                return 1;
          }

          string mqtt_host = config->getString("MQTT.host", "localhost");
          int mqtt_port = config->getInt("MQTT.port", 1883);
          string mqtt_user = config->getString("MQTT.user", "");
          string mqtt_pass = config->getString("MQTT.pass", "");
          string mqtt_topic = config->getString("MQTT.clubStatusTopic",    "/public/clubstatus");
          bool relayactive = config->getBool("Relay.active", true);
          uint8_t relayaddress = config->getInt("Relay.address", 0x20);
```

在本节中，我们初始化 MQTT 库（libmosquittopp），并尝试打开配置文件，如果命令行参数中没有指定，则使用默认路径和名称。

POCO 的 `IniFileConfiguration` 类用于打开和读取配置文件，如果找不到或无法打开，则会抛出异常。POCO 的 `AutoPtr` 与 C++11 的 `unique_ptr` 相当，允许我们创建一个新的基于堆的实例，而无需担心以后如何处理它。

接下来，我们将读取我们对 MQTT 和继电器板功能感兴趣的价值，并在合理的地方指定默认值：

```cpp
Listener listener("ClubStatus", mqtt_host, mqtt_port, mqtt_user, mqtt_pass);

    Club::log(LOG_INFO, "Created listener, entering loop...");

    UInt16 port = config->getInt("HTTP.port", 80);
    HTTPServerParams* params = new HTTPServerParams;
    params->setMaxQueued(100);
    params->setMaxThreads(10);
    HTTPServer httpd(new RequestHandlerFactory, port, params);
    try {
          httpd.start();
    }
    catch (Poco::IOException &e) {
          Club::log(LOG_FATAL, "I/O Exception on HTTP server: port already in use?");
          return 1;
    }
    catch (...) {
          Club::log(LOG_FATAL, "Exception thrown for HTTP server start. Aborting.");
          return 1;
    }
```

在本节中，我们启动 MQTT 类，向其提供连接到 MQTT 代理所需的参数。接下来，读取 HTTP 服务器的配置细节并创建一个新的`HTTPServer`实例。

服务器实例配置了提供的端口和一些限制，即 HTTP 服务器允许使用的最大线程数以及它可以保持的最大队列连接数。这些参数对于优化系统性能并将代码放入资源较少的系统中有用。

新的客户端连接由自定义的`RequestHandlerFactory`类处理，我们稍后会看到：

```cpp

             Club::mqtt = &listener;
             Club::start(relayactive, relayaddress, mqtt_topic);

             while(1) {
                   rc = listener.loop();
                   if (rc){
                         Club::log(LOG_ERROR, "Disconnected. Trying to 
                         reconnect...");
                         listener.reconnect();
                   }
             }

             mosqpp::lib_cleanup();
             httpd.stop();
             Club::stop();

             return 0;
 }
```

最后，我们将我们创建的 Listener 实例的引用分配给静态`Club`类的`mqtt`成员。这将使得`Listener`对象在以后更容易使用，正如我们将看到的。

通过在`Club`上调用`start()`，将处理连接硬件的监控和配置，我们在主函数中完成这一方面。

最后，我们进入 MQTT 类的循环，确保它保持连接到 MQTT 代理。在离开循环时，我们将清理资源并停止 HTTP 服务器和其他服务。然而，由于我们在这里处于无限循环中，这段代码将不会在这个实现中被达到。

由于这个实现将以 24/7 运行的服务形式运行，因此提供一个干净地终止服务的方法不是绝对必要的。一个相对简单的方法是添加一个信号处理器，一旦触发就会中断循环。为了简单起见，这个项目已经省略了这一点。

# Listener

`Listener`类的类声明如下：

```cpp
class Listener : public mosqpp::mosquittopp {
          //

 public:
          Listener(string clientId, string host, int port, string user, string pass);
          ~Listener();

          void on_connect(int rc);
          void on_message(const struct mosquitto_message* message);
          void on_subscribe(int mid, int qos_count, const int* granted_qos);

          void sendMessage(string topic, string& message);
          void sendMessage(string& topic, char* message, int msgLength);
 };
```

这个类提供了一个简单的 API 来连接到 MQTT 代理并向该代理发送消息。我们继承自`mosquittopp`类，重新实现了多个回调方法来处理连接新接收的消息和完成对 MQTT 主题的订阅的事件。

接下来，让我们看看实现：

```cpp
#include "listener.h"

 #include <iostream>

 using namespace std;
 Listener::Listener(string clientId, string host, int port, string user, string pass) : mosquittopp(clientId.c_str()) {
          int keepalive = 60;
          username_pw_set(user.c_str(), pass.c_str());
          connect(host.c_str(), port, keepalive);
 }

 Listener::~Listener() {
          //
 }
```

在构造函数中，我们使用 mosquittopp 类的构造函数分配唯一的 MQTT 客户端标识字符串。我们为保持连接设置使用默认值 60 秒，这意味着我们将保持与 MQTT 代理的连接打开，在此期间没有任何端点发送控制或其他消息。

在设置用户名和密码后，我们连接到 MQTT 代理：

```cpp
void Listener::on_connect(int rc) {
    cout << "Connected. Subscribing to topics...n";

          if (rc == 0) {
                // Subscribe to desired topics.
                string topic = "/club/status";
                subscribe(0, topic.c_str(), 1);
          }
          else {
                cerr << "Connection failed. Aborting subscribing.n";
          }
 }
```

这个回调函数在尝试连接到 MQTT 代理时被调用。我们检查`rc`的值，如果值为零——表示成功——我们开始订阅任何所需的主题。在这里，我们只订阅一个主题：/club/status。如果任何其他 MQTT 客户端向此主题发送消息，我们将在以下回调函数中接收到它：

```cpp

 void Listener::on_message(const struct mosquitto_message* message) {
          string topic = message->topic;
          string payload = string((const char*) message->payload, message->payloadlen);

          if (topic == "/club/status") {
                string topic = "/club/status/response";
                char payload[] = { 0x01 }; 
                publish(0, topic.c_str(), 1, payload, 1); // QoS 1\.   
          }     
 }
```

在这个回调函数中，我们接收一个包含 MQTT 主题和负载的结构体。然后我们将主题与我们所订阅的主题字符串进行比较，在这个例子中就是 /club/status 主题。在接收到这个主题的消息后，我们发布一个新的 MQTT 消息，包含主题和负载。最后一个参数是服务质量（**QoS**）值，在这个例子中设置为 *至少发送一次* 标志。这保证了至少有一个其他的 MQTT 客户端会接收到我们的消息。

MQTT 负载始终是二进制的，即在这个例子中是 `1`。为了使其反映俱乐部房间的状态（开启或关闭），我们必须集成来自静态 `Club` 类的响应，我们将在下一节中查看。

首先，我们来看 `Listener` 类的剩余函数：

```cpp
 void Listener::on_subscribe(int mid, int qos_count, const int* granted_qos) {
          // 
 }

 void Listener::sendMessage(string topic, string &message) {
          publish(0, topic.c_str(), message.length(), message.c_str(), true);
 }

 void Listener::sendMessage(string &topic, char* message, int msgLength) {
          publish(0, topic.c_str(), msgLength, message, true);
 }
```

这里留空的新的订阅回调函数，但可以用来添加日志记录或其他功能。此外，我们还有一个重载的 `sendMessage()` 函数，它允许应用程序的其他部分也可以发布 MQTT 消息。

有这两个不同函数的主要原因是，有时使用 `char*` 数组发送数据更容易，例如，作为二进制协议的一部分发送一个 8 位整数的数组，而有时 STL 字符串更方便。这样，我们就能同时获得两者的优点，而无需在需要将 MQTT 消息发送到代码中的任何位置时进行转换。

`publish()` 函数的第一个参数是消息 ID，这是一个我们可以自行分配的自定义整数。在这里，我们将其保留为零。我们还使用了 *保留* 标志（最后一个参数），将其设置为 true。这意味着每当一个新的 MQTT 客户端订阅了我们发布保留消息的主题时，该客户端总是会接收到该特定主题上最后发布的消息。

由于我们将通过 MQTT 主题发布俱乐部房间的状态，因此希望 MQTT 代理保留最后的状态消息，这样任何使用这些信息的客户端在连接到代理时都能立即接收到当前状态，而无需等待下一次状态更新。

# 俱乐部

俱乐部头文件声明了构成项目核心的类，并负责处理开关的输入、控制继电器以及更新俱乐部房间的状态：

```cpp
#include <wiringPi.h>
 #include <wiringPiI2C.h>
```

在这个头文件中，首先值得注意的是包含部分。它们为我们添加了基本的 WiringPi GPIO 功能，以及用于 I2C 使用的功能。对于其他需要此类功能的项目，还可以包括 SPI、UART（串行）、软件 PWM、Raspberry Pi（Broadcom SoC）特定功能以及其他功能：

```cpp
enum Log_level {
    LOG_FATAL = 1,
    LOG_ERROR = 2,
    LOG_WARNING = 3,
    LOG_INFO = 4,
    LOG_DEBUG = 5
 };
```

我们定义了我们将要使用的不同日志级别，作为一个 `enum`：

```cpp
 class Listener;
```

我们提前声明了 `Listener` 类，因为我们将在这些类的实现中使用它，但还不希望包含它的整个头文件：

```cpp
class ClubUpdater : public Runnable {
          TimerCallback<ClubUpdater>* cb;
          uint8_t regDir0;
          uint8_t regOut0;
          int i2cHandle;
          Timer* timer;
          Mutex mutex;
          Mutex timerMutex;
          Condition timerCnd;
          bool powerTimerActive;
          bool powerTimerStarted;

 public:
          void run();
          void updateStatus();
          void writeRelayOutputs();
          void setPowerState(Timer &t);
 };
```

`ClubUpdater`类负责配置基于 I2C 的 GPIO 扩展器，该扩展器控制继电器，以及处理任何关于俱乐部状态的更新。使用 POCO 框架中的`Timer`实例来为电源状态继电器添加延迟，正如我们在查看实现时将看到的那样。

这个类继承自 POCO 的`Runnable`类，这是 POCO `Thread`类期望的基类，而`Thread`类是原生线程的包装器。

这两个`uint8_t`成员变量反映了 I2C GPIO 扩展器设备上的两个寄存器，允许我们设置设备上输出引脚的方向和值，这实际上控制了连接的继电器：

```cpp
class Club {
          static Thread updateThread;
          static ClubUpdater updater;

          static void lockISRCallback();
          static void statusISRCallback();

 public:
          static bool clubOff;
          static bool clubLocked;
          static bool powerOn;
          static Listener* mqtt;
          static bool relayActive;
          static uint8_t relayAddress;
          static string mqttTopic;      // Topic we publish status updates on.

          static Condition clubCnd;
          static Mutex clubCndMutex;
          static Mutex logMutex;
          static bool clubChanged ;
          static bool running;
          static bool clubIsClosed;
          static bool firstRun;
          static bool lockChanged;
          static bool statusChanged;
          static bool previousLockValue;
          static bool previousStatusValue;

          static bool start(bool relayactive, uint8_t relayaddress, string topic);
          static void stop();
          static void setRelay();
          static void log(Log_level level, string msg);
 };
```

`Club`类可以被视为系统的输入端，设置和处理 ISRs（中断处理程序），同时作为包含所有与俱乐部状态相关的变量（如锁开关状态、状态开关状态和电源系统状态（俱乐部开启或关闭））的中心（静态）类。

这个类被完全设置为静态，以便程序的不同部分可以自由地使用它来查询房间状态。

接下来，这是实现部分：

```cpp
#include "club.h"

 #include <iostream>

 using namespace std;

 #include <Poco/NumberFormatter.h>

 using namespace Poco;

 #include "listener.h"
```

在这里，我们包含了`Listener`头文件，以便我们可以使用它。我们还包含了 POCO 的`NumberFormatter`类，以便我们可以格式化整数值以供日志记录：

```cpp
 #define REG_INPUT_PORT0              0x00
 #define REG_INPUT_PORT1              0x01
 #define REG_OUTPUT_PORT0             0x02
 #define REG_OUTPUT_PORT1             0x03
 #define REG_POL_INV_PORT0            0x04
 #define REG_POL_INV_PORT1            0x05
 #define REG_CONF_PORT0               0x06
 #define REG_CONG_PORT1               0x07
 #define REG_OUT_DRV_STRENGTH_PORT0_L 0x40
 #define REG_OUT_DRV_STRENGTH_PORT0_H 0x41
 #define REG_OUT_DRV_STRENGTH_PORT1_L 0x42
 #define REG_OUT_DRV_STRENGTH_PORT1_H 0x43
 #define REG_INPUT_LATCH_PORT0        0x44
 #define REG_INPUT_LATCH_PORT1        0x45
 #define REG_PUD_EN_PORT0             0x46
 #define REG_PUD_EN_PORT1             0x47
 #define REG_PUD_SEL_PORT0            0x48
 #define REG_PUD_SEL_PORT1            0x49
 #define REG_INT_MASK_PORT0           0x4A
 #define REG_INT_MASK_PORT1           0x4B
 #define REG_INT_STATUS_PORT0         0x4C
 #define REG_INT_STATUS_PORT1         0x4D
 #define REG_OUTPUT_PORT_CONF         0x4F
```

接下来，我们定义了目标 GPIO 扩展器设备 NXP PCAL9535A 的所有寄存器。尽管我们只使用了这些寄存器中的两个，但通常将完整的列表添加进来是一个很好的实践，这样可以简化后续代码的扩展。也可以使用单独的头文件，以便于在不进行重大更改或完全不更改代码的情况下轻松使用不同的 GPIO 扩展器：

```cpp
 #define RELAY_POWER 0
 #define RELAY_GREEN 1
 #define RELAY_YELLOW 2
 #define RELAY_RED 3
```

在这里，我们定义了哪些功能连接到哪个继电器，对应于 GPIO 扩展器芯片的特定输出引脚。由于我们有四个继电器，因此使用了四个引脚。这些引脚连接到芯片上的第一个银行（总共两个银行）的八个引脚。

自然地，这些定义与物理连接到继电器的部分相匹配是很重要的。根据使用情况，也可以将其设置为可配置的：

```cpp
bool Club::clubOff;
 bool Club::clubLocked;
 bool Club::powerOn;
 Thread Club::updateThread;
 ClubUpdater Club::updater;
 bool Club::relayActive;
 uint8_t Club::relayAddress;
 string Club::mqttTopic;
 Listener* Club::mqtt = 0;

 Condition Club::clubCnd;
 Mutex Club::clubCndMutex;
 Mutex Club::logMutex;
 bool Club::clubChanged = false;
 bool Club::running = false;
 bool Club::clubIsClosed = true;
 bool Club::firstRun = true;
 bool Club::lockChanged = false;
 bool Club::statusChanged = false;
 bool Club::previousLockValue = false;
 bool Club::previousStatusValue = false;
```

由于`Club`是一个完全静态的类，我们在进入`ClubUpdater`类的实现之前初始化了其所有成员变量：

```cpp
void ClubUpdater::run() {
    regDir0 = 0x00;
    regOut0 = 0x00;
    Club::powerOn = false;
    powerTimerActive = false;
    powerTimerStarted = false;
    cb = new TimerCallback<ClubUpdater>(*this, &ClubUpdater::setPowerState);
    timer = new Timer(10 * 1000, 0);
```

当我们启动这个类的实例时，它的`run()`函数会被调用。在这里，我们设置了一些默认值。方向和输出寄存器变量最初被设置为零。俱乐部房间的电源状态被设置为 false，与电源计时器相关的布尔值也被设置为 false，因为电源计时器尚未激活。这个计时器用于在打开或关闭电源之前设置一个延迟，正如我们稍后将更详细地看到的那样。

默认情况下，这个计时器的延迟是十秒。当然，这也可以设置为可配置的：

```cpp
if (Club::relayActive) {
    Club::log(LOG_INFO, "ClubUpdater: Starting i2c relay device.");
    i2cHandle = wiringPiI2CSetup(Club::relayAddress);
    if (i2cHandle == -1) {
        Club::log(LOG_FATAL, string("ClubUpdater: error starting          
        i2c relay device."));
        return;
    }

    wiringPiI2CWriteReg8(i2cHandle, REG_CONF_PORT0, 0x00);
    wiringPiI2CWriteReg8(i2cHandle, REG_OUTPUT_PORT0, 0x00);

    Club::log(LOG_INFO, "ClubUpdater: Finished configuring the i2c 
    relay device's registers.");
}
```

接下来，我们设置 I2C GPIO 扩展器。这需要 I2C 设备地址，我们之前将其传递给了`Club`类。此设置函数的作用是确保在 I2C 总线上存在一个活动 I2C 设备在该地址。在此之后，它应该准备好进行通信。也可以通过将`relayActive`变量设置为 false 来跳过此步骤。这是通过在配置文件中设置适当的值来完成的，这在在没有 I2C 总线或连接设备的情况下运行集成测试时很有用。

完成设置后，我们为第一个存储器的方向寄存器和输出寄存器写入初始值。这两个寄存器都使用空字节写入，以确保它们控制的八个引脚都设置为输出模式，并且处于二进制零（低）状态。这样，连接到前四个引脚的所有继电器最初都是关闭的：

```cpp
          updateStatus();

          Club::log(LOG_INFO, "ClubUpdater: Initial status update complete.");
          Club::log(LOG_INFO, "ClubUpdater: Entering waiting condition.");

          while (Club::running) {
                Club::clubCndMutex.lock();
                if (!Club::clubCnd.tryWait(Club::clubCndMutex, 60 * 1000)) {.
                      Club::clubCndMutex.unlock();
                      if (!Club::clubChanged) { continue; }
                }
                else {
                      Club::clubCndMutex.unlock();
                }

                updateStatus();
          }
 }
```

完成这些配置步骤后，我们使用稍后当输入改变时也将调用的相同函数运行第一个俱乐部房间状态的更新。这导致检查所有输入，并将输出设置为相应的状态。

最后，我们进入一个等待循环。此循环由`Club::running`布尔变量控制，允许我们通过信号处理程序或类似方式从中退出。实际的等待是通过一个条件变量来执行的，我们在这里等待，直到一分钟等待超时（之后，我们在快速检查后返回等待），或者我们被稍后将为输入设置的其中一个中断之一所信号。

接下来，我们看看用于更新输出状态的函数：

```cpp
void ClubUpdater::updateStatus() {
    Club::clubChanged = false;

    if (Club::lockChanged) {
          string state = (Club::clubLocked) ? "locked" : "unlocked";
          Club::log(LOG_INFO, string("ClubUpdater: lock status changed to ") + state);
          Club::lockChanged = false;

          if (Club::clubLocked == Club::previousLockValue) {
                Club::log(LOG_WARNING, string("ClubUpdater: lock interrupt triggered, but value hasn't changed. Aborting."));
                return;
          }

          Club::previousLockValue = Club::clubLocked;
    }
    else if (Club::statusChanged) {           
          string state = (Club::clubOff) ? "off" : "on";
          Club::log(LOG_INFO, string("ClubUpdater: status switch status changed to ") + state);
          Club::statusChanged = false;

          if (Club::clubOff == Club::previousStatusValue) {
                Club::log(LOG_WARNING, string("ClubUpdater: status interrupt triggered, but value hasn't changed. Aborting."));
                return;
          }

          Club::previousStatusValue = Club::clubOff;
    }
    else if (Club::firstRun) {
          Club::log(LOG_INFO, string("ClubUpdater: starting initial update run."));
          Club::firstRun = false;
    }
    else {
          Club::log(LOG_ERROR, string("ClubUpdater: update triggered, but no change detected. Aborting."));
          return;
    }
```

当我们进入此更新函数时，我们首先确保`Club::clubChanged`布尔值设置为 false，以便它可以由其中一个中断处理程序再次设置。

然后，我们检查输入中确切发生了什么变化。如果锁开关被触发，其布尔变量将被设置为 true，或者状态开关的变量可能已经被触发。如果是这种情况，我们将重置变量，并将新读取的值与该输入的已知最后值进行比较。

作为一种合理性检查，如果值没有变化，我们将忽略触发。这可能发生在由于噪声而触发中断的情况下，例如当开关的信号线靠近电源线时。后者的任何波动都会在前者中引起浪涌，从而触发 GPIO 引脚的中断。这是处理非理想物理世界现实的一个明显例子，也是硬件和软件如何影响系统可靠性的重要性的展示。

除了这个检查之外，我们还使用我们的中央记录器记录事件，并更新缓冲的输入值，以便在下次运行中使用。

if/else 语句中的最后两个情况处理初始运行以及默认处理程序。当我们最初以我们之前看到的方式运行此函数时，没有触发中断，因此显然我们必须为状态和锁定开关添加第三种情况：

```cpp
    if (Club::clubIsClosed && !Club::clubOff) {
          Club::clubIsClosed = false;

          Club::log(LOG_INFO, string("ClubUpdater: Opening club."));

          Club::powerOn = true;
          try {
                if (!powerTimerStarted) {
                      timer->start(*cb);
                      powerTimerStarted = true;
                }
                else { 
                      timer->stop();
                      timer->start(*cb);
                }
          }
          catch (Poco::IllegalStateException &e) {
                Club::log(LOG_ERROR, "ClubUpdater: IllegalStateException on timer start: " + e.message());
                return;
          }
          catch (...) {
                Club::log(LOG_ERROR, "ClubUpdater: Unknown exception on timer start.");
                return;
          }

          powerTimerActive = true;

          Club::log(LOG_INFO, "ClubUpdater: Started power timer...");

          char msg = { '1' };
          Club::mqtt->sendMessage(Club::mqttTopic, &msg, 1);

          Club::log(LOG_DEBUG, "ClubUpdater: Sent MQTT message.");
    }
    else if (!Club::clubIsClosed && Club::clubOff) {
          Club::clubIsClosed = true;

          Club::log(LOG_INFO, string("ClubUpdater: Closing club."));

          Club::powerOn = false;

          try {
                if (!powerTimerStarted) {
                      timer->start(*cb);
                      powerTimerStarted = true;
                }
                else { 
                      timer->stop();
                      timer->start(*cb);
                }
          }
          catch (Poco::IllegalStateException &e) {
                Club::log(LOG_ERROR, "ClubUpdater: IllegalStateException on timer start: " + e.message());
                return;
          }
          catch (...) {
                Club::log(LOG_ERROR, "ClubUpdater: Unknown exception on timer start.");
                return;
          }

          powerTimerActive = true;

          Club::log(LOG_INFO, "ClubUpdater: Started power timer...");

          char msg = { '0' };
          Club::mqtt->sendMessage(Club::mqttTopic, &msg, 1);

          Club::log(LOG_DEBUG, "ClubUpdater: Sent MQTT message.");
    }
```

接下来，我们检查是否需要将俱乐部房间的状态从关闭更改为开启，或者相反。这是通过检查俱乐部状态（`Club::clubOff`）布尔值相对于存储最后已知状态的 `Club::clubIsClosed` 布尔值是否发生变化来确定的。

实际上，如果状态开关从开启更改为关闭或相反，这将被检测到，并开始新的状态更改。这意味着将启动一个电源定时器，该定时器将在预设延迟后打开或关闭俱乐部房间中的非永久电源。

POCO `Timer` 类要求我们在启动定时器之前先停止定时器，如果它之前已经被启动。这需要我们添加一个额外的检查。

此外，我们还使用我们参考的 MQTT 客户端类将更新后的俱乐部房间状态的消息发送到 MQTT 代理，这里为 ASCII 1 或 0。这条消息可以用来触发其他系统，这些系统可以更新俱乐部房间的在线状态，或者用于更富有创意的应用。

当然，消息的确切负载可以设置为可配置的。

在下一节中，我们将更新状态灯的颜色，考虑到房间中的电源状态。为此，我们使用以下表格：

| **颜色** | **状态切换** | **锁定切换** | **电源状态** |
| --- | --- | --- | --- |
| 绿色 | 开启 | 未锁定 | 开启 |
| 黄色 | 关闭 | 未锁定 | 关闭 |
| 红色 | 关闭 | 锁定 | 关闭 |
| 黄色和红色 | 开启 | 锁定 | 开启 |

这里是实现的示例：

```cpp

    if (Club::clubOff) {
          Club::log(LOG_INFO, string("ClubUpdater: New lights, clubstatus off."));

          mutex.lock();
          string state = (Club::powerOn) ? "on" : "off";
          if (powerTimerActive) {
                Club::log(LOG_DEBUG, string("ClubUpdater: Power timer active, inverting power state from: ") + state);
                regOut0 = !Club::powerOn;
          }
          else {
                Club::log(LOG_DEBUG, string("ClubUpdater: Power timer not active, using current power state: ") + state);
                regOut0 = Club::powerOn; 
          }

          if (Club::clubLocked) {
                Club::log(LOG_INFO, string("ClubUpdater: Red on."));
                regOut0 |= (1UL << RELAY_RED); 
          } 
          else {
                Club::log(LOG_INFO, string("ClubUpdater: Yellow on."));
                regOut0 |= (1UL << RELAY_YELLOW);
          } 

          Club::log(LOG_DEBUG, "ClubUpdater: Changing output register to: 0x" + NumberFormatter::formatHex(regOut0));

          writeRelayOutputs();
          mutex.unlock();
    }
```

我们首先检查俱乐部房间电源的状态，这告诉我们应该使用输出寄存器的第一个位值。如果电源定时器处于活动状态，我们必须反转电源状态，因为我们想写入当前的电源状态，而不是存储在电源状态布尔值中的未来状态。

如果俱乐部房间的状态开关处于关闭位置，那么锁定开关的状态将决定最终的颜色。当俱乐部房间被锁定时，我们触发红色继电器，否则触发黄色继电器。后者表示中间状态，即俱乐部房间关闭但尚未锁定。

在这里使用互斥锁是为了确保以同步方式写入 I2C 设备的输出寄存器以及更新本地寄存器变量：

```cpp
    else { 
                Club::log(LOG_INFO, string("ClubUpdater: New lights, clubstatus on."));

                mutex.lock();
                string state = (Club::powerOn) ? "on" : "off";
                if (powerTimerActive) {
                      Club::log(LOG_DEBUG, string("ClubUpdater: Power timer active,    inverting power state from: ") + state);
                      regOut0 = !Club::powerOn; // Take the inverse of what the timer    callback will set.
                }
                else {
                      Club::log(LOG_DEBUG, string("ClubUpdater: Power timer not active,    using current power state: ") + state);
                      regOut0 = Club::powerOn; // Use the current power state value.
                }

                if (Club::clubLocked) {
                      Club::log(LOG_INFO, string("ClubUpdater: Yellow & Red on."));
                      regOut0 |= (1UL << RELAY_YELLOW);
                      regOut0 |= (1UL << RELAY_RED);
                }
                else {
                      Club::log(LOG_INFO, string("ClubUpdater: Green on."));
                      regOut0 |= (1UL << RELAY_GREEN);
                }

                Club::log(LOG_DEBUG, "ClubUpdater: Changing output register to: 0x" +    NumberFormatter::formatHex(regOut0));

                writeRelayOutputs();
                mutex.unlock();
          }
 }
```

如果俱乐部房间的状态开关设置为开启，我们将得到另外两种颜色选择，绿色通常是其中之一，它表示俱乐部房间未锁定且状态开关已启用。然而，如果后者开启但房间被锁定，我们将得到黄色和红色。

在完成输出寄存器的新内容后，我们始终使用 `writeRelayOutputs()` 函数将我们的本地版本写入远程设备，从而触发新的继电器状态：

```cpp
void ClubUpdater::writeRelayOutputs() {
    wiringPiI2CWriteReg8(i2cHandle, REG_OUTPUT_PORT0, regOut0);

    Club::log(LOG_DEBUG, "ClubUpdater: Finished writing relay outputs with: 0x" 
                + NumberFormatter::formatHex(regOut0));
 }
```

此函数非常简单，并使用 WiringPi 的 I2C API 将单个 8 位值写入连接设备的输出寄存器。我们还在这里记录写入的值：

```cpp
   void ClubUpdater::setPowerState(Timer &t) {
          Club::log(LOG_INFO, string("ClubUpdater: setPowerState called."));

          mutex.lock();
          if (Club::powerOn) { regOut0 |= (1UL << RELAY_POWER); }
          else { regOut0 &= ~(1UL << RELAY_POWER); }

          Club::log(LOG_DEBUG, "ClubUpdater: Writing relay with: 0x" +    NumberFormatter::formatHex(regOut0));

          writeRelayOutputs();

          powerTimerActive = false;
          mutex.unlock();
 }
```

在此函数中，我们将俱乐部房间的电源状态设置为布尔变量包含的任何值。我们使用与更新俱乐部房间状态颜色时相同的互斥锁。然而，我们在这里不是从头开始创建输出寄存器的内容，而是选择切换其变量的第一个位。

在切换此位后，我们像往常一样写入远程设备，这将导致俱乐部房间的电源状态切换。

接下来，我们看看静态 `Club` 类，从我们调用的第一个初始化函数开始：

```cpp
bool Club::start(bool relayactive, uint8_t relayaddress, string topic) {
          Club::log(LOG_INFO, "Club: starting up...");

          relayActive = relayactive;
          relayAddress = relayaddress;
          mqttTopic = topic;

          wiringPiSetup();

          Club::log(LOG_INFO,  "Club: Finished wiringPi setup.");

          pinMode(0, INPUT);
          pinMode(7, INPUT);
          pullUpDnControl(0, PUD_DOWN);
          pullUpDnControl(7, PUD_DOWN);
          clubLocked = digitalRead(0);
          clubOff = !digitalRead(7);

          previousLockValue = clubLocked;
          previousStatusValue = clubOff;

          Club::log(LOG_INFO, "Club: Finished configuring pins.");

          wiringPiISR(0, INT_EDGE_BOTH, &lockISRCallback);
          wiringPiISR(7, INT_EDGE_BOTH, &statusISRCallback);

          Club::log(LOG_INFO, "Club: Configured interrupts.");

          running = true;
          updateThread.start(updater);

          Club::log(LOG_INFO, "Club: Started update thread.");

          return true;
 }
```

使用此函数，我们启动整个俱乐部监控系统，正如我们在应用程序入口点之前所看到的。它接受一些参数，允许我们打开或关闭继电器功能，继电器的 I2C 地址（如果使用继电器），以及用于发布俱乐部房间状态变化的 MQTT 主题。

使用这些参数设置成员变量的值后，我们初始化 WiringPi 框架。WiringPi 提供了多种不同的初始化函数，这些函数基本上在如何访问 GPIO 引脚方面有所不同。

我们在这里使用的 `wiringPiSetup()` 函数通常是使用起来最方便的，因为它将使用虚拟引脚号，这些引脚号映射到底层的 Broadcom SoC 引脚。WiringPi 编号的主要优势是它在不同版本的 Raspberry Pi SBC 之间保持不变。

使用 Broadcom (BCM) 号码或 SBC 电路板上的引脚在引脚上的物理位置，我们面临的风险是这可能在板修订之间发生变化，但 WiringPi 编号方案可以对此进行补偿。

对于我们的目的，我们在 SBC 上使用以下引脚：

|  | **锁开关** | **状态开关** |
| --- | --- | --- |
| BCM | 17 | 4 |
| 物理位置 | 11 | 7 |
| WiringPi | 0 | 7 |

在初始化 WiringPi 库之后，我们设置所需的引脚模式，将我们的两个引脚都设置为输入。然后我们为每个引脚启用下拉。这将在 SoC 中启用一个内置的下拉电阻，它将始终尝试将输入信号拉低（相对于地）。是否需要为输入（或输出）引脚启用下拉或上拉电阻取决于情况，特别是连接的电路。

重要的是要查看连接电路的行为；如果连接电路有使线上值“浮动”的倾向，这将在输入引脚上引起不良行为，值随机变化。通过将线路拉低或拉高，我们可以确信我们在引脚上读取的不是噪声。

在我们向 `ClubUpdate` 的 `run` 循环等待的条件变量上发出信号之前，我们先为两个引脚注册我们的中断方法。

中断处理程序不过是一个回调，当指定的引脚上发生指定的事件时会被调用。WiringPi ISR 函数接受引脚号、事件类型以及我们希望使用的处理程序函数的引用。对于这里选择的事件类型，我们的中断处理程序将在输入引脚的值从高变低或从低变高时被触发。这意味着当连接的开关从开启变为关闭，或从关闭变为开启时，它会被触发。

最后，我们通过使用 `ClubUpdater` 类实例并将其推入其自己的线程来启动更新线程：

```cpp
void Club::stop() {
          running = false;
 }
```

调用此函数将允许 `ClubUpdater` 的 `run()` 函数中的循环结束，这将终止它运行的线程，允许应用程序的其余部分安全关闭：

```cpp
void Club::lockISRCallback() {
          clubLocked = digitalRead(0);
          lockChanged = true;

          clubChanged = true;
          clubCnd.signal();
 }

 void Club::statusISRCallback() {
          clubOff = !digitalRead(7);
          statusChanged = true;

          clubChanged = true;
          clubCnd.signal();
 }
```

我们的两个中断处理程序都非常简单。当操作系统接收到中断时，它会触发相应的中断处理程序，这会导致它们读取输入引脚的当前值，并根据需要反转该值。将 `statusChanged` 或 `lockChanged` 变量设置为 true 以指示更新函数哪个中断被触发。

在发出信号之前，我们同样对 `clubChanged` 布尔变量做同样的处理。

这个类的最后一部分是日志函数：

```cpp
void Club::log(Log_level level, string msg) {
    logMutex.lock();
    switch (level) {
          case LOG_FATAL: {
                cerr << "FATAL:t" << msg << endl;
                string message = string("ClubStatus FATAL: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/fatal", message);
                }

                break;
          }
          case LOG_ERROR: {
                cerr << "ERROR:t" << msg << endl;
                string message = string("ClubStatus ERROR: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/error", message);
                }

                break;
          }
          case LOG_WARNING: {
                cerr << "WARNING:t" << msg << endl;
                string message = string("ClubStatus WARNING: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/warning", message);
                }

                break;
          }
          case LOG_INFO: {
                cout << "INFO: t" << msg << endl;
                string message = string("ClubStatus INFO: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/info", message);
                }

                break;
          }
          case LOG_DEBUG: {
                cout << "DEBUG:t" << msg << endl;
                string message = string("ClubStatus DEBUG: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/debug", message);
                }

                break;
          }
          default:
                break;
    }

    logMutex.unlock();
 }
```

我们在这里使用另一个互斥锁来同步系统日志（或控制台）中的日志输出，并防止当应用程序的不同部分同时调用此函数时并发访问 MQTT 类。正如我们稍后将看到的，这个日志函数也被其他类使用。

使用这个日志函数，我们可以在本地（系统日志）和远程使用 MQTT 进行日志记录。

# HTTP 请求处理程序

每当 POCO 的 HTTP 服务器接收到一个新的客户端连接时，它都会使用我们 `RequestHandlerFactory` 类的一个新实例来获取特定请求的处理程序。因为这个类非常简单，所以它完全在头文件中实现：

```cpp
#include <Poco/Net/HTTPRequestHandlerFactory.h>
 #include <Poco/Net/HTTPServerRequest.h>

 using namespace Poco::Net;

 #include "statushandler.h"
 #include "datahandler.h"

 class RequestHandlerFactory: public HTTPRequestHandlerFactory { 
 public:
          RequestHandlerFactory() {}
          HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& request) {
                if (request.getURI().compare(0, 12, "/clubstatus/") == 0) { 
                     return new StatusHandler(); 
               }
                else { return new DataHandler(); }
          }
 };
```

我们这个类所做的不仅仅是比较 HTTP 服务器提供的 URL，以确定要实例化哪种类型的处理程序并返回。在这里，我们可以看到如果 URL 字符串以 `/clubstatus` 开头，我们返回状态处理程序，该处理程序实现了 REST API。

默认处理程序是一个简单的文件服务器，它试图将请求解释为文件名，正如我们稍后将看到的。

# 状态处理程序

此处理程序实现了一个简单的 REST API，返回包含当前俱乐部状态的 JSON 结构。这可以被外部应用程序用来显示系统上的实时信息，这对于仪表板或网站来说很有用。

由于其简单性，这个类也完全在它的头文件中实现：

```cpp
#include <Poco/Net/HTTPRequestHandler.h>
 #include <Poco/Net/HTTPServerResponse.h>
 #include <Poco/Net/HTTPServerRequest.h>
 #include <Poco/URI.h>

 using namespace Poco;
 using namespace Poco::Net;

 #include "club.h"

 class StatusHandler: public HTTPRequestHandler { 
 public: 
          void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response)  {         
                Club::log(LOG_INFO, "StatusHandler: Request from " +                                                     request.clientAddress().toString());

                URI uri(request.getURI());
                vector<string> parts;
                uri.getPathSegments(parts);

                response.setContentType("application/json");
                response.setChunkedTransferEncoding(true); 

                if (parts.size() == 1) {
                      ostream& ostr = response.send();
                      ostr << "{ "clubstatus": " << !Club::clubOff << ",";
                      ostr << ""lock": " << Club::clubLocked << ",";
                      ostr << ""power": " << Club::powerOn << "";
                      ostr << "}";
                }
                else {
                      response.setStatus(HTTPResponse::HTTP_BAD_REQUEST);
                      ostream& ostr = response.send();
                      ostr << "{ "error": "Invalid request." }";
                }
          }
 };
```

我们在这里使用 `Club` 类的中央日志记录功能来记录传入请求的详细信息。在这里，我们只是记录客户端的 IP 地址，但可以使用 POCO `HTTPServerRequest` 类的 API 请求更详细的信息。

接下来，我们从请求中获取 URI，并将 URL 的路径部分拆分为一个向量实例。在设置响应对象的内容类型和传输编码设置后，我们检查我们确实得到了预期的 REST API 调用，此时我们组成 JSON 字符串，从 `Club` 类获取俱乐部房间状态信息，并返回这个信息。

在 JSON 对象中，我们包括有关俱乐部房间状态的一般信息，反转其布尔变量，以及锁的状态和电源状态，分别用 1 表示锁是关闭的或电源是开启的。

如果 URL 路径有更多段，则它是一个未识别的 API 调用，这将导致我们返回 HTTP 400（错误请求）错误。

# 数据处理器

数据处理器在请求处理器工厂未识别任何 REST API 调用时被调用。它尝试找到指定的文件，从磁盘读取它，并返回它，同时附带适当的 HTTP 标头。这个类也完全在它的头文件中实现：

```cpp
#include <Poco/Net/HTTPRequestHandler.h>
 #include <Poco/Net/HTTPServerResponse.h>
 #include <Poco/Net/HTTPServerRequest.h>
 #include <Poco/URI.h>
 #include <Poco/File.h>

 using namespace Poco::Net;
 using namespace Poco;

 class DataHandler: public HTTPRequestHandler { 
 public: 
    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) {
          Club::log(LOG_INFO, "DataHandler: Request from " + request.clientAddress().toString());

          // Get the path and check for any endpoints to filter on.
          URI uri(request.getURI());
          string path = uri.getPath();

          string fileroot = "htdocs";
          if (path.empty() || path == "/") { path = "/index.html"; }

          File file(fileroot + path);

          Club::log(LOG_INFO, "DataHandler: Request for " + file.path());
```

我们在这里假设要服务的任何文件都可以在这个服务运行的文件夹的子文件夹中找到。文件名（和路径）是从请求 URL 获取的。如果路径为空，我们分配一个默认的索引文件来提供服务：

```cpp
          if (!file.exists() || file.isDirectory()) {
                response.setStatus(HTTPResponse::HTTP_NOT_FOUND);
                ostream& ostr = response.send();
                ostr << "File Not Found.";
                return;
          }

          string::size_type idx = path.rfind('.');
          string ext = "";
          if (idx != std::string::npos) {
                ext = path.substr(idx + 1);
          }

          string mime = "text/plain";
          if (ext == "html") { mime = "text/html"; }
          if (ext == "css") { mime = "text/css"; }
          else if (ext == "js") { mime = "application/javascript"; }
          else if (ext == "zip") { mime = "application/zip"; }
          else if (ext == "json") { mime = "application/json"; }
          else if (ext == "png") { mime = "image/png"; }
          else if (ext == "jpeg" || ext == "jpg") { mime = "image/jpeg"; }
          else if (ext == "gif") { mime = "image/gif"; }
          else if (ext == "svg") { mime = "image/svg"; }
```

我们首先检查生成的文件路径是否有效，并且它是一个常规文件，而不是一个目录。如果这个检查失败，我们返回 HTTP 404 文件未找到错误。

通过这个检查后，我们尝试从文件路径中获取文件扩展名，以尝试确定文件的特定 MIME 类型。如果失败，我们使用默认的 MIME 类型用于纯文本：

```cpp
                try {
                      response.sendFile(file.path(), mime);
                }
                catch (FileNotFoundException &e) {
                      Club::log(LOG_ERROR, "DataHandler: File not found exception    triggered...");
                      cerr << e.displayText() << endl;

                      response.setStatus(HTTPResponse::HTTP_NOT_FOUND);
                      ostream& ostr = response.send();
                      ostr << "File Not Found.";
                      return;
                }
                catch (OpenFileException &e) {
                      Club::log(LOG_ERROR, "DataHandler: Open file exception triggered: " +    e.displayText());

                      response.setStatus(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR);
                      ostream& ostr = response.send();
                      ostr << "Internal Server Error. Couldn't open file.";
                      return;
                }
          }
 };
```

作为最后一步，我们使用响应对象的 `sendFile()` 方法将文件发送到客户端，以及我们之前确定的 MIME 类型。

我们还处理这个方法可能抛出的两个异常。第一个异常发生在由于某种原因找不到文件时。这导致我们返回另一个 HTTP 404 错误。

如果由于某种原因无法打开文件，我们返回 HTTP 500 内部服务器错误，并附带异常文本。

# 服务配置

对于 Raspberry Pi SBC 的 Raspbian Linux 发行版，系统服务通常使用 `systemd` 管理。这使用一个简单的配置文件，我们的俱乐部监控服务使用类似以下的内容：

```cpp
[Unit] 
Description=ClubStatus monitoring & control 

[Service] 
ExecStart=/home/user/clubstatus/clubstatus /home/user/clubstatus/config.ini 
User=user 
WorkingDirectory=/home/user/clubstatus 
Restart=always 
RestartSec=5 

[Install] 
WantedBy=multi-user.target 
```

此服务配置指定了服务的名称，服务从"`user`"用户账户的文件夹启动，服务的配置文件位于同一文件夹中。我们设置了服务的当前工作目录，并启用服务在五秒后自动重启，以防因任何原因失败。

最后，在系统启动到用户可以登录系统的程度后，服务将被启动。这样，我们就可以确保网络和其他功能已经启动。如果过早地启动系统服务，可能会因为尚未初始化而导致的功能缺失而失败。

接下来，这是 INI 文件配置文件：

```cpp
[MQTT]
 ; URL and port of the MQTT server.
 host = localhost
 port = 1883

 ; Authentication
 user = user
 pass = password

 ; The topic status on which changes will be published.
 clubStatusTopic = /my/topic

 [HTTP]
 port = 8080

 [Relay]
 ; Whether an i2c relay board is connected. 0 (false) or 1 (true).
 active = 0
 ; i2c address, in decimal or hexadecimal.
 address = 0x20
```

配置文件分为三个部分，MQTT、HTTP 和继电器，每个部分包含相关的变量。

对于 MQTT，我们提供了连接到 MQTT 代理的预期选项，包括基于密码的认证。我们还指定了将在此发布俱乐部状态更新的主题。

HTTP 部分仅包含我们将要监听的端口，默认情况下服务器监听所有接口。如果需要，可以在启动 HTTP 服务器之前将此属性设置为可配置的，从而使网络接口也成为可配置的。

最后，继电器部分允许我们打开或关闭继电器板功能，如果使用此功能，还可以配置 I2C 设备地址。

# 权限

由于 GPIO 和 I2C 都被视为常见的 Linux 设备，它们都有自己的权限集。假设某人希望避免以 root 用户运行服务，我们需要在`gpio`和`i2c`用户组中添加一个运行服务的账户：

```cpp
    sudo usermod -a -G gpio user
    sudo usermod -a -G i2c user
```

之后，我们需要重新启动系统（或注销并重新登录）以使更改生效。现在我们应该能够无任何问题地运行服务。

# 最终结果

在目标 SBC 上配置和安装了应用程序和`systemd`服务后，它将自动启动并配置自身。为了完成系统，你可以将它与合适的电源一起安装到机箱中，并将开关、网络电缆等信号线连接到机箱中。

该系统的这一实现已安装在德国卡尔斯鲁厄的 Entropia 黑客空间。此配置使用了一个真正的交通灯（合法获得）在俱乐部门外，带有 12 伏 LED 灯用于状态指示。SBC、继电器板、消抖板和电源（5V 和 12V MeanWell 工业电源）都集成在一个单独的激光切割木制机箱中：

![图片](img/fdcc5ed3-6f1c-4c43-a51e-d14a36225368.png)

然而，你可以自由地以任何你希望的方式集成组件。这里要考虑的主要是电子设备必须得到安全保护，避免受到损害和意外接触，因为继电器板可能会切换主电压，以及可能为电源提供的主电压线路。

# 示例 - 基本媒体播放器

另一个基于 SBC 的嵌入式系统的基本示例是媒体播放器。这可能涉及音频和音频-视频（AV）媒体格式。使用常规键盘和鼠标输入播放媒体时基于 SBC 的系统与嵌入式 SBC 媒体播放器之间的区别在于，在后者的情况下，系统只能用于该目的，软件和用户界面（物理和软件方面）都针对媒体播放器使用进行了优化。

为了达到这个目的，必须开发一个基于软件的前端界面，以及一个物理接口外围设备，通过它媒体播放器可以被控制。这可能只是一系列连接到 GPIO 引脚的开关，以及一个常规的 HDMI 显示器用于输出。或者，也可以使用触摸屏，尽管这需要更复杂的驱动程序设置。

由于我们的媒体播放器系统存储媒体文件是本地的，我们希望使用支持 SD 卡之外的外部存储的 SBC。一些 SBC 带有 SATA 连接，允许我们连接容量远超过 SD 卡的硬盘驱动器（HDD）。即使我们坚持使用与许多流行的 SBC 大小相似的紧凑型 2.5" HDD，我们也可以轻松且相对便宜地获得数 TB 的存储空间。

除了存储需求之外，我们还需要一个数字视频输出，我们希望使用 GPIO 或 USB 侧的用户界面按钮。

对于这个目的，一个非常适合的板是 LeMaker Banana Pro，它配备了 H3 ARM SoC、硬件 SATA、千兆以太网支持，以及带有 4k 视频解码支持的完整尺寸 HDMI 输出：

![图片](img/74a1aea9-04a3-4e25-9ac3-f4bc3020306d.png)

在了解了在 SBC 上安装 Armbian 或类似操作系统的基础知识之后，我们可以在系统上设置媒体播放器应用程序，使其与操作系统一起启动，并配置它加载播放列表并监听多个 GPIO 引脚上的事件。这些 GPIO 引脚将连接到多个控制开关，使我们能够浏览播放列表并开始、暂停和停止播放列表项目。

其他交互方法也是可能的，例如红外或基于无线电的遥控器，每种方法都有其自身的优缺点。

在接下来的章节中，我们将通过创建这个媒体播放器系统并将其转变为信息娱乐系统：

+   第六章，*基于 OS 的应用程序测试*

+   第八章，*示例 - 基于 Linux 的信息娱乐系统*

+   第十一章，*使用 Qt 开发嵌入式系统*

# 摘要

在本章中，我们探讨了基于操作系统的嵌入式系统，研究了我们可用的许多操作系统，特别是实时操作系统的显著差异。我们还看到了如何将 RTC 外围设备集成到基于 SBC 的 Linux 系统中，并探讨了基于用户空间和内核空间的驱动模块，以及它们的优缺点。

结合本章中的示例项目，读者现在应该对如何将一组需求转化为基于操作系统的嵌入式系统有了很好的了解。读者将知道如何添加外部外围设备并从操作系统中使用它们。

在下一章中，我们将探讨为资源受限的嵌入式系统进行开发，包括 8 位微控制器及其更大的同族成员。
