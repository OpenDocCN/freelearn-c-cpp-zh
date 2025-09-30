# 测试基于操作系统的应用程序

通常，嵌入式系统使用的是一个或多或少常规的**操作系统**（**OS**），这意味着在运行时环境和工具方面，与我们的桌面 OS 有很多相同之处，尤其是在针对嵌入式 Linux 时。然而，嵌入式硬件与我们的 PC 在性能和提供的访问方面存在的差异，使得考虑在哪里执行开发和测试的哪些部分，以及如何将其集成到我们的开发工作流程中变得至关重要。

在本章中，我们将涵盖以下主题：

+   开发跨平台代码

+   在 Linux 下调试和测试跨平台代码

+   有效使用交叉编译器

+   创建支持多个目标的构建系统

# 避免使用真实硬件

在基于操作系统的平台上进行开发（如嵌入式 Linux）的最大优势之一是它与常规桌面 Linux 安装非常相似。特别是当在 SoC 上运行基于 Debian 的 Linux 发行版（Armbian、Raspbian 等）时，我们实际上拥有相同的工具，包括整个包管理器、编译器集合和库，只需几个按键即可使用。

然而，这同时也是其最大的陷阱。

我们可以编写代码，将其复制到 SBC 上，在那里编译，运行测试，并在重复此过程之前对代码进行修改。或者，我们甚至可以直接在 SBC 上编写代码，实际上将其作为我们的唯一开发平台。

我们永远不应该这样做的主要原因如下：

+   现代 PC 要快得多。

+   在开发后期阶段之前，不应在真实硬件上进行测试。

+   自动集成测试变得更加困难。

在这里，第一个观点似乎相当明显。单个或双核 ARM SoC 需要一分钟才能编译的任务，在具有 3+ GHz 的相对现代多核多线程处理器和能够支持多核编译的工具链的情况下，将从编译开始到链接对象只需十秒或更少。

这意味着，我们不必等待半分钟或更长时间才能运行新的测试或开始调试会话，我们几乎可以立即这样做。

下面的两点是相关的。虽然在实际硬件上进行测试可能看起来有利，但它也伴随着自己的复杂性。一方面，这种硬件依赖于许多外部因素才能正常工作，包括电源供应、电源之间的任何布线、外围设备和信号接口。电磁干扰等问题也可能导致信号退化，以及由于电磁耦合而触发的中断。

在开发第三章第三章“为嵌入式 Linux 和类似系统开发”的俱乐部状态服务项目时，一个电磁耦合的例子变得明显。在这里，开关的一个信号线与 230V 交流电线并行。主电线上的电流变化在信号线上引起脉冲，导致错误的中断触发事件。

所有这些潜在的硬件相关问题表明，这样的测试远不如我们希望的那样确定。这种潜在的结果是项目开发时间比计划的长得多，由于冲突和非确定性的测试结果，调试变得复杂。

专注于在真实硬件上开发和为真实硬件开发的效果之一是，它使得自动化测试变得更加困难。原因是我们不能使用任何通用的构建集群，例如，像主流**持续****集成**（**CI**）服务中常见的基于 Linux 虚拟机（VM）的测试环境。

与此相反，我们可能需要以某种方式将类似 SBC（单板计算机）这样的组件集成到 CI（持续集成）系统中，使其能够交叉编译并将二进制文件复制到 SBC 上以运行测试，或者直接在 SBC 上编译，这又回到了第一个问题。

在接下来的几节中，我们将探讨一些方法，使基于嵌入式 Linux 的开发尽可能无痛苦，从交叉编译开始。

# 为 SBC 进行交叉编译

编译过程将源文件转换为中间格式，然后可以使用这种格式针对特定的 CPU 架构进行编译。对我们来说，这意味着我们不仅限于在 SBC 上编译该 SBC 上的应用程序，我们还可以在我们的开发 PC 上这样做。

要为像树莓派（基于 Broadcom Cortex-A 的 ARM SoC）这样的 SBC 进行交叉编译，我们需要安装`arm-linux-gnueabihf`工具链，它针对具有硬件浮点（硬件浮点）支持的 ARM 架构，输出兼容 Linux 的二进制文件。

在基于 Debian 的 Linux 系统上，我们可以使用以下命令安装整个工具链：

```cpp
sudo apt install build-essential
sudo apt install g++-arm-linux-gnueabihf
sudo apt install gdb-multiarch  
```

第一个命令安装了系统本地的基于 GCC 的工具链（如果尚未安装），以及任何相关的常用工具和实用程序，包括`make`、`libtool`、`flex`等。第二个命令安装实际的交叉编译器。最后，第三个包是支持多个架构的 GDB 调试器版本，我们稍后需要它来进行真实硬件上的远程调试，以及分析应用程序崩溃时产生的核心转储。

现在，我们可以使用命令行上的完整名称来使用目标 SBC 的 g++编译器：

```cpp
arm-linux-gnueabihf-g++  
```

要测试工具链是否正确安装，我们可以执行以下命令，它应该会告诉我们编译器的详细信息，包括版本：

```cpp
arm-linux-gnueabihf-g++ -v  
```

此外，我们可能还需要与目标系统上存在的某些共享库链接。为此，我们可以复制 `/lib` 和 `/usr` 文件夹的全部内容，并将它们作为编译器的系统根的一部分包含进来：

```cpp
mkdir ~/raspberry/sysroot
scp -r pi@Pi-system:/lib ~/raspberry/sysroot
scp -r pi@Pi-system:/usr ~/raspberry/sysroot  
```

这里，`Pi-system` 是 Raspberry Pi 或类似系统的 IP 地址或网络名称。在此之后，我们可以告诉 GCC 使用 `sysroot` 标志来使用这些文件夹而不是标准路径：

```cpp
--sysroot=dir  
```

这里 `dir` 是我们复制这些文件夹的文件夹，在这个例子中，那将是 `~/raspberry/sysroot`。

或者，我们只需复制所需的头文件和库文件，并将它们作为源树的一部分添加。哪种方法更容易，主要取决于相关项目的依赖项。

对于俱乐部状态服务项目，我们至少需要 WiringPi 的头文件和库，以及 POCO 项目及其依赖项的头文件和库。我们可以确定所需的依赖项，并复制之前安装的工具链中缺少的所需包含文件和库文件。除非有迫切需要这样做，否则直接复制 SBC 操作系统中的整个文件夹要容易得多。

作为使用 `sysroot` 方法的替代方案，我们也可以在链接代码时显式定义我们希望使用的共享库的路径。这当然有其自身的优缺点。

# 俱乐部状态服务集成测试

在我们开始交叉编译和在实际硬件上进行测试之前，为了在常规桌面 Linux（或 macOS 或 Windows）系统上测试俱乐部状态服务，我们编写了一个简单的集成测试，该测试使用 GPIO 和 I2C 外围设备的模拟。

在 第三章 的项目源代码中，*为嵌入式 Linux 和类似系统开发*，这些外围设备的文件位于该项目的 `wiring` 文件夹中。

我们从 `wiringPi.h` 头文件开始：

```cpp
#include <Poco/Timer.h>

#define  INPUT              0
#define  OUTPUT                   1
#define  PWM_OUTPUT         2
#define  GPIO_CLOCK         3
#define  SOFT_PWM_OUTPUT          4
#define  SOFT_TONE_OUTPUT   5
#define  PWM_TONE_OUTPUT          6
```

我们包含来自 POCO 框架的头文件，以便我们稍后可以轻松创建定时器实例。然后，我们定义所有可能的引脚模式，就像实际的 WiringPi 头文件定义的那样：

```cpp
#define  LOW                0
#define  HIGH               1

#define  PUD_OFF                  0
#define  PUD_DOWN           1
#define  PUD_UP                   2

#define  INT_EDGE_SETUP          0
#define  INT_EDGE_FALLING  1
#define  INT_EDGE_RISING         2
#define  INT_EDGE_BOTH           3
```

这些定义进一步定义了引脚模式，包括数字输入电平、引脚上上拉和下拉的可能状态，以及最后可能的中断类型，定义了中断的触发器或触发器：

```cpp
typedef void (*ISRCB)(void); 
```

这个 `typedef` 定义了中断回调函数指针的格式。

现在让我们看看 `WiringTimer` 类：

```cpp
class WiringTimer {
    Poco::Timer* wiringTimer;
    Poco::TimerCallback<WiringTimer>* cb;
    uint8_t triggerCnt;

 public:
    ISRCB isrcb_0;
    ISRCB isrcb_7;
    bool isr_0_set;
    bool isr_7_set;

    WiringTimer();
    ~WiringTimer();
    void start();
    void trigger(Poco::Timer &t);
 };
```

这个类是我们模拟实现 GPIO 侧的组成部分。其主要目的是跟踪我们感兴趣的哪一个中断已被注册，并使用定时器定期触发它们，正如我们稍后将要看到的：

```cpp
int wiringPiSetup(); 
void pinMode(int pin, int mode); 
void pullUpDnControl(int pin, int pud); 
int digitalRead(int pin);
int wiringPiISR(int pin, int mode, void (*function)(void));
```

最后，我们在继续实现之前定义了标准的 WiringPi 函数：

```cpp
#include "wiringPi.h"

#include <fstream>
#include <memory>

WiringTimer::WiringTimer() {
   triggerCnt = 0;
   isrcb_0 = 0;
   isrcb_7 = 0;
   isr_0_set = false;
   isr_7_set = false;

   wiringTimer = new Poco::Timer(10 * 1000, 10 * 1000);
   cb = new Poco::TimerCallback<WiringTimer>(*this, 
   &WiringTimer::trigger);
}
```

在类构造函数中，我们在创建计时器实例之前设置默认值，将其配置为每 10 秒调用我们的回调函数，初始延迟为 10 秒：

```cpp
WiringTimer::~WiringTimer() {
   delete wiringTimer;
   delete cb;
}
```

在析构函数中，我们删除计时器回调实例：

```cpp
void WiringTimer::start() {
   wiringTimer->start(*cb);
}
```

在这个函数中，我们实际上启动了计时器：

```cpp
void WiringTimer::trigger(Poco::Timer &t) {
    if (triggerCnt == 0) {
          char val = 0x00;
          std::ofstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
          PIN0VAL.put(val);
          PIN0VAL.close();

          isrcb_0();

          ++triggerCnt;
    }
    else if (triggerCnt == 1) {
          char val = 0x01;
          std::ofstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
          PIN7VAL.put(val);
          PIN7VAL.close();

          isrcb_7();

          ++triggerCnt;
    }
    else if (triggerCnt == 2) {
          char val = 0x00;
          std::ofstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
          PIN7VAL.put(val);
          PIN7VAL.close();

          isrcb_7();

          ++triggerCnt;
    }
    else if (triggerCnt == 3) {
          char val = 0x01;
          std::ofstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
          PIN0VAL.put(val);
          PIN0VAL.close();

          isrcb_0();

          triggerCnt = 0;
    }
 }

```

类中的最后一个函数是定时器的回调函数。其工作方式是跟踪它被触发的次数，并以文件中的值形式设置适当的引脚电平，这些文件是我们写入磁盘的。

在初始延迟之后，第一次触发将锁定开关设置为`false`，第二次将状态开关设置为`true`，第三次将状态开关再次设置为`false`，最后第四次触发将锁定开关再次设置为`true`，然后重置计数器并重新开始：

```cpp
namespace Wiring {
   std::unique_ptr<WiringTimer> wt;
   bool initialized = false;
}
```

我们在全局命名空间中添加了一个`unique_ptr`实例，用于`WiringTimer`类实例，以及一个初始化状态指示器。

```cpp
int wiringPiSetup() {
    char val = 0x01;
    std::ofstream PIN0VAL;
    std::ofstream PIN7VAL;
    PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
    PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
    PIN0VAL.put(val);
    val = 0x00;
    PIN7VAL.put(val);
    PIN0VAL.close();
    PIN7VAL.close();

    Wiring::wt = std::make_unique<WiringTimer>();
    Wiring::initialized = true;

    return 0;
 }
```

设置函数用于将模拟 GPIO 引脚输入值的默认值写入磁盘。我们在这里也创建了`WiringTimer`实例的指针：

```cpp
 void pinMode(int pin, int mode) {
    // 

    return;
 }

 void pullUpDnControl(int pin, int pud) {
    // 

    return;
 }
```

由于我们的模拟实现决定了引脚的行为，我们可以忽略这些函数上的任何输入。为了测试目的，我们可以在这些函数被正确时间以适当设置调用时添加一个断言来验证：

```cpp
 int digitalRead(int pin) {
    if (pin == 0) {
          std::ifstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary);
          int val = PIN0VAL.get();
          PIN0VAL.close();

          return val;
    }
    else if (pin == 7) {
          std::ifstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary);
          int val = PIN7VAL.get();
          PIN7VAL.close();

          return val;
    }

    return 0;
 }
```

当读取两个模拟引脚之一的值时，我们打开其相应的文件并读取其内容，该内容是设置函数或回调设置的 1 或 0：

```cpp
//This value is then returned to the calling function.

 int wiringPiISR(int pin, int mode, void (*function)(void)) {
    if (!Wiring::initialized) { 
          return 1;
    }

    if (pin == 0) { 
          Wiring::wt->isrcb_0 = function;
          Wiring::wt->isr_0_set = true;
    }
    else if (pin == 7) {
          Wiring::wt->isrcb_7 = function;
          Wiring::wt->isr_7_set = true;
    }

    if (Wiring::wt->isr_0_set && Wiring::wt->isr_7_set) {
          Wiring::wt->start();
    }

    return 0;
 }
```

这个函数用于注册中断及其关联的回调函数。在初始检查模拟已被设置函数初始化之后，我们继续为两个指定引脚之一注册中断。

一旦为两个引脚都设置了中断，我们就开始计时器，这将反过来开始为中断回调生成事件。

接下来是 I2C 总线模拟：

```cpp
int wiringPiI2CSetup(const int devId);
int wiringPiI2CWriteReg8(int fd, int reg, int data);
```

我们只需要在这里添加两个函数：设置函数和简单的单字节寄存器写入函数。

实现如下：

```cpp
#include "wiringPiI2C.h"

#include "../club.h"

#include <Poco/NumberFormatter.h>

using namespace Poco;

int wiringPiI2CSetup(const int devId) {
   Club::log(LOG_INFO, "wiringPiI2CSetup: setting up device ID: 0x" 
                                        + NumberFormatter::formatHex(devId));

   return 0;
}
```

在设置函数中，我们记录请求的设备 ID（I2C 总线地址）并返回一个标准设备句柄。在这里，我们使用`Club`类的`log()`函数使模拟集成到代码的其余部分：

```cpp
int wiringPiI2CWriteReg8(int fd, int reg, int data) {
    Club::log(LOG_INFO, "wiringPiI2CWriteReg8: Device handle 0x" + NumberFormatter::formatHex(fd) 
                                        + ", Register 0x" + NumberFormatter::formatHex(reg)
                                        + " set to: 0x" + NumberFormatter::formatHex(data));

    return 0;
}
```

由于调用此函数的代码不会期望任何响应，除了简单确认数据已被接收之外，我们只需在这里记录接收到的数据和更多详细信息。这里也使用了 POCO 的`NumberFormatter`类，用于将整数数据格式化为十六进制值，以保持与应用程序的一致性。

现在我们编译项目并使用以下命令行命令：

```cpp
make TEST=1  
```

在 GDB 下运行应用程序（以查看何时创建/销毁新线程）现在得到以下输出：

```cpp
 Starting ClubStatus server...
 Initialised C++ Mosquitto library.
 Created listener, entering loop...
 [New Thread 0x7ffff49c9700 (LWP 35462)]
 [New Thread 0x7ffff41c8700 (LWP 35463)]
 [New Thread 0x7ffff39c7700 (LWP 35464)]
 Initialised the HTTP server.
 INFO:       Club: starting up...
 INFO:       Club: Finished wiringPi setup.
 INFO:       Club: Finished configuring pins.
 INFO:       Club: Configured interrupts.
 [New Thread 0x7ffff31c6700 (LWP 35465)]
 INFO:       Club: Started update thread.
 Connected. Subscribing to topics...
 INFO:       ClubUpdater: Starting i2c relay device.
 INFO:       wiringPiI2CSetup: setting up device ID: 0x20
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x6 set to: 0x0
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x0
 INFO:       ClubUpdater: Finished configuring the i2c relay device's registers.  
```

在这一点上，系统已经配置完毕，所有中断都已设置，应用程序已配置 I2C 设备。定时器已经开始初始倒计时：

```cpp
 INFO:       ClubUpdater: starting initial update run.
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Red on.
 DEBUG:      ClubUpdater: Changing output register to: 0x8
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x8
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x8
 INFO:       ClubUpdater: Initial status update complete.  
```

已读取 GPIO 引脚的初始状态，并且两个开关都发现处于`关闭`位置，因此我们通过在寄存器中写入其位置来激活交通灯指示器的红灯：

```cpp
 INFO:       ClubUpdater: Entering waiting condition. INFO:       ClubUpdater: lock status changed to unlocked
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Yellow on.
 DEBUG:      ClubUpdater: Changing output register to: 0x4
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x4
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x4
 INFO:       ClubUpdater: status switch status changed to on
 INFO:       ClubUpdater: Opening club.
 INFO:       ClubUpdater: Started power timer...
 DEBUG:      ClubUpdater: Sent MQTT message.
 INFO:       ClubUpdater: New lights, clubstatus on.
 DEBUG:      ClubUpdater: Power timer active, inverting power state from: on
 INFO:       ClubUpdater: Green on.
 DEBUG:      ClubUpdater: Changing output register to: 0x2
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x2
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x2
 INFO:       ClubUpdater: status switch status changed to off
 INFO:       ClubUpdater: Closing club.
 INFO:       ClubUpdater: Started timer.
 INFO:       ClubUpdater: Started power timer...
 DEBUG:      ClubUpdater: Sent MQTT message.
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer active, inverting power state from: off
 INFO:       ClubUpdater: Yellow on.
 DEBUG:      ClubUpdater: Changing output register to: 0x5
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x5
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x5
 INFO:       ClubUpdater: setPowerState called.
 DEBUG:      ClubUpdater: Writing relay with: 0x4
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x4
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x4
 DEBUG:      ClubUpdater: Written relay outputs.
 DEBUG:      ClubUpdater: Finished setPowerState.
 INFO:       ClubUpdater: lock status changed to locked
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Red on.
 DEBUG:      ClubUpdater: Changing output register to: 0x8
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x8
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x8  
```

接下来，定时器开始重复触发回调函数，使其通过不同的阶段。这使我们能够确定代码的基本功能是正确的。

在这一点上，我们可以开始实现更复杂的测试用例，甚至可能使用嵌入式 Lua、Python 运行时或类似的脚本化测试用例。

# 模拟与硬件

当模拟大量代码和硬件外设时，一个明显的问题是要问模拟结果的真实性如何。显然，在我们将测试转移到目标系统之前，我们希望我们的集成测试能够尽可能覆盖更多的真实场景。

如果我们想知道在模拟中要覆盖哪些测试用例，我们必须查看我们的项目需求（它应该能够处理什么），以及在实际场景中可能发生的情况和输入。

对于这一点，我们会分析底层代码以查看可能发生的条件，并决定哪些对我们来说是相关的。

在我们之前查看的 WiringPi 模拟中，快速查看库实现的源代码清楚地表明，与我们在目标系统上使用的版本相比，我们简化了代码。

查看基本的 WiringPi 设置函数，我们看到它执行以下操作：

+   确定确切的板型和 SoC 以获取 GPIO 布局

+   打开 Linux 设备以访问内存映射的 GPIO 引脚

+   设置 GPIO 设备的内存偏移量，并使用`mmap()`将 PWM、定时器和 GPIO 等特定外设映射到内存中

与忽略`pinMode()`调用不同，实现执行以下操作：

+   在 SoC 中适当地设置硬件 GPIO 方向寄存器（用于输入/输出模式）

+   在引脚上启动 PWM、软 PWM 或 Tone 模式（如请求所示）；子功能设置适当的寄存器

这继续到 I2C 方面，其中设置函数的实现如下：

```cpp
int wiringPiI2CSetup (const int devId) { 
   int rev; 
   const char *device; 

   rev = piGpioLayout(); 

   if (rev == 1) { 
         device = "/dev/i2c-0"; 
   } 
   else { 
         device = "/dev/i2c-1"; 
   } 

   return wiringPiI2CSetupInterface (device, devId); 
} 
```

与我们的模拟实现相比，主要区别在于预期的 I2C 外设在操作系统的内存文件系统中，而板型版本决定了我们选择哪一个。

最后被调用的函数尝试打开设备，就像在 Linux 和类似的操作系统中，每个设备都是一个我们可以打开并获取文件句柄的文件，如果成功的话。这个文件句柄是函数返回时返回的 ID：

```cpp
int wiringPiI2CSetupInterface (const char *device, int devId) { 
   int fd; 
   if ((fd = open (device, O_RDWR)) < 0) { 
         return wiringPiFailure (WPI_ALMOST, "Unable to open I2C device: %s\n", 
                                                                                                strerror (errno)); 
   } 

   if (ioctl (fd, I2C_SLAVE, devId) < 0) { 
         return wiringPiFailure (WPI_ALMOST, "Unable to select I2C device: %s\n",                                                                                                strerror (errno)); 
   } 

   return fd; 
} 
```

在打开 I2C 设备后，Linux 系统函数 `ioctl()` 用于向 I2C 外设发送数据，在这种情况下，是我们希望使用的 I2C 从设备地址。如果成功，我们将获得一个非负响应，并返回我们的文件句柄整数。

使用 `ioctl()` 函数来读写 I2C 总线也是通过该函数实现的，正如我们在同一源文件中看到的：

```cpp
static inline int i2c_smbus_access (int fd, char rw, uint8_t command, int size, union i2c_smbus_data *data) { 
   struct i2c_smbus_ioctl_data args; 

   args.read_write = rw; 
   args.command    = command; 
   args.size       = size; 
   args.data       = data; 
   return ioctl(fd, I2C_SMBUS, &args); 
} 
```

对于每次 I2C 总线访问，都会调用相同的内联函数。在我们希望使用的 I2C 设备已经选定的情况下，我们可以简单地针对 I2C 外设，并让它将有效载荷传输到设备。

在这里，`i2c_smbus_data` 类型是一个简单的联合体，用于支持返回值的多种大小（在执行读取操作时）：

```cpp
union i2c_smbus_data { 
   uint8_t byte; 
   uint16_t word; 
   uint8_t block[I2C_SMBUS_BLOCK_MAX + 2]; 
}; 
```

在这里，我们主要看到了使用抽象 API 的好处。没有它，我们的代码中会充斥着低级调用，这将使模拟变得非常困难。我们还看到，有一些条件我们可能也需要测试，例如缺失的 I2C 从设备、I2C 总线上的读写错误，这些错误可能导致意外的行为，以及 GPIO 引脚上的意外输入，包括中断引脚，正如在本章开头已经提到的。

虽然显然不可能为所有场景都做出计划，但应该努力记录所有现实场景并将它们纳入模拟实现中，以便在集成和回归测试以及调试期间可以随意启用。

# 使用 Valgrind 进行测试

Valgrind 是最常用的开源工具集合，用于分析和应用性能，从应用程序的缓存和堆行为到内存泄漏和潜在的线程问题。它与底层操作系统协同工作，因为根据使用的工具，它必须拦截从内存分配到与多线程相关的指令等所有内容。这也是为什么它仅在 64 位 x86_64 架构的 Linux 上完全支持的原因。

在其他支持的平台上使用 Valgrind（例如 x86、PowerPC、ARM、S390、MIPS 和 ARM，以及 Solaris 和 macOS）当然也是一个选项，但 Valgrind 项目的首要开发目标是 x86_64/Linux，这使得它成为进行性能分析和调试的最佳平台，即使以后会针对其他平台。

在 Valgrind 网站上 [`valgrind.org/info/platforms.html`](http://valgrind.org/info/platforms.html)，我们可以看到目前支持的所有平台的全面概述。

Valgrind 非常吸引人的一个特性是，它的所有工具都不需要我们以任何方式修改源代码或生成的二进制文件。这使得它很容易集成到现有的工作流程中，包括自动化测试和集成系统。

在基于 Windows 的系统上，也提供了诸如 Dr. Memory ([`drmemory.org/`](http://drmemory.org/))之类的工具，这些工具可以处理至少与内存相关行为的分析。这个特定的工具还附带了一个名为 Dr. Fuzz 的工具，该工具可以重复调用具有不同输入的函数，这对于集成测试可能很有用。

通过使用类似于我们在上一节中查看的集成测试，我们可以从我们 PC 的舒适环境中完全分析我们代码的行为。由于 Valgrind 的所有工具都会显著减慢我们代码的执行速度（10-100 倍），能够在快速系统上进行大部分调试和性能分析意味着我们可以在开始测试目标硬件之前节省大量时间。

在我们可能会最频繁使用的工具中，**Memcheck**、**Helgrind**和**DRD**对于检测内存分配和多线程问题很有用。一旦我们的代码通过了这三个工具，同时使用一个提供广泛代码覆盖的广泛集成测试，我们就可以继续进行性能分析和优化。

为了分析我们的代码，我们随后使用**Callgrind**来查看我们的代码在执行时花费了最多的时间在哪里，然后使用**Massif**来进行堆分配的分析。从这些数据中我们可以获取的信息，我们可以对代码进行修改，以简化常见的分配和释放情况。它也可能显示在哪里使用缓存来重用资源而不是从内存中丢弃它们是有意义的。

最后，我们还会运行一轮 MemCheck、Helgrind 和 DRD，以确保我们的更改没有引起任何回归。一旦我们满意，我们就可以继续在目标系统上部署代码，并查看其在那里的表现。

如果目标系统也运行 Linux 或其他支持的操作系统，我们也可以在那里使用 Valgrind，以检查我们是否遗漏了任何内容。根据确切的平台（操作系统和 CPU 架构），我们可能会遇到该平台 Valgrind 端口的限制。这可以包括诸如*未处理的指令*之类的错误，其中工具尚未实现 CPU 指令，因此 Valgrind 无法继续。

通过将集成测试扩展到使用 SBC 而不是本地进程，我们可以设置一个持续集成系统，在该系统中，除了对本地进程的测试之外，我们还会在真实硬件上运行它们，考虑到与用于大多数测试的基于 x86_64 的 Linux 系统相比，真实硬件平台存在的限制。

# 多目标构建系统

交叉编译和多目标构建系统是那些往往会让很多人感到害怕的词汇之一，主要是因为它们会唤起大量复杂的构建脚本的图像，这些脚本需要使用晦涩的咒语才能执行所需的操作。在本章中，我们将探讨一个基于简单 Makefile 的构建系统，该系统已在各种硬件目标上的商业项目中得到应用。

使构建系统易于使用的一点是能够以最小的麻烦设置好编译环境，并有一个中心位置，我们可以从中控制构建项目或其部分的所有相关方面，包括构建和运行测试。

因此，我们在项目的顶部有一个单独的 Makefile，它处理所有基本任务，包括确定我们运行的平台。我们在这里所做的唯一简化是假设一个类 Unix 环境，在 Windows 上使用 MSYS2 或 Cygwin，Linux、BSD 和 OS X/macOS 以及其他使用它们的本地 shell 环境。然而，我们也可以将其修改为允许使用 Microsoft Visual Studio、**Intel 编译器集合**（**ICC**）和其他编译器，只要它们提供基本工具。

构建系统的关键在于简单的 Makefiles，在其中我们定义了目标平台的具体细节，例如，对于在 x86_x64 硬件上运行的标准 Linux 系统：

```cpp
 TARGET_OS = linux
 TARGET_ARCH = x86_64

 export CC = gcc
 export CXX = g++
 export CPP = cpp
 export AR = ar
 export LD = g++
 export STRIP = strip
 export OBJCOPY = objcopy

 PLATFORM_FLAGS = -D__PLATFORM_LINUX__ -D_LARGEFILE64_SOURCE -D __LINUX__
 STD_FLAGS = $(PLATFORM_FLAGS) -Og -g3 -Wall -c -fmessage-length=0 -ffunction-sections -fdata-sections -DPOCO_HAVE_GCC_ATOMICS -DPOCO_UTIL_NO_XMLCONFIGURATION -DPOCO_HAVE_FD_EPOLL
 STD_CFLAGS = $(STD_FLAGS)
 STD_CXXFLAGS = -std=c++11 $(STD_FLAGS)
 STD_LDFLAGS = -L $(TOP)/build/$(TARGET)/libboost/lib \
                         -L $(TOP)/build/$(TARGET)/poco/lib \
                         -Wl,--gc-sections
 STD_INCLUDE = -I. -I $(TOP)/build/$(TARGET)/libboost/include \
                         -I $(TOP)/build/$(TARGET)/poco/include \
                         -I $(TOP)/extern/boost-1.58.0
 STD_LIBDIRS = $(STD_LDFLAGS)
 STD_LIBS = -ldl -lrt -lboost_system -lssl -lcrypto -lpthread

```

在这里，我们可以设置我们将用于编译、创建存档、从二进制文件中剥离调试符号等命令行工具的名称。构建系统将使用目标操作系统和架构来保持创建的二进制文件分开，这样我们就可以使用相同的源树在一次运行中为所有目标平台创建二进制文件。

我们可以看到如何将传递给编译器和链接器的标志分为不同的类别：平台特定的标志、通用（标准）标志，以及针对 C 和 C++ 编译器的特定标志。当集成已集成到源树中的外部依赖项，而这些依赖项是用 C 编写的时，这种做法很有用。这些依赖项我们将在稍后更详细地看到，它们位于 `extern` 文件夹中。

这种类型的文件将被高度定制以适应特定的项目，添加所需的包含文件、库和编译标志。对于这个示例文件，我们可以看到一个使用 POCO 和 Boost 库，以及 OpenSSL 的项目，并对 POCO 库针对目标平台进行了调整。

首先，让我们看看 macOS 配置文件的开头：

```cpp
TARGET_OS = osx
 TARGET_ARCH = x86_64

 export CC = clang
 export CXX = clang++
 export CPP = cpp
 export AR = ar
 export LD = clang++
 export STRIP = strip
 export OBJCOPY = objcopy
```

尽管文件的其他部分几乎相同，但在这里我们可以看到一个很好的例子，说明如何泛化工具的名称。尽管 Clang 支持与 GCC 相同的标志，但其工具的名称不同。采用这种方法，我们只需在这个文件中写一次不同的名称，一切就会正常工作。

这继续到 Linux on ARM 目标，它被设置为交叉编译目标：

```cpp
TARGET_OS = linux
 TARGET_ARCH = armv7
 TOOLCHAIN_NAME = arm-linux-gnueabihf

 export CC = $(TOOLCHAIN_NAME)-gcc
 export CXX = $(TOOLCHAIN_NAME)-g++
 export AR = $(TOOLCHAIN_NAME)-ar
 export LD = $(TOOLCHAIN_NAME)-g++
 export STRIP = $(TOOLCHAIN_NAME)-strip
 export OBJCOPY = $(TOOLCHAIN_NAME)-objcopy
```

在这里，我们看到了之前在本章中提到的 ARM Linux 平台的交叉编译工具链的再次出现。为了节省我们输入，我们定义了一个基本名称，这样就可以轻松地重新定义。这也显示了 Makefile 的灵活性。通过一些更多的创意，我们可以创建一组模板，将整个工具链概括成一个简单的 Makefile，根据平台 Makefile（或其他配置文件）中的提示将其包含在主 Makefile 中，从而使其非常灵活。

接下来，我们将查看项目根目录下的主要 Makefile：

```cpp
ifndef TARGET
 $(error TARGET parameter not provided.)
 endif
```

由于我们无法猜测用户希望我们针对哪个平台，因此我们需要指定目标，平台名称作为值，例如，`linux-x86_x64`：

```cpp
export TOP := $(CURDIR)
 export TARGET
```

在系统的后续部分，我们需要知道我们在本地文件系统中的哪个文件夹，以便我们可以指定绝对路径。我们使用标准的 Make 变量来完成此操作，并将其作为我们自己的环境变量导出，同时导出构建目标名称：

```cpp
UNAME := $(shell uname)
 ifeq ($(UNAME), Linux)
 export HOST = linux
 else
 export HOST = win32
 export FILE_EXT = .exe
 endif
```

使用（命令行）`uname`命令，我们可以检查我们正在运行的操作系统，每个在 shell 中支持该命令的操作系统都会返回其名称，例如，Linux 的`Linux`和 macOS 的`Darwin`。在纯 Windows（没有 MSYS2 或 Cygwin）上，该命令不存在，这将得到这个`if/else`语句的第二部分。

这条语句可以根据构建系统的需求扩展以支持更多的操作系统。在这种情况下，它仅用于确定我们创建的可执行文件是否应该有文件扩展名：

```cpp
ifeq ($(HOST), linux)
 export MKDIR   = mkdir -p
 export RM            = rm -rf
 export CP            = cp -RL
 else
 export MKDIR   = mkdir -p
 export RM            = rm -rf
 export CP            = cp -RL
 endif
```

在这个`if/else`语句中，我们可以设置常见文件操作的正确命令行命令。由于我们选择了简单的方法，我们假设在 Windows 上使用 MSYS2 或类似的 Bash shell。

在这一点上，我们也可以进一步推广概念，将操作系统文件 CLI 工具作为一组独立的 Makefile 分离出来，然后我们可以将其作为操作系统特定设置的组成部分包含进来：

```cpp
include Makefile.$(TARGET)

 export TARGET_OS
 export TARGET_ARCH
 export TOOLCHAIN_NAME
```

在这一点上，我们使用 Makefile 提供的目标参数来包含适当的配置文件。在从中导出一些详细信息后，我们现在有一个配置好的构建系统：

```cpp
all: extern-$(TARGET) core

 extern:
    $(MAKE) -C ./extern $(LIBRARY)

 extern-$(TARGET):
    $(MAKE) -C ./extern all-$(TARGET)

 core:
    $(MAKE) -C ./Core

 clean: clean-core clean-extern

 clean-extern:
    $(MAKE) -C ./extern clean-$(TARGET)

 clean-core:
    $(MAKE) -C ./Core clean

 .PHONY: all clean core extern clean-extern clean-core extern-$(TARGET)
```

从这个单一的 Makefile 中，我们可以选择编译整个项目或仅编译依赖项或核心项目。我们还可以编译特定的外部依赖项，而不编译其他任何内容。

最后，我们可以清理核心项目、依赖项或两者。

这个顶级 Makefile 主要用于控制底层的 Makefile。接下来的两个 Makefile 位于`Core`和`extern`文件夹中。在这些文件夹中，`Core` Makefile 直接编译项目的核心：

```cpp
include ../Makefile.$(TARGET) 

OUTPUT := CoreProject 

INCLUDE = $(STD_INCLUDE) 
LIBDIRS = $(STD_LIBDIRS) 

include ../version 
VERSIONINFO = -D__VERSION="\"$(VERSION)\"" 
```

作为第一步，我们包含目标平台的 Makefile 配置，以便我们可以访问其所有定义。这些也可以在主 Makefile 中导出，但这样我们可以更自由地自定义构建系统。

在执行一些小任务之前，我们指定要构建的输出二进制文件名，包括在项目根目录下打开`version`文件（使用 Makefile 语法），其中包含我们从源代码构建的版本号。这是准备传递给编译器的预处理器定义：

```cpp
ifdef RELEASE 
TIMESTAMP = $(shell date --date=@`git show -s --format=%ct $(RELEASE)^{commit}` -u +%Y-%m-%dT%H:%M:%SZ) 
else ifdef GITTIME 
TIMESTAMP = $(shell date --date=@`git show -s --format=%ct` -u +%Y-%m-%dT%H:%M:%SZ) 
TS_SAFE = _$(shell date --date=@`git show -s --format=%ct` -u +%Y-%m-%dT%H%M%SZ) 
else 
TIMESTAMP = $(shell date -u +%Y-%m-%dT%H:%M:%SZ) 
TS_SAFE = _$(shell date -u +%Y-%m-%dT%H%M%SZ) 
endif 
```

这是另一个我们需要有一个 Bash shell 或兼容工具的章节，因为我们使用日期命令来为构建创建时间戳。格式取决于传递给主 Makefile 的参数。如果我们正在构建一个发布版本，我们从 Git 仓库中获取时间戳，使用 Git 提交标签名称来检索该标签的提交时间戳，然后再进行格式化。

如果传递了`GITTIME`参数，则使用最近的 Git 提交的时间戳。否则，使用当前时间和日期（UTC）。

这段代码旨在解决与大量测试和集成构建相关的问题之一：跟踪哪些构建了何时以及使用源代码的哪个版本。它可以适应其他文件版本控制系统，只要它支持类似的功能，包括检索特定的时间戳。

值得注意的是，我们创建的第二个时间戳。这是一个与附加到生成的二进制文件的时间戳略有不同格式的版本，除非我们在发布模式下构建：

```cpp
CFLAGS = $(STD_CFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 
CXXFLAGS = $(STD_CXXFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 

OBJROOT := $(TOP)/build/$(TARGET)/obj 
CPP_SOURCES := $(wildcard *.cpp) 
CPP_OBJECTS := $(addprefix $(OBJROOT)/,$(CPP_SOURCES:.cpp=.o)) 
OBJECTS := $(CPP_OBJECTS) 
```

在这里，我们设置希望传递给编译器的标志，包括版本和时间戳，这两者都作为预处理器定义传递。

最后，当前项目文件夹中的源代码被收集，对象文件的输出文件夹被设置。正如我们所看到的，我们将对象文件写入项目根目录下的一个文件夹中，并通过编译目标进行进一步分离：

```cpp
.PHONY: all clean 

all: makedirs $(CPP_OBJECTS) $(C_OBJECTS) $(TOP)/build/bin/$(TARGET)/$(OUTPUT)_$(VERSION)_$(TARGET)$(TS_SAFE) 

makedirs: 
   $(MKDIR) $(TOP)/build/bin/$(TARGET) 
   $(MKDIR) $(OBJROOT) 

$(OBJROOT)/%.o: %.cpp 
   $(CXX) -o $@ $< $(CXXFLAGS) 
```

这部分对于 Makefile 来说相当通用。我们有`all`目标，以及一个用于在文件系统中创建文件夹（如果尚不存在）的目标。最后，我们在下一个目标中接收源文件数组，按照配置编译它们，并将对象文件输出到适当的文件夹：

```cpp
$(TOP)/build/bin/$(TARGET)/$(OUTPUT)_$(VERSION)_$(TARGET)$(TS_SAFE): $(OBJECTS) 
   $(LD) -o $@ $(OBJECTS) $(LIBDIRS) $(LIBS) 
   $(CP) $@ $@.debug 
ifeq ($(TARGET_OS), osx) 
   $(STRIP) -S $@ 
else 
   $(STRIP) -S --strip-unneeded $@      
endif 
```

在我们从源文件创建所有对象文件之后，我们希望将它们链接在一起，这一步骤就是为此而设。我们还可以看到二进制文件最终将放在哪里：在项目构建文件夹的`bin`子文件夹中。

调用链接器，并创建结果的二进制文件的副本，我们将其后缀为`.debug`以指示它是包含所有调试信息的版本。然后，原始二进制文件被剥离其调试符号和其他不需要的信息，留下一个较小的二进制文件，可以复制到远程测试系统，以及一个包含所有调试信息的大版本，以便我们在需要分析核心转储或进行远程调试时使用。

在这里，我们还看到了由于 Clang 链接器不支持命令行标志而添加的一个小技巧，需要实现一个特殊案例。在处理跨平台编译和类似任务时，很可能会遇到许多这样的小细节，所有这些都使得编写一个简单有效的通用构建系统变得复杂：

```cpp
clean: 
   $(RM) $(CPP_OBJECTS) 
   $(RM) $(C_OBJECTS) 
```

作为最后一步，我们允许删除生成的目标文件。

`extern` 中的第二个子 Makefile 也值得关注，因为它控制了所有底层依赖：

```cpp
ifndef TARGET 
$(error TARGET parameter not provided.) 
endif 

all: libboost poco 

all-linux-%: 
   $(MAKE) libboost poco 

all-qnx-%: 
   $(MAKE) libboost poco 

all-osx-%: 
   $(MAKE) libboost poco 

all-windows: 
   $(MAKE) libboost poco 
```

这里有一个有趣的功能，即基于目标平台的依赖选择器。如果我们有不应为特定平台编译的依赖项，我们可以在这里跳过它们。此功能还允许我们直接指示此 Makefile 编译特定平台的全部依赖项。在这里，我们允许针对 QNX、Linux、OS X/macOS 和 Windows 进行目标定位，同时忽略架构：

```cpp
libboost: 
   cd boost-1.58.0 && $(MAKE) 

poco: 
   cd poco-1.7.4 && $(MAKE) 
```

实际的目标仅仅是调用依赖项目顶部的另一个 Makefile，该 Makefile 依次编译该依赖并将其添加到构建文件夹中，以便 `Core` 的 Makefile 可以使用。

当然，我们也可以直接使用现有的构建系统，如这里用于 OpenSSL 的构建系统：

```cpp
openssl: 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl/include 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl/lib 
   cd openssl-1.0.2 && ./Configure --openssldir="$(TOP)/build/$(TARGET)/openssl" shared os/compiler:$(TOOLCHAIN_NAME):$(OPENSSL_PARAMS) && \ 
     $(MAKE) build_libs 
   $(CP) openssl-1.0.2/include $(TOP)/build/$(TARGET)/openssl 
   $(CP) openssl-1.0.2/libcrypto.a $(TOP)/build/$(TARGET)/openssl/lib/. 
   $(CP) openssl-1.0.2/libssl.a $(TOP)/build/$(TARGET)/openssl/lib/. 
```

此代码通过手动构建 OpenSSL 的所有常规步骤工作，然后将生成的二进制文件复制到目标文件夹中。

可能会注意到的一个跨平台构建系统的问题是，像 Autoconf 这样的常见 GNU 工具在 Windows 等操作系统上非常慢，因为它在运行时启动了数百个进程进行测试。即使在 Linux 上，这个过程也可能需要很长时间，这在每天多次运行相同的构建过程中非常令人烦恼且耗时。

理想的情况是拥有一个简单的 Makefile，其中所有内容都预先定义且处于已知状态，因此无需进行库发现等操作。这是将 POCO 库源代码添加到项目中并使用简单的 Makefile 编译它的动机之一：

```cpp
include ../../Makefile.$(TARGET) 

all: poco-foundation poco-json poco-net poco-util 

poco-foundation: 
   cd Foundation && $(MAKE) 

poco-json: 
   cd JSON && $(MAKE) 

poco-net: 
   cd Net && $(MAKE) 

poco-util: 
   cd Util && $(MAKE) 

clean: 
   cd Foundation && $(MAKE) clean 
   cd JSON && $(MAKE) clean 
   cd Net && $(MAKE) clean 
   cd Util && $(MAKE) clean 
```

此 Makefile 然后调用每个模块的单独 Makefile，例如在这个示例中：

```cpp
include ../../../Makefile.$(TARGET) 

OUTPUT = libPocoNet.a 
INCLUDE = $(STD_INCLUDE) -Iinclude 
CFLAGS = $(STD_CFLAGS) $(INCLUDE) 
OBJROOT = $(TOP)/extern/poco-1.7.4/Net/$(TARGET) 
INCLOUT = $(TOP)/build/$(TARGET)/poco 
SOURCES := $(wildcard src/*.cpp) 
HEADERS := $(addprefix $(INCLOUT)/,$(wildcard include/Poco/Net/*.h)) 

OBJECTS := $(addprefix $(OBJROOT)/,$(notdir $(SOURCES:.cpp=.o))) 

all: makedir $(OBJECTS) $(TOP)/build/$(TARGET)/poco/lib/$(OUTPUT) $(HEADERS) 

$(OBJROOT)/%.o: src/%.cpp 
   $(CC) -c -o $@ $< $(CFLAGS) 

makedir: 
   $(MKDIR) $(TARGET) 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/lib 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include/Poco 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include/Poco/Net 

$(INCLOUT)/%.h: %.h 
   $(CP) $< $(INCLOUT)/$< 

$(TOP)/build/$(TARGET)/poco/lib/$(OUTPUT): $(OBJECTS) 
   -rm -f $@ 
   $(AR) rcs $@ $^ 

clean: 
   $(RM) $(OBJECTS) 
```

此 Makefile 编译了库的整个 `Net` 模块。其结构与编译项目核心源文件的 Makefile 类似。除了编译目标文件外，它还将它们放入存档中，以便我们可以在以后链接到它，并将此存档以及头文件复制到构建文件夹中的相应位置。

编译库到项目中的主要原因是允许进行特定的优化和调整，这些优化和调整在预编译库中是不可用的。通过从库的原始构建系统中移除所有非基本内容，尝试不同的设置变得非常容易，甚至在 Windows 上也能工作。

# 在真实硬件上进行远程测试

在我们对代码进行所有本地测试并且合理确信它应该在真实硬件上工作之后，我们可以使用交叉编译构建系统创建一个二进制文件，然后我们可以在目标系统上运行它。

在这一点上，我们可以简单地复制生成的二进制文件和相关文件到目标系统，看看它是否工作。更科学的方法是使用 GDB。在目标 Linux 系统上安装了 GDB 服务器服务后，我们可以通过我们的 PC 连接到它，无论是通过网络还是串行连接。

对于运行基于 Debian 的 Linux 安装的 SBC，GDB 服务器可以很容易地安装：

```cpp
sudo apt install gdbserver  
```

虽然它被称为 `gdbserver`，但其基本功能是作为调试器的远程存根实现，它在主机系统上运行。这使得 `gdbserver` 对于新目标来说非常轻量级且易于实现。

在此之后，我们想要确保 `gdbserver` 正在运行，可以通过登录系统并以多种方式之一启动它。我们可以这样在网络上的 TPC 连接上做到这一点：

```cpp
gdbserver host:2345 <program> <parameters>  
```

或者我们可以将其附加到正在运行的进程上：

```cpp
gdbserver host:2345 --attach <PID>  
```

第一个参数的 `host` 部分指的是将要连接的主机系统的名称（或 IP 地址）。这个参数目前被忽略，这意味着它也可以留空。端口部分必须是一个目标系统上当前未使用的端口。

或者我们可以使用某种串行连接：

```cpp
gdbserver /dev/tty0 <program> <parameters>
gdbserver --attach /dev/tty0 <PID>  
```

当我们启动 `gdbserver` 时，如果目标应用已经在运行，它会暂停其执行，使我们能够从主机系统连接到调试器。在目标系统上，我们可以运行一个去除了调试符号的二进制文件；这些符号必须存在于我们在主机端使用的二进制文件中：

```cpp
$ gdb-multiarch <program>
(gdb) target remote <IP>:<port>
Remote debugging using <IP>:<port>  
```

在这一点上，调试符号将从二进制文件中加载，以及从任何依赖项（如果可用）中加载。通过串行连接连接看起来类似，只是将地址和端口替换为串行接口路径或名称。串行连接的 `baud` 率（如果不是默认的 9,600 波特）在启动时作为参数指定给 GDB：

```cpp
$ gdb-multiarch -baud <baud rate> <program>  
```

一旦我们告诉 GDB 远程目标的详细信息，我们应该看到通常的 GDB 命令行界面出现，允许我们逐步执行、分析和调试程序，就像它在我们的系统上本地运行一样。

如本章前面所述，我们使用 `gdb-multiarch`，因为这个版本的 GDB 调试器支持不同的架构，这在我们将很可能在 x86_64 系统上运行调试器时很有用，而单板计算机（SBC）很可能基于 ARM，但也可能是 MIPS 或 x86（i686）。

除了直接使用 `gdbserver` 运行应用程序外，我们还可以启动 `gdbserver` 以等待调试器连接：

```cpp
gdbserver --multi <host>:<port>  
```

或者我们可以这样做：

```cpp
gdbserver --multi <serial port>  
```

我们将以此方式连接到这个远程目标：

```cpp
$ gdb-multiarch <program>
(gdb) target extended-remote <remote IP>:<port>
(gdb) set remote exec-file <remote file path>
(gdb) run  
```

到目前为止，我们应该再次回到 GDB 命令行界面，程序二进制文件已加载在目标和主机上。

这种方法的优点之一是，当被调试的应用程序退出时，`gdbserver`不会退出。此外，这种模式允许我们在同一目标上同时调试不同的应用程序，前提是目标支持这种模式。

# 摘要

在本章中，我们探讨了如何开发和测试基于操作系统的嵌入式应用程序。我们学习了如何安装和使用交叉编译工具链，如何使用 GDB 进行远程调试，以及如何编写一个构建系统，使我们能够以最小的努力为各种目标系统编译。

到目前为止，你应能够开发并调试基于 Linux 的 SBC 或类似设备的嵌入式应用程序，同时能够以高效的方式工作。

在下一章中，我们将探讨如何为更受限制的基于 MCU 的平台开发和测试应用程序。
