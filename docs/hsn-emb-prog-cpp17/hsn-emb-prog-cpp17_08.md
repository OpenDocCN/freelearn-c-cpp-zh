# 第六章：测试基于操作系统的应用程序

通常，嵌入式系统使用更或多或少常规的操作系统（OS），这意味着在运行时环境和工具方面，嵌入式 Linux 的目标与我们的桌面 OS 大致相同。然而，嵌入式硬件与我们的 PC 在性能和访问方面的差异使得必须考虑在哪里执行开发和测试的各个部分，以及如何将其整合到我们的开发工作流程中。

在本章中，我们将涵盖以下主题：

+   开发跨平台代码

+   在 Linux 下调试和测试跨平台代码

+   有效使用交叉编译器

+   创建支持多个目标的构建系统

# 避免真实硬件

在嵌入式 Linux 等平台上进行基于操作系统的开发的最大优势之一是它与常规桌面 Linux 安装非常相似。特别是在 SoC 上运行像基于 Debian 的 Linux 发行版（Armbian、Raspbian 等）时，我们几乎可以使用相同的工具，只需按几下键即可获得整个软件包管理器、编译器集合和库。

然而，这也是它最大的缺点。

我们可以编写代码，将其复制到 SBC 上，在那里进行编译、运行测试，并在重复该过程之前对代码进行更改。或者，我们甚至可以在 SBC 上编写代码，基本上将其用作我们唯一的开发平台。

我们绝对不应该这样做的主要原因如下：

+   现代 PC 速度更快。

+   在开发的最后阶段之前，不应该在真实硬件上进行测试。

+   自动集成测试变得更加困难。

第一个观点似乎很明显。单核或双核 ARM SoC 编译需要大约一分钟的时间，而在相对现代的多核、多线程处理器（3+ GHz）和支持多核编译的工具链下，从编译开始到链接对象只需要十秒钟或更短的时间。

这意味着，我们不必等待半分钟或更长时间才能运行新的测试或开始调试会话，几乎可以立即进行。

接下来的两点是相关的。虽然在真实硬件上进行测试似乎是有利的，但它也带来了自己的复杂性。其中一点是，这些硬件依赖于许多外部因素才能正常工作，包括其电源供应、电源之间的任何布线、外围设备和信号接口。诸如电磁干扰之类的事物也可能引起问题，包括信号衰减以及由于电磁耦合而触发的中断。

在第三章的俱乐部状态服务项目开发过程中，出现了电磁耦合的一个例子，*为嵌入式 Linux 和类似系统开发*。在这里，开关的一个信号线与 230V 交流电线并排。这些主线布线上电流的变化在信号线上引起脉冲，导致虚假的中断触发事件。

所有这些潜在的与硬件相关的问题表明，这些测试并不像我们希望的那样确定。这可能导致项目开发时间比计划的要长得多，由于冲突和非确定性的测试结果，调试变得更加复杂。

专注于在真实硬件上进行开发的一个影响是，这使得自动化测试变得更加困难。原因在于我们无法使用任何通用的构建集群，例如基于 Linux VM 的测试环境，这在主流的持续集成（CI）服务中很常见。

与此相反，我们必须以某种方式将诸如 SBC 之类的东西整合到 CI 系统中，使其可以交叉编译并将二进制文件复制到 SBC 上进行测试，或者在 SBC 上进行编译，这又回到了第一个观点。

在接下来的几节中，我们将探讨一些方法，使基于嵌入式 Linux 的开发尽可能轻松，从交叉编译开始。

# 为 SBC 进行交叉编译

编译过程将源文件转换为中间格式，然后可以使用此格式来针对特定的 CPU 架构。对我们来说，这意味着我们不仅仅局限于在 SBC 上为 SBC 编译应用程序，而是可以在我们的开发 PC 上进行编译。

要为树莓派（Broadcom Cortex-A 架构的 ARM SoC）这样的 SBC 进行此操作，我们需要安装`arm-linux-gnueabihf`工具链，该工具链针对具有硬件浮点（hardware floating point）支持的 ARM 架构，输出 Linux 兼容的二进制文件。

在基于 Debian 的 Linux 系统上，我们可以使用以下命令安装整个工具链：

```cpp
sudo apt install build-essential
sudo apt install g++-arm-linux-gnueabihf
sudo apt install gdb-multiarch  
```

第一条命令安装了系统的本机基于 GCC 的工具链（如果尚未安装），以及任何常见的相关工具和实用程序，包括`make`，`libtool`，`flex`等。第二条命令安装了实际的交叉编译器。最后，第三个软件包是支持多种架构的 GDB 调试器的版本，我们以后需要用它来在真实硬件上进行远程调试，以及分析应用程序崩溃时产生的核心转储。

我们现在可以在命令行上使用 g++编译器为目标 SBC 使用其完整名称：

```cpp
arm-linux-gnueabihf-g++  
```

为了测试工具链是否正确安装，我们可以执行以下命令，这应该告诉我们编译器的详细信息，包括版本：

```cpp
arm-linux-gnueabihf-g++ -v  
```

除此之外，我们可能需要链接一些存在于目标系统上的共享库。为此，我们可以复制`/lib`和`/usr`文件夹的全部内容，并将其包含为编译器的系统根的一部分：

```cpp
mkdir ~/raspberry/sysroot
scp -r pi@Pi-system:/lib ~/raspberry/sysroot
scp -r pi@Pi-system:/usr ~/raspberry/sysroot  
```

在这里，`Pi-system`是树莓派或类似系统的 IP 地址或网络名称。之后，我们可以告诉 GCC 使用这些文件夹，而不是使用标准路径，使用`sysroot`标志：

```cpp
--sysroot=dir  
```

这里的`dir`将是我们将这些文件夹复制到的文件夹，在这个例子中将是`~/raspberry/sysroot`。

或者，我们可以只复制所需的头文件和库文件，并将它们添加为源树的一部分。哪种方法最容易主要取决于所涉及项目的依赖关系。

对于俱乐部状态服务项目，我们至少需要 WiringPi 的头文件和库，以及 POCO 项目及其依赖项的头文件和库。我们可以确定我们需要的依赖关系，并复制我们之前安装的工具链中缺少的所需包含和库文件。除非有迫切需要这样做，否则最容易的方法是直接从 SBC 的操作系统中复制整个文件夹。

作为使用`sysroot`方法的替代方案，我们还可以在链接代码时明确定义我们希望使用的共享库的路径。当然，这也有其自身的优缺点。

# 俱乐部状态服务的集成测试

为了在进行交叉编译并在真实硬件上测试之前，在常规桌面 Linux（或 macOS 或 Windows）系统上测试俱乐部状态服务，编写了一个简单的集成测试，该测试使用 GPIO 和 I2C 外围设备的模拟。

在第三章中涵盖的项目的源代码中，*为嵌入式 Linux 和类似系统开发*，这些外围设备的文件位于该项目的`wiring`文件夹中。

我们从`wiringPi.h`头文件开始：

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

我们包含了 POCO 框架的一个头文件，以便我们稍后可以轻松创建一个定时器实例。然后，我们定义了所有可能的引脚模式，就像实际的 WiringPi 头文件定义的那样：

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

这些定义进一步定义了引脚模式，包括数字输入电平，引脚上上拉和下拉的可能状态，最后是中断的可能类型，定义中断的触发器：

```cpp
typedef void (*ISRCB)(void); 
```

这个`typedef`定义了中断回调函数指针的格式。

现在让我们看一下`WiringTimer`类：

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

该类是我们模拟实现的 GPIO 端的重要部分。其主要目的是跟踪我们感兴趣的两个中断是否已注册，并使用定时器定期触发它们，正如我们将在下一刻看到的：

```cpp
int wiringPiSetup(); 
void pinMode(int pin, int mode); 
void pullUpDnControl(int pin, int pud); 
int digitalRead(int pin);
int wiringPiISR(int pin, int mode, void (*function)(void));
```

最后，在继续实现之前，我们定义标准的 WiringPi 函数：

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

在类构造函数中，我们在创建定时器实例之前设置默认值，并将其配置为在初始 10 秒延迟后每十秒调用我们的回调函数一次：

```cpp
WiringTimer::~WiringTimer() {
   delete wiringTimer;
   delete cb;
}
```

在析构函数中，我们删除了定时器回调实例：

```cpp
void WiringTimer::start() {
   wiringTimer->start(*cb);
}
```

在这个函数中，我们实际上启动了定时器：

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

该类中的最后一个函数是定时器的回调函数。它的功能是跟踪触发的次数，并将适当的引脚电平设置为我们写入磁盘的文件中的值。

在初始延迟之后，第一个触发器将将锁定开关设置为`false`，第二个将状态开关设置为`true`，第三个将状态开关设置回`false`，最后第四个触发器将锁定开关设置回`true`，然后重置计数器并重新开始：

```cpp
namespace Wiring {
   std::unique_ptr<WiringTimer> wt;
   bool initialized = false;
}
```

我们在其中添加了一个全局命名空间，其中有一个`WiringTimer`类实例的`unique_ptr`实例，以及一个初始化状态指示器。

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

设置函数用于将模拟 GPIO 引脚输入值的默认值写入磁盘。我们还在这里创建了一个`WiringTimer`实例的指针：

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

由于我们的模拟实现确定了引脚的行为，我们可以忽略这些函数的任何输入。为了测试目的，我们可以添加一个断言来验证这些函数在适当的时间以及具有适当的设置被调用：

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

在读取两个模拟引脚之一的值时，我们打开其相应的文件并读取其内容，这是由设置函数或回调设置的 1 或 0：

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

此函数用于注册中断及其关联的回调函数。在通过设置函数初始化模拟后，我们继续注册两个指定引脚中的一个的中断。

一旦两个引脚都设置了中断，我们就启动定时器，定时器将开始生成中断回调的事件。

接下来是 I2C 总线模拟：

```cpp
int wiringPiI2CSetup(const int devId);
int wiringPiI2CWriteReg8(int fd, int reg, int data);
```

这里我们只需要两个函数：设置函数和简单的一字节寄存器写入函数。

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

在设置函数中，我们记录请求的设备 ID（I2C 总线地址），并返回一个标准设备句柄。在这里，我们使用`Club`类中的`log()`函数，使模拟集成到其余代码中：

```cpp
int wiringPiI2CWriteReg8(int fd, int reg, int data) {
    Club::log(LOG_INFO, "wiringPiI2CWriteReg8: Device handle 0x" + NumberFormatter::formatHex(fd) 
                                        + ", Register 0x" + NumberFormatter::formatHex(reg)
                                        + " set to: 0x" + NumberFormatter::formatHex(data));

    return 0;
}
```

由于调用此函数的代码不会期望除了简单的确认数据已被接收之外的响应，我们可以在这里记录接收到的数据和更多细节。同样，为了一致性，这里也使用了 POCO 的`NumberFormatter`类来格式化整数数据为十六进制值，就像在应用程序中一样。

现在我们编译项目并使用以下命令行命令：

```cpp
make TEST=1  
```

现在运行应用程序（在 GDB 下，以查看何时创建/销毁新线程）会得到以下输出：

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

此时，系统已配置所有中断并由应用程序配置了 I2C 设备。定时器已经开始了初始倒计时：

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

GPIO 引脚的初始状态已被读取，两个开关都处于“关闭”位置，因此我们通过将其位置写入寄存器来激活交通灯指示灯上的红灯：

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

接下来，定时器开始触发回调函数，导致它经历不同的阶段。这使我们能够确定代码的基本功能是正确的。

在这一点上，我们可以开始实施更复杂的测试用例，甚至可以使用嵌入式 Lua、Python 运行时或类似的工具来实施可编写脚本的测试用例。

# 模拟与硬件

在模拟大段代码和硬件外设时，一个明显的问题是最终模拟的结果有多现实。显然，我们希望在将测试移至目标系统之前，能够尽可能多地覆盖真实场景的集成测试。

如果我们想知道我们希望在模拟中覆盖哪些测试用例，我们必须同时查看我们的项目需求（它应该能够处理什么）以及真实场景中可能发生的情况和输入。

为此，我们将分析底层代码，看看可能发生什么情况，并决定哪些情况对我们来说是相关的。

在我们之前查看的 WiringPi 模拟中，快速查看库实现的源代码就清楚地表明，与我们将在目标系统上使用的版本相比，我们简化了我们的代码。

查看基本的 WiringPi 设置函数，我们看到它执行以下操作：

+   确定确切的板型和 SoC 以获取 GPIO 布局

+   打开 Linux 设备以进行内存映射的 GPIO 引脚

+   设置 GPIO 设备的内存偏移，并使用`mmap()`将特定的外设（如 PWM、定时器和 GPIO）映射到内存中

与忽略`pinMode()`的调用不同，实现如下：

+   适当设置 SoC 中的硬件 GPIO 方向寄存器（用于输入/输出模式）

+   在引脚上启动 PWM、软 PWM 或 Tone 模式（根据请求）；子函数设置适当的寄存器

这在 I2C 端继续进行，设置函数的实现如下：

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

与我们的模拟实现相比，主要区别在于预期在 OS 的内存文件系统上存在 I2C 外设，并且板子版本确定我们选择哪一个。

最后一个被调用的函数尝试打开设备，因为在 Linux 和类似的操作系统中，每个设备只是一个我们可以打开并获得文件句柄的文件，如果成功的话。这个文件句柄就是函数返回时返回的 ID：

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

打开 I2C 设备后，使用 Linux 系统函数`ioctl()`来向 I2C 外设发送数据，这里是我们希望使用的 I2C 从设备的地址。如果成功，我们会得到一个非负的响应，并返回作为文件句柄的整数。

写入和读取 I2C 总线也使用`ioctl()`来处理，正如我们在同一源文件中所看到的：

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

对于每个 I2C 总线访问，都会调用相同的内联函数。已经选择了我们希望使用的 I2C 设备，我们可以简单地针对 I2C 外设，并让其将有效负载传输到设备上。

这里，`i2c_smbus_data`类型是一个简单的联合体，支持返回值的各种大小（执行读操作时）：

```cpp
union i2c_smbus_data { 
   uint8_t byte; 
   uint16_t word; 
   uint8_t block[I2C_SMBUS_BLOCK_MAX + 2]; 
}; 
```

在这里，我们主要看到使用抽象 API 的好处。如果没有它，我们的代码将充斥着低级调用，这将更难以模拟。我们还看到应该测试的一些条件，例如缺少 I2C 从设备、I2C 总线上的读写错误可能导致意外行为，以及 GPIO 引脚上的意外输入，包括中断引脚，正如本章开头已经指出的那样。

尽管显然不是所有情况都可以预先计划，但应该努力记录所有现实情况，并将其纳入模拟实现中，以便在集成和回归测试以及调试期间可以随时启用它们。

# 使用 Valgrind 进行测试

Valgrind 是用于分析和分析应用程序的缓存和堆行为，以及内存泄漏和潜在多线程问题的开源工具集。它与底层操作系统协同工作，因为根据使用的工具，它必须拦截从内存分配到与多线程相关的指令的一切。这就是为什么它只在 64 位 x86_64 架构的 Linux 下得到完全支持的原因。

在其他支持的平台上使用 Valgrind（如 x86、PowerPC、ARM、S390、MIPS 和 ARM 上的 Linux，以及 Solaris 和 macOS）当然也是一个选择，但 Valgrind 项目的主要开发目标是 x86_64/Linux，这使得它成为进行分析和调试的最佳平台，即使以后会针对其他平台进行定位。

在 Valgrind 网站[`valgrind.org/info/platforms.html`](http://valgrind.org/info/platforms.html)上，我们可以看到当前支持的平台的完整概述。

Valgrind 非常吸引人的一个特性是，它的工具都不需要我们以任何方式修改源代码或生成的二进制文件。这使得它非常容易集成到现有的工作流程中，包括自动化测试和集成系统。

在基于 Windows 的系统上，也有诸如 Dr. Memory（[`drmemory.org/`](http://drmemory.org/)）之类的工具，它们也可以处理与内存相关行为的分析。这个特定的工具还配备了 Dr. Fuzz，一个可以重复调用具有不同输入的函数的工具，可能对集成测试有用。

通过使用像前一节中所看到的集成测试，我们可以自由地从我们的个人电脑上完全分析我们代码的行为。由于 Valgrind 的所有工具都会显著减慢我们代码的执行速度（10-100 倍），能够在快速系统上进行大部分调试和分析意味着我们可以节省大量时间，然后再开始在目标硬件上进行测试。

在我们可能经常使用的工具中，**Memcheck**、**Helgrind**和**DRD**对于检测内存分配和多线程问题非常有用。一旦我们的代码通过了这三个工具，并使用提供代码广泛覆盖的广泛集成测试，我们就可以进行分析和优化。

为了对我们的代码进行分析，我们使用**Callgrind**来查看代码执行时间最长的地方，然后使用**Massif**来对堆分配进行分析。通过这些数据，我们可以对代码进行更改，以简化常见的分配和释放情况。它也可能向我们展示在何处使用缓存以重用资源而不是将其从内存中丢弃是有意义的。

最后，我们将运行另一个循环的 MemCheck、Helgrind 和 DRD，以确保我们的更改没有引起任何退化。一旦我们满意，我们就会部署代码到目标系统上，并查看其在那里的表现。

如果目标系统也运行 Linux 或其他支持的操作系统，我们也可以在那里使用 Valgrind，以确保我们没有遗漏任何东西。根据确切的平台（操作系统和 CPU 架构），我们可能会遇到 Valgrind 针对该平台的限制。这些可能包括*未处理的指令*等错误，其中工具尚未实现 CPU 指令，因此 Valgrind 无法继续。

通过将集成测试扩展到使用 SBC 而不是本地进程，我们可以建立一个持续集成系统，除了在本地进程上进行测试外，还可以在真实硬件上运行测试，考虑到真实硬件平台相对于用于大部分测试的基于 x86_64 的 Linux 系统的限制。

# 多目标构建系统

交叉编译和多目标构建系统是一些让很多人感到恐惧的词语，主要是因为它们让人联想到需要神秘咒语才能执行所需操作的复杂构建脚本。在本章中，我们将看一个基于简单 Makefile 的构建系统，该构建系统已在一系列硬件目标的商业项目中得到应用。

使构建系统易于使用的一件事是能够轻松设置所有相关方面的编译，并且有一个中心位置，我们可以从中控制项目的所有相关方面，或者部分相关方面，以及构建和运行测试。

因此，我们在项目顶部只有一个 Makefile，它处理所有基本内容，包括确定我们运行的平台。我们在这里做的唯一简化是假设类 Unix 环境，使用 MSYS2 或 Cygwin 在 Windows 上，以及 Linux、BSD 和 OS X/macOS 等使用其本机 shell 环境。然而，我们也可以适应 Microsoft Visual Studio、Intel Compiler Collection（ICC）和其他编译器，只要它们提供基本工具。

构建系统的关键是简单的 Makefile，在其中我们定义目标平台的具体细节，例如，对于在 x86_x64 硬件上运行的标准 Linux 系统：

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

在这里，我们可以设置我们将用于编译、创建存档、从二进制文件中剥离调试符号等操作的命令行工具的名称。构建系统将使用目标操作系统和架构来保持创建的二进制文件分开，以便我们可以使用相同的源树在一次运行中为所有目标平台创建二进制文件。

我们可以看到我们将传递给编译器和链接器的标志分为不同的类别：特定于平台的标志，常见（标准）标志，最后是特定于 C 和 C ++编译器的标志。前者在集成已集成到源树中的外部依赖项时非常有用，但这些依赖项是用 C 编写的。我们将在`extern`文件夹中找到这些依赖项，稍后我们将更详细地看到。

这种类型的文件将被大量定制以适应特定项目，添加所需的包含文件、库和编译标志。对于这个示例文件，我们可以看到一个使用 POCO 和 Boost 库以及 OpenSSL 的项目，调整 POCO 库以适应目标平台。

首先，让我们看一下 macOS 配置文件的顶部：

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

尽管文件的其余部分几乎相同，但在这里我们可以看到一个很好的例子，说明了如何将工具的名称泛化。尽管 Clang 支持与 GCC 相同的标志，但其工具的名称不同。通过这种方法，我们只需在这个文件中写入不同的名称一次，一切都会正常工作。

这继续了 ARM 目标上的 Linux，它被设置为交叉编译目标：

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

在这里，我们看到了之前在本章中看到的用于 ARM Linux 平台的交叉编译工具链的再次出现。为了节省输入，我们定义了基本名称一次，以便重新定义。这也展示了 Makefile 的灵活性。通过更多的创造力，我们可以创建一组模板，将整个工具链泛化为一个简单的 Makefile，该 Makefile 将根据平台的 Makefile（或其他配置文件）中的提示包含在主 Makefile 中，从而使其高度灵活。

接下来，我们将看一下项目根目录中的主 Makefile：

```cpp
ifndef TARGET
 $(error TARGET parameter not provided.)
 endif
```

由于我们无法猜测用户希望我们针对哪个平台进行目标，我们要求指定目标，并将平台名称作为值，例如`linux-x86_x64`：

```cpp
export TOP := $(CURDIR)
 export TARGET
```

稍后在系统中，我们需要知道本地文件系统中的文件夹位置，以便我们可以指定绝对路径。我们使用标准的 Make 变量，并将其导出为我们自己的环境变量，以及构建目标名称：

```cpp
UNAME := $(shell uname)
 ifeq ($(UNAME), Linux)
 export HOST = linux
 else
 export HOST = win32
 export FILE_EXT = .exe
 endif
```

使用（命令行）`uname`命令，我们可以检查我们正在运行的操作系统，每个支持该命令的操作系统在其 shell 中返回其名称，例如 Linux 用于 Linux，Darwin 用于 macOS。在纯 Windows 上（没有 MSYS2 或 Cygwin），该命令不存在，这将得到我们这个`if/else`语句的第二部分。

这个语句可以扩展以支持更多的操作系统，具体取决于构建系统的要求。在这种情况下，它仅用于确定我们创建的可执行文件是否应该有文件扩展名：

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

在这个`if/else`语句中，我们可以为常见的文件操作设置适当的命令行命令。由于我们采取了简单的方式，我们假设在 Windows 上使用 MSYS2 或类似的 Bash shell。

在这一点上，我们可以进一步推广概念，将 OS 文件 CLI 工具作为自己的一组 Makefiles 拆分出来，然后将其作为 OS 特定设置的一部分包含进来：

```cpp
include Makefile.$(TARGET)

 export TARGET_OS
 export TARGET_ARCH
 export TOOLCHAIN_NAME
```

在这一点上，我们使用提供给 Makefile 的目标参数来包含适当的配置文件。在从中导出一些细节之后，我们现在有了一个配置好的构建系统：

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

通过这个单一的 Makefile，我们可以选择编译整个项目，或者只是依赖项或核心项目。我们还可以编译特定的外部依赖项，而不编译其他内容。

最后，我们可以清理核心项目、依赖项或两者。

这个顶级 Makefile 主要用于控制底层 Makefiles。接下来的两个 Makefiles 分别位于`Core`和`extern`文件夹中。其中，`Core` Makefile 直接编译项目的核心部分：

```cpp
include ../Makefile.$(TARGET) 

OUTPUT := CoreProject 

INCLUDE = $(STD_INCLUDE) 
LIBDIRS = $(STD_LIBDIRS) 

include ../version 
VERSIONINFO = -D__VERSION="\"$(VERSION)\"" 
```

作为第一步，我们包含目标平台的 Makefile 配置，以便我们可以访问其所有定义。这些也可以在主 Makefile 中导出，但这样我们可以自由定制构建系统。

我们指定正在构建的输出二进制文件的名称，然后执行一些小任务，包括在项目根目录中使用 Makefile 语法打开`version`文件，其中包含我们正在构建的源代码的版本号。这准备作为预处理器定义传递给编译器：

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

这是另一个部分，我们依赖于有一个 Bash shell 或类似的东西，因为我们使用 date 命令来为构建创建时间戳。格式取决于传递给主 Makefile 的参数。如果我们正在构建一个发布版本，我们将从 Git 存储库中获取时间戳，使用 Git 提交标签名称来检索该标签的提交时间戳，然后进行格式化。

如果传递了`GITTIME`作为参数，则使用最近的 Git 提交的时间戳。否则，使用当前的时间和日期（UTC）。

这段代码旨在解决测试和集成构建中出现的一个问题：跟踪构建的时间和源代码的修订版本。只要它支持检索特定时间戳的类似功能，它就可以适应其他文件修订系统。

值得注意的是我们正在创建的第二个时间戳。这是一个稍微不同格式的时间戳，附加到生成的二进制文件上，除非我们是在发布模式下构建：

```cpp
CFLAGS = $(STD_CFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 
CXXFLAGS = $(STD_CXXFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 

OBJROOT := $(TOP)/build/$(TARGET)/obj 
CPP_SOURCES := $(wildcard *.cpp) 
CPP_OBJECTS := $(addprefix $(OBJROOT)/,$(CPP_SOURCES:.cpp=.o)) 
OBJECTS := $(CPP_OBJECTS) 
```

在这里，我们设置希望传递给编译器的标志，包括版本和时间戳，两者都作为预处理器定义传递。

最后，我们收集当前项目文件夹中的源文件，并设置对象文件的输出文件夹。正如我们在这里看到的，我们将把对象文件写入项目根目录下的一个文件夹中，并根据编译目标进行进一步分离。

```cpp
.PHONY: all clean 

all: makedirs $(CPP_OBJECTS) $(C_OBJECTS) $(TOP)/build/bin/$(TARGET)/$(OUTPUT)_$(VERSION)_$(TARGET)$(TS_SAFE) 

makedirs: 
   $(MKDIR) $(TOP)/build/bin/$(TARGET) 
   $(MKDIR) $(OBJROOT) 

$(OBJROOT)/%.o: %.cpp 
   $(CXX) -o $@ $< $(CXXFLAGS) 
```

这部分对于 Makefile 来说是相当通用的。我们有`all`目标，以及一个用于在文件系统上创建文件夹（如果尚不存在）的目标。最后，我们在下一个目标中接收源文件数组，根据配置编译它们，并将对象文件输出到适当的文件夹中：

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

在我们从源文件创建了所有的目标文件之后，我们希望将它们链接在一起，这就是这一步发生的地方。我们还可以看到二进制文件将会出现在哪里：在项目构建文件夹的`bin`子文件夹中。

链接器被调用，我们创建了生成二进制文件的副本，我们用`.debug`后缀来表示它是带有所有调试信息的版本。然后，原始二进制文件被剥离其调试符号和其他不需要的信息，留下一个小的二进制文件复制到远程测试系统，以及一个带有所有调试信息的较大版本，以便在需要分析核心转储或进行远程调试时使用。

我们在这里看到的另一个特点是由于 Clang 的链接器不支持的命令行标志而添加的一个小技巧，需要实现一个特殊情况。在跨平台编译和类似任务中，人们很可能会遇到许多这样的小细节，所有这些都会使得编写一个简单工作的通用构建系统变得复杂。

```cpp
clean: 
   $(RM) $(CPP_OBJECTS) 
   $(RM) $(C_OBJECTS) 
```

最后一步是允许删除生成的目标文件。

`extern`中的第二个子 Makefile 也值得注意，因为它控制所有底层依赖关系：

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

这里的一个有趣特性是基于目标平台的依赖选择器。如果我们有不应该为特定平台编译的依赖关系，我们可以在这里跳过它们。这个特性还允许我们直接指示这个 Makefile 为特定平台编译所有依赖关系。在这里，我们允许针对 QNX、Linux、OS X/macOS 和 Windows 进行定位，同时忽略架构：

```cpp
libboost: 
   cd boost-1.58.0 && $(MAKE) 

poco: 
   cd poco-1.7.4 && $(MAKE) 
```

实际的目标只是调用依赖项目顶部的另一个 Makefile，然后编译该依赖项并将其添加到构建文件夹中，以便`Core`的 Makefile 使用。

当然，我们也可以直接使用现有的构建系统从这个 Makefile 编译项目，比如这里的 OpenSSL：

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

这段代码通过手动完成构建 OpenSSL 的所有常规步骤，然后将生成的二进制文件复制到它们的目标文件夹。

人们可能会注意到跨平台构建系统的一个问题是，像 Autoconf 这样的常见 GNU 工具在 Windows 等操作系统上非常慢，因为它在运行数百个测试时会启动许多进程。即使在 Linux 上，这个过程也可能需要很长时间，当一天中多次运行相同的构建过程时，这是非常令人恼火和耗时的。

理想情况是有一个简单的 Makefile，其中一切都是预定义的，并且处于已知状态，因此不需要库发现等。这是将 POCO 库源代码添加到一个项目并有一个简单的 Makefile 编译它的动机之一：

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

这个 Makefile 然后调用每个模块的单独 Makefile，就像这个例子：

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

这个 Makefile 编译了整个库的`Net`模块。它的结构类似于用于编译项目核心源文件的结构。除了编译目标文件，它还将它们放入一个存档中，以便我们以后可以链接，并将这个存档以及头文件复制到它们在构建文件夹中的位置。

为了允许特定的优化和调整，编译库的主要原因是这些优化和调整在预编译库中是不可用的。通过从库的原始构建系统中剥离除了基本内容之外的所有内容，尝试不同的设置变得非常容易，甚至在 Windows 上也可以工作。

# 在真实硬件上进行远程测试

在我们完成了所有代码的本地测试，并且相当确信它应该可以在真实硬件上运行之后，我们可以使用交叉编译构建系统来创建一个二进制文件，然后在目标系统上运行。

在这一点上，我们可以简单地将生成的二进制文件和相关文件复制到目标系统上，看看它是否有效。更科学的方法是使用 GDB。通过在目标 Linux 系统上安装 GDB 服务器服务，我们可以通过网络或串行连接从 PC 连接到它。

对于运行基于 Debian 的 Linux 安装的 SBC，GDB 服务器可以很容易地安装：

```cpp
sudo apt install gdbserver  
```

尽管它被称为`gdbserver`，但其基本功能是作为调试器的远程存根实现，在主机系统上运行。这使得`gdbserver`非常轻量级和简单，可以为新目标实现。

之后，我们要确保`gdbserver`正在运行，方法是登录到系统并以各种方式启动它。我们可以像这样为网络上的 TPC 连接这样做：

```cpp
gdbserver host:2345 <program> <parameters>  
```

或者我们可以将其附加到正在运行的进程上：

```cpp
gdbserver host:2345 --attach <PID>  
```

第一个参数的`主机`部分是将要连接的主机系统的名称（或 IP 地址）。当前该参数被忽略，这意味着它也可以留空。端口部分必须是目标系统上当前未使用的端口。

或者我们可以使用某种串行连接：

```cpp
gdbserver /dev/tty0 <program> <parameters>
gdbserver --attach /dev/tty0 <PID>  
```

一旦我们启动`gdbserver`，它会暂停目标应用程序的执行（如果它已经在运行），从而允许我们从主机系统连接调试器。在目标系统上，我们可以运行一个已经剥离了其调试符号的二进制文件；这些符号需要在我们在主机端使用的二进制文件中存在：

```cpp
$ gdb-multiarch <program>
(gdb) target remote <IP>:<port>
Remote debugging using <IP>:<port>  
```

在这一点上，调试符号将从二进制文件中加载，以及从任何依赖项中加载（如果可用）。通过串行连接进行连接看起来类似，只是地址和端口被串行接口路径或名称替换。当我们启动时，串行连接的`波特率`（如果不是默认的 9600 波特率）被指定为 GDB 的参数：

```cpp
$ gdb-multiarch -baud <baud rate> <program>  
```

一旦我们告诉 GDB 远程目标的详细信息，我们应该看到通常的 GDB 命令行界面出现，允许我们像在本地系统上运行一样步进，分析和调试程序。

正如本章前面提到的，我们使用`gdb-multiarch`，因为这个版本的 GDB 调试器支持不同的架构，这很有用，因为我们很可能会在 x86_64 系统上运行调试器，而 SBC 很可能是基于 ARM，但也可能是 MIPS 或 x86（i686）。

除了直接使用`gdbserver`运行应用程序之外，我们还可以启动`gdbserver`等待调试器连接：

```cpp
gdbserver --multi <host>:<port>  
```

或者我们可以这样做：

```cpp
gdbserver --multi <serial port>  
```

然后我们会像这样连接到这个远程目标：

```cpp
$ gdb-multiarch <program>
(gdb) target extended-remote <remote IP>:<port>
(gdb) set remote exec-file <remote file path>
(gdb) run  
```

在这一点上，我们应该再次发现自己处于 GDB 命令行界面上，目标和主机上都加载了程序二进制文件。

这种方法的一个重要优势是`gdbserver`在被调试的应用程序退出时不会退出。此外，这种模式允许我们在同一个目标上同时调试不同的应用程序，假设目标支持这一点。

# 总结

在本章中，我们学习了如何开发和测试嵌入式操作系统应用程序。我们学会了如何安装和使用交叉编译工具链，如何使用 GDB 进行远程调试，以及如何编写构建系统，使我们能够以最小的工作量为新目标系统进行编译。

在这一点上，您应该能够以高效的方式开发和调试基于 Linux 的 SBC 或类似系统的嵌入式应用程序。

在下一章中，我们将学习如何为更受限制的基于 MCU 的平台开发和测试应用程序。
