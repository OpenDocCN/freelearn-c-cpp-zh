# 第九章：外围设备

与外围设备的通信是任何嵌入式应用的重要部分。应用程序需要检查可用性和状态，并向各种设备发送数据和接收数据。

每个目标平台都不同，连接外围设备到计算单元的方式有很多种。然而，有几种硬件和软件接口已经成为与外围设备通信的行业标准。在本章中，我们将学习如何处理直接连接到处理器引脚或串行接口的外围设备。本章涵盖以下主题：

+   通过 GPIO 控制连接的设备

+   探索脉宽调制

+   使用 ioctl 访问 Linux 中的实时时钟

+   使用 libgpiod 控制 GPIO 引脚

+   控制 I2C 外围设备

本章的配方涉及与真实硬件的交互，并打算在真实的树莓派板上运行。

# 通过 GPIO 控制连接的设备

**通用输入输出**（GPIO）是将外围设备连接到 CPU 的最简单方式。每个处理器通常都有一些用于通用目的的引脚。这些引脚可以直接与外围设备的引脚电连接。嵌入式应用可以通过改变配置为输出的引脚的信号电平或读取输入引脚的信号电平来控制设备。

信号电平的解释不遵循任何协议，而是由外围设备确定。开发人员需要查阅设备数据表以便正确地编程通信。

这种类型的通信通常是在内核端使用专用设备驱动程序完成的。然而，这并不总是必需的。在这个配方中，我们将学习如何从用户空间应用程序中使用树莓派板上的 GPIO 接口。

# 如何做...

我们将创建一个简单的应用程序，控制连接到树莓派板上的通用引脚的**发光二极管**（LED）：

1.  在你的`~/test`工作目录中，创建一个名为`gpio`的子目录。

1.  使用你喜欢的文本编辑器在`gpio`子目录中创建一个`gpio.cpp`文件。

1.  将以下代码片段放入文件中：

```cpp
#include <chrono>
#include <iostream>
#include <thread>
#include <wiringPi.h>

using namespace std::literals::chrono_literals;
const int kLedPin = 0;

int main (void)
{
  if (wiringPiSetup () <0) {
    throw std::runtime_error("Failed to initialize wiringPi");
  }

  pinMode (kLedPin, OUTPUT);
  while (true) {
    digitalWrite (kLedPin, HIGH);
    std::cout << "LED on" << std::endl;
    std::this_thread::sleep_for(500ms) ;
    digitalWrite (kLedPin, LOW);
    std::cout << "LED off" << std::endl;
    std::this_thread::sleep_for(500ms) ;
  }
  return 0 ;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(gpio)
add_executable(gpio gpio.cpp)
target_link_libraries(gpio wiringPi)
```

1.  使用[WiringPI 示例](http://wiringpi.com/examples/blink/)部分的说明，将 LED 连接到树莓派板上。

1.  建立一个 SSH 连接到你的树莓派板。按照[Raspberry Pi 文档](https://www.raspberrypi.org/documentation/remote-access/ssh/)部分的说明进行操作。

1.  通过 SSH 将`gpio`文件夹的内容复制到树莓派板上。

1.  通过 SSH 登录到板上，然后构建和运行应用程序：

```cpp
$ cd gpio && cmake . && make && sudo ./gpio
```

你的应用程序应该运行，你应该能够观察到 LED 在闪烁。

# 工作原理...

树莓派板有 40 个引脚（第一代有 26 个）可以使用**内存映射输入输出**（MMIO）机制进行编程。MMIO 允许开发人员通过读取或写入系统物理内存中的特定地址来查询或设置引脚的状态。

在第六章的*使用专用内存*配方中，*内存管理*，我们学习了如何访问 MMIO 寄存器。在这个配方中，我们将把 MMIO 地址的操作交给专门的库`wiringPi`。它隐藏了内存映射和查找适当偏移量的所有复杂性，而是暴露了一个清晰的 API。

这个库已经预装在树莓派板上，所以为了简化构建过程，我们将直接在板上构建代码，而不是使用交叉编译。与其他教程不同，我们的构建规则没有提到交叉编译器 - 我们将使用板上的本机 ARM 编译器。我们只添加了对`wiringPi`库的依赖：

```cpp
target_link_libraries(gpio wiringPi)
```

这个示例的代码是对`wiringPi`用于 LED 闪烁的示例的修改。首先，我们初始化`wiringPi`库：

```cpp
if (wiringPiSetup () < 0) {
    throw std::runtime_error("Failed to initialize wiringPi");
}
```

接下来，我们进入无限循环。在每次迭代中，我们将引脚设置为`HIGH`状态：

```cpp
    digitalWrite (kLedPin, HIGH);
```

在 500 毫秒的延迟之后，我们将相同的引脚设置为`LOW`状态并添加另一个延迟：

```cpp
 digitalWrite (kLedPin, LOW);
    std::cout << "LED off" << std::endl;
 std::this_thread::sleep_for(500ms) ;
```

我们配置程序使用引脚`0`，对应于树莓派的`BCM2835`芯片的`GPIO.0`或引脚`17`：

```cpp
const int kLedPin = 0;
```

如果 LED 连接到这个引脚，它将会闪烁，打开 0.5 秒，然后关闭 0.5 秒。通过调整循环中的延迟，您可以改变闪烁模式。

由于程序进入无限循环，我们可以通过在 SSH 控制台中按下*Ctrl* + *C*来随时终止它；否则，它将永远运行。

当我们运行应用程序时，我们只会看到以下输出：

![](img/9f3a257e-977a-4e63-97e3-c39452e84ce4.png)

我们记录 LED 打开或关闭的时间，但要检查程序是否真正工作，我们需要查看连接到引脚的 LED。如果我们按照接线说明，就可以看到它是如何工作的。当程序运行时，板上的 LED 会与程序输出同步闪烁：

![](img/72b7b80d-860c-48ec-ba6c-bb4aaee983d5.png)

我们能够控制直接连接到 CPU 引脚的简单设备，而无需编写复杂的设备驱动程序。

# 探索脉宽调制

数字引脚只能处于两种状态之一：`HIGH`或`LOW`。连接到数字引脚的 LED 也只能处于两种状态之一：`on`或`off`。但是有没有办法控制 LED 的亮度？是的，我们可以使用一种称为**脉宽调制**（**PWM**）的方法。

PWM 背后的想法很简单。我们通过周期性地打开或关闭电信号来限制电信号传递的功率。这使得信号以一定频率脉冲，并且功率与脉冲宽度成正比 - 即信号处于`HIGH`状态的时间。

例如，如果我们将引脚设置为`HIGH` 10 微秒，然后在循环中再设置为`LOW` 90 微秒，连接到该引脚的设备将接收到原本的 10%的电源。

在这个教程中，我们将学习如何使用 PWM 来控制连接到树莓派板数字 GPIO 引脚的 LED 的亮度。

# 操作步骤如下...

我们将创建一个简单的应用程序，逐渐改变连接到树莓派板上的通用引脚的 LED 的亮度：

1.  在您的`~/test`工作目录中，创建一个名为`pwm`的子目录。

1.  使用您喜欢的文本编辑器在`pwm`子目录中创建一个名为`pwm.cpp`的文件。

1.  让我们添加所需的`include`函数并定义一个名为`Blink`的函数：

```cpp
#include <chrono>
#include <thread>

#include <wiringPi.h>

using namespace std::literals::chrono_literals;

const int kLedPin = 0;

void Blink(std::chrono::microseconds duration, int percent_on) {
    digitalWrite (kLedPin, HIGH);
    std::this_thread::sleep_for(
            duration * percent_on / 100) ;
    digitalWrite (kLedPin, LOW);
    std::this_thread::sleep_for(
            duration * (100 - percent_on) / 100) ;
}
```

1.  接下来是一个`main`函数：

```cpp
int main (void)
{
  if (wiringPiSetup () <0) {
    throw std::runtime_error("Failed to initialize wiringPi");
  }

  pinMode (kLedPin, OUTPUT);

  int count = 0;
  int delta = 1;
  while (true) {
    Blink(10ms, count);
    count = count + delta;
    if (count == 101) {
      delta = -1;
    } else if (count == 0) {
      delta = 1;
    }
  }
  return 0 ;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(pwm)
add_executable(pwm pwm.cpp)
target_link_libraries(pwm wiringPi)
```

1.  按照[`wiringpi.com/examples/blink/`](http://wiringpi.com/examples/blink/)中的*WiringPI 示例*部分的说明，将 LED 连接到树莓派板上。

1.  建立 SSH 连接到您的树莓派板。请按照[`www.raspberrypi.org/documentation/remote-access/ssh/`](https://www.raspberrypi.org/documentation/remote-access/ssh/)中的*Raspberry PI 文档*部分的说明进行操作。

1.  通过 SSH 将`pwm`文件夹的内容复制到树莓派板上。

1.  通过 SSH 登录到板上，然后构建和运行应用程序：

```cpp
$ cd pwm && cmake . && make && sudo ./pwm
```

您的应用程序现在应该运行，您可以观察 LED 的闪烁。

# 工作原理...

这个配方重用了从前一个配方中闪烁 LED 的代码和原理图。我们将这段代码从`main`函数移动到一个新函数`Blink`中。

`Blink`函数接受两个参数——`duration`和`percent_on`：

```cpp
void Blink(std::chrono::microseconds duration, int percent_on)
```

`duration`确定脉冲的总宽度（以微秒为单位）。`percent_on`定义了信号为`HIGH`时的时间与脉冲总持续时间的比例。

实现很简单。当调用`Blink`时，它将引脚设置为`HIGH`并等待与`percent_on`成比例的时间：

```cpp
    digitalWrite (kLedPin, HIGH);
    std::this_thread::sleep_for(
            duration * percent_on / 100);
```

之后，它将引脚设置为`LOW`并等待剩余时间：

```cpp
    digitalWrite (kLedPin, LOW);
    std::this_thread::sleep_for(
            duration * (100 - percent_on) / 100);
```

`Blink`是实现 PWM 的主要构建块。我们可以通过将`percent_on`从`0`变化到`100`来控制亮度，如果我们选择足够短的`duration`，我们将看不到任何闪烁。

电视或监视器的刷新率相等或短于持续时间是足够好的。对于 60 赫兹，持续时间为 16.6 毫秒。我们使用 10 毫秒以简化。

接下来，我们将所有内容包装在另一个无限循环中，但现在它有另一个参数`count`：

```cpp
  int count = 0;
```

它在每次迭代中更新，并在`0`和`100`之间反弹。`delta`变量定义了变化的方向——减少或增加——以及变化的量，在我们的情况下始终为`1`：

```cpp
  int delta = 1;
```

当计数达到`101`或`0`时，方向会改变：

```cpp
    if (count == 101) {
      delta = -1;
    } else if (count == 0) {
      delta = 1;
    }
```

在每次迭代中，我们调用`Blink`，传递`10ms`作为脉冲和`count`作为定义 LED 开启时间的比例，因此它的亮度（如下图所示）：

```cpp
    Blink(10ms, count);
```

![](img/98a8f41e-0940-43fa-82fe-09b2c45f7fb0.png)

由于更新频率高，我们无法确定 LED 何时从开启到关闭。

当我们将所有东西连接起来并运行程序时，我们可以看到 LED 逐渐变亮或变暗。

# 还有更多...

PWM 广泛用于嵌入式系统，用于各种目的。这是伺服控制和电压调节的常见机制。使用*脉宽调制*维基百科页面，网址为[`en.wikipedia.org/wiki/Pulse-width_modulation`](https://en.wikipedia.org/wiki/Pulse-width_modulation)，作为了解更多关于这种技术的起点。

# 使用 ioctl 访问 Linux 中的实时时钟

在我们之前的配方中，我们使用 MMIO 从用户空间 Linux 应用程序访问外围设备。然而，这种接口不是用户空间应用程序和设备驱动程序之间通信的推荐方式。

在类 Unix 操作系统（如 Linux）中，大多数外围设备可以以与常规文件相同的方式访问，使用所谓的设备文件。当应用程序打开设备文件时，它可以从中读取，从相应设备获取数据，或者向其写入，向设备发送数据。

在许多情况下，设备驱动程序无法处理非结构化的数据流。它们期望以请求和响应的形式组织的数据交换，其中每个请求和响应都有特定和固定的格式。

这种通信由`ioctl`系统调用来处理。它接受一个设备相关的请求代码作为参数。它还可能包含其他参数，用于编码请求数据或提供输出数据的存储。这些参数特定于特定设备和请求代码。

在这个配方中，我们将学习如何在用户空间应用程序中使用`ioctl`与设备驱动程序进行数据交换。

# 如何做...

我们将创建一个应用程序，从连接到树莓派板的**实时时钟**（**RTC**）中读取当前时间：

1.  在您的`~/test`工作目录中，创建一个名为`rtc`的子目录。

1.  使用您喜欢的文本编辑器在`rtc`子目录中创建一个名为`rtc.cpp`的文件。

1.  让我们把所需的`include`函数放到`rtc.cpp`文件中：

```cpp
#include <iostream>
#include <system_error>

#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/rtc.h>
```

1.  现在，我们定义一个名为`Rtc`的类，它封装了对真实时钟设备的通信：

```cpp
class Rtc {
  int fd;
  public:
    Rtc() {
      fd = open("/dev/rtc", O_RDWR);
      if (fd < 0) {
        throw std::system_error(errno,
            std::system_category(),
            "Failed to open RTC device");
      }
    }

    ~Rtc() {
      close(fd);
    }

    time_t GetTime(void) {
      union {
        struct rtc_time rtc;
        struct tm tm;
      } tm;
      int ret = ioctl(fd, RTC_RD_TIME, &tm.rtc);
      if (ret < 0) {
        throw std::system_error(errno,
            std::system_category(),
            "ioctl failed");
      }
      return mktime(&tm.tm);
    }
};
```

1.  一旦类被定义，我们将一个简单的使用示例放入`main`函数中：

```cpp
int main (void)
{
  Rtc rtc;
  time_t t = rtc.GetTime();
  std::cout << "Current time is " << ctime(&t)
            << std::endl;

  return 0 ;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(rtc)
add_executable(rtc rtc.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  构建您的应用程序并将生成的`rtc`二进制文件复制到我们的树莓派模拟器中。

# 工作原理...

我们正在实现一个直接与连接到系统的硬件 RTC 通信的应用程序。系统时钟和 RTC 之间存在差异。系统时钟仅在系统运行时处于活动状态并维护。当系统关闭电源或进入睡眠模式时，系统时钟变得无效。即使系统关闭，RTC 也处于活动状态。它维护用于在系统启动时配置系统时钟的实际时间。此外，它可以被编程为在睡眠模式下的特定时间唤醒系统。

我们将所有与 RTC 驱动程序的通信封装到一个名为`Rtc`的类中。与驱动程序的所有数据交换都通过`/dev/rtc`特殊设备文件进行。在`Rtc`类构造函数中，我们打开设备文件并将结果文件描述符存储在`fd`实例变量中：

```cpp
  fd = open("/dev/rtc", O_RDWR);
```

同样，析构函数用于关闭文件：

```cpp
    ~Rtc() {
      close(fd);
    }
```

由于设备在析构函数中关闭，一旦`Rtc`实例被销毁，我们可以使用**资源获取即初始化**（RAII）习惯用法在出现问题时抛出异常而不泄漏文件描述符：

```cpp
      if (fd < 0) {
        throw std::system_error(errno,
            std::system_category(),
            "Failed to open RTC device");
      }
```

我们的类只定义了一个成员函数—`GetTime`。它是在`RTC_RD_TIME` `ioctl`调用之上的一个包装器。此调用期望返回一个`rtc_time`结构以返回当前时间。它几乎与我们将要用来将 RTC 驱动程序返回的时间转换为 POSIX 时间戳格式的`tm`结构相同，因此我们将它们都放入相同的内存位置作为`union`数据类型：

```cpp
      union {
        struct rtc_time rtc;
        struct tm tm;
      } tm;
```

通过这种方式，我们避免了从一个结构复制相同字段到另一个结构。

数据结构准备就绪后，我们调用`ioctl`调用，将`RTC_RD_TIME`常量作为请求 ID 传递，并将指向我们结构的指针作为存储数据的地址传递：

```cpp
  int ret = ioctl(fd, RTC_RD_TIME, &tm.rtc);
```

成功后，`ioctl`返回`0`。在这种情况下，我们使用`mktime`函数将结果数据结构转换为`time_t` POSIX 时间戳格式：

```cpp
  return mktime(&tm.tm);
```

在`main`函数中，我们创建了`Rtc`类的一个实例，然后调用`GetTime`方法：

```cpp
  Rtc rtc;
  time_t t = rtc.GetTime();
```

自从 POSIX 时间戳表示自 1970 年 1 月 1 日以来的秒数，我们使用`ctime`函数将其转换为人类友好的表示，并将结果输出到控制台：

```cpp
  std::cout << "Current time is " << ctime(&t)
```

当我们运行我们的应用程序时，我们可以看到以下输出：

![](img/b7640217-c2f3-4c53-b5b8-c7901c07760f.png)

我们能够直接从硬件时钟使用`ioctl`读取当前时间。`ioctl` API 在 Linux 嵌入式应用中被广泛使用，用于与设备通信。

# 更多内容

在我们的简单示例中，我们学习了如何只使用一个`ioctl`请求。RTC 设备支持许多其他请求，可用于设置闹钟，更新时间和控制 RTC 中断。更多细节可以在[`linux.die.net/man/4/rtc`](https://linux.die.net/man/4/rtc)的*RTC ioctl 文档*部分找到。

# 使用 libgpiod 控制 GPIO 引脚

在前面的教程中，我们学习了如何使用`ioctl` API 访问 RTC。我们可以使用它来控制 GPIO 引脚吗？答案是肯定的。最近，Linux 添加了一个通用 GPIO 驱动程序，以及一个用户空间库`libgpiod`，通过在通用`ioctl` API 之上添加一个便利层来简化对连接到 GPIO 的设备的访问。此接口允许嵌入式开发人员在任何基于 Linux 的平台上管理其设备，而无需编写设备驱动程序。此外，它提供了 C++的绑定。

结果，尽管仍然被广泛使用，但`wiringPi`库已被弃用，因为其易于使用的接口。

在本教程中，我们将学习如何使用`libgpiod` C++绑定。我们将使用相同的 LED 闪烁示例来查看`wiringPi`和`libgpiod`方法的差异和相似之处。

# 如何做...

我们将创建一个应用程序，使用新的`libgpiod` API 来闪烁连接到树莓派板的 LED。

1.  在您的`~/test`工作目录中，创建一个名为`gpiod`的子目录。

1.  使用您喜欢的文本编辑器在`gpiod`子目录中创建一个`gpiod.cpp`文件。

1.  将应用程序的代码放入`rtc.cpp`文件中：

```cpp
#include <chrono>
#include <iostream>
#include <thread>

#include <gpiod.h>
#include <gpiod.hpp>

using namespace std::literals::chrono_literals;

const int kLedPin = 17;

int main (void)
{

  gpiod::chip chip("gpiochip0");
  auto line = chip.get_line(kLedPin);
  line.request({"test",
                 gpiod::line_request::DIRECTION_OUTPUT, 
                 0}, 0);

  while (true) {
    line.set_value(1);
    std::cout << "ON" << std::endl;
    std::this_thread::sleep_for(500ms);
    line.set_value(0);
    std::cout << "OFF" << std::endl;
    std::this_thread::sleep_for(500ms);
  }

  return 0 ;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(gpiod)
add_executable(gpiod gpiod.cpp)
target_link_libraries(gpiod gpiodcxx)
```

1.  使用[Raspberry PI documentation](http://wiringpi.com/examples/blink/)中的*WiringPI 示例*部分的说明，将 LED 连接到您的树莓派板。

1.  建立一个 SSH 连接到您的树莓派板。请按照[Raspberry PI documentation](https://www.raspberrypi.org/documentation/remote-access/)中的说明进行操作。

1.  通过 SSH 将`gpio`文件夹的内容复制到树莓派板上。

1.  安装`libgpiod-dev`软件包：

```cpp
$ sudo apt-get install gpiod-dev
```

1.  通过 SSH 登录到板上，然后构建和运行应用程序：

```cpp
$ cd gpiod && cmake . && make && sudo ./gpiod
```

您的应用程序应该运行，您可以观察 LED 闪烁。

# 它是如何工作的...

我们的应用程序使用了 Linux 中访问 GPIO 设备的新的推荐方式。由于它是最近才添加的，因此需要安装最新版本的 Raspbian 发行版`buster`。

`gpiod`库本身提供了用于使用`ioctl` API 与 GPIO 内核模块通信的高级包装。该接口设计用于 C 语言，其上还有一个用于 C++绑定的附加层。这一层位于`libgpiocxx`库中，它是`libgpiod2`软件包的一部分，与 C 的`libgpiod`库一起提供。

该库使用异常来报告错误，因此代码简单且不会被返回代码检查所淹没。此外，我们不需要担心释放捕获的资源；它会通过 C++ RAII 机制自动完成。

应用程序启动时，它创建了一个 chip 类的实例，该类作为 GPIO 通信的入口点。它的构造函数接受要使用的设备的名称：

```cpp
  gpiod::chip chip("gpiochip0");
```

接下来，我们创建一个 line 的实例，它代表一个特定的 GPIO 引脚：

```cpp
  auto line = chip.get_line(kLedPin);
```

请注意，与`wiringPi`实现不同，我们传递了`17`引脚号，因为`libgpiod`使用本机 Broadcom SOC 通道（**BCM**）引脚编号：

```cpp
const int kLedPin = 17;
```

创建 line 实例后，我们需要配置所需的访问模式。我们构造一个`line_request`结构的实例，传递一个消费者的名称（`"test"`）和一个指示引脚配置为输出的常量：

```cpp
  line.request({"test",
                 gpiod::line_request::DIRECTION_OUTPUT, 
                 0}, 0);
```

之后，我们可以使用`set_value`方法更改引脚状态。与`wiringPi`示例一样，我们将引脚设置为`1`或`HIGH`，持续`500ms`，然后再设置为`0`或`LOW`，再持续`500ms`，循环进行：

```cpp
    line.set_value(1);
    std::cout << "ON" << std::endl;
    std::this_thread::sleep_for(500ms);
    line.set_value(0);
    std::cout << "OFF" << std::endl;
    std::this_thread::sleep_for(500ms);
```

该程序的输出与*通过 GPIO 连接的设备进行控制*配方的输出相同。代码可能看起来更复杂，但新的 API 更通用，可以在任何 Linux 板上工作，而不仅仅是树莓派。

# 还有更多...

有关`libgpiod`和 GPIO 接口的更多信息，可以在[`github.com/brgl/libgpiod`](https://github.com/brgl/libgpiod)找到。

# 控制 I2C 外设设备

通过 GPIO 连接设备有一个缺点。处理器可用于 GPIO 的引脚数量有限且相对较小。当您需要处理大量设备或提供复杂功能的设备时，很容易用完引脚。

解决方案是使用标准串行总线之一连接外围设备。其中之一是**Inter-Integrated Circuit**（**I2C**）。由于其简单性和设备可以仅用两根导线连接到主控制器，因此这被广泛用于连接各种低速设备。

总线在硬件和软件层面都得到了很好的支持。通过使用 I2C 外设，开发人员可以在用户空间应用程序中控制它们，而无需编写复杂的设备驱动程序。

在这个教程中，我们将学习如何在树莓派板上使用 I2C 设备。我们将使用一款流行且便宜的 LCD 显示器。它有 16 个引脚，这使得它直接连接到树莓派板变得困难。然而，通过 I2C 背包，它只需要四根线来连接。

# 操作步骤...

我们将创建一个应用程序，该应用程序在连接到我们的树莓派板的 1602 LCD 显示器上显示文本：

1.  在你的`~/test`工作目录中，创建一个名为`i2c`的子目录。

1.  使用你喜欢的文本编辑器在`i2c`子目录中创建一个`i2c.cpp`文件。

1.  将以下`include`指令和常量定义放入`i2c.cpp`文件中：

```cpp
#include <thread>
#include <system_error>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

using namespace std::literals::chrono_literals;

enum class Function : uint8_t {
  clear = 0x01,
  home = 0x02,
  entry_mode_set = 0x04,
  display_control = 0x08,
  cursor_shift = 0x10,
  fn_set = 0x20,
  set_ddram_addr = 0x80
};

constexpr int En = 0b00000100;
constexpr int Rs = 0b00000001;

constexpr int kDisplayOn = 0x04;
constexpr int kEntryLeft = 0x02;
constexpr int kTwoLine = 0x08;
constexpr int kBacklightOn = 0x08;
```

1.  现在，我们定义一个新的类`Lcd`，它封装了显示控制逻辑。我们从数据字段和`public`方法开始：

```cpp
class Lcd {
  int fd;

  public:
    Lcd(const char* device, int address) {
      fd = open(device, O_RDWR);
      if (fd < 0) {
        throw std::system_error(errno,
            std::system_category(),
            "Failed to open RTC device");
      }
      if (ioctl(fd, I2C_SLAVE, address) < 0) {
        close(fd);
        throw std::system_error(errno,
            std::system_category(),
            "Failed to aquire bus address");
      }
      Init();
    }

    ~Lcd() {
      close(fd);
    }

    void Clear() {
      Call(Function::clear);
      std::this_thread::sleep_for(2000us);
    }

    void Display(const std::string& text,
                 bool second=false) {
      Call(Function::set_ddram_addr, second ? 0x40 : 0);
      for(char c : text) {
        Write(c, Rs);
      }
    }
```

1.  接下来是`private`方法。低级辅助方法首先出现：

```cpp
private:

    void SendToI2C(uint8_t byte) {
 if (write(fd, &byte, 1) != 1) {
 throw std::system_error(errno,
 std::system_category(),
 "Write to i2c device failed");
 }
    }

    void SendToLcd(uint8_t value) {
      value |= kBacklightOn;
      SendToI2C(value);
      SendToI2C(value | En);
      std::this_thread::sleep_for(1us);
      SendToI2C(value & ~En);
      std::this_thread::sleep_for(50us);
    }

    void Write(uint8_t value, uint8_t mode=0) {
      SendToLcd((value & 0xF0) | mode);
      SendToLcd((value << 4) | mode);
    }
```

1.  一旦辅助函数被定义，我们添加更高级的方法：

```cpp
    void Init() {
      // Switch to 4-bit mode
      for (int i = 0; i < 3; i++) {
        SendToLcd(0x30);
        std::this_thread::sleep_for(4500us);
      }
      SendToLcd(0x20);

      // Set display to two-line, 4 bit, 5x8 character mode
      Call(Function::fn_set, kTwoLine);
      Call(Function::display_control, kDisplayOn);
      Clear();
      Call(Function::entry_mode_set, kEntryLeft);
      Home();
    }

    void Call(Function function, uint8_t value=0) {
      Write((uint8_t)function | value);
    }

    void Home() {
      Call(Function::home);
      std::this_thread::sleep_for(2000us);
    }
};
```

1.  添加使用`Lcd`类的`main`函数：

```cpp
int main (int argc, char* argv[])
{
  Lcd lcd("/dev/i2c-1", 0x27);
  if (argc > 1) {
    lcd.Display(argv[1]);
    if (argc > 2) {
      lcd.Display(argv[2], true);
    }
  }
  return 0 ;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(i2c)
add_executable(i2c i2c.cpp)
```

1.  根据这个表格，将你的 1602LCD 显示器的`i2c`背包上的引脚连接到树莓派板上的引脚：

| **树莓派引脚名称** | **物理引脚号** | **1602 I2C 引脚** |
| --- | --- | --- |
| GND | 6 | GND |
| +5v | 2 | VSS |
| SDA.1 | 3 | SDA |
| SCL.1 | 5 | SCL |

1.  建立 SSH 连接到你的树莓派板。按照[Raspberry PI documentation](https://www.raspberrypi.org/documentation/remote-access/ssh/)部分的说明进行操作。

1.  登录到树莓派板并运行`raspi-config`工具以启用`i2c`：

```cpp
sudo raspi-config
```

1.  在菜单中，选择 Interfacing Options | I2C | Yes。

1.  重新启动板以激活新设置。

1.  通过 SSH 将`i2c`文件夹的内容复制到树莓派板上。

1.  通过 SSH 登录到板上，然后构建和运行应用程序：

```cpp
$ cd i2c && cmake . && make && ./i2c Hello, world!
```

你的应用程序应该运行，你可以观察到 LED 在闪烁。

# 工作原理...

在这个教程中，我们的外围设备——LCD 屏幕——通过 I2C 总线连接到板上。这是一种串行接口，所以连接只需要四根物理线。然而，LCD 屏幕可以做的远不止简单的 LED。这意味着用于控制它的通信协议也更复杂。

我们将只使用 1602 LCD 屏幕提供的功能的一小部分。通信逻辑松散地基于 Arduino 的`LiquidCrystal_I2C`库，适用于树莓派。

我们定义了一个`Lcd`类，它隐藏了 I2C 通信的所有复杂性和 1602 控制协议的私有方法。除了构造函数和析构函数之外，它只公开了两个公共方法：`Clear`和`Display`。

在 Linux 中，我们通过设备文件与 I2C 设备通信。要开始使用设备，我们需要使用常规的打开调用打开与 I2C 控制器对应的设备文件：

```cpp
fd = open(device, O_RDWR);
```

可能有多个设备连接到同一总线。我们需要选择要通信的设备。我们使用`ioctl`调用来实现这一点：

```cpp
if (ioctl(fd, I2C_SLAVE, address) < 0) {
```

此时，I2C 通信已配置，我们可以通过向打开的文件描述符写入数据来发出 I2C 命令。然而，这些命令对于每个外围设备都是特定的。因此，在通用 I2C 初始化之后，我们需要继续进行 LCD 初始化。

我们将所有 LCD 特定的初始化放入`Init`私有函数中。它配置操作模式、行数和显示字符的大小。为此，我们定义了辅助方法、数据类型和常量。

基本的辅助函数是`SendToI2C`。它是一个简单的方法，将数据字节写入配置为 I2C 通信的文件描述符，并在出现错误时抛出异常。

```cpp
      if (write(fd, &byte, 1) != 1) {
        throw std::system_error(errno,
            std::system_category(),
            "Write to i2c device failed");
      }
```

除了`SendToI2C`之外，我们还定义了另一个辅助方法`SendToLcd`。它向 I2C 发送一系列字节，形成 LCD 控制器可以解释的命令。这涉及设置不同的标志并处理数据块之间需要的延迟：

```cpp
      SendToI2C(value);
      SendToI2C(value | En);
      std::this_thread::sleep_for(1us);
      SendToI2C(value & ~En);
      std::this_thread::sleep_for(50us);
```

LCD 以 4 位模式工作，这意味着发送到显示器的每个字节都需要两个命令。我们定义`Write`方法来为我们执行这些操作：

```cpp
      SendToLcd((value & 0xF0) | mode);
      SendToLcd((value << 4) | mode);
```

最后，我们定义设备支持的所有可能命令，并将它们放入`Function`枚举类中。`Call`辅助函数可以用于以类型安全的方式调用函数：

```cpp
    void Call(Function function, uint8_t value=0) {
      Write((uint8_t)function | value);
    }
```

最后，我们使用这些辅助函数来定义清除屏幕和显示字符串的公共方法。

由于通信协议的所有复杂性都封装在`Lcd`类中，我们的`main`函数相对简单。

它创建了一个类的实例，传入我们将要使用的设备文件名和设备地址。默认情况下，带有 I2C 背包的 1620 LCD 的地址是`0x27`：

```cpp
  Lcd lcd("/dev/i2c-1", 0x27);
```

`Lcd`类的构造函数执行所有初始化，一旦实例被创建，我们就可以调用`Display`函数。我们不是硬编码要显示的字符串，而是使用用户通过命令行参数传递的数据。第一个参数显示在第一行。如果提供了第二个参数，它也会显示在显示器的第二行：

```cpp
    lcd.Display(argv[1]);
    if (argc > 2) {
      lcd.Display(argv[2], true);
    }
```

我们的程序已经准备好了，我们可以将其复制到树莓派板上并在那里构建。但在运行之前，我们需要将显示器连接到板上并启用 I2C 支持。

我们使用`raspi-config`工具来启用 I2C。我们只需要做一次，但除非之前未启用 I2C，否则需要重新启动：

![](img/6ea36835-ced3-40ef-a8c8-70b0c08c2f71.png)

最后，我们可以运行我们的应用程序。它将在 LCD 显示器上显示以下输出：

![](img/91251e94-ad99-47c6-b5c3-e5866ca97b1e.jpg)

现在，我们知道如何从 Linux 用户空间程序控制通过 I2C 总线连接的设备。

# 还有更多...

有关使用 I2C 设备的更多信息，请访问[`elinux.org/Interfacing_with_I2C_Devices`](https://elinux.org/Interfacing_with_I2C_Devices.)上的*与 I2C 设备接口*页面。
