# 8

# 系统滴答（SysTick）定时器

在本章中，我们将学习关于**系统滴答（SysTick）**定时器的内容，这是所有 Arm Cortex 微控制器的一个重要核心外设。我们将从介绍 SysTick 定时器及其最常见用途开始。随后，我们将详细探讨 SysTick 定时器的寄存器。最后，我们将开发一个 SysTick 定时器的驱动程序。

本章我们将涵盖以下主要主题：

+   SysTick 定时器简介

+   开发 SysTick 定时器驱动程序

到本章结束时，你将很好地理解 SysTick 定时器，并能够有效地在 Arm Cortex-M 项目中实现和利用它。

# 技术要求

本章的所有代码示例都可以在 GitHub 上找到，地址为

[`github.com/PacktPublishing/Bare-Metal-Embedded-C-Programming`](https://github.com/PacktPublishing/Bare-Metal-Embedded-C-Programming)。

# SysTick 定时器简介

系统滴答（SysTick）定时器，通常称为 SysTick，是所有 Arm Cortex 微控制器的核心组件。无论处理器核心是 Cortex-M0、Cortex-M1 还是 Cortex-M7，以及硅制造商是 STMicroelectronics、Texas Instruments 还是其他任何公司，每个 Arm Cortex 微控制器都包含一个 SysTick 定时器。在本节中，我们将了解这个基本的外设并详细探讨其寄存器。

## SysTick 定时器概述

SysTick 定时器是所有 Arm Cortex-M 处理器的**24 位向下计数器**。它被设计为提供可配置的时间基准，可用于各种目的，如**任务调度**、**系统监控**和**时间跟踪**。这个定时器为我们提供了一种简单高效的方法来**生成周期性中断**，并作为实现系统定时功能的基础，包括为**实时操作系统（RTOS）**生成**操作系统（OS）**滴答。使用 SysTick 使得我们的代码更具可移植性，因为它属于核心，而不是供应商特定的外设。

SysTick 定时器的关键特性包括以下内容：

+   **24 位可重载计数器**：计数器从指定值递减到零，然后自动重载以提供连续的定时操作

+   **核心集成**：作为核心的一部分，它需要最小的配置并提供低延迟的中断处理

+   **可配置的时钟源**：SysTick 可以从核心时钟或外部参考时钟运行，提供在定时精度和功耗方面的灵活性

+   **中断生成**：当计数器达到零时，它可以触发一个中断

SysTick 定时器通常有三个主要用途：

+   **操作系统滴答生成**：在实时操作系统（RTOS）环境中，SysTick 通常用于生成系统滴答中断，驱动操作系统调度器

+   **周期性任务执行**：它可以用来触发定期任务，如传感器采样或通信检查

+   **时间延迟函数**：SysTick 可以在固件中提供精确的延迟，用于各种定时功能

现在，让我们探索 SysTick 定时器中的寄存器。

## SysTick 定时器寄存器

SysTick 定时器由四个主要寄存器组成：

+   SysTick 控制和状态寄存器（`SYST_CSR`）

+   SysTick 重载值寄存器（`SYST_RVR`）

+   SysTick 当前值寄存器（`SYST_CVR`）

+   SysTick 校准值寄存器（`SYST_CALIB`）

让我们逐一分析它们，从控制和状态寄存器开始。

### SysTick 控制和状态寄存器（SYST_CSR）

`SYST_CSR` 寄存器控制 SysTick 定时器的操作并提供状态信息。它具有以下位：

+   **ENABLE（位 0）**：启用或禁用 SysTick 计数器

+   **TICKINT（位 1）**：启用或禁用 SysTick 中断

+   **CLKSOURCE（位 2）**：选择时钟源（0 = 外部参考时钟，1 = 处理器时钟）

+   **COUNTFLAG（位 16）**：指示自上次读取以来计数器是否已达到零（1 = 是，0 = 否）

这是 SysTick 控制和状态寄存器的结构：

![图 8.1：SysTick 控制和状态寄存器](img/B21914_08_1.jpg)

图 8.1：SysTick 控制和状态寄存器

下一个寄存器是 SysTick 重载值寄存器（`SYST_RVR`）。

### SysTick 重载值寄存器（SYST_RVR）

此寄存器指定要加载到 SysTick 当前值寄存器的起始值。这对于设置定时器的周期至关重要，并且理解其位分配和计算对于有效的 SysTick 配置是必不可少的。

它具有以下字段：

+   **位 [31:24] 保留**：这些位是保留的

+   当计数器启用且计数器达到零时，`SYST_CVR` 寄存器

这是 SysTick 重载值寄存器的结构：

![图 8.2：SysTick 重载值寄存器](img/B21914_08_2.jpg)

图 8.2：SysTick 重载值寄存器

由于 SysTick 是一个 24 位定时器，`RELOAD` 值可以是 **0x00000001** 到 **0x00FFFFFF** 范围内的任何值。

要根据所需的定时器周期计算 `RELOAD` 值，我们确定所需周期的时钟周期数，然后从该数字中减去 1 以获得 `RELOAD` 值。

例如，如果 `RELOAD` 值计算如下：

1.  计算 1 毫秒内的时钟周期数：

    **时钟周期 = 16,000,000 个周期/秒 * 0.001 秒 =** **16,000 个周期**

    注意：1ms = 0.001 秒

1.  从计算出的时钟周期数中减去 1：

    **RELOAD = 16,000 - 1 = 15,999**，因为从 0 到 15,999 的计数将给我们 16000 个滴答。

意味着，为了配置 SysTick 定时器以 1 ms 的时间周期和 16 MHz 的时钟，我们将 `RELOAD` 值设置为 `SYST_CVR`）。

### SysTick 当前值寄存器

SysTick 当前值寄存器（`SYST_CVR`）保存 SysTick 计数器的当前值。我们可以使用此寄存器来监控倒计时过程，并在必要时重置计数器。

它具有以下字段：

+   **位[31:24]保留**：这些位是保留位，不应修改。它们必须写为零。

+   `COUNTFLAG`位在 SysTick 控制和状态寄存器（`SYST_CSR`）中。这是 SysTick 当前值寄存器：

![图 8.3：SysTick 当前值寄存器](img/B21914_08_3.jpg)

图 8.3：SysTick 当前值寄存器

### SysTick 校准值寄存器

SysTick 定时器的最后一个寄存器是 SysTick 校准值寄存器（`SYST_CALIB`）。此寄存器为我们提供了 SysTick 定时器的校准属性。

这些寄存器的名称在 STM32 头文件中略有不同。*表 8.1*提供了在*Arm 通用用户指南*文档中使用的寄存器名称与 STM32 特定头文件中使用的寄存器名称之间的清晰对应关系。这种对应关系将帮助我们正确理解和在我们的代码中引用它们。

| **功能** | **Arm 通用** **用户指南** | **STM32** **头文件** |
| --- | --- | --- |
| 控制和状态 | SYST_CSR | SysTick->CTRL |
| 重载值 | SYST_RVR | SysTick->LOAD |
| 当前值 | SYST_CVR | SysTick->VAL |
| 校准值 | SYST_CALIB | SysTick->CALIB |

表 8.1：SysTick 寄存器名称对应关系

在下一节中，我们将使用我们所学到的知识开发 SysTick 定时器的驱动程序。

# 开发 SysTick 定时器的驱动程序

在本节中，我们将开发一个用于生成精确延迟的 SysTick 定时器驱动程序。

首先，在我们的集成开发环境（IDE）中按照我们在*第七章*中学到的步骤，复制我们上一个项目。将复制的项目重命名为`SysTick`。接下来，在`Src`文件夹中创建一个名为`sttyck.c`的新文件，在`Inc`文件夹中创建一个名为`sttyck.h`的新文件，就像我们在上一课中为 GPIO 驱动器所做的那样。

使用以下代码填充你的`systick.c`文件：

```cpp
#include "systick.h"
#define CTRL_ENABLE        (1U<<0)
#define CTRL_CLCKSRC    (1U<<2)
#define CTRL_COUNTFLAG    (1U<<16)
/*By default, the frequency of the MCU is 16Mhz*/
#define ONE_MSEC_LOAD     16000
void systick_msec_delay(uint32_t delay)
{
    /*Load number of clock cycles per millisecond*/
    SysTick->LOAD =  ONE_MSEC_LOAD - 1;
    /*Clear systick current value register*/
    SysTick->VAL = 0;
    /*Select internal clock source*/
    SysTick->CTRL = CTRL_CLCKSRC;
    /*Enable systick*/
    SysTick->CTRL |=CTRL_ENABLE;
    for(int i = 0; i < delay; i++)
    {
        while((SysTick->CTRL & CTRL_COUNTFLAG) == 0){}
    }
    /*Disable systick*/
    SysTick->CTRL = 0;
}
```

让我们将其分解。

我们从包含头文件开始：

```cpp
#include "systick.h"
```

这行代码包含了头文件`systick.h`，它反过来包含`stm32fxx.h`以提供对寄存器定义的访问。

接下来，我们定义所有需要的宏：

+   `#define CTRL_ENABLE (1U << 0)`: 宏定义用于启用 SysTick 定时器。

+   `#define CTRL_CLKSRC (1U << 2)`: 宏定义选择 SysTick 定时器的内部时钟源。

+   `#define CTRL_COUNTFLAG (1U << 16)`: 宏定义用于检查`COUNTFLAG`位，该位指示定时器已计数到零。

+   `#define ONE_MSEC_LOAD 16000`: 宏定义 1 毫秒中的时钟周期数。这假设微控制器的时钟频率为 16 MHz。这是 NUCLEO-F411 开发板的默认配置。

接下来，我们进入函数实现。

首先，我们有以下内容：

```cpp
SysTick->LOAD = ONE_MSEC_LOAD - 1;
```

这行代码将 SysTick 定时器加载为 1 毫秒的时钟周期数。

然后，我们使用以下方法清除当前值寄存器以重置定时器：

```cpp
SysTick->VAL = 0;
```

接下来，我们选择内部时钟源：

```cpp
SysTick->CTRL = CTRL_CLKSRC;
```

要启用 SysTick 定时器，我们使用以下方法：

```cpp
SysTick->CTRL |= CTRL_ENABLE;
```

现在，我们进入处理延迟的循环：

```cpp
for (int i = 0; i < delay; i++)
{
    while ((SysTick->CTRL & CTRL_COUNTFLAG) == 0) {}
}
```

此循环运行指定的延迟时间。在每个迭代中，它等待 `COUNTFLAG` 位被设置，这表示定时器已计数到零。

最后，我们禁用 SysTick 定时器：

```cpp
SysTick->CTRL = 0;
```

就这样！通过这些步骤，我们已成功使用 SysTick 定时器实现了延迟函数。

我们接下来的任务是填充 `systick.h` 文件。

这里是代码：

```cpp
#ifndef SYSTICK_H_
#define SYSTICK_H_
#include <stdint.h>
#include "stm32f4xx.h"
void systick_msec_delay(uint32_t delay);
#endif
```

在这里，需要 `#include <stdint.h>` 指令以确保我们可以访问由 C 标准库提供的标准整数类型定义。这些定义包括固定宽度整数类型，如 `uint32_t`、`int32_t`、`uint16_t` 等，这对于编写可移植和清晰的代码至关重要，尤其是在嵌入式系统编程中。

驱动文件完成后，我们现在可以在 `main.c` 中进行测试。

首先，让我们通过添加一个新的切换 LED 的函数来增强我们的 `gpio.c` 文件。这将通过允许我们通过单个函数调用切换 LED 而不是分别调用 `led_on()` 和 `led_off()` 来简化我们的代码。

将以下函数添加到你的 `gpio.c` 文件中：

```cpp
#define LED_PIN            (1U<<5)
void led_toggle(void)
{
    /*Toggle PA5*/
    GPIOA->ODR ^=LED_PIN;
}
```

此函数通过在 **输出数据寄存器**（**ODR**）上执行位异或操作来切换连接到 PA5 引脚的 LED 的状态。

接下来，在 `gpio.h` 文件中声明此函数，添加以下行：

```cpp
void led_toggle(void)
```

最后，按照以下示例更新你的 `main.c` 文件以调用 SysTick 延迟和 LED 切换函数：

```cpp
#include "gpio.h"
#include "systick.h"
int main(void)
{
    /*Initialize LED*/
    led_init();
    while(1){
        /*Delay for 500ms*/
        systick_msec_delay(500);
           /* Toggle the LED */
        led_toggle();
    }
}
```

在本例中，我们以 500ms 的时间间隔 **切换 LED**。构建项目并在你的开发板上运行它。你应该看到绿色 LED 闪烁。为了进一步实验，你可以修改延迟值并观察 LED 闪烁速率的变化。

# 摘要

在本章中，我们探讨了 SysTick 定时器，这是所有 Arm Cortex 微控制器的核心外设。我们首先介绍了 SysTick 定时器，讨论了其重要性和常见应用，例如在实时操作系统中生成 OS 拓扑，执行周期性任务，以及提供精确的时间延迟。

我们随后详细检查了 SysTick 定时器的寄存器。这些包括控制状态寄存器（`SYST_CSR`），它管理定时器的操作和状态；重载值寄存器（`SYST_RVR`），它设置定时器的倒计时周期；当前值寄存器（`SYST_CVR`），它持有倒计时的当前值；以及校准值寄存器（`SYST_CALIB`），它提供了精确计时所需的基本校准属性。我们还提供了 Arm 通用用户指南中使用的寄存器名称与 STM32 头文件中使用的寄存器名称之间的比较，以确保准确编码的清晰对应。

本章以开发一个 SysTick 定时器驱动程序结束。我们回顾了`systick_msec_delay`函数的创建和实现过程，该函数通过 SysTick 定时器引入毫秒级延迟。为了测试该驱动程序，我们将其与 GPIO 功能集成，以切换我们的绿色 LED，展示了如何在嵌入式系统中实现精确的时序和控制。

在下一章中，我们将学习另一个定时器外设。与 SysTick 定时器不同，这个定时器外设的配置是针对 STM32 微控制器特定的。
