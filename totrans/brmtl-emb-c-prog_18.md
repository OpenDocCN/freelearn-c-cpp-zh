# 18

# 嵌入式系统中的电源管理和能源效率

在本章中，我们将深入研究嵌入式系统中的电源管理和能源效率，这是当今技术驱动世界的一个关键方面。有效的电源管理对于延长电池寿命和确保嵌入式设备的最佳性能至关重要。本章旨在为你提供实施有效电源管理技术的必要知识和技能。

我们将首先探讨各种电源管理技术，为理解如何在嵌入式系统中降低功耗奠定基础。随后，我们将检查 STM32F4 微控制器中可用的不同睡眠模式和低功耗状态，提供它们配置和应用的详细见解。然后，我们将讨论 STM32F4 中的唤醒源和触发器，这对于确保微控制器能够迅速响应外部事件至关重要。最后，我们将通过开发一个进入待机模式和唤醒微控制器的驱动程序来将理论应用于实践，展示如何在现实场景中应用这些概念。

在本章中，我们将涵盖以下主要主题：

+   电源管理技术概述

+   STM32F4 中的低功耗模式

+   STM32F4 中的唤醒源和触发器

+   开发进入待机模式和唤醒的驱动程序

到本章结束时，你将彻底理解嵌入式系统中的电源管理，并能够使用 STM32F4 微控制器实现节能设计。这些知识将使你能够创建优化功耗并延长电池寿命的嵌入式系统，这对于现代应用至关重要。

# 技术要求

本章的所有代码示例都可以在 GitHub 上找到，链接为[`github.com/PacktPublishing/Bare-Metal-Embedded-C-Programming`](https://github.com/PacktPublishing/Bare-Metal-Embedded-C-Programming)。

# 电源管理技术概述

在本节中，我们将探索电源管理技术的世界，这是嵌入式系统设计的一个关键方面。随着我们的设备变得更加先进，我们对电池寿命的期望也在增加，了解如何有效管理电源比以往任何时候都更重要。让我们深入了解各种电源管理技术及其实现方式，通过一些案例研究来观察这些技术在实际中的应用。

嵌入式系统中的电源管理涉及一系列硬件和软件策略，旨在降低能耗。这对于电池供电设备尤为重要，高效的电源使用可以显著延长电池寿命。我们将涵盖的主要技术包括**动态电压和频率缩放**（**DVFS**）、时钟门控、电源门控以及利用低功耗模式。

让我们从 DVFS 开始。

## 动态电压和频率缩放（DVFS）

DVFS 是一种根据工作负载调整微控制器电压和频率的方法。在低活动期间降低电压和频率可以大大减少功耗。相反，在高需求期间，电压和频率会增加以确保性能。

### DVFS 是如何实现的？

在 STM32 微控制器中，可以通过特定的电源控制寄存器管理 DVFS。这些寄存器允许系统根据所需的性能级别动态调整工作点。例如，STM32F4 系列具有几种可以配置以调整系统时钟和核心电压的电源模式。

### 一个用例示例——智能手机

智能手机是 DVFS 应用的典型例子。当手机处于空闲状态时，它会降低 CPU 频率和电压以节省电池。一旦开始使用应用程序或玩游戏，CPU 会提高其频率和电压以提供必要的性能。这种性能与节能之间的平衡使得现代智能手机如此高效。

另一种常见的技术是时钟门控。

## 时钟门控

时钟门控是一种技术，当微控制器的某些部分未使用时，会关闭其时钟信号。这防止了不必要的晶体管切换，从而节省了电力。

### 时钟门控是如何实现的？

时钟门控通常通过时钟控制寄存器进行控制。在 STM32 系列中，可以通过这些寄存器单独启用或禁用每个外设的时钟。例如，如果某个外设如 ADC 不需要，其时钟可以被禁用以节省功耗。

### 一个用例示例——智能家居设备

智能家居设备，如智能恒温器或灯光，使用时钟门控来高效管理电力。这些设备大部分时间处于低功耗状态，仅在执行特定任务时唤醒。通过关闭未使用外设的时钟，这些设备可以节约能源并延长电池寿命。

另一种技术是电源门控。

## 电源门控

电源门控通过完全关闭微控制器某些部分的电源，将节能提升到一个新的层次。这种技术确保了关闭部分的零功耗。

### 电源门控是如何实现的？

电源门控比时钟门控更复杂，通常涉及微控制器内部的专用电源管理单元。这些单元控制微控制器各个域的电源。在 STM32 微控制器中，可以通过电源控制寄存器配置电源门控，以关闭特定的外设，甚至整个微控制器的部分。

### 一个用例示例——可穿戴设备

可穿戴设备，如健身追踪器，从电源门控技术中受益匪浅。这些设备需要单次充电长时间运行。通过在不用时关闭传感器和其他组件的电源，可穿戴设备可以在不牺牲功能的情况下实现更长的电池寿命。

接下来，让我们讨论低功耗模式。

## 低功耗模式

低功耗模式是微控制器中预定义的状态，通过禁用或减少各种组件的功能，可以显著降低功耗。这些模式从简单的 CPU 睡眠模式到更复杂的深度睡眠或待机模式不等。

### 如何实现低功耗模式？

低功耗模式通过电源控制寄存器实现。例如，STM32F4 微控制器提供多种低功耗模式，包括**睡眠**、**停止**和**待机**。每种模式都提供了不同的节能和唤醒时间之间的平衡。

### 一个用例示例 – 远程传感器

在农业或环境监测中使用的远程传感器通常使用低功耗模式。这些传感器可能大部分时间处于低功耗状态，定期唤醒以进行测量和传输数据。通过利用低功耗模式，这些传感器可以在单次电池充电下运行数月甚至数年。

现在，让我们更详细地研究几个案例研究，这些研究说明了这些电源管理技术在现实世界应用中的组合使用。

## 案例研究 1 – 高效智能手表

智能手表是依赖电源管理技术的一个很好的例子。这些设备需要在性能和电池寿命之间取得平衡，因为用户期望它们在单次充电下运行数天。让我们分析一下不同技术在高效智能手表设计中的作用：

+   **动态电压频率调整（DVFS）**：智能手表使用 DVFS 根据当前的工作负载调整 CPU 频率。当用户与手表交互时，CPU 频率增加以提供流畅的体验。当手表空闲时，频率降低以节省电力。

+   **时钟门控**：如 GPS 或心率监测器等外围设备仅在需要时供电。当这些功能不使用时，它们的时钟被门控以节省能源。

+   **电源门控**：当显示器关闭时，像显示驱动器这样的组件会完全断电。

+   **低功耗模式**：在非活动期间，手表进入深度睡眠模式，仅在检查通知或用户交互时唤醒。

通过结合这些技术，智能手表可以在不牺牲功能的情况下实现令人印象深刻的电池寿命。另一个优秀的例子是太阳能环境监测。

## 案例研究 2 – 太阳能环境监测器

部署在偏远地区的太阳能环境监测器必须高效运行，以确保连续的数据收集和传输。其角色如下：

+   **动态电压频率调整（DVFS）**：监控器根据阳光强度和电池充电量调整其工作频率。在阳光最强烈的小时，它以更高的频率运行以处理更多数据。

+   **时钟门控**：例如温度、湿度和空气质量等传感器仅在数据收集间隔期间活跃。当不使用时，这些传感器的时钟会被门控。

+   **电源门控**：在夜间或阴天期间，非必要组件完全关闭电源以节省能源。

+   **低功耗模式**：监控器在数据收集间隔之间进入深度睡眠模式，定期醒来进行测量和传输数据。

通过这些电源管理技术，监控器可以自主运行很长时间，仅依靠太阳能。

电源管理是嵌入式系统设计的重要方面，尤其是在设备变得更加便携和依赖电池的情况下。通过理解和实施 DVFS、时钟门控、电源门控和低功耗模式等技术，我们可以设计出既强大又节能的嵌入式系统。无论是智能手表、远程传感器还是任何其他电池供电设备，有效的电源管理确保了更长的电池寿命和更好的整体性能。随着我们不断推动嵌入式系统能力的边界，掌握这些电源管理技术将比以往任何时候都更加重要。

在下一节中，我们将探索 STM32F4 微控制器中的低功耗模式。

## STM32F4 的低功耗模式

在本节中，我们将学习 STM32F4 微控制器中可用的低功耗模式。我们将涵盖各种低功耗模式、如何配置它们以及在项目中使用它们的实际方面。

让我们从了解这些低功耗模式开始。STM32F4 微控制器中的低功耗模式旨在通过禁用或限制某些组件的功能来降低功耗。STM32F4 提供几种低功耗状态，每个状态在节能和唤醒延迟之间提供不同的平衡。这些模式包括睡眠、停止和待机模式。

我们可以通过在从中断服务例程（ISR）返回时执行**Cortex®-M4 with FPU 系统控制寄存器**中的`SLEEPONEXIT`位来将我们的系统置于低功耗模式。

让我们深入了解每种低功耗模式的细节，从睡眠模式开始。

#### 睡眠模式

睡眠模式是最基本的低功耗模式，在这种模式下，CPU 时钟停止，但外设继续运行。这种模式提供了**快速唤醒时间**，使其非常适合需要频繁在活动状态和低功耗状态之间切换的应用。

要进入睡眠模式，我们需要清除**系统控制寄存器**（**SCR**）中的`SLEEPDEEP`位，然后执行 WFI 或 WFE 指令，如下面的代码片段所示：

```cpp
void enter_sleep_mode(void) {
    // Clear the SLEEPDEEP bit to enter Sleep mode
    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    // Request Wait For Interrupt
    __WFI();
}
```

微控制器在**任何中断或事件发生时退出睡眠模式**。由于外设保持活跃，任何配置的外设中断都可以唤醒 CPU。

一个示例用例是**传感器监控**。

对于连续传感器监控等应用，睡眠模式提供了一种在不过度牺牲响应性的情况下有效降低功耗的方法。微控制器可以快速唤醒以处理传感器数据，然后返回睡眠模式。

下一个模式是停止模式。

#### 停止模式

停止模式通过停止主内部稳压器和系统时钟提供了比睡眠模式**更深层次的节能状态**。只有低速时钟（LSI 或 LSE）保持活跃。这种模式提供了**适中的唤醒时间**和显著的节能效果。

要进入停止模式，在`PWR_CR`中设置`SLEEPDEEP`位，然后执行 WFI 或 WFE 指令，如下面的代码片段所示。还可以应用其他配置以进一步降低停止模式下的功耗：

```cpp
void enter_stop_mode(void) {
    // Set SLEEPDEEP bit to enable deep sleep mode
    SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
    // Request Wait For Interrupt
    __WFI();
}
```

当发生任何**外部中断**或**配置的 EXTI 线上的唤醒事件**、**RTC 警报**或其他配置的唤醒源时，MCU 将退出停止模式。从停止模式唤醒的时间比从睡眠模式长，但它仍然允许相对快速地返回到完全操作状态。

一个示例用例是**周期性** **数据记录**。

在数据记录等应用中，微控制器可以保持在停止模式，并根据 RTC 警报定期唤醒以记录数据，然后返回停止模式。这显著降低了功耗，同时确保了定期数据记录。

最后一种模式是等待模式。

#### 等待模式

等待模式通过关闭大部分内部电路，包括主稳压器，提供了**最高的节能效果**。只有微控制器的一小部分保持供电以监控唤醒源。这种模式具有**最长的唤醒时间**，但提供了最低的功耗。

要进入等待模式，在电源控制（`PWR_CR`）寄存器中设置`PDDS`和`SLEEPDEEP`位，然后配置唤醒源。以下代码片段演示了如何进入等待模式：

```cpp
void enter_standby_mode(void) {
    // Clear Wakeup flag
    PWR->CR |= PWR_CR_CWUF;
    // Set the PDDS bit to enter Standby mode
    PWR->CR |= PWR_CR_PDDS;
    // Set the SLEEPDEEP bit to enable deep sleep mode
    SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
    // Request Wait For Interrupt
    __WFI();
}
```

当微控制器从等待模式唤醒时，它**执行完整的复位序列**，并从复位向量开始执行。

一个示例用例是**远程** **物联网设备**。

等待模式非常适合需要长时间在电池供电下运行远程物联网设备。这些设备大部分时间可以保持在等待模式，仅在关键事件或计划任务时唤醒，从而最大化电池寿命。

现在我们已经了解了如何进入各种低功耗模式，我们将探讨如何从这些模式中唤醒。

# STM32F4 低功耗模式中的唤醒源和触发器

虽然低功耗模式有助于节省能源，但确保微控制器在需要时能够迅速唤醒同样重要。STM32F4 微控制器系列提供了各种唤醒源和触发器来有效处理这种情况。在本节中，我们将探讨这些唤醒源、它们的工作方式以及它们的实际应用。

## 理解唤醒源

唤醒源是使微控制器从低功耗状态唤醒的机制。STM32F4 提供了多种唤醒源，每种都适用于不同的场景。这些包括外部中断、RTC 闹钟、看门狗定时器和各种内部事件。通过理解这些触发器，我们可以设计出在功耗效率和响应性之间取得平衡的系统。

唤醒源可以分为以下几类：

+   外部中断

+   **实时时钟**（**RTC**）闹钟

+   内部事件

让我们深入了解这些唤醒源，以了解它们的工作原理和典型用例。

### 外部中断

外部中断是 STM32F4 微控制器的主要唤醒源之一。这些中断可以由特定 GPIO 引脚上的事件触发。当微控制器处于低功耗模式时，外部信号，如按钮按下或传感器输出，可以唤醒它。

这就是它的工作原理：

+   **GPIO 配置**：配置 GPIO 引脚作为中断源。这涉及到设置引脚模式并在所需的边缘（上升沿、下降沿或两者）上启用中断。

+   **EXTI 配置**：每个 GPIO 引脚都可以映射到一个 EXTI 线路，该线路可以配置为生成中断。

+   **NVIC 配置**：在**嵌套向量中断控制器**（**NVIC**）中启用 EXTI 线路中断，以确保微控制器能够响应外部事件。

例子用例包括**智能门铃系统**和**智能照明**。

想象一个智能门铃系统。微控制器保持低功耗模式以节省电池寿命。当有人按下门铃按钮（连接到 GPIO 引脚）时，会触发一个外部中断，唤醒微控制器以处理事件并向房主发送通知。另一个优秀的例子是智能家居照明系统。

智能家居照明系统需要在节省能源的同时对用户输入做出响应。微控制器保持低功耗模式，直到外部中断（例如，运动传感器检测到运动）唤醒它。唤醒后，微控制器处理事件，打开灯光，然后在预定义的不活动期间后再次进入睡眠状态。

我们接下来要检查的下一个唤醒源是 RTC 闹钟。

### RTC 闹钟

RTC 是一个多功能的外围设备，可以在特定间隔或预定义的时间生成唤醒事件。它特别适用于需要定期唤醒的应用，如数据记录或计划任务。

这就是它的工作原理：

+   **RTC 配置**：配置 RTC 以生成报警或周期性唤醒事件。这包括设置 RTC 时钟源、启用唤醒定时器以及设置报警时间。

+   **中断处理**：在 NVIC 中启用 RTC 报警或唤醒中断，以确保微控制器在报警或定时器事件发生时唤醒。

*一个示例用例*是**环境****监测系统**。

考虑一个远程环境监测系统，该系统记录温度和湿度数据。微控制器可以被置于低功耗模式，使用 RTC 报警定期唤醒（例如，每小时一次），以读取传感器并记录数据，然后返回到低功耗状态。

我们将要检查的最后唤醒源是**内部事件**。

### 内部事件

除了外部触发器之外，内部事件也可以从低功耗模式唤醒微控制器。这些事件包括以下内容：

+   **外围事件**：由内部外围设备生成的事件，例如 ADC 转换或通信接口活动

+   **系统事件**：如电源电压检测或时钟稳定性问题等内部系统事件

这就是它的工作原理：

+   **外围配置**：配置外围设备，使其在特定事件发生时生成中断。例如，ADC 可以在转换完成后生成中断。

+   **事件处理**：在 NVIC 中启用相关中断以处理这些内部事件并唤醒微控制器。

*一个示例用例*是**健身追踪器**

一个监测心率的可穿戴健身追踪器可以使用 ADC 读取传感器数据。微控制器处于低功耗模式，当 ADC 完成转换时唤醒，从而允许其处理和存储心率数据。

在我们总结本节之前，让我们总结一些在配置唤醒源时需要记住的关键实际考虑因素。

## 实际考虑因素

在配置唤醒源时，您必须考虑以下实际方面：

+   **响应时间**：确保所选唤醒源可以提供您应用所需的最快响应时间。外部中断通常提供最快的唤醒时间。

+   **功耗**：平衡功耗和唤醒需求。RTC 报警和看门狗定时器可以配置为周期性唤醒，同时功耗最小化。

+   **可靠性**：为关键应用选择可靠的唤醒源。看门狗定时器对于确保微控制器可以从故障中恢复的安全关键系统至关重要。

+   **外围配置**：确保所需的唤醒外围设备已正确配置，并且即使在低功耗状态下，它们的时钟也保持启用。

理解并正确配置这些唤醒源确保您的嵌入式系统既节能又可靠。

在下一节中，我们将学习如何开发一个驱动程序以进入待机模式，并随后唤醒系统。

# 开发进入待机模式和唤醒的驱动程序

在您的 IDE 中创建您之前项目的副本，按照前面章节中概述的步骤进行。将此复制的项目重命名为`StandByModeWithWakeupPin`。接下来，在`Src`文件夹中创建一个名为`standby_mode.c`的新文件，然后在`Inc`文件夹中创建一个名为`standby_mode.h`的新文件。

在您的`standby_mode.c`文件中填充以下代码：

```cpp
#include "standby_mode.h"
#define PWR_MODE_STANDBY        (PWR_CR_PDDS)
#define WK_PIN                (1U<<0)
static void set_power_mode(uint32_t pwr_mode);
void wakeup_pin_init(void)
{
    //Enable clock for GPIOA
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    //Set PA0 as input pin
    GPIOA->MODER &= ~(1U<<0);
    GPIOA->MODER &= ~(1U<<1);
    //No pull
    GPIOA->PUPDR &= ~(1U<<0);
    GPIOA->PUPDR &= ~(1U<<1);
}
```

此函数负责配置`PA0`作为退出低功耗模式的唤醒引脚。它通过清除`GPIOA`模式寄存器中的相应位将`PA0`设置为输入引脚，然后配置该引脚不带上拉或下拉电阻：

```cpp
void standby_wakeup_pin_setup(void)
{
    /*Wait for wakeup pin to be released*/
    while(get_wakeup_pin_state() == 0){}
    /*Disable wakeup pin*/
    PWR->CSR &=~(1U<<8);
    /*Clear all wakeup flags*/
    PWR->CR |=(1U<<2);
    /*Enable wakeup pin*/
    PWR->CSR |=(1U<<8);
    /*Enter StandBy mode*/
    set_power_mode(PWR_MODE_STANDBY);
    /*Set SLEEPDEEP bit in the CortexM System Control Register*/
    SCB->SCR |=(1U<<2);
    /*Wait for interrupt*/
    __WFI();
}
```

此函数使微控制器准备进入待机模式，并确保它可以通过配置的唤醒引脚唤醒。它首先等待唤醒引脚释放，确保在继续之前引脚处于稳定状态。然后该函数禁用唤醒引脚以清除任何残留设置，随后清除所有唤醒标志以重置唤醒状态。重新启用唤醒引脚后，该函数通过配置适当的电源控制寄存器将电源模式设置为待机。最后，该函数执行 WFI 指令，将微控制器置于待机模式，直到发生中断，触发唤醒过程：

```cpp
uint32_t get_wakeup_pin_state(void)
{
      return ((GPIOA->IDR & WK_PIN) == WK_PIN);
}
```

此函数检查唤醒引脚（`PA0`）的当前状态。它读取`GPIOA`的输入数据寄存器（`IDR`）并执行与唤醒引脚位掩码（`WK_PIN`）的位与操作。此操作隔离了`PA0`的状态。然后该函数将此结果与位掩码本身进行比较，以确定引脚是否为高。如果`PA0`为高，则函数返回`true`；否则，返回`false`：

```cpp
static void set_power_mode(uint32_t pwr_mode)
{
  MODIFY_REG(PWR->CR, (PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_FPDS | PWR_
  CR_LPLVDS | PWR_CR_MRLVDS), pwr_mode);
}
```

此函数通过修改`PWR_CR`中的特定位来配置 STM32F4 微控制器的电源模式。此函数接受一个参数`pwr_mode`，它指定所需的电源模式设置。它使用`MODIFY_REG`宏来更新`PWR_CR`寄存器，具体针对与不同电源模式相关的位，如**PDDS**（**深度睡眠掉电**）、**LPDS**（**低功耗深度睡眠**）、**FPDS**（**停止模式中闪存掉电**）、**LPLVDS**（**深度睡眠中低电压下的低功耗稳压器**）和**MRLVDS**（**深度睡眠中低电压下的主稳压器**）。

接下来，我们将填充`standby_mode.h`文件：

```cpp
#ifndef STANDBY_MODE_H__
#define STANDBY_MODE_H__
#include <stdint.h>
#include "stm32f4xx.h"
uint32_t get_wakeup_pin_state(void);
void wakeup_pin_init(void);
void standby_wakeup_pin_setup(void);
main.c. Update your main.c file, as shown here:

```

#include <stdio.h>

#include <string.h>

#include "standby_mode.h"

#include "gpio_exti.h"

#include "uart.h"

uint8_t g_btn_press;

static void check_reset_source(void);

int main(void)

{

uart_init();

wakeup_pin_init();

/*查找复位源*/

check_reset_source();

/*初始化 EXTI*/

pc13_exti_init();

while(1)

{

}

}

```cpp

			The `main` function starts by initializing the UART for serial communication with `uart_init`, ensuring that we can send debugging information to the serial port. Next, it configures the wake-up pin by calling `wakeup_pin_init`, preparing the microcontroller to respond to external wake-up signals. The `check_reset_source` function is then called to determine the cause of the microcontroller’s reset, whether from standby mode or another source, and to handle any necessary flag clearing. Following this, the `PC13` is initialized with `pc13_exti_init`. The function then enters an infinite `while(1)`, maintaining the program’s operational state and waiting for interrupts or events to occur:

```

static void check_reset_source(void)

{

/*启用对 PWR 的时钟访问*/

RCC->APB1ENR |= RCC_APB1ENR_PWREN;

if ((PWR->CSR & PWR_CSR_SBF) == (PWR_CSR_SBF))

{

/*清除待机标志*/

PWR->CR |= PWR_CR_CSBF;

printf("系统从待机恢复.....\n\r");

/*等待唤醒引脚释放*/

while(get_wakeup_pin_state() == 0){}

}

/*检查并清除唤醒标志位*/

if((PWR->CSR & PWR_CSR_WUF) == PWR_CSR_WUF )

{

PWR->CR |= PWR_CR_CWUF;

}

}

```cpp

			This function determines the cause of the microcontroller’s reset and handles the necessary flags accordingly. It begins by enabling the clock for the power control (`PWR`) peripheral to ensure access to the power control and status registers. It then checks whether the `PWR_CSR` register, which indicates that the system has resumed from standby mode. If the flag is set, it clears the SBF and prints a message, indicating that the system has resumed from standby. The function also waits for the wake-up pin to be released, ensuring that the pin is in a stable state. Additionally, it checks whether the Wakeup flag (`WUF`) is set and, if so, clears the flag to reset the wake-up status.
			This is the interrupt callback function:

```

static void exti_callback(void)

{

standby_wakeup_pin_setup();

}

```cpp

			And finally, we have the interrupt handler:

```

void EXTI15_10_IRQHandler(void) {

if((EXTI->PR & LINE13) != 0)

{

/*清除 PR 标志位*/

EXTI->PR |= LINE13;

//执行某些操作...

exti_callback();

}

}

```cpp

			The `exti_callback` function, coupled with `EXTI15_10_IRQHandler`, ensures that the microcontroller properly handles external interrupts from the wake-up pin. The `exti_callback` function is a straightforward handler that calls `standby_wakeup_pin_setup`. The `EXTI15_10_IRQHandler` function is an interrupt service routine specifically for EXTI lines 15 to 10\. It checks whether the interrupt was triggered by line 13 (associated with the wake-up pin), and if so, it clears the interrupt pending flag to acknowledge the interrupt. After clearing the flag, it calls `exti_callback` to handle the wake-up event.
			Now, let’s test the project!
			To test the project, start by pressing the blue push button to enter standby mode. Remember that `PA0` is configured as the wake-up pin and is active low. In normal mode, connect a jumper wire from `PA0` to the ground. To trigger a wake-up event, pull out the jumper wire and connect it to 3.3V, causing a change in logic that will wake the microcontroller from standby mode.
			To test on the microcontroller, simply build the project and run it.
			Open RealTerm and configure the appropriate port and baud rate to view the printed message that confirms the system has resumed from standby mode.
			Summary
			In this chapter, we delved into the critical aspects of power management and energy efficiency in embedded systems. Efficient power management is essential for prolonging battery life and ensuring optimal performance in embedded devices. We began by exploring various power management techniques, laying the foundation for understanding how to reduce power consumption in embedded systems.
			We then examined the different low-power modes available in STM32F4 microcontrollers, providing detailed insights into their configurations and applications. Then, we discussed the wake-up sources and triggers in STM32F4, which are essential to ensure that a microcontroller can promptly come out of low-power modes.
			Finally, we put theory into practice by developing a driver to enter standby mode and wake up the microcontroller.
			With this journey into bare-metal embedded C programming now complete, it’s important to acknowledge the profound expertise you’ve gained. By mastering the nuances of microcontroller architecture and the discipline of register-level programming, you’ve equipped yourself with the tools to create efficient and reliable embedded systems from the ground up. This book was designed to offer more than just technical instruction; it also aimed to instill a deeper understanding of the hardware and a methodical approach to firmware development. As you move forward, remember that true mastery in this field lies in the continuous application and refinement of these principles.

```
