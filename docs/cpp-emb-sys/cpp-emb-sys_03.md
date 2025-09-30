

# 第二章：资源受限嵌入式系统中的挑战

如果你正在阅读这本书，那么你很可能对嵌入式系统有很好的了解。嵌入式系统有许多定义，虽然以下定义可能不是最常见的，但它捕捉到了其他定义所共有的本质。**嵌入式系统**是为特定用途而设计的专用计算系统，具有有限的责任范围，与通用计算系统形成对比。嵌入式系统可以嵌入到更大的电子或机械系统中，或者作为独立设备运行。

嵌入式系统和通用计算设备之间的界限有时是模糊的。我们都可以同意，控制烤面包机或飞机泵的系统是嵌入式系统。手机和早期的智能手机也被认为是嵌入式系统。如今，智能手机更接近通用计算设备的定义。在本书中，我们将专注于使用现代 C++在小型嵌入式系统或资源受限的嵌入式系统上进行固件开发。

资源受限的嵌入式系统通常用于安全关键的应用。它们有责任及时控制一个过程，并且不能失败，因为失败可能意味着人类生命的丧失。在本章中，我们将讨论对安全关键设备软件开发施加的限制以及 C++使用的含义。我们将学习如何减轻这些担忧。

在本章中，我们将讨论以下主要主题：

+   安全关键和硬实时嵌入式系统

+   动态内存管理

+   禁用不想要的 C++特性

# 技术要求

为了充分利用本章内容，我强烈建议你在阅读示例时使用编译器探索器([`godbolt.org/`](https://godbolt.org/))。选择 GCC 作为你的编译器，并针对 x86 架构。这将允许你看到标准输出（stdio）结果，更好地观察代码的行为。由于我们使用了大量的现代 C++特性，请确保选择 C++23 标准，通过在编译器选项框中添加`-std=c++23`。

本章的示例可在 GitHub 上找到([`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter02`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter02))。

# 安全关键和硬实时嵌入式系统

**安全关键嵌入式系统**是那些故障可能导致财产或环境损坏、人员受伤甚至生命丧失的系统。这些系统的故障是不可接受的。汽车中的刹车、转向系统和安全气囊是安全关键系统的良好例子。这些系统的正确运行对于车辆的安全运行至关重要。

接下来，我们将分析汽车安全气囊控制单元的实时要求。

## 安全气囊控制单元和实时要求

安全关键型嵌入式系统通常强制执行严格的实时性要求，这意味着任何错过截止时间都会导致系统故障。**气囊控制单元**（**ACU**）从加速度计和压力传感器收集数据，运行一个处理收集到的数据的算法，并检测侧面、正面和追尾碰撞。在检测到碰撞后，ACU 控制不同约束系统的部署，包括气囊和座椅安全带张紧器。

ACU 的实现必须能够应对不同的场景，例如传感器和电子设备故障。这些问题可以通过冗余传感器、比较传感器数据、比较数据与阈值以及自检来缓解。最重要的是，ACUs 需要满足时间要求，因为它们只有几毫秒的时间来收集数据、做出决策并启动约束系统的部署。

如果 ACU 未能及时检测到碰撞，它就会失效，但即使它部署约束系统稍晚一点，也会造成比 ACU 根本未启动部署更大的伤害给驾驶员和乘客。这就是为什么 ACU 必须满足严格的实时性要求，而在固件方面，这意味着所有最坏情况的执行时间都必须可预测。

延迟气囊部署的影响是许多关于乘客受伤的研究的主题。以下摘录是来自论文《关于气囊部署时间对正面车辆碰撞中乘客受伤水平影响的研究》的结论部分，该论文发表在 MATEC Web of Conferences 184(1):01007 上，作者为 Alexandru Ionut Radu、Corneliu Cofaru、Bogdan Tolea 和 Dragoș Sorin Dima，概述了延迟气囊部署的模拟结果：

> *“研究发现，在正面碰撞事件中增加气囊部署时间延迟，乘客头部受伤的概率会增加高达 46%。当气囊点燃时，减少乘客头部与仪表盘/方向盘之间的距离会导致气体膨胀力传递到乘客头部，产生额外的加速度，并使乘客向后移动，增加头部与头枕之间的碰撞伤害潜力。因此，在气囊部署延迟为 0 毫秒时观察到 8%的受伤概率增加，而 100 毫秒的延迟导致头部加速度值增加 54%。因此，气囊的作用被逆转，它不再具有缓冲碰撞的作用，而是产生伤害。”*

下图（来源：[`www.researchgate.net/publication/326715516_Study_regarding_the_influence_of_airbag_deployment_time_on_the_occupant_injury_level_during_a_frontal_vehicle_collision`](https://www.researchgate.net/publication/326715516_Study_regarding_the_influence_of_airbag_deployment_time_on_the_occupant_injury_level_during_a_frontal_vehicle_collision)）展示了碰撞和延迟安全气囊部署的图形说明：

![图 2.1 – 延迟部署约束系统的碰撞模拟](img/B22402_2_1.png)

图 2.1 – 延迟部署约束系统的碰撞模拟

*图 2.1* 有效说明了如果 ACU 无法满足硬实时要求并产生延迟结果的情况。该图取自论文《关于正面车辆碰撞中安全气囊部署时间对乘员伤害水平影响的研究》。

有多个原因可能导致 ACU 失败并导致无延迟或延迟部署：

+   传感器故障

+   电子设备故障

+   碰撞检测算法失败

+   固件未能按时完成任务

通过冗余、数据完整性检查、交叉比较、启动和运行时自检来减轻传感器和电子设备故障。这给固件及其正确运行增加了额外的压力。碰撞检测算法可能由于基于不良模型构建或超出固件职责的其他因素而失败。固件的职责是按时向算法提供传感器的数据，在设定的时间窗口内及时执行它，并根据算法的输出采取行动。

## 测量固件性能和非确定性

我们如何确保固件将在规定的实时要求内运行所有功能？我们进行测量。我们可以测量不同的指标，例如性能分析、对外部事件的响应和 A-B 定时。性能分析将告诉我们程序在哪些函数上花费了最多时间。对外部事件的响应将表明系统对外部事件（如中断或通信总线上的消息）做出响应所需的时间。

### A-B 定时和实时执行

处理实时要求时最重要的指标是 **A-B 定时**。我们测量固件从 A 点到 B 点执行程序所需的时间。A-B 定时可以测量函数的持续时间，但不一定。我们可以用它来测量不同的事情。从 A 点到 B 点可能需要不同的时间，这取决于系统的状态和输入。

进行 A-B 测量的简单方法是通过切换**通用输入输出**（**GPIO**）并使用示波器测量 GPIO 变化之间的时间。这是一个简单且效果良好的解决方案，但它不具可扩展性，因为我们可能需要一个 GPIO 来测量每个我们想要测量的函数，或者我们可能需要一次只测量一个函数。我们还可以使用**微控制器单元（MCU**）的内部定时器进行精确测量，并通过 UART 端口输出该信息。这需要我们仅为了测量目的而利用通用定时器。大多数微控制器都有专门的仪器和配置文件单元。

一些基于 ARM 的微控制器具有**数据观察点和跟踪**（**DWT**）单元。DWT 用于数据跟踪和系统配置文件分析，包括以下内容：

+   **程序计数器**（**PC**）采样

+   循环计数

DWT 生成事件并通过**仪器跟踪宏单元**（**ITM**）单元输出它们。ITM 单元还可以用于输出固件本身生成的数据，以`printf`风格。ITM 缓冲数据并将其发送到 ITM 接收器。**单线输出**（**SWO**）可以用作 ITM 接收器。

我们可以这样利用 DWT 和 ITM 进行配置文件分析：

1.  DWT 可以生成 PC 的周期性采样，并使用 ITM 将它们通过 SWO 发送。

1.  在主机上，我们捕获和分析接收到的数据。

1.  通过为我们固件使用链接器映射文件，我们可以生成程序中每个函数花费时间的分布。

这可以帮助我们看到哪个函数花费了最多时间。对于 A-B 时间测量来说，它并不特别有用，但它允许我们在不直接设置 DWT 和 ITM 单元的情况下，看到程序花费最多时间的地方。

### 使用 GCC 进行软件测试

**GNU 编译器集合**（**GCC**）通过使用`-finstrument-functions`标志来支持软件测试，该标志用于测试函数的入口和退出。这会在每个函数中插入具有以下签名的`entry`和`exit`调用：

```cpp
__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
}
__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
} 
```

我们可以利用 DWT 和 ITM 在`__cyg_profile_func_enter`和`__cyg_profile_func_exit`函数中发送时钟周期数，并在主机上分析它以进行 A-B 时间测量。以下是一个简化的`entry`和`exit`函数实现的示例：

```cpp
extern "C" {
__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
    printf("entry, %p, %d", this_fn, DWT_CYCCNT);
}
__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
    printf("exit, %p, %d", this_fn, DWT_CYCCNT);
}
} 
```

上述实现使用`extern` `"C"`作为`entry`和`exit`测试函数的链接语言指定符，因为它们由编译器与 C 库链接。示例还假设`printf`被重定向以使用 ITM 作为输出，并且 DWT 中的周期计数器寄存器已启动。

另一个选项是使用 ITM 的时间戳功能，并从 `entry` 和 `exit` 仪器函数发送时间戳和函数地址。借助链接器映射文件，我们随后可以重建函数调用的顺序和返回。存在用于发送跟踪的专用格式，例如**通用跟踪格式**（**CTF**），以及称为**跟踪查看器**的桌面工具，这些工具可以让我们简化软件仪器化。CTF 是一种开放格式，用于在数据包中序列化一个事件，该数据包包含一个或多个字段。专用工具，如 **barectf** ([`barectf.org/docs/barectf/3.1/index.html`](https://barectf.org/docs/barectf/3.1/index.html)) 用于简化 CTF 数据包的生成。

事件使用 **YAML Ain’t Markup Language**（**YAML**）配置文件进行描述。`barectf` 使用配置文件生成一个包含跟踪函数的简单 C 库。这些函数用于源代码中我们想要发出跟踪的地方。

CTF 跟踪可以通过不同的传输层发送，例如 ITM 或串行。可以使用工具如 Babeltrace ([`babeltrace.org`](https://babeltrace.org)) 和 TraceCompass ([`eclipse.dev/tracecompass`](https://eclipse.dev/tracecompass)) 分析跟踪。还有其他工具可以简化跟踪生成、传输和查看，例如 SEGGER SystemView。在目标侧，SEGGER 提供了一个小型的软件模块，用于调用跟踪函数。跟踪通过 SEGGER 的 **实时传输**（**RTT**）协议使用 SWD 发送，并在 SystemView 中进行分析。

我们介绍了 A-B 定时的基本方法。还有更多高级技术，它们通常取决于目标能力，因为有一些更高级的跟踪单元可以用于 A-B 测量。

### 固件中的确定性与非确定性

如果我们使用 A-B 定时方法来测量函数的持续时间，并且对于相同的输入具有相同的持续时间和函数输出，我们称该函数是**确定性的**。如果一个函数依赖于全局状态，并且对于相同的输入测量的持续时间不同，我们称它是**非确定性的**。

C++ 中的默认动态内存分配器往往是非确定性的。分配的持续时间取决于分配器的当前全局状态和分配算法的复杂性。我们可以使用不同的全局状态对相同的输入进行持续时间测量，但很难评估所有可能的全局状态，并保证使用默认分配器的**最坏情况执行时间**（**WCET**）。

动态内存分配的非确定性行为是安全性关键系统的一个问题。另一个问题是它可能会失败。如果没有更多的可用内存或内存碎片化，分配可能会失败。这就是为什么许多安全编码标准，如**汽车行业软件可靠性协会**（**MISRA**）和**汽车开放系统架构**（**AUTOSAR**），都反对使用动态内存。

我们将探讨动态内存管理的影响和安全性关键问题。

# 动态内存管理

C++标准为对象定义了以下存储持续时间：

+   **自动存储持续时间**：具有自动存储持续时间的对象在程序进入和退出定义它们的代码块时自动创建和销毁。这些通常是函数内的局部变量，除了声明为`static`、`extern`或`thread_local`的变量。

+   **静态存储持续时间**：具有静态存储持续时间的对象在程序开始时分配，在程序结束时释放。所有在命名空间作用域内声明的对象（包括全局命名空间）都具有这种静态持续时间，以及使用`static`或`extern`声明的对象。

+   **线程存储持续时间**：在 C++11 中引入，具有线程存储持续时间的对象与定义它们的线程一起创建和销毁，允许每个线程都有自己的变量实例。它们使用`thread_local`说明符声明。

+   **动态存储持续时间**：具有动态存储持续时间的对象使用动态内存分配函数（在 C++中为`new`和`delete`）显式创建和销毁，使软件开发者能够控制这些对象的生命周期。

动态存储为软件开发者提供了极大的灵活性，使得他们能够完全控制一个对象的生命周期。权力越大，责任越大。对象使用`new`运算符动态分配，并使用`delete`释放。每个动态分配的对象必须恰好释放一次，并且在释放后不应再被访问。这是一条简单的规则，但未能遵循它会导致一系列问题，例如以下所述：

+   当动态分配的内存未被正确释放时，会发生内存泄漏。随着时间的推移，这种未使用的内存会积累，可能耗尽系统资源。

+   悬挂指针发生在指针仍然引用一个已经被释放的内存位置时。访问这样的指针会导致未定义的行为。

+   当已经释放的内存再次被删除时，会发生双重释放错误，导致未定义的行为。

动态内存管理中的另一个问题是内存碎片化。

## 内存碎片化

**内存碎片化**发生在随着时间的推移，空闲内存被分成小块的非连续块时，即使总共有足够的空闲内存，也难以或无法分配大块内存。主要有两种类型：

+   **外部碎片化**：当总内存足够满足分配请求，但由于碎片化而没有足够大的单个连续块时，就会发生这种情况。这在内存分配和释放频繁且大小差异显著的系统中最常见。

+   **内部碎片化**：当分配的内存块大于请求的内存时，会导致分配块内的空间浪费。这在使用具有固定大小内存块或内存池的分配器以及旨在提供 WCET 的分配器时发生。

内存碎片化导致内存使用效率低下，降低性能或防止进一步分配，即使在看起来有足够内存可用的情况下，也会导致内存不足的情况。让我们在以下图中可视化动态内存分配保留的内存区域：

![图 2.2 – 用于动态分配的内存区域](img/B22402_2_2.png)

图 2.2 – 用于动态分配的内存区域

在*图 2.2*中，每个块代表在分配过程中分配的内存单元。未分配的区域或使用`delete`运算符释放的区域是空的。尽管有足够的内存可用，但如果请求分配四个内存单元，由于内存碎片化而没有四个连续的内存块可用，分配将失败。

默认内存分配器的非确定性行为和内存不足的情况是关键安全系统的主要关注点。MISRA 和 AUTOSAR 为在关键安全系统中使用 C++提供了编码指南。

MISRA 是一个为汽车行业使用的电子组件开发的软件提供指南的组织。它是汽车制造商、组件供应商和工程咨询公司之间的合作。MISRA 产生标准也用于航空航天、国防、太空、医疗和其他行业。

AUTOSAR 是由汽车制造商、供应商以及来自电子、半导体和软件行业的其他公司组成的全球发展伙伴关系。AUTOSAR 还制定了关于在关键和安全相关系统中使用 C++的指南。

## C++中动态内存管理的安全关键指南

MISRA C++ 2008，它涵盖了 C++03 标准，禁止使用动态内存分配，而 AUTOSAR 的*关于在关键和安全相关系统中使用 C++14 语言的指南*规定了以下规则：

+   规则 A18-5-5（必需，工具链，部分自动化）

> “内存管理函数应确保以下内容：（a）存在最坏情况执行时间的结果，具有确定性行为，（b）避免内存碎片化，（c）避免内存耗尽，（d）避免不匹配的分配或释放，（e）不依赖于内核的非确定性行为调用。”

+   规则 A18-5-6（必需，验证/工具链，非自动化）

> “应进行一项分析，以分析动态内存管理的故障模式。特别是，应分析以下故障模式：(a)由于不存在最坏情况执行时间而产生的非确定性行为，(b)内存碎片化，(c)内存耗尽，(d)分配和释放不匹配，(e)依赖于对内核的非确定性调用。”

现在严格遵循这两条规则是一项极其困难的任务。我们可以编写一个具有确定 WCET（最坏情况执行时间）并最小化碎片化的自定义分配器，但如何编写一个避免内存耗尽的分配器？或者，如果发生这种情况，我们如何确保系统的非故障？每次调用分配器都需要验证操作的成功，并在失败的情况下，以某种方式减轻其影响。或者我们需要能够准确估计分配器所需的内存量，以确保在任何情况下都不会在运行时耗尽内存。这给我们的软件设计增加了全新的复杂性，并且比通过允许动态内存分配增加的复杂性还要多。

动态内存分配策略的一种折中方法是允许在启动时使用，但在系统运行时不允许。这是**联合攻击战斗机空中车辆 C++编码标准**所使用的策略。MISRA C++ 2023 也建议在系统运行时不要使用动态内存分配，并作为缓解策略，建议在启动时使用。

C++标准库大量使用动态内存分配。异常处理机制实现也经常使用动态分配。在放弃在嵌入式项目中使用标准库的想法之前，让我们发现`std::vector`容器的工作原理，并看看 C++提供了什么来缓解我们的担忧。

## C++标准库中的动态内存管理

我们引入了`std::vector`作为标准库中的一个使用动态内存分配的容器。`vector`是一个模板类，我们可以指定底层类型。它连续存储元素，我们可以使用`data`方法直接访问底层的连续存储。

以下代码示例演示了向量的使用：

```cpp
 std::vector<std::uint8_t> vec;
  constexpr std::size_t n_elem = 8;
  for (std::uint8_t i = 0; i < n_elem; i++) {
    vec.push_back(i);
  }
  const auto print_array = [](uint8_t *arr, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
      printf("%d ", arr[i]);
    }
    printf("\r\n");
  };
  print_array(vec.data(), n_elem); 
```

我们创建了一个以`uint8_t`为底层类型的向量，并使用`push_back`方法添加了从`0`到`8`的值。示例还演示了对底层连续存储的指针的访问，我们将它作为参数传递给了`print_array` lambda。

`vector`的通常分配策略是在第一次插入时分配一个元素，然后每次达到其容量时加倍。存储从`0`到`8`的值将导致 4 个分配请求，如下面的图所示：

![图 2.3 – 向量分配请求](img/B22402_2_3.png)

图 2.3 – 向量分配请求

*图 2**.3* 描述了向量的分配请求。为了检查任何平台的向量实现，我们可以重载 `new` 和 `delete` 运算符并监控分配请求：

```cpp
void *operator new(std::size_t count) {
  printf("%s, size = %ld\r\n", __PRETTY_FUNCTION__, count);
  return std::malloc(count);
}
void operator delete(void *ptr) noexcept {
  printf("%s\r\n", __PRETTY_FUNCTION__);
  std::free(ptr);
} 
```

`new` 重载运算符将分配调用传递给 `malloc`，并打印出调用者请求的大小。`delete` 重载运算符仅打印出函数签名，以便我们可以看到它何时被调用。一些使用 GCC 的标准库实现使用 `malloc` 实现了 `new` 运算符。我们的向量分配调用将产生以下输出：

```cpp
void* operator new(std::size_t), size = 1
void* operator new(std::size_t), size = 2
void operator delete(void*)
void* operator new(std::size_t), size = 4
void operator delete(void*)
void* operator new(std::size_t), size = 8
void operator delete(void*) 
```

上述结果使用 GCC 编译器获得，对于 x86_64 和 Arm Cortex-M4 平台都是相同的。当向量填满可用内存时，它将请求分配当前使用内存的两倍数量。然后，它将数据从原始存储复制到新获得的内存中。之后，它删除之前使用的存储，正如我们从前面的生成输出中可以看到的那样。

重载 `new` 和 `delete` 运算符将允许我们全局地改变分配机制，以满足要求确定性的 WTEC 和避免内存不足场景的安全关键指南，这相当具有挑战性。

如果事先知道元素的数量，可以通过使用 `reserve` 方法优化向量的分配请求：

```cpp
 vec.reserve(8); 
```

使用 `reserve` 方法将使向量请求八个元素，并且只有当我们超出八个元素时，它才会请求更多内存。这使得它对于在启动时允许动态分配，并且我们可以保证在任何时刻元素的数量都将保持在预留内存内的项目非常有用。如果我们向向量中添加第九个元素，它将进行另一个分配请求，请求足以容纳 16 个元素的内存。

C++ 标准库还使得容器可以使用局部分配器。让我们看看向量的声明：

```cpp
template<
    class T,
    class Allocator = std::allocator<T>
> class vector; 
```

我们可以看到第二个模板参数是 `Allocator`，默认参数是 `std::allocator`，它使用 `new` 和 `delete` 运算符。C++17 引入了 `std::pmr::polymorphic_allocator`，这是一个根据其构建的 `std::pmr::memory_resource` 类型表现出不同分配行为的分配器。

可以通过提供一个初始的静态分配的缓冲区来构建一个内存资源，它被称为 `std::pmr::monotonic_buffer_resource`。单调缓冲区是为了性能而构建的，并且仅在它被销毁时释放内存。使用静态分配的缓冲区初始化它使其适合嵌入式应用。让我们看看我们如何使用它来创建一个向量：

```cpp
 using namespace std;
  using namespace std::pmr;
  array<uint8_t, sizeof(uint8_t) * 8> buffer{0};
  monotonic_buffer_resource mbr{buffer.data(), buffer.size()};
  polymorphic_allocator<uint8_t> pa{&mbr};
  std::pmr::vector<uint8_t> vec{pa}; 
```

在前面的例子中，我们做了以下操作：

1.  创建一个 `std::array` 容器，其底层类型为 `uint8_t`。

1.  构建一个单调缓冲区，并为其提供我们刚刚创建的数组作为初始缓冲区。

1.  使用单调缓冲区创建一个多态分配器，我们用它来创建一个向量。

请注意，该向量来自`std::pmr`命名空间，它只是`std::vector`的部分特化，如下所示：

```cpp
namespace pmr {
    template< class T >
    using vector = std::vector<T, std::pmr::polymorphic_allocator<T>>;
} 
```

利用单调缓冲区创建的向量将在缓冲区提供的空间中分配内存。让我们通过以下示例来检查此类向量的行为，该示例由之前解释的代码构建而成：

```cpp
#include <cstdio>
#include <cstdlib>
#include <array>
#include <memory_resource>
#include <vector>
#include <new>
void *operator new(std::size_t count, std::align_val_t al) {
  printf("%s, size = %ld\r\n", __PRETTY_FUNCTION__, count);
  return std::malloc(count);
}
int main() {
  using namespace std;
  using namespace std::pmr;
  constexpr size_t n_elem = 8;
  array<uint8_t, sizeof(uint8_t) * 8> buffer{0};
  monotonic_buffer_resource mbr{buffer.data(), buffer.size()};
  polymorphic_allocator<uint8_t> pa{&mbr};
  std::pmr::vector<uint8_t> vec{pa};
  //vec.reserve(n_elem);
for (uint8_t i = 0; i < n_elem; i++) {
    vec.push_back(i);
  }
  for (uint8_t data : buffer) {
    printf("%d ", data);
  }
  printf("\r\n");
  return 0;
} 
```

前面的程序将提供以下输出：

```cpp
void* operator new(std::size_t, std::align_val_t), size = 64
0 0 1 0 1 2 3 0 
```

我们看到，尽管我们使用了单调缓冲区，程序仍然调用了`new`运算符。您会注意到对`reserve`方法的调用被注释掉了。这将导致向量扩展策略，如之前所述。当单调缓冲区的初始内存被使用时，它将退回到上游内存资源指针。默认的上游内存资源将使用`new`和`delete`运算符。

如果我们打印用作`monotonic_buffer_resource`初始存储的缓冲区，我们可以看到向量正在分配第一个元素并将`0`存储到其中，然后将其翻倍并存储`0`和`1`，然后再次翻倍，存储`0`、`1`、`2`和`3`。当它尝试再次翻倍时，单调缓冲区将无法满足分配请求，并将退回到使用默认分配器，该分配器依赖于`new`和`delete`运算符。我们可以在以下图中可视化这一点：

![图 2.4 – 单调缓冲区资源使用的缓冲区状态](img/B22402_2_4.png)

图 2.4 – 单调缓冲区资源使用的缓冲区状态

*图 2.4* 描述了单调缓冲区资源使用的内部状态。我们可以看到，单调缓冲区资源没有以任何方式释放内存。在分配缓冲区请求时，如果缓冲区中有足够的空间来容纳请求的元素数量，它将返回初始缓冲区中最后一个可用元素的指针。

您会注意到，在此示例中使用的`new`运算符的签名与之前使用的不同。实际上，标准库定义了不同版本的`new`和匹配的`delete`运算符，并且没有检查很难确定标准库容器使用的是哪个版本。这使得全局重载它们并替换实现为自定义版本变得更加具有挑战性，使得局部分配器通常是一个更好的选择。

使用在栈上初始化的缓冲区作为单调缓冲区的多态分配器可能是一个减轻在处理标准 C++库中的容器时动态内存管理强加的一些问题的好选项。我们在向量上展示的方法可以用于标准库中的其他容器，例如`list`和`map`，也可以用于库中的其他类型，例如`basic_string`。

缓解动态内存分配的担忧是可能的，但它仍然带来了一些挑战。如果你想要绝对确定你的 C++程序没有调用`new`运算符，有方法可以确保这一点。让我们探索我们如何禁用不想要的 C++功能。

# 禁用不想要的 C++功能

你可能已经注意到，我们使用 C 标准库中的`printf`在标准输出上打印调试信息，而不是使用 C++标准库中的`std::cout`。原因有两个——`std::cout`全局对象的实现有一个很大的内存占用，并且它使用动态内存分配。C++与 C 标准库很好地协同工作，使用`printf`是资源受限系统的良好替代方案。

我们已经讨论了异常处理机制，它通常依赖于动态内存分配。在 C++中禁用异常就像向编译器传递适当的标志一样简单。在 GCC 的情况下，该标志是`–fno-exceptions`。对于**运行时类型信息**（**RTTI**）也是如此。我们可以使用`–fno-rtti`标志来禁用它。

禁用异常将在抛出异常时调用`std::terminate`。我们可以用我们自己的实现替换默认的终止处理程序，并适当地处理它，如下面的例子所示：

```cpp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <array>
int main() {
  constexpr auto my_terminate_handler = []() {
    printf("This is my_terminate_handler\r\n");
    std::abort();
  };
  std::set_terminate(my_terminate_handler);
  std::array<int, 4> arr;
  for (int i = 0; i < 5; i++) {
   arr.at(i) = i;
  }
  return 0;
} 
```

之前的例子展示了如何通过我们自己的实现使用`std::set_terminate`来设置终止处理程序。这允许我们处理在运行时不应发生的情况，并尝试从中恢复或优雅地终止它们。C++中的一些功能或行为不能通过编译器标志禁用，但还有其他方法来处理它们，

正如我们之前看到的，我们可以重新定义全局`new`和`delete`运算符。我们还可以删除它们，如果在调用`new`的软件组件中使用，这将导致编译失败，从而有效地防止任何需要的动态内存分配尝试：

```cpp
#include <cstdio>
#include <vector>
#include <new>
void *operator new(std::size_t count) = delete;
void *operator new[](std::size_t count) = delete;
void *operator new(std::size_t count, std::align_val_t al) = delete;
void *operator new[](std::size_t count, std::align_val_t al) = delete;
void *operator new(std::size_t count, const std::nothrow_t &tag) = delete;
void *operator new[](std::size_t count, const std::nothrow_t &tag) = delete;
void *operator new(std::size_t count, std::align_val_t al, const std::nothrow_t &) = delete;
void *operator new[](std::size_t count, std::align_val_t al,const std::nothrow_t &) = delete;
int main() {
  std::vector<int> vec;
  vec.push_back(123);
  printf("vec[0] = %d\r\n", vec[0]);
  return 0;
} 
```

之前的例子将因以下编译器消息（以及其他消息）而失败：

```cpp
/usr/include/c++/13/bits/new_allocator.h:143:59: error: use of deleted function 'void* operator new(std::size_t, std::align_val_t)'
  143 |             return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW (__n * sizeof(_Tp), 
```

通过删除`new`运算符，我们可以使尝试使用动态内存管理的 C++程序的编译失败。如果我们想要确保我们的程序没有使用动态内存管理，这很有用。

# 摘要

C++允许极大的灵活性。资源受限的嵌入式系统和安全关键性指南可以对某些 C++功能的用法施加一些限制，例如异常处理、RTTI 以及标准 C++库中容器和其他模块使用动态内存分配。C++承认这些担忧，并提供机制来禁用不想要的特性。在本章中，我们学习了通过本地分配器和重载全局`new`和`delete`运算符来缓解动态内存分配担忧的不同策略。

学习曲线陡峭，但值得付出努力，因此让我们继续我们的旅程，探索嵌入式系统中的 C++。

在下一章中，我们将探讨嵌入式开发中的 C++ 生态系统。
