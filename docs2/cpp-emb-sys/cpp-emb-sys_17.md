# 13

# 与 C 库协同工作

在*第六章*中，我们讨论了 C 和 C++之间的互操作性。我们学习了语言链接以及如何使用它将 C 库包含在 C++项目中。从技术角度来看，这就是我们在 C++中使用 C 所需的一切。

在本章中，我们将重点关注将 C 库集成到 C++项目中以增强代码灵活性的软件开发技术。由于许多 C++项目仍然依赖于供应商提供的 C **硬件抽象层**（**HALs**），我们将集中讨论如何有效地将这些 C 库集成到我们的项目中。

此外，本章还将涵盖**资源获取即初始化**（**RAII**）范式，并解释为什么它在嵌入式系统中特别有益。通过自动管理资源分配和释放，RAII 大大降低了泄漏和其他资源误用问题的风险，这在资源受限的嵌入式环境中尤为重要。

在本章中，我们将涵盖以下主要内容：

+   在 C++项目中使用 C HAL

+   静态类

+   使用 RAII 封装`LittleFs` C 库

# 技术要求

本章的示例可在 GitHub 上找到（[`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter13`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter13)）。为了充分利用本章内容，请在 Renode 模拟器中运行示例。

# 在 C++项目中使用 C HAL

在*第十二章*中，我们探讨了使用 C++进行 HAL 开发的优点。然而，尽管有这些优点，目标供应商提供的 HALs 是以 C 库的形式。这些库在全球数百万台设备上经过了彻底测试，供应商通常维护得很好，提供定期更新。因此，使用它们而不是在 C++中重新实现 HAL 更有意义。

接下来，我们将为 UART 外设创建一个基于接口的设计，这将为我们提供一个更灵活的软件设计，并允许我们将使用 UART 接口的组件与底层细节解耦。

## 用于灵活软件设计的 UART 接口

在*第五章*中，我们讨论了接口对于灵活软件设计的重要性。在那里，我们有一个由`uart_stm32`类实现的`uart`接口类。`gsm_lib`类依赖于`uart`接口，这意味着我们可以与不同的`uart`接口实现重用它。

来自*第五章*的`uart_stm32`类为了演示目的具有简单的实现。它使用 C 标准库中的`printf`和`putc`函数在标准输出上写入消息。现在，我们将通过实际实现`uart_stm32`类，该类已在书中 GitHub 仓库的所有示例中使用，使我们能够在 Renode 模拟器中看到输出。让我们从以下代码所示的`uart`接口`class`开始：

```cpp
#include <cstdint>
#include <span>
namespace hal
{
class uart
{
  public:
    virtual void init(std::uint32_t baudrate) = 0;
    virtual void write(std::span<const char> data) = 0;
};
}; // namespace hal 
```

`uart` 接口是一个简单的类，包含两个虚拟方法：

+   `virtual void init(std::uint32_t baudrate)`: 用于使用单个参数 `baudrate` 初始化 UART 外设的函数。

+   `virtual void write(std::span<const char> data)`: 用于通过 UART 外设发送数据的函数。它有一个 `std::span<const char>` 参数，与通常的 C 方法中使用数据缓冲区指针和长度的方法不同。使用 `std::span` 提高了代码的内存安全性。

接下来，让我们通过 `uart_stm32` 类的定义来了解其实现：

```cpp
#include <span>
#include <cstdint>
#include <uart.hpp>
#include <stm32f0xx_hal.h>
#include <stm32f072xb.h>
namespace hal
{
class uart_stm32 : public uart
{
  public:
    uart_stm32(USART_TypeDef *inst);
    void init(std::uint32_t baudrate = c_baudrate_default);
    void write(std::span<const char> data) override;
  private:
    UART_HandleTypeDef huart_;
    USART_TypeDef *instance_;
    std::uint32_t baudrate_;
    `static` constexpr std::uint32_t c_baudrate_default = 115200;
};
}; // namespace hal 
```

在 `uart_stm32` 类定义中，我们可以注意到以下内容：

+   重写了 `uart` 接口的虚拟方法 `init` 和 `write`。

+   接受 `USART_TypeDef` 指针的构造函数。此类型是一个 `struct`，它描述了 CMSIS 头文件 `stm32f072xb.h` 中 UART 外设寄存器布局。

+   在私有成员中，我们看到 `UART_HandleTypeDef`，这是一个在 `stm32f0xx_hal_uart.h` 文件中定义的类型，由 ST HAL 提供。

接下来，让我们通过这段代码中的 `uart_stm32` 类的构造函数和方法的实现来了解其实现过程：

```cpp
hal::uart_stm32::uart_stm32(USART_TypeDef *inst): instance_(inst)
{
} 
```

在此代码中，我们看到 `uart_stm32` 构造函数的实现。它只是使用初始化列表语法设置私有成员 `USART_TypeDef *instance_`。CMSIS 定义了宏 `USART1`、`USART2`、`USART3` 和 `USART4`，它们指定了这些外设的地址，我们可以使用它们来初始化 `uart_stm32` 对象。

`uart` 接口定义了 `init` 方法，因为 UART 外设初始化依赖于其他硬件初始化（例如时钟配置）。如果我们将在构造函数中实现初始化，那么如果有人定义了一个 `global` 或 `static` 的 `uart_stm32` 对象，我们可能会遇到问题。`init` 方法如下所示：

```cpp
void hal::uart_stm32::init(std::uint32_t baudrate)
{
    huart_.Instance = instance_;
    huart_.Init.BaudRate = baudrate;
    huart_.Init.WordLength = UART_WORDLENGTH_8B;
    huart_.Init.StopBits = UART_STOPBITS_1;
    huart_.Init.Parity = UART_PARITY_NONE;
    huart_.Init.Mode = UART_MODE_TX_RX;
    huart_.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart_.Init.OverSampling = UART_OVERSAMPLING_16;
    huart_.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart_.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    huart_.MspInitCallback = nullptr;
    HAL_UART_Init(&huart_);
} 
```

在 `init` 方法中，我们使用以下配置初始化成员 `UART_HandleTypeDef huart_`：

+   `instance_`: 构造函数中选择的 UART 外设的地址

+   `baudrate`

+   8 位字长

+   1 个停止位

+   同时启用 TX 和 RX 模式

+   无硬件控制

我们还将 `MspInitCallback` 设置为 `nullptr`。我们调用 ST HAL 的 `HAL_UART_Init` 函数，并提供 `huart_` 的指针。请注意，为了示例的简单性，这里没有错误处理。错误处理是一个重要的步骤，HAL 的返回代码应该在代码中适当处理。

接下来，我们将通过以下内容了解 `write` 方法的实现：

```cpp
void hal::uart_stm32::write(std::span<const char> data)
{
    // we must cast away constness due to ST HAL’s API
char * data_ptr = const_cast<char *>(data.data());
    HAL_UART_Transmit(&huart_,
                     reinterpret_cast<uint8_t *(data_ptr),
                     data.size(),
                     HAL_MAX_DELAY);
} 
```

在 `write` 方法中，我们通过传递 `std::span<const char> data` 参数中的数据指针和数据大小，调用 ST HAL 的 `HAL_UART_Transmit` 函数。值得注意的是，我们需要取消 const 属性，因为 C 的 `HAL_UART_Transmit` 函数不接受指向数据的 const 指针。只有在确定我们将指针传递给取消 const 属性的函数不会修改其内容时，这样做才是安全的。

接下来，我们将从软件设计和使用的模式的角度分析这种方法。

## 适配器模式中的 UART 接口

本例中所有软件组件之间的关系（`uart`接口、接口的实现`uart_stm32`和 ST HAL）可以用以下 UML 图表示：

![图 13.1 – `uart_stm32`类图](img/B22402_13_01.png)

图 13.1 – `uart_stm32`类图

在*图 13.1*中，我们看到`uart_stm32`类的 UML 类图。这个类有效地实现了**适配器设计模式**，这是一种结构设计模式，用于允许具有不兼容接口的类协同工作。适配器模式涉及创建一个适配器类，它包装现有的类（或模块），并提供客户端期望的新接口。

在我们的例子中，即使`stm32f0xx_hal_uart`是一个 C 模块而不是 C++类，`uart_stm32`类通过封装基于 C 的 HAL 代码并通过 C++ `uart`接口暴露它来充当适配器。这种适配允许系统中的其他类或客户端，如`GSM`库，使用标准化的 C++接口与 UART 硬件交互，而不必关心底层的 C 实现细节。

让我们从`uart`接口客户端的角度来分析这种方法，例如在`gsm_lib`类中实现的`GSM`库，其定义如下：

```cpp
class gsm_lib{
    public:
        gsm_lib(hal::uart &u) : uart_(u) {}
        // other methods
private:
        hal::uart &uart_;
}; 
```

在这段代码中，我们看到`uart`接口的一个简单客户端示例 – `gsm_lib` – 它有一个构造函数，用于初始化引用`hal::uart &uart_`。这种方法被称为**依赖注入**。`gsm_lib`类的依赖外部构建并通过构造函数作为引用提供给类。根据接口的依赖也允许松耦合，这带来了以下好处：

+   `gsm_lib`对`uart`接口的实现细节不感兴趣。它不需要了解波特率、硬件设置等。

+   `gsm_lib`与特定目标无关。我们可以通过在这些平台上实现`uart`接口来在不同的平台上重用它。

+   对`gsm_lib`的软件测试很容易，因为我们可以在测试中使用模拟的`uart`接口，并用模拟对象实例化`gsm_lib`对象。

在`uart_stm32`类中，我们不是直接使用 C HAL 库，而是可以将 C 库中的函数包装在一个所谓的`static`类中，该类具有所有参数的直接映射。

# 引入静态类

我们在这里将要讨论的`static`类概念在 C++语言标准中并不存在。我们是从像 C#这样的语言中借用的它，在 C#中，它被定义为只包含`static`成员和方法的类。它不能被实例化。在 C#中，使用`static`关键字来声明一个`static`类。

在 C++中，可以通过定义一个具有所有`static`方法和成员的类并删除默认构造函数来创建一个`static`类。删除构造函数确保无法创建类的实例，在编译时强制执行。禁用实例化向用户明确表示：*这是一个* `static` *类。你使用的函数不依赖于任何特定实例的状态，因为没有实例存在。如果存在任何内部状态，它是共享的，并将影响所有使用该类的用户*。

我们将修改之前的示例，创建一个`uart_c_hal` `static`类来封装 UART C HAL 函数，如下所示：

```cpp
`struct` uart_c_hal {
    uart_c_hal() = delete;
    static inline HAL_StatusTypeDef init(UART_HandleTypeDef *huart)
    {
        return HAL_UART_Init(huart);
    }
    static inline HAL_StatusTypeDef transmit(UART_HandleTypeDef *huart,
uint8_t *pData,
uint16_t Size,
uint32_t Timeout)
    {
        return HAL_UART_Transmit(huart, pData, Size, Timeout);
    }
}; 
```

在此代码中，我们简单地将`uart_c_hal`类的`static`方法中的 C 函数进行了映射。接下来，我们将修改`uart_stm32`类以使用`uart_c_hal`，如下所示：

```cpp
template <typename HalUart>
class uart_stm32 : public uart
{
  public:
    uart_stm32(USART_TypeDef *inst) : instance_(inst) {}
    void init(std::uint32_t baudrate = c_baudrate_default) override {
      huart_.Instance = instance_;
      huart_.Init.BaudRate = baudrate;
      // ...
// init huart_ `struct`
      HalUart::init(&huart_);
    }
    void write(std::span<const char> data) override {
      // we must cast away costness due to ST HAL’s API
char * data_ptr = const_cast<char *>(data.data());
      HalUart::transmit(&huart_,
  reinterpret_cast<uint8_t *(data_ptr),
         data.size(),
   HAL_MAX_DELAY);
    }
  private:
    UART_HandleTypeDef huart_;
    USART_TypeDef *instance_;
    std::uint32_t baudrate_;
    static constexpr std::uint32_t c_baudrate_default = 115200;
}; 
```

在此代码中，我们看到`uart_stm32`现在是一个模板类，它使用了模板参数`HalUart`中的`init`和`transmit`方法。现在我们可以使用类模板，如下所示：

```cpp
uart_stm32<uart_c_hal> uart(USART2);
uart.init();
gsm_lib gsm(uart); 
```

`uart_stm32`类模板仍然实现了`uart`接口，这意味着我们仍然可以使用它与`gsm_lib`类一起使用。在`static`类中封装 C HAL 函数并将`uart_stm32`调整为通过模板参数使用它，解耦了 C HAL 与`uart_stm32`实现。这使得可以在目标之外测试`uart_stm32`类，因为它不再依赖于特定平台的代码了。

静态类是在 C++项目中使用 C 库的一种方式。它们允许我们将 C 库中的函数封装在可以通过模板参数传递给 C++类的类型中，使代码更加灵活且易于测试。

接下来，我们将看到如何应用 RAII 技术来有效地封装**little fail-safe** (`littlefs`) 文件系统 C 库。

# 使用 RAII 封装 littlefs C 库

RAII 是一种简单而强大的 C++技术，用于通过对象的生命周期来管理资源。资源可以代表不同的事物。资源在对象的生命周期开始时获取，在对象的生命周期结束时释放。

该技术用于管理如动态分配内存等资源。为确保内存被释放并避免内存泄漏，建议仅在类内部使用动态分配。当对象被实例化时，构造函数将分配内存，当对象超出作用域时，析构函数将释放分配的内存。

RAII 技术可以应用于动态分配内存以外的其他资源，例如文件，我们将将其应用于`littlefs`文件系统库（[`github.com/littlefs-project/littlefs`](https://github.com/littlefs-project/littlefs)）。我们将从对`littlefs`的简要概述开始——这是一个为微控制器设计的文件系统。

## LittleFS – 用于微控制器的文件系统

`littlefs`文件系统是为具有以下特点的微控制器设计的：

+   **断电恢复性**：它被构建来处理意外的断电。在断电的情况下，它将回退到最后已知的好状态。

+   **动态磨损均衡**：它针对闪存进行了优化，提供跨动态块的磨损均衡。它还包括检测和绕过坏块机制，确保长期可靠性能。

+   **有限 RAM/ROM**：它针对低内存使用进行了优化。无论文件系统大小如何，RAM 消耗保持恒定，没有无界递归。动态内存限制在可配置的缓冲区中，可以设置为 `static`。

我们将首先介绍 `littlefs` 的基本用法，然后看看我们如何在 C++ 包装类中应用 RAII。我们将通过以下示例使用 `littlefs`：

+   格式化和挂载文件系统。

+   创建一个文件，向其中写入一些内容，然后关闭它。

+   打开一个文件，从中读取内容，然后关闭它。

完整示例包含在 `Chapter13/lfs_raii/app/src/main.cpp` 中。让我们从以下代码开始，它格式化和挂载文件系统，如下所示：

```cpp
lfs_t lfs;
const lfs_config * lfs_ramfs_cfg = get_ramfs_lfs_config();
lfs_format(&lfs, lfs_ramfs_cfg);
lfs_mount(&lfs, lfs_ramfs_cfg); 
```

此代码执行以下步骤：

+   它声明了一个名为 `lfs` 的文件系统对象，类型为 `lfs_t`。此对象将用于与 `littlefs` 文件系统交互。它包含文件系统的状态，并且对于所有后续的文件系统操作都是必需的。

+   函数 `get_ramfs_lfs_config()` 返回一个指向 `lfs_config` 结构的指针，该结构包含 `littlefs` 在 RAM 存储介质上运行所需的所有配置参数。这包括读取、写入和擦除的功能指针，以及如块大小、块计数和缓存大小等参数。在项目设置中，我们使用 RAM 的一部分作为存储介质。基于 RAM 的 `littlefs` 配置定义在 C 文件 `Chapter13/lfs_raii/app/src/lfs_ramfs.c` 中。

+   格式化存储介质，以便与 `littlefs` 一起使用。`lfs_format` 函数在存储介质上初始化文件系统结构。此过程擦除任何现有数据并设置必要的元数据结构。格式化通常在第一次使用文件系统之前或重置时进行一次。

+   它挂载文件系统，使其准备好进行文件操作。`lfs_mount` 函数根据存储介质上的现有结构在 RAM 中初始化文件系统状态。在执行任何读取或写入等文件操作之前，此步骤是必需的。

接下来，让我们看看如何创建一个文件并向其中写入一些数据。代码如下所示：

```cpp
lfs_file_t file;
if(lfs_file_open(&lfs, &file, “song.txt”, LFS_O_WRONLY | LFS_O_CREAT) >= 0)
{
 const char * file_content = “These are some lyrics!”;
 lfs_file_write(&lfs,
 &file,
 reinterpret_cast<const void *>(file_content),
 strlen(file_content));
    lfs_file_close(&lfs, &file);
} 
```

此代码执行以下步骤：

+   声明了一个名为 `file` 的文件对象，类型为 `lfs_file_t`。此对象代表 `littlefs` 文件系统中的一个文件。它包含文件的状态，并且对于执行读取和写入等文件操作是必需的。

+   使用函数 `lfs_file_open` 尝试打开名为 `“song.txt”` 的文件进行写入。该函数提供了以下参数：

    +   `&lfs`：指向之前初始化和挂载的文件系统对象的指针。

    +   `&file`：指向将关联到打开文件的文件对象的指针。

    +   `“song.txt”`：要打开的文件名。

    +   `LFS_O_WRONLY | LFS_O_CREAT`：指定以只写模式打开文件，如果文件不存在则创建文件。

    +   如果 `lfs_file_open` 函数返回非负值，则代码尝试使用 `lfs_file_write` 函数向其写入一些数据。

+   我们将写入的内容声明为 `file_content` 字符串字面量。

+   函数 `lfs_file_write` 提供以下参数：

    +   `&lfs`：指向文件系统对象的指针。

    +   `&file`：指向与打开文件关联的文件对象的指针。

    +   `reinterpret_cast<const void *>(file_content)`：将字符字符串转换为函数所需的 `const void*` 指针。

    +   `strlen(file_content)`：要写入的字节数，基于字符串的长度计算。

+   在写入后关闭文件以确保数据完整性。`lfs_file_close` 将任何挂起的写入刷新到存储介质，并释放与文件关联的资源。

在将数据写入文件后，我们将尝试以读取模式打开相同的文件并从中读取数据。读取文件的代码如下所示：

```cpp
if(lfs_file_open(&lfs, &file, “song.txt”, LFS_O_RDONLY)>= 0) {
    std::array<char, 64> buff = {0};
 lfs_file_read(&lfs,
               &file,
               reinterpret_cast<void *>(buff.data()),
               buff.size() - 1);
    printf(“This is content from the file\r\n%s\r\n”, buff.data());
    lfs_file_close(&lfs, &file);
} 
```

此代码执行以下步骤：

+   尝试使用带有标志 `LFS_O_RDONLY` 的函数 `lfs_file_open` 打开文件 `“song.txt”` 以进行只读访问。

+   如果 `lfs_file_open` 函数返回非负值，则代码尝试从打开的文件中读取数据。

+   `std::array<char, 64> buff = {0}` 声明了一个名为 `buff` 的数组，大小固定为 `64` 个字符，并将所有元素初始化为零（`‘\0’`），确保如果将其作为 C 字符串处理，则缓冲区为空终止。

+   使用函数 `lfs_file_read` 从 `buff` 数组中读取打开的文件数据。该函数提供了以下参数：

    +   `&lfs`：指向文件系统对象的指针。

    +   `&file`：指向与打开文件关联的文件对象的指针。

    +   `reinterpret_cast<const void *>(buff.data())`：将 `buff` 的底层数据数组指针转换为函数所需的 `const void*` 指针。

+   `buff.size() – 1`：从文件中读取的字节数。减去 1 为字符串末尾的空终止符（`‘\0’`）保留空间。

+   读取数据后关闭文件以确保数据完整性。

你可以在 Renode 模拟器中运行完整示例。启动 Visual Studio Code，将其附加到正在运行的容器，打开 `Chapter13/lfs_raii` 项目，如 *第四章* 中所述，然后在 Visual Studio Code 终端中运行以下命令，或者在容器终端中直接运行：

```cpp
$ cd Chapter13/lfs_raii
$ cmake -B build
$ cmake --build build --target run_in_renode 
```

## 引入基于 RAII 的 C++ 包装器

现在，我们将使用 RAII 技术将 `littlefs` 功能包装在一个简单的 C++ 包装器中。我们将创建一个包含 `lfs` 和 `file` 类型的 `fs` 命名空间。让我们从以下所示的 `lfs` `struct` 代码开始：

```cpp
namespace fs{
`struct` lfs {
    lfs() = delete;
    static inline lfs_t fs_lfs;
    static void init() {
        const lfs_config * lfs_ramfs_cfg = get_ramfs_lfs_config();
        lfs_format(&fs_lfs, lfs_ramfs_cfg);
        lfs_mount(&fs_lfs, lfs_ramfs_cfg);
    }   
};
}; 
```

`struct` `lfs`的目的如下：

+   持有一个名为`fs_lfs`的`lfs_t`类型文件系统对象的实例，用于与`littlefs`文件系统交互。

+   实现用于通过调用`lfs_format`和`lfs_mount`函数来初始化文件系统的`static`方法`init`。必须在执行任何文件操作之前调用`init`方法。

接下来，让我们看看`file`类的定义：

```cpp
namespace fs{
class file {
public:
    file(const char * filename, int flags = LFS_O_RDONLY);
    ~file();
    [[nodiscard]] bool is_open() const;
    int read(std::span<char> buff);
    void write(std::span<const char> buff);
private:
    bool is_open_ = false;
    lfs_file_t file_;
};
}; 
```

此代码展示了类文件的特性和数据成员。接下来，我们将逐一介绍它们，从下面所示的构造函数开始：

```cpp
file(const char * filename, int flags = LFS_O_RDONLY) {
 if(lfs_file_open(&lfs::fs_lfs, &file_, filename, flags) >= 0) {
 is_open_ = true;
 }
} 
```

下面所示的`file`构造函数打开一个指定`filename`和`flags`的文件。如果文件打开成功，则将`is_open_`设置为 true。接下来，让我们看看下面所示的析构函数：

```cpp
~file() {
if(is_open_) {
        printf(“Closing file in destructor.\r\n”);
        lfs_file_close(&lfs::fs_lfs, &file_);
 }
} 
```

下面所示的析构函数将在文件已打开时关闭文件。它调用`lfs_file_close`来关闭文件并释放资源。构造函数和析构函数实现了 RAII 技术——创建对象将获取资源，当对象的生命周期结束时，析构函数将释放它们。接下来，让我们看看读取和写入方法：

```cpp
int read(std::span<char> buff) {
return lfs_file_read(&lfs::fs_lfs,
 &file_,
                     reinterpret_cast<void *>(buff.data()),
 buff.size() - 1);
}
int write(std::span<const char> buff) {
return lfs_file_write(&lfs::fs_lfs,
 &file_,
                      reinterpret_cast<const void *>(buff.data()),
 buff.size());
} 
```

`read`和`write`方法是对`lfs_file_read`和`lfs_file_write`函数的简单包装。`read`和`write`都使用`std::span`作为函数参数，以提高类型安全性和更好的灵活性，因为我们只需提供`std::array`即可。

## 使用 RAII 进行更清晰的文件管理

现在，我们将看到如何使用`fs`和`file`包装器与`littlefs`文件系统一起工作。代码如下所示：

```cpp
fs::lfs::init();
{
    fs::file song_file(“song.txt”, LFS_O_WRONLY | LFS_O_CREAT);
 if(song_file.is_open()) {
 song_file.write(“These are some lyrics!”);
 // destructor is called on song_file object
 // ensuring the file is closed
} 
```

我们首先通过调用`fs::lfs::init()`初始化文件系统。接下来，我们引入局部作用域以演示对析构函数的调用并执行后续步骤：

+   以写入模式打开`“song.txt”`（如果不存在则创建它）。

+   如果文件成功打开，则在文件中写入一个字符串字面量。

+   当退出作用域时，将调用析构函数，确保文件被关闭。

接下来，我们将打开文件并从中读取数据。代码如下所示：

```cpp
fs::file song_file(“song.txt”);
std::array<char, 64> buff = {0};
if(song_file.is_open()) {
 song_file.read(buff);
 printf(“This is content from the file\r\n%s\r\n”,
 buff.data());
} 
```

此代码执行以下步骤：

+   以读取模式打开`“song.txt”`。

+   声明`std::array<char, 64> buff`，初始化为零。

+   如果文件打开成功，则从文件中读取`buff`中的数据。

您可以在 Renode 模拟器中运行完整示例。启动 Visual Studio Code，将其附加到正在运行的容器，打开`Chapter13/lfs_raii`项目，如*第四章*中所述，然后在 Visual Studio Code 终端中运行以下命令，或者在容器终端中直接运行它们：

```cpp
$ cd Chapter13/lfs_raii
$ cmake -B build -DMAIN_CPP_FILE_NAME=main_lfs_raii.cpp
$ cmake --build build --target run_in_renode 
```

我们为`littlefs`库编写的简单 C++包装器应用了 RAII 原则，确保在对象的生命周期结束时调用析构函数，从而正确处理资源。这确保了即使在代码中有多个返回路径的情况下，文件也会被关闭。它还简化了开发体验，因为代码更简洁、更清晰。使用`std::span`增加了安全性。

# 摘要

在本章中，我们介绍了在 C++项目中使用 C 库的几种技术。通过将 C 代码封装在 C++类中，我们可以在松散耦合的软件模块中更好地组织我们的代码。C++增加了类型安全性，编译时特性使我们能够轻松地将 C 封装器组织在`static`类中。

应用 RAII（资源获取即初始化）非常简单，它为我们提供了一个强大的机制来处理资源管理，正如我们在`littlefs`文件系统示例中所见。

在下一章中，我们将探讨裸机固件中的超级循环，并看看我们如何通过 C++中的序列器等机制来增强它。

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：

[嵌入式系统](https://packt.link/embeddedsystems)

![Discord 二维码](img/QR_code_Discord.png)
