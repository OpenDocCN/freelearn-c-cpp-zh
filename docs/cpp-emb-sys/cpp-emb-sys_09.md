

# 第七章：加强固件 - 实用的 C++错误处理方法

为了确保固件正常工作，我们必须处理来自供应商特定代码、项目中使用的库以及我们自己的代码的错误。错误代码是 C 中的标准错误处理机制，它们也在 C++中使用。然而，C++为我们提供了其他工具，最显著的是异常，由于大型二进制足迹和非确定性，这些异常通常在嵌入式项目中被避免。尽管如此，我们将在本章中讨论 C++中的异常，以展示它们在错误处理过程中的好处。

除了异常之外，C++还提供了更多用于错误处理的选项，这些选项也将在本章中讨论。本章的目标是理解错误代码的潜在问题，并了解如何在 C++中减轻这些问题。

在本章中，我们将涵盖以下主要主题：

+   错误代码和断言

+   异常

+   `std::optional`和`std::expected`

# 技术要求

为了充分利用本章内容，我强烈建议在阅读示例时使用 Compiler Explorer（[`godbolt.org/`](https://godbolt.org/)）。选择 GCC 作为您的编译器，并针对 x86 架构。这将允许您看到标准输出（stdio）结果，并更好地观察代码的行为。由于我们使用现代 C++特性，请确保选择 C++23 标准，通过在编译器选项框中添加`-std=c++23`。

Compiler Explorer 使得尝试代码、调整代码并立即看到它如何影响输出和生成的汇编变得容易。示例可在 GitHub 上找到（[`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter07`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter07)）。

# 错误代码和断言

**错误代码**是 C 中报告和处理错误的一种常见方式。它们在 C++中仍然被使用。一个失败的函数通过枚举代码报告错误，这些代码由调用者检查并适当处理。让我们从调用者和被调用者的角度分析错误代码是如何工作的。

返回错误的函数必须有一个暴露给调用者的错误列表。这个列表在软件生命周期中维护，并且可能会发生变化。枚举错误代码可以添加、删除或修改。调用者必须知道被调用者返回的错误代码，并且需要处理它们。或者，如果它不知道如何处理错误，它应该将错误进一步传播到调用栈中。

让我们观察一个返回错误的简单函数示例，并分析这对使用此函数的代码有何影响：

```cpp
enum class error {
    Ok,
    Error1,
    Error2,
    Unknown
};
error h() {
    return error::Error1;
}
error g() {
    auto err = h();
    if(err!=error::Ok) {
        if(err == error::Error1) {
       // handle error directly
        }
        else if(err == error::Error2) {
            // propagate this error
return err;
        }
        else {
            // unknown error
return error::Unknown;
        }
    }
    return error::Ok;
}
void f() {
    auto err = g();
    if(err==error::Ok) {
        printf("Succes\r\n");
    }
    else {
        // handle errors
    }
} 
```

在前面的示例中，`h`函数返回了一个`enum class error`的错误。`g`函数调用`h`函数并执行以下步骤：

1.  检查`h`是否返回了一个与`error::Ok`不同的错误。这表明`h`函数没有完成其任务，并且存在应该被处理的错误。

1.  如果`h`返回了一个错误，检查它是否是`error::Error1`。在这种情况下，`g`知道如何处理这个错误，并处理它。

1.  如果`h`返回`error::Error2`，`g`无法处理它，并将它向上传递到调用栈。

1.  返回`error::Ok`以向上传递到调用栈，表示一切顺利。

函数`g`被`f`调用，而`f`也需要了解`enum class error`中定义的错误。它应该处理它们或将它们向上传递到调用栈。

错误代码依赖于设计契约。调用者必须检查被调用者是否返回了错误，如果是，它需要处理它或将它向上传递到调用栈。现在，我们可以识别出这种简单方法的一些潜在问题：

+   我们不能通过调用者强制执行错误处理。它可能只是丢弃返回值。

+   调用者可能会忘记处理一些错误情况。

+   调用者可能会忘记将错误向上传递到调用栈。

这些是严重的设计缺陷，给代码开发增加了额外的负担。如果我们忘记处理某个错误，就没有逃逸路径。程序将处于未知状态，这可能导致不希望的行为。

我们可以使用`nodiscard`属性来解决第一个问题。它可以与函数声明或枚举声明一起使用。在我们的情况下，我们可以用它与`enum class` `error`声明一起使用，如下所示：

```cpp
enum class [[nodiscard]] error {
    Ok,
    Error1,
    Error2,
    Unknown
}; 
```

当调用返回`enum class`错误的函数时，如果丢弃返回值，编译器会鼓励发出警告。如果我们从我们的示例中调用`g`或`h`函数，GCC 将发出类似于以下警告：

```cpp
<source>:48:6: warning: ignoring returned value of type 'error', declared with attribute 'nodiscard' [-Wunused-result] 
```

如果我们将编译器设置为将所有警告视为错误，这将破坏编译过程，并迫使我们使用代码中的返回值。尽管`nodiscard`属性很有用并且应该用于类似用例，但它并不是我们问题的完整解决方案。它将强制使用返回值，但调用者仍然可能未能检查所有可能的错误代码并正确处理它们。

几乎每个应用程序都有一些无法恢复的错误类型，唯一合理的事情是记录它们，向用户显示（如果可能），并终止程序，因为继续这样的程序状态是没有意义的。对于这些类型的错误，我们可以使用全局错误处理器，因为它们太重要了，不能任其处于野生状态且可能不会被调用者处理。

## 全局错误处理器

**全局错误处理器**可以实施为自由函数。它们在系统范围内用于处理无法恢复的错误，以及在需要由于错误的严重性而停止固件执行时。

让我们看看一个使用加速度计的固件示例。如果与加速度计的 I²C 通信出现任何问题，继续代码执行是没有意义的——固件将向用户显示一条消息并终止：

```cpp
#include <cstdio>
#include <cstdint>
#include <cstdlib>
int i2c_read(uint8_t *data, size_t len) {
    return 0;
}
namespace error {
    struct i2c_failed{};
    struct spi_failed{};
    void handler(i2c_failed err) {
        printf("I2C error!\r\n");
        exit(1);
    }
    void handler(spi_failed err) {
        printf("SPI error!\r\n");
        exit(1);
    }
};
class accelerometer {
public:
    struct data {
        int16_t x;
        int16_t y;
        int16_t z;
    };
    data get_data() {
        uint8_t buff[6];
        if(i2c_read(buff, 6) != 6) {
            error::handler(error::i2c_failed{});
        }
        return data{};
    }
};
int main () {
    accelerometer accel;
    auto data = accel.get_data();
    return 0;
} 
```

在前面的例子中，我们有一个`accelerometer`类，它有一个`get_data`方法，该方法使用从供应商特定的 HAL 中导入的 C 语言的`i2c_read`函数（让我们假设这是这种情况）。

`i2c_read`函数返回读取的字节数。在我们的例子中，返回值被模拟为`0`，这样我们就可以模拟加速度计（或 I²C 总线）的错误行为。如果`i2c_read`返回与请求的字节数不同的数字，`get_data`将调用`error::handler`。

我们使用标签分派机制实现了一个错误处理器。我们通过所谓的标签或空类型重载了`error::handler`函数。在我们的例子中，我们有两个标签，`i2c_failed`和`spi_failed`，以及两个重载的错误处理器。与使用`enum`定义错误代码相比，标签分派有几个优点：

+   我们需要在代码中使用的每个标签上重载错误处理器。每个错误类型都单独实现错误处理器。这增加了代码的可读性。

+   如果我们调用了一个未重载的错误处理器，编译将失败，迫使我们实现它。

在我们的例子中，错误处理器将使用`printf`函数打印一条消息，并调用`exit`函数，从而有效地终止程序。在现实世界的情况下，我们如何处理错误取决于应用程序。例如，对于医疗设备，如果错误后关键操作变得不安全，我们首先尝试从错误中恢复。

如果恢复失败，系统将进入关键错误状态，通知医疗人员，并优雅地终止治疗操作。

在 I²C 总线上发生错误，或者更普遍地说，与外部设备通信失败，必须通过健壮的错误处理机制适当地处理。

另一方面，有一些条件表明存在编程错误——这些是在代码正确的情况下不应发生的情况。这包括违反先决条件，例如由于代码中的逻辑错误，输入参数超出预期边界。在这种情况下继续执行可能导致未定义的行为或系统不稳定。为了在开发期间检测这些编程错误，我们使用断言。

## 断言

断言主要用于开发期间，通过验证代码中特定点的某些条件是否成立来检测编程错误。当出现意外条件时，它们通过停止执行来帮助识别逻辑错误和错误的假设。标准库中的`<cassert>`定义了一个宏断言。它用于检查逻辑表达式，如果逻辑表达式为假，则打印诊断信息并调用`std::abort`，从而有效地终止程序。

为了更好地理解断言以及如何使用它们，让我们看一下以下代码示例：

```cpp
#include <cassert>
#include <cstdint>
enum class option : std::uint8_t {
    Option1 = 0,
    Option2,
    Option3,
    Last
};
option uint8_to_option(uint8_t num) {
    assert(num < static_cast<uint8_t>(option::Last));
    return static_cast<option>(num);
}
int main() {
    const option opt = uint8_to_option(3);
    return 0;
} 
```

在前面的示例中，我们已定义了以 `uint8_t` 作为底层类型的 `option` 枚举类。我们将使用它来允许用户通过网络接口选择一个选项，并确保从 `uint8_t` 到 `option` 枚举的转换始终正确。如果接收到的 `uint8_t` 参数不小于 `option::Last`，则 `uint8_to_option` 函数将断言。

在示例中，我们用参数 `3` 调用了 `uint8_to_option`，这并不小于 `option::Last`，这意味着断言宏将打印以下诊断信息，并通过调用 `std::abort` 来终止程序：

```cpp
assertion "num < static_cast<uint8_t>(option::Last)" failed: file "/home/amar/projects/Cpp-in-Embedded Systems/Chapter07/error_handling/app/src/main.cpp", line 21, function: option uint8_to_option(uint8_t) 
```

现在，这是一个相当长的调试语句。让我们看看 `assert` 宏的定义：

```cpp
#define assert(expr)                             \
     (static_cast <bool> (expr)                  \
      ? void (0)                                 \
      : __assert_fail (#expr, 
__ASSERT_FILE,            \
                       __ASSERT_LINE,            \
                       __ASSERT_FUNCTION)) 
```

我们可以看到表达式被转换为 `bool` 类型，如果表达式为真，三元运算符不执行任何操作；如果表达式为假，它将调用 `__assert_fail` 函数。`assert` 宏将表达式作为字符串字面量传递，将文件名作为字符串字面量传递，传递行号，还将函数名作为字符串字面量传递。所有这些字符串字面量都必须存储在二进制文件中，占用宝贵的内存。

断言可以通过在包含 `<cassert>` 之前定义 `NDEBUG` 宏来禁用，如下所示：

```cpp
#define NDEBUG
#include <cassert> 
```

我们也可以使用构建系统来定义 `NDEBUG`。如果 `<cassert>` 包含之前定义了 `NDEBUG`，则 `assert` 宏将不执行任何操作。这个选项留给我们使用，以防我们想要禁用断言，因为它们最常用于调试构建，而在生产构建中被禁用。它们应该在安全关键软件验证之前被禁用。

标准库中实现的 `assert` 宏不适合嵌入式系统，因为它包含了文件名、函数名和 `assert` 表达式作为字符串字面量，最终存储在嵌入式目标的闪存中。此外，断言主要用于调试期间，它们通常在生产构建中被禁用。尽管如此，在生产构建中启用断言仍然有一些好处，因为如果它们在表达式评估为 `false` 时记录数据，它们可以提供宝贵的调试信息。

我们将检查使用断言记录信息的替代方法。正如我们已经得出的结论，默认的断言宏实现不适合嵌入式目标，尽管它包含了对调试有用的信息：文件名、函数名和行号。我们不需要一个描述断言宏行在代码中确切位置的冗长字符串，我们可以简单地记录程序计数器，并使用映射文件和 `addr2line` 工具将地址转换为确切的行。我们可以在以下代码中看到一个简单的宏定义和一个辅助函数来实现这一点：

```cpp
void log_pc_and_halt(std::uint32_t pc) {
    printf("Assert at 0x%08lX\r\n", pc);
    while(true) {}
}
#define light_assert(expr)         \
        (static_cast<bool> (expr)  \
        ? void (0)                 \
        : log_pc_and_halt(hal::get_pc())    \
        ) 
```

我们定义了一个名为`light_assert`的宏，它不是调用`__assert_failed`，而是调用`log_pc_and_halt`。它将`hal::get_pc`的返回值作为参数传递给`log_pc_and_halt`。要查看此代码的实际效果，你可以查看`Chapter07/error_handling`项目中的示例。

本章的项目配置允许你配置它使用不同的主 C++文件，并使用 CMake 配置将要使用哪个文件。让我们使用以下命令启动我们的 Docker 容器：

```cpp
$ docker start dev_env
$ docker exec -it dev_env /bin/bash 
```

这应该让我们进入 Docker 终端。运行`ls –l`以确保`Cpp-in-Embedded-Systems`仓库已克隆。如果没有，使用以下命令克隆它：

```cpp
$ git clone https://github.com/PacktPublishing/Cpp-in-Embedded-Systems.git 
```

启动 Visual Studio Code，将其附加到正在运行的容器，并按照*第四章*中所述打开`Chapter07/error_handling 项目`，然后在 Visual Studio Code 终端中运行以下命令，或者直接在容器终端中运行它们：

```cpp
$ cd Chapter07/error_handling
$ cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMAIN_CPP_FILE_NAME=main_assert.cpp
$ cmake --build build --target run_in_renode 
```

前面的命令将使用`app/src/main_assert.cpp`文件构建固件，并在 Renode 模拟器中运行它。你应该在终端看到类似的输出：

```cpp
14:11:06.6293 [INFO] usart2: [host: 0.31s (+0.31s)|virt: 0s (+0s)] Assert example
14:11:06.6455 [INFO] usart2: [host: 0.32s (+15.87ms)|virt: 0.11ms (+0.11ms)] Assert at 0x08000F74 
```

正如我们所见，断言评估表达式为假，并打印出`0x08000F74`程序计数器值。我们可以使用以下命令将此值转换为源文件的行：

```cpp
$ arm-none-eabi-addr2line --exe bare.elf 0x08000F74 
```

这将产生以下输出：

```cpp
/workspace/Cpp-in-Embedded-Systems/Chapter07/error_handling/app/src/main_assert.cpp:30 (discriminator 1) 
```

正如你所见，我们能够通过这种方法获取断言源的确切行，并且只需记录 4 字节的数据（地址）。在这个实现中，`log_pc_and_halt`只是打印出地址。在生产实现中，我们可以将地址存储在非易失性存储器中，并用于事后调试。

`hal::get_pc()`函数使用`inline`指定符声明。我们使用`inline`作为对编译器的提示，将函数的指令直接插入到调用点，即不进行函数调用。编译器不一定需要遵守我们的意图，这可以通过使用`O0`优化级别构建此示例来观察到。

练习题！

作为练习，编辑`CMakeLists.txt`中的`CMAKE_C_FLAGS_DEBUG`和`CMAKE_CXX_FLAGS_DEBUG`，并将`Og`替换为`O0`。构建并运行程序，然后在输出上运行`addr2line`实用程序。为了减轻这一担忧，你可以定义一个宏来替代`hal::get_pc()`函数。

我们使用断言来捕获编程错误——如果代码正确，这些情况永远不会发生。它们通常用于验证关键函数内部的内部假设和不变性。断言的主要目的是用于调试；它们帮助开发者在开发阶段找到并修复错误。然而，正如我们所看到的，定制的断言也可以在生产构建中提供宝贵的洞察力，用于事后分析。虽然断言在开发过程中用于检测编程错误很有用，但它们不能替代生产代码中的正确错误处理。错误代码可能很繁琐，因为它们需要手动将错误传播到调用栈。C++提供了异常作为这些问题的解决方案，提供了一种结构化的方式来处理错误，而不会在代码中添加错误检查逻辑。

接下来，我们将深入了解 C++ 异常，以更好地理解它们从错误处理角度提供的优势。

# 异常

C++中的异常是基于抛出和捕获任意类型对象的原理的错误处理机制。从标准库中抛出的所有异常都源自于在 `<exception>` 头文件中定义的 `std::exception` 类。我们将可能抛出异常的代码放在 `try` 块中，并在 `catch` 子句中定义我们想要捕获的异常类型，如下面的示例所示：

```cpp
 std::array<int, 4> arr;
    try {
      arr.at(5) = 6;
    }
    catch(std::out_of_range &e) {
      printf("Array out of range!\r\n");
    } 
```

在前面的示例中，我们定义了 `std::array arr`，一个包含四个成员的整数数组。在 `try` 块中，我们尝试访问索引为 `5` 的元素，这显然超出了定义的范围，`at` 方法将抛出 `std::out_of_range` 异常。为了运行此示例，请转到 `Chapter07/error_handling` 文件夹，确保已删除 `build` 文件夹，并运行以下命令：

```cpp
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Debug -DMAIN_CPP_FILE_NAME=main_exceptions.cpp
$ make –j4
$ make run_in_renode 
```

你应该在终端中看到打印出 `Array out of range!`。

现在，在构建示例时，你可能已经注意到二进制文件的大小达到了惊人的 88 KB。发生了什么？

为了启用异常，除了使用 `-fexceptions` 编译器标志外，我们还必须禁用之前示例中使用的 nano 规范。Nano 规范定义了 C 标准库 `newlib-nano` 的使用以及大小优化的 `libstdc++` 和 `libsupc++` 库。这些库在没有异常支持的情况下构建，如果我们使用它们，任何尝试抛出异常都将导致调用 `std::abort`。通过禁用 nano 规范，我们使用了一个未优化的 C++ 标准库，这导致了 88 KB 的二进制文件大小。可以从带有异常支持的源构建大小优化的标准 C++ 库，这将有助于减少二进制文件的大小。

如果没有捕获到异常，将调用 `std::terminate_handler`。我们可以使用 `std::set_terminate` 函数替换默认的处理程序，如下面的示例所示：

```cpp
 std::set_terminate([]() {
        printf("My terminate handler!\r\n");
        while(true){}
    }); 
```

在上述示例中，我们提供了一个 lambda 作为终止处理程序。作为一个练习，尝试使用超出范围的索引访问前一个示例中的数组，但不在`try`块中。这应该会触发终止处理程序，并调用我们传递给`std::set_terminate`函数的 lambda。

异常沿着调用栈向上传播。让我们通过以下示例来演示异常传播：

```cpp
template <class T, std::size_t N> struct ring_buffer {
  std::array<T, N> arr;
  std::size_t write_idx = 0;
  void push(T t) {
    arr.at(write_idx++) = t;
  }
};
int main()
{
    ring_buffer<int, 4> rb;
    try {
      for(int i = 0; i < 6; i++) {
        rb.push(i);
      }
    }
    catch(std::out_of_range &e) {
      printf("Ring buffer out of range!\r\n");
    }
    return 0;
} 
```

上述示例基于前几章中使用的环形缓冲区，该缓冲区使用`std::array`作为底层容器。在`push`方法中，它没有检查写入索引，这意味着如果我们调用`push`方法超过`N`次，数组的`at`方法将抛出异常。在`push`方法中抛出了异常，但没有`try-catch`块，它只在`main`函数的`catch`块中被捕获。

您可以使用以下说明在 Renode 模拟器中运行前面的示例。启动 Visual Studio Code，将其附加到正在运行的容器，按照*第四章*中所述打开`Chapter07/error_handling 项目`，然后在 Visual Studio Code 终端中运行以下命令，或者在容器终端中直接运行它们：

```cpp
$ cd Chapter07/error_handling
$ cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMAIN_CPP_FILE_NAME=main_exceptions.cpp
$ cmake --build build --target run_in_renode 
```

异常传播对于我们不希望手动使用错误代码在软件层之间传播的错误类型非常有用。然而，异常的问题在于它们与错误代码不同，在函数声明中是不可见的。我们需要依赖于良好的文档来了解哪个函数抛出错误以及错误在哪里被处理。

有一种说法是，异常用于非常罕见的异常错误。但什么是异常错误？这取决于库、应用程序和用例。很难概括。加速度计的读取失败可能是一个可恢复的错误，可以通过重置它来解决。我们可以在失败的 I²C 总线通信上抛出异常，并且捕获这个错误的上层可能决定尝试重置加速度计。

如果通过 DAC 未能控制升压稳压器输出可能也是可恢复的，但因为我们正在实施医疗设备，所以我们可能希望终止程序，这可能是防止对用户造成任何损害的最佳行动。在这种情况下，我们希望尽可能快地做出反应，异常传播和堆栈展开可能不是期望的，因此我们将依赖于全局处理程序或断言。

异常伴随着代价，包括闪存和 RAM 内存消耗，并且执行时间不能总是得到保证，如果我们正在处理硬实时系统，这会成为一个问题。但它们也解决了错误传播的问题，并强制执行错误处理。如果没有为特定类型提供`catch`子句，`std::terminate_handler`将被调用，程序将不会继续执行。

错误代码和异常可以共存，并且通常如此。嵌入式 C++ 项目通常使用 C 库或遗留的 C++ 代码，这些代码通常使用错误代码。我们可以通过将它们用于非常罕见的错误来从异常中受益，这为我们的固件增加了额外的鲁棒性。然而，是否使用它们的决定受到可用内存资源和我们所从事的项目类型的影

接下来，我们将介绍 C++ 的 `std::optional` 和 `std::expected` 模板类，这些类用作函数的返回类型。

# `std::optional` 和 `std::expected`

C++17 引入了 `std::optional`，这是一个模板类，它要么有一个值，要么什么也没有。这在函数可能返回或不返回值的情况下非常有用。为了更好地理解它，让我们通过以下示例来了解：

```cpp
#include <cstdio>
#include <optional>
struct sensor {
    struct data {
        int x;
        int y;
    };
    static inline bool ret_val = true;
    static std::optional<data> get_data() {
        ret_val = !ret_val;
        if(ret_val) {
            return data{4, 5};
        }
        else {
            return std::nullopt;
        }
    }
};
int main()
{
    const auto get_data_from_main = [] () {
        auto result = sensor::get_data();
        if(result) {
            printf("x = %d, y = %d\r\n", (*result).x, (*result).y);
        }
        else {
            printf("No data!\r\n");
        }
    };
    get_data_from_main();
    get_data_from_main();
    return 0;
} 
```

在前面的示例中，我们有一个具有 `get_data` 方法的 `sensor` 结构体，该方法在某些条件满足时返回一个值。否则，它不返回值。传感器不在错误状态，它只是还没有准备好数据。为此，我们使用 `std::optional<data>` 来声明传感器可能返回或不返回 `data` 结构体。我们使用 `ret_val` 布尔值来模拟 `get_data` 函数每第二次调用时数据就绪。

在 `main` 函数中，我们创建了 `get_data_from_main` lambda 表达式，它调用传感器的 `get_data` 方法。`std::optional<data>` 返回值在 `if` 语句中被转换为布尔值。如果它被转换为 `true`，则表示它包含数据，否则它不包含任何数据。我们通过解引用 `result` 对象来访问 `data` 类型。

C++ 23 引入了 `std::expected<T, E>`，这是一个模板类，它要么包含类 `T` 的预期对象，要么包含类 `E` 的意外对象。为了更好地理解这一点，让我们通过以下示例来了解：

```cpp
#include <cstdio>
#include <expected>
struct ble_light_bulb {
    enum class error {
        disconnected,
        timeout
    };
    struct config {
        int r;
        int g;
        int b;
    };
    bool ret_val;
    std::expected<config, error> get_config() {
        ret_val = !ret_val;
        if(ret_val) {
            return config {10, 20, 30};
        }
        else {
            return std::unexpected(error::timeout);
        }
    }
};
int main()
{  
    ble_light_bulb bulb;
    const auto get_config_from_main = [&bulb]() {
        auto result = bulb.get_config();
        if(result.has_value()) {
            auto conf = result.value();
            printf("Config r %d, g %d, b %d\r\n", conf.r, conf.g, conf.b);
        } else {
            auto err = result.error();
            using bulb_error = ble_light_bulb::error;
            if(err == bulb_error::disconnected) {
                printf("The bulb is disconnected!\r\n");
            }
            else if(err == bulb_error::timeout) {
                printf("Timeout!\r\n");
            }
        }
    };
    get_config_from_main();
    get_config_from_main();
    return 0;
} 
```

在前面的示例中，我们有一个 `ble_light_bulb` 结构体，一个带有 `get_config` 方法的 BLE（蓝牙低功耗）灯泡，该方法通过 BLE 连接从灯泡读取一些配置数据。此方法返回 `config` 或 `error`。在 `main` 函数中，我们定义了 `get_config_from_main` lambda 表达式，它调用 `ble_light_bulb` 对象上的 `get_config` 方法。我们使用预期返回对象上的 `has_value` 方法来检查它是否包含预期的值。我们使用 `value` 方法来访问预期的值或使用 `error` 方法来访问 `error` 对象。

您可以使用以下说明在 Renode 模拟器中运行前面的示例。启动 Visual Studio Code，将其附加到正在运行的容器，按照 *第四章* 中所述打开 `Chapter07/error_handling project`，然后在 Visual Studio Code 终端中运行以下命令，或者直接在容器终端中运行它们：

```cpp
$ cd Chapter07/error_handling
$ cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMAIN_CPP_FILE_NAME=main_expected.cpp
$ cmake --build build --target run_in_renode 
```

# 摘要

在本章中，我们分析了 C++ 中的不同错误处理策略。我们讨论了错误代码、全局处理程序、断言、异常、`std::optional` 和 `std::expected`。我们学习了每种方法的优缺点，以及在哪些情况下应用它们是有意义的。

在下一章中，我们将更详细地介绍模板。

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

[嵌入式系统](https://packt.link/embeddedsystems)

![Discord 二维码](img/QR_code_Discord.png)
