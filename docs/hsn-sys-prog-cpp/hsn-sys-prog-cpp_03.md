# 第三章：C 和 C++的系统类型

通过系统程序，诸如整数类型之类的简单事物变得复杂。整个章节都致力于解决在进行系统编程时出现的常见问题，特别是在为多个 CPU 架构、操作系统和用户空间/内核通信（如系统调用）进行系统编程时出现的问题。

本章包括以下主题：

+   解释 C 和 C++提供的默认类型，包括大多数程序员熟悉的类型，如`char`和`int`

+   回顾`stdint.h`提供的一些标准整数类型，以解决默认类型的限制

+   结构打包和与优化和类型转换相关的复杂性

# 技术要求

要编译和执行本章中的示例，读者必须具备以下条件：

+   一个能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请访问以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter03`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter03)。

# 探索 C 和 C++的默认类型

C 和 C++语言提供了几种内置类型，无需额外的头文件或语言特性。在本节中，我们将讨论以下内容：

+   `char`，`wchar_t`

+   `short int`，`int`，`long int`

+   `float`，`double`，`long double`

+   `bool`（仅限 C++）

# 字符类型

C 和 C++中最基本的类型是以下字符类型：

```cpp
#include <iostream>

int main(void)
{
    char c = 0x42;
    std::cout << c << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// B
```

`char`是一个整数类型，在大多数平台上，它的大小为 8 位，必须能够接受无符号的值范围为[`0`，`255`]，有符号的值范围为[`-127`，`127`]。`char`与其他整数类型的区别在于，`char`具有特殊含义，对应着**美国信息交换标准代码**（**ASCII**）。在前面的示例中，大写字母`B`由 8 位值`0x42`表示。需要注意的是，虽然`char`可以用来简单表示 8 位整数类型，但它的默认含义是字符类型；这就是为什么它具有特殊含义。例如，考虑以下代码：

```cpp
#include <iostream>

int main(void)
{
    int i = 0x42;
    char c = 0x42;

    std::cout << i << '\n';
    std::cout << c << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 66
// B
```

在前面的示例中，我们使用`int`（稍后将解释）和`char`来表示相同的整数类型`0x42`。然而，这两个值以两种不同的方式输出到`stdout`。整数以整数形式输出，而使用相同的 API，`char`以其 ASCII 表示形式输出。此外，`char`类型的数组在 C 和 C++中被认为是 ASCII 字符串类型，这也具有特殊含义。以下代码显示了这一点：

```cpp
#include <iostream>

int main(void)
{
    const char *str = "Hello World\n";
    std::cout << str;
}

// > g++ scratchpad.cpp; ./a.out
// Hello World
```

从前面的示例中，我们了解到以下内容。我们使用`char`指针（在这种情况下，无界数组类型也可以）定义了一个 ASCII 字符串；`std::cout`默认情况下知道如何处理这种类型，而`char`数组具有特殊含义。将数组类型更改为`int`将无法编译，因为编译器不知道如何将字符串转换为整数数组，而`std::cout`默认情况下也不知道如何处理整数数组，尽管在某些平台上，`int`和`char`实际上可能是相同的类型。

像`bool`和`short int`一样，字符类型在表示 8 位整数时并不总是最有效的类型，正如前面的代码所暗示的，在某些平台上，`char`实际上可能比 8 位更大，这是我们在讨论整数时将进一步详细讨论的一个主题。

为了进一步研究`char`类型，以及本节讨论的其他类型，让我们利用`std::numeric_limits{}`类。这个类提供了一个简单的包装器，围绕着`limits.h`，它为我们提供了一种查询在给定平台上如何实现类型的方法，使用一组静态成员函数实时地。

例如，考虑下面的代码：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes_signed = sizeof(signed char);
    auto min_signed = std::numeric_limits<signed char>().min();
    auto max_signed = std::numeric_limits<signed char>().max();

    auto num_bytes_unsigned = sizeof(unsigned char);
    auto min_unsigned = std::numeric_limits<unsigned char>().min();
    auto max_unsigned = std::numeric_limits<unsigned char>().max();

    std::cout << "num bytes (signed): " << num_bytes_signed << '\n';
    std::cout << "min value (signed): " << +min_signed << '\n';
    std::cout << "max value (signed): " << +max_signed << '\n';

    std::cout << '\n';

    std::cout << "num bytes (unsigned): " << num_bytes_unsigned << '\n';
    std::cout << "min value (unsigned): " << +min_unsigned << '\n';
    std::cout << "max value (unsigned): " << +max_unsigned << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes (signed): 1
// min value (signed): -128
// max value (signed): 127

// num bytes (unsigned): 1
// min value (unsigned): 0
// max value (unsigned): 255
```

在前面的例子中，我们利用`std::numeric_limits{}`来告诉我们有符号和无符号`char`的最小和最大值（应该注意的是，本书中的所有示例都是在标准的英特尔 64 位 CPU 上执行的，假设这些相同的示例实际上可以在不同的平台上执行，返回的值可能是不同的）。`std::numeric_limits{}`类可以提供关于类型的实时信息，包括以下内容：

+   有符号或无符号

+   转换限制，如四舍五入和表示类型所需的总位数

+   最小值和最大值信息

在前面的例子中，64 位英特尔 CPU 上的`char`大小为 1 字节（即 8 位），对于无符号`char`取值范围为[`0`,`255`]，有符号`char`取值范围为[`-127`,`127`]，这是规范规定的。让我们来看一下宽字符`char`或`wchar_t`：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes_signed = sizeof(signed wchar_t);
    auto min_signed = std::numeric_limits<signed wchar_t>().min();
    auto max_signed = std::numeric_limits<signed wchar_t>().max();

    auto num_bytes_unsigned = sizeof(unsigned wchar_t);
    auto min_unsigned = std::numeric_limits<unsigned wchar_t>().min();
    auto max_unsigned = std::numeric_limits<unsigned wchar_t>().max();

    std::cout << "num bytes (signed): " << num_bytes_signed << '\n';
    std::cout << "min value (signed): " << +min_signed << '\n';
    std::cout << "max value (signed): " << +max_signed << '\n';

    std::cout << '\n';

    std::cout << "num bytes (unsigned): " << num_bytes_unsigned << '\n';
    std::cout << "min value (unsigned): " << +min_unsigned << '\n';
    std::cout << "max value (unsigned): " << +max_unsigned << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes (signed): 4
// min value (signed): -2147483648
// max value (signed): 2147483647

// num bytes (unsigned): 4
// min value (unsigned): 0
// max value (unsigned): 4294967295
```

`wchar_t`表示 Unicode 字符，其大小取决于操作系统。在大多数基于 Unix 的系统上，`wchar_t`为 4 字节，可以表示 UTF-32 字符类型，如前面的例子所示，而在 Windows 上，`wchar_t`为 2 字节，可以表示 UTF-16 字符类型。在这两种操作系统上执行前面的例子将得到不同的输出。

这是非常重要的，这个问题定义了整个章节的基本主题；C 和 C++提供的默认类型取决于 CPU 架构、操作系统，有时还取决于应用程序是在用户空间还是内核中运行（例如，当 32 位应用程序在 64 位内核上执行时）。在系统编程时，永远不要假设在与系统调用进行接口时，你的应用程序对特定类型的定义与 API 所假定的类型相同。这种假设往往是无效的。

# 整数类型

为了进一步解释默认的 C 和 C++类型是由它们的环境定义的，而不是由它们的大小定义的，让我们来看一下整数类型。有三种主要的整数类型——`short int`，`int`和`long int`（不包括`long long int`，在 Windows 上实际上是`long int`）。

`short int`通常比`int`小，在大多数平台上表示为 2 字节。例如，看下面的代码：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes_signed = sizeof(signed short int);
    auto min_signed = std::numeric_limits<signed short int>().min();
    auto max_signed = std::numeric_limits<signed short int>().max();

    auto num_bytes_unsigned = sizeof(unsigned short int);
    auto min_unsigned = std::numeric_limits<unsigned short int>().min();
    auto max_unsigned = std::numeric_limits<unsigned short int>().max();

    std::cout << "num bytes (signed): " << num_bytes_signed << '\n';
    std::cout << "min value (signed): " << min_signed << '\n';
    std::cout << "max value (signed): " << max_signed << '\n';

    std::cout << '\n';

    std::cout << "num bytes (unsigned): " << num_bytes_unsigned << '\n';
    std::cout << "min value (unsigned): " << min_unsigned << '\n';
    std::cout << "max value (unsigned): " << max_unsigned << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes (signed): 2
// min value (signed): -32768
// max value (signed): 32767

// num bytes (unsigned): 2
// min value (unsigned): 0
// max value (unsigned): 65535
```

如前面的例子所示，代码获取了有符号`short int`和无符号`short int`的最小值、最大值和大小。这段代码的结果表明，在运行 Ubuntu 的英特尔 64 位 CPU 上，`short int`，无论是有符号还是无符号，都返回 2 字节的表示。

英特尔 CPU 相对于其他 CPU 架构提供了一个有趣的优势，因为英特尔 CPU 被称为**复杂指令集计算机**（**CISC**），这意味着英特尔**指令集架构**（**ISA**）提供了一长串复杂的指令，旨在为英特尔汇编的编译器和手动作者提供高级功能。其中的一个特性是英特尔处理器能够在字节级别执行**算术逻辑单元**（**ALU**）操作（包括基于内存的操作），尽管大多数英特尔 CPU 都是 32 位或 64 位。并非所有的 CPU 架构都提供相同级别的细粒度。

为了更好地解释这一点，让我们看一个涉及`short int`的例子：

```cpp
#include <iostream>

int main(void)
{
    short int s = 42;

    std::cout << s << '\n';
    s++;
    std::cout << s << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// 42
// 43
```

在前面的例子中，我们取一个`short int`，将其设置为值`42`，使用`std::cout`将这个值输出到`stdout`，然后将`short int`增加`1`，再次使用`std::cout`将结果输出到`stdout`。这是一个简单的例子，但在底层，发生了很多事情。在这种情况下，一个 2 字节的值，在包含 8 字节寄存器的系统上执行（即 64 位），必须初始化为`42`，存储在内存中，递增，然后再次存储在内存中以输出到`stdout`。所有这些操作都必须涉及 CPU 寄存器来执行这些操作。

在基于英特尔的 CPU 上（32 位或 64 位），这些操作可能涉及使用 CPU 寄存器的 2 字节版本。具体来说，英特尔的 CPU 可能是 32 位或 64 位，但它们提供的寄存器大小为 1、2、4 和 8 字节（特别是在 64 位 CPU 上）。在前面的例子中，这意味着 CPU 加载一个 2 字节寄存器，存储这个值到内存（使用 2 字节的内存操作），将这个 2 字节寄存器增加 1，然后再次将这个 2 字节寄存器存储回内存中。

在**精简指令集计算机**（**RISC**）上，这个相同的操作可能会更加复杂，因为 2 字节寄存器不存在。要加载、存储、递增和再次存储只有 2 字节的数据将需要使用额外的指令。具体来说，在 32 位 CPU 上，必须将 32 位值加载到寄存器中，当这个值存储在内存中时，必须保存和恢复上 32 位（或下 32 位，取决于对齐）以确保实际上只影响了 2 字节的内存。如果进行了大量的操作，额外的对齐检查，即内存读取、掩码和存储，将导致显著的性能影响。

因此，C 和 C++提供了默认的`int`类型，通常表示 CPU 寄存器。也就是说，如果架构是 32 位，那么`int`就是 32 位，反之亦然（64 位除外，稍后将解释）。应该注意的是，像英特尔这样的 CISC 架构可以自由地以比 CPU 寄存器大小更小的粒度实现 ALU 操作，这意味着在底层，仍然可能进行相同的对齐检查和掩码操作。重点是，除非你有非常特定的原因要使用`short int`（对此有一些原因；我们将在本章末讨论这个话题），而不是`int`，在大多数情况下，使用更小的类型，即使你不需要完整的 4 或 8 字节，仍然更有效率。

让我们看一下`int`类型：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes_signed = sizeof(signed int);
    auto min_signed = std::numeric_limits<signed int>().min();
    auto max_signed = std::numeric_limits<signed int>().max();

    auto num_bytes_unsigned = sizeof(unsigned int);
    auto min_unsigned = std::numeric_limits<unsigned int>().min();
    auto max_unsigned = std::numeric_limits<unsigned int>().max();

    std::cout << "num bytes (signed): " << num_bytes_signed << '\n';
    std::cout << "min value (signed): " << min_signed << '\n';
    std::cout << "max value (signed): " << max_signed << '\n';

    std::cout << '\n';

    std::cout << "num bytes (unsigned): " << num_bytes_unsigned << '\n';
    std::cout << "min value (unsigned): " << min_unsigned << '\n';
    std::cout << "max value (unsigned): " << max_unsigned << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes (signed): 4
// min value (signed): -2147483648
// max value (signed): 2147483647

// num bytes (unsigned): 4
// min value (unsigned): 0
// max value (unsigned): 4294967295
```

在前面的例子中，`int`在 64 位英特尔 CPU 上显示为 4 字节。这是因为向后兼容性，这意味着在一些 RISC 架构上，默认的寄存器大小，导致最有效的处理，可能不是`int`，而是`long int`。问题在于实时确定这一点是痛苦的（因为使用的指令是在编译时完成的）。让我们看一下`long int`来进一步解释这一点：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes_signed = sizeof(signed long int);
    auto min_signed = std::numeric_limits<signed long int>().min();
    auto max_signed = std::numeric_limits<signed long int>().max();

    auto num_bytes_unsigned = sizeof(unsigned long int);
    auto min_unsigned = std::numeric_limits<unsigned long int>().min();
    auto max_unsigned = std::numeric_limits<unsigned long int>().max();

    std::cout << "num bytes (signed): " << num_bytes_signed << '\n';
    std::cout << "min value (signed): " << min_signed << '\n';
    std::cout << "max value (signed): " << max_signed << '\n';

    std::cout << '\n';

    std::cout << "num bytes (unsigned): " << num_bytes_unsigned << '\n';
    std::cout << "min value (unsigned): " << min_unsigned << '\n';
    std::cout << "max value (unsigned): " << max_unsigned << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes (signed): 8
// min value (signed): -9223372036854775808
// max value (signed): 9223372036854775807

// num bytes (unsigned): 8
// min value (unsigned): 0
// max value (unsigned): 18446744073709551615
```

如前面的代码所示，在运行 Ubuntu 的 64 位英特尔 CPU 上，`long int`是一个 8 字节的值。这在 Windows 上并不成立，它将`long int`表示为 32 位，而`long long int`为 64 位（再次是为了向后兼容）。

在系统编程中，您正在处理的数据大小通常非常重要，正如本节所示，除非您确切知道应用程序将在哪种 CPU、操作系统和模式上运行，否则几乎不可能知道在使用 C 和 C++提供的默认类型时您的整数类型的大小。大多数这些类型在系统编程中不应该使用，除了`int`之外，它几乎总是表示与 CPU 寄存器相同位宽的数据类型，或者至少是一个不需要额外对齐检查和掩码来执行简单算术操作的数据类型。在下一节中，我们将讨论克服这些大小问题的其他类型，并讨论它们的优缺点。

# 浮点数

在系统编程中，浮点数很少使用，但我们在这里简要讨论一下以供参考。浮点数通过减少精度来增加可以存储的值的大小。例如，使用浮点数可以存储代表`1.79769e+308`的数字，这是使用整数值甚至`long long int`都不可能实现的。然而，无法将这个值减去`1`并看到数字值的差异，浮点数也无法在保持与整数值相同的粒度的同时表示如此大的值。浮点数的另一个好处是它们能够表示次整数数值，在处理更复杂的数学计算时非常有用（这在系统编程中很少需要，因为大多数内核不使用浮点数来防止内核中发生浮点错误，最终导致没有接受浮点值的系统调用）。

主要有三种不同类型的浮点数——`float`、`double`和`long double`。例如，考虑以下代码：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes = sizeof(float);
    auto min = std::numeric_limits<float>().min();
    auto max = std::numeric_limits<float>().max();

    std::cout << "num bytes: " << num_bytes << '\n';
    std::cout << "min value: " << min << '\n';
    std::cout << "max value: " << max << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes: 4
// min value: 1.17549e-38
// max value: 3.40282e+38
```

在前面的例子中，我们利用`std::numeric_limits`来检查`float`类型，在英特尔 64 位 CPU 上是 4 字节大小。`double`如下：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes = sizeof(double);
    auto min = std::numeric_limits<double>().min();
    auto max = std::numeric_limits<double>().max();

    std::cout << "num bytes: " << num_bytes << '\n';
    std::cout << "min value: " << min << '\n';
    std::cout << "max value: " << max << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes: 8
// min value: 2.22507e-308
// max value: 1.79769e+308
```

对于`long double`，代码如下：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes = sizeof(long double);
    auto min = std::numeric_limits<long double>().min();
    auto max = std::numeric_limits<long double>().max();

    std::cout << "num bytes: " << num_bytes << '\n';
    std::cout << "min value: " << min << '\n';
    std::cout << "max value: " << max << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes: 16
// min value: 3.3621e-4932
// max value: 1.18973e+4932
```

如前面的代码所示，在英特尔 64 位 CPU 上，`long double`是 16 字节大小（或 128 位），可以存储绝对庞大的数字。

# 布尔值

标准 C 语言没有本地定义布尔类型。然而，C++有，并使用`bool`关键字定义。在 C 中，布尔值可以用任何整数类型表示，通常`false`表示`0`，`true`表示`1`。有趣的是，一些 CPU 能够比较寄存器或内存位置与`0`更快，这意味着在某些 CPU 上，布尔算术和分支实际上更快地导致*典型*情况下的`false`。

让我们看一下使用以下代码的`bool`：

```cpp
#include <iostream>

int main(void)
{
    auto num_bytes = sizeof(bool);
    auto min = std::numeric_limits<bool>().min();
    auto max = std::numeric_limits<bool>().max();

    std::cout << "num bytes: " << num_bytes << '\n';
    std::cout << "min value: " << min << '\n';
    std::cout << "max value: " << max << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// num bytes: 1
// min value: 0
// max value: 1
```

在前面的代码中，使用 C++在 64 位英特尔 CPU 上的布尔值大小为 1 字节，可以取值为`0`或`1`。值得注意的是，出于相同的原因，布尔值可以是 32 位或者 64 位，取决于 CPU 架构。在英特尔 CPU 上，支持 8 位寄存器大小（即 1 字节），布尔值只需要 1 字节大小。

布尔值的总大小很重要，特别是在磁盘上存储布尔值时。从技术上讲，布尔值只需要一个位来存储其值，但很少（如果有的话）CPU 架构支持位式寄存器和内存访问，这意味着布尔值通常占用多于一个位，有些情况下甚至可能占用多达 64 位。如果您的结果文件的大小很重要，使用内置的布尔类型存储布尔值可能不是首选（最终需要位掩码）。

# 学习标准整数类型

为了解决 C 和 C++提供的默认类型的不确定性，它们都提供了标准整数类型，可以从`stdint.h`头文件中访问。此头文件定义了以下类型：

+   `int8_t`，`uint8_t`

+   `int16_t`，`uint16_t`

+   `int32_t`，`uint32_t`

+   `int64_t`，`uint64_t`

此外，`stdint.h`提供了上述类型的*最小*和*最快*版本，以及最大类型和整数指针类型，这些都超出了本书的范围。前面的类型正是您所期望的；它们定义了具有特定位数的整数类型的宽度。例如，`int8_t`是一个有符号的 8 位整数。无论 CPU 架构、操作系统或模式如何，这些类型始终相同（唯一未定义的是它们的字节顺序，通常仅在处理网络和外部设备时才需要）。

一般来说，如果您正在处理的数据类型的大小很重要，应使用标准整数类型，而不是语言提供的默认类型。尽管标准类型确实解决了许多已经确定的问题，但它们也有自己的问题。具体来说，`stdint.h`是一个由编译器提供的头文件，对于可能的每个 CPU 架构和操作系统组合，都定义了不同的头文件。此文件中定义的类型通常在底层使用默认类型表示。这是因为编译器知道`int32_t`是`int`还是`long int`。为了证明这一点，让我们创建一个能够比较整数类型的应用程序。

我们将从以下头文件开始：

```cpp
#include <typeinfo>
#include <iostream>

#include <string>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
```

`typeinfo`头文件将为我们提供 C++支持的类型信息，最终为我们提供特定整数类型的根类型。问题在于`typeinfo`为我们提供了这些类型信息的编码版本。为了解码这些信息，我们需要`cxxabi.h`头文件，它提供了对 C++本身内置的解码器的访问：

```cpp
template<typename T>
std::string type_name()
{
    int status;
    std::string name = typeid(T).name();

    auto demangled_name =
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);

    if (status == 0) {
        name = demangled_name;
        std::free(demangled_name);
    }

    return name;
}
```

前一个函数返回提供的类型`T`的根名称。首先从 C++中获取类型的名称，然后使用解码器将编码的类型信息转换为人类可读的形式。最后，返回结果名称：

```cpp
template<typename T1, typename T2>
void
are_equal()
{
    #define red "\0331;31m"
    #define reset "\033[0m"

    std::cout << type_name<T1>() << " vs "
              << type_name<T2>() << '\n';

    if (sizeof(T1) == sizeof(T2)) {
        std::cout << " - size: both == " << sizeof(T1) << '\n';
    }
    else {
        std::cout << red " - size: "
                  << sizeof(T1)
                  << " != "
                  << sizeof(T2)
                  << reset "\n";
    }

    if (type_name<T1>() == type_name<T2>()) {
        std::cout << " - name: both == " << type_name<T1>() << '\n';
    }
    else {
        std::cout << red " - name: "
                  << type_name<T1>()
                  << " != "
                  << type_name<T2>()
                  << reset "\n";
    }
}
```

前一个函数检查类型的名称和大小是否相同，因为它们不需要相同（例如，大小可能相同，但类型的根可能不同）。应该注意的是，我们向此函数的输出（输出到`stdout`）添加了一些奇怪的字符。这些奇怪的字符告诉控制台在找不到匹配项时以红色输出，提供了一种简单的方法来查看哪些类型是相同的，哪些类型是不同的：

```cpp
int main()
{
    are_equal<uint8_t, int8_t>();
    are_equal<uint8_t, uint32_t>();

    are_equal<signed char, int8_t>();
    are_equal<unsigned char, uint8_t>();

    are_equal<signed short int, int16_t>();
    are_equal<unsigned short int, uint16_t>();
    are_equal<signed int, int32_t>();
    are_equal<unsigned int, uint32_t>();
    are_equal<signed long int, int64_t>();
    are_equal<unsigned long int, uint64_t>();
    are_equal<signed long long int, int64_t>();
    are_equal<unsigned long long int, uint64_t>();
}
```

最后，我们将比较每种标准整数类型与预期（更恰当地说是*典型*）默认类型，以查看在任何给定架构上这些类型是否实际相同。可以在任何架构上运行此示例，以查看默认类型和标准整数类型之间的差异，以便在需要系统编程时查找不一致之处。

对于在 Ubuntu 上的基于英特尔 64 位 CPU 的`uint8_t`，结果如下：

```cpp
are_equal<uint8_t, int8_t>();
are_equal<uint8_t, uint32_t>();

// unsigned char vs signed char
// - size: both == 1
// - name: unsigned char != signed char

// unsigned char vs unsigned int
// - size: 1 != 4
// - name: unsigned char != unsigned int
```

以下显示了`char`的结果：

```cpp

are_equal<signed char, int8_t>();
are_equal<unsigned char, uint8_t>();

// signed char vs signed char
// - size: both == 1
// - name: both == signed char

// unsigned char vs unsigned char
// - size: both == 1
// - name: both == unsigned char
```

最后，以下代码显示了剩余的`int`类型的结果：

```cpp
are_equal<signed short int, int16_t>();
are_equal<unsigned short int, uint16_t>();
are_equal<signed int, int32_t>();
are_equal<unsigned int, uint32_t>();
are_equal<signed long int, int64_t>();
are_equal<unsigned long int, uint64_t>();
are_equal<signed long long int, int64_t>();
are_equal<unsigned long long int, uint64_t>();

// short vs short
// - size: both == 2
// - name: both == short

// unsigned short vs unsigned short
// - size: both == 2
// - name: both == unsigned short

// int vs int
// - size: both == 4
// - name: both == int

// unsigned int vs unsigned int
// - size: both == 4
// - name: both == unsigned int

// long vs long
// - size: both == 8
// - name: both == long

// unsigned long vs unsigned long
// - size: both == 8
// - name: both == unsigned long

// long long vs long
// - size: both == 8
// - name: long long != long

// unsigned long long vs unsigned long
// - size: both == 8
// - name: unsigned long long != unsigned long
```

所有类型都相同，但有一些显著的例外：

+   前两个测试是特意提供的，以确保实际上会检测到错误。

+   在 Ubuntu 上，`int64_t`是使用`long`实现的，而不是`long long`，这意味着在 Ubuntu 上，`long`和`long long`是相同的。但在 Windows 上不是这样。

这个演示最重要的是要认识到输出中不包括标准整数类型名称，而只包含默认类型名称。这是因为，如前所示，编译器在 Ubuntu 上的 Intel 64 位 CPU 上使用`int`实现`int32_t`，对编译器来说，这些类型是一样的。不同之处在于，在另一个 CPU 架构和操作系统上，`int32_t`可能是使用`long int`实现的。

如果您关心整数类型的大小，请使用标准整数类型，并让头文件为您选择默认类型。如果您不关心整数类型的大小，或者 API 规定了类型，请使用默认类型。在下一节中，我们将向您展示，即使标准整数类型也不能保证特定大小，并且刚刚描述的规则在使用常见的系统编程模式时可能会出现问题。

# 结构打包

标准整数提供了一个编译器支持的方法，用于在编译时指定整数类型的大小。具体来说，它们将位宽映射到默认类型，这样编码人员就不必手动执行此操作。然而，标准类型并不总是保证类型的宽度，结构是一个很好的例子。为了更好地理解这个问题，让我们看一个简单的结构示例：

```cpp
#include <iostream>

struct mystruct {
    uint64_t data1;
    uint64_t data2;
};

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 16
```

在前面的例子中，我们创建了一个包含两个 64 位整数的结构。然后，使用`sizeof()`函数，我们输出了结构的大小到`stdout`，使用`std::cout`。如预期的那样，结构的总大小，以字节为单位，是`16`。值得注意的是，和本书的其余部分一样，本节中的例子都是在 64 位 Intel CPU 上执行的。

现在，让我们看看相同的例子，但其中一个数据类型被更改为 16 位整数，而不是 64 位整数，如下所示：

```cpp
#include <iostream>

struct mystruct {
    uint64_t data1;
    uint16_t data2;
};

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 16
```

在前面的例子中，我们有一个结构，其中有两种数据类型，但它们不匹配。然后，我们使用`std::cout`输出数据结构的大小到`stdout`，报告的大小是 16 字节。问题在于，我们期望是 10 字节，因为我们将结构定义为 64 位（8 字节）和 16 位（2 字节）整数的组合。

在幕后，编译器正在用 64 位整数替换 16 位整数。这是因为 C 和 C++的基本类型是`int`，编译器允许将小于`int`的类型更改为`int`，即使我们明确声明第二个整数为 16 位整数。换句话说，使用`unit16_t`并不要求使用 16 位整数，而是在 64 位 Intel CPU 上的 Ubuntu 上是`short int`的`typedef`，根据 C 和 C++规范，编译器可以随意将`short int`更改为`int`。

我们指定整数的顺序也不重要：

```cpp
#include <iostream>

struct mystruct {
    uint16_t data1;
    uint64_t data2;
};

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 16
```

如前面的例子所示，编译器再次声明结构的总大小为 16 字节，而我们期望是 10。在这个例子中，编译器更有可能进行这种类型的替换，因为它能够识别到存在对齐问题。具体来说，这段代码编译的 CPU 是 64 位 CPU，这意味着用`unit64_t`替换`uint16_t`可能会改善内存缓存，并且将`data2`对齐到 64 位边界，而不是 16 位边界，如果结构在内存中正确对齐，它将跨越两个 64 位内存位置。

结构并不是唯一可以重现这种类型替换的方法。让我们来看看以下例子：

```cpp
#include <iostream>

int main()
{
    int16_t s = 42;
    auto result = s + 42;
    std::cout << "size: " << sizeof(result) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 4
```

在前面的例子中，我们创建了一个 16 位整数，并将其设置为`42`。然后我们创建了另一个整数，并将其设置为我们的 16 位整数加上`42`。值`42`可以表示为 8 位整数，但实际上并没有。相反，编译器将`42`表示为`int`，在这种情况下，这意味着这段代码编译的系统大小为 4 字节。

编译器将`42`表示为`int`，加上`int16_t`，结果为`int`，因为这是更高宽度类型。在前面的例子中，我们使用`auto`定义了`result`变量，这确保了结果类型反映了编译器由于这种算术操作而创建的类型。我们也可以将`result`定义为另一个`int16_t`，这样也可以工作，除非我们打开整数类型转换警告。这样做会导致一个转换警告，因为编译器构造了一个`int`，作为加上`s`加上`42`的结果，然后必须自动将结果的`int`转换回`int16_t`，这将执行一个缩小转换，可能导致溢出（因此会有警告）。

所有这些问题都是编译器能够执行类型转换的结果，从较小宽度类型转换为更高宽度类型，以优化性能，减少溢出的可能性。在这种情况下，一个数字值总是一个`int`，除非该值需要更多的存储空间（例如，用`0xFFFFFFFF00000000`替换`42`）。

这种类型的转换并不总是保证的。考虑以下例子：

```cpp
#include <iostream>

struct mystruct {
    uint16_t data1;
    uint16_t data2;
};

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 4
```

在前面的例子中，我们有一个包含两个 16 位整数的结构。结构的总大小报告为 4 字节，这正是我们所期望的。在这种情况下，编译器并没有看到改变整数大小的好处，因此保持了它们不变。

位域也不会改变编译器执行这种类型转换的能力，如下例所示：

```cpp
#include <iostream>

struct mystruct {
    uint16_t data1 : 2, data2 : 14;
    uint64_t data3;
};

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 16
```

在前面的例子中，我们创建了一个包含两个整数（一个 16 位整数和一个 64 位整数）的结构，但我们不仅定义了 16 位整数，还定义了位域，使我们可以直接访问整数中的特定位（这种做法在系统编程中应该避免，即将要解释的原因）。定义这些位域并不能阻止编译器将第一个整数的总大小从 16 位改为 64 位。

前面例子的问题在于，位域经常是系统程序员在直接与硬件接口时使用的一种模式。在前面的例子中，第二个 64 位整数预计应该距离结构顶部 2 字节。然而，在这种情况下，第二个 64 位整数实际上距离结构顶部 8 字节。如果我们使用这个结构直接与硬件接口，将会导致一个难以发现的逻辑错误。

克服这个问题的方法是对结构进行打包。以下例子演示了如何做到这一点：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct mystruct {
    uint64_t data1;
    uint16_t data2;
};
#pragma pack(pop)

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 10
```

前面的例子类似于本节中的第一个例子。创建了一个包含 64 位整数和 16 位整数的结构。在前面的例子中，结构的大小为 16 字节，因为编译器用 64 位整数替换了 16 位整数。为了解决这个问题，在前面的例子中，我们用`#pragma pack`和`#pragma pop`宏包装了结构。这些宏告诉编译器（因为我们向宏传递了`1`，表示一个字节）使用字节粒度对结构进行打包，告诉编译器不允许进行替换优化。

使用这种方法，将变量的顺序更改为编译器尝试执行这种类型优化的更可能情况，仍然会导致结构不被转换，如下例所示：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct mystruct {
    uint16_t data1;
    uint64_t data2;
};
#pragma pack(pop)

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 10
```

如前面的例子所示，结构的大小仍然是 10 字节，无论整数的顺序如何。

将结构打包与标准整数类型结合使用足以（假设字节顺序不是问题）直接与硬件进行接口，但是这种模式仍然不鼓励，而是更倾向于构建访问器和利用位掩码，为用户提供一种方式来确保以受控的方式进行直接访问硬件寄存器，而不受编译器的干扰，或者优化产生不希望的结果。

为了解释为什么应该避免打包结构和位字段，让我们看一个与对齐问题相关的例子：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct mystruct {
    uint16_t data1;
    uint64_t data2;
};
#pragma pack(pop)

int main()
{
    mystruct s;
    std::cout << "addr: " << &s << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// addr: 0x7fffd11069cf
```

在上一个例子中，我们创建了一个包含 16 位整数和 64 位整数的结构，然后对结构进行了打包，以确保结构的总大小为 10 字节，并且每个数据字段都正确对齐。然而，结构的总对齐方式并不是缓存对齐，这在上一个例子中得到了证明，方法是在堆栈上创建结构的一个实例，然后使用`std::cout`将结构的地址输出到`stdout`。如图所示，地址是字节对齐的，而不是缓存对齐的。

为了对结构进行缓存对齐，我们将利用`alignas()`函数，这将在[第七章中进行解释，*内存管理的全面视图*：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct alignas(16) mystruct {
    uint16_t data1;
    uint64_t data2;
};
#pragma pack(pop)

int main()
{
    mystruct s;
    std::cout << "addr: " << &s << '\n';
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// addr: 0x7fff44ee3f40
// size: 16
```

在上一个例子中，我们在结构的定义中添加了`alignas()`函数，它在堆栈上对结构进行了缓存对齐。我们还输出了结构的总大小，就像以前的例子一样，如图所示，结构不再是紧凑的。换句话说，使用`#pragma pack#`并不能保证结构实际上会被打包。在所有情况下，编译器都可以根据需要进行更改，即使`#pragma pack`宏也只是一个提示，而不是要求。

在前面的情况下，应该注意编译器实际上在结构的末尾添加了额外的内存，这意味着结构中的数据成员仍然在它们的正确位置，如下所示：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct alignas(16) mystruct {
    uint16_t data1;
    uint64_t data2;
};
#pragma pack(pop)

int main()
{
    mystruct s;
    std::cout << "addr data1: " << &s.data1 << '\n';
    std::cout << "addr data2: " << &s.data2 << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// addr data1: 0x7ffc45dd8c90
// addr data2: 0x7ffc45dd8c92
```

在上一个例子中，每个数据成员的地址都输出到`stdout`，并且如预期的那样，第一个数据成员对齐到`0`，第二个数据成员距离结构顶部 2 字节，即使结构的总大小是 16 字节，这意味着编译器通过在结构底部添加额外的整数来获得额外的 6 字节。虽然这可能看起来无害，如果创建了这些结构的数组，并且假定由于使用了`#pragma pack`，结构的大小为 10 字节，那么将引入一个难以发现的逻辑错误。

为了结束本章，应该提供一个关于指针大小的注释。具体来说，指针的大小完全取决于 CPU 架构、操作系统和应用程序运行的模式。让我们来看下面的例子：

```cpp
#include <iostream>

#pragma pack(push, 1)
struct mystruct {
    uint16_t *data1;
    uint64_t data2;
};
#pragma pack(pop)

int main()
{
    std::cout << "size: " << sizeof(mystruct) << '\n';
}

// > g++ scratchpad.cpp; ./a.out
// size: 16
```

在上一个例子中，我们存储了一个指针和一个整数，并使用`std::cout`将结构的总大小输出到`stdout`。在运行 Ubuntu 的 64 位英特尔 CPU 上，这个结构的总大小是 16 字节。在运行 Ubuntu 的 32 位英特尔 CPU 上，这个结构的总大小将是 12 字节，因为指针只有 4 字节大小。更糟糕的是，如果应用程序被编译为 32 位应用程序，但在 64 位内核上执行，应用程序将看到这个结构为 12 字节，而内核将看到这个结构为 16 字节。尝试将这个结构传递给内核将导致错误，因为应用程序和内核会以不同的方式看待这个结构。

# 总结

在本章中，我们回顾了 C 和 C++为系统编程提供的不同整数类型（并简要回顾了浮点类型）。我们从讨论 C 和 C++提供的默认类型以及与这些类型相关的利弊开始，包括常见的`int`类型，解释了它是什么以及如何使用它。接下来，我们讨论了由`stdint.h`提供的标准整数类型以及它们如何解决默认类型的一些问题。最后，我们结束了本章，讨论了结构打包以及编译器在不同情况下可以进行的类型转换和优化的问题。

在下一章中，我们将介绍 C++17 所做的更改，一种 C++特定的技术称为**资源获取即初始化**（**RAII**），并概述**指导支持库**（**GSL**）。

# 问题

1.  `short int`和`int`之间有什么区别？

1.  `int`的大小是多少？

1.  `signed int`和`unsigned int`的大小不同吗？

1.  `int32_t`和`int`之间有什么区别？

1.  `int16_t`保证是 16 位吗？

1.  `#pragma pack`是做什么的？

1.  是否可能保证在所有情况下进行结构打包？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
