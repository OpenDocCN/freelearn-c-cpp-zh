# 使用不同的架构

桌面应用程序的开发人员通常很少关注硬件架构。首先，他们经常使用高级编程语言，隐藏这些复杂性，以牺牲性能为代价。其次，在大多数情况下，他们的代码在 x86 架构上运行，并且他们经常认为其功能是理所当然的。例如，他们可能假设`int`的大小为 32 位，但在许多情况下这是不正确的。

嵌入式开发人员处理更广泛的架构。即使他们不使用与目标平台本地的汇编语言编写代码，他们也应该知道所有 C 和 C++基本类型都是依赖于架构的；标准只保证 int 至少为 16 位。他们还应该了解特定架构的特性，如字节顺序和对齐，并考虑到在其他架构上执行浮点或 64 位数字的操作，这在 x86 架构上相对便宜，但在其他架构上可能更昂贵。

由于他们的目标是从嵌入式硬件中实现最大可能的性能，他们应该了解如何组织内存中的数据，以最有效地利用 CPU 缓存和操作系统分页机制。

在本章中，我们将涵盖以下主题：

+   探索固定宽度整数类型

+   使用`size_t`类型

+   检测平台的字节顺序

+   转换字节顺序

+   处理数据对齐

+   使用紧凑结构

+   使用缓存行对齐数据

通过研究这些主题，我们将学习如何调整我们的代码以针对平台实现最大性能和可移植性。

# 探索固定宽度整数类型

C 和 C++开发人员经常忘记基本数据类型如 char、short 和 int 的大小是依赖于架构的。与此同时，大多数硬件外设定义了关于用于数据交换的字段大小的特定要求。为了使代码与外部硬件或通信协议一起工作具有可移植性，嵌入式开发人员使用固定大小的整数类型，明确指定数据字段的大小。

一些最常用的数据类型如下：

| **宽度** | **有符号** | **无符号** |
| --- | --- | --- |
| 8 位 | `int8_t` | `uint8_t` |
| 16 位 | `int16_t` | `uint16_t` |
| 32 位 | `int32_t` | `uint32_t` |

指针大小也取决于架构。开发人员经常需要处理数组的元素，由于数组在内部表示为指针，偏移表示取决于指针的大小。`size_t`是一种特殊的数据类型，因为它以与架构无关的方式表示偏移和数据大小。

在本教程中，我们将学习如何在代码中使用固定大小的数据类型，使其在不同架构之间可移植。这样，我们可以使我们的应用程序更快地在其他目标平台上运行，并减少代码修改。

# 如何做到...

我们将创建一个模拟与外围设备进行数据交换的应用程序。按照以下步骤操作：

1.  在您的工作目录中，即`~/test`，创建一个名为`fixed_types`的子目录。

1.  使用您喜欢的文本编辑器在`fixed_types`子目录中创建名为`fixed_types.cpp`的文件。将以下代码片段复制到`fixed_types.cpp`文件中：

```cpp
#include <iostream>

void SendDataToDevice(void* buffer, uint32_t size) {
  // This is a stub function to send data pointer by
  // buffer.
  std::cout << "Sending data chunk of size " << size << std::endl;
}

int main() {
  char buffer[] = "Hello, world!";
  uint32_t size = sizeof(buffer);
  SendDataToDevice(&size, sizeof(size));
  SendDataToDevice(buffer, size);
  return 0;
}
```

1.  在 loop 子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(fixed_types)
add_executable(fixed_types fixed_types.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的步骤*设置环境*来完成。

1.  切换到目标系统的终端。如有需要，请使用您的用户凭据登录。

1.  运行二进制文件以查看其工作原理。

# 工作原理...

当您运行二进制文件时，您将看到以下输出：

![](img/b2a222c6-a006-4c01-ad57-fa2463d6e63f.png)

在这个简单的程序中，我们正在模拟与外部设备的通信。由于我们没有真正的设备，`SendDataToDevice`函数只是打印它应该发送到目标设备的数据的大小。

假设设备可以处理可变大小的数据块。每个数据块都以其大小作为前缀，并编码为 32 位无符号整数。可以描述如下：

| **大小** | **有效载荷** |
| --- | --- |
| 0-4 字节 | 5 - N 字节，其中 N 是大小 |

在我们的代码中，我们将`size`声明为`uint32_t`：

```cpp
  uint32_t size = sizeof(buffer);
```

这意味着它将在每个平台上都占用 32 位 - 16 位、32 位或 64 位。

现在，我们将大小发送到设备：

```cpp
  SendDataToDevice(&size, sizeof(size));
```

`SendDataToDevice`不会发送实际数据；相反，它会报告要发送的数据大小。正如我们所看到的，大小为`4`字节，正如预期的那样：

```cpp
  Sending data chunk of size 4
```

假设我们声明`int`数据类型，如下所示：

```cpp
  int size = sizeof(buffer);
```

在这种情况下，这段代码只能在 32 位和 64 位系统上工作，并且在 16 位系统上悄悄地产生不正确的结果，因为`sizeof(int)`在这里是 16。

# 还有更多...

我们在这个示例中实现的代码并不是完全可移植的，因为它没有考虑 32 位字中字节的顺序。这个顺序被称为**字节序**，它的影响将在本章后面讨论。

# 使用`size_t`类型

指针大小也取决于体系结构。开发人员经常需要处理数组的元素，由于数组在内部表示为指针，偏移量表示取决于指针的大小。

例如，在 32 位系统中，指针是 32 位，与`int`相同。然而，在 64 位系统中，`int`的大小仍然是 32 位，而指针是 64 位。

`size_t`是一种特殊的数据类型，因为它以与体系结构无关的方式表示偏移量和数据大小。

在这个示例中，我们将学习如何在处理数组时使用`size_t`。

# 如何做...

我们将创建一个处理可变大小数据缓冲区的应用程序。如果需要，我们需要能够访问目标平台提供的任何内存地址。按照以下步骤操作：

1.  在您的工作目录，即`~/test`，创建一个名为`sizet`的子目录。

1.  使用您喜欢的文本编辑器在`sizet`子目录中创建一个名为`sizet.cpp`的文件。将以下代码片段复制到`sizet.cpp`文件中：

```cpp
#include <iostream>

void StoreData(const char* buffer, size_t size) {
  std::cout << "Store " << size << " bytes of data" << std::endl;
}

int main() {
  char data[] = "Hello,\x1b\a\x03world!";
  const char *buffer = data;
  std::cout << "Size of buffer pointer is " << sizeof(buffer) << std::endl;
  std::cout << "Size of int is " << sizeof(int) << std::endl;
  std::cout << "Size of size_t is " << sizeof(size_t) << std::endl;
  StoreData(data, sizeof(data));
  return 0;
}
```

1.  在子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(sizet)
add_executable(sizet sizet.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的示例，*设置环境*来完成。

1.  切换到目标系统的终端。根据需要使用您的用户凭据登录。

1.  运行`sizet`应用程序可执行文件。

# 它是如何工作的...

在这个示例中，我们正在模拟一个将任意数据存储在文件或数据库中的函数。该函数接受数据指针和数据大小。但是我们应该使用什么类型来表示大小？如果我们在 64 位系统中使用无符号整数，我们就会人为地限制我们的函数处理的数据最多只能达到 4GB。

为了避免这种限制，我们使用`size_t`作为`size`的数据类型：

```cpp
void StoreData(const char* buffer, size_t size) {
```

大多数标准库 API 接受索引和大小参数，也处理`size_t`参数。例如，`memcpy` C 函数，它将数据块从源缓冲区复制到目标缓冲区，声明如下：

```cpp
void *memset(void *b, int c, size_t len);
```

运行上述代码会产生以下输出：

![](img/79d7eb07-66fc-465b-b089-36506a38e18d.png)

正如我们所看到的，在目标系统上指针的大小是 64 位，尽管`int`的大小是 32 位。在我们的程序中使用`size_t`允许它使用嵌入式板的所有内存。

# 还有更多...

C++标准定义了一个`std::size_t`类型。它与普通的 C `size_t`相同，只是它是在`std`命名空间中定义的。在你的 C++代码中使用`std::size_t`是更可取的，因为它是标准的一部分，但`std::size_t`和`size_t`都是可以互换的。

# 检测平台的字节顺序

字节顺序定义了表示大数值的字节在内存中存储的顺序。

有两种字节顺序：

+   **大端**：最重要的字节被先存储。一个 32 位的值，*0x01020304*，被存储在`ptr`地址上，如下所示：

| **内存偏移（字节）** | **值** |
| --- | --- |
| ptr | 0x01 |
| ptr + 1 | 0x02 |
| ptr + 2 | ox03 |
| ptr + 3 | 0x04 |

大端架构的例子包括 AVR32 和 Motorola 68000。

+   **小端**：最不重要的字节被先存储。一个 32 位的值，*0x01020304*，被存储在`ptr`地址上，如下所示：

| **内存偏移（字节）** | **值** |
| --- | --- |
| ptr | 0x04 |
| ptr + 1 | 0x03 |
| ptr + 2 | 0x02 |
| ptr + 3 | 0x01 |

x86 架构是小端的。

在与其他系统交换数据时，处理字节顺序尤为重要。如果开发人员将一个 32 位整数，比如 0x01020304，原样发送，如果接收者的字节顺序与发送者的字节顺序不匹配，它可能被读取为 0x04030201。这就是为什么数据应该被序列化的原因。

在这个配方中，我们将学习如何确定目标系统的字节顺序。

# 如何做...

我们将创建一个简单的程序，可以检测目标平台的字节顺序。按照以下步骤来做：

1.  在你的工作目录，即`~/test`，创建一个名为`endianness`的子目录。

1.  使用你喜欢的文本编辑器在循环子目录中创建一个名为`loop.cpp`的文件。将以下代码片段复制到`endianness.cpp`文件中：

```cpp
#include <iostream>

int main() {
  union {
    uint32_t i;
    uint8_t c[4];
  } data;
  data.i = 0x01020304;
  if (data.c[0] == 0x01) {
    std::cout << "Big-endian" << std::endl;
  } else {
    std::cout << "Little-endian" << std::endl;
  }
}
```

1.  在循环子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(endianness)
add_executable(endianness endianness.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的配方，*设置环境*，来完成这个过程。

1.  切换到目标系统的终端。如果需要，使用你的用户凭据登录。

1.  运行二进制文件。

# 它是如何工作的...

在这个配方中，我们利用了 C 语言的`union`函数的能力，将不同数据类型的表示映射到相同的内存空间。

我们定义了一个包含两个数据字段的联合体 - 一个 8 位整数数组和一个 32 位整数。这些数据字段共享相同的内存，因此对一个字段所做的更改会自动反映在另一个字段中：

```cpp
  union {
    uint32_t i;
    uint8_t c[4];
  } data
```

接下来，我们给 32 位整数字段赋予一个特别设计的值，其中每个字节都是事先知道的，并且与其他任何字节都不同。我们使用值为一、二、三和四的字节来组成目标值。

当值被赋给 32 位字段`i`时，它会自动将所有字段重写为`c`字节数组字段。现在，我们可以读取数组的第一个元素，并根据我们读取的内容推断硬件平台的字节顺序。

如果值为一，这意味着第一个字节包含最重要的字节，因此架构是大端的。否则，它是小端的。当我们运行二进制文件时，它会产生以下输出：

![](img/dca32ea5-56bc-4cb2-9515-285e8f26f7a2.png)

正如我们所看到的，该程序检测到我们的系统是小端的。这种技术可以用来检测我们运行时的字节顺序，并相应地调整应用程序逻辑。

# 还有更多...

如今，大多数广泛使用的平台，如 x86 和**Acorn RISC Machine**（**ARM**），都是小端的。然而，你的代码不应该隐式地假设系统的字节顺序。

如果需要在同一系统上运行的应用程序之间交换数据，可以安全地使用目标平台的字节序。但是，如果您的应用程序需要与其他系统交换数据，无论是通过网络协议还是常见数据存储，都应考虑将二进制数据转换为通用字节序。

基于文本的数据格式不会受到字节序的影响。使用 JSON 格式进行数据表示，这样可以实现平台无关和人类可读的数据表示。

**注意**：在目标嵌入式平台上进行二进制表示和反向转换可能会很昂贵。

# 转换字节序

虽然序列化库处理字节序，但有时开发人员可能希望自己实现轻量级通信协议的情况。

虽然 C++标准库没有提供序列化函数，但开发人员可以利用这样一个事实：在二进制网络协议中，字节顺序是被定义的，并且始终是大端序。

标准库提供了一组函数，可用于在当前平台（硬件）和大端序（网络）字节顺序之间进行转换：

+   `uint32_t` htonl (`uint32_t` value): 将`uint32_t`从硬件顺序转换为网络顺序

+   `uint32_t` ntohl (`uint32_t` value): 将`uint32_t`从网络顺序转换为硬件顺序

+   `uint16_t` htons (`uint16_t` value): 将`uint16_t`从硬件顺序转换为网络顺序

+   `uint16_t` ntohl (`uint16_t` value): 将`uint16_t`从网络顺序转换为硬件顺序

开发人员可以使用这些函数在不同平台上运行的应用程序之间交换二进制数据。

在这个示例中，我们将学习如何对字符串进行编码，以便在可能具有相同或不同字节序的两个系统之间进行交换。

# 如何操作...

在这个示例中，我们将创建两个应用程序：发送方和接收方。发送方将为接收方编写数据，以平台无关的方式对其进行编码。按照以下步骤进行操作：

1.  在您的工作目录中，即`~/test`，创建一个名为`enconv`的子目录。

1.  使用您喜欢的文本编辑器在`enconv`子目录中创建并编辑名为`sender.cpp`的文件。包括所需的头文件，如下所示：

```cpp
#include <stdexcept>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
```

1.  然后，定义一个将数据写入文件描述符的函数：

```cpp
void WriteData(int fd, const void* ptr, size_t size) {
  size_t offset =0;
  while (size) {
    const char *buffer = (const char*)ptr + offset;
    int written = write(fd, buffer, size);
    if (written < 0) {
      throw std::runtime_error("Can not write to file");
    }
    offset += written;
    size -= written;
  }
  }
```

1.  现在，我们需要定义一个格式化并写入消息的函数，以及调用它的主函数：

```cpp
void WriteMessage(int fd, const char* str) {
  uint32_t size = strlen(str);
  uint32_t encoded_size = htonl(size);
  WriteData(fd, &encoded_size, sizeof(encoded_size));
  WriteData(fd, str, size);
}

int main(int argc, char** argv) {
  int fd = open("envconv.data", 
                 O_WRONLY|O_APPEND|O_CREAT, 0666);
  for (int i = 1; i < argc; i++) {
    WriteMessage(fd, argv[i]);
  }
}
```

1.  类似地，创建一个名为`receiver.cpp`的文件，并包含相同的头文件：

```cpp
#include <stdexcept>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
```

1.  添加以下代码，从文件描述符中读取数据：

```cpp
void ReadData(int fd, void* ptr, size_t size) {
  size_t offset =0;
  while (size) {
    char *buffer = (char*)ptr + offset;
    int received = read(fd, buffer, size);
    if (received < 0) {
      throw std::runtime_error("Can not read from file");
    } else if (received == 0) {
      throw std::runtime_error("No more data");
    }
    offset += received;
    size -= received;
  }
  }
```

1.  现在，定义一个将消息读取出来的函数，以及调用它的主函数：

```cpp
std::string ReadMessage(int fd) {
  uint32_t encoded_size = 0;
  ReadData(fd, &encoded_size, sizeof(encoded_size));
  uint32_t size = ntohl(encoded_size);
  auto data = std::make_unique<char[]>(size);
  ReadData(fd, data.get(), size);
  return std::string(data.get(), size);
}

int main(void) {
  int fd = open("envconv.data", O_RDONLY, 0666);
  while(true) {
    try {
      auto s = ReadMessage(fd);
      std::cout << "Read: " << s << std::endl;
    } catch(const std::runtime_error& e) {
      std::cout << e.what() << std::endl;
      break;
    }
  }
 }
```

1.  在 loop 子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(conv)
add_executable(sender sender.cpp)
add_executable(receiver receiver.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

1.  构建应用程序并将生成的两个可执行二进制文件`sender`和`receiver`复制到目标系统。使用第二章中的设置环境的方法。

1.  切换到目标系统的终端。如果需要，使用您的用户凭据登录。

1.  运行`sender`二进制文件，并传递两个命令行参数：`Hello`和`Worlds`。这不会生成任何输出。

1.  然后，运行接收方。

1.  现在，检查用于数据交换的`sender`和`receiver`文件的内容。它将以二进制格式呈现，因此我们需要使用`xxd`工具将其转换为十六进制格式：

```cpp
$ xxd envconv.data 
0000000: 0000 0005 4865 6c6c 6f00 0000 0557 6f72 ....Hello....Wor
0000010: 6c64 ld
```

1.  文件包含两个字符串`hello`和`world`，前面是它们的大小。`size`字段总是以大端序存储，与体系结构无关。这允许发送方和接收方在具有不同字节序的两台不同计算机上运行。

# 它是如何工作的...

在这个示例中，我们创建了两个二进制文件，sender 和 receiver，模拟了两个主机之间的数据交换。我们不能对它们的字节序做出任何假设，这就是为什么数据交换格式必须是明确的原因。

发送方和接收方交换可变大小的数据块。我们将每个块编码为 4 字节的整数，以定义即将到来的块大小，然后是块内容。

当发送方不在屏幕上生成任何输出时，它会将编码的数据块保存在文件中。当我们运行接收方时，它能够读取、解码并显示发送方保存的任何信息，如下面的屏幕截图所示：

![](img/2a6bdbd6-7451-47f5-8926-0ceaa8b67dcc.png)

虽然我们在本地以平台格式保留块大小，但在发送时需要将其转换为统一表示。我们使用`htonl`函数来实现这一点：

```cpp
  uint32_t encoded_size = htonl(size);
```

此时，我们可以将编码后的大小写入输出流：

```cpp
  WriteData(fd, &encoded_size, sizeof(encoded_size));
```

块的内容如下：

```cpp
  WriteData(fd, str, size);
```

接收者反过来从输入流中读取大小：

```cpp
 uint32_t encoded_size = 0;
 ReadData(fd, &encoded_size, sizeof(encoded_size));
```

大小被编码，直到接收者使用`ntohl`函数将其转换为平台表示形式才能直接使用：

```cpp
 uint32_t size = ntohl(encoded_size);
```

只有在这样做之后，它才会知道接下来的块的大小，并且可以分配和读取它：

```cpp
 auto data = std::make_unique<char[]>(size);
 ReadData(fd, data.get(), size);
```

由于序列化的`data`大小始终表示为大端，读取函数不需要对数据写入的平台的字节顺序做出假设。它可以处理来自任何处理器架构的数据。

# 处理数据对齐

处理器不是按字节而是按内存字-与其数据地址大小匹配的块-读写数据。32 位处理器使用 32 位字，64 位处理器使用 64 位字，依此类推。

当字对齐时，读写效率最高-数据地址是字大小的倍数。例如，对于 32 位架构，地址 0x00000004 是对齐的，而 0x00000005 是不对齐的。在 x86 平台上，访问不对齐的数据比对齐的数据慢。然而，在 ARM 上，访问不对齐的数据会生成硬件异常并导致程序终止：

```cpp
Compilers align data automatically. When it comes to structures, the result may be surprising for developers who are not aware of alignment.
struct {
    uint8_t c;
    uint32_t i;
} a = {1, 1};

std::cout << sizeof(a) << std::endl;
```

前面的代码片段的输出是什么？`sizeof(uint8_t)`是 1，而`sizeof(uint32_t)`是 4。开发人员可能期望结构的大小是各个大小的总和；然而，结果高度取决于目标架构。

对于 x86，结果是`8`。在`i`之前添加一个`uint8_t`字段：

```cpp
struct {
    uint8_t c;
 uint8_t cc;
    uint32_t i;
} a = {1, 1};

std::cout << sizeof(a) << std::endl;
```

结果仍然是 8！编译器通过添加填充字节根据对齐规则优化结构内的数据字段的放置。这些规则依赖于架构，对于其他架构，结果可能不同。因此，结构不能在两个不同的系统之间直接交换，而需要*序列化*，这将在第八章中深入解释*通信和序列化*。

在这个示例中，我们将学习如何使用编译器隐式应用的规则来对齐数据以编写更节省内存的代码。

# 如何做...

我们将创建一个程序，该程序分配一个结构数组，并检查字段顺序如何影响内存消耗。按照以下步骤执行：

1.  在您的工作目录中，即`~/test`，创建一个名为`alignment`的子目录。

1.  使用您喜欢的文本编辑器在循环子目录中创建一个名为`alignment.cpp`的文件。添加所需的头文件并定义两种数据类型，即`Category`和`ObjectMetadata1`：

```cpp
#include <iostream>
enum class Category: uint8_t {
  file, directory, socket
};
struct ObjectMetadata1 {
  uint8_t access_flags;
  uint32_t size;
  uint32_t owner_id;
  Category category;
};

```

1.  现在，让我们定义另一个数据类型，称为`ObjectMetadata2`，以及使用所有这些的代码：

```cpp
struct ObjectMetadata2 {
  uint32_t size;
  uint32_t owner_id;
  uint8_t access_flags;
  Category category;
};

int main() {
  ObjectMetadata1 object_pool1[1000];
  ObjectMetadata2 object_pool2[1000];
  std::cout << "Poorly aligned:" << sizeof(object_pool1) << std::endl;
  std::cout << "Well aligned:" << sizeof(object_pool2) << std::endl;
  return 0;
}
```

1.  在循环子目录中创建一个名为`CMakeLists.txt`的文件，并添加以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(alignment)
add_executable(alignment alignment.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的配方*设置环境*来执行此操作。

1.  切换到目标系统的终端。如果需要，使用您的用户凭据登录。

1.  运行二进制文件。

# 它是如何工作的...

在我们的示例应用程序中，我们定义了两个数据结构，`ObjectMetadata1`和`ObjectMetadata2`，它们将保存有关文件对象的一些元数据。我们定义了代表对象的四个字段：

+   **访问标志**：代表文件访问类型的位的组合，例如读取、写入或执行。所有位字段都打包到一个单独的`uint8_t`字段中。

+   **大小**：作为 32 位无符号整数的对象大小。它将支持的对象大小限制为 4GB，但对于我们展示适当数据对齐的重要性来说是足够的。

+   **所有者 ID**：在我们系统中标识用户的 32 位整数。

+   **类别**：对象的类别。这可以是文件、目录或套接字。由于我们只定义了三个类别，`uint8_t`数据类型足以表示它们所有。这就是为什么我们使用`enum`类来声明它们的原因：

```cpp
enum class Category: uint8_t {
```

`ObjectMetadata1`和`ObjectMetadata2`都包含完全相同的字段；唯一的区别是它们在其结构中的排序方式。

现在，我们声明了两个对象池。两个池都包含 1,000 个对象；`object_pool1`中包含`ObjectMetadata1`结构中的元数据，而`object_pool2`使用`ObjectMetadata2`结构。现在，让我们检查应用程序的输出：

![](img/b9ba5450-5659-4cb2-a383-bd4285356c9a.png)

两个对象池在功能和性能方面是相同的。但是，如果我们检查它们占用了多少内存，我们可以看到一个显著的差异：`object_pool1`比`object_pool2`大 4KB。鉴于`object_pool2`的大小为 12KB，我们浪费了 33%的内存，因为没有注意数据对齐。在处理数据结构时要注意对齐和填充，因为不正确的字段排序可能导致内存使用效率低下，就像`object_pool2`的情况一样。使用这些简单的规则来组织数据字段，以保持它们正确对齐：

+   按照它们的大小对它们进行分组。

+   按照从最大到最小的数据类型对组进行排序。

良好对齐的数据结构速度快、内存效率高，并且不需要实现任何额外的代码。

# 还有更多...

每个硬件平台都有自己的对齐要求，其中一些是棘手的。您可能需要查阅目标平台编译器文档和最佳实践，以充分利用硬件。如果您的目标平台是 ARM，请考虑阅读 ARM 技术文章[`infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.faqs/ka15414.html`](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.faqs/ka15414.html)上的对齐期望。

虽然结构体内数据字段的正确对齐可以导致更紧凑的数据表示，但要注意性能影响。将一起使用的数据保持在同一内存区域中称为**数据局部性**，可能会显著提高数据访问性能。适合放入同一缓存行的数据元素可以比跨越缓存行边界的元素读取或写入得快得多。在许多情况下，更倾向于通过额外的内存使用来获得性能提升。我们将在*使用缓存行对齐数据*配方中更详细地讨论这种技术。

# 使用打包结构

在这个配方中，我们将学习如何定义结构，使其在数据成员之间没有填充字节。如果应用程序处理大量对象，这可能会显著减少应用程序使用的内存量。

请注意，这是有代价的。未对齐的内存访问速度较慢，导致性能不佳。对于某些架构，未对齐访问是被禁止的，因此需要 C++编译器生成比对齐访问更多的代码来访问数据字段。

尽管打包结构体可能会导致更有效的内存使用，但除非真的必要，否则避免使用这种技术。它有太多暗含的限制，可能会导致应用程序中难以发现的模糊问题。

将紧凑结构视为传输编码，并仅在应用程序外部存储、加载或交换数据时使用它们。但是，即使在这些情况下，使用适当的数据序列化也是更好的解决方案。

# 如何做...

在这个简单的应用程序中，我们将定义一个紧凑结构的数组，并查看这如何影响它所需的内存量。按照以下步骤操作：

1.  在您的工作目录`~/test`中，创建`alignment`子目录的副本。将其命名为`packed_alignment`。

1.  通过向每个结构的定义添加`__attribute__((packed))`来修改`alignment.cpp`文件：

```cpp
struct ObjectMetadata1 {
  uint8_t access_flags;
  uint32_t size;
  uint32_t owner_id;
  Category category;
} __attribute__((packed));

struct ObjectMetadata2 {
  uint32_t size;
  uint32_t owner_id;
  uint8_t access_flags;
  Category category;
} __attribute__((packed));
```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的教程*设置环境*来操作。

1.  切换到目标系统的终端。如果需要，使用您的用户凭据登录。

1.  运行二进制文件。

# 它是如何工作的...

在这个教程中，我们通过向每个结构添加一个紧凑属性来修改了*使用数据对齐*教程中的代码：

```cpp
} __attribute__((packed));
```

此属性指示编译器不要向结构添加填充字节，以符合目标平台的对齐要求。

运行上述代码会给我们以下输出：

![](img/521bd29e-1012-4d5a-b7fc-08a360183077.png)

如果编译器不添加填充字节，数据字段的顺序变得不重要。鉴于`ObjectMetadata1`和`ObjectMetadata2`结构具有完全相同的数据字段，它们在紧凑形式中的大小变得相同。

# 还有更多...

`GNU 编译器集合`（**GCC**）通过其属性为开发人员提供了对数据布局的大量控制。您可以通过访问[GCC 类型属性](https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Type-Attributes.html#Type-Attributes)页面了解所有支持的属性及其含义。

其他编译器提供类似的功能，但它们的 API 可能不同。例如，Microsoft 编译器定义了`#pragma pack`编译器指令来声明紧凑结构。更多细节可以在[Pragma Pack Reference](https://docs.microsoft.com/en-us/cpp/preprocessor/pack?view=vs-2019)页面找到。

# 使用缓存行对齐数据

在这个教程中，我们将学习如何将数据结构与缓存行对齐。数据对齐可以显著影响系统的性能，特别是在多核系统中运行多线程应用程序的情况下。

首先，如果数据结构在同一个缓存行中，频繁访问一起使用的数据会更快。如果你的程序一直访问变量 A 和变量 B，处理器每次都需要使缓存失效并重新加载，如果它们不在同一行中。

其次，您不希望将不同线程独立使用的数据放在同一个缓存行中。如果同一个缓存行被不同的 CPU 核修改，这就需要缓存同步，这会影响使用共享数据的多线程应用程序的整体性能，因为在这种情况下，内存访问时间显著增加。

# 如何做...

我们将创建一个应用程序，使用四种不同的方法分配四个缓冲区，以学习如何对齐静态和动态分配的内存。按照以下步骤操作：

1.  在您的工作目录`~/test`中创建一个名为`cache_align`的子目录。

1.  使用您喜欢的文本编辑器在`cache_align`子目录中创建一个名为`cache_align.cpp`的文件。将以下代码片段复制到`cache_align.cpp`文件中，以定义必要的常量和检测对齐的函数：

```cpp
#include <stdlib.h>
#include <stdio.h>

constexpr int kAlignSize = 128;
constexpr int kAllocBytes = 128;

constexpr int overlap(void* ptr) {
  size_t addr = (size_t)ptr;
  return addr & (kAlignSize - 1);
 }
```

1.  现在，定义几个以不同方式分配的缓冲区：

```cpp
int main() {
  char static_buffer[kAllocBytes];
  char* dynamic_buffer = new char[kAllocBytes];

  alignas(kAlignSize) char aligned_static_buffer[kAllocBytes];
  char* aligned_dynamic_buffer = nullptr;
  if (posix_memalign((void**)&aligned_dynamic_buffer,
      kAlignSize, kAllocBytes)) {
    printf("Failed to allocate aligned memory buffer\n");
  }

```

1.  添加以下代码来使用它们：

```cpp
  printf("Static buffer address: %p (%d)\n", static_buffer,
         overlap(static_buffer));
  printf("Dynamic buffer address: %p (%d)\n", dynamic_buffer,
         overlap(dynamic_buffer));
  printf("Aligned static buffer address: %p (%d)\n", aligned_static_buffer,
         overlap(aligned_static_buffer));
  printf("Aligned dynamic buffer address: %p (%d)\n", aligned_dynamic_buffer,
         overlap(aligned_dynamic_buffer));
  delete[] dynamic_buffer;
  free(aligned_dynamic_buffer);
  return 0;
  }
```

1.  在 loop 子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(cache_align)
add_executable(cache_align cache_align.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的配方，*设置环境*，来完成此操作。

1.  切换到目标系统的终端。如果需要，使用您的用户凭据登录。

1.  运行二进制文件。

# 工作原理...

在第一个代码片段中，我们创建了两对内存缓冲区。在每对中，第一个缓冲区分配给堆栈，而第二个缓冲区分配给堆。

第一对是使用标准 C++技术创建的。堆栈上的静态缓冲区声明为数组：

```cpp
  char static_buffer[kAllocBytes];
```

要创建动态缓冲区，我们使用`new` C++关键字：

```cpp
  char* dynamic_buffer = new char[kAllocBytes];
```

在第二对中，我们创建了内存对齐的缓冲区。在堆栈上声明静态缓冲区与常规静态缓冲区类似。我们使用了一个额外的属性`alignas`，这是 C++11 中引入的一种标准化和平台无关的内存对齐方式：

```cpp
 alignas(kAlignSize) char aligned_static_buffer[kAllocBytes];
```

此属性需要一个对齐大小作为参数。我们希望数据按缓存行边界对齐。根据平台的不同，缓存行大小可能不同。最常见的大小是 32、64 和 128 字节。使用 128 字节可以使我们的缓冲区对任何缓存行大小都对齐。

没有标准的方法来为动态缓冲区做同样的事情。为了在堆上分配内存，我们使用一个名为`posix_memalign`的 C 函数。这仅在**可移植操作系统接口**（**POSIX**）系统（大多是类 Unix 系统）中可用，但这并不需要 C++11 标准的支持：

```cpp
  if (posix_memalign((void**)&aligned_dynamic_buffer,
 kAlignSize, kAllocBytes)) {
```

`posix_memalign`类似于`malloc`，但有三个参数而不是一个。第二个参数是对齐大小，与对齐属性相同。第三个是要分配的内存大小。第一个参数用于返回分配内存的指针。与`malloc`不同，`posix_memalign`可能会失败，不仅是因为无法分配内存，还因为传递给函数的对齐大小不是 2 的幂。`posix_memalign`返回一个错误代码作为其结果值，以帮助开发人员区分这两种情况。

我们定义了函数 overlap 来计算指针的非对齐部分，通过屏蔽所有对齐位：

```cpp
  size_t addr = (size_t)ptr;
  return addr & (kAlignSize - 1);
```

当我们运行应用程序时，我们可以看到区别：

![](img/007b6140-8b16-49a8-b066-43024d936300.png)

第一对中两个缓冲区的地址有非对齐部分，而第二对的地址是对齐的-非对齐部分为零。因此，对第二对缓冲区的元素进行随机访问更快，因为它们都同时在缓存中可用。

# 还有更多...

CPU 访问数据对齐对于通过硬件地址转换机制高效映射内存也至关重要。现代操作系统操作 4 KB 内存块或页面，以将进程的虚拟地址空间映射到物理内存。将数据结构对齐到 4 KB 边界可以带来性能提升。

我们在这个配方中描述的相同技术可以应用于将数据对齐到内存页边界。但是，请注意，`posix_memalign`可能需要比请求的内存多两倍。对于较大的对齐块，这种内存开销增长可能是显著的。
