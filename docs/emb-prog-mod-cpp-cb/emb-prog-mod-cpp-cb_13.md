# 第十三章：实时系统的指南

实时系统是时间反应至关重要的嵌入式系统的一类。未能及时反应的后果在不同的应用程序之间有所不同。根据严重程度，实时系统被分类如下：

+   **硬实时**：错过截止日期是不可接受的，被视为系统故障。这些通常是飞机、汽车和发电厂中的关键任务系统。

+   **严格实时**：在极少数情况下错过截止日期是可以接受的。截止日期后结果的有用性为零。想想一个直播流服务。交付太晚的视频帧只能被丢弃。只要这种情况不经常发生，这是可以容忍的。

+   **软实时**：错过截止日期是可以接受的。截止日期后结果的有用性会下降，导致整体质量的下降，应该避免。一个例子是从多个传感器捕获和同步数据。

实时系统不一定需要非常快。它们需要的是可预测的反应时间。如果一个系统通常可以在 10 毫秒内响应事件，但经常需要更长时间，那么它就不是一个实时系统。如果一个系统能够在 1 秒内保证响应，那就构成了硬实时。

确定性和可预测性是实时系统的主要特征。在本章中，我们将探讨不可预测行为的潜在来源以及减轻它们的方法。

本章涵盖以下主题：

+   在 Linux 中使用实时调度器

+   使用静态分配的内存

+   避免异常处理错误

+   探索实时操作系统

本章的食谱将帮助您更好地了解实时系统的具体情况，并学习一些针对这种嵌入式系统的软件开发的最佳实践。

# 在 Linux 中使用实时调度器

Linux 是一个通用操作系统，在各种嵌入式设备中通常被使用，因为它的多功能性。它可以根据特定的硬件进行定制，并且是免费的。

Linux 不是一个实时操作系统，也不是实现硬实时系统的最佳选择。然而，它可以有效地用于构建软实时系统，因为它为时间关键的应用程序提供了实时调度器。

在本章中，我们将学习如何在我们的应用程序中在 Linux 中使用实时调度器。

# 如何做...

我们将创建一个使用实时调度器的应用程序：

1.  在您的工作目录`~/test`中，创建一个名为`realtime`的子目录。

1.  使用您喜欢的文本编辑器在`realtime`子目录中创建一个`realtime.cpp`文件。

1.  添加所有必要的包含和命名空间：

```cpp
#include <iostream>
#include <system_error>
#include <thread>
#include <chrono>

#include <pthread.h>

using namespace std::chrono_literals;
```

1.  接下来，添加一个配置线程使用实时调度器的函数：

```cpp
void ConfigureRealtime(pthread_t thread_id, int priority) {
    sched_param sch;
    sch.sched_priority = 20;
    if (pthread_setschedparam(thread_id,
                              SCHED_FIFO, &sch)) {
        throw std::system_error(errno, 
                std::system_category(),
                "Failed to set real-time priority");
    }
}
```

1.  接下来，我们定义一个希望以正常优先级运行的线程函数：

```cpp
void Measure(const char* text) {
    struct timespec prev;
    timespec_get(&prev, TIME_UTC);
    struct timespec delay{0, 10};
    for (int i = 0; i < 100000; i++) {
      nanosleep(&delay, nullptr);
    }
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    double delta = (ts.tv_sec - prev.tv_sec) + 
        (double)(ts.tv_nsec - prev.tv_nsec) / 1000000000;
    std::clog << text << " completed in " 
              << delta << " sec" << std::endl;
}
```

1.  接下来是一个实时线程函数和一个启动这两个线程的`main`函数：

```cpp
void RealTimeThread(const char* txt) {
    ConfigureRealtime(pthread_self(), 1);
    Measure(txt);
}

int main() {
    std::thread t1(RealTimeThread, "Real-time");
    std::thread t2(Measure, "Normal");
    t1.join();
    t2.join();
}
```

1.  最后，我们创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(realtime)
add_executable(realtime realtime.cpp)
target_link_libraries(realtime pthread)

SET(CMAKE_CXX_FLAGS "--std=c++14") 
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)
```

1.  现在您可以构建和运行应用程序了。

# 它是如何工作的...

Linux 有几种调度策略，它应用于应用程序进程和线程。`SCHED_OTHER`是默认的 Linux 分时策略。它适用于所有线程，不提供实时机制。

在我们的应用程序中，我们使用另一个策略`SCHED_FIFO`。这是一个简单的调度算法。使用这个调度器的所有线程只能被优先级更高的线程抢占。如果线程进入睡眠状态，它将被放置在具有相同优先级的线程队列的末尾。

`SCHED_FIFO`策略的线程优先级始终高于`SCHED_OTHER`策略的线程优先级，一旦`SCHED_FIFO`线程变为可运行状态，它立即抢占正在运行的`SCHED_OTHER`线程。从实际的角度来看，如果系统中只有一个`SCHED_FIFO`线程在运行，它可以使用所需的 CPU 时间。`SCHED_FIFO`调度程序的确定性行为和高优先级使其非常适合实时应用程序。

为了将实时优先级分配给一个线程，我们定义了一个`ConfigureRealtime`函数。它接受两个参数——线程 ID 和期望的优先级：

```cpp
void ConfigureRealtime(pthread_t thread_id, int priority) {
```

该函数为`pthread_setschedparam`函数填充数据，该函数使用操作系统的低级 API 来更改线程的调度程序和优先级：

```cpp
    if (pthread_setschedparam(thread_id,
 SCHED_FIFO, &sch)) {
```

我们定义一个`Measure`函数，运行一个繁忙循环，调用`nanosleep`函数，参数要求它休眠 10 纳秒，这对于将执行让给另一个线程来说太短了：

```cpp
    struct timespec delay{0, 10};
    for (int i = 0; i < 100000; i++) {
      nanosleep(&delay, nullptr);
    }
```

此函数在循环之前和之后捕获时间戳，并计算经过的时间（以秒为单位）：

```cpp
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    double delta = (ts.tv_sec - prev.tv_sec) + 
        (double)(ts.tv_nsec - prev.tv_nsec) / 1000000000;
```

接下来，我们将`RealTimeThread`函数定义为`Measure`函数的包装。这将当前线程的优先级设置为实时，并立即调用`Measure`：

```cpp
    ConfigureRealtime(pthread_self(), 1);
    Measure(txt);
```

在`main`函数中，我们启动两个线程，传递文本字面量作为参数以区分它们的输出。如果我们在树莓派设备上运行程序，可以看到以下输出：

![](img/56a567f4-a5ee-43ce-8a70-8e6471fc11d3.png)

实时线程所花费的时间少了四倍，因为它没有被普通线程抢占。这种技术可以有效地满足 Linux 环境中的软实时需求。

# 使用静态分配的内存

如第六章中已经讨论过的，应该避免在实时系统中使用动态内存分配，因为通用内存分配器没有时间限制。虽然在大多数情况下，内存分配不会花费太多时间，但不能保证。这对于实时系统是不可接受的。

避免动态内存分配的最直接方法是用静态分配替换它。C++开发人员经常使用`std::vector`来存储元素序列。由于它与 C 数组相似，因此它高效且易于使用，并且其接口与标准库中的其他容器一致。由于向量具有可变数量的元素，因此它们广泛使用动态内存分配。然而，在许多情况下，可以使用`std::array`类来代替`std::vector`。它具有相同的接口，只是其元素的数量是固定的，因此其实例可以静态分配。这使得它成为在内存分配时间至关重要时替代`std::vector`的良好选择。

在本示例中，我们将学习如何有效地使用`std::array`来表示固定大小的元素序列。

# 操作步骤如下...

我们将创建一个应用程序，利用 C++标准库算法的功能来生成和处理固定数据帧，而不使用动态内存分配：

1.  在您的工作目录`~/test`中，创建一个名为`array`的子目录。

1.  使用您喜欢的文本编辑器在`array`子目录中创建一个名为`array.cpp`的文件。

1.  在`array.cpp`文件中添加包含和新的类型定义：

```cpp
#include <algorithm>
#include <array>
#include <iostream>
#include <random>

using DataFrame = std::array<uint32_t, 8>;
```

1.  接下来，我们添加一个生成数据帧的函数：

```cpp
void GenerateData(DataFrame& frame) {
  std::random_device rd;
 std::generate(frame.begin(), frame.end(),
 [&rd]() { return rd() % 100; });
}
```

1.  接下来是处理数据帧的函数：

```cpp
void ProcessData(const DataFrame& frame) {
  std::cout << "Processing array of "
            << frame.size() << " elements: [";
  for (auto x : frame) {
    std::cout << x << " ";
  }
  auto mm = std::minmax_element(frame.begin(),frame.end());
  std::cout << "] min: " << *mm.first
            << ", max: " << *mm.second << std::endl;
}
```

1.  添加一个将数据生成和处理联系在一起的`main`函数：

```cpp
int main() {
  DataFrame data;

  for (int i = 0; i < 4; i++) {
    GenerateData(data);
    ProcessData(data);
  }
  return 0;
}
```

1.  最后，我们创建一个`CMakeLists.txt`文件，其中包含程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(array)
add_executable(array array.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS_RELEASE "--std=c++17") 
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} -g -DDEBUG") 

set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

1.  现在可以构建和运行应用程序了。

# 工作原理...

我们使用`std::array`模板来声明自定义的`DataFrame`数据类型。对于我们的示例应用程序，`DataFrame`是一个包含八个 32 位整数的序列：

```cpp
using DataFrame = std::array<uint32_t, 8>;
```

现在，我们可以在函数中使用新的数据类型来生成和处理数据框架。由于数据框架是一个数组，我们通过引用将其传递给`GenerateData`函数，以避免额外的复制：

```cpp
void GenerateData(DataFrame& frame) {
```

`GenerateData`用随机数填充数据框架。由于`std::array`具有与标准库中其他容器相同的接口，我们可以使用标准算法使代码更短更可读：

```cpp
 std::generate(frame.begin(), frame.end(),
 [&rd]() { return rd() % 100; });
```

我们以类似的方式定义了`ProcessData`函数。它也接受一个`DataFrame`，但不应该修改它。我们使用常量引用明确说明数据不会被修改：

```cpp
void ProcessData(const DataFrame& frame) {
```

`ProcessData`打印数据框架中的所有值，然后找到框架中的最小值和最大值。与内置数组不同，当传递给函数时，`std::arrays`不会衰减为原始指针，因此我们可以使用基于范围的循环语法。您可能会注意到，我们没有将数组的大小传递给函数，并且没有使用任何全局常量来查询它。这是`std::array`接口的一部分。它不仅减少了函数的参数数量，还确保我们在调用它时不能传递错误的大小：

```cpp
  for (auto x : frame) {
    std::cout << x << " ";
  }
```

为了找到最小值和最大值，我们使用标准库的`std::minmax_`元素函数，而不是编写自定义循环：

```cpp
auto mm = std::minmax_element(frame.begin(),frame.end());
```

在`main`函数中，我们创建了一个`DataFrame`的实例：

```cpp
DataFrame data;
```

然后，我们运行一个循环。在每次迭代中，都会生成和处理一个新的数据框架：

```cpp
GenerateData(data);
ProcessData(data);
```

如果我们运行应用程序，我们会得到以下输出：

![](img/5443008c-9e80-4ed5-818e-9b2df50b60c6.png)

我们的应用程序生成了四个数据框架，并且只使用了几行代码和静态分配的数据来处理其数据。这使得`std::array`成为实时系统开发人员的一个很好的选择。此外，与内置数组不同，我们的函数是类型安全的，我们可以在构建时检测和修复许多编码错误。

# 还有更多...

C++20 标准引入了一个新函数`to_array`，允许开发人员从一维内置数组创建`std::array`的实例。在`to_array`参考页面上查看更多细节和示例（[`en.cppreference.com/w/cpp/container/array/to_array`](https://en.cppreference.com/w/cpp/container/array/to_array)）。

# 避免使用异常进行错误处理

异常机制是 C++标准的一个组成部分。这是设计 C++程序中的错误处理的推荐方式。然而，它确实有一些限制，不总是适用于实时系统，特别是安全关键系统。

C++异常处理严重依赖于堆栈展开。一旦抛出异常，它会通过调用堆栈传播到可以处理它的 catch 块。这意味着在其路径中调用堆栈帧中的所有本地对象的析构函数，并且很难确定并正式证明此过程的最坏情况时间。

这就是为什么安全关键系统的编码指南，如 MISRA 或 JSF，明确禁止使用异常进行错误处理。

这并不意味着 C++开发人员必须回到传统的纯 C 错误代码。在这个示例中，我们将学习如何使用 C++模板来定义可以保存函数调用的结果或错误代码的数据类型。

# 如何做...

我们将创建一个应用程序，利用 C++标准库算法的强大功能来生成和处理固定数据框架，而不使用动态内存分配：

1.  在你的工作目录`~/test`中，创建一个名为`expected`的子目录。

1.  使用你喜欢的文本编辑器在`expected`子目录中创建一个`expected.cpp`文件。

1.  向`expected.cpp`文件添加包含和新的类型定义：

```cpp
#include <iostream>
#include <system_error>
#include <variant>

#include <unistd.h>
#include <sys/fcntl.h>

template <typename T>
class Expected {
  std::variant<T, std::error_code> v;

public:
  Expected(T val) : v(val) {}
  Expected(std::error_code e) : v(e) {}

  bool valid() const {
    return std::holds_alternative<T>(v);
  }

  const T& value() const {
    return std::get<T>(v);
  }

  const std::error_code& error() const {
    return std::get<std::error_code>(v);
  }
};
```

1.  接下来，我们为打开的 POSIX 函数添加一个包装器：

```cpp
Expected<int> OpenForRead(const std::string& name) {
  int fd = ::open(name.c_str(), O_RDONLY);
  if (fd < 0) {
    return Expected<int>(std::error_code(errno, 
                         std::system_category()));
  }
  return Expected<int>(fd);
}
```

1.  添加`main`函数，显示如何使用`OpenForRead`包装器：

```cpp
int main() {
  auto result = OpenForRead("nonexistent.txt");
  if (result.valid()) {
    std::cout << "File descriptor"
              << result.value() << std::endl;
  } else {
    std::cout << "Open failed: " 
              << result.error().message() << std::endl;
  }
  return 0;
}
```

1.  最后，我们创建一个`CMakeLists.txt`文件，其中包含我们程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(expected)
add_executable(expected expected.cpp)

set(CMAKE_SYSTEM_NAME Linux)
#set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++17") 

#set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
#set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

1.  现在可以构建和运行应用程序了。

# 它是如何工作的...

在我们的应用程序中，我们创建了一个数据类型，可以以类型安全的方式保存预期值或错误代码。C++17 提供了一个类型安全的联合类`std::variant`，我们将使用它作为我们的模板类`Expected`的基础数据类型。

`Expected`类封装了一个`std::variant`字段，可以容纳两种数据类型之一，即模板类型`T`或`std::error_code`，后者是错误代码的标准 C++泛化：

```cpp
  std::variant<T, std::error_code> v;
```

虽然可以直接使用`std::variant`，但我们公开了一些使其更加方便的公共方法。`valid`方法在结果持有模板类型时返回`true`，否则返回`false`：

```cpp
  bool valid() const {
    return std::holds_alternative<T>(v);
  }
```

`value`和`error`方法用于访问返回的值或错误代码：

```cpp
  const T& value() const {
    return std::get<T>(v);
  }

  const std::error_code& error() const {
    return std::get<std::error_code>(v);
  }
```

一旦定义了`Expected`类，我们就创建一个使用它的`OpenForReading`函数。这会调用打开系统函数，并根据返回值创建一个持有文件描述符或错误代码的`Expected`实例：

```cpp
  if (fd < 0) {
    return Expected<int>(std::error_code(errno, 
 std::system_category()));
  }
  return Expected<int>(fd);
```

在`main`函数中，当我们为不存在的文件调用`OpenForReading`时，预计会失败。当我们运行应用程序时，可以看到以下输出：

![](img/60b67f21-dfee-4227-8fa9-b3367d95f288.png)

我们的`Expected`类允许我们以类型安全的方式编写可能返回错误代码的函数。编译时类型验证有助于开发人员避免许多传统错误代码常见的问题，使我们的应用程序更加健壮和安全。

# 还有更多...

我们的`Expected`数据类型的实现是`std::expected`类的一个变体（[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0323r7.html`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0323r7.html)），该类被提议用于标准化，但尚未获得批准。`std::expected`的一个实现可以在 GitHub 上找到（[`github.com/TartanLlama/expected`](https://github.com/TartanLlama/expected)）。

# 探索实时操作系统

正如本章已经讨论的那样，Linux 不是实时系统。它是软实时任务的一个很好选择，但尽管它提供了一个实时调度程序，但其内核过于复杂，无法保证硬实时应用程序所需的确定性水平。

时间关键的应用程序要么需要实时操作系统来运行，要么被设计和实现为在裸机上运行，根本没有操作系统。

实时操作系统通常比 Linux 等通用操作系统简单得多。此外，它们需要根据特定的硬件平台进行定制，通常是微控制器。

有许多实时操作系统，其中大多数是专有的，而且不是免费的。FreeRTOS 是探索实时操作系统功能的良好起点。与大多数替代方案不同，它是开源的，并且可以免费使用，因为它是根据 MIT 许可证分发的。它被移植到许多微控制器和小型微处理器，但即使您没有特定的硬件，Windows 和 POSIX 模拟器也是可用的。

在这个配方中，我们将学习如何下载和运行 FreeRTOS POSIX 模拟器。

# 如何做到...

我们将在我们的构建环境中下载和构建 FreeRTOS 模拟器：

1.  切换到 Ubuntu 终端并将当前目录更改为`/mnt`：

```cpp
$ cd /mnt
```

1.  下载 FreeRTOS 模拟器的源代码：

```cpp
$ wget -O simulator.zip http://interactive.freertos.org/attachments/token/r6d5gt3998niuc4/?name=Posix_GCC_Simulator_6.0.4.zip
```

1.  提取下载的存档：

```cpp
$ unzip simulator.zip
```

1.  将当前目录更改为`Posix_GCC_Simulator/FreeRTOS_Posix/Debug`：

```cpp
$ cd Posix_GCC_Simulator/FreeRTOS_Posix/Debug
```

1.  通过运行以下命令修复`makefile`中的小错误：

```cpp
$ sed -i -e 's/\(.*gcc.*\)-lrt\(.*\)/\1\2 -lrt/' makefile
```

1.  从源代码构建模拟器：

```cpp
$ make
```

1.  启动它：

```cpp
$ ./FreeRTOS_Posix
```

此时，模拟器正在运行。

# 它是如何工作的...

正如我们已经知道的那样，实时操作系统的内核通常比通用操作系统的内核简单得多。对于 FreeRTOS 也是如此。

由于这种简单性，内核可以在通用操作系统（如 Linux 或 Windows）中作为一个进程构建和运行。当从另一个操作系统中使用时，它就不再是真正的实时，但可以作为探索 FreeRTOS API 并开始开发后续可以在目标硬件平台的实时环境中运行的应用程序的起点。

在这个教程中，我们下载并为 POSIX 操作系统构建了 FreeRTOS 内核。

构建阶段很简单。一旦代码从存档中下载并提取出来，我们运行`make`，这将构建一个单个可执行文件`FreeRTOS-POSIX`。在运行`make`命令之前，我们通过运行`sed`在`makefile`中修复了一个错误，将`-lrt`选项放在 GCC 命令行的末尾。

```cpp
$ sed -i -e 's/\(.*gcc.*\)-lrt\(.*\)/\1\2 -lrt/' makefile
```

运行应用程序会启动内核和预打包的应用程序：

![](img/592082ae-35ae-405e-8d7d-fefe26872dae.png)

我们能够在我们的构建环境中运行 FreeRTOS。您可以深入研究其代码库和文档，以更好地理解实时操作系统的内部和 API。

# 还有更多...

如果您在 Windows 环境中工作，有一个更好支持的 FreeRTOS 模拟器的 Windows 版本。它可以从[`www.freertos.org/FreeRTOS-Windows-Simulator-Emulator-for-Visual-Studio-and-Eclipse-MingW.html`](https://www.freertos.org/FreeRTOS-Windows-Simulator-Emulator-for-Visual-Studio-and-Eclipse-MingW.html)下载，还有文档和教程。
