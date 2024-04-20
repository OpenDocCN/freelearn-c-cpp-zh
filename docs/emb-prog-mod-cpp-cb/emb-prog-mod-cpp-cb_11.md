# 第十一章：时间点和间隔

嵌入式应用程序处理发生在物理世界中的事件和控制过程——这就是为什么正确处理时间和延迟对它们至关重要。交通灯的切换；声音音调的生成；来自多个传感器的数据同步——所有这些任务都依赖于正确的时间测量。

纯 C 不提供任何标准函数来处理时间。预期应用程序开发人员将使用特定于目标操作系统的时间 API——Windows、Linux 或 macOS。对于裸机嵌入式系统，开发人员必须创建自定义函数来处理时间，这些函数基于特定于目标平台的低级定时器 API。结果，代码很难移植到其他平台。

为了克服可移植性问题，C++（从 C++11 开始）定义了用于处理时间和时间间隔的数据类型和函数。这个 API 被称为`std::chrono`库，它帮助开发人员以统一的方式在任何环境和任何目标平台上处理时间。

在本章中，我们将学习如何在我们的应用程序中处理时间戳、时间间隔和延迟。我们将讨论与时间管理相关的一些常见陷阱，以及它们的适当解决方法。

我们将涵盖以下主题：

+   探索 C++ Chrono 库

+   测量时间间隔

+   处理延迟

+   使用单调时钟

+   使用**可移植操作系统接口**（**POSIX**）时间戳

使用这些示例，您将能够编写可在任何嵌入式平台上运行的时间处理的可移植代码。

# 探索 C++ Chrono 库

从 C++11 开始，C++ Chrono 库提供了标准化的数据类型和函数，用于处理时钟、时间点和时间间隔。在这个示例中，我们将探索 Chrono 库的基本功能，并学习如何处理时间点和间隔。

我们还将学习如何使用 C++字面量来更清晰地表示时间间隔。

# 如何做...

我们将创建一个简单的应用程序，创建三个时间点并将它们相互比较。

1.  在您的`~/test`工作目录中，创建一个名为`chrono`的子目录。

1.  使用您喜欢的文本编辑器在`chrono`子目录中创建一个`chrono.cpp`文件。

1.  将以下代码片段放入文件中：

```cpp
#include <iostream>
#include <chrono>

using namespace std::chrono_literals;

int main() {
  auto a = std::chrono::system_clock::now();
  auto b = a + 1s;
  auto c = a + 200ms;

  std::cout << "a < b ? " << (a < b ? "yes" : "no") << std::endl;
  std::cout << "a < c ? " << (a < c ? "yes" : "no") << std::endl;
  std::cout << "b < c ? " << (b < c ? "yes" : "no") << std::endl;

  return 0;
}
```

1.  创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(chrono)
add_executable(chrono chrono.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在您可以构建和运行应用程序。

# 它是如何工作的...

我们的应用程序创建了三个不同的时间点。第一个是使用系统时钟的`now`函数创建的：

```cpp
auto a = std::chrono::system_clock::now();
```

另外两个时间点是通过添加固定的时间间隔`1`秒和`200`毫秒从第一个时间点派生出来的：

```cpp
auto b = a + 1s;
auto c = a + 200ms;
```

请注意我们是如何在数字值旁边指定时间单位的。我们使用了一个叫做 C++字面量的特性。Chrono 库为基本时间单位定义了这样的字面量。为了使用这些定义，我们添加了以下内容：

```cpp
using namespace std::chrono_literals;
```

这是在我们的`main`函数之前添加的。

接下来，我们将比较这些时间点：

```cpp
std::cout << "a < b ? " << (a < b ? "yes" : "no") << std::endl;
std::cout << "a < c ? " << (a < c ? "yes" : "no") << std::endl;
std::cout << "b < c ? " << (b < c ? "yes" : "no") << std::endl;
```

当我们运行应用程序时，我们会看到以下输出：

![](img/00856d97-097c-4ff9-98ef-4ed42bfda18c.png)

如预期的那样，时间点`a`比`b`和`c`都要早，其中时间点`c`（即`a`+200 毫秒）比`b`（`a`+1 秒）要早。字符串字面量有助于编写更易读的代码，C++ Chrono 提供了丰富的函数集来处理时间。我们将在下一个示例中学习如何使用它们。

# 还有更多...

Chrono 库中定义的所有数据类型、模板和函数的信息可以在 Chrono 参考中找到[`en.cppreference.com/w/cpp/chrono `](https://en.cppreference.com/w/cpp/chrono)

# 测量时间间隔

与外围硬件交互或响应外部事件的每个嵌入式应用程序都必须处理超时和反应时间。为了正确地做到这一点，开发人员需要能够以足够的精度测量时间间隔。

C++ Chrono 库提供了一个用于处理任意跨度和精度的持续时间的`std::chrono::duration`模板类。在这个示例中，我们将学习如何使用这个类来测量两个时间戳之间的时间间隔，并将其与参考持续时间进行比较。

# 如何做...

我们的应用程序将测量简单控制台输出的持续时间，并将其与循环中的先前值进行比较。

1.  在您的`〜/test`工作目录中，创建一个名为`intervals`的子目录。

1.  使用您喜欢的文本编辑器在`intervals`子目录中创建一个名为`intervals.cpp`的文件。

1.  将以下代码片段复制到`intervals.cpp`文件中：

```cpp
#include <iostream>
#include <chrono>

int main() {
  std::chrono::duration<double, std::micro> prev;
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::steady_clock::now();
    std::cout << i << ": ";
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> delta = end - start;
    std::cout << "output duration is " << delta.count() <<" us";
    if (i) {
      auto diff = (delta - prev).count();
      if (diff >= 0) {
        std::cout << ", " << diff << " us slower";
      } else {
        std::cout << ", " << -diff << " us faster";
      }
    }
    std::cout << std::endl;
    prev = delta;
  }
  return 0;
}
```

1.  最后，创建一个`CMakeLists.txt`文件，其中包含我们程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(interval)
add_executable(interval interval.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在，您可以构建并运行应用程序。

# 它是如何工作的...

在应用程序循环的每次迭代中，我们测量一个输出操作的性能。为此，我们在操作之前捕获一个时间戳，操作完成后捕获另一个时间戳：

```cpp
 auto start = std::chrono::steady_clock::now();
    std::cout << i << ": ";
 auto end = std::chrono::steady_clock::now();
```

我们使用 C++11 的`auto`让编译器推断时间戳的数据类型。现在，我们需要计算这些时间戳之间的时间间隔。从一个时间戳减去另一个时间戳就可以完成任务。我们明确将结果变量定义为`std::chrono::duration`类，该类跟踪`double`值中的微秒：

```cpp
 std::chrono::duration<double, std::micro> delta = end - start;
```

我们使用另一个相同类型的`duration`变量来保存先前的值。除了第一次迭代之外的每次迭代，我们计算这两个持续时间之间的差异：

```cpp
    auto diff = (delta - prev).count();
```

在每次迭代中，持续时间和差异都会打印到终端上。当我们运行应用程序时，我们会得到这个输出：

![](img/ec323f6d-4496-4050-a609-dc90436a90c5.png)

正如我们所看到的，现代 C++提供了方便的方法来处理应用程序中的时间间隔。由于重载运算符，很容易获得两个时间点之间的持续时间，并且可以添加、减去或比较持续时间。

# 还有更多...

从 C++20 开始，Chrono 库支持直接将持续时间写入输出流并从输入流中解析持续时间。无需将持续时间显式序列化为整数或浮点值。这使得处理持续时间对于 C++开发人员更加方便。

# 处理延迟

周期性数据处理是许多嵌入式应用程序中的常见模式。代码不需要一直运行。如果我们预先知道何时需要处理，应用程序或工作线程可以大部分时间处于非活动状态，只有在需要时才唤醒并处理数据。这样可以节省电力消耗，或者在应用程序空闲时让设备上运行的其他应用程序使用 CPU 资源。

有几种组织周期性处理的技术。运行一个带有延迟的循环的工作线程是其中最简单和最常见的技术之一。

C++提供了标准函数来向当前执行线程添加延迟。在这个示例中，我们将学习两种向应用程序添加延迟的方法，并讨论它们的优缺点。

# 如何做...

我们将创建一个具有两个处理循环的应用程序。这些循环使用不同的函数来暂停当前线程的执行。

1.  在您的`〜/test`工作目录中，创建一个名为`delays`的子目录。

1.  使用您喜欢的文本编辑器在`delays`子目录中创建一个名为`delays.cpp`的文件。

1.  让我们首先添加一个名为`sleep_for`的函数，以及必要的包含：

```cpp
#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

void sleep_for(int count, auto delay) {
  for (int i = 0; i < count; i++) {
    auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_for(delay);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta = end - start;
    std::cout << "Sleep for: " << delta.count() << std::endl;
  }
}
```

1.  它后面是第二个函数`sleep_until`：

```cpp
void sleep_until(int count, 
                 std::chrono::milliseconds delay) {
  auto wake_up = std::chrono::system_clock::now();
  for (int i = 0; i < 10; i++) {
    wake_up += delay;
    auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_until(wake_up);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta = end - start;
    std::cout << "Sleep until: " << delta.count() << std::endl;
  }
}
```

1.  接下来，添加一个简单的`main`函数来调用它们：

```cpp
int main() {
  sleep_for(10, 100ms);
  sleep_until(10, 100ms);
  return 0;
}
```

1.  最后，创建一个`CMakeLists.txt`文件，其中包含我们程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(delays)
add_executable(delays delays.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在，您可以构建并运行应用程序了。

# 它是如何工作的...

在我们的应用程序中，我们创建了两个函数，`sleep_for`和`sleep_until`。它们几乎相同，只是`sleep_for`使用`std::this_thread::sleep_for`来添加延迟，而`sleep_until`使用`std::this_thread::sleep_until`。

让我们更仔细地看看`sleep_for`函数。它接受两个参数——`count`和`delay`。第一个参数定义了循环中的迭代次数，第二个参数指定了延迟。我们使用`auto`作为`delay`参数的数据类型，让 C++为我们推断实际的数据类型。

函数体由一个循环组成：

```cpp
  for (int i = 0; i < count; i++) {
```

在每次迭代中，我们运行`delay`并通过在`delay`之前和之后获取时间戳来测量其实际持续时间。`std::this_thread::sleep_for`函数接受时间间隔作为参数：

```cpp
    auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_for(delay);
    auto end = std::chrono::system_clock::now();
```

实际延迟以毫秒为单位测量，我们使用`double`值作为毫秒计数器：

```cpp
std::chrono::duration<double, std::milli> delta = end - start;
```

`wait_until`函数只是稍有不同。它使用`std::current_thred::wait_until`函数，该函数接受一个时间点来唤醒，而不是一个时间间隔。我们引入了一个额外的`wake_up`变量来跟踪唤醒时间点：

```cpp
auto wake_up = std::chrono::system_clock::now();
```

最初，它被设置为当前时间，并在每次迭代中，将作为函数参数传递的延迟添加到其值中：

```cpp
wake_up += delay;
```

函数的其余部分与`sleep_for`实现相同，除了`delay`函数：

```cpp
std::this_thread::sleep_until(wake_up);
```

我们运行两个函数，使用相同数量的迭代和相同的延迟。请注意我们如何使用 C++字符串字面量将毫秒传递给函数，以使代码更易读。为了使用字符串字面量，我们添加了以下内容：

```cpp
sleep_for(10, 100ms);
sleep_until(10, 100ms);
```

这是在函数定义之上完成的，就像这样：

```cpp
using namespace std::chrono_literals;
```

不同的延迟函数会有什么不同吗？毕竟，我们在两种实现中都使用了相同的延迟。让我们运行代码并比较结果：

![](img/f3b5c599-d2cd-4f38-9cdc-a5a908f9ce68.png)

有趣的是，我们可以看到`sleep_for`的所有实际延迟都大于`100`毫秒，而`sleep_until`的一些结果低于这个值。我们的第一个函数`delay_for`没有考虑打印数据到控制台所需的时间。当您确切地知道需要等待多长时间时，`sleep_for`是一个不错的选择。然而，如果您的目标是以特定的周期性唤醒，`sleep_until`可能是一个更好的选择。

# 还有更多...

`sleep_for`和`sleep_until`之间还有其他微妙的差异。系统定时器通常不太精确，并且可能会被时间同步服务（如**网络时间协议** **守护程序**（**ntpd**））调整。这些时钟调整不会影响`sleep_for`，但会影响`sleep_until`。如果您的应用程序依赖于特定时间而不是时间间隔，请使用它；例如，如果您需要每秒重新绘制时钟显示上的数字。

# 使用单调时钟

C++ Chrono 库提供了三种类型的时钟：

+   系统时钟

+   稳定时钟

+   高分辨率时钟

高分辨率时钟通常被实现为系统时钟或稳定时钟的别名。然而，系统时钟和稳定时钟是非常不同的。

系统时钟反映系统时间，因此不是单调的。它可以随时通过时间同步服务（如**网络时间协议**（**NTP**））进行调整，因此甚至可以倒退。

这使得系统时钟成为处理精确持续时间的不良选择。稳定时钟是单调的；它永远不会被调整，也永远不会倒退。这个属性有它的代价——它与挂钟时间无关，通常表示自上次重启以来的时间。

稳定时钟不应该用于需要在重启后保持有效的持久时间戳，例如序列化到文件或保存到数据库。此外，稳定时钟不应该用于涉及来自不同来源的时间的任何时间计算，例如远程系统或外围设备。

在这个示例中，我们将学习如何使用稳定时钟来实现一个简单的软件看门狗。在运行后台工作线程时，重要的是要知道它是否正常工作或因编码错误或无响应的外围设备而挂起。线程定期更新时间戳，而监视例程则将时间戳与当前时间进行比较，如果超过阈值，则执行某种恢复操作。

# 如何做...

在我们的应用程序中，我们将创建一个在后台运行的简单迭代函数，以及在主线程中运行的监视循环。

1.  在您的`~/test`工作目录中，创建一个名为`monotonic`的子目录。

1.  使用您喜欢的文本编辑器在`monotonic`子目录中创建一个`monotonic.cpp`文件。

1.  让我们添加头文件并定义我们例程中使用的全局变量：

```cpp
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>

auto touched = std::chrono::steady_clock::now();
std::mutex m;
std::atomic_bool ready{ false };
```

1.  它们后面是后台工作线程例程的代码：

```cpp
void Worker() {
  for (int i = 0; i < 10; i++) {
    std::this_thread::sleep_for(
         std::chrono::milliseconds(100 + (i % 4) * 10));
    std::cout << "Step " << i << std::endl;
    {
      std::lock_guard<std::mutex> l(m);
      touched = std::chrono::steady_clock::now();
    }
  }
  ready = true;
}
```

1.  添加包含监视例程的`main`函数：

```cpp
int main() {
  std::thread t(Worker);
  std::chrono::milliseconds threshold(120);
  while(!ready) {
    auto now = std::chrono::steady_clock::now();
    std::chrono::milliseconds delta;
    {
      std::lock_guard<std::mutex> l(m);
      auto delta = now - touched;
      if (delta > threshold) {
        std::cout << "Execution threshold exceeded" << std::endl;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  }
  t.join();
  return 0;
}
```

1.  最后，创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(monotonic)
add_executable(monotonic monotonic.cpp)
target_link_libraries(monotonic pthread)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

```

现在可以构建和运行应用程序了。

# 它是如何工作的...

我们的应用程序是多线程的——它由运行监视的主线程和后台工作线程组成。我们使用三个全局变量进行同步。

`touched`变量保存了由`Worker`线程定期更新的时间戳。由于时间戳被两个线程访问，需要进行保护。我们使用一个`m`互斥锁来实现。最后，为了指示工作线程已经完成了它的工作，使用了一个原子变量`ready`。

工作线程是一个包含人为延迟的循环。延迟是基于步骤编号计算的，导致延迟从 100 毫秒到 130 毫秒不等：

```cpp
std::this_thread::sleep_for(
         std::chrono::milliseconds(100 + (i % 4) * 10));
```

在每次迭代中，`Worker`线程更新时间戳。使用锁保护同步访问时间戳：

```cpp
    {
      std::lock_guard<std::mutex> l(m);
      touched = std::chrono::steady_clock::now();
    }
```

监视例程在`Worker`线程运行时循环运行。在每次迭代中，它计算当前时间和上次更新之间的时间间隔：

```cpp
      std::lock_guard<std::mutex> l(m);
      auto delta = now - touched;
```

如果超过阈值，函数会打印警告消息，如下所示：

```cpp
      if (delta > threshold) {
        std::cout << "Execution threshold exceeded" << std::endl;
      }
```

在许多情况下，应用程序可能调用恢复函数来重置外围设备或重新启动线程。我们在监视循环中添加了`10`毫秒的延迟：

```cpp
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
```

这有助于减少资源消耗，同时实现可接受的反应时间。运行应用程序会产生以下输出：

![](img/7962b124-e6ed-45a4-b2da-ffa84adf2d9b.png)

我们可以在输出中看到几个警告，表明`worker`线程中的一些迭代所花费的时间超过了`120`毫秒的阈值。这是可以预料的，因为`worker`函数是这样编写的。重要的是我们用一个单调的`std::chrono::steady_clock`函数进行监视。使用系统时钟可能会导致在时钟调整期间对恢复函数的虚假调用。

# 还有更多...

C++20 定义了几种其他类型的时钟，比如`gps_clock`，表示**全球定位系统**（**GPS**）时间，或者`file_clock`，用于处理文件时间戳。这些时钟可能是稳定的，也可能不是。使用`is_steady`成员函数来检查时钟是否是单调的。

# 使用 POSIX 时间戳

POSIX 时间戳是 Unix 操作系统中时间的传统内部表示。POSIX 时间戳被定义为自纪元以来的秒数，即**协调世界时**（**UTC**）1970 年 1 月 1 日的 00:00:00。

由于其简单性，这种表示在网络协议、文件元数据或序列化中被广泛使用。

在这个示例中，我们将学习如何将 C++时间点转换为 POSIX 时间戳，并从 POSIX 时间戳创建 C++时间点。

# 如何做...

我们将创建一个应用程序，将时间点转换为 POSIX 时间戳，然后从该时间戳中恢复时间点。

1.  在你的`~/test`工作目录中，创建一个名为`timestamps`的子目录。

1.  使用你喜欢的文本编辑器在`timestamps`子目录中创建一个名为`timestamps.cpp`的文件。

1.  将以下代码片段放入文件中：

```cpp
#include <iostream>
#include <chrono>

int main() {
  auto now = std::chrono::system_clock::now();

  std::time_t ts = std::chrono::system_clock::to_time_t(now);
  std::cout << "POSIX timestamp: " << ts << std::endl;

  auto restored = std::chrono::system_clock::from_time_t(ts);

  std::chrono::duration<double, std::milli> delta = now - restored;
  std::cout << "Recovered time delta " << delta.count() << std::endl;
  return 0;
}
```

1.  创建一个包含我们程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(timestamps)
add_executable(timestamps timestamps.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在，你可以构建并运行应用程序。

# 它是如何工作的...

首先，我们使用系统时钟为当前时间创建一个时间点对象：

```cpp
auto now = std::chrono::system_clock::now();
```

由于 POSIX 时间戳表示自纪元以来的时间，我们不能使用稳定时钟。然而，系统时钟知道如何将其内部表示转换为 POSIX 格式。它提供了一个`to_time_t`静态函数来实现这个目的：

```cpp
std::time_t ts = std::chrono::system_clock::to_time_t(now);
```

结果被定义为具有类型`std::time_t`，但这是一个整数类型，而不是对象。与时间点实例不同，我们可以直接将其写入输出流：

```cpp
std::cout << "POSIX timestamp: " << ts << std::endl;
```

让我们尝试从这个整数时间戳中恢复一个时间点。我们使用一个`from_time_t`静态函数：

```cpp
auto restored = std::chrono::system_clock::from_time_t(ts);
```

现在，我们有两个时间戳。它们是相同的吗？让我们计算并显示差异：

```cpp
std::chrono::duration<double, std::milli> delta = now - restored;
std::cout << "Recovered time delta " << delta.count() << std::endl;
```

当我们运行应用程序时，我们会得到以下输出：

![](img/1b9142d4-dd21-4eed-a8f0-d2457fd084f2.png)

时间戳是不同的，但差异始终小于 1,000。由于 POSIX 时间戳被定义为自纪元以来的秒数，我们丢失了毫秒和微秒等细粒度时间。

尽管存在这样的限制，POSIX 时间戳仍然是时间的重要和广泛使用的传输表示，我们学会了如何在需要时将它们转换为内部 C++表示。

# 还有更多...

在许多情况下，直接使用 POSIX 时间戳就足够了。由于它们被表示为数字，可以使用简单的数字比较来决定哪个时间戳更新或更旧。类似地，从一个时间戳中减去另一个时间戳会给出它们之间的秒数时间间隔。如果性能是一个瓶颈，这种方法可能比与本机 C++时间点进行比较更可取。
