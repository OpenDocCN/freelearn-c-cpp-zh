# 第六章：内存管理

内存效率是嵌入式应用的主要要求之一。由于目标嵌入式平台通常具有有限的性能和内存能力，开发人员需要知道如何以最有效的方式使用可用内存。

令人惊讶的是，最有效的方式并不一定意味着使用最少的内存。由于嵌入式系统是专用的，开发人员预先知道将在系统上执行哪些应用程序或组件。在一个应用程序中节省内存并不会带来任何收益，除非同一系统中运行的另一个应用程序可以使用额外的内存。这就是嵌入式系统中内存管理最重要的特征是确定性或可预测性的原因。知道一个应用程序在任何负载下可以使用两兆字节的内存比知道一个应用程序大部分时间可以使用一兆字节的内存，但偶尔可能需要三兆字节更重要得多。

同样，可预测性也适用于内存分配和释放时间。在许多情况下，嵌入式应用更倾向于花费更多内存以实现确定性定时。

在本章中，我们将学习嵌入式应用中广泛使用的几种内存管理技术。本章涵盖的技术如下：

+   使用动态内存分配

+   探索对象池

+   使用环形缓冲区

+   使用共享内存

+   使用专用内存

这些技术将帮助您了解内存管理的最佳实践，并可在处理应用程序中的内存分配时用作构建块。

# 使用动态内存分配

动态内存分配是 C++开发人员常见的做法，在 C++标准库中被广泛使用；然而，在嵌入式系统的环境中，它经常成为难以发现和难以避免的问题的根源。

最显著的问题是时间。内存分配的最坏情况时间是不受限制的；然而，嵌入式系统，特别是那些控制真实世界进程或设备的系统，通常需要在特定时间内做出响应。

另一个问题是碎片化。当分配和释放不同大小的内存块时，会出现技术上是空闲的内存区域，但由于太小而无法分配给应用程序请求。内存碎片随着时间的推移而增加，可能导致内存分配请求失败，尽管总的空闲内存量相当大。

避免这类问题的一个简单而强大的策略是在编译时或启动时预先分配应用程序可能需要的所有内存。然后应用程序根据需要使用这些内存。一旦分配了这些内存，直到应用程序终止，就不会释放这些内存。

这种方法的缺点是应用程序分配的内存比实际使用的内存多，而不是让其他应用程序使用它。在实践中，这对于嵌入式应用来说并不是问题，因为它们在受控环境中运行，所有应用程序及其内存需求都是预先知道的。

# 如何做到...

在本技术中，我们将学习如何预先分配内存并在应用程序中使用它：

1.  在您的工作`〜/test`目录中，创建一个名为`prealloc`的子目录。

1.  使用您喜欢的文本编辑器在`prealloc`子目录中创建一个名为`prealloc.cpp`的文件。将以下代码片段复制到`prealloc.cpp`文件中以定义`SerialDevice`类：

```cpp
#include <cstdint>
#include <string.h>

constexpr size_t kMaxFileNameSize = 256;
constexpr size_t kBufferSize = 4096;
constexpr size_t kMaxDevices = 16;

class SerialDevice {
    char device_file_name[256];
    uint8_t input_buffer[kBufferSize];
    uint8_t output_buffer[kBufferSize];
    int file_descriptor;
    size_t input_length;
    size_t output_length;

  public:
    SerialDevice():
      file_descriptor(-1), input_length(0), output_length(0) {}

    bool Init(const char* name) {
      strncpy(device_file_name, name, sizeof(device_file_name));
    }

    bool Write(const uint8_t* data, size_t size) {
      if (size > sizeof(output_buffer)) {
        throw "Data size exceeds the limit";
      }
      memcpy(output_buffer, data, size);
    }

    size_t Read(uint8_t* data, size_t size) {
      if (size < input_length) {
        throw "Read buffer is too small";
      }
      memcpy(data, input_buffer, input_length);
      return input_length;
    }
};
```

1.  添加使用`SerialDevice`类的`main`函数：

```cpp
int main() {
  SerialDevice devices[kMaxDevices];
  size_t number_of_devices = 0;

  uint8_t data[] = "Hello";
  devices[0].Init("test");
  devices[0].Write(data, sizeof(data));
  number_of_devices = 1;

  return 0;
}
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(prealloc)
add_executable(prealloc prealloc.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++17")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在可以构建和运行应用程序。它不会输出任何数据，因为它的目的是演示我们如何预先分配内存，而不知道设备的数量和我们与设备交换的消息的大小。

# 工作原理...

在这个配方中，我们定义了封装与串行设备进行数据交换的对象。设备由可变长度的设备文件名字符串标识。我们可以向设备发送和接收可变长度的消息。

由于我们只能在运行时发现连接到系统的设备数量，我们可能会在发现时创建设备对象。同样，由于我们不知道发送和接收的消息大小，因此自然而然地要动态分配消息的内存。

相反，我们预分配未初始化设备对象的数组：

```cpp
  SerialDevice devices[kMaxDevices];
```

反过来，每个对象都预分配了足够的内存来存储消息和设备文件名：

```cpp
  char device_file_name[kMaxFileNameSize];
  uint8_t input_buffer[kBufferSize];
  uint8_t output_buffer[kBufferSize];
```

我们使用局部变量来跟踪输入和输出缓冲区中数据的实际大小。无需跟踪文件名的大小，因为预期它是以零结尾的：

```cpp
  size_t input_length;
  size_t output_length;
```

同样，我们跟踪实际发现的设备数量：

```cpp
  size_t number_of_devices = 0;
```

通过这种方式，我们避免了动态内存分配。尽管这样做有成本：我们人为地限制了支持的最大设备数量和消息的最大大小。其次，大量分配的内存从未被使用。例如，如果我们支持最多 16 个设备，而系统中只有 1 个设备，那么实际上我们只使用了分配内存的 1/16。如前所述，这对于嵌入式系统来说并不是问题，因为所有应用程序及其要求都是预定义的。没有应用程序可以从它可以分配的额外内存中受益。

# 探索对象池

正如我们在本章的第一个配方中讨论的那样，预分配应用程序使用的所有内存是一种有效的策略，有助于嵌入式应用程序避免与内存碎片化和分配时间相关的各种问题。

临时内存预分配的一个缺点是，应用程序现在负责跟踪预分配对象的使用情况。

对象池旨在通过提供类似于动态内存分配但使用预分配数组中的对象的泛化和便利接口来隐藏对象跟踪的负担。

# 如何做...

在这个配方中，我们将创建一个对象池的简单实现，并学习如何在应用程序中使用它：

1.  在您的工作`~/test`目录中，创建一个名为`objpool`的子目录。

1.  使用您喜欢的文本编辑器在`objpool`子目录中创建一个`objpool.cpp`文件。让我们定义一个模板化的`ObjectPool`类。我们从私有数据成员和构造函数开始：

```cpp
#include <iostream>

template<class T, size_t N>
class ObjectPool {
  private:
    T objects[N];
    size_t available[N];
    size_t top = 0;
  public:
    ObjectPool(): top(0) {
      for (size_t i = 0; i < N; i++) {
        available[i] = i;
      }
    }
```

1.  现在让我们添加一个从池中获取元素的方法：

```cpp
    T& get() {
      if (top < N) {
        size_t idx = available[top++];
        return objects[idx];
      } else {
        throw std::runtime_error("All objects are in use");
      }
    }
```

1.  接下来，我们添加一个将元素返回到池中的方法：

```cpp
    void free(const T& obj) {
      const T* ptr = &obj;
      size_t idx = (ptr - objects) / sizeof(T);
      if (idx < N) {
        if (top) {
          top--;
          available[top] = idx;
        } else {
          throw std::runtime_error("Some object was freed more than once");
        }
      } else {
        throw std::runtime_error("Freeing object that does not belong to
       the pool");
      }
     }
```

1.  然后，用一个小函数包装类定义，该函数返回从池中请求的元素数量：

```cpp
    size_t requested() const { return top; }
    };
```

1.  按照以下代码所示定义要存储在对象池中的数据类型：

```cpp
struct Point {
  int x, y;
};
```

1.  然后添加与对象池一起工作的代码：

```cpp
int main() {
  ObjectPool<Point, 10> points;

  Point& a = points.get();
  a.x = 10; a.y=20;
  std::cout << "Point a (" << a.x << ", " << a.y << ") initialized, requested "        <<
    points.requested() << std::endl;

  Point& b = points.get();
  std::cout << "Point b (" << b.x << ", " << b.y << ") not initialized, requested " <<
    points.requested() << std::endl;

  points.free(a);
  std::cout << "Point a(" << a.x << ", " << a.y << ") returned, requested " <<
    points.requested() << std::endl;

  Point& c = points.get();
  std::cout << "Point c(" << c.x << ", " << c.y << ") not intialized, requested " <<
    points.requested() << std::endl;

  Point local;
  try {
    points.free(local);
  } catch (std::runtime_error e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
  }
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(objpool)
add_executable(objpool objpool.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  构建应用程序并将生成的可执行二进制文件复制到目标系统。使用第二章中的配方，*设置环境*来完成。

1.  切换到目标系统终端。如果需要，使用用户凭据登录。

1.  运行二进制文件。

# 工作原理...

在这个应用程序中，我们使用了与第一个配方中相同的想法（预分配对象的静态数组），但是我们将其封装到一个模板化的`ObjectPool`类中，以提供处理不同类型对象的通用接口。

我们的模板有两个参数——存储在`ObjectPool`类实例中的对象的类或数据类型，以及池的大小。这些参数用于定义类的两个私有数据字段——对象数组和空闲索引数组：

```cpp
     T objects[N];
     size_t available[N];
```

由于模板参数在编译时被解析，这些数组是静态分配的。此外，该类有一个名为`top`的私有数据成员，它充当`available`数组中的索引，并指向下一个可用对象。

可用数组包含当前可用于使用的`objects`数组中所有对象的索引。在最开始，所有对象都是空闲的，并且可用数组中填充了对象数组中所有元素的索引：

```cpp
      for (size_t i = 0; i < N; i++) {
        available[i] = i;
      }
```

当应用程序需要从池中获取元素时，它调用`get`方法。该方法使用顶部变量来获取池中下一个可用元素的索引：

```cpp
      size_t idx = available[top++];
      return objects[idx];
```

当`top`索引达到数组大小时，意味着不能再分配更多元素，因此该方法会抛出异常以指示错误条件：

```cpp
      throw std::runtime_error("All objects are in use");
```

可以使用`free`将对象返回到池中。首先，它根据其地址检测元素的索引。索引被计算为对象地址与池起始地址的差异。由于池对象在内存中是连续存储的，我们可以轻松地过滤出相同类型的对象，但不能过滤出来自该池的对象：

```cpp
      const T* ptr = &obj;
      size_t idx = (ptr - objects) / sizeof(T);
```

请注意，由于`size_t`类型是无符号的，我们不需要检查结果索引是否小于零——这是不可能的。如果我们尝试将不属于池的对象返回到池中，并且其地址小于池的起始地址，它将被视为正索引。

如果我们返回的对象属于池，我们会更新顶部计数器，并将结果索引放入可用数组以供进一步使用：

```cpp
  top--;
  available[top] = idx;
```

否则，我们会抛出异常，指示我们试图返回一个不属于该池的对象：

```cpp
     throw std::runtime_error("Freeing object that does not belong to the pool");
```

所请求的方法用于跟踪池对象的使用情况。它返回顶部变量，该变量有效地跟踪已经被索取但尚未返回到池中的对象数量。

```cpp
     size_t requested() const { return top; }
```

让我们定义一个数据类型并尝试使用来自池的对象。我们声明一个名为`Point`的结构体，其中包含两个`int`字段，如下面的代码所示：

```cpp
 struct Point {
  int x, y;
 };
```

现在我们创建一个大小为`10`的`Point`对象池：

```cpp
    ObjectPool<Point, 10> points;
```

我们从池中获取一个对象并填充其数据字段：

```cpp
 Point& a = points.get();
 a.x = 10; a.y=20;
```

程序产生了以下输出：

![](img/aac88c6f-a95e-44b3-8e8c-3173dac428a9.png)

输出的第一行报告了一个请求的对象。

我们请求了一个额外的对象并打印其数据字段，而不进行任何初始化。池报告说已经请求了两个对象，这是预期的。

现在我们将第一个对象返回到池中，并确保请求的对象数量减少。我们还可以注意到，即使将对象返回到池中，我们仍然可以从中读取数据。

让我们从池中再索取一个对象。请求的数量增加，但请求的对象与我们在上一步中返回的对象相同。

我们可以看到`Point c`在从池中取出后没有被初始化，但其字段包含与`Point a`相同的值。实际上，现在`a`和`c`是对池中相同对象的引用，因此对变量`a`的修改将影响变量`c`。这是我们对象池实现的一个限制。

最后，我们创建一个本地的`Point`对象并尝试将其返回到池中：

```cpp
  Point local;
  try {
    points.free(local);
  } catch (std::runtime_error e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
```

预计会出现异常，并且确实如此。在程序输出中，您可以看到一个`Exception caught: Freeing object that does not belong to the pool`的消息。

# 还有更多...

尽管对象池的实现简化了与预分配对象的工作，但它有许多限制。

首先，所有对象都是在最开始创建的。因此，调用我们的池的`get`方法不会触发对象构造函数，调用`free`方法也不会调用析构函数。开发人员需要使用各种变通方法来初始化和去初始化对象。

一个可能的解决方法是定义目标对象的特殊方法，比如`initialize`和`deinitialize`，分别由`ObjectPool`类的`get`和`free`方法调用。然而，这种方法将类的实现与`ObjectPool`的实现耦合在一起。在本章的后面，我们将看到更高级的技术来克服这个限制。

我们的池的实现没有检测`free`方法是否对一个对象调用了多次。这是一个错误，但是很常见，并导致难以调试的问题。虽然在技术上是可行的，但它给实现增加了不必要的额外复杂性。

# 使用环形缓冲区

环形缓冲区，或循环缓冲区，在嵌入式世界中是一个广泛使用的数据结构。它作为一个队列放置在固定大小的内存数组之上。缓冲区可以包含固定数量的元素。生成这些元素的函数将它们顺序放入缓冲区中。当达到缓冲区的末尾时，它会切换到缓冲区的开头，就好像它的第一个元素跟在最后一个元素后面。

当涉及到组织数据生产者和消费者之间的数据交换时，这种设计被证明是非常高效的，因为它们是独立的，不能等待对方，这在嵌入式开发中是常见的情况。例如，中断服务例程应该快速地将来自设备的数据排队等待进一步处理，而中断被禁用。如果处理数据的函数落后，它不能等待中断服务例程。同时，处理函数不需要完全与**中断服务例程**（**ISR**）同步；它可以一次处理多个元素，并在稍后赶上 ISR。

这个特性，以及它们可以在静态情况下预先分配，使得环形缓冲区在许多情况下成为最佳选择。

# 如何做...

在这个示例中，我们将学习如何在 C++数组之上创建和使用环形缓冲区：

1.  在您的工作`~/test`目录中，创建一个名为`ringbuf`的子目录。

1.  使用您喜欢的文本编辑器在`ringbuf`子目录中创建一个`ringbuf.cpp`文件。

1.  从`private`数据字段开始定义`RingBuffer`类。

```cpp
#include <iostream>

template<class T, size_t N>
class RingBuffer {
  private:
    T objects[N];
    size_t read;
    size_t write;
    size_t queued;
  public:
    RingBuffer(): read(0), write(0), queued(0) {}
```

1.  现在我们添加一个将数据推送到缓冲区的方法：

```cpp
    T& push() {
      T& current = objects[write];
      write = (write + 1) % N;
      queued++;
      if (queued > N) {
        queued = N;
        read = write;
      }
      return current;
    }

```

1.  接下来，我们添加一个从缓冲区中拉取数据的方法：

```cpp
    const T& pull() {
      if (!queued) {
        throw std::runtime_error("No data in the ring buffer");
      }
      T& current = objects[read];
      read = (read + 1) % N;
      queued--;
      return current;
    }
```

1.  让我们添加一个小方法来检查缓冲区是否包含任何数据，并完成类的定义：

```cpp
bool has_data() {
  return queued != 0;
}
};
```

1.  有了`RingBuffer`的定义，我们现在可以添加使用它的代码了。首先，让我们定义我们将要使用的数据类型：

```cpp
struct Frame {
  uint32_t index;
  uint8_t data[1024];
};
```

1.  其次，添加`main`函数，并定义`RingBuffer`的一个实例作为其变量，以及尝试使用空缓冲区的代码：

```cpp
int main() {
  RingBuffer<Frame, 10> frames;

  std::cout << "Frames " << (frames.has_data() ? "" : "do not ")
      << "contain data" << std::endl;
  try {
    const Frame& frame = frames.pull();
  } catch (std::runtime_error e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
```

1.  接下来，添加使用缓冲区中五个元素的代码：

```cpp
for (size_t i = 0; i < 5; i++) {
Frame& out = frames.push();
out.index = i;
out.data[0] = 'a' + i;
out.data[1] = '\0';
  }
std::cout << "Frames " << (frames.has_data() ? "" : "do not ")
<< "contain data" << std::endl;
while (frames.has_data()) {
const Frame& in = frames.pull();
    std::cout << "Frame " << in.index << ": " << in.data << std::endl;
  }
```

1.  之后，添加类似的代码，处理可以添加的更多元素的情况：

```cpp
    for (size_t i = 0; i < 26; i++) {
    Frame& out = frames.push();
    out.index = i;
    out.data[0] = 'a' + i;
    out.data[1] = '\0';
    }
    std::cout << "Frames " << (frames.has_data() ? "" : "do not ")
      << "contain data" << std::endl;
    while (frames.has_data()) {
    const Frame& in = frames.pull();
    std::cout << "Frame " << in.index << ": " << in.data << std::endl;
    }
    }
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(ringbuf)
add_executable(ringbuf ringbuf.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  构建应用程序，并将生成的可执行二进制文件复制到目标系统。使用第二章中的示例，*设置环境*。

1.  切换到目标系统终端。如果需要，使用用户凭据登录。

1.  运行二进制文件。

# 它是如何工作的...

我们将我们的环形缓冲区实现为一个模板化的 C++类，它有三个私有数据字段：

+   `objects`: 类型为`T`的`N`个元素的静态数组

+   `read`: 一个用于读取元素的索引

+   `write`: 用于写入元素的索引

`RingBuffer`类公开了三个公共方法：

+   `push()`: 将数据写入缓冲区

+   `pull()`: 从缓冲区中读取数据

+   `has_data()`: 检查缓冲区是否包含数据

让我们仔细看看它们是如何工作的。

`push()`方法旨在被函数用于将数据存储在缓冲区中。与动态队列或动态栈的类似`push()`方法不同，后者接受一个要存储的值作为参数，我们的实现不接受任何参数。由于所有元素在编译时都是预分配的，它返回对要更新的缓冲区中的值的引用。

`push()`方法的实现很简单；它通过`write`索引获取对元素的指针，然后推进`write`索引并增加存储在缓冲区中的元素数量。请注意，取模运算符用于在`write`索引达到大小限制时将其包装到数组的开头：

```cpp
T& current = objects[write];
write = (write + 1) % N;
queued++;
```

如果我们尝试推送的元素数量超过`objects`数组的容量处理能力会发生什么？这取决于我们计划存储在缓冲区中的数据的性质。在我们的实现中，我们假设接收方对最近的数据感兴趣，并且如果它无法赶上发送方，则可以容忍中间数据的丢失。如果接收方太慢，那么在接收方`read`数据之前发送方运行了多少圈都无所谓：在这一点上超过`N`步的所有数据都被覆盖。这就是为什么一旦存储的元素数量超过`N`，我们开始推进`read`索引以及`write`索引，使它们确切地相隔`N`步：

```cpp
 if (queued > N) {
  queued = N;
  read = write;
 }
```

`pull()`方法由从缓冲区读取数据的函数使用。与`push()`方法类似，它不接受任何参数，并返回对缓冲区中元素的引用。不过，与`push()`方法不同的是，它返回一个常量引用（如下面的代码所示），以表明它不应该修改缓冲区中的数据：

```cpp
 const T& pull() {
```

首先，它检查缓冲区中是否有数据，并且如果缓冲区不包含元素，则抛出异常：

```cpp
  if (!queued) {
   throw std::runtime_error("No data in the ring buffer");
  }
```

它通过读取索引获取对元素的引用，然后推进`read`索引，应用与`push()`方法为`write`索引所做的相同的取模运算符：

```cpp
  read = (read + 1) % N;
  queued--;
```

`has_data()`方法的实现是微不足道的。如果对象计数为零，则返回`false`，否则返回`true`：

```cpp
  bool has_data() {
  return queued != 0;
  }
```

现在，让我们尝试实际操作。我们声明一个简单的数据结构`Frame`，模拟设备生成的数据。它包含一个帧索引和一个不透明的数据缓冲区：

```cpp
  uint32_t index;
  uint8_t data[1024];
  };
```

我们定义了一个容量为`10`个`frame`类型元素的环形缓冲区：

```cpp
  RingBuffer<Frame, 10> frames;
```

让我们来看看程序的输出：

![](img/45ab92d8-96c5-42ce-aee0-b49bd991217a.png)

首先，我们尝试从空缓冲区中读取并得到一个异常，这是预期的。

然后，我们将五个元素写入缓冲区，使用拉丁字母表的字符作为数据载荷：

```cpp
  for (size_t i = 0; i < 5; i++) {
    Frame& out = frames.push();
    out.index = i;
    out.data[0] = 'a' + i;
    out.data[1] = '\0';
  }
```

注意我们如何获取对元素的引用，然后在原地更新它，而不是将`frame`的本地副本推入环形缓冲区。然后我们读取缓冲区中的所有数据并将其打印在屏幕上：

```cpp
  while (frames.has_data()) {
    const Frame& in = frames.pull();
    std::cout << "Frame " << in.index << ": " << in.data << std::endl;
  }
```

程序输出表明我们可以成功读取所有五个元素。现在我们尝试将拉丁字母表的所有 26 个字母写入数组，远远超过其容量。

```cpp
 for (size_t i = 0; i < 26; i++) {
    Frame& out = frames.push();
    out.index = i;
    out.data[0] = 'a' + i;
    out.data[1] = '\0';
  }
```

然后我们以与五个元素相同的方式读取数据。读取是成功的，但我们只收到了最后写入的 10 个元素；所有其他帧都已丢失并被覆盖。对于我们的示例应用程序来说这并不重要，但对于许多其他应用程序来说可能是不可接受的。确保数据不会丢失的最佳方法是保证接收方的激活频率高于发送方。有时，如果缓冲区中没有可用数据，接收方将被激活，但这是为了避免数据丢失而可以接受的代价。

# 使用共享内存

在运行在支持**MMU**（内存管理单元）的硬件上的现代操作系统中，每个应用程序作为一个进程运行，并且其内存与其他应用程序隔离。

这种隔离带来了重要的可靠性优势。一个应用程序不能意外地破坏另一个应用程序的内存。同样，一个意外破坏自己内存并崩溃的应用程序可以被操作系统关闭，而不会影响系统中的其他应用程序。将嵌入式系统的功能解耦为几个相互通信的隔离应用程序，通过一个明确定义的 API 显著减少了实现的复杂性，从而提高了稳定性。

然而，隔离会产生成本。由于每个进程都有自己独立的地址空间，两个应用程序之间的数据交换意味着数据复制、上下文切换和使用操作系统内核同步机制，这可能是相对昂贵的。

共享内存是许多操作系统提供的一种机制，用于声明某些内存区域为共享。这样，应用程序可以在不复制数据的情况下交换数据。这对于交换大型数据对象（如视频帧或音频样本）尤为重要。

# 如何做...

在这个示例中，我们将学习如何使用 Linux 共享内存 API 在两个或多个应用程序之间进行数据交换。

1.  在您的工作`~/test`目录中，创建一个名为`shmem`的子目录。

1.  使用您喜欢的文本编辑器在`shmem`子目录中创建一个`shmem.cpp`文件。从常见的头文件和常量开始定义`SharedMem`类：

```cpp
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

const char* kSharedMemPath = "/sample_point";
const size_t kPayloadSize = 16;

using namespace std::literals;

template<class T>
class SharedMem {
  int fd;
  T* ptr;
  const char* name;

  public:
```

1.  然后，定义一个大部分工作的构造函数：

```cpp
SharedMem(const char* name, bool owner=false) {
fd = shm_open(name, O_RDWR | O_CREAT, 0600);
if (fd == -1) {
throw std::runtime_error("Failed to open a shared memory region");
}
if (ftruncate(fd, sizeof(T)) < 0) {
close(fd);
throw std::runtime_error("Failed to set size of a shared memory 
region");
};
ptr = (T*)mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, 
MAP_SHARED, fd, 0);
if (!ptr) {
close(fd);
    throw std::runtime_error("Failed to mmap a shared memory region");
}
    this->name = owner ? name : nullptr;
    std::cout << "Opened shared mem instance " << name << std::endl;
}
```

1.  添加析构函数的定义：

```cpp
    ~SharedMem() {
      munmap(ptr, sizeof(T));
      close(fd);
      if (name) {
        std::cout << "Remove shared mem instance " << name << std::endl;
        shm_unlink(name);
      }
      }
```

1.  用一个小方法来完成类定义，返回一个对共享对象的引用：

```cpp
    T& get() const {
      return *ptr;
    }
    };
```

1.  我们的`SharedMem`类可以处理不同的数据类型。让我们声明一个自定义数据结构，我们想要使用：

```cpp
struct Payload {
  uint32_t index;
  uint8_t raw[kPayloadSize];
};
```

1.  现在添加代码，将数据写入共享内存：

```cpp
void producer() {
  SharedMem<Payload> writer(kSharedMemPath);
  Payload& pw = writer.get();
  for (int i = 0; i < 5; i++) {
    pw.index = i;
    std::fill_n(pw.raw, sizeof(pw.raw) - 1, 'a' + i);
    pw.raw[sizeof(pw.raw) - 1] = '\0';
    std::this_thread::sleep_for(150ms);
  }
}
```

1.  还要添加从共享内存中读取数据的代码：

```cpp
void consumer() {
  SharedMem<Payload> point_reader(kSharedMemPath, true);
  Payload& pr = point_reader.get();
  for (int i = 0; i < 10; i++) {
    std::cout << "Read data frame " << pr.index << ": " << pr.raw << std::endl;
    std::this_thread::sleep_for(100ms);
  }
  }
```

1.  添加`main`函数，将所有内容联系在一起，如下面的代码所示：

```cpp
int main() {

  if (fork()) {
    consumer();
  } else {
    producer();
  }
  }
```

1.  在`loop`子目录中创建一个名为`CMakeLists.txt`的文件，内容如下：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(shmem)
add_executable(shmem shmem.cpp)
target_link_libraries(shmem rt)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  构建应用程序，并将生成的可执行二进制文件复制到目标系统。使用第二章中的设置环境的方法来完成。

1.  切换到目标系统终端。如果需要，使用用户凭据登录。

1.  运行二进制文件。

# 它是如何工作的...

在这个示例中，我们使用**POSIX**（**可移植操作系统接口**的缩写）API 来处理共享内存。这是一个灵活和细粒度的 C API，有很多可以调整或配置的参数。我们的目标是通过在其上实现一个更方便和类型安全的 C++包装器来隐藏这个低级 API 的复杂性。我们将使用**RAII**（**资源获取即初始化**的缩写）习惯，以确保所有分配的资源都得到适当的释放，我们的应用程序中没有内存或文件描述符泄漏。

我们定义了一个模板化的`SharedMem`类。模板参数定义了存储在我们的共享内存实例中的数据类型。这样，我们使`SharedMem`类的实例类型安全。我们不再需要在应用程序代码中使用 void 指针和类型转换，C++编译器会自动为我们完成：

```cpp
template<class T>
class SharedMem {
```

所有共享内存分配和初始化都在`SharedMem`构造函数中实现。它接受两个参数：

+   一个共享内存对象名称

+   一个所有权标志

POSIX 定义了一个`shm_open`API，其中共享内存对象由名称标识，类似于文件名。这样，使用相同名称的两个独立进程可以引用相同的共享内存对象。共享对象的生命周期是什么？当为相同的对象名称调用`shm_unlink`函数时，共享对象被销毁。如果对象被多个进程使用，第一个调用`shm_open`的进程将创建它，其他进程将重用相同的对象。但是它们中的哪一个负责删除它？这就是所有权标志的用途。当设置为`true`时，它表示`SharedMem`实例在销毁时负责共享对象的清理。

构造函数依次调用三个 POSIX API 函数。首先，它使用`shm_open`创建一个共享对象。虽然该函数接受访问标志和文件权限作为参数，但我们总是使用读写访问模式和当前用户的读写访问权限：

```cpp
fd = shm_open(name, O_RDWR | O_CREAT, 0600);
```

接下来，我们使用`ftruncate`调用定义共享区域的大小。我们使用模板数据类型的大小来实现这个目的：

```cpp
if (ftruncate(fd, sizeof(T)) < 0) {
```

最后，我们使用`mmap`函数将共享区域映射到我们的进程内存地址空间。它返回一个指针，我们可以用来引用我们的数据实例：

```cpp
ptr = (T*)mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```

该对象将文件描述符和内存区域的指针保存为其私有成员。析构函数在对象被销毁时对它们进行释放。如果设置了所有者标志，我们还保留对象名称，以便我们可以删除它：

```cpp
int fd;
T* ptr;
const char* name;
```

`SharedMem`析构函数将共享内存对象从地址空间中取消映射：

```cpp
 munmap(ptr, sizeof(T));
```

如果对象是所有者，我们可以使用`shm_unlink`调用来删除它。请注意，自从名称设置为`nullptr`后，我们不再需要所有者标志，除非对象是所有者：

```cpp
 if (name) {
   std::cout << "Remove shared mem instance " << name << std::endl;
 shm_unlink(name);
 }
```

为了访问共享数据，该类提供了一个简单的`get`方法。它返回存储在共享内存中的对象的引用：

```cpp
  T& get() const {
      return *ptr;
  }
```

让我们创建两个使用我们创建的共享内存 API 的独立进程。我们使用 POSIX 的`fork`函数来生成一个子进程。子进程将是数据生产者，父进程将是数据消费者：

```cpp
  if (fork()) {
    consumer();
  } else {
    producer();
  }
```

我们定义了一个`Payload`数据类型，生产者和消费者都用于数据交换：

```cpp
  struct Payload {
  uint32_t index;
  uint8_t raw[kPayloadSize];
  };
```

数据生产者创建一个`SharedMem`实例：

```cpp
  SharedMem<Payload> writer(kSharedMemPath);
```

它使用`get`方法接收的引用每 150 毫秒更新一次共享对象。每次，它增加有效载荷的索引字段，并用与索引匹配的拉丁字母填充其数据。

消费者和生产者一样简单。它创建一个与生产者同名的`SharedMem`实例，但它声明了对该对象的所有权。这意味着它将负责删除它，如下面的代码所示：

```cpp
  SharedMem<Payload> point_reader(kSharedMemPath, true);
```

运行应用程序并观察以下输出：

![](img/714b3428-c62d-4794-b090-b8a3bd2a72ee.png)

每 100 毫秒，应用程序从共享对象中读取数据并将其打印到屏幕上。在消费者输出中，我们可以看到它接收到了生产者写入的数据。由于消费者和生产者周期的持续时间不匹配，我们可以看到有时相同的数据被读取两次

在这个例子中故意省略的逻辑的一个重要部分是生产者和消费者的同步。由于它们作为独立的项目运行，不能保证生产者在消费者尝试读取数据时已经更新了任何数据。以下是我们在结果输出中看到的内容：

```cpp
Opened shared mem instance /sample_point
Read data frame 0: 
Opened shared mem instance /sample_point
```

我们可以看到，在生产者打开相同的对象之前，消费者打开了共享内存对象并读取了一些数据。

同样，当消费者尝试读取数据时，无法保证生产者是否完全更新数据字段。我们将在下一章中更详细地讨论这个话题。

# 还有更多...

共享内存本身是一种快速高效的进程间通信机制，但当与环形缓冲区结合时，它真正发挥作用。通过将环形缓冲区放入共享内存中，开发人员可以允许独立的数据生产者和数据消费者异步交换数据，并且同步的开销很小。

# 使用专用内存

嵌入式系统通常通过特定的内存地址范围提供对其外围设备的访问。当程序访问这个区域中的地址时，它不会读取或写入内存中的值。相反，数据被发送到该地址映射的设备或从该设备读取。

这种技术通常被称为**MMIO**（内存映射输入/输出）。在这个教程中，我们将学习如何从用户空间的 Linux 应用程序中使用 MMIO 访问 Raspberry PI 的外围设备。

# 如何做...

Raspberry PI 有许多外围设备可以通过 MMIO 访问。为了演示 MMIO 的工作原理，我们的应用程序将访问系统定时器：

1.  在您的工作`~/test`目录中，创建一个名为`timer`的子目录。

1.  使用您最喜欢的文本编辑器在`timer`子目录中创建名为`timer.cpp`的文件。

1.  将所需的头文件、常量和类型声明放入`timer.cpp`中：

```cpp
#include <iostream>
#include <chrono>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>

constexpr uint32_t kTimerBase = 0x3F003000;

struct SystemTimer {
  uint32_t CS;
  uint32_t counter_lo;
  uint32_t counter_hi;
};
```

1.  添加`main`函数，其中包含程序的所有逻辑：

```cpp
int main() {

  int memfd = open("/dev/mem", O_RDWR | O_SYNC);
  if (memfd < 0) {
  throw std::system_error(errno, std::generic_category(),
  "Failed to open /dev/mem. Make sure you run as root.");
  }

  SystemTimer *timer = (SystemTimer*)mmap(NULL, sizeof(SystemTimer),
  PROT_READ|PROT_WRITE, MAP_SHARED,
  memfd, kTimerBase);
  if (timer == MAP_FAILED) {
  throw std::system_error(errno, std::generic_category(),
  "Memory mapping failed");
  }

  uint64_t prev = 0;
  for (int i = 0; i < 10; i++) {
   uint64_t time = ((uint64_t)timer->counter_hi << 32) + timer->counter_lo;
   std::cout << "System timer: " << time;
   if (i > 0) {
   std::cout << ", diff " << time - prev;
    }
    prev = time;
    std::cout << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
 }
```

1.  在`timer`子目录中创建一个名为`CMakeLists.txt`的文件，并包含以下内容：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(timer)
add_executable(timer timer.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建和运行应用程序了。

请注意，它应该在真正的 Raspberry PI 3 设备上以`root`身份运行。

# 它是如何工作的...

系统定时器是一个外围设备，通过 MMIO 接口连接到处理器。这意味着它有一系列专用的物理地址，每个地址都有特定的格式和用途。

我们的应用程序使用两个 32 位值表示的计时器计数器。组合在一起，它们形成一个 64 位的只读计数器，在系统运行时始终递增。

对于 Raspberry PI 3，为系统定时器分配的物理内存地址范围的偏移量为`0x3F003000`（根据 Raspberry PI 硬件版本的不同可能会有所不同）。我们将其定义为一个常量。

```cpp
constexpr uint32_t kTimerBase = 0x3F003000;
```

为了访问区域内的各个字段，我们定义了一个`SystemTimer`结构：

```cpp
struct SystemTimer {
  uint32_t CS;
  uint32_t counter_lo;
  uint32_t counter_hi;
};
```

现在，我们需要获取指向定时器地址范围的指针，并将其转换为指向`SystemTimer`的指针。这样，我们就可以通过读取`SystemTimer`的数据字段来访问计数器的地址。

然而，我们需要解决一个问题。我们知道物理地址空间中的偏移量，但我们的 Linux 应用程序在虚拟地址空间中运行。我们需要找到一种将物理地址映射到虚拟地址的方法。

Linux 通过特殊的`/proc/mem`文件提供对物理内存地址的访问。由于它包含所有物理内存的快照，因此只能由`root`访问。

我们使用`open`函数将其作为常规文件打开：

```cpp
int memfd = open("/dev/mem", O_RDWR | O_SYNC);
```

一旦文件打开并且我们知道它的描述符，我们就可以将其映射到我们的虚拟地址空间中。我们不需要映射整个物理内存。与定时器相关的区域就足够了，这就是为什么我们将系统定时器范围的起始位置作为偏移参数传递，将`SystemTimer`结构的大小作为大小参数传递：

```cpp
SystemTimer *timer = (SystemTimer*)mmap(NULL, sizeof(SystemTimer),
PROT_READ|PROT_WRITE, MAP_SHARED, memfd, kTimerBase);
```

现在我们可以访问定时器字段了。我们在循环中读取定时器计数器，并显示其当前值及其与前一个值的差异。当我们以`root`身份运行我们的应用程序时，我们会得到以下输出：

![](img/aa941e90-c2ed-49d6-a79c-c813bc3b95aa.png)

正如我们所看到的，从这个内存地址读取返回递增的值。差值的值大约为 10,000，而且非常恒定。由于我们在计数器读取循环中添加了 10 毫秒的延迟，我们可以推断这个内存地址与定时器相关，而不是常规内存，定时器计数器的粒度为 1 微秒。

# 还有更多...

树莓派有许多外围设备可以通过 MMIO 访问。您可以在*BCM2835 ARM 外围设备手册*中找到关于它们的地址范围和访问语义的详细信息，该手册可在[`www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2835/BCM2835-ARM-Peripherals.pdf`](https://www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2835/BCM2835-ARM-Peripherals.pdf)上找到。

请注意，开发人员在处理可以同时被多个设备访问的内存时必须非常小心。当内存可以被多个处理器或同一处理器的多个核心访问时，您可能需要使用高级同步技术，如内存屏障，以避免同步问题。我们将在下一章讨论其中一些技术。如果您使用直接内存访问（DMA）或 MMIO，情况会变得更加复杂。由于 CPU 可能不知道内存被外部硬件更改，其缓存可能不同步，导致数据一致性问题。
