# 错误处理和容错

难以高估嵌入式软件中错误处理的重要性。嵌入式系统应该在各种物理条件下无需监督地工作，例如控制可能故障或不总是提供可靠通信线路的外部外围设备。在许多情况下，系统的故障要么很昂贵，要么很不安全。

在本章中，我们将学习有助于编写可靠和容错的嵌入式应用程序的常见策略和最佳实践。

我们将在本章中介绍以下食谱：

+   处理错误代码

+   使用异常处理错误

+   在捕获异常时使用常量引用

+   解决静态对象

+   处理看门狗

+   探索高可用系统的心跳

+   实现软件去抖动逻辑

这些食谱将帮助您了解错误处理设计的重要性，学习最佳实践，并避免在此领域出现问题。

# 处理错误代码

在设计新函数时，开发人员经常需要一种机制来指示函数无法完成其工作，因为出现了某种错误。这可能是无效的，从外围设备接收到意外结果，或者是资源分配问题。

报告错误条件的最传统和广泛使用的方法之一是通过错误代码。这是一种高效且无处不在的机制，不依赖于编程语言或操作系统。由于其效率、多功能性和跨各种平台边界的能力，它在嵌入式软件开发中被广泛使用。

设计一个既返回值又返回错误代码的函数接口可能会很棘手，特别是如果值和错误代码具有不同的类型。在这个食谱中，我们将探讨设计这种类型的函数接口的几种方法。

# 操作步骤...

我们将创建一个简单的程序，其中包含一个名为`Receive`的函数的三个实现。所有三个实现都具有相同的行为，但接口不同。按照以下步骤进行：

1.  在您的工作目录`~/test`中，创建一个名为`errcode`的子目录。

1.  使用您喜欢的文本编辑器在`errcode`子目录中创建一个名为`errcode.cpp`的文件。

1.  将第一个函数的实现添加到`errcode.cpp`文件中：

```cpp
#include <iostream>

int Receive(int input, std::string& output) {
  if (input < 0) {
    return -1;
  }

  output = "Hello";
  return 0;
}
```

1.  接下来，我们添加第二个实现：

```cpp
std::string Receive(int input, int& error) {
  if (input < 0) {
    error = -1;
    return "";
  }
  error = 0;
  return "Hello";
}
```

1.  `Receive`函数的第三个实现如下：

```cpp
std::pair<int, std::string> Receive(int input) {
  std::pair<int, std::string> result;
  if (input < 0) {
    result.first = -1;
  } else {
    result.second = "Hello";
  }
  return result;
}
```

1.  现在，我们定义一个名为`Display`的辅助函数来显示结果：

```cpp
void Display(const char* prefix, int err, const std::string& result) {
  if (err < 0) {
    std::cout << prefix << " error: " << err << std::endl;
  } else {
    std::cout << prefix << " result: " << result << std::endl;
  }
}
```

1.  然后，我们添加一个名为`Test`的函数，调用所有三个实现：

```cpp
void Test(int input) {
  std::string outputResult;
  int err = Receive(input, outputResult);
  Display(" Receive 1", err, outputResult);

  int outputErr = -1;
  std::string result = Receive(input, outputErr);
  Display(" Receive 2", outputErr, result);

  std::pair<int, std::string> ret = Receive(input);
  Display(" Receive 3", ret.first, ret.second);
}
```

1.  `main`函数将所有内容联系在一起：

```cpp
int main() {
  std::cout << "Input: -1" << std::endl;
  Test(-1);
  std::cout << "Input: 1" << std::endl;
  Test(1);

  return 0;
}
```

1.  最后，我们创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(errcode)
add_executable(errcode errcode.cpp)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  您现在可以构建和运行应用程序了。

# 工作原理...

在我们的应用程序中，我们定义了一个从某个设备接收数据的函数的三种不同实现。它应该将接收到的数据作为字符串返回，但在出现错误时，应返回表示错误原因的整数错误代码。

由于结果和错误代码具有不同的类型，我们无法重用相同的值。要在 C++中返回多个值，我们需要使用输出参数或创建一个复合数据类型。

我们的实现同时探索了这两种策略。我们使用 C++函数重载来定义`Receive`函数，它具有相同的名称，但不同类型的参数和返回值。

第一个实现返回一个错误代码，并将结果存储在输出参数中：

```cpp
int Receive(int input, std::string& output)
```

输出参数是一个通过引用传递的字符串，让函数修改其内容。第二个实现颠倒了参数。它将接收到的字符串作为结果返回，并接受错误代码作为输出参数：

```cpp
std::string Receive(int input, int& error)
```

由于我们希望错误代码由函数内部设置，因此我们也通过引用传递它。最后，第三种实现将结果和错误代码组合并返回一个 C++ `pair`：

```cpp
std::pair<int, std::string> Receive(int input)
```

该函数总是创建一个`std::pair<int, std::string>`实例。由于我们没有向其构造函数传递任何值，因此对象是默认初始化的。整数元素设置为`0`，字符串元素设置为空字符串。

这种方法不需要`output`参数，更易读，但构造和销毁`pair`对象的开销略高。

当所有三种实现都被定义后，我们在`Test`函数中测试它们。我们将相同的参数传递给每个实现并显示结果。我们期望它们每个都生成相同的结果。

有两次调用`Test`。首先，我们将`-1`作为参数传递，这应该触发错误路径，然后我们传递`1`，这将激活正常操作路径：

```cpp
  std::cout << "Input: -1" << std::endl;
  Test(-1);
  std::cout << "Input: 1" << std::endl;
  Test(1);
```

当我们运行我们的程序时，我们会看到以下输出：

![](img/4e6428d7-7667-4b9f-8b0c-bc17d09787d0.png)

所有三种实现都根据输入参数正确返回结果或错误代码。您可以根据整体设计准则或个人偏好在应用程序中使用任何方法。

# 还有更多...

作为 C++17 标准的一部分，标准库中添加了一个名为`std::optional`的模板。它可以表示可能丢失的可选值。它可以用作可能失败的函数的返回值。但是，它不能表示失败的原因，只能表示一个布尔值，指示该值是否有效。有关更多信息，请查看[`en.cppreference.com/w/cpp/utility/optional`](https://en.cppreference.com/w/cpp/utility/optional)上的`std::optional`参考。

# 使用异常进行错误处理

虽然错误代码仍然是嵌入式编程中最常见的错误处理技术，但 C++提供了另一种称为异常的机制。

异常旨在简化错误处理并使其更可靠。当使用错误代码时，开发人员必须检查每个函数的结果是否有错误，并将结果传播到调用函数。这会使代码充斥着大量的 if-else 结构，使函数逻辑更加晦涩。

当使用异常时，开发人员无需在每个函数调用后检查错误。异常会自动通过调用堆栈传播，直到达到可以通过记录、重试或终止应用程序来正确处理它的代码。

虽然异常是 C++标准库的默认错误处理机制，但与外围设备或底层操作系统层通信仍涉及错误代码。在本教程中，我们将学习如何使用`std::system_error`异常类将低级错误处理与 C++异常进行桥接。

# 如何做...

我们将创建一个简单的应用程序，通过串行链路与设备通信。请按照以下步骤操作：

1.  在您的工作目录中，即`~/test`，创建一个名为`except`的子目录。

1.  使用您喜欢的文本编辑器在`except`子目录中创建一个名为`except.cpp`的文件。

1.  将所需的包含放入`except.cpp`文件中：

```cpp
#include <iostream>
#include <system_error>
#include <fcntl.h>
#include <unistd.h>
```

1.  接下来，我们定义一个抽象通信设备的`Device`类。我们从构造函数和析构函数开始：

```cpp
class Device {
  int fd;

  public:
    Device(const std::string& deviceName) {
      fd = open(deviceName.c_str(), O_RDWR);
      if (fd < 0) {
        throw std::system_error(errno, std::system_category(),
                                "Failed to open device file");
      }
    }

    ~Device() {
      close(fd);
    }

```

1.  然后，我们添加一个发送数据到设备的方法，如下所示：

```cpp
    void Send(const std::string& data) {
      size_t offset = 0;
      size_t len = data.size();
      while (offset < data.size() - 1) {
        int sent = write(fd, data.data() + offset, 
                         data.size() - offset);
        if (sent < 0) {
          throw std::system_error(errno, 
                                  std::system_category(),
                                  "Failed to send data");
        }
        offset += sent;
      }
    }
};
```

1.  在我们的类被定义后，我们添加`main`函数来使用它：

```cpp
int main() {
  try {
    Device serial("/dev/ttyUSB0");
    serial.Send("Hello");
  } catch (std::system_error& e) {
    std::cout << "Error: " << e.what() << std::endl;
    std::cout << "Code: " << e.code() << " means \"" 
              << e.code().message()
              << "\"" << std::endl;
  }

  return 0;
}
```

1.  最后，我们创建一个`CMakeLists.txt`文件，其中包含程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(except)
add_executable(except except.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  您现在可以构建和运行应用程序。

# 工作原理...

我们的应用程序与通过串行连接的外部设备通信。在 POSIX 操作系统中，与设备通信类似于对常规文件的操作，并使用相同的 API；即`open`、`close`、`read`和`write`函数。

所有这些函数返回错误代码，指示各种错误条件。我们将通信包装在一个名为`Device`的类中，而不是直接使用它们。

它的构造函数尝试打开由`deviceName`构造函数参数引用的文件。构造函数检查错误代码，如果指示出现错误，则创建并抛出`std::system_error`异常：

```cpp
  throw std::system_error(errno, std::system_category(),
                          "Failed to open device file");
```

我们使用三个参数构造`std::system_error`实例。第一个是我们想要包装在异常中的错误代码。我们使用`open`函数在返回错误时设置的`errno`变量的值。第二个参数是错误类别。由于我们使用特定于操作系统的错误代码，我们使用`std::system_category`的实例。第一个参数是我们想要与异常关联的消息。它可以是任何有助于我们在发生错误时识别错误的内容。

类似地，我们定义了`Send`函数，它向设备发送数据。它是`write`系统函数的包装器，如果`write`返回错误，我们创建并抛出`std::system_error`实例。唯一的区别是消息字符串，因为我们希望在日志中区分这两种情况：

```cpp
throw std::system_error(errno, std::system_category(),
                         "Failed to send data");
}
```

在定义了`Device`类之后，我们可以使用它。我们只需创建`Device`类的一个实例并向其发送数据，而不是打开设备并检查错误，然后再次写入设备并再次检查错误：

```cpp
Device serial("/dev/ttyUSB0");
serial.Send("Hello");
```

所有错误处理都在主逻辑之后的`catch`块中。如果抛出系统错误，我们将其记录到标准输出。此外，我们打印嵌入在异常中的错误代码的信息。

```cpp
  } catch (std::system_error& e) {
    std::cout << "Error: " << e.what() << std::endl;
    std::cout << "Code: " << e.code() << " means \"" << e.code().message()
        << "\"" << std::endl;
  }
```

当我们构建和运行应用程序时，如果没有设备连接为`/dev/ttyUSB0`，它将显示以下输出：

![](img/e6da888b-864b-412d-8c25-e35b1e4323e1.png)

如预期的那样，检测到了错误条件，我们可以看到所有必需的细节，包括底层操作系统错误代码及其描述。请注意，使用包装类与设备通信的代码是简洁易读的。

# 还有更多...

C++标准库提供了许多预定义的异常和错误类别。有关更多详细信息，请查看 C++错误处理参考[`en.cppreference.com/w/cpp/error`](https://en.cppreference.com/w/cpp/error)。

# 在捕获异常时使用常量引用

C++异常为异常处理设计提供了强大的基础。它们灵活多样，可以以多种不同的方式使用。您可以抛出任何类型的异常，包括指针和整数。您可以通过值或引用捕获异常。在选择数据类型时做出错误选择可能会导致性能损失或资源泄漏。

在这个配方中，我们将分析潜在的陷阱，并学习如何在 catch 块中使用常量引用来进行高效和安全的错误处理。

# 如何做...

我们将创建一个样本应用程序，抛出并捕获自定义异常，并分析数据类型选择如何影响效率。按照以下步骤进行：

1.  在您的工作目录中，即`~/test`，创建一个名为`catch`的子目录。

1.  使用您喜欢的文本编辑器在`catch`子目录中创建一个名为`catch.cpp`的文件。

1.  将`Error`类的定义放在`catch.cpp`文件中：

```cpp
#include <iostream>

class Error {
  int code;

  public:
    Error(int code): code(code) {
      std::cout << " Error instance " << code << " was created"
                << std::endl;
    }
    Error(const Error& other): code(other.code) {
      std::cout << " Error instance " << code << " was cloned"
                << std::endl;
    }
    ~Error() {
      std::cout << " Error instance " << code << " was destroyed"
                << std::endl;
    }
};
```

1.  接下来，我们添加辅助函数来测试三种不同的抛出和处理错误的方式。我们从通过值捕获异常的函数开始：

```cpp
void CatchByValue() {
  std::cout << "Catch by value" << std::endl;
  try {
    throw Error(1);
  }
  catch (Error e) {
    std::cout << " Error caught" << std::endl;
  }
}
```

1.  然后，我们添加一个抛出指针并通过指针捕获异常的函数，如下所示：

```cpp
void CatchByPointer() {
  std::cout << "Catch by pointer" << std::endl;
  try {
    throw new Error(2);
  }
  catch (Error* e) {
    std::cout << " Error caught" << std::endl;
  }
}
```

1.  接下来，我们添加一个使用`const`引用来捕获异常的函数：

```cpp
void CatchByReference() {
  std::cout << "Catch by reference" << std::endl;
  try {
    throw Error(3);
  }
  catch (const Error& e) {
    std::cout << " Error caught" << std::endl;
  }
}
```

1.  在定义了所有辅助函数之后，我们添加`main`函数来将所有内容联系在一起：

```cpp
int main() {
  CatchByValue();
  CatchByPointer();
  CatchByReference();
  return 0;
}
```

1.  我们将应用程序的构建规则放入`CMakeLists.txt`文件中：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(catch)
add_executable(catch catch.cpp)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在我们可以构建和运行应用程序了。

# 工作原理...

在我们的应用程序中，我们定义了一个名为`Error`的自定义类，当抛出和捕获异常时将使用该类。该类提供了一个构造函数、一个复制构造函数和一个仅将信息记录到控制台的析构函数。我们需要它来评估不同异常捕获方法的效率。

`Error`类只包含`code`数据字段，用于区分类的实例：

```cpp
class Error {
  int code;
```

我们评估了三种异常处理方法。第一种`CatchByValue`是最直接的。我们创建并抛出`Error`类的一个实例：

```cpp
throw Error(1);
```

然后，我们通过值捕获它：

```cpp
catch (Error e) {
```

第二种实现`CatchByPointer`，使用`new`运算符动态创建`Error`的实例：

```cpp
throw new Error(2);
```

我们使用指针来捕获异常：

```cpp
catch (Error* e) {
```

最后，`CatchByReference`引发类似于`CatchByValue`的异常，但在捕获时使用`Error`的`const`引用：

```cpp
catch (const Error& e) {
```

有什么区别吗？当我们运行程序时，我们会得到以下输出：

![](img/0f5b8cf3-a4d9-4337-972f-32d2fdc7772c.png)

如您所见，通过值捕获对象时，会创建异常对象的副本。虽然在示例应用程序中不是关键问题，但这种效率低下可能会导致高负载应用程序的性能问题。

通过指针捕获异常时不会出现效率低下，但我们可以看到对象的析构函数没有被调用，导致内存泄漏。可以通过在`catch`块中调用`delete`来避免这种情况，但这很容易出错，因为并不总是清楚谁负责销毁指针引用的对象。

引用方法是最安全和最有效的方法。没有内存泄漏和不必要的复制。同时，使引用为常量会给编译器一个提示，表明它不会被更改，因此可以在底层更好地进行优化。

# 还有更多...

错误处理是一个复杂的领域，有许多最佳实践、提示和建议。考虑阅读 C++异常和错误处理 FAQ [`isocpp.org/wiki/faq/exceptions`](https://isocpp.org/wiki/faq/exceptions) 来掌握异常处理技能。

# 解决静态对象问题

在 C++中，如果对象无法正确实例化，对象构造函数会抛出异常。通常，这不会引起任何问题。在堆栈上构造的对象或使用`new`关键字动态创建的对象引发的异常可以通过 try-catch 块处理，该块位于创建对象的代码周围。

对于静态对象来说，情况会变得更加复杂。这些对象在执行进入`main`函数之前就被实例化，因此它们无法被程序的 try-catch 块包裹。C++编译器通过调用`std::terminate`函数来处理这种情况，该函数打印错误消息并终止程序。即使异常是非致命的，也没有办法恢复。

有几种方法可以避免陷阱。作为一般规则，只应静态分配简单的整数数据类型。如果仍然需要具有复杂静态对象，请确保其构造函数不会引发异常。

在本教程中，我们将学习如何为静态对象实现构造函数。

# 如何做...

我们将创建一个自定义类，该类分配指定数量的内存并静态分配两个类的实例。按照以下步骤进行：

1.  在您的工作目录中，即`〜/test`，创建一个名为`static`的子目录。

1.  使用您喜欢的文本编辑器在`static`子目录中创建一个名为`static.cpp`的文件。

1.  让我们定义一个名为`Complex`的类。将其私有字段和构造函数放在`static.cpp`文件中：

```cpp
#include <iostream>
#include <stdint.h>

class Complex {
  char* ptr;

  public:
    Complex(size_t size) noexcept {
      try {
        ptr = new(std::nothrow) char[size];
        if (ptr) {
          std::cout << "Successfully allocated "
                    << size << " bytes" << std::endl;
        } else {
          std::cout << "Failed to allocate "
                    << size << " bytes" << std::endl;
        }
      } catch (...) {
        // Do nothing
      }
    }
```

1.  然后，定义一个析构函数和`IsValid`方法：

```cpp
    ~Complex() {
      try {
        if (ptr) {
          delete[] ptr;
          std::cout << "Deallocated memory" << std::endl;
        } else {
          std::cout << "Memory was not allocated" 
                    << std::endl;
        }
      } catch (...) {
        // Do nothing
      }
    }

    bool IsValid() const { return nullptr != ptr; }
};
```

1.  类定义后，我们定义了两个全局对象`small`和`large`，以及使用它们的`main`函数：

```cpp
Complex small(100);
Complex large(SIZE_MAX);
int main() {
  std::cout << "Small object is " 
            << (small.IsValid()? "valid" : "invalid")
            << std::endl;
  std::cout << "Large object is " 
            << (large.IsValid()? "valid" : "invalid")
            << std::endl;

  return 0;
}
```

1.  最后，我们创建一个`CMakeLists.txt`文件，其中包含我们程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(static)
add_executable(static static.cpp)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建和运行应用程序。

# 工作原理...

在这里，我们定义了`Complex`类，并且我们打算静态分配此类的实例。为了安全起见，我们需要确保此类的构造函数和析构函数都不会引发异常。

然而，构造函数和析构函数都调用可能引发异常的操作。构造函数执行内存分配，而析构函数将日志写入标准输出。

构造函数使用`new`运算符分配内存，如果无法分配内存，则会引发`std::bad_alloc`异常。我们使用`std::nothrow`常量来选择`new`的不抛出实现。`new`将返回`nullptr`而不是引发异常，如果它无法分配任何内存：

```cpp
ptr = new(std::nothrow) char[size];
```

我们将构造函数的主体放在`try`块中以捕获所有异常。`catch`块为空-如果构造函数失败，我们无能为力：

```cpp
} catch (...) {
        // Do nothing
}
```

由于我们不允许任何异常传播到上一级，因此我们使用 C++关键字`noexcept`将我们的构造函数标记为不抛出异常：

```cpp
Complex(size_t size) noexcept {
```

然而，我们需要知道对象是否被正确创建。为此，我们定义了一个名为`IsValid`的方法。如果内存已分配，则返回`true`，否则返回`false`：

```cpp
bool IsValid() const { return nullptr != ptr; }
```

析构函数则相反。它释放内存并将释放状态记录到控制台。对于构造函数，我们不希望任何异常传播到上一级，因此我们将析构函数主体包装在 try-catch 块中：

```cpp
 try {
        if (ptr) {
 delete[] ptr;
          std::cout << "Deallocated memory" << std::endl;
        } else {
          std::cout << "Memory was not allocated" << std::endl;
        }
      } catch (...) {
        // Do nothing
      }
```

现在，我们声明了两个全局对象`small`和`large`。全局对象是静态分配的。对象的大小是人为选择的，`small`对象将被正确分配，但`large`对象的分配应该失败：

```cpp
Complex small(100);
Complex large(SIZE_MAX);
```

在我们的`main`函数中，检查并打印对象是否有效：

```cpp
  std::cout << "Small object is " << (small.IsValid()? "valid" : "invalid")
            << std::endl;
  std::cout << "Large object is " << (large.IsValid()? "valid" : "invalid")
            << std::endl;
```

当我们运行程序时，我们会看到以下输出：

![](img/33ae692c-8dd9-4803-b71e-6bdfd2d91a90.png)

正如我们所看到的，小对象被正确分配和释放。大对象的初始化失败，但由于它被设计为不引发任何异常，因此并未导致我们应用程序的异常终止。您可以使用类似的技术来为静态分配的对象编写健壮且安全的应用程序。

# 使用看门狗

嵌入式应用程序被设计为无需监督即可运行。这包括从错误中恢复的能力。如果应用程序崩溃，可以自动重新启动。但是，如果应用程序由于进入无限循环或由于死锁而挂起，我们该怎么办呢？

硬件或软件看门狗用于防止这种情况发生。应用程序应定期通知或*喂养*它们，以指示它们保持正常运行。如果在特定时间间隔内未喂养看门狗，则它将终止应用程序或重新启动系统。

存在许多不同的看门狗实现，但它们的接口本质上是相同的。它们提供一个函数，应用程序可以使用该函数重置看门狗定时器。

在本教程中，我们将学习如何在 POSIX 信号子系统之上创建一个简单的软件看门狗。相同的技术可以用于处理硬件看门狗定时器或更复杂的软件看门狗服务。

# 如何做...

我们将创建一个应用程序，定义`Watchdog`类并提供其用法示例。按照以下步骤进行：

1.  在您的工作目录中，即`~/test`，创建一个名为`watchdog`的子目录。

1.  使用您喜欢的文本编辑器在`watchdog`子目录中创建一个名为`watchdog.cpp`的文件。

1.  将所需的包含放在`watchdog.cpp`文件中：

```cpp
#include <chrono>
#include <iostream>
#include <thread>

#include <unistd.h>

using namespace std::chrono_literals;
```

1.  接下来，我们定义`Watchdog`类本身：

```cpp
class Watchdog {
  std::chrono::seconds seconds;

  public:
    Watchdog(std::chrono::seconds seconds):
      seconds(seconds) {
        feed();
    }

    ~Watchdog() {
      alarm(0);
    }

    void feed() {
      alarm(seconds.count());
    }
};
```

1.  添加`main`函数，作为我们看门狗的用法示例：

```cpp
int main() {
  Watchdog watchdog(2s);
  std::chrono::milliseconds delay = 700ms;
  for (int i = 0; i < 10; i++) {
    watchdog.feed();
    std::cout << delay.count() << "ms delay" << std::endl;
    std::this_thread::sleep_for(delay);
    delay += 300ms;
  }
}
```

1.  添加一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(watchdog)
add_executable(watchdog watchdog.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建并运行应用程序。

# 工作原理...

我们需要一种机制来在应用程序挂起时终止它。虽然我们可以生成一个特殊的监控线程或进程，但还有另一种更简单的方法——POSIX 信号。

在 POSIX 操作系统中运行的任何进程都可以接收多个信号。为了向进程传递信号，操作系统会停止进程的正常执行并调用相应的信号处理程序。

可以传递给进程的信号之一称为`alarm`，默认情况下，它的处理程序会终止应用程序。这正是我们需要实现看门狗的地方。

我们的`Watchdog`类的构造函数接受一个参数`seconds`：

```cpp
Watchdog(std::chrono::seconds seconds):
```

这是我们的看门狗的时间间隔，它立即传递到`feed`方法中以激活看门狗定时器：

```cpp
feed();
```

`feed`方法调用了一个 POSIX 函数`alarm`来设置计时器。如果计时器已经设置，它会用新值更新它：

```cpp
void feed() {
  alarm(seconds.count());
}
```

最后，在析构函数中调用相同的`alarm`函数来通过传递值`0`来禁用计时器：

```cpp
alarm(0);
```

现在，每次我们调用`feed`函数时，都会改变进程接收`alarm`信号的时间。然而，如果在计时器到期之前我们没有调用这个函数，它就会触发`alarm`处理程序，终止我们的进程。

为了检查它，我们创建了一个简单的示例。这是一个有 10 次迭代的循环。在每次迭代中，我们显示一条消息并休眠一段特定的时间间隔。初始间隔为 700 毫秒，每次迭代增加 300 毫秒；例如，700 毫秒，1,000 毫秒，1,300 毫秒等等：

```cpp
delay += 300ms;
```

我们的看门狗设置为 2 秒的间隔：

```cpp
Watchdog watchdog(2s);
```

让我们运行应用程序并检查它的工作原理。它产生以下输出：

![](img/b755a216-3c0d-4381-9129-554f07f472ba.png)

正如我们所看到的，应用程序在第六次迭代后被终止，因为延迟超过了看门狗的间隔。此外，由于它是异常终止的，它的返回代码是非零的。如果应用程序是由另一个应用程序或脚本生成的，这表明应用程序需要重新启动。

看门狗技术是构建健壮嵌入式应用程序的一种简单有效的方法。

# 探索高可用系统的心跳。

在前面的示例中，我们学习了如何使用看门狗定时器来防止软件挂起。类似的技术可以用来实现高可用系统，它由一个或多个软件或硬件组件组成，可以执行相同的功能。如果其中一个组件失败，另一个组件可以接管。

当前活动的组件应定期向其他被动组件广告其健康状态，使用称为**心跳**的消息。当它报告不健康状态或在特定时间内没有报告时，被动组件会检测到并激活自己。当失败的组件恢复时，它可以转换为被动模式，监视现在活动的组件是否失败，或者启动故障恢复过程来重新获得活动状态。

在这个示例中，我们将学习如何在我们的应用程序中实现一个简单的心跳监视器。

# 如何做...

我们将创建一个定义了`Watchdog`类并提供其用法示例的应用程序。按照以下步骤进行：

1.  在你的工作目录中，即`~/test`，创建一个名为`heartbeat`的子目录。

1.  使用你喜欢的文本编辑器在`heartbeat`子目录中创建一个名为`heartbeat.cpp`的文件。

1.  在`heatbeat.cpp`文件中放入所需的包含文件：

```cpp
#include <chrono>
#include <iostream>
#include <system_error>
#include <thread>

#include <unistd.h>
#include <poll.h>
#include <signal.h>

using namespace std::chrono_literals;
```

1.  接下来，我们定义一个`enum`来报告活动工作者的健康状态：

```cpp
enum class Health : uint8_t {
  Ok,
  Unhealthy,
  ShutDown
};
```

1.  现在，让我们创建一个封装心跳报告和监控的类。我们从类定义、私有字段和构造函数开始：

```cpp
class Heartbeat {
  int channel[2];
  std::chrono::milliseconds delay;

  public:
    Heartbeat(std::chrono::milliseconds delay):
        delay(delay) {
      int rv = pipe(channel);
      if (rv < 0) {
        throw std::system_error(errno,         
                                std::system_category(),
                                "Failed to open pipe");
      }
    }

```

1.  接下来，我们添加一个报告健康状态的方法：

```cpp
    void Report(Health status) {
      int rv = write(channel[1], &status, sizeof(status));
      if (rv < 0) {
        throw std::system_error(errno, 
                        std::system_category(),
                        "Failed to report health status");
      }
    }
```

1.  接下来是健康监控方法：

```cpp
    bool Monitor() {
      struct pollfd fds[1];
      fds[0].fd = channel[0];
      fds[0].events = POLLIN;
      bool takeover = true;
      bool polling = true;
      while(polling) {
        fds[0].revents = 0;
        int rv = poll(fds, 1, delay.count());
        if (rv) {
          if (fds[0].revents & (POLLERR | POLLHUP)) {
            std::cout << "Polling error occured" 
                      << std::endl;
            takeover = false;
            polling = false;
            break;
          }

          Health status;
          int count = read(fds[0].fd, &status, 
                           sizeof(status));
          if (count < sizeof(status)) {
            std::cout << "Failed to read heartbeat data" 
                      << std::endl;
            break;
          }
          switch(status) {
            case Health::Ok:
              std::cout << "Active process is healthy" 
                        << std::endl;
              break;
            case Health::ShutDown:
              std::cout << "Shut down signalled" 
                        << std::endl;
              takeover = false;
              polling = false;
              break;
            default:
              std::cout << "Unhealthy status reported" 
                        << std::endl;
              polling = false;
              break;
          }
        } else if (!rv) {
          std::cout << "Timeout" << std::endl;
          polling = false;
        } else {
          if (errno != EINTR) {
            std::cout << "Error reading heartbeat data, retrying" << std::endl;
          }
        }
      }
      return takeover;
    }
};
```

1.  一旦心跳逻辑被定义，我们创建一些函数，以便在我们的测试应用程序中使用它：

```cpp
void Worker(Heartbeat& hb) {
  for (int i = 0; i < 5; i++) {
    hb.Report(Health::Ok);
    std::cout << "Processing" << std::endl;
    std::this_thread::sleep_for(100ms);
  }
  hb.Report(Health::Unhealthy);
}

int main() {
  Heartbeat hb(200ms);
  if (fork()) {
    if (hb.Monitor()) {
      std::cout << "Taking over" << std::endl;
      Worker(hb);
    }
  } else {
    Worker(hb);
  }
}
```

1.  接下来，我们添加一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(heartbeat)
add_executable(heartbeat heartbeat.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建和运行应用程序了。

# 工作原理...

心跳机制需要某种通信渠道，让一个组件向其他组件报告其状态。在一个围绕多个处理单元构建的系统中，最好的选择是基于网络的套接字通信。我们的应用程序在单个节点上运行，因此我们可以使用本地 IPC 机制之一。

我们将使用 POSIX 管道机制进行心跳传输。创建管道时，它提供两个文件描述符进行通信——一个用于读取数据，另一个用于写入数据。

除了通信传输，我们还需要选择接管的时间间隔。如果监控过程在此间隔内未收到心跳消息，则应将另一个组件视为不健康或失败，并执行一些接管操作。

我们首先定义应用程序可能的健康状态。我们使用 C++的`enum class`使状态严格类型化，如下所示：

```cpp
enum class Health : uint8_t {
  Ok,
  Unhealthy,
  ShutDown
};
```

我们的应用程序很简单，只有三种状态：`Ok`、`Unhealthy`和`ShutDown`。`ShutDown`状态表示活动进程将正常关闭，不需要接管操作。

然后，我们定义`Heartbeat`类，它封装了所有消息交换、健康报告和监控功能。

它有两个数据字段，表示监控时间间隔和用于消息交换的 POSIX 管道：

```cpp
  int channel[2];
  std::chrono::milliseconds delay;
```

构造函数创建管道，并在失败时抛出异常：

```cpp
 int rv = pipe(channel);
      if (rv < 0) {
        throw std::system_error(errno,         
                                std::system_category(),
                                "Failed to open pipe");

```

健康报告方法是`write`函数的简单包装。它将状态以无符号 8 位整数值的形式写入管道的`write`文件描述符：

```cpp
int rv = write(channel[1], &status, sizeof(status));
```

监控方法更复杂。它使用 POSIX 的`poll`函数等待一个或多个文件描述符中的数据。在我们的情况下，我们只对一个文件描述符中的数据感兴趣——管道的读端。我们填充`poll`使用的`fds`结构，其中包括文件描述符和我们感兴趣的事件类型：

```cpp
      struct pollfd fds[1];
      fds[0].fd = channel[0];
      fds[0].events = POLLIN | POLLERR | POLLHUP;
```

两个布尔标志控制轮询循环。`takeover`标志指示我们退出循环时是否应执行接管操作，而`polling`标志指示循环是否应该存在：

```cpp
      bool takeover = true;
      bool polling = true;
```

在循环的每次迭代中，我们使用`poll`函数在套接字中轮询新数据。我们使用传入构造函数的监控间隔作为轮询超时：

```cpp
        int rv = poll(fds, 1, delay.count());
```

`poll`函数的结果指示三种可能的结果之一：

+   如果大于零，我们可以从通信管道中读取新数据。我们从通信通道中读取状态并进行分析。

+   如果状态是`Ok`，我们记录下来并进入下一个轮询迭代。

+   如果状态是`ShutDown`，我们需要退出轮询循环，但也要阻止`takeover`操作。为此，我们相应地设置我们的布尔标志：

```cpp
            case Health::ShutDown:
              std::cout << "Shut down signalled"
                        << std::endl;
 takeover = false;
 polling = false;
```

对于任何其他健康状态，我们会以`takeover`标志设置为`true`退出循环：

```cpp
              std::cout << "Unhealthy status reported"
                        << std::endl;
 polling = false;
```

在超时的情况下，`poll`返回零。与`Unhealthy`状态类似，我们需要从循环中退出并执行`takeover`操作：

```cpp
        } else if (!rv) {
          std::cout << "Timeout" << std::endl;
          polling = false;
```

最后，如果`poll`返回的值小于零，表示出现错误。系统调用失败有几种原因，其中一个非常常见的原因是被信号中断。这不是真正的错误；我们只需要再次调用`poll`。对于所有其他情况，我们会写入日志消息并继续轮询。

监控方法在监控循环运行时会阻塞，并返回一个布尔值，让调用者知道是否应执行`takeover`操作：

```cpp
 bool Monitor() {
```

现在，让我们尝试在一个玩具示例中使用这个类。我们将定义一个接受`Heartbeat`实例引用并表示要完成的工作的`Worker`函数：

```cpp
void Worker(Heartbeat& hb) {
```

在内部循环的每次迭代中，`Worker`报告其健康状态：

```cpp
hb.Report(Health::Ok);
```

在某个时刻，它报告其状态为`Unhealthy`：

```cpp
  hb.Report(Health::Unhealthy);
```

在`main`函数中，我们使用 200 毫秒的轮询间隔创建了一个`Heartbeat`类的实例：

```cpp
  Heartbeat hb(200ms);
```

然后，我们生成两个独立的进程。父进程开始监视，并且如果需要接管，运行`Worker`方法：

```cpp
    if (hb.Monitor()) {
      std::cout << "Taking over" << std::endl;
      Worker(hb);
    }
```

子类只是运行`Worker`方法。让我们运行应用程序并检查它的工作原理。它产生以下输出：

![](img/d55f961e-37da-4689-8aa0-f0d9f2e9a02a.png)

正如我们所看到的，`Worker`方法报告它正在处理数据，监视器检测到它的状态是健康的。然而，在`Worker`方法报告其状态为`Unhealthy`后，监视器立即检测到并重新运行工作程序，以继续处理。这种策略可以用于构建更复杂的健康监控和故障恢复逻辑，以实现您设计和开发的系统的高可用性。

# 还有更多...

在我们的示例中，我们使用了两个同时运行并相互监视的相同组件。但是，如果其中一个组件包含软件错误，在某些条件下导致组件发生故障，那么另一个相同的组件也很可能受到这个问题的影响。在安全关键系统中，您可能需要开发两个完全不同的实现。这种方法会增加成本和开发时间，但会提高系统的可靠性。

# 实现软件去抖动逻辑

嵌入式应用的常见任务之一是与外部物理控件（如按钮或开关）进行交互。尽管这些对象只有两种状态 - 开和关 - 但检测按钮或开关改变状态的时刻并不像看起来那么简单。

当物理按钮被按下时，需要一些时间才能建立联系。在此期间，可能会触发虚假中断，就好像按钮在开和关状态之间跳动。应用程序不应该对每个中断做出反应，而应该能够过滤掉虚假的转换。这就是**去抖动**。

尽管它可以在硬件级别实现，但最常见的方法是通过软件来实现。在本教程中，我们将学习如何实现一个简单通用的去抖动函数，可以用于任何类型的输入。

# 如何做...

我们将创建一个应用程序，定义一个通用的去抖动函数以及一个测试输入。通过用真实输入替换测试输入，可以将此函数用于任何实际目的。按照以下步骤进行：

1.  在您的工作目录中，即`~/test`，创建一个名为`debounce`的子目录。

1.  使用您喜欢的文本编辑器在`debounce`子目录中创建一个名为`debounce.cpp`的文件。

1.  让我们在`debounce.cpp`文件中添加包含和一个名为`debounce`的函数：

```cpp
#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

bool debounce(std::chrono::milliseconds timeout, bool (*handler)(void)) {
  bool prev = handler();
  auto ts = std::chrono::steady_clock::now();
  while (true) {
    std::this_thread::sleep_for(1ms);
    bool value = handler();
    auto now = std::chrono::steady_clock::now();
    if (value == prev) {
      if (now - ts > timeout) {
        break;
      }
    } else {
      prev = value;
      ts = now;
    }
  }
  return prev;
}
```

1.  然后，我们添加`main`函数，展示如何使用它：

```cpp
int main() {
  bool result = debounce(10ms, []() {
    return true;
  });
  std::cout << "Result: " << result << std::endl;
}
```

1.  添加一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(debounce)
add_executable(debounce debounce.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++14")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建和运行应用程序了。

# 工作原理...

我们的目标是检测按钮在开和关状态之间停止跳动的时刻。我们假设如果所有连续尝试读取按钮状态在特定时间间隔内返回相同的值（开或关），我们就可以知道按钮是真正开着还是关着。

我们使用这个逻辑来实现`debounce`函数。由于我们希望去抖动逻辑尽可能通用，函数不应该知道如何读取按钮的状态。这就是为什么函数接受两个参数的原因：

```cpp
bool debounce(std::chrono::milliseconds timeout, bool (*handler)(void)) {
```

第一个参数`timeout`定义了我们需要等待报告状态变化的特定时间间隔。第二个参数`handler`是一个函数或类似函数的对象，它知道如何读取按钮的状态。它被定义为指向没有参数的布尔函数的指针。

`debounce`函数运行一个循环。在每次迭代中，它调用处理程序来读取按钮的状态并将其与先前的值进行比较。如果值相等，我们检查自最近状态变化以来的时间。如果超过超时时间，我们退出循环并返回：

```cpp
auto now = std::chrono::steady_clock::now();
    if (value == prev) {
      if (now - ts > timeout) {
        break;
      }
```

如果值不相等，我们会重置最近状态变化的时间并继续等待：

```cpp
} else {
      prev = value;
      ts = now;
    }
```

为了最小化 CPU 负载并让其他进程做一些工作，我们在读取之间添加了 1 毫秒的延迟。如果函数打算用于不运行多任务操作系统的微控制器上，则不需要这个延迟：

```cpp
std::this_thread::sleep_for(1ms);
```

我们的`main`函数包含了对`debounce`函数的使用示例。我们使用 C++ lambda 来定义一个简单的规则来读取按钮。它总是返回`true`：

```cpp
  bool result = debounce(10ms, []() {
 return true;
 });
```

我们将`10ms`作为`debounce`超时传递。如果我们运行我们的程序，我们将看到以下输出：

![](img/15406ee1-e262-4eca-b5d6-f744a0738e85.png)

`debounce`函数工作了 10 毫秒并返回`true`，因为测试输入中没有出现意外的状态变化。在实际输入的情况下，可能需要更多的时间才能使按钮状态稳定下来。这个简单而高效的去抖动函数可以应用于各种真实的输入。
