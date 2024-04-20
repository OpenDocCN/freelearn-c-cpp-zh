# 通信和序列化

复杂的嵌入式系统很少由单个应用程序组成。将所有逻辑放在同一个应用程序中是脆弱的、容易出错的，有时甚至难以实现，因为系统的不同功能可能由不同的团队甚至不同的供应商开发。这就是为什么将函数的逻辑隔离在独立的应用程序中，并使用明确定义的协议相互通信是一种常见的方法，用于扩展嵌入式软件。此外，这种隔离可以通过最小的修改与托管在远程系统上的应用程序通信，使其更具可扩展性。我们将学习如何通过将其逻辑分割为相互通信的独立组件来构建健壮和可扩展的应用程序。

在本章中，我们将涵盖以下主题：

+   在应用程序中使用进程间通信

+   探索进程间通信的机制

+   学习消息队列和发布-订阅模型

+   使用 C++ lambda 进行回调

+   探索数据序列化

+   使用 FlatBuffers 库

本章中的示例将帮助您了解可扩展和平台无关的数据交换的基本概念。它们可以用于实现从嵌入式系统到云端或远程后端的数据传输，或者使用微服务架构设计嵌入式系统。

# 在应用程序中使用进程间通信

大多数现代操作系统使用底层硬件平台提供的内存虚拟化支持，以将应用程序进程彼此隔离。

每个进程都有自己完全独立于其他应用程序的虚拟地址空间。这为开发人员带来了巨大的好处。由于应用程序的地址进程是独立的，一个应用程序不能意外地破坏另一个应用程序的内存。因此，一个应用程序的失败不会影响整个系统。由于所有其他应用程序都在继续工作，系统可以通过重新启动失败的应用程序来恢复。

内存隔离的好处是有代价的。由于一个进程无法访问另一个进程的内存，它需要使用专用的**应用程序编程接口**（**API**）进行数据交换，或者由操作系统提供的**进程间通信**（**IPC**）。

在这个示例中，我们将学习如何使用共享文件在两个进程之间交换信息。这可能不是最高效的机制，但它是无处不在的，易于使用，并且对于各种实际用例来说足够好。

# 如何做...

在这个示例中，我们将创建一个示例应用程序，创建两个进程。一个进程生成数据，而另一个读取数据并将其打印到控制台：

1.  在您的工作目录（`~/test`）中，创建一个名为`ipc1`的子目录。

1.  使用您喜欢的文本编辑器在`ipc1`子目录中创建一个名为`ipc1.cpp`的文件。

1.  我们将定义两个模板类来组织我们的数据交换。第一个类`Writer`用于将数据写入文件。让我们将其定义放在`ipc1.cpp`文件中：

```cpp
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <unistd.h>

std::string kSharedFile = "/tmp/test.bin";

template<class T>
class Writer {
  private:
    std::ofstream out;
  public:
    Writer(std::string& name):
      out(name, std::ofstream::binary) {}

    void Write(const T& data) {
      out.write(reinterpret_cast<const char*>(&data), sizeof(T));
    }
};
```

1.  接下来是`Reader`类的定义，它负责从文件中读取数据：

```cpp
template<class T>
class Reader {
  private:
    std::ifstream in;
  public:
    Reader(std::string& name) {
      for(int count=10; count && !in.is_open(); count--) {
        in.open(name, std::ifstream::binary);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    T Read() {
      int count = 10;
      for (;count && in.eof(); count--) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      T data;
      in.read(reinterpret_cast<char*>(&data), sizeof(data));
      if (!in) {
        throw std::runtime_error("Failed to read a message");
      }
      return data;
    }
};
```

1.  接下来，我们定义将用于我们数据的数据类型：

```cpp
struct Message {
  int x, y;
};

std::ostream& operator<<(std::ostream& o, const Message& m) {
  o << "(x=" << m.x << ", y=" << m.y << ")";
}
```

1.  为了将所有内容整合在一起，我们定义了`DoWrites`和`DoReads`函数，以及调用它们的`main`函数：

```cpp
void DoWrites() {
  std::vector<Message> messages {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Writer<Message> writer(kSharedFile);
  for (const auto& m : messages) {
    std::cout << "Write " << m << std::endl;
    writer.Write(m);
  }
}

void DoReads() {
  Reader<Message> reader(kSharedFile);
  try {
    while(true) {
      std::cout << "Read " << reader.Read() << std::endl;
    }
  } catch (const std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }
}

int main(int argc, char** argv) {
  if (fork()) {
    DoWrites();
  } else {
    DoReads();
  }
}
```

1.  最后，创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(ipc1)
add_executable(ipc1 ipc1.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在可以构建和运行应用程序了。

# 工作原理...

在我们的应用程序中，我们探索了在文件系统中使用共享文件在两个独立进程之间进行数据交换。一个进程向文件写入数据，另一个从同一文件读取数据。

文件可以存储任何非结构化的字节序列。在我们的应用程序中，我们利用 C++模板的能力来处理严格类型化的 C++值，而不是原始字节流。这种方法有助于编写干净且无错误的代码。

我们从`Write`类的定义开始。它是标准 C++ `fstream`类的简单包装，用于文件输入/输出。该类的构造函数只打开一个文件流以进行以下写入：

```cpp
Writer(std::string& name):
      out(name, std::ofstream::binary) {}
```

除了构造函数，该类只包含一个名为`Write`的方法，负责向文件写入数据。由于文件 API 操作的是字节流，我们首先需要将我们的模板数据类型转换为原始字符缓冲区。我们可以使用 C++的`reinterpret_cast`来实现这一点：

```cpp
out.write(reinterpret_cast<const char*>(&data), sizeof(T));
```

`Reader`类的工作与`Writer`类相反——它读取`Writer`类写入的数据。它的构造函数稍微复杂一些。由于数据文件可能在创建`Reader`类的实例时还没有准备好，构造函数会尝试在循环中打开它，直到成功打开为止。它会尝试 10 次，每次间隔 10 毫秒：

```cpp
for(int count=10; count && !in.is_open(); count--) {
        in.open(name, std::ifstream::binary);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
```

`Read`方法从输入流中读取数据到临时值，并将其返回给调用者。与`Write`方法类似，我们使用`reinterpret_cast`来访问我们的数据对象的内存作为原始字符缓冲区：

```cpp
in.read(reinterpret_cast<char*>(&data), sizeof(data));
```

我们还在`Read`方法中添加了一个等待循环，等待`Write`写入数据。如果我们到达文件的末尾，我们等待最多 1 秒钟获取新数据：

```cpp
      for (;count && in.eof(); count--) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
```

如果此时文件中没有可用的数据，或者出现 I/O 错误，我们会抛出异常来指示它：

```cpp
      if (!in) {
        throw std::runtime_error("Failed to read a message");
      }
```

请注意，我们不需要添加任何代码来处理文件在 1 秒内无法打开的情况，或者数据在一秒内不准备好的情况。这两种情况都由前面的代码处理。

现在`Writer`和`Reader`类已经实现，我们可以为我们的数据交换定义一个数据类型。在我们的应用程序中，我们将交换坐标，表示为`x`和`y`的整数值。我们的数据消息看起来像这样：

```cpp
struct Message {
  int x, y;
};
```

为了方便起见，我们重写了`Message`结构的`<<`运算符。每当`Message`的实例被写入输出流时，它都会被格式化为`(x, y)`：

```cpp
std::ostream& operator<<(std::ostream& o, const Message& m) {
  o << "(x=" << m.x << ", y=" << m.y << ")";
}
```

准备工作已经就绪，让我们编写数据交换的函数。`DoWrites`函数定义了一个包含四个坐标的向量，并创建了一个`Writer`对象：

```cpp
  std::vector<Message> messages {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Writer<Message> writer(kSharedFile);
```

然后，它在循环中写入所有的坐标：

```cpp
  for (const auto& m : messages) {
    std::cout << "Write " << m << std::endl;
    writer.Write(m);
  }
```

`DoReads`函数创建一个`Reader`类的实例，使用与之前的`Writer`实例相同的文件名。它进入一个无限循环，尝试读取文件中的所有消息：

```cpp
 while(true) {
      std::cout << "Read " << reader.Read() << std::endl;
    }
```

当没有更多的消息可用时，`Read`方法会抛出一个异常来中断循环：

```cpp
  } catch (const std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }
```

`main`函数创建了两个独立的进程，在其中一个进程中运行`DoWrites`，在另一个进程中运行`DoReads`。运行应用程序后，我们得到以下输出：

![](img/8f5ce532-4e3c-401e-8716-87f43e5d0c8c.png)

正如我们所看到的，写入者确实写入了四个坐标，读取者能够使用共享文件读取相同的四个坐标。

# 还有更多...

我们设计应用程序尽可能简单，专注于严格类型化的数据交换，并将数据同步和数据序列化排除在范围之外。我们将使用这个应用程序作为更高级技术的基础，这些技术将在接下来的示例中描述。

# 探索进程间通信的机制

现代操作系统提供了许多 IPC 机制，除了我们已经了解的共享文件之外，还有以下机制：

+   管道

+   命名管道

+   本地套接字

+   网络套接字

+   共享内存

有趣的是，其中许多提供的 API 与我们在使用常规文件时使用的 API 完全相同。因此，在这些类型的 IPC 之间切换是微不足道的，我们用来读写本地文件的相同代码可以用来与运行在远程网络主机上的应用程序进行通信。

在这个示例中，我们将学习如何使用名为**POSIX**的可移植操作系统接口（**POSIX**）命名管道来在同一台计算机上的两个应用程序之间进行通信。

# 准备工作

我们将使用作为*在应用程序中使用进程间通信*示例的一部分创建的应用程序的源代码作为本示例的起点。

# 如何做...

在这个示例中，我们将从使用常规文件进行 IPC 的源代码开始。我们将修改它以使用一种名为**命名管道**的 IPC 机制：

1.  将`ipc1`目录的内容复制到一个名为`ipc2`的新目录中。

1.  打开`ipc1.cpp`文件，在`#include <unistd.h>`后添加两个`include`实例：

```cpp
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
```

1.  通过在`Writer`类的`Write`方法中添加一行来修改`Write`方法：

```cpp
    void Write(const T& data) {
      out.write(reinterpret_cast<const char*>(&data), sizeof(T));
 out.flush();
    }
```

1.  `Reader`类中的修改更为重要。构造函数和`Read`方法都受到影响：

```cpp
template<class T>
class Reader {
  private:
    std::ifstream in;
  public:
    Reader(std::string& name):
      in(name, std::ofstream::binary) {}

    T Read() {
      T data;
      in.read(reinterpret_cast<char*>(&data), sizeof(data));
      if (!in) {
        throw std::runtime_error("Failed to read a message");
      }
      return data;
    }
};
```

1.  对`DoWrites`函数进行小的更改。唯一的区别是在发送每条消息后添加 10 毫秒的延迟：

```cpp
void DoWrites() {
  std::vector<Message> messages {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Writer<Message> writer(kSharedFile);
  for (const auto& m : messages) {
    std::cout << "Write " << m << std::endl;
    writer.Write(m);
 std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
```

1.  最后，修改我们的`main`函数，创建一个命名管道而不是一个常规文件：

```cpp
int main(int argc, char** argv) {
 int ret = mkfifo(kSharedFile.c_str(), 0600);
 if (!ret) {
 throw std::runtime_error("Failed to create named pipe");
 }
  if (fork()) {
    DoWrites();
  } else {
    DoReads();
  }
}
```

现在可以构建和运行应用程序了。

# 工作原理...

正如你所看到的，我们对应用程序的代码进行了最少量的更改。所有读写数据的机制和 API 保持不变。关键的区别隐藏在一行代码后面：

```cpp
 int ret = mkfifo(kSharedFile.c_str(), 0600);
```

这一行创建了一种特殊类型的文件，称为`命名管道`。它看起来像一个常规文件——它有一个名称、权限属性和修改时间。但是，它不存储任何真实的数据。写入到该文件的所有内容都会立即传递给从该文件读取的进程。

这种差异有一系列后果。由于文件中没有存储任何真实数据，所有的读取尝试都会被阻塞，直到有数据被写入。同样，写入也会被阻塞，直到读取者读取了先前的数据。

因此，不再需要外部数据同步。看一下`Reader`类的实现。它在构造函数或`Read`方法中都没有重试循环。

为了测试我们确实不需要使用任何额外的同步，我们在每条消息写入后添加了人为的延迟：

```cpp
 std::this_thread::sleep_for(std::chrono::milliseconds(10));
```

当我们构建和运行应用程序时，我们可以看到以下输出：

![](img/6f71e499-25db-4757-96ab-21a9a119babf.png)

每个`Write`方法后面都跟着适当的`Read`方法，尽管我们在`Reader`代码中没有添加任何延迟或检查。操作系统的 IPC 机制会透明地为我们处理数据同步，从而使代码更清晰和可读。

# 还有更多...

如你所见，使用命名管道与使用常规函数一样简单。套接字 API 是 IPC 的另一种广泛使用的机制。它稍微复杂一些，但提供了更多的灵活性。通过选择不同的传输层，开发人员可以使用相同的套接字 API 来进行本地数据交换和与远程主机的网络连接。

有关套接字 API 的更多信息，请访问[`man7.org/linux/man-pages/man7/socket.7.html`](http://man7.org/linux/man-pages/man7/socket.7.html)。

# 学习消息队列和发布-订阅模型

POSIX 操作系统提供的大多数 IPC 机制都非常基本。它们的 API 是使用文件描述符构建的，并且将输入和输出通道视为原始的字节序列。

然而，应用程序往往使用特定长度和目的的数据片段进行数据交换消息。尽管操作系统的 API 机制灵活且通用，但并不总是方便进行消息交换。这就是为什么在默认 IPC 机制之上构建了专用库和组件，以简化消息交换模式。

在这篇文章中，我们将学习如何使用**发布者-订阅者**（pub-sub）模型在两个应用程序之间实现异步数据交换。

这种模型易于理解，并且被广泛用于开发软件系统，这些系统被设计为相互独立、松散耦合的组件集合，它们之间进行通信。函数的隔离和异步数据交换使我们能够构建灵活、可扩展和健壮的解决方案。

在发布-订阅模型中，应用程序可以充当发布者、订阅者或两者兼而有之。应用程序不需要向特定应用程序发送请求并期望它们做出响应，而是可以向特定主题发布消息或订阅接收感兴趣的主题上的消息。在发布消息时，应用程序不关心有多少订阅者正在监听该主题。同样，订阅者不知道哪个应用程序将在特定主题上发送消息，或者何时期望收到消息。

# 操作方法...

我们在*探索 IPC 机制*配方中创建的应用程序已经包含了许多我们可以重用的构建模块，以实现发布/订阅通信。

`Writer`类可以充当发布者，`Reader`类可以充当订阅者。我们实现它们来处理严格定义的数据类型，这些数据类型将定义我们的消息。我们在前面的示例中使用的命名管道机制是在字节级别上工作的，并不能保证消息会自动传递。

为了克服这一限制，我们将使用 POSIX 消息队列 API，而不是命名管道。在它们的构造函数中，`Reader`和`Writer`都将接受用于标识消息队列的名称作为主题：

1.  将我们在上一篇文章中创建的`ipc2`目录的内容复制到一个新目录：`ipc3`。

1.  让我们为 POSIX 消息队列 API 创建一个 C++包装器。在编辑器中打开`ipc1.cpp`并添加所需的头文件和常量定义：

```cpp
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>

std::string kQueueName = "/test";
```

1.  然后，定义一个`MessageQueue`类。它将一个消息队列句柄作为其私有数据成员。我们可以使用构造函数和析构函数来管理句柄的安全打开和关闭，使用 C++ RAII 习惯用法。

```cpp
class MessageQueue {
  private:
    mqd_t handle;
  public:
    MessageQueue(const std::string& name, int flags) {
      handle = mq_open(name.c_str(), flags);
      if (handle < 0) {
        throw std::runtime_error("Failed to open a queue for 
         writing");
      }
    }

    MessageQueue(const std::string& name, int flags, int max_count, 
     int max_size) {
      struct mq_attr attrs = { 0, max_count, max_size, 0 };
      handle = mq_open(name.c_str(), flags | O_CREAT, 0666, 
       &attrs);
      if (handle < 0) {
        throw std::runtime_error("Failed to create a queue");
      }
    }

    ~MessageQueue() {
      mq_close(handle);
    }

```

1.  然后，我们定义两个简单的方法来将消息写入队列和从队列中读取消息：

```cpp
    void Send(const char* data, size_t len) {
      if (mq_send(handle, data, len, 0) < 0) {
        throw std::runtime_error("Failed to send a message");
      }
    }

    void Receive(char* data, size_t len) {
      if (mq_receive(handle, data, len, 0) < len) {
        throw std::runtime_error("Failed to receive a message");
      }
    }
};
```

1.  我们现在修改我们的`Writer`和`Reader`类，以适应新的 API。我们的`MessageQueue`包装器完成了大部分繁重的工作，代码更改很小。`Writer`类现在看起来像这样：

```cpp
template<class T>
class Writer {
  private:
    MessageQueue queue;
  public:
    Writer(std::string& name):
      queue(name, O_WRONLY) {}

    void Write(const T& data) {
      queue.Send(reinterpret_cast<const char*>(&data), sizeof(data));
    }
};
```

1.  `Reader`类中的修改更加实质性。我们让它充当订阅者，并将直接从队列中获取和处理消息的逻辑封装到类中：

```cpp
template<class T>
class Reader {
  private:
    MessageQueue queue;
  public:
    Reader(std::string& name):
      queue(name, O_RDONLY) {}

    void Run() {
      T data;
      while(true) {
        queue.Receive(reinterpret_cast<char*>(&data), 
          sizeof(data));
        Callback(data);
      }
    }

  protected:
    virtual void Callback(const T& data) = 0;
};
```

1.  由于我们仍然希望尽可能地保持`Reader`类的通用性，我们将定义一个新的类（`CoordLogger`），它是从`Reader`派生出来的，用于定义我们消息的特定处理方式：

```cpp
class CoordLogger : public Reader<Message> {
  using Reader<Message>::Reader;

  protected:
    void Callback(const Message& data) override {
      std::cout << "Received coordinate " << data << std::endl;
    }
};
```

1.  `DoWrites`代码基本保持不变；唯一的变化是我们使用不同的常量来标识我们的队列：

```cpp
void DoWrites() {
  std::vector<Message> messages {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Writer<Message> writer(kQueueName);
  for (const auto& m : messages) {
    std::cout << "Write " << m << std::endl;
    writer.Write(m);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
```

1.  由于消息处理逻辑已经移动到`Reader`和`CoordLogger`类中，`DoReads`现在就像这样简单：

```cpp
void DoReads() {
 CoordLogger logger(kQueueName);
 logger.Run();
}
```

1.  更新后的`main`函数如下：

```cpp
int main(int argc, char** argv) {
  MessageQueue q(kQueueName, O_WRONLY, 10, sizeof(Message));
  pid_t pid = fork();
  if (pid) {
    DoWrites();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    kill(pid, SIGTERM);
  } else {
    DoReads();
  }
}
```

1.  最后，我们的应用程序需要链接`rt`库。我们通过在`CMakeLists.txt`文件中添加一行来实现这一点：

```cpp
target_link_libraries(ipc3 rt)
```

现在可以构建和运行应用程序了。

# 它是如何工作的...

在我们的应用程序中，我们重用了前面一篇文章中创建的应用程序的大部分代码，*探索 IPC 机制*。为了实现发布-订阅模型，我们需要进行两个重要的更改：

+   使我们的 IPC 基于消息。我们应该能够自动发送和接收消息。一个发布者发送的消息不应该破坏其他发布者发送的消息，订阅者应该能够整体读取消息。

+   让订阅者定义在新消息可用时调用的回调。

为了进行基于消息的通信，我们从命名管道切换到了 POSIX 消息队列 API。消息队列 API 与命名管道的常规基于文件的 API 不同，这就是为什么我们在 Linux 标准库提供的纯 C 接口之上实现了一个 C++包装器。

包装器的主要目标是使用**资源获取即初始化**（**RAII**）习语提供安全的资源管理。我们通过定义通过调用`mq_open`获取队列处理程序的构造函数和使用`mq_close`释放它的析构函数来实现这一点。这样，当`MessageQueue`类的相应实例被销毁时，队列会自动关闭。

包装器类有两个构造函数。一个构造函数用于打开现有队列。它接受两个参数——队列名称和访问标志。第二个构造函数用于创建一个新队列。它接受两个额外的参数——消息长度和队列中消息的最大大小。

在我们的应用程序中，我们在`main`函数中创建一个队列，将`10`作为可以存储在队列中的消息数量。`Message`结构的大小是我们队列中消息的最大大小：

```cpp
  MessageQueue q(kQueueName, O_WRONLY, 10, sizeof(Message));
```

然后，`DoWrites`和`DoReads`函数打开了已经使用相同名称创建的队列。

由于我们的`MessageQueue`类的公共 API 类似于我们用于使用命名管道进行 IPC 的`fstream`接口，因此只需要对写入器和读取器进行最小的更改，使它们能够与另一种 IPC 机制一起工作。我们使用`MessageQueue`的实例而不是`fstream`作为数据成员，保持其他逻辑不变。

为了让订阅者定义他们的回调方法，我们需要修改`Reader`类。我们引入了`Run`方法，而不是读取并返回单个方法的`Read`方法。它循环遍历队列中所有可用的消息。对于每个被读取的方法，它调用一个回调方法：

```cpp
      while(true) {
        queue.Receive(reinterpret_cast<char*>(&data), sizeof(data));
        Callback(data);
      }
```

我们的目标是使`Reader`类通用且可重用于不同类型的消息。然而，并不存在通用的回调。每个回调都是特定的，应该由`Reader`类的用户定义。

解决这个矛盾的一种方法是将`Reader`定义为抽象类。我们将`Callback`方法定义为`virtual`函数：

```cpp
  protected:
    virtual void Callback(const T& data) = 0;
```

现在，由于`Reader`是抽象的，我们无法创建这个类的实例。我们必须继承它，并在一个名为`CoordLogger`的派生类中提供`Callback`方法的定义：

```cpp
  protected:
    void Callback(const Message& data) override {
      std::cout << "Received coordinate " << data << std::endl;
    }
```

请注意，由于`Reader`构造函数接受一个参数，我们需要在继承类中定义构造函数。我们将使用 C++11 标准中添加的继承构造函数：

```cpp
  using Reader<Message>::Reader;
```

现在，有了一个能够处理`Message`类型消息的`CoordLogger`类，我们可以在我们的`DoReads`实现中使用它。我们只需要创建这个类的一个实例并调用它的`Run`方法：

```cpp
  CoordLogger logger(kQueueName);
  logger.Run();
```

当我们运行应用程序时，我们会得到以下输出：

![](img/39b06101-cfed-4a08-9bdd-53fe51e643b6.png)

这个输出与前面的输出并没有太大的不同，但现在实现的可扩展性更强了。`DoReads`方法并没有针对消息做任何特定的操作。它的唯一任务是创建和运行订阅者。所有数据处理都封装在特定的类中。您可以在不改变应用程序架构的情况下添加、替换和组合发布者和订阅者。

# 还有更多...

POSIX 消息队列 API 提供了消息队列的基本功能，但它也有许多限制。使用一个消息队列无法向多个订阅者发送消息。您必须为每个订阅者创建一个单独的队列，否则只有一个订阅者从队列中读取消息。

有许多详细的消息队列和发布-订阅中间件可用作外部库。ZeroMQ 是一个功能强大、灵活且轻量级的传输库。这使它成为使用数据交换的发布-订阅模型构建的嵌入式应用程序的理想选择。

# 使用 C++ lambda 进行回调

在发布-订阅模型中，订阅者通常注册一个回调，当发布者的消息传递给订阅者时会被调用。

在前面的示例中，我们创建了一个使用继承和抽象类注册回调的机制。这不是 C++中唯一可用的机制。C++中提供的 lambda 函数，从 C++11 标准开始，可以作为替代解决方案。这消除了定义派生类所需的大量样板代码，并且在大多数情况下，允许开发人员以更清晰的方式表达他们的意图。

在这个示例中，我们将学习如何使用 C++ lambda 函数来定义回调。

# 如何做...

我们将使用前面示例中大部分代码，*学习消息队列和发布-订阅模型*。我们将修改`Reader`类以接受回调作为参数。通过这种修改，我们可以直接使用`Reader`，而不需要依赖继承来定义回调：

1.  将我们在前面示例中创建的`ipc3`目录的内容复制到一个新目录`ipc4`中。

1.  保持所有代码不变，除了`Reader`类。让我们用以下代码片段替换它：

```cpp
template<class T>
class Reader {
  private:
    MessageQueue queue;
    void (*func)(const T&);
  public:
    Reader(std::string& name, void (*func)(const T&)):
      queue(name, O_RDONLY), func(func) {}

    void Run() {
      T data;
      while(true) {
        queue.Receive(reinterpret_cast<char*>(&data), 
         sizeof(data));
        func(data);
      }
    }
};
```

1.  现在我们的`Reader`类已经改变，我们可以更新`DoReads`方法。我们可以使用 lambda 函数来定义一个回调处理程序，并将其传递给`Reader`的构造函数：

```cpp
void DoReads() {
  Reader<Message> logger(kQueueName, [](const Message& data) {
    std::cout << "Received coordinate " << data << std::endl;
  });
  logger.Run();
}
```

1.  `CoordLogger`类不再需要，因此我们可以完全从我们的代码中安全地删除它。

1.  您可以构建和运行应用程序。

# 它是如何工作的...

在这个示例中，我们修改了之前定义的`Reader`类，以接受其构造函数中的额外参数。这个参数有一个特定的数据类型——一个指向函数的指针，它将被用作回调：

```cpp
Reader(std::string& name, void (*func)(const T&)):
```

处理程序存储在数据字段中以供将来使用：

```cpp
void (*func)(const T&);
```

现在，每当`Run`方法读取消息时，它会调用存储在`func`字段中的函数，而不是我们需要重写的`Callback`方法：

```cpp
queue.Receive(reinterpret_cast<char*>(&data), sizeof(data));
func(data);
```

将`Callback`函数去掉使`Reader`成为一个具体的类，我们可以直接创建它的实例。然而，现在我们需要在它的构造函数中提供一个处理程序作为参数。

使用纯 C，我们必须定义一个`named`函数并将其名称作为参数传递。在 C++中，这种方法也是可能的，但 C++还提供了匿名函数或 lambda 函数的机制，可以直接在现场定义。

在`DoReads`方法中，我们创建一个 lambda 函数，并直接将其传递给`Reader`的构造函数：

```cpp
  Reader<Message> logger(kQueueName, [](const Message& data) {
 std::cout << "Received coordinate " << data << std::endl;
 });
```

构建和运行应用程序会产生以下输出：

![](img/6c9b5c11-a5ea-4454-8e09-6d3b1ff2e1cd.png)

正如我们所看到的，它与我们在前面的示例中创建的应用程序的输出相同。然而，我们用更少的代码以更可读的方式实现了它。

Lambda 函数应该明智地使用。如果保持最小，它们会使代码更易读。如果一个函数变得比五行更长，请考虑使用命名函数。

# 还有更多...

C++提供了灵活的机制来处理类似函数的对象，并将它们与参数绑定在一起。这些机制被广泛用于转发调用和构建函数适配器。[`en.cppreference.com/w/cpp/utility/functional`](https://en.cppreference.com/w/cpp/utility/functional)上的*函数对象*页面是深入了解这些主题的好起点。

# 探索数据序列化

我们已经在第三章 *使用不同的架构*中简要涉及了序列化的一些方面。在数据交换方面，序列化是至关重要的。序列化的任务是以一种可以被接收应用程序明确读取的方式表示发送应用程序发送的所有数据。鉴于发送方和接收方可能在不同的硬件平台上运行，并通过各种传输链路连接 - **传输控制协议**/**互联网协议**（**TCP/IP**）网络，**串行外围接口**（**SPI**）总线或串行链路，这个任务并不那么简单。

根据要求实现序列化的方式有很多种，这就是为什么 C++标准库没有提供序列化的原因。

在这个示例中，我们将学习如何在 C++应用程序中实现简单的通用序列化和反序列化。

# 如何做...

序列化的目标是以一种可以在另一个系统或另一个应用程序中正确解码的方式对任何数据进行编码。开发人员通常面临的典型障碍如下：

+   平台特定的差异，如数据对齐和字节顺序。

+   内存中分散的数据；例如，链表的元素可能相距甚远。由指针连接的断开块的表示对于内存是自然的，但在传输到另一个进程时，无法自动转换为字节序列。

解决这个问题的通用方法是让一个类定义将其内容转换为序列化形式并从序列化形式中恢复类实例的函数。

在我们的应用程序中，我们将重载输出流的`operator<<`和输入流的`operator>>`，分别用于序列化和反序列化数据：

1.  在您的`~/test`工作目录中，创建一个名为`stream`的子目录。

1.  使用您喜欢的文本编辑器在`stream`子目录中创建一个`stream.cpp`文件。

1.  从定义要序列化的数据结构开始：

```cpp
#include <iostream>
#include <sstream>
#include <list>

struct Point {
  int x, y;
};

struct Paths {
  Point source;
  std::list<Point> destinations;
};
```

1.  接下来，我们重载`<<`和`>>`运算符，负责将`Point`对象分别写入和从流中读取。对于`Point`数据类型，输入以下内容：

```cpp
std::ostream& operator<<(std::ostream& o, const Point& p) {
  o << p.x << " " << p.y << " ";
  return o;
}

std::istream& operator>>(std::istream& is, Point& p) {
  is >> p.x;
  is >> p.y;
  return is;
}
```

1.  它们后面是`Paths`对象的`<<`和`>>`重载运算符：

```cpp
std::ostream& operator<<(std::ostream& o, const Paths& paths) {
  o << paths.source << paths.destinations.size() << " ";
  for (const auto& x : paths.destinations) {
    o << x;
  }
  return o;
}

std::istream& operator>>(std::istream& is, Paths& paths) {
  size_t size;
  is >> paths.source;
  is >> size;
  for (;size;size--) {
    Point tmp;
    is >> tmp;
    paths.destinations.push_back(tmp);
  }
  return is;
}
```

1.  现在，让我们在`main`函数中总结一切：

```cpp
int main(int argc, char** argv) {
  Paths paths = {{0, 0}, {{1, 1}, {0, 1}, {1, 0}}};

  std::stringstream in;
  in << paths;
  std::string serialized = in.str();
  std::cout << "Serialized paths into the string: ["
            << serialized << "]" << std::endl;

  std::stringstream out(serialized);
  Paths paths2;
  out >> paths2;
  std::cout << "Original: " << paths.destinations.size()
            << " destinations" << std::endl;
  std::cout << "Restored: " << paths2.destinations.size()
            << " destinations" << std::endl;

  return 0;
}
```

1.  最后，创建一个包含程序构建规则的`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(stream)
add_executable(stream stream.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")

set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

现在您可以构建和运行应用程序了。

# 它是如何工作的...

在我们的测试应用程序中，我们定义了一种数据类型，用于表示从源点到多个目标点的路径。我们故意使用了分散在内存中的分层结构，以演示如何以通用方式解决这个问题。

如果我们对性能没有特定要求，序列化的一种可能方法是以文本格式存储数据。除了它的简单性外，它还有两个主要优点：

+   文本编码自动解决了与字节顺序、对齐和整数数据类型大小相关的所有问题。

+   它可供人类阅读。开发人员可以使用序列化数据进行调试，而无需任何额外的工具。

为了使用文本表示，我们可以使用标准库提供的输入和输出流。它们已经定义了写入和读取格式化数字的函数。

`Point`结构被定义为两个整数值：`x`和`y`。我们重写了这种数据类型的`operator<<`，以便写入`x`和`y`值，后面跟着空格。这样，我们可以在重写的`operator>>`操作中按顺序读取它们。

`Path`数据类型有点棘手。它包含一个目的地的链表。由于列表的大小可能会变化，我们需要在序列化其内容之前写入列表的实际大小，以便在反序列化期间能够正确恢复它：

```cpp
  o << paths.source << paths.destinations.size() << " ";
```

由于我们已经重写了`Point`方法的`<<`和`>>`操作符，我们可以在`Paths`方法中使用它们。这样，我们可以将`Point`对象写入流或从流中读取，而不知道它们的数据字段的内容。层次化数据结构被递归处理：

```cpp
  for (const auto& x : paths.destinations) {
    o << x;
  }
```

最后，我们测试我们的序列化和反序列化实现。我们创建一个`Paths`对象的示例实例：

```cpp
Paths paths = {{0, 0}, {{1, 1}, {0, 1}, {1, 0}}};
```

然后，我们使用`std::stringstream`数据类型将其内容序列化为字符串：

```cpp
  std::stringstream in;
  in << paths;
  std::string serialized = in.str();
```

接下来，我们创建一个空的`Path`对象，并将字符串的内容反序列化到其中：

```cpp
  Paths paths2;
  out >> paths2;
```

最后，我们检查它们是否匹配。当我们运行应用程序时，我们可以使用以下输出来进行检查：

![](img/f862eddd-5a3d-4234-8b4b-7edb0bf4c0ac.png)

恢复对象的`destinations`列表的大小与原始对象的`destinations`列表的大小相匹配。我们还可以看到序列化数据的内容。

这个示例展示了如何为任何数据类型构建自定义序列化。可以在没有任何外部库的情况下完成。然而，在性能和内存效率要求的情况下，使用第三方序列化库将是更实用的方法。

# 还有更多...

从头开始实现序列化是困难的。cereal 库在[`uscilab.github.io/cereal/`](https://uscilab.github.io/cereal/)和 boost 库在[`www.boost.org/doc/libs/1_71_0/libs/serialization/doc/index.html`](https://www.boost.org/doc/libs/1_71_0/libs/serialization/doc/index.html)提供了一个基础，可以帮助您更快速、更容易地向应用程序添加序列化。

# 使用 FlatBuffers 库

序列化和反序列化是一个复杂的主题。虽然临时序列化看起来简单直接，但要使其通用、易于使用和快速是困难的。幸运的是，有一些库处理了所有这些复杂性。

在这个示例中，我们将学习如何使用其中一个序列化库：FlatBuffers。它是专为嵌入式编程设计的，使序列化和反序列化内存高效且快速。

FlatBuffers 使用**接口定义语言**（**IDL**）来定义数据模式。该模式描述了我们需要序列化的数据结构的所有字段。当设计模式时，我们使用一个名为**flatc**的特殊工具来为特定的编程语言生成代码，这在我们的情况下是 C++。

生成的代码以序列化形式存储所有数据，并为开发人员提供所谓的**getter**和**setter**方法来访问数据字段。getter 在使用时执行反序列化。将数据存储在序列化形式中使得 FlatBuffers 真正的内存高效。不需要额外的内存来存储序列化数据，并且在大多数情况下，反序列化的开销很低。

在这个示例中，我们将学习如何在我们的应用程序中开始使用 FlatBuffers 进行数据序列化。

# 如何做...

FlatBuffers 是一组工具和库。在使用之前，我们需要下载并构建它：

1.  下载最新的 FlatBuffers 存档，可在[`codeload.github.com/google/flatbuffers/zip/master`](https://codeload.github.com/google/flatbuffers/zip/master)下载，并将其提取到`test`目录中。这将创建一个名为`flatbuffers-master`的新目录。

1.  切换到构建控制台，将目录更改为`flatbuffers-master`，并运行以下命令来构建和安装库和工具。确保以 root 用户身份运行。如果没有，请按*Ctrl* + *C*退出用户 shell：

```cpp
# cmake .
# make
# make install
```

现在，我们准备在我们的应用程序中使用 FlatBuffers。让我们重用我们在以前的配方中创建的应用程序：

1.  将`ipc4`目录的内容复制到新创建的名为`flat`的目录中。

1.  创建一个名为`message.fbs`的文件，打开它并输入以下代码：

```cpp
 struct Message {
 x: int;
 y: int;
}
```

1.  从`message.fbs`生成 C++源代码，运行以下命令：

```cpp
$ flatc --cpp message.fbs
```

这将创建一个名为`message_generated.h`的新文件。

1.  在编辑器中打开`ipc1.cpp`。在`mqueue.h`包含之后，添加一个`include`指令用于生成的`message_generated.h`文件：

```cpp
#include <mqueue.h>

#include "message_generated.h"
```

1.  现在，摆脱我们代码中声明的`Message`结构。我们将使用 FlatBuffers 模式文件中生成的结构。

1.  由于 FlatBuffers 使用 getter 方法而不是直接访问结构字段，我们需要修改我们重新定义的`operator<<`操作的主体，用于将点数据打印到控制台。更改很小——我们只是为每个数据字段添加括号：

```cpp
 std::ostream& operator<<(std::ostream& o, const Message& m) {
  o << "(x=" << m.x() << ", y=" << m.y() << ")";
}
```

1.  代码修改已完成。现在，我们需要更新构建规则以链接 FlatBuffers 库。打开`CMakeLists.txt`，并输入以下行：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(flat)
add_executable(flat ipc1.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS_RELEASE "--std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} -g -DDEBUG")
target_link_libraries(flat rt flatbuffers)

set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

1.  切换到构建控制台，然后切换到用户 shell：

```cpp
# su - user
$
```

1.  构建并运行应用程序。

# 工作原理...

FlatBuffers 是一个外部库，不在 Ubuntu 软件包存储库中，因此我们需要先下载、构建和安装它。安装完成后，我们可以在我们的应用程序中使用它。

我们使用了我们为*使用 C++ lambda 进行回调*配方创建的现有应用程序作为起点。在该应用程序中，我们定义了一个名为`Message`的结构，用于表示我们用于 IPC 的数据类型。我们将用 FlatBuffers 提供的新数据类型替换它。这种新数据类型将为我们执行所有必要的序列化和反序列化。

我们完全从我们的代码中删除了`Message`结构的定义。相反，我们生成了一个名为`message_generated.h`的新头文件。这个文件是从`message.fbs`的 FlatBuffers 模式文件生成的。这个模式文件定义了一个具有两个整数字段`x`和`y`的结构：

```cpp
  x: int;
  y: int;
```

这个定义与我们之前的定义相同；唯一的区别是语法——FlatBuffers 的模式使用冒号将字段名与字段类型分隔开。

一旦`message_generated.h`由`flatc`命令调用创建，我们就可以在我们的代码中使用它。我们添加适当的`include`如下：

```cpp
#include "message_generated.h"
```

生成的消息与我们之前使用的消息结构相同，但正如我们之前讨论的，FlatBuffers 以序列化形式存储数据，并且需要在运行时对其进行反序列化。这就是为什么，我们不直接访问数据字段，而是使用`x()`访问器方法而不是只是`x`，以及`y()`访问器方法而不只是`y`。

我们唯一使用直接访问消息数据字段的地方是在重写的`operator<<`操作中。我们添加括号将直接字段访问转换为调用 FlatBuffers 的 getter 方法：

```cpp
  o << "(x=" << m.x() << ", y=" << m.y() << ")";
```

让我们构建并运行应用程序。我们将看到以下输出：

![](img/96e44321-cffb-4c7f-9989-b51539a077ff.png)

输出与我们自定义消息数据类型的输出相同。在我们的代码中只进行了少量修改，我们就将我们的消息迁移到了 FlatBuffers。现在，我们可以在多台计算机上运行我们的发布者和订阅者——这些计算机可以具有不同的架构，并确保它们每个都正确解释消息。

# 还有更多...

除了 FlatBuffers 之外，还有许多其他序列化库和技术，每种都有其优缺点。请参考[C++序列化 FAQ](https://isocpp.org/wiki/faq/serialization)以更好地了解如何在您的应用程序中设计序列化。
