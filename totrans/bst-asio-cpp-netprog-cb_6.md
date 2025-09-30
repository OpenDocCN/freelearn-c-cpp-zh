# 第六章。其他主题

在本章中，我们将介绍以下食谱：

+   使用复合缓冲区进行散射/收集操作

+   使用定时器

+   获取和设置套接字选项

+   执行基于流的 I/O

# 简介

本章包含四个食谱，这些食谱与之前章节中展示的核心 Boost.Asio 概念略有不同，涵盖了大多数典型用例。但这并不意味着本章中展示的食谱不重要。相反，它们非常重要，甚至在某些特定情况下是关键的。然而，在典型的分布式应用程序中，它们的使用频率较低。

尽管大多数应用程序可能不需要散射/收集 I/O 操作和复合缓冲区，但对于某些将消息的不同部分保存在单独缓冲区中的应用程序来说，这些功能可能非常实用和方便。

Boost.Asio 定时器是一个强大的工具，允许测量时间间隔。通常，这用于为可能持续很长时间的操作设置截止日期，并在运行一定时间后未完成这些操作时中断它们。考虑到 Boost.Asio 不提供为可能长时间运行的操作指定超时的方式，这种工具对于许多分布式应用程序来说是至关重要的。此外，Boost.Asio 提供的定时器还可以用于解决与网络通信无关的其他任务。

允许获取和设置套接字选项的工具同样非常重要。在开发简单的网络应用程序时，开发者可能对套接字配备的默认选项值感到非常满意，这些选项值在套接字对象实例化时自动设置。然而，在更复杂的情况下，可能绝对有必要通过自定义选项值来重新配置套接字。

Boost.Asio 类封装了套接字并提供了一个类似流的接口，这使得我们能够创建简单而优雅的分布式应用程序。而且，简单性被认为是优秀软件的关键特征之一。

现在，让我们详细考虑所提到的主题。

# 使用复合缓冲区进行散射/收集操作

第二章中“使用固定长度 I/O 缓冲区”的食谱介绍了简单的 I/O 缓冲区，但只是略微触及了散射/收集操作和复合缓冲区。在本食谱中，我们将更详细地考虑这个主题。

复合缓冲区基本上是一个复杂的缓冲区，由两个或更多简单缓冲区（内存的连续块）组成，这些缓冲区分布在进程的地址空间中。这种缓冲区在两种情况下特别有用。

第一种情况是当应用程序需要缓冲区来存储在将其发送到远程应用程序之前的消息，或者接收远程应用程序发送的消息。问题是消息的大小如此之大，以至于可能由于进程地址空间碎片化而无法分配足够存储它的单个连续缓冲区。在这种情况下，分配多个较小的缓冲区，其总大小足以存储数据，并将它们组合成一个复合缓冲区是解决问题的良好方法。

另一种情况实际上是第一种情况的反转。由于应用程序设计的特定性，要发送到远程应用程序的消息被分成几个部分并存储在不同的缓冲区中，或者如果需要将接收自远程应用程序的消息分成几个部分，每个部分都应该存储在单独的缓冲区中以供进一步处理。在这两种情况下，将几个缓冲区组合成一个复合缓冲区，然后使用分散发送或收集接收操作将是解决问题的良好方法。

在这个菜谱中，我们将看到如何创建复合缓冲区并在分散/收集 I/O 操作中使用它们。

## 准备工作...

为了理解本菜谱中呈现的内容，熟悉第二章中“使用固定长度 I/O 缓冲区”菜谱的内容是有益的，该菜谱提供了 Boost.Asio 固定长度 I/O 缓冲区的一般概述。因此，建议在继续进行此菜谱之前熟悉“使用固定长度 I/O 缓冲区”菜谱。

## 如何操作...

让我们考虑两个算法和相应的代码示例，描述了如何创建和准备用于 Boost.Asio I/O 操作的复合缓冲区。第一个算法处理用于收集输出操作的复合缓冲区，第二个算法用于分散输入操作。

### 准备用于收集输出操作的复合缓冲区

以下是一个算法和相应的代码示例，描述了如何准备用于与套接字方法（如`asio::ip::tcp::socket::send()`或自由函数如`asio::write()`）执行输出操作的复合缓冲区：

1.  分配所需数量的内存缓冲区以执行当前任务。请注意，此步骤不涉及任何来自 Boost.Asio 的功能或数据类型。

1.  用要输出的数据填充缓冲区。

1.  创建一个满足`ConstBufferSequence`或`MultipleBufferSequence`概念要求的类的实例，代表一个复合缓冲区。

1.  将简单缓冲区添加到复合缓冲区中。每个简单缓冲区应表示为`asio::const_buffer`或`asio::mutable_buffer`类的实例。

1.  复合缓冲区已准备好与 Boost.Asio 输出函数一起使用。

假设我们想要将字符串`Hello my friend!`发送到远程应用程序，但我们的消息被分割成了三部分，并且每一部分都存储在单独的缓冲区中。我们可以做的是将我们的三个缓冲区表示为一个复合缓冲区，然后，在输出操作中使用它。以下是如何在以下代码中实现它的方法：

```cpp
#include <boost/asio.hpp>

using namespace boost;

int main()
{
  // Steps 1 and 2\. Create and fill simple buffers.
  const char* part1 = "Hello ";
  const char* part2 = "my ";
  const char* part3 = "friend!";

  // Step 3\. Create an object representing a composite buffer.
  std::vector<asio::const_buffer> composite_buffer;

  // Step 4\. Add simple buffers to the composite buffer.
  composite_buffer.push_back(asio::const_buffer(part1, 6));
  composite_buffer.push_back(asio::const_buffer(part2, 3));
  composite_buffer.push_back(asio::const_buffer(part3, 7));

  // Step 5\. Now composite_buffer can be used with Boost.Asio
  // output operations as if it was a simple buffer represented
  // by contiguous block of memory.

  return 0;
}
```

### 准备用于输入操作的复合缓冲区

以下是一个算法和相应的代码示例，描述了如何准备用于`socket`方法（如`asio::ip::tcp::socket::receive()`或自由函数如`asio::read()`）输入操作的复合缓冲区：

1.  分配所需的内存缓冲区数量以执行当前任务。缓冲区大小的总和必须等于或大于预期接收到的消息的大小。请注意，此步骤不涉及任何 Boost.Asio 的功能或数据类型。

1.  创建一个满足`MutableBufferSequence`概念要求的类的实例，该类表示一个复合缓冲区。

1.  将简单缓冲区添加到复合缓冲区中。每个简单缓冲区应表示为`asio::mutable_buffer`类的实例。

1.  复合缓冲区已准备好用于 Boost.Asio 输入操作。

让我们想象一个假设的情况，我们想要从服务器接收 16 字节长的消息。然而，我们没有可以容纳整个消息的缓冲区。相反，我们有三个缓冲区：6 字节、3 字节和 7 字节长。为了创建一个可以接收 16 字节数据的缓冲区，我们可以将我们的三个小缓冲区合并成一个复合缓冲区。以下是如何在以下代码中实现它的方法：

```cpp
#include <boost/asio.hpp>

using namespace boost;

int main()
{
  // Step 1\. Allocate simple buffers.
  char part1[6];
  char part2[3];
  char part3[7];

  // Step 2\. Create an object representing a composite buffer.
  std::vector<asio::mutable_buffer> composite_buffer;

  // Step 3\. Add simple buffers to the composite buffer object.
  composite_buffer.push_back(asio::mutable_buffer(part1,
  sizeof(part1)));
  composite_buffer.push_back(asio::mutable_buffer(part2,
  sizeof(part2)));
  composite_buffer.push_back(asio::mutable_buffer(part3,
  sizeof(part3)));

  // Now composite_buffer can be used with Boost.Asio 
  // input operation as if it was a simple buffer 
  // represented by contiguous block of memory.

  return 0;
}
```

## 它是如何工作的…

让我们看看第一个示例是如何工作的。它从分配三个只读缓冲区开始，这些缓冲区填充了消息字符串`Hello my friend!`的部分。

在下一步中，创建了一个`std::vector<asio::const_buffer>`类的实例，这是复合缓冲区的具体体现。这个实例被赋予了相应的名称，`composite_buffer`。因为`std::vector<asio::const_buffer>`类满足`ConstBufferSequence`的要求，所以它的对象可以用作复合缓冲区，并且可以作为表示数据源的参数传递给 Boost.Asio 的聚集输出函数和方法。

在第 4 步中，我们的三个缓冲区中的每一个都被表示为`asio::const_buffer`类的实例，并添加到复合缓冲区中。因为所有与固定大小缓冲区一起工作的 Boost.Asio 输出函数和方法都设计为也可以与复合缓冲区一起工作，所以我们的`composite_buffer`对象可以像简单缓冲区一样使用。

第二个示例与第一个示例非常相似。唯一的区别是，由于在这个示例中创建的复合缓冲区旨在用作数据目的地（而不是像第一个示例中的数据源），因此添加到其中的三个简单缓冲区被创建为可写缓冲区，并且在添加到复合缓冲区时表示为 `asio::mutable_buffer` 类的实例。

关于第二个示例的另一件事是，由于在这个示例中创建的复合缓冲区是由可变缓冲区组成的，因此它可以用于聚集输出和分散输入操作。在这个特定的示例中，初始缓冲区（`part1`、`part2` 和 `part3`）没有填充任何数据，它们包含垃圾数据；因此，除非它们填充了有意义的数据，否则在输出操作中使用它们是没有意义的。

## 参见

+   第二章中的*使用固定长度 I/O 缓冲区*食谱提供了有关固定大小简单缓冲区更多信息。

+   第二章 *I/O 操作*中的*使用可扩展的流式 I/O 缓冲区*食谱演示了如何使用 Boost.Asio 提供的类，代表不同类型的缓冲区——可扩展缓冲区。

# 使用计时器

时间是软件系统（尤其是分布式应用程序）的一个重要方面。因此，硬件计时器——一种用于测量时间间隔的设备——是任何计算机和所有现代操作系统的基本组件，所有现代操作系统都提供了允许应用程序使用它的接口。

与计时器相关的有两个典型用例。第一个用例假设应用程序想要知道当前时间，并要求操作系统找出它。第二个用例是当应用程序要求操作系统在经过一定时间后通知它（通常是通过调用回调函数）时。

当涉及到使用 Boost.Asio 开发分布式应用程序时，第二个用例尤为重要，因为计时器是实现异步操作超时机制的唯一方式。

Boost.Asio 库包含几个实现计时器的类，我们将在本食谱中考虑这些类。

## 如何做到这一点...

Boost.Asio 库提供了两个模板类来实现计时器。其中之一是 `asio::basic_deadline_timer<>`，在 Boost.Asio 1.49 版本发布之前，这是唯一可用的。在版本 1.49 中，引入了第二个计时器 `asio::basic_waitable_timer<>` 类模板。

`asio::basic_deadline_timer<>` 类模板被设计为与 Boost.Chrono 库兼容，并在内部依赖于它提供的功能。这个模板类有些过时，并且功能有限。因此，我们不会在本食谱中考虑它。

相反，一个较新的 `asio::basic_waitable_timer<>` 类模板，与 C++11 `chrono` 库兼容，更加灵活，并提供了更多功能。Boost.Asio 包括三个 `typedefs`，用于从 `asio::basic_waitable_timer<>` 模板类泛型派生的类：

```cpp
typedef basic_waitable_timer< std::chrono::system_clock >
   system_timer;
typedef basic_waitable_timer< std::chrono::steady_clock > 
   steady_timer;
typedef basic_waitable_timer< std::chrono::high_resolution_clock >
   high_resolution_timer;
```

`asio::system_timer` 类基于 `std::chrono::system_clock` 类，它代表一个系统范围内的实时时钟。这个时钟（以及相应的计时器）会受到当前系统时间外部变化的影响。因此，当我们需要设置一个在某个绝对时间点（例如，13 小时 15 分钟 45 秒）通知我们的计时器时，`asio::system_timer` 类是一个好选择，考虑到计时器设置后系统时钟的偏移。然而，这个计时器在测量时间间隔（例如，从现在起 35 秒）方面并不擅长，因为系统时钟的偏移可能会导致计时器比实际间隔早或晚到期。

`asio::steady_timer` 类基于 `std::chrono::steady_clock` 类，它代表一个不受系统时钟变化影响的稳定时钟。这意味着 `asio::steady_timer` 是测量间隔的一个好选择。

最后一个计时器类 `asio::high_resolution_timer` 是基于 `std::chrono::high_resolution_clock` 类，它代表一个高精度系统时钟。在需要高精度时间测量的情况下可以使用它。

在使用 Boost.Asio 库实现的分布式应用程序中，计时器通常用于实现异步操作的超时周期。异步操作开始后（例如，`asio::async_read()`），应用程序将启动一个计时器，该计时器在一段时间后到期，即“超时周期”。当计时器到期时，应用程序检查异步操作是否已完成，如果没有完成，则认为操作超时，并将其取消。

由于稳定计时器不受系统时钟偏移的影响，它是实现超时机制的最佳选择。

### 注意

注意，在某些平台上，稳定的时钟不可用，代表 `std::chrono::steady_clock` 的相应类表现出与 `std::chrono::system_clock` 相同的行为，这意味着它就像后者一样，会受到系统时钟变化的影响。建议参考平台和相应的 C++标准库实现文档，以了解稳定的时钟是否真正是“稳定的”。

让我们考虑一个多少有些不切实际但具有代表性的示例应用程序，该应用程序演示了如何创建、启动和取消 Boost.Asio 计时器。在我们的示例中，我们将逐一创建和启动两个稳定的计时器。当第一个计时器到期时，我们将取消第二个计时器，在它有机会到期之前。

我们从包含必要的 Boost.Asio 头文件和 `using` 指令开始我们的示例应用程序：

```cpp
#include <boost/asio/steady_timer.hpp>
#include <iostream>

using namespace boost;
```

接下来，我们定义我们应用程序中唯一的组件：`main()` 入口点函数：

```cpp
int main()
{
```

就像几乎所有的非平凡 Boost.Asio 应用程序一样，我们需要一个 `asio::io_service` 类的实例：

```cpp
  asio::io_service ios;
```

然后，我们创建并启动第一个 `t1` 定时器，该定时器被设置为在 2 秒后过期：

```cpp
  asio::steady_timer t1(ios);
  t1.expires_from_now(std::chrono::seconds(2));
```

然后，我们创建并启动第二个 `t2` 定时器，该定时器被设置为在 5 秒后过期。它应该肯定比第一个定时器晚过期：

```cpp
  asio::steady_timer t2(ios);
  t2.expires_from_now(std::chrono::seconds(5));
```

现在，我们定义并设置一个回调函数，当第一个定时器过期时将被调用：

```cpp
   t1.async_wait(&t2 {
      if (ec == 0) {
         std::cout << "Timer #2 has expired!" << std::endl;
      }
      else if (ec == asio::error::operation_aborted) {
         std::cout << "Timer #2 has been cancelled!" 
                     << std::endl;
      }
      else {
         std::cout << "Error occured! Error code = "
            << ec.value()
            << ". Message: " << ec.message() 
                      << std::endl;
      }

      t2.cancel();
   });
```

然后，我们定义并设置另一个回调函数，当第二个定时器过期时将被调用：

```cpp
   t2.async_wait([](boost::system::error_code ec) {
      if (ec == 0) {
         std::cout << "Timer #2 has expired!" << std::endl;
      }
      else if (ec == asio::error::operation_aborted) {
         std::cout << "Timer #2 has been cancelled!" 
<< std::endl;
      }
      else {
         std::cout << "Error occured! Error code = "
            << ec.value()
            << ". Message: " << ec.message() 
<< std::endl;
      }
   });
```

在最后一步，我们在 `asio::io_service` 类的实例上调用 `run()` 方法：

```cpp
  ios.run();

  return 0;
}
```

现在，我们的示例应用程序已经准备好了。

## 它是如何工作的…

现在，让我们跟踪应用程序的执行路径，以更好地理解它是如何工作的。

`main()` 函数从创建 `asio::io_service` 类的实例开始。我们需要它，因为就像套接字、接受者、解析器以及由 Boost.Asio 库定义的其他使用操作系统服务的组件一样，定时器也需要 `asio::io_service` 类的实例。

在下一步中，我们实例化了第一个定时器 `t1`，然后对其调用 `expires_from_now()` 方法。此方法将定时器切换到非过期状态并启动它。它接受一个表示定时器应在之后过期的时距的参数。在我们的示例中，我们传递一个表示 2 秒时距的参数，这意味着从定时器开始的那一刻起，2 秒后定时器将过期，所有等待此定时器过期事件的等待者都将被通知。

接下来，创建第二个名为 `t2` 的定时器，然后启动它并设置为在 5 秒后过期。

当两个定时器都启动后，我们异步等待定时器的过期事件。换句话说，我们在每个定时器上注册回调函数，这些回调函数将在相应的定时器过期时被调用。为此，我们调用定时器的 `async_wait()` 方法，并将回调函数的指针作为参数传递。`async_wait()` 方法期望其参数是一个具有以下签名的函数的指针：

```cpp
void callback(
  const boost::system::error_code& ec);
```

回调函数接受一个单个的 `ec` 参数，它指定了等待完成的状况。在我们的示例应用程序中，我们使用 lambda 函数作为两个定时器的过期回调。

当两个定时器的过期回调都设置好后，在 `ios` 对象上调用 `run()` 方法。该方法会阻塞，直到两个定时器都过期。在调用 `run()` 方法的线程上下文中，将使用该线程来调用过期回调。

当第一个计时器到期时，相应的回调函数被调用。它检查等待完成状态，并向标准输出流输出相应的消息。然后通过在`t2`对象上调用`cancel()`方法取消第二个计时器。

取消第二个计时器导致到期回调以状态码调用，通知计时器在到期之前被取消。第二个计时器的到期回调检查到期状态，并向标准输出流输出相应的消息，然后返回。

当两个回调都完成后，`run()`方法返回，`main()`函数的执行继续到末尾。这是应用程序执行完成的时候。

# 获取和设置套接字选项

可以通过更改其各种选项的值来配置套接字的属性及其行为。当套接字对象被实例化时，其选项具有默认值。在许多情况下，默认配置的套接字是完美的选择，而在其他情况下，可能需要通过更改其选项的值来微调套接字，以便满足应用程序的要求。

在这个配方中，我们将了解如何使用 Boost.Asio 获取和设置套接字选项。

## 准备工作...

此配方假设熟悉第一章中提供的内容，*基础知识*。

## 如何操作...

每个可以通过 Boost.Asio 提供的功能设置或获取其值的套接字选项都由一个单独的类表示。支持 Boost.Asio 设置或获取套接字选项的类的完整列表可以在 Boost.Asio 文档页面上找到，网址为[`www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/socket_base.html`](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/socket_base.html)。

注意，此页面上列出的表示套接字选项的类比可以从本地套接字（底层操作系统的对象）设置或获取的选项要少。这是因为 Boost.Asio 仅支持有限数量的套接字选项。为了设置或获取其他套接字选项的值，开发者可能需要通过添加表示所需选项的类来扩展 Boost.Asio 库。然而，关于扩展 Boost.Asio 库的主题超出了本书的范围。我们将专注于如何使用库中开箱即用的套接字选项进行操作。

让我们考虑一个假设的情况，我们希望将套接字接收缓冲区的大小增加到现在的两倍。为此，我们首先需要获取缓冲区的当前大小，然后将其乘以二，最后将乘法后的值设置为新的接收缓冲区大小。

以下示例演示了如何在以下代码中执行此操作：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

int main()
{
  try {
    asio::io_service ios;

    // Create and open a TCP socket.
    asio::ip::tcp::socket sock(ios, asio::ip::tcp::v4());

    // Create an object representing receive buffer
      // size option.
    asio::socket_base::receive_buffer_size cur_buf_size;

    // Get the currently set value of the option. 
    sock.get_option(cur_buf_size);

    std::cout << "Current receive buffer size is "
      << cur_buf_size.value() << " bytes."
      << std::endl;

    // Create an object representing receive buffer
      // size option with new value.
    asio::socket_base::receive_buffer_size
      new_buf_size(cur_buf_size.value() * 2);

    // Set new value of the option.
    sock.set_option(new_buf_size);

    std::cout << "New receive buffer size is "
      << new_buf_size.value() << " bytes."
      << std::endl;
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的...

我们的示例由一个单一组件组成：`main()`入口点函数。这个函数从创建一个`asio::io_service`类的实例开始。然后，使用这个实例创建一个代表 TCP 套接字的对象。

注意套接字类构造函数的使用，它创建并*打开*了套接字。在我们能够获取或设置特定套接字对象上的选项之前，相应的套接字必须被打开。这是因为，在 Boost.Asio 套接字对象打开之前，相应操作系统的底层*套接字*对象尚未分配，因此没有可以设置选项或从中获取选项的对象。

接下来，实例化了一个`asio::socket_base::receive_buffer_size`类的实例。这个类代表了一个控制套接字接收缓冲区大小的选项。为了获取该选项的当前值，需要在套接字对象上调用`get_option()`方法，并将选项对象的引用作为参数传递给它。

`get_option()`方法通过传递给它的参数类型推断出请求的选项。然后，它将相应的选项值存储在选项对象中并返回。可以通过调用表示相应选项的对象的`value()`方法来从对象中获取选项的值，该方法返回选项的值。

在获取接收缓冲区大小选项的当前值并将其输出到标准输出流之后，为了设置该选项的新值，`main()`函数继续创建一个名为`new_buf_size`的`asio::socket_base::receive_buffer_size`类的新实例。这个实例与第一个实例`cur_buf_size`具有相同的选项，但包含新值。新的选项值作为构造函数的参数传递给选项对象。

在构造包含新的接收缓冲区大小选项值的选项对象之后，将对该对象的引用作为参数传递给套接字的`set_option()`方法。与`get_option()`类似，这个方法通过传递给它的参数类型推断出要设置的选项，然后设置相应的选项值，使新值等于存储在选项对象中的值。

在最后一步，新的选项值被输出到标准输出流。

# 执行基于流的 I/O

当正确使用时，流和基于流的 I/O 的概念在表达力和优雅性方面非常强大。有时，应用程序的大部分源代码可能由基于流的 I/O 操作组成。如果通过网络通信模块通过基于流的操作实现，此类应用程序的源代码可读性和可维护性将会提高。

幸运的是，Boost.Asio 提供了允许我们以基于流的方式实现进程间通信的工具。在本例中，我们将看到如何使用它们。

## 如何做到这一点...

Boost.Asio 库包含 `asio::ip::tcp::iostream` 封装类，它为 TCP 套接字对象提供类似 I/O 流的接口，这使得我们可以用基于流的操作来表示进程间通信操作。

让我们考虑一个利用 Boost.Asio 提供的基于流的 I/O 的 TCP 客户端应用程序。当使用这种方法时，TCP 客户端应用程序变得像以下代码一样简单：

```cpp
#include <boost/asio.hpp>
#include <iostream>

using namespace boost;

int main() 
{
  asio::ip::tcp::iostream stream("localhost", "3333");
  if (!stream) {
    std::cout << "Error occurred! Error code = " 
      << stream.error().value()
      << ". Message = " << stream.error().message()
      << std::endl;

    return -1;
  }

  stream << "Request.";
  stream.flush();

  std::cout << "Response: " << stream.rdbuf();

  return 0;
} 
```

## 它是如何工作的...

示例 TCP 客户端非常简单，仅由一个组件组成：`main()` 入口点函数。`main()` 函数从创建 `asio::ip::tcp::iostream` 类的实例开始，该实例封装了一个 TCP 套接字，并为它提供了一个类似 I/O 流的接口。

`stream` 对象使用接受服务器 DNS 名称和协议端口号的构造函数构建，并自动尝试解析 DNS 名称，然后尝试连接到该服务器。请注意，端口号以字符串形式表示，而不是整数。这是因为传递给此构造函数的两个参数都直接用于创建解析器查询，该查询需要端口号以字符串形式表示（它应该表示为服务名称，如 `http`、`ftp` 等，或者表示为字符串的端口号，如 "`80`"、"`8081`"、"`3333`" 等）。

或者，我们可以使用默认构造函数来构建 `stream` 对象，该构造函数不执行 DNS 名称解析和连接。然后，当对象被构建时，我们可以通过指定 DNS 名称和协议端口号来调用其上的 `connect()` 方法，以执行解析并连接套接字。

接下来，测试流对象的当前状态，以确定连接是否成功。如果流对象处于不良或错误状态，则将适当的消息输出到标准输出流，并退出应用程序。`asio::ip::tcp::iostream` 类的 `error()` 方法返回 `boost::system::error_code` 类的实例，该实例提供了关于流中发生的最后错误的详细信息。

然而，如果流已成功连接到服务器，则在其上执行输出操作，向服务器发送字符串 `Request`。之后，在流对象上调用 `flush()` 方法，以确保所有缓冲数据都推送到服务器。

在最后一步，对流执行输入操作，以读取从服务器接收的所有数据作为响应。接收到的消息输出到标准输出流。之后，`main()` 函数返回，应用程序退出。

## 还有更多...

我们不仅可以使用`asio::ip::tcp::iostream`类以流式方式实现客户端的 I/O，还可以在服务器端执行 I/O 操作。此外，这个类允许我们为操作指定超时时间，这使得基于流的 I/O 比正常的同步 I/O 更有优势。让我们看看这是如何实现的。

### 实现服务器端 I/O

以下代码片段演示了如何使用`asio::ip::tcp::iostream`类实现一个简单的基于流的 I/O 服务器：

```cpp
  // ... 
  asio::io_service io_service;

  asio::ip::tcp::acceptor acceptor(io_service,
    asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 3333));

  asio::ip::tcp::iostream stream;

acceptor.accept(*stream.rdbuf());
std::cout << "Request: " << stream.rdbuf();
stream << "Response.";
// ...
```

这个代码片段演示了一个简单服务器应用程序的源代码片段。它创建了接受者和`asio::ip::tcp::iostream`类的实例。然后，有趣的事情发生了。

在`acceptor`对象上调用`accept()`方法。该方法传递一个对象作为参数，该参数是通过在`stream`对象上调用`rdbuf()`方法返回的指针。`stream`对象的`rdbuf()`方法返回指向流缓冲区对象的指针。这个流缓冲区对象是一个从`asio::ip::tcp::socket`类继承的类的实例，这意味着`asio::ip::tcp::iostream`类对象使用的流缓冲区扮演两个角色：一个是流缓冲区，另一个是套接字。因此，这个“双重”流缓冲区/套接字对象可以用作正常的活动套接字来连接和与客户端应用程序通信。

当连接请求被接受并且建立了连接后，与客户端的进一步通信将以流式风格进行，就像在客户端应用程序中执行的那样，正如在之前的菜谱中所示。

### 设置超时间隔

因为`asio::ip::tcp::stream`类提供的 I/O 操作会阻塞执行线程，并且它们可能需要运行相当长的时间，所以该类提供了一种设置超时时间的方法。当超时时间耗尽时，如果当前有操作正在阻塞线程，则会中断该操作。

超时间隔可以通过`asio::ip::tcp::stream`类的`expires_from_now()`方法设置。此方法接受超时间隔的持续时间作为输入参数并启动内部计时器。如果在计时器到期时 I/O 操作仍在进行中，则该操作被视为超时，因此被强制中断。
