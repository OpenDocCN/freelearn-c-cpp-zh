

# 使用协程进行异步编程

在前面的章节中，我们看到了在 C++中编写异步代码的不同方法。我们使用了线程，这是执行的基本单元，以及一些高级异步代码机制，如 futures、promises 和**std::async**。我们将在下一章中查看 Boost.Asio 库。所有这些方法通常使用多个系统线程，由内核创建和管理。

例如，我们程序的主线程可能需要访问数据库。这种访问可能很慢，所以我们将在不同的线程中读取数据，以便主线程可以继续执行其他任务。另一个例子是生产者-消费者模型，其中一个或多个线程生成要处理的数据项，一个或多个线程以完全异步的方式处理这些项。

上述两个示例都使用了线程，也称为系统（内核）线程，并需要不同的执行单元，每个线程一个。

在本章中，我们将研究一种不同的异步代码编写方式——协程。协程是一个来自 20 世纪 50 年代末的老概念，直到 C++20 才被添加到 C++中。它们不需要单独的线程（当然，我们可以在不同的线程中运行协程）。协程是一种机制，它使我们能够在单线程中执行多个任务。

在本章中，我们将涵盖以下主要主题：

+   协程是什么？它们是如何被 C++实现和支持的？

+   实现基本协程以了解 C++协程的要求

+   生成器协程和新的 C++23 **std::generator**

+   用于解析整数的字符串解析器

+   协程中的异常

本章介绍的是不使用任何第三方库实现的 C++协程。这种方式编写协程相当底层，我们需要编写代码来支持编译器。

# 技术要求

对于本章，你需要一个 C++20 编译器。对于生成器示例，你需要一个 C++23 编译器。我们已经测试了这些示例与 GCC **14.1** 兼容。代码是平台无关的，因此尽管本书关注 Linux，但所有示例都应在 macOS 和 Windows 上运行。请注意，Visual Studio **17.11** 还不支持 C++23 **std::generator**。

本章的代码可以在本书的 GitHub 仓库中找到：[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)。

# 协程

在我们开始用 C++实现协程之前，我们将从概念上介绍协程，并看看它们在我们的程序中如何有用。

让我们从定义开始。**协程**是一个可以暂停自己的函数。协程在等待输入值（在它们暂停时，它们不执行）或产生一个值，如计算的输出后暂停自己。一旦输入值可用或调用者请求另一个值，协程将恢复执行。我们很快将回到 C++中的协程，但让我们通过一个现实生活中的例子来看看协程是如何工作的。

想象一下有人在当助手。他们开始一天的工作是阅读电子邮件。

其中一封电子邮件是要求一份报告。在阅读电子邮件后，他们开始撰写所需的文档。一旦他们写完了引言段落，他们注意到他们需要从同事那里获取一份报告，以获取上一季度的会计结果。他们停止撰写报告，给同事写了一封电子邮件，请求所需的信息，然后阅读下一封电子邮件，这是一封要求预订下午重要会议的会议室的请求。他们打开公司开发的一个专门用于自动预订会议室以优化其使用的应用程序来预订会议室。

过了一段时间，他们从同事那里收到了所需的会计数据，然后继续撰写报告。

助手总是忙于处理他们的任务。撰写报告是协程的一个好例子：他们开始撰写报告，然后在等待所需信息时暂停写作，一旦信息到达，他们继续写作。当然，助手不想浪费时间，在等待时，他们会继续做其他任务。如果他们等待请求并发出适当的响应，他们的同事可以被视为另一个协程。

现在，让我们回到软件。假设我们需要编写一个函数，在处理一些输入信息后，将数据存储到数据库中。

如果数据一次性到达，我们只需实现一个函数。该函数将读取输入，对数据进行必要的处理，最后将结果写入数据库。但如果要处理的数据以块的形式到达，并且处理每个块都需要前一个块处理的结果（为了这个例子，我们可以假设第一个块的处理只需要一些默认值）呢？

解决我们问题的可能方法是在每个数据块到达时让函数等待，处理它，将结果存储在数据库中，然后等待下一个，依此类推。但如果我们这样做，我们可能会在等待每个数据块到达时浪费很多时间。

在阅读了前面的章节后，你可能正在考虑不同的潜在解决方案：我们可以创建一个线程来读取数据，将块复制到队列中，然后第二个线程（可能是主线程）将处理数据。这是一个可接受的解决方案，但使用多个线程可能有些过度。

另一种解决方案可能是实现一个只处理一个数据块的函数。调用者将等待输入传递给函数，并保留处理每个数据块所需的上一块处理的结果。在这个解决方案中，我们必须在另一个函数中保留数据处理函数所需的状态。对于简单的示例可能是可接受的，但一旦处理变得更为复杂（例如，需要保留不同中间结果的多步处理），代码可能难以理解和维护。

我们可以用协程解决这个问题。让我们看看处理数据块并保留中间结果的协程的一些可能的伪代码：

```cpp
processing_result process_data(data_block data) {
    while (do_processing == true) {
        result_type result{ 0 };
        result = process_data_block(previous_result);
        update_database();
        yield result;
    }
}
```

前面的协程从调用者那里接收一个数据块，执行所有处理，更新数据库，并保留处理下一个数据块所需的结果。在将结果传回调用者（关于传回的更多内容稍后讨论）之后，它将自己暂停。当调用者再次调用协程请求处理新的数据块时，其执行将恢复。

这样的协程简化了状态管理，因为它可以在调用之间保持状态。

在对协程进行概念介绍之后，我们将开始使用 C++20 实现它们。

# C++协程

正如我们所见，协程只是函数，但它们并不像我们习惯的函数。它们具有我们将在本章中学习的特殊属性。在本节中，我们将专注于 C++中的协程。

函数在调用时开始执行，并通常通过返回语句或当函数的末尾到达时正常终止。

函数从开始到结束运行。它可能调用另一个函数（或者如果是递归的，甚至可以调用自己），它可能抛出异常或具有不同的返回点。但它总是从开始到结束运行。

协程是不同的。协程是一个可以暂停自己的函数。协程的流程可能如下伪代码所示：

```cpp
 void coroutine() {
    do_something();
    co_yield;
    do_something_else();
    co_yield;
    do_more_work();
    co_return;
}
```

我们很快就会看到那些带有**co_**前缀的术语的含义。

对于协程，我们需要一个机制来保持执行状态，以便能够暂停/恢复协程。这是由编译器为我们完成的，但我们必须编写一些**辅助**代码，以便让编译器帮助我们。

C++中的协程是无堆栈的。这意味着我们需要存储以能够暂停/恢复协程的状态存储在堆中，通过调用**new**/**delete**来分配/释放动态内存。这些调用是由编译器创建的。

## 新关键字

因为协程本质上是一个函数（具有一些特殊属性，但仍然是一个函数），编译器需要某种方式来确定给定的函数是否是协程。C++20 引入了三个新的关键字：**co_yield**、**co_await**和**co_return**。如果一个函数使用了这三个关键字中的至少一个，那么编译器就知道它是一个协程。

下表总结了新关键字的函数：

| **关键字** | **输入/输出** | **协程状态** |
| --- | --- | --- |
| **co_yield** | 输出 | 暂停 |
| **co_await** | 输入 | 暂停 |
| **co_return** | 输出 | 终止 |

表 8.1：新的协程关键字

在前面的表中，我们看到在 **co_yield** 和 **co_await** 之后，协程会暂停，而在 **co_return** 之后，它会终止（**co_return** 在 C++函数中相当于 **return** 语句）。协程不能有 **return** 语句；它必须始终使用 **co_return**。如果协程不返回任何值，并且使用了其他两个协程关键字之一，则可以省略 **co_return** 语句。

## 协程限制

我们已经说过，协程是使用新协程关键字的函数。但协程有以下限制：

+   使用 **varargs** 的具有可变数量参数的函数不能是协程（一个变长函数模板可以是协程）

+   类构造函数或析构函数不能是协程

+   **constexpr** 和 **consteval** 函数不能是协程

+   返回 **auto** 的函数不能是协程，但带有尾随返回类型的 **auto** 可以是

+   **main()** 函数不能是协程

+   Lambda 可以是协程

在学习了协程的限制（基本上是哪些 C++函数不能是协程）之后，我们将在下一节开始实现协程。

# 实现基本协程

在上一节中，我们学习了协程的基本知识，包括它们是什么以及一些用例。

在本节中，我们将实现三个简单的协程来展示实现和使用它们的基本方法：

+   只返回的最简单协程

+   协程向调用者发送值

+   从调用者获取值的协程

## 最简单的协程

我们知道协程是一个可以暂停自己的函数，并且可以被调用者恢复。我们还知道，如果函数至少使用了一个 **co_yield**、**co_await** 或 **co_return** 表达式，编译器会将该函数识别为协程。

编译器将转换协程源代码，并创建一些数据结构和函数，使协程能够正常工作，并能够暂停和恢复。这是为了保持协程状态并能够与协程进行通信。

编译器将处理所有这些细节，但请注意，C++对协程的支持相当底层。有一些库可以帮助我们在 C++中更轻松地处理协程。其中一些是 **Lewis Baker 的 cppcoro** 和 **Boost.Cobalt**。**Boost.Asio** 库也支持协程。这些库是下一章的主题。

让我们从零开始。这里的“从零开始”是指绝对的零起点。我们将编写一些代码，并通过编译器错误和 C++参考来编写一个基本但功能齐全的协程。

以下代码是协程的最简单实现：

```cpp
void coro_func() {
    co_return;
}
int main() {
    coro_func();
}
```

简单，不是吗？我们的第一个协程将只返回空值。它不会做任何其他事情。遗憾的是，前面的代码对于功能协程来说太简单了，无法编译。当使用 GCC **14.1** 编译时，我们得到以下错误：

```cpp
error: coroutines require a traits template; cannot find 'std::coroutine_traits'
```

我们还得到了以下提示：

```cpp
note: perhaps '#include <coroutine>' is missing
```

编译器给我们一个提示：我们可能遗漏了包含一个必需的文件。让我们包含**<coroutine>**头文件。我们将在一会儿处理关于 traits 模板的错误：

```cpp
#include <coroutine>
void coro_func() {
    co_return;
}
int main() {
    coro_func();
}
```

在编译前面的代码时，我们遇到了以下错误：

```cpp
 error: unable to find the promise type for this coroutine
```

我们协程的第一个版本给我们带来了一个编译错误，说找不到类型**std::coroutine_traits**模板。现在我们得到了一个与所谓的*promise 类型*有关的错误。

查看 C++参考，我们看到**std::coroutine_traits**模板决定了协程的返回类型和参数类型。参考还指出，协程的返回类型必须定义一个名为**promise_type**的类型。遵循参考建议，我们可以编写我们协程的新版本：

```cpp
#include <coroutine>
struct return_type {
    struct promise_type {
    };
};
template<>
struct std::coroutine_traits<return_type> {
    using promise_type = return_type::promise_type;
};
return_type coro_func() {
    co_return;
}
int main() {
    coro_func();
}
```

请注意，协程的返回类型可以有任何名称（我们在这里将其称为**return_type**，因为这在这个简单示例中很方便）。

再次编译前面的代码时，我们遇到了一些错误（为了清晰起见，错误已被编辑）。所有错误都与**promise_type**结构中缺少的函数有关：

```cpp
error: no member named 'return_void' in 'std::__n4861::coroutine_traits<return_type>::promise_type'
error: no member named 'initial_suspend' in 'std::__n4861::coroutine_traits<return_type>::promise_type'
error: no member named 'unhandled_exception' in 'std::__n4861::coroutine_traits<return_type>::promise_type'
error: no member named 'final_suspend' in 'std::__n4861::coroutine_traits<return_type>::promise_type'
error: no member named 'get_return_object' in 'std::__n4861::coroutine_traits<return_type>::promise_type'
```

我们到目前为止看到的所有编译错误都与我们的代码中缺少的功能有关。在 C++中编写协程需要遵循一些规则，并帮助编译器生成有效的代码。

以下是最简单的协程的最终版本：

```cpp
#include <coroutine>
struct return_type {
    struct promise_type {
        return_type get_return_object() noexcept {
            return return_type{ *this };
        }
        void return_void() noexcept {}
        std::suspend_always initial_suspend() noexcept {
            return {};
        }
        std::suspend_always final_suspend() noexcept {
            return {};
        }
        void unhandled_exception() noexcept {}
    };
    explicit return_type(promise_type&) {
    }
    ~return_type() noexcept {
    }
};
return_type coro_func() {
    co_return;
}
int main() {
    coro_func();
}
```

你可能已经注意到我们已经移除了**std::coroutine_traits**模板。实现返回和 promise 类型就足够了。

前面的代码编译没有任何错误，你可以运行它。它确实...什么也不做！但这是我们第一个协程，我们已经了解到我们需要提供一些编译器所需的代码来创建协程。

### promise 类型

**promise 类型**是编译器所要求的。我们需要始终定义此类型（它可以是类或结构体），它必须命名为**promise_type**，并且必须实现 C++参考中指定的某些函数。我们已经看到，如果我们不这样做，编译器会抱怨并给出错误。

promise 类型必须在协程返回的类型内部定义，否则代码将无法编译。返回的类型（有时也称为**wrapper 类型**，因为它封装了**promise_type**）可以任意命名。

## 一个产生结果的协程

一个什么也不做的协程对于说明一些基本概念很有用。我们现在将实现另一个可以将数据发送回调用者的协程。

在这个第二个例子中，我们将实现一个产生消息的协程。它将是协程的“hello world”。协程将说你好，调用函数将打印从协程接收到的消息。

为了实现该功能，我们需要从协程到调用者建立一个通信通道。这个通道是允许协程向调用者传递值并从它那里接收信息的机制。这个通道是通过协程的 **承诺类型** 和 **句柄** 建立的，它们管理协程的状态。

通信通道按以下方式工作：

+   **协程帧** : 当协程被调用时，它创建一个 **协程帧** ，其中包含暂停和恢复其执行所需的所有状态信息。这包括局部变量、承诺类型以及任何内部状态。

+   **承诺类型** : 每个协程都有一个相关的 **承诺类型** ，它负责管理协程与调用函数之间的交互。承诺是存储协程返回值的地方，它提供了控制协程行为的函数。我们将在本章的示例中看到这些函数。承诺是调用者与协程交互的接口。

+   **协程句柄** : 协程句柄是一种类型，它提供了对协程帧（协程的内部状态）的访问权限，并允许调用者恢复或销毁协程。句柄是调用者可以在协程被挂起后（例如，在 **co_await** 或 **co_yield** 之后）恢复协程的东西。句柄还可以用来检查协程是否完成或清理其资源。

+   **挂起和恢复机制** : 当协程 yield 一个值（ **co_yield** ）或等待异步操作（ **co_await** ）时，它挂起其执行，将其状态保存在协程帧中。然后调用者可以在稍后恢复协程，通过协程句柄检索 yielded 或 awaited 的值并继续执行。

我们将在以下示例中看到，这个通信通道需要我们在自己的这一侧编写相当数量的代码，以帮助编译器生成协程功能所需的全部代码。

以下代码是调用函数和协程的新版本：

```cpp
return_type coro_func() {
    co_yield "Hello from the coroutine\n"s;
    co_return;
}
int main() {
    auto rt = coro_func();
    std::cout << rt.get() << std::endl;
    return 0;
}
```

变更如下：

+   **[1]** : 协程 *yield* 并向调用者发送一些数据（在这种情况下，一个 **std::string** 对象）

+   **[2]** : 调用者读取那些数据并将其打印出来

所需的通信机制在承诺类型和返回类型（这是一个承诺类型包装器）中实现。

当编译器读取 **co_yield** 表达式时，它将生成对在承诺类型中定义的 **yield_value** 函数的调用。

以下代码是我们版本的该函数的实现，该函数生成（或 yield）一个 **std::string** 对象：

```cpp
std::suspend_always yield_value(std::string msg) noexcept {
    output_data = std::move(msg);
    return {};
}
```

函数获取一个 **std::string** 对象并将其移动到承诺类型的 **output_data** 成员变量中。但这只是将数据保留在承诺类型内部。我们需要一种机制来将那个字符串从协程中取出。

### 句柄类型

一旦我们需要一个协程的通信通道，我们需要一种方式来引用一个挂起或正在执行的协程。C++标准库在所谓的**协程句柄**中实现了这样的机制。它的类型是**std::coroutine_handle**，它是返回类型的成员变量。这个结构也负责句柄的完整生命周期，包括创建和销毁它。

以下代码片段是我们添加到返回类型中以管理协程句柄的功能：

```cpp
std::coroutine_handle<promise_type> handle{};
explicit return_type(promise_type& promise) : handle{ std::coroutine_handle<promise_type>::from_promise(promise)} {
}
~return_type() noexcept {
    if (handle) {
        handle.destroy();
    }
}
```

前面的代码声明了一个类型为**std::coroutine_handle<promise_type>**的协程句柄，并在返回类型构造函数中创建句柄。句柄在返回类型析构函数中被销毁。

现在，回到我们的产生值的协程。唯一缺少的部分是调用函数的**get()**函数，以便能够访问协程生成的字符串：

```cpp
std::string get() {
    if (!handle.done()) {
        handle.resume();
    }
    return std::move(handle.promise().output_data);
}
```

**get()**函数在协程未终止的情况下恢复协程，然后返回字符串对象。

以下是我们第二个协程的完整代码：

```cpp
#include <coroutine>
#include <iostream>
#include <string>
using namespace std::string_literals;
struct return_type {
    struct promise_type {
        std::string output_data { };
        return_type get_return_object() noexcept {
            std::cout << "get_return_object\n";
            return return_type{ *this };
        }
        void return_void() noexcept {
            std::cout << "return_void\n";
        }
        std::suspend_always yield_value(
                         std::string msg) noexcept {
            std::cout << "yield_value\n";
            output_data = std::move(msg);
            return {};
        }
        std::suspend_always initial_suspend() noexcept {
            std::cout << "initial_suspend\n";
            return {};
        }
        std::suspend_always final_suspend() noexcept {
            std::cout << "final_suspend\n";
            return {};
        }
        void unhandled_exception() noexcept {
            std::cout << "unhandled_exception\n";
        }
    };
    std::coroutine_handle<promise_type> handle{};
    explicit return_type(promise_type& promise)
       : handle{ std::coroutine_handle<
                 promise_type>::from_promise(promise)}{
        std::cout << "return_type()\n";
    }
    ~return_type() noexcept {
        if (handle) {
            handle.destroy();
        }
        std::cout << "~return_type()\n";
    }
    std::string get() {
        std::cout << "get()\n";
        if (!handle.done()) {
            handle.resume();
        }
        return std::move(handle.promise().output_data);
    }
};
return_type coro_func() {
    co_yield "Hello from the coroutine\n"s;
    co_return;
}
int main() {
    auto rt = coro_func();
    std::cout << rt.get() << std::endl;
    return 0;
}
```

运行前面的代码会打印以下消息：

```cpp
get_return_object
return_type()
initial_suspend
get()
yield_value
Hello from the coroutine
~return_type()
```

这个输出显示了协程执行期间发生的情况：

1.  **return_type**对象在调用**get_return_object**之后创建

1.  协程最初是挂起的

1.  调用者想要从协程中获取消息，因此调用**get()**

1.  **yield_value**被调用，协程被恢复，并且消息被复制到承诺的成员变量中

1.  最后，调用函数打印消息，协程返回

注意，承诺（以及承诺类型）与在*第六章*中解释的 C++标准库**std::promise**类型无关。

## 等待中的协程

在前面的例子中，我们看到了如何实现一个可以通过发送**std::string**对象来回调者通信的协程。现在，我们将实现一个可以等待调用者发送的输入数据的协程。在我们的例子中，协程将等待直到它接收到一个**std::string**对象，然后打印它。当我们说协程“等待”时，我们的意思是它是挂起的（即，没有执行）直到数据接收。

让我们从协程和调用函数的更改开始：

```cpp
return_type coro_func() {
    std::cout << co_await std::string{ };
    co_return;
}
int main() {
    auto rt = coro_func();
    rt.put("Hello from main\n"s);
    return 0;
}
```

在前面的代码中，调用函数调用**put()**函数（返回类型结构中的方法）和协程调用**co_await**等待从调用者那里来的**std::string**对象。

返回类型的更改很简单，即只是添加**put()**函数：

```cpp
void put(std::string msg) {
    handle.promise().input_data = std::move(msg);
    if (!handle.done()) {
        handle.resume();
    }
}
```

我们需要将**input_data**变量添加到承诺结构中。但是，仅仅通过对我们第一个示例所做的更改（我们将它作为本章其余示例的起点，因为它是最少的代码来实现协程）以及上一个示例中的协程句柄，代码无法编译。编译器给我们以下错误：

```cpp
error: no member named 'await_ready' in 'std::string' {aka 'std::__cxx11::basic_string<char>'}
```

回到 C++参考，我们看到当协程调用**co_await**时，编译器将生成代码来调用承诺对象中的函数**await_transform**，该函数的参数类型与协程等待的数据类型相同。正如其名所示，**await_transform**是一个将任何对象（在我们的例子中，**std::string**）转换为可等待对象的函数。**std::string**是不可等待的，因此之前的编译器错误。

**await_transform**必须返回一个**awaiter**对象。这只是一个简单的结构，实现了使编译器能够使用 awaiter 所需的基本接口。

以下代码展示了我们实现的**await_transform**函数和**awaiter**结构：

```cpp
auto await_transform(std::string) noexcept {
    struct awaiter {
        promise_type& promise;
        bool await_ready() const noexcept {
            return true;
        }
        std::string await_resume() const noexcept {
            return std::move(promise.input_data);
        }
        void await_suspend(std::coroutine_handle<
                           promise_type>) const noexcept {
        }
   };
   return awaiter(*this);
}
```

编译器需要**promise_type**函数**await_transform**。我们不能为这个函数使用不同的标识符。参数类型必须与协程等待的对象类型相同。**awaiter**结构可以命名为任何名称。我们在这里使用**awaiter**是因为它具有描述性。**awaiter**结构必须实现三个函数：

+   **await_ready**：这个函数用于检查协程是否被挂起。如果是这种情况，它返回**false**。在我们的例子中，它总是返回**true**，表示协程没有被挂起。

+   **await_resume**：这个函数恢复协程并生成**co_await**表达式的结果。

+   **await_suspend**：在我们的简单 awaiter 中，这个函数返回**void**，意味着控制权传递给调用者，协程被挂起。**await_suspend**也可以返回一个布尔值。在这种情况下返回**true**就像返回**void**一样。返回**false**意味着协程被恢复。

这是等待协程完整示例的代码：

```cpp
#include <coroutine>
#include <iostream>
#include <string>
using namespace std::string_literals;
struct return_type {
    struct promise_type {
        std::string input_data { };
        return_type get_return_object() noexcept {
            return return_type{ *this };
        }
        void return_void() noexcept {
        }
        std::suspend_always initial_suspend() noexcept {
            return {};
        }
        std::suspend_always final_suspend() noexcept {
            return {};
        }
        void unhandled_exception() noexcept {
        }
        auto await_transform(std::string) noexcept {
            struct awaiter {
                promise_type& promise;
                bool await_ready() const noexcept {
                    return true;
                }
                std::string await_resume() const noexcept {
                    return std::move(promise.input_data);
                }
                void await_suspend(std::coroutine_handle<
                                  promise_type>) const noexcept {
                }
            };
            return awaiter(*this);
        }
    };
    std::coroutine_handle<promise_type> handle{};
    explicit return_type(promise_type& promise)
      : handle{ std::coroutine_handle<
                         promise_type>::from_promise(promise)} {
    }
    ~return_type() noexcept {
        if (handle) {
            handle.destroy();
        }
    }
    void put(std::string msg) {
        handle.promise().input_data = std::move(msg);
        if (!handle.done()) {
            handle.resume();
        }
    }
};
return_type coro_func() {
    std::cout << co_await std::string{ };
    co_return;
}
int main() {
    auto rt = coro_func();
    rt.put("Hello from main\n"s);
    return 0;
}
```

在本节中，我们看到了协程的三个基本示例。我们实现了最简单的协程，然后是具有通信通道的协程，这些协程既为调用者生成数据（**co_yield**），又从调用者那里等待数据（**co_await**）。

在下一节中，我们将实现一种称为生成器的协程类型，并生成数字序列。

# 协程生成器

**生成器**是一个协程，通过反复从它被挂起的位置恢复自身来生成一系列元素。

生成器可以被视为一个**无限**序列，因为它可以生成任意数量的元素。调用函数可以从生成器获取它所需的所有新元素。

当我们说无限时，我们指的是理论上。生成器协程将产生元素，没有明确的最后一个元素（可以实现具有有限范围的生成器），但在实践中，我们必须处理诸如数值序列中的溢出等问题。

让我们从零开始实现一个生成器，应用我们在本章前几节学到的知识。

## 斐波那契序列生成器

想象我们正在实现一个应用程序，并且需要使用斐波那契序列。您可能已经知道，**斐波那契序列**是一个序列，其中每个数字都是前两个数字的和。第一个元素是 0，第二个元素是 1，然后我们应用定义并逐个生成元素。

![<math display="block"><mrow><mrow><mi>F</mi><mi>i</mi><mi>b</mi><mi>o</mi><mi>n</mi><mi>a</mi><mi>c</mi><mi>c</mi><mi>i</mi><mi>o</mi><mi>n</mi><mi>e</mi><mi>q</mi><mi>u</mi><mi>e</mi><mi>n</mi><mi>c</mi><mi>e</mi><mo>:</mo><mi>F</mi><mfenced open="(" close=")"><mi>n</mi></mfenced><mo>=</mo><mi>F</mi><mfenced open="(" close=")"><mrow><mi>n</mi><mo>−</mo><mn>2</mn></mrow></mfenced><mo>+</mo><mi>F</mi><mfenced open="(" close=")"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfenced><mo>;</mo><mi>F</mi><mfenced open="(" close=")"><mn>0</mn></mfenced><mo>=</mo><mn>0</mn><mo>,</mo><mi>F</mi><mfenced open="(" close=")"><mn>1</mn></mfenced><mo>=</mo><mn>1</mn></mrow></mrow></math>](img/12.png)

我们总是可以用一个 **for** 循环生成这些数字。但如果我们需要在程序的不同点生成它们，我们需要实现一种存储序列状态的方法。我们需要在我们的程序中某个地方保留我们生成的最后一个元素是什么。是第五个还是可能是第十个？

协程是解决这个问题的非常好的解决方案；它会自己保持所需的状态，并且它会在我们请求序列中的下一个数字时暂停。

下面是使用生成器协程的代码：

```cpp
int main() {
    sequence_generator<int64_t> fib = fibonacci();
    std::cout << "Generate ten Fibonacci numbers\n"s;
    for (int i = 0; i < 10; ++i) {
        fib.next();
        std::cout << fib.value() << " ";
    }
    std::cout << std::endl;
    std::cout << "Generate ten more\n"s;
    for (int i = 0; i < 10; ++i) {
        fib.next();
        std::cout << fib.value() << " ";
    }
    std::cout << std::endl;
    std::cout << "Let's do five more\n"s;
    for (int i = 0; i < 5; ++i) {
        fib.next();
        std::cout << fib.value() << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

如您在前面的代码中看到的，我们生成所需的数字时无需担心最后一个元素是什么。序列是由协程生成的。

注意，尽管在理论上序列是无限的，但我们的程序必须意识到非常大的斐波那契数可能存在溢出的潜在风险。

要实现生成器协程，我们遵循本章之前解释的原则。

首先，我们实现协程函数：

```cpp
sequence_generator<int64_t> fibonacci() {
    int64_t a{ 0 };
    int64_t b{ 1 };
    int64_t c{ 0 };
    while (true) {
        co_yield a;
        c = a + b;
        a = b;
        b = c;
    }
}
```

协程通过应用公式生成斐波那契序列的下一个元素。元素在无限循环中生成，但协程在 **co_yield** 后会暂停自己。

返回类型是 **sequence_generator** 结构体（我们使用模板以便能够使用 32 位或 64 位整数）。它包含一个承诺类型，与我们在前一个部分中看到的产生式协程中的承诺类型非常相似。

在 **sequence_generator** 结构体中，我们添加了两个在实现序列生成器时有用的函数。

```cpp
void next() {
    if (!handle.done()) {
        handle.resume();
    }
}
```

**next()** 函数用于恢复协程以生成序列中要生成的下一个斐波那契数。

```cpp
int64_t value() {
    return handle.promise().output_data;
}
```

**value()** 函数返回最后一个生成的斐波那契数。

这样，我们就解耦了元素生成和其检索 Q 值。

请在本书的配套 GitHub 仓库中找到此示例的完整代码。

### C++23 std::generator

我们已经看到，即使在 C++ 中实现最基础的协程也需要一定量的代码。这可能在 C++26 中改变，因为 C++ 标准库对协程的支持将更多，这将使我们能够更容易地编写协程。

C++23 引入了 **std::generator** 模板类。通过使用它，我们可以编写基于协程的生成器，而无需编写任何所需的代码，例如承诺类型、返回类型及其所有函数。要运行此示例，您需要一个 C++23 编译器。我们使用了 GCC 14.1。**std::generator** 在 Clang 中不可用。

让我们看看使用新的 C++23 标准库特性的斐波那契数列生成器：

```cpp
#include <generator>
#include <iostream>
std::generator<int> fibonacci_generator() {
    int a{ };
    int b{ 1 };
    while (true) {
        co_yield a;
        int c = a + b;
        a = b;
        b = c;
    }
}
auto fib = fibonacci_generator();
int main() {
    int i = 0;
    for (auto f = fib.begin(); f != fib.end(); ++f) {
        if (i == 10) {
            break;
        }
        std::cout << *f << " ";
        ++i;
    }
    std::cout << std::endl;
}
```

第一步是包含 **<generator>** 头文件。然后，我们只需编写协程，因为所有其他所需的代码都已经为我们编写好了。在前面的代码中，我们使用迭代器（由 C++ 标准库提供）访问生成的元素。这允许我们使用范围-for 循环、算法和范围。

还可以编写一个斐波那契生成器的版本，生成一定数量的元素而不是无限序列：

```cpp
std::generator<int> fibonacci_generator(int limit) {
    int a{ };
    int b{ 1 };
    while (limit--) {
        co_yield a;
        int c = a + b;
        a = b;
        b = c;
    }
}
```

代码更改非常简单：只需传递我们希望生成器生成的元素数量，并在 **while** 循环中将其用作终止条件。

在本节中，我们实现了最常见的协程类型之一——生成器。我们从头开始实现了生成器，也使用了 C++23 的 **std::generator** 类模板。

在下一节中，我们将实现一个简单的字符串解析器协程。

# 简单的协程字符串解析器

在本节中，我们将实现我们的最后一个示例：一个简单的字符串解析器。协程将等待输入，一个 **std::string** 对象，并在解析输入字符串后产生输出，即一个数字。为了简化示例，我们将假设数字的字符串表示没有错误，并且数字的结尾由哈希字符，**#** 表示。我们还将假设数字类型是 **int64_t**，并且字符串不会包含该整数类型范围之外的任何值。

## 解析算法

让我们看看如何将表示整数的字符串转换为数字。例如，字符串 **"-12321#"** 表示数字 -12321。要将字符串转换为数字，我们可以编写一个像这样的函数：

```cpp
int64_t parse_string(const std::string& str) {
    int64_t num{ 0 };
    int64_t sign { 1 };
    std::size_t c = 0;
    while (c < str.size()) {
        if (str[c] == '-') {
            sign = -1;
        }
        else if (std::isdigit(str[c])) {
            num = num * 10 + (str[c] - '0');
        }
        else if (str[c] == '#') {
            break;
        }
        ++c;
    }
    return num * sign;
}
```

由于假设字符串是良好形成的，代码相当简单。如果我们读取负号，**-**，则将符号更改为 -1（默认情况下，我们假设正数，如果有 **+** 符号，则简单地忽略它）。然后，逐个读取数字，并按以下方式计算数字值。

**num** 的初始值是 **0**。我们读取第一个数字，并将其数值加到当前 **num** 值乘以 10 上。这就是我们读取数字的方式：最左边的数字将乘以 10，次数等于其右侧数字的数量。

当我们使用字符来表示数字时，它们根据 ASCII 表示法有一定的值（我们假设没有使用宽字符或其他任何字符类型）。字符*0*到*9*具有连续的 ASCII 码，因此我们可以通过简单地减去*0*来轻松地将它们转换为数字。

即使对于前面的代码，最后的字符检查可能不是必要的，但我们还是在这里包含了它。当解析器例程找到**#**字符时，它将终止解析循环并返回最终的数值。

我们可以使用这个函数解析任何字符串并获取数值，但我们需要完整的字符串来将其转换为数字。

让我们考虑这个场景：字符串正在从网络连接接收，我们需要解析它并将其转换为数字。我们可能将字符保存到一个临时字符串中，然后调用前面的函数。

但还有一个问题：如果字符以每几秒一次的速度缓慢到达，那会怎样？因为这就是它们传输的方式？我们希望保持 CPU 忙碌，并在可能的情况下，在等待每个字符到达时执行其他任务（或多个任务）。

解决这个问题有不同的方法。我们可以创建一个线程并发处理字符串，但这对于这样一个简单的任务来说可能会在计算机时间上代价高昂。我们也可以使用**std::async**。

## 解析协程

在本章中，我们正在使用协程，因此我们将使用 C++协程实现字符串解析。我们不需要额外的线程，并且由于协程的异步性质，在字符到达时执行任何其他处理将非常容易。

我们需要的解析协程的样板代码与我们在前面的示例中已经看到的代码几乎相同。解析器本身则相当不同。请看以下代码：

```cpp
async_parse<int64_t, char> parse_string() {
    while (true) {
        char c = co_await char{ };
        int64_t number { };
        int64_t sign { 1 };
        if (c != '-' && c != '+' && !std::isdigit(c)) {
            continue;
        }
        if (c == '-') {
            sign = -1;
        }
        else if (std::isdigit(c)) {
            number = number * 10 + c - '0';
        }
        while (true) {
            c = co_await char{};
            if (std::isdigit(c)) {
                number = number * 10 + c - '0';
            }
            else {
                break;
            }
        }
        co_yield number * sign;
    }
}
```

我认为你现在可以轻松地识别返回类型（**async_parse<int64_t, char>**），并且知道解析协程会在等待输入字符时挂起。一旦解析完成，协程会在返回数字后挂起自己。

但你也会看到，前面的代码并不像我们第一次尝试将字符串解析为数字那样简单。

首先，解析协程逐个解析字符。它不获取完整的字符串来解析，因此有无限循环**while (true)**。我们不知道完整字符串中有多少个字符，因此我们需要继续接收和解析它们。

外层循环意味着协程将解析数字，一个接一个，随着字符的到达——永远。但请记住，它会挂起自己以等待字符，所以我们不会浪费 CPU 时间。

现在，一个字符到达。首先检查这个字符是否是我们数字的有效字符。如果字符既不是负号**-**，也不是正号**+**，也不是一个数字，那么解析器将等待下一个字符。

如果下一个字符是有效的，那么以下适用：

+   如果是减号，我们将符号值更改为-1

+   如果是加号，我们忽略它

+   如果是数字，我们将其解析到数字中，使用与我们在解析器的第一个版本中看到的方法更新当前数字值。

在第一个有效字符之后，我们进入一个新的循环来接收其余的字符，无论是数字还是分隔符字符（**#**）。注意，当我们说有效字符时，我们是指对数值转换好的。我们仍然假设输入字符形成一个有效的数字，并且正确终止。

一旦数字被转换，它就会被协程产生，外层循环再次执行。这里需要一个终止字符，因为输入字符流在理论上是无尽的，它可以包含许多数字。

协程其余部分的代码可以在 GitHub 仓库中找到。它遵循任何其他协程相同的约定。首先，我们定义返回类型：

```cpp
template <typename Out, typename In>
struct async_parse {
// …
};
```

我们使用模板以提高灵活性，因为它允许我们参数化输入和输出数据类型。在这种情况下，这些类型分别是**int64_t**和**char**。

输入和输出数据项如下：

```cpp
std::optional<In> input_data { };
Out output_data { };
```

对于输入，我们使用**std::optional<In>**，因为我们需要一种方式来知道我们是否收到了一个字符。我们使用**put()**函数将字符发送到解析器：

```cpp
 void put(char c) {
    handle.promise().input_data = c;
    if (!handle.done()) {
        handle.resume();
    }
}
```

这个函数只是将值赋给**std::optional** **input_data**变量。为了管理等待字符，我们实现以下 awaiter 类型：

```cpp
auto await_transform(char) noexcept {
    struct awaiter {
        promise_type& promise;
        [[nodiscard]] bool await_ready() const noexcept {
            return promise.input_data.has_value();
        }
        [[nodiscard]] char await_resume() const noexcept {
            assert (promise.input_data.has_value());
            return *std::exchange(
                            promise.input_data,
                            std::nullopt);
        }
        void await_suspend(std::coroutine_handle<
                           promise_type>) const noexcept {
        }
    };
    return awaiter(*this);
}
```

**awaiter**结构体实现了两个函数来处理输入数据：

+   **await_ready()**：如果可选的**input_data**变量包含有效值，则返回**true**。否则返回**false**。

+   **await_resume()**：返回存储在可选**input_data**变量中的值，并将其清空，赋值为**std::nullopt**。

在本节中，我们看到了如何使用 C++协程实现一个简单的解析器。这是我们最后的示例，展示了使用协程的一个非常基本的流处理函数。在下一节中，我们将看到协程中的异常。

# 协程和异常

在前面的章节中，我们实现了一些基本示例来学习主要的 C++协程概念。我们首先实现了一个非常基本的协程，以了解编译器对我们有什么要求：返回类型（有时称为包装类型，因为它包装了承诺类型）和承诺类型。

即使对于这样一个简单的协程，我们也必须实现我们在编写示例时解释的一些函数。但有一个函数尚未解释：

```cpp
void unhandled_exception() noexcept {}
```

我们当时假设协程不能抛出异常，但事实是它们可以。我们可以在**unhandled_exception()**函数体中添加处理异常的功能。

协程中的异常可能在创建返回类型或承诺类型对象时发生，也可能在协程执行时发生（就像正常函数一样，协程可以抛出异常）。

差别在于，如果在协程执行之前抛出异常，创建协程的代码必须处理该异常，而如果在协程执行时抛出异常，则调用**unhandled_exception()**。

第一种情况只是通常的异常处理，没有调用特殊函数。我们可以在**try-catch**块中放置协程创建，并像我们通常在代码中那样处理可能的异常。

如果另一方面，调用了**unhandled_exception()**（在 promise 类型内部），我们必须在该函数内部实现异常处理功能。

处理此类异常有不同的策略。其中之一如下：

+   重新抛出异常，这样我们就可以在 promise 类型之外（即在我们的代码中）处理它。

+   终止程序（例如，调用**std::terminate**）。

+   留下函数为空。在这种情况下，协程将崩溃，并且它很可能导致程序崩溃。

因为我们实现了非常简单的协程，所以我们留下了函数为空。

在本节的最后，我们介绍了协程的异常处理机制。正确处理异常非常重要。例如，如果你知道协程内部发生异常后无法恢复；那么，可能更好的做法是让协程崩溃，并在程序的另一部分（通常是从调用函数）处理异常。

# 概述

在本章中，我们介绍了协程，这是 C++中最近引入的一个特性，允许我们编写不需要创建新线程的异步代码。我们实现了一些简单的协程来解释 C++协程的基本要求。此外，我们还学习了如何实现生成器和字符串解析器。最后，我们看到了协程中的异常。

协程在异步编程中很重要，因为它们允许程序在特定点挂起执行并在稍后恢复，同时允许在此期间运行其他任务，所有这些都在同一个线程中运行。它们允许更好的资源利用，减少等待时间，并提高应用程序的可扩展性。

在下一章中，我们将介绍 Boost.Asio – 一个用于在 C++中编写异步代码的非常强大的库。

# 进一步阅读

+   *C++协程入门* ，Andreas Fertig，Meeting C++在线，2024

+   *解码协程* ，Andreas Weiss，CppCon 2022

# 第四部分：使用 Boost 库的高级异步编程

在这部分，我们将学习使用强大的 Boost 库进行高级异步编程技术，使我们能够高效地管理与外部资源和系统级服务交互的任务。我们将探索**Boost.Asio**和**Boost.Cobalt**库，了解它们如何简化异步应用程序的开发，同时提供对复杂过程（如任务管理和协程执行）的精细控制。通过实际示例，我们将看到 Boost.Asio 如何在单线程和多线程环境中处理异步 I/O 操作，以及 Boost.Cobalt 如何抽象出 C++20 协程的复杂性，使我们能够专注于功能而不是低级协程管理。

本部分包含以下章节：

+   *第九章* ，*使用 Boost.Asio 进行异步编程*

+   *第十章* ，*使用 Boost.Cobalt 的协程*
