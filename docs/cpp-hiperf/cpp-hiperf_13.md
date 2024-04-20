# 13

# 使用协程进行异步编程

在上一章中实现的生成器类帮助我们使用协程构建惰性求值序列。C++协程也可以用于异步编程，通过让协程表示异步计算或**异步任务**。尽管异步编程是 C++中协程的最重要驱动因素，但标准库中没有基于协程的异步任务支持。如果你想使用协程进行异步编程，我建议你找到并使用一个补充 C++20 协程的库。我已经推荐了 CppCoro（[`github.com/lewissbaker/cppcoro`](https://github.com/lewissbaker/cppcoro)），在撰写本文时似乎是最有前途的替代方案。还可以使用成熟的 Boost.Asio 库来使用异步协程，稍后在本章中将会看到。

本章将展示使用协程进行异步编程是可能的，并且有可用的库来补充 C++20 协程。更具体地，我们将重点关注：

+   `co_await`关键字和可等待类型

+   实现了一个基本任务类型——一种可以从执行一些异步工作的协程中返回的类型

+   使用协程来举例说明 Boost.Asio 中的异步编程

在继续之前，还应该说一下，本章没有涉及与性能相关的主题，也没有提出很多指导方针和最佳实践。相反，本章更多地作为 C++中异步协程的新特性的介绍。我们将通过探索可等待类型和`co_await`语句来开始这个介绍。

# 重新审视可等待类型

我们在上一章已经谈到了一些关于可等待类型的内容。但现在我们需要更具体地了解`co_await`的作用以及可等待类型是什么。关键字`co_await`是一个一元运算符，意味着它接受一个参数。我们传递给`co_await`的参数需要满足本节中将要探讨的一些要求。

当我们在代码中使用`co_await`时，我们表达了我们正在*等待*一些可能或可能不准备好的东西。如果它还没有准备好，`co_await`会暂停当前执行的协程，并将控制返回给它的调用者。当异步任务完成时，它应该将控制权转回最初等待任务完成的协程。从现在开始，我通常会将等待函数称为**续体**。

现在考虑以下表达式：

```cpp
co_await X{}; 
```

为了使这段代码编译通过，`X`需要是一个可等待类型。到目前为止，我们只使用了一些简单的可等待类型：`std::suspend_always`和`std::suspend_never`。任何直接实现了接下来列出的三个成员函数，或者另外定义了`operator co_wait()`以产生一个具有这些成员函数的对象的类型，都是可等待类型：

+   `await_ready()`返回一个`bool`，指示结果是否已准备就绪（`true`），或者是否需要暂停当前协程并等待结果变得就绪。

+   `await_suspend(coroutine_handle)` - 如果`await_ready()`返回`false`，将调用此函数，传递一个执行`co_await`的协程的句柄。这个函数给了我们一个机会来开始异步工作，并订阅一个通知，当任务完成后触发通知，然后恢复协程。

+   `await_resume()`是负责将结果（或错误）解包回协程的函数。如果在`await_suspend()`启动的工作中发生了错误，这个函数可以重新抛出捕获的错误，或者返回一个错误代码。整个`co_await`表达式的结果是`await_resume()`返回的内容。

```cpp
operator co_await for a time interval:
```

```cpp
using namespace std::chrono;
template <class Rep, class Period> 
auto operator co_await(duration<Rep, Period> d) { 
  struct Awaitable {     
    system_clock::duration d_;
    Awaitable(system_clock::duration d) : d_(d) {} 
    bool await_ready() const { return d_.count() <= 0; }
    void await_suspend(std::coroutine_handle<> h) { /* ... */ } 
    void await_resume() {}
  }; 
  return Awaitable{d};
} 
```

有了这个重载，我们现在可以将一个时间间隔传递给`co_await`运算符，如下所示：

```cpp
std::cout << "just about to go to sleep...\n";
co_await 10ms;                   // Calls operator co_await()
std::cout << "resumed\n"; 
```

示例并不完整，但是给出了如何使用一元运算符`co_await`的提示。正如您可能已经注意到的那样，三个`await_*()`函数不是直接由我们调用的；相反，它们是由编译器插入的代码调用的。另一个示例将澄清编译器所做的转换。假设编译器在我们的代码中遇到以下语句：

```cpp
auto result = co_await expr; 
```

然后编译器将（非常）粗略地将代码转换为以下内容：

```cpp
// Pseudo code
auto&& a = expr;         // Evaluate expr, a is the awaitable
if (!a.await_ready()) {  // Not ready, wait for result
  a.await_suspend(h);    // Handle to current coroutine
                         // Suspend/resume happens here
}
auto result = a.await_resume(); 
```

首先调用`await_ready()`函数来检查是否需要挂起。如果需要，将使用一个句柄调用`await_suspend()`，该句柄将挂起协程（具有`co_await`语句的协程）。最后，请求等待结果并将其分配给`result`变量。

## 隐式挂起点

正如您在众多示例中所看到的，协程通过使用`co_await`和`co_yield`来定义*显式*挂起点。每个协程还有两个*隐式*挂起点：

+   **初始挂起点**，在协程体执行之前协程的初始调用时发生

+   **最终挂起点**，在协程体执行后和协程被销毁前发生

承诺类型通过实现`initial_suspend()`和`final_suspend()`来定义这两个点的行为。这两个函数都返回可等待的对象。通常，我们从`initial_suspend()`函数中传递`std::suspend_always`，以便协程是懒惰启动而不是急切启动。

最终的挂起点对于异步任务非常重要，因为它使我们能够调整`co_await`的行为。通常，已经`co_await:`的协程应在最终挂起点恢复等待的协程。

接下来，让我们更好地了解三个可等待函数的用法以及它们如何与`co_await`运算符配合。

# 实现一个基本的任务类型

我们即将实现的任务类型是可以从代表异步任务的协程中返回的类型。任务是调用者可以使用`co_await`等待的东西。目标是能够编写看起来像这样的异步应用程序代码：

```cpp
auto image = co_await load("image.jpg");
auto thumbnail = co_await resize(image, 100, 100);
co_await save(thumbnail, "thumbnail.jpg"); 
```

标准库已经提供了一种类型，允许函数返回一个调用者可以用于等待计算结果的对象，即`std::future`。我们可以将`std::future`封装成符合可等待接口的东西。但是，`std::future`不支持连续性，这意味着每当我们尝试从`std::future`获取值时，我们都会阻塞当前线程。换句话说，在使用`std::future`时，没有办法组合异步操作而不阻塞。

另一种选择是使用`std::experimental::future`或 Boost 库中的 future 类型，它支持连续性。但是这些 future 类型会分配堆内存，并包含不需要的同步原语。相反，我们将创建一个新类型，具有最小的开销和以下职责：

+   将返回值和异常转发给调用者

+   恢复等待结果的调用者

已经提出了协程任务类型（请参阅[`www7.open-std.org/JTC1/SC22/WG21/docs/papers/2018/p1056r0.html`](http://www7.open-std.org/JTC1/SC22/WG21/docs/papers/2018/p1056r0.html)的 P1056R0），该提案为我们提供了关于我们需要的组件的良好提示。接下来的实现基于 Gor Nishanov 提出的工作和 Lewis Baker 分享的源代码，该源代码可在 CppCoro 库中找到。

这是用于表示异步任务的类模板的实现：

```cpp
template <typename T>
class [[nodiscard]] Task {
  struct Promise { /* ... */ };          // See below
  std::coroutine_handle<Promise> h_;
  explicit Task(Promise & p) noexcept
      : h_{std::coroutine_handle<Promise>::from_promise(p)} {}
 public:
  using promise_type = Promise;
  Task(Task&& t) noexcept : h_{std::exchange(t.h_, {})} {}
  ~Task() { if (h_) h_.destroy(); }
  // Awaitable interface
  bool await_ready() { return false; }
  auto await_suspend(std::coroutine_handle<> c) {
    h_.promise().continuation_ = c;
    return h_;
  }
  auto await_resume() -> T {
    auto& result = h_.promise().result_;
    if (result.index() == 1) {
      return std::get<1>(std::move(result));
    } else {
      std::rethrow_exception(std::get<2>(std::move(result)));
    }
  }
}; 
```

接下来将在后续部分解释每个部分，但首先我们需要实现一个 promise 类型，该类型使用`std::variant`来保存值或错误。promise 还保持对使用`continuation_`数据成员等待任务完成的协程的引用：

```cpp
struct Promise {
  std::variant<std::monostate, T, std::exception_ptr> result_;
  std::coroutine_handle<> continuation_;  // A waiting coroutine
  auto get_return_object() noexcept { return Task{*this}; }
  void return_value(T value) { 
    result_.template emplace<1>(std::move(value)); 
  }
  void unhandled_exception() noexcept {
    result_.template emplace<2>(std::current_exception());
  }
  auto initial_suspend() { return std::suspend_always{}; }
  auto final_suspend() noexcept {
    struct Awaitable {
      bool await_ready() noexcept { return false; }
      auto await_suspend(std::coroutine_handle<Promise> h) noexcept {
        return h.promise().continuation_;
      }
      void await_resume() noexcept {}
    };
    return Awaitable{};
  }
}; 
```

重要的是要区分我们正在使用的两个协程句柄：标识*当前协程*的句柄和标识*继续执行*的句柄。

请注意，由于`std::variant`的限制，此实现不支持`Task<void>`，并且我们不能在同一个 promise 类型上同时具有`return_value()`和`return_void()`的限制。不支持`Task<void>`是不幸的，因为并非所有异步任务都必然返回值。我们将通过为`Task<void>`提供模板特化来克服这个限制。

由于我们在上一章中实现了一些协程返回类型（`Resumable`和`Generator`），您已经熟悉了可以从协程返回的类型的要求。在这里，我们将专注于对您新的事物，例如异常处理和恢复当前等待我们的调用者的能力。让我们开始看一下`Task`和`Promise`如何处理返回值和异常。

## 处理返回值和异常

异步任务可以通过返回（一个值或`void`）或抛出异常来完成。值和错误需要交给调用者，调用者一直在等待任务完成。通常情况下，这是 promise 对象的责任。

`Promise`类使用`std::variant`来存储三种可能结果的结果：

+   根本没有值（`std::monostate`）。我们在我们的 variant 中使用这个来使其默认可构造，但不需要其他两种类型是默认可构造的。

+   类型为`T`的返回值，其中`T`是`Task`的模板参数。

+   `std::exception_ptr`，它是对先前抛出的异常的句柄。

异常是通过在`Promise::unhandled_exception()`函数内部使用`std::current_exception()`函数来捕获的。通过存储`std::exception_ptr`，我们可以在另一个上下文中重新抛出此异常。当异常在线程之间传递时，也是使用的机制。

使用`co_return value;`的协程必须具有实现`return_value()`的 promise 类型。然而，使用`co_return;`或在没有返回值的情况下运行的协程必须具有实现`return_void()`的 promise 类型。实现同时包含`return_void()`和`return_value()`的 promise 类型会生成编译错误。

## 恢复等待的协程

当异步任务完成时，它应该将控制权转移到等待任务完成的协程。为了能够恢复这个继续执行，`Task`对象需要`coroutine_handle`到继续执行的协程。这个句柄被传递给`Task`对象的`await_suspend()`函数，并且我们方便地确保将该句柄保存到 promise 对象中：

```cpp
class Task {
  // ...
  auto await_suspend(std::coroutine_handle<> c) {
    h_.promise().continuation_ = c;      // Save handle
    return h_;
  }
  // ... 
```

`final_suspend()`函数负责在此协程的最终挂起点挂起，并将执行转移到等待的协程。这是`Promise`的相关部分，供您参考：

```cpp
auto Promise::final_suspend() noexcept {
  struct Awaitable {
    bool await_ready() noexcept { return false; } // Suspend
    auto await_suspend(std::coroutine_handle<Promise> h) noexcept{
      return h.promise().continuation_;  // Transfer control to
    }                                    // the waiting coroutine
    void await_resume() noexcept {}
  };
  return Awaitable{};
} 
```

首先，从`await_ready()`返回`false`将使协程在最终挂起点挂起。我们这样做的原因是为了保持 promise 仍然存活，并且可以让继续执行有机会从 promise 中取出结果。

接下来，让我们看一下`await_suspend()`函数。这是我们想要恢复执行的地方。我们可以直接在`continuation_`句柄上调用`resume()`，并等待它完成，就像这样：

```cpp
// ...
auto await_suspend(std::coroutine_handle<Promise> h) noexcept {
  h.promise().resume();         // Not recommended
}
// ... 
```

然而，这样做会有在堆栈上创建一长串嵌套调用帧的风险，最终可能导致堆栈溢出。让我们看看通过一个简短的例子使用两个协程`a()`和`b()`会发生什么：

```cpp
auto a() -> Task<int> {  co_return 42; } 
auto b() -> Task<int> {         // The continuation
  auto sum = 0;
  for (auto i = 0; i < 1'000'000; ++i) {
    sum += co_await a();
  }
  co_return sum;
} 
```

如果与协程`a()`关联的`Promise`对象直接在协程`b()`的句柄上调用`resume()`，则在`a()`的调用帧之上会在堆栈上创建一个新的调用帧来恢复`b()`。这个过程会在循环中一遍又一遍地重复，为每次迭代在堆栈上创建新的嵌套调用帧。当两个函数互相调用时，这种调用顺序是一种递归形式，有时被称为相互递归：

![](img/B15619_13_01.png)

图 13.1：协程 b()调用协程 a()，协程 a()恢复 b()，协程 b()调用 a()，协程 a()恢复 b()，依此类推

尽管为`b()`创建了一个协程帧，但每次调用`resume()`来恢复协程`b()`都会在堆栈上创建一个新的帧。避免这个问题的解决方案称为**对称传输**。任务对象不是直接从即将完成的协程中恢复继续，而是从`await_suspend()`中返回标识继续的`coroutine_handle`：

```cpp
// ...
auto await_suspend(std::coroutine_handle<Promise> h) noexcept {
  return h.promise().continuation_;     // Symmetric transfer
}
// ... 
```

然后编译器保证会发生一种叫做*尾递归优化*的优化。在我们的情况下，这意味着编译器将能够直接将控制转移到继续，而不会创建新的嵌套调用帧。

我们不会再花更多时间讨论对称传输和尾调用的细节，但可以在 Lewis Baker 的文章*C++ Coroutines: Understanding Symmetric Transfer*中找到关于这些主题的出色且更深入的解释，网址为[`lewissbaker.github.io/2020/05/11/understanding_symmetric_transfer`](https://lewissbaker.github.io/2020/05/11/understanding_symmetric_transfer)。

如前所述，我们的`Task`模板有一个限制，即不能处理`void`类型的模板参数。现在是时候修复这个问题了。

## 支持 void 任务

为了克服之前提到的关于无法处理不产生任何值的任务的限制，我们需要为`Task<void>`进行模板特化。这里为了完整起见进行了详细说明，但除了之前定义的一般`Task`模板之外，并没有添加太多新的见解：

```cpp
template <>
class [[nodiscard]] Task<void> {

  struct Promise {
    std::exception_ptr e_;   // No std::variant, only exception
    std::coroutine_handle<> continuation_; 
    auto get_return_object() noexcept { return Task{*this}; }
    void return_void() {}   // Instead of return_value() 
    void unhandled_exception() noexcept { 
      e_ = std::current_exception(); 
    }
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept {
      struct Awaitable {
        bool await_ready() noexcept { return false; }
        auto await_suspend(std::coroutine_handle<Promise> h) noexcept {
          return h.promise().continuation_;
        }
        void await_resume() noexcept {}
      };
      return Awaitable{};
    }
  };
  std::coroutine_handle<Promise> h_;
  explicit Task(Promise& p) noexcept 
      : h_{std::coroutine_handle<Promise>::from_promise(p)} {}
public:
  using promise_type = Promise;

  Task(Task&& t) noexcept : h_{std::exchange(t.h_, {})} {}
  ~Task() { if (h_) h_.destroy(); }
  // Awaitable interface
  bool await_ready() { return false; }
  auto await_suspend(std::coroutine_handle<> c) {
    h_.promise().continuation_ = c;
    return h_;
  }
  void await_resume() {
    if (h_.promise().e_)
      std::rethrow_exception(h_.promise().e_);
  }
}; 
```

这个模板特化中的 promise 类型只保留对潜在未处理异常的引用。而不是定义`return_value()`，promise 包含成员函数`return_void()`。

我们现在可以表示返回值或`void`的任务。但在我们实际构建一个独立程序来测试我们的`Task`类型之前，还有一些工作要做。

## 同步等待任务完成

`Task`类型的一个重要方面是，无论是什么调用了返回`Task`的协程，都必须对其进行`co_await`，因此也是一个协程。这创建了一系列协程（继续）。例如，假设我们有这样一个协程：

```cpp
Task<void> async_func() {      // A coroutine
  co_await some_func();
} 
```

然后，就不可能以以下方式使用它：

```cpp
void f() {                          
  co_await async_func(); // Error: A coroutine can't return void
} 
```

一旦我们调用返回`Task`的异步函数，我们需要对其进行`co_await`，否则什么都不会发生。这也是我们声明`Task`为`nodiscard`的原因：这样如果忽略返回值，它会生成编译警告，就像这样：

```cpp
void g() {        
  async_func();          // Warning: Does nothing
} 
```

协程的强制链接具有一个有趣的效果，即我们最终到达程序的`main()`函数，而 C++标准规定不允许它是一个协程。这需要以某种方式解决，提出的解决方案是提供至少一个函数来同步等待异步链完成。例如，CppCoro 库包括函数`sync_wait()`，它具有打破协程链的效果，使得普通函数可以使用协程成为可能。

不幸的是，实现 `sync_wait()` 相当复杂，但为了至少使得编译和测试我们的 `Task` 类成为可能，我将在这里提供一个基于标准 C++ 提案 P1171R0 的简化版本，[`wg21.link/P1171R0`](https://wg21.link/P1171R0)。我们的目标是能够编写如下的测试程序：

```cpp
auto some_async_func() -> Task<int> { /* ... */ }
int main() { 
  auto result = sync_wait(some_async_func());
  return result;
} 
```

为了测试和运行异步任务，让我们继续实现 `sync_wait()`。

### 实现 sync_wait()

`sync_wait()` 在内部使用了一个专门为我们的目的设计的自定义任务类，称为 `SyncWaitTask`。它的定义将在稍后揭示，但首先让我们看一下函数模板 `sync_wait()` 的定义：

```cpp
template<typename T>
using Result = decltype(std::declval<T&>().await_resume());
template <typename T>
Result<T> sync_wait(T&& task) {
  if constexpr (std::is_void_v<Result<T>>) {
    struct Empty {};
    auto coro = [&]() -> detail::SyncWaitTask<Empty> {
      co_await std::forward<T>(task);
      co_yield Empty{};
      assert(false);
    };
    coro().get();
  } else {
    auto coro = [&]() -> detail::SyncWaitTask<Result<T>> {
      co_yield co_await std::forward<T>(task);
      // This coroutine will be destroyed before it
      // has a chance to return.
      assert(false);
    };
    return coro().get();
  }
} 
```

首先，为了指定任务返回的类型，我们使用了 `decltype` 和 `declval` 的组合。这种相当繁琐的 `using` 表达式给出了由传递给 `sync_wait()` 的任务的类型 `T::await_resume()` 返回的类型。

在 `sync_wait()` 中，我们区分返回值和返回 `void` 的任务。我们在这里做出区分，以避免需要实现 `SyncWaitTask` 的模板特化来处理 `void` 和非 `void` 类型。通过引入一个空的 `struct`，可以将这两种情况类似地处理，该结构可以作为模板参数提供给 `SyncWaitTask`，用于处理 `void` 任务。

在实际返回值的情况下，使用 lambda 表达式来定义一个协程，该协程将在结果上进行 `co_await`，然后最终产生其值。重要的是要注意，协程可能会从 `co_await` 在另一个线程上恢复，这要求我们在 `SyncWaitTask` 的实现中使用同步原语。

在协程 lambda 上调用 `get()` 会恢复协程，直到它产生一个值。`SyncWaitTask` 的实现保证协程 lambda 在 `co_yield` 语句之后永远不会有机会再次恢复。

在前一章中我们广泛使用了 `co_yield`，但没有提及它与 `co_await` 的关系；即以下 `co_yield` 表达式：

```cpp
 co_yield some_value; 
```

被编译器转换为：

```cpp
co_await promise.yield_value(some_value); 
```

`promise` 是与当前执行的协程关联的 promise 对象。当尝试理解 `sync_wait()` 和 `SyncWaitTask` 类之间的控制流时，了解这一点是有帮助的。

### 实现 SyncWaitTask

现在我们准备检查 `SyncWaitTask`，这是一种类型，只用作 `sync_wait()` 的辅助。因此，我们将其添加到名为 `detail` 的命名空间下，以明确表示这个类是一个实现细节：

```cpp
namespace detail { // Implementation detail
template <typename T>
class SyncWaitTask {  // A helper class only used by sync_wait()
  struct Promise { /* ... */ }; // See below
  std::coroutine_handle<Promise> h_;
  explicit SyncWaitTask(Promise& p) noexcept
      : h_{std::coroutine_handle<Promise>::from_promise(p)} {}
 public:
  using promise_type = Promise;

  SyncWaitTask(SyncWaitTask&& t) noexcept 
      : h_{std::exchange(t.h_, {})} {}
  ~SyncWaitTask() { if (h_) h_.destroy();}
  // Called from sync_wait(). Will block and retrieve the
  // value or error from the task passed to sync_wait()
  T&& get() {
    auto& p = h_.promise();
    h_.resume();
    p.semaphore_.acquire();               // Block until signal
    if (p.error_)
      std::rethrow_exception(p.error_);
    return static_cast<T&&>(*p.value_);
  }
  // No awaitable interface, this class will not be co_await:ed
};
} // namespace detail 
```

最值得注意的部分是函数 `get()` 及其对 promise 对象拥有的信号量的 `acquire()` 的阻塞调用。这是使得这种任务类型同步等待结果准备好的关键。拥有二进制信号量的 promise 类型如下：

```cpp
struct Promise {
  T* value_{nullptr};
  std::exception_ptr error_;
  std::binary_semaphore semaphore_;
  SyncWaitTask get_return_object() noexcept { 
    return SyncWaitTask{*this}; 
  }
  void unhandled_exception() noexcept { 
    error_ = std::current_exception(); 
  }
  auto yield_value(T&& x) noexcept {     // Result has arrived
    value_ = std::addressof(x);
    return final_suspend();
  }
  auto initial_suspend() noexcept { 
    return std::suspend_always{}; 
  }
  auto final_suspend() noexcept { 
  struct Awaitable {
      bool await_ready() noexcept { return false; }
      void await_suspend(std::coroutine_handle<Promise> h) noexcept {
        h.promise().semaphore_.release();          // Signal! 
      }
      void await_resume() noexcept {}
    };
    return Awaitable{};
  }
  void return_void() noexcept { assert(false); }
}; 
```

这里有很多我们已经讨论过的样板代码。但要特别注意 `yield_value()` 和 `final_suspend()`，这是这个类的有趣部分。回想一下，在 `sync_wait()` 中，协程 lambda 产生了返回值，如下所示：

```cpp
// ...
auto coro = [&]() -> detail::SyncWaitTask<Result<T>> {
  co_yield co_await std::forward<T>(task);  
  // ... 
```

因此，一旦值被产出，我们就会进入 promise 对象的 `yield_value()`。而 `yield_value()` 可以返回一个可等待类型的事实，使我们有机会定制 `co_yield` 关键字的行为。在这种情况下，`yield_value()` 返回一个可等待对象，该对象将通过二进制信号量发出信号，表明原始 `Task` 对象已经产生了一个值。

在 `await_suspend()` 中发出信号。我们不能比这更早发出信号，因为等待信号的代码的另一端最终会销毁协程。销毁协程只能在协程处于挂起状态时发生。

`SyncWaitTask::get()`中对`semaphore_`.`acquire()`的阻塞调用将在信号上返回，最终计算值将被传递给调用`sync_wait()`的客户端。

## 使用 sync_wait()测试异步任务

最后，可以构建一个使用`Task`和`sync_wait()`的小型异步测试程序，如下所示：

```cpp
auto height() -> Task<int> { co_return 20; }     // Dummy coroutines
auto width() -> Task<int> { co_return 30; }
auto area() -> Task<int> { 
  co_return co_await height() * co_await width(); 
}

int main() {
  auto a = area();
  int value = sync_wait(a);
  std::cout << value;          // Outputs: 600
} 
```

我们已经实现了使用 C++协程的最低限度基础设施。然而，为了有效地使用协程进行异步编程，还需要更多的基础设施。这与生成器（在上一章中介绍）有很大的不同，生成器在我们真正受益之前需要进行相当少量的准备工作。为了更接近现实世界，我们将在接下来的章节中探索一些使用 Boost.Asio 的示例。我们将首先尝试将基于回调的 API 包装在与 C++协程兼容的 API 中。

# 包装基于回调的 API

有许多基于回调的异步 API。通常，异步函数接受调用者提供的回调函数。异步函数立即返回，然后最终在异步函数计算出一个值或完成等待某事时调用回调（完成处理程序）。

为了向您展示异步基于回调的 API 是什么样子，我们将一窥名为**Boost.Asio**的异步 I/O 的 Boost 库。关于 Boost.Asio 有很多内容需要学习，这里不会涉及到太多；我只会描述与 C++协程直接相关的 Boost 代码的绝对最低限度。

为了使代码适应本书的页面，示例假设每当我们使用 Boost.Asio 的代码时，已经定义了以下命名空间别名：

```cpp
namespace asio = boost::asio; 
```

这是使用 Boost.Asio 延迟函数调用但不阻塞当前线程的完整示例。这个异步示例在单个线程中运行：

```cpp
#include <boost/asio.hpp>
#include <chrono>
#include <iostream>
using namespace std::chrono;
namespace asio = boost::asio;
int main() {
  auto ctx = asio::io_context{};
  auto timer = asio::system_timer{ctx};
  timer.expires_from_now(1000ms);
  timer.async_wait([](auto error) {       // Callback
    // Ignore errors..                          
    std::cout << "Hello from delayed callback\n"; 
  });
  std::cout << "Hello from main\n";
  ctx.run();
} 
```

编译和运行此程序将生成以下输出：

```cpp
Hello from main
Hello from delayed callback 
```

在使用 Boost.Asio 时，我们总是需要创建一个运行事件处理循环的`io_context`对象。对`async_wait()`的调用是异步的；它立即返回到`main()`并在计时器到期时调用回调（lambda）。

计时器示例不使用协程，而是使用回调 API 来提供异步性。Boost.Asio 也与 C++20 协程兼容，我稍后会进行演示。但在探索可等待类型的过程中，我们将绕道而行，而是假设我们需要在 Boost.Asio 的基于回调的 API 之上提供一个基于协程的 API，该 API 返回可等待类型。这样，我们可以使用`co_await`表达式来调用并等待（但不阻塞当前线程）异步任务完成。我们希望能够写出类似这样的代码，而不是使用回调：

```cpp
std::cout << "Hello! ";
co_await async_sleep(ctx, 100ms);
std::cout << "Delayed output\n"; 
```

让我们看看如何实现函数`async_sleep()`，以便可以与`co_await`一起使用。我们将遵循的模式是让`async_sleep()`返回一个可等待对象，该对象将实现三个必需的函数：`await_ready()`、`await_suspend()`和`await_resume()`。代码解释将在此后跟随：

```cpp
template <typename R, typename P>
auto async_sleep(asio::io_context& ctx,
                 std::chrono::duration<R, P> d) {
  struct Awaitable {
    asio::system_timer t_;
    std::chrono::duration<R, P> d_;
    boost::system::error_code ec_{};
    bool await_ready() { return d_.count() <= 0; }
    void await_suspend(std::coroutine_handle<> h) {
      t_.expires_from_now(d_);
      t_.async_wait(this, h mutable {
        this->ec_ = ec;
        h.resume();
      });
    } 
    void await_resume() {
      if (ec_) throw boost::system::system_error(ec_);
    }
  };
  return Awaitable{asio::system_timer{ctx}, d};
} 
```

再次，我们正在创建一个自定义的可等待类型，它完成了所有必要的工作：

+   除非计时器已经达到零，否则`await_ready()`将返回`false`。

+   `await_suspend()`启动异步操作并传递一个回调，当计时器到期或产生错误时将调用该回调。回调保存错误代码（如果有）并恢复挂起的协程。

+   `await_resume()`没有结果需要解包，因为我们正在包装的异步函数`boost::asio::timer::async_wait()`除了可选的错误代码外不返回任何值。

在我们实际测试`async_sleep()`的独立程序之前，我们需要一种方法来启动`io_context`运行循环并打破协程链，就像我们之前测试`Task`类型时所做的那样。我们将通过实现两个函数`run_task()`和`run_task_impl()`以及一个称为`Detached`的天真协程返回类型来以一种相当巧妙的方式来做到这一点，该类型忽略错误处理并可以被调用者丢弃：

```cpp
// This code is here just to get our example up and running
struct Detached { 
  struct promise_type {
    auto get_return_object() { return Detached{}; }
    auto initial_suspend() { return std::suspend_never{}; }
    auto final_suspend() noexcept { return std::suspend_never{};}
    void unhandled_exception() { std::terminate(); } // Ignore
    void return_void() {}
  };
};
Detached run_task_impl(asio::io_context& ctx, Task<void>&& t) {
  auto wg = asio::executor_work_guard{ctx.get_executor()};
  co_await t;
}
void run_task(asio::io_context& ctx, Task<void>&& t) {
  run_task_impl(ctx, std::move(t));
  ctx.run();
} 
```

`Detached`类型使协程立即启动并从调用者分离运行。`executor_work_guard`防止`run()`调用在协程`run_task_impl()`完成之前返回。

通常应避免启动操作并分离它们。这类似于分离的线程或分配的没有任何引用的内存。然而，此示例的目的是演示我们可以使用可等待类型以及如何编写异步程序并在单线程中运行。

一切就绪；名为`async_sleep()`的包装器返回一个`Task`和一个名为`run_task()`的函数，该函数可用于执行任务。是时候编写一个小的协程来测试我们实现的新代码了：

```cpp
auto test_sleep(asio::io_context& ctx) -> Task<void> {
  std::cout << "Hello!  ";
  co_await async_sleep(ctx, 100ms);
  std::cout << "Delayed output\n";
}
int main() {
  auto ctx = asio::io_context{};
  auto task = test_sleep(ctx);
  run_task(ctx, std::move(task));  
}; 
```

执行此程序将生成以下输出：

```cpp
Hello! Delayed output 
```

您已经看到了如何将基于回调的 API 包装在可以被`co_await`使用的函数中，因此允许我们使用协程而不是回调进行异步编程。该程序还提供了可等待类型中的函数如何使用的典型示例。然而，正如前面提到的，最近的 Boost 版本，从 1.70 开始，已经提供了与 C++20 协程兼容的接口。在下一节中，我们将在构建一个小型 TCP 服务器时使用这个新的协程 API。

# 使用 Boost.Asio 的并发服务器

本节将演示如何编写并发程序，该程序具有多个执行线程，但仅使用单个操作系统线程。我们将要实现一个基本的并发单线程 TCP 服务器，可以处理多个客户端。C++标准库中没有网络功能，但幸运的是，Boost.Asio 为我们提供了一个平台无关的接口，用于处理套接字通信。

我将演示如何使用`boost::asio::awaitable`类，而不是包装基于回调的 Boost.Asio API，以展示使用协程进行异步应用程序编程的更真实的示例。类模板`boost::asio::awaitable`对应于我们之前创建的`Task`模板；它用作表示异步计算的协程的返回类型。

## 实施服务器

服务器非常简单；一旦客户端连接，它就开始更新一个数字计数器，并在更新时写回该值。这次我们将从上到下跟踪代码，从`main()`函数开始：

```cpp
#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/use_awaitable.hpp>
using namespace std::chrono;
namespace asio = boost::asio;
using boost::asio::ip::tcp;
int main() {
  auto server = [] {
    auto endpoint = tcp::endpoint{tcp::v4(), 37259};
    auto awaitable = listen(endpoint);
    return awaitable;
  };
  auto ctx = asio::io_context{};
  asio::co_spawn(ctx, server, asio::detached);
  ctx.run(); // Run event loop from main thread
} 
```

强制性的`io_context`运行事件处理循环。也可以从多个线程调用`run()`，如果我们希望服务器执行多个操作系统线程。在我们的情况下，我们只使用一个线程，但具有多个并发流。函数`boost::asio::co_spawn()`启动一个分离的并发流。服务器使用 lambda 表达式实现；它定义了一个 TCP 端点（端口 37259），并开始在端点上监听传入的客户端连接。

协程`listen()`相当简单，如下所示：

```cpp
auto listen(tcp::endpoint endpoint) -> asio::awaitable<void> {
  auto ex = co_await asio::this_coro::executor;
  auto a = tcp::acceptor{ex, endpoint};
  while (true) {
    auto socket = co_await a.async_accept(asio::use_awaitable);
    auto session = [s = std::move(socket)]() mutable {
      auto awaitable = serve_client(std::move(s));
      return awaitable;
    };
    asio::co_spawn(ex, std::move(session), asio::detached);
  }
} 
```

执行器是实际执行我们的异步函数的对象。执行器可以表示线程池或单个系统线程，例如。我们很可能会在即将推出的 C++版本中看到某种形式的执行器，以便让我们程序员更多地控制和灵活地执行我们的代码（包括 GPU）。

接下来，协程运行一个无限循环，并等待 TCP 客户端连接。第一个`co_await`表达式在新客户端成功连接到服务器时返回一个套接字。然后将套接字对象移动到协程`serve_client()`中，该协程将为新连接的客户端提供服务，直到客户端断开连接。

服务器的主要应用逻辑发生在处理每个客户端的协程中。下面是它的样子：

```cpp
auto serve_client(tcp::socket socket) -> asio::awaitable<void> {
  std::cout << "New client connected\n";
  auto ex = co_await asio::this_coro::executor;
  auto timer = asio::system_timer{ex};
  auto counter = 0;
  while (true) {
    try {
      auto s = std::to_string(counter) + "\n";
      auto buf = asio::buffer(s.data(), s.size());
      auto n = co_await async_write(socket, buf, asio::use_awaitable);
      std::cout << "Wrote " << n << " byte(s)\n";
      ++counter;
      timer.expires_from_now(100ms);
      co_await timer.async_wait(asio::use_awaitable);
    } catch (...) {
      // Error or client disconnected
      break;
    }
  }
} 
```

每个协程调用在整个客户端会话期间为一个唯一的客户端提供服务；它在客户端断开连接之前一直运行。协程会定期更新计数器（每 100 毫秒一次），并使用`async_write()`异步将值写回给客户端。请注意，尽管它调用了两个异步操作：`async_write()`和`async_wait()`，但我们可以以线性方式编写函数`serve_client()`。

## 运行和连接到服务器

一旦我们启动了这个服务器，我们可以在端口 37259 上连接客户端。为了尝试这个，我使用了一个叫做`nc`（netcat）的工具，它可以用于通过 TCP 和 UDP 进行通信。下面是一个客户端连接到运行在本地主机上的服务器的短会话的示例：

```cpp
[client] $ nc localhost 37259              
0
1
2
3 
```

我们可以启动多个客户端，它们都将由专用的`serve_client()`协程调用来提供服务，并且拥有自己的递增计数变量的副本，如下面的屏幕截图所示：

![](img/B15619_13_02.png)

图 13.2：运行中的服务器与两个连接的客户端

另一种创建同时为多个会话提供服务的应用程序的方法是为每个连接的新客户端创建一个线程。然而，与使用协程的模型相比，线程的内存开销会大大降低会话数量的限制。

这个例子中的协程都在同一个线程上执行，这使得共享资源的锁定变得不必要。想象一下，如果我们有一个每个会话都会更新的全局计数器。如果我们使用多个线程，对全局计数器的访问就需要某种形式的同步（使用互斥锁或原子数据类型）。但是对于在同一线程上执行的协程来说，这是不必要的。换句话说，在同一线程上执行的协程可以共享状态，而不需要使用任何锁定原语。

## 我们通过服务器实现了什么（以及我们没有实现的）

Boost.Asio 示例应用程序演示了协程可以用于异步编程。我们可以使用`co_await`语句以线性方式编写代码，而不是使用嵌套回调来实现延续。然而，这个例子很简单，避开了一些真正重要的异步编程方面，比如：

+   异步读写操作。服务器只向其客户端写入数据，并忽略了同步读写操作的挑战。

+   取消异步任务和优雅关闭。服务器在一个无限循环中运行，完全忽略了干净关闭的挑战。

+   使用多个`co_await`语句时的错误处理和异常安全。

这些主题非常重要，但超出了本书的范围。我已经提到了最好避免使用分离的操作。使用`boost::asio::co_spawn()`创建分离的任务，应该非常谨慎。一个相当新的用于避免分离工作的编程范式被称为**结构化并发**。它旨在通过将并发封装到通用和可重用的算法中（例如`when_all()`和`stop_when()`）来解决异常安全和多个异步任务的取消。关键思想是永远不允许某个子任务的生命周期超过其父任务。这使得可以安全地通过引用传递本地变量给异步子操作，并且性能更好。严格嵌套的并发任务生命周期也使得代码更容易理解。

另一个重要的方面是，异步任务应该始终是懒惰的（立即挂起），这样在抛出任何异常之前就可以附加继续。如果您想要能够以安全的方式取消任务，这也是一个要求。

未来几年很可能会有很多关于这一重要主题的讲座、库和文章。CppCon 2019 的两场讲座涉及了这个主题。

+   *用于 C++中异步的统一抽象*，Eric Neibler 和 D.S. Hollman，[`sched.co/SfrC`](https://sched.co/SfrC)

+   *结构化并发：使用协程和算法编写更安全的并发代码*，Lewis Baker，[`sched.co/SfsU`](https://sched.co/SfsU)

# 总结

在本章中，您已经看到了如何使用 C++协程来编写异步任务。为了能够以`Task`类型和`sync_wait()`函数的形式实现基础设施，您需要充分理解可等待类型的概念以及它们如何用于自定义 C++中协程的行为。

通过使用 Boost.Asio，我们可以构建一个真正最小但完全功能的并发服务器应用程序，该应用程序在单个线程上执行，同时处理多个客户会话。

最后，我简要介绍了一种称为结构化并发的方法论，并指出了一些关于这个主题的更多信息的方向。

在下一章中，我们将继续探讨并行算法，这是一种通过利用多个核心来加速并发程序的方法。
