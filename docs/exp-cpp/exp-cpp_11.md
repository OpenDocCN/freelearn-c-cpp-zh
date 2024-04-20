# 第十一章：设计并发数据结构

在上一章中，我们简要介绍了 C++中并发和多线程的基础知识。并发代码设计中最大的挑战之一是正确处理数据竞争。线程同步和协调并不是一个容易理解的话题，尽管我们可能认为它是最重要的话题。虽然我们可以在任何我们对数据竞争有丝毫怀疑的地方使用互斥量等同步原语，但这并不是我们建议的最佳实践。

设计并发代码的更好方式是尽量避免使用锁。这不仅会提高应用程序的性能，还会使其比以前更安全。说起来容易做起来难——无锁编程是一个具有挑战性的话题，我们将在本章中介绍。特别是，我们将更深入地了解设计无锁算法和数据结构的基础知识。这是一个由许多杰出的开发人员不断研究的难题。我们将简要介绍无锁编程的基础知识，这将让您了解如何以高效的方式构建代码。阅读完本章后，您将更好地能够理解数据竞争问题，并获得设计并发算法和数据结构所需的基本知识。这也可能有助于您的一般设计技能，以构建容错系统。

本章将涵盖以下主题：

+   理解数据竞争和基于锁的解决方案

+   在 C++代码中使用原子操作

+   设计无锁数据结构

# 技术要求

本章中使用 g++编译器的`-std=c++2a`选项来编译示例。您可以在以下链接找到本章中使用的源文件：[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# 更深入地了解数据竞争

正如已经多次提到的，数据竞争是程序员们尽量避免的情况。在上一章中，我们讨论了死锁及其避免方法。上一章中我们使用的最后一个示例是创建一个线程安全的单例模式。假设我们使用一个类来创建数据库连接（一个经典的例子）。

以下是一个跟踪数据库连接的模式的简单实现。每次需要访问数据库时保持单独的连接并不是一个好的做法。相反，我们可以重用现有的连接来从程序的不同部分查询数据库：

```cpp
namespace Db {
  class ConnectionManager 
  {
  public:
    static std::shared_ptr<ConnectionManager> get_instance()
 {
 if (instance_ == nullptr) {
 instance_.reset(new ConnectionManager());
 }
 return instance_;
 }

    // Database connection related code omitted
  private:
    static std::shared_ptr<ConnectionManager> instance_{nullptr};
  };
}
```

让我们更详细地讨论这个例子。在上一章中，我们加入了锁来保护`get_instance()`函数免受数据竞争的影响。让我们详细说明为什么这样做。为了简化这个例子，以下是我们感兴趣的四行：

```cpp
get_instance()
  if (_instance == nullptr)
    instance_.reset(new)
  return instance_;
```

现在，想象一下我们运行一个访问`get_instance()`函数的线程。我们称它为`线程 A`，它执行的第一行是条件语句，如下所示：

```cpp
get_instance()
  if (_instance == nullptr)   <--- Thread A
    instance_.reset(new)
  return instance_;
```

它将逐行执行指令。我们更感兴趣的是第二个线程（标记为`线程 B`），它开始并发执行`线程 A`的函数。在函数并发执行期间可能出现以下情况：

```cpp
get_instance()
  if (_instance == nullptr)   <--- Thread B (checking)
    instance_.reset(new)      <--- Thread A (already checked)
  return instance_;
```

`线程 B`在将`instance_`与`nullptr`进行比较时得到了一个正结果。`线程 A`已经通过了相同的检查，并将`instance_`设置为一个新对象。从`线程 A`的角度来看，一切都很正常，它刚刚通过了条件检查，重置了`instances`，并将继续执行下一行返回`instance_`。然而，`线程 B`在它的值改变之前就比较了`instance_`。因此，`线程 B`也继续设置`instance_`的值：

```cpp
get_instance()
  if (_instance == nullptr)   
    instance_.reset(new)      <--- Thread B (already checked)
  return instance_;           <--- Thread A (returns)
```

前面的问题是`线程 B`在`instance_`已经被设置之后重置了它。此外，我们将`get_instance()`视为一个单独的操作；它由几条指令组成，每条指令都由一个线程按顺序执行。为了让两个线程不相互干扰，操作不应该包含多于一条指令。

我们关注数据竞争的原因是代码块中的间隙。代码行之间的这个间隙允许线程相互干扰。当你使用互斥锁等同步原语设计解决方案时，你应该考虑你错过的所有间隙，因为解决方案可能不正确。下面的修改使用了在前一章讨论过的互斥锁和`双重检查`锁定模式：

```cpp
static std::shared_ptr<ConnectionManager> get_instance()
{
  if (instance_ == nullptr) {
    // mutex_ is declared in the private section
 std::lock_guard lg{mutex_};
 if (instance_ == nullptr) { // double-checking
 instance_.reset(new ConnectionManager());
 }
  }
  return instance_;
}
```

当两个线程尝试访问`instance_`对象时会发生什么：

```cpp
get_instance()
  if (instance_ == nullptr)     <--- Thread B
    lock mutex                  <--- Thread A (locks the mutex)
    if (instance_ == nullptr)
      instance_.reset(new)
    unlock mutex
  return instance_
```

现在，即使两个线程都通过了第一次检查，其中一个线程也会锁定互斥锁。当一个线程尝试锁定互斥锁时，另一个线程会重置实例。为了确保它尚未设置，我们使用第二次检查（这就是为什么它被称为**双重检查锁定**）：

```cpp
get_instance()
  if (instance_ == nullptr)
    lock mutex                  <--- Thread B (tries to lock, waits)
    if (instance_ == nullptr)   <--- Thread A (double check)
      instance_.reset(new)      
    unlock mutex
  return instance_
```

当`线程 A`完成设置`instance_`后，它会解锁互斥锁，这样`线程 B`就可以继续锁定和重置`instance_`：

```cpp
get_instance()
  if (instance_ == nullptr)
    lock mutex                  <--- Thread B (finally locks the mutex)
    if (instance_ == nullptr)   <--- Thread B (check is not passed)
      instance_.reset(new)      
    unlock mutex                <--- Thread A (unlocked the mutex)
  return instance_              <--- Thread A (returns)  
```

根据经验法则，你应该总是查看代码中的细节。两个语句之间总是有一个间隙，这个间隙会导致两个或更多的线程相互干扰。接下来的部分将详细讨论一个经典的递增数字的例子。

# 同步递增

几乎每本涉及线程同步主题的书都使用递增数字的经典例子作为数据竞争的例子。这本书也不例外。例子如下：

```cpp
#include <thread>

int counter = 0;

void foo()
{
 counter++;
}

int main()
{
  std::jthread A{foo};
  std::jthread B{foo};
  std::jthread C{[]{foo();}};
  std::jthread D{
    []{
      for (int ix = 0; ix < 10; ++ix) { foo(); }
    }
  };
}
```

我们添加了几个线程，使示例变得更加复杂。前面的代码只是使用四个不同的线程递增`counter`变量。乍一看，任何时候只有一个线程递增`counter`。然而，正如我们在前一节中提到的，我们应该注意并寻找代码中的间隙。`foo()`函数似乎缺少一个。递增运算符的行为如下（伪代码）：

```cpp
auto res = counter;
counter = counter + 1;
return res;
```

现在，我们发现了本不应该有的间隙。因此，任何时候只有一个线程执行前面三条指令中的一条。也就是说，类似下面的情况是可能的：

```cpp
auto res = counter;     <--- thread A
counter = counter + 1;  <--- thread B
return res;             <--- thread C
```

例如，`线程 B`可能在`线程 A`读取其先前值时修改`counter`的值。这意味着`线程 A`在`线程 B`已经完成递增`counter`时会给`counter`赋予一个新的递增值。混乱引入了混乱，迟早，我们的大脑会因为尝试理解操作的顺序而爆炸。作为一个经典的例子，我们将继续使用线程锁定机制来解决这个问题。以下是一个常见的解决方案：

```cpp
#include <thread>
#include <mutex>

int counter = 0;
std::mutex m;

void foo()
{
 std::lock_guard g{m};
  counter++;
}

int main()
{
  // code omitted for brevity
}
```

无论哪个线程首先到达`lock_guard`都会锁定`mutex`，如下所示：

```cpp
lock mutex;             <--- thread A, B, D wait for the locked mutex 
auto res = counter;     <--- thread C has locked the mutex
counter = counter + 1;
unlock mutex;           *<--- A, B, D are blocked until C reaches here*
return res;             
```

使用锁定的问题在于性能。理论上，我们使用线程来加快程序执行，更具体地说，是数据处理。在处理大量数据的情况下，使用多个线程可能会极大地提高程序的性能。然而，在多线程环境中，我们首先要处理并发访问，因为使用多个线程访问集合可能会导致其损坏。例如，让我们看一个线程安全的堆栈实现。

# 实现线程安全的堆栈

回想一下来自第六章的栈数据结构适配器，《深入 STL 中的数据结构和算法》。我们将使用锁来实现栈的线程安全版本。栈有两个基本操作，`push`和`pop`。它们都修改容器的状态。正如您所知，栈本身不是一个容器；它是一个包装容器并提供适应接口以进行访问的适配器。我们将在一个新的类中包装`std::stack`，并加入线程安全性。除了构造和销毁函数外，`std::stack`提供以下函数：

+   `top()`: 访问栈顶元素

+   `empty()`: 如果栈为空则返回 true

+   `size()`: 返回栈的当前大小

+   `push()`: 将新项插入栈中（在顶部）

+   `emplace()`: 在栈顶就地构造一个元素

+   `pop()`: 移除栈顶元素

+   `swap()`: 与另一个栈交换内容

我们将保持简单，专注于线程安全的概念，而不是制作功能强大的完整功能栈。这里的主要关注点是修改底层数据结构的函数。我们感兴趣的是`push()`和`pop()`函数。这些函数可能在多个线程相互干扰时破坏数据结构。因此，以下声明是表示线程安全栈的类：

```cpp
template <typename T>
class safe_stack
{
public:
  safe_stack();
  safe_stack(const safe_stack& other);
  void push(T value); // we will std::move it instead of copy-referencing
  void pop();
  T& top();
  bool empty() const;

private:
  std::stack<T> wrappee_;
  mutable std::mutex mutex_;
};
```

请注意，我们将`mutex_`声明为可变的，因为我们在`empty()` const 函数中对其进行了锁定。这可能是一个比去除`empty()`的 const 性更好的设计选择。然而，您现在应该知道，对于任何数据成员使用可变性都意味着我们做出了糟糕的设计选择。无论如何，`safe_stack`的客户端代码不会太关心实现的内部细节；它甚至不知道栈使用互斥锁来同步并发访问。

现在让我们来看一下其成员函数的实现以及简短的描述。让我们从复制构造函数开始：

```cpp
safe_stack::safe_stack(const safe_stack& other)
{
  std::lock_guard<std::mutex> lock(other.mutex_);
  wrappee_ = other.wrappee_;
}
```

请注意，我们锁定了另一个栈的互斥锁。尽管这看起来不公平，但我们需要确保在复制它时，另一个栈的底层数据不会被修改。

接下来，让我们来看一下`push()`函数的实现。显然很简单；我们锁定互斥锁并将数据推入底层栈：

```cpp
void safe_stack::push(T value)
{
  std::lock_guard<std::mutex> lock(mutex_);
  // note how we std::move the value
  wrappee_.push(std::move(value));
}
```

几乎所有函数都以相同的方式包含线程同步：锁定互斥锁，执行任务，然后解锁互斥锁。这确保了一次只有一个线程访问数据。也就是说，为了保护数据免受竞态条件的影响，我们必须确保函数不变量不被破坏。

如果您不喜欢输入长的 C++类型名称，比如`std::lock_guard<std::mutex>`，可以使用`using`关键字为类型创建短别名，例如，使用`locker = std::guard<std::mutex>;`。

现在，让我们来看一下`pop()`函数，我们可以修改类声明，使`pop()`直接返回栈顶的值。我们这样做主要是因为我们不希望有人在另一个线程中访问栈顶（通过引用），然后从中弹出数据。因此，我们将修改`pop()`函数以创建一个共享对象，然后返回栈元素：

```cpp
std::shared_ptr<T> pop()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (wrappee_.empty()) {
    throw std::exception("The stack is empty");
  }
  std::shared_ptr<T> top_element{std::make_shared<T>(std::move(wrappee_.top()))};
  wrappee_.pop();
  return top_element;
}
```

请注意，`safe_stack`类的声明也应根据`pop()`函数的修改而改变。此外，我们不再需要`top()`。

# 设计无锁数据结构

如果至少有一个线程保证可以取得进展，那么我们称它是无锁函数。与基于锁的函数相比，其中一个线程可以阻塞另一个线程，它们可能都在等待某些条件才能取得进展，无锁状态确保至少一个线程取得进展。我们说使用数据同步原语的算法和数据结构是阻塞的，也就是说，线程被挂起，直到另一个线程执行操作。这意味着线程在解除阻塞之前无法取得进展（通常是解锁互斥锁）。我们感兴趣的是不使用阻塞函数的数据结构和算法。我们称其中一些为无锁，尽管我们应该区分非阻塞算法和数据结构的类型。

# 使用原子类型

在本章的前面，我们介绍了源代码行之间的间隙是数据竞争的原因。每当您有一个由多个指令组成的操作时，您的大脑都应该警惕可能出现的问题。然而，无论您多么努力使操作独立和单一，大多数情况下，您都无法在不将操作分解为涉及多个指令的步骤的情况下取得任何成果。C++通过提供原子类型来拯救我们。

首先，让我们了解为什么使用原子这个词。一般来说，我们理解原子是指不能分解成更小部分的东西。也就是说，原子操作是一个无法半途而废的操作：要么完成了，要么没有。原子操作的一个例子可能是对整数的简单赋值：

```cpp
num = 37;
```

如果两个线程访问这行代码，它们都不可能遇到它是半成品的情况。换句话说，赋值之间没有间隙。当然，如果`num`表示具有用户定义赋值运算符的复杂对象，同一语句可能会有很多间隙。

原子操作是不可分割的操作。

另一方面，非原子操作可能被视为半成品。经典的例子是我们之前讨论过的增量操作。在 C++中，对原子类型的所有操作也是原子的。这意味着我们可以通过使用原子类型来避免行之间的间隙。在使用原子操作之前，我们可以通过使用互斥锁来创建原子操作。例如，我们可能会考虑以下函数是原子的：

```cpp
void foo()
{
  mutex.lock();
  int a{41};
  int b{a + 1};
  mutex.unlock();
}
```

真正的原子操作和我们刚刚制作的假操作之间的区别在于原子操作不需要锁。这实际上是一个很大的区别，因为诸如互斥锁之类的同步机制会带来开销和性能惩罚。更准确地说，原子类型利用低级机制来确保指令的独立和原子执行。标准原子类型在`<atomic>`头文件中定义。然而，标准原子类型可能也使用内部锁。为了确保它们不使用内部锁，标准库中的所有原子类型都公开了`is_lock_free()`函数。

唯一没有`is_lock_free()`成员函数的原子类型是`std::atomic_flag`。对这种类型的操作要求是无锁的。它是一个布尔标志，大多数情况下被用作实现其他无锁类型的基础。

也就是说，如果`obj.is_lock_free()`返回`true`，则表示对`obj`的操作是直接使用原子指令完成的。如果返回 false，则表示使用了内部锁。更重要的是，`static constexpr`函数`is_always_lock_free()`在所有支持的硬件上返回`true`，如果原子类型始终是无锁的。由于该函数是`constexpr`，它允许我们在编译时定义类型是否是无锁的。这是一个重大进步，以良好的方式影响代码的组织和执行。例如，`std::atomic<int>::is_always_lock_free()`返回`true`，因为`std::atomic<int>`很可能始终是无锁的。

在希腊语中，a 意味着不，tomo 意味着切。原子一词源自希腊语 atomos，意思是不可分割的。也就是说，原子意味着不可分割的最小单位。我们使用原子类型和操作来避免指令之间的间隙。

我们使用原子类型的特化，例如 `std::atomic<long>`；但是，您可以参考以下表格以获取更方便的原子类型名称。表格的左列包含原子类型，右列包含其特化：

| **原子类型** | **特化** |
| --- | --- |
| `atomic_bool` | `std::atomic<bool>` |
| `atomic_char` | `std::atomic<char>` |
| `atomic_schar` | `std::atomic<signed char>` |
| `atomic_uchar` | `std::atomic<unsigned char>` |
| `atomic_int` | `std::atomic<int>` |
| `atomic_uint` | `std::atomic<unsigned>` |
| `atomic_short` | `std::atomic<short>` |
| `atomic_ushort` | `std::atomic<unsigned short>` |
| `atomic_long` | `std::atomic<long>` |
| `atomic_ulong` | `std::atomic<unsigned long>` |
| `atomic_llong` | `std::atomic<long long>` |
| `atomic_ullong` | `std::atomic<unsigned long long>` |
| `atomic_char16_t` | `std::atomic<char16_t>` |
| `atomic_char32_t` | `std::atomic<char32_t>` |
| `atomic_wchar_t` | `std::atomic<wchar_t>` |

上表代表了基本的原子类型。常规类型和原子类型之间的根本区别在于我们可以对它们应用的操作类型。现在让我们更详细地讨论原子操作。

# 原子类型的操作

回想一下我们在前一节讨论的间隙。原子类型的目标是要么消除指令之间的间隙，要么提供将多个指令组合在一起作为单个指令执行的操作。以下是原子类型的操作：

+   `load()`

+   `store()`

+   `exchange()`

+   `compare_exchange_weak()`

+   `compare_exchange_strong()`

+   `wait()`

+   `notify_one()`

+   `notify_all()`

`load()` 操作原子地加载并返回原子变量的值。`store()` 原子地用提供的非原子参数替换原子变量的值。

`load()` 和 `store()` 与非原子变量的常规读取和赋值操作类似。每当我们访问对象的值时，我们执行一个读取指令。例如，以下代码打印了 `double` 变量的内容：

```cpp
double d{4.2}; // "store" 4.2 into "d"
std::cout << d; // "read" the contents of "d"
```

对于原子类型，类似的读取操作转换为：

```cpp
atomic_int m;
m.store(42);             // atomically "store" the value
std::cout << m.load();   // atomically "read" the contents 
```

尽管上述代码没有实际意义，但我们包含了这个例子来表示对待原子类型的不同方式。应该通过原子操作来访问原子变量。以下代码表示了 `load()`、`store()` 和 `exchange()` 函数的定义： 

```cpp
T load(std::memory_order order = std::memory_order_seq_cst) const noexcept;
void store(T value, std::memory_order order = 
            std::memory_order_seq_cst) noexcept;
T exchange(T value, std::memory_order order = 
            std::memory_order_seq_cst) noexcept;
```

正如您所见，还有一个名为 `order` 的额外参数，类型为 `std::memory_order`。我们很快会对它进行描述。`exchange()` 函数以一种方式包含了 `store()` 和 `load()` 函数，以便原子地用提供的参数替换值，并原子地获取先前的值。

`compare_exchange_weak()` 和 `compare_exchange_strong()` 函数的工作方式相似。它们的定义如下：

```cpp
bool compare_exchange_weak(T& expected_value, T target_value, 
                           std::memory_order order = 
                            std::memory_order_seq_cst) noexcept;
bool compare_exchange_strong(T& expected_value, T target_value,
                            std::memory_order order =
                             std::memory_order_seq_cst) noexcept;
```

它们将第一个参数（`expected_value`）与原子变量进行比较，如果它们相等，则用第二个参数（`target_value`）替换变量。否则，它们会原子地将值加载到第一个参数中（这就是为什么它是通过引用传递的）。弱交换和强交换之间的区别在于 `compare_exchange_weak()` 允许出现错误（称为**虚假失败**），也就是说，即使 `expected_value` 等于底层值，该函数也会将它们视为不相等。这是因为在某些平台上，这会提高性能。

自 C++20 以来，已添加了`wait()`、`notify_one()`和`notify_all()`函数。`wait()`函数阻塞线程，直到原子对象的值修改。它接受一个参数与原子对象的值进行比较。如果值相等，它会阻塞线程。要手动解除线程阻塞，我们可以调用`notify_one()`或`notify_all()`。它们之间的区别在于`notify_one()`解除至少一个被阻塞的操作，而`notify_all()`解除所有这样的操作。

现在，让我们讨论我们在先前声明的原子类型成员函数中遇到的内存顺序。`std::memory_order`定义了原子操作周围的内存访问顺序。当多个线程同时读取和写入变量时，一个线程可以按照与另一个线程存储它们的顺序不同的顺序读取更改。原子操作的默认顺序是顺序一致的顺序 - 这就是`std::memory_order_seq_cst`的作用。有几种类型的顺序，包括`memory_order_relaxed`、`memory_order_consume`、`memory_order_acquire`、`memory_order_release`、`memory_order_acq_rel`和`memory_order_seq_cst`。在下一节中，我们将设计一个使用默认内存顺序的原子类型的无锁堆栈。

# 设计无锁堆栈

设计堆栈时要牢记的关键事项之一是确保从另一个线程返回的推送值是安全的。同样重要的是确保只有一个线程返回一个值。

在前面的章节中，我们实现了一个基于锁的堆栈，它包装了`std::stack`。我们知道堆栈不是一个真正的数据结构，而是一个适配器。通常，在实现堆栈时，我们选择向量或链表作为其基础数据结构。让我们看一个基于链表的无锁堆栈的例子。将新元素推入堆栈涉及创建一个新的列表节点，将其`next`指针设置为当前的`head`节点，然后将`head`节点设置为指向新插入的节点。

如果您对头指针或下一个指针这些术语感到困惑，请重新阅读第六章《深入 STL 中的数据结构和算法》，在那里我们详细讨论了链表。

在单线程环境中，上述步骤是可以的；但是，如果有多个线程修改堆栈，我们应该开始担心。让我们找出`push()`操作的陷阱。当将新元素推入堆栈时，发生了三个主要步骤：

1.  `node* new_elem = new node(data);`

1.  `new_elem->next = head_;`

1.  `head_ = new_elem;`

在第一步中，我们声明将插入到基础链表中的新节点。第二步描述了我们将其插入到列表的前面 - 这就是为什么新节点的`next`指针指向`head_`。最后，由于`head_`指针表示列表的起始点，我们应该重置其值以指向新添加的节点，就像第 3 步中所做的那样。

节点类型是我们在堆栈中用于表示列表节点的内部结构。以下是它的定义：

```cpp
template <typename T>
class lock_free_stack
{
private:
 struct node {
 T data;
 node* next;
 node(const T& d) : data(d) {}
 }  node* head_;
// the rest of the body is omitted for brevity
};
```

我们建议您首先查找代码中的空白 - 不是在前面的代码中，而是在我们描述将新元素推入堆栈时的步骤中。仔细看看。想象两个线程同时添加节点。一个线程在第 2 步中将新元素的下一个指针设置为指向`head_`。另一个线程使`head_`指向另一个新元素。很明显，这可能导致数据损坏。对于线程来说，在步骤 2 和 3 中有相同的`head_`是至关重要的。为了解决步骤 2 和 3 之间的竞争条件，我们应该使用原子比较/交换操作来保证在读取其值之前`head_`没有被修改。由于我们需要以原子方式访问头指针，这是我们如何修改`lock_free_stack`类中的`head_`成员的方式：

```cpp
template <typename T>
class lock_free_stack
{
private:
  // code omitted for brevity
 std::atomic<node*> head_;  // code omitted for brevity
};
```

这是我们如何在原子`head_`指针周围实现无锁`push（）`的方式：

```cpp
void push(const T& data)
{
  node* new_elem = new node(data);
  new_elem->next = head_.load();
  while (!head_.compare_exchange_weak(new_elem->next, new_elem));
}
```

我们使用`compare_exchange_weak（）`来确保`head_`指针的值与我们存储在`new_elem->next`中的值相同。如果是，我们将其设置为`new_elem`。一旦`compare_exchange_weak（）`成功，我们就可以确定节点已成功插入到列表中。

看看我们如何使用原子操作访问节点。类型为`T`的指针的原子形式-`std::atomic<T*>`-提供相同的接口。除此之外，`std::atomic<T*>`还提供指针的算术操作`fetch_add（）`和`fetch_sub（）`。它们对存储的地址进行原子加法和减法。这是一个例子：

```cpp
struct some_struct {};
any arr[10];
std::atomic<some_struct*> ap(arr);
some_struct* old = ap.fetch_add(2);
// now old is equal to arr
// ap.load() is equal to &arr[2]
```

我们故意将指针命名为`old`，因为`fetch_add（）`将数字添加到指针的地址并返回`old`值。这就是为什么`old`指向与`arr`指向的相同地址。

在下一节中，我们将介绍更多可用于原子类型的操作。现在，让我们回到我们的无锁栈。要`pop（）`一个元素，也就是移除一个节点，我们需要读取`head_`并将其设置为`head_`的下一个元素，如下所示：

```cpp
void pop(T& popped_element)
{
  node* old_head = head_;
  popped_element = old_head->data;
  head_ = head_->next;
  delete old_head;
}
```

现在，好好看看前面的代码。想象几个线程同时执行它。如果两个从堆栈中移除项目的线程读取相同的`head_`值会怎样？这和其他一些竞争条件导致我们采用以下实现：

```cpp
void pop(T& popped_element)
{
  node* old_head = head_.load();
  while (!head_.compare_exchange_weak(old_head, old_head->next));
  popped_element = old_head->data;
}
```

我们在前面的代码中几乎应用了与`push（）`函数相同的逻辑。前面的代码并不完美；它应该得到改进。我们建议您努力修改它以消除内存泄漏。

我们已经看到，无锁实现严重依赖于原子类型和操作。我们在上一节讨论的操作并不是最终的。现在让我们发现一些更多的原子操作。

# 原子操作的更多操作

在上一节中，我们在用户定义类型的指针上使用了`std::atomic<>`。也就是说，我们为列表节点声明了以下结构：

```cpp
// the node struct is internal to 
// the lock_free_stack class defined above
struct node
{
  T data;
  node* next;
};
```

节点结构是用户定义的类型。尽管在上一节中我们实例化了`std::atomic<node*>`，但以同样的方式，我们几乎可以为任何用户定义的类型实例化`std::atomic<>`，也就是`std::atomic<T>`。但是，您应该注意`std::atomic<T>`的接口仅限于以下函数：

+   `load（）`

+   `store（）`

+   `exchange（）`

+   `compare_exchange_weak（）`

+   `compare_exchange_strong（）`

+   `wait（）`

+   `notify_one（）`

+   `notify_all（）`

现在让我们根据底层类型的特定情况来查看原子类型上可用的操作的完整列表。

`std::atomic<>`与整数类型（如整数或指针）实例化具有以下操作，以及我们之前列出的操作：

+   `fetch_add（）`

+   `fetch_sub（）`

+   `fetch_or（）`

+   `fetch_and（）`

+   `fetch_xor（）`

此外，除了增量（`++`）和减量（`--`）之外，还有以下运算符可用：`+=`，`-=`，`|=`，`&=`和`^=`。

最后，有一种特殊的原子类型称为`atomic_flag`，具有两种可用操作：

+   `clear（）`

+   `test_and_set（）`

您应该将`std::atomic_flag`视为具有原子操作的位。`clear（）`函数将其清除，而`test_and_set（）`将值更改为`true`并返回先前的值。

# 总结

在本章中，我们介绍了一个相当简单的设计堆栈的例子。还有更复杂的例子可以研究和遵循。当我们讨论设计并发堆栈时，我们看了两个版本，其中一个代表无锁堆栈。与基于锁的解决方案相比，无锁数据结构和算法是程序员的最终目标，因为它们提供了避免数据竞争的机制，甚至无需同步资源。

我们还介绍了原子类型和操作，您可以在项目中使用它们来确保指令是不可分割的。正如您已经知道的那样，如果一条指令是原子的，就不需要担心它的同步。我们强烈建议您继续研究这个主题，并构建更健壮和复杂的无锁数据结构。在下一章中，我们将看到如何设计面向世界的应用程序。

# 问题

1.  在多线程单例实现中为什么要检查实例两次？

1.  在基于锁的栈的复制构造函数的实现中，为什么要锁定另一个栈的互斥量？

1.  原子类型和原子操作是什么？

1.  为什么对原子类型使用`load()`和`store()`？

1.  `std::atomic<T*>`支持哪些额外操作？

# 进一步阅读

+   《Atul Khot 的并发模式与最佳实践》，网址为[`www.packtpub.com/application-development/concurrent-patterns-and-best-practices`](https://www.packtpub.com/application-development/concurrent-patterns-and-best-practices)

+   《Maya Posch 的 C++多线程编程》,网址为[`www.packtpub.com/application-development/mastering-c-multithreading`](https://www.packtpub.com/application-development/mastering-c-multithreading)
