# 9

# 非典型分配机制

我们在使用 C++ 进行内存管理方面的探索中取得了进展。在*第七章*中，我们探讨了可以通过哪些语法方式来重载 `operator new()` 和 `operator delete()`（以及它们的数组对应物），而在*第八章*中，我们编写了一个实际的真实例子（一个内存泄漏检测器），这个例子依赖于编写这样的重载的能力。这是一个很好的开始，具体地展示了这些知识有实际的应用，但你可能会（正确地）想知道在控制内存管理功能时我们还能做些什么。

本章将与其他章节略有不同。在这里，我们将展示一系列非详尽的方法，说明如何通过控制 C++ 的内存分配函数来受益。更确切地说，我们将展示以下内容：

+   如何使用 placement `new` 高效地驱动内存映射硬件

+   如何通过 `operator new()` 的 `nothrow` 版本简化错误管理

+   如何安装和使用 `std::new_handler` 来使处理内存不足情况变得更加容易

+   如何通过标准 C++ 的中介处理“奇特”的内存，如共享内存或持久内存

在本章结束时，我们将对 C++ 基本内存分配功能为我们提供的机遇有一个更广阔的视角。后续章节将回到更具体的话题，例如基于区域的分配(*第十章*)、延迟回收(*第十一章*)，以及在后续章节中，如何使用容器和分配器来控制内存分配。

# 技术要求

你可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter9`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter9)。

# Placement new 和内存映射硬件

placement `new`（如你可能记得，在第 *7 章* 中讨论的一个重要特性）有许多用途，但其中一个特别有趣的使用是它允许我们将软件对象映射到内存映射硬件，从而有效地允许我们像操作软件一样驱动硬件。

要编写一个这个特性的工作示例会相当棘手，因为我们可能会发现自己处于“非可移植代码地带”，使用操作系统特定的功能来获取特定设备的地址，并讨论获取通常由软件驱动程序访问的内存位置的读写权限的方法。因此，我们将构建一个人工但具有说明性的示例，并要求您，尊敬的读者，想象这个示例中缺失的部分。

首先，假设我们正在开发一个新显卡的驱动程序，这款显卡非常出色，其代号为`super_video_card`。为了说明这一点，我们将通过以下类来模拟：

```cpp
#include <cstdint>
class super_video_card {
  // ...
public:
  // super duper registers
  volatile std::uint32_t r0{}, r1{}, r2{}, r3{};
  static_assert(sizeof(float) == 4); // sanity check
  volatile float f0{}, f1{}, f2{}, f3{};
  // etc.
  // initialize the video card's state
  super_video_card() = default;
  super_video_card(const super_video_card&) = delete;
  super_video_card&
    operator=(const super_video_card&) = delete;
  // could be used to reset the video card's state
  ~super_video_card() = default;
  // various services (omitted for brevity)
};
// ...
```

对于我们的目的，这个类的重要方面如下：

+   它是一个不可复制的类型，因为它旨在映射到特定的内存区域。复制此类对象至少是无效的。

+   它被设计成这样的方式，其状态在概念上可以叠加到其硬件等价物上。例如，给定前面的类声明，从硬件内存布局的开始处开始，我们期望有四个 32 位整数寄存器，然后是四个 32 位浮点寄存器。我们使用了`<cstdint>`来获取我们编译器上固定宽度整数类型的别名。

+   在这种情况下，我们应该通过`static_assert`来表述我们的期望。此外，由于硬件寄存器的状态可以通过我们程序之外的其他操作而改变，我们将寄存器等价物标记为`volatile`，这样对这些成员变量的访问将等同于 C++抽象机中的 I/O 操作。

为什么我们在这个例子中使用`volatile`变量？

如果你不太习惯`volatile`变量，你可能会想知道为什么我们在内存映射硬件表示类的数据成员上使用了这个限定符。这样做之所以重要，是因为我们希望避免编译器基于（在这种情况下是错误的）假设来优化代码，即如果我们的代码没有触摸这些变量，那么它们的状态不会改变，或者如果我们的代码中对这些变量的写入没有跟随读取，那么可以假设没有效果。通过`volatile`限定的变量，我们实际上在告诉编译器“*这里有一些你不知道的事情在这些对象上发生，所以请不要假设太多*。”

为了简单起见，我们使用了一个将数据成员清零的构造函数和一个平凡的析构函数，但在实践中，我们本可以使用构造函数（默认或其他）来初始化内存映射设备的状态以符合我们的需求，并使用析构函数将设备状态重置为某种可接受的状态。

通常，为了程序能够访问内存映射的硬件，我们可能会通过操作系统提供的服务与操作系统通信，这些服务接受作为参数的所需信息来识别我们寻求地址的设备。在我们的情况下，我们将简单地让它看起来我们可以访问一个正确大小和对齐的内存区域，我们可以从中读取和写入。内存地址以原始内存（`void*`类型）的形式暴露出来，这是在类似情况下我们可以从操作系统函数中合理期望的：

```cpp
// somewhere in memory where we have read / write
// access privileges is a memory-mapped hardware
// that corresponds to the actual device
alignas(super_video_card) char
  mem_mapped_device[sizeof(super_video_card)];
void* get_super_card_address() {
  return mem_mapped_device;
}
// ...
```

然后，我们到达了如何使用放置`new`将对象映射到某些内存映射硬件位置的方法。请注意，我们需要包含`<new>`头文件，因为这是放置`new`定义的地方。达到我们目标的方法如下：

1.  首先，获取我们想要映射我们精心制作的`super_video_card`对象的地址。

1.  然后，通过在该地址的放置`new`，构造一个`super_video_card`对象，使得该对象的数据成员对应于它们所代表的寄存器的地址。

1.  在该对象的生命周期内，通过相应的指针（以下代码摘录中的`the_card`变量）使用该对象。

1.  当我们完成工作后，我们最不想做的事情就是在`the_card`上应用`operator delete()`，因为我们一开始就没有分配相关的内存。然而，我们确实希望通过`~super_video_card()`来最终化这个对象，以确保运行该对象的清理或重置代码（如果有的话）。

因此，我们得到了以下结果：

```cpp
// ...
#include <new>
int main() {
  // map our object to the hardware
  void* p = get_super_card_address();
  auto the_card =
      new(p) super_video_card{ /* args */ };
  // through pointer the_card, use the actual memory-
  // mapped hardware
  // ...
  the_card->~super_video_card();
}
```

如果显式析构函数调用是一个问题，例如在可能抛出异常的代码中，我们可以使用一个带有自定义删除器的`std::unique_ptr`对象来最终化`super_video_card`对象（参见*第五章*）：

```cpp
// ...
#include <new>
#include <memory>
int main() {
  // map our object to the hardware
  void* p = get_super_card_address();
  std::unique_ptr<
      super_video_card,
      decltype([](super_video_card *p) {
        p->~super_video_card(); // do not call delete p!
      })
  > the_card {
      new(p) super_video_card{ /* args */ }
};
  // through pointer the_card, use the actual memory-
  // mapped hardware
   // ...
   // implicit call to the_card->~super_video_card()
}
```

在这种情况下，`std::unique_ptr`对象最终化了指针（即`super_video_card`对象），但没有释放其内存存储，这使得在`the_card`变量生命周期中存在异常时，代码更加健壮。

# 简化 nothrow new 的使用

如*第七章*所述，当`operator new()`无法执行分配请求时，其默认行为是抛出异常。这可能是由于内存不足或其他无法满足分配请求的情况，在这种情况下，通常抛出`std::bad_alloc`；由于数组长度不正确（例如，一个长度为负的一维数组超过了实现定义的限制），通常会导致抛出`std::bad_array_new_length`；或者由于在`operator new()`完成后未能完成对象的后续构造，在这种情况下，将被抛出的异常将是来自失败构造函数的任何异常。

异常是 C++函数表示未能满足函数后置条件的“正常”方式。在某些情况下，例如构造函数或重载运算符，这是唯一真正可行的方式：构造函数没有返回值，并且重载运算符的函数签名通常没有为额外的参数或错误报告返回值留下空间，尽管对于某些类型（如`std::optional`或`std::expected`）可以提出一些重载运算符使用情况的替代方案。

当然，有些领域通常不使用异常：例如，许多视频游戏在没有异常支持的情况下编译，同样，为嵌入式系统编写的许多程序也是如此。提出的原因从技术上的（对内存空间消耗、执行速度或两者都视为不希望的开销的恐惧）到更哲学上的（不喜欢被视为隐藏的控制路径），但无论原因是什么，事实是，没有异常支持的 C++代码确实存在，`nothrow`版本的`operator new()`是一个现实。

当然，这也意味着即使是看似简单的代码，如以下所示，也可能导致**未定义行为**（UB）：

```cpp
#include <new>
#include <iostream>
struct X {
  int n;
  X(int n) : n { n } { }
};
int main() {
  auto p = new (std::nothrow) X{ 3 };
  std::cout << p->n; // <-- HERE
  delete p;
}
```

这种潜在的不确定行为（UB）的原因是，如果`operator new()`的`nothrow`版本失败（虽然不太可能，但并非不可能，尤其是在内存受限的情况下），那么`p`将会是空指针，通过`p`访问`n`数据成员将是一个非常糟糕的想法。

当然，解决方案很简单，鉴于你是一位敏锐的读者，你可能已经注意到了：在使用它之前先测试指针！当然，这在这里是有效的，如下所示：

```cpp
#include <new>
#include <iostream>
struct X {
  int n;
  X(int n) : n { n } { }
};
int main() {
  auto p = new (std::nothrow) X{ 3 };
  if(p) {
      std::cout << p->n; // ...use *p as needed...
  }
  delete p; // fine even in p is null
}
```

这种方法的缺点是代码很快就会充满测试，因为程序中很少只有一个指针，这提醒我们，使用异常的代码之美在于不需要担心这些测试。使用异常，要么`operator new()`和随后的构造都成功了，可以自信地使用结果指针，要么这些步骤中有一个失败了，代码执行没有达到可能陷入麻烦的点：

```cpp
#include <new>
#include <iostream>
struct X {
  int n;
  X(int n) : n { n } { }
};
int main() {
  auto p = new X{ 3 }; // throws if operator new() or
                        // X::X(int) fails
  std::cout << p->n; // ...use *p as needed...
  delete p;
}
```

当然，即使有异常，也可能遇到麻烦，例如，如果存在一个执行路径让`p`保持为空或未初始化，而其他路径则不会发生这种情况（你通常可以通过在声明时初始化对象来避免这种情况，但这并不总是可能的）；让我们现在暂时将这些代码卫生考虑放在一边，因为它们会偏离我们感兴趣的主题。

面对分配失败的情况时，一个重要的考虑是当它发生时应该做什么。无论我们的代码库是否使用异常，我们很可能不希望程序执行继续，从而通过诸如不正确使用空指针之类的操作导致未定义行为（UB）。

在失败分配点停止执行的一种常见方法是在某些代码结构中包装尝试分配和构造操作、对结果指针的后续测试以及如果指针为空要采取的行动。我们想要包装的代码可能如下所示，假设我们想要分配并构造一个`int`对象：

```cpp
// ...
int *p = new int{ 3 };
if(!p) std::abort(); // for example
return p;
// ...
```

此代码使用`std::abort()`作为结束程序执行的机制；异常会给我们提供可能可恢复的错误，但没有异常，我们可用的大多数标准机制都会导致程序终止，在这种情况下，`std::abort()`是一个合理的选择。

结束程序执行的方式

C++程序可以以许多不同的方式结束：到达`main()`函数的末尾是最明显的一种，但还有其他例子。例如，`std::exit()`用于带有清理步骤的正常程序终止；`std::quick_exit()`用于不带清理步骤的程序终止。可以使用`std::atexit()`和`std::at_quick_exit()`注册一些在退出前要调用的函数，而`std::abort()`用于在没有清理步骤的情况下发出程序异常终止的信号。当在文档列表中的某些不愉快情况发生时（这个列表包括从`static`变量的构造函数或`noexcept`函数体中抛出的异常等情况），使用`std::terminate()`函数。在我们的情况下，唯一真正适合的机制是`std::abort()`。

解决此问题的一个可能方法是使用一个宏和一个立即调用的函数表达式（**IIFE**），这是对一个匿名 lambda 表达式所构成的、立即创建、执行和丢弃的表达式的称呼。为了使我们的解决方案通用，我们需要能够做到以下几步：

+   指定要创建的对象类型

+   使宏可变参数，因为我们需要能够将任何类型和数量的参数传递给对象的构造函数

这样一个宏的可能实现是`TRY_NEW`，如下所示：

```cpp
#include <new>
#include <cstdlib>
#define TRY_NEW(T,...) [&] { \
  auto p = new (std::nothrow) T(__VA_ARGS__); \
  if(!p) std::abort(); \
  return p; \
}()
struct dies_when_newed {
  void* operator new(std::size_t, std::nothrow_t) {
      return {};
  }
};
int main() {
  // p0 is int*, points to an int{ 0 }
  auto p0 = TRY_NEW(int);
  // p1 is int*, points to an int{ 3 }
  auto p1 = TRY_NEW(int, 3);
  auto q = TRY_NEW(dies_when_newed); // calls abort()
}
```

并非每个人都熟悉可变参数宏，所以让我们一步一步来：

+   我们宏的“签名”是`TRY_NEW(T,...)`，这意味着`T`是必需的，而`...`可以是任何数量的标记（包括一个都没有），由逗号分隔。不出所料，我们将使用`T`来表示要构造的类型，而`...`用于传递给将被调用的构造函数的参数。

+   由于我们为了可读性将宏写在了多行上，除了最后一行外，每一行都以一个空格后跟一个反斜杠结束，以通知预处理器它应该在下一行继续解析。

+   `...`上的符号通过名为`__VA_ARGS__`的特殊宏进行中继，该宏展开为`...`包含的内容，如果`...`本身为空，则可以是空的。这在 C 和 C++中都有效。请注意，我们在构造函数调用中使用括号而不是花括号，因为我们想避免无意中构建一个初始化列表，如果`__VA_ARGS__`的所有元素都是同一类型的话。

+   我们测试由调用`operator new()`的`std::nothrow`版本产生的`p`指针，如果`p`为空，则调用`std::abort()`。

+   如前所述，整个操作序列被一个立即执行函数表达式（IIFE）包裹，并返回新分配的指针。请注意，如果我们愿意，我们也可以从那个 lambda 表达式返回一个`std::unique_ptr<T>`对象。另外，请注意，这个 lambda 表达式使用了一个`[&]`捕获块来确保在 lambda 的作用域内`__VA_ARGS__`中的标记可用。

一个小但有趣的影响

注意，由于我们使用了括号（同样适用于花括号），一个空的`__VAR_ARGS__`将导致这个宏将基本类型（如`int`）初始化为零，而不是将它们留作未初始化。您可以比较：截至 C++23，`new int;`产生一个指向未初始化`int`对象的指针，但`new int();`和`new int{};`都将分配的块初始化为零。这有一个优点，就像这个宏一样，即使对于平凡类型，我们也不会得到一个指向未初始化对象的指针。然而，也有一个缺点，因为我们甚至在不必要的情况下也要为初始化付费。

另一种方法可能是使用变长参数函数模板，这在实践中可能会带来更好的调试体验。它的客户端代码看起来略有不同，但在使用和效果上与其他类似：

```cpp
#include <new>
#include <cstdlib>
#include <utility>
template <class T, class ... Args>
  auto try_new(Args &&... args) {
      auto p =
        new (std::nothrow) T(std::forward<Args>(args)...);
      if(!p) std::abort();
      return p;
  }
struct dies_when_newed {
  void* operator new(std::size_t, std::nothrow_t) {
      return {};
  }
};
int main() {
  // p0 is int*, points to an int{ 0 }
  auto p0 = try_new<int>();
  // p1 is int*, points to an int{ 3 }
  auto p1 = try_new<int>(3);
  auto q = try_new<dies_when_newed>(); // calls abort()
}
```

可变参数函数版本的调用语法看起来像是一个类型转换，传递给`try_new()`的参数被完美转发到`T`的构造函数中，以确保最终调用预期的构造函数。就像宏的情况一样，我们可以选择用这个函数返回一个`std::unique_ptr<T>`对象，而不是`T*`对象。

# 内存不足的情况和 new_handler

到目前为止，包括本章在内，我们已声明`operator new()`和`operator new[]()`在无法分配内存时通常会抛出`std::bad_alloc`异常。这在很大程度上是正确的，但我们之前避免了一个细微之处，现在我们将花些时间和精力来关注它。

想象一种情况，用户代码已经专门化了内存分配函数，以便从具有有趣性能特性的预分配数据结构中获取内存块。假设这个数据结构最初为少量块分配空间，然后在用户代码耗尽初始分配的块之后继续分配更多空间。换句话说：在这种情况下，我们有一个初始的快速设置（让我们称它为“乐观”状态）和一个次要设置（让我们称它为“第二次机会”状态），允许用户代码在“乐观”状态的资源耗尽后继续分配。

为了使此类场景无缝，在不显式干预用户代码的情况下实现透明的分配策略更改，仅显式抛出`std::bad_alloc`是不够的。抛出会完成`operator new()`的执行，客户端代码可以捕获异常并采取行动，当然，但在这种（合理的）场景中，我们希望分配失败导致采取某些行动，并且`operator new()`在更新后的状态（如果有的话）下再次尝试。

在 C++中，此类场景通过`std::new_handler`来处理，它是类型为`void(*)()`的函数指针的别名。需要了解的是以下内容：

+   程序中有一个全局的`std::new_handler`，默认情况下其值为`nullptr`。

+   可以通过`std::set_new_handler()`函数设置活动的`std::new_handler`，并且可以通过`std::get_new_handler()`函数获取活动的`std::new_handler`。请注意，为了方便起见，`std::set_new_handler()`返回正在被替换的`std::new_handler`。

+   当一个分配函数，如`operator new()`失败时，它应该首先获取活动的`std::new_handler`。如果该指针为空，则分配函数应该抛出`std::bad_alloc`，就像我们迄今为止所做的那样；否则，它应该调用该`std::new_handler`并在新条件下重试。

如预期的那样，你的标准库应该已经实现了这个算法，但我们的`operator new()`和`operator new[]()`的重载函数还没有这样做，至少到目前为止是这样。为了展示如何从`std::new_handler`中受益，我们现在将实现上述两步场景的人工版本。

这个玩具实现将使用某些`X`类型的分配操作符的成员版本，并表现得好像我们最初有足够内存来存储该类型的`limit`个对象（通常，我们实际上会管理这些内存，你可以在*第十章*中看到一个这样的管理示例，我们将提供一个更现实的例子）。我们将安装一个`std::new_handler`，当被调用时，将`limit`改为一个更大的数字，然后重置活动处理程序为`nullptr`，这样后续尝试分配`X`对象失败将导致抛出`std::bad_alloc`：

```cpp
#include <new>
#include <vector>
#include <iostream>
struct X {
  // toy example, not thread-safe
  static inline int limit = 5;
  void* operator new(std::size_t n) {
      std::cout << "X::operator new() called with "
                << limit << " blocks left\n";
      while (limit <= 0) {
        if (auto hdl = std::get_new_handler(); hdl)
            hdl();
        else
            throw std::bad_alloc{};
      }
      --limit;
      return ::operator new(n);
  }
  void operator delete(void* p) {
      std::cout << "X::operator delete()\n";
      ::operator delete(p);
  }
  // same for the array versions
};
int main() {
  std::set_new_handler([]() noexcept {
      std::cout << "allocation failure, "
                   "fetching more memory\n";
      X::limit = 10;
      std::set_new_handler(nullptr); // as per default
  });
  std::vector<X*> v;
  v.reserve(100);
  try {
      for (int i = 0; i != 10; ++i)
         v.emplace_back(new X);
  } catch(...) {
      // this will never be reached with this program
      std::cerr << "out of memory\n";
  }
  for (auto p : v) delete p;
}
```

注意`X::operator new()`处理失败的方式：如果它注意到它将无法满足其后续条件，它会获取活动的`std::new_handler`，如果它不为空，则在再次尝试之前调用它。这意味着当`std::new_handler`被调用时，它必须以某种方式改变情况，使得后续的尝试分配可以成功，或者将`std::new_handler`改为`nullptr`，这样失败将导致抛出异常。不遵守这些规则可能导致无限循环，并随之而来的是许多悲伤。

在 `main()` 中为这个玩具示例安装的处理程序执行以下操作：当被调用时，它改变执行分配的条件（它增加了 `X::limit` 的值）。然后，它使用 `nullptr` 调用 `std::set_new_handler()`，因为我们没有计划在“乐观”和“第二次机会”情况之后采取另一种方法，所以如果我们耗尽了第二次机会的资源，我们（正如他们所说）就完蛋了。

lambda 作为 new_handler？

您可能已经注意到，我们将 `std::new_handler` 类型描述为 `void(*)()` 类型函数指针的别名，然而在我们的玩具示例中，我们安装了一个 lambda。为什么这行得通？好吧，碰巧无状态的 lambda——一个空的捕获块的 lambda 表达式——可以隐式转换为具有相同调用签名的函数指针。这在许多情况下都是很有用的，比如当编写与 C 代码或操作系统 API 交互的 C++ 代码时。

我们现在即将进入本章的一个奇怪且相当技术性的部分，我们将看到如何利用 C++ 来处理非典型内存。

# 标准 C++和奇异内存

在这个有点奇怪的章节的最后，我们关注的是我们可以如何编写处理“奇异”内存的标准 C++ 程序。通过“奇异”，我们指的是需要显式操作来“接触”（分配、读取、写入、释放等）的内存，并且与我们的程序控制的“正常”内存块不同，例如本章前面使用 placement `new` 的内存映射使用示例。这类内存的例子包括持久（非易失）内存或共享内存，但实际上任何 *非同寻常的* 都可以。

由于我们必须选择一个示例，我们将编写一个使用（虚构的）共享内存块的示例。

一个小小的谎言……

重要的是要理解，我们正在描述的是一个通常会在 *进程* 之间共享的内存机制，但进程间通信是操作系统的领域。标准 C++ 只描述了在进程中的 *线程* 之间共享数据的规则；因此，我们将说一个小小的谎言，并编写一个多线程系统，而不是多进程系统，使用该内存来共享数据。我们的重点是内存管理功能，而不是进程间通信，所以这不应该构成问题。

按照本章前面部分的做法，我们将构建一个可移植的示例，展示如何在代码中管理非典型内存，并让您将这些细节映射到您选择平台的服务。我们的示例代码将具有以下形式：

+   将分配一个共享内存块。我们将让它看起来这个内存是特殊的，因为需要特殊的操作系统函数来创建它、分配它或释放它，但我们故意避免使用实际的操作系统函数。这意味着，如果您想将本节中的代码用于实际应用，您需要将其适配到您选择的平台 API。

+   我们将制作一个使用这个虚构的共享内存 API 的“手工”版本玩具程序，以此来展示在这些情况下用户代码会是什么样子。

+   然后，我们将展示理解 C++的内存管理功能如何帮助我们编写更愉快且“看起来更正常”的用户代码，这些代码与“手工”代码做同样的事情……甚至更好。

虚构的现实感？

在本节中，我们将讨论 C++和异构内存，希望这将是有趣的，我们将编写的代码将力求在内存管理方面具有现实性。如前所述，由于 C++标准在多进程系统方面的内容很少，我们将尝试使多线程代码看起来有点像多进程代码。我希望你，敏锐的读者，会接受这个提议。

请注意，本节用户代码中会有一些低级同步，包括一些通过原子变量。我尽量保持其最小化且合理现实，希望即使我不会详细解释所有内容，你也能接受，因为本书的重点是内存管理而不是并发计算（当然，这也是一个很好的主题）。如果你想知道更多关于等待原子变量或使用线程栅栏等事情，请自由使用你喜欢的并发编程资源。

准备好了吗？让我们开始吧！

## 一个虚构的共享内存 API

我们将编写一个虚构但受大多数操作系统启发（除了我们将通过异常报告错误以简化用户代码之外）的 API。操作系统主要通过从返回值中表达的错误代码来报告错误，但这会导致用户代码更加复杂。我希望这对你，亲爱的读者，来说似乎是一个可以接受的折衷方案。

正如大多数操作系统所做的那样，我们将通过一种形式的手柄或键来抽象实际的资源；创建一个某个大小的“共享内存”段将产生一个键（一个整数标识符），之后，访问该内存将需要这个键，销毁该内存也是如此。正如预期的那样，对于一个旨在用于在进程之间共享数据的设施，销毁内存不会最终确定其中的对象，因此用户代码需要确保在释放共享内存段之前销毁共享内存中的对象。

我们 API 的签名和类型如下：

```cpp
// ...
#include <cstddef> // std::size_t
#include <new> // std::bad_alloc
#include <utility> // std::pair
class invalid_shared_mem_key {};
enum shared_mem_id : std::size_t;
shared_mem_id create_shared_mem(std::size_t size);
std::pair<void*, std::size_t>
  get_shared_mem(shared_mem_id);
void destroy_shared_mem(shared_mem_id);
// ...
```

你可能会注意到我们正在使用`enum`类型为`shared_mem_id`。这样做的原因是`enum`类型在 C++中是不同的类型，而不仅仅是`using`或`typedef`会得到的一个别名。当基于它们的参数类型重载函数时，具有不同的类型可能很有用。这是一个有用的技巧：如果我们编写两个具有相同名称的函数（一个接受`shared_mem_id`类型的参数，另一个接受`std::size_t`类型的参数），这些将是不同的函数，尽管`shared_mem_id`的底层类型是`std::size_t`。

由于我们正在构建一个“共享内存”的人工实现来展示内存分配函数如何简化用户代码，因此我们的 API 函数实现将编写得简单，但让我们编写客户端代码，使其表现得好像它正在使用共享内存。我们将定义一个共享内存段为一个由字节数组及其字节大小组成的`shared_mem_block`对。我们将保持一个该类型的`std::vector`对象，使用该数组中的索引作为`shared_mem_id`。这意味着当`shared_mem_block`对象被销毁时，我们不会在`std::vector`中重用其索引（容器最终会有“空洞”，换句话说）。

我们的实施方案如下。请注意，它不是线程安全的，但这不会影响我们与内存管理相关的讨论：

```cpp
// ...
#include <vector>
#include <memory>
#include <utility>
struct shared_mem_block {
  std::unique_ptr<char[]> mem;
  std::size_t size;
};
std::vector<shared_mem_block> shared_mems;
std::pair<void*, std::size_t>
  get_shared_mem(shared_mem_id id) {
  if (id < std::size(shared_mems))
      return { shared_mems[id].mem.get(),
               shared_mems[id].size };
  return { nullptr, 0 };
}
shared_mem_id create_shared_mem(std::size_t size) {
  auto p = std::make_unique<char[]>(size);
  shared_mems.emplace_back(std::move(p), size);
  // note the parentheses
  return shared_mem_id(std::size(shared_mems) - 1);
}
// function for internal purposes only
bool is_valid_shared_mem_key(shared_mem_id id) {
  return id < std::size(shared_mems) &&
         shared_mems[id].mem;
}
void destroy_shared_mem(shared_mem_id id) {
  if (!is_valid_shared_mem_key(id))
      throw invalid_shared_mem_key{};
  shared_mems[id].mem.reset();
}
```

如果你想进行实验，你可以用你选择的操作系统的等效实现替换这些函数的实现，如果需要，调整 API。

配备了这个实现，我们现在可以比较一个使用共享内存的“手工”代码示例和一个受益于 C++设施的实现。我们将通过以下代码进行这种比较：一个从共享内存段分配一些数据块，然后启动两个线程（一个写线程和一个读线程）。写线程将写入共享数据，然后（通过最小化同步）读线程将从它读取。如前所述，我们的代码将使用*进程内*同步（C++原子变量），但在实际代码中，你应该使用操作系统提供的*进程间*同步机制。

关于生命周期

你可能还记得从*第一章*中了解到，每个对象都有一个关联的生命周期，编译器会在你的程序中跟踪这一事实。我们的虚构多进程示例实际上是一个单进程、多线程示例，因此通常的 C++生命周期规则适用。

如果你想将本节中的代码用于编写一个真正的多进程系统以运行一些测试，你可能需要考虑在这些没有明确创建`data`对象的进程中使用 C++23 中的`std::start_lifetime_as()`，并避免基于编译器的推理产生的有害优化，即在这些进程中，对象从未被构造。在早期的编译器中，一个通常有效的方法是将未正式构造的对象的`std::memcpy()`调用到自身上，从而有效地开始其生命周期。

在我们的“手工制作”和标准外观的实现中，我们将使用由一个`int`值和一个布尔`ready`标志组成的`data`对象：

```cpp
struct data {
  bool ready;
  int value;
};
```

在单进程实现中，对于完成标志，更好的选择是`atomic<bool>`对象，因为我们想确保在写入`ready`标志之前写入值，但由于我们希望这个示例看起来像我们正在使用进程间共享内存，我们将限制自己使用简单的`bool`，并通过其他方式确保这种同步。

关于同步的一席话

在一个现代程序中，优化编译器通常会重新排序看似独立的操作以生成更好的代码，处理器在代码生成后也会进行同样的操作，以最大化处理器内部流水线的使用。并发代码有时会包含既对编译器也对处理器不可见的依赖。在我们的示例中，我们希望`ready`完成标志仅在`value`写入操作完成后变为`true`；这个顺序之所以重要，是因为写入操作在一个线程中执行，但另一个线程将查看`ready`以确定是否可以读取`value`。

如果不通过某种形式的同步强制执行`value`-`ready`写入顺序，编译器或处理器可能会重新排序这些（看似独立的）写入，并破坏我们对`ready`含义的假设。

## 一个手工的用户代码示例

我们当然可以编写使用我们虚构的 API 的用户代码，而无需求助于 C++的特殊内存管理设施，仅仅依靠如*第七章*中看到的放置`new`的使用。可能会诱使人们认为放置`new`是一种特殊设施，因为你可能从这本书中了解到它，但如果这是你的观点，我们邀请你重新考虑：放置`new`机制是一种几乎在所有程序中使用的根本性内存管理工具，无论用户代码是否意识到它。

作为提醒，我们的示例程序将执行以下操作：

+   创建一个指定大小的共享内存段（在这种情况下，我们将分配比所需更多的内存）。

+   在该段的开始处构造一个`data`对象，显然是通过放置`new`来完成的。

+   启动一个线程，该线程将等待 `go` 变量（类型为 `atomic<bool>`）上的信号，然后获得对共享内存段的访问权限，写入 `value` 数据成员，然后仅通过 `ready` 数据成员发出写入已发生的信号。

+   启动另一个线程，该线程将获得对共享内存段的访问权限，获取其中共享 `data` 对象的指针，然后对 `ready` 标志进行一些（非常低效的）忙等待以改变状态，之后将读取并使用 `value`。一旦完成，将通过 `done` 标志（类型为 `atomic<bool>`）发出完成信号。

+   然后我们的程序将从键盘读取一个键，向线程（实际上是写入线程）发出信号，表明是时候开始工作了，并在释放共享内存段并结束工作之前等待它们完成。

因此，我们最终得到以下结果：

```cpp
// ...
#include <thread>
#include <atomic>
#include <iostream>
int main() {
  // we need a N-bytes shared memory block
  constexpr std::size_t N = 1'000'000;
  auto key = create_shared_mem(N);
  // map a data object in the shared memory block
  auto [p, sz] = get_shared_mem(key);
  if (!p) return -1;
  // start the lifetime of a non-ready data object
  auto p_data = new (p) data{ false };
  std::atomic<bool> go{ false };
  std::atomic<bool> done{ false };
  std::jthread writer{ [key, &go] {
      go.wait(false);
      auto [p, sz] = get_shared_mem(key);
      if (p) {
        auto p_data = static_cast<data*>(p);
        p_data->value = 3;
        std::atomic_thread_fence(
            std::memory_order_release
        );
        p_data->ready = true;
      }
  } };
  std::jthread reader{ [key, &done] {
auto [p, sz] = get_shared_mem(key);
      if (p) {
        auto p_data = static_cast<data*>(p);
        while (!p_data->ready)
            ; // busy waiting, not cool
        std::cout << "read value "
                  << p_data->value << '\n';
      }
      done = true;
      done.notify_all();
  } };
  if (char c; !std::cin.get(c)) exit(-1);
  go = true;
  go.notify_all();
  // writer and reader run to completion, then complete
  done.wait(false);
  p_data->~data();
  destroy_shared_mem(key);
}
```

我们使这项工作得以实现：我们有一种某种形式的基础设施来管理共享内存段，我们可以使用这些内存块来共享数据，我们可以编写读取该共享数据的代码，也可以写入它。请注意，我们在每个线程中捕获了 `key` 变量中的密钥，然后通过该密钥在每个 lambda 中获取内存块，但简单地捕获 `p_data` 指针并使用它也是合理的。

然而，请注意，我们并没有真正管理那个块：我们创建了它，并在开始时使用了一个大小为 `sizeof(data)` 的小块。现在，如果我们想在那个区域创建多个对象呢？如果我们想编写既创建又销毁对象的代码，引入了在给定时间管理该块哪些部分正在使用的需求呢？根据我们刚才写的，这意味着所有这些都在用户代码中完成，这是一个相当繁重的任务。

记住这一点，我们现在将用不同的方法解决相同的问题。

## 一个看起来标准的用户代码等效

那么，如果我们想以更习惯的方式使用“奇特”内存，C++ 提供了什么机制呢？嗯，这样做的一种方法如下：

+   编写一个用于“奇特”内存的管理器类，封装对操作系统的不可移植接口，并公开更接近 C++ 用户代码预期的服务

+   编写内存分配操作符的重载（如 `operator new()`、`operator delete()` 等），这些重载接受对这样一个管理对象的引用作为额外的参数

+   通过在内存管理器对象上委托来使这些重载的内存分配操作符在可移植和非可移植代码之间架起桥梁

这样，用户代码可以基本上写成“看起来正常”的代码，调用 `new` 和 `delete` 操作符，只是这些调用将使用与 *第七章* 中类似的那种扩展符号，例如 `nothrow` 或放置版本的 `operator new()`。

我们的 `shared_mem_mgr` 类将使用本节前面描述的虚构操作系统 API，但通常，人们会编写一个封装所需操作系统服务的类，以便在程序中使用目标内存。

由于这是一个为了简单而制作的示例，主要是为了展示功能的工作方式和如何使用，聪明的读者你可能会看到很多改进和优化的空间…确实，这个管理器非常慢，且占用内存，它保持一个 `std::vector<bool>` 对象，其中每个 `bool` 值指示内存块中的字节是否已被占用，并且每当有分配请求时，都会在这个容器中进行简单的线性搜索（此外，它不是线程安全的，这是不好的！）。我们将在 *第十章* 中检查一些实现质量的考虑因素，但没有任何阻止你在同时期将 `shared_mem_mgr` 改进得更好的事情。

你会注意到 `shared_mem_mgr` 被表达为一个 RAII 类型：它的构造函数创建一个共享内存段，它的析构函数释放该内存段，并且 `shared_mem_mgr` 类型已经被设置为不可复制的，这在 RAII 类型中很常见。在下面的代码摘录中，需要查看的关键成员函数是 `allocate()` 和 `deallocate()`；前者尝试从共享内存段分配一个块并记录这一行为，而后者释放与块内地址关联的内存：

```cpp
#include <algorithm>
#include <iterator>
#include <new>
class shared_mem_mgr {
  shared_mem_id key;
  std::vector<bool> taken;
  void *mem;
  auto find_first_free(std::size_t from = 0) {
      using namespace std;
      auto p = find(begin(taken) + from, end(taken),
                    false);
      return distance(begin(taken), p);
  }
  bool at_least_free_from(std::size_t from, int n) {
      using namespace std;
      return from + n < size(taken) &&
             count(begin(taken) + from,
                   begin(taken) + from + n,
                   false) == n;
  }
  void take(std::size_t from, std::size_t to) {
      using namespace std;
      fill(begin(taken) + from, begin(taken) + to,
          begin(taken) + from, true);
  }
  void free(std::size_t from, std::size_t to) {
      using namespace std;
      fill(begin(taken) + from, begin(taken) + to,
          begin(taken) + from, false);
  }
public:
  // create shared memory block
  shared_mem_mgr(std::size_t size)
      : key{ create_shared_mem(size) }, taken(size) {
      auto [p, sz] = get_shared_mem(key);
      if (!p) throw invalid_shared_mem_key{};
mem = p;
  }
  shared_mem_mgr(const shared_mem_mgr&) = delete;
  shared_mem_mgr&
      operator=(const shared_mem_mgr&) = delete;
  void* allocate(std::size_t n) {
      using namespace std;
      std::size_t i = find_first_free();
      // insanely inefficient
      while (!at_least_free_from(i, n) && i != size(taken))
        i = find_first_free(i + 1);
      if (i == size(taken)) throw bad_alloc{};
      take(i, i + n);
      return static_cast<char*>(mem) + i;
  }
  void deallocate(void *p, std::size_t n) {
      using namespace std;
      auto i = distance(
         static_cast<char*>(mem), static_cast<char*>(p)
      );
      take(i, i + n);
  }
  ~shared_mem_mgr() {
      destroy_shared_mem(key);
  }
};
```

如你所见，`shared_mem_mgr` 确实是一个管理内存块段的类，其中并不涉及任何魔法。如果有人想要改进内存管理算法，他们可以在不触及这个类的接口的情况下做到这一点，从而受益于封装带来的低耦合。

如果你想玩…

一种有趣的改进 `shared_mem_mgr` 的方法可能是首先让这个类负责分配和释放共享内存，正如它已经做的那样，然后编写一个不同的类来管理该共享内存块内的内存，最后使它们协同工作。这样，人们可以使用 `shared_mem_mgr` 与不同的内存管理算法，并根据个别程序或其部分的需求选择管理策略。如果你想要找些乐子，这是一个可以尝试的方法！

下一步是实现接受 `shared_mem_mgr&` 类型参数的分配运算符重载。这基本上是微不足道的，因为这些重载只需要将工作委托给管理器：

```cpp
void* operator new(std::size_t n, shared_mem_mgr& mgr) {
  return mgr.allocate(n);
}
void* operator new[](std::size_t n, shared_mem_mgr& mgr) {
  return mgr.allocate(n);
}
void operator delete(void *p, std::size_t n,
                    shared_mem_mgr& mgr) {
  mgr.deallocate(p, n);
}
void operator delete[](void *p, std::size_t n,
                      shared_mem_mgr& mgr) {
  mgr.deallocate(p, n);
}
```

配备了我们的管理器和这些重载，我们可以编写我们的测试程序，该程序执行与上一节中“手工”相同的任务。然而，在这种情况下，有一些不同之处：

+   我们不需要管理共享内存段的创建和销毁。这些任务由 `shared_mem_mgr` 对象作为其实现 RAII 习语的组成部分来处理。

+   我们根本不需要管理共享内存块，因为这个任务分配给了`shared_mem_mgr`对象。在块中找到一个放置对象的位置，跟踪块如何被对象使用，确保可以区分已使用区域和未使用区域，等等，这些都是该类职责的一部分。

+   作为推论，在“手工”版本中，我们在共享内存块的开始处构建了一个对象，并指出构建更多对象或管理共享内存段以考虑对`new`和`delete`操作符的多次调用将给用户代码带来负担，但在这个实现中，我们可以自由地调用`new`和`delete`，因为我们希望这种内存管理对客户端代码来说是透明的。

在非典型内存中构建对象方面，相当简单：只需在调用`new`和`new[]`操作符时传递额外的参数即可。然而，通过此类经理对象管理的对象的最终化部分则稍微复杂一些：我们不能在我们的指针上写`delete p`这样的代码，因为这会尝试最终化对象并通过“正常”方式释放内存。相反，我们需要手动最终化对象，然后手动调用适当的`operator delete()`函数版本，以执行异构的内存清理任务。当然，鉴于我们在*第六章*中写的内容，你可以将这些任务封装在你自己的智能指针中，以获得更简单、更安全的用户代码。

我们最终得到了以下示例程序：

```cpp
int main() {
  // we need a N-bytes shared memory block
  constexpr std::size_t N = 1'000'000;
  // HERE
  shared_mem_mgr mgr{ N };
  // start the lifetime of a non-ready data object
auto p_data = new (mgr) data{ false };
  std::atomic<bool> go{ false };
  std::atomic<bool> done{ false };
  std::jthread writer{ [p_data, &go] {
      go.wait(false);
      p_data->value = 3;
      std::atomic_thread_fence(std::memory_order_release);
      p_data->ready = true;
  } };
  std::jthread reader{ [p_data, &done] {
      while (!p_data->ready)
        ; // busy waiting, not cool
      std::cout << "read value " << p_data->value << '\n';
      done = true;
      done.notify_all();
  } };
  if (char c; !std::cin.get(c)) exit(-1);
  go = true;
  go.notify_all();
  // writer and reader run to completion, then complete
  done.wait(false);
  p_data->~data();
  operator delete(p_data, sizeof(data), mgr);
}
```

这仍然不是一个简单的例子，但内存管理方面显然比“手工”版本简单，任务的模块化使得优化内存管理方式变得更容易。

然后……我们就完成了。呼！这真是一次相当刺激的旅程，又一次！

# 摘要

本章探讨了各种使用 C++内存管理设施的特殊方式：将对象映射到内存映射硬件上，将基本的错误处理与`nothrow`版本的`operator new()`集成，使用`std::exception_handler`来应对内存不足的情况，以及通过“正常”分配操作符和经理对象的专业化来访问非典型内存。这为我们提供了对 C++内存管理设施的更广泛概述，以及如何利用它们来发挥优势。

我们提到过但尚未讨论的一件事是优化：如何在满足某些条件时使内存分配和内存释放变得快速，甚至非常快，并且在执行速度方面是确定的。这就是我们在*第十章*中解释如何编写基于竞技场的分配代码时将要做的。

哦，而且作为额外奖励，我们将消灭奥克瑞斯。

奥克瑞斯？你在说什么？

兽人是一种虚构的生物，出现在众多虚构幻想作品中，通常被用作敌人，与精灵（另一种虚构生物，通常有更好的声誉）有不健康的关联。由于你的友好作者在过去几十年里与游戏程序员合作了很多，兽人往往会出现在他的例子中，并且将是我们在*第十章*中编写的代码的核心。

听起来不错吗？那么，接下来是下一章！
