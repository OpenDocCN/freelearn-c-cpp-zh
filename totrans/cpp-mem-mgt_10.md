# 10

# 基于区域的内存管理和其他优化

我们的内存管理工具箱随着每一章的增长而增长。我们现在知道如何重载内存分配运算符（*第七章*）以及如何将这项技能应用于解决各种具体问题的方法（*第八章*和*第九章*都提供了一些说明性的、现实世界的例子）。

想要控制内存分配机制的一个重要原因是**性能**。现在，声称能够轻易击败库供应商提供的这些函数的实现是轻率的（而且显然是错误的！），因为这些实现对于平均情况来说通常是好的，很多时候是非常好的。当然，前一句话的关键元素是“对于平均情况”。当一个人的使用案例在事先已知其特定性时，有时可以利用这些信息，设计出一个超越任何可能为优秀**平均**性能设计的实现的实现。

本章是关于使用我们想要解决的内存管理问题的知识来构建一个对我们来说表现卓越的解决方案。这可能意味着一个平均情况下更快、在最坏情况下也足够快、显示确定性的执行时间、减少内存碎片等解决方案。毕竟，现实世界程序中有许多不同的需求和约束，我们经常不得不做出选择。

一旦本章结束，我们的工具箱将扩展，使我们能够做到以下事情：

+   编写针对事先已知约束优化的基于区域的分配策略算法

+   编写按内存块大小分配的策略

+   理解与这些技术相关的益处以及风险

本章涵盖的技术将引导我们探索与某些专用应用领域中内存分配运算符重载非常接近的使用案例。因此，我们最初将它们应用于一个“真实生活”问题：中世纪幻想游戏中兽人与精灵之间的战斗。

关于优化的（有时是减少的）回报

由于我们将在本章中讨论优化技术（以及其他内容），因此需要一些警告：**优化是一件棘手的事情**，是一个移动的目标，今天使代码变得更好的东西，明天可能会使其变差。同样，理论上看起来不错的主意，一旦实施和测试，可能会在实践中导致减速，有时人们可能会花费大量时间优化很少使用的代码，实际上是在浪费时间和金钱。

在尝试优化程序的部分之前，通常明智的做法是测量，理想情况下使用分析工具，并确定可能从你的努力中受益的部分。然后，保留一个简单（但正确）的代码版本，并使用它作为基线。每次尝试优化时，将结果与基线代码进行比较，并定期运行这些测试，尤其是在更改硬件、库、编译器或其版本时。有时，例如编译器升级可能会引入一种新的优化，它“看穿”简单的基线代码，使其比精心制作的替代方案更快。要谦逊，要合理，要尽早测量，要经常测量。

# 技术要求

您可以在此处找到本书中该章节的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter10`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter10)。

# 基于竞技场的内存管理

基于竞技场内存管理的理念是在程序中的某个已知时刻分配一块内存，并根据对情况或问题域的了解，将其管理为一个“小而个性化的堆”。

在这个一般主题上有许多变体，包括以下内容：

+   在游戏中，通过场景或级别分配和管理内存，在场景或级别结束时作为一个单独的块释放它。这有助于减少程序中的内存碎片。

+   当已知分配和释放的条件遵循给定的模式或具有有限的内存需求时，专门化分配函数以利用这些信息。

+   以一种方式表达对一组相似对象的所有权，以便在程序稍后某个时刻一次性销毁它们，而不是逐个销毁。

解释基于竞技场分配的工作原理的最佳方式可能是编写一个使用它的示例程序，并展示它所做的工作以及它带来的好处。我们将以这种方式编写代码，以便根据宏的存在，使用标准库提供的分配函数或我们自己的专用实现，并且当然，我们将测量分配和释放代码，以查看我们的努力是否有益。

## 具体示例 - 基于大小的实现

假设我们正在制作一款视频游戏，其中动作汇聚到一个壮丽的终局，兽人和精灵在一场宏大的战斗中相遇。没有人真的记得为什么这两个群体彼此仇恨，但有一种怀疑，有一天，一个精灵对一个兽人说：“你知道，你今天闻起来并不那么糟糕！”而这个兽人如此受辱，以至于它开始了一场至今仍在进行的纷争。无论如何，这是一个谣言。

在这个游戏中，关于使用兽人代码的行为有一些了解，具体如下：

+   总共动态分配的`Orc`对象数量将不会超过某个特定数量，因此我们有存储这些生物所需空间的上限。

+   在那个游戏中，死亡的兽人将不会复活，因为没有萨满可以将其复活。换句话说，没有必要实现一个在对象被销毁后重用`Orc`对象存储的策略。

这两个属性为我们提供了算法选择：

+   如果我们有足够的内存可用，我们可以预先分配一个足够大的内存块，以便将所有`Orc`对象放入游戏中，因为我们知道最坏的情况是什么。

+   由于我们知道我们不需要重用与单个`Orc`对象关联的内存，我们可以实现一个简单（并且非常快速）的分配策略，这个策略几乎不做记录，并且正如我们将看到的，让我们能够实现针对这种类型的确定性、常数时间分配。

为了这个例子，`Orc`类将由三个数据成员表示，`name`（一个`char[4]`，因为这些生物的词汇有限），`strength`（类型为`int`），和`smell`（类型为`double`，因为这些生物有…声誉），如下所示：

```cpp
class Orc {
  char name[4]{ 'U', 'R', 'G' };
  int strength = 100;
  double smell = 1000.0;
public:
  static constexpr int NB_MAX = 1'000'000;
  // ...
};
```

在这个例子中，我们将为`Orc`对象使用任意默认值，因为我们只关心这个例子中的分配和释放。当然，如果你愿意，你可以编写更复杂的测试代码来使用非默认值，但这不会影响我们的讨论，所以我们将目标定为简单。

由于我们通过基于大小的竞技场预先讨论了大型内存块的内存分配，我们需要查看`Orc`对象的内存大小消耗。假设`sizeof(int)==4`和`sizeof(double)==8`，并且假设作为基本类型，它们的对齐要求与它们各自的大小相匹配，在这种情况下，我们可以假设`sizeof(Orc)==16`。如果我们旨在一次性为所有`Orc`对象分配足够的空间，确保`sizeof(Orc)`对于我们的资源来说是合理的，这一点很重要。例如，将程序中`Orc`对象的最大数量定义为`Orc::NB_MAX`，以及我们可以一次性为`Orc`对象分配的最大内存量定义为某个假设的常量`THRESHOLD`，我们可以在源代码中留下一个如下的`static_assert`作为*约束检查*的形式：

```cpp
static_assert(Orc::NB_MAX*sizeof(Orc) <= THRESHOLD);
```

这样，如果我们最终将`Orc`类发展到资源成为问题的情况，代码将无法编译，我们就能重新评估情况。在我们的例子中，考虑到大约 16 MB 的内存消耗，我们假设我们处于预算范围内，可以继续我们的竞技场开发。

我们将想要将我们的基于地盘的实现与基线实现进行比较，在这种情况下，将是标准库提供的内存分配函数的实现。重要的是要提前指出，每个标准库实现都提供自己版本的这些函数，因此您可能需要在多个实现上运行我们将要编写的代码，以更好地了解我们的技术的影响。

要编写允许我们进行适当比较的代码，我们需要两个不同的可执行文件，因为我们将处于一个非此即彼的情况（我们要么得到标准版本，要么得到我们正在编写的“自制”版本），因此这是一个基于宏的条件编译的好用例。因此，我们将编写一组单一的源文件，这些文件将条件性地替换标准库提供的分配运算符版本为我们自己的版本，但其他方面基本上是相同的。

我们将使用三个文件进行工作：`Orc.h`，它声明了 `Orc` 类和条件定义的分配运算符重载；`Orc.cpp`，它提供了这些重载的实现以及本身的地盘实现；以及一个测试程序，该程序分配 `Orc::NB_MAX` 个 `Orc` 类型的对象，然后稍后销毁它们，并测量执行这两个操作所需的时间。当然，像大多数微基准测试一样，对这些测量结果要持保留态度：在真实程序中，分配操作会与其他代码交织在一起，因此这些数字不会相同，但至少我们将对分配运算符的两个实现应用相同的测试，所以比较应该是相对公平的。

### 声明 `Orc` 类

首先，让我们检查 `Orc.h`，我们已经在之前展示 `Orc` 类的数据成员布局时部分看到了它：

```cpp
#ifndef ORC_H
#define ORC_H
// #define HOMEMADE_VERSION
#include <cstddef>
#include <new>
class Orc {
  char name[4]{ 'U', 'R', 'G' };
  int strength = 100;
  double smell = 1000.0;
public:
  static constexpr int NB_MAX = 1'000'000;
#ifdef HOMEMADE_VERSION
   void * operator new(std::size_t);
   void * operator new[](std::size_t);
   void operator delete(void *) noexcept;
   void operator delete[](void *) noexcept;
#endif
};
#endif
```

`HOMEMADE_VERSION` 宏可以取消注释以使用我们版本的分配函数。正如预期的那样，因为我们正在为 `Orc` 类及其预期的使用模式应用一种特殊策略，所以我们使用成员函数重载来处理分配运算符。（我们难道愿意像对待奥克一样对待 `int` 对象，或者——想象一下！——精灵吗？我想不是吧。）

### 定义 `Orc` 类和实现地盘

与内存管理相关的代码的核心将在 `Orc.cpp` 中。我们将分两步进行，地盘实现和分配运算符重载，并将分别分析不同的重要部分。此文件中找到的整个实现将根据 `HOMEMADE_VERSION` 宏进行条件编译。

我们将命名我们的地盘类为 `Tribe`，它将是一个单例。是的，我们在 *第八章* 中使用过的那个被诅咒的设计模式再次出现，但我们在程序中确实想要一个单一的 `Tribe` 对象，这样就能很好地传达意图。我们实现的重要部分如下：

+   `Tribe` 类的默认（也是唯一）构造函数分配了一个 `Orc::NB_MAX*sizeof(Orc)` 字节的单个块。重要的是要立即指出，这个块中没有 `Orc` 对象：这个内存块的大小和形状正好适合放置我们需要的所有 `Orc` 对象。基于竞技场分配的一个关键思想是，至少对于这个实现来说，*竞技场管理的是原始内存，而不是对象*：对象的构造和析构是用户代码的范畴，任何在程序结束时没有正确析构的对象都是用户代码的过错，而不是竞技场的过错。

+   我们立即验证分配是否成功。在这种情况下，我使用了 `assert()`，因为代码的其余部分都依赖于这个成功，但抛出 `std::bad_alloc` 或调用 `std::abort()` 也是一个合理的选择。`Tribe` 对象保持两个指针，`p` 和 `cur`，两者最初都指向块的开始。我们将使用 `p` 作为 *块开始* 标记，而 `cur` 作为 *返回下一个块的指针*；因此，`p` 将在整个程序执行过程中保持稳定，而 `cur` 将随着每次分配向前移动 `sizeof(Orc)` 字节。

使用 char* 或 Orc*

这个 `Tribe` 实现使用 `char*` 作为 `p` 和 `cur` 指针，但 `Orc*` 也是一个正确的选择。只需记住，对于 `Tribe` 对象而言，竞技场中没有 `Orc` 对象，使用 `Orc*` 类型仅仅是一个方便的谎言，以简化指针运算。这种改变将涉及在构造函数中将 `static_cast<char*>` 替换为 `static_cast<Orc*>`，以及在 `allocate()` 成员函数的实现中将 `cur+=sizeof(Orc)` 替换为 `++cur`。这主要是一个风格和个人偏好的问题。

+   析构函数释放了由 `Tribe` 对象管理的整个内存块。这是一个非常高效的程序：它比单独释放较小的块要快，并且导致非常少的内存碎片。

+   这个第一个实现使用了在 *第八章* 中看到的梅耶斯单例技术，但我们在本章的后面将使用不同的方法来比较两种实现策略对同一设计模式性能的影响……因为确实有这种影响，正如我们将看到的。

我们基于预期使用模式的先验知识，我们的基于大小的竞技场实现将如何受益如下：

+   每次分配都将返回一个顺序“分配”的 `Orc` 大小块，这意味着没有必要搜索一个合适大小的块——我们始终知道它在哪里。

+   在释放内存时没有工作要做，因为我们一旦使用过这些块就不会再重用它们。请注意，根据标准规则，分配和释放函数必须是线程安全的，这解释了为什么在这个实现中我们使用了 `std::mutex`。

代码如下：

```cpp
#include "Orc.h"
#ifdef HOMEMADE_VERSION
#include <cassert>
#include <cstdlib>
#include <mutex>
class Tribe {
  std::mutex m;
  char *p, *cur;
  Tribe() : p{ static_cast<char*>(
      std::malloc(Orc::NB_MAX * sizeof(Orc))
  ) } {
      assert(p);
      cur = p;
  }
  Tribe(const Tribe&) = delete;
  Tribe& operator=(const Tribe&) = delete;
public:
  ~Tribe() {
      std::free(p);
  }
  static auto &get() {
      static Tribe singleton;
      return singleton;
  }
  void * allocate() {
      std::lock_guard _ { m };
      auto q = cur;
      cur += sizeof(Orc);
      return q;
  }
  void deallocate(void *) noexcept {
  }
};
// ...
```

如你所猜想的，这些分配条件几乎是最优的，但在实际应用中发生的频率比我们想象的要高。一个类似高效的用法模式可以模拟栈（最后分配的块是下一个释放的块），而我们每天编写的使用局部变量的代码往往就是底层内存的常用最优使用模式，而我们可能并没有意识到这一点。

接下来，我们将讨论重载的分配运算符。为了使此实现简单，我们将假设不会有 `Orc` 对象的数组需要分配，但你可以将实现细化以考虑数组（这不是一个困难的任务；只是编写相关测试代码更复杂）。这些函数所起的作用是将工作委托给底层区域，并且它们仅用于 `Orc` 类（这一点在后面的 *当参数改变* 部分会有所讨论）。因此，它们几乎是微不足道的：

```cpp
// ...
void * Orc::operator new(std::size_t) {
  return Tribe::get().allocate();
}
void * Orc::operator new[](std::size_t) {
  assert(false);
}
void Orc::operator delete(void *p) noexcept {
  Tribe::get().deallocate(p);
}
void Orc::operator delete[](void *) noexcept {
  assert(false);
}
#endif // HOMEMADE_VERSION
```

### 测试我们的实现

接下来，我们将讨论我们将使用的测试代码实现。此程序将由一个名为 `test()` 的微基准函数和一个 `main()` 函数组成。我们将分别检查这两个函数。

`test()` 函数将接受一个非 `void` 函数 `f()`，一个可变参数包 `args`，并调用 `f(args...)`，确保在该调用中使用完美转发来传递参数，以确保参数以原始调用中预期的语义传递。它在调用 `f()` 之前和之后读取时钟，并返回一个由 `f(args...)` 执行的结果和此调用期间经过的时间组成的 `pair`。我在我的代码中使用了 `high_resolution_clock`，但在此情况下使用 `system_clock` 或 `steady_clock` 也有合理的理由：

```cpp
#include <chrono>
#include <utility>
template <class F, class ... Args>
  auto test(F f, Args &&... args) {
      using namespace std;
      using namespace std::chrono;
      auto pre = high_resolution_clock::now();
      auto res = f(std::forward<Args>(args)...);
      auto post = high_resolution_clock::now();
      return pair{ res, post - pre };
  }
// ...
```

你可能会想知道为什么我们要求非 `void` 函数，即使在某些情况下返回值可能有些人为，也要返回 `f(args...)` 的调用结果。这里的想法是确保编译器认为 `f(args...)` 的结果是有用的，并且不会将其优化掉。编译器确实很聪明，可以根据所谓的“as-if 规则”删除看似无用的代码（简单来说，如果调用函数没有明显的效果，就把它去掉！）。

对于测试程序本身，请注意以下方面：

+   首先，我们将使用 `std::vector<Orc*>`，而不是 `std::vector<Orc>`。一开始这可能会显得有些奇怪，但既然我们正在测试 `Orc::operator new()` 和 `Orc::operator delete()` 的速度，我们确实需要调用这些运算符！如果我们使用 `Orc` 对象的容器，那么根本不会调用我们的运算符。

+   在运行测试之前，我们在那个 `std::vector` 对象上调用 `reserve()`，为我们将要构建的 `Orc` 对象的指针分配空间。这是我们测量中的一个重要方面：在 `std::vector` 对象中对 `push_back()` 和类似的插入函数的调用，如果尝试向一个满容器添加元素，将需要重新分配，这种重新分配会给我们的基准测试增加噪声，因此确保容器在测试期间不需要重新分配有助于我们专注于我们想要测量的内容。

+   我们用 `test()` 函数（已经在本书中多次使用）测量的东西是一系列 `Orc::NB_MAX` 次对 `Orc::operator new()` 的调用，最终由相同数量的 `Orc::operator delete()` 调用跟随。我们假设在构建和销毁之间的时间有一个某种程度的破坏，但我们出于对您，亲爱的读者的尊重，没有展示这种破坏。

+   一旦我们到达终点，我们就打印出我们的测量结果，使用微秒作为测量单位——我们今天的计算机足够快，以至于毫秒可能不够精确。

以下是代码：

```cpp
// ...
#include "Orc.h"
#include <print>
#include <vector>
int main() {
  using namespace std;
  using namespace std::chrono;
#ifdef HOMEMADE_VERSION
  print("HOMEMADE VERSION\n");
#else
  print("STANDARD LIBRARY VERSION\n");
#endif
  vector<Orc*> orcs;
  auto [r0, dt0] = test([&orcs] {
      for(int i = 0; i != Orc::NB_MAX; ++i)
        orcs.push_back(new Orc);
      return size(orcs);
  });
  // ...
  // CARNAGE (CENSORED)
  // ...
  auto [r1, dt1] = test([&orcs] {
      for(auto p : orcs)
        delete p;
      return size(orcs);
  });
  print("Construction: {} orcs in {}\n",
        size(orcs), duration_cast<microseconds>(dt0));
  print("Destruction:  {} orcs in {}\n",
        size(orcs), duration_cast<microseconds>(dt1));
}
```

在这一点上，你可能会想知道这一切是否值得努力。毕竟，我们的标准库可能非常高效（实际上，它们平均来说非常出色！）。唯一知道结果是否会让我们满意的方法是运行测试代码并亲自查看。

### 看看这些数字

使用带有 -O2 优化级别的在线 gcc 15 编译器和运行此代码两次（一次使用标准库版本，一次使用使用 Meyers 单例的家用版本），我在 `Orc::NB_MAX`（此处为 106）个对象上对 `new` 和 `delete` 操作符的调用得到了以下数字：

|  |  | 自制 |
| --- | --- | --- |
| N=106 | 标准库 | Meyers 单例 |
| `operator new()` | 23433μs | 17906μs |
| `operator delete()` | 7943μs | 638μs |

表 10.1 – 与 Meyers 单例实现的性能比较

实际数字会因各种因素而有所不同，但比较中有趣的是比率：我们自制的 `operator new()` 只用了标准库提供的版本的 76.4% 的时间，而我们自制的 `operator delete()` 则用了基准的… 8.03% 的时间。

这些结果相当令人愉快，但它们实际上并不应该让我们感到惊讶：我们执行了常数时间的分配和几乎“无时间”的释放。我们确实在每个分配上花费时间锁定和解锁一个 `std::mutex` 对象，但大多数标准库实现的互斥锁在低争用情况下预期并且在这些情况下非常快，而且我们的程序确实进行了单线程的分配和释放，这导致代码明显没有争用。

现在，你敏锐的推理能力可能会让你惊讶，分配实际上并没有比我们刚刚测量的更快。毕竟，我们调用的是一个空函数，那么是什么消耗了这些 CPU 时间？

答案是…我们的单例，或者更准确地说，对用于 Meyers 实现的`static`局部变量的访问。记得从*第八章*中，这种技术旨在确保在需要时创建单例，`static`局部变量是在其封装函数第一次被调用时构造的。

C++实现了“魔法静态”机制，其中对`static`局部对象的构造函数的调用被同步机制保护，确保对象只被构造一次。正如我们所看到的，这种同步虽然高效，但并非免费。在我们的情况下，如果我们能保证在调用`main()`之前没有其他全局对象需要调用`Tribe::get()`，我们可以用一种更经典的方法替换 Meyers 方法，其中单例只是`Tribe`类的`static`数据成员，在类的范围内声明并在全局作用域中定义：

```cpp
// ...
// "global" singleton implementation (the rest of
// the code remains unchanged)
class Tribe {
  std::mutex m;
  char *p, *cur;
  Tribe() : p{ static_cast<char*>(
      std::malloc(Orc::NB_MAX * sizeof(Orc))
  ) } {
      assert(p);
      cur = p;
  }
  Tribe(const Tribe&) = delete;
  Tribe& operator=(const Tribe&) = delete;
  static Tribe singleton;
public:
  ~Tribe() {
      std::free(p);
  }
static auto &get() {
      return singleton;
  }
  void * allocate() {
      std::lock_guard _ { m };
      auto q = cur;
      cur += sizeof(Orc);
      return q;
  }
  void deallocate(void *) noexcept {
  }
};
// in a .cpp file somewhere, within a block surrounded
// with #ifdef HOMEMADE_VERSION and #endif
Tribe Tribe::singleton;
// ...
```

将单例对象的定义从函数内部移出——放置在全局作用域中——消除了对其构造函数调用周围的同步需求。现在，我们可以将这种实现与之前的结果进行比较，以评估涉及的成本和可获得的收益（如果有的话）。

使用之前相同的测试设置，将“全局”单例添加到比较的实现集合中，我们得到以下结果：

| N=106 |  | 自制 |
| --- | --- | --- |
| 标准库 | Meyers 单例 | 全局单例 |
| `Operator new()` | 23433μs | 17906μs | 17573μs |
| `Operator delete()` | 7943μs | 638μs | 0μs |

表 10.2 – 与 Meyers 和“全局”单例实现的性能比较

现在，这更像样子了！对`operator new()`的调用比之前快，74.99%（与标准库版本相比，以及 98.14%与 Meyers 单例相比），但`operator delete()`的调用已经变成了空操作。这已经很难做得更好了！

那么，这样做值得吗？当然，这取决于你的需求。速度是一个因素；在某些程序中，速度的提升可能是一个必需品，但在其他程序中，它可能不是一个因素，或者几乎可以忽略不计。在内存碎片减少方面，在某些程序中也能产生很大的影响，有些程序正是出于这个原因使用区域。关键是：如果你需要这样做，现在你知道如何了。

## 将 SizeBasedArena<T,N>泛化

按照编写的`Tribe`类似乎特定于`Orc`类，但在实践中，它实际上特定于`Orc`-*大小*的对象，因为它从未调用过`Orc`类的任何函数；它从未构造过`Orc`对象，也从未销毁过。这意味着我们可以将这个类转换成一个通用类，并为其在其他类似约束下预期使用的类型重用。

为了实现这一点，我们将区域代码与`Orc`类解耦，并将其放入一个单独的文件中，例如可能叫做`SizeBasedArena.h`：

```cpp
#ifndef SIZE_BASED_ARENA_H
#define SIZE_BASED_ARENA_H
#include <cassert>
#include <cstdlib>
#include <mutex>
template <class T, std::size_t N>
class SizeBasedArena {
  std::mutex m;
  char *p, *cur;
  SizeBasedArena() : p{ static_cast<char*>(
      std::malloc(N * sizeof(T))
  ) } {
      assert(p);
      cur = p;
  }
  SizeBasedArena(const SizeBasedArena&) = delete;
  SizeBasedArena&
      operator=(const SizeBasedArena&) = delete;
public:
  ~SizeBasedArena() {
      std::free(p);
  }
  static auto &get() {
      static SizeBasedArena singleton;
      return singleton;
  }
  void * allocate_one() {
      std::lock_guard _ { m };
      auto q = cur;
      cur += sizeof(T);
      return q;
  }
  void * allocate_n(std::size_t n) {
      std::lock_guard _ { m };
      auto q = cur;
      cur += n * sizeof(T);
      return q;
  }
  void deallocate_one(void *) noexcept {
  }
  void deallocate_n(void *) noexcept {
  }
};
#endif
```

可能会令人惊讶的是，我们使用了`T`和`N`作为模板参数。如果我们不在区域中使用`T`，为什么不用初始化为`sizeof(T)`的整数来代替`T`？嗯，如果`Elf`类（例如）也使用基于大小的区域，并且如果我们不幸到`sizeof(Orc)==sizeof(Elf)`，那么如果我们基于类型的尺寸而不是类型本身，并且如果它们各自的`N`参数的值相同，可能会导致`Orc`和`Elf`使用相同的区域……而我们不想这样（他们也不想！）。

为了简化这个泛型示例中单例的初始化，我们回到了梅耶斯技术。在编写泛型代码时，比编写`Orc`特定等效代码更难保证全局对象在构造时的相互依赖性不存在，因为转向泛型代码显著扩大了潜在用户基础。

`Orc.cpp`中的实现现在如下所示：

```cpp
#include "Orc.h"
#ifdef HOMEMADE_VERSION
#include "SizeBasedArena.h"
using Tribe = SizeBasedArena<Orc, Orc::NB_MAX>;
void * Orc::operator new(std::size_t) {
  return Tribe::get().allocate_one();
}
void * Orc::operator new[](std::size_t n) {
  return Tribe::get().allocate_n(n / sizeof(Orc));
}
void Orc::operator delete(void *p) noexcept {
  Tribe::get().deallocate_one(p);
}
void Orc::operator delete[](void *p) noexcept {
  Tribe::get().deallocate_n(p);
}
#endif
```

你可能已经注意到，由于`SizeBasedArena<T,N>`实现了单个对象或`n`个对象的数组的分配函数，我们已经扩展了`Orc`类的成员函数分配运算符重载，以覆盖`operator new[]()`和`operator delete[]()`。在这个点上，真的没有不这样做的原因。

# 当参数改变时

我们基于大小的区域实现非常具体：它假设了顺序分配的可能性，并且能够忽略（通常很重要）的问题，即释放内存后是否可以重用内存。

任何基于大小的实现的一个重要注意事项显然是，我们依赖于一个特定的尺寸。因此，要知道，在这个约束下，我们当前的实现稍微有些危险。确实，考虑以下我们程序的发展，我们设想了更强大、更狡猾的`Orc`子类，如下所示：

```cpp
class MeanOrc : public Orc {
  float attackBonus; // oops!
  // ...
};
```

起初可能并不明显，但我们可能已经在这个新类中破坏了一些重要的东西，因为**成员函数分配运算符被派生类继承**。这意味着`Tribe`类，也被称为相对嘈杂的名称`SizeBasedArena<Orc,Orc::NB_MAX>`，将实现一个针对`sizeof(Orc)`字节块的策略，但（意外地）也会用于大小为`MeanOrc`的对象。这只会导致痛苦。

我们可以通过两种方式保护自己免受这种灾难性的情况。对于`Orc`类，我们可以通过将类标记为`final`来完全禁止派生类：

```cpp
class Orc final {
  // ...
};
```

这消除了将`MeanOrc`作为`Orc`的派生类的可能性；我们仍然可以编写`MeanOrc`，但通过组合或其他技术，这样可以绕过继承的运算符问题。

从`SizeBasedArena<T,N>`本身的视角来看，我们也可以选择将我们的实现限制为`final`类型，如下例所示：

```cpp
// ...
#include <type_traits>
template <class T, std::size_t N>
class SizeBasedArena {
  static_assert(std::is_final_v<T>);
   // ...
};
```

然而，最后一部分可能并不适合所有人。有许多类型（例如基本类型）不是`final`的，并且可以在基于大小的竞技场中合理使用，所以这取决于你，看看这对你所写的代码来说是否是一个好主意。如果你觉得不好，那么这些约束可以用散文而不是代码来表达。

基于大小的竞技场远非内存竞技场的唯一用例。我们可以在基于大小的主题和分配策略上设想许多变体。

例如，假设我们在游戏中引入了萨满，并且内存重用成为现实需求。我们可能会遇到这样的情况：在程序中，最多有`Orc::NB_MAX`个`Orc`类型的对象*同时存在*，但在整个程序执行期间，总数可能超过这个数字。在这种情况下，我们需要考虑以下事项：

+   如果我们允许数组，我们将在竞技场内部处理*内部*碎片化，因此我们可能想要考虑一种实现方式，为每个竞技场分配超过`N*sizeof(T)`字节的内存，但要多多少呢？

+   我们需要一种策略来重用内存。我们有多种方法可供选择，包括维护一个有序的`begin,end`对列表来界定空闲块（并且更容易将它们融合以减少碎片化）或者保留一个堆栈（可能是一系列基于块大小的堆栈）来存储最近释放的块，以便更快地重用这些块。

对于“*我们代码库的最佳方法是什么？*”这样的问题，部分是技术性的，部分是政治性的：什么使得分配快速可能会减慢释放，什么使得分配速度确定性可能会在内存空间开销上付出更多，等等。问题是要确定在我们的情况下哪些权衡效果最好，并测量以确保我们获得预期的收益。如果我们无法做得比标准库更好，那么无论如何，使用标准库吧！

# 分块池

我们基于大小的竞技场示例是为了优化单个块大小和特定的使用模式，但还有许多其他原因想要应用专门的分配策略。在本节中，我们将探讨“分块池”的概念，或者说是预分配选定块大小的原始内存的池。这更多的是作为一个学术示例来构建，而不是作为生产中使用的示例；接下来的代码将相当快速，并且可以变得非常快速，但在这本书中，我们将关注一般方法，并让你，亲爱的读者，去享受优化它到你满意的程度。

在这个例子中的想法是，用户代码计划分配大小相似（但不一定是相同的）的各种类型和对象，并假设最大对象数量的上限。这给我们提供了额外的知识；利用这些知识，我们将编写一个 `ChunkSizedAllocator<N,Sz...>` 类型，其中 `N` 将是每个“尺寸类别”的对象数量，而 `Sz...` 中的每个整数值将是一个不同的尺寸类别。

为了给出一个澄清的例子，一个 `ChunkSizedAllocator<10,20,40,80,160>` 对象将预先分配足够的原始内存来容纳 10 个大小为 20 字节、40 字节、80 字节和 160 字节的对象，总共至少 3,000 字节（每个尺寸类别所需的最小尺寸之和为 *200 + 400 + 800 + 1600*）。我们在这里说“至少”，是因为为了有用，我们的类需要考虑对齐，并且为了避免分配未对齐的对象，通常需要比最小内存量更多的内存。

为了理解我们将要做什么，这里有一些提示（当然，是字面上的意思）：

+   在整数值的变长序列 `Sz...` 中，我们将要求值按升序排序，因为这会使进一步的查找更快（线性复杂度而不是二次复杂度）。由于这些值在编译时已知，是类型模板参数的一部分，因此这没有运行时成本，更多的是对用户施加的约束。当然，我们将在编译时验证这一点，以避免不愉快的事故。

+   在 C++ 中，变长参数包可以是空的，但在这个例子中，一个空的尺寸类别集合将没有意义，因此我们将确保这种情况不会发生（当然是在编译时）。显然，`N` 必须大于零，这样这个类才有用，因此我们也将对此进行验证。

+   可能不明显的是，`Sz...` 中的值至少要大于 `sizeof(std::max_align_t)`（我们也可以测试 `alignof`，但对于基本类型来说这是多余的）并且实际上，我们需要将有效的尺寸类别设置为 2 的幂，以确保可以分配任意类型。这部分将内部处理，因为对用户代码施加这一点更复杂。

通过查看代码，我们可以看到这些约束被明确地表达出来。请注意，为了使“代码叙述”更容易理解，接下来的代码是逐步展示的，所以如果你想尝试它，请确保查看完整的示例：

```cpp
#include <algorithm>
#include <vector>
#include <utility>
#include <memory>
#include <cassert>
#include <concepts>
#include <limits>
#include <array>
#include <iterator>
#include <mutex>
// ... helper functions (shown below)...
template <int N, auto ... Sz>
  class ChunkSizedAllocator {
      static_assert(is_sorted(make_array(Sz...)));
      static_assert(sizeof...(Sz) > 0);
      static_assert(
        ((Sz >= sizeof(std::max_align_t)) && ...)
      );
      static_assert(N > 0);
      static constexpr unsigned long long sizes[] {
        next_power_of_two(Sz)...
      };
      using raw_ptr = void*;
      raw_ptr blocks[sizeof...(Sz)];
      int cur[sizeof...(Sz)] {}; // initialized to zero
      // ...
```

注意，我们有两个数据成员——即 `blocks`，它将包含每个尺寸类别的原始内存块的指针，以及 `cur`，它将包含每个尺寸类别内下一个分配的索引（默认初始化为零，因为我们将在每种情况下从头开始）。

本类的代码将很快继续。目前，你可能注意到一些未解释的辅助函数：

+   我们使用`make_array(Sz...)`，这是一个`constexpr`函数，它从`Sz...`的值构建一个类型为`std::array<T,N>`的对象，期望所有值都是同一类型（`Sz...`的第一个值的类型）。我们知道`N`对于结果`std::array<T,N>`是一个编译时常数，因为它是从`Sz...`中的值的数量计算出来的。

+   我们使用`is_sorted()`谓词在`std::array<T,N>`对象上，以确保在编译时值是按升序排序的，正如我们所期望的那样。不出所料，这会简单地调用`std::is_sorted()`算法，它是一个`constexpr`，因此可以在这种上下文中使用。

+   命名为`sizes`的非`static`成员数组将包含`Sz...`中每个值（包括该值）的下一个 2 的幂：如果该值已经是 2 的幂，那太好了！因此，如果`Sz...`是`10,20,32`，那么`sizes`将包含`16,32,32`。

为什么是 2 的幂？

在实践中，如果我们连续分配不是 2 的幂的块，那么在第一次分配之后，这些块将导致对象对齐错误，为了避免这种情况而管理填充可能会变得可能，但这将显著复杂化我们的实现。为了使分配更快，我们在编译时计算`Sz...`每个元素的下一个 2 的幂，并将它们存储在`sizes`数组中。这意味着我们可能会有两个最终大小相同的尺寸类别（例如，`40`和`60`都会导致 64 字节的块），但这是一个小问题（因为代码仍然可以工作），考虑到这是一个为知识渊博的用户设计的专用设施。

这些辅助函数的代码，在实践中，是在`ChunkSizedAllocator<N,Sz...>`类的声明之前定义的，如下所示：

```cpp
// ...
template <class T, std::same_as<T> ... Ts>
  constexpr std::array<T, sizeof...(Ts)+1>
      make_array(T n, Ts ... ns) {
        return { n, ns... };
      }
constexpr bool is_power_of_two(std::integral auto n) {
  return n && ((n & (n - 1)) == 0);
}
class integral_value_too_big {};
constexpr auto next_power_of_two(std::integral auto n) {
  constexpr auto upper_limit =
      std::numeric_limits<decltype(n)>::max();
  for(; n != upper_limit && !is_power_of_two(n); ++n)
       ;
  if(!is_power_of_two(n)) throw integral_value_too_big{};
  return n;
}
template <class T>
  constexpr bool is_sorted(const T &c) {
      return std::is_sorted(std::begin(c), std::end(c));
  }
// ...
```

注意，`make_array()`使用概念来约束所有值都是同一类型，`is_power_of_two(n)`确保测试`n`的正确位以使此测试快速（它还测试`n`以确保我们不报告`0`为 2 的幂）。`next_power_of_two()`函数可能可以做得更快，但在这里这影响不大，因为它仅在编译时使用（我们可以通过将其改为`consteval`而不是`constexpr`来强制执行这一点，但可能有一些用户想要在运行时和编译时使用之间进行选择，所以我们将给他们这个选择）。

在简要讨论了辅助函数之后，我们回到`ChunkSizedAllocator<N,Sz...>`实现，这里有一个名为`within_block(p,i)`的成员函数，它仅在指针`p`位于`blocks[i]`内时返回`true`，`blocks[i]`是我们对象内存的`i`-th 预分配块。该函数的逻辑看似简单：人们可能只想测试类似`blocks[i]<=p&&p<blocks[i]+N`的东西，但考虑到`blocks[i]`变量是`void*`类型，这阻止了指针运算，但在 C++中这实际上是错误的（记得我们在*第二章*中关于指针运算复杂性的讨论）。在实践中，这可能因为与 C 代码的兼容性而有效，但这不是你想要依赖的东西。

到目前为止，正在进行讨论，以添加一个标准库函数来测试一个指针是否位于两个其他指针之间，但直到这种情况发生，我们至少可以使用标准库提供的`std::less`函数对象来使比较变得合法。我知道这并不令人满意，但今天它可能适用于所有编译器……通过将这个测试局部化到一个专用函数中，一旦我们有一个真正的标准解决方案来解决这个问题，我们就可以简化源代码的更新：

```cpp
      // ...
      bool within_block(void *p, int i) {
        void* b = blocks[i];
        void* e = static_cast<char*>(b) + N * sizes[i];
        return p == b ||
                (std::less{}(b, p) && std::less{}(p, e));
      }
      // ...
```

没有必要使`ChunkSizedAllocator<N,Sz...>`对象全局可用：这是一个可以在程序中多次实例化并用于解决各种问题的工具。然而，我们不希望该类型是可复制的（我们可以这样做，但这会真正复杂化设计，而回报有限）。

通过`std::malloc()`，我们的构造函数为`Sz...`中的各种大小分配了原始内存块，或者至少是每个这些大小的下一个 2 的幂，正如本节前面所解释的，之后确保所有分配都成功。我们使用了`assert()`来做到这一点，但也可以在成功分配内存块后抛出`std::bad_alloc`异常，前提是必须小心地调用`std::free()`。

我们的析构函数，不出所料，对每个内存块调用`std::free()`：正如本章前面提到的区域实现一样，`ChunkSizedAllocator<N,Sz...>`对象负责内存，而不是客户端代码放入其中的对象，因此我们必须假设在调用该对象的析构函数之前，客户端代码已经销毁了存储在`ChunkSizedAllocator`对象内存块中的所有对象。

注意存在一个`std::mutex`数据成员，因为我们稍后需要这个（或某种其他同步工具）来确保分配和释放是线程安全的：

```cpp
      // ...
      std::mutex m;
  public:
      ChunkSizedAllocator(const ChunkSizedAllocator&)
         = delete;
      ChunkSizedAllocator&
        operator=(const ChunkSizedAllocator&) = delete;
      ChunkSizedAllocator() {
        int i = 0;
        for(auto sz : sizes)
            blocks[i++] = std::malloc(N * sz);
        assert(std::none_of(
std::begin(blocks), std::end(blocks),
            [](auto p) { return !p; }
        ));
      }
      ~ChunkSizedAllocator() {
        for(auto p : blocks)
            std::free(p);
      }
      // ...
```

最后，我们通过`allocate()`和`deallocate()`成员函数到达了我们努力的精髓。在`allocate(n)`中，我们寻找最小的元素`sizes[i]`，其分配的块大小足够大，可以容纳`n`字节。一旦找到这样一个块，我们就锁定我们的`std::mutex`对象以避免竞争条件，然后查看`blocks[i]`中是否至少还有一个可用的块；这种实现按顺序取用它们，并且不重复使用它们，以保持讨论简单。如果有，我们就取用这个块，更新`cur[i]`，并将适当的地址返回给用户代码。

注意，当我们没有在我们的预分配块中找到空闲块，或者当`n`太大而无法使用我们事先分配的块时，我们将分配责任委托给`::operator new()`，这样分配请求仍然可能成功。我们也可以在这种情况下抛出`std::bad_alloc`，这取决于意图：如果我们认为分配必须在我们自己的块中进行，而不是在其他地方，那么抛出或以其他方式失败是一个更好的选择。

失败怎么会是好事呢？

一些应用，尤其是在低延迟或实时系统领域的嵌入式系统中，软件即使提供了正确答案或产生了正确的计算，但如果没有及时完成，其效果与产生错误答案的软件一样糟糕。例如，考虑一个控制汽车刹车的系统：一辆在碰撞后停止的汽车实际上作用有限。这样的系统在发布之前会进行严格的测试以捕捉故障，并将依赖于特定的运行时行为；因此，在开发过程中，它们可能更愿意失败（在测试阶段会被捕捉到），而不是默认采用可能有时无法满足其时序要求的策略。当然，请不要发布在现实生活中使用时停止工作的关键系统：请充分测试它们，并确保用户的安全！但也许你正在开发一个系统，如果发生糟糕的事情，你更愿意在某个地方打印“抱歉，我们搞砸了”，然后只是重新启动程序，有时这也是完全可以接受的。

`deallocate(p)`释放函数会遍历每个内存块，查看`p`是否在该块内。记住，我们的`within_block()`函数将受益于一个指针比较测试，而截至本文撰写时，标准还没有提供这个测试，所以如果你在实际中使用此代码，请确保你给自己留一个笔记，以便一旦该新功能可用就应用它。如果`p`不在我们的任何块中，那么它可能通过`::operator new()`分配，所以我们确保通过`::operator delete()`释放它，就像我们应该做的那样。

如前所述，我们的实现一旦释放内存就不会重用内存，但重用应该发生的位置已被留在了注释中（以及锁定该部分的互斥锁的代码），所以如果你想要实现内存块重用逻辑，请随意：

```cpp
      // ...
      auto allocate(std::size_t n) {
        using std::size;
        // use smallest block available
        for(std::size_t i = 0; i != size(sizes); ++i) {
            if(n < sizes[i]) {
              std::lock_guard _ { m };
              if(cur[i] < N) {
                  void *p = static_cast<char*>(blocks[i]) +
                            cur[i] * sizes[i];
                  ++cur[i];
                  return p;
              }
            }
        }
        // either no block fits or no block left
        return ::operator new(n);
      }
      void deallocate (void *p) {
        using std::size;
        for(std::size_t i = 0; i != size(sizes); ++i) {
            if(within_block(p, i)) {
              //std::lock_guard _ { m };
              // if you want to reuse the memory,
              // it's in blocks[i]
              return;
            }
        }
        // p is not in our blocks
        ::operator delete(p);
      }
  };
  // ...
```

由于这是一种客户端代码按需使用的特殊分配形式，我们将使用分配运算符的特殊重载。正如预期的那样，这些重载将是基于要使用的`ChunkSizedAllocator`对象参数的模板：

```cpp
template <int N, auto ... Sz>
  void *operator new(std::size_t n, ChunkSizedAllocator<
      N, Sz...
  > &chunks) {
      return chunks.allocate(n);
  }
template <int N, auto ... Sz>
  void operator delete (void *p, ChunkSizedAllocator<
      N, Sz...
  > &chunks) {
      return chunks.deallocate(p);
  }
// new[] and delete[] left as an exercise ;)
```

现在，我们已经编写了这些分配设施，但我们需要测试它们，因为我们需要看到这种方法的益处。

### 测试 ChunkSizedAllocator

我们现在将编写一个简单的测试程序，该程序使用一个具有适当大小类别的`ChunkSizedAllocator`对象，然后以应该对我们类有益的方式分配和释放适合这些类别的对象大小。通过这样做，我们假设这个类的用户这样做是为了从先验已知的大小类别中受益。还可以进行其他测试，以验证代码在请求不适当的大小或在存在抛出构造函数的情况下行为，例如，所以请随意编写比我们为执行速度相关讨论提供的更详尽的测试框架。

在本章前面用于测试基于大小的竞技场的`test()`函数将再次在这里使用。参见该部分以了解其工作原理。

编写一个良好的测试程序来验证分配和释放各种大小对象的程序的行为并非易事。我们将要做的是使用一个`dummy<N>`类型，其对象在内存中将各自占用`N`字节的空间（由于我们将使用`char[N]`数据成员来获取这个结果，我们知道对于所有有效的`N`值，`alignof(dummy<N>)==1`）。

我们还将编写两个不同的`test_dummy<N>()`函数。每个这样的函数都将分配并构造`dummy<N>`对象，并设置相关的销毁然后释放代码，但一个将使用标准库实现的分配运算符，另一个将使用我们的重载。

你会注意到我们的两个`test_dummy<N>()`函数都返回一对值：一个将是分配对象的指针，另一个将是销毁和释放该对象的代码。由于我们将在此存储信息，我们需要这些对是共享公共类型的抽象，这解释了我们为什么使用`void*`作为地址和`std::function<void(void*)>`作为销毁代码。我们需要`std::function`或类似的东西：函数指针不足以作为销毁代码，因为销毁代码可以是状态化的（我们有时需要记住用于管理分配的对象）。

这些工具的代码如下：

```cpp
#include <chrono>
#include <utility>
#include <functional>
template <class F, class ... Args>
  auto test(F f, Args &&... args) {
      using namespace std;
      using namespace std::chrono;
      auto pre = high_resolution_clock::now();
      auto res = f(std::forward<Args>(args)...);
      auto post = high_resolution_clock::now();
      return pair{ res, post - pre };
  }
template <int N> struct dummy { char _[N] {}; };
template <int N> auto test_dummy() {
  return std::pair<void *, std::function<void(void*)>> {
      new dummy<N>{},
      [](void *p) { delete static_cast<dummy<N>*>(p); }
  };
}
template <int N, class T> auto test_dummy(T &alloc) {
  return std::pair<void *, std::function<void(void*)>> {
      new (alloc) dummy<N>{},
&alloc { ::operator delete(p, alloc); }
  };
}
// ...
```

最后，我们必须编写测试程序。我们将逐步讨论这个程序，以确保我们掌握过程中涉及的所有细微差别。

我们程序首先为`ChunkSizedAllocator`对象确定一个`N`的值，以及内存管理器要使用的`Sz...`大小类别（我选择的`N`的值是任意的）。我故意使用了一个非 2 的幂的大小类别，以表明这些值被适当地“向上取整”到下一个 2 的幂：`62`的大小请求在构建我们的类型的数据成员`sizes`时被转换为`64`。然后我们构建这个对象，并将其命名为`chunks`，因为……好吧，为什么不呢？

```cpp
// ...
#include <print>
#include <vector>
int main() {
  using namespace std;
  using namespace std::chrono;
  constexpr int N = 100'000;
  using Alloc = ChunkSizedAllocator<
      N, 32, 62 /* 64 */, 128
  >;
  Alloc chunks; // construct the ChunkSizedAllocator
  // ...
```

接下来的测试对于标准库和我们的专用设施具有相同的形式。让我们详细看看：

1.  我们创建了一个名为`ptrs`的`std::vector`对象对，填充了默认值（空指针和非可调用函数），用于三个大小类别的`N`个对象。这确保了`std::vector`对象使用的空间分配在我们测量之前（在传递给`test()`的 lambda 表达式执行之前）进行，并且不会干扰它们。请注意，每个测试的 lambda 都是可变的，因为它需要修改捕获的`ptrs`对象。

1.  对于三个大小类别中的每一个，我们随后分配适合该类别的`N`个对象，并通过返回的`pair`记住该对象的地址以及稍后正确终结它的代码。

1.  然后，为了结束每个测试，我们使用每个对象的 finalization 代码，将其销毁并重新分配。

幸运的是，这听起来比实际情况要糟糕。一旦测试运行完成，我们就打印出每个测试的执行时间，以微秒为单位：

```cpp
  // ...
  auto [r0, dt0] = test([ptrs = std::vector<
      std::pair<
         void*, std::function<void(void*)>
      >>(N * 3)]() mutable {
      // allocation
      for(int i = 0; i != N * 3; i += 3) {
        ptrs[i] = test_dummy<30>();
        ptrs[i + 1] = test_dummy<60>();
        ptrs[i + 2] = test_dummy<100>();
      }
      // cleanup
      for(auto & p : ptrs)
        p.second(p.first);
      return std::size(ptrs);
  });
  auto [r1, dt1] = test([&chunks, ptrs = std::vector<
      std::pair<
        void*, std::function<void(void*)>
      >>(N * 3)]() mutable {
      // allocation
      for(int i = 0; i != N * 3; i += 3) {
        ptrs[i] = test_dummy<30>(chunks);
        ptrs[i + 1] = test_dummy<60>(chunks);
        ptrs[i + 2] = test_dummy<100>(chunks);
      }
      // cleanup
      for(auto & p : ptrs)
         p.second(p.first);
      return std::size(ptrs);
  });
   std::print("Standard version : {}\n",
              duration_cast<microseconds>(dt0));
  std::print("Chunked version  : {}\n",
              duration_cast<microseconds>(dt1));
}
```

好吧，所以这稍微有点复杂，但希望是有教育意义的。这值得麻烦吗？嗯，这取决于你的需求。

当我用相同的在线 gcc 15 编译器运行此代码，并且使用与基于大小的区域相同的-O2 优化级别时，标准库版本报告的执行时间为 13,360 微秒，而“分块”版本报告的时间为 12,032 微秒，相当于标准版本执行时间的 90.05%。只要我们记住，在`chunks`对象的构造函数中进行的初始分配没有被测量：这里的想法是表明，当时间重要时，我们可以节省时间，并且当我们不急于求成时，我们愿意为此付费。

重要的是要记住，这种实现不会重用内存，但标准版本会这样做，这意味着我们的加速可能被功能性的损失所抵消（如果这是你需要的功能的话）。在我进行的测试中，锁定`std::mutex`对象或不锁定它对加速有显著影响，所以（a）根据你的平台，你可能会有更好的同步机制可供选择，并且（b）这种实现可能过于天真，如果`deallocate()`成员函数也需要锁定`std::mutex`对象，那么它可能无法带来任何好处。

当然，可以对这种（相当学术的）版本进行相当多的优化，我邀请亲爱的读者们这样做（并且每一步都测试结果！）本节的目的更多的是为了展示（a）基于块大小的分配是可以实现的，（b）从架构的角度来看如何实现，以及（c）指出沿途的一些风险和潜在陷阱。

那很有趣，不是吗？

# 摘要

作为提醒，在本章中，我们通过一个具体的例子（基于大小的特定使用模式的区域）考察了基于区域的分配，并看到我们可以从中获得显著的结果，然后看到了另一个使用预分配内存块的用例，我们从其中挑选了放置对象的块，再次看到了一些好处。这些技术展示了控制内存管理的新方法，但它们绝对不是对这一主题的全面讨论。说实话，整本书都不可能对这一主题进行全面论述，但希望它能给我们一些启发！

我们旅程的下一步将是扩展本章中看到的技巧，并编写一些实际上不是垃圾回收器但在某些方面较弱且在某些方面更好的内容：延迟回收内存区域。这将是我们开始讨论容器内存管理之前的最后一步。
