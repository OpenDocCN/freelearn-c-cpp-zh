

# 第十五章：当代问题

我们即将结束这段旅程，亲爱的读者。在这本书的过程中，我们探讨了 C++ 对象模型的基本方面，并讨论了低级编程的危险之处。我们通过 RAII 习语研究了 C++ 中资源管理的根本，了解了智能指针的使用方法，并探讨了如何编写此类类型。我们还掌握了可用的内存分配函数（我们以多种方式做到了这一点！），并编写了能够自行管理内存以及通过其他对象或类型（包括分配器）来管理内存的容器。

那是一次相当的经历！

我们还需要讨论什么？嗯，很多……但是一本书能包含的内容是有限的。因此，为了总结我们对 C++ 内存管理的讨论，我想我们可以聊一聊（是的，亲爱的读者，就你和我）一些当代 C++ 内存管理中的有趣话题。是的，一些最近才标准化（截至本书写作时）的事情，大多数（如果不是所有）库还没有实现它们，以及标准委员会正在积极工作的内容。

重要的是要看看 C++ 当前的样子以及它可能近期的样子，因为该语言仍在不断进化，而且速度相当快：每三年就会发布一个新的 C++ 标准版本，这种情况自 2011 年以来一直如此。C++ 的进化对于一些人来说太慢，对于另一些人来说又太快，但它是不懈的（我们称这种发布节奏为“火车模型”，以强调其持续的步伐），并为我们所热爱的这种语言带来了定期的进步和创新。

截至 2025 年初，C++23 是一个新采用的标准，于 2024 年 11 月正式化（是的，我知道：ISO 流程确实需要一些时间），委员会正在讨论旨在 C++26（是的，已经！）和 C++29 的提案。

本章我们将讨论的与内存管理相关的话题，要么是 C++23 标准中我们尚未在本书中讨论的方面，要么是随着本章的编写正在进行讨论的，即将到来的标准中的某些话题。亲爱的读者，请注意，你现在将读到的内容可能会以你将读到的方式成为现实，但也可能在 C++ 标准委员会的讨论和辩论后以另一种形式出现……或者最终可能永远不会出现。

即使这些话题最终没有以最初讨论的形式进入 C++ 标准，你也会知道它们已经被讨论过，以及它们旨在解决的问题，并且这些特性可能在某个时刻成为语言的一部分。谁知道呢；也许你会有顿悟，找到将这些想法中的一个转化为提案，然后 C++ 标准委员会将讨论并采纳。

在本章中，我们将涵盖以下主题：

+   明确地开始一个或多个对象的生命周期，而不使用它们的构造函数

+   简单重定位：它的含义以及标准委员会试图以何种方式解决它

+   类型感知的分配和释放函数：它们会做什么以及如何从中受益

本章我们将通过解决我们试图解决的问题的视角来介绍这些新特性（或即将推出的特性）。这种方法的意图是清楚地表明这些特性解决实际问题，并将帮助真正的程序员更好地完成工作。

我希望这一章能给您提供一个有趣的（尽管不是详尽的）当代内存管理和相关设施问题的见解，这些问题与 C++相关。

关于本章代码示例的说明

如果您尝试编译本章的示例，尊敬的读者，您可能会因为一些示例无法编译而感到沮丧，而其他示例可能需要一段时间才能编译，或者永远无法编译。对于这样一个章节来说，这种情况是正常的：我们将讨论最近添加到 C++语言中的特性组合（最近到以至于在撰写本书时尚未实现）以及 C++标准委员会正在讨论的特性。因此，将这些示例作为说明，并根据特性的更正式形式进行调整。

# 技术要求

您可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter15`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter15).

# 不使用构造函数开始对象的生命周期

考虑一个程序，它从流中消费序列化数据并试图从该数据中创建对象的情况。以下是一个示例：

```cpp
#include <fstream>
#include <cstdint>
#include <array>
#include <memory>
#include <string_view>
struct Point3D {
   float x{}, y{}, z{};
   Point3D() = default;
   constexpr Point3D(float x, float y, float z)
      : x{ x }, y{ y }, z{ z } {
   }
};
// ...
// reads at most N bytes from file named file_name and
// writes these bytes into buf. Returns the number of
// bytes read (postcondition: return value <= N)
//
template <int N>
   int read_from_stream(std::array<unsigned char, N> &buf,
                        std::string_view file_name) {
   // ...
}
// ...
```

如您所见，在这个例子中，我们有`Point3D`类。此类对象代表一组*x, y, z*坐标。我们还有一个`read_from_stream<N>()`函数，它从文件中读取字节。该函数然后将最多`N`字节存储到通过引用传递的参数`buf`中，并返回读取的字节数（可能为零，但永远不会超过`N`）。

为了这个例子，我们将假设我们计划从中读取的文件已知包含序列化的`Point3D`对象的二进制形式，相当于按组三序列化的`float`类型对象。现在，考虑以下程序，它从名为`some_file.dat`的文件中消费最多四个`Point3D`类型对象的字节表示：

```cpp
// ...
#include <print>
#include <cassert>
using namespace std::literals;
int main() {
   static constexpr int NB_PTS = 4;
   static constexpr int NB_BYTES =
      NB_PTS * sizeof(Point3D);
   alignas(Point3D)
      std::array<unsigned char, NB_BYTES> buf{};
   if (int n = read_from_stream<NB_BYTES>(
          buf, "some_file.dat"sv
       ); n != 0) {
      // print out the bytes: 0-filled left, 2
      // characters-wide, hex format
      for (int i = 0; i != n; ++i)
         std::print("{:0<2x} ", buf[i]);
      std::println();
      // if we want to treat the bytes as Point3D objects,
      // we need to start the lifetime of these Point3D
      // objects. If we do not, we are in UB territory (it
      // might work or it might not, and even if it works
      // we cannot count on it)
      const Point3D* pts =
         std::start_lifetime_as_array(buf.data(), n);
      assert(n % 3 == 0);
      for (std::size_t i = 0;
           i != n / sizeof(Point3D); ++i)
         std::print("{} {} {}\n",
                    pts[i].x, pts[i].y, pts[i].z);
   }
}
```

这个示例程序从文件中读取字节到足够容纳四个`Point3D`类型对象字节的`std::array`对象中，首先确保如果这个数组要容纳该类型的对象，其对齐方式是适当的。这种对齐考虑是至关重要的，因为我们计划在读取这些字节后将其作为该类型的对象来处理。

这个示例的目的是，一旦读取了字节，程序员可以确信（嗯，尽可能确信）对于一些假设的`Point3D`对象，所有的字节都是正确的，但仍然不能使用这些对象，因为它们的生命周期尚未开始。

这种情况通常会让许多 C 程序员微笑，而一些 C++程序员则会皱眉：C++对象模型对程序施加了约束，使得程序在对象的生命周期之外使用对象成为**UB**（见*第二章*），即使所有字节都是正确的，并且对齐约束得到了遵守，而 C 语言则不那么严格。为了使用我们刚才用来从该文件读取内容的缓冲区的内容，我们的传统选项如下：

+   要遍历字节数组，将适当大小的字节子集写入`float`类型的对象中，然后调用`Point3D`对象的构造函数并将它们放入另一个容器中。

+   将字节数组`reinterpret_cast`为`Point3D`对象数组，并寄希望于最好，这可能导致可能或可能不工作的代码，并且由于是 UB，因此无论如何都不具有可移植性（甚至不是在给定编译器的版本之间）。使用我们的`Point3D`对象，它可能会给出人们希望得到的结果，但将这些替换为，比如说，来自标准库的`std::complex<float>`对象（这种类型可能具有与我们的`Point3D`类型相似的内结构）……嗯，谁知道会发生什么呢？

+   将字节数组`std::memcpy()`到自身，将返回值转换为`Point3D*`类型，并使用得到的指针作为`Point3D`对象数组来使用。这实际上是有效的（`std::memcpy()`函数是允许启动对象生命周期的函数集的一部分）。当然，存在创建实际字节副本的风险（这将浪费执行时间）；据说某些标准库能够识别该模式，并且表现得就像调用是一个 no-op 一样，但这是一个可以启动对象生命周期的特殊类型的 no-op。

然而，这些选项似乎都不真正令人满意，因此需要一个更干净、不依赖于编译器特定优化的解决方案。为此，C++23 标准引入了一组`constexpr`函数（附带一些重载），它们被称为`std::start_lifetime_as_array<T>(p,n)`和`std::start_lifetime_as<T>(p)`。这两个都是告知编译器字节是正确的，并且要考虑引用对象的生存期已开始的便携式魔法 no-op 函数。

当然，如果出于某种原因，指针的目标有非平凡的析构函数，你应该确保你的代码在适当的时候调用这些析构函数。预期这种情况很少见且不寻常。由于我们从某些数据源中消耗了原始字节并将这些字节转换成了对象，因此结果对象拥有资源的可能性相对较小。当然，一旦它们的生命周期开始，这些对象可以获取资源。让我们坦诚地说，亲爱的读者；如果 C++程序员不是富有创造力的，那还有什么？

这套`std::start_lifetime_...`函数预计将成为网络程序员的福音，尤其是这些人。他们经常接收到格式良好的字节序列的数据帧，他们需要将其转换为对象以进行进一步处理。这些函数也预计将对从文件中消耗字节以形成聚合的程序有用。许多程序员认为，只需将字节读取到字节数组中并将该数组转换为预期的类型（或其数组）就足以访问其中的（假设的）对象（或对象），当他们的 C++代码开始出现意外行为时感到惊讶。C++是一种系统编程语言，由这些`std::start_lifetime_...`函数组成的集合填补了可能表现不佳的空白。

当然，由于涉及的风险，这些函数形成了一个非常锋利的工具集：以这种方式开始生命周期的非平凡可析构对象尤其可疑，你必须高度信任提供字节以手动和显式启动对象生命周期的任何设施。因此，这些设施应该非常小心地使用。

关于本节的注意事项：截至本文撰写时，还没有主要的编译器实现这些函数，尽管它们已经被标准化，并且是 C++23 的一部分。也许在你读到这篇文章的时候，它们已经被实现了，谁知道呢？

# 简单重定位

如您所知，亲爱的读者，C++在编程社区中是那些我们需要从计算机或任何感兴趣的硬件平台上获取最大性能的语言之一。该语言的一些信条可以概括为“你不应为未使用的东西付费”和“不应有更低级语言的空间（除了偶尔的汇编代码）”，毕竟。后者解释了上一节中`std::start_lifetime_...`函数的重要性。

这可能就是为什么，当显而易见我们可以比现在做得更好，在执行速度方面，这成为 C++程序员社区普遍感兴趣的话题，特别是 C++标准委员会的成员。我们都把语言的这些核心信条铭记在心。

我们可以做得更好的一个例子是，当我们遇到可以实际用 `std::memcpy()` 调用替换将源对象移动到目标对象，然后销毁原始对象的类型时：直接复制字节数组比执行一系列移动和析构函数更快（即使它不是，可能需要在您的 `std::memcpy()` 实现上做一些工作），尽管移动赋值和析构函数组合起来速度很快。

结果表明，有许多类型可以考虑这种优化，包括 `std::string`、`std::any` 和 `std::optional<T>`（取决于 `T` 的类型），例如前一部分中的 `Point3D` 类，任何未定义六个特殊成员函数的类型（包括基本类型），等等。

为了理解其影响，考虑以下名为 `resize()` 的自由函数，它模拟了某些容器 `C` 的 `C::resize()` 成员函数，该容器管理连续内存，例如本书中看到的各种形式的我们的 `Vector<T>` 类型。此函数将 `arr` 从 `old_cap`（旧容量）调整大小到 `new_cap`（新容量），并在末尾填充默认的 `T` 对象。该函数中高亮显示的行是我们这里感兴趣的部分：

```cpp
//
// This is not a good function interface, but we want to
// keep the example relatively simple
//
template <class T>
  void resize
    (T *&arr, std::size_t old_cap, std::size_t new_cap) {
    //
    // we could deal with throwing a default constructor
    // but it would complicate our code a bit and these
    // added complexities, worthwhile as they are, are
    // besides the point for what we are discussing here
    //
    static_assert(
      std::is_nothrow_default_contructible_v<T>
    );
    //
    // sometimes, there's just nothing to do
    //
    if(new_cap <= old_cap) return arr;
    //
    // allocate a chunk of raw memory (no object created)
    //
    auto p = static_cast<T*>(
      std::malloc(new_cap * sizeof(T))
    );
    if(!p) throw std::bad_alloc{};
    // ...
```

在这个阶段，我们已经准备好复制（或移动）对象：

```cpp
    // ...
    //
    // if move assignment does not throw, be aggressive
    //
    if constexpr(std::is_nothrow_move_assignable_v<T>) {
      std::uninitialized_move(arr, arr + old_cap, p);
      std::destroy(arr, arr + old_cap);
    } else {
      //
      // since move assignment could throw, let's be
      // conservative and copy instead
      //
      try {
        std::uninitialized_copy(arr, arr + old_cap, p);
        std::destroy(arr, arr + old_cap);
      } catch (...) {
        std::free(p);
        throw;
      }
    }
    //
    // fill the remaining space with default objects
    // (remember: we statically asserted that T::T() is
    // non-throwing)
    //
    std::uninitialized_default_construct(
      p + old_cap, p + new_cap
    );
    //
    // replace the old memory block (now without objects)
    // with the new one
    //
    std::free(arr);
    arr = p;
  }
```

观察该函数中高亮显示的行，尽管 `std::uninitialized_move()` 后跟 `std::destroy()` 的组合提供了一条快速路径，但我们甚至可以比这更快，用一个 `std::memcpy()` 调用替换一系列移动赋值运算符和一系列析构函数调用。

我们如何实现这一点？嗯，Arthur O’Dwyer、Mingxin Wang、Alisdair Meredith 和 Mungo Gill 等人提出了许多相互竞争的提案。每个提案都有其优点，但这些提案有以下共同因素：

+   在编译时提供一种测试类型是否具有“简单可重新定位性”的方法，例如，一个 `std::is_trivially_relocatable_v<T>` 特性。

+   提供一个实际重新定位对象的函数，例如 `std::relocate()` 或 `std::trivially_relocate()`，该函数接受源指针和目标指针作为参数，并将源对象重新定位到目标位置，结束原始对象的生命周期，然后开始新对象的生命周期

+   提供一种方法在编译时标记类型为简单可重新定位，例如通过关键字或属性

+   提供规则在编译时推断类型的简单可重新定位性

具体细节可能因方法而异，但如果我们假设这些工具，相同的 `resize()` 函数可以通过对之前提出的实现进行轻微调整从简单的重新定位中受益：

```cpp
template <class T>
   void resize
     (T * &arr, std::size_t old_cap, std::size_t new_cap) {
      static_assert(
         std::is_nothrow_default_contructible_v<T>
      );
      if(new_cap <= old_cap) return arr;
      auto p = static_cast<T*>(
         std::malloc(new_cap * sizeof(T))
      );
      if(!p) throw std::bad_alloc{};
      //
      // this is our ideal case
      //
      if constexpr (std::is_trivially_relocatable_v<T>) {
         // equivalent to memcpy() plus consider the
         // lifetime of objects in arr, arr + old_cap)
         // finished and the lifetime of objects in
         // [p, p + old_cap) started
         //
         // note: this supposes that the trait
         // std::is_trivially_relocatable<T>
         // implies std::is_trivially_destructible<T>
         std::relocate(arr, arr + old_cap, p);
      //
      // if move assignment does not throw, be aggressive
      //
      } else if constexpr(
           std::is_nothrow_move_assignable_v<T>
      ){
         std::uninitialized_move(arr, arr + old_cap, p);
         std::destroy(arr, arr + old_cap);
      } else {
         // ... see previous code example for the rest
      }
   }
```

这种看似简单的优化已被报道提供了相当大的好处，有些人声称在常见情况下速度提高了高达 30%，但这是一项实验性工作，如果提议（正如我们预期的那样）合并成将被集成到 C++标准中的东西，我们预计会有更多的基准测试出现。

这种潜在的速度提升是 C++语言旨在实现的目标的一部分，因此我们可以合理地预期微小的迁移性将在可预见的未来成为现实。问题是“如何”：编译器应该如何检测微小的迁移性属性？当默认的微小迁移性推导规则不适用时，程序员应该如何在自己的类型上表明这种属性？

截至 2025 年 2 月，标准委员会投票将微小的迁移纳入将成为 C++26 标准的范畴。这意味着我们可以预期，一些用 C++语言先前标准编译的程序，在用 C++26 重新编译后，可以仅通过不修改任何源代码行就能运行得更快。

# 类型感知的分配和释放函数

我们在本章的最后讨论了关于内存管理和与对象生命周期相关的优化机会的新方法，即类型感知的分配和释放函数。这是一种针对用户代码可能希望以某种方式使用有关正在进行的分配（以及最终构建）的类型信息来指导分配过程的分配函数的新方法。

当我们在[*第九章*中描述`T::operator delete()`将`T*`作为参数传递而不是抽象的`void*`时，我们看到了这些特性的一个方面，这是因为它因此负责对象的最终化和其底层存储的释放。我们看到，在某些情况下，这揭示了有趣的优化机会。

对于 C++26 正在讨论的是一组新的`operator new()`和`operator delete()`成员函数，以及接受`std::type_identity<T>`对象作为第一个参数的免费函数，对于某些类型`T`，这些函数将引导选定的操作符针对该类型`T`执行一些特殊行为。请注意，这些类型感知的分配函数实际上是分配函数：它们不执行构造，它们的释放对应函数也不执行最终化。

`std::type_identity<T>`特性是什么？

表达式`typename std::type_identity<T>::type`对应于`T`。好吧，这似乎很简单。那么，这个特性在当代 C++编程中扮演什么角色呢？实际上，特性`std::type_identity<T>`，自 C++20 引入以来，是一种通常用于在泛型函数中提供对参数类型推导额外控制的工具。

例如，具有函数签名 `template <class T> void f(T,T)`，您可以调用 `f(3,3)`，因为两个参数都是同一类型，但不能调用 `f(3,3.0)`，因为 `int` 和 `double` 是不同的类型。但话虽如此，通过将任一参数类型替换为 `std::type_identity_t<T>`，您可以调用 `f(3,3.0)`，并且由于 `T` 将根据另一个参数（类型为 `T` 的参数）推导出来，因此该类型将用于另一个参数（类型为 `std::type_identity_t<T>` 的参数）。这将导致两个参数都是 `int` 或 `double`，具体取决于哪个参数是 `T` 类型。

使用 `std::type_identity<T>`（而不是 `std::type_identity_t<T>`）而不是 `T` 作为类型感知分配函数中第一个参数的类型，是为了清楚地表明我们正在使用这个特定的特殊重载的 `operator new()`，并且这不是一个意外或调用此分配函数的其他特殊形式，例如在*第九章*中描述的那些。

这意味着您可以通过以下函数签名为特定类 `X` 提供专门的分配函数：

```cpp
#include <new>
#include <type_traits>
void* operator new(std::type_identity<X>, std::size_t n);
void operator delete(new X, for example, the specialized form will be preferred to the usual form of operator new() and operator delete(), being assumed to be more appropriate unless the programmer takes steps to prevent it.
It also means that, given a specialized allocation algorithm that applies to type `T` only if `special_alloc_alg<T>` is satisfied, you could provide allocation functions that use this specialized algorithm for type `T` through the following function signatures:

```

包含 `<new>`

包含 `<type_traits>`

模板 <class T> requires special_alloc_alg<T>

void* operator new(std::type_identity<T>, std::size_t n);

模板 <class T> requires special_alloc_alg<T>

void operator delete(X and Y, but that algorithm does not apply to other classes, such as Z:

```cpp
#include <concepts>
#include <type_traits>
class X { /* ... */ };
class Y { /* ... */ };
class Z { /* ... */ };
template <class C>
   concept cool_alloc_algorithm =
      std::is_same_v<C, X> || std::is_same_v<C, Y>;
template <class T> requires cool_alloc_algorithm<T>
  void* operator new(std::type_identity<T>, std::size_t n){
     // apply the cool allocation algorithm
  }
template <class T> requires cool_alloc_algorithm<T>
  void operator delete(std::type_identity<T>, void* p) {
     // apply the cool deallocation algorithm
  }
#include <memory>
int main() {
   // uses the "cool" allocation algorithm
   auto p = std::make_unique<X>();
   // uses the standard allocation algorithm
   auto q = std::make_unique<Z>();
} // uses the standard deallocation algorithm for q
  // uses the "cool" deallocation algorithm for p
```

类型感知分配函数也可以是成员函数重载，导致算法适用于定义这些函数的类，以及这些类的派生类。

考虑以下示例，该示例灵感来源于在[`wg21.link/p2719`](https://wg21.link/p2719)中找到的更复杂示例，该示例描述了该特性的提案：

```cpp
class D0; // forward class declaration
struct B {
  // i)
  template <class T>
  void* operator new(std::type_identity<T>, std::size_t);
  // ii)
  void* operator new(std::type_identity<D0>, std::size_t);
};
// ...
```

如所述，`i)` 适用于 `B` 及其派生类，但 `ii)` 适用于已声明的类 `D0` 的特定情况，并且只有在 `D0` 确实是 `B` 的派生类时才会使用。

继续这个示例，我们现在添加三个从 `B` 派生的类，其中 `D2` 添加了 `iii)`，这是一个非类型感知的 `operator new()` 成员函数重载：

```cpp
// ...
struct D0 : B { };
struct D1 : B { };
struct D2 : B {
  // iii)
  void *operator new(std::size_t);
};
// ...
```

给定这些重载，以下是一些调用重载 `i)`、`ii)` 和 `iii)` 的表达式示例：

```cpp
// ...
void f() {
  new B;       // i) where T is B
  new D0;      // ii)
  new D1;      // i) where T is D1
  new D2;      // iii)
  ::new B;     // uses appropriate global operator new
}
```

正如您所看到的，亲爱的读者，如果类型感知分配函数被纳入 C++标准，它们将提供新的方法来控制将使用哪种内存分配算法（根据情况而定），同时仍然让用户代码保持控制，使其能够根据需要推迟到全局 `operator new()` 函数，正如前一个示例中 `f()` 函数的最后一行所示。

与 C++20 的销毁删除功能相反，该功能同时执行对象的最终化和底层存储的分配，类型感知版本的`operator new()`和`operator delete()`只是分配函数，截至本文撰写时，没有计划提供销毁删除的类型感知版本。

摘要

在本章中，我们通过`std::start_lifetime_...`函数窥见了 C++23 的未来，但这些函数尚未被任何主要编译器实现。我们还研究了 C++未来的可能（但尚未官方）部分，包括对平凡可重定位性的潜在支持以及引入类型感知版本的`operator new()`和`operator delete()`的可能性。

随着每一步的迈进，C++成为了一个更丰富、更通用的语言，我们可以用它做更多的事情，并以更精确的方式表达我们的想法。C++是一种提供对我们程序行为更多控制的编程语言。尽管 C++今天如此强大，它让像我们这样的程序员如此强大，但本章表明我们仍然可以继续进步。

我们已经到达了旅程的终点，至少目前是这样。希望这次旅行对您来说既有趣又愉快，尊敬的读者，并且您在旅途中学到了一些东西。我还希望这里讨论的一些想法能帮助您完成任务，丰富您对 C++编程的视角。

感谢您一直陪伴我。希望您未来的旅程愉快，就像我希望这本书能让您的工具箱更完善，并且您会继续独立探索。一路顺风。

```cpp

```

# 附录：

# 您应该知道的事情

本书假设读者具备一些技术背景，这些背景可能不被一些人认为是“常识”。在以下章节中，您可能会找到有助于您充分利用本书的补充信息。根据需要参考，并享受阅读！

如果您认为您已经很好地了解了以下章节的内容，可以随意浏览，对于那些您不太熟悉的章节，可以仔细研究。您甚至可以跳过这个整个章节，在阅读本书时意识到这些主题并不是您想象中那么熟悉的情况下再回来。 

总体目标是在阅读完这本书后获得最大收益！

# 结构体和类

在 C++中，`struct`和`class`这两个词基本上意味着相同的东西，以下代码是完全合法的：

```cpp
struct Drawable {
   virtual void draw() = 0;
   virtual ~Drawable() = default;
};
class Painting : public Drawable {
   void draw() override;
};
```

这里有一些需要注意的细节：

+   C++没有像其他一些语言那样的`abstract`关键字。C++中的抽象成员函数是`virtual`的，并且用`=0`代替定义。`virtual`关键字意味着*可以被派生类特化*（`=0`部分本质上意味着*必须被特化…*）。我们经常谈论`virtual`成员函数。必须被重写的函数被称为**纯****虚**函数。

为纯虚函数提供默认实现

可以为抽象成员函数提供一个定义：这不是典型情况，但这是可能的。这在基类想要提供一个服务的默认实现，但要求派生类至少考虑提供它们自己的情况下可能很有用。以下是一个示例：

`#``include <iostream>`

`struct X { virtual int f() const =` `0; };`

`int X::f() const { return` `3; }`

`struct D : X { int f() const` `override {`

`return X::f() +` `1; }`

`};`

`void g(X &x) { std::cout << x.f() << '\``n'; }`

`int` `main() {`

`D d;`

`// X x; // 非法：X 有一个纯虚成员函数`

`g(d);`

`}`

+   C++ 类具有析构函数，用于处理对象生命周期结束时发生的情况。与许多其他流行的语言不同，C++ 中的自动和静态对象具有确定的生存期，并且在该语言中有效地使用析构函数是惯用的。在 `virtual` 成员函数中，通常会有一个 `virtual` 析构函数（这里为 `virtual ~Drawable()`），以表明在以下情况中，通过间接方式（如 `p`）使用的对象被销毁时，应该有效地销毁指向的对象（`Painting`），而不是指针静态类型表示的对象（`Drawable`）：

    ```cpp
    //
    // the following supposes that Painting is a public
    // derived class of Drawable as suggested earlier in
    // this section
    //
    Drawable *p = new Painting;
    // ...
    delete p; // <-- here
    ```

+   一个 `class` 可以从 `struct` 继承，就像 `struct` 可以从 `class` 继承一样，因为它们在结构上是等效的。主要区别在于，对于 `struct`，继承默认是 `public`（但可以使用 `protected` 或 `private` 来更改），成员也是如此，而对于 `class`，继承和成员默认是 `private`（但同样，也可以更改）。

顺便提一下，在 C++ 中，在基类（例如，`Drawable::draw()`，它是 `public`）和派生类（例如，`Painting::draw()`，它是 `private`）中有一个具有访问限定符的成员函数是完全正常的。一些其他流行的语言不允许这样做。

# std::size_t

类型 `std::size_t` 是某些无符号整型的一个别名，但实际类型可能因编译器而异（可能是 `unsigned int`、`unsigned long`、`unsigned long long` 等）。

讨论容器大小和对象在内存中占用的空间时，经常会遇到类型 `std::size_t`，这是通过 `sizeof` 运算符表示的。

# sizeof 运算符

sizeof 运算符返回对象或类型的字节大小。它在编译时进行评估，并且在本书中将被广泛使用，因为我们需要这些信息来正确分配内存块：

```cpp
auto s0 = sizeof(int); // s0 is the number of bytes in an
                       // int (parentheses required)
int n;
auto s1 = sizeof n; // s1 is the number of bytes occupied
                    // by s1, which is identical to s0.
                    // Note: for objects, parentheses are
                    // allowed but not mandated
```

对象大小是内存管理的关键组成部分之一，它影响着程序执行的速度。因此，这一点在本书中反复出现。

# 断言

“`int` 占用四个字节的存储空间。”在后一种情况下，我们有一个基于不可移植假设编写的程序，我们必须接受这个选择，但我们不希望我们的代码在那些假设不成立的平台上编译。

对于动态断言，通常使用 `<cassert>` 头文件中的 `assert()` 宏。该宏将布尔表达式作为参数，如果它评估为 `false`，则停止程序执行：

```cpp
void f(int *p) {
   assert(p); // we hold p != nullptr to be true
   // use *p
}
```

注意，许多项目在产品代码中禁用了 `assert()`，这可以通过在编译前定义 `NDEBUG` 宏来实现。因此，请确保不要在 `assert()` 中放置有副作用的表达式，因为它可能会被编译器选项移除：

```cpp
int *obtain_buf(int);
void danger(int n) {
   int *p; // uninitialized
   assert(p = obtain_buf(n)); // dangerous!!!
   // use *p, but p might be uninitialized if assert()
   // has been disabled. This is very bad
}
```

与库宏 `assert()` 相反，`static_assert` 是一种语言特性，如果其条件不满足，则阻止编译。基于前面提到的例子，其中一家公司可能基于不可移植的假设（如 `sizeof(int)==4`）构建了软件，我们可以确保代码在这些实际上不支持的平台上的编译（和做坏事）：

```cpp
static_assert(sizeof(int)==4); // only compiles if the
                               // condition holds
```

在发布软件产品之前修复错误对于开发者和用户来说都远远优于软件发布到“野外”后修复错误。因此，`static_assert` 可以被视为交付更高品质产品的强大工具。

在这本书中，我们将经常使用 `static_assert`：它没有运行时成本，并以可验证的方式记录我们的断言。这是一种基本上没有缺点特性。

# 未定义行为

**未定义行为**，通常简称为 **UB**，是由于标准没有规定特定行为的情况。在 C++ 标准中，UB 是没有要求的行为。它可能导致问题被忽略，也可能导致诊断或程序终止。关键思想是，如果你的程序有未定义行为，那么它没有按照语言的规则行事，是错误的；它的行为在你的平台上没有保证，它不能在不同的平台或编译器之间移植，也不能依赖。

一个正确编写的 C++ 程序没有未定义行为。当面对包含未定义行为的函数时，编译器可以对那个函数中的代码做几乎所有的事情，这使得从源代码中进行推理基本上是不可能的。

未定义行为是列在 *第二章* 中需要小心处理的“事项”之一。努力避免未定义行为：如果你留下它，它总是会反过来咬你。

# 类型特性

多年来，C++程序员已经开发出各种技术来推理他们类型的属性，大多数是在编译时。推断诸如*“类型`T`是`const`吗？”*或*“类型`T`是否可以平凡复制？”*等问题非常有用，尤其是在泛型代码的上下文中。这些技术产生的结构被称为`<type_traits>`头文件。

标准类型特性表达的方式随着时间的推移而标准化，从像`std::numeric_limits<T>`这样的复杂野兽，它为类型`T`提供了许多不同的服务，到更具体的服务，如`std::is_const<T>`（*类型`T`实际上是`const`吗？*）或`std::remove_const<T>`（*请给我一个类似于`T`的类型，如果有的话，不要`const`修饰*），它们产生一个单一的类型或一个单一值。实践表明，产生类型（命名为`type`）或编译时已知值（命名为`value`）的小型、单一类型特性可以被认为是“最佳实践”，并且大多数当代类型特性（包括标准特性）都是这样编写的。

自 C++14 以来，产生类型的特性有了以`_t`结尾的别名（例如，不再需要写相当痛苦的`typename std::remove_const<T>::type`咒语，现在可以写`std::remove_const_t<T>`），而自 C++17 以来，产生值的特性有了以`_v`结尾的别名（例如，不再需要写`std::is_const<T>::value`，现在可以写`std::is_const_v<T>`）。

那么，概念呢？

类型特性是一种 C++几十年来就有的编程技术，但自从 C++20 以来，我们有了**概念**，概念有点像特性（通常，它们是通过特性表达的），但在意义上更强，因为它们是类型系统的一部分。这本书并没有大量使用概念，但你（作为一个程序员）真的应该熟悉它们。它们非常强大，对当代 C++编程非常有用。

# std::true_type 和 std::false_type 特性

当表达类型特性时，标准库采用了使用`type`作为类型名称和`value`作为值名称的常见做法，如下例所示：

```cpp
// hand-made is_const<T> and remove_const<T> traits
// (please use the standard versions from <type_traits>
// instead of writing your own!)
template <class> struct is_const {
   static constexpr bool value = false; // general case
};
// specialization for const types
template <class T> struct is_const<const T> {
   static constexpr bool value = true;
};
// general case
template <class T> struct remove_const {
   using type = T;
};
// specialization for const T
template <class T> struct remove_const<const T> {
   using type = T;
};
```

事实上，许多类型特性都有布尔值。为了简化编写此类特性的任务并确保这些特性的形式是一致的，你将在`<type_traits>`头文件中找到`std::true_type`和`std::false_type`类型。这些类型可以被视为类型系统中的常量`true`和`false`的对应物。

使用这些类型，我们可以将特性如`is_const`重写如下：

```cpp
#include <type_traits>
// hand-made is_const<T> (prefer the std:: versions...)
template <class> struct is_const : std::false_type {
};
template <class T>
   struct is_const<const T> : std::true_type {
   };
```

这些类型既是便利，也是更清晰地表达思想的方式。

# std::conditional<B,T,F>特性

有时根据编译时已知的条件在两种类型之间进行选择是有用的。考虑以下示例，其中我们试图实现某种类型`T`的两个值的比较，该类型对于浮点类型和“其他”类型（如`int`）的行为不同，所有这些类型都为了简单起见而组合在一起：

```cpp
#include <cmath>
// we will allow comparisons between exact representations
// or floating point representations based on so-called tag
// types (empty classes used to distinguish function
// signatures)
struct floating {};
struct exact {};
// the three-argument versions are not meant to be called
// directly from user code
template <class T>
   bool close_enough(T a, T b, exact) {
      return a == b; // fine for int, short, bool, etc.
   }
template <class T>
   bool close_enough(T a, T b, floating) {
      // note: this could benefit from more rigor, but
      // that's orthogonal to our discussion
      return std::abs(a - b) < static_cast<T>(0.000001);
   }
// this two-argument version is the one user code is
// meant to call
template <class T>
   bool close_enough(T a, T b) {
      // OUR GOAL: call the "floating" version for types
      // float, double and long double; call the "exact"
      // version otherwise
   }
```

你可能会注意到，在我们的`close_enough()`函数中，我们没有为类型`exact`和`floating`命名参数。这没关系，因为我们根本没使用这些对象；这些参数的原因是确保两个函数具有不同的签名。

`<type_traits>`头文件中有一个`std::is_floating_point<T>`特性，对于浮点数其值为`true`，否则为`false`。如果没有这个特性，我们可以自己编写：

```cpp
// we could write is_floating_point<T> as follows
// (but please use std::is_floating_point<T> instead!
template <class> struct is_floating_point
   : std::false_type {}; // general case
// specializations
template <> struct is_floating_point<float>
   : std::true_type {};
template <> struct is_floating_point<double>
   : std::true_type {};
template <> struct is_floating_point<long double>
   : std::true_type {};
// convenience to simplify user code
template <class T>
   constexpr bool is_floating_point_v =
      is_floating_point<T>::value;
```

我们可以使用这个特性来做出决定。然而，我们不想在这里做出运行时决定，因为类型`T`的本质在编译时是完全已知的，而且没有人愿意为比较整数时的一条分支指令付费！

可以使用`std::conditional<B,T,F>`特性来做出这样的决定。如果我们自己编写，它可能看起来像这样：

```cpp
// example, home-made conditional<B,T,F> type trait
// (prefer the std:: version in <type_traits>)
// general case (incomplete type)
template <bool, class T, class F> struct conditional;
// specializations
template < class T, class F>
   struct conditional<true, T, F> {
      using type = T; // constant true, picks type T
   };
template < class T, class F>
   struct conditional<false, T, F> {
   using type = F; // constant true, picks type F
};
// convenience to simplify user code
template <bool B, class T, class F>
   using conditional_t = typename conditional<B,T,F>::type;
```

给定这个特性，我们可以在编译时根据编译时布尔值选择两种类型中的一种，这正是我们试图做到的：

```cpp
// ...
// this version will be called from user code
template <class T>
   bool close_enough(T a, T b) {
      return close_enough(
         a, b, conditional_t<
            is_floating_point_v<T>,
            floating,
            exact
         > {}
      );
   }
```

这样理解这个调用：`close_enough()`调用中的第三个参数（在我们的双参数用户界面`close_enough()`函数中找到）将是一个`floating`类型的对象或一个`exact`类型的对象，但确切类型将在编译时根据`is_floating_point_v<T>`编译时常量的值来选择。最终结果是实例化这两个空类中的一个对象，调用适当的算法，让函数内联来完成其余工作并优化整个框架。

# 算法

C++标准库包含了许多精华，其中之一是一组算法。这些函数中的每一个都执行一个非常好的循环所能完成的任务，但具有特定的名称、复杂度保证和优化。因此，让我们说我们编写以下代码：

```cpp
int vals[]{ 2,3,5,7,11 };
int dest[5];
for(int i = 0; i != 5; ++i)
   dest[i] = vals[i];
```

在 C++中，编写以下代码是惯例：

```cpp
int vals[]{ 2,3,5,7,11 };
int dest[5];
[begin,end), meaning that for all algorithms, the beginning iterator (here, begin(vals)) is included and the ending iterator (here, end(vals)) is excluded, making [begin,end) a half-open range. All algorithms in <algorithm> and in its cousin header, <numeric>, follow that simple convention.
What about ranges?
The `<ranges>` library is a major addition to the C++ standard library since C++20 and can sometimes be used to lead to even better code than the already tremendous `<algorithm>` library. This book does not use ranges much, but that does not mean this library is not wonderful, so please feel free to use it and investigate ways through which it can be used to make your code better.
Functors (function objects) and lambdas
It is customary in C++ to use **functors**, otherwise called **function objects**, to represent stateful computations. Think, for example, of a program that would print integers to the standard output using an algorithm:

```

#include <iostream>

#include <algorithm>

#include <iterator>

using namespace std;

void display(int n) { cout << n << ' '; }

int main() {

int vals[]{ 2,3,5,7,11 };

for_each(begin(vals), end(vals), display);

}

```cpp

 This small program works fine, but should we want to print elsewhere than on the standard output, we would find ourselves in an unpleasant situation: the `for_each()` algorithm expects a unary function in the sense of “function accepting a single argument” (here, the value to print), so there’s no syntactic space to add an argument such as the output stream to use. We could “solve” this issue through a global variable, or using a different function for every output stream, but that would fall short of a reasonable design.
If we replace the `display` function with a class, which we’ll name `Display` to make them visually distinct, we end up with the following:

```

#include <iostream>

#include <algorithm>

#include <iterator>

#include <fstream>

using namespace std;

class Display {

ostream &os;

public:

Display(ostream &os) : os{ os } {

}

void operator()(int n) const { os << n << ' '; }

};

int main() {

int vals[]{ 2,3,5,7,11 };

// 在标准输出上显示

for_each(begin(vals), end(vals), Display{ cout });

ofstream out{"out.txt"};

// 将内容写入文件 out.txt

for_each(begin(vals), end(vals), Display{ out });

}

```cpp

 This leads to nice, readable code with added flexibility. Note that, conceptually, lambda expressions are functors (you can even use lambdas as base classes!), so the previous example can be rewritten equivalently as follows:

```

#include <iostream>

#include <algorithm>

#include <iterator>

#include <fstream>

using namespace std;

int main() {

int vals[]{ 2,3,5,7,11 };

// 在标准输出上显示

for_each(begin(vals), end(vals), [](int n) {

cout << n << ' ';

});

ofstream out{"out.txt" };

// write to file out.txt

for_each(begin(vals), end(vals), &out {

out << n << ' ';

});

}

```cpp

 Lambdas are thus essentially functors that limit themselves to a constructor and an `operator()` member function, and this combination represents the most common case by far for such objects. You can, of course, still use full-blown, explicit functors if you want more than this.
Friends
C++ offers an access qualifier that’s not commonly found in other languages and is often misunderstood: the `friend` qualifier. A class can specify another class or a function as one of its friends, giving said `friend` qualifier full access to all of that class’s members, including those qualified as `protected` or `private`.
Some consider `friend` to break encapsulation, and indeed it can do this if used recklessly, but the intent here is to provide privileged access to specific entities rather than exposing them as `public` or `protected` members that were not designed to that end, leading to an even wider encapsulation breakage.
Consider, for example, the following classes, where `thing` is something that is meant to be built from the contents of a file named `name` by a `thing_factory` that’s able to validate the file’s content before constructing the `thing`:

```

class thing {

thing(string_view); // note: private

// ... various interesting members

// thing_factory can access private members of

// class thing

friend class thing_factory;

};

// in case we read an incorrect file

class invalid_format{};

class thing_factory {

// ... various interesting things here too

string read_file(const string &name) const {

ifstream in{ name };

// consume the file in one fell swoop, returning

// the entire contents in a single string

return { istreambuf_iterator<char>{ in },

istreambuf_iterator<char>{ } };

}

bool is_valid_content(string_view) const;

public:

thing create_thing_from(const string &name) const {

auto contents = read_file(name);

if(!is_valid_content(contents))

throw invalid_format{};

// note: calls private thing constructor

return { contents };

}

};

```cpp

 We do not want the whole world to be able to call the `private`-qualified `thing` constructor that takes an arbitrary `string_view` as an argument since that constructor is not meant to handle character strings that have not been validated in the first place. For this reason, we only let `thing_factory` use it, thus strengthening encapsulation rather than weakening it.
It is customary to put a class and its friends together when shipping code as they go together: a friend of a class, in essence, is an external addition to that class’s interface. Finally, note that restrictions apply to friendship. Friendship is not reflexive; if `A` declares `B` to be its friend, it does not follow that `B` declares `A` to be its friend:

```

class A {

int n = 3;

friend class B;

public:

void f(B);

};

class B {

int m = 4;

public:

void f(A);

};

void A::f(B b) {

// int val = b.m; // no, A is not a friend of B

}

void B::f(A a) {

int val = a.n; // Ok, B is a friend of A

}

```cpp

 Friendship is not transitive; if `A` declares `B` to be its friend and `B` declares `C` to be its friend, it does not follow that `A` declares `C` to be its friend:

```

class A {

int n = 3;

friend class B;

};

class B {

friend class C;

public:

void f(A a) {

int val = a.n; // Ok, B is a friend of A

}

};

class C {

public:

void f(A a) {

// int val = a.n; // no, C is not a friend of A

}

};

```cpp

 Last but not least, friendship is not inherited; if `A` declares `B` to be its friend, it does not follow that if `C` is a child class of `B`, `A` has declared `C` to be its friend:

```

class A {

int n = 3;

friend class B;

};

class B {

public:

void f(A a) {

int val = a.n; // Ok, B is a friend of A

}

};

class C : B {

public:

void f(A a) {

// int val = a.n; // no, C is not a friend of A

}

};

```cpp

 Used judiciously, `friend` solves encapsulation problems that would be difficult to deal with otherwise.
The decltype operator
The type system of C++ is powerful and nuanced, offering (among other things) a set of type deduction facilities. The best-known type deduction tool is probably `auto`, used to infer the type of an expression from the type of its initializer:

```

const int n = f();

auto m = n; // m is of type int

auto & r = m; // r is of type int&

const auto & cr0 = m; // cr0 is of type const int&

auto & cr1 = n; // cr1 is of type const int&

```cpp

 As you might notice from the preceding example, by default, `auto` makes copies (see the declaration of variable `m` ), but you can qualify `auto` with `&`, `&&`, `const`, and so on if needed.
Sometimes, you want to deduce the type of an expression with more precision, keeping the various qualifiers that accompany it. That might be useful when inferring the type of an arithmetic expression, the type of a lambda, the return type of a complicated generic function, and so on. For this, you have the `decltype` operator:

```

template <class T>

T& pass_thru(T &arg) {

return arg;

}

int main() {

int n = 3;

auto m = pass_thru(n); // m is an int

++m;

cout << n << ' ' << m << '\n'; // 3 4

decltype(pass_thru(n)) r = pass_thru(n); // r is an int&

++r;

cout << n << ' ' << r << '\n'; // 4 4

}

```cpp

 The use of `auto` has become commonplace in C++ code since C++11, at least in some circles. The `decltype` operator, also part of C++ since C++11, is a sharper tool, still widely used but for more specialized use cases.
When the types get painful to spell
In the preceding `decltype` example, we spelled `pass_thru(n)` twice: once in the `decltype` operator and once in the actual function call. That’s not practical in general since it duplicates the maintenance effort and… well, it’s just noise, really. Since C++14, one can use `decltype(auto)` to express “the fully qualified type of the initializing expression.”
Thus, we would customarily write `decltype(auto) r = pass_thru(n);` to express that `r` is to have the fully qualified type of the expression `pass_thru(n)` .
Perfect forwarding
The advent of variadic templates in C++11 has made it necessary to ensure there is a way for the semantics at the call site of a function to be conveyed throughout the call chain. This might seem abstract but it’s quite real and has implications on the effect of function calls.
Consider the following class:

```

#include <string>

struct X {

X(int, const std::string&); // A

X(int, std::string&&); // B

// ... other constructors and various members

};

```cpp

 This class exposes at least two constructors, one that takes an `int` and `const string&` as argument and another that takes an `int` and a `string&&` instead. To make the example more general, we’ll also suppose the existence of other `X` constructors that we might want to call while still focusing on these two. If we called these two constructors explicitly, we could do so with the following:

```

X x0{ 3, "hello" }; // calls A

string s = "hi!";

X x1{ 4, s }; // also calls A

X x2{ 5, string{ "there" } }; // calls B

X x3{ 5, "there too"s }; // also calls B

```cpp

 The constructor of `x0` calls `A`, as `"hello"` is a `const char(&)[6]` (including the trailing `'\0'`), not a `string` type, but the compiler’s allowed to synthesize a temporary `string` to pass as a `const string&` in this case (it could not if the `string&` was non-`const` as it would require referring to a modifiable object).
The constructor of `x1` also calls `A`, as `s` is a named `string` type, which means it cannot be implicitly passed by movement.
The constructors of `x2` and `x3` both call `B`, which takes a `string&&` as an argument, as they are both passed temporary, anonymous `string` objects that can be implicitly passed by movement.
Now, suppose we want to write a factory of `X` objects that relays arguments to the appropriate `X` constructor (one of the two we’re looking at or any other `X` constructor) after having done some preliminary work; for the sake of this example, we’ll simply log the fact that we are constructing an `X` object. Let’s say we wrote it this way:

```

template <class ... Args>

X makeX(Args ... args) {

clog << "Creating a X object\n";

return X(args...); // <-- HERE

}

```cpp

 In this case, arguments would all have names and be passed by value, so the constructor that takes a `string&&` would never be chosen.
Now, let’s say we wrote it this way:

```

template <class ... Args>

X makeX(Args &... args) {

clog << "Creating a X object\n";

return X(args...); // <-- HERE

}

```cpp

 In this case, arguments would all be passed by reference, and a call that passed a `char` array such as `"hello"` as an argument would not compile. What we need to do is write our factory function in such a way that each argument keeps the semantics it had at the function’s call site, and is forwarded by the function with the exact same semantics.
The way to express this in C++ involves `std::forward<T>()` (from `<utility>`), which behaves as a cast. A forwarding reference superficially and syntactically looks like the `rvalue` references used for move semantics, but their impact on argument semantics is quite different. Consider the following example:

```

// v passed by movement (type vector<int> fully specified

void f0(vector<int> &&v);

// v passed by movement (type vector<T> fully specified

// for some type T)

template <class T>

void f1(vector<T> &&v);

// v is a forwarding reference (type discovered by

// the compiler)

template <class T>

f2():

```cpp
// T is vector<int>&& (pass by movement)
f2(vector<int>{ 2,3,5,7,11 });
vector<int> v0{ 2,3,5,7,11 };
f2(v0); // T is vector<int>& (pass by reference)
const vector<int> v1{ 2,3,5,7,11 };
X objects, in this case, the appropriate signature for makeX() would be as follows:

```

template <class ... Args>

X makeX(Args ... args) {

clog << "Creating a X object\n";

return X(args...); // <-- HERE (仍然不正确)

}

```cpp

 This version of our function almost works. The signature of `makeX()` is correct as each argument will be accepted with the type used at the call site, be it a reference, a reference to `const`, or an `rvalue` reference. What’s missing is that the arguments we are receiving as `rvalue` references now have a name within `makeX()` (they’re part of the pack named `args`!), so when calling the constructor of `X`, there’s no implicit move involved anymore.
What we need to do to complete our effort is to *cast back each argument to the type it had at the call site*. That type is inscribed in `Args`, the type of our pack, and the way to perform that cast is to apply `std::forward<T>()` to each argument in the pack. A correct `makeX()` function, at long last, would be as follows:

```

template <class ... Args>

X makeX(Args &&... args) {

clog << "Creating a X object\n";

return X(std::forward<Args>(args)...); // <-- HERE

}

```cpp

 Whew! There are simpler syntaxes indeed, but we made it.
The singleton design pattern
There are many design patterns out there. Design patterns are a topic of their own, representing well-known ways of solving problems that one can represent in the abstract, give a name to, explain to others, and then reify within the constraints and idioms of one’s chosen programming language.
The **singleton** design pattern describes ways in which we can write a class that ensures it is instantiated only once in a program.
Singleton is not a well-liked pattern: it makes testing difficult, introduces dependencies on global state, represents a single point of failure in a program as well as a potential program-wide bottleneck, complicates multithreading (if the singleton is mutable, then its state requires synchronization), and so on, but it has its uses, is used in practice, and we use it on occasion in this book.
There are many ways to write a class that is instantiated only once in a program with the C++ language. All of them share some key characteristics:

*   The type’s `copy` operations have to be deleted. If one can copy a singleton, then there will be more than one instance of that type, which leads to a contradiction.
*   There should be no `public` constructor. If there were, the client code could call it and create more than one instance.
*   There should be no `protected` members. Objects of derived classes are also, conceptually, objects of the base class, again leading to a contradiction (there would, in practice, be more than one instance of the singleton!).
*   Since there is no `public` constructor, there should be a `private` constructor (probably a default constructor), and that one will only be accessible to the class itself or to its friends (if any). For simplicity, we’ll suppose that the way to access a singleton is to go through a `static` (obviously) member function of the singleton.

We’ll look at ways to implement an overly simplistic singleton in C++. For the sake of this example, the singleton will provide sequential integers on demand. The general idea for that class will be the following:

```

#include <atomic>

class SequentialIdProvider {

// ...

std::atomic<long long> cur; // 状态（同步）

// 默认构造函数（私有）

SequentialIdProvider() : cur{ 0LL } {

}

public:

// 单例提供的服务（同步）

auto next() { return cur++; }

// 删除复制操作

SequentialIdProvider(const SequentialIdProvider&)

= delete;

SequentialIdProvider&

operator=(const SequentialIdProvider&) = delete;

// ...

};

```cpp

 The following subsections show two different techniques to create and provide access to the singleton.
Instantiation at program startup
One way to instantiate a singleton is to create it before `main()` starts by actually making it a `static` data member of its class. This requires *declaring* the singleton in the class and *defining* it in a separate source file in order to avoid ODR problems.
ODR, you say?
The **One Definition Rule** (**ODR**) and associated issues are described in *Chapter 2* of this book, but the gist of it is that in C++, every object can have many declarations but only one definition.
A possible implementation would be as follows:

```

#include <atomic>

class SequentialIdProvider {

// 声明（私有）

static SequentialIdProvider singleton;

std::atomic<long long> cur; // 状态（同步）

// 默认构造函数（私有）

SequentialIdProvider() : cur{ 0LL } {

}

public:

// 提供对对象的静态成员函数访问

static auto & get() { return singleton; }

// 单例提供的服务（同步）

auto next() { return cur++; }

// 删除复制操作

SequentialIdProvider(const SequentialIdProvider&)

= delete;

SequentialIdProvider&

operator=(const SequentialIdProvider&) = delete;

// ...

};

// 在某个源文件中，例如 SequentialIdProvider.cpp

#include "SequentialIdProvider.h"

// 定义（调用默认构造函数）

SequentialIdProvider，如果我们遇到麻烦，因为 C++不保证来自多个文件的全局对象实例化的顺序。

可能的客户端代码实现如下：

```cpp
auto & provider = SequentialIdProvider::get();
for(int i = 0; i != 5; ++i)
   cout << provider.next() << ' ';
```

这将显示单调递增的整数，可能是连续的（只要没有其他线程同时调用单例的服务）。

首次调用的实例化

实例化单例的另一种方法是，在首次请求其服务时创建它，使其成为提供单例访问权限的函数的`static`变量。这样，由于`static`局部变量在函数首次调用时创建并保持其状态，单例可以为其他单例提供服务，只要这不会创建循环。

可能的实现如下：

```cpp
#include <atomic>
class SequentialIdProvider {
   std::atomic<long long> cur; // state (synchronized)
   // default constructor (private)
   SequentialIdProvider() : cur{ 0LL } {
   }
public:
   // static member function providing access to the object
   static auto & get() {
      static SequentialIdProvider singleton; // definition
      return singleton;
   }
   // service offered by the singleton (synchronized)
   auto next() { return cur++; }
   // deleted copy operations
   SequentialIdProvider(const SequentialIdProvider&)
      = delete;
   SequentialIdProvider&
      operator=(const SequentialIdProvider&) = delete;
   // ...
};
```

可能的客户端代码实现如下：

```cpp
auto & provider = SequentialIdProvider::get();
for(int i = 0; i != 5; ++i)
   cout << provider.next() << ' ';
```

这将显示单调递增的整数，可能是连续的（只要没有其他线程同时调用单例的服务）。

注意，这个版本有一个隐藏的成本：函数本地的`static`变量被称为`static`变量涉及一些同步，并且这种同步在每次调用该函数时都会付出代价。前面的客户端代码通过一次调用`SequentialIdProvider::get()`来减轻这种成本，然后在该调用之后重用通过该调用获得的引用；是`get()`的调用引入了同步成本。

The std::exchange() function

在 `<utility>` 头文件中隐藏着（至少）两个非常有用且基本的功能。一个是众所周知的，并且已经存在很长时间：`std::swap()`，它在标准库的许多用途以及用户代码中都被使用。

另一个较新的一个是 `std::exchange()`。其中 `swap(a,b)` 交换对象 `a` 和 `b` 的值，表达式 `a = exchange(b,c)` 将 `b` 的值与 `c` 的值交换，并返回 `b` 的旧值（以便将其赋值给 `a`）。一开始这可能看起来有些奇怪，但实际上这是一个非常实用的功能。

考虑以下简化版的 `fixed_size_array` 的移动构造函数：

```cpp
template <class T>
   class fixed_size_array {
      T *elems{};
      std::size_t nelems{};
   public:
      // ...
      fixed_size_array(fixed_size_array &&other)
         : elems{ other.elems }, nelems{ other.nelems } {
         other.elems = nullptr;
         other.nelems = 0;
      }
      // ...
   };
```

你可能会注意到这个构造函数做了两件事：它从 `other` 中获取数据成员，然后使用默认值替换 `other` 的成员。这就是 `std::exchange()` 的典型应用，因此这个构造函数可以简化如下：

```cpp
template <class T>
   class fixed_size_array {
      T *elems{};
      std::size_t nelems{};
   public:
      // ...
      fixed_size_array(fixed_size_array &&other)
         : elems{ std::exchange(other.elems, nullptr) },
           nelems{ std::exchange(other.nelems, 0) } {
      }
      // ...
   };
```

使用 `std::exchange()`，这个常见的两步操作可以简化为一个函数调用，简化代码并提高效率（在这种情况下，将赋值转换为构造函数调用）。

```cpp

```

```cpp

```
