

# 第十四章：使用分配器支持编写泛型容器

自本书开始以来，我们已经走得很远了。最近几章探讨了如何编写内存高效的容器，描述了在显式进行内存管理时（在*第十二章*）以及通过智能指针隐式进行内存管理时（在*第十三章*）如何做到这一点。选择内存管理方法不是非此即彼的问题；每种方法都有其自身的用途，并解决根据应用领域而定的实际用例。

然而，我们之前所介绍的所有方法都不符合标准库容器的做法。实际上，标准库容器（以及许多其他可以动态分配内存的标准库类型）都是来自一个区域（参见*第十章*）或来自堆栈上的固定容量缓冲区的 `std::vector`。

分配器在 C++98 中随着标准库容器一起正式成为 C++ 语言的一部分，但它们随着时间的推移而发展和多样化。使用 C++11，编写分配器变得显著简单，而 C++17 通过引入具有 **多态内存资源**（**PMR**）分配器和容器的全新内存分配方法。

在本章中，您将执行以下操作：

+   理解和使用传统分配器

+   为特定应用领域编写传统分配器

+   学习如何在容器移动或复制时管理分配器的生命周期

+   克隆分配器的类型

+   理解和使用 PMR 分配器和容器

拥有分配器和它们如何与容器交互的知识，本章将丰富您的内存管理工具箱，并开辟将数据组织与存储获取方式相结合的新途径。理解分配器甚至可能使编写新容器变得不那么必要；有时，与其尝试创建一个全新的容器，不如将合适的数据组织策略与合适的存储管理方法结合起来。

# 技术要求

您可以在此处找到本书中该章节的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter14`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter14).

关于本章示例的一些建议

与*第十三章*的情况一样，本章将展示不完整的示例，以避免与之前找到的摘录重复，尤其是那些在*第十二章*中的。分配器改变了容器与内存管理设施交互的方式，但它们不需要完全重写容器，因此为给定容器编写的代码在内存管理方式如何的情况下仍然保持稳定。您在 GitHub 仓库中找到的代码当然是完整的。

还要注意，本章在容器的背景下讨论分配器，但这一想法可以扩展到许多需要动态分配内存的类型。有时这样做很困难；例如，C++17 中移除了`std::function`中的分配器支持，因为没有任何已知的标准库实现能够使其工作。尽管如此，分配器可以被视为一个通用概念，而不仅仅是局限于容器，您可以在其他上下文中设想使用分配器。

# 为什么需要分配器？

分配器往往会让人感到害怕，包括一些专家在内，但您不会感到害怕，因为您已经掌握了大量的内存管理知识和技能（鉴于您正在阅读这本书，您可能对这一主题有更多的好奇心）。了解这一点后，我们首先需要解决的问题，甚至在表达分配器是什么之前，就是“为什么分配器存在？”。我们为什么要关心内存管理代码中额外的复杂性层次？

嗯，这是 C++，C++的一切都是关于给用户提供*控制*，所以我们的解释从这里开始。为了做一个类比，想想迭代器：为什么它们有用，以及它们如何使程序员的编程生活变得更好。它们将遍历序列元素的方式与元素在序列中的组织方式解耦，这样您就可以编写代码来计算诸如`std::list<int>`或`std::vector<short>`中值的总和，而无需知道在第一种情况下，您正在通过指针相互链接的节点进行导航，在第二种情况下，您正在遍历存储在连续内存中的对象。

迭代器的美妙之处在于迭代和数据组织之间的解耦。同样，分配器将数据组织与底层存储的获取或释放方式解耦。这使得我们可以独立于内存管理的属性来推理容器的属性，从而使容器在更多情况下变得有用。

非常，非常薄的一层…

对于容器来说，分配器（至少是我们即将讨论的“传统”模型中的那些）代表了对硬件的薄薄（非常薄）一层抽象。对于容器来说，分配器表达了诸如“地址是什么？”，“如何将对象放在某个地方？”，“如何销毁某个位置的对象？”等问题。从某种意义上说，对于容器来说，分配器本质上就是硬件。

# 传统分配器

如前所述，分配器已经成为了 C++几十年的支柱，但它们以几种不同的形式和形状存在。在本章中，我们将采用一种类似时间顺序的方法，从较早（更复杂）的分配器类型开始，逐步过渡到更简单（更灵活）的类型。

要理解这一章，一个关键的想法是记住，容器类型如 `std::vector<T>` 并不存在。真正存在的是 `std::vector<T,A>` 类型，其中默认情况下 `A` 是 `std::allocator<T>`，它通过 `::operator new()` 分配内存，通过 `::operator delete()` 释放内存。我们所说的 **传统分配器** 指的是容器类型的一部分的分配器类型（这并不是今天编写分配器的唯一可能方法，正如我们在本章后面讨论 PMR 分配器时将会看到的）。

我们将首先检查在 C++11 之前编写分配器所需的内容，以及容器如 `std::vector<T,A>` 如何使用 `A` 类型的对象来抽象其内存分配任务。分配器表达方式的改进将在本章后面的部分中讨论。

## 在 C++11 之前

在 C++11 之前编写的传统分配器必须实现一系列成员，这使得编写分配器的任务对许多人来说似乎很艰巨。考虑一下那些日子人们需要编写的内容，并请注意，以下内容并非全部在撰写本文时仍然有效，因为分配器的 API 随时间而演变。

跟踪不断演变的 API 的难度

对分配器的要求随着 C++03 以来每个版本的 C++ 而变化，如今，编写编译为 C++11 的示例并不总是容易（或相关）。因此，我们将详细编写的示例将使用 C++11 分配器，以展示这实际上意味着什么，但将使用 C++17 标准编译，以便代码更易于阅读（和编写）。

我们将检查这样一个分配器，`small_allocator<T>`，并以类似于 `std::allocator<T>` 的方式实现它，以突出在 C++11 时代编写分配器的意义，然后将其与标准更近版本的表达式进行比较。在我们的实现中，我们将使用 C++17 特性，因为我们不希望在已经微妙的话题中引入不必要的复杂性。

在介绍 `small_allocator<T>` 之后，我们将展示如何将来自 *第十二章* 和 *第十三章* 的 `Vector<T>` 进行增强，并成为 `Vector<T,A>`，以及 `A` 可以是 `std::allocator<T>`、`small_allocator<T>` 或任何其他符合规范的分配器类型。

### 类型别名

`T` 类型的分配器必须公开 `value_type`、`size_type`、`difference_type`（从两个 `pointer` 对象相减得到的结果类型）、`pointer`、`const_pointer`、`reference` 和 `const_reference` 的类型别名。可以这样理解：对于一个容器来说，分配器代表底层内存，从而定义了最佳描述这些底层概念的类型。容器可以将自己的别名映射到分配器的别名上，以保持一致性。

在我们的 `small_allocator<T>` 类型中，这会转化为以下内容：

```cpp
template <class T>
struct small_allocator {
   using value_type = T;
   using pointer = T*;
   using const_pointer = const T*;
   using reference = T&;
   using const_reference = const T&;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;
   // ...
```

实际上，对于 `T` 类型的分配器，我们可以在所有但最奇怪的情况下期望这些类型别名与 `small_allocator<T>` 中显示的类型别名相对应：只要 `value_type` 被定义，我们几乎总能推断出其他类型。

### 成员函数

`T` 类型的分配器必须公开一个成员函数 `max_size()`，该函数应该返回分配器实际可以分配的最大块的大小。

实际上，这通常证明是不可实现的，因为对于某些操作系统，分配总是成功的（但如果程序分配过多，分配的内存的使用可能会失败），因此该函数通常在给定平台上以尽力而为的方式实现。一个可能的实现如下所示：

```cpp
   // ...
   constexpr size_type max_size() const {
      return std::numeric_limits<size_type>::max(); // bah
   }
   // ...
```

`T` 类型的分配器还必须公开两个函数重载，这些函数使用了作者的学生们“喜欢”的词汇（欣赏这种讽刺！）。考虑 `pointer address(reference r)` 以及 `const` 对象的等效函数，即 `const_pointer address(const_reference r)`。这里的意图是抽象出获取对象地址的方式。

实现这些函数为 `return &r;` 可能很有诱惑力，但在实践中，这是危险的，因为用户被允许为他们自己的类型重载一元 `operator&()`，这意味着这种实现会调用任意代码，这确实是一个令人恐惧的前景……除非你真的、真的有很好的理由这样做，否则请考虑解决你问题的其他替代方法！

一种更好的实现技术是通过 `return std::addressof(r);` 来表达这些函数，其中 `std::addressof()` 是来自 `<memory>` 的一个“神奇”标准库函数（即 `constexpr`），它返回对象的地址，而不通过可重载的设施：

```cpp
   // ...
   constexpr pointer address(reference r) const {
      return std::addressof(r);
   }
   constexpr
      const_pointer address(const_reference r) const {
      return std::addressof(r);
   }
   // ...
```

显然，分配器需要公开成员函数来执行实际的内存分配。这些函数的签名是 `allocate(size_type n)` 和 `deallocate(pointer p, size_type n)`。这两个函数的简单实现可能如下所示：

```cpp
   // ...
   pointer allocate(size_type n) {
      auto p = static_cast<pointer>(
         malloc(n * sizeof(value_type))
      );
      if (!p) throw std::bad_alloc{};
      return p;
   }
   void deallocate(pointer p, size_type) {
      free(p);
   }
   // ...
```

`allocate()` 成员函数过去接受一个名为 `hint` 的 `void*` 类型的第二个参数，默认初始化为 `nullptr`。这个参数的目的是通知分配器一个可能用于提供存储的位置，如果容器知道这样的位置。这个特性在实践中的使用似乎很少（如果有的话），并在 C++17 中被弃用，然后在 C++20 中被移除。

这两个函数是分配器存在本质的原因：`allocate()` 返回足够容纳 `n` 个连续 `value_type` 元素的内存块，在失败时抛出 `bad_alloc`，而 `deallocate()` 释放足够容纳 `n` 个连续 `value_type` 元素的内存块。当编写一个分配器时，通常寻求为这个特定问题提供答案。

字节或对象

有趣的是，与接受字节数作为参数的 `operator new()` 相反，`allocate()` 和 `deallocate()` 都接受对象的数量作为参数。这是因为传统的分配器是类型感知的（毕竟它们是某种类型 `T` 的分配器），而 `operator new()` 和其相关函数（主要）是无类型感知的。你会在本章后面注意到，PMR 分配器（人们可能会称之为“退一步”）使用无类型感知的内存资源，如 `malloc()` 或 `operator new()`。

`allocate()` 和 `deallocate()` 都故意向客户端代码撒谎：它们交易的是原始内存，既不创建也不销毁类型 `T` 的对象，但 `allocate()` 返回一个 `pointer`（本质上是一个 `T*`），而 `deallocate()` 接受一个 `pointer` 作为参数，尽管假设所有 `T` 对象在此之前都已销毁。

这些函数欺骗类型系统在某种程度上是好事，因为它减轻了容器执行此操作的负担。当然，容器必须了解这些函数的作用，并且不应该假设 `allocate()` 返回或传递给 `deallocate()` 的内存中存在对象。

最后，分配器必须公开成员函数以将原始内存转换为对象，反之亦然。`construct(pointer p,const_reference r)` 和 `destroy(pointer p)` 函数分别用于在位置 `p`（假设之前已分配）处构造 `r` 的副本，并销毁位置 `p` 处的对象（不释放底层存储）：

```cpp
   // ...
   void construct(pointer p, const_reference r) {
      new (static_cast<void*>(p)) value_type(r);
   }
   void destroy(const_pointer p) {
      if(p) p->~value_type();
   }
   // ...
   template <class U>
   struct rebind {
      using other = small_allocator<U>;
   };
};
```

可以预期大多数实现将基本上做前面代码所做的事情。有其他选择，但在实践中很少遇到。

再次强调，这些函数欺骗了类型系统：`construct()` 接受一个 `pointer`（实践中是一个 `T*`）作为参数，但函数被调用时，该指针指向的是原始内存，而不是类型 `T` 的对象。

那重绑定（rebind）呢？

你会注意到我们没有讨论 `rebind` 公共模板类型，但这仅仅是因为当面对这个类型旨在解决的问题时，这个类型背后的想法更容易理解。我们将在本章后面通过我们的 `ForwardList<T,A>` 类讨论分配器感知的基于节点的容器时遇到这种情况。

超过这一点，对分配器的需求是定义不同类型的两个分配器对象是否相等。一个可能的实现如下：

```cpp
// ...
template <class T, class U>
constexpr bool operator==(const small_allocator<T>&,
                          const small_allocator<U>&) {
   return true;
}
template <class T, class U>
constexpr bool operator!=(const small_allocator<T>&,
                          const small_allocator<U>&) {
   return false;
}
```

换句话说，两个针对不同类型的`small_allocator`特化描述了相同的策略，因此被认为是相等的。“但是等等！”你说，“在这个计算中你是如何考虑分配器的状态的？”但是这里有一个启示：C++11 之前的分配器基本上被认为是*无状态的*。

好吧，它们并不是，但如果一个分配器与一个容器对象相关联，并且该对象被复制，那么会发生什么并不清楚。你看，如果一个分配器有*状态*，我们必须知道在分配器被复制时如何处理这个状态。这个状态是被复制了吗？是被共享了吗？在 C++11 之前，我们不知道在这种情况下应该怎么做，所以除非容器被用于不会复制的上下文中，比如函数局部向量以及与使用栈空间作为存储的分配器相关联的情况，否则大多数人完全避免使用有状态的分配器。

但有状态的分配器又如何呢？

正如暗示的那样，当时有状态的分配器是可能的（它们存在，并且在实际中得到了应用）。人们预期如何定义有状态分配器的分配器相等性（以及分配器的一般相等性）呢？一般想法是，如果从其中一个分配器分配的内存可以从另一个分配器释放，那么两个分配器应该相等。对于将分配任务委托给如`std::malloc()`或`::operator new()`等自由函数的分配器，相等性是显而易见的`true`，但有状态的分配器要求我们思考如何定义这种关系。

在我们查看如何编写分配器感知容器之前，我们将退一步看看如何将*第十二章*和*第十三章*中使用的某些未初始化内存算法适应以使用分配器的服务。这将减少在后续过程中所需的重构工作量。

### 一些分配器感知支持算法

由于我们使用分配器来弥合原始存储和对象之间的差距，因此我们无法在我们的分配器感知实现中使用在*第十二章*和*第十三章*中看到的原始内存算法。

我们有选择在每个容器调用点详细编写这些算法的版本，但这会很繁琐（并且容易出错）。相反，我们将编写这些低级内存管理算法的简化版本，并使这些简化版本使用作为参数传递的分配器。通过这样做，我们将减少容器分配器感知对实现的影响。

这三个算法中的前三个将是初始化值范围的分配器感知版本的算法，以及一个销毁此类范围的算法。为了最小化对现有实现的影响，我们将基本上使用与它们的非分配器感知对应版本相同的签名，但添加一个引用分配器的参数。对于用某个值填充原始内存块块的算法，我们有以下内容：

```cpp
template <class A, class IIt, class T>
void uninitialized_fill_with_allocator(
   A& alloc, IIt bd, IIt ed, T init
) {
   // bd: beginning of destination¸
   // ed: end of destination
   auto p = bd;
   try {
      for (; p != ed; ++p)
         alloc.construct(p, init);
   } catch (...) {
      for (auto q = bd; q != p; ++q)
         alloc.destroy(q);
      throw;
   }
}
```

然后，对于将值序列复制到原始内存块块的算法，我们有以下内容：

```cpp
template <class A, class IIt, class OIt>
void uninitialized_copy_with_allocator(
   A& alloc, IIt bs, IIt es, OIt bd
) {
   // bs: beginning of source
   // es: end of source
   // bd: beginning of destination¸
   auto p = bd;
   try {
      for (auto q = bs; q != es; ++q) {
         alloc.construct(p, *q);
         ++p;
      }
   } catch (...) {
      for (auto q = bd; q != p; ++q)
         alloc.destroy(q);
      throw;
   }
}
```

对于将值序列移动到原始内存块块的算法，我们有以下内容：

```cpp
template <class A, class IIt, class OIt>
void uninitialized_move_with_allocator(
   A& alloc, IIt bs, IIt es, OIt bd
) {
   // bs: beginning of source
   // es: end of source
   // bd: beginning of destination¸
   auto p = bd;
   try {
      for (auto q = bs; q != es; ++q) {
         alloc.construct(p, std::move(*q));
         ++p;
      }
   } catch (...) {
      for (auto q = bd; q != p; ++q)
         alloc.destroy(q);
      throw;
   }
}
```

最后，对于将对象序列转换为原始内存块块的算法，我们有以下内容：

```cpp
template <class A, class It>
   void destroy_with_allocator(A &alloc, It b, It e) {
      for (; b != e; ++b)
         alloc.destroy(b);
   }
```

注意，在每种情况下，如果发生异常，则对象将按照构造的相反顺序销毁，这将使实现更加符合规范。请随意实现这个小的调整；这并不困难，但会在我们的示例中引入一些噪音。

我们将要重写的另一个标准设施是`cmp_less()`，它允许在不被 C 语言的整数提升规则捕获的情况下比较有符号值和无符号值。它与内存直接相关，但我们需要它在我们的`Vector<T>`实现中，并且这是一个 C++20 特性，这使得在为 C++17 编译时不可用：

```cpp
template<class T, class U>
   constexpr bool cmp_less(T a, U b) noexcept {
      if constexpr (std::is_signed_v<T> ==
                    std::is_signed_v<U>)
         return a < b;
      else if constexpr (std::is_signed_v<T>)
         return a < 0 || std::make_unsigned_t<T>(a) < b;
      else
         return b >= 0 && a < std::make_unsigned_t<U>(b);
   }
```

`std::is_signed<T>`特性以及`std::make_unsigned<T>()`函数都可以在头文件`<type_traits>`中找到。

条件编译和特征测试宏

作为旁注，如果你发现自己必须维护代码，其中可能或可能没有`std::cmp_less()`这样的功能，例如有时为 C++20 编译有时为 C++17 编译的源文件，考虑通过测试相关的特征测试宏条件包含你的“自制解决方案”版本。

对于这个特定的情况，可以通过使用`#ifndef __cpp_lib_integer_comparison_functions`将我们个人版本的`cmp_less()`函数定义包裹起来，以确保只有在没有标准库实现提供版本的情况下才提供它。

现在，让我们看看这些分配器和我们的支持算法如何被容器使用，首先是一个使用连续存储的容器（我们的`Vector<T,A>`类），然后是一个基于节点的容器（我们的`ForwardList<T,A>`类）。

### 分配器感知的`Vector<T,A>`类

现在我们准备看看在连续内存使用的容器（更具体地说，我们的`Vector<T>`类）中引入分配器意识如何影响该容器的实现。请注意，我们将使用第*第十二章*中明确的内存管理方法作为基准，因为我们想探索分配器意识的影响，这将帮助我们更明显地做出实现更改。如果你愿意，可以自由地根据隐式内存管理方法调整本章中的代码。

从模板的签名本身开始，我们现在有一个双类型模板，其中`T`是元素类型，`A`是分配器类型，但为`A`提供了一个合理的默认类型，这样普通用户就不需要担心这样的技术细节：

```cpp
template <class T, class A = std::allocator<T>>
class Vector : A { // note: private inheritance
public:
   using value_type = typename A::value_type;
   using size_type = typename A::size_type;
   using pointer = typename A::pointer;
   using const_pointer = typename A::const_pointer;
   using reference = typename A::reference;
   using const_reference = typename A::const_reference;
private:
   // deliberately self-exposing selected members
   // of the private base class as our own
   using A::allocate;
   using A::deallocate;
   using A::construct;
   using A::destroy;
   // ...
```

注意以下技术：

+   由于我们期望`A`是无状态的，我们使用了私有继承，并使`A`成为`Vector<T,A>`的基类，从而实现了空基优化。或者，我们也可以在每个`Vector<T,A>`对象内部使用类型为`A`的数据成员（可能带来轻微的大小惩罚）。

+   我们从其分配器的类型别名推导出容器的类型别名。在实践中，这可能与我们在前几章中使用的别名没有太大变化，但`A`可能在进行一些“花哨的技巧”（永远不能太过小心）。

+   在我们类的私有部分，我们将基类的一些选定成员公开为我们的成员。这将使代码在以后变得更加简洁，例如，我们可以编写`allocate(n)`而不是`this->A::allocate(n)`。

我们类中非分配成员没有变化，这是预料之中的。数据成员保持不变，基本访问器如`size()`、`empty()`、`begin()`、`end()`、`front()`、`operator[]`等也是如此。默认构造函数也没有变化，因为它不分配内存，因此不需要与它的分配器交互。

需要一个新的构造函数，它接受一个分配器作为参数。这个构造函数在状态化分配器的情况下特别有用：

```cpp
   // ...
   Vector(A &alloc) : A{ alloc } {
   }
   // ...
```

当然，当遇到需要分配内存的构造函数时，情况变得更加有趣。以一个接受元素数量和初始值的构造函数为例：

```cpp
   // ...
   Vector(size_type n, const_reference init)
      : A{},elems{ allocate(n) },
        nelems{ n }, cap{ n } {
      try {
         uninitialized_fill_with_allocator(
            *static_cast<A*>(this), begin(), end(), init
         );
      } catch (...) {
         deallocate(elems, capacity());
         throw;
      }
   }
   // ...
```

这里有很多要说的：

+   将作为我们容器底层存储的内存块是通过调用基类的`allocate()`成员函数分配的。记住，尽管这产生了一个`指针`（一个`T*`），但这是一种谎言，新分配的块中没有`T`对象。

+   我们通过我们自制的分配器感知版本的`std::uninitialized_fill()`（见`_with_allocator`后缀）来填充那个未初始化的内存块，用`T`对象填充。注意我们如何将分配器作为参数传递给算法：`Vector<T,A>`和`A`之间的继承关系是`private`，但派生类知道这一点，并且可以通过`static_cast`使用这些信息。

+   如果在初始化该内存块的过程中使用的任何一个构造函数抛出异常，算法将像往常一样销毁它所创建的对象，然后我们拦截该异常，在重新抛出该异常之前释放存储，以实现异常中立。

在其他分配构造函数中，也使用了类似的操作，但用于初始化分配存储的算法不同。移动构造函数和`swap()`成员函数不分配内存，因此保持不变，赋值运算符也是如此：它们是由其他成员函数构建的，并且不需要自己分配或释放内存。

如您可能已经猜到的，我们容器的析构函数将使用分配器来销毁对象并释放底层存储：

```cpp
   // ...
   ~Vector() {
      destroy_with_allocator(
         *static_cast<A*>(this), begin(), end()
      );
      deallocate(elems, capacity());
   }
   // ...
```

`push_back()`和`emplace_back()`成员函数本身不分配内存，而是委托给我们的私有`grow()`成员函数，该函数反过来委托给`reserve()`进行分配，但它们确实需要在容器的末尾`construct()`一个对象：

```cpp
   // ...
   void push_back(const_reference val) {
      if (full()) grow();
      construct(end(), val);
      ++nelems;
   }
   void push_back(T&& val) {
      if (full()) grow();
      construct(end(), std::move(val));
      ++nelems;
   }
   template <class ... Args>
   reference emplace_back(Args &&...args) {
      if (full()) grow();
      construct(end(), std::forward<Args>(args)...);
      ++nelems;
      return back();
   }
   // ...
```

我们类中内存分配的主要工具可能是`reserve()`和`resize()`。在这两种情况下，算法保持不变，但底层内存管理任务被委托给分配器。对于`reserve()`，这导致以下情况：

```cpp
   // ...
   void reserve(size_type new_cap) {
      if (new_cap <= capacity()) return;
      auto p = allocate(new_cap);
      if constexpr (std::is_nothrow_move_assignable_v<T>) {
         uninitialized_move_with_allocator(
            *static_cast<A*>(this), begin(), end(), p
         );
      } else {
         auto src_p = begin();
         auto b = p, e = p + size();
         try {
            uninitialized_copy_with_allocator(
               *static_cast<A*>(this), begin(), end(), p
            );
         } catch (...) {
            deallocate(p, new_cap);
            throw;
         }
      }
      deallocate(elems, capacity());
      elems = p;
      cap = new_cap;
   }
   // ...
```

而对于`resize()`，我们现在有以下情况：

```cpp
   // ...
   void resize(size_type new_cap) {
      if (new_cap <= capacity()) return;
      auto p = allocate(new_cap);
      if constexpr (std::is_nothrow_move_assignable_v<T>) {
         uninitialized_move_with_allocator(
            *static_cast<A*>(this), begin(), end(), p
         );
      } else {
         uninitialized_copy_with_allocator(
            *static_cast<A*>(this), begin(), end(), p
         );
      }
      try {
         uninitialized_fill_with_allocator(
            *static_cast<A*>(this),
            p + size(), p + new_cap, value_type{}
         );
         destroy_with_allocator(
            *static_cast<A*>(this), begin(), end()
         );
         deallocate(elems, capacity());
         elems = p;
         nelems = cap = new_cap;
      } catch(...) {
         destroy_with_allocator(
            *static_cast<A*>(this), p, p + size()
         );
         deallocate(p, new_cap);
         throw;
      }
   }
   // ...
```

在`Vector<T>`类的先前实现中，我们为`insert()`和`erase()`各自实现了一个版本，因为实现所有这些函数会使这本书变得过于庞大。由于这两个函数都涉及已初始化和未初始化的内存，它们需要调整以使用分配器的服务而不是进行自己的内存管理。

在`insert()`的情况下，需要调整函数的关键方面是那些将对象复制或移动到原始内存中的方面：

```cpp
   // ...
   template <class It>
   iterator insert(const_iterator pos, It first, It last) {
      iterator pos_ = const_cast<iterator>(pos);
      const auto remaining = capacity() - size();
      const auto n = std::distance(first, last);
//      if (std::cmp_less(remaining, n)) { // needs C++20
      if(cmp_less(remaining, n)) {
         auto index = std::distance(begin(), pos_);
         reserve(capacity() + n - remaining);
         pos_ = std::next(begin(), index);
      }
      const auto nb_to_uninit_displace =
         std::min<std::ptrdiff_t>(n, end() - pos_);
      auto where_to_uninit_displace =
         end() + n - nb_to_uninit_displace;
      if constexpr (
         std::is_nothrow_move_constructible_v<T>
      )
         uninitialized_move_with_allocator(
            *static_cast<A*>(this),
            end() - nb_to_uninit_displace, end(),
            where_to_uninit_displace
         );
      else
         uninitialized_copy_with_allocator(
            *static_cast<A*>(this),
            end() - nb_to_uninit_displace, end(),
            where_to_uninit_displace
         );
      // note : might be zero
      const auto nb_to_uninit_insert =
         std::max<std::ptrdiff_t>(
            0, n - nb_to_uninit_displace
         );
      auto where_to_uninit_insert = end();
      uninitialized_copy_with_allocator(
         *static_cast<A*>(this),
         last - nb_to_uninit_insert, last,
         where_to_uninit_insert
      );
      // note : might be zero
      const auto nb_to_backward_displace =
         std::max<std::ptrdiff_t>(
            0, end() - pos_ - nb_to_uninit_displace
         );
      auto where_to_backward_displace = end();
      if constexpr (std::is_nothrow_move_assignable_v<T>)
         std::move_backward(
            pos_, pos_ + nb_to_backward_displace,
            where_to_backward_displace
         );
      else
         std::copy_backward(
            pos_, pos_ + nb_to_backward_displace,
            where_to_backward_displace
         );
      std::copy(
         first, first + n - nb_to_uninit_insert, pos_
      );
      nelems += n;
      return pos_;
   }
   // ...
```

在`erase()`的情况下，我们执行的操作是将被删除对象之后的所有对象“向左”移动一个位置；在此复制操作完成后，序列末尾的对象必须被销毁，为此，我们需要使用分配器的服务。以下是一个示例：

```cpp
   // ...
   iterator erase(const_iterator pos) {
      iterator pos_ = const_cast<iterator>(pos);
      if (pos_ == end()) return pos_;
      std::copy(std::next(pos_), end(), pos_);
      destroy(std::prev(end()));
      --nelems;
      return pos_;
   }
};
```

如您此时可能已经收集到的，我们可以以多种方式优化或简化这些函数，例如以下方式：

+   `reserve()` 和 `resize()` 之间存在共同的核心功能，因此我们可以说 `resize()` 在很大程度上类似于 `reserve()` 后跟一个未初始化的填充，并以此表达。

+   在 `erase()` 的情况下，在编译时，我们可以测试 `std::is_nothrow_move_assignable_v<T>` 特性的值，如果该条件成立，则将 `std::copy()` 的调用替换为 `std::move()` 的调用。

+   我们可以使 `insert()` 和 `erase()` 比现在更异常安全，尽管这会使这本书的代码稍微长一些。

到目前为止，我们有一个分配器感知的容器，它管理连续内存。现在将很有趣地看到分配器感知对基于节点的容器的影响，我们将通过 `ForwardList<T>` 类的分配器感知版本来解决这个问题。

### 一个分配器感知的 `ForwardList<T,A>` 类

当编写分配器感知的基于节点的容器时，会发生一件有趣的事情。请注意我们的 `ForwardList<T,A>` 类的开始部分：

```cpp
template <class T, class A = std::allocator<T>>
class ForwardList {
public:
   using value_type = typename A::value_type;
   // likewise for the other aliases
private:
   struct Node {
      value_type value;
      Node *next = nullptr;
      Node(const_reference value) : value { value } {
      }
      Node(value_type &&value)
         : value { std::move(value) } {
      }
   };
   Node *head {};
   size_type nelems {};
   // ...
```

你注意到类型 `A` 的有趣之处了吗？想想看…

是的，就是这样：`A` *是错误类型*！像 `ForwardList<T,A>` 这样的基于节点的容器永远不会分配类型 `T` 的对象：它分配 *节点*，这些节点（很可能）包含 `T` 对象和其他东西，例如，在这种情况下，指向序列中下一个 `Node` 的指针。

了解这一点后，如果我们提供了一些类似于我们在 *第十章* 中用于 `Orc` 对象的竞技场分配策略的 `A` 分配器，使分配器了解 `T`（因此，了解 `sizeof(T)`），这将导致管理错误大小对象的竞技场。这可不是什么好事！

我们面临一个有趣的困境：用户代码提供给我们一个分配器，因为它希望我们的容器能够充分利用 *分配策略*。这种分配策略作为容器模板参数出现，这就是为什么它与元素的类型相关联（在我们容器类的定义这一点上，我们不知道节点将是什么）。只有在我们定义了容器中的节点是什么之后，我们才能真正地说出需要分配什么，但那时 `A` 已经存在，并且已经与 `T` 相关联，而不是我们真正需要的类型，即 `ForwardList<T,A>::Node`。

注意，我们已经实例化了类型 `A`，但尚未构造该类型的任何对象。幸运的是，那样会非常浪费（我们永远不会使用它！）。我们真正需要的是一个与 `A` 类似的类型，但能够分配我们的 `Node` 类型对象，而不是类型 `T` 的对象。我们需要一种方法来 *克隆 `A` 描述的分配策略* 并将其应用于另一个类型。

这正是 `rebind` 的用途。记住，我们在之前编写 `small_allocator<T>` 时提到了这个模板类型，但说我们会等到可以用到它的时候再回来？现在我们就在这里，亲爱的读者。作为提醒，在分配器的上下文中，`rebind` 表现如下：

```cpp
template <class T>
   class small_allocator { // for example
   // ...
   template <class U>
      struct rebind {
         using other = small_allocator<U>;
      };
   // ...
};
```

你可以将 `rebind` 视为某种奇怪的诗意代码：这是分配器说“如果你想要与我相同类型但应用于某些 `U` 类型而不是 `T`，那么这个类型会是什么样子”的一种方式。

返回到我们的 `ForwardList<T,A>` 类，既然我们已经知道了 `rebind` 的用途，我们就可以创建我们自己的内部分配器类型，`Alloc`。这将类似于分配器类型 `A`，但应用于 `Node` 而不是 `T`，并创建一个该类型的对象（在我们的实现中偶然命名为 `alloc`），我们将使用它来执行容器中的内存管理任务：

```cpp
   // ...
   using Alloc = typename A::rebind<Node>::other;
   Alloc alloc;
   // ...
```

这是个不错的技巧，不是吗？记住，我们克隆的是 *策略*，即类型，而不是一个实际的对象，所以一些假设的 `A` 对象可能拥有的任何状态不一定是我们新的 `Alloc` 类型的一部分（至少不是不进行一些非平凡的杂技表演的情况下）。这又是另一个提醒，按照传统的分配器设计，复制和移动分配器状态是一个复杂的问题。

就像从 `Vector<T>` 转换到 `Vector<T,A>` 一样，我们的大部分 `List<T>` 实现涉及不到内存分配，因此不需要随着 `List<T,A>` 而改变。这包括 `size()`、`empty()`、`begin()`、`end()`、`swap()`、`front()` 和 `operator==()` 成员函数，以及其他许多 `List<T,A>::Iterator<U>` 类定义。由于我们的 `ForwardList<T,A>` 实现有时需要访问迭代器的私有数据成员 `cur`，我们给它 `friend` 权限：

```cpp
   // ...
   template <class U> class Iterator {
      // ...
   private:
      Node *cur {};
      friend class ForwardList<T,A>;
      // ...
   };
   // ...
```

当然，`ForwardList<T,A>` 有一些使用内存分配机制的成员函数。其中之一是 `clear()`，其作用是销毁容器中的节点。`Node` 对象的销毁和重新分配必须通过分配器执行，用一对函数调用替换对 `operator delete()` 的调用：

```cpp
   // ...
   void clear() noexcept {
      for(auto p = head; p; ) {
         auto q = p->next;
         alloc.destroy(p);
         alloc.deallocate(p, 1);
         p = q;
      }
      nelems = 0;
   }
   // ...
```

在 `ForwardList<T>` 中，我们将所有分配构造函数汇聚到一个接受一对迭代器（类型 `It`）作为参数的单个序列构造函数中。这将 `ForwardList<T,A>` 中构造函数所需的变化局部化到那个单一函数中，这简化了我们的任务。

在 `ForwardList<T>` 中，我们通过 `std::forward_iterator` 概念约束了模板参数 `It`，但概念是 C++20 的特性，而我们在这个实现中编译的是 C++17，所以（遗憾的是）我们将暂时放弃这个约束。

必须分步骤执行分配和构造使我们的实现稍微复杂一些，但我认为你们尊贵的读者不会觉得这是不可逾越的：

```cpp
   // ...
   template <class It> // <std::forward_iterator It>
      ForwardList(It b, It e) {
         if(b == e) return;
         try {
            head = alloc.allocate(1);
            alloc.construct(head, *b);
            auto q = head;
            ++nelems;
            for(++b; b != e; ++b) {
               auto ptr = alloc.allocate(1);
               alloc.construct(ptr, *b);
               q->next = ptr;
               q = q->next;
               ++nelems;
            }
         } catch (...) {
            clear();
            throw;
         }
      }
   // ...
```

我们还为 `ForwardList<T>` 编写了插入成员函数，因此这些函数也需要适应使用 `ForwardList<T,A>` 中的分配器。我们有两个 `push_front()` 的重载版本：

```cpp
   // ...
   void push_front(const_reference val) {
      auto p = alloc.allocate(1);
      alloc.construct(p, val);
      p->next = head;
      head = p;
      ++nelems;
   }
   void push_front(T&& val) {
      auto p = alloc.allocate(1);
      alloc.construct(p, std::move(val));
      p->next = head;
      head = p;
      ++nelems;
   }
   // ...
```

我们还为 `insert_after()` 提供了两个重载版本，一个用于插入单个值，另一个用于插入半开区间内的元素。在后一种情况下，由于我们正在为 C++17 编译，我们需要再次放宽对类型 `It` 的 `std::forward_iterator` 约束：

```cpp
   // ...
   iterator
      insert_after(iterator pos, const_reference value) {
      auto p = alloc.allocate(1);
      alloc.construct(p, value);
      p->next = pos.cur->next;
      pos.cur->next = p;
      ++nelems;
      return { p };
   }
   template <class It> // <std::input_iterator It>
      iterator insert_after(iterator pos, It b, It e) {
         for(; b != e; ++b)
            pos = insert_after(pos, *b);
         return pos;
      }
   // ...
```

我们的 `erase_after()` 成员函数也进行了类似的调整：

```cpp
   // ...
   iterator erase_after(iterator pos) {
      if (pos == end() || std::next(pos) == end())
         return end();
      auto p = pos.cur->next->next;
      alloc.destroy(pos.cur->next);
      alloc.deallocate(pos.cur->next, 1);
      --nelems;
      pos.cur->next = p;
      return { p->next };
   }
};
```

这就完成了我们将 `ForwardList<T>` 转换为分配器感知的 `ForwardList<T,A>` 类型的转换。我希望，亲爱的读者，这个过程并没有像一些人担心的那样困难：鉴于我们对本书中提出的原理和基本技术的理解，在这个阶段，将分配器感知集成到容器中应该对大多数人来说是有意义的。

现在我们已经看到了如何编写“传统”迭代器，以及如何使容器分配器感知的示例，你可能想知道使用分配器的优点。我们知道分配器让我们能够控制容器管理内存的方式，但我们可以从这种控制中获得什么？

### 示例用法 – 顺序缓冲区分配器

分配器使用的经典例子是，不是从自由存储中分配内存，而是管理预分配的内存块。这些内存不必来自线程的执行栈，但在实践中通常是这样做的，所以我们的示例代码也将这样做。

在阅读以下示例之前，你需要知道的是：

+   这种类型的分配器是专门为特定用户设计的工具。我们期望用户知道自己在做什么。

+   在我们的例子中，由分配器管理的预分配缓冲区必须适当地对齐以存储其中的对象。如果你想要将这个例子修改为处理任何自然对齐对象的内存分配，需要做额外的工作（你希望分配器提供 `std::max_align_t` 边界对齐的地址，而我们的示例分配器并不这样做）。

+   如果客户端代码尝试“过度分配”，请求比管理缓冲区能提供的更多内存，那么需要特别注意。在这个例子中，我们将像往常一样抛出 `std::bad_alloc`，但存在其他替代方案。

当 bad_alloc 不是一个选项时…

对于某些应用，抛出异常或其他方式失败分配不是一种选择。对于这些应用，如果专门的分配器无法满足分配请求，不应该抛出异常，因为抛出异常意味着“我无法满足这个函数的后置条件。”

当顺序缓冲区分配器耗尽内存时，一些应用程序会简单地调用`::operator new()`并承受不确定的分配时间“打击”，但会在某个地方留下痕迹（可能是日志），表明发生了这种情况。这意味着程序将泄漏内存，但对于某些应用程序（比如每天都会重新启动的股票市场交易程序），可以预期这些泄漏的数量相对较少，而且有痕迹表明发生了泄漏将让程序员在第二天之前查看问题并（希望）修复它。“两害相权取其轻”，正如有些人所说。

我们的顺序缓冲区分配器将看起来像这样：

```cpp
#include <cstdint>
template <class T>
struct seq_buf_allocator {
   using value_type = T;
   // pointer, reference and other aliases are as
   // usual, and so is max_size()
private:
   char *buf;
   pointer cur;
   size_type cap;
public:
   seq_buf_allocator(char *buf, size_type cap) noexcept
      : buf{ buf }, cap{ cap } {
      cur = reinterpret_cast<pointer>(buf);
   }
   // ...
```

如你所见，这个分配器的状态类似于我们在*第十章*中为基于大小的区域所做的：我们知道要管理的缓冲区从哪里开始（`buf`），有多大（`cap`），以及我们在顺序分配过程中的位置（`cur`）。

我们将`cur`设为一个`pointer`类型的对象，以便在之后的`allocate()`成员函数中简化计算，但这只是一个便利，并非必需。

在某种意义上，`allocate()`成员函数非常简单，因为它执行的是常数时间的计算，从底层存储中返回连续分配的对象，甚至在内存释放后也不需要重新使用该内存。`allocate()`中完成的部分工作需要避免过度分配，为此，我们将比较指针，但可能需要比较分配内存块内的指针与块外的指针（这完全取决于我们的参数值）。这可能会导致未定义的行为，这是我们想要避免的，因此我们将指针转换为`std::intptr_t`对象，并比较得到的整数值。

如果我的平台上没有提供`std::intptr_t`呢？

在 C++中，`std::intptr_t`和`std::uintptr_t`类型是条件支持的，这意味着可能存在不提供这些类型别名的供应商。如果你发现自己处于这种不太可能但并非不可能的情况，你可以简单地跟踪分配的对象数量，并将其与`cap`数据成员进行比较，以达到相同的效果。

我们最终得到以下`allocate()`实现，伴随着相应的`deallocate()`成员函数，在这种情况下，实际上是一个空操作：

```cpp
   // ...
   // rebind, address(), construct() and destroy()
   // are all as usual
   pointer allocate(size_type n) {
      auto
         request = reinterpret_cast<
            std::intptr_t
         >(cur + n),
         limit = reinterpret_cast<
            std::intptr_t
         >(buf + cap);
      if(request >= limit)
         throw std::bad_alloc{};
      auto q = cur;
      cur += n;
      return q;
   }
   void deallocate(pointer, size_type) {
   }
};
// ...
```

由于这个分配器是有状态的，我们需要考虑分配器的等价性。在这种情况下，我们将这样做：

```cpp
template <class T, class U>
  constexpr bool operator==(const seq_buf_allocator<T> &a,
                            const seq_buf_allocator<U> &b){
     return a.cur == b.cur; // maybe?
  }
template <class T, class U>
  constexpr bool operator!=(const seq_buf_allocator<T> &a,
                            const seq_buf_allocator<U> &b){
     return !(a == b);
  }
```

这些等价运算符只在特定时刻有意义，但这个分配器类型实际上并不打算在实践中进行复制；如果你计划使用这样的缓冲区并共享其内部状态，你需要考虑原始副本和副本之间如何共享内部状态并保持一致性——在这种情况下我们不需要这样做。

正如你所见，我们在分配时测试溢出，如果分配请求会导致缓冲区溢出，则抛出`std::bad_alloc`，但这只是我们之前在本章中讨论的多种选择之一：

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
#include <iostream>
#include <vector>
struct Data { int n; };
int main() {
   using namespace std::chrono;
   enum { N = 500'000 };
   {
      std::vector<Data> v;
      auto [r, dt] = test([](auto & v) {
         v.reserve(N);
         for(int i = 0; i != N; ++i)
            v.push_back({ i + 1 });
         return v.back();
      }, v);
      std::cout << "vector<Data>:\n\t"
                << v.size()
                << " insertions in "
                << duration_cast<microseconds>(dt).count()
                << " us\n";
   }
   {
      alignas(Data) char buf[N * sizeof(Data)];
      seq_buf_allocator<Data> alloc{ buf, sizeof buf };
      std::vector<Data, seq_buf_allocator<Data>> v(alloc);
      auto [r, dt] = test([](auto & v) {
         v.reserve(N);
         for(int i = 0; i != N; ++i)
            v.push_back({ i + 1 });
         return v.back();
      }, v);
      std::cout
         << "vector<Data, seq_buf_allocator<Data>>:\n\t"
         << v.size()
         << " insertions in "
         << duration_cast<microseconds>(dt).count()
         << " us\n";
   }
   // do the same replacing std::vector with Vector
}
```

在这一点上，你可能需要注意以下几点：

+   测试代码无论选择哪种分配器都是相同的。

+   当使用有状态的分配器时，我们需要使用一个参数化构造函数，该构造函数接受分配器作为参数。

+   使用`seq_buf_allocator<T>`时，缓冲区的大小和对齐的责任落在用户代码的（隐喻性）肩膀上。再次提醒，这是一个专业工具，因此预期用户知道自己在做什么。

+   如果你在一个符合规范的编译器上运行这个测试，你可能会注意到顺序缓冲区分配器的一些有趣的性能，你可能会注意到`Vector<T,A>`比`std::vector<T,A>`表现更好，但`Vector<T,A>`并不像它的`std::`对应物那样完整和严谨。在实践中，请优先使用标准设施。

+   由于堆栈空间是一种有限的资源（通常总共有一到两个兆字节，所以我们可用的空间少于这个），提供给顺序缓冲区分配器的缓冲区大小有限制。尽管如此，这项技术是有用的，并且在实践中被用于低延迟系统中。

+   如果你使用基于节点的容器列表`ForwardList<T,A>`来应用这种分配器，请记住，每个节点都有一个大小开销，因此请相应地计划缓冲区的大小。

当然，那是一个遵守 C++17 标准的实现。自那时以来，关于分配器的变化有哪些？

## 传统分配器与当代标准

如前所述，将分配器类型封装在相关容器类型中的传统方法仍然存在，但分配器本身的表达方式随着时间的推移而发生了变化，并且上一节中的分配器，无论是`small_allocator<T>`还是`seq_buf_allocator<T>`，在 C++20 编译器上按原样编写是无法编译的。在认为这是令人难过的事情之前，要知道我们仍然可以编写这些分配器，但我们必须以更简单的方式编写它们。呼！

### 简化与基于特质的实现的出现

分配器简化工作的第一步是认识到，在大多数情况下，分配器中编写的代码中很大一部分是我们所说的“样板代码”，即从类到类相同的代码，可以被认为是“噪音”。

为了达到这个目的，C++11 引入了`std::allocator_traits<A>`。其想法是，给定某些`typename A::value_type`类型，只要提供了`allocate()`和`deallocate()`的实现，就可以为大多数分配器服务（包括类型别名，如`pointer`或`size_type`）生成合理且高效的默认实现。

以`small_allocator<T>`为例，我们现在可以用以下方式简单地表达整个分配器类型：

```cpp
template <class T>
struct small_allocator {
   using value_type = T;
   T* allocate(std::size_t n) {
      auto p = static_cast<T*>(
         malloc(n * sizeof(value_type))
      );
      if (!p) throw std::bad_alloc{};
      return p;
   }
   void deallocate(T *p, std::size_t) {
      free(p);
   }
};
// ... insert the equality operators here
```

如你所见，这是一个相当简化的表示！这样，一个容器如`Vector<T,A>`现在可以在引用某些分配器`A`的成员时使用`std::allocator_traits<A>`而不是直接使用`A`。由于特性是一个非常薄的抽象层，几乎不带来任何运行时成本，它们对某些成员`M`所做的是“如果`A`公开了成员`M`，则使用`A::M`；否则，这里有一些合理的默认实现。”当然，在实践中这里不会有分支，因为所有内容都是在编译时确定的。

例如，基于我们之前的`small_allocator<T>`类型，考虑到`small_allocator<T>::allocate()`返回`T*`，那么我们可以确定`std::allocator_traits<small_allocator<T>>::pointer`将等同于`T*`，并且一个容器如`Vector<T,A>`将使其`pointer`类型别名对应于`std::allocator_traits<A>::pointer`所表示的类型。

举例来说，`seq_buf_allocator<T>`现在可以这样表示：

```cpp
template <class T>
struct seq_buf_allocator {
   using value_type = T;
   using pointer = T*;
   using size_type = std::size_t;
   char* buf;
   pointer cur;
   size_type cap;
   seq_buf_allocator(char* buf, size_type cap) noexcept
      : buf{ buf }, cap{ cap } {
      cur = reinterpret_cast<pointer>(buf);
   }
   pointer allocate(size_type n) {
      auto request =
         reinterpret_cast<std::intptr_t>(cur + n),
           limit =
         reinterpret_cast<std::intptr_t>(buf + cap);
      if (request > limit) {
         throw std::bad_alloc{};
      }
      auto q = cur;
      cur += n;
      return q;
   }
   void deallocate(pointer, size_type) {
   }
};
// ... insert equality operators here
```

在这种情况下，即使不是必需的，类型`seq_buf_allocator<T>`也公开了`pointer`和`size_type`别名，这意味着对于此类型，`std::allocator_traits`将使用分配器提供的版本，而不是尝试合成一个替代方案。正如你所看到的，当代基于特性的分配器方法非常方便。

类型`std::allocator_traits<A>`究竟提供了哪些服务？嗯，正如预期的那样，此类型公开了`value_type`的常用类型别名（它本身是`A::value_type`的别名），`pointer`，`const_pointer`，`size_type`和`difference_type`。为了方便，它还公开了别名`allocator_type`（相当于`A`）：`void_pointer`和`const_void_pointer`（在大多数情况下分别相当于`void*`和`const void*`）。记住，特性可以被特化，因此，这些看似明显的类型别名有时可能会映射到更复杂的结构。

类型`std::allocator_traits<A>`还公开了分配器的传统服务，但以`static`成员函数的形式，这些函数将分配器作为第一个参数，包括`construct()`，`destroy()`，`allocate()`，`deallocate()`和`max_size()`。C++23 向这个集合中添加了另一个`static`成员函数：`allocate_at_least()`。此函数返回一个由分配的指针和实际分配的块的大小（以对象数量表示）组成的`std::allocation_result`对象，尽管在分配完成后，该内存块中没有对象。

`rebind`机制通过类型`std::rebind_alloc<A>`和`std::rebind_traits<T>`来表示。当克隆一个分配策略（对于节点容器来说主要是这样）时，通过这些设施提供的`typename A::rebind<T>::other`的等效表示有些冗长：

```cpp
// ...
   typename std::allocator_traits<
      A
   >::template rebind_alloc<Node>;
// ...
```

注意到存在`template`关键字，这是为了语法歧义。是的，我知道你现在在想什么：这是一个多么复杂的语言！但在实践中，我们很少需要使用这个关键字，只有在那些编译器会混淆地看到后面的`<`而不知道它是模板签名的一部分还是小于运算符的情况下。

除了`std::allocator_traits<A>`带来的新功能外，还有一些处理分配器生命周期管理的新功能，这是我们多年来学会做的：

+   三个类型别名，告知容器在容器生命周期的关键时刻应该对分配器做什么。这些类型是`propagate_on_container_copy_assignment`（也称为`propagate_on_container_move_assignment`，也称为`propagate_on_container_swap`，也称为`constexpr`函数，返回`true`或`false`（默认情况下，它们等同于`std::false_type`，因为默认情况下，分配器不应该被复制或移动）。例如，如果一个分配器公开类型别名 POCMA，等同于`std::true_type`，那么使用该分配器的容器应该将分配器与分配的数据一起移动。请注意，在这三种情况下，此特性等同于`std::true_type`意味着分配器的复制、移动或交换操作（分别）是`noexcept`的。

+   类型别名`is_always_equal`；这意味着该类型的分配器将不考虑要分配的对象类型进行比较（这减轻了对`operator==()`和`operator!=()`的需求，它们比较相同模板但不同`value_type`别名的两个分配器）。不过，不要在这个问题上花费太多时间；它已经在 C++23 中被弃用，并且很可能会在 C++26 中被移除。

+   `select_on_container_copy_construction()`成员函数。这是一个`static`成员函数，它接受一个分配器，如果其分配器特性表明这是正确的事情，则复制它，否则返回原始分配器。

好吧，这种分配器生命周期管理是新的，可能令人惊讶。我们该如何处理这些信息？

## 管理传统分配器生命周期

容器在移动或复制操作中应该对分配器做什么？好吧，这里有一些细节。

在容器的复制构造函数中，最好的做法可能是使用`select_on_container_copy_construction()`。毕竟，这是该函数的目的。请勿在其他地方使用该函数：它真正适用于容器的复制构造函数。一旦正在构建的容器获得了其分配器，就可以使用该分配器来执行剩余的内存分配任务。

在容器的移动构造函数中，要做的就是移动构造分配器，并从源容器中窃取资源。

在容器的复制赋值运算符中，如果类型别名`propagate_on_container_copy_assignment`等同于`std::true_type`并且两个分配器比较不等，目标容器首先必须释放所有内存（这可能在后续过程中不可能）。超过这个点，如果`propagate_on_container_copy_assignment`等同于`std::true_type`，那么分配器应该被复制赋值。只有完成所有这些，元素才应该被复制。

容器的移动赋值运算符更复杂（记住，*移动*是一种优化，我们希望它能带来回报！）我们面临的选择如下：

+   类型别名`propagate_on_container_move_assignment`等同于`std::true_type`。在这种情况下，要执行的步骤是（a）确保目标容器释放其责任下的所有内存（它可能无法在稍后做到这一点），（b）移动赋值分配器，然后（c）从源容器将内存所有权转移到目标容器。

+   类型别名`propagate_on_container_move_assignment`等同于`std::false_type`并且分配器比较相等。在这种情况下，你可以执行与上一个案例相同的步骤，但不要移动容器。

+   类型别名`propagate_on_container_move_assignment`等同于`std::false_type`并且分配器比较不等。在这种情况下，实际上无法转移所有权，所以最好的办法是将对象本身从源容器移动到目标容器。

幸运的是，所有这些分配器属性都可以在编译时进行测试，因此决策过程不需要产生任何运行时成本。

我们为了简洁所做的事情...

你会注意到我们的`Vector<T,A>`和`ForwardList<T,A>`类型没有执行整个“分配器生命周期管理舞蹈”，以使我们的示例保持合理长度，并且因为我们对分配器复制和移动的管理方式是一个有趣的设计方面，这可能会要求在这本已经相当大的书中至少增加一章。请读者宽容，亲爱的读者。

### 在分配器感知容器中使用基于特质的分配器

在基于特质的传统分配器中，剩余的问题是：容器如何使用它们？

我们首先需要做的是调整我们对标准未初始化内存算法的分配器感知适配。例如，我们个人对`std::uninitialized_copy()`的适配如下：

```cpp
template <class A, class IIt, class OIt>
void uninitialized_copy_with_allocator
   (A &a, IIt bs, IIt es, OIt bd) {
   auto p = bd;
   try {
      for (auto q = bs; q != es; ++q) {
         std::allocator_traits<A>::construct(a, p, *q);
         ++p;
      }
   } catch (...) {
      for (auto q = bd; q != p; ++q)
         std::allocator_traits<A>::destroy(a, q);
      throw;
   }
}
```

正如你所见，我们现在使用`std::allocator_traits<A>`而不是直接使用`A`，这为定制提供了机会，并且由于`std::allocator_traits<A>`的成员函数都是静态的，所以将分配器作为第一个参数传递。相同的调整可以应用于我们编写的其他分配器感知算法的版本，具有相同的调用模式和将分配器作为第一个参数传递。

然后，我们到达了`Vector<T,A>`类型。我们如何调整其实现以使用基于特性的现代分配器？首先要做的事情是调整容器的类型别名来源：

```cpp
template <class T, class A = std::allocator<T>>
class Vector : A { // note: private inheritance
public:
   using value_type =
      typename std::allocator_traits<A>::value_type;
   using size_type =
      typename std::allocator_traits<A>::size_type;
   using pointer =
      typename std::allocator_traits<A>::pointer;
   using const_pointer =
      typename std::allocator_traits<A>::const_pointer;
   using reference = value_type&;
   using const_reference = const value_type&;
   // ...
```

你可能会惊讶，类型别名`reference`和`const_reference`并不是从`std::allocator_traits<A>`中获取的，但这是有原因的。在 C++中，正如本文所述，我们可以设计出类似“智能指针”的行为的类型（我们甚至在这本书中也这样做过；参见*第六章*），因此抽象在分配器提供非原始指针的情况下是有用的，但目前还没有已知的方法来编写“智能引用”（这将需要能够重载`operator.()`，并且关于这一点的提案至今未能被接受）。

唯一的行为类似于`T`的引用类型的类型是…嗯，`T&`。因此，这些类型别名在 C++17 中被弃用，并在 C++20 中被移除。我们仍然可以提供它们来澄清我们的类型成员函数签名，但它们不再是标准所要求的。

对于`Vector<T,A>`的成员函数而言，一般思路是将对`A`成员函数的所有调用替换为对`std::allocator_traits<A>`的`static`成员函数的调用，该函数以对`A`对象的引用作为参数（记住，在我们的`Vector<T,A>`实现中，`A`是容器的`private`基类）。以下是一个示例：

```cpp
   Vector(size_type n, const_reference init)
      : A{},
        elems{ std::allocator_traits<A>::allocate(
           static_cast<A&>(*this), n)
        },
        nelems{ n }, cap{ n } {
      try {
         uninitialized_fill_with_allocator(
            static_cast<A&>(*this), begin(), end(), init
         );
      } catch (...) {
         std::allocator_traits<A>::deallocate(
            static_cast<A&>(*this), elems, capacity()
         );
         throw;
      }
   }
```

如果你对于在数据成员初始化器中使用`*this`感到不适，你可以放心，因为我们只使用了`*this`的`A`部分，并且在那个点上基类子对象已经被完全初始化。这是`*this`的一个安全部分来使用。

同样的调整必须应用于整个容器（在数十个地方）并且显然会使源代码更加冗长，但好消息是这为我们获得了一个零运行时成本的抽象层，并帮助了所有实际编写分配器的开发者。

对于像`ForwardList<T,A>`这样的基于节点的容器，情况类似但略有不同。一方面，类型别名很棘手；其中一些是为用户代码设计的，应该根据容器的`value_type`来表示，而其他则应该基于通过其特性表示的分配器类型：

```cpp
template <class T, class A = std::allocator<T>>
class ForwardList {
public:
   // note: these are the forward-facing types, expressed
   // in terms where T is the value_type
   using value_type = T;
   using size_type =
      typename std::allocator_traits<A>::size_type;
   using pointer = value_type*;
   using const_pointer = const value_type*;
   using reference = value_type&;
   using const_reference = const value_type&;
   // ...
```

在容器内部，我们需要将`A`重新绑定到我们内部`Node`类型的分配器：

```cpp
   // ...
private:
   struct Node {
      value_type value;
      Node *next = nullptr;
      Node(const_reference value) : value { value } {
      }
      Node(value_type &&value) : value{ std::move(value) }{
      }
   };
   using Alloc = typename std::allocator_traits<
      A
   >::template rebind_alloc<Node>;
   Alloc alloc;
   // ...
```

在这一点之后，我们将使用`std::allocator_traits<Alloc>`类型的`static`成员函数来执行内存管理任务，将`alloc`数据成员作为参数传递，如下例所示：

```cpp
   // ...
   void clear() noexcept {
      for(auto p = head; p; ) {
         auto q = p->next;
std::allocator_traits<Alloc>::destroy(alloc, p); 
         std::allocator_traits<Alloc>::deallocate(
            alloc, p, 1
         );
         p = q;
      }
      nelems = 0;
   }
   template <std::forward_iterator It>
      ForwardList(It b, It e) {
         if(b == e) return;
         try {
            head = std::allocator_traits<
               Alloc
            >::allocate(alloc, 1);
            std::allocator_traits<Alloc>::construct(
               alloc, head, *b
            );
            auto q = head;
            ++nelems;
            for(++b; b != e; ++b) {
               auto ptr = std::allocator_traits<
                  Alloc
               >::allocate(alloc, 1);
               std::allocator_traits<
                  Alloc
               >::construct(alloc, ptr, *b);
               q->next = ptr;
               q = q->next;
               ++nelems;
            }
         } catch (...) {
            clear();
            throw;
         }
      }
   // ...
```

当然，同样的技术需要应用于整个容器，但复杂性保持不变。

现在我们已经看到，传统的分配器，其位于容器的类型中，已经从其原始的（相当复杂）合同演变为当代基于特性和简化的实现（容器有些冗长），这让人想到我们已经达到了某种形式的优化。这是对也是错。

## 传统分配器的烦恼

传统方法在运行时对分配器是最佳的，因为可以无开销地调用这种分配器的服务，如果分配器是无状态的，那么在容器中引入分配器在空间上没有成本。还不错！

当然，没有运行时成本并不意味着没有成本：

+   由于额外的（编译时）分层，容器的实现可能会变得相当复杂，编写、理解和维护源代码都有成本。这种专业知识并非普遍存在；当然，亲爱的读者，您拥有它，但其他人并不一定与您分享这种优势。

+   在本质上几乎在所有方面都相同，但在管理内存的方式上不同的两个容器（使用不同分配器的两个容器）在实践中将是不同类型，这可能会在具有多个容器-分配器组合的程序中减慢编译时间。

+   一些可能应该是简单的操作变得更为复杂。例如，如果试图比较容器`v0`和`v1`的相等性，并且如果`v0`是`Vector<T,A0>`而`v1`是`Vector<T,A1>`，那么就需要编写一个`operator==()`函数来处理两种不同的类型……即使容器的分配器可能不是其显著属性之一，并且在这种情况下，在比较两个容器的大小和值时，分配器不应该是关注的焦点。

同样的推理也适用于许多其他与容器相关的操作：在传统方法中，分配器（allocator）是容器类型的一部分，但许多操作与`value_type`相关，与分配器无关。我们是运行时最优的，但我们在代码生成复杂性方面有额外的成本（这可能导致更大的二进制文件，可能会影响运行速度），并且增加维护工作量（包括理解代码的源代码）也有代价。

即使像使分配器类型感知（毕竟，传统的分配器是某些类型`T`的分配器`T`）这样看似简单的事情有时也是具有争议的。毕竟，低级内存分配函数如`std::malloc()`或`::operator new()`是在处理原始字节，所以这是否意味着我们的传统分配器模型是可完善的？

# 多态内存资源分配器

在 C++17 中，C++ 语言添加了所谓的 PMR 分配器。PMR 容器将分配器信息存储为运行时值，而不是其类型的编译时部分。在这个模型中，PMR 容器包含一个指向 PMR 分配器的指针，减少了所需的类型数量，但在使用内存分配服务时增加了虚拟函数调用。

这再次不是无成本的决策，并且与传统模型相比存在权衡：

+   这种新的分配器模型假设容器存储一个指向分配策略的指针，这通常（但不总是）使得 PMR 容器比它们的非 PMR 对应物更大。有趣的是，这也意味着 `std::pmr::vector<T>` 与 `std::vector<T>` 是不同的容器，这有时会导致非常真实的不便。例如，没有隐式的方法可以将 `std::pmr::string` 的内容复制到 `std::string` 中，但幸运的是，编写这样的函数非常简单。

+   每次分配或释放服务调用都会产生多态间接成本。在调用函数执行一些重要计算的程序中，这可能是微不足道的，但当调用函数执行的计算很少时，相同的成本可能会很痛苦。

+   PMR 容器在内存资源上参数化，PMR 内存资源以字节为单位进行交易，而不是以对象为单位。这不清楚这是好事还是坏事（这可能是视角的问题），但两种方法都有效，但以字节（最简单的共同分母）进行交易使得减少程序中的类型数量更容易。

PMR 方法也有其优势：

+   容器的类型不受其分配器类型的影响。所有 PMR 容器仅持有指向所有 PMR 内存资源基类 `std::pmr::memory_resource` 的指针。

+   实现 PMR 分配器所需的工作非常小，因为只需要重写三个虚拟成员函数。这为表达可重用分配器库开辟了途径，例如。

在 PMR 模型下，一个 `std::pmr::polymorphic_allocator<T>` 对象使用一个 `std::pmr::memory_resource*` 来确定内存是如何管理的。在大多数情况下，当设计内存分配策略时，你所做的是编写一个专门化 `std::memory_resource` 的类，并确定使用该策略分配或释放内存的含义。

让我们看看一个简单的 PMR 容器示例，它具有顺序缓冲区内存资源，正如我们刚刚使用传统分配器实现了这样的机制：

```cpp
#include <print>
#include <vector>
#include <string>
#include <memory_resource>
int main() {
   enum { N = 10'000 };
   alignas(int) char buf[N * sizeof(int)]{};
   std::pmr::monotonic_buffer_resource
      res{ std::begin(buf), std::size(buf) };
   std::pmr::vector<int> v{ &res };
   v.reserve(N);
   for (int i = 0; i != N; ++i)
      v.emplace_back(i + 1);
   for (auto n : v)
      std::print("{} ", n);
   std::print("\n {}\n", std::string(70, '-'));
   for (char * p = buf; p != buf + std::size(buf);
        p += sizeof(int))
      std::print("{} ", *reinterpret_cast<int*>(p));
}
```

这相当简单，不是吗？你可能需要注意以下几点：

+   该程序旨在在线程执行栈上的字节数组中“分配”对象。由于这些对象是 `int` 类型，我们确保缓冲区 `buf` 适当对齐，并且足够大，可以容纳要存储的对象。

+   一个名为 `res` 的 `std::pmr::monotonic_buffer_resource` 对象知道要管理的缓冲区从哪里开始以及有多大。它代表了对连续内存的视角。

+   在这个程序中使用的 `std::pmr::vector<int>` 了解 `res` 并使用该资源来分配和释放内存。

就这些了。实际上，这个程序甚至没有从自由存储中分配一个字节来存储 `int` 对象。与过去为了达到类似效果所必须做的事情相比，这可能会让人感到有些欣慰。在程序结束时，遍历字节数组和遍历容器会产生相同的结果。

这工作得很好，并且几乎不需要编写代码，但如果我们想表达类似 `string` 对象的向量，同时希望向量和它存储的 `string` 对象都使用相同的分配策略怎么办？

## 嵌套分配器

嗯，碰巧 PMR 分配器默认会传播分配策略。考虑以下示例：

```cpp
#include <print>
#include <vector>
#include <string>
#include <memory_resource>
int main() {
   auto make_str = [](const char *p, int n) ->
      std::pmr::string {
      auto s = std::string{ p } + std::to_string(n);
      return { std::begin(s), std::end(s) };
   };
   enum { N = 2'000 };
   alignas(std::pmr::string) char buf[N]{};
   std::pmr::monotonic_buffer_resource
      res{ std::begin(buf), std::size(buf) };
   std::pmr::vector<std::pmr::string> v{ &res };
   for (int i = 0; i != 10; ++i)
      v.emplace_back(make_str("I love my instructor ", i));
   for (const auto &s : v)
      std::print("{} ", s);
   std::print("\n {}\n", std::string(70, '-'));
   for (char c : buf)
      std::print("{} ", c);
}
```

此示例也使用堆栈上的缓冲区，但该缓冲区既用于 `std::pmr::vector` 对象及其元数据，也用于其中的 `std::string` 对象。从封装容器到封装容器的分配策略传播是隐式的。

请注意，该程序中的 `make_str` lambda 表达式用于将格式化后以整数结尾的 `std::string` 转换为 `std::pmr::string`。如前所述，从 `std` 命名空间和 `std::pmr` 命名空间中集成类型有时需要一点努力，但这两个命名空间中类的 API 足够相似，使得这种努力仍然是合理的。

如果你使用这个程序，你会注意到 `std::pmr::string` 对象包含预期的文本，但你也许也会从最后一个循环中注意到缓冲区 `buf` 包含（以及其他事物）字符串中的文本。这是因为我们的字符串相当短，并且在大多数标准库实现中，`std::pmr::string` 并不是单独分配的。这清楚地表明，由我们的 `std::pmr::monotonic_buffer_resource` 类型的对象表示的相同分配策略已经从 `std::pmr::vector` 对象传播到了封装的 `std::pmr::string` 对象。

作用域分配器和传统模型

尽管我们在这本书中没有这样做，但使用传统的分配器方法，仍然可以使用作用域分配器系统。如果你好奇，可以自由地探索类型 `std::scoped_allocator_adapter` 以获取更多信息。

我们现在将查看最后一个示例，该示例使用分配器来跟踪内存分配过程。

## 分配器和数据收集

正如我们在*第八章*中编写我们自己的谦逊但功能性的泄漏检测器时所看到的，内存管理工具通常用于收集信息。对于非详尽的列表，要知道一些公司使用它们来跟踪内存碎片化或评估对象在内存中的位置，可能是在寻求优化缓存使用。其他人想要评估在程序执行过程中何时何地发生分配，以了解代码重组是否可能导致更好的性能。当然，检测泄漏是有用的，但我们已经知道了这一点。

作为 PMR 分配使用的第三个也是最后一个示例，我们将实现一个*跟踪资源*，也就是说，我们将跟踪容器从分配和释放请求，以了解该容器所做的某些实现选择。为了这个示例，我们将使用标准库的`std::pmr::vector`并尝试理解它在尝试向满容器插入对象时增加其容量的方法。记住，标准要求操作如`push_back()`具有摊销常数复杂度，这意味着容量应该很少增长，并且大多数插入操作应该花费常数时间。然而，它并没有强制特定的增长策略：例如，一个实现可能以 2 的倍数增长，另一个可能以 1.5 的倍数增长，另一个可能更倾向于 1.67。其他选项也存在；每个选项都有其权衡，每个库都做出自己的选择。

我们将把这个工具表示为类`tracing_resource`，它从`std::pmr::memory_resource`派生，正如`std::pmr`容器所期望的那样。这使得我们能够展示如何轻松地将内存资源类型添加到这个框架中：

+   基类公开了三个需要重写的成员函数：`do_allocate()`，它旨在执行分配请求，`do_deallocate()`，其角色是，不出所料，释放通过`do_allocate()`分配的内存，以及`do_is_equal()`，它旨在让用户代码测试两个内存资源是否相等。请注意，在这种意义上的“相等”意味着从一个分配的内存可以从另一个中释放。

+   由于我们想要跟踪分配请求，但又不想自己实现实际的内存分配策略，我们将使用一个`upstream`资源，它会为我们进行分配和释放。在我们的测试实现中，这个资源将是一个全局资源，通过`std::pmr::new_delete_resource()`获得，该资源调用`::operator new()`和`::operator delete()`来实现这一目标。

+   因此，我们的分配函数将简单地“记录”（在我们的情况下，打印）请求的分配和释放大小，然后将分配工作委托给`upstream`资源。

完整的实现如下：

```cpp
#include <print>
#include <iostream>
#include <vector>
#include <string>
#include <memory_resource>
class tracing_resource : public std::pmr::memory_resource {
   void* do_allocate(
      std::size_t bytes, std::size_t alignment
   ) override {
       std::print ("do_allocate of {} bytes\n", bytes);
       return upstream->allocate(bytes, alignment);
   }
   void do_deallocate(
      void* p, std::size_t bytes, std::size_t alignment
   ) override {
       std::print ("do_deallocate of {} bytes\n", bytes);
       return upstream->deallocate(p, bytes, alignment);
   }
   bool do_is_equal(
      const std::pmr::memory_resource& other
   ) const noexcept override {
       return upstream->is_equal(other);
   }
   std::pmr::memory_resource *upstream;
public:
   tracing_resource(std::pmr::memory_resource *upstream)
      noexcept : upstream{ upstream } {
   }
};
int main() {
   enum { N = 100 };
   tracing_resource tracer{
      std::pmr::new_delete_resource()
   };
   std::pmr::vector<int> v{ &tracer };
   for (int i = 0; i != N; ++i)
      v.emplace_back(i + 1);
   for (auto s : v)
      std::print("{} ", s);
}
```

如果你运行这个非常简单的程序，你将能够对标准库 `std::pmr::vector` 实现的增长策略有一个直观的认识。

## 优点和成本

正如我们所看到的，PMR 模型有很多值得称赞的地方。它使用简单，相对容易理解，并且易于扩展。在许多应用领域，它的速度足够快，可以满足大多数程序员的需。

当然，也有一些领域需要传统分配器模型提供的对执行时间和运行时行为的增加控制：没有来自模型的间接引用，没有对象大小方面的开销……有时，你只需要尽可能多的控制。这意味着这两种模型都有效，并且都有其存在的合理理由。

PMR 分配器的一个非常实际的优点是，它们使得构建可以组合和构建的分配器和资源库变得更容易。标准库从 `<memory_resource>` 头文件提供了一些有用的示例：

+   我们已经看到了函数 `std::pmr::new_delete_resource()`，它提供了一个系统范围内的资源，其中分配和释放是通过 `::operator new()` 和 `::operator delete()` 实现的，就像我们看到的 `std::pmr::monotonic_buffer_resource` 类，它正式化了在现有缓冲区内部进行顺序分配的过程。

+   `std::pmr::synchronized_pool_resource` 和 `std::pmr::unsynchronized_pool_resource` 类模拟从某些大小的块池中分配对象。当然，对于多线程代码，使用同步的版本。

+   有 `std::pmr::get_default_resource()` 和 `std::pmr::set_default_resource()` 函数，分别获取或替换程序的默认内存资源。默认内存资源，正如预期的那样，与 `std::pmr::new_delete_resource()` 函数返回的内容相同。

+   此外，还有一个函数 `std::pmr::null_memory_resource()`，它返回一个永远不会分配资源的对象（其 `do_allocate()` 成员函数在调用时抛出 `std::bad_alloc` 异常）。这作为一个“上游”措施是很有趣的：考虑一个通过 `std::pmr::monotonic_buffer_resource` 实现的顺序缓冲区分配器系统，其中对内存分配的请求可能导致缓冲区溢出。由于默认情况下，内存资源的“上游”使用另一个调用 `::operator new()` 和 `::operator delete()` 的资源，这种潜在的溢出将导致实际的分配，这可能会对性能产生不良影响。为“上游”资源选择 `std::pmr::null_memory_resource` 确保不会发生此类分配。

正如我们所看到和执行的，通过 PMR 模型添加到这个小集合的内存资源并定制容器的行为以适应你的需求是很容易的。

# 摘要

这确实是一个充满事件的一章，不是吗？在*第十二章*和*第十三章*中探讨了显式和隐式内存分配实现之后，本章探讨了分配器以及这些设施如何让我们定制容器中内存分配的行为以满足我们的需求。

我们看到了一个传统的分配器，它嵌入在其封装容器的类型中，是如何实现和使用的。我们使用了一个以连续内存为交易条件的容器，以及一个基于节点的容器。我们还探讨了编写（和使用）此类分配器的任务是如何随着时间演变，最终成为当代基于特性的分配器，这些分配器隐式地综合了大多数分配器服务的默认实现。

我们随后研究了较新的 PMR 分配器模型，它代表了内存分配的不同观点，并讨论了其优点和缺点。凭借本章的知识，你应该有了关于如何定制容器以满足你需求的想法。

我们的旅程即将结束。在下一章（也是最后一章）中，我们将探讨 C++中内存分配的一些当代问题，并开始思考近未来等待我们的是什么。
