# 15

# 基于策略的设计

基于策略的设计是 C++中最著名的模式之一。自从 1998 年引入标准模板库以来，很少有新的想法比基于策略的设计对 C++程序设计方式的影响更大。

基于策略的设计完全是关于灵活性、可扩展性和定制性。这是一种设计软件的方法，可以使软件能够进化，并能够适应不断变化的需求，其中一些需求在最初设计构思时甚至无法预见。一个设计良好的基于策略的系统可以在结构层面上多年保持不变，并在不妥协的情况下满足不断变化的需求和新要求。

不幸的是，这也是构建能够做所有这些事情（如果有人能弄清楚它是如何工作的）的软件的方法。本章的目标是教会你设计和理解前一种类型的系统，同时避免导致后一种类型灾难的过度行为。

本章将涵盖以下主题：

+   策略模式和基于策略的设计

+   C++中的编译时策略

+   基于策略的类的实现

+   策略的使用指南

# 技术要求

本章的示例代码可以在以下 GitHub 链接中找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP_Second_Edition/tree/master/Chapter15`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP_Second_Edition/tree/master/Chapter15)。

# 策略模式和基于策略的设计

经典的策略模式是一种行为设计模式，它允许在运行时选择特定行为的具体算法，通常是从预定义的算法族中选择。这种模式也被称为*策略模式*；其名称早于其在 C++泛型编程中的应用。策略模式的目标是允许设计有更大的灵活性。

注意

在经典的对象导向策略模式中，关于使用哪个具体算法的决定被推迟到运行时。

就像许多经典模式一样，C++中的泛型编程在编译时算法选择上采用相同的方法 - 它允许通过从一系列相关、兼容的算法中选择来对系统行为的特定方面进行编译时定制。我们现在将学习如何在 C++中实现具有策略的类的基础知识，然后继续研究更复杂和多样化的基于策略设计的方法。

## 基于策略设计的原理

当我们设计一个执行某些操作的系统，但具体操作的实施是不确定的、多样的或系统实施后可能发生变化时，应该考虑使用策略模式——换句话说，当我们知道系统必须做什么（*what the system must do*），但不知道如何做（*how*）时。同样，编译时策略（或策略）是实现一个具有特定功能（*what*）的类的方法，但实现该功能的方式不止一种（*how*）。

在本章中，我们将设计一个智能指针类来展示如何使用策略。除了策略之外，智能指针还有许多其他必需和可选的功能，我们不会涵盖所有这些功能——对于智能指针的完整实现，你将被指引到诸如 C++标准智能指针（`unique_ptr`和`shared_ptr`）、Boost 智能指针或 Loki 智能指针（[`loki-lib.sourceforge.net/`](http://loki-lib.sourceforge.net/)）等示例。本章中介绍的材料将帮助你理解这些库的实现者所做的选择，以及如何设计自己的基于策略的类。

一个智能指针的最小初始实现可能看起来像这样：

```cpp
// Example 01
template <typename T>
  T* p_;
  class SmartPtr {
  public:
  explicit SmartPtr(T* p = nullptr) : p_(p) {}
  ~SmartPtr() {
    delete p_;
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
  T& operator*() { return *p_; }
  const T& operator*() const { return *p_; }
  SmartPtr(const SmartPtr&) = delete;
  SmartPtr& operator=(const SmartPtr&) = delete;
  SmartPtr(SmartPtr&& that) :
    p_(std::exchange(that.p_, nullptr)) {}
  SmartPtr& operator=(SmartPtr&& that) {
    delete p_;
    p_ = std::exchange(that.p_, nullptr);
  }
};
```

此指针有一个从相同类型的原始指针构造函数和通常的（对于指针）操作符，即`*`和`->`。这里最有趣的部分是析构函数——当指针被销毁时，它将自动删除对象（在删除之前不需要检查指针的`null`值；`operator delete`需要接受一个空指针并执行无操作）。因此，这个智能指针的预期使用方式如下：

```cpp
// Example 01
Class C { ... };
{
  SmartPtr<C> p(new C);
  ... use p ...
} // Object *p is deleted automatically
```

这是一个 RAII 类的基本示例。RAII 对象——在我们的例子中是智能指针——拥有资源（已构造的对象）并在拥有对象本身被删除时释放（删除）它。在*第五章*“全面审视 RAII”中详细考虑的常见应用，重点是确保在程序退出此作用域时删除在作用域内构造的对象，无论后者是如何完成的（例如，如果在代码的中间某处抛出异常，RAII 析构函数将保证对象被销毁）。

智能指针的两个更多成员函数被提及，不是它们的实现，而是它们的缺失——指针被设计为不可复制的，因为它的拷贝构造函数和赋值运算符都被禁用了。这个有时被忽视的细节对于任何 RAII 类至关重要——由于指针的析构函数会删除所拥有的对象，因此绝对不应该有两个智能指针指向并尝试删除同一个对象。另一方面，移动指针是一个有效的操作：它将所有权从旧指针转移到新指针。移动构造函数对于工厂函数的工作是必要的（至少在 C++17 之前）。

我们这里所拥有的指针是功能性的，但其实现是受限的。特别是，它只能拥有和删除使用标准 `operator new` 构造的对象，并且只能是一个对象。虽然它可以捕获从自定义 `operator new` 获得的指针或指向元素数组的指针，但它并不能正确地删除这样的对象。

我们可以为在用户定义的堆上创建的对象实现不同的智能指针，为在客户端管理的内存中创建的对象实现另一个智能指针，等等，为每种类型的对象构造及其相应的删除方式实现一个。这些指针的大部分代码都会重复——它们都是指针，整个指针-like API 将必须复制到每个类中。我们可以观察到，所有这些不同的类在本质上都是同一类——对于问题“这是什么类型？”的回答总是相同的——*它是一个* *智能指针*。

唯一的区别在于删除的实现方式。这种在行为的一个特定方面有差异但意图相同的情况表明了使用策略模式。我们可以实现一个更通用的智能指针，其中处理对象删除的细节被委托给任何数量的删除策略之一：

```cpp
// Example 02
template <typename T, typename DeletionPolicy>
class SmartPtr {
  T* p_;
  DeletionPolicy deletion_policy_;
  public:
  explicit SmartPtr(
    T* p = nullptr,
    const DeletionPolicy& del_policy = DeletionPolicy()) :
    p_(p), deletion_policy_(del_policy)
  {}
  ~SmartPtr() {
    deletion_policy_(p_);
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
  T& operator*() { return *p_; }
  const T& operator*() const { return *p_; }
  SmartPtr(const SmartPtr&) = delete;
  SmartPtr& operator=(const SmartPtr&) = delete;
  SmartPtr(SmartPtr&& that) :
    p_(std::exchange(that.p_, nullptr)),
    deletion_policy_(std::move(deletion_policy_))
 {}
  SmartPtr& operator=(SmartPtr&& that) {
    deletion_policy_(p_);
    p_ = std::exchange(that.p_, nullptr);
    deletion_policy_ = std::move(deletion_policy_);
  }
};
```

删除策略是一个额外的模板参数，并且将删除策略类型的对象传递给智能指针的构造函数（默认情况下，这样的对象是默认构造的）。删除策略对象存储在智能指针中，并在其析构函数中使用它来删除指针所指向的对象。

在实现基于策略的类的复制和移动构造函数时必须小心：很容易忘记策略也需要移动或复制到新对象中。在我们的例子中，复制被禁用，但移动操作是支持的。它们必须移动的不仅仅是指针本身，还有策略对象。我们像处理任何其他类一样做这件事：通过移动对象（移动指针更复杂，因为它们是内置类型，但所有类都假定能够正确处理自己的移动操作或删除它们）。在赋值运算符中，记住指针当前拥有的对象必须由相应的，即旧的策略删除；只有在这种情况下，我们才将策略从赋值运算符的右侧移动过来。

对于删除策略类型的要求只有一个，那就是它应该是可调用的——策略被调用，就像一个带有一个参数的函数，以及指向必须删除的对象的指针。例如，我们原始指针在对象上调用 `operator delete` 的行为可以用以下删除策略来复制：

```cpp
// Example 02
template <typename T>
struct DeleteByOperator {
  void operator()(T* p) const {
    delete p;
  }
};
```

要使用此策略，我们必须在构造智能指针时指定其类型，并且可以选择性地将此类型的对象传递给构造函数，尽管在这种情况下，默认构造的对象将工作得很好：

```cpp
class C { ... };
SmartPtr<C, DeleteByOperator<C>> p(new C(42));
```

在 C++17 中，**构造模板参数推导（CTAD）** 通常可以推导出模板参数：

```cpp
class C { ... };
SmartPtr p(new C(42));
```

如果删除策略与对象类型不匹配，将报告无效调用 `operator()` 的语法错误。这通常是不希望的：错误消息并不特别友好，并且通常，对策略的要求必须从模板中策略的使用推断出来（我们的策略只有一个要求，但这是我们第一个也是最简单的策略）。为具有策略的类编写的好做法是明确并在一个地方验证和记录策略的所有要求。在 C++20 中，可以使用概念来完成此操作：

```cpp
// Example 03
template <typename T, typename F> concept Callable1 =
  requires(F f, T* p) { { f(p) } -> std::same_as<void>; };
template <typename T, typename DeletionPolicy>
requires Callable1<T, DeletionPolicy>
class SmartPtr {
  ...
};
```

在 C++20 之前，我们可以通过编译时断言来实现相同的结果：

```cpp
// Example 04
template <typename T, typename DeletionPolicy>
requires Callable1<T, DeletionPolicy>
class SmartPtr {
  ...
  static_assert(std::is_same<
    void, decltype(deletion_policy_(p_))>::value, "");
};
```

即使在 C++20 中，你也可能更喜欢 assert 错误消息。这两个选项都实现了相同的目标：它们验证策略满足所有要求，并在代码的一个地方以可读的方式表达这些要求。是否在要求中包含“可移动”取决于你：严格来说，策略只需要是可移动的，如果你需要移动智能指针本身。允许非可移动策略并在需要时才要求移动操作是合理的。

对于以不同方式分配的对象，需要其他删除策略。例如，如果一个对象是在用户提供的包含 `allocate()` 和 `deallocate()` 成员函数的堆对象上创建的，分别用于分配和释放内存，我们可以使用以下堆删除策略：

```cpp
// Example 02
template <typename T> struct DeleteHeap {
  explicit DeleteHeap(Heap& heap) : heap_(heap) {}
  void operator()(T* p) const {
    p->~T();
    heap_.deallocate(p);
  }
  private:
  Heap& heap_;
};
```

另一方面，如果一个对象是在由调用者单独管理的内存中构造的，那么只需要调用对象的析构函数：

```cpp
// Example 02
template <typename T> struct DeleteDestructorOnly {
  void operator()(T* p) const {
    p->~T();
  }
};
```

我们之前提到，由于策略被用作可调用的实体，`deletion_policy_(p_)`，它可以任何可以像函数一样调用的类型。这包括实际的函数：

```cpp
// Example 02
using delete_int_t = void (*)(int*);
void delete_int(int* p) { delete p; }
SmartPtr<int, delete_int_t> p(new int(42), delete_int);
```

模板实例化也是一个函数，可以以相同的方式使用：

```cpp
template <typename T> void delete_T(T* p) { delete p; }
SmartPtr<int, delete_int_t> p(new int(42), delete_T<int>);
```

在所有可能的删除策略中，其中一个通常是最常用的。在大多数程序中，它很可能会是默认的`operator delete`函数的删除。如果是这样，避免每次使用时都指定这个策略，并使其成为默认策略是有意义的：

```cpp
// Example 02
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr {
  ...
};
```

现在，我们的基于策略的智能指针可以像原始版本一样使用，只需删除一个选项：

```cpp
SmartPtr<C> p(new C(42));
```

在这里，第二个模板参数被保留为其默认值，`DeleteByOperator<C>`，并将此类型的默认构造对象传递给构造函数作为默认的第二个参数。

在这一点上，我必须警告你，在实现这样的基于策略的类时可能会犯的一个微妙错误。请注意，策略对象是通过`const`引用在智能指针的构造函数中被捕获的：

```cpp
explicit SmartPtr(T* p = nullptr,
  const DeletionPolicy& del_policy = DeletionPolicy());
```

这里的`const`引用很重要，因为非`const`引用不能绑定到临时对象（我们将在本节稍后考虑右值引用）。然而，策略是通过值存储在对象本身中的，因此必须制作策略对象的副本：

```cpp
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr {
  T* p_;
  DeletionPolicy deletion_policy_;
  ...
};
```

可能会诱使人们避免复制，并在智能指针中通过引用捕获策略：

```cpp
// Example 05
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr {
  T* p_;
  const DeletionPolicy& deletion_policy_;
  ...
};
```

在某些情况下，这甚至可以工作，例如：

```cpp
Heap h;
DeleteHeap<C> del_h(h);
SmartPtr<C, DeleteHeap<C>> p(new (&heap) C, del_h);
```

然而，这对于默认创建智能指针或以临时策略对象初始化的任何其他智能指针都不会起作用：

```cpp
SmartPtr<C> p(new C, DeleteByOperator<C>());
```

这段代码可以编译。不幸的是，它是错误的——临时`DeleteByOperator<C>`对象在调用`SmartPtr`构造函数之前被构造，但在语句结束时被销毁。`SmartPtr`对象内部的引用留成了悬垂引用。乍一看，这不应该让人感到惊讶——当然，临时对象不会比它被创建的语句活得久——它最晚在语句的闭合分号处被删除。一个对语言细节更熟悉的读者可能会问——*标准不是特别扩展了绑定到常量引用的临时对象的生存期吗？* 确实如此；例如：

```cpp
{
  const C& c = C();
  ... c is not dangling! ...
} // the temporary is deleted here
```

在这个代码片段中，临时对象`C()`在句子的末尾没有被删除，而是在它所绑定引用的生命周期结束时才被删除。那么，为什么同样的技巧没有在我们的删除策略对象上起作用呢？答案是，它某种程度上是起作用的 - 当构造函数的参数被评估并绑定到`const`引用参数时创建的临时对象，在其引用的生命周期内没有被销毁，这就是构造函数调用的持续时间。实际上，它本来就不会被销毁 - 在函数参数评估过程中创建的所有临时对象都在包含函数调用的句子的末尾被删除，即关闭分号处。在我们的情况下，函数是对象的构造函数，因此临时对象的生命周期跨越了整个构造函数调用。然而，它并不扩展到对象的生命周期 - 对象的`const`引用成员不是绑定到临时对象，而是绑定到构造函数参数，而构造函数参数本身也是一个`const`引用。

生命周期扩展只能使用一次 - 将引用绑定到临时对象会延长其生命周期。另一个绑定到第一个引用的引用不会做任何事情，如果对象被销毁，它可能会留下悬挂引用（GCC 和 CLANG 的**地址清理器**（**ASAN**）有助于找到这样的错误）。因此，如果策略对象需要作为智能指针的数据成员存储，它必须被复制。

通常，策略对象很小，复制它们是微不足道的。然而，有时策略对象可能具有非平凡的内部状态，复制起来代价高昂。你也可以想象一个不可复制的策略对象。在这些情况下，将参数对象移动到数据成员对象中可能是有意义的。如果我们声明一个类似于移动构造函数的重载，这很容易做到：

```cpp
// Example 06
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr {
  T* p_;
  DeletionPolicy deletion_policy_;
  public:
  explicit SmartPtr(T* p = nullptr,
      DeletionPolicy&& del_policy = DeletionPolicy())
    : p_(p), deletion_policy_(std::move(del_policy))
  {}
  ...
};
```

正如我们所说的，策略对象通常很小，所以复制它们很少成为问题。如果你确实需要两个构造函数，请确保只有一个有默认参数，这样调用无参数或无策略参数的构造函数就不会产生歧义。

我们现在有一个已经实现一次的智能指针类，但其删除实现可以在编译时通过指定删除策略进行定制。我们甚至可以添加一个在类设计时不存在的新删除策略，只要它符合相同的调用接口，它就会正常工作。接下来，我们将考虑实现策略对象的不同方法。

## 策略的实现

在上一节中，我们学习了如何实现最简单的策略对象。只要策略符合接口约定，它可以是任何类型，并且作为数据成员存储在类中。策略对象最常见的是通过模板生成的；然而，它也可以是一个特定于特定指针类型的常规非模板对象，甚至是一个函数。策略的使用仅限于特定的行为方面，例如智能指针拥有的对象的删除。

有几种方法可以实现和使用此类策略。首先，让我们回顾一下具有删除策略的智能指针的声明：

```cpp
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr { ... };
```

接下来，让我们看看我们如何构造一个智能指针对象：

```cpp
class C { ... };
SmartPtr<C, DeleteByOperator<C>> p(
  new C(42), DeleteByOperator<C>());
```

这种设计的缺点立即显现出来——类型`C`在对象`p`的定义中提到了四次——它必须在所有四个地方保持一致，否则代码将无法编译。C++17 允许我们稍微简化定义：

```cpp
SmartPtr p(new C, DeleteByOperator<C>());
```

在这里，构造函数用于从构造函数参数推导出`class`模板的参数，方式类似于函数模板。仍然有两个关于类型`C`的提及必须保持一致。

一种适用于无状态策略以及内部状态不依赖于主模板类型（在我们的例子中，是`SmartPtr`模板的类型`T`）的策略对象实现方法是，将策略本身做成非模板对象，但给它一个模板成员函数。例如，`DeleteByOperator`策略是无状态的（该对象没有数据成员）并且可以不使用类模板来实现：

```cpp
// Example 07
struct DeleteByOperator {
  template <typename T> void operator()(T* p) const {
    delete p;
  }
};
```

这是一个非模板对象，因此不需要类型参数。成员函数模板在需要删除的对象类型上实例化——类型由编译器推导。由于策略对象类型始终相同，我们不必担心在创建智能指针对象时指定一致的类型：

```cpp
// Example 07
SmartPtr<C, DeleteByOperator> p(
  new C, DeleteByOperator());             // Before C++17
SmartPtr p(new C, DeleteByOperator());     // C++17
```

此对象可以直接用于我们的智能指针，无需对`SmartPtr`模板进行任何修改，尽管我们可能想要更改默认模板参数：

```cpp
template <typename T,
          typename DeletionPolicy = DeleteByOperator>
class SmartPtr { ... };
```

更复杂的策略，如堆删除策略，仍然可以使用这种方法实现：

```cpp
struct DeleteHeap {
  explicit DeleteHeap(SmallHeap& heap) : heap_(heap) {}
  template <typename T> void operator()(T* p) const {
    p->~T();
    heap_.deallocate(p);
  }
  private:
  Heap& heap_;
};
```

此策略有一个内部状态——对堆的引用——但在此策略对象中，除了`operator()`成员函数外，没有任何内容依赖于我们需要删除的对象的类型`T`。因此，策略不需要通过对象类型进行参数化。

由于主模板`SmartPtr`在将我们的策略从类模板转换为具有模板成员函数的非模板类时无需更改，因此我们没有理由不能使用相同类中的两种类型策略。实际上，前一小节中的任何模板类策略仍然有效，因此我们可以将一些删除策略实现为类，而将其他策略实现为类模板。后者在策略具有依赖于智能指针对象类型的成员数据类型时很有用。

如果策略作为类模板实现，我们必须指定正确的类型来实例化策略，以便与每个特定的基于策略的类一起使用。在许多情况下，这是一个非常重复的过程 - 相同的类型用于参数化主模板及其策略。如果我们使用整个模板而不是其特定的实例作为策略，我们可以让编译器为我们完成这项工作：

```cpp
// Example 08
template <typename T,
          template <typename> class DeletionPolicy =
                                    DeleteByOperator>
class SmartPtr {
  public:
  explicit SmartPtr(T* p = nullptr,
    const DeletionPolicy<T>& del_policy =
                             DeletionPolicy<T>())
  : p_(p), deletion_policy_(deletion_policy)
  {}
  ~SmartPtr() {
    deletion_policy_(p_);
  }
  ...
};
```

注意第二个模板参数的语法 - `template <typename> class DeletionPolicy`。这被称为*模板模板*参数 - 模板的参数本身也是一个模板。在 C++14 及之前版本中，`class`关键字是必要的；在 C++17 中，它可以被`typename`替换。要使用此参数，我们需要用某种类型实例化它；在我们的例子中，它是主模板类型参数`T`。这确保了在主要智能指针模板及其策略中的对象类型的一致性，尽管构造函数的参数仍然必须用正确的类型构造：

```cpp
SmartPtr<C, DeleteByOperator> p(
  new C, DeleteByOperator<C>());
```

再次，在 C++17 中，类模板参数可以由构造函数推导；这也适用于模板模板参数：

```cpp
SmartPtr p(new C, DeleteByOperator<C>());
```

当类型是从模板实例化时，模板模板参数似乎是一个吸引人的替代方案，为什么我们总是不使用它们呢？首先，正如你所见，它们在灵活性方面略逊于模板类参数：当策略是一个与类本身具有相同第一个参数的模板时，它们在常见情况下可以节省输入，但在任何其他情况下都不起作用（策略可能是一个非模板或需要多个参数的模板）。另一个问题是，按照目前的写法，模板模板参数有一个显著的限制 - 模板参数的数量必须与指定完全匹配，包括默认参数。换句话说，假设我有以下模板：

```cpp
template <typename T, typename Heap = MyHeap> class DeleteHeap { ... };
```

此模板不能用作先前智能指针的参数 - 它有两个模板参数，而我们在`SmartPtr`的声明中只指定了一个（具有默认值的参数仍然是一个参数）。这个限制很容易解决：我们只需要将模板模板参数定义为变长模板：

```cpp
// Example 09
template <typename T,
          template <typename...> class DeletionPolicy =
                                    DeleteByOperator>
class SmartPtr {
  ...
};
```

现在，删除策略模板可以有任何数量的类型参数，只要它们有默认值（`DeletionPolicy<T>`是我们用于`SmartPtr`的，它必须能编译）。相比之下，我们可以使用`DeleteHeap`模板的一个实例来为智能指针提供`DeletionPolicy`作为类型参数，而不是模板模板参数——我们只需要一个类，`DeleteHeap<int, MyHeap>`就能做得很好。

到目前为止，我们总是将策略对象捕获为基于策略的类的数据成员。将类集成到更大的类中的这种做法被称为**组合**。还有其他方法可以让主模板访问策略提供的定制行为算法，我们将在下一部分考虑。

## 策略对象的使用

到目前为止，我们的所有示例都将策略对象存储为类的数据成员。这通常是存储策略的首选方式，但它有一个显著的缺点——数据成员总是有非零大小。考虑我们的具有某种删除策略的智能指针：

```cpp
template <typename T> struct DeleteByOperator {
  void operator()(T* p) const {
    delete p;
  }
};
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr {
  T* p_;
  DeletionPolicy deletion_policy_;
  ...
};
```

注意，策略对象没有数据成员。然而，对象的大小不是零，而是 1 个字节（我们可以通过打印`sizeof(DeleteByOperator<int>)`的值来验证这一点）。这是必要的，因为 C++程序中的每个对象都必须有一个唯一的地址：

```cpp
DeleteByOperator<int> d1;     // &d1 = ....
DeleteByOperator<long> d2; // &d2 must be != &d1
```

当两个对象在内存中连续布局时，它们地址之间的差异是第一个对象的大小（如果需要，加上填充）。为了防止`d1`和`d2`对象位于相同的地址，标准规定它们的大小至少为 1 个字节。

当作为另一个类的数据成员使用时，一个对象将占用至少与其大小相等的空间，在我们的例子中，是 1 个字节。假设指针占用 8 个字节，因此整个对象长度为 9 个字节。但是，对象的大小也必须填充到满足对齐要求的最接近的值——如果指针的地址需要对齐到 8 个字节，对象可以是 8 个字节或 16 个字节，但不能介于两者之间。因此，向类中添加一个空的策略对象最终将其大小从 8 个字节增加到 16 个字节。这纯粹是内存的浪费，通常是不希望的，尤其是对于大量创建的对象，如指针。无法说服编译器创建零大小的数据成员；标准禁止这样做。但是，策略还可以以另一种方式使用，而不产生开销。

组合的替代方法是继承——我们可以将策略作为主类的基类：

```cpp
// Example 10
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>>
class SmartPtr : private DeletionPolicy {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy())
  : DeletionPolicy(std::move(deletion_policy)), p_(p)
  {}
  ~SmartPtr() {
    DeletionPolicy::operator()(p_);
  }
  ...
};
```

这种方法依赖于特定的优化——如果一个基类为空（没有非静态数据成员），它可以完全从派生类的布局中优化掉。这被称为`SmartPtr`类的大小仅取决于其数据成员的必要大小——在我们的例子中，是 8 个字节。

当使用继承策略时，必须在公共继承或私有继承之间做出选择。通常，策略用于为行为的一个特定方面提供实现。这种实现继承通过私有继承来表示。在某些情况下，策略可能用于更改类的公共接口；在这种情况下，应使用公共继承。对于删除策略，我们没有更改类的接口 - 智能指针在其生命周期结束时始终删除对象；唯一的问题是怎样做。因此，删除策略应使用私有继承。

虽然使用 `operator delete` 的删除策略是无状态的，但某些策略具有必须从构造函数中给出的对象中保留的数据成员。因此，通常，基类策略应通过复制或移动到基类中从构造函数参数初始化，类似于我们初始化数据成员的方式。基类总是在派生类的数据成员之前在成员初始化列表中初始化。最后，可以使用 `base_type::function_name()` 语法来调用基类的成员函数；在我们的情况下，`DeletionPolicy::operator()(p_)`。

继承或组合是将策略类集成到主类中的两种选择。通常，应首选组合，除非有使用继承的理由。我们已经看到了这样一个理由 - 空基类优化。如果我们想影响类的公共接口，继承也是一个必要的选项。

我们智能指针目前缺少一些在大多数智能指针实现中常见的重要功能。其中一个功能是释放指针的能力，即防止对象自动销毁。在某些情况下，如果对象通过其他方式销毁，或者如果需要延长对象的生存期并将其所有权传递给另一个拥有资源的对象，这可能很有用。我们可以轻松地将此功能添加到我们的智能指针中：

```cpp
template <typename T,
          typename DeletionPolicy>
class SmartPtr : private DeletionPolicy {
  T* p_;
  public:
  ~SmartPtr() {
    if (p) DeletionPolicy::operator()(p_);
  }
  void release() { p_ = nullptr; }
  ...
};
```

现在，我们可以在我们的智能指针上调用 `p.release()`，析构函数将不会执行任何操作。我们可以将释放功能硬编码到指针中，但有时你可能希望强制执行与指针中相同的删除操作，而不进行释放。这需要使释放功能成为可选的，由另一个策略控制。我们可以添加一个 `ReleasePolicy` 模板参数来控制 `release()` 成员函数是否存在，但它应该做什么呢？当然，我们可以将 `SmartPtr::release()` 的实现移动到策略中：

```cpp
// Example 11
template <typename T> struct WithRelease {
  void release(T*& p) { p = nullptr; }
};
```

现在，`SmartPtr` 的实现只需要调用 `ReleasePolicy::release(p_)` 来将 `release()` 的适当处理委托给策略。但如果我们不希望支持释放功能，应该怎么处理呢？我们的无释放策略可以简单地什么都不做，但这会误导用户——用户期望如果调用了 `release()`，对象就不会被销毁。我们可以在运行时断言并终止程序。这会将程序员逻辑错误——尝试释放一个无释放智能指针——转换为运行时错误。最好的方式是，如果不需要，`SmartPtr` 类根本就不应该有 `release()` 成员函数。这样，错误的代码就无法编译。实现这一点的唯一方法是将策略注入到主要模板的公共接口中。这可以通过公共继承来完成：

```cpp
template <typename T,
          typename DeletionPolicy,
          typename ReleasePolicy>
class SmartPtr : private DeletionPolicy,
                 public ReleasePolicy {
  ...
};
```

现在，如果释放策略有一个名为 `release()` 的公共成员函数，那么 `SmartPtr` 类也有。

这解决了接口问题。现在，只剩下实现的小问题。`release()` 成员函数现在已经移动到策略类中，但它必须操作父类的数据成员 `p_`。一种方法是在构造过程中从派生类传递这个指针的引用到基策略类。这是一个丑陋的实现——它浪费了 8 个字节的内存来存储一个几乎“就在那里”的数据成员的引用，这个数据成员存储在派生类中，紧挨着基类本身。一个更好的方法是从基类转换到正确的派生类。当然，为了使这可行，基类需要知道正确的派生类是什么。这个问题的解决方案是我们在本书中研究的**奇特重复模板模式**（**CRTP**）：策略应该是一个模板（因此我们需要一个模板模板参数），它在派生类类型上实例化。

这样，`SmartPtr` 类既是释放策略的派生类，也是它的模板参数：

```cpp
// Example 11
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          template <typename...> class ReleasePolicy =
                                       WithRelease>
class SmartPtr : private DeletionPolicy,
                 public ReleasePolicy<SmartPtr<T,
                          DeletionPolicy, ReleasePolicy>>
{ ... };
```

注意，`ReleasePolicy` 模板被特化为 `SmartPtr` 模板的实际实例化，包括所有策略，以及 `ReleasePolicy` 本身。

现在，释放策略知道派生类的类型，并且可以将其自身转换成那个类型。这个情况总是安全的，因为正确的派生类在构造时得到了保证：

```cpp
// Example 11
template <typename P> struct WithRelease {
  void release() { static_cast<P*>(this)->p_ = nullptr; }
};
```

模板参数 `P` 将被替换为智能指针的类型。一旦智能指针公开继承自释放策略，策略的公共成员函数 `release()` 就会被继承并成为智能指针公共接口的一部分。

关于释放策略实现的最后一个细节与访问有关。正如我们迄今为止所写的，数据成员`p_`在`SmartPtr`类中是私有的，并且其基类不能直接访问它。解决这个问题的方法是声明相应的基类为派生类的友元：

```cpp
// Example 11
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          template <typename...> class ReleasePolicy =
                                       WithRelease>
class SmartPtr : private DeletionPolicy,
  public ReleasePolicy<SmartPtr<T, DeletionPolicy,
                                   ReleasePolicy>>
{
  friend class ReleasePolicy<SmartPtr>;
  T* p_;
  ...
};
```

注意，在`SmartPtr`类的主体内部，我们不需要重复所有模板参数。简写`SmartPtr`指的是当前实例化的模板。这并不扩展到类声明中开括号之前的部分，因此当我们指定策略作为基类时，我们必须重复模板参数。

无释放策略的编写同样简单：

```cpp
// Example 11
template <typename P> struct NoRelease {};
```

这里没有`release()`函数，所以尝试使用此策略调用智能指针上的`release()`将无法编译。这解决了我们提出的只在有调用意义时才需要`release()`公共成员函数的要求。基于策略的设计是一个复杂的模式，很少只限于一种做事的方式。还有另一种实现相同目标的方法，我们将在本章后面的部分，即在*使用策略控制* *公共接口*的节中，对其进行研究。

政策对象有时还可以以另一种方式使用。这仅适用于任何策略版本都没有内部状态的情况，这是设计上的要求。例如，我们的删除策略有时是无状态的，但引用调用者堆的那个不是，所以这是一个不一定是无状态的策略。释放策略始终可以被认为是无状态的；我们没有理由向其中添加数据成员，但它被限制通过公共继承来使用，因为它的主要效果是注入一个新的公共成员函数。

让我们考虑另一个我们可能想要定制的方面——调试或日志记录。出于调试目的，当对象被智能指针拥有或被删除时打印信息可能很方便。我们可以在智能指针上添加一个调试策略来支持这一点。调试策略只需做一件事，那就是在智能指针构造或销毁时打印一些信息。如果我们将指针的值传递给打印函数，它不需要访问智能指针。因此，我们可以在调试策略中将打印函数声明为静态的，并且根本不需要在智能指针类中存储此策略：

```cpp
// Example 12
template <typename T,
          typename DeletionPolicy,
          typename DebugPolicy = NoDebug>
class SmartPtr : private DeletionPolicy {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy())
  : DeletionPolicy(std::move(deletion_policy)), p_(p) {
    DebugPolicy::constructed(p_);
  }
  ~SmartPtr() {
    DebugPolicy::deleted(p_);
    DeletionPolicy::operator()(p_);
  }
  ...
};
```

为了简单起见，我们省略了释放策略，但多个策略很容易组合。调试策略实现很简单：

```cpp
// Example 12
struct Debug {
  template <typename T>
  static void constructed(const T* p) {
    std::cout << "Constructed SmartPtr for object " <<
      static_cast<const void*>(p) << std::endl;
  }
  template <typename T>
  static void deleted(const T* p) {
    std::cout << "Destroyed SmartPtr for object " <<
      static_cast<const void*>(p) << std::endl;
  }
};
```

我们选择将策略实现为一个具有模板静态成员函数的非模板类。或者，我们也可以将其实现为一个模板，参数化对象类型`T`。策略的无调试版本，即默认版本，甚至更简单。它必须定义相同的函数，但它们什么都不做：

```cpp
// Example 12
struct NoDebug {
  template <typename T>
    static void constructed(const T* p) {}
  template <typename T> static void deleted(const T* p) {}
};
```

我们可以期待编译器在调用位置内联空模板函数，并优化整个调用，因为不需要生成任何代码。

注意，通过选择这种政策的实现方式，我们做出了一些限制性的设计决策——所有版本的调试政策都必须是无状态的。如果我们需要，比如，在调试政策中存储自定义输出流而不是默认的`std::cout`，我们可能会后悔这个决定。但即使在这种情况下，也只有智能指针类的实现需要改变——客户端代码将继续工作而无需任何更改。

我们已经考虑了三种将策略对象纳入基于策略类的方法——通过组合、通过继承（公开或私有），以及仅通过编译时结合，在这种情况下，策略对象在运行时不需要存储在主要对象中。我们现在将转向基于策略设计的更高级技术。

# 高级策略设计

我们在上一节中介绍的技术构成了基于策略设计的基石——策略可以是类、模板实例化或模板（由模板模板参数使用）。策略类可以在编译时组合、继承或静态使用。如果一个策略需要知道主要基于策略的类的类型，可以使用 CRTP。其余的都是在同一主题上的变体，以及巧妙地结合几种技术以实现新的功能。我们现在将考虑这些更高级的技术。

## 构造函数政策

政策可以用来定制实现几乎任何方面，以及改变类接口。然而，当我们尝试使用政策定制类构造函数时，会出现一些独特的挑战。

例如，让我们考虑我们当前智能指针的另一个限制。到目前为止，智能指针拥有的对象总是在智能指针被删除时删除。如果智能指针支持释放，那么我们可以调用`release()`成员函数，并完全负责对象的删除。但我们如何确保这种删除呢？最可能的方式是，我们将让另一个智能指针拥有它：

```cpp
SmartPtr<C> p1(new C);
SmartPtr<C> p2(&*p1); // Now two pointers own one object
p1.release();
```

这种方法冗长且容易出错——我们暂时让两个指针拥有同一个对象。如果此时发生任何导致两个指针都被删除的情况，我们将两次销毁同一个对象。我们还必须记住始终只释放这些指针中的一个。我们应该从更高的角度看待这个问题——我们试图将对象的拥有权从第一个智能指针传递到另一个。

做这件事更好的方法是移动第一个指针到第二个：

```cpp
SmartPtr<C> p1(new C);
SmartPtr<C> p2(std::move(p1));
```

现在，第一个指针保留在移动前的状态，我们可以定义它（唯一的要求是析构函数调用必须是有效的）。我们选择将其定义为不拥有任何对象的指针，即处于释放状态的指针。第二个指针接收对象的所有权，并将适时删除它。

为了支持这个功能，我们必须实现移动构造函数。然而，可能存在某些情况下我们希望阻止所有权的转移。因此，我们可能希望同时拥有可移动和不可移动的指针。这需要另一种策略来控制是否支持移动：

```cpp
template <typename T,
  typename DeletionPolicy = DeleteByOperator<T>,
  typename MovePolicy = MoveForbidden
>
class SmartPtr ...;
```

为了简化，我们已回退到仅使用另一项政策——删除政策。我们考虑的其他政策可以与新的`MovePolicy`一起添加。删除政策可以通过我们已学到的任何一种方式实现。由于它可能从空基优化中受益，我们将继续使用基于继承的实现方式。移动策略可以通过几种不同的方式实现，但继承可能是最简单的方法：

```cpp
// Example 13
template <typename T,
  typename DeletionPolicy = DeleteByOperator<T>,
  typename MovePolicy = MoveForbidden>
class SmartPtr : private DeletionPolicy,
                 private MovePolicy {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy())
    : DeletionPolicy(std::move(deletion_policy)),
      MovePolicy(), p_(p) {}
  … SmartPtr code unchanged …
  SmartPtr(SmartPtr&& that) :
    DeletionPolicy(std::move(that)),
    MovePolicy(std::move(that)),
    p_(std::exchange(that.p_, nullptr)) {}
  SmartPtr(const SmartPtr&) = delete;
};
```

通过使用私有继承将两种策略集成，我们现在有一个具有多个基类的派生对象。在 C++的基于策略的设计中，这种多重继承相当常见，不应让你感到惊讶。这种技术有时被称为*混入*，因为派生类的实现是从基类提供的部分中混合而成的。在 C++中，*混入*这个术语也用来指代与 CRTP 相关的一种完全不同的继承方案，因此这个术语的使用常常造成混淆（在大多数面向对象的语言中，*混入*明确地指代我们在这里看到的多重继承应用）。

我们智能指针类的新特性是移动构造函数。移动构造函数在`SmartPtr`类中无条件存在。然而，它的实现要求所有基类都是可移动的。这为我们提供了一种通过不可移动的移动策略来禁用移动支持的方法：

```cpp
// Example 13
struct MoveForbidden {
  MoveForbidden() = default;
  MoveForbidden(MoveForbidden&&) = delete;
  MoveForbidden(const MoveForbidden&) = delete;
  MoveForbidden& operator=(MoveForbidden&&) = delete;
  MoveForbidden& operator=(const MoveForbidden&) = delete;
};
```

可移动策略要简单得多：

```cpp
// Example 13
struct MoveAllowed {
};
```

我们现在可以构造一个可移动指针和一个不可移动指针：

```cpp
class C { ... };
SmartPtr<C, DeleteByOperator<C>, MoveAllowed> p = ...;
auto p1(std::move(p)); // OK
SmartPtr<C, DeleteByOperator<C>, MoveForbidden> q = ...;
auto q1(std::move(q)); // Does not compile
```

尝试移动一个不可移动指针无法编译，因为其中一个基类`MoveForbidden`是不可移动的（没有移动构造函数）。请注意，在先前的例子中，移动前的指针`p`可以安全地删除，但不能以任何其他方式使用。特别是，它不能被解引用。

当我们处理可移动指针时，提供移动赋值运算符也是有意义的：

```cpp
// Example 13
template <typename T,
  typename DeletionPolicy = DeleteByOperator<T>,
  typename MovePolicy = MoveForbidden>
class SmartPtr : private DeletionPolicy,
                 private MovePolicy {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy())
    : DeletionPolicy(std::move(deletion_policy)),
      MovePolicy(), p_(p) {}
  … SmartPtr code unchanged …
  SmartPtr& operator=(SmartPtr&& that) {
    if (this == &that) return *this;
    DeletionPolicy::operator()(p_);
    p_ = std::exchange(that.p_, nullptr);
    DeletionPolicy::operator=(std::move(that));
    MovePolicy::operator=(std::move(that));
    return *this;
  }
  SmartPtr& operator=(const SmartPtr&) = delete;
};
```

注意自我赋值的检查。与必须对自我赋值不执行任何操作的复制赋值不同，移动赋值受标准的约束较少。唯一确定的要求是自我移动应始终使对象处于一个良好定义的状态（已移动的状态是一个这样的状态）。不执行任何操作的自我移动不是必需的，但也是有效的。还要注意基类是如何进行移动赋值的——最简单的方法是直接调用每个基类的移动赋值运算符。没有必要将派生类`that`转换为每个基类型——这是一个隐式执行的转换。我们绝对不能忘记将已移动的指针设置为`nullptr`，否则，这些指针拥有的对象将被删除两次。

为了简单起见，我们忽略了之前引入的所有策略。这没问题——不是所有的设计都需要通过策略来控制所有内容，而且无论如何，组合多个策略都是非常直接的。然而，这是一个指出不同策略有时相关的好机会——例如，如果我们同时使用释放策略和移动策略，使用可移动的移动策略强烈暗示该对象必须支持释放（已释放的指针类似于已移动的指针）。如果需要，我们可以使用模板元编程来强制策略之间的这种依赖关系。

注意，一个需要禁用或启用构造函数的策略并不一定必须用作基类 - 移动赋值或构造函数也会移动所有数据成员，因此，一个不可移动的数据成员同样可以禁用移动操作。在这里使用继承的更重要原因是空基类优化：如果我们把一个`MovePolicy`数据成员引入我们的类中，它会在 64 位机器上将对象大小从 8 字节增加到 16 字节。

我们考虑过使我们的指针可移动。但复制呢？到目前为止，我们明确禁止了复制——在我们的智能指针中，从一开始就删除了复制构造函数和复制赋值运算符。这到目前为止是有意义的——我们不希望有两个智能指针拥有同一个对象并删除它两次。但还有一种所有权的类型，复制操作是完美的——这就是引用计数共享指针所实现的所有权。这种类型的指针允许复制指针，现在两个指针都平等地拥有指向的对象。维护一个引用计数来统计程序中指向同一对象的指针数量。当拥有特定对象的最后一个指针被删除时，该对象本身也会被删除，因为没有更多的引用指向它。

实现引用计数的共享指针有几种方法，但让我们从类及其策略的设计开始。我们仍然需要一个删除策略，并且让一个策略控制移动和复制操作是有意义的。为了简单起见，我们再次限制自己只探索当前正在探索的策略：

```cpp
// Example 14
template <typename T,
  typename DeletionPolicy = DeleteByOperator<T>,
  typename CopyMovePolicy = NoMoveNoCopy
>
class SmartPtr : private DeletionPolicy,
                 public CopyMovePolicy {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy())
    : DeletionPolicy(std::move(deletion_policy)), p_(p)
  {}
  SmartPtr(SmartPtr&& other) :
    DeletionPolicy(std::move(other)),
    CopyMovePolicy(std::move(other)),
    p_(std::exchange(that.p_, nullptr)) {}
  SmartPtr(const SmartPtr& other) :
    DeletionPolicy(other),
    CopyMovePolicy(other),
    p_(other.p_) {}
  ~SmartPtr() {
    if (CopyMovePolicy::must_delete())
      DeletionPolicy::operator()(p_);
  }
};
```

复制操作不再无条件删除。提供了复制和移动构造函数（为了简洁，省略了两个赋值运算符，但应按照之前的方式实现）。

智能指针析构函数中对象的删除不再是无条件的 - 在引用计数的指针的情况下，复制策略维护引用计数并知道对于特定对象只有一个智能指针副本时。

智能指针类本身提供了策略类的需求。无移动、无复制策略必须禁止所有复制和移动操作：

```cpp
// Example 14
class NoMoveNoCopy {
  protected:
  NoMoveNoCopy() = default;
  NoMoveNoCopy(NoMoveNoCopy&&) = delete;
  NoMoveNoCopy(const NoMoveNoCopy&) = delete;
  NoMoveNoCopy& operator=(NoMoveNoCopy&&) = delete;
  NoMoveNoCopy& operator=(const NoMoveNoCopy&) = delete;
  constexpr bool must_delete() const { return true; }
};
```

此外，不可复制的智能指针在其析构函数中始终删除它所拥有的对象，因此`must_delete()`成员函数应始终返回`true`。请注意，此函数必须由所有复制策略实现，即使它是微不足道的，否则智能指针类将无法编译。然而，我们可以完全期待编译器优化调用并无条件调用析构函数，当使用此策略时。

仅移动策略与之前我们使用的可移动策略类似，但现在我们必须明确启用移动操作并禁用复制操作：

```cpp
// Example 14
class MoveNoCopy {
  protected:
  MoveNoCopy() = default;
  MoveNoCopy(MoveNoCopy&&) = default;
  MoveNoCopy(const MoveNoCopy&) = delete;
  MoveNoCopy& operator=(MoveNoCopy&&) = default;
  MoveNoCopy& operator=(const MoveNoCopy&) = delete;
  constexpr bool must_delete() const { return true; }
};
```

再次强调，删除是无条件的（如果对象被移动，智能指针对象内的指针可以是空的，但这并不阻止我们对其调用`operator delete`）。此策略允许移动构造函数和移动赋值运算符编译；`SmartPtr`类为这些操作提供了正确的实现，不需要策略的额外支持。

基于引用计数的复制策略要复杂得多。在这里，我们必须决定共享指针的实现。最简单的实现是在单独的内存分配中分配引用计数器，该分配由复制策略管理。让我们从一个不允许移动操作的引用计数复制策略开始：

```cpp
// Example 14
class NoMoveCopyRefCounted {
  size_t* count_;
  protected:
  NoMoveCopyRefCounted() : count_(new size_t(1)) {}
  NoMoveCopyRefCounted(const NoMoveCopyRefCounted& other) :
    count_(other.count_)
  {
    ++(*count_);
  }
  NoMoveCopyRefCounted(NoMoveCopyRefCounted&&) = delete;
  ~NoMoveCopyRefCounted() {
    --(*count_);
    if (*count_ == 0) {
      delete count_;
    }
  }
  bool must_delete() const { return *count_ == 1; }
};
```

当具有这种复制策略的智能指针被构造时，会分配并初始化一个新的引用计数器，其值为一（我们有一个智能指针指向特定的对象——我们现在正在构造的那个对象）。当智能指针被复制时，包括复制策略在内的所有基类也会被复制。这个策略的复制构造函数只是简单地增加引用计数。当智能指针被删除时，引用计数会减少。最后一个被删除的智能指针也会删除计数器本身。复制策略还控制指向的对象何时被删除——它发生在引用计数达到一的时候，这意味着我们即将删除指向该对象的最后一个指针。当然，确保在调用`must_delete()`函数之前不删除计数器非常重要。这可以通过基类的析构函数在派生类的析构函数之后运行来保证——最后一个智能指针的派生类将看到计数器的值为一，并将删除对象；然后，复制策略的析构函数将再次减少计数器，看到它降到零，并删除计数器本身。

使用这种策略，我们可以实现对象共享所有权：

```cpp
SmartPtr<C, DeleteByOperator<C>, NoMoveCopyRefCounted> p1{new C};
auto p2(p1);
```

现在，我们有两个指向同一对象的指针，引用计数为两个。当最后一个指针被删除时，对象被删除，前提是在此之前没有创建更多副本。智能指针是可复制的，但不可移动：

```cpp
SmartPtr<C, DeleteByOperator<C>, NoMoveCopyRefCounted> p1{new C};
auto p2(std::move(p1)); // Does not compile
```

通常情况下，一旦支持引用计数复制，可能就没有理由禁止移动操作，除非它们根本不需要（在这种情况下，无移动实现可以稍微高效一些）。为了支持移动操作，我们必须考虑引用计数策略的移动后状态——显然，当它被删除时，它不能减少引用计数，因为移动后的指针不再拥有该对象。最简单的方法是将指针重置到引用计数器，这样它就不再可以从复制策略中访问，但此时复制策略必须支持空计数指针的特殊情况：

```cpp
// Example 15
class MoveCopyRefCounted {
  size_t* count_;
  protected:
  MoveCopyRefCounted() : count_(new size_t(1)) {}
  MoveCopyRefCounted(const MoveCopyRefCounted& other) :
    count_(other.count_)
  {
    if (count_) ++(*count_);
  }
  ~MoveCopyRefCounted() {
    if (!count_) return;
    --(*count_);
    if (*count_ == 0) {
      delete count_;
    }
  }
  MoveCopyRefCounted(MoveCopyRefCounted&& other) :
    count_(std::exchange(other.count_, nullptr)) {}
  bool must_delete() const {
    return count_ && *count_ == 1;
  }
};
```

最后，引用计数复制策略还必须支持赋值操作。这些操作与复制或移动构造函数的实现方式类似（但请注意，在将新值赋给策略之前，必须先使用左侧策略删除左侧对象）。 

正如你所见，一些政策实施可能相当复杂，它们的交互甚至更为复杂。幸运的是，基于策略的设计特别适合编写可测试的对象。这种基于策略的设计应用非常重要，值得特别提及。

## 测试策略

现在，我们将向读者展示如何使用基于策略的设计来编写更好的测试。特别是，可以通过替换策略的特殊测试版本来使代码更容易通过单元测试进行测试。这可以通过用常规版本替换策略的特殊测试版本来实现。让我们通过之前小节中引用计数策略的例子来演示这一点。

该策略的主要挑战当然是维护正确的引用计数。我们可以轻松地开发一些测试，这些测试应该能够测试引用计数的所有边界情况：

```cpp
// Test 1: only one pointer
{
  SmartPtr<C, ... policies ...> p(new C);
} // C should be deleted here
// Test 2: one copy
{
  SmartPtr<C, ... policies ...> p(new C);
  {
    auto p1(p); // Reference count should be 2
  } // C should not be deleted here
} // C should be deleted here
```

实际上测试所有这些代码是否按预期工作是很困难的。我们知道引用计数应该是多少，但我们没有检查它实际是多少的方法（将公共函数`count()`添加到智能指针中可以解决这个问题，但这只是困难中的一小部分）。我们知道对象应该在何时被删除，但很难验证它实际上是否被删除了。如果我们删除对象两次，我们可能会遇到崩溃，但这并不确定。如果对象根本没有被删除，那就更难捕捉到这种情况。一个清理器可以找到这样的问题，至少如果我们使用标准的内存管理，但它们并不在所有环境中都可用，并且除非测试被设计为与清理器一起运行，否则可能会产生非常嘈杂的输出。

幸运的是，我们可以使用策略来让我们的测试能够窥视对象的内部工作原理。例如，如果我们没有在我们的引用计数策略中实现公共的`count()`方法，我们可以为引用计数策略创建一个可测试的包装器：

```cpp
class NoMoveCopyRefCounted {
  protected:
  size_t* count_;
  ...
};
class NoMoveCopyRefCountedTest :
  public NoMoveCopyRefCounted {
  public:
  using NoMoveCopyRefCounted::NoMoveCopyRefCounted;
  size_t count() const { return *count_; }
};
```

注意，我们必须将主复制策略中的`count_`数据成员从私有改为保护。我们也可以将测试策略声明为友元，但那样的话，我们就必须为每个新的测试策略都这样做。现在，我们实际上可以实施我们的测试：

```cpp
// Test 1: only one pointer
{
  SmartPtr<C, ... NoMoveCopyRefCountedTest> p(new C);
  assert(p.count() == 1);
} // C should be deleted here
// Test 2: one copy
{
  SmartPtr<C, ... NoMoveCopyRefCountedTest> p(new C);
  {
  auto p1(p); // Reference count should be 2
    assert(p.count() == 2);
    assert(p1.count() == 2);
    assert(&*p == &*p1);
  } // C should not be deleted here
  assert(p.count == 1);
} // C should be deleted here
```

同样，我们可以创建一个可测量的删除策略，检查对象是否将被删除，或者记录在某个外部日志对象中，表明它实际上已被删除，并测试删除是否已正确记录。我们需要对我们的智能指针实现进行测量，以便调用调试或测试策略：

```cpp
// Example 16:
template <... template parameters ...,
          typename DebugPolicy = NoDebug>
class SmartPtr : ... base policies ... {
  T* p_;
  public:
  explicit SmartPtr(T* p = nullptr,
    DeletionPolicy&& deletion_policy = DeletionPolicy()) :
    DeletionPolicy(std::move(deletion_policy)), p_(p)
  {
    DebugPolicy::construct(this, p);
  }
  ~SmartPtr() {
    DebugPolicy::destroy(this, p_,
                         CopyMovePolicy::must_delete());
  if (CopyMovePolicy::must_delete())
    DeletionPolicy::operator()(p_);
  }
  ...
};
```

调试和生产的（非调试）策略都必须包含类中引用的所有方法，但非调试策略的空方法将被内联并优化为无。

```cpp
// Example 16
struct NoDebug {
  template <typename P, typename T>
  static void construct(const P* ptr, const T* p) {}
  template <typename P, typename T>
  static void destroy(const P* ptr, const T* p,
                      bool must_delete) {}
  ... other events ...
};
```

调试策略各不相同，基本的策略只是记录所有可调试的事件：

```cpp
// Example 16
struct Debug {
  template <typename P, typename T>
  static void construct(const P* ptr, const T* p) {
    std::cout << "Constructed SmartPtr at " << ptr <<
      ", object " << static_cast<const void*>(p) <<
      std::endl;
  }
  template <typename P, typename T>
  static void destroy(const P* ptr, const T* p,
                      bool must_delete) {
    std::cout << "Destroyed SmartPtr at " << ptr <<
      ", object " << static_cast<const void*>(p) <<
      (must_delete ? " is" : " is not") << " deleted" <<
      std::endl;
  }
};
```

更复杂的策略可以验证对象的内部状态是否符合要求，并且类的不变性是否得到维护。

到现在为止，读者可能已经注意到基于策略的对象声明可能相当长：

```cpp
SmartPtr<C, DeleteByOperator<T>, MoveNoCopy,
         WithRelease, Debug> p( ... );
```

这是基于策略设计中最常见的观察问题之一，我们应该考虑一些减轻这种问题的方法。

## 策略适配器和别名

可能最明显的缺点是基于策略的设计中我们必须声明具体对象的方式 - 特别是，必须每次都重复的长策略列表。明智地使用默认参数有助于简化最常用的案例。例如，让我们看看以下的长声明：

```cpp
SmartPtr<C, DeleteByOperator<T>, MoveNoCopy,
         WithRelease, NoDebug>
p( ... );
```

有时，这可以简化为以下内容：

```cpp
SmartPtr<C> p( ... );
```

如果默认值代表了一个可移动的非调试指针最常见的使用情况，并且使用了`operator delete`，那么这可以做到。然而，如果我们不打算使用这些策略，添加它们又有什么意义呢？一个经过深思熟虑的策略参数顺序有助于使更常见的策略组合更短。例如，如果最常见的变体是删除策略，那么可以声明一个新的指针，它具有不同的删除策略和默认剩余策略，而不需要重复我们不需要更改的策略：

```cpp
SmartPtr<C, DeleteHeap<T>> p( ... );
```

这仍然留下了不常用策略的问题。此外，策略通常在添加额外功能后作为设计的一部分添加。这些策略几乎总是添加到参数列表的末尾。否则，需要重写声明基于策略类的所有代码，以重新排序其参数。然而，这些后来添加的策略并不一定是不常用的，这种设计演变可能导致许多策略参数必须明确写出，即使是在它们的默认值上，以便可以更改其中一个尾随参数。

虽然在传统基于策略的设计框架内没有通用的解决方案，但在实践中，通常只有少数常用策略组，然后是一些频繁的变化。例如，我们的大多数智能指针可能使用`operator delete`并支持移动和释放，但我们经常需要在调试和非调试版本之间交替。这可以通过创建适配器来实现，这些适配器将具有许多策略的类转换为一个新的接口，该接口仅暴露我们经常想要更改的策略，并将其他策略固定在其常用值上。任何大型设计都可能需要多个这样的适配器，因为常用的策略集可能不同。

编写此类适配器的最简单方法是使用`using`别名：

```cpp
// Example 17
template <typename T, typename DebugPolicy = NoDebug>
using SmartPtrAdapter =
  SmartPtr<T, DeleteByOperator<T>, MoveNoCopy,
              WithRelease, DebugPolicy>;
```

另一个选择是使用继承：

```cpp
// Example 18
template <typename T, typename DebugPolicy = NoDebug>
class SmartPtrAdapter : public SmartPtr<T,
  DeleteByOperator<T>, MoveNoCopy,
  WithRelease, DebugPolicy>
{...};
```

这创建了一个派生类模板，它固定了基类模板的一些参数，而将其他参数保持为参数化。基类的整个公共接口被继承，但需要特别注意基类的构造函数。默认情况下，它们不会被继承，因此新派生的类将具有默认的编译器生成的构造函数。这可能不是我们想要的，因此我们必须将基类的构造函数（以及可能的赋值运算符）引入派生类：

```cpp
// Example 18
template <typename T, typename DebugPolicy = NoDebug>
class SmartPtrAdapter : public SmartPtr<T,
  DeleteByOperator<T>, MoveNoCopy,
  WithRelease, DebugPolicy>
{
  using base_t = SmartPtr<T, DeleteByOperator<T>,
    MoveNoCopy, WithRelease, DebugPolicy>;
  using base_t::SmartPtr;
  using base_t::operator=;
};
```

`using`别名无疑更容易编写和维护，但如果需要同时适配一些成员函数、嵌套类型等，派生类适配器则提供了更多的灵活性。

当我们需要一个具有预设策略的智能指针，但需要快速更改调试策略时，我们现在可以使用新的适配器。

```cpp
SmartPtrAdapter<C, Debug> p1{new C); // Debug pointer
SmartPtrAdapter<C> p2{new C); // Non-debug pointer
```

正如我们一开始所说的，策略最常见的应用是选择类行为某个方面的特定实现。有时，这种实现上的变化也会反映在类的公共接口上——某些操作可能只适用于某些实现，而不适用于其他实现，确保与实现不兼容的操作不被调用的最佳方式是简单地不提供它。

现在，让我们重新审视使用策略选择性地启用公共接口部分的问题。

## 使用策略来控制公共接口

我们之前曾使用策略以两种方式控制公共接口：首先，通过从策略继承，我们能够注入一个公共成员函数。这种方法相当灵活且强大，但有两个缺点——首先，一旦我们公开继承自策略，我们就无法控制要注入的接口——策略的每个公共成员函数都成为派生类接口的一部分。其次，要以此方式实现任何有用的功能，我们必须让策略类将自己转换为派生类，然后它必须能够访问所有数据成员以及可能的其他策略。我们尝试的第二种方法依赖于构造函数的特定属性——要复制或移动一个类，我们必须复制或移动其所有基类或数据成员；如果其中之一是不可复制的或不可移动的，整个构造函数将无法编译。不幸的是，它通常以一个相当不明显的语法错误而失败——没有找到这个对象的复制构造函数。我们可以将这种技术扩展到其他成员函数，例如赋值运算符，但它会变得复杂。

现在，我们将学习一种更直接的方式来操作基于策略的类的公共接口。首先，让我们区分条件性地禁用现有成员函数和添加新成员函数。前者是合理的且通常安全：如果某个特定实现不支持接口提供的某些操作，那么它们从一开始就不应该被提供。后者是危险的，因为它允许对类的公共接口进行任意和不受控制的扩展。因此，我们将专注于提供基于策略的类的所有可能预期用途的接口，然后在某些策略选择下，禁用该接口的部分功能。

C++语言中已经存在一种机制来有选择性地启用和禁用成员函数。在 C++20 之前，这个机制通常通过概念（如果可用）或`std::enable_if`来实现，但其背后的基础是我们已经在*第七章*中学习过的 SFINAE 惯用法，即*SFINAE、概念和重载解析管理*。在 C++20 中，更强大的概念可以在许多情况下取代`std::enable_if`。

为了说明如何使用 SFINAE 让策略有选择性地启用成员函数，我们将重新实现控制公共`release()`成员函数的策略。我们已经在本章中通过从可能提供或不提供`release()`成员函数的`ReleasePolicy`继承来实现过一次；如果提供了，就必须使用 CRTP 来实现它。现在，我们将使用 C++20 的概念来完成同样的工作。

正如我们刚才所说的，依赖于 SFINAE 和概念的策略不能向类的接口添加任何新的成员函数；它只能禁用其中的一些。因此，第一步是将`release()`函数添加到`SmartPtr`类本身：

```cpp
// Example 19
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename ReleasePolicy = NoRelease>
class SmartPtr : private DeletionPolicy {
  T* p_;
  public:
  void release() { p_ = nullptr; }
  ...
};
```

目前，它始终处于启用状态，因此我们需要使用`ReleasePolicy`的某个属性来有条件地启用它：

```cpp
// Example 19
struct WithRelease {
  static constexpr bool enabled = true;
};
struct NoRelease {
  static constexpr bool enabled = false;
};
```

现在，我们需要使用约束有条件地启用`release()`成员函数：

```cpp
// Example 19
template <...> class SmartPtr ... {
  ...
  void release() requires ReleasePolicy::enabled {
    p_ = nullptr;
  }
};
```

在 C++20 中，我们需要的就这些。请注意，我们不需要从`ReleasePolicy`继承，因为其中除了一个常量值之外没有其他内容。同样地，我们也不需要移动或复制这个策略。

在 C++20 和概念出现之前，我们必须使用`std::enable_if`来启用或禁用特定的成员函数——一般来说，表达式`std::enable_if<value, type>`如果`value`为`true`（它必须是一个编译时，或`constexpr`，布尔值）将会编译并产生指定的`type`。如果`value`为`false`，类型替换将失败（不会产生任何类型结果）。这个模板元函数的正确用途是在 SFINAE 上下文中，类型替换的失败不会导致编译错误，而只是禁用导致失败的函数（更准确地说，是从重载解析集中移除它）。

策略本身根本不需要改变：SFINAE 和约束都需要一个`constexpr bool`值。改变的是用来禁用成员函数的表达式。简单地写成如下形式是有诱惑力的：

```cpp
template <...> class SmartPtr ... {
  ...
  std::enable_if_t<ReleasePolicy::enabled> release() {
    p_ = nullptr;
  }
};
```

不幸的是，这行不通：对于 `NoRelease` 策略，即使我们不尝试调用 `release()`，代码也无法编译。原因是 SFINAE 只在模板参数替换时才起作用（`release()` 函数必须是模板，而且，更重要的是，潜在的替换失败必须发生在模板参数替换过程中。我们不需要任何模板参数来声明 `release()`，但我们必须引入一个虚拟参数来使用 SFINAE：

```cpp
// Example 20
template <...> class SmartPtr ... {
  ...
  template<typename U = T>
  std::enable_if_t<sizeof(U) != 0 &&
                   ReleasePolicy::enabled> release() {
    p_ = nullptr;
  }
};
```

当我们在 *第七章**，SFINAE、概念和重载解析管理* 中描述“概念工具”时，我们看到了这样的“假模板”——在 C++20 之前模仿概念的一种方法。现在我们有一个模板类型参数；它永远不会被使用，并且始终设置为默认值，这并不会改变任何事情。返回类型中的条件表达式使用这个模板参数（尽管表达式依赖于参数的部分永远不会失败）。因此，我们现在处于 SFINAE 规则之内。

现在我们有了选择性地禁用成员函数的方法，我们可以重新审视条件启用构造函数，看看我们如何启用和禁用构造函数。

在 C++20 中，答案是“完全相同的方式。”我们需要一个具有 `constexpr` 布尔值和 `restrict` 约束的策略来禁用任何构造函数：

```cpp
// Example 21
struct MoveForbidden {
  static constexpr bool enabled = false;
};
struct MoveAllowed {
  static constexpr bool enabled = true;
};
```

我们可以使用这个策略来约束任何成员函数，包括构造函数：

```cpp
// Example 21
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename MovePolicy = MoveForbidden>
class SmartPtr : private DeletionPolicy {
  public:
  SmartPtr(SmartPtr&& other)
    requires MovePolicy::enabled :
    DeletionPolicy(std::move(other)),
    p_(std::exchange(other.p_, nullptr)) {}
  ...
};
```

在 C++20 之前，我们必须使用 SFINAE。这里的复杂性在于构造函数没有返回类型，我们必须在其他地方隐藏 SFINAE 测试。此外，我们再次必须使构造函数成为模板。我们还可以再次使用虚拟模板参数：

```cpp
// Example 22
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename MovePolicy = MoveForbidden>
class SmartPtr : private DeletionPolicy {
  public:
  template <typename U = T,
    std::enable_if_t<sizeof(U) != 0 && MovePolicy::enabled,
                     bool> = true>
  SmartPtr(SmartPtr&& other) :
    DeletionPolicy(std::move(other)),
    p_(std::exchange(other.p_, nullptr)) {}
  ...
};
```

如果你使用 *第七章** 中的概念工具，SFINAE、概念和重载解析管理，代码将看起来更简单、更直接，尽管仍然需要一个虚拟模板参数：

```cpp
// Example 22
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename MovePolicy = MoveForbidden>
class SmartPtr : private DeletionPolicy {
  public:
  template <typename U = T,
    REQUIRES(sizeof(U) != 0 && MovePolicy::enabled)>
  SmartPtr(SmartPtr&& other) :
    DeletionPolicy(std::move(other)),
    p_(std::exchange(other.p_, nullptr)) {}
  ...
};
```

现在我们有了完全通用的方法来启用或禁用特定的成员函数，包括构造函数，读者可能会想知道，引入早期方法的意义何在？首先，为了简单起见——`enable_if` 表达式必须在正确的上下文中使用，如果稍有错误，生成的编译器错误并不美观。另一方面，一个不可复制的基类使整个派生类不可复制的概念非常基础，并且每次都有效。这种技术甚至可以在 C++03 中使用，那时 SFINAE 的限制更多，而且更难正确实现。

此外，我们已经看到，有时策略需要向类中添加成员变量而不是（或除了）成员函数。我们的引用计数指针是一个完美的例子：如果某个策略提供了引用计数，它也必须包含计数。成员变量不能使用约束进行限制，因此它们必须来自基策略类。

另一个至少要知道如何通过策略注入公有成员函数的理由是，有时`enable_if`替代方案要求在主类模板中声明所有可能的函数集，然后可以选择性地禁用其中一些。有时，这个函数集是自相矛盾的，不能同时存在。一个例子是一组转换运算符。目前，我们的智能指针不能转换回原始指针。我们可以启用这些转换并要求它们是显式的，或者允许隐式转换：

```cpp
void f(C*);
SmartPtr<C> p(...);
f((C*)(p));     // Explicit conversion
f(p);         // Implicit conversion
```

转换运算符的定义如下：

```cpp
template <typename T, ...>
class SmartPtr ... {
  T* p_;
  public:
  explicit operator T*() { return p_; } // Explicit
  operator T*() { return p_; }          // Implicit
  ...
};
```

我们已经决定不希望这些运算符无条件地存在；相反，我们希望它们由原始转换策略控制。让我们从上次用于启用成员函数的策略的相同方法开始：

```cpp
// Example 23
struct NoRaw {
  static constexpr bool implicit_conv = false;
  static constexpr bool explicit_conv = false;
};
struct ExplicitRaw {
  static constexpr bool implicit_conv = false;
  static constexpr bool explicit_conv = true;
};
struct ImplicitRaw {
  static constexpr bool implicit_conv = true;
  static constexpr bool explicit_conv = false;
};
```

再次，我们将首先编写 C++20 代码，在那里我们可以使用约束来限制显式和隐式运算符：

```cpp
// Example 23
template <typename T, ..., typename ConversionPolicy>
class SmartPtr : ... {
  T* p_;
  public:
  explicit operator T*()
    requires ConversionPolicy::explicit_conv
    { return p_; }
  operator T*()
    requires ConversionPolicy::implicit_conv
    { return p_; }
  explicit operator const T*()
    requires ConversionPolicy::explicit_conv const
    { return p_; }
  operator const T*()
    requires ConversionPolicy::implicit_conv const
    { return p_; }
};
```

为了完整性，我们还提供了转换到`const`原始指针的转换。请注意，在 C++20 中，使用条件显式指定符（另一个 C++20 特性）提供这些运算符有更简单的方法：

```cpp
// Example 24
template <typename T, ..., typename ConversionPolicy>
class SmartPtr : ... {
  T* p_;
  public:
  explicit (ConversionPolicy::explicit_conv)
  operator T*()
    requires (ConversionPolicy::explicit_conv ||
              ConversionPolicy::implicit_conv)
    { return p_; }
  explicit (ConversionPolicy::explicit_conv)
  operator const T*()
    requires (ConversionPolicy::explicit_conv const ||
              ConversionPolicy::implicit_conv const)
    { return p_; }
};
```

在 C++20 之前，我们可以尝试使用`std::enable_if`和 SFINAE 来启用这些运算符之一，再次基于转换策略。问题是，即使后来禁用，我们也不能声明到同一类型的隐式和显式转换。这些运算符一开始就不能在同一个重载集中：

```cpp
// Example 25 – does not compile!
template <typename T, ..., typename ConversionPolicy>
class SmartPtr : ... {
  T* p_;
  public:
  template <typename U = T,
            REQUIRES(ConversionPolicy::explicit_conv)>
  explicit operator T*() { return p_; }
  template <typename U = T,
            REQUIRES(ConversionPolicy::implicit_conv)>
  operator T*() { return p_; }
  ...
};
```

如果我们想在智能指针类中选择这些运算符之一，我们必须让它们由基类策略生成。由于策略需要了解智能指针类型，我们必须再次使用 CRTP。以下是一组策略来控制从智能指针到原始指针的转换：

```cpp
// Example 26
template <typename P, typename T> struct NoRaw {
};
template <typename P, typename T> struct ExplicitRaw {
  explicit operator T*() {
    return static_cast<P*>(this)->p_;
  }
  explicit operator const T*() const {
    return static_cast<const P*>(this)->p_;
  }
};
template <typename P, typename T> struct ImplicitRaw {
  operator T*() {
    return static_cast<P*>(this)->p_;
  }
  operator const T*() const {
    return static_cast<const P*>(this)->p_;
  }
};
```

这些策略将所需的公有成员函数运算符添加到派生类中。由于它们是模板，需要用派生类类型实例化，因此转换策略是一个模板模板参数，其使用遵循 CRTP：

```cpp
// Example 26
template <typename T, ... other policies ...
          template <typename, typename>
          class ConversionPolicy = ExplicitRaw>
class SmartPtr : ... other base policies ...,
  public ConversionPolicy<SmartPtr<... paramerers ...>, T>
{
  T* p_;
  template<typename, typename>
  friend class ConversionPolicy;
  public:
  ...
};
```

再次注意模板模板参数的使用：模板参数`ConversionPolicy`不是一个类型，而是一个模板。当我们从该策略的一个实例继承时，我们必须写出我们`SmartPtr`类的完整类型，包括所有模板参数。我们将转换策略做成一个接受两个参数的模板（第二个参数是对象类型`T`）。我们也可以从第一个模板参数（智能指针类型）推导出类型`T`，这主要是一个风格问题。

选定的转换策略将它的公共接口（如果有），添加到派生类的接口中。一个策略添加了一组显式的转换操作符，而另一个则提供了隐式转换。就像在早期的 CRTP 示例中一样，基类需要访问派生类的私有数据成员。我们可以授予整个模板（及其所有实例化）友情权限，或者更具体地，授予用作每个智能指针基类的特定实例化：

```cpp
friend class ConversionPolicy<
  SmartPtr<T, ... parameters ..., ConversionPolicy>, T>;
```

我们已经学习了多种实现新策略的方法。有时，挑战在于重用我们已有的策略。下一节将展示一种实现方法。

## 重新绑定策略

如我们所见，策略列表可能会变得相当长。通常，我们只想更改一条策略，并创建一个与另一个类似但略有不同的类。至少有两种方法可以做到这一点。

第一种方法非常通用，但有些冗长。第一步是将模板参数作为别名暴露在主模板中。无论如何，这是一个好习惯——如果没有这样的别名，在编译时很难找出模板参数是什么，以防我们需要在模板外使用它。例如，我们有一个智能指针，我们想知道删除策略是什么。到目前为止，最简单的方法是借助智能指针类本身的一些帮助：

```cpp
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename CopyMovePolicy = NoMoveNoCopy,
          template <typename, typename>
            class ConversionPolicy = ExplicitRaw>
class SmartPtr : ... base policies ... {
  T* p_;
  public:
  using value_type = T;
  using deletion_policy_t = DeletionPolicy;
  using copy_move_policy_t = CopyMovePolicy;
  template <typename P, typename T1>
  using conversion_policy_t = ConversionPolicy<P, T1>;
  ...
};
```

注意，我们在这里使用了两种不同类型的别名——对于像`DeletionPolicy`这样的常规模板参数，我们可以使用`using`别名。对于模板模板参数，我们必须使用模板别名，有时称为模板`typedef`——为了使用另一个智能指针重复相同的策略，我们需要知道模板本身，而不是模板实例化，例如`ConversionPolicy<SmartPtr, T>`。现在，如果我们需要创建另一个具有一些相同策略的智能指针，我们可以简单地查询原始对象的策略：

```cpp
// Example 27
SmartPtr<int,
  DeleteByOperator<int>, MoveNoCopy, ImplicitRaw>
  p1(new int(42));
using ptr_t = decltype(p1); // The exact type of p1
SmartPtr<ptr_t::value_type,
  ptr_t::deletion_policy_t, ptr_t::copy_move_policy_t,
  ptr_t::conversion_policy_t> p2;
SmartPtr<double,
  ptr_t::deletion_policy_t, ptr_t::copy_move_policy_t,
  ptr_t::conversion_policy_t> p3;
```

现在，`p2`和`p1`具有完全相同的类型。当然，还有更简单的方法可以做到这一点。但关键是，我们可以更改列表中的任何一种类型，保留其余的，并得到一个像`p1`一样的指针，*除了一个变化。*例如，指针`p2`具有相同的策略，但指向一个`double`。

后者实际上是一个非常常见的案例，并且有一种方法可以在保持其余参数不变的情况下，简化模板到不同类型的*重新绑定*。为此，主模板及其所有策略都需要支持这种重新绑定：

```cpp
// Example 27
template <typename T> struct DeleteByOperator {
  void operator()(T* p) const { delete p; }
  template <typename U>
    using rebind_type = DeleteByOperator<U>;
};
template <typename T,
          typename DeletionPolicy = DeleteByOperator<T>,
          typename CopyMovePolicy = NoMoveNoCopy,
          template <typename, typename>
            class ConversionPolicy = ExplicitRaw>
class SmartPtr : private DeletionPolicy,
  public CopyMovePolicy,
  public ConversionPolicy<SmartPtr<T, DeletionPolicy,
    CopyMovePolicy, ConversionPolicy>, T> {
  T* p_;
  public:
  ...
  template <typename U>
  using rebind = SmartPtr<U,
    typename DeletionPolicy::template rebind<U>,
    CopyMovePolicy, ConversionPolicy>;
};
```

`rebind` 别名定义了一个只有一个参数的新模板——我们可以更改的类型。其余的参数来自主模板本身。其中一些参数也是依赖于主类型 `T` 的类型，并且自身也需要重新绑定（在我们的例子中，是删除策略）。通过选择不重新绑定复制/移动策略，我们强加了一个要求，即这些策略中没有任何一个依赖于主类型，否则这个策略也需要重新绑定。最后，模板转换策略不需要重新绑定——我们在这里可以访问整个模板，因此它将使用新的主类型实例化。现在，我们可以使用重新绑定机制来创建一个*类似*的指针类型：

```cpp
SmartPtr<int,
  DeleteByOperator<int>, MoveNoCopy, ImplicitRaw>
p(new int(42));
using dptr_t = decltype(p)::rebind<double>;
dptr_t q(new double(4.2));
```

如果我们直接访问智能指针类型，我们可以用它来进行重新绑定（例如，在模板上下文中）。否则，我们可以使用 `decltype()` 从此类变量的类型中获取类型。指针 `q` 与 `p` 具有相同的策略，但指向一个 `double`，并且根据类型依赖的策略（如删除策略）进行了相应的更新。

我们已经介绍了策略可以实施和用于定制基于策略类的主要方式。现在是时候回顾我们所学的，并就基于策略设计的使用提出一些一般性指南。

# 建议和指南

基于策略的设计允许在创建精细可定制的类时具有非凡的灵活性。有时，这种灵活性和力量反而会成为良好设计的敌人。在本节中，我们将回顾基于策略设计的优点和缺点，并提出一些一般性建议。

## 基于策略设计的优点

基于策略设计的主要优点是设计的灵活性和可扩展性。从高层次来看，这些是策略模式提供的相同好处，只是在编译时实现。基于策略的设计允许程序员在编译时为系统执行的每个特定任务或操作选择多个算法之一。由于算法的唯一约束是绑定它们的接口对整个系统的要求，因此通过编写新策略来扩展系统也是同样可能的。

在高层次上，基于策略的设计允许软件系统由组件构建。在高层次上，这几乎不是一个新颖的想法，当然也不限于基于策略的设计。基于策略设计的重点是使用组件来定义行为和实现单个类。策略和回调之间有一些相似之处——两者都允许在特定事件发生时执行用户指定的操作。然而，策略比回调更通用——回调是一个函数，而策略是整个类，具有多个函数，可能还有非平凡的内部状态。

这些通用概念转化为设计的一套独特优势，主要围绕灵活性和可扩展性。由于系统的整体结构和其高级组件由高级设计确定，因此策略允许在原始设计强加的约束内进行各种低级定制。策略可以扩展类接口（添加公共成员函数），实现或扩展类的状态（添加数据成员），并指定实现（添加私有成员函数）。原始设计在设定类的整体结构和它们之间的交互时，实际上授权每个策略拥有这些角色中的一个或多个。

结果是一个可扩展的系统，可以修改以应对不断变化的需求，甚至包括在系统设计时未预见或未知的那些需求。整体架构保持稳定，而可能策略的选择及其接口的约束提供了一种系统化、规范化的方式来修改和扩展软件。

## 基于策略设计的缺点

考虑到基于策略的设计的第一个问题，是我们已经遇到的问题——具有特定策略集的基于策略类的声明非常冗长，尤其是如果列表末尾的策略需要更改的话。考虑一下声明一个智能指针，其中包含我们在本章中实现的所有策略，合并在一起：

```cpp
SmartPtr<int, DeleteByOperator<int>, NoMoveNoCopy, ExplicitRaw, WithoutArrow, NoDebug> p;
```

这只是针对智能指针的——一个接口相对简单且功能有限的类。尽管不太可能有人需要具有所有这些定制化可能性的单个指针，但基于策略的类往往有很多策略。这个问题可能最为明显，但实际上并不是最糟糕的。模板别名有助于为特定应用实际使用的少量策略组合提供简洁的名称。在模板上下文中，用作函数参数的智能指针类型会被推导出来，无需显式指定。在常规代码中，可以使用`auto`来节省大量输入，并使代码更加健壮——当必须保持一致性的复杂类型声明被替换为自动生成这些一致类型的方式时，由于在两个不同位置输入略有不同而导致的错误就会消失（一般来说，如果有一个方法可以使编译器生成正确构造的代码，就使用它）。

更为显著，尽管不太明显的问题是，所有这些具有不同策略的策略类型实际上都是不同的类型。两个指向相同对象类型但具有不同删除策略的智能指针是不同类型。两个在其他方面相同但具有不同复制策略的智能指针也是不同类型。这为什么会成为问题呢？考虑一个被调用来处理通过智能指针传递给函数的对象的函数。这个函数不会复制智能指针，因此复制策略应该无关紧要——它永远不会被使用。然而，参数类型应该是什么？没有一种类型可以容纳所有智能指针，即使是功能非常相似的智能指针。

这里有几个可能的解决方案。最直接的一个是将所有使用策略类型的函数都做成模板。这确实简化了编码，并减少了代码重复（至少是源代码的重复），但它也有其自身的缺点——由于每个函数都有多个副本，机器代码会变得更大，并且所有模板代码都必须放在头文件中。

另一个选择是擦除策略类型。我们在*第六章*中看到了类型擦除技术，*理解类型擦除*。类型擦除解决了存在许多相似类型的问题——我们可以使所有智能指针，无论其策略如何，都具有相同的类型（当然，仅限于策略决定实现而不是公共接口）。然而，这代价很高。

模板的一般缺点，尤其是基于策略的设计，在于模板提供了一个零开销的抽象——我们可以用方便的高级抽象和概念来表示我们的程序，但编译器会移除所有这些，内联所有模板，并生成必要的最小代码。类型擦除不仅否定了这一优势，而且产生了相反的效果——它增加了非常高的内存分配和间接函数调用的开销。

最后一个选择是避免使用基于策略的类型，至少对于某些操作来说是这样。有时，这种选择会带来一些额外的成本——例如，一个需要操作对象但不删除或拥有它的函数应该通过引用而不是智能指针来获取对象（参见*第三章*，*内存和所有权*）。除了清楚地表达函数不会拥有对象的事实外，这还巧妙地解决了参数应该是什么类型的问题——引用与来自哪个智能指针无关，都是相同的类型。然而，这是一个有限的方法——大多数情况下，我们确实需要操作整个基于策略的对象，这些对象通常比简单的指针复杂得多（例如，自定义容器通常使用策略实现）。

最后一个缺点是基于策略的类型的一般复杂性，尽管这样的说法应该谨慎对待——重要的是，与什么相比的复杂性？基于策略的设计通常用于解决复杂的设计问题，其中一系列类似类型服务于相同的基本目的（*什么*），但以略有不同的方式（*如何*）。这导致我们对策略使用的建议。

## 基于策略的设计指南

基于策略的设计指南总结为管理复杂性和确保结果合理——设计的灵活性和结果的优雅性应该证明实现复杂性和使用的合理性。

由于大多数复杂性都来自策略数量的增加，这是大多数指南的重点。一些策略最终将非常不同的类型组合在一起，这些类型恰好有类似的实现。这种基于策略的类型的目的是减少代码重复。虽然这是一个值得追求的目标，但这通常不足以将众多不同的策略选项暴露给类型的最终用户。如果两种不同的类型或类型家族恰好有类似的实现，那么这种实现可以被分解出来。设计的私有、隐藏、仅实现部分可以使用策略，如果这使实现更容易的话。

然而，这些隐藏的策略不应该由客户端选择——客户端应该指定在应用程序中有意义的类型和定制可见行为的策略。从这些类型和策略中，实现可以根据需要派生额外的类型。这与调用一个通用函数来执行操作没有什么不同，比如从几个不同且无关的算法中找到序列中的最小元素。通用代码没有被重复，也没有暴露给用户。

因此，何时应该将基于策略的类型分解成两个或更多部分？一个很好的方法是问，具有特定策略集的主要类型是否有描述它的良好特定名称。例如，不可复制的拥有指针，无论是否可移动，是*唯一指针*——在任何给定时间，每个对象只有一个这样的指针。这适用于任何删除或转换策略。

另一方面，引用计数的指针是*共享指针*，再次，可以搭配其他任何策略。这表明我们的“终结所有智能指针的智能指针”可能最好分成两个——一个不可复制的唯一指针和一个可复制的共享指针。我们仍然可以获得一些代码重用，因为删除策略，例如，对于这两种指针类型都是通用的，不需要实现两次。这确实是 C++标准所做的选择。《std::unique_ptr》只有一个策略，即删除策略。《std::shared_ptr》也有相同的策略，可以使用相同的策略对象，但它进行了类型擦除，因此指向特定对象的共享指针都是同一类型。

但是，其他策略又是如何呢？在这里，我们来到了第二个指导原则——限制类使用的策略应该由它们试图防止的错误可能造成的成本来证明其合理性。例如，我们真的需要非移动策略吗？一方面，如果对象的拥有权绝对不能转让，这可以防止编程错误。另一方面，在许多情况下，程序员会简单地更改代码以使用可移动指针。此外，我们被迫使用可移动指针从工厂函数中按值返回它们。然而，不可复制的策略通常是有理由的，并且应该是默认设置。例如，有很好的理由使大多数容器默认不可复制：复制大量数据几乎总是糟糕编码的结果，通常是在向函数传递参数时。

同样，虽然可能希望防止隐式转换为原始指针作为基本的编码纪律，但总有一种方法可以将智能指针显式地转换为原始指针——如果不是其他的话，`&*p`应该始终有效。再次强调，精心限制的接口的好处可能不足以证明添加此策略的合理性。然而，它为一系列可以用来创建更复杂、更有用的策略的技术提供了一个很好的紧凑学习示例，因此我们花费在学习这个策略如何工作上的时间是完全有理由的。

当一个影响公共接口的策略是合理的，我们必须在基于约束的策略和基于 CRTP 的策略之间做出选择，前者限制现有的成员函数，后者添加它们。一般来说，依赖于约束的设计是首选的，即使在 C++20 之前，我们也必须使用“伪概念”。然而，这种方法不能用来向类中添加成员变量，只能添加成员函数。

另一种看待正确策略集和策略应该分成哪些单独群体的问题的方法是回到基于策略设计的根本优势——不同策略表达的行为的可组合性。如果我们有一个具有四个不同策略的类，每个策略都可以有四种不同的实现，那么这就是 256 个不同的类版本。当然，我们不太可能需要所有 256 个。但关键是，在我们实现类的时候，我们不知道我们将来实际上需要哪个版本。我们可以猜测并只实现最有可能的几个。如果我们错了，这将导致大量的代码重复和粘贴。使用基于策略的设计，我们有潜力实现任何行为组合，而实际上并不需要一开始就明确地写出所有这些行为。

现在我们已经了解了基于策略设计的这种优势，我们可以利用它来评估一组特定的策略——它们是否需要可组合的？我们是否需要以不同的方式将它们结合起来？如果某些策略总是以特定的组合或群体出现，这就需要从主要用户指定的策略中自动推导出这些策略。另一方面，一组可以任意组合的相对独立的策略可能是一组很好的策略。

解决基于策略的设计的一些弱点的一种方法是通过不同的手段尝试实现相同的目标。没有替代品可以完全替代策略提供的全部功能——策略模式的存在是有原因的。然而，有一些替代模式提供了一些表面的相似性，并且可以用来解决基于策略的设计所解决的问题。当我们讨论装饰器时，我们将在*第十六章*中看到这样一个替代方案，*适配器和装饰器*。它并不像策略那样通用，但当它起作用时，它可以提供策略的所有优势，特别是可组合性，而不存在一些问题。

# 摘要

在本章中，我们广泛研究了策略模式（也称为策略模式）在 C++泛型编程中的应用。这两种方法的结合产生了 C++程序员武器库中最强大的工具之一——基于策略的类设计。这种方法通过允许我们从许多构建块或策略中组合类的行为，为每个策略负责行为的一个特定方面，提供了极大的灵活性。

我们已经学习了不同的实现策略的方法——这些可以是模板，具有模板成员函数的类，具有静态函数的类，甚至是具有常量值的类。我们可以通过组合、继承或直接访问静态成员来使用策略的方式同样多种多样。策略参数可以是类型或模板，每种都有其自身的优势和局限性。

基于策略的设计这样的强大工具也容易被误用或在不恰当的情况下应用。这种情况通常源于软件逐渐向更复杂的方向发展。为了减轻这种不幸，我们提供了一套指南和建议，这些指南和建议侧重于基于策略的设计为程序员提供的核心优势，并建议了最大化这种优势的技术和约束。

在下一章中，我们将考虑一种更有限的设计模式，有时可以用来模仿基于策略的方法，而不存在其缺点。本章专门介绍装饰器模式以及更通用的适配器模式。两者都是 C++的魔法技巧——它们使一个对象看起来像它不是的东西。

# 问题

1.  策略模式是什么？

1.  策略模式是如何使用 C++泛型编程在编译时实现的？

1.  可以用作策略的类型有哪些？

1.  如何将策略集成到主模板中？

1.  我应该使用具有公共成员函数的策略还是具有约束变量的策略？

1.  基于策略的设计的主要缺点是什么？
