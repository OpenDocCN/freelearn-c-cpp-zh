# 8

# 奇特重复模板模式

我们已经熟悉了继承、多态和虚函数的概念。派生类从基类继承，并通过重写基类的虚函数来自定义基类的行为。所有操作都是在基类的一个实例上多态执行的。当基类对象实际上是派生类的实例时，会调用正确的自定义重写。基类对派生类一无所知，派生类可能甚至在基类代码编写和编译时还没有被编写。**奇特重复模板模式**（**CRTP**）将这个有序的画面颠倒过来，并彻底翻转。

本章将涵盖以下主题：

+   CRTP 是什么？

+   静态多态是什么，它与动态多态有什么区别？

+   虚函数调用的缺点是什么，为什么可能更希望在编译时解决这些调用？

+   CRTP 还有其他什么用途？

# 技术要求

Google Benchmark 库：[`github.com/google/benchmark`](https://github.com/google/benchmark)

示例代码：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter08`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter08)

理解 CRTP

CRTP（Curiously Recurring Template Pattern）这个名称最早由 James Coplien 在 1995 年提出，在他的文章《C++ Report》中。它是一种更一般的有界多态的特殊形式（Peter S. Canning 等人，*面向对象编程的有界多态，功能编程语言和计算机架构会议，1989 年*）。虽然它不是虚函数的一般替代品，但它为 C++程序员提供了一个在适当情况下提供几个优势的类似工具。

## 虚函数有什么问题？

在我们讨论虚函数的更好替代方案之前，我们应该考虑为什么我们想要有替代方案。有什么不喜欢的虚函数？

问题在于性能开销。虚函数调用可能比非虚调用贵几倍，对于本来可以内联但因为是虚函数而不能内联的非常简单的函数来说更是如此（回想一下，虚函数永远不能内联）。我们可以通过微基准测试来衡量这种差异，微基准测试是衡量代码小片段性能的理想工具。现在有很多微基准测试库和工具；在这本书中，我们将使用 Google Benchmark 库。为了跟随本章的示例，您必须首先下载并安装该库（详细说明可以在*第五章*，*RAII 的全面探讨*中找到）。然后，您可以编译并运行示例。

现在我们已经准备好了微基准测试库，我们可以测量虚函数调用的开销。我们将比较一个非常简单的虚函数，代码量最少，与执行相同操作的非虚函数进行对比。下面是我们的虚函数：

```cpp
// Example 01
class B {
  public:
  B() : i_(0) {}
  virtual ~B() {}
  virtual void f(int i) = 0;
  int get() const { return i_; }
  protected:
  int i_;
};
class D : public B {
  public:
  void f(int i) override { i_ += i; }
};
```

下面是等效的非虚函数：

```cpp
// Example 01
class A {
  public:
  A() : i_(0) {}
  void f(int i) { i_ += i; }
  int get() const { return i_; }
  protected:
  int i_;
};
```

我们现在可以在微基准测试环境中调用这两个函数，并测量每个调用所需的时间：

```cpp
void BM_none(benchmark::State& state) {
  A* a = new A;
  int i = 0;
  for (auto _ : state) a->f(++i);
  benchmark::DoNotOptimize(a->get());
  delete a;
}
void BM_dynamic(benchmark::State& state) {
  B* b = new D;
  int i = 0;
  for (auto _ : state) b->f(++i);
  benchmark::DoNotOptimize(b->get());
  delete b;
}
```

`benchmark::DoNotOptimize` 包装器阻止编译器优化掉未使用的对象，以及随之移除的整个函数调用集，因为它们被认为是多余的。注意，在测量虚函数调用时间时存在一个细微之处；编写代码的一个更简单的方法是避免使用 `new` 和 `delete` 操作符，而直接在栈上构造派生对象：

```cpp
void BM_dynamic(benchmark::State& state) {
  D d;
  int i = 0;
  for (auto _ : state) d.f(++i);
  benchmark::DoNotOptimize(b->get());
}
```

然而，这个基准测试可能产生的结果与非虚函数调用相同。原因不是虚函数调用没有开销。相反，在这个代码中，编译器能够推断出对虚函数的调用，即 `f()`，总是调用 `D::f()`（这得益于调用不是通过基类指针，而是通过派生类引用进行的，所以它几乎不可能是其他任何东西）。一个优秀的优化编译器会取消这种调用的虚化，例如，生成一个直接调用 `D::f()` 的调用，而不需要间接和 `v-table` 的引用。这样的调用甚至可以被内联。

另一个可能的复杂情况是，这两个微基准测试，尤其是非虚函数调用，可能太快——基准测试循环的主体可能花费的时间少于循环的开销。我们可以通过在循环体内部进行多次调用来解决这个问题。这可以通过编辑器的复制粘贴功能或使用 C++预处理器宏来完成：

```cpp
#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)
```

现在，在基准测试循环中，我们可以编写以下代码：

```cpp
REPEAT(b->f(++i);)
```

基准测试报告的每次迭代时间现在指的是 `32` 次函数调用。虽然这并不影响比较两次调用，但可能方便基准测试本身通过在基准测试环境末尾添加此行来报告每秒真正的调用次数：

```cpp
state.SetItemsProcessed(32*state.iterations());
```

我们现在可以比较两个基准测试的结果：

```cpp
Benchmark           Time UserCounters...
BM_none          1.60 ns items_per_second=19.9878G/s
BM_dynamic       9.04 ns items_per_second=3.54089G/s
```

我们看到虚函数调用几乎是非虚函数调用的 10 倍昂贵。注意，这并不是一个完全公平的比较；虚调用提供了额外的功能。然而，其中一些功能可以通过其他方式实现，而不必付出性能开销。

## 介绍 CRTP

现在，我们将介绍 CRTP，它颠覆了继承的传统：

```cpp
template <typename D> class B {
  ...
};
class D : public B<D> {
  ...
};
```

第一个引人注目的变化是，基类现在是一个`class`模板。派生类仍然从基类继承，但现在是从基类模板的具体实例化继承——独立地！类`B`在类`D`上实例化，而类`D`则从类`B`的该实例化继承，这个实例化是在类`D`上进行的，它又从类`B`继承，以此类推——这就是递归的作用。习惯它吧，因为在本章中你经常会看到它。

这个令人困惑的模式背后的动机是什么？考虑一下，现在基类有了关于派生类的编译时信息。因此，以前是虚拟函数调用现在可以在编译时绑定到正确的函数：

```cpp
// Example 01
template <typename D> class B {
  public:
  B() : i_(0) {}
  void f(int i) { static_cast<D*>(this)->f(i); }
  int get() const { return i_; }
  protected:
  int i_;
};
class D : public B<D> {
  public:
  void f(int i) { i_ += i; }
};
```

调用本身仍然可以在基类指针上进行：

```cpp
B<D>* b = ...;
b->f(5);
```

没有间接引用和虚拟调用的开销。编译器可以在编译时跟踪调用，直到实际调用的函数，甚至可以内联它：

```cpp
void BM_static(benchmark::State& state) {
  B<D>* b = new D;
  int i = 0;
  for (auto _ : state) {
    REPEAT(b->f(++i);)
  }
  benchmark::DoNotOptimize(b->get());
  state.SetItemsProcessed(32*state.iterations());
}
```

基准测试表明，通过 CRTP 进行的函数调用所需的时间与常规函数调用完全相同：

```cpp
Benchmark           Time UserCounters...
BM_none          1.60 ns items_per_second=19.9878G/s
BM_dynamic       9.04 ns items_per_second=3.54089G/s
BM_static        1.55 ns items_per_second=20.646G/s
```

CRTP 的主要限制是基类`B`的大小不能依赖于其模板参数`D`。更普遍地说，类`B`的模板必须实例化，使得类型`D`是一个不完整类型。例如，以下代码将无法编译：

```cpp
template <typename D> class B {
  using T = typename D::T;
  T* p_;
};
class D : public B<D> {
  using T = int;
};
```

这种代码无法编译的认识可能会让人有些惊讶，考虑到它与许多广泛使用的模板非常相似，这些模板引用了它们的模板参数的嵌套类型。例如，考虑以下模板，它将任何具有`push_back()`和`pop_back()`函数的序列容器转换为栈：

```cpp
template <typename C> class stack {
  C c_;
  public:
  using value_type = typename C::value_type;
  void push(const valuetype& v) { c.push_back(v); }
  value_type pop() {
    value_type v = c.back();
    c.pop_back();
    return v;
  }
};
stack<std::vector<int>> s;
```

注意，`using`类型别名`value_type`看起来与我们在尝试声明类`B`时使用的完全相同。那么，`B`中的那个有什么问题？实际上，类`B`本身并没有问题。在类似于我们的栈类的上下文中，它完全可以编译：

```cpp
class A {
  public:
  using T = int;
  T x_;
};
B<A> b; // Compiles with no problems
```

问题不在于类`B`本身，而在于我们对其的预期使用：

```cpp
class D : public B<D> ...
```

在`B<D>`必须被知道的时候，类型`D`还没有被声明。它不能被声明——类`D`的声明需要我们知道基类`B<D>`的确切内容。所以，如果类`D`还没有被声明，编译器怎么知道标识的`D`甚至指的是一个类型呢？毕竟，你不能在完全未知的类型上实例化一个模板。答案就在其中——类`D`是提前声明的，就像我们有了以下代码一样：

```cpp
class A;
B<A> b; // Now does not compile
```

一些模板可以在前置声明的类型上实例化，而另一些则不能。确切规则可以从标准中痛苦地收集到，但精髓是这样的——任何可能影响类大小的元素都必须完全声明。例如，`using T = typename D::T`中对内部声明的类型的引用，将是一个嵌套类的提前声明，这些也是不允许的。

另一方面，类模板的成员函数的体直到被调用时才会实例化。事实上，对于给定的模板参数，成员函数甚至不需要编译，只要它没有被调用。因此，在基类的成员函数内部对派生类的引用、其嵌套类型及其成员函数的引用是完全正常的。此外，由于派生类类型在基类内部被认为是前向声明的，我们可以声明指向它的指针和引用。以下是一个非常常见的 CRTP 基类重构示例，它将静态转换的使用集中在了一个地方：

```cpp
template <typename D> class B {
  ...
  void f(int i) { derived()->f(i); }
  D* derived() { return static_cast<D*>(this); }
};
class D : public B<D> {
  ...
  void f(int i) { i_ += i; }
};
```

基类声明拥有一个指向不完整（前向声明的）类型 `D` 的指针。它就像任何指向不完整类型的指针一样工作；在指针被解引用之前，类型必须完整。在我们的例子中，这发生在成员函数的体内；`B::f()`，正如我们讨论的那样，它只有在客户端代码调用它时才会被编译。

那么，如果我们需要在编写基类时使用派生类的嵌套类型，我们应该怎么办？在函数体内，没有问题。如果我们需要在基类本身中使用嵌套类型，通常有两个原因。第一个是声明成员函数的返回类型：

```cpp
// Example 01a
template <typename D> class B {
  typename D::value_type get() const {
    return static_cast<const D*>(this)->get();
  }
  …
};
D : public B<D> {
  using value_type = int;
  value_type get() const { … };
  …
};
```

正如我们刚才讨论的，这不会编译。幸运的是，这个问题很容易解决，我们只需要让编译器推断出返回类型：

```cpp
// Example 01a
template <typename D> class B {
  auto get() const {
    return static_cast<const D*>(this)->get();
  }
  …
};
```

第二种情况更困难：需要嵌套类型来声明数据成员或参数。在这种情况下，只剩下一种选择：类型应该作为额外的模板参数传递给基类。当然，这会在代码中引入一些冗余，但这是不可避免的：

```cpp
// Example 01a
template <typename T, typename value_type> class B {
  value_type value_;
  …
};
class D : public B<D, int> {
  using value_type = int;
  value_type get() const { … }
  …
};
```

现在我们已经知道了 CRTP 是什么以及如何编码它，让我们看看它能解决哪些设计问题。

# CRTP 和静态多态

由于 CRTP 允许我们用派生类的函数覆盖基类函数，它实现了多态行为。关键的区别在于多态发生在编译时，而不是运行时。

## 编译时多态

正如我们刚才看到的，CRTP 可以用来允许派生类自定义基类的行为：

```cpp
template <typename D> class B {
  public:
  ...
  void f(int i) { static_cast<D*>(this)->f(i); }
  protected:
  int i_;
};
class D : public B<D> {
  public:
  void f(int i) { i_ += i; }
};
```

如果调用基类 `B::f()` 方法，它将调用传递给实际派生类的方法，就像虚拟函数一样。当然，为了充分利用这种多态，我们必须能够通过基类指针调用基类的方法。如果没有这种能力，我们只是在调用我们已知类型的派生类的方法：

```cpp
D* d = ...; // Get an object of type D
d->f(5);
B<D>* b = ...; // Also has to be an object of type D
b->f(5);
```

注意，函数调用看起来与任何带有基类指针的虚拟函数类完全一样。被调用的实际函数，`f()`，来自派生类，`D::f()`。然而，有一个显著的区别——派生类`D`的实际类型必须在编译时已知——基类指针不是`B*`而是`B<D>*`，这意味着派生对象是类型`D`。如果程序员必须知道实际类型，这种*多态*似乎没有太多意义。但是，这是因为我们还没有完全想清楚*编译时多态*真正意味着什么。正如虚拟函数的好处在于我们可以调用我们甚至不知道存在的类型的成员函数一样，静态多态也必须具有同样的好处才能有用。

我们如何编写一个必须为未知类型参数编译的函数？当然，使用函数模板：

```cpp
// Example 01
template <typename D> void apply(B<D>* b, int& i) {
  b->f(++i);
}
```

这是一个模板函数，可以在任何基类指针上调用，并且它自动推断派生类`D`的类型。现在，我们可以编写看起来像常规多态代码的东西：

```cpp
B<D>* b = new D;    // 1
apply(b);         // 2
```

注意，在第一行，对象必须使用实际类型的知识来构建。这始终是这种情况；对于具有虚拟函数的常规运行时多态也是如此：

```cpp
void apply(B* b) { ... }
B* b = new D;    // 1
apply(b);        // 2
```

在两种情况下，在第二行，我们调用了一些只了解基类知识编写的代码。区别在于，在运行时多态中，我们有一个共同的基类和一些操作它的函数。在 CRTP 和静态多态中，有一个共同的基类模板，但没有单个共同的基类（模板不是类型）并且每个操作这个基类模板的函数本身也成为了一个模板。为了使两种多态类型的对称性（而不是等价性！）完整，我们只需要找出另外两个特殊情况：纯虚拟函数和多态析构。让我们从前者开始。

## 编译时纯虚拟函数

在 CRTP 场景中，纯虚拟函数的等价物是什么？纯虚拟函数必须在所有派生类中实现。声明纯虚拟函数的类，或者继承了一个纯虚拟函数但没有重写的类，是一个抽象类；它可以进一步派生，但不能实例化。

当我们考虑静态多态中纯虚拟函数的等价物时，我们意识到我们的 CRTP 实现存在一个主要漏洞。如果我们忘记在派生类中重写*编译时虚拟函数*，`f()`，会发生什么？

```cpp
// Example 02
template <typename D> class B {
  public:
  ...
  void f(int i) { static_cast<D*>(this)->f(i); }
};
class D : public B<D> {
  // no f() here!
};
...
B<D>* b = ...;
b->f(5); // 1
```

这段代码编译时没有错误或警告——在第一行，我们调用`B::f()`，它反过来调用`D::f()`。类`D`没有声明自己的`f()`成员函数版本，所以调用的是从基类继承的版本。也就是说，当然是之前已经见过的成员函数`B::f()`，它再次调用`D::f()`，实际上是`B::f()` `...`，我们得到了一个无限循环。

这里的问题是没有要求我们在派生类中覆盖成员函数`f()`，但如果我们不覆盖，程序就会不完整。问题的根源在于我们将接口和实现混合在一起——基类中的公共成员函数声明表明所有派生类都必须有一个`void f(int)`函数作为它们公共接口的一部分。派生类中相同函数的版本提供了实际实现。我们将在*第十四章*“模板方法模式和不可虚拟语法的非虚拟方法”中介绍如何分离接口和实现，但到目前为止，只需说如果这些函数有不同的名字，我们的生活就会容易得多：

```cpp
// Example 03
template <typename D> class B {
  public:
  ...
  void f(int i) { static_cast<D*>(this)->f_impl(i); }
};
class D : public B<D> {
  void f_impl(int i) { i_ += i; }
};
...
B<D>* b = ...;
b->f(5);
```

现在如果我们忘记实现`D::f_impl()`会发生什么？代码无法编译，因为类`D`中既没有这样的成员函数，也没有通过继承。因此，我们已经实现了一个编译时纯虚函数！请注意，虚拟函数实际上是`f_impl()`，而不是`f()`。

完成这个任务后，我们该如何实现一个常规的虚函数，它有一个默认实现，可以被可选地覆盖？如果我们遵循分离接口和实现的相同模式，我们只需要提供`B::f_impl()`的默认实现：

```cpp
// Example 03
template <typename D> class B {
  public:
  ...
  void f(int i) { static_cast<D*>(this)->f_impl(i); }
  void f_impl(int i) {}
};
class D1 : public B<D1> {
  void f_impl(int i) { i_ += i; }
};
class D2 : public B<D2> {
  // No f() here
};
...
B<D1>* b = ...;
b->f(5); // Calls D1::f_impl()
B<D2>* b1 = ...;
b1->f(5); // Calls B::f_impl() by default
```

我们需要处理的最后一个特殊情况是多态销毁。

## 析构函数和多态删除

到目前为止，我们故意避免以某种多态方式处理使用 CRTP 实现的删除对象的问题。实际上，如果你回顾并重新阅读了介绍完整代码的示例，例如在“介绍 CRTP”部分中的基准测试组件`BM_static`，我们要么完全避免了删除对象，要么在栈上构建了一个派生对象。这是因为多态删除带来了额外的复杂性，我们终于准备好处理它了。

首先，让我们注意，在许多情况下，多态删除并不是一个问题。所有对象都是已知其实际类型的情况下创建的。如果构建对象的代码也拥有并最终删除它们，那么关于“被删除对象的类型是什么？”这个问题就永远不会真正被提出。同样，如果对象存储在容器中，它们不是通过基类指针或引用来删除的：

```cpp
template <typename D> void apply(B<D>& b) {
  ... operate on b ...
}
{
  std::vector<D> v;
  v.push_back(D(...)); // Objects created as D
  ...
  apply(v[0]); // Objects processed as B&
} // Objects deleted as D
```

在许多情况下，正如前一个示例所示，对象以其实际类型构建和删除，并且在此过程中没有涉及多态，但在这之间对它们进行操作的代码是通用的，编写为针对基类型工作，因此，任何从该基类型派生的类。

但是，如果我们需要通过基类指针实际删除对象呢？这并不容易。首先，对`delete`运算符的简单调用将做错事：

```cpp
B<D>* b = new D;
...
delete b;
```

这段代码可以编译。更糟糕的是，甚至那些通常在类有虚拟函数但没有虚拟析构函数时发出警告的编译器在这种情况下也不会生成任何警告，因为没有虚拟函数，并且编译器不将 CRTP 多态识别为潜在的问题来源。然而，问题在于只调用了基类析构函数`B<D>`本身；`D`的析构函数从未被调用！

你可能会倾向于以处理其他*编译时虚拟*函数相同的方式解决这个问题，通过转换为已知的派生类型并调用派生类的缩进成员函数：

```cpp
template <typename D> class B {
  public:
  ~B() { static_cast<D*>(this)->~D(); }
};
```

与常规函数不同，这种多态尝试严重错误，原因有两个——首先，在基类的析构函数中，实际对象不再是派生类型，对其调用派生类的任何成员函数都会导致未定义行为。其次，即使这 somehow 成功了，派生类的析构函数将执行其工作并调用基类的析构函数，这会导致无限循环。

有两种解决这个问题的方法。一个选择是将编译时多态扩展到与任何其他操作相同的删除操作，使用一个函数模板：

```cpp
template <typename D> void destroy(B<D>* b) {
  delete static_cast<D*>(b);
}
```

这是明确定义的。`delete`运算符被调用在实型的指针上，`D`，并且调用正确的析构函数。然而，你必须注意始终使用这个`destroy()`函数来删除这些对象，而不是调用`delete`运算符。

第二种选择是实际上使析构函数成为虚拟的。这确实会带来虚拟函数调用的开销，但仅限于析构函数。它还会使对象大小增加虚拟指针的大小。如果这两个开销来源都不是问题，你可以使用这种混合静态-动态多态，其中所有虚拟函数调用都在编译时绑定，并且没有开销，除了析构函数之外。

## CRTP 和访问控制

在实现 CRTP 类时，你必须担心访问权限——你想调用的任何方法都必须是可访问的。要么方法必须是公共的，要么调用者必须具有特殊访问权限。这与调用虚函数的方式略有不同——在调用虚函数时，调用者必须有权访问在调用中命名的成员函数。例如，调用基类函数 `B::f()` 要求 `B::f()` 是公共的，或者调用者有权访问非公共成员函数（类 `B` 的另一个成员函数可以调用 `B::f()` 即使它是私有的）。然后，如果 `B::f()` 是虚函数并被派生类 `D` 覆盖，那么覆盖 `D::f()` 实际上是在 `D::f()` 可从原始调用点访问时调用的；例如，`D::f()` 可以是私有的。

CRTP 多态调用的情形略有不同。所有调用在代码中都是显式的，调用者必须有权访问他们调用的函数。通常这意味着基类必须有权访问派生类的成员函数。考虑以下来自早期部分的示例，但现在带有显式访问控制：

```cpp
template <typename D> class B {
  public:
  ...
  void f(int i) { static_cast<D*>(this)->f_impl(i); }
  private:
  void f_impl(int i) {}
};
class D : public B<D> {
  private:
  void f_impl(int i) { i_ += i; }
  friend class B<D>;
};
```

在这里，两个函数，`B::f_impl()` 和 `D::f_impl()`，在各自的类中都是私有的。基类对派生类没有特殊访问权限，无法调用其私有成员函数。除非我们想将成员函数 `D::f_impl()` 从私有改为公共，并允许任何调用者访问它，否则我们必须声明基类为派生类的友元。

反过来操作也有一些好处。让我们创建一个新的派生类 `D1`，它有一个不同的实现函数 `f_impl()` 的覆盖：

```cpp
class D1 : public B<D> {
  private:
  void f_impl(int i) { i_ -= i; }
  friend class B<D1>;
};
```

这个类有一个细微的错误——它实际上并没有从 `B<D1>` 派生，而是从旧类 `B<D>` 派生；在从一个旧模板创建新类时容易犯的错误。如果我们尝试多态地使用这个类，这个错误就会被发现：

```cpp
B<D1>* b = new D1;
```

这无法编译，因为 `B<D1>` 不是 `D1` 的基类。然而，并非所有 CRTP 的使用都涉及多态调用。无论如何，如果在类 `D1` 首次声明时就能捕获这个错误会更好。我们可以通过将类 `B` 变成一种抽象类来实现这一点，仅从静态多态的角度来看。只需将类 `B` 的构造函数设为私有，并将派生类声明为友元即可：

```cpp
template <typename D> class B {
  int i_;
  B() : i_(0) {}
  friend D;
  public:
  void f(int i) { static_cast<D*>(this)->f_impl(i); }
  private:
  void f_impl(int i) {}
};
```

注意友元声明的形式有些不寻常——`friend D` 而不是 `friend class D`。这是为模板参数编写友元声明的方式。现在，唯一可以构造 `B<D>` 类实例的类型是特定派生类 `D`，它用作模板参数，而错误的代码 `class D1 : public B<D>` 现在无法编译。

现在我们知道了 CRTP 的工作原理，让我们看看它有什么用途。

# CRTP 作为一种委托模式

到目前为止，我们已将 CRTP 作为动态多态的编译时等价物使用，包括通过基指针进行的类似虚函数的调用（当然，是编译时，通过模板函数）。这并不是 CRTP 可以使用的唯一方式。实际上，更常见的情况是函数直接在派生类上调用。这是一个非常基本的不同点——通常，公有继承表示 *is-a* 关系——派生对象是一种基对象。接口和泛型代码在基类中，而派生类覆盖了特定的实现。当通过基类指针或引用访问 CRTP 对象时，这种关系仍然成立。这种使用 CRTP 的方式有时也被称为 **静态接口**。

当直接使用派生对象时，情况就完全不同了——基类不再是接口，派生类不仅仅是实现。派生类扩展了基类的接口，基类将一些行为委派给派生类。

## 扩展接口

让我们考虑几个使用 CRTP 将行为从基类委派到派生类的例子。

第一个例子非常简单——对于任何提供 `operator+=()` 的类，我们希望自动生成 `operator+()`，它使用前者：

```cpp
// Example 04
template <typename D> struct plus_base {
  D operator+(const D& rhs) const {
    D tmp = rhs;
    tmp += static_cast<const D&>(*this);
    return tmp;
  }
};
class D : public plus_base<D> {
  int i_;
  public:
  explicit D(int i) : i_(i) {}
  D& operator+=(const D& rhs) {
    i_ += rhs.i_;
    return *this;
  }
};
```

任何以这种方式从 `plus_base` 继承的类都会自动获得一个加法运算符，该运算符保证与提供的增量运算符匹配。你们中的一些人可能会指出，我们在这里声明运算符 `+` 的方式很奇怪。二元运算符不应该是非成员函数吗？的确，它们通常是这样的。标准中没有规定它们必须是，前面的代码在技术上也是有效的。二元运算符如 `==`、`+` 等通常声明为非成员函数的原因与隐式转换有关——如果我们有一个加法操作 `x + y`，并且预期的 `operator+` 是成员函数，它必须是 `x` 对象的成员函数。不是任何可以隐式转换为 `x` 类型的对象，而是 `x` 本身——这是对 `x` 的成员函数调用。相比之下，`y` 对象必须隐式转换为那个成员 `operator+` 的参数类型，通常与 `x` 相同。为了恢复对称性并允许在 `+` 符号的左右两侧进行隐式转换（如果提供了的话），我们必须将 `operator+` 声明为非成员函数。通常，这样的函数需要访问类的私有数据成员，就像之前的例子一样，因此它必须被声明为友元。将所有这些放在一起，我们得到了这个替代实现：

```cpp
// Example 05
template <typename D> struct plus_base {
  friend D operator+(const D& lhs, const D& rhs) {
    D tmp = lhs;
    tmp += rhs;
    return tmp;
  }
};
class D : public plus_base<D> {
  int i_;
  public:
  explicit D(int i) : i_(i) {}
  D& operator+=(const D& rhs) {
    i_ += rhs.i_;
    return *this;
  }
};
```

与我们之前看到的 CRTP 的使用相比，这里有一个显著的区别——程序中将使用的对象是类型`C`，它将永远不会通过`plus_base<C>`的指针来访问。后者并不是任何事物的完整接口，而是一个利用派生类提供的接口的实现。在这里，CRTP 被用作实现技术，而不是设计模式。然而，两者之间的界限并不总是清晰的：一些实现技术非常强大，以至于它们可以改变设计选择。

一个例子是生成的比较和排序操作。在 C++20 中，设计值类型（或任何其他可比较和排序的类型）的接口的推荐选择是只提供两个运算符，`operator==()`和`operator<=>()`。编译器将生成其余部分。如果你喜欢这种接口设计方法，并想在 C++的早期版本中使用它，你需要一种实现它的方法。CRTP 为我们提供了一个可能的实现。我们需要一个基础类，它将从派生类的`operator==()`生成`operator!=()`。它还将生成所有排序运算符；当然，在 C++20 之前我们不能使用`operator<=>()`，但我们可以使用任何我们同意的成员函数名称，例如`cmp()`：

```cpp
template <typename D> struct compare_base {
  friend bool operator!=(const D& lhs, const D& rhs) {
    return !(lhs == rhs); }
  friend bool operator<=(const D& lhs, const D& rhs) {
    return lhs.cmp(rhs) <= 0;
  }
  friend bool operator>=(const D& lhs, const D& rhs) {
    return lhs.cmp(rhs) >= 0;
  }
  friend bool operator< (const D& lhs, const D& rhs) {
    return lhs.cmp(rhs) <  0;
  }
  friend bool operator> (const D& lhs, const D& rhs) {
    return lhs.cmp(rhs) >  0;
  }
};
class D : public compare_base<D> {
  int i_;
  public:
  explicit D(int i) : i_(i) {}
  auto cmp(const D& rhs) const {
    return (i_ < rhs.i_) ? -1 : ((i_ > rhs.i_) ? 1 : 0);
  }
  bool operator==(const D& rhs) const {
    return i_ == rhs.i_;
  }
};
```

在 CRTP 的文献中可以找到许多这样的例子。随着这些例子，你还可以找到关于 C++20 概念是否提供了更好的替代方案的讨论。下一节将解释这是关于什么的。

## CRTP 和概念

第一眼看上去，不清楚概念如何取代 CRTP。概念（你可以在*第七章*，*SFINAE、概念和重载解析管理）都是关于限制接口的，而 CRTP 扩展了接口。

有一些讨论是由那些概念和 CRTP 都可以通过完全不同的方式解决相同问题的案例引发的。回想一下我们使用 CRTP 从`operator+=()`自动生成`operator+()`的情况；我们唯一需要做的就是从特殊的基础类模板继承：

```cpp
// Example 05
template <typename D> struct plus_base {
  friend D operator+(const D& lhs, const D& rhs) { … }
};
class D : public plus_base<D> {
  D& operator+=(const D& rhs) { … }
};
```

我们的基础类有两个作用。首先，它从`operator+=()`生成`operator+()`。其次，它为类提供了一个选择加入这种自动化的机制：要接收生成的`operator+()`，一个类必须继承自`plus_base`。

第一个问题本身很容易解决，我们只需定义一个全局的`operator+()`模板：

```cpp
template <typename T>
T operator+(const T& lhs, const T& rhs) {
  T tmp = lhs;
  tmp += rhs;
  return tmp;
}
```

这个模板有一个“轻微”的问题：我们为程序中的每一个类型都提供了一个全局的`operator+()`，无论它是否需要。此外，大多数时候甚至无法编译，因为并非所有类都定义了`operator+=()`。

这就是概念发挥作用的地方：我们可以限制我们新`operator+()`的应用范围，最终，它只为与我们原本从`plus_base`继承的类型生成，而不为其他类型生成。

做这件事的一种方法是要求数据模板参数类型 T 至少要有增量操作符：

```cpp
template <typename T>
requires( requires(T a, T b) { a += b; } )
T operator+(const T& lhs, const T& rhs) { … }
```

然而，这并不是我们用 CRTP 得到的相同结果。在某些情况下，它可能是一个更好的结果：我们不需要为每个类选择加入自动生成 `operator+()`，我们只为满足某些限制的每个类做了这件事。但在其他情况下，任何对这些限制的合理描述都会产生过于宽泛的结果，我们必须逐个选择加入我们的类型。使用概念也可以这样做，但使用的技巧并不广为人知。你所需要做的就是定义一个其通用情况为假的（布尔变量就足够了）概念：

```cpp
template <typename T>
constexpr inline bool gen_plus = false; // General
template <typename T>
requires gen_plus<T>
T operator+(const T& lhs, const T& rhs) { … }
```

然后，对于需要选择加入的每种类型，我们专门化这个概念：

```cpp
class D { // No special base
  D& operator+=(const D& rhs) { … }
};
template <>
constexpr inline bool generate_plus<D> = true; // Opt-in
```

这两种方法都有一些优点：CRTP 使用一个基类，它可以比仅仅是一个操作符定义的包装更复杂；而概念可以在适当的时候结合显式选择加入和一些更一般的限制。然而，这些讨论忽略了一个更重要的区别：CRTP 可以用来通过成员和非成员函数扩展类接口，而概念只能用于非成员函数，包括非成员操作符。当概念和基于 CRTP 的解决方案都适用时，你应该选择最合适的一个（对于像 `operator+()` 这样的简单函数，概念可能更容易阅读）。此外，你不必等到 C++20 才能使用基于概念的限制：在 *第七章*，*SFINAE、概念和重载解析管理* 中展示的模拟概念的技巧在这里是绰绰有余的。

当然，我们可以使用与 CRTP 一起的概念，而不是试图替换 CRTP：如果 CRTP 的基类模板对我们想要从中派生的类型有一些要求，我们可以通过概念来强制执行这些要求。在这里使用概念的方式与我们发现的那一章中的方式没有不同。但我们将继续使用 CRTP 以及它还能做什么。

# CRTP 作为一种实现技术

正如我们之前指出的，CRTP 通常被用作一种纯实现模式；然而，即使在这个角色中，它也可以影响设计：一些设计选择是可取的但难以实现，如果出现一种好的实现技术，设计选择通常会改变。因此，让我们看看 CRTP 可以解决哪些问题。

## CRTP 用于代码复用

让我们从具体实现问题开始：我们有多达多个具有一些共同代码的类。通常，我们会为它们写一个基类。但共同代码并不真正通用：它为所有类做同样的事情，除了它使用的类型。我们需要的是一个通用基类模板。这把我们带到了 CRTP。

一个例子是对象注册表。出于调试目的，可能需要知道当前存在多少个特定类型的对象，甚至可能需要维护这样一个对象的列表。我们肯定不希望将注册机制应用于每个类，因此我们希望将其移动到基类。但是，现在我们遇到了一个问题——如果我们有两个派生类，`C` 和 `D`，它们都从同一个基类 `B` 继承，那么 `B` 的实例计数将是 `C` 和 `D` 的总和。问题不在于基类无法确定派生类的实际类型——如果愿意承担运行时多态的成本，它是可以确定的。问题在于基类只有一个计数器（或者类中编码的任何数量），而派生类的数量是无限的。我们可以实现一个非常复杂、昂贵且不可移植的解决方案，使用 `typeid` 来确定类名并维护一个名称和计数器的映射。但，我们真正需要的是每个派生类型一个计数器，而唯一的方法是在编译时让基类知道派生类类型。这又把我们带回了 CRTP：

```cpp
// Example 08
template <typename D> class registry {
  public:
  static size_t count;
  static D* head;
  D* prev;
  D* next;
  protected:
  registry() {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<D*>(this);
    if (next) next->prev = head;
  }
  registry(const registry&) {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<D*>(this);
    if (next) next->prev = head;
  }
  ~registry() {
    --count;
    if (prev) prev->next = next;
    if (next) next->prev = prev;
    if (head == this) head = next;
  }
};
template <typename D> size_t registry<D>::count(0);
template <typename D> D* registry<D>::head(nullptr);
```

我们将构造函数和析构函数声明为受保护的，因为我们不希望除了派生类之外创建任何注册对象。同样重要的是不要忘记复制构造函数，否则编译器会生成默认的复制构造函数，它不会增加计数器或更新列表（但析构函数会，所以计数器会变成负数并溢出）。对于每个派生类 `D`，基类是 `registry<D>`，这是一个独立的类型，具有自己的静态数据成员，`count` 和 `head`（后者是当前活动对象列表的头部）。任何需要维护活动对象运行时注册的类型现在只需要从 `registry` 继承：

```cpp
// Example 08
class C : public registry<C> {
  int i_;
  public:
  C(int i) : i_(i) {}
};
```

另一个类似的例子，其中基类需要知道派生类的类型并使用它来声明自己的成员，可以在*第九章*中找到，*命名参数、方法链和构建器模式*。接下来，我们将看到另一个 CRTP 的例子，这次实现的可用性为特定的设计选择打开了大门。

## 泛型接口的 CRTP

另一个经常需要将行为委托给派生类的情况是访问问题。在广义上，访问者是指被调用以处理一组数据对象并对每个对象依次执行函数的对象。通常，存在访问者层次结构，其中派生类定制或改变基类行为的一些方面。虽然访问者的最常见实现使用动态多态和虚函数调用，但静态访问者提供了我们之前看到的相同类型的性能优势。访问者通常不是通过多态调用的；你创建你想要的访问者并运行它。然而，基访问者类会调用在编译时可能被调度到派生类的成员函数，如果它们有正确的覆盖。考虑以下用于动物集合的通用访问者：

```cpp
// Example 09
struct Animal {
  public:
  enum Type { CAT, DOG, RAT };
  Animal(Type t, const char* n) : type(t), name(n) {}
  const Type type;
  const char* const name;
};
template <typename D> class GenericVisitor {
  public:
  template <typename it> void visit(it from, it to) {
    for (it i = from; i != to; ++i) {
      this->visit(*i);
    }
  }
  private:
  D& derived() { return *static_cast<D*>(this); }
  void visit(const Animal& animal) {
    switch (animal.type) {
      case Animal::CAT:
        derived().visit_cat(animal); break;
      case Animal::DOG:
        derived().visit_dog(animal); break;
      case Animal::RAT:
        derived().visit_rat(animal); break;
    }
  }
  void visit_cat(const Animal& animal) {
    cout << "Feed the cat " << animal.name << endl;
  }
  void visit_dog(const Animal& animal) {
    cout << "Wash the dog " << animal.name << endl;
  }
  void visit_rat(const Animal& animal) {
  cout << "Eeek!" << endl;
}
  friend D;
  GenericVisitor() = default;
};
```

注意，主要访问方法是一个模板成员函数（一个模板中的模板！），它接受任何可以遍历`Animal`对象序列的迭代器。此外，通过在类底部声明一个私有默认构造函数，我们保护自己免于在派生类中错误地指定其自己的继承类型。现在，我们可以开始创建一些访问者。默认访问者简单地接受通用访问者提供的默认操作：

```cpp
class DefaultVisitor :
  public GenericVisitor<DefaultVisitor> {
};
```

我们可以访问任何`Animal`对象的序列，例如，一个向量：

```cpp
std::vector<Animal> animals {
  {Animal::CAT, "Fluffy"},
  {Animal::DOG, "Fido"},
  {Animal::RAT, "Stinky"}};
DefaultVisitor().visit(animals.begin(), animals.end());
```

访问产生了预期的结果：

```cpp
Feed the cat Fluffy
Wash the dog Fido
Eeek!
```

但是，我们不必局限于默认操作——我们可以覆盖一个或多个动物类型的访问操作：

```cpp
class TrainerVisitor :
  public GenericVisitor<TrainerVisitor> {
  friend class GenericVisitor<TrainerVisitor>;
  void visit_dog(const Animal& animal) {
    cout << "Train the dog " << animal.name << endl;
  }
};
class FelineVisitor :
  public GenericVisitor<FelineVisitor> {
  friend class GenericVisitor<FelineVisitor>;
  void visit_cat(const Animal& animal) {
    cout << "Hiss at the cat " << animal.name << endl;
  }
  void visit_dog(const Animal& animal) {
    cout << "Growl at the dog " << animal.name << endl;
  }
  void visit_rat(const Animal& animal) {
    cout << "Eat the rat " << animal.name << endl;
  }
};
```

当一名狗训练师选择访问我们的动物时，我们使用`TrainerVisitor`：

```cpp
Feed the cat Fluffy
Train the dog Fido
Eeek!
```

最后，一只访问猫将有一套它自己的动作：

```cpp
Hiss at the cat Fluffy
Growl at the dog Fido
Eat the rat Stinky
```

我们将在*第十七章*，*访问者模式和多重分派*中学习更多关于不同类型访问者的知识。然而，现在我们将探索 CRTP 与另一个常见模式结合使用的情况。

# CRTP 和基于策略的设计

基于策略的设计是众所周知的策略模式的编译时变体；我们有一个专门章节介绍它，*第十五章*，恰当地命名为*基于策略的设计*。在这里，我们将专注于使用 CRTP 为派生类提供额外功能。具体来说，我们将泛化 CRTP 基类的使用，以扩展派生类的接口。

到目前为止，我们已使用一个基类来为派生类添加功能：

```cpp
template <typename D> struct plus_base {…};
class D : public plus_base<D> {…};
```

然而，如果我们想以多种方式扩展派生类的接口，单个基类会带来不必要的限制。首先，如果我们添加几个成员函数，基类可能会变得相当大。其次，我们可能希望接口设计有更模块化的方法。例如，我们可以有一个基类模板，它为任何派生类添加工厂构建方法：

```cpp
// Example 10
template <typename D> struct Factory {
  template <typename... Args>
  static D* create(Args&&... args) {
    return new D(std::forward<Args>(args)...);
  }
  static void destroy(D* d) { delete d; }
};
```

我们甚至可以有多个不同的工厂，它们提供相同的接口但以不同的方式分配内存。我们可以有一个另一个基类模板，它为任何具有流插入运算符的类添加字符串转换功能：

```cpp
// Example 10
template <typename D> struct Stringify {
  operator std::string() const {
    std::stringstream S;
    S << *static_cast<const D*>(this);
    return S.str();
  }
};
```

将这两个结合成一个单一的基类是没有意义的。在一个大型系统中，可能会有更多这样的类，每个类都为派生类添加特定的功能，并使用 CRTP 来实现它。但并非每个派生类都需要这些功能中的每一个。有了多个基类可供选择，很容易构建一个具有特定功能集的派生类：

```cpp
// Example 10
class C1 : public Stringify<C1>, public Factory<C1> {…};
```

这方法可行，但如果需要实现几个具有非常相似行为（除了 CRTP 基类提供的特性）的派生类，我们就有重复编写几乎完全相同的代码的风险。例如，如果我们还有一个工厂，它在线程局部内存中构建对象以加快并发程序的性能（让我们称它为`TLFactory`），我们可能不得不编写如下代码：

```cpp
class C2 : public Stringify<C2>, public TLFactory<C2> {…};
```

但`C1`和`C2`这两个类除了基类之外完全相同，然而，按照目前的写法，我们仍然需要实现和维护两份相同的代码。如果能编写一个单一代码模板，并根据需要将其不同的基类插入其中，那就更好了。这就是基于策略设计的主要思想；对此有几种不同的方法，你可以在*第十五章*，*基于策略的设计*中了解它们。现在，让我们专注于在模板中使用 CRTP 基类。由于我们现在需要一个可以接受多个基类类型的类模板，我们将需要使用变长模板。我们需要类似以下的内容：

```cpp
template <typename… Policies>
class C : public Policies… {};
```

基于策略的设计有使用这种确切模板的版本；但在这个例子中，如果我们尝试使用`Factory`或`Stringify`作为策略，它将无法编译。原因是它们不是类型（类），因此不能用作类型名称。它们是模板，因此我们必须将模板`C`的模板参数声明为模板本身（这被称为模板模板参数）。如果我们首先回忆一下如何声明单个模板模板参数，语法就更容易理解：

```cpp
template <template <typename> class B> class C;
```

如果我们想要从这个类模板`B`的特定实例继承，我们会写成：

```cpp
template <template <typename> class B>
class C : public B<template argument> {…};
```

在使用 CRTP 时，模板参数是派生类本身的类型，`C<B>`：

```cpp
template <template <typename> class B>
class C : public B<C<B>> {…};
```

将此推广到参数包是直接的：

```cpp
// Example 11
template <template <typename> class... Policies>
class C : public Policies<C<Policies...>>... {…};
```

模板参数是一个包（任何数量的模板而不是单个类）。派生类从整个包 `Policies…` 继承，除了 `Policies` 是模板，我们需要指定这些模板的实际实例化。包中的每个模板都在派生类上实例化，其类型为 `C<Policies…>`。

如果我们需要额外的模板参数，例如，为了在类 `C` 中启用使用不同的值类型，我们可以将它们与策略结合：

```cpp
// Example 11
template <typename T,
          template <typename> class... Policies>
class C : public Policies<C<T, Policies...>>... {
  T t_;
  public:
  explicit C(T t) : t_(t) {}
  const T& get() const { return t_; }
  friend std::ostream&
  operator<<(std::ostream& out, const C c) {
    out << c.t_;
    return out;
  }
};
```

要使用这个类与特定的策略集，定义一些别名是方便的：

```cpp
using X = C<int, Factory, Stringify>;
```

如果我们想使用具有相同 Policies 的几个类，我们也可以定义一个模板别名：

```cpp
template <typename T> using Y = C<T, Factory, Stringify>;
```

我们将在 *第十五章* 的 “基于策略的设计” 中了解更多关于策略的内容。我们将在那里和其他章节中遇到我们刚刚研究的技巧，CRPT ——它是一个灵活且强大的工具。

# 摘要

我们已经检查了一个相当复杂的设计模式，它结合了 C++ 的两个方面——泛型编程（模板）和面向对象编程（继承）。正如其名，Curiously Recurring Template Pattern 创建了一个循环，其中派生类从基类继承接口和实现，而基类通过模板参数访问派生类的接口。CRTP 有两种主要的使用模式——真正的静态多态，或 *静态接口*，其中对象主要作为基类型访问，以及扩展接口，或委托，其中直接访问派生类，但实现使用 CRTP 提供共同功能。后者可以从简单添加一到两个方法到从多个构建块或策略组合派生类接口的复杂任务。

下一章介绍了一个习语，它利用了我们刚刚学到的模式。这个习语也改变了我们按参数顺序传递函数参数的标准方式，并允许我们有顺序无关的命名参数。继续阅读以了解如何！

# 问题

1.  虚函数调用有多昂贵，为什么？

1.  为什么类似的函数调用，在编译时解析，没有这样的性能开销？

1.  你会如何实现编译时多态函数调用？

1.  你会如何使用 CRTP 来扩展基类的接口？

1.  在单个派生类中使用多个 CRTP 基类需要什么？
