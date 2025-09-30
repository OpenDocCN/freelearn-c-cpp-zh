

# 第六章：理解类型擦除

类型擦除通常被视为一种神秘、神秘的编程技术。它不仅限于 C++（大多数关于类型擦除的教程都使用 Java 作为示例）。本章的目标是揭开神秘的面纱，教您什么是类型擦除以及如何在 C++ 中使用它。

本章将涵盖以下主题：

+   什么是类型擦除？

+   类型擦除是设计模式，还是实现技术？

+   我们如何实现类型擦除？

+   在决定使用类型擦除时，必须考虑哪些设计和性能方面的因素？对于类型擦除的使用，还可以提供哪些其他指导方针？

# 技术要求

示例代码可以在以下链接找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter06`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter06)

您需要安装和配置 Google Benchmark 库，具体细节可以在此处找到：[`github.com/google/benchmark`](https://github.com/google/benchmark)（参见*第四章*，*从简单到微妙*）

# 什么是类型擦除？

**类型擦除**，一般而言，是一种编程技术，通过该技术从程序中移除显式的类型信息。这是一种抽象类型，确保程序不显式依赖于某些数据类型。

这个定义虽然完全正确，但也完美地服务于将类型擦除笼罩在神秘之中。它通过一种循环推理的方式做到这一点——在你面前悬挂着一种看似不可能的希望——用强类型语言编写的程序实际上不使用类型。这怎么可能？当然是通过抽象掉类型！因此，希望和神秘得以延续。

很难想象一个不明确提及类型的程序（至少是一个 C++ 程序；当然，肯定有语言在运行时所有类型都不是最终的）。

因此，我们首先通过一个示例来展示类型擦除的含义。这应该能让我们对类型擦除有一个直观的理解，在本书的后续章节中，我们将对其进行发展和完善。这里的目的是提高抽象级别——而不是编写一些特定类型的代码，可能为不同类型编写几个版本，我们可以编写一个更抽象的版本，表达概念——例如，而不是编写一个接口表达“对整数数组进行排序”的概念的函数，我们希望编写一个更抽象的函数，“排序”任何数组。

## 通过示例进行类型擦除

我们将详细解释什么是类型擦除以及如何在 C++ 中实现它。但首先，让我们看看一个从程序中移除了显式类型信息的程序是什么样的。

我们从一个使用唯一指针的简单例子开始，`std::unique_ptr`：

```cpp
std::unique_ptr<int> p(new int(0));
```

这是一个拥有指针（参见*第三章*，*内存和所有权*）——包含此指针的实体，例如对象或函数作用域，也控制我们分配的整数的生命周期，并负责其删除。删除在代码中不是显式的——当`p`指针被删除时（例如，当它超出作用域时）将发生删除。这种删除方式也不是显式的——默认情况下，`std::unique_ptr`使用`operator delete`删除它拥有的对象，或者更准确地说，通过调用`std::default_delete`，它反过来调用`operator delete`。如果我们不想使用常规的标准`delete`呢？例如，我们可能有在自定义堆上分配的对象：

```cpp
class MyHeap {
  public:
  ...
  void* allocate(size_t size);
  void deallocate(void* p);
  ...
};
void* operator new(size_t size, MyHeap* heap) {
  return heap->allocate(size);
}
```

分配没有问题，借助重载的`operator new`：

```cpp
MyHeap heap;
std::unique_ptr<int> p(new(&heap) int(0));
```

这个语法调用了双参数的`operator new`函数；第一个参数总是大小，由编译器添加，第二个参数是堆指针。由于我们声明了这样的重载，它将被调用，并返回从堆中分配的内存。但我们没有做任何改变对象删除方式的事情。常规的`operator delete`函数将被调用，并尝试将一些未从那里分配的内存返回给全局堆。结果很可能是内存损坏，并且可能崩溃。我们可以定义一个具有相同额外参数的`operator delete`函数，但这在这里对我们没有好处——与`operator new`不同，没有地方可以传递参数给`delete`（你经常会看到定义这样的`operator delete`函数，并且它应该这样行为，但它与程序中看到的任何`delete`都没有关系；它用于构造函数抛出异常时的栈回溯）。

某种程度上，我们需要告诉唯一指针，这个特定的对象要以不同的方式被删除。结果发现`std::unique_ptr`有一个第二个`template`参数。你通常看不到它，因为它默认为`std::default_delete`，但这是可以改变的，可以定义一个自定义的`deleter`对象来匹配分配机制。`deleter`有一个非常简单的接口——它需要是可调用的：

```cpp
template <typename T> struct MyDeleter {
  void operator()(T* p);
};
```

`std::default_delete`策略的实现基本上就是这样，它简单地调用`p`指针上的`delete`。我们的自定义`deleter`需要一个非平凡的构造函数来存储堆指针。请注意，虽然`deleter`通常需要能够删除任何可以分配的类型的对象，但它不必是一个模板类。一个具有模板成员函数的非模板类也可以做到这一点，只要类的数据成员不依赖于被删除的类型。在我们的情况下，数据成员只依赖于堆的类型，而不是被删除的内容：

```cpp
class MyDeleter {
  MyHeap* heap_;
  public:
  MyDeleter(MyHeap* heap) : heap_(heap) {}
  template <typename T> void operator()(T* p) {
    p->~T();
    heap_->deallocate(p);
  }
};
```

`deleter`必须执行标准`operator delete`函数的两个函数的等效操作——它必须调用被删除对象的析构函数，然后它必须释放为该对象分配的内存。

现在我们有了合适的`deleter`，我们终于可以使用我们自己的堆来使用`std::unique_ptr`：

```cpp
// Example 01
MyHeap heap;
MyDeleter deleter(&heap);
std::unique_ptr<int, MyDeleter> p(
  new(&heap) int(0), deleter);
```

注意，`deleter`对象通常在需要时创建，即在分配点：

```cpp
MyHeap heap;
std::unique_ptr<int, MyDeleter> p(
  new(&heap) int(0), MyDeleter(&heap));
```

无论哪种方式，`deleter`都必须是不可抛出复制的或不可抛出移动的；也就是说，它必须有一个复制构造函数或移动构造函数，并且构造函数必须声明为`noexcept`。内置类型，如原始指针，当然是可复制的，并且默认的编译器生成的构造函数不会抛出异常。任何将一个或多个这些类型作为数据成员的组合聚合类型，例如我们的`deleter`，都有一个默认构造函数，也不会抛出异常（除非它已经被重新定义，当然）。

注意，`deleter`是唯一指针类型的一部分。拥有相同类型对象但具有不同`deleter`的唯一指针是不同的类型：

```cpp
// Example 02
MyHeap heap;
std::unique_ptr<int, MyDeleter> p(
  new(&heap) int(0), MyDeleter(&heap));
std::unique_ptr<int> q(new int(0));
p = std::move(q);    // Error: p and q are different types
```

同样，唯一指针必须使用正确类型的`deleter`来构造：

```cpp
std::unique_ptr<int> p(new(&heap) int(0),
  MyDeleter(&heap));    // Does not compile
```

作为旁注，在实验不同类型的唯一指针时，你可能会注意到前面代码中的两个指针`p`和`q`，虽然不可赋值，但可以比较：`p == q`可以编译。这是因为比较运算符实际上是一个模板——它接受两种不同类型的唯一指针并比较其底层原始指针（如果该类型也不同，编译错误可能不会提到唯一指针，而是说一些关于比较没有转换的不同类型的指针的事情）。

现在，让我们用共享指针`std::shared_ptr`来做同样的例子。首先，我们将共享指针指向使用常规`operator new`函数构造的对象，如下所示：

```cpp
std::unique_ptr<int> p(new int(0));
std::shared_ptr<int> q(new int(0));
```

为了比较，我们也将唯一指针的声明留在了那里。这两个智能指针以完全相同的方式声明和构造。现在，在下面的代码块中，是分配在我们`heap`上的对象的共享指针：

```cpp
MyHeap heap;
std::unique_ptr<int, MyDeleter> p(
  new(&heap) int(0), MyDeleter(&heap));
std::shared_ptr<int> q(
  new(&heap) int(0), MyDeleter(&heap));
```

现在你可以看到区别了——使用自定义`deleter`创建的共享指针，尽管如此，其类型与使用默认`deleter`的指针相同！实际上，所有指向`int`的共享指针都具有相同的类型，`std::shared_ptr<int>`——模板没有另一个参数。仔细思考一下——`deleter`在构造函数中指定，但仅在析构函数中使用，因此它必须存储在智能指针对象中，直到需要时。如果我们失去了在构造过程中给出的对象，就没有办法恢复它。`std::shared_ptr`和`std::unique_ptr`都必须在指针对象内部存储任意类型的`deleter`对象。但只有`std::unique_ptr`类在其类型中包含删除器信息。`std::shared_ptr`类对所有删除器类型都是相同的。回到本节的开头，使用`std::shared_ptr<int>`的程序没有关于删除器类型的任何显式信息。

这个类型已经被从程序中擦除。这就是类型擦除程序的样子：

```cpp
// Example 03
void some_function(std::shared_ptr<int>);     // no deleter
MyHeap heap;
{
  std::shared_ptr<int> p(    // No deleter in the type
    new(&heap) int(0),
    MyDeleter(&heap));    // Deleter in constructor only
  std::shared_ptr<int> q(p);    // No deleter type anywhere
  some_function(p);    // uses p, no deleter
}    // Deletion happens, MyDeleter is invoked
```

我们花费了大量的时间来剖析`std::shared_ptr`，因为它提供了一个非常简单的类型擦除示例，尤其是当我们将其与必须解决相同问题但选择相反方法的`std::unique_ptr`进行对比时。然而，这个简单的例子并没有突出选择类型擦除的设计含义，也没有说明这个模式解决了哪些设计问题。为了了解这一点，我们应该看看 C++中典型的类型擦除对象：`std::function`。

## 从例子到一般化

在 C++中，`std::function`是一个通用的多态函数包装器，或者是一个通用的可调用对象。它用于存储任何可调用实体，如函数、lambda 表达式、仿函数（具有`operator()`的对象）或成员函数指针。这些不同可调用实体的唯一要求是它们必须具有相同的调用签名，即接受相同的参数并返回相同类型的结果。签名是在声明特定`std::function`对象时指定的：

```cpp
std::function<int(long, double)> f;
```

我们刚刚声明了一个可以接受两个参数（`long`和`double`，或者更准确地说，接受任何两个可以转换为`long`和`double`的参数）的可调用对象，并且返回的结果可以被转换为`int`。它对参数做了什么，结果是什么？这取决于分配给`f`的具体可调用实体：

```cpp
// Example 04
std::function<size_t(const std::string&)> f;
size_t f1(const std::string& s) { return s.capacity(); }
f = f1;
std::cout << f("abcde");    // 15
char c = 'b';
f = = { return s.find(c); };
std::cout << f("abcde");    // 1
f = &std::string::size;
std::cout << f("abcde");    // 5
```

在这个例子中，我们首先将一个非成员函数`f1`赋值给`f`；现在调用`f(s)`返回字符串`s`的容量，因为这就是`f1`所做的事情。接下来，我们将`f`改为包含一个 lambda 表达式；现在调用`f(s)`将调用该表达式。这两个函数唯一共同之处是接口：它们接受相同的参数并具有相同的返回类型。最后，我们将成员函数指针赋值给`f`；虽然`std::string::size()`函数不接受任何参数，但所有成员函数都有一个隐含的第一个参数，即对对象的引用，因此它符合接口的要求。

现在，我们可以看到类型擦除的更一般形式：它是对许多不同实现提供相同行为的抽象。让我们考虑它打开了哪些设计能力。

# 类型擦除作为设计模式

我们已经看到了类型擦除在程序中的表现：代码期望某些语义行为，但不是处理提供它的特定类型，而是使用一个抽象并“擦除”那些与当前任务无关的类型属性（从类型的名称开始）。

这样，类型擦除具有其他几个设计模式的属性，但它并不等同于任何一种。它可以合理地被认为是一种独立的设计模式。那么，类型擦除作为设计模式提供了什么？

在类型擦除中，我们发现了一种对某些行为（如函数调用）的抽象表达，可以用来将接口与实现分离。到目前为止，这听起来非常类似于继承。现在回想一下，在上一个部分的结尾，我们是如何让一个`std::function`对象调用几个完全不同的可调用对象的：一个函数、一个 lambda 表达式和一个成员函数。这说明了类型擦除与继承之间的基本区别：在继承中，基类决定了抽象行为（接口），任何需要实现该接口的类都必须从同一个基类派生。而在类型擦除中，没有这样的要求：提供共同行为的类型不需要形成任何特定的层次结构；实际上，它们甚至不需要是类。

可以说类型擦除提供了一种非侵入式的方法来分离接口和实现。当我们说“侵入式”时，指的是我们必须改变类型才能使用抽象：例如，我们可能有一个具有所需行为的类，但为了能够多态地使用，它还必须继承自公共基类。这就是“侵入” —— 我们必须对原本非常好的类进行强制更改，以便使其能够作为某个抽象接口的具体实现使用。正如我们刚才看到的，类型擦除没有这样的需求。只要类（或任何其他类型）具有所需的行为——通常，以函数调用类似的方式使用某些参数来调用它——就可以用来实现这种行为。类型的其他属性对我们关注的接口支持并不相关，并且被“擦除”。

我们也可以说类型擦除提供了“外部多态性”：不需要统一的层次结构，可以用来实现特定抽象的类型集是可扩展的，不仅限于从公共基类派生的类。

那么，为什么类型擦除不能完全取代 C++中的继承？在某种程度上，这是传统；不过，不要过于迅速地摒弃传统——传统的另一个名字是“惯例”，惯例代码也是熟悉且易于理解的代码。但还有两个“真实”的原因。第一个是性能。我们将在本章后面研究类型擦除的实现及其相应的性能；然而，不提前剧透，我们可以这样说，高性能的类型擦除实现最近才变得可用。第二个原因是便利性，我们已经在其中看到了这一点。如果我们需要为一系列相关操作声明一个抽象，我们可以声明一个具有必要虚拟成员函数的基类。如果我们使用`std::function`方法，类型擦除的实现将不得不分别处理这些操作中的每一个。正如我们很快就会看到的，这不是一个要求——我们可以一次性实现一组操作的类型擦除抽象。然而，使用继承来做这件事更容易。此外，记住，所有隐藏在类型擦除背后的具体类型都必须提供所需的行为；如果我们要求所有这些类型支持多个不同的成员函数，那么它们更有可能来自相同的层次结构，出于其他原因。

## 类型擦除作为一种实现技术

并非每个类型擦除的使用都背后有一个宏伟的设计理念。通常，类型擦除纯粹作为一种实现技术（继承也是如此，我们即将看到这样一个用法）。特别是，类型擦除是打破大型系统组件之间依赖关系的一个伟大工具。

这里有一个简单的例子。我们正在构建一个大型分布式软件系统，因此我们的核心组件之一是网络通信层：

```cpp
class Network {
  …
  void send(const char* data);
  void receive(const char* buffer);
  …
};
```

当然，这是一个非常简化和抽象的组件视图，这个组件至多是非平凡的，但我们现在不想关注通过网络发送数据。重要的是，这是我们的基础组件之一，系统的其余部分都依赖于它。我们可能有几个不同的程序作为我们的软件解决方案的一部分构建；它们都包含这个通信库。

现在，在某个具体的应用程序中，我们需要在网络发送数据包之前和之后处理这些数据包；这可能是一个需要高级加密的高安全性系统，或者它可能是我们系统中唯一设计用于在不可靠网络上工作并需要插入错误纠正代码的工具。重点是，网络层的开发者现在被要求引入对来自更高层应用程序特定组件的外部代码的依赖：

```cpp
class Network {
  …
  bool needs_processing;
  void send(const char* data) {
    if (needs_processing) apply_processing(buffer);
    …
  }
  …
};
```

虽然这段代码看起来很简单，但它却是一个依赖噩梦：低级库现在必须使用特定应用程序的`apply_processing()`函数来构建。更糟糕的是，所有不需要这个功能的其他程序仍然必须编译和链接这段代码，即使它们从未设置`needs_processing`。

虽然这个问题可以用“老式”的方法处理——使用一些函数指针或（更糟的是）全局变量，但类型擦除提供了一个优雅的解决方案：

```cpp
// Example 05
class Network {
  static const char* default_processor(const char* data) {
    std::cout << "Default processing" << std::endl;
    return data;
  }
  std::function<const char*(const char*)> processor =
    default_processor;
  void send(const char* data) {
    data = processor(data);
    …
  }
  public:
  template <typename F>
  void set_processor(F&& f) { processor = f; }
};
```

这是一个策略设计模式的例子，其中特定行为的实现可以在运行时选择。现在，系统的任何更高层组件都可以指定它自己的处理器函数（或 lambda 表达式，或可调用对象），而无需强迫软件的其他部分与其代码链接：

```cpp
Network N;
N.set_processor([](const char* s){ char* c; …; return c; };
```

现在我们已经知道了类型擦除的样子以及它如何作为设计模式和方便的实现技术帮助解耦组件，只剩下最后一个问题——它是如何工作的？

# 类型擦除在 C++中是如何实现的？

我们已经看到了 C++中类型擦除的样子，现在我们理解了程序不显式依赖于类型意味着什么。但谜团仍然存在——程序没有提及类型，然而，在正确的时间，它却调用了它一无所知的类型的操作。如何？这正是我们即将看到的。

## 非常古老的擦除类型

编写没有显式类型信息的程序的想法当然不是新的。实际上，它比面向对象编程和对象的概念要早得多。以这个 C 程序（这里没有 C++）为例：

```cpp
// Example 06
int less(const void* a, const int* b) {
  return *(const int*)a - *(const int*)b;
}
int main() {
  int a[10] = { 1, 10, 2, 9, 3, 8, 4, 7, 5, 0 };
  qsort(a, 10, sizeof(int), less);
}
```

现在记住标准`C`库中`qsort`函数的声明：

```cpp
void qsort(void *base, size_t nmemb, size_t size,
  int (*compare)(const void *, const void *));
```

注意，虽然我们使用它来对整数数组进行排序，但`qsort`函数本身没有任何显式类型——它使用`void*`来传递要排序的数组。同样，比较函数接受两个`void*`指针，其声明中没有显式类型信息。当然，在某个时候，我们需要知道如何比较实际类型。在我们的 C 程序中，理论上可以指向任何内容的指针被重新解释为指向整数的指针。这种反转抽象的行为被称为**具体化**。

在 C 语言中，恢复具体类型完全是程序员的职责——我们的`less()`比较函数实际上只比较整数，但从接口中无法推断出来。同样，在程序运行时验证整个程序中是否使用了正确的类型也是不可能的，程序自动选择运行时实际类型的正确比较操作当然也是不可能的。

尽管如此，这个简单的例子让我们揭开了类型擦除的神秘面纱：一般代码确实不依赖于被擦除的具体类型，但这种类型隐藏在通过类型擦除接口调用的函数的代码中。在我们的例子中，是比较函数：

```cpp
int less(const void* a, const int* b) {
  return *(const int*)a - *(const int*)b;
}
```

调用代码对类型`int`一无所知，但`less()`函数的实现操作的是这个类型。类型“隐藏”在通过类型无关接口调用的函数的代码中。

这种 C 语言方法的重大缺点是程序员必须完全负责确保所有类型擦除代码的各个部分保持一致；在我们的例子中，这是排序数据和比较函数必须引用相同的类型。

在 C++中，我们可以做得更好，但理念仍然是相同的：被擦除的类型通过通过类型无关接口调用的某些特定类型代码的实现而具体化。关键的区别是我们将强迫编译器为我们生成此代码。从根本上讲，有两种技术可以使用。第一种依赖于运行时多态（继承），第二种使用模板魔法。让我们从多态实现开始。

## 使用继承进行类型擦除

我们现在将看到`std::shared_ptr`是如何施展其魔法的。我们将用一个简化的智能指针示例来完成，这个示例专门关注类型擦除方面。了解到这是通过泛型和面向对象编程的组合来完成，你不会感到惊讶：

```cpp
// Example 07
template <typename T> class smartptr {
  struct destroy_base {
    virtual void operator()(void*) = 0;
    virtual ~deleter_base() {}
  };
  template <typename Deleter>
  struct destroy : public destroy _base {
    destroy (Deleter d) : d_(d) {}
    void operator()(void* p) override {
      d_(static_cast<T*>(p));
    }
    Deleter d_;
  };
  public:
  template <typename Deleter> smartptr(T* p, Deleter d) :
    p_(p), d_(new destroy<Deleter>(d)) {}
  ~smartptr() { (*d_)(p_); delete d_; }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
  private:
  T* p_;
  destroy _base* d_;
};
```

`smartptr`模板只有一个类型参数。由于擦除的类型不是智能指针类型的组成部分，它必须被捕获在其他某个对象中。在我们的例子中，这个对象是嵌套的`smartptr<T>::destroy`模板的一个实例化。这个对象是由构造函数创建的，这是代码中显式存在删除器类型的最后一个点。但是`smartptr`必须通过一个不依赖于`destroy`（因为智能指针对象对所有删除器都有相同的类型）的指针来引用`destroy`实例。因此，所有`destroy`模板的实例都继承自同一个基类，`destroy_base`，实际的删除器是通过一个虚拟函数调用的。构造函数是一个模板，它推导出删除器的类型，但这个类型只是隐藏的，因为它实际上是这个模板特定实例化的声明的一部分。智能指针类本身，特别是它的析构函数，实际上使用了删除器类型，进行了擦除。编译时类型检测用于创建一个在运行时重新发现删除器类型并执行正确操作的构造正确性的多态对象。因此，我们不需要动态类型转换，而可以使用静态类型转换，这只有在我们知道实际的派生类型时才有效（我们确实知道）。

同样的技术可以用来实现`std::function`和其他类型擦除类型，例如终极类型擦除类`std::any`（在 C++17 及以上版本）。这是一个类，而不是模板，但它可以持有任何类型的值：

```cpp
// Example 08
std::any a(5);
int i = std::any_cast<int>(a);    // i == 5
std::any_cast<long>(a);        // throws bad_any_cast
```

当然，如果不了解类型，`std::any`就无法提供任何接口。你可以将它存储任何值，如果你知道正确的类型（或者你可以请求类型并获取一个`std::type_info`对象）。

在我们学习其他（通常更有效）实现类型擦除的方法之前，我们必须解决我们设计中一个明显低效的问题：每次我们创建或删除一个共享指针或一个按上述方式实现的`std::function`对象时，我们必须为隐藏擦除类型的派生对象分配和释放内存。

## 无内存分配的类型擦除

然而，有方法可以优化类型擦除指针（以及任何其他类型擦除数据结构），并避免在构建多态的`smartptr::destroy`对象时发生的额外内存分配。我们可以通过预先为这些对象分配内存缓冲区来避免这种分配，至少在某些情况下可以这样做。这种优化的细节以及它的限制在*第十章*，*本地缓冲区优化*中进行了讨论。以下是优化的要点：

```cpp
// Example 07
template <typename T> class smartptr {
  …
  public:
  template <typename Deleter> smartptr(T* p, Deleter d) :
    p_(p) {
    static_assert(sizeof(Deleter) <= sizeof(buf_));
    ::new (static_cast<void*>(buf_)) destroy<Deleter>(d));
  }
  ~smartptr() {
    destroy_base* d = (destroy_base*)buf_;
    (*d)(p_);
    d->~destroy_base();
  }
  private:
  T* p_;
  alignas(8) char buf_[16];
};
```

本地缓冲区优化确实使类型擦除指针和函数变得更加高效，正如我们将在本章后面看到的那样。当然，它对删除器的尺寸施加了限制；因此，大多数实际应用使用本地缓冲区来存储足够小的擦除类型，而对于不适合缓冲区的类型则使用动态内存。上述替代方案——断言并强制程序员增加缓冲区——在非常高性能的应用中通常被采用。

使用这种优化的某些细微后果：删除器（或另一个擦除对象）现在作为类的一部分存储，并且必须与类的其余部分一起复制。我们如何复制一个我们不再知道其类型的对象？这个问题和其他细节将留待第*第十章**，本地缓冲区优化*中讨论。现在，我们将继续在其余的示例中使用本地缓冲区优化，以展示其用法并简化代码。

## 无继承的类型擦除

类型擦除的另一种实现不使用内部类层次结构来存储擦除类型。相反，类型被捕获在函数的实现中，就像在 C 中做的那样：

```cpp
void erased_func(void* p) {
  TE* q = static_cast<T*>(p);
  … do work on type TE …
}
```

在 C++中，我们将函数做成模板，以便编译器为每个我们需要的类型 `TE` 生成实例：

```cpp
template <typename TE> void erased_func(void* p) {
  TE* q = static_cast<T*>(p);
  … do work on type TE …
}
```

这是一个有些不寻常的模板函数：类型参数不能从参数中推断出来，必须显式指定。我们已经知道这将在类型擦除类的构造函数中完成，例如我们的智能指针：在那里，我们仍然知道即将被擦除的类型。另一个非常重要的点是，由前面的模板生成的任何函数都可以通过相同的函数指针调用：

```cpp
void(*)(void*) fp = erased_func<int>; // or any other type
```

现在我们可以看到类型擦除的魔法是如何工作的：我们有一个函数指针，其类型不依赖于我们正在擦除的类型 `TE`。我们将生成一个使用此类型的实现，并将其分配给此指针。当我们需要使用擦除类型 `TE` 时，例如使用指定的删除器删除对象，我们将通过这个指针调用一个函数；我们可以做到这一点而无需知道 `TE` 是什么。我们只需将这些全部组合成一个正确构建的实现，这就是我们的类型擦除智能指针：

```cpp
// Example 07
template <typename T>
class smartptr_te_static {
  T* p_;
  using destroy_t = void(*)(T*, void*);
  destroy_t destroy_;
  alignas(8) char buf_[8];
  template<typename Deleter>
  static void invoke_destroy(T* p, void* d) {
    (*static_cast<Deleter*>(d))(p);
  }
  public:
  template <typename Deleter>
  smartptr_te_static(T* p, Deleter d)
    : p_(p), destroy_(invoke_destroy<Deleter>)
  {
    static_assert(sizeof(Deleter) <= sizeof(buf_));
    ::new (static_cast<void*>(buf_)) Deleter(d);
  }
  ~smartptr_te_static() {
    this->destroy_(p_, buf_);
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
};
```

我们将用户提供的删除器存储在一个小的本地缓冲区中；在这个例子中，我们没有展示对于需要动态内存分配的较大删除器的替代实现。保留关于擦除类型信息的函数模板是 `invoke_destroy()`。请注意，它是一个静态函数；静态函数可以通过常规函数指针而不是更繁琐的成员函数指针来调用。

在`smartptr`类的构造函数中，我们实例化`invoke_destroy<Deleter>`并将其赋值给`destroy_`函数指针。我们还需要删除器对象的副本，因为删除器可能包含状态（例如，指向为智能指针拥有的对象提供内存的分配器的指针）。我们在局部缓冲区`buf_`提供的空间中构建这个删除器。此时，原始的`Deleter`类型已被擦除：我们拥有的只是一个不依赖于`Deleter`类型的函数指针和一个字符数组。

当需要销毁共享指针拥有的对象时，我们需要调用删除器。相反，我们通过`destroy_`指针调用函数，并将要销毁的对象以及删除器所在的缓冲区传递给它。擦除的`Deleter`类型无处可寻，但它隐藏在`invoke_destroy()`的具体实现中。在那里，缓冲区的指针被转换回实际存储在缓冲区中的类型（`Deleter`），然后调用删除器。

这个例子可能是 C++中类型擦除机制最简洁的演示。但它并不完全等同于前一个章节中我们使用继承的例子。当我们对智能指针拥有的类型为`T`的对象调用删除器时，我们并没有对删除器对象本身进行任何销毁操作，特别是我们存储在局部缓冲区内的副本。这里的局部缓冲区并不是问题：如果我们动态分配内存，它仍然会通过一个通用指针，如`char*`或`void*`来访问，而现在我们知道如何正确地删除它。为此，我们需要另一个可以具体化原始类型的函数。好吧，也许：平凡可销毁的删除器（以及在一般情况下，平凡可销毁的可调用对象）非常常见。所有函数指针、成员函数指针、无状态对象以及不通过值捕获任何非平凡对象的 lambda 表达式都是平凡可销毁的。因此，我们可以在构造函数中简单地添加一个静态断言，并将我们的智能指针限制为平凡可销毁的删除器，实际上，在大多数情况下它都会很好地为我们服务。但我也想向你展示一个更通用的解决方案。

我们当然可以使用另一个指向静态销毁删除器的指针，并在构造函数中以正确的类型实例化它。但析构函数并不是我们需要的结束：通常，我们还需要复制和移动删除器，甚至可能需要比较它们。这会导致我们的`smartptr`类变得臃肿。相比之下，基于继承的实现只需要一个指向`destroy`对象的指针（存储为基类`destroy_base`的指针）就完成了所有操作。有一种方法我们可以做到同样的事情。对于这个例子，没有好的方法可以逐步揭示魔法，所以我们不得不直接跳进去，并逐行进行解释：

```cpp
// Example 07
template <typename T>
class smartptr_te_vtable {
  T* p_;
  struct vtable_t {
    using destroy_t = void(*)(T*, void*);
    using destructor_t = void(*)(void*);
    destroy_t destroy_;
    destructor_t destructor_;
  };
  const vtable_t* vtable_ = nullptr;
  template <typename Deleter>
  constexpr static vtable_t vtable = {
    smartptr_te_vtable::template destroy<Deleter>,
    smartptr_te_vtable::template destructor<Deleter>
  };
  template <typename Deleter>
  static void destroy(T* p, void* d) {
    (*static_cast<Deleter*>(d))(p);
  }
  template <typename Deleter>
  static void destructor(void* d) {
    static_cast<Deleter*>(d)->~Deleter();
  }
  alignas(8) char buf_[8];
  public:
  template <typename Deleter>
  smartptr_te_vtable(T* p, Deleter d)
    : p_(p), vtable_(&vtable<Deleter>)
  {
    static_assert(sizeof(Deleter) <= sizeof(buf_));
    ::new (static_cast<void*>(buf_)) Deleter(d);
  }
  ~smartptr_te_vtable() {
    this->vtable_->destroy_(p_, buf_);
    this->vtable_->destructor_(buf_);
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
};
```

让我们解释一下这段代码是如何工作的。首先，我们声明一个 `struct vtable_t`，它包含指向我们需要在擦除的 `Deleter` 类型上实现的所有操作的函数指针。在我们的例子中，只有两个：对一个将被销毁的对象调用删除器，以及销毁删除器本身。一般来说，我们至少会在那里有复制和移动操作（你将在 *第十章**，局部缓冲区优化*）中找到这样的实现）。接下来，我们有 `vtable_` 指针。在对象构建之后，它将指向 `vtable_t` 类型的对象。虽然这可能会暗示接下来会有动态内存分配，但我们将做得更好。接下来是一个变量模板 `vtable`；在具体的 `Deleter` 类型上实例化它将创建一个 `vtable_t` 类型的静态变量实例。这可能是最棘手的部分：通常，当我们有一个类的静态数据成员时，它只是一个变量，我们可以通过名称访问它。但这里有一些不同：名称 `vtable` 可以引用许多对象，它们都是同一类型 `vtable_t`。我们都没有显式创建它们：我们不会为它们分配内存，也不会调用 `operator new` 来构建它们。编译器为每个我们使用的 `Deleter` 类型创建一个这样的对象。对于每个 `smartptr` 对象，我们想要它使用的特定 `vtable` 对象的地址存储在 `vtable_` 指针中。

类型为 `vtable_t` 的对象包含指向静态函数的指针。我们的 `vtable` 也必须这样做：正如你所见，我们在 `vtable` 中初始化了函数指针，使其指向 `smartptr` 类静态成员函数的实例化。这些实例化是为了与 `vtable` 本身实例化的相同 `Deleter` 类型。

`vtable` 这个名字不是随便选择的：我们确实实现了一个虚表；编译器为每个多态层次结构构建了一个非常类似的结构，其中包含函数指针，每个虚拟类都有一个虚拟指针，指向其原始类型（它被构建时的类型）的表。

在 `vtable` 之后，我们有两个静态函数模板。这就是擦除的类型真正隐藏的地方，稍后会被重新实现。正如我们之前所看到的，函数签名不依赖于 `Deleter` 类型，但它们的实现是。最后，我们有一个相同的缓冲区用于在本地存储删除器对象。

如前所述，构造函数将一切联系在一起；这是必须的，因为构造函数是此代码中唯一一个显式知道 `Deleter` 类型的位置。我们的构造函数做了三件事：首先，它存储了对象的指针，就像任何其他智能指针一样。其次，它将 `vtable_` 指针指向正确的 `Deleter` 类型的静态 `vtable` 变量的一个实例。最后，它在局部缓冲区中构造了删除器的副本。此时，`Deleter` 类型被擦除：`smartptr` 对象中的任何内容都没有显式依赖于它。

当调用智能指针的析构函数，需要销毁所拥有的对象和析构函数本身时，销毁者和它的真实类型再次发挥作用。这些操作都是通过间接函数调用来完成的。要调用的函数存储在 `*vtable_` 对象中（就像对于多态类，正确的虚拟函数重写可以在函数指针的虚表中找到）。销毁者通过缓冲区的地址传递给这些函数——那里没有类型信息。但是，这些函数是为特定的 `Deleter` 类型生成的，因此它们将 `void*` 缓冲区地址转换为正确的类型，并使用存储在缓冲区中的销毁者。

此实现允许我们在对象本身中只存储一个 `vtable_` 指针的同时执行多个类型擦除操作。

我们也可以结合两种方法：通过虚拟表调用一些操作，并为其他操作保留专用函数指针。为什么？可能是性能：通过虚拟表调用函数可能稍微慢一些。这需要针对任何特定应用程序进行测量。

到目前为止，我们使用类型擦除来提供针对非常特定行为的抽象接口。我们已经知道类型擦除不需要如此受限——我们已经看到了 `std::function` 的例子。本节最后的例子将是我们的通用类型擦除函数。

## 高效的类型擦除

在上一节的示例和解释之后，类型擦除函数不会构成太大的挑战。尽管如此，在这里展示它的价值仍然存在。我们将展示一个非常高效的类型擦除实现（本书中找到的实现受到了 Arthur O’Dwyer 和 Eduardo Magrid 的工作的启发）。

通用函数的模板如下所示：

```cpp
template<typename Signature> class Function;
```

这里 `Signature` 类似于 `int(std::string)`，这是一个接受字符串并返回整数的函数。只要可以像具有指定签名的函数一样调用此类型的对象，该函数就可以构造为调用任何 `Callable` 类型。

我们将再次使用局部缓冲区，但不是将其硬编码到类中，而是将模板参数添加到其中以控制缓冲区大小和对齐：

```cpp
template<typename Signature, size_t Size = 16,
         size_t Alignment = 8> struct Function;
```

为了便于编码，将函数签名拆分为参数类型 `Args...` 和返回类型 `Res` 是方便的。这样做最简单的方法是使用类模板特化：

```cpp
// Example 09
template<typename Signature, size_t Size = 16,
         size_t Alignment = 8> struct Function;
template<size_t Size, size_t Alignment,
         typename Res, typename... Args>
struct Function<Res(Args...), Size, Alignment> {…};
```

现在剩下的只是实现的小问题。首先，我们需要一个缓冲区来存储其中的 `Callable` 对象：

```cpp
// Example 09
alignas(Alignment) char space_[Size];
```

其次，我们需要一个函数指针 `executor_` 来存储从模板生成的静态函数 `executor` 的地址，该模板具有 `Callable` 对象的类型：

```cpp
// Example 09
using executor_t = Res(*)(Args..., void*);
executor_t executor_;
template<typename Callable>
static Res executor(Args... args, void* this_function) {
  return (*reinterpret_cast<Callable*>(
    static_cast<Function*>(this_function)->space_))
  (std::forward<Args>(args)...);
}
```

接下来，在构造函数中，我们必须初始化执行器并将可调用对象存储在缓冲区中：

```cpp
// Example 09
template <typename CallableArg,
          typename Callable = std::decay_t<CallableArg>>
  requires(!std::same_as<Function, Callable>)
Function(CallableArg&& callable) :
  executor_(executor<Callable>)
{
  ::new (static_cast<void*>(space_))
    Callable(std::forward<CallableArg>(callable));
}
```

构造函数有两个微妙之处。第一个是处理可调用对象类型的方式：我们推断其为`CallableArg`，但随后将其用作`Callable`。这是因为`CallableArg`可能是指向可调用对象类型的引用，例如函数指针，我们不希望构造一个引用的副本。第二个是概念限制：`Function`本身是一个具有相同签名的可调用对象，但我们不希望这个构造函数在这种情况下适用——那是复制构造函数的工作。如果你不使用 C++20，你必须使用 SFINAE 来实现相同的效果（有关详细信息，请参阅*第七章**，SFINAE、概念和重载解析管理*）。如果你喜欢概念风格，你可以在一定程度上模拟它：

```cpp
#define REQUIRES(...) \
  std::enable_if_t<__VA_ARGS__, int> = 0
template <typename CallableArg,
          typename Callable = std::decay_t<CallableArg>,
          REQUIRES(!std::is_same_v<Function, Callable>)>)
Function(CallableArg&& callable) …
```

谈到复制，我们的`Function`函数仅适用于可以轻易复制和轻易破坏的`Callable`类型，因为我们没有提供任何销毁或复制存储在缓冲区中的可调用对象的手段。这仍然覆盖了很多领域，但我们可以使用 vtable 方法处理非平凡的可调用对象（你可以在*第十章**，本地* *缓冲区优化*）中找到一个示例）。

现在我们还需要注意的一个细节是：`std::function`可以无任何可调用对象进行默认构造；调用这样的“空”函数会抛出`std::bad_function_call`异常。如果我们初始化执行器为一个预定义的什么也不做只是抛出这个异常的函数，我们也可以这样做：

```cpp
// Example 09
static constexpr Res default_executor(Args..., void*) {
  throw std::bad_function_call();
}
constexpr static executor_t default_executor_ =
  default_executor;
executor_t executor_ = default_executor_;
```

现在我们有一个与`std::function`非常相似（或者如果添加了对调用成员函数、复制和移动语义以及少数缺失的成员函数的支持，它将是这样）的泛型函数。它确实以相同的方式工作：

```cpp
Function<int(int, int, int, int)> f =
  [](int a, int b, int c, int d) { return a + b + c + d; };
int res = f(1, 2, 3, 4);
```

那这种便利性又让我们付出了什么代价呢？所有性能都应该被衡量，但通过检查编译器在必须调用类型擦除函数时生成的机器代码，我们也可以得到一些启示。以下是我们使用与我们刚刚使用的相同签名的`std::function`进行调用的过程：

```cpp
// Example 09
using Signature = int(int, int, int, int);
using SF = std::function<Signature>;
auto invoke_sf(int a, int b, int c, int d, const SF& f) {
  return f(a, b, c, d);
}
```

编译器（GCC-11 与 O3）将此代码转换为以下形式：

```cpp
endbr64
sub    $0x28,%rsp
mov    %r8,%rax
mov    %fs:0x28,%r8
mov    %r8,0x18(%rsp)
xor    %r8d,%r8d
cmpq   $0x0,0x10(%rax)
mov    %edi,0x8(%rsp)
mov    %esi,0xc(%rsp)
mov    %edx,0x10(%rsp)
mov    %ecx,0x14(%rsp)
je     62 <_Z9invoke_sfiiiiRKSt8functionIFiiiiiEE+0x62>
lea    0xc(%rsp),%rdx
lea    0x10(%rsp),%rcx
mov    %rax,%rdi
lea    0x8(%rsp),%rsi
lea    0x14(%rsp),%r8
callq  *0x18(%rax)
mov    0x18(%rsp),%rdx
sub    %fs:0x28,%rdx
jne    67 <_Z9invoke_sfiiiiRKSt8functionIFiiiiiEE+0x67>
add    $0x28,%rsp
retq
callq  67 <_Z9invoke_sfiiiiRKSt8functionIFiiiiiEE+0x67>
callq  6c <_Z9invoke_sfiiiiRKSt8functionIFiiiiiEE+0x6c>
```

现在我们的函数：

```cpp
// Example 09
using F = Function<Signature>;
auto invoke_f(int a, int b, int c, int d, const F& f) {
  return f(a, b, c, d);
}
```

这次，编译器可以做得更好：

```cpp
endbr64
jmpq   *0x10(%r8)
```

我们看到的是所谓的尾调用：编译器简单地将执行权转移到需要调用的实际可调用对象。你可能会问，难道不是总是这样吗？通常不是：大多数函数调用都是通过 `call` 和 `ret` 指令实现的。为了调用一个函数，其参数必须存储在预定义的位置，然后返回地址被压入栈中，通过 `call` 指令将执行权转移到函数入口点。返回指令 `ret` 从栈中取出地址并将执行权转移到它。尾调用的美妙之处在于：虽然我们希望 `Function` 调用最终调用原始可调用对象，但我们不需要执行权返回到 `Function` 对象。如果在 `Function` 执行器中没有其他事情要做，除了将控制权返回给调用者，我们完全可以简单地保留原始返回地址不变，让可调用对象将控制权返回到正确的位置，而不需要额外的间接引用。当然，这假设执行器在调用后没有其他事情要做。

我们代码中有两个关键的优化使得这种紧凑的实现成为可能。第一个是处理空函数的方式：大多数 `std::function` 的实现将执行器初始化为 `nullptr` 并在每次调用时进行指针比较。我们没有进行这样的比较；我们总是调用执行器。但是，我们的执行器永远不会是空的：除非其他初始化，它指向默认的执行器。

第二种优化更为微妙。你可能已经注意到执行器比可调用对象多一个参数：为了调用具有签名 `int(int, int)` 的函数，我们的执行器需要两个原始函数参数（当然）以及指向可调用对象的指针（在我们的例子中存储在局部缓冲区中）。因此，我们的执行器签名是 `int(int, int, void*)`。为什么不先传递对象呢？这正是 `std::function` 所做的（至少是我们刚刚看到的汇编版本）。问题是原始函数参数也位于栈上。在栈末尾添加一个参数很容易。但是，为了插入新的第一个参数，我们必须将所有现有参数移动一个位置（这就是为什么 `std::function` 生成的代码随着参数数量的增加而变长的原因）。

虽然听起来很有说服力，但关于性能的任何推测几乎不值得用电子来记录下来。性能必须始终测量，这是我们在这个章节中剩下的最后一个任务。

# 类型擦除的性能

我们将要测量类型擦除的泛型函数和类型擦除的智能指针删除器的性能。首先，我们需要正确的工具；在这种情况下，一个微基准测试库。

## 安装微基准测试库

在我们的案例中，我们感兴趣的是使用不同类型的智能指针构建和删除对象的非常小代码片段的效率。测量小代码片段性能的适当工具是微基准测试。现在有许多微基准测试库和工具；在这本书中，我们将使用 Google Benchmark 库。要跟随本节中的示例，你必须首先下载并安装库（为此，请遵循 `Readme.md` 文件中的说明）。然后你可以编译并运行示例。你可以构建库中包含的示例文件，以了解如何在你的特定系统上构建基准测试。例如，在 Linux 机器上，构建和运行 `smartptr.C` 基准测试程序的命令可能看起来像这样：

```cpp
$CXX smartptr.C smartptr_ext.C -o smartptr -g –O3 \
  -I. -I$GBENCH_DIR/include \
  -Wall -Wextra -Werror -pedantic --std=c++20 \
  $GBENCH_DIR/lib/libbenchmark.a -lpthread -lrt -lm && \
./smartptr
```

在这里，`$CXX` 是你的 C++ 编译器，例如 `clang++` 或 `g++-11`，而 `$GBENCH_DIR` 是基准测试安装的目录。

## 类型擦除的开销

每个基准测试都需要一个基线。在我们的案例中，基线是原始指针。我们可以合理地假设没有任何智能指针能够超越原始指针，并且最好的智能指针将没有开销。因此，我们首先测量使用原始指针构建和销毁一个小对象所需的时间：

```cpp
// Example 07
struct deleter {
  template <typename T> void operator()(T* p) { delete p; }
};
deleter d;
void BM_rawptr(benchmark::State& state) {
  for (auto _ : state) {
    int* p = new int(0);
    d(p);
  }
  state.SetItemsProcessed(state.iterations());
}
```

一个好的优化编译器可以通过优化“不必要的”工作（实际上，这是程序所做的所有工作）对像这样的微基准测试造成很大的破坏。我们可以通过将分配移动到不同的编译单元来防止此类优化：

```cpp
// 07_smartptr.C:
void BM_rawptr(benchmark::State& state) {
  for (auto _ : state) {
    int* p = get_raw_ptr()
    d(p);
  }
  state.SetItemsProcessed(state.iterations());
}
// 07_smartptr_ext.C:
int* get_raw_ptr() { return new int(0); }
```

如果你有一个可以进行整个程序优化的编译器，请为这个基准测试关闭它。但是不要关闭每个文件的优化：我们希望分析优化后的代码，因为这才是实际程序将使用的代码。

基准测试报告的实际数字当然取决于运行它的机器。但我们对相对变化感兴趣，所以任何机器都可以，只要我们在所有测量中都使用它：

```cpp
Benchmark                      Time
BM_rawptr                   8.72 ns
```

现在，我们可以验证 `std::unique_ptr` 确实没有开销（当然，只要我们以相同的方式构建和删除对象）：

```cpp
// smartptr.C
void BM_uniqueptr(benchmark::State& state) {
  for (auto _ : state) {
    auto p(get_unique_ptr());
  }
  state.SetItemsProcessed(state.iterations());
}
// smartptr_ext.C
auto get_unique_ptr() {
  return std::unique_ptr<int, deleter>(new int(0), d);
}
```

结果在原始指针的测量噪声范围内，如下所示：

```cpp
Benchmark                      Time
BM_uniqueptr                8.82 ns
```

我们可以类似地测量 `std::shared_ptr` 以及我们自己的智能指针的不同版本的性能：

```cpp
Benchmark                      Time
BM_sharedptr                22.9 ns
BM_make_sharedptr           17.5 ns
BM_smartptr_te              19.5 ns
```

第一行，`BM_sharedptr`，使用我们的自定义删除器构建和删除`std::shared_ptr<int>`。共享指针比唯一指针昂贵得多。当然，这不止一个原因——`std::shared_ptr`是一个引用计数智能指针，维护引用计数有其自身的开销。使用`std::make_shared`来分配共享指针使其创建和删除显著更快，正如我们在`BM_make_sharedptr`基准测试中所看到的，但为了确保我们只测量类型擦除的开销，我们应该实现一个类型擦除唯一指针。但我们已经做到了——这是我们在本章*如何在 C++中实现类型擦除*部分看到的`smartptr`。它具有刚好足够的功能来测量与其他所有指针相同的基准测试的性能：

```cpp
void BM_smartptr_te(benchmark::State& state) {
  for (auto _ : state) {
    auto get_smartptr_te();
  }
  state.SetItemsProcessed(state.iterations());
}
```

在这里，`smartptr_te`代表使用继承实现的智能指针的类型擦除版本。它比`std::shared_ptr`略快，这证实了我们的怀疑，即后者有多个开销来源。就像`std::shared_ptr`一样，删除`smartptr_te`会触及两个内存位置：在我们的案例中，它是被删除的对象和删除器（内嵌在多态对象中）。这正是`std::make_shared`通过合并`std::shared_ptr`的两个内存位置来避免的，这肯定是有益的。我们可以合理地假设第二个内存分配也是我们类型擦除智能指针性能不佳（大约是原始或唯一指针的两倍慢）的原因。如果我们使用智能指针对象内部预留的内部缓冲区，我们可以避免这种分配。我们已经在*类型擦除不涉及内存分配*部分看到了智能指针的本地缓冲区实现（在这个基准测试中，它被重命名为`smartptr_te_lb0`）。这里以`BM_smartptr_te_lb0`的名字进行了基准测试。当可能时使用本地缓冲区，但对于较大的删除器切换到动态分配的版本被命名为`smartptr_te_lb`，并且略慢（`BM_smartptr_te_lb`）：

```cpp
Benchmark                      Time
BM_smartptr_te_lb           11.3 ns
BM_smartptr_te_lb0          10.5 ns
BM_smartptr_te_static       9.58 ns
BM_smartptr_te_vtable       10.4 ns
```

我们还对两种不使用继承实现的类型擦除智能指针进行了基准测试。静态函数版本`BM_smartptr_te_static`比使用虚表的版本`BM_smartptr_te_vtable`略快。这两个版本都使用本地缓冲区；编译器生成的虚表与我们所精心制作的等效结构表现完全相同，这并不令人惊讶。

总体而言，即使是最好的类型擦除实现也存在一些开销，在我们的案例中不到 10%。是否可以接受这个开销，取决于应用程序。

我们还应该测量泛型类型擦除函数的性能。我们可以用任何可调用实体来测量其性能，例如，一个 lambda 表达式：

```cpp
// Example 09
void BM_fast_lambda(benchmark::State& state) {
  int a = rand(), b = rand(), c = rand(), d = rand();
  int x = rand();
  Function<int(int, int, int, int)> f {
    = {
      return x + a + b + c + d; }
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(f(a, b, c, d));
    benchmark::ClobberMemory();
  }
}
```

我们也可以对`std::function`进行相同的测量，并比较结果：

```cpp
Benchmark                      Time
BM_fast_lambda                 0.884 ns
BM_std_lambda                   1.33 ns
```

虽然这可能看起来是一个巨大的成功，但这个基准也隐藏了对过度使用类型擦除的警告。要揭示这个警告，我们只需测量对同一 lambda 的直接调用的性能：

```cpp
Benchmark                      Time
BM_lambda                      0.219 ns
```

我们如何将这种主要的减速与我们在比较智能指针和原始指针时看到的类型擦除的微小成本相协调？

重要的是要注意正在擦除的内容。一个实现良好的类型擦除接口可以提供与虚拟函数调用非常相似的性能。非内联的非虚拟函数调用将稍微快一点（在我们的例子中，耗时不到 9 纳秒的调用产生了大约 10%的开销）。但类型擦除的调用始终是间接的。它无法接近的一个竞争点是内联函数调用。这正是我们在比较类型擦除和直接调用 lambda 的性能时观察到的。

在我们了解了类型擦除的性能之后，我们何时可以推荐使用它？

# 使用类型擦除的指南

类型擦除解决了哪些问题，何时解决方案的成本是可以接受的？首先，重要的是不要忘记原始目标：类型擦除是一种设计模式，有助于关注点的分离，这是一种非常强大的设计技术。它用于在实现该行为可以由一组可能无关的类型提供时，为某种行为创建一个抽象。

它还用作实现技术，主要用来帮助打破编译单元和其他程序组件之间的依赖关系。

在我们能够回答“*类型擦除值得付出代价吗？*”这个问题之前，我们需要考虑替代方案。在许多情况下，替代方案是另一种实现相同抽象的方法：多态类层次结构或函数指针。这两种选项的性能与类型擦除（在其最佳实现中）相似，所以这取决于便利性和代码质量。对于单个函数，使用类型擦除函数比开发新的类层次结构更容易，比使用函数指针更灵活。对于具有许多成员函数的类，维护类层次结构通常更容易且更不容易出错。

另一个可能的替代方案是不采取任何行动，允许设计部分之间更紧密的耦合。这种决定的缺点通常与其性能收益成反比：系统紧密耦合的部分通常需要协调实现以达到良好的性能，但它们紧密耦合是有原因的。逻辑上分离良好的组件不应进行大量交互，因此，这种交互的性能不应是关键的。

当性能很重要但我们仍然需要抽象时，我们该怎么办？通常，这是类型擦除的直接对立面：我们将一切变成模板。

考虑 C++20 的范围。一方面，它们是抽象序列。我们可以编写一个操作范围的函数，并用向量、deque、从这些容器之一创建的范围、该范围的子范围或过滤视图来调用它。只要可以从`begin()`迭代到`end()`，任何东西都是范围。但是，从向量和一个 deque 创建的范围是不同的类型，尽管它们在接口上是序列的抽象。标准库提供了多个范围适配器和范围视图，它们都是模板。操作这些范围的函数也是模板。

我们能否实现一个类型擦除的范围？是的，这甚至并不难。我们最终得到一个单一的类型，`GenericRange`，可以从向量、deque、列表或其他具有`begin()`、`end()`和前向迭代器的任何东西中构建。我们还得到一些东西，其速度大约是大多数容器迭代器的一半，除了向量：它们的迭代器实际上只是指针，向量化的编译器可以进行优化，至少将代码速度提高一个数量级。当我们擦除原始容器的类型时，这种优化的可能性就丢失了。

C++设计者做出了决定，一方面，范围提供了一种对某些行为的抽象，并让我们将接口与实现分离。另一方面，他们不愿意牺牲性能。因此，他们选择将范围及其所有操作代码都做成模板。

作为软件系统的设计者，你可能不得不做出类似的决策。一般准则是在这种耦合对于性能至关重要的情况下，更倾向于紧密耦合相关组件。相反，对于交互不需要高效率的松耦合组件，更倾向于更好的分离。当处于这个领域时，类型擦除至少应该与多态和其他解耦技术同等考虑。

# 摘要

在本章中，我们希望已经解除了被称为类型擦除的编程技术的神秘感。我们展示了如何编写一个程序，其中并非所有的类型信息都是显式可见的，以及为什么这可能是一种理想的实现方式的原因。我们还展示了，当高效实现并明智使用时，它是一种强大的技术，可能导致更简单、更灵活的接口和明显分离的组件。

下一章将改变方向——我们已经处理了一些抽象惯用法一段时间了，现在转向 C++惯用法，这些惯用法有助于将模板组件绑定到复杂交互系统中。我们首先从 SFINAE 惯用法开始。

# 问题

1.  真正的类型擦除是什么？

1.  类型擦除在 C++中是如何实现的？

1.  在`auto`后面隐藏类型和擦除它之间有什么区别？

1.  当程序需要使用具体类型时，它是如何被具体化的？

1.  类型擦除的性能开销是什么？
