# 第六章：设计模式和 C++

C++不仅仅是一种面向对象的语言，它不仅仅提供动态多态性，因此在 C++中设计不仅仅是关于四人帮的模式。在本章中，你将学习关于常用的 C++习语和设计模式以及它们的使用场景。

本章将涵盖以下主题：

+   编写习惯用法的 C++

+   奇异递归模板模式

+   创建对象

+   跟踪状态和访问对象在 C++中

+   高效处理内存

这是一个相当长的列表！让我们不浪费时间，直接开始吧。

# 技术要求

本章的代码需要以下工具来构建和运行：

+   支持 C++20 的编译器

+   CMake 3.15+

本章的源代码片段可以在[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter06`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter06)找到。

# 编写习惯用法的 C++

如果你熟悉面向对象的编程语言，你一定听说过四人帮的设计模式。虽然它们可以在 C++中实现（而且经常被实现），但这种多范式语言通常采用不同的方法来实现相同的目标。如果你想要超越 Java 或 C#等所谓的基于咖啡的语言的性能，有时付出虚拟调度的代价太大了。在许多情况下，你会提前知道你将处理的类型。如果发生这种情况，你通常可以使用语言和标准库中提供的工具编写更高性能的代码。其中有一个我们将从本章开始的一组 - 语言习语。让我们通过查看其中一些来开始我们的旅程。

根据定义，习语是在特定语言中反复出现的构造，是特定于该语言的表达。C++的“母语者”应该凭直觉知道它的习语。我们已经提到智能指针，这是最常见的之一。现在让我们讨论一个类似的。

## 使用 RAII 保护自动化作用域退出操作

C++中最强大的表达之一是用于关闭作用域的大括号。这是析构函数被调用和 RAII 魔术发生的地方。为了驯服这个咒语，你不需要使用智能指针。你只需要一个 RAII 保护 - 一个对象，当构造时，将记住它在销毁时需要做什么。这样，无论作用域是正常退出还是由异常退出，工作都会自动发生。

最好的部分 - 你甚至不需要从头开始编写一个 RAII 保护。经过充分测试的实现已经存在于各种库中。如果你使用我们在上一章中提到的 GSL，你可以使用`gsl::finally()`。考虑以下例子：

```cpp
using namespace std::chrono;


void self_measuring_function() {

  auto timestamp_begin = high_resolution_clock::now();


  auto cleanup = gsl::finally([timestamp_begin] {

    auto timestamp_end = high_resolution_clock::now();

    std::cout << "Execution took: " << duration_cast<microseconds>(timestamp_end - timestamp_begin).count() << " us";

  });
```

```cpp
  // perform work

  // throw std::runtime_error{"Unexpected fault"};

}
```

在这里，我们在函数开始时取一个时间戳，然后在结束时再取一个。尝试运行这个例子，看看取消注释`throw`语句如何影响执行。在这两种情况下，我们的 RAII 保护将正确打印执行时间（假设异常在某处被捕获）。

现在让我们讨论一些更流行的 C++习语。

## 管理可复制性和可移动性

在 C++中设计新类型时，重要的是决定它是否可以复制和移动。更重要的是正确实现类的语义。现在让我们讨论这些问题。

### 实现不可复制类型

有些情况下，你不希望你的类被复制。非常昂贵的复制类是一个例子。另一个例子是由于切片而导致错误的类。过去，防止这些对象复制的常见方法是使用不可复制的习语：

```cpp
struct Noncopyable {

  Noncopyable() = default;

  Noncopyable(const Noncopyable&) = delete;

  Noncopyable& operator=(const Noncopyable&) = delete;

};


class MyType : NonCopyable {};
```

然而，请注意，这样的类也是不可移动的，尽管在阅读类定义时很容易忽略这一点。更好的方法是明确地添加两个缺失的成员（移动构造函数和移动赋值运算符）。作为一个经验法则，当声明这样的特殊成员函数时，总是声明所有这些函数。这意味着从 C++11 开始，首选的方法是编写以下内容：

```cpp
struct MyTypeV2 {

  MyTypeV2() = default;

  MyTypeV2(const MyTypeV2 &) = delete;

  MyTypeV2 & operator=(const MyTypeV2 &) = delete;

  MyTypeV2(MyTypeV2 &&) = delete;

  MyTypeV2 & operator=(MyTypeV2 &&) = delete;

};
```

这一次，成员是直接在目标类型中定义的，而没有辅助的`NonCopyable`类型。

### 遵循三和五法则

在讨论特殊成员函数时，还有一件事需要提到：如果您不删除它们并提供自己的实现，很可能需要定义所有这些函数，包括析构函数。在 C++98 中，这被称为三法则（由于需要定义三个函数：复制构造函数、复制赋值运算符和析构函数），自 C++11 的移动操作以来，它现在被五法则取代（另外两个是移动构造函数和移动赋值运算符）。应用这些规则可以帮助您避免资源管理问题。

### 遵循零法则

另一方面，如果您只使用所有特殊成员函数的默认实现，那么根本不要声明它们。这清楚地表明您想要默认行为。这也是最不令人困惑的。考虑以下类型：

```cpp
class PotentiallyMisleading {

public:

  PotentiallyMisleading() = default;

  PotentiallyMisleading(const PotentiallyMisleading &) = default;

  PotentiallyMisleading &operator=(const PotentiallyMisleading &) = default;

  PotentiallyMisleading(PotentiallyMisleading &&) = default;

  PotentiallyMisleading &operator=(PotentiallyMisleading &&) = default;

  ~PotentiallyMisleading() = default;


private:

  std::unique_ptr<int> int_;

};
```

尽管我们默认了所有成员，但这个类仍然是不可复制的。这是因为它有一个`unique_ptr`成员，它本身是不可复制的。幸运的是，Clang 会警告您，但 GCC 默认情况下不会。更好的方法是应用零规则，而不是写以下内容：

```cpp
class RuleOfZero {

  std::unique_ptr<int> int_;

};
```

现在我们有了更少的样板代码，并且通过查看成员，更容易注意到它不支持复制。

在讨论复制时，还有一个重要的习惯用法需要了解，您将在一分钟内了解到。在此之前，我们将涉及另一个习惯用法，可以（并且应该）用于实现第一个习惯用法。

## 使用隐藏友元

实质上，隐藏的友元是在声明它们为友元的类型的主体中定义的非成员函数。这使得这样的函数无法通过其他方式调用，而只能通过**参数相关查找**（**ADL**）来调用，有效地使它们隐藏起来。因为它们减少了编译器考虑的重载数量，它们也加快了编译速度。这样做的额外好处是，它们提供比其替代品更短的错误消息。它们的最后一个有趣的特性是，如果应该首先发生隐式转换，它们就不能被调用。这可以帮助您避免这种意外转换。

尽管在 C++中通常不建议使用友元，但对于隐藏的友元，情况看起来不同；如果前面段落中的优势不能说服您，您还应该知道，它们应该是实现定制点的首选方式。现在，您可能想知道这些定制点是什么。简而言之，它们是库代码使用的可调用对象，用户可以为其类型进行专门化。标准库为这些保留了相当多的名称，例如`begin`、`end`及其反向和`const`变体，`swap`、`(s)size`、`(c)data`和许多运算符，等等。如果您决定为任何这些定制点提供自己的实现，最好是符合标准库的期望。

好了，现在理论够了。让我们看看如何在实践中使用隐藏的友元来提供定制点专门化。例如，让我们创建一个过于简化的类来管理类型的数组：

```cpp
template <typename T> class Array {

public:

  Array(T *array, int size) : array_{array}, size_{size} {}


  ~Array() { delete[] array_; }


  T &operator[](int index) { return array_[index]; }

  int size() const { return size_; }

  friend void swap(Array &left, Array &right) noexcept {
    using std::swap;
    swap(left.array_, right.array_);
    swap(left.size_, right.size_);
  }


private:

  T *array_;

  int size_;

};
```

正如您所看到的，我们定义了一个析构函数，这意味着我们还应该提供其他特殊成员函数。我们将在下一节中使用我们隐藏的友元`swap`来实现它们。请注意，尽管在我们的`Array`类的主体中声明，但这个`swap`函数仍然是一个非成员函数。它接受两个`Array`实例，并且没有访问权限。

使用`std::swap`行使编译器首先在交换成员的命名空间中查找`swap`函数。如果找不到，它将退回到`std::swap`。这被称为“两步 ADL 和回退惯用语”，或简称为“两步”，因为我们首先使`std::swap`可见，然后调用`swap`。`noexcept`关键字告诉编译器我们的`swap`函数不会抛出异常，这允许它在某些情况下生成更快的代码。除了`swap`，出于同样的原因，始终使用这个关键字标记默认和移动构造函数。

既然我们有一个`swap`函数，让我们使用它来应用另一个惯用语到我们的`Array`类。

## 使用复制和交换惯用语提供异常安全性

正如我们在上一节中提到的，因为我们的`Array`类定义了一个析构函数，根据五法则，它还应该定义其他特殊成员函数。在本节中，您将了解一种惯用语，让我们可以在没有样板文件的情况下做到这一点，同时还额外提供强异常安全性。

如果您不熟悉异常安全级别，这里是您的函数和类型可以提供的级别的快速回顾：

+   **无保证**：这是最基本的级别。在对象在使用时抛出异常后，不对其状态做任何保证。

+   **基本异常安全性**：可能会有副作用，但您的对象不会泄漏任何资源，将处于有效状态，并且将包含有效数据（不一定与操作之前相同）。您的类型应该至少提供这个级别。

+   **强异常安全性**：不会发生任何副作用。对象的状态将与操作之前相同。

+   **无抛出保证**：操作将始终成功。如果在操作期间抛出异常，它将被内部捕获和处理，因此操作不会在外部抛出异常。此类操作可以标记为`noexcept`。

那么，我们如何一举两得地写出无样板文件的特殊成员，并提供强异常安全性呢？实际上很容易。由于我们有我们的`swap`函数，让我们使用它来实现赋值运算符：

```cpp
  Array &operator=(Array other) noexcept {

    swap(*this, other);

    return *this;

  }
```

在我们的情况下，一个运算符就足够了，既适用于复制赋值，也适用于移动赋值。在复制的情况下，我们通过值来获取参数，因此这是临时复制正在进行的地方。然后，我们所需要做的就是交换成员。我们不仅实现了强异常安全性，而且还能够在赋值运算符的主体中不抛出异常。然而，在函数被调用之前，当复制发生时，仍然可能抛出异常。在移动赋值的情况下，不会进行复制，因为通过值获取将只获取移动的对象。

现在，让我们定义复制构造函数：

```cpp
  Array(const Array &other) : array_{new T[other.size_]}, size_{other.size_} {

    std::copy_n(other.array_, size_, array_);

  }
```

这个函数可以根据`T`和分配内存而抛出异常。现在，让我们也定义移动构造函数：

```cpp
  Array(Array &&other) noexcept

      : array_{std::exchange(other.array_, nullptr)}, size_{std::exchange(other.size_, 0)} {}
```

在这里，我们使用`std::exchange`来初始化我们的成员，并在初始化列表上清理`other`的成员。构造函数声明为`noexcept`是出于性能原因。例如，如果`std::vector`只在移动构造时是`noexcept`可移动的，否则将进行复制。

就是这样。我们创建了一个提供强异常安全性的`array`类，而且几乎没有代码重复。

现在，让我们来解决另一个 C++惯用语，它可以在标准库的几个地方找到。

## 编写 niebloids

Niebloids，以 Eric Niebler 的名字命名，是 C++17 及以后标准使用的一种函数对象类型，用于定制点。随着标准范围的引入，它们的流行度开始增长，但它们最早是在 2014 年由 Niebler 提出的。它们的目的是在不需要时禁用 ADL，因此编译器不会考虑来自其他命名空间的重载。还记得前面章节中的*两步法*吗？由于它不方便且容易忘记，所以引入了*定制点对象*的概念。本质上，这些是为您执行*两步法*的函数对象。

如果您的库应该提供定制点，最好使用 niebloids 来实现它们。C++17 及以后引入的标准库中的所有定制点都是出于某种原因以这种方式实现的。即使您只需要创建一个函数对象，仍然要考虑使用 niebloids。它们提供了 ADL 的所有优点，同时减少了缺点。它们允许特化，并且与概念一起，它们为您提供了一种定制可调用函数重载集合的方法。它们还允许更好地定制算法，只是写的代码比通常多一点。

在这一部分，我们将创建一个简单的范围算法，我们将其实现为 niebloid。让我们称之为`contains`，因为它将简单地返回一个布尔值，表示范围中是否找到了给定的元素。首先，让我们创建函数对象本身，从其基于迭代器的调用操作符的声明开始：

```cpp
namespace detail {

struct contains_fn final {

  template <std::input_iterator It, std::sentinel_for<It> Sent, typename T,

            typename Proj = std::identity>

  requires std::indirect_binary_predicate<

      std::ranges::equal_to, std::projected<It, Proj>, const T *> constexpr bool

  operator()(It first, Sent last, const T &value, Proj projection = {}) const {
```

看起来冗长，但所有这些代码都有其目的。我们使我们的结构`final`以帮助编译器生成更高效的代码。如果您查看模板参数，您会看到迭代器和哨兵 - 每个标准范围的基本构建块。哨兵通常是一个迭代器，但它可以是任何可以与迭代器比较的半正则类型（半正则类型是可复制和默认可初始化的）。接下来，`T`是要搜索的元素类型，而`Proj`表示投影 - 在比较之前对每个范围元素应用的操作（`std::identity`的默认值只是将其输入作为输出传递）。 

在模板参数之后，有它们的要求；操作符要求我们可以比较投影值和搜索值是否相等。在这些约束之后，我们只需指定函数参数。

现在让我们看看它是如何实现的：

```cpp
    while (first != last && std::invoke(projection, *first) != value)

      ++first;

    return first != last;

  }
```

在这里，我们只是遍历元素，对每个元素调用投影并将其与搜索值进行比较。如果找到则返回`true`，否则返回`false`（当`first == last`时）。

即使我们没有使用标准范围，前面的函数也可以工作；我们还需要为范围重载。它的声明可以如下所示：

```cpp
  template <std::ranges::input_range Range, typename T,

            typename Proj = std::identity>

  requires std::indirect_binary_predicate<

      std::ranges::equal_to,

      std::projected<std::ranges::iterator_t<Range>, Proj>,

      const T *> constexpr bool

  operator()(Range &&range, const T &value, Proj projection = {}) const {
```

这一次，我们使用满足`input_range`概念的类型，元素值和投影类型作为模板参数。我们要求在调用投影后，范围的迭代器可以与类型为`T`的对象进行比较，与之前类似。最后，我们使用范围、值和投影作为我们重载的参数。

这个操作符的主体也会非常简单：

```cpp
    return (*this)(std::ranges::begin(range), std::ranges::end(range), value,

                   std::move(projection));

  }

};

}  // namespace detail
```

我们只需使用给定范围的迭代器和哨兵调用先前的重载，同时传递值和我们的投影不变。现在，对于最后一部分，我们需要提供一个`contains` niebloid，而不仅仅是`contains_fn`可调用：

```cpp
inline constexpr detail::contains_fn contains{};
```

通过声明一个名为`contains`的内联变量，类型为`contains_fn`，我们允许任何人使用变量名调用我们的 niebloid。现在，让我们自己调用它看看它是否有效：

```cpp
int main() {

  auto ints = std::ranges::views::iota(0) | std::ranges::views::take(5);


  return contains(ints, 42);

}
```

就是这样。我们的抑制 ADL 的函数符合预期工作。

如果你认为所有这些都有点啰嗦，那么你可能会对`tag_invoke`感兴趣，它可能会在将来的某个时候成为标准的一部分。请参考*进一步阅读*部分，了解有关这个主题的论文和 YouTube 视频，其中详细解释了 ADL、niebloids、隐藏的友元和`tag_invoke`。

现在让我们转向另一个有用的 C++习惯用法。

## 基于策略的设计模式

基于策略的设计最初是由 Andrei Alexandrescu 在他出色的*现代 C++设计*书中引入的。尽管该书于 2001 年出版，但其中许多想法今天仍在使用。我们建议阅读它；你可以在本章末尾的*进一步阅读*部分找到它的链接。策略习惯用法基本上是 Gang of Four 的策略模式的编译时等价物。如果您需要编写一个具有可定制行为的类，您可以将其作为模板与适当的策略作为模板参数。这在实际中的一个例子可能是标准分配器，作为最后一个模板参数传递给许多 C++容器作为策略。

让我们回到我们的`Array`类，并为调试打印添加一个策略：

```cpp
template <typename T, typename DebugPrintingPolicy = NullPrintingPolicy>

class Array {
```

如你所见，我们可以使用一个不会打印任何东西的默认策略。`NullPrintingPolicy`可以实现如下：

```cpp
struct NullPrintingPolicy {

  template <typename... Args> void operator()(Args...) {}

};
```

如你所见，无论给定什么参数，它都不会做任何事情。编译器会完全优化它，因此在不使用调试打印功能时不会产生任何开销。

如果我们希望我们的类更加冗长，我们可以使用不同的策略：

```cpp
struct CoutPrintingPolicy {

  void operator()(std::string_view text) { std::cout << text << std::endl; }

};
```

这次，我们只需将传递给策略的文本打印到`cout`。我们还需要修改我们的类来实际使用我们的策略：

```cpp
  Array(T *array, int size) : array_{array}, size_{size} {

    DebugPrintingPolicy{}("constructor");

  }


  Array(const Array &other) : array_{new T[other.size_]}, size_{other.size_} {

    DebugPrintingPolicy{}("copy constructor");

    std::copy_n(other.array_, size_, array_);

  }


  // ... other members ... 
```

我们只需调用策略的`operator()`，将要打印的文本传递进去。由于我们的策略是无状态的，我们可以在需要使用它时每次实例化它，而不会产生额外的成本。另一种选择也可以是直接从中调用静态函数。

现在，我们只需要用所需的策略实例化我们的`Array`类并使用它：

```cpp
Array<T, CoutPrintingPolicy>(new T[size], size);
```

使用编译时策略的一个缺点是使用不同策略的模板实例化是不同类型的。这意味着需要更多的工作，例如从常规的`Array`类分配到具有`CoutPrintingPolicy`的类。为此，您需要将策略作为模板参数实现赋值运算符作为模板函数。

有时，使用特征作为使用策略的替代方案。例如，`std::iterator_traits`可以用于在编写使用迭代器的算法时使用有关迭代器的各种信息。例如，`std::iterator_traits<T>::value_type`可以适用于定义了`value_type`成员的自定义迭代器，以及简单的迭代器，比如指针（在这种情况下，`value_type`将指向被指向的类型）。

关于基于策略的设计就说这么多。接下来我们要讨论的是一个可以应用于多种情景的强大习惯用法。

# 奇异递归模板模式

尽管它的名字中有*模式*一词，**奇异递归模板模式**（**CRTP**）是 C++中的一种习惯用法。它可以用于实现其他习惯用法和设计模式，并应用静态多态性，等等。让我们从最后一个开始，因为我们稍后会涵盖其他内容。

## 了解何时使用动态多态性与静态多态性

在提到多态性时，许多程序员会想到动态多态性，其中执行函数调用所需的信息在运行时收集。与此相反，静态多态性是关于在编译时确定调用的。前者的优势在于你可以在运行时修改类型列表，允许通过插件和库扩展你的类层次结构。后者的优势在于，如果你提前知道类型，它可以获得更好的性能。当然，在第一种情况下，你有时可以期望编译器去虚拟化你的调用，但你不能总是指望它这样做。然而，在第二种情况下，你可以获得更长的编译时间。

看起来你不能在所有情况下都赢。不过，为你的类型选择正确的多态类型可以走很远。如果性能受到影响，我们强烈建议你考虑静态多态性。CRTP 是一种可以用来应用它的习惯用法。

许多设计模式可以以一种或另一种方式实现。由于动态多态性的成本并不总是值得的，四人帮设计模式在 C++中通常不是最好的解决方案。如果你的类型层次结构应该在运行时扩展，或者编译时间对你来说比性能更重要（而且你不打算很快使用模块），那么四人帮模式的经典实现可能是一个很好的选择。否则，你可以尝试使用静态多态性来实现它们，或者通过应用更简单的面向 C++的解决方案，其中我们在本章中描述了一些。关键是选择最适合工作的工具。

## 实现静态多态性

现在让我们实现我们的静态多态类层次结构。我们需要一个基本模板类：

```cpp
template <typename ConcreteItem> class GlamorousItem {

public:

  void appear_in_full_glory() {

    static_cast<ConcreteItem *>(this)->appear_in_full_glory();

  }

};
```

基类的模板参数是派生类。这一开始可能看起来很奇怪，但它允许我们在我们的接口函数中`static_cast`到正确的类型，这种情况下，命名为`appear_in_full_glory`。然后我们在派生类中调用这个函数的实现。派生类可以这样实现：

```cpp
class PinkHeels : public GlamorousItem<PinkHeels> {

public:

  void appear_in_full_glory() {

    std::cout << "Pink high heels suddenly appeared in all their beauty\n";

  }

};


class GoldenWatch : public GlamorousItem<GoldenWatch> {

public:

  void appear_in_full_glory() {

    std::cout << "Everyone wanted to watch this watch\n";

  }

};
```

这些类中的每一个都使用自身作为模板参数从我们的`GlamorousItem`基类派生。每个也实现了所需的函数。

请注意，与动态多态性相反，CRTP 中的基类是一个模板，因此你将为你的派生类得到不同的基本类型。这意味着你不能轻松地创建一个`GlamorousItem`基类的容器。然而，你可以做一些事情：

+   将它们存储在一个元组中。

+   创建你的派生类的`std::variant`。

+   添加一个通用类来包装所有`Base`的实例化。你也可以为这个使用一个变体。

在第一种情况下，我们可以按照以下方式使用该类。首先，创建`base`实例的元组：

```cpp
template <typename... Args>

using PreciousItems = std::tuple<GlamorousItem<Args>...>;


auto glamorous_items = PreciousItems<PinkHeels, GoldenWatch>{};
```

我们的类型别名元组将能够存储任何迷人的物品。现在，我们需要做的就是调用有趣的函数：

```cpp
  std::apply(

      []<typename... T>(GlamorousItem<T>... items) {    
          (items.appear_in_full_glory(), ...); },

      glamorous_items);
```

因为我们试图迭代一个元组，最简单的方法是调用`std::apply`，它在给定元组的所有元素上调用给定的可调用对象。在我们的情况下，可调用对象是一个只接受`GlamorousItem`基类的 lambda。我们使用 C++17 引入的折叠表达式来确保我们的函数将被所有元素调用。

如果我们要使用变体而不是元组，我们需要使用`std::visit`，就像这样：

```cpp
  using GlamorousVariant = std::variant<PinkHeels, GoldenWatch>;

  auto glamorous_items = std::array{GlamorousVariant{PinkHeels{}}, GlamorousVariant{GoldenWatch{}}};

  for (auto& elem : glamorous_items) {

    std::visit([]<typename T>(GlamorousItem<T> item){ item.appear_in_full_glory(); }, elem);

  }
```

`std::visit`函数基本上接受变体并在其中存储的对象上调用传递的 lambda。在这里，我们创建了一个我们迷人变体的数组，所以我们可以像对待任何其他容器一样迭代它，用适当的 lambda 访问每个变体。

如果你觉得从接口用户的角度来写不直观，考虑下一种方法，将变体包装到另一个类中，我们这里称为`CommonGlamorousItem`：

```cpp
class CommonGlamorousItem {

public:

  template <typename T> requires std::is_base_of_v<GlamorousItem<T>, T>

  explicit CommonGlamorousItem(T &&item)

      : item_{std::forward<T>(item)} {}

private:

  GlamorousVariant item_;

};
```

为了构造我们的包装器，我们使用了一个转发构造函数（`templated T&&`是它的参数）。然后我们转发而不是移动来创建`item_`包装变体，因为这样我们只移动了右值输入。我们还约束了模板参数，因此一方面，我们只包装`GlamorousItem`基类，另一方面，我们的模板不会被用作移动或复制构造函数。

我们还需要包装我们的成员函数：

```cpp
  void appear_in_full_glory() {

    std::visit(

        []<typename T>(GlamorousItem<T> item) { 
            item.appear_in_full_glory(); },

        item_);

  }
```

这次，`std::visit`调用是一个实现细节。用户可以以以下方式使用这个包装器类：

```cpp
auto glamorous_items = std::array{CommonGlamorousItem{PinkHeels{}},

                                  CommonGlamorousItem{GoldenWatch{}}};

    for (auto& elem : glamorous_items) {

      elem.appear_in_full_glory();

    }
```

这种方法让类的使用者编写易于理解的代码，但仍然保持了静态多态性的性能。

为了提供类似的用户体验，尽管性能较差，您也可以使用一种称为类型擦除的技术，我们将在下面讨论。

## 插曲-使用类型擦除

尽管类型擦除与 CRTP 无关，但它与我们当前的示例非常契合，这就是为什么我们在这里展示它的原因。

类型擦除习惯是关于在多态接口下隐藏具体类型。这种方法的一个很好的例子可以在 Sean Parent 的演讲*Inheritance Is The Base Class of Evil*中找到，这是*GoingNative 2013*会议上的一个很好的例子。我们强烈建议您在空闲时间观看它；您可以在*进一步阅读*部分找到它的链接。在标准库中，您可以在`std::function`、`std::shared_ptr`的删除器或`std::any`等中找到它。

使用方便和灵活性是有代价的-这种习惯用法需要使用指针和虚拟分发，这使得标准库中提到的实用程序在性能导向的用例中使用起来不好。小心。

为了将类型擦除引入我们的示例中，我们不再需要 CRTP。这次，我们的`GlamorousItem`类将使用智能指针来包装动态多态对象。

```cpp
class GlamorousItem {

public:

  template <typename T>

  explicit GlamorousItem(T t)

      : item_{std::make_unique<TypeErasedItem<T>>(std::move(t))} {}


  void appear_in_full_glory() { item_->appear_in_full_glory_impl(); }


private:

  std::unique_ptr<TypeErasedItemBase> item_;

};  
```

这次，我们存储了一个指向基类（`TypeErasedItemBase`）的指针，它将指向我们项目的派生包装器（`TypeErasedItem<T>`）。基类可以定义如下：

```cpp
  struct TypeErasedItemBase {

    virtual ~TypeErasedItemBase() = default;

    virtual void appear_in_full_glory_impl() = 0;

  };
```

每个派生的包装器也需要实现这个接口：

```cpp
  template <typename T> class TypeErasedItem final : public TypeErasedItemBase {

  public:

    explicit TypeErasedItem(T t) : t_{std::move(t)} {}

    void appear_in_full_glory_impl() override { t_.appear_in_full_glory(); }


  private:

    T t_;

  };
```

通过调用包装对象的函数来实现基类的接口。请注意，这种习惯用法被称为“类型擦除”，因为`GlamorousItem`类不知道它实际包装的是什么`T`。当项目被构造时，`information`类型被擦除了，但这一切都能正常工作，因为`T`实现了所需的方法。

具体的项目可以以更简单的方式实现，如下所示：

```cpp
class PinkHeels {

public:

  void appear_in_full_glory() {

    std::cout << "Pink high heels suddenly appeared in all their beauty\n";

  }

};


class GoldenWatch {

public:

  void appear_in_full_glory() {

    std::cout << "Everyone wanted to watch this watch\n";

  }

};
```

这次，它们不需要继承任何基类。我们只需要鸭子类型-如果它像鸭子一样嘎嘎叫，那么它可能是一只鸭子。如果它可以以全荣耀出现，那么它可能是迷人的。

我们的类型擦除 API 可以如下使用：

```cpp
  auto glamorous_items =

      std::array{GlamorousItem{PinkHeels{}}, GlamorousItem{GoldenWatch{}}};

  for (auto &item : glamorous_items) {

    item.appear_in_full_glory();

  }
```

我们只需创建一个包装器数组，并对其进行迭代，所有这些都使用简单的基于值的语义。我们发现这是最愉快的使用方式，因为多态性对调用者来说是作为实现细节隐藏的。

然而，这种方法的一个很大的缺点是，正如我们之前提到的，性能较差。类型擦除是有代价的，因此应该谨慎使用，绝对不要在热路径中使用。

现在我们已经描述了如何包装和擦除类型，让我们转而讨论如何创建它们。

# 创建对象

在本节中，我们将讨论与对象创建相关的常见问题的解决方案。我们将讨论各种类型的对象工厂，通过构建者，并涉及组合和原型。然而，我们将采用与四人帮在描述他们的解决方案时略有不同的方法。他们提出了复杂的、动态多态的类层次结构作为他们模式的适当实现。在 C++世界中，许多模式可以应用于现实世界的问题，而不引入太多的类和动态分派的开销。这就是为什么在我们的情况下，实现将是不同的，在许多情况下更简单或更高效（尽管在四人帮的意义上更专业化和不那么“通用”）。让我们马上开始。

## 使用工厂

我们将在这里讨论的第一种创建模式是工厂。当对象的构造可以在单个步骤中完成时（如果不能在工厂之后立即完成的模式很有用），但构造函数本身并不够好时，它们是有用的。有三种类型的工厂-工厂方法、工厂函数和工厂类。让我们依次介绍它们。

### 使用工厂方法

工厂方法，也称为“命名构造函数惯用法”，基本上是调用私有构造函数的成员函数。我们什么时候使用它们？以下是一些情况：

+   **当有许多不同的方法来构造一个对象，这可能会导致错误**。例如，想象一下构造一个用于存储给定像素的不同颜色通道的类；每个通道由一个字节值表示。仅使用构造函数会使得很容易传递错误的通道顺序，或者值是为完全不同的调色板而设计的。此外，切换像素的颜色内部表示会变得非常棘手。你可以说我们应该有不同类型来表示这些不同格式的颜色，但通常，使用工厂方法也是一个有效的方法。

+   **当你想要强制对象在堆上或在另一个特定的内存区域中创建**。如果你的对象在堆栈上占用大量空间，而你担心会用尽堆栈内存，使用工厂方法是一个解决方案。如果你要求所有实例都在设备上的某个内存区域中创建，也是一样。

+   **当构造对象可能失败，但你不能抛出异常**。你应该使用异常而不是其他错误处理方法。当使用正确时，它们可以产生更清洁和性能更好的代码。然而，一些项目或环境要求禁用异常。在这种情况下，使用工厂方法将允许您报告在构造过程中发生的错误。

我们描述的第一种情况的工厂方法可能如下所示：

```cpp
class Pixel {

public:

  static Pixel fromRgba(char r, char b, char g, char a) {

    return Pixel{r, g, b, a};

  }

  static Pixel fromBgra(char b, char g, char r, char a) {

    return Pixel{r, g, b, a};

  }


  // other members


private:

  Pixel(char r, char g, char b, char a) : r_(r), g_(g), b_(b), a_(a) {}

  char r_, g_, b_, a_;

}
```

这个类有两个工厂方法（实际上，C++标准不承认术语“方法”，而是称它们为“成员函数”）：`fromRgba`和`fromBgra`。现在更难出错并以错误的顺序初始化通道。

请注意，拥有私有构造函数实际上会阻止任何类从您的类型继承，因为没有访问其构造函数，就无法创建实例。然而，如果这是您的目标而不是副作用，您应该更喜欢将您的类标记为最终。

### 使用工厂函数

与使用工厂成员函数相反，我们也可以使用非成员函数来实现它们。这样，我们可以提供更好的封装，正如 Scott Meyers 在他的文章中所描述的。

在我们的`Pixel`的情况下，我们也可以创建一个自由函数来制造它的实例。这样，我们的类型可以有更简单的代码：

```cpp
struct Pixel {

  char r, g, b, a;

};


Pixel makePixelFromRgba(char r, char b, char g, char a) {

  return Pixel{r, g, b, a};

}


Pixel makePixelFromBgra(char b, char g, char r, char a) {

  return Pixel{r, g, b, a};

}
```

使用这种方法使我们的设计符合第一章*，软件架构的重要性和优秀设计原则*中描述的开闭原则。可以很容易地添加更多的工厂函数来处理其他颜色调色板，而无需修改`Pixel`结构本身。

这种`Pixel`的实现允许用户手动初始化它，而不是使用我们提供的函数之一。如果我们希望，可以通过更改类声明来禁止这种行为。修复后的样子如下：

```cpp
struct Pixel {

  char r, g, b, a;


private:

  Pixel(char r, char g, char b, char a) : r(r), g(g), b(b), a(a) {}

  friend Pixel makePixelFromRgba(char r, char g, char b, char a);

  friend Pixel makePixelFromBgra(char b, char g, char r, char a);

};
```

这一次，我们的工厂函数是我们类的朋友。然而，类型不再是一个聚合，所以我们不能再使用聚合初始化（`Pixel{}`），包括指定的初始化器。此外，我们放弃了开闭原则。这两种方法提供了不同的权衡，所以要明智选择。

### 选择工厂的返回类型

在实现对象工厂时，您还应该选择它应该返回的实际类型。让我们讨论各种方法。

对于`Pixel`这种值类型而不是多态类型的情况，最简单的方法效果最好——我们只需返回值。如果您生成多态类型，请使用智能指针返回它（**永远**不要使用裸指针，因为这将在某个时候导致内存泄漏）。如果调用者应该拥有创建的对象，通常将其返回到基类的`unique_ptr`中是最好的方法。在不太常见的情况下，您的工厂和调用者都必须拥有对象时，使用`shared_ptr`或其他引用计数的替代方案。有时，工厂跟踪对象但不存储它就足够了。在这种情况下，在工厂内部存储`weak_ptr`，在外部返回`shared_ptr`。

一些 C++程序员会认为您应该使用输出参数返回特定类型，但在大多数情况下，这不是最佳方法。在性能方面，按值返回通常是最佳选择，因为编译器不会对对象进行额外的复制。如果问题是类型不可复制，从 C++17 开始，标准规定了复制省略是强制性的，因此通常按值返回这些类型不是问题。如果您的函数返回多个对象，请使用 pair、tuple、struct 或容器。

如果在构建过程中出现问题，您有几种选择：

+   如果不需要向调用者提供错误消息，则返回您的类型的`std::optional`。

+   如果在构建过程中出现错误很少且应该传播，则抛出异常。

+   如果在构建过程中出现错误很常见（请参阅 Abseil 文档中的模板），则返回您的类型的`absl::StatusOr`（请参阅*进一步阅读*部分）。

现在您知道应该返回什么了，让我们讨论我们最后一种工厂类型。

### 使用工厂类

工厂类是可以为我们制造对象的类型。它们可以帮助解耦多态对象类型与其调用者。它们可以允许使用对象池（其中可重用的对象被保留，这样您就不需要不断分配和释放它们）或其他分配方案。这些只是它们可以有用的一些例子。让我们更仔细地看看另一个例子。想象一下，您需要根据输入参数创建不同的多态类型。在某些情况下，像下面显示的多态工厂函数一样的多态工厂函数是不够的：

```cpp
std::unique_ptr<IDocument> open(std::string_view path) {

    if (path.ends_with(".pdf")) return std::make_unique<PdfDocument>();

    if (name == ".html") return std::make_unique<HtmlDocument>();


    return nullptr;

}
```

如果我们还想打开其他类型的文档，比如 OpenDocument 文本文件，可能会讽刺地发现前面的打开工厂不适用于扩展。如果我们拥有代码库，这可能不是一个大问题，但如果我们库的消费者需要注册自己的类型，这可能是一个问题。为了解决这个问题，让我们使用一个工厂类，允许注册函数来打开不同类型的文档，如下所示：

```cpp
class DocumentOpener {

public:

  using DocumentType = std::unique_ptr<IDocument>;

  using ConcreteOpener = DocumentType (*)(std::string_view);


private:

  std::unordered_map<std::string_view, ConcreteOpener> openerByExtension;

};
```

这个类目前还没有做太多事情，但它有一个从扩展到应该调用以打开给定类型文件的函数的映射。现在我们将添加两个公共成员函数。第一个将注册新的文件类型：

```cpp
  void Register(std::string_view extension, ConcreteOpener opener) {

    openerByExtension.emplace(extension, opener);

  }
```

现在我们有了填充映射的方法。第二个新的公共函数将使用适当的打开者打开文档：

```cpp
  DocumentType open(std::string_view path) {

    if (auto last_dot = path.find_last_of('.');

        last_dot != std::string_view::npos) {

      auto extension = path.substr(last_dot + 1);

      return openerByExtension.at(extension)(path);

    } else {

      throw std::invalid_argument{"Trying to open a file with no extension"};

    }

  }
```

基本上，我们从文件路径中提取扩展名，如果为空则抛出异常，如果不为空，则在我们的映射中寻找打开者。如果找到，我们使用它来打开给定的文件，如果没有，映射将为我们抛出另一个异常。

现在我们可以实例化我们的工厂并注册自定义文件类型，比如 OpenDocument 文本格式：

```cpp
auto document_opener = DocumentOpener{};


document_opener.Register(

    "odt", [](auto path) -> DocumentOpener::DocumentType {

      return std::make_unique<OdtDocument>(path);

    });
```

请注意，我们注册了一个 lambda，因为它可以转换为我们的`ConcreteOpener`类型，这是一个函数指针。但是，如果我们的 lambda 有状态，情况就不同了。在这种情况下，我们需要使用一些东西来包装我们。这样的东西可能是`std::function`，但这样做的缺点是每次运行函数时都需要付出类型擦除的代价。在打开文件的情况下，这可能没问题。但是，如果你需要更好的性能，考虑使用`function_ref`这样的类型。

提议的这个实用程序的示例实现（尚未被接受）可以在 Sy Brand 的 GitHub 存储库中找到，该存储库在*进一步阅读*部分中有引用。

好了，现在我们在工厂中注册了我们的打开者，让我们使用它来打开一个文件并提取一些文本出来：

```cpp
  auto document = document_opener.open("file.odt");

  std::cout << document->extract_text().front();
```

就是这样！如果你想为你的库的消费者提供一种注册他们自己类型的方式，他们必须在运行时访问你的映射。你可以提供一个 API 让他们访问它，或者将工厂设为静态，并允许他们从代码的任何地方注册。

这就是工厂和在单一步骤中构建对象的全部内容。让我们讨论另一个流行的模式，如果工厂不合适的话可以使用。

## 使用构建者

构建者类似于工厂，是来自四人帮的一种创建模式。与工厂不同，它们可以帮助你构建更复杂的对象：那些无法在单一步骤中构建的对象，例如由许多单独部分组装而成的类型。它们还为你提供了一种自定义对象构建的方式。在我们的例子中，我们将跳过设计复杂的构建者层次结构。相反，我们将展示构建者如何帮助。我们将把实现层次结构的工作留给你作为练习。

当一个对象无法在单一步骤中产生时，就需要构建者，但如果单一步骤不是微不足道的话，具有流畅接口只会让它们更加愉快。让我们使用 CRTP 来演示创建流畅的构建者层次结构。

在我们的情况下，我们将创建一个 CRTP，`GenericItemBuilder`，它将作为我们的基本构建者，以及`FetchingItemBuilder`，它将是一个更专业的构建者，可以使用远程地址获取数据（如果支持的话）。这样的专业化甚至可以存在于不同的库中，例如，使用可能在构建时可用或不可用的不同 API。

为了演示目的，我们将从第五章*，利用 C++语言特性*构建我们的`Item`结构的实例：

```cpp
struct Item {

  std::string name;

  std::optional<std::string> photo_url;

  std::string description;

  std::optional<float> price;

  time_point<system_clock> date_added{};

  bool featured{};

};
```

如果你愿意，你可以通过将默认构造函数设为私有并使构建者成为友元来强制使用构建者构建`Item`实例。

```cpp
  template <typename ConcreteBuilder> friend class GenericItemBuilder;
```

我们的构建者实现可以从以下开始：

```cpp
template <typename ConcreteBuilder> class GenericItemBuilder {

public:

  explicit GenericItemBuilder(std::string name)

      : item_{.name = std::move(name)} {}

protected:

  Item item_;
```

尽管通常不建议创建受保护的成员，但我们希望我们的后代构建者能够访问我们的项目。另一种方法是在派生类中只使用基本构建器的公共方法。

我们在构建器的构造函数中接受名称，因为它是来自用户的单个输入，在创建项目时需要设置。这样，我们确保它将被设置。另一种选择是在建造的最后阶段检查它是否可以，当对象被释放给用户时。在我们的情况下，构建步骤可以实现如下：

```cpp
  Item build() && {

    item_.date_added = system_clock::now();

    return std::move(item_);

  }
```

我们强制要求在调用此方法时“消耗”构建器；它必须是一个 r 值。这意味着我们可以在一行中使用构建器，或者在最后一步将其移动以标记其工作结束。然后我们设置我们的项目的创建时间并将其移出构建器。

我们的构建器 API 可以提供以下功能：

```cpp
  ConcreteBuilder &&with_description(std::string description) {

    item_.description = std::move(description);

    return static_cast<ConcreteBuilder &&>(*this);

  }


  ConcreteBuilder &&marked_as_featured() {

    item_.featured = true;

    return static_cast<ConcreteBuilder &&>(*this);

  }
```

它们中的每一个都将具体（派生）构建器对象作为 r 值引用返回。也许出乎意料的是，这次应该优先返回此返回类型，而不是按值返回。这是为了避免在构建时不必要地复制`item_`。另一方面，通过 l 值引用返回可能导致悬空引用，并且会使调用`build()`变得更加困难，因为返回的 l 值引用将无法匹配预期的 r 值引用。

最终的构建器类型可能如下所示：

```cpp
class ItemBuilder final : public GenericItemBuilder<ItemBuilder> {

  using GenericItemBuilder<ItemBuilder>::GenericItemBuilder;

};
```

它只是一个重用我们通用构建器的构造函数的类。可以如下使用：

```cpp
  auto directly_loaded_item = ItemBuilder{"Pot"}

                                  .with_description("A decent one")

                                  .with_price(100)

                                  .build();
```

正如您所看到的，最终的接口可以使用函数链接调用，并且方法名称使整个调用流畅易读，因此称为*流畅接口*。

如果我们不直接加载每个项目，而是使用一个更专门的构建器，可以从远程端点加载数据的部分，会怎么样？我们可以定义如下：

```cpp
class FetchingItemBuilder final

    : public GenericItemBuilder<FetchingItemBuilder> {

public:

  explicit FetchingItemBuilder(std::string name)

      : GenericItemBuilder(std::move(name)) {}


  FetchingItemBuilder&& using_data_from(std::string_view url) && {

    item_ = fetch_item(url);

    return std::move(*this);

  }

};
```

我们还使用 CRTP 从我们的通用构建器继承，并强制要求给我们一个名称。然而，这一次，我们用我们自己的函数扩展基本构建器，以获取内容并将其放入我们正在构建的项目中。由于 CRTP，当我们从基本构建器调用函数时，我们将得到派生的返回，这使得接口更容易使用。可以以以下方式调用：

```cpp
  auto fetched_item =

      FetchingItemBuilder{"Linen blouse"}

          .using_data_from("https://example.com/items/linen_blouse")

          .marked_as_featured()

          .build();
```

一切都很好！

如果您需要始终创建不可变对象，构建器也很有用。由于构建器可以访问类的私有成员，它可以修改它们，即使类没有为它们提供任何设置器。当然，这并不是您可以从使用它们中受益的唯一情况。

### 使用复合和原型构建

您需要使用构建器的情况是创建复合体。复合体是一种设计模式，其中一组对象被视为一个对象，所有对象共享相同的接口（或相同的基本类型）。一个例子是图形，您可以将其组合成子图形，或者文档，它可以嵌套其他文档。当您在这样的对象上调用`print()`时，所有子对象都会按顺序调用它们的`print()`函数以打印整个复合体。构建器模式对于创建每个子对象并将它们全部组合在一起非常有用。

原型是另一种可以用于对象构建的模式。如果您的类型创建成本很高，或者您只想要一个基本对象来构建，您可能想要使用这种模式。它归结为提供一种克隆对象的方法，您稍后可以单独使用它，或者修改它以使其成为应该成为的样子。在多态层次结构的情况下，只需添加`clone()`，如下所示：

```cpp
class Map {

public:

    virtual std::unique_ptr<Map> clone() const;

    // ... other members ...

};


class MapWithPointsOfInterests {

public:

    std::unique_ptr<Map> clone() override const;

    // ... other members ...

private:

    std::vector<PointOfInterest> pois_;

};
```

我们的`MapWithPointsOfInterests`对象也可以克隆这些点，这样我们就不需要手动重新添加每一个。这样，当用户创建自己的地图时，我们可以为其提供一些默认值。还要注意，在某些情况下，简单的复制构造函数就足够了，而不是使用原型。

我们现在已经涵盖了对象创建。我们沿途提到了变体，那么为什么不重新访问它们（双关语）以看看它们如何帮助我们？

# 在 C++中跟踪状态和访问对象

状态是一种设计模式，旨在在对象的内部状态发生变化时帮助改变对象的行为。不同状态的行为应该彼此独立，以便添加新状态不会影响当前状态。在状态对象中实现所有行为的简单方法不具有可扩展性。使用状态模式，可以通过引入新的状态类并定义它们之间的转换来添加新行为。在本节中，我们将展示一种使用`std::variant`和静态多态双重分派来实现状态和状态机的方法。

首先，让我们定义我们的状态。在我们的示例中，让我们模拟商店中产品的状态。它们可以如下所示：

```cpp
namespace state {


struct Depleted {};


struct Available {

  int count;

};


struct Discontinued {};

} // namespace state
```

我们的状态可以有自己的属性，比如剩余物品的数量。与动态多态性相反，它们不需要从一个共同的基类继承。相反，它们都存储在一个变体中，如下所示：

```cpp
using State = std::variant<state::Depleted, state::Available, state::Discontinued>;
```

除了状态，我们还需要用于状态转换的事件。检查以下代码：

```cpp
namespace event {


struct DeliveryArrived {

  int count;

};


struct Purchased {

  int count;

};


struct Discontinued {};


} // namespace event
```

如您所见，我们的事件也可以有属性，并且不需要从一个共同的基类继承。现在，我们需要实现状态之间的转换。可以按以下方式完成：

```cpp
State on_event(state::Available available, event::DeliveryArrived delivered) {

  available.count += delivered.count;

  return available;

}


State on_event(state::Available available, event::Purchased purchased) {

  available.count -= purchased.count;

  if (available.count > 0)

    return available;

  return state::Depleted{};

}
```

如果进行购买，状态可以改变，但也可以保持不变。我们还可以使用模板一次处理多个状态：

```cpp
template <typename S> State on_event(S, event::Discontinued) {

  return state::Discontinued{};

}
```

如果商品停产，无论它处于什么状态都无所谓。好的，现在让我们实现最后一个受支持的转换：

```cpp
State on_event(state::Depleted depleted, event::DeliveryArrived delivered) {

  return state::Available{delivered.count};

}
```

我们需要的下一个拼图是一种定义多个调用运算符的方式，以便可以调用最佳匹配的重载。我们稍后将使用它来调用我们刚刚定义的转换。我们的辅助程序可以如下所示：

```cpp
template<class... Ts> struct overload : Ts... { using Ts::operator()...; };

template<class... Ts> overload(Ts...) -> overload<Ts...>;
```

我们创建了一个`overload`结构，它将在构造期间提供所有传递给它的调用运算符，使用可变模板、折叠表达式和类模板参数推导指南。有关此的更深入解释，以及实现访问的另一种替代方式，请参阅 Bartłomiej Filipek 在*进一步阅读*部分中的博客文章。

现在我们可以开始实现状态机本身：

```cpp
class ItemStateMachine {

public:

  template <typename Event> void process_event(Event &&event) {

    state_ = std::visit(overload{

        & requires std::is_same_v<

            decltype(on_event(state, std::forward<Event>(event))), State> {

          return on_event(state, std::forward<Event>(event));

        },

        [](const auto &unsupported_state) -> State {

          throw std::logic_error{"Unsupported state transition"};

        }

      },

      state_);

  }


private:

  State state_;

};
```

我们的`process_event`函数将接受我们定义的任何事件。它将使用当前状态和传递的事件调用适当的`on_event`函数，并切换到新状态。如果找到给定状态和事件的`on_event`重载，将调用第一个 lambda。否则，约束条件将不满足，并将调用第二个更通用的重载。这意味着如果存在不受支持的状态转换，我们将抛出异常。

现在，让我们提供一种报告当前状态的方法：

```cpp
      std::string report_current_state() {

        return std::visit(

            overload{[](const state::Available &state) -> std::string {

                       return std::to_string(state.count) + 
                       " items available";

                     },

                     [](const state::Depleted) -> std::string {

                       return "Item is temporarily out of stock";

                     },

                     [](const state::Discontinued) -> std::string {

                       return "Item has been discontinued";

                     }},

            state_);

      }
```

在这里，我们使用我们的重载来传递三个 lambda，每个 lambda 返回一个通过访问我们的状态对象生成的报告字符串。

现在我们可以调用我们的解决方案：

```cpp
        auto fsm = ItemStateMachine{};

        std::cout << fsm.report_current_state() << '\n';

        fsm.process_event(event::DeliveryArrived{3});

        std::cout << fsm.report_current_state() << '\n';

        fsm.process_event(event::Purchased{2});

        std::cout << fsm.report_current_state() << '\n';

        fsm.process_event(event::DeliveryArrived{2});

        std::cout << fsm.report_current_state() << '\n';

        fsm.process_event(event::Purchased{3});

        std::cout << fsm.report_current_state() << '\n';

        fsm.process_event(event::Discontinued{});

        std::cout << fsm.report_current_state() << '\n';

        // fsm.process_event(event::DeliveryArrived{1});
```

运行后，将产生以下输出：

```cpp
Item is temporarily out of stock

3 items available

1 items available

3 items available

Item is temporarily out of stock

Item has been discontinued
```

也就是说，除非您取消注释具有不受支持的转换的最后一行，否则在最后将抛出异常。

我们的解决方案比基于动态多态性的解决方案要高效得多，尽管受支持的状态和事件列表受限于编译时提供的状态。有关状态、变体和各种访问方式的更多信息，请参阅 Mateusz Pusz 在 CppCon 2018 的演讲，也列在*进一步阅读*部分中。

在我们结束本章之前，我们想让您了解的最后一件事是处理内存。让我们开始我们的最后一节。

# 高效处理内存

即使您没有非常有限的内存，查看您如何使用它也是一个好主意。通常，内存吞吐量是现代系统的性能瓶颈，因此始终重要的是充分利用它。执行太多的动态分配可能会减慢程序速度并导致内存碎片化。让我们学习一些减轻这些问题的方法。

## 使用 SSO/SOO 减少动态分配

动态分配有时会给您带来麻烦，不仅在构造对象时抛出异常，而且还会花费 CPU 周期并导致内存碎片化。幸运的是，有一种方法可以防范这种情况。如果您曾经使用过`std::string`（GCC 5.0 之后），您很可能使用了一种称为**小字符串优化**（**SSO**）的优化。这是**小对象优化**（**SSO**）的一个更普遍的优化的例子，可以在 Abseil 的 InlinedVector 等类型中找到。其主要思想非常简单：如果动态分配的对象足够小，它应该存储在拥有它的类内部，而不是动态分配。在`std::string`的情况下，通常有容量、长度和实际要存储的字符串。如果字符串足够短（在 GCC 的情况下，在 64 位平台上，它是 15 个字节），它将存储在其中的某些成员中。

将对象存储在原地而不是在其他地方分配并仅存储指针还有一个好处：减少指针追踪。每次需要访问指针后面存储的数据时，都会增加 CPU 缓存的压力，并有可能需要从主内存中获取数据。如果这是一个常见的模式，它可能会影响您的应用程序的整体性能，特别是如果 CPU 的预取器没有猜测到指向的地址。使用 SSO 和 SOO 等技术在减少这些问题方面是非常宝贵的。

## 通过管理 COW 来节省内存

如果您在 GCC 5.0 之前使用过 GCC 的`std::string`，您可能使用了一种称为**写时复制**（**COW**）的不同优化。当使用相同的基础字符数组创建多个实例时，COW 字符串实现实际上共享相同的内存地址。当字符串被写入时，基础存储被复制，因此得名。

这种技术有助于节省内存并保持高速缓存热度，并且通常在单线程上提供了可靠的性能。但要注意在多线程环境中使用它。使用锁可能会严重影响性能。与任何与性能相关的主题一样，最好的方法是测量在您的情况下是否是最佳工具。

现在让我们讨论一下 C++17 的一个功能，它可以帮助您实现动态分配的良好性能。

## 利用多态分配器

我们正在讨论的功能是多态分配器。具体来说，是`std::pmr::polymorphic_allocator`和多态`std::pmr::memory_resource`类，分配器使用它来分配内存。

本质上，它允许您轻松地链接内存资源，以充分利用您的内存。链可以简单到一个资源保留一个大块并分配它，然后退回到另一个资源，如果它耗尽内存，就简单地调用`new`和`delete`。它们也可以更复杂：您可以构建一个长链的内存资源，处理不同大小的池，仅在需要时提供线程安全性，绕过堆直接使用系统内存，返回您最后释放的内存块以提供高速缓存，以及执行其他花哨的操作。并非所有这些功能都由标准多态内存资源提供，但由于它们的设计，很容易扩展它们。

让我们首先讨论内存区域的主题。

### 使用内存区域

内存区域，也称为区域，只是存在有限时间的大块内存。您可以使用它来分配您在区域的生命周期内使用的较小对象。区域中的对象可以像往常一样被释放，或者在称为*闪烁*的过程中一次性擦除。我们稍后会描述它。

区域相对于通常的分配和释放具有几个巨大的优势-它们提高了性能，因为它们限制了需要获取上游资源的内存分配。它们还减少了内存的碎片化，因为任何碎片化都将发生在区域内。一旦释放了区域的内存，碎片化也就消失了。一个很好的主意是为每个线程创建单独的区域。如果只有一个线程使用区域，它就不需要使用任何锁定或其他线程安全机制，减少了线程争用，并为您提供了良好的性能提升。

如果您的程序是单线程的，提高其性能的低成本解决方案可能如下：

```cpp
  auto single_threaded_pool = std::pmr::unsynchronized_pool_resource();

  std::pmr::set_default_resource(&single_threaded_pool);
```

如果您不明确设置任何资源，那么默认资源将是`new_delete_resource`，它每次调用`new`和`delete`，就像常规的`std::allocator`一样，并且具有它提供的所有线程安全性（和成本）。

如果您使用前面的代码片段，那么使用`pmr`分配器进行的所有分配都将不使用锁。但是，您仍然需要实际使用`pmr`类型。例如，要在标准容器中使用，您只需将`std::pmr::polymorphic_allocator<T>`作为分配器模板参数传递。许多标准容器都有启用`pmr`的类型别名。接下来创建的两个变量是相同类型，并且都将使用默认内存资源：

```cpp
  auto ints = std::vector<int, std::pmr::polymorphic_allocator<int>>(std::pmr::get_default_resource());

  auto also_ints = std::pmr::vector<int>{};
```

第一个显式传递资源。现在让我们来看看`pmr`中可用的资源。

### 使用单调内存资源

我们将讨论的第一个是`std::pmr::monotonic_buffer_resource`。它是一个只分配内存而不在释放时执行任何操作的资源。它只会在资源被销毁或显式调用`release()`时释放内存。这种类型与无线程安全连接，使其极其高效。如果您的应用程序偶尔需要在给定线程上执行大量分配的任务，然后随后一次性释放所有使用的对象，使用单调资源将带来巨大的收益。它也是链式资源的一个很好的基本构建块。

### 使用池资源

两种资源的常见组合是在单调缓冲区资源之上使用池资源。标准池资源创建不同大小块的池。在`std::pmr`中有两种类型，`unsynchronized_pool_resource`用于仅有一个线程从中分配和释放的情况，`synchronized_pool_resource`用于多线程使用。与全局分配器相比，两者都应该提供更好的性能，特别是在使用单调缓冲区作为上游资源时。如果您想知道如何链接它们，下面是方法：

```cpp
  auto buffer = std::array<std::byte, 1 * 1024 * 1024>{};

  auto monotonic_resource =

      std::pmr::monotonic_buffer_resource{buffer.data(), buffer.size()};

  auto pool_options = std::pmr::pool_options{.max_blocks_per_chunk = 0,

      .largest_required_pool_block = 512};

  auto arena =

      std::pmr::unsynchronized_pool_resource{pool_options, &monotonic_resource};
```

我们为区域创建了一个 1 MB 的缓冲区以供重复使用。我们将其传递给单调资源，然后传递给不同步的池资源，从而创建一个简单而有效的分配器链，直到使用完所有初始缓冲区之前都不会调用 new。

您可以将`std::pmr::pool_options`对象传递给两种池类型，以限制给定大小的块的最大数量（`max_blocks_per_chunk`）或最大块的大小（`largest_required_pool_block`）。传递 0 会导致使用实现的默认值。在 GCC 库的情况下，实际块的数量取决于块的大小。如果超过最大大小，池资源将直接从其上游资源分配。如果初始内存耗尽，它也会转向上游资源。在这种情况下，它会分配几何增长的内存块。

### 编写自己的内存资源

如果标准内存资源不适合您的所有需求，您总是可以相当简单地创建自定义资源。例如，不是所有标准库实现都提供的一个很好的优化是跟踪已释放的给定大小的最后一块块，并在下一个给定大小的分配上将它们返回。这个`最近使用`缓存可以帮助您增加数据缓存的热度，这应该有助于您的应用程序性能。您可以将其视为一组用于块的 LIFO 队列。

有时，您可能还希望调试分配和释放。在下面的代码片段中，我编写了一个简单的资源，可以帮助您完成这项任务：

```cpp
class verbose_resource : public std::pmr::memory_resource {

  std::pmr::memory_resource *upstream_resource_;

public:

  explicit verbose_resource(std::pmr::memory_resource *upstream_resource)

      : upstream_resource_(upstream_resource) {}
```

我们的冗长资源继承自多态基础资源。它还接受一个上游资源，它将用于实际分配。它必须实现三个私有函数 - 一个用于分配，一个用于释放，一个用于比较资源实例。这是第一个：

```cpp
private:

  void *do_allocate(size_t bytes, size_t alignment) override {

    std::cout << "Allocating " << bytes << " bytes\n";

    return upstream_resource_->allocate(bytes, alignment);

  }
```

它只是在标准输出上打印分配大小，然后使用上游资源来分配内存。下一个将类似：

```cpp
  void do_deallocate(void *p, size_t bytes, size_t alignment) override {

    std::cout << "Deallocating " << bytes << " bytes\n";

    upstream_resource_->deallocate(p, bytes, alignment);

  }
```

我们记录我们释放多少内存并使用上游执行任务。现在下一个所需的最后一个函数被陈述如下：

```cpp
  [[nodiscard]] bool

  do_is_equal(const memory_resource &other) const noexcept override {

    return this == &other;

  }
```

我们只需比较实例的地址，以知道它们是否相等。`[[nodiscard]]`属性可以帮助我们确保调用者实际上消耗了返回的值，这可以帮助我们避免意外滥用我们的函数。

就是这样。对于`pmr`分配器这样一个强大的功能，API 现在并不那么复杂，是吗？

除了跟踪分配之外，我们还可以使用`pmr`来防止在不应该分配时进行分配。

### 确保没有意外的分配

特殊的`std::pmr::null_memory_resource()`将在任何人尝试使用它分配内存时抛出异常。您可以通过将其设置为默认资源来防止使用`pmr`执行任何分配，如下所示：

```cpp
std::pmr::set_default_resource(null_memory_resource());
```

您还可以使用它来限制在不应该发生时从上游分配。检查以下代码：

```cpp
  auto buffer = std::array<std::byte, 640 * 1024>{}; // 640K ought to be enough for anybody

  auto resource = std::pmr::monotonic_buffer_resource{

      buffer.data(), buffer.size(), std::pmr::null_memory_resource()};
```

如果有人尝试分配超过我们设置的缓冲区大小，将抛出`std::bad_alloc`。

让我们继续进行本章的最后一项任务。

### 眨眼内存

有时不需要释放内存，就像单调缓冲资源一样，对性能来说还不够。一种称为*眨眼*的特殊技术可以在这里帮助。眨眼对象意味着它们不仅不是逐个释放，而且它们的构造函数也不会被调用。对象只是蒸发，节省了通常用于调用每个对象及其成员（和它们的成员...）的析构函数所需的时间。

注意：这是一个高级主题。在使用这种技术时要小心，并且只有在可能的收益值得时才使用它。

这种技术可以节省您宝贵的 CPU 周期，但并非总是可能使用它。如果您的对象处理的资源不仅仅是内存，避免眨眼内存。否则，您将会出现资源泄漏。如果您依赖于对象的析构函数可能产生的任何副作用，情况也是如此。

现在让我们看看眨眼的实际效果：

```cpp
  auto verbose = verbose_resource(std::pmr::get_default_resource());

  auto monotonic = std::pmr::monotonic_buffer_resource(&verbose);

  std::pmr::set_default_resource(&monotonic);


  auto alloc = std::pmr::polymorphic_allocator{};

  auto *vector = alloc.new_object<std::pmr::vector<std::pmr::string>>();

  vector->push_back("first one");

  vector->emplace_back("long second one that must allocate");
```

在这里，我们手工创建了一个多态分配器，它将使用我们的默认资源——一个每次到达上游时都会记录的单调资源。为了创建对象，我们将使用 C++20 中对`pmr`的新增功能`new_object`函数。我们创建了一个字符串向量。我们可以使用`push_back`传递第一个字符串，因为它足够小，可以适应我们由于 SSO 而拥有的小字符串缓冲区。第二个字符串需要使用默认资源分配一个字符串，然后才能将其传递给我们的向量，如果我们使用`push_back`。将其置于内部会导致字符串在调用之前（而不是之前）在向量的函数内部构造，因此它将使用向量的分配器。最后，我们没有在任何地方调用分配对象的析构函数，只有在退出作用域时才释放所有内容。这应该给我们带来难以超越的性能。

这是本章的最后一项内容。让我们总结一下我们学到的东西。

# 总结

在本章中，我们介绍了 C++世界中使用的各种习语和模式。现在你应该能够流利地编写 C++。我们已经揭开了如何执行自动清理的神秘面纱。您现在可以编写更安全的类型，正确地移动、复制和交换。您学会了如何利用 ADL 来优化编译时间和编写定制点。我们讨论了如何在静态和动态多态性之间进行选择。我们还学会了如何向类型引入策略，何时使用类型擦除，何时不使用。

此外，我们还讨论了如何使用工厂和流畅构建器创建对象。此外，使用内存区域也不再是神秘的魔法。使用诸如变体之类的工具编写状态机也是如此。

我们还触及了一些额外的话题。哦！我们旅程的下一站将是关于构建软件和打包的内容。

# 问题

1.  三、五和零的规则是什么？

1.  我们何时使用 niebloids 而不是隐藏的友元？

1.  如何改进数组接口以使其更适合生产？

1.  折叠表达式是什么？

1.  何时不应该使用静态多态性？

1.  在眨眼示例中，我们如何节省一次额外的分配？

# 进一步阅读

1.  *tag_invoke：支持可定制函数的通用模式*，Lewis Baker，Eric Niebler，Kirk Shoop，ISO C++提案，[`wg21.link/p1895`](https://wg21.link/p1895)

1.  *tag_invoke :: niebloids 进化*，Gašper Ažman 为 Core C++会议制作的 YouTube 视频，[`www.youtube.com/watch?v=oQ26YL0J6DU`](https://www.youtube.com/watch?v=oQ26YL0J6DU)

1.  *继承是邪恶的基类*，Sean Parent 为 GoingNative 2013 会议制作的 Channel9 视频，[`channel9.msdn.com/Events/GoingNative/2013/Inheritance-Is-The-Base-Class-of-Evil`](https://channel9.msdn.com/Events/GoingNative/2013/Inheritance-Is-The-Base-Class-of-Evil)

1.  *现代 C++设计*，Andrei Alexandrescu，Addison-Wesley，2001

1.  *非成员函数如何改进封装*，Scott Meyers，Dr. Dobbs 文章，[`www.drdobbs.com/cpp/how-non-member-functions-improve-encapsu/184401197`](https://www.drdobbs.com/cpp/how-non-member-functions-improve-encapsu/184401197)

1.  *返回状态或值*，Status 用户指南，Abseil 文档，[`abseil.io/docs/cpp/guides/status#returning-a-status-or-a-value`](https://abseil.io/docs/cpp/guides/status#returning-a-status-or-a-value)

1.  `function_ref`，GitHub 存储库，[`github.com/TartanLlama/function_ref`](https://github.com/TartanLlama/function_ref)

1.  *如何使用 std::visit 处理多个变体*，Bartłomiej Filipek，Bartek 的编码博客文章，[`www.bfilipek.com/2018/09/visit-variants.html`](https://www.bfilipek.com/2018/09/visit-variants.html)

1.  CppCon 2018：Mateusz Pusz，*使用 std::variant 有效替代动态多态性*，YouTube 视频，[`www.youtube.com/watch?v=gKbORJtnVu8`](https://www.youtube.com/watch?v=gKbORJtnVu8)
