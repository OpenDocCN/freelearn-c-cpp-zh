

# 第二十一章：STL 与概念和协程的交互

本章将探讨 STL 与 C++ 的两个高级特性之间的相互作用：概念和协程。本章旨在加深你对这些现代 C++ 特性如何增强并与 STL 交互的理解。

我们首先学习关于概念的知识，从介绍开始，逐步探索它们在细化 STL 算法约束、增强数据结构和开发自定义概念中的作用。这一部分对于理解显式类型约束如何导致更健壮和可读的代码至关重要。

接下来，我们将重点关注协程，在检查它们与 STL 算法和数据结构的集成之前，提供一个复习。这包括探索与范围和视图的潜在协同作用，最终讨论协程可能预示着 C++ 编程范式的转变。

本章将全面了解并深入探讨如何有效地使用这些特性，强调它们在现代 C++ 开发中的重要性及其潜在挑战。

在本章中，我们将涵盖以下主题：

+   概念

+   协程

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 概念

C++20 中概念的引入标志着朝着更安全和更具表达性的模板编程迈出的关键一步。凭借其指定模板参数约束的固有能力，概念承诺将重塑我们与 **标准模板库**（**STL**）交互和利用的方式。让我们发现概念如何与 STL 算法和数据结构的丰富织锦交织，以创建一个更健壮和声明性的 C++ 编程范式。

## 概念简介

**概念**提供了一种指定和检查模板参数约束的机制。本质上，它们允许开发者对传递给模板的类型提出要求。概念旨在使模板错误更易于阅读，帮助避免常见陷阱，并促进更通用和可重用代码的创建。

考虑以下关于算术类型的概念：

```cpp
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;
```

使用这个概念，可以限制一个函数只接受算术类型：

```cpp
template<Arithmetic T>
T add(T a, T b) { return (a + b); }
```

## STL 算法中的细化约束

历史上，STL 算法依赖于其模板参数的复杂、有时模糊的要求。有了概念，这些要求变得明确和可理解。例如，`std::sort` 算法需要随机访问迭代器，现在可以使用概念来断言。如果错误地使用列表（仅提供双向迭代器），这将导致更精确的错误消息。

## 有效地约束模板

在使用 C++ 模板编程时，确保给定的类型满足一组特定的要求在历史上一直是一个挑战。在引入概念之前，开发者会依赖于涉及**替换失败不是错误**（**SFINAE**）或专门的特质类等复杂技术的技术。这些方法冗长且容易出错，通常会导致难以理解的错误信息。

概念允许开发者定义一组类型必须满足的谓词，提供了一种更结构化和可读性的方式来约束模板。使用概念，你可以指定模板参数必须满足的要求。当一个类型不符合概念定义的约束时，编译器将拒绝模板实例化，生成更直接、更有意义的错误信息。这增强了模板代码的可读性、可维护性和健壮性。使用概念，编译器可以快速确定类型对给定模板的适用性，确保只使用合适的类型，从而最小化运行时错误或未定义行为的风险。

下面是一个代码示例，展示了概念的使用以及在没有引入概念之前如何完成相同的任务：

```cpp
#include <iostream>
#include <type_traits>
// Create a class that is not inherently printable.
struct NotPrintable
{
  int foo{0};
  int bar{0};
};
// Concept definition using the 'requires' clause
template <typename T>
concept Printable = requires(T t) {
  // Requires that t can be printed to std::cout
  std::cout << t;
};
// Before C++20:
// A Function template that uses SFINAE to implement a
// "Printable concept"
template <typename T,
          typename = typename std::enable_if<std::is_same<
              decltype(std::cout << std::declval<T>()),
              std::ostream &>::value>::type>
void printValueSFINAE(const T &value) {
  std::cout << "Value: " << value << "\n";
}
// After C++20:
// A Function template that uses the Printable concept
template <Printable T> void printValue(const T &value) {
  std::cout << "Value: " << value << "\n";
}
int main() {
  const int num = 42;
  const NotPrintable np;
  const std::string str = "Hello, Concepts!";
  // Using the function template with SFINAE
  printValueSFINAE(num);
  // This line would fail to compile:
  // printValueSFINAE(np);
  printValueSFINAE(str);
  // Using the function template with concepts
  printValue(num);
  // This line would fail to compile
  // printValue(np);
  printValue(str);
  return 0;
}
```

下面是示例输出：

```cpp
Value: 42
Value: Hello, Concepts!
Value: 42
Value: Hello, Concepts!
```

在这个例子中，我们使用所需子句定义了一个名为 `Printable` 的概念。`Printable` 概念检查一个类型是否可以被打印到 `std::cout`。然后我们有两个函数模板 `printValue` 和 `printValueSFINAE`，分别用于满足概念或 SFINAE 条件时打印值。

当使用带有 `Printable` 概念的 `printValue` 函数模板时，编译器将确保传递给它的类型可以被打印，如果不能，它将生成清晰的错误信息。这使得代码更易于阅读，并提供了有意义的错误信息。

另一方面，当使用 `printValueSFINAE` 函数模板时，我们依赖于 SFINAE 来完成相同的任务。这种方法更冗长且容易出错，因为它涉及到复杂的 `std::enable_if` 构造，并且在约束未满足时可能导致难以理解的错误信息。

通过比较这两种方法，你可以看到概念如何提高 C++ 模板代码的可读性、可维护性和健壮性，使其更容易指定和执行类型要求。

## 带有显式要求的增强数据结构

STL 容器，如 `std::vector` 或 `std::map`，通常对存储的类型有要求，例如必须是可复制的或可赋值的。概念可以非常清晰地表达这些要求。

想象一个自定义容器，它要求其元素必须能够使用默认构造函数。这个要求可以用概念来优雅地表达，从而确保容器行为更安全、更可预测。

## 自定义概念和 STL 交互

概念的一个优点是它们不仅限于标准库提供的那些。开发者可以创建定制的概念，以满足特定需求，确保 STL 结构和算法可以适应独特和复杂的场景，同时不牺牲类型安全。

例如，如果某个算法需要具有特定接口的类型（例如具有 `draw()` 成员函数），则可以设计一个概念来强制执行此要求，从而实现更直观和自文档化的代码。让我们看看一个代码示例：

```cpp
#include <concepts>
#include <iostream>
#include <vector>
template <typename T>
concept Drawable = requires(T obj) {
  { obj.draw() } -> std::convertible_to<void>;
};
class Circle {
public:
  void draw() const { std::cout << "Drawing a circle.\n"; }
};
class Square {
public:
  // No draw() member function
};
template <Drawable T> void drawShape(const T &shape) {
  shape.draw();
}
int main() {
  Circle circle;
  Square square;
  drawShape(circle);
  // Uncommenting the line below would result in
  // 'drawShape': no matching overloaded function found:
  // drawShape(square);
  return 0;
}
```

下面是示例输出：

```cpp
Drawing a circle.
```

在前面的代码示例中，我们做了以下操作：

+   我们定义了一个名为 `Drawable` 的自定义概念，它要求类型具有返回 `void` 的 `draw()` 成员函数。

+   我们创建了两个示例类：`Circle`，它通过具有 `draw()` 成员函数满足 `Drawable` 概念，而 `Square` 则不满足该概念，因为它缺少 `draw()` 成员函数。

+   我们定义了一个名为 `drawShape` 的泛型函数，它接受一个 `Drawable` 类型作为参数并调用其 `draw()` 成员函数。

+   在 `main` 函数中，我们创建了 `Circle` 和 `Square` 的实例，并演示了 `drawShape` 可以用 `Drawable` 类型（例如 `Circle`）调用，但不能用不满足 `Drawable` 概念的类型（例如 `Square`）调用。

此示例说明了如何使用自定义概念来强制执行特定的接口要求，确保类型安全，并在处理 C++ 中的复杂场景和算法时使代码更直观和自文档化。

## 潜在的挑战和注意事项

虽然概念无疑很强大，但有一些考虑事项需要考虑：

+   **复杂性**：设计复杂的自定义概念可能具有挑战性，可能会增加新手的学习曲线。

+   **编译时间**：与大多数基于模板的功能一样，过度依赖或误用可能会增加编译时间。

+   **向后兼容性**：较老的代码库可能需要重构才能充分利用或完全符合新的概念驱动约束。

本节介绍了 C++ 中一个强大的功能，允许我们指定模板参数的约束。我们首先简要介绍了概念，理解它们在增强代码可表达性和安全性方面的作用。然后，我们探讨了如何将精细的约束应用于 STL 算法，从而实现更健壮和可读的代码。我们还学习了如何有效地约束模板，这对于防止代码误用和确保代码按预期行为至关重要。

然而，我们也承认了与概念相关的潜在挑战和注意事项。虽然它们提供了许多好处，但明智地使用它们以避免不必要的复杂性和潜在陷阱是很重要的。

从本节中获得的知识是无价的，因为它为我们提供了使用 STL 编写更安全、更表达性和更高效代码的工具。它还为我们为下一节做准备，我们将探索 C++ 的另一个令人兴奋的特性：协程。

下一节将刷新我们对协程的理解，并讨论它们与 STL 算法和数据结构的集成。我们还将探索与范围和视图的潜在协同作用，这可能导致更高效、更优雅的代码。最后，我们将探讨协程如何代表我们在编写异步代码时的一种范式转变。

# 协程

协程集成到 C++20 中引入了异步编程的新范式，这使得代码更易读、更直观。通过允许函数暂停和恢复，协程为在异步 C++ 代码中常见的回调密集型风格提供了一种替代方案。这种演变本身具有变革性，同时也提供了与尊贵的 STL 交互的新鲜、创新方式。研究协程与 STL 算法和数据结构的交互揭示了它们如何简化异步操作。

## 理解协程——复习

`co_await`、`co_return` 和 `co_yield`：

+   `co_await`: 暂停当前协程，直到等待的表达式准备好，此时协程继续

+   `co_return`: 这用于结束协程，可能返回一个值

+   `co_yield`: 以生成器的方式产生一个值，允许对协程进行迭代

## STL 算法和协程集成

使用协程后，之前需要更复杂异步方法的 STL 算法现在可以以直接、线性的逻辑优雅地编写。考虑在序列或范围内操作的计算算法；它们可以与协程结合，以异步方式生成值。

例如，一个协程可以异步产生值，然后使用 `std::transform` 或 `std::for_each` 处理这些值，将异步代码与同步 STL 算法无缝结合。

## 协程和 STL 数据结构

协程的魔法也触及了 STL 数据结构的领域。协程为容器如 `std::vector` 或 `std::list` 提供了有趣的潜在用途：`填充`（异步）。

想象一个场景，其中必须从网络源获取数据并将其存储在 `std::vector` 中。可以使用协程异步获取数据，在数据到达时产生值，然后直接将这些值插入到向量中。这种异步与 STL 数据结构直接性的结合简化了代码并减少了认知开销。

## 与范围和视图的潜在协同作用

随着 C++语言的演变，其他特性，如范围和视图，与协程结合，可以提供一种更表达性的方式来处理数据操作和转换。协程可以生成范围，这些范围可以延迟评估、过滤和转换，使用视图，从而实现一个强大且可组合的异步编程模型。

让我们看看以下涉及以下步骤的代码示例：

+   `std::vector<int>` 用于存储数字序列。

+   **协程**：一个异步生成数字以填充我们的向量的生成器。

+   `std::ranges::copy_if`。

+   `std::views::transform`，我们将每个数字乘以二。首先，我们必须创建一个特殊的 `promise_type` 结构的 `generator` 类，我们的协程将使用它。在这个代码中，生成器类模板及其嵌套的 `promise_type` 结构是实现 C++中生成值序列协程的关键组件。

根据请求，一次一个 `T`。它封装了协程的状态，并提供了一个接口来控制其执行和访问产生的值。

在生成器内部嵌套的 `promise_type` 是协程的生命周期和状态管理核心。它保存要产生的当前值（value）并定义了几个关键函数：

+   `get_return_object`：返回与该协程关联的生成器对象

+   `initial_suspend` 和 `final_suspend`：控制协程的执行，初始暂停并在完成后暂停

+   `unhandled_exception`：定义未处理异常的行为，终止程序

+   `return_void`：当协程到达其末尾时的占位符

+   `yield_value`：当产生一个值时（co_yield），暂停协程并存储产生的值

下面的代码示例被分成几个部分（完整的示例可以在书籍的 GitHub 仓库中找到）：

```cpp
template <typename T> class generator {
public:
  struct promise_type {
    T value;
    auto get_return_object() {
      return generator{handle_type::from_promise(*this)};
    }
    auto initial_suspend() {
      return std::suspend_always{};
    }
    auto final_suspend() noexcept {
      return std::suspend_always{};
    }
    void unhandled_exception() { std::terminate(); }
    void return_void() {}
    auto yield_value(T x) {
      value = x;
      return std::suspend_always{};
    }
  };
  using handle_type = std::coroutine_handle<promise_type>;
  generator(handle_type h) : m_handle(h) {}
  generator(const generator &) = delete;
  generator(generator &&o) noexcept
      : m_handle(std::exchange(o.m_handle, {})) {}
  ~generator() {
    if (m_handle) m_handle.destroy();
  }
  bool next() {
    m_handle.resume();
    return !m_handle.done();
  }
  T value() const { return m_handle.promise().value; }
private:
  handle_type m_handle;
};
```

上述代码定义了一个名为 `generator` 的泛型模板类。这个类被实例化为 `generate_numbers` 函数的返回类型，该函数创建从开始到结束的整数序列。当被调用时，它启动一个协程，迭代地产生指定范围内的整数。每次迭代都会暂停协程，使当前值对调用者可用。生成器类提供了恢复协程（`next()`）和检索当前值（`value()`）的机制。生成器的构造函数、移动构造函数、析构函数和已删除的复制构造函数管理协程的生命周期并确保适当的资源管理。

这就是难点所在。现在，我们可以开始构建和使用我们的协程：

```cpp
generator<int> generate_numbers(int start, int end) {
  for (int i = start; i <= end; ++i) { co_yield i; }
}
int main() {
  std::vector<int> numbers;
  auto gen = generate_numbers(1, 10);
  while (gen.next()) { numbers.push_back(gen.value()); }
  std::vector<int> evenNumbers;
  std::ranges::copy_if(numbers,
                       std::back_inserter(evenNumbers),
                       [](int n) { return n % 2 == 0; });
  const auto transformed =
      evenNumbers |
      std::views::transform([](int n) { return n * 2; });
  for (int n : transformed) { std::cout << n << " "; }
  return 0;
}
```

以下是示例输出：

```cpp
4 8 12 16 20
```

在这个例子中，我们做了以下几件事：

+   我们创建了一个 `generator` 类来表示异步生成器。

+   我们使用 `generate_numbers` 协程异步生成从 1 到 10 的数字。

+   使用范围，我们只过滤出偶数并将它们存储在另一个向量中。

+   使用视图，我们将这些偶数乘以二进行转换。

+   最后，我们输出了转换后的序列。

## 展望未来——范式转变

C++ 中的协程代表了异步编程领域的一项重大进步。通过引入处理异步任务的标准方式，协程促进了编写非阻塞、高效和可维护的代码。当与 STL 结合使用时，协程有可能简化复杂操作，改变 C++ 编程的格局。

STL 为数据操作和算法实现提供了一个健壮的框架。协程的引入通过提供一种比传统线程机制更不易出错、更直观的并发模型来增强这个框架。这种协同作用允许开发复杂的异步程序，利用 STL 容器、迭代器和算法的全部力量，而不牺牲性能。

随着协程在 STL 中的集成越来越紧密，我们预期将出现一种范式转变，其中高性能代码不仅以其速度为特征，还将以其清晰性和模块化结构为特征。协程的采用预计将扩大，这得益于它们产生可扩展和响应性软件的能力。

C++ 标准的未来迭代可能会引入更多功能，以补充协程-STL 接口，为开发者提供更丰富的工具集。这种演变将巩固 C++ 作为开发高性能、异步应用程序的首选语言的地位。C++ 社区对持续改进的承诺保持了该语言在解决现代编程挑战中的相关性和有效性。

# 摘要

本章揭示了 C++20 的概念和协程与 STL 的集成。我们首先探讨了概念在模板编程中的作用。概念通过强制类型约束和增强模板使用的可表达性和安全性来增强代码的健壮性。它们用更易读和声明性的语法替换了易出错的 SFINAE 技术。我们看到概念如何提高算法要求的可清晰性，从而产生更易于维护的代码。

接下来，我们探讨了协程如何为 C++ 中的异步编程引入一个新的复杂层次。我们讨论了协程的机制，强调使用 `co_await`、`co_return` 和 `co_yield` 来创建非阻塞操作。我们研究了协程如何与 STL 数据结构和算法交互，使异步和同步代码能够无缝融合。

理解概念、协程和 STL 之间的相互作用至关重要。它使我们能够编写不仅性能出色，而且清晰和可靠的代码。这种知识使我们能够自信和有远见地应对复杂的编程场景。

接下来，我们将专注于应用能够使 STL 算法实现并行的执行策略。本章将引导我们了解并行执行策略的细微差别，`constexpr` 在提升编译时优化中的作用，以及在并发环境中实现最佳性能的最佳实践。
