# 10

# 代理对象和延迟评估

在本章中，您将学习如何使用代理对象和延迟评估，以推迟执行某些代码直到需要。使用代理对象可以在后台进行优化，从而保持公开的接口不变。

本章涵盖了：

+   懒惰和急切评估

+   使用代理对象避免多余的计算

+   在使用代理对象时重载运算符

# 引入延迟评估和代理对象

首先，本章中使用的技术是用于隐藏库中的优化技术，不让库的用户看到。这很有用，因为将每个单独的优化技术公开为一个单独的函数需要用户的大量关注和教育。它还使代码库膨胀了大量特定的函数，使其难以阅读和理解。通过使用代理对象，我们可以在后台实现优化；结果代码既经过优化又易读。

## 懒惰与急切评估

**懒惰** **评估**是一种技术，用于推迟操作，直到真正需要其结果。相反，立即执行操作的情况称为**急切评估**。在某些情况下，急切评估是不希望的，因为我们可能最终构造一个从未使用的值。

为了演示急切和懒惰评估之间的差异，让我们假设我们正在编写某种具有多个级别的游戏。每当完成一个级别时，我们需要显示当前分数。在这里，我们将专注于游戏的一些组件：

+   一个`ScoreView`类负责显示用户的分数，如果获得了奖励，则显示可选的奖励图像

+   代表加载到内存中的图像的`Image`类

+   从磁盘加载图像的`load()`函数

在这个例子中，类和函数的实现并不重要，但声明看起来是这样的：

```cpp
class Image { /* ... */ };                   // Buffer with JPG data
auto load(std::string_view path) -> Image;   // Load image at path
class ScoreView {
public:
  // Eager, requires loaded bonus image
  void display(const Image& bonus);
  // Lazy, only load bonus image if necessary
  void display(std::function<Image()> bonus);
  // ...
}; 
```

提供了两个`display()`版本：第一个需要完全加载的奖励图像，而第二个接受一个只在需要奖励图像时调用的函数。使用第一个*急切*版本会是这样：

```cpp
// Always load bonus image eagerly
const auto eager = load("/images/stars.jpg");
score.display(eager); 
```

使用第二个*懒惰*版本会是这样：

```cpp
// Load default image lazily if needed
auto lazy = [] { return load("/images/stars.jpg"); }; 
score.display(lazy); 
```

急切版本将始终将默认图像加载到内存中，即使它从未显示过。然而，奖励图像的延迟加载将确保只有在`ScoreView`真正需要显示奖励图像时才加载图像。

这是一个非常简单的例子，但其思想是，您的代码几乎以与急切声明相同的方式表达。隐藏代码懒惰评估的技术是使用代理对象。

## 代理对象

代理对象是内部库对象，不打算对库的用户可见。它们的任务是推迟操作直到需要，并收集表达式的数据，直到可以评估和优化。然而，代理对象在黑暗中行事；库的用户应该能够处理表达式，就好像代理对象不存在一样。换句话说，使用代理对象，您可以在库中封装优化，同时保持接口不变。现在您将学习如何使用代理对象来懒惰地评估更高级的表达式。

# 使用代理对象避免构造对象

急切评估可能会导致不必要地构造对象。通常这不是问题，但如果对象昂贵（例如因为堆分配），可能有合理的理由优化掉无用的短暂对象的构造。

## 使用代理对象比较连接的字符串

现在我们将通过一个使用代理对象的最小示例，让您了解它们是什么以及可以用于什么。它并不意味着为您提供一个通用的生产就绪的优化字符串比较解决方案。

话虽如此，看看这段代码片段，它连接两个字符串并比较结果：

```cpp
auto a = std::string{"Cole"}; 
auto b = std::string{"Porter"}; 
auto c = std::string{"ColePorter"}; 
auto is_equal = (a + b) == c;        // true 
```

这是前面代码片段的可视化表示：

![](img/B15619_10_01.png)

图 10.1：将两个字符串连接成一个新字符串

问题在于，(`a + b`)构造了一个新的临时字符串，以便将其与`c`进行比较。我们可以直接比较连接，而不是构造一个新的字符串，就像这样：

```cpp
auto is_concat_equal(const std::string& a, const std::string& b,
                     const std::string& c) { 
  return  
    a.size() + b.size() == c.size() && 
    std::equal(a.begin(), a.end(), c.begin()) &&  
    std::equal(b.begin(), b.end(), c.begin() + a.size()); 
} 
```

然后我们可以这样使用它：

```cpp
auto is_equal = is_concat_equal(a, b, c); 
```

就性能而言，我们取得了胜利，但从语法上讲，一个代码库中充斥着这种特殊情况的便利函数很难维护。因此，让我们看看如何在保持原始语法不变的情况下实现这种优化。

## 实现代理

首先，我们将创建一个代表两个字符串连接的代理类：

```cpp
struct ConcatProxy { 
  const std::string& a; 
  const std::string& b; 
}; 
```

然后，我们将构建自己的`String`类，其中包含一个`std::string`和一个重载的`operator+()`函数。请注意，这是如何创建和使用代理对象的示例；创建自己的`String`类不是我推荐的做法：

```cpp
class String { 
public: 
  String() = default; 
  String(std::string str) : str_{std::move(str)} {} 
  std::string str_{};
}; 

auto operator+(const String& a, const String& b) {
   return ConcatProxy{a.str_, b.str_};
} 
```

这是前面代码片段的可视化表示：

![](img/B15619_10_02.png)

图 10.2：代表两个字符串连接的代理对象

最后，我们将创建一个全局的`operator==()`函数，该函数将使用优化的`is_concat_equal()`函数，如下所示：

```cpp
auto operator==(ConcatProxy&& concat, const String& str) {
  return is_concat_equal(concat.a, concat.b, str.str_); 
} 
```

现在我们已经准备就绪，可以兼得两全：

```cpp
auto a = String{"Cole"}; 
auto b = String{"Porter"}; 
auto c = String{"ColePorter"}; 
auto is_equal = (a + b) == c;     // true 
```

换句话说，我们在保留使用`operator==()`的表达语法的同时，获得了`is_concat_equal()`的性能。

## rvalue 修饰符

在前面的代码中，全局的`operator==()`函数只接受`ConcatProxy` rvalues：

```cpp
auto operator==(ConcatProxy&& concat, const String& str) { // ... 
```

如果我们接受一个`ConcatProxy` lvalue，我们可能会意外地误用代理，就像这样：

```cpp
auto concat = String{"Cole"} + String{"Porter"};
auto is_cole_porter = concat == String{"ColePorter"}; 
```

问题在于，持有`"Cole"`和`"Porter"`的临时`String`对象在比较执行时已被销毁，导致失败。（请记住，`ConcatProxy`类只持有对字符串的引用。）但由于我们强制`concat`对象为 rvalue，前面的代码将无法编译，从而避免了可能的运行时崩溃。当然，你可以通过使用`std::move(concat) == String("ColePorter")`将其强制编译，但这不是一个现实的情况。

## 分配一个连接的代理

现在，你可能会想，如果我们实际上想将连接的字符串存储为一个新的字符串而不仅仅是比较它，该怎么办？我们所做的就是简单地重载一个`operator String()`函数，如下所示：

```cpp
struct ConcatProxy {
  const std::string& a;
  const std::string& b;
  operator String() const && { return String{a + b}; }
}; 
```

两个字符串的连接现在可以隐式转换为一个字符串：

```cpp
String c = String{"Marc"} + String{"Chagall"}; 
```

不过，有一个小问题：我们无法使用`auto`关键字初始化新的`String`对象，因为这将导致`ConcatProxy`：

```cpp
auto c = String{"Marc"} + String{"Chagall"};
// c is a ConcatProxy due to the auto keyword here 
```

不幸的是，我们无法绕过这一点；结果必须显式转换为`String`。

现在是时候看看我们优化版本与正常情况相比有多快了。

## 性能评估

为了评估性能优势，我们将使用以下基准测试，连接并比较大小为`50`的`10,000`个字符串：

```cpp
template <typename T>
auto create_strings(int n, size_t length) -> std::vector<T> {
  // Create n random strings of the specified length
  // ...
}
template <typename T> 
void bm_string_compare(benchmark::State& state) {
  const auto n = 10'000, length = 50;
  const auto a = create_strings<T>(n, length);
  const auto b = create_strings<T>(n, length);
  const auto c = create_strings<T>(n, length * 2);
  for (auto _ : state) {
    for (auto i = 0; i < n; ++i) {
      auto is_equal = a[i] + b[i] == c[i];
      benchmark::DoNotOptimize(is_equal);
    }
  }
}
BENCHMARK_TEMPLATE(bm_string_compare, std::string);
BENCHMARK_TEMPLATE(bm_string_compare, String);
BENCHMARK_MAIN(); 
```

在 Intel Core i7 CPU 上执行时，我使用 gcc 实现了 40 倍的加速。直接使用`std::string`的版本完成时间为 1.6 毫秒，而使用`String`的代理版本仅为 0.04 毫秒。当使用长度为 10 的短字符串进行相同的测试时，加速约为 20 倍。造成这种巨大变化的一个原因是，小字符串将通过利用*第七章* *内存管理*中讨论的小字符串优化来避免堆分配。基准测试告诉我们，当我们摆脱临时字符串和可能伴随其而来的堆分配时，使用代理对象的加速是相当可观的。

`ConcatProxy` 类帮助我们隐藏了在比较字符串时的优化。希望这个简单的例子能激发您开始思考在实现性能优化的同时保持 API 设计清晰的方法。

接下来，您将看到另一个有用的优化，可以隐藏在代理类后面。

# 推迟 sqrt 计算

本节将向您展示如何使用代理对象来推迟或甚至避免在比较二维向量长度时使用计算量大的 `std::sqrt()` 函数。

## 一个简单的二维向量类

让我们从一个简单的二维向量类开始。它有 *x* 和 *y* 坐标，以及一个名为 `length()` 的成员函数，用于计算从原点到位置 *(x, y)* 的距离。我们将这个类称为 `Vec2D`。以下是定义：

```cpp
class Vec2D {
public:
  Vec2D(float x, float y) : x_{x}, y_{y} {}
  auto length() const {
    auto squared = x_*x_ + y_*y_;
    return std::sqrt(squared);
  }
private:
  float x_{};
  float y_{};
}; 
```

以下是客户端如何使用 `Vec2D` 的示例：

```cpp
auto a = Vec2D{3, 4}; 
auto b = Vec2D{4, 4};
auto shortest = a.length() < b.length() ? a : b;
auto length = shortest.length();
std::cout << length; // Prints 5 
```

该示例创建了两个向量并比较它们的长度。然后将最短向量的长度打印到标准输出。*图 10.3* 说明了向量和到原点的计算长度：

![](img/B15619_10_03.png)

图 10.3：两个长度不同的二维向量。向量 a 的长度为 5。

## 底层数学

在计算的数学中，您可能会注意到一些有趣的事情。用于长度的公式如下：

![](img/B15619_10_001.png)

然而，如果我们只需要比较两个向量之间的距离，平方长度就足够了，如下面的公式所示：

![](img/B15619_10_002.png)

平方根可以使用函数 `std::sqrt()` 计算。但是，正如前面提到的，如果我们只想比较两个向量的长度，就不需要进行平方根运算。好处在于 `std::sqrt()` 是一个相对缓慢的操作，这意味着如果我们通过长度比较许多向量，就可以获得一些性能。问题是，我们如何在保持清晰语法的同时实现这一点？让我们看看如何使用代理对象在比较长度时在后台执行这种优化。

为了清晰起见，我们从原始的 `Vec2D` 类开始，但是我们将 `length()` 函数分成两部分 - `length_squared()` 和 `length()`，如下所示：

```cpp
class Vec2D {
public:
  Vec2D(float x, float y) : x_{x}, y_{y} {}  
  auto length_squared() const {
    return x_*x_ + y_*y_;  
  }
  auto length() const {
    return std::sqrt(length_squared());
  }
private:
  float x_{};
  float y_{};
}; 
```

现在，我们 `Vec2D` 类的客户端可以使用 `length_squared()` 来获得一些性能优势，当只比较不同向量的长度时。

假设我们想要实现一个方便的实用函数，返回一系列 `Vec2D` 对象的最小长度。现在我们有两个选择：在进行比较时使用 `length()` 函数或 `length_squared()` 函数。它们对应的实现如下示例所示：

```cpp
// Simple version using length()
auto min_length(const auto& r) -> float {
  assert(!r.empty());
  auto cmp = [](auto&& a, auto&& b) {
    return a.length () < b.length();
  };
  auto it = std::ranges::min_element(r, cmp);
  return it->length();
} 
```

使用 `length_squared()` 进行比较的第二个优化版本将如下所示：

```cpp
// Fast version using length_squared()
auto min_length(const auto& r) -> float {
  assert(!r.empty());
  auto cmp = [](auto&& a, auto&& b) {
    return a.length_squared() < b.length_squared(); // Faster
  };
  auto it = std::ranges::min_element(r, cmp);
  return it->length(); // But remember to use length() here!
} 
```

使用 `cmp` 内部的 `length()` 的第一个版本具有更可读和更容易正确的优势，而第二个版本具有更快的优势。提醒一下，第二个版本的加速是因为我们可以避免在 `cmp` lambda 内部调用 `std::sqrt()`。

最佳解决方案是具有使用 `length()` 语法的第一个版本和使用 `length_squared()` 性能的第二个版本。

根据这个类将被使用的上下文，可能有很好的理由暴露 `length_squared()` 这样的函数。但是让我们假设我们团队中的其他开发人员不理解为什么有 `length_squared()` 函数，并且觉得这个类很混乱。因此，我们决定想出更好的方法，避免有两个暴露向量长度属性的函数版本。正如您可能已经猜到的那样，是时候使用代理类来隐藏这种复杂性了。

为了实现这一点，我们不是从`length()`成员函数中返回一个`float`值，而是返回一个对用户隐藏的中间对象。根据用户如何使用隐藏的代理对象，它应该避免`std::sqrt()`操作，直到真正需要。在接下来的部分中，我们将实现一个名为`LengthProxy`的类，它将是我们从`Vec2D::length()`返回的代理对象的类型。

## 实现 LengthProxy 对象

现在是时候实现`LengthProxy`类了，其中包含一个代表平方长度的`float`数据成员。实际的平方长度永远不会暴露出来，以防止类的用户将平方长度与常规长度混淆。相反，`LengthProxy`有一个隐藏的`friend`函数，用于比较其平方长度和常规长度，如下所示：

```cpp
class LengthProxy { 
public: 
  LengthProxy(float x, float y) : squared_{x * x + y * y} {} 
  bool operator==(const LengthProxy& other) const = default; 
  auto operator<=>(const LengthProxy& other) const = default; 
  friend auto operator<=>(const LengthProxy& proxy, float len) { 
    return proxy.squared_ <=> len*len;   // C++20
  } 
  operator float() const {      // Allow implicit cast to float
    return std::sqrt(squared_); 
  }  
private: 
  float squared_{}; 
}; 
```

我们已经定义了`operator float()`，以允许从`LengthProxy`到`float`的隐式转换。`LengthProxy`对象也可以相互比较。通过使用新的 C++20 比较，我们简单地将等号运算符和三路比较运算符设置为`default`，让编译器为我们生成所有必要的比较运算符。

接下来，我们重写`Vec2D`类，以返回`LengthProxy`类的对象，而不是实际的`float`长度：

```cpp
class Vec2D { 
public: 
  Vec2D(float x, float y) : x_{x}, y_{y} {} 
  auto length() const { 
    return LengthProxy{x_, y_};    // Return proxy object
  } 
  float x_{}; 
  float y_{}; 
}; 
```

有了这些补充，现在是时候使用我们的新代理类了。

## 使用 LengthProxy 比较长度

在这个例子中，我们将比较两个向量`a`和`b`，并确定`a`是否比`b`短。请注意，代码在语法上看起来与我们没有使用代理类时完全相同：

```cpp
auto a = Vec2D{23, 42}; 
auto b = Vec2D{33, 40}; 
bool a_is_shortest = a.length() < b.length(); 
```

在后台，最终语句会扩展为类似于这样的内容：

```cpp
// These LengthProxy objects are never visible from the outside
LengthProxy a_length = a.length(); 
LengthProxy b_length = b.length(); 
// Member operator< on LengthProxy is invoked, 
// which compares member squared_ 
auto a_is_shortest = a_length < b_length; 
```

不错！`std::sqrt()`操作被省略，而`Vec2D`类的接口仍然完整。我们之前实现的`min_length()`的简化版本现在执行比较更有效，因为省略了`std::sqrt()`操作。接下来是简化的实现，现在也变得高效了：

```cpp
// Simple and efficient 
auto min_length(const auto& r) -> float { 
  assert(!r.empty()); 
  auto cmp = [](auto&& a, auto&& b) { 
    return a.length () < b.length(); 
  }; 
  auto it = std::ranges::min_element(r, cmp); 
  return it->length(); 
} 
```

`Vec2D`对象之间的优化长度比较现在是在后台进行的。实现`min_length()`函数的程序员不需要知道这种优化，就能从中受益。让我们看看如果我们需要实际长度会是什么样子。

## 使用 LengthProxy 计算长度

当请求实际长度时，调用代码会有一些变化。为了触发对`float`的隐式转换，我们必须在声明下面的`len`变量时承诺一个`float`；也就是说，我们不能像通常那样只使用`auto`：

```cpp
auto a = Vec2D{23, 42};
float len = a.length(); // Note, we cannot use auto here 
```

如果我们只写`auto`，`len`对象将是`LengthProxy`类型，而不是`float`。我们不希望我们的代码库的用户明确处理`LengthProxy`对象；代理对象应该在暗中运行，只有它们的结果应该被利用（在这种情况下，比较结果或实际距离值是`float`）。尽管我们无法完全隐藏代理对象，让我们看看如何收紧它们以防止误用。

### 防止误用 LengthProxy

您可能已经注意到，使用`LengthProxy`类可能会导致性能变差的情况。在接下来的示例中，根据程序员对长度值的请求，多次调用`std::sqrt()`函数：

```cpp
auto a = Vec2D{23, 42};
auto len = a.length();
float f0 = len;       // Assignment invoked std::sqrt()
float f1 = len;       // std::sqrt() of len is invoked again 
```

尽管这是一个人为的例子，但在现实世界中可能会出现这种情况，我们希望强制`Vec2d`的用户每个`LengthProxy`对象只调用一次`operator float()`。为了防止误用，我们使`operator float()`成员函数只能在 rvalue 上调用；也就是说，只有当`LengthProxy`对象没有绑定到变量时，才能将其转换为浮点数。

我们通过在`operator float()`成员函数上使用`&&`作为修饰符来强制执行此行为。`&&`修饰符的工作原理与`const`修饰符相同，但是`const`修饰符强制成员函数不修改对象，而`&&`修饰符强制函数在临时对象上操作。

修改如下：

```cpp
operator float() const && { return std::sqrt(squared_); } 
```

如果我们在绑定到变量的`LengthProxy`对象上调用`operator float()`，例如以下示例中的`dist`对象，编译器将拒绝编译：

```cpp
auto a = Vec2D{23, 42};
auto len = a.length(); // len is of type LenghtProxy
float f = len;         // Doesn't compile: len is not an rvalue 
```

但是，我们仍然可以直接在从`length()`返回的 rvalue 上调用`operator float()`，就像这样：

```cpp
auto a = Vec2D{23, 42}; 
float f = a.length();    // OK: call operator float() on rvalue 
```

临时的`LengthProxy`实例仍将在后台创建，但由于它没有绑定到变量，因此我们可以将其隐式转换为`float`。这将防止滥用，例如在`LengthProxy`对象上多次调用`operator float()`。

## 性能评估

为了看看我们实际获得了多少性能，让我们来测试一下`min_element()`的以下版本：

```cpp
auto min_length(const auto& r) -> float {
  assert(!r.empty());
  auto it = std::ranges::min_element(r, [](auto&& a, auto&& b) {
    return a.length () < b.length(); });
  return it->length();
} 
```

为了将代理对象优化与其他内容进行比较，我们将定义一个另一版本`Vec2DSlow`，它总是使用`std::sqrt()`计算实际长度：

```cpp
struct Vec2DSlow {
  float length() const {                  // Always compute
    auto squared = x_ * x_ + y_ * y_;     // actual length
    return std::sqrt(squared);            // using sqrt()
  }
  float x_, y_;
}; 
```

使用 Google Benchmark 和函数模板，我们可以看到在查找 1,000 个向量的最小长度时获得了多少性能提升：

```cpp
template <typename T> 
void bm_min_length(benchmark::State& state) {
  auto v = std::vector<T>{};
  std::generate_n(std::back_inserter(v), 1000, [] {
    auto x = static_cast<float>(std::rand());
    auto y = static_cast<float>(std::rand());
    return T{x, y};
  });
  for (auto _ : state) {
    auto res = min_length(v);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK_TEMPLATE(bm_min_length, Vec2DSlow);
BENCHMARK_TEMPLATE(bm_min_length, Vec2D);
BENCHMARK_MAIN(); 
```

在 Intel i7 CPU 上运行此基准测试生成了以下结果：

+   使用未优化的`Vec2DSlow`和`std::sqrt()`花费了 7,900 ns

+   使用`LengthProxy`的`Vec2D`花费了 1,800 ns

这种性能提升相当于超过 4 倍的加速。

这是一个例子，说明了我们如何在某些情况下避免不必要的计算。但是，我们成功地将优化封装在代理对象内部，而不是使`Vec2D`的接口更加复杂，以便所有客户端都能从优化中受益，而不会牺牲清晰度。

C++中用于优化表达式的相关技术是**表达式模板**。这利用模板元编程在编译时生成表达式树。该技术可用于避免临时变量并实现延迟评估。表达式模板是使 Boost **基本线性代数库**（**uBLAS**）和**Eigen**中的线性代数算法和矩阵运算快速的技术之一，[`eigen.tuxfamily.org`](http://eigen.tuxfamily.org)。您可以在 Bjarne Stroustrup 的《C++程序设计语言》第四版中了解有关如何在设计矩阵类时使用表达式模板和融合操作的更多信息。

我们将通过查看与重载运算符结合使用代理对象时的其他受益方式来结束本章。

# 创造性的运算符重载和代理对象

正如您可能已经知道的那样，C++具有重载多个运算符的能力，包括标准数学运算符，例如加号和减号。重载的数学运算符可用于创建自定义数学类，使其行为类似于内置数值类型，以使代码更易读。另一个例子是流运算符，在标准库中重载以将对象转换为流，如下所示：

```cpp
std::cout << "iostream " << "uses " << "overloaded " << "operators."; 
```

然而，一些库在其他上下文中使用重载。如前所述，Ranges 库使用重载来组合视图，如下所示：

```cpp
const auto r = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
auto odd_positive_numbers = r 
  | std::views::filter([](auto v) { return v > 0; }) 
  | std::views::filter([](auto v) { return (v % 2) == 1; }); 
```

接下来，我们将探讨如何在代理类中使用管道运算符。

## 管道运算符作为扩展方法

与其他语言相比，例如 C＃，Swift 和 JavaScript，C++不支持扩展方法；也就是说，您不能在本地使用新的成员函数扩展类。

例如，您不能使用如下所示的`std::vector`扩展`contains(T val)`函数：

```cpp
auto numbers = std::vector{1, 2, 3, 4};
auto has_two = numbers.contains(2); 
```

但是，您可以重载管道运算符以实现这种几乎等效的语法：

```cpp
auto has_two = numbers | contains(2); 
```

通过使用代理类，可以轻松实现这一点。

### 管道运算符

我们的目标是实现一个简单的管道操作符，以便我们可以编写以下内容：

```cpp
auto numbers = std::vector{1, 3, 5, 7, 9}; 
auto seven = 7; 
bool has_seven = numbers | contains(seven); 
```

使用可管道化语法的`contains()`函数有两个参数：`numbers`和`seven`。由于左参数`numbers`可以是任何东西，我们需要重载以在右侧包含一些唯一的东西。因此，我们创建了一个名为`ContainsProxy`的`struct`模板，它保存右侧的参数；这样，重载的管道操作符可以识别重载：

```cpp
template <typename T>
struct ContainsProxy { const T& value_; };
template <typename Range, typename T>
auto operator|(const Range& r, const ContainsProxy<T>& proxy) {
  const auto& v = proxy.value_;
  return std::find(r.begin(), r.end(), v) != r.end();
} 
```

现在我们可以像这样使用`ContainsProxy`：

```cpp
auto numbers = std::vector{1, 3, 5, 7, 9}; 
auto seven = 7; 
auto proxy = ContainsProxy<decltype(seven)>{seven};  
bool has_seven = numbers | proxy; 
```

管道操作符有效，尽管语法仍然很丑陋，因为我们需要指定类型。为了使语法更整洁，我们可以简单地创建一个方便的函数，它接受该值并创建一个包含类型的代理：

```cpp
template <typename T>
auto contains(const T& v) { return ContainsProxy<T>{v}; } 
```

这就是我们需要的全部；现在我们可以将其用于任何类型或容器：

```cpp
auto penguins = std::vector<std::string>{"Ping","Roy","Silo"};
bool has_silo = penguins | contains("Silo"); 
```

本节涵盖的示例展示了实现管道操作符的一种基本方法。例如，Paul Fultz 的 Ranges 库和 Fit 库（可在[`github.com/pfultz2/Fit`](https://github.com/pfultz2/Fit)找到）实现了适配器，它们接受常规函数并赋予其使用管道语法的能力。

# 总结

在本章中，您学会了惰性求值和急性求值之间的区别。您还学会了如何使用隐藏的代理对象在幕后实现惰性求值，这意味着您现在了解如何在保留类的易于使用的接口的同时实现惰性求值优化。将复杂的优化隐藏在库类中，而不是在应用程序代码中暴露它们，可以使应用程序代码更易读，更少出错。

在下一章中，我们将转移重点，转向使用 C++进行并发和并行编程。
