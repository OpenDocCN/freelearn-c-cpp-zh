# 9

# 基本实用程序

本章将介绍 C++**实用库**中的一些基本类。在处理包含不同类型元素的集合时，将使用前一章介绍的一些元编程技术以便有效地工作。

C++容器是同类的，意味着它们只能存储单一类型的元素。`std::vector<int>`存储一组整数，`std::list<Boat>`中存储的所有对象都是`Boat`类型。但有时，我们需要跟踪不同类型的元素集合。我将这些集合称为**异类集合**。在异类集合中，元素可能具有不同的类型。下图显示了一个整数的同类集合的示例和一个具有不同类型元素的异类集合的示例：

![](img/B15619_09_01.png)

图 9.1：同类和异类集合

本章将涵盖 C++实用库中一组有用的模板，这些模板可用于存储各种类型的多个值。本章分为四个部分：

+   使用`std::optional`表示可选值

+   使用`std::pair`、`std::tuple`和`std::tie()`来固定大小的集合

+   使用标准容器存储具有`std::any`和`std::variant`类型的元素的动态大小集合

+   一些真实世界的例子展示了`std::tuple`和`std::tie()`的有用性，以及我们在*第八章*中涵盖的元编程概念

让我们首先探索`std::optional`及其一些重要的用例。

# 使用 std::optional 表示可选值

尽管在 C++17 中是一个相当次要的特性，`std::optional`是标准库的一个不错的补充。它简化了一个以前无法以清晰和直接的方式表达的常见情况。简而言之，它是任何类型的一个小包装器，其中包装的类型可以是*初始化*或*未初始化*。

用 C++术语来说，`std::optional`是一个*最大大小为一的栈分配容器*。

## 可选返回值

在引入`std::optional`之前，没有明确的方法来定义可能不返回定义值的函数，例如两条线段的交点。引入`std::optional`后，这样的可选返回值可以得到清晰的表达。接下来是一个返回两条线之间的可选交点的函数的实现：

```cpp
// Prerequisite
struct Point { /* ... */ }; 
struct Line { /* ... */ };  
auto lines_are_parallel(Line a, Line b) -> bool { /* ... */ }
auto compute_intersection(Line a, Line b) -> Point { /* ... */ }
auto get_intersection(const Line& a, const Line& b) 
  -> std::optional<Point> 
{
  if (lines_are_parallel(a, b))
    return std::optional{compute_intersection(a, b)};
  else
    return {};
} 
```

`std::optional`的语法类似于指针；值通过`operator*()`或`operator->()`访问。尝试使用`operator*()`或`operator->()`访问空的可选值的值是未定义行为。还可以使用`value()`成员函数访问值，如果可选值不包含值，则会抛出`std::bad_optional_access`异常。接下来是一个返回的简单`std::optional`的示例：

```cpp
auto set_magic_point(Point p) { /* ... */ }
auto intersection = get_intersection(line0, line1);
if (intersection.has_value()) {
  set_magic_point(*intersection);
} 
```

`std::optional`持有的对象始终是栈分配的，将类型包装到`std::optional`的内存开销是一个布尔值的大小（通常为一个字节），加上可能的填充。

## 可选成员变量

假设我们有一个表示人头的类。头部可以戴一顶帽子，也可以不戴帽子。通过使用`std::optional`来表示帽子成员变量，实现就可以尽可能地表达出来：

```cpp
struct Hat { /* ... */ };
class Head {
public:
  Head() { assert(!hat_); }      // hat_ is empty by default
  auto set_hat(const Hat& h) { 
    hat_ = h; 
  }
  auto has_hat() const { 
    return hat_.has_value(); 
  }
  auto& get_hat() const { 
    assert(hat_.has_value()); 
    return *hat_; 
  }
  auto remove_hat() { 
    hat_ = {};        // Hat is cleared by assigning to {}
  } 
private:
  std::optional<Hat> hat_;
}; 
```

如果没有`std::optional`，表示可选成员变量将依赖于例如指针或额外的`bool`成员变量。两者都有缺点，例如在堆上分配，或者在没有警告的情况下意外访问被认为是空的可选。

## 避免枚举中的空状态

在旧的 C++代码库中可以看到的一个模式是`enum`中的*空状态*或*空状态*。这是一个例子：

```cpp
enum class Color { red, blue, none };  // Don't do this! 
```

在前面的`enum`中，`none`是所谓的空状态。在`Color`的`enum`中添加`none`值的原因是为了表示可选颜色，例如：

```cpp
auto get_color() -> Color; // Returns an optional color 
```

然而，使用这种设计，没有办法表示非可选颜色，这使得*所有*代码都必须处理额外的空状态`none`。

更好的替代方案是避免额外的空状态，而是用类型`std::optional<Color>`表示可选颜色：

```cpp
enum class Color { red, blue };
auto get_color() -> std::optional<Color>; 
```

这清楚地表明我们可能无法得到一个颜色。但我们也知道一旦有了`Color`对象，它就不可能为空：

```cpp
auto set_color(Color c) { /* c is a valid color, now use it ... */ } 
```

在实现`set_color()`时，我们知道客户端传递了有效的颜色。

## 排序和比较 std::optional

`std::optional`同样可以使用下表中显示的规则进行比较和排序：

| 两个*空*可选值被认为是相等的。 | 空的可选值被认为*小于*非空的可选值。 |
| --- | --- |

|

```cpp
auto a = std::optional<int>{};
auto b = std::optional<int>{};
auto c = std::optional<int>{4};
assert(a == b);
assert(b != c); 
```

|

```cpp
auto a = std::optional<int>{};
auto b = std::optional<int>{4};
auto c = std::optional<int>{5};
assert(a < b);
assert(b < c); 
```

|

因此，如果对`std::optional<T>`的容器进行排序，空的可选值将出现在容器的开头，而非空的可选值将像通常一样排序，如下所示：

```cpp
auto c = std::vector<std::optional<int>>{{3}, {}, {1}, {}, {2}};
std::sort(c.begin(), c.end());
// c is {}, {}, {1}, {2}, {3} 
```

如果您习惯使用指针表示可选值，设计使用输出参数的 API，或在枚举中添加特殊的空状态，那么现在是时候将`std::optional`添加到您的工具箱中了，因为它提供了这些反模式的高效且安全的替代方案。

让我们继续探讨可以容纳不同类型元素的固定大小集合。

# 固定大小异构集合

C++实用库包括两个可以用于存储不同类型的多个值的类模板：`std::pair`和`std::tuple`。它们都是固定大小的集合。就像`std::array`一样，在运行时动态添加更多值是不可能的。

`std::pair`和`std::tuple`之间的主要区别在于`std::pair`只能容纳两个值，而`std::tuple`可以在编译时用任意大小进行实例化。我们将从简要介绍`std::pair`开始，然后转向`std::tuple`。

## 使用 std::pair

类模板`std::pair`位于`<utility>`头文件中，并且自从标准模板库引入以来就一直可用于 C++。它在标准库中用于算法需要返回两个值的情况，比如`std::minmax()`，它可以返回初始化列表的最小值和最大值：

```cpp
std::pair<int, int> v = std::minmax({4, 3, 2, 4, 5, 1});
std::cout << v.first << " " << v.second;     // Outputs: "1 5" 
```

前面的例子显示了可以通过成员`first`和`second`访问`std::pair`的元素。

在这里，`std::pair`保存相同类型的值，因此也可以在这里返回一个数组。但是`std::pair`更有趣的地方在于它可以保存*不同*类型的值。这就是为什么我们认为这是一个异构集合的原因，尽管它只能容纳两个值。

标准库中`std::pair`保存不同值的一个例子是关联容器`std::map`。`std::map`的值类型是一个由键和与键关联的元素组成的对：

```cpp
auto scores = std::map<std::string, int>{};
scores.insert(std::pair{"Neo", 12}); // Correct but ineffecient
scores.emplace("Tri", 45);           // Use emplace() instead
scores.emplace("Ari", 33);
for (auto&& it : scores) { // "it" is a std::pair
  auto key = it.first;
  auto val = it.second;
  std::cout << key << ": " << val << '\n';
} 
```

显式命名`std::pair`类型的要求已经减少，在现代 C++中，使用初始化列表和结构化绑定来隐藏我们正在处理`std::pair`值的事实是很常见的。下面的例子表达了相同的事情，但没有明确提到底层的`std::pair`：

```cpp
auto scores = std::map<std::string, int> {
  {"Neo", 12},                            // Initializer lists
  {"Tri", 45},
  {"Ari", 33}
};
for (auto&& [key, val] : scores) {       // Structured bindings
  std::cout << key << ": " << val << '\n';
} 
```

我们将在本章后面更多地讨论结构化绑定。

正如其名称所示，`std::pair`只能容纳两个值。C++11 引入了一个名为`std::tuple`的新实用类，它是`std::pair`的泛化，可以容纳任意数量的元素。

## std::tuple

`std::tuple`可以用作固定大小的异构集合，可以声明为任意大小。与`std::vector`相比，它的大小在运行时不能改变；您不能添加或删除元素。

元组可以这样构造，其成员类型明确指定：

```cpp
auto t = std::tuple<int, std::string, bool>{}; 
```

或者，我们可以使用类模板参数推导进行初始化，如下所示：

```cpp
auto t = std::tuple{0, std::string{}, false}; 
```

这将使编译器生成一个类，大致可以看作是这样的：

```cpp
struct Tuple {
  int data0_{};
  std::string data1_{};
  bool data2_{};
}; 
```

与 C++标准库中的许多其他类一样，`std::tuple`也有一个对应的`std::make_tuple()`函数，它可以从参数中自动推断类型：

```cpp
auto t = std::make_tuple(42, std::string{"hi"}, true); 
```

但正如前面所述，从 C++17 开始，许多这些`std::make_`函数都是多余的，因为 C++17 类可以从构造函数中推断出这些类型。

### 访问元组的成员

可以使用自由函数模板`std::get<Index>()`访问`std::tuple`的各个元素。你可能会想为什么不能像常规容器一样使用`at(size_t index)`成员函数访问成员。原因是`at()`这样的成员函数只允许返回一个类型，而元组在不同索引处包含不同类型。相反，使用带有索引的函数模板`std::get()`作为模板参数：

```cpp
auto a = std::get<0>(t);     // int
auto b = std::get<1>(t);     // std::string
auto c = std::get<2>(t);     // bool 
```

我们可以想象`std::get()`函数的实现类似于这样：

```cpp
template <size_t Index, typename Tuple>
auto& get(const Tuple& t) {
  if constexpr(Index == 0) {
    return t.data0_;
  } else if constexpr(Index == 1) {
    return t.data1_;
  } else if constexpr(Index == 2) {
    return t.data2_;
  }
} 
```

这意味着当我们创建和访问元组时：

```cpp
auto t = std::tuple(42, true);
auto v = std::get<0>(t); 
```

编译器大致生成以下代码：

```cpp
// The Tuple class is generated first:
class Tuple { 
  int data0_{};
  bool data1_{};
public:
  Tuple(int v0, bool v1) : data0_{v0}, data1_{v1} {} 
};
// get<0>(Tuple) is then generated to something like this:
auto& get(const Tuple& tpl) { return data0_; }

// The generated function is then utilized:
auto t = Tuple(42, true); 
auto v = get(t); 
```

请注意，这个例子只能被认为是一种简单的想象，用来想象编译器在构造`std::tuple`时生成的内容；`std::tuple`的内部非常复杂。然而，重要的是要理解，`std::tuple`类基本上是一个简单的结构，其成员可以通过编译时索引访问。

`std::get()`函数模板也可以使用 typename 作为参数。它的使用方式如下：

```cpp
auto number = std::get<int>(tuple);
auto str = std::get<std::string>(tuple); 
```

只有当指定的类型在元组中包含一次时才可能。

### 迭代 std::tuple 成员

从程序员的角度来看，似乎`std::tuple`可以像任何其他容器一样使用常规的基于范围的`for`循环进行迭代，如下所示：

```cpp
auto t = std::tuple(1, true, std::string{"Jedi"});
for (const auto& v : t) {
  std::cout << v << " ";
} 
```

这不可能的原因是`const auto& v`的类型只被评估一次，而由于`std::tuple`包含不同类型的元素，这段代码根本无法编译。

对于常规算法也是一样，因为迭代器不会改变指向的类型；因此，`std::tuple`不提供`begin()`或`end()`成员函数，也不提供用于访问值的下标运算符`[]`。因此，我们需要想出其他方法来展开元组。

### 展开元组

由于元组不能像通常那样进行迭代，我们需要使用元编程来展开循环。从前面的例子中，我们希望编译器生成类似于这样的东西：

```cpp
auto t = std::tuple(1, true, std::string{"Jedi"});
std::cout << std::get<0>(t) << " ";
std::cout << std::get<1>(t) << " ";
std::cout << std::get<2>(t) << " ";
// Prints "1 true Jedi" 
```

如你所见，我们迭代元组的每个索引，这意味着我们需要知道元组中包含的类型/值的数量。然后，由于元组包含不同类型，我们需要编写一个生成元组中每种类型的新函数的元函数。

如果我们从一个为特定索引生成调用的函数开始，它会看起来像这样：

```cpp
template <size_t Index, typename Tuple, typename Func> 
void tuple_at(const Tuple& t, Func f) {
  const auto& v = std::get<Index>(t);
  std::invoke(f, v);
} 
```

然后我们可以将其与通用 lambda 结合使用，就像你在*第二章* *Essential C++ Techniques*中学到的那样：

```cpp
auto t = std::tuple{1, true, std::string{"Jedi"}};
auto f = [](const auto& v) { std::cout << v << " "; };
tuple_at<0>(t, f);
tuple_at<1>(t, f);
tuple_at<2>(t, f);
// Prints "1 true Jedi" 
```

有了`tuple_at()`函数，我们就可以继续进行实际的迭代。我们首先需要的是元组中值的数量作为编译时常量。幸运的是，这个值可以通过类型特征`std::tuple_size_v<Tuple>`获得。使用`if constexpr`，我们可以通过创建一个类似的函数来展开迭代，根据索引采取不同的操作：

+   如果索引等于元组大小，它会生成一个空函数

+   否则，它会在传递的索引处执行 lambda，并生成一个索引增加 1 的新函数

代码将如下所示：

```cpp
template <typename Tuple, typename Func, size_t Index = 0> void tuple_for_each(const Tuple& t, const Func& f) {
  constexpr auto n = std::tuple_size_v<Tuple>;
  if constexpr(Index < n) {
    tuple_at<Index>(t, f);
    tuple_for_each<Tuple, Func, Index+1>(t, f);
  }
} 
```

如你所见，默认索引设置为零，这样在迭代时就不必指定它。然后可以像这样调用`tuple_for_each()`函数，直接放在 lambda 的位置：

```cpp
auto t = std::tuple{1, true, std::string{"Jedi"}};
tuple_for_each(t, [](const auto& v) { std::cout << v << " "; });
// Prints "1 true Jedi" 
```

相当不错；从语法上看，它看起来与`std::for_each()`算法非常相似。

#### 为元组实现其他算法

在`tuple_for_each()`的基础上，可以以类似的方式实现迭代元组的不同算法。以下是`std::any_of()`为元组实现的示例：

```cpp
template <typename Tuple, typename Func, size_t Index = 0> 
auto tuple_any_of(const Tuple& t, const Func& f) -> bool { 
  constexpr auto n = std::tuple_size_v<Tuple>; 
  if constexpr(Index < n) { 
    bool success = std::invoke(f, std::get<Index>(t)); 
    if (success) {
      return true;
    }
    return tuple_any_of<Tuple, Func, Index+1>(t, f); 
  } else { 
    return false; 
  } 
} 
```

它可以这样使用：

```cpp
auto t = std::tuple{42, 43.0f, 44.0}; 
auto has_44 = tuple_any_of(t, [](auto v) { return v == 44; }); 
```

函数模板`tuple_any_of()`遍历元组中的每种类型，并为当前索引处的元素生成一个 lambda 函数，然后将其与`44`进行比较。在这种情况下，`has_44`将评估为`true`，因为最后一个元素，即`double`值，是`44`。如果我们添加一个与`44`不可比较的类型的元素，比如`std::string`，我们将得到一个编译错误。

### 访问元组元素

在 C++17 之前，有两种标准方法可以访问`std::tuple`的元素：

+   为了访问单个元素，使用了函数`std::get<N>(tuple)`。

+   为了访问多个元素，使用了函数`std::tie()`。

尽管它们都起作用，但执行这样一个简单任务的语法非常冗长，如下例所示：

```cpp
// Prerequisite 
using namespace std::string_literals;  // "..."s
auto make_saturn() { return std::tuple{"Saturn"s, 82, true}; }
int main() {
  // Using std::get<N>()
  {
    auto t = make_saturn();
    auto name = std::get<0>(t);
    auto n_moons = std::get<1>(t);
    auto rings = std::get<2>(t);
    std::cout << name << ' ' << n_moons << ' ' << rings << '\n';
    // Output: Saturn 82 true   }
    // Using std::tie()
  {
    auto name = std::string{};
    auto n_moons = int{};
    auto rings = bool{};
    std::tie(name, n_moons, rings) = make_saturn();
    std::cout << name << ' ' << n_moons << ' ' << rings << '\n';
  }
} 
```

为了能够优雅地执行这个常见任务，C++17 引入了结构化绑定。

#### 结构化绑定

使用结构化绑定，可以使用`auto`和括号声明列表一次初始化多个变量。与一般情况下的`auto`关键字一样，可以通过使用相应的修饰符来控制变量是否应该是可变引用、前向引用、const 引用或值。在下面的示例中，正在构造`const`引用的结构化绑定：

```cpp
const auto& [name, n_moons, rings] = make_saturn();
std::cout << name << ' ' << n_moons << ' ' << rings << '\n'; 
```

结构化绑定也可以用于在`for`循环中提取元组的各个成员，如下所示：

```cpp
auto planets = { 
  std::tuple{"Mars"s, 2, false}, 
  std::tuple{"Neptune"s, 14, true} 
};
for (auto&& [name, n_moons, rings] : planets) { 
   std::cout << name << ' ' << n_moons << ' ' << rings << '\n'; 
} 
// Output:
// Mars 2 false 
// Neptune 14 true 
```

这里有一个快速提示。如果你想要返回具有命名变量的多个参数，而不是元组索引，可以在函数内部定义一个结构体并使用自动返回类型推导：

```cpp
auto make_earth() {
  struct Planet { std::string name; int n_moons; bool rings; };
  return Planet{"Earth", 1, false}; 
}
// ...
auto p = make_earth(); 
std::cout << p.name << ' ' << p.n_moons << ' ' << p.rings << '\n'; 
```

结构化绑定也适用于结构体，因此，我们可以直接捕获各个数据成员，如下所示，即使它是一个结构体：

```cpp
auto [name, num_moons, has_rings] = make_earth(); 
```

在这种情况下，我们可以选择任意名称作为标识符，因为`Planet`的数据成员的顺序是相关的，就像返回元组时一样。

现在，我们将看看在处理任意数量的函数参数时，`std::tuple`和`std::tie()`的另一个用例。

### 可变模板参数包

**可变模板参数包**使程序员能够创建可以接受任意数量参数的模板函数。

#### 具有可变数量参数的函数示例

如果我们要创建一个将任意数量的参数转换为字符串的函数，而不使用可变模板参数包，我们需要使用 C 风格的可变参数（就像`printf()`一样）或为每个参数数量创建一个单独的函数：

```cpp
auto make_string(const auto& v0) { 
  auto ss = std::ostringstream{}; 
  ss << v0; 
  return ss.str(); 
} 
auto make_string(const auto& v0, const auto& v1) { 
   return make_string(v0) + " " + make_string(v1); 
}
auto make_string(const auto& v0, const auto& v1, const auto& v2) { 
  return make_string(v0, v1) + " " + make_string(v2); 
} 
// ... and so on for as many parameters we might need 
```

这是我们函数的预期用法：

```cpp
auto str0 = make_string(42);
auto str1 = make_string(42, "hi");
auto str2 = make_string(42, "hi", true); 
```

如果我们需要大量的参数，这变得很繁琐，但是使用参数包，我们可以将其实现为一个接受任意数量参数的函数。

#### 如何构造可变参数包

参数包通过在类型名称前面放置三个点和在可变参数后面放置三个点来识别，用逗号分隔扩展包：

```cpp
template<typename ...Ts> 
auto f(Ts... values) {
  g(values...);
} 
```

这是个语法解释：

+   `Ts`是类型列表

+   `<typename ...Ts>`表示函数处理一个列表

+   `values...`扩展包，使得每个值之间都添加了逗号。

将其转化为代码，考虑这个`expand_pack()`函数模板：

```cpp
template <typename ...Ts>
auto expand_pack(const Ts& ...values) {
   auto tuple = std::tie(values...);
} 
```

让我们这样调用前面的函数：

```cpp
expand_pack(42, std::string{"hi"}); 
```

在这种情况下，编译器将生成一个类似于这样的函数：

```cpp
auto expand_pack(const int& v0, const std::string& v1) {
  auto tuple = std::tie(v0, v1);
} 
```

这是各个参数包部分扩展到的内容：

| 表达式： | 扩展为： |
| --- | --- |
| `template <typename... Ts>` | `template <typename T0, typename T1>` |
| `expand_pack(const Ts& ...values)` | `expand_pack(const T0& v0, const T1& v1)` |
| `std::tie(values...)` | `std::tie(v0, v1)` |

表 9.1：扩展表达式

现在，让我们看看如何创建一个带有可变参数包的`make_string()`函数。

进一步扩展初始的`make_string()`函数，为了从每个参数创建一个字符串，我们需要迭代参数包。没有直接迭代参数包的方法，但一个简单的解决方法是将其转换为元组，然后使用`tuple_for_each()`函数模板进行迭代，如下所示：

```cpp
template <typename ...Ts> 
auto make_string(const Ts& ...values) { 
  auto ss = std::ostringstream{}; 
  // Create a tuple of the variadic parameter pack 
  auto tuple = std::tie(values...); 
  // Iterate the tuple 
  tuple_for_each(tuple, &ss { ss << v; }); 
  return ss.str();
}
// ...
auto str = make_string("C++", 20);  // OK: str is "C++" 
```

参数包被转换为`std::tuple`，然后使用`tuple_for_each()`进行迭代。回顾一下，我们需要使用`std::tuple`来处理参数的原因是因为我们希望支持各种类型的任意数量的参数。如果我们只需要支持特定类型的参数，我们可以使用带有范围`for`循环的`std::array`，如下所示：

```cpp
template <typename ...Ts>
auto make_string(const Ts& ...values) {
  auto ss = std::ostringstream{};
  auto a = std::array{values...};     // Only supports one type
  for (auto&& v : a) { ss << v; }
  return ss.str();
}
// ...
auto a = make_string("A", "B", "C");  // OK: Only one type
auto b = make_string(100, 200, 300);  // OK: Only one type
auto c = make_string("C++", 20);      // Error: Mixed types 
```

正如您所见，`std::tuple`是一个具有固定大小和固定元素位置的异构集合，更或多或少类似于常规结构，但没有命名的成员变量。

我们如何扩展这个以创建一个动态大小的集合（例如`std::vector`和`std::list`），但具有存储混合类型元素的能力？我们将在下一节中看到这个问题的解决方案。

# 动态大小的异构集合

我们在本章开始时指出，C++提供的动态大小容器是同质的，这意味着我们只能存储单一类型的元素。但有时，我们需要跟踪一个大小动态的集合，其中包含不同类型的元素。为了能够做到这一点，我们将使用包含`std::any`或`std::variant`类型元素的容器。

最简单的解决方案是使用`std::any`作为基本类型。`std::any`对象可以存储其中的任何类型的值：

```cpp
auto container = std::vector<std::any>{42, "hi", true}; 
```

然而，它也有一些缺点。首先，每次访问其中的值时，必须在运行时测试类型。换句话说，我们在编译时完全失去了存储值的类型信息。相反，我们必须依赖运行时类型检查来获取信息。其次，它在堆上分配对象而不是栈上，这可能会对性能产生重大影响。

如果我们想要迭代我们的容器，我们需要明确告诉每个`std::any`对象：*如果你是一个 int，就这样做，如果你是一个 char 指针，就那样做*。这是不可取的，因为它需要重复的源代码，并且比使用其他替代方案效率低，我们将在本章后面介绍。

以下示例已编译；类型已明确测试并转换：

```cpp
for (const auto& a : container) {
  if (a.type() == typeid(int)) {
    const auto& value = std::any_cast<int>(a);
    std::cout << value;
  }
  else if (a.type() == typeid(const char*)) {
    const auto& value = std::any_cast<const char*>(a);
    std::cout << value;
  }
  else if (a.type() == typeid(bool)) {
    const auto& value = std::any_cast<bool>(a);
    std::cout << value;
  }
} 
```

我们无法使用常规流操作符打印它，因为`std::any`对象不知道如何访问其存储的值。因此，以下代码不会编译；编译器不知道`std::any`中存储了什么：

```cpp
for (const auto& a : container) { 
  std::cout << a;                // Does not compile
} 
```

通常我们不需要`std::any`提供的类型的完全灵活性，在许多情况下，我们最好使用`std::variant`，接下来我们将介绍。

## std::variant

如果我们不需要在容器中存储*任何*类型，而是想要集中在容器初始化时声明的固定类型集合上，那么`std::variant`是更好的选择。

`std::variant`相对于`std::any`有两个主要优势：

+   它不会将其包含的类型存储在堆上（不像`std::any`）

+   它可以通过通用 lambda 调用，这意味着您不必明确知道其当前包含的类型（本章后面将更多介绍）

`std::variant`的工作方式与元组有些类似，只是它一次只存储一个对象。包含的类型和值是我们最后分配的类型和值。以下图示了在使用相同类型实例化`std::tuple`和`std::variant`时它们之间的区别：

![](img/B15619_09_02.png)

图 9.2：类型元组与类型变体

以下是使用`std::variant`的示例：

```cpp
using VariantType = std::variant<int, std::string, bool>; 
VariantType v{}; 
std::holds_alternative<int>(v);  // true, int is first alternative
v = 7; 
std::holds_alternative<int>(v);  // true
v = std::string{"Anne"};
std::holds_alternative<int>(v);  // false, int was overwritten 
v = false; 
std::holds_alternative<bool>(v); // true, v is now bool 
```

我们使用`std::holds_alternative<T>()`来检查变体当前是否持有给定类型。您可以看到，当我们为变体分配新值时，类型会发生变化。

除了存储实际值外，`std::variant`还通过使用通常为`std::size_t`大小的索引来跟踪当前持有的备用。这意味着`std::variant`的总大小通常是最大备用的大小加上索引的大小。我们可以通过使用`sizeof`运算符来验证我们的类型：

```cpp
std::cout << "VariantType: "<< sizeof(VariantType) << '\n';
std::cout << "std::string: "<< sizeof(std::string) << '\n';
std::cout << "std::size_t: "<< sizeof(std::size_t) << '\n'; 
```

使用带有 libc++的 Clang 10.0 编译和运行此代码将生成以下输出：

```cpp
VariantType: 32
std::string: 24
std::size_t: 8 
```

如您所见，`VariantType`的大小是`std::string`和`std::size_t`的总和。

### std::variant 的异常安全性

当向`std::variant`对象分配新值时，它被放置在变体当前持有值的相同位置。如果由于某种原因，新值的构造或分配失败并引发异常，则可能不会恢复旧值。相反，变体可以变为**无值**。您可以使用成员函数`valueless_by_exception()`来检查变体对象是否无值。这可以在尝试使用`emplace()`成员函数构造对象时进行演示：

```cpp
struct Widget {
  explicit Widget(int) {    // Throwing constructor
    throw std::exception{};
  }
};
auto var = std::variant<double, Widget>{1.0};
try {
  var.emplace<1>(42); // Try to construct a Widget instance
} catch (...) {
  std::cout << "exception caught\n";
  if (var.valueless_by_exception()) {  // var may or may not 
    std::cout << "valueless\n";        // be valueless
  } else {
    std::cout << std::get<0>(var) << '\n';
  }
} 
```

在异常被抛出并捕获后，初始的`double`值 1.0 可能存在，也可能不存在。操作不能保证回滚，这通常是我们可以从标准库容器中期望的。换句话说，`std::variant`不提供强异常安全性保证的原因是性能开销，因为这将要求`std::variant`使用堆分配。`std::variant`的这种行为是一个有用的特性，而不是一个缺点，因为这意味着您可以在具有实时要求的代码中安全地使用`std::variant`。

如果您希望使用堆分配版本，但具有强异常安全性保证和“永不为空”的保证，`boost::variant`提供了这种功能。如果您对实现这种类型的挑战感兴趣，[`www.boost.org/doc/libs/1_74_0/doc/html/variant/design.html`](https://www.boost.org/doc/libs/1_74_0/doc/html/variant/design.html)提供了一个有趣的阅读。

### 访问变体

访问`std::variant`中的变量时，我们使用全局函数`std::visit()`。正如你可能已经猜到的那样，当处理异构类型时，我们必须使用我们的主要伴侣：通用 lambda：

```cpp
auto var = std::variant<int, bool, float>{};
std::visit([](auto&& val) { std::cout << val; }, var); 
```

在示例中使用通用 lambda 和变体`var`调用`std::visit()`时，编译器会将 lambda 概念上转换为一个常规类，该类对变体中的每种类型进行`operator()`重载。这将看起来类似于这样：

```cpp
struct GeneratedFunctorImpl {
  auto operator()(int&& v)   { std::cout << v; }
  auto operator()(bool&& v)  { std::cout << v; }
  auto operator()(float&& v) { std::cout << v; }
}; 
```

然后，`std::visit()`函数扩展为使用`std::holds_alternative<T>()`的`if...else`链，或使用`std::variant`的索引生成正确的调用`std::get<T>()`的跳转表。

在前面的示例中，我们直接将通用 lambda 中的值传递给`std::cout`，而不考虑当前持有的备用。但是，如果我们想要根据正在访问的类型执行不同的操作怎么办？在这种情况下可能使用的一种模式是定义一个可变类模板，该模板将继承一组 lambda。然后，我们需要为要访问的每种类型定义这个。听起来有点复杂，不是吗？这一开始可能看起来有点神奇，也考验了我们的元编程技能，但是一旦我们有了可变类模板，使用起来就很容易了。

我们将从可变类模板开始。以下是它的外观：

```cpp
template<class... Lambdas>
struct Overloaded : Lambdas... {
  using Lambdas::operator()...;
}; 
```

如果您使用的是 C++17 编译器，还需要添加一个显式的推导指南，但在 C++20 中不需要：

```cpp
template<class... Lambdas> 
Overloaded(Lambdas...) -> Overloaded<Lambdas...>; 
```

就是这样。模板类`Overloaded`将继承我们将使用模板实例化的所有 lambda，并且函数调用运算符`operator()()`将被每个 lambda 重载一次。现在可以创建一个只包含调用运算符的多个重载的无状态对象：

```cpp
auto overloaded_lambdas = Overloaded{
  [](int v)   { std::cout << "Int: " << v; },
  [](bool v)  { std::cout << "Bool: " << v; },
  [](float v) { std::cout << "Float: " << v; }
}; 
```

我们可以使用不同的参数进行测试，并验证是否调用了正确的重载：

```cpp
overloaded_lambdas(30031);    // Prints "Int: 30031"
overloaded_lambdas(2.71828f); // Prints "Float: 2.71828" 
```

现在，我们可以在不需要将`Overloaded`对象存储在左值中的情况下使用`std::visit()`。最终的效果如下：

```cpp
auto var = std::variant<int, bool, float>{42};
std::visit(Overloaded{
  [](int v)   { std::cout << "Int: " << v; },
  [](bool v)  { std::cout << "Bool: " << v; },
  [](float v) { std::cout << "Float: " << v; }
}, var);
// Outputs: "Int: 42" 
```

因此，一旦我们有了`Overloaded`模板，我们就可以使用这种方便的方式来指定一组不同类型参数的 lambda。在下一节中，我们将开始使用`std::variant`和标准容器。

## 使用变体的异构集合

现在我们有了一个可以存储所提供列表中任何类型的变体，我们可以将其扩展为异构集合。我们只需创建一个我们的变体的`std::vector`：

```cpp
using VariantType = std::variant<int, std::string, bool>;
auto container = std::vector<VariantType>{}; 
```

现在，我们可以向向量中推送不同类型的元素：

```cpp
container.push_back(false);
container.push_back("I am a string"s);
container.push_back("I am also a string"s);
container.push_back(13); 
```

现在，向内存中的向量看起来是这样的，其中向量中的每个元素都包含变体的大小，本例中为`sizeof(std::size_t) + sizeof(std::string)`：

![](img/B15619_09_03.png)

图 9.3：变体的向量

当然，我们也可以使用`pop_back()`或以容器允许的任何其他方式修改容器：

```cpp
container.pop_back();
std::reverse(container.begin(), container.end());
// etc... 
```

## 访问我们的变体容器中的值

现在我们有了一个大小动态的异构集合的样板，让我们看看如何像常规的`std::vector`一样使用它：

1.  **构造异构变体容器**：在这里，我们构造了一个包含不同类型的`std::vector`。请注意，初始化列表包含不同的类型：

```cpp
using VariantType = std::variant<int, std::string, bool>;
auto v = std::vector<VariantType>{ 42, "needle"s, true }; 
```

1.  **使用常规 for 循环迭代打印内容**：要使用常规`for`循环迭代容器，我们利用`std::visit()`和一个通用 lambda。全局函数`std::visit()`负责类型转换。该示例将每个值打印到`std::cout`，而不考虑类型：

```cpp
for (const auto& item : v) { 
  std::visit([](const auto& x) { std::cout << x << '\n';}, item);
} 
```

1.  **检查容器中的类型**：在这里，我们通过类型检查容器的每个元素。这是通过使用全局函数`std::holds_alternative<type>`实现的，该函数在变体当前持有所要求的类型时返回`true`。以下示例计算当前容器中包含的布尔值的数量：

```cpp
auto num_bools = std::count_if(v.begin(), v.end(),
                               [](auto&& item) {
  return std::holds_alternative<bool>(item);
}); 
```

1.  **通过包含的类型和值查找内容**：在此示例中，我们通过结合`std::holds_alternative()`和`std::get()`来检查容器的类型和值。此示例检查容器是否包含值为`"needle"`的`std::string`：

```cpp
auto contains = std::any_of(v.begin(), v.end(),
                            [](auto&& item) {
  return std::holds_alternative<std::string>(item) &&
    std::get<std::string>(item) == "needle";
}); 
```

### 全局函数 std::get()

全局函数模板`std::get()`可用于`std::tuple`、`std::pair`、`std::variant`和`std::array`。有两种实例化`std::get()`的方式，一种是使用索引，一种是使用类型：

+   `std::get<Index>()`: 当`std::get()`与索引一起使用时，如`std::get<1>(v)`，它返回`std::tuple`、`std::pair`或`std::array`中相应索引处的值。

+   `std::get<Type>()`: 当`std::get()`与类型一起使用时，如`std::get<int>(v)`，返回`std::tuple`、`std::pair`或`std::variant`中的相应值。对于`std::variant`，如果变体当前不持有该类型，则会抛出`std::bad_variant_access`异常。请注意，如果`v`是`std::tuple`，并且`Type`包含多次，则必须使用索引来访问该类型。

在讨论了实用程序库中的基本模板之后，让我们看一些实际应用，以了解本章涵盖的内容在实践中的应用。

# 一些实际示例

我们将通过检查两个示例来结束本章，其中`std::tuple`、`std::tie()`和一些模板元编程可以帮助我们编写清晰和高效的代码。

## 示例 1：投影和比较运算符

在 C++20 中，需要为类实现比较运算符的情况大大减少，但仍然有一些情况下，我们需要为特定场景中的对象提供自定义比较函数。考虑以下类：

```cpp
struct Player {
  std::string name_{};
  int level_{};
  int score_{};
  // etc...
};
auto players = std::vector<Player>{};
// Add players here... 
```

假设我们想按照他们的属性对玩家进行排序：首要排序顺序是`level_`，次要排序顺序是`score_`。在实现比较和排序时，看到这样的代码并不罕见：

```cpp
auto cmp = [](const Player& lhs, const Player& rhs) {
  if (lhs.level_ == rhs.level_) {
    return lhs.score_ < rhs.score_;
  }
  else {
    return lhs.level_ < rhs.level_;
  }
};
std::sort(players.begin(), players.end(), cmp); 
```

当属性数量增加时，使用嵌套的`if-else`块编写这种风格的比较运算符很容易出错。我们真正想表达的是我们正在比较`Player`属性的*投影*（在这种情况下是一个严格的子集）。`std::tuple`可以帮助我们以更清晰的方式重写这段代码，而不需要`if-else`语句。

让我们使用`std::tie()`，它创建一个包含我们传递给它的 lvalue 引用的`std::tuple`。以下代码创建了两个投影，`p1`和`p2`，并使用`<`运算符进行比较：

```cpp
auto cmp = [](const Player& lhs, const Player& rhs) {
  auto p1 = std::tie(lhs.level_, lhs.score_); // Projection
  auto p2 = std::tie(lhs.level_, lhs.score_); // Projection
  return p1 < p2;
};
std::sort(players.begin(), players.end(), cmp); 
```

与使用`if-else`语句的初始版本相比，这非常清晰易读。但这真的有效吗？看起来我们需要创建临时对象来比较两个玩家。在微基准测试中运行这个代码并检查生成的代码时，使用`std::tie()`实际上没有任何开销；事实上，在这个例子中，使用`std::tie()`的版本比使用`if-else`语句的版本稍微快一些。

使用范围算法，我们可以通过将投影作为参数提供给`std::ranges::sort()`来进行排序，使代码更加清晰：

```cpp
std::ranges::sort(players, std::less{}, [](const Player& p) {
  return std::tie(p.level_, p.score_); 
}); 
```

这是`std::tuple`在不需要完整的具有命名成员的结构的情况下使用的一个例子，而不会在代码中牺牲任何清晰度。

## 例 2：反射

术语**反射**指的是在不知道类的内容的情况下检查类的能力。与许多其他编程语言不同，C++没有内置的反射，这意味着我们必须自己编写反射功能。反射计划包括在未来版本的 C++标准中；希望我们能在 C++23 中看到这个功能。

在这个例子中，我们将限制反射，使类能够迭代它们的成员，就像我们可以迭代元组的成员一样。通过使用反射，我们可以创建用于序列化或记录的通用函数，这些函数可以自动适用于任何类。这减少了在 C++中传统上需要的大量样板代码。

### 使一个类反映其成员

由于我们需要自己实现所有的反射功能，我们将从通过一个名为`reflect()`的函数公开成员变量开始。我们将继续使用在上一节中介绍的`Player`类。在这里，我们添加`reflect()`成员函数和一个构造函数的样子如下：

```cpp
class Player {
public:
  Player(std::string name, int level, int score)
      : name_{std::move(name)}, level_{level}, score_{score} {}

  auto reflect() const {
    return std::tie(name_, level_, score_);
  } 
private:
  std::string name_;
  int level_{};
  int score_{};
}; 
```

`reflect()`成员函数通过调用`std::tie()`返回成员变量的引用的元组。我们现在可以开始使用`reflect()`函数，但首先，关于使用手工制作的反射的替代方案的说明。

### 简化反射的 C++库

在 C++库世界中已经有了相当多的尝试来简化反射的创建。一个例子是 Louis Dionne 的元编程库*Boost Hana*，它通过一个简单的宏为类提供了反射能力。最近，*Boost*还添加了*Precise and Flat Reflection*，由 Anthony Polukhin 编写，它*自动*反映类的公共内容，只要所有成员都是简单类型。

然而，为了清晰起见，在这个例子中，我们只会使用我们自己的`reflect()`成员函数。

### 使用反射

现在`Player`类具有反射其成员变量的能力，我们可以自动创建大量功能，否则需要我们重新输入每个成员变量。正如您可能已经知道的，C++可以自动生成构造函数、析构函数和比较运算符，但其他运算符必须由程序员实现。其中一个这样的函数是`operator<<()`，它将其内容输出到流中以便将其存储在文件中，或更常见的是在应用程序日志中记录它们。

通过重载`operator<<()`并使用我们在本章前面实现的`tuple_for_each()`函数模板，我们可以简化为类创建`std::ostream`输出的过程，如下所示：

```cpp
auto& operator<<(std::ostream& ostr, const Player& p) { 
  tuple_for_each(p.reflect(), &ostr { 
    ostr << m << " "; 
  }); 
  return ostr; 
} 
```

现在，该类可以与任何`std::ostream`类型一起使用，如下所示：

```cpp
auto v = Player{"Kai", 4, 2568}; 
std::cout << v;                  // Prints: "Kai 4 2568 " 
```

通过通过元组反射我们的类成员，我们只需要在类中添加/删除成员时更新我们的反射函数，而不是更新每个函数并迭代所有成员变量。

### 有条件地重载全局函数

现在，我们有了一个使用反射而不是手动输入每个变量来编写大量函数的机制，但我们仍然需要为每种类型输入简化的大量函数。如果我们希望这些函数为每种可以反射的类型生成呢？

我们可以通过使用约束条件来有条件地为所有具有`reflect()`成员函数的类启用`operator<<()`。

首先，我们需要创建一个指向`reflect()`成员函数的新概念：

```cpp
template <typename T> 
concept Reflectable = requires (T& t) {
  t.reflect();
}; 
```

当然，这个概念只是检查一个类是否有一个名为`reflect()`的成员函数；它并不总是返回一个元组。总的来说，我们应该对这种只使用单个成员函数的弱概念持怀疑态度，但它对于例子来说是有用的。无论如何，我们现在可以在全局命名空间中重载`operator<<()`，使所有可反射的类都能够被比较并打印到`std::ostream`中：

```cpp
auto& operator<<(std::ostream& os, const Reflectable auto& v) {
  tuple_for_each(v.reflect(), &os {
    os << m << " ";
  });
  return os;
} 
```

前面的函数模板只会为包含`reflect()`成员函数的类型实例化，并因此不会与任何其他重载发生冲突。

### 测试反射能力

现在，我们已经准备就绪：

+   我们将测试的`Player`类有一个`reflect()`成员函数，返回对其成员的引用的元组

+   全局`std::ostream& operator<<()`已经重载了可反射类型

下面是一个简单的测试，用于验证这个功能：

```cpp
int main() {
  auto kai = Player{"Kai", 4, 2568}; 
  auto ari = Player{"Ari", 2, 1068}; 

  std::cout << kai; // Prints "Kai 4 2568" 
  std::cout << ari; // Prints "Ari 2 1068" 
} 
```

这些例子展示了`std::tie()`和`std::tuple`等小而重要的实用工具与一点元编程结合时的用处。

# 总结

在本章中，您已经学会了如何使用`std::optional`来表示代码中的可选值。您还看到了如何将`std::pair`、`std::tuple`、`std::any`和`std::variant`与标准容器和元编程结合在一起，以存储和迭代不同类型的元素。您还了解到`std::tie()`是一个概念上简单但功能强大的工具，可用于投影和反射。

在下一章中，您将了解如何进一步扩展您的 C++工具箱，通过学习如何构建隐藏的代理对象来创建库。
