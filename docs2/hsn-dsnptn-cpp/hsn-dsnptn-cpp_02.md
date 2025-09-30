

# 类和函数模板

C++的模板编程特性是一个庞大而复杂的主题，许多书籍专门用于教授这些特性。在这本书中，我们将使用许多高级的 C++泛型编程特性。那么，我们应该如何准备自己，以便理解这些语言结构，它们将在本书的各个部分出现？本章采用非正式的方法——而不是精确的定义，我们通过示例演示模板的使用，并解释不同的语言特性是如何工作的。如果你在这个时候发现自己的知识不足，鼓励你寻求更深入的理解，并阅读一本或更多专注于解释 C++语言语法和语义的书籍。当然，如果你希望有一个更精确、更正式的描述，你可以参考 C++标准或参考书籍。

本章将涵盖以下主题：

+   C++中的模板

+   类和函数模板

+   模板实例化

+   模板特化

+   模板函数的重载

+   可变模板

+   Lambda 表达式

+   概念

# C++中的模板

C++最伟大的优势之一是它对泛型编程的支持。在泛型编程中，算法和数据结构是用泛型类型编写的，这些类型将在以后指定。这允许程序员一次实现一个函数或一个类，然后，为许多不同的类型实例化它。模板是 C++的一个特性，允许在泛型类型上定义类和函数。C++支持三种类型的模板——函数、类和变量模板。

## 函数模板

函数模板是泛型函数——与常规函数不同，模板函数不声明其参数类型。相反，类型是模板参数：

```cpp
// Example 01
template <typename T>
T increment(T x) { return x + 1; }
```

此模板函数可用于将任何类型的值增加一，其中加一是一个有效的操作：

```cpp
increment(5);    // T is int, returns 6
increment(4.2);    // T is double, return 5.2 char c[10];
increment(c);    // T is char*, returns &c[1]
```

大多数模板函数对其模板参数的类型有一些限制。例如，我们的`increment()`函数要求表达式`x + 1`对`x`的类型是有效的。否则，尝试实例化模板将失败，并伴随着一些冗长的编译错误。

非成员函数和类成员函数都可以是函数模板；然而，虚函数不能是模板。泛型类型不仅可以用来声明函数参数，还可以用来声明函数体内的任何变量：

```cpp
template <typename T> T sum(T from, T to, T step) {
  T res = from;
  while ((from += step) < to) { res += from; }
  return res;
}
```

在 C++20 中，简单的模板声明可以被缩写：我们不需要写

```cpp
template <typename T> void f(T t);
```

我们可以写

```cpp
// Example 01a
void f(auto t);
```

除了更简洁的声明外，这种缩写没有特别的优势，这个特性相当有限。首先，`auto`只能用作“顶级”参数类型；例如，这是无效的（但某些编译器允许）：

```cpp
void f(std::vector<auto>& v);
```

并且仍然需要写成

```cpp
template <typename T> void f(std::vector<T>& v);
```

此外，如果你需要在函数声明的其他地方使用模板类型参数，你不能简化它们：

```cpp
template <typename T> T f(T t);
```

当然，你可以将返回类型声明为`auto`并使用尾随返回类型：

```cpp
auto f(auto t) -> decltype(t);
```

但在这个阶段，模板实际上并没有“简化”。

我们将在稍后看到更多关于函数模板的内容，但接下来我们先介绍类模板。

## 类模板

类模板是使用泛型类型的类，通常用于声明其数据成员，但也可以在它们内部声明方法和局部变量：

```cpp
// Example 02
template <typename T> class ArrayOf2 {
  public:
  T& operator[](size_t i) { return a_[i]; }
  const T& operator[](size_t i) const { return a_[i]; }
  T sum() const { return a_[0] + a_[1]; }
  private:
  T a_[2];
};
```

这个类只实现一次，然后可以用来定义任何类型的两个元素的数组：

```cpp
ArrayOf2<int> i; i[0] = 1; i[1] = 5;
std::cout << i.sum();                       // 6
ArrayOf2<double> x; x[0] = -3.5; x[1] = 4;
std::cout << x.sum();                       // 0.5
ArrayOf2<char*> c; char s[] = "Hello";
c[0] = s; c[1] = s + 2;
```

特别注意最后一个例子——你可能认为`ArrayOf2`模板与`char*`这样的类型不兼容——毕竟，它有一个`sum()`方法，如果`a_[0]`和`a_[1]`的类型是指针，则无法编译。然而，我们的示例按原样编译——类模板的方法不需要在我们尝试使用它之前就有效。如果我们从未调用`c.sum()`，那么它无法编译的事实永远不会出现，程序仍然有效。如果我们调用了一个对于所选模板参数无法编译的成员函数，我们会在模板体中得到语法错误（在我们的例子中，关于无法将两个指针相加的错误）。这些错误信息很少直接明了。即使它们是直接的，也不清楚问题是否出在函数体中，或者函数从一开始就不应该被调用。在本章的后面部分，我们将看到如何改善这种情况。

## 变量模板

C++中的最后一种模板是变量模板，它在 C++14 中引入。这种模板允许我们定义一个具有泛型类型的变量：

```cpp
// Example 03
template <typename T> constexpr T pi =
T(3.14159265358979323846264338327950288419716939937510582097494459230781L);
pi<float>;      // 3.141592
pi<double>;     // 3.141592653589793
```

变量模板在大多数情况下非常简单易用，主要用于定义自己的常量，但也有一些有趣的模式可以利用它们；我们将在下一节中看到一个例子。

## 非类型模板参数

通常，模板参数是类型，但 C++还允许几种非类型参数。首先，模板参数可以是整数或枚举类型的值：

```cpp
// Example 04
template <typename T, size_t N> class Array {
  public:
  T& operator[](size_t i) {
    if (i >= N) throw std::out_of_range("Bad index");
     return data_[i];
  }
  private:
  T data_[N];
};
Array<int, 5> a;      // OK
cin >> a[0];
Array<int, a[0]> b;   // Error
```

这是一个有两个参数的模板——第一个是类型，但第二个不是。它是一个`size_t`类型的值，用于确定数组的大小；这种模板与内置的 C 风格数组相比的优势在于它可以进行范围检查。C++标准库中有一个`std::array`类模板，应该用于替代在任意实际程序中实现自己的数组，但它确实是一个易于理解的示例。

用于实例化模板的非类型参数的值必须是编译时常量或 `constexpr` 值——前一个示例中的最后一行是无效的，因为 `a[0]` 的值直到程序在运行时读取它才已知。C++20 允许非类型模板参数使用浮点数和用户定义的类型；在此之前，参数仅限于整型、指针（包括函数和成员指针）、引用和枚举。当然，非类型参数的值必须是编译时常量，因此，例如，不允许指向局部变量的指针。

在 C++ 中，数值模板参数曾经非常流行，因为它们允许实现复杂的编译时计算，但在最近的版本标准中，`constexpr` 函数可以用来达到相同的效果，并且更容易阅读。当然，标准一手拿走，一手又给予，因此非模板参数与 `constexpr` 函数结合出现了一个有趣的新用例：这些函数首次在 C++11 中引入，用于定义“即时函数”，或编译时评估的函数。`constexpr` 函数的问题在于它们*可能*在编译时评估，但这不是必需的；它们也可能在运行时评估：

```cpp
constexpr size_t length(const char* s) {
  size_t res = 0;
  while (*(s++)) ++res;
  return res;
}
std::cout << length("abc") << std::endl;
char s[] = "runtime";
std::cout << length(s) << std::endl;
```

这里有一个 `constexpr` 函数 `length()`。长度计算实际上是在编译时发生的吗？除了检查生成的汇编代码（这可能会因编译器而异）外，没有其他方法可以知道。唯一确定的方法是在编译时上下文中调用该函数，例如：

```cpp
static_assert(length("abc") == 3, ""); // OK
char s[] = "runtime";
static_assert(length(s) == 7, ""); // Fails
```

第一个断言可以编译，而第二个即使值 7 是正确的也无法编译：论点是它不是一个编译时值，因此评估必须在运行时发生。

在 C++20 中，函数可以被声明为 `consteval` 而不是 `constexpr`：这保证了评估发生在编译时或者根本不发生（因此，前一个示例中的第二个 `cout` 语句将无法编译）。在 C++20 之前，我们必须发挥创意。这里有一种强制编译时执行的方法：

```cpp
// Example 05c
template <auto V>
static constexpr auto force_consteval = V;
```

`force_consteval` 变量模板可以用来强制编译时评估，如下所示：

```cpp
std::cout << force_consteval<length("abc")> << std::endl;
char s[] = "runtime";
std::cout << force_consteval<length(s)> << std::endl;
```

第二个 `cout` 语句无法编译，因为 `length()` 函数不能被评估为即时函数。变量模板 `force_consteval` 使用一个非类型模板参数，其类型未指定，而是从模板参数（一个 `auto` 模板参数）推导得出。这是一个 C++17 的特性；在 C++14 中，我们必须使用一个相当不优雅的宏来实现相同的结果：

```cpp
// Example 05d
template <typename T, T V>
static constexpr auto force_consteval_helper = V;
#define force_consteval(V)
force_consteval_helper<decltype(V), (V)>
std::cout << force_consteval(length("abc")) << std::endl;
```

如果一个非类型模板参数看起来“不如类型”，你会喜欢下一个选项，一个肯定比简单类型更复杂的参数。

## 模板模板参数

值得一提的第二种非类型模板参数是**模板模板**参数——即本身也是模板的模板参数。在本书的后续章节中我们将需要它们。这个模板参数的替换不是用类的名称，而是用整个模板的名称。

这里有一个具有模板模板参数的类模板：

```cpp
// Example 06a
template <typename T,
         template <typename> typename Container>
class Builder {
  Container<T> data_;
  public:
  void add(const T& t) { data_.push_back(t); }
  void print() const {
    for (const auto& x : data_) std::cout << x << " ";
    std::cout << std::endl;
  }
};
```

`Builder`模板声明了一个用于构造（构建）任意类型`T`的容器的类。容器本身没有特定的类型，它本身就是一个模板。

它可以用任何接受一个类型参数的容器模板实例化：

```cpp
template <typename T> class my_vector { … };
Builder<int, my_vector> b;
b.add(1);
b.add(2);
b.print();
```

当然，对`Container`模板还有额外的要求：它必须有一个单一的类型参数`T`（其余可以默认），它应该可以默认构造，它必须有一个`push_back()`方法，等等。C++20 为我们提供了一种简洁的方式来表述这些要求，并将它们作为模板接口的一部分；我们将在本章的*概念*部分学习它。

这里有一个具有两个模板模板参数的函数模板：

```cpp
// Example 06b
template <template <typename> class Out_container,
          template <typename> class In_container,
          typename T> Out_container<T>
resequence(const In_container<T>& in_container) {
  Out_container<T> out_container;
  for (auto x : in_container) {
    out_container.push_back(x);
  }
  return out_container;
}
```

这个函数接受一个任意的容器作为参数，并返回另一个容器，一个不同的模板，但实例化在相同的类型上，其值从输入容器复制而来：

```cpp
my_vector<int> v { 1, 2, 3, 4, 5 };
template <typename T> class my_deque { … };
auto d = resequence<my_deque>(v);// my_deque with 1 … 5
```

注意，编译器推导出模板参数的类型（`In_container`作为`my_vector`）以及其模板参数的类型（`T`作为`int`）。当然，剩余的模板参数`Out_container`无法推导（它不在模板函数的任何参数中使用），必须显式指定，这符合我们的预期用途。

模板模板参数有一个主要的限制，由于不同的编译器执行不均，这使得问题更加复杂（即，一些编译器通过了本应无法编译的代码，但你确实希望它能编译）。限制是，为模板模板指定的模板参数数量必须与参数的数量匹配。考虑这个模板函数：

```cpp
template <template <typename> class Container, typename T>
void print(const Container<T>& container) {
  for (auto x : container) { std::cout << x << " "; }
  std::cout << std::endl;
}
std::vector<int> v { 1, 2, 3, 4, 5 };
print(v);
```

这段代码可能可以编译，但这取决于标准的版本和编译器对标准的严格遵循：`std::vector`模板有两个模板参数，而不是一个。第二个参数是分配器；它有一个默认值，这就是为什么在声明向量对象时我们不必指定分配器类型。GCC、Clang 和 MSVC 都在一定程度上放宽了这一要求（但程度不同）。变长模板，我们将在本章后面看到，提供了一个更稳健的解决方案（至少在 C++17 及以后版本中）。

模板是生成代码的一种配方。接下来，我们将看到如何将这些配方转换为我们可以运行的实际代码。

# 模板实例化

模板名称不是一个类型，不能用来声明变量或调用函数。要创建类型或函数，模板必须被实例化。大多数时候，模板在使用时隐式实例化。我们再次从函数模板开始。

## 函数模板

要使用函数模板生成函数，我们必须指定所有模板类型参数应使用的类型。我们可以直接指定这些类型：

```cpp
template <typename T> T half(T x) { return x/2; }
int i = half<int>(5);
```

这将 `half` 函数模板实例化为 `int` 类型。类型是显式指定的；我们可以用另一种类型的参数调用该函数，只要它可转换为请求的类型：

```cpp
double x = half<double>(5);
```

即使参数是 `int` 类型，实例化的是 `half<double>`，返回类型是 `double`。整数值 `5` 被隐式转换为 `double`。

尽管每个函数模板都可以通过指定所有类型参数来实例化，但这很少发生。函数模板的大部分使用都涉及到类型的自动推导。考虑以下情况：

```cpp
auto x = half(8);    // int
auto y = half(1.5);    // double
```

模板类型只能从模板函数参数推导——编译器将尝试选择 `T` 参数的类型以匹配声明为相同类型的函数参数。在我们的例子中，函数模板具有 `T` 类型的 `x` 参数。对这个函数的任何调用都必须为这个参数提供某个值，并且这个值必须有一个类型。编译器将推导出 `T` 必须是那种类型。在前面的代码块中的第一次调用中，参数是 `5`，其类型是 `int`。在这种情况下，假设 `T` 应该是 `int` 是最好的选择。同样，在第二次调用中，我们可以推导出 `T` 必须是 `double`。

在此推导之后，编译器执行类型替换：所有其他对 `T` 类型的提及都被推导出的类型所替换；在我们的例子中，`T` 的其他使用只有一个是返回类型。

模板参数推导广泛用于捕获我们难以确定类型的场景：

```cpp
long x = ...;
unsigned int y = ...;
auto x = half(y + z);
```

在这里，我们推导出 `T` 类型为表达式 `y + z` 的类型（它是 `long`，但使用模板推导时，我们不需要显式指定，并且推导出的类型将*跟随*参数类型，如果我们更改 `y` 和 `z` 的类型）。考虑以下示例：

```cpp
template <typename U> auto f(U);
half(f(5));
```

我们推导出 `T` 以匹配 `f()` 模板函数对于 `int` 参数返回的类型（当然，在调用之前必须提供 `f()` 模板函数的定义，但我们不需要深入到定义 `f()` 的头文件中，因为编译器会为我们推导正确的类型）。

只有用于声明函数参数的类型才能被推导。没有规则要求所有模板类型参数都必须以某种方式出现在参数列表中，但任何无法推导的参数必须显式指定：

```cpp
template <typename U, typename V> U half(V x) {
  return x/2;
}
auto y = half<double>(8);
```

在这里，第一个模板类型参数被显式指定，所以`U`是`double`，而`V`被推断为`int`。

有时候，编译器无法推断模板类型参数，即使它们被用来声明参数：

```cpp
template <typename T> T Max(T x, T y) {
  return (x > y) ? x : y;
}
auto x = Max(7L, 11); // Error
```

在这里，我们可以从第一个论据中推断出`T`必须是`long`，但从第二个论据中，我们推断出`T`必须是`int`。对于学习模板的程序员来说，通常很令人惊讶的是在这种情况下`long`类型没有被推断出来——毕竟，如果我们将`long`替换到每个地方，第二个论据将隐式转换，函数将能够编译。那么为什么没有推断出*更大的*类型呢？因为编译器并不试图找到一个类型，使得所有参数转换都是可能的：毕竟，通常有不止一个这样的类型。在我们的例子中，`T`可以是`double`或`unsigned long`，函数仍然有效。如果一个类型可以从多个参数中推断出来，那么所有这些推断的结果必须相同。

否则，模板实例化被认为是模糊的。

类型推断并不总是像使用类型参数的类型那样直接。参数可能被声明为一个比类型参数本身更复杂的类型：

```cpp
template <typename T> T decrement(T* p) {
  return --(*p);
}
int i = 7;
decrement(&i);    // i == 6
```

这里，参数的类型是一个指向`int`的*指针*，但推断给`T`的类型是`int`。类型的推断可以是任意复杂的，只要它是明确的：

```cpp
template <typename T> T first(const std::vector<T>& v) {
  return v[0];
}
std::vector<int> v{11, 25, 67};
first(v);    // T is int, returns 11
```

这里，参数是另一个模板`std::vector`的实例化，我们必须从创建这个向量实例化的类型中推断模板参数类型。

正如我们所看到的，如果一个类型可以从多个函数参数中推断出来，那么这些推断的结果必须相同。另一方面，一个参数可以用来推断多个类型：

```cpp
template <typename U, typename V>
std::pair<V, U> swap12(const std::pair<U, V>& x) {
  return std::pair<V, U>(x.second, x.first);
}
swap12(std::make_pair(7, 4.2)); // pair of 4.2, 7
```

在这里，我们从单个论据中推断出两个类型，`U`和`V`，然后使用这两个类型来形成一个新类型，`std::pair<V, U>`。这个例子过于冗长，我们可以利用一些更多的 C++特性来使其更加紧凑且易于维护。首先，标准已经有一个函数可以推断参数类型并使用它们来声明一个 pair，我们甚至已经使用了这个函数——`std::make_pair()`。

其次，函数的返回类型可以从`return`语句中的表达式推断出来（这是一个 C++14 特性）。这种推断的规则与模板参数类型推断的规则相似。有了这些简化，我们的例子变成了以下这样：

```cpp
template <typename U, typename V>
auto swap12(const std::pair<U, V>& x) {
  return std::make_pair(x.second, x.first);
}
```

注意，我们不再显式使用类型`U`和`V`。我们仍然需要这个函数是一个模板，因为它操作的是通用类型，即两个类型的组合，我们不知道直到实例化函数。然而，我们可以只使用一个模板参数来代表参数的类型：

```cpp
template <typename T> auto swap12(const T& x) {
  return std::make_pair(x.second, x.first);
}
```

这两个变体之间存在一个显著的区别——最后一个函数模板将从任何只有一个参数的调用中成功推导出类型，无论该参数的类型如何。如果该参数不是 `std::pair`，或者更一般地说，如果参数不是一个类或结构体，或者它没有 `first` 和 `second` 数据成员，推导仍然会成功，但类型替换会失败。另一方面，上一个版本甚至不会考虑不是某些类型对的参数。对于任何 `std::pair` 参数，将推导出对类型，并且替换应该没有问题。我们能否使用最后一个声明并仍然将类型 `T` 限制为对或具有类似接口的另一个类？是的，我们将在本书后面的部分看到几种方法来实现这一点。

成员函数模板与非成员函数模板非常相似，它们的参数也是类似推导的。成员函数模板可以在类或类模板中使用，我们将在下一节中回顾这一点。

## 类模板

类模板的实例化类似于函数模板的实例化——使用模板创建类型会隐式地实例化模板。要使用类模板，我们需要指定模板参数的类型参数：

```cpp
template <typename N, typename D> class Ratio {
  public:
  Ratio() : num_(), denom_() {}
  Ratio(const N& num, const D& denom) :
    num_(num), denom_(denom) {}
  explicit operator double() const {
    return double(num_)/double(denom_);
  }
  private:
  N num_;
  D denom_;
};
Ratio<int, double> r;
```

`r` 变量的定义隐式地实例化了 `Ratio` 类模板的 `int` 和 `double` 类型。它还实例化了该类的默认构造函数。在这个代码中，第二个构造函数没有被使用，也没有被实例化。这是类模板的一个特性——实例化一个模板会实例化所有数据成员，但直到它们被使用时才实例化方法——这使得我们能够编写只针对某些类型编译部分方法的类模板。如果我们使用第二个构造函数来初始化 `Ratio` 的值，那么这个构造函数就会被实例化，并且必须对给定的类型有效：

```cpp
Ratio<int, double> r(5, 0.1);
```

在 C++17 中，这些构造函数可以用来从构造函数参数推导出类模板的类型：

```cpp
Ratio r(5, 0.1);
```

当然，这只有在有足够的构造函数参数可以推导类型时才有效。例如，默认构造的 `Ratio` 对象必须使用显式指定的类型来实例化；没有其他方法可以推导它们。在 C++17 之前，经常使用辅助函数模板来构造一个可以从参数推导出类型的对象。类似于我们之前看过的 `std::make_pair()`，我们可以实现一个 `make_ratio` 函数，它将执行与 C++17 构造函数参数推导相同的功能：

```cpp
template <typename N, typename D>
Ratio<N, D> make_ratio(const N& num, const D& denom) {
  return { num, denom };
}
auto r(make_ratio(5, 0.1));
```

如果 C++17 提供了推导模板参数的方式，应该优先使用：它不需要编写另一个本质上重复类构造函数的函数，也不需要调用复制或移动构造函数来初始化对象（尽管在实践中，大多数编译器都会执行返回值优化并优化掉对复制或移动构造函数的调用）。

当使用模板生成类型时，它会隐式实例化。类和函数模板也可以显式实例化。这样做会实例化模板而不使用它：

```cpp
template class Ratio<long, long>;
template Ratio<long, long> make_ratio(const long&,
                                      const long&);
```

显式实例化很少需要，并且本书的其他地方不会使用。

虽然具有特定模板参数的类模板实例化行为（主要是）像常规类一样，但类模板的静态数据成员值得特别提及。首先，让我们回顾一下静态类数据成员的常见挑战：它们必须在某个地方定义，并且只能定义一次：

```cpp
// In the header:
class A {
  static int n;
};
// In a C file:
int A::n = 0;
std::cout << A::n;
```

没有这样的定义，程序将无法链接：名称 `A::n` 未定义。但如果定义被移动到头文件中，并且头文件被包含在几个编译单元中，程序也将无法链接，这次名称 `A::n` 是多重定义的。

对于类模板，要求恰好定义一次静态数据成员是不切实际的：我们需要为模板实例化的每一组模板参数定义它们，我们无法在任何一个编译单元中做到这一点（其他编译单元可能以不同的类型实例化相同的模板）。幸运的是，这并不必要。类模板的静态成员可以（并且应该）与模板本身一起定义：

```cpp
// In the header:
template <typename T> class A {
  static T n;
};
template <typename T> T A<T>::n {};
```

虽然从技术上讲这会导致多个定义，但链接器的任务是合并它们，这样我们就能得到一个单一的定义（对于相同类型的所有对象，静态成员变量的值只有一个）。

在 C++17 中，内联变量提供了一个更简单的解决方案：

```cpp
// In the header:
template <typename T> class A {
  static inline T n {};
};
```

这也适用于非模板类：

```cpp
// In the header:
class A {
  static inline int n = 0;
};
```

如果类模板的静态数据成员有一个非平凡的构造函数，则该构造函数会为每个模板实例化调用一次（而不是每个对象——对于相同类型的所有对象，静态成员变量只有一个实例）。

到目前为止，我们使用的类模板允许我们声明泛型类，即可以用许多不同类型实例化的类。到目前为止，所有这些类看起来完全相同，除了类型外，并且生成相同的代码。这并不总是希望的——不同的类型可能需要以某种方式有所不同地处理。

例如，假设我们想要能够表示不仅存储在`Ratio`对象中的两个数字的比例，而且还想表示存储在其他地方的两个数字的比例，其中`Ratio`对象包含对这些数字的指针。显然，如果对象存储了分子和分母的指针，那么`Ratio`对象的一些方法，如转换为`double`的转换操作符，需要以不同的方式实现。在 C++中，这是通过特化模板来实现的，我们将在下面进行操作。

# 模板特化

模板特化允许我们对某些类型生成不同的模板代码——不仅仅是用不同类型替换后的相同代码，而是完全不同的代码。在 C++中，模板特化有两种类型——显式特化（或完全特化）和部分特化。让我们先从前者开始。

## 显式特化

显式模板特化定义了针对一组特定类型的模板的特殊版本。在显式特化中，所有泛型类型都被替换为具体的、具体的类型。由于显式特化不是一个泛型类或函数，因此它不需要在以后进行实例化。出于同样的原因，有时它被称为**完全特化**。如果泛型类型被完全替换，就没有任何泛型剩余了。显式特化不应与显式模板实例化混淆——虽然两者都为给定的一组类型参数创建了一个模板的实例，但显式实例化创建了一个泛型代码的实例，其中泛型类型被具体类型替换。显式特化创建了一个具有相同名称的函数或类的实例，但它覆盖了实现，因此生成的代码可以完全不同。一个例子可以帮助我们理解这种区别。

让我们从类模板开始。假设，如果`Ratio`的分子和分母都是`double`类型，我们想要计算这个比例并将其存储为一个单独的数字。通用的`Ratio`代码应该保持不变，但对于一组特定的类型，我们希望类看起来完全不同。我们可以通过显式特化来实现这一点：

```cpp
template <> class Ratio<double, double> {
  public:
  Ratio() : value_() {}
  template <typename N, typename D>
    Ratio(const N& num, const D& denom) :
      value_(double(num)/double(denom)) {}
  explicit operator double() const { return value_; }
  private:
  double value_;
};
```

两个模板类型参数都被指定为`double`。类的实现与通用版本完全不同——我们只有一个数据成员，而不是两个；转换操作符简单地返回值，而构造函数现在计算分子和分母的比例。但这甚至不是同一个构造函数——我们提供了一个模板构造函数，它可以接受任何类型的两个参数，只要它们可以转换为`double`，而不是通用版本中如果为两个`double`模板参数实例化时将拥有的非模板构造函数`Ratio(const double&, const double&)`。

有时，我们不需要特化整个类模板，因为大部分泛型代码仍然适用。然而，我们可能想要更改一个或几个成员函数的实现。我们也可以显式特化成员函数：

```cpp
template <> Ratio<float, float>::operator double() const {
  return num_/denom_;
}
```

模板函数也可以显式特化。同样，与显式实例化不同，我们可以编写函数体，并以我们想要的方式实现它：

```cpp
template <typename T> T do_something(T x) {
  return ++x;
}
template <> double do_something<double>(double x) {
  return x/2;
}
do_something(3);        // 4
do_something(3.0);    // 1.5
```

然而，我们无法更改参数的数量或类型，或者返回类型——它们必须与泛型类型的替换结果相匹配，因此以下代码无法编译：

```cpp
template <> long do_something<int>(int x) { return x*x; }
```

在使用模板之前，必须显式声明一个特化，以避免对相同类型的泛型模板进行隐式实例化。这很有道理——隐式实例化将创建一个与显式特化具有相同名称和相同类型的类或函数。现在程序中会有两个相同类或函数的版本，这违反了单一定义规则，并使程序无效（具体规则可以在标准中的*[基本定义.ODR]*部分找到）。

当我们有一个或几个类型需要模板以非常不同的方式行为时，显式特化是有用的。然而，这并没有解决我们关于指针比例的问题——我们想要一个仍然*部分泛型*的特化，即它可以处理任何类型的指针，但不能处理其他类型的指针。这是通过部分特化实现的，我们将在下一节中探讨。

## 部分特化

现在，我们正在进入 C++模板编程的真正有趣部分——部分模板特化。当一个类模板部分特化时，它仍然保持为泛型代码，但比原始模板*不那么泛型*。部分模板的最简单形式是其中一些泛型类型被具体类型替换，但其他类型仍然是泛型：

```cpp
template <typename N, typename D> class Ratio {
  .....
};
template <typename D> class Ratio<double, D> {
  public:
  Ratio() : value_() {}
  Ratio(const double& num, const D& denom) :
    value_(num/double(denom)) {}
  explicit operator double() const { return value_; }
  private:
  double value_;
};
```

在这里，如果分子是`double`类型，无论分母类型如何，我们都将`Ratio`转换为`double`值。可以为同一个模板定义多个部分特化。例如，我们还可以为分母是`double`而分子是任何类型的情况进行特化：

```cpp
template <typename N> class Ratio<N, double> {
  public:
  Ratio() : value_() {}
  Ratio(const N& num, const double& denom) :
    value_(double(num)/denom) {}
  explicit operator double() const { return value_; }
  private:
  double value_;
};
```

当模板被实例化时，会选择给定类型集的最佳特殊化。在我们的情况下，如果分子和分母都不是`double`，那么必须实例化通用模板——没有其他选择。如果分子是`double`，那么第一个部分特殊化比通用模板是一个更好的（更具体的）匹配。如果分母是`double`，那么第二个部分特殊化是一个更好的匹配。但如果两个项都是`double`呢？在这种情况下，两个部分特殊化是等效的；没有一个比另一个更具体。这种情况被认为是模糊的，实例化失败。请注意，只有这个特定的实例化`Ratio<double, double>`失败——定义两个特殊化不是错误（至少，不是一个语法错误），但请求一个无法唯一解析到最窄特殊化的实例化是错误的。为了允许我们的模板的任何实例化，我们必须消除这种模糊性，而做到这一点的方法是提供一个比其他两个更窄的特殊化。在我们的情况下，只有一个选项——为`Ratio<double, double>`提供一个完全特殊化：

```cpp
template <> class Ratio<double, double> {
  public:
  Ratio() : value_() {}
  template <typename N, typename D>
    Ratio(const N& num, const D& denom) :
      value_(double(num)/double(denom)) {}
  explicit operator double() const { return value_; }
  private:
  double value_;
};
```

现在，对于`Ratio<double, double>`实例化而言，部分特殊化是不明确的这一事实已不再相关——我们有一个比它们任何一个都更具体的模板版本，因此该版本优先于两者。

部分特殊化不必完全指定一些泛型类型。因此，可以保持所有类型都是泛型，但对其施加一些限制。例如，我们仍然想要一个特殊化，其中分子和分母都是指针。它们可以是任何东西的指针，所以它们是泛型类型，但比通用模板的任意类型*不那么泛型*：

```cpp
template <typename N, typename D> class Ratio<N*, D*> {
  public:
  Ratio(N* num, D* denom) : num_(num), denom_(denom) {}
  explicit operator double() const {
    return double(*num_)/double(*denom_);
  }
  private:
  N* const num_;
  D* const denom_;
};
int i = 5; double x = 10;
auto r(make_ratio(&i, &x));        // Ratio<int*, double*>
double(r);                    // 0.5
x = 2.5;
double(r);                    // 2
```

这个部分特殊化仍然有两个泛型类型，但它们都是指针类型，`N*`和`D*`，对于任何`N`和`D`类型。其实现与通用模板完全不同。当用两个指针类型实例化时，部分特殊化比通用模板更具体，被认为是更好的匹配。请注意，在我们的例子中，分母是`double`。那么为什么没有考虑对`double`分母的特殊化呢？那是因为，尽管从程序逻辑的角度来看，分母是`double`，但从技术上讲，它是`double*`，这是一个完全不同的类型，我们没有为其提供特殊化。

要定义一个特殊化，必须首先声明一个通用模板。然而，它不需要被定义——可以特殊化在通用情况下不存在的模板。为此，我们必须提前声明通用模板，然后定义我们需要的所有特殊化：

```cpp
template <typename T> class Value; // Declaration 
template <typename T> class Value<T*> {
  public:
  explicit Value(T* p) : v_(*p) {} private:
  T v_;
};
template <typename T> class Value<T&> {
  public:
  explicit Value(T& p) : v_(p) {}
  private:
  T v_;
};
int i = 5; int* p = &i; int& r = i;
Value<int*> v1(p); // T* specialization
Value<int&> v2(r); // T& specialization
```

在这里，我们没有通用的`Value`模板，但我们为任何指针或引用类型提供了部分特殊化。如果我们尝试在某种其他类型上实例化该模板，例如`int`，我们将得到一个错误，指出`Value<int>`类型是不完整的——这和尝试仅使用类的声明前缀来定义一个对象没有区别。

到目前为止，我们只看到了类模板的部分特殊化示例。与前面关于完全特殊化的讨论不同，我们没有在这里看到单个函数特殊化。这有一个非常好的原因——C++中不存在部分函数模板特殊化。有时错误地称为部分特殊化的是模板函数重载的简单形式。另一方面，模板函数的重载可以变得相当复杂，值得学习——我们将在下一节中介绍这一点。

## 模板函数重载

我们习惯于常规函数或类方法的重载——具有相同名称的多个函数具有不同的参数类型。每个调用都会调用与调用参数参数类型最佳匹配的函数，如下例所示：

```cpp
// Example 07
void whatami(int x) {
  std::cout << x << " is int" << std::endl;
}
void whatami(long x) {
  std::cout << x << " is long" << std::endl;
}
whatami(5);    // 5 is int
whatami(5.0);    // Compilation error
```

如果参数与给定名称的重载函数之一完全匹配，则调用该函数。否则，编译器会考虑将参数类型转换为可用函数的转换。如果其中一个函数提供了*更好的*转换，则选择该函数。否则，调用是模糊的，就像在前面示例的最后一行一样。关于构成*最佳*转换的确切定义可以在标准中找到（见*重载*部分，更具体地说，见子部分*[over.match]*）。一般来说，*最便宜*的转换是添加`const`或移除引用等；然后，有内置类型之间的转换，从派生类指针到基类指针的转换等。在多个参数的情况下，所选函数的每个参数都必须有最佳转换。没有*投票*——如果一个函数有三个参数，其中两个与第一个重载完全匹配，而第三个与第二个重载完全匹配，那么即使剩余的参数可以隐式转换为相应的参数类型，重载调用仍然是模糊的。

模板的存在使得重载解析变得更加复杂。除了非模板函数外，还可以定义具有相同名称和可能相同数量的参数的多个函数模板。所有这些函数都是重载函数调用的候选者，但函数模板可以生成具有不同参数类型的函数，那么我们如何决定实际的重载函数是什么？确切的规则甚至比非模板函数的规则更复杂，但基本思想是这样的——如果有一个非模板函数与调用参数几乎完美匹配，则选择该函数。当然，标准使用比“几乎完美”更精确的术语，但“平凡”转换，如添加`const`，属于这一类别——你可以“免费”获得它们。如果没有这样的函数，编译器将尝试将所有具有相同名称的函数模板实例化成与调用参数几乎完美匹配的形式，使用模板参数推导。如果恰好只有一个模板被实例化，则调用由这种实例化创建的函数。否则，重载解析将在非模板函数中继续以通常的方式进行。

这是对一个非常复杂过程的非常简化的描述，但有两个重要的要点——首先，如果对一个模板函数和一个非模板函数的调用有同样好的匹配，则优先选择非模板函数，其次，编译器不会尝试将函数模板实例化成可能转换为所需类型的对象。模板函数在参数类型推导后必须几乎完美匹配调用，否则根本不会被调用。让我们在我们的上一个例子中添加一个模板：

```cpp
void whatami(int x); // Same as above
void whatami(long x); // Same as above
template <typename T> void whatami(T* x) {
  std::cout << x << " is a pointer" << std::endl;
}
int i = 5;
whatami(i);    // 5 is int
whatami(&i);    // 0x???? is a pointer
```

这里看起来像是一个函数模板的部分特化。但实际上并不是——它只是一个函数模板——没有一般模板可以特化。相反，它只是一个类型参数从相同参数推导出来的函数模板，但使用不同的规则。如果参数是任何类型的指针，则可以推导出模板的类型。这包括指向`const`的指针——`T`可以是`const`类型，所以如果我们调用`whatami(ptr)`，其中`ptr`是`const int*`，那么当`T`是`const int`时，第一个模板重载是一个完美的匹配。如果推导成功，由模板生成的函数，即模板实例化，将被添加到重载集中。

对于`int*`参数，它是唯一可以工作的重载，因此被调用。但如果多个函数模板可以匹配调用，并且两种实例化都是有效的重载会发生什么？让我们再添加一个模板：

```cpp
void whatami(int x); // Same as above
void whatami(long x); // Same as above
template <typename T> void whatami(T* x); // Same as above
template <typename T> void whatami(T&& x) {
  std::cout << "Something weird" << std::endl;
}
class C {    };
C c;
whatami(c);    // Something weird
whatami(&c);    // 0x???? is a pointer
```

这个模板函数通过通用引用接受其参数，因此它可以针对任何只有一个参数的 `whatami()` 调用进行实例化。第一次调用，`whatami(c)`，很简单——最后一个重载，带有 `T&&`，是唯一可以调用的。没有从 `c` 到指针或整数的转换。但第二次调用很棘手——我们有一个，而不是一个，与调用完全匹配的模板实例化，且不需要任何转换。那么为什么这不是一个模糊的重载呢？因为解决重载函数模板的规则与非模板函数的规则不同，类似于选择类模板部分特殊化的规则（这也是为什么函数模板重载经常与部分特殊化混淆的另一个原因）。更具体的模板是一个更好的匹配。

在我们的情况下，第一个模板更具体——它可以接受任何指针参数，但只能是指针。第二个模板可以接受任何参数，所以每次第一个模板是一个可能的匹配时，第二个也是，但反之则不然。如果可以使用更具体的模板实例化一个有效的重载函数，那么就使用这个模板。

否则，我们必须回退到更通用的模板。

重载集中的非常通用的模板函数有时会导致意想不到的结果。假设我们有以下三个针对 `int`、`double` 和任何东西的重载：

```cpp
void whatami(int x) {
  std::cout << x << " is int" << std::endl;
}
void whatami(double x) {
  std::cout << x << " is double" << std::endl;
}
template <typename T> void whatami(T&& x) {
  std::cout << "Something weird" << std::endl;
}
int i = 5;
float x = 4.2;
whatami(i);    // i is int
whatami(x);    // Something weird
whatami(1.2);    // 1.2 is double
```

第一次调用有一个 `int` 参数，所以 `whatami(int)` 是一个完美的匹配。如果我没有模板重载，第二次调用将转到 `whatami(double)`——从 `float` 到 `double` 的转换是隐式的（从 `float` 到 `int` 的转换也是隐式的，但到 `double` 的转换更优先）。但这仍然是一个转换，所以当函数模板实例化到 `whatami(float&&)` 的完美匹配时，这是最佳匹配，也是选择的重载。最后一个调用有一个 `double` 参数，再次我们有一个与非模板函数 `whatami(double)` 的完美匹配，所以它比任何其他替代方案更优先。

应该注意的是，为相同参数类型重载按值传递和按引用传递的函数通常会在重载解析中产生歧义。例如，这两个函数几乎总是模糊的：

```cpp
template <typename T> void whatami(T&& x) {
  std::cout << "Something weird" << std::endl;
}
template <typename T> void whatami(T x) {
  std::cout << "Something copyable" << std::endl;
}
class C {};
C c;
whatami(c);
```

只要函数的参数可以被复制（并且我们的对象 `c` 是可复制的），重载就是模糊的，调用将无法编译。当更具体的函数重载更一般的函数时，问题不会发生（在我们所有的前例中，`whatami(int)` 使用按值传递且没有问题），但将两种类型的参数传递混合在类似通用的函数中是不明智的。

最后，还有一种函数在重载解析顺序中有一个特殊的位置——可变参数函数。

可变参数函数使用 `...` 而不是参数来声明，并且可以用任何类型和数量的参数调用（`printf` 就是一个这样的函数）。这个函数是最后的手段重载——只有当没有其他重载可用时才会调用：

```cpp
void whatami(...) {
  std::cout << "It's something or somethings" << std::endl;
}
```

只要我们有 `whatami(T&& x)` 重载可用，可变参数函数就不会是首选的重载，至少对于任何只有一个参数的 `whatami()` 调用来说是这样。没有那个模板，`whatami(...)` 将会用于任何非数字或指针类型的参数。可变参数函数自 C 语言时代起就存在了，不要与在 C++11 中引入的可变参数模板混淆，这正是我们接下来要讨论的内容。

# 可变参数模板

C 和 C++ 之间最显著的差异可能是类型安全。在 C 中可以编写泛型代码——标准函数 `qsort()` 就是一个完美的例子——它可以对任何类型的值进行排序，并且它们通过 `void*` 指针传递，这实际上可以是任何类型的指针。当然，程序员必须知道实际的类型，并将指针转换为正确的类型。在泛型 C++ 程序中，类型要么在实例化时显式指定，要么推导出来，泛型类型的类型系统与常规类型的类型系统一样强大。除非我们想要一个具有未知数量参数的函数，即在 C++11 之前，唯一的方法是使用旧的 C 风格的可变参数函数，其中编译器根本不知道参数类型是什么；程序员只需知道并正确地解包可变参数即可。

C++11 引入了可变参数函数的现代等价物——可变参数模板。现在我们可以声明具有任何数量参数的泛型函数：

```cpp
template <typename ... T> auto sum(const T& ... x);
```

这个函数接受一个或多个参数，这些参数可能具有不同的类型，并计算它们的总和。返回类型不容易确定，但幸运的是，我们可以让编译器来决定——我们只需将返回类型声明为 `auto`。我们实际上如何实现这个函数来累加未知数量的值，而这些值的类型我们甚至无法命名，甚至不能作为泛型类型？在 C++17 中，这很容易，因为有了折叠表达式：

```cpp
// Example 08a
template <typename ... T> auto sum(const T& ... x) {
  return (x + ...);
}
sum(5, 7, 3);        // 15, int
sum(5, 7L, 3);        // 15, long
sum(5, 7L, 2.9);        // 14.9, double
```

您可以验证结果类型正是我们所声称的类型：

```cpp
static_assert(std::is_same_v<
  decltype(sum(5, 7L, 2.9)), double>);
```

在 C++14 以及 C++17 中，当折叠表达式不足以使用（并且它们仅在有限的上下文中有用，主要是在使用二进制或一元运算符组合参数时），标准技术是递归，这在模板编程中一直很受欢迎：

```cpp
// Example 08b
template <typename T1> auto sum(const T1& x1) {
  return x1;
}
template <typename T1, typename ... T>
auto sum(const T1& x1, const T& ... x) {
  return x1 + sum(x ...);
}
```

第一个重载（不是部分特化！）是为具有任何类型单个参数的 `sum()` 函数。该值将被返回。第二个重载用于一个以上的参数，并且第一个参数被明确地加到剩余参数的总和中。递归继续进行，直到只剩下一个参数，此时将调用另一个重载并停止递归。这是在变长模板中解开参数包的标准技术，我们将在本书中多次看到这一点。编译器将内联所有递归函数调用并生成直接将所有参数相加的代码。

类模板也可以是变长的——它们有任意数量的类型参数，并且可以从不同类型的不同数量的对象构建类。其声明与函数模板类似。例如，让我们构建一个类模板 `Group`，它可以保存任何数量的不同类型的对象，并在转换为它所保存的类型时返回正确的对象：

```cpp
// Example 09
template <typename ... T> struct Group;
```

这种模板的通常实现仍然是递归的，使用深度嵌套的继承，尽管有时可能存在非递归实现。我们将在下一节中看到一个例子。当只剩下一个类型参数时，递归必须终止。这是通过部分特化来完成的，因此我们将之前显示的通用模板仅作为声明保留，并为一个类型参数定义一个特化：

```cpp
template <typename ... T> struct Group;
template <typename T1> struct Group<T1> {
  T1 t1_;
  Group() = default;
  explicit Group(const T1& t1) : t1_(t1) {}
  explicit Group(T1&& t1) : t1_(std::move(t1)) {}
  explicit operator const T1&() const { return t1_; }
  explicit operator T1&() { return t1_; }
};
```

这个类保存一个类型 `T1` 的值，通过复制或移动来初始化它，并在转换为 `T1` 类型时返回对其的引用。对于任意数量的类型参数的特化包含第一个类型作为数据成员，以及相应的初始化和转换方法，并从剩余类型的 `Group` 类模板继承：

```cpp
template <typename T1, typename ... T>
struct Group<T1, T ...> : Group<T ...> {
  T1 t1_;
  Group() = default;
  explicit Group(const T1& t1, T&& ... t) :
    Group<T ...>(std::forward<T>(t) ...), t1_(t1) {}
  explicit Group(T1&& t1, T&& ... t) :
    Group<T...>(std::forward<T>(t)...),
                t1_(std::move(t1)) {}
  explicit operator const T1&() const { return t1_; }
  explicit operator T1&() { return t1_; }
};
```

对于 `Group` 类中包含的每个类型，它有两种可能的初始化方式——复制或移动。幸运的是，我们不必为复制和移动操作的每种组合指定构造函数。相反，我们有两种构造函数版本，用于初始化第一个参数（存储在特化中的那个）；我们使用完美转发来处理剩余的参数。

现在，我们可以使用我们的 `Group` 类模板来保存不同类型的一些值（它无法处理相同类型的多个值，因为检索此类型的尝试将是模糊的）：

```cpp
Group<int, long> g(3, 5);
int(g);    // 3
long(g);    // 5
```

明确写出所有组类型并确保它们与参数类型匹配相当不方便。在 C++17 中，我们可以使用推导指南来启用从构造函数的类模板参数推导：

```cpp
template <typename ... T> Group(T&&... t) -> Group<T...>;
Group g(3, 2.2, std::string("xyz"));
int(g);            // 3
double(g);            // 2.2
std::string(g);        // "xyz"
```

在 C++17 之前，解决这个问题的通常方法是使用辅助函数模板（当然是一个变长模板）来利用模板参数推导：

```cpp
template <typename ... T> auto makeGroup(T&& ... t) {
  return Group<T ...>(std::forward<T>(t) ...);
}
auto g = makeGroup(3, 2.2, std::string("xyz"));
```

注意，C++ 标准库包含一个类模板 `std::tuple`，这是我们的 `Group` 的一个更完整、功能更丰富的版本。

变长模板也可以有非类型参数；在这种情况下，`makeGroup` 模板可以用任意数量的参数实例化。通常，这些非类型参数包会与 `auto`（推导）类型结合使用。例如，这里有一个模板，它持有不同类型编译时常量值的列表：

```cpp
// Example 10
template <auto... Values> struct value_list {};
```

没有使用 `auto`（即，在 C++17 之前），几乎不可能声明这样的模板，因为类型必须被显式指定。请注意，这是一个完整的模板：它将其定义的一部分作为常量值持有。为了提取它们，我们需要另一个变长模板：

```cpp
template <size_t N, auto... Values>
struct nth_value_helper;
template <size_t n, auto v1, auto... Values>
struct nth_value_helper<n, v1, Values...> {
  static constexpr auto value =
    nth_value_helper<n - 1, Values...>::value;
};
template <auto v1, auto... Values>
struct nth_value_helper<0, v1, Values...> {
  static constexpr auto value = v1;
};
template <size_t N, auto... Values>
constexpr auto nth_value(value_list<Values...>) {
  return nth_value_helper<N, Values...>::value;
}
```

模板函数 `nth_value` 从 `value_list` 参数（该参数本身不包含数据，除了其类型外没有其他兴趣）的类型推导出参数包 `Values`。然后使用部分类特化的递归实例化来遍历参数包，直到我们得到第 `N` 个值。请注意，为了以这种方式存储浮点常量，我们需要 C++20。

变长模板可以与模板模板参数结合使用，以解决例如当标准库容器用作模板模板参数的替代参数时产生的一些问题。一个简单的解决方案是将参数声明为接受任意数量的类型：

```cpp
template <template <typename...> class Container,
         typename... T>
void print(const Container<T...>& container);
std::vector<int> v{ … };
print(v);
```

注意，`std::vector` 模板有两个类型参数。在 C++17 中，一个标准更改使其成为 `Container` 模板模板参数中指定的参数包的有效匹配。大多数编译器甚至更早之前就允许这种匹配。

变长模板，特别是与完美转发结合使用，对于编写非常通用的模板类非常有用——例如，一个向量可以包含任意类型的对象，为了就地构造这些对象而不是复制它们，我们必须调用具有不同数量参数的构造函数。当编写向量模板时，无法知道需要多少参数来初始化向量将包含的对象，因此必须使用变长模板（实际上，`std::vector` 的就地构造函数，如 `emplace_back`，是变长模板）。

在 C++ 中，我们还要提到一种类似于模板的实体，它既有类又有函数的外观——这就是 lambda 表达式。下一节将专门介绍这个内容。

# Lambda 表达式

在 C++ 中，常规函数语法通过 *可调用* 的概念得到了扩展，可调用是 *可调用实体* 的简称——可调用是像函数一样可以调用的东西。可调用的例子包括函数（当然）、函数指针或具有 `operator()` 的对象，也称为**仿函数**：

```cpp
void f(int i); struct G {
  void operator()(int i);
};
f(5);            // Function
G g; g(5);        // Functor
```

在局部上下文中定义一个可调用实体通常很有用，就在它被使用的地方附近。例如，为了对一个对象的序列进行排序，我们可能需要定义一个自定义的比较函数。我们可以使用普通函数来完成这个任务：

```cpp
bool compare(int i, int j) { return i < j; }
void do_work() {
  std::vector<int> v;
  .....
  std::sort(v.begin(), v.end(), compare);
}
```

然而，在 C++中，函数不能在其它函数内部定义，因此我们的`compare()`函数可能必须定义在它被使用的地方相当远的地方。如果它是一个单次使用的比较函数，这种分离是不方便的，并且会降低代码的可读性和可维护性。

有一种方法可以绕过这个限制——虽然我们无法在函数内部声明函数，但我们可以在函数内部声明类，并且类是可以调用的：

```cpp
void do_work() {
  std::vector<int> v;
  .....
  struct compare {
    bool operator()(int i, int j) const { return i < j; }
  };
  std::sort(v.begin(), v.end(), compare());
}
```

这既紧凑又局部，但相当冗长。实际上，我们并不需要给这个类命名，而且我们只希望有一个这个类的实例。在 C++11 中，我们有一个更好的选择，那就是 Lambda 表达式：

```cpp
void do_work() {
  std::vector<int> v;
  .....
  auto compare = [](int i, int j) { return i < j; };
  std::sort(v.begin(), v.end(), compare);
}
```

如果我们只为一次调用`std::sort`使用这个比较函数，甚至不需要给它命名，可以直接在调用内部定义：

```cpp
  std::sort(v.begin(), v.end(),
            [](int i, int j) { return i < j; });
```

这是最紧凑的方式。可以指定返回类型，但通常可以由编译器推导出来。Lambda 表达式创建了一个对象，因此它有一个类型，但这个类型是由编译器生成的，因此对象声明必须使用`auto`。

Lambda 表达式是对象，因此它们可以有数据成员。当然，局部可调用类也可以有数据成员。通常，它们是从包含作用域中的局部变量初始化的：

```cpp
// Example 11
void do_work() {
  std::vector<double> v;
  .....
  struct compare_with_tolerance {
    const double tolerance;
    explicit compare_with_tolerance(double tol) :
      tolerance(tol) {}
    bool operator()(double x, double y) const {
      return x < y && std::abs(x - y) > tolerance;
    }
  };
  double tolerance = 0.01;
  std::sort(v.begin(), v.end(),
            compare_with_tolerance(tolerance));
}
```

再次强调，这是一种非常冗长的简单操作方式。我们必须三次提到容差变量——作为数据成员、构造函数参数以及在成员初始化列表中。Lambda 表达式使这段代码更简洁，因为它可以捕获局部变量。在局部类中，我们不允许通过构造函数参数以外的任何方式引用包含作用域中的变量，但对于 Lambda 表达式，编译器会自动生成一个构造函数来捕获表达式体中提到的所有局部变量：

```cpp
void do_work() {
  std::vector<double> v;
  .....
  double tolerance = 0.01;
  auto compare_with_tolerance = = {
    return x < y && std::abs(x - y) > tolerance;
  };
  std::sort(v.begin(), v.end(), compare_with_tolerance);
}
```

在这里，Lambda 表达式内部的`tolerance`这个名字指的是具有相同名称的局部变量。变量是通过值捕获的，这在 Lambda 表达式的捕获子句`[=]`中指定。我们也可以通过`[&]`像这样使用引用捕获：

```cpp
auto compare_with_tolerance = & {
  return x < y && std::abs(x - y) > tolerance;
};
```

不同之处在于，当按值捕获时，在 Lambda 对象构造的点会创建捕获变量的副本。这个局部副本默认是`const`的，尽管我们可以声明 Lambda 为可变的，这样我们就可以改变捕获的值：

```cpp
double tolerance = 0.01;
size_t count = 0; // line 2
auto compare_with_tolerance = = mutable {
  std::cout << "called " << ++count << " times\n";
  return x < y && std::abs(x - y) > tolerance;
};
std::vector<double> v;
… store values in v …
// Counts calls but does not change the value on line 2
std::sort(v.begin(), v.end(), compare_with_tolerance);
```

另一方面，通过引用捕获外部作用域中的变量会使 Lambda 表达式内部对这个变量的每次提及都指向原始变量。通过引用捕获的值可以被更改：

```cpp
double tolerance = 0.01;
size_t count = 0;
auto compare_with_tolerance = & mutable {
  ++count; // Changes count above
  return x < y && std::abs(x - y) > tolerance;
};
std::vector<double> v;
… store values in v …
std::sort(v.begin(), v.end(), compare_with_tolerance);
std::cout << "lambda called " << count << " times\n";
```

还可以通过值或引用显式捕获一些变量；例如，捕获 `[=, &count]` 通过值捕获了一切，除了 `count`，它通过引用捕获。

而不是将早期示例中的 lambda 表达式的参数从 `int` 改为 `double`，我们可以将它们声明为 `auto`，这实际上使得 lambda 表达式的 `operator()` 成为一个模板（这是一个 C++14 特性）。

Lambda 表达式最常被用作局部函数。然而，它们实际上并不是函数；它们是可调用对象，因此它们缺少函数的一个特性——重载它们的能力。在本节中我们将学习的最后一个技巧是如何绕过这一点，并从 lambda 表达式中创建一个重载集。

首先，主要思想——确实不可能重载可调用对象。另一方面，在同一个对象中重载多个 `operator()` 方法非常容易——方法的重载就像任何其他函数一样。当然，lambda 表达式对象的 `operator()` 是由编译器生成的，而不是由我们声明的，因此不可能强迫编译器在同一个 lambda 表达式中生成多个 `operator()`。但类有自己的优点，主要优点是我们可以从它们继承。

Lambda 表达式是对象——它们的类型是类，因此我们也可以从它们继承。如果一个类公开继承自基类，那么基类的所有公共方法都成为派生类的公共方法。如果一个类公开继承自多个基类（多重继承），那么它的公共接口是由所有基类的所有公共方法组成的。如果在这个集合中有多个同名方法，它们将变成重载方法，并应用通常的重载解析规则（特别是，可能创建一个模糊的重载集，在这种情况下程序将无法编译）。

因此，我们需要创建一个类，它可以自动从任意数量的基类继承。我们刚刚看到了完成这个任务的正确工具——变长模板。正如我们在上一节中学到的，遍历变长模板参数包中的任意数量项的通常方法是递归：

```cpp
// Example 12a
template <typename ... F> struct overload_set;
template <typename F1>
struct overload_set<F1> : public F1 {
  overload_set(F1&& f1) : F1(std::move(f1)) {}
  overload_set(const F1& f1) : F1(f1) {}
  using F1::operator();
};
template <typename F1, typename ... F>
struct overload_set<F1, F ...> :
    public F1, public overload_set<F ...> {
  overload_set(F1&& f1, F&& ... f) :
    F1(std::move(f1)),
    overload_set<F ...>(std::forward<F>(f) ...) {}
  overload_set(const F1& f1, F&& ... f) :
    F1(f1), overload_set<F ...>(std::forward<F>(f) ...) {}
  using F1::operator();
  using overload_set<F ...>::operator();
};
template <typename ... F> auto overload(F&& ... f) {
  return overload_set<F ...>(std::forward<F>(f) ...);
}
```

`overload_set` 是一个变长类模板；在我们可以特化它之前，必须先声明通用模板，但它没有定义。第一个定义是为只有一个 lambda 表达式的特殊情况——`overload_set` 类从 lambda 表达式继承，并将其 `operator()` 添加到其公共接口中。对于 `N` 个 lambda 表达式（`N>1`）的特化从第一个继承，并从剩余的 `N-1` 个 lambda 表达式构成的 `overload_set` 继承。最后，我们有一个辅助函数，可以从任意数量的 lambda 表达式中构建重载集——在我们的情况下，这是一个必要性而不是便利性，因为我们不能显式指定 lambda 表达式的类型，而必须让函数模板推导它们。现在，我们可以从任意数量的 lambda 表达式中构建重载集：

```cpp
int i = 5;
double d = 7.3;
auto l = overload(
  [](int* i) { std::cout << "i=" << *i << std::endl; },
  [](double* d) { std::cout << "d=" << *d << std::endl; }
);
l(&i);    // i=5
l(&d);    // d=5.3
```

这种解决方案并不完美，因为它处理模糊重载的能力不佳。在 C++17 中，我们可以做得更好，这给了我们一个机会来展示一种使用参数包的替代方法，这种方法不需要递归。以下是 C++17 版本：

```cpp
// Example 12b
template <typename ... F>
struct overload_set : public F ... {
  overload_set(F&& ... f) : F(std::forward<F>(f)) ... {}
  using F::operator() ...;    // C++17
};
template <typename ... F> auto overload(F&& ... f) {
  return overload_set<F ...>(std::forward<F>(f) ...);
}
```

可变参数模板不再依赖于部分特化；相反，它直接从参数包继承（这部分实现也在 C++14 中工作，但 `using` 声明需要 C++17）。模板辅助函数与之前相同——它推导所有 lambda 表达式的类型，并从具有这些类型的 `overload_set` 实例化中构建一个对象。lambda 表达式本身通过完美转发传递给基类，在那里它们用于初始化 `overload_set` 对象的所有基对象（lambda 表达式是可移动的）。无需递归或部分特化，这是一个更加紧凑和直接的模板。它的使用与 `overload_set` 的上一个版本相同，但它更好地处理了几乎模糊的重载。

我们还可以弃用模板函数，并使用模板推导指南：

```cpp
// Example 12c
template <typename ... F>
struct overload : public F ... {
  using F::operator() ...;
};
template <typename ... F> // Deduction guide
overload(F&& ... ) -> overload<F ...>;
```

`overload` 模板的使用基本保持不变；注意用于构建对象的括号：

```cpp
int i = 5;
double d = 7.3;
auto l = overload{
  [](int* i) { std::cout << "i=" << *i << std::endl; },
  [](double* d) { std::cout << "d=" << *d << std::endl; },
};
l(&i);    // i=5
l(&d);    // d=5.3
```

在本书的后续章节中，我们将广泛使用 lambda 表达式，当我们需要编写一段代码并将其附加到对象上以便稍后执行时。

接下来，我们将学习一个新的 C++ 特性，它在某种程度上与我们迄今为止试图做的相反：它使模板 *更不通用*。正如我们已经看到的，使用模板过度承诺很容易：我们可以定义模板，其定义在某些情况下无法编译。最好将任何对模板参数的限制作为声明的一部分，让我们看看这是如何实现的。

# 概念

C++20 对 C++ 模板机制进行了重大增强：概念。

在 C++20 中，模板（包括类模板和函数模板）以及非模板函数（通常是类模板的成员）可以使用约束来指定对模板参数的要求。这些约束对于生成更好的错误消息很有用，但在需要根据模板参数的一些属性来选择函数重载或模板特化时，它们确实是不可或缺的。

约束的基本语法相当简单：约束是通过关键字 `requires` 引入的，它可以在函数声明之后或返回类型之前指定（在这本书中，我们两种方式交替使用，以便读者熟悉不同的代码编写风格）。表达式本身通常使用模板参数，并且必须评估为布尔值，例如：

```cpp
// Example 13a
template <typename T> T copy(T&& t)
  requires (sizeof(T) > 1)
{
  return std::forward<T>(t);
}
```

在这里，函数 `copy()` 要求其参数的类型至少有 2 个字节的长度。如果我们尝试用 `char` 参数调用此函数，该调用将无法编译。请注意，如果违反了约束，它就相当于在特定调用中该函数不存在：如果有另一个重载，即使在没有约束的情况下重载是模糊的，它也会被考虑。

这里是一个更复杂（也更实用）的例子：

```cpp
template <typename T1, typename T2>
std::common_type_t<T1, T2> min2(T1&& t1, T2&& t2)
{
  if (t1 < t2) return std::forward<T1>(t1);
  return std::forward<T2>(t2);
}
```

这是一个类似于 `std::min` 的函数，但它接受两个不同类型的参数。这会引发两个潜在问题：首先，返回类型是什么？返回值是两个参数之一，但必须有一个单一的返回类型。我们可以使用来自 `<type_traits>` 头文件的 `std::common_type` 特性作为合理的答案：对于数值类型，它执行通常的类型提升，对于类，如果可能，它将基类转换为派生类，并且它尊重隐式用户指定的转换。但还有一个问题：如果表达式 `t1 < t2` 无法编译，我们会在函数体中得到一个错误。这是不幸的，因为错误很难分析，可能具有误导性：它暗示函数体实现不正确。我们可以通过添加一个静态断言来解决第二个问题：

```cpp
static_assert(sizeof(t1 < t2) > 0);
```

这至少清楚地表明，如果没有匹配的 `operator<()`，我们希望代码无法编译。注意我们表达断言的奇怪方式：表达式 `t1 < t2` 本身通常必须在运行时评估，并且很可能为假。我们需要一个编译时值，而不关心哪个参数较小，只关心它们可以进行比较。因此，我们断言的不是比较的结果，而是这个结果的大小：`sizeof()` 总是编译时值，而在 C++ 中任何东西的大小至少为 1。这个断言唯一可能失败的方式是表达式根本无法编译。

这仍然没有解决问题的另一部分：对参数类型的约束没有被包含在函数的接口中。函数可以在任何两种类型上调用，然后可能编译或不编译。使用 C++20 约束，我们可以将要求从函数体内的隐式（编译失败）或显式（静态断言）错误移动到函数声明中，并使其成为函数接口的一部分：

```cpp
// Example 13b
template <typename T1, typename T2>
std::common_type_t<T1, T2> min2(T1&& t1, T2&& t2)
  requires (sizeof(t1 < t2) > 0)
{
  if (t1 < t2) return std::forward<T1>(t1);
  return std::forward<T2>(t2);
}
```

当您学习构建更复杂的约束时，重要的是要记住，约束表达式必须评估为 `bool` 值；不允许任何转换，这就是为什么一个非常类似的表达式不起作用的原因：

```cpp
template <typename T1, typename T2>
std::common_type_t<T1, T2> min2(T1&& t1, T2&& t2)
  requires (sizeof(t1 < t2));
```

`sizeof()` 的整数值始终非零，并且会转换为 `true`，但在这个上下文中不会。好消息是，我们根本不需要使用 `sizeof()` 诡计来编写约束。还有一种类型的约束表达式，即 *requires 表达式*，它更强大，并且能更清晰地表达我们的意图：

```cpp
// Example 13b
template <typename T1, typename T2>
std::common_type_t<T1, T2> min2(T1&& t1, T2&& t2)
  requires (requires { t1 < t2; });
```

`requires` 表达式以 `requires` 关键字开头，后跟大括号 `{}`；它可以包含任意数量的必须编译的表达式，或者整个 `requires` 表达式的值为假（这些表达式的结果如何并不重要，它们只需是有效的 C++ 表达式）。您还可以使用类型、类型特性和不同类型要求的组合。由于语言的一个特性，`requires` 表达式周围的大括号是可选的，这意味着您可能会看到像 `requires requires { t1 < t2 }` 这样的代码，其中第一个和第二个 `requires` 是完全不同的关键字。

模板类型的要求可能相当复杂；通常，相同的约束在许多不同的模板中都适用。这样的要求集可以命名并定义以供以后使用；这些命名要求被称为概念。每个概念都是在约束中使用时在编译时评估的条件。

约束的语法类似于模板：

```cpp
// Example 13c
template <typename T1, typename T2> concept Comparable =
  requires(T1 t1, T2 t2) { t1 < t2; };
```

我们在这本书中不会详细讲解语法——对于这一点，请使用像 [cppreference.com](http://cppreference.com) 这样的参考资源。一个概念可以用它所包含的要求来代替：

```cpp
template <typename T1, typename T2>
std::common_type_t<T1, T2> min2(T1&& t1, T2&& t2)
  requires Comparable<T1, T2>;
```

限制单个类型的概念也可以用作模板参数占位符。让我们考虑一个 `swap()` 函数的例子。对于整型，有一个技巧允许我们在不使用临时变量的情况下交换两个值。它依赖于位异或操作的性质。为了演示的目的，我们假设，在特定的硬件上，这个版本比通常实现交换的方式更快。我们希望编写一个自动检测类型 T 是否支持异或操作并使用它的 `MySwap(T& a, T& b)` 模板函数；如果可用，否则我们回退到常规的交换。

首先，我们需要一个支持异或操作的类型的概念：

```cpp
// Example 14a,b
template <typename T> concept HasXOR =
  requires(T a, T b) { a ^ b; };
```

该概念有一个`requires`表达式；大括号内的每个表达式都必须编译，否则，概念的要求没有得到满足。

现在，我们可以实现一个基于 XOR 的交换模板。我们可以使用`requires`约束来实现，但有一个更紧凑的方法：

```cpp
template <HasXOR T> void MySwap(T& x, T& y) {
     x = x ^ y;
     y = x ^ y;
     x = x ^ y;
}
```

概念名称`HasXOR`可以用作`typename`关键字来声明模板参数。这限制了我们的`MySwap()`函数只能用于具有`operator^()`的类型。但我们也需要一个通用情况的重载。我们还应该注意，在我们的情况下，“通用”并不意味着“任何”：类型必须支持移动赋值和移动构造。我们需要另一个概念：

```cpp
template <typename T> concept Assignable =
  requires(T a, T b) {
    T(std::move(b));
    b = std::move(a);
  };
```

这是一个非常类似的概念，只不过我们有两个表达式；这两个表达式都必须有效，这个概念才是正确的。

第二个`MySwap()`重载接受所有`Assignable`类型。然而，我们必须明确排除具有 XOR 的类型，否则我们将有模糊的重载。这是一个很好的例子，说明了我们可以将概念作为模板占位符与要求中的概念结合起来：

```cpp
template <Assignable T> void MySwap(T& x, T& y)
  requires (!HasXOR<T>)
{
  T tmp(std::move(x));
  x = std::move(y);
  y = std::move(tmp);
}
```

现在如果可能的话，调用`MySwap()`将选择基于 XOR 的重载，否则，它将使用通用重载（交换不可赋值类型根本无法编译）。

最后，让我们回到本章的第一个例子之一：在“*类模板*”部分中的类模板`ArrayOf2`。回想一下，它有一个成员函数`sum()`，这个函数对模板类型的要求比类中的其他部分要严格得多：它添加数组元素的值。如果元素没有`operator+()`，只要我们不调用`sum()`，就没有问题，但如果我们调用它，就会得到一个语法错误。如果这个函数根本不是类接口的一部分，除非类型支持它，那就更好了。我们可以通过一个约束来实现这一点：

```cpp
// Example 15
template <typename T> class ArrayOf2 {
  public:
  T& operator[](size_t i) { return a_[i]; }
  const T& operator[](size_t i) const { return a_[i]; }
  T sum() const requires (requires (T a, T b) { a + b; }) {
    return a_[0] + a_[1];
  }
  private:
  T a_[2];
};
```

如果表达式`a + b`无法编译，代码的行为就像在类接口中没有声明成员函数`sum()`一样。当然，我们也可以使用一个命名的概念来实现这一点。

我们将在*第七章*中看到更多管理模板参数要求的方法。现在，让我们回顾我们已经学到的内容，并继续使用这些工具来解决常见的 C++问题。

# 摘要

模板、变长模板和 lambda 表达式都是 C++的强大功能，它们在用法上简单，但在细节上丰富。本章中的示例应该有助于为读者准备本书的后续章节，在这些章节中，我们将使用这些技术使用现代 C++语言工具来实现经典和新型设计模式。希望学习如何充分利用这些复杂而强大的工具的读者可以参考其他专门教授这些主题的书籍，其中一些可以在本章末尾找到。

现在，读者已经准备好学习常见的 C++惯用法，从下一章中表达内存所有权的惯用法开始。

# 问题

1.  类型与模板之间有什么区别？

1.  C++ 有哪些类型的模板？

1.  C++ 模板有哪些类型的模板参数？

1.  模板特化与模板实例化之间有什么区别？

1.  你如何访问变长模板的参数包？

1.  lambda 表达式有什么用途？

1.  概念是如何细化模板接口的？

# 进一步阅读

+   *C++* 基础知识：[`www.packtpub.com/product/c-fundamentals`](https://www.packtpub.com/product/c-fundamentals)

+   *C++ 数据结构与算法*：[`www.packtpub.com/product/c-data-structures-and-algorithms`](https://www.packtpub.com/product/c-data-structures-and-algorithms)

+   *精通 C++ 编程*：[`www.packtpub.com/product/mastering-c-programming`](https://www.packtpub.com/product/mastering-c_programming)
