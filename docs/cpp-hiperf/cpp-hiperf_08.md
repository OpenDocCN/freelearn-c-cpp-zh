# 第八章：8

# 编译时编程

C++具有在编译时评估表达式的能力，这意味着值在程序执行时已经计算出来。尽管自 C++98 以来就一直可以进行元编程，但由于其复杂的基于模板的语法，最初非常复杂。随着`constexpr`、`if constexpr`的引入，以及最近的 C++ *概念*，元编程变得更类似于编写常规代码。

本章将简要介绍 C++中的编译时表达式求值以及它们如何用于优化。

我们将涵盖以下主题：

+   使用 C++模板进行元编程以及如何在 C++20 中编写缩写函数模板

+   在编译时使用类型特征检查和操作类型

+   编译器评估的常量表达式

+   C++20 概念以及如何使用它们为我们的模板参数添加约束

+   元编程的一些真实例子

我们将从介绍模板元编程开始。

# 介绍模板元编程

在编写常规 C++代码时，最终会将其转换为机器代码。另一方面，**元编程**允许我们编写能够将自身转换为常规 C++代码的代码。更一般地说，元编程是一种技术，我们编写能够转换或生成其他代码的代码。通过使用元编程，我们可以避免重复使用仅基于我们使用的数据类型略有不同的代码，或者通过预先计算在最终程序执行之前就可以知道的值来最小化运行时成本。没有什么能阻止我们使用其他语言生成 C++代码。例如，我们可以通过广泛使用预处理器宏或编写一个生成或修改 C++文件的 Python 脚本来进行元编程：

![](img/B15619_08_01.png)

图 8.1：一个元程序生成将被编译成机器代码的常规 C++代码

尽管我们可以使用任何语言来生成常规代码，但是使用 C++，我们有特权在语言本身内部使用**模板**和**常量表达式**编写元程序。C++编译器可以执行我们的元程序，并生成编译器将进一步转换为机器代码的常规 C++代码。

在 C++中直接使用模板和常量表达式进行元编程，而不是使用其他技术，有许多优势：

+   我们不必解析 C++代码（编译器会为我们做这个工作）。

+   在使用 C++模板元编程时，对分析和操作 C++类型有很好的支持。

+   元程序的代码和常规非通用代码混合在 C++源代码中。有时，这可能使人难以理解哪些部分分别在运行时和编译时执行。然而，总的来说，这是使 C++元编程有效使用的一个非常重要的方面。

在其最简单和最常见的形式中，C++中的模板元编程用于生成接受不同类型的函数、值和类。当编译器使用模板生成类或函数时，称模板被**实例化**。编译器通过**评估**常量表达式来生成常量值：

![](img/B15619_08_02.png)

图 8.2：C++中的编译时编程。将生成常规 C++代码的元程序是用 C++本身编写的。

这是一个相对简化的观点；没有什么规定 C++编译器必须以这种方式执行转换。然而，将 C++元编程视为在这两个不同阶段进行的是很有用的：

+   初始阶段，模板和常量表达式生成函数、类和常量值的常规 C++代码。这个阶段通常被称为**常量评估**。

+   第二阶段，编译器最终将常规 C++代码编译成机器代码。

在本章后面，我将把从元编程生成的 C++代码称为*常规 C++代码*。

在使用元编程时，重要的是要记住它的主要用例是制作出色的库，并因此隐藏用户代码中的复杂构造/优化。请注意，无论代码的内部多么复杂，都很重要将其隐藏在良好的接口后面，以便用户代码库易于阅读和使用。

让我们继续创建我们的第一个用于生成函数和类的模板。

## 创建模板

让我们看一个简单的`pow()`函数和一个`Rectangle`类。通过使用**类型模板参数**，`pow()`函数和`Rectangle`类可以与任何整数或浮点类型一起使用。没有模板，我们将不得不为每种基本类型创建一个单独的函数/类。

编写元编程代码可能非常复杂；使其变得更容易的一点是想象预期的常规 C++代码的意图。

下面是一个简单函数模板的示例：

```cpp
// pow_n accepts any number type 
template <typename T> 
auto pow_n(const T& v, int n) { 
  auto product = T{1}; 
  for (int i = 0; i < n; ++i) { 
    product *= v; 
  }
  return product; 
} 
```

使用此函数将生成一个返回类型取决于模板参数类型的函数：

```cpp
auto x = pow_n<float>(2.0f, 3); // x is a float 
auto y = pow_n<int>(3, 3);      // y is an int 
```

显式模板参数类型（在这种情况下为`float`和`int`）可以（最好）省略，而编译器可以自行解决这个问题。这种机制称为**模板参数推断**，因为编译器*推断*模板参数。以下示例将导致与先前显示的相同的模板实例化：

```cpp
auto x = pow_n(2.0f, 3);  // x is a float 
auto y = pow_n(3, 3);     // y is an int 
```

相应地，可以定义一个简单的类模板如下：

```cpp
// Rectangle can be of any type 
template <typename T> 
class Rectangle { 
public: 
  Rectangle(T x, T y, T w, T h) : x_{x}, y_{y}, w_{w}, h_{h} {} 
  auto area() const { return w_ * h_; } 
  auto width() const { return w_; } 
  auto height() const { return h_; } 
private:
  T x_{}, y_{}, w_{}, h_{}; 
}; 
```

当使用类模板时，我们可以明确指定模板应为其生成代码的类型，如下所示：

```cpp
auto r1 = Rectangle<float>{2.0f, 2.0f, 4.0f, 4.0f}; 
```

但也可以从**类模板参数推断**（**CTAD**）中受益，并让编译器为我们推断参数类型。以下代码将实例化一个`Rectangle<int>`：

```cpp
auto r2 = Rectangle{-2, -2, 4, 4};   // Rectangle<int> 
```

然后，函数模板可以接受一个`Rectangle`对象，其中矩形的尺寸是使用任意类型`T`定义的，如下所示：

```cpp
template <typename T> 
auto is_square(const Rectangle<T>& r) { 
  return r.width() == r.height(); 
} 
```

类型模板参数是最常见的模板参数。接下来，您将看到如何使用数值参数而不是类型参数。

## 使用整数作为模板参数

除了一般类型，模板还可以是其他类型，例如整数类型和浮点类型。在下面的示例中，我们将在模板中使用`int`，这意味着编译器将为每个唯一的整数传递的模板参数生成一个新函数：

```cpp
template <int N, typename T> 
auto const_pow_n(const T& v) { 
  auto product = T{1}; 
  for (int i = 0; i < N; ++i) { 
    product *= v; 
  }
  return product; 
} 
```

以下代码将强制编译器实例化两个不同的函数：一个平方值，一个立方值：

```cpp
auto x2 = const_pow_n<2>(4.0f);   // Square
auto x3 = const_pow_n<3>(4.0f);   // Cube 
```

请注意模板参数`N`和函数参数`v`之间的差异。对于每个`N`的值，编译器都会生成一个新函数。但是，`v`作为常规参数传递，因此不会导致生成新函数。

## 提供模板的特化

默认情况下，每当我们使用新参数的模板时，编译器将生成常规的 C++代码。但也可以为模板参数的某些值提供自定义实现。例如，假设我们希望在使用整数并且`N`的值为`2`时，提供我们的`const_pow_n()`函数的常规 C++代码。我们可以为这种情况编写一个**模板特化**，如下所示：

```cpp
template<>
auto const_pow_n<2, int>(const int& v) {
  return v * v;
} 
```

对于函数模板，当编写特化时，我们需要固定*所有*模板参数。例如，不可能只指定`N`的值，而让类型参数`T`未指定。但是，对于类模板，可以只指定模板参数的子集。这称为**部分模板特化**。编译器将首先选择最具体的模板。

我们不能对函数应用部分模板特化的原因是函数可以重载（而类不能）。如果允许混合重载和部分特化，那将很难理解。

## 编译器如何处理模板函数

当编译器处理模板函数时，它会构造一个展开了模板参数的常规函数。以下代码将使编译器生成常规函数，因为它使用了模板：

```cpp
auto a = pow_n(42, 3);          // 1\. Generate new function
auto b = pow_n(42.f, 2);        // 2\. Generate new function
auto c = pow_n(17.f, 5);        // 3.
auto d = const_pow_n<2>(42.f);  // 4\. Generate new function
auto e = const_pow_n<2>(99.f);  // 5.
auto f = const_pow_n<3>(42.f);  // 6\. Generate new function 
```

因此，当编译时，与常规函数不同，编译器将为每组唯一的*模板参数*生成新函数。这意味着它相当于手动创建了四个不同的函数，看起来像这样：

```cpp
auto pow_n__float(float v, int n) {/*...*/}   // Used by: 1
auto pow_n__int(int v, int n) {/*...*/}       // Used by: 2 and 3
auto const_pow_n__2_float (float v) {/*...*/} // Used by: 4 and 5
auto const_pow_n__3_float(float v) {/*...*/}  // Used by: 6 
```

这对于理解元编程的工作原理非常重要。模板代码生成非模板化的 C++代码，然后作为常规代码执行。如果生成的 C++代码无法编译，错误将在编译时被捕获。

## 缩写函数模板

C++20 引入了一种新的缩写语法，用于编写函数模板，采用了通用 lambda 使用的相同风格。通过使用`auto`作为函数参数类型，我们实际上创建的是一个函数模板，而不是一个常规函数。回想一下我们最初的`pow_n()`模板，它是这样声明的：

```cpp
template <typename T>
auto pow_n(const T& v, int n) { 
  // ... 
```

使用缩写的函数模板语法，我们可以使用`auto`来声明它：

```cpp
auto pow_n(const auto& v, int n) { // Declares a function template
  // ... 
```

这两个版本之间的区别在于缩写版本没有变量`v`的显式占位符。由于我们在实现中使用了占位符`T`，这段代码将不幸地无法编译：

```cpp
auto pow_n(const auto& v, int n) {
  auto product = T{1}; // Error: What is T?
  for (int i = 0; i < n; ++i) { 
    product *= v; 
  } 
  return product;
} 
```

为了解决这个问题，我们可以使用`decltype`指定符。

## 使用 decltype 接收变量的类型

`decltype`指定符用于检索变量的类型，并且在没有显式类型名称可用时使用。

有时，我们需要一个显式的类型占位符，但没有可用的，只有变量名。这在我们之前实现`pow_n()`函数时发生过，当使用缩写的函数模板语法时。

让我们通过修复`pow_n()`的实现来看一个使用`decltype`的例子：

```cpp
auto pow_n(const auto& v, int n) {
  auto product = decltype(v){1};   // Instead of T{1}
  for (int i = 0; i < n; ++i) { product *= v; } 
  return product;
} 
```

尽管这段代码编译并工作，但我们有点幸运，因为`v`的类型实际上是一个`const`引用，而不是我们想要的变量`product`的类型。我们可以通过使用从左到右的声明样式来解决这个问题。但是，试图将定义产品的行重写为看起来相同的东西会揭示一个问题：

```cpp
auto pow_n(const auto& v, int n) {
  decltype(v) product{1};
  for (int i = 0; i < n; ++i) { product *= v; } // Error!
  return product;
} 
```

现在，我们得到了一个编译错误，因为`product`是一个`const`引用，可能无法分配新值。

我们真正想要的是从变量`v`的类型中去掉`const`引用，当定义变量`product`时。我们可以使用一个方便的模板`std::remove_cvref`来实现这个目的。我们的`product`的定义将如下所示：

```cpp
typename std::remove_cvref<decltype(v)>::type product{1}; 
```

哦！在这种特殊情况下，也许最好还是坚持最初的`template <typename T>`语法。但现在，您已经学会了在编写通用 C++代码时如何使用`std::remove_cvref`和`decltype`，这是一个常见的模式。

在 C++20 之前，在通用 lambda 的主体中经常看到`decltype`。然而，现在可以通过向通用 lambda 添加显式模板参数来避免相当不方便的`decltype`：

```cpp
auto pow_n = []<class T>(const T& v, int n) { 
  auto product = T{1};
  for (int i = 0; i < n; ++i) { product *= v; }
  return product;
}; 
```

在 lambda 的定义中，我们写`<class T>`以获取一个可以在函数体内使用的参数类型的标识符。

也许需要一些时间来习惯使用`decltype`和操纵类型的工具。也许`std::remove_cvref`一开始看起来有点神秘。它是`<type_traits>`头文件中的一个模板，我们将在下一节中进一步了解它。

# 类型特征

在进行模板元编程时，您可能经常会发现自己处于需要在编译时获取有关您正在处理的类型的信息的情况。在编写常规（非泛型）C++代码时，我们使用完全了解的具体类型，但在编写模板时情况并非如此；具体类型直到编译器实例化模板时才确定。类型特征允许我们提取有关我们模板处理的类型的信息，以生成高效和正确的 C++代码。

为了提取有关模板类型的信息，标准库提供了一个类型特征库，该库在`<type_traits>`头文件中可用。所有类型特征都在编译时评估。

## 类型特征类别

有两类类型特征：

+   返回关于类型信息的类型特征，作为布尔值或整数值。

+   返回新类型的类型特征。这些类型特征也被称为元函数。

第一类返回`true`或`false`，取决于输入，并以`_v`结尾（代表值）。

`_v`后缀是在 C++17 中添加的。如果您的库实现不提供类型特征的`_v`后缀，则可以使用旧版本`std::is_floating_point<float>::value`。换句话说，删除`_v`扩展并在末尾添加`::value`。

以下是使用类型特征对基本类型进行编译时类型检查的一些示例：

```cpp
auto same_type = std::is_same_v<uint8_t, unsigned char>; 
auto is_float_or_double = std::is_floating_point_v<decltype(3.f)>; 
```

类型特征也可以用于用户定义的类型：

```cpp
class Planet {};
class Mars : public Planet {};
class Sun {};
static_assert(std::is_base_of_v<Planet, Mars>);
static_assert(!std::is_base_of_v<Planet, Sun>); 
```

类型特征的第二类返回一个新类型，并以`_t`结尾（代表类型）。当处理指针和引用时，这些类型特征转换（或元函数）非常方便：

```cpp
// Examples of type traits which transforms types
using value_type = std::remove_pointer_t<int*>;  // -> int
using ptr_type = std::add_pointer_t<float>;      // -> float* 
```

我们之前使用的类型特征`std::remove_cvref`也属于这个类别。它从类型中移除引用部分（如果有）以及`const`和`volatile`限定符。`std::remove_cvref`是在 C++20 中引入的。在那之前，通常使用`std::decay`来执行此任务。

## 使用类型特征

如前所述，所有类型特征都在编译时评估。例如，以下函数如果值大于或等于零则返回`1`，否则返回`-1`，对于无符号整数可以立即返回`1`，如下所示：

```cpp
template<typename T>
auto sign_func(T v) -> int {
  if (std::is_unsigned_v<T>) { 
    return 1; 
  } 
  return v < 0 ? -1 : 1; 
} 
```

由于类型特征在编译时评估，因此当使用无符号整数和有符号整数调用时，编译器将生成下表中显示的代码：

| 与无符号整数一起使用... | ...生成的函数： |
| --- | --- |

|

```cpp
auto unsigned_v = uint32_t{42};
auto sign = sign_func(unsigned_v); 
```

|

```cpp
int sign_func(uint32_t v) {
  if (true) { 
    return 1; 
  } 
  return v < 0 ? -1 : 1; 
} 
```

|

| 与有符号整数一起使用... | ...生成的函数： |
| --- | --- |

|

```cpp
auto signed_v = int32_t{-42}; 
auto sign = sign_func(signed_v); 
```

|

```cpp
int sign_func(int32_t v) {
  if (false) { 
    return 1; 
  } 
  return v < 0 ? -1 : 1; 
} 
```

|

表 8.1：基于我们传递给`sign_func()`的类型（在左列），编译器生成不同的函数（在右列）。

接下来，让我们谈谈常量表达式。

# 使用常量表达式进行编程

使用`constexpr`关键字前缀的表达式告诉编译器应在编译时评估该表达式：

```cpp
constexpr auto v = 43 + 12; // Constant expression 
```

`constexpr`关键字也可以与函数一起使用。在这种情况下，它告诉编译器某个函数打算在编译时评估，如果满足所有允许进行编译时评估的条件，则会在运行时执行，就像常规函数一样。

`constexpr`函数有一些限制；不允许执行以下操作：

+   处理本地静态变量

+   处理`thread_local`变量

+   调用任何函数，本身不是`constexpr`函数

使用`constexpr`关键字，编写编译时评估的函数与编写常规函数一样容易，因为它的参数是常规参数而不是模板参数。

考虑以下`constexpr`函数：

```cpp
constexpr auto sum(int x, int y, int z) { return x + y + z; } 
```

让我们这样调用函数：

```cpp
constexpr auto value = sum(3, 4, 5); 
```

由于`sum()`的结果用于常量表达式，并且其所有参数都可以在编译时确定，因此编译器将生成以下常规的 C++代码：

```cpp
const auto value = 12; 
```

然后像往常一样将其编译成机器代码。换句话说，编译器评估`constexpr`函数并生成常规的 C++代码，其中计算结果。

如果我们调用`sum()`并将结果存储在未标记为`constexpr`的变量中，编译器可能（很可能）在编译时评估`sum()`：

```cpp
auto value = sum(3, 4, 5); // value is not constexpr 
```

总之，如果从常量表达式调用`constexpr`函数，并且其所有参数都是常量表达式，那么它保证在编译时评估。

## 运行时上下文中的 Constexpr 函数

在前面的例子中，编译器在编译时已知的值（`3`、`4`、`5`）是已知的，但是`constexpr`函数如何处理直到运行时才知道值的变量？如前一节所述，`constexpr`是编译器的指示，表明在某些条件下，函数可以在编译时评估。如果直到运行时调用时才知道值的变量，它们将像常规函数一样被评估。

在下面的例子中，`x`、`y`和`z`的值是在运行时由用户提供的，因此编译器无法在编译时计算总和：

```cpp
int x, y, z; 
std::cin >> x >> y >> z;      // Get user input
auto value = sum(x, y, z); 
```

如果我们根本不打算在运行时使用`sum()`，我们可以通过将其设置为立即函数来禁止这种用法。

## 使用`consteval`声明立即函数

`constexpr`函数可以在运行时或编译时调用。如果我们想限制函数的使用，使其只在编译时调用，我们可以使用关键字`consteval`而不是`constexpr`。假设我们想禁止在运行时使用`sum()`。使用 C++20，我们可以通过以下代码实现：

```cpp
consteval auto sum(int x, int y, int z) { return x + y + z; } 
```

使用`consteval`声明的函数称为**立即函数**，只能生成常量。如果我们想调用`sum()`，我们需要在常量表达式中调用它，否则编译将失败：

```cpp
constexpr auto s = sum(1, 2, 3); // OK
auto x = 10;
auto s = sum(x, 2, 3);           // Error, expression is not const 
```

如果我们尝试在编译时使用参数不明确的`sum()`，编译器也会报错：

```cpp
int x, y, z; 
std::cin >> x >> y >> z; 
constexpr auto s = sum(x, y, z); // Error 
```

接下来讨论`if` `constexpr`语句。

## if constexpr 语句

`if constexpr`语句允许模板函数在同一函数中在编译时评估不同的作用域（也称为编译时多态）。看看下面的例子，其中一个名为`speak()`的函数模板尝试根据类型区分成员函数：

```cpp
struct Bear { auto roar() const { std::cout << "roar\n"; } }; 
struct Duck { auto quack() const { std::cout << "quack\n"; } }; 
template <typename Animal> 
auto speak(const Animal& a) { 
  if (std::is_same_v<Animal, Bear>) { a.roar(); } 
  else if (std::is_same_v<Animal, Duck>) { a.quack(); } 
} 
```

假设我们编译以下行：

```cpp
auto bear = Bear{};
speak(bear); 
```

然后编译器将生成一个类似于这样的`speak()`函数：

```cpp
auto speak(const Bear& a) {
  if (true) { a.roar(); }
  else if (false) { a.quack(); } // This line will not compile
} 
```

如您所见，编译器将保留对成员函数`quack()`的调用，然后由于`Bear`不包含`quack()`成员函数而无法编译。这甚至会发生在`quack()`成员函数由于`else if (false)`语句而永远不会被执行的情况下。

为了使`speak()`函数无论类型如何都能编译，我们需要告诉编译器，如果`if`语句为`false`，我们希望完全忽略作用域。方便的是，这正是`if constexpr`所做的。

以下是我们如何编写`speak()`函数，以便处理`Bear`和`Duck`，即使它们没有共同的接口：

```cpp
template <typename Animal> 
auto speak(const Animal& a) { 
  if constexpr (std::is_same_v<Animal, Bear>) { a.roar(); } 
  else if constexpr (std::is_same_v<Animal, Duck>) { a.quack(); } 
} 
```

当使用`Animal == Bear`调用`speak()`时，如下所示：

```cpp
auto bear = Bear{};
speak(bear); 
```

编译器生成以下函数：

```cpp
auto speak(const Bear& animal) { animal.roar(); } 
```

当使用`Animal == Duck`调用`speak()`时，如下所示：

```cpp
auto duck = Duck{};
speak(duck); 
```

编译器生成以下函数：

```cpp
auto speak(const Duck& animal) { animal.quack(); } 
```

如果使用任何其他原始类型调用`speak()`，例如`Animal == int`，如下所示：

```cpp
speak(42); 
```

编译器生成一个空函数：

```cpp
auto speak(const int& animal) {} 
```

与常规的`if`语句不同，编译器现在能够生成多个不同的函数：一个使用`Bear`，另一个使用`Duck`，如果类型既不是`Bear`也不是`Duck`，则生成最后一个。如果我们想让这第三种情况成为编译错误，我们可以通过添加一个带有`static_assert`的`else`语句来实现：

```cpp
template <typename Animal> 
auto speak(const Animal& a) { 
  if constexpr (std::is_same_v<Animal, Bear>) { a.roar(); } 
  else if constexpr (std::is_same_v<Animal, Duck>) { a.quack(); }
  else { static_assert(false); } // Trig compilation error
} 
```

我们稍后会更多地讨论`static_assert`的用处。

如前所述，这里使用`constexpr`的方式可以称为编译时多态。那么，它与运行时多态有什么关系呢？

### 与运行时多态的比较

顺便说一句，如果我们使用传统的运行时多态来实现前面的例子，使用继承和虚函数来实现相同的功能，实现将如下所示：

```cpp
struct AnimalBase {
  virtual ~AnimalBase() {}
  virtual auto speak() const -> void {}
};
struct Bear : public AnimalBase {
  auto roar() const { std::cout << "roar\n"; } 
  auto speak() const -> void override { roar(); }
};
struct Duck : public AnimalBase {
  auto quack() const { std::cout << "quack\n"; }
  auto speak() const -> void override { quack(); }
}; 
auto speak(const AnimalBase& a) { 
  a.speak();
} 
```

对象必须使用指针或引用进行访问，并且类型在*运行时*推断，这导致性能损失与编译时版本相比，其中应用程序执行时一切都是可用的。下面的图像显示了 C++中两种多态类型之间的区别：

![](img/B15619_08_03.png)

图 8.3：运行时多态由虚函数支持，而编译时多态由函数/操作符重载和 if constexpr 支持。

现在，我们将继续看看如何使用`if constexpr`来做一些更有用的事情。

### 使用 if constexpr 的通用模数函数示例

这个例子将向您展示如何使用`if constexpr`来区分运算符和全局函数。在 C++中，`%`运算符用于获取整数的模，而`std::fmod()`用于浮点类型。假设我们想要将我们的代码库泛化，并创建一个名为`generic_mod()`的通用模数函数。

如果我们使用常规的`if`语句来实现`generic_mod()`，如下所示：

```cpp
template <typename T> 
auto generic_mod(const T& v, const T& n) -> T {
  assert(n != 0);
  if (std::is_floating_point_v<T>) { return std::fmod(v, n); }
  else { return v % n; }
} 
```

如果以`T == float`调用它，它将失败，因为编译器将生成以下函数，这将无法编译通过：

```cpp
auto generic_mod(const float& v, const float& n) -> float {
  assert(n != 0);
  if (true) { return std::fmod(v, n); }
  else { return v % n; } // Will not compile
} 
```

尽管应用程序无法到达它，编译器将生成`return v % n;`这一行，这与`float`不兼容。编译器不在乎应用程序是否能到达它——因为它无法为其生成汇编代码，所以它将无法编译通过。

与前面的例子一样，我们将`if`语句更改为`if constexpr`语句：

```cpp
template <typename T> 
auto generic_mod(const T& v, const T& n) -> T { 
  assert(n != 0);
  if constexpr (std::is_floating_point_v<T>) {
    return std::fmod(v, n);
  } else {                 // If T is a floating point,
    return v % n;          // this code is eradicated
  }
} 
```

现在，当使用浮点类型调用函数时，它将生成以下函数，其中`v % n`操作被消除：

```cpp
auto generic_mod(const float& v, const float& n) -> float { 
  assert(n != 0);
  return std::fmod(v, n); 
} 
```

运行时的`assert()`告诉我们，如果第二个参数为 0，我们不能调用这个函数。

## 在编译时检查编程错误

Assert 语句是验证代码库中调用者和被调用者之间不变性和契约的简单但非常强大的工具（见*第二章*，*Essential C++ Techniques*）。使用`assert()`可以在执行程序时检查编程错误。但我们应该始终努力尽早检测错误，如果有常量表达式，我们可以使用`static_assert()`在编译程序时捕获编程错误。

### 使用 assert 在运行时触发错误

回顾`pow_n()`的模板版本。假设我们想要阻止它使用负指数（`n`值）进行调用。在运行时版本中，其中`n`是一个常规参数，我们可以添加一个运行时断言来阻止这种情况：

```cpp
template <typename T> 
auto pow_n(const T& v, int n) { 
  assert(n >= 0); // Only works for positive numbers 
  auto product = T{1}; 
  for (int i = 0; i < n; ++i) {
    product *= v; 
  }
  return product; 
} 
```

如果函数被调用时`n`的值为负数，程序将中断并告知我们应该从哪里开始寻找错误。这很好，但如果我们能在编译时而不是运行时跟踪这个错误会更好。

### 使用`static_assert`在编译时触发错误

如果我们对模板版本做同样的事情，我们可以利用`static_assert()`。与常规的 assert 不同，`static_assert()`声明如果条件不满足将拒绝编译。因此，最好是在编译时中断构建，而不是在运行时中断程序。在下面的例子中，如果模板参数`N`是一个负数，`static_assert()`将阻止函数编译：

```cpp
template <int N, typename T>
auto const_pow_n(const T& v) {
  static_assert(N >= 0, "N must be positive"); 
  auto product = T{1}; 
  for (int i = 0; i < N; ++i) { 
    product *= v; 
  } 
  return product; 
}
auto x = const_pow_n<5>(2);  // Compiles, N is positive
auto y = const_pow_n<-1>(2); // Does not compile, N is negative 
```

换句话说，对于常规变量，编译器只知道类型，不知道它包含什么。对于编译时值，编译器既知道类型又知道值。这使得编译器能够计算其他编译时值。

我们可以（应该）使用无符号整数而不是使用`int`并断言它是非负的。在这个例子中，我们只是使用有符号的`int`来演示`assert()`和`static_assert()`的使用。

使用编译时断言是一种在编译时检查约束的方法。这是一个简单但非常有用的工具。在过去几年中，C++的编译时编程支持取得了一些非常令人兴奋的进展。现在，我们将继续介绍 C++20 中的一个最重要的特性，将约束检查提升到一个新的水平。

# 约束和概念

到目前为止，我们已经涵盖了写 C++元编程的一些重要技术。您已经看到模板如何利用类型特征库为我们生成具体的类和函数。此外，您已经看到了`constexpr`、`consteval`和`if constexpr`的使用可以帮助我们将计算从运行时移动到编译时。通过这种方式，我们可以在编译时检测编程错误，并编写具有较低运行时成本的程序。这很棒，但在编写和使用 C++中的通用代码方面仍有很大的改进空间。我们尚未解决的一些问题包括：

1.  接口太通用。当使用具有任意类型的模板时，很难知道该类型的要求是什么。如果我们只检查模板接口，这使得模板难以使用。相反，我们必须依赖文档或深入到模板的实现中。

1.  类型错误由编译器晚期捕获。编译器最终会在编译常规 C++代码时检查类型，但错误消息通常很难解释。相反，我们希望在实例化阶段捕获类型错误。

1.  无约束的模板参数使元编程变得困难。到目前为止，在本章中我们编写的代码都使用了无约束的模板参数，除了一些静态断言。这对于小例子来说是可以管理的，但如果我们能够像类型系统帮助我们编写正确的非通用 C++代码一样，获得更有意义的类型，那么编写和推理我们的元编程将会更容易。

1.  使用`if constexpr`可以进行条件代码生成（编译时多态），但在较大规模上很快变得难以阅读和编写。

正如您将在本节中看到的，C++概念以一种优雅而有效的方式解决了这些问题，引入了两个新关键字：`concept`和`requires`。在探讨约束和概念之前，我们将花一些时间考虑没有概念的模板元编程的缺点。然后，我们将使用约束和概念来加强我们的代码。

## Point2D 模板的无约束版本

假设我们正在编写一个处理二维坐标系的程序。我们有一个类模板，表示具有`x`和`y`坐标的点，如下所示：

```cpp
template <typename T>
class Point2D {
public:
  Point2D(T x, T y) : x_{x}, y_{y} {}
  auto x() { return x_; }
  auto y() { return y_; }
  // ...
private:
  T x_{};
  T y_{};
}; 
```

假设我们需要找到两点**p1**和**p2**之间的欧几里德距离，如下所示：

![](img/B15619_08_04.png)

图 8.4：找到 p1 和 p2 之间的欧几里得距离

为了计算距离，我们实现了一个接受两个点并使用勾股定理的自由函数（这里实际的数学并不重要）：

```cpp
auto dist(auto p1, auto p2) {
  auto a = p1.x() - p2.x();
  auto b = p1.y() - p2.y();
  return std::sqrt(a*a + b*b);
} 
```

一个小的测试程序验证了我们可以用整数实例化`Point2D`模板，并计算两点之间的距离：

```cpp
int main() {
  auto p1 = Point2D{2, 2};
  auto p2 = Point2D{6, 5};
  auto d = dist(p1, p2);
  std::cout << d;
} 
```

这段代码编译和运行都很好，并在控制台输出`5`。

### 通用接口和糟糕的错误消息

在继续之前，让我们稍微偏离一下，对函数模板`dist()`进行一段时间的反思。假设我们无法轻松访问`dist()`的实现，只能读取接口：

```cpp
auto dist(auto p1, auto p2) // Interface part 
```

我们可以说返回类型和`p1`和`p2`的类型有什么？实际上几乎没有——因为`p1`和`p2`完全*未受约束*，`dist()`的接口对我们来说没有透露任何信息。这并不意味着我们可以将任何东西传递给`dist()`，因为最终生成的常规 C++代码必须编译。

例如，如果我们尝试用两个整数而不是`Point2D`对象来实例化我们的`dist()`模板，就像这样：

```cpp
 auto d = dist(3, 4); 
```

编译器将很乐意生成一个常规的 C++函数，类似于这样：

```cpp
auto dist(int p1, int p2) {
  auto a = p1.x() – p2.x();  // Will generate an error:
  auto b = p1.y() – p2.y();  // int does not have x() and y()
  return std::sqrt(a*a + b*b);
} 
```

当编译器检查常规的 C++代码时，错误将在稍后被捕获。当尝试用两个整数实例化`dist()`时，Clang 生成以下错误消息：

```cpp
error: member reference base type 'int' is not a structure or union
auto a = p1.x() – p2.y(); 
```

这个错误消息是指`dist()`的*实现*，这是调用函数`dist()`的调用者不需要知道的东西。这是一个微不足道的例子，但是尝试解释由于向复杂的模板库提供错误类型而引起的错误消息可能是一个真正的挑战。

更糟糕的是，如果我们真的很不幸，通过提供根本没有意义的类型来完成整个编译。在这种情况下，我们正在用`const char*`实例化`Point2D`：

```cpp
int main() {
  auto from = Point2D{"2.0", "2.0"}; // Ouch!
  auto to = Point2D{"6.0", "5.0"};   // Point2D<const char*>
  auto d = dist(from, to);
  std::cout << d;
} 
```

它编译并运行，但输出可能不是我们所期望的。我们希望在过程的早期阶段捕获这些类型的错误，这是我们可以通过使用约束和概念来实现的，如下图所示：

![](img/B15619_08_05.png)

图 8.5：使用约束和概念可以在实例化阶段捕获类型错误

稍后，您将看到如何使此代码更具表现力，以便更容易正确使用并更难滥用。我们将通过向我们的代码添加概念和约束来实现这一点。但首先，我将快速概述如何定义和使用概念。

## 约束和概念的语法概述

本节是对约束和概念的简要介绍。我们不会在本书中完全覆盖它们，但我会为您提供足够的材料来提高生产力。

### 定义新概念

使用您已经熟悉的类型特征，可以轻松地定义新概念。以下示例使用关键字`concept`定义了概念`FloatingPoint`：

```cpp
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>; 
```

赋值表达式的右侧是我们可以指定类型`T`的约束的地方。还可以使用`||`（逻辑或）和`&&`（逻辑与）来组合多个约束。以下示例使用`||`将浮点数和整数组合成`Number`概念：

```cpp
template <typename T>
concept Number = FloatingPoint<T> || std::is_integral_v<T>; 
```

您将注意到，还可以使用右侧已定义的概念构建概念。标准库包含一个`<concepts>`头文件，其中定义了许多有用的概念，例如`std::floating_point`（我们应该使用它而不是定义自己的）。

此外，我们可以使用`requires`关键字来添加一组语句，这些语句应该添加到我们的概念定义中。例如，这是来自 Ranges 库的概念`std::range`的定义：

```cpp
template<typename T>
concept range = requires(T& t) {
  ranges::begin(t);
  ranges::end(t);
}; 
```

简而言之，这个概念说明了范围是我们可以传递给`std::ranges::begin()`和`std::ranges::end()`的东西。

可以编写比这更复杂的`requires`子句，稍后您将看到更多内容。

### 使用概念约束类型

我们可以通过使用`requires`关键字向模板参数类型添加约束。以下模板只能使用`std::integral`概念实例化整数类型的参数：

```cpp
template <typename T>
requires std::integral<T>
auto mod(T v, T n) { 
  return v % n;
} 
```

在定义类模板时也可以使用相同的技术：

```cpp
template <typename T>
requires std::integral<T>
struct Foo {
  T value;
}; 
```

另一种语法允许我们以更紧凑的方式编写，通过直接用概念替换`typename`：

```cpp
template <std::integral T>
auto mod(T v, T n) { 
  return v % n;
} 
```

这种形式也可以用于类模板：

```cpp
template <std::integral T>
struct Foo {
  T value;
}; 
```

如果我们想在定义函数模板时使用缩写的函数模板形式，我们可以在`auto`关键字前面添加概念：

```cpp
auto mod(std::integral auto v, std::integral auto n) {
  return v % n;
} 
```

返回类型也可以通过使用概念来约束：

```cpp
std::integral auto mod(std::integral auto v, std::integral auto n) {
  return v % n;
} 
```

正如你所看到的，有许多方法可以指定相同的事情。缩写形式与概念的结合确实使有限函数模板的阅读和编写变得非常容易。C++概念的另一个强大特性是以清晰和表达性的方式重载函数。

### 函数重载

回想一下我们之前使用`if constexpr`实现的`generic_mod()`函数。它看起来像这样：

```cpp
template <typename T> 
auto generic_mod(T v, T n) -> T { 
  if constexpr (std::is_floating_point_v<T>) {
    return std::fmod(v, n);
  } else {
    return v % n;
  } 
} 
```

通过使用概念，我们可以重载一个函数模板，类似于我们如果编写了一个常规的 C++函数：

```cpp
template <std::integral T>
auto generic_mod(T v, T n) -> T {             // Integral version
  return v % n;
}
template <std::floating_point T>
auto generic_mod(T v, T n) -> T {             // Floating point version
  return std::fmod(v, n);
} 
```

有了你对约束和概念的新知识，现在是时候回到我们的`Point2D`模板的例子，看看它如何改进。

## Point2D 模板的约束版本

现在你知道如何定义和使用概念了，让我们通过编写一个更好的模板`Point2D`和`dist()`来使用它们。记住，我们的目标是一个更具表现力的接口，并且使由无关参数类型引起的错误在模板实例化时出现。

我们将首先创建一个算术类型的概念：

```cpp
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>; 
```

接下来，我们将创建一个名为`Point`的概念，它定义了一个点应该具有成员函数`x()`和`y()`返回相同类型，并且这个类型应该支持算术操作：

```cpp
template <typename T>
concept Point = requires(T p) {
  requires std::is_same_v<decltype(p.x()), decltype(p.y())>;
  requires Arithmetic<decltype(p.x())>;
}; 
```

这个概念现在可以通过显式约束使`dist()`的接口更好：

```cpp
auto dist(Point auto p1, Point auto p2) {
  // Same as before ... 
```

这看起来真的很有希望，所以让我们也对我们的返回类型添加一个约束。虽然`Point2D`可能被实例化为整数类型，但我们知道距离可以是浮点数。标准库中的概念`std::floating_point`非常适合这个。这是`dist()`的最终版本：

```cpp
std::floating_point auto dist(Point auto p1, Point auto p2) { 
  auto a = p1.x() - p2.x();
  auto b = p1.y() - p2.y();
  return std::sqrt(a*a + b*b);
} 
```

我们的接口现在更加描述性，当我们尝试用错误的参数类型实例化它时，我们将在实例化阶段而不是最终编译阶段获得错误。

现在我们应该对我们的`Point2D`模板做同样的事情，以避免有人意外地用它实例化它不打算处理的类型。例如，我们希望阻止有人用`const char*`实例化`Point2D`类，就像这样：

```cpp
auto p1 = Point2D{"2.0", "2.0"}; // How can we prevent this? 
```

我们已经创建了`Arithmetic`概念，我们可以在这里使用它来在`Point2D`的模板参数中放置约束。这是我们如何做到的：

```cpp
template <Arithmetic T> // T is now constrained!
class Point2D {
public:
  Point2D(T x, T y) : x_{x}, y_{y} {}
  auto x() { return x_; }
  auto y() { return y_; }
  // ...
private:
  T x_{};
  T y_{};
}; 
```

我们唯一需要改变的是指定类型`T`应该支持概念`Arithmetic`指定的操作。尝试使用`const char*`实例化模板现在将生成一个直接的错误消息，而编译器尝试实例化`Point2D<const char*>`类。

## 向你的代码添加约束

概念的实用性远远超出了模板元编程。这是 C++20 的一个基本特性，改变了我们使用概念而不是具体类型或完全无约束的变量声明`auto`来编写和推理代码的方式。

概念非常类似于类型（如`int`、`float`或`Plot2D<int>`）。类型和概念都指定了对象上支持的一组操作。通过检查类型或概念，我们可以确定某些对象如何构造、移动、比较和通过成员函数访问等。然而，一个重大的区别是，概念并不说任何关于对象如何存储在内存中，而类型除了其支持的操作集之外还提供了这些信息。例如，我们可以在类型上使用`sizeof`运算符，但不能在概念上使用。

通过概念和`auto`，我们可以声明变量而无需明确指出确切的类型，但仍然非常清楚地表达我们的意图。看一下以下代码片段：

```cpp
const auto& v = get_by_id(42); // What can I do with v? 
```

大多数时候，当我们遇到这样的代码时，我们更感兴趣的是我们可以在`v`上执行哪些操作，而不是知道确切的类型。在`auto`前面添加一个概念会产生不同的效果：

```cpp
const Person auto& v = get_by_id(42);
v.get_name(); 
```

几乎可以在几乎所有可以使用关键字 `auto` 的上下文中使用概念：局部变量、返回值、函数参数等等。在我们的代码中使用概念使得阅读更加容易。在撰写本书时（2020 年中），已经建立的 C++ IDE 中目前还没有对概念的额外支持。然而，代码补全以及其他基于概念的有用编辑器功能很快就会可用，使得 C++ 编码更加有趣和安全。

## 标准库中的概念

C++20 还包括一个新的 `<concepts>` 头文件，其中包含预定义的概念。您已经看到其中一些概念的作用。许多概念都是基于类型特性库中的特性。然而，有一些基本概念以前没有用特性表达。其中最重要的是比较概念，如 `std::equality_comparable` 和 `std::totally_ordered`，以及对象概念，如 `std::movable`、`std::copyable`、`std::regular` 和 `std::semiregular`。我们不会在标准库的概念上花费更多时间，但在开始定义自己的概念之前，请记住将它们牢记在心。在正确的泛化级别上定义概念并不是件容易的事，通常明智的做法是基于已经存在的概念定义新的概念。

让我们通过查看 C++ 中一些实际的元编程示例来结束本章。

# 元编程的实际例子

高级元编程可能看起来非常学术化，因此为了展示其有用性，让我们看一些不仅演示元编程语法的例子，还演示它如何在实践中使用。

## 示例 1：创建一个通用的安全转换函数

在 C++ 中进行数据类型转换时，有多种不同的方式会出错：

+   如果将值转换为比特长度较低的整数类型，可能会丢失一个值。

+   如果将负值转换为无符号整数，可能会丢失一个值。

+   如果从指针转换为任何其他整数而不是 `uintptr_t`，正确的地址可能会变得不正确。这是因为 C++ 仅保证 `uintptr_t` 是唯一可以保存地址的整数类型。

+   如果从 `double` 转换为 `float`，结果可能是 `int`，如果 `double` 值太大，`float` 无法容纳。

+   如果使用 `static_cast()` 在指针之间进行转换，如果类型没有共同的基类，可能会得到未定义的行为。

为了使我们的代码更加健壮，我们可以创建一个通用的检查转换函数，在调试模式下验证我们的转换，并在发布模式下尽可能快地执行我们的转换。

根据被转换的类型，会执行不同的检查。如果我们尝试在未经验证的类型之间进行转换，它将无法编译。

这些是 `safe_cast()` 旨在处理的情况：

+   相同类型：显然，如果我们转换相同类型，我们只需返回输入值。

+   指针到指针：如果在指针之间进行转换，`safe_cast()` 在调试模式下执行动态转换以验证是否可转换。

+   双精度浮点数到浮点数：`safe_cast()` 在从 `double` 转换为 `float` 时接受精度损失，但有一个例外 - 如果从 `double` 转换为 `float`，则有可能 `double` 太大，使得 `float` 无法处理结果。

+   算术到算术：如果在算术类型之间进行转换，值将被转换回其原始类型以验证是否丢失精度。

+   指针到非指针：如果从指针转换为非指针类型，`safe_cast()` 验证目标类型是否为 `uintptr_t` 或 `intptr_t`，这是唯一保证能够保存地址的整数类型。

在任何其他情况下，`safe_cast()` 函数将无法编译。

让我们看看如何实现这一点。我们首先获取有关我们的转换操作的`constexpr`布尔值的信息。它们是`constexpr`布尔值而不是`const`布尔值的原因是，我们将在稍后的`if constexpr`表达式中使用它们，这些表达式需要`constexpr`条件：

```cpp
template <typename T> constexpr auto make_false() { return false; }
template <typename Dst, typename Src> 
auto safe_cast(const Src& v) -> Dst{ 
  using namespace std;
  constexpr auto is_same_type = is_same_v<Src, Dst>;
  constexpr auto is_pointer_to_pointer =  
    is_pointer_v<Src> && is_pointer_v<Dst>; 
  constexpr auto is_float_to_float =  
    is_floating_point_v<Src> && is_floating_point_v<Dst>; 
  constexpr auto is_number_to_number =  
    is_arithmetic_v<Src> && is_arithmetic_v<Dst>; 
  constexpr auto is_intptr_to_ptr = 
    (is_same_v<uintptr_t,Src> || is_same_v<intptr_t,Src>)
    && is_pointer_v<Dst>;
  constexpr auto is_ptr_to_intptr =
    is_pointer_v<Src> &&
    (is_same_v<uintptr_t,Dst> || is_same_v<intptr_t,Dst>); 
```

因此，现在我们已经获得了关于转换的所有必要信息，作为`constexpr`布尔值，我们在编译时断言我们可以执行转换。如前所述，如果条件不满足，`static_assert()`将无法编译通过（与常规 assert 不同，后者在运行时验证条件）。

请注意在`if`/`else`链的末尾使用了`static_assert()`和`make_false<T>`。我们不能只输入`static_assert(false)`，因为那样会完全阻止`safe_cast()`的编译；相反，我们利用模板函数`make_false<T>()`来推迟生成，直到需要时。

当执行实际的`static_cast()`时，我们将回到原始类型并验证结果是否等于未转换的参数，使用常规的运行时`assert()`。这样，我们可以确保`static_cast()`没有丢失任何数据：

```cpp
 if constexpr(is_same_type) { 
    return v; 
  }
  else if constexpr(is_intptr_to_ptr || is_ptr_to_intptr){
    return reinterpret_cast<Dst>(v); 
  } 
  else if constexpr(is_pointer_to_pointer) { 
    assert(dynamic_cast<Dst>(v) != nullptr); 
    return static_cast<Dst>(v); 
  } 
  else if constexpr (is_float_to_float) { 
    auto casted = static_cast<Dst>(v); 
    auto casted_back = static_cast<Src>(v); 
    assert(!isnan(casted_back) && !isinf(casted_back)); 
    return casted; 
  }  
  else if constexpr (is_number_to_number) { 
    auto casted = static_cast<Dst>(v); 
    auto casted_back = static_cast<Src>(casted); 
    assert(casted == casted_back); 
    return casted; 
  } 
  else {
    static_assert(make_false<Src>(),"CastError");
    return Dst{}; // This can never happen, 
    // the static_assert should have failed 
  }
} 
```

请注意我们如何使用`if constexpr`来使函数有条件地编译。如果我们使用普通的`if`语句，函数将无法编译通过。

```cpp
auto x = safe_cast<int>(42.0f); 
```

这是因为编译器将尝试编译以下行，而`dynamic_cast`只接受指针：

```cpp
// type To is an integer
assert(dynamic_cast<int>(v) != nullptr); // Does not compile 
```

然而，由于`if constexpr`和`safe_cast<int>(42.0f)`的构造，以下函数可以正确编译：

```cpp
auto safe_cast(const float& v) -> int {
  constexpr auto is_same_type = false;
  constexpr auto is_pointer_to_pointer = false;
  constexpr auto is_float_to_float = false;
  constexpr auto is_number_to_number = true;
  constexpr auto is_intptr_to_ptr = false;
  constexpr auto is_ptr_to_intptr = false
  if constexpr(is_same_type) { /* Eradicated */ }
  else if constexpr(is_intptr_to_ptr||is_ptr_to_intptr){/* Eradicated */}
  else if constexpr(is_pointer_to_pointer) {/* Eradicated */}
  else if constexpr(is_float_to_float) {/* Eradicated */}
  else if constexpr(is_number_to_number) {
    auto casted = static_cast<int>(v);
    auto casted_back = static_cast<float>(casted);
    assert(casted == casted_back);
    return casted;
  }
  else { /* Eradicated */ }
} 
```

如你所见，除了`is_number_to_number`子句之外，在`if constexpr`语句之间的所有内容都已经被完全消除，从而使函数能够编译。

## 示例 2：在编译时对字符串进行哈希处理

假设我们有一个资源系统，其中包含一个无序映射的字符串，用于标识位图。如果位图已经加载，系统将返回已加载的位图；否则，它将加载位图并返回：

```cpp
// External function which loads a bitmap from the filesystem
auto load_bitmap_from_filesystem(const char* path) -> Bitmap {/* ... */}
// Bitmap cache 
auto get_bitmap_resource(const std::string& path) -> const Bitmap& { 
  // Static storage of all loaded bitmaps
  static auto loaded = std::unordered_map<std::string, Bitmap>{};
  // If the bitmap is already in loaded_bitmaps, return it
  if (loaded.count(path) > 0) {
    return loaded.at(path);
  } 
  // The bitmap isn't already loaded, load and return it 
  auto bitmap = load_bitmap_from_filesystem(path.c_str());
  loaded.emplace(path, std::move(bitmap)); 
  return loaded.at(path); 
} 
```

然后在需要位图资源的地方使用位图缓存：

+   如果尚未加载，`get_bitmap_resource()`函数将加载并返回它

+   如果已经在其他地方加载过，`get_bitmap_resource()`将简单地返回已加载的函数。

因此，无论哪个绘制函数先执行，第二个函数都不必从磁盘加载位图：

```cpp
auto draw_something() {
  const auto& bm = get_bitmap_resource("my_bitmap.png");
  draw_bitmap(bm);
}
auto draw_something_again() {
  const auto& bm = get_bitmap_resource("my_bitmap.png");
  draw_bitmap(bm);
} 
```

由于我们使用了无序映射，每当我们检查位图资源时都需要计算哈希值。现在您将看到我们如何通过将计算移动到编译时来优化运行时代码。

### 编译时哈希值计算的优势

我们将尝试解决的问题是，每次执行`get_bitmap_resource("my_bitmap.png")`这一行时，应用程序都会在运行时计算字符串`"my_bitmap.png"`的哈希值。我们希望在编译时执行这个计算，这样当应用程序执行时，哈希值已经被计算出来。换句话说，就像你们学习使用元编程在编译时生成函数和类一样，我们现在要让它在编译时生成哈希值。

你可能已经得出结论，这是所谓的*微优化*：计算一个小字符串的哈希值不会对应用程序的性能产生任何影响，因为这是一个非常小的操作。这可能完全正确；这只是一个将计算从运行时移动到编译时的示例，可能还有其他情况下这可能会产生显著的性能影响。

顺便说一句，当为弱硬件编写软件时，字符串哈希是一种纯粹的奢侈，但在编译时对字符串进行哈希处理可以让我们在任何平台上都享受到这种奢侈，因为一切都是在编译时计算的。

### 实现和验证编译时哈希函数

为了使编译器能够在编译时计算哈希和，我们重写`hash_function()`，使其以一个高级类（如`std::string`）的原始空终止`char`字符串作为参数，这在编译时无法计算。现在，我们可以将`hash_function()`标记为`constexpr`：

```cpp
constexpr auto hash_function(const char* str) -> size_t {
  auto sum = size_t{0};
  for (auto ptr = str; *ptr != '\0'; ++ptr)
    sum += *ptr;
  return sum;
} 
```

现在，让我们使用在编译时已知的原始字面字符串调用它：

```cpp
auto hash = hash_function("abc"); 
```

编译器将生成以下代码片段，这是与`a`，`b`和`c`对应的 ASCII 值的总和（`97`，`98`和`99`）：

```cpp
auto hash = size_t{294}; 
```

只是累积单个值是一个非常糟糕的哈希函数；在实际应用中不要这样做。这里只是因为它容易理解。一个更好的哈希函数是将所有单个字符与`boost::hash_combine()`结合起来，如*第四章*，*数据结构*中所解释的那样。

`hash_function()`只有在编译器在编译时知道字符串时才会在编译时计算；如果不知道，编译器将像任何其他表达式一样在运行时执行`constexpr`。

既然我们已经有了哈希函数，现在是时候创建一个使用它的字符串类了。

### 构造一个 PrehashedString 类

我们现在准备实现一个用于预哈希字符串的类，它将使用我们创建的哈希函数。这个类包括以下内容：

+   一个以原始字符串作为参数并在构造时计算哈希的构造函数。

+   比较运算符。

+   一个`get_hash()`成员函数，返回哈希值。

+   `std::hash()`的重载，简单地返回哈希值。这个重载被`std::unordered_map`，`std::unordered_set`或标准库中使用哈希值的任何其他类使用。简单地说，这使得容器意识到`PrehashedString`存在一个哈希函数。

这是`PrehashedString`类的基本实现：

```cpp
class PrehashedString {
public:
  template <size_t N>
  constexpr PrehashedString(const char(&str)[N])
      : hash_{hash_function(&str[0])}, size_{N - 1},
      // The subtraction is to avoid null at end
        strptr_{&str[0]} {}
  auto operator==(const PrehashedString& s) const {
    return
      size_ == s.size_ &&
      std::equal(c_str(), c_str() + size_, s.c_str());
  }
  auto operator!=(const PrehashedString& s) const {
    return !(*this == s); }
  constexpr auto size()const{ return size_; }
  constexpr auto get_hash()const{ return hash_; }
  constexpr auto c_str()const->const char*{ return strptr_; }
private:
  size_t hash_{};
  size_t size_{};
  const char* strptr_{nullptr};
};
namespace std {
template <>
struct hash<PrehashedString> {
  constexpr auto operator()(const PrehashedString& s) const {
    return s.get_hash();
  }
};
} // namespace std 
```

请注意构造函数中的模板技巧。这迫使`PrehashedString`只接受编译时字符串字面值。这样做的原因是`PrehashedString`类不拥有`const char* ptr`，因此我们只能在编译时使用它创建的字符串字面值：

```cpp
// This compiles
auto prehashed_string = PrehashedString{"my_string"};
// This does not compile
// The prehashed_string object would be broken if the str is modified
auto str = std::string{"my_string"};
auto prehashed_string = PrehashedString{str.c_str()};
// This does not compile.
// The prehashed_string object would be broken if the strptr is deleted
auto* strptr = new char[5];
auto prehashed_string = PrehashedString{strptr}; 
```

所以，既然我们已经准备就绪，让我们看看编译器如何处理`PrehashedString`。

### 评估 PrehashedString

这是一个简单的测试函数，返回字符串`"abc"`的哈希值（为了简单起见）：

```cpp
auto test_prehashed_string() {
  const auto& hash_fn = std::hash<PrehashedString>{};
  const auto& str = PrehashedString("abc");
  return hash_fn(str);
} 
```

由于我们的哈希函数只是对值求和，而`"abc"`中的字母具有 ASCII 值*a* = 97，*b* = 98，*c* = 99，由 Clang 生成的汇编代码应该输出和为 97 + 98 + 99 = 294。检查汇编代码，我们可以看到`test_prehashed_string()`函数编译成了一个`return`语句，返回`294`：

```cpp
mov eax, 294
ret 
```

这意味着整个`test_prehashed_string()`函数已经在编译时执行；当应用程序执行时，哈希和已经被计算！

### 使用 PrehashedString 评估 get_bitmap_resource()

让我们回到最初的`get_bitmap_resource()`函数，最初使用的`std::string`已经被替换为`PrehashedString`：

```cpp
// Bitmap cache
auto get_bitmap_resource(const PrehashedString& path) -> const Bitmap& 
{
  // Static storage of all loaded bitmaps
  static auto loaded_bitmaps =
    std::unordered_map<PrehashedString, Bitmap>{};
  // If the bitmap is already in loaded_bitmaps, return it
  if (loaded_bitmaps.count(path) > 0) {
    return loaded_bitmaps.at(path);
  }
  // The bitmap isn't already loaded, load and return it
  auto bitmap = load_bitmap_from_filesystem(path.c_str());
  loaded_bitmaps.emplace(path, std::move(bitmap));
  return loaded_bitmaps.at(path);
} 
```

我们还需要一个测试函数：

```cpp
auto test_get_bitmap_resource() { return get_bitmap_resource("abc"); } 
```

我们想知道的是这个函数是否预先计算了哈希和。由于`get_bitmap_resource()`做了很多事情（构造静态`std::unordered_map`，检查映射等），生成的汇编代码大约有 500 行。尽管如此，如果我们的魔术哈希和在汇编代码中找到，这意味着我们成功了。

当检查由 Clang 生成的汇编代码时，我们将找到一行对应于我们的哈希和，`294`：

```cpp
.quad   294                     # 0x126 
```

为了确认这一点，我们将字符串从`"abc"`改为`"aaa"`，这应该将汇编代码中的这一行改为 97 * 3 = 291，但其他一切应该完全相同。

我们这样做是为了确保这不只是一些其他与哈希和毫不相关的魔术数字。

检查生成的汇编代码，我们将找到期望的结果：

```cpp
.quad   291                     # 0x123 
```

除了这一行之外，其他都是相同的，因此我们可以安全地假设哈希是在编译时计算的。

我们所看到的示例表明，我们可以将编译时编程用于非常不同的事情。添加可以在编译时验证的安全检查，使我们能够在不运行程序并通过覆盖测试搜索错误的情况下找到错误。并且将昂贵的运行时操作转移到编译时可以使我们的最终程序更快。

# 总结

在本章中，您已经学会了如何使用元编程来在编译时而不是运行时生成函数和值。您还发现了如何以现代 C++的方式使用模板、`constexpr`、`static_assert()`和`if constexpr`、类型特征和概念来实现这一点。此外，通过常量字符串哈希，您看到了如何在实际环境中使用编译时评估。

在下一章中，您将学习如何进一步扩展您的 C++工具箱，以便您可以通过构建隐藏的代理对象来创建库。
