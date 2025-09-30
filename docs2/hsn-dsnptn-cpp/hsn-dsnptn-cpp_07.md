

# SFINAE、概念和重载解析管理

本章我们研究的惯用表达式**替换失败不是错误**（**SFINAE**）在使用的语言特性方面更为复杂。因此，它往往吸引大量的 C++程序员的关注。这个特性中有些东西符合典型 C++程序员的思维方式——普通人认为，如果它没有坏，就不要去动它。程序员，尤其是 C++程序员，往往认为，如果它没有坏，你就没有充分利用它。我们只能说，SFINAE 有很大的潜力。

在本章中，我们将涵盖以下主题：

+   函数重载和重载解析是什么？类型推导和替换是什么？

+   SFINAE 是什么，为什么它在 C++中是必要的？

+   如何使用 SFINAE 编写极其复杂，有时有用的程序？

# 技术要求

本章的示例代码可以在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) [Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter07](http://Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter07)找到。

# 重载解析和重载集

本节将测试你对 C++标准最新和最先进补充的了解。我们将从 C++最基本的功能之一——函数及其重载开始。

## C++函数重载

`f(x)`，那么必须存在多个名为`f`的函数。如果出现这种情况，我们就处于一个重载情况，必须进行重载解析以确定应该调用这些函数中的哪一个。

让我们从简单的例子开始：

```cpp
// Example 01
void f(int i) { cout << “f(int)” << endl; }        // 1
void f(long i) { cout << “f(long)” << endl; }    // 2
void f(double i) { cout << “f(double)” << endl; }    // 3
f(5);        // 1
f(5l);    // 2
f(5.0);    // 3
```

在这里，我们有三个名为`f`的函数定义和三个函数调用。请注意，函数签名都是不同的（参数类型不同）。这是一个要求——重载函数必须在参数上有所不同。不可能有两个重载接受完全相同的参数，但返回类型或函数体不同。此外，请注意，虽然这个例子是针对常规函数的，但相同的规则也适用于重载成员函数，因此我们不会特别关注成员函数。

回到我们的例子，每一行调用的是哪个`f()`函数？为了理解这一点，我们需要知道 C++中重载函数是如何解析的。重载解析的确切规则相当复杂，并且在不同版本的规范中存在细微的差异，但大部分情况下，它们被设计成编译器会做你期望它在最常见情况下做的事情。我们预计`f(5)`会调用接受整数参数的重载，因为`5`是一个`int`变量。确实如此。同样，`5l`具有长类型，因此`f(5l)`调用第二个重载。最后，`5.0`是一个浮点数，因此调用最后一个重载。

这并不难，对吧？但如果参数与参数类型不完全匹配会发生什么？那么，编译器必须考虑类型转换。例如，`5.0`字面量的类型是`double`。让我们看看如果我们用`float`类型的参数调用`f()`会发生什么：

```cpp
f(5.0f);
```

现在我们必须将参数从`float`类型转换为`int`、`long`或`double`类型之一。同样，标准有规则，但将转换为`double`视为首选并且调用重载应该不会令人感到意外。

让我们看看不同整数类型会发生什么，比如说，`unsigned int`：

```cpp
f(5u);
```

现在我们有两个选择；将`unsigned int`转换为`signed int`，或者转换为`signed long`。虽然可以争论转换为`long`更安全，因此更好，但标准认为这两种转换非常接近，以至于编译器无法选择。这个调用无法编译，因为重载解析被认为是模糊的；错误信息应该说明这一点。如果你在代码中遇到这样的问题，你必须通过将参数转换为使解析无歧义的类型来帮助编译器。通常，最简单的方法是将重载函数所需参数的类型进行转换：

```cpp
unsigned int i = 5u;
f(static_cast<int>(i));
```

到目前为止，我们已经处理了参数类型不同但数量相同的情况。当然，如果不同名称的函数声明中参数的数量不同，只需要考虑可以接受所需数量参数的函数。以下是一个具有相同名称但参数数量不同的两个函数的示例：

```cpp
void f(int i) { cout << “f(int)” << endl; }            // 1
void f(long i, long j) { cout << “f(long2)” << endl; }    // 2
f(5.0, 7);
```

在这里，重载解析非常简单——我们需要一个可以接受两个参数的函数，而且只有一个选择。两个参数都将必须转换为`long`。但如果存在多个具有相同参数数量的函数，会怎样呢？让我们看看以下示例：

```cpp
// Example 02
void f(int i, int j) { cout << “f(int, int)” << endl; }// 1
void f(long i, long j) { cout << “f(long2)” << endl; }    // 2
void f(double i) { cout << “f(double)” << endl; }      // 3
f(5, 5);    // 1
f(5l, 5l);    // 2
f(5, 5.0);    // 1
f(5, 5l);    // ?
```

首先，最明显的情况 - 如果所有参数的类型与某个重载的参数类型完全匹配，则调用该重载。接下来，事情开始变得有趣 - 如果没有完全匹配，我们可以在每个参数上进行转换。让我们考虑第三次调用，`f(5, 5.0)`。第一个参数，`int`，与第一个重载完全匹配，但在必要时可以转换为`long`。第二个参数，`double`，与任何重载都不匹配，但可以转换为匹配两者。第一个重载是更好的匹配 - 它需要的参数转换更少。最后，关于最后一行呢？第一个重载可以通过对第二个参数进行转换来调用。第二个重载也可以通过在第一个参数上进行转换来工作。这又是一个模糊的重载，并且这一行将无法编译。请注意，通常情况下，转换较少的重载并不总是获胜；在更复杂的情况下，即使一个重载需要的转换较少，也可能有模糊的重载（一般规则是，如果有一个重载在所有参数上都有最佳的转换，则获胜；否则，调用是模糊的）。为了解决这种歧义，你必须通过类型转换（通常是通过类型转换，在我们的例子中是通过改变数字字面量的类型）来改变一些参数的类型，以便使预期的重载成为首选的重载。

注意第三个重载是如何完全被忽略的，因为它对于所有函数调用都具有错误的参数数量。但这并不总是那么简单 - 函数可以有默认参数，这意味着参数的数量并不总是必须与参数的数量匹配。

考虑以下代码块：

```cpp
// Example 03
void f(int i) { cout << “f(int)” << endl; }            // 1
void f(long i, long j) { cout << “f(long2)” << endl; }    // 2
void f(double i, double j = 0) {                    // 3
  cout << “f(double, double = 0)” << endl;
}
f(5);        // 1
f(5l, 5);    // 2
f(5, 5);    // ?
f(5.0);    // 3
f(5.0f);    // 3
f(5l);    // ?
```

我们现在有三个重载。第一个和第二个永远不会混淆，因为它们具有不同数量的参数。然而，第三个重载可以用一个或两个参数调用；在前一种情况下，第二个参数被假定为零。第一次调用是最简单的 - 一个参数，其中类型与第一个重载的参数类型完全匹配。第二次调用让我们想起了之前见过的案例 - 两个参数，其中第一个与某个重载完全匹配，但第二个需要转换。替代重载需要在两个参数上执行转换，因此第二个函数定义是最好的匹配。

第三个调用似乎足够简单，因为它有两个整数参数，但这种简单性具有欺骗性——存在两个接受两个参数的重载版本，并且在两种重载情况下，两个参数都需要转换。虽然从`int`到`long`的转换可能看起来比从`int`到`double`的转换更好，但 C++并不这样认为。这个调用是模糊的。下一个调用`f(5.0)`只有一个参数，它可以转换为`int`，这是单参数重载中参数的类型。但它仍然更适合第三个重载，在那里它根本不需要转换。将参数类型从`double`改为`float`，我们得到下一个调用。转换为`double`比转换为`int`更好，并且利用默认参数不被视为转换，因此在重载比较时不会带来任何其他*惩罚*。最后一个调用再次是模糊的——转换为`double`和转换为`int`都被认为是同等重要的，因此第一个和第三个重载同样好。第二个重载提供了对第一个参数的精确匹配；然而，没有方法可以不提供第二个参数就调用该重载，因此它甚至没有被考虑。

到目前为止，我们只考虑了普通的 C++函数，尽管我们所学的所有内容同样适用于成员函数。现在，我们需要将模板函数也加入其中。

## 模板函数

除了常规函数之外，对于参数类型已知的函数，C++还有`template`函数。当调用这些函数时，参数类型是从调用位置的参数类型中推断出来的。模板函数可以与非模板函数具有相同的名称，也可以有多个模板函数具有相同的名称，因此我们需要了解在模板存在的情况下如何进行重载解析。

考虑以下示例：

```cpp
// Example 04
void f(int i) { cout << “f(int)” << endl; }        // 1
void f(long i) { cout << “f(long)” << endl; }    // 2
template <typename T>
void f(T i) { cout << “f(T)” << endl; }        // 3
f(5);        // 1
f(5l);    // 2
f(5.0);    // 3
```

`f`函数名可以指代这三个函数中的任何一个，其中一个是一个模板。每次都会从这三个中选择最佳的重载。考虑特定函数调用重载解析的函数集合被称为`f()`匹配正好与重载集中第一个非模板函数相匹配——参数类型是`int`，第一个函数是`f(int)`。如果在重载集中找到了与非模板函数的精确匹配，它总是被认为是最佳的重载。

模板函数也可以通过精确匹配进行实例化——用具体类型替换模板参数的过程称为模板参数替换（或类型替换），如果将 `int` 替换为 `T` 模板参数，那么我们得到另一个与调用完全匹配的函数。然而，与调用完全匹配的非模板函数被认为是一个更好的重载。第二次调用以类似的方式处理，但它与重载集中的第二个函数完全匹配，因此将调用该函数。最后一个调用有一个 `double` 类型的参数，可以转换为 `int` 或 `long`，或者替换 `T` 以使模板实例化与调用完全匹配。由于没有与调用完全匹配的非模板函数，因此实例化为精确匹配的模板函数是下一个最佳重载并被选中。

但是，当有多个模板函数可以替换其模板参数以匹配调用参数类型时会发生什么？让我们来看看：

```cpp
// Example 05
void f(int i) { cout << “f(int)” << endl; }    // 1
template <typename T>
void f(T i) { cout << “f(T)” << endl; }    // 2
template <typename T>
void f(T* i) { cout << “f(T*)” << endl; }    // 3
f(5);        // 1
f(5l);    // 2
int i = 0;
f(&i);    // 3
```

第一次调用再次与非模板函数完全匹配，因此得到解决。第二次调用与第一个非模板重载匹配，通过转换，或者如果将 `long` 类型替换为 `T`，则与第二个重载完全匹配。最后一个重载与这两个调用都不匹配——没有替换可以使 `T*` 参数类型与 `int` 或 `long` 匹配。然而，如果将 `int` 替换为 `T`，最后一个调用可以与第三个重载匹配。问题是，如果将 `int*` 替换为 `T`，它也可以与第二个重载匹配。那么选择哪个模板重载呢？答案是更具体的那个——第一个重载 `f(T)` 可以与任何单参数函数调用匹配，而第二个重载 `f(T*)` 只能与带有指针参数的调用匹配。更具体、更窄的重载被认为是一个更好的匹配，并被选中。这是一个新的概念，专门针对模板——我们选择更难实例化的重载，而不是选择更好的转换（通常，*更少* 或 *更简单* 的转换）。

这条规则似乎对空指针不适用：`f(NULL)` 可以调用第一个或第二个重载（`f(int)` 或 `f(T)`），而 `f(nullptr)` 则调用第二个重载，`f(T)`。指针重载从未被调用，尽管 `NULL` 和 `nullptr` 都被认为是空指针。然而，这实际上是编译器严格遵循规则的情况。C++ 中的 `NULL` 是一个整数零，实际上是一个宏：

```cpp
#define NULL 0 // Or 0L
```

根据 `0` 或 `0L` 的定义，调用 `f(int)` 或 `f(T)`（其中 `T==long`）取决于 `f(int)`。尽管名称中有“ptr”，但 `nullptr` 实际上是一个 `nullptr_t` 类型的常量值。它可以 *转换为* 任何指针类型，但它不是任何指针类型。这就是为什么在处理接受不同类型指针的函数时，经常声明带有 `nullptr_t` 参数的重载。

最后，还有一种函数可以匹配几乎任何具有相同名称的函数调用，那就是接受可变参数的函数：

```cpp
// Example 06
void f(int i) { cout << “f(int)” << endl; }    // 1
void f(...) { cout << “f(...)” << endl; }    // 2
f(5);        // 1
f(5l);    // 1
f(5.0);    // 1
struct A {};
A a;
f(a);    {};    // 2
```

重载中的第一个可以用于前三个函数调用 - 它是第一个调用的精确匹配，并且存在转换可以使其他两个调用适合`f()`的第一个重载的签名。在这个示例中的第二个函数可以用任何数量和类型的参数调用。这被认为是最后的手段 - 具有可以转换为正确转换以匹配调用的特定参数的函数更受欢迎。这包括用户定义的转换，如下所示：

```cpp
struct B {
  operator int() const { return 0; }
};
B b;
f(b);        // 1
```

只有在没有转换可以让我们避免调用`f(...)`可变函数的情况下，才必须调用它。

现在我们知道了重载解析的顺序 - 首先选择一个与参数完全匹配的非模板函数。如果在重载集中没有这样的匹配，则选择一个模板函数，如果其参数可以用具体类型替换以给出精确匹配。如果有多个这样的模板函数选项，则更具体的重载优先于更一般的一个。如果以这种方式尝试匹配模板函数也失败，则如果参数可以转换为参数类型，则调用非模板函数。最后，如果所有其他方法都失败，但有一个接受可变参数的正确名称的函数可用，则调用该函数。请注意，某些转换被认为是*平凡的*，并包含在*精确匹配*的概念中，例如，从`T`到`const` `T`的转换。在每一步中，如果存在多个同样好的选项，则重载被认为是模糊的，程序是不良的。

模板函数中的类型替换过程决定了模板函数参数的最终类型，以及它们与函数调用参数的匹配程度。这个过程可能会导致一些意想不到的结果，必须更详细地考虑。

# 模板函数中的类型替换

在实例化模板函数以匹配特定调用时，我们必须仔细区分两个步骤 - 首先，从参数类型推导出模板参数的类型（这个过程称为类型推导）。一旦推导出类型，就用具体类型替换所有参数类型（这个过程称为**类型替换**）。当函数有多个参数时，这种差异变得更加明显。

## 类型推导和替换

类型推导和替换密切相关，但并不完全相同。推导是“*猜测：*”为了匹配调用，模板类型或类型应该是什么的过程？当然，编译器并不是真的猜测，而是应用标准中定义的一组规则。考虑以下示例：

```cpp
// Example 07
template <typename T>
void f(T i, T* p) { std::cout << “f(T, T*)” << std::endl; }
int i;
f(5, &i);    // T == int
f(5l, &i);    // ?
```

在考虑第一次调用时，我们可以从第一个参数推导出`T`模板参数应该是`int`。因此，`int`被替换为函数的两个参数中的`T`。模板被实例化为`f(int, int*)`，与参数类型完全匹配。在考虑第二次调用时，我们可以从第一个参数推导出`T`应该是`long`，或者我们可以从第二个参数推导出`T`应该是`int`。这种歧义导致类型推导过程失败。如果这是唯一可用的重载，则不会选择任何选项，程序无法编译。如果存在更多重载，它们将依次考虑，包括可能是最后的手段，即`f(...)`可变函数的重载。在这里需要注意的一个重要细节是，在推导模板类型时不会考虑转换——将`T`推导为`int`将为第二次调用产生`f(int, int*)`，这是调用`f(long, int*)`时转换第一个参数的一个可行选项。然而，这个选项根本没有被考虑，并且类型推导因为歧义而失败。

可以通过明确指定模板类型来解决模糊的推导，从而消除类型推导的需要：

```cpp
f<int>(5l, &i);    // T == int
```

现在，类型推导根本就没有进行：我们从函数调用中知道`T`是什么，因为它被明确指定了。另一方面，类型替换仍然必须发生——第一个参数是`int`类型，第二个是`int*`类型。函数调用通过转换第一个参数而成功。我们也可以强制进行反向推导：

```cpp
f<long>(5l, &i);    // T == long
```

再次，由于我们知道`T`是什么，所以不需要推导。替换过程是直接的，我们最终得到`f(long, long*)`。由于没有从`int*`到`long*`的有效转换，这个函数不能使用`int*`作为第二个参数来调用。因此，程序无法编译。请注意，通过明确指定类型，我们也指定了`f()`必须是一个模板函数。对于`f()`的非模板重载不再考虑。另一方面，如果有多个`f()`模板函数，那么这些重载将按常规考虑，但这次是使用我们通过明确指定强制进行的参数推导的结果。

模板函数可以有默认参数，就像非模板函数一样，然而，这些参数的值并不用于推导类型（在 C++11 中，模板函数可以为它们的类型参数提供默认值，这提供了一种替代方案）。考虑以下示例：

```cpp
// Example 08
void f(int i, int j = 1) {                      // 1
  cout << “f(int2)” << endl;
}
template <typename T> void f(T i, T* p = nullptr) {    // 2
  cout << “f(T, T*)” << endl;
}
int i;
f(5);        // 1
f(5l);    // 2
```

第一次调用与`f(int, int)`非模板函数完全匹配，第二个参数的默认值为`1`。请注意，如果我们将函数声明为`f(int i, int j = 1L)`，默认值作为`long`，这也不会有任何区别。默认参数的类型并不重要——如果它可以转换为指定的参数类型，那么就使用那个值，否则，程序将从第一行开始就无法编译。第二次调用与`f(T, T*)`模板函数完全匹配，`T == long`，第二个参数的默认值为`NULL`。同样，那个值的类型不是`long*`并不重要。

我们现在理解了类型推导和类型替换之间的区别。当可以从不同的参数推导出不同的具体类型时，类型推导可能会产生歧义。如果发生这种情况，这意味着我们没有推导出参数类型，无法使用此模板函数。类型替换永远不会产生歧义——一旦我们知道`T`是什么，我们每次在函数定义中看到`T`时，就简单地替换那个类型。这个过程也可能失败，但方式不同。

## 替换失败

一旦我们推导出模板参数类型，类型替换就是一个纯粹机械的过程：

```cpp
// Example 09
template <typename T> T* f(T i, T& j) {
  j = 2*i;
  return new T(i);
}
int i = 5, j = 7;
const int* p = f(i, j);
```

在这个例子中，`T`类型可以从第一个参数推导为`int`。它也可以从第二个参数推导为`int`。请注意，返回类型不用于类型推导。由于对`T`只有一个可能的推导，我们现在在函数定义中每次看到`T`时，都将其替换为`int`：

```cpp
int* f(int i, int& j) {
  j = 2*i;
  return new int(i);
}
```

然而，并非所有类型都是平等的，有些类型比其他类型允许更多的自由度。考虑以下代码：

```cpp
// Example 10
template <typename T>
void f(T i, typename T::t& j) {
  std::cout << “f(T, T::t)” << std::endl;
}
template <typename T>
void f(T i, T j) {
  std::cout << “f(T, T)” << std::endl;
}
struct A {
struct t { int i; }; t i; };
A a{5};
f(a, a.i);    // T == A
f(5, 7);    // T == int
```

在考虑第一次调用时，编译器从第一个和第二个参数推导出`T`模板参数为`A`类型——第一个参数是`A`类型的值，第二个参数是`A::t`嵌套类型的引用，如果我们坚持我们最初对`T`作为`A`的推导，它匹配`T::t`。第二个重载从两个参数中为`T`提供冲突的值，因此不能使用。因此，调用第一个重载。

现在，仔细看看第二个调用。对于两个重载，`T` 类型都可以从第一个参数推导为 `int`。然而，将 `int` 替换为 `T`，在第一个重载的第二个参数中会产生一些奇怪的东西 - `int::t`。当然，这不会编译 - `int` 不是一个类，也没有任何嵌套类型。实际上，我们可以预期，对于任何不是类或没有名为 `t` 的嵌套类型的 `T` 类型，第一个模板重载都将无法编译。确实，尝试在第一个模板函数中将 `int` 替换为 `T` 将因为第二个参数的类型无效而失败。然而，这种替换失败并不意味着整个程序无法编译。相反，它被默默地忽略，并且原本可能是不良的重载从重载集中移除。然后，重载解析继续如常进行。当然，我们可能会发现没有任何重载与函数调用匹配，程序仍然无法编译，但错误信息不会提到 `int::t` 无效的问题；它只会说没有可以调用的函数。

再次，区分类型推导失败和类型替换失败非常重要。我们可以完全不考虑前者：

```cpp
f<int>(5, 7);    // T == int
```

现在，推导是不必要的，但将 `int` 替换为 `T` 的过程仍然必须发生，并且这种替换在第一个重载中产生了一个无效的表达式。再次，这种替换失败将这个 `f()` 的候选者从重载集中排除，并且重载解析继续（在这种情况下，成功）使用剩余的候选者。通常，这将是我们的重载练习的结束：模板生成的代码无法编译，因此整个程序也不应该编译。幸运的是，C++ 在这种情况下更为宽容，并且有一个特殊的异常，我们需要了解。

## 替换失败不是错误

由表达式引起的替换失败，该表达式在指定的或推导的类型中将是无效的，不会使整个程序无效的规则被称为**替换失败不是错误**（**SFINAE**）。这个规则对于在 C++ 中使用模板函数是必不可少的；没有 SFINAE，将无法编写许多其他方面完全有效的程序。考虑以下模板重载，它区分了普通指针和成员指针：

```cpp
// Example 11
template <typename T> void f(T* i) {        // 1
  std::cout << “f(T*)” << std::endl;
}
template <typename T> void f(int T::* p) {    // 2
  std::cout << “f(T::*)” << std::endl;
}
struct A { int i; };
A a;
f(&a.i);    // 1
f(&A::i);    // 2
```

到目前为止，一切顺利 - 第一次调用时，函数使用特定变量的指针调用，`a.i`，`T` 类型被推导为 `int`。第二次调用是使用 `A` 类的数据成员的指针，其中 `T` 被推导为 `A`。但现在，让我们用指向不同类型的指针来调用 `f()`：

```cpp
int i;
f(&i);    // 1
```

第一个重载仍然工作得很好，这正是我们想要调用的。但第二个重载不仅不太合适，它完全是无效的——如果我们尝试用 `int` 替换 `T`，它将导致语法错误。这个语法错误被编译器观察到并被静默忽略，连同重载本身。

注意，SFINAE 规则不仅限于无效类型，例如对不存在类成员的引用。有几种方式可能导致替换失败：

```cpp
// Example 12
template <size_t N>
void f(char(*)[N % 2] = nullptr) {    // 1
  std::cout << “N=” << N << “ is odd” << std::endl;
}
template <size_t N>
void f(char(*)[1 - N % 2] = nullptr) { // 2
  std::cout << “N=” << N << “ is even” << std::endl;
}
f<5>();
f<8>();
```

在这里，模板参数是一个值，而不是一个类型。我们有两个模板重载，它们都接受字符数组指针，并且数组大小表达式仅对 `N` 的某些值有效。具体来说，零大小数组在 C++ 中是无效的。因此，第一个重载仅在 `N` `%` `2` 非零时有效，即 `N` 是奇数。同样，第二个重载仅在 `N` 是偶数时有效。没有给函数提供任何参数，所以我们打算使用默认参数。如果没有这两个重载在所有方面都是模糊的，那么在两次调用中，其中一个重载在模板参数替换期间失败并被静默移除。

上述示例非常简洁——特别是模板参数值推断，相当于数值参数的类型推断被显式指定禁用。我们可以恢复推断，并且替换可能成功或失败，这取决于表达式是否有效：

```cpp
// Example 13
template <typename T, size_t N = T::N>
void f(T t, char(*)[N % 2] = NULL) {
  std::cout << “N=” << N << “ is odd” << std::endl;
}
template <typename T, size_t N = T::N>
void f(T t, char(*)[1 - N % 2] = NULL) {
  std::cout << “N=” << N << “ is even” << std::endl;
}
struct A { enum {N = 5}; };
struct B { enum {N = 8}; };
A a;
B b;
f(a);
f(b);
```

现在，编译器必须从第一个参数中推断类型。对于第一次调用，`f(a)`，`A` 类型很容易推断出来。无法推断第二个模板参数，`N`，因此使用默认值（我们现在处于 C++11 领域）。推断出两个模板参数后，我们现在进行替换，其中 `T` 被替换为 `A`，`N` 被替换为 `5`。这种替换在第二个重载中失败，但在第一个重载中成功。在重载集中只剩下一个候选者时，重载解析结束。同样，第二次调用 `f(b)` 最终调用的是第二个重载。

注意，前一个示例和更早的示例之间存在一个微妙但非常重要的区别，其中我们有了这个函数：

```cpp
template <typename T> void f(T i, typename T::t& j);
```

在这个模板中，替换失败是“*自然的*”：可能引起失败的参数是必需的，并且意图是成员指针类型。在前一个情况下，模板参数 `N` 是多余的：它除了人为地导致替换失败和禁用一些重载之外，对任何其他事情都不需要。你为什么要人为地造成替换失败呢？我们已经看到了一个原因，即强制选择其他情况下模糊的重载。更普遍的原因与这样一个事实有关，即类型替换有时可能导致错误。

## 当替换失败仍然是一个错误时

注意，SFINAE 并不能保护我们免受在模板实例化过程中可能发生的任何语法错误。例如，如果模板参数被推导，并且模板参数被替换，我们仍然可能得到一个无效的模板函数：

```cpp
// Example 14
template <typename T> void f(T) {
  std::cout << sizeof(T::i) << std::endl;
}
void f(...) { std::cout << “f(...)” << std::endl; }
f(0);
```

这个代码片段与我们之前考虑的代码片段非常相似，只有一个例外——我们直到检查函数体时才知道模板重载假设`T`类型是一个类，并且有一个名为`T::i`的数据成员。到那时，已经太晚了，因为重载解析仅基于函数声明——参数、默认参数和返回类型（后者不用于推导类型或选择更好的重载，但仍会进行类型替换并受 SFINAE 覆盖）。一旦模板被实例化并被重载解析选择，任何语法错误，如函数体内的无效表达式，都不会被忽略——这种失败是一个非常严重的错误。替换失败是否被忽略的确切上下文列表由标准定义；它在 C++11 中得到了显著扩展，后续标准进行了一些细微的调整。

另有一种情况，尝试使用 SFINAE 会导致错误。以下是一个例子：

```cpp
// Example 15a
template <typename T> struct S {
  typename T::value_type f();
};
```

在这里，我们有一个类模板。如果类型`T`没有嵌套类型`value_type`，类型替换会导致错误，这是一个真正的错误，不会被忽略。你甚至不能使用没有`value_type`的类型实例化这个类模板。将函数变成模板并不能解决这个问题：

```cpp
template <typename T> struct S {
  template <typename U> typename T::value_type f();
};
```

记住这一点非常重要，即 SFINAE 仅在模板函数推导的类型替换过程中发生错误时才适用。在上一个例子中，替换错误并不依赖于模板类型参数`U`，因此它始终会是一个错误。如果你真的需要解决这个问题，你必须使用成员函数模板，并使用模板类型参数来触发替换错误。由于我们不需要额外的模板参数，我们可以将其默认为与类模板类型参数`T`相同：

```cpp
// Example 15b
template <typename T> struct S {
  template <typename U = T>
  std::enable_if_t<std::is_same_v<U, T>
  typename U::value_type f();
};
```

现在，如果存在替换错误，它将发生在依赖于模板类型参数`U::value_type`的类型上。我们不需要指定类型`U`，因为它默认为`T`，并且由于类型`U`和`T`必须相同的要求（否则函数的返回类型无效，这是一个 SFINAE 错误），它不能是其他任何东西。因此，我们的模板成员函数`f()`（几乎）完全做了原始非模板函数`f()`所做的事情（如果函数在类中有重载，则存在细微的差异）。所以，如果你真的需要“隐藏”由类模板参数引起的替换错误，你可以通过引入冗余的函数模板参数并将这两个参数限制为始终相同来实现。

在继续之前，让我们回顾一下我们遇到的三种替换失败类型。

## 替换失败发生在哪里？为什么会发生？

为了理解本章的其余部分，我们必须清楚地区分在模板函数中可能发生的几种替换失败类型。

第一种发生在模板声明使用依赖类型或其他可能导致失败的构造时，并且它们的使用对于正确声明模板是必要的。以下是一个旨在使用容器参数调用的模板函数（所有 STL 容器都有一个嵌套类型 `value_type`）：

```cpp
// Example 16
template <typename T>
bool find(const T& cont, typename T::value_type val);
```

如果我们尝试用没有定义嵌套类型 `value_type` 的参数调用这个函数，函数调用将无法编译（假设我们没有其他重载）。还有许多其他例子，我们自然使用依赖类型和其他可能对某些模板参数值无效的表达式。这些无效表达式会导致替换失败。它不必发生在参数声明中。以下是一个返回类型可能未定义的模板：

```cpp
// Example 16
template <typename U, typename V>
std::common_type_t<U, V> compute(U u, V v);
```

在这个模板中，返回类型是两个模板参数类型的公共类型。但如果模板参数的类型 `U` 和 `V` 没有公共类型，会发生什么？那么类型表达式 `std::common_type_t<U, V>` 是无效的，类型替换失败。以下又是另一个例子：

```cpp
// Example 15
template <typename T>
auto process(T p) -> decltype(*p);
```

在这里，再次，替换失败可能发生在返回类型中，但我们使用尾随返回类型，这样我们可以直接检查表达式 `*p` 是否可以编译（或者更正式地说，是否有效）。如果是，结果类型就是返回类型。否则，替换失败。请注意，这种声明与以下类似的东西之间有一个区别：

```cpp
template <typename T> T process(T* p);
```

如果函数参数是一个原始指针，两种版本都相当于同一件事。但第一种变体也可以为任何可以解引用的类型编译，例如容器迭代器和智能指针，而第二种版本仅适用于原始指针。

第二种替换失败发生在函数声明成功编译，包括类型替换，然后我们在函数体中得到语法错误。我们可以轻松地修改这些示例，看看这种情况是如何发生的。让我们从 `find()` 函数开始：

```cpp
// Example 17
template <typename T, typename V>
bool find(const T& cont, V val) {
  for (typename T::value_type x : cont) {
    if (x == val) return true;
  }
  return false;
}
```

这次，我们决定接受任何类型的值。这本身并不一定错误，但我们的模板函数的主体是在假设容器类型 `T` 有嵌套类型 `value_type`，并且这个类型可以与类型 `V` 相比较的情况下编写的。如果我们用错误的参数调用函数，调用仍然可以编译，因为模板声明中的替换对参数类型没有特别的要求。但然后我们在模板本身的主体中得到语法错误，而不是在调用位置。

类似的情况也可能发生在`compute()`模板中：

```cpp
// Example 17
template <typename U, typename V>
auto compute(U u, V v) {
  std::common_type_t<U, V> res = (u > v) ? u : v;
  return res;
}
```

这个模板函数可以对任何两个参数进行调用，但如果没有为两者提供一个共同类型，那么它将无法编译。

注意两种替换失败之间的非常显著的区别：如果失败发生在 SFINAE 上下文中，函数将从重载解析中移除，就像它不存在一样。如果有另一个重载（具有相同名称的函数），它将被考虑，并最终可能被调用。如果没有，我们将在调用位置得到一个语法错误，这归结为“没有这样的函数。”另一方面，如果失败发生在模板的主体中（或在 SFINAE 规则未覆盖的其他地方），假设函数是最好的或唯一的重载，它将被调用。客户端代码——调用本身——将编译正常，但模板将不会。

有几个原因使得第一个选项更可取。首先，调用者可能想要调用一个不同的重载版本，这个版本可以正常编译，但由于模板重载解析的规则复杂，错误地选择了错误的重载。调用者可能很难修复这个错误并强制选择预期的重载。其次，当模板的主体在编译失败时，你收到的错误信息通常难以理解。我们的例子很简单，但在更现实的情况下，你可能会看到一个涉及一些你一无所知的内部类型和对象的错误。最后一个原因是更概念性的陈述：模板函数的接口，就像任何其他接口一样，应该尽可能完整地描述对调用者的要求。接口是一个合同；如果调用者遵守了它，函数的实现者必须履行承诺。

假设我们有一个模板函数，其主体对类型参数有一些要求，而这些要求没有以自然、直接的方式（类型替换成功但模板无法编译）被接口捕获。将硬替换失败转换为 SFINAE 失败的唯一方法是在 SFINAE 上下文中使其发生。为此，我们需要在接口中添加一些不是声明函数所必需的东西。这种添加的唯一目的是触发替换失败，并在函数主体中导致编译错误之前，将函数从重载解析集中移除。这种“人工”的失败是第三种替换失败。以下是一个例子，我们强制要求类型是指针，尽管接口本身即使没有这个要求也可以正常工作：

```cpp
// Example 18
template <typename U, typename V>
auto compare(U pu, V pv) -> decltype(bool(*pu == *pv)) {
  return *pu < *pv;
}
```

此函数接受两个指针（或任何其他可以解引用的指针类对象）并返回比较它们指向的值的布尔结果。为了使函数体能够编译，两个参数都必须是可以解引用的。此外，解引用它们的结果必须可以比较相等。最后，比较的结果必须可以转换为 bool。尾随的返回类型声明是不必要的：我们本来可以直接声明函数返回 `bool`。但它确实有影响：它将可能的替换失败从函数体移动到其声明中，在那里它成为了一个 SFINAE 失败。除非 `decltype()` 内部的表达式无效，否则返回类型始终是 `bool`。这可能会发生的原因与函数体无法编译的原因相同：其中一个参数无法解引用，值不可比较，或者比较的结果无法转换为 `bool`（后者通常是多余的，但我们还是应该强制整个合同）。

注意到“自然”和“人工”替换失败之间的界限并不总是清晰的。例如，有人可能会争论，之前使用 `std::common_type_t<U, V>` 作为返回类型是人工的（第三种替换失败，而不是第一种），“自然”的方式应该是声明返回类型为 `auto`，并让函数体在无法推导出公共类型时失败。确实，这种差异通常归结为程序员的风格和意图：如果不是因为需要强制类型限制，程序员是否仍然会在模板声明中写出类型表达式？

第一类失败的解决方法是直接的：模板接口本身形成一个合同，尝试调用违反了合同，并且函数没有被调用。理想情况下，应完全避免第二类失败。但为了做到这一点，我们必须使用第三类失败，即 SFINAE 上下文中的人工替换失败。本章的其余部分将讨论编码此类接口限制模板合同的方法。自从 C++的第一天起，SFINAE 技术就被用来人为地引起替换失败，从而将这些函数从重载集中删除。C++20 为解决这个问题添加了一种全新的机制：概念。在我们讨论使用 SFINAE 控制重载解析之前，我们需要更多地了解这种语言最新的补充。

# C++20 中的概念和约束

本章的其余部分都是关于添加到模板声明中用于对模板参数施加限制的“人工”替换失败。在本节中，我们将了解 C++20 中编码这些限制的新方法。在下一节中，我们将展示如果你不能使用 C++20 但仍想约束你的模板，你可以做什么。

## C++20 中的约束

C++20 通过引入概念和约束来改变我们使用 SFINAE 限制模板参数的方式。尽管整体特性通常被称为“概念”，但约束才是最重要的部分。以下内容不是这些特性的完整或正式描述，而是一种最佳实践的演示（由于社区仍在确定哪些是足够广泛接受的，因此说“模式”可能还为时过早）。

指定约束的第一种方式是通过编写一个具有以下形式的 `requires` 子句：

```cpp
requires(constant-boolean-expression)
```

关键字 `requires` 和括号中的常量（编译时）表达式必须出现在模板参数之后，或者作为函数声明的最后一个元素：

```cpp
// Example 19
template <typename T> requires(sizeof(T) == 8) void f();
template <typename T> void g(T p) requires(sizeof(*p) < 8);
```

就像在上一节中一样，声明末尾的约束可以按名称引用函数参数，而参数列表之后的约束只能引用模板参数（这两种语法之间的其他，更微妙的不同超出了本章的范围）。与上一节不同的是，如果约束失败，编译器通常会提供一个清晰的诊断信息，而不是简单地报告“找不到函数 f”和模板推导失败。

在 `requires` 子句中的常量表达式中可以使用什么？实际上，任何可以在编译时计算的表达式都可以，只要整体结果是 `bool`。例如，可以使用类型特性如 `std::is_convertible_v` 或 `std::is_default_constructible_v` 来限制类型。如果表达式复杂，`constexpr` 函数可以帮助简化它们：

```cpp
template <typename V> constexpr bool valid_type() {
  return sizeof(T) == 8 && alignof(T) == 8 &&
    std::is_default_constructible_v<T>;
}
template <typename T> requires(valid_type<T>()) void f();
```

但有一个我们之前没有见过的特殊表达式——`requires` 表达式。这个表达式可以用来检查某个任意表达式是否可以编译（技术上，它“是有效的”）：

```cpp
requires { a + b; }
```

假设 `a` 和 `b` 的值是在表达式使用的上下文中定义的，如果表达式 `a + b` 是有效的，则此表达式评估为 `true`。如果我们知道我们想要测试的类型，但没有变量怎么办？那么我们可以使用 `requires` 表达式的第二种形式：

```cpp
requires(A a, B b) { a + b; }
```

类型 `A` 和 `B` 通常指的是模板参数或某些依赖类型。

注意，我们说的是“任意表达式是有效的”，而不是“任意代码是有效的”。这是一个重要的区别。例如，你不能写

```cpp
requires(C cont) { for (auto x: cont) {}; }
```

并要求类型 `C` 满足范围-for 循环的所有要求。大多数时候，你测试的是像 `cont.begin()` 和 `cont.end()` 这样的表达式。然而，你也可以通过在 lambda 表达式中隐藏代码来提出更复杂的要求：

```cpp
requires(C cont) {
  [](auto&& c) {
    for (auto x: cont) { return x; };
  }(cont);
}
```

如果这样的代码失败，你必须找出错误信息，那可就糟糕了。

当在模板约束中使用 `requires` 表达式时，模板的限制不是由特定的特性，而是由类型所需的行为来决定的：

```cpp
// Example 20
template <typename T, typename P>
void f(T i, P p) requires( requires { i = *p; } );
template <typename T, typename P>
void f(T i, P p) requires( requires { i.*p; } );
```

首先，是的，有两个关键字 `requires`（顺便说一句，在这种情况下括号是可选的，你可以找到这个约束被写成 `requires requires`）。第一个 `requires` 引入了一个约束，一个 `requires` 子句。第二个 `requires` 开始了 `requires` 表达式。第一个函数 `f()` 中的表达式在第二个模板参数 `p` 可以解引用（它可以是一个指针，但不一定必须是）并且结果可以赋值给第一个参数 `i` 时是有效的。我们不需要要求赋值两边的类型相同，甚至不需要 `*p` 可以转换为 `T`（通常是这样的，但不是必需的）。我们只需要 `i = *p` 这个表达式能够编译。最后，如果我们没有现成的正确变量，我们可以将它们声明为 `requires` 表达式的参数：

```cpp
// Example 20
template <typename T, typename P>
requires(requires(T t, P p) { t = *p; }) void f(T i, P p);
template <typename T, typename P>
requires(requires(T t, P p) { t.*p; }) void f(T i, P p);
```

这两个例子还表明，我们可以使用约束来进行 SFINAE 覆盖控制：如果约束失败，模板函数将从重载解析集中移除，并且解析继续。

如我们所见，有时我们需要检查的不是表达式，而是一个依赖类型；我们也可以在 `requires` 表达式中这样做：

```cpp
requires { typename T::value_type; }
```

`requires` 表达式计算结果为 bool，因此它可以用在逻辑表达式中：

```cpp
requires(
  requires { typename T::value_type; } &&
  sizeof(T) <= 32
)
```

我们可以通过这种方式组合多个 `requires` 表达式，但也可以在单个表达式中编写更多的代码：

```cpp
requires(T t) { typename T::value_type; t[0]; }
```

在这里，我们要求类型 `T` 有一个嵌套类型 `value_type` 和一个接受整数索引的索引运算符。

最后，有时我们需要检查的不仅仅是某个表达式能否编译，还要检查它的结果是否具有某种类型（或满足某些类型要求）。这可以通过 `requires` 表达式的复合形式来完成：

```cpp
requires(T t) { { t + 1 } -> std::same_as<T>; }
```

在这里，我们要求表达式 `t + 1` 能够编译，并且产生与变量 `t` 本身相同类型的结果。最后一部分是通过一个概念完成的；你将在下一节中了解到它们，但现在是把它看作是编写 `std::is_same_v` 类型特性的一种替代方法。

说到概念……到目前为止我们所描述的一切都可以在任何 C++20 书籍的“概念”标题下找到，但我们还没有提到概念本身。这即将改变。

## C++20 中的概念

概念只是对一组要求的命名集合——就是我们刚刚学习到的那些要求。从某种意义上说，它们类似于 `constexpr` 函数，除了它们操作的是类型，而不是值。

当有一组经常引用的要求，或者你想要给它们一个有意义的名称时，你会使用概念。例如，范围由一个非常简单的要求定义：它必须有一个开始迭代器和结束迭代器。每次我们声明一个接受范围参数的函数模板时，我们都可以写一个简单的 `requires` 表达式，但这既方便又易于阅读，给这个要求一个名称：

```cpp
// Example 21
template <typename R> concept Range = requires(R r) {
  std::begin(r);
  std::end(r);
};
```

我们刚刚介绍了一个名为`Range`的概念，它有一个模板类型参数`R`；此类型必须具有开始和结束迭代器（我们使用`std::begin()`而不是成员函数`begin()`的原因是 C 数组也是范围，但没有成员函数）。

注意，C++20 有一个范围库和相应的一组概念（包括`std::ranges::range`，在任何实际代码中应使用它而不是我们自制的`Range`），但范围的概念是方便的教学材料，我们将用它来驱动示例。

一旦我们有一个命名的概念，我们就可以用它来代替在模板约束中详细说明的要求：

```cpp
// Example 21
template <typename R> requires(Range<R>) void sort(R&& r);
```

如你所见，概念可以在`requires`子句中使用，就像它是一个类型为`bool`的`constexpr`变量一样。确实，概念也可以在静态断言等上下文中使用：

```cpp
static_assert(Range<std::vector<int>>);
static_assert(!Range<int>);
```

对于概念是整个要求的简单模板声明，语言提供了一种更简单的方式来表述它：

```cpp
// Example 21
template <Range R> void sort(R&& r);
```

换句话说，可以在模板声明中使用概念名称代替`typename`关键字。这样做会自动将相应的类型参数限制为满足该概念的类型。如果需要，仍然可以使用`requires`子句来定义额外的约束。最后，概念也可以与新的 C++20 模板语法一起使用：

```cpp
// Example 21
void sort(Range auto&& r);
```

所有三个声明具有相同的效果，选择主要取决于风格和便利性。

## 概念和类型限制

我们已经看到如何使用概念和约束来对函数模板的参数施加限制。`requires`子句可以出现在模板参数之后或函数声明的末尾；这两个地方都是 SFINAE 上下文，任一位置的替换失败都不会停止整个程序的编译。在这方面，概念与替换失败没有本质的不同：虽然你可以在 SFINAE 上下文之外使用约束，但替换失败仍然是一个错误。例如，你不能通过使用约束来断言一个类型没有嵌套类型`value_type`：

```cpp
static_assert(!requires{ typename T::value_type; });
```

你可能期望如果未满足要求，`requires`表达式将评估为假，但在此情况下，它根本无法编译（你会得到错误信息，即`T::value_type`未引用一个有效的类型）。

然而，使用概念可以实施一些以前无法实现的限制。这些是针对类模板的要求。在最简单的情况下，我们可以使用一个概念来限制类模板的类型参数：

```cpp
// Example 21
template <Range R> class C { … };
```

此类模板只能用满足`Range`概念的类型实例化。

然后，我们可以约束单个成员函数，无论它们是否是模板：

```cpp
// Example 21
template <typename T> struct holder {
  T& value;
  holder(T& t) : value(t) {}
  void sort() requires(Range<T>) {
    std::sort(std::begin(value), std::end(value));
  }
};
```

现在类模板本身可以在任何类型上实例化。然而，如果类型满足`Range`约束，其接口才包括一个成员函数`sort()`。

这是在约束和旧 SFINAE 之间的一个非常重要的区别：人工替换失败只有在替换函数模板中推导出的类型参数时才会起到帮助作用。在本章的早期，我们不得不向一个成员函数添加一个虚拟模板类型参数，只是为了能够创建一个 SFINAE 失败。有了概念，就无需做这些了。

概念和约束是指定模板参数限制的最佳方式。它们使多年来发明的许多 SFINAE 技巧变得过时。但并非每个人都能访问 C++20。此外，即使有概念，一些 SFINAE 技术仍然在使用。在最后一节中，我们将学习这些技术，并了解如果你没有 C++20 但仍然想要约束模板类型，可以做什么。

# SFINAE 技术

模板参数替换失败不是错误——SFINAE 规则——必须添加到语言中，只是为了使某些狭窄定义的模板函数成为可能。但是，C++程序员的独创性没有界限，因此 SFINAE 被重新定位并利用来通过故意造成替换失败来手动控制重载集。多年来，发明了大量的基于 SFINAE 的技术，直到 C++20 的概念使其中大多数变得过时。

尽管如此，即使在 C++20 中，一些 SFINAE 的使用仍然存在，而且还有大量的 C++20 之前的代码，你可能需要阅读、理解和维护。

让我们从 SFINAE 的应用开始，即使有概念可用，这些应用仍然是有用的。

## C++20 中的 SFINAE

首先，即使在 C++20 中，仍然存在“自然”的类型替换失败。例如，你可能想编写这个函数：

```cpp
template <typename T> typename T::value_type f(T&& t);
```

这仍然是可行的，假设你真的想返回由嵌套的`value_type`给出的类型。然而，在你匆忙回答“是的”之前，你应该仔细检查你真正想要返回的类型是什么。你想要强制执行与调用者的哪种契约？也许`value_type`的存在被用作真实要求的代理，例如类型 T 具有索引操作符或可以用作迭代范围。在这种情况下，你现在可以直接声明这些要求，例如：

```cpp
template <typename T> auto f(T&& t)
requires( requires { *t.begin(); t.begin() != t.end(); } );
```

这意味着你真正需要的是一个具有成员函数`begin()`和`end()`的类型。这些函数返回的值（假设是迭代器）被解引用并比较；如果这些操作受支持，返回值对我们来说足够接近迭代器。最后，在前面的例子中，我们让编译器确定返回类型。这通常很方便，但缺点是接口——我们的契约——没有说明返回类型是什么；我们的代码的客户必须阅读实现。假设我们返回通过解引用迭代器得到的值，我们可以明确地指出这一点：

```cpp
template <typename T> auto f(T&& t)->decltype(*t.begin())
requires( requires {
  *t.begin();
  t.begin() != t.end();
  ++t.begin();
} );
```

这是一个非常全面的客户合同，当然，前提是我们作为实施者保证如果满足所列要求，函数的主体将能够编译。否则，合同是不完整的：例如，如果我们确实在函数的主体中使用了`T::value_type`，我们应该将`typename T::value_type`添加到要求列表中，无论这最终返回的类型是什么（如果是，我们仍然可以使用 SFINAE 来处理返回类型，这没有问题）。

当使用依赖类型声明模板函数参数时，也存在类似的考虑，例如：

```cpp
template <typename T>
bool find(const T& t, typename T::value_type x);
```

再次，我们应该问自己这些是否真的是我们想要施加的要求。假设函数正在查找容器`t`中的值`x`，只要它可以与容器中存储的值进行比较，我们是否真的关心`x`的类型？考虑这个替代方案：

```cpp
template <typename T, typename X>
bool find(const T& t, X x)
requires( requires {
  *t.begin() == x;
  t.begin() == t.end();
  ++t.begin();
} );
```

现在我们要求容器具有范围-for 循环所需的一切，并且容器中存储的值可以与`x`进行比较以实现相等。假设我们只是迭代容器，如果找到等于`x`的值则返回 true，这就是我们从调用者那里需要的要求。

你不应该推断出在 C++20 中“自然”的 SFINAE 不再使用，而是被独立的模板参数和约束绑定所取代。我们建议的只是检查你的代码，以确定接口表达并通过 SFINAE 执行的合同是否真的是你想要的，或者仅仅是编写代码方便的。在后一种情况下，概念提供了一种表达你真正想要要求但无法（但请继续阅读，因为有一些受概念启发的技术可以在 C++20 之前使用并满足相同需求）的方法。另一方面，如果模板函数最好以在客户端提供无效参数时触发替换失败的方式编写，那么，无论如何，继续使用 SFINAE——没有必要重写一切以使用概念。

即使是“人工”的 SFINAE 在 C++20 中仍然有用途，正如我们即将看到的。

## SFINAE 和类型特性

在 C++20 中，“人工”SFINAE 最重要的应用是编写类型特性。类型特性不会消失：即使你在代码中将`std::is_same_v`（特性）替换为`std::same_as`（概念），你应该知道概念的实现使用了它所取代的特性。

并非所有类型特性都需要使用 SFINAE，但许多确实需要。这些特性检查某些语法特征的存在，例如嵌套类型的存在。这些特性的实现面临一个共同问题：如果类型没有所需的功能，某些代码将无法编译。但我们不希望出现编译错误。我们希望表达式评估为`false`。那么我们如何让编译器忽略错误呢？当然是通过使其在 SFINAE 上下文中发生。

让我们从整个章节中一直阻碍我们的一个例子开始：我们将编写一个特性来检查一个类型是否有嵌套类型 `value_type`。我们将使用 SFINAE，因此需要一个模板函数。这个函数必须在一个 SFINAE 上下文中使用嵌套类型。有几个选择。通常，添加一个依赖于可能失败的表达式的模板参数是很方便的，例如：

```cpp
template <typename T, typename = T::value_type> void f();
```

注意，第二个参数没有名字——我们从未使用过它。如果我们尝试用任何没有嵌套 `value_type` 的类型 `T` 实例化这个模板，例如 `f<int>()`，替换将失败，但这不是错误（SFINAE！）。当然，当我们写 `f(ptr)` 时没有函数可以调用是一个错误，所以我们必须提供一个后备的重载：

```cpp
template <typename T> void f(…);
```

你可能会觉得“双重通用”的 `template f(...)` 函数的概念很奇特——它可以接受任何类型的任何参数，甚至在没有模板的情况下，那么为什么还要使用模板呢？当然，如果一个显式指定类型的调用，例如 `f<int>()`，会把这个函数视为一个可能的重载（记住，通过指定模板参数类型，我们也排除了所有非模板函数的考虑）。然而，我们希望这个重载的优先级尽可能低，只要存在，就优先选择第一个重载。这就是为什么我们使用 `f(…)`，这是“最后的手段”的重载。唉，`f()` 和 `f(…)` 的重载仍然被认为是模糊的，所以我们需要至少有一个参数。只要我们能轻松地构造出该类型的对象，参数的类型就无关紧要：

```cpp
template <typename T, typename = T::value_type>
void f(int);
template <typename T> void f(…);
```

现在，如果 `T::value_type` 是一个有效的类型，调用 `f<T>(0)` 将选择第一个重载。否则，只有一个重载可供选择，即第二个。我们需要的只是一个方法来确定如果进行调用，会选择哪个重载，而不必实际进行调用。

这实际上非常简单：我们可以使用 `decltype()` 来检查函数的结果类型（在 C++11 之前，使用 `sizeof()`）。现在，我们只需要给这两个重载不同的返回类型。可以使用任何两种不同的类型。然后，我们可以根据这些类型编写一些条件代码。然而，记住我们正在编写一个类型特性，检查存在性的特性通常会在存在时结束为 `std::true_type`，不存在时为 `std::false_type`。我们没有理由使我们的实现过于复杂——我们只需从两个重载中返回所需类型，并将其用作特性：

```cpp
// Example 22
namespace detail {
template <typename T, typename = T::value_type>
void test_value_type(int);
template <typename T> void test_value_type (…);
}
template <typename T> using has_value_type =
  decltype(detail::test_value_type <T>(0));
```

由于函数从未被调用，只是在`decltype()`内部使用，我们不需要提供函数的定义，只需要它们的声明（但请参见下一节，以获得更完整和细致的解释）。为了避免将客户端不需要关心的测试函数污染全局命名空间，通常的做法是将它们隐藏在`detail`或`internal`等命名空间中。说到惯例，我们应该定义两个别名：

```cpp
template <typename T>
using has_value_type_t = has_value_type<T>::type;
template <typename T> inline constexpr
bool has_value_type_v = has_value_type<T>::value;
```

现在，我们可以像使用任何标准特质一样使用我们的特质，例如：

```cpp
static_assert(has_value_type_v<T>, “I require value_type”);
```

正如我们之前看到的，我们还可以使用几个其他 SFINAE 上下文来“隐藏”使用`T::value_type`时可能出现的潜在错误。可以使用尾返回类型，但这并不方便，因为我们已经有一个需要的返回类型（有一种方法可以绕过这一点，但它比其他替代方案更复杂）。此外，如果我们需要使用 SFINAE 与构造函数一起，那么返回类型不是一种选择。

另一种常见的技术是在函数中添加额外的参数；替换错误发生在参数类型中，并且参数必须具有默认值，这样调用者甚至不知道它们的存在。这曾经更受欢迎，但我们正在远离这种做法：虚拟参数可能会干扰重载解析，并且可能很难为这样的参数找到可靠的默认值。

另一种正在成为标准做法的技术是将替换失败发生在可选的非类型模板参数中：

```cpp
// Example 22a
template <typename T, std::enable_if_t<
  sizeof(typename T::value_type) !=0, bool> = true>
std::true_type test_value_type(int);
```

这里我们有一个非类型模板参数（一个类型为`bool`的值）和一个默认值`true`。在这个参数中替换类型`T`可能会以与这一节中所有早期失败相同的方式失败：如果嵌套类型`T::value_type`不存在（如果存在，逻辑表达式`sizeof(…) != 0`永远不会失败，因为任何类型的尺寸都是非负的）。这种方法的优点是，如果我们需要同时检查多个失败，更容易组合多个表达式，例如：

```cpp
template <typename T, std::enable_if_t<
  sizeof(typename T::value_type) !=0 &&
  sizeof(typename T::size_type) !=0, bool> = true>
std::true_type test_value_type(int);
```

这种技术有时会与默认值中的失败表达式一起使用，而不是使用类型：

```cpp
template <typename T,
          bool = sizeof(typename T::value_type)>
std::true_type test_value_type(int);
```

这是一个坏习惯：虽然它有时有效，看起来也更容易编写，但它有一个主要的缺点。通常，你需要声明几个具有不同条件的重载，以便只有一个成功。你可以使用前面的方法来做到这一点：

```cpp
template <typename T, std::enable_if_t<cond1, bool> = true>
res_t func();
template <typename T, std::enable_if_t<cond2, bool> = true>
res_t func(); // OK as long as only one cond1,2 is true
```

但你不能这样做：

```cpp
template <typename T, bool = cond1> = true>
res_t func();
template <typename T, bool = cond2 > = true>
res_t func();
```

两个具有相同参数但不同默认值的模板被认为是重复声明，即使其中一个条件`cond1`或`cond2`总是导致替换失败。更好的做法是养成在非类型参数的类型中使用（可能失败的）条件的习惯。

为了回顾我们关于 SFINAE 所学到的一切，让我们再写一个特质。这次，我们将检查一个类型是否是类：

```cpp
// Example 23
namespace detail {
template <typename T> std::true_type test(int T::*);
template <typename T> std::false_type test(...);
}
template <typename T>
using is_class = decltype(detail::test<T>(nullptr));
```

类与不是类之间的关键区别在于，类有成员，因此有成员指针。这次最简单的方法是声明一个成员函数参数，该参数是成员指针（无论是什么类型的成员，我们都不会调用该函数）。如果类型 T 没有任何成员，则在参数类型`T::*`中发生替换失败。

这几乎与标准特质`std::is_class`的定义完全相同，但它还检查了联合：联合不被`std::is_class`视为类，但实现`std::is_union`需要编译器支持，而不是 SFINAE。

我们学到的技术使我们能够编写任何检查类型特定属性的特质：它是否是一个指针，它是否有嵌套类型或成员等。另一方面，概念使得检查行为变得容易：类型是否可以被解引用，两个类型是否可以比较等？请注意，我说的是“容易”而不是“可能”：你可以使用概念来检查非常狭窄定义的特征，你可以使用特质来检测行为，但这并不直接。

本章主要面向那些在应用代码中编写模板和模板库的程序员：如果你编写了一个具有 STL 复杂性和严谨性的库，你需要在你的定义上非常精确（你还需要一个标准委员会来辩论并精确到必要的程度）。对于我们其他人来说，“如果`*p`可以编译，则调用`f(p)`”提供的正式程度通常足够。在 C++20 中，我们可以使用概念来实现这一点。如果你还没有使用 C++20，你必须使用 SFINAE 技术之一。本章讨论了几种这样的技术；社区在多年中发展了更多。然而，概念的发展对这些实践产生了有趣的影响：除了我们可以在 C++20 中直接使用的工具外，标准还提供了一种思考这个问题的方法，这种方法的应用范围更广。因此，一些与概念相似的技术（例如，在尾随的`decltype()`中测试行为）变得越来越受欢迎，而其他实践则不再受欢迎。甚至有人尝试使用 C++20 之前的语言特性来实现一个概念库。当然，不可能复制概念；在许多方面，我们甚至无法接近。然而，即使我们不能使用该语言本身，我们仍然可以从开发概念语言所投入的思维中受益。因此，我们可以“在精神上”使用 SFINAE，这提供了一种一致的方式来实现基于 SFINAE 的限制，而不是一个临时的技术集合。以下是一种在不使用 C++20 的情况下实现类似概念限制的方法。

## 概念之前的技术

我们的目标并不是在这里实现一个完整的概念库：你可以在网上找到这样的库，这本书是关于设计模式和最佳实践的，而不是编写特定的库。本节的目标是从众多可用的选项中挑选出一些最佳的基于 SFINAE 的技术。这些技术尽可能符合基于概念的心态。我们没有选择的方法和技巧并不一定比其他方法差，但本节提供了一套 SFINAE 工具和实践，它是一致的、统一的，并且足以满足绝大多数应用程序开发者的需求。

正如与真实的概念一样，我们需要两种类型的实体：概念和限制。

如果你看看概念的使用方式，它们强烈地类似于常量布尔变量：

```cpp
template <typename R> concept Range = …;
template <typename R> requires(Range<R>) void sort(…);
```

`requires()`子句需要一个布尔值，它不仅限于概念（考虑表达式`requires(std::is_class_v<T>)`）。因此，`Range<R>`概念就像一个布尔值。不可避免地，我们将使用`constexpr bool`变量来代替概念以模拟它们的行为。从`Range<R>`与`std::is_class_v<T>`的比较中，我们还可以推断出，类似于特质的机制可能是实现概念的最佳选择：毕竟`std::is_class_v`也是一个`constexpr bool`变量。

从上一节中我们学习的特质的实现中，我们知道我们需要两个重载：

```cpp
template <typename R> constexpr yes_t RangeTest(some-args);
template <typename R> constexpr no_t RangeTest(...);
```

第一个重载对于满足`Range`要求的任何类型`R`都是有效且首选的（一旦我们弄清楚如何实现）。第二个重载始终可用但不是首选的，因此只有在没有其他重载可用时才会调用。

我们可以通过返回类型（`yes_t`和`no_t`只是对我们尚未选择的某些类型的占位符）来确定调用了哪个重载。但有一个更简单的方法；我们需要的只是`Range`“概念”的一个常量布尔值，所以为什么不直接让`constexpr`函数返回正确的值，如下所示：

```cpp
template <typename R> constexpr bool RangeTest(some-args) {
  return true;
}
template <typename R> constexpr bool RangeTest(...) {
  return false;
}
template <typename R>
constexpr inline bool Range = RangeTest<R>(0);
```

最后两个语句（变量和后备重载）已经完成。“所有”我们需要的只是确保当`R`不是范围时，第一个重载会因替换失败而失败。那么，在我们的目的中，什么是范围呢？就像我们在*Concepts in C++20*这一节中所做的那样，我们将定义范围为任何具有`begin()`和`end()`方法的类型。由于我们正在测试特定的行为，这可能会失败编译，但不应导致错误，因此我们应该在 SFINAE 上下文中触发这种失败。正如我们已经看到的，这种可能无效的代码的最容易放置位置是尾随返回类型：

```cpp
template <typename R>
constexpr auto RangeTest(??? r) -> decltype(
  std::begin(r),        // Ranges have begin()
  std::end(r),         // Ranges have end()
  bool{}            // But return type should be bool
) { return true; }
```

后置返回类型让我们能够编写使用参数名称的代码。我们只需要一个类型为`R`的参数`r`。当使用 SFINAE 调用任何预期被调用的模板函数时，这很容易做到。但这个函数永远不会用实际的范围来调用。我们可以尝试声明一个类型为`R&`的参数，然后用默认构造的范围`R{}`来调用该函数，但这不会起作用，因为`constexpr`函数必须具有`constexpr`参数（否则它们仍然可以被调用，但不是在常量表达式中，即在编译时），而`R{}`对于大多数范围来说不会是一个`constexpr`值。

我们可以完全放弃使用引用，改用指针：

```cpp
// Example 24
template <typename R>
constexpr auto RangeTest(R* r) -> decltype(
  std::begin(*r),    // Ranges have begin()
  std::end(*r),         // Ranges have end()
  bool{}            // But return type should be bool
) { return true; }
template <typename R> constexpr bool RangeTest(...) {
  return false;
}
template <typename R>
constexpr inline bool Range = RangeTest<R>(nullptr);
```

虽然你可能预计“概念类似”的 SFINAE 将会非常复杂，但实际上这正是你需要定义一个如`Range`这样的概念：

```cpp
static_assert(Range<std::vector<int>>);
static_assert(!Range<int>);
```

这两个语句看起来与它们的 C++20 等价物完全一样！我们的“概念”甚至在 C++14 中也能工作，只是那里没有`inline`变量，所以我们必须使用`static`代替。

在现在完成概念之后，我们还需要对约束做一些处理。在这里，我们的成功将受到很大的限制。首先，由于我们正在使用 SFINAE，我们只能对模板函数参数施加限制（正如我们所见，C++20 约束甚至可以应用于非模板函数，例如类模板的成员函数）。此外，我们在哪里可以编写这些约束也非常有限。最通用的方法是将非模板参数添加到模板中并在那里测试约束：

```cpp
template <typename R,
    std::enable_if_t<Range<R>, bool> = true>
void sort(R&& r);
```

我们可以在宏中隐藏模板代码：

```cpp
// Example 24
#define REQUIRES(...) \
  std::enable_if_t<(__VA_ARGS__), bool> = true
template <typename R, REQUIRES(Range<R>)> void sort(R&& r);
```

可变参数宏巧妙地解决了宏在参数为代码时常见的难题：逗号被解释为参数之间的分隔符。这当然没有 C++20 约束那么方便，但几乎是最接近的方法了。

现在让我们回到概念上来。我们之前写的内容是有效的，但有两个问题：首先，那里也有大量的模板代码。其次，我们必须使用指针来引入我们稍后可以用来测试所需行为的函数参数名称。这限制了我们可以要求的行为，因为函数可以通过引用传递参数，行为可能依赖于所使用的引用类型，而我们无法形成引用的指针。实际上，我们刚才写的代码在许多情况下是无法编译的，因为模板函数`sort()`的参数`R`的类型被推断为引用。为了可靠地使用它，我们必须检查其底层类型：

```cpp
// Example 24
template <typename R, REQUIRES(Range<std::decay_t<R>>)>
void sort(R&& r);
```

如果我们可以使用引用参数，那将更加方便，但这样我们又回到了之前遇到的问题：如何调用这样的函数？我们不能使用对应类型的值，例如`R{}`，因为它不是一个常量表达式。如果我们尝试将`R{}`用作默认参数值，也会出现同样的问题——它仍然不是一个常量表达式。

就像软件工程中的大多数问题一样，这个问题可以通过添加另一个间接层来解决：

```cpp
template <typename R>
constexpr static auto RangeTest(R& r = ???) ->
  decltype(std::begin(r), std::end(r));
template <typename R>        // Overload for success
constexpr static auto RangeTest(int) ->
  decltype(RangeTest<R>(), bool{}) { return true; }
template <typename R>        // Fallback overload
constexpr bool RangeTest(...) { return false; }
template <typename R>
constexpr static bool Range = RangeTest<R>(0);
```

我们的后备重载保持不变，但如果 SFINAE 测试成功，将要被调用的重载现在尝试在 `decltype` 上下文中调用 `RangeTest(r)`（此外，我们回到了使用 `int` 而不是指针作为占位参数）。最后一个问题是使用什么作为参数 `r` 的默认值。

在代码中获取永远不会被调用的对象的引用的常用方法是 `std::declval`，因此我们可能想要这样写：

```cpp
template <typename R>
constexpr static auto RangeTest(R& r=std::declval<R>()) ->
  decltype(std::begin(r), std::end(r));
```

不幸的是，这不会编译，错误信息可能类似于“`std::declval` 不能使用。”这很奇怪，我们实际上并没有使用它（整个函数仅在 `decltype()` 内部使用），但让我们尝试解决这个问题。毕竟，`std::declval` 中没有魔法，我们只需要一个返回我们对象引用的函数：

```cpp
template <typename T> constexpr T& lvalue();
template <typename R>
constexpr static auto RangeTest(R& r = lvalue<R>()) ->
  decltype(std::begin(r), std::end(r));
```

在一个符合标准的编译器上，这同样不会编译，但错误将不同，这次编译器会说出类似这样的话：

```cpp
inline function 'lvalue<std::vector<int>>' is used but not defined."
```

好的，我们可以定义这个函数，但请确保它永远不会被调用：

```cpp
template <typename T> constexpr T& lvalue() { abort(); }
template <typename R>
constexpr static auto RangeTest(R& r = lvalue<R>()) ->
  decltype(std::begin(r), std::end(r));
```

添加 `{ abort(); }` 会带来很大的不同——程序现在可以编译了，并且在添加了其余缺失的部分之后，它可以在不终止的情况下运行。这正是应该的：函数 `lvalue()` 仅在 `decltype` 内部使用，其实现根本无关紧要。我不会再让你悬着了，这是一个与标准本身相关的问题；如果你想要深入了解棘手细节，可以在这里跟随 *核心问题 1581*：[`www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0859r0.html`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0859r0.html)。现在，我们只能保留这个无用的函数体（这不会伤害到任何东西）。当然，我们也可以为初始化默认的右值参数以及 `const` 左值引用定义类似的函数，并将它们包含在某个仅用于实现的命名空间中：

```cpp
namespace concept_detail {
template <typename T>
  constexpr const T& clvalue() { abort(); }
template <typename T> constexpr T& lvalue() { abort(); }
template <typename T> constexpr T&& rvalue() { abort(); }
}
```

现在我们可以定义测试我们想要的引用类型的行为的概念：

```cpp
// Example 24a
template <typename R>
constexpr static auto RangeTest(
  R& r = concept_detail::lvalue<R>()) ->
  decltype(std::begin(r), std::end(r));
template <typename R>
constexpr static auto RangeTest(int) ->
  decltype(RangeTest<R>(), bool{}) { return true; }
template <typename R>
constexpr bool RangeTest(...) { return false; }
template <typename R>
constexpr static bool Range = RangeTest<R>(0);
```

包括我们的 `REQUIRES` 宏在内的约束仍然以完全相同的方式工作（毕竟，概念本身并没有改变——`Range` 仍然是一个常量布尔变量）。

仍然存在样板代码的问题；实际上，我们有了更多难以处理的默认参数值。不过，借助一些宏，这最容易处理：

```cpp
// Example 25
#define CLVALUE(TYPE, NAME) const TYPE& NAME = \
  Concepts::concept_detail::clvalue<TYPE>()
#define LVALUE(TYPE, NAME) TYPE& NAME = \
  Concepts::concept_detail::lvalue<TYPE>()
#define RVALUE(TYPE, NAME) TYPE&& NAME = \
  Concepts::concept_detail::rvalue<TYPE>()
```

在这三个模板函数（如 `RangeTest`）中，第一个函数相当于 C++20 的 `concept` 声明——这就是我们想要要求的行为被编码的地方。除了这些宏之外，实际上无法再缩短它：

```cpp
// Example 25
template <typename R> CONCEPT RangeTest(RVALUE(R, r)) ->
  decltype(std::begin(r), std::end(r));
```

在这里，我们也定义了一个宏：

```cpp
#define CONCEPT constexpr inline auto
```

这样做并不是为了缩短代码，而是为了让读者（如果不是编译器）清楚地知道我们正在定义一个概念。将其与 C++20 的语句进行比较：

```cpp
template <typename R> concept Range =
  requires(R r) { std::begin(r); std::end(r); };
```

其他两个重载（`RangeTest(int)`和`RangeTest(…)`），以及概念变量的定义本身，可以很容易地使任何概念通用（当然，除了名称之外）。事实上，唯一从一个概念到另一个概念有所不同的声明是第一个：

```cpp
template <typename R>
constexpr static auto RangeTest(int) ->
  decltype(RangeTest<R>(), bool{}) { return true; }
```

如果我们使用变长模板，我们可以使它适用于任何概念测试函数：

```cpp
// Example 25
template <typename… T>
constexpr static auto RangeTest(int) ->
  decltype(RangeTest<T…>(), bool{}) { return true; }
```

由于我们所有的参数宏，例如`LVALUE()`，都包含了每个参数的默认值，所以函数总是可以在没有参数的情况下调用。我们必须注意我们定义的测试函数可能与`RangeTest(int)`函数冲突的可能性。这里不会发生这种情况，因为`int`不是一个有效的范围，但对于其他参数可能会发生。由于我们控制这些常见的重载以及概念变量的定义本身，我们可以确保它们使用一个不会与我们在常规代码中可能编写的任何内容冲突的参数：

```cpp
// Example 25
struct ConceptArg {};
template <typename… T>
constexpr static auto RangeTest(ConceptArg, int) ->
  decltype(RangeTest<T…>(), bool{}) { return true; }
template <typename T>
constexpr bool RangeTest(ConceptArg, ...) { return false; }
template <typename R>
constexpr static bool Range = RangeTest<R>(ConceptArg{},0);
```

这段代码对于所有概念都是相同的，除了像`Range`和`RangeTest`这样的名称。一个宏可以仅通过两个命名参数生成所有这些行：

```cpp
// Example 25
#define DECLARE_CONCEPT(NAME, SUFFIX) \
template <typename... T> constexpr inline auto     \
  NAME ## SUFFIX(ConceptArg, int) -> \
  decltype(NAME ## SUFFIX<T...>(), bool{}){return true;} \
template <typename... T> constexpr inline bool \
  NAME ## SUFFIX(ConceptArg, ...) { return false; } \
template <typename... T> constexpr static bool NAME = \
  NAME ## SUFFIX<T...>(ConceptArg{}, 0)
```

我们没有这样做是为了简洁，但如果你想在代码中使用这些类似概念的工具，你应该将这些所有实现细节隐藏在一个命名空间中。

现在我们可以如下定义我们的范围概念：

```cpp
// Example 25
template <typename R> CONCEPT RangeTest(RVALUE(R, r)) ->
  decltype(std::begin(r), std::end(r));
DECLARE_CONCEPT(Range, Test);
```

多亏了变长模板，我们不仅限于只有一个模板参数的概念。这是一个可以相加的两个类型的概念：

```cpp
// Example 25
template <typename U, typename V> CONCEPT
  AddableTest(CLVALUE(U, u), CLVALUE(V, v)) ->
  decltype(u + v);
DECLARE_CONCEPT(Addable, Test);
```

作为比较，这是 C++20 版本的样子：

```cpp
template <typename U, typename V> concept Addable =
  require(U u, V v) { u + v; }
```

当然，它要短得多，也强大得多。但 C++14 版本几乎是最接近的（这不是唯一的方法，但它们都产生类似的结果）。

这些“假概念”可以用来约束模板，就像 C++20 概念一样：

```cpp
 // Example 25
template <typename R, REQUIRES(Range<R>)> void sort(R&& r);
```

好吧，并不完全像 C++20 的概念——我们限制在模板函数中，任何要求都必须至少涉及一个模板参数。所以，如果你想限制模板类的非模板成员函数，你必须玩模板游戏：

```cpp
template <typename T> class C {
  template <typename U = T, REQUIRE(Range<U>)> void f(T&);
  …
};
```

但我们最终确实得到了相同的结果：对向量的排序调用可以编译，而对非范围对象的排序调用则不行：

```cpp
std::vector<int> v = …;
sort(v);         // OK
sort(0);        // Does not compile
```

不幸的是，我们的伪概念在错误信息方面确实存在不足——一个 C++20 编译器通常会告诉我们哪个概念没有满足以及原因，而模板替换错误信息并不容易解读。

顺便说一句，当你编写测试以确保某些内容无法编译时，你现在可以使用一个概念（或伪概念）来做这件事：

```cpp
// Example 25
template <typename R>
CONCEPT SortCompilesTest(RVALUE(R, r))->decltype(sort(r));
DECLARE_CONCEPT(SortCompiles, Test);
static_assert(SortCompiles<std::vector<int>>);
static_assert(!SortCompiles<int>);
```

C++20 版本留给你作为练习。

在我们结束这一章之前，让我们看看 SFINAE 和概念在模板中使用时的推荐和最佳实践。

# 限制模板——最佳实践

我们在整章中遇到了最有用的 SFINAE 和概念化技术，并推荐了它们，但由于要覆盖的材料很多，因此简要重申这些指南可能是有帮助的。这些指南主要针对在应用程序代码中使用模板的程序员。这包括基础代码，如应用程序的核心模板库，但编写库（如 STL）的程序员，该库旨在在极端多变的情况下尽可能广泛地使用，并在正式标准中非常精确地记录，会发现这些指南在精确性和形式上有所欠缺：

+   学习 SFINAE 的基本规则：它在哪些上下文中适用（声明）以及在哪些上下文中不适用（函数体）。

+   从在模板声明中使用参数依赖类型和在尾随返回类型中使用参数依赖表达式产生的 SFINAE 的“自然”使用几乎总是表达模板参数约束的最简单方式（但请参见下一条指南）。

+   反问自己你是否使用了依赖类型，例如`T::value_type`，因为这正是你在使用它的上下文中正确的类型，或者它只是比在接口上编写真实约束（例如“任何可以转换为`T::value_type`的类型”）更简单？在后一种情况下，这一章应该已经说服你，这样的限制并不难表达。

+   在合理的情况下，通过使用额外的模板参数及其必要的约束来使你的模板更通用（而不是使用`T::value_type`作为参数类型，使用另一个模板参数并将其约束为可转换为`T::value_type`）。

+   如果你使用 C++20 并且可以访问概念，请避免使用“人工”的 SFINAE，即不要创建仅用于约束模板的替换失败。根据需要使用`requires`子句，无论是否使用概念。

+   如果你不能使用 C++20 的概念，选择一个通用的统一方法来处理基于 SFINAE 的约束，并遵循它。即使你不能使用语言工具，也要利用为 C++20 开发的概念化方法：在应用基于 SFINAE 的技术时，遵循相同的风格和模式。上一节介绍了一种这样的方法。

+   理想情况下，如果一个模板声明满足所有指定的限制，模板体中不应出现替换错误（即如果函数被调用，则可以编译）。在实践中，这是一个难以实现的目标：限制条件可能变得冗长，有时难以编写，你可能甚至没有意识到你的实现隐式地要求的所有约束。即使是设计时受益于委员会审查其要求的每个词的 STL，也没有完全达到这个目标。尽管如此，这是一个好的实践目标。此外，如果你必须允许函数被调用但不能编译，至少在函数体中将要求编码为静态断言——它们比用户从未听说过的类型中的奇怪替换错误更容易理解。

阅读本章后，这些指南对你来说不应太过令人畏惧。

# 摘要

SFINAE 是 C++ 标准中的一种较为晦涩的特性——它复杂且有许多细微之处。虽然它通常在 *手动控制重载解析* 的上下文中被提及，但它的主要目的实际上并不是为了允许非常复杂的专家级代码，而是为了让常规（自动）的重载解析按照程序员期望的方式工作。在这个角色中，它通常能如预期般工作，且无需额外努力——实际上，程序员通常甚至不需要意识到这个特性。大多数时候，当你编写一个泛型重载和一个针对指针的特殊重载时，你期望后者在非指针类型上不会被调用。大多数时候，你可能甚至不会停下来注意到被拒绝的重载会是无效的——谁在乎呢，它本就不应该被使用。但为了找出它不应该被使用，必须替换类型，这会导致无效的代码。SFINAE 打破了这种“先有鸡还是先有蛋”的问题——为了找出重载应该被拒绝，我们必须替换类型，但那样会创建不应该编译的代码，这不应该是一个问题，因为重载最初就应该被拒绝，但我们直到替换了类型才知道这一点，如此循环。这就是我们所说的“自然”SFINAE。

当然，我们并没有翻阅几十页的书籍只是为了学习编译器神奇地做正确的事情，你不必担心。SFINAE 更为精细的使用是创建一个人工的替换失败，通过移除一些重载来控制重载解析。在本章中，我们学习了这些最终被 SFINAE 抑制的 *临时* 错误的 *安全* 环境。通过谨慎的应用，这项技术可以在编译时检查和区分从不同类型的基本特性（*这是一个类吗？*）到任何数量的 C++ 语言特性可以提供的复杂行为（*有没有办法添加这两种类型？*）。在 C++20 中，通过引入约束和概念，此类代码得到了极大的简化。然而，我们甚至可以将概念启发的思维应用于为早期标准编写的代码。

在下一章中，我们将介绍另一种用于极大地增强 C++ 中类层次结构能力的先进模板模式：类继承使我们能够从基类传递信息到派生类，而 Curiously Recurring Template Pattern 则相反，它使基类意识到派生类。

# 问题

1.  重载集是什么？

1.  重载解析是什么？

1.  类型推导和类型替换是什么？

1.  SFINAE 是什么？

1.  在什么情况下可能无效的代码可以存在而不触发编译错误，除非该代码实际上需要？

1.  我们如何确定选择了哪个重载，而不实际调用它？

1.  SFINAE 如何用于控制条件编译？

1.  为什么 C++20 的约束优于 SFINAE 用于模板约束？

1.  C++20 的概念标准如何使使用早期语言版本的程序员受益？

# 第三部分：C++ 设计模式

本部分从本书的主要内容开始。它介绍了最重要的、最常用的 C++ 设计模式。每个模式通常被用作解决某一类问题的公认方法。确切的问题是什么，差异很大：有些是系统架构挑战，有些是接口设计问题，还有些是处理程序性能。

本部分包含以下章节：

+   *第八章*，*Curiously Recurring Template Pattern*

+   *第九章*，*命名参数、方法链和 Builder 模式*

+   *第十章*，*局部缓冲区优化*

+   *第十一章*，*作用域保护*

+   *第十二章*，*友元工厂*

+   *第十三章*，*虚构造函数和工厂*

+   *第十四章*，*模板方法模式和 Non-Virtual 习语*
