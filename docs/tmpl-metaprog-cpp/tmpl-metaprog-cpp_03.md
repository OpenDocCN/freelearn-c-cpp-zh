# 第二章：*第二章*：模板基础

在上一章中，我们简要介绍了模板。它们是什么，它们如何有帮助，使用模板的优缺点，以及一些函数和类模板的例子。在本章中，我们将详细探讨这个领域，并查看模板参数、实例化、特化、别名等方面。本章的主要学习内容包括以下内容：

+   如何定义函数模板、类模板、变量模板和别名模板

+   存在哪些类型的模板参数？

+   什么是模板实例化？

+   什么是模板特化？

+   如何使用泛型 lambda 和 lambda 模板

到本章结束时，你将熟悉 C++ 中模板的核心基础，能够理解大量模板代码，并能够自己编写模板。

为了开始本章，我们将探讨定义和使用函数模板的细节。

# 定义函数模板

函数模板的定义方式与常规函数类似，只是函数声明前面是关键字 `template`，后面跟着一个用尖括号括起来的模板参数列表。以下是一个简单的函数模板示例：

```cpp
template <typename T>
```

```cpp
T add(T const a, T const b)
```

```cpp
{
```

```cpp
   return a + b;
```

```cpp
}
```

这个函数有两个参数，称为 `a` 和 `b`，它们都是相同的 `T` 类型。这个类型列在模板参数列表中，由关键字 `typename` 或 `class`（在本例和整本书中使用了前者）引入。这个函数所做的只是将两个参数相加并返回这个操作的结果，这个结果应该具有相同的 `T` 类型。

函数模板只是创建实际函数的蓝图，并且只存在于源代码中。除非在源代码中显式调用，否则函数模板将不会出现在编译后的可执行文件中。然而，当编译器遇到对函数模板的调用并且能够将提供的参数及其类型与函数模板的参数匹配时，它将根据模板和用于调用的参数生成一个实际函数。为了理解这一点，让我们看看一些例子：

```cpp
auto a = add(42, 21);
```

在这个片段中，我们使用两个 `int` 参数 `42` 和 `21` 调用 `add` 函数。编译器能够从提供的参数类型中推断出模板参数 `T`，因此不需要显式提供它。然而，以下两种调用也是可能的，并且实际上与前面的一种相同：

```cpp
auto a = add<int>(42, 21);
```

```cpp
auto a = add<>(42, 21);
```

从这个调用中，编译器将生成以下函数（请注意，实际代码可能因编译器的不同而有所不同）：

```cpp
int add(const int a, const int b)
```

```cpp
{
```

```cpp
  return a + b;
```

```cpp
}
```

然而，如果我们改变调用形式，我们明确为模板参数 `T` 提供了 `short` 类型的参数：

```cpp
auto b = add<short>(42, 21);
```

在这种情况下，编译器将生成这个函数的另一个实例，用 `short` 代替 `int`。这个新的实例将如下所示：

```cpp
short add(const short a, const int b)
```

```cpp
{
```

```cpp
  return static_cast<short>(a + b);
```

```cpp
}
```

如果两个参数的类型不明确，编译器将无法自动推断它们。以下调用就是这种情况：

```cpp
auto d = add(41.0, 21);
```

在这个例子中，`41.0`是一个`double`类型，而`21`是一个`int`类型。`add`函数模板有两个相同类型的参数，因此编译器无法将其与提供的参数匹配，并将引发错误。为了避免这种情况，假设你希望它为`double`类型实例化，你必须显式指定类型，如下面的代码片段所示：

```cpp
auto d = add<double>(41.0, 21);
```

只要两个参数具有相同的类型，并且`+`运算符对于参数类型是可用的，你就可以以前面显示的方式调用函数模板`add`。然而，如果`+`运算符不可用，那么即使模板参数被正确解析，编译器也无法生成实例化。这如下面的代码片段所示：

```cpp
class foo
```

```cpp
{
```

```cpp
   int value;
```

```cpp
public:
```

```cpp
   explicit foo(int const i):value(i)
```

```cpp
   { }
```

```cpp
   explicit operator int() const { return value; }
```

```cpp
};
```

```cpp
auto f = add(foo(42), foo(41));
```

在这种情况下，编译器将发出错误，指出找不到类型`foo`的二进制`+`运算符。当然，不同的编译器会有不同的实际消息，所有错误都是如此。为了能够为类型`foo`的参数调用`add`，你必须为此类型重载`+`运算符。一个可能的实现如下：

```cpp
foo operator+(foo const a, foo const b)
```

```cpp
{
```

```cpp
  return foo((int)a + (int)b);
```

```cpp
}
```

我们迄今为止看到的所有示例都表示了只有一个模板参数的模板。然而，一个模板可以有任意数量的参数，甚至可以有可变数量的参数。这个后者的主题将在*第三章*中讨论，*可变参数模板*。下一个函数是一个有两个类型模板参数的函数模板：

```cpp
template <typename Input, typename Predicate>
```

```cpp
int count_if(Input start, Input end, Predicate p)
```

```cpp
{
```

```cpp
   int total = 0;
```

```cpp
   for (Input i = start; i != end; i++)
```

```cpp
   {
```

```cpp
      if (p(*i))
```

```cpp
         total++;
```

```cpp
   }
```

```cpp
   return total;
```

```cpp
}
```

此函数接受两个输入迭代器，分别指向范围的开头和结尾，以及一个谓词，并返回范围内匹配谓词的元素数量。这个函数，至少在概念上，与标准库中`<algorithm>`头文件中的`std::count_if`通用函数非常相似，你应该始终优先使用标准算法而不是手工实现的算法。然而，为了本主题的目的，这个函数是一个很好的例子，可以帮助你理解模板是如何工作的。

我们可以使用`count_if`函数如下：

```cpp
int main()
```

```cpp
{
```

```cpp
   int arr[]{ 1,1,2,3,5,8,11 };
```

```cpp
   int odds = count_if(
```

```cpp
                 std::begin(arr), std::end(arr), 
```

```cpp
                 [](int const n) { return n % 2 == 1; });
```

```cpp
   std::cout << odds << '\n';
```

```cpp
}
```

再次强调，没有必要显式指定类型模板参数的参数（输入迭代器的类型和一元谓词的类型），因为编译器能够从调用中推断它们。

虽然还有更多关于函数模板的知识需要学习，但本节提供了关于如何使用它们的介绍。现在让我们学习定义类模板的基础知识。

# 定义类模板

类模板的声明方式与类声明非常相似，使用 `template` 关键字和模板参数列表在类声明之前。我们已经在引言章节中看到了第一个例子。下面的代码片段展示了名为 `wrapper` 的类模板。它有一个单一的模板参数，称为 `T`，用作数据成员、参数和函数返回类型的类型：

```cpp
template <typename T>
```

```cpp
class wrapper
```

```cpp
{
```

```cpp
public:
```

```cpp
   wrapper(T const v): value(v)
```

```cpp
   { }
```

```cpp
   T const& get() const { return value; }
```

```cpp
private:
```

```cpp
   T value;
```

```cpp
};
```

只要类模板在您的源代码中任何地方都没有使用，编译器就不会从它生成代码。为了实现这一点，类模板必须被实例化，并且所有参数都必须正确地与参数匹配，无论是用户显式地匹配，还是编译器隐式地匹配。下面展示了实例化此类模板的例子：

```cpp
wrapper a(42);           // wraps an int
```

```cpp
wrapper<int> b(42);      // wraps an int
```

```cpp
wrapper<short> c(42);    // wraps a short
```

```cpp
wrapper<double> d(42.0); // wraps a double
```

```cpp
wrapper e("42");         // wraps a char const *
```

由于一个名为 `wrapper<int>` 或 `wrapper<char const*>` 的特性，这个片段中 `a` 和 `e` 的定义仅在 C++17 及以后版本中有效。

类模板可以在不定义的情况下声明，并在允许不完整类型的环境中使用，例如函数的声明，如下所示：

```cpp
template <typename T>
```

```cpp
class wrapper;
```

```cpp
void use_foo(wrapper<int>* ptr);
```

然而，类模板必须在模板实例化的点定义；否则，编译器将生成错误。以下代码片段展示了这一点：

```cpp
template <typename T>
```

```cpp
class wrapper;                       // OK
```

```cpp
void use_wrapper(wrapper<int>* ptr); // OK
```

```cpp
int main()
```

```cpp
{
```

```cpp
   wrapper<int> a(42);            // error, incomplete type
```

```cpp
   use_wrapper(&a);
```

```cpp
}
```

```cpp
template <typename T>
```

```cpp
class wrapper
```

```cpp
{
```

```cpp
   // template definition
```

```cpp
};
```

```cpp
void use_wrapper(wrapper<int>* ptr)
```

```cpp
{
```

```cpp
   std::cout << ptr->get() << '\n';
```

```cpp
}
```

在声明 `use_wrapper` 函数时，类模板 `wrapper` 只被声明，而没有定义。然而，在这个上下文中允许不完整类型，这使得在这一点上使用 `wrapper<T>` 是可以的。然而，在 `main` 函数中，我们正在实例化 `wrapper` 类模板的对象。这将生成编译器错误，因为在这个点上类模板的定义必须是可用的。为了修复这个特定的例子，我们必须将 `main` 函数的定义移动到末尾，在 `wrapper` 和 `use_wrapper` 定义之后。

在这个例子中，类模板是使用 `class` 关键字定义的。然而，在 C++ 中，使用 `class` 或 `struct` 关键字声明类之间几乎没有区别：

+   使用 `struct` 时，默认成员访问权限是公共的，而使用 `class` 则是私有的。

+   使用 `struct` 时，基类继承的默认访问修饰符是公共的，而使用 `class` 则是私有的。

您可以使用 `struct` 关键字定义类模板，就像我们在这里使用 `class` 关键字一样。使用 `struct` 或 `class` 关键字定义的类之间的差异也适用于使用 `struct` 或 `class` 关键字定义的类模板。

不论类是否是模板，都可能包含成员函数模板。这些定义的方式将在下一节中讨论。

# 定义成员函数模板

到目前为止，我们已经学习了关于函数模板和类模板的知识。也可以定义成员函数模板，无论是在非模板类中还是在类模板中。在本节中，我们将学习如何做到这一点。为了理解差异，让我们从以下示例开始：

```cpp
template <typename T>
```

```cpp
class composition
```

```cpp
{
```

```cpp
public:
```

```cpp
   T add(T const a, T const b)
```

```cpp
   {
```

```cpp
      return a + b;
```

```cpp
   }
```

```cpp
};
```

`composition` 类是一个类模板。它有一个名为 `add` 的单一成员函数，该函数使用类型参数 `T`。这个类可以这样使用：

```cpp
composition<int> c;
```

```cpp
c.add(41, 21);
```

我们首先需要实例化 `composition` 类的对象。请注意，我们必须显式指定类型参数 `T` 的参数，因为编译器无法自行推断出来（没有上下文可以从中推断）。当我们调用 `add` 函数时，我们只需提供参数。它们的类型，由之前解析为 `int` 的 `T` 类型模板参数表示，已经已知。例如 `c.add<int>(42, 21)` 这样的调用将触发编译器错误。`add` 函数不是一个函数模板，而是一个类模板 `composition` 的成员函数。

在下一个示例中，`composition` 类略有变化，但意义重大。让我们首先看看定义：

```cpp
class composition
```

```cpp
{
```

```cpp
public:
```

```cpp
   template <typename T>
```

```cpp
   T add(T const a, T const b)
```

```cpp
   {
```

```cpp
      return a + b;
```

```cpp
   }
```

```cpp
};
```

这次，`composition` 是一个非模板类。然而，`add` 函数是一个函数模板。因此，要调用此函数，我们必须执行以下操作：

```cpp
composition c;
```

```cpp
c.add<int>(41, 21);
```

对于 `T` 类型模板参数的 `int` 类型显式指定是多余的，因为编译器可以从调用参数中自行推断出来。然而，这里展示了这样做有助于更好地理解这两种实现之间的差异。

除了这两种情况之外，我们还可以有类模板的成员函数模板。在这种情况下，成员函数模板的模板参数必须与类模板的模板参数不同；否则，编译器将生成错误。让我们回到 `wrapper` 类模板示例，并按如下方式修改它：

```cpp
template <typename T>
```

```cpp
class wrapper
```

```cpp
{
```

```cpp
public:
```

```cpp
   wrapper(T const v) :value(v)
```

```cpp
   {}
```

```cpp
   T const& get() const { return value; }
```

```cpp
   template <typename U>
```

```cpp
   U as() const
```

```cpp
   {
```

```cpp
      return static_cast<U>(value);
```

```cpp
   }
```

```cpp
private:
```

```cpp
   T value;
```

```cpp
};
```

如您所见，这个实现增加了一个成员，一个名为 `as` 的函数。这是一个函数模板，有一个名为 `U` 的类型模板参数。这个函数用于将包装值从类型 `T` 转换为类型 `U`，并将其返回给调用者。我们可以如下使用这个实现：

```cpp
wrapper<double> a(42.0);
```

```cpp
auto d = a.get();       // double
```

```cpp
auto n = a.as<int>();   // int
```

在实例化 `wrapper` 类（`double`）时指定了模板参数的参数 - 虽然在 C++17 中这是多余的，并且在调用 `as` 函数（`int`）以执行转换时。

在我们继续其他主题之前，例如实例化、特化和其他形式的模板，包括变量和别名之前，重要的是我们要花时间更多地了解模板参数。这将使下一节的主题更加清晰。

# 理解模板参数

到目前为止，本书中我们已经看到了多个具有一个或多个参数的模板示例。在所有这些示例中，参数代表在实例化时提供的类型，无论是用户明确提供的，还是编译器在可以推断它们时隐式提供的。这类参数被称为**类型模板参数**。然而，模板也可以有**非类型模板参数**和**模板模板参数**。在以下章节中，我们将探讨所有这些参数。 

## 类型模板参数

如前所述，这些是在模板实例化过程中作为参数提供的类型参数。它们通过`typename`或`class`关键字引入。使用这两个关键字没有区别。类型模板参数可以有一个默认值，这是一个类型。这可以通过与指定函数参数默认值相同的方式指定。以下是一些示例：

```cpp
template <typename T>
```

```cpp
class wrapper { /* ... */ };
```

```cpp
template <typename T = int>
```

```cpp
class wrapper { /* ... */ };
```

类型模板参数的名称可以省略，这在转发声明中可能很有用：

```cpp
template <typename>
```

```cpp
class wrapper;
```

```cpp
template <typename = int>
```

```cpp
class wrapper;
```

C++11 引入了可变模板，这些模板具有可变数量的参数。接受零个或多个参数的模板参数称为**参数包**。**类型模板参数包**具有以下形式：

```cpp
template <typename... T>
```

```cpp
class wrapper { /* ... */ };
```

可变模板将在*第三章*，“可变模板”中讨论。因此，我们在此不会深入讨论这类参数的细节。

C++20 引入了`typename`或`class`关键字。以下是一些示例，包括具有默认值的概念和受约束的类型模板参数包：

```cpp
template <WrappableType T>
```

```cpp
class wrapper { /* ... */ };
```

```cpp
template <WrappableType T = int>
```

```cpp
class wrapper { /* ... */ };
```

```cpp
template <WrappableType... T>
```

```cpp
class wrapper { /* ... */ };
```

概念和约束在第*第六章*“概念和约束”中讨论。我们将在那一章中了解更多关于这类参数的信息。现在，让我们看看第二种模板参数，非类型模板参数。

## 非类型模板参数

模板参数不总是必须代表类型。它们也可以是编译时表达式，例如常量、具有外部链接的函数或对象的地址，或静态类成员的地址。使用编译时表达式提供的参数称为**非类型模板参数**。这类参数只能具有**结构化类型**。以下是一些结构化类型：

+   整数类型

+   浮点类型，自 C++20 起

+   枚举

+   指针类型（指向对象或函数）

+   成员类型指针（指向成员对象或成员函数）

+   左值引用类型（指向对象或函数）

+   符合以下要求的字面类类型：

    +   所有基类都是公开且不可变的。

    +   所有非静态数据成员都是公开且不可变的。

    +   所有基类和非静态数据成员的类型也是结构化类型或其数组。

这些类型的 cv-限定形式也可以用于非类型模板参数。非类型模板参数可以以不同的方式指定。可能的形式在以下片段中显示：

```cpp
template <int V>
```

```cpp
class foo { /*...*/ };
```

```cpp
template <int V = 42>
```

```cpp
class foo { /*...*/ };
```

```cpp
template <int... V>
```

```cpp
class foo { /*...*/ };
```

在所有这些例子中，非类型模板参数的类型是 `int`。第一个和第二个例子是相似的，除了第二个例子使用了默认值。第三个例子显著不同，因为参数实际上是一个参数包。这将在下一章中讨论。

为了更好地理解非类型模板参数，让我们看看以下示例，其中我们草拟了一个固定大小的数组类，称为 `buffer`：

```cpp
template <typename T, size_t S>
```

```cpp
class buffer
```

```cpp
{
```

```cpp
   T data_[S];
```

```cpp
public:
```

```cpp
   constexpr T const * data() const { return data_; }
```

```cpp
   constexpr T& operator[](size_t const index)
```

```cpp
   {
```

```cpp
      return data_[index];
```

```cpp
   }
```

```cpp
   constexpr T const & operator[](size_t const index) const
```

```cpp
   {
```

```cpp
      return data_[index];
```

```cpp
   }
```

```cpp
};
```

这个 `buffer` 类包含一个内部数组，该数组有 `S` 个元素，类型为 `T`。因此，`S` 需要是一个编译时值。这个类可以如下实例化：

```cpp
buffer<int, 10> b1;
```

```cpp
buffer<int, 2*5> b2;
```

这两个定义是等价的，`b1` 和 `b2` 都是两个包含 10 个整数的缓冲区。此外，它们是同一类型，因为 2*5 和 10 是两个在编译时评估为相同值的表达式。你可以通过以下语句轻松检查这一点：

```cpp
static_assert(std::is_same_v<decltype(b1), decltype(b2)>);
```

这种情况不再适用，因为 `b3` 对象的类型声明如下：

```cpp
buffer<int, 3*5> b3;
```

在这个例子中，`b3` 是一个包含 15 个整数的 `buffer`，这与上一个例子中包含 10 个整数的 `buffer` 类型不同。从概念上讲，编译器会生成以下代码：

```cpp
template <typename T, size_t S>
```

```cpp
class buffer
```

```cpp
{
```

```cpp
   T data_[S];
```

```cpp
public:
```

```cpp
   constexpr T* data() const { return data_; }
```

```cpp
   constexpr T& operator[](size_t const index)
```

```cpp
   {
```

```cpp
      return data_[index];
```

```cpp
   }
```

```cpp
   constexpr T const & operator[](size_t const index) const
```

```cpp
   {
```

```cpp
      return data_[index];
```

```cpp
   }
```

```cpp
};
```

这是主模板的代码，但接下来还将展示几个特化示例：

```cpp
template<>
```

```cpp
class buffer<int, 10>
```

```cpp
{
```

```cpp
  int data_[10];
```

```cpp
public: 
```

```cpp
  constexpr int * data() const;
```

```cpp
  constexpr int & operator[](const size_t index); 
```

```cpp
  constexpr const int & operator[](
```

```cpp
    const size_t index) const;
```

```cpp
};
```

```cpp
template<>
```

```cpp
class buffer<int, 15>
```

```cpp
{
```

```cpp
  int data_[15]; 
```

```cpp
public: 
```

```cpp
  constexpr int * data() const;
```

```cpp
  constexpr int & operator[](const size_t index);
```

```cpp
  constexpr const int & operator[](
```

```cpp
    const size_t index) const;
```

```cpp
};
```

在这个代码示例中看到的特化概念将在本章的 *理解模板特化* 部分中进一步详细说明。目前，你应该注意到两种不同的 `buffer` 类型。再次强调，可以通过以下语句验证 `b1` 和 `b3` 的类型不同：

```cpp
static_assert(!std::is_same_v<decltype(b1), decltype(b3)>);
```

在实践中，使用结构化类型（如整数、浮点数或枚举类型）的情况比其他情况更为常见。理解它们的使用和找到有用的示例可能更容易。然而，也存在使用指针或引用的场景。在以下示例中，我们将检查函数参数指针的使用。让我们先看看代码：

```cpp
struct device
```

```cpp
{
```

```cpp
   virtual void output() = 0;
```

```cpp
   virtual ~device() {}
```

```cpp
};
```

```cpp
template <void (*action)()>
```

```cpp
struct smart_device : device
```

```cpp
{
```

```cpp
   void output() override
```

```cpp
   {
```

```cpp
      (*action)();
```

```cpp
   }
```

```cpp
};
```

在这个片段中，`device` 是一个具有纯虚函数 `output`（以及虚析构函数）的基类。这是 `smart_device` 类模板的基类，该类模板通过调用函数指针来实现 `output` 虚函数。这个函数指针传递了一个参数给类模板的非类型模板参数。以下示例展示了它的用法：

```cpp
void say_hello_in_english()
```

```cpp
{
```

```cpp
   std::cout << "Hello, world!\n";
```

```cpp
}
```

```cpp
void say_hello_in_spanish()
```

```cpp
{
```

```cpp
   std::cout << "Hola mundo!\n";
```

```cpp
}
```

```cpp
auto w1 =
```

```cpp
   std::make_unique<smart_device<&say_hello_in_english>>();
```

```cpp
w1->output();
```

```cpp
auto w2 =
```

```cpp
   std::make_unique<smart_device<&say_hello_in_spanish>>();
```

```cpp
w2->output();
```

在这里，`w1`和`w2`是两个`unique_ptr`对象。尽管表面上它们指向相同类型的对象，但这并不正确，因为`smart_device<&say_hello_in_english>`和`smart_device<&say_hello_in_spanish>`是不同的类型，因为它们使用不同的函数指针值实例化。这可以通过以下语句轻松检查：

```cpp
static_assert(!std::is_same_v<decltype(w1), decltype(w2)>);
```

相反，如果我们将`auto`指定符改为`std::unique_ptr<device>`，如下面的代码片段所示，那么`w1`和`w2`就是基类 device 的智能指针，因此它们具有相同的类型：

```cpp
std::unique_ptr<device> w1 = 
```

```cpp
   std::make_unique<smart_device<&say_hello_in_english>>();
```

```cpp
w1->output();
```

```cpp
std::unique_ptr<device> w2 = 
```

```cpp
   std::make_unique<smart_device<&say_hello_in_spanish>>();
```

```cpp
w2->output();
```

```cpp
static_assert(std::is_same_v<decltype(w1), decltype(w2)>);
```

虽然这个例子使用了函数指针，但也可以构思一个类似的使用成员函数指针的例子。前一个例子可以转换为以下形式（仍然使用相同的基类 device）：

```cpp
template <typename Command, void (Command::*action)()>
```

```cpp
struct smart_device : device
```

```cpp
{
```

```cpp
   smart_device(Command& command) : cmd(command) {}
```

```cpp
   void output() override
```

```cpp
   {
```

```cpp
      (cmd.*action)();
```

```cpp
   }
```

```cpp
private:
```

```cpp
   Command& cmd;
```

```cpp
};
```

```cpp
struct hello_command
```

```cpp
{
```

```cpp
   void say_hello_in_english()
```

```cpp
   {
```

```cpp
      std::cout << "Hello, world!\n";
```

```cpp
   }
```

```cpp
   void say_hello_in_spanish()
```

```cpp
   {
```

```cpp
      std::cout << "Hola mundo!\n";
```

```cpp
   }
```

```cpp
};
```

这些类可以如下使用：

```cpp
hello_command cmd;
```

```cpp
auto w1 = std::make_unique<
```

```cpp
   smart_device<hello_command, 
```

```cpp
      &hello_command::say_hello_in_english>>(cmd);
```

```cpp
w1->output();
```

```cpp
auto w2 = std::make_unique<
```

```cpp
   smart_device<hello_command, 
```

```cpp
      &hello_command::say_hello_in_spanish>>(cmd);
```

```cpp
w2->output();
```

在 C++17 中，引入了一种新的指定非类型模板参数的形式，使用`auto`指定符（包括`auto*`和`auto&`形式）或`decltype(auto)`代替类型的名称。这允许编译器从提供的表达式推断参数的类型。如果推断的类型不允许作为非类型模板参数，编译器将生成错误。让我们看一个例子：

```cpp
template <auto x>
```

```cpp
struct foo
```

```cpp
{ /* … */ };
```

这个类模板可以如下使用：

```cpp
foo<42>   f1;  // foo<int>
```

```cpp
foo<42.0> f2;  // foo<double> in C++20, error for older 
```

```cpp
               // versions
```

```cpp
foo<"42"> f3;  // error
```

在第一个例子中，对于`f1`，编译器推断参数的类型为`int`。在第二个例子中，对于`f2`，编译器推断的类型为`double`。然而，这只适用于 C++20。在标准的前版本中，这一行会产生错误，因为 C++20 之前不允许将浮点类型作为非类型模板参数的参数。然而，最后一行产生错误，因为`"42"`是一个字符串字面量，而字符串字面量不能用作非类型模板参数的参数。

然而，在 C++20 中，可以通过将字面量字符串包裹在结构字面量类中来绕过最后一个例子。这个类将字符串字面量的字符存储在固定长度的数组中。以下代码片段展示了这一点：

```cpp
template<size_t N>
```

```cpp
struct string_literal
```

```cpp
{
```

```cpp
   constexpr string_literal(const char(&str)[N])
```

```cpp
   {
```

```cpp
      std::copy_n(str, N, value);
```

```cpp
   }
```

```cpp
   char value[N];
```

```cpp
};
```

然而，前面展示的`foo`类模板需要修改，以显式使用`string_literal`而不是`auto`指定符：

```cpp
template <string_literal x>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
};
```

在此基础上，前面展示的`foo<"42"> f;`声明在 C++20 中将无错误编译。

`auto`指定符也可以与非类型模板参数包一起使用。在这种情况下，每个模板参数的类型都是独立推断的。模板参数的类型不需要相同。以下代码片段展示了这一点：

```cpp
template<auto... x>
```

```cpp
struct foo
```

```cpp
{ /* ... */ };
```

```cpp
foo<42, 42.0, false, 'x'> f;
```

在这个例子中，编译器推断模板参数的类型分别为`int`、`double`、`bool`和`char`。

第三种也是最后一种模板参数类别是**模板模板参数**。我们将在下一节中探讨它们。

## 模板模板参数

尽管这个名字听起来可能有点奇怪，但它指的是一类模板参数，这些参数本身也是模板。这些参数可以像类型模板参数一样指定，带有或没有名称，带有或没有默认值，以及带有或没有名称的参数包。截至 C++17，可以使用关键字 `class` 和 `typename` 来引入模板模板参数。在此版本之前，只能使用 `class` 关键字。

为了展示模板模板参数的使用，让我们首先考虑以下两个类模板：

```cpp
template <typename T>
```

```cpp
class simple_wrapper
```

```cpp
{
```

```cpp
public:
```

```cpp
   T value;
```

```cpp
};
```

```cpp
template <typename T>
```

```cpp
class fancy_wrapper
```

```cpp
{
```

```cpp
public:
```

```cpp
   fancy_wrapper(T const v) :value(v)
```

```cpp
   {
```

```cpp
   }
```

```cpp
   T const& get() const { return value; }
```

```cpp
   template <typename U>
```

```cpp
   U as() const
```

```cpp
   {
```

```cpp
      return static_cast<U>(value);
```

```cpp
   }
```

```cpp
private:
```

```cpp
   T value;
```

```cpp
};
```

`simple_wrapper` 类是一个非常简单的类模板，它持有类型模板参数 `T` 的值。另一方面，`fancy_wrapper` 是一个更复杂的包装实现，它隐藏了包装的值并公开了数据访问的成员函数。接下来，我们实现一个名为 `wrapping_pair` 的类模板，它包含两个包装类型的值。这可以是 `simpler_wrapper`、`fancy_wrapper` 或其他类似的东西：

```cpp
template <typename T, typename U, 
```

```cpp
          template<typename> typename W = fancy_wrapper>
```

```cpp
class wrapping_pair
```

```cpp
{
```

```cpp
public:
```

```cpp
   wrapping_pair(T const a, U const b) :
```

```cpp
      item1(a), item2(b)
```

```cpp
   {
```

```cpp
   }
```

```cpp
   W<T> item1;
```

```cpp
   W<U> item2;
```

```cpp
};   
```

`wrapping_pair` 类模板有三个参数。前两个是类型模板参数，分别命名为 `T` 和 `U`。第三个参数是模板模板参数，称为 `W`，它有一个默认值，即 `fancy_wrapper` 类型。我们可以像以下代码片段所示使用这个类模板：

```cpp
wrapping_pair<int, double> p1(42, 42.0);
```

```cpp
std::cout << p1.item1.get() << ' '
```

```cpp
          << p1.item2.get() << '\n';
```

```cpp
wrapping_pair<int, double, simple_wrapper> p2(42, 42.0);
```

```cpp
std::cout << p2.item1.value << ' '
```

```cpp
          << p2.item2.value << '\n';
```

在这个例子中，`p1` 是一个包含两个值（一个 `int` 和一个 `double`，每个都包装在一个 `fancy_wrapper` 对象中）的 `wrapping_pair` 对象。这不是显式指定的，而是模板模板参数的默认值。另一方面，`p2` 也是一个 `wrapping_pair` 对象，也包含一个 `int` 和一个 `double`，但这些被一个 `simple_wrapper` 对象包装，现在在模板实例化中显式指定。

在这个例子中，我们看到了模板参数的默认模板参数的使用。这个主题将在下一节中详细探讨。

## 默认模板参数

默认模板参数的指定方式与默认函数参数类似，在等号后面的参数列表中。以下规则适用于默认模板参数：

+   它们可以与任何类型的模板参数一起使用，除了参数包。

+   如果在类模板、变量模板或类型别名中为模板参数指定了一个默认值，那么所有后续的模板参数也必须有一个默认值。例外是最后一个参数，如果它是模板参数包。

+   如果在函数模板中为模板参数指定了一个默认值，那么后续的模板参数不受限制，也必须有一个默认值。

+   在函数模板中，参数包之后可以跟有更多类型参数，前提是它们有默认参数或编译器可以从函数参数中推断出它们的值。

+   它们不允许在友元类模板的声明中使用。

+   它们只允许在友元函数模板的声明中使用，如果该声明也是一个定义，并且在同一翻译单元中没有其他函数声明。

+   它们不允许在函数模板或成员函数模板的显式特化的声明或定义中使用。

以下代码片段展示了使用默认模板参数的示例：

```cpp
template <typename T = int>
```

```cpp
class foo { /*...*/ };
```

```cpp
template <typename T = int, typename U = double>
```

```cpp
class bar { /*...*/ };
```

如前所述，在声明类模板时，带有默认参数的模板参数不能后面跟着没有默认参数的参数，但这种限制不适用于函数模板。这将在下一个代码片段中展示：

```cpp
template <typename T = int, typename U>
```

```cpp
class bar { };   // error
```

```cpp
template <typename T = int, typename U>
```

```cpp
void func() {}   // OK
```

一个模板可以有多个声明（但只有一个定义）。所有声明和定义中的默认模板参数被合并（与默认函数参数合并的方式相同）。让我们通过一个例子来了解它是如何工作的：

```cpp
template <typename T, typename U = double>
```

```cpp
struct foo;
```

```cpp
template <typename T = int, typename U>
```

```cpp
struct foo;
```

```cpp
template <typename T, typename U>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   T a;
```

```cpp
   U b;
```

```cpp
};
```

这在语义上等同于以下定义：

```cpp
template <typename T = int, typename U = double>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   T a;
```

```cpp
   U b;
```

```cpp
};
```

然而，这些具有不同默认模板参数的多个声明不能以任何顺序提供。前面提到的规则仍然适用。因此，第一个参数具有默认参数而后续参数没有的类模板声明是不合法的：

```cpp
template <typename T = int, typename U>
```

```cpp
struct foo;  // error, U does not have a default argument
```

```cpp
template <typename T, typename U = double>
```

```cpp
struct foo;
```

默认模板参数的另一个限制是，在同一个作用域内不能为同一个模板参数指定多个默认值。因此，下一个示例将产生错误：

```cpp
template <typename T = int>
```

```cpp
struct foo;
```

```cpp
template <typename T = int> // error redefinition
```

```cpp
                            // of default parameter
```

```cpp
struct foo {};
```

当默认模板参数使用类中的名称时，成员访问限制是在声明时检查的，而不是在模板实例化时：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
protected:
```

```cpp
   using value_type = T;
```

```cpp
};
```

```cpp
template <typename T, typename U = typename T::value_type>
```

```cpp
struct bar
```

```cpp
{
```

```cpp
   using value_type = U;
```

```cpp
};
```

```cpp
bar<foo<int>> x;
```

当定义`x`变量时，bar 类模板被实例化，但`foo::value_type`类型别名是受保护的，因此不能在`foo`之外使用。结果是`bar`类模板声明时出现编译错误。

通过这些说明，我们结束了模板参数这一主题。在下一节中，我们将探讨模板实例化，这是从模板定义和一组模板参数创建函数、类或变量新定义的过程。

# 理解模板实例化

如前所述，模板只是编译器在遇到它们的使用时创建实际代码的蓝图。从模板声明创建函数、类或变量定义的行为称为**模板实例化**。这可以是**显式的**，即当你告诉编译器何时生成定义时，或者**隐式的**，即编译器根据需要生成新的定义。我们将在下一节中详细探讨这两种形式。

## 隐式实例化

当编译器根据模板的使用生成定义，并且没有显式实例化时，会发生隐式实例化。隐式实例化的模板定义在模板所在的同一命名空间中。然而，编译器从模板创建定义的方式可能不同。这将在以下示例中看到。让我们考虑以下代码：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
  void f() {}
```

```cpp
};
```

```cpp
int main()
```

```cpp
{
```

```cpp
  foo<int> x;
```

```cpp
}
```

这里，我们有一个名为 `foo` 的类模板，它有一个成员函数 `f`。在 `main` 中，我们定义了一个 `foo<int>` 类型的变量，但没有使用它的任何成员。因为编译器遇到了对 `foo` 的这种使用，它会隐式地为 `int` 类型定义 `foo` 的一个特化。如果你使用在 Clang 上运行的 [cppinsights.io](http://cppinsights.io)，你会看到以下代码：

```cpp
template<>
```

```cpp
struct foo<int>
```

```cpp
{
```

```cpp
  inline void f();
```

```cpp
};
```

因为我们的代码中没有调用函数 `f`，它只被声明而没有定义。如果我们向 `main` 中添加一个 `f` 调用，特化将如下改变：

```cpp
template<>
```

```cpp
struct foo<int>
```

```cpp
{
```

```cpp
  inline void f() { }
```

```cpp
};
```

然而，如果我们添加一个名为 `g` 的额外函数，其实现包含一个错误，我们将根据不同的编译器获得不同的行为：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
  void f() {}
```

```cpp
  void g() {int a = "42";}
```

```cpp
};
```

```cpp
int main()
```

```cpp
{
```

```cpp
  foo<int> x;
```

```cpp
  x.f();
```

```cpp
}
```

`g` 函数体中存在一个错误（你也可以使用 `static_assert(false)` 语句作为替代）。这段代码在 VC++ 中编译没有任何问题，但在 Clang 和 GCC 中会失败。这是因为 VC++ 忽略了模板中未使用的部分，只要代码在语法上是正确的，但其他编译器在继续模板实例化之前会进行语义验证。

对于函数模板，当用户代码在需要其定义存在的上下文中引用一个函数时，会发生隐式实例化。对于类模板，当用户代码在需要完整类型或类型的完整性影响代码的上下文中引用模板时，会发生隐式实例化。此类上下文的典型例子是在构造此类类型的对象时。然而，在声明类模板的指针时并非如此。为了理解它是如何工作的，让我们考虑以下示例：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
  void f() {}
```

```cpp
  void g() {}
```

```cpp
};
```

```cpp
int main()
```

```cpp
{
```

```cpp
  foo<int>* p;
```

```cpp
  foo<int> x;
```

```cpp
  foo<double>* q;
```

```cpp
}
```

在这个片段中，我们使用了之前示例中的相同 `foo` 类模板，并声明了几个变量：`p` 是指向 `foo<int>` 的指针，`x` 是一个 `foo<int>`，`q` 是指向 `foo<double>` 的指针。由于 `x` 的声明，编译器此时只需要实例化 `foo<int>`。现在，让我们考虑一些成员函数 `f` 和 `g` 的调用，如下所示：

```cpp
int main()
```

```cpp
{
```

```cpp
  foo<int>* p;
```

```cpp
  foo<int> x;
```

```cpp
  foo<double>* q;
```

```cpp
  x.f();
```

```cpp
  q->g();
```

```cpp
}
```

随着这些更改，编译器需要实例化以下内容：

+   当声明 `x` 变量时 `foo<int>`

+   当发生 `x.f()` 调用时 `foo<int>::f()`

+   当发生 `q->g()` 调用时 `foo<double>` 和 `foo<double>::g()`

另一方面，当声明 `p` 指针时，编译器不需要实例化 `foo<int>`，当声明 `q` 指针时，也不需要实例化 `foo<double>`。然而，当类模板特化涉及指针转换时，编译器确实需要隐式实例化。以下示例展示了这一点：

```cpp
template <typename T>
```

```cpp
struct control
```

```cpp
{};
```

```cpp
template <typename T>
```

```cpp
struct button : public control<T>
```

```cpp
{};
```

```cpp
void show(button<int>* ptr)
```

```cpp
{
```

```cpp
   control<int>* c = ptr;
```

```cpp
}
```

在 `show` 函数中，发生 `button<int>*` 和 `control<int>*` 之间的转换。因此，在这个点上，编译器必须实例化 `button<int>`。

当一个类模板包含静态成员时，这些成员在编译器隐式实例化类模板时不会隐式实例化，但只有在编译器需要它们的定义时才会实例化。另一方面，每个类模板的特化都有自己的静态成员副本，如下面的代码片段所示：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   static T data;
```

```cpp
};
```

```cpp
template <typename T> T foo<T>::data = 0;
```

```cpp
int main()
```

```cpp
{
```

```cpp
   foo<int> a;
```

```cpp
   foo<double> b;
```

```cpp
   foo<double> c;
```

```cpp
   std::cout << a.data << '\n'; // 0
```

```cpp
   std::cout << b.data << '\n'; // 0
```

```cpp
   std::cout << c.data << '\n'; // 0
```

```cpp
   b.data = 42;
```

```cpp
   std::cout << a.data << '\n'; // 0
```

```cpp
   std::cout << b.data << '\n'; // 42
```

```cpp
   std::cout << c.data << '\n'; // 42
```

```cpp
}
```

类模板 `foo` 有一个名为 `data` 的静态成员变量，它在 `foo` 的定义之后初始化。在 `main` 函数中，我们将变量 `a` 声明为 `foo<int>` 的对象，而 `b` 和 `c` 则是 `foo<double>` 的对象。最初，它们的所有成员字段 `data` 都初始化为 0。然而，变量 `b` 和 `c` 共享同一份数据副本。因此，在执行 `b.data = 42` 赋值操作后，`a.data` 仍然是 0，但 `b.data` 和 `c.data` 都是 `42`。

在了解了隐式实例化的工作原理之后，现在是时候继续前进，了解模板实例化的另一种形式，即显式实例化。

## 显式实例化

作为用户，你可以明确告诉编译器实例化一个类模板或函数模板。这被称为显式实例化，并且有两种形式：**显式实例化定义**和**显式实例化声明**。我们将按此顺序讨论它们。

### 显式实例化定义

显式实例化定义可能出现在程序中的任何位置，但必须在引用的模板定义之后。显式模板实例化定义的语法有以下形式：

+   类模板的语法如下：

    ```cpp
    template class-key template-name <argument-list>
    ```

+   函数模板的语法如下：

    ```cpp
    template return-type name<argument-list>(parameter-list);
    template return-type name(parameter-list);
    ```

如您所见，在所有情况下，显式实例化定义都是以 `template` 关键字开始的，但不跟任何参数列表。对于类模板，`class-key` 可以是 `class`、`struct` 或 `union` 关键字之一。对于类和函数模板，具有给定参数列表的显式实例化定义在整个程序中只能出现一次。

我们将通过一些示例来了解这是如何工作的。以下是第一个示例：

```cpp
namespace ns
```

```cpp
{
```

```cpp
   template <typename T>
```

```cpp
   struct wrapper
```

```cpp
   {
```

```cpp
      T value;
```

```cpp
   };
```

```cpp
   template struct wrapper<int>;       // [1]
```

```cpp
}
```

```cpp
template struct ns::wrapper<double>;   // [2]
```

```cpp
int main() {}
```

在这个片段中，`wrapper<T>`是在`ns`命名空间中定义的类模板。代码中标记为`[1]`和`[2]`的语句都代表显式实例化定义，分别对应`wrapper<int>`和`wrapper<double>`。显式实例化定义只能出现在它所引用的模板所在的同一命名空间中（如`[1]`所示），或者它必须是完全限定的（如`[2]`所示）。我们可以为函数模板编写类似的显式模板定义：

```cpp
namespace ns
```

```cpp
{
```

```cpp
   template <typename T>
```

```cpp
   T add(T const a, T const b)
```

```cpp
   {
```

```cpp
      return a + b;
```

```cpp
   }
```

```cpp
   template int add(int, int);           // [1]
```

```cpp
}
```

```cpp
template double ns::add(double, double); // [2]
```

```cpp
int main() { }
```

第二个示例与第一个示例有惊人的相似之处。`[1]`和`[2]`都代表`add<int>()`和`add<double>()`的显式模板定义。

如果显式实例化定义不在与模板相同的命名空间中，则名称必须完全限定。使用`using`语句不会使名称在当前命名空间中可见。以下示例展示了这一点：

```cpp
namespace ns
```

```cpp
{
```

```cpp
   template <typename T>
```

```cpp
   struct wrapper { T value; };
```

```cpp
}
```

```cpp
using namespace ns;
```

```cpp
template struct wrapper<double>;   // error
```

在这个示例的最后一行中，会生成一个编译错误，因为`wrapper`是一个未知名称，必须用命名空间名称限定，如`ns::wrapper`。

当类成员用作返回类型或参数类型时，在显式实例化定义中会忽略成员访问指定。以下代码片段展示了示例：

```cpp
template <typename T>
```

```cpp
class foo
```

```cpp
{
```

```cpp
   struct bar {};
```

```cpp
   T f(bar const arg)
```

```cpp
   {
```

```cpp
      return {};
```

```cpp
   }
```

```cpp
};
```

```cpp
template int foo<int>::f(foo<int>::bar);
```

`X<T>::bar`类和`foo<T>::f()`函数都是`foo<T>`类的私有成员，但它们可以在最后一行显示的显式实例化定义中使用。

在了解了显式实例化定义及其工作原理之后，出现的问题是何时它是有用的。你为什么要告诉编译器从模板中生成实例化？答案是这有助于分发库、减少构建时间和可执行文件大小。如果你正在构建一个你想以`.lib`文件形式分发的库，并且该库使用模板，那么没有实例化的模板定义不会被放入库中。但这会导致每次使用库时用户代码的构建时间增加。通过强制在库中实例化模板，那些定义被放入对象文件和你要分发的`.lib`文件中。因此，你的用户代码只需要链接到库文件中可用的那些函数。这就是 Microsoft MSVC CRT 库为所有流、区域设置和字符串类所做的事情。`libstdc++`库对字符串类和其他类也做了同样的事情。

模板实例化可能引发的问题是你可能会得到多个定义，每个翻译单元一个。如果包含模板的相同头文件被包含在多个翻译单元（`.cpp`文件）中，并且使用了相同的模板实例化（例如，从我们之前的示例中的`wrapper<int>`），那么这些实例化的相同副本将被放入每个翻译单元中。这会导致对象大小增加。可以通过显式实例化声明来解决此问题，我们将在下一节中探讨。

### 显式实例化声明

显式实例化声明（自 C++11 起可用）是你可以告诉编译器模板实例化的定义位于不同的翻译单元中，并且不应生成新定义的方法。其语法与显式实例化定义相同，只是在声明前使用了 `extern` 关键字：

+   类模板的语法如下：

    ```cpp
    extern template class-key template-name <argument-list>
    ```

+   函数模板的语法如下：

    ```cpp
    extern template return-type name<argument-list>(parameter-list);
    extern template return-type name(parameter-list);
    ```

如果你提供了一个显式的实例化声明，但在程序的任何翻译单元中都没有实例化定义，那么结果将是编译器警告和链接错误。技术是在一个源文件中声明显式的模板实例化，在剩余的源文件中声明显式的模板声明。这将减少编译时间和目标文件大小。

让我们看看以下示例：

```cpp
// wrapper.h
```

```cpp
template <typename T>
```

```cpp
struct wrapper
```

```cpp
{
```

```cpp
   T data;
```

```cpp
}; 
```

```cpp
extern template wrapper<int>;   // [1]
```

```cpp
// source1.cpp
```

```cpp
#include "wrapper.h"
```

```cpp
#include <iostream>
```

```cpp
template wrapper<int>;          // [2]
```

```cpp
void f()
```

```cpp
{
```

```cpp
   ext::wrapper<int> a{ 42 };
```

```cpp
   std::cout << a.data << '\n';
```

```cpp
}
```

```cpp
// source2.cpp
```

```cpp
#include "wrapper.h"
```

```cpp
#include <iostream>
```

```cpp
void g()
```

```cpp
{
```

```cpp
   wrapper<int> a{ 100 };
```

```cpp
   std::cout << a.data << '\n';
```

```cpp
}
```

```cpp
// main.cpp
```

```cpp
#include "wrapper.h"
```

```cpp
int main()
```

```cpp
{
```

```cpp
   wrapper<int> a{ 0 };
```

```cpp
}
```

在这个示例中，我们可以看到以下内容：

+   `wrapper.h` 头文件包含一个名为 `wrapper<T>` 的类模板。在标记为 `[1]` 的行中有一个对 `wrapper<int>` 的显式实例化声明，告诉编译器在编译包含此头文件的源文件（翻译单元）时不要为这个实例化生成定义。

+   `source1.cpp` 文件包含了 `wrapper.h`，在标记为 `[2]` 的行中包含了对 `wrapper<int>` 的显式实例化定义。这是整个程序中这个实例化的唯一定义。

+   源文件 `source2.cpp` 和 `main.cpp` 都使用了 `wrapper<int>`，但没有任何显式实例化定义或声明。这是因为当头文件包含在每个这些文件中时，`wrapper.h` 中的显式声明是可见的。

或者，可以将显式实例化声明从头文件中移除，但然后它必须添加到包含该头文件的每个源文件中，这很可能会被遗忘。

当你进行显式模板声明时，请记住，定义在类体内部的类成员函数始终被认为是内联的，因此它总是会实例化。因此，你只能使用 `extern` 关键字来定义类体之外的成员函数。

现在我们已经了解了模板实例化是什么，我们将继续讨论另一个重要主题，**模板特化**，这是从模板实例化中创建的定义，用于处理特定的模板参数集。

# 理解模板特化

**模板特化**是从模板实例化创建的定义。正在特化的模板称为**主模板**。你可以为给定的一组模板参数提供一个显式特化的定义，从而覆盖编译器会生成的隐式代码。这是支持诸如类型特性和条件编译等特性的技术，这些是我们在 *第五章*，*类型特性和条件编译* 中将要探讨的元编程概念。

模板特化有两种形式：**显式（完整）特化**和**部分特化**。我们将在以下章节中详细探讨这两个方面。

## 显式特化

显式特化（也称为完整特化）发生在你为具有完整模板参数集的模板实例提供定义时。以下内容可以完全特化：

+   函数模板

+   类模板

+   变量模板（自 C++14 起可用）

+   类模板的成员函数、类和枚举

+   类或类模板的成员函数模板和类模板

+   类模板的静态数据成员

让我们从以下示例开始：

```cpp
template <typename T>
```

```cpp
struct is_floating_point
```

```cpp
{
```

```cpp
   constexpr static bool value = false;
```

```cpp
};
```

```cpp
template <>
```

```cpp
struct is_floating_point<float>
```

```cpp
{
```

```cpp
   constexpr static bool value = true;
```

```cpp
};
```

```cpp
template <>
```

```cpp
struct is_floating_point<double>
```

```cpp
{
```

```cpp
   constexpr static bool value = true;
```

```cpp
};
```

```cpp
template <>
```

```cpp
struct is_floating_point<long double>
```

```cpp
{
```

```cpp
   constexpr static bool value = true;
```

```cpp
};
```

在此代码片段中，`is_floating_point` 是主模板。它包含一个名为 `value` 的 `constexpr` 静态布尔数据成员，其初始值为 `false`。然后，我们有三个针对 `float`、`double` 和 `long double` 类型的完整特化。这些新定义改变了 `value` 使用 `true` 而不是 `false` 初始化的方式。因此，我们可以使用此模板编写如下代码：

```cpp
std::cout << is_floating_point<int>::value         << '\n';
```

```cpp
std::cout << is_floating_point<float>::value       << '\n';
```

```cpp
std::cout << is_floating_point<double>::value      << '\n';
```

```cpp
std::cout << is_floating_point<long double>::value << '\n';
```

```cpp
std::cout << is_floating_point<std::string>::value << '\n';
```

第一行和最后一行打印 `0`（对于 `false`）；其他行打印 `1`（对于 `true`）。这个示例演示了 `type` 特性是如何工作的。实际上，标准库在 `<type_traits>` 头文件中定义了一个名为 `is_floating_point` 的类模板，位于 `std` 命名空间中。我们将在 *第五章*，*类型特性和条件编译* 中了解更多关于这个主题的内容。

正如你在示例中看到的，静态类成员可以被完全特化。然而，每个特化都有自己的静态成员副本，以下示例进行了演示：

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   static T value;
```

```cpp
};
```

```cpp
template <typename T> T foo<T>::value = 0;
```

```cpp
template <> int foo<int>::value = 42;
```

```cpp
foo<double> a, b;  // a.value=0, b.value=0
```

```cpp
foo<int> c;        // c.value=42
```

```cpp
a.value = 100;     // a.value=100, b.value=100, c.value=42
```

在这里，`foo<T>` 是一个具有单个静态成员的类模板，该成员称为 `value`。对于主模板，它被初始化为 `0`，而对于 `int` 特化，它被初始化为 `42`。在声明变量 `a`、`b` 和 `c` 之后，`a.value` 是 `0`，`b.value` 也是 `0`，而 `c.value` 是 `42`。然而，在将值 `100` 赋给 `a.value` 之后，`b.value` 也变成了 `100`，而 `c.value` 保持为 `42`。

显式特化必须出现在主模板声明之后。它不需要在显式特化之前提供主模板的定义。因此，以下代码是有效的：

```cpp
template <typename T>
```

```cpp
struct is_floating_point;
```

```cpp
template <>
```

```cpp
struct is_floating_point<float>
```

```cpp
{
```

```cpp
   constexpr static bool value = true;
```

```cpp
};
```

```cpp
template <typename T>
```

```cpp
struct is_floating_point
```

```cpp
{
```

```cpp
   constexpr static bool value = false;
```

```cpp
};
```

模板特化也可以只声明而不定义。这样的模板特化可以像任何其他不完整类型一样使用。您可以在以下示例中看到这一点：

```cpp
template <typename>
```

```cpp
struct foo {};    // primary template
```

```cpp
template <>
```

```cpp
struct foo<int>;  // explicit specialization declaration
```

```cpp
foo<double> a; // OK
```

```cpp
foo<int>* b;   // OK
```

```cpp
foo<int> c;    // error, foo<int> incomplete type
```

在这个例子中，`foo<T>` 是一个主模板，对于它存在一个针对 `int` 类型的显式特化的声明。这使得可以使用 `foo<double>` 和 `foo<int>*`（支持声明指向部分类型的指针）。然而，在声明 `c` 变量时，完整的类型 `foo<int>` 不可用，因为缺少对 `int` 的完整特化的定义。这会生成编译器错误。

当特化一个函数模板时，如果编译器能够从函数参数的类型中推断出一个模板参数，那么这个模板参数是可选的。以下示例展示了这一点：

```cpp
template <typename T>
```

```cpp
struct foo {};
```

```cpp
template <typename T>
```

```cpp
void func(foo<T>) 
```

```cpp
{
```

```cpp
   std::cout << "primary template\n";
```

```cpp
}
```

```cpp
template<>
```

```cpp
void func(foo<int>) 
```

```cpp
{
```

```cpp
   std::cout << "int specialization\n";
```

```cpp
}
```

`func` 函数模板的 `int` 完整特化的语法应该是 `template<> func<int>(foo<int>)`。然而，编译器能够从函数参数的类型中推断出 `T` 实际表示的类型。因此，在定义特化时我们不必指定它。

另一方面，函数模板和成员函数模板的声明或定义不允许包含默认函数参数。因此，在以下示例中，编译器将发出错误：

```cpp
template <typename T>
```

```cpp
void func(T a)
```

```cpp
{
```

```cpp
   std::cout << "primary template\n";
```

```cpp
}
```

```cpp
template <>
```

```cpp
void func(int a = 0) // error: default argument not allowed
```

```cpp
{
```

```cpp
   std::cout << "int specialization\n";
```

```cpp
}
```

在所有这些示例中，模板只有一个模板参数。然而，在实际应用中，许多模板有多个参数。显式特化需要一个包含完整参数集的定义。以下代码片段展示了这一点：

```cpp
template <typename T, typename U>
```

```cpp
void func(T a, U b)
```

```cpp
{
```

```cpp
   std::cout << "primary template\n";
```

```cpp
}
```

```cpp
template <>
```

```cpp
void func(int a, int b)
```

```cpp
{
```

```cpp
   std::cout << "int-int specialization\n";
```

```cpp
}
```

```cpp
template <>
```

```cpp
void func(int a, double b)
```

```cpp
{
```

```cpp
   std::cout << "int-double specialization\n";
```

```cpp
}
```

```cpp
func(1, 2);      // int-int specialization
```

```cpp
func(1, 2.0);    // int-double specialization
```

```cpp
func(1.0, 2.0);  // primary template
```

在这些内容的基础上，我们可以继续前进，研究部分特化，它基本上是显式（完整）特化的泛化。

## 部分特化

当你对一个主模板进行部分特化，但只指定了一些模板参数时，就会发生部分特化。这意味着部分特化既有模板参数列表（紧随模板关键字之后）和模板参数列表（紧随模板名称之后）。然而，只有类才能进行部分特化。

让我们通过以下示例来了解它是如何工作的：

```cpp
template <typename T, int S>
```

```cpp
struct collection
```

```cpp
{
```

```cpp
   void operator()()
```

```cpp
   { std::cout << "primary template\n"; }
```

```cpp
};
```

```cpp
template <typename T>
```

```cpp
struct collection<T, 10>
```

```cpp
{
```

```cpp
   void operator()()
```

```cpp
   { std::cout << "partial specialization <T, 10>\n"; }
```

```cpp
};
```

```cpp
template <int S>
```

```cpp
struct collection<int, S>
```

```cpp
{ 
```

```cpp
   void operator()()
```

```cpp
   { std::cout << "partial specialization <int, S>\n"; }
```

```cpp
};
```

```cpp
template <typename T, int S>
```

```cpp
struct collection<T*, S>
```

```cpp
{ 
```

```cpp
   void operator()()
```

```cpp
   { std::cout << "partial specialization <T*, S>\n"; }
```

```cpp
};
```

我们有一个名为 `collection` 的主模板，它有两个模板参数（一个类型模板参数和一个非类型模板参数），并且我们有三个部分特化，如下所示：

+   非类型模板参数 `S` 的值为 `10` 的特化

+   `int` 类型的特化

+   指针类型 `T*` 的特化

这些模板可以按照以下代码片段所示使用：

```cpp
collection<char, 42> a;  // primary template
```

```cpp
collection<int,  42> b;  // partial specialization <int, S>
```

```cpp
collection<char, 10> c;  // partial specialization <T, 10>
```

```cpp
collection<int*, 20> d;  // partial specialization <T*, S>
```

如注释中所述，`a` 从主模板实例化，`b` 从 `int` 的部分特化（`collection<int, S>`）实例化，`c` 从 `10` 的部分特化（`collection<T, 10>`）实例化，而 `d` 从指针的部分特化（`collection<T*, S>`）实例化。然而，由于它们是模糊的，编译器无法选择使用哪个模板实例化，因此某些组合是不可能的。以下是一些示例：

```cpp
collection<int,   10> e; // error: collection<T,10> or 
```

```cpp
                         //        collection<int,S>
```

```cpp
collection<char*, 10> f; // error: collection<T,10> or 
```

```cpp
                         //        collection<T*,S>
```

在第一种情况下，`collection<T, 10>` 和 `collection<int, S>` 的部分特化都与类型 `collection<int, 10>` 匹配，而在第二种情况下，可以是 `collection<T, 10>` 或 `collection<T*, S>`。

在定义主模板的特化时，你需要记住以下几点：

+   部分特化的模板参数列表中的参数不能有默认值。

+   模板参数列表暗示了模板参数列表中参数的顺序，这仅在部分特化中才有特征。部分特化的模板参数列表不能与模板参数列表暗示的列表相同。

+   在模板参数列表中，你只能使用非类型模板参数的标识符。在此上下文中不允许使用表达式。以下示例展示了这一点：

    ```cpp
    template <int A, int B> struct foo {};
    template <int A> struct foo<A, A> {};     // OK
    template <int A> struct foo<A, A + 1> {}; // error
    ```

当一个类模板有部分特化时，编译器必须决定从哪个特化生成最佳匹配的定义。为此，它将模板特化的模板参数与主模板和部分特化的模板参数列表进行匹配。根据此匹配过程的结果，编译器执行以下操作：

+   如果没有找到匹配项，则从主模板生成一个定义。

+   如果找到一个单独的部分特化，则从该特化生成一个定义。

+   如果找到多个部分特化，则从最特化的部分特化生成一个定义，但前提是它是唯一的。否则，编译器会生成一个错误（如我们之前所见）。如果模板 `A` 接受的类型是模板 `B` 接受的子集，但反之则不然，则认为模板 `A` 比模板 `B` 更特化。

然而，部分特化不是通过名称查找来找到的，只有在通过名称查找找到主模板时才会考虑。

要了解部分特化的有用性，让我们看看一个现实世界的例子。

在这个例子中，我们想要创建一个函数，以优雅的方式格式化数组的内 容并将其输出到流中。格式化后的数组内容应看起来像 `[1,2,3,4,5]`。然而，对于 `char` 元素数组，元素之间不应用逗号分隔，而应显示为方括号内的字符串，例如 `[demo]`。为此，我们将考虑使用 `std::array` 类。以下实现使用分隔符格式化数组的内容：

```cpp
template <typename T, size_t S>
```

```cpp
std::ostream& pretty_print(std::ostream& os, 
```

```cpp
                           std::array<T, S> const& arr)
```

```cpp
{
```

```cpp
   os << '[';
```

```cpp
   if (S > 0)
```

```cpp
   {
```

```cpp
      size_t i = 0;
```

```cpp
      for (; i < S - 1; ++i)
```

```cpp
         os << arr[i] << ',';
```

```cpp
      os << arr[S-1];
```

```cpp
   }
```

```cpp
   os << ']';
```

```cpp
   return os;
```

```cpp
}
```

```cpp
std::array<int, 9> arr {1, 1, 2, 3, 5, 8, 13, 21};
```

```cpp
pretty_print(std::cout, arr);  // [1,1,2,3,5,8,13,21]
```

```cpp
std::array<char, 9> str;
```

```cpp
std::strcpy(str.data(), "template");
```

```cpp
pretty_print(std::cout, str);  // [t,e,m,p,l,a,t,e]
```

在这个片段中，`pretty_print` 是一个有两个模板参数的函数模板，与 `std::array` 类的模板参数相匹配。当用 `arr` 数组作为参数调用时，它打印 `[1,1,2,3,5,8,13,21]`。当用 `str` 数组作为参数调用时，它打印 `[t,e,m,p,l,a,t,e]`。然而，我们的意图是在后一种情况下打印 `[template]`。为此，我们需要另一个实现，它专门针对 `char` 类型：

```cpp
template <size_t S>
```

```cpp
std::ostream& pretty_print(std::ostream& os, 
```

```cpp
                           std::array<char, S> const& arr)
```

```cpp
{
```

```cpp
   os << '[';
```

```cpp
   for (auto const& e : arr)
```

```cpp
      os << e;
```

```cpp
   os << ']';
```

```cpp
   return os;
```

```cpp
}
```

```cpp
std::array<char, 9> str;
```

```cpp
std::strcpy(str.data(), "template");
```

```cpp
pretty_print(std::cout, str);  // [template]
```

在这个第二个实现中，`pretty_print` 是一个只有一个模板参数的函数模板，这个模板参数是一个非类型模板参数，表示数组的尺寸。类型模板参数被显式指定为 `char`，在 `std::array<char, S>` 中。这次，使用 `str` 数组调用 `pretty_print` 将 `[template]` 打印到控制台。

这里关键要理解的是，不是 `pretty_print` 函数模板被部分特化，而是 `std::array` 类模板。函数模板不能被特化，而我们这里有的是重载函数。然而，`std::array<char,S>` 是 `std::array<T, S>` 主类模板的一个特化。

本章中我们看到的所有示例要么是函数模板，要么是类模板。然而，变量也可以是模板，这将是下一节的主题。

# 定义变量模板

变量模板是在 C++14 中引入的，允许我们在命名空间作用域或类作用域中定义模板变量，在这种情况下，它们代表一组全局变量或静态数据成员。

变量模板在命名空间作用域中声明，如下面的代码片段所示。这是一个典型的例子，你可以在文献中找到，但我们可以用它来阐述变量模板的好处：

```cpp
template<class T>
```

```cpp
constexpr T PI = T(3.1415926535897932385L);
```

语法类似于声明变量（或数据成员），但结合了声明模板的语法。

产生的问题是变量模板实际上是如何有帮助的。为了回答这个问题，让我们构建一个示例来展示这个观点。假设我们想要编写一个函数模板，给定一个球体的半径，返回其体积。球体的体积是 `4πr³ / 3`。因此，一个可能的实现如下：

```cpp
constexpr double PI = 3.1415926535897932385L;
```

```cpp
template <typename T>
```

```cpp
T sphere_volume(T const r)
```

```cpp
{
```

```cpp
   return 4 * PI * r * r * r / 3;
```

```cpp
}
```

在这个例子中，`PI`被定义为`double`类型的编译时常量。如果我们使用`float`等类型作为类型模板参数`T`，这将生成编译器警告：

```cpp
float v1 = sphere_volume(42.0f); // warning
```

```cpp
double v2 = sphere_volume(42.0); // OK
```

解决这个问题的潜在方法是将`PI`作为模板类的静态数据成员，其类型由类型模板参数确定。这种实现可以如下所示：

```cpp
template <typename T>
```

```cpp
struct PI
```

```cpp
{
```

```cpp
   static const T value;
```

```cpp
};
```

```cpp
template <typename T> 
```

```cpp
const T PI<T>::value = T(3.1415926535897932385L);
```

```cpp
template <typename T>
```

```cpp
T sphere_volume(T const r)
```

```cpp
{
```

```cpp
   return 4 * PI<T>::value * r * r * r / 3;
```

```cpp
}
```

这方法是可行的，尽管使用`PI<T>::value`并不理想。如果能简单地写`PI<T>`会更好。这正是本节开头展示的变量模板`PI`允许我们做到的。下面是完整的解决方案：

```cpp
template<class T>
```

```cpp
constexpr T PI = T(3.1415926535897932385L);
```

```cpp
template <typename T>
```

```cpp
T sphere_volume(T const r)
```

```cpp
{
```

```cpp
   return 4 * PI<T> * r * r * r / 3;
```

```cpp
}
```

下一个示例展示了另一种可能的用法，并演示了变量模板的显式特化：

```cpp
template<typename T> 
```

```cpp
constexpr T SEPARATOR = '\n';
```

```cpp
template<> 
```

```cpp
constexpr wchar_t SEPARATOR<wchar_t> = L'\n';
```

```cpp
template <typename T>
```

```cpp
std::basic_ostream<T>& show_parts(
```

```cpp
   std::basic_ostream<T>& s, 
```

```cpp
   std::basic_string_view<T> const& str)
```

```cpp
{
```

```cpp
   using size_type = 
```

```cpp
      typename std::basic_string_view<T>::size_type;
```

```cpp
   size_type start = 0;
```

```cpp
   size_type end;
```

```cpp
   do
```

```cpp
   {
```

```cpp
      end = str.find(SEPARATOR<T>, start);
```

```cpp
      s << '[' << str.substr(start, end - start) << ']' 
```

```cpp
        << SEPARATOR<T>;
```

```cpp
      start = end+1;
```

```cpp
   } while (end != std::string::npos);
```

```cpp
   return s;
```

```cpp
}
```

```cpp
show_parts<char>(std::cout, "one\ntwo\nthree");
```

```cpp
show_parts<wchar_t>(std::wcout, L"one line");
```

在这个例子中，我们有一个名为`show_parts`的函数模板，它处理一个输入字符串，在分割由分隔符分隔的部分之后。分隔符是一个在（全局）命名空间作用域中定义的变量模板，并显式特化为`wchar_t`类型。

如前所述，变量模板可以是类的成员。在这种情况下，它们代表静态数据成员，需要使用`static`关键字进行声明。以下示例演示了这一点：

```cpp
struct math_constants
```

```cpp
{
```

```cpp
   template<class T>
```

```cpp
   static constexpr T PI = T(3.1415926535897932385L);
```

```cpp
};
```

```cpp
template <typename T>
```

```cpp
T sphere_volume(T const r)
```

```cpp
{
```

```cpp
   return 4 * math_constants::PI<T> *r * r * r / 3;
```

```cpp
}
```

你可以在类中声明一个变量模板，然后在其外部提供其定义。请注意，在这种情况下，变量模板必须使用`static const`声明，而不是`static constexpr`，因为后者需要在类内初始化：

```cpp
struct math_constants
```

```cpp
{
```

```cpp
   template<class T>
```

```cpp
   static const T PI;
```

```cpp
};
```

```cpp
template<class T>
```

```cpp
const T math_constants::PI = T(3.1415926535897932385L);
```

变量模板用于简化类型特性的使用。*显式特化*部分包含了一个名为`is_floating_point`的类型特性的示例。这里再次是主要模板：

```cpp
template <typename T>
```

```cpp
struct is_floating_point
```

```cpp
{
```

```cpp
   constexpr static bool value = false;
```

```cpp
};
```

有几个显式特化，这里不再列出。然而，这个`type`特性可以如下使用：

```cpp
std::cout << is_floating_point<float>::value << '\n';
```

使用`is_floating_point<float>::value`相当繁琐，但可以通过以下定义的变量模板来避免：

```cpp
template <typename T>
```

```cpp
inline constexpr bool is_floating_point_v = 
```

```cpp
   is_floating_point<T>::value;
```

这个`is_floating_point_v`变量模板有助于编写更简单、更易于阅读的代码。以下片段是我更倾向于使用，而不是使用`::value`的冗长变体的形式：

```cpp
std::cout << is_floating_point_v<float> << '\n';
```

标准库定义了一系列以`_v`后缀结尾的变量模板，用于`::value`，就像我们的例子一样（例如`std::is_floating_point_v`或`std::is_same_v`）。我们将在*第五章*中更详细地讨论这个主题，*类型特性与条件编译*。

变量模板的实例化方式类似于函数模板和类模板。这可以通过显式实例化或显式特化来实现，或者由编译器隐式地完成。当变量模板在需要存在变量定义的上下文中使用时，或者变量需要用于表达式的常量评估时，编译器会生成一个定义。

然后，我们转向别名模板的主题，它允许我们为类模板定义别名。

# 定义别名模板

在 C++ 中，可以使用 `typedef` 声明或 `using` 声明（后者是在 C++11 中引入的）。以下是一些使用 `typedef` 的示例：

```cpp
typedef int index_t;
```

```cpp
typedef std::vector<
```

```cpp
           std::pair<int, std::string>> NameValueList;
```

```cpp
typedef int (*fn_ptr)(int, char);
```

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   typedef T value_type;
```

```cpp
};
```

在这个例子中，`index_t` 是 `int` 的别名，`NameValueList` 是 `std::vector<std::pair<int, std::string>>` 的别名，而 `fn_ptr` 是返回 `int` 并有两个 `int` 和 `char` 类型的参数的函数指针类型的别名。最后，`foo::value_type` 是类型模板 `T` 的别名。

自 C++11 以来，这些类型别名可以通过以下形式的 **using 声明** 来创建：

```cpp
using index_t = int;
```

```cpp
using NameValueList = 
```

```cpp
   std::vector<std::pair<int, std::string>>;
```

```cpp
using fn_ptr = int(*)(int, char);
```

```cpp
template <typename T>
```

```cpp
struct foo
```

```cpp
{
```

```cpp
   using value_type = T;
```

```cpp
};
```

现在，使用声明比 `typedef` 声明更受欢迎，因为它们更容易使用，也更易于阅读（从左到右）。然而，它们比 `typedef` 有一个重要的优势，即允许我们为模板创建别名。**别名模板** 是一个名称，它不仅指向一个类型，而且指向一系列类型。记住，模板不是一个类、函数或变量，而是一个蓝图，它允许创建一系列类型、函数或变量。

要了解别名模板是如何工作的，让我们考虑以下示例：

```cpp
template <typename T>
```

```cpp
using customer_addresses_t = 
```

```cpp
   std::map<int, std::vector<T>>;            // [1]
```

```cpp
struct delivery_address_t {};
```

```cpp
struct invoice_address_t {};
```

```cpp
using customer_delivery_addresses_t =
```

```cpp
   customer_addresses_t<delivery_address_t>; // [2]
```

```cpp
using customer_invoice_addresses_t =
```

```cpp
   customer_addresses_t<invoice_address_t>;  // [3]
```

在第 `[1]` 行的声明中引入了别名模板 `customer_addresses_t`。它是一个映射类型的别名，其键类型为 `int`，值类型为 `std::vector<T>`。由于 `std::vector<T>` 不是一个类型，而是一系列类型，因此 `customer_addresses_t<T>` 定义了一系列类型。在第 `[2]` 和 `[3]` 行的 `using` 声明中，从上述类型系列中引入了两个类型别名，`customer_delivery_addresses_t` 和 `customer_invoice_addresses_t`。

别名模板可以出现在命名空间或类作用域中，就像任何模板声明一样。另一方面，它们既不能完全也不能部分特化。然而，有方法可以克服这种限制。一种解决方案是创建一个具有类型别名成员的类模板并特化该类。然后可以创建一个引用类型别名成员的别名模板。让我们通过以下示例来演示这一点。

虽然以下代码不是有效的 C++ 代码，但它代表了我想要实现的目标，如果别名模板的特化是可能的：

```cpp
template <typename T, size_t S>
```

```cpp
using list_t = std::vector<T>;
```

```cpp
template <typename T>
```

```cpp
using list_t<T, 1> = T;
```

在这个例子中，如果集合的大小大于 `1`，则 `list_t` 是 `std::vector<T>` 的别名模板。然而，如果只有一个元素，则 `list_t` 应该是类型模板参数 `T` 的别名。实际上实现这一点的示例如下：

```cpp
template <typename T, size_t S>
```

```cpp
struct list
```

```cpp
{
```

```cpp
   using type = std::vector<T>;
```

```cpp
};
```

```cpp
template <typename T>
```

```cpp
struct list<T, 1>
```

```cpp
{
```

```cpp
   using type = T;
```

```cpp
};
```

```cpp
template <typename T, size_t S>
```

```cpp
using list_t = typename list<T, S>::type;
```

在这个例子中，`list<T,S>` 是一个具有名为 `T` 的成员类型别名的类模板。在主模板中，这是一个 `std::vector<T>` 的别名。在部分特化 `list<T,1>` 中，它是 `T` 的别名。然后，`list_t` 被定义为 `list<T, S>::type` 的别名模板。以下断言证明了这一机制是有效的：

```cpp
static_assert(std::is_same_v<list_t<int, 1>, int>);
```

```cpp
static_assert(std::is_same_v<list_t<int, 2>, std::vector<int>>);
```

在我们结束本章之前，还有一个需要解决的问题：泛型 lambda 及其 C++20 改进，lambda 模板。

# 探索泛型 lambda 和 lambda 模板

Lambda 表达式，正式称为**lambda 表达式**，是在需要的地方简化定义函数对象的一种方法。这通常包括传递给算法的谓词或比较函数。尽管我们不会一般性地讨论 lambda 表达式，但让我们看看以下示例：

```cpp
int arr[] = { 1,6,3,8,4,2,9 };
```

```cpp
std::sort(
```

```cpp
   std::begin(arr), std::end(arr),
```

```cpp
   [](int const a, int const b) {return a > b; });
```

```cpp
int pivot = 5;
```

```cpp
auto count = std::count_if(
```

```cpp
   std::begin(arr), std::end(arr),
```

```cpp
   pivot {return a > pivot; });
```

Lambda 表达式是语法糖，是一种简化定义匿名函数对象的方法。当遇到 lambda 表达式时，编译器会生成一个具有函数调用操作符的类。对于前面的例子，它们可能看起来如下：

```cpp
struct __lambda_1
```

```cpp
{
```

```cpp
   inline bool operator()(const int a, const int b) const
```

```cpp
   {
```

```cpp
      return a > b;
```

```cpp
   }
```

```cpp
};
```

```cpp
struct __lambda_2
```

```cpp
{
```

```cpp
   __lambda_2(int & _pivot) : pivot{_pivot}
```

```cpp
   {} 
```

```cpp
   inline bool operator()(const int a) const
```

```cpp
   {
```

```cpp
      return a > pivot;
```

```cpp
   }
```

```cpp
private: 
```

```cpp
   int pivot;
```

```cpp
};
```

这里选择的名字是任意的，每个编译器都会生成不同的名字。此外，实现细节可能不同，这里看到的是编译器应该生成的最基本的内容。注意，第一个 lambda 和第二个 lambda 之间的区别在于后者包含通过值捕获的状态。

Lambda 表达式，在 C++11 中引入，在标准后来的版本中收到了几个更新。其中有两个特别值得注意，将在本章中讨论：

+   使用`auto`指定符而不是显式指定类型。这会将生成的函数对象转换为一个具有模板函数调用操作符的对象。

+   **模板 lambda**，在 C++20 中引入，允许我们使用模板语法显式指定模板化函数调用操作符的形状。

为了理解这些之间的区别以及泛型和模板 lambda 如何有帮助，让我们探索以下示例：

```cpp
auto l1 = [](int a) {return a + a; };  // C++11, regular 
```

```cpp
                                       // lambda
```

```cpp
auto l2 = [](auto a) {return a + a; }; // C++14, generic 
```

```cpp
                                       // lambda
```

```cpp
auto l3 = []<typename T>(T a) 
```

```cpp
          { return a + a; };   // C++20, template lambda
```

```cpp
auto v1 = l1(42);                      // OK
```

```cpp
auto v2 = l1(42.0);                    // warning
```

```cpp
auto v3 = l1(std::string{ "42" });     // error
```

```cpp
auto v5 = l2(42);                      // OK
```

```cpp
auto v6 = l2(42.0);                    // OK
```

```cpp
auto v7 = l2(std::string{"42"});       // OK
```

```cpp
auto v8 = l3(42);                      // OK
```

```cpp
auto v9 = l3(42.0);                    // OK
```

```cpp
auto v10 = l3(std::string{ "42" });    // OK
```

这里，我们有三个不同的 lambda：`l1`是一个常规 lambda，`l2`是一个泛型 lambda，因为至少有一个参数是用`auto`指定符定义的，而`l3`是一个模板 lambda，使用模板语法定义，但没有使用`template`关键字。

我们可以用一个整数来调用`l1`；我们也可以用`double`来调用它，但这次编译器将产生一个关于可能数据丢失的警告。然而，尝试用字符串参数调用它将产生编译错误，因为`std::string`不能转换为`int`。另一方面，`l2`是一个泛型 lambda。编译器将为其调用的所有参数类型实例化它的特化，在这个例子中是`int`、`double`和`std::string`。以下代码片段显示了生成的函数对象可能的样子，至少在概念上是这样：

```cpp
struct __lambda_3
```

```cpp
{
```

```cpp
   template<typename T1>
```

```cpp
   inline auto operator()(T1 a) const
```

```cpp
   {
```

```cpp
     return a + a;
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline int operator()(int a) const
```

```cpp
   {
```

```cpp
     return a + a;
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline double operator()(double a) const
```

```cpp
   {
```

```cpp
     return a + a;
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline std::string operator()(std::string a) const
```

```cpp
   {
```

```cpp
     return std::operator+(a, a);
```

```cpp
   }
```

```cpp
};
```

你可以在这里看到函数调用操作符的主要模板，以及我们提到的三个特殊化。不出所料，编译器将为第三个 lambda 表达式`l3`生成相同的代码，这是一个仅在 C++20 中可用的模板 lambda。由此产生的问题是泛型 lambda 和 lambda 模板有何不同？为了回答这个问题，让我们稍微修改一下之前的例子：

```cpp
auto l1 = [](int a, int b) {return a + b; };
```

```cpp
auto l2 = [](auto a, auto b) {return a + b; };
```

```cpp
auto l3 = []<typename T, typename U>(T a, U b) 
```

```cpp
          { return a + b; };
```

```cpp
auto v1 = l1(42, 1);                    // OK
```

```cpp
auto v2 = l1(42.0, 1.0);                // warning
```

```cpp
auto v3 = l1(std::string{ "42" }, '1'); // error
```

```cpp
auto v4 = l2(42, 1);                    // OK
```

```cpp
auto v5 = l2(42.0, 1);                  // OK
```

```cpp
auto v6 = l2(std::string{ "42" }, '1'); // OK
```

```cpp
auto v7 = l2(std::string{ "42" }, std::string{ "1" }); // OK 
```

```cpp
auto v8 = l3(42, 1);                    // OK
```

```cpp
auto v9 = l3(42.0, 1);                  // OK
```

```cpp
auto v10 = l3(std::string{ "42" }, '1'); // OK
```

```cpp
auto v11 = l3(std::string{ "42" }, std::string{ "42" }); // OK 
```

新的 lambda 表达式接受两个参数。再次，我们可以用两个整数或一个`int`和一个`double`调用`l1`（尽管这又会产生警告），但我们不能用字符串和`char`调用它。然而，我们可以使用泛型 lambda `l2`和 lambda 模板 `l3`做所有这些。编译器生成的代码对于`l2`和`l3`是相同的，从语义上看如下所示：

```cpp
struct __lambda_4
```

```cpp
{
```

```cpp
   template<typename T1, typename T2>
```

```cpp
   inline auto operator()(T1 a, T2 b) const
```

```cpp
   {
```

```cpp
     return a + b;
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline int operator()(int a, int b) const
```

```cpp
   {
```

```cpp
     return a + b;
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline double operator()(double a, int b) const
```

```cpp
   {
```

```cpp
     return a + static_cast<double>(b);
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline std::string operator()(std::string a, 
```

```cpp
                                 char b) const
```

```cpp
   {
```

```cpp
     return std::operator+(a, b);
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline std::string operator()(std::string a, 
```

```cpp
                                 std::string b) const
```

```cpp
   {
```

```cpp
     return std::operator+(a, b);
```

```cpp
   }
```

```cpp
};
```

在这个片段中，我们看到函数调用操作符的主要模板，以及几个完整的显式特殊化：对于两个`int`值，对于`double`和`int`，对于字符串和`char`，以及对于两个字符串对象。但如果我们想限制泛型 lambda `l2`的使用，使其仅限于相同类型的参数呢？这是不可能的。编译器无法推断我们的意图，因此，它将为参数列表中每个`auto`指定符的出现生成不同的类型模板参数。然而，C++20 中的 lambda 模板确实允许我们指定函数调用操作符的形式。看看下面的例子：

```cpp
auto l5 = []<typename T>(T a, T b) { return a + b; };
```

```cpp
auto v1 = l5(42, 1);        // OK
```

```cpp
auto v2 = l5(42, 1.0);      // error
```

```cpp
auto v4 = l5(42.0, 1.0);    // OK
```

```cpp
auto v5 = l5(42, false);    // error
```

```cpp
auto v6 = l5(std::string{ "42" }, std::string{ "1" }); // OK 
```

```cpp
auto v6 = l5(std::string{ "42" }, '1'); // error               
```

使用任何两种不同类型的两个参数调用 lambda 模板是不可能的，即使它们可以隐式转换，例如从`int`到`double`。编译器将生成一个错误。在调用模板 lambda 时，无法显式提供模板参数，例如在`l5<double>(42, 1.0)`中。这也会生成编译器错误。

`decltype`类型指定符允许我们告诉编译器从表达式推导类型。这个主题在*第四章*，*高级模板概念*中详细讨论。然而，在 C++14 中，我们可以在泛型 lambda 中使用它来声明上一个泛型 lambda 表达式中第二个参数的类型与第一个参数相同。更确切地说，这看起来如下所示：

```cpp
auto l4 = [](auto a, decltype(a) b) {return a + b; };
```

然而，这暗示了第二个参数`b`的类型必须可以转换为第一个参数`a`的类型。这允许我们编写以下调用：

```cpp
auto v1 = l4(42.0, 1);                  // OK
```

```cpp
auto v2 = l4(42, 1.0);                  // warning
```

```cpp
auto v3 = l4(std::string{ "42" }, '1'); // error
```

第一次调用编译没有任何问题，因为`int`可以隐式转换为`double`。第二次调用编译时会有警告，因为从`double`转换为`int`可能会丢失数据。然而，第三次调用会生成错误，因为`char`不能隐式转换为`std::string`。尽管`l4` lambda 比之前看到的泛型 lambda `l2`有所改进，但它仍然不能完全限制不同类型参数的调用。这只有通过前面展示的 lambda 模板才能实现。

下一个片段展示了 lambda 模板的另一个示例。这个 lambda 有一个单一参数，一个`std::array`。然而，数组的元素类型和数组的大小被指定为 lambda 模板的模板参数：

```cpp
auto l = []<typename T, size_t N>(
```

```cpp
            std::array<T, N> const& arr) 
```

```cpp
{ 
```

```cpp
   return std::accumulate(arr.begin(), arr.end(), 
```

```cpp
                          static_cast<T>(0));
```

```cpp
};
```

```cpp
auto v1 = l(1);                           // error
```

```cpp
auto v2 = l(std::array<int, 3>{1, 2, 3}); // OK
```

尝试使用除`std::array`对象以外的任何东西调用这个 lambda 会产生编译器错误。编译器生成的函数对象可能看起来如下：

```cpp
struct __lambda_5
```

```cpp
{
```

```cpp
   template<typename T, size_t N>
```

```cpp
   inline auto operator()(
```

```cpp
      const std::array<T, N> & arr) const
```

```cpp
   {
```

```cpp
     return std::accumulate(arr.begin(), arr.end(), 
```

```cpp
                            static_cast<T>(0));
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline int operator()(
```

```cpp
      const std::array<int, 3> & arr) const
```

```cpp
   {
```

```cpp
     return std::accumulate(arr.begin(), arr.end(), 
```

```cpp
                            static_cast<int>(0));
```

```cpp
   }
```

```cpp
};
```

与常规 lambda 相比，泛型 lambda 的一个有趣的好处是关于递归 lambda。Lambda 没有名字；它们是无名的，因此你不能直接递归调用它们。相反，你必须定义一个`std::function`对象，将 lambda 表达式赋值给它，并在捕获列表中通过引用捕获它。以下是一个递归 lambda 的示例，它计算一个数的阶乘：

```cpp
std::function<int(int)> factorial;
```

```cpp
factorial = &factorial {
```

```cpp
   if (n < 2) return 1;
```

```cpp
      else return n * factorial(n - 1);
```

```cpp
};
```

```cpp
factorial(5);
```

这可以使用泛型 lambda 简化。它们不需要`std::function`及其捕获。一个递归泛型 lambda 可以如下实现：

```cpp
auto factorial = [](auto f, int const n) {
```

```cpp
   if (n < 2) return 1;
```

```cpp
   else return n * f(f, n - 1);
```

```cpp
};
```

```cpp
factorial(factorial, 5);
```

如果理解这一点有困难，编译器生成的代码应该能帮助你弄清楚：

```cpp
struct __lambda_6
```

```cpp
{
```

```cpp
   template<class T1>
```

```cpp
   inline auto operator()(T1 f, const int n) const
```

```cpp
   {
```

```cpp
     if(n < 2) return 1;
```

```cpp
     else return n * f(f, n - 1);
```

```cpp
   }
```

```cpp
   template<>
```

```cpp
   inline int operator()(__lambda_6 f, const int n) const
```

```cpp
   {
```

```cpp
     if(n < 2) return 1;
```

```cpp
     else return n * f.operator()(__lambda_6(f), n - 1);
```

```cpp
   }
```

```cpp
};
```

```cpp
__lambda_6 factorial = __lambda_6{};
```

```cpp
factorial(factorial, 5);
```

一个通用的 lambda 是一个具有模板函数调用操作符的函数对象。第一个参数，使用`auto`指定，可以是任何东西，包括 lambda 本身。因此，编译器将为生成的类的类型提供一个完整的显式特化调用操作符。

Lambda 表达式帮助我们避免在需要将函数对象作为参数传递给其他函数时编写显式代码。相反，编译器为我们生成这些代码。C++14 中引入的泛型 lambda 帮助我们避免为不同类型编写相同的 lambda。C++20 的 lambda 模板允许我们使用模板语法和语义指定生成的调用操作符的形式。

# 摘要

本章是对 C++模板核心特性的概述。我们学习了如何定义类模板、函数模板、变量模板和别名模板。在学习模板参数之后，我们详细研究了模板实例化和模板特化。我们还学习了泛型 lambda 和 lambda 模板以及它们与常规 lambda 相比的优势。通过完成本章，你现在熟悉了模板基础知识，这应该允许你理解大量模板代码，并自己编写模板。

在下一章中，我们将探讨另一个重要主题，即具有可变数量参数的模板，称为变长模板。

# 问题

1.  哪些类型的类别可以用于非类型模板参数？

1.  默认模板参数不允许在哪些地方使用？

1.  显式实例化声明是什么，它与显式实例化定义在语法上有什么区别？

1.  什么是别名模板？

1.  什么是模板 lambda？

# 进一步阅读

+   C++ 模板：快速更新查看(C++11/14/17/20)，[`www.vishalchovatiya.com/c-template-a-quick-uptodate-look/`](http://www.vishalchovatiya.com/c-template-a-quick-uptodate-look/)

+   C++ 的模板别名，[`www.stroustrup.com/template-aliases.pdf`](https://www.stroustrup.com/template-aliases.pdf)

+   Lambda：从 C++11 到 C++20，第二部分，[`www.cppstories.com/2019/03/lambdas-story-part2/`](https://www.cppstories.com/2019/03/lambdas-story-part2/)
