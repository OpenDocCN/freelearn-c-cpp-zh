

# 第八章：使用模板构建通用和可重用代码

在本书的先前示例中，我们已经使用了类模板，但没有对其进行详细解释。到现在为止，你应该对 C++中的模板有基本的了解，并且知道如何使用标准库中的模板容器类来专门化具有不同底层类型的容器。我们还介绍了`std::optional`和`std::expected`模板类，我们可以使用它们来处理函数的不同返回类型。

正如你所看到的，模板在 C++标准库中被广泛使用。它们允许我们对不同类型实现相同的功能，使我们的代码可重用和通用，这是 C++的一个优点。模板是一个极其复杂的话题；关于 C++模板和元编程的整本书都已经被写出来了。本章将帮助你更详细地了解 C++中的模板。

在本章中，我们将涵盖以下主要主题：

+   模板基础

+   元编程

+   概念

+   编译时多态

# 技术要求

为了充分利用本章内容，我强烈建议你在阅读示例时使用 Compiler Explorer ([`godbolt.org/`](https://godbolt.org/))。选择 GCC 作为 x86 架构的编译器。这将允许你看到标准输出并更好地观察代码的行为。由于我们使用现代 C++，请确保选择 C++23 标准，通过在编译器选项框中添加`-std=c++23`来实现。

Compiler Explorer 使得尝试代码、调整代码并立即看到它如何影响输出和生成的汇编代码变得容易。本章的示例可以在 GitHub 上找到（[`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter08`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter08))。

# 模板基础

“模板”这个词的一个定义是“一个用于指导正在制作的零件形状的量规、图案或模具（如薄板或板）。”这个定义可以应用于 C++中的模板。

在 C++中，模板充当函数和类的模式或模具，允许创建实际的函数和类。从这个角度来看，模板本身不是真正的函数或类型；相反，它们作为生成具体函数和类型的指南。为了更好地理解这个定义，让我们看一下以下代码示例：

```cpp
#include <cstdio>
template<typename T>
T add(T a, T b) {
   return a + b;
}
int main() {
    int result_int = add(1, 4);
    float result_float = add(1.11f, 1.91f);
    printf("result_int = %d\r\n", result_int);
    printf("result_float = %.2f\r\n", result_float);
    return 0;
} 
```

在这个示例中，我们有一个模板函数`add`，其模板类型参数为`T`。在`main`函数中，我们看到对`add`函数的两次调用：

+   第一个模板接受整数参数，并将返回值存储在`result_int`

+   第二个模板接受浮点数参数，并将返回值存储在`result_float`浮点变量中

现在，我们之前说过，模板类型和函数不是真正的类型和函数，那么如果它不是一个真正的函数，我们如何调用模板函数呢？

## 调用模板函数

在这个例子中，当编译器看到对模板函数的调用时，它会推断模板参数，并用类型 `int` 替换第一个调用中的模板参数 `T`，在第二个调用中用 `float` 替换。在参数推断之后，模板被实例化；也就是说，编译器创建了两个 `add` 函数实例：一个接受整数作为参数，另一个接受浮点数。我们可以在前面示例的汇编输出中看到这一点：

```cpp
_Z3addIiET_S0_S0_:
        push    rbp
        mov     rbp, rsp
        mov     DWORD PTR [rbp-4], edi
        mov     DWORD PTR [rbp-8], esi
        mov     edx, DWORD PTR [rbp-4]
        mov     eax, DWORD PTR [rbp-8]
        add     eax, edx
        pop     rbp
        ret
_Z3addIfET_S0_S0_:
        push    rbp
        mov     rbp, rsp
        movss   DWORD PTR s[rbp-4], xmm0
        movss   DWORD PTR [rbp-8], xmm1
        movss   xmm0, DWORD PTR [rbp-4]
        addss   xmm0, DWORD PTR [rbp-8]
        pop     rbp
        ret 
```

在前面的汇编输出中，我们看到有两个 `add` 函数实例：`_Z3addIiET_S0_S0_`，接受整数，和 `_Z3addIfET_S0_S0_`，接受浮点数。编译器在确定了此函数调用点的模板参数后，从 `add` 模板函数中实例化了这两个函数。这是 C++ 中模板的基本工作原理。

在 `add` 模板函数的例子中，编译器将为每个定义了 `operator+` 的类型实例化一个新的函数。那么，如果我们尝试对一个没有定义 `operator+` 的类型调用 `add` 模板函数会发生什么？让我们看一下以下例子：

```cpp
struct point {
    int x;
    int y;
};
int main() {
    point a{1, 2};
    point b{2, 1};
    auto c = add(a, b);
    return 0;
} 
```

在前面的例子中，我们定义了一个 `point` 结构体，对于它没有定义 `operator+`，并且调用了 `add` 模板函数。这将导致编译器错误，类似于下面显示的错误：

```cpp
<source>: In instantiation of 'T add(T, T) [with T = point]':
<source>:25:17:   required from here
   25 |     auto c = add(a, b);
      |              ~~~^~~~~~
<source>:6:13: error: no match for 'operator+' (operand types are 'point' and 'point')
    6 |    return a + b;
      |           ~~^~~ 
```

那么，发生了什么？当编译器尝试使用 `add` 模板并使用 `point` 作为类型 `T` 实例化函数时，由于 `no match for 'operator+' (operand types are 'point' and 'point')`，编译失败。我们可以通过如下定义 `point` 结构体的 `operator+` 来解决这个问题：

```cpp
struct point {
    int x;
    int y;
    point operator+(const point& other) const {
        return point{x + other.x, y + other.y};
    }
    void print() {
        printf("x = %d, y = %d\r\n", x, y);
    }
}; 
```

在前面的实现中，我们为 `point` 结构体定义了 `operator+`，并且还定义了 `print` 函数，这将帮助我们打印点。在此更改之后，我们可以成功编译示例。

如果出于某种原因，我们想让 `add` 函数在类型 `point` 上使用时表现得与直接应用 `operator+` 不同，会怎样？比如说，我们想在求和后同时将 `x` 和 `y` 增加 1。我们可以使用模板特化来实现这一点。

## 模板特化

**模板特化** 允许我们向编译器提供特定类型的模板函数的实现，如下面的例子所示，特化了 `add` 函数以适用于类型 `point`：

```cpp
template<>
point add<point>(point a, point b) {
   return point{a.x+b.x+1, a.y+b.y+1};
} 
```

在这种情况下，当使用类型 `point` 的参数调用 `add` 函数时，编译器会跳过泛型模板实例化，而是使用这个专门的版本。这允许我们针对点对象特别定制函数的行为，当两个点实例相加时，每个坐标都会增加 1。现在让我们看一下完整的 `main` 函数：

```cpp
int main() {
    point a{1, 2};
    point b{2, 1};
    auto c = add(a, b);
    c.print();
    static_assert(std::is_same_v<decltype(c), point>);
    return 0;
} 
```

如果我们运行之前步骤中的模板特化示例，我们将得到以下输出：

```cpp
x = 4, y = 4 
```

编译器为`point`类型使用了函数特化。模板特化使模板成为一个灵活的工具，允许我们在需要时向编译器提供自定义实现。

在前面的例子中，我们可以看到对于变量`c`，我们使用了`auto`作为类型指定符。`auto`关键字是在 C++11 中引入的，当使用时，编译器会从初始化表达式中推导出变量的实际类型。为了确认变量`c`推导出的类型是`point`，我们使用了`static_assert`，它执行编译时断言检查。

作为`static_assert`的参数，我们使用元编程库中的类型特性`std::is_same_v`，它检查两个类型是否相同，如果相同则评估为`true`。我们使用`decltype`指定符确定`c`的类型，它会在编译时检索表达式的类型。这允许我们验证为`c`推导出的类型确实是`point`。如果这个断言失败，编译器将生成错误。

# 模板元编程

**模板元编程**涉及使用模板编写在编译时根据模板参数中使用的类型生成不同函数、类型和常量的代码。模板元编程是现代 C++库中广泛使用的高级技术。它可能令人难以理解，所以如果它看起来很难理解，那完全没问题。把这仅仅看作是对这个有趣主题的介绍和探索。

让我们回到`add`模板函数的例子。如果我们想强制这个模板函数只用于整数和浮点数等算术类型，我们能做些什么呢？

来自元编程库的`<type_traits>`头文件为我们提供了`std::enable_if`模板类型，它接受两个参数，一个布尔值和一个类型。如果布尔值为真，结果类型将有一个公共`typedef`成员`type`。让我们看看以下例子：

```cpp
#include <type_traits>
template<typename T>
std::enable_if<true, T>::type
add(T a, T b) {
   return a + b;
} 
```

在前面的例子中，我们用`std::enable_if`代替了`add`模板函数的返回类型。因为我们设置了布尔参数为`true`，它将有一个公共`typedef`类型`T`，这意味着`add`函数模板的返回类型将是`T`。

我们将使用类型特性类模板`std::is_arithmetic<T>`扩展这个例子，它将有一个名为`value`的公共布尔值，如果`T`是算术类型则设置为`true`。前面的例子将产生以下代码：

```cpp
template<typename T>
std::enable_if<std::is_arithmetic<T>::value, T>::type
add(T a, T b) {
   return a + b;
} 
```

在前面的例子中，我们不是将`true`硬编码为`std::enable_if`的条件，而是使用了`std::is_arithmetic<T>::value`。让我们看看使用这个模板函数和前面例子中的`point`类型的`main`函数：

```cpp
int main() {
    auto a = add(1, 2); // OK
    auto b = add(1.1, 2.1); // OK
    point p_a{1, 2};
    point p_b{2, 1}; 
    auto p_c = add(p_a, p_b); // compile-error
    return 0;
} 
```

如果我们尝试编译这段代码，编译将失败，并显示一个包含以下内容的冗长错误消息：

```cpp
<source>: In function 'int main()':
<source>:30:17: error: no matching function for call to 'add(point&, point&)'
  30 |     auto c = add(p_a, p_b); // compile-error
     |              ~~~^~~~~~~~~~
<source>:30:17: note: there is 1 candidate
<source>:19:1: note: candidate 1: 'template<class T> typename std::enable_if<std::is_arithmetic<_Tp>::value, T>::type add(T, T)'
  19 | add(T a, T b) {
     | ^~~
<source>:19:1: note: template argument deduction/substitution failed:
<source>: In substitution of 'template<class T> typename std::enable_if<std::is_arithmetic<_Tp>::value, T>::type add(T, T) [with T = point]':
<source>:30:17:   required from here
  30 |     auto c = add(p_a, p_b); // compile-error
     |              ~~~^~~~~~~~~~
<source>:19:1: error: no type named 'type' in 'struct std::enable_if<false, point>'
  19 | add(T a, T b) {
     | ^~~ 
```

前面的编译器错误看起来很吓人，很难阅读。这是模板臭名昭著的问题之一。在我们解决这个问题之前，让我们专注于分析这个案例中发生了什么。

模板参数推导/替换失败，因为 `std::is_arithmetic<point>::value` 结果为 `false`，这意味着 `std::enable_if` 模板类型将不会有公共 typedef `type T`。实际上，任何尝试在这个例子中使用非算术类型的 `add` 模板函数都将导致编译器错误，即使该类型定义了 `operator+`。我们可以将 `std::enable_if` 视为 C++ 中模板函数的启用器或禁用器。

让我们修改 `add` 模板函数，使其打印求和操作的结果。由于整数和浮点数都是算术类型，我们需要对它们进行不同的处理。我们可以使用 `std::enable_if` 并创建两个模板函数，使用 `std::is_integral` 和 `std::is_floating_point` 类型特性，如下例所示：

```cpp
template<typename T>
std::enable_if<std::is_integral<T>::value, T>::type
add(T a, T b) {
    T result = a + b;
    printf("%d + %d = %d\r\n", a, b, result);
    return result;
}
template<typename T>
std::enable_if<std::is_floating_point<T>::value, T>::type
add(T a, T b) {
    T result = a + b;
    printf("%.2f + %.2f = %.2f\r\n", a, b, result);
    return result;
} 
```

如你所记，`std::enable_if` 是一个模板启用器或禁用器，意味着它将启用整数类型的第一个模板函数，并使用 `printf` 和 `%d` 格式说明符打印它们。对于整数类型的第二个模板函数，模板替换将失败，但不会被视为错误，因为第一个模板对于整数参数有一个有效的函数候选者。这个原则被称为 **替换失败不是错误**（**SFINAE**）。对于浮点类型，第一个模板函数将被禁用，但第二个将被启用。

现在，我们使用的示例函数非常简单，但让我们暂时假设 `add` 函数模板正在执行一个繁重的任务，并且整数和浮点数版本之间的唯一区别是我们如何打印结果。因此，如果我们使用两个不同的函数模板，我们将复制大量的相同代码。我们可以通过使用 `constexpr if` 来避免这种情况，它将在编译时启用或禁用代码中的某些路径。让我们看看一个修改后的示例：

```cpp
std::enable_if_t<std::is_arithmetic_v<T>, T>
add(T a, T b) {
    T result = a + b;
    if constexpr (std::is_integral_v<T>) {
        printf("%d + %d = %d\r\n", a, b, result);
    } else if constexpr (std::is_floating_point_v<T>) {
        printf("%.2f + %.2f = %.2f\r\n", a, b, result);
    }
    return a + b;
} 
```

在前面的例子中，我们使用了 `constexpr if` 语句根据 `std::is_integral_v<T>` 和 `std::is_floating_point_v<T>` 表达式的编译时评估来启用程序的某些路径。`constexpr if` 是在 C++17 中引入的。你还可以注意到，我们使用了类型特性的别名作为 `std::enable_if_t<T>`，它等价于 `std::enable_if<T>::type`，以及 `std::is_floating_point_v<T>`，它等价于 `std::is_floating_point<T>::value`。

在这个例子中，我们使用了类型特性和 `std::enable_if` 来仅对算术类型启用 `add` 函数模板。C++20 引入了概念，我们可以用它来对模板类型施加限制。

# 概念

**概念**是模板参数要求的命名集合。它们在编译时评估，并在重载解析期间用于选择最合适的函数重载；也就是说，它们用于确定哪个函数模板将被实例化和编译。

我们将创建一个用于算术类型的概念，并在我们的`add`模板函数中使用它，如下所示：

```cpp
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;
template<Arithmetic T>
T add(T a, T b) {
    T result = a + b;
    if constexpr (std::is_integral_v<T>) {
        printf("%d + %d = %d\r\n", a, b, result);
    } else if constexpr (std::is_floating_point_v<T>) {
        printf("%.2f + %.2f = %.2f\r\n", a, b, result);
    }
    return a + b;
} 
```

在前面的代码中，我们创建了`Arithmetic`概念，并在`add`函数模板中使用它来对`T`模板类型提出要求。现在`add`模板函数更容易阅读。从模板声明中可以看出，类型`T`必须满足`Arithmetic`概念的要求，这使得代码更容易阅读和理解。

概念不仅使代码更容易阅读，还提高了编译器错误的可读性。如果我们尝试在`point`类型上调用函数模板`add`，我们现在会得到一个类似于以下错误的错误：

```cpp
<source>: In function 'int main()':
<source>:41:17: error: no matching function for call to 'add(point&, point&)'
  41 |     auto c = add(p_a, p_b); // compile-error
     |              ~~~^~~~~~~~~~
<source>:41:17: note: there is 1 candidate
<source>:22:3: note: candidate 1: 'template<class T>  requires  Arithmetic<T> T add(T, T)'
  22 | T add(T a, T b) {
     |   ^~~
<source>:22:3: note: template argument deduction/substitution failed:
<source>:22:3: note: constraints not satisfied
<source>: In substitution of 'template<class T>  requires  Arithmetic<T> T add(T, T) [with T = point]':
<source>:41:17:   required from here
  41 |     auto c = add(p_a, p_b); // compile-error
     |              ~~~^~~~~~~~~~
<source>:18:9:   required for the satisfaction of 'Arithmetic<T>' [with T = point]
<source>:18:27: note: the expression 'is_arithmetic_v<T> [with T = point]' evaluated to 'false'
  18 | concept Arithmetic = std::is_arithmetic_v<T>;
     |                      ~~~~~^~~~~~~~~~~~~~~~~~ 
```

之前的编译器错误比我们之前没有使用概念时的错误更容易阅读和理解发生了什么。我们可以轻松地追踪错误的起源，即`Arithmetic`概念对`point`类型施加的约束没有得到满足。

接下来，我们将继续讨论编译时多态，并看看我们如何利用概念来帮助我们强制执行强接口。

# 编译时多态

在*第五章*中，我们讨论了动态的或运行时多态。我们用它来定义`uart`的接口，该接口由`uart_stm32`类实现。`gsm_lib`类只依赖于`uart`接口，而不是具体的实现，该实现包含在`uart_stm32`中。这被称为**松耦合**，使我们能够为`gsm_lib`类拥有可移植的代码。

我们可以轻松地在不同的硬件平台上为`gsm_lib`提供另一个`uart`接口实现。这个原则被称为**依赖倒置**。它表示高层模块（类）不应该依赖于低层模块，而两者都应该依赖于抽象（接口）。我们可以通过在 C++中使用继承和虚函数来实现这个原则。

虚函数会导致间接引用，从而增加运行时开销和实现所需的二进制大小。它们允许运行时调度函数调用，但这也带来了代价。在嵌入式应用中，我们通常知道所有我们的类型，这意味着我们可以使用模板和重载解析来进行静态或编译时调度函数调用。

### 使用类模板进行编译时多态

我们可以将`gsm_lib`制作成一个类模板，它有一个参数，我们将用它来指定`uart`类型，如下面的示例所示：

```cpp
#include <span>
#include <cstdio>
#include <cstdint>
class uart_stm32 {
public:
    void init(std::uint32_t baudrate = 9600) {
        printf("uart_stm32::init: setting baudrate to %d\r\n", baudrate);
    }
    void write(std::span<const char> data) {
        printf("uart_stm32::write: ");
        for(auto ch: data) {
            putc(ch, stdout);
        }
    }
};
template<typename T>
class gsm_lib{
public:
    gsm_lib(T &u) : uart_(u) {}
    void init() {
        printf("gsm_lib::init: sending AT command\r\n");
        uart_.write("AT");
    }
private:
    T &uart_;
};
int main() {
    uart_stm32 uart_stm32_obj;
    uart_stm32_obj.init(115200);
    gsm_lib gsm(uart_stm32_obj);
    gsm.init();
    return 0;
} 
```

在上述示例中，编译器将使用`uart_stm32`类作为模板参数实例化`gsm_lib`模板类。这将导致在`gsm_lib`代码中使用`uart_stm32`类的对象引用。我们仍然可以通过使用提供所有编译所需方法的不同类型来轻松重用`gsm_lib`。在这个例子中，与`gsm_lib`类模板一起使用的类型必须提供一个接受`std::span<char>`作为其参数的`write`方法。但这同时也意味着任何具有此类方法的类型都将允许我们编译代码。

动态多态需要接口类在具体类中实现并在高级代码中使用。当阅读代码时，它使代码的预期行为变得清晰。我们能否使用模板做类似的事情？结果是我们可以的。我们可以使用**奇特重复的模板模式**（**CRTP**）来实现编译时子类型多态。

## 奇特重复的模板模式 (CRTP)

CRTP 是 C++的一种惯用法，其中派生类使用一个以自身作为基类的模板类实例化。是的，听起来很复杂，所以让我们通过代码更好地理解这一点：

```cpp
template<typename U>
class uart_interface {
public:
    void init(std::uint32_t baudrate = 9600) {
       static_cast<U*>(this)->initImpl(baudrate);
    }
};
class uart_stm32 : public uart_interface<uart_stm32> {
public:
    void initImpl(std::uint32_t baudrate = 9600) {
        printf("uart_stm32::init: setting baudrate to %d\r\n", baudrate);
    }
}; 
```

上述代码实现了 CRTP。`uart_stm32`派生类从使用`uart_stm32`类本身实例化的`uart_interface`类模板继承。基类模板公开了一个接口，它可以通过对`this`（指向自身的指针）使用`static_cast`来访问派生类。它提供了`init`方法，该方法在`uart_stm32`类的对象上调用`initImpl`。

CRTP 允许我们在基类中定义我们的接口并在派生类中实现它，类似于我们用于运行时多态的继承机制。为了确保此接口在`gsm_lib`中使用，我们需要使用概念创建类型约束，如下所示：

```cpp
template<typename T>
concept TheUart = std::derived_from<T, uart_interface<T>>; 
```

上述代码是我们将用于限制`gsm_lib`类模板接受类型的概念。它将仅接受由该类型本身实例化的`uart_interface`类模板派生的类型。以下是一个完整的代码示例：

```cpp
#include <span>
#include <cstdio>
#include <cstdint>
template<typename U>
class uart_interface {
public:
    void init(std::uint32_t baudrate = 9600) {
       static_cast<U*>(this)->initImpl(baudrate);
    }
    void write(std::span<const char> data) {
       static_cast<U*>(this)->writeImpl(data);
    }
};
class uart_stm32 : public uart_interface<uart_stm32> {
public:
    void initImpl(std::uint32_t baudrate = 9600) {
        printf("uart_stm32::init: setting baudrate to %d\r\n", baudrate);
    }
    void writeImpl(std::span<const char> data) {
        printf("uart_stm32::write: ");
        for(auto ch: data) {
            putc(ch, stdout);
        }
    }
};
template<typename T>
concept TheUart = std::derived_from<T, uart_interface<T>>;
template<TheUart T>
class gsm_lib{
public:
    gsm_lib(T &u) : uart_(u) {}
    void init() {
        printf("gsm_lib::init: sending AT command\r\n");
        uart_.write("AT");
    }
private:
    T &uart_;
};
int main() {
    uart_stm32 uart_stm32_obj;
    uart_stm32_obj.init(115200);
    gsm_lib gsm(uart_stm32_obj);
    gsm.init();
    return 0;
} 
```

在上述代码中，我们使用 CRTP 实现了编译时或静态子类型多态。`uart_stm32`是一个具体类，它依赖于由`uart_interface`类模板定义的接口。我们使用`TheUart`概念来约束`gsm_lib`中从`uart_interface`派生的类型的高级代码。我们通过 CRTP 和概念实现了依赖反转，并且它得到了清晰的定义。

与继承（运行时多态）相比，编译时多态的主要优势是静态绑定；也就是说，没有虚拟函数。这以模板语法为代价，可能会使代码更难阅读和理解。

# 摘要

在本章中，我们介绍了模板基础、模板元编程、概念以及编译时多态。虽然模板是一个包含许多更深层次概念的先进主题，但本章旨在为新学习者提供一个坚实的起点。通过理解这里涵盖的基本原理，你应该能够很好地探索模板的更复杂方面，并在嵌入式系统编程中充分利用它们的全部潜力。

在下一章中，我们将讨论 C++ 中的类型安全。
