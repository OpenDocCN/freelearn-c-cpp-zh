# 3

# 类型转换和 cv-qualifications

我们正在进步。在*第一章*中，我们探讨了内存、对象和指针是什么，因为我们知道如果我们想要掌握内存管理机制，我们就需要理解这些基本概念。然后在*第二章*中，我们查看了一些低级构造，如果误用可能会给我们带来麻烦，但在某些情况下理解这些构造对于掌握程序如何管理内存是至关重要的。这是一个相对枯燥的开端，但也意味着我们工作的有趣部分还在后面。我希望这能给你带来鼓舞！

在*第二章*的结尾，我们探讨了类型欺骗的方法，这是一种绕过类型系统的方法，包括一些被认为可以工作但实际上并不奏效的方法。C++提供了一些受控和明确的方式来与类型系统交互，通知编译器它应该将表达式的类型视为与从源代码中推断出的不同。这些工具，即**类型转换**（或简称*转换*），是本章的主题。

我们首先将探讨在一般意义上什么是类型转换，区分进行类型转换的各种基本原因，并说明为什么在 C++程序中 C 风格类型转换通常是不合适的（除了某些特定情况）。然后，我们将快速查看 C++系统的一个与安全相关的方面，即**cv-qualifications**，并讨论 cv 限定符在 C++代码的卫生性和整体质量中的作用。之后，我们将检查我们可用的六个 C++类型转换。最后，我们将回到 C 类型转换，以展示它们在何种有限情况下可能仍然适用。

在本章中，我们将学习以下内容：

+   类型转换是什么以及它们在程序中的含义

+   cv-qualifications 是什么以及它们如何与类型转换交互

+   C++类型转换是什么，包括 C 类型转换，以及何时应该使用它们

# 技术要求

你可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter3`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter3)。

# 什么是类型转换？

你将使用类型转换来调整编译器对表达式类型的看法。问题是，编译器看到我们的源代码，理解我们写了什么，以及别人的代码表达了什么。大多数时候（希望如此），这段代码是有意义的，编译器将把你的源代码转换为适当的二进制文件而不会抱怨。

当然，有时程序员意图与代码之间会有（希望是暂时的）差异，这种差异通过编译器看到的源代码表达出来。大多数时候，编译器是正确的，程序员会重写源代码，至少部分地，以便更好地表达意图，受到揭示问题的错误或警告信息的启发（以它们自己诗意的方式）。当然，有时源代码与程序员的意图相匹配，但仍然与编译器存在分歧，需要调整以达到某种程度的共识。例如，假设程序员想要分配一个足够大的缓冲区来存储大量的整数（`lots` 是一个太大以至于无法合理使用栈或编译时未知的值）；实现这一目标的一种（低级且容易出错但仍然合法）方法就是调用 `std::malloc()` 函数：

```cpp
// ...
int *p = std::malloc(lots * sizeof(int)); // <-- HERE
if(p) {
    // use p as an array of int objects
    std::free(p);
}
// ...
```

如你所知，这段代码摘录不是有效的 C++代码 – `std::malloc()` 返回 `void*`（一个指向至少请求大小的原始内存块的指针，如果分配失败则返回 `nullptr`），而在 C++中 `void*` 不能隐式转换为 `int*`（反之亦然，当然，`int*` 可以隐式转换为 `void*`）。

注意，在这种情况下，我们可以用 `new int[lots]` 替换 `std::malloc(lots*sizeof(int))`（这是一个过于简化的例子），但事情并不总是这么简单，有时我们需要对类型系统撒谎，即使只是一瞬间。这就是类型转换的作用所在。

那么，什么是类型转换呢？类型转换是一种受控的方式来引导编译器的类型系统理解程序员的意图。类型转换还在源代码中提供了关于这种暂时性谎言背后原因的信息；它们记录了程序员在需要撒谎的那一刻的意图。C++的类型转换在传达意图方面非常明确，在效果上非常精确；C 风格类型转换（在其他语言中也可见）在意图方面更为模糊，正如我们将在本章后面看到的那样，并且可以在具有如此丰富类型系统的 C++语言中执行不适当的转换。

# 类型系统中的安全性 – cv-资格

C++在其类型系统中提供了两个与安全性相关的资格符。这些被称为 `const` 和 `volatile`，它们在许多方面都有关联。

`const` 资格符表示被此资格符指定的对象在当前作用域中被认为是不可变的，例如以下情况：

```cpp
const int N = 3; // global constant
class X {
    int n; // note: not const
public:
    X(int n) : n{ n } {
    }
    int g() { // note: not const
      return n += N; // thus, n's state can be mutated
    }
    int f() const { // const applies to this, and
                    // transitively to its members
      // return g(); // illegal as g() is not const
      return n + 1;
    }
};
int f(const int &n) { // f() will not mutate argument n
    return X{ n }.f() + 1; // X::X(int) takes its argument
                          // by value so n remains intact
}
int main() {
    int a = 4;
    a = f(a); // a is not const in main()
}
```

将一个对象标记为 `const` 意味着在它被标记为这样的上下文中，它不能被修改。在类成员的情况下，`const` 保证通过 `const` 成员函数传递，也就是说，一个 `const` 成员函数不能修改 `*this` 的成员，也不能调用同一对象的非 `const` 成员函数。在前面的例子中，`X::f` 是 `const` 的，因此它不能调用 `X::g`，后者不提供这种保证；允许 `X::f` 调用 `X::g` 将实际上破坏 `const` 保证，因为 `X::g` 可以修改 `*this`，而 `X::f` 不能。

`const` 标记在 C++ 中是众所周知且文档齐全的。通常认为“`const`-correct”是良好的代码卫生习惯，并且在实践中应该努力做到；在合理的地方使用 `const` 是 C++ 语言最强大的特性之一，许多声称自己是“类型安全”的语言缺乏这一基本特性，没有它，正确性就难以实现。

`volatile` 关键字是 `const` 的对应词；因此，术语 *cv-qualifier* 指的是这两个术语。在标准中定义得相当不充分，`volatile` 有几种含义。

当应用于基本类型（例如，`volatile int`）时，它意味着它所指定的对象可能通过编译器所不知的方式访问，并且不一定从源代码中可见。因此，这个术语在编写设备驱动程序时非常有用，其中程序本身之外的动作（例如，按键的物理压力）可能会改变与对象关联的内存，或者当某些硬件或软件组件（在源代码之外）可以观察该对象状态的变化时。

非正式地说，如果源代码声明“*请读取那个 `volatile` 对象的值”，那么生成的代码应该读取那个值，即使程序看起来没有以任何方式修改它；同样，如果源代码声明“*请写入那个* `volatile` 对象”，那么应该向那个内存位置写入，即使程序看起来在随后的操作中没有从那个内存位置读取。因此，`volatile` 可以被视为一种防止编译器执行其本可以执行优化的机制。

在 C++ 的抽象机器中，访问 `volatile` 标记的对象相当于 I/O 操作的道德等价物——它可以改变程序的状态。对于某些类类型的对象，`volatile` 可以应用于成员函数，就像 `const` 一样。实际上，一个非 `static` 成员函数可以是 `const`、`volatile`、`const volatile` 或这些都不是（以及其他事项）。

在之前的描述中，关于在成员函数上应用`const`限定符的意义是通过`X::f`成员函数来阐述的——`*this`是`const`；在该函数中，其非`mutable`、非`static`的数据成员是`const`的，并且只有那些带有`const`限定符的成员函数才能通过`*this`来调用。同样，被`volatile`限定的非`static`成员函数也非常相似——在该函数执行期间，`*this`是`volatile`的，以及它的所有成员也都是`volatile`的，这会影响你可以对这些对象执行的操作。例如，取`volatile int`的地址会得到`volatile int*`，这不能隐式转换为`int*`，因为转换会丢失一些安全保证。这也是我们为什么有类型转换的原因之一。

# C++的类型转换

传统上，C++支持四种执行我们称为类型转换的显式类型转换方式——`static_cast`、`dynamic_cast`、`const_cast`和`reinterpret_cast`。C++11 添加了第五种，`duration_cast`，它与本书相关，但有时会出现在示例中，尤其是在我们测量函数执行时间时。最后，C++20 引入了第六种情况，`bit_cast`，这对于本书中的工作很有兴趣。

以下几节简要概述了每种 C++类型转换，并附带了一些示例，说明它们何时以及如何有用。

## 你最好的朋友（大多数时候）——`static_cast`

在我们的类型转换工具集中，`static_cast`是最好的、最有效的工具。它大多数情况下是安全的，基本上不花费任何成本，并且可以在`constexpr`上下文中使用，这使得它适合于编译时操作。

你可以在涉及潜在风险的情境中使用`static_cast`，例如将`int`转换为`float`或相反。在后一种情况下，它明确承认了小数部分的丢失。你还可以使用`static_cast`将指针或引用从派生类转换为它的直接或间接基类（只要没有歧义），这是完全安全的，也可以隐式地进行，以及从基类转换为它的派生类。使用`static_cast`从基类到派生类的转换效率很高，但如果转换不正确，风险极高，因为它不执行运行时检查。

下面有一些示例：

```cpp
struct B { virtual ~B() = default; /* ... */ };
struct D0 : B { /* ... */ };
struct D1 : B { /* ... */ };
class X {
public:
    X(int, double);
};
void f(D0&);
void f(D1*);
int main() {
    const float x = 3.14159f;
    int n = static_cast<int>(x); // Ok, no warning
    X x0{ 3, 3.5 }; // Ok
    // compiles, probably warns (narrowing conversion)
    X x1(3.5,0);
    // does not compile, narrowing not allowed with braces
    // X x2{ 3.5, 0 };
    X x3{ static_cast<int>(x), 3 }; // Ok
    D0 d0;
    // illegal, no base-derived relationship with D0 and D1
    // D1* d1 = static_cast<D1*>(&d0);
    // Ok, static_cast could be omitted
    B *b = static_cast<B*>(&d0);
    // f(*b); // illegal
    f(*static_cast<D0*>(b)); // Ok
    f(static_cast<D1*>(b)); // compiles but very dangerous!
}
```

特别注意前一个示例中`static_cast`的最后使用——从基类转换为其派生类之一是适当地使用`static_cast`完成的。然而，你必须确保转换会导致所选类型的对象，因为不会对转换的有效性进行运行时验证；正如其名称所暗示的，这个转换只进行编译时检查。如果你不确定如何使用向下转换，这不是你需要的工具。

`static_cast` 不仅改变编译器对表达式类型的看法；它还可以调整访问的内存地址，以考虑转换中涉及的类型。例如，当 `D` 类至少有两个非空的基类 `B0` 和 `B1` 时，这个派生类的这两个部分在 `D` 对象中的地址并不相同（如果它们是相同的，它们就会重叠！），所以从 `D*` 到其基类之一的 `static_cast` 可能会产生与 `D*` 本身不同的地址。我们将在讨论 `reinterpret_cast` 时回到这一点，对于 `reinterpret_cast`，其行为不同（且更危险）。

## 出现问题的迹象——`dynamic_cast`

有时会遇到这样的情况，你有一个指向某个类类型对象的指针或引用，而这个类型恰好与所需的类型不同（但相关）。这种情况经常发生——例如，在游戏引擎中，大多数类都从某个 `Component` 基类派生，函数通常接受 `Component*` 参数，但需要访问期望的派生类对象的成员。

这里的主要问题是，通常，函数的接口是错误的——它接受类型不足够精确的参数。尽管如此，我们都有软件要交付，有时，即使我们在过程中做出了我们可能希望以后重新审视的一些选择，我们也需要让事情工作。

进行此类转换的安全方法是 `dynamic_cast`。这种转换允许你将指针或引用从一个类型转换为另一个相关类型，以便你可以测试转换是否成功；对于指针，不正确的转换会产生 `nullptr`，而对于引用，不正确的转换会抛出 `std::bad_cast`。`dynamic_cast` 的类型相关性不仅限于基类派生关系，还包括在多重继承设计中从一个基类到另一个基类的转换。然而，请注意，在大多数情况下，`dynamic_cast` 要求要转换的表达式是具有至少一个 `virtual` 成员函数的多态类型。

这里有一些例子：

```cpp
struct B0 {
    virtual int f() const = 0;
    virtual ~B0() = default;
};
struct B1 {
    virtual int g() const = 0;
    virtual ~B1() = default;
};
class D0 : public B0 {
    public: int f() const override { return 3; }
};
class D1 : public B1 {
    public: int g() const override { return 4; }
};
class D : public D0, public D1 {};
int f(D *p) {
    return p? p->f() + p->g() : -1; // Ok
}
// g has the wrong interface: it accepts a D0& but
// tries to use it as a D1&, which makes sense if
// the referred object is publicly D0 and D1 (for
// example, class D
int g(D0 &d0) {
    D1 &d1 = dynamic_cast<D1&>(d0); // throws if wrong
    return d1.g();
}
#include <iostream>
int main() {
    D d;
    f(&d); // Ok
    g(d); // Ok, a D is a D0
    D0 d0;
    // calls f(nullptr) as &d0 does not point to a D
    std::cout << f(dynamic_cast<D*>(&d0)) << '\n'; // -1
    try {
      g(d0); // compiles but will throw bad_cast
    } catch(std::bad_cast&) {
      std::cerr << "Nice try\n";
    }
}
```

注意，尽管这个例子在抛出 `std::bad_cast` 时显示了一条消息，但这绝对不能称为异常处理；我们没有解决“问题”，代码执行在可能已损坏的状态下继续，这可能会在更严重的代码中使事情变得更糟。在这个玩具示例中，只是让代码失败并停止执行也是一个合理的选择。

在实践中，`dynamic_cast` 的使用应该是罕见的，因为它往往是我们以可完善的方式选择了函数接口的标志。请注意，`dynamic_cast` 需要编译时包含 **运行时类型信息**（**RTTI**），这会导致二进制文件更大。不出所料，由于这些成本，一些应用领域可能会避免使用这种转换，我们也会这样做。

## 玩弄安全性的把戏——`const_cast`

无论是 `static_cast` 还是 `dynamic_cast`（甚至包括 `reinterpret_cast`），都不能改变表达式的 cv-限定符；要实现这一点，你需要 `const_cast`。使用 `const_cast`，你可以从表达式中添加或移除 `const` 或 `volatile` 限定符。正如你可能已经猜到的，这仅在指针或引用上才有意义。

为什么你会做诸如从表达式中移除 `const` 限定符之类的事情呢？令人惊讶的是，有许多情况下这很有用，但一个常见的情况是允许在 `const` 限定符未适当使用的情况下使用 `const`-correct 类型——例如，未使用 `const` 的遗留代码，如下所示：

```cpp
#include <vector>
struct ResourceHandle { /* ... */ };
// this function observes a resource without modifying it,
// but the type system is not aware of that fact (the
// argument is not const)
void observe_resource(ResourceHandle*);
class ResourceManager {
    std::vector<ResourceHandle *> resources;
    // ...
public:
    // note: const member function
    void observe_resources() const {
      // we want to observe each resource, for example
      // to collect data
      for(const ResourceHandle * h : resources) {
       // does not compile, h is const
       // observe_resource(h);
      // temporarily dismiss constness
          observe_resource(const_cast<ResourceHandle*>(h));
      }
    }
    // ...
};
```

`const_cast` 是一个用于玩弄类型系统安全性的工具；它应在特定、受控的情况下使用，而不是做不合理的事情，比如改变数学常数（如 pi）的值。如果你尝试这样做，你将遇到 **未定义行为**（**UB**）——这是理所当然的。

## “相信我，编译器”—— reinterpret_cast

有时候，你只是要让编译器相信你。例如，知道在你的平台上 `sizeof(int)==4`，你可能想将 `int` 作为 `char[4]` 来与期望该类型的现有 API 进行交互。请注意，你应该确保这个属性成立（可能通过 `static_assert`），而不是依赖于所有平台上这个属性都成立（它并不成立）。

这就是 `reinterpret_cast` 给你的——将某种类型的指针转换为无关类型的指针的能力。这可以在我们看到的*第二章*中寻求利用指针互转换性的情况下使用，就像这也可以以几种相当危险且不便携的方式欺骗类型系统一样。

以从整数到四个字节的数组的上述转换为例——如果目的是为了便于对单个字节进行寻址，你必须意识到整数的字节序取决于平台，以及除非采取一些谨慎的措施，否则所编写的代码可能是不便携的。

此外，请注意，`reinterpret_cast` 只改变与表达式关联的类型——例如，它不会执行 `static_cast` 在多重继承情况下从派生类转换为基类时所做的轻微地址调整。

以下示例显示了这两种转换之间的区别：

```cpp
struct B0 { int n = 3; };
struct B1 { float f = 3.5f; };
// B0 is the first base subobject of D
class D : public B0, public B1 { };
int main() {
    D d;
    // b0 and &d point to the same address
    // b1 and &d do not point to the same address
    B0 *b0 = static_cast<B0*>(&d);
    B1 *b1 = static_cast<B1*>(&d);
    int n0 = b0->n; // Ok
    float f0 = b1->f; // Ok
    // r0 and &d point to the same address
    // r1 and &d also point to the same address... oops!
    B0 *r0 = reinterpret_cast<B0*>(&d); // fragile
    B1 *r1 = reinterpret_cast<B1*>(&d); // bad idea
    int nr0 = r0->n; // Ok but fragile
    float fr0 = r1->f; // UB
}
```

请谨慎使用 `reinterpret_cast`。相对安全的使用包括在给定足够宽的整型类型时将指针转换为整型表示（反之亦然），在转换不同类型的空指针之间，以及在函数指针类型之间进行转换——尽管在这种情况下，通过结果指针调用函数的结果是未定义的。如果您想了解更多，可以查看使用此转换可以执行的所有转换的完整列表，请参阅 [wg21.link/expr.reinterpret.cast](http://wg21.link/expr.reinterpret.cast)。

## 我知道位是正确的——`bit_cast`

C++20 引入了 `bit_cast`，这是一种新的转换，可以用来从一个对象复制位到另一个相同宽度的对象，在复制过程中开始目标对象（以及其中可能包含的对象）的生命周期，只要源和目标类型都是简单可复制的。这个有点神奇的库函数可以在 `<bit>` 头文件中找到，并且是 `constexpr` 的。

这里有一个例子：

```cpp
#include <bit>
struct A { int a; double b; };
struct B { unsigned int c; double d; };
int main() {
    constexpr A a{ 3, 3.5 }; // ok
    constexpr B b = std::bit_cast<B>(a); // Ok
    static_assert(a.a == b.c && a.b == b.d); // Ok
    static_assert((void*)&a != (void*)&b); // Ok
}
```

如此例所示，`A` 和 `B` 都是在编译时构建的，并且它们在位上是相同的，但它们的地址是不同的，因为它们是完全不同的对象。它们的数据成员部分是不同类型的，但大小相同，顺序相同，并且都是简单可复制的。

此外，请注意在此示例的最后一行使用了 C 风格的转换。正如我们很快将要讨论的，这是 C 风格转换的少数合理用途之一（我们也可以在这里使用 `static_cast`，它同样高效）。

## 有点不相关，但仍然——`duration_cast`

我们不会过多地讨论 `duration_cast`，因为它与我们感兴趣的主题只有间接关系，但既然它将是本书中微基准测试工具集的一部分，它至少值得提一下。

`duration_cast` 库函数可以在 `<chrono>` 头文件中找到，它是 `std::chrono` 命名空间的一部分。它是 `constexpr` 的，并且可以用来在表示不同测量单位的表达式之间进行转换。

例如，假设我们想要测量执行某个函数 `f()` 所花费的时间，使用我们库供应商提供的 `system_clock`。我们可以在调用 `f()` 之前和之后使用它的 `now()` 静态成员函数来读取那个时钟，这给了我们该时钟的两个 `time_point` 对象（两个时间点），然后计算它们之间的差异以获得该时钟的 `duration`。我们不知道用来表示该持续时间的测量单位是什么，但如果我们想以，比如说，`microseconds` 的形式使用它，我们使用 `duration_cast` 来执行那个转换：

```cpp
#include <chrono>
#include <iostream>
int f() { /* ... */ }
int main() {
    using std::cout;
    using namespace std::chrono;
    auto pre = system_clock::now();
    int res = f();
    auto post = system_clock::now();
    cout << "Computed " << res << " in "
        << duration_cast<microseconds>(post - pre);
}
```

我们将在本书的后面部分系统地介绍我们的基准测试实践，展示一种更正式的方式来衡量函数或代码块的执行时间，但 `duration_cast` 将成为我们选择用来确保我们展示结果格式的工具。

## 可恶的一个——C 转换

当需要类型转换时，你可能想使用 C 风格的转换，因为 C 语法出现在其他语言中，并且通常可以简洁地表达——`(T)expr`将表达式`expr`视为类型`T`。这种简洁性实际上是一个缺点，而不是优点，正如我们将看到的。在 C++代码中将 C 风格的转换限制在最小范围内：

+   当在源代码文本中执行自动搜索时，C 风格的转换更难找到，因为它们看起来像函数调用中的参数。由于转换是我们欺骗类型系统的方式，因此时不时地回顾使用它们的决定是值得的，因此能够找到它们是有价值的。相比之下，C++的转换是关键字，这使得它们更容易找到。

+   C 风格的转换不传达关于转换发生原因的信息。当编写`(T)expr`时，我们并没有说明我们是否想要更改 cv 限定符、导航类层次结构、仅更改指针类型，等等。特别是，当在指向不同类型的指针之间进行转换时，C 风格的转换通常表现得像`reinterpret_cast`，正如我们所看到的，在某些情况下可能会导致灾难性的结果。

你有时会在 C++代码中看到 C 风格的转换，大多数情况下是因为意图非常明确。我们在`bit_cast`部分的末尾看到了一个例子。另一个例子是消除编译器警告——例如，当调用一个标记为`[[nodiscard]]`的函数，但出于某种原因仍然想要丢弃结果时。

在另一个例子中，考虑以下泛型函数：

```cpp
template <class ItA, class ItB>
    bool all_equal(ItA bA, ItA eA, ItB bB, ItB eB) {
      for(; bA != eA && bB != eB; ++bA, (void) ++bB)
          if (*bA != bB)
            return false;
      return true;
    }
```

此函数遍历两个分别由`bA,eA)`和`[bB,eB)`（确保在处理完最短序列后立即停止）分隔的序列，比较这两个序列中“相同位置”的元素，并且只有在那些两个序列之间的所有元素比较都相等时才返回`true`。

注意，在这个代码中，将类型转换为`void`使用了 C 风格的转换，在`bA`和`bB`的增量之间进行转换，将`++bB`的结果转换为`void`。这看起来可能有些奇怪，但这是几乎任何人，包括敌对（或分心的）用户都可以在许多情况下使用的代码。假设有人决定在`operator++(ItA)`和`operator++(ItB)`的类型之间重载逗号运算符（是的，你可以这样做）。那个人就可以基本上劫持我们的函数来运行意外的代码。通过将其中一个参数转换为`void`，我们确保这是不可能的。

# 概述

这就结束了我们对 C++中转换和 cv 限定符的快速概述。现在我们已经看到了一些欺骗类型系统并陷入麻烦的方法，以及为什么我们应该谨慎（如果有的话）做这些事情的原因，我们可以开始用 C++构建美丽的事物，并朝着在编写正确程序以控制我们管理内存的尝试中实现安全、高效的抽象而努力。

在下一章中，我们将首先使用语言的一个定义性特征，即析构函数，来自动化我们代码处理资源的方式，特别是关注内存的处理方式。

# 第二部分：隐式内存管理技术

在这部分，我们将探讨一些在 C++中实现隐式资源管理（包括内存管理）的知名方法。这些都是你可以在日常编程实践中使用的技巧，它们将使你的程序比显式管理内存时更加简单和安全。可以说，这部分章节涉及人们所说的“现代”或“当代”C++。

本部分包含以下章节：

+   [*第四章*, *使用析构函数*

+   *第五章*, *使用标准智能指针*

+   *第六章*, *编写智能指针*
