# 2

# 需要小心的事情

因此，你决定阅读一本关于 C++内存管理的书，你愿意查看高级方法和技巧，就像你愿意“动手”一样，以便对内存管理过程有精细的控制。多么出色的计划！

由于你知道你将编写非常高级的代码，但也会编写非常底层的代码，有一些事情我们需要确保你意识到，这样你就不会陷入麻烦或编写看似工作但实际上并不工作（至少不是可移植的）的代码。

在本章中，我们将指出一些在本书中将发挥作用但你应该小心处理的 C++编程方面。这看起来可能像（非常）小的不良实践汇编或鼓励你陷入麻烦，但请将以下内容视为使用某些危险或棘手特性的好方法。你使用 C++，你有很大的表达自由，并且如果你了解并理解它们，你可以访问一些有用的特性。

我们希望代码干净高效，我们希望有责任感的程序员。让我们共同努力实现这个目标。

在本章中，我们将学习以下内容：

+   我们将涵盖一些可能导致麻烦的 C++代码的方式。确实，有些事情编译器无法可靠地诊断，就像有些事情 C++标准没有说明会发生什么一样，编写执行这些事情的代码是灾难的配方——至少是令人惊讶或不可移植的行为。

+   尤其是我们将探讨一个人如何因为指针而陷入麻烦。由于这本书讨论了内存管理，我们将经常使用指针和指针运算，能够区分适当的用法和不适当的用法将非常有价值。

+   最后，我们将讨论我们可以不使用类型转换（第三章的主要主题*）进行哪些类型转换，以及这与普遍看法相反，这种情况很少是好的主意。

我们的整体目标将是学习我们不应该做的事情（尽管有时我们也会做一些类似的操作），并在之后避免它们，希望理解我们这样做的原因。解决了这个问题之后，我们将有大量的章节来探讨我们应该做的事情，以及如何做好它们！

# 不同的邪恶类型

在深入研究需要谨慎处理的一些实际实践之前，看看如果我们的代码不遵守语言规则，我们可能会遇到的主要风险类别是很有趣的。每个这样的类别都伴随着一种我们应该努力避免的不愉快。

## 形式不当，无需诊断

C++中的一些结构被称为**不合法，无需诊断**（**IFNDR**）。确实，你会在标准中找到许多类似“如果[...], 程序是不合法的，无需诊断。”的表述。当某物是 IFNDR 时，意味着你的程序是有问题的。可能会发生一些不好的事情，但编译器不需要告诉你（实际上，有时编译器没有足够的信息来诊断问题情况）。

`alignas`)在不同的翻译单元（基本上是不同的源文件）中，或者有一个构造函数直接或间接地委托给自己。以下是一个示例：

```cpp
class X {
public:
    // #0 delegates to #1 which delegates to #0 which...
    X(float x) : X{ static_cast<int>(x) } { // #0
    }
    X(int n) : X{ n + 0.5f } { // #1
    }
};
int main() {}
```

注意，你的编译器可能会给出诊断信息；但这不是强制要求的。并不是编译器懒惰——在某些情况下，它们甚至可能无法提供诊断信息！因此，要小心不要编写导致 IFNDR（无需诊断）情况的代码。

## 不确定行为

我们在*第一章*中提到了**不确定行为**（**UB**）。UB 通常被视为 C++程序员头痛和痛苦的原因，但它指的是 C++标准没有要求的任何行为。在实践中，这意味着如果你编写的代码包含 UB，你不知道运行时会发生什么（至少如果你希望代码具有一定的可移植性）。UB 的典型例子包括解引用空指针或未初始化的指针：这样做会让你陷入严重的麻烦。

对于编译器来说，UB 不应该发生（毕竟，尊重语言规则的代码不包含 UB）。因此，编译器会“围绕”包含 UB 的代码进行优化，有时会产生令人惊讶的效果：它们可能会开始移除测试和分支、优化循环等。

UB 的影响往往局限于局部。例如，在以下示例中，有一个测试确保在使用`*p`之前`p`不是空指针，但至少有一个对`*p`的访问是没有检查的。这段代码是有问题的（未检查的`*p`访问是 UB），因此编译器允许以这种方式重写它，从而有效地移除所有验证`p`不是空指针的测试。毕竟，如果`p`是`nullptr`，那么损害已经造成，因此编译器有权利假设程序员传递了一个非空指针给函数！

```cpp
int g(int);
int f(int *p) {
    if(p != nullptr)
        return g(*p); // Ok, we know p is not null
    return *p; // oops, if p == nullptr this is UB
}
```

在这种情况下，编译器可以合法地将整个`f()`函数体重写为`return g(*p)`，将`return *p`语句转换为不可达代码。

语言中存在潜在的不确定行为（UB）的多个地方，包括有符号整数溢出、访问数组越界、数据竞争等。目前有持续的努力在减少潜在 UB 案例的数量（甚至有一个专门致力于此的**SG12**研究小组），但 UB 可能在未来一段时间内仍然是语言的一部分，我们需要对此有所警觉。

## 实现定义的行为

标准中的一些部分属于**实现定义的行为**范畴，或者说是你可以依赖特定平台的行为。这种行为是你选择的平台应该记录的，但并不保证可以移植到其他平台。

实现定义的行为出现在许多情况下，包括如下事物：实现定义的限制，例如最大嵌套括号数；switch 语句中的最大 case 标签数；对象的实际大小；`constexpr`函数中的最大递归调用数；字节中的位数；等等。其他已知的实现定义行为案例包括`int`对象中的字节数或`char`类型是有符号还是无符号整型。

实现定义的行为本身并不是邪恶的源头，但如果追求可移植代码但依赖于一些不可移植的假设，则可能会出现问题。有时，当假设可以在编译时或类似的潜在运行时机制中验证时，通过`static_assert`在代码中表达这些假设是有用的，以便在为时已晚之前意识到这些假设对于特定目标平台是错误的。

例如：

```cpp
int main() {
    // our code supposes int is four bytes wide, a non-
    // portable assumption
    static_assert(sizeof(int)==4);
    // only compiles if condition is true...
}
```

除非你确信你的代码永远不会需要移植到另一个平台，否则应尽可能少地依赖实现定义的行为，并且如果确实需要，确保通过`static_assert`（如果可能的话）或运行时（如果没有其他选择）验证并记录这种情况。这可能会帮助你避免未来的一些令人不快的惊喜。

## 未指定行为（未记录）

当实现定义的行为在特定平台上不可移植但有文档记录时，未指定行为是指即使对于给定正确数据的良好格式程序，其行为也依赖于实现但不需要记录的行为。

一些未指定行为的案例包括已移动对象的状体（例如，`f(g(),h())`将首先评估`g()`或`h()`，新分配内存块中的值等）。这个后者的例子对我们研究很有趣；调试构建可能会用可识别的位模式填充新分配的内存块以帮助调试过程，而使用相同工具集的优化构建可能会留下新分配内存块初始位的“未初始化”，保留分配时的位，以获得速度提升。

## ODR

ODR（One Definition Rule，单一定义规则）简单来说，就是在一个翻译单元中，每个“事物”（函数、作用域中的对象、枚举、模板等）只能有一个定义，尽管可以有多个声明。

```cpp
int f(int); // declaration
int f(int n); // Ok, declaration again
int f(int m) { return m; } // Ok, definition
// int f(int) { return 3; } // not Ok (ODR violation)
```

在 C++ 中，避免 ODR 违反很重要，因为这些“邪恶”可以逃过编译器的审查，落入 IFNDR 情境。例如，由于源文件的独立编译，包含非 `inline` 函数定义的头文件会导致该定义在每个包含该头文件的源文件中重复。然后，每次编译可能都会成功，而同一构建中该函数存在多个定义的事实可能在稍后（在链接时）被发现，或者根本未被检测到，从而造成混乱。

## 错误行为

C++ 中持续进行的与安全相关的工作导致了对一种新类型的“邪恶”的讨论，这种类型暂时被命名为 *错误行为*。这个新类别旨在涵盖过去可能被视为未定义行为（UB）的情况，但对于这些情况，我们可以提供诊断并定义良好的行为。这种行为仍然是不正确的，但错误行为在某种程度上为后果提供了边界。请注意，截至本文撰写时，错误行为的这项工作仍在进行中，这个新的措辞功能可能针对 C++26。

错误行为的预期用例之一是从未初始化的变量中读取，实现（出于安全原因）可以为读取的位提供固定值，从读取该变量产生的概念性错误是实施者鼓励诊断的东西。另一个用例是忘记从非 void 赋值运算符返回值。

现在我们已经探讨了如果不行为可能会影响我们程序的许多“不愉快”的“家族”，让我们深入研究一些可能会让我们陷入麻烦的主要设施，并看看我们应该避免做什么。

# 指针

*第一章* 讨论了 C++ 中指针的概念及其所代表的意义。它描述了指针算术是什么，以及它允许我们做什么。现在，我们将探讨指针算术的实际应用，包括这个低级（但有时宝贵）工具的恰当和不恰当使用。

## 在数组中使用指针算术

指针算术是一个既好又实用的工具，但它是一把锋利的工具，往往被误用。对于原始数组，以下两个标记为 `A` 和 `B` 的循环的行为完全相同：

```cpp
void f(int);
int main() {
    int vals[]{ 2,3,5,7,11 };
    enum { N = sizeof vals / sizeof vals[0] };
    for(int i = 0; i != N; ++i) // A
      f(vals[i]);
    for(int *p = vals; p != vals + N; ++p) // B
      f(*p);
}
```

你可能会对循环 `B` 中的 `vals + N` 部分感到好奇，但它是有效的（并且是惯用的）C++ 代码。你可以观察到数组末尾之后的指针，尽管你不允许观察它指向的内容；标准保证这个特定的一个超出末尾的地址对你的程序是可访问的。然而，对于下一个地址，没有这样的保证，所以请小心！

只要你遵守规则，你就可以使用指针在数组内部跳来跳去。如果你超出了范围，并使用指针超出数组末尾一个位置，你将进入 UB 区域；也就是说，你可能会尝试访问不在你的进程地址空间中的地址：

```cpp
int arr[10]{ }; // all elements initialized to zero
int *p = &arr[3];
p += 4; assert(p == &arr[7]);
--p;    assert(p == &arr[6]);
p += 4; // still Ok as long as you don't try to access *p
++p; // UB, not guaranteed to be valid
```

## 指针可转换性

C++标准定义了对象如何进行`reinterpret_cast`（我们将在*第三章*中详细说明），因为它们具有相同的地址。广义上，以下几点是正确的：

+   一个对象与其自身是可指针转换的

+   一个`union`与其数据成员是可指针转换的，如果它们是复合类型，则还包括其第一个数据成员

+   在某些限制下，如果`x`是一个对象而`y`是那个对象的第一个非静态数据成员的类型，那么`x`和`y`是可指针转换的

这里包含了一些示例：

```cpp
struct X { int n; };
struct Y : X {};
union U { X x; short s; };
int main() {
    X x;
    Y y;
    U u;
    // x is pointer-interconvertible with x
    // u is pointer-interconvertible with u.x
    // u is pointer-interconvertible with u.s
    // y is pointer-interconvertible with y.x
}
```

如果你尝试以不尊重指针可转换性规则的方式应用`reinterpret_cast`，你的代码在技术上是不正确的，并且在实践中不一定能保证工作。不要这样做。

我们将在代码示例中偶尔使用指针可转换性属性，包括在下一节中。

## 在对象内使用指针算术的应用

在 C++中，对象内的指针算术也是允许的，尽管人们应该小心处理这一点（使用适当的类型转换，我们将在*第三章*中探讨，并确保适当地执行指针算术）。

例如，以下代码是正确的，尽管这不是人们应该追求的事情（这没有意义，它以不必要的复杂方式做事，但它是合法的，并且不会造成伤害）：

```cpp
struct A {
    int a;
    short s;
};
short * f(A &a) {
    // pointer interconvertibility in action!
    int *p = reinterpret_cast<int*>(&a);
    p++;
    return reinterpret_cast<short*>(p); // Ok, within the
                                       // same object
}
int main() {
    A a;
    short *p = f(a);
    *p = 3; // fine, technically
}
```

我们不会在本书中滥用 C++语言的这一方面，但我们需要意识到它，以便编写正确、低级别的代码。

关于指针和地址的区别

为了加强硬件和软件安全，人们已经在可以提供“指针标记”形式的硬件架构上进行了工作，这允许硬件跟踪指针来源，以及其他方面。两个著名的例子是 CHERI 架构([`packt.link/cJeLo`](https://packt.link/cJeLo))和**内存标记扩展**(**MTEs**)（Linux: [`packt.link/KXeRn`](https://packt.link/KXeRn) | Android: [`packt.link/JDfEo`](https://packt.link/JDfEo), 和 [`packt.link/fQM2T`](https://packt.link/fQM2T)| Windows: [`packt.link/DgSaH`](https://packt.link/DgSaH))）。

为了利用这样的硬件，语言需要区分地址的低级概念和指针的高级概念，因为后者需要考虑到指针不仅仅是内存位置。如果你的代码绝对需要比较无关的指针以确定顺序，你可以做的一件事是将指针转换为`std::intptr_t`或`std::uintptr_t`，然后比较（数值）结果而不是比较实际的指针。请注意，编译器对这两种类型的支持是可选的，尽管所有主要的编译器供应商都提供了它。

### 空指针

空指针作为指向无效位置的指针的可识别值的想法可以追溯到 C.A.R. Hoare ([`packt.link/ByfeX`](https://packt.link/ByfeX))。在 C 语言中，通过`NULL`宏，它最初被表示为一个值为`0`的`char*`，然后是一个值为`0`的`void*`，然后在 C++中，由于像`int *p = NULL;`这样的带有类型`NULL`的语句在 C 中是合法的，但在 C++中不是，所以它简单地表示值为`0`。这是因为 C++的类型系统更加严格。请注意，值为`0`的指针并不意味着“指向地址零”，因为这个地址本身是完全有效的，并且在许多平台上被这样使用。

在 C++中，表达空指针的首选方式是`nullptr`，这是一个`std::nullptr_t`类型的对象，它可以转换为任何类型的指针，并按预期行为。这解决了 C++中一些长期存在的问题，如下所示：

```cpp
int f(int); //#0
int f(char*); // #1
int main() {
    int n = 3;
    char c;
    f(n); // calls #0
    f(&c); // calls #1
    f(0); // ambiguous before C++11, calls #0 since
    f(nullptr); // only since C++11; unambiguously calls #1
}
```

注意，`nullptr`不是一个指针；它是一个可以隐式转换为指针的对象。因此，`std::is_pointer_v<nullptr>`特性是假的，C++提供了一个名为`std::is_null_pointer<T>`的独立特性，用于静态测试`T`是否是`std::nullptr_t`（考虑`const`和`volatile`）。

解引用空指针是未定义的行为，就像解引用未初始化的指针一样。在代码中使用`nullptr`的目的就是为了使这种状态可识别：`nullptr`是一个可区分的值，而未初始化的指针可能什么都是。

在 C++中（与 C 不同），对空指针进行算术运算是有明确定义的……只要你在空指针上加上零。或者，换一种说法：如果你在空指针上加上零，代码仍然是有定义的，但如果你加上任何其他东西，那就得你自己负责了。在 wg21.link/c++draft/expr.add#4.1 中有一个明确的规定。这意味着以下情况是正确的，就像空`数组`的情况一样，`begin()`返回`nullptr`，`size()`返回零，所以`end()`实际上计算的是`nullptr+0`，这符合规则：

```cpp
template <class T> class Array {
    T *elems = nullptr; // pointer to the beginning
    std::size_t nelems = 0; // number of elements
public:
    Array() = default; // =empty array
    // ...
    auto size() const noexcept { return nelems; }
    // note: could return nullptr
    auto begin() noexcept { return elems; }
    auto end() noexcept { return begin() + size(); }
};
```

我们将在第十二章、第十三章和第十四章中更详细地回到这个`数组`示例；这将帮助我们讨论高效内存管理技术的一些重要方面。现在，让我们看看另一个危险的编程操作来源。

# 类型转换

C++程序员可能陷入麻烦的另一个领域是**类型欺骗**。通过类型欺骗，我们指的是在一定程度上颠覆语言类型系统的技术。执行类型转换的圣洁工具是类型转换，因为它们在源代码文本中是显式的，并且（除 C 风格类型转换外）表达了转换的意图，但这个主题值得单独成章（*第三章*，如果你想知道的话）。

在本节中，我们将探讨其他实现类型欺骗的方法，包括可推荐的方法和应避免的方法。

## 通过联合成员进行类型欺骗

联合是一种成员都位于同一地址的类型。联合的大小是其最大成员的大小，联合的对齐是其成员的最严格对齐。

考虑以下示例：

```cpp
struct X {
    char c[5]; short s;
} x;
// one byte of padding between x.c and x.s
static_assert(sizeof x.s == 2 && sizeof x == 8);
static_assert(alignof(x) == alignof(short));
union U {
    int n; X x;
} u;
static_assert(sizeof u == sizeof u.x);
static_assert(alignof(u) == alignof(u.n));
int main() {}
```

很容易想到，可以使用`union`隐式地将诸如四字节的浮点数转换为四字节的整数，在 C 语言（而不是 C++）中，这确实是可能的。

尽管广泛认为这种做法在 C++中是合法的，但实际情况并非如此（有一个特殊的注意事项，我们将在稍后探讨）。实际上，在 C++中，已写入的联合的最后一个成员被称为联合的`constexpr`函数：

```cpp
union U {
    float f;
    int n;
};
constexpr int f() {
    U u{ 1.5f };
    return u.n; // UB (u.f is the active member)
}
int main() {
    // constexpr auto r0 = f(); // would not compile
    auto r1 = f(); // compiles, as not a constexpr
                  // context, but still UB
}
```

如你所知，在先前的示例中，像`f()`这样的`constexpr`函数不能包含在`constexpr`上下文中调用时会导致未定义行为的代码。这有时使其成为一个有趣的表达观点的工具。

在`union`成员之间的转换方面存在一个注意事项，这个注意事项与公共初始序列有关。

### 公共初始序列

如在 wg21.link/class.mem.general#23 中解释的那样，`A`和`B`由它们的前两个成员组成（`int`与`const int`布局兼容，`float`与`volatile float`布局兼容）：

```cpp
struct A { int n; float f; char c; };
struct B{ const int b0; volatile float x; };
```

如果读取的值是成员的公共初始序列和活动成员的一部分，则可以使用`union`从非活动成员中读取。以下是一个示例：

```cpp
struct A { int n0; char c0; };
struct B { int n1; char c1; float x; };
union U {
    A a;
    B b;
};
int f() {
    U u{ { 1, '2' } }; // initializes u.a
    return u.b.n1; // not UB
}
int main() {
    return f(); // Ok
}
```

注意，这种类型欺骗应尽量减少，因为它可能会使推理源代码变得更加困难，但它非常有用。例如，它可以用来实现一些有趣的底层表示，这些表示对于可以有两个不同表示的类（例如`optional`或`string`）来说是有用的，这使得从一个切换到另一个变得更加容易。可以基于此构建一些有用的优化。

## intptr_t 和 uintptr_t 类型

如本章前面所述，在 C++中，无法以定义良好的方式直接比较指向内存中任意位置的指针。然而，可以以定义良好的方式比较与指针相关联的整数值，如下所示：

```cpp
#include <iostream>
#include <cstdint>
int main() {
    using namespace std;
    int m,
       n;
    // simply comparing &m with &n is not allowed
    if(reinterpret_cast<intptr_t>(&m) <
      reinterpret_cast<intptr_t>(&n))
      cout << "m precedes n in address order\n";
    else
      cout << "n precedes m in address order\n";
}
```

`std::intptr_t` 和 `std::uintptr_t` 类型是足够大的整数类型的别名，可以容纳地址。对于可能导致负值操作（例如，减法）的情况，请使用有符号类型 `intptr_t`。

## `std::memcpy()` 函数

由于历史（和与 C 的兼容性）原因，`std::memcpy()` 是特殊的，因为它如果使用得当可以启动对象的生命周期。对 `std::memcpy()` 的错误使用进行类型转换可能如下所示：

```cpp
// suppose this holds for this example
static_assert(sizeof(int) == sizeof(float));
#include <cassert>
#include <cstdlib>
#include <cstring>
int main() {
    float f = 1.5f;
    void *p = malloc(sizeof f);
    assert(p);
    int *q = std::memcpy(p, &f, sizeof f);
    int value = *q; // UB
    //
}
```

这之所以非法，是因为对 `std::memcpy()` 的调用将一个 `float` 对象复制到由 `p` 指向的存储中，实际上是在那个存储中启动了一个 `float` 对象的生命周期。由于 `q` 是一个 `int*`，解引用它是未定义行为（UB）。

另一方面，以下操作是合法的，展示了如何使用 `std::memcpy()` 进行类型转换：

```cpp
// suppose this holds for this example
static_assert(sizeof(int) == sizeof(float));
#include <cassert>
#include <cstring>
int main() {
    float f = 1.5f;
    int value;
    std::memcpy(&value, &f, sizeof f); // Ok
    // ...
}
```

的确，在这个第二个例子中，使用 `std::memcpy()` 从 `f` 复制位到 `value` 启动了 `value` 的生命周期。从那时起，该对象可以像任何其他 `int` 一样使用。

## `char*`、`unsigned char*` 和 `std::byte*` 的特殊情况

`char*`、`unsigned char*`（不是 `signed char*`）和 `std::byte*` 类型在 C++ 中具有特殊地位，因为它们可以指向任何地方并代表任何类型（[wg21.link/basic.lval#11](http://wg21.link/basic.lval#11)）。因此，如果您需要访问对象的值表示形式下的底层字节，这些类型是您工具箱中的重要工具。

在本书的后续内容中，我们偶尔会使用这些类型来执行低级字节操作。请注意，此类操作本质上是脆弱且不可移植的，因为整数中字节的顺序可能会因平台而异。请谨慎使用此类低级设施。

## `std::start_lifetime_as<T>()` 函数

本章最后介绍的一组设施是 `std::start_lifetime_as<T>()` 和 `std::start_lifetime_as_array<T>()`。这些函数讨论了多年，但直到 C++23 才真正发挥其作用。它们的作用是将原始内存字节数组作为参数，并返回一个指向 `T` 的指针（指向该缓冲区），其生命周期已开始，从而可以从该点开始将指针所指的内容用作 `T` 类型：

```cpp
static_assert(sizeof(short) == 2);
#include <memory>
int main() {
    char buf[]{ 0x00, 0x01, 0x02, 0x03 };
    short* p = std::start_lifetime_as<short>(buf);
    // use *p as a short
}
```

这同样是一个需要谨慎使用的低级特性。这里的意图是能够用纯 C++ 实现诸如低级文件 I/O 和网络代码（例如，接收 UDP 数据包并将其值表示形式视为现有对象）等，而不会陷入未定义行为的陷阱。我们将在*第十五章*中更详细地讨论这些函数。

# 摘要

本章探讨了我们将有时使用的一些低级和有时令人不快的设施，目的是设置适当的“警告标志”，并提醒我们必须负责任地编写合理且正确的代码，尽管我们选择的语言提供了很大的自由度。

当在本书的后续章节中编写高级内存管理功能时，这些危险的设施有时对我们是有用的。受到本章关于需要注意的事项的内容的启发，我们将谨慎、小心地使用这些设施，并使其难以被误用。

在我们接下来的章节中，我们将探讨置于我们手中的关键 C++类型转换；目的是让我们了解每种转换的作用，以及何时（以及为了什么目的）应该使用它，这样我们就可以构建我们想要使用的强大内存管理抽象。
