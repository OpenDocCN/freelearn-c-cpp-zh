# 9

# 使用强类型提高类型安全

C++是一种静态类型语言，这意味着每个表达式在编译时都会被分配一个类型，要么是由开发者（在大多数情况下），要么是在使用关键字 auto 时由编译器推导出来。尽管如此，这并不意味着它是一个类型安全的语言。

C++和 C 都允许具有可变数量的参数的函数（`va_arg`），或变长函数和类型转换，并支持隐式类型转换。与 C++和 C 的性能相关的这些底层功能往往是程序中 bug 的来源。在本章中，我们将介绍用于在 C++中提高类型安全性的良好实践。

**类型安全**是安全关键系统程序的一个重要方面。这就是为什么像 MISRA 和 AUTOSAR 这样的组织提供的编码标准会限制使用违反类型安全的特性。在本章中，我们将涵盖以下主要内容：

+   隐式转换

+   显式转换

+   强类型

# 技术要求

为了充分利用本章内容，我强烈建议你在阅读示例时使用编译器探索器([`godbolt.org/`](https://godbolt.org/))。将 GCC 作为你的 x86 架构的编译器。这将允许你看到标准输出（stdio）结果，更好地观察代码的行为。由于我们使用了大量的现代 C++特性，请确保在编译器选项框中添加`-std=c++23`以选择 C++23 标准。

编译器探索器使得尝试代码、调整代码并立即看到它如何影响输出和生成的汇编变得容易。大多数示例也可以在 Arm Cortex-M0 目标的 Renode 模拟器上运行，并在 GitHub 上提供([`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter09`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter09)))。

# 隐式转换

当你调用一个期望整数参数的函数，但你传递了一个浮点数作为参数时，编译器会愉快地编译程序。同样，如果你将整数数组传递给一个期望整数指针的函数，程序也会编译。这些场景在 C 和 C++中都变得如此正常，以至于它们通常被默认接受，而不考虑编译过程中的实际情况。

在这两种描述的场景中，编译器正在进行隐式转换。在第一种场景中，它将浮点数转换为整数，在第二种场景中，它传递数组的第一个元素的指针，这个过程被称为**数组到指针退化**。

虽然隐式转换使代码更简洁、更容易编写，但它们也打开了与类型安全相关的一系列问题的门。将浮点数转换为整数会导致精度损失，而假设数组总是像指针一样行为可能导致对数组边界的错误解释，这可能导致缓冲区溢出或其他内存问题。

隐式转换在以下情况下执行：

+   当一个函数以与参数类型不同的类型调用时。例如：

    ```cpp
    #include <cstdio>
    void print_int(int value) {
        printf("value = %d\n", value);
    }
    int main() {
        float pi = 3.14f;
     // int implicitly converts to float
     print_int(pi);
     return 0;
    } 
    ```

+   当返回语句中指定的值类型与函数声明中指定的类型不同时。例如：

    ```cpp
    int get_int() {
        float pi = 3.14;
     // float implicitly converts to int
     return pi;
    } 
    ```

+   在具有不同算术类型操作数的二元运算符表达式中。例如：

    ```cpp
    #include <cstdio>
    int main() {
        int int_value = 5;
        float float_value = 4.2;
     // int converts to float
     auto result = int_value + float_value;
     printf("result = %f\n", result);

     return 0;
    } 
    ```

+   在将整数类型作为`switch`语句的目标时。例如：

    ```cpp
    char input = 'B';
    // implicit conversion from char to int
    switch (input) {
     case 65:
     printf("Input is 'A'\n");
     break;
     case 66:
     printf("Input is 'B'\n");
     break;
     default:
     printf("Unknown input");
    } 
    ```

+   在`if`语句中，类型可以被转换为`bool`类型。例如：

    ```cpp
    #include <cstdio>
    int main() {
        int int_value = 10;
     // int implicitly converts to bool
     if (int_value) {
     printf("true\n");
     }
     return 0;
    } 
    ```

编译器处理不同类型的隐式转换，其中一些最重要的转换包括：

+   数字提升和转换

+   数组到指针转换

+   函数到指针转换

接下来，我们将通过示例讨论上述隐式转换。

## 数字提升和转换

算术类型可以被提升或转换为其他算术类型。类型提升不会改变值或丢失精度。`std::uint8_t`可以被提升为`int`，或者`float`可以被提升为`double`。如果一个类型可以被完全转换为目标类型，而不丢失精度，那么它正在进行提升。

算术运算符不接受小于`int`的类型。当作为算术运算符的操作数传递时，算术类型可以被提升。根据它们的类型，整数和浮点数的提升有一些特定的规则：

+   布尔提升：如果`bool`类型设置为`false`，则提升为`int`类型，值为`0`；如果设置为`true`，则提升为`int`类型，值为`1`。

+   其他整型类型，包括位域，将被转换为以下列表中最小的类型，该类型可以表示转换类型的所有值：

    +   `int`

    +   `unsigned int`

    +   `long`

    +   `unsigned long`

    +   `long long`

    +   `unsigned long long`

+   `float`类型可以被提升为`double`类型。

为了更好地理解整数提升规则，我们将通过下一个例子进行说明：

```cpp
#include <cstdint>
#include <type_traits>
int main() {
    std::uint8_t a = 1;
    std::uint16_t b = 42;
    auto res1 = a + b;
    static_assert(std::is_same_v<int, decltype(res1)>);
    return 0;
} 
```

在上述例子中，我们添加了`uint8_t`和`uint16_t`。根据提升规则，这两种类型都将提升为`int`，因为它们可以被`int`完全表示。加法的结果存储在变量`res1`中，该变量被声明为`auto`，这意味着编译器将推导其类型。我们期望它是一个`int`，我们使用`static_assert`和`std::is_same_v`来验证这一点。

在这个例子中，两种类型都被提升到了相同的类型。如果我们提升后有不同的类型，那么它们将根据**常规算术转换规则**转换为一种公共类型。

常规算术转换的目标是将类型转换为一种公共类型，这同时也是结果类型。常规算术转换有一些规则：

+   如果两种类型都是有符号或无符号整数，那么公共类型是具有更高整数转换等级的类型。等级按降序排列如下（无符号整数的等级对应于匹配的有符号整数的等级）：

    +   `long long`

    +   `long`

    +   `int`

    +   `short`

    +   `signed char`

+   如果其中一个类型是有符号整数，另一个是无符号整数，则适用以下规则：

    +   如果无符号类型的整数转换等级大于或等于有符号类型，则公共类型是无符号类型。

    +   否则，如果有符号类型可以表示无符号类型的所有值，则公共类型是有符号类型。

    +   否则，公共类型是有符号整数的无符号整数类型。

+   如果其中一个类型是浮点类型，另一个是整数，则整数转换为该浮点类型。

+   如果两种类型都是浮点类型但浮点转换等级不同，则将转换等级较低的类型转换为另一个类型。浮点转换等级按降序排列如下：

    +   `long double`

    +   `double`

    +   `float`

让我们通过以下示例来更好地理解常规算术转换的规则：

```cpp
#include <type_traits>
int main() {
    struct bitfield{
        long long a:31;
    };
    bitfield b {4};
    int c = 1;
    auto res1 = b.a + c;  
    static_assert(sizeof(int) == 4);
    static_assert(sizeof(long long) == 8);
    static_assert(std::is_same_v<int, decltype(res1)>);
    long e = 5;
    auto res2 = e - b.a;
    static_assert(std::is_same_v<long, decltype(res2)>);
    return 0;
} 
```

在上述示例中，我们有一个 31 位的 `bitfield`，其底层类型为 `long long`。我们首先将 `b.a` 和类型为 `int` 的变量 c 相加。如果我们在一个 `int` 的大小为 4 字节的平台上，位字段将被提升为 `int`，尽管底层类型 `long long` 的大小为 8 字节。提升后的位字段将加到 `int` 类型的 `c` 上，因此这个操作的最终结果也将是 `int`，我们可以通过使用 `std::is_same_v` 检查 res1 的类型来验证这一点。

在示例的第二部分，我们从 `long` 类型的 `e` 中减去位字段。在这种情况下，位字段首先提升为 `int`；然后，根据常规算术转换的规则，它被转换为 `long`，这意味着结果类型也将是 `long`。

你可以从本书的 GitHub 仓库运行上述示例。它位于 `Chapter09/type_safety` 目录下，你可以使用以下命令构建和运行它：

```cpp
$ cmake -B build -DMAIN_CPP_FILE_NAME="main_usual_arithmetic_conversion.cpp"
$ cmake --build build --target run_in_renode 
```

程序成功构建的事实就足以确认常规算术转换的结果，因为我们使用了 `static_assert` 来验证它。

现在，让我们看看一个可能令人惊讶的结果的示例：

```cpp
#include <cstdio>
int main() {
    int a = -4;
    unsigned int b = 3;
    if(a + b > 0) {
        printf("%d + % u is greater than 0\r\n", a, b);
    }
    return 0;
} 
```

如果你运行这个示例，`if` 子句中的表达式将评估为真。根据常规算术转换的规则，有符号的 `int` a 将被转换为 `unsigned int`，这意味着表达式 `a + b` 确实大于 `0`。在算术表达式中混合无符号和有符号类型可能会由于隐式转换而导致不期望的行为和潜在的错误。

我们可以使用 GCC 的编译器标志 `–Wconversion` 和 `-Wsign-conversion` 来使其在隐式转换可能改变值和符号时发出警告。然而，在算术表达式中混合有符号和无符号类型应该避免，因为这可能导致错误的结果。

接下来，我们将讨论数组到指针转换及其影响。

## 数组到指针转换

一个数组可以被隐式转换为指针。生成的指针指向数组的第一个元素。许多在数据数组上工作的 C 和 C++函数都是设计有指针和大小参数的。这些接口基于合同设计。合同如下：

+   调用者将传递一个指向数组第一个元素的指针

+   调用者将传递数组的大小

这是一个简单的约定，但没有办法强制执行。让我们看看以下简单的例子：

```cpp
#include <cstdio> 
void print_ints(int * arr, std::size_t len) {
    for(std::size_t i = 0; i < len; i++) {
        printf("%d\r\n", arr[i]);
    }
}
 int main() { 
    int array_ints[3] = {1, 2, 3};
    print_ints(array_ints, 3);
    return 0; 
} 
```

在上面的例子中，我们有`print_ints`函数，它有一个指向`int`的指针`arr`和`len`，一个`std::size_t`参数。在`main`函数中，我们通过传递`array_ints`，一个包含 3 个整数的数组，以及`3`作为参数来调用`print_ints`函数。数组`array_ints`将被隐式转换为指向其第一个元素的指针。`print_ints`函数有几个潜在问题：

+   它期望我们传递给它的指针是有效的。它不会验证这一点。

+   它期望它接收的`len`参数是它操作的数组的实际大小。调用者可能传递一个可能导致越界访问的大小。

+   由于它直接操作指针，如果在函数中使用指针算术，总有可能发生越界访问。

为了消除这些潜在问题，在 C++中，我们不是使用指向数据数组的指针来工作，而是可以使用类模板`std::span`。它是一个连续对象序列的包装器，序列的第一个元素位于位置零。它可以由 C 风格数组构造，它有`size`方法，我们可以在它上面使用基于范围的`for`循环。让我们用`std::span`而不是指针重写之前的例子：

```cpp
#include <cstdio> 
#include <span>
void print_ints(const std::span<int> arr) {
    for(int elem: arr) {
        printf("%d\r\n", elem);
    }
}
int main() { 
    int arr[3] = {1, 2, 3};
    print_ints(arr);
    return 0; 
} 
```

在上面的例子中，我们可以看到函数`print_ints`现在看起来简单多了。它接受整数的`std::span`，并使用基于范围的 for 循环遍历元素。在调用位置，我们现在只需传递`arr`，一个包含 3 个整数的数组。它会被隐式转换为`std::span`。

类模板`std::span`也有`size`方法、操作符`[]`以及`begin`和`end`迭代器，这意味着我们可以将其用于标准库算法。我们还可以从`span`构造子 span。它可以由 C 风格数组构造，也可以由容器如`std::array`和`std::vector`构造。它是解决通常依赖于指针和大小参数的接口潜在问题的绝佳解决方案。

## 函数到指针的转换

一个函数可以被隐式转换为该函数的指针。以下示例演示了这一点：

```cpp
#include <cstdio> 
#include <type_traits>
void print_hello() {
    printf("Hello!\r\n");
}
int main() { 
    void(*fptr)() = print_hello;
    fptr();
    fptr = &print_hello;
    (*fptr)();
    static_assert(std::is_same_v<decltype(fptr), void(*)()>);
    static_assert(std::is_same_v<decltype(print_hello), void()>);
    return 0; 
} 
```

在上述示例中，我们将函数 `print_hello` 赋值给函数指针 `fptr`。在 C++ 中，我们不需要使用地址运算符与函数名一起使用来将其赋值给函数指针。同样，我们不需要取消引用函数指针来通过它调用函数。尽管如此，`print_hello` 和 `fptr` 是两种不同的类型，我们使用 `static_assert` 和 `is_same` 类型特性来确认这一点。

C++ 中的隐式转换使编写代码更加容易。有时它们可能导致不期望的行为和程序中的潜在问题。为了减轻这些担忧，我们可以在需要时显式转换类型。

接下来，我们将介绍显式转换。

# 显式转换

C++ 支持使用 C 风格的类型转换显式转换，也支持函数式类型转换和以下类型转换运算符：

+   `const_cast`

+   `static_cast`

+   `dynamic_cast`

+   `reinterpret_cast`

我们将介绍类型转换运算符，从 `const_cast` 开始。

## const_cast

`const_cast` 用于移除 const 属性以与非 const 正确函数一起工作。我们将通过以下示例来更好地理解它：

```cpp
#include <cstdio>
void print_num(int & num) {
    printf("num is %d\r\n", num);
}
int main() {
    const int num = 42;
    print_num(const_cast<int&>(num));
    int & num_ref = const_cast<int&>(num);
    num_ref = 16;
    return num;
} 
```

在上述示例中，我们使用了 `const_cast` 在两种不同的场景中。我们首先使用它来从 `const int num` 中移除 const 属性，以便将其传递给 `print_num` 函数。`print_num` 函数有一个单个参数 – 一个对 `int` 的非 const 引用。正如我们所知，这个函数不会尝试修改引用所绑定到的对象，因此我们决定移除 const 属性，这样我们就可以将 const int 的引用传递给它，而不会导致编译器生成错误。

然后，我们使用了 `const_cast` 来从 `num` 中移除 const 属性，以便将其赋值给非 const 引用 `num_ref`。如果你在 Compiler Explorer 中运行此示例，你将看到以下输出：

```cpp
Program returned: 42
num is 42 
```

程序返回了 `42`，也就是说，`num` 的值是 `42`，尽管我们试图通过 `num_ref` 将其设置为 `16`。这是因为通过非 const 引用或指针修改 const 变量是未定义的行为。

`const_cast` 主要用于与非 const 正确函数接口。尽管如此，这很危险，应该避免使用，因为我们无法保证我们传递给 const-cast-away 指针或引用的函数不会尝试修改指针所指向的对象或引用所绑定到的对象。接下来，我们将介绍 `static_cast`。

## static_cast

在 C++ 中，最常用的类型转换运算符是 `static_cast`，它用于以下场景：

+   将基类指针向上转换为派生类指针，或将派生类指针向下转换为基类指针

+   要丢弃一个值表达式

+   在已知转换路径的类型之间进行转换，例如 int 到 float，`enum` 到 int，int 到 `enum` 等

我们将通过以下示例来介绍 `static_cast` 的几种用法：

```cpp
#include <cstdio>
struct Base {
    void hi() {
        printf("Hi from Base\r\n");
    }
};
struct Derived : public Base {
    void hi() {
        printf("Hi from Derived\r\n");
    }
};
int main() {
    // unsigned to signed int 
int a = -4; 
    unsigned int b = 3; 
    if(a + static_cast<int>(b) > 0) { 
        printf("%d + %d is greater than 0\r\n", a, b); 
    } 
    else {
        printf("%d + %d is not greater than 0\r\n", a,b); 
    }
    // discard an expression
int c;
    static_cast<void>(c);
    Derived derived;
    // implicit upcast
    Base * base_ptr = &derived;
    base_ptr->hi();
    // downcast
    Derived *derived_p = static_cast<Derived*>(base_ptr);
    derived_p->hi();
    return 0;
} 
```

如果我们运行上面的示例，我们将得到以下输出：

```cpp
-4 + 3 is not greater than 0
Hi from Base
Hi from Derived 
```

在上述示例中，我们使用 `static_cast` 将 `unsigned int` 转换为有符号的 `int`，这有助于缓解由隐式转换引入的混合符号整数比较问题。然而，我们仍需要确保转换是安全的，因为 `static_cast` 不会进行任何运行时检查。

使用 `static_cast` 将变量 `c` 转换为 `void` 是一种用于抑制编译器关于未使用变量警告的技术。这表明我们了解该变量，但我们故意不使用它。

在上述示例的另一部分中，我们可以看到 `Derived` 类对象的地址可以隐式转换为 `Base` 类指针。如果我们在一个指向 `Derived` 类对象的 `Base` 类指针上调用函数 `hi`，我们实际上会调用在 `Base` 类中定义的 `hi` 函数。然后我们使用 `static_cast` 将 `Base` 指针向下转型为 `Derived` 指针。

使用 `static_cast` 进行向下转型可能很危险，因为 `static_cast` 不会进行任何运行时检查以确保指针实际上指向转换的类型。`Derived` 类的对象也是 `Base` 类的对象，但反之则不成立——`Base` 不是 `Derived`。以下示例演示了为什么这很危险：

```cpp
#include <cstdio>
struct Base {
    void hi() {
 printf("Hi from Base\r\n");
 }
};
struct Derived : public Base {
    void hi() {
 printf("Hi from Derived, x = %d\r\n", x);
 }
    int x = 42;
};
int main() {
 Base base;
 Derived *derived_ptr = static_cast<Derived*>(&base);
 derived_ptr->hi();
 return 0;
} 
```

在此代码中，我们试图在基类对象上访问 `Derived` 类的成员 `x`。由于我们使用了 `static_cast`，编译器将不会报错，这将导致未定义行为，因为基类没有成员 `x`。该程序的可能输出如下所示：

```cpp
Hi from Derived, x = 1574921984 
```

为了避免这个问题，我们可以使用 `dynamic_cast`，我们将在下一节中介绍。

## dynamic_cast

`dynamic_cast` 执行类型的运行时检查，并在 `Base` 指针实际上不指向 `Derived` 类对象的情况下将结果设置为 `nullptr`。我们将通过一个示例来更好地理解它：

```cpp
#include <cstdio>
struct Base {
    virtual void hi() {
        printf("Hi from Base\r\n");
    }
};
struct Derived : public Base {
    void hi() override {
        printf("Hi from Derived\r\n");
    }
    void derived_only() {
        printf("Derived only method\r\n");
    }
};
void process(Base *base) {
    base->hi();
    if(auto ptr = dynamic_cast<Derived*>(base); ptr ! = nullptr) 
    {
        ptr->derived_only();
    }
}
int main() {
    Base base;
    Derived derived;
    Base * base_ptr = &derived;
    process(&base);
    process(base_ptr);
    return 0;
} 
```

在上述示例中，我们有一个带有 `Base` 类指针参数的 `process` 函数。该函数使用 `dynamic_cast` 将 `Base` 指针向下转型为 `Derived` 指针。在带有初始化器的 **if 语句** 中，我们使用 `dynamic_cast<Derived*>` 对 `Base` 指针的结果初始化 `ptr`。在 `if` 语句的条件中，我们检查 `ptr` 是否与 `nullptr` 不同，如果是，则可以安全地将其用作指向 `Derived` 类对象的指针。接下来，我们将介绍 `reinterpret_cast`。

## reinterpret_cast

`reinterpret_cast` 用于通过重新解释底层位来在类型之间进行转换。它可以在以下情况下使用：

+   将指针转换为足够大的整数，以容纳其所有值。

+   将整数值转换为指针。将指针转换为整数再转换回其原始类型保证具有原始值，并且可以安全地解引用。

+   要在不同类型之间转换指针，例如在 `T1` 和 `T2` 之间。只有当结果指针是 `char`、`unsigned char`、`std::byte` 或 `T1` 时，指向 `T2` 的结果指针才能安全地解引用。

+   要将函数指针 `F1` 转换为指向不同函数 `F2` 的指针。将 `F2` 转换回 `F1` 将导致指向 `F1` 的指针。

为了更好地理解 `reinterpret_cast`，我们将通过以下示例：

```cpp
#include <cstdio>
#include <cstdint>
int fun() {
    printf("fun\r\n");
    return 42;
}
int main() {
    float f = 3.14f;
    // initialize pointer to an int with float address
auto a = reinterpret_cast<int*>(&f);
    printf("a = %d\r\n", *a);
    // the above is same as:
    a = static_cast<int*>(static_cast<void*>(&f));
    printf("a = %d\r\n", *a);
    // casting back to float pointer
auto fptr = reinterpret_cast<float*>(a);
    printf("f = %.2f\r\n", *fptr);
    // converting a pointer to integer
auto int_val = reinterpret_cast<std::uintptr_t>(fptr);
    printf("Address of float f is 0x%8X\r\n", int_val);
    auto fun_void_ptr = reinterpret_cast<void(*)()>(fun);
    // undefined behavior
fun_void_ptr();
    auto fun_int_ptr = reinterpret_cast<int(*)()>(fun);
    // safe call
printf("fun_int_ptr returns %d\r\n", fun_int_ptr());
    return 0;
} 
```

您可以从本书的 GitHub 仓库运行上述示例。它位于 `Chapter09/type_safety` 目录下，您可以使用以下命令构建和运行它：

```cpp
$ cmake -B build -DMAIN_CPP_FILE_NAME="main_reinterpret_cast.cpp"
$ cmake --build build --target run_in_renode 
```

在 Renode 中运行示例将提供以下输出：

```cpp
a = 1078523331
a = 1078523331
f = 3.14
Address of float f is 0x20003F18
fun
fun
fun_int_ptr returns 42 
```

上述示例演示了 `reinterpret_cast` 的用法。我们首先使用 `reinterpret_cast<int*>(&f)` 通过浮点地址初始化了一个指向整数的指针，这相当于使用 `static_cast`，即 `static_cast<int*>(static_cast<void*>(&f))`。我们打印了解引用整型指针的值，它是 `1078523331`。这是 `float` 变量 `f` 中包含的实际位模式。它是 `3.14` 的 IEEE-754 浮点表示。

然而，根据 C++ 标准，使用浮点地址初始化的整型指针的解引用不是定义良好的行为。这被称为**类型欺骗**——将一个类型的对象当作另一个类型处理。使用 `reinterpret_cast` 进行类型欺骗是常见的，尽管它引入了未定义的行为，但在大多数平台上它确实产生了预期的结果。在通过这个示例之后，我们将讨论替代方案。

如果我们将指向整数的指针转换回指向浮点数的指针，则可以安全地解引用结果指针。

接下来，我们将指针转换为浮点整数以打印它包含的地址。我们使用了 `std::uintptr_t`，这是一种能够容纳指向 `void` 的指针的整型。在此之后，我们初始化了 `fun_void_ptr` —— 一个指向返回 `void` 的函数的指针，该函数名为 `fun`，它返回 `int`。我们对 `fun_void_ptr` 指针进行了调用，它打印了预期的输出，但仍然是未定义的。将 `fun_void_ptr` 转换为与函数 `fun` 签名匹配的指针——`fun_int_ptr`——将使通过结果指针调用 `fun` 变得安全。

接下来，我们将通过 C++ 中的类型欺骗和替代方案来使用 `reinterpret_cast`。

### 类型欺骗

尽管使用 `reinterpret_cast` 进行类型欺骗是常见的做法，尽管它引入了未定义的行为。别名规则决定了我们在 C++ 中如何访问一个对象，简单来说，我们可以通过一个指针及其 const 版本、包含该对象的 struct 或 union 以及通过 `char`、`unsigned char` 和 `std::byte` 来访问一个对象。

我们将通过以下示例来更好地理解 C++ 中的类型欺骗：

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
namespace {
struct my_struct {
    int a;
    char c;
};
void print_my_struct (const my_struct & str) {
    printf("a = %d, c = %c\r\n", str.a, str.c);
}
void process_data(const char * data) {
    const auto *pstr = reinterpret_cast<const my_struct *>(data);
    printf("%s\r\n", __func__);
    print_my_struct(pstr[0]);
    print_my_struct(pstr[1]);
}
void process_data_memcpy(const char * data) {
    my_struct my_structs[2];
    std::memcpy(my_structs, data, sizeof(my_structs));
    printf("%s\r\n", __func__);
    print_my_struct(my_structs[0]);
    print_my_struct(my_structs[1]);
}
};
int main() {
    int i = 42;
    auto * i_ptr = reinterpret_cast<char*>(&i);
    if(i_ptr[0]==42) {
        printf("Little endian!\r\n");
    }
    else {
        printf("Big endian!\r\n");
    }
    my_struct my_structs_arr[] = {{4, 'a'}, {5, 'b'}};
    char arr[128];
    std::memcpy(&arr, my_structs_arr, sizeof(my_structs_arr));
    process_data(arr);
    process_data_memcpy(arr);
    return 0;
} 
```

您可以从本书的 GitHub 仓库运行上述示例。它位于 `Chapter09/type_safety` 目录下，您可以使用以下命令构建和运行它：

```cpp
$ cmake -B build -DMAIN_CPP_FILE_NAME="main_type_punning.cpp"
$ cmake --build build --target run_in_renode 
```

在 Renode 中运行示例将提供以下输出：

```cpp
Little endian!
process_data
a = 4, c = a
a = 5, c = b
process_data_memcpy
a = 4, c = a
a = 5, c = b 
```

在上面的示例中，我们使用了 `reinterpret_cast` 将整数 `i` 作为 `chars` 数组来处理。通过检查所提及数组第一个元素的值，我们可以确定我们是在大端还是小端系统上。根据别名规则，这是一种有效的方法，但将 `chars` 数组作为其他类型处理将是未定义的行为。我们在 `void process_data` 函数中这样做，在该函数中我们将 `chars` 数组重新解释为 `my_struct` 对象的数组。程序输出正如我们所预期的那样，尽管我们引入了未定义的行为。为了减轻这个问题，我们可以使用 `std::memcpy`。

### 类型转换 - 正确的方法

使用 `std::memcpy` 是 C++ 中类型转换的唯一（截至 C++23）可用选项。在上面的示例中，我们在 `process_data_memcpy` 函数中展示了这一点。通常会有关于字节复制的担忧，使用额外的内存和运行时开销，但事实是 `memcpy` 的调用通常会被编译器优化掉。您可以通过在 Compiler Explorer 中运行上述示例并尝试不同的优化级别来验证这一点。

C++20 引入了 `std::bit_cast`，它也可以用于类型转换，如下面的示例所示：

```cpp
#include <cstdio>
#include <bit>
int main() {
    float f = 3.14f;
    auto a = std::bit_cast<int>(f);
    printf("a = %d\r\n", a);
    return 0;
} 
```

上面的程序输出如下：

```cpp
a = 1078523331 
```

上面的示例和程序输出展示了 `std::bit_cast` 用于类型转换的用法。`std::bit_cast` 将返回一个对象。我们指定要转换到的类型作为模板参数。这将是 `std::bit_cast` 的返回类型。转换类型的尺寸和我们转换到的类型必须相同。这意味着 `std::bit_cast` 不是一个将一种类型的数组解释为另一种类型数组的选项，为此我们仍然需要使用 `std::memcpy`。

接下来，我们将看到如何使用 C++ 中的强类型来提高类型安全性。

# 强类型

当我们谈论类型安全性时，我们也应该讨论使用常用类型（如整数和浮点数）来表示物理单位（如时间、长度和体积）的接口的安全性。让我们看看以下来自供应商 SDK 的函数：

```cpp
/**
  * @brief Start the direct connection establishment procedure.
A LE_Create_Connection call will be made to the controller by GAP with the initiator filter policy set to "ignore whitelist and
process connectable advertising packets only for the specified
device".
  * @param LE_Scan_Interval This is defined as the time interval from when the Controller started its last LE scan until it begins the subsequent LE scan.
Time = N * 0.625 msec.
  * Values:
  - 0x0004 (2.500 ms)  ... 0x4000 (10240.000 ms)
  * @param LE_Scan_Window Amount of time for the duration of the LE scan. LE_Scan_Window
shall be less than or equal to LE_Scan_Interval.
Time = N * 0.625 msec.
  * Values:
  - 0x0004 (2.500 ms)  ... 0x4000 (10240.000 ms)
  * @param Peer_Address_Type The address type of the peer device.
  * Values:
  - 0x00: Public Device Address
  - 0x01: Random Device Address
  * @param Peer_Address Public Device Address or Random Device Address of the device
to be connected.
    * @param Conn_Interval_Min Minimum value for the connection event interval. This shall be less than or equal to Conn_Interval_Max.
Time = N * 1.25 msec.
  * Values:
  - 0x0006 (7.50 ms)  ... 0x0C80 (4000.00 ms)
  * @param Conn_Interval_Max Maximum value for the connection event interval. This shall be
greater than or equal to Conn_Interval_Min.
Time = N * 1.25 msec.
  * Values:
  - 0x0006 (7.50 ms)  ... 0x0C80 (4000.00 ms)
  * @param Conn_Latency Slave latency for the connection in number of connection events.
  * Values:
  - 0x0000 ... 0x01F3
  * @param Supervision_Timeout Supervision timeout for the LE Link.
It shall be a multiple of 10 ms and larger than (1 + connSlaveLatency) * connInterval * 2.
Time = N * 10 msec.
  * Values:
  - 0x000A (100 ms)  ... 0x0C80 (32000 ms)
  * @param Minimum_CE_Length Information parameter about the minimum length of connection needed for this LE connection.
Time = N * 0.625 msec.
  * Values:
  - 0x0000 (0.000 ms)  ... 0xFFFF (40959.375 ms)
  * @param Maximum_CE_Length Information parameter about the maximum length of connection needed
for this LE connection.
Time = N * 0.625 msec.
  * Values:
  - 0x0000 (0.000 ms)  ... 0xFFFF (40959.375 ms)
  * @retval Value indicating success or error code.
*/
tBleStatus aci_gap_create_connection(
 uint16_t LE_Scan_Interval,
 uint16_t LE_Scan_Window,
 uint8_t Peer_Address_Type,
 uint8_t Peer_Address[6],
 uint16_t Conn_Interval_Min,
 uint16_t Conn_Interval_Max,
 uint16_t Conn_Latency,
 uint16_t Supervision_Timeout,
 uint16_t Minimum_CE_Length,
 uint16_t Maximum_CE_Length); 
```

这是一个文档良好的函数。尽管如此，理解它接受的参数及其确切单位仍然需要大量的努力。大多数参数代表时间，但以不同的方式表示。

`LE_Scan_Interval`、`LE_Scan_Window`、`Conn_Interval_Min`、`Conn_Interval_Max`、`Supervision_Timeout`、`Minimum_CE_Length` 和 `Maximum_CE_Length` 都是时间相关的参数，但它们代表不同的单位。它们是 0.625、1.25 或 10 毫秒的倍数。上述函数的供应商还提供了以下宏：

```cpp
#define CONN_L(x) ((int)((x) / 0.625f))
#define CONN_P(x) ((int)((x) / 1.25f)) 
```

下面是使用提供的宏调用上述函数的示例：

```cpp
tBleStatus status = aci_gap_create_connection(CONN_L(80), CONN_L(120), PUBLIC_ADDR, mac_addr, CONN_P(50), CONN_P(60), 0, SUPERV_TIMEOUT, CONN_L(10), CONN_L(15)); 
```

宏定义有助于提高可读性，但向此函数传递错误值的问题仍然存在。很容易出错，交换`CONN_L`和`CONN_P`宏，从而在程序中引入难以发现的错误。我们本可以用`uint16_t`，但可以定义并使用类型`conn_l`和`conn_p`。如果我们用这些修正来包装函数，我们将得到以下包装函数：

```cpp
tBleStatus aci_gap_create_connection_wrapper(
                            conn_l LE_Scan_Interval,
                            conn_l LE_Scan_Window,
 uint8_t Peer_Address_Type,
 uint8_t Peer_Address[6],
                            conn_p Conn_Interval_Min,
                            conn_p Conn_Interval_Max,
 uint16_t Conn_Latency,
 uint16_t Supervision_Timeout,
                            conn_l Minimum_CE_Length,
                            conn_l Maximum_CE_Length); 
```

在上述示例中，我们使用`conn_l`和`conn_p`类型而不是`uint16_t`，我们将如下定义这些类型：

```cpp
class conn_l {
private:
    uint16_t time_;
public:
 explicit conn_l(float time_ms) : time_(time_ms/0.625f){}
    uint16_t & get() {return time_;}
};
class conn_p {
private:
    uint16_t time_;
public:
 explicit conn_p(float time_ms) : time_(time_ms/1.25f){}
    uint16_t & get() {return time_;}
}; 
```

使用上述强类型`conn_l`和`conn_p`，我们可以像下面这样调用包装函数：

```cpp
 tBleStatus stat = aci_gap_create_connection_wrapper(
            conn_l(80),
            conn_l(120),
            PUBLIC_ADDR,
            nullptr,
            conn_p(50),
            conn_p(60),
            0,
            SUPERV_TIMEOUT,
            conn_l(10),
            conn_l(15)
    ); 
```

通过在`conn_l`和`conn_p`类型的构造函数前使用关键字`explicit`，我们确保编译器不会从整数类型进行隐式转换。这使得无法传递可以用来构造`conn_l`和`conn_p`的整数或浮点数到`aci_gap_create_connection_wrapper`。

您可以从书的 GitHub 仓库运行整个示例。它位于`Chapter09/type_safety`下，您可以使用以下命令构建和运行它：

```cpp
$ cmake -B build -DMAIN_CPP_FILE_NAME="main_strong_types.cpp"
$ cmake --build build --target run_in_renode 
```

成功编译示例意味着我们向`aci_gap_create_connection_wrapper`传递了所有正确的参数。作为一个练习，尝试用整数值而不是`conn_l`和`conn_p`参数来传递，看看它们如何阻止编译器进行隐式转换。之后，尝试从`conn_l`和`conn_p`构造函数中移除`explicit`关键字，看看会发生什么。

我们可以通过引入一个表示时间持续时间的强类型`time`来进一步改进示例，并将其作为`conn_l`和`conn_p`类型的私有成员。代码将如下所示：

```cpp
class time {
private:
    uint16_t time_in_ms_;
public:
 explicit time(uint16_t time_in_ms) : time_in_ms_(time_in_ms){}
    uint16_t & get_ms() {return time_in_ms_;}
};
time operator""_ms(unsigned long long t) {
    return time(t);
}
class conn_l {
private:
    uint16_t val_;
public:
 explicit conn_l(time t) : val_(t.get_ms()/0.625f){}
    uint16_t & get() {return val_;}
};
class conn_p {
private:
    uint16_t val_;
public:
 explicit conn_p(time t) : val_(t.get_ms()/1.25f){}
    uint16_t & get() {return val_;}
}; 
```

在上述示例中，我们创建了一个强类型时间，并将其用作`conn_l`和`conn_p`类型的私有成员。我们还使用`operator""_ms`创建了一个用户定义字面量，以使以下函数调用成为可能：

```cpp
 tBleStatus stat = aci_gap_create_connection_wrapper(
            conn_l(80_ms),
            conn_l(120_ms),
            PUBLIC_ADDR,
            nullptr,
            conn_p(50_ms),
            conn_p(60_ms),
            0_ms,
            4000_ms,
            conn_l(10_ms),
            conn_l(15_ms)
    ); 
```

在上述示例中，我们使用用户定义字面量`operator""_ms`来创建强类型时间的对象，这些对象用于实例化`conn_l`和`conn_p`对象。

以上对原始接口的更改提高了代码的可读性和编译时错误检测。使用强类型，我们使向函数传递错误值变得更加困难，从而增加了代码库的类型安全性。

# 摘要

类型安全性是任何用于关键应用的编程语言的重要方面。理解隐式转换的潜在问题对于减轻类型安全性问题至关重要。类型欺骗是 C++中另一个值得特别注意的领域，我们学习了如何正确处理它。我们还学习了如何使用强类型来减轻向具有相同类型的参数传递错误值的问题。

接下来，我们将介绍 C++中的 lambda 表达式。

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/embeddedsystems`](https://packt.link/embeddedsystems)

![二维码](img/QR_code_Discord.png)
