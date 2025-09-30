# 6

# 超越类 – 基本 C++概念

从历史上看，C++ 是从 C 语言加上类开始的，这使得类成为具有 C 背景的开发者要学习的第一个概念。在前一章中，我们详细介绍了类，在继续探讨更高级的概念之前，我们将介绍其他使 C++ 远远超出具有类的 C 的基本 C++ 概念。

在我们继续探讨更高级的主题之前，探索使 C++ 独特的其他基本概念是很重要的。在本章中，我们将涵盖以下主要主题：

+   命名空间

+   函数重载

+   与 C 的互操作性

+   引用

+   标准库容器和算法

# 技术要求

为了充分利用本章内容，我强烈建议你在阅读示例时使用 Compiler Explorer ([`godbolt.org/`](https://godbolt.org/))。选择 GCC 作为你的编译器，并针对 x86 架构。这将允许你看到标准输出（stdio）结果，并更好地观察代码的行为。由于我们使用的是现代 C++ 功能，请确保选择 C++23 标准，通过在编译器选项框中添加 `-std=c++23`。

Compiler Explorer 使得尝试代码、调整代码并立即看到它如何影响输出和生成的汇编变得容易。示例可在 GitHub 上找到 ([`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter06`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter06))。

# 命名空间

**C++ 中的命名空间**用作访问类型名称、函数、变量等的作用域指定符。它们允许我们在使用许多软件组件且经常有相似标识符的大型代码库中更容易地区分类型和函数名称。

在 C 语言中，我们通常会给类型和函数添加前缀，以便更容易区分，例如：

```cpp
typedef struct hal_uart_stm32{
    UART_HandleTypeDef huart_;
    USART_TypeDef *instance_; 
} hal_uart_stm32;
void hal_init();
uint32_t hal_get_ms(); 
```

在 C++ 中，我们可以使用命名空间而不是 C 风格的标识符前缀来组织代码的逻辑组，如下面的示例所示：

```cpp
namespace hal {
void init();
std::uint32_t tick_count;
std::uint32_t get_ms() {
    return tick_count;
}
class uart_stm32 {
private:
    UART_HandleTypeDef huart_;
    USART_TypeDef *instance_; 
};
}; 
```

`hal` 命名空间的所有成员都可以在命名空间内部无修饰地访问。要访问 `hal` 命名空间中的标识符，在命名空间外部的代码中，我们使用命名空间作为限定符，后跟作用域解析运算符（`::`），如下面的示例所示：

```cpp
hal::init();
std::uint32_t time_now = hal::get_ms(); 
```

在这个例子中，除了 `hal` 命名空间外，我们还看到了 `std` 命名空间，我们在前面的例子中使用过它。C++ 标准库类型和函数在 `std` 命名空间中声明。

我们可以使用 `using` 指令来访问无修饰的标识符，如下面的示例所示：

```cpp
using std::array;
array<int, 4> arr; 
```

`using` 指令也可以用于整个命名空间，如下面的示例所示：

```cpp
using namespace std;
array<int, 4> arr;
vector<int> vec; 
```

建议谨慎使用 `using` 指令，特别是与 `std` 一起使用，用于有限的作用域，或者更好的做法是仅引入单个标识符。

同一个命名空间可以在不同的头文件中使用来声明标识符。例如，`std::vector` 在 `vector.h` 头文件中声明，而 `std::array` 在 `array.h` 头文件中声明。这允许我们将属于同一逻辑组的来自不同头文件的代码组织在命名空间中。

未在显式命名空间内声明的函数和类型是全局命名空间的一部分。将所有代码组织在命名空间中是一种良好的做法。唯一不能在命名空间内声明而必须位于全局命名空间中的函数是 `main`。要访问全局命名空间中的标识符，我们使用作用域解析运算符，如下面的示例所示：

```cpp
const int ret_val = 0;
int main() {
    return ::ret_val;
} 
```

代码行 `return ::ret_val;` 使用了作用域解析运算符 `::`，但没有指定命名空间。这意味着它引用的是全局命名空间。因此，`::ret_val` 访问的是在函数或类外部定义的 `ret_val` 变量——即在全局作用域中。

## 未命名的命名空间

命名空间可以不使用名称限定符进行声明。这允许我们声明属于它们声明的翻译单元本地的函数和类型。在下面的示例中，我们可以看到一个未命名的命名空间的例子：

```cpp
namespace {
constexpr std::size_t c_max_retries;
std::size_t counter;
}; 
```

在代码中，我们有一个包含一些变量声明的未命名的命名空间。它们具有**内部链接**，这意味着它们不能被来自其他翻译单元的代码访问。我们可以在 C 和 C++中使用 `static` 存储指定符来实现相同的效果。

## 嵌套命名空间

命名空间也可以嵌套。我们可以在一个命名空间内部有另一个命名空间，如下面的示例所示：

```cpp
namespace sensors {
namespace environmental {
class temperature {
};
class humidity {
};
};
namespace indoor_air_quality{
class c02{
};
class pm2_5{
};
};
}; 
```

在这个例子中，我们已经在命名空间中组织了传感器。我们有一个顶级命名空间 `sensors`，它包含两个命名空间：`environmental` 和 `indoor_air_quality`。C++17 标准允许我们编写命名空间，如下面的示例所示：

```cpp
namespace sensors::environmental {
class temperature {
};
class humidity {
};
}; 
```

命名空间是使代码更易读的好方法，因为它们允许我们保持标识符短，而不需要 C 风格的前缀。

# 函数重载

在上一章中，当我们讨论继承时，我们提到了**静态绑定**。我们看到了可以为属于不同类的函数使用相同的函数名。然而，我们也可以为不同的函数参数使用相同的函数名，如下面的示例所示：

```cpp
#include <cstdio>
void print(int a) {
    printf("Int %d\r\n", a);
}
void print(float a) {
    printf("Float %2.f\r\n", a);
}
int main() {
    print(2);
    print(2.f);
    return 0;
} 
```

在这个例子中，我们有两个 `print` 函数。其中一个有一个 `int` 类型的参数，另一个有一个 `float` 类型的参数。在调用位置，编译器将根据传递给函数调用的参数选择一个 `print` 函数。

在同一作用域内具有相同名称的函数称为**重载函数**。我们不需要为这两个函数使用两个不同的名称，如 `print_int` 和 `print_float`，我们可以为这两个函数使用相同的名称，让编译器决定调用哪个函数。

为了区分两个重载的 `print` 函数——一个接受 `int` 参数，另一个接受 `float`——编译器采用了一种称为 **名称修饰** 的技术。名称修饰通过将额外的信息，如参数类型，编码到函数名称中，来修改函数名称。这确保了每个重载函数在编译代码中都有一个唯一的符号。如果我们检查上一个示例的汇编输出，我们可以观察到这些修饰过的名称：

```cpp
_Z5printi:
        mov     r1, r0
        ldr     r0, .L2
        b       printf
_Z5printf:
        vcvt.f64.f32    d16, s0
        ldr     r0, .L5
        vmov    r2, r3, d16
        b       printf 
```

我们可以看到编译器将 `_Z5printi` 和 `_Z5printf` 标签分配给了具有 `int` 和 `float` 参数的 `print` 函数。这使得它能够根据参数匹配来调度函数调用。

重载函数可以有不同数量的参数。不能使用返回类型进行函数重载。具有相同名称和相同参数的两个函数不能有不同的返回类型。以下代码将导致编译错误：

```cpp
int print(int a);
void print(int a); 
```

这段代码将被编译器视为函数重新声明，并导致错误。

函数重载是 C++的一个基本但强大的特性，它提供了一种在编译时或静态多态的机制。

# 与 C 的互操作性

你能够在 Renode 模拟器中运行的上一章的代码示例使用了 C++和 C 代码。我们使用了供应商提供的 HAL 库和 Arm 的 **通用微控制器软件接口标准** (**CMSIS**)，两者都是用 C 编写的，并包含在 `platform` 文件夹中。

如果你查看 `CMakeLists.txt` 文件以及其中的 `add_executable` 函数，你会看到列出了来自 `platform` 文件夹的 C 文件以及仅有的几个 C++文件。构建项目将提供以下控制台输出：

```cpp
[  7%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal.c.o
[ 15%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal_cortex.c.o
[ 23%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal_gpio.c.o
[ 30%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal_rcc.c.o
[ 38%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal_uart.c.o
[ 46%] Building C object CMakeFiles/bare.elf.dir/platform/STM32F0xx_HAL_Driver/Src/stm32f0xx_hal_uart_ex.c.o
[ 53%] Building ASM object CMakeFiles/bare.elf.dir/platform/startup_stm32f072xb.s.o
[ 61%] Building C object CMakeFiles/bare.elf.dir/platform/src/stm32f0xx_hal_msp.c.o
[ 69%] Building C object CMakeFiles/bare.elf.dir/platform/src/stm32f0xx_it.c.o
[ 76%] Building C object CMakeFiles/bare.elf.dir/platform/src/system_stm32f0xx.c.o
[ 84%] Building CXX object CMakeFiles/bare.elf.dir/app/src/main.cpp.o
[ 92%] Building CXX object CMakeFiles/bare.elf.dir/hal/uart/src/uart_stm32.cpp.o
[100%] Linking CXX executable bare.elf 
```

每个 C 和 C++文件都被视为一个翻译单元，并由各自的 C 和 C++编译器分别单独构建。编译完成后，C 和 C++目标文件将被链接成一个单一的 ELF 文件。

## C++中的外部和语言链接

可以从其他翻译单元引用的变量和函数具有 **外部链接**。这允许它们与在其他文件中提供的代码链接，前提是编译器可以访问声明。它们还有一个称为 **语言链接** 的属性。这个属性允许 C++与 C 代码链接。在 C++中使用 C 语言链接的语法如下：

```cpp
extern "C" {
void c_func();
} 
```

使用 C 语言链接的声明将根据 C 语言链接约定进行链接，以防止名称修饰（以及其他事项），确保与 C 翻译单元内编译的代码正确链接。

## C++中的 C 标准库

C++封装了 C 标准库，并提供与 C 语言版本同名但带有 `c` 前缀且无扩展名的头文件。例如，C 语言头文件 `<stdlib.h>` 的 C++等价文件是 `<cstdlib>`。

在 GCC 中，实现 C++包装器包括 C 标准库头文件；例如，`<cstdio>`包括`<stdio.h>`。如果你深入研究`<stdio.h>`，你可以看到它使用`__BEGIN_DECLS`和`__END_DECLS`宏保护函数声明。以下是这些宏的定义：

```cpp
/* C++ needs to know that types and declarations are C, not C++.  */
#ifdef    __cplusplus
# define __BEGIN_DECLS    extern "C" {
# define __END_DECLS    }
#else
# define __BEGIN_DECLS
# define __END_DECLS
#endif 
```

在这里，我们可以看到标准 C 库头文件通过添加语言链接指定符来处理 C++兼容性，如果使用 C++编译器。这种做法也被许多微控制器供应商提供的许多 HAL 实现所采用。如果你打开`platform/STM32F0xx_HAL_Driver/Inc`中的任何 C 头文件，你会看到当它们被 C++编译器访问时，声明被 C 语言链接指定符保护，如下所示：

```cpp
#ifdef __cplusplus
extern "C" {
#endif
// Declarations
#ifdef __cplusplus
}
#endif 
```

C 库通常被 C++程序使用，尤其是在嵌入式领域，因此总是用语言链接指定符保护它们是个好主意。如果我们在一个 C++程序中使用 C 库，并且头文件没有内部保护，我们可以在`include`位置保护头文件，如下所示：

```cpp
extern "C" {
#include "c_library.h"
} 
```

C 语言的语言链接指定符确保了使用 C 代码的 C++代码的正确链接，这在嵌入式项目中通常是这种情况。

# 引用

在前一章中，我们简要提到了引用，但没有详细解释。引用是对象的别名；也就是说，它们指向对象，因此它们必须立即初始化。它们不是对象，所以没有指向引用的指针或引用数组。

C++中有两种不同的引用类型：**左值**和**右值**引用。

## 值类别

C++表达式要么是左值要么是右值值类别。值类别有更详细的划分，但我们将保持这个简单的类别，它有一个历史起源。

**左值**通常出现在赋值表达式的左侧，但这并不总是如此。左值有一个程序可以访问的地址。以下是一些左值的示例：

```cpp
void bar();
int a = 42; // a is lvalue
int b = a; // a can also appear on the right side
int * p = &a; // pointer p is lvalue
void(*bar_ptr)() = bar; // func pointer bar_ptr is lvalue 
```

**右值**通常出现在赋值表达式的右侧。例如，字面量、不返回引用的函数调用和内置运算符调用。我们可以把它们看作是临时值。以下是一个右值的示例：

```cpp
int a = 42; // 42 is rvalue
int b = a + 16; // a + 16 is rvalue
std::size_t size = sizeof(int); // sizeof(int) is rvalue 
```

这里还有一个完整的示例，帮助你更好地理解右值：

```cpp
#include <cstdio>
struct my_struct {
    int a_;
    my_struct() : a_(0) {}
    my_struct(int a) : a_(a) {}
};
int main() {
    printf("a_ = %d\r\n", my_struct().a_);
    printf("a_ = %d\r\n", (my_struct()=my_struct(16)).a_);
    return 0;
} 
```

在前面的例子中，我们可以看到赋值运算符左侧的`my_struct()`右值表达式。示例的输出如下：

```cpp
a_ = 0
a_ = 16 
```

在第一个`printf`调用中，我们调用`my_struct`的构造函数，它返回一个临时对象，并访问`a_`成员。在下一行，我们有以下表达式：`my_struct()=my_struct(16)`。在这个表达式的左侧，我们有一个对默认构造函数的调用，它返回一个临时对象。然后我们将构造函数接受`int`的结果赋值给左侧的临时对象，这将把一个临时对象复制到另一个临时对象中。

## 左值引用

**左值引用**用于现有对象的别名。它们也可以是 const 限定。我们通过在类型名称中添加`&`来声明它们。以下代码演示了左值引用的用法：

```cpp
#include <cstdio>
int main() {
    int a = 42;
    int& a_ref = a;
    const int& a_const_ref = a;
    printf("a = %d\r\n", a);
    a_ref = 16;
    printf("a = %d\r\n", a);
    // a_const_ref = 16; compiler error
    return 0;
} 
```

如示例所示，我们可以使用引用来操作对象。在常量引用的情况下，任何尝试更改值的操作都将导致编译器错误。

## 右值引用

**右值引用**用于扩展临时右值的生命周期。我们通过在类型名称旁边使用`&&`来声明它们。以下是一些右值引用的示例用法：

```cpp
int&& a = 42;
int b = 0;
// int&& b_ref = b; compiler error
int&& b_ref = b + 10; // ok, b + 10 is rvalue 
```

右值引用不能绑定到左值。尝试这样做将导致编译器错误。右值引用对于资源管理很重要，并且它们用于移动语义，这允许资源从一个对象移动到另一个对象。

如果我们查看`std::vector`的`push_back`方法的文档，我们将看到两个声明：

```cpp
void push_back( const T& value );
void push_back( T&& value ); 
```

第一个声明用于通过复制`value`来初始化新的向量成员。带有右值引用的第二个声明将移动`value`，这意味着新的向量成员将接管`value`对象从动态分配的资源。让我们看一下以下示例，以了解移动语义的基本知识：

```cpp
#include <string>
#include <vector>
#include <cstdio>
int main()
{
    std::string str = "Hello world, this is move semantics demo!!!";
    printf("str.data address is %p\r\n", (void*)str.data());
    std::vector<std::string> v;
    v.push_back(str);
    printf("str after copy is <%s>\r\n", str.data());
    v.push_back(std::move(str));
    //v.push_back(static_cast<std::string&&>(str));
    printf("str after move is <%s>\r\n", str.data());

    for(const auto & s:v) {
        printf("s is <%s>\r\n", s.data());
        printf("s.data address is %p\r\n", (void*)s.data());
    }
    return 0;
} 
```

在这个示例中，我们对`std::vector<std::string>`的`push_back`方法进行了两次调用。第一次调用`v.push_back(str);`将`str`复制到向量中。在此操作之后，原始的`str`保持不变，这由输出得到证实：

```cpp
str.data address is 0x84c2b0
str after copy is <Hello world, this is move semantics demo!!!> 
```

第二次调用`v.push_back(std::move(str));`使用`std::move`将`str`转换为右值引用。这向编译器表明`str`的资源可以被移动而不是复制。因此，`str`的内部数据被转移到向量中的新字符串，而`str`被留下处于有效但未指定的状态，通常变为空：

```cpp
str after move is <>
s is <Hello world, this is move semantics demo!!!>
s.data address is 0x84d330
s is <Hello world, this is move semantics demo!!!>
s.data address is 0x84c2b0 
```

在前面的输出中，我们还使用`s.data()`和`str.data()`打印了字符串底层字符数组的地址。以下是发生的情况：

+   原始的`str`其数据位于地址`0x84c2b0`

+   在将字符串`str`复制到向量中后，第一个元素`v[0]`拥有其数据的不同地址的副本（`0x84d330`），这证实了一个深拷贝已被创建

移动之后，向量中的第二个元素`v[1]`现在指向原始数据地址`0x84c2b0`。这表明`str`的内部数据被移动到`v[1]`而没有复制。这只是移动语义的一瞥；还有更多内容，但由于它主要用于管理动态分配的资源，我们不会更详细地介绍它。

# 标准库容器和算法

我们已经在之前的章节中讨论了一些 C++库中的容器，例如`std::vector`和`std::array`。由于`std::vector`依赖于动态内存分配，`std::array`通常在嵌入式应用中是首选的容器。

## 数组

标准库中的数组在栈上分配一个连续的内存块。我们可以将数组视为一个简单的包装器，它包含一个 C 风格数组的类型，并在其中包含数组的大小。它是一个模板类型，使用底层数据类型和大小实例化。

我们可以使用一个方法来访问数组成员，如果使用越界索引访问，它将抛出一个异常。这使得它比 C 风格数组更安全，因为它允许我们在运行时捕获越界访问错误并处理它们。如果禁用了异常，我们可以设置一个全局终止处理程序来执行我们的功能。我们有机会在本书的*第二章*中看到这一点，当时我们正在讨论异常。

我们可以使用 `std::array` 创建一个类似向量的容器，我们可以使用它与容器适配器，如 `std::stack` 或 `std::priority` 队列。我们将我们的新类型称为 `fixed_vector`。它将继承自 `std::array` 并实现 `push_back`、`pop_back`、`empty` 和 `end` 方法。以下是使用标准库中的数组实现我们的新类型的示例：

```cpp
template <typename T, size_t S> class fixed_vector : public std::array<T, S> {
  public:
    void push_back(const T &el) {
        if(cnt_ < S) {
            this->at(cnt_) = el;
            ++cnt_;
        }
    }
    T &back() {
        return this->at(cnt_-1);
    }
    void pop_back() {
        if(cnt_) {
            --cnt_;
        }
    }
    auto end() {
        return std::array<T, S>::begin() + cnt_;
    }
    bool empty() const {
        return cnt_ == 0;
    }
  private:
    size_t cnt_ = 0;
}; 
```

我们的新类型 `fixed_vector` 利用底层的 `std::array` 并实现 `push_back` 函数来向数组的末尾添加元素。如果我们尝试添加比可能更多的元素，它将静默失败。此行为可以根据应用程序的要求进行调整。它还实现了 `back` 方法，该方法返回对最后一个元素的左值引用，以及 `pop_back`，它递减用于跟踪容器中存储的元素数量的私有成员 `cnt_`。

我们可以使用我们的新容器类型 `fixed_vector` 作为容器适配器（如栈和优先队列）的底层容器类型。

## 容器适配器

栈是一个简单的**后进先出（LIFO**）容器适配器，优先队列在插入元素时会对其进行排序。我们可以在以下示例中看到如何使用 `fixed_vector`：

```cpp
int main() {
    std::priority_queue<int, fixed_vector<int, 10>> pq;
    pq.push(10);
    pq.push(4);
    pq.push(8);
    pq.push(1);
    pq.push(2);
    printf("Popping elements from priority queue: ");
    while(!pq.empty()) {
       printf("%d ", pq.top());
       pq.pop();
    }
    std::stack<int, fixed_vector<int, 10>> st;
    st.push(10);
    st.push(4);
    st.push(8);
    st.push(1);
    st.push(2);
    printf("\r\nPopping elements from stack (LIFO): ");
    while(!st.empty()) {
       printf("%d ", st.top());
       st.pop();
    }
    return 0;
} 
```

在这个例子中，我们使用 `fixed_vector` 实例化 `std::stack` 和 `std::priority_queue` 模板类型。如果我们运行这个程序，我们将得到以下输出：

```cpp
Popping elements from priority queue: 10 8 4 2 1
Popping elements from stack (LIFO): 2 1 8 4 10 
```

如您从输出中看到的，优先队列中的元素是排序的，而栈中的元素是按照后进先出（LIFO）原则弹出的。

标准库提供了各种容器，我们刚刚触及了它提供的可能性的一角。它还提供了在容器上操作的算法。

## 算法

C++ 标准库提供了包含在 `algorithm` 头文件中的大量模板算法函数，这些函数与不同的容器类型配合良好。我们现在将介绍其中的一些。

### std::copy 和 std::copy_if

`std::copy` 和 `std::copy_if` 用于将元素从一个容器复制到另一个容器。`std::copy_if` 还接受一个谓词函数，用于控制是否复制成员，如下面的示例所示：

```cpp
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
void print_container(const auto& container) {
    for(auto& elem: container) {
       printf("%d ", elem);
    }
       printf("\r\n");
}
int main() {
    std::array<int, 10> src{0};
    std::array<int, 10> dst{0};
    std::iota(src.begin(), src.end(), 0);
    std::copy_if(src.begin(), src.end(), dst.begin(),[] 
        (int x) {return x > 3;});
    print_container(src);
    print_container(dst);
    return 0;
} 
```

在这个例子中，我们使用 `std::iota` 从数值头文件初始化 `src` 数组，以递增的值开始，从 `0` 开始。然后，我们使用 `std::copy_if` 将 `src` 数组中所有大于 3 的元素复制到 `dst` 数组中。

### std::sort

`std::sort` 用于对容器中的元素进行排序。在下面的例子中，我们将生成随机元素并对其进行排序：

```cpp
int main() {
    std::array<int, 10> src{0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 6);
    auto rand = & -> int {
        return distrib(gen);
    };
    std::transform(src.begin(), src.end(), src.begin(), rand);
    print_container(src);
    std::sort(src.begin(), src.end());
    print_container(src);
    return 0;
} 
```

在这个例子中，我们使用 `std::transform` 来填充 `src` 数组，它将 `rand` lambda 应用到 `src` 数组的每个成员上。我们使用了 `random` 头文件中的类型来生成介于 1 和 6 之间的随机数。在我们用随机数填充数组之后，我们使用 `std::sort` 对其进行排序。这个程序的可能的输出如下所示：

```cpp
6 6 1 1 6 5 4 4 1 1
1 1 1 1 4 4 5 6 6 6 
```

我们首先看到排序和应用 `std::sort` 之前的数组中的值。我们本可以用 `for` 循环来填充初始数组，但我们利用这个机会在这里展示了 `std::transform`。

这些是从 C++ 标准库中的一些算法；还有更多可以用来有效地解决容器中常见任务的算法。

# 摘要

在本章中，我们介绍了 C++ 的基础知识，例如命名空间、函数重载、引用以及标准库容器和算法。我们还学习了如何在 C++ 程序中实现和使用 C 兼容性。

在下一章中，我们将学习 C++ 中的错误处理机制。
