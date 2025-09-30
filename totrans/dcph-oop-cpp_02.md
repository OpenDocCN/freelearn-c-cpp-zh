# 2

# 添加语言必要性

本章将介绍 C++中必要的非面向对象特性，这些特性是 C++面向对象特性的关键构建块。本章中介绍的特性代表了从这一点开始在书中将直接使用的主题。C++是一种充满灰色地带的语言；从本章开始，你将不仅熟悉语言特性，还将熟悉语言的细微差别。本章的目标将是开始提升你的技能，从一名普通 C++程序员转变为能够在创建可维护代码的同时成功操作语言细微差别的人。

在本章中，我们将涵盖以下主要主题：

+   `const`限定符

+   函数原型

+   函数重载

到本章结束时，你将理解非面向对象特性，如`const`限定符、函数原型（包括使用默认值）和函数重载（包括标准类型转换如何影响重载函数的选择以及可能产生的潜在歧义）。许多这些看似简单的话题都包含了一系列有趣的细节和细微差别。这些技能对于成功进行本书的下一章至关重要。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 中找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter02`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter02)。每个完整程序示例都可以在 GitHub 的相应章节标题（子目录）下找到，对应章节的文件名，后面跟着一个连字符，然后是当前章节中的示例编号。例如，*第二章*中的第一个完整程序，*添加语言必要性*，可以在上述 GitHub 目录下的`Chapter02`子目录中的`Chp2-Ex1.cpp`文件中找到。

本章的 CiA 视频可以在以下链接查看：[`bit.ly/3CM65dF`](https://bit.ly/3CM65dF)。

# 使用`const`和`constexpr`限定符

在本节中，我们将向变量添加`const`和`constexpr`限定符，并讨论它们如何添加到函数的输入参数和返回值中。随着我们在 C++语言中的前进，这些限定符将被广泛使用。使用`const`和`constexpr`可以使值被初始化，但之后不再修改。函数可以通过使用`const`或`constexpr`来声明它们不会修改其输入参数，或者它们的返回值可能只能被捕获（但不能修改）。这些限定符有助于使 C++成为一种更安全的语言。让我们看看`const`和`constexpr`的实际应用。

## `const`和`constexpr`变量

一个有资格的 `const` 变量是一个必须初始化且永远不会被赋予新值的变量。将 `const` 和变量的使用放在一起似乎是一个悖论——`const` 意味着不要改变，而变量的概念是天生可以持有不同的值。尽管如此，拥有一个强类型检查的变量，其唯一值可以在运行时确定，是非常有用的。关键字 `const` 被添加到变量声明中。

类似地，使用 `constexpr` 声明的变量是一个有资格的常量变量——它可以被初始化，但永远不会被赋予新的值。只要可能，`constexpr` 的使用正在变得越来越受欢迎。

在某些情况下，常量的值在编译时是未知的。一个例子可能是如果使用用户输入或函数的返回值来初始化一个常量。一个 `const` 变量可以在运行时轻松初始化。`constexpr` 变量通常可以在运行时初始化，但并不总是如此。在我们的例子中，我们将考虑各种情况。

让我们在下面的程序中考虑几个例子。我们将把这个程序分成两个部分进行更具体的解释，然而，完整的程序示例可以在以下链接中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
#include <cstring> // though, we'll prefer std:: string,
// char [ ] demos the const qualifier easily in cases below
using std::cout;     // preferable to: using namespace std;
using std::cin;
using std::endl;
using std::setw;
// simple const variable declaration and initialization
// Convention will capitalize those known at compile time
// (those taking the place of a former macro #define)
const int MAX = 50; 
// simple constexpr var. declaration and init. (preferred)
constexpr int LARGEST = 50;
constexpr int Minimum(int a, int b)  
// function definition w formal parameters
{
    return (a < b)? a : b;   // conditional operator ?: 
}
```

在前面的程序段中，注意我们如何使用 `const` 修饰符在数据类型之前声明一个变量。这里，`const int MAX = 50;` 简单地将 `MAX` 初始化为 50。`MAX` 在代码的后续部分不能通过赋值来修改。出于惯例，简单的 `const` 和 `constexpr` 资格变量（取代了曾经使用的 `#define` 宏）通常使用大写字母，而计算（或可能计算）的值则使用典型的命名约定。接下来，我们使用 `constexpr int LARGEST = 50;` 引入一个常量变量，同样不能被修改。这个选项正在成为首选用法，但并不总是可以使用的。

接下来，我们有函数 `Minimum()` 的定义；注意在这个函数体中使用了三元条件运算符 `?:`。同时注意，这个函数的返回值被 `constexpr` 赋予了资格（我们很快就会检查这一点）。接下来，让我们在继续这个程序的其余部分时检查 `main()` 函数的主体：

```cpp
int main()
{
    int x = 0, y = 0;
    // Since 'a', 'b' could be calculated at runtime
    // (such as from values read in), we will use lowercase
    constexpr int a = 10, b = 15;// both 'a', 'b' are const
    cout << "Enter two <int> values: ";
    cin >> x >> y;
    // const variable initialized w return val. of a fn.
    const int min = Minimum(x, y);  
    cout << "Minimum is: " << min << endl;
    // constexpr initialized with return value of function
    constexpr int smallest = Minimum(a, b);           
    cout << "Smallest of " << a << " " << b << " is: " 
         << smallest << endl;
    char bigName[MAX] = {""};  // const used to size array
    char largeName[LARGEST] = {""}; // same for constexpr 
    cout << "Enter two names: ";
    cin >> setw(MAX) >> bigName >> setw(LARGEST) >>
           largeName;
    const int namelen = strlen(bigName);   
    cout << "Length of name 1: " << namelen << endl;
    cout << "Length of name 2: " << strlen(largeName) <<
             endl;
    return 0;
}
```

在 `main()` 函数中，让我们考虑以下代码序列：我们提示用户输入 `"Enter two values: "` 并分别将它们存储在变量 `x` 和 `y` 中。在这里，我们调用函数 `Minimum(x,y)` 并将刚刚使用 `cin` 和提取操作符 `>>` 读取的两个值 `x` 和 `y` 作为实际参数传递。请注意，在 `min` 的 `const` 变量声明旁边，我们使用函数调用 `Minimum()` 的返回值初始化 `min`。重要的是要注意，设置 `min` 是作为一个单独的声明和初始化捆绑在一起的。如果将此拆分为两行代码——变量声明后跟赋值——编译器将会报错。标记为 `const` 的变量只能初始化为一个值，并且在声明后不能赋值。

接下来，我们将函数 `Minimum(a, b)` 的返回值初始化给 `smallest`。请注意，参数 `a` 和 `b` 是可以在编译时确定的字面量值。同时请注意，`Minimum()` 函数的返回值已经被 `constexpr` 标记。这种标记是必要的，以便 `constexpr smallest` 能够使用函数的返回值进行初始化。注意，如果我们尝试将 `x` 和 `y` 传递给 `Minimum()` 来设置 `smallest`，将会得到一个错误，因为 `x` 和 `y` 的值不是字面量值。

在上一个示例的最后一段代码中，请注意我们使用 `MAX`（在程序示例的早期部分定义）来在声明 `char bigName[MAX];` 中为固定大小数组 `bigName` 定义一个大小。我们同样使用 `LARGEST` 来为固定大小数组 `largeName` 定义一个大小。在这里，我们看到可以使用 `const` 或 `constexpr` 来以这种方式定义数组的大小。然后我们进一步在 `setw(MAX)` 中使用 `MAX`，在 `setw(LARGEST)` 中使用 `LARGEST`，以确保在读取键盘输入时使用 `cin` 和提取操作符 `>>` 不溢出 `bigName` 或 `largeName`。最后，我们使用函数 `strlen(bigname)` 的返回值初始化变量 `const int namelen` 并使用 `cout` 打印这个值。请注意，因为 `strlen()` 不是一个其值被 `constexpr` 标记的函数，所以我们不能使用这个返回值来初始化一个 `constexpr`。

伴随上述完整程序示例的输出如下：

```cpp
Enter two <int> values: 39 17
Minimum is: 17
Smallest of 10 15 is: 10
Enter two names: Gabby Dorothy
Length of name 1: 5
Length of name 2: 7
```

现在我们已经看到了如何使用 `const` 和 `constexpr` 标记变量，让我们考虑函数的常量标记。

## 函数的 `const` 标记

关键字`const`和`constexpr`也可以与函数一起使用。这些修饰符可以在参数之间使用，以指示参数本身不会被修改。这是一个有用的特性——函数的调用者将理解该函数不会修改以这种方式修饰的输入参数。然而，由于非指针（和非引用）变量作为栈上实际参数的副本以值传递的方式传递给函数，因此`const`或`constexpr`修饰这些参数的内在副本并不起作用。因此，不需要对标准数据类型的参数进行`const`或`constexpr`修饰。

同样的原则也适用于函数的返回值。函数的返回值可以是`const`或`constexpr`修饰的；然而，除非返回的是一个指针（或引用），否则作为返回值传递回栈上的项是一个副本。因此，当返回类型是指向常量对象的指针（我们将在*第三章*，*间接寻址：指针*，以及更后面讨论）时，`const`修饰的返回值更有意义。注意，如果一个函数的返回值将被用来初始化一个`constexpr`变量，那么需要这个函数有一个`constexpr`修饰的返回值，正如我们在之前的例子中所看到的。作为`const`的最后一次使用，当我们转向类的 OO 细节时，我们可以使用这个关键字来指定特定的成员函数将不会修改该类的任何数据成员。我们将在*第五章*，*详细探索类*中探讨这种情况。

现在我们已经了解了`const`和`constexpr`修饰符在变量中的用法，并看到了`const`和`constexpr`与函数结合使用的潜在用途，让我们继续本章的下一个语言特性：函数原型。

# 与函数原型一起工作

在本节中，我们将检查函数原型的机制，例如在文件中以及跨多个文件放置的必要性，以增加程序灵活性。我们还将为原型参数添加可选名称，并理解我们为什么可能选择在 C++原型中添加默认值。函数原型确保 C++代码进行强类型检查。

在进入函数原型之前，让我们花一点时间回顾一些必要的编程术语。**函数定义**指的是一个函数的代码主体，而函数的声明（也称为**前向声明**）只是引入一个函数名及其返回类型和参数类型。前向声明允许编译器通过比较调用与前向声明来执行函数调用和定义之间的强类型检查。前向声明是有用的，因为函数定义并不总是出现在函数调用之前的文件中；有时，函数定义出现在与它们的调用不同的文件中。

## 定义函数原型

**函数原型**是函数的前向声明，它描述了函数应该如何正确调用。原型确保了函数调用和其定义之间的强类型检查。一个简单的函数原型包括以下内容：

+   函数的返回类型

+   函数的名称

+   函数的类型和参数数量

函数原型允许函数调用先于函数的定义，或者允许调用存在于单独文件中的函数。随着我们学习更多 C++语言特性，例如异常，我们将看到更多元素有助于函数的扩展原型（和扩展签名）。现在，让我们看看一个简单的例子：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex2.cpp)

```cpp
#include <iostream>
using std::cout;     // preferred to: using namespace std;
using std:: endl;
[[nodiscard]] int Minimum(int, int);   // fn. prototype

int main()
{
    int x = 5, y = 89;
    // function call with actual parameters
    cout << Minimum(x, y) << endl;     
    return 0;                          
}
[[nodiscard]] int Minimum(int a, int b) // fn. definition
                                 // with formal parameters 
{
    return (a < b)? a : b;  
}
```

注意，我们在上述示例的早期原型化了`int Minimum(int, int);`。这个原型让编译器知道任何对`Minimum()`的调用都应该接受两个整数参数，并返回一个整数值（我们将在本节稍后讨论类型转换）。 

还要注意在函数返回类型之前使用`[[nodiscard]]`。这表示程序员应该存储返回值或以其他方式使用返回值（例如在表达式中）。如果忽略此函数的返回值，编译器将发出警告。

接下来，在`main()`函数中，我们调用函数`Minimum(x, y)`。此时，编译器检查函数调用是否与上述原型在类型和参数数量以及返回类型上匹配。也就是说，两个参数都是整数（或者可以轻松转换为整数），返回类型是整数（或者可以轻松转换为整数）。返回值将被用作`cout`打印的值。最后，在文件中定义了`Minimum()`函数。如果函数定义与原型不匹配，编译器将引发错误。

原型的存在使得在编译器看到函数定义之前，可以完全对给定函数的调用进行类型检查。当前的例子当然是人为设计的，以说明这一点；我们本可以交换文件中`Minimum()`和`main()`出现的顺序。然而，想象一下`Minimum()`的定义包含在一个单独的文件中（这是更典型的情况）。在这种情况下，原型将出现在调用此函数的文件顶部（以及头文件包含），以便可以对原型进行完全的类型检查。

在上述多文件场景中，包含函数定义的文件将单独编译。然后，链接器的任务将确保当多个文件链接在一起时，函数定义和所有原型匹配，以便链接器可以解决对这种函数调用的任何引用。如果原型与函数定义不匹配，链接器将无法将代码的不同部分链接成一个编译单元。

让我们看看这个例子的输出：

```cpp
5
```

现在我们已经了解了函数原型的基础知识，让我们看看我们如何可以向函数原型添加可选的参数名称。

## 在函数原型中命名参数

函数原型可以可选地包含与形式参数列表或实际参数列表中不同的名称。参数名称被编译器忽略，但通常可以增强可读性。让我们回顾一下我们之前的例子，在函数原型中添加可选的参数名称：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex3.cpp)

```cpp
#include <iostream>
using std::cout;    // preferred to: using namespace std;
using std::endl;
// function prototype with optional argument names
[[nodiscard]] int Minimum(int arg1, int arg2);

int main()
{
    int x = 5, y = 89;
    cout << Minimum(x, y) << endl;      // function call
    return 0;
}
[[nodiscard]] int Minimum(int a, int b) // fn. definition
{
    return (a < b)? a : b;  
}
```

这个例子几乎与前面的例子相同。然而，请注意，函数原型包含命名参数`arg1`和`arg2`。这些标识符立即被编译器忽略。因此，这些命名参数不需要与函数的形式参数或实际参数匹配，并且只是可选的，仅为了提高可读性。

伴随这个例子的输出与前面的例子相同：

```cpp
5
```

接下来，让我们通过向函数原型添加一个有用的功能来继续我们的讨论：默认值。

## 在函数原型中添加默认值

**默认值**可以在函数原型中指定。这些值将在函数调用中缺少实际参数时使用，并作为实际参数本身。默认值遵循以下标准：

+   默认值必须在函数原型中从右到左指定，不能省略任何值。

+   实际参数在函数调用中从左到右进行替换；因此，在原型中指定默认值的顺序是重要的。

函数原型可以全部、部分或没有任何值被默认值填充，只要默认值符合上述规范。

让我们通过一个使用默认值的示例来了解一下：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex4.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex4.cpp)

```cpp
#include <iostream>
using std::cout;    // preferred to: using namespaces std;
using std::endl;
// fn. prototype with one default value
[[nodiscard]] int Minimum(int arg1, int arg2 = 100000);  
int main()
{
    int x = 5, y = 89;
    cout << Minimum(x) << endl; // function call with only
                             // one argument (uses default)
    cout << Minimum(x, y) << endl; // no default vals used
    return 0;
}
[[nodiscard]] int Minimum(int a, int b) // fn. definition
{
    return (a < b)? a : b;  
}
```

在这个例子中，请注意，在 `int Minimum(int arg1, int arg2 = 100000);` 的函数原型中添加了一个默认值到最右边的参数。这意味着当从 `main()` 调用 `Minimum()` 时，它可以带有一个参数调用 `Minimum(x)`，或者带有两个参数调用 `Minimum(x, y)`。当 `Minimum()` 带有一个参数被调用时，单个参数绑定到函数形式参数列表中的最左边参数，默认值绑定到下一个顺序参数。然而，当 `Minimum()` 带有两个参数被调用时，两个实际参数都绑定到函数的形式参数；默认值不会被使用。

下面是这个示例的输出：

```cpp
5
5
```

现在我们已经掌握了函数原型中的默认值，让我们通过在程序的不同作用域中使用不同的默认值来扩展这个想法。

## 在不同作用域中使用不同的默认值进行原型设计

函数可以在不同的作用域中使用不同的默认值进行原型设计。这允许函数以通用方式构建，并通过原型在多个应用程序或代码的多个部分中进行定制。

下面是一个示例，展示了同一函数（在不同作用域）使用不同默认值的多重原型：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex5.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex5.cpp)

```cpp
#include <iostream>
using std::cout;    // preferred to: using namespace std;
using std::endl;
// standard function prototype
[[nodiscard]] int Minimum(int, int);   
void Function1(int x)
{   
    // local prototype with default value
    [[nodiscard]] int Minimum(int arg1, int arg2 = 500); 
    cout << Minimum(x) << endl; 
}
void Function2(int x)
{
    // local prototype with default value
    [[nodiscard]] int Minimum(int arg1, int arg2 = 90);  
    cout << Minimum(x) << endl; 
}

[[nodiscard]] int Minimum(int a, int b) // fn. definition
{ 
    return (a < b)? a : b;   
}
int main()
{
    Function1(30);    
    Function2(450);
    return 0;
}
```

在这个例子中，请注意，`int Minimum(int, int);` 的原型在文件顶部附近被定义。然后请注意，在 `Function1()` 的更局部作用域中，函数 `Minimum()` 被重新原型化为 `int Minimum(int arg1, int arg2 = 500);`，为最右边的参数指定了默认值 `500`。同样，在 `Function2()` 的作用域中，函数 `Minimum()` 被重新原型化为 `int Minimum(int arg1, int arg2 = 90);`，在右边的参数中指定了默认值 `90`。当在 `Function1()` 或 `Function2()` 内部调用 `Minimum()` 时，每个函数作用域中的局部原型分别会被使用——每个都有自己的默认值。

以这种方式，程序的具体区域可以很容易地通过具有特定应用部分中具有意义的默认值进行定制。然而，请确保**仅**在调用函数的作用域内使用具有个性化默认值的函数重原型，以确保这种定制可以轻松地包含在非常有限的范围内。永远不要在全局作用域中用不同的默认值重原型化函数——这可能导致意外和错误的结果。

以下是对该例子的输出：

```cpp
30
90
```

现在我们已经探讨了与单文件和多文件中的默认使用、原型中的默认值以及在不同作用域中用个性化默认值重原型化函数相关的函数原型，我们现在可以继续本章的最后一个重要主题：函数重载。

# 理解函数重载

C++ 允许存在两个或多个具有相似目的但参数类型或数量不同的函数，它们可以与相同的函数名称共存，这被称为**函数重载**。这允许进行更通用的函数调用，让编译器根据使用函数的变量（对象）的类型选择正确的函数版本。在本节中，我们将向函数重载的基本知识中添加默认值，以提供灵活性和定制。我们还将了解标准类型转换可能如何影响函数重载，以及可能出现的潜在歧义（以及如何解决这些类型的疑问）。

## 学习函数重载的基本知识

当存在两个或多个具有相同名称的函数时，这些类似函数之间的区别因素将是它们的签名。通过改变函数的签名，可以在同一个命名空间中存在两个或多个具有其他名称相同的函数。函数重载依赖于函数的签名如下：

+   **函数的签名**指的是函数的名称，加上它的类型和参数数量。

+   函数的返回类型不包括在其签名部分。

+   具有相同目的的两个或多个函数可以共享相同的名称，前提是它们的签名不同。

函数的签名有助于为每个函数提供一个内部，“混淆”的名称。这种编码方案保证了每个函数在编译器内部都有唯一的表示。

让我们花几分钟时间理解一个稍微大一点的例子，这个例子将包含函数重载。为了简化解释，这个例子被分为三个部分；尽管如此，完整的程序可以在以下链接中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex6.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex6.cpp)

```cpp
#include <iostream>
#include <cmath>
using std::cout;    // preferred to: using namespace std;
using std::endl;
constexpr float PI = 3.14159;
class Circle     // simple user defined type declarations
{
public:
   float radius;
   float area;
};
class Rectangle
{
public:
   float length;
   float width;
   float area;
};
void Display(Circle);     // 'overloaded' fn. prototypes
void Display(Rectangle);  // since they differ in signature
```

在这个示例的开始，注意我们使用`#include <cmath>`包含数学库，以提供对基本数学函数，如`pow()`的访问。接下来，注意`Circle`和`Rectangle`的类定义，每个类都有相关的数据成员（`Circle`的`radius`和`area`；`Rectangle`的`length`、`width`和`area`）。一旦定义了这些类型，就显示了两个重载的`Display()`函数的原型。由于两个显示函数的原型使用了用户定义的类型`Circle`和`Rectangle`，因此重要的是`Circle`和`Rectangle`之前已经被定义。现在，让我们在继续程序的下一个部分时检查`main()`函数的主体：

```cpp
int main()
{
    Circle myCircle;
    Rectangle myRect;
    Rectangle mySquare;
    myCircle.radius = 5.0;
    myCircle.area = PI * pow(myCircle.radius, 2.0);
    myRect.length = 2.0;
    myRect.width = 4.0;
    myRect.area = myRect.length * myRect.width;
    mySquare.length = 4.0;
    mySquare.width = 4.0;
    mySquare.area = mySquare.length * mySquare.width;
    Display(myCircle);   // invoke: void display(Circle)
    Display(myRect);     // invoke: void display(Rectangle)
    Display(mySquare);
    return 0;
}
```

现在，在`main()`函数中，我们声明了一个`Circle`类型的变量和两个`Rectangle`类型的变量。然后我们使用点操作符（`.`）和适当的值，在`main()`函数中为这些变量的数据成员加载。接下来在`main()`函数中，有三次对`Display()`的调用。第一次函数调用`Display(myCircle)`将调用接受一个`Circle`作为形式参数的`Display()`版本，因为传递给这个函数的实际参数实际上是用户定义的类型`Circle`。接下来的两个函数调用`Display(myRect)`和`Display(mySquare)`将调用接受一个`Rectangle`作为形式参数的重载版本，因为在这两个调用中传递的实际参数本身就是`Rectangle`类型。让我们通过检查`Display()`的两个函数定义来完成这个程序：

```cpp
void Display (Circle c)
{
   cout << "Circle with radius " << c.radius;
   cout << " has an area of " << c.area << endl; 
}

void Display (Rectangle r)
{
   cout << "Rectangle with length " << r.length;
   cout << " and width " << r.width;
   cout << " has an area of " << r.area << endl; 
}
```

注意在这个示例的最后部分，定义了`Display()`的两个版本。其中一个函数接受一个`Circle`作为形式参数，而重载版本接受一个`Rectangle`作为其形式参数。每个函数体访问特定于其形式参数类型的成员数据，然而每个函数的整体功能相似，因为在每种情况下，都显示了一个特定的形状（`Circle`或`Rectangle`）。

让我们看看这个完整程序示例的输出：

```cpp
Circle with radius 5 has an area of 78.5397
Rectangle with length 2 and width 4 has an area of 8
Rectangle with length 4 and width 4 has an area of 16
```

接下来，让我们通过了解标准类型转换如何允许一个函数被多个数据类型使用来扩展我们对函数重载的讨论。这可以使函数重载的使用更加选择性地进行。

## 使用标准类型转换消除过度重载

基本语言类型可以由编译器自动从一种类型转换为另一种类型。这允许语言提供比其他情况下所需更少的操作符来操作标准类型。标准类型转换还可以在保持函数参数的精确数据类型不是至关重要的情况下消除函数重载的需要。在包括赋值和操作的表达式中，标准类型之间的提升和降级通常被透明地处理，无需显式类型转换。

这里有一个示例，说明了简单的标准类型转换。此示例不包括函数重载：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex7.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex7.cpp)

```cpp
#include <iostream>
using std::cout;    // preferred to: using namespace std;
using std::endl;
int Maximum(double, double);      // function prototype

int main()
{
    int result = 0;
    int m = 6, n = 10;
    float x = 5.7, y = 9.89;

    result =  Maximum(x, y); 
    cout << "Result is: " << result << endl;
    cout << "The maximum is: " << Maximum(m, n) << endl;
    return 0;
}
int Maximum(double a, double b)  // function definition
{
    return (a > b)? a : b;
}
```

在此示例中，`Maximum()`函数接受两个双精度浮点数作为参数，并将结果作为`int`类型返回。首先请注意，`int Maximum(double, double);`在程序顶部附近进行了原型声明，并在同一文件的底部进行了定义。

现在，在`main()`函数中，请注意我们定义了三个`int`类型的变量：`result`、`a`和`x`。后两个变量分别初始化为`6`和`10`。我们还定义了两个浮点数并初始化：`float x = 5.7, y = 9.89;`。在第一次调用`Maximum()`函数时，我们使用`x`和`y`作为实际参数。这两个浮点数被提升为双精度浮点数，并且函数按预期被调用。

这是一个标准类型转换的示例。让我们注意到`int Maximum(double, double)`的返回值是一个整数——不是一个双精度数。这意味着从这个函数返回的值（无论是形式参数`a`还是`b`）将是一个`a`或`b`的副本，首先将其截断为整数，然后用作返回值。这个返回值被整洁地分配给在`main()`中声明的`int`类型的`result`。这些都是标准类型转换的例子。

接下来，使用实际参数`m`和`n`调用`Maximum()`函数。与之前的函数调用类似，整数`m`和`n`被提升为双精度，函数按预期被调用。返回值也将被截断回`int`类型，并将此值传递给`cout`以打印为整数。

此示例的输出如下：

```cpp
Result is: 9
The maximum is: 10
```

现在我们已经了解了函数重载和标准类型转换的工作原理，让我们考察一个两种情况结合可能会产生模糊函数调用的场景。

## 函数重载和类型转换引起的歧义

当一个函数被调用，其形式参数和实际参数在类型上完全匹配时，关于应该调用选择中的哪个重载函数不会产生歧义——与完全匹配的函数是明显的选择。然而，当一个函数被调用且其形式参数和实际参数在类型上不同时，可能需要对实际参数执行标准类型转换。然而，存在形式参数和实际参数类型不匹配的情况，并且存在重载函数。在这些情况下，编译器可能难以选择哪个函数应该被选为最佳匹配。在这些情况下，编译器会生成一个错误，表明与函数调用本身配对的可用选择是不确定的。显式类型转换或在更局部的作用域中重新原型化所需的选择可以帮助纠正这些其他情况下可能的不确定性。

让我们回顾一个简单的函数，它展示了函数重载、标准类型转换以及潜在的不确定性：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex8.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter02/Chp2-Ex8.cpp)

```cpp
#include <iostream>
using std::cout;    // preferred to: using namespace std;
using std::endl;
int Maximum (int, int);   // overloaded function prototypes
float Maximum (float, float); 
int main()
{
    char a = 'A', b = 'B';
    float x = 5.7, y = 9.89;
    int m = 6, n = 10;
    cout << "The max is: " << Maximum(a, b) << endl;
    cout << "The max is: " << Maximum(x, y) << endl;
    cout << "The max is: " << Maximum(m, n) << endl;
    // The following (ambiguous) line generates a compiler 
// error - there are two equally good fn. candidates 
    // cout << "The maximum is: " << Maximum(a, y) << endl;
    // We can force a choice by using an explicit typecast
    cout << "The max is: " << 
             Maximum(static_cast<float>(a), y) << endl;
    return 0;
}
int Maximum (int arg1, int arg2)    // function definition
{
    return (arg1 > arg2)? arg1 : arg2;
}
float Maximum (float arg1, float arg2)  // overloaded fn.
{                                    
    return (arg1 > arg2)? arg1 : arg2;
}
```

在这个先前的简单示例中，`Maximum()` 的两个版本都被原型化和定义了。这些函数是重载的；注意，它们的名称相同，但它们使用的参数类型不同。还要注意，它们的返回类型不同；然而，由于返回类型不是函数签名的一部分，因此返回类型不需要匹配。

接下来，在 `main()` 中，声明并初始化了两个类型为 `char`、`int` 和 `float` 的变量。接下来，调用 `Maximum(a, b)` 并将两个 `char` 实际参数转换为整数（使用它们的 ASCII 等价物）以匹配此函数的 `Maximum(int, int)` 版本。这是与 `a` 和 `b` 的 `char` 参数类型最接近的匹配：`Maximum(int, int)` 与 `Maximum(float, float)`。然后，调用 `Maximum(x, y)` 并使用两个浮点数，这个调用将正好匹配此函数的 `Maximum(float, float)` 版本。同样，`Maximum(m, n)` 将被调用，并将完美匹配此函数的 `Maximum(int, int)` 版本。

现在，注意下一个函数调用（这并非巧合，它是被注释掉的）：`Maximum(a, y)`。在这里，第一个实际参数完美匹配 `Maximum(int, int)` 中的第一个参数，而第二个实际参数完美匹配 `Maximum(float, float)` 中的第二个参数。对于不匹配的参数，可以应用类型转换——但并没有！相反，这个函数调用被编译器标记为歧义函数调用，因为任一重载函数都可能是一个合适的匹配。

在代码行`Maximum((float) a, y)`中，注意函数调用`Maximum((float) a, y)`强制对第一个实际参数`a`进行显式类型转换，解决了调用哪个重载函数的潜在歧义。现在参数`a`被转换成`float`类型，这个函数调用很容易匹配`Maximum(float, float)`，不再被认为是歧义的。类型转换可以是一种解决这类疯狂情况的工具。

这里是伴随我们示例的输出：

```cpp
The maximum is: 66
The maximum is: 9.89
The maximum is: 10
The maximum is: 65
```

# 摘要

在本章中，我们学习了 C++中一些额外的非面向对象特性，这些特性是构建 C++面向对象特性的必要基石。这些语言需求包括使用`const`限定符、理解函数原型、在原型中使用默认值、函数重载、标准类型转换如何影响重载函数的选择，以及可能出现的歧义（以及如何解决）。

非常重要的是，你现在可以向前推进到下一章，我们将详细探讨使用指针进行间接寻址。你在本章积累的事实技能将帮助你更容易地导航每一章，确保你准备好轻松应对从*第五章*开始的 OO 概念，即*探索类的细节*。

记住，C++是一种比大多数其他语言都有更多灰色区域的语言。你通过技能集积累的微妙差异将提高你作为 C++开发者的价值——不仅能导航和理解现有的微妙代码，而且能创建易于维护的代码。

# 问题

1.  函数的签名是什么，函数的签名在 C++中与名称修饰有何关系？你认为这如何帮助编译器内部处理重载函数？

1.  编写一个小型的 C++程序，提示用户输入有关`Student`的信息，并打印出这些数据。使用以下步骤编写你的代码：

    1.  使用`class`或`struct`创建一个`Student`数据类型。`Student`信息至少应包括`firstName`、`lastName`、`gpa`以及`Student`注册的`currentCourse`。这些信息可以存储在一个简单的类中。你可以使用`char`数组来表示字符串字段，因为我们还没有介绍指针，或者你可以（更推荐）使用`string`类型。此外，你可以在`main()`函数中读取这些信息，而不是创建一个单独的函数来读取数据（因为后者将需要了解指针或引用）。请勿使用全局变量（即外部变量）。

    1.  创建一个函数来打印出`Student`的所有数据。记住要为这个函数原型。在函数原型中使用`gpa`的默认值`4.0`。以两种方式调用这个函数：一次显式传递每个参数，一次使用默认的`gpa`。

    1.  现在，重载 print 函数，使其能够打印出选定的数据（例如，`lastName`和`gpa`），或者使用这个函数的版本，它接受一个`Student`作为参数（但不能是`Student`的指针或引用——我们稍后会这样做）。记得要为这个函数编写原型。

    1.  使用 iostreams 进行输入输出。
