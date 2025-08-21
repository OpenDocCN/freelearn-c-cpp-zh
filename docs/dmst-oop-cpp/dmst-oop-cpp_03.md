# 第二章：添加语言必需性

本章将介绍 C++的非面向对象特性，这些特性是 C++面向对象特性的重要基石。本章介绍的特性代表了从本章开始在本书中将被毫不犹豫地使用的主题。C++是一门笼罩在灰色地带的语言；从本章开始，您将不仅熟悉语言特性，还将熟悉语言的微妙之处。本章的目标将是从一个普通的 C++程序员的技能开始，使其能够成功地在创建可维护的代码的同时在语言的微妙之处中操作。

在本章中，我们将涵盖以下主要主题：

+   `const`限定符

+   函数原型

+   函数重载

通过本章结束时，您将了解非面向对象的特性，如`const`限定符，函数原型（包括使用默认值），函数重载（包括标准类型转换如何影响重载函数选择并可能创建潜在的歧义）。许多这些看似简单的主题包括各种有趣的细节和微妙之处。这些技能将是成功地继续阅读本书后续章节所必需的。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02)。每个完整的程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节号相对应，后跟破折号，再跟随所在章节中的示例编号。例如，*第二章*，*添加语言必需性*中的第一个完整程序可以在名为`Chp2-Ex1.cpp`的文件中的`Chapter02`子目录中找到上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/3cTYgnB`](https://bit.ly/3cTYgnB)。

# 使用 const 限定符

在本节中，我们将向变量添加`const`限定符，并讨论如何将其添加到函数的输入参数和返回值中。随着我们在 C++语言中的进一步学习，`const`限定符将被广泛使用。使用`const`可以使值被初始化，但永远不会再次修改。函数可以声明它们不会修改其输入参数，或者它们的返回值只能被捕获（但不能被修改）使用`const`。`const`限定符有助于使 C++成为一种更安全的语言。让我们看看`const`的实际应用。

## 常量变量

一个`const`限定的变量是一个必须被初始化的变量，永远不能被赋予新值。将`const`和变量一起使用似乎是一个悖论-`const`意味着不改变，然而变量的概念本质上是持有不同的值。尽管如此，拥有一个在运行时可以确定其唯一值的强类型检查变量是有用的。关键字`const`被添加到变量声明中。

让我们在以下程序中考虑一些例子。我们将把这个程序分成两个部分，以便更有针对性地解释，但是完整的程序示例可以在以下链接中找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace std;
// simple const variable declaration and initialization
const int MAX = 50; 
int minimum(int a, int b)  // function definition with
{                          // formal parameters
    return (a < b)? a : b;   // conditional operator ?: 
}
```

在前面的程序段中，请注意我们在数据类型之前使用`const`限定符声明变量。在这里，`const int MAX = 50;`简单地将`MAX`初始化为`50`。`MAX`不能通过赋值在代码中后期修改。按照惯例，简单的`const`限定变量通常大写。接下来，我们有函数`minimum()`的定义；请注意在这个函数体中使用了三元条件运算符`?:`。接下来，让我们继续查看`main()`函数的主体，继续进行本程序的其余部分：

```cpp
int main()
{
    int x, y;
    cout << "Enter two values: ";
    cin >> x >> y;
    const int MIN = minimum(x, y);  // const var initialized 
                             // with a function's return value
    cout << "Minimum is: " << MIN << endl;
    char bigName[MAX];      // const var used to size an array
    cout << "Enter a name: ";
    cin >> setw(MAX) >> bigName;
    const int NAMELEN = strlen(bigName); // another const
    cout << "Length of name: " << NAMELEN << endl;
    return 0;
}
```

在`main()`中，让我们考虑代码的顺序，提示用户将“输入两个值：”分别存入变量`x`和`y`中。在这里，我们调用函数`minimum(x,y)`，并将我们刚刚使用`cin`和提取运算符`>>`读取的两个值`x`和`y`作为实际参数传递。请注意，除了`MIN`的`const`变量声明之外，我们还使用函数调用`minimum()`的返回值初始化了`MIN`。重要的是要注意，设置`MIN`被捆绑为单个声明和初始化。如果这被分成两行代码--变量声明后跟一个赋值--编译器将会标记一个错误。`const`变量只能在声明后用一个值初始化，不能在声明后赋值。

在上面的最后一段代码中，请注意我们使用`MAX`（在这个完整程序示例的早期部分定义）来定义固定大小数组`bigName`的大小：`char bigName[MAX];`。然后，我们在`setw(MAX)`中进一步使用`MAX`来确保我们在使用`cin`和提取运算符`>>`读取键盘输入时不会溢出`bigName`。最后，我们使用函数`strlen(bigname)`的返回值初始化变量`const int NAMELEN`，并使用`cout`打印出这个值。

上面完整程序示例的输出如下：

```cpp
Enter two values: 39 17
Minimum is: 17
Enter a name: Gabby
Length of name: 5
```

现在我们已经看到了如何对变量进行`const`限定，让我们考虑对函数进行`const`限定。

## 函数的 const 限定

关键字`const`也可以与函数一起使用。`const`限定符可以用于参数中，表示参数本身不会被修改。这是一个有用的特性--函数的调用者将了解到以这种方式限定的输入参数不会被修改。然而，因为非指针（和非引用）变量被作为“按值”传递给函数，作为实际参数在堆栈上的副本，对这些固有参数的`const`限定并没有任何意义。因此，对标准数据类型的参数进行`const`限定是不必要的。

相同的原则也适用于函数的返回值。函数的返回值可以被`const`限定，然而，除非返回一个指针（或引用），作为返回值传回堆栈的项目是一个副本。因此，当返回类型是指向常量对象的指针时，`const`限定返回值更有意义（我们将在*第三章*中介绍，*间接寻址：指针*及以后内容）。作为`const`的最后一个用途，我们可以在类的 OO 细节中使用这个关键字，以指定特定成员函数不会修改该类的任何数据成员。我们将在*第五章*中探讨这种情况，*详细探讨类*。

现在我们了解了`const`限定符用于变量，并看到了与函数一起使用`const`的潜在用途，让我们继续前进到本章的下一个语言特性：函数原型。

# 使用函数原型

在本节中，我们将研究函数原型的机制，比如在文件中的必要放置和跨多个文件以实现更大的程序灵活性。我们还将为原型参数添加可选名称，并了解我们为什么可以选择向 C++原型添加默认值。函数原型确保了 C++代码的强类型检查。

在继续讨论函数原型之前，让我们花一点时间回顾一些必要的编程术语。**函数定义**指的是组成函数的代码主体。而函数的声明（也称为**前向声明**）仅仅是引入了函数名及其返回类型和参数类型，前向声明允许编译器通过将调用与前向声明进行比较而执行强类型检查。前向声明很有用，因为函数定义并不总是在函数调用之前出现在一个文件中；有时，函数定义出现在与它们的调用分开的文件中。

## 定义函数原型

**函数原型**是对函数的前向声明，描述了函数应该如何被正确调用。原型确保了函数调用和定义之间的强类型检查。函数原型包括：

+   函数的返回类型

+   函数的名称

+   函数的类型和参数数量

函数原型允许函数调用在函数的定义之前，或者允许调用存在于不同的文件中的函数。让我们来看一个简单的例子：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex2.cpp)

```cpp
#include <iostream>
using namespace std;
int minimum(int, int);     // function prototype

int main()
{
    int x = 5, y = 89;
    // function call with actual parameters
    cout << minimum(x, y) << endl;     
    return 0;                          
}
int minimum(int a, int b)  // function definition with
{                          // formal parameters
    return (a < b)? a : b;  
}
```

注意，我们在上面的例子中在开头原型了`int minimum(int, int);`。这个原型让编译器知道对`minimum()`的任何调用都应该带有两个整数参数，并且应该返回一个整数值（我们将在本节后面讨论类型转换）。

接下来，在`main()`函数中，我们调用函数`minimum(x, y)`。此时，编译器检查函数调用是否与前面提到的原型匹配，包括参数的类型和数量以及返回类型。也就是说，这两个参数是整数（或者可以轻松转换为整数），返回类型是整数（或者可以轻松转换为整数）。返回值将被用作`cout`打印的值。最后，在文件中定义了函数`minimum()`。如果函数定义与原型不匹配，编译器将引发错误。

原型的存在允许对给定函数的调用在编译器看到函数定义之前进行完全的类型检查。上面的例子当然是为了演示这一点而捏造的；我们也可以改变`minimum()`和`main()`在文件中出现的顺序。然而，想象一下`minimum()`的定义包含在一个单独的文件中（更典型的情况）。在这种情况下，原型将出现在调用这个函数的文件的顶部（以及头文件的包含），以便函数调用可以完全根据原型进行类型检查。

在上述的多文件情况下，包含函数定义的文件将被单独编译。然后链接器的工作是确保当这两个文件链接在一起时，函数定义和原型匹配，以便链接器可以解析对这样的函数调用的任何引用。如果原型和定义不匹配，链接器将无法将代码的这两部分链接成一个编译单元。

让我们来看一下这个例子的输出：

```cpp
5
```

现在我们了解了函数原型基础知识，让我们看看如何向函数原型添加可选参数名称。

## 在函数原型中命名参数

函数原型可以选择包含名称，这些名称可能与形式参数或实际参数列表中的名称不同。参数名称会被编译器忽略，但通常可以增强可读性。让我们重新看一下我们之前的示例，在函数原型中添加可选参数名称。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex3.cpp)

```cpp
#include <iostream>
using namespace std;
int minimum(int arg1, int arg2);    // function prototype with
                                    // optional argument names
int main()
{
    int x = 5, y = 89;
    cout << minimum(x, y) << endl;   // function call
    return 0;
}
int minimum(int a, int b)            // function definition
{
    return (a < b)? a : b;  
}
```

这个示例几乎与前面的示例相同。但是，请注意函数原型包含了命名参数`arg1`和`arg2`。这些标识符会被编译器立即忽略。因此，这些命名参数不需要与函数的形式参数或实际参数匹配，仅仅是为了增强可读性而可选地存在。

与上一个示例相同，此示例的输出如下：

```cpp
5
```

接下来，让我们通过向函数原型添加一个有用的功能来继续我们的讨论：默认值。

## 向函数原型添加默认值

**默认值**可以在函数原型中指定。这些值将在函数调用中没有实际参数时使用，并将作为实际参数本身。默认值必须符合以下标准：

+   必须从右到左在函数原型中指定默认值，不能省略任何值。

+   实际参数在函数调用中从左到右进行替换；因此，在原型中从右到左指定默认值的顺序是重要的。

函数原型可以有全部、部分或没有默认值填充，只要默认值符合上述规定。

让我们看一个使用默认值的示例：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex4.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex4.cpp)

```cpp
#include <iostream>
using namespace std;
int minimum(int arg1, int arg2 = 100000);  // fn. prototype
                                    // with one default value
int main()
{
    int x = 5, y = 89;
    cout << minimum(x) << endl; // function call with only
                                // one argument (uses default)
    cout << minimum(x, y) << endl; // no default values used
    return 0;
}
int minimum(int a, int b)            // function definition
{
    return (a < b)? a : b;  
}
```

在这个示例中，请注意在函数原型`int minimum(int arg1, int arg2 = 100000);`中向最右边的参数添加了一个默认值。这意味着当从`main()`中调用`minimum`时，可以使用一个参数调用：`minimum(x)`，也可以使用两个参数调用：`minimum(x, y)`。当使用一个参数调用`minimum()`时，单个参数绑定到函数的形式参数中的最左边参数，而默认值绑定到形式参数列表中的下一个顺序参数。但是，当使用两个参数调用`minimum()`时，实际参数都绑定到函数中的形式参数；默认值不会被使用。

这个示例的输出如下：

```cpp
5
5
```

现在我们已经掌握了函数原型中的默认值，让我们通过在各种程序作用域中使用不同的默认值来扩展这个想法。

## 在不同作用域中使用不同默认值进行原型化

函数可以在不同的作用域中使用不同的默认值进行原型化。这允许函数在多个应用程序中以通用方式构建，并通过原型在多个代码部分中进行定制。

这是一个示例，演示了相同函数的多个原型（在不同的作用域中）使用不同的默认值。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex5.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex5.cpp)

```cpp
#include <iostream>
using namespace std;
int minimum(int, int);   // standard function prototype
void function1(int x)
{   
    int minimum(int arg1, int arg2 = 500); // local prototype
                                           // with default value
    cout << minimum(x) << endl; 
}
void function2(int x)
{
    int minimum(int arg1, int arg2 = 90);  // local prototype
                                           // with default value
    cout << minimum(x) << endl; 
}

int minimum(int a, int b)            // function definition
{ 
    return (a < b)? a : b;   
}
int main()
{
    function1(30);    
    function2(450);
    return 0;
}
```

在这个示例中，请注意在文件顶部附近原型化了`int minimum(int, int);`，然后注意在`function1()`的更局部范围内重新定义了`minimum()`，作为`int minimum(int arg1, int arg2 = 500);`，为其最右边的参数指定了默认值`500`。同样，在`function2()`的范围内，函数`minimum()`被重新定义为：`int minimum(int arg1, int arg2 = 90);`，为其最右边的参数指定了默认值`90`。当在`function1()`或`function2()`中调用`minimum()`时，将分别使用每个函数范围内的本地原型-每个都有自己的默认值。

通过这种方式，程序的特定部分可以很容易地使用默认值进行定制，这些默认值在应用程序的特定部分可能是有意义的。但是，请确保*仅*在调用函数的范围内使用重新定义函数的个性化默认值，以确保这种定制可以轻松地包含在非常有限的范围内。永远不要在全局范围内重新定义具有不同默认值的函数原型-这可能会导致意外和容易出错的结果。

示例的输出如下：

```cpp
30
90
```

在单个和多个文件中探索了函数原型的默认用法，使用原型中的默认值，并在不同范围内重新定义函数以及使用个别默认值后，我们现在可以继续进行本章的最后一个主要主题：函数重载。

# 理解函数重载

C++允许两个或更多个函数共享相似的目的，但在它们所接受的参数类型或数量上有所不同，以相同的函数名称共存。这被称为**函数重载**。这允许进行更通用的函数调用，让编译器根据使用函数的变量（对象）的类型选择正确的函数版本。在本节中，我们将在函数重载的基础上添加默认值，以提供灵活性和定制。我们还将学习标准类型转换如何影响函数重载，以及可能出现的歧义（以及如何解决这些类型的不确定性）。

## 学习函数重载的基础知识

当存在两个或更多个同名函数时，这些相似函数之间的区别因素将是它们的签名。通过改变函数的签名，两个或更多个在同一命名空间中具有相同名称的函数可以存在。函数重载取决于函数的签名，如下所示：

+   **函数的签名**指的是函数的名称，以及其参数的类型和数量。

+   函数的返回类型不包括在其签名中。

+   两个或更多个具有相同目的的函数可以共享相同的名称，只要它们的签名不同。

函数的签名有助于为每个函数提供一个内部的“混淆”名称。这种编码方案保证每个函数在编译器内部都有唯一的表示。

让我们花几分钟来理解一个稍微复杂的示例，其中将包含函数重载。为了简化解释，这个示例被分成了三个部分；然而，完整的程序可以在以下链接中找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex6.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex6.cpp)

```cpp
#include <iostream>
#include <cmath>
using namespace std;
const float PI = 3.14159;
class Circle        // user defined type declarations
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
void display(Circle);     // 'overloaded' function prototypes
void display(Rectangle);  // since they differ in signature
```

在这个例子的开头，注意我们用 `#include <cmath>` 包含了 math 库，以便访问基本的数学函数，比如 `pow()`。接下来，注意 `Circle` 和 `Rectangle` 的类定义，每个类都有相关的数据成员（`Circle` 的 `radius` 和 `area`；`Rectangle` 的 `length`、`width` 和 `area`）。一旦这些类型被定义，就会显示两个重载的显示函数的原型。由于这两个显示函数的原型使用了用户定义的类型 `Circle` 和 `Rectangle`，所以很重要的是 `Circle` 和 `Rectangle` 必须先被定义。现在，让我们继续查看 `main()` 函数的主体部分：

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
    display(myCircle);     // invoke: void display(Circle)
    display(myRect);       // invoke: void display(Rectangle)
    display(mySquare);
    return 0;
}
```

现在，在 `main()` 函数中，我们声明了一个 `Circle` 类型的变量和两个 `Rectangle` 类型的变量。然后我们使用适当的值在 `main()` 中使用点运算符 `.` 加载了每个变量的数据成员。接下来，在 `main()` 中，有三次对 `display()` 的调用。第一个函数调用 `display(myCircle)`，将调用以 `Circle` 作为形式参数的 `display()` 版本，因为传递给这个函数的实际参数实际上是用户定义的类型 `Circle`。接下来的两个函数调用 `display(myRect)` 和 `display(mySquare)`，将调用重载版本的 `display()`，因为这两个调用中传递的实际参数本身就是 `Rectangle`。让我们通过查看 `display()` 的两个函数定义来完成这个程序：

```cpp
void display (Circle c)
{
   cout << "Circle with radius " << c.radius;
   cout << " has an area of " << c.area << endl; 
}

void display (Rectangle r)
{
   cout << "Rectangle with length " << r.length;
   cout << " and width " << r.width;
   cout << " has an area of " << r.area << endl; 
}
```

请注意在这个示例的最后部分，定义了 `display()` 的两个版本。其中一个函数以 `Circle` 作为形式参数，重载版本以 `Rectangle` 作为形式参数。每个函数体访问特定于其形式参数类型的数据成员，但每个函数的整体功能都是相似的，因为在每种情况下都显示了一个特定的形状（`Circle` 或 `Rectangle`）。

让我们来看看这个完整程序示例的输出：

```cpp
Circle with radius 5 has an area of 78.5397
Rectangle with length 2 and width 4 has an area of 8
Rectangle with length 4 and width 4 has an area of 16
```

接下来，让我们通过理解标准类型转换如何允许一个函数被多个数据类型使用，来扩展我们对函数重载的讨论。这可以让函数重载更有选择性地使用。

## 通过标准类型转换消除过多的重载

编译器可以自动将基本语言类型从一种类型转换为另一种类型。这使得语言可以提供一个更小的操作符集来操作标准类型，而不需要更多的操作符。标准类型转换也可以消除函数重载的需要，当保留函数参数的确切数据类型不是至关重要的时候。标准类型之间的提升和降级通常是透明处理的，在包括赋值和操作的表达式中，不需要显式转换。

这是一个示例，说明了简单的标准类型转换。这个例子不包括函数重载。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex7.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex7.cpp)

```cpp
#include <iostream>
using namespace std; 
int maximum(double, double);      // function prototype

int main()
{
    int result;
    int m = 6, n = 10;
    float x = 5.7, y = 9.89;

    result =  maximum(x, y); 
    cout << "Result is: " << result << endl;
    cout << "The maximum is: " << maximum(m, n) << endl;
    return 0;
}
int maximum(double a, double b)  // function definition
{
    return (a > b)? a : b;
}
```

在这个例子中，`maximum()` 函数以两个双精度浮点数作为参数，并将结果作为 `int` 返回。首先，注意在程序的顶部附近原型化了 `int maximum(double, double);`，并且在同一个文件的底部定义了它。

现在，在`main（）`函数中，请注意我们定义了三个 int 变量：`result`，`a`和`x`。后两者分别初始化为`6`和`10`的值。我们还定义并初始化了两个浮点数：`float x = 5.7, y = 9.89;`。在第一次调用`maximum（）`函数时，我们使用`x`和`y`作为实际参数。这两个浮点数被提升为双精度浮点数，并且函数被按预期调用。

这是标准类型转换的一个例子。让我们注意`int maximum(double, double)`的返回值是一个整数 - 而不是双精度。这意味着从这个函数返回的值（形式参数`a`或`b`）将首先被截断为整数，然后作为返回值使用。这个返回值被整洁地赋给了`result`，它在`main（）`中被声明为`int`。这些都是标准类型转换的例子。

接下来，`maximum（）`被调用，实际参数为`m`和`n`。与前一个函数调用类似，整数`m`和`n`被提升为双精度，并且函数被按预期调用。返回值也将被截断为`int`，并且该值将作为整数传递给`cout`进行打印。

这个示例的输出是：

```cpp
Result is: 9
The maximum is: 10
```

现在我们了解了函数重载和标准类型转换的工作原理，让我们来看一个情况，其中两者结合可能会产生一个模棱两可的函数调用。

## 函数重载和类型转换引起的歧义

当调用函数时，形式和实际参数在类型上完全匹配时，不会出现关于应该调用哪个重载函数的歧义 - 具有完全匹配的函数是显而易见的选择。然而，当调用函数时，形式和实际参数在类型上不同时，可能需要对实际参数进行标准类型转换。然而，在形式和实际参数类型不匹配且存在重载函数的情况下，编译器可能难以选择哪个函数应该被选为最佳匹配。在这些情况下，编译器会生成一个错误，指示可用的选择与函数调用本身是模棱两可的。显式类型转换或在更局部的范围内重新原型化所需的选择可以帮助纠正这些否则模棱两可的情况。

让我们回顾一个简单的函数，说明函数重载、标准类型转换和潜在的歧义。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex8.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter02/Chp2-Ex8.cpp)

```cpp
#include <iostream>
using namespace std;
int maximum (int, int);     // overloaded function prototypes
float maximum (float, float); 
int main()
{
    char a = 'A', b = 'B';
    float x = 5.7, y = 9.89;
    int m = 6, n = 10;
    cout << "The max is: " << maximum(a, b) << endl;
    cout << "The max is: " << maximum(x, y) << endl;
    cout << "The max is: " << maximum(m, n) << endl;
    // The following (ambiguous) line generates a compiler 
    // error since there are two equally good fn. candidates 
    // cout << "The maximum is: " << maximum(a, y) << endl;
    // We can force a choice by using an explicit typecast
    cout << "The max is: " << maximum((float)a, y) << endl;
    return 0;
}
int maximum (int arg1, int arg2)        // function definition
{
    return (arg1 > arg2)? arg1 : arg2;
}
float maximum (float arg1, float arg2)  // overloaded function
{                                    
    return (arg1 > arg2)? arg1 : arg2;
}
```

在前面的简单示例中，`maximum（）`的两个版本都被原型化和定义。这些函数被重载；请注意它们的名称相同，但它们在使用的参数类型上不同。还要注意它们的返回类型不同；但是，由于返回类型不是函数签名的一部分，因此返回类型不需要匹配。

接下来，在`main（）`中，声明并初始化了两个`char`，`int`和`float`类型的变量。接下来，调用`maximum（a，b）`，两个`char`实际参数被转换为整数（使用它们的 ASCII 等价物）以匹配该函数的`maximum(int, int)`版本。这是与`a`和`b`的`char`参数类型最接近的匹配：`maximum(int, int)`与`maximum(float, float)`。然后，使用两个浮点数调用`maximum（x，y）`，这个调用将完全匹配该函数的`maximum(float, float)`版本。类似地，`maximum（m，n）`将被调用，并且将完全匹配该函数的`maximum(int, int)`版本。

现在，注意下一个函数调用（不巧的是，它被注释掉了）：`maximum(a, y)`。在这里，第一个实际参数完全匹配 `maximum(int, int)` 中的第一个参数，但第二个实际参数完全匹配 `maximum(float, float)` 中的第二个参数。对于不匹配的参数，可以应用类型转换——但没有！相反，编译器将此函数调用标记为模棱两可的函数调用，因为任何一个重载函数都可能是一个合适的匹配。

在代码行 `maximum((float) a, y)` 上，注意到对 `maximum((float) a, y)` 的函数调用强制对第一个实际参数 `a` 进行显式类型转换，解决了调用哪个重载函数的潜在歧义。现在，参数 `a` 被转换为 `float`，这个函数调用很容易匹配 `maximum(float, float)`，不再被视为模棱两可。类型转换可以是一个工具，用来消除这类疯狂情况的歧义。

以下是与我们示例配套的输出：

```cpp
The maximum is: 66
The maximum is: 9.89
The maximum is: 10
The maximum is: 65
```

# 总结

在本章中，我们学习了额外的非面向对象的 C++ 特性，这些特性是构建 C++ 面向对象特性所必需的基本组成部分。这些语言必需品包括使用 `const` 限定符，理解函数原型，使用原型中的默认值，函数重载，标准类型转换如何影响重载函数的选择，以及可能出现的歧义如何解决。

非常重要的是，您现在已经准备好进入下一章，我们将在其中详细探讨使用指针进行间接寻址。您在本章积累的事实技能将帮助您更轻松地导航每一个逐渐更详细的章节，以确保您准备好轻松应对从*第五章* 开始的面向对象概念，*详细探索类*。

请记住，C++ 是一种充满了比大多数其他语言更多灰色地带的语言。您积累的微妙细微之处将增强您作为 C++ 开发人员的价值——一个不仅可以导航和理解现有微妙代码的人，还可以创建易于维护的代码。

# 问题

1.  函数的签名是什么，函数的签名如何与 C++ 中的名称修饰相关联？您认为这如何促进编译器内部处理重载函数？

1.  编写一个小的 C++ 程序，提示用户输入有关 `学生` 的信息，并打印出数据。

a. `学生` 信息应至少包括名字、姓氏、GPA 和 `学生` 注册的当前课程。这些信息可以存储在一个简单的类中。您可以利用数组来表示字符串字段，因为我们还没有涉及指针。此外，您可以在主函数中读取这些信息，而不是创建一个单独的函数来读取数据（因为后者需要指针或引用的知识）。请不要使用全局（即 extern 变量）。

b. 创建一个函数来打印 `学生` 的所有数据。记得对这个函数进行原型声明。在这个函数的原型中，使用默认值 4.0 作为 GPA。以两种方式调用这个函数：一次显式传入每个参数，一次使用默认的 GPA。

c. 现在，重载 `Print` 函数，其中一个打印出选定的数据（即姓氏和 GPA），或者使用接受 `Student` 作为参数的版本的函数（但不是 `Student` 的指针或引用——我们稍后会做）。记得对这个函数进行原型声明。

d. 使用 iostream 进行 I/O。
