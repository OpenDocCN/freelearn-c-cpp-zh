# 第四章：间接寻址：引用

本章将探讨如何在 C++中利用引用。引用通常可以用作间接寻址的替代方案，但并非总是如此。尽管您在上一章中使用指针有间接寻址的经验，我们将从头开始理解 C++引用。

引用和指针一样，是您必须能够轻松使用的语言特性。许多其他语言使用引用进行间接寻址，而不需要像 C++那样深入理解才能正确使用指针和引用。与指针一样，您会经常在其他程序员的代码中看到引用的使用。与指针相比，使用引用在编写应用程序时提供了更简洁的表示方式，这可能会让您感到满意。

遗憾的是，在所有需要间接寻址的情况下，引用不能替代指针。因此，在 C++中，深入理解使用指针和引用进行间接寻址是成功创建可维护代码的必要条件。

本章的目标是通过了解如何使用 C++引用作为替代方案来补充您对使用指针进行间接寻址的理解。了解两种间接寻址技术将使您成为一名更优秀的程序员，轻松理解和修改他人的代码，并自己编写原始、成熟和有竞争力的 C++代码。

在本章中，我们将涵盖以下主要主题：

+   引用基础 - 声明、初始化、访问和引用现有对象

+   将引用用作函数的参数和返回值

+   在引用中使用 const 限定符

+   理解底层实现，以及引用不能使用的情况

在本章结束时，您将了解如何声明、初始化和访问引用；您将了解如何引用内存中现有的对象。您将能够将引用用作函数的参数，并了解它们如何作为函数的返回值使用。

您还将了解 const 限定符如何适用于引用作为变量，并且如何与函数的参数和返回类型一起使用。您将能够区分引用何时可以替代指针，以及它们不能替代指针的情况。这些技能将是成功阅读本书后续章节的必要条件。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master/Chapter04)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节编号相对应，后跟破折号，再跟随所在章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp4-Ex1.cpp`的文件中的`Chapter04`子目录中找到。

本章的 CiA 视频可在以下链接观看：[`bit.ly/2OM7GJP`](https://bit.ly/2OM7GJP)

# 理解引用基础

在本节中，我们将重新讨论引用基础，并介绍适用于引用的运算符，如引用运算符`&`。我们将使用引用运算符`&`来建立对现有变量的引用。与指针变量一样，引用变量指向在其他地方定义的内存。

使用引用变量允许我们使用比指针间接访问内存时更简单的符号。许多程序员欣赏引用与指针变量的符号的清晰度。但是，在幕后，内存必须始终被正确分配和释放；被引用的一部分内存可能来自堆。程序员无疑需要处理指针来处理其整体代码的一部分。

我们将分辨引用和指针何时可以互换使用，何时不可以。让我们从声明和使用引用变量的基本符号开始。

## 声明、初始化和访问引用

让我们从引用变量的含义开始。C++中的`&`。引用必须在声明时初始化，并且永远不能被分配给引用另一个对象。引用和初始化器必须是相同类型。由于引用和被引用的对象共享相同的内存，任一变量都可以用来修改共享内存位置的内容。

引用变量，在幕后，可以与指针变量相比较——因为它保存了它引用的变量的地址。与指针变量不同，引用变量的任何使用都会自动取消引用变量以转到它包含的地址；取消引用运算符`*`在引用中是不需要的。取消引用是自动的，并且隐含在每次使用引用变量时。

让我们看一个说明引用基础的例子：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex1.cpp)

```cpp
#include <iostream>
using namespace std;
int main()
{
    int x = 10;
    int *p = new int;    // allocate memory for ptr variable
    *p = 20;             // dereference and assign value 
    int &refInt1 = x;  // reference to an integer
    int &refInt2 = *p; // also a reference to an integer
    cout << x << " " << *p << " ";
    cout << refInt1 << " " << refInt2 << endl;
    x++;      // updates x and refInt1
    (*p)++;   // updates *p and refInt2
    cout << x << " " << *p << " ";
    cout << refInt1 << " " << refInt2 << endl;
    refInt1++;    // updates refInt1 and x
    refInt2++;    // updates refInt2 and *p
    cout << x << " " << *p << " ";
    cout << refInt1 << " " << refInt2 << endl;
    return 0;
}
```

在前面的例子中，我们首先声明并初始化`int x = 10;`，然后声明并分配`int *p = new int;`。然后我们将整数值 20 分配给`*p`。

接下来，我们声明并初始化两个引用变量，`refInt1`和`refInt2`。在第一个引用声明和初始化中，`int &refInt1 = x;`，我们建立`refInt1`引用变量指向变量`x`。从右向左阅读引用声明有助于理解。在这里，我们说要使用`x`来初始化`refInt1`，它是一个整数的引用（`&`）。注意初始化器`x`是一个整数，并且`refInt1`声明为整数的引用；它们的类型匹配。这很重要。如果类型不同，代码将无法编译。同样，声明和初始化`int &refInt2 = *p;`也将`refInt2`建立为整数的引用。哪一个？由`p`指向的那个。这就是为什么使用`*`对`p`进行取消引用以获得整数本身。

现在，我们打印出`x`、`*p`、`refInt1`和`refInt2`；我们可以验证`x`和`refInt1`的值相同为`10`，而`*p`和`refInt2`的值也相同为`20`。

接下来，使用原始变量，我们将`x`和`*p`都增加一。这不仅增加了`x`和`*p`的值，还增加了`refInt1`和`refInt2`的值。重复打印这四个值，我们再次注意到`x`和`refInt1`的值为`11`，而`*p`和`refInt2`的值为`21`。

最后，我们使用引用变量来增加共享内存。我们将`refInt1`和`*refint2`都增加一，这也增加了原始变量`x`和`*p`的值。这是因为内存是原始变量和引用到该变量的相同。也就是说，引用可以被视为原始变量的别名。我们通过再次打印这四个变量来结束程序。

以下是输出：

```cpp
10 20 10 20
11 21 11 21
12 22 12 22
```

重要提示

记住，引用变量必须初始化为它将引用的变量。引用永远不能被分配给另一个变量。引用和它的初始化器必须是相同类型。

现在我们已经掌握了如何声明简单引用，让我们更全面地看一下引用现有对象，比如用户定义类型的对象。

## 引用现有的用户定义类型的对象

如果定义一个`struct`或`class`类型的对象的引用，那么被引用的对象可以简单地使用`.`（成员选择运算符）访问。同样，不需要（就像指针一样）首先使用取消引用运算符来访问被引用的对象，然后选择所需的成员。

让我们看一个引用用户定义类型的例子：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex2.cpp)

```cpp
#include <iostream>
#include <cstring>
using namespace std;
class Student
{
public:
    char name[20];
    float gpa;
};
int main()
{
    Student s1;
    Student &sRef = s1;  // establish a reference to s1
    strcpy(s1.name, "Katje Katz");   // fill in the data
    s1.gpa = 3.75;
    cout << s1.name << " has GPA: " << s1.gpa << endl; 
    cout << sRef.name << " has GPA: " << sRef.gpa << endl; 
    strcpy(sRef.name, "George Katz");  // change the data
    sRef.gpa = 3.25;
    cout << s1.name << " has GPA: " << s1.gpa << endl; 
    cout << sRef.name << " has GPA: " << sRef.gpa << endl; 
    return 0;
}
```

在程序的第一部分中，我们使用`class`定义了一个用户定义类型`Student`。接下来，我们使用`Student s1;`声明了一个类型为`Student`的变量`s1`。现在，我们使用`Student &sRef = s1;`声明并初始化了一个`Student`的引用。在这里，我们声明`sRef`引用特定的`Student`，即`s1`。注意，`s1`是`Student`类型，而`sRef`的引用类型也是`Student`类型。

现在，我们使用`strcpy()`加载一些初始数据到`s1`中，然后进行简单赋值。因此，这改变了`sRef`的值，因为`s1`和`sRef`引用相同的内存。也就是说，`sRef`是`S1`的别名。

我们打印出`s1`和`sRef`的各种数据成员，并注意到它们包含相同的值。

现在，我们加载新的值到`sRef`中，也使用`strcpy()`和简单赋值。同样，我们打印出`s1`和`sRef`的各种数据成员，并注意到它们的值再次发生了改变。我们可以看到它们引用相同的内存。

程序输出如下：

```cpp
Katje Katz has GPA: 3.75
Katje Katz has GPA: 3.75
George Katz has GPA: 3.25
George Katz has GPA: 3.25
```

现在，让我们通过考虑在函数中使用引用来进一步了解引用的用法。

# 使用引用与函数

到目前为止，我们已经通过使用引用来为现有变量建立别名来最小程度地演示了引用。相反，让我们提出引用的有意义用法，比如在函数调用中使用它们。我们知道 C++中的大多数函数将接受参数，并且在前几章中我们已经看到了许多示例，说明了函数原型和函数定义。现在，让我们通过将引用作为函数的参数传递，并使用引用作为函数的返回值来增进我们对函数的理解。

## 将引用作为函数的参数传递

引用可以作为函数的参数来实现按引用传递，而不是按值传递参数。引用可以减轻在所涉及的函数范围内以及调用该函数时使用指针表示的需要。对于引用的形式参数，使用对象或`.`（成员选择）表示法来访问`struct`或`class`成员。

为了修改作为参数传递给函数的变量的内容，必须使用对该参数的引用（或指针）作为函数参数。就像指针一样，当引用传递给函数时，传递给函数的是表示引用的地址的副本。然而，在函数内部，任何使用引用作为形式参数的用法都会自动隐式地取消引用，允许用户使用对象而不是指针表示。与传递指针变量一样，将引用变量传递给函数将允许修改由该参数引用的内存。

在检查函数调用时（除了其原型），如果传递给该函数的对象是按值传递还是按引用传递，这将不明显。也就是说，整个对象是否将在堆栈上复制，还是堆栈上将传递对该对象的引用。这是因为在操作引用时使用对象表示法，并且这两种情况的函数调用将使用相同的语法。

勤奋使用函数原型将解决函数定义的外观以及其参数是对象还是对象引用的神秘。请记住，函数定义可以在与该函数的任何调用分开的文件中定义，并且不容易查看。请注意，指定在函数调用中的指针不会出现这种模棱两可的情况；根据变量的声明方式，立即就能明显地知道地址被发送到函数。

让我们花几分钟来理解一个示例，说明将引用作为参数传递给函数。在这里，我们将从检查有助于以下完整程序示例的三个函数开始：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master/Chapter04/Chp4-Ex3.cpp)

```cpp
void AddOne(int &arg)   // These two functions are overloaded
{
    arg++;
}
void AddOne(int *arg)   // Overloaded function definition
{
    (*arg)++;
}
void Display(int &arg)  // This fn passes a reference to arg
{                       
    cout << arg << " " << flush;
}
```

在上面的函数中，注意`AddOne（int＆arg）`将引用作为形式参数，而`AddOne（int *arg）`将指针作为形式参数。这些函数是重载的。它们的实际参数的类型将决定调用哪个版本。

现在让我们考虑`Display（int＆arg）`。此函数接受对整数的引用。请注意，在此函数的定义中，使用对象（而不是指针）表示法来打印`arg`。

现在，让我们检查此程序的其余部分：

```cpp
#include <iostream>
using namespace std;
void AddOne(int &);    // function prototypes
void AddOne(int *);
void Display(int &);
int main()
{
    int x = 10, *y;
    y = new int;    // allocate y's memory
    *y = 15;        // dereference y to assign a value
    Display(x);
    Display(*y);

    AddOne(x);    // calls reference version (with an object) 
    AddOne(*y);   // also calls reference version 
    Display(x);   // Based on prototype, we see we are passing
    Display(*y);  // by reference. Without prototype, we might
                  // have guessed it was by value.
    AddOne(&x);   // calls pointer version
    AddOne(y);    // also calls pointer version
    Display(x);
    Display(*y);
    return 0;
}
```

请注意此程序段顶部的函数原型。它们将与先前代码段中的函数定义匹配。现在，在`main（）`函数中，我们声明并初始化`int x = 10;`并声明一个指针`int *y;`。我们使用`new（）`为`y`分配内存，然后通过解引用指针赋值`*y = 15;`。我们使用连续调用`Display（）`打印出`x`和`*y`的相应值作为基线。

接下来，我们调用`AddOne（x）`，然后是`AddOne（*y）`。变量`x`被声明为整数，`*y`指的是`y`指向的整数。在这两种情况下，我们都将整数作为实际参数传递给带有签名`void AddOne（int＆）`的重载函数版本。在这两种情况下，形式参数将在函数中更改，因为我们是通过引用传递的。当它们的相应值在接下来的连续调用`Display（）`中打印时，我们可以验证这一点。请注意，在函数调用`AddOne（x）`中，实际参数`x`的引用是在函数调用时由形式参数`arg`（在函数的参数列表中）建立的。

相比之下，我们接下来调用`AddOne（＆x）`，然后是`AddOne（y）`。在这两种情况下，我们都调用了带有签名`void AddOne（int *）`的此函数的重载版本。在每种情况下，我们都将地址的副本作为实际参数传递给函数。自然地，`＆x`是变量`x`的地址，所以这有效。同样，`y`本身就是一个地址-它被声明为指针变量。我们再次验证它们的相应值是否再次更改，使用两次`Display（）`调用。

请注意，在每次调用`Display()`时，我们都传递了一个`int`类型的对象。仅仅看函数调用本身，我们无法确定这个函数是否将以实际参数`int`（这意味着值不能被更改）或者以实际参数`int &`（这意味着值可以被修改）的形式接受。这两种情况都是可能的。然而，通过查看函数原型，我们可以清楚地看到这个函数以`int &`作为参数，从中我们可以理解参数很可能会被修改。这是函数原型有帮助的众多原因之一。

以下是完整程序示例的输出：

```cpp
10 15 11 16 12 17
```

现在，让我们通过使用引用作为函数的返回值来扩展我们对使用引用的讨论。

## 使用引用作为函数返回值

函数可以通过它们的返回语句返回对数据的引用。我们将在*第十二章*中看到需要通过引用返回数据的情况，*友元和运算符重载*。使用运算符重载，使用指针从函数返回值将不是一个选项，以保留运算符的原始语法；我们必须返回一个引用（或者一个带有 const 限定符的引用）。此外，了解如何通过引用返回对象将是有用的，因为我们在*第十四章*中探讨 C++标准模板库时会用到，*理解 STL 基础*。

当通过函数的返回语句返回引用时，请确保被引用的内存在函数调用完成后仍然存在。**不要**返回对函数内部栈上定义的局部变量的引用；这些内存将在函数完成时从栈上弹出。

由于我们无法从函数内部返回对局部变量的引用，并且因为返回对外部变量的引用是没有意义的，您可能会问我们返回的引用所指向的数据将存放在哪里？这些数据将不可避免地位于堆上。堆内存将存在于函数调用的范围之外。在大多数情况下，堆内存将在其他地方分配；然而，在很少的情况下，内存可能已经在此函数内分配。在这种情况下，当不再需要时，您必须记得放弃已分配的堆内存。

通过引用（而不是指针）变量删除堆内存将需要您使用取地址运算符`&`将所需的地址传递给`delete()`运算符。即使引用变量包含它们引用的对象的地址，但引用标识符的使用始终处于其取消引用状态。很少会出现使用引用变量删除内存的情况；我们将在*第十章*中讨论一个有意义（但很少）的例子，*实现关联、聚合和组合*。

让我们看一个例子来说明使用引用作为函数返回值的机制：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex4.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master/Chapter04/Chp4-Ex4.cpp)

```cpp
#include <iostream>
using namespace std;
int &CreateId();  // function prototype

int main()    
{
    int &id1 = CreateId();  // reference established
    int &id2 = CreateId();
    cout << "Id1: " << id1 << " Id2: " << id2 << endl;
    delete &id1;  // Here, '&' is address-of, not reference
    delete &id2;  // to calculate address to pass delete()
    return 0;
}
int &CreateId()   // Function returns a reference to an int
{
    static int count = 100;  // initialize with first id 
    int *memory = new int;
    *memory = count++;  // use count as id, then increment
    return *memory;
}
```

在这个例子中，我们看到程序顶部有`int &CreateId();`的原型。这告诉我们`CreateId()`将返回一个整数的引用。返回值必须用来初始化一个`int &`类型的变量。

在程序底部，我们看到了`CreateId()`的函数定义。请注意，此函数首先声明了一个`static`计数器，它被初始化为`100`。因为这个局部变量是`static`的，它将保留从函数调用到函数调用的值。然后我们在几行后递增这个计数器。静态变量`count`将被用作生成唯一 ID 的基础。

接下来在`CreateId()`中，我们在堆上为一个整数分配空间，并使用局部变量`memory`指向它。然后我们将`*memory`加载为`count`的值，然后为下一次进入这个函数增加`count`。然后我们使用`*memory`作为这个函数的返回值。请注意，`*memory`是一个整数（由变量`memory`在堆上指向的整数）。当我们从函数中返回它时，它作为对该整数的引用返回。当从函数中返回引用时，始终确保被引用的内存存在于函数的范围之外。

现在，让我们看看我们的`main()`函数。在这里，我们使用第一次调用`CreateId()`的返回值初始化了一个引用变量`id1`，如下所示的函数调用和初始化：`int &id1 = CreateId();`。请注意，引用`id1`在声明时必须被初始化，我们已经通过上述代码行满足了这个要求。

我们重复这个过程，用`CreateId()`的返回值初始化这个引用`id2`。然后我们打印`id1`和`id2`。通过打印`id1`和`id2`，您可以看到每个 id 变量都有自己的内存并保持自己的数据值。

接下来，我们必须记得释放`CreateId()`分配的内存。我们必须使用`delete()`运算符。等等，`delete()`运算符需要一个指向将被删除的内存的指针。变量`id1`和`id2`都是引用，而不是指针。是的，它们各自包含一个地址，因为每个都是作为指针实现的，但是它们各自的标识符的任何使用总是处于解引用状态。为了规避这个困境，我们只需在调用`delete()`之前取引用变量`id1`和`id2`的地址，比如`delete &id1;`。*很少*情况下，您可能需要通过引用变量删除内存，但现在您知道在需要时如何做。

这个例子的输出是：

```cpp
Id1: 100 Id2: 101
```

现在我们了解了引用如何在函数参数中使用以及作为函数的返回值，让我们继续通过进一步研究引用的微妙之处。

# 使用 const 限定符与引用

`const`限定符可以用来限定引用初始化或*引用的*数据。我们还可以将`const`限定的引用用作函数的参数和函数的返回值。

重要的是要理解，在 C++中，引用被实现为一个常量指针。也就是说，引用变量中包含的地址是一个固定的地址。这解释了为什么引用变量必须初始化为它将引用的对象，并且不能以后使用赋值来更新。这也解释了为什么仅对引用本身（而不仅仅是它引用的数据）进行常量限定是没有意义的。这种`const`限定的变体已经隐含在其底层实现中。

让我们看看在引用中使用`const`的各种情况。

## 使用对常量对象的引用

`const`限定符可以用来指示引用初始化的数据是不可修改的。这样，别名总是引用一个固定的内存块，该变量的值不能使用别名本身来改变。一旦指定为常量，引用意味着既不会改变引用本身，也不会改变其值。同样，由于其底层实现为常量限定指针，`const`限定的引用不能在任何赋值中用作*l 值*。

注意

回想一下，左值意味着可以修改的值，并且出现在赋值的左侧。

让我们举一个简单的例子来理解这种情况：

https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex5.cpp

```cpp
#include <iostream>
using namespace std;
int main()
{
   int x = 5;
   const int &refInt = x;
   cout << x << " " << refInt << endl;
   // refInt = 6;  // Illegal -- refInt is const 
   x = 7;   // we can inadvertently change refInt
   cout << x << " " << refInt << endl;
   return 0;
}
```

在前面的例子中，注意我们声明`int x = 5;`，然后我们用声明`const int &refInt = x;`建立对该整数的常量引用。接下来，我们打印出基线的两个值，并注意它们是相同的。这是有道理的，它们引用相同的整数内存。

接下来，在被注释掉的代码片段中，`//refInt = 6;`，我们试图修改引用所指向的数据。因为`refInt`被限定为`const`，这是非法的；因此这就是我们注释掉这行代码的原因。

然而，在下一行代码中，我们给`x`赋值为`7`。由于`refInt`引用了相同的内存，它的值也将被修改。等等，`refInt`不是常量吗？是的，通过将`refInt`限定为`const`，我们指示使用标识符`refInt`时其值不会被修改。这个内存仍然可以使用`x`来修改。

但等等，这不是一个问题吗？不，如果`refInt`真的想要引用不可修改的东西，它可以用`const int`而不是`int`来初始化自己。这是 C++中一个微妙的点，因此你可以编写完全符合你意图的代码，理解每种选择的重要性和后果。

这个例子的输出是：

```cpp
5 5
7 7
```

接下来，让我们看一下`const`限定符主题的变化。

## 使用指向常量对象的指针作为函数参数和作为函数的返回类型

使用`const`限定符与函数参数可以允许通过引用传递参数的速度，但通过值传递参数的安全性。这是 C++中一个有用的特性。

一个函数将一个对象的引用作为参数通常比将对象的副本作为参数的函数版本具有更少的开销。当在堆栈上复制的对象类型很大时，这种情况最为明显。将引用作为形式参数传递更快，但允许在函数范围内可能修改实际参数。将常量对象的引用作为函数参数提供了参数的速度和安全性。在参数列表中限定为`const`的引用在所讨论的函数范围内可能不是一个左值。

`const`限定符引用的同样好处也存在于函数的返回值中。常量限定所引用的数据坚持要求函数的调用者也必须将返回值存储在对常量对象的引用中，确保对象不会被修改。

让我们看一个例子：

https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex6.cpp

```cpp
#include <iostream>      
using namespace std;
class Collection
{
public:
    int x;
    float y;
};
void Update(Collection &);   // function prototypes
void Print(const Collection &);
int main()
{
    Collection collect1, *collect2;
    collect2 = new Collection;  // allocate memory from heap
    Update(collect1);   // a ref to the object will be passed
    Update(*collect2);  // same here -- *collect2 is an object
    Print(collect1);  
    Print(*collect2);
    delete collect2;    // delete heap memory
    return 0;
}
void Update(Collection &c)
{
    cout << "Enter x and y members: ";
    cin >> c.x >> c.y;
}

void Print(const Collection &c)
{
    cout << "x member: " << c.x;
    cout << "   y member: " << c.y << endl;
}
```

在这个例子中，我们首先定义了一个简单的`class Collection`，其中包含数据成员`x`和`y`。接下来，我们原型化了`Update(Collection &);`和`Print(const Collection &);`。请注意，`Print()`对被引用的数据进行了常量限定作为输入参数。这意味着该函数将通过引用传递此参数，享受传递参数的速度，但通过值传递参数的安全性。

注意，在程序的末尾，我们看到了`Update()`和`Print()`的定义。两者都采用引用作为参数，但是`Print()`的参数是常量限定的：`void Print(const Collection &);`。请注意，两个函数在每个函数体内使用`.`（成员选择）符号来访问相关的数据成员。

在`main()`中，我们声明了两个变量，`collect1`类型为`Collection`，`collect2`是指向`Collection`的指针（并且其内存随后被分配）。我们为`collect1`和`*collect2`都调用了`Update()`，在每种情况下，都将适用对象的引用传递给`Update()`函数。对于`collect2`，它是一个指针变量，实际参数必须首先解引用`*collect2`，然后调用此函数。

最后，在`main()`中，我们连续为`collect1`和`*collect2`调用`Print()`。在这里，`Print()`将引用每个对象作为常量限定的引用数据，确保在`Print()`函数范围内不可能修改任何输入参数。

这是我们示例的输出：

```cpp
Enter x and y members: 33 23.77
Enter x and y members: 10 12.11
x member: 33   y member: 23.77
x member: 10   y member: 12.11
```

现在我们已经了解了`const`限定引用何时有用，让我们看看何时可以使用引用代替指针，以及何时不可以。

# 实现底层实现和限制

引用可以简化间接引用所需的符号。但是，在某些情况下，引用根本无法取代指针。要了解这些情况，有必要回顾一下 C++中引用的底层实现。

引用被实现为常量指针，因此必须初始化。一旦初始化，引用就不能引用不同的对象（尽管被引用的对象的值可以更改）。

为了理解实现，让我们考虑一个样本引用声明：`int &intVar = x;`。从实现的角度来看，前一个变量声明实际上被声明为`int *const intVar = &x;`。请注意，初始化左侧显示的`&`符号具有引用的含义，而初始化或赋值右侧显示的`&`符号意味着取地址。这两个声明说明了引用的定义与其底层实现。

接下来，让我们了解在哪些情况下不能使用引用。

## 了解何时必须使用指针而不是引用

根据引用的底层实现（作为`const`指针），大多数引用使用的限制都是有道理的。例如，不允许引用引用；每个间接级别都需要提前初始化，这通常需要多个步骤，例如使用指针。也不允许引用数组（每个元素都需要立即初始化）；尽管如此，指针数组始终是一个选择。还不允许指向引用的指针；但是，允许引用指针（以及指向指针的指针）。

让我们来看看一个有趣的允许引用的机制，这是我们尚未探讨的。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter04/Chp4-Ex7.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master/Chapter04/Chp4-Ex7.cpp)

```cpp
#include <iostream>   
using namespace std;
int main()
{
    int *ptr = new int;
    *ptr = 20;
    int *&refPtr = ptr;  // establish a reference to a pointer
    cout << *ptr << " " << *refPtr << endl; 
    return 0;
}
```

在这个例子中，我们声明`int *ptr;`，然后为`ptr`分配内存（在一行上合并）。然后我们给`*p`赋值为`20`。

接下来，我们声明`int *&refPtr = ptr;`，这是一个指向`int`类型指针的引用。最好从右向左阅读声明。因此，我们使用`ptr`来初始化`refPtr`，它是指向`int`的指针的引用。在这种情况下，两种类型匹配：`ptr`是指向`int`的指针，因此`refPtr`必须引用指向`int`的指针。然后我们打印出`*ptr`和`*refPtr`的值，可以看到它们是相同的。

以下是我们程序的输出：

```cpp
20 20
```

通过这个例子，我们看到了另一个有趣的引用用法。我们也了解了使用引用所施加的限制，所有这些限制都是由它们的基础实现驱动的。

# 总结

在本章中，我们学习了 C++引用的许多方面。我们花时间了解了引用的基础知识，比如声明和初始化引用变量到现有对象，以及如何访问基本类型和用户定义类型的引用组件。

我们已经看到如何在函数中有意义地利用引用，既作为输入参数，又作为返回值。我们还看到了何时合理地对引用应用`const`限定符，以及如何将这个概念与函数的参数和返回值相结合。最后，我们看到了引用的基础实现。这有助于解释引用所包含的一些限制，以及帮助我们理解间接寻址的哪些情况将需要使用指针而不是引用。

与指针一样，本章中使用引用的所有技能将在接下来的章节中自由使用。C++允许程序员使用引用来更方便地进行间接寻址的表示；然而，程序员预计可以相对轻松地利用指针进行间接寻址。

最后，您现在可以继续前往*第五章*，*详细探讨类*，在这一章中，我们将开始 C++的面向对象特性。这就是我们一直在等待的；让我们开始吧！

# 问题

1.  修改并增强您的 C++程序，从*第三章*，*间接寻址-指针*，*练习 1*，如下所示：

a. 重载您的`ReadData()`函数，使用接受`Student &`参数的版本，以允许从键盘在函数内输入`firstName`、`lastName`、`currentCourseEnrolled`和`gpa`。

b. 替换您先前解决方案中的`Print()`函数，该函数取一个`Student`，而是取一个`const``Student &`作为`Print()`的参数。

c. 在`main()`中创建`Student`类型和`Student *`类型的变量。现在，调用各种版本的`ReadData()`和`Print()`。指针变量是否必须调用接受指针的这些函数的版本，非指针变量是否必须调用接受引用的这些函数的版本？为什么？
