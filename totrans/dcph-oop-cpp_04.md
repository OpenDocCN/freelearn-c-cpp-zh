# 4

# 间接寻址 – 引用

本章将探讨如何在 C++ 中使用引用。引用通常，但不总是，可以用作间接寻址的替代品，而指针则不行。尽管您已经从上一章通过使用指针获得了间接寻址的经验，但我们仍将从基础开始，以理解 C++ 中的引用。

与指针一样，引用是您必须能够轻松利用的语言特性。许多其他语言在不需要像 C++ 那样彻底理解的情况下，使用引用进行间接寻址。正如指针一样，您将在其他程序员的代码中经常看到引用的使用。您可能会很高兴地发现，与指针相比，使用引用在编写应用程序时将提供更简洁的记法。

不幸的是，引用不能在所有需要间接寻址的情况下替代指针。因此，在 C++ 中，对使用指针和引用进行间接寻址的彻底理解是编写成功且可维护代码的必要条件。

本章的目标将是补充您对使用指针进行间接寻址的理解，并了解如何使用 C++ 引用作为替代品。理解这两种间接寻址技术将使您成为一名更好的程序员，能够轻松理解并修改他人的代码，以及自己编写原创、成熟和合格的 C++ 代码。

在本章中，我们将涵盖以下主要主题：

+   引用基础 – 声明、初始化、访问和引用现有对象

+   将引用用作函数的参数和返回值

+   使用引用的 `const` 限定符

+   理解底层实现，以及何时无法使用引用

到本章结束时，您将了解如何声明、初始化和访问引用；您将了解如何引用内存中的现有对象。您将能够将引用用作函数的参数，并理解它们如何作为函数的返回值使用。

您还将理解 `const` 限定符如何应用于引用作为变量，以及如何与函数的参数和返回类型一起使用。您将能够区分在哪些情况下可以使用引用代替指针，以及在哪些情况下它们不能作为指针的替代品。为了成功进行本书后续章节的学习，这些技能是必要的。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 中找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter04`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter04)。每个完整程序示例都可以在 GitHub 的相应章节标题（子目录）下的文件中找到，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter04`子目录中的`Chp4-Ex1.cpp`文件中找到。

本章的 CiA 视频可在以下网址查看：[`bit.ly/3ptaMRK`](https://bit.ly/3ptaMRK)。

# 理解引用基础

在本节中，我们将回顾引用基础，并介绍适用于引用的运算符，例如引用操作符 `&`。我们将使用引用操作符（`&`）来建立对现有变量的引用。与指针变量一样，引用变量引用的是在别处定义的内存。

使用引用变量使我们能够使用比指针在间接访问内存时使用的符号更直接的符号。许多程序员都欣赏引用变量与指针变量在符号上的清晰度。然而，在幕后，内存必须始终得到适当的分配和释放；被引用的部分内存可能来自堆。程序员无疑需要处理他们整体代码中的一些指针。

我们将区分引用和指针何时可以互换，何时不能。让我们从声明和使用引用变量的基本符号开始。

## 声明、初始化和访问引用

让我们从引用变量的含义开始。C++中的 `&`。引用必须在声明时初始化，并且永远不能被分配以引用另一个对象。引用和初始化器必须是同一类型。由于引用和被引用的对象共享相同的内存，因此可以使用任一变量来修改共享内存位置的内存内容。

在幕后，引用变量可以与指针变量相提并论，因为它持有它所引用的变量的地址。与指针变量不同，引用变量的任何使用都会自动解除引用变量，以到达它所包含的地址；引用操作符 `*` 对于引用来说是不需要的。解除引用是自动的，并且在使用引用变量时是隐含的。

让我们看看一个说明引用基础的示例：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex1.cpp)

```cpp
#include <iostream>
using std::cout;
using std::endl;
int main()
{
    int x = 10;
    int *p = new int;   // allocate memory for ptr variable
    *p = 20;            // dereference and assign value 
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
    delete p;       // relinquish p's memory
    return 0;
}
```

在前面的例子中，我们首先声明并初始化`int x = 10;`，然后声明并分配`int *p = new int;`。然后我们将整数值`20`赋给`*p`。

接下来，我们声明并初始化两个引用变量，`refInt1`和`refInt2`。在第一个引用声明和初始化中，`int &refInt1 = x;`，我们将`refInt1`设置为引用变量`x`。从右到左阅读引用声明有助于理解。在这里，我们说的是使用`x`来初始化`refInt1`，它是一个指向整数的引用（`&`）。注意，初始化器`x`是一个整数，而`refInt1`被声明为指向整数的引用；它们的类型匹配。这是很重要的。如果类型不匹配，代码将无法编译。同样，声明和初始化`int &refInt2 = *p;`也将`refInt2`声明为指向整数的引用。哪一个？由`p`指向的那个。这就是为什么使用`*`解引用`p`以到达整数本身的原因。

现在，我们打印出`x`、`*p`、`refInt1`和`refInt2`；我们可以验证`x`和`refInt1`具有相同的值`10`，而`*p`和`refInt2`也具有相同的值`20`。

接下来，使用原始变量，我们将`x`和`*p`都增加一。这不仅增加了`x`和`*p`的值，还增加了`refInt1`和`refInt2`的值。重复打印这四个值，我们再次注意到`x`和`refInt1`的值为`11`，而`*p`和`refInt2`的值为`21`。

最后，我们使用引用变量来增加共享内存。我们将`refInt1`和`*refint2`都增加一，这也增加了原始变量`x`和`*p`的值。这是因为原始变量和对其的引用之间的内存是相同的。也就是说，引用可以被视为原始变量的别名。我们通过再次打印出这四个变量来结束程序。

下面是输出结果：

```cpp
10 20 10 20
11 21 11 21
12 22 12 22
```

重要提示

记住，引用变量必须初始化为它将要引用的变量。引用永远不能分配给另一个变量。更准确地说，我们不能重新绑定引用到另一个实体。引用及其初始化器必须是同一类型。

现在我们已经掌握了如何声明简单的引用，让我们更全面地看看如何引用现有对象，例如用户定义的类型。

## 引用用户定义类型的现有对象

如果需要定义对`struct`或`class`类型对象的引用，则可以通过使用`.`（成员选择）运算符直接访问被引用的对象。再次强调，与指针不同，在选择所需的成员之前，没有必要首先使用解引用运算符来访问被引用的对象。

让我们看看一个例子，其中我们引用一个用户定义的类型：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex2.cpp)

```cpp
#include <iostream>
using std::cout;
using std::endl;
using std::string;
class Student    // very simple class – we will add to it 
{                // in our next chapter
public:
    string name;
    float gpa;
};
int main()
{
    Student s1;
    Student &sRef = s1;  // establish a reference to s1
    s1.name = "Katje Katz";   // fill in the data
    s1.gpa = 3.75;
    cout << s1.name << " has GPA: " << s1.gpa << endl; 
    cout << sRef.name << " has GPA: " << sRef.gpa << endl; 
    sRef.name = "George Katz";  // change the data
    sRef.gpa = 3.25;
    cout << s1.name << " has GPA: " << s1.gpa << endl; 
    cout << sRef.name << " has GPA: " << sRef.gpa << endl; 
    return 0;
}
```

在本程序的第一个部分，我们使用 `class` 定义了一个用户自定义类型 `Student`。接下来，我们使用 `Student s1;` 声明了一个类型为 `Student` 的变量 `s1`。现在，我们声明并初始化了一个指向 `Student` 的引用 `Student &sRef = s1;`。在这里，我们声明 `sRef` 来引用一个特定的 `Student`，即 `s1`。请注意，`s1` 是 `Student` 类型，而 `sRef` 的引用类型也是 `Student` 类型。

现在，我们通过两个简单的赋值将一些初始数据加载到 `s1.name` 和 `s1.gpa` 中。因此，这改变了 `sRef` 的值，因为 `s1` 和 `sRef` 引用的是相同的内存。也就是说，`sRef` 是 `s1` 的别名。

我们打印出 `s1` 和 `sRef` 的各种数据成员，并注意到它们包含相同的值。

现在，我们通过赋值将新的值加载到 `sRef.name` 和 `sRef.gpa` 中。同样，我们打印出 `s1` 和 `sRef` 的各种数据成员，并注意到两者的值都发生了变化。再次，我们可以看到它们引用的是相同的内存。

伴随此程序的输出如下：

```cpp
Katje Katz has GPA: 3.75
Katje Katz has GPA: 3.75
George Katz has GPA: 3.25
George Katz has GPA: 3.25
```

让我们通过考虑引用在函数中的使用来进一步理解引用的概念。

# 在函数中使用引用

到目前为止，我们通过使用它们为现有变量创建别名来最小化地展示了引用。相反，让我们提出一个有意义的引用用法，例如在函数调用中使用它们。我们知道大多数 C++ 函数都会接受参数，我们已经在之前的章节中看到了许多示例，说明了函数原型和函数定义。现在，让我们通过将引用作为函数参数传递，并使用引用作为函数的返回值来增强我们对函数的理解。

## 将引用作为函数参数传递

引用可以用作函数的参数，以实现按引用传递参数，而不是按值传递参数。引用还可以在所讨论的函数的作用域内以及在调用该函数时减少对指针符号的需求。对于引用形式的正式参数，使用对象或 `.`（成员选择）符号来访问 `struct` 或 `class` 成员。

为了修改传递给函数的变量的内容，必须使用该参数的引用（或指针）作为函数参数。就像指针一样，当将引用传递给函数时，传递的是表示该引用的地址的副本。然而，在函数内部，任何使用形式参数为引用的用法都将自动和隐式地解引用，使用户可以使用对象而不是指针表示法。与传递指针变量一样，将引用变量传递给函数将允许修改该参数引用的内存。

当检查一个函数调用（除了其原型）时，不会很明显地看出传递给该函数的对象是按值传递还是按引用传递。也就是说，整个对象是否会在栈上复制，或者是否将对该对象的引用传递到栈上。这是因为当操作引用时使用了对象表示法，并且这两种情况下的函数调用将使用相同的语法。

仔细使用函数原型将解决函数定义看起来是什么样子以及其参数是对象还是对象引用的神秘。记住，函数定义可能定义在一个与该函数的任何调用都分开的文件中，并且不容易查看。注意，这种歧义不会出现在函数调用中指定的指针；根据变量的声明方式，立即可以明显看出地址是否被发送到函数。

让我们花几分钟时间理解一个示例，该示例说明了将引用作为参数传递给函数。在这里，我们将首先检查三个函数，这些函数有助于以下完整的程序示例：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex3.cpp)

```cpp
void AddOne(int &arg)   // These two fns. are overloaded
{
    arg++;
}
void AddOne(int *arg)   // Overloaded function definition
{
    (*arg)++;
}
void Display(int &arg)  // Function parameter establishes 
                       // a reference to arg
{
    cout << arg << " " << flush;
}
```

检查前面的函数，注意`AddOne(int &arg)`将一个`int`的引用作为形式参数，而`AddOne(int *arg)`将一个`int`的指针作为形式参数。这些函数是重载的。它们的实际参数的类型将确定调用哪个版本。

现在，让我们考虑`Display(int &arg)`。这个函数接受一个整数的引用。注意，在这个函数的定义中，使用的是对象（而不是指针）表示法来打印`arg`。

现在，让我们检查这个程序的其余部分：

```cpp
#include <iostream>
using std::cout;
using std::flush;
void AddOne(int &);    // function prototypes
void AddOne(int *);
void Display(int &);
int main()
{
    int x = 10, *y = nullptr;
    y = new int;    // allocate y's memory
    *y = 15;        // dereference y to assign a value
    Display(x);
    Display(*y);

    AddOne(x);    // calls ref. version (with an object) 
    AddOne(*y);   // also calls reference version 
    Display(x);   // Based on prototype, we see we are 
    Display(*y);  // passing by ref. Without prototype, 
                  // we may have guessed it was by value.
    AddOne(&x);   // calls pointer version
    AddOne(y);    // also calls pointer version
    Display(x);
    Display(*y);
    delete y;     // relinquish y's memory
    return 0;
}
```

注意这个程序段顶部的函数原型。它们将与代码前一段中的函数定义相匹配。现在，在 `main()` 函数中，我们声明并初始化 `int x = 10;` 并声明一个指针 `int *y;`。我们使用 `new()` 分配 `y` 的内存，然后通过解引用指针赋值 `*y = 15;`。我们使用连续的 `Display()` 调用来打印 `x` 和 `*y` 的相应值作为基准。

接下来，我们调用 `AddOne(x)` 然后是 `AddOne(*y)`。变量 `x` 被声明为整数，`*y` 指向 `y` 所指向的整数。在这两种情况下，我们都在将整数作为实际参数传递给具有签名 `void AddOne(int &);` 的重载函数版本。在这两种情况下，形式参数将在函数中被更改，因为我们是通过引用传递的。我们可以在使用连续的 `Display()` 调用打印它们的相应值时验证这一点。注意，在函数调用 `AddOne(x);` 中，实际参数 `x` 的引用是通过函数调用时参数列表中的形式参数 `arg` 建立的。

相比之下，我们随后调用 `AddOne(&x);` 然后是 `AddOne(y);`。在这两种情况下，我们都在调用具有签名 `void AddOne(int *);` 的重载函数版本。在每种情况下，我们都在函数中将地址的副本作为实际参数传递。自然地，`&x` 是变量 `x` 的地址，所以这行得通。同样，`y` 本身也是一个地址——它被声明为一个指针变量。我们再次通过两次调用 `Display()` 验证它们的相应值是否被更改。

注意，在每次调用 `Display()` 时，我们传递一个 `int` 类型的对象。仅从函数调用本身来看，我们无法确定这个函数是否将接受一个 `int` 作为实际参数（这意味着值不能被更改），或者接受一个 `int &` 作为实际参数（这意味着值可以被修改）。这两种情况都是可能的。然而，通过查看函数原型，我们可以清楚地看到这个函数接受一个 `int &` 作为参数，并且由此我们可以理解参数可能被修改。这是函数原型有很多好处之一。

下面是完整程序示例的输出：

```cpp
10 15 11 16 12 17
```

现在，让我们通过使用函数的引用作为返回值来补充我们对使用引用与函数的讨论。

## 使用引用作为函数的返回值

函数可以通过返回语句返回数据的引用。当我们为用户定义的类型重载运算符时，在*第十二章*，*朋友与运算符重载*中，我们将看到需要通过引用返回数据的需求。在运算符重载中，使用指针从函数返回值将不会是保留运算符原始语法的选项。我们必须返回一个引用（或带有`const`修饰的引用）；这也会使重载的运算符能够享受级联使用。此外，当我们探索 C++标准模板库时，在*第十四章*，*理解 STL 基础*中，了解如何通过引用返回对象将是有用的。

当通过函数的返回语句返回引用时，请确保所引用的内存在函数调用完成后仍然存在。**不要**返回函数中定义在栈上的局部变量的引用；这个内存将在函数完成时从栈上弹出。

由于我们无法在函数内部返回局部变量的引用，并且返回外部变量的引用是没有意义的，您可能会问我们返回引用的数据将驻留在何处。这些数据不可避免地将在堆上。堆内存将存在于函数调用的范围之外。在大多数情况下，堆内存已经在其他地方分配；然而，在罕见的情况下，内存可能在此函数内部分配。在这种情况下，您必须记住在不再需要时释放分配的堆内存。

通过引用变量删除堆内存（与指针相比）将需要您使用地址运算符`&`将所需的地址传递给`delete()`运算符。尽管引用变量包含它们所引用的对象的地址，但引用标识符的使用始终处于解引用状态。使用引用变量删除内存的需求**很少**出现；我们将在*第十章*，*实现关联、聚合和组合*中讨论一个有意义的（尽管很少见）示例。

重要提示

以下示例说明了如何从函数中返回引用的语法，您将在我们重载运算符以允许其级联使用时使用它，例如。然而，不建议使用引用返回新分配的堆内存（在大多数情况下，堆内存已经在其他地方分配）。这是一个常见的约定，使用引用向其他程序员发出信号，表明该变量不需要内存管理。尽管如此，在现有代码中可能会看到通过引用进行此类删除的罕见场景（如上述与关联的罕见使用），因此了解如何进行这种罕见的删除是有用的。

让我们通过一个示例来展示将引用作为函数返回值的机制：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex4.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex4.cpp)

```cpp
#include <iostream>
using std::cout;
using std::endl;
int &CreateId();  // function prototype

int main()    
{
    int &id1 = CreateId();  // reference established
    int &id2 = CreateId();
    cout << "Id1: " << id1 << " Id2: " << id2 << endl;
    delete &id1; // Here, '&' is address-of, not reference
    delete &id2; // to calculate address to pass delete()
    return 0;  // It is unusual to delete in fashion shown,
}          // using the addr. of a ref. Also, deleting in 
           // a diff. scope than alloc. can be error prone
int &CreateId()   // Function returns a reference to an int
{
    static int count = 100;  // initialize with first id 
    int *memory = new int;
    *memory = count++;  // use count as id, then increment
    return *memory;
}
```

在此示例中，我们看到`int &CreateId();`在程序顶部进行了原型声明。这告诉我们`CreateId()`将返回一个整数的引用。返回值必须用于初始化类型为`int &`的变量。

在程序底部，我们看到`CreateId()`函数的定义。请注意，此函数首先声明了一个`static`计数器，它被初始化一次为`100`。因为这个局部变量是`static`的，它将在函数调用之间保留其值。然后我们在几行之后将这个计数器增加一。静态变量`count`将作为生成唯一 ID 的基础。

接下来，在`CreateId()`中，我们在堆上为整数分配空间，并使用局部变量`memory`指向它。然后我们将`*memory`加载为`count`的值，然后增加`count`以便下次进入此函数时使用。然后我们使用`*memory`作为此函数的返回值。请注意，`*memory`是一个整数（由变量`memory`在堆上指向的整数）。当我们从函数返回它时，它作为对该整数的引用返回。当从函数返回引用时，始终确保所引用的内存存在于函数的作用域之外。

现在，让我们看看我们的`main()`函数。在这里，我们用以下函数调用和初始化的返回值初始化引用变量`id1`：`int &id1 = CreateId();`。请注意，引用`id1`必须在声明时初始化，我们通过上述代码满足了这一要求。

我们用`id2`重复此过程，用`CreateId()`的返回值初始化这个引用。然后我们打印`id1`和`id2`。通过打印`id1`和`id2`，你可以看到每个 ID 变量都有自己的内存并保持自己的数据值。

接下来，我们必须记住释放`CreateId()`代表我们分配的内存。我们必须使用`delete()`运算符。等等，`delete()`运算符期望一个指向将被删除内存的指针。变量`id1`和`id2`都是引用，不是指针。确实，它们各自包含一个地址，因为每个都是作为指针实现的，但任何使用它们各自标识符的操作总是处于解引用状态。为了解决这个问题，我们在调用`delete()`之前简单地取引用变量`id1`和`id2`的地址，例如`delete &id1;`。通过引用变量删除内存的情况**很少见**，但现在你知道了如果需要这样做该如何操作。

此示例的输出如下：

```cpp
Id1: 100 Id2: 101
```

现在我们已经了解了如何在函数参数中使用引用以及如何从函数返回引用，让我们进一步探讨引用的细微差别。

# 使用 const 关键字与引用

`const` 关键字可以用来限定引用初始化或*引用*的数据。我们还可以使用 `const` 限定的引用作为函数的参数和函数的返回值。

重要的是要理解在 C++ 中，引用被实现为一个常量指针。也就是说，引用变量中包含的地址是一个固定的地址。这解释了为什么引用变量必须初始化为它将要引用的对象，并且不能通过赋值来更新。这也解释了为什么对引用本身进行常量限定（而不仅仅是它引用的数据）没有意义。这种 `const` 限定的变体已经隐含在其底层实现中。

让我们通过使用 `const` 与引用的各种场景来看看这些。

## 使用常量对象引用

`const` 关键字可以用来表明引用初始化的数据是不可修改的。以这种方式，别名始终指向一个固定的内存块，该变量的值不能通过别名本身来更改。一旦引用被指定为常量，就暗示了引用及其值都不能更改。再次强调，由于引用本身作为常量限定指针的底层实现，引用本身不能更改。`const` 限定的引用不能用作任何赋值中的 *左值*。

注意

回想一下，**左值** 是一个可以修改的值，它出现在赋值语句的左侧。

让我们通过一个简单的例子来了解这种情况：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex5.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex5.cpp)

```cpp
#include <iostream>
using std::cout;
using std::endl;
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

在上一个示例中，请注意我们声明了 `int x = 5;` 然后通过以下声明建立对该整数的常量引用：`const int &refInt = x;`。接下来，我们打印出这两个值作为基准，并注意到它们是相同的。这是有道理的；它们引用了相同的整数内存。

接下来，在注释掉的代码片段 `//refInt = 6;` 中，我们尝试修改引用所引用的数据。因为 `refInt` 被标记为 `const`，这是非法的；这就是为什么我们注释掉了这一行代码。

然而，在下一行代码中，我们将 `x` 的值赋为 `7`。由于 `refInt` 引用了相同的内存，其值也将被修改。等等，`refInt` 不是常量吗？是的，通过将 `refInt` 标记为 `const`，我们表明其值不会通过 `refInt` 这个标识符来修改。这个内存仍然可以通过 `x` 来修改。

但是，这难道不是一个问题吗？不，如果`refInt`确实想要引用不可修改的对象，它可以用`const int`而不是`int`来初始化自己。这个微妙之处是 C++中需要记住的，这样你就可以编写出符合你意图的代码，理解每个选择的意义和后果。

本例的输出如下：

```cpp
5 5
7 7
```

接下来，让我们看看`const`修饰符主题的一个变体。

## 使用指向常量对象的指针作为函数参数，以及作为函数的返回类型

使用`const`修饰符作为函数参数不仅允许通过引用传递参数的速度，还允许通过值传递参数的安全性。这是 C++中的一个有用特性。

一个以对象引用作为参数的函数通常比一个以对象副本作为参数的函数开销更小。这最明显地发生在对象类型本应复制到栈上且很大时。将引用作为形式参数传递更快，同时允许在函数的作用域内修改实际参数。将常量对象的引用作为函数参数提供了对所讨论参数的速度和安全性。在参数列表中修饰为`const`的引用在相关函数的作用域内可能不是`*l-value*`。

`const`修饰的引用对函数的返回值也有同样的好处。对引用的数据进行常量修饰坚持要求函数的调用者必须也将返回值存储在常量对象的引用中，确保对象不会被修改。

让我们看看一个例子：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex6.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex6.cpp)

```cpp
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
struct collection
{
    int x;
    float y;
};
void Update(collection &);   // function prototypes
void Print(const collection &);
int main()
{
    collection collect1, *collect2 = nullptr;
    collect2 = new collection;  // allocate mem. from heap
    Update(collect1);  // a ref to the object is passed
    Update(*collect2); // same here: *collect2 is an object
    Print(collect1);  
    Print(*collect2);
    delete collect2;   // delete heap memory
    return 0;
}
void Update(collection &c)
{
    cout << "Enter <int> and <float> members: ";
    cin >> c.x >> c.y;
}

void Print(const collection &c)
{
    cout << "x member: " << c.x;
    cout << "   y member: " << c.y << endl;
}
```

在这个例子中，我们首先定义了一个简单的`struct` `collection`，其中包含数据成员`x`和`y`。接下来，我们原型化了`Update(collection &);`和`Print(const collection &);`。请注意，`Print()`的常量修饰符指定了被引用的数据作为输入参数。这意味着这个函数将享受通过引用传递参数的速度，以及通过值传递参数的安全性。

注意，在程序末尾，我们看到了`Update()`和`Print()`的定义。两个函数都接受引用作为参数，然而，`Print()`的参数是常量修饰的：`void Print(const collection &);`。请注意，两个函数都在函数体内使用`.`（成员选择）运算符来访问相关数据成员。

在 `main()` 中，我们声明了两个变量，`collect1` 类型为 `collection`，以及 `collect2`，它是一个指向 `collection` 的指针（其内存随后被分配）。我们对 `collect1` 和 `*collect2` 都调用了 `Update()`，在每种情况下，都向 `Update()` 函数传递了适用对象的引用。在 `collect2` 的情况下，由于它是一个指针变量，在调用此函数之前必须首先取消引用 `*collect2` 以到达被引用的对象。

最后，在 `main()` 函数中，我们依次对 `collect1` 和 `*collect2` 调用 `Print()`。在这里，`Print()` 将引用每个作为形式参数的对象作为常量合格引用数据，确保在 `Print()` 函数的作用域内不可能修改任何输入参数。

下面是伴随我们示例的输出：

```cpp
Enter x and y members: 33 23.77
Enter x and y members: 10 12.11
x member: 33   y member: 23.77
x member: 10   y member: 12.11
```

现在我们已经了解了何时使用常量合格引用是有用的，让我们看看何时可以使用引用代替指针，以及何时不能。

# 理解底层实现和限制

引用可以简化间接引用所需的符号。然而，有些情况下引用根本不能取代指针。为了理解这些情况，回顾 C++ 中引用的底层实现是有用的。

引用被实现为常量指针，因此它们必须被初始化。一旦初始化，引用就不能指向不同的对象（尽管被引用的对象的值可以改变）。

为了理解实现，让我们考虑一个示例引用声明：`int &intVar = x;`。从实现的角度来看，这就像前面的变量声明被改为 `int *const intVar = &x;`。请注意，初始化左侧的 `&` 符号表示引用的意义，而初始化或赋值右侧的 `&` 符号表示取地址。这两个声明说明了引用的定义与其底层实现之间的关系。

尽管引用被实现为常量指针，但引用变量的使用就像底层常量指针已被取消引用一样。因此，你不能用 `nullptr` 初始化引用——不仅 `nullptr` 不能被取消引用，而且由于引用只能初始化而不能重置，就会失去将引用变量设置为指向有意义对象的机遇。这也适用于指针的引用。

接下来，让我们了解在哪些情况下我们不能使用引用。

## 理解何时必须使用指针而不是引用

基于引用的底层实现（作为`const`指针），大多数引用使用的限制都是有意义的。例如，引用的引用通常是不允许的；每个间接级别都需要预先初始化，这通常需要多个步骤，例如使用指针。然而，我们将在*第十五章*中看到`&&`)，*测试类和组件*，我们将检查各种*移动*操作。引用数组也是不允许的（每个元素都需要立即初始化）；然而，指针数组始终是一个选项。此外，不允许指向引用的指针；但是，允许引用指针（以及指针的指针）。

让我们看看一个有趣的允许引用案例的机制，我们尚未探索：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex7.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter04/Chp4-Ex7.cpp)

```cpp
#include <iostream>   
using std::cout;
using std::endl;
int main()
{
    int *ptr = new int;
    *ptr = 20;
    int *&refPtr = ptr;  // establish a reference to a ptr
    cout << *ptr << " " << *refPtr << endl; 
    delete ptr;
    return 0;
}
```

在这个例子中，我们声明`int *ptr;`然后为`ptr`分配内存（合并在一行中）。然后我们将值`20`赋给`*p`。

接下来，我们声明`int *&refPtr = ptr;`，这是一个指向`int`类型指针的引用。有助于从右到左阅读声明。因此，我们使用`ptr`来初始化`refPtr`，它是指向`int`的指针的引用。在这种情况下，两种类型匹配；`ptr`是`int`的指针，所以`refPtr`也必须引用一个指向`int`的指针。然后我们打印出`*ptr`和`*refPtr`的值，可以看到它们是相同的。

这里是伴随我们程序的输出：

```cpp
20 20
```

通过这个例子，我们看到了引用的另一种有趣用途。我们还理解了使用引用的限制，所有这些限制都是由它们的底层实现驱动的。

# 摘要

在本章中，我们学习了 C++引用的众多方面。我们花时间理解了引用的基础，例如将引用变量声明和初始化为现有对象，以及如何访问基本和用户定义类型的引用组件。

我们看到了如何以有意义的方式在函数中使用引用，无论是作为输入参数还是作为返回值。我们还看到了何时合理地将`const`限定符应用于引用，以及看到了如何将此概念与函数的参数和返回值相结合。最后，我们看到了引用的底层实现。这有助于解释引用包含的一些限制，以及理解哪些间接寻址的情况需要使用指针而不是引用。

与指针一样，本章中使用的所有关于引用的技能将在接下来的章节中自由使用。C++允许程序员使用引用来更方便地实现间接寻址；然而，程序员应预期能够相对容易地使用引用进行间接寻址。

最后，你现在可以向前推进到*第五章*，*详细探索类*，其中我们将开始 C++的面向对象特性。这是我们一直在等待的；让我们开始吧！

# 问题

1.  修改并增强你的 C++程序，从*第三章*，*间接寻址 – 指针*，*问题 1*，如下进行：

    1.  重载你的`ReadData()`函数，添加一个接受`Student &`参数的版本，以便在函数内部从键盘输入`firstName`、`lastName`、`currentCourseEnrolled`和`gpa`。

    1.  将你之前解决方案中的`Print()`函数替换为接受一个`const Student &`作为参数的函数。

    1.  在`main()`中创建`Student`类型和`Student *`类型的变量。现在，调用`ReadData()`和`Print()`的各种版本。指针变量是否必须调用接受指针的这些函数版本，非指针变量是否必须调用接受引用的这些函数版本？为什么或为什么不？

# 第二部分：在 C++中实现面向对象的概念

本部分的目标是理解如何使用 C++语言特性和经过验证的编程技术来实现 OO 设计。C++可以用于许多编码范式；程序员必须努力在 C++中以 OO 方式编程（这不是自动的）。这是本书最大的章节，因为理解如何将语言特性和实现技术映射到 OO 概念是至关重要的。

本节的第一章详细探讨了类，从描述 OO 概念中的封装和信息隐藏开始。深入探讨了语言特性，如成员函数、`this`指针、详细访问区域、构造函数（包括拷贝构造函数、成员初始化列表和类内初始化）、析构函数、成员函数的限定符（`const`、`static`和`inline`），以及数据成员的限定符（`const`和`static`）。

本节下一章探讨单继承的基本概念，使用 OO 概念中的泛化和特化，详细介绍了通过成员初始化列表继承的构造函数、构造和析构的顺序，以及理解继承的访问区域。探讨了最终类。本章通过探索公有与保护以及私有基类，以及这些语言特性如何改变继承的 OO 意义来进一步深入。

下一章深入探讨了面向对象的泛型概念，包括对这一概念的理解以及如何在 C++中使用虚函数实现。探讨了`virtual`、`override`和`final`关键字。检查了将操作动态绑定到特定方法。通过探索虚函数表来解释运行时绑定。

下一章详细解释了抽象类，将面向对象（OO）概念与其使用纯虚函数的实现相结合。介绍了接口的 OO 概念（在 C++中不是显式定义的），并回顾了其实施方法。通过继承层次结构的向上和向下转换完成本章内容。

下一章探讨了多重继承及其可能引发的问题。详细介绍了虚拟基类以及用于确定多重继承是否是特定场景的最佳设计的 OO 概念——区分器。如果存在其他可能的设计。

本节最后一章介绍了关联、聚合和组合的概念，以及如何使用指针或引用、指针集或内嵌对象来实现这些常见的对象关系。

本部分包括以下章节：

+   *第五章*，*详细探索类*

+   *第六章*，*使用单继承实现层次结构*

+   *第七章*，*通过多态利用动态绑定*

+   *第八章*，*掌握抽象类*

+   *第九章*，*探索多重继承*

+   *第十章*，*实现关联、聚合和组合*

第二部分：在 C++中实现面向对象的概念
