

# 与模板一起工作

本章将继续我们提高 C++编程知识库的追求，超越面向对象的概念，并继续以编写更可扩展的代码为目标。我们将接下来探索使用 C++模板创建泛型代码——包括**模板函数**和**模板类**。我们将学习如何正确编写模板代码，使其成为代码重用的顶峰。除了探索如何创建模板函数和模板类之外，我们还将了解适当使用运算符重载如何使模板函数对几乎所有类型的数据都具有可重用性。

在本章中，我们将涵盖以下主要主题：

+   探索模板基础以泛化代码

+   理解如何创建和使用模板函数和模板类

+   理解运算符重载如何使模板更易于扩展

许多面向对象的语言包括使用泛型的编程概念，允许类和接口的类型自身被参数化。在一些语言中，泛型仅仅是用于将对象转换为所需类型的包装器。在 C++中，泛型的概念更为全面，并且通过模板来实现。

到本章结束时，你将能够通过构建模板函数和模板类来设计更通用的代码。你将了解运算符重载如何确保模板函数可以针对任何数据类型进行高度扩展。通过将精心设计的模板成员函数与运算符重载相结合，你将能够在 C++中创建高度可重用和可扩展的模板类。

让我们通过探索模板来扩展我们的编程知识，以加深对 C++的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 中找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter13`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter13)。每个完整的程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter13`子目录中找到，文件名为`Chp13-Ex1.cpp`。

本章的 CiA 视频可以在以下链接查看：[`bit.ly/3A7lx0U`](https://bit.ly/3A7lx0U)。

# 探索模板基础以泛化代码

模板允许以抽象数据类型的方式在相关函数或类中指定代码，这些数据类型主要用于相关函数或类。创建模板的动机是为了泛型指定我们反复想要利用的函数和类的定义，但数据类型可以不同。这些组件的个性化版本在核心数据类型上可能有所不同；然后可以从这些关键数据类型中提取并泛型编写。

当我们选择使用此类或函数的特定类型时，而不是从类似类或函数（具有预置的数据类型）复制粘贴现有代码并稍作修改，预处理器会取模板代码并将其*展开*为我们请求的、真正的类型。这种模板*展开*能力允许程序员只编写和维护一个泛型化代码版本，而不是需要编写的许多特定类型版本的代码。好处是，预处理器将比我们使用复制、粘贴和轻微修改方法做得更精确地展开模板代码到真正的类型。

让我们花点时间进一步研究在代码中使用模板的动机。

## 检查模板的动机

假设我们希望创建一个类来安全地处理动态分配的数组，数据类型为`int`，就像我们在*第十二章*的*问题 3*的解决方案中所创建的那样，*朋友和运算符重载*。我们的动机可能是有一种可以增长或缩小的数组类型（与原生的固定大小数组不同），同时具有边界检查以确保安全使用（与使用`int *`实现的原始动态数组操作不同，这会无耻地允许我们访问超出动态数组分配长度的元素）。

我们可能决定创建一个`ArrayInt`类，以下是其初始框架：

```cpp
class ArrayInt
{
private: 
    int numElements = 0;     // in-class initialization
    int *contents = nullptr; // dynamically allocated array
public:
    ArrayInt(int size): numElements(size) 
    { 
        contents = new int [size];
    }
    ~ArrayInt() { delete [] contents; }       
    int &operator[](int index) // returns a referenceable
    {                // memory location or throws exception
        if (index < numElements) 
            return contents[index];
        else         // index selected is out of bounds
            throw std::out_of_range(std::to_string(index));
    }                                
};
int main()
{
    ArrayInt a1(5); // Create an ArrayInt of 5 elements
    try    // operator[] could throw an exception
    {
        a1[4] = 7;      // a1.operator[](4) = 7;
    }
    catch (const std::out_of_range &e)
    {
        cout << "Out of range: element " << e.what();
        cout << endl;
    }
}   
```

在前面的代码段中，请注意，我们的`ArrayInt`类使用`int *contents`实现数组的数据结构，该结构在构造函数中动态分配到所需的大小。我们已经重载了`operator[]`，以确保只返回数组中正确的范围内的索引值，否则抛出`std::out_of_range`异常。我们可以添加`Resize()`方法到`ArrayInt`等。总的来说，我们喜欢这个类的安全性和灵活性。

现在，我们可能想要有一个 `ArrayFloat` 类（或者稍后，一个 `ArrayStudent` 类）。与其复制我们的基线 `ArrayInt` 类并稍作修改来创建一个 `ArrayFloat` 类，例如，我们可能会问是否有更自动化的方式来完成这种替换。毕竟，在创建一个以 `ArrayInt` 类为起点的 `ArrayFloat` 类时，我们会改变什么？我们会改变数据成员 `contents` 的 *类型* – 从 `int *` 到 `float *`。我们会在构造函数中的内存分配中将 `contents = new int [size];` 改为使用 `float` 而不是 `int`（在任何实际重新分配中也是如此，例如在 `Resize()` 方法中）。

与其复制、粘贴并稍微修改一个 `ArrayInt` 类来创建一个 `ArrayFloat` 类，我们完全可以简单地使用一个 **模板类** 来泛化这个类内部操作的数据的 *类型*。同样，任何依赖于特定数据类型的函数都将成为 **模板函数**。我们将在稍后考察创建和使用模板的语法。

使用模板，我们可以创建一个名为 `Array` 的单一模板类，其中类型被泛化。在编译时，如果预处理器检测到我们在代码中使用了类型 `int` 或 `float` 的这个类，预处理器将为我们提供必要的模板 *展开*。也就是说，通过复制和粘贴（在幕后）每个模板类（及其方法），并用预处理器识别的我们正在使用的数据类型进行替换。

扩展后的代码，一旦在底层展开，并不比我们亲自为每个类编写代码更小。但重点是，我们不必费力地创建、修改、测试，并在以后维护每个略有不同的类。这是 C++ 为我们做的事情。这就是模板类和模板函数值得注意的目的。

模板不仅限于与原始数据类型一起使用。例如，我们可能希望创建一个 `Array`，其类型是用户定义的，如 `Student`。我们需要确保所有模板成员函数对我们实际扩展模板类以使用的所有数据类型都是有意义的。我们可能需要重载选定的运算符，以便我们的模板成员函数可以像与原始类型一样无缝地与用户定义的类型一起工作。

在本章后面，我们将看到一个示例，说明如果我们选择扩展模板类以用于用户定义的类型，我们可能需要重载哪些选定的运算符，以便类的成员函数可以流畅地与任何数据类型一起工作。幸运的是，我们知道如何重载运算符！

让我们继续前进，探索指定和使用模板函数和模板类的机制。

# 理解模板函数和类

**模板**提供了通过抽象与这些函数和类关联的数据类型来创建泛型函数和类的能力。模板函数和类都可以被精心编写，以便泛化这些函数和类所依赖的相关数据类型。

让我们先来探讨如何创建和使用模板函数。

## 创建和使用模板函数

**模板函数**除了参数化函数的参数本身之外，还参数化了函数的参数类型。模板函数要求函数体适用于几乎任何数据类型。模板函数可以是成员函数或非成员函数。运算符重载可以帮助确保模板函数的主体适用于用户定义的类型——我们很快就会看到更多关于这一点的内容。

关键字`template`，以及尖括号`<` `>`和类型名称的占位符，用于指定模板函数及其原型。

让我们看看一个不属于类成员的模板函数（我们很快就会看到模板成员函数的例子）。这个例子作为一个完整的可运行程序，可以在我们的 GitHub 仓库中找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex1.cpp)

```cpp
// template function prototype
template <class Type1, class Type2>   // template preamble
Type2 ChooseFirst(Type1, Type2);
// template function definition
template <class Type1, class Type2>  // template preamble
Type2 ChooseFirst(Type1 x, Type2 y)
{
    if (x < y) 
        return static_cast<Type2>(x);
    else 
        return y; 
}   
int main()
{
    int value1 = 4, value2 = 7;
    float value3 = 5.67f;
    cout << "First: " << ChooseFirst(value1, value3); 
    cout << endl;
    cout << "First: " << ChooseFirst(value2, value1); 
    cout << endl;
}
```

看看之前的函数示例，我们首先看到一个模板函数原型。`template <class Type1, class Type2>`的声明表明，这个原型将是一个模板原型，并且将使用占位符`Type1`和`Type2`而不是实际的数据类型。占位符`Type1`和`Type2`可以是（几乎）任何遵循标识符创建规则的名称。

然后，为了完成原型，我们看到`Type2 ChooseFirst(Type1, Type2);`，这表明这个函数的返回类型将是`Type2`，并且`ChooseFirst()`函数的参数将是`Type1`和`Type2`（当然，它们也可以是相同类型）。

接下来，我们看到函数的定义。它同样以`template <class Type1, class Type2>`的声明开始。与原型类似，函数头`Type2 ChooseFirst(Type1 x, Type2 y)`表明形式参数`x`和`y`分别是`Type1`和`Type2`类型。这个函数的主体相当直接。我们只需通过使用`<`运算符进行简单的比较，来确定两个参数中哪一个应该在两个值排序中排在前面。

现在，在 `main()` 函数中，当编译器的预处理部分看到对 `ChooseFirst()` 的调用，并带有实际参数 `int value1` 和 `float value3` 时，预处理程序会注意到 `ChooseFirst()` 是一个模板函数。如果还没有这样的 `ChooseFirst()` 版本来处理 `int` 和 `float` 类型，预处理程序会复制这个模板函数，并将 `Type1` 替换为 `int`，将 `Type2` 替换为 `float` —— 代表我们创建适合我们需求的这个函数的适当版本。注意，当调用 `ChooseFirst(value2, value1)` 并且类型都是整数时，模板函数在预处理程序再次在我们的代码中（在幕后）展开时，`Type1` 和 `Type2` 的占位符类型都将被替换为 `int`。

虽然 `ChooseFirst()` 是一个简单的函数，但通过它我们可以看到创建泛化关键数据类型的模板函数的直接机制。我们还可以看到预处理程序如何注意到模板函数的使用，并代表我们承担起根据我们的特定类型使用情况展开这个函数的努力。

让我们看看这个程序的输出：

```cpp
First: 4
First: 4
```

现在我们已经了解了模板函数的基本机制，让我们继续前进，了解我们如何扩展这个过程以包括模板类。

## 创建和使用模板类

**模板类**参数化了类定义的最终类型，并且还需要为任何需要知道正在操作的核心数据类型的任何方法提供模板成员函数。

关键字 `template` 和 `class`，以及尖括号 `<` `>` 和类型名称的占位符，用于指定模板类定义。

让我们看看一个模板类定义及其支持的模板成员函数。这个例子可以作为完整的程序（包含必要的 `#include` 和 `using` 语句）在我们的 GitHub 仓库中找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex2.cpp)

```cpp
template <class Type>   // template class preamble
class Array
{
private:
    int numElements = 0;   // in-class initialization
    Type *contents = nullptr;// dynamically allocated array
public:
    // Constructor and destructor will allocate, deallocate
    // heap memory to allow Array to be fluid in its size.
    // Later, you can use a smart pointer – or use the STL
    // vector class (we're building a similar class here!)
    Array(int size): numElements(size), 
                     contents(new Type [size])
    { // note: allocation is handled in member init. list
    }
    ~Array() { delete [] contents; }  
    void Print() const;     
    Type &operator[](int index) // returns a referenceable
    {               // memory location or throws exception
        if (index < numElements) 
            return contents[index];
        else   // index is out of bounds
            throw std::out_of_range
                             (std::to_string (index));    
    }                                
    void operator+(Type);   // prototype only
};
template <class Type>
void Array<Type>::operator+(Type item)  
{
    // resize array as necessary, add new data element and
    // increment numElements
}
template <class Type>
void Array<Type>::Print() const
{
    for (int i = 0; i < numElements; i++)
        cout << contents[i] << " ";
    cout << endl;
}
int main()
{                    
    // Creation of int Array will trigger template
    // expansion by the preprocessor.
    Array<int> a1(3); // Create an int Array of 3 elements
    try    // operator[] could throw an exception
    {
        a1[2] = 12;      
        a1[1] = 70;       // a1.operator[](1) = 70;
        a1[0] = 2;
        a1[100] = 10;// this assignment throws an exception
    }
    catch (const std::out_of_range &e)
    {
        cout << "Out of range: index " << e.what() << endl;
    } 
    a1.Print();
}   
```

在前面的类定义中，我们首先注意到 `template <class Type>` 的模板类前缀。这个前缀指定即将到来的类定义将是一个模板类，并且占位符 `Type` 将用于泛化在这个类中主要使用的数据类型。

我们接下来看到`Array`类的定义。数据成员`contents`将是`Type`类型的占位符。当然，并不是所有数据类型都需要泛型化。数据成员`int numElements`作为一个整数是完全可以接受的。接下来，我们看到一系列成员函数的原型定义和一些内联定义，包括重载的`operator[]`。对于内联定义的成员函数，在函数定义前不需要模板声明。我们只需要对内联函数进行泛型化，使用我们的占位符`Type`。

现在，让我们看看选定的成员函数。在构造函数中，我们注意到`contents = new Type [size];`的内存分配仅仅使用了占位符`Type`代替实际的数据类型。同样，对于重载的`operator[]`，这个方法的返回类型也是`Type`。

然而，当我们查看一个非内联的成员函数时，我们注意到`template <class Type>`的模板声明必须位于成员函数定义之前。例如，让我们考虑`void Array<Type>::operator+(Type item);`的成员函数定义。除了声明之外，在函数定义中，类名（在成员函数名称和作用域解析运算符`::`之前）必须增加包括占位符类型`<Type>`的尖括号。此外，任何泛型函数参数都必须使用`Type`的占位符类型。

现在，在我们的`main()`函数中，我们仅仅使用`Array<int>`的数据类型来实例化一个安全、易于调整大小的整数数组。如果我们想实例化一个浮点数数组，我们可以使用`Array<float>`。在底层，当我们创建特定数组类型的实例时，预处理器会注意到我们是否已经为该类型扩展了此类。如果没有，类定义和相关的模板成员函数会为我们复制，并且占位符类型会被替换为我们需要的类型。这并不比我们自己复制、粘贴并稍作修改代码少；然而，重点是，我们只需要指定和维护一个版本。这更少出错，并且更容易进行长期维护。

让我们看看这个程序的输出：

```cpp
2 70 12
```

一个有趣的话题——std::optional

在前面的示例中，`Array<Type>::operator[]`在所选索引超出范围时抛出`out_of_range`异常。有时，异常处理可能具有程序上的成本。在这种情况下，使用可选返回类型可能是一个有用的替代方案。记住，`operator[]`的有效返回值是对相关数组元素内存位置的引用。对于超出范围的索引场景，我们知道我们无法从这个方法中返回数组元素的相应内存位置（这没有意义），因此异常处理的替代方案可能是使用`std::optional<Type>`作为函数的返回值。

让我们接下来看看一个不同的完整程序示例，以结合模板函数和模板类。

## 检查一个完整程序示例

看到一个额外的示例，说明模板函数和模板类是有用的。让我们扩展我们在*第十二章*中最近审查的`LinkList`程序，*友元和运算符重载*；我们将升级此程序以利用模板。

这个完整的程序可以在我们的 GitHub 仓库中找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter13/Chp13-Ex3.cpp)

```cpp
#include <iostream>
using std::cout;   // preferred to: using namespace std;
using std::endl;
// forward declaration with template preamble
template <class Type> class LinkList;  
template <class Type>   // template preamble for class def.
class LinkListElement
{
private:
    Type *data = nullptr;
    LinkListElement *next = nullptr;
    // private access methods to be used in scope of friend
    Type *GetData() const { return data; } 
    LinkListElement *GetNext() const { return next; }
    void SetNext(LinkListElement *e) { next = e; }
public:
    friend class LinkList<Type>;   
    LinkListElement() = default;
    LinkListElement(Type *i): data(i), next(nullptr) { }
    ~LinkListElement(){ delete data; next = nullptr; }
};
// LinkList should only be extended as a protected/private
// base class; it does not contain a virtual destructor. It
// can be used as-is, or as implementation for another ADT.
template <class Type>
class LinkList
{
private:
    LinkListElement<Type> *head = nullptr, *tail = nullptr,
                                 *current = nullptr;
public:
    LinkList() = default;
    LinkList(LinkListElement<Type> *e) 
        { head = tail = current = e; }
    void InsertAtFront(Type *);
    LinkListElement<Type> *RemoveAtFront();  
    void DeleteAtFront()  { delete RemoveAtFront(); }
    bool IsEmpty() const { return head == nullptr; } 
    void Print() const;    
    ~LinkList(){ while (!IsEmpty()) DeleteAtFront(); }
};
```

让我们检查`LinkListElement`和`LinkList`的前置模板类定义。最初，我们注意到`LinkList`类的声明中包含了必要的模板前缀`template class <Type>`。我们还应该注意到，每个类定义本身也包含相同的模板前缀，以双重指定该类将是一个模板类，并且数据类型的占位符将是标识符`Type`。

在`LinkListElement`类中，请注意数据类型将是`Type`（占位符类型）。同时请注意，在`LinkList`的友元类指定中，类型占位符将是必要的，即`friend class LinkList<Type>;`。

在`LinkList`类中，请注意对`LinkListElement`相关类的任何引用都将包括类型占位符`<Type>`。例如，注意在`LinkListElement<Type> *head;`的数据成员声明中或`RemoveAtFront()`的返回类型中的占位符使用，其返回类型为`LinkListElement<Type>`。此外，请注意内联函数定义不需要在每个方法前使用模板前缀；我们仍然受到在类定义本身之前出现的模板前缀的保护。

现在，让我们继续前进，看看`LinkList`类的三个非内联成员函数：

```cpp
template <class Type>     // template preamble
void LinkList<Type>::InsertAtFront(Type *theItem)
{
    LinkListElement<Type> *newHead = nullptr;
    newHead = new LinkListElement<Type>(theItem);
    newHead->SetNext(head);  // newHead->next = head;
    head = newHead;
}
template <class Type>    // template preamble
LinkListElement<Type> *LinkList<Type>::RemoveAtFront()
{
    LinkListElement<Type> *remove = head;
    head = head->GetNext();  // head = head->next;
    current = head;    // reset current for usage elsewhere
    return remove;
}

template <class Type>    // template preamble
void LinkList<Type>::Print() const
{
    if (!head)
        cout << "<EMPTY>" << endl;
    LinkListElement<Type> *traverse = head;
    while (traverse)
    {
        Type output = *(traverse->GetData());
        cout << output << ' ';
        traverse = traverse->GetNext();
    }
    cout << endl;
}
```

在检查前面的代码时，我们可以看到在`LinkList`的非内联方法中，`template <class Type>`模板前缀出现在每个成员函数定义之前。我们还可以看到，与作用域解析运算符结合的类名被`<Type>`增强，例如，`void LinkList<Type>::Print()`。

我们注意到上述模板成员函数需要它们的方法的一部分来使用占位符类型`Type`。例如，`InsertAtFront(Type *theItem)`方法使用占位符`Type`作为形式参数`theItem`的数据类型，并在声明局部指针变量`temp`时指定相关的类`LinkListElement<Type>`。`RemoveAtFront()`方法同样使用类型为`LinkListElement<Type>`的局部变量，因此需要将其用作模板函数。同样，`Print()`引入了一个类型为`Type`的局部变量来帮助输出。

现在我们来看看我们的`main()`函数，看看我们如何利用我们的模板类：

```cpp
int main()
{
    LinkList<int> list1;  // create a LinkList of integers
    list1.InsertAtFront(new int (3000));
    list1.InsertAtFront(new int (600));
    list1.InsertAtFront(new int (475));
    cout << "List 1: ";
    list1.Print();
    // delete elements from list, one by one
    while (!(list1.IsEmpty()))
    {
       list1.DeleteAtFront();
       cout << "List 1 after removing an item: ";
       list1.Print();
    }
    LinkList<float> list2;  // create a LinkList of floats
    list2.InsertAtFront(new float(30.50));
    list2.InsertAtFront(new float (60.89));
    list2.InsertAtFront(new float (45.93));
    cout << "List 2: ";
    list2.Print();
}
```

在我们之前的`main()`函数中，我们使用我们的模板类创建两种类型的链表，即声明为`LinkList<int> list1;`的整数`LinkList`和声明为`LinkList<float> list2;`的浮点数`LinkList`。

在每种情况下，我们实例化各种链表，然后添加元素并打印相应的列表。在第一个`LinkList`实例的情况下，我们还展示了如何从列表中连续删除元素。

让我们看看这个程序的输出：

```cpp
List 1: 475 600 3000
List 1 after removing an item: 600 3000
List 1 after removing an item: 3000
List 1 after removing an item: <EMPTY>
List 2: 45.93 60.89 30.5
```

总体来看，我们看到创建`LinkList<int>`和`LinkList<float>`非常容易。模板代码在幕后简单地扩展以适应我们希望的数据类型。然后，我们可以问自己，创建`Student`实例的链表有多容易？非常容易！我们可以简单地实例化`LinkList<Student> list3;`并调用适当的`LinkList`方法，例如`list3.InsertAtFront(new Student("George", "Katz", 'C', "Mr.", 3.2, "C++", "123GWU"));`。

也许我们希望在模板`LinkList`类中包含一种对元素进行排序的方法，例如，通过添加一个`OrderedInsert()`方法（这通常依赖于`operator<`或`operator>`来进行元素比较）。这对所有数据类型都适用吗？这是一个好问题。如果方法中编写的代码对所有数据类型都是通用的，那么它就可以。操作符重载能帮助这个任务吗？是的！

现在我们已经看到了模板类和函数的机制，让我们考虑如何确保我们的模板类和函数能够完全扩展以适用于任何数据类型。为此，让我们考虑操作符重载如何有价值。

# 使模板更加灵活和可扩展

C++中模板的引入使我们能够通过程序员一次指定某些类型的类和函数，而在幕后，预处理器为我们生成许多版本的代码。然而，为了使一个类真正可扩展以扩展到许多不同的用户定义类型，成员函数中编写的代码必须适用于任何类型的数据。为了帮助这一努力，运算符重载可以用来扩展可能容易存在于标准类型中的操作，包括为用户定义类型提供定义。

总结一下，我们知道运算符重载可以使简单的运算符不仅与标准类型一起工作，还可以与用户定义的类型一起工作。通过在我们的模板代码中重载运算符，我们可以确保我们的模板代码具有高度的复用性和可扩展性。

让我们考虑一下如何通过运算符重载来加强模板。

## 通过添加运算符重载进一步泛化模板代码

回想一下，当重载运算符时，重要的是要传达与标准类型相同的运算符意义。想象一下，如果我们想给我们的`LinkList`类添加一个`OrderedInsert()`方法。这个成员函数的主体可能需要比较两个元素以确定哪个应该排在另一个之前。做到这一点最简单的方法是使用`operator<`。这个运算符很容易定义为与标准类型一起工作，但它会与用户定义的类型一起工作吗？它可以，前提是我们重载运算符以与所需类型一起工作。

让我们看看一个例子，我们将需要重载一个运算符以使成员函数代码具有普遍适用性：

```cpp
template <class Type>
void LinkList<Type>::OrderedInsert(Type *theItem)
{
    current = head;    
    if (*theItem < *(head->GetData()))  
        InsertAtFront(theItem);  // add theItem before head
    else
        // Traverse list, add theItem in proper location
}
```

在前面的模板成员函数中，我们依赖于`operator<`能够与任何我们希望利用此模板类的数据类型一起工作。也就是说，当预处理器为特定的用户定义类型展开此代码时，`<`运算符必须适用于此方法被特定展开的任何数据类型。

如果我们希望创建一个包含`Student`实例的`LinkList`并应用一个`Student`实例相对于另一个实例的`OrderedInsert()`，那么我们需要确保两个`Student`实例之间的比较`operator<`是定义好的。当然，默认情况下，`operator<`只为标准类型定义。但是，如果我们简单地为`Student`重载`operator<`，我们可以确保`LinkList<Type>::OrderedInsert()`方法也将适用于`Student`数据类型。

让我们看看如何为`Student`实例重载`operator<`，无论是作为成员函数还是作为非成员函数：

```cpp
// overload operator < As a member function of Student
bool Student::operator<(const Student &s)
{   // if this->gpa < s.gpa return true, else return false
    return this->gpa < s.gpa;
}
// OR, overload operator < as a non-member function
bool operator<(const Student &s1, const Student &s2)
{   // if s1.gpa < s2.gpa return true, else return false
    return s1.gpa < s2.gpa;
}
```

在前面的代码中，我们可以识别出 `operator<` 被实现为 `Student` 的成员函数或非成员函数。如果你可以访问 `Student` 类的定义，那么首选的方法是使用该运算符函数的成员函数定义。然而，有时我们无法修改类。在这种情况下，我们必须使用非成员函数方法。无论如何，在两种实现中，我们只是比较两个 `Student` 实例的 `gpa`，如果第一个实例的 `gpa` 低于第二个 `Student` 实例，则返回 `true`，否则返回 `false`。

现在已经为两个 `Student` 实例定义了 `operator<`，我们可以回到之前的模板函数 `LinkList<Type>::OrderedInsert(Type *)`，该函数在 `LinkList` 中使用 `operator <` 来比较类型为 `Type` 的两个对象。当在代码的某个地方创建 `LinkList<Student>` 时，`LinkList` 和 `LinkListElement` 的模板代码将由预处理程序为 `Student` 展开；`Type` 将被替换为 `Student`。当展开的代码被编译时，展开的 `LinkList<Student>::OrderedInsert()` 代码将无错误编译，因为已经为两个 `Student` 对象定义了 `operator<`。

然而，如果我们忽略为给定类型重载 `operator<`，但在我们的代码中从未调用 `OrderedInsert()`（或依赖于 `operator<` 的其他方法）在相同展开的模板类型对象上，会发生什么？信不信由你，代码将编译并正常工作。在这种情况下，我们实际上没有调用需要为该类型实现 `operator<` 的函数（即 `OrderedInsert()`）。因为该函数从未被调用，所以该成员函数的模板展开被跳过。编译器没有理由发现应该为该类型重载 `operator<`（以便方法能够成功编译）。未调用的方法只是没有被编译器验证展开。

通过使用运算符重载来补充模板类和函数，我们可以通过确保方法体中使用的典型运算符可以应用于模板展开中我们想要利用的任何类型，从而使模板代码更加可扩展。我们的代码变得更加通用。

我们现在已经看到了如何使用模板函数和类，以及运算符重载如何增强模板以创建更可扩展的代码。在继续下一章之前，让我们简要回顾这些概念。

# 摘要

在本章中，我们将 C++ 编程知识从面向对象语言特性扩展到包括其他语言特性，这些特性将使我们能够编写更可扩展的代码。我们学习了如何使用模板函数和模板类，以及运算符重载如何很好地支持这些工作。

我们已经看到，模板可以让我们根据类或函数内部主要使用的数据类型泛型地指定一个类或函数。我们已经看到，模板类不可避免地会使用模板函数，因为这些方法通常需要泛型地使用构建类所依据的数据。我们已经看到，通过利用用户定义类型的运算符重载，我们可以利用使用简单运算符编写的代码体，以适应更复杂的数据类型的用法，从而使模板代码更加有用和可扩展。

模板的力量加上运算符重载（使方法可用于几乎所有类型）使得 C++对泛型的实现比简单的类型替换要强大得多。

我们现在明白，使用模板可以让我们更抽象地指定一个类或函数一次，并允许预处理器根据应用程序中可能需要的特定数据类型为我们生成该类或函数的多个版本。

通过允许预处理器根据应用程序中需要的类型为我们扩展模板类或模板函数的多个版本，创建许多类似类或函数（以及维护这些版本）的工作就转移给了 C++，而不是程序员。除了用户需要维护的代码更少之外，对模板类或函数所做的更改只需在一个地方进行——当需要时，预处理器将重新扩展代码而不会出错。

通过研究模板，我们已经增加了额外的、有用的功能到我们的 C++工具箱中，这些功能与运算符重载相结合，将确保我们可以为几乎所有数据类型编写高度可扩展和可重用的代码。我们现在准备好继续前进到*第十四章*，*理解 STL 基础*，这样我们就可以继续使用有用的 C++库功能来扩展我们的 C++编程技能，使我们成为更好的程序员。让我们继续前进！

# 问题

1.  将你的`ArrayInt`类从*第十二章*，*友元和运算符重载*，转换为模板`Array`类，以支持任何数据类型的动态分配数组，这些数组可以轻松调整大小并具有内置的边界检查。

a. 考虑你将需要重载哪些运算符，以便在每种方法内的泛型代码支持你可能在模板`Array`类型中希望存储的任何用户定义类型。

b. 使用你的模板`Array`类，创建一个`Student`实例的数组。利用各种成员函数来演示各种模板函数能正确运行。

1.  使用模板`LinkList`类，完成`LinkList<Type>::OrderedInsert()`的实现。在`main()`函数中创建一个`Student`实例的`LinkList`。在列表中使用`OrderedInsert()`插入几个`Student`实例后，通过显示每个`Student`及其`gpa`来验证此方法是否正确工作。`Student`实例应按最低到最高的`gpa`顺序排列。您可能希望将在线代码作为起点。
