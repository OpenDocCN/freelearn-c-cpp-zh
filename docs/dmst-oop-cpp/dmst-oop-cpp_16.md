# 第十三章：使用模板

本章将继续追求扩展您的 C++编程技能，超越面向对象编程概念，继续编写更具可扩展性的代码。我们将探索使用 C++模板创建通用代码 - 包括**模板函数**和**模板类**。我们将学习如何编写正确的模板代码，以实现代码重用的最高境界。我们将探讨如何创建模板函数和模板类，以及理解适当使用运算符重载如何使模板函数可重用于几乎任何类型的数据。

在本章中，我们将涵盖以下主要主题：

+   探索模板基础知识以通用化代码

+   理解如何创建和使用模板函数和模板类

+   理解运算符重载如何使模板更具可扩展性

通过本章结束时，您将能够通过构建模板函数和模板类来设计更通用的代码。您将了解运算符重载如何确保模板函数对任何数据类型都具有高度可扩展性。通过将精心设计的模板成员函数与运算符重载配对使用，您将能够在 C++中创建高度可重用和可扩展的模板类。

让我们通过探索模板来扩展您的编程技能，从而增进对 C++的理解。

# 技术要求

完整程序示例的在线代码可在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下，文件名与所在章节编号对应，后跟破折号，再跟上所在章节中示例编号。例如，本章的第一个完整程序可以在`Chapter13`子目录中的名为`Chp13-Ex1.cpp`的文件中找到，位于上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/2OUaLrb`](https://bit.ly/2OUaLrb)。

# 探索模板基础知识以通用化代码

模板允许以一种抽象的方式对代码进行通用指定，这种方式与主要用于相关函数或类中的数据类型无关。创建模板的动机是为了通用指定我们反复想要使用的函数和类的定义，但使用不同的数据类型。这些组件的个性化版本在核心数据类型上会有所不同；这些关键数据类型可以被提取并以通用方式编写。

当我们选择使用特定类型的类或函数时，而不是复制和粘贴现有代码（带有预设数据类型）并稍作修改，预处理器会取代模板代码并为我们请求的类型进行*扩展*。这种模板*扩展*能力使程序员只需编写和维护通用化代码的一个版本，而不是需要编写许多特定类型版本的代码。另一个好处是，预处理器将更准确地将模板代码扩展为请求的类型，而不是我们可能使用复制、粘贴和轻微修改方法所做的扩展。

让我们花点时间进一步探讨在我们的代码中使用模板的动机。

## 审视使用模板的动机

假设我们希望创建一个类来安全地处理动态分配的`int`数据类型的数组，就像我们在*第十二章*的*问题 3*解决方案中创建的那样，*运算符重载和友元*。我们的动机可能是要有一个数组类型，可以增长或缩小到任何大小（不像本地的固定大小数组），但对于安全使用有边界检查（不像使用`int *`实现的动态数组的原始操作，它会肆意地允许我们访问远远超出我们动态数组分配长度的元素）。

我们可能决定创建一个以下开始框架的`ArrayInt`类：

```cpp
class ArrayInt
{
private:
    int numElements;
    int *contents;   // dynamically allocated array
public:
    ArrayInt(int size) : numElements(size) 
    { 
        contents = new int [size];
    }
    ~ArrayInt() { delete contents; }       
    int &operator[](int index) // returns a referenceable
    {                          // memory location 
        if (index < numElements) return contents[index];
        else cout << "Out of Bounds"; // or better – throw an
    }                                 // OutOfBounds exception
};
int main()
{
    ArrayInt a1(5); // Create an ArrayInt of 5 elements
    a1[4] = 7;      // a1.operator[](4) = 7;
}   
```

在前面的代码段中，请注意我们的`ArrayInt`类使用`int *contents;`来模拟数组的数据，它在构造函数中动态分配到所需的大小。我们已经重载了`operator[]`，以安全地返回数组中范围内的索引值。我们可以添加`Resize()`和`ArrayInt`等方法。总的来说，我们喜欢这个类的安全性和灵活性。

现在，我们可能想要有一个`ArrayFloat`类（或者以后是`ArrayStudent`类）。例如，我们可能会问是否有一种更自动化的方法来进行这种替换，而不是复制我们的基线`ArrayInt`类并稍微修改它以创建一个`ArrayFloat`类。毕竟，如果我们使用`ArrayInt`类作为起点创建`ArrayFloat`类，我们会改变什么呢？我们会改变数据成员`contents`的*类型* - 从`int *`到`float *`。我们会在构造函数中改变内存分配中的*类型*，从`contents = new int [size];`到使用`float`而不是`int`（以及在任何重新分配中也是如此，比如在`Resize()`方法中）。

与其复制、粘贴和稍微修改`ArrayInt`类以创建`ArrayFloat`类，我们可以简单地使用**模板类**来泛型化与该类中操作的数据相关联的*类型*。同样，依赖于特定数据类型的任何函数将成为**模板函数**。我们将很快研究创建和使用模板的语法。

使用模板，我们可以创建一个名为`Array`的模板类，其中类型是泛型化的。在编译时，如果预处理器检测到我们在代码中使用了这个类来处理`int`或`float`类型，那么预处理器将为我们提供必要的模板*扩展*。也就是说，通过复制和粘贴（在幕后）每个模板类（及其方法）并替换预处理器识别出我们正在使用的数据类型。

扩展后的代码在幕后并不比我们自己为每个单独的类编写代码要小。但关键是，我们不必费力地创建、修改、测试和后续维护每个略有不同的类。这是 C++代表我们完成的。这就是模板类和模板函数的值得注意的目的。

模板不仅限于与原始数据类型一起使用。例如，我们可能希望创建一个用户定义类型的`Array`，比如`Student`。我们需要确保我们的模板成员函数对我们实际扩展模板类以利用的数据类型是有意义的。我们可能需要重载选定的运算符，以便我们的模板成员函数可以与用户定义的类型无缝地工作，就像它们与原始类型一样。

在本章的后面部分，我们将看到一个例子，说明如果我们选择扩展模板类以适用于用户定义的类型，我们可能需要重载选定的运算符，以便类的成员函数可以与任何数据类型流畅地工作。幸运的是，我们知道如何重载运算符！

让我们继续探索指定和利用模板函数和模板类的机制。

# 理解模板函数和类

**模板**通过抽象与这些函数和类相关的数据类型，提供了创建通用函数和类的能力。模板函数和类都可以被精心编写，以使这些函数和类的相关数据类型通用化。

让我们首先来看看如何创建和利用模板函数。

## 创建和使用模板函数

**模板函数**将函数中的参数类型参数化，除了参数本身。模板函数要求函数体适用于大多数任何数据类型。模板函数可以是成员函数或非成员函数。运算符重载可以帮助确保模板函数的函数体适用于用户定义的类型 - 我们很快会看到更多。

关键字`template`，以及尖括号`<` `>`和*类型*名称的占位符，用于指定模板函数及其原型。

让我们来看一个不是类成员的模板函数（我们将很快看到模板成员函数的例子）。这个例子可以在我们的 GitHub 仓库中找到，作为一个完整的工作程序，如下所示：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex1.cpp)

```cpp
// template function prototype
template <class Type1, class Type2>   // template preamble
Type2 ChooseFirst(Type1, Type2);
// template function definition
template <class Type1, class Type2>  // template preamble
Type2 ChooseFirst(Type1 x, Type2 y)
{
    if (x < y) return (Type2) x;
    else return y; 
}   
int main()
{
    int value1 = 4, value2 = 7;
    float value3 = 5.67f;
    cout << "First: " << ChooseFirst(value1, value3) << endl;
    cout << "First: " << ChooseFirst(value2, value1) << endl;
}
```

看一下前面的函数示例，我们首先看到一个模板函数原型。前言`template <class Type1, class Type 2>`表示原型将是一个模板原型，并且占位符`Type1`和`Type2`将被用来代替实际数据类型。占位符`Type1`和`Type2`可以是（几乎）任何名称，遵循创建标识符的规则。

然后，为了完成原型，我们看到`Type2 ChooseFirst(Type1, Type2);`，这表明这个函数的返回类型将是`Type2`，`ChooseFirst()`函数的参数将是`Type1`和`Type2`（它们肯定可以扩展为相同的类型）。

接下来，我们看到函数定义。它也以`template <class Type1, class Type 2>`开头。与原型类似，函数头`Type2 ChooseFirst(Type1 x, Type2 y)`表示形式参数`x`和`y`分别是类型`Type1`和`Type2`。这个函数的主体非常简单。我们只需使用`<`运算符进行简单比较，确定这两个参数中哪一个应该在这两个值的排序中排在第一位。

现在，在`main()`中，当编译器的预处理部分看到对`ChooseFirst()`的调用，实际参数为`int value1`和`float value3`时，预处理器注意到`ChooseFirst()`是一个模板函数。如果还没有这样的`ChooseFirst()`版本来处理`int`和`float`，预处理器将复制这个模板函数，并用`int`替换`Type1`，用`float`替换`Type2` - 为我们创建适合我们需求的函数的适当版本。请注意，当调用`ChooseFirst(value2, value1)`并且类型都是整数时，当预处理器再次扩展（在代码底层）模板函数时，占位符类型`Type1`和`Type2`将都被`int`替换。

虽然`ChooseFirst()`是一个简单的函数，但通过它，我们可以看到创建通用关键数据类型的模板函数的简单机制。我们还可以看到预处理器注意到模板函数的使用方式，并代表我们扩展这个函数，根据我们特定的类型使用需求。

让我们来看一下这个程序的输出：

```cpp
First: 4
First: 4
```

现在我们已经看到了模板函数的基本机制，让我们继续了解如何将这个过程扩展到包括模板类。

## 创建和使用模板类

**模板类**参数化类定义的最终类型，并且还需要模板成员函数来处理需要知道被操作的核心数据类型的任何方法。

关键字`template`和`class`，以及尖括号`<``>`和*type*名称的占位符，用于指定模板类定义。

让我们来看一个模板类定义及其支持的模板成员函数。这个例子可以在我们的 GitHub 存储库中找到，作为一个完整的程序。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex2.cpp)

```cpp
template <class Type>   // template class preamble
class Array
{
private:
    int numElements;
    Type *contents;   // dynamically allocated array
public:
    Array(int size) : numElements(size)
    { 
        contents = new Type [size];
    }
    ~Array() { delete contents; }  
    void Print() const;     
    Type &operator[](int index) // returns a referenceable
    {                          // memory location 
        if (index < numElements) return contents[index];
        else cout << "Out of Bounds"; // or better – throw an
    }                                 // OutOfBounds exception
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
    // Creation of int array will trigger template expansion
    Array<int> a1(3); // Create an int Array of 3 int elements
    a1[2] = 12;      
    a1[1] = 70;       // a1.operator[](1) = 70;
    a1[0] = 2;
    a1.Print();
}   
```

在前面的类定义中，让我们首先注意`template <class Type>`的模板类前言。这个前言指定了即将到来的类定义将是一个模板类，占位符`Type`将用于泛型化主要在这个类中使用的数据类型。

然后我们看到了`Array`的类定义。数据成员`contents`将是占位符类型`Type`。当然，并不是所有的数据类型都需要泛型化。数据成员`int numElements`作为整数是完全合理的。接下来，我们看到了一系列成员函数的原型，以及一些内联定义的成员函数，包括重载的`operator[]`。对于内联定义的成员函数，在函数定义前不需要模板前言。我们唯一需要做的是使用我们的占位符`Type`泛型化数据类型。

现在让我们来看一下选定的成员函数。在构造函数中，我们现在注意到`contents = new Type [size];`的内存分配仅仅使用了占位符`Type`而不是实际的数据类型。同样，对于重载的`operator[]`，这个方法的返回类型是`Type`。

然而，看一个不是内联的成员函数，我们注意到模板前言`template <class Type>`必须在成员函数定义之前。例如，让我们考虑`void Array<Type>::operator+(Type item);`的成员函数定义。除了前言之外，在函数定义中类名（在成员函数名和作用域解析运算符`::`之前）必须增加占位符类型`<Type>`在尖括号中。此外，任何通用函数参数必须使用占位符类型`Type`。

现在，在我们的`main()`函数中，我们仅使用`Array<int>`的数据类型来实例化一个安全、易于调整大小的整数数组。如果我们想要实例化一个浮点数数组，我们可以选择使用`Array<float>`。在幕后，当我们创建特定数组类型的实例时，预处理器会注意到我们是否先前为该*type*扩展了这个类。如果没有，类定义和适用的模板成员函数将被复制，占位符类型将被替换为我们需要的类型。这并不比我们自己复制、粘贴和稍微修改代码少一行；然而，重点是我们只需要指定和维护一个版本。这样做更不容易出错，更容易进行长期维护。

让我们来看一下这个程序的输出：

```cpp
2 70 12
```

接下来让我们看一个不同的完整程序例子，来整合模板函数和模板类。

## 检查一个完整的程序例子

有必要看一个额外的例子，说明模板函数和模板类。让我们扩展我们最近在*第十二章*中审查的`LinkList`程序，*运算符重载和友元*；我们将升级这个程序以利用模板。

这个完整的程序可以在我们的 GitHub 存储库中找到。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter13/Chp13-Ex3.cpp)

```cpp
#include <iostream>
using namespace std;
template <class Type> class LinkList;  // forward declaration
                                     // with template preamble
template <class Type>   // template preamble for class def
class LinkListElement
{
private:
    Type *data;
    LinkListElement *next;
    // private access methods to be used in scope of friend
    Type *GetData() { return data; } 
    LinkListElement *GetNext() { return next; }
    void SetNext(LinkListElement *e) { next = e; }
public:
    friend class LinkList<Type>;   
    LinkListElement() { data = 0; next = 0; }
    LinkListElement(Type *i) { data = i; next = 0; }
    ~LinkListElement(){ delete data; next = 0;}
};
// LinkList should only be extended as a protected or private
// base class; it does not contain a virtual destructor. It
// can be used as-is, or as implementation for another ADT.
template <class Type>
class LinkList
{
private:
    LinkListElement<Type> *head, *tail, *current;
public:
    LinkList() { head = tail = current = 0; }
    LinkList(LinkListElement<Type> *e) 
        { head = tail = current = e; }
    void InsertAtFront(Type *);
    LinkListElement<Type> *RemoveAtFront();  
    void DeleteAtFront()  { delete RemoveAtFront(); }
    int IsEmpty() { return head == 0; } 
    void Print();    
    ~LinkList(){ while (!IsEmpty()) DeleteAtFront(); }
};
```

让我们来检查`LinkListElement`和`LinkList`的前面的模板类定义。最初，我们注意到`LinkList`类的前向声明包含了必要的`template class <Type>`的模板前言。我们还应该注意到每个类定义本身都包含相同的模板前言，以双重指定该类将是一个模板类，并且数据类型的占位符将是标识符`Type`。

在`LinkListElement`类中，注意到数据类型将是`Type`（占位符类型）。另外，注意到类型的占位符在`LinkList`的友元类规范中是必要的，即`friend class LinkList<Type>;`。

在`LinkList`类中，注意到任何与`LinkListElement`的关联类的引用都将包括`<Type>`的类型占位符。例如，在`LinkListElement<Type> *head;`的数据成员声明中或者`RemoveAtFront()`的返回类型中，都使用了占位符。此外，注意到内联函数定义不需要在每个方法之前加上模板前言；我们仍然受到类定义本身之前的前言的覆盖。

现在，让我们继续来看看`LinkList`类的三个非内联成员函数：

```cpp
template <class Type>     // template preamble
void LinkList<Type>::InsertAtFront(Type *theItem)
{
    LinkListElement<Type> *temp;
    temp = new LinkListElement<Type>(theItem);
    temp->SetNext(head);  // temp->next = head;
    head = temp;
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
void LinkList<Type>::Print()
{
    Type output;
    if (!head)
        cout << "<EMPTY>" << endl;
    current = head;
    while (current)
    {
        output = *(current->GetData());
        cout << output << " ";
        current = current->GetNext();
    }
    cout << endl;
}
```

当我们检查前面的代码时，我们可以看到在`LinkList`的非内联方法中，`template <class Type>`的模板前言出现在每个成员函数定义之前。我们还看到与作用域解析运算符相关联的类名被增加了`<Type>`；例如，`void LinkList<Type>::Print()`。

我们注意到前面提到的模板成员函数需要利用占位符类型`Type`的一部分来实现它们的方法。例如，`InsertAtFront(Type *theItem)`方法将占位符`Type`用作形式参数`theItem`的数据类型，并在声明一个本地指针变量`temp`时指定关联类`LinkListElement<Type>`。`RemoveAtFront()`方法类似地利用了类型为`LinkListElement<Type>`的本地变量，因此需要将其用作模板函数。同样，`Print()`引入了一个类型为`Type`的本地变量来辅助输出。

现在让我们来看看我们的`main()`函数，看看我们如何利用我们的模板类：

```cpp
int main()
{
    LinkList<int> list1; // create a LinkList of integers
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
    LinkList<float> list2;  // now make a LinkList of floats
    list2.InsertAtFront(new float(30.50));
    list2.InsertAtFront(new float (60.89));
    list2.InsertAtFront(new float (45.93));
    cout << "List 2: ";
    list2.Print();
}
```

在我们前面的`main()`函数中，我们利用我们的模板类创建了两种类型的链表，即整数的`LinkList`声明为`LinkList<int> list1;`和浮点数的`LinkList`声明为`LinkList<float> list2;`。

在每种情况下，我们实例化各种链表，然后添加元素并打印相应的列表。在第一个`LinkList`实例的情况下，我们还演示了如何连续从列表中删除元素。

让我们来看看这个程序的输出：

```cpp
List 1: 475 600 3000
List 1 after removing an item: 600 3000
List 1 after removing an item: 3000
List 1 after removing an item: <EMPTY>
List 2: 45.93 60.89 30.5
```

总的来说，我们看到创建`LinkList<int>`和`LinkList<float>`非常容易。模板代码在幕后被简单地扩展，以适应我们所需的每种数据类型。然后我们可能会问自己，创建`Student`实例的链表有多容易？非常容易！我们可以简单地实例化`LinkList<Student> list3;`并调用适当的`LinkList`方法，比如`list3.InsertAtFront(new Student("George", "Katz", 'C', "Mr.", 3.2, "C++", "123GWU"));`。

也许我们想在模板`LinkList`类中包含一种方法来对我们的元素进行排序，比如添加一个`OrderedInsert()`方法（通常依赖于`operator<`或`operator>`来比较元素）。这对所有数据类型都适用吗？这是一个很好的问题。只要方法中的代码是通用的，可以适用于所有数据类型，它就可以，运算符重载可以帮助实现这个目标。是的！

现在我们已经看到了模板类和函数的工作原理，让我们考虑如何确保我们的模板类和函数能够完全扩展以适用于任何数据类型。为了做到这一点，让我们考虑运算符重载如何有价值。

# 使模板更灵活和可扩展

在 C++中添加模板使我们能够让程序员一次性地指定某些类型的类和函数，而在幕后，预处理器会代表我们生成许多版本的代码。然而，为了使一个类真正可扩展以适用于许多不同的用户定义类型，成员函数中编写的代码必须普遍适用于任何类型的数据。为了帮助实现这个目标，可以使用运算符重载来扩展可能轻松存在于标准类型的操作，以包括对用户定义类型的定义。

总结一下，我们知道运算符重载可以使简单的运算符不仅适用于标准类型，还适用于用户定义的类型。通过在模板代码中重载运算符，我们可以确保模板代码具有高度的可重用性和可扩展性。

让我们考虑如何通过运算符重载来加强模板。

## 通过添加运算符重载来进一步泛化模板代码。

回想一下，当重载运算符时，重要的是要促进与标准类型相同的含义。想象一下，我们想要在我们的`LinkList`类中添加一个`OrderedInsert()`方法。这个成员函数的主体可能依赖于比较两个元素，以确定哪个应该排在另一个之前。最简单的方法是使用`operator<`。这个运算符很容易定义为与标准类型一起使用，但它是否适用于用户定义的类型？只要我们重载运算符以适用于所需的类型，它就可以适用。

让我们看一个例子，我们需要重载一个运算符，使成员函数代码普遍适用：

```cpp
template <class Type>
void LinkList<Type>::OrderedInsert(Type *theItem)
{
    current = head;    
    if (theItem < head->GetData())  
        InsertAtFront(theItem);   // add theItem before head
    else
        // Traverse list, add theItem in the proper location
}
```

在前面的模板成员函数中，我们依赖于`operator<`能够与我们想要使用这个模板类的任何数据类型一起工作。也就是说，当预处理器为特定的用户定义类型扩展这段代码时，`<`运算符必须适用于此方法特定扩展的任何数据类型。

如果我们希望创建一个`LinkList`的`Student`实例，并对一个`Student`与另一个`Student`进行`OrderedInsert()`，那么我们需要确保为两个`Student`实例定义了`operator<`的比较。当然，默认情况下，`operator<`仅适用于标准类型。但是，如果我们简单地为`Student`重载`operator<`，我们就可以确保`LinkList<Type>::OrderedInsert()`方法也适用于`Student`数据类型。

让我们看看如何为`Student`实例重载`operator<`，无论是作为成员函数还是非成员函数：

```cpp
// overload operator < As a member function of Student
bool Student::operator<(const Student &s)
{
    if (this->gpa < s.gpa)  
        return true;
    else
        return false;
}
// OR, overload operator < as a non-member function
bool operator<(const Student &s1, const Student &s2)
{
    if (s1.gpa < s2.gpa)  
        return true;
    else
        return false;
}
```

在前面的代码中，我们可以识别`operator<`被实现为`Student`的成员函数，或者作为非成员函数。如果你可以访问`Student`类的定义，首选的方法是利用成员函数定义来实现这个运算符函数。然而，有时我们无法访问修改一个类。在这种情况下，我们必须使用非成员函数的方法。无论如何，在任何一种实现中，我们只是比较两个`Student`实例的`gpa`，如果第一个实例的`gpa`低于第二个`Student`实例，则返回`true`，否则返回`false`。

现在`operator<`已经为两个`Student`实例定义了，我们可以回到我们之前的`LinkList<Type>::OrderedInsert(Type *)`模板函数，它利用`LinkList`中类型为`Type`的两个对象进行比较。当我们的代码中某处创建了`LinkList<Student>`时，`LinkList`和`LinkListElement`的模板代码将被预处理器为`Student`进行扩展；`Type`将被替换为`Student`。然后编译扩展后的代码时，扩展的`LinkList<Student>::OrderedInsert()`中的代码将会无错误地编译，因为`operator<`已经为两个`Student`对象定义了。

然而，如果我们忽略为给定类型重载`operator<`会发生什么，然而，`OrderedInsert()`（或者另一个依赖于`operator<`的方法）在我们的代码中对该扩展模板类型的对象从未被调用？信不信由你，代码将会编译并且正常工作。在这种情况下，我们实际上并没有调用一个需要为该类型实现`operator<`的函数（即`OrderedInsert()`）。因为这个函数从未被调用，该成员函数的模板扩展被跳过。编译器没有理由去发现`operator<`应该为该类型重载（为了使方法成功编译）。未被调用的方法只是没有被扩展，以供编译器验证。

通过运算符重载来补充模板类和函数，我们可以通过确保在方法体中使用的典型运算符可以应用于模板扩展中我们想要使用的任何类型，使模板代码变得更具可扩展性。我们的代码变得更加普适。

我们现在已经看到了如何使用模板函数和类，以及如何运算符重载可以增强模板，创建更具可扩展性的代码。在继续前进到下一章之前，让我们简要回顾一下这些概念。

# 总结

在这一章中，我们进一步加强了我们的 C++编程知识，超越了面向对象编程语言特性，包括了额外的语言特性，使我们能够编写更具可扩展性的代码。我们学会了如何利用模板函数和模板类，以及运算符重载如何很好地支持这些努力。

我们已经看到，模板可以让我们以泛型方式指定一个类或函数，与该类或函数中主要使用的数据类型相关。我们已经看到，模板类不可避免地利用模板函数，因为这些方法通常需要泛型地使用构建类的数据。我们已经看到，通过利用用户定义类型的运算符重载，我们可以利用使用简单运算符编写的方法体来适应更复杂的数据类型的使用，使模板代码变得更加有用和可扩展。

我们现在明白，使用模板可以让我们更抽象地指定一个类或函数，让预处理器为我们生成许多该类或函数的版本，基于应用程序中可能需要的特定数据类型。

通过允许预处理器根据应用程序中需要的类型来扩展模板类或一组模板函数的许多版本，创建许多类似的类或函数（并维护这些版本）的工作被传递给了 C++，而不是程序员。除了减少用户需要维护的代码外，模板类或函数中所做的更改只需要在一个地方进行 – 预处理器在需要时将重新扩展代码而不会出错。

我们通过研究模板为我们的 C++技能库增加了额外的有用功能，结合运算符重载，这将确保我们可以为几乎任何数据类型编写高度可扩展和可重用的代码。我们现在准备继续进行*第十四章*，*理解 STL 基础*，以便我们可以继续扩展我们的 C++编程技能，使用有用的 C++库功能，这将使我们成为更好的程序员。让我们继续前进！

# 问题

1.  将您的`ArrayInt`类从*第十二章*，*运算符重载和友元*，转换为一个模板`Array`类，以支持可以轻松调整大小并具有内置边界检查的任何数据类型的动态分配数组。

a. 考虑一下，如果需要的话，您将需要重载哪些运算符，以支持模板的`Array`类型中存储的任何用户定义类型的通用代码。

b. 使用您的模板的`Array`类，创建`Student`实例的数组。利用各种成员函数来演示各种模板函数是否正确运行。

1.  使用模板的`LinkList`类，完成`LinkList<Type>::OrderedInsert()`的实现。在`main()`中创建`Student`实例的`LinkList`。在列表中使用`OrderedInsert()`插入了几个`Student`实例后，通过显示每个`Student`及其`gpa`来验证该方法是否正确工作。`Student`实例应按`gpa`从低到高排序。您可能希望使用在线代码作为起点。
