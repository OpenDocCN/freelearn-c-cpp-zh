# 第二十章：使用 pImpl 模式去除实现细节

本章将结束我们扩展您的 C++编程技能的探索，超越核心面向对象编程概念，旨在进一步赋予您解决重复类型的编码问题的能力，利用常见的设计模式。在编码中应用设计模式不仅可以提供精炼的解决方案，还可以有助于更轻松地维护代码，并提供代码重用的潜力。

我们将学习如何在 C++中有效实现**pImpl 模式**的下一个设计模式。

在本章中，我们将涵盖以下主要主题：

+   理解 pImpl 模式以及它如何减少编译时的依赖关系

+   了解如何在 C++中使用关联和唯一指针实现 pImpl 模式

+   识别与 pImpl 相关的性能问题和必要的权衡

本章结束时，您将了解 pImpl 模式以及如何使用它来将实现细节与类接口分离，以减少编译器依赖性。将额外的设计模式添加到您的技能集中将帮助您成为更有价值的程序员。

让我们通过研究另一个常见的设计模式，pImpl 模式，来增强我们的编程技能。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，文件名与所在章节编号相对应，后跟破折号，再跟随该章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp20-Ex1.cpp`的文件中的`Chapter20`子目录中找到上述 GitHub 目录。一些程序位于适用的子目录中，如示例中所示。

本章的 CiA 视频可在以下链接观看：[`bit.ly/2OT5K1W`](https://bit.ly/2OT5K1W)。

# 理解 pImpl 模式

**pImpl 模式**（**p**ointer to **Impl**ementation idiom）是一种结构设计模式，它将类的实现与其公共接口分离。这种模式最初被**四人组**（**GofF**）称为**桥接模式**，也被称为**切尔西猫**，**编译器防火墙习惯**，**d 指针**，**不透明指针**或**句柄模式**。

该模式的主要目的是最小化编译时的依赖关系。减少编译时的依赖关系的结果是，类定义（尤其是私有访问区域）的更改不会在开发或部署的应用程序中引发及时的重新编译。相反，必要的重新编译代码可以被隔离到类本身的*实现*中。依赖于类定义的应用程序的其他部分将不再受重新编译的影响。

类定义中的私有成员可能会影响类的重新编译。这是因为更改数据成员可能会改变该类型的实例的大小。此外，私有成员函数必须与函数调用的签名匹配，以进行重载解析以及潜在的类型转换。

传统的头文件（`.h`）和源代码文件（`.cpp`）指定依赖关系的方式会触发重新编译。通过将类内部实现细节从类头文件中移除（并将这些细节放在源文件中），我们可以消除许多依赖关系。我们可以更改其他头文件在其他头文件和源代码文件中的包含方式，简化依赖关系，从而减轻重新编译的负担。

pImpl 模式将迫使对类定义进行以下调整：

+   私有（非虚拟）成员将被替换为指向嵌套类类型的指针，该类型包括以前的私有数据成员和方法。嵌套类的前向声明也是必要的。

+   指向实现的指针（`pImpl`）将是一个关联，类实现的方法调用将被委托给它。

+   修订后的类定义将存在于一个采用这种习惯用法的类的头文件中。以前包含在这个头文件中的任何头文件现在将被移动到该类的源代码文件中。

+   现在，其他类包括 pImpl 类的头文件将不会面临重新编译，如果类的实现在其私有访问区域内被修改。

+   为了有效地管理代表实现的关联对象的动态内存资源，我们将使用一个唯一指针（智能指针）。

修订后的类定义中的编译自由度利用了指针只需要类类型的前向声明才能编译的事实。

让我们继续检查 pImpl 模式的基本实现，然后是精炼的实现。

# 实现 pImpl 模式

为了实现 pImpl 模式，我们需要重新审视典型的头文件和源文件组成。然后，我们将用指向实现的指针替换典型类定义中的私有成员，利用关联。实现将被封装在我们目标类的嵌套类中。我们的 pImpl 指针将把所有请求委托给我们的关联对象，该对象提供内部类的详细信息或实现。

内部（嵌套）类将被称为**实现类**。原始的、现在是外部的类将被称为**目标**或**接口类**。

我们将首先回顾典型（非 pImpl 模式）文件组成，其中包含类定义和成员函数定义。

## 组织文件和类内容以应用模式基础知识

首先让我们回顾一下典型的 C++类的组织策略，关于类定义和成员函数定义的文件放置。接下来，我们将考虑使用 pImpl 模式的类的修订组织策略。

### 回顾典型的文件和类布局

让我们看一下典型的类定义，以及我们以前如何组织类与源文件和头文件相关的内容，比如我们在[*第五章*]（B15702_05_Final_NM_ePub.xhtml#_idTextAnchor199）中的讨论，*详细探讨类*，以及[*第十五章*]（B15702_15_Final_NM_ePub.xhtml#_idTextAnchor572），*测试 OO 程序和组件*。

回想一下，我们将每个类组织成一个头（`.h`）文件，其中包含类定义和内联函数定义，以及一个相应的源代码（`.cpp`）文件，其中包含非内联成员函数定义。让我们回顾一个熟悉的样本类定义，`Person`：

```cpp
#ifndef _PERSON_H  // preprocessor directives to avoid 
#define _PERSON_H  // multiple inclusion of header
class Person
{
private:
    char *firstName, *lastName, *title;
    char middleInitial;
protected:
    void ModifyTitle(const char *);
public:
    Person();   // default constructor
    Person(const char *, const char *, char, const char *);
    Person(const Person &);  // copy constructor
    virtual ~Person();  // virtual destructor
    const char *GetFirstName() const { return firstName; }
    const char *GetLastName() const { return lastName; }
    const char *GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    virtual void Print() const;
    virtual void IsA();
    virtual void Greeting(const char *);
    Person &operator=(const Person &);  // overloaded op =
};
#endif
```

在上述的头文件（`Person.h`）中，我们已经包含了我们的`Person`类的类定义，以及类的内联函数定义。任何不在类定义中出现的较大的内联函数定义（在原型中使用关键字`inline`表示）也会出现在这个文件中，在类定义本身之后。请注意使用预处理指令来确保每个编译单元只包含一次类定义。

接下来让我们回顾相应的源代码文件`Person.cpp`的内容：

```cpp
#include <iostream>  // also include other relevant libraries
#include "Person.h"  // include the header file
using namespace std;
// Include all the non-inline Person member functions
// The default constructor is one example of many in the file
Person::Person()
{
    firstName = lastName = title = 0;  // NULL pointer
    middleInitial = '\0';   // null character
}
```

在先前定义的源代码文件中，我们为类`Person`定义了所有非内联成员函数。虽然并非所有方法都显示出来，但所有方法都可以在我们的 GitHub 代码中找到。此外，如果类定义包含任何静态数据成员，应该在源代码文件中包含外部变量的定义，指定这个成员的内存。

现在让我们考虑如何通过应用 pImpl 模式，从`Person`类定义及其对应的头文件中删除实现细节。

### 应用修订后的类和文件布局的 pImpl 模式

为了使用 pImpl 模式，我们将重新组织我们的类定义及其相应的实现。我们将在现有类定义中添加一个嵌套类，以表示原始类的私有成员和其实现的核心。我们的外部类将包括一个指向内部类类型的指针，作为与我们实现的关联。我们的外部类将把所有实现请求委托给内部关联对象。我们将重新构造头文件和源代码文件中类和源代码的放置方式。

让我们仔细看一下我们修订后的类的实现，以了解实现 pImpl 模式所需的每个新细节。这个例子由一个源文件`PersonImpl.cpp`和一个头文件`Person.h`组成，可以在与我们的 GitHub 存储库中测试该模式的简单驱动程序相同的目录中找到。为了创建一个完整的可执行文件，您需要编译和链接`PersonImp.cpp`和`Chp20-Ex1.cpp`（驱动程序），这两个文件都在同一个目录中。以下是驱动程序的 GitHub 存储库 URL：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/Chp20-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/Chp20-Ex1.cpp)

```cpp
#ifndef _PERSON_H    // Person.h header file definition
#define _PERSON_H
class Person
{
private:
    class PersonImpl;  // forward declaration of nested class
    PersonImpl *pImpl; // pointer to implementation of class
protected:
    void ModifyTitle(const char *);
public:
    Person();   // default constructor
    Person(const char *, const char *, char, const char *);
    Person(const Person &);  // copy constructor
    virtual ~Person();  // virtual destructor
    const char *GetFirstName() const; // no longer inline
    const char *GetLastName() const; 
    const char *GetTitle() const; 
    char GetMiddleInitial() const; 
    virtual void Print() const;
    virtual void IsA();
    virtual void Greeting(const char *);
    Person &operator=(const Person &);  // overloaded =
};
#endif
```

在我们前面提到的`Person`的修订类定义中，请注意我们已经删除了私有访问区域中的数据成员。任何非虚拟的私有方法，如果存在的话，也将被删除。相反，我们使用`class PersonImpl;`对我们的嵌套类进行了前向声明。我们还声明了一个指向实现的指针，使用`PersonImpl *pImpl;`，它代表了封装实现的嵌套类成员的关联。在我们的初始实现中，我们将使用一个本地（原始的）C++指针来指定与嵌套类的关联。随后我们将修改我们的实现以利用*unique pointer*。

请注意，我们的`Person`的公共接口与以前大致相同。所有现有的公共和受保护的方法都存在于预期的接口中。然而，我们注意到依赖于数据成员实现的内联函数已被替换为非内联成员函数原型。

让我们继续看一下我们嵌套类`PersonImpl`的类定义，以及`PersonImpl`和`Person`的成员函数在一个共同的源代码文件`PersonImpl.cpp`中的放置。我们将从嵌套`PersonImpl`类定义开始：

```cpp
// PersonImpl.cpp source code file includes the nested class
// Nested class definition supports implementation
class Person::PersonImpl
{
private:
    char *firstName, *lastName, *title;
    char middleInitial;
public:
    PersonImpl();   // default constructor
    PersonImpl(const char *, const char *, char, 
               const char *);
    PersonImpl(const PersonImpl &);  
    virtual ~PersonImpl();  
    const char *GetFirstName() const { return firstName; }
    const char *GetLastName() const { return lastName; }
    const char *GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    void ModifyTitle(const char *);
    virtual void Print() const;
    virtual void IsA() { cout << "Person" << endl; }
    virtual void Greeting(const char *msg) 
        { cout << msg << endl; }
    PersonImpl &operator=(const PersonImpl &); 
};
```

在前面提到的`PersonImpl`的嵌套类定义中，请注意这个类看起来与`Person`的原始类定义非常相似。我们有私有数据成员和一整套成员函数原型，甚至为了简洁起见编写了一些内联函数（实际上不会被内联，因为它们是虚拟的）。`PersonImpl`代表了`Person`的实现，因此这个类能够访问所有数据并完全实现每个方法是至关重要的。请注意，在`class Person::PersonImpl`的定义中，作用域解析运算符(`::`)用于指定`PersonImpl`是`Person`的嵌套类。

让我们继续看一下`PersonImpl`的成员函数定义，它们将出现在与类定义相同的源文件`PersonImpl.cpp`中。尽管一些方法已经被缩写，但它们的完整在线代码在我们的 GitHub 存储库中是可用的：

```cpp
// File: PersonImpl.cpp -- See online code for full methods. 
// Nested class member functions. 
// Notice that the class name is Outer::Inner class
Person::PersonImpl::PersonImpl()
{
    firstName = lastName = title = 0;  // NULL pointer
    middleInitial = '\0';
}
Person::PersonImpl::PersonImpl(const char *fn, const char *ln,
                               char mi, const char *t)
{
    firstName = new char [strlen(fn) + 1];
    strcpy(firstName, fn);
    // Continue memory allocation, init. for data members
}
Person::PersonImpl::PersonImpl(const Person::PersonImpl &pers)
{
    firstName = new char [strlen(pers.firstName) + 1];
    strcpy(firstName, pers.firstName);
    // Continue memory allocation and deep copy for all
}   // pointer data members and copy for non-ptr data members
Person::PersonImpl::~PersonImpl()
{   // Delete all dynamically allocated data members
}
void Person::PersonImpl::ModifyTitle(const char *newTitle)
{   // Delete old title, reallocate space for and copy new one
}
void Person::PersonImpl::Print() const
{   // Print each data member as usual
}
Person::PersonImpl &Person::PersonImpl::operator=
                             (const PersonImpl &p)
{  // check for self-assignment, then delete destination
   // object data members. Then reallocate and copy from 
   // source object. 
   return *this;  // allow for cascaded assignments
}
```

在上述代码中，我们看到了使用嵌套类`PersonImpl`实现整体`Person`类的代码。我们看到了`PersonImpl`的成员函数定义，并注意到这些方法的实现方式与我们之前在原始`Person`类中实现方法的方式完全相同，而没有使用 pImpl 模式。同样，我们注意到使用了作用域解析运算符(`::`)来指定每个成员函数定义的类名，比如`void Person::PersonImpl::Print() const`。在这里，`Person::PersonImpl`表示`Person`类中的`PersonImpl`嵌套类。

接下来，让我们花一点时间来审查`Person`的成员函数定义，我们的类使用了 pImpl 模式。这些方法还将为`PersonImpl.cpp`源代码文件做出贡献，并且可以在我们的 GitHub 存储库中找到：

```cpp
// Person member functions – also in PersonImpl.cpp
Person::Person(): pImpl(new PersonImpl())
{  // This is the complete member function definition
}
Person::Person(const char *fn, const char *ln, char mi,
               const char *t): 
               pImpl(new PersonImpl(fn, ln, mi, t))
{  // This is the complete member function definition
}  
Person::Person(const Person &pers): 
           pImpl(new PersonImpl(*(pers.pImpl)))
{  // This is the complete member function definition
}  // No Person data members to copy from pers except deep
   // copy of *(pers.pImpl) to data member pImpl
Person::~Person()
{
    delete pImpl;   // delete associated implementation
}
void Person::ModifyTitle(const char *newTitle)
{   // delegate request to the implementation 
    pImpl->ModifyTitle(newTitle);  
}
const char *Person::GetFirstName() const
{   // no longer inline in Person;further hides implementation
    return pImpl->GetFirstName();
}
// Note: methods GetLastName(), GetTitle(), GetMiddleInitial()
// are implemented similar to GetFirstName(). See online code.
void Person::Print() const
{
    pImpl->Print();   // delegate to implementation
}                     // (same named member function)
// Note: methods IsA() and Greeting() are implemented 
// similarly to Print() – using delegation. See online code.
Person &Person::operator=(const Person &p)
{  // delegate op= to implementation portion
   pImpl->operator=(*(p.pImpl));   // call op= on impl. piece
   return *this;  // allow for cascaded assignments
}
```

在上述`Person`的成员函数定义中，我们注意到所有方法都通过关联`pImpl`委托所需的工作给嵌套类。在我们的构造函数中，我们分配了关联的`pImpl`对象并适当地初始化它（使用每个构造函数的成员初始化列表）。我们的析构函数负责使用`delete pImpl;`删除关联对象。

我们的`Person`复制构造函数将会将成员`pImpl`设置为新分配的内存，同时调用嵌套对象的`PersonImpl`复制构造函数进行对象的创建和初始化，将`*(pers.pImpl)`传递给嵌套对象的复制构造函数。也就是说，`pers.pImpl`是一个指针，所以我们使用`*`对指针进行解引用，以获得可引用的对象，用于`PersonImpl`的复制构造函数。

我们在`Person`的重载赋值运算符中使用了类似的策略。也就是说，除了`pImpl`之外，没有其他数据成员来执行深度赋值，因此我们只是在关联对象`pImpl`上调用`PersonImpl`的赋值运算符，再次将`*(p.pImpl)`作为右值传入。

最后，让我们考虑一个示例驱动程序，以演示我们模式的运行情况。有趣的是，我们的驱动程序将使用我们最初指定的非模式类（源文件和头文件）或我们修改后的 pImpl 模式特定源文件和头文件！

### 将模式组件组合在一起

最后，让我们来看看我们驱动程序源文件`Chp20-Ex1.cpp`中的`main()`函数：

```cpp
#include <iostream>
#include "Person.h"
using namespace std;
const int MAX = 3;
int main()
{
    Person *people[MAX];
    people[0] = new Person("Giselle", "LeBrun", 'R', "Ms.");
    people[1] = new Person("Zack", "Moon", 'R', "Dr.");
    people[2] = new Person("Gabby", "Doone", 'A', "Dr.");
    for (int i = 0; i < MAX; i++)
       people[i]->Print();
    for (int i = 0; i < MAX; i++)
       delete people[i];
    return 0;
}
```

审查我们上述的`main()`函数，我们只是动态分配了几个`Person`实例，调用了实例的`Person`方法，然后删除了每个实例。我们已经包含了`Person.h`头文件，如预期的那样，以便能够使用这个类。从客户端的角度来看，一切看起来*像往常一样*，并且与模式无关。

请注意，我们分别编译`PersonImp.cpp`和`Chp20-Ex1.cpp`，将对象文件链接在一起成为可执行文件。然而，由于 pImpl 模式，如果我们改变了`Person`的实现，这种改变将被封装在`PersonImp`嵌套类的实现中。只有`PersonImp.cpp`需要重新编译。客户端不需要在驱动程序`Chp20-Ex1.cpp`中重新编译，因为更改不会发生在驱动程序依赖的`Person.h`头文件中。

让我们来看看这个程序的输出。

```cpp
Ms. Giselle R. LeBrun
Dr. Zack R. Moon
Dr. Gabby A. Doone
```

在上述输出中，我们看到了我们简单驱动程序的预期结果。

让我们继续考虑如何通过使用独特指针来改进我们的 pImpl 模式的实现。

## 使用独特指针改进模式

我们使用与本机 C++指针关联的关联来实现的初始实现减轻了许多编译器依赖。这是因为编译器只需要看到 pImpl 指针类型的前向类声明，才能成功编译。到目前为止，我们已经实现了使用 pImpl 模式的核心目标-减少重新编译。

然而，总是有人批评使用本机或*原始*指针。我们需要自己管理内存，包括记住在外部类析构函数中删除分配的嵌套类类型。内存泄漏、内存滥用和内存错误是使用原始指针自己管理内存资源的潜在缺点。因此，习惯上使用**智能指针**来实现 pImpl 模式。

我们将继续实现 pImpl 的任务，通过检查通常与 pImpl 模式一起使用的关键组件——智能指针，更具体地说是`unique_ptr`。

让我们从理解智能指针的基础知识开始。

### 理解智能指针

为了习惯性地实现 pImpl 模式，我们必须首先了解智能指针。**智能指针**是一个小的包装类，封装了一个原始指针，确保它包含的指针在包装对象超出范围时自动删除。实现智能指针的类可以使用模板来为任何数据类型创建智能指针。

这是一个非常简单的智能指针示例。这个示例可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/Chp20-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/Chp20-Ex2.cpp)

```cpp
#include <iostream>
#include "Person.h"
using namespace std;
template <class Type>
class SmartPointer
{
private:
    Type *pointer;
public:
    SmartPointer(Type *ptr = NULL) { pointer = ptr; }
    virtual ~SmartPointer();  // allow specialized SmrtPtrs
    Type *operator->() { return pointer; }
    Type &operator*() { return *pointer; }
};
SmartPointer::~SmartPointer()
{
    delete pointer;
    cout << "SmartPtr Destructor" << endl;
}
int main()
{
    SmartPointer<int> p1(new int());
    SmartPointer<Person> pers1(new Person("Renee",
                               "Alexander", 'K', "Dr."));
    *p1 = 100;
    cout << *p1 << endl;
    (*pers1).Print();   // or use: pers1->Print();
    return 0;
}
```

在先前定义的简单`SmartPointer`类中，我们只是封装了一个原始指针。关键好处是，当包装对象从堆栈中弹出（对于局部实例）或在程序终止之前（对于静态和外部实例）时，`SmartPointer`析构函数将确保原始指针被销毁。当然，这个类很基础，我们必须确定复制构造函数和赋值运算符的期望行为。也就是说，允许浅复制/赋值，要求深复制/赋值，或者禁止所有复制/赋值。尽管如此，我们现在可以想象智能指针的概念。

以下是我们智能指针示例的输出：

```cpp
100
Dr. Renee K. Alexander
SmartPtr Destructor
SmartPtr Destructor
```

前面的输出显示，`SmartPointer` 中包含的每个对象的内存都是由我们管理的。我们可以很容易地通过`"SmartPtr Destructor"`输出字符串看到，当`main()`中的局部对象超出范围并从堆栈中弹出时，每个对象的析构函数会代表我们被调用。

### 理解唯一指针

标准 C++库中的`unique_ptr`是一种智能指针，它封装了对给定堆内存资源的独占所有权和访问权限。`unique_ptr`不能被复制；`unique_pointer`的所有者将独占该指针的使用权。唯一指针的所有者可以选择将这些指针移动到其他资源，但后果是原始资源将不再包含`unique_pointer`。我们必须`#include <memory>`来包含`unique_ptr`的定义。

修改我们的智能指针程序，改用`unique_pointer`，现在我们有：

```cpp
#include <iostream>
#include <memory>
#include "Person.h"
using namespace std;
int main()
{
    unique_ptr<int> p1(new int());
    unique_ptr<Person> pers1(new Person("Renee", "Alexander",
                                        'K', "Dr."));
    *p1 = 100;
    cout << *p1 << endl;
    (*pers1).Print();   // or use: pers1->Print();
    return 0;
}
```

我们的输出将类似于`SmartPointer`示例；不同之处在于不会显示`"SmartPtr Destructor"`调用消息（因为我们使用的是`unique_ptr`）。请注意，因为我们包含了`using namespace std;`，所以在唯一指针声明中我们不需要用`std::`来限定`unique_ptr`。

有了这个知识，让我们将唯一指针添加到我们的 pImpl 模式中。

### 将唯一指针添加到模式中

为了使用`unique_ptr`实现 pImpl 模式，我们将对先前的实现进行最小的更改，从我们的`Person.h`头文件开始。我们的 pImpl 模式利用`unique_ptr`的完整程序示例可以在我们的 GitHub 存储库中找到，并且还将包括一个修订后的`PersonImpl.cpp`文件。这是驱动程序`Chp20-Ex3.cpp`的 URL；请注意我们的 GitHub 存储库中的子目录，用于这个完整示例：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/unique/Chp20-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter20/unique/Chp20-Ex3.cpp)

```cpp
#ifndef _PERSON_H    // Person.h header file definition
#define _PERSON_H
#include <memory>
class Person
{
private:
    class PersonImpl;  // forward declaration to nested class
    std::unique_ptr<PersonImpl> pImpl; // unique ptr to impl
protected:
    void ModifyTitle(const char *);
public:
    Person();   // default constructor
    Person(const char *, const char *, char, const char *);
    Person(const Person &);  // copy constructor
    virtual ~Person();  // virtual destructor
    const char *GetFirstName() const; // no longer inline
    const char *GetLastName() const; 
    const char *GetTitle() const; 
    char GetMiddleInitial() const; 
    virtual void Print() const;
    virtual void IsA();
    virtual void Greeting(const char *);
    Person &operator=(const Person &);  // overloaded =
};
#endif
```

请注意，在前面修改过的`Person`类定义中，有`std::unique_ptr<PersonImpl> pImpl;`的独占指针声明。在这里，我们使用`std::`限定符，因为标准命名空间没有在我们的头文件中明确包含。我们还`#include <memory>`来获取`unique_ptr`的定义。类的其余部分与我们最初使用原始指针实现的 pImpl 实现是相同的。

接下来，让我们了解一下我们的源代码需要从最初的 pImpl 实现中进行多少修改。现在让我们来看一下我们源文件`PersonImpl.cpp`中需要修改的成员函数：

```cpp
// Source file PersonImpl.cpp
// Person destructor no longer needs to delete pImpl member
Person::~Person()
{  // unique_pointer pImpl will delete its own resources
}
```

看一下前面提到需要修改的成员函数，我们发现只有`Person`的析构函数！因为我们使用了一个独占指针来实现对嵌套类实现的关联，我们不再需要自己管理这个资源的内存。这非常好！通过这些小的改变，我们的 pImpl 模式现在使用`unique_ptr`来指定类的实现。

接下来，让我们来检查一些与使用 pImpl 模式相关的性能问题。

# 理解 pImpl 模式的权衡取舍。

将 pImpl 模式纳入生产代码中既有好处又有缺点。让我们分别审查一下，以便更好地理解可能需要部署这种模式的情况。

可忽略的性能问题包括大部分的缺点。也就是说，几乎每个对目标（接口）类的请求都需要委托给其嵌套实现类。唯一可以由外部类处理的请求是那些不涉及任何数据成员的请求；这种情况将非常罕见！另一个缺点包括实例需要更高的内存需求来容纳作为模式实现的一部分添加的指针。这些问题在嵌入式软件系统和需要最佳性能的系统中将是至关重要的，但在其他情况下相对较小。

对于使用 pImpl 模式的类来说，维护将会更加困难，这是一个不幸的缺点。每个目标类现在都与一个额外的（实现）类配对，包括一组转发方法来将请求委托给实现。

也可能会出现一些实现困难。例如，如果任何私有成员（现在在嵌套实现类中）需要访问外部接口类的受保护或公共方法中的任何一个，我们将需要在嵌套类中包含一个反向链接，以便访问该成员。为什么？内部类中的`this`指针将是嵌套对象类型的。然而，外部对象中的受保护和公共方法将期望一个`this`指针指向外部对象 - 即使这些公共方法将重新委托请求调用私有的嵌套类方法来帮助。还需要这个反向链接来从内部类（实现）的范围内调用接口的公共虚函数。然而，请记住，我们通过每个对象添加的另一个指针和委托来影响性能，这将影响性能。

利用 pImpl 模式的优势有很多，提供了重要的考虑因素。其中最重要的是，在开发和维护代码期间重新编译的时间显著减少。此外，类的编译二进制接口变得独立于类的底层实现。更改类的实现只需要重新编译和链接嵌套实现类。外部类的用户不受影响。作为一个额外的好处，pImpl 模式提供了一种隐藏类的底层私有细节的方法，这在分发类库或其他专有代码时可能会有用。

在我们的 pImpl 实现中包含`unique_pointer`的一个优势是，我们保证了关联实现类的正确销毁。我们还有可能避免程序员引入的指针和内存错误！

使用 pImpl 模式是一种权衡。对每个类和所涉及的应用进行仔细分析将有助于确定 pImpl 模式是否适合您的设计。

我们现在已经看到了最初使用原始指针的 pImpl 模式的实现，然后应用了`unique_pointer`。让我们现在简要回顾一下我们在结束本书的最后一章之前所学到的与模式相关的内容。

# 总结

在本章中，我们通过进一步提高我们的编程技能，探索了另一个核心设计模式，进一步推进了成为更不可或缺的 C++程序员的目标。我们通过使用本地 C++指针和关联来探索了 pImpl 模式的初始实现，然后通过使用 unique 指针来改进我们的实现。通过检查实现，我们很容易理解 pImpl 模式如何减少编译时的依赖，并且可以使我们的代码更依赖于实现。

利用核心设计模式，比如 pImpl 模式，将帮助您更轻松地为其他熟悉常见设计模式的程序员理解的可重用、可维护的代码做出贡献。您的软件解决方案将基于创造性和经过良好测试的设计解决方案。

我们现在一起完成了我们的最后一个设计模式，结束了对 C++面向对象编程的长期探索。您现在拥有了许多技能，包括对面向对象编程的深入理解、扩展的语言特性和核心设计模式，这些都使您成为了一名更有价值的程序员。

尽管 C++是一种复杂的语言，具有额外的特性、补充技术和额外的设计模式需要探索，但您已经具备了坚实的基础和专业水平，可以轻松地掌握和获取任何额外的语言特性、库和模式。您已经走了很长的路；这是一次充满冒险的旅程！我享受我们的探索过程的每一分钟，希望您也一样。

我们从审查基本语言语法和理解 C++基础知识开始，这些知识对我们即将开始的面向对象编程之旅至关重要。然后，我们一起将 C++作为面向对象的语言，不仅学习了基本的面向对象概念，还学会了如何使用 C++语言特性、编码技巧或两者都使用来实现它们。然后，我们通过添加异常处理、友元、运算符重载、模板、STL 基础知识以及测试面向对象类和组件来扩展您的技能。然后，我们通过应用感兴趣的每个模式深入代码，进入了复杂的编程技术。

这些所获得的技能段代表了 C++知识和掌握的新层次。每一个都将帮助您更轻松地创建可维护和健壮的代码。您作为一名精通的、熟练的 C++面向对象程序员的未来正在等待。现在，让我们开始编程吧！

# 问题

1.  修改本章中使用 unique 指针的 pImpl 模式示例，另外在嵌套类的实现中引入 unique 指针。

1.  将您以前章节解决方案中的`Student`类简单地继承自本章中采用 pImpl 模式的`Person`类。您遇到了什么困难吗？现在，修改您的`Student`类，另外利用独特指针实现 pImpl 模式。现在，您遇到了什么困难吗？

1.  您能想象其他什么例子可能合理地将 pImpl 模式纳入相对独立的实现中？
