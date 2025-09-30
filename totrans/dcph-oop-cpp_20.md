# 20

# 使用 pImpl 模式移除实现细节

本章将结束我们扩展你的 C++ 编程知识库的旅程，目标是进一步赋予你解决常见编码问题的能力，利用常见的设计模式。在你的编码中融入设计模式不仅可以提供更精细的解决方案，还有助于简化代码维护并提供潜在的代码重用。

我们接下来将学习如何在 C++ 中有效地实现下一个设计模式——**pImpl 模式**。

在本章中，我们将涵盖以下主要主题：

+   理解 pImpl 模式及其如何减少编译时依赖

+   理解如何在 C++ 中使用关联和唯一指针实现 pImpl 模式

+   识别与 pImpl 相关的性能问题以及必要的权衡

到本章结束时，你将理解 pImpl 模式以及如何将其用于将实现细节从类接口中分离出来，以减少编译器依赖。将额外的设计模式添加到你的技能集中将帮助你成为一个更有价值的程序员。

让我们通过检查另一个常见的设计模式——pImpl 模式，来提高我们的编程技能。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter20`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter20)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，文件名对应章节编号，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter20` 子目录中找到，文件名为 `Chp20-Ex1.cpp`。一些程序位于示例中指示的可应用子目录中。

本章的 CiA 视频可以在以下网址查看：[`bit.ly/3CfQxhR`](https://bit.ly/3CfQxhR)。

# 理解 pImpl 模式

**pImpl 模式**（**p**ointer to **Impl**ementation 习语）是一种结构化设计模式，它将类的实现与其公共接口分离。这个模式最初被称为 **Bridge 模式**，由 **Gang of Four**（**GoF**）提出，也被称为 **Cheshire Cat**、**compiler-firewall 习语**、**d-pointer**、**opaque pointer** 或 **Handle 模式**。

此模式的主要目的是最小化编译时依赖。减少编译时依赖的结果是，类定义中的更改（最明显的是私有访问区域）不会在开发或部署的应用程序中引发一系列及时的重新编译。相反，必要的重新编译代码可以隔离到类的*实现*本身。依赖于类定义的应用程序的其他部分将不再受重新编译的影响。

类定义内部的私有成员可能会影响类的重新编译。这是因为更改数据成员可能会改变该类型实例的大小。此外，私有成员函数必须与函数调用签名匹配，以便进行重载解析以及潜在的类型转换。

传统头文件（`.h` 或 `.hpp`）和源代码文件（`.cpp`）中指定依赖关系的方式会触发重新编译。通过将类内部实现细节从类头文件中移除（并将这些细节放在源文件中），我们可以消除许多依赖。我们可以更改其他头文件和源代码文件中包含的头文件，简化依赖关系，从而减轻重新编译的负担。

pImpl 模式将强制对类定义进行以下调整：

+   私有（非虚拟）成员将被替换为指向包含以前私有数据成员和方法的嵌套类类型的指针。还需要对嵌套类进行前向声明。

+   实现的指针（`pImpl`）将是一个关联，类实现的函数调用将被委派到这个关联上。

+   修订后的类定义将存在于采用此习语的类的头文件中。任何以前由该头文件依赖的已包含的头文件现在将移动到源代码文件中，而不是包含在头文件中。

+   如果修改了类在其私有访问区域内的实现，现在包括 pImpl 类的头文件在内的其他类将不会面临重新编译。

+   为了有效地管理表示实现的关联对象的动态内存资源，我们将使用唯一指针（智能指针）。

修订后的类定义中的编译自由利用了这样一个事实：指针只需要对指针指向的类类型进行前向声明即可编译。

让我们继续前进，首先考察一个基本的，然后是一个改进的 pImpl 模式的实现。

# 实现 pImpl 模式

为了实现 pImpl 模式，我们需要重新审视典型的头文件和源文件组成。然后，我们将典型类定义中的私有成员替换为指向实现的指针，利用关联的优势。实现将被封装在我们目标类的嵌套类中。我们的 pImpl 指针将委托所有请求到提供内部类细节或实现的关联对象。

内部（嵌套）类将被称为**实现类**。原始的，现在外部的，类将被称为**目标**或**接口类**。

我们将首先回顾包含类定义和成员函数定义的典型（非 pImpl 模式）文件组成。

## 组织文件和类内容以应用模式基础

让我们首先回顾典型 C++类在文件放置方面的组织策略，包括类定义和成员函数定义。接下来，我们将考虑使用 pImpl 模式的类的修改后的组织策略。

### 回顾典型文件和类布局

让我们看看一个典型的类定义以及我们之前是如何根据源文件和头文件组织类的，例如在*第五章*“详细探索类”和*第十五章*“测试类和组件”中的讨论。

回想一下，我们将每个类组织到一个包含类定义和内联函数定义的头文件（`.h`或`.hpp`）中，以及一个包含非内联成员函数定义的相应源代码文件（`.cpp`）。让我们回顾一个熟悉的样本类定义，`Person`：

```cpp
#ifndef _PERSON_H  // preprocessor directives to avoid 
#define _PERSON_H  // multiple inclusion of header
using std::string;
class Person
{
private:
    string firstName, lastName, title;
    char middleInitial = '\0';   // in-class initialization
protected:
    void ModifyTitle(const string &);
public:
    Person() = default;   // default constructor
    Person(const string &, const string &, char, 
           const string &);  // alternate constructor
    // prototype not needed for default copy constructor
    // Person(const Person &) = default;  // copy ctor
    virtual ~Person() = default;  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; }
    const string &GetLastName() const { return lastName; }
    const string &GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    virtual void Print() const;
    virtual void IsA() const;
    virtual void Greeting(const string &) const;
    Person &operator=(const Person &);  // overloaded op =
};
#endif
```

在上述头文件（`Person.h`）中，我们包含了`Person`类的类定义以及类的内联函数定义。任何未出现在类定义中（在原型中用关键字`inline`指示）的较大内联函数定义也将出现在此文件中，在类定义之后。注意预处理指令的使用，以确保每个编译单元只包含一次类定义。

让我们接下来回顾相应的源代码文件的内容，`Person.cpp`：

```cpp
#include <iostream>  // also incl. other relevant libraries
#include "Person.h"  // include the header file
using std::cout;     // preferred to: using namespace std;
using std::endl; 
using std::string;
// Include all the non-inline Person member functions
// The alt. constructor is one example of many in the file
Person::Person(const string &fn, const string &ln, char mi,
             const string &t): firstName(fn), lastName(ln),
                               middleInitial(mi), title(t)
{
   // dynamically alloc. memory for any ptr data mbrs here
}
```

在之前定义的源代码文件中，我们为类`Person`定义了所有非内联成员函数。尽管不是所有方法都显示出来，但所有方法都可以在我们的 GitHub 代码中找到。此外，如果类定义包含任何静态数据成员，这些成员的内存指定的外部变量定义也应包含在源代码文件中。

现在让我们考虑如何通过应用 pImpl 模式，从`Person`类定义及其相应的头文件中移除实现细节。

### 应用 pImpl 模式并修改类和文件布局

要使用 pImpl 模式，我们将重新组织我们的类定义及其相应的实现。我们将在现有的类定义内添加一个嵌套类，以表示原始类的私有成员和其实现的核心。我们的外部类将包含一个指向内部类类型的指针，作为对我们实现的关联。我们的外部类将委派所有实现请求到关联的内部对象。我们将重新结构化头文件和源代码文件中类的放置。

让我们更仔细地看看我们对类的修订版实现，以了解实现 pImpl 模式所需的所有新细节。这个例子由一个源文件 `PersonImpl.cpp` 和一个头文件 `Person.h` 组成，可以在我们的 GitHub 仓库中的同一目录下找到，作为测试该模式的简单驱动程序。要制作一个完整的可执行文件，您需要编译并链接同一目录下的 `PersonImp.cpp` 和 `Chp20-Ex1.cpp`（驱动程序）。以下是驱动程序的 GitHub 仓库 URL：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/Chp20-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/Chp20-Ex1.cpp)

```cpp
#ifndef _PERSON_H    // Person.h header file definition
#define _PERSON_H
class Person
{
private:
    class PersonImpl;  // forward declaration nested class
    PersonImpl *pImpl = nullptr; // ptr to implementation 
                                 // of class
protected:
    void ModifyTitle(const string &);
public:
    Person();   // default constructor
    Person(const string &, const string &, char, 
           const string &);
    Person(const Person &);  // copy const. will be defined
    virtual ~Person();  // virtual destructor
    const string &GetFirstName() const; // no longer inline
    const string &GetLastName() const; 
    const string &GetTitle() const; 
    char GetMiddleInitial() const; 
    virtual void Print() const;
    virtual void IsA() const;
    virtual void Greeting(const string &) const;
    Person &operator=(const Person &);  // overloaded =
};
#endif
```

在我们之前提到的针对 `Person` 的修订版类定义中，请注意我们已经移除了私有访问区域中的数据成员。任何非虚私有方法，如果存在的话，也会被移除。相反，我们通过 `class PersonImpl;` 对嵌套类进行了前置声明。我们还声明了一个指向实现的指针 `PersonImpl *pImpl;`，它代表了对封装实现的嵌套类成员的关联。在我们的初始实现中，我们将使用原生（原始）C++指针来指定对嵌套类的关联。我们将随后修订我们的实现以利用 *唯一指针*。

注意，我们的 `Person` 的公共接口与之前几乎相同。我们现有的所有公共和受保护方法在接口上如预期存在。然而，我们注意到，内联函数（依赖于数据成员的实现）已被非内联成员函数原型所取代。

让我们继续前进，看看我们嵌套类 `PersonImpl` 的类定义，以及 `PersonImpl` 和 `Person` 的成员函数在公共源代码文件 `PersonImpl.cpp` 中的放置。我们将从嵌套的 `PersonImpl` 类定义开始：

```cpp
// PersonImpl.cpp source code file includes nested class
// Nested class definition supports implementation
class Person::PersonImpl
{
private:
    string firstName, lastName, title;
    char middleInitial = '\0';  // in-class initialization
public:
    PersonImpl() = default;   // default constructor
    PersonImpl(const string &, const string &, char, 
               const string &);
    // Default copy ctor does not need to be prototyped
    // PersonImpl(const PersonImpl &) = default;  
    virtual ~PersonImpl() = default;  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; }
    const string &GetLastName() const { return lastName; }
    const string &GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    void ModifyTitle(const string &);
    virtual void Print() const;
    virtual void IsA() const { cout << "Person" << endl; }
    virtual void Greeting(const string &msg) const
        { cout << msg << endl; }
    PersonImpl &operator=(const PersonImpl &); 
};
```

在之前提到的`PersonImpl`嵌套类定义中，请注意，这个类看起来与原始的`Person`类定义惊人地相似。我们有私有数据成员和一系列完整的成员函数原型，甚至为了简洁而编写的某些内联函数（实际上它们不会内联，因为它们是虚拟的）。`PersonImpl`代表`Person`的实现，因此这个类能够访问所有数据并完全实现每个方法至关重要。请注意，在`class Person::PersonImpl`的定义中使用的作用域解析运算符（`::`）用于指定`PersonImpl`是`Person`的嵌套类。

让我们继续，通过查看`PersonImpl`的成员函数定义来继续，这些定义将出现在与类定义相同的源文件`PersonImpl.cpp`中。尽管一些方法已经缩写，但它们的完整在线代码可以在我们的 GitHub 仓库中找到：

```cpp
// File: PersonImpl.cpp - See online code for full methods 
// Nested class member functions. 
// Notice that the class name is Outer::Inner class
// Notice that we are using the system-supplied definitions
// for default constructor, copy constructor and destructor
// alternate constructor
Person::PersonImpl::PersonImpl(const string &fn, 
             const string &ln, char mi, const string &t): 
             firstName(fn), lastName(ln), 
             middleInitial(mi), title(t)   
{
}
void Person::PersonImpl::ModifyTitle(const string &newTitle)
{   
    title = newTitle;
}
void Person::PersonImpl::Print() const
{   // Print each data member as usual
}
// Note: same as default op=, but it is good to review what 
// is involved in implementing op= for upcoming discussion
Person::PersonImpl &Person::PersonImpl::operator=
                             (const PersonImpl &p)
{  
    if (this != &p)  // check for self-assignment
    {
        firstName = p.firstName;
        lastName = p.lastName;
        middleInitial = p.middleInitial;
        title = p.title;
   }
   return *this;  // allow for cascaded assignments
}
```

在上述代码中，我们看到使用嵌套类`PersonImpl`实现的整体`Person`类的实现。我们看到`PersonImpl`的成员函数定义，并注意到这些方法的主体与我们之前在原始`Person`类中（没有使用 pImpl 模式）实现的方法完全相同。再次，我们注意到使用作用域解析运算符（`::`）来指定每个成员函数定义的类名，例如`void Person::PersonImpl::Print() const`。在这里，`Person::PersonImpl`表示`Person`类内部的嵌套类`PersonImpl`。

接下来，让我们花一点时间回顾`Person`类的成员函数定义，我们在这个类中使用了 pImpl 模式。这些方法还将贡献到`PersonImpl.cpp`源代码文件中，可以在我们的 GitHub 仓库中找到：

```cpp
// Person member functions – also in PersonImpl.cpp
Person::Person(): pImpl(new PersonImpl())
{ // As shown, this is the complete member fn. definition
}
Person::Person(const string &fn, const string &ln, char mi,
               const string &t): 
               pImpl(new PersonImpl(fn, ln, mi, t))
{ // As shown, this is the complete member fn. definition
}  
Person::Person(const Person &p): 
               pImpl(new PersonImpl(*(p.pImpl)))
{  // This is the complete copy constructor definition
}  // No Person data members to copy from 'p' except deep
   // copy of *(p.pImpl) to data member pImpl
Person::~Person()
{
    delete pImpl;   // delete associated implementation
}
void Person::ModifyTitle(const string &newTitle)
{   // delegate request to the implementation 
    pImpl->ModifyTitle(newTitle);  
}
const string &Person::GetFirstName() const
{   // no longer inline in Person; 
    // non-inline method further hides implementation
    return pImpl->GetFirstName();
}
// Note: methods GetLastName(), GetTitle(), and  
// GetMiddleInitial() are implemented similar to
// GetFirstName(). See online code
void Person::Print() const
{
    pImpl->Print();   // delegate to implementation
}                     // (same named member function)
// Note: methods IsA() and Greeting() are implemented 
// similarly to Print() – using delegation. See online code
Person &Person::operator=(const Person &p)
{  // delegate op= to implementation portion
   pImpl->operator=(*(p.pImpl)); // call op= on impl. piece
   return *this;  // allow for cascaded assignments
}
```

在之前提到的`Person`成员函数定义中，我们注意到所有方法都通过关联的`pImpl`将所需的工作委托给嵌套类。在我们的构造函数中，我们分配关联的`pImpl`对象并适当地初始化它（使用每个构造函数的成员初始化列表）。我们的析构函数负责使用`delete pImpl;`删除关联的对象。

我们的`Person`拷贝构造函数将成员`pImpl`设置为新的分配的内存，同时调用嵌套对象的创建和初始化的`PersonImpl`拷贝构造函数，将`*(p.pImpl)`传递给嵌套对象的拷贝构造函数。也就是说，`p.pImpl`是一个指针，所以我们使用`*`解引用指针以获得对`PersonImpl`拷贝构造函数的可引用对象。

我们在`Person`的重载赋值运算符中也使用了类似的策略。也就是说，除了`pImpl`之外没有其他数据成员来执行深度赋值，所以我们只是调用关联对象`pImpl`上的`PersonImpl`赋值运算符，再次传入`*(p.pImpl)`作为右侧值。

最后，让我们考虑一个示例驱动程序来展示我们的模式在实际应用中的效果。有趣的是，我们的驱动程序可以与最初指定的非模式类（源文件和头文件）一起工作，也可以与经过修订的 pImpl 模式特定源文件和头文件一起工作！

### 将模式组件组合在一起

让我们最后看看我们的驱动程序源文件`Chp20-Ex1.cpp`中的`main()`函数，看看我们的模式是如何编排的：

```cpp
#include <iostream>
#include "Person.h"
using std::cout;  // preferred to: using namespace std;
using std::endl;
constexpr int MAX = 3;
int main()
{
    Person *people[MAX] = { }; // initialized to nullptrs
    people[0] = new Person("Elle", "LeBrun", 'R',"Ms.");
    people[1] = new Person("Zack", "Moon", 'R', "Dr.");
    people[2] = new Person("Gabby", "Doone", 'A', "Dr.");
    for (auto *individual : people)
       individual->Print();
    for (auto *individual : people)
       delete individual;
    return 0;
}
```

回顾我们之前提到的`main()`函数，我们只是动态分配了几个`Person`实例，在实例上调用选定的`Person`方法（`Print()`），然后删除每个实例。我们如预期那样包含了`Person.h`头文件，以便能够使用这个类。从客户端的角度来看，一切看起来*都很正常*，看起来没有使用模式。

注意，我们分别编译`PersonImp.cpp`和`Chp20-Ex1.cpp`，将目标文件链接在一起生成可执行文件。然而，由于使用了 pImpl 模式，如果我们更改`Person`的实现，这种更改将被封装在其`PersonImp`嵌套类中的实现中。只有`PersonImp.cpp`需要重新编译。客户端不需要重新编译驱动程序`Chp20-Ex1.cpp`，因为更改不会发生在`Person.h`头文件中（驱动程序依赖于该头文件）。

让我们看看这个程序的输出：

```cpp
Ms. Elle R. LeBrun
Dr. Zack R. Moon
Dr. Gabby A. Doone
```

在上述输出中，我们看到了我们简单驱动程序的预期结果。

让我们继续前进，考虑如何使用唯一指针改进我们的 pImpl 模式实现。

## 使用唯一指针改进模式

我们最初使用与原生 C++指针关联的实现减少了编译器的依赖。这是因为编译器只需要看到 pImpl 指针类型的类前向声明就能成功编译。到目前为止，我们已经实现了使用 pImpl 模式的核心目标——减少重新编译。

然而，使用原生或*原始*指针总是存在批评。我们负责自己管理内存，包括记住在外部类析构函数中删除分配的嵌套类类型。使用原始指针自行管理内存资源可能会导致内存泄漏、内存误用和内存错误等潜在缺点。因此，通常使用**智能指针**来实现 pImpl 模式。

我们将继续我们的任务，通过检查与 pImpl 模式经常一起使用的关键组件——智能指针，或者更具体地说，是`unique_ptr`——来实现 pImpl。

让我们首先理解智能指针的基本知识。

### 理解智能指针

要实现 pImpl 模式，我们首先必须理解智能指针。**智能指针**是一个小的包装类，它封装了一个原始指针，确保当包装对象超出作用域时，它所包含的指针会自动删除。实现智能指针的类可以使用模板来实现，为任何数据类型创建智能指针。

这里是一个非常简单的智能指针示例。这个示例可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/Chp20-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/Chp20-Ex2.cpp)

```cpp
#include <iostream>
#include "Person.h"
using std::cout;   // preferred to: using namespace std;
using std::endl;
template <class Type>
class SmartPointer
{
private:
    Type *pointer = nullptr;  // in-class initialization
public:
    // Below ctor also handles default construction 
    SmartPointer(Type *ptr = nullptr): pointer(ptr) { }
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

在之前定义的简单`SmartPointer`类中，我们只是封装了一个原始指针。关键好处是`SmartPointer`的析构函数将确保在包装对象从栈中弹出（对于局部实例）或程序终止之前（对于静态和外部实例）时，原始指针被销毁。当然，这个类是基础的，我们必须确定所需的复制构造函数和赋值运算符的行为。也就是说，允许浅复制/赋值，要求深复制/赋值，或者禁止所有复制/赋值。尽管如此，我们现在可以可视化智能指针的概念。

这里是我们智能指针示例的输出：

```cpp
100
Dr. Renee K. Alexander
SmartPtr Destructor
SmartPtr Destructor
```

如上述输出所示，`SmartPointer`中包含的每个对象的内存都是由我们管理的。我们可以很容易地通过“`SmartPtr Destructor`”输出字符串看到，当`main()`中的局部对象超出作用域并被从栈中弹出时，会代表我们调用每个对象的析构函数。

### 理解唯一指针

标准 C++库中的`unique_ptr`是一种封装了给定堆内存资源独占所有权和访问权的智能指针类型。`unique_ptr`不能被复制；`unique_ptr`的所有者将独占使用该指针。`unique_ptr`的所有者可以选择将这些指针移动到其他资源，但后果是原始资源将不再包含`unique_ptr`。我们必须`#include <memory>`来包含`unique_ptr`的定义。

其他类型的智能指针

标准 C++库中除了`unique_ptr`之外，还有其他类型的智能指针可用，例如`weak_ptr`和`shared_ptr`。这些额外的智能指针类型将在*第二十一章* *《使 C++更安全》*中探讨。

将我们的智能指针程序修改为使用`unique_ptr`，我们现在有以下内容：

```cpp
#include <iostream>
#include <memory>
#include "Person.h"
using std::cout;   // preferred to: using namespace std;
using std::endl;
using std::unique_ptr;
int main()
{
    unique_ptr<int> p1(new int());
    unique_ptr<Person> pers1(new Person("Renee", 
                             "Alexander", 'K', "Dr."));
    *p1 = 100;
    cout << *p1 << endl;
    (*pers1).Print();   // or use: pers1->Print();
    return 0;
}
```

我们的输出将与 `SmartPointer` 示例类似；区别在于不会显示 `"SmartPtr Destructor"` 调用消息（因为我们使用的是 `unique_ptr`）。注意，因为我们包含了 `using std::unique_ptr;`，所以我们不需要在唯一指针声明中对 `unique_ptr` 进行 `std::` 限定。

带着这些知识，让我们将唯一指针添加到我们的 pImpl 模式中。

### 向模式中添加唯一指针

要使用 `unique_ptr` 实现 pImpl 模式，我们将对我们的先前实现进行最小程度的修改，从我们的 `Person.h` 头文件开始。我们使用 `unique_ptr` 的 pImpl 模式完整程序示例可以在我们的 GitHub 仓库中找到，并将包括对 `PersonImpl.cpp` 的修订文件。以下是驱动程序的 URL，`Chp20-Ex3.cpp`；注意我们 GitHub 仓库中此完整示例的子目录，`unique`：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/unique/Chp20-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter20/unique/Chp20-Ex3.cpp)

```cpp
#ifndef _PERSON_H    // Person.h header file definition
#define _PERSON_H
#include <memory>
class Person
{
private:
    class PersonImpl;  // forward declaration nested class
    std::unique_ptr<PersonImpl> pImpl; //unique ptr to impl
protected:
    void ModifyTitle(const string &);
public:
    Person();   // default constructor
    Person(const string &, const string &, char, 
           const string &);
    Person(const Person &);  // copy constructor
    virtual ~Person();  // virtual destructor
    const string &GetFirstName() const; // no longer inline
    const string &GetLastName() const; 
    const string &GetTitle() const; 
    char GetMiddleInitial() const; 
    virtual void Print() const;
    virtual void IsA() const;
    virtual void Greeting(const string &) const;
    Person &operator=(const Person &);  // overloaded =
};
#endif
```

注意，在修改后的 `Person` 类定义中，`std::unique_ptr<PersonImpl> pImpl;` 的唯一指针声明。在这里，我们使用 `std::` 限定符，因为标准命名空间尚未在我们的头文件中显式包含。我们还 `#include <memory>` 以获取 `unique_ptr` 的定义。类的其余部分与我们的初始 pImpl 实现相同，该实现使用原始指针实现的关联。

接下来，让我们了解我们的源代码需要从初始的 pImpl 实现中修改到何种程度。现在，让我们查看源文件 `PersonImpl.cpp` 中必要的修改后的成员函数：

```cpp
// Source file PersonImpl.cpp
// Person destructor no longer needs to delete pImpl member
// and hence can simply be the default destructor!
// Note: prototyped with virtual in header file.
Person::~Person() = default;
// unique_pointer pImpl will delete its own resources
```

查看需要修改的上述成员函数，我们发现只有 `Person` 析构函数！因为我们使用唯一指针来实现对嵌套类实现的关联，所以我们不再需要自己管理这个资源的内存。这真是太好了！通过这些小的改动，我们的 pImpl 模式现在具有一个 `unique_ptr` 来指定类的实现。

接下来，让我们检查一些与使用 pImpl 模式相关的性能问题。

# 理解 pImpl 模式的权衡

将 pImpl 模式集成到生产代码中既有优点也有缺点。让我们逐一回顾，以便我们更好地理解可能需要部署此模式的情况。

可忽略的性能问题涵盖了大多数缺点。也就是说，针对目标（接口）类的几乎所有请求都需要委派给其嵌套的实现类。唯一可以由外部类处理的请求将是不涉及任何数据成员的请求；这些情况将极其罕见！另一个缺点包括实例的内存需求略有增加，以适应模式实现中添加的指针。这些问题在嵌入式软件系统和需要峰值性能的系统中将至关重要，但在其他情况下相对较小。

对于采用 pImpl 模式的类，维护将稍微困难一些，这是一个不幸的缺点。每个目标类现在都配有一个额外的（实现）类，包括一组转发方法，用于将请求委派给实现类。

也可能出现一些实现困难。例如，如果任何私有成员（现在在嵌套实现类中）需要访问外部（接口）类的任何受保护或公共方法，我们需要从嵌套类到外部类包含一个回链以访问该成员。为什么？内部类中的`this`指针将是嵌套对象类型。然而，外部对象中的受保护和公共方法将期望一个指向外部对象的`this`指针——即使这些公共方法随后将请求重新委派以调用私有嵌套类方法以获得帮助。此回链还用于从内部类（实现）的作用域调用接口的公共虚拟函数。然而，请记住，我们通过为每个对象添加另一个指针以及委派调用相关对象中的每个方法来影响性能。

利用 pImpl 模式有几个优点，提供了重要的考虑因素。最重要的是，在代码的开发和维护期间，重新编译时间显著减少。此外，类的编译后的二进制接口与类的底层实现无关。仅需要重新编译和链接嵌套的实现类即可更改类的实现。外部类的用户不受影响。作为额外的好处，pImpl 模式提供了一种隐藏类底层私有细节的方法，这在分发类库或其他专有代码时可能很有用。

在我们的 pImpl 实现中包含`unique_ptr`的一个优点是，我们保证了相关实现类的正确销毁。我们还有潜力避免程序员无意中引入的指针和内存错误！

使用 pImpl 模式是一种权衡。仔细分析每个类和当前的应用程序将有助于确定 pImpl 模式是否适合您的设计。

我们现在已经看到了 pImpl 模式的实现，最初使用原始指针，然后应用了`unique_ptr`。现在，让我们简要回顾一下与模式相关的学习内容，然后进入我们书籍的附加章节，*第二十一章*，*使 C++更安全*。

# 摘要

在本章中，我们通过进一步掌握另一个核心设计模式来提高我们的编程技能，从而实现了成为更不可或缺的 C++程序员的宏伟目标。我们探讨了 pImpl 模式，最初使用原生 C++指针和关联进行实现，然后通过使用唯一指针来改进我们的实现。通过检查实现，我们很容易理解 pImpl 模式如何减少编译时依赖，并使我们的代码更依赖于实现。

利用核心设计模式，如 pImpl 模式，将帮助你更轻松地贡献可重用、可维护且其他熟悉常见设计模式的程序员可以理解的代码。你的软件解决方案将基于创造性和经过充分测试的设计解决方案。

我们现在一起完成了我们的最后一个设计模式，结束了在 C++中理解面向对象编程的漫长旅程。你现在拥有了许多技能，包括对面向对象有深入的理解、扩展的语言特性和核心设计模式，所有这些都使你成为一个更有价值的程序员。

尽管 C++是一种复杂的语言，具有额外的功能、补充技术和额外的设计模式需要探索，但你已经拥有了一个坚实的基础和专业知识水平，可以轻松地导航和接受你可能希望获得的任何额外的语言功能、库和模式。你已经走了很长的路；这已经是一次冒险的旅程！我享受了我们这次探索的每一分钟，我希望你也一样。

我们首先回顾了基本语言语法，并理解了 C++的必要要素，这些要素是我们当时即将开始的面向对象编程（OOP）之旅的基石。然后，我们将 C++视为一种面向对象的编程语言，不仅学习了必要的面向对象概念，还学习了如何使用 C++语言特性、编码技术或两者结合来实现这些概念。接着，我们通过添加异常处理、友元、运算符重载、模板、STL 基础以及测试面向对象类和组件的知识来扩展你的技能。然后，我们通过采用核心设计模式和深入应用感兴趣的模式来应用代码，我们冒险进入了更复杂的编程技术。

这些获得的知识技能块代表了 C++知识掌握的新层次。每个都将帮助你创建更易于维护和健壮的代码。你作为一个熟练的 C++面向对象程序员的未来正在等待。现在，让我们继续我们的附加章节，然后，让我们开始编程！

# 问题

1.  修改本章中使用的唯一指针的 pImpl 模式示例，在嵌套类的实现中进一步引入唯一指针。

1.  修改之前章节中的`Student`类，使其简单地从本章中采用 pImpl 模式的`Person`类继承。你遇到什么困难吗？现在，修改`Student`类，使其额外利用唯一指针的 pImpl 模式。一个建议的`Student`类是包含与`Course`关联的类。现在，你遇到什么困难吗？

1.  你能想象出哪些其他示例可以合理地结合 pImpl 模式以实现相对的实现独立性？

# 第五部分：C++中更安全编程的考虑因素

本部分的目标是了解作为程序员我们可以做什么来使 C++成为一种更安全的语言，这反过来将有助于使我们的程序更健壮。到目前为止，我们已经对 C++有了很多了解，从语言基础到在 C++中实现 OO 设计。我们已经增加了额外的技能，例如使用友元和运算符重载、异常处理、模板和 STL。我们甚至深入研究了几个流行的设计模式。我们知道我们几乎可以在 C++中做任何事情，但我们也已经看到，拥有如此大的能力可能会留下粗心编程和严重错误的空间，这可能导致难以维护的代码。

在本节中，我们将以敏锐的目光回顾全书所学内容，了解我们如何努力使我们的代码更加健壮。我们将致力于制定一套核心编程指南，目标只有一个：使我们的程序更安全！

我们将重新审视并扩展我们对智能指针（唯一、共享和弱引用）的知识，以及介绍一个互补的惯用语，RAII。我们将回顾与原生 C++指针相关的安全问题，并总结我们的安全担忧，以编程指南的形式：在新的 C++代码中始终优先使用智能指针。

我们将回顾现代编程特性，例如基于范围的`for`循环和 for-each 风格的循环，以了解这些简单的结构如何帮助我们避免常见错误。我们将重新审视`auto`关键字，而不是显式类型，以增加代码的安全性。我们将重新审视使用经过良好测试的 STL 类型，以确保我们的代码在使用临时容器时不会出错。我们将重新审视`const`限定符以多种方式增加代码的安全性。通过回顾全书使用的具体语言特性，我们将重新审视每个特性如何增加代码的安全性。我们还将考虑线程安全性以及我们全书所见的各种主题如何与线程安全性相关。

最后，我们将讨论核心编程指南，例如优先初始化而不是赋值，或者使用`virtual`、`override`或`final`之一来指定多态操作及其方法。我们将理解采用编程指南的重要性，并了解可用于支持在 C++中安全编程的资源。

本部分包含以下章节：

+   *第二十一章*，*使 C++更安全*

第五部分：C++更安全编程的考虑因素
