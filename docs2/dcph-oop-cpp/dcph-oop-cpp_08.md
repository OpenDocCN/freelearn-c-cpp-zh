# 8

# 掌握抽象类

本章将继续扩展我们对 C++面向对象编程知识的理解。我们将从探索一个强大的面向对象概念——**抽象类**开始，然后进一步了解这个想法是如何通过**直接语言支持**在 C++中实现的。

我们将使用纯虚函数实现抽象类，以最终支持相关类层次结构中的改进。我们将了解抽象类如何增强和与我们的多态理解相匹配。我们还将认识到本章中提出的抽象类面向对象概念将支持强大且灵活的设计，使我们能够轻松创建可扩展的 C++代码。

在本章中，我们将涵盖以下主要内容：

+   理解抽象类的面向对象概念

+   使用纯虚函数实现抽象类

+   使用抽象类和纯虚函数创建接口

+   使用抽象类泛化派生类对象，以及向上和向下转换

到本章结束时，您将理解抽象类的面向对象概念，以及如何通过纯虚函数在 C++中实现这一想法。您将了解仅包含纯虚函数的抽象类如何定义一个面向对象的概念——接口。您将理解抽象类和接口如何有助于强大的面向对象设计。

您将看到我们如何非常容易地使用抽象类型集合泛化相关、专业的对象组。我们将进一步探索在层次结构中的向上和向下转换，以了解允许什么以及何时进行此类类型转换是合理的。

通过理解 C++中抽象类的直接语言支持以及为什么创建接口是有用的，您将拥有更多工具来创建一个可扩展的相关类层次结构。让我们通过理解这些概念在 C++中的实现来扩展我们对 C++作为面向对象语言的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 中找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter08`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter08)。每个完整程序示例都可以在 GitHub 的相应章节标题（子目录）下找到，该文件对应章节编号，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter08`子目录中的名为`Chp8-Ex1.cpp`的文件中找到。

本章的 CiA 视频可以在以下链接中查看：[`bit.ly/3SZv0jy`](https://bit.ly/3SZv0jy)。

# 理解抽象类的面向对象概念

在本节中，我们将介绍一个基本面向对象的概念，即抽象类。这个概念将丰富你对关键 OO（面向对象）思想的了解，包括封装、信息隐藏、泛化、特化和多态。你知道如何封装一个类。你也知道如何使用单继承构建继承层次结构，以及构建层次结构的各种原因，例如支持*是*关系或支持实现继承的较少使用原因。此外，你知道如何通过多态的概念使用运行时绑定方法到操作，这是通过虚函数实现的。让我们通过探索**抽象类**来扩展我们不断增长的 OO 术语。

**抽象类**是一个基类，旨在收集派生类中可能存在的共同点，目的是在派生类上断言一个公共接口（即一组操作）。抽象类不代表一个旨在实例化的类。只有派生类类型的对象可以被实例化。

让我们从 C++语言特性开始，它允许我们实现抽象类，即纯虚函数。

# 使用纯虚函数实现抽象类

抽象类是通过在类定义中引入至少一个抽象方法（即纯虚函数原型）来指定的。抽象方法的概念是仅指定操作的协议（即仅成员函数的*名称*和*签名*），但没有函数定义。抽象方法将是多态的，因为没有定义，它预期将被派生类重新定义。

函数参数后跟一个 `=0`。此外，理解有关纯虚函数的以下细微差别也很重要：

+   通常不提供纯虚函数的定义。这相当于在基类级别指定操作（仅原型），并在派生类级别提供所有方法（成员函数定义）。

+   派生类没有为其基类引入的所有纯虚函数提供方法，也被认为是抽象的，因此不能实例化。

+   原型中的 `=0` 仅是向链接器指示，在创建可执行程序时，不需要（或解决）此函数的定义。

注意

抽象类是通过在类定义中包含一个或多个纯虚函数原型来指定的。通常不提供这些方法的可选定义。

纯虚函数通常不会有定义的原因是它们旨在为在派生类中实现的多态操作提供一个使用协议。纯虚函数指定一个类为抽象类；抽象类不能被实例化。因此，在纯虚函数中提供的定义永远不会被选为多态操作的正确方法，因为抽象类型的实例永远不会存在。话虽如此，纯虚函数仍然可以提供一个定义，该定义可以使用作用域解析运算符（`::`）和基类名称显式调用。也许，这种默认行为可能对作为派生类实现中辅助函数的有意义。

让我们从简要概述一下指定抽象类所需的语法开始。记住，使用*abstract*关键字本身并不用于指定抽象类。相反，仅仅通过引入一个或多个纯虚函数，我们就已经指明了该类是一个抽象类：

```cpp
class LifeForm    // Abstract class definition
{
private:
    // all LifeForms have a lifeExpectancy
    int lifeExpectancy = 0; // in-class initialization
public:
    LifeForm() = default; // def. ctor, uses in-class init 
    LifeForm(int life): lifeExpectancy(life) { }
    // Remember, we get default copy, even w/o proto below
    // LifeForm(const LifeForm &form) = default; 
    // Must include prototype to specify virtual destructor
    virtual ~LifeForm() = default;   // virtual destructor
    // Recall, [[nodiscard]] requires ret. value to be used
    [[nodiscard]] int GetLifeExpectancy() const 
        { return lifeExpectancy; }
    virtual void Print() const = 0; // pure virtual fns. 
    virtual string IsA() const = 0;   
    virtual string Speak() const = 0;
};
```

注意，在抽象类定义中，我们引入了四个虚函数，其中三个是纯虚函数。虚析构函数没有内存需要释放，但被标记为`virtual`，以便它是多态的，并且可以应用于存储为基类类型指针的派生类实例的正确销毁顺序。

三个纯虚函数，`Print()`、`IsA()`和`Speak()`，在它们的原型中用`=0`表示。这些操作没有定义（尽管可以有选择地定义）。纯虚函数可以有默认实现，但不能作为内联函数。提供这些操作的方法的责任将落在派生类身上，使用由基类定义指定的接口（即签名）。在这里，纯虚函数为在派生类定义中定义的多态操作提供了*接口*。

重要提示

抽象类肯定会有派生类（因为我们不能实例化抽象类本身）。为了确保虚拟析构机制在最终层次结构中适当工作，请确保在抽象类定义中包含一个*虚拟析构函数*。这将确保所有派生类析构函数都是虚拟的，并且可以被重写以在对象的销毁顺序中提供正确的入口点。

现在，让我们从面向对象的角度更深入地探讨拥有接口的含义。

# 创建接口

**接口类**是一个类的面向对象概念，它是抽象类的一个进一步细化。而抽象类可以包含泛化属性和默认行为（通过包含数据成员和纯虚函数的默认定义，或者通过提供非虚成员函数），接口类将只包含抽象方法。在 C++中，只包含抽象方法（即没有可选定义的纯虚函数）的抽象类可以被视为**接口类**。

当考虑 C++中实现的接口类时，记住以下内容是有用的：

+   抽象类是不可实例化的；它们通过继承提供（即，接口，即操作）派生类必须提供的接口。

+   虽然纯虚函数在抽象类中可能包含一个可选的实现（即，方法体），但如果类希望被视为在纯面向对象术语中的接口类，则不应提供此实现。

+   尽管抽象类可能包含数据成员，但如果类希望被视为接口类，则不应包含。

+   在面向对象术语中，抽象方法是一个没有方法的操作；它仅是接口，并在 C++中作为纯虚函数实现。

+   作为提醒，请确保在接口类定义中包含虚拟析构函数原型；这将确保派生类的析构函数将是虚拟的。析构函数定义应该是空的。

让我们考虑在面向对象编程（OOP）实现技术中拥有接口类的各种动机。一些面向对象编程（OOP）语言遵循非常严格的面向对象概念，并且只允许实现非常纯粹的面向对象设计。其他面向对象编程（OOP）语言，如 C++，提供了更多的灵活性，允许通过语言直接实现更激进的面向对象思想。

例如，在纯面向对象术语中，继承应该保留用于“是...的”关系。我们已经看到了实现继承，这是 C++通过私有和受保护基类支持的。我们已经看到了一些可接受的实现继承的使用，即，通过另一个（使用受保护和公共基类使用的能力来隐藏底层实现）来实现一个新的类。

另一个边缘面向对象编程（OOP）特性的例子是多继承。我们将在*第九章*，“探索多继承”中看到，C++允许一个类从多个基类派生。在某些情况下，我们确实是在说派生类与可能许多基类之间存在“是...的”关系，但并非总是如此。

一些面向对象的语言不允许多重继承，而那些不允许多重继承的语言则更多地依赖于接口类来混合（否则）多个基类的功能。在这些情况下，面向对象的语言可以允许派生类根据多个接口类中指定的功能实现，而不实际使用多重继承。理想情况下，接口用于从多个类中*混合*功能。这些类，不出所料，有时被称为**混合**类。在这些情况下，我们并不是说派生类和基类之间必然存在 Is-A 关系。

在 C++中，当我们引入一个只包含纯虚函数的抽象类时，我们可以将其视为创建一个接口类。当一个新类从多个接口中混合功能时，我们可以从面向对象的角度将其视为使用每个接口类作为混合所需接口以实现行为的一种手段。请注意，派生类必须用自己的实现覆盖每个纯虚函数；我们只是在混合所需的 API。

C++实现面向对象的接口概念仅仅是包含纯虚函数的抽象类。在这里，我们使用从抽象类的公共继承以及多态来模拟面向对象的接口类概念。请注意，其他语言（如 Java）直接在语言中实现这个想法（但那些语言不支持多重继承）。在 C++中，我们可以做几乎所有的事情，但了解如何以合理和有意义的方式实现面向对象的理念（即使这些理念没有直接的语言支持）仍然很重要。

让我们通过一个示例来展示如何使用抽象类来实现接口类：

```cpp
class Charitable    // interface class definition
{                   // implemented using an abstract class
public:
    virtual void Give(float) = 0; // interface for 'giving'
    // must include prototype to specify virtual destructor
    virtual ~Charitable() = default; // remember virt. dest
};
class Person: public Charitable   // mix-in an 'interface'
{
    // Assume typical Person class definition w/ data
    // members, constructors, member functions exist.
public:
    virtual void Give(float amt) override
    {  // implement a means for giving here 
    }
    ~Person() override;  // virtual destructor prototype
};
// Student Is-A Person which mixes-in Charitable interface
class Student: public Person 
{   
    // Assume typical Student class definition w/ data
    // members, constructors, member functions exist.
public:
    virtual void Give(float amt) override
    {  // Should a Student have little money to give,
       // perhaps they can donate their time equivalent to
       // the desired monetary amount they'd like to give
    }
    ~Student() override;  // virtual destructor prototype 
};
```

在上述类定义中，我们首先注意到一个简单的接口类`Charitable`，它使用受限的抽象类实现。我们不包含数据成员，而是一个纯虚函数`virtual void Give(float) = 0;`来定义接口类。我们还包含一个虚析构函数。

接下来，`Person`通过公共继承从`Charitable`派生出来，以实现`Charitable`接口。我们简单地覆盖`virtual void Give(float);`以提供默认的*捐赠*定义。然后我们从`Person`派生出`Student`；请注意，*学生是 Person 的一个混合（或实现）Charitable 接口的类*。在我们的`Student`类中，我们选择重新定义`virtual void Give(float);`以提供更适合`Student`实例的`Give()`定义。也许学生财务有限，选择捐赠相当于预定金额的时间。

在这里，我们使用 C++中的抽象类来模拟面向对象的接口类概念。

让我们继续讨论与抽象类相关的内容，通过考察派生类对象如何被抽象类类型收集来展开。

# 将派生类对象泛化为抽象类型

我们在*第七章*，“通过多态利用动态绑定”，看到有时将相关的派生类实例分组到一个使用基类指针存储的集合中是合理的。这样做允许使用基类指定的多态操作对相关的派生类类型进行统一处理。我们还知道，当调用多态基类操作时，由于 C++中实现多态的虚函数和内部 v-table，将在运行时调用正确的派生类方法。

然而，你可以思考一下，是否有可能通过一个抽象基类类型来收集一组相关的派生类类型。记住，抽象类是不可实例化的，那么我们如何将派生类对象存储为一个不能实例化的对象呢？解决方案是使用*指针*（甚至是一个引用）。由于我们不能在抽象基类实例的集合中收集派生类实例（这些类型不能实例化），我们可以在抽象类类型的指针集合中收集派生类实例。我们还可以让抽象类类型的引用指向派生类实例。自从我们学习了多态性以来，我们就一直在做这种类型的分组（使用基类指针）。

专门对象的泛化组使用隐式向上转型。撤销这种向上转型必须使用显式向下转型来完成，程序员需要确保之前泛化的派生类型是正确的。错误的向下转型到错误类型将导致运行时错误。

在什么情况下有必要通过基类类型（包括抽象基类类型）收集派生类对象？答案是当在你的应用程序中按更通用的方式处理相关的派生类类型是有意义的时候，也就是说，当基类类型中指定的操作涵盖了您希望利用的所有操作时。不可否认，你可能会发现同样多的情况，其中保持派生类实例在其自己的类型中（以利用在派生类级别引入的专用操作）是合理的。现在你理解了可能发生的情况。

让我们继续通过检查一个展示抽象类在行动中的综合示例来继续。

# 将所有部件组合在一起

到目前为止，在本章中，我们已经理解了抽象类的微妙之处，包括纯虚函数，以及如何使用抽象类和纯虚函数创建接口类。始终重要的是要看到我们的代码在行动中的表现，以及其所有各种组件及其各种细微差别。

让我们看看一个更复杂、完整的程序示例，以完全说明使用纯虚函数实现的抽象类。在这个例子中，我们不会进一步将抽象类指定为接口类，但我们将有机会使用它们抽象基类类型的一组指针收集相关的派生类类型。这个例子将被分成多个部分；完整的程序可以在以下 GitHub 位置找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter08/Chp8-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter08/Chp8-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
using std::cout;     // preferred to:  using namespace std;
using std::endl;
using std::setprecision;
using std::string;
using std::to_string;
constexpr int MAX = 5;
class LifeForm   // abstract class definition
{
private:
   int lifeExpectancy = 0;  // in-class initialization
public:
   LifeForm() = default;
   LifeForm(int life): lifeExpectancy(life) { }
   // Remember, we get the default copy ctor included,
   // even without the prototype below:
   // LifeForm(const LifeForm &) = default; 
   // Must include prototype to specify virtual destructor
   virtual ~LifeForm() = default;     // virtual destructor
   [[nodiscard]] int GetLifeExpectancy() const 
       { return lifeExpectancy; }
   virtual void Print() const = 0;   // pure virtual fns. 
   virtual string IsA() const = 0;   
   virtual string Speak() const = 0;
};
```

在上述类定义中，我们注意到`LifeForm`是一个抽象类。它是一个抽象类，因为它至少包含一个纯虚函数的定义。实际上，它包含三个纯虚函数的定义，即`Print()`、`IsA()`和`Speak()`。

现在，让我们通过一个具体的派生类`Cat`来扩展`Lifeform`：

```cpp
class Cat: public LifeForm
{
private:
   int numberLivesLeft = 9;  // in-class initialization
   string name;
   static constexpr int CAT_LIFE = 15;  // Life exp for cat
public:
   Cat(): LifeForm(CAT_LIFE) { } // note prior in-class init
   Cat(int lives): LifeForm(CAT_LIFE),
                   numberLivesLeft(lives) { }
   Cat(const string &);
   // Because base class destructor is virtual, ~Cat() is 
   // automatically virtual (overridden) whether or not 
   // explicitly prototyped. Below prototype not needed:
   // ~Cat() override = default;   // virtual destructor
   const string &GetName() const { return name; }
   int GetNumberLivesLeft() const 
       { return numberLivesLeft; }
   void Print() const override; // redef pure virt fns
   string IsA() const override { return "Cat"; }
   string Speak() const override { return "Meow!"; }
};
Cat::Cat(const string &n) : LifeForm(CAT_LIFE), name(n)
{  // numLivesLeft will be set with in-class initialization
}
void Cat::Print() const
{
   cout << "\t" << name << " has " << numberLivesLeft;
   cout << " lives left" << endl;
}
```

在之前的代码段中，我们看到了`Cat`类的定义。注意，`Cat`通过在`Cat`类中为这些方法提供定义，重新定义了`LifeForm`的纯虚函数`Print()`、`IsA()`和`Speak()`。由于这些函数已经有了现有的方法，任何`Cat`的派生类都可以选择性地重新定义这些方法以使用更合适的版本（但它们不再有义务这样做）。

注意，如果`Cat`未能重新定义`LifeForm`的任何一个纯虚函数，那么`Cat`也将被视为一个抽象类，因此不能实例化。

作为提醒，尽管`IsA()`和`Speak()`虚函数被内联编写以缩短代码，但编译器几乎永远不会内联虚函数，因为它们的正确方法必须在运行时确定（除了涉及编译器去虚化、final 方法或实例的动态类型已知的一些情况）。

注意，在`Cat`构造函数中，成员初始化列表被用来选择接受一个整型参数的`LifeForm`构造函数（即`:LifeForm(CAT_LIFE)`）。值`15`（`CAT_LIFE`）被传递给`LifeForm`构造函数，以将`LifeForm`中定义的`lifeExpectancy`初始化为`15`。此外，成员初始化列表还用于初始化`Cat`类中定义的数据成员，在类内初始化未使用的情况下（即，值由方法的参数确定）。

现在，让我们继续前进到`Person`类的定义，以及它的内联函数：

```cpp
class Person: public LifeForm
{
private: 
    string firstName;
    string lastName;
    char middleInitial = '\0';
    string title;  // Mr., Ms., Mrs., Miss, Dr., etc.
    static constexpr int PERSON_LIFE = 80;  // Life exp of
protected:                                  // a Person
    void ModifyTitle(const string &);  
public:
    Person();   // programmer-specified default constructor
    Person(const string &, const string &, char, 
           const string &);  
    // Default copy constructor prototype is not necessary:
    // Person(const Person &) = default;  // copy const.
    // Because base class destructor is virtual, ~Person() 
    // is automatically virtual (overridden) whether or not 
    // explicitly prototyped. Below prototype not needed:
    // ~Person() override = default;  // destructor
    const string &GetFirstName() const 
        { return firstName; }  
    const string &GetLastName() const 
        { return lastName; }    
    const string &GetTitle() const { return title; } 
    char GetMiddleInitial() const { return middleInitial; }
    void Print() const override; // redef pure virt fns
    string IsA() const override;   
    string Speak() const override;
};
```

注意现在`Person`使用公有继承扩展了`LifeForm`。在之前的章节中，`Person`是继承层次结构顶部的基类。`Person`重新定义了来自`LifeForm`的纯虚函数，即`Print()`、`IsA()`和`Speak()`。因此，`Person`现在是一个具体类，可以被实例化。

现在，让我们回顾`Person`的成员函数定义：

```cpp
// select the desired base constructor using mbr. init list
Person::Person(): LifeForm(PERSON_LIFE) 
{  // Remember, middleInitial will be set w/ in-class init
   // and the strings will be default constructed to empty
}
Person::Person(const string &fn, const string &ln, char mi,
               const string &t): LifeForm(PERSON_LIFE), 
                               firstName(fn), lastName(ln),
                               middleInitial(mi), title(t)
{
}
// We're using the default copy constructor. But if we did
// choose to prototype and define it, the method would be:
// Person::Person(const Person &p): LifeForm(p),
//           firstName(p.firstName), lastName(p.lastName),
//           middleInitial(p.middleInitial), title(p.title)
// {
// }
void Person::ModifyTitle(const string &newTitle)
{
   title = newTitle;
}
void Person::Print() const
{
   cout << "\t" << title << " " << firstName << " ";
   cout << middleInitial << ". " << lastName << endl;
}
string Person::IsA() const
{  
   return "Person";  
}
string Person::Speak() const 
{  
   return "Hello!";  
}  
```

在`Person`成员函数中，请注意我们为`Print()`、`IsA()`和`Speak()`提供了实现。此外，请注意在两个`Person`构造函数中，我们在它们的成员初始化列表中选择了`:LifeForm(PERSON_LIFE)`来调用`LifeForm(int)`构造函数。这个调用将设置私有继承数据成员`LifeExpectancy`为`80`（`PERSON_LIFE`）在给定`Person`实例的`LifeForm`子对象中。

接下来，让我们回顾`Student`类的定义，以及它的内联函数定义：

```cpp
class Student: public Person
{
private: 
    float gpa = 0.0;  // in-class initialization
    string currentCourse;
    const string studentId;  
    static int numStudents;
public:
    Student();  // programmer-supplied default constructor
    Student(const string &, const string &, char, 
            const string &, float, const string &, 
            const string &); 
    Student(const Student &);  // copy constructor
    ~Student() override;  // virtual destructor
    void EarnPhD();  
    float GetGpa() const { return gpa; }
    const string &GetCurrentCourse() const 
       { return currentCourse; }
    const string &GetStudentId() const 
       { return studentId; }
    void SetCurrentCourse(const string &);
    // Redefine not all of the virtrtual function; don't 
    // override Person::Speak(). Also, mark Print() as 
    // the final override
    void Print() const final override; 
    string IsA() const override;
    static int GetNumberStudents();  
};
int Student::numStudents = 0; // static data mbr def/init
inline void Student::SetCurrentCourse(const string &c)
{
    currentCourse = c; 
}
inline int Student::GetNumberStudents()
{
    return numStudents;
}
```

上述`Student`类的定义看起来与我们过去看到的非常相似。`Student`使用公有继承扩展了`Person`，因为`Student`是`Person`的一个子类。

接下来，我们将回顾非内联的`Student`类成员函数：

```cpp
// default constructor
Student::Student(): studentId(to_string(numStudents + 100) 
                                         + "Id")
{   // Set const studentId in mbr init list with unique id 
    // (based upon numStudents counter + 100), concatenated
    // with the string "Id". Remember, string member
    // currentCourse will be default constructed with
    // an empty string - it is a member object
    numStudents++;
}
// Alternate constructor member function definition
Student::Student(const string &fn, const string &ln, 
                 char mi, const string &t, float avg, 
                 const string &course, const string &id):
                 Person(fn, ln, mi, t), gpa(avg),
                 currentCourse(course), studentId(id)
{
    numStudents++;
}
// Copy constructor definition
Student::Student(const Student &s) : Person(s), 
                 gpa(s.gpa), 
                 currentCourse(s.currentCourse),
                 studentId(s.studentId)
{
    numStudents++;
}
// destructor definition
Student::~Student()
{
    numStudents--;
}
void Student::EarnPhD()  
{   
   ModifyTitle("Dr.");  
}
void Student::Print() const
{
   cout << "\t" << GetTitle() << " " << GetFirstName();
   cout << " " << GetMiddleInitial() << ". " 
        << GetLastName();
   cout << " id: " << studentId << "\n\twith gpa: ";
   cout << setprecision(3) << " " << gpa 
        << " enrolled in: " << currentCourse << endl;
}
string Student::IsA() const
{  
   return "Student";  
}
```

在之前列出的代码部分中，我们看到`Student`的非内联成员函数定义。到这一点，完整的类定义对我们来说在很大程度上是熟悉的。

因此，让我们检查`main()`函数：

```cpp
int main()
{
   // Notice that we are creating an array of POINTERS to
   // LifeForms. Since LifeForm cannot be instantiated, 
   // we could not create an array of LifeForm(s).
   LifeForm *entity[MAX] = { }; // init. with nullptrs
   entity[0] = new Person("Joy", "Lin", 'M', "Ms.");
   entity[1] = new Student("Renee", "Alexander", 'Z',
                           "Dr.", 3.95, "C++", "21-MIT"); 
   entity[2] = new Student("Gabby", "Doone", 'A', "Ms.", 
                            3.95, "C++", "18-GWU"); 
   entity[3] = new Cat("Katje");
   entity[4] = new Person("Giselle", "LeBrun", 'R',
                          "Miss");
   // Use range for-loop to process each element of entity
   for (LifeForm *item : entity)  // each item is a 
   {                              // LifeForm *       
      cout << item->Speak();
      cout << " I am a " << item->IsA() << endl;
      item->Print();
      cout << "\tHas a life expectancy of: ";
      cout << item->GetLifeExpectancy();
      cout << "\n";
   }
   for (LifeForm *item : entity) // process each element 
   {                             // in the entity array    
      delete item;
      item = nullptr;   // ensure deleted ptr isn't used
   }
   return 0;
}
```

在这里，在`main()`函数中，我们声明了一个指向`LifeForm`的指针数组。回想一下，`LifeForm`是一个抽象类。我们不能创建一个`LifeForm`对象的数组，因为这需要我们能够实例化一个`LifeForm`；我们不能——`LifeForm`是一个抽象类。

然而，我们可以创建一个抽象类型的指针集合，这允许我们收集相关类型，例如在这个集合中收集`Person`、`Student`和`Cat`实例。当然，我们可能应用于以这种泛型方式存储的实例的操作仅限于在抽象基类`LifeForm`中找到的操作。

然后，我们分配了各种`Person`、`Student`和`Cat`实例，通过类型为`LifeForm`的泛型指针集合的元素存储每个实例。当任何这些派生类实例以这种方式存储时，将执行隐式向上转换到抽象基类类型（但实例本身不会被任何方式改变——我们只是在指向构成整个内存布局的最基础类子对象）。

现在，我们通过循环应用在抽象类`LifeForm`中找到的操作，将这些操作应用于这个泛型集合中的所有实例，例如`Speak()`、`Print()`和`IsA()`。这些操作恰好是多态的，允许通过动态绑定利用每个实例最合适的实现。我们还对每个这些实例调用了`GetLifeExpectancy()`，这是一个在`LifeForm`级别找到的非虚函数。这个函数仅仅返回所讨论的`LifeForm`的生命预期。

最后，我们再次通过使用通用的`LifeForm`指针来遍历删除`Person`、`Student`和`Cat`的动态分配实例。我们知道`delete()`会调用析构函数，并且由于析构函数是虚函数，因此将开始适当的析构函数起始级别和正确的销毁顺序。此外，通过设置`item = nullptr;`，我们确保被删除的指针不会错误地用作有效地址（我们正在用`nullptr`覆盖每个释放的地址）。

在这个例子中，抽象类`LifeForm`的效用在于，它的使用允许我们将所有`LifeForm`对象的共同方面和行为集中在一个基类中（例如`lifeExpectancy`和`GetLifeExpectancy()`）。这些共同行为还扩展到一组具有所需接口的纯虚函数，即所有`LifeForm`对象都应该有的`Print()`、`IsA()`和`Speak()`。

重要提示

抽象类是收集派生类共同特性的类，但它本身并不代表一个有形的实体或对象，不应该被实例化。为了指定一个类为抽象类，它必须至少包含一个纯虚函数。

观察上述程序的输出，我们可以看到各种相关派生类类型的对象被实例化和统一处理。在这里，我们通过它们的抽象基类类型收集了这些对象，并在各种派生类中对基类中的纯虚函数进行了有意义的定义。

下面是完整程序示例的输出：

```cpp
Hello! I am a Person
        Ms. Joy M. Lin
        Has a life expectancy of: 80
Hello! I am a Student
        Dr. Renee Z. Alexander id: 21-MIT
        with gpa:  3.95 enrolled in: C++
        Has a life expectancy of: 80
Hello! I am a Student
        Ms. Gabby A. Doone id: 18-GWU
        with gpa:  3.95 enrolled in: C++
        Has a life expectancy of: 80
Meow! I am a Cat
        Katje has 9 lives left
        Has a life expectancy of: 15
Hello! I am a Person
        Miss Giselle R. LeBrun
        Has a life expectancy of: 80     
```

我们现在已经彻底研究了抽象类的 OO 概念及其在 C++中使用纯虚函数的实现，以及这些想法如何扩展到创建 OO 接口。在继续下一章之前，让我们简要回顾一下本章中我们涵盖的语言特性和 OO 概念。

# 摘要

在本章中，我们继续通过面向对象编程来推进我们的学习，首先，通过理解 C++中的纯虚函数如何直接提供对抽象类 OO 概念的语言支持。我们探讨了没有数据成员且不包含非虚函数的抽象类如何支持 OO 理想中的接口类。我们讨论了其他 OOP 语言如何利用接口类，以及 C++可能如何通过使用这种受限的抽象类来支持这种范式。我们将相关的派生类类型向上转换为存储为抽象基类类型的指针，这是一种典型且非常有用的编程技术。

我们已经看到，抽象类如何通过提供指定派生类共享的公共属性和行为，不仅补充了多态性，而且最值得注意的是，为相关类提供了多态行为的接口，因为抽象类本身是不可实例化的。

通过将抽象类和可能的对象导向概念中的接口类添加到我们的 C++编程资源中，我们能够实现促进代码易于扩展的设计。

我们现在准备继续学习第九章《探索多重继承》，通过学习如何和何时恰当地利用多重继承的概念来增强我们的面向对象编程技能，同时理解权衡和潜在的设计替代方案。让我们继续前进！

# 问题

1.  使用以下指南创建形状的层次结构：

    1.  创建一个名为`Shape`的抽象基类，该类定义了一个计算形状面积的操作。不要包含`Area()`操作的任何方法。提示：使用纯虚函数。

    1.  从`Shape`类使用公有继承派生出`Rectangle`、`Circle`和`Triangle`类。可选地，从`Rectangle`类派生出`Square`类。在每个派生类中重新定义`Shape`类引入的`Area()`操作。确保为每个派生类提供支持该操作的方法，以便你以后可以实例化每种类型的`Shape`。

    1.  根据需要添加数据成员和其他成员函数以完成新引入的类定义。记住，只有公共属性和操作应该在`Shape`中指定——所有其他属性都属于它们各自的派生类。不要忘记在每个类定义中实现复制构造函数和访问函数。

    1.  创建一个抽象类`Shape`类型的指针数组。将数组中的元素分配给`Rectangle`、`Square`、`Circle`和`Triangle`类型的实例。由于你现在将派生类对象作为通用的`Shape`对象处理，因此遍历指针数组并对每个实例调用`Area()`函数。确保删除你分配的任何动态分配的内存。

    1.  你的抽象`Shape`类在概念面向对象术语中也是一个接口类吗？为什么，或者为什么不？
