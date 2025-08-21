# 第八章：掌握抽象类

本章将继续扩展我们对 C++面向对象编程的知识。我们将首先探讨一个强大的面向对象概念，**抽象类**，然后逐步理解这一概念如何通过*直接语言支持*在 C++中实现。

我们将使用纯虚函数实现抽象类，最终支持相关类层次结构中的细化。我们将了解抽象类如何增强和配合我们对多态性的理解。我们还将认识到本章介绍的抽象类的面向对象概念将支持强大且灵活的设计，使我们能够轻松创建可扩展的 C++代码。

在本章中，我们将涵盖以下主要主题：

+   理解抽象类的面向对象概念

+   使用纯虚函数实现抽象类

+   使用抽象类和纯虚函数创建接口

+   使用抽象类泛化派生类对象；向上转型和向下转型

通过本章结束时，您将理解抽象类的面向对象概念，以及如何通过纯虚函数在 C++中实现这一概念。您将学会仅包含纯虚函数的抽象类如何定义面向对象概念的接口。您将了解为什么抽象类和接口有助于强大的面向对象设计。

您将看到我们如何非常容易地使用一组抽象类型来泛化相关的专门对象。我们还将进一步探讨层次结构中的向上转型和向下转型，以了解何时允许以及何时合理使用此类类型转换。

通过理解 C++中抽象类的直接语言支持，使用纯虚函数，以及创建接口的有用性，您将拥有更多工具来创建相关类的可扩展层次结构。让我们通过了解这些概念在 C++中的实现来扩展我们对 C++作为面向对象编程语言的理解。

# 技术要求

完整程序示例的在线代码可在以下 GitHub 网址找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter08`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter08)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节编号相对应，后跟破折号，再跟所在章节中的示例编号。例如，本章的第一个完整程序可以在`Chapter08`子目录中的名为`Chp8-Ex1.cpp`的文件中找到。

本章的 CiA 视频可在以下网址观看：[`bit.ly/2Pa6XBT`](https://bit.ly/2Pa6XBT)。

# 理解抽象类的面向对象概念

在本节中，我们将介绍一个重要的面向对象概念，即抽象类。考虑到您对关键面向对象思想的知识基础不断增长，包括封装、信息隐藏、泛化、特化和多态性，您知道如何封装一个类。您还知道如何使用单继承构建继承层次结构（以及构建层次结构的各种原因，例如支持**是一个**关系或支持实现继承的较少使用的原因）。此外，您知道如何使用虚函数实现方法到操作的运行时绑定，从而实现多态性的概念。让我们通过探索**抽象类**来扩展我们不断增长的面向对象术语。

**抽象类**是一个旨在收集派生类中可能存在的共同点，以便在派生类上断言一个公共接口（即一组操作）的基类。抽象类不代表一个用于实例化的类。只有派生类类型的对象可以被实例化。

让我们首先看一下 C++语言特性，允许我们实现抽象类，即纯虚拟函数。

# 使用纯虚拟函数实现抽象类

通过在类定义中引入至少一个抽象方法（即纯虚拟函数原型）来指定抽象类。**抽象方法**的面向对象概念是指定一个仅具有其使用协议（即成员函数的*名称*和*签名*）的操作，但没有函数定义。抽象方法将是多态的，因为没有定义，它预计会被派生类重新定义。

函数参数后面的`=0`。此外，重要的是要理解关于纯虚拟函数的以下微妙之处：

+   通常不提供纯虚拟函数的定义。这相当于在基类级别指定操作（仅原型），并在派生类级别提供所有方法（成员函数定义）。

+   未为其基类引入的所有纯虚拟函数提供方法的派生类也被视为抽象类，因此不能被实例化。

+   原型中的`=0`只是向链接器指示，在创建可执行程序时，不需要链接（或解析）此函数的定义。

注意

通过在类定义中包含一个或多个纯虚拟函数原型来指定抽象类。通常不提供这些方法的可选定义。

纯虚拟函数通常不提供定义的原因是它们旨在为多态操作提供使用协议，以在派生类中实现。纯虚拟函数指定一个类为抽象；抽象类不能被实例化。因此，纯虚拟函数中提供的定义永远不会被选择为多态操作的适当方法，因为抽象类型的实例永远不会存在。也就是说，纯虚拟函数仍然可以提供一个定义，可以通过作用域解析运算符（`::`）和基类名称显式调用。也许，这种默认行为可能作为派生类实现中使用的辅助函数具有意义。

让我们首先简要概述指定抽象类所需的语法。请记住，*abstract*可能是一个用于指定抽象类的关键字。相反，仅仅通过引入一个或多个纯虚拟函数，我们已经指示该类是一个抽象类：

```cpp
class LifeForm    // Abstract class definition
{
private:
    int lifeExpectancy; // all LifeForms have a lifeExpectancy
public:
    LifeForm() { lifeExpectancy = 0; }
    LifeForm(int life) { lifeExpectancy = life; }
    LifeForm(const LifeForm &form) 
       { lifeExpectancy = form.lifeExpectancy; }
    virtual ~LifeForm() { }   // virtual destructor
    int GetLifeExpectancy() const { return lifeExpectancy; }
    virtual void Print() const = 0; // pure virtual functions 
    virtual const char *IsA() = 0;   
    virtual const char *Speak() = 0;
};
```

请注意，在抽象类定义中，我们引入了四个虚拟函数，其中三个是纯虚拟函数。虚拟析构函数没有要释放的内存，但被指定为`virtual`，以便它是多态的，并且可以应用正确的销毁顺序到存储为基类类型指针的派生类实例。

三个纯虚拟函数`Print()`、`IsA()`和`Speak()`在它们的原型中被指定为`=0`。这些操作没有定义（尽管可以选择性地提供）。纯虚拟函数可以有默认实现，但不能作为内联函数。派生类的责任是使用基类定义指定的接口（即签名）为这些操作提供方法。在这里，纯虚拟函数为多态操作提供了*接口*，这些操作将在派生类定义中定义。

注意

抽象类肯定会有派生类（因为我们不能实例化抽象类本身）。为了确保虚析构函数机制在最终层次结构中能够正常工作，请确保在抽象类定义中包含虚析构函数。这将确保所有派生类的析构函数都是`virtual`，并且可以被重写以提供对象销毁序列中的正确入口点。

现在，让我们更深入地了解从面向对象的角度来拥有接口意味着什么。

# 创建接口。

接口类是面向对象概念中的一个类，它是抽象类的进一步细化。抽象类可以包含通用属性和默认行为（通过包含数据成员和纯虚函数的默认定义，或者通过提供非虚拟成员函数），而接口类只包含抽象方法。在 C++中，一个只包含抽象方法的抽象类（即没有可选定义的纯虚函数）可以被视为接口类。

在考虑 C++中实现的接口类时，有几点需要记住：

+   抽象类不可实例化；它们通过继承提供了派生类必须提供的接口（即操作）。

+   虽然在抽象类中纯虚函数可能包含可选实现（即方法体），但如果类希望在纯面向对象的术语中被视为接口类，则不应提供此实现。

+   虽然抽象类可能有数据成员，但如果类希望被视为接口类，则不应该有数据成员。

+   在面向对象的术语中，抽象方法是没有方法的操作；它只是接口，并且在 C++中实现为纯虚函数。

+   作为提醒，请确保在接口类定义中包含虚析构函数原型；这将确保派生类的析构函数是虚拟的。析构函数定义应为空。

让我们考虑在面向对象编程实现技术中拥有接口类的各种动机。一些面向对象编程语言遵循非常严格的面向对象概念，只允许实现非常纯粹的面向对象设计。其他面向对象编程语言，如 C++，通过直接允许实现更激进的面向对象思想，提供了更多的灵活性。

例如，在纯面向对象的术语中，继承应该保留给 Is-A 关系。我们已经看到了 C++支持的实现继承，通过私有和受保护的基类。我们已经看到了一些可接受的实现继承的用法，即以另一个类的术语实现一个新类（通过使用受保护和公共基类来隐藏底层实现）。

另一个面向对象编程特性的例子是多重继承。我们将在接下来的章节中看到，C++允许一个类从多个基类派生。在某些情况下，我们确实在说派生类与许多基类可能存在 Is-A 关系，但并非总是如此。

一些面向对象编程语言不允许多重继承，而那些不允许的语言更多地依赖于接口类来混合（否则）多个基类的功能。在这些情况下，面向对象编程语言可以允许派生类实现多个接口类中指定的功能，而不实际使用多重继承。理想情况下，接口用于混合多个类的功能。这些类，不出所料，有时被称为**混入**类。在这些情况下，我们并不一定说派生类和基类之间存在 Is-A 关系。

在 C++中，当我们引入一个只有纯虚函数的抽象类时，我们可以认为创建了一个接口类。当一个新类混合了来自多个接口的功能时，我们可以在面向对象的术语中将其视为使用接口类来混合所需的行为接口。请注意，派生类必须用自己的实现重写每个纯虚函数；我们只混合所需的 API。

C++对面向对象概念中的接口的实现仅仅是一个只包含纯虚函数的抽象类。在这里，我们使用公共继承自抽象类，配合多态性来模拟面向对象概念中的接口类。请注意，其他语言（如 Java）直接在语言中实现了这个想法（但是这些语言不支持多重继承）。在 C++中，我们几乎可以做任何事情，但重要的是要理解如何以合理和有意义的方式实现面向对象理想（即使这些理想在直接语言支持中没有提供）。

让我们看一个例子来说明使用抽象类实现接口类：

```cpp
class Charitable    // interface class definition
{                   // implemented using an abstract class
public:
    virtual void Give(float) = 0; // interface for 'giving'
    virtual ~Charitable() { } // remember virtual destructor
};
class Person: public Charitable   // mix-in an 'interface'
{
    // Assume typical Person class definition w/ data members,
    // constructors, member functions exist.
public:
    virtual void Give(float amt) override
    {  // implement a means for giving here 
    }
    virtual ~Person();  // prototype
};               
class Student: public Person 
{   // Student Is-A Person which mixes-in Charitable interface
    // Assume typical Student class definition w/ data
    // members, constructors, member functions exist.
public:
    virtual void Give(float amt) override
    {  // Should a Student have little money to give,
       // perhaps they can donate their time equivalent to
       // the desired monetary amount they'd like to give
    }
    virtual ~Student();  // prototype
};
```

在上述的类定义中，我们首先注意到一个简单的接口类`Charitable`，使用受限的抽象类实现。我们不包括数据成员，一个纯虚函数来定义`virtual void Give(float) = 0;`接口，以及一个虚析构函数。

接下来，`Person`从`Charitable`派生，使用公共继承来实现`Charitable`接口。我们简单地重写`virtual void Give(float);`来为*给予*提供一个默认定义。然后我们从`Person`派生`Student`；请注意*学生是一个实现了 Charitable 接口的人*。在我们的`Student`类中，我们选择重新定义`virtual void Give(float);`来为`Student`实例提供更合适的`Give()`定义。也许`Student`实例财务有限，选择捐赠一个等同于预定货币金额的时间量。

在这里，我们在 C++中使用抽象类来模拟面向对象概念中的接口类。

让我们继续讨论关于抽象类的整体问题，通过检查派生类对象如何被抽象类类型收集。

# 将派生类对象泛化为抽象类型

我们在*第七章*中已经看到，*通过多态性利用动态绑定*，有时将相关的派生类实例分组存储在使用基类指针的集合中是合理的。这样做允许使用基类指定的多态操作对相关的派生类类型进行统一处理。我们也知道，当调用多态基类操作时，由于 C++中实现多态性的虚函数和内部虚表，将在运行时调用正确的派生类方法。

然而，你可能会思考，是否可能通过抽象类类型来收集一组相关的派生类类型？请记住，抽象类是不可实例化的，那么我们如何将一个派生类对象存储为一个不能被实例化的对象呢？解决方案是使用*指针*。虽然我们不能将派生类实例收集在一组抽象基类实例中（这些类型不能被实例化），但我们可以将派生类实例收集在抽象类类型的指针集合中。自从我们学习了多态性以来，我们一直在做这种类型的分组（使用基类指针）。

广义的专门对象组使用隐式向上转型。撤消这样的向上转型必须使用显式向下转型，并且程序员需要正确地确定先前泛化的派生类型。对错误的向下转型将导致运行时错误。

何时需要按基类类型收集派生类对象，包括抽象基类类型？答案是，当在应用程序中以更通用的方式处理相关的派生类类型时，即当基类类型中指定的操作涵盖了您想要利用的所有操作时。毫无疑问，您可能会发现同样多的情况，即保留派生类实例在其自己的类型中（以利用在派生类级别引入的专门操作）是合理的。现在您明白了可能发生的情况。

让我们继续通过检查一个全面的示例来展示抽象类的实际应用。

# 将所有部分放在一起

到目前为止，在本章中，我们已经了解了抽象类的微妙之处，包括纯虚函数，以及如何使用抽象类和纯虚函数创建接口类。始终重要的是看到我们的代码在各种组件及其各种细微差别中的运行情况。

让我们看一个更复杂的、完整的程序示例，以充分说明在 C++中使用纯虚函数实现抽象类。在这个例子中，我们不会进一步将抽象类指定为接口类，但我们将利用机会使用一组指向其抽象基类类型的指针来收集相关的派生类类型。这个例子将被分解成许多段落；完整的程序可以在以下 GitHub 位置找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter08/Chp8-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter08/Chp8-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace std;
const int MAX = 5;
class LifeForm   // abstract class definition
{
private:
   int lifeExpectancy;
public:
   LifeForm() { lifeExpectancy = 0; }
   LifeForm(int life) { lifeExpectancy = life; }
   LifeForm(const LifeForm &form) 
       { lifeExpectancy = form.lifeExpectancy; }
   virtual ~LifeForm() { }     // virtual destructor
   int GetLifeExpectancy() const { return lifeExpectancy; }
   virtual void Print() const = 0;   // pure virtual functions 
   virtual const char *IsA() = 0;   
   virtual const char *Speak() = 0;
};
```

在上述的类定义中，我们注意到`LifeForm`是一个抽象类。它是一个抽象类，因为它包含至少一个纯虚函数定义。事实上，它包含了三个纯虚函数定义，即`Print()`、`IsA()`和`Speak()`。

现在，让我们用一个具体的派生类`Cat`来扩展`LifeForm`：

```cpp
class Cat: public LifeForm
{
private:
   int numberLivesLeft;
   char *name;
public:
   Cat() : LifeForm(15) { numberLivesLeft = 9; name = 0; }
   Cat(int lives) : LifeForm(15) { numberLivesLeft = lives; }
   Cat(const char *n);
   virtual ~Cat() { delete name; }   // virtual destructor
   const char *GetName() const { return name; }
   int GetNumberLivesLeft() const { return numberLivesLeft; }
   virtual void Print() const override; // redef pure virt fns
   virtual const char *IsA() override { return "Cat"; }
   virtual const char *Speak() override { return "Meow!"; }
};
Cat::Cat(const char *n) : LifeForm(15)
{
   name = new char [strlen(n) + 1];
   strcpy(name, n);
   numberLivesLeft = 9;
}
void Cat::Print() const
{
   cout << "\t" << name << " has " << GetNumberLivesLeft();
   cout << " lives left" << endl;
}
```

在前面的代码段中，我们看到了`Cat`的类定义。请注意，`Cat`已经重新定义了`LifeForm`的纯虚函数`Print()`、`IsA()`和`Speak()`，并为`Cat`类中的每个方法提供了定义。有了这些函数的现有方法，`Cat`的任何派生类都可以选择重新定义这些方法，使用更合适的版本（但它们不再有义务这样做）。

请注意，如果`Cat`未能重新定义`LifeForm`的任何一个纯虚函数，那么`Cat`也将被视为抽象类，因此无法实例化。

作为提醒，虚函数`IsA()`和`Speak()`虽然是内联写的以缩短代码，但编译器永远不会将虚函数内联，因为它们的正确方法必须在运行时确定。

请注意，在`Cat`构造函数中，成员初始化列表用于选择接受整数参数的`LifeForm`构造函数（即`:LifeForm(15)`）。将值`15`传递给`LifeForm`构造函数，以初始化`LifeForm`中定义的`lifeExpectancy`为`15`。

现在，让我们继续前进到`Person`的类定义，以及它的内联函数：

```cpp
class Person: public LifeForm
{
private: 
   // data members
   char *firstName;
   char *lastName;
   char middleInitial;
   char *title;  // Mr., Ms., Mrs., Miss, Dr., etc.
protected:
   void ModifyTitle(const char *);  
public:
   Person();   // default constructor
   Person(const char *, const char *, char, const char *);  
   Person(const Person &);  // copy constructor
   virtual ~Person();  // destructor
   const char *GetFirstName() const { return firstName; }  
   const char *GetLastName() const { return lastName; }    
   const char *GetTitle() const { return title; } 
   char GetMiddleInitial() const { return middleInitial; }
   virtual void Print() const override; // redef pure virt fns
   virtual const char *IsA() override;   
   virtual const char *Speak() override;
};
```

请注意，`Person`现在使用公共继承扩展了`LifeForm`。在之前的章节中，`Person`是继承层次结构顶部的基类。`Person`重新定义了来自`LifeForm`的纯虚函数，即`Print()`、`IsA()`和`Speak()`。因此，`Person`现在是一个具体类，可以被实例化。

现在，让我们回顾一下`Person`的成员函数定义：

```cpp
Person::Person(): LifeForm(80)
{
   firstName = lastName = 0;  // NULL pointer
   middleInitial = '\0';
   title = 0;
}
Person::Person(const char *fn, const char *ln, char mi, 
               const char *t): LifeForm(80)
{
   firstName = new char [strlen(fn) + 1];
   strcpy(firstName, fn);
   lastName = new char [strlen(ln) + 1];
   strcpy(lastName, ln);
   middleInitial = mi;
   title = new char [strlen(t) + 1];
   strcpy(title, t);
}
Person::Person(const Person &pers): LifeForm(pers)
{
   firstName = new char [strlen(pers.firstName) + 1];
   strcpy(firstName, pers.firstName);
   lastName = new char [strlen(pers.lastName) + 1];
   strcpy(lastName, pers.lastName);
   middleInitial = pers.middleInitial;
   title = new char [strlen(pers.title) + 1];
   strcpy(title, pers.title);
}
Person::~Person()
{
   delete firstName;
   delete lastName;
   delete title;
}
void Person::ModifyTitle(const char *newTitle)
{
   delete title;  // delete old title
   title = new char [strlen(newTitle) + 1];
   strcpy(title, newTitle);
}
void Person::Print() const
{
   cout << "\t" << title << " " << firstName << " ";
   cout << middleInitial << ". " << lastName << endl;
}
const char *Person::IsA() {  return "Person";  }
const char *Person::Speak() {  return "Hello!";  }   
```

在`Person`成员函数中，请注意我们为`Print()`、`IsA()`和`Speak()`实现了功能。另外，请注意在两个`Person`构造函数中，我们在它们的成员初始化列表中选择了`:LifeForm(80)`来调用`LifeForm(int)`构造函数。这个调用将在给定`Person`实例的`LifeForm`子对象中将私有继承的数据成员`LifeExpectancy`设置为`80`。

接下来，让我们回顾`Student`类的定义，以及它的内联函数定义：

```cpp
class Student: public Person
{
private: 
   // data members
   float gpa;
   char *currentCourse;
   const char *studentId;  
public:
   Student();  // default constructor
   Student(const char *, const char *, char, const char *,
           float, const char *, const char *); 
   Student(const Student &);  // copy constructor
   virtual ~Student();  // virtual destructor
   void EarnPhD();  
   float GetGpa() const { return gpa; }
   const char *GetCurrentCourse() const 
       { return currentCourse; }
   const char *GetStudentId() const { return studentId; }
   void SetCurrentCourse(const char *);
   virtual void Print() const override; // redefine not all 
   virtual const char *IsA() override;  // virtual functions
};
inline void Student::SetCurrentCourse(const char *c)
{
   delete currentCourse;   // delete existing course
   currentCourse = new char [strlen(c) + 1];
   strcpy(currentCourse, c); 
}
```

前面提到的`Student`类定义看起来很像我们以前见过的。`Student`使用公共继承扩展了`Person`，因为`Student` *是一个* `Person`。

接下来，我们将回顾非内联的`Student`类成员函数：

```cpp
Student::Student(): studentId (0)  // default constructor
{
   gpa = 0.0;
   currentCourse = 0;
}
// Alternate constructor member function definition
Student::Student(const char *fn, const char *ln, char mi, 
                 const char *t, float avg, const char *course,
                 const char *id): Person(fn, ln, mi, t)
{
   gpa = avg;
   currentCourse = new char [strlen(course) + 1];
   strcpy(currentCourse, course);
   char *temp = new char [strlen(id) + 1];
   strcpy (temp, id); 
   studentId = temp;
}
// Copy constructor definition
Student::Student(const Student &ps): Person(ps)
{
   gpa = ps.gpa;
   currentCourse = new char [strlen(ps.currentCourse) + 1];
   strcpy(currentCourse, ps.currentCourse);
   char *temp = new char [strlen(ps.studentId) + 1];
   strcpy (temp, ps.studentId); 
   studentId = temp;
}

// destructor definition
Student::~Student()
{
   delete currentCourse;
   delete (char *) studentId;
}
void Student::EarnPhD()  {   ModifyTitle("Dr.");  }
void Student::Print() const
{
   cout << "\t" << GetTitle() << " " << GetFirstName() << " ";
   cout << GetMiddleInitial() << ". " << GetLastName();
   cout << " with id: " << studentId << " has a gpa of: ";
   cout << setprecision(2) <<  " " << gpa << " enrolled in: ";
   cout << currentCourse << endl;
}
const char *Student::IsA() {  return "Student";  }
```

在前面列出的代码部分中，我们看到了`Student`的非内联成员函数定义。到目前为止，完整的类定义对我们来说已经非常熟悉了。

因此，让我们来审查一下`main()`函数：

```cpp
int main()
{
   // Notice that we are creating an array of POINTERS to
   // LifeForms. Since LifeForm cannot be instantiated, 
   // we could not create an array of LifeForm (s).
   LifeForm *entity[MAX];
   entity[0] = new Person("Joy", "Lin", 'M', "Ms.");
   entity[1] = new Student("Renee", "Alexander", 'Z', "Dr.",
                            3.95, "C++", "21-MIT"); 
   entity[2] = new Student("Gabby", "Doone", 'A', "Ms.", 
                            3.95, "C++", "18-GWU"); 
   entity[3] = new Cat("Katje"); 
   entity[4] = new Person("Giselle", "LeBrun", 'R', "Miss");
   for (int i = 0; i < MAX; i++)
   {
      cout << entity[i]->Speak();
      cout << " I am a " << entity[i]->IsA() << endl;
      entity[i]->Print();
      cout << "Has a life expectancy of: ";
      cout << entity[i]->GetLifeExpectancy();
      cout << "\n";
   } 
   for (int i = 0; i < MAX; i++)
      delete entity[i];
   return 0;
}
```

在`main()`中，我们声明了一个指向`LifeForm`的指针数组。回想一下，`LifeForm`是一个抽象类。我们无法创建`LifeForm`对象的数组，因为那将要求我们能够实例化一个`LifeForm`；我们不能这样做——`LifeForm`是一个抽象类。

然而，我们可以创建一个指向抽象类型的指针集合，这使我们能够收集相关类型——在这个集合中的`Person`、`Student`和`Cat`实例。当然，我们可以对以这种泛化方式存储的实例应用的唯一操作是在抽象基类`LifeForm`中找到的那些操作。

接下来，我们分配了各种`Person`、`Student`和`Cat`实例，将每个实例存储在类型为`LifeForm`的泛化指针集合的元素中。当以这种方式存储任何这些派生类实例时，将执行隐式向上转型到抽象基类类型（但实例不会以任何方式被改变——我们只是指向整个内存布局组成部分的最基类子对象）。

现在，我们通过循环来对这个泛化集合中的所有实例应用在抽象类`LifeForm`中找到的操作，比如`Speak()`、`Print()`和`IsA()`。这些操作恰好是多态的，允许通过动态绑定使用每个实例的最适当实现。我们还在每个实例上调用`GetLifeExpectancy()`，这是在`LifeForm`级别找到的非虚拟函数。这个函数只是返回了相关`LifeForm`的寿命预期。

最后，我们通过循环使用泛化的`LifeForm`指针再次删除动态分配的`Person`、`Student`和`Cat`实例。我们知道`delete()`将会调用析构函数，并且因为析构函数是虚拟的，适当的析构顺序将会开始。

在这个例子中，`LifeForm`抽象类的实用性在于它的使用允许我们将所有`LifeForm`对象的共同特征和行为概括在一个基类中（比如`lifeExpectancy`和`GetLifeExpectancy()`）。这些共同行为还扩展到一组具有所需接口的纯虚函数，所有`LifeForm`对象都应该有，即`Print()`、`IsA()`和`Speak()`。

重要提醒

抽象类是收集派生类的共同特征，但本身并不代表应该被实例化的有形实体或对象。为了将一个类指定为抽象类，它必须包含至少一个纯虚函数。

查看上述程序的输出，我们可以看到各种相关的派生类类型的对象被实例化并统一处理。在这里，我们通过它们的抽象基类类型收集了这些对象，并且在各种派生类中用有意义的定义覆盖了基类中的纯虚函数。

以下是完整程序示例的输出：

```cpp
Hello! I am a Person
        Ms. Joy M. Lin
        Has a life expectancy of: 80
Hello! I am a Student
        Dr. Renee Z. Alexander with id: 21-MIT has a gpa of:  4 enrolled in: C++
        Has a life expectancy of: 80
Hello! I am a Student
        Ms. Gabby A. Doone with id: 18-GWU has a gpa of: 4 enrolled in: C++
        Has a life expectancy of: 80
Meow! I am a Cat
        Katje has 9 lives left
        Has a life expectancy of: 15
Hello! I am a Person
        Miss Giselle R. LeBrun
        Has a life expectancy of: 80
```

我们已经彻底研究了抽象类的面向对象概念以及在 C++中如何使用纯虚函数实现，以及这些概念如何扩展到创建面向对象接口。在继续前进到下一章之前，让我们简要回顾一下本章涵盖的语言特性和面向对象概念。

# 总结

在本章中，我们继续了解面向对象编程，首先是通过理解 C++中纯虚函数如何直接支持抽象类的面向对象概念。我们探讨了没有数据成员且不包含非虚函数的抽象类如何支持接口类的面向对象理想。我们谈到了其他面向对象编程语言如何利用接口类，以及 C++如何选择支持这种范式，通过使用这种受限制的抽象类。我们将相关的派生类类型向上转换为抽象基类类型的指针存储，这是一种典型且非常有用的编程技术。

我们已经看到抽象类如何通过提供一个类来指定派生类共享的共同属性和行为，以及最重要的是为相关类提供多态行为的接口，因为抽象类本身是不可实例化的。

通过在 C++中添加抽象类和可能的面向对象接口类的概念，我们能够实现促进易于扩展的代码设计。

我们现在准备继续*第九章*，*探索多重继承*，通过学习如何以及何时适当地利用多重继承的概念，同时理解权衡和潜在的设计替代方案，来增强我们的面向对象编程技能。让我们继续前进吧！

# 问题

1.  使用以下指南创建形状的层次结构：

a. 创建一个名为`Shape`的抽象基类，它定义了计算`Shape`面积的操作。不要包括`Area()`操作的方法。提示：使用纯虚函数。

b. 使用公共继承从`Shape`派生`Rectangle`、`Circle`和`Triangle`类。可选择从`Rectangle`派生`Square`类。在每个派生类中重新定义`Shape`引入的`Area()`操作。确保在每个派生类中提供支持该操作的方法，以便稍后实例化每种`Shape`类型。

c. 根据需要添加数据成员和其他成员函数来完成新引入的类定义。记住，只有共同的属性和操作应该在`Shape`中指定 - 所有其他属性和操作都属于它们各自的派生类。不要忘记在每个类定义中实现复制构造函数和访问函数。

d. 创建一个抽象类类型`Shape`的指针数组。将该数组中的元素指向`Rectangle`、`Square`、`Circle`和`Triangle`类型的实例。由于现在你正在将派生类对象视为通用的`Shape`对象，所以循环遍历指针数组，并为每个调用`Area()`函数。确保`delete()`任何动态分配的内存。

e. 在概念上，你的抽象`Shape`类也是一个接口类吗？为什么或为什么不是？
