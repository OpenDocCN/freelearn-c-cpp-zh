# 第十五章：测试类和组件

本章将继续探索如何通过探索测试组成我们面向对象程序的类和组件的方法，来增加您的 C++编程技能库。我们将探索各种策略，以确保我们编写的代码经过充分测试并且健壮。

本章将展示如何通过测试单个类以及测试一起工作的各种组件来测试您的面向对象程序。

在本章中，我们将涵盖以下主要主题：

+   理解规范类形式；创建健壮的类

+   创建驱动程序来测试类

+   测试通过继承、关联或聚合相关的类

+   测试异常处理机制

通过本章结束时，您将掌握各种技术，确保您的代码在投入生产之前经过充分测试。具备持续产生健壮代码的技能将帮助您成为更有益的程序员。

让我们通过研究各种面向对象测试技术来增强我们的 C++技能。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名对应于章节号，后跟破折号，再跟随该章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp15-Ex1.cpp`的文件中的`Chapter15`子目录中找到上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/314TI8h`](https://bit.ly/314TI8h)。

# 思考面向对象测试

在部署任何代码之前，软件测试非常重要。测试面向对象的软件将需要不同于其他类型软件的技术。因为面向对象的软件包含类之间的关系，我们必须了解如何测试可能存在的类之间的依赖关系和关系。此外，每个对象可能会根据对每个实例应用操作的顺序以及与相关对象的特定交互而进入不同的状态（例如，通过关联）。与过程性应用程序相比，面向对象应用程序的整体控制流程要复杂得多，因为应用于给定对象的操作的组合和顺序以及相关对象的影响是多种多样的。

然而，我们可以应用指标和流程来测试面向对象的软件。这些范围从理解我们可以应用于类规范的习语和模式，到创建驱动程序来独立测试类以及它们与其他类的关系。这些流程还可以包括创建场景，以提供对象可能经历的事件或状态的可能序列。对象之间的关系，如继承、关联和聚合，在测试中变得非常重要；相关对象可以影响现有对象的状态。

让我们从理解我们经常可以应用于开发的类的简单模式开始，来开始我们在测试面向对象软件中的探索。这种习语将确保一个类可能是完整的，没有意外的行为。我们将从规范类形式开始。

# 理解规范类形式

对于 C++中的许多类来说，遵循类规范的模式是合理的，以确保新类包含所需的全部组件。规范类形式是一个强大的类规范，使得类实例能够在初始化、赋值、参数传递和从函数返回值的使用等方面提供统一的行为（类似于标准数据类型）。规范类形式将适用于大多数既用于实例化的类，又用于作为新派生类的公共基类的类。打算作为私有或受保护基类的类（即使它们可能被实例化）可能不遵循这种习惯的所有部分。

遵循正统规范形式的类将包括：

+   一个默认构造函数

+   一个复制构造函数

+   一个过载的赋值运算符

+   虚析构函数

遵循扩展规范形式的类还将包括：

+   一个“移动”复制构造函数

+   一个“移动”赋值运算符

让我们在下面的子节中看看规范类形式的每个组件。

## 默认构造函数

简单实例化需要一个默认构造函数。虽然如果一个类不包含构造函数，将会提供一个默认（空）构造函数，但重要的是要记住，如果一个类包含其他签名的构造函数，将不会提供默认构造函数。最好提供一个合理的基本初始化的默认构造函数。

此外，在成员初始化列表中没有指定替代基类构造函数的情况下，将调用给定类的基类的默认构造函数。如果基类没有这样的默认构造函数（并且没有提供另一个签名的构造函数），则对基类构造函数的隐式调用将被标记为错误。

让我们还考虑多重继承情况，其中出现了菱形继承结构，并且使用虚基类来消除最派生类实例中大多数基类子对象的重复。在这种情况下，除非在负责创建菱形形状的派生类的成员初始化列表中另有规定，否则现在*共享*基类子对象的默认构造函数将被调用。即使在中间级别指定了非默认构造函数，当中间级别指定了一个可能共享的虚基类时，这些规定也会被忽略。

## 复制构造函数

对于包含指针数据成员的所有对象来说，复制构造函数是至关重要的。除非程序员提供了复制构造函数，否则系统将在应用程序中必要时链接系统提供的复制构造函数。系统提供的复制构造函数执行所有数据成员的成员逐一（浅层）复制。这意味着一个类的多个实例可能包含指向共享内存块的指针，这些内存块代表应该是个体化的数据。此外，记得在派生类的复制构造函数中使用成员初始化列表来指定基类的复制构造函数以复制基类的数据成员。当然，在深度方式中复制基类子对象是至关重要的；此外，基类数据成员不可避免地是私有的，因此在派生类的成员初始化列表中选择基类复制构造函数非常重要。

通过指定一个复制构造函数，我们还帮助提供了一个对象通过值从函数传递（或返回）的预期方式。在这些情况下确保深层复制是至关重要的。用户可能认为这些复制是“通过值”，但如果它们的指针数据成员实际上与源实例共享，那么它实际上并不是通过值传递（或返回）对象。

## 过载的赋值运算符

一个**重载的赋值运算符**，就像复制构造函数一样，对于所有包含指针数据成员的对象也是至关重要的。系统提供的赋值运算符的默认行为是从源对象到目标对象的浅赋值。同样，当数据成员是指针时，强烈建议重载赋值运算符以为任何这样的指针数据成员分配空间。

另外，请记住，重载的赋值运算符不会*继承*；每个类都负责编写自己的版本。这是有道理的，因为派生类不可避免地有更多的数据成员需要复制，而其基类中的赋值运算符则可能是私有的或无法访问的。然而，在派生类中重载赋值运算符时，请记住调用基类的赋值运算符来执行继承的基类成员的深度赋值（这些成员可能是私有的或无法访问的）。

## 虚析构函数

虚析构函数在使用公共继承时是必需的。通常，派生类实例被收集在一组中，并由一组基类指针进行泛化。请记住，以这种方式进行向上转型只可能对公共基类进行（而不是对受保护或私有基类）。当以这种方式对对象的指针进行泛化时，虚析构函数对于通过动态（即运行时）绑定确定正确的析构函数起始点至关重要，而不是静态绑定。请记住，静态绑定会根据指针的类型选择起始析构函数，而不是对象实际的类型。一个很好的经验法则是，如果一个类有一个或多个虚函数，请确保你也有一个虚析构函数。

## 移动复制构造函数

一个`this`。然后我们必须将源对象的指针置空，以便这两个实例不共享动态分配的数据成员。实质上，我们已经移动了（内存中的）指针数据成员。

那么非指针数据成员呢？这些数据成员的内存将像往常一样被复制。非指针数据成员的内存和指针本身的内存（而不是指针指向的内存）仍然驻留在源实例中。因此，我们能做的最好的事情就是为源对象的指针指定一个空值，并在非指针数据成员中放置一个`0`（或类似的）值，以指示这些成员不再相关。

我们将使用 C++标准库中的`move()`函数来指示移动复制构造函数如下：

```cpp
Person p1("Alexa", "Gutierrez", 'R', "Ms.");
Person p2(move(p1));  // move copy constructor
Person p3 = move(p2); // also the move copy constructor
```

此外，对于通过继承相关的类，我们还将在派生类构造函数的成员初始化列表中使用`move()`。这将指定基类移动复制构造函数来帮助初始化子对象。

## 移动赋值运算符

**移动赋值运算符**与重载的赋值运算符非常相似，但其目标是再次通过*移动*源对象的动态分配数据来节省内存（而不是执行深度赋值）。与重载的赋值运算符一样，我们将测试自我赋值，然后从（已存在的）目标对象中删除任何先前动态分配的数据成员。然后，我们将简单地将源对象中的指针数据成员复制到目标对象中的指针数据成员。我们还将将源对象中的指针置空，以便这两个实例不共享这些动态分配的数据成员。

此外，就像移动复制构造函数一样，非指针数据成员将简单地从源对象复制到目标对象，并在源对象中用`0`值替换以指示不使用。

我们将再次使用`move()`函数如下：

```cpp
Person p3("Alexa", "Gutierrez", 'R', "Ms.");
Person p5("Xander", "LeBrun", 'R', "Dr.");
p5 = move(p3);  // move assignment; replaces p5
```

此外，对于通过继承相关的类，我们可以再次指定派生类的移动赋值运算符将调用基类的移动赋值运算符来帮助完成任务。

## 将规范类形式的组件结合在一起

让我们看一个采用规范类形式的一对类的例子。我们将从我们的`Person`类开始。这个例子可以在我们的 GitHub 上找到一个完整的程序：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15/Chp15-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15/Chp15-Ex1.cpp)

```cpp
class Person
{
private:    // Assume all usual data members exist
protected:  // Assume usual protected member functions exist 
public:
    Person();                // default constructor
    // Assume other usual constructors exist  
    Person(const Person &);  // copy constructor
    Person(Person &&);       // move copy constructor
    virtual ~Person();       // virtual destructor
    // Assume usual access functions and virtual fns. exist 
    Person &operator=(const Person &);  // assignment operator
    Person &operator=(Person &&);  // move assignment operator
};
// copy constructor
Person::Person(const Person &pers)     
{  
    // Assume deep copy is implemented here  
}
// overloaded assignment operator
Person &Person::operator=(const Person &p)
{
    if (this != &p)  // check for self-assignment
    {
       // Delete existing Person ptr data members for 'this',
       // then re-allocate correct size and copy from source
    }
    return *this;  // allow for cascaded assignments
}
```

在先前的类定义中，我们注意到`Person`包含默认构造函数、复制构造函数、重载赋值运算符和虚析构函数。在这里，我们已经采用了正统的规范类形式作为一个模式，适用于可能有一天作为公共基类的类。还要注意，我们已经添加了移动复制构造函数和移动赋值运算符的原型，以进一步采用扩展的规范类形式。

移动复制构造函数`Person(Person &&);`和移动赋值运算符`Person &operator=(Person &&);`的原型包含类型为`Person &&`的参数。这些是`Person &`的例子，将绑定到原始复制构造函数和重载赋值运算符，而 r 值引用参数将绑定到适用的移动方法。

现在让我们看一下有助于`Person`扩展规范类形式的方法定义 - 移动复制构造函数和移动赋值运算符：

```cpp
// move copy constructor
Person::Person(const Person &&pers)   
{   // overtake source object's dynamically allocated memory
    // and null-out source object's pointers to that memory
    firstName = pers.firstName;
    pers.firstName = 0;
    lastName = pers.lastName;
    pers.lastName = 0;
    middleInitial = pers.middleInitial;
    pers.middleInitial = '\0'; // null char indicates non-use
    title = pers.title;
    pers.title = 0;
}
// move overloaded assignment operator
Person &Person::operator=(const Person &p)
{ 
    if (this != &p)       // check for self-assignment
    {
        delete firstName;  // or call ~Person(); (unusual)
        delete lastName;   // Delete existing object's
        delete title;      // allocated data members
        // overtake source object's dynamically alloc memory
        // and null source object's pointers to that memory
        firstName = p.firstName;
        p.firstName = 0;
        lastName = p.lastName;
        p.lastName = 0;
        middleInitial = p.middleInitial;
        p.middleInitial = '\0'; // null char indicates non-use
        title = p.title;
        p.title = 0;   
    }
    return *this;  // allow for cascaded assignments  
}
```

请注意，在前面的移动复制构造函数中，我们通过简单的指针赋值（而不是内存分配，如我们在深复制构造函数中所使用的）接管源对象的动态分配内存。然后我们在源对象的指针数据成员中放置一个空值。对于非指针数据成员，我们只是将值从源对象复制到目标对象，并在源对象中放置一个零值（例如`p.middleInitial`的`'\0'`）以表示其进一步的非使用。

在移动赋值运算符中，我们检查自我赋值，然后采用相同的方案，仅仅通过简单的指针赋值将动态分配的内存从源对象移动到目标对象。我们也复制简单的数据成员，并且当然用空指针或零值替换源对象数据值，以表示进一步的非使用。`*this`的返回值允许级联赋值。

现在，让我们看看派生类`Student`如何在利用其基类组件来辅助实现选定的成语方法时，同时使用正统和扩展的规范类形式：

```cpp
class Student: public Person
{
private:  // Assume usual data members exist
public:
    Student();                 // default constructor
    // Assume other usual constructors exist  
    Student(const Student &);  // copy constructor
    Student(Student &&);       // move copy constructor
    virtual ~Student();        // virtual destructor
    // Assume usual access functions exist 
    // as well as virtual overrides and additional methods
    Student &operator=(const Student &);  // assignment op.
    Student &operator=(Student &&);  // move assignment op.
};
// copy constructor
Student::Student(const Student &s): Person(s)
{   // Use member init. list to specify base copy constructor
    // to initialize base sub-object
    // Assume deep copy for Student is implemented here  
}
// Overloaded assignment operator
Student &Student::operator=(const Student &s)
{
   if (this != &s)   // check for self-assignment
   {
       Person::operator=(s);  // call base class assignment op
       // delete existing Student ptr data members for 'this'
       // then reallocate correct size and copy from source
   }
}
```

在先前的类定义中，我们再次看到`Student`包含默认构造函数、复制构造函数、重载赋值运算符和虚析构函数，以完成正统的规范类形式。

然而，请注意，在`Student`复制构造函数中，我们通过成员初始化列表指定了`Person`复制构造函数的使用。同样，在`Student`重载赋值运算符中，一旦我们检查自我赋值，我们调用`Person`中的重载赋值运算符来帮助我们使用`Person::operator=(s);`完成任务。

现在让我们看一下有助于`Student`扩展规范类形式的方法定义 - 移动复制构造函数和移动赋值运算符：

```cpp
// move copy constructor
Student::Student(Student &&ps): Person(move(ps))   
{   // Use member init. list to specify base move copy 
    // constructor to initialize base sub-object
    gpa = ps.gpa;
    ps.gpa = 0.0;
    currentCourse = ps.currentCourse;
    ps.currentCourse = 0;
    studentId = ps.studentId;  
    ps.studentId = 0;
}
// move assignment operator
Student &Student::operator=(Student &&s)
{
   // make sure we're not assigning an object to itself
   if (this != &s)
   {
      Person::operator=(move(s));  // call base move oper=
      delete currentCourse;  // delete existing data members
      delete studentId;
      gpa = s.gpa;  
      s.gpa = 0.0;
      currentCourse = s.currentCourse;
      s.currentCourse = 0;
      studentId = s.studentId;
      s.studentId = 0;
   }
   return *this;  // allow for cascaded assignments
}
```

请注意，在先前列出的`Student`移动复制构造函数中，我们在成员初始化列表中指定了基类的移动复制构造函数的使用。`Student`移动复制构造函数的其余部分与`Person`基类中的类似。

同样，让我们注意，在`Student`移动赋值运算符中，调用基类的移动`operator=`与`Person::operator=(move(s);`。这个方法的其余部分与基类中的类似。

一个很好的经验法则是，大多数非平凡的类应该至少使用正统的规范类形式。当然，也有例外。例如，一个只用作受保护或私有基类的类不需要具有虚析构函数，因为派生类实例不能通过非公共继承边界向上转型。同样，如果我们有充分的理由不希望复制或禁止赋值，我们可以在这些方法的扩展签名中使用`= delete`规范来禁止复制或赋值。

尽管如此，规范类形式将为采用这种习惯的类增加健壮性。采用这种习惯的类在初始化、赋值和参数传递方面的统一性将受到程序员的重视。

让我们继续来看看与规范类形式相辅相成的一个概念，即健壮性。

## 确保类是健壮的

C++的一个重要特性是能够构建用于广泛重用的类库。无论我们希望实现这个目标，还是只是希望为我们自己组织的使用提供可靠的代码，重要的是我们的代码是健壮的。一个健壮的类将经过充分测试，应该遵循规范的类形式（除了在受保护和私有基类中需要虚析构函数），并且是可移植的（或包含在特定平台的库中）。任何候选重用的类，或者将在任何专业环境中使用的类，绝对必须是健壮的。

健壮的类必须确保给定类的所有实例都完全构造。**完全构造的对象**是指所有数据成员都得到适当初始化的对象。必须验证给定类的所有构造函数（包括复制构造函数）以初始化所有数据成员。应检查加载数据成员的值是否适合范围。记住，未初始化的数据成员是潜在的灾难！应该在给定构造函数未能正确完成或数据成员的初始值不合适的情况下采取预防措施。

可以使用各种技术来验证完全构造的对象。一种基本的技术是在每个类中嵌入一个状态数据成员（或派生或嵌入一个状态祖先/成员）。在成员初始化列表中将状态成员设置为`0`，并在构造函数的最后一行将其设置为`1`。在实例化后探测这个值。这种方法的巨大缺陷是用户肯定会忘记探测*完全构造*的成功标志。

一个更好的技术是利用异常处理。在每个构造函数内嵌异常处理是理想的。如果数据成员未在合适范围内初始化，首先尝试重新输入它们的值，或者例如打开备用数据库进行输入。作为最后手段，您可以抛出异常来报告*未完全构造的对象*。我们将在本章后面更仔细地研究关于测试的异常处理。

与此同时，让我们继续使用一种技术来严格测试我们的类和组件——创建驱动程序来测试类。

# 创建驱动程序来测试类

在*第五章*中，*详细探讨类*，我们简要讨论了将代码分解为源文件和头文件的方法。让我们简要回顾一下。通常，头文件将以类的名称命名（如`Student.h`），并包含类定义，以及任何内联成员函数定义。通过将内联函数放在头文件中，它们将在其实现更改时被正确地重新扩展（因为头文件随后包含在每个源文件中，与该头文件创建了依赖关系）。

每个类的方法实现将被放置在相应的源代码文件中（比如`Student.cpp`），它将包括它所基于的头文件（即`#include "Student.h"`）。请注意，双引号意味着这个头文件在我们当前的工作目录中；我们也可以指定一个路径来找到头文件。相比之下，C++库使用的尖括号告诉预处理器在编译器预先指定的目录中查找。另外，请注意，每个派生类的头文件将包括其基类的头文件（以便它可以看到成员函数的原型）。

考虑到这种头文件和源代码文件结构，我们现在可以创建一个驱动程序来测试每个单独的类或每组紧密相关的类（例如通过关联或聚合相关的类）。通过继承相关的类可以在它们自己的单独的驱动程序文件中进行测试。每个驱动程序文件可以被命名为反映正在测试的类的名称，比如`StudentDriver.cpp`。驱动程序文件将包括正在测试的类的相关头文件。当然，所涉及类的源文件将作为编译过程的一部分被编译和链接到驱动程序文件中。

驱动程序文件可以简单地包含一个`main()`函数，作为一个测试平台来实例化相关的类，并作为测试每个成员函数的范围。驱动程序将测试默认实例化、典型实例化、复制构造、对象之间的赋值，以及类中的每个附加方法。如果存在虚析构函数或其他虚函数，我们应该实例化派生类实例（在派生类的驱动程序中），将这些实例向上转型为基类指针进行存储，然后调用虚函数以验证发生了正确的行为。在虚析构函数的情况下，我们可以通过删除动态分配的实例（或等待栈实例超出范围）并通过调试器逐步验证一切是否符合预期来跟踪销毁顺序的入口点。

我们还可以测试对象是否完全构造；我们很快将在这个主题上看到更多。

假设我们有我们通常的`Person`和`Student`类层次结构，这里有一个简单的驱动程序来测试`Student`类。这个驱动程序可以在我们的 GitHub 存储库中找到。为了创建一个完整的程序，您还需要编译和链接在同一目录中找到的`Student.cpp`和`Person.cpp`文件。这是驱动程序的 GitHub URL：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15/Chp15-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter15/Chp15-Ex2.cpp)

```cpp
#include "Person.h"    // include relevant class header files
#include "Student.h"
using namespace std;
const int MAX = 3;
int main()   // Driver to test Student class. Stored in above
{            // filename for chapter example consistency 
    // Test all means for instantiation, including copy const.
    Student s0; // Default construction
    // alternate constructor
    Student s1("Jo", "Li", 'H', "Ms.", 3.7, "C++", "UD1234");
    Student s2("Sam", "Lo", 'A', "Mr.", 3.5, "C++", "UD2245");
    // These initializations implicitly invoke copy const.
    Student s3(s1);
    Student s4 = s2;   // This is also initialization
    // Test the assignment operator
    Student s5("Ren", "Ze", 'A', "Dr.", 3.8, "C++", "BU5563");
    Student s6;
    s6 = s5;  // this is an assignment, not initialization
    // Test each public method. A sample is shown here
    s1.Print();  // Be sure to test each method! 

    // Generalize derived instances as base types 
    // Do the polymorphic operations work as expected?
    Person *people[MAX];
    // base instance for comparison
    people[0] = new Person("Juliet", "Martinez", 'M', "Ms.");
    // derived instances, generalized with base class ptrs.   
    people[1] = new Student("Zack", "Moon", 'R', "Dr.", 3.8,
                            "C++", "UMD1234");  
    people[2] = new Student("Gabby", "Doone", 'A', "Dr.", 3.9,
                            "C++", "GWU4321");
    for (int i = 0; i < MAX; i++)
    {
       people[i]->IsA();
       cout << "  ";
       people[i]->Print();
    }
    // Test destruction sequence (dynam. allocated instances)
    for (int i = 0; i < MAX; i++)
       delete people[i];   // engage virtual dest. sequence
    return 0;
}
```

简要回顾前面的程序片段，我们可以看到我们已经测试了每种实例化方式，包括复制构造函数。我们还测试了赋值运算符，验证了每个成员函数的工作（示例方法显示了），并验证了虚函数（包括虚析构函数）按预期工作。

既然我们已经看到了一个基本的驱动程序测试我们的类，让我们考虑一些额外的指标，当测试通过继承、关联或聚合相关的类时可以使用。

# 测试相关类

对于面向对象的程序，仅仅测试单个类的完整性和健壮性是不够的，尽管这些是很好的起点。完整性不仅包括遵循规范的类形式，还包括确保数据成员具有安全的访问方式，使用适当的访问方法（在不修改实例时标记为`const`）。完整性还验证了按照面向对象设计规范实现了所需的接口。

健壮性要求我们验证所有上述方法是否在适当的驱动程序中进行了测试，评估其平台独立性，并验证每种实例化方式是否导致完全构造的对象。我们可以通过对实例的数据成员进行阈值测试来增强这种类型的测试，注意当抛出异常时。完整性和健壮性，尽管看似全面，实际上是 OO 组件测试最直接的手段。

测试相关类之间交互的一种更具挑战性的手段是测试聚合和关联之间的交互。

## 通过继承、关联或聚合相关的类进行测试

通过各种对象关系相关的类需要各种额外的组件测试手段。具有各种关系的对象之间的相互影响可能会影响应用程序中给定实例的生命周期内的状态变化。这种类型的测试将需要最详细的努力。我们会发现场景对于帮助我们捕捉相关对象之间的常规交互是有用的，从而导致更全面的测试相互交互的类的方式。

让我们首先考虑如何测试通过继承相关的类。

### 添加测试继承的策略

通过公共继承相关的类需要验证虚函数。例如，所有预期的派生类方法是否已被覆盖？记住，如果基类行为在派生类级别仍然被认为是适当的，那么派生类不需要覆盖其基类中指定的所有虚函数。将需要将实现与设计进行比较，以确保我们已经用适当的方法覆盖了所有必需的多态操作。

当然，虚函数的绑定是在运行时完成的（即动态绑定）。重要的是创建派生类实例并使用基类指针存储它们，以便可以应用多态操作。然后我们需要验证派生类的行为是否突出。如果没有，也许我们会发现自己处于一个意外的函数隐藏情况，或者基类操作没有像预期的那样标记为虚拟（请记住，虚拟和覆盖关键字在派生类级别，虽然很好并且推荐，但是是可选的，不会影响动态行为）。

尽管通过继承相关的类具有独特的测试策略，但要记住实例化将创建一个单一对象，即基类或派生类类型的对象。当我们实例化这样的类型时，我们有一个实例，而不是一对共同工作的实例。派生类仅具有基类子对象，该子对象是其自身的一部分。让我们考虑一下这与关联对象或聚合物的比较，它们可以是单独的对象（关联），可能与其伴侣进行交互。

### 添加测试聚合和关联的策略

通过关联或聚合相关的类可能是多个实例之间的通信，并且彼此引起状态变化。这显然比继承的对象关系更复杂。

通过聚合相关的类通常比通过关联相关的类更容易测试。考虑到最常见的聚合形式（组合），内嵌（内部）对象是外部（整体）对象的一部分。当实例化外部对象时，我们得到内部对象嵌入在“整体”中的内存。与包含基类子对象的派生类实例的内存布局相比，内存布局并没有非常不同（除了可能的排序）。在每种情况下，我们仍然处理单个实例（即使它有嵌入的“部分”）。然而，与测试进行比较的重点是，应用于“整体”的操作通常被委托给“部分”或组件。我们将严格需要测试整体上的操作，以确保它们将必要的信息委托给每个部分。

通过一般聚合的较少使用的形式相关的类（其中整体包含指向部分的指针，而不是典型的组合的嵌入对象实现）与关联有类似的问题，因为实现是相似的。考虑到这一点，让我们来看看与相关对象有关的测试问题。

通过关联相关的类通常是独立存在的对象，在应用程序的某个时刻彼此创建了链接。在应用于一个对象上的操作可能会导致关联对象的变化。例如，让我们考虑一个“学生”和一个“课程”。两者可能独立存在，然后在应用程序的某个时刻，“学生”可能通过`Student::AddCourse()`添加一个“课程”。通过这样做，不仅特定的“学生”实例现在包含到特定的“课程”实例的链接中，而且`Student::AddCourse()`操作已经导致了“课程”类的变化。特定的“学生”实例现在是特定“课程”实例名单的一部分。在任何时候，“课程”可能被取消，从而影响到所有已经在该“课程”中注册的“学生”实例。这些变化反映了每个关联对象可能存在的状态。例如，“学生”可能处于“当前注册”或“退出”“课程”的状态。有很多可能性。我们如何测试它们？

### 添加场景以帮助测试对象关系

在面向对象分析中，场景的概念被提出作为创建 OO 设计和测试的手段。**场景**是对应用程序中可能发生的一系列事件的描述性步行。场景将展示类以及它们如何在特定情况下相互作用。许多相关场景可以被收集到 OO 概念的**用例**中。在 OO 分析和设计阶段，场景有助于确定应用程序中可能存在的类，以及每个类可能具有的操作和关系。在测试中，场景可以被重复使用，形成测试各种对象关系的驱动程序创建的基础。考虑到这一点，可以开发一系列驱动程序来测试多种场景（即用例）。这种建模方式将更彻底地为相关对象提供一个测试基础，而不仅仅是最初的简单测试完整性和健壮性的手段。

与任何类型的相关类之间的另一个关注领域是版本控制。例如，如果基类定义或默认行为发生了变化会发生什么？这将如何影响派生类？这将如何影响相关对象？随着每次变化，我们不可避免地需要重新审视所有相关类的组件测试。

接下来，让我们考虑异常处理机制如何影响 OO 组件测试。

# 测试异常处理机制

现在我们可以创建驱动程序来测试每个类（或一组相关类），我们将想要了解我们代码中哪些方法可能会抛出异常。对于这些情况，我们将希望在驱动程序中添加 try 块，以确保我们知道如何处理每个可能抛出的异常。在这样做之前，我们应该问自己，在开发过程中我们的代码是否包含了足够的异常处理？例如，考虑实例化，我们的构造函数是否检查对象是否完全构造？如果没有，它们会抛出异常吗？如果答案是否定的，我们的类可能不像我们预期的那样健壮。

让我们考虑将异常处理嵌入到构造函数中，以及我们如何构建一个驱动程序来测试所有可能的实例化方式。

## 将异常处理嵌入到构造函数中以创建健壮的类

我们可能还记得我们最近的*第十一章*，*处理异常*，我们可以创建自己的异常类，从 C++标准库`exception`类派生而来。假设我们已经创建了这样一个类，即`ConstructionException`。如果在构造函数的任何时候我们无法正确初始化给定实例以提供一个完全构造的对象，我们可以从任何构造函数中抛出`ConstructionException`。潜在抛出`ConstructionException`的含义是我们现在应该在 try 块中封闭实例化，并添加匹配的 catch 块来预期可能抛出的`ConstructionException`。然而，请记住，在 try 块范围内声明的实例只在 try-catch 配对内部有效。

好消息是，如果一个对象没有完成构造（也就是说，在构造函数完成之前抛出异常），那么这个对象在技术上就不存在。如果一个对象在技术上不存在，就不需要清理部分实例化的对象。然而，我们需要考虑如果我们预期的实例没有完全构造，这对我们的应用意味着什么。这将如何改变我们代码中的进展？测试的一部分是确保我们已经考虑了我们的代码可能被使用的所有方式，并相应地进行防护！

重要的是要注意，引入`try`和`catch`块可能会改变我们的程序流程，包括这种类型的测试对我们的驱动程序是至关重要的。我们可能会寻找考虑`try`和`catch`块的场景，当我们进行测试时。

我们现在已经看到了如何增强我们的测试驱动程序以适应可能抛出异常的类。在本章中，我们还讨论了在我们的驱动程序中添加场景，以帮助跟踪具有关系的对象之间的状态，当然，我们还讨论了可以遵循的简单类习惯，以便为成功做好准备。在继续下一章之前，让我们简要回顾一下这些概念。

# 总结

在本章中，我们通过检查各种 OO 类和组件测试实践和策略，增强了成为更好的 C++程序员的能力。我们的主要目标是确保我们的代码是健壮的，经过充分测试，并且可以无错误地部署到我们的各个组织中。

我们已经考虑了编程习惯，比如遵循规范的类形式，以确保我们的类是完整的，并且在构造/销毁、赋值以及在参数传递和作为函数返回值中的使用方面具有预期的行为。我们已经讨论了创建健壮类的含义 - 一个遵循规范的类形式，也经过充分测试，独立于平台，并且针对完全构造的对象进行了测试。

我们还探讨了如何创建驱动程序来测试单个类或一组相关类。我们已经建立了一个测试单个类的项目清单。我们更深入地研究了对象关系，以了解彼此交互的对象需要更复杂的测试。也就是说，当对象从一种状态转移到另一种状态时，它们可能会受到相关对象的影响，这可能会进一步改变它们的进展方向。我们已经添加了使用场景作为我们的驱动程序的测试用例，以更好地捕捉实例可能在应用程序中移动的动态状态。

最后，我们已经看了一下异常处理机制如何影响我们测试代码，增强我们的驱动程序以考虑 try 和 catch 块在我们的应用程序中可能操纵的控制流。

我们现在准备继续我们书的下一部分，C++中的设计模式和习惯用法。我们将从*第十六章*开始，*使用观察者模式*。在剩下的章节中，我们将了解如何应用流行的设计模式，在我们的编码中使用它们。这些技能将使我们成为更好的程序员。让我们继续前进！

# 问题

1.  考虑一对包含对象关系的类，来自你以前的练习（提示：公共继承比关联更容易考虑）。

a. 你的类遵循规范的类形式吗？是正统的还是扩展的？为什么？如果不是，而应该是，修改类以遵循这种习惯用法。

b. 你认为你的类健壮吗？为什么？为什么不？

1.  创建一个（或两个）驱动程序来测试你的一对类。

a. 确保测试通常的项目清单（构造、赋值、销毁、公共接口、向上转型（如果适用）和使用虚函数）。

b.（可选）如果您选择了两个与关联相关的类，请创建一个单独的驱动程序，以详细描述这两个类的交互的典型场景。

c. 确保在您的一个测试驱动程序中包括异常处理的测试。

1.  创建一个`ConstructionException`类（从 C++标准库`exception`派生）。在样本类的构造函数中嵌入检查，以在必要时抛出`ConstructionException`。确保将此类的所有实例化形式都包含在适当的`try`和`catch`块配对中。
