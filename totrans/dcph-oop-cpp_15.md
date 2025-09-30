# 15

# 测试类和组件

本章将继续我们的追求，通过探索测试构成我们面向对象程序类和组件的方法，来增加你的 C++编程知识库，而不仅仅是面向对象的概念。我们将探讨各种策略，以确保我们编写的代码经过充分测试且稳健。

本章展示了如何通过测试单个类以及测试协同工作的各种组件来测试你的面向对象程序。

在本章中，我们将涵盖以下主要主题：

+   理解典型类形式和创建稳健类

+   创建驱动程序以测试类

+   测试通过继承、关联或聚合相关的类

+   测试异常处理机制

在本章结束时，你将拥有各种技术，以确保你的代码在生产前经过充分测试。具备持续生产稳健代码的技能将帮助你成为一个更有益的程序员。

通过研究面向对象测试的各种技术，让我们增加我们的 C++技能集。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter15`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter15)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter15`子目录中找到，文件名为`Chp15-Ex1.cpp`。

本章的 CiA 视频可在以下网址查看：[`bit.ly/3AxyLFH`](https://bit.ly/3AxyLFH)。

# 思考面向对象测试

在任何代码部署之前进行软件测试至关重要。测试面向对象软件需要不同于其他类型软件的技术。因为面向对象软件包含类之间的关系，我们必须了解如何测试可能存在于类之间的依赖关系和关系。此外，每个对象可能根据应用于每个实例的操作顺序以及与相关对象的特定交互（例如，通过关联）而处于不同的状态。与过程式应用程序相比，面向对象应用程序的整体控制流程要复杂得多，因为应用于特定对象的操作组合和顺序以及来自相关对象的影响众多。

尽管如此，我们仍然可以应用一些指标和流程来测试面向对象（OO）软件。这些指标和流程包括理解我们可以应用于类指定的惯用语句和模式，以及创建驱动程序来独立测试类以及它们与其他类之间的关系。这些流程还可以包括创建场景，以提供对象可能经历的事件或状态的预期序列。对象之间的关系，如继承、关联和聚合，在测试中变得非常重要；相关对象可以影响现有对象的状态。

让我们通过理解一个简单的模式开始我们的面向对象软件测试之旅，这个模式我们经常可以应用于我们开发的类。这个惯用语句将确保一个类可能是完整的，没有意外的行为。我们将从规范类形式开始。

# 理解规范类形式

对于 C++中的许多类，遵循类指定的模式以确保新类包含所需的所有组件是合理的。**规范类形式**是对类的一个稳健的规范，它使类实例能够在初始化、赋值、参数传递以及函数返回值的使用等方面提供统一的行为（类似于标准数据类型）。规范类形式适用于大多数旨在实例化或将成为新派生类公共基类的类。旨在作为私有或保护基类的类（即使它们可能被实例化）可能不会遵循这个惯用语句的所有部分。

一个遵循**正统**规范形式的类将包括以下内容：

+   一个默认构造函数（或一个`=default`原型，以显式允许此接口）

+   一个复制构造函数

+   一个重载的赋值运算符

+   一个虚析构函数

尽管上述任何组件都可以使用`=default`原型来显式利用默认的系统提供的实现，但现代的偏好正在远离这种做法（因为这些原型通常是多余的）。例外是默认构造函数，如果没有使用`=default`，在其他构造函数存在的情况下，你将无法获得其接口。

一个遵循**扩展**规范形式的类将还包括以下内容：

+   一个*移动*复制构造函数

+   一个*移动*赋值运算符

让我们接下来在下一小节中查看规范类形式的每个组成部分。

## 默认构造函数

对默认构造函数原型的`=default`；这在利用类内初始化时特别有用。

此外，如果没有在成员初始化列表中指定其他基类构造函数，将调用给定类的基类的默认构造函数。如果一个基类没有这样的默认构造函数（并且没有提供，因为存在具有不同签名的构造函数），对基类构造函数的隐式调用将被标记为错误。

让我们也考虑多继承的情况，其中出现菱形层次结构，并且使用虚拟基类来消除最派生类实例中最基本类子对象的重复。在这种情况下，除非在负责创建菱形结构的派生类的成员初始化列表中另有指定，否则将调用现在*共享*的基类子对象的默认构造函数。即使在中级别的成员初始化列表中指定了非默认构造函数，这种情况也会发生；记住，当中级别指定一个可能共享的虚拟基类时，这些指定将被忽略。

## 拷贝构造函数

对于包含指针数据成员的所有对象，**拷贝构造函数**通常至关重要。除非程序员提供了拷贝构造函数，否则在应用程序需要时将链接系统提供的拷贝构造函数。系统提供的拷贝构造函数执行所有数据成员的成员级（浅）拷贝。这意味着类的多个实例可能包含指向*共享*内存块的指针，这些内存块代表应该个性化的数据。除非有意资源共享，否则新实例化的对象中的原始指针数据成员将想要分配自己的内存并将数据值从源对象复制到这个内存中。此外，请记住在派生类的拷贝构造函数中使用成员初始化列表来指定基类的拷贝构造函数以复制基类数据成员。当然，以深度方式复制基类子对象至关重要；此外，基类数据成员必然是私有的，因此在派生类的成员初始化列表中选择基类拷贝构造函数非常重要。

通过指定拷贝构造函数，我们也有助于提供从函数传递（或返回）值时创建对象的预期方式。在这些情况下确保深度拷贝至关重要。用户可能会认为这些拷贝是*按值*进行的，但如果它们的指针数据成员实际上与源实例共享，那么这并不是真正按值传递（或返回）对象。

## 重载赋值运算符

一个**重载的赋值运算符**，就像拷贝构造函数一样，对于所有包含指针数据成员的对象通常也是至关重要的。系统提供的赋值运算符的默认行为是从源对象到目标对象的浅拷贝。同样，当数据成员是原始指针时，除非两个对象想要共享堆数据成员的资源，否则强烈建议重载赋值运算符。目标对象中分配的空间应等于任何此类指针数据成员的数据成员大小。然后应该将每个指针数据成员的内容（数据）从源对象复制到目标对象。

此外，请记住，重载的赋值运算符不是*继承*的；每个类都负责编写自己的版本。这很有意义，因为派生类不可避免地有比其基类赋值运算符函数更多的数据成员要复制。然而，当在派生类中重载赋值运算符时，请记住调用基类的赋值运算符以执行继承基类成员的深度赋值（这些成员可能是私有的且无法访问）。

## 虚析构函数

A `=default`).

## 移动拷贝构造函数

一个`this`。然后我们必须将源对象对这些数据成员的指针设置为空，这样两个实例就不会*共享*动态分配的数据成员。本质上，我们已经移动了（这些指针的内存）。

那么非指针数据成员怎么办？这些数据成员的内存将按常规复制。非指针数据成员的内存以及指针本身的内存（不是那些指针指向的内存）仍然位于源实例中。因此，我们能做的最好的事情是为源对象的指针指定一个空值（`nullptr`），并在非指针数据成员中放置一个`0`（或类似值）以指示这些成员不再相关。

我们将使用 C++标准库中找到的`move()`函数，如下指示移动拷贝构造函数：

```cpp
Person p1("Alexa", "Gutierrez", 'R', "Ms.");
Person p2(move(p1));  // move copy constructor
Person p3 = move(p2); // also the move copy constructor
```

此外，对于通过继承相关联的类，我们也会在派生类移动拷贝构造函数的成员初始化列表中使用`move()`。这将指定基类的移动拷贝构造函数以帮助初始化子对象。

## 移动赋值运算符

**移动赋值运算符**与重载的赋值运算符非常相似，对于所有包含指针数据成员的对象来说通常至关重要。然而，目标是再次通过 *移动* 源对象的动态分配数据到目标对象（而不是执行深拷贝）来节省内存。与重载的赋值运算符一样，我们将测试自赋值，然后从（现有的）目标对象中删除任何先前动态分配的数据成员。然而，然后我们将简单地从源对象复制指针数据成员到目标对象。我们还将使源对象中的指针为空，这样两个实例就不会共享这些动态分配的数据成员。

类似于移动复制构造函数，非指针数据成员将简单地从源对象复制到目标对象，并在源对象中用 `nullptr` 值替换，以指示未使用。

我们将再次使用 `move()` 函数，如下所示：

```cpp
Person p3("Alexa", "Gutierrez", 'R', "Ms.");
Person p5("Xander", "LeBrun", 'R', "Dr.");
p5 = move(p3);  // move assignment; replaces p5
```

此外，对于通过继承相关联的类，我们还可以指定派生类的移动赋值运算符将调用基类的移动赋值运算符以帮助完成任务。

## 将规范类形式的组件组合在一起

让我们看看一对采用规范类形式的类的示例。我们将从我们的 `Person` 类开始。这个例子可以作为完整的程序在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter15/Chp15-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter15/Chp15-Ex1.cpp)

```cpp
class Person
{
private:    // Note slightly modified data members
    string firstName, lastName;
    char middleInitial = '\0';   // in-class initialization
    // pointer data member to demo deep copy and operator =
    char *title = nullptr;      // in-class initialization
protected: // Assume usual protected member functions exist 
public:
    Person() = default;      // default constructor
    // Assume other usual constructors exist  
    Person(const Person &);  // copy constructor
    Person(Person &&);       // move copy constructor
    virtual ~Person() { delete [] title }; // virtual dest.
    // Assume usual access functions and virtual fns. exist 
    Person &operator=(const Person &);  // assignment op.
    Person &operator=(Person &&);  // move assignment op.
};
// copy constructor
Person::Person(const Person &p): firstName(p.firstName),
      lastName(p.lastName), middleInitial(p.middleInitial)
{ 
    // Perform a deep copy for the pointer data member 
    // That is, allocate memory, then copy contents
    title = new char [strlen(p.title) + 1];
    strcpy(title, p.title);
}
// overloaded assignment operator
Person &Person::operator=(const Person &p)
{
    if (this != &p)  // check for self-assignment
    {
       // delete existing Person ptr data mbrs. for 'this'
       delete [] title;
       // Now re-allocate correct size and copy from source
       // Non-pointer data members are simply copied from
       // source to destination object
       firstName = p.firstName; // assignment btwn. strings
       lastName = p.lastName;
       middleInitial = p.middleInitial;
       title = new char [strlen(p.title) + 1]; // mem alloc 
       strcpy(title, p.title);
    }
    return *this;  // allow for cascaded assignments
}
```

在前面的类定义中，我们注意到 `Person` 包含一个默认构造函数、复制构造函数、重载的赋值运算符和一个虚析构函数。在这里，我们采用了传统的规范类形式作为可能有一天会作为公共基类的类的模式。同时请注意，我们添加了移动复制构造函数和移动赋值运算符的原型，以进一步采用扩展规范类形式。

移动复制构造函数 `Person(Person &&);` 和移动赋值运算符 `Person &operator=(Person &&);` 的原型包含类型为 `Person &&` 的参数。这些都是 `Person &` 的例子，将绑定到原始的复制构造函数和重载的赋值运算符，而右值引用参数将绑定到相应的移动方法。

现在我们来看一下对扩展规范类形式有所贡献的方法的定义——`Person` 的移动构造函数和移动赋值运算符：

```cpp
// move copy constructor
Person::Person(Person &&p): firstName(p.firstName), 
    lastName(p.lastName), middleInitial(p.middleInitial),
    title(p.title)  // dest ptr takes over src ptr's memory
{   
    // Overtake source object's dynamically alloc. memory
    // or use simple assignments (non-ptr data members) 
    // to copy source object's members in member init. list 
    // Then null-out source object's ptrs to that memory
    // Clear source obj's string mbrs, or set w null char
    p.firstName.clear(); // set src object to empty string
    p.lastName.clear();
    p.middleInitial = '\0'; // null char indicates non-use
    p.title = nullptr; // null out src ptr; don't share mem
}
// move overloaded assignment operator
Person &Person::operator=(Person &&p)
{ 
    if (this != &p)       // check for self-assignment
    {
        // delete destination object's ptr data members
        delete [] title;      
        // for ptr mbrs: overtake src obj's dynam alloc mem
        // and null source object's pointers to that memory
        // for non-ptr mbrs, a simple assignment suffices
        // followed by clearing source data member
        firstName = p.firstName;  // string assignment
        p.firstName.clear();   // clear source data member 
        lastName = p.lastName;
        p.lastName.clear();
        middleInitial = p.middleInitial; // simple =
        p.middleInitial = '\0'; // null char shows non-use
        title = p.title; // ptr assignment to take over mem
        p.title = nullptr;   // null out src pointer
    }
    return *this;  // allow for cascaded assignments  
}
```

注意，在前面的移动构造函数中，对于指针类型的成员变量，我们通过在成员初始化列表中使用简单的指针赋值来接管源对象的动态分配的内存（而不是像在深度复制构造函数中那样使用内存分配）。然后我们在构造函数的主体中将源对象的指针数据成员设置为`nullptr`值。对于非指针数据成员，我们简单地从源对象复制值到目标对象，并在源对象中放置一个零值或空值（例如，对于`p.middleInitial`使用`'\0'`或使用`clear()`对于`p.firstName`），以指示其进一步的非使用。

在移动赋值运算符中，我们检查自赋值，然后采用相同的方案，仅通过简单的指针赋值将动态分配的内存从源对象移动到目标对象。我们也复制简单的数据成员，当然，用空指针（`nullptr`）或零值替换源对象的数据值，以指示其进一步的非使用。`*this`的返回值允许级联赋值。

现在，让我们看看派生类`Student`如何利用其基类组件，同时采用正统和扩展规范类形式来实现选定的惯用方法：

```cpp
class Student: public Person
{
private:  
    float gpa = 0.0;        // in-class initialization
    string currentCourse;
    // one pointer data member to demo deep copy and op=
    const char *studentId = nullptr; // in-class init.
    static int numStudents; 
public:
    Student();                 // default constructor
    // Assume other usual constructors exist  
    Student(const Student &);  // copy constructor
    Student(Student &&);       // move copy constructor
    ~Student() override;       // virtual destructor
    // Assume usual access functions exist 
    // as well as virtual overrides and additional methods
    Student &operator=(const Student &);  // assignment op.
    Student &operator=(Student &&);  // move assignment op.
};
// See online code for default constructor implementation
// as well as implementation for other usual member fns.
// copy constructor
Student::Student(const Student &s): Person(s), 
                 gpa(s.gpa), currentCourse(s.currentCourse)
{   // Use member init. list to specify base copy 
    // constructor to initialize base sub-object
    // Also use mbr init list to set most derived data mbrs
    // Perform deep copy for Student ptr data members 
    // use temp - const data can't be directly modified 
    char *temp = new char [strlen(s.studentId) + 1];
    strcpy (temp, s.studentId);
    studentId = temp;
    numStudents++;
}
// Overloaded assignment operator
Student &Student::operator=(const Student &s)
{
   if (this != &s)   // check for self-assignment
   {   // call base class assignment operator
       Person::operator=(s); 
       // delete existing Student ptr data mbrs for 'this'
       delete [] studentId;
       // for ptr members, reallocate correct size and copy
       // from source; for non-ptr members, just use =
       gpa = s.gpa;  // simple assignment
       currentCourse = s.currentCourse;
       // deep copy of pointer data mbr (use a temp since
       // data is const and can't be directly modified)
       char *temp = new char [strlen(s.studentId) + 1];
       strcpy (temp, s.studentId);
       studentId = temp;
   }
   return *this;
}
```

在前面的类定义中，我们再次看到`Student`包含一个默认构造函数、一个复制构造函数、一个重载的赋值运算符和一个虚析构函数，以完成正统的规范类形式。

然而，请注意，在`Student`复制构造函数中，我们通过成员初始化列表指定了使用`Person`复制构造函数。同样，在`Student`重载的赋值运算符中，一旦我们检查到自赋值，我们就调用`Person`中的重载赋值运算符来帮助我们完成任务，使用`Person::operator=(s);`。

现在让我们看看对`Student`扩展规范类形式做出贡献的方法定义——移动复制构造函数和移动赋值运算符：

```cpp
// move copy constructor
Student::Student(Student &&s): Person(move(s)), gpa(s.gpa),
    currentCourse(s.currentCourse), 
    studentId(s.studentId) // take over src obj's resource 
{   
    // First, use mbr. init. list to specify base move copy 
    // constructor to initialize base sub-object. Then
    // overtake source object's dynamically allocated mem.
    // or use simple assignments (non-ptr data members) 
    // to copy source object's members in mbr. init. list.
    // Then null-out source object's ptrs to that memory or 
    // clear out source obj's string mbrs. in method body
    s.gpa = 0.0;     // then zero-out source object member
    s.currentCourse.clear();  // clear out source member
    s.studentId = nullptr; // null out src ptr data member
    numStudents++;  // it is a design choice whether or not 
    // to inc. counter; src obj is empty but still exists
}
// move assignment operator
Student &Student::operator=(Student &&s)
{
   // make sure we're not assigning an object to itself
   if (this != &s)
   {
      Person::operator=(move(s));  // call base move oper=
      delete [] studentId;  // delete existing ptr data mbr
      // for ptr data members, take over src objects memory
      // for non-ptr data members, simple assignment is ok
      gpa = s.gpa; // assignment of source to dest data mbr
      s.gpa = 0.0; // zero out source object data member
      currentCourse = s.currentCourse; // string assignment
      s.currentCourse.clear(); // set src to empty string
      studentId = s.studentId; // pointer assignment
      s.studentId = nullptr;   // null out src ptr data mbr
   }
   return *this;  // allow for cascaded assignments
}
```

注意，在之前列出的`Student`移动复制构造函数中，我们在成员初始化列表中指定了使用基类的移动复制构造函数。`Student`移动复制构造函数的其余部分与`Person`基类中的类似。

同样，让我们注意到在`Student`移动赋值运算符中，调用了基类的移动赋值运算符`Person::operator=(move(s));`。此方法的其他部分与基类中的类似。

一个好的经验法则是，大多数非平凡类应该至少利用正统的规范类形式。当然，也有一些例外。例如，一个仅作为受保护的或私有基类的类不需要有虚拟析构函数，因为派生类实例不能超出非公共继承边界向上转换。同样，如果我们有充分的理由不希望有副本或禁止赋值，我们可以在这些方法的扩展签名中使用`= delete`指定来禁止副本或赋值。

尽管如此，规范类形式将为采用这种语法的类增加健壮性。程序员将重视使用这种语法在初始化、赋值和参数传递方面的类之间的统一性。

让我们继续前进，看看与规范类形式互补的一个想法，那就是健壮性。

## 确保类是健壮的

C++的一个重要特性是能够构建用于广泛重用的类库。无论我们是否希望实现这一目标，或者只是希望为我们自己的组织提供可靠的代码，我们的代码都必须是健壮的。一个**健壮的类**应该经过良好的测试，应遵循规范类形式（除了在受保护的和私有的基类中需要虚拟析构函数外），并且应该是可移植的（或包含在特定平台的库中）。任何可能被重用或将在任何专业场合使用的类，绝对必须是健壮的。

一个健壮的类必须确保给定类的所有实例都是完全构建的。一个**完全构建的对象**是指所有数据成员都适当地初始化了。给定类的所有构造函数（包括拷贝构造函数）都必须经过验证以确保初始化所有数据成员。数据成员加载的值应该检查范围适宜性。记住，未初始化的数据成员可能是一个潜在的灾难！如果给定的构造函数没有正确完成或者数据成员的初始值不合适，应该采取预防措施。

可以使用各种技术来验证完全构建的对象。一种（不推荐）的基本技术是在每个类中嵌入一个状态数据成员（或从或嵌入一个状态祖先/成员）。在成员初始化列表中将状态成员设置为`0`，在构造函数的最后一条语句中将它设置为`1`。在实例化后检查这个值。这种方法的一个巨大缺陷是用户肯定会忘记检查**完全构建**的成功标志。

与上述简单方案不同的替代方案是，对于所有简单数据类型，利用课堂初始化，将这些成员在替代构造函数的成员初始化列表中重置为所需的值。实例化之后，可以再次探测这些值，以确定是否成功完成了替代构造函数。这仍然远非理想的实现。

一种更好的技术是利用异常处理。将异常处理嵌入到每个构造函数中是理想的。如果数据成员没有在合适的范围内初始化，首先尝试重新输入它们的值，或者打开一个备用的数据库进行输入，例如。作为最后的手段，你可以抛出一个异常来报告*未完全构建的对象*。我们将在本章稍后更详细地检查与测试相关的异常处理。

同时，让我们继续探讨一种严格测试我们的类和组件的技术——创建用于测试类的驱动程序。

# 创建驱动程序以测试类

在*第五章*《详细探索类》中，我们简要地讨论了将代码拆分为源文件和头文件。让我们简要回顾一下。通常，头文件将以类的名称命名（例如`Student.h`），并包含类的定义以及任何内联成员函数的定义。通过将内联函数放在头文件中，如果它们的实现发生变化（因为头文件随后包含在每个源文件中，从而与该头文件建立依赖关系），它们将被正确地重新展开。

每个类的方法实现将放置在相应的源代码文件中（例如`Student.cpp`），该文件将包含基于它的头文件（即`#include "Student.h"`）。注意，双引号表示该头文件位于我们的当前工作目录中；我们也可以指定一个路径来查找头文件。相比之下，与 C++库一起使用的尖括号告诉预处理器在编译器预先指定的目录中查找。此外，请注意，每个派生类的头文件将包含其基类的头文件（这样它可以看到成员函数原型）。

注意，任何静态数据成员或方法定义都将出现在它们相应的源代码文件中（这样每个应用程序将只有一个定义）。

在考虑到这个头文件和源代码文件结构的情况下，我们现在可以创建一个驱动程序来测试每个单独的类或每个紧密相关的类的分组（例如，通过关联或聚合相关的类）。通过继承相关的类可以在它们自己的单独的驱动程序文件中进行测试。每个驱动程序文件可以命名为反映正在测试的类，例如`StudentDriver.cpp`。驱动程序文件将包含正在测试的类（的）相关头文件。当然，在编译过程中，相关类的源文件会被编译并链接到驱动程序文件中。

驱动文件可以简单地包含一个`main()`函数作为测试床，以实例化相关的类，并作为测试每个成员函数的范围。驱动程序将测试默认实例化、典型实例化、拷贝构造、对象之间的赋值以及类中每个额外的方法。如果存在虚拟析构函数或其他虚拟函数，我们应该实例化派生类实例（在派生类的驱动程序中），将这些实例向上转换为使用基类指针存储，然后调用虚拟函数以验证是否发生正确的行为。在虚拟析构函数的情况下，我们可以通过删除动态分配的实例（或等待栈实例超出作用域）并使用调试器单步执行来跟踪析构序列中的入口点，以验证一切是否如预期。

我们还可以测试对象是否完全构造；我们将在稍后了解更多关于这个话题的内容。

假设我们有一个常见的`Person`和`Student`类层次结构，以下是一个简单的驱动程序（包含`main()`的文件）来测试`Student`类。这个驱动程序可以在我们的 GitHub 仓库中找到。要制作一个完整的程序，您还需要编译并链接同一目录中找到的`Student.cpp`和`Person.cpp`文件。以下是驱动程序的 GitHub 仓库 URL：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter15/Chp15-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter15/Chp15-Ex2.cpp)

```cpp
#include "Person.h"  // include relevant class header files
#include "Student.h"
using std::cout;    // preferred to: using namespace std;
using std::endl;
constexpr int MAX = 3;
int main() // Driver to test Student class, stored in above
{          // filename for chapter example consistency 
    // Test all instantiation means, even copy constructor
    Student s0; // Default construction
    // alternate constructor
    Student s1("Jo", "Li", 'H', "Ms.", 3.7, "C++", 
               "UD1234");
    Student s2("Sam", "Lo", 'A', "Mr.", 3.5, "C++",
               "UD2245");
    // These initializations implicitly invoke copy const.
    Student s3(s1);
    Student s4 = s2;   // This is also initialization
    // Test the assignment operator
    Student s5("Ren", "Ze", 'A', "Dr.", 3.8, "C++",
               "BU5563");
    Student s6;
    s6 = s5;  // this is an assignment, not initialization
    // Test each public method. A sample is shown here
    s1.Print();  // Be sure to test each method! 

    // Generalize derived instances as base types 
    // Do the polymorphic operations work as expected?
    Person *people[MAX] = { }; // initialized with nullptrs
    // base instance for comparison
    people[0] = new Person("Juliet", "Martinez", 'M',
                           "Ms.");
    // derived instances, generalized with base class ptrs.   
    people[1] = new Student("Zack", "Moon", 'R', "Dr.",
                            3.8, "C++", "UMD1234");  
    people[2] = new Student("Gabby", "Doone", 'A', "Dr.", 
                            3.9, "C++", "GWU4321");
    for (auto *item : people)  // loop through all elements
    {
       item->IsA();
       cout << "  ";
       item->Print();
    }
    // Test destruction sequence (dynam. alloc. instances)
    for (auto *item : people)  // loop thru all elements
       delete item;   // engage virtual dest. sequence
    return 0;
}
```

简要回顾前面的程序片段，我们可以看到我们已经测试了每种实例化方式，包括拷贝构造函数。我们还测试了赋值运算符，验证了每个成员函数的工作情况（示例方法如下），并验证了虚拟函数（包括虚拟析构函数）按预期工作。

现在我们已经看到基本驱动程序测试了我们的类，让我们考虑一些在测试通过继承、关联或聚合相关的类时可以使用的额外指标。

# 测试相关类

在面向对象程序中，仅仅测试单个类以验证完整性和健壮性是不够的，尽管这些都是好的起点。完整性不仅意味着遵循规范类形式，而且还确保数据成员有使用适当访问方法（当不修改实例时标记为`const`）的安全访问方式。完整性还验证了面向对象设计指定的所需接口是否已实现。

健壮性使我们验证所有上述方法是否已在适当的驱动程序中测试过，评估了平台独立性，并验证了每种实例化的方式都导致一个完全构建的对象。我们可以通过数据成员的阈值测试来增强这种类型的测试，例如，注意何时抛出异常。虽然看似全面，但完整性和健壮性实际上是面向对象组件测试的最直接手段。

更具挑战性的测试方法是测试相关类之间的交互。

## 测试通过继承、关联或聚合相关联的类

通过各种对象关系相关联的类需要各种额外的测试手段。具有彼此之间不同关系的对象可能会影响给定实例在其应用程序生命周期内可能具有的状态进展。这种测试将需要最详细的工作。我们将发现场景对于帮助我们捕捉相关对象之间的通常交互是有用的，从而导致测试相互交互的类的更全面的方法。

让我们先考虑如何测试与继承相关的类。

### 添加测试继承的策略

通过公共继承相关联的类需要验证虚函数。例如，所有预期的派生类方法是否都被覆盖了？记住，如果基类行为在派生类级别仍然被认为是合适的，派生类不需要覆盖其基类中指定的所有虚函数。将实现与设计进行比较是必要的，以确保我们已用适当的方法覆盖了所有必需的多态操作。

当然，虚函数的绑定是在运行时完成的（即动态绑定）。创建派生类实例并使用基类指针存储它们，以便应用多态操作，这将是重要的。然后我们需要验证派生类行为是否显现出来。如果没有，我们可能发现自己处于一个意外的函数隐藏情况，或者可能是基类操作没有按照预期标记为`virtual`（记住，在派生类级别，关键字`virtual`和`override`虽然很好且推荐，但不是必需的，并且不影响动态行为）。

虽然通过继承相关联的类有独特的测试策略，但请记住，实例化将创建一个单一的对象，即基类或派生类类型。当我们实例化此类类型时，我们有一个这样的实例，而不是一对共同工作的实例。派生类仅仅有一个基类子对象，它是它自身的一部分。让我们考虑一下这与关联对象或聚合体相比如何，这些可以是独立的对象（关联），可能与其伴侣相互作用。

### 添加测试聚合和关联的策略

通过关联或聚合相关联的类可能存在多个实例相互通信并相互引起状态变化。这肯定比继承的对象关系更复杂。

通过聚合相关联的类通常比通过关联相关联的类更容易测试。考虑到最常见的聚合形式（组合），嵌入式（内部）对象是外部（整体）对象的一部分。当外部对象被实例化时，我们得到嵌入在*整体*中的内部对象的内存。与包含基类子对象的派生类实例的内存布局相比，内存布局并没有太大不同（除了可能的顺序）。在每种情况下，我们仍然在处理一个单一实例（尽管它有嵌入式*部分*）。然而，在测试中的比较点是，应用于*整体*的操作通常被委托给*部分*或组件。我们将严格需要测试整体上的操作，以确保它们将必要的信息委托给每个部分。

通过较少使用的通用聚合形式（其中整体包含对部分的指针，而不是典型的组合嵌入式对象实现）相关联的类，与关联有类似的问题，因为实现是相似的。考虑到这一点，让我们看看与关联对象相关的测试问题。

通过关联相关联的类通常是独立存在的对象，它们在应用中的某个时刻创建了彼此之间的链接。在应用中，两个对象创建彼此之间的链接可能有一个预定的点，也可能没有。对一个对象应用的操作可能会引起相关对象的变化。例如，让我们考虑一个`Student`（学生）和一个`Course`（课程）。它们可能独立存在，然后在应用中的某个时刻，一个`Student`可能通过`Student::AddCourse()`添加一个`Course`。通过这样做，不仅特定的`Student`实例现在包含了对特定`Course`实例的链接，而且`Student::AddCourse()`操作已经引起了`Course`类的变化。那个特定的`Student`实例现在成为特定`Course`实例名单的一部分。在任何时候，`Course`都可能被取消，从而影响所有注册该`Course`的`Student`实例。这些变化反映了每个相关对象可能存在的状态。例如，一个`Student`可能处于**当前注册**或**退课**的状态。有许多可能性。我们如何测试所有这些情况？

### 添加场景以帮助测试对象关系

在面向对象分析中，场景的概念被提出作为一种既可创建面向对象设计又可对其进行测试的手段。**场景**是对应用中可能发生的一系列事件的描述性遍历。一个场景将展示类以及它们在特定情况下可能如何相互交互。许多相关的场景可以收集到面向对象的**用例**概念中。在面向对象分析和设计阶段，场景有助于确定应用中可能存在的类以及每个类可能具有的操作和关系。在测试中，场景可以被重用来形成驱动器创建的基础，以测试各种对象关系。考虑到这一点，可以开发一系列驱动器来测试许多场景（即用例）。这种类型的建模将能够比最初简单测试完整性和鲁棒性的方法更全面地为相关对象提供测试平台。

任何类型的相关类之间的另一个关注领域是版本控制。例如，如果基类定义或默认行为发生变化，会发生什么？这将对派生类有何影响？这将对相关对象有何影响？随着每次变化，我们不可避免地需要重新访问所有相关类的组件测试。

接下来，让我们考虑异常处理机制如何在面向对象组件测试中发挥作用。

# 测试异常处理机制

现在我们能够创建用于测试每个类（或相关类的组合）的驱动程序，我们将希望了解我们代码中的哪些方法可能会抛出异常。对于这些场景，我们希望在驱动程序中添加 try 块以确保我们知道如何处理每个潜在的异常。在这样做之前，我们应该问自己，在开发过程中，我们是否在我们的代码中包含了足够的异常处理？例如，考虑到实例化，我们的构造函数是否检查对象是否完全构建？如果没有，它们是否会抛出异常？如果答案是“否”，那么我们的类可能没有我们预期的那么健壮。

让我们考虑将异常处理嵌入到构造函数中，以及我们如何构建一个驱动程序来测试所有可能的实例化方法。

## 在构造函数中嵌入异常处理以创建健壮的类

我们可能还记得我们最近的*第十一章*，*处理异常*，我们可以创建自己的异常类，这些类是从 C++ 标准库 `exception` 类派生出来的。让我们假设我们已经创建了一个这样的类，即 `ConstructionException`。如果在构造函数的任何点上我们无法正确初始化给定的实例以提供一个完全构建的对象，我们可以从任何构造函数中抛出 `ConstructionException`。抛出 `ConstructionException` 的潜在含义是，我们现在应该将实例化包含在 try 块中，并添加匹配的 catch 块来预测可能抛出的 `ConstructionException`。然而，请记住，在 try 块作用域内声明的实例的作用域仅限于 `try`-`catch` 对。

好消息是，如果一个对象没有完成构建（即，如果在构造函数完成之前抛出了异常），那么从技术上讲，该对象将不存在。如果一个对象在技术上不存在，那么就没有必要清理部分实例化的对象。然而，我们需要考虑，如果我们预期的实例没有完全构建，这对我们的应用程序意味着什么？这将如何改变我们代码的执行流程？测试的一部分是确保我们已经考虑了我们的代码可能被使用的所有方式，并相应地使我们的代码更加健壮！

重要的是要注意，引入 `try` 和 `catch` 块可能会改变我们的程序流程，并且在我们的驱动程序中包含这种类型的测试是至关重要的。在我们进行测试时，我们可能会寻找考虑 `try` 和 `catch` 块的场景。

我们已经看到如何增强我们的测试驱动程序以适应可能会抛出异常的类。我们也在本章中讨论了在我们的驱动程序中添加场景以帮助跟踪具有关系的对象之间的状态，以及当然，我们可以遵循的简单类习惯用法，以帮助我们取得成功。现在，在我们继续到下一章之前，让我们简要回顾这些概念。

# 摘要

在本章中，通过检查各种面向对象类和组件测试实践和策略，我们提高了成为更好的 C++ 程序员的能力。我们的主要目标是确保我们的代码健壮、经过良好测试，并且可以无错误地部署到我们的各个组织中。

我们考虑了编程惯用法，例如遵循经典类形式以确保我们的类完整，并且对于构造/析构、赋值以及在参数传递和函数返回值中的使用具有预期的行为。我们讨论了创建健壮类意味着什么——即遵循经典类形式且经过良好测试、平台无关且对完全构造的对象进行了测试的类。

我们还探讨了如何创建驱动程序来测试单个类或相关类的集合。我们为驱动程序中的单个类建立了一个测试清单。我们更深入地研究了对象关系，以了解相互交互的对象需要更复杂的测试。也就是说，当对象从一个状态移动到另一个状态时，它们可能会受到相关对象的影响，这可能会进一步改变它们的进程。我们为我们的驱动程序添加了利用场景作为测试用例，以更好地捕捉实例在应用程序中可能移动的动态状态。

最后，我们已经探讨了异常处理机制如何影响我们测试代码的方式。我们已经增强了我们的驱动程序，以考虑 try 和 catch 块可能会将我们的应用程序从预期的典型流程中重定向的流程控制。

我们现在可以继续前进，进入我们书籍的下一部分，即 C++ 中的设计模式和惯用法。我们将从*第十六章*开始，*使用观察者模式*。在接下来的章节中，我们将了解如何应用流行的设计模式并将它们应用于我们的编码。这些技能将使我们成为更好的程序员。让我们继续前进！

# 问题

1.  考虑到你的前一个练习中的一个包含对象关系的类对（提示——与关联相比，公共继承更容易考虑）。

    1.  你的类遵循经典类形式吗？是正则的还是扩展的？为什么，或者为什么不？如果它们不遵循并且应该遵循，请修改类以遵循这个惯用法。

    1.  你认为你的类是健壮的吗？为什么，或者为什么不？

1.  创建一个（或两个）驱动程序来测试你的类对：

    1.  确保测试以下常规清单项目（构造、赋值、析构、公共接口、向上转型（如果适用）以及虚拟函数的使用）。

    1.  （可选）如果你选择了通过关联使用的关系相关的两个类，创建一个单独的驱动程序来遵循典型场景，详细说明两个类的交互。

    1.  确保在你的测试驱动程序中包含异常处理的测试。

1.  创建一个`ConstructionException`类（从 C++标准库的`exception`派生）。在示例类中嵌入检查，在必要时抛出`ConstructionException`。确保将此类的所有实例化形式都包含在适当的`try`和`catch`块对中。

# 第四部分：C++中的设计模式和惯用法

本部分的目标是扩展你的 C++技能库，不仅限于面向对象编程和其他必要技能，还包括核心设计模式的知识。设计模式提供了经过验证的技术和策略来解决重复出现的 OO 问题。本节介绍了常见的设计模式，并通过在书中构建创意示例深入展示了如何应用这些模式。每一章都包含详细的代码示例，以说明每个模式。

本节的第一章介绍了设计模式的概念，并讨论了在编码解决方案中利用此类模式的优势。第一章还介绍了观察者模式，并提供了深入理解该模式各个组成部分的程序示例。

下一章解释了工厂方法模式，并同样提供了详细的程序，展示了如何实现带和不带对象工厂的工厂方法模式。本章还比较了对象工厂与抽象工厂。

下一章介绍了适配器模式，并提供了使用继承与关联实现适配器类的实现策略和程序示例。此外，还展示了适配器作为一个简单的包装类。

下一章将探讨单例模式。在两个简单的示例之后，展示了使用配对类实现的详细示例。还介绍了用于容纳单例的注册表。

本节和本书的最后一章介绍了 pImpl 模式，以减少代码中的编译时依赖。提供了一个基本实现，然后使用唯一指针进行扩展。此外，还探讨了与该模式相关的性能问题。

本部分包括以下章节：

+   *第十六章*, *使用观察者模式*

+   *第十七章*, *应用工厂模式*

+   *第十八章*, *应用适配器模式*

+   *第十九章*, *使用单例模式*

+   *第二十章*, *使用 pImpl 模式去除实现细节*

第四部分：C++中的设计模式和惯用法
