

# 第十章：实现关联、聚合和组合

本章将继续深化我们对 C++ 面向对象编程知识的理解。我们将通过探索关联、聚合和组合等面向对象概念来增强我们对对象关系的理解。这些面向对象的概念在 C++ 中没有直接的语言支持；我们将学习多种编程技术来实现这些想法。我们还将了解针对各种概念首选的实现技术，以及各种实践的优势和陷阱。

关联、聚合和组合在面向对象（OO）设计中频繁出现。理解如何实现这些重要的对象关系至关重要。

在本章中，我们将涵盖以下主要内容：

+   理解聚合和组合的面向对象概念及其各种实现

+   理解关联的面向对象概念及其实现，包括后链维护的重要性以及引用计数的实用性

到本章结束时，你将理解关联、聚合和组合的面向对象概念，以及如何在 C++ 中实现这些关系。你还将了解许多必要的维护方法，以保持这些关系最新，例如引用计数和后链维护。尽管这些概念相对简单，但你将了解为什么需要大量的记录来保持这些类型对象关系的准确性。

通过探索这些核心对象关系，让我们扩展对 C++ 作为面向对象（OOP）语言的了解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter10`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter10)。每个完整程序示例都可以在 GitHub 上找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter10` 子目录中找到，文件名为 `Chp10-Ex1.cpp`。

本章的 CiA 视频可在以下网址查看：[`bit.ly/3clgvGe`](https://bit.ly/3clgvGe)。

# 理解聚合和组合

聚合的面向对象概念在许多面向对象设计中出现。它出现的频率与继承一样，用于指定对象关系。"聚合"用于指定“拥有”（Has-A）、整体-部分关系，在某些情况下，还用于包含关系。一个类可以包含其他对象的聚合。聚合可以分为两类——*组合*以及一种不那么严格的*泛化*聚合形式。

通用聚合和组合都暗示了“拥有”或整体-部分关系。然而，这两个相关对象的存在要求之间存在差异。在通用聚合中，对象可以独立存在，而在组合中，对象不能没有对方而存在。

让我们来看看聚合的每一种类型，从组合开始。

## 定义和实现组合

**组合**是聚合最特殊的形式，并且通常是大多数面向对象的设计师和程序员在考虑聚合时首先想到的。组合意味着包含，通常与整体-部分关系同义——也就是说，整体实体由一个或多个部分组成。整体包含部分。拥有关系也将适用于组合。

外部对象，或整体，可以由部分组成。在组合中，部分的存在依赖于整体。实现通常是嵌入的对象——也就是说，是包含对象类型的成员数据。在罕见的情况下，外部对象将包含指向包含对象类型的指针或引用；然而，当这种情况发生时，外部对象将负责创建和销毁内部对象。没有外部层，包含对象就没有任何目的。同样，没有其内部包含的部分，外部层也不是理想上完整的。

让我们看看一个通常实现的组合。示例将展示包含——一个`Student`拥有一个`Id`。更进一步，我们将暗示`Id`是`Student`的一个必要部分，没有`Student`它将不存在。单独的`Id`对象没有任何作用。如果它们不是赋予它们目的的主要对象的一部分，`Id`对象就无需存在。同样，你可能会说，没有`Id`，`Student`就不完整，尽管这有点主观！我们将使用嵌入在整体中的对象来实现部分。

组合示例将被分成许多部分。尽管只展示了示例的一部分，但完整的程序可以在以下 GitHub 位置找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter10/Chp10-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter10/Chp10-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setprecision;
using std::string;
using std::to_string;
class Id final  // the contained 'part'
{        // this class is not intended to be extended 
private:
    string idNumber;
public:
    Id() = default;
    Id(const string &id): idNumber(id) { }
    // We get default copy constructor, destructor
    // without including without including prototype
    // Id(const Id &id) = default;
    // ~Id() = default;
    const string &GetId() const { return idNumber; }
};
```

在之前的代码片段中，我们定义了一个`Id`类。`Id`将是一个可以被其他需要完全功能的`Id`能力的类所包含的类。`Id`将成为任何可能选择包含它的任何整体对象的一部分。

让我们继续前进，构建一组最终将包含这个`Id`的类。我们将从一个我们熟悉的类`Person`开始：

```cpp
class Person
{
private:
    string firstName;
    string lastName;
    char middleInitial = '\0';   // in-class initialization
    string title;  // Mr., Ms., Mrs., Miss, Dr., etc.
protected:
    void ModifyTitle(const string &);
public:
    Person() = default;   // default constructor
    Person(const string &, const string &, char, 
           const string &);
    // We get default copy constructor w/o prototype 
    // Person(const Person &) = default;  // copy ctor.
    // But, we need prototype destructor to add 'virtual' 
    virtual ~Person() = default;  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; }
    const string &GetLastName() const { return lastName; }
    const string &GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    // virtual functions
    virtual void Print() const;   
    virtual void IsA() const;
    virtual void Greeting(const string &) const;
};
//  Assume the member functions for Person exist here
//  (they are the same as in previous chapters)
```

在之前的代码段中，我们已经定义了`Person`类，正如我们习惯描述的那样。为了简化这个例子，让我们假设伴随的成员函数如上述类定义中原型所示存在。您可以在之前提供的 GitHub 链接中参考这些成员函数的在线代码。

现在，让我们定义我们的`Student`类。尽管它将包含我们习惯看到的元素，但`Student`还将包含一个作为嵌入对象的`Id`：

```cpp
class Student: public Person  // 'whole' object
{
private:
    float gpa = 0.0;    // in-class initialization
    string currentCourse;
    static int numStudents;  
    Id studentId;  // is composed of a 'part'
public:    
    Student();  // default constructor
    Student(const string &, const string &, char, 
            const string &, float, const string &, 
            const string &);
    Student(const Student &);  // copy constructor
    ~Student() override;  // destructor
    // various member functions (many are inline)
    void EarnPhD() { ModifyTitle("Dr."); } 
    float GetGpa() const { return gpa; }         
    const string &GetCurrentCourse() const
        { return currentCourse; }
    void SetCurrentCourse(const string &); // proto. only
    void Print() const override;
    void IsA() const override 
        { cout << "Student" << endl; }
    static int GetNumberStudents() { return numStudents; }
    // Access function for embedded Id object
    const string &GetStudentId() const;   // prototype only
};
int Student::numStudents = 0;  // static data member
inline void Student::SetCurrentCourse(const string &c)
{
    currentCourse = c;
}
```

在先前的`Student`类中，我们通常会注意到`Student`是从`Person`派生出来的。正如我们已经知道的，这意味着一个`Student`实例将包含一个`Person`的内存布局，作为一个`Person`子对象。

然而，请注意`Student`类定义中的数据成员`Id studentId;`。在这里，`studentId`是`Id`类型。它不是一个指针，也不是`Id`的引用。数据成员`studentId`是一个嵌入对象（即聚合或成员对象）。这意味着当`Student`类被实例化时，不仅将包含从继承的类中继承的内存，还包括任何嵌入对象的内存。我们需要提供一种初始化嵌入对象`studentId`的方法。注意，我们之前已经见过成员对象，例如类型为`string`的数据成员；即数据成员是另一个类类型。

让我们继续前进，通过`Student`成员函数来了解我们如何初始化、操作和访问嵌入的对象：

```cpp
Student::Student(): studentId(to_string(numStudents + 100) 
                                         + "Id") 
{
    numStudents++;   // increment static counter
}
Student::Student(const string &fn, const string &ln, 
                 char mi, const string &t, float avg, 
                 const string &course, const string &id):  
                 Person(fn, ln, mi, t), gpa(avg),
                 currentCourse(course), studentId(id)
{
    numStudents++;
}
Student::Student(const Student &s): Person(s),
                gpa(s.gpa), currentCourse(s.currentCourse),
                studentId(s.studentId)
{
    numStudents++;
}
Student::~Student()   // destructor definition
{
    numStudents--;    // decrement static counter
    // embedded object studentId will also be destructed
}
void Student::Print() const
{
    cout << GetTitle() << " " << GetFirstName() << " ";
    cout << GetMiddleInitial() << ". " << GetLastName();
    cout << " with id: " << studentId.GetId() << " GPA: ";
    cout << setprecision(3) <<  " " << gpa;
    cout << " Course: " << currentCourse << endl;
}    
const string &GetStudentId() const 
{   
    return studentId.GetId();   
} 
```

在之前列出的`Student`成员函数中，让我们从构造函数开始。注意在默认构造函数中，我们利用成员初始化列表（`:`）来指定`studentId(to_string(numStudents + 100) + "Id")`。因为`studentId`是一个成员对象，所以我们有机会（通过成员初始化列表）选择用于其初始化的构造函数。在这里，我们只是选择具有`Id(const string &)`签名的构造函数。如果没有特定的值用于初始化`Id`，我们将制造一个字符串值来作为所需的 ID。

同样，在`Student`的另一个构造函数中，我们使用成员初始化列表来指定`studentId(id)`，这也会选择`Id(const string &)`构造函数，并将参数`id`传递给这个构造函数。

`Student`的拷贝构造函数还额外指定了如何使用成员初始化列表中的`studentId(s.studentId)`规范来初始化`studentId`成员对象。在这里，我们只是调用了`Id`的拷贝构造函数。

在我们的 `Student` 析构函数中，我们不需要释放 `studentId` 的内存。因为这个数据成员是一个嵌入的（聚合）对象，其内存将在外部对象的内存释放时消失。当然，因为 `studentId` 本身也是一个对象，所以它的析构函数将在其内存释放之前首先被调用。在底层，编译器将（隐式地）在 `Student` 析构函数的最后一条代码中插入对 `Id` 析构函数的调用。实际上，这将是析构函数中的倒数第二行隐式插入的代码——最后将被隐式插入的将是调用 `Person` 析构函数（以继续销毁序列）。

最后，在之前的代码段中，让我们注意对 `studentId.GetId()` 的调用，这在 `Student::Print()` 和 `Student::GetStudentId()` 中都发生了。在这里，嵌入对象 `studentId` 调用其自己的公共函数 `Id::GetId()` 来检索其在 `Student` 类作用域内的私有数据成员。因为 `studentId` 在 `Student` 中是私有的，这个嵌入对象只能在其作用域内（即 `Student` 的成员函数）访问。然而，`Student::GetStudentId()` 的添加为其他作用域中的 `Student` 实例提供了一个公共包装器来检索此信息。

最后，让我们看一下我们的 `main()` 函数：

```cpp
int main()
{
    Student s1("Cyrus", "Bond", 'I', "Mr.", 3.65, "C++",
               "6996CU");
    Student s2("Anne", "Brennan", 'M', "Ms.", 3.95, "C++",
               "909EU");
    cout << s1.GetFirstName() << " " << s1.GetLastName();
    cout << " has id #: " << s1.GetStudentId() << endl;
    cout << s2.GetFirstName() << " " << s2.GetLastName();
    cout << " has id #: " << s2.GetStudentId() << endl;
    return 0;
}
```

在前面提到的 `main()` 函数中，我们创建了两个 `Student` 实例：`s1` 和 `s2`。当为每个 `Student` 创建内存（在这种情况下，在栈上）时，任何继承的类的内存也将作为子对象包含在内。此外，任何嵌入的对象，如 `Id`，也将作为 `Student` 内部的子对象进行布局。包含的对象，或称为 *部分*，将与外部对象，或称为 *整体*，的分配一起分配。

接下来，让我们注意对包含部分的访问，即嵌入的 `Id` 对象。我们从调用 `s1.GetStudentId()` 开始；`s1` 访问一个 `Student` 成员函数，`GetStudentId()`。这个学生成员函数将利用 `studentId` 的成员对象来调用 `Id::GetId()`，这是 `Id` 类型的内部对象。`Student::GetStudentId()` 成员函数可以通过简单地返回 `Id::GetId()` 在嵌入对象上返回的值来实现所需的公共访问。

让我们看一下上述程序的输出：

```cpp
Cyrus Bond has id #: 6996CU
Anne Brennan has id #: 909EU 
```

这个例子详细介绍了组合及其典型实现，即嵌入对象。现在，让我们看看一种使用较少的、替代的实现方式——继承。

### 考虑为组合使用另一种实现方式

有必要了解，组合也可以通过继承来实现，然而，这种方法极具争议。记住，继承最常用于实现“是”某种类型（*Is-A*）和“拥有”某种类型（*Has-A*）的关系。我们在*第九章*中简要描述了使用继承来实现“拥有”关系，即*探索多重继承*。

总结一下，你只需从“部分”继承，而不是将部分作为数据成员嵌入。这样做时，你不再需要为“部分”提供“包装”函数，例如我们之前程序中看到的，`Student::GetStudentId()` 方法调用 `studentId.GetId()` 以提供对其内嵌部分的访问。在嵌入式对象示例中，包装函数是必要的，因为部分（`Id`）在整体（`Student`）中是私有的。程序员无法在 `Student` 的作用域之外访问私有的 `studentId` 数据成员。当然，`Student` 的成员函数（如 `GetStudentId()`）可以访问它们自己类的私有数据成员，并在这样做时，可以实施 `Student::GetStudentId()` 包装函数以提供这样的（安全）访问。

如果使用了继承，`Id::GetId()` 的公共接口将简单地作为 `Student` 中的公共接口继承，提供简单的访问，无需首先显式地通过嵌入式对象。

尽管在某些方面继承“部分”很简单，但它极大地增加了多重继承的复杂性。我们知道多重继承可以带来许多潜在的问题。此外，使用继承，整体只能包含每种“部分”的一个实例——不能有多个“部分”的实例。

此外，当你将实现与面向对象设计进行比较时，使用继承实现整体-部分关系可能会令人困惑。记住，继承通常意味着“是”某种类型（*Is-A*）而不是“拥有”某种类型（*Has-A*）。因此，聚合最典型和最受欢迎的实现方式是通过内嵌对象。

接下来，让我们继续探讨更一般的聚合形式。

## 定义和实现泛化聚合

我们已经探讨了面向对象设计中最常用的聚合形式，即组合。最值得注意的是，通过组合，我们看到部分没有理由在没有整体的情况下存在。然而，存在一种更通用的（但不太常见）的聚合形式，有时在面向对象设计中指定。我们现在将考虑这种不太常见的聚合形式。

在**泛化聚合**中，一个“部分”可能存在于没有“整体”的情况下。一个部分将单独创建，然后在稍后的时间点附加到整体上。当“整体”消失时，一个“部分”可能仍然可以用于与另一个外部或“整体”对象一起使用。

在广义聚合中，Has-A 关系当然适用，整个部分指定也适用。区别在于，整体对象不会创建或销毁部分子对象。考虑一个简单的例子，一个`Car` *具有* 一个`Engine`。`Car`对象也*具有*一套四个`Tire`对象。`Engine`或`Tire`对象可以单独制造，然后传递给`Car`的构造函数，为整体提供这些部分。然而，如果销毁一个`Engine`，可以很容易地用新的`Engine`替换（使用成员函数），而不需要销毁整个`Car`并重新构建。

广义聚合相当于 Has-A 关系，但我们将其视为具有更多灵活性和个体部分的永久性，就像我们在组合中所做的那样。我们将其视为聚合关系，仅仅因为我们希望将对象等同于具有 Has-A 意义。在`Car`、`Engine`和`Tire`示例中的 Has-A 关系是强烈的；`Engine`和`Tire`是必要的部分，是构成整个`Car`所必需的。

在这里，实现通常是整体包含指向部分（或一组指针）的指针。重要的是要注意，部分将被传递到外部对象的构造函数（或另一个成员函数）中，以建立这种关系。关键标记是整体不会创建（也不会销毁）部分，部分永远不会销毁整体。

顺便提一下，广义聚合的各个部分（和基本实现）的持久性将类似于我们下一个主题——关联。让我们继续前进到下一个部分，了解广义聚合和关联之间的相似性，以及 OO 概念上的差异（有时微妙）。

# 理解关联

**关联**模型了存在于不同类类型之间的关系。关联可以提供对象交互的方式以满足这些关系。然而，关联不用于 Has-A 关系，但在某些情况下，我们描述的是一个真正的 Has-A 关系，还是仅仅因为听起来在语言上合适而使用 Has-A 这个短语，这之间可能存在灰色地带。

关联的多重性存在：一对一、一对多、多对一或多对多。例如，一个`Student`可以与一个单独的`University`相关联，而这个`University`可以与许多`Student`实例相关联；这是一个一对多关联。

关联对象具有独立的存在性。也就是说，两个或多个对象可能被实例化并独立存在于应用程序的一部分。在某个时刻，一个对象可能希望断言与其他对象的依赖关系或关系。在应用程序的后期，关联对象可能分道扬镳，继续它们各自无关的路径。

例如，考虑`课程`和`讲师`之间的关系。一个`课程`与一个`讲师`相关联。一个`课程`需要有一个`讲师`；`讲师`对`课程`是必不可少的。一个`讲师`可以与多个`课程`(s)相关联。然而，每个部分都是独立存在的——一个不会创建或摧毁另一个。讲师也可以独立于课程存在；也许讲师正在花时间写一本书，正在休假，或者是一位正在进行研究的教授。

在这个例子中，关联与泛化聚合非常相似。在这两种情况下，相关对象也都是独立存在的。在这种情况下，无论是说`课程`拥有`讲师`，还是说`课程`对`讲师`有依赖关系，都可能是一种灰色地带。你可能自己问自己——是口语使我选择了“拥有”这个词吗？我是不是意味着两者之间存在必要的联系？也许这种关系是一种关联，其描述性修饰语（进一步描述关联的性质）是*教授*。你可能对两种选择都有支持性的论点。因此，泛化聚合可以被认为是关联的特殊类型；我们将看到，它们使用独立存在的对象实现时是相同的。尽管如此，我们将区分典型的关联为对象之间的关系，这种关系明确不支持真正的“拥有”关系。

例如，考虑`大学`和`讲师`之间的关系。我们与其将其视为“拥有”关系，不如将其视为两者之间的关联关系；我们可以将描述这种关系的修饰语视为*雇佣*。同样，`大学`与许多`学生`对象建立关系。这里的关联可以用修饰语*教育*来描述。可以区分的是，`大学`由`系`对象、`建筑`对象以及此类组件组成，以通过包含支持其“拥有”关系，然而其与`讲师`对象、`学生`对象等的关系则是通过关联来实现的。

现在我们已经区分了典型的关联和泛化聚合，让我们来看看我们如何实现关联以及其中涉及的一些复杂性。

## 实现关联

通常，两个或多个对象之间的关联是通过指针或指针集合实现的。一方使用指向相关对象的指针来实现，而关系的多方使用指向相关对象的指针集合来实现。指针集合可能是一个指针数组、指针链表，或者真正任何指针集合。每种类型的集合都有自己的优点和缺点。例如，指针数组易于使用，可以直接访问特定成员，但项目数量是固定的。指针链表可以容纳任何数量的项目，但访问特定元素需要遍历其他元素以找到所需的项目。

有时，可能会使用引用来实现关联的一侧。回想一下，引用必须初始化，并且以后不能重置以引用另一个对象。使用引用来建模关联意味着在主对象存在期间，一个实例将与另一个精确的实例相关联。这是非常限制性的，因此引用很少用于实现关联。

无论实现方式如何，当主对象消失时，它不会干扰（即删除）相关对象。

让我们看看一个典型的例子，它说明了如何实现一对多关联的首选方法，在一方使用指针，而在多方使用指针集合。在这个例子中，一个`University`将与多个`Student`实例相关联。为了简单起见，一个`Student`将与一个单一的`University`相关联。

为了节省空间，本程序中与上一个示例相同的部分将不会显示；然而，整个程序可以在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter10/Chp10-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter10/Chp10-Ex2.cpp)

```cpp
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setprecision;
using std::string;
using std::to_string;
// classes Id and Person are omitted here to save space.
// They will be as shown in previous example: Chp10-Ex1.cpp
class Student; // forward declaration
class University
{
private:
    string name;
    static constexpr int MAX = 25; // max students allowed
    // Notice: each studentBody element is set to a nullptr 
    // using in-class initialization 
    Student *studentBody[MAX] = { }; // Association to
                                     // many students
    int currentNumStudents = 0;  // in-class initialization
public:
    University();
    University(const string &);
    University(const University &) = delete; // no copies
    ~University();
    void EnrollStudent(Student *);
    const string &GetName() const { return name; }
    void PrintStudents() const;
};
```

在前面的部分中，我们首先注意到`class Student;`的前向声明。这个声明将允许我们的代码在`Student`类定义之前引用`Student`类型。在`University`类定义中，我们看到有一个指向`Student`的指针数组。我们还看到`EnrollStudent()`方法接受一个`Student *`作为参数。前向声明使得在定义之前可以使用`Student`。

我们还注意到，`University`有一个简单的接口，包括构造函数、析构函数和一些成员函数。

接下来，让我们看看`University`成员函数的定义：

```cpp
// Remember, currentNumStudents will be set w in-class init
// and name, as a string member object, will be init to 
// empty. And studentBody (array of ptrs) will also set w
// in-class initialization.
University::University()
{
    // in-lieu of in-class init, we could alternatively set
    // studentBody[i] to nullptr iteratively in a loop:
    // (the student body will start out empty)   
    // for (int i = 0; i < MAX; i++) 
    //    studentBody[i] = nullptr; 
}
University::University(const string &n): name(n)
{   
    // see default constructor for alt init of studentBody
}
University::~University()
{
    // The University will not delete the students
    for (int i = 0; i < MAX; i++)   // only null out 
       studentBody[i] = nullptr;    // their link
}                      
void University::EnrollStudent(Student *s)
{
    // set an open slot in the studentBody to point to the
    // Student passed in as an input parameter
    studentBody[currentNumStudents++] = s;
}
void University::PrintStudents()const
{
    cout << name << " has the following students:" << endl;
    // Simple loop to process set of students, however we
    // will soon see safer, more modern ways to iterate 
    // over partial arrays w/o writing explicit 'for' loops
    for (int i = 0; i < currentNumStudents; i++)
    {
       cout << "\t" << studentBody[i]->GetFirstName();
       cout << " " << studentBody[i]->GetLastName();
       cout << endl;
    }
}
```

仔细观察前面提到的 `University` 方法，我们可以看到，在 `University` 的两个构造函数中，我们可以选择使用 `nullptr` 来将组成我们的 `studentBody` 的指针置为空（而不是我们选择在类内初始化，这同样会初始化每个元素）。同样，在析构函数中，我们也将关联到相关 `Student` 实例的链接置为空。在本节稍后，我们将看到还需要进行一些额外的反向链接维护，但到目前为止，重点是我们将不会删除相关的 `Student` 对象。

由于 `University` 对象和 `Student` 对象将独立存在，因此它们都不会创建或销毁其他类型的实例。

我们还遇到了一个有趣的成员函数，`EnrollStudent(Student *)`。在这个方法中，将传递一个指向特定 `Student` 的指针作为输入参数。我们只是索引到我们的 `Student` 对象指针数组 `studentBody`，并将未使用的数组元素指向新注册的 `Student`。我们使用 `currentNumStudents` 计数器跟踪当前 `Student` 对象的数量，该计数器在将指针分配给数组后通过后增量增加。

我们还注意到，`University` 类有一个 `Print()` 方法，它会打印大学的名称，然后是其当前的学生阵容。它是通过简单地访问 `studentBody` 中的每个相关 `Student` 对象，并要求每个 `Student` 实例调用 `Student::GetFirstName()` 和 `Student::GetLastName()` 方法来实现的。

接下来，现在让我们来看看我们的 `Student` 类定义，包括其内联函数。回想一下，我们假设的 `Person` 类与本章前面看到的相同：

```cpp
class Student: public Person  
{
private:
    // data members
    float gpa = 0.0;  // in-class initialization
    string currentCourse;
    static int numStudents;
    Id studentId;  // part, Student Has-A studentId
    University *univ = nullptr;  // Assoc. to Univ object
public:                          
    // member function prototypes
    Student();  // default constructor
    Student(const string &, const string &, char, 
            const string &, float, const string &, 
            const string &, University *);
    Student(const Student &);  // copy constructor
    ~Student() override;  // destructor
    void EarnPhD() { ModifyTitle("Dr."); }
    float GetGpa() const { return gpa; }
    const string &GetCurrentCourse() const 
        { return currentCourse; }
    void SetCurrentCourse(const string &); // proto. only
    void Print() const override;
    void IsA() const override 
        { cout << "Student" << endl; }
    static int GetNumberStudents() { return numStudents; }
    // Access functions for aggregate/associated objects
    const string &GetStudentId() const 
        { return studentId.GetId(); }
    const string &GetUniversity() const 
        { return univ->GetName(); }
};
int Student::numStudents = 0;  // def. of static data mbr.
inline void Student::SetCurrentCourse(const string &c)
{
    currentCourse = c;
}
```

在前面的代码段中，我们看到 `Student` 类的定义。请注意，我们有一个与大学关联的指针数据成员 `University *univ = nullptr;`，并且该成员使用类内初始化设置为 `nullptr`。

在 `Student` 类的定义中，我们还可以看到有一个包装函数来封装对学生的大学名称的访问，即 `Student::GetUniversity()`。在这里，我们允许关联的对象 `univ` 调用其公共方法 `University::GetName()`，并将该值作为 `Student::GetUniversity()` 的结果返回。

现在，让我们来看看 `Student` 的非内联成员函数：

```cpp
Student::Student(): studentId(to_string(numStudents + 100) 
                                        + "Id")
{
    // no current University association (set to nullptr 
    // with in-class initialization)
    numStudents++;
}
Student::Student(const string &fn, const string &ln, 
          char mi, const string &t, float avg, 
          const string &course, const string &id, 
          University *univ): Person(fn, ln, mi, t), 
          gpa(avg), currentCourse(course), studentId(id)
{
    // establish link to University, then back link
    // note: forward link could also be set in the
    // member initialization list
    this->univ = univ;  // required use of ‹this›
    univ->EnrollStudent(this);  // another required 'this'
    numStudents++;
}
Student::Student(const Student &s): Person(s), 
          gpa(s.gpa), currentCourse(s.currentCourse),
          studentId(s.studentId)
{
    // Notice, these three lines of code are the same as 
    // in the alternate constructor – we could instead make
    // a private helper method with this otherwise 
    // duplicative code as a means to simplify code 
    // maintenance. 
    this->univ = s.univ;    
    univ->EnrollStudent(this);
    numStudents++;
}
Student::~Student()  // destructor
{
    numStudents--;
    univ = nullptr;  // a Student does not delete its Univ
    // embedded object studentId will also be destructed
}
void Student::Print() const
{
    cout << GetTitle() << " " << GetFirstName() << " ";
    cout << GetMiddleInitial() << ". " << GetLastName();
    cout << " with id: " << studentId.GetId() << " GPA: ";
    cout << setprecision(3) <<  " " << gpa;
    cout << " Course: " << currentCourse << endl;
}
```

在前面的代码段中，请注意，默认的 `Student` 构造函数和析构函数都只将它们与 `University` 对象的链接置为空（使用 `nullptr`）。默认构造函数无法将此链接设置为现有对象，并且绝对不应该创建一个 `University` 实例来这样做。同样，`Student` 析构函数也不应该仅仅因为 `Student` 对象的生命周期结束就删除 `University`。

上述代码中最有趣的部分发生在 `Student` 类的替代构造函数和复制构造函数中。让我们来检查替代构造函数。在这里，我们建立了与关联的 `University` 的链接，以及从 `University` 返回到 `Student` 的反向链接。

在代码行 `this->univ = univ;` 中，我们通过将 `this` 指针指向的 `univ` 数据成员设置为指向输入参数 `univ` 指向的位置来赋值。仔细看看之前的类定义——`University *` 的标识符被命名为 `univ`。此外，在替代构造函数中 `University *` 的输入参数也被命名为 `univ`。我们在这个构造函数体（或成员初始化列表）中不能简单地使用 `univ = univ;` 来赋值。在这个最局部作用域中的 `univ` 标识符是输入参数 `univ`。将 `univ = univ;` 赋值将会使这个参数指向自己。相反，我们使用 `this` 指针来消除赋值表达式左侧的 `univ` 的歧义。语句 `this->univ = univ;` 将数据成员 `univ` 设置为输入参数 `univ`。我们能否仅仅将输入参数重命名为不同的名称，比如 `u`？当然可以，但重要的是要理解在需要这样做时如何消除具有相同标识符的输入参数和数据成员的歧义。

现在，让我们检查下一行代码，`univ->EnrollStudent(this);`。由于 `univ` 和 `this->univ` 指向同一个对象，使用哪一个来设置反向链接并不重要。在这里，`univ` 调用 `EnrollStudent()`，这是 `University` 类中的一个公共成员函数。没问题，`univ` 是 `University` 类型。`University::EnrollStudent(Student *)` 期望传入一个指向 `Student` 的指针以在 `University` 一侧完成链接。幸运的是，我们 `Student` 替代构造函数（调用函数的作用域）中的 `this` 指针是一个 `Student *`。`this` 指针（在替代构造函数中）正是我们需要用来创建反向链接的 `Student *`。这是一个需要显式使用 `this` 指针来完成任务的另一个例子。

让我们继续到我们的 `main()` 函数：

```cpp
int main()
{
    University u1("The George Washington University");
    Student s1("Gabby", "Doone", 'A', "Miss", 3.85, "C++",
               "4225GWU", &u1);
    Student s2("Giselle", "LeBrun", 'A', "Ms.", 3.45,
               "C++", "1227GWU", &u1);
    Student s3("Eve", "Kendall", 'B', "Ms.", 3.71, "C++",
               "5542GWU", &u1);
    cout << s1.GetFirstName() << " " << s1.GetLastName();
    cout << " attends " << s1.GetUniversity() << endl;
    cout << s2.GetFirstName() << " " << s2.GetLastName();
    cout << " attends " << s2.GetUniversity() << endl;
    cout << s3.GetFirstName() << " " << s3.GetLastName();
    cout << " attends " << s3.GetUniversity() << endl;
    u1.PrintStudents();
    return 0;
}
```

最后，在我们的 `main()` 函数中的前一个代码片段中，我们可以创建几个独立存在的对象，在它们之间建立关联，然后观察这种关系在实际中的应用。

首先，我们实例化一个 `University`，即 `u1`。然后，我们实例化三个 `Student` 对象，`s1`、`s2` 和 `s3`，并将它们分别关联到 `University u1` 上。请注意，这种关联可以在实例化 `Student` 时设置，或者稍后进行，例如，如果 `Student` 类支持一个 `SelectUniversity(University *)` 接口来这样做的话。

我们随后打印出每个`Student`，以及每个`Student`所就读的`University`的名称。然后，我们打印出我们`University`，`u1`的学生名单。我们注意到，在相关对象之间建立的联系在两个方向上都是完整的。

让我们看看上述程序的输出：

```cpp
Gabby Doone attends The George Washington University
Giselle LeBrun attends The George Washington University
Eve Kendall attends The George Washington University
The George Washington University has the following students:
        Gabby Doone
        Giselle LeBrun
        Eve Kendall
```

我们已经看到，在相关对象之间建立和利用关联是多么容易。然而，实现关联会产生很多维护工作。让我们继续前进，了解参考计数和反向链接维护的必要和相关问题，这将有助于这些维护工作。

## 利用反向链接维护和引用计数

在前面的子节中，我们看到了如何使用指针实现关联。我们看到了如何通过指针将一个对象与关联实例中的另一个对象链接起来。我们还看到了如何通过建立反向链接来完成循环的双向关系。

然而，正如关联对象通常所表现的那样，关系是流动的，并且会随时间变化。例如，对于特定的`University`，其学生名单会经常变化，或者`Instructor`将教授的各种`Course`集合在每个学期也会变化。因此，通常需要移除特定对象与另一个对象之间的关联，或许还会与该类别的不同实例建立关联。但这同时也意味着关联对象必须知道如何移除其与第一个提到的对象的链接。这变得复杂了。

例如，考虑`Student`和`Course`之间的关系。一名`Student`注册了许多`Course`实例。一个`Course`包含了对许多`Student`实例的关联。这是一个多对多关联。让我们想象一下，如果`Student`想要退选一个`Course`，仅仅移除特定`Student`实例对特定`Course`实例的指针是不够的。此外，`Student`必须让特定的`Course`实例知道，相关的`Student`应该从该`Course`的名单中移除。这被称为反向链接维护。

考虑一下，如果一名`Student`只是简单地将其与所退选的`Course`之间的链接置为 null，并且不再采取任何进一步行动，上述场景会发生什么。相关的`Student`实例将没问题。然而，之前关联的`Course`实例仍然会包含一个指向该`Student`的指针。也许这相当于`Student`在`Course`中得到了不及格的成绩，因为`Instructor`仍然认为该`Student`是注册的，但还没有提交作业。最终，`Student`还是受到了影响，得到了不及格的成绩。

记住，对于关联对象，当一个对象完成与另一个对象的工作时，它不会删除另一个对象。例如，当一个`Student`退选一门`Course`时，他们不会删除那门`Course`——只会移除他们对那门`Course`的指针（并且肯定还要处理所需的反向链接维护）。

一个帮助我们进行整体链接维护的想法是考虑**引用计数**。引用计数的目的是跟踪可能指向给定实例的指针数量。例如，如果其他对象指向一个给定的实例，则不应删除该实例。否则，其他对象中的指针将指向已释放的内存，这将导致许多运行时错误。

让我们考虑一个具有多重性的关联，例如`Student`和`Course`之间的关系。`Student`应该跟踪有多少`Course`指针指向`Student`，即`Student`正在选修多少门`Course`。在多个`Course`指向该`Student`的情况下，不应删除`Student`。否则，`Course`将指向已删除的内存。处理这种情况的一种方法是在`Student`析构函数中检查对象（`this`）是否包含任何非空`Course`实例指针。如果对象包含，它需要通过每个活动的`Course`实例调用一个方法，请求从每个这样的`Course`中删除对`Student`的链接。在每个链接被删除后，对应于`Course`实例集合的引用计数可以递减。

同样，链接维护应在`Course`类中进行，以利于`Student`实例。在所有注册该`Course`的`Student`实例被通知之前，不应删除`Course`实例。通过引用计数保持指向特定`Course`实例的`Student`实例数量的计数器是有帮助的。在这个例子中，这就像维护一个变量来反映当前注册该`Course`的`Student`实例数量一样简单。

我们可以细致地自行进行链接维护，或者我们可能选择使用智能指针来管理关联对象的生存期。**智能指针**可以在 C++标准库中找到。它们封装了一个指针（即，在类中包装一个指针）以添加智能功能，包括引用计数和内存管理。由于智能指针使用模板，而我们将不会在*第十三章*“使用模板”中介绍模板，我们在这里只提及其潜在用途。

我们现在已经看到了反向链接维护的重要性以及引用计数在完全支持关联及其成功实现中的实用性。在继续下一章之前，让我们简要回顾一下本章中涵盖的面向对象概念——关联、聚合和组合。

# 摘要

在本章中，我们通过探索各种对象关系——关联、聚合和组合——来继续我们面向对象编程的追求。我们已经理解了代表这些关系的各种 OO 设计概念，并看到 C++不通过关键字或特定语言特性直接提供语言支持来实现这些概念。

尽管如此，我们已经学习了实现这些核心 OO 关系的技术，例如用于组合的内嵌对象和泛化聚合，或使用指针来实现关联。我们研究了这些关系下对象典型存在期限；例如，在聚合中，通过创建和销毁其内部部分（通过内嵌对象，或者更少的情况下，通过分配和释放指针成员）。或者通过关联对象的独立存在，这些对象既不创建也不销毁对方。我们还深入研究了实现关联（特别是具有多重性的关联）所需的维护工作，特别是通过检查反向链接维护和引用计数。

通过理解如何实现关联、聚合和组合，我们已经增加了我们的 OOP 技能的关键特性。我们看到了这些关系如何在 OO 设计中比继承更加普遍的例子。通过掌握这些技能，我们已经完成了在 C++中实现基本 OO 概念的技能集。

我们现在准备继续前进到*第十一章*，*处理异常*，这将开始我们扩展 C++编程技能库的探索。让我们继续前进！

# 问题

1.  在本章的`University`/`Student`示例中添加一个额外的`Student`构造函数，以便通过引用接受`University`构造函数参数，而不是通过指针。例如，除了具有签名`Student::Student(const string &fn, const string &ln, char mi, const string &t, float avg, const string &course, const string &id, University *univ);`的构造函数外，还可以重载此函数，但最后一个参数为`University &univ`。这种改变如何影响对这个构造函数的隐式调用？

提示：在你的重载构造函数中，你现在需要取`University`引用参数的地址（`&`）来设置关联（该关联以指针形式存储）。你可能需要切换到对象表示法（`.`）来设置反向链接（如果你使用参数`univ`，而不是数据成员`this->univ`）。

1.  编写一个 C++程序以实现`Course`类型和`Student`类型对象之间的多对多关联。你可以选择基于之前封装`Student`的程序进行构建。多对多关系应按以下方式工作：

    1.  特定的 `Student` 可以选择零到多个 `Course`，而特定的 `Course` 将与多个 `Student` 实例相关联。将 `Course` 类封装起来，至少包含课程名称、一组指向相关联的 `Student` 实例的指针和一个引用计数，以跟踪在 `Course` 中的 `Student` 实例数量（这将等同于指向特定 `Course` 实例的 `Student` 实例数量）。添加适当的接口以合理封装此类。

    1.  在 `Student` 类中添加一组指向该学生已注册的 `Course` 实例的指针。此外，跟踪特定学生当前注册的 `Course` 实例数量。添加适当的成员函数以支持这一新功能。

    1.  使用指针链表（即，数据部分是关联对象的指针）或关联对象的指针数组来模拟多方面的关联。请注意，数组将限制您可以拥有的关联对象数量，但这可能是合理的，因为特定的 `Course` 只能容纳一定数量的 `Student`，而学生每学期可能只能注册一定数量的 `Course`。如果您选择指针数组方法，请确保您的实现包括错误检查，以适应每个数组中关联对象数量的上限。

    1.  一定要检查简单的错误，例如尝试将 `Student` 添加到一个已满的 `Course` 中，或者将过多的 `Course` 添加到学生的课程表（假设每学期最多五门课程）中。

    1.  确保您的析构函数不会删除关联的实例。

    1.  引入至少三个 `Student` 对象，每个对象选择两个或更多的 `Course`。此外，确保每个 `Course` 有多个注册的学生。打印每个学生，包括他们注册的每个 `Course`。同样，打印每个 `Course`，显示注册在该 `Course` 中的每个学生。

1.  （可选）增强您的程序在 *练习 2* 中，以获得关于反向链接维护和引用计数的经验如下：

    1.  为 `Student` 实现 `DropCourse()` 接口。也就是说，在 `Student` 中创建一个 `Student::DropCourse(Course *)` 方法。在这里，找到学生想要从课程列表中删除的 `Course`，但在删除 `Course` 之前，调用该 `Course` 上的一个方法来从 `Course` 中删除上述 `Student`（即 `this`）。提示：您可以创建一个 `Course::RemoveStudent(Student *)` 方法来帮助进行反向链接删除。

    1.  现在，完全实现适当的析构函数。当 `Course` 被析构时，`Course` 析构函数首先告诉每个剩余的关联 `Student` 移除它们对该 `Course` 的链接。同样，当 `Student` 被析构时，遍历 `Student` 的课程列表，要求那些 `Courses` 从他们的学生列表中移除上述 `Student`（即 `this`）。您可能会发现每个类中的引用计数（即通过检查 `numStudents` 或 `numCourses`）对于确定是否需要执行这些任务很有帮助。

# 第三部分：扩展您的 C++ 编程曲目

本部分的目标是扩展您的 C++ 编程技能，不仅限于面向对象编程技能，还要涵盖 C++ 的其他关键特性。

本节的第一章通过理解 `try`、`throw` 和 `catch` 的机制，并通过检查许多示例来深入各种异常处理场景，来探索 C++ 中的异常处理。此外，本章还通过引入新的异常类来扩展异常类层次结构。

下一章深入探讨正确使用友元函数和友元类，以及运算符重载（有时可能需要友元），以使内置类型和用户定义类型之间的操作多态化。

下一章探讨使用 C++ 模板来帮助使代码通用并可用于各种数据类型，使用模板函数和模板类。此外，本章还解释了运算符重载如何帮助使模板代码对几乎所有数据类型都具有可扩展性。

在下一章中，将介绍 C++ 的标准模板库（Standard Template Library），并检查核心 STL 容器，如 `list`、`iterator`、`deque`、`stack`、`queue`、`priority_queue` 和 `map`（包括一个使用函数对象的例子）。此外，还将介绍 STL 算法和函数对象。

本节的最后一章通过探索规范类形式、创建组件测试驱动程序、测试通过继承、关联、聚合相关联的类以及测试异常处理机制，来概述测试面向对象程序和组件。

本部分包括以下章节：

+   *第十一章*，*处理异常*

+   *第十二章*，*友元和运算符重载*

+   *第十三章*，*使用模板*

+   *第十四章*，*理解 STL 基础*

+   *第十五章*，*测试类和组件*

第三部分：扩展您的 C++ 编程曲目
