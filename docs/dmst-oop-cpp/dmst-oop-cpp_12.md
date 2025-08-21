# 第十章：实现关联、聚合和组合

本章将继续推进我们对 C++面向对象编程的了解。我们将通过探索关联、聚合和组合的面向对象概念来增进我们对对象关系的理解。这些 OO 概念在 C++中没有直接的语言支持；相反，我们将学习多种编程技术来实现这些想法。我们还将了解对于各种概念，哪些实现技术是首选的，以及各种实践的优势和缺陷。

关联、聚合和组合在面向对象设计中经常出现。了解如何实现这些重要的对象关系是至关重要的。

在本章中，我们将涵盖以下主要主题：

+   理解聚合和组合的 OO 概念，以及各种实现

+   理解关联的 OO 概念及其实现，包括反向链接维护的重要性和引用计数的实用性

通过本章的学习，您将了解关联、聚合和组合的 OO 概念，以及如何在 C++中实现这些关系。您还将了解许多必要的维护方法，如引用计数和反向链接维护，以保持这些关系的最新状态。尽管这些概念相对简单，但您将看到为了保持这些类型的对象关系的准确性，需要大量的簿记工作。

通过探索这些核心对象关系，让我们扩展对 C++作为面向对象编程语言的理解。

# 技术要求

完整程序示例的在线代码可在以下 GitHub 链接找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，文件名与所在章节编号相对应，后跟破折号，再跟随所在章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp10-Ex1.cpp`的文件中的子目录`Chapter10`中找到，位于上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/3sag0RY`](https://bit.ly/3sag0RY)。

# 理解聚合和组合

面向对象的聚合概念在许多面向对象设计中出现。它与继承一样频繁，用于指定对象关系。**聚合**用于指定具有-一个、整体-部分以及在某些情况下的包含关系。一个类可以包含其他对象的聚合。聚合可以分为两类——*组合*以及一种不太严格和*泛化*的聚合形式。

**泛化聚合**和**组合**都意味着具有-一个或整体-部分关系。然而，两者在两个相关对象之间的存在要求上有所不同。对于泛化聚合，对象可以独立存在；但对于组合，对象不能没有彼此存在。

让我们来看看每种聚合的变体，从组合开始。

## 定义和实现组合

**组合**是聚合的最专业形式，通常是大多数 OO 设计师和程序员在考虑聚合时所想到的。组合意味着包含，并且通常与整体-部分关系同义——即整体由一个或多个部分组成。整体*包含*部分。具有-一个关系也适用于组合。

外部对象，或*整体*，可以由*部分*组成。通过组合，部分不存在于整体之外。实现通常是一个嵌入对象 - 也就是说，一个包含对象类型的数据成员。在极少数情况下，外部对象将包含对包含对象类型的指针或引用；然而，当发生这种情况时，外部对象将负责创建和销毁内部对象。包含的对象没有其外层没有目的。同样，外层也不是*理想*的完整，没有内部的，包含的部分。

让我们看一个通常实现的组合示例。该示例将说明包含 - `Student` *有一个* `Id`。更重要的是，我们将暗示`Id`是`Student`的一个必要部分，并且没有`Student`就不会存在。`Id`对象本身没有任何目的。如果它们不是给予它们目的的主要对象的一部分，`Id`对象根本不需要存在。同样，您可能会认为`Student`没有`Id`是不完整的，尽管这有点主观！我们将使用嵌入对象在*整体*中实现*部分*。

组合示例将被分成许多部分。虽然只显示了示例的部分，完整的程序可以在以下 GitHub 位置找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10/Chp10-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10/Chp10-Ex1.cpp)

```cpp
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace std;
class Id  // the contained 'part'
{
private:
    char *idNumber;
public:
    Id() { idNumber = 0; }
    Id(const char *); 
    Id(const Id &);  
    ~Id() { delete idNumber; }
    const char *GetId() const { return idNumber; }
};
Id::Id(const char *id)
{
    idNumber = new char [strlen(id) + 1];
    strcpy(idNumber, id);
} 
Id::Id(const Id &id)
{
   idNumber = new char [strlen(id.idNumber) + 1];
   strcpy(idNumber, id.idNumber);
}
```

在前面的代码片段中，我们已经定义了一个`Id`类。`Id`将是一个可以被其他需要完全功能的`Id`的类包含的类。`Id`将成为可能选择包含它的*整体*对象的*部分*。

让我们继续构建一组最终将包含这个`Id`的类。我们将从一个我们熟悉的类`Person`开始：

```cpp
class Person
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
    virtual ~Person();  // virtual destructor
    const char *GetFirstName() const { return firstName; }
    const char *GetLastName() const { return lastName; }
    const char *GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    // virtual functions
    virtual void Print() const;   
    virtual void IsA();
    virtual void Greeting(const char *);
};
//  Assume the member functions for Person exist here
//  (they are the same as in previous chapters)
```

在先前的代码片段中，我们已经定义了`Person`类，就像我们习惯描述的那样。为了缩写这个示例，让我们假设伴随的成员函数存在于前述的类定义中。您可以在之前提供的 GitHub 链接中引用这些成员函数的在线代码。

现在，让我们定义我们的`Student`类。虽然它将包含我们习惯看到的元素，`Student`还将包含一个`Id`，作为一个嵌入对象：

```cpp
class Student: public Person  // 'whole' object
{
private:
    // data members
    float gpa;
    char *currentCourse;
    static int numStudents;  
    Id studentId;  // is composed of a 'part'
public:
    // member function prototypes
    Student();  // default constructor
    Student(const char *, const char *, char, const char *,
            float, const char *, const char *);
    Student(const Student &);  // copy constructor
    virtual ~Student();  // destructor
    void EarnPhD() { ModifyTitle("Dr."); } // various inline
    float GetGpa() const { return gpa; }         // functions
    const char *GetCurrentCourse() const
        { return currentCourse; }
    void SetCurrentCourse(const char *); // prototype only
    virtual void Print() const override;
    virtual void IsA() override { cout << "Student" << endl; }
    static int GetNumberStudents() { return numStudents; }
    // Access function for embedded Id object
    const char *GetStudentId() const;   // prototype only
};
int Student::numStudents = 0;  // static data member
inline void Student::SetCurrentCourse(const char *c)
{
    delete currentCourse;   // delete existing course
    currentCourse = new char [strlen(c) + 1];
    strcpy(currentCourse, c);
}
```

在前面的`Student`类中，我们经常注意到`Student`是从`Person`派生的。正如我们已经知道的那样，这意味着`Student`实例将包括`Person`的内存布局，作为`Person`子对象。

但是，请注意`Student`类定义中的数据成员`Id studentId;`。在这里，`studentId`是`Id`类型。它不是指针，也不是对`Id`的引用。数据成员`studentId`是一个嵌入对象。这意味着当实例化`Student`类时，不仅将包括从继承类中继承的内存，还将包括任何嵌入对象的内存。我们需要提供一种初始化嵌入对象`studentId`的方法。

让我们继续`Student`成员函数，以了解如何初始化，操作和访问嵌入对象：

```cpp
// constructor definitions
Student::Student(): studentId ("None") 
{
    gpa = 0.0;
    currentCourse = 0;
    numStudents++;
}
Student::Student(const char *fn, const char *ln, char mi,
                 const char *t, float avg, const char *course,
                 const char *id): Person(fn, ln, mi, t),
                 studentId(id)
{
    gpa = avg;
    currentCourse = new char [strlen(course) + 1];
    strcpy(currentCourse, course);
    numStudents++;
}
Student::Student(const Student &ps): Person(ps),
                 studentId(ps.studentId)
{
    gpa = ps.gpa;
    currentCourse = new char [strlen(ps.currentCourse) + 1];
    strcpy(currentCourse, ps.currentCourse);
    numStudents++;
}
Student::~Student()   // destructor definition
{
    delete currentCourse;
    numStudents--;
    // the embedded object studentId will also be destructed
}
void Student::Print() const
{
    cout << GetTitle() << " " << GetFirstName() << " ";
    cout << GetMiddleInitial() << ". " << GetLastName();
    cout << " with id: " << studentId.GetId() << " GPA: ";
    cout << setprecision(3) <<  " " << gpa;
    cout << " Course: " << currentCourse << endl;
}    
const char *GetStudentId() const 
{   
    return studentId.GetId();   
} 
```

在`Student`的先前列出的成员函数中，让我们从我们的构造函数开始。请注意，在默认构造函数中，我们利用成员初始化列表（`:`）来指定`studentId("None")`。因为`studentId`是一个成员对象，我们有机会选择（通过成员初始化列表）应该用于其初始化的构造函数。在这里，我们仅仅选择具有`Id(const char *)`签名的构造函数。

类似地，在`Student`的替代构造函数中，我们使用成员初始化列表来指定`studentId(id)`，这也将选择`Id(const char *)`构造函数，将参数`id`传递给此构造函数。

`Student`的复制构造函数还指定了如何使用成员初始化列表中的`studentId(ps.studentId)`来初始化`studentId`成员对象。在这里，我们只是调用了`Id`的复制构造函数。

在我们的`Student`析构函数中，我们不需要释放`studentId`。因为这个数据成员是一个嵌入对象，当外部对象的内存消失时，它的内存也会消失。当然，因为`studentId`本身也是一个对象，它的析构函数会在释放内存之前首先被调用。在幕后，编译器会（隐秘地）在`Student`析构函数的最后一行代码中补充一个对`studentId`的`Id`析构函数的调用。

最后，在前面的代码段中，让我们注意一下`studentId.GetId()`在`Student::Print()`和`Student::GetStudentId()`中的调用。在这里，嵌入对象`studentId`调用它自己的公共函数`Id::GetId()`来检索它在`Student`类作用域内的私有数据成员。因为`studentId`在`Student`中是私有的，所以这个嵌入对象只能在`Student`的作用域内被访问（也就是`Student`的成员函数）。然而，`Student::GetStudentId()`的添加为`Student`实例提供了一个公共的包装器，使得其他作用域中的`Student`实例可以检索这些信息。

最后，让我们来看一下我们的`main()`函数：

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

在上述的`main()`函数中，我们实例化了两个`Student`实例，`s1`和`s2`。当为每个`Student`创建内存（在这种情况下，是在堆栈上）时，任何继承类的内存也将被包含为子对象。此外，任何嵌入对象的内存，比如`Id`，也将被布置为`Student`的子对象。包含对象或*部分*的内存将与外部对象或*整体*的分配一起分配。

接下来，让我们注意一下对包含的部分，即嵌入的`Id`对象的访问。我们从调用`s1.GetStudentId()`开始；`s1`访问了一个`Student`成员函数`GetStudentId()`。这个学生成员函数将利用`studentId`的成员对象来调用`Id::GetId()`，从而访问`Id`类型的这个内部对象。`Student::GetStudentId()`成员函数可以通过简单地返回`Id::GetId()`在嵌入对象上返回的值来实现这种期望的公共访问。

让我们来看上述程序的输出：

```cpp
Cyrus Bond has id #: 6996CU
Anne Brennan has id #: 909EU 
```

这个例子详细介绍了组合及其典型实现，即嵌入对象。现在让我们来看一个使用较少的、替代的实现方式——继承。

### 考虑组合的另一种实现方式

值得理解的是，组合也可以用继承来实现；然而，这是极具争议的。记住，继承通常用于实现*是一个*关系，而不是*有一个*关系。我们在*第九章*中简要描述了使用继承来实现*有一个*关系，即*探索多重继承*。

简而言之，你只需从*部分*继承，而不是将部分作为数据成员嵌入。这样做时，你就不再需要为*部分*提供*包装器*函数，就像我们在前面的程序中看到的那样，`Student::GetStudentId()`方法调用`studentId.GetId()`来提供对其嵌入部分的访问。在嵌入对象的例子中，包装器函数是必要的，因为部分（`Id`）在整体（`Student`）中是私有的。程序员无法在`Student`的作用域之外访问`Student`的私有`studentId`数据成员。当然，`Student`的成员函数（如`GetStudentId()`）可以访问它们自己类的私有数据成员，并通过这样做来实现`Student::GetStudentId()`包装器函数，以提供这种（安全的）访问。

如果使用了继承，Id::GetId()的公共接口将会被简单地继承为 Student 的公共接口，无需通过嵌入对象显式地进行访问。

尽管在某些方面继承*部分*很简单，但它大大增加了多重继承的复杂性。我们知道多重继承可能会带来许多潜在的复杂性。此外，使用继承，*整体*只能包含一个*部分*的实例，而不是多个*部分*的实例。

此外，使用继承实现整体-部分关系可能会在将实现与 OO 设计进行比较时产生混淆。请记住，继承通常意味着 Is-A 而不是 Has-A。因此，最典型和受欢迎的聚合实现是通过嵌入对象。

接下来，让我们继续看一下更一般形式的聚合。

## 定义和实现泛化聚合

我们已经看过 OO 设计中最常用的聚合形式，即组合。特别是，通过组合，我们已经看到部分没有理由在没有整体的情况下存在。尽管如此，还存在一种更一般的（但不太常见）聚合形式，并且有时会在 OO 设计中进行指定。我们现在将考虑这种不太常见的聚合形式。

在**泛化聚合**中，*部分*可以存在而不需要*整体*。部分将被单独创建，然后在以后的某个时间点附加到整体上。当*整体*消失时，*部分*可能会留下来以供与另一个外部或*整体*对象一起使用。

在泛化聚合中，Has-A 关系当然适用，整体-部分的指定也适用。不同之处在于*整体*对象不会创建也不会销毁*部分*子对象。考虑一个简单的例子，汽车*Has-A(n)*发动机。汽车对象还*Has-A*一组 4 个轮胎对象。发动机或轮胎对象可以单独制造，然后传递给汽车的构造函数，以提供这些部分给整体。然而，如果发动机被销毁，可以轻松地替换为新的发动机（使用成员函数），而无需销毁整个汽车然后重新构建。

泛化聚合等同于 Has-A 关系，但我们认为这种关系比组合更灵活，个体部分的持久性更强。我们将这种关系视为聚合，只是因为我们希望赋予对象 Has-A 的含义。在“汽车”、“发动机”、“轮胎”的例子中，Has-A 关系很强；发动机和轮胎是必要的部分，需要组成整个汽车。

在这里，实现通常是*整体*包含指向*部分*（们）的指针。重要的是要注意，部分将被传递到外部对象的构造函数（或其他成员函数）中以建立关系。关键的标志是整体不会创建（也不会销毁）部分。部分也永远不会销毁整体。

顺便说一句，泛化聚合的个体部分的持久性（和基本实现）将类似于我们下一个主题 - 关联。让我们继续前进到我们的下一节，以了解泛化聚合和关联之间的相似之处以及 OO 概念上的差异（有时是微妙的）。

# 理解关联

**关联**模拟了存在于否则无关的类类型之间的关系。关联可以提供对象相互作用以实现这些关系的方式。关联不用于 Has-A 关系；然而，在某些情况下，我们描述的是*真正的*Has-A 关系，或者我们只是因为在语言上听起来合适而使用 Has-A 短语。

关联的多重性存在：一对一，一对多，多对一，或多对多。例如，一个`学生`可能与一个`大学`相关联，而那个`大学`可能与许多`学生`实例相关联；这是一对多的关联。

相关的对象具有独立的存在。也就是说，两个或更多的对象可以在应用程序的某个部分被实例化并独立存在。在应用程序的某个时刻，一个对象可能希望断言与另一个对象的依赖或关系。在应用程序的后续部分，相关的对象可能分道扬镳，继续各自独立的路径。

例如，考虑`课程`和`教师`之间的关系。一个`课程`与一个`教师`相关联。一个`课程`需要一个`教师`；一个`教师`对`课程`是必不可少的。一个`教师`可能与许多`课程`相关联。然而，每个部分都是独立存在的 - 一个不会创造也不会摧毁另一个。教师也可以独立存在而没有课程；也许一个教师正在花时间写书，或者正在休假，或者是一位进行研究的教授。

在这个例子中，关联非常类似于广义聚合。在这两种情况下，相关的对象也是独立存在的。在这种情况下，无论是说`课程`拥有`教师`还是`课程`对`教师`有依赖都可以是灰色的。你可能会问自己 - 是不是只是口头语言让我选择了“拥有”的措辞？我是不是指两者之间存在必要的联系？也许这种关系是一种关联，它的描述性修饰（进一步描述关联的性质）是*教*。你可能有支持任何选择的论点。因此，广义聚合可以被认为是关联的专门类型；我们将看到它们的实现是相同的，使用独立存在的对象。尽管如此，我们将区分典型关联作为对象之间明确不支持真正拥有关系的关系。

例如，考虑`大学`和`教师`之间的关系。我们可以考虑这种关系不是拥有关系，而是关联关系；我们可以认为描述这种关系的修饰是*雇用*。同样，`大学`与许多`学生`对象有关系。这里的关联可以用*教育*来描述。可以区分出`大学`由`系`对象，`楼`对象和这类组件组成，以支持其通过包含的拥有关系，然而它与`教师`对象，`学生`对象等的关系是使用关联来建立的。

既然我们已经区分了典型关联和广义聚合，让我们看看如何实现关联以及涉及的一些复杂性。

## 实现关联

通常，两个或更多对象之间的关联是使用指针或指针集来实现的。*一*方使用指向相关对象的指针来实现，而关系的*多*方则以指向相关对象的指针集合的形式实现。指针集合可以是指针数组，指针链表，或者真正的任何指针集合。每种类型的集合都有其自己的优点和缺点。例如，指针数组易于使用，可以直接访问特定成员，但项目数量是固定的。指针链表可以容纳任意数量的项目，但访问特定元素需要遍历其他元素以找到所需的项目。

偶尔，引用可能被用来实现关联的*one*一侧。请记住，引用必须被初始化，并且不能在以后被重置为引用另一个对象。使用引用来建模关联意味着一个实例将与另一个特定实例相关联，而主对象存在期间不能更改。这是非常限制性的；因此，引用很少用于实现关联。

无论实现方式如何，当主对象消失时，它都不会影响（即删除）关联的对象。

让我们看一个典型的例子，说明了首选的一对多关联实现，利用*one*一侧的指针和*many*一侧的指针集合。在这个例子中，一个`University`将与许多`Student`实例相关联。而且，为了简单起见，一个`Student`将与一个`University`相关联。

为了节省空间，本程序中与上一个示例相同的部分将不会显示；但是，整个程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10/Chp10-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter10/Chp10-Ex2.cpp)

```cpp
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace std;
const int MAX = 25;
// class Id and class Person are omitted here to save space.
// They will be as shown in previous example (Chp10-Ex1.cpp)
class Student; // forward declaration
class University
{
private:
    char *name;
    Student *studentBody[MAX]; // Association to many students
    int currentNumStudents;
    University(const University &);  // prohibit copies
public:
    University();
    University(const char *);
    ~University();
    void EnrollStudent(Student *);
    const char *GetName() const { return name; }
    void PrintStudents() const;
};
```

在前面的段落中，让我们首先注意`class Student;`的前向声明。这个声明允许我们的代码在`Student`类定义之前引用`Student`类型。在`University`类定义中，我们看到有一个指向`Student`的指针数组。我们还看到`EnrollStudent()`方法以`Student *`作为参数。前向声明使得在定义之前可以使用`Student`。

我们还注意到`University`具有一个简单的接口，包括构造函数、析构函数和一些成员函数。

接下来，让我们来看一下`University`成员函数的定义：

```cpp
University::University()
{
    name = 0;
    for (int i = 0; i < MAX; i++)  // the student body
       studentBody[i] = 0;         // will start out empty 
    currentNumStudents = 0;
}
University::University(const char *n)
{
    name = new char [strlen(n) + 1];
    strcpy(name, n);
    for (int i = 0; i < MAX; i++) // the student body will
       studentBody[i] = 0;        // start out empty
    currentNumStudents = 0;
}
University::~University()
{
    delete name;
    // The students will delete themselves
    for (int i = 0; i < MAX; i++)
       studentBody[i] = 0;  // only NULL out their link
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
    for (int i = 0; i < currentNumStudents; i++)
    {
       cout << "\t" << studentBody[i]->GetFirstName() << " ";
       cout << studentBody[i]->GetLastName() << endl;
    }
}
```

仔细观察前面的`University`方法，我们可以看到在`University`的两个构造函数中，我们只是将组成`studentBody`的指针`NULL`。同样，在析构函数中，我们也将与关联的`Students`的链接`NULL`。不久，在本节中，我们将看到还需要一些额外的反向链接维护，但现在的重点是我们不会删除关联的`Student`对象。

由于`University`对象和`Student`对象是独立存在的，因此它们之间既不会创建也不会销毁对方类型的实例。

我们还遇到了一个有趣的成员函数`EnrollStudent(Student *)`。在这个方法中，将传入一个指向特定`Student`的指针作为输入参数。我们只是索引到我们的`Student`对象指针数组`studentBody`中，并将一个未使用的数组元素指向新注册的`Student`。我们使用`currentNumStudents`计数器跟踪当前存在的`Student`对象数量，在指针分配后进行后置递增。

我们还注意到`University`有一个`Print()`方法，它打印大学的名称，然后是它当前的学生人数。它通过简单地访问`studentBody`中的每个关联的`Student`对象，并要求每个`Student`实例调用`Student::GetFirstName()`和`Student::GetLastName()`方法来实现这一点。

接下来，让我们来看一下我们的`Student`类定义，以及它的内联函数。请记住，我们假设`Person`类与本章前面看到的一样：

```cpp
class Student: public Person  
{
private:
    // data members
    float gpa;
    char *currentCourse;
    static int numStudents;
    Id studentId;  // part, Student Has-A studentId
    University *univ;  // Association to University object
public:
    // member function prototypes
    Student();  // default constructor
    Student(const char *, const char *, char, const char *,
            float, const char *, const char *, University *);
    Student(const Student &);  // copy constructor
    virtual ~Student();  // destructor
    void EarnPhD() { ModifyTitle("Dr."); }
    float GetGpa() const { return gpa; }
    const char *GetCurrentCourse() const 
        { return currentCourse; }
    void SetCurrentCourse(const char *); // prototype only
    virtual void Print() const override;
    virtual void IsA() override { cout << "Student" << endl; }
    static int GetNumberStudents() { return numStudents; }
    // Access functions for aggregate/associated objects
    const char *GetStudentId() const 
        { return studentId.GetId(); }
    const char *GetUniversity() const 
        { return univ->GetName(); }
};
int Student::numStudents = 0;  // def. of static data member
inline void Student::SetCurrentCourse(const char *c)
{
    delete currentCourse;   // delete existing course
    currentCourse = new char [strlen(c) + 1];
    strcpy(currentCourse, c);
}
```

在前面的代码段中，我们看到了`Student`类的定义。请注意，我们使用指针数据成员`University *univ;`与`University`关联。

在`Student`的类定义中，我们还可以看到有一个包装函数来封装对学生所在大学名称的访问，即`Student::GetUniversity()`。在这里，我们允许关联对象`univ`调用其公共方法`University::GetName()`，并将该值作为`Student::GetUniversity()`的结果返回。

现在，让我们来看看`Student`的非内联成员函数：

```cpp
Student::Student(): studentId ("None")
{
    gpa = 0.0;
    currentCourse = 0;  
    univ = 0;    // no current University association
    numStudents++;
}
Student::Student(const char *fn, const char *ln, char mi,
                 const char *t, float avg, const char *course,
                 const char *id, University *univ):
                 Person(fn, ln, mi, t), studentId(id)
{
    gpa = avg;
    currentCourse = new char [strlen(course) + 1];
    strcpy(currentCourse, course);
    // establish link to University, then back link
    this->univ = univ;  // required use of 'this'
    univ->EnrollStudent(this);  // another required 'this'
    numStudents++;
}
Student::Student(const Student &ps): 
                 Person(ps), studentId(ps.studentId)
{
    gpa = ps.gpa;
    currentCourse = new char [strlen(ps.currentCourse) + 1];
    strcpy(currentCourse, ps.currentCourse);
    this->univ = ps.univ;    
    univ->EnrollStudent(this);
    numStudents++;
}
Student::~Student()  // destructor
{
    delete currentCourse;
    numStudents--;
    univ = 0;  // the University will delete itself
    // the embedded object studentId will also be destructed
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

在前面的代码段中，请注意默认的`Student`构造函数和析构函数都只将它们与`University`对象的链接`NULL`。默认构造函数无法将此链接设置为现有对象，并且肯定不应该创建`University`实例来这样做。同样，`Student`析构函数不应该仅仅因为`Student`对象的寿命已经结束就删除`University`。

前面代码中最有趣的部分发生在`Student`的备用构造函数和复制构造函数中。让我们来看看备用构造函数。在这里，我们建立了与关联的`University`的链接，以及从`University`返回到`Student`的反向链接。

在代码行`this->univ = univ;`中，我们通过将数据成员`univ`（由`this`指针指向）设置为指向输入参数`univ`指向的位置来进行赋值。仔细看前面的类定义 - `University *`的标识符名为`univ`。此外，备用构造函数中`University *`的输入参数也被命名为`univ`。我们不能简单地在这个构造函数的主体中赋值`univ = univ;`。最本地范围内的`univ`标识符是输入参数`univ`。赋值`univ = univ;`会将该参数设置为自身。相反，我们使用`this`指针来消除赋值左侧的`univ`的歧义。语句`this->univ = univ;`将数据成员`univ`设置为输入参数`univ`。我们是否可以简单地将输入参数重命名为不同的名称，比如`u`？当然可以，但重要的是要理解在需要时如何消除具有相同标识符的输入参数和数据成员的歧义。

现在，让我们来看看下一行代码`univ->EnrollStudent(this);`。现在`univ`和`this->univ`指向同一个对象，无论使用哪一个来设置反向链接都没有关系。在这里，`univ`调用`EnrollStudent()`，这是`University`类中的一个公共成员函数。没有问题，`univ`的类型是`University`。`University::EnrollStudent(Student *)`期望传递一个指向`Student`的指针来完成`University`端的链接。幸运的是，在我们的`Student`备用构造函数中（调用函数的作用域），`this`指针是一个`Student *`。`this`就是我们需要创建反向链接的`Student *`。这是另一个需要显式使用`this`指针来完成手头任务的例子。

让我们继续前进到我们的`main()`函数：

```cpp
int main()
{
    University u1("The George Washington University");
    Student s1("Gabby", "Doone", 'A', "Miss", 3.85, "C++",
               "4225GWU", &u1);
    Student s2("Giselle", "LeBrun", 'A', "Ms.", 3.45, "C++",
               "1227GWU", &u1);
    Student s3("Eve", "Kendall", 'B', "Ms.", 3.71, "C++",
               "5542GWU", &u1);
    cout << s1.GetFirstName() << " " << s1.GetLastName();
    cout << " attends " << s1.GetUniversity() << endl;
    cout << s2.GetFirstName() << " " << s2.GetLastName();
    cout << " attends " << s2.GetUniversity() << endl;
    cout << s3.GetFirstName() << " " << s3.GetLastName();
    cout << " attends " << s2.GetUniversity() << endl;
    u1.PrintStudents();
    return 0;
}
```

最后，在我们的`main()`函数中的前面代码片段中，我们可以创建几个独立存在的对象，创建它们之间的关联，然后查看这种关系的实际情况。

首先，我们实例化一个`University`，即`u1`。接下来，我们实例化三个`Students`，`s1`，`s2`和`s3`，并将每个关联到`University u1`。请注意，当我们实例化一个`Student`时，可以设置这种关联，或者稍后进行设置，例如，如果`Student`类支持`SelectUniversity(University *)`接口来这样做。

然后，我们打印出每个`Student`，以及每个`Student`所就读的`University`的名称。然后我们打印出我们的`University u1`的学生人数。我们注意到，关联对象之间建立的链接在两个方向上都是完整的。

让我们来看看上述程序的输出：

```cpp
Gabby Doone attends The George Washington University
Giselle LeBrun attends The George Washington University
Eve Kendall attends The George Washington University
The George Washington University has the following students:
        Gabby Doone
        Giselle LeBrun
        Eve Kendall
```

我们已经看到了如何在相关对象之间轻松建立和利用关联。然而，从实现关联中会产生大量的维护工作。让我们继续了解引用计数和反向链接维护的必要和相关问题，这将有助于这些维护工作。

## 利用反向链接维护和引用计数

在前面的小节中，我们已经看到了如何使用指针来实现关联。我们已经看到了如何使用指向关联实例中的对象的指针来建立对象之间的关系。我们也看到了如何通过建立反向链接来完成循环的双向关系。

然而，与关联对象一样，关系是流动的，随着时间的推移会发生变化。例如，给定“大学”的“学生”群体会经常发生变化，或者“教师”将在每个学期教授的各种“课程”也会发生变化。因此，通常会删除特定对象与另一个对象的关联，并可能改为与该类的另一个实例关联。但这也意味着关联的对象必须知道如何删除与第一个提到的对象的链接。这变得复杂起来。

举例来说，考虑“学生”和“课程”的关系。一个“学生”可以注册多个“课程”实例。一个“课程”包含对多个“学生”实例的关联。这是一种多对多的关联。假设“学生”希望退出一门“课程”。仅仅让特定的“学生”实例移除指向特定“课程”实例的指针是不够的。此外，“学生”必须让特定的“课程”实例知道，应该将相关的“学生”从该“课程”的名单中移除。这被称为反向链接维护。

考虑一下，在上述情况下，如果一个“学生”简单地将其与要退出的“课程”的链接设置为`NULL`，然后不再进行任何操作，会发生什么。受影响的“学生”实例将不会有问题。然而，以前关联的“课程”实例仍将包含指向该“学生”的指针。也许这会导致“学生”在“教师”仍然认为该“学生”已注册但没有交作业的情况下获得不及格分数。最终，这位“学生”还是受到了影响，得到了不及格分数。

记住，对于关联的对象，一个对象在完成与另一个对象的交互后不会删除另一个对象。例如，当一个“学生”退出一门“课程”时，他们不会删除那门“课程” - 只是移除他们对相关“课程”的指针（并且肯定也要处理所需的反向链接维护）。

一个帮助我们进行整体链接维护的想法是考虑**引用计数**。引用计数的目的是跟踪有多少指针可能指向给定的实例。例如，如果其他对象指向给定的实例，那么该实例就不应该被删除。否则，其他对象中的指针将指向已释放的内存，这将导致大量的运行时错误。

让我们考虑一个具有多重性的关联。比如“学生”和“课程”之间的关系。一个“学生”应该跟踪有多少“课程”指针指向该“学生”，也就是说，该“学生”正在上多少门“课程”。只要有多个“课程”指向该“学生”，就不应该删除该“学生”。否则，“课程”将指向已删除的内存。处理这种情况的一种方法是在“学生”析构函数中检查对象（this）是否包含指向“课程”的非`NULL`指针。如果对象包含这样的指针，那么它需要通过每个活跃的“课程”调用一个方法，请求从每个这样的“课程”中移除对“学生”的链接。在移除每个链接之后，与“课程”实例集对应的引用计数可以递减。

同样，链接维护应该发生在`Course`类中，而不是`Student`实例中。在通知所有在该`Course`中注册的`Student`实例之前，不应删除`Course`实例。通过引用计数来跟踪有多少`Student`实例指向`Course`的特定实例是有帮助的。在这个例子中，只需维护一个变量来反映当前注册在`Course`中的`Student`实例的数量就可以了。

我们可以自己精心进行链接维护，或者选择使用智能指针来管理关联对象的生命周期。**智能指针**可以在 C++标准库中找到。它们封装了一个指针（即在类中包装一个指针）以添加智能特性，包括引用计数和内存管理。由于智能指针使用了模板，而我们直到*第十三章*，*使用模板*，我们才会涵盖，所以我们在这里只是提到了它们的潜在实用性。

我们现在已经看到了后向链接维护的重要性，以及引用计数的实用性，以充分支持关联及其成功的实现。在继续前进到下一章之前，让我们简要回顾一下本章涵盖的面向对象的概念——关联、聚合和组合。

# 总结

在本章中，我们通过探索各种对象关系——关联、聚合和组合，继续推进我们对面向对象编程的追求。我们已经理解了代表这些关系的各种面向对象设计概念，并且已经看到 C++并没有通过关键字或特定的语言特性直接提供语言支持来实现这些概念。

尽管如此，我们已经学会了几种实现这些核心面向对象关系的技术，比如使用嵌入对象来实现组合和广义聚合，或者使用指针来实现关联。我们已经研究了这些关系中对象存在的典型寿命，例如通过创建和销毁其内部部分（通过嵌入对象，或者更少见地通过分配和释放指针成员），或者通过相关对象的独立存在，它们既不创建也不销毁彼此。我们还深入研究了实现关联所需的内部工作，特别是那些具有多重性的关联，通过检查后向链接维护和引用计数。

通过理解如何实现关联、聚合和组合，我们已经为我们的面向对象编程技能增添了关键特性。我们已经看到了这些关系在面向对象设计中甚至可能比继承更为常见的例子。通过掌握这些技能，我们已经完成了在 C++中实现基本面向对象概念的核心技能组合。

我们现在准备继续到*第十一章*，*处理异常*，这将开始我们扩展 C++编程技能的探索。让我们继续前进！

# 问题

1.  在本章的`University`-`Student`示例中添加一个额外的`Student`构造函数，以接受引用而不是指针的`University`构造参数。例如，除了带有签名`Student::Student(const char *fn, const char *ln, char mi, const char *t, float avg, const char *course, const char *id, University *univ);`的构造函数外，重载此函数，但最后一个参数为`University &univ`。这如何改变对此构造函数的隐式调用？

提示：在您重载的构造函数中，您现在需要取`University`引用参数的地址（即`&`）来设置关联（存储为指针）。您可能需要切换到对象表示法（`.`）来设置后向链接（如果您使用参数`univ`，而不是数据成员`this->univ`）。

1.  编写一个 C++程序来实现“课程”类型对象和“学生”类型对象之间的多对多关联。您可以选择在之前封装“学生”的程序基础上构建。多对多关系应该按以下方式工作：

a. 给定的“学生”可以选修零到多门“课程”，而给定的“课程”将与多个“学生”实例关联。封装“课程”类，至少包含课程名称、指向关联“学生”实例的指针集，以及一个引用计数，用于跟踪在“课程”中的“学生”实例数量（这将等同于多少“学生”实例指向给定的“课程”实例）。添加适当的接口来合理封装这个类。

b. 在您的“学生”类中添加指向该“学生”注册的“课程”实例的指针集。此外，跟踪给定“学生”注册的“课程”实例数量。添加适当的成员函数来支持这种新功能。

c. 使用指针的链表（即，数据部分是指向关联对象的指针）或作为关联对象的指针数组来对多边关联进行建模。请注意，数组将对您可以拥有的关联对象数量施加限制；但是，这可能是合理的，因为给定的“课程”只能容纳最大数量的“学生”，而“学生”每学期只能注册最大数量的“课程”。如果您选择指针数组的方法，请确保您的实现包括错误检查，以适应每个数组中关联对象数量超过最大限制的情况。

d. 一定要检查简单的错误，比如尝试在已满的“课程”中添加“学生”，或者向“学生”的课程表中添加过多的“课程”（假设每学期最多有 5 门课程）。

e. 确保您的析构函数不会删除关联的实例。

f. 引入至少三个“学生”对象，每个对象都选修两门或更多门“课程”。此外，请确保每门“课程”都有多个“学生”注册。打印每个“学生”，包括他们注册的每门“课程”。同样，打印每门“课程”，显示注册在该“课程”中的每个“学生”。

1.  （可选）增强您在*练习 2*中的程序，以获得以下反向链接维护和引用计数的经验：

a. 为“学生”实现一个`DropCourse()`接口。也就是，在“学生”中创建一个“Student::DropCourse(Course *)”方法。在这里，找到“学生”希望在他们的课程列表中删除的“课程”，但在删除“课程”之前，调用该“课程”的一个方法，从该“课程”中删除前述的“学生”（即，`this`）。提示：您可以创建一个`Course::RemoveStudent(Student *)`方法来帮助删除反向链接。

b. 现在，完全实现适当的析构函数。当一个“课程”被销毁时，让“课程”析构函数首先告诉每个剩余的关联“学生”删除他们与该“课程”的链接。同样，当一个“学生”被销毁时，循环遍历“学生”的课程列表，要求那些“课程”从他们的学生列表中删除前述的“学生”（即，`this`）。您可能会发现每个类中的引用计数（即，通过检查`numStudents`或`numCourses`）有助于确定是否必须执行这些任务。
