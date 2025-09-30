

# 友元与操作符重载

本章将继续我们扩展你的 C++编程知识库的追求，目标是编写更可扩展的代码。我们将接下来探索 C++中的**友元函数**、**友元类**和**操作符重载**。我们将了解操作符重载如何将操作符的使用扩展到标准类型之外，以与用户定义的类型保持一致，以及为什么这是一个强大的面向对象编程工具。我们将学习如何安全地使用友元函数和类来实现这一目标。

在本章中，我们将涵盖以下主要主题：

+   理解友元函数和友元类，适当的使用理由以及增加其使用安全性的措施

+   了解操作符重载的基本知识——如何以及为什么重载操作符，确保操作符在标准类型和用户定义类型之间是多态的

+   实现操作符函数以及了解何时可能需要友元

到本章结束时，你将解锁友元的正确使用方法，并了解它们在利用 C++重载操作符的能力中的用途。尽管友元函数和类的使用可能被滥用，但你将坚持只在两个紧密耦合的类中仅限内部使用。你将了解正确使用友元如何增强操作符重载，使操作符能够扩展以支持用户定义的类型，从而与操作数关联操作。

让我们通过探索友元函数、友元类和操作符重载来扩展我们对 C++的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 中找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter12`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter12)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter12`子目录中找到，文件名为`Chp12-Ex1.cpp`。

本章的 CiA 视频可在以下网址查看：[`bit.ly/3K0f4tb`](https://bit.ly/3K0f4tb)。

# 理解友元类和友元函数

封装是 C++通过正确使用类和访问区域提供的有价值的面向对象编程（OOP）特性。封装提供了在处理数据和行为方面的统一性。一般来说，放弃类提供的封装保护是不明智的。

然而，在某些编程场景中，稍微打破封装性被认为比提供一个过于公开的类接口更可接受。也就是说，当一个类需要为两个类提供协作的方法时，尽管通常这些方法不适合公开访问。

让我们考虑一个可能导致我们考虑稍微放弃（即，打破）神圣的面向对象编程概念封装性的场景：

+   可能存在两个紧密耦合的类，它们之间没有其他关系。一个类可能与其他类有一个或多个关联，并需要操作另一个类的成员。然而，允许访问这些成员的公共接口会使这些内部成员过于公开，并容易受到超出紧密耦合类对需求之外的操纵。

+   在这种情况下，允许紧密耦合对中的其中一个类访问另一个类的成员，比在另一个类中提供一个允许比通常更安全地操作这些成员的公共接口更好。我们将很快看到如何最小化这种潜在的封装性损失。

+   我们将很快看到的某些选定的运算符重载场景可能需要实例在类作用域之外的函数中访问其成员。再次强调，一个完全可访问的公共接口可能被认为是危险的。

**友元函数**和**友元类**允许这种选择性的封装性打破。打破封装性是严重的，不应仅仅为了覆盖访问区域而进行。相反，当选择稍微打破两个紧密耦合类之间的封装性或提供一个过于公开的接口，该接口可能导致从应用程序的各个作用域对另一个类的成员有更大的、可能是不希望有的访问时，可以使用友元（并采取额外的安全措施）。

让我们看看每个如何使用，然后我们将添加我们应该坚持采用的相关安全措施。让我们从友元函数和友元类开始。

## 使用友元函数和友元类

**友元函数**是那些被个别授予扩展作用域以包括它们所关联的类的函数。让我们考察其影响和后勤：

+   在友元函数的作用域内，关联类型的实例可以访问其自己的成员，就像它们在自己的类作用域内一样。

+   一个友元函数需要在放弃访问权限（即扩展其作用域）的类的类定义中作为友元进行原型化。

+   关键字`friend`用于提供访问的函数原型之前。

+   函数重载友元函数不被视为友元。

**友元类**是那些其每个成员函数都是关联类的友元函数的类。让我们考察其后勤：

+   友元类应该在提供访问其成员（即作用域）的类的类定义中有一个前向声明。

+   关键字 `friend` 应该位于获得访问权限的类的前向声明之前（即，其作用域已被扩展）。

重要提示

友元类和友元函数应该谨慎使用，只有在选择性地稍微打破封装时，它才比提供一个过于公开的接口（即，一个在应用程序的任何作用域中都会提供对所选成员的不希望访问的公共接口）更好。

让我们从检查友元类和友元函数声明的语法开始。以下类不代表完整的类定义；然而，完整的程序可以在我们的 GitHub 仓库中找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex1.cpp)

```cpp
class Student;  // forward declaration of Student class
class Id  // Partial class – full class can be found online
{
private:
    string idNumber;
    Student *student = nullptr;  // in-class initialization
public:  // Assume constructors, destructor, etc. exist
    void SetStudent(Student *);
    // all member fns. of Student are friend fns to/of Id
    friend class Student;
};
// Note: Person class is as often defined; see online code
class Student : public Person
{
private:
    float gpa = 0.0;    // in-class initialization
    string currentCourse;
    static int numStudents;
    Id *studentId = nullptr;
public:   // Assume constructors, destructor, etc. exist
    // only the following mbr fn. of Id is a friend fn.
    friend void Id::SetStudent(Student *); // to/of Student
};
```

在前面的代码片段中，我们首先注意到 `Id` 类中的一个友元类定义。语句 `friend class Student;` 表示 `Student` 中的所有成员函数都是 `Id` 的友元函数。这个包含性的声明代替了将 `Student` 类的每个函数都命名为 `Id` 的友元函数。

此外，在 `Student` 类中，请注意 `friend void Id::SetStudent(Student *);` 的声明。这个友元函数声明表示，只有 `Id` 的这个特定成员函数是 `Student` 的友元函数。

友元函数原型 `friend void Id::SetStudent(Student *);` 的含义是，如果一个 `Student` 在 `Id::SetStudent()` 方法的范围内，那么这个 `Student` 可以像在自己的作用域内一样操作自己的成员，即 `Student` 的作用域。你可能想知道，哪个 `Student` 会发现自己处于 `Id::SetStudent(Student *)` 的作用域中？这很简单，就是作为输入参数传递给方法的那个。结果是，`Id::SetStudent()` 方法中的类型为 `Student *` 的输入参数可以像 `Student` 实例在自己的类作用域内一样访问其自己的私有和受保护的成员——它处于友元函数的作用域中。

类似地，`Id` 类中找到的友元类前向声明 `friend class Student;` 的含义是，如果任何 `Id` 实例发现自己处于 `Student` 方法中，那么这个 `Id` 实例可以像在自己的类中一样访问其自己的私有或受保护方法。`Id` 实例可能在其友元类 `Student` 的任何成员函数中，就像那些方法被扩展到也有 `Id` 类的作用域一样。

注意放弃访问权的类——即扩展范围的类——是宣布友情的类。也就是说，`Id`中的`friend class Student;`语句表示：如果任何`Id`恰好位于`Student`的任何成员函数中，允许该`Id`完全访问其成员，就像它在自己的作用域中一样。同样，`Student`中的友元函数语句表示，如果在这个特定的`Id`方法中找到一个`Student`实例（通过输入参数），它可以完全访问其元素，就像它在自己的类成员函数中一样。从增加作用域的角度考虑友情。

现在我们已经看到了友元函数和友元类的基本机制，让我们使用一个简单的协议来使其更具吸引力，以便选择性地破坏封装。

## 使用友元时，使访问更安全

我们已经看到，两个紧密耦合的类，例如通过关联相关联的类，可能需要稍微扩展它们的范围，以便通过使用**友元函数**或**友元类**来选择性地包含彼此。另一种选择是提供一个公共接口来选择每个类的元素。然而，考虑到你可能不希望这些元素的公共接口在应用程序的任何范围内都可以统一访问，你可以使用。你真正面临的是一个艰难的选择：利用友元还是提供一个**过于公开**的接口。

虽然一开始使用友元可能会让你感到不适，但它可能比提供对类元素的不希望公开的接口更安全。

为了减轻你对友元允许的选择性破坏封装所感到的恐慌，考虑将以下协议添加到你的友元使用中：

+   当使用友元时，为了减少封装的损失，一个类可以提供对另一个类数据成员的私有访问方法。考虑到这些方法是简单的访问方法（通常是单行方法，不太可能通过扩展增加软件膨胀），可以考虑将这些方法内联以提高效率。

+   被讨论的实例应同意仅使用在友元函数的作用域内适当访问其所需成员而创建的私有访问方法。这种非正式的理解当然是一种绅士协议，而不是语言强加的。

以下是一个简单的示例，说明如何使用`main()`函数和几个方法来适当地使用两个紧密耦合的类，为了节省空间，没有显示所有方法，完整的示例可以在我们的 GitHub 仓库中找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex2.cpp)

```cpp
using Item = int;  
class LinkList;  // forward declaration
class LinkListElement
{
private:
   void *data = nullptr;   // in-class initialization
   LinkListElement *next = nullptr;
   // private access methods to be used in scope of friend 
   void *GetData() const { return data; } 
   LinkListElement *GetNext() const { return next; }
   void SetNext(LinkListElement *e) { next = e; }
public:
// All member functions of LinkList are friend 
   // functions of LinkListElement 
   friend class LinkList;   
   LinkListElement() = default;
   LinkListElement(Item *i): data(i), next(nullptr) { }
   ~LinkListElement() { delete static_cast<Item *>(data); 
                        next = nullptr; }
};
// LinkList should only be extended as a protected/private
// base class; it does not contain a virtual destructor. It
// can be used as-is, or as implementation for another ADT.
class LinkList
{
private:
   LinkListElement *head = nullptr, *tail = nullptr, 
                   *current = nullptr;  // in-class init.
public:
   LinkList() = default;
   LinkList(LinkListElement *e) 
       { head = tail = current = e; }
   void InsertAtFront(Item *);
   LinkListElement *RemoveAtFront();  
   void DeleteAtFront()  { delete RemoveAtFront(); }
   bool IsEmpty() const { return head == nullptr; } 
   void Print() const;    // see online definition
   ~LinkList() { while (!IsEmpty()) DeleteAtFront(); }
};
```

让我们检查前面定义的`LinkListElement`和`LinkList`类。注意，在`LinkListElement`类中，我们有三个私有成员函数：`void *GetData();`、`LinkListElement *GetNext();`和`void SetNext(LinkListElement *);`。这三个成员函数不应成为公共类接口的一部分。只有当这些方法在`LinkList`的范围内使用时才是合适的，`LinkList`是与`LinkListElement`紧密耦合的类。

接下来，注意在`LinkListElement`类中的`friend class LinkList;`前向声明。这个声明意味着`LinkList`的所有成员函数都是`LinkListElement`的友元函数。因此，任何发现自己处于`LinkList`方法中的`LinkListElement`实例可以简单地访问它们之前提到的私有`GetData()`、`GetNext()`和`SetNext()`方法，因为它们将处于友元类的范围内。

接下来，让我们看看前面代码中的`LinkList`类。类定义本身没有关于友元的独特声明。毕竟，是`LinkListElement`类扩大了其作用域以包括`LinkedList`类的方法，而不是相反。

现在，让我们看看`LinkList`类的两个选定的成员函数。这些方法的完整集合可以在之前提到的 URL 上找到：

```cpp
void LinkList::InsertAtFront(Item *theItem)
{
   LinkListElement *newHead = new LinkListElement(theItem);
   // Note: temp can access private SetNext() as if it were
   // in its own scope – it is in the scope of a friend fn.
   newHead->SetNext(head);// same as: newHead->next = head;
   head = newHead;
}
LinkListElement *LinkList::RemoveAtFront()
{
   LinkListElement *remove = head;
   head = head->GetNext();  // head = head->next;
   current = head;    // reset current for usage elsewhere
   return remove;
}
```

在检查上述代码时，我们可以看到在`LinkList`方法的样本中，一个`LinkListElement`可以在友元函数（本质上是其自己的作用域的扩展）的作用域内调用其私有的方法。例如，在`LinkList::InsertAtFront()`中，`LinkListElement *temp`使用`temp->SetNext(head)`将其`next`成员设置为`head`。当然，我们也可以直接使用`temp->next = head;`来访问私有数据成员。然而，我们通过`LinkListElement`提供私有访问函数（如`SetNext()`），并要求`LinkList`方法（友元函数）让`temp`使用私有方法`SetNext()`，而不是直接操作数据成员本身，从而保持了一定的封装性。

由于`LinkListElement`中的`GetData()`、`GetNext()`和`SetNext()`是内联函数，我们通过提供封装访问成员`data`和`next`的感觉，并没有失去性能。

我们可以同样看到`LinkList`的其他成员函数，例如`RemoveAtFront()`（以及出现在在线代码中的`Print()`）使用`LinkListElement`实例利用其私有访问方法，而不是允许`LinkListElement`实例直接获取它们的私有`data`和`next`成员。

`LinkListElement` 和 `LinkList` 是两个紧密耦合的类的标志性例子，在这种情况下，最好扩展一个类以包含另一个类的访问范围，而不是提供一个**过于公开**的接口。毕竟，我们不想让 `main()` 中的用户能够接触到 `LinkListElement` 并应用 `SetNext()`，例如，这可能会在没有 `LinkList` 类知识的情况下改变整个 `LinkedList`。

既然我们已经了解了友元函数和类的机制以及建议的用法，让我们探索另一种可能需要使用友元的语言特性——操作符重载。

# 解密操作符重载的基本原理

C++ 语言中包含多种操作符。C++ 允许大多数操作符被重新定义，以便包括与用户定义类型的用法；这被称为**操作符重载**。通过这种方式，用户定义的类型可以利用与标准类型相同的表示法来执行这些已知的操作。我们可以将重载的操作符视为多态的，因为它的相同形式可以与各种类型（标准类型和用户定义类型）一起使用。

并非所有操作符都可以在 C++ 中重载。以下操作符不能重载：成员访问操作符 (`.`)、三元条件操作符 (`?:`)、作用域解析操作符 (`::`)、成员指针操作符 (`.*`)、`sizeof()` 操作符和 `typeid()` 操作符。所有其余的操作符都可以重载，前提是至少有一个操作数是用户定义类型。

在重载操作符时，重要的是要传达操作符对标准类型所具有的相同含义。例如，当与 `cout` 一起使用时，提取操作符 (`<<`) 被定义为打印到标准输出。此操作符可以应用于各种标准类型，例如整数、浮点数、字符字符串等。如果为用户定义的类型（如 `Student`）重载提取操作符 (`<<`)，它也应该意味着打印到标准输出。以这种方式，操作符 `<<` 在输出缓冲区（如 `cout`）的上下文中是多态的；也就是说，它对所有类型具有相同的意义但不同的实现。

重要的是要注意，在 C++ 中重载操作符时，我们可能不能改变操作符在语言中出现的预定义优先级。这很有意义——我们并不是在重写编译器以不同方式解析和解释表达式。我们只是在扩展操作符的含义，从它与标准类型的用法扩展到包括与用户定义类型的用法。操作符优先级将保持不变。

一个**操作符**，后跟表示你想要重载的操作符的符号。

让我们看看操作符函数原型的简单语法：

```cpp
Student &operator+(float gpa, const Student &s);
```

在这里，我们旨在提供一种使用 C++加法运算符（`+`）将浮点数和`Student`实例相加的方法。这种加法的意义可能是将新的浮点数与学生的现有平均成绩点数平均。在这里，运算符函数的名称是`operator+()`。

在上述原型中，运算符函数不是任何类的成员函数。左操作数预期为`float`类型，右操作数为`Student`类型。函数的返回类型（`Student &`）允许我们使用多个操作数级联使用`+`，或者与多个运算符配对，例如`s1 = 3.45 + s2;`。整体概念是，只要至少有一个操作数是用户定义类型，我们就可以定义如何使用`+`与多个类型一起使用。

实际上，涉及的内容远不止前面原型中显示的简单语法。在我们全面检查详细示例之前，让我们首先看看更多与实现运算符函数相关的后勤问题。

## 实现运算符函数以及知道何时可能需要友元

**运算符函数**，重载运算符的机制，可以作为一个成员函数或作为一个常规的外部函数实现。让我们以下列关键点总结实现运算符函数的机制：

+   作为成员函数实现的运算符函数将接收一个隐式参数（`this`指针），最多还有一个显式参数。如果重载操作中的左操作数是用户定义类型，并且可以轻松修改该类，则将运算符函数实现为成员函数是合理且首选的。

+   作为外部函数实现的运算符函数将接收一个或两个显式参数。如果重载操作中的左操作数是标准类型或不可修改的类类型，则必须使用外部（非成员）函数来重载此运算符。这个外部函数可能需要成为任何用作右手函数参数的对象类型的`friend`。

+   运算符函数通常应该相互实现。也就是说，在重载二元运算符时，确保它已经定义为无论数据类型（如果它们不同）在运算符中出现的顺序如何都能工作。

让我们看看一个完整的程序示例，以说明运算符重载的机制，包括成员函数和非成员函数，以及需要使用友元的场景。尽管为了节省空间，已经排除了程序的一些知名部分，但完整的程序示例可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter12/Chp12-Ex3.cpp)

```cpp
// Assume usual header files and std namespace inclusions
class Person
{
private: 
    string firstName, lastname;
    char middleInitial = '\0';
    char *title = nullptr; // use ptr member to demonstrate
                           // deep assignment
protected:
    void ModifyTitle(const string &); // converts to char *
public:                               
    Person() = default;   // default constructor
    Person(const string &, const string &, char, 
           const char *);  
    Person(const Person &);  // copy constructor
    virtual ~Person();  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; }  
    const string &GetLastName() const { return lastName; }    
    const char *GetTitle() const { return title; } 
    char GetMiddleInitial() const { return middleInitial; }
    virtual void Print() const;
    virtual void IsA() const;
    // overloaded operator functions
    Person &operator=(const Person &); // overloaded assign
    bool operator==(const Person &);   // overloaded
                                       // comparison
    Person &operator+(const string &); // overloaded plus
    // non-mbr friend fn. for op+ (to make associative)
    friend Person &operator+(const string &, Person &);  
};
```

让我们从查看`Person`类的先前定义开始，检查我们的代码。除了我们习惯看到的类元素之外，我们还有四个原型化的运算符函数：`operator=()`, `operator==()`, 和 `operator+()`（它被实现两次，以便`+`的运算数可以反转）。

`operator=()`, `operator==()`和`operator+()`的一个版本将被实现为这个类的成员函数，而另一个`operator+()`，具有`const char *`和`Person`参数，将被实现为一个非成员函数，并且还需要使用友元函数。

### 赋值运算符重载

让我们继续前进，检查这个类的适用运算符函数定义，首先从重载赋值运算符开始：

```cpp
// Assume the required constructors, destructor and basic
// member functions prototyped in the class def. exist.
// overloaded assignment operator
Person &Person::operator=(const Person &p)
{
    if (this != &p)  // make sure we're not assigning an 
    {                // object to itself
        // delete any previously dynamically allocated data 
        // from the destination object
        delete title;
        // Also, remember to reallocate memory for any 
        // data members that are pointers
        // Then, copy from source to destination object 
        // for each data member
        firstName = p.firstName;
        lastName = p.lastName;
        middleInitial = p.middleInitial;
        // Note: a pointer is used for title to demo the
        // necessary steps to implement a deep assignment -
        // otherwise, we would implement title with string
        title = new char[strlen(p.title) + 1]; // mem alloc
        strcpy(title, p.title);
    }
    return *this;  // allow for cascaded assignments
}
```

现在我们来回顾一下前面代码中重载的赋值运算符。它由成员函数`Person &Person::operator=(const Person &p);`指定。在这里，我们将从源对象分配内存，该源对象将是输入参数`p`，到目标对象，该对象将由`this`指向。

我们的首要任务是确保我们不是将对象赋值给自己。如果是这种情况，就没有工作要做！我们通过测试`if (this != &p)`来检查两个地址是否指向同一个对象。如果不是将对象赋值给自己，我们继续。

接下来，在条件语句（`if`）内部，我们首先释放由`this`指向的动态分配的数据成员所使用的现有内存。毕竟，赋值运算符左侧的对象是预先存在的，并且无疑为这些数据成员进行了分配。

现在，我们注意到条件语句中的核心代码看起来与复制构造函数非常相似。也就是说，我们仔细分配空间给指针数据成员，以匹配从输入参数`p`的相应数据成员所需的大小。然后，我们将输入参数`p`的适用数据成员复制到由`this`指向的数据成员。对于`char`数据成员`middleInitial`，不需要内存分配；我们只是使用赋值。对于`string`数据成员`firstName`和`lastName`也是如此。在这段代码中，我们确保我们对任何指针数据成员执行了深度赋值。浅拷贝（指针赋值），其中源对象和目标对象本应共享数据成员指针的数据部分内存，将会是一个即将发生的灾难。

最后，在我们的 `operator=()` 实现的末尾，我们返回 `*this`。请注意，此函数的返回类型是 `Person` 的引用。由于 `this` 是一个指针，我们只需取消引用它，以便返回一个可引用的对象。这样做是为了使 `Person` 实例之间的赋值可以级联；也就是说，`p1 = p2 = p3;` 其中 `p1`、`p2` 和 `p3` 分别是 `Person` 的实例。

注意

当重载 `operator=` 时，始终检查自赋值。也就是说，确保你不会将一个对象赋值给自己。在自赋值的情况下，实际上没有工作要做，但继续进行不必要的自赋值实际上可能会产生意外的错误！例如，如果我们有动态分配的数据成员，我们将释放目标对象的内存，并根据源对象内存的细节重新分配这些数据成员（当它们是同一个对象时，这些内存已经被释放）。这种行为可能是不可预测的，并且容易出错。

如果程序员希望禁止两个对象之间的赋值，可以在重载赋值运算符的原型中使用 `delete` 关键字，如下所示：

```cpp
    // disallow assignment
    Person &operator=(const Person &) = delete;
```

记住，重载的赋值运算符与拷贝构造函数有很多相似之处；对这两种语言特性都应采取相同的谨慎和注意。然而，请注意，赋值运算符将在进行两个现有对象之间的赋值时被调用，而拷贝构造函数是在创建新实例后隐式调用的初始化。在拷贝构造函数中，新实例使用现有实例作为其初始化的基础；同样，赋值运算符的左侧对象使用右侧对象作为其赋值的基础。

重要提示

重载的赋值运算符不会被派生类继承；因此，它必须由层次结构中的每个类定义。忽略为类重载 `operator=` 将迫使编译器为该类提供一个默认的浅拷贝赋值运算符；这对于包含指针数据成员的任何类都是危险的。

### 赋值运算符重载

接下来，让我们看看我们对重载的赋值运算符的实现：

```cpp
// overloaded comparison operator
bool Person::operator==(const Person &p)
{   
    // if the objects are the same object, or if the
    // contents are equal, return true. Otherwise, false.
    if (this == &p) 
        return true;
    else if ( (!firstName.compare(p.firstName)) &&
              (!lastName.compare(p.lastName)) &&
              (!strcmp(title, p.title)) &&
              (middleInitial == p.middleInitial) )
        return true;
    else
        return false;
}
```

继续使用我们之前的程序段，我们重载比较运算符。它由成员函数 `int Person::operator==(const Person &p);` 指定。在这里，我们将比较运算符右侧的 `Person` 对象，该对象将通过输入参数 `p` 进行引用，与运算符左侧的 `Person` 对象进行比较，该对象将由 `this` 指向。

同样，我们的首要任务是测试 `if (this != &p)` 中的对象是否指向同一个对象。如果两个地址都指向同一个对象，我们返回布尔值 (`bool`) 的 `true`。

接下来，我们检查两个 `Person` 对象是否包含相同的值。它们可能在内存中是分离的对象，但如果它们包含相同的值，我们同样可以选择返回 `bool` 值为 `true`。如果没有匹配，我们则返回 `bool` 值为 `false`。

### 将加法操作符作为成员函数重载

现在，让我们看看如何为 `Person` 和 `string` 重载 `operator+`：

```cpp
// overloaded operator + (member function)
Person &Person::operator+(const string &t)
{
    ModifyTitle(t);
    return *this;
}
```

继续使用前面的程序，我们将加法操作符 (`+`) 重载以与 `Person` 和 `string` 一起使用。操作符函数由成员函数原型 `Person& Person::operator+(const string &t);` 指定。参数 `t` 将代表 `operator+` 的右操作数，它是一个字符字符串（它将绑定到一个字符串的引用）。左操作数将由 `this` 指向。一个可能的用法是 `p1 + "Miss"`，其中我们希望使用 `operator+` 给 `Person` `p1` 添加一个 `title`。

在这个成员函数的主体中，我们仅仅将输入参数 `t` 作为 `ModifyTitle()` 的参数使用，即 `ModifyTitle(t);`。然后我们返回 `*this` 以便我们可以级联使用这个操作符（注意返回类型是 `Person &`）。

### 将加法操作符作为非成员函数重载（使用友元）

现在，让我们使用 `operator+` 反转操作数的顺序，以便使用 `string` 和 `Person`：

```cpp
// overloaded + operator (not a mbr function) 
Person &operator+(const string &t, Person &p)
{
    p.ModifyTitle(t);
    return p;
}
```

继续使用前面的程序，我们希望 `operator+` 不仅与 `Person` 和 `string` 一起工作，而且与操作数顺序相反，即与 `string` 和 `Person` 一起工作。没有理由这个操作符应该以一种方式工作，而不是另一种方式。

要完全实现 `operator+`，我们接下来重载 `operator+()` 以与 `const string &` 和 `Person` 一起使用。操作符函数由非成员函数 `Person& operator+(const string &t, Person &p);` 指定，它有两个显式输入参数。第一个参数 `t` 将代表 `operator+` 的左操作数，它是一个字符串（将此参数绑定到操作符函数中的第一个形式参数的字符串引用）。第二个参数 `p` 将是 `operator+` 中使用的右操作数的引用。一个可能的用法是 `"Miss" + p1`，其中我们希望使用 `operator+` 给 `Person p1` 添加一个头衔。注意，`"Miss"` 将使用 `std::string(const char *)` 构造函数构造为一个 `string`——字符串字面量只是字符串对象的初始值。

在这个非成员函数的函数体内，我们仅仅接受输入参数 `p` 并使用参数 `t` 指定的字符串字符调用受保护的 `ModifyTitle()` 方法，即 `p.ModifyTitle(t)`。然而，因为 `Person::ModifyTitle()` 是受保护的，`Person &p` 可能不能在 `Person` 的成员函数之外调用此方法。我们处于外部函数中；我们不在 `Person` 的作用域内。因此，除非这个成员函数是 `Person` 的 `friend`，否则 `p` 不能调用 `ModifyTitle()`。幸运的是，`Person &operator+(const string &, Person &);` 已经在 `Person` 类中作为友元函数原型，为 `p` 提供了必要的范围，允许它调用其受保护的方法。这就像 `p` 在 `Person` 的作用域内；它是在 `Person` 的友元函数的作用域内！

让我们继续前进到我们的 `main()` 函数，将我们之前提到的许多代码段结合起来，以便我们可以看到如何使用重载运算符调用我们的运算符函数：

```cpp
int main()
{
    Person p1;      // default constructed Person
    Person p2("Gabby", "Doone", 'A', "Miss");
    Person p3("Renee", "Alexander", 'Z', "Dr.");
    p1.Print();
    p2.Print();
    p3.Print();  
    p1 = p2;       // invoke overloaded assignment operator
    p1.Print();
    p2 = "Ms." + p2;  // invoke overloaded + operator
    p2.Print();       // then invoke overloaded = operator
    p1 = p2 = p3;     // overloaded = can handle cascaded =
    p2.Print();     
    p1.Print();
    if (p2 == p2)   // overloaded comparison operator
       cout << "Same people" << endl;
    if (p1 == p3)
       cout << "Same people" << endl;
   return 0;
}
```

最后，让我们检查前面程序中的 `main()` 函数。我们首先实例化三个 `Person` 实例，即 `p1`、`p2` 和 `p3`；然后使用每个实例的成员函数 `Print()` 打印它们的值。

现在，我们用语句 `p1 = p2;` 调用我们的重载赋值运算符。在底层，这翻译成以下运算符函数调用：`p1.operator=(p2);`。从这我们可以清楚地看到，我们正在调用之前定义的 `operator=()` 方法，该方法从源对象 `p2` 深度复制到目标对象 `p1`。我们使用 `p1.Print();` 来查看我们的复制结果。

接下来，我们用 `"Ms." + p2` 调用我们的重载 `operator+`。这一行代码的这一部分翻译成以下运算符函数调用：`operator+("Ms.", p2);`。在这里，我们简单地调用之前描述的 `operator+()` 函数，这是一个非成员函数，也是 `Person` 类的 `friend`。因为此函数返回一个 `Person &`，我们可以级联这个函数调用，使其看起来更像通常的加法上下文，并额外写出 `p2 = "Ms." + p2;`。在这整行代码中，首先调用 `operator+()` 对 `"Ms." + p2`。这个调用的返回值是 `p2`，然后被用作级联调用 `operator=` 的右操作数。注意，`operator=` 的左操作数也恰好是 `p2`。幸运的是，重载的赋值运算符检查自赋值。

现在，我们看到一个级联赋值 `p1 = p2 = p3;`。在这里，我们两次调用了重载的赋值运算符。首先，我们用 `p2` 和 `p3` 调用 `operator=`。翻译后的调用将是 `p2.operator=(p3);`。然后，使用第一次函数调用的返回值，我们再次调用 `operator=`。对于 `p1 = p2 = p3;` 的嵌套翻译调用看起来像这样：`p1.operator=(p2.operator=(p3));`。

最后在这个程序中，我们调用了两次重载的比较操作符。例如，每次比较`if (p2 == p2)`或`if (p1 == p3)`仅仅调用我们之前定义的`operator==`成员函数。回想一下，我们编写这个函数是为了报告对象在内存中相同或简单地包含相同的值时返回`true`，否则返回`false`。

让我们看看这个程序的输出：

```cpp
No first name No last name
Miss Gabby A. Doone
Dr. Renee Z. Alexander
Miss Gabby A. Doone
Ms. Gabby A. Doone
Dr. Renee Z. Alexander
Dr. Renee Z. Alexander
Same people
Same people
```

我们现在已经看到了如何指定和利用友元类和友元函数，如何重载 C++操作符，以及这些概念可以相互补充的情况。在我们继续前进到下一章之前，让我们简要回顾一下本章我们学到的特性。

# 摘要

在本章中，我们将 C++编程努力扩展到了面向对象语言特性之外，包括将使我们能够编写更可扩展程序的功能。我们已经学习了如何利用友元函数和友元类，我们也学习了如何在 C++中重载操作符。

我们已经看到，应该谨慎且少量地使用友元函数和类。它们不是为了提供一种明显的绕过访问区域的方法。相反，它们是为了处理编程情况，允许两个紧密耦合的类之间进行访问，而不在任一类中提供过于公开的接口，这可能在更广泛的范围内被滥用。

我们已经看到了如何使用操作符函数在 C++中重载操作符，无论是作为成员函数还是非成员函数。我们已经了解到，重载操作符将允许我们扩展 C++操作符的意义，使其包括用户定义的类型，就像它们包含标准类型一样。我们还看到，在某些情况下，友元函数或类可能很有用，可以帮助实现操作符函数，以便它们可以关联地行为。

通过探索友元和操作符重载，我们已经为我们的 C++工具箱添加了重要功能，后者将帮助我们确保我们很快将使用模板编写的代码可以用于几乎任何数据类型，从而有助于高度可扩展和可重用的代码。我们现在可以继续前进到*第十三章*，*使用模板*，这样我们就可以继续使用基本语言特性来扩展我们的 C++编程技能，这些特性将使我们成为更好的程序员。让我们继续前进吧！

# 问题

1.  在*第八章*，*掌握抽象类*的`Shape`练习中重载`operator=`，或者，也可以在你的正在进行的`LifeForm`/`Person`/`Student`类中重载`operator=`，如下所示：

在`Shape`（或`LifeForm`）中定义`operator=`，并在所有其派生类中重写此方法。提示：派生类中`operator=()`的实现将比其祖先做更多的工作，但仍可以调用祖先的实现来执行基类的工作部分。

1.  在你的 `Shape` 类（或 `LifeForm` 类）中重载 `operator<<` 操作符以打印每个 `Shape`（或 `LifeForm`）的信息。此函数的参数应该是一个 `ostream &` 和一个 `Shape &`（或一个 `LifeForm &`）。注意，`ostream` 来自 C++ 标准库（`using namespace std;`）。

你可以提供一个函数，`ostream &operator<<(ostream &, Shape &);`，并从该函数中调用在 `Shape` 中定义并在每个派生类中重新定义的多态 `Print()`，或者提供多个 `operator<<` 方法来实现此功能（每个派生类一个）。如果使用 `Lifeform` 层次结构，在上述 `operator<<` 函数签名中将 `LifeForm` 替换为 `Shape`。

1.  创建一个 `ArrayInt` 类以提供具有边界检查的安全整数数组。重载 `operator[]` 以返回数组中存在的元素，或者在元素不存在时抛出 `OutOfBounds` 异常。向 `ArrayInt` 添加其他方法，例如 `Resize()`、`RemoveElement()` 等。使用动态分配的数组（即使用 `int *contents`）来表示数组中的数据，这样你可以轻松地处理调整大小。代码可以从以下内容开始：

    ```cpp
    class ArrayInt // starting point for the class def.
    {          // be sure to add: using std::to_string;
    private:   // and also: using std::out_of_range;
        int numElements = 0;     // in-class init.
        int *contents = nullptr; // dynam. alloc. array
    public:
        ArrayInt(int size); // set numElements and
                            // allocate contents
        // returns a referenceable memory location or
        // throws an exception
        int &operator[](int index) 
        {             
            if (index < numElements) 
                return contents[index];
            else    // index is out of bounds
                throw std::out_of_range(
                                  std::to_string(index));
        }                        
    };
    int main()
    {
        ArrayInt a1(5); // Create ArrayInt of 5 elements
        try
        {
            a1[4] = 7;      // a1.operator[](4) = 7;
        }
        catch (const std::out_of_range &e)
        {
            cout << "Out of range: " << e.what() << endl;
        }
    }
    ```
