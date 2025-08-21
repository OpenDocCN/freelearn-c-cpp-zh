# 第十二章：友元和运算符重载

本章将继续扩展你的 C++编程技能，超越 OOP 概念，目标是编写更具可扩展性的代码。接下来，我们将探索**友元函数**、**友元类**和**运算符重载**在 C++中的应用。我们将了解运算符重载如何将运算符扩展到与用户定义类型一致的行为，以及为什么这是一个强大的 OOP 工具。我们将学习如何安全地使用友元函数和类来实现这一目标。

在本章中，我们将涵盖以下主要主题：

+   理解友元函数和友元类，适当使用它们的原因，以及增加安全性的措施

+   学习运算符重载的基本要点——如何以及为何重载运算符，并确保运算符在标准类型和用户定义类型之间是多态的

+   实现运算符函数；了解何时需要友元

在本章结束时，您将掌握友元的正确使用，并了解它们在利用 C++重载运算符的能力方面的实用性。尽管可以利用友元函数和类的使用，但您将只学习它们在两个紧密耦合的类中的受限使用。您将了解如何正确使用友元可以增强运算符重载，使运算符能够扩展以支持用户定义类型，以便它们可以与其操作数关联工作。

让我们通过探索友元函数、友元类和运算符重载来扩展你的 C++编程技能，增进对 C++的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节编号相对应，后跟破折号，再跟所在章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp12-Ex1.cpp`的文件中的`Chapter12`子目录下找到。

本章的 CiA 视频可在以下网址观看：[`bit.ly/3f3tIm4`](https://bit.ly/3f3tIm4)。

# 理解友元类和友元函数

封装是 C++通过类和访问区域的正确使用提供的宝贵的 OOP 特性。封装提供了数据和行为被操作的统一方式。总的来说，放弃类提供的封装保护是不明智的。

然而，在某些编程情况下，略微破坏封装性被认为比提供一个*过度公开*的类接口更可接受，也就是说，当一个类需要为两个类提供合作的方法时，但总的来说，这些方法不适合公开访问时。

让我们考虑一个可能导致我们稍微放弃（即破坏）封装的情景：

+   可能存在两个紧密耦合的类，它们在其他方面没有关联。一个类可能与另一个类有一个或多个关联，并且需要操作另一个类的成员。然而，为了允许访问这些成员的公共接口会使这些内部*过度公开*，并且容易受到远远超出这对紧密耦合类的需求的操纵。

+   在这种情况下，允许紧密耦合的一对类中的一个类访问另一个类的成员比在另一个类中提供一个公共接口更好，这个公共接口允许对这些成员进行更多操作，而这通常是不安全的。我们将看到，如何最小化这种潜在的封装损失。

+   我们很快将看到，选定的运算符重载情况可能需要一个实例在其类作用域之外的函数中访问其成员。再次强调，一个完全可访问的公共接口可能被认为是危险的。

**友元函数**和**友元类**允许这种有选择性地打破封装。打破封装是严肃的，不应该简单地用来覆盖访问区域。相反，当在两个紧密耦合的类之间轻微打破封装或提供一个过度公开的接口时，可以使用友元函数和友元类，同时加入安全措施，这样做可能会从应用程序的各个作用域中获得更大且可能不受欢迎的对另一个类成员的访问。

让我们看一下如何使用每个，然后我们将添加我们应该坚持使用的相关安全措施。让我们从友元函数和友元类开始。

## 使用友元函数和友元类

**友元函数**是被单独授予*扩展作用域*的函数，以包括它们所关联的类。让我们来看一下其含义和具体情况：

+   在友元函数的作用域中，关联类型的实例可以访问自己的成员，就好像它在自己的类作用域中一样。

+   友元函数需要在放弃访问权限的类的类定义中作为友元进行原型声明（即扩展其作用域）。

+   关键字`friend`用于提供访问权限的原型前面。

+   重载友元函数的函数不被视为友元。

**友元类**是指该类的每个成员函数都是关联类的友元函数。让我们来看一下具体情况：

+   友元类应该在提供访问权限的类的类定义中进行前向声明（即作用域）。

+   关键字`friend`应该在获得访问权限的类的前向声明之前。

注意

友元类和友元函数应该谨慎使用，只有在有选择地和轻微地打破封装比提供一个*过度公开*的接口更好的选择时才使用（即一个普遍提供对应用程序中任何作用域中的选定成员的不受欢迎访问的公共接口）。

让我们首先来看一下友元类和友元函数声明的语法。以下类并不代表完整的类定义；然而，完整的程序可以在我们的在线 GitHub 存储库中找到，链接如下：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex1.cpp)

```cpp
class Student;  // forward declaration of Student class
class Id   
{
private:
    char *idNumber;
    Student *student;
public:  // Assume constructors, destructor, etc. exist
    void SetStudent(Student *);
    // all member functions of Student are friend fns to/of Id
    friend class Student;
};
class Student
{
private:
    char *name;
    float gpa;
    Id *studentId;
public:   // Assume constructors, destructor, etc. exist
    // only the following mbr function of Id is a friend fn.
    friend void Id::SetStudent(Student *);    // to/of Student
};
```

在前面的代码片段中，我们首先注意到了`Id`类中的友元类定义。语句`friend class Student;`表明`Student`中的所有成员函数都是`Id`的友元函数。这个包容性的语句用来代替将`Student`类的每个函数都命名为`Id`的友元函数。

另外，在`Student`类中，注意`friend void Id::SetStudent(Student *);`的声明。这个友元函数声明表明只有`Id`的这个特定成员函数是`Student`的友元函数。

友元函数原型`friend void Id::SetStudent(Student *);`的含义是，如果一个`Student`发现自己在`Id::SetStudent()`方法的范围内，那么这个`Student`可以操纵自己的成员，就好像它在自己的范围内一样，也就是`Student`的范围。你可能会问：哪个`Student`可能会发现自己在`Id::SetStudent(Student *)`的范围内？很简单。就是作为输入参数传递给方法的那个。结果是，在`Id::SetStudent()`方法中的`Student *`类型的输入参数可以访问自己的私有和受保护成员，就好像`Student`实例在自己的类范围内一样——它在友元函数的范围内。

同样，`Id`类中的友元类前向声明`friend class Student;`的含义是，如果任何`Id`实例发现自己在`Student`方法中，那么这个`Id`实例可以访问自己的私有或受保护方法，就好像它在自己的类中一样。`Id`实例可以在其友元类`Student`的任何成员函数中，就好像这些方法也扩展到了`Id`类的范围一样。

请注意，放弃访问的类——也就是扩大其范围的类——是宣布友谊的类。也就是说，在`Id`中的`friend class Student;`语句表示：如果任何`Id`恰好在`Student`的任何成员函数中，允许该`Id`完全访问其成员，就好像它在自己的范围内一样。同样，在`Student`中的友元函数语句表示：如果`Student`实例（通过输入参数）在`Id`的特定方法中被找到，它可以完全访问其元素，就好像它在自己类的成员函数中一样。以友谊作为扩大范围的手段来思考。

现在我们已经了解了友元函数和友元类的基本机制，让我们使用一个简单的约定来使其更具吸引力，以有选择地打破封装。

## 在使用友元时使访问更安全

我们已经看到，通过关联相关的两个紧密耦合的类可能需要通过使用**友元函数**或**友元类**来有选择地扩展它们的范围。另一种选择是为每个类提供公共接口。然而，请考虑您可能不希望这些元素的公共接口在应用程序的任何范围内都是统一可访问的。您确实面临着一个艰难的选择：使用友元或提供一个*过度公共*的接口。

虽然最初使用友元可能会让您感到不安，但这可能比提供不需要的公共接口给类元素更安全。

为了减少对友元允许的选择性打破封装的恐慌，考虑在使用友元时添加以下约定：

+   在使用友元时，为了减少封装的损失，一个类可以为另一个类的数据成员提供私有访问方法。尽可能将这些方法设置为内联，以提高效率。

+   问题实例应同意只使用创建的私有访问方法来适当地访问其所需的成员，而在友元函数的范围内（即使它实际上可以在友元函数的范围内无限制地访问自己类型的任何数据或方法）。

这里有一个简单的例子来说明两个紧密耦合的类如何适当地使用`main()`函数，为了节省空间，省略了几个方法，完整的例子可以在我们的 GitHub 存储库中找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex2.cpp)

```cpp
typedef int Item;  
class LinkList;  // forward declaration
class LinkListElement
{
private:
   void *data;
   LinkListElement *next;
   // private access methods to be used in scope of friend 
   void *GetData() { return data; } 
   LinkListElement *GetNext() { return next; }
   void SetNext(LinkListElement *e) { next = e; }
public:
// All mbr fns of LinkList are friend fns of LinkListElement 
   friend class LinkList;   
   LinkListElement() { data = 0; next = 0; }
   LinkListElement(Item *i) { data = i; next = 0; }
   ~LinkListElement(){ delete (Item *)data; next = 0;}
};
// LinkList should only be extended as a protected or private
// base class; it does not contain a virtual destructor. It
// can be used as-is, or as implementation for another ADT.
class LinkList
{
private:
   LinkListElement *head, *tail, *current;
public:
   LinkList() { head = tail = current = 0; }
   LinkList(LinkListElement *e) { head = tail = current = e; }
   void InsertAtFront(Item *);
   LinkListElement *RemoveAtFront();  
   void DeleteAtFront()  { delete RemoveAtFront(); }
   int IsEmpty() { return head == 0; } 
   void Print();    // see online definition
   ~LinkList() { while (!IsEmpty()) DeleteAtFront(); }
};
```

让我们来看看`LinkListElement`和`LinkList`的前面的类定义。请注意，在`LinkListElement`类中，我们有三个私有成员函数，即`void *GetData();`，`LinkListElement *GetNext();`和`void SetNext(LinkListElement *);`。这三个成员函数不应该是公共类接口的一部分。这些方法只适合在`LinkList`的范围内使用，这是与`LinkListElement`紧密耦合的类。

接下来，请注意`LinkListElement`类中的`friend class LinkList;`前向声明。这个声明意味着`LinkList`的所有成员函数都是`LinkListElement`的友元函数。因此，任何发现自己在`LinkList`方法中的`LinkListElement`实例都可以访问自己前面提到的私有`GetData()`，`GetNext()`和`SetNext()`方法，因为它们将在友元类的范围内。

接下来，让我们看看前面代码中的`LinkList`类。类定义本身没有与友好相关的唯一声明。毕竟，是`LinkListElement`类扩大了其范围以包括`LinkedList`类的方法，而不是相反。

现在，让我们来看一下`LinkList`类的两个选定的成员函数。这些方法的完整组合可以在网上找到，就像之前提到的 URL 中一样。

```cpp
void LinkList::InsertAtFront(Item *theItem)
{
   LinkListElement *temp = new LinkListElement(theItem);
   // Note: temp can access private SetNext() as if it were
   // in its own scope – it is in the scope of a friend fn.
   temp->SetNext(head);  // same as: temp->next = head;
   head = temp;
}
LinkListElement *LinkList::RemoveAtFront()
{
   LinkListElement *remove = head;
   head = head->GetNext();  // head = head->next;
   current = head;    // reset current for usage elsewhere
   return remove;
}
```

当我们检查前面的代码时，我们可以看到在`LinkList`方法的抽样中，`LinkListElement`可以调用自己的私有方法，因为它在友元函数的范围内（本质上是自己的范围，扩大了）。例如，在`LinkList::InsertAtFront()`中，`LinkListElement *temp`使用`temp->SetNext(head)`将其`next`成员设置为`head`。当然，我们也可以直接在这里访问私有数据成员，使用`temp->next = head;`。但是，通过`LinkListElement`提供私有访问函数，如`SetNext()`，并要求`LinkList`方法（友元函数）让`temp`利用私有方法`SetNext()`，而不是直接操作数据成员本身，我们保持了封装的程度。

因为`LinkListElement`中的`GetData()`，`GetNext()`和`SetNext()`是内联函数，所以我们不会因为提供对成员`data`和`next`的封装访问而损失性能。

我们还可以看到`LinkList`的其他成员函数，比如`RemoveAtFront()`（以及在线代码中出现的`Print()`），都有`LinkListElement`实例利用其私有访问方法，而不是允许`LinkListElement`实例直接获取其私有的`data`和`next`成员。

`LinkListElement`和`LinkList`是两个紧密耦合的类的标志性示例，也许最好是扩展一个类以包含另一个类的范围，以便访问，而不是提供一个过度公开的接口。毕竟，我们不希望`main()`中的用户接触到`LinkListElement`并应用`SetNext()`，例如，这可能会在不知道`LinkList`类的情况下改变整个`LinkedList`。

现在我们已经看到了友元函数和类的机制以及建议的用法，让我们探索另一个可能需要利用友元的语言特性 - 运算符重载。

# 解密运算符重载要点

C++语言中有各种运算符。C++允许大多数运算符重新定义以包括与用户定义类型的使用；这被称为**运算符重载**。通过这种方式，用户定义的类型可以利用与标准类型相同的符号来执行这些众所周知的操作。我们可以将重载的运算符视为多态的，因为它的相同形式可以与各种类型 - 标准和用户定义的类型一起使用。

并非所有运算符都可以在 C++中重载。以下运算符无法重载：成员访问（`。`），三元条件运算符（`？：`），作用域解析运算符（`::`），成员指针运算符（`.*`），`sizeof（）`运算符和`typeid（）`运算符。其余的都可以重载，只要至少有一个操作数是用户定义的类型。

在重载运算符时，重要的是要促进与标准类型相同的含义。例如，当与`cout`一起使用时，提取运算符（`<<`）被定义为打印到标准输出。这个运算符可以应用于各种标准类型，如整数，浮点数和字符串。如果提取运算符（`<<`）被重载为用户定义的类型，如`Student`，它也应该意味着打印到标准输出。这样，运算符`<<`在输出缓冲区的上下文中是多态的；也就是说，对于所有类型，它具有相同的含义，但不同的实现。

重载 C++中的运算符时，重要的是要注意，我们不能改变语言中运算符的预定义优先级。这是有道理的 - 我们不是在重写编译器以解析和解释表达式。我们只是将运算符的含义从其与标准类型的使用扩展到包括与用户定义类型的使用。运算符优先级将保持不变。

运算符，后跟表示您希望重载的运算符的符号。

让我们来看看运算符函数原型的简单语法：

```cpp
Student &operator+(float gpa, const Student &s);
```

在这里，我们打算提供一种方法，使用 C++加法运算符（`+`）来添加一个浮点数和一个`Student`实例。这种加法的含义可能是将新的浮点数与学生现有的平均成绩进行平均。在这里，运算符函数的名称是`operator+()`。

在上述原型中，运算符函数不是任何类的成员函数。左操作数将是`float`，右操作数将是`Student`。函数的返回类型（`Student＆`）允许我们将`+`与多个操作数级联使用，或者与多个运算符配对使用，例如`s1 = 3.45 + s2;`。总体概念是我们可以定义如何使用`+`与多种类型，只要至少有一个操作数是用户定义的类型。

实际上，比上面显示的简单语法涉及的内容要多得多。在我们完全检查详细示例之前，让我们首先看一下与实现运算符函数相关的更多后勤事项。

## 实现运算符函数并知道何时可能需要友元

**运算符函数**，重载运算符的机制，可以作为成员函数或常规外部函数实现。让我们总结实现运算符函数的机制，以下是关键点：

+   作为成员函数实现的运算符函数将接收一个隐式参数（`this`指针），最多一个显式参数。如果重载操作中的左操作数是可以轻松修改类的用户定义类型，则将运算符函数实现为成员函数是合理且首选的。

+   作为外部函数实现的运算符函数将接收一个或两个显式参数。如果重载操作中的左操作数是不可修改的标准类型或类类型，则必须使用外部（非成员）函数来重载此运算符。这个外部函数可能需要是用作右操作数的任何对象类型的“友元”。

+   运算符函数通常应该被互相实现。也就是说，当重载二元运算符时，确保它已经被定义为可以工作，无论数据类型（如果它们不同）以何种顺序出现在运算符中。

让我们看一个完整的程序示例，以说明运算符重载的机制，包括成员和非成员函数，以及需要使用友元的情况。尽管为了节省空间，程序的一些众所周知的部分已被排除在外，但完整的程序示例可以在我们的 GitHub 存储库中找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter12/Chp12-Ex3.cpp)

```cpp
// Assume usual header files and std namespace
class Person
{
private: 
    char *firstName, *lastname, *title;
    char middleInitial;
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
    virtual void Print() const;
    // overloaded operator functions
    Person &operator=(const Person &);  // overloaded assign
    bool operator==(const Person &); // overloaded comparison
    Person &operator+(const char *); // overloaded plus
    // non-mbr friend fn. for operator+ (to make associative)
    friend Person &operator+(const char *, Person &);  
};
```

让我们从代码审查开始，首先查看前面的`Person`类定义。除了我们习惯看到的类元素之外，我们还有四个运算符函数的原型：`operator=()`、`operator==()`和`operator+()`，它被实现了两次 - 以便可以颠倒`+`的操作数。

`operator=()`、`operator==()`和`operator+()`的一个版本将作为此类的成员函数实现，而另一个`operator+()`，带有`const char *`和`Person`参数，将作为非成员函数实现，并且还需要使用友元函数。

### 重载赋值运算符

让我们继续检查此类的适用运算符函数定义，首先是重载赋值运算符：

```cpp
// Assume the required constructors, destructor and basic
// member functions prototyped in the class definition exist.
// overloaded assignment operator
Person &Person::operator=(const Person &p)
{
    if (this != &p)  // make sure we're not assigning an 
    {                // object to itself
        delete firstName;  // or call ~Person() to release
        delete lastName;   // this memory (unconventional)
        delete title; 
        firstName = new char [strlen(p.firstName) + 1];
        strcpy(firstName, p.firstName);
        lastName = new char [strlen(p.lastName) + 1];
        strcpy(lastName, p.lastName);
        middleInitial = p.middleInitial;
        title = new char [strlen(p.title) + 1];
        strcpy(title, p.title);
    }
    return *this;  // allow for cascaded assignments
}
```

现在让我们回顾一下前面代码中重载的赋值运算符。它由成员函数`Person &Person::operator=(const Person &p);`指定。在这里，我们将从源对象（输入参数`p`）分配内存到目标对象（由`this`指向）。

我们的首要任务是确保我们没有将对象分配给自身。如果是这种情况，就没有工作要做！我们通过测试`if (this != &p)`来检查这一点，看看两个地址是否指向同一个对象。如果我们没有将对象分配给自身，我们继续。

接下来，在条件语句（`if`）中，我们首先释放由`this`指向的动态分配的数据成员的现有内存。毕竟，赋值语句左侧的对象已经存在，并且无疑为这些数据成员分配了内存。

现在，我们注意到条件语句中的核心代码看起来与复制构造函数非常相似。也就是说，我们仔细为指针数据成员分配空间，以匹配输入参数`p`的相应数据成员所需的大小。然后，我们将适用的数据成员从输入参数`p`复制到由`this`指向的数据成员。对于`char`数据成员`middleInitial`，不需要内存分配；我们仅使用赋值。在这段代码中，我们确保已执行了深度赋值。浅赋值，其中源对象和目标对象否则会共享数据成员的内存部分的指针，将是一场等待发生的灾难。

最后，在我们对`operator=()`的实现结束时，我们返回`*this`。请注意，此函数的返回类型是`Person`的引用。由于`this`是一个指针，我们只需对其进行解引用，以便返回一个可引用的对象。这样做是为了使`Person`实例之间的赋值可以级联；也就是说，`p1 = p2 = p3;`其中`p1`、`p2`和`p3`分别是`Person`的实例。

注意

重载的赋值运算符不会被派生类继承，因此必须由层次结构中的每个类定义。如果忽略为类重载`operator=`，编译器将为该类提供默认的浅赋值运算符；这对于包含指针数据成员的任何类都是危险的。

如果程序员希望禁止两个对象之间的赋值，可以在重载的赋值操作符的原型中使用关键字`delete`。

```cpp
    // disallow assignment
    Person &operator=(const Person &) = delete;
```

有必要记住，重载的赋值操作符与复制构造函数有许多相似之处；对这两种语言特性都需要同样的小心和谨慎。然而，赋值操作符将在两个已存在对象之间进行赋值时被调用，而复制构造函数在创建新实例后隐式被调用进行初始化。对于复制构造函数，新实例使用现有实例作为其初始化的基础；同样，赋值操作符的左操作数使用右操作数作为其赋值的基础。

### 重载比较操作符

接下来，让我们看看我们对重载比较操作符的实现：

```cpp
// overloaded comparison operator
bool Person::operator==(const Person &p)
{   
    // if the objects are the same object, or if the
    // contents are equal, return true. Otherwise, false.
    if (this == &p) 
        return 1;
    else if ( (!strcmp(firstName, p.firstName)) &&
              (!strcmp(lastName, p.lastName)) &&
              (!strcmp(title, p.title)) &&
              (middleInitial == p.middleInitial) )
        return 1;
    else
        return 0;
}
```

继续我们之前程序的一部分，我们重载比较操作符。它由成员函数`int Person::operator==(const Person &p);`指定。在这里，我们将比较右操作数上的`Person`对象，它将由输入参数`p`引用，与左操作数上的`Person`对象进行比较，它将由`this`指向。

同样，我们的首要任务是测试`if (this != &p)`，看看两个地址是否指向同一个对象。如果两个地址指向同一个对象，我们返回`true`的布尔值。

接下来，我们检查两个`Person`对象是否包含相同的值。它们可能是内存中的不同对象，但如果它们包含相同的值，我们同样可以选择返回`true`的`bool`值。如果没有匹配，我们返回`false`的`bool`值。

### 作为成员函数重载加法操作符

现在，让我们看看如何为`Person`和`const char *`重载`operator+`：

```cpp
// overloaded operator + (member function)
Person &Person::operator+(const char *t)
{
    ModifyTitle(t);
    return *this;
}
```

继续前面的程序，我们重载加法操作符（`+`），用于`Person`和`const char *`。操作符函数由成员函数原型`Person& Person::operator+(const char *t);`指定。参数`t`代表`operator+`的右操作数，即一个字符串。左操作数将由`this`指向。一个例子是`p1 + "Miss"`，我们希望使用`operator+`给`Person p1`添加一个称号。

在这个成员函数的主体中，我们仅仅将输入参数`t`作为`ModifyTitle()`的参数使用，即`ModifyTitle(t);`。然后我们返回`*this`，以便我们可以级联使用这个操作符（注意返回类型是`Person &`）。

### 作为非成员函数重载加法操作符（使用友元）

现在，让我们颠倒`operator+`的操作数顺序，允许`const char *`和`Person`：

```cpp
// overloaded + operator (not a mbr function) 
Person &operator+(const char *t, Person &p)
{
    p.ModifyTitle(t);
    return p;
}
```

继续前面的程序，我们理想地希望`operator+`不仅适用于`Person`和`const char *`，还适用于操作数的顺序颠倒；也就是说，`const char *`和`Person`。没有理由这个操作符只能单向工作。

为了完全实现`operator+`，接下来我们将重载`operator+()`，用于`const char *`和`Person`。操作符函数由非成员函数`Person& operator+(const char *t, Person &p);`指定，有两个显式输入参数。第一个参数`t`代表`operator+`的左操作数，即一个字符串。第二个参数`p`是用于`operator+`的右操作数的引用。一个例子是`"Miss" + p1`，我们希望使用`operator+`给`Person p1`添加一个称号。

在这个非成员函数的主体中，我们只是取输入参数`p`，并使用参数`t`指定的字符串应用受保护的方法`ModifyTitle()`。也就是说，`p.ModifyTitle(t)`。然而，因为`Person::ModifyTitle()`是受保护的，`Person &p`不能在`Person`的成员函数之外调用这个方法。我们在一个外部函数中；我们不在`Person`的范围内。因此，除非这个成员函数是`Person`的`friend`，否则`p`不能调用`ModifyTitle()`。幸运的是，在`Person`类中已经将`Person &operator+(const char *, Person &);`原型化为`friend`函数，为`p`提供了必要的范围，使其能够调用它的受保护方法。就好像`p`在`Person`的范围内一样；它在`Person`的`friend`函数的范围内！

最后，让我们继续前进到我们的`main()`函数，将我们之前提到的许多代码段联系在一起，这样我们就可以看到如何调用我们的操作函数，利用我们重载的运算符：

```cpp
int main()
{
    Person p1;      // default constructed Person
    Person p2("Gabby", "Doone", 'A', "Miss");
    Person p3("Renee", "Alexander", 'Z', "Dr.");
    p1.Print();
    p2.Print();
    p3.Print();  
    p1 = p2;        // invoke overloaded assignment operator
    p1.Print();
    p2 = "Ms." + p2;   // invoke overloaded + operator
    p2.Print();        // then invoke overloaded =  operator
    p1 = p2 = p3;   // overloaded = can handle cascaded =
    p2.Print();     
    p1.Print();
    if (p2 == p2)   // overloaded comparison operator
       cout << "Same people" << endl;
    if (p1 == p3)
       cout << "Same people" << endl;
   return 0;
}
```

最后，让我们来检查一下前面程序的`main()`函数。我们首先实例化了三个`Person`的实例，即`p1`、`p2`和`p3`；然后我们使用成员函数`Print()`打印它们的值。

现在，我们用语句`p1 = p2;`调用了我们重载的赋值运算符。在底层，这转换成了以下的操作函数调用：`p1.operator=(p2);`。从这里，我们可以清楚地看到，我们正在调用之前定义的`Person`的`operator=()`方法，它从源对象`p2`深度复制到目标对象`p1`。我们应用`p1.Print();`来查看我们的复制结果。

接下来，我们使用重载的`operator+`来处理`"Ms." + p2`。这行代码的一部分转换成以下的操作函数调用：`operator+("Ms.", p2);`。在这里，我们简单地调用了之前描述的`operator+()`函数，这是一个`Person`类的非成员函数和`friend`。因为这个函数返回一个`Person &`，我们可以将这个函数调用级联，看起来更像是通常的加法上下文，并且额外地写成`p2 = "Ms." + p2;`。在这行完整的代码中，首先对`"Ms." + p2`调用了`operator+()`。这个调用的返回值是`p2`，然后被用作级联调用`operator=`的右操作数。注意到`operator=`的左操作数也恰好是`p2`。幸运的是，重载的赋值运算符会检查自我赋值。

现在，我们看到了`p1 = p2 = p3;`的级联赋值。在这里，我们两次调用了重载的赋值运算符。首先，我们用`p2`和`p3`调用了`operator=`。翻译后的调用将是`p2.operator=(p3);`。然后，使用第一个函数调用的返回值，我们将第二次调用`operator=`。`p1 = p2 = p3;`的嵌套、翻译后的调用看起来像`p1.operator=(p2.operator=(p3));`。

最后，在这个程序中，我们两次调用了重载的比较运算符。例如，每次比较`if (p2 == p2)`或`if (p1 == p3)`只是调用了我们上面定义的`operator==`成员函数。回想一下，我们已经编写了这个函数，如果对象在内存中相同或者只是包含相同的值，就报告`true`，否则返回`false`。

让我们来看一下这个程序的输出：

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

我们现在已经看到了如何指定和使用友元类和友元函数，如何在 C++中重载运算符，以及这两个概念如何互补。在继续前往下一章之前，让我们简要回顾一下我们在本章学到的特性。

# 总结

在本章中，我们将我们的 C++编程努力进一步推进，超越了面向对象编程语言特性，包括了能够编写更具扩展性的程序的特性。我们已经学会了如何利用友元函数和友元类，以及如何在 C++中重载运算符。

我们已经看到友元函数和类应该谨慎使用。它们并不是为了提供一个明显的方法来绕过访问区域。相反，它们的目的是处理编程情况，允许两个紧密耦合的类之间进行访问，而不在这些类中的任何一个提供*过度公开*的接口，这可能会被广泛滥用。

我们已经看到如何在 C++中使用运算符函数重载运算符，既作为成员函数又作为非成员函数。我们已经了解到，重载运算符将允许我们扩展 C++运算符的含义，以包括用户定义类型，就像它们包含标准类型一样。我们还看到，在某些情况下，友元函数或类可能会派上用场，以帮助实现运算符函数，使其可以进行关联行为。

通过探索友元和运算符重载，我们已经为我们的 C++技能库添加了重要的功能，后者将帮助我们确保我们即将使用模板编写的代码可以用于几乎任何数据类型，从而为高度可扩展和可重用的代码做出贡献。我们现在准备继续前进到[*第十三章*]，*使用模板*，以便我们可以继续扩展我们的 C++编程技能，使用将使我们成为更好的程序员的基本语言特性。让我们继续前进！

# 问题

1.  在[*第八章*]（B15702_08_Final_NM_ePub.xhtml#_idTextAnchor335）的`Shape`练习中重载`operator=`，*掌握抽象类*，或者在你正在进行的`LifeForm`/`Person`/`Student`类中重载`operator=`如下：

a. 在`Shape`（或`LifeForm`）中定义`operator=`，并在其所有派生类中重写这个方法。提示：`operator=()`的派生实现将比其祖先做更多的工作，但可以调用其祖先的实现来执行基类部分的工作。

1.  在你的`Shape`类（或`LifeForm`类）中重载`operator<<`，以打印关于每个`Shape`（或`LifeForm`）的信息。这个函数的参数应该是`ostream &`和`Shape &`（或`LifeForm &`）。注意，`ostream`来自 C++标准库（`using namespace std;`）。

a. 你可以提供一个函数`ostream &operator<<(ostream &, Shape &);`，并从中调用多态的`Print()`，它在`Shape`中定义，并在每个派生类中重新定义），或者提供多个`operator<<`方法来实现这个功能（每个派生类一个）。如果使用`Lifeform`层次结构，将`Shape`替换为`LifeForm`。

1.  创建一个`ArrayInt`类，提供带边界检查的安全整数数组。重载`operator[]`，如果数组中存在元素，则返回该元素，否则抛出异常`OutOfBounds`。在你的`ArrayInt`中添加其他方法，比如`Resize()`和`RemoveElement()`。使用动态分配数组（即使用`int *contents`）来模拟数组的数据，这样你就可以轻松处理调整大小。代码将以以下方式开始：

```cpp
class ArrayInt
{
private:
    int numElements;
    int *contents;   // dynamically allocated array
public:
    ArrayInt(int size);// set numElements, alloc contents
    int &operator[](int index) // returns a referenceable
    {                          // memory location 
        if (index < numElements) return contents[index];
        else cout << "error"; // or throw OutOfBounds
    }                         // exception
};
int main()
{
    ArrayInt a1(5); // Create an ArrayInt of 5 elements
    A1[4] = 7;      // a1.operator[](4) = 7;
}
```
