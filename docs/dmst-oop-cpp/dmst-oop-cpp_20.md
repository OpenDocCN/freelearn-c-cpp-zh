# 第十六章：使用观察者模式

本章将开始我们的探索，将您的 C++编程技能库扩展到 OOP 概念之外，目标是使您能够通过利用常见的设计模式来解决重复出现的编码问题。设计模式还将增强代码维护，并为潜在的代码重用提供途径。本书的第四部分，从本章开始，旨在演示和解释流行的设计模式和习语，并学习如何在 C++中有效实现它们。

在本章中，我们将涵盖以下主要主题：

+   理解利用设计模式的优势

+   理解观察者模式及其对面向对象编程的贡献

+   理解如何在 C++中实现观察者模式

通过本章结束，您将了解在您的代码中使用设计模式的效用，以及了解流行的**观察者模式**。我们将在 C++中看到这种模式的示例实现。利用常见的设计模式将帮助您成为一个更有益和有价值的程序员，使您能够接纳更复杂的编程技术。

让我们通过研究各种设计模式来增强我们的编程技能，从本章开始使用观察者模式。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter16`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter16)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，该文件与所在章节编号对应，后跟破折号，再跟所在章节中示例编号。例如，本章的第一个完整程序可以在子目录`Chapter16`中的名为`Chp16-Ex1.cpp`的文件中找到，该文件位于上述 GitHub 目录下。

本章的 CiA 视频可以在以下链接观看：[`bit.ly/3vYprq2`](https://bit.ly/3vYprq2)。

# 利用设计模式

**设计模式**代表了针对重复出现的编程难题的一组经过充分测试的编程解决方案。设计模式代表了设计问题的高级概念，以及类之间的通用协作如何提供解决方案，可以以多种方式实现。

在过去 25 年多的软件开发中，已经识别和描述了许多设计模式。我们将在本书的剩余章节中查看一些流行的模式，以便让您了解如何将流行的软件设计解决方案纳入我们的编码技术库中。

为什么我们选择使用设计模式？首先，一旦我们确定了一种编程问题类型，我们可以利用其他程序员充分测试过的*经过验证的*解决方案。此外，一旦我们使用了设计模式，其他程序员在沉浸于我们的代码（用于维护或未来增强）时，将对我们选择的技术有基本的了解，因为核心设计模式已成为行业标准。

一些最早的设计模式大约 50 年前出现，随着**模型-视图-控制器**范式的出现，后来有时简化为**主题-视图**。例如，主题-视图是一个基本的模式，其中一个感兴趣的对象（**主题**）将与其显示方法（**视图**）松散耦合。主题及其视图之间有一对一的关联。有时主题可以有多个视图，这种情况下，主题与许多视图对象相关联。如果一个视图发生变化，状态更新可以发送到主题，然后主题可以向其他视图发送必要的消息，以便它们也可以更新以反映新状态可能如何修改它们的特定视图。

最初的**模型-视图-控制器**（**MVC**）模式，源自早期的面向对象编程语言，如 Smalltalk，具有类似的前提，只是控制器对象在模型（即主题）和其视图（或视图）之间委托事件。这些初步范例影响了早期的设计模式；主题-视图或 MVC 的元素在概念上可以被视为今天核心设计模式的基础。

我们将在本书的其余部分中审查的许多设计模式都是由*四人组*（Erich Gamma，Richard Helm，Ralph Johnson 和 John Vlissides）在*设计模式，可重用面向对象软件的元素*中最初描述的模式的改编。我们将应用和调整这些模式来解决我们在本书早期章节中介绍的应用程序所引发的问题。

让我们开始我们对理解和利用流行设计模式的追求，通过调查一个正在实施的模式。我们将从一个被称为**观察者模式**的行为模式开始。

# 理解观察者模式

在**观察者模式**中，一个感兴趣的对象将维护一个对主要对象状态更新感兴趣的观察者列表。观察者将维护与他们感兴趣的对象的链接。我们将主要感兴趣的对象称为**主题**。感兴趣的对象列表统称为**观察者**。主题将通知任何观察者相关状态的变化。一旦观察者被通知主题的任何状态变化，它们将自行采取任何适当的下一步行动（通常通过主题在每个观察者上调用的虚函数来执行）。

我们已经可以想象如何使用关联来实现观察者模式。事实上，观察者代表了一对多的关联。例如，主题可以使用 STL 的`list`（或`vector`）来收集一组观察者。每个观察者将包含与主题的关联。我们可以想象主题上的一个重要操作，对应于主题中的状态改变，发出对其观察者列表的更新，以*通知*它们状态的改变。`Notify()`方法实际上是在主题的状态改变时被调用，并统一地应用于主题的观察者列表上的多态观察者`Update()`方法。在我们陷入实现之前，让我们考虑构成观察者模式的关键组件。

观察者模式将包括：

+   主题，或感兴趣的对象。主题将维护一个观察者对象的列表（多边关联）。

+   主题将提供一个接口来`Register()`或`Remove()`一个观察者。

+   主题将包括一个`Notify()`接口，当主题的状态发生变化时，将更新其观察者。主题将通过在其集合中的每个观察者上调用多态的`Update()`方法来`Notify()`观察者。

+   观察者类将被建模为一个抽象类（或接口）。

+   观察者接口将提供一个抽象的、多态的`Update()`方法，当其关联的主题改变其状态时将被调用。

+   从每个 Observer 到其 Subject 的关联将在一个具体类中维护，该类派生自 Observer。这样做将减轻尴尬的转换（与在抽象 Observer 类中维护 Subject 链接相比）。

+   两个类将能够维护它们的当前状态。

上述的 Subject 和 Observer 类是通用指定的，以便它们可以与各种具体类（主要通过继承）结合使用观察者模式。通用的 Subject 和 Observer 提供了很好的重用机会。通过设计模式，模式的许多核心元素通常可以更通用地设置，以允许代码本身更大程度的重用，不仅是解决方案概念的重用。

让我们继续看观察者模式的一个示例实现。

# 实现观察者模式

为了实现观察者模式，我们首先需要定义我们的`Subject`和`Observer`类。然后，我们需要从这些类派生具体类，以合并我们的应用程序特定内容并启动我们的模式。让我们开始吧！

## 创建 Observer、Subject 和特定领域的派生类

在我们的示例中，我们将创建`Subject`和`Observer`类来建立*注册*`Observer`与`Subject`以及`Subject`通知其一组观察者可能存在的状态更改的框架。然后，我们将从这些基类派生出我们习惯看到的派生类 - `Course`和`Student`，其中`Course`将是我们的具体`Subject`，而`Student`将成为我们的具体`Observer`。

我们将建模的应用程序涉及课程注册系统和等待列表的概念。正如我们之前在*第十章*的*问题 2*中所看到的，*实现关联、聚合和组合*，我们将对`Student`进行建模，将其与许多`Course`实例关联，并且`Course`与许多`Student`实例关联。当我们建模我们的等待列表时，观察者模式将发挥作用。

我们的`Course`类将派生自`Subject`。我们的`Course`将继承的观察者列表将代表这个`Course`等待列表上的`Student`实例。 `Course`还将有一个`Student`实例列表，代表已成功注册该课程的学生。

我们的`Student`类将派生自`Person`和`Observer`。 `Student`将包括`Student`当前注册的`Course`实例列表。 `Student`还将有一个成员，`waitList`，它将对应于`Student`正在等待的`Course`的关联。这个*等待列表*`Course`代表我们将收到通知的`Subject`。通知将对应于状态更改，指示`Course`现在有空间让`Student`添加`Course`。

正是从`Observer`那里，`Student`将继承多态操作`Update()`，这将对应于`Student`被通知现在`Course`中有一个空位。在这里，在`Student::Update()`中，我们将添加机制，将`Student`从等待列表（有一个`waitList`数据成员）移动到`Course`中的实际当前学生列表（以及该学生的当前课程列表）。

### 指定 Observer 和 Subject

让我们将我们的示例分解成组件，从指定我们的`Observer`和`Subject`类开始。完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter16/Chp16-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter16/Chp16-Ex1.cpp)

```cpp
#include <list>   // partial list of #includes
#include <iterator>
using namespace std;
const int MAXCOURSES = 5, MAXSTUDENTS = 5;
class Subject;  // forward declarations
class Student;
class Observer  // Observer is an abstract class
{
private:
    int observerState;
protected:
    Observer() { observerState = 0; }
    Observer(int s) { observerState = s; }
    void SetState(int s) { observerState = s; }
public:
    int GetState() const { return observerState; }
    virtual ~Observer() {}
    virtual void Update() = 0;
};
```

在前面的类定义中，我们介绍了我们的抽象`Observer`类。在这里，我们包括一个`observerState`和受保护的构造函数来初始化这个状态。我们包括一个受保护的`SetState()`方法，以便从派生类的范围更新这个状态。我们还包括一个公共的`GetState()`方法。`GetState()`的添加将通过允许我们轻松检查`Observer`的状态是否已更改，有助于在我们的`Subject`的`Notify()`方法中实现。尽管状态信息历来是添加到`Observer`和`Subject`的派生类中，但我们将在这些基类中通用化状态信息。这将使我们的派生类保持更加独立于模式，并集中于应用程序的本质。

请注意，我们的析构函数是虚拟的，并且我们引入了一个抽象方法`virtual void Update() = 0;`来指定我们的`Subject`将在其观察者列表上调用的接口，以将更新委托给这些`Observer`实例。

现在，让我们来看看我们的`Subject`基类：

```cpp
class Subject   // Treated as an abstract class, due to
{               // protected constructors. However, there's no 
private:        // pure virtual function
    list<class Observer *> observers;
    int numObservers;
    int subjectState;
    list<Observer *>::iterator newIter;
protected:
    Subject() { subjectState = 0; numObservers = 0; }
    Subject(int s) { subjectState = s; numObservers = 0; }
    void SetState(int s) { subjectState = s; }
public:
    int GetState() const { return subjectState; }
    int GetNumObservers() const { return numObservers; }
    virtual ~Subject() {}
    virtual void Register(Observer *);
    virtual void Release(Observer *);
    virtual void Notify();
};
```

在上述的`Subject`类定义中，我们看到我们的`Subject`包括一个 STL`list`来收集它的`Observer`实例。它还包括`subjectState`和一个计数器来反映观察者的数量。此外，我们还包括一个数据成员来跟踪一个未损坏的迭代器。一旦我们擦除一个元素（`list::erase()`是一个会使当前迭代器失效的操作），我们将看到这将会很方便。

我们的`Subject`类还将具有受保护的构造函数和一个`SetState()`方法，该方法初始化或设置`Subject`的状态。虽然这个类在技术上不是抽象的（它不包含纯虚函数），但它的构造函数是受保护的，以模拟抽象类；这个类只打算作为派生类实例中的子对象来构造。

在公共接口中，我们有一些访问函数来获取当前状态或观察者的数量。我们还有一个虚析构函数，以及`Register()`、`Release()`和`Notify()`的虚函数。我们将在这个基类级别为后三个方法提供实现。

接下来让我们看看在我们的`Subject`基类中`Register()`、`Release()`和`Notify()`的默认实现。

```cpp
void Subject::Register(Observer *ob)
{
    observers.push_back(ob);   // Add an Observer to the list
    numObservers++;
}
void Subject::Release(Observer *ob) // Remove an Observer 
{                                   // from the list
    bool found;
    // loop until we find the desired Observer
    for (list<Observer *>::iterator iter = observers.begin();
         iter != observers.end() && !found; iter++)
    {
        Observer *temp = *iter;
        if (temp == ob)  // if we found observer which we seek
        {
            // erase() element, iterator is now corrupt; Save
            // returned (good) iterator, we'll need it later
            newIter = observers.erase(iter);
            found = true;  // exit loop after found
            numObservers--;
        }
    }
}
void Subject::Notify()
{   // Notify all Observers
    for (list<Observer *>::iterator iter = observers.begin(); 
         iter != observers.end(); iter++)
    {
        Observer *temp = *iter;
        temp->Update(); // AddCourse, then Release Observer   
        // State 1 means we added course, got off waitlist 
        // (waitlist had a Release), so update the iterator
        if (temp->GetState() == 1)
            iter = newIter;  // update the iterator since
    }                        // erase() invalidated this one
    if (observers.size() != 0)
    {   // Update last item on waitlist
        Observer *last = *newIter; 
        last->Update();
    }
}
```

在上述的`Subject`成员函数中，让我们从检查`void Subject::Register(Observer *)`方法开始。在这里，我们只是将指定的`Observer *`添加到我们的 STL 观察者列表中（并增加观察者数量的计数）。

接下来，让我们通过审查`void Subject::Release(Observer *)`来考虑`Register()`的反向操作。在这里，我们遍历观察者列表，直到找到我们正在寻找的观察者。然后我们在当前项目上调用`list::erase()`，将我们的`found`标志设置为`true`（以退出循环），并减少观察者的数量。还要注意，我们保存了`list::erase()`的返回值，这是更新的（有效的）观察者列表的迭代器。循环中的迭代器`iter`在我们调用`list::erase()`时已经失效。我们将这个修改后的迭代器保存在一个数据成员`newIter`中，以便稍后访问它。

最后，让我们来看看`Subject`中的`Notify()`方法。一旦`Subject`中有状态变化，就会调用这个方法。目标是`Update()`所有`Subject`观察者列表上的观察者。为了做到这一点，我们逐个查看我们的列表。我们使用`Observer *temp = *iter;`使用列表迭代器逐个获取`Observer`。我们使用`temp->Update();`在当前`Observer`上调用`Update()`。我们可以通过检查观察者的状态`if (temp->GetState() == 1)`来判断给定`Observer`的更新是否成功。状态为`1`时，我们知道观察者的操作将导致我们刚刚审查的`Release()`函数被调用。因为`Release()`中使用的`list::erase()`已经使迭代器无效，所以我们现在使用`iter = newIter;`获取正确和修订后的迭代器。最后，在循环外，我们在观察者列表中的最后一项上调用`Update()`。

### 从 Subject 和 Observer 派生具体类

让我们继续向前推进这个例子，看看我们从`Subject`或`Observer`派生的具体类。让我们从`Course`开始：

```cpp
class Course: public Subject  
{ // inherits Observer list; represents Students on wait-list
private:
    char *title;
    int number, totalStudents; // course num; total students
    Student *students[MAXSTUDENTS];  // students cur. enrolled
public:
    Course(const char *title, int num): number(num)
    {
        this->title = new char[strlen(title) + 1];
        strcpy(this->title, title);
        totalStudents = 0;
        for (int i = 0; i < MAXSTUDENTS; i++)
            students[i] = 0; 
    }
    virtual ~Course() { delete title; } // There's more work!
    int GetCourseNum() const { return number; }
    const char *GetTitle() const { return title; }
    void Open() { SetState(1); Notify(); } 
    void PrintStudents();
};
bool Course::AddStudent(Student *s)
{  // Should also check Student isn't already added to Course.
    if (totalStudents < MAXSTUDENTS)  // course not full
    {
        students[totalStudents++] = s;
        return true;
    }
    else return false;
}
void Course::PrintStudents()
{
    cout << "Course: (" << GetTitle() << ") has the following
             students: " << endl;
    for (int i = 0; i < MAXSTUDENTS && students[i] != 0; i++)
    {
        cout << "\t" << students[i]->GetFirstName() << " ";
        cout << students[i]->GetLastName() << endl;
    }
}
```

在上述的`Course`类中，我们包括了课程标题和编号的数据成员，以及当前已注册学生的总数。我们还有我们当前已注册学生的列表，用`Student *students[MAXNUMBERSTUDENTS];`表示。此外，请记住我们从基类`Subject`继承了`Observer`的 STL`list`。这个`Observer`实例列表将代表`Course`的等待列表中的`Student`实例。

`Course`类另外包括一个构造函数，一个虚析构函数和简单的访问函数。请注意，虚析构函数的工作比所示的更多 - 如果一个`Course`被销毁，我们必须首先记住从`Course`中删除（但不删除）`Student`实例。我们的`bool Course::AddStudent(Student *)`接口将允许我们向`Course`添加一个`Student`。当然，我们应该确保在这个方法的主体中`Student`尚未添加到`Course`中。

我们的`void Course::Open();`方法将在`Course`上调用，表示该课程现在可以添加学生。在这里，我们首先将状态设置为`1`（表示*开放招生*），然后调用`Notify()`。我们基类`Subject`中的`Notify()`方法循环遍历每个`Observer`，对每个观察者调用多态的`Update()`。每个观察者都是一个`Student`；`Student::Update()`将允许等待列表上的每个`Student`尝试添加现在可以接收学生的`Course`。成功添加到课程的当前学生列表后，`Student`将请求在等待列表上释放其位置（作为`Observer`）。

接下来，让我们来看看我们从`Person`和`Observer`派生的具体类`Student`的类定义：

```cpp
class Person { };  // Assume this is our typical Person class
class Student: public Person, public Observer
{
private:
    float gpa;
    const char *studentId;
    int currentNumCourses;
    Course *courses[MAXCOURSES]; // currently enrolled courses
    // Course we'd like to take - we're on the waitlist. 
    Course *waitList;// This is our Subject (specialized form)
public:
    Student();  // default constructor
    Student(const char *, const char *, char, const char *, 
            float, const char *, Course *);
    Student(const char *, const char *, char, const char *,
            float, const char *);
    Student(const Student &) = delete;  // Copies disallowed
    virtual ~Student();  
    void EarnPhD();
    float GetGpa() const { return gpa; }
    const char *GetStudentId() const { return studentId; }
    virtual void Print() const override;
    virtual void IsA() override;
    virtual void Update() override;
    virtual void Graduate();   // newly introduced virtual fn.
    bool AddCourse(Course *);
    void PrintCourses();
};
```

简要回顾上述`Student`类的类定义，我们可以看到这个类是通过多重继承从`Person`和`Observer`派生的。让我们假设我们的`Person`类就像我们过去多次使用的那样。

除了我们`Student`类的通常组件之外，我们还添加了数据成员`Course *waitList;`，它将模拟与我们的`Subject`的关联。这个数据成员将模拟我们非常希望添加的`Course`，但目前无法添加的*等待列表*课程的概念。请注意，这个链接是以派生类型`Course`而不是基本类型`Subject`声明的。这在观察者模式中很典型，并将帮助我们避免在`Student`中覆盖`Update()`方法时可怕的向下转换。通过这个链接，我们将与我们的`Subject`进行交互，并通过这种方式接收我们的`Subject`状态的更新。

我们还注意到在`Student`中有`virtual void Update() override;`的原型。这个方法将允许我们覆盖`Observer`指定的纯虚拟`Update()`方法。

接下来，让我们审查`Student`的各种新成员函数的选择：

```cpp
// Assume most Student member functions are as we are
// accustomed to seeing. Let's look at those which may differ:
Student::Student(const char *fn, const char *ln, char mi,
                const char *t, float avg, const char *id,
                Course *c) : Person(fn, ln, mi, t), Observer()
{
    // Most data members are set as usual - see online code 
    waitList = c;      // Set waitlist to Course (Subject)
    c->Register(this); // Add the Student (Observer) to 
}                      // the Subject's list of Observers
bool Student::AddCourse(Course *c)
{ 
    // Should also check that Student isn't already in Course
    if (currentNumCourses < MAXCOURSES)
    {
        courses[currentNumCourses++] = c;  // set association
        c->AddStudent(this);               // set back-link
        return true;
    }
    else  // if we can't add the course,
    {     // add Student (Observer) to the Course's Waitlist, 
        c->Register(this);  // stored in Subject base class
        waitList = c;// set Student (Observer) link to Subject
        return false;
    }
}
```

让我们回顾之前列出的成员函数。由于我们已经习惯了`Student`类中大部分必要的组件和机制，我们将专注于新添加的`Student`方法，从一个替代构造函数开始。在这里，让我们假设我们像往常一样设置了大部分数据成员。这里的关键额外代码行是`waitList = c;`将我们的等待列表条目设置为所需的`Course`（`Subject`），以及`c->Register(this);`，其中我们将`Student`（`Observer`）添加到`Subject`的列表（课程的正式等待列表）。

接下来，在我们的`bool Student::AddCourse(Course *)`方法中，我们首先检查是否已超过最大允许的课程数。如果没有，我们将通过机制来添加关联，以在两个方向上链接`Student`和`Course`。也就是说，`courses[currentNumCourses++] = c;`将学生当前的课程列表包含到新的`Course`的关联中，以及`c->AddStudent(this);`要求当前的`Course`将`Student`（`this`）添加到其已注册学生列表中。

让我们继续审查`Student`的其余新成员函数：

```cpp
void Student::Update()
{   // Course state changed to 'Open' so we can now add it.
    if (waitList->GetState() == 1)  
    {
        if (AddCourse(waitList))  // if success in Adding 
        {
            cout << GetFirstName() << " " << GetLastName();
            cout << " removed from waitlist and added to ";
            cout << waitList->GetTitle() << endl;
            SetState(1); // set Obser's state to "Add Success"
            // Remove Student from Course's waitlist
            waitList->Release(this); // Remove Obs from Subj
            waitList = 0;  // Set our link to Subject to Null
        }
    }
}
void Student::PrintCourses()
{
    cout << "Student: (" << GetFirstName() << " ";
    cout << GetLastName() << ") enrolled in: " << endl;
    for (int i = 0; i < MAXCOURSES && courses[i] != 0; i++)
        cout << "\t" << courses[i]->GetTitle() << endl;
}
```

继续我们之前提到的`Student`成员函数的其余部分，接下来，在我们的多态`void Student::Update()`方法中，我们进行了所需的等待列表课程添加。回想一下，当我们的`Subject`（`Course`）上有状态变化时，`Notify()`将被调用。这样的状态变化可能是当一个`Course`*开放注册*，或者可能是在`Student`退出`Course`后现在存在*新的空位可用*的状态。`Notify()`然后在每个`Observer`上调用`Update()`。我们在`Student`中重写了`Update()`来获取`Course`（`Subject`）的状态。如果状态表明`Course`现在*开放注册*，我们尝试`AddCourse(waitList);`。如果成功，我们将`Student`（`Observer`）的状态设置为`1`（*添加成功*），以表明我们在我们的`Update()`中取得了成功，这意味着我们已经添加了`Course`。接下来，因为我们已经将所需的课程添加到了我们当前的课程列表中，我们现在可以从`Course`的等待列表中移除自己。也就是说，我们将使用`waitList->Release(this);`将自己（`Student`）从`Subject`（`Course`的等待列表）中移除。现在我们已经添加了我们想要的等待列表课程，我们还可以使用`waitList = 0;`来移除我们与`Subject`的链接。

最后，我们上述的`Student`代码包括一个方法来打印`Student`当前注册的课程，即`void Student::PrintCourses();`。这个方法非常简单。

### 将模式组件组合在一起

让我们现在通过查看我们的`main()`函数来将所有各种组件组合在一起，看看我们的观察者模式是如何被编排的：

```cpp
int main()
{   // Instantiate several courses
    Course *c1 = new Course("C++", 230);  
    Course *c2 = new Course("Advanced C++", 430);
    Course *c3 = new Course("Design Patterns in C++", 550);
    // Instantiate Students, select a course to be on the 
    // waitlist for -- to be added when registration starts
    Student s1("Anne", "Chu", 'M', "Ms.", 3.9, "555CU", c1);
    Student s2("Joley", "Putt", 'I', "Ms.", 3.1, "585UD", c1);
    Student s3("Geoff", "Curt", 'K', "Mr.", 3.1, "667UD", c1);
    Student s4("Ling", "Mau", 'I', "Ms.", 3.1, "55UD", c1);
    Student s5("Jiang", "Wu", 'Q', "Dr.", 3.8, "883TU", c1);
    cout << "Registration is Open. Waitlist Students to be
             added to Courses" << endl;
    // Sends a message to Students that Course is Open. 
    c1->Open();   // Students on wait-list will automatically
    c2->Open();   // be Added (as room allows)
    c3->Open();
    // Now that registration is open, add more courses 
    cout << "During open registration, Students now adding
             additional courses" << endl;
    s1.AddCourse(c2);  // Try to add more courses
    s2.AddCourse(c2);  // If full, we'll be added to wait-list
    s4.AddCourse(c2);  
    s5.AddCourse(c2);  
    s1.AddCourse(c3);  
    s3.AddCourse(c3);  
    s5.AddCourse(c3);
    cout << "Registration complete\n" << endl;
    c1->PrintStudents();   // print each Course's roster
    c2->PrintStudents();
    c3->PrintStudents();
    s1.PrintCourses();     // print each Student's course list
    s2.PrintCourses();
    s3.PrintCourses();
    s4.PrintCourses();
    s5.PrintCourses();
    delete c1;
    delete c2;
    delete c3;
    return 0;
}
```

回顾我们之前提到的`main()`函数，我们首先实例化了三个`Course`实例。接下来，我们实例化了五个`Student`实例，利用一个构造函数，允许我们在课程注册开始时提供每个`Student`想要添加的初始`Course`。请注意，这些`Students`（`Observers`）将被添加到他们所需课程的等待列表（`Subject`）。在这里，一个`Subject`（`Course`）将有一个希望在注册开放时添加课程的`Observers`（`Students`）列表。

接下来，我们看到许多`Student`实例都希望的`Course`变为*开放注册*，使用`c1->Open();`进行注册。 `Course::Open()`将`Subject`的状态设置为`1`，表示课程*开放注册*，然后调用`Notify()`。正如我们所知，`Subject::Notify()`将在`Subject`的观察者列表上调用`Update()`。在这里，初始等待列表的`Course`实例将被添加到学生的日程表中，并随后从`Subject`的等待列表中作为`Observer`被移除。

现在注册已经开放，每个`Student`将尝试以通常的方式添加更多课程，比如使用`bool Student::AddCourse(Course *)`，比如`s1.AddCourse(c2);`。如果一个`Course`已满，该`Student`将被添加到`Course`的等待列表（作为继承的`Subject`的观察者列表）。记住，`Course`继承自`Subject`，它保留了对特定课程感兴趣的学生的列表（观察者的等待列表）。当`Course`状态变为*新空间可用*时，等待列表上的学生（观察者）将收到通知，并且每个`Student`的`Update()`方法随后将为该`Student`调用`AddCourse()`。

一旦我们添加了各种课程，我们将看到每个`Course`打印其学生名单，比如`c2->PrintStudents()`。同样，我们将看到每个`Student`打印他们所注册的课程，比如`s5.PrintCourses();`。

让我们来看一下这个程序的输出：

```cpp
Registration is Open. Waitlist Students to be added to Courses
Anne Chu removed from waitlist and added to C++
Goeff Curt removed from waitlist and added to C++
Jiang Wu removed from waitlist and added to C++
Joley Putt removed from waitlist and added to C++
Ling Mau removed from waitlist and added to C++
During open registration, Students now adding more courses
Registration complete
Course: (C++) has the following students:
        Anne Chu
        Goeff Curt
        Jiang Wu
        Joley Putt
        Ling Mau
Course: (Advanced C++) has the following students:
        Anne Chu
        Joley Putt
        Ling Mau
        Jiang Wu
Course: (Design Patterns in C++) has the following students:
        Anne Chu
        Goeff Curt
        Jiang Wu
Student: (Anne Chu) enrolled in:
        C++
        Advanced C++
        Design Patterns in C++
Student: (Joley Putt) enrolled in:
        C++
        Advanced C++
Student: (Goeff Curt) enrolled in:
        C++
        Design Patterns in C++
Student: (Ling Mau) enrolled in:
        C++
        Advanced C++
Student: (Jiang Wu) enrolled in:
        C++
        Advanced C++
        Design Patterns in C++
```

我们现在已经看到了观察者模式的实现。我们已经将更通用的`Subject`和`Observer`类折叠到了我们习惯看到的类的框架中，即`Course`、`Person`和`Student`。让我们现在简要回顾一下我们在模式方面学到的东西，然后继续下一章。

# 总结

在本章中，我们已经开始通过将我们的技能范围扩展到包括设计模式的利用，来使自己成为更好的 C++程序员。我们的主要目标是通过应用常见的设计模式来解决重复类型的编程问题，从而使您能够使用*经过验证的*解决方案。

我们首先理解了设计模式的目的，以及在我们的代码中使用它们的优势。然后，我们具体理解了观察者模式的前提以及它对面向对象编程的贡献。最后，我们看了一下如何在 C++中实现观察者模式。

利用常见的设计模式，比如观察者模式，将帮助您更轻松地解决其他程序员理解的重复类型的编程问题。面向对象编程的一个关键原则是尽可能地重用组件。通过利用设计模式，您将为更复杂的编程技术做出可重用的解决方案。

我们现在准备继续前进，进入我们下一个设计模式[*第十七章*]（B15702_17_Final_NM_ePub.xhtml#_idTextAnchor649），*实现工厂模式*。向我们的技能集合中添加更多的模式将使我们成为更多才多艺和受人重视的程序员。让我们继续前进！

# 问题

1.  使用本章示例中的在线代码作为起点，并使用之前练习的解决方案（*问题 3*，[*第十章*]（B15702_10_Final_NM_ePub.xhtml#_idTextAnchor386），*实现关联、聚合和组合*）：

a. 实现（或修改之前的）`Student::DropCourse()`。当一个`Student`退课时，这个事件将导致`Course`的状态变为状态`2`，*新空间可用*。状态改变后，`Course`（`Subject`）上的`Notify()`将被调用，然后`Update()`将更新观察者列表（等待列表上的学生）。间接地，`Update()`将允许等待列表上的`Student`实例，如果有的话，现在添加这门`Course`。

b. 最后，在`DropCourse()`中，记得从学生当前的课程列表中移除已经退课的课程。

1.  你能想象其他容易融入观察者模式的例子吗？
