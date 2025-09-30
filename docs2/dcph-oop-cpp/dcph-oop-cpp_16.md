

# 第十六章：使用观察者模式

本章将开始我们的探索之旅，旨在扩展您的 C++ 编程知识库，超越面向对象的概念，目标是让您能够通过利用常见的设计模式来解决重复出现的编程问题。设计模式还将增强代码维护性，并为潜在的代码重用提供途径。

从本章开始，本书的第四部分的目标是展示和解释流行的设计模式和惯用法，并学习如何在 C++ 中有效地实现它们。

本章将涵盖以下主要主题：

+   理解利用设计模式的优势

+   理解观察者模式及其对面向对象编程的贡献

+   理解如何在 C++ 中实现观察者模式

到本章结束时，您将理解在代码中采用设计模式的价值，以及理解流行的**观察者模式**。我们将通过 C++ 中的示例实现来展示这个模式。利用常见的设计模式将帮助您成为一个更有益和有价值的程序员，使您能够掌握更复杂的编程技术。

让我们通过研究各种设计模式来提高我们的编程技能集，从本章的观察者模式开始。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter16`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter16)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，文件名对应章节编号，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter16` 子目录中找到一个名为 `Chp16-Ex1.cpp` 的文件。

本章的 CiA 视频可以在以下网址观看：[`bit.ly/3A8ZWoy`](https://bit.ly/3A8ZWoy)。

# 利用设计模式

**设计模式**代表了一组经过良好测试的编程解决方案，用于解决重复出现的编程难题。设计模式代表了设计问题的概念层面，以及类之间如何进行通用协作以提供多种实现方式的解决方案。

在过去 25+ 年的软件开发中，已经识别和描述了许多公认的设计模式。我们将在本书的剩余章节中探讨一些流行的模式，以让您了解如何将流行的软件设计解决方案融入我们的技术编码库中。

我们为什么可能选择使用设计模式？首先，一旦我们确定了一种编程问题类型，我们可以使用其他程序员已经全面测试过的*经过验证的*解决方案。此外，一旦我们采用设计模式，其他沉浸在我们的代码中（用于维护或未来的增强）的程序员将对我们选择的技术有一个基本理解，因为核心设计模式已成为行业标准。

一些最早的设计模式几乎在 50 年前出现，随着**模型-视图-控制器**（Model-View-Controller，简称 MVC）范式的出现，后来有时简化为**主题-视图**。例如，主题-视图是一个基本的模式，其中感兴趣的物体（即**主题**）将与它的显示方法（即它的**视图**）松散耦合。主题和它的视图通过一对一的关联进行通信。有时主题可以有多个视图，在这种情况下，主题与多个视图对象相关联。如果一个视图发生变化，可以发送一个状态更新到主题，然后主题可以发送必要的消息到其他视图，以便它们也能更新以反映新的状态可能对其特定视图的修改。

来自早期面向对象语言（如 Smalltalk）的原始**模型-视图-控制器**（MVC）模式有一个类似的假设，只不过是一个控制器对象在模型（即主题）和它的视图（或视图）之间委派事件。这些初步范式影响了早期的设计模式；主题-视图或 MVC 的元素可以从概念上被视为今天核心设计模式的基本基础。

在本书的剩余部分，我们将回顾的许多设计模式将是*四人帮*（Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides）在《设计模式：可复用面向对象软件元素》中最初描述的模式的改编。我们将应用和改编这些模式来解决本书早期章节中介绍的应用程序产生的问题。

让我们通过调查一个实际应用中的模式来开始我们对理解和利用流行设计模式的追求。我们将从一个被称为**观察者模式**的行为模式开始。

# 理解观察者模式

在**观察者模式**中，一个感兴趣的物体将维护一个观察者的列表，这些观察者对主要物体的状态更新感兴趣。观察者将维护对其感兴趣物体的链接。我们将把感兴趣的物体称为**主题**。感兴趣物体的列表统称为**观察者**。主题将通知任何观察者相关的状态变化。一旦观察者被通知主题的任何状态变化，它们将自行采取任何适当的后续行动（通常是通过主题在每个观察者上调用虚拟函数来实现）。

已经，我们可以想象如何使用关联来实现观察者模式。事实上，观察者代表了一对多关联。例如，主题可能使用 STL `list`（或`vector`）来收集一组观察者。每个观察者都将包含对主题的关联。我们可以想象一个对主题的重要操作，对应于主题的状态变化，向其观察者列表发出更新，以*通知*它们状态变化。实际上，当主题的状态发生变化时，`Notify()`方法会被调用，并在主题的每个观察者列表上统一应用多态的观察者`Update()`方法。在我们陷入实现之前，让我们考虑构成观察者模式的关键组件。

观察者模式将包括以下内容：

+   主题，或感兴趣的物体。主题将维护一个观察者对象列表（多边关联）。

+   主题将提供一个接口来`Register()`或`Remove()`观察者。

+   主题将包括一个`Notify()`接口，当主题的状态发生变化时，将更新其观察者。主题将通过在其集合中的每个观察者上调用多态的`Update()`方法来`Notify()`观察者。

+   观察者类将被建模为一个抽象类（或接口）。

+   观察者接口将提供一个抽象的多态`Update()`方法，当其关联的主题改变其状态时将被调用。

+   每个观察者与其主题之间的关联将在一个从观察者派生的具体类中维护。这样做将减轻尴尬的类型转换（与在抽象观察者类中维护主题链接相比）。

+   这两个类都将能够维护它们当前的状态。

上述`Subject`和`Observer`类被指定为通用类型，以便它们可以与各种具体的类（主要通过继承）结合使用，这些类希望使用观察者模式。通用的主题和观察者提供了很好的重用机会。在设计模式中，模式的核心元素通常可以更通用地设置，以便允许代码本身的重用，而不仅仅是解决方案（模式）概念的重用。

让我们继续前进，看看观察者模式的示例实现。

# 实现观察者模式

要实现观察者模式，我们首先需要定义我们的`Subject`和`Observer`类。然后，我们需要从这些类派生出具体的类，以包含我们的应用程序特定内容，并使模式生效。让我们开始吧！

## 创建观察者、主题和特定领域的派生类

在我们的示例中，我们将创建 `Subject` 和 `Observer` 类来建立将 `Observer` 注册到 `Subject` 的框架，以及 `Subject` 通知其观察者可能的状态变化的机制。然后，我们将从这些基类派生出我们习惯看到的派生类 - `Course` 和 `Student`，其中 `Course` 将是我们的具体 `Subject`，而 `Student` 将成为我们的具体 `Observer`。

我们将要模拟的应用将涉及课程注册系统和等待名单的概念。正如我们在 *第十章* 的 *问题 2* 中所看到的，*实现关联、聚合和组合*，我们将模拟一个 `Student` 与多个 `Course` 实例的关联，以及一个 `Course` 与多个 `Student` 实例的关联。当我们模拟等待名单时，观察者模式将发挥作用。

我们的 `Course` 类将派生自 `Subject`。我们将继承的观察者列表将代表此 `Course` 的等待名单上的 `Student` 实例。`Course` 还将有一个 `Student` 实例列表，代表成功注册了当前课程的 `Student`。

我们的 `Student` 类将派生自 `Person` 和 `Observer`。`Student` 将包括一个 `Course` 实例列表，其中包含该 `Student` 当前注册的课程。`Student` 还将有一个数据成员 `waitListedCourse`，它对应于一个 `Student` 正在等待添加的 `Course` 的关联。这个 *等待名单* 的 `Course` 代表我们将从中接收通知的 `Subject`。一个通知将对应于一个状态变化，表明 `Course` 现在有空间让一个 `Student` 添加该 `Course`。

`Student` 将从 `Observer` 继承多态操作 `Update()`，这对应于 `Student` 被通知 `Course` 中现在有空位。在这里，在 `Student::Update()` 中，我们将包括添加学生的 `waitListedCourse`（前提是课程开放且有可用座位）的机制。如果添加成功，我们将从课程的等待名单（`Course` 从 `Subject` 继承的观察者列表）中释放 `Student`。自然地，`Student` 将被添加到 `Course` 的当前学生名单中，并且该 `Course` 将出现在该学生的当前课程列表中。

### 指定观察者和主题

让我们将我们的示例分解成组件，从指定我们的 `Observer` 和 `Subject` 的类对开始。完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter16/Chp16-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter16/Chp16-Ex1.cpp)

```cpp
#include <list>    // partial list of #includes
#include <iterator>
using std::cout;   // prefered to: using namespace std;
using std::endl;
using std::setprecision;
using std::string;
using std::to_string;
using std::list;
constexpr int MAXCOURSES = 5, MAXSTUDENTS = 5;
// Simple enums for states; we could have also made a
// hierarchy of states, but let's keep it simple
enum State { Initial = 0, Success = 1, Failure = 2 };
// More specific states for readability in subsequent code
enum StudentState { AddSuccess = State::Success, 
                    AddFailure = State::Failure };
enum CourseState { OpenForEnrollment = State::Success,
                   NewSpaceAvailable = State::Success, 
                   Full = State::Failure };
class Subject;  // forward declarations
class Student;
class Observer  // Observer is an abstract class
{
private:
    // Represent a state as an int, to eliminate type
    // conversions between specific and basic states
    int observerState = State::Initial;  // in-class init.
protected:
    Observer() = default;
    Observer(int s): observerState(s) { }
    void SetState(int s) { observerState = s; }
public:
    int GetState() const { return observerState; }
    virtual ~Observer() = default;
    virtual void Update() = 0;
};
```

在前面的类定义中，我们引入了我们的抽象`Observer`类。在这里，我们包括一个`observerState`和受保护的构造函数来初始化这个状态。我们包括一个受保护的`SetState()`方法，从派生类的范围更新这个状态。我们还包括一个公共的`GetState()`方法。`GetState()`的添加将有助于在`Subject`的`Notify()`方法中的实现，因为它允许我们轻松地检查我们的`Observer`的状态是否已更改。尽管状态信息传统上被添加到`Observer`和`Subject`的派生类中，但我们将在这两个基类中泛化状态信息。这将允许我们的派生类保持更独立于模式，并专注于应用程序的本质。

注意，我们的析构函数是虚的，我们引入了一个抽象方法`virtual void Update() = 0;`来指定`Subject`将在其观察者列表上调用该接口，以将这些更新委托给这些`Observer`实例。

现在，让我们看看我们的`Subject`基类：

```cpp
class Subject   // Treated as an abstract class, due to
{               // protected constructors. However, there's 
private:        // no pure virtual function
    list<class Observer *> observers;
    int numObservers = 0;
    // Represent a state as an int, to eliminate
    // type conversions between specific and basic states
    int subjectState = State::Initial;
    list<Observer *>::iterator newIter;
protected:
    Subject() = default;
    Subject(int s): subjectState(s) { } // note in-class
                                        // init. above
    void SetState(int s) { subjectState = s; }
public:
    int GetState() const { return subjectState; }
    int GetNumObservers() const { return numObservers; }
    virtual ~Subject() = default;
    virtual void Register(Observer *);
    virtual void Release(Observer *);
    virtual void Notify();
};
```

在上述`Subject`类定义中，我们看到我们的`Subject`包括一个 STL `list`来收集其`Observer`实例。它还包括`subjectState`和一个计数器，以反映观察者的数量。此外，我们还包括一个数据成员来跟踪一个未损坏的迭代器。我们将看到这将在我们删除一个元素时很有用（`list::erase()`是一个将使当前迭代器无效的操作）。

我们的`Subject`类也将拥有受保护的构造函数和一个`SetState()`方法，该方法用于初始化或设置`Subject`的状态。尽管这个类在技术上不是抽象的（它不包含纯虚函数），但其构造函数是受保护的，以模拟抽象类；这个类仅打算在派生类实例内部作为子对象进行构造。

在公共接口中，我们有一些访问函数来获取当前状态或观察者的数量。我们还有一个虚析构函数，以及`Register()`、`Release()`和`Notify()`的虚函数。我们将在基类级别提供后三个方法的实现。

接下来，让我们看看`Subject`基类中`Register()`、`Release()`和`Notify()`的默认实现：

```cpp
void Subject::Register(Observer *ob)
{
    observers.push_back(ob); // Add an Observer to the list
    numObservers++;
}
void Subject::Release(Observer *ob) // Remove an Observer 
{                                   // from the list
    bool found = false;
    // loop until we find the desired Observer
    // Note auto iter will be: list<Observer *>::iterator
    for (auto iter = observers.begin();
         iter != observers.end() && !found; ++iter)
    {
        if (*iter == ob)// if we find observer that we seek
        {
            // erase() element, iterator is now corrupt.
            // Save returned (good) iterator; 
            // we'll need it later
            newIter = observers.erase(iter);
            found = true;  // exit loop after found
            numObservers--;
        }
    }
}
void Subject::Notify()
{   // Notify all Observers
    // Note auto iter will be: list<Observer *>::iterator
    for (auto iter = observers.begin(); 
         iter != observers.end(); ++iter)
    {
        (*iter)->Update(); // AddCourse, then Release   
        // Observer. State 'Success' is represented
        // generally for Observer (at this level we have 
        // no knowledge of how Subject and Observer have
        // been specialized). In our application, this
        // means a Student (observer) added a course,
        // got off waitlist (so waitlist had a Release),
        // so we update the iterator
        if ((*iter)->GetState() == State::Success)
            iter = newIter; // update the iterator since
    }                       // erase() invalidated this one
    if (!observers.empty())
    {   // Update last item on waitlist
        Observer *last = *newIter; 
        last->Update();
    }
}
```

在上述`Subject`成员函数中，让我们首先检查`void Subject::Register(Observer *)`方法。在这里，我们只是将作为参数指定的`Observer *`添加到我们的 STL 观察者`list`中（并增加观察者数量的计数器）。

接下来，让我们考虑`Register()`的逆操作，通过审查`void Subject::Release(Observer *)`。在这里，我们遍历观察者列表，直到找到我们正在寻找的观察者。然后我们在当前项上调用`list::erase()`，将我们的`found`标志设置为`true`（以退出循环），并减少观察者的数量。注意，我们还保存了`list::erase()`的返回值，这是一个更新（且有效）的观察者列表迭代器。循环中的迭代器`iter`在我们的`list::erase()`调用后已失效。我们将这个修订的迭代器保存在数据成员`newIter`中，以便我们稍后可以访问它。

最后，让我们看看`Subject`中的`Notify()`方法。当`Subject`中发生状态变化时，此方法将被调用。目标是更新`Subject`的观察者列表中的所有观察者。为了做到这一点，我们遍历我们的列表。一个接一个地，我们使用列表迭代器`iter`获取一个`Observer`。我们通过`(*iter)->Update();`在当前`Observer`上调用`Update()`。我们可以通过使用`if ((*iter)->GetState() == State::Success)`检查观察者的状态来判断给定的`Observer`的更新是否成功。当状态为*Success*时，我们知道观察者的操作将导致我们刚刚审查的`Release()`函数被调用。因为`Release()`中使用的`list::erase()`已使迭代器无效，所以我们现在使用`iter = newIter;`获取正确和修订的迭代器。最后，在循环外部，我们在观察者列表的最后一个项目上调用`Update()`。

### 从 Subject 和 Observer 派生具体类

让我们通过查看从`Subject`或`Observer`派生的具体类来继续这个例子。让我们从`Subject`派生的`Course`开始：

```cpp
class Course: public Subject  
{   // inherits Observer list; 
    // Observer list represents Students on waitlist
private:
    string title;
    int number = 0;  // course num, total num students set
    int totalStudents = 0; // using in-class initialization
    Student *students[MAXSTUDENTS] = { }; // initialize to
                                          // nullptrs
public:                             
    Course(const string &title, int num): number(num)
    {
        this->title = title;  // or rename parameter
        // Note: in-class init. is in-lieu of below:
        // for (int i = 0; i < MAXSTUDENTS; i++)
            // students[i] = nullptr; 
    }
    // destructor body shown as place holder to add more
    // work that will be necessary
    ~Course() override 
    {     /* There's more work to add here! */    }
    int GetCourseNum() const { return number; }
    const string &GetTitle() const { return title; }
    const AddStudent(Student *);
    void Open() 
{    SetState(CourseState::OpenForEnrollment); 
Notify(); 
    } 
    void PrintStudents() const;
};
bool Course::AddStudent(Student *s)
{  // Should also check Student hasn't been added to Course
    if (totalStudents < MAXSTUDENTS)  // course not full
    {
        students[totalStudents++] = s;
        return true;
    }
    else return false;
}
void Course::PrintStudents() const
{
    cout << "Course: (" << GetTitle() << 
            ") has the following students: " << endl;
    for (int i = 0; i < MAXSTUDENTS && 
                        students[i] != nullptr; i++)
    {
        cout << "\t" << students[i]->GetFirstName() << " ";
        cout << students[i]->GetLastName() << endl;
    }
}
```

在我们之前提到的`Course`类中，我们包括课程标题和编号的数据成员，以及当前注册的学生总数。我们还有当前注册的学生列表，表示为`Student *students[MAXNUMBERSTUDENTS];`。此外，请记住，我们从`Subject`基类继承了 STL `list`观察者。这个`Observer`实例的列表将代表我们的`Course`（学生）等待列表。

`Course`类还包括一个构造函数、一个虚析构函数和简单的访问函数。请注意，虚析构函数要做的工作比显示的更多——如果`Course`被销毁，我们必须记住首先从`Course`中移除（但不删除）`Student`实例。我们的`bool Course::AddStudent(Student *)`接口将允许我们将`Student`添加到`Course`中。当然，我们应该确保`Student`没有在这个方法的主体中添加`Course`。

我们的 `void Course::Open();` 方法将在 `Course` 对象上被调用，以指示该课程现在可以添加学生。在这里，我们首先将状态设置为 `Course::OpenForEnrollment`（通过枚举类型明确表示 *Open for Enrollment*），然后调用 `Notify()`。我们的基类 `Subject` 中的 `Notify()` 方法会遍历每个 `Observer`，对每个观察者调用多态的 `Update()` 方法。每个 `Observer` 是一个 `Student`；`Student::Update()` 将允许等待名单上的每个 `Student` 尝试添加 `Course`，现在该课程对学生开放。一旦成功添加到课程的当前学生名单中，`Student` 将请求其在等待名单上的位置 `Release()`（作为一个 `Observer`）。

接下来，让我们看一下我们的 `Student` 类定义，这是我们从 `Person` 和 `Observer` 派生出来的具体类：

```cpp
class Person { }; // Assume our typical Person class here
class Student: public Person, public Observer
{
private:
    float gpa = 0.0;     // in-class initialization
    const string studentId;
    int currentNumCourses = 0;
    Course *courses[MAXCOURSES] = { }; // set to nullptrs
    // Course we'd like to take - we're on the waitlist. 
    Course *waitListedCourse = nullptr;  // Our Subject
                                // (in specialized form)
    static int numStudents;
public:
    Student();  // default constructor
    Student(const string &, const string &, char, 
            const string &, float, const string &, Course *);
    Student(const string &, const string &, char, 
            const string &, float, const string &);
    Student(const Student &) = delete; // Copies disallowed
    ~Student() override;   // virtual destructor
    void EarnPhD();
    float GetGpa() const { return gpa; }
    const string &GetStudentId() const 
       { return studentId; }
    void Print() const override;  // from Person
    void IsA() const override;  // from Person
    void Update() override;     // from Observer
    virtual void Graduate(); // newly introduced virtual fn
    bool AddCourse(Course *);
    void PrintCourses() const;
    static int GetNumberStudents() { return numStudents; } 
};
```

简要回顾一下之前提到的 `Student` 类定义，我们可以看到这个类通过多继承同时从 `Person` 和 `Observer` 派生出来。让我们假设我们的 `Person` 类与我们过去多次使用的是一样的。

除了我们 `Student` 类的常规组件外，我们添加了数据成员 `Course *waitListedCourse;`，它将模拟与我们的 `Subject` 的关联。这个数据成员将模拟我们非常希望添加，但目前无法添加的 `Course` 的概念，即一个 *等待名单* 的课程。在这里，我们正在实现单个等待名单课程的概念，但我们可以轻松地扩展示例以包括支持多个等待名单课程的列表。请注意，这个链接（数据成员）是以派生类型 `Course` 的形式声明的，而不是基类型 `Subject`。这在观察者模式中很典型，并且将帮助我们避免在 `Student` 中重写 `Update()` 方法时的讨厌的向下转型。正是通过这个链接，我们将与我们的 `Subject` 进行交互，以及我们接收来自 `Subject` 更新状态的方式。

我们还注意到，我们在 `Student` 中声明了 `virtual void Update() override;`。这个方法将允许我们重写由 `Observer` 指定的纯虚 `Update()` 方法。

接下来，让我们回顾一下 `Student` 的各种新成员函数：

```cpp
// Assume most Student member functions are as we are
// accustomed to seeing. All are available online.
// Let's look at ONLY those that may differ:
// Note that the default constructor for Observer() will be
// invoked implicitly, thus it is not needed in init list
// below (it is shown in comment as a reminder it's called)
Student::Student(const string &fn, const string &ln, 
    char mi, const string &t, float avg, const string &id,
    Course *c): Person(fn, ln, mi, t), // Observer(),
    gpa(avg), studentId(id), currentNumCourses(0)
{ 
    // Below nullptr assignment is no longer needed with
    // above in-class initialization; otherwise, add here:
    // for (int i = 0; i < MAXCOURSES; i++)
        // courses[i] = nullptr;
    waitListedCourse = c;  // set initial waitlisted Course
                           // (Subject)
    c->Register(this); // Add the Student (Observer) to 
                       // the Subject's list of Observers
    numStudents++;
}
bool Student::AddCourse(Course *c)
{ 
    // Should also check Student isn't already in Course
    if (currentNumCourses < MAXCOURSES)
    {
        courses[currentNumCourses++] = c;  // set assoc.
        c->AddStudent(this);               // set back-link
        return true;
    }
    else  // if we can't add the course,
    {   // add Student (Observer) to the Course's Waitlist, 
        c->Register(this);  // stored in Subject base class
        waitListedCourse = c; // set Student (Observer) 
                              // link to Subject
        return false;
    }
}
```

让我们回顾一下之前列出的成员函数。由于我们已经习惯了 `Student` 类中的大多数必要组件和机制，我们将重点关注新添加的 `Student` 方法，从备用构造函数开始。在这个构造函数中，让我们假设我们像往常一样设置了大多数数据成员。这里的关键代码行是 `waitListedCourse = c;`，将我们的等待名单条目设置为所需的 `Course`（`Subject`），以及 `c->Register(this);`，其中我们将 `Student`（`Observer`）添加到 `Subject` 的列表（课程的正式等待名单）中。

接下来，在我们的 `bool Student::AddCourse(Course *)` 方法中，我们首先检查我们是否没有超过允许的最大课程数。如果没有，我们就进行添加关联的机制，以便在两个方向上链接一个 `Student` 和 `Course`。也就是说，`courses[currentNumCourses++] = c;` 使得学生的当前课程列表包含对新 `Course` 的关联，以及 `c->AddStudent(this);` 请求当前 `Course` 将 `Student`（即 `this`）添加到其注册学生名单中。

让我们继续回顾 `Student` 的新成员函数的其余部分：

```cpp
void Student::Update()
{   // Course state changed to 'Open For Enrollment', etc.
    // so we can now add it.
    if ((waitListedCourse->GetState() == 
         CourseState::OpenForEnrollment) ||
        (waitListedCourse->GetState() == 
         CourseState::NewSpaceAvailable))
    {
        if (AddCourse(waitListedCourse)) // success Adding 
        {
            cout << GetFirstName() << " " << GetLastName();
            cout << " removed from waitlist and added to ";
            cout << waitListedCourse->GetTitle() << endl;
            // Set observer's state to AddSuccess
            SetState(StudentState::AddSuccess); 
            // Remove Student from Course's waitlist
            waitListedCourse->Release(this); // Remove Obs.
                                            // from Subject
            waitListedCourse = nullptr; // Set Subject link 
        }                               // to null
    }
}
void Student::PrintCourses() const
{
    cout << "Student: (" << GetFirstName() << " ";
    cout << GetLastName() << ") enrolled in: " << endl;
    for (int i = 0; i < MAXCOURSES && 
                    courses[i] != nullptr; i++)
        cout << "\t" << courses[i]->GetTitle() << endl;
}
```

继续我们之前提到的 `Student` 成员函数的其余部分，接下来，在我们的多态 `void Student::Update()` 方法中，我们执行所需的添加等待名单中的课程。回想一下，当我们的 `Subject`（课程）发生状态变化时，将调用 `Notify()`。一种这样的状态变化可能是当 `Course` 对注册开放，或者当 `Student` 放弃 `Course` 后，现在存在一个 *New Space Available*（新空间可用）的状态。`Notify()` 然后对每个 `Observer` 调用 `Update()`。我们的 `Update()` 在 `Student` 中被重写以获取 `Course`（主题）的状态。如果状态表明课程现在对注册开放或有一个 *New Space Available*，我们尝试 `AddCourse(waitListedCourse);`。如果成功，我们将 `Student`（观察者）的状态设置为 `StudentState::AddSuccess`（添加成功）以指示我们在 `Update()` 中成功，这意味着我们已添加了课程。接下来，由于我们已经将期望的课程添加到我们的当前课程列表中，我们现在可以自己从 `Course` 的等待名单中移除。也就是说，我们将使用 `waitListedCourse->Release(this);` 将自己（学生）作为 `Observer` 从 `Subject`（课程的等待名单）中移除。现在我们已经添加了我们的期望等待名单课程，我们也可以使用 `waitListedCourse = nullptr;` 移除我们与 `Subject` 的链接。

最后，我们之前提到的 `Student` 代码包括一个方法来打印 `Student` 当前注册的课程，即 `void Student::PrintCourses();`。这个方法相当直接。

### 将模式组件组合在一起

让我们现在通过查看我们的 `main()` 函数来查看我们的观察者模式是如何编排的：

```cpp
int main()
{   // Instantiate several courses
    Course *c1 = new Course("C++", 230);  
    Course *c2 = new Course("Advanced C++", 430);
    Course *c3 = new Course("C++ Design Patterns", 550);
    // Instantiate Students, select a course to be on the 
    // waitlist for -- to be added when registration starts
    Student s1("Anne", "Chu", 'M', "Ms.", 3.9, "66CU", c1);
    Student s2("Joley", "Putt", 'I', "Ms.", 3.1, 
               "585UD", c1);
    Student s3("Geoff", "Curt", 'K', "Mr.", 3.1, 
               "667UD", c1);
    Student s4("Ling", "Mau", 'I', "Ms.", 3.1, "55TU", c1);
    Student s5("Jiang", "Wu", 'Q', "Dr.", 3.8, "88TU", c1);
    cout << "Registration is Open" << "\n";
    cout << "Waitlist Students to be added to Courses"; 
    cout << endl;
    // Sends a message to Students that Course is Open. 
    c1->Open(); // Students on waitlist will automatically
    c2->Open(); // be Added (as room allows)
    c3->Open();
    // Now that registration is open, add more courses 
    cout << "During open registration, Students now adding
             additional courses" << endl;
    s1.AddCourse(c2);  // Try to add more courses
    s2.AddCourse(c2);  // If full, we'll be added to 
    s4.AddCourse(c2);  // a waitlist
    s5.AddCourse(c2);  
    s1.AddCourse(c3);  
    s3.AddCourse(c3);  
    s5.AddCourse(c3);
    cout << "Registration complete\n" << endl;
    c1->PrintStudents();   // print each Course's roster
    c2->PrintStudents();
    c3->PrintStudents();
    s1.PrintCourses();  // print each Student's course list
    s2.PrintCourses();
    s3.PrintCourses();
    s4.PrintCourses();
    s5.PrintCourses();
    return 0;
}
```

回顾我们之前提到的 `main()` 函数，我们首先创建了三个 `Course` 实例。接下来，我们创建了五个 `Student` 实例，使用一个构造函数，允许我们在课程注册开始时为每个 `Student` 提供一个他们想要添加的初始 `Course`。请注意，这些 `Students`（观察者）将被添加到他们期望课程的等待名单（主题）中。在这里，一个 `Subject`（课程）将有一个 `Observer`（学生）列表，这些学生希望在注册开放时添加该课程。

接下来，我们看到许多`Student`实例渴望的`Course`变为*开放注册*，可以通过`c1->Open();`进行注册。`Course::Open()`将`Subject`的状态设置为`CourseState::OpenForEnrollment`，这很容易表明课程是*开放注册*的，然后调用`Notify()`。正如我们所知，`Subject::Notify()`将在`Subject`的观察者列表上调用`Update()`。正是在这里，一个初始的等待注册的`Course`实例将被添加到学生的日程表中，并随后作为`Observer`从`Subject`的等待列表中移除。

现在注册已经开放，每个`Student`将尝试使用`bool Student::AddCourse(Course *)`以通常的方式添加更多课程，例如`s1.AddCourse(c2);`。如果一个`Course`已满，`Student`将被添加到`Course`的等待列表中（作为继承自`Subject`的观察者列表，实际上是由派生自`Student`类型的观察者组成）。回想一下，`Course`继承自`Subject`，它保留了一个对添加特定课程感兴趣的学生列表（观察者的等待列表）。当`Course`状态变为*新空间可用*时，等待列表中的学生（通过数据成员`observers`）将被通知，并且每个`Student`上的`Update()`方法将随后调用该`Student`的`AddCourse()`方法。

一旦我们添加了各种课程，我们就会看到每个`Course`打印其学生名单，例如`c2->PrintStudents()`。同样，我们也会看到每个`Student`打印他们所注册的课程，例如使用`s5.PrintCourses()`。

让我们看看这个程序的输出：

```cpp
Registration is Open 
Waitlist Students to be added to Courses
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
Course: (C++ Design Patterns) has the following students:
        Anne Chu
        Goeff Curt
        Jiang Wu
Student: (Anne Chu) enrolled in:
        C++
        Advanced C++
        C++ Design Patterns
Student: (Joley Putt) enrolled in:
        C++
        Advanced C++
Student: (Goeff Curt) enrolled in:
        C++
        C++ Design Patterns
Student: (Ling Mau) enrolled in:
        C++
        Advanced C++
Student: (Jiang Wu) enrolled in:
        C++
        Advanced C++
        C++ Design Patterns
```

我们现在已经看到了观察者模式的实现。我们将更通用的`Subject`和`Observer`类折叠到我们习惯看到的类框架中，即`Course`、`Person`和`Student`。现在，让我们简要回顾一下与模式相关的学习内容，然后再进入下一章。

# 摘要

在本章中，我们开始追求成为更好的 C++ 程序员，通过将我们的知识库从面向对象的概念扩展到包括设计模式的应用。我们的主要目标是让您能够通过应用常见的模式，使用**经过验证和可靠的**解决方案来解决重复出现的编程问题。

我们首先理解了设计模式的目的以及在我们代码中采用它们的优势。然后，我们具体理解了观察者模式背后的前提以及它是如何贡献于面向对象的。最后，我们查看了一下如何在 C++ 中实现观察者模式。

利用常见的模式，如观察者模式，将帮助您更轻松地以其他程序员能理解的方式解决重复出现的编程问题。面向对象的一个关键原则是尽可能追求组件的重用。通过利用设计模式，您将为具有更复杂编程技术的可重用解决方案做出贡献。

现在我们已经准备好继续前进，进入我们的下一个设计模式*第十七章*，*实现工厂模式*。将更多模式添加到我们的技能集合中，使我们成为更灵活、更有价值的程序员。让我们继续前进！

# 问题

1.  以本章示例的在线代码作为起点，以及之前练习的解决方案（*问题 3*，*第十章*，*实现关联、聚合和组合*）：

    1.  实现（或修改你之前的）`Student::DropCourse()`。当`Student`取消选课时，这个事件将导致`Course`状态变为状态`2`，*新空间可用*。随着状态的变化，`Notify()`将被调用在`Course`（主题）上，然后它将`Update()`观察者列表（等待名单上的学生）。`Update()`将间接允许等待名单上的`Student`实例（如果有），现在可以添加该课程。

    1.  最后，在`DropCourse()`中，记得从学生的当前课程列表中移除已取消的课程。

1.  你能想象出哪些其他例子可以轻松地融入观察者模式？
