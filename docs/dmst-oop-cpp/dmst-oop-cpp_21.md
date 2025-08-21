# 第十七章：应用工厂模式

本章将继续扩展您的 C++编程技能，超越核心面向对象编程概念，目标是使您能够利用常见的设计模式解决重复出现的编码问题。我们知道，应用设计模式可以增强代码维护性，并为潜在的代码重用提供途径。

继续演示和解释流行的设计模式和习语，并学习如何在 C++中有效实现它们，我们将继续我们的探索，工厂模式，更准确地说是**工厂方法模式**。

在本章中，我们将涵盖以下主要主题：

+   理解工厂方法模式及其对面向对象编程的贡献

+   理解如何使用对象工厂和不使用对象工厂来实现工厂方法模式；比较对象工厂和抽象工厂

在本章结束时，您将理解流行的工厂方法模式。我们将在 C++中看到这种模式的两个示例实现。将额外的核心设计模式添加到您的编程技能中，将使您成为一个更复杂和有价值的程序员。

让我们通过研究这种常见的设计模式，工厂方法模式，来增加我们的编程技能。

# 技术要求

本章示例程序的完整代码可在以下 GitHub 链接找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17)。每个完整的示例程序都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节的章节号对应，后跟破折号，再跟上所在章节中的示例编号。例如，本章的第一个完整程序可以在子目录`Chapter17`中的名为`Chp17-Ex1.cpp`的文件中找到，位于上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/2PdlSLB`](https://bit.ly/2PdlSLB)。

# 理解工厂方法模式

**工厂模式**或**工厂方法模式**是一种创建型设计模式，允许创建对象而无需指定将实例化的确切（派生）类。工厂方法模式提供了一个创建对象的接口，但允许创建方法内的细节决定实例化哪个（派生）类。

工厂方法模式也被称为**虚拟构造函数**。就像虚拟析构函数具有特定的析构函数（这是销毁序列的入口点）在运行时通过动态绑定确定一样，虚拟构造函数的概念是所需的对象在运行时统一确定。

使用工厂方法模式，我们将指定一个抽象类（或接口）来收集和指定我们希望创建的派生类的一般行为。在这种模式中，抽象类或接口被称为**产品**。然后我们创建我们可能想要实例化的派生类，覆盖任何必要的抽象方法。各种具体的派生类被称为**具体产品**。

然后我们指定一个工厂方法，其目的是为了统一创建具体产品的实例。工厂方法可以放在抽象产品类中，也可以放在单独的对象工厂类中；对象工厂代表一个负责创建具体产品的类。如果将工厂方法放在抽象产品类中，那么这个工厂（创建）方法将是静态的，如果放在对象工厂类中，那么它可以选择是静态的。工厂方法将根据一致的输入参数列表决定要制造哪个具体产品，然后返回一个通用的产品指针给具体产品。多态方法可以应用于新创建的对象，以引出其特定的行为。

工厂方法模式将包括以下内容：

+   一个抽象产品类（或接口）。

+   多个具体产品派生类。

+   在抽象产品类或单独的对象工厂类中的工厂方法。工厂方法将具有一个统一的接口来创建任何具体产品类型的实例。

+   具体产品将由工厂方法作为通用产品实例返回。

请记住，工厂方法（无论是在对象工厂中）都会生产产品。工厂方法提供了一种统一的方式来生产许多相关的产品类型。

让我们继续看两个工厂方法模式的示例实现。

# 实现工厂方法模式

我们将探讨工厂方法模式的两种常见实现。每种实现都有设计权衡，值得讨论！

让我们从将工厂方法放在抽象产品类中的技术开始。

## 包括工厂方法在产品类中

要实现工厂方法模式，我们首先需要创建我们的抽象产品类以及我们的具体产品类。这些类定义将为我们构建模式奠定基础。

在我们的例子中，我们将使用一个我们习惯看到的类`Student`来创建我们的产品。然后我们将创建具体的产品类，即`GradStudent`，`UnderGradStudent`和`NonDegreeStudent`。我们将在我们的产品（`Student`）类中包含一个工厂方法，以创建任何派生产品类型的统一接口。

我们将通过添加类来区分学生的教育学位目标，为我们现有的`Student`应用程序补充我们的框架。新的组件为大学入学（新生入学）系统提供了基础。

假设我们的应用程序不是实例化一个`Student`，而是实例化各种类型的`Student` - `GradStudent`，`UnderGradStudent`或`NonDegreeStudent` - 基于他们的学习目标。`Student`类将包括一个抽象的多态`Graduate()`操作；每个派生类将使用不同的实现重写这个方法。例如，寻求博士学位的`GradStudent`可能在`GradStudent::Graduate()`方法中有更多与学位相关的标准要满足，而其他`Student`的专业化可能不需要。他们可能需要验证学分小时数，验证通过的平均成绩，以及验证他们的论文是否被接受。相比之下，`UnderGradStudent`可能只需要验证他们的学分小时数和总体平均成绩。

抽象产品类将包括一个静态方法`MatriculateStudent()`，作为创建各种类型学生（具体产品类型）的工厂方法。

### 定义抽象产品类

让我们首先看一下我们的工厂方法实现的机制，从检查我们的抽象产品类`Student`的定义开始。这个例子可以在我们的 GitHub 存储库中找到一个完整的程序：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17/Chp17-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17/Chp17-Ex1.cpp)

```cpp
// Assume Person class exists with its usual implementation
class Student: public Person  // Notice that Student is now  
{                             // an abstract class
private:
    float gpa;
    char *currentCourse;
    const char *studentId;
public:
    Student();  // default constructor
    Student(const char *, const char *, char, const char *,
            float, const char *, const char *);
    Student(const Student &);  // copy constructor
    virtual ~Student();  // destructor
    float GetGpa() const { return gpa; }
    const char *GetCurrentCourse() const 
       { return currentCourse; }
    const char *GetStudentId() const { return studentId; }
    void SetCurrentCourse(const char *); // prototype only
    virtual void Print() const override;
    virtual const char *IsA() override { return "Student"; }
    virtual void Graduate() = 0;  // Now Student is abstract
    // Creates a derived Student type based on degree sought
    static Student *MatriculateStudent(const char *,
       const char *, const char *, char, const char *,
       float, const char *, const char *);
};
// Assume all the usual Student member functions exist 
```

在之前的类定义中，我们介绍了抽象的`Student`类，它是从`Person`（一个具体的、因此可实例化的类）派生出来的。这是通过引入抽象方法`virtual void Graduate() = 0;`来实现的。在我们的学生入学示例中，我们将遵循一个设计决策，即只有特定类型的学生应该被实例化；也就是说，`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`的派生类类型。

在前面的类定义中，注意我们的工厂方法，具有`static Student *MatriculateStudent();`原型。这个方法将使用统一的接口，并提供了创建各种`Student`派生类类型的手段。一旦我们看到了派生类的类定义，我们将详细研究这个方法。

### 定义具体产品类

现在，让我们来看看我们的具体产品类，从`GradStudent`开始：

```cpp
class GradStudent: public Student
{
private:
    char *degree;  // PhD, MS, MA, etc.
public:
    GradStudent() { degree = 0; }  // default constructor
    GradStudent(const char *, const char *, const char *,
       char, const char *, float, const char *, const char *);
    GradStudent(const GradStudent &);  // copy constructor
    virtual ~GradStudent() { delete degree; } // destructor
    void EarnPhD();
    virtual const char *IsA() override 
       { return "GradStudent"; }
    virtual void Graduate() override;
};
// Assume alternate and copy constructors are implemented
// as expected. See online code for full implementation.
void GradStudent::EarnPhD()
{
    if (!strcmp(degree, "PhD")) // only PhD candidates can 
        ModifyTitle("Dr.");     // EarnPhd(), not MA and MS 
}                               // candidates
void GradStudent::Graduate()
{   // Here, we can check that the required number of credits
    // have been met with a passing gpa, and that their 
    // doctoral or master's thesis has been completed.
    EarnPhD();  // Will change title only if a PhD candidate
    cout << "GradStudent::Graduate()" << endl;
}
```

在上述的`GradStudent`类定义中，我们添加了一个`degree`数据成员，用于指示`"PhD"`、“MS”或`"MA"`学位，并根据需要调整构造函数和析构函数。我们已经将`EarnPhD()`移到`GradStudent`，因为这个方法并不适用于所有的`Student`实例。相反，`EarnPhD()`适用于`GradStudent`实例的一个子集；我们只会授予`"Dr."`头衔给博士候选人。

在这个类中，我们重写了`IsA()`，返回`"GradStudent"`。我们还重写了`Graduate()`，以便进行适用于研究生的毕业清单，如果满足了这些清单项目，就调用`EarnPhD()`。

现在，让我们来看看我们的下一个具体产品类，`UnderGradStudent`：

```cpp
class UnderGradStudent: public Student
{
private:
    char *degree;  // BS, BA, etc
public:
    UnderGradStudent() { degree = 0; }  // default constructor
    UnderGradStudent(const char *, const char *, const char *,
       char, const char *, float, const char *, const char *);
    UnderGradStudent(const UnderGradStudent &);  
    virtual ~UnderGradStudent() { delete degree; } 
    virtual const char *IsA() override 
        { return "UnderGradStudent"; }
    virtual void Graduate() override;
};
// Assume alternate and copy constructors are implemented
// as expected. See online code for full implementation.
void UnderGradStudent::Graduate()
{   // Verify that number of credits and gpa requirements have
    // been met for major and any minors or concentrations.
    // Have all applicable university fees been paid?
    cout << "UnderGradStudent::Graduate()" << endl;
}
```

快速看一下之前定义的`UnderGradStudent`类，我们注意到它与`GradStudent`非常相似。这个类甚至包括一个`degree`数据成员。请记住，并非所有的`Student`实例都会获得学位，所以我们不希望通过在`Student`中定义它来概括这个属性。虽然我们可以引入一个共享的基类`DegreeSeekingStudent`，用于收集`UnderGradStudent`和`GradStudent`的共同点，但这种细粒度的层次几乎是不必要的。这里的重复是一个设计权衡。

这两个兄弟类之间的关键区别是重写的`Graduate()`方法。我们可以想象，本科生毕业的清单可能与研究生不同。因此，我们可以合理地区分这两个类。否则，它们基本上是一样的。

现在，让我们来看看我们的下一个具体产品类，`NonDegreeStudent`：

```cpp
class NonDegreeStudent: public Student
{
public:
    NonDegreeStudent() { }  // default constructor
    NonDegreeStudent(const char *, const char *, char, 
       const char *, float, const char *, const char *);
    NonDegreeStudent(const NonDegreeStudent &s): Student(s){ }  
    virtual ~NonDegreeStudent() { } // destructor
    virtual const char *IsA() override 
       { return "NonDegreeStudent"; }
    virtual void Graduate() override;
};
// Assume alternate constructor is implemented as expected.
// Notice copy constructor is inline above (as is default)
// See online code for full implementation.
void NonDegreeStudent::Graduate()
{   // Check if applicable tuition has been paid. 
    // There is no credit or gpa requirement.
    cout << "NonDegreeStudent::Graduate()" << endl;
}
```

快速看一下上述的`NonDegreeStudent`类，我们注意到这个具体产品与它的兄弟类相似。然而，在这个类中没有学位数据成员。此外，重写的`Graduate()`方法需要进行的验证比`GradStudent`或`UnderGradStudent`类中的重写版本少。

### 检查工厂方法定义

接下来，让我们来看看我们的工厂方法，即我们产品（`Student`）类中的静态方法：

```cpp
// Creates a Student based on the degree they seek
// This is a static method of Student (keyword in prototype)
Student *Student::MatriculateStudent(const char *degree, 
    const char *fn, const char *ln, char mi, const char *t,
    float avg, const char *course, const char *id)
{
    if (!strcmp(degree, "PhD") || !strcmp(degree, "MS") 
        || !strcmp(degree, "MA"))
        return new GradStudent(degree, fn, ln, mi, t, avg,
                               course, id);
    else if (!strcmp(degree, "BS") || !strcmp(degree, "BA"))
        return new UnderGradStudent(degree, fn, ln, mi, t,
                                    avg, course, id);
    else if (!strcmp(degree, "None"))
        return new NonDegreeStudent(fn, ln, mi, t, avg,
                                    course, id);
}
```

前面提到的`Student`的静态方法`MatriculateStudent()`代表了工厂方法，用于创建各种产品（具体`Student`实例）。在这里，根据`Student`所寻求的学位类型，将实例化`GradStudent`，`UnderGradStudent`或`NonDegreeStudent`中的一个。请注意，`MatriculateStudent()`的签名可以处理任何派生类构造函数的参数要求。还要注意，任何这些专门的实例类型都将作为抽象产品类型（`Student`）的基类指针返回。

工厂方法`MatriculateStudent()`中的一个有趣选项是，这个方法并不一定要实例化一个新的派生类实例。相反，它可以重用之前可能仍然可用的实例。例如，想象一下，一个`Student`暂时未在大学注册（因为费用支付迟到），但仍然被保留在*待定学生*名单上。`MatriculateStudent()`方法可以选择返回指向这样一个现有`Student`的指针。*回收*是工厂方法中的一种替代方法！

### 将模式组件整合在一起

最后，让我们通过查看我们的`main()`函数来将所有不同的组件整合在一起，看看我们的工厂方法模式是如何被编排的：

```cpp
int main()
{
    Student *scholars[MAX];
    // Student is now abstract....cannot instantiate directly
    // Use the Factory Method to make derived types uniformly
    scholars[0] = Student::MatriculateStudent("PhD", "Sara",
                "Kato", 'B', "Ms.", 3.9, "C++", "272PSU");
    scholars[1] = Student::MatriculateStudent("BS", "Ana",
                "Sato", 'U', "Ms.", 3.8, "C++", "178PSU");
    scholars[2] = Student::MatriculateStudent("None", "Elle",
                "LeBrun", 'R', "Miss", 3.5, "C++", "111BU");
    for (int i = 0; i < MAX; i++)
    {
       scholars[i]->Graduate();
       scholars[i]->Print();
    }
    for (int i = 0; i < MAX; i++)
       delete scholars[i];   // engage virtual dest. sequence
    return 0;
}
```

回顾我们前面提到的`main()`函数，我们首先创建一个指向潜在专业化`Student`实例的指针数组，以它们的一般化`Student`形式。接下来，我们在抽象产品类中调用静态工厂方法`Student::MatriculateStudent()`来创建适当的具体产品（派生`Student`类类型）。我们创建每个派生`Student`类型 - `GradStudent`，`UnderGradStudent`和`NonDegreeStudent`各一个。

然后，我们通过我们的一般化集合循环，为每个实例调用`Graduate()`，然后调用`Print()`。对于获得博士学位的学生（`GradStudent`实例），他们的头衔将被`GradStudent::Graduate()`方法更改为`"Dr."`。最后，我们通过另一个循环来释放每个实例的内存。幸运的是，`Student`已经包含了一个虚析构函数，以便销毁顺序从适当的级别开始。

让我们来看看这个程序的输出：

```cpp
GradStudent::Graduate()
  Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
UnderGradStudent::Graduate()
  Ms. Ana U. Sato with id: 178PSU GPA:  3.8 Course: C++
NonDegreeStudent::Graduate()
  Miss Elle R. LeBrun with id: 111BU GPA:  3.5 Course: C++
```

前面实现的一个优点是它非常直接。然而，我们可以看到抽象产品类包含工厂方法（用于构造派生类类型）和派生具体产品之间存在着紧密的耦合。然而，在面向对象编程中，基类通常不会了解任何派生类型。

这种紧密耦合实现的一个缺点是，抽象产品类必须在其静态创建方法`MatriculateStudent()`中包含每个后代的实例化手段。添加新的派生类现在会影响抽象基类的定义 - 需要重新编译。如果我们没有访问这个基类的源代码怎么办？有没有一种方法来解耦工厂方法和工厂方法将创建的产品之间存在的依赖关系？是的，有一种替代实现。

让我们现在来看一下工厂方法模式的另一种实现。我们将使用一个对象工厂类来封装我们的`MatriculateStudent()`工厂方法，而不是将这个方法包含在抽象产品类中。

## 创建一个对象工厂类来封装工厂方法

对于工厂方法模式的另一种实现，我们将对抽象产品类进行轻微偏离其先前的定义。然而，我们将像以前一样创建我们的具体产品类。这些类定义将再次开始构建我们模式的框架。

在我们修改后的示例中，我们将再次将我们的产品定义为`Student`类。我们还将再次派生具体的产品类`GradStudent`，`UnderGradStudent`和`NonDegreeStudent`。然而，这一次，我们不会在我们的产品（`Student`）类中包含工厂方法。相反，我们将创建一个单独的对象工厂类，其中将包括工厂方法。与之前一样，工厂方法将具有统一的接口来创建任何派生产品类型。工厂方法不需要是静态的，就像在我们上一次的实现中一样。

我们的对象工厂类将包括`MatriculateStudent()`作为工厂方法来创建各种`Student`实例（具体产品类型）。

### 定义不包含工厂方法的抽象产品类

让我们来看看我们对工厂方法模式的替代实现的机制，首先检查我们的抽象产品类`Student`的定义。这个例子可以在我们的 GitHub 存储库中找到一个完整的程序：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17/Chp17-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter17/Chp17-Ex2.cpp)

```cpp
// Assume Person class exists with its usual implementation
class Student: public Person   // Notice Student is 
{                              // an abstract class
private:
    float gpa;
    char *currentCourse;
    const char *studentId;
public:
    Student();  // default constructor
    Student(const char *, const char *, char, const char *,
            float, const char *, const char *);
    Student(const Student &);  // copy constructor
    virtual ~Student();  // destructor
    float GetGpa() const { return gpa; }
    const char *GetCurrentCourse() const 
       { return currentCourse; }
    const char *GetStudentId() const { return studentId; }
    void SetCurrentCourse(const char *); // prototype only
    virtual void Print() const override;
    virtual const char *IsA() override { return "Student"; }
    virtual void Graduate() = 0;  // Student is abstract
};
```

在我们上述的`Student`类定义中，与我们之前的实现的关键区别是，这个类不再包含一个静态的`MatriculateStudent()`方法作为工厂方法。`Student`只是一个抽象基类。

### 定义具体产品类

有了这个想法，让我们来看看派生（具体产品）类：

```cpp
class GradStudent: public Student
{   // Implemented as in our last example
};
class UnderGradStudent: public Student
{   // Implemented as in our last example
};
class NonDegreeStudent: public Student
{   // Implemented as in our last example
};
```

在我们之前列出的类定义中，我们可以看到我们的具体派生产品类与我们在第一个示例中实现这些类的方式是相同的。

### 将对象工厂类添加到工厂方法

接下来，让我们介绍一个包括我们工厂方法的对象工厂类：

```cpp
class StudentFactory    // Object Factory class
{
public:   
    // Factory Method – creates Student based on degree sought
    Student *MatriculateStudent(const char *degree, 
       const char *fn, const char *ln, char mi, const char *t,
       float avg, const char *course, const char *id)
    {
        if (!strcmp(degree, "PhD") || !strcmp(degree, "MS") 
            || !strcmp(degree, "MA"))
            return new GradStudent(degree, fn, ln, mi, t, 
                                   avg, course, id);
        else if (!strcmp(degree, "BS") || 
                 !strcmp(degree, "BA"))
            return new UnderGradStudent(degree, fn, ln, mi, t,
                                        avg, course, id);
        else if (!strcmp(degree, "None"))
            return new NonDegreeStudent(fn, ln, mi, t, avg,
                                        course, id);
    }
};
```

在上述的对象工厂类定义（`StudentFactory`类）中，我们最少包括工厂方法规范，即`MatriculateStudent()`。该方法与我们之前的示例中的方法非常相似。然而，通过在对象工厂中捕获具体产品的创建，我们已经解耦了抽象产品和工厂方法之间的关系。

### 将模式组件结合在一起

接下来，让我们将我们的`main()`函数与我们原始示例的函数进行比较，以可视化我们修改后的组件如何实现工厂方法模式：

```cpp
int main()
{
    Student *scholars[MAX];
    // Create an Object Factory for Students
    StudentFactory *UofD = new StudentFactory();
    // Student is now abstract....cannot instantiate directly
    // Ask the Object Factory to create a Student
    scholars[0] = UofD->MatriculateStudent("PhD", "Sara", 
                  "Kato", 'B', "Ms.", 3.9, "C++", "272PSU");
    scholars[1] = UofD->MatriculateStudent("BS", "Ana", "Sato"
                  'U', "Dr.", 3.8, "C++", "178PSU");
    scholars[2] = UofD->MatriculateStudent("None", "Elle",
                  "LeBrun", 'R', "Miss", 3.5, "c++", "111BU");
    for (int i = 0; i < MAX; i++)
    {
       scholars[i]->Graduate();
       scholars[i]->Print();
    }
    for (int i = 0; i < MAX; i++)
       delete scholars[i];   // engage virtual dest. sequence
    return 0;
}
```

考虑到我们之前列出的`main()`函数，我们可以看到我们再次创建了指向抽象产品类型（`Student`）的指针数组。然后，我们实例化了一个可以创建各种具体产品类型的`Student`实例的对象工厂，即`StudentFactory *UofD = new StudentFactory();`。与之前的示例一样，对象工厂根据每个学生所寻求的学位类型创建了每个派生类型的`GradStudent`，`UnderGradStudent`和`NonDegreeStudent`的一个实例。`main()`中的其余代码与我们之前的示例中一样。

我们的输出将与我们上一个示例相同。

对象工厂类的优势在于，我们已经从抽象产品类（在工厂方法中）中移除了对象创建的依赖，并知道派生类类型是什么。也就是说，如果我们扩展层次结构以包括新的具体产品类型，我们不必修改抽象产品类。当然，我们需要访问修改我们的对象工厂类`StudentFactory`，以增强我们的`MatriculateStudent()`工厂方法。

与这种实现相关的一种模式，**抽象工厂**，是另一种模式，它允许具有类似目的的单个工厂被分组在一起。抽象工厂可以被指定为提供统一类似对象工厂的方法；它是一个将创建工厂的工厂，为我们原始模式添加了另一层抽象。

我们现在已经看到了工厂方法模式的两种实现。我们已经将产品和工厂方法的概念融入了我们习惯看到的类框架中，即`Student`和`Student`的派生类。在继续前往下一章之前，让我们简要地回顾一下我们在模式方面学到的东西。

# 总结

在本章中，我们继续努力成为更好的 C++程序员，扩展我们对设计模式的知识。特别是，我们从概念上和通过两种常见的实现探讨了工厂方法模式。我们的第一个实现包括将工厂方法放在我们的抽象产品类中。我们的第二个实现通过添加一个对象工厂类来包含我们的工厂方法，消除了我们的抽象产品和工厂方法之间的依赖关系。我们还非常简要地讨论了抽象工厂的概念。

利用常见的设计模式，比如工厂方法模式，将帮助您更轻松地解决其他程序员理解的重复类型的编程问题。通过利用核心设计模式，您将为使用更复杂的编程技术提供了被理解和可重用的解决方案。

我们现在准备继续前进到我们的下一个设计模式*第十八章*，*实现适配器模式*。向我们的技能集合中添加更多的模式使我们成为更多才多艺和有价值的程序员。让我们继续前进吧！

# 问题

1.  使用*问题 1*中的解决方案，*第八章*，*掌握抽象类*：

a. 实现工厂方法模式来创建各种形状。您已经创建了一个名为 Shape 的抽象基类，以及派生类，比如 Rectangle、Circle、Triangle，可能还有 Square。

b. 选择在`Shape`中将工厂方法实现为静态方法，或者作为`ShapeFactory`类中的方法（如果需要的话引入后者类）。

1.  您能想象其他哪些例子可能很容易地融入工厂方法模式？
