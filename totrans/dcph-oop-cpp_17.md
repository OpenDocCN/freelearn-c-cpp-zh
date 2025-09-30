# 17

# 应用工厂模式

本章将继续我们的追求，以扩展你的 C++编程工具箱，使其超越核心 OOP 概念，目标是使你能够利用常见的设计模式解决重复出现的编码问题。我们知道，结合设计模式可以增强代码维护性，并为潜在的代码重用提供途径。

继续演示和解释流行的设计模式和惯用法，并学习如何在 C++中有效地实现它们，我们继续我们的探索之旅，这次是工厂模式，更确切地说是**工厂方法模式**。

在本章中，我们将涵盖以下主要内容：

+   理解工厂方法模式及其对面向对象编程（OOP）的贡献

+   理解如何使用和没有对象工厂实现工厂方法模式，以及比较对象工厂和抽象工厂

到本章结束时，你将理解流行的工厂方法模式。我们将看到 C++中该模式的两个示例实现。将更多的核心设计模式添加到你的编程工具箱中，将使你成为一个更复杂且更有价值的程序员。

通过研究另一个常见的设计模式，即工厂方法模式，让我们提高我们的编程技能。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter17`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter17)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter17`子目录中找到，文件名为`Chp17-Ex1.cpp`。

本章的 CiA 视频可以在以下网址查看：[`bit.ly/3QOmCC1`](https://bit.ly/3QOmCC1)。

# 理解工厂方法模式

**工厂模式**，或称为**工厂方法模式**，是一种创建型设计模式，它允许在不指定将要实例化的确切（派生）类的情况下创建对象。工厂方法模式提供了一个创建对象的接口，同时允许创建方法中的细节决定要实例化哪个（派生）类。

工厂方法模式也被称为**虚拟构造函数**。正如虚拟析构函数具有特定的析构函数（它是销毁序列的入口点），通过动态绑定在运行时确定一样，虚拟构造函数的概念是，在运行时统一确定要实例化的所需对象。

我们无法总是预见到在应用程序中需要的特定相关派生类对象的混合。工厂方法（或虚拟构造函数）可以根据提供的输入，在请求时创建许多相关派生类类型中的一种实例。工厂方法将派生类对象作为其基类类型返回，允许对象以更通用的方式创建和存储。可以将多态操作应用于新创建的（向上转型）实例，使相关的派生类行为得以展现。工厂方法通过消除在客户端代码中绑定特定派生类类型的需要，促进了与客户端代码的松耦合。客户端只需利用工厂方法来创建和提供适当的实例。

使用工厂方法模式，我们将指定一个抽象类（或接口）来收集和指定我们希望创建的派生类的通用行为。在这个模式中的抽象类或接口被称为**产品**。然后我们创建可能想要实例化的派生类，覆盖任何必要的抽象方法。各种具体的派生类被称为**具体产品**。

然后我们指定一个工厂方法，其目的是提供一个接口，以统一创建具体产品实例。工厂方法可以放在抽象产品类中，也可以放在单独的对象工厂类中；**对象工厂**代表一个具有创建具体产品任务的类。如果放在抽象产品类中，这个工厂（创建）方法将是静态的；如果放在对象工厂类中，则可选地是静态的。工厂方法将根据一致的输入参数列表决定制造哪个具体的产品。工厂方法将返回一个指向具体产品的通用产品指针。可以将多态方法应用于新创建的对象，以引发其特定行为。

工厂方法模式将包括以下内容：

+   一个抽象的**产品**类（或接口）。

+   多个**具体产品**派生类。

+   在抽象产品类或单独的**对象工厂**类中的**工厂方法**。工厂方法将具有统一的接口来创建任何具体产品类型的实例。

+   具体产品将由工厂方法作为通用产品实例返回。

请记住，工厂方法（无论是否在对象工厂中）产生产品。工厂方法提供了一种统一的方式产生许多相关的产品类型。可以存在多个工厂方法来生产独特的产品线；每个工厂方法可以通过一个有意义的名称来区分，即使它们的签名碰巧是相同的。

让我们继续前进，看看工厂方法模式的两个示例实现。

# 实现工厂方法模式

我们将探讨两种常见的工厂方法模式的实现。每种实现都会有设计权衡，当然值得讨论！

让我们从将工厂方法放置在抽象产品类中的技术开始。

## 在产品类中包含工厂方法

为了实现工厂方法模式，我们首先需要创建我们的抽象产品类以及我们的具体产物类。这些类定义将为我们构建模式的基础。

在我们的例子中，我们将使用我们习惯看到的类来创建我们的产品 – `Student`。然后我们将创建具体的产物类，即`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`。我们将在我们的产品（`Student`）类中包含一个工厂方法，它具有一致的接口来创建任何派生产品类型。

我们将要建模的组件将通过添加基于他们教育学位目标的类来补充我们现有的`Student`应用程序的框架。这些新组件为大学入学（新的`Student`录取）系统提供了基础。

让我们假设，而不是实例化一个`Student`，我们的应用程序将根据他们的学习目标实例化各种类型的`Student` – `GradStudent`、`UnderGradStudent`或`NonDegreeStudent`。`Student`类将包括一个抽象的多态`Graduate()`操作；每个派生类将使用不同的实现覆盖此方法。例如，寻求博士学位的`GradStudent`可能在`GradStudent::Graduate()`方法中需要满足比其他`Student`特殊化更多的学位相关标准。他们可能需要验证学分小时数，验证通过的平均成绩点，以及验证他们的论文已被接受。相比之下，`UnderGradStudent`可能只需要验证他们的学分小时数和整体平均成绩点。

抽象产品类将包括一个静态方法`MatriculateStudent()`作为工厂方法来创建各种类型的学生（具体的产物类型）。

### 定义抽象产品类

让我们先看看实现我们的工厂方法的具体机制，从检查我们的抽象产品类`Student`的定义开始。这个例子可以作为完整的程序在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter17/Chp17-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter17/Chp17-Ex1.cpp)

```cpp
// Assume Person class exists with its usual implementation
class Student: public Person  // Notice that Student is now
{                             // an abstract class
private:
    float gpa = 0.0;  // in-class initialization
    string currentCourse;
    const string studentId;
    static int numStudents;
public:
    Student();  // default constructor
    Student(const string &, const string &, char, 
       const string &, float, const string &, 
       const string &);
    Student(const Student &);  // copy constructor
    ~Student() override;  // virtual destructor
    float GetGpa() const { return gpa; }
    const string &GetCurrentCourse() const 
       { return currentCourse; }
    const string &GetStudentId() const 
       { return studentId; }
    void SetCurrentCourse(const string &); // proto. only
    void Print() const override;
    string IsA() const override { return "Student"; }
    virtual void Graduate() = 0;  // Student is abstract
    // Create a derived Student type based on degree sought
    static Student *MatriculateStudent(const string &,
       const string &, const string &, char, 
       const string &, float, const string &, 
       const string &);
    static int GetNumStudents() { return numStudents; }
};
// Assume all the usual Student member functions exist 
```

在前面的类定义中，我们引入了我们的抽象`Student`类，它从`Person`（一个具体类，因此可以实例化）派生而来。这是通过引入抽象方法`virtual void Graduate() = 0;`实现的。在我们的学生注册示例中，我们将遵循这样的设计决策：只有特定类型的学生的实例应该被创建，即派生类类型`GradStudent`、`UnderGradStudent`或`NonDegreeStudent`。

在前面的类定义中，请注意我们的工厂方法，其原型为`static Student *MatriculateStudent();`。这个方法将使用统一的接口，并提供创建`Student`的各种派生类类型的手段。一旦我们看到了派生类的类定义，我们将详细研究这个方法。

### 定义具体产品类

现在，让我们来看看我们的具体产品类，从`GradStudent`开始：

```cpp
class GradStudent: public Student
{
private:
    string degree;  // PhD, MS, MA, etc.
public:
    GradStudent() = default;// default constructor
    GradStudent(const string &, const string &, 
       const string &, char, const string &, float, 
       const string &, const string &);
    // Prototyping default copy constructor isn't necessary
    // GradStudent(const GradStudent &) = default;
    // Since the most base class has virt dtor prototyped,
    // it is not necessary to prototype default destructor
    // ~GradStudent() override = default; // virtual dtor
    void EarnPhD();
    string IsA() const override { return "GradStudent"; }
    void Graduate() override;
};
// Assume alternate constructor is implemented
// as expected. See online code for full implementation.
void GradStudent::EarnPhD()
{
    if (!degree.compare("PhD")) // only PhD candidates can 
        ModifyTitle("Dr.");     // EarnPhd(), not MA and MS 
}                               // candidates
void GradStudent::Graduate()
{   // Here, we can check that the required num of credits
    // have been met with a passing gpa, and that their 
    // doctoral or master's thesis has been completed.
    EarnPhD(); // Will change title only if a PhD candidate
    cout << "GradStudent::Graduate()" << endl;
}
```

在上述`GradStudent`类定义中，我们添加了一个`degree`数据成员来表示`"PhD"`、`"MS"`或`"MA"`学位，并根据需要调整构造函数和析构函数。我们将`EarnPhD()`移动到`GradStudent`中，因为这个方法并不适用于所有`Student`实例。相反，`EarnPhD()`适用于`GradStudent`实例的一个子集；我们只会授予`"Dr."`头衔给博士候选人。

在这个类中，我们重写了`IsA()`以返回`"GradStudent"`。我们还重写了`Graduate()`，以执行适用于研究生的毕业清单，如果清单项目已经满足，则调用`EarnPhD()`。

现在，让我们来看看我们的下一个具体产品类，`UnderGradStudent`：

```cpp
class UnderGradStudent: public Student
{
private:
    string degree;  // BS, BA, etc
public:
    UnderGradStudent() = default;// default constructor
    UnderGradStudent(const string &, const string &, 
       const string &, char, const string &, float, 
       const string &, const string &);
    // Prototyping default copy constructor isn't necessary
    // UnderGradStudent(const UnderGradStudent &) =default; 
    // Since the most base class has virt dtor prototyped,
    // it is not necessary to prototype default destructor
    // ~UnderGradStudent() override = default; // virt dtor
    string IsA() const override 
       { return "UnderGradStudent"; }
    void Graduate() override;
};
// Assume alternate constructor is implemented
// as expected. See online code for full implementation.
void UnderGradStudent::Graduate()
{   // Verify that num of credits and gpa requirements have
    // been met for major and any minors or concentrations.
    // Have all applicable university fees been paid?
    cout << "UnderGradStudent::Graduate()" << endl;
}
```

快速看一下之前定义的`UnderGradStudent`类，我们会发现它非常类似于`GradStudent`。这个类甚至包括一个`degree`数据成员。记住，并不是所有的`Student`实例都会获得学位，所以我们不希望在`Student`中定义这个属性，从而将其泛化。虽然我们可以为`UnderGradStudent`和`GradStudent`引入一个共享的基类`DegreeSeekingStudent`来收集这种共性，但这种细粒度的层次结构几乎是不必要的。这里的重复是一个设计权衡。

这两个兄弟类之间的关键区别在于重写的`Graduate()`方法。我们可以想象，本科生毕业的清单可能和研究生毕业的清单有很大不同。因此，我们可以合理地区分这两个类。否则，它们非常相似。

现在，让我们来看看我们的下一个具体产品类，`NonDegreeStudent`：

```cpp
class NonDegreeStudent: public Student
{
public:
    NonDegreeStudent() = default;  // default constructor
    NonDegreeStudent(const string &, const string &, char, 
       const string &, float, const string &, 
       const string &);
    // Prototyping default copy constructor isn't necessary
    // NonDegreeStudent(const NonDegreeStudent &s)
    //     =default;
    // Since the most base class has virt dtor prototyped,
    // it is not necessary to prototype default destructor
    // ~NonDegreeStudent() override = default; // virt dtor
    string IsA() const override  
       { return "NonDegreeStudent"; }
    void Graduate() override;
};
// Assume alternate constructor is implemented as expected.
// See online code for full implementation.
void NonDegreeStudent::Graduate()
{   // Check if applicable tuition has been paid. 
    // There is no credit or gpa requirement.
    cout << "NonDegreeStudent::Graduate()" << endl;
}
```

快速看一下前面提到的`NonDegreeStudent`类，我们会注意到这个具体产品与其兄弟类相似。然而，这个类中没有`degree`数据成员。此外，重写的`Graduate()`方法比`GradStudent`或`UnderGradStudent`类中此方法的重写版本需要进行的验证要少。

### 检查工厂方法定义

接下来，让我们看看我们的工厂方法，这是我们的产品（`Student`）类中的一个静态方法：

```cpp
// Creates a Student based on the degree they seek
// This is a static Student method (keyword in prototype)
Student *Student::MatriculateStudent(const string &degree, 
    const string &fn, const string &ln, char mi, 
    const string &t, float avg, const string &course, 
    const string &id)
{
    if (!degree.compare("PhD") || !degree.compare("MS") 
        || !degree.compare("MA"))
        return new GradStudent(degree, fn, ln, mi, t, avg,
                               course, id);
    else if (!degree.compare("BS") || 
             !degree.compare("BA"))
        return new UnderGradStudent(degree, fn, ln, mi, t,
                                    avg, course, id);
    else if (!degree.compare("None"))
        return new NonDegreeStudent(fn, ln, mi, t, avg,
                                    course, id);
}
```

前面提到的`Student`类的静态方法`MatriculateStudent()`代表了工厂方法来创建各种产品（具体的`Student`实例）。在这里，根据`Student`寻求的学位类型，将实例化`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`之一。注意，`MatriculateStudent()`的签名可以处理任何派生类构造函数的参数要求。也请注意，这些专门的实例类型将作为抽象产品类型（`Student`）的基类指针返回。

工厂方法中的一个有趣选项是`MatriculateStudent()`，这个方法并不强制实例化一个新的派生类实例。相反，它可能回收一个可能仍然可用的先前实例。例如，想象一个`Student`因为延迟付款而暂时在大学中未注册，但仍然保留在*待处理学生*名单上。`MatriculateStudent()`方法可以选择返回这样一个现有`Student`的指针。*回收*是工厂方法中的一个替代方案！

### 将模式组件组合在一起

最后，现在让我们通过查看`main()`函数来将所有各种组件组合在一起，看看我们的工厂方法模式是如何编排的：

```cpp
int main()
{
    Student *scholars[MAX] = { }; // init. to nullptrs
    // Student is now abstract; cannot instantiate directly
    // Use Factory Method to make derived types uniformly
    scholars[0] = Student::MatriculateStudent("PhD", 
       "Sara", "Kato", 'B', "Ms.", 3.9, "C++", "272PSU");
    scholars[1] = Student::MatriculateStudent("BS", 
       "Ana", "Sato", 'U', "Ms.", 3.8, "C++", "178PSU");
    scholars[2] = Student::MatriculateStudent("None", 
       "Elle", "LeBrun", 'R', "Miss", 3.5, "C++", "111BU");
    for (auto *oneStudent : scholars)
    {
       oneStudent->Graduate();
       oneStudent->Print();
    }
    for (auto *oneStudent : scholars)
       delete oneStudent;   // engage virt dtor sequence
    return 0;
}
```

回顾我们前面提到的`main()`函数，我们首先创建了一个指针数组，用于潜在的专门`Student`实例，以它们的泛化`Student`形式存在。接下来，我们在抽象产品类中调用静态工厂方法`Student::MatriculateStudent()`，以创建适当的具体产品（派生`Student`类类型）。我们创建了每种派生`Student`类型的一个实例——`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`。

我们随后遍历我们的泛化集合，对每个实例调用`Graduate()`方法，然后调用`Print()`方法。对于获得博士学位的学生（`GradStudent`实例），他们的头衔将通过`GradStudent::Graduate()`方法更改为`"Dr."`。最后，我们通过另一个循环来释放每个实例的内存。幸运的是，`Student`类包含了一个虚析构函数，这样销毁序列就会从正确的级别开始。

让我们看看这个程序的输出：

```cpp
GradStudent::Graduate()
  Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
UnderGradStudent::Graduate()
  Ms. Ana U. Sato with id: 178PSU GPA:  3.8 Course: C++
NonDegreeStudent::Graduate()
  Miss Elle R. LeBrun with id: 111BU GPA:  3.5 Course: C++
```

之前实现的一个优点是它非常直接。然而，我们可以看到抽象 Product（包含 Factory Method，它构建派生类类型）和派生具体 Product 之间存在紧密耦合。然而，在面向对象编程中，基类理想情况下对任何子类类型一无所知。

这种紧密耦合的实现的一个缺点是，抽象的 Product 类必须在它的静态创建方法`MatriculateStudent()`中包含一个实例化的方法，用于其每个子类。现在添加新的派生类会影响抽象基类定义——它需要重新编译。如果我们无法访问这个基类的源代码怎么办？有没有一种方法可以解耦 Factory Method 和 Factory Method 将要创建的 Products 之间的依赖关系？是的，有一种替代实现。

让我们现在看看 Factory Method 模式的另一种实现。我们将使用一个 Object Factory 类来封装我们的`MatriculateStudent()`Factory Method，而不是将其包含在抽象 Product 类中。

## 创建一个封装 Factory Method 的对象工厂类

对于我们的 Factory Method 模式的替代实现，我们将创建我们的抽象 Product 类，与之前的定义略有不同。然而，我们仍然会像以前一样创建我们的具体 Product 类。这些类定义共同构成了我们模式的基础框架。

在我们的修改后的例子中，我们将再次将 Product 定义为`Student`类。我们也将再次派生`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`的具体产品类。然而，这一次，我们不会在我们的 Product（`Student`）类中包含 Factory Method。相反，我们将创建一个单独的对象工厂类，该类将包含 Factory Method。像以前一样，Factory Method 将有一个统一的接口来创建任何派生产品类型。Factory Method 不需要是静态的，就像我们上一个实现中那样。

我们的 Object Factory 类将包括`MatriculateStudent()`作为 Factory Method 来创建各种类型的`Student`实例（具体产品类型）。

### 不包含 Factory Method 的抽象 Product 类定义

让我们来看看 Factory Method 模式替代实现的机制，首先从我们的抽象 Product 类`Student`的定义开始。这个例子可以作为完整的程序，在我们的 GitHub 仓库中找到，以下 URL：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter17/Chp17-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter17/Chp17-Ex2.cpp)

```cpp
// Assume Person class exists with its usual implementation
class Student: public Person   // Notice Student is 
{                              // an abstract class
private:
    float gpa = 0.0;   // in-class initialization
    string currentCourse;
    const string studentId;
    static int numStudents; // Remember, static data mbrs 
                // are also shared by all derived instances
public:          
    Student();  // default constructor
    Student(const string &, const string &, char, 
       const string &, float, const string &, 
       const string &);
    Student(const Student &);  // copy constructor
    ~Student() override;  // destructor
    float GetGpa() const { return gpa; }
    const string &GetCurrentCourse() const 
       { return currentCourse; }
    const string &GetStudentId() const 
       { return studentId; }
    void SetCurrentCourse(const string &); // proto. only
    void Print() const override;
    string IsA() const override { return "Student"; }
    virtual void Graduate() = 0;  // Student is abstract
    static int GetNumStudents() { return numStudents; }
};
```

在我们之前提到的`Student`类定义中，与之前的实现相比，关键的不同之处在于这个类不再包含一个静态的`MatriculateStudent()`方法作为工厂方法。`Student`仅仅是一个抽象基类。记住，所有的研究生、本科生和非学位学生都是`Student`的特化形式，因此`static int numStudents`是所有`Student`类型的一个共享、集体计数。

### 定义具体产品类

考虑到这一点，让我们看看派生（具体产品）类：

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

在我们之前列出的类定义中，我们可以看到我们的具体派生产品类与我们在第一个示例中的这些类的实现是相同的。

### 添加带有工厂方法的对象工厂类

接下来，让我们介绍一个包含我们的工厂方法的对象工厂类：

```cpp
class StudentFactory    // Object Factory class
{
public:   
   // Factory Method creates Student based on degree sought
    Student *MatriculateStudent(const string &degree, 
       const string &fn, const string &ln, char mi, 
       const string &t, float avg, const string &course, 
       const string &id)
    {
        if (!degree.compare("PhD") || !degree.compare("MS") 
            || !degree.compare("MA"))
            return new GradStudent(degree, fn, ln, mi, t, 
                                   avg, course, id);
        else if (!degree.compare("BS") || 
                 !degree.compare("BA"))
            return new UnderGradStudent(degree, fn, ln, mi,
                                       t, avg, course, id);
        else if (!degree.compare("None"))
            return new NonDegreeStudent(fn, ln, mi, t, avg,
                                        course, id);
    }
};
```

在之前提到的对象工厂类定义（`StudentFactory`类）中，我们最小化地包含了工厂方法规范，即`MatriculateStudent()`。该方法与之前的示例非常相似。然而，通过在对象工厂中捕获具体产品的创建，我们将抽象产品与工厂方法之间的关系解耦了。

### 将模式组件组合在一起

接下来，让我们比较我们的`main()`函数与原始示例，以可视化我们修改后的组件如何实现工厂方法模式：

```cpp
int main()
{
    Student *scholars[MAX] = { }; // init. to nullptrs
    // Create an Object Factory for Students
    StudentFactory *UofD = new StudentFactory();
    // Student is now abstract, cannot instantiate directly
    // Ask the Object Factory to create a Student
    scholars[0] = UofD->MatriculateStudent("PhD", "Sara", 
               "Kato", 'B', "Ms.", 3.9, "C++", "272PSU");
    scholars[1] = UofD->MatriculateStudent("BS", "Ana", 
               "Sato", 'U', "Dr.", 3.8, "C++", "178PSU");
    scholars[2] = UofD->MatriculateStudent("None", "Elle",
               "LeBrun", 'R', "Miss", 3.5, "C++", "111BU");
    for (auto *oneStudent : scholars)
    {
       oneStudent->Graduate();
       oneStudent->Print();
    }
    for (auto *oneStudent : scholars)
       delete oneStudent;   // engage virt dtor sequence
    delete UofD; // delete factory that created various 
    return 0;    // types of students
}
```

考虑我们之前列出的`main()`函数，我们看到我们再次创建了一个指向抽象产品类型（`Student`）的指针数组。然后我们实例化了一个对象工厂，它可以创建各种具体产品类型的`Student`实例，使用`StudentFactory *UofD = new StudentFactory();`。与之前的示例一样，根据每个学生的学位类型，对象工厂创建了每种派生类型`GradStudent`、`UnderGradStudent`和`NonDegreeStudent`的一个实例。`main()`中的其余代码与之前的示例相同。

我们的结果将与我们的上一个示例相同。

与我们之前的方法相比，对象工厂类的优势在于我们消除了从抽象产品类（在工厂方法中）创建对象的知识依赖。也就是说，如果我们扩展我们的层次结构以包括新的具体产品类型，我们不需要修改抽象产品类。当然，我们需要能够修改我们的对象工厂类`StudentFactory`，以增强我们的`MatriculateStudent()`工厂方法。

与此实现相关的模式，即**抽象工厂**，是一种允许具有相似目的的单独工厂被分组在一起的附加模式。抽象工厂可以指定提供一种统一类似对象工厂的方法；它是一个将创建工厂的工厂，为我们的原始模式增加了另一个抽象层次。

我们现在已经看到了工厂方法模式的两种实现。我们将产品和工厂方法的概念融合到了我们习惯看到的类框架中，即`Student`及其`Student`的派生类。现在，让我们简要回顾一下与模式相关的内容，然后再继续下一章。

# 摘要

在本章中，我们通过扩展我们对设计模式的知识，继续追求成为更好的 C++ 程序员。特别是，我们探讨了工厂方法模式，从概念上以及通过两种常见的实现方式进行了研究。我们的第一个实现是将工厂方法放置在我们的抽象产品类中。我们的第二个实现通过添加一个包含工厂方法的对象工厂类，而不是在抽象产品和工厂方法之间建立依赖关系，从而消除了这种依赖。我们还非常简短地讨论了抽象工厂的概念。

利用常见的模式，如工厂方法模式，将帮助您更轻松地以其他程序员能理解的方式解决重复出现的编程问题。通过利用核心设计模式，您将为具有更复杂编程技术的可理解和可重用解决方案做出贡献。

我们现在准备继续前进，学习下一个设计模式，即*第十八章*中的*实现适配器模式*。将更多模式添加到我们的技能集合中，使我们成为更灵活、更有价值的程序员。让我们继续前进！

# 问题

1.  使用之前练习的解决方案（*问题 1*，*第八章*，*掌握抽象类*），按照以下方式增强您的代码：

    1.  实现工厂方法模式以创建各种形状。你将已经创建了一个抽象基类`Shape`以及派生类，如`Rectangle`、`Circle`、`Triangle`，以及可能的`Square`。

    1.  选择是否将您的工厂方法实现为`Shape`中的静态方法，或者作为`ShapeFactory`类中的方法（如果需要，引入此类）。

1.  你能想象出哪些其他例子可以轻松地结合工厂方法模式？
