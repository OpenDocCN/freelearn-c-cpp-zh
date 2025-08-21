# 第十八章：应用适配器模式

本章将扩展我们的探索，超越核心面向对象编程概念，旨在使您能够利用常见的设计模式解决重复出现的编码问题。在编码解决方案中应用设计模式不仅可以提供优雅的解决方案，还可以增强代码的维护性，并为代码重用提供潜在机会。

我们将学习如何在 C++中有效实现**适配器模式**。

在本章中，我们将涵盖以下主要主题：

+   理解适配器模式及其对面向对象编程的贡献

+   理解如何在 C++中实现适配器模式

本章结束时，您将了解基本的适配器模式以及如何使用它来允许两个不兼容的类进行通信，或者将不合适的代码升级为设计良好的面向对象代码。向您的知识库中添加另一个关键设计模式将使您的编程技能得到提升，帮助您成为更有价值的程序员。

让我们通过研究另一个常见的设计模式，即适配器模式，来增加我们的编程技能。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与所在章节编号相对应，后跟该章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录中的`Chapter18`子目录中的名为`Chp18-Ex1.cpp`的文件中找到。

本章的 CiA 视频可在以下链接观看：[`bit.ly/2Pfg9VA`](https://bit.ly/2Pfg9VA)。

# 理解适配器模式

**适配器模式**是一种结构设计模式，提供了一种将现有类的不良接口转换为另一个类所期望的接口的方法。**适配器类**将成为两个现有组件之间通信的链接，调整接口以便两者可以共享和交换信息。适配器允许两个或更多类一起工作，否则它们无法这样做。

理想情况下，适配器不会添加功能，而是会添加所需的接口以便允许一个类以预期的方式使用，或者使两个不兼容的类相互通信。在其最简单的形式中，适配器只是将现有的类转换为支持 OO 设计中可能指定的预期接口。

适配器可以与其提供自适应接口的类相关联或派生自该类。如果使用继承，适合使用私有或受保护的基类来隐藏底层实现。如果适配器类与具有不良接口的类相关联，适配器类中的方法（具有新接口）将仅将工作委托给其关联类。

适配器模式还可以用于为一系列函数或其他类添加 OO 接口（即*在一系列函数或其他类周围包装 OO 接口*），从而使各种现有组件在 OO 系统中更自然地被利用。这种特定类型的适配器称为`extern C`，以允许链接器解析两种语言之间的链接约定。

利用适配器模式有好处。适配器允许通过提供共享接口来重用现有代码，以便否则无关的类进行通信。面向对象的程序员现在可以直接使用适配器类，从而更容易地维护应用程序。也就是说，大多数程序员的交互将是与设计良好的适配器类，而不是与两个或更多奇怪的组件。使用适配器的一个小缺点是由于增加了代码层，性能略有下降。然而，通常情况下，通过提供清晰的接口来支持它们的交互来重用现有组件是一个成功的选择，尽管会有（希望是小的）性能折衷。

适配器模式将包括以下内容：

+   一个**Adaptee**类，代表具有可取用功能的类，但以不合适或不符合预期的形式存在。

+   一个**适配器**类，它将适配 Adaptee 类的接口以满足所需接口的需求。

+   一个**目标**类，代表应用程序所需接口的具体接口。一个类可以既是目标又是适配器。

+   可选的**客户端**类，它们将与目标类交互，以完全定义正在进行的应用程序。

适配器模式允许重用合格的现有组件，这些组件不符合当前应用程序设计的接口需求。

让我们继续看适配器模式的两个常见应用；其中一个将有两种潜在的实现方式。

# 实现适配器模式

让我们探讨适配器模式的两种常见用法。即，创建一个适配器来弥合两个不兼容的类接口之间的差距，或者创建一个适配器来简单地用 OO 接口包装一组现有函数。

我们将从使用*适配器*提供连接器来连接两个（或更多）不兼容的类开始。*Adaptee*将是一个经过充分测试的类，我们希望重用它（但它具有不理想的接口），*Target*类将是我们在进行中的应用程序的 OO 设计中指定的类。现在让我们指定一个适配器，以使我们的 Adaptee 能够与我们的 Target 类一起工作。

## 使用适配器为现有类提供必要的接口

要实现适配器模式，我们首先需要确定我们的 Adaptee 类。然后我们将创建一个适配器类来修改 Adaptee 的接口。我们还将确定我们的 Target 类，代表我们需要根据我们的 OO 设计来建模的类。有时，我们的适配器和目标可能会合并成一个单一的类。在实际应用中，我们还将有客户端类，代表着最终应用程序中的所有类。让我们从 Adaptee 和 Adapter 类开始，因为这些类定义将为我们构建模式奠定基础。

在我们的例子中，我们将指定我们习惯看到的 Adaptee 类为`Person`。我们将想象我们的星球最近意识到许多其他能够支持生命的系外行星，并且我们已经与每个文明友好地结盟。进一步想象，地球上的各种软件系统希望欢迎和包容我们的新朋友，包括`Romulans`和`Orkans`，我们希望调整一些现有软件以轻松适应我们系外行星邻居的新人口统计。考虑到这一点，我们将通过创建一个适配器类`Humanoid`来将我们的`Person`类转换为包含更多系外行星术语。

在我们即将实现的代码中，我们将使用私有继承来从`Person`（被适配者）继承`Humanoid`（适配器），从而隐藏被适配者的底层实现。我们也可以将`Humanoid`关联到`Person`（这也是我们将在本节中审查的一种实现）。然后，我们可以在我们的层次结构中完善一些`Humanoid`的派生类，比如`Orkan`、`Romulan`和`Earthling`，以适应手头的星际应用。`Orkan`、`Romulan`和`Earthling`类可以被视为我们的目标类，或者我们的应用将实例化的类。我们选择将我们的适配器类`Humanoid`设为抽象，以便它不能直接实例化。因为我们的具体派生类（目标类）可以在我们的应用程序（客户端）中由它们的抽象基类类型（`Humanoid`）进行泛化，所以我们也可以将`Humanoid`视为目标类。也就是说，`Humanoid`可以被视为主要是一个适配器，但次要是一个泛化的目标类。

我们的各种客户端类可以利用`Humanoid`的派生类，创建每个具体后代的实例。这些实例可以存储在它们自己的专门类型中，或者使用`Humanoid`指针进行泛型化。我们的实现是对广泛使用的适配器设计模式的现代化改进。

### 指定被适配者和适配器（私有继承技术）

让我们来看看我们的适配器模式的第一个用法的机制，首先回顾我们的被适配者类`Person`的定义。这个例子可以在我们的 GitHub 存储库中找到一个完整的程序。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex1.cpp)

```cpp
// Person is the Adaptee class; the class requiring adaptation
class Person
{
private:
    char *firstName, *lastName, *title, *greeting;
    char middleInitial;
protected:
    void ModifyTitle(const char *);  
public:
    Person();   // default constructor
    Person(const char *, const char *, char, const char *);  
    Person(const Person &);  // copy constructor
    Person &operator=(const Person &); // assignment operator
    virtual ~Person();  // destructor
    const char *GetFirstName() const { return firstName; }  
    const char *GetLastName() const { return lastName; }    
    const char *GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    void SetGreeting(const char *);
    virtual const char *Speak() { return greeting; }
    virtual void Print();
};
// Assume constructors, destructor, and non-inline methods are 
// implemented as expected (see online code)
```

在前面的类定义中，我们注意到我们的`Person`类定义与本书中许多其他示例中看到的一样。这个类是可实例化的；然而，在我们的星际应用中，`Person`不是一个适当的类来实例化。相反，预期的接口应该是利用`Humanoid`中找到的接口。

考虑到这一点，让我们来看看我们的适配器类`Humanoid`：

```cpp
class Humanoid: private Person   // Humanoid is abstract
{                           
protected:
    void SetTitle(const char *t) { ModifyTitle(t); }
public:
    Humanoid();   
    Humanoid(const char *, const char *, const char *,
             const char *);
    Humanoid(const Humanoid &h) : Person(h) { }  
    Humanoid &operator=(const Humanoid &h) 
        { return (Humanoid &) Person::operator=(h); }
    virtual ~Humanoid() { }  
    const char *GetSecondaryName() const 
        { return GetFirstName(); }  
    const char *GetPrimaryName() const 
        { return GetLastName(); } 
    // scope resolution needed in method to avoid recursion 
    const char *GetTitle() const { return Person::GetTitle();}
    void SetSalutation(const char *m) { SetGreeting(m); }
    virtual void GetInfo() { Print(); }
    virtual const char *Converse() = 0;  // abstract class
};
Humanoid::Humanoid(const char *n2, const char *n1, 
    const char *planetNation, const char *greeting):
    Person(n2, n1, ' ', planetNation)
{
    SetGreeting(greeting);
}
const char *Humanoid::Converse()  // default definition for  
{                           // pure virtual function - unusual                           
    return Speak();
}
```

在上述的`Humanoid`类中，我们的目标是提供一个适配器，以满足我们星际应用所需的接口。我们只需使用私有继承，将`Humanoid`从`Person`派生，将`Person`中的公共接口隐藏在`Humanoid`的范围之外。我们知道目标应用（客户端）不希望`Person`中的公共接口被`Humanoid`的各种子类型实例使用。请注意，我们并没有添加功能，只是在适配接口。

然后，我们注意到`Humanoid`中引入的公共方法，为目标类提供了所需的接口。这些接口的实现通常很简单。我们只需调用`Person`中定义的继承方法，就可以轻松完成手头的任务（但使用了不可接受的接口）。例如，我们的`Humanoid::GetPrimaryName()`方法只是调用`Person::GetLastName();`来完成任务。然而，`GetPrimaryName()`可能更多地代表适当的星际术语，而不是`Person::GetLastName()`。我们可以看到`Humanoid`是作为`Person`的适配器。

请注意，在`Humanoid`方法中调用`Person`基类方法时，不需要在调用前加上`Person::`（除非`Humanoid`方法调用`Person`中同名的方法，比如`GetTitle()`）。`Person::`的作用域解析用法避免了这些情况中的潜在递归。

我们还注意到`Humanoid`引入了一个抽象的多态方法（即纯虚函数），其规范为`virtual const char *Converse() = 0;`。我们已经做出了设计决策，即只有`Humanoid`的派生类才能被实例化。尽管如此，我们理解公共的派生类仍然可以被其基类类型`Humanoid`收集。在这里，`Humanoid`主要作为适配器类，其次作为一个目标类，提供一套可接受的接口。

请注意，我们的纯虚函数`virtual const char *Converse() = 0;`包括一个默认实现。这是罕见的，但只要实现不是内联写的，就是允许的。在这里，我们利用机会通过简单调用`Person::Speak()`来为`Humanoid::Converse()`指定默认行为。

### 从适配器派生具体类

接下来，让我们扩展我们的适配器（`Humanoid`）并看看我们的一个具体的、派生的目标类`Orkan`：

```cpp
class Orkan: public Humanoid
{
public:
    Orkan();   // default constructor
    Orkan(const char *n2, const char *n1, const char *t): 
       Humanoid(n2, n1, t, "Nanu nanu") { }
    Orkan(const Orkan &h) : Humanoid(h) { }  
    Orkan &operator=(const Orkan &h) 
        { return (Orkan &) Humanoid::operator=(h); }
    virtual ~Orkan() { }  
    virtual const char *Converse() override;  
};
const char *Orkan::Converse()  // Must override to make
{                              // Orkan a concrete class
    return Humanoid::Converse(); // use scope resolution to
}                                // avoid recursion
```

在我们前面提到的`Orkan`类中，我们使用公共继承来从`Humanoid`派生`Orkan`。`Orkan` *是一个* `Humanoid`。因此，`Humanoid`中的所有公共接口都对`Orkan`实例可用。请注意，我们的替代构造函数将默认问候消息设置为`"Nanu nanu"`，符合`Orkan`方言。

因为我们希望`Orkan`是一个具体的、可实例化的类，所以我们必须重写`Humanoid::Converse()`并在`Orkan`类中提供一个实现。然而，请注意，`Orkan::Converse()`只是调用了`Humanoid::Converse();`。也许`Orkan`认为其基类中的默认实现是可以接受的。请注意，我们在`Orkan::Converse()`方法中使用`Humanoid::`作用域解析来限定`Converse()`，以避免递归。

有趣的是，如果`Humanoid`不是一个抽象类，`Orkan`就不需要重写`Converse()` - 默认行为会自动继承。然而，由于`Humanoid`被定义为抽象类，所以在`Orkan`中重写`Converse()`是必要的，否则`Orkan`也会被视为抽象类。别担心！我们可以通过在`Orkan::Converse()`中调用`Humanoid::Converse()`来利用`Humanoid::Converse()`的默认行为。这将满足使`Orkan`具体化的要求，同时允许`Humanoid`保持抽象，同时为`Converse()`提供罕见的默认行为！

现在，让我们看一下我们的下一个具体的目标类`Romulan`：

```cpp
class Romulan: public Humanoid
{
public:
    Romulan();   // default constructor
    Romulan(const char *n2, const char *n1, const char *t): 
        Humanoid(n2, n1, t, "jolan'tru") { }
    Romulan(const Romulan &h) : Humanoid(h) { } 
    Romulan &operator=(const Romulan &h) 
        { return (Romulan &) Humanoid::operator=(h); }
    virtual ~Romulan() { }  
    virtual const char *Converse() override;  
};
const char *Romulan::Converse()   // Must override to make
{                                 // Romulan a concrete class
    return Humanoid::Converse();   // use scope resolution to
}                                  // avoid recursion                  
```

快速看一下前面提到的`Romulan`类，我们注意到这个具体的目标与其兄弟类`Orkan`相似。我们注意到传递给我们基类构造函数的默认问候消息是`"jolan'tru"`，以反映`Romulan`方言。虽然我们可以使`Romulan::Converse()`的实现更加复杂，但我们选择不这样做。我们可以快速理解这个类的全部范围。

接下来，让我们看一下我们的第三个目标类`Earthling`：

```cpp
class Earthling: public Humanoid
{
public:
    Earthling();   // default constructor
    Earthling(const char *n2, const char *n1, const char *t):
        Humanoid(n2, n1, t, "Hello") { }
    Earthling(const Romulan &h) : Humanoid(h) { }  
    Earthling &operator=(const Earthling &h) 
        { return (Earthling &) Humanoid::operator=(h); }
    virtual ~Earthling() { }  
    virtual const char *Converse() override;  
};
const char *Earthling::Converse()   // Must override to make
{                                // Earthling a concrete class  
    return Humanoid::Converse();  // use scope resolution to
}                                 // avoid recursion
```

再次快速看一下前面提到的`Earthling`类，我们注意到这个具体的目标与其兄弟类`Orkan`和`Romulan`相似。

现在我们已经定义了我们的被适配者、适配器和多个目标类，让我们通过检查程序的客户端部分来将这些部分组合在一起。

### 将模式组件结合在一起

最后，让我们考虑一下我们整个应用程序中的一个示例客户端可能是什么样子。当然，它可能由许多文件和各种类组成。在其最简单的形式中，如下所示，我们的客户端将包含一个`main()`函数来驱动应用程序。

现在让我们看一下我们的`main()`函数，看看我们的模式是如何被编排的：

```cpp
int main()
{
    list<Humanoid *> allies;
    Orkan *o1 = new Orkan("Mork", "McConnell", "Orkan");
    Romulan *r1 = new Romulan("Donatra", "Jarok", "Romulan");
    Earthling *e1 = new Earthling("Eve", "Xu", "Earthling");
    // Add each specific type of Humanoid to the generic list
    allies.push_back(o1);
    allies.push_back(r1);
    allies.push_back(e1);
    // Create a list iterator; set to first item in the list
    list <Humanoid *>::iterator listIter = allies.begin();
    while (listIter != allies.end())
    {
        (*listIter)->GetInfo();
        cout << (*listIter)->Converse() << endl;
        listIter++;
    }
    // Though each type of Humanoid has a default Salutation,
    // each may expand their skills with an alternate language
    e1->SetSalutation("Bonjour");
    e1->GetInfo();
    cout << e1->Converse() << endl;  // Show the Earthling's 
                             // revised language capabilities
    delete o1;   // delete the heap instances
    delete r1;
    delete e1;
    return 0;
}
```

回顾我们上述的`main()`函数，我们首先创建一个`STL` `list` of `Humanoid`指针，使用`list<Humanoid *> allies;`。然后，我们实例化一个`Orkan`，`Romulan`和一个`Earthling`，并使用`allies.push_back()`将它们添加到列表中。再次使用`STL`，我们接下来创建一个列表迭代器，以遍历指向`Humanoid`实例的指针列表。当我们遍历我们的盟友的通用列表时，我们对列表中的每个项目调用`GetInfo()`和`Converse()`的批准接口（也就是说，对于每种特定类型的`Humanoid`）。

接下来，我们指定一个特定的`Humanoid`，一个`Earthling`，并通过调用`e1->SetSalutation("Bonjour");`来更改这个实例的默认问候语。通过再次在这个实例上调用`Converse()`（我们首先在上述循环中以通用方式这样做），我们可以请求`Earthling`使用`"Bonjour"`来向盟友打招呼，而不是使用`"Hello"`（`Earthling`的默认问候语）。

让我们来看看这个程序的输出：

```cpp
Orkan Mork McConnell
Nanu nanu
Romulan Donatra Jarok
jolan'tru
Earthling Eve Xu
Hello
Earthling Eve Xu
Bonjour
```

在上述输出中，请注意每个`Humanoid`的行星规格（`Orkan`，`Romulan`，`Earthling`），然后显示它们的次要和主要名称。然后显示特定`Humanoid`的适当问候语。请注意，`Earthling` `Eve Xu`首先使用`"Hello"`进行对话，然后稍后使用`"Bonjour"`进行对话。

前述实现的优点（使用私有基类从 Adaptee 派生 Adapter）是编码非常简单。通过这种方法，Adaptee 类中的任何受保护的方法都可以轻松地传递下来在 Adapter 方法的范围内使用。我们很快会看到，如果我们改用关联作为连接 Adapter 到 Adaptee 的手段，受保护的成员将成为一个问题。

前述方法的缺点是它是一个特定于 C++的实现。其他语言不支持私有基类。另外，使用公共基类来定义 Adapter 和 Adaptee 之间的关系将无法隐藏不需要的 Adaptee 接口，并且是一个非常糟糕的设计选择。

### 考虑 Adaptee 和 Adapter 的替代规范（关联）

现在，让我们简要地考虑一下稍微修改过的上述 Adapter 模式实现。我们将使用关联来模拟 Adaptee 和 Adapter 之间的关系。具体的派生类（Targets）仍将像以前一样从 Adapter 派生。

这是我们 Adapter 类`Humanoid`的另一种实现，使用 Adapter 和 Adaptee 之间的关联。虽然我们只会审查与我们最初的方法不同的代码部分，但完整的实现可以在我们的 GitHub 上找到作为一个完整的程序：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex2.cpp)

```cpp
// Assume that Person exists mostly as before – however,
// Person::ModifyTitle() must be moved from protected to
// public - or be unused if modifying Person is not possible.
// Let's assume we moved Person::ModifyTitle() to public.
class Humanoid    // Humanoid is abstract
{
private:
    Person *life;  // delegate all requests to assoc. object
protected:
    void SetTitle(const char *t) { life->ModifyTitle(t); }
public:
    Humanoid() { life = 0; }
    Humanoid(const char *, const char *, const char *, 
             const char *);
    Humanoid(const Humanoid &h);
    Humanoid &operator=(const Humanoid &);
    virtual ~Humanoid() { delete life; }  
    const char *GetSecondaryName() const 
        { return life->GetFirstName(); }
    const char *GetPrimaryName() const 
        { return life->GetLastName(); }    
    const char *GetTitle() const { return life->GetTitle(); }
    void SetSalutation(const char *m) { life->SetGreeting(m);}
    virtual void GetInfo() { life->Print(); }
    virtual const char *Converse() = 0;  // abstract class
};
Humanoid::Humanoid(const char *n2, const char *n1, 
          const char *planetNation, const char *greeting)
{
    life = new Person(n2, n1, ' ', planetNation);
    life->SetGreeting(greeting);
}
Humanoid::Humanoid(const Humanoid &h)
{  // Remember life data member is of type Person
    delete life;  // delete former associated object
    life = new Person(h.GetSecondaryName(),
                      h.GetPrimaryName(),' ', h.GetTitle());
    life->SetGreeting(h.life->Speak());  
}
Humanoid &Humanoid::operator=(const Humanoid &h)
{
    if (this != &h)
        life->Person::operator=((Person &) h);
    return *this;
}
const char *Humanoid::Converse() //default definition for
{                                // pure virtual fn - unusual
    return life->Speak();
}
```

请注意，在我们上述的 Adapter 类的实现中，`Humanoid`不再是从`Person`派生的。相反，`Humanoid`将添加一个私有数据成员`Person *life;`，它将表示 Adapter（`Humanoid`）和 Adaptee（`Person`）之间的关联。在我们的 Humanoid 构造函数中，我们需要分配 Adaptee（`Person`）的基础实现。我们还需要在析构函数中删除 Adaptee（`Person`）。

与我们上次的实现类似，`Humanoid`在其公共接口中提供相同的成员函数。但是，请注意，每个`Humanoid`方法通过关联对象委托调用适当的 Adaptee 方法。例如，`Humanoid::GetSecondaryName()`仅调用`life->GetFirstName();`来委托请求（而不是调用继承的相应 Adaptee 方法）。

与我们最初的实现一样，我们从`Humanoid`（`Orkan`，`Romulan`和`Earthling`）派生的类以相同的方式指定，我们的客户端也在`main()`函数中。

### 选择被适配者和适配器之间的关系

在选择适配器和被适配者之间的关系时，一个有趣的点是选择私有继承还是关联的关系，这取决于被适配者是否包含任何受保护的成员。回想一下，`Person`的原始代码包括一个受保护的`ModifyTitle()`方法。如果被适配者类中存在受保护的成员，私有基类实现允许在适配器类的范围内继续访问这些继承的受保护成员（也就是适配器的方法）。然而，使用基于关联的实现，被适配者（`Person`）中的受保护方法在适配器的范围内是无法使用的。为了使这个例子工作，我们需要将`Person::ModifyTitle()`移到公共访问区域。然而，修改被适配者类并不总是可能的，也不一定推荐。考虑到受保护成员的问题，我们最初使用私有基类的实现是更强大的实现，因为它不依赖于我们修改被适配者（`Person`）的类定义。

现在让我们简要地看一下适配器模式的另一种用法。我们将简单地使用一个适配器类作为包装类。我们将为一个本来松散排列的一组函数添加一个面向对象的接口，这些函数工作得很好，但缺乏我们的应用程序（客户端）所需的接口。

## 使用适配器作为包装器

作为适配器模式的另一种用法，我们将在一组相关的外部函数周围包装一个面向对象的接口。也就是说，我们将创建一个包装类来封装这些函数。

在我们的示例中，外部函数将代表一套现有的数据库访问函数。我们将假设核心数据库功能对于我们的数据类型（`Person`）已经经过了充分测试，并且已经被无问题地使用。然而，这些外部函数本身提供了一个不可取和意外的功能接口。

相反，我们将通过创建一个适配器类来封装这些外部函数的集体功能。我们的适配器类将是`CitizenDataBase`，代表了一个封装的方式，用于从数据库中读取和写入`Person`实例。我们现有的外部函数将为我们的`CitizenDataBase`成员函数提供实现。让我们假设在我们的适配器类中定义的面向对象的接口满足我们的面向对象设计的要求。

让我们来看看我们简单包装的适配器模式的机制，首先要检查提供数据库访问功能的外部函数。这个例子可以在我们的 GitHub 仓库中找到一个完整的程序：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter18/Chp18-Ex3.cpp)

```cpp
// Assume Person class exists with its usual implementation
Person objectRead;  // holds the object from the current read
                    // to support a simulation of a DB read
void db_open(const char *dbName)
{   // Assume implementation exists
    cout << "Opening database: " << dbName << endl;
}
void db_close(const char *dbName)
{   // Assume implementation exists
    cout << "Closing database: " << dbName << endl;
}
Person &db_read(const char *dbName, const char *key)
{   // Assume implementation exists
    cout << "Reading from: " << dbName << " using key: ";
    cout << key << endl;
    // In a true implementation, we would read the data
    // using the key and return the object we read in
    return objectRead;  // a non-stack instance for simulation
}
const char *db_write(const char *dbName, Person &data)
{   // Assume implementation exists
    const char *key = data.GetLastName();
    cout << "Writing: " << key << " to: " << dbName << endl;
    return key;
}
```

在我们之前定义的外部函数中，让我们假设所有函数都经过了充分测试，并且允许从数据库中读取或写入`Person`实例。为了支持这个模拟，我们创建了一个外部`Person`实例`Person objectRead;`，提供了一个简短的、非堆栈位置的存储位置，用于新读取的实例（被`db_read()`使用），直到新读取的实例被捕获为返回值。请记住，现有的外部函数并不代表一个封装的解决方案。

现在，让我们创建一个简单的包装类来封装这些外部函数。包装类`CitizensDataBase`将代表我们的适配器类：

```cpp
// CitizenDataBase is the Adapter class 
class CitizenDataBase  (Adapter wraps the undesired interface)
{
private:
    char *name;
public:
    // No default constructor (unusual)
    CitizenDataBase(const char *);
    CitizenDataBase(const CitizenDataBase &) = delete;
    CitizenDataBase &operator=(const CitizenDataBase &) 
                               = delete;  
    virtual ~CitizenDataBase();  
    Person &Read(const char *);
    const char *Write(Person &);
};
CitizenDataBase::CitizenDataBase(const char *n)
{
    name = new char [strlen(n) + 1];
    strcpy(name, n);
    db_open(name);   // call existing external function
}
CitizenDataBase::~CitizenDataBase()
{
    db_close(name);  // close database with external function
    delete name;
}
Person &CitizenDataBase::Read(const char *key)
{
    return db_read(name, key);   // call external function
}
const char *CitizenDataBase::Write(Person &data)
{
    return db_write(name, data);  // call external function
}
```

在我们上述的适配器类定义中，我们只是在`CitizenDataBase`类中封装了外部数据库功能。在这里，`CitizenDataBase`不仅是我们的适配器类，也是我们的目标类，因为它包含了我们手头应用程序（客户端）期望的接口。

现在，让我们来看看我们的`main()`函数，这是一个客户端的简化版本：

```cpp
int main()
{
    const char *key;
    char name[] = "PersonData"; // name of database
    Person p1("Curt", "Jeffreys", 'M', "Mr.");
    Person p2("Frank", "Burns", 'W', "Mr.");
    Person p3;
    CitizenDataBase People(name);   // open requested Database
    key = People.Write(p1); // write a Person object
    p3 = People.Read(key);  // using a key, retrieve Person
    return 0;
}                           // destruction will close database
```

在上述的`main()`函数中，我们首先实例化了三个`Person`实例。然后实例化了一个`CitizenDataBase`，以提供封装的访问权限，将我们的`Person`实例写入或从数据库中读取。我们的`CitizenDataBase`构造函数的方法调用外部函数`db_open()`来打开数据库。同样，析构函数调用`db_close()`。正如预期的那样，我们的`CitizenDataBase`的`Read()`和`Write()`方法分别调用外部函数`db_read()`或`db_write()`。

让我们来看看这个程序的输出：

```cpp
Opening database: PersonData
Writing: Jeffreys to: PersonData
Reading from: PersonData using key: Jeffreys
Closing database: PersonData
```

在上述输出中，我们可以注意到各个成员函数与包装的外部函数之间的相关性，通过构造、调用写入和读取，然后销毁数据库。

我们简单的`CitizenDataBase`包装器是适配器模式的一个非常简单但合理的用法。有趣的是，我们的`CitizenDataBase`也与**数据访问对象模式**有共同之处，因为这个包装器提供了一个干净的接口来访问数据存储机制，隐藏了对底层数据库的实现（访问）。

我们现在已经看到了适配器模式的三种实现。我们已经将适配器、被适配者、目标和客户端的概念融入到我们习惯看到的类的框架中，即`Person`，以及我们适配器的后代（`Orkan`、`Romulan`、`Earthling`，就像我们前两个例子中的那样）。让我们现在简要地回顾一下我们在移动到下一章之前学到的与模式相关的知识。

# 总结

在本章中，我们通过扩展我们对设计模式的知识，进一步提高了成为更好的 C++程序员的追求。我们已经在概念和多种实现中探讨了适配器模式。我们的第一个实现使用私有继承从被适配者类派生适配器。我们将适配器指定为抽象类，然后使用公共继承根据适配器类提供的接口引入了几个基于接口的目标类。我们的第二个实现则使用关联来建模适配器和被适配者之间的关系。然后我们看了一个适配器作为包装器的示例用法，简单地为现有基于函数的应用组件添加了面向对象的接口。

利用常见的设计模式，比如适配器模式，将帮助你更容易地重用现有的经过充分测试的代码部分，以一种其他程序员能理解的方式。通过利用核心设计模式，你将为更复杂的编程技术做出贡献，提供了被理解和可重用的解决方案。

我们现在准备继续前进，进入我们的下一个设计模式[*第十九章*]，*使用单例模式*。增加更多的模式到我们的编程技能库中，使我们成为更多才多艺和有价值的程序员。让我们继续前进！

# 问题

1.  使用本章中找到的适配器示例：

a. 实现一个`CitizenDataBase`，用于存储各种类型的`Humanoid`实例（`Orkan`、`Romulan`、`Earthling`，也许还有`Martian`）。决定你是使用私有基类适配器-被适配者关系，还是适配器和被适配者之间的关联关系（提示：私有继承版本会更容易）。

b. 注意`CitizenDataBase`处理`Person`实例，这个类是否可以原样用来存储各种类型的`Humanoid`实例，还是必须以某种方式进行适配？请记住，`Person`是`Humanoid`的基类（如果你选择了这种实现方式），但也要记住我们永远不能向上转型超出非公共继承边界。

1.  你能想象哪些其他例子可能很容易地应用适配器模式？
