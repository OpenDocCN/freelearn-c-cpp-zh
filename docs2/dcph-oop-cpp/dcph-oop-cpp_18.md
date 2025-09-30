

# 应用适配器模式

本章将扩展我们的探索，旨在将你的 C++编程技能扩展到核心面向对象概念之外，目标是使你能够利用常见设计模式解决重复出现的编程问题。在编码解决方案中采用设计模式不仅可以提供优雅的解决方案，还可以提高代码维护性，并提供代码重用的潜在机会。

我们接下来将学习如何在 C++中有效地实现下一个核心设计模式——**适配器模式**。

在本章中，我们将涵盖以下主要内容：

+   理解适配器模式及其对面向对象编程（OOP）的贡献

+   理解如何在 C++中实现适配器模式

到本章结束时，你将理解基本的适配器模式以及如何使用它来允许两个不兼容的类进行通信，或者将不合适的代码升级为良好的面向对象代码。将另一个关键设计模式添加到你的知识库中，将提高你的编程技能，帮助你成为一个更有价值的程序员。

让我们通过研究另一个常见的设计模式——适配器模式，来增加我们的编程技能集。

# 技术要求

在以下 GitHub URL 中可以找到完整程序示例的在线代码：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter18`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter18)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的`Chapter18`子目录中的名为`Chp18-Ex1.cpp`的文件中找到。

本章的 CiA 视频可在以下链接查看：[`bit.ly/3Kaxckc`](https://bit.ly/3Kaxckc)。

# 理解适配器模式

**适配器模式**是一种结构型设计模式，它提供了一种将现有类的不理想接口转换为另一个类期望的接口的方法。**适配器类**将是两个现有组件之间通信的链接，通过适配接口使得这两个组件可以共享和交换信息。适配器允许两个或更多类协同工作，否则它们无法这样做。

理想情况下，适配器不会添加功能，而是添加使用（或转换）的首选接口，以便允许一个类以预期的方式使用，或者使两个原本不兼容的类能够相互通信。在其最简单的形式中，适配器只是将现有的类转换为支持预期的接口，正如在面向对象设计中可能指定的那样。

适配器可以与它提供适配接口的类相关联或从该类派生。如果使用继承，则适当的私有或受保护的基类可以隐藏底层实现。如果相反，适配器类与具有不理想接口的类相关联，适配器类中的方法（具有新接口）将仅将工作委托给其关联的类。

适配器模式还可以用来给一系列函数或其他类添加面向对象（OO）接口（即，在周围包裹一个 OO 接口），使得各种现有组件在面向对象系统中更自然地被利用。这种特定的适配器类型被称为 `extern C`，以便链接器解决两种语言之间的链接约定）。

利用适配器模式有好处。适配器通过提供一个共享接口，允许原本不相关的类进行通信，从而允许重用现有代码。现在面向对象的程序员将直接使用适配器类，这有助于应用程序的维护。也就是说，大多数程序员的交互将是一个设计良好的适配器类，而不是与两个或更多奇特的组件交互。使用适配器的一个小缺点是，由于增加了代码层，性能略有下降。然而，通常情况下，通过提供一个干净的接口来支持现有组件的交互，以重用现有组件，这是一个有利可图的方案，尽管可能会有（希望是小的）性能权衡。

适配器模式将包括以下内容：

+   一个 **Adaptee** 类，它代表了具有理想实用工具的类，但它的存在形式不适合或不理想。

+   一个 **Adapter** 类，它将 Adaptee 类的接口适配以满足所需接口的需求。

+   一个 **Target** 类，它代表了当前应用程序的具体、所需的接口。一个类可能既是 Target 也是 Adapter。

+   可选的 **Client** 类，它们将与 Target 类交互，以完全定义当前的应用程序。

适配器模式允许重用符合当前应用程序设计接口需求的现有合格组件。

让我们继续前进，看看适配器模式的两种常见应用；其中一种将有两种潜在的实现方式。

# 实现适配器模式

让我们探索适配器模式（Adapter pattern）的两种常见用法。那就是创建一个适配器来弥合两个不兼容的类接口之间的差距，或者创建一个适配器来简单地用面向对象（OO）接口包装现有的函数集。

我们将从使用一个提供两个（或更多）不兼容类之间连接器的适配器（*Adapter*）的用法开始。*Adaptee* 将是一个经过良好测试的类，我们希望重用它（但它的接口可能不理想），而*Target* 类将是我们在正在制作的应用程序中的 OO 设计所指定的。现在让我们指定一个适配器，以便我们的 Adaptee 能够与我们的 Target 类一起工作。

## 使用适配器为现有类提供必要的接口

要实现适配器模式，我们首先需要确定我们的 Adaptee 类。然后我们将创建一个适配器类来修改 Adaptee 的接口。我们还将确定我们的 Target 类，代表我们需要根据 OO 设计进行建模的类。有时，适配器和目标可能会合并成一个类。在实际应用中，我们还将有 Client 类，代表最终应用中发现的完整类集。让我们从 Adaptee 和 Adapter 类开始，因为这些类定义将是我们构建模式的基础。

在我们的例子中，我们将指定我们的 Adaptee 类为我们习惯看到的类——`Person`。我们将设想我们的星球最近意识到了许多其他能够支持生命的系外行星，并且我们已经善意地与每个这样的文明结盟。进一步设想地球上的各种软件系统都希望欢迎并包括我们的新朋友，包括`Romulans`和`Orkans`，我们希望调整一些现有的软件以轻松适应我们系外行星邻居的新人口结构。考虑到这一点，我们将通过创建一个适配器类`Humanoid`来将我们的`Person`类转换为包含更多星际术语。 

在我们即将到来的实现中，我们将使用私有继承从`Person`（Adaptee）继承`Humanoid`（Adapter），因此隐藏了 Adaptee 的底层实现。我们还可以将一个`Humanoid`与一个`Person`关联（我们将在本节中回顾这种实现）。然后我们可以在我们的层次结构中充实一些`Humanoid`的派生类，例如`Orkan`、`Romulan`和`Earthling`，以适应手头的星际应用。`Orkan`、`Romulan`和`Earthling`类可以被视为我们的 Target 类，或者是我们应用将要实例化的类。我们将选择使我们的适配器类`Humanoid`抽象，这样它就不能直接实例化。因为我们的特定派生类（目标类）可以通过我们的应用（Client）中的抽象基类类型（`Humanoid`）进行泛化，我们也可以将`Humanoid`视为一个目标类。也就是说，`Humanoid`主要被视为一个适配器，但次要地被视为一个泛化的目标类。

我们的各个 Client 类可以利用`Humanoid`的派生类，创建其具体后代的实例。这些实例可以存储在自己的专用类型中，或者使用`Humanoid`指针进行泛化。我们的实现是对广泛使用的适配器设计模式的一种现代诠释。

### 指定 Adaptee 和 Adapter（私有继承技术）

让我们看看我们适配器模式第一次使用时的机制，首先回顾一下我们的 Adaptee 类`Person`的定义。这个例子可以作为完整的程序在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex1.cpp)

```cpp
// Person is the Adaptee class (class requiring adaptation)
class Person
{
private:
    string firstName, lastName, title, greeting;
    char middleInitial  = '\0';  // in-class initialization
protected:
    void ModifyTitle(const string &);  
public:
    Person() = default;   // default constructor
    Person(const string &, const string &, char, 
           const string &);
    // default copy constructor prototype is not necessary
    // Person(const Person &) = default;  // copy ctor
    // Default op= suffices, so we'll comment out proto.
    // (see online code for review of implementation)
    // Person &operator=(const Person &); // assignment op.
    virtual ~Person()= default;  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; }  
    const string &GetLastName() const 
        { return lastName; }    
    const string &GetTitle() const { return title; }
    char GetMiddleInitial() const { return middleInitial; }
    void SetGreeting(const string &);
    virtual const string &Speak() { return greeting; }
    virtual void Print() const;
};
// Assume constructors, destructor, and non-inline methods
// are implemented as expected (see online code)
```

在前面的类定义中，我们注意到我们的 `Person` 类定义与我们在这本书的许多其他示例中看到的一样。这个类是可以实例化的；然而，在我们的星际应用程序中，`Person` 并不是一个合适的类来实例化。相反，预期的接口应该是使用 `Humanoid` 中的接口。

在这个前提下，让我们来看看我们的适配器类 `Humanoid`：

```cpp
class Humanoid: private Person   // Humanoid is abstract
{                           
protected:
    void SetTitle(const string &t) { ModifyTitle(t); }
public:
    Humanoid() = default;   
    Humanoid(const string &, const string &, 
             const string &, const string &);
    // default copy constructor prototype not required
    // Humanoid(const Humanoid &h) = default; 
    // default op= suffices, so commented out below, but
    // let's review how we'd write op= if needed
    // note explicit Humanoid downcast after calling base  
    // class Person::op= to match return type needed here
    // Humanoid &operator=(const Humanoid &h) 
    //     { return dynamic_cast<Humanoid &> 
    //              (Person::operator=(h)); }
    // dtor proto. not required since base dtor is virt.
    // ~Humanoid() override = default; // virt destructor
    // Added interfaces for the Adapter class 
    const string &GetSecondaryName() const 
        { return GetFirstName(); }  
    const string &GetPrimaryName() const 
        { return GetLastName(); } 
    // scope resolution needed in method to avoid recursion 
    const string &GetTitle() const 
        { return Person::GetTitle();}
    void SetSalutation(const string &m) { SetGreeting(m); }
    virtual void GetInfo() const { Print(); }
    virtual const string &Converse() = 0; // abstract class
};
Humanoid::Humanoid(const string &n2, const string &n1, 
    const string &planetNation, const string &greeting):
    Person(n2, n1, ' ', planetNation)
{
    SetGreeting(greeting);
}
const string &Humanoid::Converse()  // default definition 
{                    // for pure virtual function - unusual
    return Speak();
}
```

在上述 `Humanoid` 类中，我们的目标是提供一个适配器，以贡献我们星际应用程序所需的预期接口。我们简单地使用私有继承从 `Person` 派生 `Humanoid`，隐藏了 `Person` 中发现的公共接口，使其在 `Humanoid` 的作用域之外不被使用。我们理解目标应用程序（客户端）不希望 `Person` 中的公共接口被 `Humanoid` 实例的各种子类型所利用。请注意，我们不是添加功能，只是在适配接口。

我们接着注意到 `Humanoid` 中引入的公共方法，为目标类（们）提供了所需的接口。这些接口的实现通常是直接的。我们只需调用在 `Person` 中定义的继承方法，就可以轻松完成当前任务（但这样做使用的是不可接受的接口）。例如，我们的 `Humanoid::GetPrimaryName()` 方法简单地调用 `Person::GetLastName();` 来完成任务。然而，`GetPrimaryName()` 可能更多地代表了适当的星际语言，而不是 `Person::GetLastName()`。我们可以看到 `Humanoid` 如何作为 `Person` 的适配器。我们还可以看到适配器类 `Humanoid` 的大多数成员函数如何使用内联函数简单地封装 `Person` 方法，以提供更合适的接口，同时不增加任何开销。

注意，在 Humanoid 方法中对 `Person` 基类方法的调用前不需要使用 `Person::`（除非 `Humanoid` 方法调用 `Person` 中相同名称的方法，例如 `GetTitle()`）。使用 `Person::` 的作用域解析避免了这些情况中的潜在递归。

我们还注意到，`Humanoid` 引入了一个抽象的多态方法（即纯虚函数），其指定为 `virtual const string &Converse() = 0;`。我们已经做出了设计决定，只有 `Humanoid` 的派生类才能被实例化。尽管如此，我们理解公共派生类仍然可以被其基类类型 `Humanoid` 收集。在这里，`Humanoid` 主要作为适配器类，次要作为提供一系列可接受接口的目标类。

注意，我们的纯虚函数`virtual const String &Converse() = 0;`包含了一个默认实现。这种情况很少见，但允许这样做，只要实现不是内联编写的。在这里，我们利用这个机会为`Humanoid::Converse()`指定一个默认行为，只需简单地调用`Person::Speak()`。

### 从适配器派生具体类

接下来，让我们扩展我们的适配器（`Humanoid`）并查看我们的一个具体派生目标类`Orkan`：

```cpp
class Orkan: public Humanoid
{
public:
    Orkan() = default;   // default constructor
    Orkan(const string &n2, const string &n1, 
        const string &t): Humanoid(n2, n1, t, "Nanu nanu")
        { }
    // default copy constructor prototype not required
    // Orkan(const Orkan &h) = default;  
    // default op= suffices, so commented out below, but
    // let's review how we'd write it if needed
    // note explicit Orkan downcast after calling base  
    // class Humanoid::op= to match return type needed here
    // Orkan &operator=(const Orkan &h) 
    //    { return dynamic_cast<Orkan &>
    //             (Humanoid::operator=(h)); }
    // dtor proto. not required since base dtor is virt.
    // ~Orkan() override = default; // virtual destructor
    const string &Converse() override;  
};
// Must override Converse to make Orkan a concrete class
const string &Orkan::Converse()  
{                                
    return Humanoid::Converse(); // use scope resolution to
}                                // avoid recursion
```

在我们前面提到的`Orkan`类中，我们使用公有继承从`Humanoid`派生`Orkan`。一个`Orkan` *是* 一个`Humanoid`。因此，`Humanoid`中的所有公共接口都对`Orkan`实例可用。注意，我们的备用构造函数将默认问候信息设置为`"Nanu nanu"`，按照`Orkan`方言。

由于我们希望`Orkan`成为一个具体可实例化的类，我们必须在`Orkan`类中重写`Humanoid::Converse()`并提供实现。然而，请注意，`Orkan::Converse()`只是调用`Humanoid::Converse();`。也许`Orkan`认为其基类中的默认实现是可以接受的。注意，我们使用`Humanoid::`作用域解析符在`Orkan::Converse()`方法中限定`Converse()`，以避免递归。

有趣的是，如果`Humanoid`不是一个抽象类，`Orkan`就不需要重写`Converse()`方法——默认行为会自动继承。然而，由于`Humanoid`被定义为抽象类，`Orkan`中必须重写`Converse()`方法，否则`Orkan`也会被视为一个抽象类。不用担心！我们只需在`Orkan::Converse()`中调用`Humanoid::Converse()`，就可以利用`Humanoid::Converse()`的默认行为的好处。这将满足使`Orkan`具体化的要求，同时允许`Humanoid`保持抽象状态，同时仍然为`Converse()`提供罕见的默认行为！

现在，让我们看一下我们的下一个具体目标类`Romulan`：

```cpp
class Romulan: public Humanoid
{
public:
    Romulan() = default;   // default constructor
    Romulan(const string &n2, const string &n1, 
        const string &t): Humanoid(n2, n1, t, "jolan'tru")
        { }
    // default copy constructor prototype not required
    // Romulan(const Romulan &h) = default;
    // default op= suffices, so commented out below, but
    // let's review how we'd write it if so needed
    // note explicit Romulan downcast after calling base  
    // class Humanoid::op= to match return type needed here
    // Romulan &operator=(const Romulan &h) 
    //    { return dynamic_cast<Romulan &>
    //             (Humanoid::operator=(h)); }
    // dtor proto. not required since base dtor is virt.
    // ~Romulan() override = default;  // virt destructor
    const string &Converse() override;  
};
// Must override Converse to make Romulan a concrete class
const string &Romulan::Converse()  
{                               
    return Humanoid::Converse(); // use scope resolution to
}                                // avoid recursion        
```

快速看一下前面提到的`Romulan`类，我们会注意到这个具体的目标类与它的兄弟类`Orkan`相似。我们会注意到传递给基类构造函数的默认问候信息是`"jolan'tru"`，以反映`Romulan`方言。尽管我们可以使我们的`Romulan::Converse()`实现更加复杂，但我们选择不这样做。我们可以快速理解这个类的全部范围。

接下来，让我们看一下我们的第三个目标类`Earthling`：

```cpp
class Earthling: public Humanoid
{
public:
    Earthling() = default;   // default constructor
    Earthling(const string &n2, const string &n1, 
        const string &t): Humanoid(n2, n1, t, "Hello") { }
    // default copy constructor prototype not required
    // Earthling(const Romulan &h) = default;  
    // default op= suffices, so commented out below, but
    // let's review how we'd write it if so needed
    // note explicit Earthling downcast after calling base
    // class Humanoid::op= to match return type needed here
    // Earthling &operator=(const Earthling &h) 
    //    { return dynamic_cast<Earthling &>
    //             (Humanoid::operator=(h)); }
    // dtor proto. not required since base dtor is virt.
    // ~Earthling() override = default; // virt destructor
    const string &Converse() override;  
};
// Must override to make Earthling a concrete class
const string &Earthling::Converse() 
{                                                          
    return Humanoid::Converse(); // use scope resolution to
}                                // avoid recursion
```

再次，快速看一下前面提到的`Earthling`类，我们会注意到这个具体的目标类与它的兄弟类`Orkan`和`Romulan`相似。

现在我们已经定义了我们的适配器、适配器和多个目标类，让我们通过检查程序的客户端部分来将这些组件组合在一起。

### 将模式组件组合在一起

最后，让我们考虑一下在我们的整体应用程序中一个示例客户端可能的样子。当然，它可能由许多具有各种类的文件组成。在其最简单的形式中，如以下所示，我们的客户端将包含一个`main()`函数来驱动应用程序。

现在我们来看看我们的`main()`函数，看看我们的模式是如何编排的：

```cpp
int main()
{
    list<Humanoid *> allies;
    Orkan *o1 = new Orkan("Mork", "McConnell", "Orkan");
    Romulan *r1 = new Romulan("Donatra", "Jarok", 
                              "Romulan");
    Earthling *e1 = new Earthling("Eve", "Xu",
                                  "Earthling");
    // Add each specific type of Humanoid to generic list
    allies.push_back(o1);
    allies.push_back(r1);
    allies.push_back(e1);

    // Process the list of allies (which are Humanoid *'s 
    // Actually, each is a specialization of Humanoid!)
    for (auto *entity : allies)
    {
        entity->GetInfo();
        cout << entity->Converse() << endl;
    }
    // Though each type of Humanoid has a default
    // Salutation, each may expand their skills with 
    // an alternate language
    e1->SetSalutation("Bonjour");
    e1->GetInfo();
    cout << e1->Converse() << endl; // Show the Earthling's 
                           // revised language capabilities
    delete o1;   // delete the heap instances
    delete r1;
    delete e1;
    return 0;
}
```

回顾我们之前提到的`main()`函数，我们首先使用`list<Humanoid *>`创建一个 STL 列表，名为`allies;`。然后我们实例化一个`Orkan`、`Romulan`和一个`Earthling`，并使用`allies.push_back()`将它们每个都添加到列表中。再次使用标准模板库，我们接下来创建一个列表迭代器来遍历指向`Humanoid`实例的指针列表。当我们遍历我们的通用盟友列表时，我们在列表中的每个项目上调用`GetInfo()`和`Converse()`的批准接口（即，对于每种特定的`Humanoid`类型）。

接下来，我们指定一个特定的`Humanoid`，一个`Earthling`，并通过调用`e1->SetSalutation("Bonjour");`来更改这个实例的默认问候语。通过再次在这个实例上调用`Converse()`（我们首先在上面的循环中泛型地这样做），我们可以要求`Earthling`使用`"Bonjour"`来问候盟友，而不是使用默认的问候语`"Hello"`（`Earthling`的默认问候语）。

让我们看看这个程序的输出：

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

在上述输出中，请注意，每个`Humanoid`的行星规范都显示了出来（`Orkan`、`Romulan`和`Earthling`），然后是它们的次要和主要名称。然后，显示特定`Humanoid`的适当问候语。请注意，`Earthling` `Eve Xu`首先使用`"Hello"`进行对话，然后后来使用`"Bonjour"`进行对话。

之前实现（使用私有基类从 Adaptee 派生 Adapter）的一个优点是代码非常直接。使用这种方法，Adaptee 类中的任何受保护的方法都可以轻松地传递到 Adapter 方法的作用域内。我们很快就会看到，如果我们将关联用作将 Adapter 连接到 Adaptee 的手段，受保护成员将是一个问题。

之前提到的方法的一个缺点是它是一个 C++特定的实现。其他语言不支持私有基类。另外，使用公有基类来定义 Adapter 和 Adaptee 之间的关系将无法隐藏不想要的 Adaptee 接口，这将是一个非常糟糕的设计选择。

### 考虑 Adaptee 和 Adapter（关联）的另一种规范

让我们现在简要考虑一下之前提到的 Adapter 模式实现的略微修改版本。我们将使用关联来模拟 Adaptee 和 Adapter 之间的关系。具体的派生类（目标）仍然会像之前一样从 Adapter 派生。

这里是我们的 Adapter 类 `Humanoid` 的另一种实现，使用 Adapter 和 Adaptee 之间的关联。尽管我们只会审查与我们的初始方法不同的代码部分，但完整的实现可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex2.cpp)

```cpp
// Assume that Person exists mostly as before – however,
// Person::ModifyTitle() must be moved from protected to
// public or be unused if modifying Person is not possible.
// Let's assume we moved Person::ModifyTitle() to public.
class Humanoid    // Humanoid is abstract
{
private:
    Person *life = nullptr;  // delegate all requests to
                             // the associated object
protected:
    void SetTitle(const string &t) 
        { life->ModifyTitle(t); }
public:
    Humanoid() = default;
    Humanoid(const string &, const string &, 
             const string &, const string &);
    Humanoid(const Humanoid &h);// we have work for copies!
    Humanoid &operator=(const Humanoid &); // and for op=
    virtual ~Humanoid()  // virtual destructor
        { delete life; life = nullptr; }  
    // Added interfaces for the Adapter class
    const string &GetSecondaryName() const 
        { return life->GetFirstName(); }
    const string &GetPrimaryName() const 
        { return life->GetLastName(); }    
    const string &GetTitle() const 
        { return life->GetTitle();}
    void SetSalutation(const string &m) 
        { life->SetGreeting(m); }
    virtual void GetInfo() const { life->Print(); }
    virtual const string &Converse() = 0; // abstract class
};
Humanoid::Humanoid(const string &n2, const string &n1, 
          const string &planetNation, const string &greeting)
{
    life = new Person(n2, n1, ' ', planetNation);
    life->SetGreeting(greeting);
}
// copy constructor (we need to write it ourselves)
Humanoid::Humanoid(const Humanoid &h)  
{  // Remember life data member is of type Person
    delete life;  // delete former associated object
    life = new Person(h.GetSecondaryName(),
                     h.GetPrimaryName(),' ', h.GetTitle());
    life->SetGreeting(h.life->Speak());  
}
// overloaded operator= (we need to write it ourselves)
Humanoid &Humanoid::operator=(const Humanoid &h)
{
    if (this != &h)
        life->Person::operator=(dynamic_cast
                                <const Person &>(h));
    return *this;
}
const string &Humanoid::Converse() //default definition for
{                              // pure virtual fn - unusual
    return life->Speak();
}
```

在上述 Adapter 类的实现中，`Humanoid` 已不再从 `Person` 派生。相反，`Humanoid` 将添加一个私有数据成员 `Person *life;`，它将代表 Adapter (`Humanoid`) 和 Adaptee (`Person`) 之间的关联。在我们的 Humanoid 构造函数中，我们需要分配 Adaptee (`Person`) 的底层实现。我们还需要在我们的析构函数中删除 Adaptee (`Person`)。

与我们上一次的实现类似，`Humanoid` 在其公共接口中提供了相同的成员函数。然而，请注意，每个 `Humanoid` 方法都通过关联对象将调用委托给适当的 Adaptee 方法。例如，`Humanoid::GetSecondaryName()` 仅调用 `life->GetFirstName();` 来委托请求（而不是调用继承的相应 Adaptee 方法）。

与我们最初的实现一样，我们的从 `Humanoid` 派生的类（`Orkan`、`Romulan` 和 `Earthling`）以相同的方式指定，同样，我们的 `main()` 函数中的客户端也是这样。

### 选择 Adaptee 和 Adapter 之间的关系

在选择私有继承或关联作为 Adapter 和 Adaptee 之间的关系时，一个值得考虑的有趣点是 Adaptee 是否包含任何受保护的成员。回想一下，`Person` 的原始代码包括一个受保护的 `ModifyTitle()` 方法。Adaptee 类中应该存在受保护的成员吗？私有基类实现允许那些继承的受保护成员在 Adapter 类的作用域内继续被访问（即通过 Adapter 的方法）。然而，使用基于关联的实现，Adaptee (`Person`) 中的受保护方法在 Adapter 的作用域内不可用。为了使这个例子工作，我们需要将 `Person::ModifyTitle()` 移到公共访问区域。然而，修改 Adaptee 类并不总是可能的，也不一定是推荐的。考虑到受保护成员的问题，我们最初使用私有基类的实现是更强的实现，因为它不依赖于我们修改 Adaptee (`Person`) 的类定义。

现在，让我们简要地看一下适配器模式的一种替代用法。我们只是将适配器类用作包装类。我们将向一个原本松散排列但工作良好的函数集添加面向对象的接口，但这些函数缺少我们应用程序（客户端）所需的目标接口。

## 使用适配器作为包装器

作为适配器模式的一种替代用法，我们将围绕一组相关的外部函数包装一个面向对象（OO）接口。也就是说，我们将创建一个包装类来封装这些函数。

在我们的例子中，外部函数将代表一组现有的数据库访问函数。我们假设核心数据库功能已经针对我们的数据类型（`Person`）进行了良好的测试，并且没有问题地使用过。然而，这些外部函数本身提供了一个不理想且意外的功能接口。

我们将通过创建一个适配器类来封装这些外部函数的功能。我们的适配器类将是`CitizenDataBase`，代表一种封装的从数据库读取和写入`Person`实例的方法。我们现有的外部函数将为我们的`CitizenDataBase`成员函数提供实现。让我们假设在适配器类中定义的面向对象接口符合我们的面向对象设计要求。

让我们来看看我们简单包装适配器模式的机制，首先从检查提供数据库访问功能的外部函数开始。这个例子作为完整的程序可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter18/Chp18-Ex3.cpp)

```cpp
// Assume Person class exists with its usual implementation
Person objectRead; // holds the object from current read
                   // to support a simulation of a DB read
void db_open(const string &dbName)
{   // Assume implementation exists
    cout << "Opening database: " << dbName << endl;
}
void db_close(const string &dbName)
{   // Assume implementation exists
    cout << "Closing database: " << dbName << endl;
}
Person &db_read(const string &dbName, const string &key)
{   // Assume implementation exists
    cout << "Reading from: " << dbName << " using key: ";
    cout << key << endl;
    // In a true implementation, we would read the data
    // using the key and return the object we read in
    return objectRead; // non-stack instance for simulation
}
const string &db_write(const string &dbName, Person &data)
{   // Assume implementation exists
    const string &key = data.GetLastName();
    cout << "Writing: " << key << " to: " << 
             dbName << endl;
    return key;
}
```

在我们之前定义的外部函数中，让我们假设所有函数都经过了良好的测试，并允许从数据库中读取或写入`Person`实例。为了支持这种模拟，我们创建了一个外部`Person`实例`Person objectRead;`，为刚读取的实例提供一个简短的、非堆栈位置的存储空间（由`db_read()`使用），直到新读取的实例被捕获为返回值。请注意，现有的外部函数并不代表一个封装的解决方案。

现在，让我们创建一个简单的包装类来封装这些外部函数。这个包装类，`CitizensDataBase`，将代表我们的适配器类：

```cpp
// CitizenDataBase is the Adapter class 
class CitizenDataBase  // Adapter wraps undesired interface
{
private:
    string name;
public:
    // No default constructor (unusual)
    CitizenDataBase(const string &);
    CitizenDataBase(const CitizenDataBase &) = delete;
    CitizenDataBase &operator=(const CitizenDataBase &) 
                               = delete;  // disallow =
    virtual ~CitizenDataBase();  // virtual destructor
    inline Person &Read(const string &);
    inline const string &Write(Person &);
};
CitizenDataBase::CitizenDataBase(const string &n): name(n)
{
    db_open(name);   // call existing external function
}
CitizenDataBase::~CitizenDataBase()
{
    db_close(name);  // close database with external
}                    // function
Person &CitizenDataBase::Read(const string &key)
{
    return db_read(name, key);   // call external function
}
const string &CitizenDataBase::Write(Person &data)
{
    return db_write(name, data);  // call external function
}
```

在我们之前为适配器类定义的类中，我们只是简单地将外部数据库功能封装在 `CitizenDataBase` 类中。在这里，`CitizenDataBase` 不仅是我们的适配器类，也是我们的目标类，因为它包含了我们的应用程序（客户端）所期望的接口。请注意，`CitizenDataBase` 的 `Read()` 和 `Write()` 方法都已经在类定义中内联了；它们的方法只是调用外部函数。这是一个示例，说明了具有内联函数的包装类可以是一个低成本适配器类，仅添加非常小的开销（构造函数、析构函数以及可能的其他非内联方法）。

现在，让我们来看看我们的 `main()` 函数，它是客户端的一个精简版本：

```cpp
int main()
{
    string key;
    string name("PersonData"); // name of database
    Person p1("Curt", "Jeffreys", 'M', "Mr.");
    Person p2("Frank", "Burns", 'W', "Mr.");
    Person p3;
    CitizenDataBase People(name);   // open Database
    key = People.Write(p1); // write a Person object
    p3 = People.Read(key);  // using a key, retrieve Person
    return 0;
}                        // destruction will close database
```

在上述 `main()` 函数中，我们首先创建了三个 `Person` 实例。然后，我们创建了一个 `CitizenDataBase` 实例，以提供封装的访问权限来写入或读取我们的 `Person` 实例，到或从数据库中。我们的 `CitizenDataBase` 构造函数的方法调用外部函数 `db_open()` 来打开数据库。同样，析构函数调用 `db_close()`。正如预期的那样，我们的 `CitizenDataBase` 的 `Read()` 和 `Write()` 方法将分别调用外部函数 `db_read()` 或 `db_write()`。

让我们看看这个程序的输出：

```cpp
Opening database: PersonData
Writing: Jeffreys to: PersonData
Reading from: PersonData using key: Jeffreys
Closing database: PersonData
```

在上述输出中，我们可以注意到各种成员函数与包装的外部函数之间的关联，通过构造函数、写入和读取的调用，以及数据库的销毁。

我们的简单 `CitizenDataBase` 包装器是适配器模式的一个非常直接但合理的应用。有趣的是，我们的 `CitizenDataBase` 也与 **数据访问对象模式** 有相似之处，因为这个包装器提供了一个干净的接口来访问数据存储机制，隐藏了底层数据库的实现（访问）。

我们现在已经看到了适配器模式的三个实现。我们将适配器、适配者、目标和客户端的概念融合到了我们习惯看到的类框架中，即 `Person` 类，以及我们的适配器的后代（`Orkan`、`Romulan` 和 `Earthling`，如我们的前两个例子所示）。现在，让我们简要回顾一下我们在学习模式之前所学的知识，然后继续到下一章。

# 摘要

在本章中，我们通过扩展我们对设计模式的知识来提高我们成为更好的 C++程序员的追求。我们探讨了适配器模式的概念及其多种实现。我们的第一个实现使用私有继承从适配器类派生出适配器。我们指定适配器为一个抽象类，然后使用公共继承根据适配器类提供的接口引入几个目标类。我们的第二个实现则使用关联来模拟适配器和适配器之间的关系。然后我们查看了一个适配器作为包装器的示例用法，简单地为现有的基于函数的应用程序组件添加 OO 接口。

利用常见的设计模式，例如适配器模式，将帮助您更轻松地重用现有经过良好测试的代码部分，并且其他程序员也能理解。通过利用核心设计模式，您将为理解良好且可重用的解决方案做出贡献，并使用更复杂的编程技术。

我们现在准备继续前进，学习下一个设计模式，在*第十九章*中，*使用单例模式*。将更多模式添加到我们的编程技能库中，使我们成为更灵活且更有价值的程序员。让我们继续前进！

# 问题

1.  使用本章中找到的适配器示例，创建一个如下所示的程序：

    1.  实现一个`CitizenDataBase`，该数据库存储各种类型的`Humanoid`实例（`Orkan`、`Romulan`、`Earthling`以及可能还有`Martian`）。决定您将使用私有基类适配器-适配器关系还是适配器和适配器之间的关联关系（提示：私有继承版本将更容易）。

    1.  注意到`CitizenDataBase`处理`Person`实例，这个类是否可以原样使用来存储各种类型的`Humanoid`实例，或者必须以某种方式对其进行适配？回想一下，`Person`是`Humanoid`的基类（如果您选择了这种实现），但也要记住，我们永远不能超出非公共继承边界进行向上转型。

1.  你能想象出哪些其他例子可以轻松地结合适配器模式？
