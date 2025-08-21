# 第十九章：使用单例模式

本章将继续扩展您的 C++编程技能，超越核心面向对象编程概念，旨在让您能够利用核心设计模式解决重复出现的编码难题。在编码解决方案中使用设计模式不仅可以提供精炼的解决方案，还有助于更轻松地维护代码，并为代码重用提供潜在机会。

我们将学习如何在 C++中有效实现**单例模式**，这是下一个核心设计模式。

在本章中，我们将涵盖以下主要主题：

+   理解单例模式及其对面向对象编程的贡献

+   在 C++中实现单例模式（使用简单的对类方法和配对类方法的方法）；使用注册表允许多个类利用单例模式

通过本章结束时，您将了解单例模式以及如何使用它来确保给定类型只能存在一个实例。将另一个核心设计模式添加到您的知识体系中，将进一步增强您的编程技能，帮助您成为更有价值的程序员。

通过研究另一个常见的设计模式，单例模式，来增强我们的编程技能。

# 技术要求

本章中完整程序示例的代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19)。每个完整程序示例都可以在 GitHub 存储库中的适当章节标题（子目录）下找到，文件名与当前章节号对应，后跟当前章节中的示例编号。例如，本章中的第一个完整程序可以在名为`Chp19-Ex1.cpp`的文件中的`Chapter19`子目录中找到上述 GitHub 存储库中。

本章的 CiA 视频可在以下链接观看：[`bit.ly/3f2dKZb`](https://bit.ly/3f2dKZb)。

# 理解单例模式

单例模式是一种创建型设计模式，它保证了一个类只会存在一个实例；该类型的两个或更多实例根本不可能同时存在。采用这种模式的类被称为**单例**。

单例模式可以使用静态数据成员和静态方法来实现。这意味着单例将在全局范围内访问当前实例。这一影响起初似乎很危险；将全局状态信息引入代码是对单例模式的一种批评，有时会被认为是一种反模式。然而，通过对定义单例的静态数据成员使用访问区域的适当使用，我们可以坚持只使用当前类的适当静态方法访问单例（除了初始化），从而减轻这种潜在的模式问题。

该模式的另一个批评是它不是线程安全的。可能存在竞争条件，以进入创建单例实例的代码段。如果不保证对该关键代码区域的互斥性，单例模式将会破坏，允许多个这样的实例存在。因此，如果将使用多线程编程，必须使用适当的锁定机制来保护创建单例的关键代码区域。使用静态内存实现的单例存储在同一进程中的线程之间的共享内存中；有时会因为垄断资源而批评单例。

Singleton 模式可以利用多种实现技术。每种实现方式都必然会有利弊。我们将使用一对相关的类`Singleton`和`SingletonDestroyer`来强大地实现该模式。虽然还有更简单、直接的实现方式（我们将简要回顾其中一种），但最简单的技术留下了 Singleton 可能不会被充分销毁的可能性。请记住，析构函数可能包括重要和必要的活动。

Singleton 通常具有长寿命；因此，在应用程序终止之前销毁 Singleton 是合适的。许多客户端可能有指向 Singleton 的指针，因此没有一个客户端应该删除 Singleton。我们将看到`Singleton`将是*自行创建*的，因此它应该理想地*自行销毁*（即通过其`SingletonDestroyer`的帮助）。因此，配对类方法虽然不那么简单，但将确保正确的`Singleton`销毁。请注意，我们的实现也将允许直接删除 Singleton；这是罕见的，但我们的代码也将处理这种情况。

带有配对类实现的 Singleton 模式将包括以下内容：

+   一个代表实现 Singleton 概念所需的核心机制的**Singleton**类。

+   一个**SingletonDestroyer**类，它将作为 Singleton 的辅助类，确保给定的 Singleton 被正确销毁。

+   从 Singleton 派生的类，代表我们希望确保在特定时间只能创建一个其类型实例的类。这将是我们的**目标**类。

+   可选地，目标类可以既从 Singleton 派生，又从另一个类派生，这个类可能代表我们想要专门化或简单包含的现有功能（即*混入*）。在这种情况下，我们将从一个特定于应用程序的类和 Singleton 类中继承。

+   可选的**客户端**类，它们将与目标类交互，以完全定义手头的应用程序。

+   或者，Singleton 也可以在目标类内部实现，将类的功能捆绑在一个单一类中。

+   真正的 Singleton 模式可以扩展到允许创建多个（离散的）实例，但不是一个确定数量的实例。这是罕见的。

我们将专注于传统的 Singleton 模式，以确保在任何给定时间只存在一个类的实例。

让我们继续前进，首先检查一个简单的实现，然后是我们首选的配对类实现，Singleton 模式。

# 实现 Singleton 模式

Singleton 模式将用于确保给定类只能实例化该类的单个实例。然而，真正的 Singleton 模式还将具有扩展功能，以允许多个（但数量明确定义的）实例被创建。这种 Singleton 模式的罕见且不太为人所知的特殊情况。

我们将从一个简单的 Singleton 实现开始，以了解其局限性。然后我们将进一步实现 Singleton 的更强大的配对类实现，最常见的模式目标是只允许在任何给定时间内实例化一个目标类。

## 使用简单实现

为了实现一个非常简单的 Singleton，我们将使用一个简单的单类规范来定义 Singleton 本身。我们将定义一个名为`Singleton`的类来封装该模式。我们将确保我们的构造函数是私有的，这样它们就不能被应用超过一次。我们还将添加一个静态的`instance()`方法来提供`Singleton`对象的实例化接口。这个方法将确保私有构造只发生一次。

让我们先来看一下这个简单的实现，可以在我们的 GitHub 存储库中找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19/Chp19-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19/Chp19-Ex1.cpp)

```cpp
class Singleton
{
private:
    static Singleton *theInstance;
    Singleton();  // private to prevent multiple instantiation
public:
    static Singleton *instance(); // interface for creation
    virtual ~Singleton();  // never called, unless you delete
};                         // Singleton explicitly, which is
                           // unlikely and atypical
Singleton *Singleton::theInstance = NULL; // external variable
                                         // to hold static mbr
Singleton::Singleton()
{
    cout << "Constructor" << endl;
    theInstance = NULL;
}
Singleton::~Singleton()  // the destructor is not called in
{                        // the typical pattern usage
    cout << "Destructor" << endl;
    if (theInstance != NULL)  
    {  
       Singleton *temp = theInstance;
       theInstance = NULL;       // removes ptr to Singleton
       temp->theInstance = NULL; // prevents recursion
       delete temp;              // delete the Singleton
    }                 
}
Singleton *Singleton::instance()
{
    if (theInstance == NULL)
        theInstance = new Singleton();  // allocate Singleton
    return theInstance;
}
int main()
{
    Singleton *s1 = Singleton::instance(); // create Singleton
    Singleton *s2 = Singleton::instance(); // returns existing
    cout << s1 << " " << s2 << endl; // addresses are the same
}                                         
```

在上述的类定义中，我们注意到包括数据成员`static Singleton *theInstance;`来表示`Singleton`实例本身。我们的构造函数是私有的，这样就不能多次使用它来创建多个`Singleton`实例。相反，我们添加了一个`static Singleton *instance()`方法来创建`Singleton`。在这个方法中，我们检查数据成员`theInstance`是否为`NULL`，如果是，我们就实例化唯一的`Singleton`实例。

在类定义之外，我们看到了外部变量（及其初始化）来支持静态数据成员的内存需求，定义为`Singleton *Singleton::theInstance = NULL;`。我们还看到在`main()`中，我们调用静态的`instance()`方法来使用`Singleton::instance()`创建一个 Singleton 实例。对这个方法的第一次调用将实例化一个`Singleton`，而对这个方法的后续调用将仅仅返回指向现有`Singleton`对象的指针。我们可以通过打印这些对象的地址来验证这些实例是相同的。

让我们来看一下这个简单程序的输出：

```cpp
Constructor
0xee1938 0xee1938
```

在上述输出中，我们注意到了一些意外的事情 - 析构函数没有被调用！如果析构函数有关键的任务要执行怎么办呢？

### 理解简单 Singleton 实现的一个关键缺陷

在简单实现中，我们的`Singleton`的析构函数没有被调用，仅仅是因为我们没有通过`s1`或`s2`标识符删除动态分配的`Singleton`实例。为什么呢？显然可能有多个指针（句柄）指向一个`Singleton`对象。决定哪个句柄应该负责删除`Singleton`是很难确定的 - 这些句柄至少需要合作或使用引用计数。

此外，`Singleton`往往存在于应用程序的整个生命周期。这种长期存在进一步表明，`Singleton`应该负责自己的销毁。但是如何做呢？我们很快将看到一个实现，它将允许`Singleton`通过一个辅助类来控制自己的销毁。然而，使用简单实现，我们可能只能举手投降，并建议操作系统在应用程序终止时回收内存资源 - 包括这个小`Singleton`的堆内存。这是正确的；然而，如果在析构函数中需要完成重要任务呢？我们在简单模式实现中遇到了限制。

如果我们需要调用析构函数，我们是否应该允许其中一个句柄使用，例如`delete s1;`来删除实例？我们之前已经讨论过是否允许任何一个句柄执行删除的问题，但现在让我们进一步检查析构函数本身可能存在的问题。例如，如果我们的析构函数假设只包括`delete theInstance;`，我们将会有一个递归函数调用。也就是说，调用`delete s1;`将调用`Singleton`的析构函数，然后在析构函数体内部调用`delete theInstance;`将把`theInstance`识别为`Singleton`类型，并再次调用`Singleton`的析构函数 - *递归*。

不用担心！如所示，我们的析构函数通过首先检查`theInstance`数据成员是否不是`NULL`，然后安排`temp`指向`theInstance`来管理递归，以保存我们需要删除的实例的句柄。然后我们进行`temp->theInstance = NULL;`的赋值，以防止在`delete temp;`时递归。为什么？因为`delete temp;`也会调用`Singleton`的析构函数。在这个析构函数调用时，`temp`将绑定到`this`，并且在第一次递归函数调用时不满足条件测试`if (theInstance != NULL)`，使我们退出持续的递归。请注意，我们即将使用成对类方法的实现不会有这个潜在问题。

重要的是要注意，在实际应用中，我们不会创建一个领域不明确的`Singleton`实例。相反，我们将应用程序分解到设计中以使用该模式。毕竟，我们希望有一个有意义的类类型的`Singleton`实例。要使用我们简单的`Singleton`类作为基础来做到这一点，我们只需将我们的目标（特定于应用程序）类从`Singleton`继承。目标类也将有私有构造函数 - 接受足以充分实例化目标类的参数。然后，我们将静态的`instance()`方法从`Singleton`移到目标类，并确保`instance()`的参数列表接受传递给私有目标构造函数的必要参数。

总之，我们简单的实现存在固有的设计缺陷，即`Singleton`本身没有保证的适当销毁。让操作系统在应用程序终止时收集内存不会调用析构函数。选择一个可以删除内存的`Singleton`句柄虽然可能，但需要协调，也破坏了模式的通常应用，即允许`Singleton`在应用程序的持续时间内存在。

现在，因为我们理解了简单的`Singleton`实现的局限性，我们将转而前进到首选的成对类实现 Singleton 模式。成对类方法将确保我们的`Singleton`在应用程序允许`Singleton`在应用程序终止之前被销毁（最常见的情况）或者在应用程序中罕见地提前销毁`Singleton`时，能够进行适当的销毁。

## 使用更健壮的成对类实现

为了以一种良好封装的方式实现成对类方法的 Singleton 模式，我们将定义一个 Singleton 类，纯粹添加创建单个实例的核心机制。我们将把这个类命名为`Singleton`。然后，我们将添加一个辅助类到`Singleton`，称为`SingletonDestroyer`，以确保我们的`Singleton`实例在应用程序终止之前始终进行适当的销毁。这一对类将通过聚合和关联进行关联。更具体地说，`Singleton`类将在概念上包含一个`SingletonDestroyer`（聚合），而`SingletonDestroyer`类将持有一个关联到（外部）`Singleton`的关联。因为`Singleton`和`SingletonDestroyer`的实现是通过静态数据成员，聚合是概念性的 - 静态成员被存储为外部变量。

一旦定义了这些核心类，我们将考虑如何将 Singleton 模式纳入我们熟悉的类层次结构中。假设我们想要实现一个类来封装“总统”的概念。无论是一个国家的总统还是大学的校长，都很重要的是在特定时间只有一个总统。 “总统”将是我们的目标类；因此，“总统”是一个很好的候选者来利用我们的 Singleton 模式。

有趣的是，尽管在特定时间只会有一位总统，但是可以替换总统。例如，美国总统的任期一次只有四年，可以连任一届。大学校长可能也有类似的条件。总统可能因辞职、弹劾或死亡而提前离任，或者在任期到期后简单地离任。一旦现任总统的存在被移除，那么实例化一个新的 Singleton `President`就是可以接受的。因此，我们的 Singleton 模式在特定时间只允许一个 Target 类的 Singleton。

反思我们如何最好地实现`President`类，我们意识到`President` *是* `Person`，并且还需要*混入* `Singleton`的功能。有了这个想法，我们现在有了我们的设计。`President`将使用多重继承来扩展`Person`的概念，并混入`Singleton`的功能。

当然，我们可以从头开始构建一个`President`类，但是当`President`类的`Person`组件在一个经过充分测试和可用的类中表示时，为什么要这样做呢？同样，当然，我们可以将`Singleton`类的信息嵌入到我们的`President`类中，而不是继承一个单独的`Singleton`类。绝对，这也是一个选择。然而，我们的应用程序将封装解决方案的每个部分。这将使未来的重用更容易。尽管如此，设计选择很多。

### 指定 Singleton 和 SingletonDestroyer 类

让我们来看看我们的 Singleton 模式的机制，首先检查`Singleton`和`SingletonDestroyer`类的定义。这些类合作实现 Singleton 模式。这个例子可以在我们的 GitHub 存储库中找到完整的程序。

https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19/Chp19-Ex2.cpp

```cpp
class Singleton;    // Necessary forward class declarations
class SingletonDestroyer;
class Person;
class President;
class SingletonDestroyer   
{
private:
    Singleton *theSingleton;
public:
    SingletonDestroyer(Singleton *s = 0) { theSingleton = s; }
    SingletonDestroyer(const SingletonDestroyer &) = delete; 
    SingletonDestroyer &operator=(const SingletonDestroyer &)                                  = delete; 
    ~SingletonDestroyer(); // destructor shown further below
    void setSingleton(Singleton *s) { theSingleton = s; }
    Singleton *getSingleton() { return theSingleton; }
};
```

在上述代码段中，我们从几个前向类声明开始，比如`class Singleton;`。这些声明允许在编译器看到它们的完整类定义之前就可以引用这些数据类型。

接下来，让我们来看看我们的`SingletonDestroyer`类定义。这个简单的类包含一个私有数据成员`Singleton *theSingleton;`，表示`SingletonDestroyer`将来将负责释放的`Singleton`的关联（我们将很快检查`SingletonDestroyer`的析构函数定义）。请注意，我们的析构函数不是虚拟的，因为这个类不打算被专门化。

请注意，我们的构造函数为`Singleton *`指定了默认值`0`（`NULL`）。`SingletonDestroyer`还包含两个成员函数`setSingleton()`和`getSingleton()`，仅提供了设置和获取相关`Singleton`成员的方法。

还要注意，`SingletonDestroyer`中的复制构造函数和重载赋值运算符在其原型中使用`=delete`进行了禁止。

在我们检查这个类的析构函数之前，让我们先看看`Singleton`的类定义。

```cpp
// Singleton will be mixed-in using inheritance with a Target
// class. If Singleton is used stand-alone, the data members
// would be private, and add a Static *Singleton instance();
// method to the public access region.
class Singleton
{
protected:
    static Singleton *theInstance;
    static SingletonDestroyer destroyer;
protected:
    Singleton() {}
    Singleton(const Singleton &) = delete; // disallow copies
    Singleton &operator=(const Singleton &) = delete; // and =
    friend class SingletonDestroyer;
    virtual ~Singleton() 
        { cout << "Singleton destructor" << endl; }
};
```

上述的`Singleton`类包含受保护的数据成员`static Singleton *theInstance;`，它将表示为采用 Singleton 习惯用法分配给类的唯一实例的指针。

受保护的数据成员`static SingletonDestroyer destroyer`代表一个概念上的聚合或包含成员。这种包含实际上只是概念性的，因为静态数据成员不存储在任何实例的内存布局中；它们实际上存储在外部内存中，并且*name-mangled*以显示为类的一部分。这个（概念上的）聚合子对象`destroyer`将负责正确销毁`Singleton`。请记住，`SingletonDestroyer`与唯一的`Singleton`有关，代表了`SingletonDestroyer`概念上包含的外部对象。这种关联是`SingletonDestroyer`将如何访问 Singleton 的方式。

当实现静态数据成员`static SingletonDestroyer destroyer;`的外部变量的内存在应用程序结束时消失时，将调用`SingletonDestroyer`（静态的概念性子对象）的析构函数。这个析构函数将运行`delete theSingleton;`，确保外部动态分配的`Singleton`对象将有适当的析构顺序运行。因为`Singleton`中的析构函数是受保护的，所以需要将`SingletonDestructor`指定为`Singleton`的友元类。

请注意，`Singleton`中复制构造函数和重载赋值运算符的使用都已经在它们的原型中使用`=delete`禁止了。

在我们的实现中，我们假设`Singleton`将通过继承混入到派生的目标类中。在派生类（打算使用 Singleton 习惯用法的类）中，我们提供了所需的静态`instance()`方法来创建`Singleton`实例。请注意，如果`Singleton`被用作独立类来创建单例，我们将在`Singleton`的公共访问区域中添加`static Singleton* instance()`。然后我们将数据成员从受保护的访问区域移动到私有访问区域。然而，拥有一个与应用程序无关的 Singleton 只能用来演示概念。相反，我们将把 Singleton 习惯用法应用到需要使用这种习惯用法的实际类型上。

有了我们的`Singleton`和`SingletonDestroyer`类定义，让我们接下来检查这些类的其余必要实现需求：

```cpp
// External (name mangled) variables to hold static data mbrs
Singleton *Singleton::theInstance = 0;
SingletonDestroyer Singleton::destroyer;
// SingletonDestroyer destructor definition must appear after 
// class definition for Singleton because it is deleting a 
// Singleton (so its destructor can be seen)
// This is not an issue when using header and source files.
SingletonDestroyer::~SingletonDestroyer()
{   
    if (theSingleton == NULL)
        cout << "SingletonDestroyer destructor: Singleton                  has already been destructed" << endl;
    else
    {
        cout << "SingletonDestroyer destructor" << endl;
        delete theSingleton;   
    }                          
}
```

在上述代码片段中，首先注意两个外部变量定义，提供内存以支持`Singleton`类中的两个静态数据成员——即`Singleton *Singleton::theInstance = 0;`和`SingletonDestroyer Singleton::destroyer;`。请记住，静态数据成员不存储在其指定类的任何实例中。相反，它们存储在外部变量中；这两个定义指定了内存。请注意，数据成员都标记为受保护。这意味着虽然我们可以直接定义它们的外部存储，但我们不能通过`Singleton`的静态成员函数以外的方式访问这些数据成员。这将给我们一些安心。虽然静态数据成员有潜在的全局访问点，但它们的受保护访问区域要求使用`Singleton`类的适当静态方法来正确操作这些重要成员。

接下来，注意`SingletonDestroyer`的析构函数。这个巧妙的析构函数首先检查它是否与它负责的`Singleton`的关联是否为`NULL`。这将很少发生，并且只会在非常不寻常的情况下发生，即客户端直接使用显式的`delete`释放`Singleton`对象。

`SingletonDestroyer`析构函数中的通常销毁场景将是执行`else`子句，其中`SingletonDestructor`作为静态对象将负责删除其配对的`Singleton`，从而销毁它。请记住，`Singleton`中将包含一个`SingletonDestroyer`对象。这个静态（概念上的）子对象的内存不会在应用程序结束之前消失。请记住，静态内存实际上并不是任何实例的一部分。因此，当`SingletonDestroyer`被销毁时，它通常的情况将是`delete theSingleton;`，这将释放其配对的 Singleton 的内存，使得`Singleton`能够被正确销毁。

单例模式背后的驱动设计决策是，单例是一个长期存在的对象，它的销毁通常应该在应用程序的最后发生。单例负责创建自己的内部目标对象，因此单例不应该被客户端删除（因此也不会被销毁）。相反，首选的机制是，当作为静态对象移除时，`SingletonDestroyer`会删除其配对的`Singleton`。

尽管如此，偶尔也会有合理的情况需要在应用程序中间删除一个`Singleton`。如果一个替代的`Singleton`从未被创建，我们的`SingletonDestroyer`析构函数仍将正确工作，识别到其配对的`Singleton`已经被释放。然而，更有可能的情况是我们的`Singleton`将在应用程序的某个地方被另一个`Singleton`实例替换。回想一下我们的应用程序示例，总统可能会被弹劾、辞职或去世，但会被另一位总统取代。在这些情况下，直接删除`Singleton`是可以接受的，然后创建一个新的`Singleton`。在这种情况下，`SingletonDestroyer`现在将引用替代的`Singleton`。

### 从 Singleton 派生目标类

接下来，让我们看看如何从`Singleton`创建我们的目标类`President`：

```cpp
// Assume our Person class definition is as we are accustomed
// A President Is-A Person and also mixes-in Singleton 
class President: public Person, public Singleton
{
private:
    President(const char *, const char *, char, const char *);
public:
    virtual ~President();
    President(const President &) = delete;  // disallow copies
    President &operator=(const President &) = delete; // and =
    static President *instance(const char *, const char *,
                               char, const char *);
};
President::President(const char *fn, const char *ln, char mi,
    const char *t) : Person(fn, ln, mi, t), Singleton()
{
}
President::~President()
{
    destroyer.setSingleton(NULL);  
    cout << "President destructor" << endl;
}
President *President::instance(const char *fn, const char *ln,
                               char mi, const char *t)
{
    if (theInstance == NULL)
    {
        theInstance = new President(fn, ln, mi, t);
        destroyer.setSingleton(theInstance);
        cout << "Creating the Singleton" << endl;
    }
    else
        cout << "Singleton previously created.                  Returning existing singleton" << endl;
    return (President *) theInstance; // cast necessary since
}                              // theInstance is a Singleton * 
```

在我们上述的目标类`President`中，我们仅仅使用公共继承从`Person`继承`President`，然后通过多重继承从`Singleton`继承`President`来*混入*`Singleton`机制。

我们将构造函数放在私有访问区域。静态方法`instance()`将在内部使用这个构造函数来创建唯一允许的`Singleton`实例，以符合模式。没有默认构造函数（不寻常），因为我们不希望允许创建没有相关细节的`President`实例。请记住，如果我们提供了替代的构造函数接口，C++将不会链接默认构造函数。由于我们不希望复制`President`或将`President`分配给另一个潜在的`President`，我们已经在这些方法的原型中使用`=delete`规范来禁止复制和分配。

我们的`President`析构函数很简单，但至关重要。在我们明确删除`Singleton`对象的情况下，我们通过设置`destroyer.setSingleton(NULL);`来做好准备。请记住，`President`继承了受保护的`static SingletonDestroyer destroyer;`数据成员。在这里，我们将销毁者的关联`Singleton`设置为`NULL`。然后，我们的`President`析构函数中的这行代码使得`SingletonDestroyer`的析构函数能够准确地依赖于检查其关联的`Singleton`是否已经在开始其`Singleton`对应部分的通常删除之前被删除。

最后，我们定义了一个静态方法，为我们的`President`提供`Singleton`的创建接口，使用`static President *instance(const char *, const char *, char, const char *);`。在`instance()`的定义中，我们首先检查继承的受保护数据成员`Singleton *theInstance`是否为`NULL`。如果我们还没有分配`Singleton`，我们使用上述的私有构造函数分配`President`并将这个新分配的`President`实例分配给`theInstance`。这是从`President *`向`Singleton *`的向上转型，在公共继承边界上没有问题。然而，如果在`instance()`方法中，我们发现`theInstance`不是`NULL`，我们只需返回指向先前分配的`Singleton`对象的指针。由于用户无疑会想要将此对象用作`President`来享受继承的`Person`功能，我们将`theInstance`向下转型为`President *`，作为此方法的返回值。

最后，让我们考虑一下我们整个应用程序中一个示例客户端的后勤。在其最简单的形式中，我们的客户端将包含一个`main()`函数来驱动应用程序并展示我们的 Singleton 模式。

### 将模式组件在客户端中组合在一起

现在让我们来看看我们的`main()`函数是如何组织我们的模式的：

```cpp
int main()
{ 
    // Create a Singleton President
    President *p1 = President::instance("John", "Adams", 
                                        'Q', "President");
    // This second request will fail, returning orig. instance
    President *p2 = President::instance("William", "Harrison",
                                        'H', "President");
    if (p1 == p2)   // Verification there's only one object
        cout << "Same instance (only one Singleton)" << endl;
    p1->Print();
    // SingletonDestroyer will release Singleton at end
    return 0;
}
```

回顾我们在前面的代码中的`main()`函数，我们首先使用`President *p1 = President::instance("John", "Adams", 'Q', "President");`分配一个 Singleton `President`。然后我们尝试在下一行代码中分配另一个`President`，使用`*p2`。因为我们只能有一个`Singleton`（`President` *混入*了一个`Singleton`），一个指针被返回到我们现有的`President`并存储在`p2`中。我们通过比较`p1 == p2`来验证只有一个`Singleton`；指针确实指向同一个实例。

接下来，我们利用我们的`President`实例以其预期的方式使用，比如使用从`Person`继承的一些成员函数。例如，我们调用`p1->Print();`。当然，我们的`President`类可以添加适合在我们的客户端中使用的专门功能。

现在，在`main()`的末尾，我们的静态对象`SingletonDestroyer Singleton::destroyer;`将在其内存被回收之前被适当地销毁。正如我们所看到的，`SingletonDestroyer`的析构函数（通常）会使用`delete theSingleton;`向其关联的`Singleton`（实际上是`President`）发出`delete`。这将触发我们的`President`析构函数、`Singleton`析构函数和`Person`析构函数分别被调用和执行（从最专门的到最一般的子对象）。由于我们的`Singleton`析构函数是虚拟的，我们保证从正确的级别开始销毁并包括所有析构函数。

让我们看看这个程序的输出：

```cpp
Creating the Singleton
Singleton previously created. Returning existing singleton
Same instance (only one Singleton)
President John Q Adams
SingletonDestroyer destructor
President destructor
Singleton destructor
Person destructor
```

在前面的输出中，我们可以看到 Singleton `President`的创建，以及第二个`instance()`请求一个`President`只是返回现有的`President`。然后我们看到打印出的`President`的细节。

最有趣的是，我们可以看到`Singleton`的销毁顺序，这是由`SingletonDestroyer`的静态对象回收驱动的。通过在`SingletonDestroyer`析构函数中正确删除`Singleton`，我们看到`President`、`Singleton`和`Person`的析构函数都被调用，因为它们共同构成了完整的`President`对象。

### 检查显式单例删除及其对 SingletonDestroyer 析构函数的影响

让我们看看客户端的另一个版本，其中有一个替代的`main()`函数。在这里，我们强制删除我们的`Singleton`；这是罕见的。在这种情况下，我们的`SingletonDestroyer`不会删除其配对的`Singleton`。这个例子可以在我们的 GitHub 存储库中找到作为一个完整的程序。

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19/Chp19-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter19/Chp19-Ex3.cpp)

```cpp
int main()
{
    President *p1 = President::instance("John", "Adams", 
                                        'Q', "President");
    President *p2 = President::instance("William", "Harrison",
                                        'H', "President");
    if (p1 == p2)  // Verification there's only one object
        cout << "Same instance (only one Singleton)" << endl;
    p1->Print();
    delete p1;  // Delete the Singleton – unusual.
    return 0;   // Upon checking, the SingletonDestroyer will
}           // no longer need to destroy its paired Singleton
```

在上述的`main()`函数中，注意我们明确地使用`delete p1;`来释放我们的单例`President`，而不是让实例在程序结束时通过静态对象删除来回收。幸运的是，我们在我们的`SingletonDestroyer`析构函数中包含了一个测试，让我们知道`SingletonDestroyer`是否必须删除其关联的`Singleton`，或者这个删除已经发生。

让我们来看一下修改后的输出，注意与我们原来的`main()`函数的区别：

```cpp
Creating the Singleton
Singleton previously created. Returning existing singleton
Same instance (only one Singleton)
President John Q Adams
President destructor
Singleton destructor
Person destructor
SingletonDestroyer destructor: Singleton has already been destructed
```

在我们修改后的客户端的输出中，我们可以再次看到单例`President`的创建，第二个`President`的*失败*创建请求，等等。

让我们注意一下销毁顺序以及它与我们第一个客户端的不同之处。在这里，单例`President`被明确地释放。我们可以看到`President`的正确删除，通过在`President`，`Singleton`和`Person`中的析构函数的调用和执行。现在，当应用程序即将结束并且静态`SingletonDestroyer`即将回收其内存时，我们可以看到`SingletonDestroyer`上的析构函数被调用。然而，这个析构函数不再删除其关联的`Singleton`。

### 理解设计的优势和劣势

前面（成对类）实现的单例模式的一个优点（无论使用哪个`main()`）是，我们保证了`Singleton`的正确销毁。这不管`Singleton`是长寿命的，并且通过其关联的`SingletonDestroyer`以通常方式被删除，还是在应用程序中较早地直接删除（一个罕见的情况）。

这种实现的一个缺点是继承自`Singleton`的概念。也就是说，只能有一个派生类`Singleton`包含`Singleton`类的特定机制。因为我们从`Singleton`继承了`President`，我们正在使用`President`和`President`独自使用的单例逻辑（即静态数据成员，存储在外部变量中）。如果另一个类希望从`Singleton`派生以采用这种习惯用法，`Singleton`的内部实现已经被用于`President`。哎呀！这看起来不公平。

不用担心！我们的设计可以很容易地扩展，以适应希望使用我们的`Singleton`基类的多个类。我们将扩展我们的设计以容纳多个`Singleton`对象。然而，我们仍然假设每个类类型只有一个`Singleton`实例。

现在让我们简要地看一下如何扩展单例模式来解决这个问题。

## 使用注册表允许多个类使用单例

让我们更仔细地检查一下我们当前单例模式实现的一个缺点。目前，只能有一个派生类`Singleton`能有效地利用`Singleton`类。为什么呢？`Singleton`是一个带有外部变量定义的类，用于支持类内的静态数据成员。代表`theInstance`的静态数据成员（使用外部变量`Singleton *Singleton::theInstance`实现）只能设置为一个`Singleton`实例。*不是每个类一个* - 只有一组外部变量创建了关键的`Singleton`数据成员`theInstance`和`destroyer`的内存。问题就在这里。

相反，我们可以指定一个`Registry`类来跟踪应用单例模式的类。有许多**Registry**的实现，我们将审查其中一种实现。

在我们的实现中，`Registry`将是一个类，它将类名（对于使用 Singleton 模式的类）与每个注册类的单个允许实例的`Singleton`指针配对。我们仍然将每个 Target 类从`Singleton`派生（以及根据我们的设计认为合适的任何其他类）。

我们从`Singleton`派生的每个类中的`instance()`方法将被修改如下：

+   我们在`instance()`中的第一个检查将是调用`Registry`方法（使用派生类的名称），询问该类是否以前创建过`Singleton`。如果`Registry`方法确定已经为请求的派生类型实例化了`Singleton`，则`instance()`将返回对现有实例的指针。

+   相反，如果`Registry`允许分配`Singleton`，`instance()`将分配`Singleton`，就像以前一样，将`theInstance`的继承受保护数据成员设置为分配的派生`Singleton`。静态`instance()`方法还将通过使用`setSingleton()`设置继承受保护的销毁者数据成员的反向链接。然后，我们将新实例化的派生类实例（即`Singleton`）传递给`Registry`方法，以在`Registry`中`Store()`新分配的`Singleton`。

我们注意到存在四个指向相同`Singleton`的指针。一个是从我们的派生类`instance()`方法返回的派生类类型的专用指针。这个指针将被传递给我们的客户端进行应用使用。第二个`Singleton`指针将是存储在我们继承的受保护数据成员`theInstance`中的指针。第三个`Singleton`指针将是存储在`SingletonDestroyer`中的指针。第四个指向`Singleton`的指针将存储在`Registry`中。没有问题，我们可以有多个指向`Singleton`的指针。这是`SingletonDestroyer`在其传统销毁功能中使用的一个原因-它将在应用程序结束时销毁每种类型的唯一`Singleton`。

我们的`Registry`将维护每个使用`Singleton`模式的类的一对，包括类名和相应类的（最终）指针到特定`Singleton`。每个特定`Singleton`实例的指针将是一个静态数据成员，并且还需要一个外部变量来获取其底层内存。结果是每个拥抱 Singleton 模式的类的一个额外的外部变量。

`Registry`的想法如果我们选择另外容纳 Singleton 模式的罕见使用，可以进一步扩展。如果我们选择另外容纳 Singleton 模式的罕见使用，`Registry`的想法可以进一步扩展。在这种扩展模式中的一个例子可能是，我们选择对一个只有一个校长但有多个副校长的高中进行建模。`Principal`将是`Singleton`的一个预期派生类，而多个副校长将代表`Vice-Principal`类的固定数量的实例（派生自`Singleton`）。我们的注册表可以扩展到允许`Vice-Principal`类型的`N`个注册的`Singleton`对象。

我们现在已经看到了使用成对类方法实现 Singleton 模式。我们已经将`Singleton`、`SingetonDestroyer`、Target 和 Client 的概念折叠到我们习惯看到的类框架中，即`Person`，以及我们的`Singleton`和`Person`的后代类（`President`）。让我们现在简要回顾一下我们在模式方面学到的东西，然后继续下一章。

# 总结

在本章中，我们通过接受另一个设计模式来扩展我们的编程技能，从而实现了成为更好的 C++程序员的目标。我们首先采用了一种简单的方法来探讨 Singleton 模式，然后使用`Singleton`和`SingletonDestroyer`进行了成对类的实现。我们的方法使用继承将 Singleton 的实现合并到我们的 Target 类中。可选地，我们使用多重继承将一个有用的现有基类合并到我们的 Target 类中。

利用核心设计模式，如 Singleton 模式，将帮助您更轻松地重用现有的经过充分测试的代码部分，以一种其他程序员理解的方式。通过使用熟悉的设计模式，您将为众所周知和可重用的解决方案做出贡献，采用前卫的编程技术。

现在，我们准备继续前往我们的最终设计模式，在*第二十章*中，*使用 pImpl 模式去除实现细节*。将更多的模式添加到我们的编程技能库中，使我们成为更多才多艺和有价值的程序员。让我们继续前进！

# 问题

1.  使用本章中找到的 Singleton 模式示例：

a. 实现一个`President`到`辞职()`的接口，或者实现一个接口来`弹劾()`一个`President`。您的方法应删除当前的 Singleton`President`（并从`SingletonDestroyer`中删除该链接）。`SingletonDestroyer`有一个`setSingleton()`，可能有助于帮助您删除反向链接。

b. 注意到前任的 Singleton`President`已被移除，使用`President::instance()`创建一个新的`President`。验证新的`President`已经安装。

c.（*可选*）创建一个`Registry`，允许在多个类中有效地使用`Singleton`（不是互斥的，而是当前的实现）。

1.  为什么不能将`Singleton`中的`static instance()`方法标记为虚拟，并在`President`中重写它？

1.  您能想象哪些其他例子可能很容易地融入 Singleton 模式？
