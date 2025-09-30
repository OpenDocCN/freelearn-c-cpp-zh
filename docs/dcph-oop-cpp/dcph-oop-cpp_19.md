

# 第十九章：使用单例模式

本章将继续我们的目标，即扩展你的 C++ 编程技能，使其超越核心面向对象编程（OOP）概念，目标是让你能够利用核心设计模式解决重复出现的编程难题。在编码解决方案中使用设计模式不仅可以提供更精细的解决方案，还有助于简化代码维护，并提供代码重用的潜在机会。

我们接下来将学习如何在 C++ 中有效地实现下一个核心设计模式——**单例模式**。

在本章中，我们将涵盖以下主要主题：

+   理解单例模式及其对面向对象编程（OOP）的贡献

+   在 C++ 中实现单例模式（使用简单技术与配对类方法），并使用注册表允许许多类使用单例模式

到本章结束时，你将理解单例模式及其如何确保给定类型只能存在一个实例。将另一个核心设计模式添加到你的知识体系中将进一步增强你的编程技能，帮助你成为一个更有价值的程序员。

让我们通过检查另一个常见的设计模式——单例模式，来提高我们的编程技能集。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter19`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter19)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter19` 子目录中找到，文件名为 `Chp19-Ex1.cpp`。

本章的 CiA 视频可以在以下网址观看：[`bit.ly/3ThNKe0`](https://bit.ly/3ThNKe0)。

# 理解单例模式

单例模式是一种创建型设计模式，它保证一个采用这种习惯用法的类只有一个实例；该类型的两个或多个实例可能根本无法同时存在。采用这种模式的类将被称为**单例**。

Singleton 可以通过静态数据成员和静态方法来实现。这意味着 Singleton 将拥有对当前实例的全局访问点。这种影响最初看起来很危险；将全局状态信息引入代码是导致 Singleton 有时被认为是一种反模式的一个批评。然而，通过适当使用定义 Singleton 的静态数据成员的访问区域，我们可以坚持认为对 Singleton（除了初始化之外）的访问只能使用当前类适当的静态方法（并缓解这种潜在的模式关注）。

对该模式的另一个批评是它不是线程安全的。可能存在竞争条件来进入创建 Singleton 实例的代码段。如果没有保证对那个关键代码区域的互斥性，Singleton 模式将会破裂，允许多个此类实例。因此，如果使用多线程编程，那么也必须使用适当的锁定机制来保护 Singleton 实例化的关键代码区域。使用静态内存实现的 Singleton 是同一进程中的线程之间的共享内存；有时，Singleton 可能会因为垄断资源而受到批评。

Singleton 模式可以利用几种技术来实现。每种实现方式不可避免地都会有优点和缺点。我们将使用一对相关的类，`Singleton` 和 `SingletonDestroyer`，来稳健地实现该模式。虽然存在更简单、更直接的实施方法（其中两种我们将简要回顾），但最简单的技术留下了 Singleton 可能不会被充分销毁的可能性。回想一下，析构函数可能包括重要且必要的活动。

Singleton 往往是长期存在的；因此，在应用程序终止之前销毁 Singleton 是合适的。许多客户端可能指向 Singleton，因此不应有单个客户端删除 Singleton。我们将看到 `Singleton` 将是 *自我创建* 的，因此它应该理想地是 *自我销毁*（即通过其 `SingletonDestroyer` 的帮助）。因此，配对类方法，虽然不那么简单，但将确保适当的 `Singleton` 销毁。请注意，我们的实现还将允许直接删除 Singleton；这很少见，但我们的代码也将处理这种情况。

配对类实现的 Singleton 模式将包括以下内容：

+   一个代表实现 Singleton 概念所需核心机制的 **Singleton** 类。

+   `Singleton` 确保给定的 Singleton 被正确地销毁。

+   从 `Singleton` 派生出的类代表一个我们想要确保在给定时间只能创建其类型的一个实例的类。这将是我们的 **目标** 类。

+   可选地，目标类可以同时从 `Singleton` 和另一个类派生，该类可能代表我们想要专门化或简单地包含（即 *mix-in*）的现有功能。在这种情况下，我们将从应用程序特定的类和 Singleton 类进行多重继承。

+   可选的 **Client** 类，这些类将与目标类（们）交互，以完全定义当前的应用程序。

+   或者，Singleton 也可以在目标类中实现，将类功能捆绑在一个类中。

+   真正的 Singleton 模式可以扩展以允许创建多个（离散的），但不是不确定数量的实例。这种情况很少见。

我们将关注一个传统的 Singleton 模式，确保在给定时间只有一个实例的类采用此模式存在。

让我们继续前进，首先考察两种简单的实现，然后是我们的首选配对类实现 Singleton 模式。

# 实现 Singleton 模式

Singleton 模式将被用来确保给定的类只能实例化该类的一个实例。然而，一个真正的 Singleton 模式也将具有扩展能力，允许创建多个（但数量是明确定义的）实例。Singleton 模式这个不常见且不太为人所知的限制条件是很少见的。

我们将从两个简单的 Singleton 实现开始，以了解它们的局限性。然后我们将过渡到更健壮的配对类 Singleton 实现，其最常见的模式目标是任何给定时间只允许一个目标类实例化。

## 使用简单实现

为了实现一个非常简单的 Singleton，我们将为 Singleton 本身使用一个直接的单类规范。我们将定义一个名为 `Singleton` 的类，以封装此模式。我们将确保我们的构造函数是私有的，这样就不能被多次应用。我们还将添加一个静态的 `instance()` 方法，以提供 `Singleton` 对象实例化的接口。此方法将确保私有构造只发生一次。

让我们看看这个简单的实现，它可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex1.cpp)

```cpp
class Singleton
{
private:
    static Singleton *theInstance;   // initialized below
    Singleton();  // private to prevent multiple
                  // instantiation
public:
    static Singleton *instance(); // interface for creation
    virtual ~Singleton(); // never called, unless you
};                        // delete Singleton explicitly, 
                          // which is unlikely and atypical
Singleton *Singleton::theInstance = nullptr; // extern var
                                   // to hold static member
Singleton::Singleton()
{
    cout << "Constructor" << endl;
    // Below line of code is not necessary and therefore
    // commented out – see static member init. above
    // theInstance = nullptr;
}
Singleton::~Singleton()  // the destructor is not called in
{                        // the typical pattern usage
    cout << "Destructor" << endl;
    if (theInstance != nullptr)  
    {  
       Singleton *temp = theInstance;
       // Remove pointer to Singleton and prevent recursion
       // Remember, theInstance is static, so
       // temp->theInstance = nullptr; would be duplicative 
       theInstance = nullptr;    
       delete temp;              // delete the Singleton
       // Note, delete theInstance; without temp usage
       // above would be recursive 
    }                 
}
Singleton *Singleton::instance()
{
    if (theInstance == nullptr)
        theInstance = new Singleton();// allocate Singleton
    return theInstance;
}
int main()
{
    // create Singleton
    Singleton *s1 = Singleton::instance(); 
    // returns existing Singleton (not a new one)
    Singleton *s2 = Singleton::instance(); 
    // note: addresses are the same (same Singleton!)
    cout << s1 << " " << s2 << endl; 
    return 0;
}                                         
```

注意，在上面的类定义中，我们包括数据成员 `static Singleton *theInstance;` 来表示 `Singleton` 实例本身。我们的构造函数是私有的，因此不能被多次使用来创建多个 `Singleton` 实例。相反，我们添加一个 `static Singleton *instance()` 方法来创建 `Singleton`。在此方法中，我们检查数据成员 `theInstance` 是否等于 `nullptr`，如果是，则实例化唯一的 `Singleton` 实例。

在类定义之外，我们看到外部变量（及其初始化）通过定义 `Singleton *Singleton::theInstance = nullptr;` 来支持静态数据成员的内存需求。我们还可以看到在 `main()` 函数中，我们如何调用静态 `instance()` 方法来使用 `Singleton::instance()` 创建一个 `Singleton` 实例。第一次调用此方法将实例化一个 `Singleton`，而后续调用此方法将仅返回现有 `Singleton` 对象的指针。我们可以通过打印这些对象地址来验证实例是相同的。

让我们看看这个简单程序的输出：

```cpp
Constructor
0xee1938 0xee1938
```

在之前提到的输出中，我们注意到一些可能意想不到的事情——析构函数没有被调用！如果析构函数有重要的任务要执行会怎样？

### 理解简单 Singleton 实现的关键缺陷

在简单实现中，我们没有调用我们的 `Singleton` 的析构函数，仅仅是因为我们没有通过 `s1` 或 `s2` 标识符删除动态分配的 `Singleton` 实例。为什么没有呢？显然可能有多个指向 `Singleton` 对象的指针（句柄）。决定哪个句柄应该负责删除 `Singleton` 是困难的——句柄至少需要协作或采用引用计数。

此外，`Singleton` 往往存在于整个应用程序的运行期间。这种长寿进一步表明 `Singleton` 应该负责自己的销毁。但是如何做到呢？我们很快就会看到一个实现，它将允许 `Singleton` 使用辅助类来控制自己的销毁。然而，在简单实现中，我们可能会简单地举手表示无能为力，并建议操作系统在应用程序终止时回收内存资源——包括这个小型 `Singleton` 的堆内存。这是真的；然而，如果在析构函数中需要完成一个重要的任务会怎样？我们正在遇到简单模式实现中的局限性。

如果我们需要调用析构函数，我们是否应该允许一个句柄使用例如 `delete s1;` 来删除实例？我们之前已经审查了是否允许任何句柄执行删除的问题，但现在让我们进一步检查析构函数本身可能存在的问题。例如，如果我们的析构函数假设上只包括 `delete theInstance;`，我们将有一个递归函数调用。也就是说，调用 `delete s1;` 将调用 `Singleton` 析构函数，而析构函数体内的 `delete theInstance;` 将将 `theInstance` 识别为 `Singleton` 类型并再次调用 `Singleton` 析构函数——*递归地*。

不要担心！正如所示，我们的析构函数通过首先检查`theInstance`数据成员是否不等于`nullptr`来管理递归，然后安排`temp`指向`theInstance`以保存我们需要删除的实例的句柄。然后我们执行`temp->theInstance = nullptr;`赋值操作，以防止在执行`delete temp;`时发生递归。为什么？因为`delete temp;`也会调用`Singleton`析构函数。在这个析构函数调用期间，`temp`将绑定到`this`，并在第一次递归函数调用中失败条件测试`if (theInstance != nullptr)`，从而退出递归。请注意，我们即将实施的配对类方法实现将不会出现这个问题。

重要的是要注意，在实际应用中，我们不会创建一个域无关的`Singleton`实例。相反，我们会将应用程序分解到设计中以使用该模式。毕竟，我们希望有一个有意义的类类型的`Singleton`实例。为此，我们可以以简单的`Singleton`类为基础，简单地从`Singleton`继承我们的目标（应用程序特定）类。目标类也将有私有构造函数——接受必要的参数以充分实例化目标类。然后我们将`Singleton`中的静态`instance()`方法移动到目标类，并确保`instance()`的参数列表接受传递给私有目标构造函数的必要参数。

总结来说，我们的简单实现存在固有的设计缺陷，即无法保证`Singleton`本身的正确销毁。当应用程序终止时，让操作系统收集内存不会调用析构函数。虽然选择多个句柄之一来删除`Singleton`的内存是可能的，但这需要协调，并且也破坏了通常的应用模式，允许`Singleton`在应用程序运行期间存在。

让我们接下来考虑一个使用静态局部内存引用而不是堆内存指针的简单实现，用于我们的单例（Singleton）。

## 另一种简单的实现

作为实现一个非常简单的单例的替代方法，我们将修改之前的简单类定义。首先，我们将移除静态指针数据成员（它是在`Singleton::instance()`中动态分配的）。我们不会在类中使用静态数据成员，而是在`instance()`方法中使用一个（非指针）静态局部变量来表示单例。

让我们看看这个替代实现，它可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex1b.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex1b.cpp)

```cpp
class Singleton
{ 
private:
    string data;
    Singleton(string d); // private to prevent multiple 
public:                  // instantiation
    static Singleton &instance(string); // return reference
    // destructor is called for the static local variable
    // declared in instance() before the application ends
    virtual ~Singleton();   // destructor is now called
    const string &getData() const { return data; }
};
Singleton::Singleton(string d): data(d)  // initialize data
{                                   
    cout << "Constructor" << endl;
}
Singleton::~Singleton()
{
    cout << "Destructor" << endl;
}
// Note that instance() takes a parameter to reflect how we
// can provide meaningful data to the Singleton constructor
Singleton &Singleton::instance(string d)
{   // create the Singleton with desired constructor; But,
    // we can never replace the Singleton in this approach!
    // Remember, static local vars are ONLY created and 
    // initialized once - guaranteeing one Singleton
    static Singleton theInstance(d);   
    return theInstance;
}
int main()
{   
    // First call, creates/initializes Singleton
    Singleton &s1 = Singleton::instance("Unique data"); 
    // Second call returns existing Singleton
    // (the static local declaration is ignored)
    Singleton &s2 = Singleton::instance("More data"); 
    cout << s1.getData() << " " << s2.getData() << endl;
    return 0;
}                                        
```

注意，在上述单例类定义中，我们不再包含一个静态数据成员（以及支持此数据成员的外部静态变量声明）来表示`Singleton`实例本身。相反，我们使用静态局部（非指针）变量在静态`instance()`方法中指定了单例的实现。我们的构造函数是私有的；它可以在类的静态成员函数中调用以初始化这个静态局部变量。作为静态（并且不是指针分配），这个局部变量只会在创建和初始化一次。它的空间将在应用程序启动时预留，并且静态变量将在第一次调用`instance()`时初始化。随后的`instance()`调用不会产生这个`Singleton`的替换；除了第一次调用`instance()`之外，静态局部变量声明将被忽略。请注意，`instance()`的返回值现在是对这个静态局部`Singleton`实例的引用。记住，静态局部变量将存在于整个应用程序中（它不会像其他局部变量一样存储在栈上）。

此外，非常重要的一点是，请注意我们通过参数列表将数据传递给初始化单例的`instance()`方法；这些数据随后传递给了`Singleton`构造函数。能够使用适当的数据构造单例是非常重要的。通过将单例实现为一个静态局部（非指针）变量在静态`instance()`方法中，我们有机会在这个方法内构造单例。请注意，在类中定义的静态指针数据成员也具有这种能力，因为分配（以及因此构造，如前例所示）也是在`instance()`方法内进行的。然而，类的非指针静态数据成员不允许提供有意义的构造函数参数，因为实例将在程序开始时创建和初始化，而这样的初始化器将在此之前可用（实际上不在`instance()`方法内）。在后一种情况下，单例将只从`instance()`返回，而不会在其中初始化。

现在，请注意，在`main()`中，我们调用静态`instance()`方法，使用`Singleton::instance()`创建一个`Singleton`实例。我们使用从`Singleton::instance()`返回的单例的引用创建了一个别名`s1`。对这个方法的第一次调用将实例化单例，而对该方法的后续调用将仅返回现有`Singleton`对象的引用。我们可以通过打印单例中包含的数据来验证这两个别名（`s1`和`s2`）引用的是同一个对象。

让我们看看这个简单程序的输出：

```cpp
Constructor
Unique data 
Unique data
Destructor
```

在之前提到的输出中，我们注意到在应用程序结束之前，析构函数会自动被调用以清理 Singleton。我们还注意到，尝试创建第二个 `Singleton` 实例只会返回现有的 `Singleton`。这是因为静态局部变量 `theInstance` 只在应用程序中创建和初始化一次，无论 `instance()` 被调用多少次（静态局部变量的一个简单属性）。然而，这种实现也有潜在的缺点；让我们看看。

### 理解替代简单 Singleton 实现的限制

在 `instance()` 中使用非指针静态局部变量来实现 Singleton 并没有给我们提供改变 Singleton 的灵活性。在函数中，任何静态局部变量在应用程序开始时都会为其分配内存；这个内存只初始化一次（在第一次调用 `instance()` 时）。这意味着我们总是在应用程序中恰好有一个 `Singleton`。即使我们从未调用 `instance()` 来初始化它，这个 `Singleton` 的空间也存在。

此外，由于静态局部变量的实现方式，这个实现中的 `Singleton` 不能被替换为另一个 `Singleton` 对象。在某些应用程序中，我们可能一次只想有一个 `Singleton` 对象，但同时也希望能够将一个 `Singleton` 的实例替换为另一个实例。例如，想象一个组织可以有一个总统；然而，希望（Singleton）总统每隔几年可以被不同的（Singleton）总统所取代。使用指针的初始简单实现允许这种可能性，但存在潜在的缺陷，即其析构函数从未被调用。每个简单实现都有潜在的缺点。

现在，因为我们理解了简单 Singleton 实现的限制，我们将转向 Singleton 模式的首选配对类实现。配对类方法将保证我们的 `Singleton` 能够正确销毁，无论是应用程序允许在应用程序终止前通过故意配对类（最常见的情况）销毁 `Singleton`，还是在应用程序中提前销毁 `Singleton` 的罕见情况下。这种方法还将允许我们用一个 Singleton 的另一个实例替换 Singleton。

## 使用更健壮的配对类实现

为了以良好的封装方式实现带有配对类方法的单例模式，我们将定义一个单例类，仅用于添加创建单个实例的核心机制。我们将这个类命名为`Singleton`。然后，我们将添加一个辅助类到`Singleton`中，称为`SingletonDestroyer`，以确保在应用程序终止之前，我们的`Singleton`实例总是经过适当的销毁。这两个类将通过聚合和关联相关联。更具体地说，`Singleton`在概念上包含一个`SingletonDestroyer`（聚合），而`SingletonDestroyer`将保持对其（外部）`Singleton`的关联，它在概念上是嵌入的。由于`Singleton`和`SingletonDestroyer`的实现是通过静态数据成员，这种聚合是概念性的——静态成员作为外部变量存储。

一旦定义了这些核心类，我们将考虑如何将单例模式融入到我们熟悉的类层次结构中。让我们设想，我们想要实现一个封装“总统”概念的类。无论是国家的总统还是大学的校长，在某个特定的时间点只有一个总统是很重要的。“总统”将是我们的目标类；“总统”是利用我们的单例模式的好候选者。

有趣的是，虽然某个特定时间点只有一个总统，但总统是可以被替换的。例如，美国总统的任期只有四年，可能还会再连任一个任期。大学校长可能也有类似的情况。总统可能通过辞职、弹劾或死亡提前离开，或者简单地在其任期结束时离开。一旦现任总统的存在被移除，那么实例化一个新的单例“总统”就是可以接受的。因此，我们的单例模式允许在某个特定时间点只有一个目标类的单例。

反思如何最好地实现“总统”类，我们意识到“总统”是“人”的一种，并且还需要“混合”单例功能。考虑到这一点，我们现在有了我们的设计。“总统”将使用多重继承来扩展“人”的概念，并混合单例的功能。

当然，我们可以从头开始构建“总统”类，但为什么这样做，当“总统”类中的“人”组件已经由一个经过良好测试和可用的类表示呢？同样，当然，我们可以将单例类信息嵌入到我们的“总统”类中，而不是从单独的单例类继承它。绝对，这也是一个选择。然而，我们的应用程序将封装解决方案的每一部分。这将使未来的重用更加容易。尽管如此，设计选择是多种多样的。

### 指定单例和 SingletonDestroyer

让我们看一下我们的单例模式的机制，首先检查`Singleton`和`SingletonDestroyer`类定义。这些类协同工作以实现单例模式。这个例子作为一个完整的程序，可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex2.cpp)

```cpp
class Singleton;    // Necessary forward class declarations
class SingletonDestroyer;
class Person;
class President;
class SingletonDestroyer   
{
private:
    Singleton *theSingleton = nullptr;
public:
    SingletonDestroyer(Singleton *s = nullptr) 
        { theSingleton = s; }
    // disallow copies and assignment
    SingletonDestroyer(const SingletonDestroyer &) 
                                    = delete; 
    SingletonDestroyer &operator=
       (const SingletonDestroyer &) = delete;
    ~SingletonDestroyer(); // dtor shown further below
    void setSingleton(Singleton *s) { theSingleton = s; }
    Singleton *getSingleton() { return theSingleton; }
};
```

在上述代码段中，我们首先声明了几个前向类声明，例如`class Singleton;`。这些声明允许在编译器看到它们的完整类定义之前对这些数据类型进行引用。

接下来，让我们看一下我们的`SingletonDestroyer`类定义。这个简单的类包含一个私有数据成员`Singleton *theSingleton;`，它将关联到`Singleton`，`SingletonDestroyer`将有一天负责释放它（我们很快将检查`SingletonDestroyer`的析构函数定义）。注意，我们的析构函数不是虚拟的，因为这个类不是用来专门化的。

注意，我们的构造函数为`Singleton *`指定了默认值`nullptr`，这是一个输入参数。`SingletonDestroyer`还包含两个成员函数`setSingleton()`和`getSingleton()`，它们仅仅提供了设置和获取相关联的`Singleton`成员的手段。

还要注意，在`SingletonDestroyer`中使用复制构造函数和重载的赋值运算符都已被在它们的原型中使用`=delete`禁止。

在我们检查这个类的析构函数之前，让我们看一下`Singleton`的类定义：

```cpp
// Singleton will be mixed-in using inheritance with a
// Target class. If Singleton is used stand-alone, the data 
// members would be private. Also be sure to add a
// Static *Singleton instance(); 
// method to the public access region.
class Singleton
{
protected:    // protected data members
    static Singleton *theInstance;
    static SingletonDestroyer destroyer;
protected:   // and protected member functions
    Singleton() = default;
    // disallow copies and assignment
    Singleton(const Singleton &) = delete; 
    Singleton &operator=(const Singleton &) = delete; 
    friend class SingletonDestroyer;
    virtual ~Singleton() 
        { cout << "Singleton destructor" << endl; }
};
```

上述`Singleton`类包含受保护的`static Singleton *theInstance;`数据成员，它将代表（当分配时）指向使用单例语法的类分配的唯一实例的指针。

受保护的`static SingletonDestroyer destroyer;`数据成员代表一个概念上的聚合或包含成员。这种包含仅仅是概念上的，因为静态数据成员不会存储在任何实例的内存布局中；相反，它们存储在外部内存中，并通过名称混淆来显示为类的一部分。这个（概念上的）聚合子对象`destroyer`将负责正确地销毁`Singleton`。回想一下，`SingletonDestroyer`与唯一的`Singleton`有关联，代表`SingletonDestroyer`在概念上包含的外部对象。这种关联是`SingletonDestroyer`访问 Singleton 的方式。

当实现静态数据成员`static SingletonDestroyer destroyer;`的外部变量的内存在使用结束时消失时，`SingletonDestroyer`的析构函数（静态的、概念上的子对象）将被调用。这个析构函数将`delete theSingleton;`，确保外部的`Singleton`对象（它是动态分配的）将运行适当的析构函数序列。因为`Singleton`中的析构函数是受保护的，所以必须将`SingletonDestroyer`指定为`Singleton`的友元类。

注意，`Singleton`中复制构造函数和重载赋值运算符的使用都已被在它们的原型中使用`=delete`禁止。

在我们的实现中，我们假设`Singleton`将通过继承混合到派生目标类中。它将位于派生类（即打算使用 Singleton 惯用语的类）中，我们将提供所需的静态`instance()`方法来创建`Singleton`实例。请注意，如果`Singleton`被用作独立的类来创建单例，我们将在`Singleton`的公共访问区域添加`static Singleton* instance()`。我们还将把数据成员从受保护的访问区域移动到私有访问区域。然而，具有应用程序无关的单例仅用于演示概念。相反，我们将 Singleton 惯用语应用于实际需要使用此惯用语的类型。

在我们的`Singleton`和`SingletonDestroyer`类定义就绪后，接下来让我们检查这些类的剩余实现必要性：

```cpp
// External (name mangled) vars to hold static data mbrs
Singleton *Singleton::theInstance = nullptr;
SingletonDestroyer Singleton::destroyer;
// SingletonDestroyer destructor definition must appear 
// after class definition for Singleton because it is 
// deleting a Singleton (so its destructor can be seen)
// This is not an issue when using header and source files.
SingletonDestroyer::~SingletonDestroyer()
{   
    if (theSingleton == nullptr)
        cout << "SingletonDestroyer destructor: Singleton 
                 has already been destructed" << endl;
    else
    {
        cout << "SingletonDestroyer destructor" << endl;
        delete theSingleton;   
    }                          
}
```

在上述代码片段中，我们首先注意到两个外部变量定义，它们为`Singleton`类中的两个静态数据成员提供内存支持——即`Singleton *Singleton::theInstance = nullptr;`和`SingletonDestroyer Singleton::destroyer;`。回想一下，静态数据成员不存储在其指定的类实例中。相反，它们存储在外部变量中；这两个定义指定了内存。请注意，数据成员都被标记为`protected`。这意味着尽管我们可以以这种方式直接定义它们的存储，但我们不能通过`Singleton`的静态成员函数之外的方式访问这些数据成员。这将给我们带来一些安慰。尽管存在对静态数据成员的潜在全局访问点，但它们施加的受保护访问区域要求使用`Singleton`类的适当静态方法来正确操作这些重要的成员。

接下来，请关注`SingletonDestroyer`的析构函数。这个巧妙的析构函数首先检查其与负责的`Singleton`的关联是否等于`nullptr`。这种情况很少见，并且发生在非常罕见的情况下，即客户端直接使用显式的`delete`释放 Singleton 对象。

在`SingletonDestroyer`析构函数中的通常销毁场景将是执行`else`子句，其中`SingletonDestructor`作为一个静态对象将负责其配对的`Singleton`的删除和销毁。记住，在`Singleton`中会有一个包含的`SingletonDestroyer`对象。这个静态（概念上）子对象的内存不会消失，直到应用程序完成。回想一下，静态内存实际上不是任何实例的一部分。然而，静态子对象将在`main()`完成之前被销毁。因此，当`SingletonDestroyer`被销毁时，其通常情况将是`delete theSingleton;`，这将释放其配对的 Singleton 的内存，允许`Singleton`被正确销毁。

单例模式背后的驱动设计决策是，单例是一个长期存在的对象，其销毁最常正确地发生在应用程序的末尾。单例负责其自身的内部目标对象创建，因此单例不应该被客户端删除（从而销毁）。相反，首选的机制是当`SingletonDestroyer`作为一个静态对象被移除时，删除其配对的`Singleton`。

尽管如此，偶尔在应用程序过程中删除`Singleton`也有合理的场景。如果永远不会创建替换的`Singleton`，我们的`SingletonDestroyer`析构函数仍然可以正确工作，识别出其配对的`Singleton`已经被释放。然而，更有可能的是，我们的`Singleton`将在应用程序的某个地方被另一个`Singleton`实例替换。回想一下我们的应用程序示例，总统可能会被弹劾、辞职或去世，但将被另一位总统取代。在这些情况下，直接删除`Singleton`并创建一个新的`Singleton`是可以接受的。在这种情况下，`SingletonDestroyer`现在将引用替换的`Singleton`。

### 从 Singleton 派生目标类

接下来，让我们看看我们如何从`Singleton`创建我们的目标类，`President`：

```cpp
// Assume our Person class is as we are accustomed
// A President Is-A Person and also mixes-in Singleton 
class President: public Person, public Singleton
{
private:
    President(const string &, const string &, char, 
              const string &);
public:
    ~President() override;   // virtual destructor
    // disallow copies and assignment
    President(const President &) = delete;  
    President &operator=(const President &) = delete; 
    static President *instance(const string &, 
                    const string &, char, const string &);
};
President::President(const string &fn, const string &ln, 
    char mi, const string &t): Person(fn, ln, mi, t),
                               Singleton()
{
}
President::~President()
{
    destroyer.setSingleton(nullptr);  
    cout << "President destructor" << endl;
}
President *President::instance(const string &fn, 
           const string &ln, char mi, const string &t)
{
    if (theInstance == nullptr)
    {
        theInstance = new President(fn, ln, mi, t);
        destroyer.setSingleton(theInstance);
        cout << "Creating the Singleton" << endl;
    }
    else
        cout << "Singleton previously created. 
                 Returning existing singleton" << endl;
    // below cast is necessary since theInstance is 
    // a Singleton *
    return dynamic_cast<President *>(theInstance);  
}                              
```

在我们之前提到的目标类`President`中，我们只是使用公有继承从`Person`继承`President`，然后多重继承`President`从`Singleton`以*混合*`Singleton`机制。

我们将构造函数放在私有访问区域。静态方法 `instance()` 将在内部使用此构造函数创建一个且仅有一个 `Singleton` 实例，以符合模式。没有默认构造函数（不寻常），因为我们不希望允许创建没有相关详细信息的 `President` 实例。回想一下，如果我们提供了替代的构造函数接口，C++ 不会链接默认构造函数。由于我们不希望复制 `President` 或将 `President` 赋值给另一个潜在的 `President`，我们在这些方法的原型中使用 `=delete` 规范禁止复制和赋值。

我们为 `President` 定义的析构函数简单而关键。如果我们的 `Singleton` 对象将被显式删除，我们通过设置 `destroyer.setSingleton(nullptr);` 来做准备。回想一下，`President` 继承了受保护的 `static SingletonDestroyer destroyer;` 数据成员。在这里，我们将破坏者关联的 `Singleton` 设置为 `nullptr`。然后，在 `President` 的析构函数中的这一行代码使得 `SingletonDestroyer` 中的析构函数能够准确地依赖于检查其关联的 `Singleton` 是否已经被删除，然后再开始通常的 `Singleton` 对应析构。

最后，我们定义了一个静态方法来为我们的 `President` 作为 `Singleton` 提供创建接口，即 `static President *instance(const string &, const string &, char, const string &);`。在 `instance()` 的定义中，我们首先检查继承的受保护数据成员 `Singleton *theInstance` 是否等于 `nullptr`。如果我们还没有分配 `Singleton`，我们将使用上述私有构造函数分配 `President` 并将这个新分配的 `President` 实例赋值给 `theInstance`。这是一个从 `President *` 到 `Singleton *` 的向上转换，这在公共继承边界上没有问题。然而，如果在 `instance()` 方法中我们发现 `theInstance` 不等于 `nullptr`，我们只需返回之前分配的 `Singleton` 对象的指针。由于用户无疑会想将此对象用作 `President` 以享受继承的 `Person` 功能，我们将 `theInstance` 向下转换为 `President *` 以从该方法返回。

最后，让我们考虑一下我们整体应用程序中一个示例客户端的物流。在其最简单的形式中，我们的客户端将包含一个 `main()` 函数来驱动应用程序并展示我们的单例模式。

### 在客户端中将模式组件组合在一起

现在我们来看看我们的 `main()` 函数，看看我们的模式是如何编排的：

```cpp
int main()
{ 
    // Create a Singleton President
    President *p1 = President::instance("John", "Adams", 
                                        'Q', "President");
    // This second request will fail, returning 
    // the original instance
    President *p2 = President::instance("William",
                            "Harrison", 'H', "President");
    if (p1 == p2)   // Verification there's only one object
        cout << "Same instance (only 1 Singleton)" << endl;
    p1->Print();
    // SingletonDestroyer will release Singleton at end
    return 0;
}
```

回顾前面代码中的 `main()` 函数，我们首先使用 `President *p1 = President::instance("John", "Adams", 'Q', "President");` 分配了一个 Singleton `总统`。然后我们在下一行代码中尝试使用 `*p2` 分配另一个 `总统`。因为我们只能有一个 Singleton（一个 `总统` *混入* 一个 Singleton），所以返回了一个指向现有 `总统` 的指针，并将其存储在 `p2` 中。我们通过比较 `p1 == p2` 来验证只有一个 Singleton；指针确实指向了同一个实例。

接下来，我们利用 `总统` 实例的预期方式使用它，例如，通过使用从 `Person` 继承的一些成员函数。例如，我们调用 `p1->Print();`。当然，我们的 `总统` 类可以添加一些专门的功能，这些功能也适合在 Client 中使用。

现在，在 `main()` 函数的末尾，我们的静态对象 `SingletonDestroyer Singleton::destroyer;` 将在内存回收之前适当地被析构。正如我们所见，`SingletonDestroyer` 的析构函数将（大多数情况下）使用 `delete theSingleton;` 对其关联的 Singleton（实际上是一个 `总统`）执行 `delete` 操作。这将触发 `总统`、`Singleton` 和 `Person` 析构函数的调用和执行（从最专门的子对象到最一般的子对象）。由于 `Singleton` 中的析构函数是虚拟的，我们保证从正确的级别开始销毁，并包括所有析构函数。

让我们看看这个程序的输出：

```cpp
Creating the Singleton
Singleton previously created. Returning existing singleton
Same instance (only 1 Singleton)
President John Q Adams
SingletonDestroyer destructor
President destructor
Singleton destructor
Person destructor
```

在前面的输出中，我们可以可视化 Singleton `总统` 的创建过程，以及看到对 `总统` 的第二次 `instance()` 请求仅仅返回现有的 `总统`。然后我们看到打印出的 `总统` 的详细信息。

最有趣的是，我们可以看到 Singleton 的销毁序列，这是由 SingletonDestroyer 的静态对象回收驱动的。通过在 `SingletonDestroyer` 的析构函数中适当地删除 Singleton，我们看到 `总统`、`Singleton` 和 `Person` 析构函数各自被调用，因为它们有助于完整的 `总统` 对象。

### 检查显式 Singleton 销毁及其对 SingletonDestroyer 析构函数的影响

让我们看看 Client 的一个替代版本，它有一个不同的 `main()` 函数。在这里，我们强制删除我们的 Singleton；这种情况很少见。在这种情况下，我们的 `SingletonDestroyer` 不会删除其配对的 Singleton。这个例子作为一个完整的程序，可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter19/Chp19-Ex3.cpp)

```cpp
int main()
{
    President *p1 = President::instance("John", "Adams", 
                                        'Q', "President");
    President *p2 = President::instance("William",
                             "Harrison", 'H', "President");
    if (p1 == p2)  // Verification there's only one object
        cout << "Same instance (only 1 Singleton)" << endl;
    p1->Print();
    delete p1;  // Delete the Singleton – unusual.
    return 0;   // Upon checking, the SingletonDestroyer 
}   // will no longer need to destroy its paired Singleton
```

在上述`main()`函数中，请注意我们明确地使用`delete p1;`释放了我们的 Singleton `President`，而不是允许实例在程序结束时通过静态对象删除来回收。幸运的是，我们在`SingletonDestroyer`析构函数中包含了一个测试，以让我们知道是否必须删除关联的`Singleton`，或者这种删除是否已经发生。

让我们看看修订后的输出，以注意与我们的原始`main()`之间的差异：

```cpp
Creating the Singleton
Singleton previously created. Returning existing singleton
Same instance (only 1 Singleton)
President John Q Adams
President destructor
Singleton destructor
Person destructor
SingletonDestroyer destructor: Singleton has already been destructed
```

在我们修订后的`Client`的上述输出中，我们再次可以可视化`Singleton` `President`的创建，第二个`President`创建请求的*失败*，等等。

让我们注意到销毁序列以及它与我们的第一个`Client`的不同之处。在这里，Singleton `President`被明确地释放。我们可以通过调用和执行`President`、`Singleton`和`Person`中的析构函数来看到`President`的正确删除，因为每个都执行了。现在，当应用程序即将结束时，静态的`SingletonDestroyer`即将回收其内存，我们可以可视化对`SingletonDestroyer`调用的析构函数。然而，这个析构函数将不再删除其关联的`Singleton`。

### 理解设计优势和劣势

单例模式的前一个（配对类）实现的优势（无论使用哪个`main()`）在于我们保证了`Singleton`的正确销毁。这发生在无论`Singleton`是长期存在的并且通常由其关联的`SingletonDestroyer`删除，还是它较早地在应用程序中直接删除（一个罕见的情况）。

这种实现的缺点源于`Singleton`的概念。也就是说，只能有一个`Singleton`的派生类包含`Singleton`类的特定机制。因为我们从`Singleton`继承了`President`，所以我们正在为`President`和仅`President`使用 Singleton 物流（即静态数据成员，存储在外部变量中）。如果另一个类希望从`Singleton`派生以采用这种习语，`Singleton`的内部实现已经被`President`使用。哎呀！这看起来似乎不太公平。

不要担心！我们的设计可以很容易地扩展以适应希望使用我们的`Singleton`基类的多个类。我们将增强我们的设计以容纳多个`Singleton`对象。然而，我们假设我们仍然希望每个类类型只有一个`Singleton`实例。

另一个潜在的担忧是线程安全性。例如，如果将使用多线程编程，我们需要确保我们的`static President::instance()`方法表现得像是原子的，也就是说，不可中断的。我们可以通过仔细同步对静态方法本身的访问来实现这一点。

现在让我们简要地看看我们如何扩展 Singleton 模式来解决此问题。

## 使用注册表允许许多类使用 Singleton

让我们更详细地检查我们当前 Singleton 模式实现的一个缺点。目前，只有一个派生自 `Singleton` 的类可以有效地利用 `Singleton` 类。为什么是这样？`Singleton` 是一个带有外部变量定义的类，以支持类内的静态数据成员。代表 `theInstance` 的静态数据成员（使用外部变量 `Singleton *Singleton::theInstance` 实现）只能设置为单个 `Singleton` 实例。*不是每个类一个* – 只有一组外部变量为关键的 `Singleton` 数据成员 `theInstance` 和 `destroyer` 创建内存。问题就出在这里。

我们可以指定一个 `Registry` 类来跟踪应用 Singleton 模式的类。有许多 `Registry` 的实现，我们将审查其中一种实现。

在我们的实现中，`Registry` 将是一个将类名（用于采用 Singleton 模式的类）与每个注册类单个允许实例的 `Singleton` 指针配对的类。我们仍然会将每个目标类从 `Singleton` 派生出来（以及从任何其他我们认为合适的设计中派生出来）。

我们将对每个从 `Singleton` 派生出来的类的 `instance()` 方法进行修订，如下所示：

+   在 `instance()` 中的第一次检查将是调用一个 `Registry` 方法（带有派生类的名称），询问是否为该类之前创建了一个 `Singleton`。如果 `Registry` 方法确定请求的派生类型的 `Singleton` 之前已经被实例化，`instance()` 将返回现有实例的指针。

+   相反，如果 `Registry` 授予了分配 `Singleton` 的权限，`instance()` 将像以前一样分配 `Singleton`，将继承的受保护的 `theInstance` 数据成员设置为分配的派生 `Singleton`。静态 `instance()` 方法还将通过继承的受保护的销毁器数据成员使用 `setSingleton()` 设置回链。然后我们将新实例化的派生类实例（它是一个 `Singleton`）传递给 `Registry` 方法以 `Store()` 在 `Registry` 中存储新分配的 `Singleton`。

我们注意到将存在四个指向相同 `Singleton` 的指针。一个将是我们的派生类类型的专用指针，它从我们的派生类 `instance()` 方法返回。这个指针将被交给我们的客户端用于应用。第二个 `Singleton` 指针将是存储在我们继承的受保护数据成员 `theInstance` 中的指针。第三个 `Singleton` 指针将是存储在 `SingletonDestroyer` 中的指针。指向 `Singleton` 的第四个指针将是一个存储在 `Registry` 中的指针。没问题，我们可以有多个指向 `Singleton` 的指针。这是 `SingletonDestroyer` 在其传统销毁能力中如此重要的原因之一——它将在应用程序结束时销毁每个类型的唯一 `Singleton`。

我们的 `Registry` 将为每个使用 `Singleton` 模式的类维护一对，包括一个类名和对应特定 `Singleton` 的（最终）指针。每个特定 `Singleton` 实例的指针将是一个静态数据成员，并且还需要一个外部变量来获取其底层内存。结果是每个采用 Singleton 模式的类将额外有一个外部变量。

如果我们选择进一步扩展 `Registry` 的概念，以允许在罕见的情况下使用 Singleton 模式，允许每个类类型有多个（但数量有限的）`Singleton` 对象，那么这个概念还可以进一步扩展。这种受控的多个单例的罕见存在被称为 `Principal`，将是 `Singleton` 的预期派生类，而多个副校长将代表 `Vice-Principal` 类（从 `Singleton` 派生）的固定数量的实例。我们的注册表可以扩展到允许 `Vice-Principal` 类型最多注册 `N` 个 `Singleton` 对象（多例）。

我们现在已经看到了使用配对类方法实现的 Singleton 模式。我们将 `Singleton`、`SingletonDestroyer`、目标类和客户端的概念融合到我们习惯看到的类框架中，即 `Person`，以及我们的 `Singleton` 和 `Person` 的派生类（`President`）。现在，让我们简要回顾一下与模式相关的内容，然后继续下一章。

# 摘要

在本章中，我们通过采用另一个设计模式来扩展我们的编程技能，进一步实现了成为更好的 C++ 程序员的目标。我们首先采用了两种简单的方法来探索 Singleton 模式，然后使用 `Singleton` 和 `SingletonDestroyer` 的配对类实现。我们的方法使用继承将 Singleton 的实现纳入我们的目标类。可选地，我们通过多重继承将一个有用的现有基类纳入我们的目标类。

利用核心设计模式，如 Singleton 模式，将帮助您更容易地以其他程序员能理解的方式重用现有、经过良好测试的代码部分。通过使用熟悉的设计模式，您将为具有前卫编程技术的易于理解和可重用的解决方案做出贡献。

我们现在准备继续前进，进入我们的最后一个设计模式*第二十章*，*使用 pImpl 模式去除实现细节*。将更多模式添加到我们的编程技能库中，使我们成为更加多才多艺且受重视的程序员。让我们继续前进！

# 问题

1.  使用本章中找到的 Singleton 模式示例，创建一个程序来完成以下任务：

    1.  实现一个用于`President`的`Resign()`接口或实现`Impeach()`接口。你的方法应该删除当前的`Singleton` `President`（并从`SingletonDestroyer`中移除那个链接）。`SingletonDestroyer`有一个`setSingleton()`方法，这可能有助于移除回链。

    1.  注意到之前的`Singleton` `President`已经被移除，使用`President::instance()`创建一个新的`President`。验证新的`President`已经被安装。

    1.  （可选）创建一个`Registry`以允许`Singleton`在多个类中有效使用（不是相互排他地，如当前实现那样）。

1.  为什么不能将`Singleton`中的`static instance()`方法标记为虚拟并在`President`中重写它？

1.  你还能想象出哪些其他示例可以轻松地结合 Singleton 模式？
