# 21

# 使 C++ 更加安全

本附录章节将深入探讨作为 C++ 程序员，我们可以如何使语言在日常使用中尽可能安全。我们已经从基本语言特性进步到我们的核心兴趣——使用 C++ 进行面向对象编程，再到额外的有用语言特性库（异常、运算符重载、模板和 STL），以及设计模式，以提供解决重复出现的面向对象编程问题的知识库。在旅途中，我们始终看到 C++ 需要我们额外的关注，以避免棘手和可能存在问题的编程情况。C++ 是一种允许我们做任何事情的语言，但随之而来的是需要指导方针来确保我们的编程遵循安全实践。毕竟，我们的目标是创建能够成功运行且无错误的程序，并且易于维护。C++ 能够做任何事情的能力需要与良好的实践相结合，以使 C++ 更加安全。

本章的目标是回顾我们在前几章中介绍过的主题，并从安全的角度进行审查。我们还将结合与之前内容紧密相关的话题。本章的目的不是全面覆盖全新的主题或深入探讨之前的话题，而是提供一组更安全的编程实践，并鼓励在需要时进一步了解每个主题。其中一些主题本身可以涵盖整个章节（或书籍）！

在本附录章节中，我们将介绍一些流行的编程约定，以满足我们的安全挑战：

+   重新审视智能指针（唯一、共享和弱引用），以及补充的惯用法（RAII）

+   使用现代`for`循环（基于范围的、for-each）以避免常见错误

+   添加类型安全：使用`auto`代替显式类型声明

+   优先使用 STL 类型作为简单容器（如`std::vector`等）

+   适当地使用`const`以确保某些项不被修改

+   理解线程安全问题

+   考虑核心编程指导原则的基本要素，例如优先初始化而不是赋值，或者只选择`virtual`、`override`或`final`中的一个

+   采用 C++ 核心编程指南进行安全编程（如果需要，构建和组装一个）

+   理解 C++ 编程中的资源安全

在本章结束时，您将了解一些当前行业在 C++ 中安全编程的标准和关注点。本章的目的不是列出 C++ 中所有安全问题和实践的综合列表，而是展示作为成功的 C++ 程序员，您需要关注的问题类型。在某些情况下，您可能希望更深入地研究一个主题，以获得更全面的能力和熟练度。将安全性添加到您的 C++ 编程中会使您成为一个更有价值的程序员，因为您的代码将更加可靠，具有更长的生命周期和更高的成功率。

让我们通过考虑如何使 C++ 更安全来完善我们的编程技能集。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter21`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter21)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是当前章节中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter21` 子目录中找到，文件名为 `Chp21-Ex1.cpp`。一些程序位于示例中指示的可应用子目录中。

本章的 CiA 视频可以在以下网址观看：[`bit.ly/3wpOG6b`](https://bit.ly/3wpOG6b)。

# 重新审视智能指针

在整本书中，我们已经对如何使用原生或本地 C++ 指针有了合理的理解，包括与堆实例相关的内存分配和释放。我们坚持使用原生 C++ 指针，因为它们在现有的 C++ 代码中无处不在。了解如何正确利用原生指针对于处理目前广泛使用的现有 C++ 代码量至关重要。但是，对于新创建的代码，有一种更安全的方式来操作堆内存。

我们已经看到，使用原生指针进行动态内存管理是一项繁重的工作！特别是当可能有多个指针指向同一块内存时。我们讨论了引用计数到共享资源（如堆内存）以及当所有实例都完成对共享内存的操作时删除内存的机制。我们还知道，内存释放很容易被忽视，从而导致内存泄漏。

我们还亲身体验到，原生指针的错误可能代价高昂。当我们解引用我们不想访问的内存，或者解引用未初始化的原生指针（解释内存包含有效的地址和该地址上的有意义数据——这两者实际上都不有效）时，我们的程序可能会突然结束。通过指针算术遍历内存可能会被一个本应熟练的程序员的错误所困扰。当出现内存错误时，指针或堆内存误用往往是罪魁祸首。

当然，使用引用可以减轻许多原生指针错误带来的负担。但引用仍然可以指向某人忘记释放的已解引用堆内存。出于这些以及其他许多原因，智能指针在 C++ 中变得流行，其主要目的是使 C++ 更安全。

我们在之前的章节中讨论了智能指针，并看到了它们在我们使用 pImpl 模式（使用`unique_ptr`）时的实际应用。但除了唯一指针之外，还有更多类型的智能指针需要我们回顾：共享和弱指针。让我们还设定一个编程前提（未来风格指南的补充），即在我们的新代码中优先使用智能指针而不是原生指针，以实现指针安全的目的和价值。

回想一下，**智能指针**是一个小的包装类，它封装了一个原始指针或原生指针，确保当包装对象超出作用域时，它所包含的指针会自动删除。标准 C++库中实现的*唯一*、*共享*和*弱*智能指针使用模板来为任何数据类型创建特定的智能指针类别。

虽然我们可以为每种智能指针类型深入探讨整整一章，但我们将简要回顾每种类型，作为起点，鼓励在新创建的代码中使用它们，以支持我们使 C++更安全的目标。

现在，让我们逐一回顾每种智能指针类型。

## 使用智能指针 - 唯一

回想一下，在标准 C++库中，`unique_ptr`是一种智能指针类型，它封装了对给定堆内存资源的独占所有权和访问。`unique_ptr`不能被复制；`unique_ptr`的所有者将独占使用该指针。唯一指针的所有者可以选择将这些指针移动到其他资源，但后果是原始资源将不再包含`unique_ptr`。回想一下，我们必须使用`#include <memory>`来包含`unique_ptr`的定义。

这里有一个非常简单的例子，说明了如何创建唯一指针。这个例子可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex1.cpp)

```cpp
#include <iostream>
#include <memory>
#include "Person.h"
using std::cout;   // preferred to: using namespace std;
using std::endl;
using std::unique_ptr;
// We will create unique pointers, with and without using
// the make_unique (safe wrapper) interface
int main()
{
    unique_ptr<int> p1(new int(100));
    cout << *p1 << endl;
    unique_ptr<Person> pers1(new Person("Renee",
                             "Alexander",'K', "Dr."));
    (*pers1).Print();      // or use: pers1->Print();
    unique_ptr<Person> pers2; // currently uninitialized
    pers2 = move(pers1);// take over another unique
                           // pointer's resource
    pers2->Print();        // or use: (*pers2).Print();  
    // make_unique provides a safe wrapper, eliminating
    // obvious use of heap allocation with new()
    auto pers3 = make_unique<Person>("Giselle", "LeBrun",
                                      'R', "Ms.");
    pers3->Print();        
    return 0;
}
```

首先，请注意，因为我们包含了`using std::unique_ptr;`，所以我们不需要在唯一指针声明中对`unique_ptr`或`make_unique`进行`std::`限定。在这个小程序中，我们创建了几个唯一指针，从指向一个整数的一个指针`p1`和一个指向`Person`实例的指针`pers1`开始。由于我们使用了唯一指针，因此每个变量都独占使用它所指向的堆内存。

接下来，我们介绍一个唯一指针`pers2`，它通过`pers2 = move(pers1);`接管了原本分配并链接到`pers1`的内存。原始变量不再能访问这块内存。请注意，尽管我们可以为`pers2`分配它自己的唯一堆内存，但我们选择展示如何使用`move()`允许一个唯一指针将其内存释放给另一个唯一指针。使用`move()`来改变唯一指针的所有权是典型的，因为唯一指针不能被复制（因为这会导致两个或更多指针共享相同的内存，因此它们不是唯一的！）

最后，我们创建另一个唯一指针`pers3`，它使用`make_unique`作为包装器来为`pers3`将表示的唯一指针分配堆内存。使用`make_unique`的偏好是，`new()`的调用将内部为我们执行。此外，在对象构造期间抛出的任何异常也将由我们处理，如果底层的`new()`没有成功完成并且需要调用`delete()`，也是如此。

堆内存将由系统自动管理；这是使用智能指针的一个好处。

下面是`unique_ptr`示例的输出：

```cpp
100
Dr. Renee K. Alexander
Dr. Renee K. Alexander
Ms. Giselle LeBrun
Person destructor
Person destructor
```

在底层，当内存不再被利用时，智能指针所指向的每个对象都将自动调用析构函数。在本例中，当`main()`中的局部对象超出作用域并被从栈中弹出时，代表每个`Person`对象的析构函数将代表我们被调用。请注意，我们的`Person`析构函数包含一个`cout`语句，这样我们就可以可视化地看到只有两个`Person`对象被销毁。在这里，被销毁的`Person`对象代表通过`move()`语句从`pers1`接管实例的`pers2`，以及使用`make_unique`包装器创建的`pers3`对象。

接下来，让我们添加使用共享和弱智能指针的示例。

## 使用智能指针 – 共享

标准 C++库中的`shared_ptr`是一种智能指针类型，允许共享对给定资源的所有权和访问。对于该资源的最后一个共享指针将触发资源的销毁和内存释放。共享指针可用于多线程应用程序；然而，如果使用非常量成员函数来修改共享资源，则可能会发生竞争条件。由于共享指针仅提供引用计数，我们需要使用额外的库方法来解决这些问题（缓解竞争条件、同步对代码关键区域的访问等）。例如，标准 C++库提供了重载的原子方法来锁定、存储和比较共享指针所指向的底层数据。

我们已经看到了许多可以利用共享指针的示例程序。例如，我们利用了`Course`和`Student`类之间的关联——一个学生可以关联多个课程，一个课程也可以关联多个学生。显然，多个`Student`实例可以指向同一个`Course`实例，反之亦然。

在以前，使用原始指针时，程序员有责任使用引用计数。相比之下，使用共享指针时，内部引用计数器会原子性地增加和减少，以支持指针和线程安全。

解引用共享指针几乎与解引用原始指针一样快；然而，由于共享指针在类中代表了一个包装指针，因此构造和复制共享指针的成本更高。然而，我们感兴趣的是使 C++更安全，所以我们将简单地注意这个非常小的性能开销并继续前进。

让我们看看一个非常简单的使用`shared_ptr`的例子。这个例子可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex2.cpp)

```cpp
#include <iostream>
#include <memory>
#include "Person.h"
using std::cout;   // preferred to: using namespace std;
using std::endl;
using std::shared_ptr;
int main()
{
    shared_ptr<int> p1 = std::make_shared<int>(100);
    // alternative to preferred, previous line of code:
    // shared_ptr<int> p1(new int(100));
    shared_ptr<int> p2;// currently uninitialized (caution)
    p2 = p1; // p2 now shares the same memory as p1
    cout << *p1 << " " << *p2 << endl;
    shared_ptr<Person> pers1 = std::make_shared<Person>
                          ("Gabby", "Doone", 'A', "Miss");
    // alternative to preferred, previous lines of code:
    // shared_ptr<Person> pers1(new Person("Gabby",
    //                              "Doone",'A', "Miss"));
    shared_ptr<Person> pers2 = pers1;  // initialized
    pers1->Print();   // or use: (*pers1).Print();
    pers2->Print();   
    pers1->ModifyTitle("Dr."); // changes shared instance
    pers2->Print();   
    cout << "Number of references: " << pers1.use_count();
    return 0;
}
```

在上述程序中，我们创建了四个共享指针——两个指向相同的整数（`p1`和`p2`）和两个指向相同的`Person`实例（`pers1`和`pers2`）。由于我们使用的是共享指针（允许这种重新赋值），这些变量中的每一个都可能改变它们所指向的特定共享内存。例如，通过`pers1`对共享内存的更改，如果随后我们通过指针`pers2`查看（共享）内存，将会反映出来；这两个变量都指向相同的内存位置。

堆内存将再次由智能指针的使用自动管理。在这个例子中，当移除对内存的最后一个引用时，内存将被销毁和删除。请注意，引用计数是由我们代为进行的，并且我们可以使用`use_count()`来访问此信息。

让我们注意前一个示例中的一些有趣之处。注意 `shared_ptr` 变量 `pers1` 和 `pers2` 中 `->` 和 `.` 符号的混合使用。例如，我们使用 `pers1->Print();`，同时也使用 `pers1.use_count()`。这并非错误，而是揭示了智能指针的包装实现。考虑到这一点，我们知道 `use_count()` 是 `shared_ptr` 的一个方法。我们的共享指针 `pers1` 和 `pers2` 都被声明为 `shared_ptr` 的实例（绝对不是使用带有符号 `*` 的原始 C++ 指针）。因此，点符号是访问 `use_count()` 方法是合适的。然而，我们使用 `->` 符号来访问 `pers1->Print();`。在这里，回忆一下这个符号等同于 `(*pers1).Print();`。`shared_ptr` 类中的 `operator*` 和 `operator->` 都被重载，以便将智能指针中包含的包装原始指针委托出去。因此，我们可以使用标准指针符号来访问 `Person` 方法（通过安全包装的原始指针）。

这里是关于我们的 `shared_ptr` 指针示例的输出：

```cpp
100 100
Miss Gabby Doone
Miss Gabby Doone
Dr. Gabby Doone
Number of references: 2
Person destructor
```

共享指针似乎是一种确保多个指针指向的内存资源得到适当管理的好方法。总体来说，这是真的。然而，存在循环依赖的情况，共享指针根本无法释放其内存 – 另一个指针始终指向相关的内存。这发生在内存循环被遗弃时；也就是说，当没有外部共享指针指向循环连接时。在这些独特的情况下，我们实际上（并且反直觉地）可能会用共享指针管理内存。在这些情况下，我们可以从弱指针那里寻求帮助，以帮助我们打破循环。

考虑到这一点，接下来让我们看看弱智能指针。

## 使用智能指针 – 弱指针

在标准 C++ 库中，`weak_ptr` 是一种不拥有给定资源的智能指针类型；相反，弱指针充当观察者。弱指针可以用来帮助打破共享指针之间可能存在的循环连接；也就是说，在共享资源的销毁本应永远不会发生的情况下。在这里，一个弱指针被插入到链中，以打破共享指针单独可能创建的循环依赖。

例如，想象一下我们最初的编程示例中的 `Student` 和 `Course` 依赖关系，利用关联，或者从我们的更复杂的程序中，该程序展示了观察者模式。每个都包含关联对象类型的指针数据成员，从而有效地创建了一个潜在的循环依赖。现在，如果存在外部（来自圆圈外的）共享指针，例如外部课程列表或外部学生列表，那么可能不会出现排他性的循环依赖场景。在这种情况下，例如，课程的主列表（外部指针，与关联对象之间存在的任何循环依赖无关）将提供取消课程的方法，从而导致其最终被销毁。

同样，在我们的例子中，由大学学生群体组成的校外学生集合可以提供一个指向由 `Student` 和 `Course` 之间的关联产生的潜在循环共享指针场景的外部指针。然而，在这两种情况下，都需要做工作来从学生的课程列表中删除已取消的课程（或从课程的学生的列表中删除已退出的学生）。在这种情况下，删除关联反映了准确管理学生的日程安排或课程的出勤名单。尽管如此，我们可以想象存在循环连接的情景，但没有外部对链接的访问（与上述具有外部链接到圆圈中的情景不同）。

在存在循环依赖（没有外部影响）的情况下，我们需要将一个共享指针降级为弱指针。弱指针不会控制其所指向的资源的生命周期。

指向资源的弱指针不能直接访问该资源。这是因为 `weak_ptr` 类中没有重载操作符 `*` 和 `->`。您需要将弱指针转换为共享指针才能访问（包装的）指针类型的方法。一种方法是将 `lock()` 方法应用于弱指针，因为返回值是一个共享指针，其内容通过信号量锁定以确保对共享资源的互斥访问。

让我们通过一个使用 `weak_ptr` 的非常简单的例子来看看。这个例子可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex3.cpp)

```cpp
#include <iostream>
#include <memory>
#include "Person.h"
using std::cout;   // preferred to: using namespace std;
using std::endl;
using std::weak_ptr;
using std::shared_ptr;
int main()
{
    // construct the resource using a shared pointer
    shared_ptr<Person> pers1 = std::make_shared<Person>
                           ("Gabby", "Doone", 'A', "Miss");
    pers1->Print(); // or alternatively: (*pers1).Print();
    // Downgrade resource to a weak pointer
    weak_ptr<Person> wpers1(pers1); 
    // weak pointer cannot access the resource; 
    // must convert to a shared pointer to do so
    // wpers1->Print();   // not allowed! operator-> is not
                          // overloaded in weak_ptr class
    cout << "# references: " << pers1.use_count() << endl;
    cout << "# references: " << wpers1.use_count() << endl;
    // establish a new shared pointer to the resource
    shared_ptr<Person> pers2 = wpers1.lock();  
    pers2->Print();
    pers2->ModifyTitle("Dr.");   // modify the resource
    pers2->Print();
    cout << "# references: " << pers1.use_count() << endl;
    cout << "# references: " << wpers1.use_count() << endl;
    cout << "# references: " << pers2.use_count() << endl;
    return 0;
}
```

在上述程序中，我们使用 `pers1` 中的共享指针来分配我们的资源。现在，让我们假设在我们的程序中有理由将我们的资源降级为弱指针——也许我们想要插入一个弱指针来打破共享指针的循环。使用 `weak_ptr<Person> wpers1(pers1);`，我们为这个资源建立了一个弱指针。请注意，我们无法使用 `wpers1` 来调用 `Print();`。这是因为 `weak_ptr` 类中没有重载 `operator->` 和 `operator*`。

我们为 `pers1` 和 `wpers1` 打印出 `use_count()`，以注意到每个都显示了一个值为 `1`。这意味着只有一个非弱指针控制着相关的资源（弱指针可能暂时持有资源，但不能修改它）。

现在，假设我们想要按需将 `wpers1` 指向的资源转换为另一个共享指针，以便我们可以访问该资源。我们可以通过首先锁定弱指针来实现这一点；`lock()` 将返回一个共享指针，其内容由信号量保护。我们将这个值赋给 `pers2`。然后我们使用共享指针调用 `pers2->ModifyTitle("Dr.");` 来修改资源。

最后，我们从 `pers1`、`wpers1` 和 `pers2` 的角度打印出 `use_count()`。在每种情况下，引用计数都将为 `2`，因为有两个非弱指针引用了共享资源。弱指针不会对该资源的引用计数做出贡献，这正是弱指针可以帮助打破循环依赖的方式。通过在依赖循环中插入弱指针，共享资源的引用计数不会受到弱指针存在的影响。这种策略允许当只有对资源的弱指针剩余（且引用计数为 `0`）时删除资源。

堆内存将再次由智能指针的使用自动管理。在本例中，当移除对内存的最后一个引用时，内存将被销毁和删除。再次注意，弱指针没有对这个计数做出贡献。我们可以从 `Person` 析构函数中的 `cout` 语句中看到，只有一个实例被析构。

下面是关于我们的 `weak_ptr` 指针示例的输出：

```cpp
Miss Gabby Doone
# references: 1
# references: 1
Miss Gabby Doone
Dr. Gabby Doone
# references: 2
# references: 2
# references: 2
Person destructor
```

在本节中，我们回顾并补充了有关智能指针的基本知识。然而，每个类型的智能指针都可能单独占用一章内容。尽管如此，希望您对基本知识已经足够熟悉，可以开始在代码中包含各种智能指针，并在需要时进一步研究每种类型。

## 探索一个互补的想法——RAII

一种与智能指针（以及其他概念）相辅相成的编程惯用方法是 `move()` 操作。

许多 C++ 类库遵循 RAII（资源获取即初始化）进行资源管理，例如 `std::string` 和 `std::vector`。这些类遵循该惯用法，即它们的构造函数获取必要的资源（堆内存），并在析构函数中自动释放资源。使用这些类的用户不需要显式释放容器本身的任何内存。在这些类库中，即使不使用智能指针来管理堆内存，RAII 作为一种技术也被用来管理这些资源，其概念被封装并隐藏在类实现本身之中。

当我们在 *第二十章* 中实现自己的智能指针时，即 *使用 pImpl 模式去除实现细节*，我们无意中使用了 RAII 来确保在构造函数中分配堆资源，并在析构函数中释放资源。标准 C++ 库中实现的智能指针（`std::unique_ptr`、`std::shared_ptr` 和 `std::weak_ptr`）也采用了这种惯用法。通过使用采用这种惯用法的类（或者在自己无法做到时将其添加到类中），可以有助于确保代码更安全且易于维护。由于这种惯用法为代码增加了安全性和健壮性，熟练的开发者强烈建议我们将 RAII 作为 C++ 中最重要的实践和功能之一。

在我们努力使 C++ 更安全的努力中，接下来让我们考虑几个我们可以轻松采用的简单 C++ 功能，以确保我们的编码更加健壮。

# 采用促进安全性的额外 C++ 功能

如我们通过前 20 章编程所见，C++ 是一种广泛的语言。我们知道 C++ 有很大的能力，我们几乎可以在 C++ 中做任何事情。作为面向对象的 C++ 程序员，我们已经看到了如何采用 OO 设计，目标是使我们的代码更容易维护。

我们还积累了大量使用 C++ 中的原始（本地）指针的经验，主要是因为原始指针在现有代码中非常普遍。当需要时，你确实需要经验和熟练使用本地指针。在获得这种经验的过程中，我们亲眼目睹了可能遇到的堆内存管理陷阱——我们的程序可能崩溃，我们可能泄漏了内存，意外覆盖了内存，留下了悬垂指针，等等。在本章中，我们的首要任务是优先使用智能指针来创建新代码——以促进 C++ 的安全性。

现在，我们将探索 C++的其他领域，我们可以同样使用更安全的特性。我们在整本书中看到了这些不同的特性；建立一条指导原则，选择那些能促进 C++安全性的语言特性是很重要的。仅仅因为我们可以在 C++中做任何事情，并不意味着我们应该例行公事地将那些与高度误用相关的特性纳入我们的技能库。那些不断崩溃（或者甚至只崩溃一次）的应用程序是不可接受的。当然，我们在整本书中都提到了禁忌。在这里，让我们指出那些值得拥抱的语言特性，以进一步实现使 C++更安全的目标，使我们的应用程序更加健壮和易于维护。

让我们从回顾我们可以将其纳入日常代码中的简单项目开始。

## 重温范围 for 循环

C++有各种各样的循环结构，我们在整本书中都看到了。在处理一个完整的元素集合时，一个常见的错误是正确跟踪集合中有多少个项目，尤其是在这个计数器被用作遍历集合中所有项目的依据时。例如，当我们的集合以数组形式存储时，处理过多的元素可能会导致我们的代码不必要地抛出异常（或者更糟，可能导致程序崩溃）。

与其依赖于一个`MAX`值来遍历集合中的所有元素，不如以一种不依赖于程序员正确记住这个上限循环值的方式遍历集合中的每一个项目。相反，对于集合中的每一个项目，让我们进行某种处理。for-each 循环很好地满足了这一需求。

在处理一个不完整的元素集合时，一个常见的错误是正确跟踪当前集合中有多少个项目。例如，一门课程可能允许的最大学生人数是有限的。然而，截至今天，只有一半的潜在`Student`位置被填满。当我们查看课程中注册的学生名单时，我们需要确保我们只处理已填满的学生位置（即当前的学生人数）。处理所有最大学生位置显然是错误的，可能会导致我们的程序崩溃。在这种情况下，我们必须小心地只遍历`Course`中当前使用的`Student`位置，无论是通过在适当的时候退出循环的逻辑，还是通过选择一个当前大小代表要遍历的集合完整大小的容器类型（没有空白的*待填充*位置）；后一种情况使得 for-each 循环成为理想的选择。

此外，如果我们依赖于基于 `currentNumStudents` 计数器的循环呢？在之前示例中提到的情况下，这可能比 `MAX` 值更好，但如果我们没有正确更新那个计数器呢？我们也会在这个问题上出错。再次强调，将表示当前条目数量的容器类与 foreach 类型的循环结合起来，可以确保我们以更不易出错的方式处理完整的当前分组。

既然我们已经回顾了现代和更安全的循环风格，让我们拥抱 `auto` 以确保类型安全。然后我们将看到一个结合这些共同特性的示例。

## 使用 `auto` 进行类型安全

在许多情况下，使用 `auto` 可以使变量声明（包括循环迭代器）的编码更加容易，并且使用 `auto` 而不是显式类型化可以确保类型安全。

选择使用 `auto` 是声明具有复杂类型的变量的简单方法。使用 `auto` 还可以确保为给定变量选择最佳类型，并且不会发生隐式转换。

我们可以在各种情况下使用 `auto` 作为类型的占位符，让编译器推导出特定情况下的需求。我们甚至可以在许多情况下将 `auto` 用作函数的返回类型。使用 `auto` 可以使我们的代码看起来更通用，并且可以作为泛化的替代方案来补充模板。我们还可以将 `auto` 与 `const` 配对，并将这些限定符与引用配对；请注意，这些限定符 *结合* 不能与 `auto` 外推，必须由程序员单独指定。此外，`auto` 不能与增强类型的限定符一起使用，例如 `long` 或 `short`，也不能与 `volatile` 一起使用。虽然这超出了我们书籍的范围，但 `auto` 可以与 lambda 表达式一起使用。

当然，使用 `auto` 有一些缺点。例如，如果程序员不理解正在创建的对象的类型，程序员可能会期望编译器选择某种类型，然而却推导出另一种（意外的）类型。这可能会在您的代码中产生微妙的错误。例如，如果您为 `auto` 将选择和编译器实际推导出的 `auto` 声明类型都重载了函数，您可能会调用一个与预期不同的函数！当然，这大多可能是由于程序员在插入 `auto` 关键字时没有完全理解当前使用上下文的结果。另一个缺点是，当程序员仅仅为了强制代码编译而使用 `auto`，而没有真正地处理手头的语法并思考代码应该如何编写时。

既然我们已经回顾了在我们的代码中添加 `auto`，那么让我们回顾一下在日常代码中拥抱 STL。然后我们将看到一个结合这些共同特性的示例。

## 优先使用 STL 进行简单容器

如我们在*第十四章*“理解 STL 基础”所见，标准模板库（STL）包含了一套非常完整且健壮的容器类，这些类在 C++代码中被广泛使用。使用这些经过良好测试的组件（而不是原生 C++机制，如指针数组）来收集类似项，可以为我们的代码增加鲁棒性和可靠性。内存管理变得更容易（消除了许多潜在的错误）。

通过使用模板实现其大量的容器类，STL 允许其容器以通用方式用于程序可能遇到的任何数据类型。相比之下，如果我们使用了原生 C++机制，我们可能会将我们的实现绑定到特定的类类型，例如指向`Student`的指针数组。当然，我们可以实现一个指向模板化类型的指针数组，但为什么要在有这么多经过良好测试且易于使用的容器可供我们使用时这样做呢？

STL 容器还避免了使用`new()`和`delete()`进行内存管理，而是选择使用分配器来提高 STL 底层内存管理的效率。例如，一个向量、栈或队列可能会增长或缩小。而不是分配你可能预期的最大元素数量（这可能既难以估计，又可能对典型使用（通常不会达到最大值）来说既困难又低效），在幕后可能会预先分配一定大小的缓冲区或元素数量。这种初始分配允许在不需要为集合中的每个新添加项进行大小调整的情况下多次向容器中添加元素（否则可能会这样做以避免过度分配）。只有当底层容器的内部分配（或缓冲区）大小超过预先分配的量时，才需要进行内部重新分配（对容器用户来说是未知的）。内部重新分配或*移动*的代价是分配更大的内存块，从原始内存复制到更大的内存块，然后释放原始内存。STL 在幕后努力微调内部分配，以平衡典型使用需求与可能进行的昂贵重新分配。

既然我们已经重新审视了在代码中优先使用 STL，那么现在让我们重新审视在必要时使用`const`，以确保代码不会被修改，除非我们有意使其如此。我们将通过一个示例结束本节，该示例展示了本节中所有关键的安全点。

## 根据需要使用 const

将`const`限定符应用于对象是一种简单的方法，可以表明不应修改的实例实际上没有被修改。我们可能会记得，`const`实例只能调用`const`成员函数。而且，`const`成员函数不能修改调用该方法的任何对象的部分（`this`）。记住利用这个简单的限定符可以确保这个检查点链对于我们不打算修改的对象发生作用。

在此基础上，请记住，`const`可以在参数列表中使用，以指定对象和方法。使用`const`不仅增加了它所指定的对象和方法的可读性，还增加了宝贵的只读对象和方法强制执行。让我们记住在需要时使用`const`！

现在，让我们看看我们如何使用这些易于添加的 C++特性，这些特性有助于更安全的编程。这个例子重新审视了首选的循环风格，使用`auto`进行类型安全，使用 STL 进行简单的容器，以及适当地应用`const`。这个例子可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex4.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex4.cpp)

```cpp
#include <vector>  
using std::vector;  
// Assume additional #include/using as typically included
// Assume classes Person, Student are as typically defined
// In this const member function, no part of 'this' will
// be modified. Student::Print() can be called by const
// instances of Student, including const iterators 
void Student::Print() const
{   // need to use access functions as these data members
    // are defined in Person as private
    cout << GetTitle() << " " << GetFirstName() << " ";
    cout << GetMiddleInitial() << ". " << GetLastName();
    cout << " with id: " << studentId << " GPA: ";
    cout << setprecision(3) <<  " " << gpa;
    cout << " Course: " << currentCourse << endl;
}
int main()
{   // Utilize STL::vector instead of more native C++ data
    // structures (such as an array of pointers to Student)
    // There's less chance for us to make an error with
    // memory allocation, deallocation, deep copies, etc.
    vector<Student> studentBody;  
    studentBody.push_back(Student("Hana", "Sato", 'U', 
                           "Miss", 3.8, "C++", "178PSU"));
    studentBody.push_back(Student("Sam", "Kato", 'B', 
                           "Mr.", 3.5, "C++", "272PSU"));
    studentBody.push_back(Student("Giselle", "LeBrun", 'R',
                           "Ms.", 3.4, "C++", "299TU"));
    // Notice that our first loop uses traditional notation
    // to loop through each element of the vector.
    // Compare this loop to next loop using an iterator and
    // also to the preferred range-for loop further beyond
    // Note: had we used MAX instead of studentBody.size(),
    // we'd have a potential error – what if MAX isn't the
    // same as studentBody.size()? 
    for (int i = 0; i < studentBody.size(); i++)   
        studentBody1[i].Print();  
    // Notice auto keyword simplifies iterator declaration
    // However, an iterator is still not the most
    // preferred looping mechanism. 
    // Note, iterator type is: vector<Student>::iterator
    // the use of auto replaces this type, simplifying as: 
    for (auto iter = studentBody.begin(); 
              iter != studentBody.end(); iter++)
        (*iter).EarnPhD();
    // Preferred range-for loop 
    // Uses auto to simplify type and const to ensure no
    // modification. As a const iterator, student may only
    // call const member fns on the set it iterates thru
    for (const auto &student : studentBody)
        student.Print();
    return 0;
}
```

在上述程序中，我们最初注意到我们使用了 C++ STL 中的`std::vector`。在`main()`函数中，我们注意到使用`vector<Student> studentBody;`实例化了一个向量。利用这个经过良好测试的容器类无疑增加了我们代码的健壮性，相对于我们自行管理动态大小的数组。

接下来，注意指定了一个常量成员函数`void Student::Print() const;`。在这里，`const`指定确保调用此方法的对象（`this`）的任何部分都不能被修改。此外，如果存在任何`const`实例，它们将能够调用`Student::Print()`，因为`const`指定保证了此方法对`const`实例来说是安全的（即只读）。

接下来，我们注意到三种循环风格和机制，从最不安全到最安全的风格进行排序。第一个循环使用传统的`for`循环遍历循环中的每个元素。如果我们用`MAX`代替`studentBody.size()`作为循环条件会怎样？我们可能会尝试处理比容器中当前元素更多的元素；这种疏忽可能导致错误。

第二个循环使用迭代器和 `auto` 关键字来简化类型指定（从而对迭代器本身来说更容易且更安全）。虽然迭代器定义良好，但它们仍然不是首选的循环机制。`for` 语句中第二个语句的增量中的一个细微差别也可能导致效率低下。例如，考虑在循环条件重新测试之前执行的语句中的预增量与后增量。如果这是 `iter++`，则代码效率会较低。这是因为 `iter` 是一个对象，预增量返回对象的引用，而后增量返回一个临时对象（在每个循环迭代中创建和销毁）。后增量还使用了一个重载函数，因此编译器无法优化其使用。

最后，我们看到首选且最安全的循环机制，它结合了范围-for 循环和 `auto` 用于迭代器指定（以简化类型声明）。使用 `auto` 替换了 `vector<Student>::iterator` 作为 `iter` 的类型。任何简化符号的地方，都有更少的错误空间。此外，请注意迭代器声明中添加了 `const`，以确保循环将只调用每个迭代实例上的不可修改方法；这是一个我们可以在我们代码中采用的额外、适当的特性示例。

以下是上述程序的输出：

```cpp
Miss Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Mr. Sam B. Kato with id: 272PSU GPA:  3.5 Course: C++
Ms. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
Everyone to earn a PhD
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sam B. Kato with id: 272PSU GPA:  3.5 Course: C++
Dr. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
```

我们现在回顾了几个简单的 C++ 语言特性，这些特性可以轻松地被采用以促进我们日常编码实践中的安全性。使用范围-for 循环提供了代码简化并消除了对循环迭代中经常错误的上限的依赖。采用 `auto` 简化了变量声明，包括循环迭代器内的声明，并有助于确保类型安全与显式类型。使用经过良好测试的 STL 组件可以为我们的代码增加鲁棒性、可靠性和熟悉感。最后，将 `const` 应用于数据和方法是确保数据不会被意外修改的一种简单方法。这些原则都很容易应用，并通过增加整体安全性为我们的代码增加价值。

接下来，让我们考虑理解线程安全性如何有助于使 C++ 更加安全。

# 考虑线程安全性

C++ 的多线程编程本身就是一个完整的书籍。尽管如此，我们在本书中提到了几个可能需要考虑线程安全性的情况。值得重申这些主题，以提供一个概述，说明您可能在 C++ 编程的各个细分领域中遇到的问题。

一个程序可能由多个线程组成，每个线程可能都可能会相互竞争以访问共享资源。例如，共享资源可能是一个文件、套接字、共享内存区域或输出缓冲区。每个访问共享资源的线程都需要对资源进行仔细协调（称为互斥）的访问。

例如，想象如果有两个线程想要向你的屏幕写入输出。如果每个线程都可以访问与`cout`关联的输出缓冲区，而不必等待另一个线程完成一个连贯的语句，输出将是一团糟的随机字母和符号。显然，对共享资源的同步访问是非常重要的！

线程安全涉及理解原子操作、互斥、锁、同步等——这些都是多线程编程的方面。

让我们从线程和多线程编程的概述开始。

## 多线程编程概述

**线程**是在一个进程内部的一个独立的控制流，从概念上讲，它就像是在给定进程内部的一个子进程（或进程的进一步细分）。线程有时被称为**控制线程**。拥有许多控制线程的应用程序被称为**多线程应用程序**。

在单处理器环境中，线程给人一种多个任务同时运行的感觉。就像进程一样，线程在 CPU 之间快速切换，以便用户看起来它们正在被同时处理（尽管实际上不是）。在共享的多处理器环境中，应用程序中使用线程可以显著加快处理速度，并允许实现并行计算。即使在单处理器系统中，线程实际上（也许出人意料地）可以加快一个进程的速度，因为在等待另一个线程的 I/O 完成时，一个线程可以运行。

执行相同任务的线程可能会同时处于类的类似方法中。如果每个线程都在处理一个不同的数据集（例如，一个不同的`this`指针，即使是在同一个方法中工作），通常没有必要同步对这些方法的访问。例如，想象`s1.EarnPhd();`和`s2.EarnPhD();`。在这里，两个独立的实例处于同一个方法中（可能是并发地）。然而，每个方法中处理的数据集是不同的——在第一种情况下，`s1`将绑定到`this`；在第二种情况下，`s2`将绑定到`this`。这两个实例之间共享的数据很可能没有重叠。然而，如果这些方法正在访问静态数据（即给定类所有实例共享的数据，例如`numStudents`数据成员），那么对访问共享内存区域的代码的关键部分进行同步将是必需的。传统上，在需要互斥访问代码关键区域的 数据或函数周围添加系统依赖的锁或信号量。

C++中的多线程编程可以通过各种商业或公共领域的多线程库来实现。此外，标准 C++库在多种能力上提供了线程支持，包括使用 `std::condition_variable` 进行线程同步，`std::mutex` 确保关键资源的互斥性（通过避免竞争条件），以及 `std::semaphore` 来模拟资源计数。通过实例化 `std::thread` 对象并熟练掌握上述功能，我们可以使用已建立的 C++库添加多线程编程。此外，可以将 `std::atomic` 模板添加到类型中，将其建立为原子类型并确保类型安全的同步。`std::exception_ptr` 类型允许在协调线程之间传输异常。总的来说，有许多线程库功能需要考虑；这是一个广泛的话题。

多线程编程的细节超出了本书的范围；然而，我们可以讨论本书中可能需要增加以要求使用线程知识的场景。让我们回顾一些那些情况。

## 多线程编程场景

有许多编程场景可以从使用多线程编程中受益。我们只提及一些扩展了本书中涵盖的思想的例子。

观察者模式当然可以在多线程编程场景中使用！在这些情况下，必须小心处理 `Observer` 和 `Subject` 的 `Update()` 和 `Notify()` 方法，以添加同步和锁定机制。

智能指针，例如 `shared_ptr` 和 `weak_ptr`，可以在多线程应用程序中使用，并且已经包含了通过引用计数（以及使用原子库方法）来锁定和同步访问共享资源的手段。

通过关联相关的对象可能会在多线程编程或通过共享内存区域中出现。任何通过多线程编程使用共享资源进行访问的时候，都应该使用互斥锁（锁）来确保对这些共享资源的互斥访问。

抛出异常的对象需要相互通信，将需要在捕获块中包含同步或将异常委派给 `main()` 程序线程。使用工作线程与 `main()` 程序线程通信是典型设计模式。利用共享内存是存储需要在抛出和捕获异常本身之间协调的线程之间共享的数据的手段。可以使用 `std::exception_ptr` 实例与 `std::current_exception()` 一起使用来存储需要共享的实例。这个共享实例（在线程之间）可以使用 `std::rethrow_exception()` 重新抛给参与线程。

多线程编程本身就是一个迷人的主题，并且需要在 C++ 中安全地使用它之前进行深入理解。我们已经回顾了一些可能补充本书所涵盖内容的线程安全性考虑的区域。强烈建议在向代码中添加多线程编程之前，深入探讨 C++ 中的线程安全性。

接下来，让我们进一步探讨编程指南如何为 C++ 编程增加必要的安全性级别。

# 利用核心编程指南

编程指南远不止是一套约定，比如指示缩进多少空格或变量的命名规范，函数、类、数据成员和成员函数的命名约定。现代编程指南是组织内部程序员之间的一种契约，旨在创建遵循特定标准的代码，其最大目标是通过对这些共同标准的遵循，提供健壮且易于扩展的代码。简而言之，编程指南中包含的大多数约定都是为了使 C++ 编程更安全。

关于构成 C++ 编程指南的内容，各组织之间可能存在共识差异，但有许多资源可用（包括来自标准委员会的资源）来提供示例和指导。

让我们继续探讨编程指南的基本要素的抽样，然后讨论采用核心指南集，以及理解广泛可用的在 C++ 中安全编程的资源。

## 检查指南要素

让我们从检查一个典型的 C++ 编程指南中可以遵循的有意义的约定开始。我们在整本书中已经探讨了这些编程问题中的许多，但回顾一些项目对于选择促进 C++ 安全性的约定是有用的。

### 优先初始化而非赋值

在可能的情况下，始终选择初始化而非赋值。这既更高效也更安全！使用类内初始化或成员初始化列表。在初始化之后使用赋值可能效率较低。例如，想象一个默认构造的成员对象，它只是快速通过构造函数体内的赋值来覆盖其值。利用成员初始化列表通过另一个构造函数初始化这个成员对象会更有效率。

此外，未能为每一块内存赋予初始值可能会在安全性方面给我们带来巨大的代价——在 C++ 中，内存不是干净的，因此将未初始化变量（或数据成员）中的任何内容解释为有效内容是完全不恰当的。访问未初始化的值是未定义的行为。我们真的不知道未初始化的内存中隐藏着什么，但我们知道它绝不是用作初始化器的正确值！

让我们通过一个小程序来回顾首选的初始化。这个例子可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex5.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex5.cpp)

```cpp
class Person
{
private: 
   string firstName; // str mbrs are default constructed so
   string lastName;  // we don't need in-class initializers
   char middleInitial = '\0';  // in-class initialization
   string title;  
protected: 
   void ModifyTitle(const string &); 
public:
   Person() = default;   // default constructor
   Person(const string &, const string &, char, 
          const string &);  
   // use default copy constructor and default destructor
   // inline function definitions
   const string &GetFirstName() const { return firstName; }
   const string &GetLastName() const { return lastName; }
   const string &GetTitle() const { return title; } 
   char GetMiddleInitial() const { return middleInitial; }
};
// With in-class initialization, it often not necessary to
// write the default constructor yourself – there's often
// nothing remaining to initialize!
// alternate constructor
// Note use of member init list to initialize data members
Person::Person(const string &fn, const string &ln, char mi,
               const string &t): firstName(fn),
               lastName(ln), middleInitial(mi), title(t)
{
    // no need to assign values in body of method –
    // initialization has handled everything!
}
```

检查前面的代码，我们发现`Person`类使用类内初始化将`middleInitial`数据成员设置为空字符（`'\0'`）。对于`Person`的每个实例，`middleInitial`将在调用任何进一步初始化该实例的构造函数之前被设置为空字符。请注意，类中的其他数据成员都是`string`类型。因为`string`本身就是一个类，这些数据成员实际上是`string`类型的成员对象，并将被默认构造，适当地初始化这些字符串成员。

接下来，请注意我们没有提供默认（无参数）构造函数，允许系统提供的默认构造函数为我们链接。类内初始化，加上适当的`string`成员对象初始化，使得对于新的`Person`实例没有额外的初始化工作，因此不需要程序员指定的默认构造函数。

最后，请注意我们在`Person`类的替代构造函数中使用了成员初始化列表。在这里，每个数据成员都使用此方法参数列表中的适当值进行设置。请注意，每个数据成员都是通过初始化设置的，这样在替代构造函数的主体中就不需要任何赋值操作了。

我们前面的代码遵循了流行的代码规范：在可能的情况下，始终选择通过初始化而不是赋值来设置值。知道每个数据成员在构造过程中都有适当的值，这使我们能够提供更安全的代码。初始化也比赋值更高效。

现在，让我们考虑另一个与虚函数相关的核心 C++指南。

### 选择`virtual`、`override`或`final`中的一个

多态是一个美妙的概念，C++通过使用虚函数轻松支持。我们在*第七章*，*通过多态利用动态绑定*中了解到，关键字`virtual`用于指示多态操作——一个可能被派生类用首选方法覆盖的操作。派生类没有义务通过提供新方法来覆盖多态操作（虚函数），但可能会发现这样做是有意义的。

当派生类选择用新方法覆盖基类引入的虚函数时，被覆盖的方法可以在方法的签名中使用`virtual`和`override`关键字。然而，在这个被覆盖的（派生类）级别，只使用`override`是一个约定。

当在层次结构中引入虚函数时，在某个时候可能希望表明某个方法是此操作的*最终*实现。也就是说，所涉及的操作可能不再被覆盖。我们知道，在层次结构的这个级别上应用`final`说明符是合适的，以表明给定的方法可能不再被覆盖。尽管我们也可以在这个级别包含关键字`virtual`，但建议只使用`final`。

总结一下，在指定虚函数时，每个级别只选择一个标签：`virtual`、`override`或`final`——即使关键字`virtual`可以添加以补充`override`和`final`。这样做可以使当前虚函数是新生成的（`virtual`）、是虚函数的覆盖方法（`override`），还是虚函数的最终方法（`final`）更加清晰。清晰性导致错误发生得少，这有助于使 C++更安全。

让我们通过一个程序段来回顾使用虚函数时的首选关键字用法。完整的示例可以在我们的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex6.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter21/Chp21-Ex6.cpp)

```cpp
class Person
{
private: 
    string firstName;
    string lastName;
    char middleInitial = '\0';  // in-class initialization
    string title;  // Mr., Ms., Mrs., Miss, Dr., etc.
protected:
    void ModifyTitle(const string &); 
public:
    Person() = default;   // default constructor
    Person(const string &, const string &, char, 
           const string &); 
    virtual ~Person();  // virtual destructor
    const string &GetFirstName() const 
        { return firstName; } 
    const string &GetLastName() const { return lastName; }
    const string &GetTitle() const { return title; } 
    char GetMiddleInitial() const { return middleInitial; }
    virtual void Print() const; // polymorphic operations
    virtual void IsA() const;   // introduced at this level
    virtual void Greeting(const string &) const;
};
// Assume the non-inline member functions for Person 
// follow and are as we are accustomed to seeing
class Student: public Person
{
private: 
    float gpa = 0.0;   // in-class initialization
    string currentCourse;
    const string studentId; 
    static int numStudents;  // static data member
public:
    Student();  // default constructor
    Student(const string &, const string &, char, 
            const string &, float, const string &, 
            const string &); 
    Student(const Student &);  // copy constructor
    ~Student() override;  // virtual destructor
    void EarnPhD();  
    // inline function definitions
    float GetGpa() const { return gpa; }
    const string &GetCurrentCourse() const
        { return currentCourse; }
    const string &GetStudentId() const 
        { return studentId; }
    void SetCurrentCourse(const string &); // proto. only

    // In the derived class, keyword virtual is optional, 
    // and not currently recommended. Use override instead.
    void Print() const final; // override is optional here
    void IsA() const override;
    // note, we choose not to redefine (override):
    // Person::Greeting(const string &) const
    static int GetNumberStudents(); // static mbr. function
};
// definition for static data member 
int Student::numStudents = 0;  // notice initial value of 0
// Assume the non-inline, non-static member functions for
// Students follow and are as we are accustomed to seeing
```

在前面的示例中，我们看到我们一直在书中使用的`Person`类。作为一个基类，请注意，`Person`指定了多态操作`Print()`、`IsA()`和`Greeting()`，以及使用`virtual`关键字的析构函数。这些操作旨在由派生类使用更合适的方法覆盖（不包括析构函数），但如果派生类认为基类的实现是合适的，则不需要覆盖。

在派生类`Student`中，我们使用更合适的方法覆盖了`IsA()`。请注意，我们在该函数的签名中使用了`override`，尽管我们也可以包含`virtual`。接下来，请注意，我们没有在`Student`级别覆盖`Greeting()`；我们可以假设`Student`认为`Person`中的实现是可以接受的。另外，请注意，析构函数被覆盖以提供销毁链的入口点。回想一下，析构函数不仅会调用派生类的析构函数，还会调用基类的析构函数（隐式地作为派生类析构函数中的最后一行代码），从而确保对象的完整销毁序列能够正确开始。

最后，请注意，在`Student`类中，`Print()`函数已被重写为`final`。尽管我们也可以将`override`关键字添加到这个函数的签名中，但我们选择根据推荐的编码规范仅使用`final`。

现在，让我们看看典型 C++编程指南中的另一个典型元素，与智能指针相关。

### 在新代码中优先使用智能指针

我们在这本书中使用了许多本地的（原始的）C++指针，因为你无疑会被要求沉浸到包含大量指针的现有代码中。拥有本地指针的经验和能力，将使你在被要求进入使用本地指针的情况时成为一个更安全的程序员。

然而，出于安全考虑，大多数编程指南都会建议在新建代码中仅使用智能指针。毕竟，它们的使用开销很小，可以帮助消除程序员管理堆内存的许多潜在陷阱。智能指针还有助于异常安全性。例如，异常处理意味着代码的预期流程可能在几乎任何时间被中断，导致使用传统指针时可能发生内存泄漏。智能指针可以减轻一些这种负担，并提供异常安全性。

在原始代码中使用智能指针非常重要，这一点值得重复：在 C++中选择智能指针而不是本地指针，将导致更安全且更容易维护的代码。代码也将更容易编写，消除了许多析构函数的需求，自动阻止不希望的复制和赋值（`unique_ptr`）等。考虑到这一点，在可能的情况下，始终选择智能指针在新创建的代码中。

我们在这本书中也看到了智能指针和本地指针。现在，你可以选择在你创建的新代码中使用智能指针——这强烈推荐。当然，可能有一些情况下这是不可能的；也许你正在创建与现有本地指针代码高度交互的新代码，需要利用相同的数据结构。尽管如此，在可能的情况下，你可以努力使用智能指针，同时你拥有灵活性和经验来理解大量使用本地指针的现有代码、库和在线示例。

对于安全性来说，还有什么比在你的原始代码中拥有智能指针的能力，并辅以本地指针的知识，只在必要时使用更好呢？

有许多编程指南的例子可以轻松遵循，以使你的代码更安全。上述例子只是许多例子中的一部分，用以说明你将在一组基本 C++编程指南中看到的各种实践。

现在，让我们考虑如何组装或采用核心编程指南，以帮助使我们的代码更安全。

## 采用编程指南

无论你是自己构建或组装一套编程指南，还是遵循你所在组织管理的一套指南，采用一组核心 C++编程指南对于确保你的代码尽可能安全、健壮至关重要，这转化为更容易维护的代码。

指南应始终随着语言的发展而保持灵活。接下来，让我们考虑寻找核心 C++ 编程指南的资源，以便直接遵循或逐步回顾，以改进你所在组织接受的指南。

## 在 C++ 中安全编程的资源

在线有许多关于 C++ 编程指南的资源。然而，最重要的资源是 *ISO C++ 核心指南*，主要由 Bjarne Stroustrup 和 Herb Sutter 组装，可以在以下 GitHub 网址找到：[`github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md`](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)。他们的共同目标是帮助程序员安全且更有效地使用现代 C++。

选取的市场领域可能需要遵循一定的指南来获得或确保行业内的认证。例如，**MISRA** 是一套针对 **汽车工业软件可靠性协会**（Motor Industry Software Reliability Association）的 C++ 编码标准；MISRA 也已被其他行业采纳为标准，例如医疗系统。另一个为嵌入式系统开发的编码标准是 **CERT**，由 **卡内基梅隆大学**（Carnegie Mellon University，简称 **CMU**）开发。CERT 最初是 **计算机紧急响应团队**（Computer Emergency Response Team）的缩写，现在已成为 CMU 的注册商标。CERT 也被许多金融行业采纳。**JSF AV C++**（Joint Strike Fighter Air Vehicle C++）是洛克希德·马丁公司（Lockheed Martin）开发的一种 C++ 编码标准，用于航空航天工程领域，以确保安全关键系统的代码无错误。

毫无疑问，你加入的每个组织作为贡献者都将有一套编程指南，供组内所有程序员遵循。如果没有，明智的做法是建议采用一套核心的 C++ 编程指南。毕竟，你需要帮助维护自己的代码以及同事的代码；一套统一和预期的标准将使所有相关人员都能管理这项工作。

# 摘要

在本附录中，我们通过理解在 C++ 中安全编程的重要性，增加了成为不可或缺的 C++ 程序员的目标。毕竟，我们的主要目标是创建健壮且易于维护的代码。采用安全的编程实践将帮助我们实现这一目标。

我们回顾了书中提到的概念，以及相关思想，最终形成了一套核心编程指南，以确保更安全的编码实践。

首先，我们回顾了智能指针，检查了来自标准 C++ 库的三种类型，即 `unique_ptr`、`shared_ptr` 和 `weak_ptr`。我们了解到，这些类通过提供封装来分配和释放堆内存，从而通过我们在经过充分测试的标准库类中的行为来安全地实现 RAII 习语。我们提出了一个指南：在新建代码中始终优先考虑智能指针。

接下来，我们重申了在本书中看到的多种编程实践，我们可以利用这些实践来使我们的编码更加安全。例如，优先使用 for-each 风格的循环和 `auto` 关键字来保证类型安全。此外，使用 STL 容器而非较脆弱的本地机制，并在需要时为数据和方法添加 `const` 限定符以确保只读访问。这些实践（在许多实践中）可以帮助确保我们的代码尽可能安全。

接下来，我们介绍了 C++ 的多线程编程，并回顾了之前我们看到的可能从使用线程中受益的编程场景。我们还前瞻性地查看了一下标准 C++ 库中支持多线程编程的类，包括那些提供同步、互斥锁、信号量和创建原子类型的类。

最后，我们探讨了编程指南的要点，以便更好地理解在 C++ 核心编程指南中可能有益的规则。例如，我们回顾了优先初始化而非赋值，关于 `virtual`、`override` 和 `final` 关键字的虚拟函数使用，以及本章之前探讨的主题。我们讨论了采用一套全面的 C++ 核心编程指南的重要性，以及查找作为行业标准使用的示例指南的资源。

在应用本书中涵盖的许多特性时，了解如何使 C++ 更加安全无疑会使你成为一个更有价值的程序员。你现在拥有了核心语言技能，并且对 C++ 中的面向对象编程（基本概念及其在 C++ 中的实现方式，无论是直接语言支持还是使用编程技术）有了非常坚实的理解。我们通过异常处理、友元、运算符重载、模板、STL 基础和测试 OO 类和组件等知识丰富了你的技能。我们还接受了核心设计模式，通过综合编程示例深入研究每个模式。最后在本章中，我们回顾了如何通过在每个可用的机会选择采用更安全的编程实践来安全地组合你所学的知识。

当我们一起结束我们的附加章节时，你现在准备好独自踏上旅程，将 C++ 应用于许多新的和现有的应用。你准备好创建安全、健壮且易于维护的代码。我真诚地希望你对 C++ 的兴趣和我一样浓厚。再次，让我们开始编程吧！

# 评估

每章问题的编程解决方案可以在我们的 GitHub 仓库中找到，网址如下：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main)。每个完整的程序解决方案都可以在我们的 GitHub 仓库的“评估”子目录中找到，然后在适当的章节标题（子目录，例如`Chapter01`）下，在一个与章节编号相对应的文件中，后面跟着一个连字符，然后是当前章节中的解决方案编号。例如，*第一章*，“理解基本的 C++假设”中的*问题 3*的解决方案可以在上述 GitHub 目录下的`Assessments/Chapter01`子目录中的`Chp1-Q3.cpp`文件中找到。

非编程问题的书面回答可以在以下章节中找到，按章节组织，以及在上文提到的 GitHub 中相应章节的“评估”子目录中。例如，`Assessments/Chapter01/Chp1-WrittenQs.pdf`将包含对*第一章*，“理解基本的 C++假设”的非编程解决方案的答案。如果一个练习既有编程部分又有对程序的后续问题，那么后续问题的答案可以在下一节（以及上文提到的`.pdf`文件）中找到，也可以在 GitHub 中编程解决方案顶部的注释中找到（因为可能需要审查解决方案以完全理解后续问题的答案）。

# 第一章，理解基本的 C++假设

1.  在不希望光标移动到下一行进行输出的情况下，`flush`可能比`endl`更有用，用于清除与`cout`关联的缓冲区的内容。回想一下，`endl`操作符只是一个换行符加上缓冲区刷新。

1.  对于变量选择前置递增（`++i`）还是后置递增（`i++`），当与复合表达式一起使用时，会对代码产生影响。一个典型的例子是`result = array[i++];`与`result = array[++i];`。在后置递增（`i++`）的情况下，`array[i]`的内容将被分配给`result`，然后`i`递增。在前置递增的情况下，`i`首先递增，然后`result`将具有`array[i]`的值（即使用`i`的新值作为索引）。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter01/Chp1-Q3.cpp`。

# 第二章，添加语言必要性

1.  函数的签名是函数名加上其类型和参数数量（没有返回类型）。这与名称混淆有关，因为签名帮助编译器为每个函数提供一个唯一的内部名称。例如，`void Print(int, float);` 可能具有混淆后的名称 `Print_int_float();`。这通过为每个函数提供一个唯一的名称来简化重载函数，使得在调用时，可以通过内部函数名称明显地知道正在调用哪个函数。

1.  a – d. 请参阅 GitHub 仓库中的 `Assessments/Chapter02/Chp2-Q2.cpp`。

# 第三章，间接寻址：指针

1.  a – f. 请参阅 GitHub 仓库中的 `Assessments/Chapter03/Chp3-Q1.cpp`。

d. （后续问题）`Print(Student)` 比重载版本 `Print(const Student *)` 效率低，因为该函数的初始版本在栈上传递整个对象，而重载版本只在栈上传递指针。

1.  假设我们有一个指向类型为 `Student` 的对象的现有指针，例如：`Student *s0 = new Student`; （这个 `Student` 尚未用数据初始化）

a. `const Student *s1;` （不需要初始化）

b. `Student *const s2 = s0;` （需要初始化）

c. `const Student *const s3 = s0;` （也需要初始化）

1.  将类型为 `const Student *` 的参数传递给 `Print()` 允许将指向 `Student` 的指针传递给 `Print()` 以提高速度，但指向的对象不能被解引用和修改。然而，将 `Student * const` 作为 `Print()` 的参数传递是没有意义的，因为指针的副本将被传递给 `Print()`。将这个副本标记为 `const`（意味着不允许改变指针指向）将没有意义，因为不允许改变指针副本对原始指针本身没有影响。原始指针在函数内部地址被改变的风险中从未处于危险之中。

1.  在许多编程场景中可能会使用动态分配的 3-D 数组。例如，如果图像存储在 2-D 数组中，一组图像可能存储在 3-D 数组中。拥有动态分配的 3-D 数组允许从文件系统中读取任意数量的图像并将其存储在内部。当然，在分配 3-D 数组之前，你需要知道将要读取多少图像。例如，一个 3-D 数组可能包含 30 张图像，其中 30 是第三维，用于收集一组图像。为了概念化一个 4-D 数组，你可能希望组织上述 3-D 数组的集合。

例如，也许你有一组 31 张 1 月份的图片。这组 1 月份的图片是一个三维数组（二维用于图片，第三维用于组成 1 月份的 31 张图片的集合）。你可能希望对每个月都做同样的事情。而不是为每个月的图片集合分别设置单独的三维数组变量，我们可以创建一个第四维来收集一年的数据到一个集合中。第四维将包含一年的 12 个月份中的一个元素。那么五维数组呢？你可以通过将第五维作为收集不同年份数据的方式扩展这个图像概念，例如收集一个世纪的图片（第五维）。现在我们有按世纪组织的图片，然后按年、按月、按图片（需要前两个维度）组织。

# 第四章，间接寻址：引用

1.  a – c. 请参阅 GitHub 仓库中的`Assessments/Chapter04/Chp4-Q1.cpp`。

c. （后续问题）指针变量不仅需要调用接受`Student`指针的`ReadData(Student *)`版本，引用变量也不需要仅调用接受`Student`引用的`ReadData(Student &)`版本。例如，指针变量可以用`*`解引用然后调用接受引用的版本。同样，引用变量可以用`&`取其地址然后调用接受指针的版本（尽管这不太常见）。你只需确保数据类型与你要传递的和函数期望的类型相匹配。

# 第五章，详细探索类

1.  a – e. 请参阅 GitHub 仓库中的`Assessments/Chapter05/Chp5-Q1.cpp`。

# 第六章，使用单继承实现层次结构

1.  a – d. 请参阅 GitHub 仓库中的`Assessments/Chapter06/Chp6-Q1.cpp`。

1.  a – c. （可选）请参阅 GitHub 仓库中的`Chapter06/Assessments/Chp6-Q2.cpp`。

# 第七章，利用多态实现动态绑定

1.  a – e. 请参阅 GitHub 仓库中的`Assessments/Chapter07/Chp7-Q1.cpp`。

# 第八章，精通抽象类

1.  a – d. 请参阅 GitHub 仓库中的`Assessments/Chapter08/Chp8-Q1.cpp`。

e. 根据你的实现，你的`Shape`类可能被视为接口类，也可能不是。如果你的实现是一个不包含数据成员且只包含抽象方法（纯虚函数）的抽象类，那么你的`Shape`实现被视为接口类。然而，如果你的`Shape`类在派生类中通过重写的`Area()`方法计算了`area`后将其作为数据成员存储，那么它就只是一个抽象基类。

# 第九章，探索多重继承

1.  请参阅 GitHub 仓库中的`Assessments/Chapter09/Chp9-Q1.cpp`。

a. 有一个`LifeForm`子对象。

b. `LifeForm`构造函数和析构函数各被调用一次。

c. 如果从`Centaur`构造函数的成员初始化列表中移除了对`LifeForm(1000)`的替代构造函数的指定，则将调用`LifeForm`的默认构造函数。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter09/Chp9-Q2.cpp`。

a. 有两个`LifeForm`子对象。

b. `LifeForm`构造函数和析构函数各自被调用两次。

# 第十章，实现关联、聚合和组合

1.  请参阅 GitHub 仓库中的`Assessments/Chapter10/Chp10-Q1.cpp`。

（后续问题）一旦您重载了一个接受`University &`作为参数的构造函数，这个版本可以通过首先在构造函数调用中对`University`指针进行解引用（以创建一个可引用的对象）来使用`University *`调用。

1.  a – f. 请参阅 GitHub 仓库中的`Assessments/Chapter10/Chp10-Q2.cpp`。

1.  a – b. （可选）请参阅 GitHub 仓库中的`Assessments/Chapter10/Chp10-Q3.cpp`。

# 第十一章，处理异常

1.  a – c. 请参阅 GitHub 仓库中的`Assessments/Chapter11/Chp11-Q1.cpp`。

# 第十二章，友元和运算符重载

1.  请参阅 GitHub 仓库中的`Assessments/Chapter12/Chp12-Q1.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter12/Chp12-Q2.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter12/Chp12-Q3.cpp`。

# 第十三章，使用模板

1.  a – b. 请参阅 GitHub 仓库中的`Assessments/Chapter13/Chp13-Q1.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter13/Chp13-Q2.cpp`。

# 第十四章，理解 STL 基础

1.  a – b. 请参阅 GitHub 仓库中的`Assessments/Chapter14/Chp14-Q1.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter14/Chp14-Q2.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter14/Chp14-Q3.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter14/Chp14-Q4.cpp`。

# 第十五章，测试类和组件

1.  a. 如果您的类包含一个（用户指定的）默认构造函数、拷贝构造函数、重载的赋值运算符和一个虚析构函数，则您的类遵循正交规范类形式。如果它们还包括移动拷贝构造函数和重载的移动赋值运算符，则您的类还遵循扩展规范类形式。

b. 如果您的类遵循规范类形式并确保类的所有实例都有完全构造的手段，则您的类将被认为是健壮的。测试一个类可以确保其健壮性。

1.  a – c. 请参阅 GitHub 仓库中的`Assessments/Chapter15/Chp15-Q2.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter15/Chp15-Q3.cpp`。

# 第十六章，使用观察者模式

1.  a – b. 请参阅 GitHub 仓库中的`Assessments/Chapter16/Chp16-Q1.cpp`。

1.  其他可能容易融入观察者模式的例子包括任何需要客户接收他们所希望的后备产品通知的应用程序。例如，许多人可能希望接种新冠疫苗，并希望在疫苗分发点的等待名单上。在这里，一个`VaccineDistributionSite`（感兴趣的主体）可以继承自`Subject`并包含一个`Person`对象列表，其中`Person`继承自`Observer`。`Person`对象将包含一个指向`VaccineDistributionSite`的指针。一旦在某个`VaccineDistributionSite`（即，发生了分发事件）有足够的疫苗供应，就可以调用`Notify()`来更新`Observer`实例（等待名单上的人）。每个`Observer`都将收到一个`Update()`，这将允许那个人安排预约。如果`Update()`返回成功并且`Person`已经安排了预约，`Observer`可以使用`Subject`从等待名单中释放自己。

# 第十七章，应用工厂模式

1.  a – b. 请参阅 GitHub 仓库中的`Assessments/Chapter17/Chp17-Q1.cpp`。

1.  其他可能容易融入工厂方法模式的例子包括许多类型的应用程序，这些应用程序可能需要根据构造时提供的特定值实例化各种派生类。例如，一个工资单应用程序可能需要各种类型的`Employee`实例，如`Manager`、`Engineer`、`Vice-President`等。工厂方法可以提供一种根据在雇佣`Employee`时提供的信息实例化各种类型`Employee`的方法。工厂方法模式是一种可以应用于许多类型应用程序的模式。

# 第十八章，应用适配器模式

1.  a – b. 请参阅 GitHub 仓库中的`Assessments/Chapter18/Chp18-Q1.cpp`。

1.  其他可能容易融入适配器模式的例子包括许多将现有经过良好测试的非 OO 代码重新用于提供 OO 接口（即，适配器类型的包装器）的例子。其他例子包括创建一个适配器将以前使用的类转换为当前所需的类（再次使用重用先前创建和经过良好测试的组件的想法）。一个例子是将以前用于表示汽油发动机汽车的`Car`类适配为一个表示`ElectricCar`的类。

# 第十九章，使用单例模式

1.  a – c. 请参阅 GitHub 仓库中的`Assessments/Chapter19/Chp19-Q1.cpp`。

1.  我们不能将`Singleton`中的`static instance()`方法标记为虚拟并在`President`中重写它，仅仅是因为静态方法永远不能是虚拟的。它们是静态绑定的，并且永远不会接收一个`this`指针。此外，签名可能需要不同（没有人喜欢意外隐藏函数的情况）。

1.  其他可能容易融入单例模式的例子包括创建一个公司的单例`CEO`、一个国家的单例`TreasuryDepartment`，或者一个国家的单例`Queen`。这些单例实例都提供了建立注册表以跟踪多个单例对象的机会。也就是说，许多国家可能只有一个`Queen`。在这种情况下，注册表不仅允许每个对象类型只有一个单例，而且允许每个其他限定符（如*国家*）只有一个单例。这是一个罕见的情况，其中给定类型的多个单例对象可以出现（但总是受控数量的对象）。

# 第二十章，使用 pImpl 模式去除实现细节

1.  请参阅 GitHub 仓库中的`Assessments/Chapter20/Chp20-Q1.cpp`。

1.  请参阅 GitHub 仓库中的`Assessments/Chapter20/Chp20-Q2.cpp`。

（后续问题）在本章中，简单地从采用 pImpl 模式的`Person`类继承`Student`类不会带来任何物流困难。此外，修改`Student`类以也采用 pImpl 模式并使用唯一指针更具挑战性。可能遇到各种困难，包括处理内联函数、向下转型、避免显式调用底层实现，或者需要回指针来帮助调用虚函数。请参阅在线解决方案以获取详细信息。

1.  其他可能容易融入 pImpl 模式以实现相对独立实现的例子包括创建通用的 GUI 组件，例如`Window`、`Scrollbar`、`Textbox`等，用于各种平台（派生类）。实现细节可以轻松隐藏。另一个例子可能是开发者希望隐藏在头文件中可能看到的实现细节的专有商业类。
