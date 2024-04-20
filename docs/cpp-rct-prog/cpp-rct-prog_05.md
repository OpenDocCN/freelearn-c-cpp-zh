# 第五章：Observables 的介绍

在最后三章中，我们学习了现代 C++的语言特性：多线程、无锁编程模型等。那里涵盖的主题可以被视为开始学习响应式编程模型的先决条件。响应式编程模型需要掌握函数式编程、并发编程、调度器、对象/函数式编程、设计模式和事件流处理等技能。我们已经在上一章中涵盖或涉及了函数式编程、对象/函数式编程以及与调度相关的一些主题。这次，我们将涵盖设计模式的精彩世界，以理解响应式编程的要点以及特别是 Observables。在下一章中，我们将在跳入 RxCpp 库之前处理事件流编程的主题。设计模式运动随着一本名为*设计模式：可复用面向对象软件的元素*的书籍的出版而达到了临界质量，这本书由**四人帮**（**GoF**）编写，其中列出了一组分为创建型、结构型和行为型家族的 23 种模式。GoF 目录将观察者模式定义为行为模式的一种。我们想要在这里传达的一个关键信息是，通过了解可敬的 GoF 模式，可以理解响应式编程模型。在本章中，我们将涵盖：

+   GoF 观察者模式

+   GoF 观察者模式的局限性

+   对设计模式和 Observables 进行全面审视

+   使用复合设计模式对建模现实世界的层次结构

+   使用访问者对复合物进行行为处理

+   将复合物扁平化并通过迭代器模式进行导航

+   通过改变视角，从迭代器转换为 Observable/Observer！

# GoF 观察者模式

GoF 观察者模式在 GoF 书中也被称为*发布-订阅模式*。这个想法很简单。`EventSource`（发出事件的类）将与事件接收器（监听事件通知的类）建立一对多的关系。每个`EventSource`都将有一个机制，让事件接收器订阅以获取不同类型的通知。单个`EventSource`可能会发出多个事件。当`EventSource`的状态发生变化或其领域发生重大事件时，它可以向成千上万的订阅者（事件接收器或监听器）发送通知。`EventSource`将遍历订阅者列表并逐个通知它们。GoF 书是在世界大多数时间都在进行顺序编程的时候编写的。诸如并发性之类的主题大多与特定于平台的库或`POSIX`线程库相关。我们将编写一个简单的 C++程序来演示观察者模式的整个思想。目的是快速理解观察者模式，鲁棒性等想法被次要地给予了优先级。这个清单是自包含的并且容易理解的：

```cpp
//-------------------- Observer.cpp 
#include <iostream> 
#include  <vector> 
#include <memory> 
using namespace std; 
//---- Forward declaration of event sink 
template<class T> 
class EventSourceValueObserver; 
//----------A toy implementation of EventSource
template<class T> 
class EventSourceValueSubject{ 
   vector<EventSourceValueObserver<T> *> sinks;  
   T State; // T is expected to be a value type 
  public: 
   EventSourceValueSubject() { State = 0; } 
   ~EventSourceValueSubject() { 
       sinks.clear(); 
   } 
   bool Subscribe( EventSourceValueObserver<T> *sink ) { sinks.push_back(sink);} 
   void NotifyAll() { for (auto sink : sinks) { sink->Update(State); }} 
   T GetState() { return State; } 
   void SetState(T pstate) { State = pstate; NotifyAll(); } 
};
```

上面的代码片段实现了一个微不足道的`EventSource`，它可以潜在地存储一个整数值作为状态。在现代 C++中，我们可以使用类型特征来检测消费者是否已经用整数类型实例化了这个类。由于我们的重点是阐明，我们没有添加与类型约束相关的断言。在下一个 C++标准中，有一个称为**concept**（在其他语言中称为约束）的概念，将有助于直接强制执行这一点（而不需要类型特征）。在现实生活中，`EventSource`可能存储大量变量或值流。对它们的任何更改都将广播给所有订阅者。在`SetState`方法中，当`EventSource`类的消费者（事件接收器本身是这个类中的消费者）改变状态时，`NotifyAll()`方法将被触发。`NotifyAll()`方法通过接收器列表工作，并调用`Update()`方法。然后，事件接收器可以执行特定于其上下文的任务。我们没有实现取消订阅等方法，以便专注于核心问题：

```cpp
//--------------------- An event sink class for the preceding EventSources 
template <class T> 
class EventSourceValueObserver{ 
    T OldState; 
  public: 
    EventSourceValueObserver() { OldState = 0; } 
    virtual ~EventSorceValueObserver() {} 
    virtual void Update( T State ) { 
       cout << "Old State " << OldState << endl; 
       OldState = State; 
       cout << "Current State " << State << endl;  
    } 
}; 
```

`EventSourceValueObserver`类已经实现了`Update`方法来执行与其上下文相关的任务。在这里，它只是将旧状态和当前状态的值打印到控制台上。在现实生活中，接收器可能会修改 UX 元素或通过通知将状态的传播传递给其他对象。让我们再写一个事件接收器，它将继承自`EventSourceValueObserver`：

```cpp
//------------ A simple specialized Observe 
class AnotherObserver : public EventSourceValueObserver<double> { 
  public: 
    AnotherObserver():EventSourceValueObserver() {} 
    virtual ~AnotherObserver() {} 
    virtual void Update( double State )  
    { cout << " Specialized Observer" << State <<  endl; } 
};
```

我们为演示目的实现了观察者的专门版本。这样做是为了表明我们可以有两个类的实例（可以从`EventSourceObserver<T>`继承）作为订阅者。在这里，当我们从`EventSource`收到通知时，我们也不做太多事情：

```cpp
int main() { 
   unique_ptr<EventSourceValueSubject<double>> 
                 evsrc(new EventSourceValueSubject<double>()); 
    //---- Create Two instance of Observer and Subscribe 
   unique_ptr<AnotherObserver> evobs( new AnotherObserver());
   unique_ptr<EventSourceValueObserver<double>> 
               evobs2( new EventSourceValueObserver<double>());
   evsrc->Subscribe( evobs.get() );
   evsrc->Subscribe( evobs2.get());
   //------ Change the State of the EventSource 
   //------ This should trigger call to Update of the Sink 
   evsrc->SetState(100); 
} 
```

上面的代码片段实例化了一个`EventSource`对象并添加了两个订阅者。当我们改变`EventSource`的状态时，订阅者将收到通知。这是观察者模式的关键。在普通的面向对象编程程序中，对象的消费是以以下方式进行的：

1.  实例化对象

1.  调用方法计算某个值或改变状态

1.  根据返回值或状态变化执行有用的操作

在这里，在观察者的情况下，我们已经做了以下工作：

1.  实例化对象（`EventSource`）

1.  通过实现观察者（用于事件监听）进行通知订阅

1.  当`EventSource`发生变化时，您将收到通知

1.  对通过通知接收到的值执行某些操作

这里概述的`Method`函数有助于关注点的分离，并实现了模块化。这是实现事件驱动代码的良好机制。与其轮询事件，不如要求被通知。大多数 GUI 工具包今天都使用类似的范例。

# GoF 观察者模式的局限性

GoF 模式书是在世界真正进行顺序编程的时候编写的。从当前的编程模型世界观来看，观察者模式实现的架构有很多异常。以下是其中一些：

+   主题和观察者之间的紧密耦合。

+   `EventSource`的生命周期由观察者控制。

+   观察者（接收器）可以阻塞`EventSource`。

+   实现不是线程安全的。

+   事件过滤是在接收器级别进行的。理想情况下，数据应该在数据所在的地方（在通知之前的主题级别）进行过滤。

+   大多数时候，观察者并不做太多事情，CPU 周期将被浪费。

+   `EventSource`理想上应该将值发布到环境中。环境应该通知所有订阅者。这种间接层次可以促进诸如事件聚合、事件转换、事件过滤和规范化事件数据等技术。

随着不可变变量、函数式组合、函数式风格转换、无锁并发编程等功能编程技术的出现，我们可以规避经典 Observer 模式的限制。行业提出的解决方案是 Observables 的概念。

在经典 Observer 模式中，一个勤奋的读者可能已经看到了异步编程模型被整合的潜力。`EventSource`可以对订阅者方法进行异步调用，而不是顺序循环订阅者。通过使用一种“发射并忘记”的机制，我们可以将`EventSource`与其接收器解耦。调用可以从后台线程、异步任务或打包任务，或适合上下文的合适机制进行。通知方法的异步调用具有额外的优势，即如果任何客户端阻塞（进入无限循环或崩溃），其他客户端仍然可以收到通知。异步方法遵循以下模式：

1.  定义处理数据、异常和数据结束的方法（在事件接收器方面）

1.  Observer（事件接收器）接口应该有`OnData`、`OnError`和`OnCompleted`方法

1.  每个事件接收器应该实现 Observer 接口

1.  每个`EventSource`（Observable）应该有订阅和取消订阅的方法

1.  事件接收器应该通过订阅方法订阅 Observable 的实例

1.  当事件发生时，Observable 会通知 Observer

这些事情有些已经在第一章中提到过，*响应式编程模型-概述和历史*。当时我们没有涉及异步部分。在本章中，我们将重新审视这些想法。根据作者们在技术演示和与开发人员的互动中积累的经验，直接跳入编程的 Observable/Observer 模型并不能帮助理解。大多数开发人员对 Observable/Observer 感到困惑，因为他们不知道这种模式解决了什么特定的问题。这里给出的经典 GoF Observer 实现是为了为 Observable Streams 的讨论设定背景。

# 对 GoF 模式的整体观察

设计模式运动始于一个时期，当时世界正在努力应对面向对象软件设计方法的复杂性。GoF 书籍和相关的模式目录为开发人员提供了一套设计大型系统的技术。诸如并发和并行性之类的主题并不在设计目录的设计者们的考虑之中。（至少，他们的工作没有反映出这一点！）

我们已经看到，通过经典 Observer 模式进行事件处理存在一些局限性，这在某些情况下可能是个问题。有什么办法？我们需要重新审视事件处理的问题，退一步。我们将稍微涉及一些哲学的主题，以不同的视角看待响应式编程模型（使用 Observable Streams 进行编程！）试图解决的问题。我们的旅程将帮助我们从 GOF 模式过渡到使用函数式编程构造的响应式编程世界。

本节中的内容有些抽象，并且是为了提供一个概念性背景，从这个背景中，本书的作者们接触了本章涵盖的主题。我们解释 Observables 的方法是从 GoF Composite/Visitor 模式开始，逐步达到 Observables 的主题。这种方法的想法来自一本关于阿德瓦伊塔·维丹塔（Advaita Vedanta）的书，这是一种起源于印度的神秘哲学传统。这个主题已经用西方哲学术语解释过。如果某个问题看起来有点抽象，可以随意忽略它。

Nataraja Guru（1895-1973）是一位印度哲学家，他是阿德瓦伊塔维达塔哲学的倡导者，这是一所基于至高力量的非二元论的印度哲学学派。根据这个哲学学派，我们周围所看到的一切，无论是人类、动物还是植物，都是绝对（梵文中称为婆罗门）的表现，它唯一的积极肯定是 SAT-CHIT-ANAND（维达塔哲学使用否定和反证来描述婆罗门）。这可以被翻译成英语为存在、本质和幸福（这里幸福的隐含含义是“好”）。在 DK Print World 出版的一本名为《统一哲学》的书中，他将 SAT-CHIT-ANAND 映射到本体论、认识论和价值论（哲学的三个主要分支）。以下表格给出了 SAT-CHIT-ANAND 可能与其他意义相近的实体的映射。

| **SAT** | **CHIT** | **ANAND** |
| --- | --- | --- |
| 存在 | 本质 | 幸福 |
| 本体论 | 认识论 | 价值论 |
| 我是谁？ | 我能知道什么？ | 我应该做什么？ |
| 结构 | 行为 | 功能 |

在 Vedanta（阿德瓦伊塔学派）哲学中，整个世界被视为存在、本质和幸福。从表中，我们将软件设计世界中的问题映射为结构、行为和功能的问题。世界上的每个系统都可以从结构、行为和功能的角度来看待。面向对象程序的规范结构是层次结构。我们将感兴趣的世界建模为层次结构，并以规范的方式处理它们。GOF 模式目录中有组合模式（结构）用于建模层次结构和访问者模式（行为）用于处理它们。

# 面向对象编程模型和层次结构

这一部分在概念上有些复杂，那些没有涉足过 GoF 设计模式的人可能会觉得有些困难。最好的策略可能是跳过这一部分，专注于运行示例。一旦理解了运行示例，就可以重新访问这一部分。

面向对象编程非常擅长建模层次结构。事实上，层次结构可以被认为是面向对象数据处理的规范数据模型。在 GoF 模式世界中，我们使用组合模式来建模层次结构。组合模式被归类为结构模式。每当使用组合模式时，访问者模式也将成为系统的一部分。访问者模式适用于处理组合以向结构添加行为。访问者/组合模式在现实生活中成对出现。当然，组合的一个实例可以由不同的访问者处理。在编译器项目中，**抽象语法树**（**AST**）将被建模为一个组合，并且将有访问者实现用于类型检查、代码优化、代码生成和静态分析等。

访问者模式的问题之一是它必须对组合的结构有一定的概念才能进行处理。此外，在需要处理组合层次结构中可用数据的筛选子集的上下文中，它将导致代码膨胀。我们可能需要为每个过滤条件使用不同的访问者。GoF 模式目录中还有另一个属于行为类别的模式，称为 Iterator，这是每个 C++程序员都熟悉的东西。Iterator 模式擅长以结构无关的方式处理数据。任何层次结构都必须被线性化或扁平化，以便被 Iterator 处理。例如，树可以使用 BFS Iterator 或 DFS Iterator 进行处理。对于应用程序员来说，树突然变成了线性结构。我们需要将层次结构扁平化，使其处于适合 Iterator 处理的状态。这个过程将由实现 API 的人来实现。Iterator 模式也有一些局限性（它是基于拉的），我们将通过一种称为 Observable/Observer 的模式将系统改为基于推的。这一部分有点抽象，但在阅读整个章节后，你可以回来理解发生了什么。简而言之，我们可以总结整个过程如下：

+   我们可以使用组合模式来建模层次结构

+   我们可以使用 Visitor 模式处理组合

+   我们可以通过 Iterator 来展开或线性化组合

+   Iterators 遵循拉取方法，我们需要为基于推的方案逆转视线

+   现在，我们已经成功地实现了 Observable/Observer 的方式来实现事物

+   Observables 和 Iterators 是二进制对立的（一个人的推是另一个人的拉！）

我们将实现所有前述观点，以对 Observables 有牢固的基础。

# 用于表达式处理的组合/访问者模式

为了演示从 GoF 模式目录到 Observables 的过程，我们将模拟一个四则运算计算器作为一个运行示例。由于表达式树或 AST 本质上是层次结构的，它们将是一个很好的例子，可以作为组合模式的模型。我们故意省略了编写解析器，以保持代码清单的简洁：

```cpp
#include <iostream> 
#include <memory> 
#include <list> 
#include <stack> 
#include <functional> 
#include <thread> 
#include <future> 
#include <random> 
#include "FuncCompose.h" // available int the code base 
using namespace std; 
//---------------------List of operators supported by the evaluator 
enum class OPERATOR{ ILLEGAL,PLUS,MINUS,MUL,DIV,UNARY_PLUS,UNARY_MINUS };  
```

我们定义了一个枚举类型来表示四个二元运算符（`+`，`-`，`*`，`/`）和两个一元运算符（`+`，`-`）。除了标准的 C++头文件，我们还包含了一个自定义头文件（`FuncCompose.h`），它可以在与本书相关的 GitHub 存储库中找到。它包含了 Compose 函数和管道运算符（`|`）的代码，用于函数组合。我们可以使用 Unix 管道风格的组合来将一系列转换联系在一起：

```cpp
//------------ forward declarations for the Composites  
class Number;  //----- Stores IEEE double precision floating point number  
class BinaryExpr; //--- Node for Binary Expression 
class UnaryExpr;  //--- Node for Unary Expression 
class IExprVisitor; //---- Interface for the Visitor  
//---- Every node in the expression tree will inherit from the Expr class 
class Expr { 
  public: 
   //---- The standard Visitor double dispatch method 
   //---- Normally return value of accept method are void.... and Concrete
   //---- classes store the result which can be retrieved later
   virtual double accept(IExprVisitor& expr_vis) = 0; 
   virtual ~Expr() {} 
}; 
//----- The Visitor interface contains methods for each of the concrete node  
//----- Normal practice is to use 
struct IExprVisitor{ 
   virtual  double Visit(Number& num) = 0; 
   virtual  double Visit(BinaryExpr& bin) = 0; 
   virtual  double Visit(UnaryExpr& un)=0 ; 
}; 
```

Expr 类将作为表达式树中所有节点的基类。由于我们的目的是演示组合/访问者 GoF 模式，我们只支持常数、二元表达式和一元表达式。Expr 类中的 accept 方法接受一个 Visitor 引用作为参数，方法的主体对所有节点都是相同的。该方法将把调用重定向到 Visitor 实现上的适当处理程序。为了更深入地了解本节涵盖的整个主题，通过使用您喜欢的搜索引擎搜索*双重分派*和*Visitor 模式*。

Visitor 接口（`IExprVisitor`）包含处理层次结构支持的所有节点类型的方法。在我们的情况下，有处理常数、二元运算符和一元运算符的方法。让我们看看节点类型的代码。我们从 Number 类开始：

```cpp
//---------A class to represent IEEE 754 interface 
class Number : public Expr { 
   double NUM; 
  public: 
   double getNUM() { return NUM;}    
   void setNUM(double num)   { NUM = num; } 
   Number(double n) { this->NUM = n; } 
   ~Number() {} 
   double accept(IExprVisitor& expr_vis){ return expr_vis.Visit(*this);} 
}; 
```

Number 类封装了 IEEE 双精度浮点数。代码很明显，我们需要关心的只是`accept`方法的内容。该方法接收一个`visitor`类型的参数（`IExprVisitor&`）。该例程只是将调用反映到访问者实现的适当节点上。在这种情况下，它将在`IExpressionVisitor`上调用`Visit(Number&)`：

```cpp
//-------------- Modeling Binary Expresison  
class BinaryExpr : public Expr { 
   Expr* left; Expr* right; OPERATOR OP; 
  public: 
   BinaryExpr(Expr* l,Expr* r , OPERATOR op ) { left = l; right = r; OP = op;} 
   OPERATOR getOP() { return OP; } 
   Expr& getLeft() { return *left; } 
   Expr& getRight() { return *right; } 
   ~BinaryExpr() { delete left; delete right;left =0; right=0; } 
   double accept(IExprVisitor& expr_vis) { return expr_vis.Visit(*this);} 
};  
```

`BinaryExpr`类模拟了具有左右操作数的二元运算。操作数可以是层次结构中的任何类。候选类包括`Number`、`BinaryExpr`和`UnaryExpr`。这可以到任意深度。在我们的情况下，终端节点是 Number。先前的代码支持四个二元运算符：

```cpp
//-----------------Modeling Unary Expression 
class UnaryExpr : public Expr { 
   Expr * right; OPERATOR op; 
  public: 
   UnaryExpr( Expr *operand , OPERATOR op ) { right = operand;this-> op = op;} 
   Expr& getRight( ) { return *right; } 
   OPERATOR getOP() { return op; } 
   virtual ~UnaryExpr() { delete right; right = 0; } 
   double accept(IExprVisitor& expr_vis){ return expr_vis.Visit(*this);} 
};  
```

`UnaryExpr`方法模拟了带有运算符和右侧表达式的一元表达式。我们支持一元加和一元减。右侧表达式可以是`UnaryExpr`、`BinaryExpr`或`Number`。现在我们已经为所有支持的节点类型编写了实现，让我们专注于访问者接口的实现。我们将编写一个树遍历器和评估器来计算表达式的值：

```cpp
//--------An Evaluator for Expression Composite using Visitor Pattern  
class TreeEvaluatorVisitor : public IExprVisitor{ 
  public: 
   double Visit(Number& num){ return num.getNUM();} 
   double Visit(BinaryExpr& bin) { 
     OPERATOR temp = bin.getOP(); double lval = bin.getLeft().accept(*this); 
     double rval = bin.getRight().accept(*this); 
     return (temp == OPERATOR::PLUS) ? lval + rval: (temp == OPERATOR::MUL) ?  
         lval*rval : (temp == OPERATOR::DIV)? lval/rval : lval-rval;   
   } 
   double Visit(UnaryExpr& un) { 
     OPERATOR temp = un.getOP(); double rval = un.getRight().accept(*this); 
     return (temp == OPERATOR::UNARY_PLUS)  ? +rval : -rval; 
   } 
};
```

这将对 AST 进行深度优先遍历，并递归评估节点。让我们编写一个表达式处理器（`IExprVisitor`的实现），它将以**逆波兰表示法**（**RPN**）形式将表达式树打印到控制台上：

```cpp
//------------A Visitor to Print Expression in RPN
class ReversePolishEvaluator : public IExprVisitor {
    public:
    double Visit(Number& num){cout << num.getNUM() << " " << endl; return 42;}
    double Visit(BinaryExpr& bin){
        bin.getLeft().accept(*this); bin.getRight().accept(*this);
        OPERATOR temp = bin.getOP();
        cout << ( (temp==OPERATOR::PLUS) ? " + " :(temp==OPERATOR::MUL) ?
        " * " : (temp == OPERATOR::DIV) ? " / ": " - " ) ; return 42;
    }
    double Visit(UnaryExpr& un){
        OPERATOR temp = un.getOP();un.getRight().accept(*this);
        cout << (temp == OPERATOR::UNARY_PLUS) ?" (+) " : " (-) "; return 42;
    }
};
```

RPN 表示法也称为后缀表示法，其中运算符位于操作数之后。它们适合使用评估堆栈进行处理。它们构成了 Java 虚拟机和.NET CLR 所利用的基于堆栈的虚拟机架构的基础。现在，让我们编写一个主函数将所有内容整合在一起：

```cpp
int main( int argc, char **argv ){ 
     unique_ptr<Expr>   
            a(new BinaryExpr( new Number(10) , new Number(20) , OPERATOR::PLUS)); 
     unique_ptr<IExprVisitor> eval( new TreeEvaluatorVisitor()); 
     double result = a->accept(*eval); 
     cout << "Output is => " << result << endl; 
     unique_ptr<IExprVisitor>  exp(new ReversePolishEvaluator()); 
     a->accept(*exp); 
}
```

此代码片段创建了一个组合的实例（`BinaryExpr`的一个实例），并实例化了`TreeEvaluatorVisitor`和`ReversePolshEvaluator`的实例。然后，调用 Expr 的`accept`方法开始处理。我们将在控制台上看到表达式的值和表达式的 RPN 等价形式。在本节中，我们学习了如何创建一个组合，并使用访问者接口处理组合。组合/访问者的其他潜在示例包括存储目录内容及其遍历、XML 处理、文档处理等。普遍观点认为，如果您了解组合/访问者二者，那么您已经很好地理解了 GoF 模式目录。

我们已经看到，组合模式和访问者模式作为一对来处理系统的结构和行为方面，并提供一些功能。访问者必须以一种假定了组合结构的认知方式编写。从抽象的角度来看，这可能是一个潜在的问题。层次结构的实现者可以提供一种将层次结构展平为列表的机制（在大多数情况下是可能的）。这将使 API 实现者能够提供基于迭代器的 API。基于迭代器的 API 也适用于函数式处理。让我们看看它是如何工作的。

# 展平组合以进行迭代处理

我们已经了解到，访问者模式必须了解复合体的结构，以便有人编写访问者接口的实例。这可能会产生一个称为*抽象泄漏*的异常。GoF 模式目录中有一个模式，将帮助我们以结构不可知的方式导航树的内容。是的，你可能已经猜对了：迭代器模式是候选者！为了使迭代器发挥作用，复合体必须被扁平化为列表序列或流。让我们编写一些代码来扁平化我们在上一节中建模的表达式树。在编写扁平化复合体的逻辑之前，让我们创建一个数据结构，将 AST 的内容作为列表存储。列表中的每个节点必须存储操作符或值，具体取决于我们是否需要存储操作符或操作数。我们为此描述了一个名为`EXPR_ITEM`的数据结构：

```cpp
//////////////////////////// 
// A enum to store discriminator -> Operator or a Value? 
enum class ExprKind{  ILLEGAL_EXP,  OPERATOR , VALUE }; 
// A Data structure to store the Expression node. 
// A node will either be a Operator or Value 
struct EXPR_ITEM { 
    ExprKind knd; double Value; OPERATOR op; 
    EXPR_ITEM():op(OPERATOR::ILLEGAL),Value(0),knd(ExprKind::ILLEGAL_EXP){} 
    bool SetOperator( OPERATOR op ) 
    {  this->op = op;this->knd = ExprKind::OPERATOR; return true; } 
    bool SetValue(double value)  
    {  this->knd = ExprKind::VALUE;this->Value = value;return true;} 
    string toString() {DumpContents();return "";} 
   private: 
      void DumpContents() { //---- Code omitted for brevity } 
}; 
```

`list<EXPR_ITEM>`数据结构将以线性结构存储复合的内容。让我们编写一个类来扁平化复合体：

```cpp
//---- A Flattener for Expressions 
class FlattenVisitor : public IExprVisitor { 
        list<EXPR_ITEM>  ils; 
        EXPR_ITEM MakeListItem(double num) 
        { EXPR_ITEM temp; temp.SetValue(num); return temp; } 
        EXPR_ITEM MakeListItem(OPERATOR op) 
        { EXPR_ITEM temp;temp.SetOperator(op); return temp;} 
        public: 
        list<EXPR_ITEM> FlattenedExpr(){ return ils;} 
        FlattenVisitor(){} 
        double Visit(Number& num){ 
           ils.push_back(MakeListItem(num.getNUM()));return 42; 
        } 
        double Visit(BinaryExpr& bin) { 
            bin.getLeft().accept(*this);bin.getRight().accept(*this); 
            ils.push_back(MakeListItem(bin.getOP()));return 42; 
        } 
         double Visit(UnaryExpr& un){ 
            un.getRight().accept(*this); 
            ils.push_back(MakeListItem(un.getOP())); return 42; 
        } 
};  
```

`FlattenerVistor`类将复合`Expr`节点扁平化为`EXPR_ITEM`列表。一旦复合体被线性化，就可以使用迭代器模式处理项目。让我们编写一个小的全局函数，将`Expr`树转换为`list<EXPR_ITEM>`：

```cpp
list<EXPR_ITEM> ExprList(Expr* r) { 
   unique_ptr<FlattenVisitor> fl(new FlattenVisitor()); 
    r->accept(*fl); 
    list<EXPR_ITEM> ret = fl->FlattenedExpr();return ret; 
 }
```

全局子例程`ExprList`将扁平化一个任意表达式树的`EXPR_ITEM`列表。一旦我们扁平化了复合体，我们可以使用迭代器来处理内容。在将结构线性化为列表后，我们可以使用堆栈数据结构来评估表达式数据以产生输出：

```cpp
//-------- A minimal stack to evaluate RPN expression 
class DoubleStack : public stack<double> { 
   public: 
    DoubleStack() { } 
    void Push( double a ) { this->push(a);} 
    double Pop() { double a = this->top(); this->pop(); return a; } 
};  
```

`DoubleStack`是 STL 堆栈容器的包装器。这可以被视为一种帮助程序，以保持清单的简洁。让我们为扁平化表达式编写一个求值器。我们将遍历列表`<EXPR_ITEM>`并将值推送到堆栈中，如果遇到值的话。如果遇到操作符，我们将从堆栈中弹出值并应用操作。结果再次推入堆栈。在迭代结束时，堆栈中现有的元素将是与表达式相关联的值：

```cpp
//------Iterator through eachn element of Expression list 
double Evaluate( list<EXPR_ITEM> ls) { 
   DoubleStack stk; double n; 
   for( EXPR_ITEM s : ls ) { 
     if (s.knd == ExprKind::VALUE) { stk.Push(s.Value); } 
     else if ( s.op == OPERATOR::PLUS) { stk.Push(stk.Pop() + stk.Pop());} 
     else if (s.op == OPERATOR::MINUS ) { stk.Push(stk.Pop() - stk.Pop());} 
     else if ( s.op ==  OPERATOR::DIV) { n = stk.Pop(); stk.Push(stk.Pop() / n);} 
     else if (s.op == OPERATOR::MUL) { stk.Push(stk.Pop() * stk.Pop()); } 
     else if ( s.op == OPERATOR::UNARY_MINUS) { stk.Push(-stk.Pop()); } 
    } 
   return stk.Pop(); 
} 
//-----  Global Function Evaluate an Expression Tree 
double Evaluate( Expr* r ) { return Evaluate(ExprList(r)); } 
```

让我们编写一个主程序，调用这个函数来评估表达式。求值器中的代码清单易于理解，因为我们正在减少一个列表。在基于树的解释器中，事情并不明显：

```cpp
int main( int argc, char **argv ){      
     unique_ptr<Expr>
         a(new BinaryExpr( new Number(10) , new Number(20) , OPERATOR::PLUS)); 
     double result = Evaluate( &(*a)); 
     cout << result << endl; 
} 
```

# 列表上的 Map 和 Filter 操作

Map 是一个功能操作符，其中一个函数将被应用于列表。Filter 将对列表应用谓词并返回另一个列表。它们是任何功能处理管道的基石。它们也被称为高阶函数。我们可以编写一个通用的 Map 函数，使用`std::transform`用于`std::list`和`std::vector`：

```cpp
template <typename R, typename F> 
R Map(R r , F&& fn) { 
      std::transform(std::begin(r), std::end(r), std::begin(r), 
         std::forward<F>(fn)); 
      return r; 
} 
```

让我们还编写一个函数来过滤`std::list`（我们假设只会传递一个列表）。相同的方法也适用于`std::vector`。我们可以使用管道操作符来组合一个高阶函数。复合函数也可以作为谓词传递：

```cpp
template <typename R, typename F> 
R Filter( R r , F&& fn ) { 
   R ret(r.size()); 
   auto first = std::begin(r), last = std::end(r) , result = std::begin(ret);  
   bool inserted = false; 
   while (first!=last) { 
    if (fn(*first)) { *result = *first; inserted = true; ++result; }  
    ++first; 
   } 
   if ( !inserted ) { ret.clear(); ret.resize(0); } 
   return ret; 
}
```

在这个 Filter 的实现中，由于`std::copy_if`的限制，我们被迫自己编写迭代逻辑。通常建议使用 STL 函数的实现来编写包装器。对于这种特殊情况，我们需要检测列表是否为空：

```cpp
//------------------ Global Function to Iterate through the list  
void Iterate( list<EXPR_ITEM>& s ){ 
    for (auto n : s ) { std::cout << n.toString()  << 'n';} 
} 
```

让我们编写一个主函数将所有内容组合在一起。代码将演示如何在应用程序代码中使用`Map`和`Filter`。功能组合和管道操作符的逻辑在`FuncCompose.h`中可用：

```cpp
int main( int argc, char **argv ){ 
     unique_ptr<Expr>   
        a(new BinaryExpr( new Number(10.0) , new Number(20.0) , OPERATOR::PLUS)); 
      //------ExprList(Expr *) will flatten the list and Filter will by applied 
      auto cd = Filter( ExprList(&(*a)) , 
            [](auto as) {  return as.knd !=   ExprKind::OPERATOR;} ); 
      //-----  Square the Value and Multiply by 3... used | as composition Operator 
      //---------- See FuncCompose.h for details 
      auto cdr = Map( cd, [] (auto s ) {  s.Value *=3; return s; } |  
                  [] (auto s ) { s.Value *= s.Value; return s; } ); 
      Iterate(cdr);  
} 
```

`Filter`例程创建一个新的`list<Expr>`，其中只包含表达式中使用的值或操作数。`Map`例程在值列表上应用复合函数以返回一个新列表。

# 逆转注视可观察性！

我们已经学会了如何将复合转换为列表，并通过迭代器遍历它们。迭代器模式从数据源中提取数据，并在消费者级别操纵结果。我们面临的最重要的问题之一是我们正在耦合`EventSource`和事件接收器。GoF 观察者模式在这里也没有帮助。

让我们编写一个可以充当事件中心的类，事件接收器将订阅该类。通过拥有事件中心，我们现在将有一个对象，它将充当`EventSource`和事件接收器之间的中介。这种间接的一个优点很容易明显，即我们的类可以在到达消费者之前聚合、转换和过滤事件。消费者甚至可以在事件中心级别设置转换和过滤条件：

```cpp
//----------------- OBSERVER interface 
struct  OBSERVER { 
    int id; 
    std::function<void(const double)> ondata; 
    std::function<void()> oncompleted; 
    std::function<void(const std::exception &)> onexception; 
}; 
//--------------- Interface to be implemented by EventSource 
struct OBSERVABLE { 
   virtual bool Subscribe( OBSERVER * obs ) = 0; 
    // did not implement unsuscribe  
}; 
```

我们已经在[第一章](https://cdp.packtpub.com/c___reactive_programming/wp-admin/post.php?post=53&action=edit#post_26)中介绍了`OBSERVABLE`和`OBSERVER`，*响应式编程模型-概述和历史*和第二章，*现代 C++及其关键习惯的概览*。`EventSource`实现了`OBSERVABLE`，事件接收器实现了`OBSERVER`接口。从`OBSERVER`派生的类将实现以下方法：

+   `ondata`（用于接收数据）

+   `onexception`（异常处理）

+   `oncompleted`（数据结束）

`EventSource`类将从`OBSERVABLE`派生，并且必须实现：

+   Subscribe（订阅通知）

+   Unsubscribe（在我们的情况下未实现）

```cpp
//------------------A toy implementation of EventSource 
template<class T,class F,class M, class Marg, class Farg > 
class EventSourceValueSubject : public OBSERVABLE { 
   vector<OBSERVER> sinks;  
   T *State;  
   std::function<bool(Farg)> filter_func; 
   std::function<Marg(Marg)> map_func;
```

`map_func`和`filter_func`是可以帮助我们在将值异步分派给订阅者之前转换和过滤值的函数。在实例化`EventSource`类时，我们将这些值作为参数给出。目前，我们已经根据假设编写了代码，即只有`Expr`对象将存储在`EventSource`中。我们可以有一个表达式的列表或向量，并将值流式传输给订阅者。为此，实现可以将标量值推送到监听器：

```cpp
  public: 
   EventSourceValueSubject(Expr *n,F&& filter, M&& mapper) { 
       State = n; map_func = mapper; filter_func = filter; NotifyAll();  
   } 
   ~EventSourceValueSubject() {  sinks.clear(); } 
   //------ used Raw Pointer ...In real life, a shared_ptr<T>
   //------ is more apt here
   virtual  bool Subscribe( OBSERVER  *sink ) { sinks.push_back(*sink); return true;} 
```

我们做出了一些假设，即`Expr`对象将由调用者拥有。我们还省略了取消订阅方法的实现。构造函数接受一个`Expr`对象，一个`Filter`谓词（可以是使用|运算符的复合函数），以及一个`Mapping`函数（可以是使用`|`运算符的复合函数）：

```cpp
   void NotifyAll() { 
      double ret = Evaluate(State); 
      list<double> ls; ls.push_back(ret); 
      auto result = Map( ls, map_func);; // Apply Mapping Logic 
      auto resulttr = Filter( result,filter_func); //Apply Filter 
      if (resulttr.size() == 0 ) { return; } 
```

在评估表达式后，标量值将放入 STL 列表中。然后，将在列表上应用 Map 函数以转换值。将来，我们将处理一系列值。一旦我们映射或转换了值，我们将对列表应用过滤器。如果列表中没有值，则方法将返回而不通知订阅者：

```cpp
      double dispatch_number = resulttr.front(); 
      for (auto sink : sinks) {  
           std::packaged_task<int()> task([&]()  
           { sink.ondata(dispatch_number); return 1;  }); 
           std::future<int> result = task.get_future();task(); 
           double dresult = result.get(); 
         } 
     }
```

在此代码中，我们将调用`packaged_task`将数据分派到事件接收器。工业级库使用称为调度器的代码片段来执行此任务的一部分。由于我们使用的是 fire and forget，接收器将无法阻止`EventSource`。这是 Observables 的最重要用例之一：

```cpp
      T* GetState() { return State; } 
      void SetState(T *pstate) { State = pstate; NotifyAll(); } 
}; 
```

现在，让我们编写一个方法，根据现代 C++随机数生成器发出随机表达式，具有均匀概率分布。选择这种分布是相当任意的。我们也可以尝试其他分布，以查看不同的结果：

```cpp
Expr *getRandomExpr(int start, int end) { 
    std::random_device rd; 
    std::default_random_engine reng(rd()); 
    std::uniform_int_distribution<int> uniform_dist(start, end); 
    double mean = uniform_dist(reng); 
    return  new  
          BinaryExpr( new Number(mean*1.0) , new Number(mean*2.0) , OPERATOR::PLUS); 
} 
```

现在，让我们编写一个主函数将所有内容组合在一起。我们将使用`Expr`、`Filter`和`Mapper`实例化`EventSourceValueSubject`类：

```cpp
int main( int argc, char **argv ){ 
     unique_ptr<Expr>   
         a(new BinaryExpr( new Number(10) , new Number(20) , OPERATOR::PLUS)); 
     EventSourceValueSubject<Expr,std::function<bool(double)>, 
                    std::function<double(double)>,double,double>  
                    temp(&(*a),[] (auto s ) {   return s > 40.0;  }, 
                    []  (auto s ) { return s+ s ; }  | 
                    []  (auto s ) { return s*2;} ); 
```

在实例化对象时，我们使用管道运算符来组合两个 Lambda。这是为了演示我们可以组合任意数量的函数以形成复合函数。当我们编写 RxCpp 程序时，我们将大量利用这种技术。

```cpp
     OBSERVER obs_one ;     OBSERVER obs_two ; 
     obs_one.ondata = [](const double  r) {  cout << "*Final Value " <<  r << endl;}; 
     obs_two.ondata = [] ( const double r ){ cout << "**Final Value " << r << endl;};
```

在这段代码中，我们实例化了两个`OBSERVER`对象，并使用 Lambda 函数将它们分配给 ondata 成员。我们没有实现其他方法。这仅用于演示目的：

```cpp
     temp.Subscribe(&obs_one); temp.Subscribe(&obs_two);   
```

我们订阅了使用`OBSERVER`实例的事件通知。我们只实现了 ondata 方法。实现`onexception`和`oncompleted`是微不足道的任务：

```cpp
     Expr *expr = 0; 
     for( int i= 0; i < 10; ++i ) { 
           cout << "--------------------------" <<  i << " "<< endl; 
           expr = getRandomExpr(i*2, i*3 ); temp.SetState(expr); 
           std::this_thread::sleep_for(2s); delete expr; 
     } 
} 
```

我们通过将表达式设置为`EventSource`对象来评估一系列随机表达式。经过转换和过滤，如果还有值剩下，该值将通知给`OBSERVER`，并打印到控制台。通过这种方式，我们成功地使用`packaged_taks`编写了一个非阻塞的`EventSource`。在本章中，我们演示了以下内容：

+   使用复合对表达树进行建模

+   通过 Visitor 接口处理复合

+   将表达树展平为列表，并通过迭代器进行处理（拉）

+   从`EventSource`到事件接收端（推送）的凝视反转

# 总结

在本章中，我们涵盖了很多内容，朝着响应式编程模型迈进。我们了解了 GoF Observer 模式并理解了它的缺点。然后，我们偏离了哲学，以了解从结构、行为和功能的角度看世界的方法。我们在表达树建模的背景下学习了 GoF Composite/Visitor 模式。我们学会了如何将层次结构展平为列表，并通过迭代器对其进行导航。最后，我们稍微改变了事物的方案，以达到 Observables。通常，Observables 与 Streams 一起工作，但在我们的情况下，它是一个标量值。在下一章中，我们将学习有关事件流处理，以完成学习响应式编程的先决条件。
