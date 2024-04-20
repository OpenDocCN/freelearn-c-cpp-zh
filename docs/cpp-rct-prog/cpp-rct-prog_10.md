# 在 RxCpp 中创建自定义操作符

在过去的三章中，我们学习了 RxCpp 库及其编程模型。我们还将所学内容应用到了 GUI 编程的上下文中。从心智模型的角度来看，任何想以响应式方式编写程序的开发人员都必须理解可观察对象、观察者以及它们之间的操作符。当然，调度器和主题也很重要。响应式程序的大部分逻辑都驻留在操作符中。RxCpp 库作为其实现的一部分提供了许多内置（库存）操作符。我们已经在我们的程序中使用了其中一些。在本章中，我们将学习如何实现自定义操作符。要编写自定义操作符，我们需要深入了解与 RxCpp 库相关的一些高级主题。本章涵盖的主题如下：

+   Rx 操作符的哲学

+   链接库存操作符

+   编写基本的 RxCpp 操作符

+   编写不同类型的自定义操作符

+   使用`lift<T>`元操作符编写自定义操作符

+   向 RxCpp 库源代码中添加操作符

# Rx 操作符的哲学

如果你看任何响应式程序，我们会看到一系列操作符堆叠在可观察对象和观察者之间。开发人员使用流畅接口来链接操作符。在 RxCpp 中，可以使用点（`.`）或管道（`|`）来执行操作符链式调用。从软件接口的角度来看，每个操作符都接受一个可观察对象，并返回一个相同类型或不同类型的可观察对象。

RxCpp 可观察对象/观察者交互的一般用法（伪代码）如下：

```cpp
   Observable().     // Source Observable 
          Op1().     // First operator 
          Op2().     // Second operator 
                     ..                         
                     .. 
          Opn().subscribe( on_datahandler, 
                            on_errorhandler, 
                            on_completehandler); 
```

尽管在操作符链式调用时我们使用流畅接口，但实际上我们是在将函数组合在一起。为了将函数组合在一起，函数的返回值应该与组合链中的函数的参数类型兼容。

操作符以可观察对象作为参数，并返回另一个可观察对象。有一些情况下，它返回的是除可观察对象之外的值。只有那些返回可观察对象的操作符才能成为操作符链式调用的一部分。

要编写一个新的操作符，使其成为操作符链式调用方法的一部分，最好的方法是将它们作为`observable<T>`类型的方法添加。然而，编写一个可以在不同上下文中运行的生产质量操作符最好留给 RxCpp 内部的专家。另一个选择是使用 RxCpp 库中提供的`lift<t>`（`...`）操作符。我们将在本章中涵盖这两种策略。

每个操作符实现都应该具有的另一个非常重要的属性是它们应该是无副作用的。至少，它们不应该改变输入可观察对象的内容。换句话说，充当操作符的函数或函数对象应该是一个纯函数。

# 链接库存操作符

我们已经学到了 RxCpp 操作符是在可观察对象上操作的（作为输入接收），并返回可观察对象。这使得这些操作符可以通过操作符链式调用一一调用。链中的每个操作符都会转换从前一个操作符接收到的流中的元素。源流在这个过程中不会被改变。在链式调用操作符时，我们使用流畅接口语法。

开发人员通常在实现 GOF 构建器模式的类的消费上使用流畅接口。构建器模式的实现是以无序的方式实现的。尽管操作符链式调用的语法类似，但在响应式世界中操作符被调用的顺序确实很重要。

让我们编写一个简单的程序，帮助我们理解可观察对象操作符链式执行顺序的重要性。在这个特定的例子中，我们有一个可观察流，在这个流中我们应用 map 操作符两次：一次是为了找出平方，然后是为了找出值的两个实例。我们先应用平方函数，然后是两次函数：

```cpp
//----- operatorChaining1.cpp 
//----- Square and multiplication by 2 in order 
#include "rxcpp/rx.hpp" 
int main() 
{ 
    auto values = rxcpp::observable<>::range(1, 3). 
        map([](int x) { return x * x; }). 
        map([](int x) { return x * 2; }); 
    values.subscribe( 
        [](int v) {printf("OnNext: %dn", v); }, 
        []() {printf("OnCompletedn"); }); 
    return 0; 
} 
```

前面的程序将产生以下输出：

```cpp
OnNext: 2 
OnNext: 8 
OnNext: 18 
OnCompleted
```

现在，让我们颠倒应用顺序（先缩放 2 倍，两次，然后是参数的平方），然后查看输出，看看我们会得到不同的输出（在第一种情况下，先应用了平方，然后是缩放 2 倍）。以下程序将解释执行顺序，如果我们将程序生成的输出与之前的程序进行比较：

```cpp
//----- operatorChaining2.cpp 
//----- Multiplication by 2 and Square in order 
#include "rxcpp/rx.hpp" 
int main() 
{ 
    auto values = rxcpp::observable<>::range(1, 3). 
        map([](int x) { return x * 2; }). 
        map([](int x) { return x * x; }); 
    values.subscribe( 
        [](int v) {printf("OnNext: %dn", v); }, 
        []() {printf("OnCompletedn"); }); 
    return 0; 
} 
```

程序产生的输出如下：

```cpp
OnNext: 4 
OnNext: 16 
OnNext: 36 
OnCompleted 
```

在 C++中，我们可以很好地组合函数，因为 Lambda 函数和 Lambda 函数的惰性评估。RxCpp 库利用了这一事实来实现操作符。如果有三个函数（`F`、`G`、`H`）以`observable<T>`作为输入参数并返回`observable<T>`，我们可以象征性地将它们组合如下：

```cpp
F(G( H(x)) 
```

如果我们使用操作符链，可以写成如下形式：

```cpp
x.H().G().F() 
```

现在我们已经学会了操作符链实际上是在进行操作符组合。两者产生类似的结果，但操作符链更易读和直观。本节的一个目的是建立这样一个事实，即操作符组合和操作符链提供类似的功能。最初我们实现的操作符可以组合在一起（不能被链式调用），我们将学习如何创建适合操作符链的操作符。

# 编写基本的 RxCpp 自定义操作符

在上一节中，我们讨论了操作符链。操作符链是可能的，因为库存操作符是作为`observable<T>`类型的一部分实现的。我们最初要实现的操作符不能成为操作符链策略的一部分。在本节中，我们将实现一些 RxCpp 操作符，可以转换 Observable 并返回另一个 Observable。

# 将 RxCpp 操作符写为函数

为了开始讨论，让我们编写一个简单的操作符，它可以在 observable<string>上工作。该操作符只是在流中的每个项目之前添加文字`Hello`：

```cpp
//----------- operatorSimple.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace rxcpp; 
using namespace rxcpp::operators; 
// Write a Simple Reactive operator Takes an Observable<string> and 
// Prefix Hello to every item and return another Observable<string> 
observable<std::string> helloNames(observable<std::string> src ) { 
    return src.map([](std::string s) { return "Hello, " + s + "!"; }); 
} 
```

我们实现的自定义操作符是为了演示如何编写一个可以在 Observable 上工作的操作符。编写的操作符必须使用函数语义来调用，并且实现不适合操作符链。既然我们已经实现了一个操作符，让我们编写一个主函数来测试操作符的工作方式：

```cpp
int main() { 
     std::array< std::string,4 > a={{"Praseed", "Peter", "Sanjay","Raju"}}; 
     // Apply helloNames operator on the observable<string>  
     // This operator cannot be part of the method chaining strategy 
     // We need to invoke it as a function  
     // If we were implementing this operator as part of the
     //          RxCpp observable<T> 
     //   auto values = rxcpp::observable<>:iterate(a).helloNames(); 
     auto values = helloNames(rxcpp::observable<>::iterate(a));  
     //-------- As usual subscribe  
     values.subscribe(  
              [] (std::string f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

程序将产生以下输出：

```cpp
Hello, Praseed! 
Hello, Peter! 
Hello, Sanjay! 
Hello, Raju! 
Hello World.. 
```

# 将 RxCpp 操作符写为 Lambda 函数

我们已经将我们的第一个自定义操作符写成了一个`unary`函数。所有操作符都是以 Observables 作为参数的`unary`函数。该函数以`observable<string>`作为参数，并返回另一个`observable<string>`。我们可以通过将操作符（内联）作为 Lambda 来实现相同的效果。让我们看看如何做到：

```cpp
//----------- operatorInline.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace rxcpp; 
using namespace rxcpp::operators; 
int main() { 
     std::array< std::string,4 > a={{"Praseed", "Peter", "Sanjay","Raju"}}; 
     auto helloNames = [] (observable<std::string> src ) { 
           return src.map([](std::string s) {  
             return "Hello, " + s + "!";  
             }); 
     }; 
     // type of values will be observable<string> 
     // Lazy Evaluation  
     auto values = helloNames(rxcpp::observable<>::iterate(a));  
     //-------- As usual subscribe  
     values.subscribe(  
              [] (std::string f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

程序的输出如下：

```cpp
Hello, Praseed! 
Hello, Peter! 
Hello, Sanjay! 
Hello, Raju! 
Hello World.. 
```

输出显示，程序行为是相同的，无论是使用普通函数还是 Lambda 函数。Lambda 函数的优势在于调用站点的创建和函数的消耗。

# 组合自定义 RxCpp 操作符

我们已经在本书中学习了函数组合（第二章*，现代 C++及其关键习语之旅*）。函数组合是可能的，当一个函数的返回值与另一个函数的输入参数兼容时。在操作符的情况下，由于大多数操作符返回 Observables 并将 Observables 作为参数，它们适合函数组合。在本节中，我们的操作符适合组合，但它们还不能被链式调用。让我们看看如何组合操作符：

```cpp
//----------- operatorCompose.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace rxcpp; 
using namespace rxcpp::operators; 
int main() { 
     std::array< int ,4 > a={{10, 20,30,40}}; 
     // h-function (idempotent) 
     auto h = [] (observable<int> src ) { 
       return src.map([](int n ) { return n; }); 
     }; 
     // g-function 
     auto g = [] (observable<int> src ) { 
          return src.map([](int n ) { return n*2; }); 
     }; 
     // type of values will be observable<string> 
     // Lazy Evaluation ... apply h over observable<string> 
     // on the result, apply g  
     auto values = g(h(rxcpp::observable<>::iterate(a)));  
     //-------- As usual subscribe  
     values.subscribe(  
              [] (int f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

程序的输出如下：

```cpp
20 
40 
60 
80 
Hello World.. 
```

# 不同类型的自定义操作符

RxCpp 库包含作为库存提供的不同类型的运算符。RxCpp 的默认运算符集对于大多数应用程序来说已经足够了。可用运算符的不同类型如下：

+   创建运算符

+   转换运算符

+   过滤运算符

+   组合运算符

+   错误处理运算符

+   实用运算符

+   布尔运算符

+   数学运算符

运算符的分类为开发人员提供了一个选择适当运算符的良好框架。在本节中，我们将实现以下内容：

+   自定义创建运算符

+   自定义转换运算符

+   涉及调度程序的自定义操作

# 编写自定义创建运算符

大多数 RxCpp 运算符函数接受 Observable 并返回一个 Observable 以实现运算符的组合。我们需要做一些额外的工作，以使组合具有可链式的方式（在下一节中，我们将介绍`lift<t>`和向`RxCpp`库中的`[observable<T>]` Observable 添加运算符的主题）。我们在本节中实现的运算符将帮助我们从输入数据创建一个 Observable。我们可以从任何类型的单个值、一系列值、STL 容器的迭代器、另一个 Observable 等创建 Observable 流。让我们讨论一个接受 STL 容器并创建 Observable 的示例程序，然后进行一些转换：

```cpp
//------ CustomOperator1.cpp 
#include "rxcpp/rx.hpp" 
namespace rx { 
    using namespace rxcpp;  
    using namespace rxcpp::operators; 
    using namespace rxcpp::sources; 
    using namespace rxcpp::util; 
} 

template<typename Container> 
rx::observable<std::string> helloNames(Container items) { 
    auto str = rx::observable<>::iterate(items); 
    return str. 
    filter([](std::string s){ 
        return s.length() > 5; 
    }). 
    map([](std::string s){ 
        return "Hello, " + s + "!"; 
    }). 
    //------ Translating exception 
    on_error_resume_next([](std::exception_ptr){ 
        return rx::error<std::string>(std::runtime_error("custom exception")); 
    }); 
} 
```

`helloNames()`函数接受任何标准库容器并创建一个字符串类型的 Observable（`observable<string>`）。然后对 Observable 进行过滤，以获取长度超过五个字符的项目，并在每个项目前加上`Hello`字符串。发生的异常将通过使用标准 RxCpp 运算符`on_error_resume_next()`进行转换：现在，让我们编写主程序来看看如何使用这个运算符：

```cpp
int main() { 
    //------ Create an observable composing the custom operator 
    auto names = {"Praseed", "Peter", "Joseph", "Sanjay"}; 
    auto value = helloNames(names).take(2); 

    auto error_handler = = { 
        try { rethrow_exception(e); } 
        catch (const std::exception &ex) { 
            std::cerr << ex.what() << std::endl; 
        } 
    }; 

    value. 
    subscribe( 
              [](std::string s){printf("OnNext: %sn", s.c_str());}, 
              error_handler, 
              [](){printf("OnCompletedn");}); 
} 
```

名字列表作为参数传递到新定义的运算符中，我们得到以下输出：

```cpp
OnNext: Hello, Praseed! 
OnNext: Hello, Joseph! 
OnCompleted
```

# 编写自定义转换运算符

让我们编写一个简单的程序，通过组合其他运算符来实现一个自定义运算符，在这个程序中，我们过滤奇数的数字流，将数字转换为其平方，并仅取流中的前三个元素：

```cpp
//------ CustomOperator1.cpp 
#include "rxcpp/rx.hpp" 
namespace rx { 
    using namespace rxcpp; 
    using namespace rxcpp::operators; 
    using namespace rxcpp::sources; 
    using namespace rxcpp::util; 
} 
//------ operator to filter odd number, find square & take first three items 
std::function<rx::observable<int>(rx::observable<int>)> getOddNumSquare() { 
    return [](rx::observable<int> item) { 
        return item. 
        filter([](int v){ return v%2; }). 
        map([](const int v) { return v*v; }). 
        take(3). 
        //------ Translating exception 
        on_error_resume_next([](std::exception_ptr){ 
            return rx::error<int>(std::runtime_error("custom exception")); }); 
    }; 
} 
int main() { 
    //------ Create an observable composing the custom operator 
    auto value = rxcpp::observable<>::range(1, 7) | 
    getOddNumSquare(); 
    value. 
    subscribe( 
              [](int v){printf("OnNext: %dn", v);}, 
              [](){printf("OnCompletedn");}); 
} 
```

在这个例子中，自定义运算符是用不同的方法实现的。运算符函数不是返回所需类型的简单 Observable，而是返回一个接受并返回*int*类型的 Observable 的函数对象。这允许用户使用管道(`|`)运算符执行高阶函数的执行。在编写复杂程序时，使用用户定义的转换实现自定义运算符并将其与现有运算符组合在一起非常方便。通常最好通过组合现有运算符来组合新运算符，而不是从头实现新运算符（不要重复造轮子！）。

# 编写涉及调度程序的自定义运算符

RxCpp 库默认是单线程的，RxCpp 将在调用订阅方法的线程中安排执行。有一些运算符接受调度程序作为参数，执行可以在调度程序管理的线程中进行。让我们编写一个程序来实现一个自定义运算符，以处理调度程序参数：

```cpp
//----------- CustomOperatorScheduler.cpp 
#include "rxcpp/rx.hpp" 
template <typename Duration> 
auto generateObservable(Duration durarion) { 
    //--------- start and the period 
    auto start = rxcpp::identity_current_thread().now(); 
    auto period = durarion; 
    //--------- Observable upto 3 items 
    return rxcpp::observable<>::interval(start, period).take(3); 
} 

int main() { 
    //-------- Create a coordination 
    auto coordination = rxcpp::observe_on_event_loop(); 
    //-------- Instantiate a coordinator and create a worker 
    auto worker = coordination.create_coordinator().get_worker(); 
    //----------- Create an Observable (Replay ) 
    auto values = generateObservable(std::chrono::milliseconds(2)). 
        replay(2, coordination); 
    //--------------- Subscribe first time 
    worker.schedule(& { 
        values.subscribe([](long v) { printf("#1 -- %d : %ldn", 
            std::this_thread::get_id(), v); }, 
                         []() { printf("#1 --- OnCompletedn"); }); 
    }); 
    worker.schedule(& { 
        values.subscribe([](long v) { printf("#2 -- %d : %ldn", 
            std::this_thread::get_id(), v); }, 
                         []() { printf("#2 --- OnCompletedn"); }); }); 
    //----- Start the emission of values 
    worker.schedule(& { 
        values.connect(); 
    }); 
    //------- Add blocking subscription to see results 
    values.as_blocking().subscribe(); 
    return 0; 
} 
```

# 编写可以链式组合的自定义运算符

RxCpp 库提供的内置运算符的一个关键优点是可以使用流畅的接口链式操作运算符。这显著提高了代码的可读性。到目前为止，我们创建的自定义运算符可以组合在一起，但不能像标准运算符那样链式组合。在本节中，我们将实现可以使用以下方法进行链式组合的运算符：

+   使用`lift<T>`元运算符

+   通过向 RxCpp 库添加代码来编写新运算符

# 使用 lift<t>运算符编写自定义运算符

RxCpp 库中的`observable<T>`实现中有一个名为`lift`（`lift<t>`）的操作符。实际上，它可以被称为元操作符，因为它具有将接受普通变量（`int`、`float`、`double`、`struct`等）的`一元`函数或函数对象转换为兼容处理`observable<T>`流的能力。`observable<T>::lift`的 RxCpp 实现期望一个 Lambda，该 Lambda 以`rxcpp::subscriber<T>`作为参数，并且在 Lambda 的主体内，我们可以应用一个操作（Lambda 或函数）。在本节中，可以对`lift<t>`操作符的目的有一个概述。

lift 操作符接受任何函数或 Lambda，该函数或 Lambda 将接受 Observable 的 Subscriber 并产生一个新的 Subscriber。这旨在允许使用`make_subscriber`的外部定义的操作符连接到组合链中。lift 的函数原型如下：

```cpp
template<class ResultType , class operator > 
auto rxcpp::operators::lift(Operator && op) -> 
                 detail::lift_factory<ResultType, operator> 
```

`lift<t>`期望的 Lambda 的签名和主体如下：

```cpp
={ 
         return rxcpp::make_subscriber<T>( 
                dest,rxcpp::make_observer_dynamic<T>( 
                      ={ 
                         //---- Apply an action Lambda on each items 
                         //---- typically "action_lambda" is declared in the 
                         //---- outside scope (captured)
                         dest.on_next(action_lambda(n)); 
                      }, 
                      ={dest.on_error(e);}, 
                      [=](){dest.on_completed();})); 
}; 
```

为了理解`lift<T>`操作符的工作原理，让我们编写一个使用它的程序。`lift<T>`的优势在于所创建的操作符可以成为 RxCpp 库的操作符链式结构的一部分。

```cpp
//----------- operatorLiftFirst.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace rxcpp; 
using namespace rxcpp::operators; 

int main() { 
     std::array< int ,4 > a={{10, 20,30,40}}; 
     //////////////////////////////////////////////////// 
     // The following Lambda will be lifted  
     auto lambda_fn = [] ( int n ) { return n*2; }; 
     ///////////////////////////////////////////////////////////// 
     // The following Lambda expects a rxcpp::subscriber and returns 
     // a subscriber which implements on_next,on_error,on_completed 
     // The Lambda lifting happens because, we apply lambda_fn on  
     // each item. 
     auto transform = ={ 
         return rxcpp::make_subscriber<int>( 
                dest,rxcpp::make_observer_dynamic<int>( 
                      ={ 
                         dest.on_next(lambda_fn(n)); 
                      }, 
                      ={dest.on_error(e);}, 
                      [=](){dest.on_completed();})); 
     }; 
     // type of values will be observable<int> 
     // Lazy Evaluation  
     auto values = rxcpp::observable<>::iterate(a);  
     //-------- As usual subscribe  
     values.lift<int>(transform).subscribe(  
              [] (int f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

我们现在已经学会了如何使用`lift<t>`操作符。`observable<T>`实例及其 lift 方法接受具有特定参数类型的 Lambda 并产生一个`observable<T>`。`lift<T>`的优势在于我们可以使用操作符链式结构。

# 将任意 Lambda 转换为自定义 Rx 操作符

在前一节中，我们了解到可以使用`lift<t>`操作符来实现自定义操作符，这些操作符可以成为 RxCpp 库的操作符链式结构的一部分。`lift<T>`的工作有点复杂，我们将编写一个`Adapter`类来将接受基本类型参数的任意 Lambda 转换为`lift<T>`操作符可以应用的形式。

适配器代码将帮助我们进行这样的调用：

```cpp
observable<T>::lift<T>( liftaction( lambda<T> ) )
```

让我们编写一个`Adapter`类实现和一个通用函数包装器，以便在程序中使用：

```cpp
//----------- operatorLiftSecond.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace rxcpp; 
using namespace rxcpp::operators; 
///////////////////////////////////////////////// 
// The LiftAction class  ( an adapter class) converts an Action ( a Lambda ) 
// and wraps it into a form which can help us to connect 
// to an observable<T> using the observable<T>::lift<T> method.  
template<class Action> 
struct LiftAction { 
    typedef typename std::decay<Action>::type action_type; 
    action_type action; 

    LiftAction(action_type t): action(t){} 
    ////////////////////////////////////// 
    // Create an Internal observer to gather  
    // data from observable<T>  
    // 
    template<class Subscriber> 
    struct action_observer : public  
              rxcpp::observer_base<typename  
              std::decay<Subscriber>::type::value_type> 
    { 
        ///////////////////////////////////////////// 
        // typedefs for  
        //        * this_type (action_observer) 
        //        * base_type (observable_base)  
        //        * value_type  
        //        * dest_type 
        //        * observer_type 
        typedef action_observer<Subscriber> this_type; 
        typedef rxcpp::observer_base<typename             
                std::decay<Subscriber>::type::value_type> base_type; 
        typedef typename base_type::value_type value_type; 
        typedef typename std::decay<Subscriber>::type dest_type; 
        typedef rxcpp::observer<value_type, this_type> observer_type; 

        //------ destination subscriber and action 
        dest_type dest; 
        action_type action; 
        action_observer(dest_type d, action_type t) 
            : dest(d), action(t){} 

        //--------- subscriber/observer methods 
        //--------  on_next implementation needs more  
        //--------- robustness by supporting exception handling 
        void on_next(typename dest_type::value_type v) const  
        {dest.on_next(action(v));} 
        void on_error(std::exception_ptr e) const  
        { dest.on_error(e);} 
        void on_completed() const { 
            dest.on_completed(); 
        } 
        //--------- Create a subscriber with requisite parameter 
        //--------- types 
        static rxcpp::subscriber<value_type, observer_type>  
                 make(const dest_type& d, const action_type& t) { 
            return rxcpp::make_subscriber<value_type> 
                 (d, observer_type(this_type(d, t))); 
        } 
    }; 
```

在 RxCpp 操作符实现中，我们将有一个内部 Observer 拦截流量，并在将控制传递给链中的下一个操作符之前对项目应用一些逻辑。`action_observer`类就是按照这些方式结构的。由于我们使用 Lambda（延迟评估），只有当调度程序触发执行时，流水线中接收到数据时才会发生执行：

```cpp
    template<class Subscriber> 
    auto operator()(const Subscriber& dest) const 
        -> decltype(action_observer<Subscriber>::make(dest, action)) { 
        return      action_observer<Subscriber>::make(dest, action); 
    } 
}; 
////////////////////////////////////// 
// liftaction takes a Universal reference  
// and uses perfect forwarding  
template<class Action> 
auto liftaction(Action&& p) ->  LiftAction<typename std::decay<Action>::type> 
{  
   return  LiftAction<typename  
           std::decay<Action>::type>(std::forward<Action>(p)); 
} 
```

现在我们已经学会了如何实现`Adapter`类以将 Lambda 转换为`lift<T>`可以接受的形式，让我们编写一个程序来演示如何利用前面的代码：

```cpp
int main() { 
     std::array< int ,4 > a={{10, 20,30,40}}; 
     auto h = [] (observable<int> src ) { 
         return src.map([](int n ) { return n; }); 
     }; 
     auto g = [] (observable<int> src ) { 
         return src.map([](int n ) { return n*2; }); 
     }; 
     // type of values will be observable<int> 
     // Lazy Evaluation  ... the Lift operator 
     // converts a Lambda to be part of operator chaining
     auto values = g(h(rxcpp::observable<>::iterate(a))) 
       .lift<int> (liftaction( [] ( int r ) { return 2*r; }));  
     //-------- As usual subscribe  
     values.subscribe(  
              [] (int f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

程序的输出如下：

```cpp
40 
80 
120 
160 
Hello World.. 
```

# 在库中创建自定义 RxCpp 操作符

`RxCpp`库中的每个操作符都在`rxcpp::operators`命名空间下定义。在`rxcpp::operators`命名空间内，库设计者创建了一个名为 details 的嵌套命名空间，其中通常指定了操作符逻辑的实现。为了演示从头开始实现操作符，我们克隆了 map 操作符的实现，创建了另一个名为`eval`的操作符。`eval`的语义与`map`操作符相同。源代码清单可在与本书相关的 GitHub 存储库中的特定章节文件夹中找到。

我们决定将书中的代码移动到 GitHub 存储库，因为清单有点长，对于理解在`RxCpp`库中实现操作符的概念没有太大贡献。前面概述的`liftaction`实现向我们展示了如何编写内部 Observer。每个操作符实现都遵循一个标准模式：

+   它通过创建一个私有 Observer 订阅源 Observable

+   根据操作符的目的转换 Observable 的元素

+   将转换后的值推送给其自己的订阅者

`eval`运算符实现的骨架源代码如下。源文件的实现包括以下内容：

| **源文件** | **关键更改** |
| --- | --- |

| `rx-eval.hpp` | `eval`运算符的实现：

```cpp

//rx-eval.hpp   
#if   !defined(RXCPP_OPERATORS_RX_EVAL_HPP)   
#define   RXCPP_OPERATORS_RX_EVAL_HPP   
//------------ all headers are   included here   
#include "../rx-includes.hpp"   
namespace rxcpp {   
    namespace operators {   
        namespace detail {   
          //-------------- operator   implementation goes here   
        }
    }
}
#endif   

```

|

| `rx-includes.h` | 修改后的头文件，包含了`Rx-eval.hpp`的引入。`rx-includes.h`将在文件中添加一个额外的条目，如下所示：

```cpp
#include "operators/rx-eval.hpp"   
```

|

| `rx-operators.h` | 修改后的头文件，包含了`eval_tag`的定义。`rx-operators.h`包含以下标签条目：

```cpp
struct eval_tag {   
    template<class Included>   
    struct include_header{   
          static_assert(Included::value, 
           "missing include: please 
                   #include   <rxcpp/operators/rx-eval.hpp>");   
};   
};   
```

|

| `rx-observables.h` | 修改后的头文件，其中包含`eval`运算符的定义：

```cpp
template<class... AN>   
auto eval(AN&&... an)   const-> decltype(observable_member(eval_tag{},   
 *(this_type*)nullptr,   std::forward<AN>(an)...)){   
        return    observable_member(eval_tag{},                 
                   *this, std::forward<AN>(an)...);   
}   
```

|

让我们编写一个使用`eval`运算符的程序。`eval`运算符的原型（类似于`map`）如下：

```cpp
observaable<T>::eval<T>( lambda<T>)
```

你可以检查实现的源代码，以更好地理解`eval`运算符。现在，让我们编写一个利用`eval`运算符的程序：

```cpp
//----------- operatorComposeCustom.cpp 
#include "rxcpp/rx.hpp" 
#include "rxcpp/rx-test.hpp" 
#include <iostream> 
namespace rxu=rxcpp::util; 
#include <array> 
using namespace std; 
using namespace rxcpp; 
using namespace rxcpp::operators; 
int main() { 
     std::array< string ,4 > a={{"Bjarne","Kirk","Herb","Sean"}}; 
     auto h = [] (observable<string> src ) { 
          return src.eval([](string s ) { return s+"!"; }); 
     }; 
     //-------- We will Lift g using eval 
     auto g = [](string s) { return "Hello : " + s; }; 
     // use apply h first and then call eval 
     auto values = h(rxcpp::observable<>::iterate(a)).eval(g);  
     //-------- As usual subscribe  
     values.subscribe(  
              [] (string f) { std::cout << f <<  std::endl; } ,  
              [] () {std::cout << "Hello World.." << std::endl;} ); 
} 
```

程序的输出如下：

```cpp
Hello : Bjarne! 
Hello : Kirk! 
Hello : Herb! 
Hello : Sean! 
Hello World.. 
```

编写以通用方式实现的自定义运算符需要对 RxCpp 内部有深入的了解。在尝试自定义运算符之前，您需要了解一些基本运算符的实现。我们编写的运算符可以成为您实现此类运算符的起点。再次强调，从头开始编写自定义运算符应该是最后的选择！

# 摘要

在本章中，我们学习了如何编写自定义运算符。我们首先编写了可以执行基本任务的简单运算符。尽管我们编写的运算符（最初）是可组合的，但我们无法像标准的 RxCpp 运算符那样将它们链接在一起。在编写了不同类型的运算符之后，我们使用`lift<T>`元运算符实现了可链接的自定义运算符。最后，我们看到了如何将运算符添加到`observable<T>`中。在下一章中，我们将深入探讨 Rx 编程的设计模式和习惯用法。我们将从 GOF 设计模式开始，并实现不同的响应式编程模式。
