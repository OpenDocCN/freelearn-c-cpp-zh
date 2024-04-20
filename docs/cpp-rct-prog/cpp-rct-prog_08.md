# RxCpp - 关键元素

在上一章中，我们介绍了 RxCpp 库及其编程模型。我们编写了一些程序来了解库的工作原理，并介绍了 RxCpp 库的最基本元素。在本章中，我们将深入介绍响应式编程的关键元素，包括以下内容：

+   Observables

+   观察者及其变体（订阅者）

+   主题

+   调度程序

+   操作符

实际上，响应式编程的关键方面如下：

+   Observables 是观察者可以订阅以获取通知的流

+   主题是 Observable 和 Observer 的组合

+   调度程序执行与操作符相关的操作，并帮助数据从 Observables 流向 Observers

+   操作符是接受 Observable 并发出另一个 Observable 的函数（嗯，几乎是！）

# Observables

在上一章中，我们从头开始创建了 Observables 并订阅了这些 Observables。在我们的所有示例中，Observables 创建了`Producer`类的实例（数据）。`Producer`类产生一个事件流。换句话说，Observables 是将订阅者（观察者）连接到生产者的函数。

在我们继续之前，让我们剖析一下 Observable 及其相关的核心活动：

+   Observable 是一个以 Observer 作为参数并返回函数的函数

+   Observable 将 Observer 连接到 Producer（Producer 对 Observer 是不透明的）

+   生产者是 Observable 的值来源

+   观察者是一个具有`on_next`、`on_error`和`on_completed`方法的对象

# 生产者是什么？

简而言之，生产者是 Observable 的值来源。生产者可以是 GUI 窗口、定时器、WebSockets、DOM 树、集合/容器上的迭代器等。它们可以是任何可以成为值来源并传递给 Observer 的值的东西（在`RxCpp`中，`observer.on_next(value)`）。当然，值可以传递给操作符，然后传递给操作符的内部观察者。

# 热 Observable 与冷 Observable

在上一章的大多数示例中，我们看到 Producers 是在 Observable 函数中创建的。生产者也可以在 Observable 函数之外创建，并且可以将对生产者的引用放在 Observable 函数内。引用到在其范围之外创建的生产者的 Observable 称为热 Observable。任何我们在 Observable 中创建了生产者实例的 Observable 称为冷 Observable。为了搞清楚问题，让我们编写一个程序来演示冷 Observable：

```cpp
//---------- ColdObservable.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
int main(int argc, char *argv[])  
{
 //----------- Get a Coordination 
 auto eventloop = rxcpp::observe_on_event_loop(); 
 //----- Create a Cold Observable 
 auto values = rxcpp::observable<>::interval( 
               std::chrono::seconds(2)).take(2);
```

在上面的代码中，interval 方法创建了一个冷 Observable，因为事件流的生产者是在`interval`函数中实例化的。当订阅者或观察者附加到冷 Observable 时，它将发出数据。即使在两个观察者之间订阅存在延迟，结果也将是一致的。这意味着我们将获得 Observable 发出的所有数据的两个观察者：

```cpp
 //----- Subscribe Twice

values.subscribe_on(eventloop). 
    subscribe([](int v){printf("[1] onNext: %dn", v);}, 
        [](){printf("[1] onCompleted\n");}); 
 values.subscribe_on(eventloop). 
    subscribe([](int v){printf("[2] onNext: %dn", v);}, 
        [](){printf("[2] onCompleted\n");}); 
  //---- make a blocking subscription to see the results 
 values.as_blocking().subscribe(); 
 //----------- Wait for Two Seconds 
 rxcpp::observable<>::timer(std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 
} 
```

程序发出的输出如下。对于每次运行，控制台中内容的顺序可能会改变，因为我们在同一线程中调度执行观察者方法。但是，由于订阅延迟，不会有数据丢失：

```cpp
[1] onNext: 1 
[2] onNext: 1 
[2] onNext: 2 
[1] onNext: 2 
[2] onCompleted 
[1] onCompleted 
```

# 热 Observable

我们可以通过调用 Observable 的`publish`方法将冷 Observable 转换为热 Observable。将冷 Observable 转换为热 Observable 的后果是数据可能会被后续的订阅所错过。热 Observable 会发出数据，无论是否有订阅。以下程序演示了这种行为：

```cpp
//---------- HotObservable.cpp

#include <rxcpp/rx.hpp> 
#include <memory> 
int main(int argc, char *argv[]) { 
 auto eventloop = rxcpp::observe_on_event_loop(); 
 //----- Create a Cold Observable 
 //----- Convert Cold Observable to Hot Observable  
 //----- using .Publish(); 
 auto values = rxcpp::observable<>::interval( 
               std::chrono::seconds(2)).take(2).publish();   
 //----- Subscribe Twice 
 values. 
    subscribe_on(eventloop). 
    subscribe( 
        [](int v){printf("[1] onNext: %dn", v);}, 
        [](){printf("[1] onCompletedn");}); 
  values. 
    subscribe_on(eventloop). 
    subscribe( 
        [](int v){printf("[2] onNext: %dn", v);}, 
        [](){printf("[2] onCompletedn");}); 
 //------ Connect to Start Emitting Values 
 values.connect(); 
 //---- make a blocking subscription to see the results 
 values.as_blocking().subscribe(); 
 //----------- Wait for Two Seconds 
 rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 
} 
```

在下一个示例中，我们将看一下`RxCpp 库`支持的`publish_synchronized`机制。从编程接口的角度来看，这只是一个小改变。看一下以下程序：

```cpp
//---------- HotObservable2.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 

int main(int argc, char *argv[]) { 

 auto eventloop = rxcpp::observe_on_event_loop(); 
 //----- Create a Cold Observable 
 //----- Convert Cold Observable to Hot Observable  
 //----- using .publish_synchronized(); 
 auto values = rxcpp::observable<>::interval( 
               std::chrono::seconds(2)). 
               take(5).publish_synchronized(eventloop);   
 //----- Subscribe Twice 
 values. 
    subscribe( 
        [](int v){printf("[1] onNext: %dn", v);}, 
        [](){printf("[1] onCompletedn");}); 

 values. 
    subscribe( 
        [](int v){printf("[2] onNext: %dn", v);}, 
        [](){printf("[2] onCompletedn");}); 

 //------ Start Emitting Values 
 values.connect(); 
 //---- make a blocking subscription to see the results 
 values.as_blocking().subscribe(); 

 //----------- Wait for Two Seconds 
 rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 
} 
```

程序的输出如下。我们可以看到输出很好地同步，即输出按正确的顺序显示：

```cpp
[1] onNext: 1 
[2] onNext: 1 
[1] onNext: 2 
[2] onNext: 2 
[1] onNext: 3 
[2] onNext: 3 
[1] onNext: 4 
[2] onNext: 4 
[1] onNext: 5 
[2] onNext: 5 
[1] onCompleted 
[2] onCompleted
```

# 热可观察对象和重放机制

热可观察对象会发出数据，无论是否有订阅者可用。这在我们期望订阅者持续接收数据的情况下可能会成为问题。在响应式编程中有一种机制可以缓存数据，以便稍后的订阅者可以被通知可观察对象的可用数据。我们可以使用`.replay()`方法来创建这样的可观察对象。让我们编写一个程序来演示重放机制，这在编写涉及热可观察对象的程序时非常有用：

```cpp
//---------- ReplayAll.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
int main(int argc, char *argv[]) { 

  auto values = rxcpp::observable<>::interval( 
                std::chrono::milliseconds(50),  
                rxcpp::observe_on_new_thread()). 
                take(5).replay(); 
    // Subscribe from the beginning 
    values.subscribe( 
        [](long v){printf("[1] OnNext: %ldn", v);}, 
        [](){printf("[1] OnCompletedn");}); 
    // Start emitting 
    values.connect(); 
    // Wait before subscribing 
    rxcpp::observable<>::timer( 
         std::chrono::milliseconds(125)).subscribe(&{ 
        values.as_blocking().subscribe( 
            [](long v){printf("[2] OnNext: %ldn", v);}, 
            [](){printf("[2] OnCompletedn");}); 
    }); 
 //----------- Wait for Two Seconds 
 rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 

} 
```

在编写响应式程序时，您确实需要了解热和冷可观察对象之间的语义差异。我们只是涉及了这个主题的一些方面。请参考 RxCpp 文档和 ReactiveX 文档以了解更多关于热和冷可观察对象的信息。互联网上有无数关于这个主题的文章。

# 观察者及其变体（订阅者）

观察者订阅可观察对象并等待事件通知。观察者已经在上一章中介绍过了。因此，我们将专注于订阅者，它们是观察者和订阅的组合。订阅者有取消订阅观察者的功能，而“普通”观察者只能订阅。以下程序很好地解释了这些概念：

```cpp
//---- Subscriber.cpp 
#include "rxcpp/rx.hpp" 
int main() { 
     //----- create a subscription object 
     auto subscription = rxcpp::composite_subscription(); 
     //----- Create a Subscription  
     auto subscriber = rxcpp::make_subscriber<int>( 
        subscription, 
        &{ 
            printf("OnNext: --%dn", v); 
            if (v == 3) 
                subscription.unsubscribe(); // Demonstrates Un Subscribes 
        }, 
        [](){ printf("OnCompletedn");}); 

    rxcpp::observable<>::create<int>( 
        [](rxcpp::subscriber<int> s){ 
            for (int i = 0; i < 5; ++i) { 
                if (!s.is_subscribed())  
                    break; 
                s.on_next(i); 
           } 
            s.on_completed();   
    }).subscribe(subscriber); 
    return 0; 
} 
```

对于使用并发和动态性（异步时间变化事件）编写复杂程序，订阅和取消订阅的能力非常方便。通过查阅 RxCpp 文档来更深入地了解这个主题。

# 主题

主题是既是观察者又是可观察对象的实体。它有助于从一个可观察对象（通常）传递通知给一组观察者。我们可以使用主题来实现诸如缓存和数据缓冲之类的复杂技术。我们还可以使用主题将热可观察对象转换为冷可观察对象。在`RxCpp 库`中实现了四种主题的变体。它们如下：

+   `SimpleSubject`

+   行为主题

+   `ReplaySubject`

+   `SynchronizeSubject`

让我们编写一个简单的程序来演示主题的工作。代码清单将演示如何将数据推送到主题并使用主题的观察者端检索它们。

```cpp
//------- SimpleSubject.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
int main(int argc, char *argv[]) { 
    //----- Create an instance of Subject 
    rxcpp::subjects::subject<int> subject; 
    //----- Retreive the Observable  
    //----- attached to the Subject 
    auto observable = subject.get_observable(); 
    //------ Subscribe Twice 
    observable.subscribe( [] ( int v ) { printf("1------%dn",v ); }); 
    observable.subscribe( [] ( int v ) { printf("2------%dn",v );}); 
    //--------- Get the Subscriber Interface 
    //--------- Attached to the Subject 
    auto subscriber = subject.get_subscriber(); 
    //----------------- Emit Series of Values 
    subscriber.on_next(1); 
    subscriber.on_next(4); 
    subscriber.on_next(9); 
    subscriber.on_next(16); 
    //----------- Wait for Two Seconds 
    rxcpp::observable<>::timer(std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 
}
```

`BehaviorSubject`是 Subject 的一种变体，它作为其实现的一部分存储最后发出的（当前）值。任何新的订阅者都会立即获得*当前值*。否则，它的行为就像一个普通的 Subject。`BehaviorSubject`在某些领域中也被称为属性或单元。它在我们更新特定单元或内存区域的一系列数据时非常有用，比如在事务上下文中。让我们编写一个程序来演示`BehaviorSubject`的工作原理：

```cpp
//-------- BehaviorSubject.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 

int main(int argc, char *argv[]) { 

    rxcpp::subjects::behavior<int> behsubject(0); 

    auto observable = behsubject.get_observable(); 
    observable.subscribe( [] ( int v ) { 
        printf("1------%dn",v ); 
     }); 

     observable.subscribe( [] ( int v ) { 
        printf("2------%dn",v ); 
     }); 

    auto subscriber = behsubject.get_subscriber(); 
    subscriber.on_next(1); 
    subscriber.on_next(2); 

    int n = behsubject.get_value(); 

    printf ("Last Value ....%dn",n); 

} 
```

`ReplaySubject`是 Subject 的一种变体，它存储已经发出的数据。我们可以指定参数来指示主题必须保留多少个值。在处理热可观察对象时，这非常方便。各种重放重载的函数原型如下：

```cpp
replay (Coordination cn,[optional] composite_subscription cs) 
replay (std::size_t count, Coordination cn, [optional]composite_subscription cs) 
replay (duration period, Coordination cn, [optional] composite_subscription cs) 
replay (std::size_t count, duration period, Coordination cn,[optional] composite_subscription cs).
```

让我们编写一个程序来理解`ReplaySubject`的语义：

```cpp
//------------- ReplaySubject.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
int main(int argc, char *argv[]) { 
    //----------- instantiate a ReplaySubject 
    rxcpp::subjects::replay<int,rxcpp::observe_on_one_worker>       
           replay_subject(10,rxcpp::observe_on_new_thread()); 
    //---------- get the observable interface 
    auto observable = replay_subject.get_observable(); 
    //---------- Subscribe! 
    observable.subscribe( [] ( int v ) {printf("1------%dn",v );}); 
    //--------- get the subscriber interface 
    auto subscriber = replay_subject.get_subscriber(); 
    //---------- Emit data  
    subscriber.on_next(1); 
    subscriber.on_next(2); 
    //-------- Add a new subscriber 
    //-------- A normal subject will drop data 
    //-------- Replay subject will not 
    observable.subscribe( [] ( int v ) {  printf("2------%dn",v );}); 
     //----------- Wait for Two Seconds 
    rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 
} 
```

在本节中，我们介绍了主题的三种变体。主要用例是通过使用可观察接口从不同来源获取事件和数据，并允许一组订阅者消耗获取的数据。`SimpleSubject`可以作为可观察对象和观察者来处理一系列值。`BehaviorSubject`用于监视一段时间内属性或变量的变化，而`ReplaySubject`将帮助您避免由于订阅延迟而导致的数据丢失。最后，`SynchronizeSubject`是一个具有同步逻辑的主题。

# 调度器

`RxCpp`库拥有一个声明性的线程机制，这要归功于其内置的强大调度子系统。从一个 Observable 中，数据可以通过不同的路径流经变化传播图。通过给流处理管道提供提示，我们可以在相同线程、不同线程或后台线程中安排操作符和观察者方法的执行。这有助于更好地捕捉程序员的意图。

`RxCpp`中的声明性调度模型是可能的，因为操作符实现中的流是不可变的。流操作符将一个 Observable 作为参数，并返回一个新的 Observable 作为结果。输入参数根本没有被改变（这种行为从操作符的实现中隐含地期望）。这有助于无序执行。`RxCpp`的调度子系统包含以下构造（特定于 Rxcpp v2）：

+   调度程序

+   Worker

+   协调

+   协调员

+   可调度的

+   时间线

`RxCpp`的第 2 版从`RxJava`系统中借用了其调度架构。它依赖于`RxJava`使用的调度程序和 Worker 习语。以下是关于调度程序的一些重要事实：

+   调度程序有一个时间线。

+   调度程序可以在时间线上创建许多 Worker。

+   Worker 拥有时间线上的可调度队列。

+   `schedulable`拥有一个函数（通常称为`Action`）并拥有生命周期。

+   `Coordination`函数作为协调员的工厂，并拥有一个调度程序。

+   每个协调员都有一个 Worker，并且是以下内容的工厂：

+   协调的`schedulable`

+   协调的 Observables 和订阅者

我们一直在程序中使用 Rx 调度程序，而不用担心它们在幕后是如何工作的。让我们编写一个玩具程序，来帮助我们理解调度程序在幕后是如何工作的：

```cpp
//------------- SchedulerOne.cpp 
#include "rxcpp/rx.hpp" 
int main(){ 
    //---------- Get a Coordination  
    auto Coordination function= rxcpp::serialize_new_thread(); 
    //------- Create a Worker instance  through a factory method  
    auto worker = coordination.create_coordinator().get_worker(); 
    //--------- Create a action object 
    auto sub_action = rxcpp::schedulers::make_action( 
         [] (const rxcpp::schedulers::schedulable&) {   
          printf("Action Executed in Thread # : %dn",  
          std::this_thread::get_id());   
          } );  
    //------------- Create a schedulable and schedule the action 
    auto scheduled = rxcpp::schedulers::make_schedulable(worker,sub_action); 
    scheduled.schedule(); 
    return 0; 
} 
```

在`RxCpp`中，所有接受多个流作为输入或涉及对时间有影响的任务的操作符都将`Coordination`函数作为参数。一些使用特定调度程序的`Coordination`函数如下：

+   `identity_immediate()`

+   `identity_current_thread()`

+   `identity_same_worker(worker w)`

+   `serialize_event_loop()`

+   `serialize_new_thread()`

+   `serialize_same_worker(worker w)`

+   `observe_on_event_loop()`

+   `observe_on_new_thread()`

在前面的程序中，我们手动安排了一个操作（实际上只是一个 lambda）。让我们继续调度程序的声明方面。我们将编写一个使用`Coordination`函数安排任务的程序：

```cpp
//----------- SchedulerTwo.cpp 
#include "rxcpp/rx.hpp" 
int main(){ 
    //-------- Create a Coordination function 
    auto Coordination function= rxcpp::identity_current_thread(); 
    //-------- Instantiate a coordinator and create a worker     
    auto worker = coordination.create_coordinator().get_worker(); 
    //--------- start and the period 
    auto start = coordination.now() + std::chrono::milliseconds(1); 
    auto period = std::chrono::milliseconds(1);      
    //----------- Create an Observable (Replay ) 
    auto values = rxcpp::observable<>::interval(start,period). 
    take(5).replay(2, coordination); 
    //--------------- Subscribe first time using a Worker 
    worker.schedule(&{ 
       values.subscribe( [](long v){ printf("#1 -- %d : %ldn",  
                   std::this_thread::get_id(),v);  }, 
                        [](){ printf("#1 --- OnCompletedn");}); 
    }); 
    worker.schedule(&{ 
      values.subscribe( [](long v){printf("#2 -- %d : %ldn",  
                   std::this_thread::get_id(),v); }, 
                     [](){printf("#2 --- OnCompletedn");});  
    }); 
    //----- Start the emission of values  
   worker.schedule(& 
   { values.connect();}); 
   //------- Add blocking subscription to see results 
   values.as_blocking().subscribe(); return 0; 
}
```

我们使用重放机制创建了一个热 Observable 来处理一些观察者的延迟订阅。我们还创建了一个 Worker 来进行订阅的调度，并将观察者与 Observable 连接起来。前面的程序演示了`RxCpp`中调度程序的工作原理。

# ObserveOn 与 SubscribeOn

`ObserveOn`和`SubscribeOn`操作符的行为方式不同，这一直是反应式编程新手困惑的来源。`ObserveOn`操作符改变了其下方的操作符和观察者的线程。而`SubscribeOn`则影响其上方和下方的操作符和方法。以下程序演示了`SubscribeOn`和`ObserveOn`操作符的行为方式对程序运行时行为的微妙变化。让我们编写一个使用`ObserveOn`操作符的程序：

```cpp
//-------- ObservableOnScheduler.cpp 
#include "rxcpp/rx.hpp" 
int main(){ 
    //------- Print the main thread id 
    printf("Main Thread Id is %dn",  
             std::this_thread::get_id()); 
    //-------- We are using observe_on here 
    //-------- The Map will use the main thread 
    //-------- Subscribed Lambda will use a new thread 
    rxcpp::observable<>::range(0,15). 
        map([](int i){ 
            printf("Map %d : %dn", std::this_thread::get_id(),i);  
            return i; }). 
        take(5).observe_on(rxcpp::synchronize_new_thread()). 
        subscribe(&{ 
           printf("Subs %d : %dn", std::this_thread::get_id(),i);  
        }); 
    //----------- Wait for Two Seconds 
    rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 

    return 0; 
}
```

前述程序的输出如下：

```cpp
Main Thread Id is 1 
Map 1 : 0 
Map 1 : 1 
Subs 2 : 0 
Map 1 : 2 
Subs 2 : 1 
Map 1 : 3 
Subs 2 : 2 
Map 1 : 4 
Subs 2 : 3 
Subs 2 : 4 
```

前述程序的输出清楚地显示了`map`在主线程中工作，而`subscribe`方法在次要线程中被调度。这清楚地表明`ObserveOn`只对其下方的操作符和订阅者起作用。让我们编写一个几乎相同的程序，使用`SubscribeOn`操作符而不是`ObserveOn`操作符。看一下这个：

```cpp
//-------- SubscribeOnScheduler.cpp 
#include "rxcpp/rx.hpp" 
int main(){ 
    //------- Print the main thread id 
    printf("Main Thread Id is %dn",  
             std::this_thread::get_id()); 
    //-------- We are using subscribe_on here 
    //-------- The Map and subscribed Lambda will  
    //--------- use the secondary thread 
    rxcpp::observable<>::range(0,15). 
        map([](int i){ 
            printf("Map %d : %dn", std::this_thread::get_id(),i);  
            return i; 
        }). 
        take(5).subscribe_on(rxcpp::synchronize_new_thread()). 
        subscribe(&{ 
           printf("Subs %d : %dn", std::this_thread::get_id(),i);  
        }); 
    //----------- Wait for Two Seconds 
    rxcpp::observable<>::timer( 
       std::chrono::milliseconds(2000)). 
       subscribe(&{ }); 

    return 0; 
}
```

前述程序的输出如下：

```cpp
Main Thread Id is 1 
Map 2 : 0 
Subs 2 : 0 
Map 2 : 1 
Subs 2 : 1 
Map 2 : 2 
Subs 2 : 2 
Map 2 : 3 
Subs 2 : 3 
Map 2 : 4 
Subs 2 : 4 
```

前述程序的输出显示 map 和订阅方法都在次要线程中工作。这清楚地显示了`SubscribeOn`改变了它之前和之后的项目的线程行为。

# RunLoop 调度程序

RxCpp 库没有内置的主线程调度程序的概念。你能做的最接近的是利用`run_loop`类来模拟在主线程中进行调度。在下面的程序中，Observable 在后台线程执行，订阅方法在主线程运行。我们使用`subscribe_on`和`observe_on`来实现这个目标：

```cpp
//------------- RunLoop.cpp 
#include "rxcpp/rx.hpp" 
int main(){ 
    //------------ Print the Main Thread Id 
    printf("Main Thread Id is %dn",  
                std::this_thread::get_id()); 
    //------- Instantiate a run_loop object 
    //------- which will loop in the main thread 
    rxcpp::schedulers::run_loop rlp; 
    //------ Create a Coordination functionfor run loop 
    auto main_thread = rxcpp::observe_on_run_loop(rlp); 
    auto worker_thread = rxcpp::synchronize_new_thread(); 
    rxcpp::composite_subscription scr; 
    rxcpp::observable<>::range(0,15). 
        map([](int i){ 
            //----- This will get executed in worker 
            printf("Map %d : %dn", std::this_thread::get_id(),i);  
            return i; 
        }).take(5).subscribe_on(worker_thread). 
        observe_on(main_thread). 
        subscribe(scr, &{ 
            //--- This will get executed in main thread 
            printf("Sub %d : %dn", std::this_thread::get_id(),i); }); 
    //------------ Execute the Run Loop 
    while (scr.is_subscribed() || !rlp.empty()) { 
        while (!rlp.empty() && rlp.peek().when < rlp.now()) 
        { rlp.dispatch();} 
    }  
    return 0; 
} 
```

前述程序的输出如下：

```cpp
Main Thread Id is 1 
Map 2 : 0 
Map 2 : 1 
Sub 1 : 0 
Sub 1 : 1 
Map 2 : 2 
Map 2 : 3 
Sub 1 : 2 
Map 2 : 4 
Sub 1 : 3 
Sub 1 : 4 
```

我们可以看到 map 被调度在工作线程中，订阅方法在主线程中执行。这是因为我们巧妙地放置了 subscribe_on 和 observe_on 运算符，这是我们在前一节中介绍的。

# 运算符

运算符是应用于 Observable 以产生新的 Observable 的函数。在这个过程中，原始 Observable 没有被改变，并且可以被认为是一个纯函数。我们已经在我们编写的示例程序中涵盖了许多运算符。在[第十章](https://cdp.packtpub.com/c___reactive_programming/wp-admin/post.php?post=79&action=edit#post_86)中，*在 Rxcpp 中创建自定义运算符*，我们将学习如何创建在 Observables 上工作的自定义运算符。运算符不改变（输入）Observable 的事实是声明式调度在 Rx 编程模型中起作用的原因。Rx 运算符可以被分类如下：

+   创建运算符

+   变换运算符

+   过滤运算符

+   组合运算符

+   错误处理运算符

+   实用运算符

+   布尔运算符

+   数学运算符

还有一些更多的运算符不属于这些类别。我们将提供一个来自前述类别的关键运算符列表，作为一个快速参考的表格。作为开发人员，可以根据上面给出的表格来选择运算符，根据上下文来选择运算符。

# 创建运算符

这些运算符将帮助开发人员从输入数据中创建各种类型的 Observables。我们已经在我们的示例代码中演示了 create、from、interval 和 range 运算符的使用。请参考这些示例和 RxCpp 文档以了解更多信息。下面给出了一张包含一些运算符的表格：

| **Observables** | **描述** |
| --- | --- |
| `create` | 通过以编程方式调用 Observer 方法创建一个 Observable |
| `defer` | 为每个 Observer/Subscriber 创建一个新的 Observable |
| `empty` | 创建一个不发出任何内容的 Observable（只在完成时发出） |
| `from` | 根据参数创建一个 Observable（多态） |
| `interval` | 创建一个在时间间隔内发出一系列值的 Observable |
| `just` | 创建一个发出单个值的 Observable |
| `range` | 创建一个发出一系列值的 Observable |
| `never` | 创建一个永远不发出任何内容的 Observable |
| `repeat` | 创建一个重复发出值的 Observable |
| `timer` | 创建一个在延迟因子之后发出值的 Observable，可以将其指定为参数 |
| `throw` | 创建一个发出错误的 Observable |

# 变换运算符

这些运算符帮助开发人员创建一个新的 Observable，而不修改源 Observable。它们通过在源 Observable 上应用 lambda 或函数对象来作用于源 Observable 中的单个项目。下面给出了一张包含一些最有用的变换运算符的表格。

| **Observables** | **描述** |
| --- | --- |
| `buffer` | 收集过去的值并在收到信号时发出的 Observable |
| `flat_map` | 发出应用于源 Observable 和集合 Observable 发出的一对值的函数的结果的 Observable |
| `group_by` | 帮助从 Observable 中分组值的 Observable |
| `map` | 通过指定的函数转换源 Observable 发出的项目的 Observable |
| `scan` | 发出累加器函数的每次调用的结果的 Observable |
| `window` | 发出连接的、不重叠的项目窗口的 Observable。 每个窗口将包含特定数量的项目，该数量作为参数给出。 参数名为 count。 |

# 过滤运算符

过滤流的能力是流处理中的常见活动。 Rx 编程模型定义了许多过滤类别的运算符并不罕见。 过滤运算符主要是谓词函数或 lambda。 以下表格包含过滤运算符的列表：

| **Observables** | **Description** |
| --- | --- |
| `debounce` | 如果经过一段特定的时间间隔而没有从源 Observable 发出另一个项目，则发出一个项目的 Observable |
| `distinct` | 发出源 Observable 中不同的项目的 Observable |
| `element_at` | 发出位于指定索引位置的项目的 Observable |
| `filter` | 只发出由过滤器评估为 true 的源 Observable 发出的项目的 Observable |
| `first` | 只发出源 Observable 发出的第一个项目的 Observable |
| `ignore_eleements` | 从源 Observable 发出终止通知的 Observable |
| `last` | 只发出源 Observable 发出的最后一个项目的 Observable |
| `sample` | 在周期时间间隔内发出源 Observable 发出的最近的项目的 Observable |
| `skip` | 与源 Observable 相同的 Observable，只是它不会发出源 Observable 发出的前 t 个项目 |
| `skip_last` | 与源 Observable 相同的 Observable，只是它不会发出源 Observable 发出的最后 t 个项目 |
| `take` | 只发出源 Observable 发出的前 t 个项目，或者如果该 Observable 发出的项目少于 t 个，则发出源 Observable 的所有项目 |
| `take_last` | 只发出源 Observable 发出的最后 t 个项目的 Observable |

# 组合运算符

Rx 编程模型的主要目标之一是将事件源与事件接收器解耦。 显然，需要能够组合来自各种来源的流的运算符。 RxCpp 库实现了一组此类运算符。 以下表格概述了一组常用的组合运算符：

| **Observables** | **Description** |
| --- | --- |
| `combine_latest` | 当两个 Observables 中的任一 Observable 发出项目时，通过指定的函数组合每个 Observable 发出的最新项目，并根据该函数的结果发出项目 |
| `merge` | 通过合并它们的发射将多个 Observables 合并为一个 |
| `start_with` | 在开始发出源 Observable 的项目之前，发出指定的项目序列 |
| `switch_on_next` | 将发出 Observables 的 Observable 转换为发出最近发出的 Observable 发出的项目的单个 Observable |
| `zip` | 通过指定的函数将多个 Observables 的发射组合在一起，并根据该函数的结果发出每个组合的单个项目 |

# 错误处理运算符

这些是在管道执行过程中发生异常时帮助我们进行错误恢复的运算符。

| **Observables** | **Description** |
| --- | --- |
| `Catch` | `RxCpp`不支持 |
| `retry` | 如果调用`on_error`，则会重新订阅源 Observable 的 Observable，最多重试指定次数 |

# Observable 实用程序运算符

以下是用于处理 Observables 的有用实用程序运算符工具箱： observe_on 和 subscribe_on 运算符帮助我们进行声明式调度。 我们已经在上一章中介绍过它们。

| **Observables** | **Description** |
| --- | --- |
| `finally` | Observable 发出与源 Observable 相同的项目，然后调用给定的操作 |
| `observe_on` | 指定观察者将观察此 Observable 的调度程序 |
| `subscribe` | 对 Observable 的发射和通知进行操作 |
| `subscribe_on` | 指定 Observable 订阅时应使用的调度程序 |
| `scope` | 创建与 Observable 寿命相同的一次性资源 |

# 条件和布尔运算符

条件和布尔运算符是评估一个或多个 Observable 或 Observable 发出的项目的运算符：

| **Observables** | **Description** |
| --- | --- |
| `all` | 如果源 Observable 发出的每个项目都满足指定条件，则发出 true 的 Observable；否则，它发出 false |
| `amb` | Observable 发出与源 Observables 中首先发出项目或发送终止通知的相同序列 |
| `contains` | 如果源 Observable 发出了指定的项目，则发出 true 的 Observable；否则发出 false |
| `default_if_empty` | 如果源 Observable 发出了指定的项目，则发出 true 的 Observable；否则发出 false |
| `sequence_equal` | 只有在发出相同顺序的相同项目序列后正常终止时，Observable 才会发出 true；否则，它将发出 false |
| `skip_until` | 直到第二个 Observable 发出项目之前，丢弃由 Observable 发出的项目 |
| `skip_while` | 直到指定条件变为 false 后，丢弃由 Observable 发出的项目 |
| `take_until` | 在第二个 Observable 发出项目或终止后，丢弃由 Observable 发出的项目 |
| `take_while` | 在指定条件变为 false 后，丢弃由 Observable 发出的项目 |

# 数学和聚合运算符

这些数学和聚合运算符是一类操作符，它们对 Observable 发出的整个项目序列进行操作：它们基本上将 Observable<T>减少为类型 T 的某个值。它们不会返回 Observable。

| **Observables** | **Description** |
| --- | --- |
| `average` | 计算 Observable 发出的数字的平均值并发出此平均值 |
| `concat` | 发出两个或多个 Observable 的发射，而不对它们进行交错 |
| `count` | 计算源 Observable 发出的项目数量并仅发出此值 |
| `max` | 确定并发出 Observable 发出的最大值项目 |
| `min` | 确定并发出 Observable 发出的最小值项目 |
| `reduce` | 对 Observable 发出的每个项目依次应用函数，并发出最终值 |
| `sum` | 计算 Observable 发出的数字的总和并发出此总和 |

# 可连接的 Observable 运算符

可连接的 Observable 是具有更精确控制的订阅动态的特殊 Observable。以下表格列出了一些具有高级订阅语义的关键运算符

| **Observables** | **Description** |
| --- | --- |
| `connect` | 指示可连接的 Observable 开始向其订阅者发出项目 |
| `publish` | 将普通 Observable 转换为可连接的 Observable |
| `ref_count` | 使可连接的 Observable 表现得像普通的 Observable |
| `replay` | 确保所有观察者看到相同的发出项目序列，即使它们在 Observable 开始发出项目后订阅。此运算符与热 Observable 一起使用 |

# 摘要

在本章中，我们了解了 Rx 编程模型的各个部分是如何配合的。我们从 Observables 开始，迅速转移到热和冷 Observables 的主题。然后，我们讨论了订阅机制及其使用。接着，我们转向了 Subjects 这一重要主题，并了解了多种 Scheduler 实现的工作方式。最后，我们对 RxCpp 系统中提供的各种操作符进行了分类。在下一章中，我们将学习如何利用迄今为止所学的知识，以一种反应式的方式使用 Qt 框架编写 GUI 程序。
