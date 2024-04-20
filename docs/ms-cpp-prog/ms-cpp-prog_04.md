# 第四章：智能指针

在上一章中，您了解了模板编程和泛型编程的好处。在本章中，您将学习以下智能指针主题：

+   内存管理

+   原始指针的问题

+   循环依赖

+   智能指针：

+   `auto_ptr`

+   智能指针

+   `shared_ptr`

+   `weak_ptr`

让我们探讨 C++提供的内存管理设施。

# 内存管理

在 C++中，内存管理通常是软件开发人员的责任。这是因为 C++标准不强制在 C++编译器中支持垃圾回收；因此，这取决于编译器供应商的选择。特别是，Sun C++编译器带有一个名为`libgc`的垃圾回收库。

C++语言拥有许多强大的特性。其中，指针无疑是其中最强大和最有用的特性之一。指针非常有用，但它们也有自己的奇怪问题，因此必须负责使用。当内存管理没有得到认真对待或者没有做得很好时，会导致许多问题，包括应用程序崩溃、核心转储、分段错误、难以调试的问题、性能问题等等。悬空指针或者流氓指针有时会干扰其他无关的应用程序，而罪魁祸首应用程序却悄无声息地执行；事实上，受害应用程序可能会被多次责怪。内存泄漏最糟糕的部分在于，有时会变得非常棘手，即使是经验丰富的开发人员最终也会花费数小时来调试受害代码，而罪魁祸首代码却毫发未损。有效的内存管理有助于避免内存泄漏，并让您开发内存高效的高性能应用程序。

由于每个操作系统的内存模型都不同，因此在相同的内存泄漏问题上，每个操作系统可能在不同的时间点表现不同。内存管理是一个大课题，C++提供了许多有效的方法来处理它。我们将在以下章节讨论一些有用的技术。

# 原始指针的问题

大多数 C++开发人员有一个共同点：我们都喜欢编写复杂的东西。你问一个开发人员，“嘿，伙计，你想重用已经存在并且可用的代码，还是想自己开发一个？”虽然大多数开发人员会委婉地说在可能的情况下重用已有的代码，但他们的内心会说，“我希望我能自己设计和开发它。”复杂的数据结构和算法往往需要指针。原始指针在遇到麻烦之前确实很酷。

在使用前，原始指针必须分配内存，并且在使用后需要释放内存；就是这么简单。然而，在一个产品中，指针分配可能发生在一个地方，而释放可能发生在另一个地方。如果内存管理决策没有做出正确的选择，人们可能会认为释放内存是调用者或被调用者的责任，有时内存可能不会从任何地方释放。还有另一种可能性，同一个指针可能会从不同的地方被多次删除，这可能导致应用程序崩溃。如果这种情况发生在 Windows 设备驱动程序中，很可能会导致蓝屏。

想象一下，如果出现应用程序异常，并且抛出异常的函数有一堆在异常发生前分配了内存的指针？任何人都能猜到：会有内存泄漏。

让我们看一个使用原始指针的简单例子：

```cpp
#include <iostream>
using namespace std;

class MyClass {
      public:
           void someMethod() {

                int *ptr = new int();
                *ptr = 100;
                int result = *ptr / 0;  //division by zero error expected
                delete ptr;

           }
};

int main ( ) {

    MyClass objMyClass;
    objMyClass.someMethod();

    return 0;

}
```

现在，运行以下命令：

```cpp
g++ main.cpp -g -std=c++17
```

查看此程序的输出：

```cpp
main.cpp: In member function ‘void MyClass::someMethod()’:
main.cpp:12:21: warning: division by zero [-Wdiv-by-zero]
 int result = *ptr / 0;
```

现在，运行以下命令：

```cpp
./a.out
[1] 31674 floating point exception (core dumped) ./a.out
```

C++编译器真的很酷。看看警告消息，它指出了问题。我喜欢 Linux 操作系统。Linux 在发现行为不端的恶意应用程序方面非常聪明，并且及时将它们关闭，以免对其他应用程序或操作系统造成任何损害。核心转储实际上是好事，但在庆祝 Linux 方法时却被诅咒。猜猜，微软的 Windows 操作系统同样聪明。当它们发现一些应用程序进行可疑的内存访问时，它们会进行错误检查，Windows 操作系统也支持迷你转储和完整转储，这相当于 Linux 操作系统中的核心转储。

让我们看一下 Valgrind 工具的输出，以检查内存泄漏问题：

```cpp
valgrind --leak-check=full --show-leak-kinds=all ./a.out

==32857== Memcheck, a memory error detector
==32857== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==32857== Using Valgrind-3.12.0 and LibVEX; rerun with -h for copyright info
==32857== Command: ./a.out
==32857== 
==32857== 
==32857== Process terminating with default action of signal 8 (SIGFPE)
==32857== Integer divide by zero at address 0x802D82B86
==32857== at 0x10896A: MyClass::someMethod() (main.cpp:12)
==32857== by 0x1088C2: main (main.cpp:24)
==32857== 
==32857== HEAP SUMMARY:
==32857== in use at exit: 4 bytes in 1 blocks
==32857== total heap usage: 2 allocs, 1 frees, 72,708 bytes allocated
==32857== 
==32857== 4 bytes in 1 blocks are still reachable in loss record 1 of 1
==32857== at 0x4C2E19F: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==32857== by 0x108951: MyClass::someMethod() (main.cpp:8)
==32857== by 0x1088C2: main (main.cpp:24)
==32857== 
==32857== LEAK SUMMARY:
==32857== definitely lost: 0 bytes in 0 blocks
==32857== indirectly lost: 0 bytes in 0 blocks
==32857== possibly lost: 0 bytes in 0 blocks
==32857== still reachable: 4 bytes in 1 blocks
==32857== suppressed: 0 bytes in 0 blocks
==32857== 
==32857== For counts of detected and suppressed errors, rerun with: -v
==32857== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
[1] 32857 floating point exception (core dumped) valgrind --leak-check=full --show-leak-kinds=all ./a.out
```

在这个输出中，如果你注意**粗体**部分的文本，你会注意到 Valgrind 工具指出了导致这个核心转储的源代码行号。`main.cpp`文件中的第 12 行如下：

```cpp
 int result = *ptr / 0; //division by zero error expected 
```

在`main.cpp`文件的第 12 行发生异常时，异常下面出现的代码将永远不会被执行。在`main.cpp`文件的第 13 行，由于异常，将永远不会执行`delete`语句：

```cpp
 delete ptr;
```

在堆栈展开过程中，由于指针指向的内存在堆栈展开过程中没有被释放，因此前面的原始指针分配的内存没有被释放。每当函数抛出异常并且异常没有被同一个函数处理时，堆栈展开是有保证的。然而，只有自动本地变量在堆栈展开过程中会被清理，而不是指针指向的内存。这导致内存泄漏。

这是使用原始指针引发的奇怪问题之一；还有许多其他类似的情况。希望你现在已经相信，使用原始指针的乐趣是有代价的。但所付出的代价并不值得，因为在 C++中有很好的替代方案来解决这个问题。你是对的，使用智能指针是提供使用指针的好处而不付出原始指针附加成本的解决方案。

因此，智能指针是在 C++中安全使用指针的方法。

# 智能指针

在 C++中，智能指针让你专注于手头的问题，摆脱了处理自定义垃圾收集技术的烦恼。智能指针让你安全地使用原始指针。它们负责清理原始指针使用的内存。

C++支持许多类型的智能指针，可以在不同的场景中使用：

+   `auto_ptr`

+   `unique_ptr`

+   `shared_ptr`

+   `weak_ptr`

`auto_ptr`智能指针是在 C++11 中引入的。`auto_ptr`智能指针在超出范围时自动释放堆内存。然而，由于`auto_ptr`从一个`auto_ptr`实例转移所有权的方式，它已被弃用，并且`unique_ptr`被引入作为其替代品。`shared_ptr`智能指针帮助多个共享智能指针引用同一个对象，并负责内存管理负担。`weak_ptr`智能指针帮助解决由于应用程序设计中存在循环依赖问题而导致的`shared_ptr`使用的内存泄漏问题。

还有其他类型的智能指针和相关内容，它们并不常用，并列在以下项目列表中。然而，我强烈建议你自己探索它们，因为你永远不知道什么时候会发现它们有用：

+   无主

+   `enable_shared_from_this`

+   `bad_weak_ptr`

+   `default_delete`

`owner_less`智能指针帮助比较两个或更多个智能指针是否共享相同的原始指向对象。`enable_shared_from_this`智能指针帮助获取`this`指针的智能指针。`bad_weak_ptr`智能指针是一个异常类，意味着使用无效智能指针创建了`shared_ptr`。`default_delete`智能指针指的是`unique_ptr`使用的默认销毁策略，它调用`delete`语句，同时也支持用于数组类型的部分特化，使用`delete[]`。

在本章中，我们将逐一探讨`auto_ptr`，`shared_ptr`，`weak_ptr`和`unique-ptr`。

# auto_ptr

`auto_ptr`智能指针接受一个原始指针，封装它，并确保原始指针指向的内存在`auto_ptr`对象超出范围时被释放。在任何时候，只有一个`auto_ptr`智能指针可以指向一个对象。因此，当一个`auto_ptr`指针被赋值给另一个`auto_ptr`指针时，所有权被转移到接收赋值的`auto_ptr`实例；当一个`auto_ptr`智能指针被复制时也是如此。

通过一个简单的例子来观察这些内容将会很有趣，如下所示：

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
using namespace std;

class MyClass {
      private:
           static int count;
           string name;
      public:
           MyClass() {
                 ostringstream stringStream(ostringstream::ate);
                 stringStream << "Object";
                 stringStream << ++count;
                 name = stringStream.str();
                 cout << "\nMyClass Default constructor - " << name << endl;
           }
           ~MyClass() {
                 cout << "\nMyClass destructor - " << name << endl;
           }

           MyClass ( const MyClass &objectBeingCopied ) {
                 cout << "\nMyClass copy constructor" << endl;
           }

           MyClass& operator = ( const MyClass &objectBeingAssigned ) {
                 cout << "\nMyClass assignment operator" << endl;
           }

           void sayHello( ) {
                cout << "Hello from MyClass " << name << endl;
           }
};

int MyClass::count = 0;

int main ( ) {

   auto_ptr<MyClass> ptr1( new MyClass() );
   auto_ptr<MyClass> ptr2( new MyClass() );

   return 0;

}
```

前面程序的编译输出如下：

```cpp
g++ main.cpp -std=c++17

main.cpp: In function ‘int main()’:
main.cpp:40:2: warning: ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated-declarations]
 auto_ptr<MyClass> ptr1( new MyClass() );

In file included from /usr/include/c++/6/memory:81:0,
 from main.cpp:3:
/usr/include/c++/6/bits/unique_ptr.h:49:28: note: declared here
 template<typename> class auto_ptr;

main.cpp:41:2: warning: ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated-declarations]
 auto_ptr<MyClass> ptr2( new MyClass() );

In file included from /usr/include/c++/6/memory:81:0,
 from main.cpp:3:
/usr/include/c++/6/bits/unique_ptr.h:49:28: note: declared here
 template<typename> class auto_ptr;
```

正如你所看到的，C++编译器警告我们使用`auto_ptr`已经被弃用。因此，我不建议再使用`auto_ptr`智能指针；它已被`unique_ptr`取代。

现在，我们可以忽略警告并继续，如下所示：

```cpp
g++ main.cpp -Wno-deprecated

./a.out

MyClass Default constructor - Object1

MyClass Default constructor - Object2

MyClass destructor - Object2

MyClass destructor - Object1 
```

正如你在前面的程序输出中所看到的，分配在堆中的`Object1`和`Object2`都被自动删除了。这要归功于`auto_ptr`智能指针。

# 代码演示 - 第 1 部分

从`MyClass`的定义中，你可能已经了解到，它定义了默认的`构造函数`，`复制`构造函数和析构函数，一个`赋值`运算符和`sayHello()`方法，如下所示：

```cpp
//Definitions removed here to keep it simple 
class MyClass {
public:
      MyClass() { }  //Default constructor
      ~MyClass() { } //Destructor 
      MyClass ( const MyClass &objectBeingCopied ) {} //Copy Constructor 
      MyClass& operator = ( const MyClass &objectBeingAssigned ) { } //Assignment operator
      void sayHello();
}; 
```

`MyClass`的方法只是一个打印语句，表明方法被调用；它们纯粹是为了演示目的而设计的。

`main()`函数创建了两个`auto_ptr`智能指针，它们指向两个不同的`MyClass`对象，如下所示：

```cpp
int main ( ) {

   auto_ptr<MyClass> ptr1( new MyClass() );
   auto_ptr<MyClass> ptr2( new MyClass() );

   return 0;

}
```

正如你所理解的，`auto_ptr`是一个封装了原始指针而不是指针的本地对象。当控制流达到`return`语句时，堆栈展开过程开始，作为这一过程的一部分，堆栈对象`ptr1`和`ptr2`被销毁。这反过来调用了`auto_ptr`的析构函数，最终删除了堆栈对象`ptr1`和`ptr2`指向的`MyClass`对象。

我们还没有完成。让我们探索`auto_ptr`的更多有用功能，如下所示的`main`函数：

```cpp
int main ( ) {

    auto_ptr<MyClass> ptr1( new MyClass() );
    auto_ptr<MyClass> ptr2( new MyClass() );

    ptr1->sayHello();
    ptr2->sayHello();

    //At this point the below stuffs happen
    //1\. ptr2 smart pointer has given up ownership of MyClass Object 2
    //2\. MyClass Object 2 will be destructed as ptr2 has given up its 
    //   ownership on Object 2
    //3\. Ownership of Object 1 will be transferred to ptr2
    ptr2 = ptr1;

    //The line below if uncommented will result in core dump as ptr1 
    //has given up its ownership on Object 1 and the ownership of 
    //Object 1 is transferred to ptr2.
    // ptr1->sayHello();

    ptr2->sayHello();

    return 0;

}
```

# 代码演示 - 第 2 部分

我们刚刚看到的`main()`函数代码演示了许多有用的技术和一些`auto_ptr`智能指针的争议行为。以下代码创建了两个`auto_ptr`的实例，即`ptr1`和`ptr2`，它们封装了在堆中创建的两个`MyClass`对象：

```cpp
 auto_ptr<MyClass> ptr1( new MyClass() );
 auto_ptr<MyClass> ptr2( new MyClass() );
```

接下来，以下代码演示了如何使用`auto_ptr`调用`MyClass`支持的方法：

```cpp
 ptr1->sayHello();
 ptr2->sayHello();
```

希望你注意到了`ptr1->sayHello()`语句。它会让你相信`auto_ptr` `ptr1`对象是一个指针，但实际上，`ptr1`和`ptr2`只是作为本地变量在堆栈中创建的`auto_ptr`对象。由于`auto_ptr`类重载了`->`指针运算符和`*`解引用运算符，它看起来像一个指针。事实上，`MyClass`暴露的所有方法只能使用`->`指针运算符访问，而所有`auto_ptr`方法可以像访问堆栈对象一样访问。

以下代码演示了`auto_ptr`智能指针的内部行为，所以请密切关注；这将会非常有趣：

```cpp
ptr2 = ptr1;
```

尽管上述代码看起来像是一个简单的`赋值`语句，但它在`auto_ptr`中触发了许多活动。由于前面的`赋值`语句，发生了以下活动：

+   `ptr2`智能指针将放弃对`MyClass`对象 2 的所有权。

+   `ptr2`放弃了对`object 2`的所有权，因此`MyClass`对象 2 将被销毁。

+   `object 1`的所有权将被转移到`ptr2`。

+   此时，`ptr1`既不指向`object 1`，也不负责管理`object 1`使用的内存。

以下注释行包含一些信息：

```cpp
// ptr1->sayHello();
```

由于`ptr1`智能指针已经释放了对`object 1`的所有权，因此尝试访问`sayHello()`方法是非法的。这是因为`ptr1`实际上不再指向`object 1`，而`object 1`由`ptr2`拥有。当`ptr2`超出范围时，释放`object 1`使用的内存是`ptr2`智能指针的责任。如果取消注释上述代码，将导致核心转储。

最后，以下代码让我们使用`ptr2`智能指针在`object 1`上调用`sayHello()`方法：

```cpp
ptr2->sayHello();
return 0;
```

我们刚刚看到的`return`语句将在`main()`函数中启动堆栈展开过程。这将最终调用`ptr2`的析构函数，进而释放`object 1`使用的内存。美妙的是，所有这些都是自动发生的。在我们专注于手头的问题时，`auto_ptr`智能指针在幕后为我们努力工作。

然而，由于以下原因，从`C++11`开始，`auto_ptr`已经被弃用：

+   `auto_ptr`对象不能存储在 STL 容器中

+   `auto_ptr`复制构造函数将从原始源头那里移除所有权，也就是说，``auto_ptr``

+   `auto_ptr`复制`赋值`运算符将从原始源头那里移除所有权，也就是说，`auto_ptr`

+   `auto_ptr`的复制构造函数和`赋值`运算符违反了原始意图，因为`auto_ptr`的复制构造函数和`赋值`运算符将从右侧对象中移除源对象的所有权，并将所有权分配给左侧对象

# unique_ptr

`unique_ptr`智能指针的工作方式与`auto_ptr`完全相同，只是`unique_ptr`解决了`auto_ptr`引入的问题。因此，`unique_ptr`是`C++11`开始的`auto_ptr`的替代品。`unique_ptr`智能指针只允许一个智能指针独占拥有堆分配的对象。只能通过`std::move()`函数将一个`unique_ptr`实例的所有权转移给另一个实例。

因此，让我们重构我们之前的示例，使用`unique_ptr`来替代`auto_ptr`。

重构后的代码示例如下：

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
using namespace std;

class MyClass {
      private:
          static int count;
          string name;

      public:
          MyClass() {
                ostringstream stringStream(ostringstream::ate);
                stringStream << "Object";
                stringStream << ++count;
                name = stringStream.str();
                cout << "\nMyClass Default constructor - " << name << endl;
          }

          ~MyClass() {
                cout << "\nMyClass destructor - " << name << endl;
          }

          MyClass ( const MyClass &objectBeingCopied ) {
                cout << "\nMyClass copy constructor" << endl;
          }

          MyClass& operator = ( const MyClass &objectBeingAssigned ) {
                cout << "\nMyClass assignment operator" << endl;
          }

          void sayHello( ) {
                cout << "\nHello from MyClass" << endl;
          }

};

int MyClass::count = 0;

int main ( ) {

 unique_ptr<MyClass> ptr1( new MyClass() );
 unique_ptr<MyClass> ptr2( new MyClass() );

 ptr1->sayHello();
 ptr2->sayHello();

 //At this point the below stuffs happen
 //1\. ptr2 smart pointer has given up ownership of MyClass Object 2
 //2\. MyClass Object 2 will be destructed as ptr2 has given up its 
 // ownership on Object 2
 //3\. Ownership of Object 1 will be transferred to ptr2
 ptr2 = move( ptr1 );

 //The line below if uncommented will result in core dump as ptr1 
 //has given up its ownership on Object 1 and the ownership of 
 //Object 1 is transferred to ptr2.
 // ptr1->sayHello();

 ptr2->sayHello();

 return 0;
}
```

上述程序的输出如下：

```cpp
g++ main.cpp -std=c++17

./a.out

MyClass Default constructor - Object1

MyClass Default constructor - Object2

MyClass destructor - Object2

MyClass destructor - Object1 
```

在上述输出中，您可以注意到编译器没有报告任何警告，并且程序的输出与`auto_ptr`的输出相同。

# 代码演示

重要的是要注意`main()`函数中`auto_ptr`和`unique_ptr`之间的区别。让我们来看一下以下代码中所示的`main()`函数。这段代码在堆中创建了两个`MyClass`对象的实例，分别用`ptr1`和`ptr2`包装起来：

```cpp
 unique_ptr<MyClass> ptr1( new MyClass() );
 unique_ptr<MyClass> ptr2( new MyClass() );
```

接下来，以下代码演示了如何使用`unique_ptr`调用`MyClass`支持的方法：

```cpp
 ptr1->sayHello();
 ptr2->sayHello();
```

就像`auto_ptr`一样，`unique_ptr`智能指针`ptr1`对象重载了`->`指针运算符和`*`解引用运算符；因此，它看起来像一个指针。

以下代码演示了`unique_ptr`不支持将一个`unique_ptr`实例分配给另一个实例，只能通过`std::move()`函数实现所有权转移：

```cpp
ptr2 = std::move(ptr1);
```

`move`函数触发以下活动：

+   `ptr2`智能指针放弃了对`MyClass`对象 2 的所有权

+   `ptr2`放弃了对`object 2`的所有权，因此`MyClass`对象 2 被销毁

+   `object 1` 的所有权已转移到 `ptr2`

+   此时，`ptr1` 既不指向 `object 1`，也不负责管理 `object 1` 使用的内存

如果取消注释以下代码，将导致核心转储：

```cpp
// ptr1->sayHello();
```

最后，以下代码让我们使用 `ptr2` 智能指针调用 `object 1` 的 `sayHello()` 方法：

```cpp
ptr2->sayHello();
return 0;
```

我们刚刚看到的 `return` 语句将在 `main()` 函数中启动堆栈展开过程。这将最终调用 `ptr2` 的析构函数，从而释放 `object 1` 使用的内存。请注意，`unique_ptr` 对象可以存储在 STL 容器中，而 `auto_ptr` 对象则不行。

# shared_ptr

当一组 `shared_ptr` 对象共享堆分配的对象的所有权时，使用 `shared_ptr` 智能指针。当所有 `shared_ptr` 实例完成对共享对象的使用时，`shared_ptr` 指针释放共享对象。`shared_ptr` 指针使用引用计数机制来检查对共享对象的总引用；每当引用计数变为零时，最后一个 `shared_ptr` 实例将删除共享对象。

让我们通过一个示例来检查 `shared_ptr` 的使用，如下所示：

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
using namespace std;

class MyClass {
  private:
    static int count;
    string name;
  public:
    MyClass() {
      ostringstream stringStream(ostringstream::ate);
      stringStream << "Object";
      stringStream << ++count;

      name = stringStream.str();

      cout << "\nMyClass Default constructor - " << name << endl;
    }

    ~MyClass() {
      cout << "\nMyClass destructor - " << name << endl;
    }

    MyClass ( const MyClass &objectBeingCopied ) {
      cout << "\nMyClass copy constructor" << endl;
    }

    MyClass& operator = ( const MyClass &objectBeingAssigned ) {
      cout << "\nMyClass assignment operator" << endl;
    }

    void sayHello() {
      cout << "Hello from MyClass " << name << endl;
    }

};

int MyClass::count = 0;

int main ( ) {

  shared_ptr<MyClass> ptr1( new MyClass() );
  ptr1->sayHello();
  cout << "\nUse count is " << ptr1.use_count() << endl;

  {
      shared_ptr<MyClass> ptr2( ptr1 );
      ptr2->sayHello();
      cout << "\nUse count is " << ptr2.use_count() << endl;
  }

  shared_ptr<MyClass> ptr3 = ptr1;
  ptr3->sayHello();
  cout << "\nUse count is " << ptr3.use_count() << endl;

  return 0;
}
```

前面程序的输出如下：

```cpp
MyClass Default constructor - Object1
Hello from MyClass Object1
Use count is 1

Hello from MyClass Object1
Use count is 2

Number of smart pointers referring to MyClass object after ptr2 is destroyed is 1

Hello from MyClass Object1
Use count is 2

MyClass destructor - Object1
```

# 代码漫游

以下代码创建了一个指向堆分配的 `MyClass` 对象的 `shared_ptr` 对象实例。与其他智能指针一样，`shared_ptr` 也有重载的 `->` 和 `*` 运算符。因此，可以调用所有 `MyClass` 对象的方法，就好像使用原始指针一样。`use_count()` 方法告诉指向共享对象的智能指针的数量：

```cpp
 shared_ptr<MyClass> ptr1( new MyClass() );
 ptr1->sayHello();
 cout << "\nNumber of smart pointers referring to MyClass object is "
      << ptr1->use_count() << endl;
```

在以下代码中，智能指针 `ptr2` 的作用域被包含在花括号括起来的块中。因此，`ptr2` 将在以下代码块的末尾被销毁。代码块内的预期 `use_count` 函数为 2：

```cpp
 { 
      shared_ptr<MyClass> ptr2( ptr1 );
      ptr2->sayHello();
      cout << "\nNumber of smart pointers referring to MyClass object is "
           << ptr2->use_count() << endl;
 }
```

在以下代码中，预期的 `use_count` 值为 1，因为 `ptr2` 已被删除，这将减少 1 个引用计数：

```cpp
 cout << "\nNumber of smart pointers referring to MyClass object after ptr2 is destroyed is "
 << ptr1->use_count() << endl; 
```

以下代码将打印一个 Hello 消息，后跟 `use_count` 为 2。这是因为 `ptr1` 和 `ptr3` 现在都指向堆中的 `MyClass` 共享对象：

```cpp
shared_ptr<MyClass> ptr3 = ptr2;
ptr3->sayHello();
cout << "\nNumber of smart pointers referring to MyClass object is "
     << ptr2->use_count() << endl;
```

`main` 函数末尾的 `return 0;` 语句将销毁 `ptr1` 和 `ptr3`，将引用计数减少到零。因此，我们可以观察到输出末尾打印 `MyClass` 析构函数的语句。

# weak_ptr

到目前为止，我们已经讨论了 `shared_ptr` 的正面作用，并举例说明。但是，当应用程序设计中存在循环依赖时，`shared_ptr` 无法清理内存。要么必须重构应用程序设计以避免循环依赖，要么可以使用 `weak_ptr` 来解决循环依赖问题。

您可以查看我的 YouTube 频道，了解 `shared_ptr` 问题以及如何使用 `weak_ptr` 解决该问题：[`www.youtube.com/watch?v=SVTLTK5gbDc`](https://www.youtube.com/watch?v=SVTLTK5gbDc)。

考虑有三个类：A、B 和 C。类 A 和 B 都有一个 C 的实例，而 C 有 A 和 B 的实例。这里存在一个设计问题。A 依赖于 C，而 C 也依赖于 A。同样，B 依赖于 C，而 C 也依赖于 B。

考虑以下代码：

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
using namespace std;

class C;

class A {
      private:
           shared_ptr<C> ptr;
      public:
           A() {
                 cout << "\nA constructor" << endl;
           }

           ~A() {
                 cout << "\nA destructor" << endl;
           }

           void setObject ( shared_ptr<C> ptr ) {
                this->ptr = ptr;
           }
};

class B {
      private:
           shared_ptr<C> ptr;
      public:
           B() {
                 cout << "\nB constructor" << endl;
           }

           ~B() {
                 cout << "\nB destructor" << endl;
           }

           void setObject ( shared_ptr<C> ptr ) {
                this->ptr = ptr;
           }
};

class C {
      private:
           shared_ptr<A> ptr1;
           shared_ptr<B> ptr2;
      public:
           C(shared_ptr<A> ptr1, shared_ptr<B> ptr2) {
                   cout << "\nC constructor" << endl;
                   this->ptr1 = ptr1;
                   this->ptr2 = ptr2;
           }

           ~C() {
                   cout << "\nC destructor" << endl;
           }
};

int main ( ) {
                shared_ptr<A> a( new A() );
                shared_ptr<B> b( new B() );
                shared_ptr<C> c( new C( a, b ) );

                a->setObject ( shared_ptr<C>( c ) );
                b->setObject ( shared_ptr<C>( c ) );

                return 0;
}
```

前面程序的输出如下：

```cpp
g++ problem.cpp -std=c++17

./a.out

A constructor

B constructor

C constructor
```

在前面的输出中，您可以观察到，即使我们使用了`shared_ptr`，对象 A、B 和 C 使用的内存从未被释放。这是因为我们没有看到各自类的析构函数被调用。原因是`shared_ptr`在内部使用引用计数算法来决定是否共享对象必须被销毁。然而，它在这里失败了，因为除非删除对象 C，否则无法删除对象 A。除非删除对象 A，否则无法删除对象 C。同样，除非删除对象 A 和 B，否则无法删除对象 C。同样，除非删除对象 C，否则无法删除对象 A，除非删除对象 C，否则无法删除对象 B。

问题的关键是这是一个循环依赖设计问题。为了解决这个问题，从 C++11 开始，C++引入了`weak_ptr`。`weak_ptr`智能指针不是一个强引用。因此，所引用的对象可以在任何时候被删除，不像`shared_ptr`。

# 循环依赖

循环依赖是一个问题，如果对象 A 依赖于 B，而对象 B 又依赖于 A。现在让我们看看如何通过`shared_ptr`和`weak_ptr`的组合来解决这个问题，最终打破循环依赖，如下所示：

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
using namespace std;

class C;

class A {
      private:
 weak_ptr<C> ptr;
      public:
           A() {
                  cout << "\nA constructor" << endl;
           }

           ~A() {
                  cout << "\nA destructor" << endl;
           }

           void setObject ( weak_ptr<C> ptr ) {
                  this->ptr = ptr;
           }
};

class B {
      private:
 weak_ptr<C> ptr;
      public:
           B() {
               cout << "\nB constructor" << endl;
           }

           ~B() {
               cout << "\nB destructor" << endl;
           }

           void setObject ( weak_ptr<C> ptr ) {
                this->ptr = ptr;
           }
};

class C {
      private:
           shared_ptr<A> ptr1;
           shared_ptr<B> ptr2;
      public:
           C(shared_ptr<A> ptr1, shared_ptr<B> ptr2) {
                   cout << "\nC constructor" << endl;
                   this->ptr1 = ptr1;
                   this->ptr2 = ptr2;
           }

           ~C() {
                   cout << "\nC destructor" << endl;
           }
};

int main ( ) {
         shared_ptr<A> a( new A() );
         shared_ptr<B> b( new B() );
         shared_ptr<C> c( new C( a, b ) );

         a->setObject ( weak_ptr<C>( c ) );
         b->setObject ( weak_ptr<C>( c ) );

         return 0;
}
```

重构代码的输出如下：

```cpp
g++ solution.cpp -std=c++17

./a.out

A constructor

B constructor

C constructor

C destructor

B destructor

A destructor
```

# 摘要

在本章中，您了解到

+   由于原始指针而引起的内存泄漏问题

+   关于赋值和复制构造函数的`auto_ptr`的问题

+   `unique_ptr`及其优势

+   `shared_ptr`在内存管理中的作用及其与循环依赖相关的限制。

+   您还可以使用`weak_ptr`解决循环依赖问题。

在下一章中，您将学习如何在 C++中开发 GUI 应用程序。
