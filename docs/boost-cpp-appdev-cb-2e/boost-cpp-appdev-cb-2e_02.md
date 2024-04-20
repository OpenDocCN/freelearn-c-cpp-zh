# 管理资源

在本章中，我们将涵盖以下主题：

+   管理不离开作用域的类的本地指针

+   对跨函数使用的类指针进行引用计数

+   管理不离开作用域的数组的本地指针

+   对跨函数使用的数组指针进行引用计数

+   在变量中存储任何功能对象

+   在变量中传递函数指针

+   在变量中传递 C++11 lambda 函数

+   指针的容器

+   在作用域退出时执行！

+   通过派生类的成员初始化基类

# 介绍

在本章中，我们将继续处理 Boost 库引入的数据类型，主要关注指针的处理。我们将看到如何轻松管理资源，如何使用能够存储任何功能对象、函数和 lambda 表达式的数据类型。阅读完本章后，你的代码将变得更加可靠，内存泄漏将成为历史。

# 管理不离开作用域的类的本地指针

有时，我们需要动态分配内存并在该内存中构造一个类。问题就出在这里。看一下以下代码：

```cpp
bool foo1() { 
    foo_class* p = new foo_class("Some data"); 

    const bool something_else_happened = some_function1(*p);  
    if (something_else_happened) { 
        delete p; 
        return false; 
    } 

    some_function2(p); 

    delete p; 
    return true; 
}
```

这段代码乍一看是正确的。但是，如果`some_function1()`或`some_function2()`抛出异常怎么办？在这种情况下，`p`不会被删除。让我们以以下方式修复它：

```cpp
bool foo2() { 
    foo_class* p = new foo_class("Some data"); 
    try { 
        const bool something_else_happened = some_function1(*p); 
        if (something_else_happened) { 
            delete p; 
            return false; 
        } 
       some_function2(p); 
    } catch (...) { 
        delete p; 
        throw; 
    } 
    delete p; 
    return true; 
}
```

现在代码是正确的，但是丑陋且难以阅读。我们能做得比这更好吗？

# 入门

需要对 C++的基本知识和异常期间代码行为有所了解。

# 如何做到这一点...

只需看一下`Boost.SmartPtr`库。有一个`boost::scoped_ptr`类可能会帮到你：

```cpp
#include <boost/scoped_ptr.hpp> 

bool foo3() { 
    const boost::scoped_ptr<foo_class> p(new foo_class("Some data")); 

    const bool something_else_happened = some_function1(*p); 
    if (something_else_happened) { 
       return false; 
    } 
    some_function2(p.get()); 
    return true; 
}
```

现在，资源不会泄漏，源代码也更清晰。

如果你可以控制`some_function2(foo_class*)`，你可能希望将其重写为接受`foo_class`的引用而不是指针。具有引用的接口比具有指针的接口更直观，除非你的公司有一个特殊的约定，即输出参数只能通过指针传递。

顺便说一句，`Boost.Move`还有一个`boost::movelib::unique_ptr`，你可以用它来代替`boost::scoped_ptr`：

```cpp
#include <boost/move/make_unique.hpp> 

bool foo3_1() { 
    const boost::movelib::unique_ptr<foo_class> p
        = boost::movelib::make_unique<foo_class>("Some data"); 

    const bool something_else_happened = some_function1(*p); 
    if (something_else_happened) { 
       return false; 
    } 
    some_function2(p.get()); 
    return true; 
}
```

# 它是如何工作的...

`boost::scoped_ptr<T>`和`boost::movelib::unique_ptr`是典型的**RAII**类。当抛出异常或变量超出作用域时，堆栈被展开并调用析构函数。在析构函数中，`scoped_ptr<T>`和`unique_ptr<T>`调用`delete`来删除它们存储的指针。因为这两个类默认调用`delete`，所以如果基类的析构函数是虚拟的，通过指向`base`类的指针持有`derived`类是安全的：

```cpp
#include <iostream>
#include <string>

struct base {
    virtual ~base(){}
};

class derived: public base {
    std::string str_;

public:
    explicit derived(const char* str)
        : str_(str)
    {}

    ~derived() /*override*/ {
        std::cout << "str == " << str_ << '\n';
    }
};

void base_and_derived() {
    const boost::movelib::unique_ptr<base> p1(
        boost::movelib::make_unique<derived>("unique_ptr")
    );

    const boost::scoped_ptr<base> p2(
        new derived("scoped_ptr")
    );
}
```

运行`base_and_derived()`函数将产生以下输出：

```cpp
str == scoped_ptr
str == unique_ptr
```

在 C++中，对象的析构函数是按照相反的构造顺序调用的。这就是为什么在`scoped_ptr`的析构函数之前调用了`unique_ptr`的析构函数。

`boost::scoped_ptr<T>`类模板既不可复制也不可移动。`boost::movelib::unique_ptr`类是一个只能移动的类，并且在 C++11 之前的编译器上使用移动模拟。这两个类都存储指向它们拥有的资源的指针，并且不要求`T`是一个完整类型（`T`可以被前向声明）。

有些编译器在删除不完整类型时不会发出警告，这可能导致难以检测的错误。幸运的是，Boost 类具有特定的编译时断言来处理这种情况。这使得`scoped_ptr`和`unique_ptr`非常适合实现**Pimpl**习惯用法：

```cpp
// In header file:
struct public_interface {
    // ...
private:
    struct impl; // Forward declaration.
    boost::movelib::unique_ptr<impl> impl_;
};
```

# 还有更多...

这些类非常快。编译器会将使用`scoped_ptr`和`unique_ptr`的代码优化为机器代码，与手动编写的内存管理代码相比，几乎没有额外的开销。

C++11 有一个`std::unique_ptr<T, D>`类，它独占资源，并且与`boost::movelib::unique_ptr<T, D>`的行为完全相同。

C++标准库没有`boost::scoped_ptr<T>`，但您可以使用`const std::unique_ptr<T>`代替。唯一的区别是`boost::scoped_ptr<T>`仍然可以调用`reset()`，而`const std::unique_ptr<T>`不行。

# 另请参阅

+   `Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息。您可以在[`boost.org/libs/smart_ptr`](http://boost.org/libs/smart_ptr)上阅读有关它们的信息。

+   如果您使用`boost::movelib::unique_ptr`进行移动模拟，`Boost.Move`文档可能会帮助您[`boost.org/libs/move`](http://boost.org/libs/move)。

# 跨函数使用的类指针的引用计数

假设您有一些包含数据的动态分配的结构，并且您希望在不同的执行线程中处理它。要执行此操作的代码如下：

```cpp
#include <boost/thread.hpp> 
#include <boost/bind.hpp> 

void process1(const foo_class* p); 
void process2(const foo_class* p); 
void process3(const foo_class* p); 

void foo1() { 
    while (foo_class* p = get_data()) // C way 
    { 
        // There will be too many threads soon, see 
        // recipe 'Parallel execution of different tasks' 
        // for a good way to avoid uncontrolled growth of threads 
        boost::thread(boost::bind(&process1, p)) 
            .detach(); 
        boost::thread(boost::bind(&process2, p)) 
            .detach(); 
        boost::thread(boost::bind(&process3, p)) 
            .detach(); 

        // delete p; Oops!!!! 
    } 
}
```

我们不能在`while`循环结束时释放`p`，因为它仍然可以被运行`process`函数的线程使用。这些`process`函数不能删除`p`，因为它们不知道其他线程不再使用它。

# 准备工作

此示例使用`Boost.Thread`库，这不是一个仅头文件的库。您的程序必须链接到`boost_thread`、`boost_chrono`和`boost_system`库。在继续阅读之前，请确保您了解线程的概念。有关描述线程的配方的参考，请参阅*另请参阅*部分。

您还需要对`boost::bind`或`std::bind`有一些基本的了解，它们几乎是一样的。

# 如何做...

正如您可能已经猜到的，Boost（和 C++11）中有一个类可以帮助您解决这个问题。它被称为`boost::shared_ptr`。可以按以下方式使用：

```cpp
#include <boost/shared_ptr.hpp> 

void process_sp1(const boost::shared_ptr<foo_class>& p); 
void process_sp2(const boost::shared_ptr<foo_class>& p); 
void process_sp3(const boost::shared_ptr<foo_class>& p); 

void foo2() { 
    typedef boost::shared_ptr<foo_class> ptr_t; 
    ptr_t p; 
    while (p = ptr_t(get_data())) // C way 
    { 
        boost::thread(boost::bind(&process_sp1, p)) 
            .detach(); 
        boost::thread(boost::bind(&process_sp2, p)) 
            .detach(); 
        boost::thread(boost::bind(&process_sp3, p)) 
            .detach(); 

        // no need to anything 
    } 
}
```

另一个例子如下：

```cpp
#include <string> 
#include <boost/smart_ptr/make_shared.hpp> 

void process_str1(boost::shared_ptr<std::string> p); 
void process_str2(const boost::shared_ptr<std::string>& p); 

void foo3() { 
    boost::shared_ptr<std::string> ps = boost::make_shared<std::string>( 
        "Guess why make_shared<std::string> " 
        "is faster than shared_ptr<std::string> " 
        "ps(new std::string('this string'))" 
    ); 

    boost::thread(boost::bind(&process_str1, ps)) 
            .detach(); 
    boost::thread(boost::bind(&process_str2, ps)) 
            .detach(); 
}
```

# 它是如何工作的...

`shared_ptr`类内部有一个原子引用计数器。当您复制它时，引用计数会增加，当调用其`析构函数`时，引用计数会减少。当引用计数等于零时，将为`shred_ptr`指向的对象调用`delete`。

现在，让我们看看在`boost::thread (boost::bind(&process_sp1, p))`的情况下发生了什么。函数`process_sp1`以引用参数作为参数，那么当我们退出`while`循环时为什么它不会被释放？答案很简单。`bind()`返回的函数对象包含`shared`指针的副本，这意味着`p`指向的数据直到函数对象被销毁才会被释放。函数对象被复制到线程中，并在线程执行时保持活动状态。

回到`boost::make_shared`，让我们看看`shared_ptr<std::string> ps(new int(0))`。在这种情况下，我们有两个调用`new`：

+   通过`new int(0)`构造一个指向整数的指针

+   在构造`shared_ptr`类内部引用计数器分配在堆上

使用`make_shared<T>`只需一次调用`new`。`make_shared<T>`分配一个单一的内存块，并在该内存块中构造原子计数器和`T`对象。

# 还有更多...

原子引用计数器保证了`shared_ptr`在线程之间的正确行为，但您必须记住原子操作不如非原子操作快。`shared_ptr`在赋值、复制构造和未移动的`shared_ptr`销毁时会触及原子变量。这意味着在兼容 C++11 的编译器上，您可以尽可能使用移动构造和移动赋值来减少原子操作的次数。如果您不打算再使用`p`变量，只需使用`shared_ptr<T> p1(std::move(p))`。如果您不打算修改指向的值，建议将其设置为`const`。只需将`const`添加到智能指针的模板参数中，编译器将确保您不会修改内存：

```cpp
void process_cstr1(boost::shared_ptr<const std::string> p);
void process_cstr2(const boost::shared_ptr<const std::string>& p);

void foo3_const() {
    boost::shared_ptr<const std::string> ps
        = boost::make_shared<const std::string>(
            "Some immutable string"
        );

    boost::thread(boost::bind(&process_cstr1, ps))
            .detach();
    boost::thread(boost::bind(&process_cstr2, ps))
            .detach();

    // *ps = "qwe"; // Compile time error, string is const!
}
```

对`const`感到困惑？以下是智能指针 constness 到简单指针 constness 的映射：

| `shared_ptr<T>` | `T*`  |
| --- | --- |
| `shared_ptr<const T>` | `const T*`  |
| `const shared_ptr<T>` | `T* const` |
| `const shared_ptr<const T>` | `const T* const` |

`shared_ptr`调用和`make_shared`函数是 C++11 的一部分，它们在`std::`命名空间的头文件`<memory>`中声明。它们几乎具有与 Boost 版本相同的特性。

# 另请参阅

+   有关`Boost.Thread`和原子操作的更多信息，请参阅第五章*，* *多线程*。

+   有关如何绑定和重新排序函数参数的信息，请参阅第一章的*开始编写您的应用程序*中的*绑定和重新排序函数参数*食谱，了解更多关于`Boost.Bind`的信息。

+   有关如何将`shared_ptr<U>`转换为`shared_ptr<T>`的信息，请参阅第三章的*转换智能指针*。

+   `Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息。请参阅链接[`boost.org/libs/smart_ptr`](http://boost.org/libs/smart_ptr)了解相关内容。

# 管理不离开作用域的数组指针

我们已经看到如何在*管理不离开作用域的类的指针*食谱中管理指向资源的指针。但是，当我们处理数组时，我们需要调用`delete[]`而不是简单的`delete`。否则，将会发生内存泄漏。请看下面的代码：

```cpp
void may_throw1(char ch); 
void may_throw2(const char* buffer); 

void foo() { 
    // we cannot allocate 10MB of memory on stack, 
    // so we allocate it on heap 
    char* buffer = new char[1024 * 1024 * 10]; 

    // Oops. Here comes some code, that may throw.
    // It was a bad idea to use raw pointer as the memory may leak!!
    may_throw1(buffer[0]); 
    may_throw2(buffer); 

    delete[] buffer; 
}
```

# 准备就绪

此食谱需要了解 C++异常和模板的知识。

# 如何做...

`Boost.SmartPointer`库不仅有`scoped_ptr<>`类，还有`scoped_array<>`类：

```cpp
#include <boost/scoped_array.hpp> 

void foo_fixed() { 
    // We allocate array on heap 
    boost::scoped_array<char> buffer(new char[1024 * 1024 * 10]); 

    // Here comes some code, that may throw, 
    // but now exception won't cause a memory leak 
    may_throw1(buffer[0]); 
    may_throw2(buffer.get()); 

    // destructor of 'buffer' variable will call delete[] 
}
```

`Boost.Move`库的`boost::movelib::unique_ptr<>`类也可以与数组一起使用。您只需要在模板参数的末尾提供`[]`来指示它存储的是数组。

```cpp
#include <boost/move/make_unique.hpp> 

void foo_fixed2() { 
    // We allocate array on heap 
    const boost::movelib::unique_ptr<char[]> buffer 
        = boost::movelib::make_unique<char[]>(1024 * 1024 * 10); 

    // Here comes some code, that may throw, 
    // but now exception won't cause a memory leak 
    may_throw1(buffer[0]); 
    may_throw2(buffer.get()); 

    // destructor of 'buffer' variable will call delete[] 
}
```

# 工作原理...

`scoped_array<>`的工作原理与`scoped_ptr<>`类完全相同，但在析构函数中调用`delete[]`而不是`delete`。`unique_ptr<T[]>`也是这样做的。

# 还有更多...

`scoped_array<>`类与`scoped_ptr<>`具有相同的保证和设计。它既没有额外的内存分配，也没有虚函数的调用。它不能被复制，也不是 C++11 的一部分。`std::unique_ptr<T[]>`是 C++11 的一部分，具有与`boost::movelib::unique_ptr<T[]>`类相同的保证和性能。

实际上，`make_unique<char[]>(1024)`与`new char[1024]`不同，因为第一个进行值初始化，而第二个进行默认初始化。默认初始化的等效函数是`boost::movelib::make_unique_definit`。

请注意，Boost 版本也可以在 C++11 之前的编译器上工作，甚至在这些编译器上模拟 rvalues，使`boost::movelib::unique_ptr`成为仅移动类型。如果您的标准库没有提供`std::make_unique`，那么`Boost.SmartPtr`可能会帮助您。它提供了`boost::make_unique`，在头文件`boost/smart_ptr/make_unique.hpp`中返回一个`std::unique_ptr`。它还提供了`boost::make_unique_noinit`，用于在相同的头文件中进行默认初始化。C++17 没有`make_unique_noinit`函数。

在 C++中使用`new`进行内存分配和手动内存管理是一种不好的习惯。尽可能使用`make_unique`和`make_shared`函数。

# 另请参阅

+   `Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息，您可以在[`boost.org/libs/smart_ptr.`](http://boost.org/libs/smart_ptr)上阅读相关内容。

+   如果您希望使用`boost::movelib::unique_ptr`进行移动模拟，`Boost.Move`文档可能会对您有所帮助，请阅读[`boost.org/libs/move.`](http://boost.org/libs/move) [﻿](http://boost.org/libs/move)

# 引用计数的指向跨函数使用的数组的指针

我们继续处理指针，我们的下一个任务是对数组进行引用计数。让我们看一下从流中获取一些数据并在不同的线程中处理它的程序。代码如下：

```cpp
#include <cstring> 
#include <boost/thread.hpp> 
#include <boost/bind.hpp> 

void do_process(const char* data, std::size_t size); 

void do_process_in_background(const char* data, std::size_t size) { 
    // We need to copy data, because we do not know, 
    // when it will be deallocated by the caller.
    char* data_cpy = new char[size]; 
    std::memcpy(data_cpy, data, size); 

    // Starting thread of execution to process data.
    boost::thread(boost::bind(&do_process, data_cpy, size)) 
            .detach(); 
    boost::thread(boost::bind(&do_process, data_cpy, size)) 
            .detach();

    // Oops!!! We cannot delete[] data_cpy, because 
    // do_process() function may still work with it.
}
```

与*跨函数使用类指针的引用计数*示例中发生的相同问题。

# 准备工作

这个示例使用了`Boost.Thread`库，这不是一个仅包含头文件的库，所以你的程序需要链接`boost_thread`、`boost_chrono`和`boost_system`库。在继续阅读之前，请确保你理解了线程的概念。

你还需要一些关于`boost::bind`或`std::bind`的基本知识，它们几乎是一样的。

# 如何做...

有四种解决方案。它们之间的主要区别在于`data_cpy`变量的类型和构造方式。所有这些解决方案都完全做了本示例开头描述的相同的事情，但没有内存泄漏。这些解决方案如下：

+   第一个解决方案适用于在编译时已知数组大小的情况：

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

template <std::size_t Size>
void do_process_shared(const boost::shared_ptr<char[Size]>& data);

template <std::size_t Size>
void do_process_in_background_v1(const char* data) {
    // Same speed as in 'First solution'.
    boost::shared_ptr<char[Size]> data_cpy
        = boost::make_shared<char[Size]>();
    std::memcpy(data_cpy.get(), data, Size);

    // Starting threads of execution to process data.
    boost::thread(boost::bind(&do_process_shared<Size>, data_cpy))
            .detach();
    boost::thread(boost::bind(&do_process_shared<Size>, data_cpy))
            .detach();

    // data_cpy destructor will deallocate data when
    // reference count is zero.
}
```

+   自 Boost 1.53 以来，`shared_ptr`本身可以处理未知大小的数组。第二个解决方案：

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

void do_process_shared_ptr(
        const boost::shared_ptr<char[]>& data,
        std::size_t size);

void do_process_in_background_v2(const char* data, std::size_t size) {
    // Faster than 'First solution'.
    boost::shared_ptr<char[]> data_cpy = boost::make_shared<char[]>(size);
    std::memcpy(data_cpy.get(), data, size);

    // Starting threads of execution to process data.
    boost::thread(boost::bind(&do_process_shared_ptr, data_cpy, size))
            .detach();
    boost::thread(boost::bind(&do_process_shared_ptr, data_cpy, size))
            .detach();

    // data_cpy destructor will deallocate data when
    // reference count is zero.
}
```

+   第三个解决方案：

```cpp
#include <boost/shared_ptr.hpp>

void do_process_shared_ptr2(
        const boost::shared_ptr<char>& data,
        std::size_t size);

void do_process_in_background_v3(const char* data, std::size_t size) {
    // Same speed as in 'First solution'.
    boost::shared_ptr<char> data_cpy(
                new char[size],
                boost::checked_array_deleter<char>()
    );
    std::memcpy(data_cpy.get(), data, size);

    // Starting threads of execution to process data.
    boost::thread(boost::bind(&do_process_shared_ptr2, data_cpy, size))
            .detach();
    boost::thread(boost::bind(&do_process_shared_ptr2, data_cpy, size))
            .detach();

    // data_cpy destructor will deallocate data when
    // reference count is zero.
}
```

+   最后一个解决方案自 Boost 1.65 以来已经被弃用，但在古老的 Boost 版本中可能会有用：

```cpp
#include <boost/shared_array.hpp>

void do_process_shared_array(
        const boost::shared_array<char>& data,
        std::size_t size);

void do_process_in_background_v4(const char* data, std::size_t size) {
    // We need to copy data, because we do not know, when it will be
    // deallocated by the caller.
    boost::shared_array<char> data_cpy(new char[size]);
    std::memcpy(data_cpy.get(), data, size);

    // Starting threads of execution to process data.
    boost::thread(
        boost::bind(&do_process_shared_array, data_cpy, size)
    ).detach();
    boost::thread(
        boost::bind(&do_process_shared_array, data_cpy, size)
    ).detach();

    // No need to call delete[] for data_cpy, because
    // data_cpy destructor will deallocate data when
    // reference count is zero.
}
```

# 它是如何工作的...

在所有示例中，**智能指针**类计算引用并在引用计数变为零时调用`delete[]`释放指针。第一个和第二个示例很简单。在第三个示例中，我们为`shared`指针提供了一个自定义的`deleter`对象。当智能指针决定释放资源时，智能指针的`deleter`对象被调用。当智能指针在没有显式`deleter`的情况下构造时，会构造默认的`deleter`，它根据智能指针的模板类型调用`delete`或`delete[]`。

# 还有更多...

第四个解决方案是最保守的，因为在 Boost 1.53 之前，第二个解决方案的功能没有在`shared_ptr`中实现。第一个和第二个解决方案是最快的，因为它们只使用了一次内存分配调用。第三个解决方案可以与较旧版本的 Boost 和 C++11 标准库的`std::shared_ptr<>`一起使用（只需不要忘记将`boost::checked_array_deleter<T>()`更改为`std::default_delete<T[]>()`）。

实际上，`boost::make_shared<char[]>(size)`并不等同于`new char[size]`，因为它涉及到所有元素的值初始化。默认初始化的等效函数是`boost::make_shared_noinit`。

注意！C++11 和 C++14 版本的`std::shared_ptr`不能处理数组！只有在 C++17 中`std::shared_ptr<T[]>`才能正常工作。如果你计划编写可移植的代码，请考虑使用`boost::shared_ptr`、`boost::shared_array`，或者显式地将`deleter`传递给`std::shared_ptr`。

`boost::shared_ptr<T[]>`、`boost::shared_array`和 C++17 的`std::shared_ptr<T[]>`都有`operator[](std::size_t index)`，允许你通过索引访问共享数组的元素。`boost::shared_ptr<T>`和带有自定义`deleter`的`std::shared_ptr<T>`没有`operator[]`，这使它们不太有用。

# 另请参阅

`Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息。你可以在[`boost.org/libs/smart_ptr`](http://boost.org/libs/smart_ptr)上阅读相关内容。

# 将任何功能对象存储在变量中

考虑这样一种情况，当你开发一个库，它的 API 在头文件中声明，而在源文件中实现。这个库应该有一个接受任何功能对象的函数。看一下下面的代码：

```cpp
// making a typedef for function pointer accepting int 
// and returning nothing 
typedef void (*func_t)(int); 

// Function that accepts pointer to function and 
// calls accepted function for each integer that it has.
// It cannot work with functional objects :( 
void process_integers(func_t f); 

// Functional object 
class int_processor { 
   const int min_; 
   const int max_; 
   bool& triggered_; 

public: 
    int_processor(int min, int max, bool& triggered) 
        : min_(min) 
        , max_(max) 
        , triggered_(triggered) 
    {} 

    void operator()(int i) const { 
        if (i < min_ || i > max_) { 
            triggered_ = true; 
        } 
    } 
};
```

如何修改`process_integers`函数以接受任何功能对象？

# 准备工作

在开始本教程之前，建议先阅读第一章中的*在容器/变量中存储任何值*教程，*开始编写您的应用程序*。

# 如何做...

有一个解决方案，它被称为`Boost.Function`库。它允许您存储任何函数、成员函数或者函数对象，只要它的签名与模板参数中描述的一致：

```cpp
#include <boost/function.hpp> 

typedef boost::function<void(int)> fobject_t; 

// Now this function may accept functional objects 
void process_integers(const fobject_t& f); 

int main() { 
    bool is_triggered = false; 
    int_processor fo(0, 200, is_triggered); 
    process_integers(fo); 
    assert(is_triggered); 
}
```

# 它是如何工作的...

`fobject_t`对象在自身中存储函数对象并擦除它们的确切类型。使用`boost::function`来存储有状态的对象是安全的：

```cpp
bool g_is_triggered = false; 
void set_functional_object(fobject_t& f) {
    // Local variable
    int_processor fo( 100, 200, g_is_triggered); 

    f = fo;
    // now 'f' holds a copy of 'fo'

    // 'fo' leavs scope and will be destroyed,
    // but it's OK to use 'f' in outer scope.
}
```

`boost::function`是否记得`boost::any`类？那是因为它使用相同的技术**类型擦除**来存储任何函数对象。

# 还有更多...

`boost::function`类有一个默认构造函数并且有一个空状态。可以像这样检查是否为空/默认构造状态：

```cpp
void foo(const fobject_t& f) { 
    // boost::function is convertible to bool 
    if (f) { 
        // we have value in 'f' 
        // ... 
    } else { 
        // 'f' is empty 
        // ... 
    } 
}
```

`Boost.Function`库有大量的优化。它可以存储小型函数对象而无需额外的内存分配，并且具有优化的移动赋值运算符。它被接受为 C++11 标准库的一部分，并且在`std::`命名空间的`<functional>`头文件中定义。

`boost::function`对存储在其中的对象使用 RTTI。如果禁用 RTTI，库将继续工作，但会大幅增加编译后的二进制文件的大小。

# 另请参阅

+   Boost.Function 的官方文档包含更多示例、性能测量和类参考文档。请参考链接[`boost.org/libs/function`](http://boost.org/libs/function)进行阅读。

+   *在变量中传递函数指针*教程。

+   *在变量中传递 C++11 lambda 函数*教程。

# 在变量中传递函数指针

我们将继续使用之前的示例，现在我们想在`process_integers()`方法中传递一个函数指针。我们应该为函数指针添加一个重载，还是有更加优雅的方法？

# 准备工作

这个教程是前一个的延续。你必须先阅读前一个教程。

# 如何做...

不需要做任何事情，因为`boost::function<>`也可以从函数指针中构造：

```cpp
void my_ints_function(int i); 

int main() { 
    process_integers(&my_ints_function); 
}
```

# 它是如何工作的...

指向`my_ints_function`的指针将被存储在`boost::function`类中，并且对`boost::function`的调用将被转发到存储的指针。

# 还有更多...

`Boost.Function`库为函数指针提供了良好的性能，并且不会在堆上分配内存。标准库`std::function`也有效地存储函数指针。自 Boost 1.58 以来，`Boost.Function`库可以存储具有 rvalue 引用调用签名的函数和函数对象：

```cpp
boost::function<int(std::string&&)> f = &something;
f(std::string("Hello")); // Works
```

# 另请参阅

+   Boost.Function 的官方文档包含更多示例、性能测量和类参考文档。请访问[`boost.org/libs/function`](http://boost.org/libs/function)进行阅读。

+   *在变量中传递 C++11 lambda 函数*教程。

# 在变量中传递 C++11 lambda 函数

我们将继续使用之前的示例，现在我们想在`process_integers()`方法中使用一个 lambda 函数。

# 准备工作

这个教程是前两个教程的延续。你必须先阅读它们。你还需要一个兼容 C++11 的编译器，或者至少支持 C++11 lambda 的编译器。

# 如何做...

不需要做任何事情，因为`boost::function<>`也可以用于任何难度的 lambda 函数：

```cpp
#include <deque>
//#include "your_project/process_integers.h"

void sample() {
    // lambda function with no parameters that does nothing 
    process_integers([](int /*i*/){}); 

    // lambda function that stores a reference 
    std::deque<int> ints; 
    process_integers(&ints{ 
        ints.push_back(i); 
    }); 

    // lambda function that modifies its content 
    std::size_t match_count = 0; 
    process_integers(ints, &match_count mutable { 
        if (ints.front() == i) { 
           ++ match_count; 
        } 
        ints.pop_front(); 
    });
}
```

# 还有更多...

`Boost.Functional`中的 lambda 函数存储性能与其他情况相同。lambda 表达式生成的函数对象足够小，可以适应`boost::function`的实例，不会执行动态内存分配。调用存储在`boost::function`中的对象的速度接近通过指针调用函数的速度。只有在初始`boost::function`中存储的对象不适合在没有分配的情况下存储时，复制`boost::function`才会分配堆内存。移动实例不会分配和释放内存。

请记住，`boost::function`意味着对编译器的优化障碍。这意味着：

```cpp
    std::for_each(v.begin(), v.end(), [](int& v) { v += 10; });
```

通常由编译器优化得更好：

```cpp
    const boost::function<void(int&)> f0(
        [](int& v) { v += 10; }
    ); 
    std::for_each(v.begin(), v.end(), f0);
```

这就是为什么在不真正需要时应该尽量避免使用`Boost.Function`。在某些情况下，C++11 的`auto`关键字可能更方便：

```cpp
    const auto f1 = [](int& v) { v += 10; }; 
    std::for_each(v.begin(), v.end(), f1);
```

# 另请参阅

有关性能和`Boost.Function`的其他信息，请访问官方文档页面[`www.boost.org/libs/function`](http://www.boost.org/libs/function)。

# 指针容器

有这样的情况，当我们需要在容器中存储指针。示例包括：在容器中存储多态数据，强制在容器中快速复制数据，以及对容器中的数据操作有严格的异常要求。在这种情况下，C++程序员有以下选择：

+   在容器中存储指针并使用`delete`来处理它们的销毁：

```cpp
#include <set>
#include <algorithm>
#include <cassert>

template <class T>
struct ptr_cmp {
    template <class T1>
    bool operator()(const T1& v1, const T1& v2) const {
        return operator ()(*v1, *v2);
    }

    bool operator()(const T& v1, const T& v2) const {
        return std::less<T>()(v1, v2);
    }
};

void example1() {
    std::set<int*, ptr_cmp<int> > s;
    s.insert(new int(1));
    s.insert(new int(0));

    // ...
    assert(**s.begin() == 0);
    // ...

    // Oops! Any exception in the above code leads to
    // memory leak.

    // Deallocating resources.
    std::for_each(s.begin(), s.end(), [](int* p) { delete p; });
}
```

这种方法容易出错，需要大量编写。

+   在容器中存储 C++11 智能指针：

```cpp
#include <memory>
#include <set>

void example2_cpp11() {
    typedef std::unique_ptr<int> int_uptr_t;
    std::set<int_uptr_t, ptr_cmp<int> > s;
    s.insert(int_uptr_t(new int(1)));
    s.insert(int_uptr_t(new int(0)));

    // ...
    assert(**s.begin() == 0);
    // ...

    // Resources will be deallocated by unique_ptr<>.
}
```

这种解决方案很好，但不能在 C++03 中使用，而且您仍然需要编写一个比较器函数对象。

C++14 有一个`std::make_unique`函数用于构造`std::uniue_ptr`。使用它而不是`new`是一个很好的编码风格！

+   在容器中使用`Boost.SmartPtr`：

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

void example3() {
    typedef boost::shared_ptr<int> int_sptr_t;
    std::set<int_sptr_t, ptr_cmp<int> > s;
    s.insert(boost::make_shared<int>(1));
    s.insert(boost::make_shared<int>(0));

    // ...
    assert(**s.begin() == 0);
    // ...

    // Resources will be deallocated by shared_ptr<>.
}
```

这种解决方案是可移植的，但会增加性能损失（原子计数器需要额外的内存，并且其增量/减量不如非原子操作快），而且您仍然需要编写比较器。

# 准备工作

更好地理解本配方需要了解标准库容器的知识。

# 如何做...

`Boost.PointerContainer`库提供了一个很好的可移植解决方案：

```cpp
#include <boost/ptr_container/ptr_set.hpp> 

void correct_impl() { 
    boost::ptr_set<int> s; 
    s.insert(new int(1)); 
    s.insert(new int(0)); 

    // ... 
    assert(*s.begin() == 0); 
    // ... 

    // Resources will be deallocated by container itself.
}
```

# 工作原理...

`Boost.PointerContainer`库有`ptr_array`、`ptr_vector`、`ptr_set`、`ptr_multimap`等类。这些类根据需要释放指针，并简化了指针指向的数据的访问（在`assert(*s.begin() == 0);`中不需要额外的解引用）。

# 还有更多...

当我们想要克隆一些数据时，我们需要在要克隆的对象的命名空间中定义一个独立的函数`T*new_clone(const T& r)`。此外，如果您包含头文件`<boost/ptr_container/clone_allocator.hpp>`，则可以使用默认的`T* new_clone(const T& r)`实现，如下面的代码所示：

```cpp
#include <boost/ptr_container/clone_allocator.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <cassert>

void theres_more_example() {
    // Creating vector of 10 elements with values 100
    boost::ptr_vector<int> v;
    int value = 100;
    v.resize(10, &value); // Beware! No ownership of pointer!

    assert(v.size() == 10);
    assert(v.back() == 100);
}
```

C++标准库没有指针容器，但您可以使用`std::unique_ptr`的容器来实现相同的功能。顺便说一句，自 Boost 1.58 以来，有一个`boost::movelib::unique_ptr`类，可在 C++03 中使用。您可以将其与`Boost.Container`库中的容器混合使用，以获得存储指针的 C++11 功能：

```cpp
#include <boost/container/set.hpp>
#include <boost/move/make_unique.hpp>
#include <cassert>

void example2_cpp03() { 
    typedef boost::movelib::unique_ptr<int> int_uptr_t; 
    boost::container::set<int_uptr_t, ptr_cmp<int> > s; 
    s.insert(boost::movelib::make_unique<int>(1)); 
    s.insert(boost::movelib::make_unique<int>(0)); 
    // ... 
    assert(**s.begin() == 0); 
}
```

并非所有开发人员都很了解 Boost 库。使用具有 C++标准库替代品的函数和类更加友好，因为开发人员通常更了解标准库的特性。因此，如果对您来说没有太大区别，请使用`Boost.Container`与`boost::movelib::unique_ptr`。

# 另请参阅

+   官方文档包含了每个类的详细参考，请访问链接[`boost.org/libs/ptr_container`](http://boost.org/libs/ptr_container)了解更多信息。

+   本章的前四个配方为您提供了一些智能指针使用的示例。

+   第九章*，容器*中的多个食谱描述了`Boost.Container`库的特性。查看该章节，了解有用的快速容器。

# 在作用域退出时执行它！

如果您处理的是 Java、C#或 Delphi 等语言，显然您正在使用`try {} finally{}`结构。让我简要地描述一下这些语言结构的作用。

当程序通过返回或异常离开当前作用域时，`finally`块中的代码将被执行。这种机制用作 RAII 模式的替代：

```cpp
// Some pseudo code (suspiciously similar to Java code) 
try { 
    FileWriter f = new FileWriter("example_file.txt"); 
    // Some code that may throw or return 
    // ... 
} finally { 
    // Whatever happened in scope, this code will be executed 
    // and file will be correctly closed 
    if (f != null) { 
        f.close() 
    } 
}
```

在 C++中有办法做到这一点吗？

# 准备工作

此食谱需要基本的 C++知识。了解在抛出异常时代码的行为将会很有帮助。

# 如何做...

C++使用 RAII 模式而不是`try {} finally{}`结构。`Boost.ScopeExit`库旨在允许用户在函数体内定义 RAII 包装器：

```cpp
#include <boost/scope_exit.hpp> 
#include <cstdlib> 
#include <cstdio> 
#include <cassert> 

int main() { 
    std::FILE* f = std::fopen("example_file.txt", "w"); 
    assert(f); 

    BOOST_SCOPE_EXIT(f) { 
    // Whatever happened in outer scope, this code will be executed 
    // and file will be correctly closed. 
        std::fclose(f); 
    } BOOST_SCOPE_EXIT_END 

    // Some code that may throw or return. 
    // ... 
}
```

# 它是如何工作的...

`f`通过`BOOST_SCOPE_EXIT(f)`按值传递。当程序离开执行范围时，`BOOST_SCOPE_EXIT(f) {`和`} BOOST_SCOPE_EXIT_END`之间的代码将被执行。如果我们希望通过引用传递值，使用`BOOST_SCOPE_EXIT`宏中的`&`符号。如果我们希望传递多个值，只需用逗号分隔它们。

在某些编译器上，将引用传递给指针并不奏效。`BOOST_SCOPE_EXIT(&f)`宏在那里无法编译，这就是为什么我们在示例中没有通过引用捕获它的原因。

# 更多信息...

为了在成员函数中捕获这个，我们将使用一个特殊的符号`this_`：

```cpp
class theres_more_example { 
public: 
    void close(std::FILE*); 

    void theres_more_example_func() { 
        std::FILE* f = 0; 
        BOOST_SCOPE_EXIT(f, this_) { // Capturing `this` as 'this_' 
            this_->close(f); 
        } BOOST_SCOPE_EXIT_END 
    } 
};
```

`Boost.ScopeExit`库在堆上不分配额外的内存，也不使用虚函数。使用默认语法，不要定义`BOOST_SCOPE_EXIT_CONFIG_USE_LAMBDAS`，否则将使用`boost::function`来实现作用域退出，这可能会分配额外的内存并意味着一个优化障碍。您可以通过指定自定义的`deleter`，使用`boost::movelib::unique_ptr`或`std::unique_ptr`来实现接近`BOOST_SCOPE_EXIT`结果：

```cpp
#include <boost/move/unique_ptr.hpp>
#include <cstdio>

void unique_ptr_example() {
    boost::movelib::unique_ptr<std::FILE, int(*)(std::FILE*)> f(
        std::fopen("example_file.txt", "w"), // open file
        &std::fclose  // specific deleter
    );
    // ...
}
```

如果您为`BOOST_SCOPE_EXIT`编写了两个或更多类似的代码块，那么现在是时候考虑一些重构，并将代码移动到一个完全功能的 RAII 类中。

# 另请参阅

官方文档包含更多示例和用例。您可以在[`boost.org/libs/scope_exit.`](http://boost.org/libs/scope_exit)上阅读相关内容

# 通过派生类的成员初始化基类

让我们看一个例子。我们有一个基类，它有虚函数，并且必须用`std::ostream`对象的引用进行初始化：

```cpp
#include <boost/noncopyable.hpp> 
#include <sstream> 

class tasks_processor: boost::noncopyable { 
    std::ostream& log_; 

protected: 
    virtual void do_process() = 0; 

public: 
    explicit tasks_processor(std::ostream& log) 
        : log_(log) 
    {} 

    void process() { 
        log_ << "Starting data processing"; 
        do_process(); 
    } 
};
```

我们还有一个派生类，它有一个`std::ostream`对象，并实现了`do_process()`函数：

```cpp
class fake_tasks_processor: public tasks_processor { 
    std::ostringstream logger_; 

    virtual void do_process() { 
        logger_ << "Fake processor processed!"; 
    } 

public: 
    fake_tasks_processor() 
        : tasks_processor(logger_) // Oops! logger_ does not exist here 
        , logger_() 
    {} 
};
```

这在编程中并不是一个很常见的情况，但当发生这样的错误时，要想绕过它并不总是简单的。有些人试图通过改变`logger_`和基类型初始化的顺序来绕过它：

```cpp
    fake_tasks_processor() 
        : logger_() // Oops! It is still constructed AFTER tasks_processor. 
        , tasks_processor(logger_) 
    {}
```

直接基类在非静态数据成员之前初始化，而不管成员初始化器的顺序，这样不会按预期工作。

# 入门

这个食谱需要基本的 C++知识。

# 如何做...

`Boost.Utility`库为这种情况提供了一个快速解决方案。解决方案称为`boost::base_from_member`模板。要使用它，您需要执行以下步骤：

1.  包括`base_from_member.hpp`头文件：

```cpp
#include <boost/utility/base_from_member.hpp>
```

1.  从`boost::base_from_member<T>`派生您的类，其中`T`是在基类之前必须初始化的类型（注意基类的顺序；`boost::base_from_member<T>`必须放在使用`T`的类之前）：

```cpp
class fake_tasks_processor_fixed
    : boost::base_from_member<std::ostringstream>
    , public tasks_processor
```

1.  正确地，编写构造函数如下：

```cpp
{
    typedef boost::base_from_member<std::ostringstream> logger_t;

    virtual void do_process() {
        logger_t::member << "Fake processor processed!";
    }

public:
    fake_tasks_processor_fixed()
        : logger_t()
        , tasks_processor(logger_t::member)
    {}
};
```

# 它是如何工作的...

直接基类在非静态数据成员之前进行初始化，并且按照它们在基类指定符列表中出现的顺序进行初始化。如果我们需要用*something*初始化基类`B`，我们需要将*something*作为在`B`之前声明的基类`A`的一部分。换句话说，`boost::base_from_member`只是一个简单的类，它将其模板参数作为非静态数据成员：

```cpp
template < typename MemberType, int UniqueID = 0 >
class base_from_member {
protected:
    MemberType  member;
    //      Constructors go there...
};
```

# 还有更多...

正如你所看到的，`base_from_member`有一个整数作为第二个模板参数。这是为了在我们需要多个相同类型的`base_from_member`类的情况下使用：

```cpp
class fake_tasks_processor2 
    : boost::base_from_member<std::ostringstream, 0> 
    , boost::base_from_member<std::ostringstream, 1> 
    , public tasks_processor 
{ 
    typedef boost::base_from_member<std::ostringstream, 0> logger0_t;
    typedef boost::base_from_member<std::ostringstream, 1> logger1_t;

    virtual void do_process() { 
        logger0_t::member << "0: Fake processor2 processed!"; 
        logger1_t::member << "1: Fake processor2 processed!"; 
    } 

public: 
    fake_tasks_processor2() 
        : logger0_t() 
        , logger1_t() 
        , tasks_processor(logger0_t::member) 
    {} 
};
```

`boost::base_from_member`类既不应用额外的动态内存分配，也没有虚函数。如果您的编译器支持，当前实现支持**完美转发**和**可变模板**。

C++标准库中没有`base_from_member`。

# 另请参阅

+   `Boost.Utility`库包含许多有用的类和函数；有关更多信息的文档，请访问[`boost.org/libs/utility`](http://boost.org/libs/utility)

+   在第一章 *开始编写您的应用程序*中的*制作不可复制的类*示例中，包含了来自`Boost.Utility`的更多类的示例

+   此外，在第一章 *开始编写您的应用程序*中的*使用 C++11 移动模拟*示例中，包含了更多来自`Boost.Utility`的类的示例。
