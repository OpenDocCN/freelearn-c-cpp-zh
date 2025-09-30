# 第 3 章。管理资源

本章我们将涵盖：

+   管理未离开作用域的类的指针

+   在方法间引用计数类指针

+   管理未离开作用域的数组指针

+   在方法间引用计数数组指针

+   在变量中存储任何函数对象

+   在变量中传递函数指针

+   在变量中传递 C++11 lambda 函数

+   指针容器

+   在作用域退出时执行某些操作

+   通过派生类的成员初始化基类

# 简介

在本章中，我们将继续处理由 Boost 库引入的数据类型，主要关注指针的使用。我们将了解如何轻松管理资源，以及如何使用一种能够存储任何函数对象、函数和 lambda 表达式的数据类型。阅读本章后，你的代码将变得更加可靠，内存泄漏将成为历史。

# 管理未离开作用域的类的指针

有时候我们需要在内存中动态分配内存并构造一个类，麻烦就从这里开始了。看看下面的代码：

[PRE0]

这段代码乍一看似乎是正确的。但是，如果 `some_function1()` 或 `some_function2()` 抛出异常怎么办？在这种情况下，`p` 不会被删除。让我们以下面的方式修复它：

[PRE1]

现在代码看起来很丑陋且难以阅读，但却是正确的。也许我们可以做得更好。

## 准备工作

需要具备基本的 C++ 知识和异常期间代码的行为。

## 如何做到...

让我们看看 `Boost.SmartPtr` 库。这里有一个 `boost::scoped_ptr` 类，可能对你有所帮助：

[PRE2]

现在，资源泄漏的可能性已经不存在了，源代码也变得更加清晰。

### 注意

如果你控制 `some_function1()` 和 `some_function2()`，你可能希望重新编写它们，以便它们接受 `scoped_ptr<foo_class>`（或只是一个引用）的引用，而不是 `foo_class` 的指针。这样的接口将更加直观。

## 它是如何工作的...

在析构函数中，`boost::scoped_ptr<T>` 将为其存储的指针调用 `delete`。当抛出异常时，堆栈回溯，并调用 `scoped_ptr` 的析构函数。

`scoped_ptr<T>` 类模板是不可复制的；它只存储指向类的指针，并且不需要 `T` 是一个完整类型（它可以被前置声明）。一些编译器在删除不完整类型时不会发出警告，这可能导致难以检测的错误，但 `scoped_ptr`（以及 `Boost.SmartPtr` 中的所有类）具有针对此类情况的特定编译时断言。这使得 `scoped_ptr` 完美地实现了 `Pimpl` 习惯用法。

`boost::scoped_ptr<T>` 函数等同于 `const std::auto_ptr<T>`，但它还有一个 `reset()` 函数。

## 还有更多...

这个类非常快。在大多数情况下，编译器会将使用 `scoped_ptr` 的代码优化成接近我们手写的机器代码（如果编译器检测到某些函数不抛出异常，有时甚至更好）。

## 参见

+   `Boost.SmartPtr` 库的文档包含了许多示例以及关于所有智能指针类的其他有用信息。您可以在[http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm)上阅读它。

# 在方法间使用类指针的引用计数

想象一下，你有一些包含数据的动态分配的结构，你想要在不同的执行线程中处理它。执行此操作的代码如下：

[PRE3]

我们不能在 `while` 循环的末尾释放 `p`，因为它可能仍然被运行进程函数的线程使用。进程函数不能删除 `p`，因为它们不知道其他线程已经不再使用它了。

## 准备工作

这个配方使用了 `Boost.Thread` 库，它不是一个仅包含头文件的库，因此你的程序需要链接到 `libboost_thread` 和 `libboost_system` 库。在继续阅读之前，请确保你理解了线程的概念。有关使用线程的配方，请参阅 *参见* 部分。

你还需要一些关于 `boost::bind` 或 `std::bind` 的基本知识，它们几乎是相同的。

## 如何做...

如你所猜，在 Boost（和 C++11）中有一个类可以帮助你处理这个问题。它被称为 `boost::shared_ptr`，它可以被用作：

[PRE4]

这方面的另一个例子如下：

[PRE5]

## 它是如何工作的...

`shared_ptr` 类内部有一个原子引用计数器。当你复制它时，引用计数器会增加，当其析构函数被调用时，引用计数器会减少。当引用计数器等于零时，`delete` 会调用 `shared_ptr` 指向的对象。

现在，让我们找出在 `boost::thread` (`boost::bind(&process_sp1, p)`) 的情况下发生了什么。`process_sp1` 函数接受一个引用作为参数，那么为什么我们在退出 `while` 循环时它没有被释放呢？答案是简单的。`bind()` 返回的功能对象包含共享指针的一个副本，这意味着指向 `p` 的数据不会在功能对象被销毁之前被释放。

回到 `boost::make_shared`，让我们看看 `shared_ptr<std::string> ps(new int(0))`。在这种情况下，我们有两个 `new` 调用：首先是在构造一个指向整数的指针时，其次是在构造 `shared_ptr` 类（它使用 `new` 调用在堆上分配一个原子计数器）。但是，当我们使用 `make_shared` 构造 `shared_ptr` 时，只有一个 `new` 调用会被执行。它将分配一块内存，并在其中构造一个原子计数器和 `int` 对象。

## 还有更多...

原子引用计数器保证了`shared_ptr`在多线程中的正确行为，但您必须记住，原子操作并不像非原子操作那样快。在C++11兼容的编译器上，您可以使用`std::move`（以这种方式移动共享指针的构造函数，使得原子计数器既不增加也不减少）来减少原子操作的次数。

`shared_ptr`和`make_shared`类是C++11的一部分，并在`std::`命名空间中的头文件`<memory>`中声明。

## 参考以下内容

+   请参考[第5章](ch05.html "第5章。多线程")，*多线程*，以获取有关`Boost.Thread`和原子操作更多信息。

+   请参考[第1章](ch01.html "第1章。开始编写您的应用程序")中的*重新排序函数参数*配方，*开始编写您的应用程序*，以获取有关`Boost.Bind`更多信息。

+   请参考[第1章](ch01.html "第1章。开始编写您的应用程序")中的*将值绑定为函数参数*配方，*开始编写您的应用程序*，以获取有关`Boost.Bind`更多信息。

+   `Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息。您可以在[http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm)上阅读它。

# 管理未离开作用域的数组指针

我们已经在*管理未离开作用域的类的指针*配方中看到了如何管理资源指针。但是，当我们处理数组时，我们需要调用`delete[]`而不是简单的`delete`，否则将会有内存泄漏。请看以下代码：

[PRE6]

## 准备工作

对于这个配方，需要了解C++异常和模板。

## 如何做...

`Boost.SmartPointer`库不仅包含`scoped_ptr<>`类，还包含`scoped_array<>`类。

[PRE7]

## 它是如何工作的...

它的工作方式就像一个`scoped_ptr<>`类，但在析构函数中调用`delete[]`而不是`delete`。

## 还有更多...

`scoped_array<>`类具有与`scoped_ptr<>`相同的安全性和设计。它没有额外的内存分配，也没有虚拟函数调用。它不能被复制，也不是C++11的一部分。

## 参考以下内容

+   `Boost.SmartPtr`库的文档包含了许多关于所有智能指针类的示例和其他有用信息。您可以在[http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm)上阅读它。

# 在方法间使用数组指针的引用计数

我们继续处理指针，我们的下一个任务是引用计数一个数组。让我们看看一个从流中获取一些数据并在不同线程中处理它的程序。执行此操作的代码如下：

[PRE8]

与*在方法间使用指针的引用计数*配方中出现的相同问题。

## 准备工作

此配方使用 `Boost.Thread` 库，它不是一个仅包含头文件的库，因此您的程序需要链接到 `libboost_thread` 和 `libboost_system` 库。在继续阅读之前，请确保您理解线程的概念。

您还需要了解一些关于 `boost::bind` 或 `std::bind` 的基础知识，它们几乎相同。

## 如何操作...

有三种解决方案。它们之间的主要区别在于 `data_cpy` 变量的类型和构造。这些解决方案都做了与配方开头描述的完全相同的事情，但没有内存泄漏。解决方案如下：

+   第一种解决方案：

    [PRE9]

+   第二种解决方案：

    自从 Boost 1.53 以来，`shared_ptr` 本身就可以处理数组：

    [PRE10]

+   第三种解决方案：

    [PRE11]

## 它是如何工作的...

在这些示例中的每一个，共享类都会计算引用数，并在引用数变为零时调用 `delete[]`。前两个示例是微不足道的。在第三个示例中，我们为共享指针提供了一个 `deleter` 对象。这个 `deleter` 对象将代替默认的 `delete` 调用。这个 `deleter` 与 C++11 中的 `std::unique_ptr` 和 `std::shared_ptr` 中使用的相同。

## 还有更多...

第一种解决方案是传统的 Boost；在 Boost 1.53 之前，第二种解决方案的功能并未在 `shared_ptr` 中实现。

第二种解决方案是最快的（它使用的 `new` 调用较少），但它只能与 Boost 1.53 及更高版本一起使用。

第三种解决方案是最便携的。它可以与较老的 Boost 版本以及 C++11 STL 的 `shared_ptr<>`（只需别忘了将 `boost::checked_array_deleter<T>()` 改为 `std::default_delete<T[]>()`）一起使用。

## 另请参阅

+   `Boost.SmartPtr` 库的文档包含了许多示例和其他关于所有智能指针类的有用信息。您可以在 [http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm) 中了解它。

# 在变量中存储任何功能对象

C++ 有一种语法可以处理函数指针和成员函数指针。而且，这很好！然而，这个机制很难与功能对象一起使用。考虑当你正在开发一个库，其 API 在头文件中声明，实现则在源文件中。这个库应该有一个接受任何功能对象的函数。你将如何传递一个功能对象给它？看看下面的代码：

[PRE12]

## 准备工作

在开始此配方之前，建议阅读 [第 1 章](ch01.html "第 1 章。开始编写您的应用程序") 中关于 *在容器/变量中存储任何值* 的配方。

您还需要了解一些关于 `boost::bind` 或 `std::bind` 的基础知识，它们几乎相同。

## 如何操作...

让我们看看如何修复示例，并使 `process_integers` 接受功能对象：

1.  有一个解决方案，它被称为 `Boost.Function` 库。它允许你存储任何函数、成员函数或功能对象，如果其签名与模板参数中描述的匹配：

    [PRE13]

    `boost::function` 类有一个默认构造函数，并且处于空状态。

1.  检查空/默认构造状态可以这样做：

    [PRE14]

## 它是如何工作的...

`fobject_t` 方法在其自身中存储功能对象的 数据并擦除它们的精确类型。使用以下代码中的 `boost::function` 对象是安全的：

[PRE15]

这让你想起了 `boost::any` 类吗？它使用相同的技巧——类型擦除来存储任何函数对象。

## 还有更多...

`Boost.Function` 库有大量的优化；它可能在不进行额外内存分配的情况下存储小的函数对象，并且有优化的移动赋值运算符。它被视为 C++11 STL 库的一部分，并在 `std::` 命名空间中的 `<functional>` 头文件中定义。

但是，记住 `boost::function` 对编译器意味着一个优化障碍。这意味着：

[PRE16]

将被编译器优化得更好

[PRE17]

这就是为什么你应该尽量避免在实际上不需要时使用 `Boost.Function`。在某些情况下，C++11 的 `auto` 关键字可能更方便：

[PRE18]

## 参见

+   `Boost.Function` 的官方文档包含更多示例、性能指标和类参考文档。你可以在 [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html) 了解相关信息。

+   *在变量中传递函数指针* 的配方。

+   *在变量中传递 C++11 lambda 函数* 的配方。

# 在变量中传递函数指针

我们正在继续之前的示例，现在我们想在 `process_integeres()` 方法中传递一个函数指针。我们应该只为函数指针添加重载，还是有一个更优雅的方法？

## 准备工作

这个配方是继续之前的配方。你必须先阅读之前的配方。

## 如何做到这一点...

由于 `boost::function<>` 也可以从函数指针构造，因此无需进行任何操作：

[PRE19]

## 它是如何工作的...

将 `my_ints_function` 的指针存储在 `boost::function` 类中，并且对 `boost::function` 的调用将被转发到存储的指针。

## 还有更多...

`Boost.Function` 库为函数指针提供了良好的性能，并且它不会在堆上分配内存。然而，无论你在 `boost::function` 中存储什么，它都会使用 RTTI。如果你禁用 RTTI，它将继续工作，但会显著增加编译二进制文件的大小。

## 参见

+   `Boost.Function` 的官方文档包含更多示例、性能指标和类参考文档。你可以在 [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html) 了解相关信息。

+   `*在变量中传递 C++11 lambda 函数*` 配方。

# 在变量中传递 C++11 lambda 函数

我们将继续使用上一个示例，现在我们想要在我们的 `process_integers()` 方法中使用一个 lambda 函数。

## 准备工作

这个配方是延续前两个配方的系列。您必须先阅读它们。您还需要一个兼容 C++11 的编译器或至少一个具有 C++11 lambda 支持的编译器。

## 如何实现...

由于 `boost::function<>` 也可以与任何难度的 lambda 函数一起使用，因此无需进行任何操作：

[PRE20]

## 还有更多...

`Boost.Functional` 中 lambda 函数存储的性能与其他情况相同。当 lambda 表达式产生的功能对象足够小，可以放入 `boost::function` 的实例中时，不会执行动态内存分配。调用存储在 `boost::function` 中的对象的速度接近通过指针调用函数的速度。对象的复制速度接近构造 `boost::function` 的速度，并在类似情况下将精确地使用动态内存分配。移动对象不会分配和释放内存。

## 参考信息

+   关于性能和 `Boost.Function` 的更多信息可以在官方文档页面上找到：[http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html)

# 指针容器

有时候我们需要在容器中存储指针。例如：在容器中存储多态数据，强制容器中数据的快速复制，以及容器中操作数据的严格异常要求。在这种情况下，C++程序员有以下选择：

+   在容器中存储指针并使用运算符 `delete` 处理它们的销毁：

    [PRE21]

    这种方法容易出错，并且需要大量编写

+   在容器中存储智能指针：

    对于 C++03 版本：

    [PRE22]

    `std::auto_ptr` 方法已被弃用，不建议在容器中使用。此外，此示例在 C++11 中无法编译。

    对于 C++11 版本：

    [PRE23]

    这种解决方案是一个好方案，但不能用于 C++03，并且您仍然需要编写一个比较器功能对象

+   在容器中使用 `Boost.SmartPtr`：

    [PRE24]

    这种解决方案是可移植的，但您仍然需要编写比较器，并且它增加了性能惩罚（原子计数器需要额外的内存，其增加/减少操作不如非原子操作快）

## 准备工作

为了更好地理解这个配方，需要了解 STL 容器。

## 如何实现...

`Boost.PointerContainer` 库提供了一个良好且可移植的解决方案：

[PRE25]

## 它是如何工作的...

`Boost.PointerContainer` 库包含 `ptr_array`、`ptr_vector`、`ptr_set`、`ptr_multimap` 等类。所有这些容器都能简化你的生活。当处理指针时，它们将在析构函数中释放指针，并简化对指针所指向数据的访问（无需在 `assert(*s.begin() == 0);` 中进行额外的解引用）。

## 更多内容...

之前的示例并没有克隆指针数据，但当我们想要克隆一些数据时，我们只需要在要克隆的对象的命名空间中定义一个独立的函数，例如 `new_clone()`。此外，如果你包含了 `<boost/ptr_container/clone_allocator.hpp>` 头文件，你可以使用默认的 `T* new_clone( const T& r )` 实现，如下面的代码所示：

[PRE26]

## 参见

+   官方文档包含了每个类的详细参考，你可以在 [http://www.boost.org/doc/libs/1_53_0/libs/ptr_container/doc/ptr_container.html](http://www.boost.org/doc/libs/1_53_0/libs/ptr_container/doc/ptr_container.html) 上阅读相关信息。

+   本章的前四个示例将为你提供一些智能指针使用的例子

# 在作用域退出时执行某些操作

如果你处理的是像 Java、C# 或 Delphi 这样的语言，你显然使用了 `try{} finally{}` 构造或 D 语言的 `scope(exit)`。让我简要地描述一下这些语言构造的功能。

当程序通过返回或异常离开当前作用域时，`finally` 或 `scope(exit)` 块中的代码将被执行。这种机制非常适合实现如以下代码片段所示的 **RAII** 模式：

[PRE27]

在 C++ 中有办法做这样的事情吗？

## 准备工作

需要基本的 C++ 知识来完成这个示例。了解抛出异常时的代码行为将很有用。

## 如何实现...

`Boost.ScopeExit` 库被设计用来解决这类问题：

[PRE28]

## 它是如何工作的...

变量 `f` 通过 `BOOST_SCOPE_EXIT(f)` 以值的方式传递。当程序离开执行范围时，`BOOST_SCOPE_EXIT(f) {` 和 `} BOOST_SCOPE_EXIT_END` 之间的代码将被执行。如果我们希望通过引用传递值，请在 `BOOST_SCOPE_EXIT` 宏中使用 `&` 符号。如果我们希望传递多个值，只需用逗号将它们分开。

### 注意

在某些编译器上，对指针的传递引用效果不佳。`BOOST_SCOPE_EXIT(&f)` 宏无法在那里编译，这就是为什么我们在示例中没有通过引用捕获它的原因。

## 更多内容...

要在成员函数内部捕获它，我们使用一个特殊的符号 `this_`：

[PRE29]

`Boost.ScopeExit` 库在堆上不分配额外的内存，也不使用虚函数。使用默认语法，不要定义 `BOOST_SCOPE_EXIT_CONFIG_USE_LAMBDAS`，因为否则作用域退出将使用 `boost::function` 实现，这可能会分配额外的内存并引入优化屏障。

## 参见

+   官方文档包含了更多示例和用例。你可以在 [http://www.boost.org/doc/libs/1_53_0/libs/scope_exit/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/scope_exit/doc/html/index.html) 了解相关信息。

# 通过派生类的成员初始化基类

让我们看看以下示例。我们有一个具有虚函数并且必须使用对 `std::ostream` 对象的引用进行初始化的基类：

[PRE30]

我们还有一个具有 `std::ostream` 对象并实现 `do_process()` 函数的派生类：

[PRE31]

在编程中，这种情况并不常见，但当这种错误发生时，并不总是简单就能想到绕过它的方法。有些人试图通过改变 `logger_` 和基类初始化的顺序来绕过它：

[PRE32]

它不会像他们预期的那样工作，因为直接基类在非静态数据成员之前初始化，无论成员初始化器的顺序如何。

## 准备工作

需要具备基本的 C++ 知识才能使用此配方。

## 如何做到这一点...

`Boost.Utility` 库为这类情况提供了一个快速解决方案；它被称为 `boost::base_from_member` 模板。要使用它，你需要执行以下步骤：

1.  包含 `base_from_member.hpp` 头文件：

    [PRE33]

1.  从 `boost::base_from_member<T>` 派生你的类，其中 `T` 是必须在基类之前初始化的类型（注意基类的顺序；`boost::base_from_member<T>` 必须放在使用 `T` 的类之前）：

    [PRE34]

1.  正确编写构造函数如下：

    [PRE35]

## 它是如何工作的...

如果直接基类在非静态数据成员之前初始化，并且如果直接基类会按照它们在基类指定列表中出现的声明顺序初始化，我们需要以某种方式使基类成为我们的非静态数据成员。或者创建一个具有所需成员的成员字段的基类：

[PRE36]

## 还有更多...

正如你所见，`base_from_member` 有一个整数作为第二个模板参数。这是为了处理我们需要多个相同类型的 `base_from_member` 类的情况：

[PRE37]

`boost::base_from_member` 类既不应用额外的动态内存分配，也没有虚函数。当前的实现不支持 C++11 特性（如完美转发和变长模板），但在 Boost 的 trunk 分支中，有一个可以充分利用 C++11 优势的实现。它可能将在最近的未来合并到发布分支中。

## 参见

+   `Boost.Utility` 库包含了许多有用的类和方法；有关获取更多信息，请参阅 [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)。

+   在 [第 1 章](ch01.html "第 1 章。开始编写您的应用程序") 的 *Making a noncopyable class* 配方中，*Starting to Write Your Application*，包含了 `Boost.Utility` 中类的更多示例。

+   此外，[第1章](ch01.html "第1章。开始编写您的应用程序")中的*使用C++11移动模拟*配方包含来自`Boost.Utility`类的更多示例。
