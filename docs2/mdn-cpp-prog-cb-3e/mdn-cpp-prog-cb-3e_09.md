# 9

# 健壮性和性能

当选择以性能和灵活性为主要目标的面向对象编程语言时，C++ 常常是首选。现代 C++ 提供了语言和库功能，例如右值引用、移动语义和智能指针。

当与良好的异常处理实践、常量正确性、类型安全转换、资源分配和释放相结合时，C++ 使开发者能够编写更好、更健壮、更高效的代码。本章的食谱涵盖了所有这些基本主题。

本章包括以下食谱：

+   使用异常进行错误处理

+   为不抛出异常的函数使用 `noexcept`

+   确保程序常量正确性

+   创建编译时常量表达式

+   创建即时函数

+   在常量评估上下文中优化代码

+   在常量表达式中使用虚函数调用

+   执行正确的类型转换

+   实现移动语义

+   使用 `unique_ptr` 独特拥有内存资源

+   使用 `shared_ptr` 共享内存资源

+   使用 `<=>` 运算符进行一致比较

+   安全地比较有符号和无符号整数

我们将从这个章节开始，介绍一些处理异常的食谱。

# 使用异常进行错误处理

异常是程序运行时可能出现的异常情况的一种响应。它们使控制流转移到程序的另一部分。与返回错误代码相比，异常是一种更简单、更健壮的错误处理机制，后者可能会极大地复杂化和杂乱代码。在本食谱中，我们将探讨与抛出和处理异常相关的关键方面。

## 准备工作

本食谱要求您具备抛出异常（使用 `throw` 语句）和捕获异常（使用 `try...catch` 块）的机制的基本知识。本食谱侧重于异常周围的良好实践，而不是 C++ 语言中异常机制的细节。

## 如何操作...

使用以下实践来处理异常：

+   通过值抛出异常：

    ```cpp
    void throwing_func()
    {
      throw std::runtime_error("timed out");
    }
    void another_throwing_func()
    {
      throw std::system_error(
        std::make_error_code(std::errc::timed_out));
    } 
    ```

+   通过引用捕获异常，或者在大多数情况下，通过常量引用捕获：

    ```cpp
    try
    {
      throwing_func(); // throws std::runtime_error
    }
    catch (std::exception const & e)
    {
      std::cout << e.what() << '\n';
    } 
    ```

+   在捕获类层次结构中的多个异常时，从最派生类到层次结构的基类按顺序排列 `catch` 语句：

    ```cpp
    auto exprint = [](std::exception const & e)
    {
      std::cout << e.what() << '\n';
    };
    try
    {
      another_throwing_func(); // throws std::system_error
                               // 1st catch statements catches it
    }
    catch (std::system_error const & e)
    {
      exprint(e);
    }
    catch (std::runtime_error const & e)
    {
      exprint(e);
    }
    catch (std::exception const & e)
    {
      exprint(e);
    } 
    ```

+   使用 `catch(...)` 捕获所有异常，无论它们的类型如何：

    ```cpp
    try
    {
      throwing_func();
    }
    catch (std::exception const & e)
    {
      std::cout << e.what() << '\n';
    }
    catch (...)
    {
      std::cout << "unknown exception" << '\n';
    } 
    ```

+   使用 `throw;` 重新抛出当前异常。这可以用于为多个异常创建单个异常处理函数。

+   当您想隐藏异常的原始位置时，抛出异常对象（例如，`throw e;`）：

    ```cpp
    void handle_exception()
    {
      try
      {
        throw; // throw current exception
      }
      catch (const std::logic_error & e)
      { /* ... */ }
      catch (const std::runtime_error & e)
      { /* ... */ }
      catch (const std::exception & e)
      { /* ... */ }
    }
    try
    {
      throwing_func();
    }
    catch (...)
    {
      handle_exception();
    } 
    ```

## 它是如何工作的...

大多数函数必须指示其执行的成败。这可以通过不同的方式实现。以下是一些可能性：

+   返回一个错误代码（对于成功有一个特殊值）以指示失败的具体原因：

    ```cpp
    int f1(int& result)
    {
      if (...) return 1;
      // do something
    if (...) return 2;
      // do something more
      result = 42;
      return 0;
    }
    enum class error_codes {success, error_1, error_2};
    error_codes f2(int& result)
    {
      if (...) return error_codes::error_1;
      // do something
    if (...) return error_codes::error_2;
      // do something more
      result = 42;
      return error_codes::success;
    } 
    ```

+   这种变体是只返回布尔值来仅指示成功或失败：

    ```cpp
    bool g(int& result)
    {
      if (...) return false;
      // do something
    if (...) return false;
      // do something more
      result = 42;
      return true;
    } 
    ```

+   另一个替代方案是返回无效对象、空指针或空的`std::optional<T>`对象：

    ```cpp
    std::optional<int> h()
    {
      if (...) return {};
      // do something
    if (...) return {};
      // do something more
    return 42;
    } 
    ```

在任何情况下，都应该检查函数的返回值。这可能导致复杂、杂乱、难以阅读和维护的现实代码。此外，检查函数返回值的过程始终执行，无论函数是成功还是失败。另一方面，只有当函数失败时才会抛出并处理异常，这应该比成功的执行更少发生。这实际上可能导致比返回并测试错误代码的代码更快。

异常和错误代码不是互斥的。异常应该仅用于在异常情况下转移控制流，而不是用于控制程序中的数据流。

类构造函数是特殊的函数，它们不返回任何值。它们应该构建一个对象，但在失败的情况下，它们将无法通过返回值来指示这一点。异常应该是一个构造函数用来指示失败机制的机制。与**资源获取即初始化（RAII**）惯用法一起，这确保了在所有情况下资源的安全获取和释放。另一方面，异常不允许离开析构函数。当这种情况发生时，程序会通过调用`std::terminate()`异常终止。这是在发生另一个异常时调用析构函数进行栈回溯的情况。当发生异常时，栈从抛出异常的点回溯到处理异常的块。这个过程涉及到所有这些栈帧中所有局部对象的销毁。

如果在此过程中正在销毁的对象的析构函数抛出异常，则应开始另一个栈回溯过程，这将与已经进行的过程冲突。因此，程序会异常终止。

处理构造函数和析构函数中的异常的规则如下：

+   使用异常来指示构造函数中发生的错误。

+   不要在析构函数中抛出或让异常离开。

可以抛出任何类型的异常。然而，在大多数情况下，你应该抛出临时对象，并通过常量引用捕获异常。捕获（常量）引用的原因是避免异常类型的切片。让我们考虑以下代码片段：

```cpp
class simple_error : public std::exception
{
public:
  virtual const char* what() const noexcept override
 {
    return "simple exception";
  }
};
try
{
   throw simple_error{};
}
catch (std::exception e)
{
   std::cout << e.what() << '\n'; // prints "Unknown exception"
} 
```

我们抛出一个`simple_error`对象，但通过值捕获一个`std::exception`对象。这是`simple_error`的基类型。发生**切片**过程，派生类型信息丢失，只保留对象的`std::exception`部分。因此，打印的消息是*未知异常*，而不是预期的*简单异常*。使用引用可以避免对象切片。

以下是一些关于抛出异常的指南：

+   建议抛出标准异常或从 `std::exception` 或其他标准异常派生的自定义异常。这样做的原因是标准库提供了旨在作为表示异常首选方案的异常类。你应该使用已经可用的那些，当这些不够用时，基于标准异常构建自己的异常。这样做的主要好处是一致性，并帮助用户通过基类 `std::exception` 捕获异常。

+   避免抛出内置类型的异常，如整数。这样做的原因是数字对用户来说信息量很小，用户必须知道它代表什么，而一个对象可以提供上下文信息。例如，`throw 13;` 对用户来说没有任何说明，但 `throw access_denied_exception{};` 仅从类名本身就携带了大量的隐含信息，借助数据成员，它还可以携带关于异常情况的有用或必要信息。

+   当使用提供自己异常层次结构的库或框架时，优先抛出该层次结构中的异常或从它派生的自定义异常，至少在代码中与它紧密相关的部分。这样做的主要原因是为了保持利用库 API 的代码的一致性。

## 还有更多...

如前所述，当你需要创建自己的异常类型时，应从可用的标准异常之一派生，除非你正在使用具有自己异常层次结构的库或框架。C++标准定义了几个需要考虑此类目的的异常类别：

+   `std::logic_error` 表示指示程序逻辑错误的异常，例如无效的参数和范围之外的索引。有各种从标准派生的类，如 `std::invalid_argument`、`std::out_of_range` 和 `std::length_error`。

+   `std::runtime_error` 表示指示超出程序范围或由于各种因素（包括外部因素）无法预测的错误。C++标准还提供了从 `std::runtime_error` 派生的几个类，包括 `std::overflow_error`、`std::underflow_error`、`std::system_error` 和 C++20 中的 `std::format_error`。

+   以 `bad_` 为前缀的异常，例如 `std::bad_alloc`、`std::bad_cast` 和 `std::bad_function_call`，表示程序中的各种错误，如内存分配失败、动态类型转换失败或函数调用失败。

所有这些异常的基类是 `std::exception`。它有一个非抛出（non-throwing）的虚方法 `what()`，该方法返回一个指向字符数组的指针，该数组表示错误的描述。

当您需要从标准异常派生自定义异常时，使用适当的类别，例如逻辑错误或运行时错误。如果这些类别都不合适，则可以直接从`std::exception`派生。以下是从标准异常派生时可以使用的可能解决方案列表：

+   如果您需要从`std::exception`派生，则重写虚拟方法`what()`以提供错误描述：

    ```cpp
    class simple_error : public std::exception
    {
    public:
      virtual const char* what() const noexcept override
     {
        return "simple exception";
      }
    }; 
    ```

+   如果您从`std::logic_error`或`std::runtime_error`派生，并且只需要提供一个不依赖于运行时数据的静态描述，则将描述文本传递给基类构造函数：

    ```cpp
    class another_logic_error : public std::logic_error
    {
    public:
      another_logic_error():
        std::logic_error("simple logic exception")
      {}
    }; 
    ```

+   如果您从`std::logic_error`或`std::runtime_error`派生，但描述消息依赖于运行时数据，则提供一个带有参数的构造函数并使用它们来构建描述消息。您可以将描述消息传递给基类构造函数或从重写的`what()`方法返回它：

    ```cpp
    class advanced_error : public std::runtime_error
    {
      int error_code;
      std::string make_message(int const e)
     {
        std::stringstream ss;
        ss << "error with code " << e;
        return ss.str();
      }
    public:
      advanced_error(int const e) :
        std::runtime_error(make_message(e).c_str()),error_code(e)
      {
      }
      int error() const noexcept
     {
        return error_code;
      }
    }; 
    ```

要查看标准异常类的完整列表，您可以访问[`en.cppreference.com/w/cpp/error/exception`](https://en.cppreference.com/w/cpp/error/exception)页面。

## 相关内容

+   *第八章*，*处理线程函数抛出的异常*，了解如何处理从主线程或它所加入的线程抛出的工作线程中的异常

+   *使用`noexcept`指定不抛出异常的函数*，以了解如何通知编译器一个函数不应该抛出异常

# 使用`noexcept`指定不抛出异常的函数

异常规范是一种语言特性，可以启用性能改进，但另一方面，如果使用不当，可能会导致程序异常终止。C++03 中的异常规范，允许您指示函数可以抛出哪些类型的异常，已在 C++11 中弃用，并在 C++17 中删除。它被 C++11 的`noexcept`指定符所取代。此外，使用`throw()`指定符来指示函数抛出，而不指示可以抛出的异常类型，已在 C++17 中弃用，并在 C++20 中完全删除。`noexcept`指定符仅允许您指示函数不抛出异常（与旧的`throw`指定符相反，旧的`throw`指定符可以指示函数可以抛出的类型列表）。本食谱提供了有关 C++中现代异常规范的信息，以及何时使用它们的指南。

## 如何实现...

使用以下构造来指定或查询异常规范：

+   在函数声明中使用`noexcept`来指示该函数不会抛出任何异常：

    ```cpp
    void func_no_throw() noexcept
    {
    } 
    ```

+   在函数声明中使用`noexcept(expr)`，例如模板元编程，来指示函数可能抛出或可能不抛出异常，这取决于评估为`bool`的条件：

    ```cpp
    template <typename T>
    T generic_func_1()
     noexcept(std::is_nothrow_constructible_v<T>)
    {
      return T{};
    } 
    ```

+   在编译时使用`noexcept`运算符来检查表达式是否声明为不抛出任何异常：

    ```cpp
    template <typename T>
    T generic_func_2() noexcept(noexcept(T{}))
    {
      return T{};
    }
    template <typename F, typename A>
    auto func(F&& f, A&& arg) noexcept
    {
      static_assert(noexcept(f(arg)), "F is throwing!");
      return f(arg);
    }
    std::cout << noexcept(generic_func_2<int>) << '\n'; 
    ```

## 它是如何工作的...

截至 C++17，异常指定是函数类型的一部分，但不是函数签名的一部分；它可以作为任何函数声明的部分出现。因为异常指定不是函数签名的一部分，所以两个函数签名不能仅在异常指定上有所不同。

在 C++17 之前，异常指定不是函数类型的一部分，只能作为 lambda 声明或顶层函数声明的部分出现；它们甚至不能出现在 `typedef` 或类型别名声明中。关于异常指定的进一步讨论仅限于 C++17 标准。

抛出异常的过程可以通过几种方式来指定：

+   如果没有异常指定，则函数可能抛出异常。

+   `noexcept(false)` 等同于没有异常指定。

+   `noexcept(true)` 和 `noexcept` 表示一个函数不会抛出任何异常。

+   `throw()` 等同于 `noexcept(true)`，但在 C++17 中被弃用，并在 C++20 中完全删除。

使用异常指定必须谨慎进行，因为如果一个异常（无论是直接抛出还是从被调用的另一个函数中抛出）使一个标记为非抛出的函数结束，程序将立即以调用 `std::terminate()` 的方式异常终止。

不抛出异常的函数指针可以隐式转换为可能抛出异常的函数指针，但反之则不行。另一方面，如果一个虚函数具有非抛出异常指定，这表明所有重写声明的所有重写都必须保留此指定，除非重写的函数被声明为已删除。

在编译时，可以使用操作符 `noexcept` 来检查一个函数是否声明为非抛出。此操作符接受一个表达式，如果表达式被声明为非抛出或 `false`，则返回 `true`。它不会评估它检查的表达式。

`noexcept` 操作符，连同 `noexcept` 指定符一起，在模板元编程中特别有用，用于指示一个函数对于某些类型是否可能抛出异常。它还与 `static_assert` 声明一起使用，以检查表达式是否违反了函数的非抛出保证，如 *如何做...* 部分的示例所示。

以下代码提供了更多关于 `noexcept` 操作符如何工作的示例：

```cpp
int double_it(int const i) noexcept
{
  return i + i;
}
int half_it(int const i)
{
  throw std::runtime_error("not implemented!");
}
struct foo
{
  foo() {}
};
std::cout << std::boolalpha
  << noexcept(func_no_throw()) <<  '\n' // true
  << noexcept(generic_func_1<int>()) <<  '\n' // true
  << noexcept(generic_func_1<std::string>()) <<  '\n'// true
  << noexcept(generic_func_2<int>()) << '\n' // true
  << noexcept(generic_func_2<std::string>()) <<  '\n'// true
  << noexcept(generic_func_2<foo>()) <<  '\n' // false
  << noexcept(double_it(42)) <<  '\n' // true
  << noexcept(half_it(42)) <<  '\n' // false
  << noexcept(func(double_it, 42)) <<  '\n' // true
  << noexcept(func(half_it, 42)) << '\n';            // true 
```

重要的是要注意，`noexcept` 指定符不提供编译时对异常的检查。它只代表用户通知编译器一个函数不期望抛出异常的一种方式。编译器可以使用这一点来启用某些优化。例如，`std::vector` 如果其移动构造函数是 `noexcept`，则会移动元素，否则会复制它们。

## 更多内容...

如前所述，使用 `noexcept` 指示符声明的函数由于异常而退出会导致程序异常终止。因此，应谨慎使用 `noexcept` 指示符。它的存在可以启用代码优化，这有助于提高性能同时保持*强异常保证*。一个例子是库容器。

C++语言提供了几个异常保证级别：

+   第一级，*无异常保证*，不提供任何保证。如果发生异常，没有任何指示表明程序是否处于有效状态。资源可能会泄漏，内存可能会损坏，对象的不变性可能会被破坏。

+   *基本异常保证*是保证的最简单级别，它确保在抛出异常后，对象处于一致和可用的状态，没有资源泄漏发生，且不变性得到保留。

+   *强异常保证*指定操作要么成功完成，要么以抛出异常的方式完成，该异常使程序处于操作开始之前的状态。这确保了提交或回滚语义。

+   *无抛出异常保证*实际上是其中最强烈的，它指定操作保证不会抛出任何异常并成功完成。

许多标准容器为其一些操作提供了强异常保证。例如，vector 的 `push_back()` 方法。可以通过使用移动构造函数或移动赋值运算符而不是向量元素类型的复制构造函数或复制赋值运算符来优化此方法。然而，为了保持其强异常保证，这只能在移动构造函数或赋值运算符不抛出异常的情况下进行。如果任一抛出异常，则必须使用复制构造函数或赋值运算符。

如果其类型参数的移动构造函数带有 `noexcept` 标记，`std::move_if_noexcept()` 实用函数会这样做。能够表明移动构造函数或移动赋值运算符不会抛出异常可能是使用 `noexcept` 的最重要的场景之一。

考虑以下异常指定规则：

+   如果一个函数可能抛出异常，则不要使用任何异常指定符。

+   仅标记那些保证不会抛出异常的函数。

+   仅标记那些可能基于条件抛出异常的带有 `noexcept(expression)` 的函数。

这些规则很重要，因为，如前所述，从 `noexcept` 函数抛出异常将立即通过调用 `std::terminate()` 终止程序。

## 参见

+   *使用异常进行错误处理*，以探索在 C++ 语言中使用异常的最佳实践

# 确保程序的正确性保持恒定。

虽然没有正式的定义，但常量正确性意味着不应该被修改的对象（是不可变的）保持不变。作为开发者，您可以通过使用 `const` 关键字来声明参数、变量和成员函数来强制执行这一点。在本食谱中，我们将探讨常量正确性的好处以及如何实现它。

## 如何做到...

为了确保程序具有常量正确性，您应该始终将以下内容声明为常量：

+   函数参数不应该在函数内部被修改：

    ```cpp
    struct session {};
    session connect(std::string const & uri,
     int const timeout = 2000)
    {
      /* do something */
    return session { /* ... */ };
    } 
    ```

+   不变的类数据成员：

    ```cpp
    class user_settings
    {
    public:
      int const min_update_interval = 15;
      /* other members */
    }; 
    ```

+   从外部看，不修改对象状态的类成员函数：

    ```cpp
    class user_settings
    {
      bool show_online;
    public:
      bool can_show_online() const {return show_online;}
      /* other members */
    }; 
    ```

+   在其整个生命周期中值不改变的函数局部变量：

    ```cpp
    user_settings get_user_settings()
    {
      return user_settings {};
    }
    void update()
    {
      user_settings const us = get_user_settings();
      if(us.can_show_online()) { /* do something */ }
      /* do more */
    } 
    ```

+   应该绑定到临时（一个右值）以扩展临时寿命到（常量）引用寿命的引用：

    ```cpp
    std::string greetings()
    {
       return "Hello, World!";
    }
    const std::string & s = greetings(); // must use const
    std::cout << s << std::endl; 
    ```

## 它是如何工作的...

将对象和成员函数声明为常量具有几个重要的好处：

+   您防止了对象意外和故意的更改，这在某些情况下可能导致程序行为不正确。

+   您使编译器能够执行更好的优化。

+   您为其他用户记录代码的语义。

常量正确性不是一个个人风格的问题，而是一个应该指导 C++开发的核心理念。

不幸的是，常量正确性的重要性在书籍、C++ 社区和工作环境中尚未得到，并且仍然没有得到足够的强调。但经验法则是，所有不应该改变的内容都应该声明为常量。这应该始终如此，而不仅仅是在开发的后期阶段，当您可能需要清理和重构代码时。

当您将参数或变量声明为常量时，您可以将 `const` 关键字放在类型之前（`const T c`）或之后（`T const c`）。这两种方式是等效的，但无论您使用哪种风格，对声明的读取必须从右侧开始。`const T c` 读取为 *c 是一个常量的 T*，而 `T const c` 读取为 *c 是一个常量 T*。当涉及到指针时，这会变得稍微复杂一些。以下表格展示了各种指针声明及其含义：

| **表达式** | **描述** |
| --- | --- |
| `T* p` | `p` 是一个指向非常量 `T` 的非常量指针。 |
| `const T* p` | `p` 是一个指向常量 `T` 的非常量指针。 |
| `T const * p` | `p` 是一个指向常量 `T` 的非常量指针（与前面的点相同）。 |
| `const T * const p` | `p` 是一个指向常量 `T` 的常量指针。 |
| `T const * const p` | `p` 是一个指向常量 `T` 的常量指针（与前面的点相同）。 |
| `T** p` | `p` 是一个指向非常量指针的非常量指针，该指针指向非常量 `T`。 |
| `const T** p` | `p` 是一个指向非常量指针的非常量指针，该指针指向常量 `T`。 |
| `T const ** p` | 与 `const T** p` 相同。 |
| `const T* const * p` | `p` 是一个指向常量指针的非常量指针，该指针是一个常量 `T`。 |
| `T const * const * p` | 与 `const T* const * p` 相同。 |

表 9.1：指针声明及其含义示例

将 `const` 关键字放在类型之后更自然，因为它与语法解释的方向一致，即从右到左。因此，本书中的所有示例都使用这种风格。

当涉及到引用时，情况类似：`const T & c` 和 `T const & c` 是等价的，这意味着 *c 是指向常量 T 的引用*。然而，`T const & const c`，这意味着 *c 是指向常量 T 的常量引用*，是没有意义的，因为引用——变量的别名——在隐式上是常量的，它们不能被修改来表示指向另一个变量的别名。

一个指向非常量对象的非常量指针，即 `T*`，可以隐式转换为指向常量对象的非常量指针，`T const *`。然而，`T**` 不能隐式转换为 `T const **`（这与 `const T**` 相同）。这是因为这可能导致通过指向非常量对象的指针修改常量对象，如下面的示例所示：

```cpp
int const c = 42;
int* x;
int const ** p = &x; // this is an actual error
*p = &c;
*x = 0;              // this modifies c 
```

如果一个对象是常量，则只能调用其类的常量函数。然而，将成员函数声明为常量并不意味着该函数只能对常量对象进行调用；它也可能意味着该函数不会修改对象的状态，从外部看。这是一个关键方面，但通常被误解。一个类有一个内部状态，它可以通过其公共接口向其客户端公开。

然而，并非所有内部状态都可能被公开，从公共接口可见的内容可能没有在内部状态中的直接表示。（如果你对订单行进行建模，并在内部表示中具有项目数量和项目销售价格字段，那么你可能有一个公开的方法，通过乘以数量和价格来公开订单行金额。）因此，从其公共接口可见的对象状态是一个逻辑状态。将方法定义为常量是一个确保函数不改变逻辑状态的声明。然而，编译器阻止你使用此类方法修改数据成员。为了避免这个问题，应该从常量方法中修改的数据成员应声明为 `mutable`。

在下面的示例中，`computation` 是一个具有 `compute()` 方法的类，它执行长时间运行的计算操作。因为它不影响对象的逻辑状态，所以这个函数被声明为常量。然而，为了避免对相同输入再次计算结果，计算出的值被存储在缓存中。为了能够在常量函数中修改缓存，它被声明为 `mutable`：

```cpp
class computation
{
  double compute_value(double const input) const
 {
    /* long running operation */
return input + 42;
  }
  mutable std::map<double, double> cache;
public:
  double compute(double const input) const
 {
    auto it = cache.find(input);
    if(it != cache.end()) return it->second;
    auto result = compute_value(input);
    cache[input] = result;
    return result;
  }
}; 
```

以下类表示了类似的情况，它实现了一个线程安全的容器。对共享内部数据的访问通过`mutex`进行保护。该类提供了添加和删除值的方法，以及如`contains()`这样的方法，指示项目是否存在于容器中。因为这个成员函数不打算修改对象的逻辑状态，所以它被声明为常量。但是，访问共享内部状态必须通过互斥锁进行保护。为了锁定和解锁互斥锁，必须将修改对象状态的 mutable 操作和互斥锁都声明为`mutable`：

```cpp
template <typename T>
class container
{
  std::vector<T>     data;
  mutable std::mutex mt;
public:
  void add(T const & value)
 {
    std::lock_guard<std::mutex> lock(mt);
    data.push_back(value);
  }
  bool contains(T const & value) const
 {
    std::lock_guard<std::mutex> lock(mt);
    return std::find(std::begin(data), std::end(data), value)
           != std::end(data);
  }
}; 
```

`mutable`指定符允许我们修改使用它的类成员，即使包含的对象被声明为`const`。这是`std::mutex`类型的`mt`成员的情况，即使在声明为`const`的`contains()`方法中也会被修改。

有时，一个方法或运算符会被重载以同时具有常量和非常量版本。这种情况通常出现在下标运算符或提供直接访问内部状态的方法中。这样做的原因是，该方法应该对常量和非常量对象都可用。尽管行为应该不同：对于非常量对象，该方法应允许客户端修改它提供访问的数据，但对于常量对象，则不应修改。因此，非常量下标运算符返回对非常量对象的引用，而常量下标运算符返回对常量对象的引用：

```cpp
class contact {};
class addressbook
{
  std::vector<contact> contacts;
public:
  contact& operator[](size_t const index);
  contact const & operator[](size_t const index) const;
}; 
```

应注意，如果成员函数是常量，即使对象是常量，该成员函数返回的数据可能不是常量。

`const`的一个重要用途是定义对临时对象的引用，如*如何做…*部分最后一条所述。临时对象是一个右值，非`const`左值引用不能绑定到右值。然而，通过将左值引用变为`const`，这是可能的。这会使临时对象的生存期延长到常量引用的生存期。但是，这仅适用于基于堆栈的引用，不适用于对象成员的引用。

## 还有更多...

可以使用`const_cast`转换来移除对象的`const`限定符，但只有在你知道该对象没有被声明为常量时才应使用它。你可以在*执行正确的类型转换*菜谱中了解更多关于此内容。

## 参见

+   *创建编译时常量表达式*，了解`constexpr`指定符以及如何定义可以在编译时评估的变量和函数

+   *创建即时函数*，了解 C++20 的`consteval`指定符，它用于定义保证在编译时评估的函数

+   *执行正确的类型转换*，了解在 C++ 语言中执行正确转换的最佳实践

# 创建编译时常量表达式

在编译时评估表达式的可能性提高了运行时执行效率，因为要运行的代码更少，编译器可以执行额外的优化。编译时常量不仅可以是文本（如数字或字符串），还可以是函数执行的结果。如果函数的所有输入值（无论它们是参数、局部变量还是全局变量）在编译时都是已知的，编译器可以执行该函数，并在编译时提供结果。这就是 C++11 中引入的泛型常量表达式所实现的功能，它在 C++14 中得到了放宽，甚至在 C++20 中进一步放宽。关键字 `constexpr`（代表 *常量表达式*）可以用来声明编译时常量对象和函数。我们已经在前面章节的几个例子中看到了这一点。现在，是时候学习它实际上是如何工作的了。

## 准备工作

C++14 和 C++20 中对泛型常量表达式的处理方式已经放宽，但这给 C++11 引入了一些破坏性变化。例如，在 C++11 中，`constexpr` 函数隐式地是 `const` 的，但在 C++14 中就不再是这种情况了。在本食谱中，我们将讨论 C++20 中定义的泛型常量表达式。

## 如何操作...

当你想使用 `constexpr` 关键字时：

+   定义可以在编译时评估的非成员函数：

    ```cpp
    constexpr unsigned int factorial(unsigned int const n)
    {
      return n > 1 ? n * factorial(n-1) : 1;
    } 
    ```

+   定义可以在编译时执行以初始化 `constexpr` 对象和在此期间调用的成员函数的构造函数：

    ```cpp
    class point3d
    {
      double const x_;
      double const y_;
      double const z_;
    public:
      constexpr point3d(double const x = 0,
     double const y = 0,
     double const z = 0)
        :x_{x}, y_{y}, z_{z}
      {}
      constexpr double get_x() const {return x_;}
      constexpr double get_y() const {return y_;}
      constexpr double get_z() const {return z_;}
    }; 
    ```

+   定义可以在编译时评估其值的变量：

    ```cpp
    constexpr unsigned int size = factorial(6);
    char buffer[size] {0};
    constexpr point3d p {0, 1, 2};
    constexpr auto x = p.get_x(); 
    ```

## 它是如何工作的...

`const` 关键字用于在运行时声明变量为常量；这意味着一旦初始化，它们就不能更改。然而，评估常量表达式可能仍然意味着运行时计算。`constexpr` 关键字用于声明在编译时为常量的变量或可以在编译时执行的功能。`constexpr` 函数和对象可以替代宏和硬编码的文本，而不会产生任何性能损失。

将函数声明为 `constexpr` 并不意味着它总是会在编译时评估。它仅允许在编译时评估的表达式中使用该函数。这仅发生在函数的所有输入值都可以在编译时评估的情况下。然而，该函数也可能在运行时被调用。以下代码显示了同一函数的两个调用，首先是编译时，然后是运行时：

```cpp
constexpr unsigned int size = factorial(6);
// compile time evaluation
int n;
std::cin >> n;
auto result = factorial(n);
// runtime evaluation 
```

关于 `constexpr` 可以使用的地方有一些限制。这些限制随着时间的推移而演变，C++14 和 C++20 中有所变化。为了保持列表的合理性，这里只显示了在 C++20 中需要满足的要求：

+   一个`constexpr`变量必须满足以下要求：

    +   它的类型是一个字面量类型。

    +   它在声明时初始化。

    +   用于初始化变量的表达式是一个常量表达式。

    +   它必须有常量析构。这意味着它不能是类类型或类类型的数组；否则，类类型必须有一个`constexpr`析构函数。

+   一个`constexpr`函数必须满足以下要求：

    +   它不是一个协程。

    +   返回类型以及所有参数的类型都是字面量类型。

    +   至少有一组参数，对于该函数的调用会产生一个常量表达式。

    +   函数体不得包含`goto`语句、标签（除了在`switch`中的`case`和`default`之外），以及非字面量类型或具有静态或线程存储持续时间的局部变量。这个列表点中提到的限制在 C++23 中被移除。

+   一个`constexpr`构造函数必须满足以下要求，除了之前对函数的要求之外：

    +   该类没有虚拟基类。

    +   所有初始化非静态数据成员的构造函数，包括基类，也必须是`constexpr`。

+   自 C++20 起，一个`constexpr`析构函数必须满足以下要求，除了之前对函数的要求之外：

    +   该类没有虚拟基类。

    +   所有销毁非静态数据成员的析构函数，包括基类，也必须是`constexpr`。

这里提到的所有`constexpr`构造函数和析构函数的限制在 C++23 中都被移除了。

对于标准不同版本的要求的完整列表，你应该阅读在[`en.cppreference.com/w/cpp/language/constexpr`](https://en.cppreference.com/w/cpp/language/constexpr)可用的在线文档。

一个`constexpr`函数不是隐式`const`（截至 C++14），所以如果你希望函数不改变对象的逻辑状态，你需要显式使用`const`说明符。然而，一个`constexpr`函数是隐式`inline`的。另一方面，一个声明为`constexpr`的对象是隐式`const`的。以下两个声明是等价的：

```cpp
constexpr const unsigned int size = factorial(6);
constexpr unsigned int size = factorial(6); 
```

在某些情况下，你可能需要在声明中使用`constexpr`和`const`，因为它们会引用声明中的不同部分。在以下示例中，`p`是一个指向常量整数的`constexpr`指针：

```cpp
static constexpr int c = 42;
constexpr int const * p = &c; 
```

如果且仅如果一个引用变量别名一个具有静态存储持续时间或函数的对象，那么引用变量也可以是`constexpr`。以下是一个示例：

```cpp
static constexpr int const & r = c; 
```

在这个示例中，`r`是一个`constexpr`引用，它定义了一个对在前面代码片段中定义的编译时常量变量`c`的别名。

尽管你可以定义静态的`constexpr`变量，但在`constexpr`函数中这样做直到 C++23 之前是不可能的。以下是一个这样的示例：

```cpp
constexpr char symbol_table(int const n)
{
  static constexpr char symbols[] = "!@#$%^&*"; // error until C++23
return symbols[n % 8];
}
int main()
{
    constexpr char s = symbol_table(42);
    std::cout << s << '\n';
} 
```

声明`symbols`变量将生成编译器错误，在 C++23 之前。解决这个问题的一个可能方法是定义变量在`constexpr`函数之外，如下所示：

```cpp
static constexpr char symbols[] = "!@#$%^&*"; // OK
constexpr char symbol_table(int const n)
{
  return symbols[n % 8];
} 
```

在 C++23 中，这个问题得到了解决，它放宽了几个`constexpr`限制，使得解决方案变得不再必要。

在`constexpr`函数中还应提及的一个方面与异常有关。自 C++20 以来，允许在`constexpr`函数中使用 try-catch 块（在此版本之前无法使用）。然而，不允许从常量表达式中抛出异常。尽管你可以在`constexpr`函数中有一个抛出语句，但其行为如下：

+   当在运行时执行时，它将表现得好像没有被声明为`constexpr`。

+   当在编译时执行时，如果执行路径遇到抛出语句，则编译器会发出错误。

这在以下代码片段中得到了体现：

```cpp
constexpr int factorial2(int const n)
{
   if(n <= 0) throw std::invalid_argument("n must be positive");
   return n > 1 ? n * factorial2(n - 1) : 1;
}
int main()
{
   try
   {
      int a = factorial2(5);
      int b = factorial2(-5);
   }
   catch (std::exception const& ex)
   {
      std::cout << ex.what() << std::endl;
   }         
   constexpr int c = factorial2(5);
   constexpr int d = factorial2(-5); // error
} 
```

在此代码片段中：

+   对`factorial2()`的前两次调用是在运行时执行的。第一次调用成功并返回`60`。第二次调用由于参数为负而抛出`std::invalid_argument`异常。

+   第三次调用是在编译时评估的，因为变量`c`被声明为`constexpr`，并且所有函数的输入在编译时也是已知的。调用成功，函数评估结果为`60`。

+   第四次调用也是在编译时评估的，但由于参数为负，应该执行抛出异常的路径。然而，在常量表达式中不允许这样做，因此编译器会发出错误。

## 还有更多...

在 C++20 中，语言中添加了一个新的指定符。这个指定符被称为`constinit`，用于确保具有静态或线程存储持续时间的变量具有静态初始化。在 C++中，变量的初始化可以是静态的或动态的。静态初始化可以是零初始化（当对象的初始值设置为零时）或常量初始化（当初始值设置为编译时表达式时）。以下代码片段显示了零和常量初始化的示例：

```cpp
struct foo
{
  int a;
  int b;
};
struct bar
{
  int   value;
  int*  ptr;
  constexpr bar() :value{ 0 }, ptr{ nullptr } {}
};
std::string text {};  // zero-initialized to unspecified value
double arr[10];       // zero-initialized to ten 0.0
int* ptr;             // zero-initialized to nullptr
foo f = foo();        // zero-initialized to a=0, b=0
foo const fc{ 1, 2 }; // const-initialized at runtime
constexpr bar b;      // const-initialized at compile-time 
```

具有静态存储的变量可以具有静态或动态初始化。在后一种情况下，可能出现难以发现的错误。想象两个在不同的翻译单元中初始化的静态对象。

当一个对象的初始化依赖于另一个对象时，它们的初始化顺序很重要。这是因为依赖于对象的那个对象必须首先初始化。然而，翻译单元初始化的顺序是不确定的，因此无法保证这些对象的初始化顺序。然而，具有静态存储持续时间的变量如果具有静态初始化，则是在编译时初始化的。这意味着当执行翻译单元的动态初始化时，可以安全地使用这些对象。

这正是新指定符`constinit`的目的。它确保具有静态或线程局部存储的变量具有静态初始化，因此其初始化是在编译时执行的：

```cpp
int f() { return 42; }
constexpr int g(bool const c) { return c ? 0 : f(); }
constinit int c = g(true);  // OK
constinit int d = g(false); /* error: variable does not have
                                      a constant initializer */ 
```

它还可以用于非初始化声明中，以指示具有线程存储持续时间的变量已经初始化，如下面的示例所示：

```cpp
extern thread_local constinit int data;
int get_data() { return data; } 
```

你不能在同一个声明中使用超过一个的`constexpr`、`constinit`和`consteval`指定符。

## 参见

+   *创建立即函数*，了解 C++20 的`consteval`指定符，该指定符用于定义保证在编译时评估的函数

+   *确保程序恒定正确性*，以探索恒定正确性的好处以及如何实现它

# 创建立即函数

`constexpr`函数允许在编译时评估函数，前提是它们的所有输入（如果有的话）也必须在编译时可用。然而，这并不保证，`constexpr`函数也可能在运行时执行，正如我们在之前的配方中看到的，*创建编译时常量表达式*。在 C++20 中，引入了函数的新类别：*立即函数*。这些函数保证始终在编译时进行评估；否则，它们会产生错误。立即函数可以作为宏的替代品，并且可能在语言未来的反射和元类开发中很重要。

## 如何实现…

当你想使用`consteval`关键字时：

+   定义必须在编译时评估的非成员函数或函数模板：

    ```cpp
    consteval unsigned int factorial(unsigned int const n)
    {
      return n > 1 ? n * factorial(n-1) : 1;
    } 
    ```

+   定义必须在编译时执行的构造函数，以初始化`constexpr`对象和仅应在编译时调用的成员函数：

    ```cpp
    class point3d
    {
      double x_;
      double y_;
      double z_;
    public:
      consteval point3d(double const x = 0,
     double const y = 0,
     double const z = 0)
        :x_{x}, y_{y}, z_{z}
      {}
      consteval double get_x() const {return x_;}
      consteval double get_y() const {return y_;}
      consteval double get_z() const {return z_;}
    }; 
    ```

## 它是如何工作的…

`consteval`指定符是在 C++20 中引入的。它只能应用于函数和函数模板，并将它们定义为立即函数。这意味着任何函数调用都必须在编译时进行评估，因此产生一个编译时常量表达式。如果函数不能在编译时进行评估，则程序是不良形式，编译器会发出错误。

以下规则适用于立即函数：

+   析构函数、分配和释放函数不能是立即函数。

+   如果函数的任何声明包含`consteval`指定符，则该函数的所有声明也必须包含它。

+   `consteval`指定符不能与`constexpr`或`constinit`一起使用。

+   立即函数是一个内联的`constexpr`函数。因此，立即函数和函数模板必须满足适用于`constexpr`函数的要求。

这里是如何使用上一节中定义的`factorial()`函数和`point3d`类的示例：

```cpp
constexpr unsigned int f = factorial(6);
std::cout << f << '\n';
constexpr point3d p {0, 1, 2};
std::cout << p.get_x() << ' ' << p.get_y() << ' ' << p.get_z() << '\n'; 
```

然而，以下示例会产生编译器错误，因为即时函数`factorial()`和`point3d`的构造函数无法在编译时评估：

```cpp
unsigned int n;
std::cin >> n;
const unsigned int f2 = factorial(n); // error
double x = 0, y = 1, z = 2;
constexpr point3d p2 {x, y, z};       // error 
```

如果即时函数不是在常量表达式中，则无法获取其地址：

```cpp
using pfact = unsigned int(unsigned int);
pfact* pf = factorial;
constexpr unsigned int f3 = pf(42);   // error
consteval auto addr_factorial()
{
  return &factorial;
}
consteval unsigned int invoke_factorial(unsigned int const n)
{
  return addr_factorial()(n);
}
constexpr auto ptr = addr_factorial();
// ERROR: cannot take the pointer of an immediate function
constexpr unsigned int f2 = invoke_factorial(5);
// OK 
```

因为即时函数在运行时不可见，所以不会为它们生成符号，调试器也无法显示它们。

## 参见

+   *确保程序常量正确性*，探索常量正确性的好处以及如何实现它

+   *创建编译时常量表达式*，了解`constexpr`指定符以及如何定义可以在编译时评估的变量和函数

# 在常量评估上下文中优化代码

在前两个菜谱中，我们学习了关于*常量表达式函数*，它允许函数在所有输入在编译时都可用的情况下在编译时进行评估，以及*C++20 中的即时函数*，它们保证始终在编译时评估（否则将产生错误）。`constexpr`函数的一个重要方面是常量评估上下文；这些是在编译时评估所有表达式和函数的代码路径。常量评估上下文对于更有效地优化代码非常有用。另一方面，从`constexpr`函数中调用即时函数仅在 C++23 中可行。在本菜谱中，我们将学习如何利用常量评估上下文。

## 如何实现…

要确定函数上下文是否为常量评估，以便提供编译时实现，请使用以下方法：

+   在 C++20 中，`std::is_constant_evaluated()`库函数，在`<type_traits>`头文件中可用，使用常规的`if`语句：

    ```cpp
    constexpr double power(double base, int exponent)
    {
       if(std::is_constant_evaluated())
       {
          double result = 1.0;
          if (exponent == 0)
          {
              return result;
          }
          else if (exponent > 0) {
              for (int i = 0; i < exponent; i++) {
                  result *= base;
              }
          }
          else {
              exponent = -exponent;
              for (int i = 0; i < exponent; i++) {
                  result *= base;
              }
              result = 1.0 / result;
          }
          return result;
       }
       else
       {
           return std::pow(base, exponent);
       }
    }
    int main()
    {
       constexpr double a = power(10, 5); // compile-time eval
       std::cout << a << '\n';
       double b = power(10, 5);           // runtime eval
       std::cout << b << '\n';
    } 
    ```

+   在 C++23 中，`if consteval`语句，它是`if(std::is_constant_evaluated())`语句的简化（具有额外的优点）：

    ```cpp
    constexpr double power(double base, int exponent)
    {
       if consteval
       {
          double result = 1.0;
          if (exponent == 0)
          {
              return result;
          }
          else if (exponent > 0) {
              for (int i = 0; i < exponent; i++) {
                  result *= base;
              }
          }
          else {
              exponent = -exponent;
              for (int i = 0; i < exponent; i++) {
                  result *= base;
              }
              result = 1.0 / result;
          }
          return result;
       }
       else
       {
           return std::pow(base, exponent);
       }
    } 
    ```

## 它是如何工作的…

C++20 标准提供了一个名为`std::is_constant_evaluated()`的库函数（在`<type_traits>`头文件中），它可以检测其调用是否发生在`constexpr`函数中的常量评估上下文中。在这种情况下，它返回`true`；否则，返回`false`。

此函数使用常规的`if`语句，如前一小节中提供的示例，其中我们计算了数字的幂。从这个实现中可以得出的关键要点如下：

+   在常量评估上下文中，我们使用了一个可以在编译时由编译器执行的算法来优化代码。

+   在非常量评估上下文（即运行时）中，我们调用`std::pow()`函数来计算幂。

然而，这个函数和常量评估上下文有一些“陷阱”，你必须注意：

+   函数的参数在编译时已知，并不意味着上下文是常量评估的。在以下代码片段中，`constexpr` 函数 `power()` 的第一次调用是在常量评估上下文中，但第二次调用不是，尽管所有参数在编译时已知，并且函数被声明为 `constexpr`：

    ```cpp
    constexpr double a = power(10, 5); // [1] compile-time eval
    double b = power(10, 5);           // [2] runtime eval 
    ```

+   如果与 `constexpr` if 语句一起使用，`std::is_constant_evaluated()` 函数始终评估为 `true`（例如 GCC 和 Clang 编译器会为此细微的错误提供警告）：

    ```cpp
    constexpr double power(double base, int exponent)
    {
       if constexpr (std::is_constant_evaluated())
     {
       }
    } 
    ```

以下是一个报告错误的示例：

```cpp
prog.cc: In function 'constexpr double power(double, int)':
prog.cc:10:45: warning: 'std::is_constant_evaluated' always evaluates to true in 'if constexpr' [-Wtautological-compare]
   10 |     if constexpr (std::is_constant_evaluated())
      |                   ~~~~~~~~~~~~~~~~~~~~~~~~~~^~ 
```

C++23 标准提供了对 `std::is_constant_evaluated()` 函数的更好替代方案，即 `consteval if` 语句。这有几个优点：

+   不需要包含头文件

+   避免对使用正确形式的 `if` 语句产生混淆

+   允许在常量评估上下文中调用立即函数

在 C++23 中，幂函数的实现变为以下形式：

```cpp
constexpr double power(double base, int exponent)
{
   if consteval
   {
      /* ... */
   }
   else
   {
       return std::pow(base, exponent);
   }
} 
```

`consteval` `if` 语句始终需要花括号。否定形式也是可能的，无论是使用 `!` 还是 `not` 关键字。在以下代码片段中，每一对语句都是等价的：

```cpp
if !consteval {/*statement*/}          // [1] equivalent to [2]
if consteval {} else {/*statement*/}   // [2]
if not consteval {/*statement1*/}      // [3] equivalent to [4]
else {/*statement2*/}              
if consteval {/*statement2*/}          // [4]
else {/*statement1*/} 
```

`consteval` if 语句对于允许在 `constexpr` 函数中从常量评估上下文中立即调用函数也很重要。以下是一个 C++20 的示例：

```cpp
consteval int plus_one(int const i) 
{ 
   return i + 1; 
}
consteval int plus_two(int i)
{
   return plus_one(i) + 1;
}
constexpr int plus_two_alt(int const i)
{
   if (std::is_constant_evaluated())
   {
      return plus_one(i) + 1;
   } 
   else
   {
      return i + 2;
   }
} 
```

在这里，函数 `plus_one()` 是一个立即函数，可以从 `plus_two()` 函数（也是一个立即函数）中调用。然而，从 `plus_two_alt()` 函数中调用它是不可能的，因为它不是一个常量表达式，尽管这是一个 `constexpr` 函数，并且调用 `plus_one()` 函数的上下文是常量评估的。

这个问题通过 C++23 的 `consteval if` 语句得到解决。这使得从常量评估上下文中调用立即函数成为可能，如下面的示例所示：

```cpp
constexpr int plus_two_alt(int const i)
{
   if consteval
   {
      return plus_one(i) + 1;
   } 
   else
   {
      return i + 2;
   }
} 
```

随着 `consteval if` 语句的可用性，`std::is_constant_evaluated()` 函数变得过时。实际上，它可以使用 `consteval if` 语句如下实现：

```cpp
constexpr bool is_constant_evaluated() noexcept
{
   if consteval {
      return true;
   } else {
      return false;
   }
} 
```

当使用 C++23 编译器时，你应该始终优先选择 `consteval if` 语句，而不是过时的 `std::is_constant_evaluated()` 函数。

## 参见

+   *创建编译时常量表达式*，了解 `constexpr` 指示符以及如何定义可以在编译时评估的变量和函数

+   *创建立即函数*，了解 C++20 的 `consteval` 指示符，它用于定义保证在编译时评估的函数

# 在常量表达式中使用虚函数调用

作为一种多范式编程语言，C++ 包括对面向对象编程的支持。多态性是面向对象编程的核心原则之一，在 C++ 中有两种形式：编译时多态性，通过函数和运算符重载实现，以及运行时多态性，通过虚函数实现。虚函数允许派生类覆盖基类中的函数实现。然而，在 C++20 中，虚函数被允许在常量表达式中使用，这意味着它们可以在编译时调用。在本食谱中，你将了解这是如何工作的。

## 准备工作

在本食谱中，我们将使用以下结构来表示文档的维度以及随后的示例中的信封维度：

```cpp
struct dimension
{
   double width;
   double height;
}; 
```

## 如何实现...

你可以通过以下方式将运行时多态性移动到编译时：

+   将你想要移动到编译时调用的虚函数声明为 `constexpr`。

+   将层次结构的基类的析构函数声明为 `constexpr`。

+   将重写的虚函数声明为 `constexpr`。

+   在常量表达式中调用 `constexpr` 虚函数。

以下是一个示例片段：

```cpp
struct document_type
{
   constexpr virtual ~document_type() {};
   constexpr virtual dimension size() const = 0;
};
struct document_a5 : document_type
{
   constexpr dimension size() const override { return { 148.5, 210 }; }
};
struct envelope_type
{
   constexpr virtual ~envelope_type() {}
   constexpr virtual dimension size() const = 0;
   constexpr virtual dimension max_enclosure_size() const = 0;
};
struct envelop_commercial_8 : envelope_type
{
   constexpr dimension size() const override { return { 219, 92 }; }
   constexpr dimension max_enclosure_size() const override 
 { return { 213, 86 }; }
};
constexpr bool document_fits_envelope(document_type const& d, 
                                      envelope_type const& e)
{
   return e.max_enclosure_size().width >= d.size().width;
}
int main()
{
   constexpr envelop_commercial_8 e1;
   constexpr document_a5          d1;
   static_assert(document_fits_envelope(d1, e1));
} 
```

## 它是如何工作的...

在 C++20 之前，虚函数不能是 `constexpr`。然而，用于常量表达式的对象的动态类型必须在编译时已知。因此，将虚函数设置为 `constexpr` 的限制在 C++20 中已被取消。

有 `constexpr` 虚函数的优势在于可以将一些计算从运行时移动到编译时。尽管这不会影响实践中许多用例，但在上一节中已经给出一个示例。让我们进一步阐述，以便更好地理解。

我们有一系列各种文档纸张大小。例如，包括 *A3*、*A4*、*A5*、*legal*、*letter* 和 *half-letter*。它们有不同的尺寸。例如，A5 是 148.5 毫米 x 210 毫米，而 letter 是 215.9 毫米 x 279.4 毫米。

另一方面，我们有不同类型和大小的信封。例如，我们有一个 92 毫米 x 219 毫米的信封，最大封装尺寸为 86 毫米 x 213 毫米。我们想要编写一个函数，以确定某种类型的折叠纸张是否可以放入信封中。由于尺寸是标准的，它们在编译时是已知的。这意味着我们可以在编译时而不是运行时执行此检查。

为了这个目的，在 *如何实现...* 部分中，我们已经看到：

+   一个文档的层次结构，基类称为 `document_type`。它有两个成员：一个虚析构函数和一个名为 `size()` 的虚函数，该函数返回纸张的大小。这两个函数也都是 `constexpr`。

+   一个信封的层次结构，基类称为`envelope_type`。它有三个成员：一个虚析构函数，一个名为`size()`的虚函数，它返回信封的大小，以及一个名为`max_enclosure_size()`的虚函数，它返回可以放入信封中的（折叠）纸张的最大尺寸。所有这些都是`constexpr`。

+   一个名为`document_fits_envelope()`的免费函数通过比较两个宽度的尺寸来确定给定的文档类型是否适合特定的信封类型。这也是一个`constexpr`函数。

因为所有提到的函数都是`constexpr`，所以如果被调用的对象也是`constexpr`，则可以在常量表达式中调用`document_fits_envelope()`函数，例如在`static_assert`中。在本书附带的代码文件中，你可以找到一个关于各种纸张和信封尺寸的详细示例。

你应该记住：

+   你甚至可以将一个重写的虚拟函数声明为`constexpr`，即使它在基类中被重写的函数没有被定义为`constexpr`。

+   反过来也是可能的，派生类中重写的虚拟函数可以不是`constexpr`，尽管在基类中该函数被定义为`constexpr`。

+   如果存在多级层次结构，并且一个虚拟函数定义了一些`constexpr`重写和一些非`constexpr`重写，那么用于确定虚拟函数是否为`constexpr`的最终重写器是针对被调用的对象。

## 参见

+   *第一章*，*使用 override 和 final 对虚拟方法和类进行操作*，学习如何在虚拟方法和类上使用 override 和 final 说明符

# 正确执行类型转换

通常情况下，数据必须从一种类型转换为另一种类型。一些转换在编译时是必要的（例如`double`到`int`）；其他转换在运行时是必要的（例如向上转换和向下转换层次结构中的类指针）。该语言支持与 C 转换风格的兼容性，无论是`(type)expression`还是`type(expression)`形式。然而，这种类型的转换破坏了 C++的类型安全性。

因此，该语言也提供了几种转换：`static_cast`、`dynamic_cast`、`const_cast`和`reinterpret_cast`。它们用于更好地指示意图并编写更安全的代码。在本菜谱中，我们将探讨如何使用这些转换。

## 如何做到这一点...

使用以下转换来执行类型转换：

+   使用`static_cast`来执行非多态类型的类型转换，包括将整数转换为枚举、从浮点数转换为整数值，或从指针类型转换为另一个指针类型，例如从基类到派生类（向下转换）或从派生类到基类（向上转换），但没有任何运行时检查：

    ```cpp
    enum options {one = 1, two, three};
    int value = 1;
    options op = static_cast<options>(value);
    int x = 42, y = 13;
    double d = static_cast<double>(x) / y;
    int n = static_cast<int>(d); 
    ```

+   使用 `dynamic_cast` 来执行从基类到派生类或相反的指针或引用的多态类型的类型转换。这些检查在运行时执行，可能需要启用 **运行时类型信息**（**RTTI**）：

    ```cpp
    struct base
    {
      virtual void run() {}
      virtual ~base() {}
    };
    struct derived : public base
    {
    };
    derived d;
    base b;
    base* pb = dynamic_cast<base*>(&d);         // OK
    derived* pd = dynamic_cast<derived*>(&b);   // fail
    try
    {
      base& rb = dynamic_cast<base&>(d);       // OK
      derived& rd = dynamic_cast<derived&>(b); // fail
    }
    catch (std::bad_cast const & e)
    {
      std::cout << e.what() << '\n';
    } 
    ```

+   使用 `const_cast` 来执行具有不同 `const` 和 `volatile` 说明符的类型之间的转换，例如从未声明为 `const` 的对象中移除 `const`：

    ```cpp
    void old_api(char* str, unsigned int size)
    {
      // do something without changing the string
    }
    std::string str{"sample"};
    old_api(const_cast<char*>(str.c_str()),
            static_cast<unsigned int>(str.size())); 
    ```

+   使用 `reinterpret_cast` 来执行位重新解释，例如在整数和指针类型之间、从指针类型到整数或从指针类型到任何其他指针类型的转换，而不涉及任何运行时检查：

    ```cpp
    class widget
    {
    public:
      typedef size_t data_type;
      void set_data(data_type d) { data = d; }
      data_type get_data() const { return data; }
    private:
      data_type data;
    };
    widget w;
    user_data* ud = new user_data();
    // write
    w.set_data(reinterpret_cast<widget::data_type>(ud));
    // read
    user_data* ud2 = reinterpret_cast<user_data*>(w.get_data()); 
    ```

## 它是如何工作的...

显式类型转换，有时被称为 *C 风格转换* 或 *静态转换*，是 C++ 与 C 语言兼容性的遗产，并允许你执行各种转换，包括以下内容：

+   在算术类型之间

+   在指针类型之间

+   在整型和指针类型之间

+   在具有不同 `const` 或 `volatile` 修饰符的已修饰和未修饰类型之间

这种类型的转换在处理多态类型或在模板中使用时效果不佳。正因为如此，C++ 提供了我们在前面的示例中看到的四种转换。使用这些转换可以带来几个重要的好处：

+   它们更好地表达了用户的意图，无论是对于编译器还是阅读代码的其他人。

+   它们使得在不同类型之间的转换更加安全（除了 `reinterpret_cast`）。

+   它们可以在源代码中轻松搜索。

`static_cast` 并不是显式类型转换或静态转换的直接等价物，尽管名称可能暗示这一点。这种转换在编译时执行，可以用来执行隐式转换、隐式转换的反向转换以及从类层次结构中的指针到类型的转换。它不能用来触发不相关指针类型之间的转换。因此，在以下示例中，使用 `static_cast` 从 `int*` 转换到 `double*` 会产生编译器错误：

```cpp
int* pi = new int{ 42 };
double* pd = static_cast<double*>(pi);   // compiler error 
```

然而，从 `base*` 转换到 `derived*`（其中 `base` 和 `derived` 是在 *如何做...* 部分中显示的类）不会产生编译器错误，但在尝试使用新获得的指针时会产生运行时错误：

```cpp
base b;
derived* pd = static_cast<derived*>(&b); // compilers OK, runtime error
base* pb1 = static_cast<base*>(pd);      // OK 
```

另一方面，`static_cast` 不能用来移除 `const` 和 `volatile` 修饰符。以下代码片段说明了这一点：

```cpp
int const c = 42;
int* pc = static_cast<int*>(&c);         // compiler error 
```

使用 `dynamic_cast` 可以在继承层次结构中安全地进行向上、向下或侧向的类型转换。这种转换在运行时执行，并要求启用 RTTI。正因为如此，它会产生运行时开销。动态转换只能用于指针和引用。

当使用 `dynamic_cast` 将表达式转换为指针类型且操作失败时，结果是一个空指针。当它用于将表达式转换为引用类型且操作失败时，会抛出 `std::bad_cast` 异常。因此，总是将 `dynamic_cast` 转换到引用类型的操作放在 `try...catch` 块中。

RTTI 是一种在运行时暴露对象数据类型信息的机制。这仅适用于多态类型（至少有一个虚方法，包括虚析构函数，所有基类都应该有）。RTTI 通常是一个可选的编译器功能（或者可能根本不支持），这意味着使用此功能可能需要使用编译器开关。

虽然动态转换是在运行时执行的，但如果尝试在非多态类型之间进行转换，将会得到编译器错误：

```cpp
struct struct1 {};
struct struct2 {};
struct1 s1;
struct2* ps2 = dynamic_cast<struct2*>(&s1); // compiler error 
```

`reinterpret_cast` 更像是一个编译器指令。它不会转换为任何 CPU 指令；它只是指示编译器将表达式的二进制表示解释为另一种指定的类型。这是一种类型不安全的转换，应该谨慎使用。它可以用于在整数类型和指针、指针类型和函数指针类型之间转换表达式。因为没有任何检查，`reinterpret_cast` 可以成功用于在无关类型之间转换表达式，例如从 `int*` 转换到 `double*`，这将产生未定义的行为：

```cpp
int* pi = new int{ 42 };
double* pd = reinterpret_cast<double*>(pi); 
```

`reinterpret_cast` 的典型用途是在使用操作系统或供应商特定 API 的代码中在类型之间转换表达式。许多 API 以指针或整数类型的形式存储用户数据。因此，如果您需要将用户定义类型的地址传递给此类 API，则需要将无关指针类型或指针类型值转换为整数类型值。在上一节中提供了一个类似的例子，其中 `widget` 是一个类，它在数据成员中存储用户定义的数据，并提供了访问它的方法：`set_data()` 和 `get_data()`。如果您需要在 `widget` 中存储对象的指针，那么请使用 `reinterpret_cast`，如下面的例子所示。

`const_cast` 在某种程度上与 `reinterpret_cast` 相似，因为它是一个编译器指令，并不转换为 CPU 指令。它用于去除 `const` 或 `volatile` 修饰符，这是其他三种转换所不能做的操作。

`const_cast` 应仅用于在对象未声明为 `const` 或 `volatile` 时去除 `const` 或 `volatile` 修饰符。否则，将会引入未定义的行为，如下面的例子所示：

```cpp
int const a = 42;
int const * p = &a;
int* q = const_cast<int*>(p);
*q = 0; // undefined behavior 
```

在这个例子中，变量 `p` 指向一个对象（变量 `a`），该对象被声明为常量。通过移除 `const` 修饰符，尝试修改指向的对象会引入未定义的行为。

## 还有更多...

当使用形式为`(type)expression`的显式类型转换时，请注意它将选择以下列表中满足特定转换要求的第一个选择：

1.  `const_cast<type>(expression)`

1.  `static_cast<type>(expression)`

1.  `static_cast<type>(expression) + const_cast<type>(expression)`

1.  `reinterpret_cast<type>(expression)`

1.  `reinterpret_cast<type>(expression) + const_cast<type>(expression)`

此外，与特定的 C++转换不同，静态转换可以用于在未完整类类型之间进行转换。如果`type`和`expression`都是指向未完整类型的指针，则未指定是选择`static_cast`还是`reinterpret_cast`。

## 参见

+   *确保程序常量正确性*，以探索常量正确性的好处以及如何实现它

# 实现移动语义

移动语义是推动现代 C++性能提升的关键特性。它们允许移动资源，而不是复制，或者更一般地说，复制代价高昂的对象。然而，它要求类实现移动构造函数和移动赋值运算符。在某些情况下，编译器提供了这些，但在实践中，通常您必须显式地编写它们。在本配方中，我们将看到如何实现移动构造函数和移动赋值运算符。

## 准备工作

预期您具备关于右值引用和特殊类函数（构造函数、赋值运算符和析构函数）的基本知识。我们将演示如何使用以下`Buffer`类来实现移动构造函数和赋值运算符：

```cpp
class Buffer
{
  unsigned char* ptr;
  size_t length;
public:
  Buffer(): ptr(nullptr), length(0)
  {}
  explicit Buffer(size_t const size):
    ptr(new unsigned char[size] {0}), length(size)
  {}
  ~Buffer()
  {
    delete[] ptr;
  }
  Buffer(Buffer const& other):
    ptr(new unsigned char[other.length]),
    length(other.length)
  {
    std::copy(other.ptr, other.ptr + other.length, ptr);
  }
  Buffer& operator=(Buffer const& other)
  {
    if (this != &other)
    {
      delete[] ptr;
      ptr = new unsigned char[other.length];
      length = other.length;
      std::copy(other.ptr, other.ptr + other.length, ptr);
    }
    return *this;
  }
  size_t size() const { return length;}
  unsigned char* data() const { return ptr; }
}; 
```

让我们继续到下一节，在那里您将学习如何修改这个类以利用移动语义。

## 如何做到这一点...

要为类实现移动构造函数，请执行以下操作：

1.  编写一个接受类类型右值引用的构造函数：

    ```cpp
    Buffer(Buffer&& other)
    {
    } 
    ```

1.  将所有数据成员从右值引用赋值到当前对象。这可以在构造函数体中完成，如下所示，或者在初始化列表中完成，这是首选方式：

    ```cpp
    ptr = other.ptr;
    length = other.length; 
    ```

1.  可选地，将数据成员从右值引用赋值为默认值（以确保被移动的对象处于可析构状态）：

    ```cpp
    other.ptr = nullptr;
    other.length = 0; 
    ```

将`Buffer`类的移动构造函数整合如下：

```cpp
Buffer(Buffer&& other)
{
  ptr = other.ptr;
  length = other.length;
  other.ptr = nullptr;
  other.length = 0;
} 
```

要为类实现移动赋值运算符，请执行以下操作：

1.  编写一个接受类类型右值引用并返回其引用的赋值运算符：

    ```cpp
    Buffer& operator=(Buffer&& other)
    {
    } 
    ```

1.  检查右值引用是否指向与`this`相同的对象，如果它们不同，则执行步骤 3 到步骤 5：

    ```cpp
    if (this != &other)
    {
    } 
    ```

1.  处理当前对象的所有资源（如内存、句柄等）：

    ```cpp
    delete[] ptr; 
    ```

1.  将所有数据成员从右值引用赋值到当前对象：

    ```cpp
    ptr = other.ptr;
    length = other.length; 
    ```

1.  将右值引用的数据成员赋值为默认值：

    ```cpp
    other.ptr = nullptr;
    other.length = 0; 
    ```

1.  不论是否执行了步骤 3 到步骤 5，都返回当前对象的引用：

    ```cpp
    return *this; 
    ```

将所有内容放在一起，`Buffer`类的移动赋值运算符看起来像这样：

```cpp
Buffer& operator=(Buffer&& other)
{
  if (this != &other)
  {
    delete[] ptr;
    ptr = other.ptr;
    length = other.length;
    other.ptr = nullptr;
    other.length = 0;
  }
  return *this;
} 
```

## 它是如何工作的...

移动构造函数和移动赋值运算符由编译器默认提供，除非已经存在用户定义的复制构造函数、移动构造函数、复制赋值运算符、移动赋值运算符或析构函数。当由编译器提供时，它们以成员方式执行移动。移动构造函数递归地调用类数据成员的移动构造函数；同样，移动赋值运算符递归地调用类数据成员的移动赋值运算符。

在这种情况下，移动代表了对太大而无法复制（如字符串或容器）或不应被复制（如`unique_ptr`智能指针）的对象的性能优势。并非所有类都应该实现复制和移动语义。一些类应该只可移动，而另一些类则应该可复制和可移动。另一方面，一个类可复制但不移动并没有太多意义，尽管技术上可以实现。

并非所有类型都从移动语义中受益。对于内置类型（如`bool`、`int`或`double`）、数组或 PODs，移动实际上是一个复制操作。另一方面，移动语义在 rvalue 的上下文中提供了性能优势，即临时对象。rvalue 是一个没有名字的对象；它在表达式的评估期间临时存在，并在下一个分号处被销毁：

```cpp
T a;
T b = a;
T c = a + b; 
```

在前面的例子中，`a`、`b`和`c`是 lvalue；它们是有名字的对象，可以在其生命周期的任何时刻通过这个名字来引用该对象。另一方面，当你评估表达式`a+b`时，编译器创建了一个临时对象（在这种情况下，被分配给`c`），然后在遇到分号时被销毁。这些临时对象被称为 rvalue，因为它们通常出现在赋值表达式的右侧。在 C++11 中，我们可以通过`&&`这样的 rvalue 引用来引用这些对象。

移动语义在 rvalue 的上下文中非常重要。这是因为它们允许你在临时对象被销毁后获取其资源所有权，而客户端在移动操作完成后无法再使用它。另一方面，lvalue 不能被移动；它们只能被复制。这是因为它们可以在移动操作之后被访问，客户端期望对象处于相同的状态。例如，在前面的例子中，表达式`b = a`将`a`赋值给`b`。

在此操作完成后，lvalue 类型的对象`a`仍然可以被客户端使用，并且应该处于与之前相同的状态。另一方面，`a+b`的结果是临时的，其数据可以安全地移动到`c`。

移动构造函数与复制构造函数不同，因为它接受对类类型`T`的右值引用`T(T&&)`，而复制构造函数接受左值引用`T(T const&)`。同样，移动赋值也接受右值引用，即`T& operator=(T&&)`，而复制赋值运算符接受左值引用，即`T& operator=(T const &)`。即使两者都返回对`T&`类的引用，这也是正确的。编译器根据参数的类型（右值或左值）选择合适的构造函数或赋值运算符。

当存在移动构造函数/赋值运算符时，右值会被自动移动。左值也可以被移动，但这需要显式地将其转换为右值引用。这可以通过使用`std::move()`函数来完成，它基本上执行了一个`static_cast<T&&>`操作：

```cpp
std::vector<Buffer> c;
c.push_back(Buffer(100));  // move
Buffer b(200);
c.push_back(b);            // copy
c.push_back(std::move(b)); // move 
```

对象移动后，它必须保持在一个有效状态。然而，没有关于这个状态应该是什么的要求。为了保持一致性，你应该将所有成员字段设置为它们的默认值（数值类型为`0`，指针为`nullptr`，布尔值为`false`等）。

以下示例展示了`Buffer`对象可以以不同的方式被构造和赋值：

```cpp
Buffer b1;                // default constructor
Buffer b2(100);           // explicit constructor
Buffer b3(b2);            // copy constructor
b1 = b3;                  // assignment operator
Buffer b4(std::move(b1)); // move constructor
b3 = std::move(b4);       // move assignment 
```

每一行注释中提到的构造函数或赋值运算符都涉及到了对象`b1`、`b2`、`b3`和`b4`的创建或赋值。

## 还有更多...

如`Buffer`示例所示，实现移动构造函数和移动赋值运算符都涉及到编写类似的代码（移动构造函数的整个代码也存在于移动赋值运算符中）。实际上，可以通过在移动构造函数中调用移动赋值运算符（或者，作为替代方案，将赋值代码分解成一个私有函数，该函数由移动构造函数和移动赋值运算符共同调用）来避免这种情况：

```cpp
Buffer(Buffer&& other) : ptr(nullptr), length(0)
{
  *this = std::move(other);
} 
```

在这个例子中有两个需要注意的点：

+   在构造函数的初始化列表中进行成员初始化是必要的，因为这些成员可能会在以后的移动赋值运算符中使用（例如，本例中的`ptr`成员）。

+   将`other`显式地转换为右值引用。如果没有这个显式转换，将会调用复制赋值运算符。这是因为即使将右值作为参数传递给这个构造函数，当它被赋予一个名称时，它绑定到一个左值上。因此，`other`实际上是一个左值，必须转换为右值引用才能调用移动赋值运算符。

## 参见

+   *第三章*，*默认和删除函数*，了解在特殊成员函数上使用`default`指定符以及如何使用`delete`指定符定义已删除的函数

# 使用`unique_ptr`来唯一拥有内存资源

手动处理堆内存分配和释放（使用 `new` 和 `delete`）是 C++ 中最具争议的特性之一。所有分配都必须与正确的范围内的相应删除操作正确配对。如果内存分配在函数中完成，并且需要在函数返回之前释放，例如，那么这必须在所有返回路径上发生，包括函数由于异常而返回的不正常情况。C++11 特性，如右值和移动语义，使得更好的智能指针（因为一些，如 `auto_ptr`，在 C++11 之前就已存在）的开发成为可能；这些指针可以管理内存资源，并在智能指针被销毁时自动释放。在这个菜谱中，我们将查看 `std::unique_ptr`，这是一个拥有并管理在堆上分配的另一个对象或对象数组的智能指针，并在智能指针超出作用域时执行销毁操作。

## 准备工作

在以下示例中，我们将使用以下类：

```cpp
class foo
{
  int a;
  double b;
  std::string c;
public:
  foo(int const a = 0, double const b = 0, 
 std::string const & c = "") :a(a), b(b), c(c)
  {}
  void print() const
 {
    std::cout << '(' << a << ',' << b << ',' << std::quoted(c) << ')'
              << '\n';
  }
}; 
```

对于这个菜谱，你需要熟悉移动语义和 `std::move()` 转换函数。`unique_ptr` 类在 `<memory>` 头文件中的 `std` 命名空间中可用。

## 如何做到这一点...

以下是在使用 `std::unique_ptr` 时需要了解的一些典型操作列表：

+   使用可用的重载构造函数创建一个通过指针管理对象或对象数组的 `std::unique_ptr`。默认构造函数创建一个不管理任何对象的指针：

    ```cpp
    std::unique_ptr<int>   pnull;
    std::unique_ptr<int>   pi(new int(42));
    std::unique_ptr<int[]> pa(new int[3]{ 1,2,3 });
    std::unique_ptr<foo>   pf(new foo(42, 42.0, "42")); 
    ```

+   或者，使用 C++14 中可用的 `std::make_unique()` 函数模板创建 `std::unique_ptr` 对象：

    ```cpp
    std::unique_ptr<int>   pi = std::make_unique<int>(42);
    std::unique_ptr<int[]> pa = std::make_unique<int[]>(3);
    std::unique_ptr<foo>   pf = std::make_unique<foo>(42, 42.0, "42"); 
    ```

+   使用 C++20 中可用的 `std::make_unique_for_overwrite()` 函数模板，创建指向默认初始化的对象或对象数组的 `std::unique_ptr`。这些对象应稍后用确定的值覆盖：

    ```cpp
    std::unique_ptr<int>   pi = std::make_unique_for_overwrite<int>();
    std::unique_ptr<foo[]> pa = std::make_unique_for_overwrite<foo[]>(); 
    ```

+   如果默认的 `delete` 操作符不适用于销毁托管对象或数组，请使用重载构造函数，该构造函数接受自定义删除器：

    ```cpp
    struct foo_deleter
    {
      void operator()(foo* pf) const
     {
        std::cout << "deleting foo..." << '\n';
        delete pf;
      }
    };
    std::unique_ptr<foo, foo_deleter> pf(
     new foo(42, 42.0, "42"),
        foo_deleter()); 
    ```

+   使用 `std::move()` 将对象的所有权从一个 `std::unique_ptr` 转移到另一个：

    ```cpp
    auto pi = std::make_unique<int>(42);
    auto qi = std::move(pi);
    assert(pi.get() == nullptr);
    assert(qi.get() != nullptr); 
    ```

+   要访问托管对象的原始指针，如果你想保留对象的所有权，请使用 `get()`；如果你想释放所有权，请使用 `release()`：

    ```cpp
    void func(int* ptr)
    {
      if (ptr != nullptr)
        std::cout << *ptr << '\n';
      else
        std::cout << "null" << '\n';
    }
    std::unique_ptr<int> pi;
    func(pi.get()); // prints null
    pi = std::make_unique<int>(42);
    func(pi.get()); // prints 42 
    ```

+   使用 `operator*` 和 `operator->` 解引用托管对象的指针：

    ```cpp
    auto pi = std::make_unique<int>(42);
    *pi = 21;
    auto pf1 = std::make_unique<foo>();
    pf1->print(); // prints (0,0,"")
    auto pf2 = std::make_unique<foo>(42, 42.0, "42");
    pf2->print(); // prints (42,42,"42") 
    ```

+   如果 `std::unique_ptr` 管理一个对象数组，可以使用 `operator[]` 访问数组的单个元素：

    ```cpp
    std::unique_ptr<int[]> pa = std::make_unique<int[]>(3);
    for (int i = 0; i < 3; ++i)
      pa[i] = i + 1; 
    ```

+   要检查 `std::unique_ptr` 是否可以管理一个对象，请使用显式操作符 `bool` 或检查 `get() != nullptr`（这是操作符 `bool` 所做的）：

    ```cpp
    std::unique_ptr<int> pi(new int(42));
    if (pi) std::cout << "not null" << '\n'; 
    ```

+   `std::unique_ptr`对象可以存储在容器中。由`make_unique()`返回的对象可以直接存储。如果想要将管理对象的所有权放弃给容器中的`std::unique_ptr`对象，可以将一个左值对象通过`std::move()`静态转换为右值对象：

    ```cpp
    std::vector<std::unique_ptr<foo>> data;
    for (int i = 0; i < 5; i++)
      data.push_back(
    std::make_unique<foo>(i, i, std::to_string(i)));
    auto pf = std::make_unique<foo>(42, 42.0, "42");
    data.push_back(std::move(pf)); 
    ```

## 它是如何工作的...

`std::unique_ptr`是一个智能指针，它通过原始指针管理在堆上分配的对象或数组。当智能指针超出作用域、被赋予新的指针使用`operator=`或使用`release()`方法放弃所有权时，它会执行适当的销毁操作。默认情况下，使用`delete`运算符来销毁管理对象。然而，用户在构造智能指针时可以提供自定义的销毁器。这个销毁器必须是一个函数对象，要么是一个函数对象的左值引用，要么是一个函数，并且这个可调用对象必须接受一个类型为`unique_ptr<T, Deleter>::pointer`的单个参数。

C++14 添加了`std::make_unique()`实用函数模板来创建`std::unique_ptr`。它在某些特定情况下避免了内存泄漏，但也有一些限制：

+   它只能用来分配数组；你不能用它来初始化它们，这是`std::unique_ptr`构造函数所能做到的。

    以下两段示例代码是等价的：

    ```cpp
    // allocate and initialize an array
    std::unique_ptr<int[]> pa(new int[3]{ 1,2,3 });
    // allocate and then initialize an array
    std::unique_ptr<int[]> pa = std::make_unique<int[]>(3);
    for (int i = 0; i < 3; ++i)
      pa[i] = i + 1; 
    ```

+   它不能用来创建具有用户定义销毁器的`std::unique_ptr`对象。

正如我们刚才提到的，`make_unique()`的巨大优势是它帮助我们避免在某些抛出异常的上下文中发生内存泄漏。如果分配失败或它创建的对象的构造函数抛出任何异常，`make_unique()`本身可能会抛出`std::bad_alloc`。让我们考虑以下示例：

```cpp
void some_function(std::unique_ptr<foo> p)
{ /* do something */ }
some_function(std::unique_ptr<foo>(new foo()));
some_function(std::make_unique<foo>()); 
```

无论`foo`的分配和构造过程中发生什么，都不会有内存泄漏，无论你使用`make_unique()`还是`std::unique_ptr`的构造函数。然而，代码的略微不同版本会导致这种情况发生变化：

```cpp
void some_other_function(std::unique_ptr<foo> p, int const v)
{
}
int function_that_throws()
{
  throw std::runtime_error("not implemented");
}
// possible memory leak
some_other_function(std::unique_ptr<foo>(new foo),
                    function_that_throws());
// no possible memory leak
some_other_function(std::make_unique<foo>(),
                    function_that_throws()); 
```

在这个例子中，`some_other_function()` 有一个额外的参数：一个整数值。传递给这个函数的整数参数是另一个函数的返回值。如果这个函数调用抛出异常，使用 `std::unique_ptr` 构造函数创建智能指针可能会导致内存泄漏。这是因为，在调用 `some_other_function()` 时，编译器可能会首先调用 `foo`，然后是 `function_that_throws()`，最后是 `std::unique_ptr` 的构造函数。如果 `function_that_throws()` 抛出错误，那么分配的 `foo` 将会泄漏。如果调用顺序是 `function_that_throws()` 然后是 `new foo()` 和 `unique_ptr` 的构造函数，则不会发生内存泄漏；这是因为栈在 `foo` 对象分配之前就开始回溯。然而，通过使用 `make_unique()` 函数，可以避免这种情况。这是因为，唯一调用的函数是 `make_unique()` 和 `function_that_throws()`。如果首先调用 `function_that_throws()`，则 `foo` 对象根本不会分配。如果首先调用 `make_unique()`，则 `foo` 对象将被构造，并且其所有权将传递给 `std::unique_ptr`。如果稍后调用 `function_that_throws()` 抛出异常，那么当栈回溯时，`std::unique_ptr` 将被销毁，并且 `foo` 对象将从智能指针的析构函数中销毁。C++17 通过要求在开始下一个参数之前必须完全评估任何参数来解决这个问题。

在 C++20 中，添加了一个名为 `std::make_unique_for_overwrite()` 的新函数。这与 `make_unique()` 类似，但它的默认值初始化对象或对象数组。此函数可用于泛型代码，其中不知道类型模板参数是否是平凡的复制的。此函数表达了创建指向可能未初始化的对象的指针的意图，以便稍后可以覆盖它。

常量 `std::unique_ptr` 对象不能将管理对象或数组的所有权转移到另一个 `std::unique_ptr` 对象。另一方面，可以通过 `get()` 或 `release()` 获取管理对象的原始指针。第一种方法仅返回底层指针，但后者也释放了管理对象的所有权，因此得名。在调用 `release()` 之后，`std::unique_ptr` 对象将变为空，调用 `get()` 将返回 `nullptr`。

管理 `Derived` 类对象的 `std::unique_ptr` 可以隐式转换为管理 `Base` 类对象的 `std::unique_ptr`，如果 `Derived` 从 `Base` 继承。这种隐式转换只有在 `Base` 有虚拟析构函数（所有基类都应该有）的情况下才是安全的；否则，将执行未定义的行为：

```cpp
struct Base
{
  virtual ~Base()
  {
    std::cout << "~Base()" << '\n';
  }
};
struct Derived : public Base
{
  virtual ~Derived()
  {
    std::cout << "~Derived()" << '\n';
  }
};
std::unique_ptr<Derived> pd = std::make_unique<Derived>();
std::unique_ptr<Base> pb = std::move(pd); 
```

运行此代码片段的输出如下：

```cpp
~Derived()
~Base() 
```

`std::unique_ptr` 可以存储在容器中，例如 `std::vector`。因为任何时刻只有一个 `std::unique_ptr` 对象可以拥有被管理的对象，所以智能指针不能被复制到容器中；它必须被移动。这可以通过 `std::move()` 实现，它执行了一个 `static_cast` 到右值引用类型。这允许将管理对象的所有权转移到容器中创建的 `std::unique_ptr` 对象。

## 参见

+   *使用 `shared_ptr` 共享内存资源*，了解 `std::shared_ptr` 类，它表示一个智能指针，它共享堆上分配的对象或对象数组的所有权

# 使用 `shared_ptr` 共享内存资源

当对象或数组需要共享时，无法使用 `std::unique_ptr` 来管理动态分配的对象或数组。这是因为 `std::unique_ptr` 保留其唯一所有权。C++ 标准提供了一个名为 `std::shared_ptr` 的另一个智能指针；它在许多方面与 `std::unique_ptr` 类似，但不同之处在于它可以与其他 `std::shared_ptr` 对象共享对象或数组的所有权。在本配方中，我们将了解 `std::shared_ptr` 的工作原理以及它与 `std::unique_ptr` 的区别。我们还将查看 `std::weak_ptr`，它是一个非资源拥有智能指针，它持有由 `std::shared_ptr` 管理的对象的引用。

## 准备工作

确保你已经阅读了之前的配方，*使用 `unique_ptr` 唯一拥有内存资源*，以熟悉 `unique_ptr` 和 `make_unique()` 的工作方式。我们将使用本配方中定义的 `foo`、`foo_deleter`、`Base` 和 `Derived` 类，并对其进行多次引用。

`shared_ptr` 和 `weak_ptr` 类以及 `make_shared()` 函数模板都包含在 `<memory>` 头文件中的 `std` 命名空间中。

为了简单和可读性，我们不会在本配方中使用完全限定的名称 `std::unique_ptr`、`std::shared_ptr` 和 `std::weak_ptr`，而是使用 `unique_ptr`、`shared_ptr` 和 `weak_ptr`。

## 如何做...

以下是在使用 `shared_ptr` 和 `weak_ptr` 时需要了解的典型操作的列表：

+   使用可用的重载构造函数之一来创建一个通过指针管理对象的 `shared_ptr`。默认构造函数创建一个空的 `shared_ptr`，它不管理任何对象：

    ```cpp
    std::shared_ptr<int> pnull1;
    std::shared_ptr<int> pnull2(nullptr);
    std::shared_ptr<int> pi1(new int(42));
    std::shared_ptr<int> pi2 = pi1;
    std::shared_ptr<foo> pf1(new foo());
    std::shared_ptr<foo> pf2(new foo(42, 42.0, "42")); 
    ```

+   或者，使用自 C++11 起可用的 `std::make_shared()` 函数模板来创建 `shared_ptr` 对象：

    ```cpp
    std::shared_ptr<int> pi  = std::make_shared<int>(42);
    std::shared_ptr<foo> pf1 = std::make_shared<foo>();
    std::shared_ptr<foo> pf2 = std::make_shared<foo>(42, 42.0, "42"); 
    ```

+   使用 C++20 中可用的 `std::make_shared_for_overwrite()` 函数模板来创建指向默认初始化的对象或对象数组的 `shared_ptr`。这些对象应稍后用确定的值覆盖：

    ```cpp
    std::shared_ptr<int> pi = std::make_shared_for_overwrite<int>();
    std::shared_ptr<foo[]> pa = std::make_shared_for_overwrite<foo[]>(3); 
    ```

+   如果默认的删除操作不适用于销毁被管理的对象，请使用重载的构造函数，它接受一个自定义删除器：

    ```cpp
    std::shared_ptr<foo> pf1(new foo(42, 42.0, "42"),
                             foo_deleter());
    std::shared_ptr<foo> pf2(
     new foo(42, 42.0, "42"),
            [](foo* p) {
              std::cout << "deleting foo from lambda..." << '\n';
     delete p;}); 
    ```

+   在管理对象的数组时，始终指定一个删除器。删除器可以是 `std::default_delete` 的数组部分特化，或者任何接受模板类型指针的函数：

    ```cpp
    std::shared_ptr<int> pa1(
     new int[3]{ 1, 2, 3 },
      std::default_delete<int[]>());
    std::shared_ptr<int> pa2(
     new int[3]{ 1, 2, 3 },
      [](auto p) {delete[] p; }); 
    ```

+   要访问管理对象的原始指针，请使用 `get()` 函数：

    ```cpp
    void func(int* ptr)
    {
      if (ptr != nullptr)
        std::cout << *ptr << '\n';
      else
        std::cout << "null" << '\n';
    }
    std::shared_ptr<int> pi;
    func(pi.get());
    pi = std::make_shared<int>(42);
    func(pi.get()); 
    ```

+   使用 `operator*` 和 `operator->` 解引用管理对象的指针：

    ```cpp
    std::shared_ptr<int> pi = std::make_shared<int>(42);
    *pi = 21;
    std::shared_ptr<foo> pf = std::make_shared<foo>(42, 42.0, "42");
    pf->print(); 
    ```

+   如果 `shared_ptr` 管理对象的数组，可以使用 `operator[]` 访问数组的各个元素。这仅在 C++17 中可用：

    ```cpp
    std::shared_ptr<int[]> pa1(
     new int[3]{ 1, 2, 3 },
      std::default_delete<int[]>());
    for (int i = 0; i < 3; ++i)
      pa1[i] *= 2; 
    ```

+   要检查 `shared_ptr` 是否可以管理一个对象，请使用显式操作符 `bool` 或检查 `get() != nullptr`（这是操作符 `bool` 所做的）：

    ```cpp
    std::shared_ptr<int> pnull;
    if (pnull) std::cout << "not null" << '\n';
    std::shared_ptr<int> pi(new int(42));
    if (pi) std::cout << "not null" << '\n'; 
    ```

+   `shared_ptr` 对象可以存储在容器中，例如 `std::vector`：

    ```cpp
    std::vector<std::shared_ptr<foo>> data;
    for (int i = 0; i < 5; i++)
      data.push_back(
        std::make_shared<foo>(i, i, std::to_string(i)));
    auto pf = std::make_shared<foo>(42, 42.0, "42");
    data.push_back(std::move(pf));
    assert(!pf); 
    ```

+   使用 `weak_ptr` 维护对共享对象的非拥有引用，稍后可以通过从 `weak_ptr` 对象构造的 `shared_ptr` 访问：

    ```cpp
    auto sp1 = std::make_shared<int>(42);
    assert(sp1.use_count() == 1);
    std::weak_ptr<int> wpi = sp1;
    assert(sp1.use_count() == 1);
    auto sp2 = wpi.lock(); // sp2 type is std::shared_ptr<int>
    assert(sp1.use_count() == 2);
    assert(sp2.use_count() == 2);
    sp1.reset();
    assert(sp1.use_count() == 0);
    assert(sp2.use_count() == 1); 
    ```

+   当你需要为已由另一个 `shared_ptr` 对象管理的实例创建 `shared_ptr` 对象时，请将 `std::enable_shared_from_this` 类模板用作类型的基类：

    ```cpp
    struct Apprentice;
    struct Master : std::enable_shared_from_this<Master>
    {
      ~Master() { std::cout << "~Master" << '\n'; }
      void take_apprentice(std::shared_ptr<Apprentice> a);
    private:
      std::shared_ptr<Apprentice> apprentice;
    };
    struct Apprentice
    {
      ~Apprentice() { std::cout << "~Apprentice" << '\n'; }
      void take_master(std::weak_ptr<Master> m);
    private:
      std::weak_ptr<Master> master;
    };
    void Master::take_apprentice(std::shared_ptr<Apprentice> a)
    {
      apprentice = a;
      apprentice->take_master(shared_from_this());
    }
    void Apprentice::take_master(std::weak_ptr<Master> m)
    {
      master = m;
    }
    auto m = std::make_shared<Master>();
    auto a = std::make_shared<Apprentice>();
    m->take_apprentice(a); 
    ```

## 它是如何工作的...

在许多方面，`shared_ptr` 与 `unique_ptr` 非常相似；然而，它服务于不同的目的：共享对象或数组的所有权。两个或多个 `shared_ptr` 智能指针可以管理同一个动态分配的对象或数组，当最后一个智能指针超出作用域、使用 `operator=` 分配新指针或使用 `reset()` 方法重置时，该对象或数组将被自动销毁。默认情况下，对象使用 `operator delete` 销毁；然而，用户可以向构造函数提供一个自定义删除器，这是使用 `std::make_shared()` 所不可能做到的。如果 `shared_ptr` 用于管理对象的数组，必须提供一个自定义删除器。在这种情况下，你可以使用 `std::default_delete<T[]>`，它是 `std::default_delete` 类模板的部分特化，使用 `operator delete[]` 来删除动态分配的数组。

与自 C++14 才可用的 `std::make_unique()` 不同，`std::make_shared()`（自 C++11 起可用）应用于创建智能指针，除非你需要提供一个自定义删除器。主要原因与 `make_unique()` 相同：避免在某些上下文中抛出异常时潜在的内存泄漏。有关更多信息，请阅读前一道菜谱中提供的 `std::make_unique()` 的解释。

在 C++20 中，添加了一个新的函数，称为 `std::make_shared_for_overwrite()`。这个函数与 `make_shared()` 类似，但默认初始化对象或对象数组。这个函数可以在未知类型模板参数是否是平凡可复制的泛型代码中使用。这个函数表达了创建一个可能未初始化的对象的指针的意图，以便稍后可以覆盖它。

此外，与 `unique_ptr` 的情况类似，一个管理 `Derived` 类对象的 `shared_ptr` 可以隐式转换为管理 `Base` 类对象的 `shared_ptr`。这只有在 `Derived` 类从 `Base` 类派生时才可能。这种隐式转换只有在 `Base` 类有一个虚析构函数（正如所有基类在应该通过基类的指针或引用多态删除对象时应该有的那样）时才是安全的；否则，将执行未定义的行为。在 C++17 中，添加了几个新的非成员函数：`std::static_pointer_cast()`、`std::dynamic_pointer_cast()`、`std::const_pointer_cast()` 和 `std::reinterpret_pointer_cast()`。这些函数将 `static_cast`、`dynamic_cast`、`const_cast` 和 `reinterpret_cast` 应用于存储的指针，并返回一个新的指向指定类型的 `shared_ptr`。

在以下示例中，`Base` 和 `Derived` 是我们在上一个示例中使用的相同类：

```cpp
std::shared_ptr<Derived> pd = std::make_shared<Derived>();
std::shared_ptr<Base> pb = pd;
std::static_pointer_cast<Derived>(pb)->print(); 
```

有时你需要一个智能指针来管理共享对象，但又不希望它对共享所有权做出贡献。假设你模拟一个树结构，其中节点对其子节点有引用，并且它们由 `shared_ptr` 对象表示。另一方面，假设一个节点需要保持对其父节点的引用。如果这个引用也是 `shared_ptr`，那么它将创建循环引用，并且没有任何对象会被自动销毁。

`weak_ptr` 是一种智能指针，用于打破这种循环依赖。它持有对由 `shared_ptr` 管理的对象或数组的非拥有引用。可以从 `shared_ptr` 对象创建 `weak_ptr`。为了访问管理的对象，你需要获取一个临时的 `shared_ptr` 对象。为此，我们需要使用 `lock()` 方法。此方法原子性地检查所引用的对象是否仍然存在，如果对象不再存在，则返回一个空的 `shared_ptr`，如果对象仍然存在，则返回一个拥有该对象的 `shared_ptr`。由于 `weak_ptr` 是一个非拥有智能指针，因此所引用的对象可以在 `weak_ptr` 超出作用域之前或当所有拥有 `shared_ptr` 对象被销毁、重置或分配给其他指针时被销毁。可以使用 `expired()` 方法来检查所引用的对象是否已被销毁或仍然可用。

在*如何做...*部分，前面的示例模拟了一个师徒关系。有一个`Master`类和一个`Apprentice`类。`Master`类有一个对`Apprentice`类的引用和一个名为`take_apprentice()`的方法来设置`Apprentice`对象。`Apprentice`类有一个对`Master`类的引用和一个名为`take_master()`的方法来设置`Master`对象。为了避免循环依赖，这些引用中的一个必须由一个`weak_ptr`表示。在提出的示例中，`Master`类有一个`shared_ptr`来拥有`Apprentice`对象，而`Apprentice`类有一个`weak_ptr`来跟踪对`Master`对象的引用。然而，这个示例稍微复杂一些，因为在这里，`Apprentice::take_master()`方法是从`Master::take_apprentice()`中调用的，并且需要一个`weak_ptr<Master>`。为了在`Master`类内部调用它，我们必须能够在`Master`类中使用`this`指针创建一个`shared_ptr<Master>`。在安全的方式中做到这一点的唯一方法是使用`std::enable_shared_from_this`。

`std::enable_shared_from_this`是一个类模板，必须用作所有需要为当前对象（`this`指针）创建`shared_ptr`的类的基类，当此对象已被另一个`shared_ptr`管理时。它的类型模板参数必须是派生自它的类，就像在好奇的递归模板模式中一样。它有两个方法：`shared_from_this()`，它返回一个`shared_ptr`，共享`this`对象的拥有权，和`weak_from_this()`，它返回一个`weak_ptr`，共享对`this`对象的非拥有引用。后者方法仅在 C++17 中可用。这些方法只能在由现有`shared_ptr`管理的对象上调用；否则，它们会抛出`std::bad_weak_ptr`异常，自 C++17 起。在 C++17 之前，行为是未定义的。

不使用`std::enable_shared_from_this`并直接创建`shared_ptr<T>(this)`会导致有多个`shared_ptr`对象独立管理同一个对象，彼此之间不知道。当这种情况发生时，对象最终会被不同的`shared_ptr`对象多次销毁。

## 参见

+   *使用`unique_ptr`来唯一拥有内存资源*，学习`std::unique_ptr`类，它代表一个智能指针，它拥有并管理在堆上分配的另一个对象或对象数组

# 与操作符`<=>`的一致比较

C++语言定义了六个关系运算符来执行比较：`==`、`!=`、`<`、`<=`、`>`和`>=`。尽管`!=`可以用`==`来实现，而`<=`、`>=`和`>`可以用`<`来实现，但如果你想让用户定义的类型支持相等比较，你仍然必须实现`==`和`!=`；如果你想让它支持排序，你必须实现`<`、`<=`、`>`和`>=`。

这意味着如果你的类型——让我们称它为 `T`——的对象要可比较，那么需要 6 个函数；如果它们要与另一个类型 `U` 可比较，则需要 12 个函数；如果还要使 `U` 类型的值与你的 `T` 类型可比较，则需要 18 个函数，依此类推。新的 C++20 标准通过引入一个新的比较操作符，称为三向比较，将这个数字减少到 1 或 2，或者这些数字的倍数（取决于与其他类型的比较），这个新的比较操作符用符号 `<=>` 表示，因此它通常被称为 *飞船操作符*。这个新操作符帮助我们编写更少的代码，更好地描述关系的强度，并避免手动实现比较操作符时可能出现的性能问题。

## 准备工作

在定义或实现三向比较操作符时，必须包含头文件 `<compare>`。这个新的 C++20 头文件是标准通用工具库的一部分，它提供了用于实现比较的类、函数和概念。

## 如何做到这一点…

要在 C++20 中最优地实现比较，请执行以下操作：

+   如果你只想让你的类型支持相等比较（包括 `==` 和 `!=`），则只需实现 `==` 操作符并返回一个 `bool`。你可以默认实现，以便编译器执行逐成员比较：

    ```cpp
    class foo
    {
      int value;
    public:
      foo(int const v):value(v){}
      bool operator==(foo const&) const = default;
    }; 
    ```

+   如果你希望你的类型同时支持相等和排序，并且默认的成员比较就足够了，那么只需定义 `<=>` 操作符，返回 `auto`，并默认其实现：

    ```cpp
    class foo
    {
      int value;
    public:
      foo(int const v) :value(v) {}
      auto operator<=>(foo const&) const = default;
    }; 
    ```

+   如果你希望你的类型同时支持相等和排序，并且需要执行自定义比较，那么实现 `==` 操作符（用于相等）和 `<=>` 操作符（用于排序）：

    ```cpp
    class foo
    {
      int value;
    public:
      foo(int const v) :value(v) {}
      bool operator==(foo const& other) const
      { return value == other.value; }
      auto operator<=>(foo const& other) const
      { return value <=> other.value; }
    }; 
    ```

在实现三向比较操作符时，请遵循以下指南：

+   仅实现三向比较操作符，但在比较值时始终使用双向比较操作符 `<`、`<=`、`>` 和 `>=`。

+   即使你想要比较操作符的第一个操作数是除你的类之外的其他类型，也要将三向比较操作符实现为成员函数。

+   仅当你在两个参数上想要隐式转换时，才将三向比较操作符实现为非成员函数（这意味着比较两个对象，它们都不是你的类）。

## 它是如何工作的…

新的三向比较操作符类似于 `memcmp()`/`strcmp()` C 函数和 `std::string::compare()` 方法。这些函数接受两个参数，并返回一个整数值，如果第一个小于第二个，则返回小于零的值；如果它们相等，则返回零；如果第一个参数大于第二个参数，则返回大于零的值。三向比较操作符不返回整数，而是返回比较类别类型的值。

这可以是以下之一：

+   `std::strong_ordering` 表示支持所有六个关系运算符的三向比较的结果，不允许不可比较的值（这意味着 `a < b`、`a == b` 和 `a > b` 至少有一个必须为真），并且意味着可替换性。这是一个属性，如果 `a == b` 并且 `f` 是一个只读取比较显著状态（通过参数的公共常量成员访问）的函数，那么 `f(a) == f(b)`。

+   `std::weak_ordering` 支持所有六个关系运算符，不支持不可比较的值（这意味着 `a < b`、`a == b` 和 `a > b` 都可能不为真），但也不意味着可替换性。一个典型的定义弱排序的类型是无大小写敏感的字符串类型。

+   `std::partial_ordering` 支持所有六个关系运算符，但不意味着可替换性，并且其值可能不可比较（例如，浮点数 `NaN` 不能与任何其他值进行比较）。

`std::strong_ordering` 类型是所有这些类别类型中最强的。它不能隐式转换为任何其他类别，但它可以隐式转换为 `std::weak_ordering` 和 `std::partial_ordering`。`std::weak_ordering` 也可以隐式转换为 `std::partial_ordering`。我们已在以下表格中总结了所有这些属性：

| **类别** | **运算符** | **可替换性** | **可比较值** | **隐式转换** |
| --- | --- | --- | --- | --- |
| `std::strong_ordering` | `==`, `!=`, `<`, `<=`, `>`, `>=` | 是 | 是 | ![](img/B21549_09_001.png) |
| `std::weak_ordering` | `==`, `!=`, `<`, `<=`, `>`, `>=` | 否 | 是 | ![](img/B21549_09_001.png) |
| `std::partial_ordering` | `==`, `!=`, `<`, `<=`, `>`, `>=` | 否 | 否 |  |

表 9.2：类别类型属性

这些比较类别具有隐式可比较于文字零（但不能与零的整数变量）的值。它们的值列在以下表格中：

| **类别** | **数值** | **非数值** |
| --- | --- | --- |
| -1 | 0 | 1 |
| `strong_ordering` | 小于 | 等价 | 大于 |  |
| `weak_ordering` | 小于 | 等价 | 大于 |  |
| `partial_ordering` | 小于 | 等价 | 大于 | 无序 |

表 9.3：隐式可比较于文字零的比较类别值

为了更好地理解其工作原理，让我们看看以下示例：

```cpp
class cost_unit_t
{
  // data members
public:
  std::strong_ordering operator<=>(cost_unit_t const & other) const noexcept = default;
};
class project_t : public cost_unit_t
{
  int         id;
  int         type;
  std::string name;
public:
  bool operator==(project_t const& other) const noexcept
  {
    return (cost_unit_t&)(*this) == (cost_unit_t&)other &&
           name == other.name &&
           type == other.type &&
           id == other.id;
  }
  std::strong_ordering operator<=>(project_t const & other) const noexcept
  {
    // compare the base class members
if (auto cmp = (cost_unit_t&)(*this) <=> (cost_unit_t&)other;
        cmp != 0)
      return cmp;
    // compare this class members in custom order
if (auto cmp = name.compare(other.name); cmp != 0)
      return cmp < 0 ? std::strong_ordering::less :
                       std::strong_ordering::greater;
    if (auto cmp = type <=> other.type; cmp != 0)
      return cmp;
    return id <=> other.id;
  }
}; 
```

在这里，`cost_unit_t`是一个基类，它包含一些（未指定的）数据成员并定义了`<=>`运算符，尽管它是由编译器默认实现的。这意味着编译器还将提供`==`和`!=`运算符，而不仅仅是`<`、`<=`、`>`和`>=`。这个类通过`project_t`派生，它包含几个数据字段：项目的标识符、类型和名称。然而，对于这种类型，我们不能默认实现运算符的实现，因为我们不想逐字段比较成员，而是按照自定义的顺序：首先是名称，然后是类型，最后是标识符。在这种情况下，我们实现了`==`运算符，它返回`bool`并测试成员字段是否相等，以及`<=>`运算符，它返回`std::strong_ordering`并使用其自身的`<=>`运算符来比较两个参数的值。

```cpp
employee_t that models employees in a company. An employee can have a manager, and an employee who is a manager has people that they manage. Conceptually, such a type could look as follows:
```

```cpp
struct employee_t
{
  bool is_managed_by(employee_t const&) const { /* ... */ }
  bool is_manager_of(employee_t const&) const { /* ... */ }
  bool is_same(employee_t const&) const { /* ... */ }
  bool operator==(employee_t const & other) const
  {
    return is_same(other);
  }
  std::partial_ordering operator<=>(employee_t const& other) const noexcept
  {
    if (is_same(other))
      return std::partial_ordering::equivalent;
    if (is_managed_by(other))
      return std::partial_ordering::less;
    if (is_manager_of(other))
      return std::partial_ordering::greater;
    return std::partial_ordering::unordered;
  }
}; 
```

`is_same()`、`is_manager_of()`和`is_managed_by()`方法返回两个员工之间的关系。然而，可能存在没有关系的员工；例如，来自不同团队的员工，或者没有经理-下属结构的团队。在这里，我们可以实现相等和排序。然而，由于我们不能比较所有员工，`<=>`运算符必须返回`std::partial_ordering`值。如果值代表相同的员工，则返回值是`partial_ordering::equivalent`；如果当前员工由提供的员工管理，则返回`partial_ordering::less`；如果当前员工是提供的员工的管理者，则返回`partial_ordering::greater`；在其他所有情况下返回`partial_ordering::unorder`。

让我们再看一个例子来理解三向比较运算符是如何工作的。在以下示例中，`ipv4`类模拟了一个 IP 版本 4 地址。它支持与其他`ipv4`类型的对象以及`unsigned long`值的比较（因为有一个`to_unlong()`方法，它将 IP 地址转换为 32 位无符号整数值）：

```cpp
struct ipv4
{
  explicit ipv4(unsigned char const a=0, unsigned char const b=0,
 unsigned char const c=0, unsigned char const d=0) noexcept :
    data{ a,b,c,d }
  {}
  unsigned long to_ulong() const noexcept
 {
    return
      (static_cast<unsigned long>(data[0]) << 24) |
      (static_cast<unsigned long>(data[1]) << 16) |
      (static_cast<unsigned long>(data[2]) << 8) |
      static_cast<unsigned long>(data[3]);
  }
  auto operator<=>(ipv4 const&) const noexcept = default;
  bool operator==(unsigned long const other) const noexcept
  {
    return to_ulong() == other;
  }
  std::strong_ordering
  operator<=>(unsigned long const other) const noexcept
  {
    return to_ulong() <=> other;
  }
private:
  std::array<unsigned char, 4> data;
}; 
```

在这个例子中，我们重载了`<=>`运算符并允许它默认实现。但我们还明确实现了`operator==`和`operator<=>`的重载，这些运算符用于比较`ipv4`对象与`unsigned long`值。因为这些运算符，我们可以写出以下任何一种形式：

```cpp
ipv4 ip(127, 0, 0, 1);
if(ip == 0x7F000001) {}
if(ip != 0x7F000001) {}
if(0x7F000001 == ip) {}
if(0x7F000001 != ip) {}
if(ip < 0x7F000001)  {}
if(0x7F000001 < ip)  {} 
```

这里有两个需要注意的地方：第一个是，尽管我们只重载了`==`运算符，我们也可以使用`!=`运算符；第二个是，尽管我们重载了`==`运算符和`<=>`运算符来比较`ipv4`值与`unsigned long`值，我们也可以比较`unsigned long`值与`ipv4`值。这是因为编译器执行对称重载解析。这意味着对于表达式`a@b`（其中`@`是一个双向关系运算符），它执行`a@b`、`a<=>b`和`b<=>a`的名称查找。以下表格显示了所有可能的关系运算符转换：

| `a == b` | `b == a` |  |
| --- | --- | --- |
| `a != b` | `!(a == b)` | `!(b == a)` |
| `a <=> b` | `0 <=> (b <=> a)` |  |
| `a < b` | `(a <=> b) < 0` | `0 > (b <=> a)` |
| `a <= b` | `(a <=> b) <= 0` | `0 >= (b <=> a)` |
| `a > b` | `(a <=> b) > 0` | `0 < (b <=> a)` |
| `a >= b` | `(a <=> b) >= 0` | `0 <= (b <=> a)` |

表 9.4：关系运算符的可能转换

这大大减少了你必须显式提供的重载数量，以支持不同形式的比较。三向比较运算符可以实施为成员函数或非成员函数。通常，你应该优先选择成员实现。

只有在你想要两个参数都进行隐式转换时才应使用非成员形式。以下是一个示例：

```cpp
struct A { int i; };
struct B
{
  B(A a) : i(a.i) { }
  int i;
};
inline auto
operator<=>(B const& lhs, B const& rhs) noexcept
{
  return lhs.i <=> rhs.i;
}
assert(A{ 2 } > A{ 1 }); 
```

虽然`<=>`运算符为类型`B`定义，因为它是一个非成员运算符，并且由于`A`可以隐式转换为`B`，我们可以对`A`类型的对象执行比较操作。

## 参见

+   *第一章*，*使用类模板参数推导简化代码*，学习如何在不显式指定模板参数的情况下使用类模板

+   *确保程序常量正确性*，以探索常量正确性的好处以及如何实现它

# 安全地比较有符号和无符号整数

C++语言具有多种整型：`short`、`int`、`long`和`long long`，以及它们的无符号对应类型`unsigned short`、`unsigned int`、`unsigned long`和`unsigned long long`。在 C++11 中，引入了固定宽度的整型，例如`int32_t`和`uint32_t`，以及许多类似的类型。除此之外，还有`char`、`signed char`、`unsigned char`、`wchar_t`、`char8_t`、`char16_t`和`char32_t`这些类型，尽管它们不是为了存储数字而是为了存储字符。此外，用于存储`true`或`false`值的`bool`类型也是一个整型。这些类型值的比较是一个常见的操作，但比较有符号和无符号值是容易出错的。如果没有一些特定的编译器开关来将这些操作标记为警告或错误，你就可以执行这些操作并得到意外的结果。例如，比较`-1 < 42u`（比较有符号的-1 和无符号的 42）将返回`false`。C++20 标准提供了一套用于执行有符号和无符号值安全比较的函数，我们将在本食谱中学习这些函数。

## 如何做…

要执行确保负有符号整数始终比较小于无符号整数的无符号和有符号整数的安全比较，请使用`<utility>`头文件中的以下比较函数之一：

| **函数** | **对应的比较运算符** |
| --- | --- |
| `std::cmp_equal` | `==` |
| `std::cmp_not_equal` | `!=` |
| `std::cmp_less` | `<` |
| `std::cmp_less_equal` | `<=` |
| `std::cmp_greater` | `>` |
| `std::cmp_greater_equal` | `>=` |

表 9.5：新的 C++20 比较函数及其对应的比较运算符

以下是一个示例：

```cpp
int a = -1;
unsigned int b = 42;
if (std::cmp_less(a, b)) // a is less than b so this returns true
{
   std::cout << "-1 < 42\n";
}
else
{
   std::cout << "-1 >= 42\n";
} 
```

## 它是如何工作的…

比较两个有符号或两个无符号值很简单，但比较一个有符号和一个无符号整数则容易出错。当发生此类比较时，有符号值会被转换为无符号。例如，整数-1 变为 4294967295。这是因为有符号数在内存中的存储方式如下：

+   最高有效位表示符号：正数为 0，负数为 1。

+   负值通过取正数的位反并加 1 来存储。

这种表示法被称为**二进制补码**。例如，假设是一个 8 位有符号表示，值 1 存储为`0000001`，但值-1 存储为`11111111`。这是因为正数的 7 个最低有效位是`0000001`，取反后是`1111110`。通过加 1，我们得到`1111111`。与符号位一起，这构成了`11111111`。对于 32 位有符号整数，值-1 存储为`11111111'11111111'11111111'11111111`。

```cpp
signed -1 and unsigned 42 will print *-1 >= 42* because the actual comparison occurs between unsigned 4294967295 and unsigned 42.
```

```cpp
int a = -1;
unsigned int b = 42;
if(a < b)
{
   std::cout << "-1 < 42\n";
}
else
{
   std::cout << "-1 >= 42\n";
} 
```

这适用于所有六个相等（`==`，`!=`）和不相等（`<`，`<=`，`>`，`>=`）运算符。为了得到正确的结果，我们需要检查有符号值是否为负。之前显示的`if`语句的正确条件如下：

```cpp
if(a < 0 || static_cast<unsigned int>(a) < b) 
```

为了简化此类表达式的编写，C++20 标准引入了表 9.5 中列出的六个函数，应在比较有符号和无符号整数时用作相应运算符的替代。

```cpp
if(std::cmp_less(a, b))
{
   std::cout << "-1 < 42\n";
}
else
{
   std::cout << "-1 >= 42\n";
} 
```

下面的代码片段展示了`std::cmp_less()`函数的一个可能的实现：

```cpp
template<class T, class U>
constexpr bool cmp_less(T t, U u) noexcept
{
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
 return t < u;
    else if constexpr (std::is_signed_v<T>)
        return t < 0 || std::make_unsigned_t<T>(t) < u;
    else
return u >= 0 && t < std::make_unsigned_t<U>(u);
} 
```

这所做的是以下内容：

+   如果两个参数都是有符号的，它使用内置的`<`比较运算符来比较它们。

+   如果第一个参数是有符号的，第二个是无符号的，那么它检查第一个是否是本地的（负值总是小于正值）或者使用内置运算符`<`将第一个参数转换为无符号并与第二个参数进行比较。

+   如果第一个参数是无符号的，第二个可以是有符号或无符号的。第一个参数只能小于第二个，如果第二个是正数，并且第一个参数小于将其转换为无符号的第二个参数。

当你使用这些函数时，请记住它们只适用于：

+   `short`，`int`，`long`，`long long`及其无符号对应类型

+   固定宽度整数类型，如`int32_t`，`int_least32_t`，`int_fast32_t`及其无符号对应类型

+   扩展整数类型（这些是编译器特定的类型，如`__int64`或`__int128`及其大多数编译器支持的无符号对应类型）

下面的代码片段提供了一个使用扩展类型（在这种情况下是 Microsoft 特定的）和标准固定宽度整数类型的示例。

```cpp
__int64 a = -1;
unsigned __int64 b = 42;
if (std::cmp_less(a, b))  // OK
{ }
int32_t  a = -1;
uint32_t b = 42;
if (std::cmp_less(a, b))  // OK
{ } 
```

然而，你不能用它们来比较枚举，`std::byte`，`char`，`char8_t`，`char16_t`，`char32_t`，`wchar_t`和`bool`。在这种情况下，你会得到编译器错误：

```cpp
if (std::cmp_equal(true, 1)) // error
{ } 
```

## 参见

+   *第二章*，*理解各种数值类型*，了解可用的整数和浮点类型

+   *执行正确的类型转换*，了解在 C++ 中执行类型转换的正确方法

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
