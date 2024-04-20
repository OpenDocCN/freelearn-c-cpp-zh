# 第十章：探索函数

本章包含的示例如下：

+   默认和删除的函数

+   使用 lambda 与标准算法

+   使用通用 lambda

+   编写递归 lambda

+   编写具有可变数量参数的函数模板

+   使用折叠表达式简化可变参数函数模板

+   实现高阶函数 map 和 fold

+   将函数组合成高阶函数

+   统一调用任何可调用的东西

# 默认和删除的函数

在 C++中，类有特殊成员（构造函数、析构函数和运算符），可以由编译器默认实现，也可以由开发人员提供。然而，可以默认实现的规则有点复杂，可能会导致问题。另一方面，开发人员有时希望阻止对象以特定方式被复制、移动或构造。通过使用这些特殊成员实现不同的技巧是可能的。C++11 标准通过允许函数被删除或默认实现简化了许多这样的问题，我们将在下一节中看到。

# 入门

对于这个示例，你需要知道什么是特殊成员函数，以及可复制和可移动的含义。

# 如何做...

使用以下语法指定如何处理函数：

+   要默认一个函数，使用`=default`而不是函数体。只有具有默认值的特殊类成员函数可以被默认：

```cpp
        struct foo 
        { 
          foo() = default; 
        };
```

+   要删除一个函数，使用`=delete`而不是函数体。任何函数，包括非成员函数，都可以被删除：

```cpp
        struct foo 
        { 
          foo(foo const &) = delete; 
        }; 

        void func(int) = delete;
```

使用默认和删除的函数来实现各种设计目标，例如以下示例：

+   要实现一个不可复制且隐式不可移动的类，将复制操作声明为已删除：

```cpp
        class foo_not_copyable 
        { 
        public: 
          foo_not_copyable() = default; 

          foo_not_copyable(foo_not_copyable const &) = delete; 
          foo_not_copyable& operator=(foo_not_copyable const&) = delete; 
        };
```

+   要实现一个不可复制但可移动的类，将复制操作声明为已删除，并显式实现移动操作（并提供任何需要的其他构造函数）：

```cpp
        class data_wrapper 
        { 
          Data* data; 
        public: 
          data_wrapper(Data* d = nullptr) : data(d) {} 
          ~data_wrapper() { delete data; } 

          data_wrapper(data_wrapper const&) = delete; 
          data_wrapper& operator=(data_wrapper const &) = delete; 

          data_wrapper(data_wrapper&& o) :data(std::move(o.data))  
          {  
            o.data = nullptr;  
          } 

          data_wrapper& operator=(data_wrapper&& o) 
          { 
            if (this != &o) 
            { 
              delete data; 
              data = std::move(o.data); 
              o.data = nullptr; 
            } 

            return *this; 
          } 
        };
```

+   为了确保一个函数只能被特定类型的对象调用，并可能防止类型提升，为函数提供已删除的重载（以下示例中的自由函数也可以应用于任何类成员函数）：

```cpp
        template <typename T> 
        void run(T val) = delete; 

        void run(long val) {} // can only be called with long integers
```

# 工作原理...

一个类有几个特殊成员，可以由编译器默认实现。这些是默认构造函数、复制构造函数、移动构造函数、复制赋值、移动赋值和析构函数。如果你不实现它们，那么编译器会这样做，以便可以创建、移动、复制和销毁类的实例。然而，如果你显式提供了其中一个或多个特殊方法，那么编译器将根据以下规则不生成其他方法：

+   如果存在用户定义的构造函数，则默认构造函数不会被默认生成。

+   如果存在用户定义的虚拟析构函数，则默认构造函数不会被默认生成。

+   如果存在用户定义的移动构造函数或移动赋值运算符，则默认不会生成复制构造函数和复制赋值运算符。

+   如果存在用户定义的复制构造函数、移动构造函数、复制赋值运算符、移动赋值运算符或析构函数，则默认不会生成移动构造函数和移动赋值运算符。

+   如果存在用户定义的复制构造函数或析构函数，则默认生成复制赋值运算符。

+   如果存在用户定义的复制赋值运算符或析构函数，则默认生成复制构造函数。

请注意，前面列表中的最后两条规则是被弃用的规则，可能不再被你的编译器支持。

有时，开发人员需要提供这些特殊成员的空实现或隐藏它们，以防止以特定方式构造类的实例。一个典型的例子是一个不应该被复制的类。这种情况的经典模式是提供一个默认构造函数并隐藏复制构造函数和复制赋值运算符。虽然这样可以工作，但显式定义的默认构造函数确保了该类不再被视为平凡的，因此不再是 POD 类型。这种情况的现代替代方法是使用前面部分所示的删除函数。

当编译器在函数定义中遇到`=default`时，它将提供默认实现。之前提到的特殊成员函数的规则仍然适用。如果函数是内联的，函数可以在类的主体之外声明为`=default`：

```cpp
    class foo 
    { 
    public: 
      foo() = default; 

      inline foo& operator=(foo const &); 
    }; 

    inline foo& foo::operator=(foo const &) = default;
```

当编译器在函数定义中遇到`=delete`时，它将阻止调用该函数。但是，在重载解析期间仍然会考虑该函数，只有在删除的函数是最佳匹配时，编译器才会生成错误。例如，通过为`run()`函数给出先前定义的重载，只有长整数的调用是可能的。对于任何其他类型的参数，包括`int`，其中存在自动类型提升为`long`的情况，将确定删除的重载被认为是最佳匹配，因此编译器将生成错误：

```cpp
    run(42);  // error, matches a deleted overload 
    run(42L); // OK, long integer arguments are allowed
```

请注意，之前声明的函数不能被删除，因为`=delete`定义必须是翻译单元中的第一个声明：

```cpp
    void forward_declared_function(); 
    // ... 
    void forward_declared_function() = delete; // error
```

经验法则（也称为*五大法则*）适用于类特殊成员函数，即，如果您明确定义了任何复制构造函数、移动构造函数、复制赋值运算符、移动赋值运算符或析构函数，则您必须明确定义或默认所有这些函数。

# 使用标准算法与 lambda

C++最重要的现代特性之一是 lambda 表达式，也称为 lambda 函数或简单的 lambda。Lambda 表达式使我们能够定义可以捕获作用域中的变量并被调用或作为参数传递给函数的匿名函数对象。Lambda 在许多方面都很有用，在这个配方中，我们将看到如何将它们与标准算法一起使用。

# 准备就绪

在这个配方中，我们讨论了接受作为其迭代的元素的函数或谓词参数的标准算法。您需要了解什么是一元和二元函数，以及什么是谓词和比较函数。您还需要熟悉函数对象，因为 lambda 表达式是函数对象的语法糖。

# 如何做...

您应该更倾向于使用 lambda 表达式将回调传递给标准算法，而不是函数或函数对象：

+   如果您只需要在一个地方使用 lambda，则在调用的地方定义匿名 lambda 表达式：

```cpp
        auto numbers =  
          std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 }; 
        auto positives = std::count_if( 
          std::begin(numbers), std::end(numbers),  
          [](int const n) {return n > 0; });
```

+   如果您需要在多个地方调用 lambda，则定义一个命名 lambda，即分配给变量的 lambda（通常使用`auto`指定符为类型）：

```cpp
        auto ispositive = [](int const n) {return n > 0; }; 
        auto positives = std::count_if( 
          std::begin(numbers), std::end(numbers), ispositive);
```

+   如果您需要在参数类型上有所不同的 lambda，则使用通用 lambda 表达式（自 C++14 起可用）：

```cpp
        auto positives = std::count_if( 
          std::begin(numbers), std::end(numbers),  
          [](auto const n) {return n > 0; });
```

# 它是如何工作的...

在之前的第二个项目符号中显示的非通用 lambda 表达式接受一个常量整数，并在大于`0`时返回`true`，否则返回`false`。编译器定义了一个具有 lambda 表达式签名的无名函数对象的调用运算符：

```cpp
    struct __lambda_name__ 
    { 
      bool operator()(int const n) const { return n > 0; } 
    };
```

编译器定义的未命名函数对象的方式取决于我们定义 lambda 表达式的方式，它可以捕获变量，使用`mutable`说明符或异常规范，或具有尾部返回类型。之前显示的`__lambda_name__`函数对象实际上是编译器生成的简化版本，因为它还定义了默认的复制和移动构造函数，默认的析构函数和已删除的赋值运算符。

必须充分理解，lambda 表达式实际上是一个类。为了调用它，编译器需要实例化一个类的对象。从 lambda 表达式实例化的对象称为*lambda 闭包*。

在下一个例子中，我们想要计算范围内大于或等于 5 且小于或等于 10 的元素的数量。在这种情况下，lambda 表达式将如下所示：

```cpp
    auto numbers = std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 }; 
    auto start{ 5 }; 
    auto end{ 10 }; 
    auto inrange = std::count_if( 
             std::begin(numbers), std::end(numbers),  
             start, end {
                return start <= n && n <= end;});
```

此 lambda 通过复制（即值）捕获两个变量`start`和`end`。编译器创建的结果未命名函数对象看起来非常像我们之前定义的那个。通过前面提到的默认和已删除的特殊成员，该类如下所示：

```cpp
    class __lambda_name_2__ 
    { 
      int start_; 
      int end_; 
    public: 
      explicit __lambda_name_2__(int const start, int const end) : 
        start_(start), end_(end) 
      {} 

      __lambda_name_2__(const __lambda_name_2__&) = default; 
      __lambda_name_2__(__lambda_name_2__&&) = default; 
      __lambda_name_2__& operator=(const __lambda_name_2__&)  
         = delete; 
      ~__lambda_name_2__() = default; 

      bool operator() (int const n) const 
      { 
        return start_ <= n && n <= end_; 
      } 
    };
```

lambda 表达式可以通过复制（或值）或引用捕获变量，两者的不同组合是可能的。但是，不可能多次捕获变量，并且只能在捕获列表的开头使用`&`或`=`。

lambda 只能捕获封闭函数范围内的变量。它不能捕获具有静态存储期限的变量（即在命名空间范围内声明或使用`static`或`external`说明符声明的变量）。

以下表格显示了 lambda 捕获语义的各种组合。

| 描述 |
| --- |
| 不捕获任何东西 |
| 通过引用捕获一切 |
| 通过复制捕获一切 |
| 仅通过引用捕获`x` |
| 仅通过复制捕获`x` |
| 通过引用捕获包扩展`x` |
| 通过复制捕获包扩展`x` |
| 通过引用捕获一切，除了通过复制捕获的`x` |
| 通过复制捕获一切，除了通过引用捕获的`x` |
| 通过引用捕获一切，除了指针`this`被复制捕获（`this`始终被复制捕获） |
| 错误，`x`被捕获两次 |
| 错误，一切都被引用捕获，不能再次指定通过引用捕获`x` |
| 错误，一切都被复制捕获，不能再次指定通过复制捕获`x` |
| 错误，指针`this`始终被复制捕获 |
| 错误，不能同时通过复制和引用捕获一切 |

截至 C++17，lambda 表达式的一般形式如下：

```cpp
    capture-list mutable constexpr exception attr -> ret
    { body }
```

此语法中显示的所有部分实际上都是可选的，除了捕获列表，但是可以为空，并且主体也可以为空。如果不需要参数，则可以省略参数列表。不需要指定返回类型，因为编译器可以从返回表达式的类型推断出来。`mutable`说明符（告诉编译器 lambda 实际上可以修改通过复制捕获的变量），`constexpr`说明符（告诉编译器生成`constexpr`调用运算符），异常说明符和属性都是可选的。

最简单的 lambda 表达式是`[]{}`，尽管通常写作`[](){}`。

# 还有更多...

有时 lambda 表达式只在其参数的类型上有所不同。在这种情况下，lambda 可以以通用的方式编写，就像模板一样，但是使用`auto`说明符作为类型参数（不涉及模板语法）。这在下一个配方中讨论，见*另请参阅*部分。

# 另请参阅

+   *使用通用 lambda*

+   *编写递归 lambda*

# 使用通用 lambda：

在前面的文章中，我们看到了如何编写 lambda 表达式并将其与标准算法一起使用。在 C++中，lambda 基本上是未命名函数对象的语法糖，这些函数对象是实现调用运算符的类。然而，就像任何其他函数一样，这可以通过模板来实现。C++14 利用了这一点，并引入了通用 lambda，它们不需要为参数指定实际类型，而是使用`auto`关键字。虽然没有用这个名字，通用 lambda 基本上就是 lambda 模板。它们在我们想要使用相同 lambda 但参数类型不同的情况下非常有用。

# 入门

建议在继续阅读本文之前，先阅读前一篇文章《使用 lambda 与标准算法》。

# 操作步骤如下：

编写通用 lambda：

+   使用`auto`关键字而不是实际类型来定义 lambda 表达式的参数。

+   当需要使用多个 lambda，它们之间只有参数类型不同。

以下示例展示了一个通用 lambda 首先与整数向量一起使用`std::accumulate()`算法，然后与字符串向量一起使用。

```cpp
        auto numbers =
          std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};  
        auto texts =  
          std::vector<std::string>{"hello"s, " "s, "world"s, "!"s}; 

        auto lsum = [](auto const s, auto const n) {return s + n;}; 

        auto sum = std::accumulate( 
          std::begin(numbers), std::end(numbers), 0, lsum); 
          // sum = 22 

        auto text = std::accumulate( 
          std::begin(texts), std::end(texts), ""s, lsum); 
          // sum = "hello world!"s
```

# 工作原理：

在前一节的示例中，我们定义了一个命名的 lambda 表达式，也就是说，一个具有其闭包分配给变量的 lambda 表达式。然后将这个变量作为参数传递给`std::accumulate()`函数。这个通用算法接受定义范围的开始和结束迭代器，一个初始值进行累积，并一个函数，该函数应该将范围内的每个值累积到总和中。这个函数接受一个表示当前累积值的第一个参数和一个表示要累积到总和中的当前值的第二个参数，并返回新的累积值。

请注意，我没有使用术语`add`，因为它不仅仅用于加法。它也可以用于计算乘积、连接或其他将值聚合在一起的操作。

在这个例子中，两次调用`std::accumulate()`几乎相同，只是参数的类型不同：

+   在第一个调用中，我们传递整数范围的迭代器（来自`vector<int>`），初始和为 0，并传递一个将两个整数相加并返回它们的和的 lambda。这将产生范围内所有整数的和；在这个例子中，结果是 22。

+   在第二次调用中，我们传递字符串范围的迭代器（来自`vector<string>`），一个空字符串作为初始值，并传递一个将两个字符串连接在一起并返回结果的 lambda。这将产生一个包含范围内所有字符串的字符串，这个例子中结果是"hello world!"。

虽然通用 lambda 可以在调用它们的地方匿名定义，但这实际上没有意义，因为通用 lambda（基本上就是前面提到的 lambda 表达式模板）的目的是被重用，就像在*操作步骤如下*部分的示例中所示的那样。

在定义用于多次调用`std::accumulate()`的 lambda 表达式时，我们使用了`auto`关键字而不是具体类型来指定 lambda 参数（比如`int`或`std::string`），让编译器推断类型。当遇到 lambda 表达式的参数类型带有`auto`关键字时，编译器会生成一个没有名字的函数对象，该对象具有调用运算符模板。在这个例子中，通用 lambda 表达式的函数对象如下：

```cpp
    struct __lambda_name__ 
    { 
      template<typename T1, typename T2> 
      auto operator()(T1 const s, T2 const n) const { return s + n; } 

       __lambda_name__(const __lambda_name__&) = default; 
       __lambda_name__(__lambda_name__&&) = default; 
       __lambda_name__& operator=(const __lambda_name__&) = delete; 
       ~__lambda_name__() = default; 
    };
```

调用运算符是一个模板，对于 lambda 中使用`auto`指定的每个参数，都有一个类型参数。调用运算符的返回类型也是`auto`，这意味着编译器将从返回值的类型中推断出它。这个操作符模板将使用编译器在使用通用 lambda 的上下文中识别的实际类型进行实例化。

# 另请参阅

+   *使用标准算法与 lambda*

+   *尽可能使用 auto* 第八章 的配方，*学习现代核心语言特性*

# 编写递归 lambda

Lambda 基本上是无名函数对象，这意味着应该可以递归调用它们。事实上，它们可以被递归调用；但是，这样做的机制并不明显，因为它需要将 lambda 分配给函数包装器，并通过引用捕获包装器。虽然可以说递归 lambda 实际上并没有太多意义，函数可能是更好的设计选择，但在这个配方中，我们将看看如何编写递归 lambda。

# 准备工作

为了演示如何编写递归 lambda，我们将考虑著名的斐波那契函数的例子。在 C++中通常以递归方式实现如下：

```cpp
    constexpr int fib(int const n) 
    { 
      return n <= 2 ? 1 : fib(n - 1) + fib(n - 2); 
    }
```

# 如何做...

为了编写递归 lambda 函数，您必须执行以下操作：

+   在函数范围内定义 lambda。

+   将 lambda 分配给`std::function`包装器。

+   通过引用在 lambda 中捕获`std::function`对象，以便递归调用它。

以下是递归 lambda 的示例：

+   在从定义它的范围调用的函数范围内的递归斐波那契 lambda 表达式：

```cpp
        void sample() 
        { 
          std::function<int(int const)> lfib =  
            &lfib 
            { 
              return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2); 
            }; 

          auto f10 = lfib(10); 
        }
```

+   通过函数返回的递归斐波那契 lambda 表达式，可以从任何范围调用：

```cpp
        std::function<int(int const)> fib_create() 
        { 
          std::function<int(int const)> f = [](int const n)  
          { 
            std::function<int(int const)> lfib = &lfib 
            { 
              return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2); 
            }; 
            return lfib(n); 
          }; 
          return f; 
        } 

        void sample() 
        { 
          auto lfib = fib_create(); 
          auto f10 = lfib(10); 
        }
```

# 它是如何工作的...

编写递归 lambda 时需要考虑的第一件事是，lambda 表达式是一个函数对象，为了从 lambda 的主体递归调用它，lambda 必须捕获其闭包（即 lambda 的实例化）。换句话说，lambda 必须捕获自身，这有几个含义：

+   首先，lambda 必须有一个名称；无名 lambda 不能被捕获以便再次调用。

+   其次，lambda 只能在函数范围内定义。原因是 lambda 只能捕获函数范围内的变量；它不能捕获任何具有静态存储期的变量。在命名空间范围内或使用 static 或 external 说明符定义的对象具有静态存储期。如果 lambda 在命名空间范围内定义，它的闭包将具有静态存储期，因此 lambda 将无法捕获它。

+   第三个含义是 lambda 闭包的类型不能保持未指定，也就是说，不能使用 auto 说明符声明它。因为在处理初始化程序时，变量的类型是未知的，所以无法使用 auto 类型说明符声明的变量出现在自己的初始化程序中。因此，您必须指定 lambda 闭包的类型。我们可以使用通用目的的函数包装器`std::function`来做到这一点。

+   最后但并非最不重要的是，lambda 闭包必须通过引用捕获。如果我们通过复制（或值）捕获，那么将会创建函数包装器的副本，但是当捕获完成时，包装器将未初始化。我们最终得到一个无法调用的对象。尽管编译器不会抱怨通过值捕获，但当调用闭包时，会抛出`std::bad_function_call`。

在*如何做...*部分的第一个示例中，递归 lambda 是在另一个名为`sample()`的函数内部定义的。lambda 表达式的签名和主体与介绍部分中定义的常规递归函数`fib()`的相同。lambda 闭包被分配给一个名为`lfib`的函数包装器，然后被 lambda 引用并从其主体递归调用。由于闭包被引用捕获，它将在必须从 lambda 的主体中调用时初始化。

在第二个示例中，我们定义了一个函数，该函数返回一个 lambda 表达式的闭包，该闭包又定义并调用了一个递归 lambda，并使用它被调用的参数。当需要从函数返回递归 lambda 时，必须实现这种模式。这是必要的，因为在递归 lambda 被调用时，lambda 闭包仍然必须可用。如果在那之前它被销毁，我们将得到一个悬空引用，并且调用它将导致程序异常终止。这种错误的情况在以下示例中得到了说明：

```cpp
    // this implementation of fib_create is faulty
    std::function<int(int const)> fib_create() 
    { 
      std::function<int(int const)> lfib = &lfib 
      { 
        return n <= 2 ? 1 : lfib(n - 1) + lfib(n - 2); 
      }; 

      return lfib; 
    } 

    void sample() 
    { 
      auto lfib = fib_create();
      auto f10 = lfib(10);       // crash 
    }
```

解决方案是在*如何做...*部分中创建两个嵌套的 lambda 表达式。`fib_create()`方法返回一个函数包装器，当调用时创建捕获自身的递归 lambda。这与前面示例中的实现略有不同，但基本上是不同的。外部的`f` lambda 不捕获任何东西，特别是不捕获引用；因此，我们不会遇到悬空引用的问题。然而，当调用时，它创建了嵌套 lambda 的闭包，我们感兴趣的实际 lambda，并返回将递归的`lfib` lambda 应用于其参数的结果。

# 编写具有可变数量参数的函数模板

有时编写具有可变数量参数的函数或具有可变数量成员的类是很有用的。典型的例子包括`printf`这样的函数，它接受格式和可变数量的参数，或者`tuple`这样的类。在 C++11 之前，前者只能通过使用可变宏（只能编写不安全类型的函数）实现，而后者根本不可能。C++11 引入了可变模板，这是具有可变数量参数的模板，可以编写具有可变数量参数的类型安全函数模板，也可以编写具有可变数量成员的类模板。在本示例中，我们将看看如何编写函数模板。

# 准备工作

具有可变数量参数的函数称为*可变函数*。具有可变数量参数的函数模板称为*可变函数模板*。学习如何编写可变函数模板并不需要了解 C++可变宏（`va_start`、`va_end`、`va_arg`和`va_copy`、`va_list`），但它代表了一个很好的起点。

我们已经在之前的示例中使用了可变模板，但这个示例将提供详细的解释。

# 如何做...

要编写可变函数模板，必须执行以下步骤：

1.  如果可变函数模板的语义要求，可以定义一个带有固定数量参数的重载来结束编译时递归（参见以下代码中的`[1]`）。

1.  定义一个模板参数包，引入一个可以容纳任意数量参数的模板参数，包括零个；这些参数可以是类型、非类型或模板（参见`[2]`）。

1.  定义一个函数参数包，用于保存任意数量的函数参数，包括零个；模板参数包的大小和相应的函数参数包的大小相同，并且可以使用`sizeof...`运算符确定（参见`[3]`）。

1.  扩展参数包，以替换为提供的实际参数（参考`[4]`）。

以下示例说明了所有前面的观点，是一个可变参数函数模板，它使用`operator+`来添加可变数量的参数：

```cpp
    template <typename T>                 // [1] overload with fixed 
    T add(T value)                        //     number of arguments 
    { 
      return value; 
    } 

    template <typename T, typename... Ts> // [2] typename... Ts 
    T add(T head, Ts... rest)             // [3] Ts... rest 
    { 
      return head + add(rest...);         // [4] rest...  
    }
```

# 它是如何工作的...

乍一看，前面的实现看起来像是递归，因为函数`add()`调用了自身，从某种意义上来说确实是，但它是一种不会产生任何运行时递归和开销的编译时递归。编译器实际上会生成几个具有不同参数数量的函数，基于可变参数函数模板的使用，因此只涉及函数重载，而不涉及任何递归。然而，实现是按照参数会以递归方式处理并具有结束条件的方式进行的。

在前面的代码中，我们可以识别出以下关键部分：

+   `Typename... Ts`是指示可变数量模板类型参数的模板参数包。

+   `Ts... rest`是指示可变数量函数参数的函数参数包。

+   `Rest...`是函数参数包的扩展。

省略号的位置在语法上并不重要。`typename... Ts`，`typename ... Ts`和`typename ...Ts`都是等效的。

在`add(T head, Ts... rest)`参数中，`head`是参数列表的第一个元素，`...rest`是列表中其余参数的包（可以是零个或多个）。在函数的主体中，`rest...`是函数参数包的扩展。这意味着编译器会用它们的顺序替换参数包中的元素。在`add()`函数中，我们基本上将第一个参数添加到其余参数的总和中，这给人一种递归处理的印象。当只剩下一个参数时，递归就会结束，在这种情况下，将调用第一个`add()`重载（带有单个参数）并返回其参数的值。

这个函数模板`add()`的实现使我们能够编写如下代码：

```cpp
    auto s1 = add(1, 2, 3, 4, 5);  
    // s1 = 15 
    auto s2 = add("hello"s, " "s, "world"s, "!"s);  
    // s2 = "hello world!"
```

当编译器遇到`add(1, 2, 3, 4, 5)`时，它会生成以下函数（`arg1`，`arg2`等等，并不是编译器生成的实际名称），显示这实际上只是对重载函数的调用，而不是递归：

```cpp
    int add(int head, int arg1, int arg2, int arg3, int arg4)  
    {return head + add(arg1, arg2, arg3, arg4);} 
    int add(int head, int arg1, int arg2, int arg3)  
    {return head + add(arg1, arg2, arg3);} 
    int add(int head, int arg1, int arg2)  
    {return head + add(arg1, arg2);} 
    int add(int head, int arg1)  
    {return head + add(arg1);} 
    int add(int value)  
    {return value;}
```

使用 GCC 和 Clang，您可以使用`__PRETTY_FUNCTION__`宏来打印函数的名称和签名。

通过在我们编写的两个函数的开头添加`std::cout << __PRETTY_FUNCTION__ << std::endl`，在运行代码时我们得到以下结果：

```cpp
    T add(T, Ts ...) [with T = int; Ts = {int, int, int, int}] 
    T add(T, Ts ...) [with T = int; Ts = {int, int, int}] 
    T add(T, Ts ...) [with T = int; Ts = {int, int}] 
    T add(T, Ts ...) [with T = int; Ts = {int}] 
    T add(T) [with T = int]
```

由于这是一个函数模板，它可以与支持`operator+`的任何类型一起使用。另一个例子，`add("hello"s, " "s, "world"s, "!"s)`，产生了字符串`"hello world!"`。然而，`std::basic_string`类型有不同的`operator+`重载，包括一个可以将字符串连接到字符的重载，因此我们应该也能够编写以下内容：

```cpp
    auto s3 = add("hello"s, ' ', "world"s, '!');  
    // s3 = "hello world!"
```

然而，这将生成如下的编译器错误（请注意，我实际上用字符串“hello world”替换了`std::basic_string<char, std::char_traits<char>, std::allocator<char> >`以简化）：

```cpp
In instantiation of 'T add(T, Ts ...) [with T = char; Ts = {string, char}]': 
16:29:   required from 'T add(T, Ts ...) [with T = string; Ts = {char, string, char}]' 
22:46:   required from here 
16:29: error: cannot convert 'string' to 'char' in return 
 In function 'T add(T, Ts ...) [with T = char; Ts = {string, char}]': 
17:1: warning: control reaches end of non-void function [-Wreturn-type]
```

发生的情况是，编译器生成了下面显示的代码，其中返回类型与第一个参数的类型相同。然而，第一个参数是`std::string`或`char`（再次，`std::basic_string<char, std::char_traits<char>, std::allocator<char> >`被替换为`string`以简化）。在第一个参数的类型为`char`的情况下，返回值的类型`head+add(...)`是`std::string`，它与函数返回类型不匹配，并且没有隐式转换为它：

```cpp
    string add(string head, char arg1, string arg2, char arg3)  
    {return head + add(arg1, arg2, arg3);} 
    char add(char head, string arg1, char arg2)  
    {return head + add(arg1, arg2);} 
    string add(string head, char arg1)  
    {return head + add(arg1);} 
    char add(char value)  
    {return value;}
```

我们可以通过修改可变参数函数模板，将返回类型改为`auto`而不是`T`来解决这个问题。在这种情况下，返回类型总是从返回表达式中推断出来，在我们的例子中，它将始终是`std::string`。

```cpp
    template <typename T, typename... Ts> 
    auto add(T head, Ts... rest) 
    { 
      return head + add(rest...); 
    }
```

还应该进一步补充的是，参数包可以出现在大括号初始化中，并且可以使用`sizeof...`运算符确定其大小。此外，可变函数模板并不一定意味着编译时递归，正如我们在本配方中所示的那样。所有这些都在以下示例中展示，其中我们定义了一个创建具有偶数成员的元组的函数。我们首先使用`sizeof...(a)`来确保我们有偶数个参数，并通过生成编译器错误来断言否则。`sizeof...`运算符既可以用于模板参数包，也可以用于函数参数包。`sizeof...(a)`和`sizeof...(T)`将产生相同的值。然后，我们创建并返回一个元组。模板参数包`T`被展开（使用`T...`）为`std::tuple`类模板的类型参数，并且函数参数包`a`被展开（使用`a...`）为元组成员的值，使用大括号初始化：

```cpp
    template<typename... T> 
    auto make_even_tuple(T... a) 
    { 
      static_assert(sizeof...(a) % 2 == 0,  
                    "expected an even number of arguments"); 
      std::tuple<T...> t { a... }; 

      return t; 
    } 

    auto t1 = make_even_tuple(1, 2, 3, 4); // OK 

    // error: expected an even number of arguments 
    auto t2 = make_even_tuple(1, 2, 3);
```

# 另请参阅

+   *使用折叠表达式简化可变函数模板*

+   *在第九章的*创建原始用户定义字面量*配方中，*使用数字和*

*字符串*

# 使用折叠表达式简化可变函数模板

在本章中，我们多次讨论了折叠；这是一种将二元函数应用于一系列值以产生单个值的操作。我们在讨论可变函数模板时已经看到了这一点，并且将在高阶函数中再次看到。事实证明，在编写可变函数模板中参数包的展开基本上是一种折叠操作的情况相当多。为了简化编写这样的可变函数模板，C++17 引入了折叠表达式，它将参数包的展开折叠到二元运算符上。在本配方中，我们将看到如何使用折叠表达式来简化编写可变函数模板。

# 准备工作

本配方中的示例基于我们在上一个配方*编写具有可变数量参数的函数模板*中编写的可变函数模板`add()`。该实现是一个左折叠操作。为简单起见，我们再次呈现该函数：

```cpp
    template <typename T> 
    T add(T value) 
    { 
      return value; 
    } 

    template <typename T, typename... Ts> 
    T add(T head, Ts... rest) 
    { 
      return head + add(rest...); 
    }
```

# 如何做...

要在二元运算符上折叠参数包，请使用以下形式之一：

+   一元形式的左折叠`(... op pack)`：

```cpp
        template <typename... Ts> 
        auto add(Ts... args) 
        { 
          return (... + args); 
        }
```

+   二元形式的左折叠`(init op ... op pack)`：

```cpp
        template <typename... Ts> 
        auto add_to_one(Ts... args) 
        { 
          return (1 + ... + args); 
        }
```

+   一元形式的右折叠`(pack op ...)`：

```cpp
        template <typename... Ts> 
        auto add(Ts... args) 
        { 
          return (args + ...); 
        }
```

+   一元形式的右折叠`(pack op ... op init)`：

```cpp
        template <typename... Ts> 
        auto add_to_one(Ts... args) 
        { 
          return (args + ... + 1); 
        }
```

上面显示的括号是折叠表达式的一部分，不能省略。

# 它是如何工作的...

当编译器遇到折叠表达式时，它会将其扩展为以下表达式之一：

| **表达式** | **展开** |
| --- | --- |
| `(... op pack)` | ((pack$1 op pack$2) op ...) op pack$n |
| `(init op ... op pack)` | (((init op pack$1) op pack$2) op ...) op pack$n |
| `(pack op ...)` | pack$1 op (... op (pack$n-1 op pack$n)) |
| `(pack op ... op init)` | pack$1 op (... op (pack$n-1 op (pack$n op init))) |

当使用二元形式时，省略号的左右两侧的运算符必须相同，并且初始化值不能包含未展开的参数包。

以下二元运算符支持折叠表达式：

| 加 | 减 | 乘 | 除 | 取余 | 指数 | 与 | 或 | 等于 | 小于 | 大于 | 左移 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >> | += | -= | *= | /= | %= | ^= | &= | &#124;= | <<= | >>= | == |
| != | <= | >= | && | &#124;&#124; | , | .* | ->*. |  |  |  |  |

在使用一元形式时，只允许使用诸如`*`，`+`，`&`，`|`，`&&`，`||`和`,`（逗号）等运算符与空参数包一起。在这种情况下，空包的值如下：

| `+` | `0` |
| --- | --- |
| `*` | `1` |
| `&` | `-1` |
| `&#124;` | `0` |
| `&&` | `true` |
| `&#124;&#124;` | `false` |
| `,` | `void()` |

现在我们已经实现了之前的函数模板（让我们考虑左折叠版本），我们可以编写以下代码：

```cpp
    auto sum = add(1, 2, 3, 4, 5);         // sum = 15 
    auto sum1 = add_to_one(1, 2, 3, 4, 5); // sum = 16
```

考虑到`add(1, 2, 3, 4, 5)`的调用，它将产生以下函数：

```cpp
    int add(int arg1, int arg2, int arg3, int arg4, int arg5) 
    { 
      return ((((arg1 + arg2) + arg3) + arg4) + arg5); 
    }
```

由于现代编译器进行优化的激进方式，这个函数可以被内联，最终得到一个表达式，如`auto sum = 1 + 2 + 3 + 4 + 5`。

# 还有更多...

Fold 表达式适用于所有支持的二元运算符的重载，但不适用于任意的二元函数。可以通过提供一个包装类型来实现对此的解决方法，以保存一个值和一个重载的运算符来实现：

```cpp
    template <typename T> 
    struct wrapper 
    { 
      T const & value; 
    }; 

    template <typename T> 
    constexpr auto operator<(wrapper<T> const & lhs,  
                             wrapper<T> const & rhs)  
    { 
      return wrapper<T> { 
        lhs.value < rhs.value ? lhs.value : rhs.value}; 
    } 

    template <typename... Ts> 
    constexpr auto min(Ts&&... args)  
    { 
      return (wrapper<Ts>{args} < ...).value; 
    }
```

在前面的代码中，`wrapper`是一个简单的类模板，它保存了类型为`T`的值的常量引用。为这个类模板提供了重载的`operator<`；这个重载并不返回一个布尔值来指示第一个参数是否小于第二个参数，而是实际上返回`wrapper`类类型的一个实例，以保存这两个参数的最小值。可变函数模板`min()`使用这个重载的`operator<`来将展开为`wrapper`类模板实例的参数包进行折叠：

```cpp
    auto m = min(1, 2, 3, 4, 5); // m = 1
```

# 另请参阅

+   *实现高阶函数 map 和 fold*

# 实现高阶函数 map 和 fold

在本书的前面几个示例中，我们使用了通用算法`std::transform()`和`std::accumulate()`，例如实现字符串工具来创建字符串的大写或小写副本，或者对范围的值进行求和。这些基本上是高阶函数`map`和`fold`的实现。高阶函数是一个接受一个或多个其他函数作为参数并将它们应用于范围（列表、向量、映射、树等）的函数，产生一个新的范围或值。在这个示例中，我们将看到如何实现`map`和`fold`函数来处理 C++标准容器。

# 准备工作

*Map*是一个高阶函数，它将一个函数应用于范围的元素，并按相同的顺序返回一个新的范围。

*Fold*是一个高阶函数，它将一个组合函数应用于范围的元素，产生一个单一的结果。由于处理的顺序可能很重要，通常有两个版本的这个函数--`foldleft`，从左到右处理元素，和**`foldright`**，从右到左组合元素。

大多数对 map 函数的描述表明它适用于`list`，但这是一个通用术语，可以表示不同的顺序类型，如列表、向量和数组，还有字典（即映射）、队列等。因此，我更喜欢在描述这些高阶函数时使用术语范围。

# 如何做...

要实现`map`函数，您应该：

+   在支持迭代和对元素进行赋值的容器上使用`std::transform`，如`std::vector`或`std::list`：

```cpp
        template <typename F, typename R> 
        R mapf(F&& f, R r) 
        { 
          std::transform( 
            std::begin(r), std::end(r), std::begin(r),  
            std::forward<F>(f)); 
          return r; 
        }
```

+   对于不支持对元素进行赋值的容器，如`std::map`，请使用显式迭代和插入等其他方法：

```cpp
        template<typename F, typename T, typename U> 
        std::map<T, U> mapf(F&& f, std::map<T, U> const & m) 
        { 
          std::map<T, U> r; 
          for (auto const kvp : m) 
            r.insert(f(kvp)); 
          return r; 
        } 

        template<typename F, typename T> 
        std::queue<T> mapf(F&& f, std::queue<T> q) 
        { 
          std::queue<T> r; 
          while (!q.empty()) 
          { 
            r.push(f(q.front())); 
            q.pop(); 
          } 
          return r; 
        }
```

要实现`fold`函数，您应该：

+   在支持迭代的容器上使用`std::accumulate()`：

```cpp
        template <typename F, typename R, typename T> 
        constexpr T foldl(F&& f, R&& r, T i) 
        { 
          return std::accumulate( 
            std::begin(r), std::end(r),  
            std::move(i),  
            std::forward<F>(f)); 
        } 

        template <typename F, typename R, typename T> 
        constexpr T foldr(F&& f, R&& r, T i) 
        { 
          return std::accumulate( 
            std::rbegin(r), std::rend(r),  
            std::move(i),  
            std::forward<F>(f)); 
        }
```

+   使用其他方法显式处理不支持迭代的容器，如`std::queue`：

```cpp
        template <typename F, typename T> 
        constexpr T foldl(F&& f, std::queue<T> q, T i) 
        { 
          while (!q.empty()) 
          { 
            i = f(i, q.front()); 
            q.pop(); 
          } 
          return i; 
        }
```

# 它是如何工作的...

在前面的示例中，我们以一种功能方式实现了 map，没有副作用。这意味着它保留了原始范围并返回了一个新的范围。函数的参数是要应用的函数和范围。为了避免与`std::map`容器混淆，我们将这个函数称为`mapf`。有几个`mapf`的重载，如前面所示：

+   第一个重载适用于支持迭代和对其元素赋值的容器；这包括`std::vector`、`std::list`和`std::array`，还有类似 C 的数组。该函数接受一个对函数的`rvalue`引用和一个范围，其中`std::begin()`和`std::end()`被定义。范围通过值传递，这样修改本地副本不会影响原始范围。通过应用给定函数对每个元素使用标准算法`std::transform()`来转换范围；然后返回转换后的范围。

+   第二个重载专门针对不支持直接赋值给其元素（`std::pair<T, U>`）的`std::map`。因此，这个重载创建一个新的映射，然后使用基于范围的 for 循环遍历其元素，并将应用输入函数的结果插入到新映射中。

+   第三个重载专门针对`std::queue`，这是一个不支持迭代的容器。可以说队列不是一个典型的映射结构，但为了演示不同的可能实现，我们考虑它。为了遍历队列的元素，必须改变队列--需要从前面弹出元素，直到列表为空。这就是第三个重载所做的--它处理输入队列的每个元素（通过值传递），并将应用给定函数的结果推送到剩余队列的前端元素。

现在我们已经实现了这些重载，我们可以将它们应用到许多容器中，如下面的例子所示（请注意，这里使用的 map 和 fold 函数在附带书籍的代码中实现在名为 funclib 的命名空间中，因此显示为完全限定名称）：

+   保留向量中的绝对值。在这个例子中，向量包含负值和正值。应用映射后，结果是一个只包含正值的新向量。

```cpp
        auto vnums =  
          std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};  
        auto r = funclib::mapf([](int const i) { 
          return std::abs(i); }, vnums);  
        // r = {0, 2, 3, 5, 1, 6, 8, 4, 9}
```

+   对列表中的数值进行平方。在这个例子中，列表包含整数值。应用映射后，结果是一个包含初始值的平方的列表。

```cpp
        auto lnums = std::list<int>{1, 2, 3, 4, 5}; 
        auto l = funclib::mapf([](int const i) { 
          return i*i; }, lnums); 
        // l = {1, 4, 9, 16, 25}
```

+   浮点数的四舍五入金额。在这个例子中，我们需要使用`std::round()`；然而，这个函数对所有浮点类型都有重载，这使得编译器无法选择正确的重载。因此，我们要么编写一个接受特定浮点类型参数并返回应用于该值的`std::round()`值的 lambda，要么创建一个函数对象模板，包装`std::round()`并仅对浮点类型启用其调用运算符。这种技术在下面的例子中使用：

```cpp
        template<class T = double> 
        struct fround 
        {   
          typename std::enable_if< 
            std::is_floating_point<T>::value, T>::type 
          operator()(const T& value) const 
          { 
            return std::round(value); 
          } 
        }; 

        auto amounts =  
          std::array<double, 5> {10.42, 2.50, 100.0, 23.75, 12.99}; 
        auto a = funclib::mapf(fround<>(), amounts); 
        // a = {10.0, 3.0, 100.0, 24.0, 13.0}
```

+   将单词映射的地图键大写（其中键是单词，值是在文本中出现的次数）。请注意，创建字符串的大写副本本身就是一个映射操作。因此，在这个例子中，我们使用`mapf`将`toupper()`应用于表示键的字符串的元素，以产生一个大写副本。

```cpp
        auto words = std::map<std::string, int>{  
          {"one", 1}, {"two", 2}, {"three", 3}  
        }; 
        auto m = funclib::mapf( 
          [](std::pair<std::string, int> const kvp) { 
            return std::make_pair( 
              funclib::mapf(toupper, kvp.first),  
              kvp.second); 
          }, 
          words); 
        // m = {{"ONE", 1}, {"TWO", 2}, {"THREE", 3}}
```

+   从优先级队列中规范化数值--最初，数值范围是 1 到 100，但我们希望将它们规范化为两个值，1=高和 2=正常。所有初始优先级的值最多为 30 的变为高优先级，其他的变为正常优先级：

```cpp
        auto priorities = std::queue<int>(); 
        priorities.push(10); 
        priorities.push(20); 
        priorities.push(30); 
        priorities.push(40); 
        priorities.push(50); 
        auto p = funclib::mapf( 
          [](int const i) { return i > 30 ? 2 : 1; },  
          priorities); 
        // p = {1, 1, 1, 2, 2}
```

要实现`fold`，我们实际上必须考虑两种可能的折叠类型，即从左到右和从右到左。因此，我们提供了两个名为`foldl`（用于左折叠）和`foldr`（用于右折叠）的函数。在前一节中显示的实现非常相似--它们都接受一个函数、一个范围和一个初始值，并调用`std::algorithm()`将范围的值折叠成一个值。然而，`foldl`使用直接迭代器，而`foldr`使用反向迭代器来遍历和处理范围。第二个重载是`std::queue`类型的特化，它没有迭代器。

基于这些折叠实现，我们可以进行以下示例：

+   添加整数向量的值。在这种情况下，左折叠和右折叠将产生相同的结果。在以下示例中，我们传递一个 lambda，它接受一个和一个数字并返回一个新的和，或者从标准库中使用`std::plus<>`函数对象，它将`operator+`应用于相同类型的两个操作数（基本上类似于 lambda 的闭包）：

```cpp
        auto vnums =  
           std::vector<int>{0, 2, -3, 5, -1, 6, 8, -4, 9};  

        auto s1 = funclib::foldl( 
           [](const int s, const int n) {return s + n; },  
           vnums, 0);                // s1 = 22 

        auto s2 = funclib::foldl( 
           std::plus<>(), vnums, 0); // s2 = 22 

        auto s3 = funclib::foldr( 
           [](const int s, const int n) {return s + n; },  
           vnums, 0);                // s3 = 22 

        auto s4 = funclib::foldr( 
           std::plus<>(), vnums, 0); // s4 = 22
```

+   将字符串从向量连接成一个字符串：

```cpp
        auto texts =  
           std::vector<std::string>{"hello"s, " "s, "world"s, "!"s}; 

        auto txt1 = funclib::foldl( 
           [](std::string const & s, std::string const & n) { 
           return s + n;},  
           texts, ""s);    // txt1 = "hello world!" 

        auto txt2 = funclib::foldr( 
           [](std::string const & s, std::string const & n) { 
           return s + n; },  
           texts, ""s);    // txt2 = "!world hello"
```

+   将字符数组连接成一个字符串：

```cpp
        char chars[] = {'c','i','v','i','c'}; 

        auto str1 = funclib::foldl(std::plus<>(), chars, ""s);  
        // str1 = "civic" 

        auto str2 = funclib::foldr(std::plus<>(), chars, ""s);  
        // str2 = "civic"
```

+   根据`map<string, int>`中已计算出现次数的单词数量来计算文本中单词的数量：

```cpp
        auto words = std::map<std::string, int>{  
           {"one", 1}, {"two", 2}, {"three", 3} }; 

        auto count = funclib::foldl( 
           [](int const s, std::pair<std::string, int> const kvp) { 
              return s + kvp.second; }, 
           words, 0); // count = 6
```

# 还有更多...

这些函数可以被串联，也就是说，它们可以用另一个函数调用另一个函数的结果。以下示例将整数范围映射为正整数范围，方法是将`std::abs()`函数应用于其元素。然后将结果映射到另一个平方范围。然后通过在范围上应用左折叠将它们相加：

```cpp
    auto vnums = std::vector<int>{ 0, 2, -3, 5, -1, 6, 8, -4, 9 }; 

    auto s = funclib::foldl( 
      std::plus<>(), 
      funclib::mapf( 
        [](int const i) {return i*i; },  
        funclib::mapf( 
          [](int const i) {return std::abs(i); }, 
          vnums)), 
      0); // s = 236
```

作为练习，我们可以按照前面配方中所见的方式，将 fold 函数实现为一个可变参数函数模板。执行实际折叠的函数作为参数提供：

```cpp
    template <typename F, typename T1, typename T2> 
    auto foldl(F&&f, T1 arg1, T2 arg2) 
    { 
      return f(arg1, arg2); 
    } 

    template <typename F, typename T, typename... Ts> 
    auto foldl(F&& f, T head, Ts... rest) 
    { 
      return f(head, foldl(std::forward<F>(f), rest...)); 
    }
```

当我们将这与我们在配方*编写具有可变数量参数的函数模板*中编写的`add()`函数模板进行比较时，我们可以注意到几个不同之处：

+   第一个参数是一个函数，在递归调用`foldl`时可以完全转发。

+   结束情况是一个需要两个参数的函数，因为我们用于折叠的函数是一个二元函数（接受两个参数）。

+   我们编写的两个函数的返回类型声明为`auto`，因为它必须匹配提供的二元函数`f`的返回类型，直到我们调用`foldl`为止，这是不知道的：

```cpp
    auto s1 = foldl(std::plus<>(), 1, 2, 3, 4, 5);  
    // s1 = 15 
    auto s2 = foldl(std::plus<>(), "hello"s, ' ', "world"s, '!');  
    // s2 = "hello world!" 
    auto s3 = foldl(std::plus<>(), 1); // error, too few arguments
```

# 参见

+   *创建字符串助手库* 第九章的配方[9830e5b8-a9ca-41e8-b565-8800a82d9caa.xhtml]，*处理数字和字符串*

+   *编写具有可变数量参数的函数模板*

+   *将函数组合成高阶函数*

# 将函数组合成高阶函数

在上一个配方中，我们实现了两个高阶函数，map 和 fold，并看到了它们的各种使用示例。在配方的结尾，我们看到它们如何可以被串联起来，在对原始数据进行多次转换后产生最终值。管道是一种组合形式，意味着从两个或更多给定函数创建一个新函数。在上述示例中，我们实际上并没有组合函数；我们只是调用了一个函数，其结果由另一个函数产生，但在这个配方中，我们将看到如何将函数实际组合到一起成为一个新函数。为简单起见，我们只考虑一元函数（只接受一个参数的函数）。

# 准备工作

在继续之前，建议您阅读前一篇配方*实现高阶函数 map 和 fol*d。这不是理解本配方的必要条件，但我们将引用这里实现的 map 和 fold 函数。 

# 操作步骤

要将一元函数组合成高阶函数，您应该：

+   要组合两个函数，提供一个接受两个函数`f`和`g`作为参数并返回一个新函数（lambda）的函数，该函数返回`f(g(x))`，其中`x`是组合函数的参数：

```cpp
        template <typename F, typename G> 
        auto compose(F&& f, G&& g) 
        {  
          return = { return f(g(x)); }; 
        } 

        auto v = compose( 
          [](int const n) {return std::to_string(n); }, 
          [](int const n) {return n * n; })(-3); // v = "9"
```

+   要组合可变数量的函数，提供先前描述的函数的可变模板重载：

```cpp
        template <typename F, typename... R> 
        auto compose(F&& f, R&&... r) 
        { 
          return = { return f(compose(r...)(x)); }; 
        } 

        auto n = compose( 
          [](int const n) {return std::to_string(n); }, 
          [](int const n) {return n * n; }, 
          [](int const n) {return n + n; }, 
          [](int const n) {return std::abs(n); })(-3); // n = "36"
```

# 工作原理...

将两个一元函数组合成一个新函数相对较简单。创建一个我们在之前的示例中称为`compose()`的模板函数，它有两个参数--`f`和`g`--代表函数，并返回一个接受一个参数`x`并返回`f(g(x))`的函数。但是重要的是，`g`函数返回的值的类型与`f`函数的参数的类型相同。`compose`函数的返回值是一个闭包，即一个 lambda 的实例。

在实践中，能够组合不止两个函数是很有用的。这可以通过编写`compose()`函数的可变模板版本来实现。可变模板在*编写具有可变数量参数的函数模板*配方中有更详细的解释。可变模板意味着通过扩展参数包进行编译时递归。这个实现与`compose()`的第一个版本非常相似，只是如下：

+   它接受可变数量的函数作为参数。

+   返回的闭包使用扩展的参数包递归调用`compose()`；递归在只剩下两个函数时结束，在这种情况下，调用先前实现的重载。

即使代码看起来像是发生了递归，这并不是真正的递归。这可以称为编译时递归，但是随着每次扩展，我们会得到对另一个具有相同名称但不同数量参数的方法的调用，这并不代表递归。

现在我们已经实现了这些可变模板重载，我们可以重写上一个配方*实现高阶函数 map 和 fold*中的最后一个示例。有一个初始整数向量，我们通过对每个元素应用`std::abs()`将其映射到只有正值的新向量。然后，将结果映射到一个新向量，方法是将每个元素的值加倍。最后，将结果向量中的值通过将它们添加到初始值 0 来折叠在一起：

```cpp
    auto s = compose( 
      [](std::vector<int> const & v) { 
        return foldl(std::plus<>(), v, 0); }, 
      [](std::vector<int> const & v) { 
        return mapf([](int const i) {return i + i; }, v); }, 
      [](std::vector<int> const & v) { 
        return mapf([](int const i) {return std::abs(i); }, v); })(vnums);
```

# 还有更多...

组合通常用点（`.`）或星号（`*`）表示，比如`f . g`或`f * g`。我们实际上可以在 C++中做类似的事情，通过重载`operator*`（尝试重载操作符点没有多大意义）。与`compose()`函数类似，`operator*`应该适用于任意数量的参数；因此，我们将有两个重载，就像在`compose()`的情况下一样：

+   第一个重载接受两个参数并调用`compose()`返回一个新函数。

+   第二个重载是一个可变模板函数，再次通过扩展参数包调用`operator*`：

```cpp
    template <typename F, typename G> 
    auto operator*(F&& f, G&& g) 
    { 
      return compose(std::forward<F>(f), std::forward<G>(g)); 
    } 

    template <typename F, typename... R> 
    auto operator*(F&& f, R&&... r) 
```

```cpp
    { 
      return operator*(std::forward<F>(f), r...); 
    }
```

现在，我们可以通过应用`operator*`来简化函数的实际组合，而不是更冗长地调用 compose：

```cpp
    auto n = 
      ([](int const n) {return std::to_string(n); } * 
       [](int const n) {return n * n; } * 
       [](int const n) {return n + n; } * 
       [](int const n) {return std::abs(n); })(-3); // n = "36" 

    auto c =  
      [](std::vector<int> const & v) { 
        return foldl(std::plus<>(), v, 0); } * 
      [](std::vector<int> const & v) { 
        return mapf([](int const i) {return i + i; }, v); } * 
      [](std::vector<int> const & v) { 
        return mapf([](int const i) {return std::abs(i); }, v); }; 

    auto s = c(vnums); // s = 76
```

# 另请参阅

+   *编写具有可变数量参数的函数模板*

# 统一调用任何可调用对象

开发人员，特别是那些实现库的人，有时需要以统一的方式调用可调用对象。这可以是一个函数，一个指向函数的指针，一个指向成员函数的指针，或者一个函数对象。这种情况的例子包括`std::bind`，`std::function`，`std::mem_fn`和`std::thread::thread`。C++17 定义了一个名为`std::invoke()`的标准函数，可以使用提供的参数调用任何可调用对象。这并不意味着要取代对函数或函数对象的直接调用，但在模板元编程中实现各种库函数时非常有用。

# 准备就绪

对于这个配方，您应该熟悉如何定义和使用函数指针。

为了举例说明 `std::invoke()` 如何在不同的上下文中使用，我们将使用以下函数和类：

```cpp
    int add(int const a, int const b) 
    { 
      return a + b; 
    } 

    struct foo 
    { 
      int x = 0; 

      void increment_by(int const n) { x += n; } 
    };
```

# 如何做...

`std::invoke()` 函数是一个可变参数的函数模板，它将可调用对象作为第一个参数，并传递给调用的可变参数列表。`std::invoke()` 可以用来调用以下内容：

+   自由函数：

```cpp
        auto a1 = std::invoke(add, 1, 2);   // a1 = 3
```

+   通过函数指针调用自由函数：

```cpp
        auto a2 = std::invoke(&add, 1, 2);  // a2 = 3 
        int(*fadd)(int const, int const) = &add; 
        auto a3 = std::invoke(fadd, 1, 2);  // a3 = 3
```

+   通过成员函数指针调用成员函数：

```cpp
        foo f; 
        std::invoke(&foo::increment_by, f, 10);
```

+   数据成员：

```cpp
        foo f; 
        auto x1 = std::invoke(&foo::x, f);  // x1 = 0
```

+   函数对象：

```cpp
        foo f; 
        auto x3 = std::invoke(std::plus<>(),  
          std::invoke(&foo::x, f), 3); // x3 = 3
```

+   Lambda 表达式：

```cpp
        auto l = [](auto a, auto b) {return a + b; }; 
        auto a = std::invoke(l, 1, 2); // a = 3
```

在实践中，`std::invoke()` 应该在模板元编程中被用来调用带有任意数量参数的函数。为了举例说明这样的情况，我们提供了我们的 `std::apply()` 函数的可能实现，以及作为 C++17 标准库的一部分的一个调用函数的实现，通过将元组的成员解包成函数的参数：

```cpp
    namespace details 
    { 
      template <class F, class T, std::size_t... I> 
      auto apply(F&& f, T&& t, std::index_sequence<I...>) 
      { 
        return std::invoke( 
          std::forward<F>(f), 
          std::get<I>(std::forward<T>(t))...); 
      } 
    } 

    template <class F, class T> 
    auto apply(F&& f, T&& t) 
    { 
      return details::apply( 
        std::forward<F>(f), 
        std::forward<T>(t), 
        std::make_index_sequence< 
          std::tuple_size<std::decay_t<T>>::value> {}); 
    }
```

# 它是如何工作的...

在我们看到 `std::invoke()` 如何工作之前，让我们简要看一下不同可调用对象如何被调用。给定一个函数，显然，调用它的普遍方式是直接传递必要的参数给它。然而，我们也可以使用函数指针来调用函数。函数指针的问题在于定义指针的类型可能很麻烦。使用 `auto` 可以简化事情（如下面的代码所示），但在实践中，通常需要先定义函数指针的类型，然后定义一个对象并用正确的函数地址进行初始化。以下是几个例子：

```cpp
    // direct call 
    auto a1 = add(1, 2);    // a1 = 3 

    // call through function pointer 
    int(*fadd)(int const, int const) = &add; 
    auto a2 = fadd(1, 2);   // a2 = 3 

    auto fadd2 = &add; 
    auto a3 = fadd2(1, 2);  // a3 = 3
```

当您需要通过一个是类的实例的对象来调用类函数时，通过函数指针进行调用变得更加麻烦。定义成员函数的指针和调用它的语法并不简单：

```cpp
    foo f; 
    f.increment_by(3); 
    auto x1 = f.x;    // x1 = 3 

    void(foo::*finc)(int const) = &foo::increment_by; 
    (f.*finc)(3); 
    auto x2 = f.x;    // x2 = 6 

    auto finc2 = &foo::increment_by; 
    (f.*finc2)(3); 
    auto x3 = f.x;    // x3 = 9
```

无论这种调用看起来多么麻烦，实际问题是编写能够以统一方式调用任何这些类型的可调用对象的库组件（函数或类）。这就是实践中从标准函数（如 `std::invoke()`）中受益的地方。

`std::invoke()` 的实现细节很复杂，但它的工作原理可以用简单的术语来解释。假设调用的形式是 `invoke(f, arg1, arg2, ..., argN)`，那么考虑以下情况：

+   如果 `f` 是 `T` 类的成员函数的指针，那么调用等价于：

+   `(arg1.*f)(arg2, ..., argN)`，如果 `arg1` 是 `T` 的一个实例

+   `(arg1.get().*f)(arg2, ..., argN)`，如果 `arg1` 是 `reference_wrapper` 的一个特化

+   `((*arg1).*f)(arg2, ..., argN)`，如果是其他情况

+   如果 `f` 是 `T` 类的数据成员的指针，并且有一个参数，换句话说，调用的形式是 `invoke(f, arg1)`，那么调用等价于：

+   `arg1.*f`，如果 `arg1` 是 `T` 类的一个实例

+   `arg1.get().*f`，如果 `arg1` 是 `reference_wrapper` 的一个特化

+   `(*arg1).*f`，如果是其他情况

+   如果 `f` 是一个函数对象，那么调用等价于 `f(arg1, arg2, ..., argN)`

# 另请参阅

+   *编写一个带有可变数量参数的函数模板*
