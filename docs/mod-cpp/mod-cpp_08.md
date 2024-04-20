# 学习现代核心语言特性

本章包括以下配方：

+   尽可能使用 auto

+   创建类型别名和别名模板

+   理解统一初始化

+   理解各种形式的非静态成员初始化

+   控制和查询对象对齐

+   使用作用域枚举

+   使用 override 和 final 来定义虚方法

+   使用基于范围的 for 循环迭代范围

+   为自定义类型启用基于范围的 for 循环

+   使用显式构造函数和转换运算符来避免隐式转换

+   使用未命名命名空间代替静态全局变量

+   使用内联命名空间进行符号版本控制

+   使用结构化绑定处理多返回值

# 尽可能使用 auto

自动类型推断是现代 C++中最重要和广泛使用的特性之一。新的 C++标准使得在各种情况下使用`auto`作为类型的占位符并让编译器推断实际类型成为可能。在 C++11 中，`auto`可用于声明局部变量和带有尾随返回类型的函数的返回类型。在 C++14 中，`auto`可用于不指定尾随类型的函数的返回类型以及 lambda 表达式中的参数声明。

未来的标准版本可能会扩展`auto`的使用范围。在这些情况下使用`auto`有几个重要的好处。开发人员应该意识到这一点，并尽可能使用`auto`。Andrei Alexandrescu 提出了一个实际的术语，并由 Herb Sutter 推广--*almost always auto* (*AAA*)。

# 如何做...

考虑在以下情况下使用`auto`作为实际类型的占位符：

+   使用形式为`auto name = expression`的局部变量，当你不想承诺特定类型时：

```cpp
        auto i = 42;          // int 
        auto d = 42.5;        // double 
        auto s = "text";      // char const * 
        auto v = { 1, 2, 3 }; // std::initializer_list<int> 
```

+   使用`auto name = type-id { expression }`形式声明局部变量时，当你需要承诺特定类型时：

```cpp
        auto b  = new char[10]{ 0 };            // char* 
        auto s1 = std::string {"text"};         // std::string
        auto v1 = std::vector<int> { 1, 2, 3 }; // std::vector<int>
        auto p  = std::make_shared<int>(42);    // std::shared_ptr<int>
```

+   声明命名的 lambda 函数，形式为`auto name = lambda-expression`，除非 lambda 需要传递或返回给函数：

```cpp
        auto upper = [](char const c) {return toupper(c); };
```

+   声明 lambda 参数和返回值：

```cpp
        auto add = [](auto const a, auto const b) {return a + b;};
```

+   在不想承诺特定类型时声明函数返回类型：

```cpp
        template <typename F, typename T> 
        auto apply(F&& f, T value) 
        { 
          return f(value); 
        }
```

# 工作原理...

`auto`修饰符基本上是实际类型的占位符。使用`auto`时，编译器从以下实例中推断出实际类型：

+   从用于初始化变量的表达式的类型，当使用`auto`声明变量时。

+   从函数的尾随返回类型或返回表达式的类型，当`auto`用作函数的返回类型的占位符时。

在某些情况下，有必要承诺特定类型。例如，在前面的例子中，编译器推断`s`的类型为`char const *`。如果意图是要有一个`std::string`，那么类型必须明确指定。类似地，`v`的类型被推断为`std::initializer_list<int>`。然而，意图可能是要有一个`std::vector<int>`。在这种情况下，类型必须在赋值的右侧明确指定。

使用 auto 修饰符而不是实际类型有一些重要的好处；以下是可能最重要的一些：

+   不可能让变量未初始化。这是一个常见的错误，开发人员在声明变量时指定实际类型时会犯这个错误，但是对于需要初始化变量以推断类型的`auto`来说是不可能的。

+   使用`auto`可以确保你始终使用正确的类型，不会发生隐式转换。考虑以下示例，我们将向局部变量检索向量的大小。在第一种情况下，变量的类型是`int`，尽管`size()`方法返回`size_t`。这意味着将发生从`size_t`到`int`的隐式转换。然而，使用`auto`类型将推断出正确的类型，即`size_t`：

```cpp
        auto v = std::vector<int>{ 1, 2, 3 }; 
        int size1 = v.size();       
        // implicit conversion, possible loss of data 
        auto size2 = v.size(); 
        auto size3 = int{ v.size() };  // ill-formed (warning in gcc/clang, error in VC++)
```

+   使用`auto`有助于促进良好的面向对象的实践，比如更喜欢接口而不是实现。指定的类型越少，代码就越通用，更容易进行未来的更改，这是面向对象编程的基本原则。

+   这意味着更少的输入和更少关心我们根本不关心的实际类型。经常情况下，即使我们明确指定了类型，我们实际上并不关心它。迭代器就是一个非常常见的例子，但还有很多其他情况。当你想要遍历一个范围时，你并不关心迭代器的实际类型。你只对迭代器本身感兴趣；因此，使用`auto`可以节省输入可能很长的名称所用的时间，并帮助你专注于实际的代码而不是类型名称。在下面的例子中，在第一个`for`循环中，我们明确使用了迭代器的类型。这是很多文本需要输入，长语句实际上可能使代码不太可读，而且你还需要知道你实际上并不关心的类型名称。而使用`auto`关键字的第二个循环看起来更简单，可以节省你的输入和关心实际类型的时间。

```cpp
        std::map<int, std::string> m; 
        for (std::map<int,std::string>::const_iterator it = m.cbegin();
          it != m.cend(); ++it) 
        { /*...*/ } 

        for (auto it = m.cbegin(); it != m.cend(); ++it)
        { /*...*/ }
```

+   使用`auto`声明变量提供了一致的编码风格，类型总是在右侧。如果动态分配对象，你需要在赋值的左右两侧都写上类型，例如`int* p = new int(42)`。而使用`auto`，类型只在右侧指定一次。

然而，使用`auto`时有一些需要注意的地方：

+   `auto`关键字只是类型的占位符，而不是`const`/`volatile`和引用限定符的占位符。如果需要`const`/`volatile`和/或引用类型，那么需要显式指定它们。在下面的例子中，`foo.get()`返回一个`int`的引用；当变量`x`从返回值初始化时，编译器推断的类型是`int`，而不是`int&`。因此，对`x`的任何更改都不会传播到`foo.x_`。为了做到这一点，应该使用`auto&`：

```cpp
        class foo { 
          int x_; 
        public: 
          foo(int const x = 0) :x_{ x } {} 
          int& get() { return x_; } 
        }; 

        foo f(42); 
        auto x = f.get(); 
        x = 100; 
        std::cout << f.get() << std::endl; // prints 42
```

+   不可能对不可移动的类型使用`auto`：

```cpp
        auto ai = std::atomic<int>(42); // error
```

+   不可能使用`auto`来表示多个单词的类型，比如`long long`、`long double`或`struct foo`。然而，在第一种情况下，可能的解决方法是使用字面量或类型别名；至于第二种情况，使用`struct`/`class`的形式只在 C++中支持 C 兼容性，并且应该尽量避免：

```cpp
        auto l1 = long long{ 42 }; // error 
        auto l2 = llong{ 42 };     // OK 
        auto l3 = 42LL;            // OK
```

+   如果你使用`auto`关键字但仍然需要知道类型，你可以在任何 IDE 中将光标放在变量上来查看类型。然而，如果你离开 IDE，那就不可能了，唯一的方法是从初始化表达式中自己推断出类型，这可能意味着需要在代码中搜索函数返回类型。

`auto`可以用来指定函数的返回类型。在 C++11 中，这需要在函数声明中使用尾返回类型。在 C++14 中，这已经放宽，返回值的类型由编译器从`return`表达式中推断出来。如果有多个返回值，它们应该有相同的类型：

```cpp
    // C++11 
    auto func1(int const i) -> int 
    { return 2*i; } 

    // C++14 
    auto func2(int const i) 
    { return 2*i; }
```

如前所述，`auto`不保留`const`/`volatile`和引用限定符。这导致了`auto`作为函数返回类型的占位符出现问题。为了解释这一点，让我们考虑前面提到的`foo.get()`的例子。这次我们有一个名为`proxy_get()`的包装函数，它接受一个`foo`的引用，调用`get()`，并返回`get()`返回的值，即`int&`。然而，编译器会推断`proxy_get()`的返回类型为`int`，而不是`int&`。尝试将该值分配给`int&`会导致错误：

```cpp
    class foo 
    { 
      int x_; 
    public: 
      foo(int const x = 0) :x_{ x } {} 
      int& get() { return x_; } 
    }; 

    auto proxy_get(foo& f) { return f.get(); } 

    auto f = foo{ 42 }; 
    auto& x = proxy_get(f); // cannot convert from 'int' to 'int &'
```

要解决这个问题，我们需要实际返回`auto&`。然而，这是一个关于模板和完美转发返回类型的问题，而不知道这是一个值还是一个引用。在 C++14 中解决这个问题的方法是`decltype(auto)`，它将正确推断类型：

```cpp
    decltype(auto) proxy_get(foo& f) { return f.get(); } 
    auto f = foo{ 42 }; 
    decltype(auto) x = proxy_get(f);
```

`auto`可以用于 lambda 的另一个重要情况是。从 C++14 开始，lambda 的返回类型和参数类型都可以是`auto`。这样的 lambda 被称为*通用 lambda*，因为 lambda 定义的闭包类型具有模板化的调用运算符。以下是一个接受两个`auto`参数并返回应用于实际类型的`operator+`结果的通用 lambda 的示例：

```cpp
    auto ladd = [] (auto const a, auto const b) { return a + b; }; 
    struct 
    { 
       template<typename T, typename U> 
       auto operator () (T const a, U const b) const { return a+b; } 
    } L;
```

这个 lambda 可以用来添加任何定义了`operator+`的内容。在下面的例子中，我们使用 lambda 来添加两个整数并连接两个`std::string`对象（使用 C++14 用户定义的字面量`operator ""s`）：

```cpp
    auto i = ladd(40, 2);            // 42 
    auto s = ladd("forty"s, "two"s); // "fortytwo"s
```

# 另请参阅

+   *创建类型别名和别名模板*

+   *理解统一初始化*

# 创建类型别名和别名模板

在 C++中，可以创建可以代替类型名称的同义词。这是通过创建`typedef`声明实现的。这在几种情况下很有用，比如为类型创建更短或更有意义的名称，或者为函数指针创建名称。然而，`typedef`声明不能与模板一起用于创建`模板类型别名`。例如，`std::vector<T>`不是一个类型（`std::vector<int>`是一个类型），而是一种当类型占位符`T`被实际类型替换时可以创建的所有类型的一种家族。

在 C++11 中，类型别名是另一个已声明类型的名称，而别名模板是另一个已声明模板的名称。这两种类型的别名都是用新的`using`语法引入的。

# 如何做到...

+   使用以下形式创建类型别名`using identifier = type-id`，如下例所示：

```cpp
        using byte    = unsigned char; 
        using pbyte   = unsigned char *; 
        using array_t = int[10]; 
        using fn      = void(byte, double); 

        void func(byte b, double d) { /*...*/ } 

        byte b {42}; 
        pbyte pb = new byte[10] {0}; 
        array_t a{0,1,2,3,4,5,6,7,8,9}; 
        fn* f = func;
```

+   使用以下形式创建别名模板`template<template-params-list> identifier = type-id`，如下例所示：

```cpp
        template <class T> 
        class custom_allocator { /* ... */}; 

        template <typename T> 
        using vec_t = std::vector<T, custom_allocator<T>>; 

        vec_t<int>           vi; 
        vec_t<std::string>   vs; 
```

为了一致性和可读性，您应该做到以下几点：

+   不要混合使用`typedef`和`using`声明来创建别名。

+   使用`using`语法创建函数指针类型的名称。

# 它是如何工作的...

`typedef`声明引入了一个类型的同义词（或者换句话说是别名）。它不引入另一个类型（比如`class`、`struct`、`union`或`enum`声明）。使用`typedef`声明引入的类型名称遵循与标识符名称相同的隐藏规则。它们也可以被重新声明，但只能引用相同的类型（因此，只要它是同一个类型的同义词，就可以在一个翻译单元中有多个有效的`typedef`声明）。以下是`typedef`声明的典型示例：

```cpp
    typedef unsigned char   byte; 
    typedef unsigned char * pbyte; 
    typedef int             array_t[10]; 
    typedef void(*fn)(byte, double); 

    template<typename T> 
    class foo { 
      typedef T value_type; 
    }; 

    typedef std::vector<int> vint_t;
```

类型别名声明等同于`typedef`声明。它可以出现在块作用域、类作用域或命名空间作用域。根据 C++11 段落 7.1.3.2：

typedef 名称也可以通过别名声明引入。在使用关键字后面的标识符成为 typedef 名称，后面的可选属性说明符序列与该 typedef 名称相关。它具有与通过 typedef 说明符引入的语义相同。特别是，它不定义新类型，也不应出现在类型标识符中。

然而，别名声明在创建数组类型和函数指针类型的别名时更易读且更清晰。在*如何做到...*部分的示例中，很容易理解`array_t`是 10 个整数的数组类型的名称，`fn`是一个接受两个`byte`和`double`类型参数并返回`void`的函数类型的名称。这也与声明`std::function`对象的语法一致（例如，`std::function<void(byte, double)> f`）。

新语法的驱动目的是定义别名模板。这些模板在特化时，等同于将别名模板的模板参数替换为`type-id`中的模板参数的结果。

重要的是要注意以下事项：

+   别名模板不能部分或显式地进行特化。

+   当推断模板参数时，别名模板永远不会通过模板参数推断进行推断。

+   当特化别名模板时产生的类型不允许直接或间接地使用其自身类型。

# 理解统一初始化

花括号初始化是 C++11 中初始化数据的统一方法。因此，它也被称为*统一初始化*。这可以说是 C++11 中开发人员应该理解和使用的最重要的功能之一。它消除了以前在初始化基本类型、聚合和非聚合类型以及数组和标准容器之间的区别。

# 准备工作

要继续使用这个示例，你需要熟悉直接初始化（从显式的构造函数参数集初始化对象）和复制初始化（从另一个对象初始化对象）。以下是这两种初始化类型的简单示例，但是要了解更多细节，你应该查看其他资源：

```cpp
    std::string s1("test");   // direct initialization 
    std::string s2 = "test";  // copy initialization
```

# 如何做...

为了统一初始化对象，无论其类型如何，都可以使用花括号初始化形式`{}`，它可以用于直接初始化和复制初始化。在使用花括号初始化时，这些被称为直接列表初始化和复制列表初始化。

```cpp
    T object {other};   // direct list initialization 
    T object = {other}; // copy list initialization
```

统一初始化的示例如下：

+   标准容器：

```cpp
        std::vector<int> v { 1, 2, 3 };
        std::map<int, std::string> m { {1, "one"}, { 2, "two" }};
```

+   动态分配的数组：

```cpp
        int* arr2 = new int[3]{ 1, 2, 3 };    
```

+   数组：

```cpp
        int arr1[3] { 1, 2, 3 }; 
```

+   内置类型：

```cpp
        int i { 42 };
        double d { 1.2 };    
```

+   用户定义的类型：

```cpp
        class foo
        {
          int a_;
          double b_;
        public:
          foo():a_(0), b_(0) {}
          foo(int a, double b = 0.0):a_(a), b_(b) {}
        }; 

        foo f1{}; 
        foo f2{ 42, 1.2 }; 
        foo f3{ 42 };
```

+   用户定义的 POD 类型：

```cpp
        struct bar { int a_; double b_;};
        bar b{ 42, 1.2 };
```

# 工作原理... 

在 C++11 之前，对象根据其类型需要不同类型的初始化：

+   基本类型可以使用赋值进行初始化：

```cpp
        int a = 42; 
        double b = 1.2;
```

+   如果类对象具有转换构造函数（在 C++11 之前，具有单个参数的构造函数被称为*转换构造函数*），也可以使用赋值进行初始化：

```cpp
        class foo 
        { 
          int a_; 
        public: 
          foo(int a):a_(a) {} 
        }; 
        foo f1 = 42;
```

+   非聚合类可以在提供参数时使用括号（函数形式）进行初始化，只有在执行默认初始化（调用默认构造函数）时才可以不使用任何括号。在下一个示例中，`foo`是在*如何做...*部分中定义的结构：

```cpp
        foo f1;           // default initialization 
        foo f2(42, 1.2); 
        foo f3(42); 
        foo f4();         // function declaration
```

+   聚合和 POD 类型可以使用花括号初始化进行初始化。在下一个示例中，`bar`是在*如何做...*部分中定义的结构：

```cpp
        bar b = {42, 1.2}; 
        int a[] = {1, 2, 3, 4, 5};
```

除了初始化数据的不同方法外，还有一些限制。例如，初始化标准容器的唯一方法是首先声明一个对象，然后将元素插入其中；vector 是一个例外，因为可以从可以使用聚合初始化进行先期初始化的数组中分配值。另一方面，动态分配的聚合体不能直接初始化。

*如何做...*部分中的所有示例都使用了直接初始化，但是使用花括号初始化也可以进行复制初始化。这两种形式，直接初始化和复制初始化，在大多数情况下可能是等效的，但是复制初始化不够宽松，因为它在其隐式转换序列中不考虑显式构造函数，必须直接从初始化程序产生对象，而直接初始化期望从初始化程序到构造函数的参数的隐式转换。动态分配的数组只能使用直接初始化进行初始化。

在前面的示例中显示的类中，`foo`是唯一既有默认构造函数又有带参数的构造函数的类。要使用默认构造函数执行默认初始化，我们需要使用空括号，即`{}`。要使用带参数的构造函数，我们需要在括号`{}`中提供所有参数的值。与非聚合类型不同，其中默认初始化意味着调用默认构造函数，对于聚合类型，默认初始化意味着用零初始化。

在 C++11 中，初始化标准容器（如上面显示的向量和映射）是可能的，因为所有标准容器都有一个额外的构造函数，该构造函数接受类型为`std::initializer_list<T>`的参数。这基本上是类型为`T const`的元素数组的轻量级代理。然后，这些构造函数从初始化列表中的值初始化内部数据。

使用`std::initializer_list`进行初始化的方式如下：

+   编译器解析初始化列表中元素的类型（所有元素必须具有相同类型）。

+   编译器创建一个具有初始化列表中元素的数组。

+   编译器创建一个`std::initializer_list<T>`对象来包装先前创建的数组。

+   `std::initializer_list<T>`对象作为参数传递给构造函数。

初始化列表始终优先于其他构造函数，其中使用大括号初始化。如果类存在这样的构造函数，那么在执行大括号初始化时将调用它：

```cpp
    class foo 
    { 
      int a_; 
      int b_; 
    public: 
      foo() :a_(0), b_(0) {} 

      foo(int a, int b = 0) :a_(a), b_(b) {} 
      foo(std::initializer_list<int> l) {} 
    }; 

    foo f{ 1, 2 }; // calls constructor with initializer_list<int>
```

优先规则适用于任何函数，而不仅仅是构造函数。在下面的示例中，存在相同函数的两个重载。使用初始化列表调用函数将解析为调用具有`std::initializer_list`的重载：

```cpp
    void func(int const a, int const b, int const c) 
    { 
      std::cout << a << b << c << std::endl; 
    } 

    void func(std::initializer_list<int> const l) 
    { 
      for (auto const & e : l) 
        std::cout << e << std::endl; 
    } 

    func({ 1,2,3 }); // calls second overload
```

然而，这可能导致错误。例如，考虑向量类型。在向量的构造函数中，有一个构造函数有一个表示要分配的初始元素数量的单个参数，另一个构造函数有一个`std::initializer_list`作为参数。如果意图是创建一个预分配大小的向量，使用大括号初始化将不起作用，因为具有`std::initializer_list`的构造函数将是要调用的最佳重载：

```cpp
    std::vector<int> v {5};
```

上面的代码不会创建一个具有五个元素的向量，而是创建一个具有一个值为 5 的元素的向量。要实际创建一个具有五个元素的向量，必须使用括号形式的初始化：

```cpp
    std::vector<int> v (5);
```

另一个需要注意的是，大括号初始化不允许缩小转换。根据 C++标准（参见标准的第 8.5.4 段），缩小转换是一种隐式转换：

- 从浮点类型到整数类型

- 从长双精度到双精度或浮点数，或从双精度到浮点数，除非源是常量表达式，并且转换后的实际值在可以表示的值范围内（即使不能精确表示）

- 从整数类型或未作用域的枚举类型到浮点类型，除非源是常量表达式，并且转换后的实际值适合目标类型，并且在转换回原始类型时会产生原始值

- 从整数类型或未作用域的枚举类型到不能表示原始类型所有值的整数类型，除非源是常量表达式，并且转换后的实际值适合目标类型，并且在转换回原始类型时会产生原始值。

以下声明触发编译器错误，因为它们需要缩小转换：

```cpp
    int i{ 1.2 };           // error 

    double d = 47 / 13; 
    float f1{ d };          // error 
    float f2{47/13};        // OK
```

要修复错误，必须进行显式转换：

```cpp
    int i{ static_cast<int>(1.2) }; 

    double d = 47 / 13; 
    float f1{ static_cast<float>(d) };
```

花括号初始化列表不是一个表达式，也没有类型。因此，`decltype`不能用于花括号初始化列表，模板类型推断也不能推断与花括号初始化列表匹配的类型。

# 还有更多

以下示例展示了直接列表初始化和复制列表初始化的几个例子。在 C++11 中，所有这些表达式的推断类型都是`std::initializer_list<int>`。

```cpp
auto a = {42};   // std::initializer_list<int>
auto b {42};     // std::initializer_list<int>
auto c = {4, 2}; // std::initializer_list<int>
auto d {4, 2};   // std::initializer_list<int>
```

C++17 已经改变了列表初始化的规则，区分了直接列表初始化和复制列表初始化。类型推断的新规则如下：

+   对于复制列表初始化，如果列表中的所有元素具有相同类型，auto 推断将推断出一个`std::initializer_list<T>`，否则会形成错误。

+   对于直接列表初始化，如果列表只有一个元素，auto 推断会推断出一个`T`，如果有多个元素，则会形成错误。

根据新规则，前面的示例将会发生变化：`a`和`c`被推断为`std::initializer_list<int>`；`b`被推断为`int`；`d`使用直接初始化，在花括号初始化列表中有多个值，会触发编译错误。

```cpp
auto a = {42};   // std::initializer_list<int>
auto b {42};     // int
auto c = {4, 2}; // std::initializer_list<int>
auto d {4, 2};   // error, too many
```

# 另请参阅

+   *尽可能使用 auto*

+   理解非静态成员初始化的各种形式

# 理解非静态成员初始化的各种形式

构造函数是进行非静态类成员初始化的地方。许多开发人员更喜欢在构造函数体中进行赋值。除了几种特殊情况需要在构造函数体中进行赋值之外，非静态成员的初始化应该在构造函数的初始化列表中进行，或者在 C++11 中，当它们在类中声明时使用默认成员初始化。在 C++11 之前，类的常量和非常量非静态数据成员必须在构造函数中初始化。在类中声明时初始化只对静态常量可能。正如我们将在后面看到的，C++11 消除了这种限制，允许在类声明中初始化非静态成员。这种初始化称为*默认成员初始化*，并在接下来的章节中进行解释。

本文将探讨非静态成员初始化应该如何进行的方式。

# 如何做...

要初始化类的非静态成员，应该：

+   对于具有多个构造函数的类的成员提供默认值，应使用默认成员初始化来为这些成员提供公共初始化器（见以下代码中的`[3]`和`[4]`）。

+   对于常量，无论是静态的还是非静态的，应使用默认成员初始化（见以下代码中的`[1]`和`[2]`）。

+   使用构造函数初始化列表来初始化没有默认值但依赖于构造函数参数的成员（见以下代码中的`[5]`和`[6]`）。

+   当其他选项不可行时，在构造函数中使用赋值（例如使用指针`this`初始化数据成员，检查构造函数参数值，并在使用这些值或两个非静态数据成员的自引用初始化成员之前抛出异常）。

以下示例展示了这些初始化形式：

```cpp
    struct Control 
    { 
      const int DefaultHeigh = 14;                  // [1] 
      const int DefaultWidth = 80;                  // [2] 

      TextVAligment valign = TextVAligment::Middle; // [3] 
      TextHAligment halign = TextHAligment::Left;   // [4] 

      std::string text; 

      Control(std::string const & t) : text(t)       // [5] 
      {} 

      Control(std::string const & t, 
        TextVerticalAligment const va, 
        TextHorizontalAligment const ha):  
      text(t), valign(va), halign(ha)                 // [6] 
      {} 
    };
```

# 工作原理...

应该在构造函数的初始化列表中初始化非静态数据成员，如下例所示：

```cpp
    struct Point 
    { 
      double X, Y; 
      Point(double const x = 0.0, double const y = 0.0) : X(x), Y(y)  {} 
    };
```

然而，许多开发人员并不使用初始化列表，而更喜欢在构造函数体中进行赋值，甚至混合使用赋值和初始化列表。这可能是由于几个原因--对于具有许多成员的较大类，构造函数赋值可能看起来比长长的初始化列表更容易阅读，也许分成许多行，或者可能是因为他们熟悉其他没有初始化列表的编程语言，或者不幸的是，由于各种原因，他们甚至不知道它的存在。

重要的是要注意，非静态数据成员初始化的顺序是它们在类定义中声明的顺序，而不是它们在构造函数初始化列表中初始化的顺序。另一方面，非静态数据成员销毁的顺序是构造的相反顺序。

在构造函数中使用赋值是不高效的，因为这可能会创建稍后被丢弃的临时对象。如果在初始化列表中未初始化非静态成员，则这些成员将通过它们的默认构造函数进行初始化，然后在构造函数体中赋值时调用赋值运算符。如果默认构造函数分配资源（如内存或文件），并且必须在赋值运算符中进行释放和重新分配，则可能会导致效率低下：

```cpp
    struct foo 
    { 
      foo()  
      { std::cout << "default constructor" << std::endl; } 
      foo(std::string const & text)  
      { std::cout << "constructor '" << text << "'" << std::endl; } 
      foo(foo const & other)
      { std::cout << "copy constructor" << std::endl; } 
      foo(foo&& other)  
      { std::cout << "move constructor" << std::endl; }; 
      foo& operator=(foo const & other)  
      { std::cout << "assignment" << std::endl; return *this; } 
      foo& operator=(foo&& other)  
      { std::cout << "move assignment" << std::endl; return *this;} 
      ~foo()  
      { std::cout << "destructor" << std::endl; } 
    }; 

    struct bar 
    { 
      foo f; 

      bar(foo const & value) 
      { 
        f = value; 
      } 
    }; 

    foo f; 
    bar b(f);
```

前面的代码产生以下输出，显示数据成员`f`首先进行默认初始化，然后赋予一个新值：

```cpp
default constructor 
default constructor 
assignment 
destructor 
destructor
```

将初始化从构造函数体中的赋值更改为初始化列表将用复制构造函数替换默认构造函数加赋值运算符的调用：

```cpp
    bar(foo const & value) : f(value) { }
```

添加前面的代码行会产生以下输出：

```cpp
default constructor 
copy constructor 
destructor 
destructor
```

由于这些原因，至少对于内置类型之外的其他类型（如`bool`、`char`、`int`、`float`、`double`或指针），你应该更喜欢构造函数初始化列表。然而，为了保持初始化风格的一致性，当可能时，你应该始终更喜欢构造函数初始化列表。有几种情况下使用初始化列表是不可能的；这些情况包括以下情况（但列表可能会扩展到其他情况）：

+   如果一个成员必须用指向包含它的对象的指针或引用进行初始化，使用初始化列表中的`this`指针可能会触发一些编译器的警告，因为它在对象构造之前被使用。

+   如果有两个数据成员必须包含对彼此的引用。

+   如果要测试输入参数并在使用参数的值初始化非静态数据成员之前抛出异常。

从 C++11 开始，非静态数据成员可以在类中声明时进行初始化。这称为*默认成员初始化*，因为它应该表示使用默认值进行初始化。默认成员初始化适用于常量和不是基于构造函数参数进行初始化的成员（换句话说，其值不取决于对象的构造方式）：

```cpp
    enum class TextFlow { LeftToRight, RightToLeft }; 

    struct Control 
    { 
      const int DefaultHeight = 20; 
      const int DefaultWidth = 100; 

      TextFlow textFlow = TextFlow::LeftToRight; 
      std::string text; 

      Control(std::string t) : text(t) 
      {} 
    };
```

在前面的例子中，`DefaultHeight`和`DefaultWidth`都是常量；因此，这些值不取决于对象的构造方式，因此它们在声明时进行初始化。`textFlow`对象是一个非常量非静态数据成员，其值也不取决于对象的初始化方式（它可以通过另一个成员函数进行更改），因此在声明时也使用默认成员初始化进行初始化。另一方面，`text`也是一个非常量非静态数据成员，但它的初始值取决于对象的构造方式，因此它在构造函数的初始化列表中使用作为参数传递给构造函数的值进行初始化，而不是在构造函数体中进行赋值。

如果数据成员既使用默认成员初始化，又使用构造函数初始化列表进行初始化，后者优先，并且默认值将被丢弃。为了举例说明，让我们再次考虑之前的`foo`类和使用它的以下`bar`类：

```cpp
    struct bar 
    { 
      foo f{"default value"}; 

      bar() : f{"constructor initializer"} 
      { 
      } 
    }; 

    bar b;
```

在这种情况下，输出不同，因为默认初始化列表中的值被丢弃，对象不会被初始化两次：

```cpp
constructor
constructor initializer
destructor
```

为每个成员使用适当的初始化方法不仅可以产生更高效的代码，还可以产生更有组织和更易读的代码。

# 控制和查询对象对齐

C++11 提供了标准化的方法来指定和查询类型的对齐要求（以前只能通过特定于编译器的方法实现）。控制对齐对于提高不同处理器上的性能并启用一些仅适用于特定对齐数据的指令非常重要。例如，Intel SSE 和 Intel SSE2 要求数据的对齐为 16 字节，而对于 Intel 高级矢量扩展（或 Intel AVX），强烈建议使用 32 字节对齐。本教程探讨了`alignas`说明符用于控制对齐要求以及`alignof`运算符用于检索类型的对齐要求。

# 准备工作

您应该熟悉数据对齐是什么以及编译器如何执行默认数据对齐。但是，关于后者的基本信息在*它是如何工作的...*部分中提供。

# 如何做...

+   控制类型的对齐方式（无论是在类级别还是数据成员级别）或对象时，请使用`alignas`说明符：

```cpp
        struct alignas(4) foo 
        { 
          char a; 
          char b; 
        }; 
        struct bar 
        { 
          alignas(2) char a; 
          alignas(8) int  b; 
        }; 
        alignas(8)   int a; 
        alignas(256) long b[4];
```

+   要查询类型的对齐方式，请使用`alignof`运算符：

```cpp
        auto align = alignof(foo);
```

# 它是如何工作的...

处理器不是一次访问一个字节的内存，而是以 2 的幂次（2、4、8、16、32 等）的较大块访问。因此，编译器对内存中的数据进行对齐非常重要，以便处理器可以轻松访问。如果这些数据未对齐，编译器必须额外工作来访问数据；它必须读取多个数据块，移位，丢弃不必要的字节并将其余部分组合在一起。

C++编译器根据其数据类型的大小对变量进行对齐：`bool`和`char`为 1 字节，`short`为 2 字节，`int`、`long`和`float`为 4 字节，`double`和`long long`为 8 字节，依此类推。在涉及结构或联合时，对齐必须与最大成员的大小匹配，以避免性能问题。举例来说，让我们考虑以下数据结构：

```cpp
    struct foo1    // size = 1, alignment = 1 
    { 
      char a; 
    }; 

    struct foo2    // size = 2, alignment = 1 
    { 
      char a; 
      char b; 
    }; 

    struct foo3    // size = 8, alignment = 4 
    { 
      char a; 
      int  b; 
    };
```

`foo1`和`foo2`的大小不同，但对齐方式相同--即 1--因为所有数据成员都是`char`类型，其大小为 1。在结构`foo3`中，第二个成员是一个整数，其大小为 4。因此，该结构的成员的对齐是在地址的倍数为 4 的地方进行的。为了实现这一点，编译器引入了填充字节。实际上，结构`foo3`被转换为以下内容：

```cpp
    struct foo3_ 
    { 
      char a;        // 1 byte 
      char _pad0[3]; // 3 bytes padding to put b on a 4-byte boundary 
      int  b;        // 4 bytes 
    };
```

同样，以下结构的大小为 32 字节，对齐为 8；这是因为最大的成员是一个大小为 8 的`double`。然而，该结构需要在几个地方填充，以确保所有成员都可以在地址的倍数为 8 的地方访问：

```cpp
    struct foo4 
    { 
      int a; 
      char b; 
      float c; 
      double d; 
      bool e; 
    };
```

编译器创建的等效结构如下：

```cpp
    struct foo4_ 
    { 
      int a;         // 4 bytes 
      char b;        // 1 byte 
      char _pad0[3]; // 3 bytes padding to put c on a 8-byte boundary  
      float c;       // 4 bytes 
      char _pad1[4]; // 4 bytes padding to put d on a 8-byte boundary 
      double d;      // 8 bytes 
      bool e;        // 1 byte 
      char _pad2[7]; // 7 bytes padding to make sizeof struct multiple of 8 
    };
```

在 C++11 中，使用`alignas`说明符指定对象或类型的对齐方式。这可以采用表达式（求值为 0 或对齐的有效值的整数常量表达式）、类型标识或参数包。`alignas`说明符可以应用于不表示位字段的变量或类数据成员的声明，也可以应用于类、联合或枚举的声明。应用`alignas`规范的类型或对象的对齐要求将等于声明中使用的所有`alignas`规范的最大且大于零的表达式。

使用`alignas`说明符时有一些限制：

+   唯一有效的对齐方式是 2 的幂次（1、2、4、8、16、32 等）。任何其他值都是非法的，程序被视为不合法；这不一定会产生错误，因为编译器可能选择忽略该规范。

+   对齐方式为 0 始终被忽略。

+   如果声明中最大的`alignas`小于没有`alignas`说明符的自然对齐方式，则程序也被视为不合法。

在下面的例子中，`alignas`修饰符应用于类声明。没有`alignas`修饰符的自然对齐将是 1，但是使用`alignas(4)`后变为 4：

```cpp
    struct alignas(4) foo 
    { 
      char a; 
      char b; 
    };
```

换句话说，编译器将前面的类转换为以下内容：

```cpp
    struct foo 
    { 
      char a; 
      char b; 
      char _pad0[2]; 
    };
```

`alignas`修饰符可以应用于类声明和成员数据声明。在这种情况下，最严格（即最大）的值获胜。在下面的例子中，成员`a`的自然大小为 1，需要 2 的对齐；成员`b`的自然大小为 4，需要 8 的对齐，因此，最严格的对齐将是 8。整个类的对齐要求是 4，这比最严格的所需对齐要弱（即更小），因此它将被忽略，尽管编译器会产生警告：

```cpp
    struct alignas(4) foo 
    { 
      alignas(2) char a; 
      alignas(8) int  b; 
    };
```

结果是一个看起来像这样的结构：

```cpp
    struct foo 
    { 
      char a; 
      char _pad0[7]; 
      int b; 
      char _pad1[4]; 
    };
```

`alignas`修饰符也可以应用于变量。在下一个例子中，整数变量`a`需要放置在内存中的 8 的倍数处。下一个变量，4 个整数的数组`a`，需要放置在内存中的 8 的倍数处。下一个变量，4 个`long`的数组，需要放置在内存中的 256 的倍数处。因此，编译器将在两个变量之间引入多达 244 字节的填充（取决于变量`a`在内存中的位置，即 8 的倍数的地址）：

```cpp
    alignas(8)   int a;   
    alignas(256) long b[4]; 

    printf("%pn", &a); // eg. 0000006C0D9EF908 
    printf("%pn", &b); // eg. 0000006C0D9EFA00
```

通过查看地址，我们可以看到`a`的地址确实是 8 的倍数，而`b`的地址是 256 的倍数（十六进制 100）。

要查询类型的对齐方式，我们使用`alignof`运算符。与`sizeof`不同，此运算符只能应用于类型标识，而不能应用于变量或类数据成员。它可以应用于的类型可以是完整类型、数组类型或引用类型。对于数组，返回的值是元素类型的对齐方式；对于引用，返回的值是引用类型的对齐方式。以下是几个例子：

| 表达式 | 评估 |
| --- | --- |
| `alignof(char)` | 1，因为`char`的自然对齐是 1 |
| `alignof(int)` | 4，因为`int`的自然对齐是 4 |
| `alignof(int*)` | 32 位为 4，64 位为 8，指针的对齐方式 |
| `alignof(int[4])` | 4，因为元素类型的自然对齐是 4 |
| `alignof(foo&)` | 8，因为类`foo`的指定对齐方式（如上例所示）是 8 |

# 使用作用域的枚举

枚举是 C++中的基本类型，定义了一组值，始终是整数基础类型。它们的命名值，即常量，称为枚举器。使用关键字`enum`声明的枚举称为*未作用域的枚举*，而使用`enum class`或`enum struct`声明的枚举称为*作用域的枚举*。后者是在 C++11 中引入的，旨在解决未作用域的枚举的几个问题。

# 如何做...

+   最好使用作用域的枚举而不是未作用域的枚举。

+   为了使用作用域的枚举，应该使用`enum class`或`enum struct`声明枚举：

```cpp
        enum class Status { Unknown, Created, Connected };
        Status s = Status::Created;
```

`enum class`和`enum struct`声明是等效的，在本示例和本书的其余部分中，我们将使用`enum class`。 

# 它是如何工作的...

未作用域的枚举存在一些问题，给开发人员带来了问题：

+   它们将它们的枚举器导出到周围的作用域（因此称为未作用域的枚举），这有以下两个缺点：如果同一命名空间中的两个枚举具有相同名称的枚举器，可能会导致名称冲突，并且不可能使用其完全限定的名称使用枚举器：

```cpp
        enum Status {Unknown, Created, Connected};
        enum Codes {OK, Failure, Unknown};   // error 
        auto status = Status::Created;       // error
```

+   在 C++ 11 之前，它们无法指定所需的基础类型，该类型必须是整数类型。除非枚举值无法适应有符号或无符号整数，否则该类型不得大于`int`。由于这个原因，无法进行枚举的前向声明。原因是枚举的大小是未知的，因为在定义枚举值之前，基础类型是未知的，以便编译器可以选择适当的整数类型。这在 C++11 中已经得到解决。

+   枚举器的值隐式转换为`int`。这意味着您可以故意或意外地混合具有特定含义的枚举和整数（甚至可能与枚举的含义无关），编译器将无法警告您：

```cpp
        enum Codes { OK, Failure }; 
        void include_offset(int pixels) {/*...*/} 
        include_offset(Failure);
```

作用域枚举基本上是不同于非作用域枚举的强类型枚举：

+   它们不会将它们的枚举器导出到周围的范围。前面显示的两个枚举将更改为以下内容，不再生成名称冲突，并且可以完全限定枚举器的名称：

```cpp
        enum class Status { Unknown, Created, Connected }; 
        enum class Codes { OK, Failure, Unknown }; // OK 
        Codes code = Codes::Unknown;               // OK
```

+   您可以指定基础类型。非作用域枚举的基础类型的相同规则也适用于作用域枚举，只是用户可以明确指定基础类型。这也解决了前向声明的问题，因为在定义可用之前可以知道基础类型：

```cpp
        enum class Codes : unsigned int; 

        void print_code(Codes const code) {} 

        enum class Codes : unsigned int 
        {  
           OK = 0,  
           Failure = 1,  
           Unknown = 0xFFFF0000U 
        };
```

+   作用域枚举的值不再隐式转换为`int`。将`enum class`的值分配给整数变量将触发编译器错误，除非指定显式转换：

```cpp
        Codes c1 = Codes::OK;                       // OK 
        int c2 = Codes::Failure;                    // error 
        int c3 = static_cast<int>(Codes::Failure);  // OK
```

# 使用 override 和 final 来进行虚方法

与其他类似的编程语言不同，C++没有特定的语法来声明接口（基本上是只有纯虚方法的类），并且还存在一些与如何声明虚方法相关的缺陷。在 C++中，虚方法是用`virtual`关键字引入的。但是，在派生类中声明覆盖时，`virtual`关键字是可选的，这可能会导致处理大型类或层次结构时产生混淆。您可能需要在整个层次结构中导航到基类，以确定函数是否是虚拟的。另一方面，有时，确保虚函数甚至派生类不能再被覆盖或进一步派生是有用的。在本教程中，我们将看到如何使用 C++11 的特殊标识符`override`和`final`来声明虚函数或类。

# 准备工作

您应该熟悉 C++中的继承和多态的概念，例如抽象类、纯虚指定符、虚拟和重写方法。

# 如何做到...

为了确保在基类和派生类中正确声明虚方法，并增加可读性，请执行以下操作：

+   在派生类中声明虚函数时，始终使用`virtual`关键字，这些函数应该覆盖来自基类的虚函数，并且

+   在虚函数声明或定义的声明部分之后始终使用`override`特殊标识符。

```cpp
        class Base 
        { 
          virtual void foo() = 0;
          virtual void bar() {} 
          virtual void foobar() = 0; 
        };

        void Base::foobar() {}

        class Derived1 : public Base 
        { 
          virtual void foo() override = 0;
          virtual void bar() override {}
          virtual void foobar() override {} 
        }; 

        class Derived2 : public Derived1 
        { 
          virtual void foo() override {} 
        };
```

声明符是函数类型的一部分，不包括返回类型。

为了确保函数不能进一步被覆盖或类不能再被派生，使用`final`特殊标识符：

+   在虚函数声明或定义的声明部分之后，以防止在派生类中进行进一步的覆盖：

```cpp
        class Derived2 : public Derived1 
        { 
          virtual void foo() final {} 
        };
```

+   在类的声明中类的名称之后，以防止进一步派生类：

```cpp
        class Derived4 final : public Derived1 
        { 
          virtual void foo() override {} 
        };
```

# 它是如何工作的...

`override`的工作方式非常简单；在虚函数声明或定义中，它确保函数实际上是覆盖了基类函数，否则，编译器将触发错误。

需要注意的是，`override` 和 `final` 关键字都是特殊标识符，只在成员函数声明或定义中具有特定含义。它们不是保留关键字，仍然可以在程序的其他地方作为用户定义的标识符使用。

使用`override`特殊标识符有助于编译器检测虚拟方法不像下面的示例中覆盖另一个方法的情况：

```cpp
    class Base 
    { 
    public: 
      virtual void foo() {}
      virtual void bar() {}
    }; 

    class Derived1 : public Base 
    { 
    public:    
      void foo() override {}
      // for readability use the virtual keyword    

      virtual void bar(char const c) override {}
      // error, no Base::bar(char const) 
    };
```

另一个特殊标识符`final`在成员函数声明或定义中使用，表示该函数是虚拟的，不能在派生类中被覆盖。如果派生类尝试覆盖虚拟函数，编译器会触发错误：

```cpp
    class Derived2 : public Derived1 
    { 
      virtual void foo() final {} 
    }; 

    class Derived3 : public Derived2 
    { 
      virtual void foo() override {} // error 
    };
```

`final`修饰符也可以在类声明中使用，表示它不能被派生：

```cpp
    class Derived4 final : public Derived1 
    { 
      virtual void foo() override {} 
    };

    class Derived5 : public Derived4 // error 
    { 
    };
```

由于`override`和`final`在定义的上下文中具有特殊含义，并且实际上不是保留关键字，因此您仍然可以在 C++代码的其他任何地方使用它们。这确保了在 C++11 之前编写的现有代码不会因为使用这些标识符而中断：

```cpp
    class foo 
    { 
      int final = 0; 
      void override() {} 
    };
```

# 使用范围-based for 循环迭代范围

许多编程语言支持一种称为`for each`的`for`循环变体，即在集合的元素上重复一组语句。直到 C++11 之前，C++没有对此提供核心语言支持。最接近的功能是标准库中的通用算法`std::for_each`，它将函数应用于范围内的所有元素。C++11 为`for each`带来了语言支持，实际上称为*范围-based for 循环*。新的 C++17 标准对原始语言功能进行了几项改进。

# 准备工作

在 C++11 中，范围-based for 循环具有以下一般语法：

```cpp
    for ( range_declaration : range_expression ) loop_statement
```

为了举例说明使用范围-based for 循环的各种方式，我们将使用以下返回元素序列的函数：

```cpp
    std::vector<int> getRates() 
    { 
      return std::vector<int> {1, 1, 2, 3, 5, 8, 13}; 
    } 

    std::multimap<int, bool> getRates2() 
    { 
      return std::multimap<int, bool> { 
        { 1, true }, 
        { 1, true }, 
        { 2, false }, 
        { 3, true }, 
        { 5, true }, 
        { 8, false }, 
        { 13, true } 
      }; 
    }
```

# 如何做...

范围-based for 循环可以以各种方式使用：

+   通过承诺特定类型的元素序列：

```cpp
        auto rates = getRates();
        for (int rate : rates) 
          std::cout << rate << std::endl; 
        for (int& rate : rates) 
          rate *= 2;
```

+   通过不指定类型并让编译器推断它：

```cpp
        for (auto&& rate : getRates()) 
          std::cout << rate << std::endl; 

        for (auto & rate : rates) 
          rate *= 2; 

        for (auto const & rate : rates) 
          std::cout << rate << std::endl;
```

+   通过在 C++17 中使用结构化绑定和分解声明：

```cpp
        for (auto&& [rate, flag] : getRates2()) 
          std::cout << rate << std::endl;
```

# 工作原理...

在*如何做...*部分显示的范围-based for 循环的表达式基本上是一种语法糖，因为编译器将其转换为其他形式。在 C++17 之前，编译器生成的代码通常是以下内容：

```cpp
    { 
      auto && __range = range_expression; 
      for (auto __begin = begin_expr, __end = end_expr; 
      __begin != __end; ++__begin) { 
        range_declaration = *__begin; 
        loop_statement 
      } 
    }
```

此代码中的`begin_expr`和`end_expr`取决于范围的类型：

+   对于类似 C 的数组：`__range`和`__range + __bound`（其中`__bound`是数组中的元素数）

+   对于具有`begin`和`end`成员（无论其类型和可访问性）的类类型：`__range.begin()`和`__range.end()`。

+   对于其他人来说，`begin(__range)`和`end(__range)`是通过参数相关查找确定的。

重要的是要注意，如果一个类包含任何名为`begin`或`end`的成员（函数、数据成员或枚举器），无论其类型和可访问性如何，它们都将被选为`begin_expr`和`end_expr`。因此，这样的类类型不能用于范围-based for 循环。

在 C++17 中，编译器生成的代码略有不同：

```cpp
    { 
      auto && __range = range_expression; 
      auto __begin = begin_expr; 
      auto __end = end_expr; 
      for (; __begin != __end; ++__begin) { 
        range_declaration = *__begin; 
        loop_statement 
      } 
    }
```

新标准已经删除了开始表达式和结束表达式必须具有相同类型的约束。结束表达式不需要是实际的迭代器，但必须能够与迭代器进行不等比较。其中一个好处是范围可以由谓词限定。

# 另请参阅

+   *为自定义类型启用范围-based for 循环*

# 为自定义类型启用范围-based for 循环

正如我们在前面的教程中所看到的，基于范围的 for 循环（在其他编程语言中称为`for each`）允许您在范围的元素上进行迭代，提供了一种简化的语法，使得代码在许多情况下更易读。但是，基于范围的 for 循环不能直接与表示范围的任何类型一起使用，而是需要存在`begin()`和`end()`函数（对于非数组类型）作为成员或自由函数。在本教程中，我们将看到如何使自定义类型可以在基于范围的 for 循环中使用。

# 准备就绪

建议在继续本教程之前阅读*使用基于范围的 for 循环对范围进行迭代*，以便了解基于范围的 for 循环的工作原理以及编译器为这样的循环生成的代码。

为了演示如何使自定义类型表示序列的范围可以用于基于范围的 for 循环，我们将使用以下简单数组的实现：

```cpp
    template <typename T, size_t const Size> 
    class dummy_array 
    { 
      T data[Size] = {}; 

    public: 
      T const & GetAt(size_t const index) const 
      { 
        if (index < Size) return data[index]; 
        throw std::out_of_range("index out of range"); 
      } 

      void SetAt(size_t const index, T const & value) 
      { 
        if (index < Size) data[index] = value; 
        else throw std::out_of_range("index out of range"); 
      } 

      size_t GetSize() const { return Size; } 
    };
```

本教程的目的是使编写以下代码成为可能：

```cpp
    dummy_array<int, 3> arr; 
    arr.SetAt(0, 1); 
    arr.SetAt(1, 2); 
    arr.SetAt(2, 3); 

    for(auto&& e : arr) 
    {  
      std::cout << e << std::endl; 
    }
```

# 如何做到...

为了使自定义类型可以在基于范围的`for`循环中使用，您需要执行以下操作：

+   为必须实现以下运算符的类型创建可变和常量迭代器：

+   用于增加迭代器的`operator++`。

+   `operator*`用于对迭代器进行取消引用并访问迭代器指向的实际元素。

+   `operator!=`用于与另一个迭代器进行比较。

+   为该类型提供自由的`begin()`和`end()`函数。

鉴于前面的简单范围示例，我们需要提供以下内容：

1.  迭代器类的以下最小实现：

```cpp
        template <typename T, typename C, size_t const Size> 
        class dummy_array_iterator_type 
        { 
        public: 
          dummy_array_iterator_type(C& collection,  
                                    size_t const index) : 
          index(index), collection(collection) 
          { } 

        bool operator!= (dummy_array_iterator_type const & other) const 
        { 
          return index != other.index; 
        } 

        T const & operator* () const 
        { 
          return collection.GetAt(index); 
        } 

        dummy_array_iterator_type const & operator++ () 
        { 
          ++index; 
          return *this; 
        } 

        private: 
          size_t   index; 
          C&       collection; 
        };
```

1.  可变和常量迭代器的别名模板：

```cpp
        template <typename T, size_t const Size> 
        using dummy_array_iterator =  
           dummy_array_iterator_type< 
             T, dummy_array<T, Size>, Size>; 

        template <typename T, size_t const Size> 
        using dummy_array_const_iterator =  
           dummy_array_iterator_type< 
             T, dummy_array<T, Size> const, Size>;
```

1.  返回相应的开始和结束迭代器的自由`begin()`和`end()`函数，对于两个别名模板都进行重载：

```cpp
        template <typename T, size_t const Size> 
        inline dummy_array_iterator<T, Size> begin(
          dummy_array<T, Size>& collection) 
        { 
          return dummy_array_iterator<T, Size>(collection, 0); 
        } 

        template <typename T, size_t const Size> 
        inline dummy_array_iterator<T, Size> end(
          dummy_array<T, Size>& collection) 
        { 
          return dummy_array_iterator<T, Size>(
            collection, collection.GetSize()); 
        } 

        template <typename T, size_t const Size> 
        inline dummy_array_const_iterator<T, Size> begin( 
          dummy_array<T, Size> const & collection) 
        { 
          return dummy_array_const_iterator<T, Size>( 
            collection, 0); 
        } 

        template <typename T, size_t const Size> 
        inline dummy_array_const_iterator<T, Size> end( 
          dummy_array<T, Size> const & collection) 
        { 
          return dummy_array_const_iterator<T, Size>( 
            collection, collection.GetSize()); 
        }
```

# 它是如何工作的...

有了这个实现，前面显示的基于范围的 for 循环将如预期般编译和执行。在执行参数相关查找时，编译器将识别我们编写的两个`begin()`和`end()`函数（它们接受对`dummy_array`的引用），因此生成的代码变得有效。

在前面的示例中，我们定义了一个迭代器类模板和两个别名模板，称为`dummy_array_iterator`和`dummy_array_const_iterator`。`begin()`和`end()`函数都有这两种类型的迭代器的两个重载。这是必要的，以便我们考虑的容器可以在具有常量和非常量实例的基于范围的 for 循环中使用：

```cpp
    template <typename T, const size_t Size> 
    void print_dummy_array(dummy_array<T, Size> const & arr) 
    { 
      for (auto && e : arr) 
      { 
        std::cout << e << std::endl; 
      } 
    }
```

为了使我们在本教程中考虑的简单范围类能够用于基于范围的 for 循环，可能的替代方法是提供成员`begin()`和`end()`函数。一般来说，只有在您拥有并且可以修改源代码时才有意义。另一方面，本教程中展示的解决方案在所有情况下都适用，并且应优先于其他替代方案。

# 另请参阅

+   *创建类型别名和别名模板*

# 使用显式构造函数和转换运算符以避免隐式转换

在 C++11 之前，具有单个参数的构造函数被视为转换构造函数。使用 C++11，没有`explicit`说明符的每个构造函数都被视为转换构造函数。这样的构造函数定义了从其参数的类型到类的类型的隐式转换。类还可以定义转换运算符，将类的类型转换为另一指定类型。所有这些在某些情况下很有用，但在其他情况下可能会创建问题。在本教程中，我们将看到如何使用显式构造函数和转换运算符。

# 准备就绪

对于这个示例，您需要熟悉转换构造函数和转换运算符。在这个示例中，您将学习如何编写显式构造函数和转换运算符，以避免对类型进行隐式转换。显式构造函数和转换运算符（称为*用户定义的转换函数*）的使用使编译器能够产生错误--在某些情况下是编码错误--并允许开发人员快速发现并修复这些错误。

# 如何做...

要声明显式构造函数和转换运算符（无论它们是函数还是函数模板），在声明中使用 `explicit` 说明符。

以下示例显示了显式构造函数和转换运算符：

```cpp
    struct handle_t 
    { 
      explicit handle_t(int const h) : handle(h) {} 

      explicit operator bool() const { return handle != 0; }; 
    private: 
      int handle; 
    };
```

# 工作原理...

为了理解为什么需要显式构造函数以及它们的工作原理，我们首先来看看转换构造函数。以下类有三个构造函数：一个默认构造函数（没有参数），一个带有 `int` 参数的构造函数，以及一个带有两个参数的构造函数，一个 `int` 和一个 `double`。它们什么也不做，只是打印一条消息。从 C++11 开始，这些都被认为是转换构造函数。该类还有一个转换运算符，将类型转换为 `bool`：

```cpp
    struct foo 
    { 
      foo()
      { std::cout << "foo" << std::endl; }
      foo(int const a)
      { std::cout << "foo(a)" << std::endl; }
      foo(int const a, double const b)
      { std::cout << "foo(a, b)" << std::endl; } 

      operator bool() const { return true; } 
    };
```

基于此，以下对象的定义是可能的（请注意，注释表示控制台输出）：

```cpp
    foo f1;              // foo 
    foo f2 {};           // foo 

    foo f3(1);           // foo(a) 
    foo f4 = 1;          // foo(a) 
    foo f5 { 1 };        // foo(a) 
    foo f6 = { 1 };      // foo(a) 

    foo f7(1, 2.0);      // foo(a, b) 
    foo f8 { 1, 2.0 };   // foo(a, b) 
    foo f9 = { 1, 2.0 }; // foo(a, b)
```

`f1` 和 `f2` 调用默认构造函数。`f3`、`f4`、`f5` 和 `f6` 调用带有 `int` 参数的构造函数。请注意，所有这些对象的定义是等效的，即使它们看起来不同（`f3` 使用函数形式初始化，`f4` 和 `f6` 使用复制初始化，`f5` 使用大括号初始化列表直接初始化）。同样，`f7`、`f8` 和 `f9` 调用带有两个参数的构造函数。

值得注意的是，如果 `foo` 定义了一个接受 `std::initializer_list` 的构造函数，那么所有使用 `{}` 进行初始化的情况都将解析为该构造函数：

```cpp
    foo(std::initializer_list<int> l)  
    { std::cout << "foo(l)" << std::endl; }
```

在这种情况下，`f5` 和 `f6` 将打印 `foo(l)`，而 `f8` 和 `f9` 将生成编译器错误，因为初始化列表的所有元素都应该是整数。

这些看起来都是正确的，但隐式转换构造函数使得可能出现隐式转换不是我们想要的情况：

```cpp
    void bar(foo const f) 
    { 
    } 

    bar({});             // foo() 
    bar(1);              // foo(a) 
    bar({ 1, 2.0 });     // foo(a, b)
```

上面示例中的转换运算符也使我们能够在需要布尔值的地方使用 `foo` 对象：

```cpp
    bool flag = f1; 
    if(f2) {} 
    std::cout << f3 + f4 << std::endl; 
    if(f5 == f6) {}
```

前两个例子中 `foo` 预期被用作布尔值，但最后两个例子中的加法和相等测试可能是不正确的，因为我们很可能期望添加 `foo` 对象并测试 `foo` 对象的相等性，而不是它们隐式转换为的布尔值。

也许一个更现实的例子来理解问题可能出现的地方是考虑一个字符串缓冲区实现。这将是一个包含字符内部缓冲区的类。该类可能提供几个转换构造函数：一个默认构造函数，一个带有 `size_t` 参数表示预分配缓冲区大小的构造函数，以及一个带有 `char` 指针的构造函数，用于分配和初始化内部缓冲区。简而言之，这样的字符串缓冲区可能如下所示：

```cpp
    class string_buffer 
    { 
    public: 
      string_buffer() {} 

      string_buffer(size_t const size) {} 

      string_buffer(char const * const ptr) {} 

      size_t size() const { return ...; } 
      operator bool() const { return ...; } 
      operator char * const () const { return ...; } 
    };
```

基于这个定义，我们可以构造以下对象：

```cpp
    std::shared_ptr<char> str; 
    string_buffer sb1;             // empty buffer 
    string_buffer sb2(20);         // buffer of 20 characters 
    string_buffer sb3(str.get());   
    // buffer initialized from input parameter
```

`sb1` 是使用默认构造函数创建的，因此具有一个空缓冲区；`sb2` 是使用带有单个参数的构造函数初始化的，参数的值表示内部缓冲区的字符大小；`sb3` 是用现有缓冲区初始化的，并且用于定义内部缓冲区的大小并将其值复制到内部缓冲区。然而，同样的定义也使以下对象定义成为可能：

```cpp
    enum ItemSizes {DefaultHeight, Large, MaxSize}; 

    string_buffer b4 = 'a'; 
    string_buffer b5 = MaxSize;
```

在这种情况下，`b4`用`char`初始化。由于存在到`size_t`的隐式转换，将调用带有单个参数的构造函数。这里的意图并不一定清楚；也许应该是`"a"`而不是`'a'`，在这种情况下将调用第三个构造函数。然而，`b5`很可能是一个错误，因为`MaxSize`是代表`ItemSizes`的枚举器，与字符串缓冲区大小无关。这些错误的情况并不会以任何方式被编译器标记。

在构造函数的声明中使用`explicit`限定符，该构造函数将成为显式构造函数，不再允许隐式构造类类型的对象。为了举例说明这一点，我们将稍微修改之前的`string_buffer`类，声明所有构造函数为显式：

```cpp
    class string_buffer 
    { 
    public: 
      explicit string_buffer() {} 

      explicit string_buffer(size_t const size) {} 

      explicit string_buffer(char const * const ptr) {} 

      explicit operator bool() const { return ...; } 
      explicit operator char * const () const { return ...; } 
    };
```

更改是微小的，但在之前的示例中`b4`和`b5`的定义不再起作用，并且是不正确的，因为从`char`或`int`到`size_t`的隐式转换在重载解析期间不再可用于确定应调用哪个构造函数。结果是`b4`和`b5`都会出现编译错误。请注意，即使构造函数是显式的，`b1`、`b2`和`b3`仍然是有效的定义。

在这种情况下解决问题的唯一方法是提供从`char`或`int`到`string_buffer`的显式转换：

```cpp
    string_buffer b4 = string_buffer('a'); 
    string_buffer b5 = static_cast<string_buffer>(MaxSize); 
    string_buffer b6 = string_buffer{ "a" };
```

使用显式构造函数，编译器能够立即标记错误的情况，开发人员可以相应地做出反应，要么修复初始化并提供正确的值，要么提供显式转换。

只有在使用复制初始化时才会出现这种情况，而在使用函数式或通用初始化时不会。

以下定义仍然可能（并且是错误的）使用显式构造函数：

```cpp
    string_buffer b7{ 'a' }; 
    string_buffer b8('a');
```

与构造函数类似，转换运算符可以声明为显式（如前所示）。在这种情况下，从对象类型到转换运算符指定的类型的隐式转换不再可能，需要显式转换。考虑`b1`和`b2`，之前定义的`string_buffer`对象，以下不再可能使用显式转换`operator bool`：

```cpp
    std::cout << b1 + b2 << std::endl; 
    if(b1 == b2) {}
```

相反，它们需要显式转换为`bool`：

```cpp
    std::cout << static_cast<bool>(b1) + static_cast<bool>(b2);
    if(static_cast<bool>(b1) == static_cast<bool>(b2)) {}
```

# 另请参阅

+   *理解统一初始化*

# 使用未命名命名空间而不是静态全局变量

程序越大，您的程序在链接时可能遇到文件局部名称冲突的机会就越大。在源文件中声明并应该是翻译单元局部的函数或变量可能会与另一个翻译单元中声明的其他类似函数或变量发生冲突。这是因为所有未声明为静态的符号都具有外部链接，它们的名称必须在整个程序中是唯一的。这个问题的典型 C 解决方案是将这些符号声明为静态，将它们的链接从外部改为内部，从而使它们成为翻译单元的局部。在这个教程中，我们将看看 C++对这个问题的解决方案。

# 准备就绪

在这个教程中，我们将讨论全局函数、静态函数和变量、命名空间和翻译单元等概念。除此之外，您需要理解内部链接和外部链接之间的区别；这对于这个教程至关重要。

# 如何做...

当您处于需要将全局符号声明为静态以避免链接问题的情况时，最好使用未命名的命名空间：

1.  在您的源文件中声明一个没有名称的命名空间。

1.  将全局函数或变量的定义放在未命名的命名空间中，而不使它们成为`static`。

下面的示例显示了两个不同翻译单元中分别称为`print()`的函数；它们每个都在一个未命名的命名空间中定义：

```cpp
    // file1.cpp 
    namespace 
    { 
      void print(std::string message) 
      { 
        std::cout << "[file1] " << message << std::endl; 
      } 
    } 

    void file1_run() 
    { 
      print("run"); 
    } 

    // file2.cpp 
    namespace 
    { 
      void print(std::string message) 
      { 
        std::cout << "[file2] " << message << std::endl; 
      } 
    } 

    void file2_run() 
    { 
      print("run"); 
    }
```

# 它是如何工作的...

当在翻译单元中声明函数时，它具有外部链接。这意味着来自两个不同翻译单元的具有相同名称的两个函数会生成链接错误，因为不可能有两个具有相同名称的符号。在 C 中以及一些 C++中解决这个问题的方法是将函数或变量声明为静态，并将其链接从外部更改为内部。在这种情况下，它的名称不再在翻译单元之外导出，链接问题得到避免。

C++中的正确解决方案是使用未命名命名空间。当您像上面所示地定义一个命名空间时，编译器将其转换为以下形式：

```cpp
    // file1.cpp 
    namespace _unique_name_ {} 
    using namespace _unique_name_; 
    namespace _unique_name_ 
    { 
      void print(std::string message) 
      { 
        std::cout << "[file1] " << message << std::endl; 
      } 
    } 

    void file1_run() 
    { 
      print("run"); 
    }
```

首先，它声明了一个具有唯一名称的命名空间（名称是什么以及如何生成该名称是编译器实现的细节，不应该成为关注点）。此时，命名空间是空的，这一行的目的基本上是建立命名空间。其次，使用指令将`_unique_name_`命名空间中的所有内容引入当前命名空间。第三，使用编译器生成的名称定义了命名空间，就像在原始源代码中一样（当它没有名称时）。

通过在未命名命名空间中定义翻译单元本地的`print()`函数，它们只有本地可见性，但它们的外部链接不再产生链接错误，因为它们现在具有外部唯一名称。

未命名命名空间在涉及模板的可能更晦涩的情况下也可以工作。在 C++11 之前，模板非类型参数不能具有内部链接的名称，因此无法使用静态变量。另一方面，未命名命名空间中的符号具有外部链接，并且可以用作模板参数。尽管 C++11 解除了模板非类型参数的此链接限制，但在最新版本的 VC++编译器中仍然存在。下面的示例显示了这个问题，其中声明 t1 会产生编译错误，因为非类型参数表达式具有内部链接，但`t2`是正确的，因为`Size2`具有外部链接。（请注意，使用 Clang 和 gcc 编译下面的代码不会产生任何错误。）

```cpp
    template <int const& Size> 
    class test {}; 

    static int Size1 = 10; 

    namespace 
    { 
      int Size2 = 10; 
    } 

    test<Size1> t1; 
    test<Size2> t2;
```

# 另请参阅

+   *使用内联命名空间进行符号版本控制*

# 使用内联命名空间进行符号版本控制

C++11 标准引入了一种新类型的命名空间，称为*内联命名空间*，它基本上是一种使嵌套命名空间的声明看起来和行为像是属于周围命名空间的机制。内联命名空间使用`inline`关键字在命名空间声明中声明（未命名命名空间也可以内联）。这是一个有用的库版本控制功能，在这个示例中，我们将看到如何使用内联命名空间来对符号进行版本控制。从这个示例中，您将学习如何使用内联命名空间和条件编译对源代码进行版本控制。

# 准备工作

在这个示例中，我们将讨论命名空间和嵌套命名空间、模板和模板特化，以及使用预处理器宏进行条件编译。熟悉这些概念是继续进行示例所必需的。

# 操作步骤如下：

提供库的多个版本，并让用户决定使用哪个版本，可以采取以下步骤：

+   在命名空间中定义库的内容。

+   在内部内联命名空间中定义库的每个版本或其部分。

+   使用预处理器宏和`#if`指令来启用库的特定版本。

以下示例显示了一个库，其中有两个客户可以使用的版本：

```cpp
    namespace modernlib 
    { 
      #ifndef LIB_VERSION_2 
      inline namespace version_1 
      { 
        template<typename T> 
        int test(T value) { return 1; } 
      } 
      #endif 

      #ifdef LIB_VERSION_2 
      inline namespace version_2 
      { 
        template<typename T> 
        int test(T value) { return 2; } 
      } 
      #endif 
    }
```

# 工作原理...

内联命名空间的成员被视为是周围命名空间的成员。这样的成员可以部分特化、显式实例化或显式特化。这是一个传递属性，这意味着如果命名空间 A 包含一个内联命名空间 B，B 包含一个内联命名空间 C，那么 C 的成员会出现为 B 和 A 的成员，B 的成员会出现为 A 的成员。

为了更好地理解内联命名空间的帮助，让我们考虑从第一个版本到第二个版本（以及更多版本）随时间演变的库的情况。这个库在名为`modernlib`的命名空间下定义了所有类型和函数。在第一个版本中，这个库可能是这样的：

```cpp
    namespace modernlib 
    { 
      template<typename T> 
      int test(T value) { return 1; } 
    }
```

库的客户端可以进行以下调用并获得值 1：

```cpp
    auto x = modernlib::test(42);
```

然而，客户端可能决定将模板函数`test()`特化为以下内容：

```cpp
    struct foo { int a; }; 

    namespace modernlib 
    { 
      template<> 
      int test(foo value) { return value.a; } 
    } 

    auto y = modernlib::test(foo{ 42 });
```

在这种情况下，`y`的值不再是 1，而是 42，因为调用了用户专用的函数。

一切都运行正常，但作为库开发人员，您决定创建库的第二个版本，但仍然同时发布第一和第二个版本，并让用户使用宏来控制使用哪个版本。在第二个版本中，您提供了`test()`函数的新实现，不再返回 1，而是返回 2。为了能够提供第一和第二个版本的实现，您将它们放在名为`version_1`和`version_2`的嵌套命名空间中，并使用预处理宏有条件地编译库：

```cpp
    namespace modernlib 
    { 
      namespace version_1 
      { 
        template<typename T> 
        int test(T value) { return 1; } 
      } 

      #ifndef LIB_VERSION_2 
      using namespace version_1; 
      #endif 

      namespace version_2  
      { 
        template<typename T> 
        int test(T value) { return 2; } 
      } 

      #ifdef LIB_VERSION_2 
      using namespace version_2; 
      #endif 
    }
```

突然之间，无论客户端使用库的第一个版本还是第二个版本，客户端代码都会中断。这是因为`test`函数现在位于嵌套命名空间中，并且对`foo`的特化是在`modernlib`命名空间中进行的，而实际上应该在`modernlib::version_1`或`modernlib::version_2`中进行。这是因为要求在模板的声明命名空间中进行模板特化。在这种情况下，客户端需要像这样更改代码：

```cpp
    #define LIB_VERSION_2 

    #include "modernlib.h" 

    struct foo { int a; }; 

    namespace modernlib 
    { 
      namespace version_2  
      { 
        template<> 
        int test(foo value) { return value.a; } 
      } 
    }
```

这是一个问题，因为库泄漏了实现细节，客户端需要意识到这些细节才能进行模板特化。这些内部细节在本文中所示的内联命名空间中隐藏。有了`modernlib`库的定义，`modernlib`命名空间中`test()`函数的特化的客户端代码不再中断，因为`version_1::test()`或`version_2::test()`（取决于客户端实际使用的版本）在模板特化时作为封闭`modernlib`命名空间的一部分。实现的细节现在对客户端隐藏，客户端只看到周围的命名空间`modernlib`。

但是，您应该记住：

+   `std`命名空间保留用于标准，不应内联。

+   如果命名空间在其第一次定义中不是内联的，则不应内联定义命名空间。

# 另请参阅

+   *使用未命名命名空间代替静态全局变量*

# 使用结构化绑定处理多返回值

从函数返回多个值是非常常见的事情，但在 C++中没有直接启用它的一流解决方案。开发人员必须在通过引用参数返回多个值的函数之间进行选择，定义一个包含多个值的结构，或者返回`std::pair`或`std::tuple`。前两种方法使用具有优势的命名变量，它们清楚地指示返回值的含义，但缺点是它们必须明确定义。`std::pair`具有名为`first`和`second`的成员，而`std::tuple`具有无名成员，只能通过函数调用检索，但可以使用`std::tie()`将其复制到命名变量中。这些解决方案都不是理想的。

C++17 将`std::tie()`的语义用法扩展为一种一流的核心语言特性，它使得可以将元组的值解包到命名变量中。这个特性被称为*结构化绑定*。

# 准备工作

对于这个示例，您应该熟悉标准实用类型`std::pair`和`std::tuple`以及实用函数`std::tie()`。

# 如何操作...

要使用支持 C++17 的编译器从函数返回多个值，应该执行以下操作：

1.  使用`std::tuple`作为返回类型。

```cpp
        std::tuple<int, std::string, double> find() 
        { 
          return std::make_tuple(1, "marius", 1234.5); 
        }
```

1.  使用结构化绑定将元组的值解包到命名对象中。

```cpp
        auto [id, name, score] = find();
```

1.  使用分解声明将返回的值绑定到`if`语句或`switch`语句中的变量。

```cpp
        if (auto [id, name, score] = find(); score > 1000) 
        { 
          std::cout << name << std::endl; 
        }
```

# 工作原理...

结构化绑定是一种语言特性，其工作方式与`std::tie()`完全相同，只是我们不必为每个需要使用`std::tie()`显式解包的值定义命名变量。通过结构化绑定，我们可以使用`auto`说明符在单个定义中定义所有命名变量，以便编译器可以推断出每个变量的正确类型。

为了举例说明，让我们考虑在`std::map`中插入项目的情况。插入方法返回一个`std::pair`，其中包含插入的元素的迭代器或阻止插入的元素，以及一个布尔值，指示插入是否成功。下面的代码非常明确，使用`second`或`first->second`使得代码更难阅读，因为您需要不断弄清它们代表什么：

```cpp
    std::map<int, std::string> m; 

    auto result = m.insert({ 1, "one" }); 
    std::cout << "inserted = " << result.second << std::endl 
              << "value = " << result.first->second << std::endl;
```

通过使用`std::tie`，可以使上述代码更易读，它将元组解包为单独的对象（并且可以与`std::pair`一起使用，因为`std::tuple`从`std::pair`具有转换赋值）：

```cpp
    std::map<int, std::string> m; 
    std::map<int, std::string>::iterator it; 
    bool inserted; 

    std::tie(it, inserted) = m.insert({ 1, "one" }); 
    std::cout << "inserted = " << inserted << std::endl 
              << "value = " << it->second << std::endl; 

    std::tie(it, inserted) = m.insert({ 1, "two" }); 
    std::cout << "inserted = " << inserted << std::endl 
              << "value = " << it->second << std::endl;
```

这段代码并不一定更简单，因为它需要预先定义成对解包的对象。同样，元组的元素越多，你需要定义的对象就越多，但使用命名对象使得代码更易于阅读。

C++17 结构化绑定将元组元素的解包提升为一种语言特性；它不需要使用`std::tie()`，并且在声明时初始化对象：

```cpp
    std::map<int, std::string> m; 
    { 
      auto[it, inserted] = m.insert({ 1, "one" }); 
      std::cout << "inserted = " << inserted << std::endl 
                << "value = " << it->second << std::endl; 
    } 

    { 
      auto[it, inserted] = m.insert({ 1, "two" }); 
      std::cout << "inserted = " << inserted << std::endl 
                << "value = " << it->second << std::endl; 
    }
```

上面示例中使用多个块是必要的，因为变量不能在同一块中重新声明，并且结构化绑定意味着使用`auto`说明符进行声明。因此，如果您需要像上面的示例中那样进行多次调用并使用结构化绑定，您必须使用不同的变量名称或像上面所示的多个块。一个替代方法是避免结构化绑定并使用`std::tie()`，因为它可以多次使用相同的变量进行调用，因此您只需要声明它们一次。

在 C++17 中，还可以使用`if(init; condition)`和`switch(init; condition)`的形式在`if`和`switch`语句中声明变量。这可以与结构化绑定结合使用以生成更简单的代码。在下面的示例中，我们尝试将一个新值插入到 map 中。调用的结果被解包到两个变量`it`和`inserted`中，它们在`if`语句的初始化部分中定义。`if`语句的条件是从插入的对象的值中评估出来的：

```cpp
    if(auto [it, inserted] = m.insert({ 1, "two" }); inserted)
    { std::cout << it->second << std::endl; }
```
