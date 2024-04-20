# 第二十一章：新的 C++17 功能

在本章中，我们将涵盖以下内容：

+   使用结构化绑定来解包捆绑的返回值

+   将变量范围限制为`if`和`switch`语句

+   从新的括号初始化规则中获益

+   让构造函数自动推断结果模板类类型

+   使用 constexpr-if 简化编译时决策

+   使用内联变量启用仅头文件库

+   使用折叠表达式实现方便的辅助函数

# 介绍

C++在 C++11、C++14 和最近的 C++17 中增加了很多内容。到目前为止，它与十年前完全不同。C++标准不仅标准化了语言，因为它需要被编译器理解，还标准化了 C++标准模板库（STL）。

本书将解释如何通过大量示例充分利用 STL。但首先，本章将集中讨论最重要的新语言特性。掌握它们将极大地帮助您编写可读、可维护和富有表现力的代码。

我们将看到如何使用结构化绑定舒适地访问对、元组和结构的单个成员，以及如何使用新的`if`和`switch`变量初始化功能来限制变量范围。C++11 引入了新的括号初始化语法，它看起来与初始化列表相同，引入了语法上的歧义，这些问题已经通过*新的括号初始化规则*得到解决。现在可以从实际构造函数参数中*推断*模板类实例的确切*类型*，如果模板类的不同特化将导致完全不同的代码，现在可以使用 constexpr-if 轻松表达。在许多情况下，使用新的*折叠表达式*可以使模板函数中的可变参数包处理变得更加容易。最后，使用新的内联变量声明静态全局可访问对象在仅头文件库中变得更加舒适，这在之前只对函数可行。

本章中的一些示例对库的实现者可能更有趣，而对于实现应用程序的开发人员来说可能不那么重要。虽然出于完整性的原因我们将研究这些特性，但不需要立即理解本章的所有示例就能理解本书的其余部分。

# 使用结构化绑定来解包捆绑的返回值

C++17 带来了一个新特性，结合了语法糖和自动类型推断：**结构化绑定**。这有助于将对、元组和结构的值分配给单独的变量。在其他编程语言中，这也被称为**解包**。

# 如何做...

应用结构化绑定以从一个捆绑结构中分配多个变量始终是一步。让我们首先看看 C++17 之前是如何做的。然后，我们可以看一下多个示例，展示了我们如何在 C++17 中做到这一点：

+   访问`std::pair`的单个值：假设我们有一个数学函数`divide_remainder`，它接受*被除数*和*除数*参数，并返回两者的分数以及余数。它使用`std::pair`捆绑返回这些值：

```cpp
        std::pair<int, int> divide_remainder(int dividend, int divisor);

```

考虑以下访问结果对的单个值的方式：

```cpp
        const auto result (divide_remainder(16, 3));
        std::cout << "16 / 3 is " 
                  << result.first << " with a remainder of " 
                  << result.second << 'n';
```

我们现在可以使用有表达力的名称将单个值分配给单独的变量，这样阅读起来更好：

```cpp
 auto [fraction, remainder] = divide_remainder(16, 3);
        std::cout << "16 / 3 is " 
                  << fraction << " with a remainder of "       
                  << remainder << 'n';
```

+   结构化绑定也适用于`std::tuple`：让我们看看以下示例函数，它可以获取在线股票信息：

```cpp
        std::tuple<std::string, 
                   std::chrono::system_clock::time_point, unsigned>
        stock_info(const std::string &name);
```

将其结果分配给单独的变量看起来就像前面的示例：

```cpp
 const auto [name, valid_time, price] = stock_info("INTC");
```

+   结构化绑定也适用于自定义结构：假设有以下结构：

```cpp
        struct employee {
            unsigned id;
            std::string name;
            std::string role;
            unsigned salary;
        };
```

现在，我们可以使用结构化绑定访问这些成员。假设我们有一个整个向量：

```cpp
        int main()
        {
            std::vector<employee> employees {
                /* Initialized from somewhere */};

            for (const auto &[id, name, role, salary] : employees) {
                std::cout << "Name: "   << name
                          << "Role: "   << role
                          << "Salary: " << salary << 'n';
            }
        }
```

# 它是如何工作的...

结构化绑定总是以相同的模式应用：

```cpp
auto [var1, var2, ...] = <pair, tuple, struct, or array expression>;
```

+   变量列表`var1, var2, ...`必须与被赋值的表达式包含的变量数量完全匹配。

+   `<pair, tuple, struct, or array expression>`必须是以下之一：

+   一个`std::pair`。

+   一个`std::tuple`。

+   一个`struct`。所有成员必须是*非静态*的，并且定义在*同一个基类*中。第一个声明的成员被分配给第一个变量，第二个成员被分配给第二个变量，依此类推。

+   固定大小的数组。

+   类型可以是`auto`、`const auto`、`const auto&`，甚至`auto&&`。

不仅出于*性能*的考虑，始终确保通过在适当的情况下使用引用来最小化不必要的复制。

如果我们在方括号之间写入*太多*或*太少*的变量，编译器将报错，告诉我们我们的错误：

```cpp
std::tuple<int, float, long> tup {1, 2.0, 3};
auto [a, b] = tup; // Does not work
```

这个例子显然试图将一个包含三个成员的元组变量塞入只有两个变量的情况中。编译器立即对此进行了处理，并告诉我们我们的错误：

```cpp
error: type 'std::tuple<int, float, long>' decomposes into 3 elements, but only 2 names were provided
auto [a, b] = tup;
```

# 还有更多...

STL 中的许多基本数据结构都可以立即使用结构化绑定进行访问，而无需我们改变任何内容。例如，考虑一个循环，打印出`std::map`的所有项：

```cpp
std::map<std::string, size_t> animal_population {
    {"humans",   7000000000},
    {"chickens", 17863376000},
    {"camels",   24246291},
    {"sheep",    1086881528},
    /* … */
};

for (const auto &[species, count] : animal_population) {
    std::cout << "There are " << count << " " << species 
              << " on this planet.n";
}
```

这个特定的例子之所以有效，是因为当我们遍历一个`std::map`容器时，我们在每次迭代步骤上得到`std::pair<const key_type, value_type>`节点。正是这些节点使用结构化绑定功能（`key_type`是`species`字符串，`value_type`是人口计数`size_t`）进行拆包，以便在循环体中单独访问它们。

在 C++17 之前，可以使用`std::tie`来实现类似的效果：

```cpp
int remainder;
std::tie(std::ignore, remainder) = divide_remainder(16, 5);
std::cout << "16 % 5 is " << remainder << 'n';
```

这个例子展示了如何将结果对拆分成两个变量。`std::tie`在某种意义上比结构化绑定功能弱，因为我们必须在*之前*定义我们想要绑定的所有变量。另一方面，这个例子展示了`std::tie`的一个优势，结构化绑定没有：值`std::ignore`充当一个虚拟变量。结果的小数部分被分配给它，这导致该值被丢弃，因为在这个例子中我们不需要它。

在使用结构化绑定时，我们没有`tie`虚拟变量，因此我们必须将所有的值绑定到命名变量。尽管如此，忽略其中一些是有效的，因为编译器可以轻松地优化未使用的绑定。

回到过去，`divide_remainder`函数可以以以下方式实现，使用输出参数：

```cpp
bool divide_remainder(int dividend, int divisor, 
                      int &fraction, int &remainder);

```

访问它看起来像这样：

```cpp
int fraction, remainder;
const bool success {divide_remainder(16, 3, fraction, remainder)};
if (success) {
    std::cout << "16 / 3 is " << fraction << " with a remainder of " 
              << remainder << 'n';
}
```

很多人仍然更喜欢这种方式，而不是返回像对、元组和结构这样的复杂结构，他们认为这样代码会更*快*，因为避免了这些值的中间复制。对于现代编译器来说，这*不再是真的*，因为它们会优化掉中间复制。

除了 C 语言中缺少的语言特性外，通过返回值返回复杂结构长时间被认为是慢的，因为对象必须在返回函数中初始化，然后复制到应该包含返回值的变量中。现代编译器支持**返回值优化**（RVO），可以省略中间复制。

# 将变量范围限制在 if 和 switch 语句中

尽可能限制变量的范围是一个很好的风格。然而，有时候，我们首先需要获取一些值，只有在符合某种条件的情况下，才能进一步处理。

为此，C++17 提供了带有初始化程序的`if`和`switch`语句。

# 如何做...

在这个示例中，我们在支持的上下文中都使用了初始化程序语法，以便看到它们如何整理我们的代码：

+   `if`语句：假设我们想要使用`std::map`的`find`方法在字符映射中找到一个字符：

```cpp
       if (auto itr (character_map.find(c)); itr != character_map.end()) {
           // *itr is valid. Do something with it.
       } else {
           // itr is the end-iterator. Don't dereference.
       }
       // itr is not available here at all

```

+   `switch`语句：这是从输入中获取字符并同时在`switch`语句中检查值以控制计算机游戏的样子。

```cpp
       switch (char c (getchar()); c) {
           case 'a': move_left();  break;
           case 's': move_back();  break;
           case 'w': move_fwd();   break;
           case 'd': move_right(); break;
           case 'q': quit_game();  break;

           case '0'...'9': select_tool('0' - c); break;

           default:
               std::cout << "invalid input: " << c << 'n';
       }
```

# 工作原理...

带有初始化器的`if`和`switch`语句基本上只是语法糖。以下两个示例是等效的：

*C++17 之前*：

```cpp
{
    auto var (init_value);
    if (condition) {
        // branch A. var is accessible
    } else {
        // branch B. var is accessible
    }
    // var is still accessible
}
```

*自* C++17：

```cpp
if (auto var (init_value); condition) {
    // branch A. var is accessible
} else {
    // branch B. var is accessible
}
// var is not accessible any longer
```

同样适用于`switch`语句：

在 C++17 之前：

```cpp
{
    auto var (init_value);
    switch (var) {
    case 1: ...
    case 2: ...
    ...
    }
    // var is still accessible
}
```

自 C++17 以来：

```cpp
switch (auto var (init_value); var) {
case 1: ...
case 2: ...
  ...
}
// var is not accessible any longer
```

这个特性非常有用，可以使变量的作用域尽可能短。在 C++17 之前，只能在代码周围使用额外的大括号来实现这一点，正如 C++17 之前的示例所示。短暂的生命周期减少了作用域中的变量数量，使我们的代码整洁，并且更容易重构。

# 还有更多...

另一个有趣的用例是临界区的有限作用域。考虑以下例子：

```cpp
if (std::lock_guard<std::mutex> lg {my_mutex}; some_condition) {
    // Do something
}
```

首先，创建一个`std::lock_guard`。这是一个接受互斥体参数作为构造函数参数的类。它在其构造函数中*锁定*互斥体，并且当它超出作用域时，在其析构函数中再次*解锁*它。这样，忘记解锁互斥体是不可能的。在 C++17 之前，需要一对额外的大括号来确定它再次解锁的作用域。

另一个有趣的用例是弱指针的作用域。考虑以下情况：

```cpp
if (auto shared_pointer (weak_pointer.lock()); shared_pointer != nullptr) {
    // Yes, the shared object does still exist
} else {
    // shared_pointer var is accessible, but a null pointer
}
// shared_pointer is not accessible any longer
```

这是另一个例子，我们会有一个无用的`shared_pointer`变量泄漏到当前作用域，尽管它在`if`条件块外部或有嘈杂的额外括号时可能是无用的！

带有初始化器的`if`语句在使用*遗留*API 和输出参数时特别有用：

```cpp
if (DWORD exit_code; GetExitCodeProcess(process_handle, &exit_code)) {
    std::cout << "Exit code of process was: " << exit_code << 'n';
}
// No useless exit_code variable outside the if-conditional
```

`GetExitCodeProcess`是 Windows 内核 API 函数。它返回给定进程句柄的退出代码，但只有在该句柄有效时才会返回。离开这个条件块后，变量就变得无用了，所以我们不再需要它在任何作用域中。

能够在`if`块中初始化变量在许多情况下显然非常有用，特别是在处理使用输出参数的遗留 API 时。

使用`if`和`switch`语句的初始化器来保持作用域紧凑。这样可以使您的代码更紧凑，更易于阅读，并且在代码重构会话中，移动代码会更容易。

# 从新的大括号初始化规则中获益

C++11 带来了新的大括号初始化语法`{}`。它的目的是允许*聚合*初始化，但也允许通常的构造函数调用。不幸的是，当将这个语法与`auto`变量类型结合使用时，很容易表达错误的事情。C++17 带来了增强的初始化规则。在本教程中，我们将阐明如何在 C++17 中使用哪种语法正确初始化变量。

# 如何做...

变量在一步中初始化。使用初始化语法，有两种不同的情况：

+   在不带有`auto`类型推断的大括号初始化语法中：

```cpp
       // Three identical ways to initialize an int:
       int x1 = 1;
       int x2  {1};
       int x3  (1);

       std::vector<int> v1   {1, 2, 3}; // Vector with three ints: 1, 2, 3
       std::vector<int> v2 = {1, 2, 3}; // same here
       std::vector<int> v3   (10, 20);  // Vector with 10 ints, 
                                        // each have value 20
```

+   使用带有`auto`类型推断的大括号初始化语法：

```cpp
       auto v   {1};         // v is int
       auto w   {1, 2};      // error: only single elements in direct 
                             // auto initialization allowed! (this is new)
       auto x = {1};         // x is std::initializer_list<int>
       auto y = {1, 2};      // y is std::initializer_list<int>
       auto z = {1, 2, 3.0}; // error: Cannot deduce element type
```

# 工作原理...

没有`auto`类型推断时，在使用大括号`{}`操作符初始化常规类型时，不会有太多令人惊讶的地方。当初始化容器如`std::vector`、`std::list`等时，大括号初始化将匹配该容器类的`std::initializer_list`构造函数。它以*贪婪*的方式进行匹配，这意味着不可能匹配非聚合构造函数（非聚合构造函数是通常的构造函数，与接受初始化列表的构造函数相对）。

`std::vector`，例如，提供了一个特定的非聚合构造函数，它可以用相同的值填充任意多个项目：`std::vector<int> v (N, value)`。当写成`std::vector<int> v {N, value}`时，将选择`initializer_list`构造函数，它将用两个项目`N`和`value`初始化向量。这是一个特殊的陷阱，人们应该知道。

与使用普通的`()`括号调用构造函数相比，`{}`操作符的一个好处是它们不进行隐式类型转换：`int x (1.2);` 和 `int x = 1.2;` 会将`x`初始化为值`1`，通过将浮点值四舍五入并将其转换为 int。相比之下，`int x {1.2};` 不会编译，因为它要*完全*匹配构造函数类型。

人们可以就哪种初始化样式是最好的进行有争议的讨论。

支持大括号初始化样式的人说，使用大括号使得变量被构造函数调用初始化非常明确，并且这行代码不会重新初始化任何东西。此外，使用`{}`大括号将选择唯一匹配的构造函数，而使用`()`括号的初始化行则尝试匹配最接近的构造函数，甚至进行类型转换以进行匹配。

C++17 引入的附加规则影响了使用`auto`类型推断的初始化--虽然 C++11 会正确地将变量`auto x {123};`的类型推断为只有一个元素的`std::initializer_list<int>`，但这很少是我们想要的。C++17 会将相同的变量推断为`int`。

经验法则：

+   `auto var_name {one_element};` 推断`var_name`与`one_element`的类型相同

+   `auto var_name {element1, element2, ...};` 是无效的，无法编译

+   `auto var_name = {element1, element2, ...};` 推断为一个`std::initializer_list<T>`，其中`T`与列表中所有元素的类型相同

C++17 使得意外定义初始化列表变得更加困难。

在 C++11/C++14 模式下尝试使用不同的编译器将会显示一些编译器实际上将`auto x {123};`推断为`int`，而其他编译器将其推断为`std::initializer_list<int>`。编写这样的代码可能会导致可移植性问题！

# 让构造函数自动推断出结果模板类的类型

C++中的许多类通常是专门针对类型进行特化的，这些类型可以很容易地从用户在构造函数调用中放入的变量类型中推断出来。然而，在 C++17 之前，这不是一个标准化的特性。C++17 允许编译器从构造函数调用中*自动*推断模板类型。

# 如何做...

这种情况的一个非常方便的用例是构造`std::pair`和`std::tuple`实例。这些可以在一步中进行专门化和实例化：

```cpp
std::pair  my_pair  (123, "abc");       // std::pair<int, const char*>
std::tuple my_tuple (123, 12.3, "abc"); // std::tuple<int, double,
                                        //            const char*>
```

# 它是如何工作的...

让我们定义一个示例类，其中自动模板类型推断将会有价值：

```cpp
template <typename T1, typename T2, typename T3>
class my_wrapper {
    T1 t1;
    T2 t2;
    T3 t3;

public:
    explicit my_wrapper(T1 t1_, T2 t2_, T3 t3_) 
        : t1{t1_}, t2{t2_}, t3{t3_}
    {}

    /* … */
};
```

好吧，这只是另一个模板类。以前我们必须这样写才能实例化它：

```cpp
my_wrapper<int, double, const char *> wrapper {123, 1.23, "abc"};
```

现在我们可以省略模板专门化部分：

```cpp
my_wrapper wrapper {123, 1.23, "abc"};
```

在 C++17 之前，只能通过实现*make 函数助手*来实现这一点：

```cpp
my_wrapper<T1, T2, T3> make_wrapper(T1 t1, T2 t2, T3 t3)
{
    return {t1, t2, t3};
}
```

使用这样的辅助函数，可以实现类似的效果：

```cpp
auto wrapper (make_wrapper(123, 1.23, "abc"));
```

STL 已经提供了许多类似的辅助函数，如`std::make_shared`、`std::make_unique`、`std::make_tuple`等。在 C++17 中，这些现在大多可以被视为过时。当然，它们将继续提供以确保兼容性。

# 还有更多...

我们刚刚学到的是*隐式模板类型推断*。在某些情况下，我们不能依赖隐式类型推断。考虑以下示例类：

```cpp
template <typename T>
struct sum {
    T value;

    template <typename ... Ts>
    sum(Ts&& ... values) : value{(values + ...)} {}
};
```

这个结构`sum`接受任意数量的参数，并使用折叠表达式将它们相加（稍后在本章中查看折叠表达式示例，以获取有关折叠表达式的更多详细信息）。结果的和保存在成员变量`value`中。现在的问题是，`T`是什么类型？如果我们不想明确指定它，它肯定需要依赖于构造函数中提供的值的类型。如果我们提供字符串实例，它需要是`std::string`。如果我们提供整数，它需要是`int`。如果我们提供整数、浮点数和双精度浮点数，编译器需要找出哪种类型适合所有值而不会丢失信息。为了实现这一点，我们提供了一个*显式推导指南*：

```cpp
template <typename ... Ts>
sum(Ts&& ... ts) -> sum<std::common_type_t<Ts...>>;
```

这个推导指南告诉编译器使用`std::common_type_t`特性，它能够找出适合所有值的公共类型。让我们看看如何使用它：

```cpp
sum s          {1u, 2.0, 3, 4.0f};
sum string_sum {std::string{"abc"}, "def"};

std::cout << s.value          << 'n'
          << string_sum.value << 'n';
```

在第一行中，我们使用`unsigned`，`double`，`int`和`float`类型的构造函数参数实例化了一个`sum`对象。`std::common_type_t`返回`double`作为公共类型，所以我们得到一个`sum<double>`实例。在第二行中，我们提供了一个`std::string`实例和一个 C 风格的字符串。根据我们的推导指南，编译器构造了一个`sum<std::string>`类型的实例。

运行此代码时，它将打印数字和字符串的和。

# 使用 constexpr-if 简化编译时决策

在模板化的代码中，通常需要根据模板专门化的类型来做一些不同的事情。C++17 带来了 constexpr-if 表达式，它大大简化了这种情况下的代码。

# 如何做...

在这个示例中，我们将实现一个小的辅助模板类。它可以处理不同的模板类型专门化，因为它能够根据我们为其专门化的类型在某些段落中选择完全不同的代码：

1.  编写通用部分的代码。在我们的例子中，这是一个简单的类，支持使用`add`函数将类型`U`的值添加到类型`T`的成员值中：

```cpp
       template <typename T>
       class addable
       { 
           T val;

       public:
           addable(T v) : val{v} {}

           template <typename U>
           T add(U x) const {
               return val + x;
           }
       };
```

1.  假设类型`T`是`std::vector<something>`，类型`U`只是`int`。将整数添加到整个向量意味着什么？我们说这意味着我们将整数添加到向量中的每个项目。这将在循环中完成：

```cpp
       template <typename U>
       T add(U x) 
       {
           auto copy (val); // Get a copy of the vector member
           for (auto &n : copy) { 
               n += x;
           }
           return copy;
       }
```

1.  下一步，也是最后一步是*结合*两个世界。如果`T`是`U`项的向量，则执行*循环*变体。如果不是，则只需实现*正常*的加法：

```cpp
       template <typename U>
       T add(U x) const {
           if constexpr (std::is_same_v<T, std::vector<U>>) {
               auto copy (val);
               for (auto &n : copy) { 
                   n += x;
               }
               return copy;
           } else {
               return val + x;
           }
       }

```

1.  现在可以使用该类。让我们看看它如何与完全不同的类型一起工作，例如`int`，`float`，`std::vector<int>`和`std::vector<string>`：

```cpp
       addable<int>{1}.add(2);               // is 3
       addable<float>{1.0}.add(2);           // is 3.0
       addable<std::string>{"aa"}.add("bb"); // is "aabb"

       std::vector<int> v {1, 2, 3};
       addable<std::vector<int>>{v}.add(10); 
           // is std::vector<int>{11, 12, 13}

       std::vector<std::string> sv {"a", "b", "c"};
       addable<std::vector<std::string>>{sv}.add(std::string{"z"}); 
           // is {"az", "bz", "cz"}
```

# 它是如何工作的...

新的 constexpr-if 的工作方式与通常的 if-else 结构完全相同。不同之处在于它测试的条件必须在*编译时*进行评估。编译器从我们的程序创建的所有运行时代码都不包含来自 constexpr-if 条件语句的任何分支指令。也可以说它的工作方式类似于预处理器`#if`和`#else`文本替换宏，但对于这些宏，代码甚至不需要在语法上是良好形式的。constexpr-if 结构的所有分支都需要*语法上良好形式*，但*不*采取的分支不需要*语义上有效*。

为了区分代码是否应该将值`x`添加到向量中，我们使用类型特征`std::is_same`。表达式`std::is_same<A, B>::value`在`A`和`B`是相同类型时求值为布尔值`true`。我们的条件是`std::is_same<T, std::vector<U>>::value`，如果用户将类专门化为`T = std::vector<X>`并尝试使用类型`U = X`的参数调用`add`，则求值为`true`。

当然，constexpr-if-else 块中可以有多个条件（注意`a`和`b`必须依赖于模板参数，而不仅仅是编译时常量）：

```cpp
if constexpr (a) {
    // do something
} else if constexpr (b) {
    // do something else 
} else {
    // do something completely different
}
```

使用 C++17，许多元编程情况更容易表达和阅读。

# 还有更多...

为了说明 constexpr-if 结构对 C++的改进有多大，我们可以看看在 C++17*之前*如何实现相同的事情：

```cpp
template <typename T>
class addable
{
    T val;

public:
    addable(T v) : val{v} {}

    template <typename U>
 std::enable_if_t<!std::is_same<T, std::vector<U>>::value, T>
    add(U x) const { return val + x; }

    template <typename U>
 std::enable_if_t<std::is_same<T, std::vector<U>>::value, 
                     std::vector<U>>
    add(U x) const {
        auto copy (val);
        for (auto &n : copy) { 
            n += x;
        }
        return copy;
    }
};
```

在不使用 constexpr-if 的情况下，这个类适用于我们希望的所有不同类型，但看起来非常复杂。它是如何工作的？

*两个不同*`add`函数的实现看起来很简单。它们的返回类型声明使它们看起来复杂，并且包含一个技巧--例如`std::enable_if_t<condition, type>`表达式在`condition`为`true`时评估为`type`。否则，`std::enable_if_t`表达式不会评估为任何东西。这通常被认为是一个错误，但我们将看到为什么它不是。

对于第二个`add`函数，相同的条件以*反转*的方式使用。这样，它只能同时对两个实现中的一个为`true`。

当编译器看到具有相同名称的不同模板函数并且必须选择其中一个时，一个重要的原则就会发挥作用：**SFINAE**，它代表**替换失败不是错误**。在这种情况下，这意味着如果其中一个函数的返回值无法从错误的模板表达式中推导出（如果其条件评估为`false`，则`std::enable_if`是错误的），则编译器不会报错。它将简单地继续寻找并尝试*其他*函数实现。这就是诀窍；这就是它是如何工作的。

真是麻烦。很高兴看到这在 C++17 中变得如此容易。

# 使用内联变量启用仅头文件库

虽然在 C++中一直可以声明单独的函数*内联*，但 C++17 还允许我们声明*变量*内联。这使得实现*仅头文件*库变得更加容易，这在以前只能使用变通方法实现。

# 它是如何实现的...

在这个示例中，我们创建了一个示例类，它可以作为典型的仅头文件库的成员。目标是使用`inline`关键字以静态成员的方式实例化它，并以全局可用的方式使用它，这在 C++17 之前是不可能的。

1.  `process_monitor`类应该同时包含一个静态成员并且本身应该是全局可访问的，这将在从多个翻译单元包含时产生双重定义的符号：

```cpp
       // foo_lib.hpp 

       class process_monitor { 
       public: 
           static const std::string standard_string 
               {"some static globally available string"}; 
       };

       process_monitor global_process_monitor;
```

1.  如果我们现在在多个`.cpp`文件中包含这个以便编译和链接它们，这将在链接阶段失败。为了解决这个问题，我们添加`inline`关键字：

```cpp
       // foo_lib.hpp 

       class process_monitor { 
       public: 
           static const inline std::string standard_string 
               {"some static globally available string"}; 
       };

       inline process_monitor global_process_monitor;
```

看，就是这样！

# 它是如何工作的...

C++程序通常由多个 C++源文件组成（这些文件具有`.cpp`或`.cc`后缀）。这些文件被单独编译为模块/对象文件（通常具有.o 后缀）。然后将所有模块/对象文件链接在一起成为单个可执行文件或共享/静态库是最后一步。

在链接阶段，如果链接器可以找到一个特定符号的定义*多次*，则被视为错误。例如，我们有一个带有`int foo();`签名的函数。如果两个模块定义了相同的函数，那么哪一个是正确的？链接器不能随意选择。嗯，它可以，但这很可能不是任何程序员想要发生的事情。

提供全局可用函数的传统方法是在头文件中*声明*它们，这些头文件将被任何需要调用它们的 C++模块包含。然后，这些函数的定义将被放入单独的模块文件中*一次*。然后，这些模块与希望使用这些函数的模块一起链接在一起。这也被称为**一次定义规则**（**ODR**）。查看以下插图以更好地理解：

![](img/ee850b95-1991-4682-a5d1-1c7290509001.png)

然而，如果这是唯一的方法，那么就不可能提供仅包含头文件的库。仅包含头文件的库非常方便，因为它们只需要使用`#include`包含到任何 C++程序文件中，然后立即可用。为了使用不是仅包含头文件的库，程序员还必须调整构建脚本，以便链接器将库模块与自己的模块文件一起链接。特别是对于只有非常短函数的库，这是不必要的不舒服。

对于这种情况，`inline`关键字可以用来做一个例外，以允许在不同模块中*多次*定义相同的符号。如果链接器找到具有相同签名的多个符号，但它们被声明为内联，它将只选择第一个并相信其他符号具有相同的定义。所有相等的内联符号都完全相等的定义基本上是程序员的*承诺*。

关于我们的 reciple 示例，链接器将在每个包含`foo_lib.hpp`的模块中找到`process_monitor::standard_string`符号。没有`inline`关键字，它将不知道选择哪一个，因此它将中止并报告错误。对`global_process_monitor`符号也是一样。哪一个才是正确的？

在声明两个符号`inline`后，它将只接受每个符号的第一次出现，并*丢弃*所有其他出现。

在 C++17 之前，唯一的干净方法是通过额外的 C++模块文件提供此符号，这将迫使我们的库用户在链接步骤中包含此文件。

`inline`关键字传统上还有*另一个*功能。它告诉编译器可以通过获取其实现并直接将其放在调用它的地方来*消除*函数调用。这样，调用代码包含一个函数调用少，这通常被认为更快。如果函数非常短，生成的汇编代码也会更短（假设执行函数调用的指令数量，保存和恢复堆栈等比实际有效载荷代码更高）。如果内联函数非常长，二进制大小将增长，这有时甚至可能不会导致最终更快的代码。

因此，编译器只会将`inline`关键字作为提示，并可能通过内联来消除函数调用。但它也可以内联一些函数，*而不需要*程序员声明为内联。

# 还有更多...

在 C++17 之前的一个可能的解决方法是提供一个`static`函数，它返回一个`static`对象的引用：

```cpp
class foo {
public:
    static std::string& standard_string() {
        static std::string s {"some standard string"};
        return s;
    }
};
```

这样，将头文件包含在多个模块中是完全合法的，但仍然可以在任何地方访问到完全相同的实例。然而，对象并不是在程序开始时立即构造的，而是只有在第一次调用此 getter 函数时才会构造。对于某些用例，这确实是一个问题。想象一下，我们希望静态的全局可用对象的构造函数在*程序开始*时做一些重要的事情（就像我们的 reciple 示例库类），但由于 getter 在程序结束时被调用，这就太晚了。

另一个解决方法是将非模板类`foo`变为模板类，这样它就可以从与模板相同的规则中获益。

这两种策略在 C++17 中都可以避免。

# 使用折叠表达式实现方便的辅助函数

自 C++11 以来，有可变模板参数包，它们使得实现接受任意多个参数的函数成为可能。有时，这些参数都被合并成一个表达式，以便从中导出函数结果。这在 C++17 中变得非常容易，因为它带有折叠表达式。

# 如何做...

让我们实现一个函数，它接受任意多个参数并返回它们的总和：

1.  首先，我们定义它的签名：

```cpp
      template <typename ... Ts>
      auto sum(Ts ... ts);
```

1.  所以，现在我们有一个参数包`ts`，函数应该展开所有参数并使用折叠表达式将它们相加。如果我们使用任何操作符（在这个例子中是`+`）与`...`一起，以便将其应用于参数包的所有值，我们需要用括号括起表达式：

```cpp
      template <typename ... Ts>
      auto sum(Ts ... ts)
      {
          return (ts + ...);
      }
```

1.  我们现在可以这样调用它：

```cpp
      int the_sum {sum(1, 2, 3, 4, 5)}; // Value: 15
```

1.  它不仅适用于`int`类型；我们可以用任何实现了`+`运算符的类型来调用它，比如`std::string`：

```cpp
      std::string a {"Hello "};
      std::string b {"World"};

      std::cout << sum(a, b) << 'n'; // Output: Hello World
```

# 它是如何工作的...

我们刚刚做的是对其参数进行简单的递归应用二元运算符(`+`)。这通常被称为*折叠*。C++17 带有**折叠表达式**，它可以用更少的代码表达相同的想法。

这种类型的表达式称为**一元折叠**。C++17 支持使用以下二元操作符对参数包进行折叠：`+`、`-`、`*`、`/`、`%`、`^`、`&`、`|`、`=`、`<`、`>`、`<<`、`>>`、`+=`、`-=`、`*=`、`/=`、`%=`、`^=`、`&=`、`|=`、`<<=`、`>>=`、`==`、`!=`、`<=`、`>=`、`&&`、`||`、`,`、`.*`、`->*`。

顺便说一句，在我们的示例代码中，如果我们写`(ts + ...)`或`(… + ts)`都没有关系；两者都可以。然而，在其他情况下可能会有所不同--如果`…`点在操作符的*右侧*，则折叠称为*右*折叠。如果它们在*左侧*，则是*左*折叠。

在我们的`sum`示例中，一元左折叠展开为`1 + (2 + (3 + (4 + 5)))`，而一元右折叠将展开为`(((1 + 2) + 3) + 4) + 5`。根据使用的操作符，这可能会有所不同。当添加数字时，它并不会有所不同。

# 还有更多...

如果有人用*没有*参数调用`sum()`，则变参参数包不包含可以折叠的值。对于大多数操作符来说，这是一个错误（对于一些操作符来说不是；我们将在一分钟内看到）。然后我们需要决定这是否应该保持为错误，或者空的总和是否应该导致特定的值。显而易见的想法是，什么都没有的总和是`0`。

这就是它的实现方式：

```cpp
template <typename ... Ts>
auto sum(Ts ... ts)
{
    return (ts + ... + 0);
}
```

这样，`sum()`的结果是`0`，`sum(1, 2, 3)`的结果是`(1 + (2 + (3 + 0)))`。这种带有初始值的折叠称为**二进制折叠**。

同样，如果我们写`(ts + ... + 0)`或`(0 + ... + ts)`，它也可以工作，但这会使二进制折叠再次成为二进制*右*折叠或二进制*左*折叠。看看下面的图表：

![](img/4c518bfa-0a12-435d-820f-0199ee897ce3.png)

当使用二进制折叠来实现无参数情况时，*单位*元素的概念通常很重要--在这种情况下，将`0`添加到任何数字都不会改变任何东西，这使`0`成为单位元素。由于这个属性，我们可以使用`+`或`-`运算符将`0`添加到任何折叠表达式中，这将导致在参数包中没有参数的情况下结果为`0`。从数学的角度来看，这是正确的。从实现的角度来看，我们需要根据需要定义什么是正确的。

相同的原则适用于乘法。在这里，单位元素是`1`：

```cpp
template <typename ... Ts>
auto product(Ts ... ts)
{
    return (ts * ... * 1);
}
```

`product(2, 3)`的结果是`6`，没有参数的`product()`的结果是`1`。

逻辑**和**(`&&`)和**或**(`||`)操作符带有*内置*单位元素。使用`&&`对空参数包进行折叠的结果是`true`，使用`||`对空参数包进行折叠的结果是`false`。

另一个操作符，当应用于空参数包时默认为某个表达式的逗号操作符（`,`），然后默认为`void()`。

为了激发一些灵感，让我们看看我们可以使用这个特性实现的一些更多的小助手。

# 匹配范围与单个项目

如何编写一个函数，告诉我们某个范围是否包含我们提供的变参参数中的*至少一个*值：

```cpp
template <typename R, typename ... Ts>
auto matches(const R& range, Ts ... ts)
{
    return (std::count(std::begin(range), std::end(range), ts) + ...);
}
```

帮助函数使用 STL 中的`std::count`函数。该函数接受三个参数：前两个参数是某个可迭代范围的*begin*和*end*迭代器，作为第三个参数，它接受一个*value*，该值将与范围内的所有项目进行比较。然后，`std::count`方法返回范围内等于第三个参数的所有元素的数量。

在我们的折叠表达式中，我们总是将相同参数范围的*begin*和*end*迭代器传递给`std::count`函数。然而，作为第三个参数，每次我们都将参数包中的另一个参数放入其中。最后，函数将所有结果相加并将其返回给调用者。

我们可以这样使用它：

```cpp
std::vector<int> v {1, 2, 3, 4, 5};

matches(v,         2, 5);          // returns 2
matches(v,         100, 200);      // returns 0
matches("abcdefg", 'x', 'y', 'z'); // returns 0
matches("abcdefg", 'a', 'd', 'f'); // returns 3
```

正如我们所看到的，`matches`帮助函数非常灵活--它可以直接在向量或字符串上调用。它还可以在初始化列表、`std::list`、`std::array`、`std::set`等实例上工作！

# 检查多次插入集合是否成功

让我们编写一个帮助函数，将任意数量的可变参数插入到`std::set`中，并返回所有插入是否*成功*：

```cpp
template <typename T, typename ... Ts>
bool insert_all(T &set, Ts ... ts)
{
    return (set.insert(ts).second && ...);
}
```

那么，这是如何工作的呢？`std::set`的`insert`函数具有以下签名：

```cpp
std::pair<iterator, bool> insert(const value_type& value);
```

文档表示，当我们尝试插入一个项目时，`insert`函数将返回一个对中的`iterator`和`bool`变量。如果插入成功，`bool`值为`true`。如果成功，迭代器指向集合中的*新元素*。否则，迭代器指向*现有*项目，它将与要插入的项目*冲突*。

我们的帮助函数在插入后访问`.second`字段，这只是反映成功或失败的`bool`变量。如果所有插入在所有返回对中都导致`true`，那么所有插入都成功了。折叠表达式使用`&&`运算符将所有插入结果组合在一起并返回结果。

我们可以这样使用它：

```cpp
std::set<int> my_set {1, 2, 3};

insert_all(my_set, 4, 5, 6); // Returns true
insert_all(my_set, 7, 8, 2); // Returns false, because the 2 collides
```

请注意，如果我们尝试插入三个元素，但第二个元素已经无法插入，`&& ...`折叠将会短路并停止插入所有其他元素：

```cpp
std::set<int> my_set {1, 2, 3};

insert_all(my_set, 4, 2, 5); // Returns false
// set contains {1, 2, 3, 4} now, without the 5!

```

# 检查所有参数是否在某个范围内

如果我们可以检查*一个*变量是否在某个特定范围内，我们也可以使用折叠表达式来对*多个*变量执行相同的操作。

```cpp
template <typename T, typename ... Ts>
bool within(T min, T max, Ts ...ts)
{
    return ((min <= ts && ts <= max) && ...);
}
```

表达式`(min <= ts && ts <= max)`确实告诉了参数包的每个值是否在`min`和`max`之间（*包括*`min`和`max`）。我们选择`&&`运算符将所有布尔结果减少为单个结果，只有当所有个别结果都为`true`时才为`true`。

这就是它的实际效果：

```cpp
within( 10,  20,  1, 15, 30);    // --> false
within( 10,  20,  11, 12, 13);   // --> true
within(5.0, 5.5,  5.1, 5.2, 5.3) // --> true
```

有趣的是，这个函数非常灵活，因为它对我们使用的类型的唯一要求是它们可以使用`<=`运算符进行比较。例如，`std::string`也满足这个要求：

```cpp
std::string aaa {"aaa"};
std::string bcd {"bcd"};
std::string def {"def"};
std::string zzz {"zzz"};

within(aaa, zzz,  bcd, def); // --> true
within(aaa, def,  bcd, zzz); // --> false
```

# 将多个项目推入向量

还可以编写一个不减少任何结果但处理相同类型的多个操作的帮助函数。比如将项目插入到`std::vector`中，它不返回任何结果（`std::vector::insert()`通过抛出异常来表示错误）：

```cpp
template <typename T, typename ... Ts>
void insert_all(std::vector<T> &vec, Ts ... ts)
{
    (vec.push_back(ts), ...);
}

int main()
{
    std::vector<int> v {1, 2, 3};
    insert_all(v, 4, 5, 6);
}
```

请注意，我们使用逗号（`,`）运算符来将参数包展开为单独的`vec.push_back(...)`调用，而不是折叠实际结果。这个函数也很好地处理了*空*参数包，因为逗号运算符具有隐式的单位元素`void()`，它转换为*什么也不做*。
