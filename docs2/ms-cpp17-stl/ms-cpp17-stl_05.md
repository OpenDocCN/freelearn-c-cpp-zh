# 第五章：词汇类型

在过去十年中，人们越来越认识到，标准语言或标准库的一个重要角色是提供*词汇类型*。一个“词汇”类型是一个声称为处理其领域提供一个单一*通用语言*的类型，一个共同的语言。

注意，甚至在 C++存在之前，C 编程语言就已经在某个领域的词汇方面做出了相当不错的尝试，为整数数学（`int`）、浮点数学（`double`）、以 Unix 纪元表示的时间点（`time_t`）和字节计数（`size_t`）提供了标准类型或类型别名。

在本章中，我们将学习：

+   C++中词汇类型的演变历史，从`std::string`到`std::any`

+   代数数据类型、乘积类型和求和类型的定义

+   如何操作元组和访问变体

+   `std::optional<T>`作为“可能有`T`”或“尚未有`T`”的作用

+   `std::any`作为“无限”的代数数据类型等价物

+   如何实现类型擦除，它在`std::any`和`std::function`中的使用以及其固有的限制

+   `std::function`的一些陷阱以及修复它们的第三方库

# `std::string`的故事

考虑字符字符串的领域；例如，短语`hello world`。在 C 中，处理字符串的通用语言是`char *`：

```cpp
    char *greet(const char *name) {
      char buffer[100];
      snprintf(buffer, 100, "hello %s", name);
      return strdup(buffer);
    }

    void test() {
      const char *who = "world";
      char *hw = greet(who);
      assert(strcmp(hw, "hello world") == 0);
      free(hw);
    }
```

这在一段时间内是可行的，但是处理原始的`char *`对于语言的用户以及第三方库和例程的创建者来说存在一些问题。一方面，C 语言如此古老，以至于一开始就没有发明出`const`，这意味着某些旧的例程会期望它们的字符串为`char *`，而某些较新的则期望`const char *`。另一方面，`char *`没有携带一个*长度*；因此，一些函数期望一个指针和一个长度，而一些函数只期望指针，并且根本无法处理值`'\0'`嵌入的字节。

`char *`谜团中缺失的最重要部分是*生命周期管理*和*所有权*（如第四章，*容器动物园*开头所述）。当一个 C 函数想要从其调用者那里接收一个字符串时，它接受`char *`，并且通常将字符的所有权管理留给调用者。但是，如果它想要*返回*一个字符串呢？那么它必须返回`char *`并希望调用者记得释放它（`strdup`、`asprintf`），或者从调用者那里接收一个缓冲区并希望它足够大以容纳输出（`sprintf`、`snprintf`、`strcat`）。在 C（以及在预标准的 C++）中管理字符串所有权的困难如此之大，以至于出现了大量的“字符串库”来解决这个问题：Qt 的`QString`、glib 的`GString`等等。

1998 年，C++ 以一个奇迹般的方式进入了这个混乱：一个 *标准* 字符串类！新的 `std::string` 以自然的方式封装了字符串的字节和长度；它可以正确处理嵌入的空字节；它支持以前复杂的操作，如 `hello + world`，通过静默地分配所需的精确内存量；而且由于 RAII，它永远不会泄漏内存或引起关于谁拥有底层字节的混淆。最好的是，它从 `char *` 有隐式转换：

```cpp
    std::string greet(const std::string& name) {
      return "hello " + name;
    }

    void test() {
      std::string who = "world";
      assert(greet(who) == "hello world");
    }
```

现在，C++ 函数处理字符串（如前述代码中的 `greet()`）可以接受 `std::string` 参数并返回 `std::string` 结果。甚至更好，因为字符串类型是 *标准化的*，几年后你可能会相当有信心，当你选择一些第三方库将其集成到你的代码库中时，它的任何接受字符串（文件名、错误消息等）的函数都会使用 `std::string`。通过共享 `std::string` 的 *通用语言*，每个人都可以更有效地进行沟通。

# 使用 reference_wrapper 标记引用类型

在 C++03 中引入的另一个词汇类型是 `std::reference_wrapper<T>`。它有一个简单的实现：

```cpp
    namespace std {
      template<typename T>
      class reference_wrapper {
        T *m_ptr;
        public:
        reference_wrapper(T& t) noexcept : m_ptr(&t) {}

        operator T& () const noexcept { return *m_ptr; }
        T& get() const noexcept { return *m_ptr; }
      };

      template<typename T>
      reference_wrapper<T> ref(T& t);
    } // namespace std
```

`std::reference_wrapper` 的用途与 `std::string` 和 `int` 等词汇类型略有不同；它专门作为将我们希望在其上下文中作为引用行为的“标记”值的方式：

```cpp
     int result = 0;
     auto task = [](int& r) {
       r = 42;
     };

     // Trying to use a native reference wouldn't compile.
     //std::thread t(task, result);

     // Correctly pass result "by reference" to the new thread.
     std::thread t(task, std::ref(result));
```

`std::thread` 的构造函数编写了特定的特殊情况来处理 `reference_wrapper` 参数，通过“退化”为原生引用来处理。相同的特殊情况适用于标准库函数 `make_pair`、`make_tuple`、`bind`、`invoke` 以及基于 `invoke` 的所有内容（如 `std::apply`、`std::function::operator()` 和 `std::async`）。

# C++11 和代数类型

随着 C++11 的形成，越来越多的人认识到另一个适合词汇化的领域是所谓的 *代数数据类型*。代数类型在函数式编程范式中自然出现。基本思想是考虑类型的域——即该类型所有可能值的集合。为了使事情简单，你可能想要考虑 C++ 的 `enum` 类型，因为很容易谈论 `enum` 类型的对象在某个时刻可能具有的不同值的数量：

```cpp
    enum class Color {
      RED = 1,
      BLACK = 2,
    };

    enum class Size {
      SMALL = 1,
      MEDIUM = 2,
      LARGE = 3,
    };
```

给定类型 `Color` 和 `Size`，你能创建一个实例可能具有 2 × 3 = 6 个值的类型吗？是的；这种类型代表 `Color` 和 `Size` 的“每个都只有一个”，被称为 *积类型*，因为其可能值的集合是其元素可能值集合的 *笛卡尔积*。

那么一个实例可能具有 2 + 3 = 5 个不同值的类型呢？也是；这种类型表示“要么是 `Color` 或 `Size`，但不会同时两者都是”，这被称为 *求和类型*。（令人困惑的是，数学家并不使用 *笛卡尔和* 这个术语来表示这个概念。）

在像 Haskell 这样的函数式编程语言中，这两个练习的拼写如下：

```cpp
    data SixType = ColorandSizeOf Color Size;
    data FiveType = ColorOf Color | SizeOf Size;
```

在 C++ 中，它们的拼写如下：

```cpp
    using sixtype = std::pair<Color, Size>;
    using fivetype = std::variant<Color, Size>;
```

类模板 `std::pair<A, B>` 表示一个有序元素对：一个类型为 `A` 的元素，后面跟着一个类型为 `B` 的元素。它与一个包含两个元素的普通 `struct` 非常相似，只是你不需要自己编写 `struct` 定义：

```cpp
    template<class A, class B>
    struct pair {
      A first;
      B second;
    };
```

注意到 `std::pair<A, A>` 和 `std::array<A, 2>` 之间只有细微的表面差异。我们可能会说 `pair` 是 `array` 的一个 *异构* 版本（除了 `pair` 只能持有两个元素的限制）。

# 使用 std::tuple

C++11 引入了一个完整的异构数组；它被称为 `std::tuple<Ts...>`。仅包含两种元素类型的元组——例如，`tuple<int, double>`——与 `pair<int, double>` 没有区别。但元组可以持有比一对元素更多的内容；通过 C++11 可变参数模板的魔力，它们可以持有三元组、四元组、五元组等，因此具有通用的名称 `tuple`。例如，`tuple<int, int, char, std::string>` 与一个成员分别为 `int`、另一个 `int`、一个 `char` 和最后的 `std::string` 的 `struct` 相似。

因为元组的第一个元素与第二个元素类型不同，我们不能使用“正常”的 `operator[](size_t)` 通过可能随运行时变化的索引来访问元素。相反，我们必须在 *编译时* 告诉编译器我们打算访问元组的哪个元素，这样编译器才能确定要给表达式赋予什么类型。C++ 在编译时提供信息的方法是通过模板参数强制将其纳入类型系统，这就是我们这样做的原因。当我们想访问元组 `t` 的第一个元素时，我们调用 `std::get<0>(t)`。要访问第二个元素，我们调用 `std::get<1>(t)`，依此类推。

这就成为了处理 `std::tuple` 的模式——在具有访问和操作它们的 *成员函数* 的同构容器类型中，异构代数类型倾向于有 *自由函数模板* 用于访问和操作它们。

然而，一般来说，你不会对元组进行很多**操作**。它们的主要用途，除了模板元编程之外，是在需要单个值的上下文中以经济的方式暂时将多个值绑定在一起。例如，你可能还记得从第四章[part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d]中“最简单的容器”部分中的示例中了解到的`std::tie`。这是一种将任意数量的值绑定到单个单元的便宜方法，该单元可以用`operator<`进行字典序比较。字典序比较的“感觉”取决于你绑定值的顺序：

```cpp
    using Author = std::pair<std::string, std::string>;
    std::vector<Author> authors = {
      {"Fyodor", "Dostoevsky"},
      {"Sylvia", "Plath"},
      {"Vladimir", "Nabokov"},
      {"Douglas", "Hofstadter"},
    };

    // Sort by first name then last name.
    std::sort(
      authors.begin(), authors.end(),
      [](auto&& a, auto&& b) {
        return std::tie(a.first, a.second) < std::tie(b.first, b.second);
      }
    );
    assert(authors[0] == Author("Douglas", "Hofstadter"));

    // Sort by last name then first name.
    std::sort(
      authors.begin(), authors.end(),
      [](auto&& a, auto&& b) {
        return std::tie(a.second, a.first) < std::tie(b.second, b.first);
      }
    );
    assert(authors[0] == Author("Fyodor", "Dostoevsky"));
```

`std::tie`之所以如此便宜，是因为它实际上创建了一个对其参数内存位置的**引用**元组，而不是复制其参数的值。这导致了`std::tie`的第二种常见用途：模拟像 Python 这样的语言中发现的“多重赋值”：

```cpp
    std::string s;
    int i;

    // Assign both s and i at once.
    std::tie(s, i) = std::make_tuple("hello", 42);
```

注意，前述注释中的“一次”短语与并发（见第七章[part0108.html#36VSO0-2fdac365b8984feebddfbb9250eaf20d]，*并发*）或副作用执行的顺序无关；我的意思是，两个值可以在单个赋值语句中赋值，而不是占用两行或多行。

如前例所示，`std::make_tuple(a, b, c...)`可以用来创建一个包含**值**的元组；也就是说，`make_tuple`确实会构造其参数值的副本，而不仅仅是获取它们的地址。

最后，在 C++17 中，我们可以使用构造函数模板参数推导来简单地编写`std::tuple(a, b, c...)`；但除非你确切知道你想要它的行为，否则最好避免使用这个特性。模板参数推导与`std::make_tuple`的不同之处仅在于它将保留`std::reference_wrapper`参数而不是将它们退化到原生 C++引用：

```cpp
    auto [i, j, k] = std::tuple{1, 2, 3};

    // make_tuple decays reference_wrapper...
    auto t1 = std::make_tuple(i, std::ref(j), k);
    static_assert(std::is_same_v< decltype(t1),
      std::tuple<int, int&, int>
    >);

    // ...whereas the deduced constructor does not.
    auto t2 = std::tuple(i, std::ref(j), k);
    static_assert(std::is_same_v< decltype(t2),
      std::tuple<int, std::reference_wrapper<int>, int>
    >);
```

# 操作元组值

大多数这些函数和模板仅在模板元编程的上下文中有用；你不太可能每天都会使用它们：

+   `std::get<I>(t)`: 获取对`t`的第`I`个元素的引用。

+   `std::tuple_size_v<decltype(t)>`: 表示给定元组的**大小**。因为这是元组类型的编译时常量属性，所以它被表示为一个以该类型为参数的变量模板。如果你更愿意使用更自然的语法，你可以以以下两种方式之一编写辅助函数：

```cpp
        template<class T>
        constexpr size_t tuple_size(T&&)
        {
          return std::tuple_size_v<std::remove_reference_t<T>>;
        }

        template<class... Ts>
        constexpr size_t simpler_tuple_size(const std::tuple<Ts...>&)
        {
          return sizeof...(Ts);
        }
```

+   `std::tuple_element_t<I, decltype(t)>`: 表示给定元组类型的第`I`个元素的**类型**。同样，标准库以一种比核心语言更不优雅的方式公开了这项信息。通常，要找到元组的第`I`个元素的类型，你只需编写`decltype(std::get<I>(t))`。

+   `std::tuple_cat(t1, t2, t3...)`: 将所有给定的元组从头到尾连接起来。

+   `std::forward_as_tuple(a, b, c...)`: 创建一个引用元组，就像`std::tie`；但与`std::tie`要求左值引用不同，`std::forward_as_tuple`将接受任何类型的引用作为输入，并将它们完美地转发到元组中，以便稍后可以通过`std::get<I>(t)...`提取它们：

```cpp
        template<typename F>
        void run_zeroarg(const F& f);

        template<typename F, typename... Args>
        void run_multiarg(const F& f, Args&&... args)
        {
          auto fwd_args =
            std::forward_as_tuple(std::forward<Args>(args)...);
          auto lambda = [&f, fwd_args]() {
            std::apply(f, fwd_args);
          };
          run_zeroarg(f);
        }
```

# 关于命名类的说明

正如我们在第四章“容器动物园”中看到的那样，当我们比较`std::array<double, 3>`和`struct Vec3`时，使用 STL 类模板可以缩短你的开发时间，并通过重用经过良好测试的 STL 组件来消除错误来源；但它也可能使你的代码可读性降低或给你的类型赋予*过多的*功能。在我们的第四章“容器动物园”的例子中，`std::array<double, 3>`对于`Vec3`来说是一个糟糕的选择，因为它暴露了一个不想要的`operator<`。

在你的接口和 API 中使用任何代数类型（`tuple`、`pair`、`optional`或`variant`）可能是错误的。你会发现，如果你为自己的“特定领域词汇”类型编写命名类，你的代码将更容易阅读、理解和维护，即使它们最终只是代数类型的薄包装——尤其是如果它们最终只是代数类型的薄包装。

# 使用 std::variant 表达备选方案

而`std::tuple<A,B,C>`是一个*积类型*，`std::variant<A,B,C>`是一个*和类型*。一个变体可以同时持有`A`、`B`或`C`中的一个——但一次不能同时持有（或少于）一个。这个概念的另一个名字是*有区别的联合*，因为变体在很多方面都像原生 C++的`union`；但与原生`union`不同，变体总是能够告诉你它的哪个元素，`A`、`B`或`C`，在任何给定时间点是“活跃”的。这些元素的官方名称是“备选方案”，因为一次只能有一个是活跃的：

```cpp
    std::variant<int, double> v1;

    v1 = 1; // activate the "int" member
    assert(v1.index() == 0);
    assert(std::get<0>(v1) == 1);

    v1 = 3.14; // activate the "double" member
    assert(v1.index() == 1);
    assert(std::get<1>(v1) == 3.14);
    assert(std::get<double>(v1) == 3.14);

    assert(std::holds_alternative<int>(v1) == false);
    assert(std::holds_alternative<double>(v1) == true);

    assert(std::get_if<int>(&v1) == nullptr);
    assert(*std::get_if<double>(&v1) == 3.14);
```

与`tuple`一样，你可以使用`std::get<I>(v)`获取`variant`的特定元素。如果你的变体对象的备选方案都是不同的（除非你在进行深度元编程，这应该是最常见的用例），你可以使用`std::get<T>(v)`与类型以及索引一起使用——例如，查看前面的代码示例，其中`std::get<0>(v1)`和`std::get<int>(v1)`可以互换使用，因为变体`v1`中的零索引备选方案是`int`类型。然而，与`tuple`不同，变体上的`std::get`允许失败！如果你在`v1`当前持有`int`类型值时调用`std::get<double>(v1)`，那么你会得到一个`std::bad_variant_access`类型的异常。`std::get_if`是`std::get`的“非抛出”版本。正如前面的示例所示，如果指定的备选方案是活跃的，`get_if`返回指向该备选方案的指针，否则返回空指针。因此，以下代码片段都是等效的：

```cpp
    // Worst...
    try {
      std::cout << std::get<int>(v1) << std::endl;
    } catch (const std::bad_variant_access&) {}

    // Still bad...
    if (v1.index() == 0) {
      std::cout << std::get<int>(v1) << std::endl; 
    }

    // Slightly better... 
    if (std::holds_alternative<int>(v1)) {
      std::cout << std::get<int>(v1) << std::endl;
    } 

    // ...Best.
    if (int *p = std::get_if<int>(&v1)) {
      std::cout << *p << std::endl; 
    }
```

# 访问变体

在前面的例子中，我们展示了当有一个变量 `std::variant<int, double> v` 时，调用 `std::get<double>(v)` 会给我们当前的值，前提是变体当前持有 `double`，但如果变体持有 `int`，则会抛出异常。这可能会让你觉得有些奇怪——因为 `int` 可以转换为 `double`，为什么它不能直接给我们转换后的值呢？

如果我们想要这种行为，我们不能从 `std::get` 获取。我们必须以这种方式重新表达我们的需求：“我有一个变体。如果它当前持有 `double`，称为 `d`，那么我想获取 `double(d)`。如果它持有 `int i`，那么我想获取 `double(i)`。”也就是说，我们有一个行为列表在心中，我们想要在当前由我们的变体 `v` 持有的任何替代方案上调用这其中的一个行为。标准库通过可能有些晦涩的名字 `std::visit` 来表达这个算法：

```cpp
    struct Visitor {
      double operator()(double d) { return d; }
      double operator()(int i) { return double(i); }
      double operator()(const std::string&) { return -1; }
    };

    using Var = std::variant<int, double, std::string>;

    void show(Var v)
    {
      std::cout << std::visit(Visitor{}, v) << std::endl;
    }

    void test() 
    {
      show(3.14);
      show(1);
      show("hello world");
    }
```

一般而言，当我们 `visit` 一个变体时，我们心中所想的全部行为在本质上都是相似的。因为我们是用 C++ 编写的，它具有函数和运算符的重载，我们可以一般地使用完全相同的语法来表达我们的相似行为。如果我们可以用相同的语法来表达它们，我们就可以将它们封装到一个模板函数中，或者——最常见的情况——一个 C++14 泛型 lambda，如下所示：

```cpp
    std::visit([](const auto& alt) {
      if constexpr (std::is_same_v<decltype(alt), const std::string&>) {
        std::cout << double(-1) << std::endl;
      } else {
        std::cout << double(alt) << std::endl;
      }
    }, v);
```

注意到使用了 C++17 的 `if constexpr` 来处理与其他情况根本不同的一种情况。是否更喜欢使用这种明确的 `decltype` 切换，或者创建一个辅助类，例如前面代码示例中的 `Visitor`，并依赖重载解析来选择每个可能替代的 `operator()` 的正确重载，这更多是一个个人喜好问题。

`std::visit` 也有一个可变参数版本，它接受两个、三个甚至更多的 `variant` 对象，这些对象可以是相同类型或不同类型。这个版本的 `std::visit` 可以用来实现一种“多重分派”，如下面的代码所示。然而，除非你正在进行真正的密集型元编程，否则你几乎肯定不需要这个版本的 `std::visit`：

```cpp
    struct MultiVisitor {
      template<class T, class U, class V>
      void operator()(T, U, V) const { puts("wrong"); }

      void operator()(char, int, double) const { puts("right!"); }
    };

    void test()
    {
      std::variant<int, double, char> v1 = 'x';
      std::variant<char, int, double> v2 = 1;
      std::variant<double, char, int> v3 = 3.14;
      std::visit(MultiVisitor{}, v1, v2, v3); // prints "right!"
    }
```

# 那么 `make_variant` 呢？以及关于值语义的注意事项

由于你可以使用 `std::make_tuple` 创建一个元组对象，或者使用 `make_pair` 创建一个对，你可能会合理地问：“`make_variant` 呢？”实际上，并没有这样的函数。它不存在的主要原因在于，虽然 `tuple` 和 `pair` 是积类型，但 `variant` 是和类型。要创建一个元组，你必须始终提供其所有 *n* 个元素的值，因此元素类型总是可以推断出来的。对于 `variant`，你只需要提供其一个值——假设为类型 `A`——但编译器在不知道类型 `B` 和 `C` 的身份的情况下，无法创建一个 `variant<A,B,C>` 对象。因此，提供 `my::make_variant<A,B,C>(a)` 这样的函数是没有意义的，因为实际的类构造函数可以更简洁地写成：`std::variant<A,B,C>(a)`。

我们已经提到了`make_pair`和`make_tuple`存在的次要原因：它们自动将特殊词汇类型`std::reference_wrapper<T>`衰减为`T&`，因此`std::make_pair(std::ref(a), std::cref(b))`创建了一个类型为`std::pair<A&, const B&>`的对象。具有“引用对”或“引用元组”类型的对象表现得非常奇怪：你可以使用通常的语义比较和复制它们，但当你将值赋给这种类型的对象时，而不是“重新绑定”引用元素（以便它们引用右侧的对象），赋值运算符实际上“通过”赋值，改变所引用对象的值。正如我们在“使用`std::tuple`”部分的代码示例中所看到的，这种故意的奇怪性允许我们使用`std::tie`作为一种“多重赋值”语句。

所以，我们可能期望或希望看到标准库中有一个`make_variant`函数的另一个原因可能是它的引用衰减能力。然而，这仅仅是因为一个简单的原因——标准禁止创建元素为引用类型的变体！我们将在本章后面看到，`std::optional`和`std::any`同样被禁止持有引用类型。（然而，`std::variant<std::reference_wrapper<T>, ...>`是完全合法的。）这种禁止的原因是库的设计者还没有就引用的变体应该意味着什么达成共识。或者，更确切地说，一个引用的*元组*应该意味着什么！我们今天在语言中拥有引用元组的原因仅仅是因为`std::tie`在 2011 年看起来是一个非常好的主意。到了 2017 年，没有人特别渴望通过引入变体、可选或引用的“任何”来增加混淆。

我们已经确定了`std::variant<A,B,C>`始终恰好持有类型`A`、`B`或`C`的一个值——不多也不少。嗯，这实际上并不完全正确。*在非常罕见的情况下，*有可能构造一个没有任何值的变体。唯一使这种情况发生的方法是使用类型`A`的值来构造变体，然后以这种方式给它分配一个类型`B`的值，即`A`被成功销毁，但构造函数`B`抛出异常，而`B`实际上从未被放置。当这种情况发生时，变体对象进入一个被称为“无值异常”的状态：

```cpp
    struct A {
      A() { throw "ha ha!"; }
    };
    struct B {
      operator int () { throw "ha ha!"; }
    };
    struct C {
      C() = default;
      C& operator=(C&&) = default;
      C(C&&) { throw "ha ha!"; }
    };

    void test()
    {
      std::variant<int, A, C> v1 = 42;

      try {
        v1.emplace<A>();
      } catch (const char *haha) {}
      assert(v1.valueless_by_exception());

      try {
        v1.emplace<int>(B());
      } catch (const char *haha) {}
      assert(v1.valueless_by_exception());
    }
```

这种情况永远不会发生在你身上，除非你正在编写构造函数或转换运算符会抛出异常的代码。此外，通过使用`operator=`而不是`emplace`，你可以避免在除了你有抛出异常的移动构造函数之外的所有情况下出现无值的变体：

```cpp
    v1 = 42;

    // Constructing the right-hand side of this assignment
    // will throw; yet the variant is unaffected.
    try { v1 = A(); } catch (...) {}
    assert(std::get<int>(v1) == 42);

    // In this case as well.
    try { v1 = B(); } catch (...) {}
    assert(std::get<int>(v1) == 42);

    // But a throwing move-constructor can still foul it up.
    try { v1 = C(); } catch (...) {}
    assert(v1.valueless_by_exception());
```

从第四章“容器动物园”中关于`std::vector`的讨论中回忆起来，你的类型的移动构造函数应该始终标记为`noexcept`；因此，如果你虔诚地遵循这条建议，你将能够完全避免处理`valueless_by_exception`。

无论如何，当一个变体处于这种状态时，它的`index()`方法返回`size_t(-1)`（一个也称为`std::variant_npos`的常量）并且任何尝试`std::visit`它的操作都将抛出一个类型为`std::bad_variant_access`的异常。

# 使用`std::optional`延迟初始化

你可能已经在想，`std::variant`的一个潜在用途可能是表示“也许我有一个对象，也许我没有。”例如，我们可以使用标准的标签类型`std::monostate`来表示“也许我没有”的状态：

```cpp
    std::map<std::string, int> g_limits = {
      { "memory", 655360 }
    };

    std::variant<std::monostate, int>
    get_resource_limit(const std::string& key)
    {
      if (auto it = g_limits.find(key); it != g_limits.end()) {
        return it->second;
      }
      return std::monostate{};
    }

    void test()
    {
      auto limit = get_resource_limit("memory");
      if (std::holds_alternative<int>(limit)) {
        use( std::get<int>(limit) );
      } else {
        use( some_default );
      }
    }
```

你会很高兴地知道，这*不是*实现该目标的最佳方式！标准库提供了专门用于处理“也许我有一个对象，也许我没有”这一概念的*vocabulary type* `std::optional<T>`。

```cpp
    std::optional<int>
    get_resource_limit(const std::string& key)
    {
      if (auto it = g_limits.find(key); it != g_limits.end()) {
        return it->second;
      }
      return std::nullopt;
    }

    void test()
    {
      auto limit = get_resource_limit("memory");
      if (limit.has_value()) {
        use( *limit );
      } else {
        use( some_default );
      }
    }
```

在代数数据类型的逻辑中，`std::optional<T>`是一个和类型：它具有与`T`一样多的可能值，再加上一个。这个额外值被称为“null”，“empty”或“disengaged”状态，并在源代码中由特殊常量`std::nullopt`表示。

不要将`std::nullopt`与同名的`std::nullptr`混淆！它们除了都是模糊的 null-like 之外，没有共同之处。

与具有混乱的免费（非成员）函数的`std::tuple`和`std::variant`不同，`std::optional<T>`类充满了方便的成员函数。`o.has_value()`为真，如果可选对象`o`当前持有类型为`T`的值。通常将“有值”状态称为“参与”状态；包含值的可选对象是“参与”的，而处于空状态的可选对象是“分离”的。

如果比较运算符`==`, `!=`, `<`, `<=`, `>`, 和 `>=`对于`T`是有效的，那么它们都会为`optional<T>`重载。要比较两个可选对象，或者将一个可选对象与类型为`T`的值进行比较，你需要记住的是，在分离状态下，可选对象与`T`的任何实际值比较时都“小于”。

`bool(o)`是`o.has_value()`的同义词，而`!o`是`!o.has_value()`的同义词。我个人建议你始终使用`has_value`，因为它们在运行时成本上没有区别；唯一的区别在于代码的可读性。如果你确实使用了简化的转换到`bool`的形式，请注意，对于`std::optional<bool>`，`o == false`和`!o`意味着非常不同的事情！

`o.value()`返回一个指向`o`包含的值的引用。如果`o`当前处于分离状态，则`o.value()`会抛出一个类型为`std::bad_optional_access`的异常。

使用重载的单目运算符 `operator*`，`*o` 返回 `o` 包含的值的引用，而不检查是否已连接。如果 `o` 当前未连接，并且你调用 `*o`，则这是未定义的行为，就像你调用 `*p` 在空指针上一样。你可以通过注意 C++ 标准库喜欢使用标点符号来表示其最有效、最少检查理智的操作来记住这种行为。例如，`std::vector::operator[]` 的边界检查比 `std::vector::at()` 少。因此，按照同样的逻辑，`std::optional::operator*` 的边界检查比 `std::optional::value()` 少。

`o.value_or(x)` 返回 `o` 包含的值的副本，或者如果 `o` 未连接，则返回将 `x` 转换为类型 `T` 的副本。我们可以使用 `value_or` 将前面的代码示例重写为一行简单且易于阅读的代码：

```cpp
    std::optional<int> get_resource_limit(const std::string&);

    void test() {
      auto limit = get_resource_limit("memory");
      use( limit.value_or(some_default) );
    }
```

前面的例子已经展示了如何使用 `std::optional<T>` 作为处理“可能是一个 `T`”在飞行中的方式（作为函数返回类型或参数类型）。另一种常见且有用的使用 `std::optional<T>` 的方式是作为处理“尚未是一个 `T`”在静止中的方式，作为类数据成员。例如，假设我们有一些类型 `L`，它不是默认可构造的，例如由 lambda 表达式产生的闭包类型：

```cpp
    auto make_lambda(int arg) {
      return arg { return x + arg; };
    }
    using L = decltype(make_lambda(0));

    static_assert(!std::is_default_constructible_v<L>);
    static_assert(!std::is_move_assignable_v<L>);
```

然后，具有该类型成员的类也将无法默认构造：

```cpp
    class ProblematicAdder {
      L fn_;
    };

    static_assert(!std::is_default_constructible_v<ProblematicAdder>);
```

但是，通过向我们的类提供一个类型为 `std::optional<L>` 的成员，我们允许它在需要默认构造性的上下文中使用：

```cpp
    class Adder {
      std::optional<L> fn_;
      public:
      void setup(int first_arg) {
        fn_.emplace(make_lambda(first_arg));
      }
      int call(int second_arg) {
        // this will throw unless setup() was called first
        return fn_.value()(second_arg);
      }
    };

    static_assert(std::is_default_constructible_v<Adder>);

    void test() {
      Adder adder;
      adder.setup(4);
      int result = adder.call(5);
      assert(result == 9); 
    }
```

没有使用 `std::optional`，要实现这种行为是非常困难的。你可以使用 placement-new 语法或使用 `union` 来做，但本质上你必须至少重新实现 `optional` 的一半。使用 `std::optional` 会更好！

注意，如果出于某种原因我们想要得到未定义的行为而不是从 `call()` 抛出异常的可能性，我们只需将 `fn_.value()` 替换为 `*fn_`。

`std::optional` 真的是 C++17 新特性中最大的胜利之一，通过熟悉它，你将受益匪浅。

从 `optional`，可以将其描述为一种有限的单类型 `variant`，我们现在转向另一个极端：无限代数数据类型的等价物。

# 重访变体

`variant` 数据类型擅长表示简单的选择，但截至 C++17，它并不特别适合表示如 JSON 列表之类的递归数据类型。也就是说，以下 C++17 代码将无法编译：

```cpp
    using JSONValue = std::variant<
      std::nullptr_t,
      bool,
      double,
      std::string,
      std::vector<JSONValue>,
      std::map<std::string, JSONValue>
    >;
```

有几种可能的解决方案。最稳健和正确的方法是继续使用 C++11 的 Boost 库 `boost::variant`，该库通过标记类型 `boost::recursive_variant_` 特定地支持递归变体类型：

```cpp
    using JSONValue = boost::variant<
      std::nullptr_t,
      bool,
      double,
      std::string,
      std::vector<boost::recursive_variant_>,
      std::map<std::string, boost::recursive_variant_>
    >;
```

你也可以通过引入一个新的类类型 `JSONValue` 来解决这个问题，该类型要么 **包含** 要素，要么 **是** 递归类型的 `std::variant`。

注意，在下面的例子中，我选择了 HAS-A 而不是 IS-A；从非多态的标准库类型继承几乎总是一个非常糟糕的想法。

由于 C++接受对类类型的转发引用，这将编译：

```cpp
    struct JSONValue {
      std::variant<
        std::nullptr_t,
        bool,
        double,
        std::string,
        std::vector<JSONValue>,
        std::map<std::string, JSONValue>
      > value_;
    };
```

最后的可能性是切换到标准库中的一个比`variant`更强大的代数类型。

# 使用 std::any 的无限备选方案

用亨利·福特的话来说，类型为`std::variant<A, B, C>`的对象可以存储一个值

任何类型--只要它是`A`、`B`或`C`。但是，假设我们想要存储一个*真正*任何类型的值？也许我们的程序将在运行时加载插件，这些插件可能包含无法预测的新类型。我们无法在`variant`中指定这些类型。或者，也许我们处于前一节详细描述的“递归数据类型”情况。

对于这些情况，C++17 标准库提供了一个代数数据类型的“无穷大”版本：类型`std::any`。这是一种容器（见第四章，*容器动物园*），用于存储任何类型的单个对象。容器可能是空的，也可能包含一个对象。您可以对`any`对象执行以下基本操作：

+   询问它当前是否持有对象

+   向其中放入一个新的对象（销毁旧对象，无论它是什么）

+   询问所持有对象的类型

+   通过正确命名其类型来检索所持有的对象

在代码中，这三个操作的前三个看起来像这样：

```cpp
    std::any a; // construct an empty container

    assert(!a.has_value());

    a = std::string("hello");
    assert(a.has_value());
    assert(a.type() == typeid(std::string));

    a = 42;
    assert(a.has_value());
    assert(a.type() == typeid(int));
```

第四种操作稍微有些复杂。它被称作`std::any_cast`，并且，就像`std::get`对变体一样，它有两种风味：一种类似于`std::get`的风味，在失败时抛出`std::bad_any_cast`异常，以及一种类似于`std::get_if`的风味，在失败时返回一个空指针：

```cpp
    if (std::string *p = std::any_cast<std::string>(&a)) {
      use(*p);
    } else {
      // go fish!
    }

    try {
      std::string& s = std::any_cast<std::string&>(a);
      use(s);
    } catch (const std::bad_any_cast&) {
      // go fish!
    }
```

注意，在两种情况下，您都必须命名您想要从`any`对象中检索的类型。如果您类型错误，那么您将得到一个异常或空指针。没有办法说“给我一个所持有的对象，无论它的类型是什么”，因为那样这个表达式的类型又是什么呢？

回想一下，当我们在前一节遇到与`std::variant`类似的问题时，我们通过使用`std::visit`将一些泛型代码访问到所持有的备选方案上解决了它。不幸的是，对于`any`没有等效的`std::visit`。原因是简单且无法克服的：分离编译。假设在一个源文件`a.cc`中，我有：

```cpp
    template<class T> struct Widget {};

    std::any get_widget() {
      return std::make_any<Widget<int>>();
    }
```

在另一个源文件`b.cc`中（可能编译成不同的插件，`.dll`或共享对象文件）我有：

```cpp
    template<class T> struct Widget {};

    template<class T> int size(Widget<T>& w) {
      return sizeof w;
    }

    void test()
    {
      std::any a = get_widget();
      int sz = hypothetical_any_visit([](auto&& w){
        return size(w);
      }, a);
      assert(sz == sizeof(Widget<int>));
    }
```

当编译 `b.cc` 时，编译器如何知道它需要输出 `size(Widget<int>&)` 的模板实例化，而不是，比如说，`size(Widget<double>&)`？当有人将 `a.cc` 改为返回 `make_any(Widget<char>&)` 时，编译器应该如何知道它需要使用新的 `size(Widget<char>&)` 实例重新编译 `b.cc`，而 `size(Widget<int>&)` 的实例不再需要——除非我们预计要链接到一个确实需要该实例化的 `c.cc`！基本上，编译器无法确定在可以定义为包含任何类型并触发任何代码生成的容器上，可能需要什么样的代码生成。

因此，为了提取 `any` 中包含值的任何函数，你必须事先知道该包含值的类型可能是什么。（如果你猜错了——去钓鱼吧！）

# `std::any` 与多态类类型

`std::any` 处于 `std::variant<A, B, C>` 的编译时多态和具有多态继承层次结构和 `dynamic_cast` 的运行时多态之间。你可能想知道 `std::any` 是否与 `dynamic_cast` 的机制有任何交互。答案是“没有，它没有”——也没有任何标准方法来获得这种行为。`std::any` 是百分之百的静态类型安全：没有方法可以突破它并获得“指向数据的指针”（例如，`void *`），除非你知道数据的确切静态类型：

```cpp
    struct Animal {
      virtual ~Animal() = default;
    };

    struct Cat : Animal {};

    void test()
    {
      std::any a = Cat{};

      // The held object is a "Cat"...
      assert(a.type() == typeid(Cat));
      assert(std::any_cast<Cat>(&a) != nullptr);

      // Asking for a base "Animal" will not work.
      assert(a.type() != typeid(Animal));
      assert(std::any_cast<Animal>(&a) == nullptr);

      // Asking for void* certainly will not work!
      assert(std::any_cast<void>(&a) == nullptr);
    }
```

# 简而言之，类型擦除

让我们简要地看看标准库如何实现 `std::any`。其核心思想被称为“类型擦除”，我们实现它的方法是通过识别我们想要支持的所有类型 `T` 的显著或相关操作，然后“擦除”任何特定类型 `T` 可能支持的任何其他独特操作。

对于 `std::any`，其显著的操作如下：

+   通过移动构造包含对象

+   通过移动构造包含对象

+   获取包含对象的 `typeid`

构造和销毁也是必需的，但这两个操作与包含对象本身的生存期管理有关，而不是“你可以用它做什么”，所以至少在这个情况下，我们不需要考虑它们。

因此，我们发明了一个支持仅这三种操作的多态类类型（称之为 `AnyBase`），这些操作作为可重写的 `virtual` 方法，然后每次程序员实际上将特定类型 `T` 的对象存储到 `any` 中时，我们创建一个新的派生类（称之为 `AnyImpl<T>`）：

```cpp
    class any;

    struct AnyBase {
      virtual const std::type_info& type() = 0;
      virtual void copy_to(any&) = 0;
      virtual void move_to(any&) = 0;
      virtual ~AnyBase() = default;
    };

    template<typename T>
    struct AnyImpl : AnyBase {
      T t_;
      const std::type_info& type() {
        return typeid(T);
      }
      void copy_to(any& rhs) override {
        rhs.emplace<T>(t_);
      }
      void move_to(any& rhs) override {
        rhs.emplace<T>(std::move(t_));
      }
      // the destructor doesn't need anything
      // special in this case
    };
```

使用这些辅助类，实现 `std::any` 的代码变得相当简单，尤其是当我们使用智能指针（见第六章，*智能指针*）来管理 `AnyImpl<T>` 对象的生存期时：

```cpp
    class any {
      std::unique_ptr<AnyBase> p_ = nullptr;
      public:
      template<typename T, typename... Args>
      std::decay_t<T>& emplace(Args&&... args) {
        p_ = std::make_unique<AnyImpl<T>>(std::forward<Args>(args)...);
      }

      bool has_value() const noexcept {
        return (p_ != nullptr);
      }

      void reset() noexcept {
        p_ = nullptr;
      }

      const std::type_info& type() const {
        return p_ ? p_->type() : typeid(void);
      }

      any(const any& rhs) {
        *this = rhs;
      }

      any& operator=(const any& rhs) {
        if (rhs.has_value()) {
          rhs.p_->copy_to(*this);
        }
        return *this;
      }
    };
```

前面的代码示例省略了移动赋值的实现。它可以像复制赋值一样完成，或者可以通过简单地交换指针来完成。标准库实际上在可能的情况下更喜欢交换指针，因为这保证是 `noexcept`；你可能会看到 `std::any` 不交换指针的唯一原因可能是它使用“小对象优化”来避免为非常小的、不可抛出移动构造的类型 `T` 进行堆分配。截至本文撰写时，libstdc++（GCC 使用的库）将使用小对象优化，并避免为大小最多为 8 字节类型的堆分配；libc++（Clang 使用的库）将使用小对象优化，适用于大小最多为 24 字节类型的类型。

与第四章中讨论的标准容器不同，《容器动物园》，`std::any` 不接受分配器参数，也不允许你自定义或配置其堆内存的来源。如果你在实时或内存受限的系统上使用 C++，其中不允许堆分配，那么你不应该使用 `std::any`。考虑一个替代方案，例如 Tiemo Jung 的 `tj::inplace_any<Size, Alignment>`。如果所有其他方法都失败了，你现在已经看到了如何自己实现它！

# std::any 和可复制性

注意到我们的 `AnyImpl<T>::copy_to` 定义要求 `T` 可复制构造。这对于标准的 `std::any` 也是正确的；没有方法可以将移动唯一类型存储到 `std::any` 对象中。绕过这个问题的方法是使用一种“适配器”包装器，其目的是使其移动唯一对象符合可复制构造的语法要求，同时避免任何实际的复制：

```cpp
    using Ptr = std::unique_ptr<int>;

    template<class T>
    struct Shim {
      T get() { return std::move(*t_); }

      template<class... Args>
      Shim(Args&&... args) : t_(std::in_place,
        std::forward<Args>(args)...) {}

      Shim(Shim&&) = default;
      Shim& operator=(Shim&&) = default;
      Shim(const Shim&) { throw "oops"; }
      Shim& operator=(const Shim&) { throw "oops"; }
      private:
      std::optional<T> t_;
    };

    void test()
    {
      Ptr p = std::make_unique<int>(42);

      // Ptr cannot be stored in std::any because it is move-only.
      // std::any a = std::move(p);

      // But Shim<Ptr> can be!
      std::any a = Shim<Ptr>(std::move(p));
      assert(a.type() == typeid(Shim<Ptr>));

      // Moving a Shim<Ptr> is okay...
      std::any b = std::move(a);

      try {
        // ...but copying a Shim<Ptr> will throw.
        std::any c = b;
      } catch (...) {}

      // Get the move-only Ptr back out of the Shim<Ptr>.
      Ptr r = std::any_cast<Shim<Ptr>&>(b).get();
      assert(*r == 42);
    }
```

注意前一个代码示例中 `std::optional<T>` 的使用；这保护了我们的假复制构造函数免受 `T` 可能不可默认构造的可能性。

# 再次提到类型擦除：std::function

我们观察到对于 `std::any`，显著的操作如下：

+   构建包含对象的副本

+   通过移动构造函数构建包含对象的副本

+   获取包含对象的 `typeid`

假设我们要添加一个到这组显著的操作中？让我们说我们的集合是：

+   构建包含对象的副本

+   通过移动构造函数构建包含对象的副本

+   获取包含对象的 `typeid`

+   使用特定的固定参数类型序列 `A...` 调用包含的对象，并将结果转换为某种特定的固定类型 `R`

这组操作的类型擦除对应于标准库类型 `std::function<R(A...)>`！

```cpp
    int my_abs(int x) { return x < 0 ? -x : x; }
    long unusual(long x, int y = 3) { return x + y; }

    void test()
    {
      std::function<int(int)> f; // construct an empty container
      assert(!f);

      f = my_abs; // store a function in the container
      assert(f(-42) == 42);

      f = [](long x) { return unusual(x); }; // or a lambda!
      assert(f(-42) == -39);
    }
```

如果包含的对象具有状态，则复制 `std::function` 总是会复制包含的对象。当然，如果包含的对象是函数指针，你不会观察到任何差异；但如果你尝试使用用户定义类类型的对象或具有状态的 lambda 表达式，你可以看到复制发生：

```cpp
    f = i=0 mutable { return ++i; };
    assert(f(-42) == 1); 
    assert(f(-42) == 2);

    auto g = f;
    assert(f(-42) == 3);
    assert(f(-42) == 4);
    assert(g(-42) == 3);
    assert(g(-42) == 4);
```

就像 `std::any` 一样，`std::function<R(A...)` 允许你检索包含对象的 `typeid`，或者如果你静态地知道（或可以猜测）其类型，可以检索指向该对象的指针：

+   `f.target_type()` 等同于 `a.type()`

+   `f.target<T>()` 等同于 `std::any_cast<T*>(&a)`

```cpp
    if (f.target_type() == typeid(int(*)(int))) {
      int (*p)(int) = *f.target<int (*)(int)>();
      use(p);
    } else {
      // go fish!
    }
```

话虽如此，我在现实生活中从未见过这些方法的实际用例。通常，如果你必须询问 `std::function` 的包含类型，那么你已经做错了什么。

`std::function` 最重要的用例是作为跨越模块边界的“行为”传递的词汇类型，在这种情况下使用模板是不可能的——例如，当你需要将回调传递给外部库中的函数时，或者当你编写需要从其调用者接收回调的库时：

```cpp
    // templated_for_each is a template and must be visible at the
    // point where it is called.
    template<class F>
    void templated_for_each(std::vector<int>& v, F f) {
      for (int& i : v) {
        f(i);
      }
    }

    // type_erased_for_each has a stable ABI and a fixed address.
    // It can be called with only its declaration in scope.
    extern void type_erased_for_each(std::vector<int>&,
      std::function<void(int)>);
```

我们在本章开始时讨论了 `std::string`，这是在函数之间传递字符串的标准词汇类型；现在，随着本章的结束，我们正在讨论 `std::function`，这是在函数之间传递 *函数* 的标准词汇类型！

# std::function, 可复制性和分配

就像 `std::any` 一样，`std::function` 要求存储在其内的任何对象都必须是可复制的。如果你使用了很多捕获 `std::future<T>`、`std::unique_ptr<T>` 或其他只能移动的类型（move-only types）的 lambda 表达式，这可能会带来问题：这样的 lambda 类型本身也将是只能移动的。解决这个问题的一种方法在本章的 *std::any and copyability* 部分已经演示过：我们可以引入一个在语法上可复制的适配器（shim），但如果你尝试复制它，它会抛出一个异常。

当与 `std::function` 和 lambda 捕获一起工作时，可能更倾向于通过 `shared_ptr` 捕获只能移动的 lambda 捕获。我们将在下一章介绍 `shared_ptr`：

```cpp
    auto capture = [](auto& p) {
      using T = std::decay_t<decltype(p)>;
      return std::make_shared<T>(std::move(p));
    };

   std::promise<int> p;

   std::function<void()> f = [sp = capture(p)]() {
     sp->set_value(42);
   };
```

就像 `std::any` 一样，`std::function` 不接受分配器参数，也不允许你自定义或配置其堆内存的来源。如果你在实时或内存受限的系统上使用 C++，其中不允许堆分配，那么你不应该使用 `std::function`。考虑使用如 Carl Cook 的 `sg14::inplace_function<R(A...), Size, Alignment>` 这样的替代方案。

# 概述

类似于 `std::string` 和 `std::function` 这样的词汇类型（vocabulary types）允许我们共享一个 *通用语言* 来处理常见的编程概念。在 C++17 中，我们有一套丰富的词汇类型来处理 *代数数据类型*：`std::pair` 和 `std::tuple`（积类型），`std::optional` 和 `std::variant`（和类型），以及 `std::any`（和类型的终极形式——它可以存储几乎任何东西）。然而，不要沉迷于使用 `std::tuple` 和 `std::variant` 作为每个函数的返回类型！命名类类型仍然是保持代码可读性最有效的方法。

使用 `std::optional` 来表示可能缺少的值，或者表示数据成员的“尚未存在”状态。

使用 `std::get_if<T>(&v)` 来查询 `variant` 的类型；使用 `std::any_cast<T>(&a)` 来查询 `any` 的类型。请记住，您提供的类型必须与目标类型完全匹配；如果不匹配，您将得到 `nullptr`。

请注意，`make_tuple` 和 `make_pair` 不仅构造 `tuple` 和 `pair` 对象；它们还将 `reference_wrapper` 对象解引用为原生引用。使用 `std::tie` 和 `std::forward_as_tuple` 来创建引用的元组。`std::tie` 特别适用于多重赋值和编写比较运算符。`std::forward_as_tuple` 对于元编程很有用。

请注意，`std::variant` 总是有可能处于“异常无值”状态；但要知道，除非您编写具有抛出移动构造函数的类，否则您不必担心这种情况。另外：不要编写具有抛出移动构造函数的类！

请注意，类型擦除的类型 `std::any` 和 `std::function` 隐式地使用了堆。第三方库提供了这些类型的非标准 `inplace_` 版本。请注意，`std::any` 和 `std::function` 要求其包含的类型必须是可复制的。如果出现这种情况，请使用 "通过 `shared_ptr` 捕获" 来处理。
