# 第五章：利用 C++语言特性

C++语言是一种独特的语言。它被用于各种情况，从创建固件和操作系统，桌面和移动应用程序，到服务器软件，框架和服务。C++代码在各种硬件上运行，在计算云上大规模部署，并且甚至可以在外太空中找到。如果没有这种多范式语言具有的广泛功能集，这样的成功是不可能的。

本章描述了如何利用 C++语言提供的内容，以便我们可以实现安全和高性能的解决方案。我们将展示类型安全的最佳行业实践，避免内存问题，并以同样高效的方式创建高效的代码。我们还将教您在设计 API 时如何使用某些语言特性。

在本章中，我们将涵盖以下主题：

+   管理资源和避免泄漏

+   将计算从运行时移动到编译时

+   利用安全类型的能力

+   创建易于阅读和高性能的代码

+   将代码分成模块

在这段旅程中，您将了解各种 C++标准中可用的功能和技术，从 C++98 到 C++20。这将包括声明式编程，RAII，`constexpr`，模板，概念和模块。话不多说，让我们开始这段旅程吧。

# 技术要求

您需要以下工具来构建本章中的代码：

+   支持 C++20 的编译器（建议使用 GCC 11+）

+   CMake 3.15+

本章的源代码可以在[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter05`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter05)找到。

# 设计出色的 APIs

尽管 C++允许您使用您可能熟悉的基于对象的 API，但它还有一些其他技巧。我们将在本节中提到其中一些。

## 利用 RAII

C API 和 C++ API 之间的主要区别是什么？通常，这与多态性或具有类本身无关，而是与一种称为 RAII 的习惯有关。

**RAII**代表**资源获取即初始化**，但实际上更多的是关于释放资源而不是获取资源。让我们看一下在 C 和 C++中编写的类似 API，以展示这个特性的作用：

```cpp
struct Resource;

// C API
Resource* acquireResource();
void releaseResource(Resource *resource);

// C++ API
using ResourceRaii = std::unique_ptr<Resource, decltype(&releaseResource)>;
ResourceRaii acquireResourceRaii();
```

C++ API 基于 C API，但这并不总是必须的。重要的是，在 C++ API 中，不需要单独的函数来释放我们宝贵的资源。由于 RAII 习惯，一旦`ResourceRaii`对象超出范围，它就会自动完成。这减轻了用户手动管理资源的负担，最重要的是，它不需要额外的成本。

而且，我们不需要编写任何我们自己的类 - 我们只是重用了标准库的`unique_ptr`，它是一个轻量级指针。它确保它管理的对象将始终被释放，并且将始终被精确释放一次。

由于我们管理一些特殊类型的资源而不是内存，我们必须使用自定义的删除器类型。我们的`acquireResourceRaii`函数需要将实际指针传递给`releaseResource`函数。如果您只想从 C++中使用它，C API 本身不需要暴露给用户。

这里需要注意的一点是，RAII 不仅用于管理内存：您可以使用它轻松处理任何资源的所有权，例如锁，文件句柄，数据库连接，以及任何应该在其 RAII 包装器超出范围时释放的其他资源。

## 指定 C++中容器的接口

标准库的实现是搜索惯用和高性能 C++代码的好地方。例如，如果你想阅读一些非常有趣的模板代码，你应该尝试一下`std::chrono`，因为它演示了一些有用的技术，并对此有了新的方法。在*进一步阅读*部分可以找到 libstdc++的实现链接。

当涉及到库的其他地方时，即使快速查看其容器也会发现它们的接口往往与其他编程语言中的对应物不同。为了展示这一点，让我们来看一下标准库中一个非常直接的类`std::array`，并逐个分析它：

```cpp
template <class T, size_t N>
struct array {
 // types:
 typedef T& reference;
 typedef const T& const_reference;
 typedef /*implementation-defined*/ iterator;
 typedef /*implementation-defined*/ const_iterator;
 typedef size_t size_type;
 typedef ptrdiff_t difference_type;
 typedef T value_type;
 typedef T* pointer;
 typedef const T* const_pointer;
 typedef reverse_iterator<iterator> reverse_iterator;
 typedef reverse_iterator<const_iterator> const_reverse_iterator;
```

当你开始阅读类定义时，你会看到的第一件事是它为一些类型创建了别名。这在标准容器中很常见，这些别名的名称在许多容器中都是相同的。这是由于几个原因。其中之一是最少惊讶原则 - 以这种方式减少开发人员花在思考你的意思以及特定别名的命名方式上的时间。另一个原因是你的类的用户和库编写者在编写他们自己的代码时经常依赖这样的类型特征。如果你的容器不提供这样的别名，它将使得使用它与一些标准工具或类型特征更加困难，因此你的 API 的用户将不得不解决这个问题，甚至使用完全不同的类。

即使在模板中没有使用这些类型别名，拥有这样的类型别名也是有用的。在函数参数和类成员字段中依赖于这些类型是很常见的，所以如果你正在编写一个其他人可能会使用的类，一定要记得提供它们。例如，如果你正在编写一个分配器，它的许多使用者将依赖于特定的类型别名存在。

让我们看看数组类将给我们带来什么：

```cpp
 // no explicit construct/copy/destroy for aggregate type
```

因此，关于`std::array`的另一个有趣之处是它没有定义构造函数，包括复制/移动构造函数；赋值运算符；或析构函数。这仅仅是因为拥有这些成员不会增加任何价值。通常，在不必要的情况下添加这些成员实际上对性能有害。有了非默认构造函数（`T() {}`已经是非默认的，与`T() = default;`相反），你的类不再是平凡的，也不再是平凡可构造的，这将阻止编译器对其进行优化。

让我们看看我们的类还有哪些其他声明：

```cpp
 constexpr void fill(const T& u);
 constexpr void swap(array<T, N>&) noexcept(is_nothrow_swappable_v<T&>);
```

现在，我们可以看到两个成员函数，包括一个成员交换。通常，不依赖于`std::swap`的默认行为并提供我们自己的交换函数是有利的。例如，在`std::vector`的情况下，底层存储被整体交换，而不是每个元素被交换。当你编写一个成员交换函数时，一定要引入一个名为`swap`的自由函数，以便通过**参数相关查找**（**ADL**）来检测它。它可以调用你的成员`swap`函数。

关于值得一提的交换函数的另一件事是它是有条件的`noexcept`。如果存储的类型可以在不抛出异常的情况下交换，那么数组的交换也将是`noexcept`的。具有不抛出异常的交换可以帮助你在存储我们类型的成员的类的复制操作中实现强异常安全性保证。

如下面的代码块所示，现在出现了一大堆函数，它们向我们展示了许多类的另一个重要方面 - 它们的迭代器：

```cpp
 // iterators:
 constexpr iterator begin() noexcept;
 constexpr const_iterator begin() const noexcept;
 constexpr iterator end() noexcept;
 constexpr const_iterator end() const noexcept;

 constexpr reverse_iterator rbegin() noexcept;
 constexpr const_reverse_iterator rbegin() const noexcept;
 constexpr reverse_iterator rend() noexcept;
 constexpr const_reverse_iterator rend() const noexcept;

 constexpr const_iterator cbegin() const noexcept;
 constexpr const_iterator cend() const noexcept;
 constexpr const_reverse_iterator crbegin() const noexcept;
 constexpr const_reverse_iterator crend() const noexcept;
```

迭代器对于每个容器都是至关重要的。如果您的类没有提供迭代器访问权限，您将无法在基于范围的循环中使用它，并且它将与标准库中的所有有用算法不兼容。这并不意味着您需要编写自己的迭代器类型 - 如果您的存储是连续的，您可以只使用简单的指针。提供`const`迭代器可以帮助您以不可变的方式使用类，并且提供反向迭代器可以帮助扩展容器的更多用例。

让我们看看接下来会发生什么：

```cpp
 // capacity:
 constexpr size_type size() const noexcept;
 constexpr size_type max_size() const noexcept;
 constexpr bool empty() const noexcept;

 // element access:
 constexpr reference operator[](size_type n);
 constexpr const_reference operator[](size_type n) const;
 constexpr const_reference at(size_type n) const;
 constexpr reference at(size_type n);
 constexpr reference front();
 constexpr const_reference front() const;
 constexpr reference back();
 constexpr const_reference back() const;

 constexpr T * data() noexcept;
 constexpr const T * data() const noexcept;
private:
 // the actual storage, like T elements[N];
};
```

在迭代器之后，我们有一些检查和修改容器数据的方法。在`array`的情况下，所有这些方法都是`constexpr`的。这意味着如果我们要编写一些编译时代码，我们可以使用我们的数组类。我们将在本章的*在编译时移动计算*部分中更详细地讨论这一点。

最后，我们完成了对`array`的整个定义。然而，它的接口并不仅限于此。从 C++17 开始，在类型定义之后，您可以看到类似以下的行：

```cpp
template<class T, class... U>
  array(T, U...) -> array<T, 1 + sizeof...(U)>;
```

这些语句被称为**推导指南**。它们是**类模板参数推导**（**CTAD**）功能的一部分，该功能在 C++17 中引入。它允许您在声明变量时省略模板参数。对于`array`来说，这很方便，因为现在您可以只写以下内容：

```cpp
auto ints = std::array{1, 2, 3};
```

但是，对于更复杂的类型，例如映射，它可能更方便，如下所示：

```cpp
auto legCount = std::unordered_map{ std::pair{"cat", 4}, {"human", 2}, {"mushroom", 1} };
```

然而，这里有一个问题：当我们传递第一个参数时，我们需要指定我们正在传递键值对（请注意，我们还为其使用了推导指南）。

既然我们谈到了接口，让我们指出其中的一些其他方面。

## 在接口中使用指针

您在接口中使用的类型非常重要。即使有文档，一个良好的 API 在一瞥之间仍应该是直观的。让我们看看不同的传递资源参数给函数的方法如何向 API 使用者暗示不同的事情。

考虑以下函数声明：

```cpp
void A(Resource*); 
void B(Resource&); 
void C(std::unique_ptr<Resource>); 
void D(std::unique_ptr<Resource>&);
void E(std::shared_ptr<Resource>); 
void F(std::shared_ptr<Resource>&);
```

您应该在何时使用这些函数？

由于智能指针现在是处理资源的标准方式，`A`和`B`应该留给简单的参数传递，并且如果您不对传递的对象的所有权做任何操作，则不应使用它们。`A`应该仅用于单个资源。例如，如果您想要传递多个实例，可以使用容器，例如`std::span`。如果您知道要传递的对象不为空，最好通过引用传递，例如 const 引用。如果对象不太大，也可以考虑通过值传递。

关于函数`C`到`F`的一个很好的经验法则是，如果您想要操纵指针本身，那么只应传递智能指针作为参数；例如，用于所有权转移。

函数`C`通过值接受`unique_ptr`。这意味着它是一个资源接收器。换句话说，它会消耗然后释放资源。请注意，通过选择特定类型，接口清晰地表达了其意图。

函数`D`应该仅在您想要传递包含一个资源的`unique_ptr`并在同一个`unique_ptr`中作为输出参数接收另一个资源时使用。对于简单地传递资源来说，拥有这样的函数并不是一个好主意，因为调用者需要将其专门存储在`unique_ptr`中。换句话说，如果您考虑传递`const unique_ptr<Resource>&`，只需传递`Resource*`（或`Resource&`）即可。

函数`E`用于与调用方共享资源所有权。通过值传递`shared_ptr`可能相对昂贵，因为需要增加其引用计数。然而，在这种情况下，通过值传递`shared_ptr`是可以的，因为如果调用方真的想成为共享所有者，那么必须在某个地方进行复制。

`F`函数类似于`D`，只有在你想要操作`shared_ptr`实例并通过这个输入/输出参数传播更改时才应该使用。如果你不确定函数是否应该拥有所有权，考虑传递一个`const shared_ptr&`。

## 指定前置条件和后置条件

一个函数通常会对其参数有一些要求是很常见的。每个要求都应该被陈述为一个前置条件。如果一个函数保证其结果具有某些属性——例如，它是非负的——那么函数也应该清楚地表明这一点。一些开发人员会使用注释来通知其他人，但这并不真正以任何方式强制要求。放置`if`语句会更好一些，但会隐藏检查的原因。目前，C++标准仍然没有提供处理这个问题的方法（合同最初被投票纳入 C++20 标准，后来被移除）。幸运的是，像微软的**指南支持库**（**GSL**）这样的库提供了它们自己的检查。

假设出于某种原因，我们正在编写自己的队列实现。push 成员函数可能如下所示：

```cpp
template<typename T>
T& Queue::push(T&& val) {
 gsl::Expects(!this->full());
 // push the element
 gsl::Ensures(!this->empty());
}
```

请注意，用户甚至不需要访问实现就可以确保某些检查已经就位。代码也是自我描述的，因为清楚地表明了函数需要什么以及结果将是什么。

## 利用内联命名空间

在系统编程中，通常情况下，你并不总是只是针对 API 编写代码；通常情况下，你还需要关心 ABI 兼容性。当 GCC 发布其第五个版本时，发生了一个著名的 ABI 破坏，其中一个主要变化是改变了`std::string`的类布局。这意味着使用旧版 GCC 版本的库（或者在较新的 GCC 版本中仍然使用新的 ABI，这在最近的 GCC 版本中仍然存在）将无法与使用较新 ABI 编写的代码一起工作。在发生 ABI 破坏的情况下，如果收到链接器错误，你可以算自己幸运。在某些情况下，例如将`NDEBUG`代码与调试代码混合使用，如果一个类只有在一种配置中可用的成员，你可能会遇到内存损坏；例如，为了更好地进行调试而添加特殊成员。

一些内存损坏，通常很难调试，可以很容易地通过使用 C++11 的内联命名空间转换为链接器错误。考虑以下代码：

```cpp
#ifdef NDEBUG

inline namespace release {

#else 

inline namespace debug {

#endif


struct EasilyDebuggable {

// ...

#ifndef NDEBUG

// fields helping with debugging

#endif

};


} // end namespace
```

由于前面的代码使用了内联命名空间，当你声明这个类的对象时，用户在两种构建类型之间看不到任何区别：内联命名空间中的所有声明在周围范围内都是可见的。然而，链接器最终会得到不同的符号名称，这将导致链接器在尝试链接不兼容的库时失败，给我们提供了我们正在寻找的 ABI 安全性和一个提到内联命名空间的良好错误消息。

有关提供安全和优雅的 ABI 的更多提示，请参阅*Arvid Norberg*在*C++Now* 2019 年的*The ABI Challenge*演讲，链接在*进一步阅读*部分中。

## 利用 std::optional

从 ABI 转回 API，让我们提到在本书早期设计伟大的 API 时遗漏的另一种类型。本节的英雄可以在函数的可选参数方面挽救局面，因为它可以帮助你的类型具有可能包含值的组件，也可以用于设计清晰的接口或作为指针的替代。这个英雄被称为`std::optional`，并在 C++17 中标准化。如果你不能使用 C++17，你仍然可以在 Abseil（`absl::optional`）中找到它，或者在 Boost（`boost::optional`）中找到一个非常相似的版本。使用这些类的一个重要优点是它们非常清晰地表达了意图，有助于编写清晰和自我描述的接口。让我们看看它的作用。

### 可选函数参数

我们将从向可能但不一定持有值的函数传递参数开始。你是否曾经遇到过类似以下的函数签名？

```cpp
void calculate(int param); // If param equals -1 it means "no value"


void calculate(int param = -1);
```

有时，当你不想在`param`在代码的其他地方计算时，却很容易错误地传递一个`-1`——也许在那里它甚至是一个有效的值。以下签名怎么样？

```cpp
void calculate(std::optional<int> param);
```

这一次，如果你不想传递一个`value`，该怎么做就清楚多了：只需传递一个空的 optional。意图明确，而且`-1`仍然可以作为一个有效的值，而不需要以一种类型不安全的方式赋予它任何特殊含义。

这只是我们 optional 模板的一个用法。让我们看看其他一些用法。

### 可选的函数返回值

就像接受特殊值来表示参数的*无值*一样，有时函数可能返回*无值*。你更喜欢以下哪种？

```cpp
int try_parse(std::string_view maybe_number);
bool try_parse(std::string_view maybe_number, int &parsed_number);
int *try_parse(std::string_view maybe_number);
std::optional<int> try_parse(std::string_view maybe_number);
```

你怎么知道第一个函数在出现错误时会返回什么值？或者它会抛出异常而不是返回一个魔术值？接下来看第二个签名，看起来如果出现错误会返回`false`，但是很容易忘记检查它，直接读取`parsed_number`，可能会引起麻烦。在第三种情况下，虽然可以相对安全地假设在出现错误时会返回一个`nullptr`，在成功的情况下会返回一个整数，但现在不清楚返回的`int`是否应该被释放。

通过最后一个签名，仅仅通过看它就清楚了，在出现错误的情况下将返回一个空值，而且没有其他需要做的事情。这简单、易懂、优雅。

可选的返回值也可以用来标记*无值*的返回，而不一定是发生了错误。说到这里，让我们继续讨论 optionals 的最后一个用例。

### Optional 类成员

在一个类状态中实现一致性并不总是一件容易的事情。例如，有时你希望有一个或两个成员可以简单地不设置。而不是为这种情况创建另一个类（增加代码复杂性）或保留一个特殊值（很容易被忽视），你可以使用一个 optional 类成员。考虑以下类型：

```cpp
struct UserProfile {
  std::string nickname;
  std::optional <std::string> full_name;
  std::optional <std::string> address;
  std::optional <PhoneNumber> phone;
};
```

在这里，我们可以看到哪些字段是必要的，哪些不需要填充。相同的数据可以使用空字符串存储，但这并不会清晰地从结构的定义中看出。另一种选择是使用`std::unique_ptr`，但这样我们会失去数据的局部性，这对性能通常是至关重要的。对于这种情况，`std::optional`可以有很大的价值。当你想要设计干净和直观的 API 时，它绝对应该是你的工具箱的一部分。

这些知识可以帮助你提供高质量和直观的 API。还有一件事可以进一步改进它们，这也将帮助你默认情况下编写更少的错误代码。我们将在下一节讨论这个问题。

# 编写声明式代码

你熟悉命令式与声明式编码风格吗？前者是当你的代码一步一步地告诉机器*如何*实现你想要的。后者是当你只告诉机器*你*想要实现什么。某些编程语言更偏向于其中一种。例如，C 是命令式的，而 SQL 是声明式的，就像许多函数式语言一样。有些语言允许你混合这些风格——想想 C#中的 LINQ。

C++是一个灵活的语言，允许你以两种方式编写代码。你应该更倾向于哪一种呢？事实证明，当你编写声明式代码时，通常会保持更高的抽象级别，这会导致更少的错误和更容易发现的错误。那么，我们如何声明式地编写 C++呢？有两种主要的策略可以应用。

第一种是编写函数式风格的 C++，这是你在可能的情况下更倾向于纯函数式风格（函数没有副作用）。你应该尝试使用标准库算法，而不是手动编写循环。考虑以下代码：

```cpp
auto temperatures = std::vector<double>{ -3., 2., 0., 8., -10., -7\. };

// ...

for (std::size_t i = 0; i < temperatures.size() - 1; ++i) {

    for (std::size_t j = i + 1; j < temperatures.size(); ++j) {

        if (std::abs(temperatures[i] - temperatures[j]) > 5) 
            return std::optional{i};

    }

}

return std::nullopt;
```

现在，将前面的代码与执行相同操作的以下片段进行比较：

```cpp
auto it = std::ranges::adjacent_find(temperatures, 
                                     [](double first, double second) {
    return std::abs(first - second) > 5);
});
if (it != std::end(temperatures)) 
    return std::optional{std::distance(std::begin(temperatures), it)};

return std::nullopt);
```

这两个片段都返回了最后一个具有相对稳定温度的日子。你更愿意阅读哪一个？哪一个更容易理解？即使你现在对 C++算法不太熟悉，但在代码中遇到几次后，它们就会感觉比手工编写的循环更简单、更安全、更清晰。因为它们通常就是这样。

在 C++中编写声明性代码的第二种策略在前面的片段中已经有所体现。您应该优先使用声明性 API，比如来自 ranges 库的 API。虽然我们的片段中没有使用范围视图，但它们可以产生很大的不同。考虑以下片段：

```cpp
using namespace std::ranges;

auto is_even = [](auto x) { return x % 2 == 0; };

auto to_string = [](auto x) { return std::to_string(x); };

auto my_range = views::iota(1)

    | views::filter(is_even)

    | views::take(2)
```

```cpp
    | views::reverse

    | views::transform(to_string);

std::cout << std::accumulate(begin(my_range), end(my_range), ""s) << '\n';
```

这是声明性编码的一个很好的例子：你只需指定应该发生什么，而不是如何。前面的代码获取了前两个偶数，颠倒它们的顺序，并将它们打印为一个字符串，从而打印出了生活、宇宙和一切的著名答案：42。所有这些都是以一种直观和易于修改的方式完成的。

## 展示特色项目画廊

不过，玩具示例就到此为止。还记得我们在第三章中的多米尼加展会应用程序吗，*功能和非功能需求*？让我们编写一个组件，它将从客户保存为收藏夹的商店中选择并显示一些特色项目。例如，当我们编写移动应用程序时，这可能非常方便。

让我们从一个主要是 C++17 实现开始，然后在本章中将其更新为 C++20。这将包括添加对范围的支持。

首先，让我们从获取有关当前用户的信息的一些代码开始：

```cpp
using CustomerId = int;


CustomerId get_current_customer_id() { return 42; }
```

现在，让我们添加商店所有者：

```cpp
struct Merchant {

  int id;

};
```

商店也需要有商品：

```cpp
struct Item {

  std::string name;

  std::optional<std::string> photo_url;

  std::string description;

  std::optional<float> price;

  time_point<system_clock> date_added{};

  bool featured{};

};
```

有些项目可能没有照片或价格，这就是为什么我们为这些字段使用了`std::optional`。

接下来，让我们添加一些描述我们商品的代码：

```cpp
std::ostream &operator<<(std::ostream &os, const Item &item) {

  auto stringify_optional = [](const auto &optional) {

    using optional_value_type =

        typename std::remove_cvref_t<decltype(optional)>::value_type;

    if constexpr (std::is_same_v<optional_value_type, std::string>) {

      return optional ? *optional : "missing";

    } else {

      return optional ? std::to_string(*optional) : "missing";

    }

  };


  auto time_added = system_clock::to_time_t(item.date_added);


  os << "name: " << item.name

     << ", photo_url: " << stringify_optional(item.photo_url)

     << ", description: " << item.description

     << ", price: " << std::setprecision(2) 
     << stringify_optional(item.price)

     << ", date_added: " 
     << std::put_time(std::localtime(&time_added), "%c %Z")

     << ", featured: " << item.featured;

  return os;

}
```

首先，我们创建了一个帮助 lambda，用于将我们的`optionals`转换为字符串。因为我们只想在我们的`<<`运算符中使用它，所以我们在其中定义了它。

请注意，我们使用了 C++14 的通用 lambda（auto 参数），以及 C++17 的`constexpr`和`is_same_v`类型特征，这样当我们处理可选的`<string>`时，我们就有了不同的实现。在 C++17 之前实现相同的功能需要编写带有重载的模板，导致代码更加复杂：

```cpp
enum class Category {

  Food,

  Antiques,

  Books,

  Music,

  Photography,

  Handicraft,

  Artist,

};
```

最后，我们可以定义存储本身：

```cpp
struct Store {

  gsl::not_null<const Merchant *> owner;

  std::vector<Item> items;

  std::vector<Category> categories;

};
```

这里值得注意的是使用了指南支持库中的`gsl::not_null`模板，这表明所有者将始终被设置。为什么不只使用一个普通的引用？因为我们可能希望我们的存储是可移动和可复制的。使用引用会妨碍这一点。

现在我们有了这些构建模块，让我们定义如何获取客户的收藏夹商店。为简单起见，让我们假设我们正在处理硬编码的商店和商家，而不是创建用于处理外部数据存储的代码。

首先，让我们为商店定义一个类型别名，并开始我们的函数定义：

```cpp
using Stores = std::vector<gsl::not_null<const Store *>>;


Stores get_favorite_stores_for(const CustomerId &customer_id) {
```

接下来，让我们硬编码一些商家，如下所示：

```cpp
  static const auto merchants = std::vector<Merchant>{{17}, {29}};
```

现在，让我们添加一个带有一些项目的商店，如下所示：

```cpp
  static const auto stores = std::vector<Store>{

      {.owner = &merchants[0],

       .items =

           {

               {.name = "Honey",

                .photo_url = {},

                .description = "Straight outta Compton's apiary",

                .price = 9.99f,

                .date_added = system_clock::now(),

                .featured = false},

               {.name = "Oscypek",

                .photo_url = {},

                .description = "Tasty smoked cheese from the Tatra 
                                mountains",

                .price = 1.23f,

                .date_added = system_clock::now() - 1h,

                .featured = true},

           },

       .categories = {Category::Food}},

      // more stores can be found in the complete code on GitHub

  };
```

在这里，我们介绍了我们的第一个 C++20 特性。你可能不熟悉`.field = value;`的语法，除非你在 C99 或更新的版本中编码过。从 C++20 开始，您可以使用这种表示法（官方称为指定初始化器）来初始化聚合类型。它比 C99 更受限制，因为顺序很重要，尽管它还有一些其他较小的差异。没有这些初始化器，很难理解哪个值初始化了哪个字段。有了它们，代码更冗长，但即使对于不熟悉编程的人来说，更容易理解。

一旦我们定义了我们的商店，我们就可以编写函数的最后部分，这部分将执行实际的查找：

```cpp
  static auto favorite_stores_by_customer =

      std::unordered_map<CustomerId, Stores>{{42, {&stores[0], &stores[1]}}};

  return favorite_stores_by_customer[customer_id];

}
```

现在我们有了我们的商店，让我们编写一些代码来获取这些商店的特色物品：

```cpp
using Items = std::vector<gsl::not_null<const Item *>>;


Items get_featured_items_for_store(const Store &store) {

  auto featured = Items{};

  const auto &items = store.items;

  for (const auto &item : items) {

    if (item.featured) {

      featured.emplace_back(&item);

    }

  }

  return featured;

}
```

前面的代码是为了从一个商店获取物品。让我们还编写一个函数，从所有给定的商店获取物品：

```cpp
Items get_all_featured_items(const Stores &stores) {

  auto all_featured = Items{};

  for (const auto &store : stores) {

    const auto featured_in_store = get_featured_items_for_store(*store);

    all_featured.reserve(all_featured.size() + featured_in_store.size());

    std::copy(std::begin(featured_in_store), std::end(featured_in_store),

              std::back_inserter(all_featured));

  }

  return all_featured;

}
```

前面的代码使用`std::copy`将元素插入向量，预先分配内存由保留调用。

现在我们有了一种获取有趣物品的方法，让我们按“新鲜度”对它们进行排序，以便最近添加的物品将首先显示：

```cpp
void order_items_by_date_added(Items &items) {

  auto date_comparator = [](const auto &left, const auto &right) {

    return left->date_added > right->date_added;

  };

  std::sort(std::begin(items), std::end(items), date_comparator);

}
```

如您所见，我们利用了带有自定义比较器的`std::sort`。如果愿意，您也可以强制`left`和`right`的类型相同。为了以通用方式执行此操作，让我们使用另一个 C++20 特性：模板 lambda。让我们将它们应用于前面的代码：

```cpp
void order_items_by_date_added(Items &items) {

  auto date_comparator = []<typename T>(const T &left, const T &right) {

    return left->date_added > right->date_added;

  };

  std::sort(std::begin(items), std::end(items), date_comparator);

}
```

lambda 的`T`类型将被推断，就像对于任何其他模板一样。

还缺少的最后两部分是实际的渲染代码和将所有内容粘合在一起的主函数。在我们的示例中，渲染将简单地打印到一个`ostream`：

```cpp
void render_item_gallery(const Items &items) {

  std::copy(

      std::begin(items), std::end(items),

      std::ostream_iterator<gsl::not_null<const Item *>>(std::cout, "\n"));

}
```

在我们的情况下，我们只是将每个元素复制到标准输出，并在元素之间插入一个换行符。使用`copy`和`ostream_iterator`允许您自己处理元素的分隔符。在某些情况下，这可能很方便；例如，如果您不希望在最后一个元素之后有逗号（或者在我们的情况下是换行符）。

最后，我们的主要函数将如下所示：

```cpp
int main() {

  auto fav_stores = get_favorite_stores_for(get_current_customer_id());


  auto selected_items = get_all_featured_items(fav_stores);


  order_items_by_date_added(selected_items);


  render_item_gallery(selected_items);

}
```

看吧！随时运行代码，看看它如何打印我们的特色物品：

```cpp
name: Handmade painted ceramic bowls, photo_url: http://example.com/beautiful_bowl.png, description: Hand-crafted and hand-decorated bowls made of fired clay, price: missing, date_added: Sun Jan  3 12:54:38 2021 CET, featured: 1

name: Oscypek, photo_url: missing, description: Tasty smoked cheese from the Tatra mountains, price: 1.230000, date_added: Sun Jan  3 12:06:38 2021 CET, featured: 1
```

现在我们已经完成了我们的基本实现，让我们看看如何通过使用 C++20 的一些新语言特性来改进它。

## 介绍标准范围

我们的第一个添加将是范围库。您可能还记得，它可以帮助我们实现优雅、简单和声明性的代码。为了简洁起见，首先，我们将引入`ranges`命名空间：

```cpp
#include <ranges>


using namespace std::ranges;
```

我们将保留定义商家、物品和商店的代码不变。让我们通过使用`get_featured_items_for_store`函数开始我们的修改：

```cpp
Items get_featured_items_for_store(const Store &store) {

  auto items = store.items | views::filter(&Item::featured) |

               views::transform([](const auto &item) {

                 return gsl::not_null<const Item *>(&item);

               });

  return Items(std::begin(items), std::end(items));

}
```

如您所见，将容器转换为范围很简单：只需将其传递给管道运算符。我们可以使用`views::filter`表达式而不是手工筛选特色元素的循环，将成员指针作为谓词传递。由于底层的`std::invoke`的魔法，这将正确地过滤掉所有具有我们的布尔数据成员设置为`false`的项目。

接下来，我们需要将每个项目转换为`gsl::not_null`指针，以便我们可以避免不必要的项目复制。最后，我们返回一个这样的指针向量，与我们的基本代码相同。

现在，让我们看看如何使用前面的函数来获取所有商店的特色物品：

```cpp
Items get_all_featured_items(const Stores &stores) {

  auto all_featured = stores | views::transform([](auto elem) {

                        return get_featured_items_for_store(*elem);

                      });


  auto ret = Items{};

  for_each(all_featured, & {

    ret.reserve(ret.size() + elem.size());

    copy(elem, std::back_inserter(ret));

  });

  return ret;

}
```

在这里，我们从所有商店创建了一个范围，并使用我们在上一步中创建的函数进行了转换。因为我们需要先解引用每个元素，所以我们使用了一个辅助 lambda。视图是惰性评估的，因此每次转换只有在即将被消耗时才会执行。这有时可以节省大量的时间和计算：假设您只想要前 N 个项目，您可以跳过对`get_featured_items_for_store`的不必要调用。

一旦我们有了惰性视图，类似于我们的基本实现，我们可以在向量中保留空间，并从`all_featured`视图中的每个嵌套向量中将项目复制到那里。如果您使用整个容器，范围算法更简洁。看看`copy`不需要我们编写`std::begin(elem)`和`std::end(elem)`。

现在我们有了我们的物品，让我们通过使用范围来简化我们的排序代码来处理它们：

```cpp
void order_items_by_date_added(Items &items) {

  sort(items, greater{}, &Item::date_added);

}
```

再次，您可以看到范围如何帮助您编写更简洁的代码。前面的复制和排序都是范围*算法*，而不是*视图*。它们是急切的，并允许您使用投影。在我们的情况下，我们只是传递了我们物品类的另一个成员，以便在排序时可以使用它进行比较。实际上，每个项目将被投影为其`date_added`，然后使用`greater{}`进行比较。

但等等 - 我们的 items 实际上是指向`Item`的`gsl::not_null`指针。这是如何工作的？事实证明，由于`std::invoke`的巧妙之处，我们的投影将首先解引用`gsl::not_null`指针。很巧妙！

我们可以进行的最后一个更改是在我们的“渲染”代码中：

```cpp
void render_item_gallery([[maybe_unused]] const Items &items) {

  copy(items,

       std::ostream_iterator<gsl::not_null<const Item *>>(std::cout, "\n"));

}
```

在这里，范围只是帮助我们删除一些样板代码。

当您运行我们更新版本的代码时，您应该得到与基本情况相同的输出。

如果您期望范围比简洁的代码更多，那么有好消息：它们在我们的情况下甚至可以更有效地使用。

### 使用范围减少内存开销并提高性能

您已经知道，在`std::ranges::views`中使用惰性求值可以通过消除不必要的计算来提高性能。事实证明，我们还可以使用范围来减少我们示例中的内存开销。让我们重新审视一下从商店获取特色商品的代码。它可以缩短为以下内容：

```cpp
auto get_featured_items_for_store(const Store &store) {

  return store.items | views::filter(&Item::featured) |

         views::transform(

             [](const auto &item) { return gsl::not_null(&item); });

}
```

请注意，我们的函数不再返回 items，而是依赖于 C++14 的自动返回类型推导。在我们的情况下，我们的代码将返回一个惰性视图，而不是返回一个向量。

让我们学习如何为所有商店使用这个：

```cpp
Items get_all_featured_items(const Stores &stores) {

  auto all_featured = stores | views::transform([](auto elem) {

                        return get_featured_items_for_store(*elem);

                      }) |

                      views::join;

  auto as_items = Items{};

  as_items.reserve(distance(all_featured));

  copy(all_featured, std::back_inserter(as_items));

  return as_items;

}
```

现在，因为我们之前的函数返回的是一个视图而不是向量，在调用`transform`之后，我们得到了一个视图的视图。这意味着我们可以使用另一个标准视图，称为 join，将我们的嵌套视图合并成一个统一的视图。

接下来，我们使用`std::ranges::distance`在目标向量中预先分配空间，然后进行复制。有些范围是有大小的，这种情况下您可以调用`std::ranges::size`。最终的代码只调用了一次`reserve`，这应该给我们带来良好的性能提升。

这结束了我们对代码引入范围的介绍。由于我们在这一部分结束时提到了与性能相关的内容，让我们谈谈 C++编程这一方面的另一个重要主题。

# 将计算移动到编译时

从 21 世纪初现代 C++的出现开始，C++编程变得更多地关于在编译期间计算事物，而不是将它们推迟到运行时。在编译期间检测错误要比以后调试错误要便宜得多。同样，在程序启动之前准备好结果要比以后计算结果要快得多。

起初，有模板元编程，但是从 C++11 开始，每个新标准都为编译时计算带来了额外的功能：无论是类型特征、诸如`std::enable_if`或`std::void_t`的构造，还是 C++20 的`consteval`用于仅在编译时计算的内容。

多年来改进的一个功能是`constexpr`关键字及其相关代码。C++20 真正改进并扩展了`constexpr`。现在，您不仅可以编写常规的简单`constexpr`函数，还可以在其中使用动态分配和异常，更不用说`std::vector`和`std::string`了！

还有更多：甚至虚函数现在也可以是`constexpr`：重载分辨率照常进行，但如果给定的函数是`constexpr`，它可以在编译时调用。

标准算法也进行了另一个改进。它们的非并行版本都已准备好供您在编译时代码中使用。考虑以下示例，它可以用于检查容器中是否存在给定的商家：

```cpp
#include <algorithm>

#include <array>


struct Merchant { int id; };


bool has_merchant(const Merchant &selected) {

  auto merchants = std::array{Merchant{1}, Merchant{2}, Merchant{3},

                              Merchant{4}, Merchant{5}};

  return std::binary_search(merchants.begin(), merchants.end(), selected,

                            [](auto a, auto b) { return a.id < b.id; });

}
```

正如您所看到的，我们正在对商家数组进行二进制搜索，按其 ID 排序。

为了深入了解代码及其性能，我们建议您快速查看此代码生成的汇编代码。随着编译时计算和性能追求的到来，开发的一个无价的工具之一是[`godbolt.org`](https://godbolt.org)网站。它可以用于快速测试代码，以查看不同架构、编译器、标志、库版本和实现如何影响生成的汇编代码。

我们使用 GCC trunk（在 GCC 11 正式发布之前）进行了上述代码的测试，使用了`-O3`和`--std=c++2a`。在我们的情况下，我们使用以下代码检查了生成的汇编代码：

```cpp
int main() { return has_merchant({4}); }
```

您可以通过以下 Godbolt 查看几十行汇编代码：[`godbolt.org/z/PYMTYx`](https://godbolt.org/z/PYMTYx)。

*但是* - 您可能会说*汇编中有一个函数调用，所以也许我们可以内联它，这样它可以更好地优化？* 这是一个有效的观点。通常，这会有很大帮助，尽管现在，我们只是将汇编内联（参见：[`godbolt.org/z/hPadxd`](https://godbolt.org/z/hPadxd)）。

现在，尝试将签名更改为以下内容：

```cpp
constexpr bool has_merchant(const Merchant &selected) 
```

`constexpr`函数隐式地是内联的，因此我们删除了该关键字。如果我们查看汇编代码，我们会发现发生了一些魔法：搜索被优化掉了！正如您在[`godbolt.org/z/v3hj3E`](https://godbolt.org/z/v3hj3E)中所看到的，剩下的所有汇编代码如下：

```cpp
main:

        mov     eax, 1

        ret
```

编译器优化了我们的代码，以便只剩下我们预先计算的结果被返回。这相当令人印象深刻，不是吗？

## 通过使用 const 来帮助编译器帮助您

编译器可以进行相当好的优化，即使您没有给它们`inline`或`constexpr`关键字，就像前面的例子一样。帮助它们为您实现性能的一件事是将变量和函数标记为`const`。也许更重要的是，它还可以帮助您避免在代码中犯错误。许多语言默认具有不可变变量，这可以减少错误，使代码更易于理解，并且通常可以获得更快的多线程性能。

尽管 C++默认具有可变变量，并且您需要明确地输入`const`，但我们鼓励您这样做。它确实可以帮助您停止犯与修改变量有关的棘手拼写错误。

使用`const`（或`constexpr`）代码是类型安全哲学的一部分。让我们谈谈它。

# 利用安全类型的力量

C++在很大程度上依赖于帮助您编写类型安全代码的机制。诸如显式构造函数和转换运算符之类的语言构造已经被内置到语言中很长时间了。越来越多的安全类型被引入到标准库中。有`optional`可以帮助您避免引用空值，`string_view`可以帮助您避免超出范围，`any`作为任何类型的安全包装器，只是其中之一。此外，通过其零成本抽象，建议您创建自己的类型，这些类型既有用又难以被误用。

通常，使用 C 风格的结构可能会导致不安全的代码。一个例子是 C 风格的转换。它们可以解析为`const_cast, static_cast`, `reinterpret_cast`，或者这两者之一与`const_cast`的组合。意外写入`const`对象，这是`const_cast`是未定义行为。如果从`reinterpret_cast<T>`返回的内存读取，如果 T 不是对象的原始类型（C++20 的`std::bit_cast`可以在这里帮助），也是如此。如果使用 C++转换，这两种情况都更容易避免。

当涉及类型时，C 可能过于宽松。幸运的是，C++引入了许多类型安全的替代方案来解决问题 C 构造。有流和`std::format`代替`printf`等，有`std::copy`和其他类似的算法代替不安全的`memcpy`。最后，有模板代替接受`void *`的函数（并在性能方面付出代价）。在 C++中，通过一种叫做概念的特性，模板甚至可以获得更多的类型安全。让我们看看如何通过使用它们来改进我们的代码。

## 约束模板参数

概念可以改进代码的第一种方式是使其更加通用。你还记得那些需要在一个地方改变容器类型，导致其他地方也需要改变的情况吗？如果你没有改变容器到一个完全不同语义的容器，并且你需要以不同的方式使用它，那么你的代码可能不够通用。

另一方面，你是否曾经写过一个模板或在代码中使用`auto`，然后想知道如果有人改变了底层类型，你的代码是否会出错？

概念的关键在于对你正在操作的类型施加正确级别的约束。它们约束了你的模板可以匹配的类型，并且在编译时进行检查。例如，假设你写了以下代码：

```cpp
template<typename T>

void foo(T& t) {...}
```

现在，你可以这样写：

```cpp
void foo(std::swappable auto& t) {...}
```

在这里，`foo()`必须传递一个支持`std::swap`的类型才能工作。

你还记得有些模板匹配了太多类型的情况吗？以前，你可以使用`std::enable_if`、`std::void_t`或`if constexpr`来约束它们。然而，编写`enable_if`语句有点麻烦，可能会减慢编译时间。在这里，概念再次拯救了我们，因为它们的简洁性和清晰表达了它们的意图。

C++20 中有几十个标准概念。其中大部分位于`<concepts>`头文件中，可以分为四类：

+   核心语言概念，比如`derived_from`、`integral`、`swappable`和`move_constructible`

+   比较概念，比如`boolean-testable`、`equality_comparable_with`和`totally_ordered`

+   对象概念，比如`movable`、`copyable`、`semiregular`和`regular`

+   可调用的概念，比如`invokable`、`predicate`和`strict_weak_order`

其他的概念在`<iterator>`头文件中定义。这些可以分为以下几类：

+   间接可调用的概念，比如`indirect_binary_predicate`和`indirectly_unary_invocable`

+   常见算法要求，比如`indirectly_swappable`、`permutable`、`mergeable`和`sortable`

最后，还有一些在`<ranges>`头文件中可以找到。例如`range`（duh）、`contiguous_range`和`view`。

如果这对你的需求还不够，你可以像标准定义我们刚刚涵盖的那些概念一样声明自己的概念。例如，`movable`概念的实现如下：

```cpp
template <class T>
concept movable = std::is_object_v<T> && std::move_constructible<T> && std::assignable_from<T&, T> && std::swappable<T>;
```

此外，如果你查看`std::swappable`，你会看到以下内容：

```cpp
template<class T>
concept swappable = requires(T& a, T& b) { ranges::swap(a, b); };
```

这意味着类型`T`如果`ranges::swap(a, b)`对这种类型的两个引用进行编译，则类型`T`将是`swappable`。

在定义自己的概念时，一定要确保你满足了它们的语义要求。在定义接口时指定和使用概念是对接口的消费者做出的承诺。

通常，你可以在声明中使用所谓的简写符号以缩短代码：

```cpp
void sink(std::movable auto& resource);
```

为了可读性和类型安全，建议你在约束类型时使用`auto`和概念一起使用，让你的读者知道他们正在处理的对象的类型。以这种方式编写的代码将保留类似于 auto 的通用性。你可以在常规函数和 lambda 中都使用这种方式。

使用概念的一个巨大优势是更短的错误消息。将几十行关于一个编译错误的代码减少到几行并不罕见。另一个好处是你可以在概念上进行重载。

现在，让我们回到我们的多米尼加展示的例子。这一次，我们将添加一些概念，看看它们如何改进我们的实现。

首先，让我们让`get_all_featured_items`只返回一系列项目。我们可以通过将概念添加到返回类型中来实现这一点，如下所示：

```cpp
range auto get_all_featured_items(const Stores &stores);
```

到目前为止，一切都很顺利。现在，让我们为这种类型添加另一个要求，这个要求将在调用`order_items_by_date_added`时得到执行：我们的范围必须是可排序的。`std::sortable`已经为范围迭代器定义了，但为了方便起见，让我们定义一个名为`sortable_range`的新概念：

```cpp
template <typename Range, typename Comp, typename Proj>

concept sortable_range =

    random_access_range<Range> &&std::sortable<iterator_t<Range>, Comp, Proj>;
```

与其标准库对应的是，我们可以接受一个比较器和一个投影（我们在范围中引入了它）。我们的概念由满足`random_access_range`概念的类型满足，以及具有满足前述可排序概念的迭代器。就是这么简单。

在定义概念时，您还可以使用`requires`子句来指定额外的约束。例如，如果您希望我们的范围仅存储具有`date_added`成员的元素，您可以编写以下内容：

```cpp
template <typename Range, typename Comp>

concept sortable_indirectly_dated_range =

    random_access_range<Range> &&std::sortable<iterator_t<Range>, Comp> && requires(range_value_t<Range> v) { { v->date_added }; };
```

然而，在我们的情况下，我们不需要那么多地约束类型，因为当您使用概念并定义它们时，应该留下一些灵活性，这样重用它们才有意义。

这里重要的是，您可以使用`requires`子句来指定满足概念要求的类型上应该有效调用的代码。如果您愿意，您可以指定对每个子表达式返回的类型的约束；例如，要定义可递增的内容，您可以使用以下内容：

```cpp
requires(I i) {

  { i++ } -> std::same_as<I>;

}
```

既然我们有了概念，让我们重新定义`order_items_by_date_added`函数：

```cpp
void order_items_by_date_added(

    sortable_range<greater, decltype(&Item::date_added)> auto &items) {

  sort(items, greater{}, &Item::date_added);

}
```

现在，我们的编译器将检查我们传递给它的任何范围是否是可排序的，并且包含一个可以使用`std::ranges::greater{}`进行排序的`date_added`成员。

如果我们在这里使用更受限制的概念，函数将如下所示：

```cpp
void order_items_by_date_added(

    sortable_indirectly_dated_range<greater> auto &items) {

  sort(items, greater{}, &Item::date_added);

}
```

最后，让我们重新设计我们的渲染函数：

```cpp
template <input_range Container>

requires std::is_same_v<typename Container::value_type,

                        gsl::not_null<const Item *>> void

render_item_gallery(const Container &items) {

  copy(items,

       std::ostream_iterator<typename Container::value_type>(std::cout, "\n"));

}
```

在这里，您可以看到概念名称可以在模板声明中使用，而不是`typename`关键字。在这一行的下面，您可以看到`requires`关键字也可以用来根据其特征进一步约束适当的类型。如果您不想指定一个新的概念，这可能会很方便。

概念就是这样。现在，让我们写一些模块化的 C++代码。

# 编写模块化的 C++

我们将在本章中讨论的 C++的最后一个重要特性是模块。它们是 C++20 的又一个重要补充，对构建和分区代码有很大影响。

C++现在已经使用`#include`很长时间了。然而，这种文本形式的依赖包含有其缺陷，如下所列：

+   由于需要处理大量文本（即使在预处理后，`Hello World`也有大约 50 万行代码），这很慢。这导致**一次定义规则**（**ODR**）的违反。

+   您的`includes`的顺序很重要，但不应该重要。这个问题比前一个问题严重了一倍，因为它还会导致循环依赖。

+   最后，很难封装那些只需要在头文件中的东西。即使您将一些东西放在一个详细的命名空间中，也会有人使用它，正如海伦姆定律所预测的那样。

幸运的是，这就是模块进入游戏的时候。它们应该解决前面提到的缺陷，为构建时间带来巨大的加速，并在构建时提高 C++的可扩展性。使用模块，您只导出您想要导出的内容，这会带来良好的封装。依赖包含的特定顺序也不再是问题，因为导入的顺序不重要。

不幸的是，在撰写本文时，模块的编译器支持仍然只是部分完成。这就是为什么我们决定只展示 GCC 11 中已经可用的内容。遗憾的是，这意味着诸如模块分区之类的内容在这里不会涉及。

每个模块在编译后都将被编译成对象文件和模块接口文件。这意味着编译器不需要解析所有依赖项的文件，就可以快速知道给定模块包含的类型和函数。您只需要输入以下内容：

```cpp
import my_module;
```

一旦`my_module`被编译并可用，您就可以使用它。模块本身应该在一个`.cppm`文件中定义，但目前 CMake 还不支持这一点。您最好暂时将它们命名为`.cpp`。

话不多说，让我们回到我们多米尼加展会的例子，展示如何在实践中使用它们。

首先，让我们为客户代码创建我们的第一个模块，从以下指令开始：

```cpp
module;
```

这个语句表示从这一点开始，这个模块中的所有内容都将是私有的。这是一个很好的放置包含和其他不会被导出的内容的地方。

接下来，我们必须指定导出模块的名称：

```cpp
export module customer;
```

这将是我们稍后导入模块时要使用的名称。这行必须出现在导出的内容之前。现在，让我们指定我们的模块实际上将导出什么，使用`export`关键字给定义加上前缀：

```cpp
export using CustomerId = int;


export CustomerId get_current_customer_id() { return 42; }
```

搞定了！我们的第一个模块已经准备好可以使用了。让我们为商家创建另一个模块：

```cpp
module;


export module merchant;


export struct Merchant {

  int id;

};
```

与我们的第一个模块非常相似，这里我们指定了要导出的名称和类型（与第一个模块的类型别名和函数相反）。您也可以导出其他定义，比如模板。不过，对于宏来说会有些棘手，因为您需要导入`<header_file>`才能看到它们。

顺便说一句，模块的一个很大优势是它们不允许宏传播到导入的模块。这意味着当您编写以下代码时，模块不会定义`MY_MACRO`：

```cpp
#define MY_MACRO

import my_module;
```

模块中的确定性有助于保护您免受其他模块中代码的破坏。

现在，让我们为我们的商店和商品定义第三个模块。我们不会讨论导出其他函数、枚举和其他类型，因为这与前两个模块没有区别。有趣的是模块文件的开始方式。首先，让我们在私有模块部分包含我们需要的内容：

```cpp
module;


#include <chrono>

#include <iomanip>

#include <optional>

#include <string>

#include <vector>
```

在 C++20 中，标准库头文件还不是模块，但这很可能会在不久的将来发生改变。

现在，让我们看看接下来会发生什么：

```cpp
export module store;


export import merchant;
```

这是有趣的部分。我们的商店模块导入了我们之前定义的商家模块，然后将其重新导出为商店的接口的一部分。如果您的模块是其他模块的外观，这可能会很方便，比如在不久的将来的模块分区中（也是 C++20 的一部分）。一旦可用，您将能够将模块分割成多个文件。其中一个文件可以包含以下内容：

```cpp
export module my_module:foo;


export template<typename T> foo() {}
```

正如我们之前讨论的，然后它将由您的模块的主文件导出如下：

```cpp
export module my_module;


export import :foo;
```

这结束了模块和我们在本章计划的 C++的重要特性。让我们总结一下我们学到了什么。

# 总结

在本章中，我们学习了许多 C++特性及其对编写简洁、表达力强和高性能的 C++代码的影响。我们学习了如何提供适当的 C++组件接口。您现在可以应用诸如 RAII 之类的原则，编写优雅的、没有资源泄漏的代码。您还知道如何利用`std::optional`等类型在接口中更好地表达您的意图。

接下来，我们演示了如何使用通用和模板 lambda，以及`if constexpr`来编写能够适用于许多类型的少量代码。现在，您还可以使用指定的初始化程序以清晰的方式定义对象。

之后，您学会了如何使用标准范围以声明式风格编写简单的代码，如何编写可以在编译时和运行时执行的代码，以及如何使用概念编写更受限制的模板代码。

最后，我们演示了如何使用 C++模块编写模块化代码。在下一章中，我们将讨论如何设计 C++代码，以便我们可以建立在可用的习惯用法和模式之上。

# 问题

1.  我们如何确保我们的代码将打开的每个文件在不再使用时都会关闭？

1.  在 C++代码中何时应该使用“裸”指针？

1.  什么是推导指南？

1.  何时应该使用`std::optional`和`gsl::not_null`？

1.  范围算法与视图有何不同？

1.  在定义函数时，除了指定概念的名称之外，如何通过其他方式约束类型？

1.  `import X`与`import <X>`有何不同？

# 进一步阅读

+   *C++核心指南*，*概念*部分：[`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rt-concepts`](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rt-concepts)

+   libstdc++对`std::chrono`的实现：[`code.woboq.org/gcc/libstdc++-v3/include/std/chrono.html`](https://code.woboq.org/gcc/libstdc++-v3/include/std/chrono.html)
