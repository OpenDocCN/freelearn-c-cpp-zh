# 5

# 算法

标准库中容器的使用在 C++程序员中被广泛采用。很少能找到没有引用`std::vector`或`std::string`等的 C++代码库。然而，在我的经验中，标准库算法的使用频率要低得多，尽管它们提供了与容器相同类型的好处：

+   在解决复杂问题时可以用作构建块

+   它们有很好的文档（包括参考资料、书籍和视频）

+   许多 C++程序员已经熟悉它们

+   它们的空间和运行时成本是已知的（复杂度保证）

+   它们的实现非常精心和高效

如果这还不够，C++的特性，比如 lambda、执行策略、概念和范围，都使标准算法更加强大，同时也更加友好。

在本章中，我们将看看如何使用算法库在 C++中编写高效的算法。您将学习在应用程序中使用标准库算法作为构建块的好处，无论是性能还是可读性方面。

在本章中，您将学习：

+   C++标准库中的算法

+   迭代器和范围-容器和算法之间的粘合剂

+   如何实现一个可以操作标准容器的通用算法

+   使用 C++标准算法的最佳实践

让我们首先看一下标准库算法，以及它们如何成为今天的样子。

# 介绍标准库算法

将标准库算法集成到您的 C++词汇表中是很重要的。在本介绍中，我将介绍一组可以通过使用标准库算法有效解决的常见问题。

C++20 通过引入 Ranges 库和*C++概念*的语言特性对算法库进行了重大改变。因此，在我们开始之前，我们需要简要了解 C++标准库的历史背景。

## 标准库算法的演变

您可能已经听说过 STL 算法或 STL 容器。希望您也已经听说了 C++20 引入的新的 Ranges 库。在 C++20 中，标准库有很多新增内容。在继续之前，我需要澄清一些术语。我们将从 STL 开始。

STL，或者标准模板库，最初是在上世纪 90 年代添加到 C++标准库中的一个库的名称。它包含算法、容器、迭代器和函数对象。这个名字一直很粘人，我们已经习惯了听到和谈论 STL 算法和容器。然而，C++标准并没有提到 STL；相反，它谈到了标准库及其各个组件，比如迭代器库和算法库。在本书中，我会尽量避免使用 STL 这个名字，而是在需要时谈论标准库或单独的库。

现在让我们来看看 Ranges 库以及我将称之为*受限算法*。Ranges 库是 C++20 中添加到标准库的一个库，引入了一个全新的头文件`<ranges>`，我们将在下一章中更多地谈论它。但是，Ranges 库的添加也对`<algorithm>`头文件产生了很大影响，通过引入所有先前存在的算法的重载版本。我将这些算法称为*受限算法*，因为它们使用了 C++概念进行限制。因此，`<algorithm>`头文件现在包括了旧的基于迭代器的算法和可以操作范围的使用 C++概念限制的新算法。这意味着我们将在本章讨论的算法有两种风味，如下例所示：

```cpp
#include <algorithm>
#include <vector>
auto values = std::vector{9, 2, 5, 3, 4};
// Sort using the std algorithms
std::sort(values.begin(), values.end());
// Sort using the constrained algorithms under std::ranges
std::ranges::sort(values); 
std::ranges::sort(values.begin(), values.end()); 
```

请注意，`sort()`的两个版本都位于`<algorithm>`头文件中，但它们由不同的命名空间和签名区分。本章将使用这两种版本，但一般来说，我建议尽可能使用新的约束算法。在阅读本章后，这些好处将会变得明显。

现在你已经准备好开始学习如何使用现成的算法来解决常见问题了。

## 解决日常问题

我在这里列出了一些常见的场景和有用的算法，只是为了让你对标准库中可用的算法有所了解。标准库中有许多算法，在本节中我只会介绍其中的一些。对于标准库算法的快速但完整的概述，我推荐 Jonathan Boccara 在*CppCon 2018*上的演讲，题为*Less Than an Hour*，可在[`sched.co/FnJh`](https://sched.co/FnJh)上找到。

### 遍历序列

有一个有用的短小的辅助函数，可以打印序列的元素。下面的通用函数适用于任何容器，其中包含可以使用`operator<<()`打印到输出流的元素：

```cpp
void print(auto&& r) {
  std::ranges::for_each(r, [](auto&& i) { std::cout << i << ' '; });
} 
```

`print()`函数使用了`for_each()`，这是从`<algorithm>`头文件导入的算法。`for_each()`为我们提供的函数为范围中的每个元素调用一次。我们提供的函数的返回值被忽略，并且对我们传递给`for_each()`的序列没有影响。我们可以使用`for_each()`来进行诸如打印到`stdout`之类的副作用（就像在这个例子中所做的那样）。

一个类似的非常通用的算法是`transform()`。它也为序列中的每个元素调用一个函数，但它不会忽略返回值，而是将函数的返回值存储在输出序列中，就像这样：

```cpp
auto in = std::vector{1, 2, 3, 4};
auto out = std::vector<int>(in.size());
auto lambda = [](auto&& i) { return i * i; };
std::ranges::transform(in, out.begin(), lambda);
print(out); 
// Prints: "1 4 9 16" 
print() function defined earlier. The transform() algorithm will call our lambda once for each element in the input range. To specify where the output will be stored, we provide transform() with an output iterator, out.begin(). We will talk a lot more about iterators later on in this chapter.
```

有了我们的`print()`函数和一些最常见的算法演示，我们将继续看一些用于生成元素的算法。

### 生成元素

有时我们需要为一系列元素分配一些初始值或重置整个序列。下面的例子用值-1 填充了一个向量：

```cpp
auto v = std::vector<int>(4);
std::ranges::fill(v, -1);
print(v); 
// Prints "-1 -1 -1 -1 " 
```

下一个算法`generate()`为每个元素调用一个函数，并将返回值存储在当前元素中：

```cpp
auto v = std::vector<int>(4);
std::ranges::generate(v, std::rand);
print(v);
// Possible output: "1804289383 846930886 1681692777 1714636915 " 
```

在前面的例子中，`std::rand()`函数被每个元素调用了一次。

我要提到的最后一个生成算法是`<numeric>`头文件中的`std::iota()`。它按递增顺序生成值。起始值必须作为第二个参数指定。下面是一个生成 0 到 5 之间值的简短示例：

```cpp
 auto v = std::vector<int>(6);
  std::iota(v.begin(), v.end(), 0);
  print(v); // Prints: "0 1 2 3 4 5 " 
```

这个序列已经排序好了，但更常见的情况是你有一个无序的元素集合需要排序，接下来我们会看一下。

### 元素排序

排序元素是一个非常常见的操作。有一些好的排序算法替代方案是值得了解的，但在这个介绍中，我只会展示最常规的版本，简单地命名为`sort()`：

```cpp
auto v = std::vector{4, 3, 2, 3, 6};
std::ranges::sort(v);
print(v);       // Prints: "2 3 3 4 6 " 
```

如前所述，这不是唯一的排序方式，有时我们可以使用部分排序算法来提高性能。我们将在本章后面更多地讨论排序。

### 查找元素

另一个非常常见的任务是找出特定值是否在集合中。也许我们想知道集合中有多少个特定值的实例。如果我们知道集合已经排序，那么搜索值的这些算法可以更有效地实现。你在*第三章*，*分析和测量性能*中看到了这一点，我们比较了线性搜索和二分搜索。

我们从不需要排序的`find()`算法开始：

```cpp
auto col = std::list{2, 4, 3, 2, 3, 1};
auto it = std::ranges::find(col, 2);
if (it != col.end()) {
  std::cout << *it << '\n';
} 
```

如果找不到我们要找的元素，`find()`会返回集合的`end()`迭代器。在最坏的情况下，`find()`需要检查序列中的所有元素，因此它的运行时间为*O(n)*。

### 使用二分查找进行查找

如果我们知道集合已经排序，我们可以使用二分搜索算法之一：`binary_search()`、`equal_range()`、`upper_bound()`或`lower_bound()`。如果我们将这些函数与提供对其元素进行随机访问的容器一起使用，它们都保证在*O(log n)*时间内运行。当我们在本章后面讨论迭代器和范围时（有一个名为*Iterators and Ranges*的部分即将到来），你将更好地理解算法如何提供复杂度保证，即使它们在不同的容器上操作。

在以下示例中，我们将使用一个排序的`std::vector`，其中包含以下元素：

![](img/B15619_05_01.png)

图 5.1：一个包含七个元素的排序 std::vector

`binary_search()`函数根据我们搜索的值是否能找到返回`true`或`false`：

```cpp
auto v = std::vector{2, 2, 3, 3, 3, 4, 5};    // Sorted!
bool found = std::ranges::binary_search(v, 3);
std::cout << std::boolalpha << found << '\n'; //   Output: true 
```

在调用`binary_search()`之前，你应该绝对确定集合是排序的。我们可以在代码中使用`is_sorted()`轻松断言这一点，如下所示：

```cpp
assert(std::ranges::is_sorted(v)); 
```

这个检查将在*O(n)*时间内运行，但只有在激活断言时才会被调用，因此不会影响最终程序的性能。

我们正在处理的排序集合包含多个 3。如果我们想知道集合中第一个 3 或最后一个 3 的位置，我们可以使用`lower_bound()`来找到第一个 3，或者使用`upper_bound()`来找到最后一个 3 之后的元素：

```cpp
auto v = std::vector{2, 2, 3, 3, 3, 4, 5};
auto it = std::ranges::lower_bound(v, 3);
if (it != v.end()) {
  auto index = std::distance(v.begin(), it);
  std::cout << index << '\n'; // Output: 2
} 
```

这段代码将输出`2`，因为这是第一个 3 的索引。要从迭代器获取元素的索引，我们使用`<iterator>`头文件中的`std::distance()`。

同样地，我们可以使用`upper_bound()`来获取一个迭代器，指向最后一个 3 之后的元素：

```cpp
const auto v = std::vector{2, 2, 3, 3, 3, 4, 5};
auto it = std::ranges::upper_bound(v, 3);
if (it != v.end()) {
  auto index = std::distance(v.begin(), it);
  std::cout << index << '\n'; // Output: 5
} 
```

如果你想要上下界，你可以使用`equal_range()`，它返回包含 3 的子范围：

```cpp
const auto v = std::vector{2, 2, 3, 3, 3, 4, 5};
auto subrange = std::ranges::equal_range(v, 3);
if (subrange.begin() != subrange.end()) {
  auto pos1 = std::distance(v.begin(), subrange.begin());
  auto pos2 = std::distance(v.begin(), subrange.end());
  std::cout << pos1 << " " << pos2 << '\n';
} // Output: "2 5" 
```

现在让我们探索一些用于检查集合的其他有用算法。

### 测试特定条件

有三个非常方便的算法叫做`all_of()`、`any_of()`和`none_of()`。它们都接受一个范围、一个一元谓词（接受一个参数并返回`true`或`false`的函数）和一个可选的投影函数。

假设我们有一个数字列表和一个小 lambda 函数，确定一个数字是否为负数：

```cpp
const auto v = std::vector{3, 2, 2, 1, 0, 2, 1};
const auto is_negative = [](int i) { return i < 0; }; 
```

我们可以使用`none_of()`来检查是否没有任何数字是负数：

```cpp
if (std::ranges::none_of(v, is_negative)) {
  std::cout << "Contains only natural numbers\n";
} 
```

此外，我们可以使用`all_of()`来询问列表中的所有元素是否都是负数：

```cpp
if (std::ranges::all_of(v, is_negative)) {
  std::cout << "Contains only negative numbers\n";
} 
```

最后，我们可以使用`any_of()`来查看列表是否至少包含一个负数：

```cpp
if (std::ranges::any_of(v, is_negative)) {
  std::cout << "Contains at least one negative number\n";
} 
```

很容易忘记标准库中存在的这些小而方便的构建块。但一旦你养成使用它们的习惯，你就再也不会回头手写这些了。

### 计算元素

计算等于某个值的元素数量最明显的方法是调用`count()`：

```cpp
const auto numbers = std::list{3, 3, 2, 1, 3, 1, 3};
int n = std::ranges::count(numbers, 3);
std::cout << n;                    // Prints: 4 
```

`count()`算法运行时间为线性。然而，如果我们知道序列是排序的，并且我们使用的是向量或其他随机访问数据结构，我们可以使用`equal_range()`，它将在*O(log n)*时间内运行。以下是一个例子：

```cpp
const auto v = std::vector{0, 2, 2, 3, 3, 4, 5};
assert(std::ranges::is_sorted(v)); // O(n), but not called in release
auto r = std::ranges::equal_range(v, 3);
int n = std::ranges::size(r);
std::cout << n;                    // Prints: 2 
```

`equal_range()`函数找到包含我们要计数的所有元素的子范围。一旦找到子范围，我们可以使用`<ranges>`头文件中的`size()`来检索子范围的长度。

### 最小值、最大值和夹紧

我想提到一组小但非常有用的算法，这些算法对于经验丰富的 C++程序员来说是必不可少的知识。`std::min()`、`std::max()`和`std::clamp()`函数有时会被遗忘，而我们经常发现自己编写这样的代码：

```cpp
const auto y_max = 100;
auto y = some_func();
if (y > y_max) {
  y = y_max;
} 
```

该代码确保`y`的值在某个限制范围内。这段代码可以工作，但我们可以避免使用可变变量和`if`语句，而是使用`std::min()`，如下所示：

```cpp
const auto y = std::min(some_func(), y_max); 
```

通过使用`std::min()`，我们消除了代码中的可变变量和`if`语句。对于类似的情况，我们可以使用`std::max()`。如果我们想要将一个值限制在最小值和最大值之间，我们可以这样做：

```cpp
const auto y = std::max(std::min(some_func(), y_max), y_min); 
```

但是，自 C++17 以来，我们现在有了`std::clamp()`，它可以在一个函数中为我们完成这个操作。因此，我们可以像下面这样使用`clamp()`：

```cpp
const auto y = std::clamp(some_func(), y_min, y_max); 
```

有时我们需要在未排序的元素集合中找到极值。为此，我们可以使用`minmax()`，它（不出所料地）返回序列的最小值和最大值。结合结构化绑定，我们可以按如下方式打印极值：

```cpp
const auto v = std::vector{4, 2, 1, 7, 3, 1, 5};
const auto [min, max] = std::ranges::minmax(v);
std::cout << min << " " << max;      // Prints: "1 7" 
```

我们还可以使用`min_element()`或`max_element()`找到最小或最大元素的位置。它不返回值，而是返回一个指向我们要查找的元素的迭代器。在下面的例子中，我们正在寻找最小元素：

```cpp
const auto v = std::vector{4, 2, 7, 1, 1, 3};
const auto it = std::ranges::min_element(v);
std::cout << std::distance(v.begin(), it); // Output: 3 
3, which is the index of the first minimum value that was found.
```

这是对标准库中一些最常见算法的简要介绍。算法的运行时成本在 C++标准中有规定，所有库实现都需要遵守这些规定，尽管确切的实现可能在不同的平台之间有所不同。为了理解如何保持与许多不同类型的容器一起工作的通用算法的复杂性保证，我们需要更仔细地研究迭代器和范围。

# 迭代器和范围

正如前面的例子所示，标准库算法操作的是迭代器和范围，而不是容器类型。本节将重点介绍迭代器和 C++20 中引入的新概念范围。一旦掌握了迭代器和范围，正确使用容器和算法就变得容易了。

## 介绍迭代器

迭代器构成了标准库算法和范围的基础。迭代器是数据结构和算法之间的粘合剂。正如你已经看到的，C++容器以非常不同的方式存储它们的元素。迭代器提供了一种通用的方式来遍历序列中的元素。通过让算法操作迭代器而不是容器类型，算法变得更加通用和灵活，因为它们不依赖于容器的类型以及容器在内存中排列元素的方式。

在本质上，迭代器是表示序列中位置的对象。它有两个主要责任：

+   在序列中导航

+   在当前位置读取和写入值

迭代器抽象根本不是 C++独有的概念，而是存在于大多数编程语言中。C++实现迭代器概念的不同之处在于，C++模仿了原始内存指针的语法。

基本上，迭代器可以被认为是具有与原始指针相同属性的对象；它可以移动到下一个元素并解引用（如果指向有效地址）。算法只使用指针允许的一些操作，尽管迭代器可能在内部是一个遍历类似树状的`std::map`的重对象。

直接在`std`命名空间下找到的大多数算法只对迭代器进行操作，而不是容器（即`std::vector`，`std::map`等）。许多算法返回的是迭代器而不是值。

为了能够在序列中导航而不越界，我们需要一种通用的方法来告诉迭代器何时到达序列的末尾。这就是我们有哨兵值的原因。

## 哨兵值和超出末尾的迭代器

**哨兵值**（或简称哨兵）是指示序列结束的特殊值。哨兵值使得可以在不知道序列大小的情况下迭代一系列值。哨兵值的一个示例用法是 C 风格的以 null 结尾的字符串（在这种情况下，哨兵是`'\0'`字符）。不需要跟踪以 null 结尾的字符串的长度，字符串开头的指针和末尾的哨兵就足以定义一系列字符。

约束算法使用迭代器来定义序列中的第一个元素，并使用哨兵来指示序列的结束。哨兵的唯一要求是它可以与迭代器进行比较，实际上意味着`operator==()`和`operator!=()`应该被定义为接受哨兵和迭代器的组合：

```cpp
bool operator=!(sentinel s, iterator i) {
  // ...
} 
```

现在你知道了哨兵是什么，我们如何创建一个哨兵来指示序列的结束呢？这里的诀窍是使用一个叫做**past-the-end** **iterator**的迭代器作为哨兵。它只是一个指向我们定义的序列中最后一个元素之后（或过去）的迭代器。看一下下面的代码片段和图表：

|

```cpp
auto vec = std::vector {
  'a','b','c','d'
};
auto first = vec.begin();
auto last = vec.end(); 
```

| ![](img/B15619_05_02.png) |
| --- |

如前图所示，`last`迭代器现在指向了一个想象中的`'d'`元素之后。这使得可以通过循环迭代序列中的所有元素：

```cpp
for (; first != last; ++first) {
  char value = *first; // Dereference iterator
  // ... 
```

我们可以使用 past-the-end 哨兵与我们的迭代器`it`进行比较，但是我们不能对哨兵进行解引用，因为它不指向范围的元素。这种 past-the-end 迭代器的概念有着悠久的历史，甚至适用于内置的 C 数组：

```cpp
char arr[] = {'a', 'b', 'c', 'd'};
char* end = arr + sizeof(arr);
for (char* it = arr; it != end; ++it) { // Stop at end
   std::cout << *it << ' ';} 
// Output: a b c d 
```

再次注意，`end`实际上指向了越界，因此我们不允许对其进行解引用，但是我们允许读取指针值并将其与我们的`it`变量进行比较。

## 范围

范围是指我们在引用一系列元素时使用的迭代器-哨兵对的替代品。`<range>`头文件包含了定义不同种类范围要求的多个概念，例如`input_range`，`random_access_range`等等。这些都是最基本概念`range`的细化，它的定义如下：

```cpp
template<class T>
concept range = requires(T& t) {
  ranges::begin(t);
  ranges::end(t);
}; 
```

这意味着任何暴露`begin()`和`end()`函数的类型都被认为是范围（假设这些函数返回迭代器）。

对于 C++标准容器，`begin()`和`end()`函数将返回相同类型的迭代器，而对于 C++20 范围，这通常不成立。具有相同迭代器和哨兵类型的范围满足`std::ranges::common_range`的概念。新的 C++20 视图（在下一章中介绍）返回可以是不同类型的迭代器-哨兵对。但是，它们可以使用`std::views::common`转换为具有相同迭代器和哨兵类型的视图。

在`std::ranges`命名空间中找到的约束算法可以操作范围而不是迭代器对。由于所有标准容器（`vector`，`map`，`list`等）都满足范围概念，因此我们可以直接将范围传递给约束算法，如下所示：

```cpp
auto vec = std::vector{1, 1, 0, 1, 1, 0, 0, 1};
std::cout << std::ranges::count(vec, 0); // Prints 3 
```

范围是可迭代的东西的抽象（可以循环遍历的东西），在某种程度上，它们隐藏了对 C++迭代器的直接使用。然而，迭代器仍然是 C++标准库的一个重要部分，并且在 Ranges 库中也被广泛使用。

你需要理解的下一件事是存在的不同种类的迭代器。

## 迭代器类别

现在你对范围的定义以及如何知道何时到达序列的末尾有了更好的理解，是时候更仔细地看一下迭代器可以支持的操作，以便导航，读取和写入值。

在序列中进行迭代器导航可以使用以下操作：

+   向前移动：`std::next(it)`或`++it`

+   向后移动：`std::prev(it)`或`--it`

+   跳转到任意位置：`std::advance(it, n)`或`it += n`

通过*解引用*迭代器来读取和写入迭代器表示的位置的值。下面是它的样子：

+   阅读：`auto value = *it`

+   写入：`*it = value`

这些是容器公开的迭代器的最常见操作。但此外，迭代器可能在数据源上操作，其中写入或读取意味着向前移动。这些数据源的示例可能是用户输入，网络连接或文件。这些数据源需要以下操作：

+   只读和向前移动：`auto value = *it; ++it;`

+   只写和向前移动：`*it = value; ++it;`

这些操作只能用两个连续的表达式来表示。第一个表达式的后置条件是第二个表达式必须有效。这也意味着我们只能读取或写入一个值到一个位置一次。如果我们想要读取或写入一个新值，我们必须先将迭代器推进到下一个位置。

并非所有迭代器都支持前述列表中的所有操作。例如，一些迭代器只能*读取*值和*向前移动*，而其他一些既可以*读取*，*写入*，又可以*跳转*到任意位置。

现在，如果我们考虑一些基本算法，就会显而易见地发现迭代器的要求在不同的算法之间有所不同：

+   如果算法计算值的出现次数，则需要*读取*和*向前移动*操作

+   如果算法用一个值填充容器，则需要*写入*和*向前移动*操作

+   对于排序集合上的二分搜索算法需要*读取*和*跳转*操作

一些算法可以根据迭代器支持的操作来更有效地实现。就像容器一样，标准库中的所有算法都有复杂度保证（使用大 O 表示法）。为了满足某个复杂度保证，算法对其操作的迭代器提出了*要求*。这些要求被归类为六种基本迭代器类别，它们之间的关系如下图所示：

![](img/B15619_05_03.png)

图 5.2：六种迭代器类别及其相互关系

箭头表示迭代器类别还具有它所指向的类别的所有功能。例如，如果一个算法需要一个前向迭代器，我们同样可以传递一个双向迭代器，因为双向迭代器具有前向迭代器的所有功能。

这六个要求由以下概念正式指定：

+   `std::input_iterator`：支持*只读和向前移动*（一次）。一次性算法如`std::count()`可以使用输入迭代器。`std::istream_iterator`是输入迭代器的一个例子。

+   `std::output_iterator`：支持*只写和向前移动*（一次）。请注意，输出迭代器只能写入，不能读取。`std::ostream_iterator`是输出迭代器的一个例子。

+   `std::forward_iterator`：支持*读取*，*写入*和*向前移动*。当前位置的值可以多次读取或写入。例如`std::forward_list`公开前向迭代器。

+   `std::bidirectional_iterator`：支持*读取*，*写入*，*向前移动*和*向后移动*。双向链表`std::list`公开双向迭代器。

+   `std::random_access_iterator`：支持*读取*，*写入*，*向前移动*，*向后移动*和在常数时间内*跳转*到任意位置。`std::deque`中的元素可以使用随机访问迭代器访问。

+   `std::contiguous_iterator`：与随机访问迭代器相同，但也保证底层数据是连续的内存块，例如`std::string`，`std::vector`，`std::array`，`std::span`和（很少使用的）`std::valarray`。

迭代器类别对于理解算法的时间复杂度要求非常重要。对底层数据结构有很好的理解，可以很容易地知道哪些迭代器通常属于哪些容器。

现在我们准备深入了解大多数标准库算法使用的常见模式。

# 标准算法的特性

为了更好地理解标准算法，了解一些`<algorithm>`头文件中所有算法使用的特性和常见模式是很有帮助的。正如已经提到的，`std`和`std::ranges`命名空间下的算法有很多共同之处。我们将从这里开始讨论适用于`std`算法和`std::range`下受限算法的通用原则。然后，在下一节中，我们将继续讨论`std::ranges`下特有的特性。

## 算法不会改变容器的大小

来自`<algorithm>`的函数只能修改指定范围内的元素；元素永远不会被添加或删除到底层容器中。因此，这些函数永远不会改变它们操作的容器的大小。

例如，`std::remove()`或`std::unique()`实际上并不会从容器中删除元素（尽管它们的名字是这样）。相反，它们将应该保留的元素移动到容器的前面，然后返回一个标记，定义了元素的有效范围的新结尾：

| 代码示例 | 结果向量 |
| --- | --- |

|

```cpp
// Example with std::remove()
auto v = std::vector{1,1,2,2,3,3};
auto new_end = std::remove(
  v.begin(), v.end(), 2);
v.erase(new_end, v.end()); 
```

| ![](img/B15619_05_04.png) |
| --- |

|

```cpp
// Example with std::unique()
auto v = std::vector{1,1,2,2,3,3};
auto new_end = std::unique(
  v.begin(), v.end());
v.erase(new_end, v.end()); 
```

| ![](img/B15619_05_05.png) |
| --- |

C++20 在`<vector>`头文件中添加了`std::erase()`和`std::erase_if()`函数的新版本，它们可以立即从向量中删除值，而无需先调用`remove()`再调用`erase()`。

标准库算法永远不会改变容器的大小，这意味着在调用产生输出的算法时，我们需要自己分配数据。

## 带有输出的算法需要已分配的数据

向输出迭代器写入数据的算法，如`std::copy()`或`std::transform()`，需要为输出预留已分配的数据。由于算法只使用迭代器作为参数，它们无法自行分配数据。为了扩大算法操作的容器，它们依赖于迭代器能够扩大它们迭代的容器。

如果将指向空容器的迭代器传递给输出算法，程序很可能会崩溃。下面的示例展示了这个问题，其中`squared`是空的：

```cpp
const auto square_func = [](int x) { return x * x; };
const auto v = std::vector{1, 2, 3, 4};
auto squared = std::vector<int>{};
std::ranges::transform(v, squared.begin(), square_func); 
```

相反，你必须执行以下操作之一：

+   为结果容器预先分配所需的大小，或者

+   使用插入迭代器，它在迭代时向容器中插入元素

以下代码片段展示了如何使用预分配的空间：

```cpp
const auto square_func = [](int x) { return x * x; };
const auto v = std::vector{1, 2, 3, 4};
auto squared = std::vector<int>{};
squared.resize(v.size());
std::ranges::transform(v, squared.begin(), square_func); 
std::back_inserter() and std::inserter() to insert values into a container that is not preallocated:
```

```cpp
const auto square_func = [](int x) { return x * x; };
const auto v = std::vector{1, 2, 3, 4};
// Insert into back of vector using std::back_inserter
auto squared_vec = std::vector<int>{};
auto dst_vec = std::back_inserter(squared_vec);
std::ranges::transform(v, dst_vec, square_func);
// Insert into a std::set using std::inserter
auto squared_set = std::set<int>{};
auto dst_set = std::inserter(squared_set, squared_set.end());
std::ranges::transform(v, dst_set, square_func); 
```

如果你正在操作`std::vector`并且知道结果容器的预期大小，可以在执行算法之前使用`reserve()`成员函数来预留空间，以避免不必要的分配。否则，在算法执行期间，向量可能会多次重新分配新的内存块。

## 算法默认使用`operator==()`和`operator<()`

作为比较，算法依赖于基本的`==`和`<`运算符，就像整数的情况一样。为了能够在算法中使用自定义类，类必须提供`operator==()`和`operator<()`，或者作为算法的参数提供。

通过使用三路比较运算符`operator<=>()`，我们可以让编译器生成必要的运算符。下面的示例展示了一个简单的`Flower`类，其中`std::find()`使用了`operator==()`，而`std::max_element()`使用了`operator<()`：

```cpp
struct Flower {
    auto operator<=>(const Flower& f) const = default; 
    bool operator==(const Flower&) const = default;
    int height_{};
};
auto garden = std::vector<Flower>{{67}, {28}, {14}};
// std::max_element() uses operator<()
auto tallest = std::max_element(garden.begin(), garden.end());
// std::find() uses operator==()
auto perfect = *std::find(garden.begin(), garden.end(), Flower{28}); 
```

除了使用当前类型的默认比较函数之外，还可以使用自定义比较函数，接下来我们将探讨这一点。

### 自定义比较函数

有时我们需要比较对象而不使用默认的比较运算符，例如在排序或按长度查找字符串时。在这些情况下，可以提供自定义函数作为额外参数。原始算法使用值（例如`std::find()`），具有特定运算符的版本在名称末尾附加了`_if`（`std::find_if()`、`std::count_if()`等）：

```cpp
auto names = std::vector<std::string> {
  "Ralph", "Lisa", "Homer", "Maggie", "Apu", "Bart"
};
std::sort(names.begin(), names.end(), 
          [](const std::string& a,const std::string& b) {
            return a.size() < b.size(); });
// names is now "Apu", "Lisa", "Bart", "Ralph", "Homer", "Maggie"
// Find names with length 3
auto x = std::find_if(names.begin(), names.end(), 
  [](const auto& v) { return v.size() == 3; });
// x points to "Apu" 
```

## 受限算法使用投影

`std::ranges`下的受限算法为我们提供了一个称为**投影**的方便功能，它减少了编写自定义比较函数的需求。前一节中的前面示例可以使用标准谓词`std::less`结合自定义投影进行重写：

```cpp
auto names = std::vector<std::string>{
  "Ralph", "Lisa", "Homer", "Maggie", "Apu", "Bart"
};
std::ranges::sort(names, std::less<>{}, &std::string::size);
// names is now "Apu", "Lisa", "Bart", "Ralph", "Homer", "Maggie"

// Find names with length 3
auto x = std::ranges::find(names, 3, &std::string::size);
// x points to "Apu" 
```

还可以将 lambda 作为投影参数传递，这在想要在投影中组合多个属性时非常方便：

```cpp
struct Player {
  std::string name_{};
  int level_{};
  float health_{};
  // ...
};
auto players = std::vector<Player>{
  {"Aki", 1, 9.f}, 
  {"Nao", 2, 7.f}, 
  {"Rei", 2, 3.f}};
auto level_and_health = [](const Player& p) {
  return std::tie(p.level_, p.health_);
}; 
// Order players by level, then health
std::ranges::sort(players, std::greater<>{}, level_and_health); 
```

向标准算法传递投影对象的可能性是一个非常受欢迎的功能，真正简化了自定义比较的使用。

## 算法要求移动操作不抛出异常

所有算法在移动元素时都使用`std::swap()`和`std::move()`，但只有在移动构造函数和移动赋值标记为`noexcept`时才会使用。因此，在使用算法时，对于重型对象来说，实现这些是很重要的。如果它们不可用且无异常，则元素将被复制而不是移动。

请注意，如果您在类中实现了移动构造函数和移动赋值运算符，`std::swap()`将利用它们，因此不需要指定`std::swap()`重载。

## 算法具有复杂性保证

标准库中每个算法的复杂度都使用大 O 表示法进行了规定。算法是以性能为目标创建的。因此，它们既不分配内存，也不具有高于*O(n log n)*的时间复杂度。即使它们是相当常见的操作，也不包括不符合这些标准的算法。

请注意`stable_sort()`、`inplace_merge()`和`stable_partition()`的异常。许多实现在这些操作期间倾向于临时分配内存。

例如，让我们考虑一个测试非排序范围是否包含重复项的算法。一种选择是通过迭代范围并搜索范围的其余部分来实现它。这将导致一个*O(n*²*)*复杂度的算法：

```cpp
template <typename Iterator>
auto contains_duplicates(Iterator first, Iterator last) {
  for (auto it = first; it != last; ++it)
    if (std::find(std::next(it), last, *it) != last)
      return true;
  return false;
} 
```

另一种选择是复制整个范围，对其进行排序，并查找相邻的相等元素。这将导致*O(n log n)*的时间复杂度，即`std::sort()`的复杂度。然而，由于它需要复制整个范围，因此仍然不符合构建块算法的条件。分配意味着我们不能相信它不会抛出异常：

```cpp
template <typename Iterator>
auto contains_duplicates(Iterator first, Iterator last) {
  // As (*first) returns a reference, we have to get 
  // the base type using std::decay_t
  using ValueType = std::decay_t<decltype(*first)>;
  auto c = std::vector<ValueType>(first, last);
  std::sort(c.begin(), c.end());
  return std::adjacent_find(c.begin(),c.end()) != c.end();
} 
```

复杂性保证从 C++标准库的一开始就是其巨大成功的主要原因之一。C++标准库中的算法是以性能为目标设计和实现的。

## 算法的性能与 C 库函数等价物一样好

标准 C 库配备了许多低级算法，包括`memcpy()`、`memmove()`、`memcmp()`和`memset()`。根据我的经验，有时人们使用这些函数而不是标准算法库中的等价物。原因是人们倾向于相信 C 库函数更快，因此接受类型安全的折衷。

这对于现代标准库实现来说是不正确的；等价算法`std::copy()`、`std::equal()`和`std::fill()`在可能的情况下会使用这些低级 C 函数；因此，它们既提供性能又提供类型安全。

当然，也许会有例外情况，C++编译器无法检测到可以安全地使用低级 C 函数的情况。例如，如果一个类型不是平凡可复制的，`std::copy()`就不能使用`memcpy()`。但这是有充分理由的；希望一个不是平凡可复制的类的作者有充分的理由以这种方式设计类，我们（或编译器）不应该忽视这一点，而不调用适当的构造函数。

有时，C++算法库中的函数甚至比它们的 C 库等效函数表现得更好。最突出的例子是`std::sort()`与 C 库中的`qsort()`。`std::sort()`和`qsort()`之间的一个重大区别是，`qsort()`是一个*函数*，而`std::sort()`是一个*函数模板*。当`qsort()`调用比较函数时，由于它是作为函数指针提供的，通常比使用`std::sort()`时调用的普通比较函数慢得多，后者可能会被编译器内联。

在本章的其余部分，我们将介绍在使用标准算法和实现自定义算法时的一些最佳实践。

# 编写和使用通用算法

算法库包含通用算法。为了尽可能具体，我将展示一个通用算法的实现示例。这将为您提供一些关于如何使用标准算法的见解，同时演示实现通用算法并不那么困难。我故意避免在这里解释示例代码的所有细节，因为我们将在本书的后面花费大量时间进行通用编程。

在接下来的示例中，我们将把一个简单的非通用算法转换为一个完整的通用算法。

## 非通用算法

通用算法是一种可以与各种元素范围一起使用的算法，而不仅仅是一种特定类型，比如`std::vector`。以下算法是一个非通用算法的例子，它只能与`std::vector<int>`一起使用：

```cpp
auto contains(const std::vector<int>& arr, int v) {
  for (int i = 0; i < arr.size(); ++i) {	
    if (arr[i] == v) { return true; }
  }
  return false;
} 
```

为了找到我们要找的元素，我们依赖于`std::vector`的接口，它为我们提供了`size()`函数和下标运算符（`operator[]()`）。然而，并非所有容器都提供这些函数，我也不建议您以这种方式编写原始循环。相反，我们需要创建一个在迭代器上操作的函数模板。

## 通用算法

通过用两个迭代器替换`std::vector`，用一个模板参数替换`int`，我们可以将我们的算法转换为通用版本。以下版本的`contains()`可以与任何容器一起使用：

```cpp
template <typename Iterator, typename T>
auto contains(Iterator begin, Iterator end, const T& v) {
  for (auto it = begin; it != end; ++it) {
    if (*it == v) { return true; }
  }
  return false;
} 
```

例如，要将其与`std::vector`一起使用，您需要传递`begin()`和`end()`迭代器：

```cpp
auto v = std::vector{3, 4, 2, 4};
if (contains(v.begin(), v.end(), 3)) {
 // Found the value...
} 
```

我们可以通过提供一个接受范围而不是两个单独迭代器参数的版本来改进这个算法：

```cpp
auto contains(const auto& r, const auto& x) {
  auto it = std::begin(r);
  auto sentinel = std::end(r);
  return contains(it, sentinel, x);
} 
```

这个算法不强制客户端提供`begin()`和`end()`迭代器，因为我们已经将其移到函数内部。我们使用了 C++20 的**缩写函数模板**语法，以避免明确说明这是一个函数模板。最后一步，我们可以为我们的参数类型添加约束：

```cpp
auto contains(const std::ranges::range auto& r, const auto& x) {
  auto it = std::begin(r);
  auto sentinel = std::end(r);
  return contains(it, sentinel, x);
} 
```

正如你所看到的，创建一个强大的通用算法实际上并不需要太多的代码。我们传递给算法的数据结构唯一的要求是它可以公开`begin()`和`end()`迭代器。您将在*第八章*“编译时编程”中了解更多关于约束和概念的知识。

## 可以被通用算法使用的数据结构

这让我们意识到，只要我们的新自定义数据结构公开`begin()`和`end()`迭代器或一个范围，它们就可以被标准通用算法使用。举个简单的例子，我们可以实现一个二维`Grid`结构，其中行被公开为一对迭代器，就像这样：

```cpp
struct Grid {
  Grid(std::size_t w, std::size_t h) : w_{w}, h_{h} {    data_.resize(w * h); 
  }
  auto get_row(std::size_t y); // Returns iterators or a range

  std::vector<int> data_{};
  std::size_t w_{};
  std::size_t h_{};
}; 
```

下图说明了带有迭代器对的`Grid`结构的布局：

![](img/B15619_05_06.png)

图 5.3：建立在一维向量上的二维网格

`get_row()`的可能实现将返回一个包含代表行的开始和结束的迭代器的`std::pair`：

```cpp
auto Grid::get_row(std::size_t y) {
  auto left = data_.begin() + w_ * y;
  auto right = left + w_;
  return std::make_pair(left, right);
} 
```

表示行的迭代器对然后可以被标准库算法使用。在下面的示例中，我们使用`std::generate()`和`std::count()`：

```cpp
auto grid = Grid{10, 10};
auto y = 3;
auto row = grid.get_row(y);
std::generate(row.first, row.second, std::rand);
auto num_fives = std::count(row.first, row.second, 5); 
```

虽然这样可以工作，但使用`std::pair`有点笨拙，而且还要求客户端知道如何处理迭代器对。没有明确说明`first`和`second`成员实际上表示半开范围。如果它能暴露一个强类型的范围会不会很好呢？幸运的是，我们将在下一章中探讨的 Ranges 库为我们提供了一个名为`std::ranges::subrange`的视图类型。现在，`get_row()`函数可以这样实现：

```cpp
auto Grid::get_row(std::size_t y) {
  auto first = data_.begin() + w_ * y;
  auto sentinel = first + w_;
  return std::ranges::subrange{first, sentinel};
} 
```

我们甚至可以更懒，使用为这种情况量身定制的方便视图，称为`std::views::counted()`

```cpp
auto Grid::get_row(std::size_t y) {
  auto first = data_.begin() + w_ * y;
  return std::views::counted(first, w_);
} 
```

从`Grid`类返回的行现在可以与接受范围而不是迭代器对的受限算法中的任何一个一起使用：

```cpp
auto row = grid.get_row(y);
std::ranges::generate(row, std::rand);
auto num_fives = std::ranges::count(row, 5); 
```

这完成了我们编写和使用支持迭代器对和范围的通用算法的示例。希望这给您一些关于如何以通用方式编写数据结构和算法以避免组合爆炸的见解，如果我们不得不为所有类型的数据结构编写专门的算法，那么组合爆炸就会发生。

# 最佳实践

让我们考虑一些在使用我们讨论的算法时会对您有所帮助的实践。我将首先强调实际利用标准算法的重要性。

## 使用受限算法

在 C++20 中引入的`std::ranges`下的受限算法比`std`下的基于迭代器的算法提供了一些优势。受限算法执行以下操作：

+   支持投影，简化元素的自定义比较。

+   支持范围而不是迭代器对。无需将`begin()`和`end()`迭代器作为单独的参数传递。

+   易于正确使用，并且由于受 C++概念的限制，在编译期间提供描述性错误消息。

我建议开始使用受限算法而不是基于迭代器的算法。

您可能已经注意到，本书在许多地方使用了基于迭代器的算法。这样做的原因是，在撰写本书时，并非所有标准库实现都支持受限算法。

## 仅对需要检索的数据进行排序

算法库包含三种基本排序算法：`sort()`、`partial_sort()`和`nth_element()`。此外，它还包含其中的一些变体，包括`stable_sort()`，但我们将专注于这三种，因为根据我的经验，在许多情况下，可以通过使用`nth_element()`或`partial_sort()`来避免完全排序。

虽然`sort()`对整个范围进行排序，但`partial_sort()`和`nth_element()`可以被视为检查该排序范围的部分的算法。在许多情况下，您只对排序范围的某一部分感兴趣，例如：

+   如果要计算范围的中位数，则需要排序范围中间的值。

+   如果您想创建一个可以被人口平均身高的 80%使用的身体扫描仪，您需要在排序范围内找到两个值：距离最高者 10%的值和距离最矮者 10%的值。

下图说明了`std::nth_element`和`std::partial_sort`如何处理范围，与完全排序的范围相比：

|

```cpp
auto v = std::vector{6, 3, 2, 7,
                     4, 1, 5};
auto it = v.begin() + v.size()/2; 
```

| ![](img/B15619_05_07.png) |
| --- |

|

```cpp
std::ranges::sort(v); 
```

| *![](img/B15619_05_08.png)* |
| --- |

|

```cpp
std::nth_element(v.begin(), it,
                 v.end()); 
```

| *![](img/B15619_05_09.png)* |
| --- |

|

```cpp
std::partial_sort(v.begin(), it,
                  v.end()); 
```

| *![](img/B15619_05_10.png)* |
| --- |

图 5.1：使用不同算法对范围的排序和非排序元素

下表显示了它们的算法复杂度；请注意，*m*表示正在完全排序的子范围：

| 算法 | 复杂度 |
| --- | --- |
| `std::sort()` | *O(n log n)* |
| `std::partial_sort()` | *O(n log m)* |
| `std::nth_element()` | *O(n)* |

表 5.2：算法复杂度

### 用例

现在您已经了解了`std:nth_element()`和`std::partial_sort()`，让我们看看如何将它们结合起来检查范围的部分，就好像整个范围都已排序：

|

```cpp
auto v = std::vector{6, 3, 2, 7,
                     4, 1, 5};
auto it = v.begin() + v.size()/2; 
```

| ![](img/B15619_05_07.png) |
| --- |

|

```cpp
auto left = it - 1;
auto right = it + 2;
std::nth_element(v.begin(),
                 left, v.end());
std::partial_sort(left, right,
                  v.end()); 
```

| *![](img/B15619_05_12.png)* |
| --- |

|

```cpp
std::nth_element(v.begin(), it,
                 v.end());
std::sort(it, v.end()); 
```

| *![](img/B15619_05_13.png)* |
| --- |

|

```cpp
auto left = it - 1;
auto right = it + 2;
std::nth_element(v.begin(),
                 right, v.end());
std::partial_sort(v.begin(),
                  left, right);
std::sort(right, v.end()); 
```

| *![](img/B15619_05_14.png)* |
| --- |

图 5.3：组合算法和相应的部分排序结果

正如您所看到的，通过使用`std::sort()`、`std::nth_element()`和`std::partial_sort()`的组合，有许多方法可以在绝对不需要对整个范围进行排序时避免这样做。这是提高性能的有效方法。

### 性能评估

让我们看看`std::nth_element()`和`std::partial_sort()`与`std::sort()`相比如何。我们使用了一个包含 1000 万个随机`int`元素的`std::vector`进行了测量：

| 操作 | 代码，其中`r`是操作的范围 | 时间（加速） |
| --- | --- | --- |
| 排序 |

```cpp
std::sort(r.begin(), r.end()); 
```

| 760 毫秒（1.0x） |
| --- |
| 寻找中位数 |

```cpp
auto it = r.begin() + r.size() / 2;
std::nth_element(r.begin(), it, r.end()); 
```

| 83 毫秒（9.2x） |
| --- |
| 对范围的前十分之一进行排序 |

```cpp
auto it = r.begin() + r.size() / 10;
std::partial_sort(r.begin(), it, r.end()); 
```

| 378 毫秒（2.0x） |
| --- |

表 5.3：部分排序算法的基准结果

## 使用标准算法而不是原始的 for 循环

很容易忘记复杂的算法可以通过组合标准库中的算法来实现。也许是因为习惯于手工解决问题并立即开始手工制作`for`循环并使用命令式方法解决问题。如果这听起来对您来说很熟悉，我的建议是要充分了解标准算法，以至于您开始将它们作为首选。

我推荐使用标准库算法而不是原始的`for`循环，原因有很多：

+   标准算法提供了性能。即使标准库中的一些算法看起来很琐碎，它们通常以不明显的方式进行了最优设计。

+   标准算法提供了安全性。即使是更简单的算法也可能有一些特殊情况，很容易忽视。

+   标准算法是未来的保障；如果您想利用 SIMD 扩展、并行性甚至是以后的 GPU，可以用更合适的算法替换给定的算法（参见*第十四章*，*并行算法*）。

+   标准算法有详细的文档。

此外，通过使用算法而不是`for`循环，每个操作的意图都可以通过算法的名称清楚地表示出来。如果您使用标准算法作为构建块，您的代码的读者不需要检查原始的`for`循环内部的细节来确定您的代码的作用。

一旦您养成了以算法思考的习惯，您会意识到许多`for`循环通常是一些简单算法的变体，例如`std::transform()`、`std::any_of()`、`std::copy_if()`和`std::find()`。

使用算法还将使代码更清晰。您通常可以实现函数而不需要嵌套代码块，并且同时避免可变变量。这将在下面的示例中进行演示。

### 示例 1：可读性问题和可变变量

我们的第一个示例来自一个真实的代码库，尽管变量名已经被伪装。由于这只是一个剪切，您不必理解代码的逻辑。这个例子只是为了向您展示与嵌套的`for`循环相比，使用算法时复杂度降低的情况。

原始版本如下：

```cpp
// Original version using a for-loop
auto conflicting = false;
for (const auto& info : infos) {
  if (info.params() == output.params()) {
    if (varies(info.flags())) {
      conflicting = true;
      break;
    }
  }
  else {
    conflicting = true;
    break;
  }
} 
```

在`for`-循环版本中，很难理解`conflicting`何时或为什么被设置为`true`，而在算法的后续版本中，你可以直观地看到，如果`info`满足谓词，它就会发生。此外，标准算法版本不使用可变变量，并且可以使用短 lambda 和`any_of()`的组合来编写。它看起来是这样的：

```cpp
// Version using standard algorithms
const auto in_conflict = & {
  return info.params() != output.params() || varies(info.flags());
};
const auto conflicting = std::ranges::any_of(infos, in_conflict); 
```

虽然这可能有些言过其实，但想象一下，如果我们要追踪一个 bug 或者并行化它，使用 lambda 和`any_of()`的标准算法版本将更容易理解和推理。

### 示例 2：不幸的异常和性能问题

为了进一步说明使用算法而不是`for`-循环的重要性，我想展示一些不那么明显的问题，当使用手工制作的`for`-循环而不是标准算法时，你可能会遇到的问题。

假设我们需要一个函数，将容器前面的第 n 个元素移动到后面，就像这样：

![](img/B15619_05_15.png)

图 5.4：将前三个元素移动到范围的后面

#### 方法 1：使用传统的 for 循环

一个天真的方法是在迭代它们时将前 n 个元素复制到后面，然后删除前 n 个元素：

![](img/B15619_05_16.png)

图 5.5：分配和释放以将元素移动到范围的后面

以下是相应的实现：

```cpp
template <typename Container>
auto move_n_elements_to_back(Container& c, std::size_t n) {
  // Copy the first n elements to the end of the container
  for (auto it = c.begin(); it != std::next(c.begin(), n); ++it) {
    c.emplace_back(std::move(*it));
  }
  // Erase the copied elements from front of container
  c.erase(c.begin(), std::next(c.begin(), n));
} 
```

乍一看，这可能看起来是合理的，但仔细检查会发现一个严重的问题——如果容器在迭代过程中重新分配了内存，由于`emplace_back()`，迭代器`it`将不再有效。由于算法试图访问无效的迭代器，算法将进入未定义的行为，并且在最好的情况下会崩溃。

#### 方法 2：安全的 for 循环（以性能为代价的安全）

由于未定义的行为是一个明显的问题，我们将不得不重写算法。我们仍然使用手工制作的`for`-循环，但我们将利用索引而不是迭代器：

```cpp
template <typename Container>
auto move_n_elements_to_back(Container& c, std::size_t n) {
  for (size_t i = 0; i < n; ++i) {
    auto value = *std::next(c.begin(), i);
    c.emplace_back(std::move(value));
  }
  c.erase(c.begin(), std::next(c.begin(), n));
} 
```

解决方案有效；不再崩溃。但现在，它有一个微妙的性能问题。该算法在`std::list`上比在`std::vector`上慢得多。原因是`std::next(it, n)`与`std::list::iterator`一起使用是*O(n)*，而在`std::vector::iterator`上是*O(1)*。由于`std::next(it, n)`在`for`-循环的每一步中都被调用，这个算法在诸如`std::list`的容器上将具有*O(n*²*)*的时间复杂度。除了这个性能限制，前面的代码还有以下限制：

+   由于`emplace_back()`，它不适用于静态大小的容器，比如`std::array`

+   它可能会抛出异常，因为`emplace_back()`可能会分配内存并失败（尽管这可能很少见）

#### 方法 3：查找并使用合适的标准库算法

当我们达到这个阶段时，我们应该浏览标准库，看看它是否包含一个适合用作构建块的算法。方便的是，`<algorithm>`头文件提供了一个名为`std::rotate()`的算法，它正好可以解决我们正在寻找的问题，同时避免了前面提到的所有缺点。这是我们使用`std::rotate()`算法的最终版本：

```cpp
template <typename Container>
auto move_n_elements_to_back(Container& c, std::size_t n) {
  auto new_begin = std::next(c.begin(), n);
  std::rotate(c.begin(), new_begin, c.end());
} 
```

让我们来看看使用`std::rotate()`的优势：

+   该算法不会抛出异常，因为它不会分配内存（尽管包含的对象可能会抛出异常）

+   它适用于大小无法更改的容器，比如`std::array`

+   性能是*O(n)*，无论它在哪个容器上操作

+   实现很可能针对特定硬件进行优化

也许你会觉得这种`for`-循环和标准算法之间的比较是不公平的，因为这个问题还有其他解决方案，既优雅又高效。然而，在现实世界中，当标准库中有算法等待解决你的问题时，看到像你刚刚看到的这样的实现并不罕见。

### 例 3：利用标准库的优化

这个最后的例子突显了一个事实，即即使看起来非常简单的算法可能包含你不会考虑的优化。例如，让我们来看一下`std::find()`。乍一看，似乎明显的实现无法进一步优化。这是`std::find()`算法的可能实现：

```cpp
template <typename It, typename Value>
auto find_slow(It first, It last, const Value& value) {
  for (auto it = first; it != last; ++it)
    if (*it == value)
      return it;
  return last;
} 
```

然而，通过查看 GNU libstdc++的实现，当与`random_access_iterator`一起使用时（换句话说，`std::vector`，`std::string`，`std::deque`和`std::array`），libc++实现者已经将主循环展开成一次四个循环的块，导致比较（`it != last`）执行的次数减少四分之一。

这是从 libstdc++库中取出的`std::find()`的优化版本：

```cpp
template <typename It, typename Value>
auto find_fast(It first, It last, const Value& value) {
  // Main loop unrolled into chunks of four
  auto num_trips = (last - first) / 4;
  for (auto trip_count = num_trips; trip_count > 0; --trip_count) {
    if (*first == value) {return first;} ++first;
    if (*first == value) {return first;} ++first;
    if (*first == value) {return first;} ++first;
    if (*first == value) {return first;} ++first;
  }
  // Handle the remaining elements
  switch (last - first) {
    case 3: if (*first == value) {return first;} ++first;
    case 2: if (*first == value) {return first;} ++first;
    case 1: if (*first == value) {return first;} ++first;
    case 0:
    default: return last;
  }
} 
```

请注意，实际上使用的是`std::find_if()`，而不是`std::find()`，它利用了这种循环展开优化。但`std::find()`是使用`std::find_if()`实现的。

除了`std::find()`，libstdc++中还使用`std::find_if()`实现了大量算法，例如`any_of()`，`all_of()`，`none_of()`，`find_if_not()`，`search()`，`is_partitioned()`，`remove_if()`和`is_permutation()`，这意味着所有这些都比手工制作的`for`-循环稍微快一点。

稍微地，我真的是指稍微；加速大约是 1.07 倍，如下表所示：

| 在包含 1000 万个元素的`std::vector`中查找整数 |
| --- |
| 算法 | 时间 | 加速 |
| `find_slow()` | 3.06 毫秒 | 1.00x |
| `find_fast()` | 3.26 毫秒 | 1.07x |

表 5.5：find_fast()使用在 libstdc++中找到的优化。基准测试表明 find_fast()比 find_slow()稍微快一点。

然而，即使好处几乎可以忽略不计，使用标准算法，你可以免费获得它。

#### "与零比较"优化

除了循环展开之外，一个非常微妙的优化是`trip_count`是向后迭代以与零比较而不是一个值。在一些 CPU 上，与零比较比任何其他值稍微快一点，因为它使用另一个汇编指令（在 x86 平台上，它使用`test`而不是`cmp`）。

下表显示了使用 gcc 9.2 的汇编输出的差异：

| 动作 | C++ | 汇编 x86 |
| --- | --- | --- |
| 与零比较 |

```cpp
auto cmp_zero(size_t val) {
  return val > 0;
} 
```

|

```cpp
test edi, edi
setne al
ret 
```

|

| 与另一个值比较 |
| --- |

```cpp
auto cmp_val(size_t val) {
  return val > 42;
} 
```

|

```cpp
cmp edi, 42
setba al
ret 
```

|

表 5.6：汇编输出的差异

尽管标准库实现鼓励这种优化，但不要重新排列你手工制作的循环以从这种优化中受益，除非它是一个（非常）热点。这样做会严重降低你代码的可读性；让算法来处理这些优化。

这是关于使用算法而不是`for`-循环的建议的结束。如果你还没有使用标准算法，我希望我已经给了你一些理由来说服你尝试一下。现在我们将继续我的最后一个关于有效使用算法的建议。

## 避免容器拷贝

我们将通过突出一个常见问题来结束这一章，即尝试从算法库中组合多个算法时很难避免底层容器的不必要拷贝。

一个例子将澄清我的意思。假设我们有某种`Student`类来代表特定年份和特定考试分数的学生，就像这样：

```cpp
struct Student {
  int year_{};
  int score_{};
  std::string name_{};
  // ...
}; 
```

如果我们想在一个庞大的学生集合中找到二年级成绩最高的学生，我们可能会在`score_`上使用`max_element()`，但由于我们只想考虑二年级的学生，这就变得棘手了。基本上，我们想要将`copy_if()`和`max_element()`结合起来组成一个新的算法，但是在算法库中组合算法是不可能的。相反，我们需要将所有二年级学生复制到一个新的容器中，然后迭代新容器以找到最高分数：

```cpp
auto get_max_score(const std::vector<Student>& students, int year) {
  auto by_year = = { return s.year_ == year; }; 
  // The student list needs to be copied in
  // order to filter on the year
  auto v = std::vector<Student>{};
  std::ranges::copy_if(students, std::back_inserter(v), by_year);
  auto it = std::ranges::max_element(v, std::less{}, &Student::score_);
  return it != v.end() ? it->score_ : 0; 
} 
```

这是一个诱人的地方，可以开始从头开始编写自定义算法，而不利用标准算法的优势。但正如您将在下一章中看到的，没有必要放弃标准库来执行这样的任务。组合算法的能力是使用 Ranges 库的主要动机之一，我们将在下一章中介绍。

# 总结

在本章中，您学习了如何使用算法库中的基本概念，以及使用它们作为构建模块而不是手写的`for`循环的优势，以及为什么在以后优化代码时使用标准算法库是有益的。我们还讨论了标准算法的保证和权衡，这意味着您从现在开始可以放心地使用它们。

通过使用算法的优势而不是手动的`for`循环，您的代码库已经为本书接下来的章节中将讨论的并行化技术做好了准备。标准算法缺少的一个关键特性是组合算法的可能性，这一点在我们试图避免不必要的容器复制时得到了强调。在下一章中，您将学习如何使用 C++ Ranges 库中的视图来克服标准算法的这一限制。
