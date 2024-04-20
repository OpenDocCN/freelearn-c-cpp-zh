# 第六章：范围和视图

本章将继续上一章关于算法及其局限性的内容。Ranges 库中的视图是 Algorithm 库的强大补充，它允许我们将多个转换组合成一个惰性评估的视图，覆盖元素序列。阅读完本章后，您将了解什么是范围视图，以及如何将它们与标准库中的容器、迭代器和算法结合使用。

具体来说，我们将涵盖以下主要主题：

+   算法的可组合性

+   范围适配器

+   将视图实例化为容器

+   在范围内生成、转换和抽样元素

在我们深入讨论 Ranges 库本身之前，让我们讨论一下为什么它被添加到 C++20 中，以及为什么我们想要使用它。

# Ranges 库的动机

随着 Ranges 库引入到 C++20 中，我们在实现算法时从标准库中获益的方式得到了一些重大改进。以下列表显示了新功能：

+   定义迭代器和范围要求的概念现在可以由编译器更好地检查，并在开发过程中提供更多帮助

+   `<algorithm>`头文件中所有函数的新重载都受到了刚才提到的概念的约束，并接受范围作为参数，而不是迭代器对

+   迭代器头文件中的约束迭代器

+   范围视图，使得可以组合算法

本章将重点放在最后一项上：视图的概念，它允许我们组合算法以避免将数据不必要地复制到拥有的容器中。为了充分理解这一点，让我们从算法库中的可组合性不足开始。

## 算法库的局限性

标准库算法在一个基本方面存在局限性：可组合性。让我们通过查看*第五章*，*算法*中的最后一个示例来了解这一点，我们在那里简要讨论了这个问题。如果您还记得，我们有一个类来表示特定年份和特定考试分数的`Student`。

```cpp
struct Student {
  int year_{};
  int score_{};
  std::string name_{};
  // ...
}; 
```

如果我们想要从一个大量学生的集合中找到他们第二年的最高分，我们可能会在`score_`上使用`max_element()`，但由于我们只想考虑特定年份的学生，这就变得棘手了。通过使用接受范围和投影的新算法（参见*第五章*，*算法*），我们可能会得到类似这样的结果：

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

以下是一个示例，说明了它的使用方法：

```cpp
auto students = std::vector<Student>{
  {3, 120, "Niki"},
  {2, 140, "Karo"},
  {3, 190, "Sirius"},
  {2, 110, "Rani"},
   // ...
};
auto score = get_max_score(students, 2);
std::cout << score << '\n'; 
// Prints 140 
```

这个`get_max_score()`的实现很容易理解，但在使用`copy_if()`和`std::back_inserter()`时会创建不必要的`Student`对象的副本。

您现在可能会认为`get_max_score()`可以写成一个简单的`for-`循环，这样就可以避免由于`copy_if()`而产生额外的分配。

```cpp
auto get_max_score(const std::vector<Student>& students, int year) {
  auto max_score = 0;
  for (const auto& student : students) {
    if (student.year_ == year) {
      max_score = std::max(max_score, student.score_);
    }
  }
  return max_score;
} 
```

虽然在这个小例子中很容易实现，但我们希望能够通过组合小的算法构建块来实现这个算法，而不是使用单个`for`-循环从头开始实现它。

我们希望有一种语法，它与使用算法一样易读，但又能够避免在算法的每一步中构造新的容器。这就是 Ranges 库中的视图发挥作用的地方。虽然 Ranges 库包含的不仅仅是视图，但与 Algorithm 库的主要区别在于能够将本质上不同类型的迭代器组合成惰性评估的范围。

如果使用 Ranges 库中的视图编写前面的示例，它将如下所示：

```cpp
auto max_value(auto&& range) {
  const auto it = std::ranges::max_element(range);
  return it != range.end() ? *it : 0;
}
auto get_max_score(const std::vector<Student>& students, int year) {
  const auto by_year = = { return s.year_ == year; };
  return max_value(students 
    | std::views::filter(by_year)
    | std::views::transform(&Student::score_));
} 
```

现在我们又开始使用算法，因此可以避免可变变量、`for`循环和`if`语句。在我们的初始示例中，保存特定年份学生的额外向量现在已经被消除。相反，我们已经组成了一个范围视图，它代表了所有通过`by_year`谓词过滤的学生，然后转换为只暴露分数。然后将视图传递给一个小型实用程序函数`max_value()`，该函数使用`max_element()`算法来比较所选学生的分数，以找到最大值。

通过将算法链接在一起来组成算法，并同时避免不必要的复制，这就是我们开始使用 Ranges 库中的视图的动机。

# 从 Ranges 库中理解视图

Ranges 库中的视图是对范围的惰性评估迭代。从技术上讲，它们只是具有内置逻辑的迭代器，但从语法上讲，它们为许多常见操作提供了非常愉快的语法。

以下是如何使用视图来对向量中的每个数字进行平方的示例（通过迭代）：

```cpp
auto numbers = std::vector{1, 2, 3, 4};
auto square = [](auto v) {  return v * v; };
auto squared_view = std::views::transform(numbers, square);
for (auto s : squared_view) {  // The square lambda is invoked here
  std::cout << s << " ";
}
// Output: 1 4 9 16 
```

变量`squared_view`不是`numbers`向量的值平方的副本；它是一个代理对象，有一个细微的区别——每次访问一个元素时，都会调用`std::transform()`函数。这就是为什么我们说视图是惰性评估的。

从外部来看，你仍然可以像任何常规容器一样迭代`squared_view`，因此你可以执行常规算法，比如`find()`或`count()`，但在内部，你没有创建另一个容器。

如果要存储范围，可以使用`std::ranges::copy()`将视图实现为容器。（这将在本章后面进行演示。）一旦视图被复制回容器，原始容器和转换后的容器之间就不再有任何依赖关系。

使用范围，还可以创建一个过滤视图，其中只有范围的一部分是可见的。在这种情况下，只有满足条件的元素在迭代视图时是可见的：

```cpp
auto v = std::vector{4, 5, 6, 7, 6, 5, 4};
auto odd_view = 
  std::views::filter(v, [](auto i){ return (i % 2) == 1; });
for (auto odd_number : odd_view) {
  std::cout << odd_number << " ";
}
// Output: 5 7 5 
```

Ranges 库的多功能性的另一个例子是它提供了创建一个视图的可能性，该视图可以迭代多个容器，就好像它们是一个单一的列表一样：

```cpp
auto list_of_lists = std::vector<std::vector<int>> {
  {1, 2},
  {3, 4, 5},
  {5},
  {4, 3, 2, 1}
};
auto flattened_view = std::views::join(list_of_lists);
for (auto v : flattened_view) 
  std::cout << v << " ";
// Output: 1 2 3 4 5 5 4 3 2 1

auto max_value = *std::ranges::max_element(flattened_view);
// max_value is 5 
```

现在我们已经简要地看了一些使用视图的例子，让我们来检查所有视图的共同要求和属性

## 视图是可组合的

视图的全部功能来自于能够将它们组合在一起。由于它们不复制实际数据，因此可以在数据集上表达多个操作，而在内部只迭代一次。为了理解视图是如何组成的，让我们看一下我们的初始示例，但是不使用管道运算符来组合视图；相反，让我们直接构造实际的视图类。这是它的样子：

```cpp
auto get_max_score(const std::vector<Student>& s, int year) {
  auto by_year = = { return s.year_ == year; };

  auto v1 = std::ranges::ref_view{s}; // Wrap container in a view
  auto v2 = std::ranges::filter_view{v1, by_year};
  auto v3 = std::ranges::transform_view{v2, &Student::score_};
  auto it = std::ranges::max_element(v3);
  return it != v3.end() ? *it : 0;
} 
```

我们首先创建了一个`std::ranges::ref_view`，它是一个围绕容器的薄包装。在我们的情况下，它将向量`s`转换为一个便宜的视图。我们需要这个，因为我们的下一个视图`std::ranges::filter_view`需要一个视图作为它的第一个参数。正如你所看到的，我们通过引用链中的前一个视图来组成我们的下一个视图。

这种可组合视图的链当然可以任意延长。算法`max_element()`不需要知道完整链的任何信息；它只需要迭代范围`v3`，就像它是一个普通的容器一样。

以下图是`max_element()`算法、视图和输入容器之间关系的简化视图：

![](img/B15619_06_01.png)

图 6.1：顶层算法 std::ranges::max_element()从视图中提取值，这些视图惰性地处理来自底层容器（std::vector）的元素

现在，这种组合视图的方式有点冗长，如果我们试图去除中间变量`v1`和`v2`，我们最终会得到这样的东西：

```cpp
using namespace std::ranges; // _view classes live in std::ranges
auto scores = 
  transform_view{filter_view{ref_view{s}, by_year},
    &Student::score_}; 
```

现在，这可能看起来不太语法优雅。通过摆脱中间变量，我们得到了一些即使对训练有素的人来说也很难阅读的东西。我们还被迫从内到外阅读代码以理解依赖关系。幸运的是，Ranges 库为我们提供了范围适配器，这是组合视图的首选方式。

## 范围视图配有范围适配器

正如你之前看到的，Ranges 库还允许我们使用范围适配器和管道运算符来组合视图，从而获得更加优雅的语法（你将在*第十章*，*代理对象和延迟评估*中学习如何在自己的代码中使用管道运算符）。前面的代码示例可以通过使用范围适配器对象进行重写，我们会得到类似这样的东西：

```cpp
using namespace std::views; // range adaptors live in std::views
auto scores = s | filter(by_year) | transform(&Student::score_); 
```

从左到右阅读语句的能力，而不是从内到外，使得代码更容易阅读。如果你使用过 Unix shell，你可能熟悉这种用于链接命令的表示法。

Ranges 库中的每个视图都有一个相应的范围适配器对象，可以与管道运算符一起使用。在使用范围适配器时，我们还可以跳过额外的`std::ranges::ref_view`，因为范围适配器直接与`viewable_ranges`一起工作，即可以安全转换为`view`的范围。

您可以将范围适配器视为一个全局无状态对象，它实现了两个函数：`operator()()`和`operator|()`。这两个函数都构造并返回视图对象。管道运算符是在前面的示例中使用的。但也可以使用调用运算符使用嵌套语法来形成视图，如下所示：

```cpp
using namespace std::views;
auto scores = transform(filter(s, by_year), &Student::score_); 
```

同样，在使用范围适配器时，无需将输入容器包装在`ref_view`中。

总之，Ranges 库中的每个视图包括：

+   一个类模板（实际视图类型），它操作视图对象，例如`std::ranges::transform_view`。这些视图类型可以在命名空间`std::ranges`下找到。

+   一个范围适配器对象，它从范围创建视图类的实例，例如`std::views::transform`。所有范围适配器都实现了`operator()()`和`operator|()`，这使得可以使用管道运算符或嵌套来组合转换。范围适配器对象位于命名空间`std::views`下。

## 视图是具有复杂性保证的非拥有范围

在前一章中，介绍了范围的概念。任何提供`begin()`和`end()`函数的类型，其中`begin()`返回一个迭代器，`end()`返回一个哨兵，都可以作为范围。我们得出结论，所有标准容器都是范围。容器拥有它们的元素，因此我们可以称它们为拥有范围。

视图也是一个范围，它提供`begin()`和`end()`函数。然而，与容器不同，视图不拥有它们所覆盖的范围中的元素。

视图的构造必须是一个常量时间操作，*O(1)*。它不能执行任何依赖于底层容器大小的工作。对于视图的赋值、复制、移动和销毁也是如此。这使得在使用视图来组合多个算法时，很容易推断性能。它还使得视图无法拥有元素，因为这将需要在构造和销毁时具有线性时间复杂度。

## 视图不会改变底层容器

乍一看，视图可能看起来像是输入容器的变异版本。然而，容器根本没有发生变异：所有处理都是在迭代器中进行的。视图只是一个代理对象，当迭代时，*看起来*像是一个变异的容器。

```cpp
int to std::string:
```

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto strings = ints 
  | std::views::transform([](auto i) { return std::to_string(i); }); 
```

也许我们有一个在容器上操作的函数，我们想要使用范围算法进行转换，然后我们想要返回并将其存储回容器。例如，在上面的例子中，我们可能确实想要将字符串存储在一个单独的容器中。您将在下一节中学习如何做到这一点。

## 视图可以实体化为容器

有时，我们想要将视图存储在容器中，即**实体化**视图。所有视图都可以实体化为容器，但这并不像您希望的那样容易。C++20 提出了一个名为`std::ranges::to<T>()`的函数模板，它可以将视图转换为任意容器类型`T`，但并没有完全实现。希望我们在将来的 C++版本中能够得到类似的东西。在那之前，我们需要做更多的工作来实体化视图。

在前面的例子中，我们将`ints`转换为`std::strings`，如下所示：

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto r = ints 
  | std::views::transform([](auto i) { return std::to_string(i); }); 
```

现在，如果我们想要将范围`r`实体化为一个向量，我们可以像这样使用`std::ranges::copy（）`：

```cpp
auto vec = std::vector<std::string>{};
std::ranges::copy(r, std::back_inserter(vec)); 
```

实体化视图是一个常见的操作，所以如果我们有一个通用的实用程序来处理这种情况会很方便。假设我们想要将一些任意视图实体化为`std::vector`；我们可以使用一些通用编程来得到以下方便的实用函数：

```cpp
auto to_vector(auto&& r) {
  std::vector<std::ranges::range_value_t<decltype(r)>> v;
  if constexpr(std::ranges::sized_range<decltype(r)>) {
    v.reserve(std::ranges::size(r));
  }
  std::ranges::copy(r, std::back_inserter(v));
  return v;
} 
https://timur.audio/how-to-make-a-container-from-a-c20-range, which is well worth a read. 
```

在本书中，我们还没有讨论过泛型编程，但接下来的几章将解释使用`auto`参数类型和`if constexpr`。

我们正在使用`reserve（）`来优化此函数的性能。它将为范围中的所有元素预先分配足够的空间，以避免进一步的分配。但是，我们只能在知道范围的大小时调用`reserve（）`，因此我们必须使用`if constexpr`语句在编译时检查范围是否为`size_range`。

有了这个实用程序，我们可以将某种类型的容器转换为持有另一种任意类型元素的向量。让我们看看如何使用`to_vector（）`将整数列表转换为`std::strings`的向量。这是一个例子：

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto r = ints 
  | std::views::transform([](auto i) { return std::to_string(i); });
auto strings = to_vector(r); 
// strings is now a std::vector<std::string> 
```

请记住，一旦视图被复制回容器，原始容器和转换后的容器之间就不再有任何依赖关系。这也意味着实体化是一种急切的操作，而所有视图操作都是惰性的。

## 视图是惰性评估的

视图执行的所有工作都是惰性的。这与`<algorithm>`头文件中的函数相反，后者在调用时立即对所有元素执行其工作。

您已经看到`std::views::filter`视图可以替换算法`std::copy_if（）`，而`std::views::transform`视图可以替换`std::transform（）`算法。当我们将视图用作构建块并将它们链接在一起时，我们通过避免急切算法所需的容器元素的不必要复制而受益于惰性评估。

但是`std::sort（）`呢？有对应的排序视图吗？答案是否定的，因为它需要视图首先急切地收集所有元素以找到要返回的第一个元素。相反，我们必须自己显式调用视图上的排序来做到这一点。在大多数情况下，我们还需要在排序之前实体化视图。我们可以通过一个例子来澄清这一点。假设我们有一个通过某个谓词过滤的数字向量，如下所示：

```cpp
auto vec = std::vector{4, 2, 7, 1, 2, 6, 1, 5};
auto is_odd = [](auto i) { return i % 2 == 1; };
auto odd_numbers = vec | std::views::filter(is_odd); 
```

如果我们尝试使用`std::ranges::sort（）`或`std::sort（）`对我们的视图`odd_numbers`进行排序，我们将收到编译错误：

```cpp
std::ranges::sort(odd_numbers); // Doesn't compile 
```

编译器抱怨`odd_numbers`范围提供的迭代器类型。排序算法需要随机访问迭代器，但这不是我们的视图提供的迭代器类型，即使底层输入容器是`std::vector`。我们需要在排序之前实体化视图：

```cpp
auto v = to_vector(odd_numbers);
std::ranges::sort(v);
// v is now 1, 1, 5, 7 
```

但为什么这是必要的呢？答案是这是惰性评估的结果。过滤视图（以及许多其他视图）在需要延迟读取一个元素时无法保留底层范围（在本例中为`std::vector`）的迭代器类型。

那么，有没有可以排序的视图？是的，一个例子是`std::views::take`，它返回范围中的前*n*个元素。以下示例在排序之前编译和运行良好，无需在排序之前实现视图：

```cpp
auto vec = std::vector{4, 2, 7, 1, 2, 6, 1, 5};
auto first_half = vec | std::views::take(vec.size() / 2);
std::ranges::sort(first_half);
// vec is now 1, 2, 4, 7, 2, 6, 1, 5 
```

迭代器的质量已经得到保留，因此可以对`first_half`视图进行排序。最终结果是底层向量`vec`中前一半的元素已经被排序。

您现在对来自 Ranges 库的视图以及它们的工作原理有了很好的理解。在下一节中，我们将探讨如何使用标准库中包含的视图。

# 标准库中的视图

到目前为止，在本章中，我们一直在谈论来自 Ranges 库的视图。正如前面所述，这些视图类型需要在常数时间内构造，并且还具有常数时间的复制、移动和赋值运算符。然而，在 C++中，我们在 C++20 添加 Ranges 库之前就已经谈论过视图类。这些视图类是非拥有类型，就像`std::ranges::view`一样，但没有复杂性保证。

在本节中，我们将首先探索与`std::ranges::view`概念相关联的 Ranges 库中的视图，然后转到与`std::ranges::view`不相关联的`std::string_view`和`std::span`。

## 范围视图

Ranges 库中已经有许多视图，我认为我们将在未来的 C++版本中看到更多这样的视图。本节将快速概述一些可用视图，并根据其功能将它们放入不同的类别中。

### 生成视图

```cpp
-2, -1, 0, and 1:
```

```cpp
for (auto i : std::views::iota(-2, 2)) {
  std::cout << i << ' ';
}
// Prints -2 -1 0 1 
```

通过省略第二个参数，`std::views::iota`将在请求时产生无限数量的值。

### 转换视图

转换视图是转换范围的元素或范围结构的视图。一些示例包括：

+   `std::views::transform`：转换每个元素的值和/或类型

+   `std::views::reverse`：返回输入范围的反转版本

+   `std::views::split`：拆分每个元素并将每个元素拆分为子范围。结果范围是范围的范围

+   `std::views::join`：split 的相反操作；展平所有子范围

以下示例使用`split`和`join`从逗号分隔的值字符串中提取所有数字：

```cpp
auto csv = std::string{"10,11,12"};
auto digits = csv 
  | std::views::split(',')      // [ [1, 0], [1, 1], [1, 2] ]
  | std::views::join;           // [ 1, 0, 1, 1, 1, 2 ]
for (auto i : digits) {   std::cout << i; }
// Prints 101112 
```

### 采样视图

采样视图是选择范围中的元素子集的视图，例如：

+   `std::views::filter`：仅返回满足提供的谓词的元素

+   `std::views::take`：返回范围中的*n*个第一个元素

+   `std::views::drop`：在丢弃前*n*个元素后返回范围中的所有剩余元素

在本章中，您已经看到了许多使用`std::views::filter`的示例；这是一个非常有用的视图。`std::views::take`和`std::views::drop`都有一个`_while`版本，它接受一个谓词而不是一个数字。以下是使用`take`和`drop_while`的示例：

```cpp
auto vec = std::vector{1, 2, 3, 4, 5, 4, 3, 2, 1};
 auto v = vec
   | std::views::drop_while([](auto i) { return i < 5; })
   | std::views::take(3);
 for (auto i : v) { std::cout << i << " "; }
 // Prints 5 4 3 
```

此示例使用`drop_while`从前面丢弃小于 5 的值。剩下的元素传递给`take`，它返回前三个元素。现在到我们最后一类范围视图。

### 实用视图

在本章中，您已经看到了一些实用视图的用法。当您有想要转换或视为视图的东西时，它们非常方便。在这些视图类别中的一些示例是`ref_view`、`all_view`、`subrange`、`counted`和`istream_view`。

以下示例向您展示了如何读取一个包含浮点数的文本文件，然后打印它们。

假设我们有一个名为`numbers.txt`的文本文件，其中包含重要的浮点数，如下所示：

```cpp
1.4142 1.618 2.71828 3.14159 6.283 ... 
```

然后，我们可以通过使用`std::ranges::istream_view`来创建一个`floats`的视图：

```cpp
auto ifs = std::ifstream("numbers.txt");
for (auto f : std::ranges::istream_view<float>(ifs)) {
  std::cout << f << '\n';
}
ifs.close(); 
```

通过创建一个`std::ranges::istream_view`并将其传递给一个`istream`对象，我们可以简洁地处理来自文件或任何其他输入流的数据。

Ranges 库中的视图已经经过精心选择和设计。在未来的标准版本中很可能会有更多的视图。了解不同类别的视图有助于我们将它们区分开，并在需要时更容易找到它们。

## 重新审视 std::string_view 和 std::span

值得注意的是，标准库在 Ranges 库之外还提供了其他视图。在*第四章*，*数据结构*中引入的`std::string_view`和`std::span`都是非拥有范围，非常适合与 Ranges 视图结合使用。

与 Ranges 库中的视图不同，不能保证这些视图可以在常数时间内构造。例如，从以 null 结尾的 C 风格字符串构造`std::string_view`可能会调用`strlen()`，这是一个*O(n)*操作。

假设出于某种原因，我们有一个重置范围中前`n`个值的函数：

```cpp
auto reset(std::span<int> values, int n) {
  for (auto& i : std::ranges::take_view{values, n}) {
    i = int{};
  }
} 
```

在这种情况下，不需要使用范围适配器来处理`values`，因为`values`已经是一个视图。通过使用`std::span`，我们可以传递内置数组或容器，如`std::vector`：

```cpp
int a[]{33, 44, 55, 66, 77};
reset(a, 3); 
// a is now [0, 0, 0, 66, 77]
auto v = std::vector{33, 44, 55, 66, 77};
reset(v, 2); 
// v is now [0, 0, 55, 66, 77] 
```

类似地，我们可以将`std::string_view`与 Ranges 库一起使用。以下函数将`std::string_view`的内容拆分为`std::vector`的`std::string`元素：

```cpp
auto split(std::string_view s, char delim) {
  const auto to_string = [](auto&& r) -> std::string {
    const auto cv = std::ranges::common_view{r};
    return {cv.begin(), cv.end()};
  };
  return to_vector(std::ranges::split_view{s, delim} 
    | std::views::transform(to_string));
} 
```

lambda `to_string`将一系列`char`转换为`std::string`。`std::string`构造函数需要相同的迭代器和 sentinel 类型，因此范围被包装在`std::ranges::common_view`中。实用程序`to_vector()`将视图实现并返回`std::vector<std::string>`。`to_vector()`在本章前面已经定义过。

我们的`split()`函数现在可以用于`const char*`字符串和`std::string`对象，如下所示：

```cpp
 const char* c_str = "ABC,DEF,GHI";  // C style string
  const auto v1 = split(c_str, ',');  // std::vector<std::string>
  const auto s = std::string{"ABC,DEF,GHI"};
  const auto v2 = split(s, ',');      // std::vector<std::string>
  assert(v1 == v2);                   // true 
```

我们现在将通过谈论我们期望在未来版本的 C++中看到的 Ranges 库来结束这一章。

# Ranges 库的未来

在 C++20 中被接受的 Ranges 库是基于 Eric Niebler 编写的库，可以在[`github.com/ericniebler/range-v3`](https://github.com/ericniebler/range-v3)上找到。目前，这个库中只有一小部分组件已经成为标准的一部分，但更多的东西可能很快就会被添加进来。

除了许多有用的视图尚未被接受，例如`group_by`、`zip`、`slice`和`unique`之外，还有**actions**的概念，可以像视图一样进行管道传递。但是，与视图一样，操作执行范围的急切变异，而不是像视图那样进行惰性求值。排序是典型操作的一个例子。

如果您等不及这些功能被添加到标准库中，我建议您看一下 range-v3 库。

# 总结

这一章介绍了使用范围视图构建算法背后的许多动机。通过使用视图，我们可以高效地组合算法，并使用管道操作符简洁的语法。您还学会了一个类成为视图意味着什么，以及如何使用将范围转换为视图的范围适配器。

视图不拥有其元素。构造范围视图需要是一个常数时间操作，所有视图都是惰性求值的。您已经看到了如何将容器转换为视图的示例，以及如何将视图实现为拥有容器。

最后，我们简要概述了标准库中提供的视图，以及 C++中范围的可能未来。

这一章是关于容器、迭代器、算法和范围的系列的最后一章。我们现在将转向 C++中的内存管理。
