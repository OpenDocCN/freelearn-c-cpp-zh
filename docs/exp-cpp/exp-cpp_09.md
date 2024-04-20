# 函数式编程

**面向对象编程**（**OOP**）为我们提供了一种思考对象的方式，从而以类和它们的关系来表达现实世界。函数式编程是一种完全不同的编程范式，因为它允许我们专注于*功能*结构而不是代码的*物理*结构。学习和使用函数式编程有两种用途。首先，它是一种迫使你以非常不同的方式思考的新范式。解决问题需要灵活的思维。附着于单一范式的人往往对任何问题提供类似的解决方案，而大多数优雅的解决方案需要更广泛的方法。掌握函数式编程为开发人员提供了一种新的技能，帮助他们提供更好的解决方案。其次，使用函数式编程可以减少软件中的错误数量。其中最大的原因之一是函数式编程的独特方法：它将程序分解为函数，每个函数都不修改数据的状态。

在本章中，我们将讨论函数式编程的基本模块，以及范围。在 C++20 中引入的范围为我们提供了一种很好的方式，以便将算法组合起来，使它们能够处理数据集合。将算法组合起来，以便我们可以将它们顺序应用于这些数据集合，这是函数式编程的核心。这就是为什么我们在本章中还将讨论范围。

本章将涵盖以下主题：

+   函数式编程介绍

+   介绍范围库

+   纯函数

+   高阶函数

+   深入递归

+   函数式 C++中的元编程

# 技术要求

在本章的示例中，将使用 g++编译器以及`-std=c++2a`选项。

您可以在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)找到本章的源文件。

# 揭示函数式编程

正如我们之前提到的，函数式编程是一种编程范式。您可以将范式视为构建程序时的一种思维方式。C++是一种多范式语言。我们可以使用它以过程范式开发程序，即通过依次执行语句来执行。在第三章《面向对象编程的细节》中，我们讨论了面向对象的方法，它涉及将复杂系统分解为相互通信的对象。另一方面，函数式编程鼓励我们将系统分解为函数而不是对象。它使用表达式而不是语句。基本上，您将某些东西作为输入，并将其传递给生成输出的函数。然后可以将其用作另一个函数的输入。这乍看起来可能很简单，但函数式编程包含了一些一开始感觉难以掌握的规则和实践。然而，当您掌握了这一点，您的大脑将解锁一种新的思维方式——函数式方式。

为了使这一点更清晰，让我们从一个示例开始，它将演示函数式编程的本质。假设我们已经获得了一个整数列表，并且需要计算其中偶数的数量。唯一的问题是有几个这样的向量。我们应该分别计算所有向量中的偶数，并将结果作为一个新向量产生，其中包含对每个输入向量的计算结果。

输入以矩阵形式提供，即向量的向量。在 C++中表达这一点的最简单方式是使用以下类型：

```cpp
std::vector<std::vector<int>>
```

我们可以通过使用类型别名来进一步简化前面的代码，如下所示：

```cpp
using IntMatrix = std::vector<std::vector<int>>;
```

以下是这个问题的一个例子。我们有一堆包含整数的向量，结果应该是一个包含偶数的计数的向量：

![](img/6afeff6f-80a2-4fdc-a80e-d758cdcac856.png)

看一下以下函数。它以整数向量的向量（也称为矩阵）作为其参数。该函数计算偶数的数量：

```cpp
std::vector<int> count_all_evens(const IntMatrix& numbers)
{
  std::vector<int> even_numbers_count;
  for (const auto& number_line: numbers) {
    int even{0};
 for (const auto& number: number_line) {
 if (number % 2 == 0) {
 ++even;
 }
 }
 even_numbers_count.push_back(even);
  }
  return even_numbers_count;
}
```

前面的函数保留了一个单独的向量，用于存储每个向量中偶数的计数。输入以向量的形式提供，这就是为什么函数循环遍历第一个向量以检索内部向量。对于每个检索到的向量，它循环遍历并在向量中每次遇到偶数时递增计数器。在完成每个向量的循环后，最终结果被推送到包含数字列表的向量中。虽然您可能希望回到前面的示例并改进代码，但我们现在将继续并将其分解为更小的函数。首先，我们将负责计算偶数数量的代码部分移入一个单独的函数中。

让我们将其命名为`count_evens`，如下所示：

```cpp
int count_evens(const std::vector<int>& number_line) {
  return std::count_if(number_line.begin(), 
       number_line.end(), [](int num){return num % 2 == 0;});
}
```

注意我们如何应用`count_if()`算法。它接受两个迭代器，并将它们分别放在容器的开头和结尾。它还接受第三个参数，一个*一元谓词*，它对集合的每个元素进行调用。我们传递了一个 lambda 作为一元谓词。您也可以使用任何其他可调用实体，例如函数指针、`std::`函数等。

现在我们有了一个单独的计数函数，我们可以在原始的`count_all_evens()`函数中调用它。以下是 C++中函数式编程的实现：

```cpp
std::vector<int> count_all_evens(const std::vector<std::vector<int>>& numbers) {
  return numbers | std::ranges::views::transform(count_evens);
}
```

在深入研究前面的代码之前，让我们先就引起我们注意的第一件事达成一致——不是`|`运算符的奇怪用法，而是代码的简洁性。将其与我们在本节开头介绍的代码版本进行比较。它们都完成了同样的工作，但第二个——函数式的代码——更加简洁。还要注意的是，该函数不保留或更改任何状态。它没有副作用。这在函数式编程中至关重要，因为函数必须是*纯*函数。它接受一个参数，然后在不修改它的情况下对其进行处理，并返回一个新值（通常基于输入）。函数式编程的第一个挑战是将任务分解为更小的独立函数，然后轻松地将它们组合在一起。

尽管我们是从命令式的解决方案转向函数式的解决方案，但这并不是在利用函数式编程范式时的正确方式。与其首先编写命令式代码，然后修改它以获得函数式版本，不如改变您的思维方式和解决问题的方式。您应该驯服思考函数式的过程。计算所有偶数的问题导致我们解决了一个向量的问题。如果我们能找到一种方法来解决单个向量的问题，我们就能解决所有向量的问题。`count_evens()`函数接受一个向量并产生一个单个值，如下截图所示：

![](img/f23fba4f-7441-4785-94ef-d67480148d5e.png)

解决了一个向量的问题后，我们应该继续将解决方案应用于所有向量的原始问题。`std::transform()`函数基本上做了我们需要的事情：它接受一个可以应用于单个值的函数，并将其转换为处理集合的方式。以下图片说明了我们如何使用它来实现一个函数(`count_all_evens`)，该函数可以处理来自只处理一个项目的函数(`count_evens`)的函数的项目集合：

![](img/9fa67593-d788-4281-a47a-f8e9a968a285.png)

将更大的问题分解为更小的、独立的任务是函数式编程的核心。每个函数都专门用于执行一个足够简单的任务，而不会意识到原始问题。然后将函数组合在一起，以从原始输入生成一系列转换后的项目。

现在，`count_all_evens()`函数的最终版本利用了范围。让我们找出它们是什么以及如何使用它们，因为我们将在后续示例中需要它们。

# 使用范围

范围与视图相关联。我们将在本节中同时研究它们。我们在第六章中讨论了 STL 容器和算法，*深入研究 STL 中的数据结构和算法*。它们为我们提供了一种通用的方法来组合和处理对象集合。正如您已经知道的那样，我们经常使用迭代器来循环遍历容器并处理它们的元素。迭代器是一种工具，允许我们在算法和容器之间实现松耦合。

例如，之前，我们对向量应用了`count_if()`，但`count_if()`不知道它被应用到了什么容器。看一下`count_if()`的以下声明：

```cpp
template <typename InputIterator, typename UnaryPredicate>
constexpr typename iterator_traits<InputIterator>::difference_type
  count_if(InputIterator first, InputIterator last, UnaryPredicate p);
```

正如您所看到的，除了其特定于 C++的冗长声明之外，`count_if()`不接受容器作为参数。相反，它使用迭代器 - 具体来说，输入迭代器。

输入迭代器支持使用`++`运算符向前迭代，并使用`*`运算符访问每个元素。我们还可以使用`==`和`!=`关系比较输入迭代器。

算法在不知道容器的确切类型的情况下迭代容器。我们可以在任何具有开始和结束的实体上使用`count_if()`，如下所示：

```cpp
#include <array>
#include <iostream>
#include <algorithm>

int main()
{
  std::array<int, 4> arr{1, 2, 3, 4};
 auto res = std::count_if(arr.cbegin(), arr.cend(), 
 [](int x){ return x == 3; });
  std::cout << "There are " << res << " number of elements equal to 3";
}
```

除了它们的通用性，算法不太容易组合。通常，我们将算法应用于一个集合，并将算法的结果存储为另一个集合，以便在以后的某个日期应用更多的算法。我们使用`std::transform()`将结果放入另一个容器中。例如，以下代码定义了一个产品的向量：

```cpp
// consider the Product is already declared and has a "name", "price", and "weight"
// also consider the get_products() is defined 
// and returns a vector of Product instances

using ProductList = std::vector<std::shared_ptr<Product>>;
ProductList vec{get_products()};
```

假设项目是由不同的程序员团队开发的，并且他们选择将产品的名称保留为任何数字；例如，1 代表苹果，2 代表桃子，依此类推。这意味着`vec`将包含`Product`实例，每个实例的`name`字段中将有一个数字字符（而名称的类型是`std::string` - 这就是为什么我们将数字保留为字符而不是其整数值）。现在，我们的任务是将产品的名称从数字转换为完整的字符串（`apple`，`peach`等）。我们可以使用`std::transform`来实现：

```cpp
ProductList full_named_products; // type alias has been defined above
using ProductPtr = std::shared_ptr<Product>;
std::transform(vec.cbegin(), vec.cend(), 
  std::back_inserter(full_named_products), 
  [](ProductPtr p){ /* modify the name and return */ });
```

执行上述代码后，`full_named_products`向量将包含具有完整产品名称的产品。现在，要过滤出所有的苹果并将它们复制到一个苹果向量中，我们需要使用`std::copy_if`：

```cpp
ProductList apples;
std::copy_if(full_named_products.cbegin(), full_named_products.cend(),
  std::back_inserter(apples), 
  [](ProductPtr p){ return p->name() == "apple"; });
```

前面代码示例的最大缺点之一是缺乏良好的组合，直到引入范围。范围为我们提供了一种优雅的方式来处理容器元素和组合算法。

简而言之，范围是一个可遍历的实体；也就是说，一个范围有一个`begin()`和一个`end()`，就像我们迄今为止使用的容器一样。在这些术语中，每个 STL 容器都可以被视为一个范围。STL 算法被重新定义为直接接受范围作为参数。通过这样做，它们允许我们将一个算法的结果直接传递给另一个算法，而不是将中间结果存储在本地变量中。例如，`std::transform`，我们之前使用`begin()`和`end()`，如果应用于一个范围，将具有以下形式（以下代码是伪代码）。通过使用范围，我们可以以以下方式重写前面的示例：

```cpp
ProductList apples = filter(
  transform(vec, [](ProductPtr p){/* normalize the name */}),
  [](ProductPtr p){return p->name() == "apple";}
);
```

不要忘记导入`<ranges>`头文件。transform 函数将返回一个包含已标准化名称的`Product`指针的范围；也就是说，数值将被替换为字符串值。filter 函数将接受结果并返回具有`apple`作为名称的产品范围。

请注意，我们通过省略 `std::ranges::views` 在 `filter` 和 `transform` 函数前面的部分来简化了这些代码示例。分别使用 `std::ranges::views::filter` 和 `std::ranges::views::transform`。

最后，我们在本章开头的示例中使用的重载运算符 `**|**` 允许我们将范围串联在一起。这样，我们可以组合算法以产生最终结果，如下所示：

```cpp
ProductList apples = vec | transform([](ProductPtr p){/* normalize the name */})
                         | filter([](ProductPtr p){return p->name() == "apple";});
```

我们使用管道而不是嵌套函数调用。这可能一开始会让人困惑，因为我们习惯将 `|` 运算符用作按位或。每当你看到它应用于集合时，它指的是管道范围。

`|` 运算符受 Unix shell 管道运算符的启发。在 Unix 中，我们可以将多个进程的结果串联在一起；例如，`ls -l | grep cpp | less` 将在 `ls` 命令的结果中查找 `cpp`，并使用 `less` 程序逐屏显示最终结果。

正如我们已经提到的，范围是对集合的抽象。这并不意味着它是一个集合。这就是为什么前面的示例没有带来任何额外开销 - 它只是从一个函数传递到另一个函数的范围，其中范围只提供了集合的开始和结束。它还允许我们访问底层集合元素。以下图解释了这个想法：

![](img/fa625af2-3795-4fd4-b089-5a72113aa071.png)

函数（无论是 **transform** 还是 **filter**）返回的是一个范围结构而不是一个集合。范围的 `begin()` 迭代器将指向满足谓词的源集合中的元素。范围的迭代器是一个代理对象：它与常规迭代器不同，因为它指向满足给定谓词的元素。我们有时将它们称为 **智能迭代器**，因为每次我们推进它（例如通过增量），它都会找到满足谓词的集合中的下一个元素。更有趣的是，迭代器的“智能性”取决于我们应用于集合的函数类型。例如，`filter()` 函数返回一个具有智能迭代器的范围，用于它们的增量运算符。这主要是因为过滤的结果可能包含比原始集合更少的元素。另一方面，transform 不会返回具有减少元素数量的结果 - 它只是转换元素。这意味着由 transform 返回的范围对于增量/减量操作具有相同的功能，但元素访问将不同。对于每次访问，范围的智能迭代器将从原始集合中返回转换的元素。换句话说，它只是为迭代器实现了 `*()` 运算符，类似于下面的代码片段中所示：

```cpp
auto operator*()
{
  return predicate(*current_position);
}
```

通过这种方式，我们创建了集合的新 *视图* 而不是转换元素的新集合。`filter` 和其他函数也是如此。更有趣的是，范围视图利用了 *惰性评估*。对于我们之前的示例，即使我们有两个范围转换，结果也是通过在单次遍历中评估它们来产生的。

在使用 `transform` 和 `filter` 的示例中，每个函数都定义了一个视图，但它们不会修改或评估任何内容。当我们将结果分配给结果集合时，向量是从视图中访问每个元素来构造的。这就是评估发生的地方。

就是这么简单 - 范围为我们提供了惰性评估的函数组合。我们之前简要介绍了函数式编程中使用的工具集。现在，让我们了解一下这种范式的好处。

# 为什么使用函数式编程？

首先，函数式编程引入了简洁性。与命令式对应物相比，代码要短得多。它提供了简单但高度表达的工具。当代码更少时，错误就会更少出现。

函数不会改变任何东西，这使得并行化变得更加容易。这是并发程序中的主要问题之一，因为并发任务需要在它们之间共享可变数据。大多数情况下，您必须使用诸如互斥锁之类的原语来显式同步线程。函数式编程使我们摆脱了显式同步，我们可以在多个线程上运行代码而无需进行调整。在第八章，*深入数据结构*中，我们将详细讨论数据竞争。

函数式范式将所有函数视为*纯*函数；也就是说，不会改变程序状态的函数。它们只是接受输入，以用户定义的方式进行转换，并提供输出。对于相同的输入，纯函数生成相同的结果，不受调用次数的影响。每当我们谈论函数式编程时，我们应该默认考虑所有纯函数。

以下函数以`double`作为输入，并返回其平方：

```cpp
double square(double num) { return num * num; }
```

仅编写纯函数可能会让程序运行变慢。

一些编译器，如 GCC，提供了帮助编译器优化代码的属性。例如，`[[gnu::pure]]`属性告诉编译器该函数可以被视为纯函数。这将让编译器放心，函数不会访问任何全局变量，函数的结果仅取决于其输入。

有许多情况下，*常规*函数可能会带来更快的解决方案。然而，为了适应这种范式，您应该强迫自己以函数式思维。例如，以下程序声明了一个向量，并计算了其元素的平方根：

```cpp
void calc_square_roots(std::vector<double>& vec) 
{
  for (auto& elem : vec) {
    elem = std::sqrt(elem);
  }
}

int main()
{
  std::vector<double> vec{1.1, 2.2, 4.3, 5.6, 2.4};
 calc_square_roots(vec);
}
```

在这里，我们通过引用传递向量。这意味着，如果我们在函数中对它进行更改，就会改变原始集合。显然，这不是一个纯函数，因为它改变了输入向量。函数式的替代方法是在一个新的向量中返回转换后的元素，保持输入不变：

```cpp
std::vector<double> pure_calc_square_roots(const std::vector<double>& vec)
{
 std::vector<double> new_vector;
  for (const auto& elem : vec) {
    new_vector.push_back(std::sqrt(elem));
  }
 return new_vector;
}
```

函数式思维的一个更好的例子是解决一个较小的问题，并将其应用到集合中。在这种情况下，较小的问题是计算单个数字的平方根，这已经实现为`std::sqrt`。将其应用到集合中是通过`std::ranges::views::transform`完成的，如下所示：

```cpp
#include <ranges>
#include <vector>

int main()
{
 std::vector<double> vec{1.1, 2.2, 4.3, 5.6, 2.4};
 auto result = vec | std::ranges::views::transform(std::sqrt);
}
```

正如我们已经知道的，通过使用范围，我们可以避免存储中间对象。在前面的例子中，我们直接将`transform`应用于向量。`transform`返回一个视图，而不是由源向量的转换元素组成的完整集合。当我们构造`result`向量时，实际的转换副本才会产生。另外，请注意`std::sqrt`被认为是一个纯函数。

本章开始时我们解决的例子为我们提供了函数式编程所需的视角。为了更好地掌握这种范式，我们应该熟悉它的原则。在下一节中，我们将深入探讨函数式编程的原则，以便您更好地了解何时以及如何使用这种范式。

# 函数式编程原则

尽管函数式范式很古老（诞生于 20 世纪 50 年代），但它并没有在编程世界中掀起风暴。如我们在本书和其他许多书中多次声明的那样，C++是一种**多范式语言**。这就是学习 C++的美妙之处；我们可以调整它以适应几乎每个环境。掌握这种范式并不是一件容易的事。您必须感受它并应用它，直到最终开始以这种范式思考。之后，您将能够在几秒钟内找到常规任务的解决方案。

如果您还记得第一次学习面向对象编程时，您可能会记得在能够发挥面向对象编程的真正潜力之前，您可能会有些挣扎。函数式编程也是如此。在本节中，我们将讨论函数式编程的基本概念，这将成为进一步发展的基础。您可以应用（或已经这样做）其中一些概念，而实际上并没有使用函数式范例。然而，请努力理解和应用以下每个原则。

# 纯函数

正如我们之前提到的，*如果函数不改变状态，则函数是纯的*。与非纯函数相比，纯函数可能被视为性能较差；然而，它们非常好，因为它们避免了由于状态修改而导致的代码中可能出现的大多数错误。这些错误与程序状态有关。显然，程序处理数据，因此它们组成修改状态的功能，从而为最终用户产生一些预期的结果。

在面向对象编程中，我们将程序分解为对象，每个对象都有一系列特殊功能。面向对象编程中对象的一个基本特征是其*状态*。通过向对象发送消息（换句话说，调用其方法）来修改对象的状态在面向对象编程中至关重要。通常，成员函数调用会导致对象状态的修改。在函数式编程中，我们将代码组织成一组纯函数，每个函数都有自己的目的，并且独立于其他函数。

让我们来看一个简单的例子，只是为了让这个概念清晰起来。假设我们在程序中处理用户对象，每个用户对象都包含与用户相关的年龄。`User`类型在以下代码块中被描述为`struct`：

```cpp
struct User
{
  int age;
  string name;
  string phone_number;
  string email;
};
```

有必要每年更新用户的年龄。假设我们有一个函数，每年为每个`User`对象调用一次。以下函数接受一个`User`对象作为输入，并将其`age`增加`1`：

```cpp
void update_age(User& u)
{
  u.age = u.age + 1;
}
```

`update_age()`函数通过引用接受输入并更新原始对象。这在函数式编程中并不适用。这个纯函数不是通过引用获取原始对象并改变其值，而是返回一个完全不同的`user`对象，具有相同的属性，除了更新的`age`属性：

```cpp
User pure_update_age(const User& u) // cannot modify the input argument
{
 User tmp{u};
  tmp.age = tmp.age + 1;
  return tmp;
}
```

尽管与`update_age()`相比似乎效率低下，但这种方法的优点之一是它使操作变得非常清晰（在调试代码时非常有用）。现在，可以保证`pure_update_age()`不会修改原始对象。我们可以修改前面的代码，使其按值传递对象。这样，我们将跳过创建`tmp`对象，因为参数本身就代表了一个副本：

```cpp
User pure_update_age(User u) // u is the copy of the passed object
{
  u.age = u.age + 1;
  return u;
}
```

如果一个纯函数使用相同的参数多次调用，它必须每次返回相同的结果。以下代码演示了我们的`pure_update_age()`函数在给定相同输入时返回相同的值：

```cpp
User john{.age{21}, .name{"John"}};

auto updated{pure_update_age(john)};
std::cout << updated.age; // prints 22

updated = pure_update_age(john);
std::cout << updated.age; // prints 22
```

对于一个函数来说，每次针对相同的输入数据调用时都表现相同是一个很大的好处。这意味着我们可以通过将应用程序的逻辑分解为更小的函数来设计它，每个函数都有一个确切而清晰的目的。然而，纯函数在额外临时对象方面存在开销。常规设计涉及具有包含程序状态的集中存储，该状态通过纯函数间接更新。在每次纯函数调用之后，函数将修改后的对象作为可能需要存储的新对象返回。您可以将其视为调整代码以省略传递整个对象。

# 高阶函数

在函数式编程中，函数被视为*一等*对象（你可能也会遇到一等公民）。这意味着我们应该将它们视为对象，而不是一组指令。这对我们有什么区别？嗯，在这一点上，函数被视为对象的唯一重要之处是能够将其传递给其他函数。接受其他函数作为参数的函数被称为**高阶函数**。

C++程序员将一个函数传递到另一个函数是很常见的。以下是以老式方式实现的方法：

```cpp
typedef  void (*PF)(int);
void foo(int arg) 
{
  // do something with arg
}

int bar(int arg, PF f)
{
 f(arg);
  return arg;
}

bar(42, foo);
```

在前面的代码中，我们声明了一个指向函数的指针。`PF`代表函数的类型定义，接受一个整数参数，并且不返回任何值。前面的例子是将指针函数传递给其他函数作为参数的一种常见方式。我们将函数视为对象。然而，这取决于我们对*对象*的理解。

在前面的章节中，我们将对象定义为具有状态的东西。这意味着，如果我们将函数视为对象，我们也应该能够在需要时以某种方式改变它的状态。对于函数指针来说，情况并非如此。以下是将函数传递给另一个函数的更好方法：

```cpp
class Function
{
public:
  void modify_state(int a) {
    state_ = a;
  }

  int get_state() {
    return state_;
  }

  void operator()() {
 // do something that a function would do
 }
private:
  int state_;
};

void foo(Function f)
{
 f();
  // some other useful code
}
```

看一下前面的代码。它声明了一个具有重载`operator()`的类。每当我们重载一个类的运算符时，我们使它变得*可调用*。尽管听起来很明显，但任何可调用的东西都被视为函数。因此，具有重载`operator()`的类的对象可以被视为函数（有时被称为*函数对象*）。这在某种程度上有点像一个技巧，因为我们不是将函数变成对象，而是使对象可调用。然而，这使我们能够实现我们想要的东西：具有状态的函数。以下客户端代码演示了`Function`对象具有状态：

```cpp
void foo(Function f)
{
  f();
  f.modify_state(11);
 cout << f.get_state(); // get the state
  f(); // call the "function"
}
```

通过这样做，我们可以跟踪函数被调用的次数。以下是一个跟踪调用次数的简单示例：

```cpp
class Function
{
public:
 void operator()() {    // some useful stuff ++called_; 
  }

private:
  int called_ = 0;
};
```

最后，`std::function`，它在以下代码中的`<functional>`头文件中定义，展示了另一种定义高阶函数的方法：

```cpp
#include <functional>

void print_it(int a) {
  cout << a;
}

std::function<void(int)> function_object = print_it;
```

当调用`function_object`（使用`operator()`）时，它将调用`print_it`函数。`std::function`封装了任何函数，并允许将其作为对象使用（以及将其传递给其他函数）。

在前面的例子中，接受其他函数作为参数的函数都是高阶函数的例子。返回函数的函数也被称为高阶函数。总之，高阶函数是接受或返回另一个函数或多个函数的函数。看一下以下例子：

```cpp
#include <functional>
#include <iostream>

std::function<int (int, int)> get_multiplier()
{
 return [](int a, int b) { return a * b; };
}

int main()
{
 auto multiply = get_multiplier();
  std::cout << multiply(3, 5) << std::endl; // outputs 15
}
```

`get_multiplier()`返回一个包装在`std::function`中的 lambda。然后，我们调用结果，就像调用普通函数一样。`get_multiplier()`函数是一个高阶函数。我们可以使用高阶函数来实现**柯里化**，类似于我们在前面的例子中所做的。在函数式编程中，柯里化是指我们将一个函数的多个参数转换为多个函数，每个函数只接受一个参数；例如，将`multiply(3, 5)`转换为`multiply(3)(5)`。以下是我们如何实现这一点：

```cpp
std::function<int(int)> multiply(int a)
{
 return a { return a * b; };
}

int main()
{
  std::cout << multiply(3)(5) << std::endl;
}
```

`multiply()`接受一个参数，并返回一个也接受单个参数的函数。注意 lambda 捕获：它捕获了`a`的值，以便在其主体中将其乘以`b`。

柯里化是对逻辑学家 Haskell Curry 的致敬。Haskell、Brook 和 Curry 编程语言也以他的名字命名。

柯里化最有用的特性之一是拥有我们可以组合在一起的抽象函数。我们可以创建`multiply()`的专门版本，并将它们传递给其他函数，或者在适用的地方使用它们。这可以在以下代码中看到：

```cpp
auto multiplyBy22 = multiply(22);
auto fiveTimes = multiply(5);

std::cout << multiplyBy22(10); // outputs 220
std::cout << fiveTimes(4); // outputs 20
```

在使用 STL 时，您一定会使用高阶函数。许多 STL 算法使用谓词来过滤或处理对象集合。例如，`std::find_if`函数找到满足传递的谓词对象的元素，如下例所示：

```cpp
std::vector<int> elems{1, 2, 3, 4, 5, 6};
std::find_if(elems.begin(), elems.end(), [](int el) {return el % 3 == 0;});
```

`std::find_if`以 lambda 作为其谓词，并对向量中的所有元素调用它。满足条件的任何元素都将作为请求的元素返回。

另一个高阶函数的例子是`std::transform`，我们在本章开头介绍过（不要与`ranges::view::transform`混淆）。让我们使用它将字符串转换为大写字母：

```cpp
std::string str = "lowercase";
std::transform(str.begin(), str.end(), str.begin(), 
  [](unsigned char c) { return std::toupper(c); });
std::cout << str; // "LOWERCASE"
```

第三个参数是容器的开始，是`std::transform`函数插入其当前结果的位置。

# 折叠

折叠（或减少）是将一组值组合在一起以生成减少数量的结果的过程。大多数情况下，我们说的是单个结果。折叠抽象了迭代具有递归性质的结构的过程。例如，链表或向量在元素访问方面具有递归性质。虽然向量的递归性质是有争议的，但我们将考虑它是递归的，因为它允许我们通过重复增加索引来访问其元素。为了处理这样的结构，我们通常在每一步中跟踪结果，并处理稍后要与先前结果组合的下一个项目。根据我们处理集合元素的方向，折叠称为*左*或*右*折叠。

例如，`std::accumulate`函数（另一个高阶函数的例子）是折叠功能的完美例子，因为它结合了集合中的值。看一个简单的例子：

```cpp
std::vector<double> elems{1.1, 2.2, 3.3, 4.4, 5.5};
auto sum = std::accumulate(elems.begin(), elems.end(), 0);
```

函数的最后一个参数是累加器。这是应该用作集合的第一个元素的先前值的初始值。前面的代码计算了向量元素的和。这是`std::accumulate`函数的默认行为。正如我们之前提到的，它是一个高阶函数，这意味着可以将一个函数作为其参数传递。然后将为每个元素调用该函数以产生所需的结果。例如，让我们找到先前声明的`elems`向量的乘积：

```cpp
auto product = std::accumulate(elems.begin(), elems.end(), 1, 
  [](int prev, int cur) { return prev * cur; });
```

它采用二进制操作；也就是说，具有两个参数的函数。操作的第一个参数是到目前为止已经计算的先前值，而第二个参数是当前值。二进制操作的结果将是下一步的先前值。可以使用 STL 中的现有操作之一简洁地重写前面的代码：

```cpp
auto product = std::accumulate(elems.begin(), elems.end(), 1, 
 std::multiplies<int>());
```

`std::accumulate`函数的更好替代品是`std::reduce`函数。`reduce()`类似于`accumulate()`，只是它不保留操作的顺序；也就是说，它不一定按顺序处理集合元素。您可以向`std::reduce`函数传递执行策略并更改其行为，例如并行处理元素。以下是如何使用并行执行策略将 reduce 函数应用于先前示例中的`elems`向量：

```cpp
std::reduce(std::execution::par, elems.begin(), elems.end(), 
  1, std::multiplies<int>());
```

尽管`std::reduce`与`std::accumulate`相比似乎更快，但在使用非交换二进制操作时，您应该小心。

折叠和递归是相辅相成的。递归函数也通过将问题分解为较小的任务并逐个解决它们来解决问题。

# 深入递归

我们已经在第二章 *使用 C++进行低级编程*中讨论了递归函数的主要特点。让我们来看一个简单的递归计算阶乘的例子：

```cpp
int factorial(int n)
{
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}
```

递归函数相对于它们的迭代对应物提供了优雅的解决方案。然而，你应该谨慎地考虑使用递归的决定。递归函数最常见的问题之一是堆栈溢出。

# 头递归

头递归是我们已经熟悉的常规递归。在前面的例子中，阶乘函数表现为头递归函数，意味着在处理当前步骤的结果之前进行递归调用。看一下阶乘函数中的以下一行：

```cpp
...
return n * factorial(n - 1);
...
```

为了找到并返回乘积的结果，函数阶乘以减小的参数（即`(n - 1)`）被调用。这意味着乘积（`*`运算符）有点像*暂停*，正在等待它的第二个参数由`factorial(n - 1)`返回。堆栈随着对函数的递归调用次数的增加而增长。让我们尝试将递归阶乘实现与以下迭代方法进行比较：

```cpp
int factorial(int n) 
{
  int result = 1;
  for (int ix = n; ix > 1; --ix) {
    result *= ix;
  }
  return result;
}
```

这里的一个主要区别是我们在相同的变量（名为`result`）中存储了每一步的乘积的结果。有了这个想法，让我们试着分解阶乘函数的递归实现。

很明显，每个函数调用在堆栈上占据了指定的空间。每一步的结果都应该存储在堆栈的某个地方。尽管我们知道应该，甚至必须是相同的变量，但递归函数并不在乎；它为它的变量分配空间。常规递归函数的反直觉性促使我们寻找一个解决方案，以某种方式知道每次递归调用的结果应该存储在同一个地方。

# 尾递归

尾递归是解决递归函数中存在多个不必要变量的问题的方法。尾递归函数的基本思想是在递归调用之前进行实际处理。以下是我们如何将阶乘函数转换为尾递归函数：

```cpp
int tail_factorial(int n, int result)
{
  if (n <= 1) return result;
  return tail_factorial(n - 1, n * result);
}
```

注意函数的新参数。仔细阅读前面的代码给了我们尾递归正在发生的基本概念：在递归调用之前进行处理。在`tail_factorial`再次在其主体中被调用之前，当前结果被计算（`n * result`）并传递给它。

虽然这个想法可能看起来并不吸引人，但如果编译器支持**尾调用优化（TCO）**，它确实非常高效。TCO 基本上涉及知道阶乘函数的第二个参数（尾部）可以在每次递归调用时存储在相同的位置。这允许堆栈保持相同的大小，独立于递归调用的次数。

说到编译器优化，我们不能忽略模板元编程。我们将它与编译器优化一起提到，因为我们可以将元编程视为可以对程序进行的最大优化。在编译时进行计算总是比在运行时更好。

# 函数式 C++中的元编程

元编程可以被视为另一种编程范式。这是一种完全不同的编码方法，因为我们不再处理常规的编程过程。通过常规过程，我们指的是程序在其生命周期中经历的三个阶段：编码、编译和运行。显然，当程序被执行时，它会按照预期的方式执行。通过编译和链接，编译器生成可执行文件。另一方面，元编程是代码在编译代码期间被*执行*的地方。如果你第一次接触这个，这可能听起来有点神奇。如果程序甚至还不存在，我们怎么能执行代码呢？回想一下我们在第四章中学到的关于模板的知识，*理解和设计模板*，我们知道编译器会对模板进行多次处理。在第一次通过中，编译器定义了模板类或函数中使用的必要类型和参数。在下一次通过中，编译器开始以我们熟悉的方式编译它们；也就是说，它生成一些代码，这些代码将由链接器链接以生成最终的可执行文件。

由于元编程是在代码编译期间发生的事情，我们应该已经对所使用的语言的概念和结构有所了解。任何可以在编译时计算的东西都可以用作元编程构造，比如模板。

以下是 C++中经典的令人惊叹的元编程示例：

```cpp
template <int N>
struct MetaFactorial
{
  enum {
    value = N * MetaFactorial<N - 1>::value
  };
};

template <>
struct MetaFactorial<0>
{
  enum {
    value = 1
  };
};

int main() {
  std::cout << MetaFactorial<5>::value; // outputs 120
  std::cout << MetaFactorial<6>::value; // outputs 720
}
```

为什么我们要写这么多代码来计算阶乘，而在上一节中我们只用不到五行的代码就写出了？原因在于它的效率。虽然编译代码需要花费一点时间，但与普通的阶乘函数（递归或迭代实现）相比，它的效率非常高。这种效率的原因在于阶乘的实际计算是在编译时发生的。也就是说，当可执行文件运行时，结果已经准备好了。我们只是在运行程序时使用了计算出的值；运行时不会发生计算。如果你是第一次看到这段代码，下面的解释会让你爱上元编程。

让我们详细分解和分析前面的代码。首先，`MetaFactorial` 模板声明为带有单个 `value` 属性的 `enum`。之所以选择这个 `enum`，仅仅是因为它的属性是在编译时计算的。因此，每当我们访问 `MetaFactorial` 的 value 属性时，它已经在编译时被计算（评估）了。看一下枚举的实际值。它从相同的 `MetaFactorial` 类中进行了递归依赖：

```cpp
template <int N>
struct MetaFactorial
{
  enum {
 value = N * MetaFactorial<N - 1>::value
 };
};
```

你们中的一些人可能已经注意到了这里的技巧。`MetaFactorial<N - 1>` 不是与 `MetaFactorial<N>` 相同的结构。尽管它们有相同的名称，但每个具有不同类型或值的模板都会生成一个单独的新类型。因此，假设我们调用类似以下的内容：

```cpp
std::cout << MetaFactorial<3>::value;
```

在这里，勤奋的编译器为每个值生成了三个不同的结构（以下是一些伪代码，表示我们应该如何想象编译器的工作）：

```cpp
struct MetaFactorial<3>
{
  enum {
    value = 3 * MetaFactorial<2>::value
  };
};

struct MetaFactorial<2>
{
  enum {
    value = 2 * MetaFactorial<1>::value;
  };
};

struct MetaFactorial<1>
{
  enum {
    value = 1 * MetaFactorial<0>::value;
  };
};
```

在下一次通过中，编译器将用其相应的数值替换生成的结构的每个值，如下伪代码所示：

```cpp
struct MetaFactorial<3>
{
  enum {
   value = 3 * 2
  };
};

struct MetaFactorial<2>
{
  enum {
    value = 2 * 1
  };
};

struct MetaFactorial<1>
{
  enum {
    value = 1 * 1
  };
};

```

然后，编译器删除未使用的生成的结构，只留下 `MetaFactorial<3>`，再次只用作 `MetaFactorial<3>::value`。这也可以进行优化。通过这样做，我们得到以下结果：

```cpp
std::cout << 6;
```

将此与我们之前的一行进行比较：

```cpp
std::cout << MetaFactorial<3>::value;
```

这就是元编程的美妙之处——它是在编译时完成的，不留痕迹，就像忍者一样。编译时间会更长，但程序的执行速度是可能的情况下最快的，与常规解决方案相比。我们建议您尝试实现其他成本昂贵的计算的元版本，比如计算第 n 个斐波那契数。这并不像为*运行时*而不是*编译时*编写代码那么容易，但您已经感受到了它的力量。

# 总结

在这一章中，我们对使用 C++有了新的视角。作为一种多范式语言，它可以被用作函数式编程语言。

我们学习了函数式编程的主要原则，比如纯函数、高阶函数和折叠。纯函数是不会改变状态的函数。纯函数的优点之一是它们留下的错误较少，否则会因为状态的改变而引入错误。

高阶函数是接受或返回其他函数的函数。除了在函数式编程中，C++程序员在处理 STL 时也使用高阶函数。

纯函数以及高阶函数使我们能够将整个应用程序分解为一系列函数的*装配线*。这个装配线中的每个函数负责接收数据并返回原始数据的新修改版本（而不是改变原始状态）。当结合在一起时，这些函数提供了一个良好协调的任务线。

在下一章中，我们将深入探讨多线程编程，并讨论在 C++中引入的线程支持库组件。

# 问题

1.  列出范围的优势。

1.  哪些函数被认为是纯函数？

1.  在函数式编程方面，纯虚函数和纯函数之间有什么区别？

1.  什么是折叠？

1.  尾递归相对于头递归的优势是什么？

# 进一步阅读

有关本章涵盖内容的更多信息，请查看以下链接：

+   *学习 C++函数式编程* 作者 Wisnu Anggoro：[`www.packtpub.com/application-development/learning-c-functional-programming`](https://www.packtpub.com/application-development/learning-c-functional-programming)

+   *在 C++中的函数式编程：如何利用函数式技术改进您的 C++程序* 作者伊万·库奇克（Ivan Cukic）：[`www.amazon.com/Functional-Programming-programs-functional-techniques/dp/1617293814/`](https://www.amazon.com/Functional-Programming-programs-functional-techniques/dp/1617293814/)
