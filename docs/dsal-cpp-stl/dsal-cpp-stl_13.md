

# 第十三章：数值和范围基础运算

在本章中，你将发现 C++ **标准模板库**（**STL**）强大的数值和排序操作潜力。这些函数为序列注入生命力，使得使用排序范围进行累积、转换和查询变得轻而易举。读者将深入了解基本和高级数值运算，并发现与排序集合一起工作的实用性。结合最佳实践，本章确保开发者拥有一个强大的工具集，以优化、并行化并优雅地处理数值数据。

本章将涵盖以下主要内容：

+   基本数值运算

+   高级数值运算

+   排序范围上的操作

+   最佳实践

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 基本数值运算

发现 C++ STL 数值函数的力量是一种令人耳目一新的体验。在本节中，我们将深入探讨基础数值运算。通过掌握这些，你将解锁生成序列、计算综合摘要以及高效执行连续元素上的复杂操作的能力。所以，系好安全带，让我们开始吧！

## 使用 std::iota 生成序列

我们将要挖掘的第一个宝藏是 `std::iota`。它是数值运算工具箱中的一个简单而强大的工具。`std::iota` 用连续值填充一个范围。从一个初始值开始，它将递增的值分配给范围中后续的元素。在这里，你可以看到 `std::itoa` 用五个连续整数填充了一个向量，从 1 开始：

```cpp
std::vector<int> vec(5);
std::iota(vec.begin(), vec.end(), 1);
// vec now holds: {1, 2, 3, 4, 5}
```

当你想要一个容器来存储许多连续的数字序列而不需要手动输入每一个时，这个函数将是一个福音。考虑这样一个场景，你想要一个 `std::vector` 来存储构造性模拟的时间步长：

```cpp
#include <iostream>
#include <numeric>
#include <vector>
int main() {
  const int numTimeSteps = 100;
  std::vector<double> timeSteps(numTimeSteps);
  // Generate a sequence of time steps using std::iota
  double timeStep = 0.01; // Time step size
  std::iota(timeSteps.begin(), timeSteps.end(), 0);
  // Scale the time steps to represent actual time
  for (double &t : timeSteps) { t *= timeStep; }
  // Now, timeSteps contains a sequence of time points for
  // simulation
  // Simulate a simple system over time (e.g., particle
  // movement)
  for (const double t : timeSteps) {
    // Simulate the system's behavior at time t
    // ...
    std::cout << "Time: " << t << std::endl;
  }
  return 0;
}
```

这里是示例输出：

```cpp
Time:0
Time: 0.01
Time: 0.02
Time: 0.03
Time: 0.04
Time: 0.05
...
```

在这个例子中，`std::iota` 用于生成时间步长的序列，这可以用来模拟系统随时间的行为。虽然这是一个简化的例子，但在实际应用中，你可以将 `std::iota` 作为更复杂模拟和建模场景的基础，例如物理模拟、金融建模或科学研究。

`std::iota` 有助于创建时间序列或离散事件时间线，这可以是各种计算模拟和建模任务的基本组成部分。当它集成到更大的、更复杂的系统中，时间序列或索引至关重要时，其价值变得更加明显。

## 使用 std::accumulate 求和元素

假设你有一系列数字，并希望找到它们的和（或者可能是一个乘积）。请使用 `std::accumulate`。此算法主要用于计算元素范围的总和。让我们看看以下简单示例的实际操作：

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
int sum = std::accumulate(vec.begin(), vec.end(), 0);
// sum will be 15
```

它主要用于计算元素范围的和，但它的功能并不仅限于此。凭借其灵活的设计，`std::accumulate` 也可以用于其他操作，例如查找乘积或连接字符串。通过提供自定义二元操作，其应用范围显著扩大。以下是一个简单的示例，说明如何使用 `std::accumulate`：

```cpp
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
int main() {
  std::vector<std::string> words = {"Hello", ", ", "world",
                                    "!"};
  std::string concatenated = std::accumulate(
      words.begin(), words.end(), std::string(""),
      [](const std::string &x, const std::string &y) {
        return x + y;
      });
  std::cout << "Concatenated string: " << concatenated
            << std::endl;
  return 0;
}
```

这里是示例输出：

```cpp
Concatenated string: Hello, world!
```

通过一些创意，`std::accumulate` 可以成为你算法工具箱中的多功能工具。

## 相邻元素及其与 `std::adjacent_difference` 的交互

有时，我们感兴趣的是单个元素和相邻元素的对。STL 通过 `std::adjacent_difference` 为此提供了支持。

`std::adjacent_difference` 计算一个元素与其前驱之间的差值，并将其存储在另一个序列中。这种操作在计算离散导数等任务中很有用。

以下代码演示了 `std::adjacent_difference` 的用法：

```cpp
std::vector<int> vec = {2, 4, 6, 8, 10};
std::vector<int> result(5);
std::adjacent_difference(vec.begin(), vec.end(), result.begin());
// result holds: {2, 2, 2, 2, 2}
```

不仅限于差异，你还可以将自定义二元操作传递给 `std::adjacent_difference` 以实现不同的结果，例如比率。让我们看看以下示例：

```cpp
#include <iostream>
#include <numeric>
#include <vector>
int main() {
  std::vector<double> values = {8.0, 16.0, 64.0, 256.0,
                                4096.0};
  // Create a vector to store the calculated ratios
  std::vector<double> ratios(values.size());
  // Write a lambda to use in adjacent_difference
  auto lambda = [](double x, double y) {
    if (x == 0.0) {
      // Handle division by zero for the first element
      return 0.0;
    } else {
      // Calculate the ratio between y and x
      return y / x;
    }
  };
  // Calculate the ratios between consecutive elements
  std::adjacent_difference(values.begin(), values.end(),
                           ratios.begin(), lambda);
  // The first element in the ratios vector is 0.0 because
  //there's no previous element
  // Print the calculated ratios for the remaining elements
  std::cout << "Ratios between consecutive elements:\n";
  for (size_t i = 1; i < ratios.size(); ++i) {
    std::cout << "Ratio " << i << ": " << ratios[i]
              << "\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
Ratios between consecutive elements:
Ratio 1: 0.5
Ratio 2: 0.25
Ratio 3: 0.25
Ratio 4: 0.0625
```

## 使用 `std::inner_product` 的内积

对于那些涉足线性代数的人来说，这个函数是一个奇迹。`std::inner_product` 计算两个范围的点积。你可能还记得，点积是两个序列中对应对的乘积之和。让我们看看如何计算两个向量的点积：

```cpp
std::vector<int> vec1 = {1, 2, 3};
std::vector<int> vec2 = {4, 5, 6};
int product = std::inner_product(vec1.begin(), vec1.end(),
                                 vec2.begin(), 0);
// product will be 32 (because 1*4 + 2*5 + 3*6 = 32)
```

`std::inner_product` 不仅限于整数或普通乘法。自定义二元操作可以针对不同类型和操作进行定制。

这里有一些现实世界的例子，以证明 `std::inner_product` 可以与针对不同类型和操作（而不仅仅是整数和普通乘法）定制的自定义二元操作一起工作：

+   `std::inner_product` 用于计算两个容器中元素的加权平均值，其中一个容器包含值，另一个容器包含相应的权重。自定义二元操作将执行值和权重的逐元素乘法，然后将它们相加以找到加权平均值。

+   使用自定义二元操作计算投资组合的总价值，通过将资产价格乘以其相应的数量并求和。

+   `std::inner_product` 可以使用自定义二元操作来完成此目的。

+   `std::inner_product` 可以通过自定义二元操作进行适配，以高效地执行矩阵乘法。

+   使用`std::inner_product`执行复数运算，例如计算两个复数向量的内积或找到复数平方的和。自定义的二元操作将针对复数算术进行定制。

+   使用自定义的二元操作将字符串连接起来的`std::inner_product`。这允许你高效地连接字符串集合。

+   `std::inner_product`可以通过自定义的二元操作进行适配，以根据所需的算法执行颜色混合。

这些示例说明`std::inner_product`是一个多才多艺的算法，可以根据各种类型和操作进行定制。这使得它在许多现实世界的应用中非常有用，而不仅仅是简单的整数乘法。

在本节中，我们看到了 C++ STL 提供的基本数值操作为高效计算、生成和操作序列铺平了道路。它们改变了开发者解决问题的方法，使得快速有效的解决方案成为可能。正如我们所见，这些算法是多才多艺的，只需一点创意，就可以适应无数任务。

在你的工具包中有了这些工具，你现在可以生成序列，计算快速摘要，并对连续元素执行复杂操作。

# 高级数值操作

为了进一步探索 C++ STL 的数值操作之旅，让我们来看看那些提升数据处理能力并使并行性和并发成为性能追求盟友的高级数值过程。

记得我们关于生成序列和计算摘要的讨论吗？好吧，想象一下通过利用多个处理器的力量，将这些操作超级充电以高效处理大量数据。这正是高级数值操作大放异彩的地方。C++17 中引入的并行算法提供了实现这一目标的方法，确保我们的计算既快速又高效，即使在并发环境中也是如此。

当处理大量数据集时，顺序处理通常不够。以计算大量数字向量的总和为例。直接进行操作可以完成任务，但可能不是最快的。然而，通过分割数据并在多个数据块上并行工作，可以显著加快操作速度。这正是并行算法的精髓，而像`std::reduce`这样的函数就是这一点的例证。`std::reduce`不是顺序累积值，而是在并行中累积子总计，然后合并它们，为大型数据集提供了显著的性能提升。

为了看到这一过程在实际中的应用，让我们并行计算一个大型向量中所有数字的总和：

```cpp
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
int main() {
  // Create a large vector of numbers
  const int dataSize = 1000000;
  std::vector<int> numbers(dataSize);
  // Initialize the vector with some values (e.g., 1 to
  // dataSize)
  std::iota(numbers.begin(), numbers.end(), 1);
  // Calculate the sum of the numbers in parallel
  int parallelSum = std::reduce(
      std::execution::par, numbers.begin(), numbers.end());
  std::cout << "Parallel Sum: " << parallelSum
            << std::endl;
  return 0;
}
```

下面是示例输出：

```cpp
Parallel Sum: 1784293664
```

深入并行操作需要细致的方法。虽然速度的承诺很有吸引力，但必须谨慎。并行引入了挑战，如确保线程安全和处理数据竞争。幸运的是，STL 通过执行策略提供了补救措施。通过指定执行策略，例如在调用算法时使用`std::execution::par`，我们可以指导它并行运行。此外，还有`std::execution::par_unseq`，用于并行和向量化执行，确保更高的吞吐量。

说到转换，让我们来看看`std::transform_reduce`。这是`std::transform`和`std::reduce`的结合。它对每个范围元素应用转换函数，并将结果归约成一个单一值，这可以并行化。例如，如果我们有一个数字向量，并想对每个元素进行平方然后求和，`std::transform_reduce`将是我们的首选，尤其是在处理大量数据时。

让我们看看如何使用`std::transform_reduce`对向量的每个元素进行平方，然后对平方值求和：

```cpp
#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
int main() {
  const long int dataSize = 1000;
  std::vector<long int> numbers(dataSize);
  std::iota(numbers.begin(), numbers.end(), 1);
  // Use std::transform_reduce to square each element and
  // sum them up in parallel
  long int parallelSumOfSquares = std::transform_reduce(
      std::execution::par, numbers.begin(), numbers.end(),
      0, // Initial value for the accumulation
      std::plus<long int>(),
      [](long int x) { return x * x; });
  std::cout << "Parallel Sum of Squares:"
            << parallelSumOfSquares << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
Parallel Sum of Squares: 333833500
```

高级操作的另一项亮点是`std::inclusive_scan`和`std::exclusive_scan`这对组合。这些是生成前缀和的强大工具。`std::inclusive_scan`将第 i 个输入元素包含在第 i 个和中，而`std::exclusive_scan`则不包含。像它们的其他高级数值操作一样，它们也可以通过并行执行来增强性能。

重要提示

在输入序列中的`i`，输出序列中相应的元素包含从输入序列索引`0`到`i`的所有元素的总和。

并行操作可能非常耗费资源。确保硬件能够处理并行性，并且数据量足够大，足以证明并发执行的开销是必要的。此外，始终警惕潜在的问题，如数据竞争或死锁。关键在于不断权衡利弊，分析具体要求，并选择最合适的方案。

# 对排序范围的运算

排序的吸引力并不仅仅是为了整齐排列元素。相反，它赋予我们在后续操作中的强大能力——简化导航、高效查询和增强的操纵能力。对于 C++开发者来说，理解对排序范围的运算就像是获得了一套新的超级能力。凭借 C++ STL 为这些排序序列提供的工具，高效的算法操作世界变得一片开阔，等待探索。

那么，拥有排序范围有什么大不了的？考虑一下在杂乱无章的一堆书中找书和在整齐排列的书架上找书的区别。当数据排序后，算法可以采取捷径，例如分而治之，从而实现对数时间复杂度而不是线性时间复杂度。

在排序范围中，一个主要的技巧是利用 `std::lower_bound` 和 `std::upper_bound`，这两个函数是您实现此目的的首选。前者用于找到应该插入值以保持顺序的第一个位置，而后者则标识最后一个合适的点。共同使用，它们可以确定与给定值等效的条目范围。如果您曾对某些应用程序返回搜索结果的快速性感到惊奇，那么这些二分搜索技术通常要归功于它们。

继续讨论查询的话题，`std::equal_range` 作为上述函数的组合出现，返回排序范围内一个值的上下界；如果您只需要一个简单的检查，`std::binary_search` 会告诉您元素是否存在于排序范围内。这些工具简化了查询过程，使其既快速又精确。

然而，对排序范围的运算并不局限于搜索。集合运算，类似于我们的基础数学课程，在排序数据中变得生动起来。如果您有两个排序序列，并希望确定它们的公共元素，`std::set_intersection` 就是完成这项工作的工具。对于属于一个序列但不属于另一个序列的元素，转向 `std::set_difference`。如果您想要合并两个序列的元素同时保持排序顺序，`std::set_union` 就准备好了。最后但同样重要的是，为了找到每个序列中独特的元素，`std::set_symmetric_difference` 扮演着这个角色。

想象一下这些运算赋予我们的力量。比较两个大型数据集以找到共同点或差异是许多应用程序的常见需求，从数据库到数据分析。通过在排序范围内工作，这些运算变得可行且高效。

排序操作合理地假设数据是有序的。如果这个不变量没有得到维护，结果可能是不可预测的。因此，在深入这些操作之前，确保排序顺序至关重要。幸运的是，通过像 `std::is_sorted` 这样的函数，可以在进一步操作之前验证范围的排序性质。

让我们将所有这些概念结合起来，快速看一下它们是如何被使用的：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> d = {10, 20, 30, 40, 50,
                        60, 70, 80, 90};
  int tgt = 40;
  auto lb = std::lower_bound(d.begin(), d.end(), tgt);
  auto ub = std::upper_bound(d.begin(), d.end(), tgt);
  bool exists =
      std::binary_search(d.begin(), d.end(), tgt);
  std::vector<int> set1 = {10, 20, 30, 40, 50};
  std::vector<int> set2 = {30, 40, 50, 60, 70};
  std::vector<int> intersection(
      std::min(set1.size(), set2.size()));
  auto it = std::set_intersection(set1.begin(), set1.end(),
                                  set2.begin(), set2.end(),
                                  intersection.begin());
  std::vector<int> difference(
      std::max(set1.size(), set2.size()));
  auto diffEnd = std::set_difference(
      set1.begin(), set1.end(), set2.begin(), set2.end(),
      difference.begin());
  bool isSorted = std::is_sorted(d.begin(), d.end());
  std::cout << "Lower Bound:"
            << std::distance(d.begin(), lb) << "\n";
  std::cout << "Upper Bound:"
            << std::distance(d.begin(), ub) << "\n";
  std::cout << "Exists: " << exists << "\n";
  std::cout << "Intersection: ";
  for (auto i = intersection.begin(); i != it; ++i)
    std::cout << *i << " ";
  std::cout << "\n";
  std::cout << "Difference: ";
  for (auto i = difference.begin(); i != diffEnd; ++i)
    std::cout << *i << " ";
  std::cout << "\n";
  std::cout << "Is Sorted: " << isSorted << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Lower Bound: 3
Upper Bound: 4
Exists: 1
Intersection: 30 40 50
Difference: 10 20
Is Sorted: 1
```

从这些例子中可以看出，对排序范围的运算解锁了一个广阔的可能性领域。它们展示了数学理论与实际编码的结合，为开发者提供了一个强大的框架，以无与伦比的效率导航、查询和操作数据。随着我们继续前进，我们将探讨与数值和基于范围的运算相关的最佳实践，确保我们在利用它们的力量时，能够做到精确、高效和优雅。探索和掌握的旅程仍在继续！

# 最佳实践

以下是一些与数值和基于范围的运算相关的最佳实践：

+   对于几乎已排序的数据集，`std::stable_sort` 可能比其他排序方法更有效。因此，在决定适当的操作时，了解数据集的特征至关重要。

+   在进行排序操作之前推荐使用 `std::is_sorted`。

+   **明智地使用并行算法**：随着对并发的日益重视，并行算法成为提高性能的一个有吸引力的选择。C++ STL 为许多标准算法提供了并行版本。虽然这些算法利用多个 CPU 核心来提供更快的执行结果，但它们也可能引入挑战，尤其是在线程安全方面。并发编程中的一个主要问题是共享可变状态。当多个线程试图同时修改相同的数据时，就会产生问题。为了安全地使用并行算法，线程要么在独立的数据部分上工作，要么使用如互斥锁之类的同步工具来管理同时的数据修改。

    此外，并行化并不总是答案。管理多个线程的开销有时可能会抵消并行执行的好处，尤其是在处理小数据集或简单任务时。为了确定并行化在特定场景中的有效性，最好在顺序和并行配置下对代码进行性能分析。这种评估有助于选择最有效的方法。

在本节中，我们探讨了如何根据数据属性在 C++ STL 中选择合适的算法，强调了数据集特征（如大小和分布）的重要性。对于几乎已排序的数据，选择如 `std::stable_sort` 这样的适当算法对于最佳性能至关重要。我们还强调了在排序操作中维护数据顺序的必要性，使用如 `std::is_sorted` 这样的工具来确保数据完整性。讨论了并行算法，重点关注它们的优点和复杂性，如线程安全。关键要点是，虽然并行化功能强大，但需要仔细考虑，尤其是在数据集大小和任务复杂性方面。

# 摘要

在本章中，我们深入研究了 C++ STL 提供的用于处理数值序列并在排序范围内操作的算法的多样世界。我们从基本的数值操作开始，例如使用 `std::iota` 生成序列，使用 `accumulate` 累加元素，以及使用 `std::adjacent_difference` 探索相邻元素之间的交互。本章探讨了更复杂的任务，例如使用 `std::inner_product` 计算内积。

这些操作在 STL 容器中的数据处理和分析中是必不可少的，它们简化了从简单累加到复杂转换的各种任务。对于开发者来说，这些信息至关重要，因为它在执行数值计算时提高了效率和效果，并为他们准备应对高性能场景，尤其是在处理大数据集时。

本章还涵盖了高级数值运算，这在并行计算环境中尤其有益。我们学习了如何使用并行算法进行数据转换和汇总，确保在并发环境中高性能。对排序范围的操作得到了探讨，展示了二分搜索技术的效率和集合运算的功能，这些操作由于数据的排序性质而得到了显著优化。

在下一章中，我们将探索范围的概念，这代表了 C++ 中序列的更现代的方法。我们将探讨为什么转向基于范围的运算变得流行，理解这些现代 STL 组件的本质和力量，并探索它们在排序和搜索算法中的可组合性。这一即将到来的章节将赋予读者拥抱现代 STL 全部潜能的知识，使他们能够在 C++ 编程实践中做出明智的决策，了解何时以及如何应用这些新工具。
