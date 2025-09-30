# 6

# 高级序列容器使用

序列容器是 C++数据处理的核心，提供线性存储数据的结构。对于中级开发者来说，从包括向量、数组、deque 和列表在内的序列容器数组中选择正确的选项可能是至关重要的。本章将详细分析每种容器类型，强调它们的独特优势和理想使用场景。此外，深入了解最佳实践——从高效的调整大小到迭代器管理——将确保开发者选择正确的容器并有效地使用它。掌握这些细微差别将提高代码在实际应用中的效率、可读性和可维护性。

在庞大的 C++ **标准模板库**（**STL**）中，序列容器占据着显赫的位置。这不仅是因为它们通常是开发者首选的数据结构，也因为每个容器提供的独特和多功能解决方案。正如其名称所暗示的，这些容器按顺序维护元素。但当我们深入了解时，我们会发现相似之处通常到此为止。每个序列容器都带来其优势，并针对特定场景进行定制。

在本章中，我们将涵盖以下主要主题：

+   `std::array`

+   `std::vector`

+   `std::deque`

+   `std::list`

+   `std::forward_list`

+   `std::string`

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# std::array

`std::array`是一个固定大小的容器，它围绕传统的 C 风格数组。如果你来自 C 背景，甚至早期的 C++，你将熟悉原始数组的烦恼——缺乏边界检查、繁琐的语法等等。使用`std::array`，你可以获得传统数组的所有好处，如静态内存分配和常数时间访问，同时享受现代 C++的便利，包括基于范围的 for 循环和用于大小检查的成员函数。当你事先知道数据集的大小且不会改变时，使用`std::array`是完美的。在性能至关重要且内存需求静态的场景中，它非常适用。

注意

关于 C++核心指南的更多信息，请参阅*C++核心* *指南* [`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

## 目的和适用性

`std::array`是一个封装固定大小数组的容器。其优势如下：

+   可预测的固定大小

+   栈分配，提供快速访问和最小开销

在以下情况下，最好选择`std::array`：

+   数组的长度在编译时已知。

+   最小化开销和性能可预测性至关重要。

对于动态大小需求，请考虑使用`std::vector`。

## 理想使用场景

以下是一些`std::array`的理想使用场景：

+   `std::array`是首选选择。这使得它在维度大小预定义的情况下非常适合，例如在某些数学运算或游戏板表示中。

+   `std::array`不涉及动态内存分配，这可以有利于实时或性能关键的应用。

+   `std::array`提供了边界检查（通过`at()`成员函数），提供了比 C 风格数组更安全的替代方案，尤其是在处理潜在的越界访问时。

+   `std::array`保留了大小信息，这使得编写更安全、更直观的函数变得更容易。

+   `std::array`可以无缝地与 C 风格数组一起使用，使其成为具有 C 和 C++集成的项目的绝佳选择。

+   `std::array`是最佳选择。

+   `std::vector`和`std::array`提供连续的内存存储，这使得它们在迭代时对缓存友好。

+   `std::array`确保没有意外的内存分配。

+   `std::array`提供了方便的初始化语义。

然而，尽管`std::array`在传统数组之上提供了几个优势，但必须注意，它并不适合需要动态调整大小的场景。对于这些用例，可以考虑`std::vector`或其他动态容器。

## 性能

`std::array`的算法性能如下所述：

+   **插入**：由于大小固定，不适用

+   **删除**：不适用

+   **访问**：任何位置的常数*O(1)*

+   **内存开销**：由于栈分配，最小化。

+   **权衡**：固定大小的效率是以静态大小为代价的。

## 内存管理

与`std::vector`不同，`std::array`不会动态分配内存。它是栈分配的，因此没有意外的分配行为或惊喜。

## 线程安全

你是否在多线程中读取？完全可以。然而，并发写入同一元素需要同步。

## 扩展和变体

对于动态需求，`std::vector`是 STL 的主要替代方案。其他固定大小数组选项包括传统的 C 风格数组。

## 排序和搜索复杂度

+   `std::sort()`

+   `std::binary_search()`

## 接口和成员函数

存在标准函数，如`begin()`、`end()`和`size()`。值得注意的成员函数如下：

+   `fill`：将所有元素设置为某个值

+   `swap`：与相同类型和大小的另一个数组交换内容

## 比较操作

与`std::vector`相比，`std::array`不进行大小调整，但提供可预测的性能。在做出选择时，权衡动态大小需求与性能一致性。

## 与算法的交互

STL 算法由于随机访问能力而很好地与`std::array`协同工作。然而，那些期望动态大小的将无法与`std::array`一起工作。

## 异常

使用`std::array`时，越界访问（如使用`at()`）可以抛出异常，主要是`std::out_of_range`。

## 定制化

虽然不能调整大小，但可以集成自定义类型。鉴于容器栈分配的特性，确保它们高效可移动/可复制。

## 示例

在本例中，我们将展示以下最佳实践和 `std::array` 的使用：

+   使用固定大小的 `std::array`

+   使用 C++ 结构化绑定与 `std::array` 解构元素

+   使用 `std::array` 实现编译时计算（归功于其 constexpr 特性）

+   使用 `std::sort` 和 `std::find` 等算法与 `std::array`

结构化绑定

C++17 中引入的结构化绑定，允许方便且易于阅读地从元组、对或类似结构体对象中解包元素到单独命名的变量中。这种语法简化了访问从函数返回的多个元素或分解复杂数据结构的内容，增强了代码的清晰度并减少了冗余。

下面是讨论上述观点的代码示例：

```cpp
#include <algorithm>
#include <array>
#include <iostream>
struct Point {
  int x{0}, y{0};
};
constexpr int sumArray(const std::array<int, 5> &arr) {
  int sum = 0;
  for (const auto &val : arr) { sum += val; }
  return sum;
}
int main() {
  std::array<int, 5> numbers = {5, 3, 8, 1, 4};
  std::array<Point, 3> points = {{{1, 2}, {3, 4}, {5, 6}}};
  // Demonstrating structured bindings with &[x, y]
  for (const auto &[x, y] : points) {
    std::cout << "(" << x << ", " << y << ")\n";
  }
  constexpr std::array<int, 5> constNumbers = {1, 2, 3, 4,
                                               5};
  constexpr int totalSum = sumArray(constNumbers);
  std::cout << "\nCompile-time sum of array elements: "
            << totalSum << "\n";
  std::sort(numbers.begin(), numbers.end());
  std::cout << "\nSorted numbers: ";
  for (const auto &num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  int searchFor = 3;
  if (std::find(numbers.begin(), numbers.end(),
                searchFor) != numbers.end()) {
    std::cout << "\nFound " << searchFor
              << " in the array.\n";
  } else {
    std::cout << "\nDidn't find " << searchFor
              << " in the array.\n";
  }
  return 0;
}
```

此示例突出了 `std::array` 的特性和优势，包括其固定大小特性、与结构化绑定等现代 C++ 功能的兼容性，以及在编译时计算中的实用性。前面的示例还说明了如何无缝地将 STL 算法应用于 `std::array`。

## 最佳实践

让我们探索使用 `std::array` 的最佳实践：

+   `std::array` 封装了 C 风格数组的可预测性，并为其增加了额外的功能。其固定大小在数据大小已确定的情况下特别有用，使其成为此类应用的优选。

+   `std::array` 提供了一系列成员函数，提高了其功能。这使得它在当代 C++ 开发中相对于其传统对应物更具吸引力。

+   `.at()` 成员函数非常有价值。它通过在越界时抛出异常来防止越界访问。

+   `std::array` 既是其优势也是其限制。它承诺常数时间访问，但在调整大小方面缺乏灵活性。因此，在声明时精确指定所需的大小对于防止潜在问题至关重要。

+   `std::array` 采用这种循环结构最小化了边界错误的概率，促进了代码的稳定性。

+   `std::array` 可以容纳多种类型，考虑到效率至关重要。如果类型，无论是原始类型还是用户定义类型，特别大或复杂，确保其移动或复制操作已优化，以在数组操作期间保持性能。

+   `std::array` 在需要固定大小容器的场景中表现出色。然而，对于预期动态调整大小或大量数据的适用，`std::vector` 等替代方案可能提供更灵活的解决方案。

# `std::vector`

`std::vector` 是一个动态数组。它根据需要增长和缩小，在直接访问性能和大小灵活性之间提供了良好的平衡。`std::vector` 具有缓存友好的连续内存布局，在末尾提供摊销常数时间的插入，使其成为一个优秀的通用容器。当你的主要操作是索引和需要动态调整大小，但中间没有频繁的插入或删除时，性能最佳。

## 目的和适用性

`std::vector` 在 STL 中本质上是一个动态数组。它的主要优势在于以下方面：

+   提供常数时间的随机访问

+   元素插入或删除时动态调整大小

当以下要求时，它尤其适合：

+   随机访问至关重要。

+   插入或删除主要在序列的末尾进行。

+   缓存局部性至关重要。

当常数时间访问、性能或缓存友好性比其他关注点更重要时，选择 `std::vector`。

## 理想用例

以下是一些 `std::vector` 的理想用例：

+   `std::vector` 是你的最佳选择。与标准数组不同，向量会自动管理其大小并无缝处理内存分配/释放。

+   `std::vector` 提供了对任何元素的常数时间访问，使其适合频繁访问或修改特定索引的数据。

+   `std::vector` 提供了适合此目的的连续内存块。

+   `std::vector` 对其末尾的插入进行了优化，因此对于日志系统等新条目持续添加的应用程序来说是一个不错的选择。

+   由于其连续内存块和缺乏结构开销，`std::vector` 提供了存储数据最内存高效的方式，与基于链表的容器不同。

+   `std::vector` 是缓存友好的，与不连续的数据结构相比，在许多场景中性能更快。

+   `std::vector` 可以轻松地进行流式传输或写入。

+   为了实现这个目的，`std::vector` 可以有效地实现一个栈数据结构，其中元素只从末尾添加或删除。

+   `std::vector` 为此提供了一个高效且动态的容器。

然而，在使用 `std::vector` 时有一些注意事项。如果需要在中间频繁插入或删除，由于需要移动元素，`std::vector` 可能不是最有效率的选项。另外，如果你经常推送元素，使用 `reserve()` 预分配内存并避免频繁重新分配是一个好习惯。

## 性能

`std::vector` 的算法性能特点如下：

+   **插入：** 末尾的平均情况为 *O(1)*，其他位置为 *O(n)*

+   **删除：** 末尾为 *O(1)*，中间为 *O(n)*

+   **访问：** 任何位置的快速 *O(1)*

+   **内存开销：** 通常较低，但如果未管理预留容量，可能会膨胀

+   **权衡：** *O(1)* 访问的便利性被在开始或中间插入的潜在 *O(n)* 成本所抵消。

## 内存管理

`std::vector` 自动管理其内存。如果其容量耗尽，它通常会将其大小加倍，尽管这并不是强制性的。分配器可以影响这种行为，允许细粒度控制。

## 线程安全

并发读取？没问题。但是写入，或者读取和写入的混合，需要外部同步。考虑互斥锁或其他并发工具。

## 扩展和变体

虽然 `std::vector` 是一个动态数组，但 STL 提供了其他序列容器，如 `std::deque`，它提供了在两端快速插入的 API 或 `std::list`，可能优化中间插入和删除。

## 排序和搜索复杂度

排序和搜索复杂度如下所述：

+   `std::sort()`

+   `std::binary_search()`

## 特殊接口和成员函数

除了常规操作（如 `push_back`、`pop_back`、`begin` 和 `end`）之外，熟悉以下内容：

+   `emplace_back`: 直接构造元素

+   `resize`: 改变元素数量

+   `shrink_to_fit`: 减少内存使用

## 比较

与 `std::list` 和 `std::deque` 相比，`std::vector` 在随机访问方面表现出色，但在频繁修改非常大的数据类型的中间部分时可能会失败。

## 与算法的交互

许多 STL 算法与 `std::vector` 的随机访问特性非常和谐。然而，需要频繁重新排序的算法可能更适合与其他容器搭配。

## 异常

超出容量或访问越界索引可能会抛出异常。值得注意的是，操作是异常安全的，即使在操作（如插入）抛出异常的情况下，也能保留向量的状态。

## 定制化

使用自定义分配器时，调整内存分配策略。然而，`std::vector` 并不支持自定义比较器或哈希函数。

## 示例

在这个例子中，我们将展示以下最佳实践和 `std::vector` 的使用：

+   使用 `reserve` 预分配内存

+   使用 `emplace_back` 进行高效插入

+   使用迭代器进行遍历和修改

+   使用自定义对象与 `std::vector` 结合

+   使用 `std::remove` 等算法与 `std::vector` 结合

下面是代码示例：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
class Employee {
public:
  Employee(int _id, const std::string &_name)
      : id(_id), name(_name) {}
  int getId() const { return id; }
  const std::string &getName() const { return name; }
  void setName(const std::string &newName) {
    name = newName;
  }
private:
  int id{0};
  std::string name;
};
int main() {
  std::vector<Employee> employees;
  employees.reserve(5);
  employees.emplace_back(1, "Lisa");
  employees.emplace_back(2, "Corbin");
  employees.emplace_back(3, "Aaron");
  employees.emplace_back(4, "Amanda");
  employees.emplace_back(5, "Regan");
  for (const auto &emp : employees) {
    std::cout << "ID: " << emp.getId()
              << ", Name: " << emp.getName() << "\n";
  }
  auto it = std::find_if(
      employees.begin(), employees.end(),
      [](const Employee &e) { return e.getId() == 3; });
  if (it != employees.end()) { it->setName("Chuck"); }
  std::cout << "\nAfter Modification:\n";
  for (const auto &emp : employees) {
    std::cout << "ID: " << emp.getId()
              << ", Name: " << emp.getName() << "\n";
  }
  employees.erase(std::remove_if(employees.begin(),
                                 employees.end(),
                                 [](const Employee &e) {
                                   return e.getId() == 2;
                                 }),
                  employees.end());
  std::cout << "\nAfter Removal:\n";
  for (const auto &emp : employees) {
    std::cout << "ID: " << emp.getId()
              << ", Name: " << emp.getName() << "\n";
  }
  return 0;
}
```

上述示例展示了 `std::vector` 与 C++ STL 算法结合的效率和灵活性。它展示了以各种方式管理和操作 `Employee` 对象列表。

现在，让我们看看一个 `std::vector<bool>` 的示例。

`std::vector<bool>` 是 C++ 标准库中的一个有些争议的特殊化。它被设计为每个布尔值只使用一个比特位，从而节省空间。然而，这种空间优化导致了几个意想不到的行为和怪癖，尤其是在与其他类型的 `std::vector` 相比时。

由于这些原因，许多专家建议在使用 `std::vector<bool>` 时要谨慎。尽管如此，如果仍然希望使用它，以下是一个展示其使用和一些怪癖的典型示例：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<bool> boolVec = {true, false, true, true,
                               false};
  boolVec[1] = true;
  std::cout << "Second element: " << boolVec[1] << '\n';
  auto ref = boolVec[1];
  ref = false;
  std::cout << "Second element after modifying copy: "
            << boolVec[1] << '\n';
  // Iterating over the vector
  for (bool val : boolVec) { std::cout << val << ' '; }
  std::cout << '\n';
  // Pushing values
  boolVec.push_back(false);
  // Resizing
  boolVec.resize(10, true);
  // Capacity and size
  std::cout << "Size: " << boolVec.size()
            << ", Capacity: " << boolVec.capacity()
            << '\n';
  // Clearing the vector
  boolVec.clear();
  return 0;
}
```

上述代码的关键要点如下：

+   `std::vector<bool>` 通过将布尔值存储为单独的位来节省内存。

+   当从 `std::vector<bool>` 访问元素时，你不会得到与其他向量类型相同的普通引用。相反，你得到一个代理对象。这就是为什么在示例中修改 `ref` 并不会改变向量中的实际值。

+   其他操作，如迭代、调整大小和容量检查，与其他 `std::vector` 类型的工作方式类似。

对于许多应用，`std::vector<bool>` 的特殊性可能超过其内存节省的好处。如果内存优化不是至关重要的，并且行为上的怪癖可能成为问题，请考虑使用其他容器，如 `std::deque<bool>`、`std::bitset` 或第三方位集/向量库。

## 最佳实践

让我们探索使用 `std::vector` 的最佳实践：

+   `std::vector<bool>` 不仅仅是一个简单的布尔值向量。它被专门化以节省空间，这种空间效率是以牺牲为代价的：元素不是真正的布尔值，而是位字段代理。这种专门化可能导致某些操作中独特的表现，因此完全理解其复杂性至关重要。

+   `std::vector` 的动态调整大小能力。虽然这很强大，但使用 `reserve` 函数预测和引导这种调整可能会有所帮助。预分配内存有助于最小化重新分配，并确保高效性能。

+   `push_back` 是一个常用的方法来添加元素，`emplace_back` 提供了一种更高效的方法直接在向量中构造对象。就地构造对象通常可以增强性能，尤其是在复杂对象中。

+   `std::vector` 提供了优秀的随机访问性能。然而，由于需要移动后续元素，中间的操作，如插入或删除，可能会更加耗时。对于需要频繁中间操作的任务，考虑替代的 STL 容器是值得的。

+   `.at()` 成员函数另一方面，提供带边界检查的访问，如果使用无效索引，将抛出 `std::out_of_range` 异常。

+   `std::vector`。虽然 `std::vector` 本身不是线程安全的，但可以使用适当的同步工具，如互斥锁，来实现线程安全。

+   `std::vector` 并未针对频繁的中间插入或删除进行优化。然而，其缓存友好性和快速搜索的能力可能仍然意味着它是最佳的数据类型。将其用作链表可能不是最优的，但仅适用于特定的用例（可能是非常大的数据类型或非常大的数据集）。对于这种模式，容器如 `std::list` 可能更适合。然而，永远不要假设 `std::list` 仅因为需要频繁的插入和删除就会表现更好。

+   `std::map`，其中向量可能是值，不要陷入自动更新的陷阱。显式管理和更新这些嵌套容器是至关重要的。

# `std::deque`

`std::deque` 是一个双端队列。表面上，它看起来像 `std::vector`，在两端进行插入和删除操作更好。虽然这是真的，但请记住，这种灵活性是以略微复杂的内部结构为代价的。如果你的应用程序需要在两端进行快速插入和删除，但不需要 `std::vector` 的紧凑内存布局，那么 `std::deque` 就是你的首选容器。

## 目的和适用性

`std::deque` 是一个提供快速在两端进行插入和删除操作的容器。其主要优势如下：

+   两端高效的 *O(1)* 插入和删除

+   动态大小，无需手动内存管理

+   前端和后端操作具有相当好的缓存性能

`std::deque` 在以下方面表现出色：

+   你需要随机访问能力，但预计两端将频繁修改。

+   你需要一个动态大小的容器，但不想有 `std::list` 的内存开销。

如果只需要端修改，`std::vector` 可能是一个更好的选择。

## 理想用例

以下是一些 `std::deque` 的理想用例：

+   `std::vector` 主要允许在末尾快速插入，而 `std::deque` 支持在两端快速插入和删除，使其非常适合需要两端操作的场景。

+   `std::deque` 可以作为队列（FIFO 数据结构）和栈（LIFO 数据结构）。在这方面，它非常灵活，不像其他专注于一个或另一个的容器。

+   `std::vector` 和 `std::deque` 提供了对元素的常数时间随机访问，这使得它们适合需要通过索引访问元素的应用程序。

+   `std::vector` 只向一个方向增长，而 `std::deque` 可以向两个方向增长。这使得它在数据集可能在两端不可预测地扩展的情况下特别有用。

+   `std::deque` 可以在从前端（播放）消耗数据时有所帮助。新的数据可以在后端缓冲，而无需重新整理整个数据集。

+   `std::deque` 提供了一个高效的解决方案。

+   `std::deque` 可以轻松地适应其他自定义数据结构。例如，一个平衡树或特定类型的优先队列可能会利用 `std::deque` 的功能。

+   `std::deque` 可以有效地处理添加新操作和自动删除最旧的项。

考虑到 `std::deque` 时，必须权衡其双端和随机访问特性的好处与相对于 `std::vector` 的略高每元素开销。在其他只有单侧增长的场景中，其他数据结构可能更节省空间。

## 性能

`std::deque` 的算法性能如下：

+   **插入：** 前端和后端都是 *O(1)*；中间是 *O(n)*

+   **删除：** 前端和后端都是 *O(1)*；中间是 *O(n)*

+   **访问：** 随机访问保持一致的 *O(1)*

+   由于分段内存，`std::vector`

## 内存管理

`std::deque` 使用分段分配，这意味着它根据需要分配内存块。与 `std::vector` 不同，它不会加倍其大小；因此，没有过度的内存开销。自定义分配器可以影响内存分配策略。

## 线程安全

并发读取是安全的。但像大多数 STL 容器一样，同时写入或读写混合需要外部同步机制，如互斥锁。

## 扩展和变体

在性能和内存特性方面，`std::deque` 位于 `std::vector` 和 `std::list` 之间。然而，它独特地提供了两端快速操作。

## 排序和搜索的复杂度

排序和搜索的复杂度如下：

+   `std::sort()`

+   `std::binary_search()`

## 接口和成员函数

除了熟悉的（`push_back`、`push_front`、`pop_back`、`pop_front`、`begin` 和 `end`）之外，熟悉以下内容：

+   `emplace_front` 和 `emplace_back`：在各自的端进行就地构造

+   `resize`：调整容器大小，根据需要扩展或截断

## 比较

与 `std::vector` 相比，`std::deque` 提供了更好的前端操作。与 `std::list` 相比，它提供了更好的随机访问，但在中间插入/删除方面可能表现不佳。与 `std::vector` 相比，`std::deque` 的非连续存储可能在迭代元素时成为劣势，因为缓存性能较差。

## 与算法的交互

由于 `std::deque` 的随机访问特性，它可以有效地利用大多数 STL 算法。特别适合 `std::deque` 的算法是那些需要快速端修改的算法。

## 异常

超出大小或访问越界索引可能导致异常。如果插入等操作抛出异常，容器保持完整，确保异常安全。

## 定制化

`std::deque` 可以与自定义分配器一起使用，以定制内存分配行为，但它不支持自定义比较器或哈希函数。

## 示例

在本例中，我们将展示以下最佳实践和 `std::deque` 的使用：

+   使用 `std::deque` 维护元素列表，利用其动态大小

+   在前后两端插入元素

+   从前后两端高效移除元素

+   使用 `std::deque` 作为滑动窗口以分块处理元素

+   应用 STL 算法，如 `std::transform`

下面是代码示例：

```cpp
#include <algorithm>
#include <deque>
#include <iostream>
// A function to demonstrate using a deque as a sliding
// window over data.
void processInSlidingWindow(const std::deque<int> &data,
                            size_t windowSize) {
  for (size_t i = 0; i <= data.size() - windowSize; ++i) {
    int sum = 0;
    for (size_t j = i; j < i + windowSize; ++j) {
      sum += data[j];
    }
    std::cout << "Average of window starting at index "
              << i << ": "
              << static_cast<double>(sum) / windowSize
              << "\n";
  }
}
int main() {
  std::deque<int> numbers;
  for (int i = 1; i <= 5; ++i) {
    numbers.push_back(i * 10);   // 10, 20, ..., 50
    numbers.push_front(-i * 10); // -10, -20, ..., -50
  }
  std::cout << "Numbers in deque: ";
  for (const auto &num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  numbers.pop_front();
  numbers.pop_back();
  std::cout << "After removing front and back: ";
  for (const auto &num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  processInSlidingWindow(numbers, 3);
  std::transform(numbers.begin(), numbers.end(),
                 numbers.begin(),
                 [](int n) { return n * 2; });
  std::cout << "After doubling each element: ";
  for (const auto &num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  return 0;
}
```

在前面的示例中，我们执行以下操作：

+   我们通过向容器的开始和结束添加元素来展示 `std::deque` 的动态特性。

+   我们展示了 `pop_front()` 和 `pop_back()` 的有效操作。

+   滑动窗口函数以分块处理元素，利用 `std::deque` 的随机访问特性。

+   最后，我们使用 `std::transform` 算法来操作数据。

## 最佳实践

让我们探讨使用 `std::deque` 的最佳实践：

+   `std::deque`是其分段内存。这有时会导致与`std::vector`的连续内存布局相比，性能差异更难以预测的细微差别。

+   当涉及到内存行为时，`std::deque`与`std::vector`。这两个有不同的架构，导致在特定场景中性能各异。

+   `std::deque`在两端提供快速的插入和删除操作，但在中间不行。如果中间操作被认为是瓶颈，考虑其他容器，如`std::vector`和`std::list`。

+   `std::deque`的核心优势在于两端都提供常数时间操作。如果你主要只使用一端，`std::vector`可能提供更好的性能。即使有这个优势，也不要假设`std::deque`的性能会优于`std::vector`。你可能发现，`std::vector`的连续存储和缓存友好性甚至允许它在头部插入时超越`std::deque`。

+   `std::deque`不保证连续内存，当处理需要原始数组的 API 或库时可能会带来挑战。始终要意识到这种区别。

+   在添加元素时使用`emplace_front`和`emplace_back`。这些函数直接在 deque 中构建元素，优化内存使用和性能。

+   当前端和后端操作频繁且可接受的性能损失时，`std::deque`是合适的。其架构针对这些操作进行了优化，提供一致的性能。

+   `std::deque`，始终确保你处于其大小边界内，以防止未定义的行为。

+   `std::deque`，确保使用适当的同步机制，如互斥锁或锁，以确保数据完整性和防止竞态条件。

# std::list

`std::list`是一个双向链表。与之前的容器不同，它不连续存储其元素。这意味着你失去了缓存友好性，但获得了巨大的灵活性。只要你有指向位置的迭代器，无论位置如何，插入和删除操作都是常数时间操作。然而，访问时间是线性的，这使得它不太适合频繁进行随机访问的任务。`std::list`最适合那些数据集频繁在中间和两端进行插入和删除操作，且直接访问不是优先级的场景。

## 目的和适用性

`std::list`是 STL 提供的双向链表。其优势包括以下：

+   在任何位置实现常数时间插入和删除（虽然牺牲了缓存友好性和快速搜索）

+   在修改期间保持迭代器有效性（除非迭代器引用的元素被删除）

在以下情况下，它是最合适的选择：

+   预期会频繁地从容器的头部和中间进行插入和删除操作。

+   随机访问不是主要要求。

+   迭代器有效性保持至关重要。

+   每个节点中存储的（大）数据本身就不利于缓存。

在考虑不同容器时，倾向于使用 `std::list` 以获得链表的优点。如果随机访问至关重要，`std::vector` 或 `std::deque` 可能是更好的选择。

## 理想用例

以下是一些 `std::list` 的理想用例：

+   `std::forward_list` 和 `std::list` 提供双向遍历能力，允许你向前和向后迭代元素，这对某些算法是有益的。

+   如果你有要插入的位置的迭代器，`std::list` 提供任何位置的常数时间插入和删除。这使得它适合于这种操作频繁的应用程序。然而，需要注意的是，搜索插入位置的成本可能会超过插入操作本身的收益。通常，即使对于频繁的插入和删除，`std::vector` 也可能优于 `std::list`。

+   `std::list` 可以是一个好的选择。

+   `std::list` 由于其拼接能力而证明是高效的。

+   `std::queue` 是标准选择，`std::list` 可以因其双向特性被用来实现双端队列（deque）。

+   `std::list` 适合在软件应用程序中维护撤销和重做历史。

+   由于在元素间移动的高效性，`std::list` 常常是一个好的选择。

+   `std::list` 可以调整以创建循环列表，其中最后一个元素链接回第一个。

+   可以使用 `std::list`。

虽然 `std::list` 多功能，但应谨慎其局限性。它不支持直接访问或索引，与数组或 `std::vector` 不同。因此，当其特定优势与应用程序的需求很好地匹配时，选择 `std::list` 是至关重要的。它也不利于缓存，并且线性搜索的成本较高。

## 性能

`std::list` 的算法性能如下：

+   **插入**：在任何位置的时间复杂度为 *O(1)*。

+   **删除**：对于已知位置的时间复杂度为 *O(1)*。

+   **访问时间**：由于其链式结构，为 *O(n)*。

+   **内存开销**：通常高于向量，因为存储了下一个和上一个指针。

+   对于大多数用例，`std::vector` 将优于 `std::list`。

## 内存管理

与 `std::vector` 不同，`std::list` 不会大量重新分配。每个元素的分配是独立的。分配器仍然可以影响单个节点的分配，从而提供更具体的内存管理。

## 线程安全

并发读取是安全的。然而，修改或同时读取和写入需要外部同步。可以使用互斥锁或类似的结构。

## 扩展和变体

`std::forward_list` 是单链表的变体，优化了空间但失去了向后遍历的能力。

## 排序和搜索复杂度

排序和搜索的复杂度如下：

+   `std::list::sort()`，通常为 *O(n log n)*

+   由于缺乏随机访问，`std::find()`

## 接口和成员函数

值得注意的函数如下：

+   `emplace_front`/`emplace_back`：直接就地构造

+   `splice`：将元素从一个列表转移到另一个列表

+   `merge`：合并两个排序后的列表

+   `unique`：删除重复元素

## 比较操作

当与`std::vector`或`std::deque`相比时，`std::list`似乎在频繁的中部插入和删除方面更优越。然而，它并不提供前者容器那样的快速随机访问。这意味着找到执行插入或删除的位置的成本超过了插入或删除本身的利益。

## 与算法的交互

虽然`std::list`可以与许多 STL 算法一起工作，但那些需要随机访问的（例如，`std::random_shuffle`）并不理想。

## 异常

越界或非法操作可能会抛出异常。然而，`std::list`的许多操作提供了强大的异常安全性，确保列表保持一致性。

## 定制化

可以使用自定义分配器来影响节点内存分配。与`std::set`或`std::map`等容器不同，自定义比较器在`std::list`中并不常见。

## 示例

在这个例子中，我们将展示以下最佳实践和`std::list`的使用：

+   利用`std::list`的双向特性来遍历和修改元素，无论是正向还是反向方向

+   在列表的任何位置高效地插入和删除元素

+   使用`std::list`的成员函数，如`sort()`、`merge()`、`splice()`和`remove_if()`

+   应用外部 STL 算法，如`std::find`

这里是代码示例：

```cpp
#include <algorithm>
#include <iostream>
#include <list>
void display(const std::list<int> &lst) {
  for (const auto &val : lst) { std::cout << val << " "; }
  std::cout << "\n";
}
int main() {
  std::list<int> numbers = {5, 1, 8, 3, 7};
  std::cout << "Numbers in reverse: ";
  for (auto it = numbers.rbegin(); it != numbers.rend();
       ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  auto pos = std::find(numbers.begin(), numbers.end(), 8);
  numbers.insert(pos, 2112);
  std::cout << "After insertion: ";
  display(numbers);
  numbers.sort();
  std::list<int> more_numbers = {2, 6, 4};
  more_numbers.sort();
  numbers.merge(more_numbers);
  std::cout << "After sorting and merging: ";
  display(numbers);
  std::list<int> additional_numbers = {99, 100, 101};
  numbers.splice(numbers.end(), additional_numbers);
  std::cout << "After splicing: ";
  display(numbers);
  numbers.remove_if([](int n) { return n % 2 == 0; });
  std::cout << "After removing all even numbers: ";
  display(numbers);
  return 0;
}
```

在这个例子中，我们将执行以下操作：

+   我们使用反向迭代器反向遍历`std::list`。

+   我们展示了在期望位置高效插入元素的能力。

+   我们展示了使用`std::list`特定操作，如`sort()`、`merge()`和`splice()`的用法。

+   最后，我们使用 lambda 与`remove_if()`条件性地从列表中删除元素。

这个例子展示了`std::list`的各种功能，包括特别高效的容器操作和使用其双向特性的操作。

## 最佳实践

让我们探索使用`std::list`的一些最佳实践：

+   在没有针对数据类型如`std::vector`进行性能分析并发现可测量的性能改进之前，不要使用`std::list`。

+   `std::list`本身提供的`sort()`成员函数是必不可少的，而不是求助于`std::sort`。这是因为 C 需要随机访问迭代器，而`std::list`不支持。

+   由于`std::list`的双链结构，它不提供*O(1)*随机访问。对于频繁的随机访问，容器如`std::vector`或`std::deque`可能更合适。

+   `std::list`意味着它为每个元素维护两个指针。这使它能够进行双向遍历，但这也带来了内存成本。如果内存使用至关重要且不需要双向遍历，`std::forward_list`提供了一个更干净的替代方案。

+   `std::list` 可以将操作从 *O(n)* 转换为 *O(1)*。利用迭代器的力量进行更有效的插入和删除。

+   `std::list` 提供了使用 `splice` 函数在常数时间内在不同列表之间传输元素的独特能力。这个操作既高效又可简化列表操作。

+   `emplace_front` 和 `emplace_back`，你可以在原地构建元素，从而消除对临时对象的需求，并可能加快你的代码。

+   `std::list`。特别是在内存敏感的场景中，了解这种开销对于做出明智的容器选择至关重要。

# std::forward_list

`std::forward_list` 是一个单链表。它与 `std::list` 类似，但每个元素只指向下一个元素，而不是前一个。这比 `std::list` 减少了内存开销，但以双向迭代为代价。当你需要一个列表结构但不需要向后遍历且希望节省内存开销时，选择 `std::forward_list`。

## 目的和适用性

`std::forward_list` 是 STL 中的一个单链表容器。它的主要吸引力在于以下方面：

+   在列表的任何位置进行高效的插入和删除

+   比使用 `std::list` 消耗更少的内存，因为它不存储前一个指针

它在以下情况下特别适用：

+   你需要无论位置如何都能进行常数时间的插入或删除。

+   内存开销是一个需要关注的问题。

+   不需要双向迭代。

虽然 `std::vector` 在随机访问方面表现出色，但如果你更重视插入和删除效率，则转向 `std::forward_list`。

## 理想用例

以下是一些 `std::forward_list` 的理想用例：

+   `std::forward_list` 使用单链表，由于它只需要维护一个方向上的链接，所以比双链表的开销更小。这使得它在空间节约是首要考虑的场景中非常适用。

+   `std::forward_list` 提供了最优效率。

+   `std::forward_list` 可以是一个合适的选择。

+   `using std::forward_list` 确保单向移动。

+   `std::forward_list` 可以用来设计类似栈的行为。

+   `std::forward_list` 可以存储这些边。

+   `std::forward_list` 提供了必要的结构。

+   `std::forward_list`.

重要的是要理解，虽然 `std::forward_list` 在特定用例中提供了优势，但它缺乏其他容器提供的某些功能，例如在 `std::list` 中看到的双向遍历。当其优势与应用程序的需求相匹配时，选择 `std::forward_list` 是合适的。

## 性能

`std::forward_list` 的算法性能如下：

+   **插入**：无论位置如何都是 *O(1)*。

+   **删除**：任何位置都是 *O(1)*。

+   **访问**：*O(n)*，因为只有顺序访问是唯一的选择。

+   **内存开销**：最小化，因为只存储了下一个指针。

+   `std::list`通常由于其缓存不友好和与`std::vector`相比的慢速搜索性能而不足。一般来说，`std::vector`在大多数用例中会比`std::forward_list`表现更好。

## 内存管理

元素插入时分配内存。每个节点存储元素和指向下一个节点的指针。自定义分配器可以调整这种分配策略。

## 线程安全

并发读取是安全的。然而，写入或读取和写入的组合需要外部同步。

## 扩展和变体

对于希望获得双向迭代能力的人来说，`std::list`（一个双链表）是一个可行的替代方案。

## 排序和搜索复杂度

排序和搜索的复杂度如下：

+   `std::sort()`

+   **搜索**：*O(n)*，因为没有随机访问

## 特殊接口和成员函数

值得注意的成员函数如下：

+   `emplace_front`：用于直接构造元素

+   `remove`：通过值移除元素

+   `splice_after:` 用于从另一个`std::forward_list`转移元素

记住，`std::forward_list`中没有`size()`或`push_back()`函数。

## 比较操作

与`std::list`相比，`std::forward_list`使用更少的内存，但不支持双向迭代。与`std::vector`相比，它不允许随机访问，但确保了一致的插入和删除时间。

## 与算法的交互

由于其单向性质，`std::forward_list`可能不适合需要双向或随机访问迭代器的算法。

## 异常

在内存分配失败期间可能会出现异常。大多数对`std::forward_list`的操作都提供强大的异常安全保证。

## 自定义

您可以使用自定义分配器调整内存分配策略。`std::forward_list`本身不支持自定义比较器或哈希函数。

## 示例

`std::forward_list`是一个单链表，在从前端进行插入/删除操作时特别高效。它比`std::list`消耗更少的内存，因为它不存储每个元素的向后指针。

`std::forward_list`的一个常见用途是实现具有链表的哈希表以解决冲突。以下是一个使用`std::forward_list`的基本链式哈希表版本：

```cpp
#include <forward_list>
#include <iostream>
#include <vector>
template <typename KeyType, typename ValueType>
class ChainedHashTable {
public:
  ChainedHashTable(size_t capacity) : capacity(capacity) {
    table.resize(capacity);
  }
  bool get(const KeyType &key, ValueType &value) const {
    const auto &list = table[hash(key)];
    for (const auto &bucket : list) {
      if (bucket.key == key) {
        value = bucket.value;
        return true;
      }
    }
    return false;
  }
  void put(const KeyType &key, const ValueType &value) {
    auto &list = table[hash(key)];
    for (auto &bucket : list) {
      if (bucket.key == key) {
        bucket.value = value;
        return;
      }
    }
    list.emplace_front(key, value);
  }
  bool remove(const KeyType &key) {
    auto &list = table[hash(key)];
    return list.remove_if(& {
      return bucket.key == key;
    });
  }
private:
  struct Bucket {
    KeyType key;
    ValueType value;
    Bucket(KeyType k, ValueType v) : key(k), value(v) {}
  };
  std::vector<std::forward_list<Bucket>> table;
  size_t capacity;
  size_t hash(const KeyType &key) const {
    return std::hash<KeyType>{}(key) % capacity;
  }
};
int main() {
  ChainedHashTable<std::string, int> hashTable(10);
  hashTable.put("apple", 10);
  hashTable.put("banana", 20);
  hashTable.put("cherry", 30);
  int value;
  if (hashTable.get("apple", value)) {
    std::cout << "apple: " << value << "\n";
  }
  if (hashTable.get("banana", value)) {
    std::cout << "banana: " << value << "\n";
  }
  hashTable.remove("banana");
  if (!hashTable.get("banana", value)) {
    std::cout << "banana not found!\n";
  }
  return 0;
}
```

在这个例子中，我们做以下操作：

+   哈希表由一个名为`table`的`std::vector`组成，其中包含`std::forward_list`。向量的每个槽位对应一个哈希值，并且可能包含多个与该哈希值冲突的键（在一个`forward_list`中）。

+   在这个上下文中，`forward_list`的`emplace_front`函数特别有用，因为我们可以在常数时间内向列表的前端添加新的键值对。

+   我们使用`forward_list::remove_if`来移除键值对，它遍历列表并移除第一个匹配的键。

## 最佳实践

让我们探索使用`std::forward_list`的最佳实践：

+   在没有对代码与 `std::vector` 等数据类型进行性能分析并发现可测量的性能改进之前，不要使用 `std::forward_list`。

+   `std::forward_list` 是一个针对单链表世界中的特定场景进行优化的专用容器。理解其优势和局限性对于有效地使用它至关重要。

+   `std::forward_list` 是一个不错的选择。然而，它缺乏对元素的快速直接访问，需要 *O(n)* 操作。

+   `std::forward_list` 仅支持正向迭代。如果需要双向遍历，请考虑其他容器，例如 `std::list`。

+   **无随机访问**：此容器不适用于需要快速随机访问元素的场景。

+   `size()` 成员函数意味着确定列表的大小需要 *O(n)* 操作。为了快速检查列表是否为空，请使用 `empty()` 函数，它效率很高。

+   `std::forward_list` 提供高效的插入和删除。特别是，`emplace_front` 对于就地元素构造非常有用，可以减少开销。

+   使用 `sort()` 函数来维护元素顺序。要移除连续的重复元素，请应用 `unique()` 函数。

+   **对迭代器的注意事项**：在修改后，尤其是插入或删除后，务必重新检查迭代器的有效性，因为它们可能会失效。

+   在多线程应用程序中使用 `std::forward_list` 以防止数据竞争或不一致性。

+   `std::list`，由于 `std::forward_list` 只维护每个元素一个指针（前向指针），因此它在使用双向迭代不是必需时，是一个更节省内存的选择。

# `std::string`

在 STL 中，`std::string` 是一个用于管理字符序列的类。`std::string` 通过提供一系列字符串操作和分析功能简化了文本处理。尽管在正式的 C++ 标准库文档中，`std::string` 并未被归类为 *序列容器* 类别，尽管它的行为非常像。相反，它被归类为单独的 *字符串* 类别，以认可其通用的容器行为和其在文本处理方面的专用性质。

## 目的和适用性

`std::string` 代表一个动态的字符序列，本质上是对 `std::vector<char>` 的特殊化。它被设计用于以下目的：

+   操作文本数据

+   与期望字符串输入或产生字符串输出的函数交互

它在以下情况下尤其适用：

+   动态文本修改频繁。

+   希望能够高效地访问单个字符。

对于大多数字符串操作任务，请选择 `std::string`。如果您需要无所有权的字符串视图，请考虑 `std::string_view`。

## 理想用例

以下是一些 `std::string` 的理想用例：

+   **文本处理**：解析文件、处理日志或任何需要动态文本操作的其它任务

+   **用户输入/输出**: 接受用户输入；生成人类可读的输出

+   **数据序列化**: 将数据编码为字符串以进行传输/存储

## 性能

`std::string`的算法性能如下所述：

+   **插入**: 平均 *O(1)* 在末尾，其他地方为 *O(n)*

+   **删除**: 由于元素可能需要移动，因此 *O(n)*

+   **访问**: 任何位置的快速 *O(1)*

+   **内存开销**: 通常较低，但如果未使用预留容量，则可能会增长

## 内存管理

`std::string`动态分配内存。当缓冲区填满时，它会重新分配，通常加倍其大小。自定义分配器可以修改此行为。

## 线程安全

并发读取是安全的，但同时修改需要同步，通常使用互斥锁（mutexes）。

## 扩展和变体

`std::wstring`是宽字符版本，适用于某些本地化任务。`std::string_view`提供对字符串的非拥有视图，在特定场景中提高性能。还应考虑`std::u8string`、`std::u16string`和`std::u32string`。

## 排序和搜索复杂度

`std::string`的算法性能如下所述：

+   **搜索**: 线性搜索的 *O(n)*

+   对于排序序列，可以进行`std::binary_search()`。

## 特殊接口和成员函数

除了众所周知的（如`substr`、`find`和`append`）之外，熟悉以下内容：

+   `c_str()`: 返回一个 C 风格字符串（提供与以 null 终止的 C 字符串交互的功能）

+   `data()`: 直接访问底层字符数据

+   `resize()`: 调整字符串长度

+   `shrink_to_fit()`: 减少内存使用

## 比较操作

虽然`std::string`管理文本，但`std::vector<char>`可能看起来相似，但它缺乏字符串语义，例如自动空终止。

## 与算法的交互

STL 算法与`std::string`无缝工作，尽管某些算法，如排序，可能很少应用于文本内容。

## 异常

恶意访问（例如，`at()`）可能会抛出异常。操作通常是异常安全的，这意味着即使在操作抛出异常的情况下，字符串仍然有效。

## 定制化

`std::string`支持自定义分配器，但自定义比较器或哈希函数不适用。

## 示例

C++中的`std::string`是一个多用途容器，提供了一系列用于不同目的的成员函数，从文本操作到搜索和比较。以下是一个使用`std::string`的最佳实践的高级示例：

```cpp
#include <algorithm>
#include <iostream>
#include <string>
int main() {
  std::string s = "Hello, C++ World!";
  std::cout << "Size: " << s.size() << "\n";
  std::cout << "First char: " << s[0] << "\n";
  std::string greet = "Hello";
  std::string target = "World";
  std::string combined = greet + ", " + target + "!";
  std::cout << "Combined: " << combined << "\n";
  if (s.find("C++") != std::string::npos) {
    std::cout << "String contains 'C++'\n";
  }
  std::transform(
      s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return std::toupper(c); });
  std::cout << "Uppercase: " << s << "\n";
  std::transform(
      s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return std::tolower(c); });
  std::cout << "Lowercase: " << s << "\n";
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  std::cout << "Without spaces: " << s << "\n";
  std::string first = "apple";
  std::string second = "banana";
  if (first < second) {
    std::cout << first << " comes before " << second
              << "\n";
  }
  int number = 2112;
  std::string numStr = std::to_string(number);
  std::cout << "Number as string: " << numStr << "\n";
  int convertedBack = std::stoi(numStr);
  std::cout << "String back to number: " << convertedBack
            << "\n";
  return 0;
}
```

在前面的示例中，我们做了以下操作：

+   我们演示了基本的字符串操作，包括构造、访问字符和连接。

+   我们使用`find`函数检查子字符串。

+   我们使用`std::transform`与`std::toupper`和`std::tolower`将整个字符串分别转换为大写和小写。

+   我们使用`erase`结合`std::remove`从字符串中删除字符。

+   我们使用`std::string`的重载比较运算符提供的自然排序比较了两个字符串。

+   我们使用 `std::to_string` 和 `std::stoi` 函数将数字转换为字符串，反之亦然。

这些操作展示了各种 `std::string` 最佳实践及其与其他 STL 组件的无缝集成。

## 最佳实践

让我们探索使用 `std::string` 的最佳实践：

+   用于字符串连接的 `+` 操作符可能会影响性能，考虑到可能的重新分配和复制。在循环中使用 `+=` 来提高效率。

+   使用 `reserve()` 预分配足够的内存，减少重新分配并提高性能。

+   **迭代调制谨慎**：在迭代过程中修改字符串可能会给你带来惊喜。请谨慎行事，并在迭代时避免并发修改。

+   `std::string` 成员函数，如 `find()`、`replace()` 和 `substr()`。它们简化了代码，增强了可读性，并可能提高性能。

+   **受保护元素访问**：在深入字符串元素之前，验证你的索引。越界访问是通向未定义行为的单程票。

+   `std::string_view` 用于对字符串的部分或全部进行轻量级引用。当没有修改计划时，它是传统字符串切片的有效替代方案。

+   `std::string`。它是 `std::basic_string` 模板的衍生，可以满足自定义字符类型和特定字符行为的需求。

+   `std::string` 用于 ASCII 和 UTF-8 需要。你是在探索 UTF-16 或 UTF-32 领域吗？转向 `std::wstring` 及其宽字符同伴。始终对编码保持警惕，以避免潜在的数据错误。

+   **利用内部优化**：**小字符串优化**（**SSO**）是许多标准库袖子中的王牌。它允许直接在字符串对象中存储小字符串，避免动态分配。对于小字符串来说，这是一个性能上的福音。

小字符串究竟有多小？

*小字符串* 的确切长度因实现而异。然而，小字符串缓冲区的大小通常在 15 到 23 个字符之间。

+   `std::string` 的 `compare()` 函数比 `==` 操作符提供了更多的粒度。它可以提供对词法排序的见解，这对于排序操作可能是至关重要的。

+   `std::stringstream` 提供了一种灵活的方式来连接和转换字符串，但它可能伴随着开销。当性能至关重要时，优先选择直接字符串操作。

+   `std::stoi` 和 `std::to_string` 等函数。它们比手动解析更安全且通常更高效。
