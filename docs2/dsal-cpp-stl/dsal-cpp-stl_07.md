# 7

# 高级有序关联容器使用

C++中的关联容器允许开发者以更符合现实场景的方式管理数据，例如使用键来检索值。本章将探讨有序和无序关联容器，它们的独特属性以及理想的应用环境。对于中级 C++开发者来说，理解何时使用 map 而不是 unordered_map，或者理解 set 和 multiset 之间的细微差别，对于优化性能、内存使用和数据检索速度至关重要。此外，掌握最佳实践将使开发者能够编写高效、可维护且无错误的代码，确保容器在多种应用场景中有效地发挥作用。

从本质上讲，有序关联容器，由于其严格的顺序和唯一（有时不是那么唯一）的元素，为 C++开发者提供了强大的工具。它们专为涉及关系、排序和唯一性的场景量身定制。理解它们的特性和用例是充分发挥其潜力的第一步。

本章提供了以下容器的参考：

+   `std::set`

+   `std::map`

+   `std::multiset`

+   `std::multimap`

# 技术要求

本章的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# std::set

在其核心，`std::set`容器是一个包含唯一元素的集合，其中每个元素都遵循严格的顺序。你可以将其想象为一个俱乐部，每个成员都是独特的，并且都有特定的等级。该容器确保没有两个元素是相同的，这使得它在不需要重复元素的情况下非常有用。

## 目的和适用性

`std::set`是一个关联容器，旨在存储类型为`Key`的有序唯一对象集合。其优势如下：

+   确保所有元素都是唯一的

+   在插入元素时自动排序

它特别适合以下场景：

+   当不希望有重复元素时

+   当元素的排序很重要时

+   当预期会有频繁的查找和插入操作时

## 理想用例

以下是一些`std::set`的理想用例：

+   `std::set`自然地强制执行这一点。例如，当收集唯一的学生 ID 列表或产品代码时，它将非常有用。

+   `std::set`根据比较标准保持其元素排序。当需要数据天生排序时，这很有益，例如在维护一个排行榜，其中分数会持续插入，但始终应该是有序的。

+   `std::set`提供了对查找操作的对数时间复杂度。这使得它在需要频繁成员检查的场景中非常适用——例如，检查特定用户是否是 VIP 名单的一部分。

+   `std::set` 在某些情况下非常有价值。它特别适用于你可能想要在两个集合之间找到共同元素或确定哪些元素仅属于一个集合的情况。

+   `std::set` 容器可以用来跟踪这些时间。由于其有序性，你可以迅速确定下一个事件或特定时间段是否已被预订。

值得注意的是，虽然 `std::set` 在这些任务上很擅长，但评估手头问题的具体要求至关重要。如果排序不是必需的，并且你主要需要快速插入、删除和查找而不考虑顺序，`std::unordered_set` 可能是一个更好的选择。

## 性能

`std::set` 的算法性能如下：

+   **插入**：通常为 *O(log n)*，因为平衡二叉搜索树结构

+   **删除**：单个元素为 *O(log n)*

+   **访问（查找元素）**：*O(log n)*

+   `std::vector` 由于树结构

## 内存管理

在内部，`std::set` 使用树结构，通常是平衡二叉搜索树。可以通过自定义分配器来影响内存分配。

## 线程安全

与 `std::vector` 类似，并发读取是安全的，但修改或读取和修改的组合需要外部同步。

## 扩展和变体

C++ 的 `std::multiset`（允许重复元素）和 `std::unordered_set`（一个哈希表，以牺牲无序为代价，提供平均 *O(1)* 插入/查找）。

## 排序和搜索复杂度

排序和搜索的复杂度如下：

+   **排序**：元素在插入时自动排序

+   `find` 成员函数

## 特殊接口和成员函数

一些值得注意的成员函数如下：

+   `emplace`：就地插入元素

+   `count`：返回元素数量（在集合中始终为 0 或 1）

+   `lower_bound` 和 `upper_bound`：为特定键提供边界

## 比较操作

与 `std::vector` 相比，`std::set` 在确保唯一性和保持顺序方面表现出色，但可能不是频繁随机访问或顺序不是关注点时的最佳选择。

## 与算法的交互

由于其双向迭代器，许多 STL 算法与 `std::set` 兼容。然而，需要随机访问的算法可能不太理想。

## 异常

由于 `std::set` 没有固定容量，因此不会因为容量问题而抛出异常。异常可能来自分配器在内存分配期间。

## 自定义

`std::set` 允许自定义分配器进行内存管理。你还可以提供一个自定义比较器来定义集合元素的排序方式。

## 示例

`std::set` 是一个有序关联容器，包含唯一元素。它通常用于表示一个集合，其中元素的存在比其出现的次数更重要。以下代码是一个示例，说明了使用 `std::set` 时的最佳实践：

```cpp
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
int main() {
  std::set<int> numbers = {5, 3, 8, 1, 4};
  auto [position, wasInserted] = numbers.insert(6);
  if (wasInserted) {
    std::cout << "6 was inserted into the set.\n";
  }
  auto result = numbers.insert(5);
  if (!result.second) {
    std::cout << "5 is already in the set.\n";
  }
  if (numbers.find(3) != numbers.end()) {
    std::cout << "3 is in the set.\n";
  }
  numbers.erase(1);
  std::cout << "Elements in the set:";
  for (int num : numbers) { std::cout << ' ' << num; }
  std::cout << '\n';
  std::set<int> moreNumbers = {9, 7, 2};
  numbers.merge(moreNumbers);
  std::cout << "After merging:";
  for (int num : numbers) { std::cout << ' ' << num; }
  std::cout << '\n';
  if (numbers.count(2)) {
    std::cout << "2 exists in the set.\n";
  }
  std::set<std::string, bool (*)(const std::string &,
                                 const std::string &)>
      caseInsensitiveSet{[](const std::string &lhs,
                            const std::string &rhs) {
        return std::lexicographical_compare(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            [](char a, char b) {
              return std::tolower(a) < std::tolower(b);
            });
      }};
  caseInsensitiveSet.insert("Hello");
  if (!caseInsensitiveSet.insert("hello").second) {
    std::cout << "Duplicate insertion (case-insensitive) "
                 "detected.\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
6 was inserted into the set.
5 is already in the set.
3 is in the set.
Elements in the set: 3 4 5 6 8
After merging: 2 3 4 5 6 7 8 9
2 exists in the set.
Duplicate insertion (case-insensitive) detected.
```

在此示例中，我们做了以下操作：

+   我们展示了基本的`std::set`操作，如插入、查找元素和删除。

+   我们展示了集合如何固有地排序其元素以及如何遍历它们。

+   使用`merge`函数的示例是为了将另一个集合合并到我们的主集合中。

+   使用`count`方法检查集合中是否存在元素，由于唯一性约束，这只能为 0 或 1。

+   最后，我们使用自定义比较器创建了一个不区分大小写的字符串集合。

## 最佳实践

让我们探讨使用`std::set`的最佳实践：

+   `std::set`，访问时间不是恒定的，如`std::vector`或`std::array`。由于其基于树的结构，元素检索通常需要对数时间。在设计算法时，要考虑这一点。

+   `std::set`容器是不可变的。直接通过迭代器修改它可能会破坏集合的内部顺序。如果必须修改，请删除旧元素并插入其更新版本。

+   `std::unordered_set`。由于其基于哈希的设计，它通常在性能指标上优于`std::set`，除非是极端情况。

+   使用`emplace`在集合中直接创建元素。这项技术防止了不必要的对象复制或移动。

+   **导航元素修改**：直接修改集合元素是不可行的。当你需要修改时，最佳方法是两步过程：移除原始元素并引入其修改后的副本。

+   `find`方法是确定元素是否存在于集合中的首选方法。在`std::set`的上下文中，它比`count`方法更简洁、更易于表达，因为集合具有唯一元素的性质。

+   `std::is_sorted`。

+   `std::set`本身不是线程安全的。如果预计会有多个线程并发访问，请使用同步原语（如`std::mutex`）保护集合，或者考虑使用某些 C++库提供的并发容器。

+   在使用`std::set`容器时，请记住元素是排序的。这通常可以免除在其他容器上可能应用的其他排序操作。

+   `std::vector`，`std::set`不支持`reserve`或`capacity`操作。树随着元素的添加而增长。为了效率，在删除元素时，考虑偶尔使用某些实现中可用的`shrink_to_fit`操作。

# std::map

`std::map`容器是`std::set`的兄弟，它关乎关系。它将唯一的键连接到特定的值，形成一个对。用通俗易懂的话来说，想象一个字典，每个词（键）都有一个唯一的定义（值）。

## 目的和适用性

`std::map`是一个有序关联容器，存储键值对，确保键的唯一性。其底层数据结构通常是平衡二叉树（如**红黑树**（**RBT**））。主要优点包括以下内容：

+   对数访问、插入和删除时间

+   通过键对键值对进行排序维护

在以下场景中使用`std::map`：

+   当你需要将值与唯一键关联时

+   当维护键的顺序很重要时

+   当需要频繁地进行访问、插入或删除操作，并且它们需要高效时

## 理想用例

以下是一些`std::map`的理想用例：

+   `std::map`在将唯一键与特定值关联时表现出色。例如，当将一个人的姓名（唯一键）映射到其联系详情或一个单词映射到其定义时，它非常有用。

+   `std::map`容器可以将不同的配置键与其相应的值关联起来，确保轻松检索和修改设置。

+   **学生记录系统**：教育机构可能维护一个记录系统，其中学生 ID（保证唯一）作为键，映射到包含姓名、课程、成绩和其他详细信息的全面学生档案。

+   `std::map`可以将不同的项目与其出现次数关联起来，确保高效的更新和检索。

+   用于此目的的`std::map`容器确保术语按顺序排列，并且可以高效地访问或更新。

+   `std::map`可以作为缓存，将输入值映射到其计算结果。

+   `std::map`根据其键的顺序维护其元素，适用于频繁依赖此顺序的操作的场景——例如，根据某些标准获取 *top 10* 或 *lowest 5*。

总是考虑你问题的具体需求。虽然`std::map`提供了排序和唯一键值关联，但如果不需要排序，`std::unordered_map`可能是一个更高效的替代方案，因为它的大多数操作的平均时间复杂度为常数时间。

## 性能

`std::map`的算法性能如下：

+   **插入**：*O(log n)*

+   **删除**：*O(log n)*

+   **访问**：*O(log n)* 定位键

+   **内存开销**：通常高于基于哈希的对应物，因为基于树的结构

主要的权衡是在内存开销与有序操作的效率和键值对操作的灵活性之间取得平衡。

## 内存管理

`std::map`有效地管理其内部内存，确保平衡树。然而，具体行为可能受到自定义分配器的影响，从而允许更多的控制。

## 线程安全

并发读取是安全的。然而，并发写入或混合读写需要外部同步，例如使用互斥锁。

## 扩展和变体

`std::multimap`允许每个键有多个值，而`std::unordered_map`提供了一个基于哈希表的替代方案，没有排序，但具有潜在的 *O(1)* 平均访问时间。

## 排序和搜索复杂度

排序和搜索复杂度如下：

+   `std::map`本身维护排序

+   **搜索**：*O(log n)*

## 特殊接口和成员函数

以下是一些值得注意的成员函数：

+   `emplace`：直接在原地构建键值对

+   `at`：如果键不存在，则抛出异常

+   `operator[]`：访问或为给定键创建一个值

+   `lower_bound` 和 `upper_bound`：提供指向相对于键位置的迭代器

## 比较操作

与 `std::unordered_map` 相比，`std::map` 在键顺序重要或数据集可能频繁增减的场景中表现更佳。对于需要原始性能且顺序不重要的场景，`std::unordered_map` 可能更可取。

## 与算法的交互

尽管许多 STL 算法可以与 `std::map` 一起工作，但其双向迭代器限制了它与需要随机访问的算法的兼容性。

## 异常

操作如 `at()` 可能会抛出越界异常。大多数对 `std::map` 的操作都提供了强异常安全性，确保在抛出异常时映射保持不变。

## 自定义

您可以提供自定义比较器来指定键的顺序，或使用自定义分配器来影响内存管理。

## 示例

在 `std::map` 中，键是有序且唯一的，这使得查找特定条目变得容易。以下代码是使用 `std::map` 的最佳实践的示例：

```cpp
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
int main() {
  std::map<std::string, int> ageMap = {
      {"Lisa", 25}, {"Corbin", 30}, {"Aaron", 22}};
  ageMap["Kristan"] = 28;
  ageMap.insert_or_assign("Lisa", 26);
  if (ageMap.find("Corbin") != ageMap.end()) {
    std::cout << "Corbin exists in the map.\n";
  }
  ageMap["Aaron"] += 1;
  std::cout << "Age records:\n";
  for (const auto &[name, age] : ageMap) {
    std::cout << name << ": " << age << '\n';
  }
  ageMap.erase("Corbin");
  if (ageMap.count("Regan") == 0) {
    std::cout << "Regan does not exist in the map.\n";
  }
  std::map<std::string, int,
           bool (*)(const std::string &,
                    const std::string &)>
      customOrderMap{[](const std::string &lhs,
                        const std::string &rhs) {
        return lhs > rhs; // reverse lexicographic order
      }};
  customOrderMap["Lisa"] = 25;
  customOrderMap["Corbin"] = 30;
  customOrderMap["Aaron"] = 22;
  std::cout << "Custom ordered map:\n";
  for (const auto &[name, age] : customOrderMap) {
    std::cout << name << ": " << age << '\n';
  }
  return 0;
}
```

以下是一个示例输出：

```cpp
Corbin exists in the map.
Age records:
Aaron: 23
Corbin: 30
Kristan: 28
Lisa: 26
Regan does not exist in the map.
Custom ordered map:
Lisa: 25
Corbin: 30
Aaron: 22
```

在这个例子中，我们做了以下操作：

+   我们展示了 `std::map` 的基本操作，例如插入、修改、检查键的存在以及遍历其元素。

+   我们使用了结构化绑定（C++17）在迭代时重新结构化键值对。

+   我们展示了如何使用 `count` 来检查键是否存在于映射中。

+   我们通过提供一个自定义比较器，该比较器按逆字典序对键进行排序，创建了一个自定义排序的映射。

## 最佳实践

让我们探讨使用 `std::map` 的最佳实践：

+   `std::map` 中，键在其元素的生命周期内保持不变。直接修改是不允许的。如果您需要更新键，正确的方法是删除旧的键值对，并插入一个新的具有所需键的键值对。

+   `std::unordered_map`。其基于哈希表的实现可能比 `std::map` 的红黑树提供许多操作的更快平均时间复杂度，从而减少了潜在的开销。

+   `emplace` 方法，在映射中就地构造元素，避免创建临时对象和不必要的复制。当与 `std::make_pair` 或 `std::piecewise_construct` 等工具配合使用时，它优化了插入的性能。

+   `operator[]` 方法，虽然方便，但可能是一把双刃剑。如果指定的键不存在，它将在映射中插入一个具有默认初始化值的键。当您只想查询，而不希望有潜在的插入时，请使用 `find` 方法。如果找到，`find` 方法返回指向元素的迭代器；如果没有找到，则返回 `end()` 方法。

+   `std::map` 可能并不总是适合使用场景。您可以在定义映射时提供一个比较器来自定义顺序。确保此比较器执行严格的弱排序，以保持映射内部结构的完整性。

+   `std::map` 容器，同步变得至关重要。考虑使用 `std::mutex` 或其他 STL 同步原语在写入操作期间锁定访问，以保持数据一致性。

+   `count` 方法可以直接得到计数结果。对于映射，这总是返回 `0` 或 `1`，这使得检查成员资格成为一种快速方式。

+   `erase` 操作使迭代器无效。使用 `erase` 返回的迭代器继续安全操作。

+   `std::map` 提供了范围方法，如 `equal_range`，它可以返回与给定键等效的元素子范围的上界和下界。利用它们进行高效的子范围操作。

+   `std::map` 支持自定义分配器。这允许更好地控制分配和释放过程。

# std::multiset

虽然 `std::set` 容器以其独特性而自豪，但 `std::multiset` 则更为宽容。它仍然保持顺序，但允许多个元素具有相同的值。这个容器就像一个俱乐部，成员有等级，但每个等级都有空间容纳多个成员。

## 目的和适用性

`std::multiset` 是一个关联容器，存储排序元素并允许元素有多个出现。其关键优势如下：

+   维护元素的排序顺序

+   允许重复

+   提供插入、删除和搜索的对数时间复杂度

它特别适合以下场景：

+   当需要保留重复值时

+   当你需要元素始终保持排序

+   当不需要随机访问时

## 理想用例

以下是一些 `std::multiset` 的理想用例：

+   `std::multiset` 容器有益。它允许存储重复值，同时保持它们的排序顺序。

+   由于其固有的排序特性和容纳重复数字的能力，`std::multiset` 容器非常宝贵。

+   `std::multiset` 容器可以帮助高效地管理和跟踪这些选择，尤其是在一个热门会话被多次选择时。

+   `std::multiset` 容器可以表示此类项目，允许根据需求轻松跟踪和补充。

+   `std::map`) 将术语映射到文档，一个 `std::multiset` 容器可以用来跟踪术语在多个文档中出现的频率，即使某些术语很常见且重复出现。

+   使用 `std::multiset` 来高效管理事件或点，尤其是在多个事件共享相同位置时。

+   `std::multiset` 闪耀着光芒。

记住，虽然 `std::multiset` 设计用于以排序方式处理相同值的多个实例，但如果排序属性不是必需的，并且你想跟踪多个项目，由于基于哈希的实现，结构如 `std::unordered_multiset` 在某些情况下可能更高效。

## 性能

`std::multiset` 的算法性能如下：

+   **插入**：*O(log n)*

+   **删除**：*O(log n)*

+   **访问**：元素以*O(log* *n)*的时间复杂度访问

+   **内存开销**：由于内部平衡（通常实现为平衡二叉搜索树），存在开销

## 内存管理

`std::multiset`不像`std::vector`那样动态调整大小。相反，它在插入元素时使用动态内存分配来管理节点。分配器可以影响节点内存管理。

## 线程安全

并发读取是安全的。然而，修改（插入或删除）需要外部同步。建议使用互斥锁或其他同步原语进行并发写入。

## 扩展和变体

`std::set`是一个不允许重复的直接变体。还有`std::unordered_multiset`，它为操作提供平均常数时间复杂度，但不保持顺序。

## 排序和搜索复杂度

排序和搜索复杂度如下描述：

+   **排序**：元素始终排序；因此，不需要排序操作

+   **搜索**：由于其基于树的本质，*O(log n)*

## 特殊接口和成员函数

虽然它提供了常规函数（`insert`、`erase`、`find`），但以下是一些实用的函数：

+   `count`：返回与指定键匹配的总元素数

+   `equal_range`：提供元素所有实例的范围（迭代器）

## 比较操作

与`std::set`相比，`std::multiset`允许重复，但代价是略微增加的内存。与如`std::vector`之类的序列容器相比，它保持排序顺序，但不提供常数时间访问。

## 与算法的交互

从排序数据中受益的算法（如二分搜索或集合操作）与`std::multiset`配合良好。那些需要随机访问或频繁重新排序的算法可能不合适。

## 异常

内存分配失败可能会抛出异常。大多数`std::multiset`操作提供强大的异常安全性保证。

## 定制化

在`std::multiset`中，定制包括以下内容：

+   可以使用自定义分配器来控制内存分配。

+   可以提供自定义比较器来指定元素存储的顺序。

## 示例

`std::multiset`是一个可以存储多个键的容器，包括重复键。键始终按从低到高的顺序排序。`std::multiset`通常用于需要维护排序元素集且允许重复项的情况。

以下代码是使用`std::multiset`的示例，展示了其一些独特特性和最佳实践：

```cpp
#include <iostream>
#include <iterator>
#include <set>
#include <string>
int main() {
  std::multiset<int> numbers = {5, 3, 8, 5, 3, 9, 4};
  numbers.insert(6);
  numbers.insert(5); // Inserting another duplicate
  for (int num : numbers) { std::cout << num << ' '; }
  std::cout << '\n';
  std::cout << "Number of 5s: " << numbers.count(5)
            << '\n';
  auto [begin, end] = numbers.equal_range(5);
  for (auto it = begin; it != end; ++it) {
    std::cout << *it << ' ';
  }
  std::cout << '\n';
  numbers.erase(5);
  std::multiset<std::string, std::greater<>> words = {
      "apple", "banana", "cherry", "apple"};
  for (const auto &word : words) {
    std::cout << word << ' ';
  }
  std::cout << '\n';
  std::multiset<int> dataset = {1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10};
  const auto start = dataset.lower_bound(4);
  const auto stop = dataset.upper_bound(7);
  std::copy(start, stop,
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << '\n';
  return 0;
}
```

这里是示例输出：

```cpp
3 3 4 5 5 5 6 8 9
Number of 5s: 3
5 5 5
cherry banana apple apple
4 5 6 7
```

从前面的示例中可以得出以下关键要点：

+   `std::multiset`自动排序键。

+   它可以存储重复键，并且可以利用此属性进行某些算法或存储模式，其中重复项是有意义的。

+   使用`equal_range`是查找键所有实例的最佳实践。此方法返回开始和结束迭代器，覆盖所有键的实例。

+   可以使用自定义比较器，如`std::greater<>`，来反转默认排序。

+   可以使用`lower_bound`和`upper_bound`进行高效的范围查询。

记住，如果你不需要存储重复项，那么`std::set`是一个更合适的选择。

## 最佳实践

让我们探讨使用`std::multiset`的最佳实践：

+   使用`std::multiset`来防止不必要的开销。相反，优先考虑`std::set`，它本质上管理唯一元素，可能更高效。

+   `std::multiset`不提供`std::vector`提供的相同时间复杂度的常数时间访问。由于底层基于树的数据结构，访问元素是对数复杂度。

+   使用`std::multiset`来存储重复元素可能导致内存使用增加，尤其是当这些重复项很多时。分析内存需求并确保容器适用于应用程序至关重要。

+   `std::multiset`，确保它施加严格的弱排序。任何排序不一致都可能引起未定义的行为。严格测试比较器以确认其可靠性。

+   `find`和`count`成员函数。它们提供了更高效和直接的方式来执行此类检查。

+   `std::multiset`，明确说明需要重复条目的原因。如果理由不强，或者重复项对应用程序逻辑的益处不大，考虑使用`std::set`。

# `std::multimap`

扩展`std::map`的原则，`std::multimap`容器允许一个键与多个值关联。它就像一个字典，一个词可能有几个相关定义。

## 目的和适用性

`std::multimap`是 STL 中的一个关联容器。其显著特点如下：

+   存储键值对

+   允许具有相同键的多个值

+   按键排序存储元素

它特别适合以下场景：

+   当你需要维护一个具有非唯一键的集合时

+   当你需要基于键的排序访问时

+   当键值映射至关重要时

当你预期在同一个键下有多个值时，选择`std::multimap`。如果需要唯一键，你可能需要考虑`std::map`。

## 理想使用场景

以下是一些`std::multimap`的理想使用场景：

+   在`std::multimap`中，你可以将一个键与多个值关联起来。

+   `std::multimap`容器可以有效地映射这些多个含义。

+   `std::multimap`容器可以将目的地（键）与各种航班详情或时间（值）关联起来。

+   使用`std::multimap`，你可以轻松跟踪特定日期的所有事件。

+   `std::multimap`在需要权重或其他与边关联的数据时尤其有用。

+   `std::multimap`容器。

+   `std::multimap`容器适用于此类用例。

+   `std::multimap`有助于根据标签有效地组织和检索媒体。

记住，当一对多关系普遍存在时，`std::multimap` 是一个不错的选择。然而，如果顺序和排序不是关键，且高效检索更重要，考虑到基于哈希的结构，如 `std::unordered_multimap`，可能会有所帮助。

## 性能

`std::multimap` 的算法性能如下：

+   **插入**：在大多数情况下是**对数** *O(log n)*

+   **删除**：通常情况下是**对数** *O(log n)*

+   **访问**：对于特定键是 *O(log n)*

+   **内存开销**：由于维护基于树的结构和潜在的平衡，它稍微高一点

## 内存管理

`std::multimap` 内部使用树结构，通常是红黑树（RBT）。因此，内存分配和平衡操作可能会发生。分配器可以影响其内存处理。

## 线程安全

多次读取是安全的。然而，写入或读取和写入的组合需要外部同步。使用互斥锁等工具是可取的。

## 扩展和变体

对于基于哈希表的键值映射，考虑 `std::unordered_multimap`。对于唯一键值映射，`std::map` 更为合适。

## 排序和搜索复杂度

排序和搜索的复杂度如下：

+   **排序**：排序是固有的，因为元素是按键顺序维护的

+   **搜索**：定位特定键是 *O(log n)*

## 特殊接口和成员函数

标准函数，如 `insert`、`erase` 和 `find` 都是可用的。以下也是可用的：

+   `count`：返回具有特定键的元素数量

+   `equal_range`：检索具有特定键的元素范围

## 比较操作

与 `std::unordered_multimap` 相比，`std::multimap` 提供了有序访问，但由于其基于树的本质，可能会有稍微高的开销。

## 与算法的交互

由于 `std::multimap` 维护有序访问，从排序数据中受益的算法（如 `std::set_intersection`）可能很有用。然而，请记住数据是按键排序的。

## 异常

尝试访问不存在的键或越界场景可能会抛出异常。大多数操作都是强异常安全的，确保即使在抛出异常的情况下容器仍然有效。

## 自定义

自定义分配器可以优化内存管理。`std::multimap` 还允许自定义比较器来指定键的排序。

## 示例

`std::multimap` 是一种容器，它维护一组键值对集合，其中多个键值对可以具有相同的键。`std::multimap` 中的键总是排序的。

以下代码是使用 `std::multimap` 的示例，展示了它的一些独特特性和最佳实践：

```cpp
#include <iostream>
#include <map>
#include <string>
int main() {
  std::multimap<std::string, int> grades;
  grades.insert({"John", 85});
  grades.insert({"Corbin", 78});
  grades.insert({"Regan", 92});
  grades.insert({"John", 90}); // John has another grade
  for (const auto &[name, score] : grades) {
    std::cout << name << " scored " << score << '\n';
  }
  std::cout << '\n';
  std::cout << "John's grade count:"
            << grades.count("John") << '\n';
  auto [begin, end] = grades.equal_range("John");
  for (auto it = begin; it != end; ++it) {
    std::cout << it->first << " scored " << it->second
              << '\n';
  }
  std::cout << '\n';
  grades.erase("John");
  std::multimap<std::string, int, std::greater<>>
      reverseGrades = {{"Mandy", 82},
                       {"Mandy", 87},
                       {"Aaron", 90},
                       {"Dan", 76}};
  for (const auto &[name, score] : reverseGrades) {
    std::cout << name << " scored " << score << '\n';
  }
  return 0;
}
```

这里是示例输出：

```cpp
Corbin scored 78
John scored 85
John scored 90
Regan scored 92
John's grade count:2
John scored 85
John scored 90
Mandy scored 82
Mandy scored 87
Dan scored 76
Aaron scored 90
```

从前面的代码中可以总结出以下要点：

+   `std::multimap` 自动排序键。

+   它可以存储具有相同键的多个键值对。

+   使用`equal_range`是查找键的所有实例的最佳实践。此方法返回开始和结束迭代器，覆盖所有键的实例。

+   `grades.count("John")` 高效地统计了具有指定键的键值对数量。

+   自定义比较器，如`std::greater<>`，可以将排序从默认的升序更改为降序。

当你需要一个支持重复键的字典样式的数据结构时，`std::multimap` 容器非常有用。如果不需要重复键，那么`std::map` 将是一个更合适的选择。

## 最佳实践

让我们探索使用`std::multimap`的最佳实践：

+   `std::multimap` 的访问复杂度是对数级的，这归因于其基于树的底层结构。

+   `std::multimap` 与`std::map`一样是唯一的。`std::multimap` 容器允许单个键有多个条目。如果你的应用程序需要唯一的键，那么`std::map`是适当的选择。

+   `std::multimap` 的元素基于键具有固有的排序特性。利用这一特性，尤其是在执行受益于有序数据的操作（如范围搜索或有序合并）时，可以发挥优势。

+   由于其哈希机制，`std::unordered_multimap` 可能是一个更合适的替代方案。然而，值得注意的是，最坏情况下的性能和内存开销可能会有所不同。

+   使用`find`或`count`。这有助于防止潜在的问题并确保代码的健壮性。

+   **使用自定义比较器进行定制排序**：如果您有与默认排序不同的特定排序要求，请使用自定义比较器。确保您的比较器强制执行严格的弱排序，以保证多映射的一致性和定义良好的行为。

+   `equal_range` 成员函数。它提供了一个特定键的所有元素的范围（开始和结束迭代器），使得对这些特定元素进行高效迭代成为可能。

+   `std::multimap` 在处理大量数据集时可能会变得低效，尤其是当频繁的插入和删除是常见操作时。在这种情况下，评估结构的性能并考虑替代方案或优化策略是值得的。
