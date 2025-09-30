

# 高级无序关联容器使用

当我们的有序关联容器之旅为我们提供了关系映射的技能和排序的权力时，现在是时候进入一个优先考虑速度而不是排序行为的领域：无序关联容器。正如它们的名称所暗示的，这些容器不保证它们元素的具体顺序，但它们通过可能更快的访问时间来弥补这一点。

在计算的世界里，总是有权衡。无序关联容器可能会放弃顺序的美感，但在许多场景中，它们通过速度来弥补这一点，尤其是在哈希操作最佳时。无论你是开发高频交易系统、缓存机制还是实时多人游戏后端，了解何时利用无序关联容器的力量可以有所区别。

本章提供了以下容器的参考：

+   `std::unordered_set`

+   `std::unordered_map`

+   `std::unordered_multiset`

+   `std::unordered_multimap`

# 技术要求

本章的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# `std::unordered_set`

这个容器类似于`std::set`，但有一个转折：它不保持元素在任何特定的顺序。相反，它使用哈希机制快速访问其元素。在给定一个好的哈希函数的情况下，这种基于哈希的方法可以为大多数操作提供平均常数时间复杂度。

## 目的和适用性

`std::unordered_set`是 C++ **标准模板库**（**STL**）中的一个基于哈希的容器，它以无特定顺序存储唯一元素。其核心优势包括以下内容：

+   提供插入、删除和搜索的平均常数时间操作

+   有效处理非平凡的数据类型

在以下场景中，你应该选择`std::unordered_set`：

+   当你需要快速检查元素的存在时

+   当元素顺序不是关注点时

+   当预期频繁的插入和删除时

然而，如果元素的排序至关重要，`std::set`可能是一个更好的选择。

## 理想使用场景

以下是一些`std::unordered_set`的理想使用场景：

+   `std::unordered_set`是你的候选。

+   使用`std::unordered_set`从现有数据集中创建唯一项的集合。

+   在快速插入和删除比保持顺序更重要的情况下使用`std::unordered_set`。

+   当元素的顺序不重要时，使用`std::unordered_set`比`std::set`更优，因为`std::unordered_set`提供了更快的查找、插入和删除操作。然而，`std::unordered_set`可能比`std::set`使用更多的内存。

## 性能

`std::unordered_set`的算法性能如下：

+   **插入**：平均情况 *O(1)*，最坏情况 *O(n)*，由于潜在的哈希冲突

+   **删除**：平均情况 *O(1)*，最坏情况 *O(n)*，由于潜在的哈希冲突

+   **访问**：*O(1)*

+   **内存开销**：由于哈希机制，通常高于有序容器

这里的关键权衡在于平均情况与最坏情况，特别是关于哈希冲突的问题。

## 内存管理

`std::unordered_set` 使用一系列桶来管理其内存以存储元素。桶的数量可以增长，通常在负载因子超过某个阈值时。使用自定义分配器可以帮助调整这种行为。

## 线程安全

并发读取是安全的。然而，修改集合的操作（如插入或删除）需要外部同步机制，例如互斥锁。

## 扩展和变体

`std::unordered_multiset` 是一个近亲，允许使用元素的多个实例。如果有序存储至关重要，`std::set` 和 `std::multiset` 就派上用场。

## 排序和搜索复杂度

其排序和搜索复杂度如下：

+   `std::unordered_set` 是无序的。

+   **搜索**：由于哈希，平均时间复杂度为 *O(1)*，但最坏情况可能为 *O(n)*，这取决于哈希质量。

## 特殊接口和成员函数

一些值得注意的成员函数如下：

+   `emplace`：这允许直接构造元素。

+   `bucket`：这可以检索给定元素的桶号。

+   `load_factor` 和 `max_load_factor`：这些是管理性能特征所必需的。

## 比较

与 `std::set` 相比，`std::unordered_set` 通常提供更快的操作，但失去了固有的顺序，并且可能具有更高的内存开销。

## 与算法的交互

由于其无序性，`std::unordered_set` 可能不是需要有序数据的 STL 算法的最佳候选者。然而，围绕唯一元素的算法可以很好地适应。

## 异常

如果分配失败或哈希函数抛出异常，操作可能会抛出异常。确保您的哈希函数无异常，以保证容器的异常安全性。

## 自定义

可以应用自定义哈希函数和等价谓词来微调容器针对特定数据类型的操作行为。此外，在某些场景下，自定义分配器也可能有益。

## 示例

`std::unordered_set` 以无特定顺序存储唯一元素。它支持的主要操作是插入、删除和成员检查。与使用平衡二叉树内部实现的 `std::set` 不同，`std::unordered_set` 使用哈希表，使得平均插入、删除和搜索复杂度为 *O(1)*，尽管常数较高且最坏情况性能较差。

以下代码示例展示了使用 `std::unordered_set` 的最佳实践：

```cpp
#include <iostream>
#include <unordered_set>
#include <vector>
void displaySet(const std::unordered_set<int> &set) {
  for (const int &num : set) { std::cout << num << " "; }
  std::cout << '\n';
}
int main() {
  std::unordered_set<int> numbers;
  for (int i = 0; i < 10; ++i) { numbers.insert(i); }
  displaySet(numbers);
  int searchValue = 5;
  if (numbers.find(searchValue) != numbers.end()) {
    std::cout << searchValue << " found in the set."
              << '\n';
  } else {
    std::cout << searchValue << " not found in the set."
              << '\n';
  }
  numbers.erase(5);
  displaySet(numbers);
  std::cout << "Size: " << numbers.size() << '\n';
  std::cout << "Load factor: " << numbers.load_factor()
            << '\n';
  numbers.rehash(50);
  std::cout << "Number of buckets after rehash: "
            << numbers.bucket_count() << '\n';
  std::vector<int> moreNumbers = {100, 101, 102, 103};
  numbers.insert(moreNumbers.begin(), moreNumbers.end());
  displaySet(numbers);
  return 0;
}
```

以下是一个示例输出：

```cpp
9 8 7 6 5 4 3 2 1 0
5 found in the set.
9 8 7 6 4 3 2 1 0
Size: 9
Load factor: 0.818182
Number of buckets after rehash: 53
103 102 101 100 9 8 7 6 4 3 2 1 0
```

以下是从前面代码中得出的几个关键要点：

+   `std::unordered_set`允许快速插入、删除和查找。

+   `find`可用于检查元素的存在。

+   `rehash`方法可以改变底层哈希表中的桶数，这在你事先知道元素数量并希望减少重新散列开销时可能有所帮助。

+   总是小心地考虑负载因子（在下面的最佳实践部分有介绍）并在必要时考虑重新散列以保持高效性能。

+   记住，`std::unordered_set`中元素的顺序是不保证的。随着元素的插入或删除，顺序可能会随时间改变。

当你需要快速查找且不关心元素顺序时，使用`std::unordered_set`是合适的。如果顺序是必需的，你可能想考虑使用`std::set`。

## 最佳实践

让我们探讨使用`std::unordered_set`的最佳实践：

+   `std::unordered_set`设计时无需维护其元素的具体顺序。不要依赖于此容器内的任何顺序一致性。

+   **哈希冲突意识**：哈希冲突会损害性能，将平均情况下的常数时间操作转换为最坏情况下的线性时间操作。始终对此保持警觉，尤其是在设计哈希函数或处理大数据集时。

+   `std::unordered_set`与其桶数和负载因子紧密相关。考虑`std::unordered_set`的负载因子和重新散列策略以进行性能调整。`bucket_count()`: 当前桶数

+   `load_factor()`: 当前元素数量除以桶数

+   `max_load_factor()`: 负载因子阈值，当超过此阈值时触发重新散列

+   `std::hash` 标准模板特化。这允许对哈希行为进行微调。*   在必要时进行 `rehash()` 或 `reserve()`。这可以帮助防止性能意外下降，尤其是在插入新元素时。*   **均匀的哈希分布**：一个好的哈希函数将在桶之间均匀分布值，最小化冲突的可能性。在将哈希函数部署到性能关键应用之前，通过测试其分布来确保您的哈希函数实现这一点。使用设计良好的哈希函数，将元素均匀分布到桶中，以避免性能下降。*   `std::unordered_set` 不是理想的选择。考虑迁移到 `std::set` 或利用 STL 中的其他有序容器。*   在多线程应用程序中，`std::unordered_set` 确保适当的位置同步机制。并发读取是安全的，但写入或同时读取和写入需要外部同步。*   `std::unordered_set` 动态管理其大小，如果您对要存储的元素数量有一个估计，则使用 `reserve()` 等函数是有益的。这有助于减少重新哈希的次数并提高性能。*   适度使用 `erase` 成员函数。记住，通过迭代器删除比通过键值删除更快（最坏情况下为 *O(1)*，而通过键值删除为 *O(n)*）。*   由于 `std::unordered_set` 的哈希机制，它可能比其他容器具有更高的内存开销。在内存敏感的应用程序中，请考虑这一点。

# std::unordered_map

将这个容器视为 `std::map` 的无序版本。它将键与值关联起来，但不强加任何顺序。相反，它依赖于哈希以实现快速操作。

## 目的和适用性

`std::unordered_map` 是 STL 中基于哈希表的键值容器。其核心优势如下：

+   快速的平均情况基于键的访问、插入和删除

+   维护键值关联的能力

在这种情况下，这个容器是首选：

+   当插入、删除和查找必须平均快速时

+   当元素顺序不是关注点时

## 理想用例

以下是一些 `std::unordered_map` 的理想用例：

+   `std::unordered_map` 提供了平均常数时间复杂度用于 `search`、`insert` 和 `delete` 操作

+   `std::unordered_map` 是理想的

+   `std::unordered_map` 允许您有效地将项目映射到它们的出现次数

+   `std::unordered_map` 可以将属性映射到对象列表或集合

+   `std::unordered_map` 可以将设置键与其当前值关联起来，以便快速查找和修改

+   `std::unordered_map` 可以作为基于唯一标识符快速记录访问的索引

+   `std::unordered_map` 提供了一种有效的方法来根据唯一键更新和访问数据类别或计数器

+   `std::unordered_map` 提供了一种高效的结构来处理键值对

+   `std::unordered_map` 非常宝贵

+   `std::unordered_map` 可以将资源键与其状态或属性关联起来

总结来说，`std::unordered_map` 对于需要快速关联查找、插入和删除，且不需要键保持任何特定顺序的场景是最佳的。如果键的序列或排序性质是优先考虑的，那么像 `std::map` 这样的结构会更合适。

## 性能

`std::unordered_map` 的算法性能如下所述：

+   **插入**：平均情况下的 *O(1)*，最坏情况下的 *O(n)*

+   **删除**：平均情况下的 *O(1)*，最坏情况下的 *O(n)*

+   **访问**：平均情况下的 *O(1)*，由于潜在的哈希冲突，最坏情况下的 *O(n)*

+   **内存开销**：由于哈希基础设施，通常高于有序映射的对应项

## 内存管理

`std::unordered_map` 自动管理其内存，当负载因子超过某些阈值时进行扩容。分配器可以提供对此过程的更精细控制。

## 线程安全

并发读取是安全的。然而，修改或混合读写需要外部同步，例如使用互斥锁。

## 扩展和变体

`std::map` 是有序的对应项，以维护顺序为代价提供 log(n) 的保证。根据您的需求，决定您是否需要顺序或平均情况的速度。

## 排序和搜索复杂度

其排序和搜索复杂度如下所述：

+   `std::unordered_map` 本质上是无序的

+   **搜索**：基于键的快速 *O(1)* 平均情况查找

## 特殊接口和成员函数

除了标准函数（`insert`、`erase`、`find`）之外，熟悉以下内容：

+   `emplace`：就地构建键值对

+   `bucket_count`：返回桶的数量

+   `load_factor`：提供当前的负载因子

## 比较项

与 `std::map` 相比，`std::unordered_map` 以牺牲顺序为代价换取了更快的平均情况操作。无序版本在常数顺序不是关键的场景中通常表现更好。

## 与算法的交互

大多数与序列一起工作的 STL 算法不能直接应用于键值映射结构。尽管如此，容器提供了针对其用例优化的方法。

## 异常

内存分配或哈希函数的失败可以抛出异常。一些操作，如 `at()`，可以抛出 `std::out_of_range`。确保异常安全性至关重要，尤其是在插入或就地构造时。

## 自定义

您可以提供自定义哈希函数和键相等函数以进一步优化或调整行为。此外，还提供了自定义分配器以进行内存管理调整。

## 示例

`std::unordered_map` 是一个将键与值关联的容器。它与 `std::map` 类似，但 `std::map` 按照键的顺序维护其元素，而 `std::unordered_map` 不维护任何顺序。内部，它使用哈希表，这使得插入、删除和查找具有 *O(1)* 的复杂度。

以下代码示例展示了使用`std::unordered_map`时的最佳实践：

```cpp
#include <iostream>
#include <unordered_map>
void displayMap(
    const std::unordered_map<std::string, int> &map) {
  for (const auto &[key, value] : map) {
    std::cout << key << ": " << value << '\n';
  }
}
int main() {
  std::unordered_map<std::string, int> ageMap;
  ageMap[„Lisa"] = 28;
  ageMap[„Corbin"] = 25;
  ageMap[„Aaron"] = 30;
  std::cout << "Corbin's age: " << ageMap["Corbin"]
            << '\n';
  if (ageMap.find("Daisy") == ageMap.end()) {
    std::cout << "Daisy not found in the map." << '\n';
  } else {
    std::cout << "Daisy's age: " << ageMap["Daisy"]
              << '\n';
  }
  ageMap["Lisa"] = 29;
  std::cout << "Lisa's updated age: " << ageMap["Lisa"]
            << '\n';
  displayMap(ageMap);
  std::cout << "Load factor: " << ageMap.load_factor()
            << '\n';
  std::cout << "Bucket count: " << ageMap.bucket_count()
            << '\n';
  ageMap.rehash(50);
  std::cout << "Bucket count after rehash:"
            << ageMap.bucket_count() << '\n';
  // Remove an entry
  ageMap.erase("Aaron");
  displayMap(ageMap);
  return 0;
}
```

这里是示例输出：

```cpp
Corbin's age: 25
Daisy not found in the map.
Lisa's updated age: 29
Aaron: 30
Corbin: 25
Lisa: 29
Load factor: 0.6
Bucket count: 5
Bucket count after rehash:53
Corbin: 25
Lisa: 29
```

以下是从前面的代码中得出的关键要点：

+   使用`operator[]`或`insert`方法向映射中添加元素。请注意，在不存在键上使用索引操作符将创建它并使用默认值。

+   `find`方法检查键的存在。当你想检查键的存在而不进行潜在插入时，它比使用`index`操作符更有效。

+   总是注意地图的负载因子，并在必要时考虑重新散列以保持高效性能。

+   与`std::unordered_set`一样，`std::unordered_map`中元素的顺序是不保证的。随着元素的插入或删除，顺序可能会改变。

当你需要快速基于键的访问且不关心元素顺序时，`std::unordered_map`是合适的。如果顺序很重要，那么`std::map`将是一个更合适的选择。

## 最佳实践

让我们探讨使用`std::unordered_map`的最佳实践：

+   **元素顺序不保证**：不要假设映射保持元素顺序。

+   **注意哈希冲突**：确保在哈希冲突场景中考虑潜在的糟糕性能。

+   使用`std::unordered_map`以保持最佳性能。定期检查负载因子，并在必要时考虑重新散列。

+   `std::unordered_map`高度依赖于所使用的哈希函数的有效性。一个设计不良的哈希函数可能导致性能不佳，因为缓存未命中和冲突解决开销。

+   使用`std::unordered_map`来提高内存效率，尤其是在插入和删除频繁的场景中。

+   **检查现有键**：在插入之前始终检查现有键以避免覆盖。

+   使用`emplace`就地构建条目，减少开销。

+   使用`operator[]`访问元素时，`std::unordered_map`成本较高，这可能是性能陷阱。

# `std::unordered_multiset`

此容器是`std::unordered_set`的灵活对应物，允许元素出现多次。它结合了散列的速度和非唯一元素的自由度。

## 目的和适用性

`std::unordered_multiset`是一个基于哈希表的容器，允许你以无序方式存储多个等效项。其主要吸引力如下：

+   快速的平均情况插入和查找时间

+   存储具有相同值的多个项的能力

它特别适合以下场景：

+   当元素的顺序不重要时

+   当你预期有多个具有相同值的元素

+   当你希望插入和查找的平均情况时间复杂度为常数时

当你在搜索允许重复且顺序不重要的容器时，`std::unordered_multiset`是一个有力的选择。

## 理想用例

以下是一些`std::unordered_multiset`的理想用例：

+   `std::unordered_multiset`是合适的。它允许存储多个相同的元素。

+   `std::unordered_multiset`可以是一个有效的结构，其中每个唯一值都与其重复项一起存储。

+   `std::unordered_multiset`可以通过存储碰撞项来有效地管理哈希冲突。

+   `std::unordered_multiset`可以存储这些重复模式以供进一步分析。

+   `std::unordered_multiset`效率高，因为它允许插入的平均复杂度为常数时间。

+   `std::unordered_multiset`可以有效地管理这些标签出现。

+   `std::unordered_multiset`提供了一种管理这些分组项的方法。

+   `std::unordered_multiset`可以作为一个高效的内存工具来管理这些冗余数据点。

`std::unordered_multiset`最适合需要快速插入和查找、允许重复元素且元素顺序不重要的场景。当需要唯一键或有序数据结构时，其他容器，如`std::unordered_set`或`std::map`可能更合适。

## 性能

`std::unordered_multiset`的算法性能如下：

+   **插入**：平均情况下为*O(1)*，但最坏情况下可以是*O(n)*

+   **删除**：平均情况下为*O(1)*

+   **访问**：没有像数组那样的直接访问，但查找元素的平均情况下为*O(1)*

+   **内存开销**：通常，由于哈希机制，这比有序容器要高

一个权衡是，虽然`std::unordered_multiset`在平均情况下提供*O(1)*的插入、查找和删除性能，但在最坏情况下性能可能会下降到*O(n)*。

## 内存管理

`std::unordered_multiset`动态管理其桶列表。容器可以调整大小，这可能在元素插入且大小超过`max_load_factor`时自动发生。可以使用分配器来影响内存分配。

## 线程安全

从容器中读取是线程安全的，但修改（例如，插入或删除）需要外部同步。多个线程同时写入`std::unordered_multiset`可能导致竞争条件。

## 扩展和变体

`std::unordered_set`功能类似，但不允许重复元素。它与`std::multiset`形成对比，后者保持其元素有序但允许重复。

## 排序和搜索复杂度

它的排序和搜索复杂度如下：

+   **排序**：不是天生有序的，但你可以将元素复制到向量中并对其进行排序

+   **搜索**：由于哈希，查找的平均复杂度为*O(1)*

## 特殊接口和成员函数

虽然它提供了标准函数（`insert`、`erase`、`find`），但你也可以探索以下内容：

+   `count`：返回与特定值匹配的元素数量

+   `bucket`：返回给定值的桶号

+   `max_load_factor`：管理容器决定何时进行大小调整

## 比较操作

与 `std::multiset` 相比，这个容器提供了更快的平均性能，但牺牲了顺序和可能更高的内存使用。

## 与算法的交互

基于哈希的容器，如 `std::unordered_multiset`，并不总是像针对有序容器优化过的 STL 算法那样受益。不依赖于元素顺序的算法更可取（即 `std::for_each`、`std::count`、`std::all_of`、`std::transform` 等）。

## 异常

对于不良分配可能会抛出标准异常。重要的是要知道对 `std::unordered_multiset` 的操作提供了强大的异常安全性。

## 定制化

容器支持自定义分配器和哈希函数，允许对内存分配和哈希行为进行精细控制。

## 示例

`std::unordered_multiset` 与 `std::unordered_set` 类似，但允许相同元素的多重出现。与其他无序容器一样，它内部使用哈希表，因此不维护元素的任何顺序。`unordered_multiset` 的关键特性是其存储重复元素的能力，这在某些应用中可能很有用，例如根据某些标准对项目进行计数或分类。

以下示例演示了使用 `std::unordered_multiset` 时的几个最佳实践：

```cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_set>
int main() {
  std::unordered_multiset<std::string> fruits;
  fruits.insert("apple");
  fruits.insert("banana");
  fruits.insert("apple");
  fruits.insert("orange");
  fruits.insert("apple");
  fruits.insert("mango");
  fruits.insert("banana");
  const auto appleCount = fruits.count("apple");
  std::cout << "Number of apples: " << appleCount << '\n';
  auto found = fruits.find("orange");
  if (found != fruits.end()) {
    std::cout << "Found: " << *found << '\n';
  } else {
    std::cout << "Orange not found!" << '\n';
  }
  auto range = fruits.equal_range("banana");
  for (auto itr = range.first; itr != range.second;
       ++itr) {
    std::cout << *itr << " ";
  }
  std::cout << '\n';
  fruits.erase("apple");
  std::cout << "Number of apples after erase:"
            << fruits.count("apple") << '\n';
  std::cout << "Load factor: " << fruits.load_factor()
            << '\n';
  std::cout << "Bucket count: " << fruits.bucket_count()
            << '\n';
  fruits.rehash(50);
  std::cout << "Bucket count after rehashing: "
            << fruits.bucket_count() << '\n';
  for (const auto &fruit : fruits) {
    std::cout << fruit << " ";
  }
  std::cout << '\n';
  return 0;
}
```

这是示例输出：

```cpp
Number of apples: 3
Found: orange
banana banana
Number of apples after erase:0
Load factor: 0.363636
Bucket count: 11
Bucket count after rehashing: 53
mango banana banana orange
```

以下是从前面代码中得出的几个关键要点：

+   `std::unordered_multiset` 可以存储重复值。使用 `count` 方法检查容器中给定元素出现的次数。

+   `equal_range` 函数提供了一个迭代器范围，指向特定元素的所有实例。

+   与其他无序容器一样，要意识到负载因子，并在必要时考虑重新哈希。

+   记住，`unordered_multiset` 中的元素是无序的。如果您需要有序数据且允许重复值，应使用 `std::multiset`。

+   您需要遍历集合并使用基于迭代器的 `erase()` 方法来删除特定重复值的实例。在前面的示例中，我们为了简单起见移除了所有 `apple` 的实例。

使用 `std::unordered_multiset` 来跟踪顺序不重要且允许重复的元素。它为插入、删除和查找提供了高效的平均常数时间复杂度。

## 最佳实践

让我们探讨使用 `std::unordered_multiset` 的最佳实践：

+   `std::unordered_multiset` 和 `std::unordered_set`。与 `std::unordered_set` 不同，`std::unordered_multiset` 允许重复。如果您的应用程序必须存储多个等效键，请选择 `std::unordered_multiset`。

+   `std::unordered_multiset` 的能力在于处理重复元素。这在需要跟踪元素多个实例的场景中特别有用。然而，这也意味着像 `find()` 这样的操作将返回元素第一个实例的迭代器，并且对于某些操作可能需要遍历所有重复项。

+   `std::unordered_set`，`std::unordered_multiset` 的性能会受到负载因子的影响。较高的负载因子可能导致更多的哈希冲突，影响性能。相反，较低的负载因子虽然减少了冲突，但可能导致内存效率低下。使用 `load_factor()` 来监控，并使用 `rehash()` 或 `max_load_factor()` 来有效地管理负载因子。

+   使用 `std::unordered_multiset` 进行高效的元素分布，尤其是在处理自定义或复杂数据类型时。使用 `std::hash` 模板特化实现专门的哈希函数，以确保均匀分布并最小化冲突。

+   由于其无序性，`std::unordered_multiset` 可能不是最佳选择。在这种情况下，考虑使用 `std::multiset`，它维护顺序但仍然允许重复。

+   使用 `erase()` 函数来删除元素。通过迭代器删除元素是一个 *O(1)* 操作，而通过值删除在最坏情况下可能需要 *O(n)*。在设计删除策略时请注意这一点，尤其是在性能关键的应用中。

+   `std::unordered_set`，`std::unordered_multiset` 由于其哈希机制，可能会有更高的内存开销。在内存受限的环境中，这应该是一个考虑因素。

+   `std::unordered_multiset` 支持并发读取，但写入或并发读取和写入需要外部同步机制。这在多线程环境中至关重要，以避免数据竞争并保持数据完整性。

+   `std::unordered_multiset`，请注意那些期望有序范围的算法，因为它们不适合无序容器。始终确保所选算法与 `std::unordered_multiset` 的特性相匹配。

# std::unordered_multimap

通过结合 `std::unordered_map` 的原则和多重性的灵活性，这个容器允许单个键与多个值相关联，而无需维护特定的顺序。

## 目的和适用性

`std::unordered_multimap` 是一种基于哈希的容器，允许单个键与多个值相关联。与 `std::unordered_map` 不同，它不强制唯一键。它特别适用于以下场景：

+   当需要快速的平均情况查找时间时

+   当你预期同一个键会有多个值时

+   当键的顺序不重要时，因为元素没有存储在任何特定的顺序

在需要非唯一键和快速查找的情况下选择 `std::unordered_multimap`。如果顺序或唯一键很重要，请考虑其他选项。

## 理想用例

以下是一些 `std::unordered_multimap` 的理想用例：

+   `std::unordered_multimap` 是一个合适的容器。例如，一个作者（键）可以在作者及其书籍（值）的数据库中拥有多本书。

+   `std::unordered_multimap` 是有益的。

+   `std::unordered_multimap` 可以通过将冲突键链接到它们相应的值来管理哈希冲突。

+   `std::unordered_multimap` 可以组织这些标签到项目或项目到标签的关系。

+   `std::unordered_multimap` 可以是一个高效的内存工具。

+   `std::unordered_multimap` 可以作为存储系统。例如，如果你按出生年份分组人员，其中一年（键）可以对应许多人（值）。

+   `std::unordered_multimap` 很有用。一个例子是颜色命名系统，其中一种颜色可以有多个相关名称。

+   `std::unordered_multimap` 可以将一个坐标与该空间中的多个对象关联起来。

`std::unordered_multimap` 是一个高度通用的工具，适用于需要快速插入和查找的应用程序，并且一个键应与多个值相关联。当需要唯一键或有序数据结构时，其他容器，如 `std::unordered_map` 或 `std::set`，可能更合适。

## 性能

`std::unordered_multimap` 的算法性能如下：

+   **插入**：平均情况 *O(1)*，最坏情况 *O(n)*

+   **删除**：平均情况 *O(1)*，最坏情况 *O(n)*

+   **访问**：平均情况 *O(1)*，最坏情况 *O(n)*

+   **内存开销**：由于哈希基础设施，适中，可能因哈希冲突而增加

它的权衡包括快速的平均情况操作，但如果哈希冲突变得普遍，可能会出现潜在的减速。

## 内存管理

当负载因子超过其最大值时，`std::unordered_multimap` 会进行大小调整。可以使用分配器来定制内存行为，包括分配和释放策略。

## 线程安全

从不同实例读取是线程安全的。然而，对同一实例的并发读取和写入需要外部同步。

## 扩展和变体

`std::unordered_map` 是一个包含唯一键的变体。如果你需要有序键行为，`std::multimap` 和 `std::map` 是基于树的替代方案。

## 排序和搜索复杂度

它的排序和搜索复杂度如下：

+   **排序**：由于其无序性，本身不可排序；必须复制到可排序的容器中

+   **搜索**：由于哈希，平均情况 *O(1)*，但在存在许多哈希冲突的情况下可能会降低

## 特殊接口和成员函数

除了常见的函数（`insert`、`find`、`erase`）之外，深入了解以下内容：

+   `emplace`：直接在容器中构建元素

+   `bucket`：获取给定键的桶号

+   `load_factor`：提供元素到桶的比率

## 比较操作

与 `std::unordered_map` 相比，这个容器允许非唯一键。如果键顺序很重要，`std::multimap` 是一个基于树的替代方案。

## 与算法的交互

由于是无序的，许多为有序序列设计的 STL 算法可能不直接适用，或者需要不同的方法。

## 异常处理

内存分配失败或哈希函数复杂性问题可能会抛出异常。容器操作提供基本的异常安全性，确保容器保持有效。

## 自定义

您可以使用自定义分配器进行内存调整。自定义哈希函数或键相等谓词也可以针对特定用例优化行为。

## 示例

`std::unordered_multimap` 与 `std::unordered_map` 类似，但允许具有等效键的多个键值对。它是一个关联容器，其值类型是通过结合其键和映射类型形成的。

以下代码示例演示了使用 `std::unordered_multimap` 的一些最佳实践：

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
int main() {
  std::unordered_multimap<std::string, int> grades;
  grades.insert({"Lisa", 85});
  grades.insert({"Corbin", 92});
  grades.insert({"Lisa", 89});
  grades.insert({"Aaron", 76});
  grades.insert({"Corbin", 88});
  grades.insert({"Regan", 91});
  size_t lisaCount = grades.count("Lisa");
  std::cout << "Number of grade entries for Lisa: "
            << lisaCount << '\n';
  auto range = grades.equal_range("Lisa");
  for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->first << " has grade: " << it->second
              << '\n';
  }
  auto lisaGrade = grades.find("Lisa");
  if (lisaGrade != grades.end()) {
    lisaGrade->second = 90; // Updating the grade
  }
  grades.erase("Corbin"); // This will erase all grade
                          // entries for Corbin
  std::cout
      << "Number of grade entries for Corbin after erase: "
      << grades.count("Corbin") << '\n';
  std::cout << "Load factor: " << grades.load_factor()
            << '\n';
  std::cout << "Bucket count: " << grades.bucket_count()
            << '\n';
  grades.rehash(50);
  std::cout << "Bucket count after rehashing: "
            << grades.bucket_count() << '\n';
  for (const auto &entry : grades) {
    std::cout << entry.first
              << " received grade: " << entry.second
              << '\n';
  }
  return 0;
}
```

下面是示例输出：

```cpp
Number of grade entries for Lisa: 2
Lisa has grade: 85
Lisa has grade: 89
Number of grade entries for Corbin after erase: 0
Load factor: 0.363636
Bucket count: 11
Bucket count after rehashing: 53
Regan received grade: 91
Aaron received grade: 76
Lisa received grade: 90
Lisa received grade: 89
```

下面是从前面的代码中得出的几个要点：

+   在 `std::unordered_multimap` 中，可以插入具有相同键的多个键值对。

+   您可以使用 `equal_range` 获取与特定键关联的所有键值对的迭代器范围。

+   `count` 方法可以帮助您确定具有特定键的键值对数量。

+   如同其他无序容器一样，您应该注意负载因子，并在必要时重新散列以实现最佳性能。

+   使用带有键的 `erase()` 方法将删除与该键关联的所有键值对。

+   由于它是一个无序容器，元素的顺序是不保证的。

+   当您需要跟踪与同一键关联的多个值且不需要对键值对进行排序时，请使用 `std::unordered_multimap`。它为大多数操作提供平均常数时间复杂度。

## 最佳实践

让我们探索使用 `std::unordered_multimap` 的最佳实践：

+   `std::unordered_multimap` 表示容器不维护其键值对的具体顺序。遍历容器不保证任何特定序列。

+   `std::unordered_multimap` 的能力是存储单个键的多个条目。在插入、删除或搜索时记住这一点，以避免意外的逻辑错误。

+   使用 `load_factor()` 函数来监控当前的负载因子。如果它变得过高，可以考虑使用 `rehash()` 函数重新散列容器。也可以使用 `max_load_factor()` 函数设置负载因子的期望上限。

+   为自定义数据类型提供 `std::hash` 模板特化，以确保高效且一致的散列。

+   **处理哈希冲突**：即使有高效的哈希函数，也可能发生冲突。容器内部处理这些冲突，但了解它们有助于做出更好的设计决策。冲突可能导致插入和搜索操作的性能下降，因此平衡负载因子和桶的数量是至关重要的。

+   在遍历与特定键关联的所有值时使用 `equal_range()`。

+   **迭代器失效**：迭代器失效可能是一个问题，尤其是在重新散列等操作之后。始终确保在可能失效之后不使用指向元素的迭代器、指针或引用。

+   `emplace` 或 `emplace_hint` 方法。这些方法允许在容器内直接构造键值对。

+   **并发考虑**: 并发读取是线程安全的，但对于任何修改或并发读取和写入，你需要外部同步。在多线程场景中使用同步原语，如互斥锁（mutexes）。

+   `std::unordered_multimap`。然而，确保选定的算法不期望排序或唯一键，因为这些假设会与容器的属性相矛盾。
