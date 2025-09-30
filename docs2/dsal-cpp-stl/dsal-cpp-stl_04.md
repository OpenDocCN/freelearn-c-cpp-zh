# 4

# 使用 std::vector 掌握算法

在本章中，我们将探讨`std::vector`与 C++ **标准模板库**（**STL**）算法的交互，以释放 C++ STL 的潜力。本章阐述了高效排序、搜索和操作向量的过程，利用头文件中提供的算法。此外，关注 lambda 表达式、自定义比较器和谓词，为可定制、简洁和高效的向量操作铺平了道路。

在本章中，我们将涵盖以下主题：

+   对向量进行排序

+   搜索元素

+   操作向量

+   自定义比较器和谓词

+   理解容器不变性和迭代器失效

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 对向量进行排序

在软件中，组织数据是一个常见的需求。在 C++中，`std::vector`经常是许多人的首选容器，并且很自然地，人们会希望对其元素进行排序。于是，`std::sort`算法应运而生，这是来自`<algorithm>`头文件的一个多功能工具，它将你的`std::vector`游戏提升到了新的水平。

## 开始使用 std::sort

`std::sort`不仅仅适用于向量；它可以对任何顺序容器进行排序。然而，它与`std::vector`的共生关系特别值得注意。最简单地说，使用`std::sort`对向量进行排序是一个直接的任务，如下面的代码所示：

```cpp
std::vector<int> numbers = {5, 1, 2, 4, 3};
std::sort(std::begin(numbers), std::end(numbers));
```

执行后，`numbers`将存储`{1, 2, 3, 4, 5}`。其美在于简单：将向量的起始和结束迭代器传递给`std::sort`，它就会处理其余部分。

## 内部引擎——introsort

在 C++ STL 提供的众多算法中，有一个始终因其有效性而突出，那就是`std::sort`。当与`std::vector`的动态特性结合时，它成为一股不可阻挡的力量，将你的代码效率推向新的高度。但是什么让它如此出色呢？

要欣赏`std::sort`背后的天才，首先必须熟悉 introsort 算法。Introsort 不仅仅是一个普通的排序算法。它是一个杰出的混合体，巧妙地融合了三种著名排序算法（快速排序、堆排序和插入排序）的优点。这种组合确保了`std::sort`能够在各种场景中适应并表现出最佳性能。

虽然我们可以深入探讨算法的复杂性，但对于日常使用来说，真正重要的是这一点：introsort 确保`std::sort`保持惊人的速度。其底层机制已经经过精炼和优化，以适应各种数据模式。

## 无与伦比的效率——O(n log n)

对于那些不深入计算机科学术语的人来说，时间复杂度可能听起来像是古老的咒语。然而，它们中蕴含着一种简单的美。当我们说`std::sort`的平均时间复杂度为*O(n log n)*时，我们表达了对速度的承诺。

将*O(n log n)*视为一个承诺。即使你的向量增长，扩展到巨大的大小，`std::sort`也能确保操作的数量不会无控制地爆炸。它找到了一个平衡点，确保排序所需的时间以可管理的速率增长，使其成为即使是最大向量也能信赖的选择。

## 降序排序

虽然升序是默认行为，但在某些情况下，你可能希望最大的值在前面。C++为你提供了支持。借助`std::greater<>()`，一个来自`<functional>`头文件预定义的比较器，你可以按以下代码所示对向量进行降序排序：

```cpp
std::sort(numbers.begin(), numbers.end(), std::greater<>());
```

执行后，如果`numbers`最初有`{1, 2, 3, 4, 5}`，现在将存储`{5, 4, 3, 2, 1}`。

## 对自定义数据类型进行排序

向量不仅限于原始类型。你可能有自定义对象的向量。为了演示这一点，我们将使用一个例子。我们将使用`Person`类和一个`Person`对象的向量。目标是首先按名称（使用内联比较器）然后按年龄（使用 lambda 函数对象作为比较器）对向量进行排序。

让我们看看一个自定义排序的例子：

```cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
struct Person {
  std::string name;
  int age{0};
  Person(std::string n, int a) : name(n), age(a) {}
  friend std::ostream &operator<<(std::ostream &os,
                                  const Person &p) {
    os << p.name << " (" << p.age << ")";
    return os;
  }
};
int main() {
  std::vector<Person> people = {Person("Regan", 30),
                                Person("Lisa", 40),
                                Person("Corbin", 45)};
  auto compareByName = [](const Person &a,
                          const Person &b) {
    return a.name < b.name;
  };
  std::sort(people.begin(), people.end(), compareByName);
  std::cout << "Sorted by name:\n";
  for (const auto &p : people) { std::cout << p << "\n"; }
  std::sort(people.begin(), people.end(),
            [](const Person &a, const Person &b) {
              return a.age < b.age;
            });
  std::cout << "\nSorted by age:\n";
  for (const auto &p : people) { std::cout << p << "\n"; }
  return 0;
}
```

下面是示例输出：

```cpp
Sorted by name:
Corbin (45)
Lisa (40)
Regan (30)
Sorted by age:
Regan (30)
Lisa (40)
Corbin (45)
```

在这个例子中，我们做了以下操作：

+   我们定义一个具有姓名和年龄属性的`Person`类。

+   我们还提供了一个内联比较函数（`compareByName`）来按姓名对`Person`对象进行排序。

+   我们然后使用内联比较器对`people`向量进行排序。

+   之后，我们使用 lambda 函数作为比较器对`people`向量按年龄进行排序。

+   结果被显示出来以验证排序操作是否按预期工作。

## 陷阱和注意事项

有一种诱惑将`std::sort`看作是一根魔杖，但请记住，虽然它很强大，但它并非无所不知。算法假设范围`(``begin, end)`是有效的；传递无效迭代器可能导致未定义的行为。此外，提供的比较器必须建立严格弱排序；否则可能会产生意外的结果。

### 严格弱排序

术语`std::sort`。这个概念涉及到用于对集合中的元素进行排序的比较函数。让我们为了清晰起见将其分解：

+   **严格性**：这意味着对于任何两个不同的元素*a*和*b*，比较函数 comp 必须不能同时报告*comp(a, b)*和*comp(b, a)*为真。用更简单的话说，如果*a*被认为是小于*b*的，那么*b*不能小于*a*。这确保了排序的一致性。

+   **弱点**：在此上下文中，“弱”一词指的是允许等价类。在严格排序（如严格全序）中，两个不同的元素不能是等效的。然而，在严格弱排序中，不同的元素可以被认为是等效的。例如，如果你有一个按年龄排序的人的列表，年龄相同的人即使在他们是不同个体的情况下也属于同一个等价类。

+   **比较的传递性**：如果 *comp(a, b)* 为真且 *comp(b, c)* 为真，那么 *comp(a, c)* 也必须为真。这确保了整个元素集合的排序一致性。

+   **等价关系的传递性**：如果 *a* 不小于 *b* 且 *b* 不小于 *a*（意味着它们在排序标准上等效），并且类似地 *b* 和 *c* 也等效，那么 *a* 和 *c* 也必须被认为是等效的。

提供严格弱排序的比较器允许 `std::sort` 正确且高效地排序元素。它确保了排序的一致性，允许对等效元素进行分组，并在比较和等效方面尊重逻辑传递性。未能遵守这些规则可能导致排序算法中出现不可预测的行为。

让我们通过一个代码示例来说明文本中提到的概念。我们将展示当向 `std::sort` 提供无效范围时会发生什么，以及如果比较器没有建立严格的弱排序会发生什么：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {3, 1, 4, 1, 5, 9};
  // Let's mistakenly provide an end iterator beyond the
  // actual end of the vector.
  std::vector<int>::iterator invalid = numbers.end() + 1;
  // Uncommenting the following line can lead to undefined
  // behavior due to the invalid range.
  // std::sort(numbers.begin(), invalidEnd);
  // This comparator will return true even when both
  // elements are equal. This violates the strict weak
  // ordering.
  auto badComparator = [](int a, int b) { return a <= b; };
  // Using such a comparator can lead to unexpected
  // results.
  std::sort(numbers.begin(), numbers.end(), badComparator);
  // Displaying the sorted array (might be unexpectedly
  // sorted or cause other issues)
  for (int num : numbers) { std::cout << num << " "; }
  std::cout << "\n";
  return 0;
}
```

在这个例子中，我们做以下操作：

+   我们看到错误地提供一个超出向量末尾的结束迭代器如何导致未定义的行为。（出于安全原因，这部分已注释掉。）

+   我们提供了一个不维护严格弱排序的比较器，因为它在两个数字相等时也返回 true。使用这样的比较器与 `std::sort` 结合可能会导致意外结果或其他未定义的行为。

通过 `std::sort`，你拥有一个高效且适应性强的工具。通过理解其默认行为，利用标准比较器的力量，并为独特场景定制比较器，你可以自信且巧妙地处理各种排序任务。随着我们继续本章的学习，请记住这项基础技能，因为我们将进一步深入到 STL 算法和 `std::vector` 的广阔领域。

在本节中，我们使用 `std::sort` 算法优化了 `std::vector` 中的元素排序，解包其 introsort 机制——一个快速排序、堆排序和插入排序的混合体，以确保最佳性能，通常具有 *O(n log n)* 的复杂度。这种理解对于算法设计中的数据处理效率和高性能应用开发至关重要。

接下来，我们将重点从排序转移到搜索，对比线性搜索和二分搜索技术，以有效地在 `std::vector` 中找到元素，分析它们在不同用例中的效率。

# 搜索元素

在集合中查找元素与存储它们一样重要。在 C++ STL 中，有一系列针对搜索的算法。无论 `std::vector` 是否排序，STL 都提供了一系列函数，可以直接使用经典的线性搜索或更快的二分搜索找到目标。使用 `std::vector`，这些技术在许多场景中变得不可或缺。

## 使用 `std::find` 进行线性搜索

最基本且直观的搜索算法是**线性搜索**。如果你不确定向量的顺序，或者它只是未排序的，这种方法就会派上用场。

考虑 `std::vector<int> numbers = {21, 12, 46, 2};`。要找到元素 `46` 的位置，我们将使用以下代码：

```cpp
auto it = std::find(numbers.begin(), numbers.end(), 46);
```

如果元素存在，它将指向其位置；否则，它将指向 `numbers.end()`。这是一个直接、无装饰的方法，从开始到结束检查每个元素。然而，它所需的时间会随着向量的大小线性增长，这使得它对于大规模数据集来说不太理想。

## 二分搜索技术

很少有算法搜索策略因其纯粹的美感和效率而脱颖而出，就像 `std::vector`，二分搜索为我们提供了一堂关于战略思维如何改变我们解决问题的方法的示范课。让我们深入探讨一半的世界，揭示二分搜索背后的 brilliance。

二分搜索基于一个简单而美丽的原则：分而治之。而不是逐个扫描每个元素，二分搜索直接跳到数据集的中心。快速评估确定所需元素位于数据集的前半部分还是后半部分。这种洞察力使它能够排除剩余元素的一半，不断缩小搜索范围，直到找到所需的元素。

为了二分搜索能够发挥作用，有一个不可协商的要求：数据集，或 `std::vector`，必须在我们这个上下文中排序。这个先决条件至关重要，因为二分搜索的效率依赖于可预测性。每次将搜索空间减半的决定都是基于对元素按特定顺序组织的信心。这种结构化的安排允许算法有信心排除大量数据，从而使搜索非常高效。

## 使用 `std::lower_bound` 和 `std::upper_bound`

但如果你想要的不仅仅是存在呢？有时，我们试图回答的问题更为复杂：如果这个元素不在向量中，根据当前的排序，它最适合放在哪里？或者，给定一个元素的多个出现，它们是从哪里开始或结束的？C++ STL 提供了两个强大的工具来解决这些查询：`std::lower_bound` 和 `std::upper_bound`。

`std::lower_bound`函数在排序向量领域扮演着关键角色。当遇到一个特定元素时，这个函数会尝试找到这个元素在向量中首次出现的位置，或者它应该正确放置的位置，以确保向量的顺序保持不变。它有效地返回一个迭代器，指向第一个不小于（即大于或等于）指定值的元素。

例如，如果我们的向量包含`{1, 3, 3, 5, 7}`，并且我们使用`std::lower_bound`来寻找`3`，函数将指向`3`的第一个出现位置。然而，如果我们正在寻找`4`，函数将指示`5`之前的位置，突出`4`最适合的位置，同时保持向量的排序特性。

另一方面，`std::upper_bound`提供了对序列结束的洞察。当给定一个元素时，它确定第一个大于指定值的元素的位置。实际上，如果你有多个元素的出现，`std::upper_bound`将指向最后一个出现之后的元素。

回到我们的向量`{1, 3, 3, 5, 7}`，如果我们使用`std::upper_bound`来搜索`3`，它将引导我们到`5`之前的位置，展示了`3`序列的结束。

让我们看看使用`std::upper_bound`和`std::lower_bound`与整数`std::vector`的完整示例。

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {1, 3, 3, 5, 7};
  int val1 = 3;
  auto low1 = std::lower_bound(numbers.begin(),
                               numbers.end(), val1);
  std::cout << "std::lower_bound for value " << val1
            << ": " << (low1 - numbers.begin()) << "\n";
  int val2 = 4;
  auto low2 = std::lower_bound(numbers.begin(),
                               numbers.end(), val2);
  std::cout << "std::lower_bound for value " << val2
            << ": " << (low2 - numbers.begin()) << "\n";
  int val3 = 3;
  auto up1 = std::upper_bound(numbers.begin(),
                              numbers.end(), val3);
  std::cout << "std::upper_bound for value " << val3
            << ": " << (up1 - numbers.begin()) << "\n";
  return 0;
}
```

当你运行前面的代码时，对于指定的值将生成以下输出：

```cpp
std::lower_bound for value 3: 1
std::lower_bound for value 4: 3
std::upper_bound for value 3: 3
```

以下是对代码示例的解释：

+   对于`std::lower_bound`和`3`，它返回一个迭代器，指向`3`的第一个出现位置，即索引`1`。

+   对于`std::lower_bound`和`4`，它指示了`4`最适合的位置，即在`5`之前（即索引`3`）。

+   对于`std::upper_bound`和`3`，它指向`3`的最后一个出现之后的元素，即在`5`之前（即索引`3`）。

虽然确认元素的存在无疑是必要的，但当我们提出更详细的问题时，使用`std::vector`的算法探索的深度才真正显现。结合`std::lower_bound`和`std::upper_bound`的能力，我们开始欣赏 STL 支持的数据分析能力。

## 二分查找与线性查找——效率和多功能性

在算法搜索技术的领域内，二分和线性搜索都成为基本策略。每个都有其独特的优势和理想的应用场景，主要应用于多功能的`std::vector`。让我们更深入地了解这两种方法的细微差别。

### 二分查找——条件下的速度高手

二分查找是一种高度有效的方法，以其对数时间复杂度而闻名。这种效率转化为显著的速度，尤其是在处理大型向量时。然而，这种迅速也有一个前提：`std::vector`必须是有序的。二分查找的本质在于其每次都能消除一半剩余元素的能力，基于元素的顺序进行有根据的猜测。

但如果这种顺序没有得到保持会发生什么？简单地说，结果变得不可预测。如果一个向量没有排序，二分查找可能无法找到即使存在的元素，或者返回不一致的结果。因此，在尝试在`std::vector`上进行二分查找之前，确保有序性是至关重要的。

### 线性查找 – 可靠的工作马

相反，线性查找以其直接的方法为特征。它系统地检查向量中的每个元素，直到找到所需的项或得出它不存在的结论。这种简单性是其优势；该方法不需要对元素的排列有任何先前的条件，使其变得灵活且适用于有序和无序向量。

然而，这种逐步检查是有代价的：线性查找具有线性时间复杂度。虽然它可能对较小的向量来说效率很高，但随着向量大小的增加，其性能可能会明显变慢，尤其是在与排序向量的快速二分查找相比时。

搜索是基础，掌握线性和二分技术可以增强你对`std::vector`的熟练程度。无论你是寻找单个元素，测量有序序列中项的位置，还是找到元素出现的范围，STL 都为你提供了强大而高效的工具来完成这些任务。随着你进一步探索`std::vector`和 STL，理解这些搜索方法是基石，确保在 C++之旅中没有任何元素被遗漏。

本节提高了我们在`std::vector`中查找元素的能力，从线性搜索的`std::find`开始，到使用`std::lower_bound`和`std::upper_bound`进行排序数据的二分搜索。与线性搜索不同，我们认识到二分搜索的速度优势，尽管它需要一个预先排序的向量。选择正确的搜索技术对于各种应用中的性能优化至关重要。

接下来，我们将探讨使用如`std::copy`等方法来更改向量内容，重点关注实际操作技巧以及保持数据结构完整性和性能的关键考虑因素。

# 操作向量

C++中的向量是动态数组，不仅存储数据，还提供了一系列操作来处理这些数据，尤其是在与 STL 提供的算法结合使用时。这些算法允许开发者以优雅的方式优化数据移动和转换任务。让我们深入探讨使用一些强大的算法来操作`std::vector`的艺术。

## 使用 std::copy 进行转换

假设你有一个向量并希望将其元素复制到另一个向量。简单的循环可能出现在你的脑海中，但有一个更高效和更表达性的方法：`std::copy`。

考虑以下代码中的两个向量：

```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination(5);
```

复制元素就像以下所示：

```cpp
std::copy(source.begin(), source.end(), destination.begin());
```

`destination` 包含 `{1, 2, 3, 4, 5}`。值得注意的是，`destination` 向量应该有足够的空间来容纳复制的元素。

## 使用 std::reverse 反转元素

经常，你可能需要反转向量的元素。而不是手动交换元素，`std::reverse` 就会派上用场，如下面的代码所示：

```cpp
std::vector<int> x = {1, 2, 3, 4, 5};
std::reverse(x.begin(), x.end());
```

向量数字现在读作 `{5, 4, 3,` `2, 1}`.

## 使用 std::rotate 旋转向量

另一个用于操作向量的实用算法是 `std::rotate`，它允许你旋转元素。假设你有一个如下向量的例子：

```cpp
std::vector<int> values = {1, 2, 3, 4, 5};
```

如果你想将其旋转，使 `3` 成为第一个元素，你将执行以下操作：

```cpp
std::rotate(values.begin(), values.begin() + 2, values.end());
```

你的向量 `values` 现在包含 `{3, 4, 5, 1, 2}`。这会将元素移动，并围绕向量回绕。

## 使用 std::fill 填充向量

可能会有一些场景，你希望将所有向量元素重置或初始化为特定值。`std::fill` 是这个任务的完美工具：

```cpp
std::vector<int> data(5);
std::fill(data.begin(), data.end(), 42);
```

现在 `data` 中的每个元素都是 `42`。

## 将操作应用于实践

一个音乐流媒体服务希望允许用户以下方式管理他们的播放列表：

+   年末时，它们有一个独特的功能：用户可以将他们最喜欢的 10 首歌曲移动到播放列表的开头，作为 *年度回顾*。

+   用户可以反转他们的播放列表，以重新发现他们很久没听过的旧歌，以特定的促销活动。

+   有时，当用户购买新专辑时，他们喜欢将其曲目插入到当前播放列表的中间，并将旧喜爱的歌曲旋转到末尾，以混合新旧歌曲。

+   对于春天的全新开始，用户可以用平静和清新的春季主题音乐填充他们的播放列表。

以下代码显示了用户如何管理他们的播放列表：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<std::string> playlist = {
      "Song A", "Song B", "Song C", "Song D",
      "Song E", "Song F", "Song G", "Song H",
      "Song I", "Song J", "Song K", "Song L"};
  std::rotate(playlist.rbegin(), playlist.rbegin() + 10,
              playlist.rend());
  std::cout << "Year in Review playlist: ";
  for (const auto &song : playlist) {
    std::cout << song << ", ";
  }
  std::cout << "\n";
  std::reverse(playlist.begin(), playlist.end());
  std::cout << "Rediscovery playlist: ";
  for (const auto &song : playlist) {
    std::cout << song << ", ";
  }
  std::cout << "\n";
  std::vector<std::string> newAlbum = {
      "New Song 1", "New Song 2", "New Song 3"};
  playlist.insert(playlist.begin() + playlist.size() / 2,
                  newAlbum.begin(), newAlbum.end());
  std::rotate(playlist.begin() + playlist.size() / 2,
              playlist.end() - newAlbum.size(),
              playlist.end());
  std::cout << "After new album purchase: ";
  for (const auto &song : playlist) {
    std::cout << song << ", ";
  }
  std::cout << "\n";
  std::vector<std::string> springSongs = {
      "Spring 1", "Spring 2", "Spring 3", "Spring 4"};
  if (playlist.size() < springSongs.size()) {
    playlist.resize(springSongs.size());
  }
  std::fill(playlist.begin(),
            playlist.begin() + springSongs.size(),
            "Spring Song");
  std::cout << "Spring Refresh: ";
  for (const auto &song : playlist) {
    std::cout << song << ", ";
  }
  std::cout << "\n";
  return 0;
}
```

这里是示例输出（已截断）：

```cpp
Year in Review playlist: Song C, Song D, Song E, Song F, Song G, Song H, [...]
Rediscovery playlist: Song B, Song A, Song L, Song K, Song J, Song I, [...]
After new album purchase: Song B, Song A, Song L, Song K, Song J, Song I, [...]
Spring Refresh: Spring Song, Spring Song, Spring Song, Spring Song, Song J, [...]
```

在这个例子中，我们做以下操作：

+   `std::rotate` 函数将用户的 10 首最受欢迎的歌曲带到列表开头。

+   `std::reverse` 函数有助于重新发现旧歌。

+   用户的新专辑购买展示了 `std::rotate` 的更实际用途。

+   `std::fill` 函数用春季主题的歌曲填充播放列表，以迎接新的开始。

## 操作注意事项

虽然这些函数提供了转换向量的强大和高效方式，但还有一些事情需要注意：

+   确保目标向量，特别是像 `std::copy` 这样的函数有足够的空间来容纳数据。如果你不确定大小，使用 `std::back_inserter` 可能会有所帮助。

+   例如，`std::rotate`这样的算法非常高效。它们最小化了元素移动的数量。然而，元素移动的顺序可能一开始并不明显。通过练习不同的场景，将培养出更精确的理解。

+   函数如`std::fill`和`std::reverse`在原地工作，转换原始向量。在应用这些函数或备份之前，始终确保你不需要原始顺序或值。

与 STL 算法配对的向量使开发者能够创建高效、表达性和简洁的操作。无论是复制、旋转、反转还是填充，都有针对该任务的算法。随着你继续使用`std::vector`，采用这些工具确保你以优雅和速度处理数据，编写出既易于阅读又易于编写的有效代码。

在本节中，我们已经掌握了使用 STL 算法修改`std::vector`的内容，特别是`std::copy`，这对于执行安全高效的数据操作至关重要。我们还涵盖了关键考虑因素，例如避免迭代器失效以维护数据完整性和性能。这种专业知识对于 C++开发者来说是无价的，因为在实际应用中，简化复杂数据操作的执行是至关重要的。

在接下来的内容中，我们将深入探讨使用比较器和谓词自定义 STL 算法行为，从而为用户定义的数据类型定义定制的排序和搜索标准。

# 定制比较器和谓词

当使用`std::vector`和 STL 算法时，你经常会遇到默认行为不符合需求的情况。有时，两个元素的比较方式或选择元素的准则必须偏离常规。这就是自定义比较器和谓词发挥作用的地方。它们是 C++ STL 强大和灵活性的证明，允许你将逻辑无缝地注入到既定的算法中。

## 理解比较器

一个`bool`。它用于指定元素顺序，尤其是在排序或搜索操作中。默认情况下，`std::sort`等操作使用`(<)`运算符来比较元素，但通过自定义比较器，你可以重新定义这一点。

想象一个整数的`std::vector`，你想要按降序排序它们。无需再编写另一个算法，你可以使用带有比较器的`std::sort`：

```cpp
std::vector<int> numbers = {1, 3, 2, 5, 4};
std::sort(numbers.begin(), numbers.end(), [](int a, int b){
    return a > b;
});
```

在这个例子中，lambda 表达式充当比较器，反转了通常的**小于**行为。

## 谓词的力量

虽然**比较器**定义了顺序，但**谓词**有助于做出决策。与比较器一样，谓词也是一个`bool`。谓词通常与需要根据某些准则进行选择或决策的算法一起使用。

例如，如果你想要计算向量中有多少个偶数，你可以使用如下代码中的`std::count_if`谓词：

```cpp
std::vector<int> x = {1, 2, 3, 4, 5};
int evens = std::count_if(x.begin(), x.end(), [](int n){
    return n % 2 == 0;
});
```

在这里，lambda 谓词检查一个数字是否为偶数，允许`std::count_if`相应地计数。

## 构建有效的比较器和谓词

以下是在构建有效的比较器和谓词时需要牢记的最佳实践：

+   **清晰性**：确保内部的逻辑清晰。比较器或谓词的目的应该在阅读后显而易见。

+   **无状态性**：比较器或谓词应该是无状态的，这意味着它不应该有任何副作用或改变调用之间的行为。

+   **效率**：由于比较器和谓词可能在算法中被反复调用，它们应该高效。避免在它们内部进行不必要的计算或调用。

## 用户定义的结构体和类

虽然 lambda 简洁方便，但定义一个结构体或类允许我们定义更复杂或更适合重用的行为。

考虑一个包含学生姓名和成绩的向量。如果你想按成绩排序，然后按姓名排序，可以使用以下代码：

```cpp
struct Student {
    std::string name;
    int grade;
};
std::vector<Student> students = { ... };
std::sort(students.begin(), students.end(), [](const Student& a, const Student& b) {
    if(a.grade == b.grade){ return (a.name < b.name); }
    return (a.grade > b.grade);
});
```

虽然 lambda 方法有效，但对于复杂的逻辑，使用结构体可能更清晰：

```cpp
struct SortByGradeThenName {
  bool operator()(const Student &first,
                  const Student &second) const {
    if (first.grade == second.grade) {
      return (first.name < second.name);
    }
    return (first.grade > second.grade);
  }
};
std::sort(students.begin(), students.end(), SortByGradeThenName());
```

自定义比较器和谓词就像给你打开了 STL 引擎室的钥匙。它们允许你利用库的原始力量，但又能精确地满足你的需求。这种精细的控制使得 C++在算法任务和数据处理方面成为一个突出的语言。

本节介绍了自定义比较器和谓词，增强了我们在`std::vector`中对元素进行排序和过滤的能力。我们学习了如何使用比较器定义排序标准，以及如何使用谓词设置条件，特别是对于用户定义的类型，允许在算法中进行复杂的数据组织。理解和利用这些工具对于开发者来说至关重要，以便在 C++中自定义和优化数据操作。

接下来，我们将探讨容器不变性和迭代器失效，学习如何管理容器稳定性并避免常见的失效问题，这对于确保健壮性，尤其是在多线程环境中至关重要。

# 理解容器不变性和迭代器失效

在 C++ STL 中，有一个关键的考虑因素经常被许多人忽视：`std::vector`，其中一个不变量可能是元素存储在连续的内存位置。然而，某些操作可能会破坏这些不变量，导致潜在的陷阱，如迭代器失效。有了这个知识，我们可以编写更健壮和高效的代码。

## 理解迭代器失效

在没有掌握**迭代器失效**的情况下，对`std::vector`的研究是不完整的。迭代器失效就像在有人重新排列了你书中的页面后尝试使用书签一样。你认为你指向了一个位置，但那里的数据可能已经改变或不存在了。

例如，当我们向向量中推送一个元素（`push_back`）时，如果预留了足够的内存（`capacity`），则元素可以无障碍地添加。但是，如果由于空间限制，向量需要分配新的内存，它可能会将所有元素重新定位到这个新的内存块。结果，任何指向旧内存块中元素的迭代器、指针或引用现在都将失效。

类似地，其他操作，如`insert`、`erase`或`resize`，也可能使迭代器失效。关键是要认识到这些操作何时可能会破坏向量的布局，并准备好处理其后果。

以下是一个代码示例，演示了使用`std::vector`的迭代器失效以及某些操作如何可能破坏容器的布局：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {1, 2, 3, 4, 5};
  std::vector<int>::iterator it = numbers.begin() + 2;
  std::cout << "The element at the iterator before"
               "push_back: "
            << *it << "\n";
  for (int i = 6; i <= 1000; i++) { numbers.push_back(i); }
  std::cout << "The element at the iterator after"
               "push_back: "
            << *it << "\n";
  it = numbers.begin() + 2;
  numbers.insert(it, 99);
  it = numbers.begin() + 3;
  numbers.erase(it);
  return 0;
}
```

在这个例子中，我们做了以下操作：

+   我们首先将一个迭代器设置为指向`numbers`向量的第三个元素。

+   在向向量中推送许多元素之后，原始内存块可能会重新分配到一个新的内存块，导致迭代器失效。

+   我们进一步展示了`insert`和`erase`操作如何使迭代器失效。

+   强调使用失效的迭代器可能导致未定义的行为，因此，在修改向量后，应始终重新获取迭代器。

在对向量进行修改操作后，始终要小心，因为它们可能会使你的迭代器失效。在这些操作之后重新获取你的迭代器，以确保它们是有效的。

## 对抗失效的策略

既然我们已经了解了迭代器可能失效的时间，现在是时候揭示绕过或优雅处理这些场景的方法了。

+   `reserve`方法。这预分配了内存，减少了在添加过程中重新分配和后续迭代器失效的需要。

+   **优先使用位置而非迭代器**：考虑存储位置（例如，索引值）而不是存储迭代器。在可能导致迭代器失效的操作之后，你可以轻松地使用位置重新创建一个有效的迭代器。

+   **操作后刷新迭代器**：在任何破坏性操作之后，避免使用任何旧的迭代器、指针或引用。相反，获取新的迭代器以确保它们指向正确的元素。

+   `<algorithm>`头文件提供了许多针对容器（如`std::vector`）优化的算法。这些算法通常内部处理潜在的失效，保护你的代码免受此类陷阱的影响。

+   **小心使用自定义比较器和谓词**：当使用需要比较器或谓词的算法时，确保它们不会以可能导致失效的方式内部修改向量。维护关注点分离的原则。

让我们看看一个集成了避免迭代器失效的关键策略的例子：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers;
  numbers.reserve(1000);
  for (int i = 1; i <= 10; ++i) { numbers.push_back(i); }
  // 0-based index for number 5 in our vector 
  size_t positionOfFive = 4;
  std::cout << "Fifth element: " << numbers[positionOfFive]
            << "\n";
  numbers.insert(numbers.begin() + 5, 99);
  std::vector<int>::iterator it =
      numbers.begin() + positionOfFive;
  std::cout << "Element at the earlier fifth position "
               "after insertion: "
            << *it << "\n";
  // After inserting, refresh the iterator
  it = numbers.begin() + 6;
  std::sort(numbers.begin(), numbers.end());
  // Caution with Custom Comparators and Predicates:
  auto isOdd = [](int num) { return num % 2 != 0; };
  auto countOdd =
      std::count_if(numbers.begin(), numbers.end(), isOdd);
  std::cout << "Number of odd values: " << countOdd
            << "\n";
  // Note: The lambda function 'isOdd' is just a read-only
  // operation and doesn't modify the vector, ensuring we
  // don't have to worry about invalidation.
  return 0;
}
```

这里是示例输出：

```cpp
Fifth element: 5
Element at the earlier fifth position after insertion: 5
Number of odd values: 6
```

这个例子做了以下操作：

+   展示了如何使用`reserve`来预分配内存，以预测大小。

+   显示位置（索引值）而不是迭代器来处理潜在的失效。

+   在破坏性操作（`insert`）之后刷新迭代器。

+   使用`<algorithm>`头文件（即`std::sort`和`std::count_if`），该文件针对容器进行了优化并尊重不变性。

+   强调了只读操作（通过`isOdd` lambda）的重要性，以避免可能的失效。（`isOdd` lambda 函数只是一个只读操作，不会修改向量，确保我们不必担心失效。）

## 处理多线程场景中的失效

虽然在单线程应用程序中迭代器失效更容易管理，但在多线程环境中事情可能会变得复杂。想象一下，一个线程正在修改一个向量，而另一个线程试图使用迭代器从中读取。混乱！灾难！以下是在多线程场景中处理失效的方法：

+   **使用互斥锁和锁定**：使用**互斥锁**保护修改向量的代码部分。这确保了在任何给定时间只有一个线程可以更改向量，防止可能导致不可预测失效的并发操作。

+   **使用原子操作**：某些操作可能是原子的，确保它们在没有中断的情况下完全完成，从而减少未同步访问和修改的可能性。

+   **考虑线程安全的容器**：如果你的应用程序以多线程为中心，考虑使用专为处理并发访问和修改而设计的**线程安全容器**，这样就不会损害不变性。

互斥锁

互斥锁（mutex），即**互斥**，是一种同步原语，用于并发编程中保护共享资源或代码的关键部分，防止多个线程同时访问。通过在访问共享资源之前锁定互斥锁并在之后解锁，一个线程确保在资源被使用时没有其他线程可以访问该资源，从而防止竞争条件并确保多线程应用程序中的数据一致性。

线程安全容器

线程安全容器指的是一种数据结构，允许多个线程并发访问和修改其内容，而不会导致数据损坏或不一致。这是通过内部机制（如锁定或原子操作）实现的，这些机制确保同步和互斥，从而在多线程环境中保持容器数据的完整性。这种容器在并发编程中对于线程之间安全高效的数据共享至关重要。

让我们看看多线程访问`std::vector`的实际例子。此示例将演示使用互斥锁来防止并发修改，确保线程安全：

```cpp
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
std::mutex vecMutex;
void add_to_vector(std::vector<int> &numbers, int value) {
  std::lock_guard<std::mutex> guard(vecMutex);
  numbers.push_back(value);
}
void print_vector(const std::vector<int> &numbers) {
  std::lock_guard<std::mutex> guard(vecMutex);
  for (int num : numbers) { std::cout << num << " "; }
  std::cout << "\n";
}
int main() {
  std::vector<int> numbers;
  std::thread t1(add_to_vector, std::ref(numbers), 1);
  std::thread t2(add_to_vector, std::ref(numbers), 2);
  t1.join();
  t2.join();
  std::thread t3(print_vector, std::ref(numbers));
  t3.join();
  return 0;
}
```

以下是示例输出：

```cpp
2 1
```

此示例说明了以下概念：

+   我们使用互斥锁（`vecMutex`）来保护共享的`std::vector`免受并发访问和修改。

+   `add_to_vector`和`print_vector`函数使用`std::lock_guard`锁定互斥锁，确保在它们的范围内对向量的独占访问。

+   我们使用`std::thread`来运行同时修改或从向量中读取的函数。使用互斥锁确保这些操作是线程安全的。

记住，虽然互斥锁可以防止并发修改，但它们也可能引入潜在的死锁并降低并行性。如果你的应用程序深度集成了多线程，你可能需要考虑其他线程安全的容器或高级同步技术。

理解和尊重容器不变性对于充分利用 STL 容器和`<algorithm>`头文件的功能至关重要。了解何时以及为什么某些不变性可能会被破坏，可以让我们创建出健壮、高效和可靠的代码。在我们继续探索`std::vector`之外的算法时，始终牢记这些原则。

在本节中，我们讨论了在容器修改过程中保持`std::vector`稳定性以及迭代器失效的风险的重要性。我们确定了导致失效的操作及其可能破坏程序完整性的潜在影响。

理解迭代器行为对于防止错误和确保我们应用程序的健壮性至关重要。我们还学习了减轻失效风险的方法，在可能危及向量一致性的操作中保持向量的一致性。

# 摘要

在整个章节中，我们通过`std::vector`及其与各种算法的交互，加深了对 STL 的理解。我们从对向量进行排序开始，探讨了`std::sort`算法及其底层引擎 introsort，并欣赏了其*O(n log n)*效率。我们进一步到向量中进行搜索，对比了线性搜索和二分搜索技术的条件和效率。

接着，章节引导我们了解有效的向量操作，包括使用`std::copy`进行转换以及防止性能下降或逻辑错误的必要考虑。我们学习了如何使用自定义比较器和谓词来扩展与用户定义的结构体和类一起使用时的标准算法的功能。最后，我们探讨了容器不变性和迭代器失效，获得了在复杂的多线程环境中保持数据完整性的策略。

重要的是，这些信息为我们提供了如何有效利用`std::vector`的实际和详细见解。掌握这些算法使开发者能够编写针对各种编程挑战的高效、健壮和适应性强代码。

接下来，我们将把我们的关注点从算法的技术复杂性转移到对为什么`std::vector`应该成为我们首选容器的大讨论上。我们将比较`std::vector`与其他容器，深入探讨其内存优势，并反思从数据处理到游戏开发的实际应用案例。这将强调`std::vector`的通用性和效率，巩固其在安全且强大的默认选择中的地位，同时作为熟练的 C++程序员众多工具之一。
