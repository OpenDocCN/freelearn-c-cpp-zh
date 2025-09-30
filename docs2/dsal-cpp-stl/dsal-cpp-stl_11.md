# 11

# 基本算法和搜索

本章涵盖了最关键和最常用的 C++ **标准模板库** (**STL**) 算法。本章通过关注排序、条件检查、查找和搜索技术，使读者能够有效地操作和分析数据。理解这些基本算法对于希望确保高效和健壮应用的开发者至关重要。本章还强调了最佳实践，确保代码正确且优化。

本章涵盖了以下主要主题：

+   排序

+   检查条件

+   计数和查找

+   搜索和比较

+   最佳实践

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 排序

**排序** 是每个程序员都会遇到的基本概念，但它不仅仅是关于元素排序。它关乎优化，理解数据的本质，并选择合适的方法来有意义地排列这些数据。C++ STL 的强大工具箱提供了一系列针对各种场景和数据集的排序算法。但如何选择？如何有效地运用这些工具以获得最佳结果？让我们共同踏上这段启发性的旅程。

首先，为什么我们要排序？排序使数据看起来更美观，并为高效搜索、数据分析以及优化数据结构铺平了道路。无论是按姓名在地址簿中排序，还是在在线商店中按价格排序产品，排序这一行为深深地融入了计算的纹理中。

STL 提供了一个主要的排序函数：`std::sort`。这个函数非常灵活，可以排序几乎任何元素序列，从数组到向量。在底层，`std::sort` 通常使用 introsort 实现的，这是一种结合了快速排序、堆排序和插入排序的混合排序算法，确保了速度和适应性。以下是一个简单的 `std::sort` 示例：

```cpp
std::vector<int> numbers = {5, 3, 8, 1, 4};
std::sort(numbers.begin(), numbers.end());
```

但排序并不总是关于升序或数字。使用 `std::sort`，自定义比较器允许你定义顺序。想象一下，你有一个产品列表，并想按名称降序排序。你可以这样做：

```cpp
std::sort(products.begin(), products.end(), [](const Product& a, const Product& b) {
    return a.name > b.name;
});
```

这不仅仅关于常规排序。当你有几乎排序好的数据时，`std::partial_sort` 就能提供帮助。这个函数对某个子范围进行排序。比如说，你想根据分数找到前三名的学生；`std::partial_sort` 可以使这项任务更高效。

然而，了解算法只是战斗的一半；理解何时使用哪个函数是至关重要的。如果你旨在对一个包含一百万个数字的列表进行排序，`std::sort` 将是你的最佳拍档。但如果你处理的是较小的数据集，其中你必须保持相等元素的原始顺序，那么 `std::stable_sort` 是一个更合适的选择。

此外，还有一些针对特定场景量身定制的排序函数。例如，当处理大型数据集且你只对排序数据的子集感兴趣时，`std::nth_element` 是一个极好的工具。它重新排列元素，使得第 n 个位置的元素在排序序列中也将位于该位置。

选择合适的算法还涉及到理解你的数据特性。如果你有一个较小的数据集或几乎排序好的列表，插入排序可能是你的最佳选择。另一方面，对于较大的数据集，更高级的算法如归并排序或快速排序可能更合适。了解这些算法的底层机制和性能指标有助于做出明智的决定。

STL 中的排序不仅仅是安排数据，而是选择最佳方式。这是理解你的数据、应用性质以及你拥有的工具之间的舞蹈。接下来，我们将学习如何检查排序数据上的各种条件。

# 检查条件

C++ STL 的优雅之处不仅在于其丰富的容器和算法集合，还在于其精细调校的能力，让开发者能够通过基于条件的操作高效地检查和验证数据。借助谓词函数的力量，这些操作使程序员能够回答诸如“这个数据集是否具有特定的属性？”和“这个范围内的所有元素都是正数吗？”等问题。

最直观和基本操作之一是 `std::all_of`。使用这个算法，你可以检查一个范围内的所有元素是否满足给定的谓词。如果你有一个学生成绩列表，你可以使用 `std::all_of` 来查看所有成绩是否都是正数（而且应该是！）。

相比之下，它的对应函数 `std::none_of` 检查一个范围内的所有元素是否都不满足给定的谓词。假设你正在处理一个学生成绩列表，并想确保没有人得分低于及格线。在这种情况下，`std::none_of` 变得非常有价值。

三元组中的最后一个函数是 `std::any_of`，它检查序列中至少有一个元素满足特定条件。这在寻找条件存在性的场景中尤其有用，例如查找是否有任何成绩是 A（>= 90）。

让我们看看一个代码示例，说明 `std::all_of`、`std::none_of` 和 `std::any_of` 的用法：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> grades = {85, 90, 78, 92,
                             88, 76, 95, 89};
  if (std::all_of(grades.begin(), grades.end(),
                  [](int grade) { return grade > 0; })) {
    std::cout << "All students have positive grades.\n";
  } else {
    std::cout << "Not all grades are positive.\n";
  }
  if (std::none_of(grades.begin(), grades.end(),
                   [](int grade) { return grade < 80; })) {
    std::cout
        << "No student has scored below passing marks.\n";
  } else {
    std::cout << "There are students who scored below "
                 "passing marks.\n";
  }
  if (std::any_of(grades.begin(), grades.end(),
                  [](int grade) { return grade >= 95; })) {
    std::cout << "There's at least one student with an "
                 "'exceptional' grade.\n";
  } else {
    std::cout
        << "No student has an 'exceptional' grade.\n";
  }
  return 0;
}
```

下面是示例输出：

```cpp
All students have positive grades.
There are students who scored below passing marks.
There's at least one student with an 'exceptional' grade.
```

在这个例子中，我们使用了一组学生成绩作为我们的数据集。我们使用描述的算法来检查所有成绩是否为正数，没有学生得分低于及格分（在本例中认为是 80 分），以及至少有一名学生取得了*优异*的成绩（90 分或以上）。

超越这些基本检查，还有更多专门的算法，例如 `std::is_sorted`，正如其名所示，它验证一个范围是否已排序。例如，对于产品价格的数据集，这个函数可以快速检查序列是否按升序排列，确保在执行其他操作之前数据的完整性。

另一个有趣的算法是 `std::is_partitioned`。想象一下，你有一个混合数据集，你已经使用某些标准对其进行了分区，例如将数字分为偶数和奇数。这个算法根据谓词检查序列中是否存在这种分区。

虽然这些函数提供了直接验证数据的方法，但有时需求更为复杂。考虑这种情况，你可能想要比较两个序列以检查它们是否是彼此的排列。STL 提供了 `std::is_permutation` 来实现这一目的。无论是字符串、数字还是自定义对象，这个函数都可以确定一个序列是否是另一个序列的重新排序。

让我们用一个产品价格的数据集来演示 `std::is_permutation` 的用法：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<double> prices = {5.99, 10.49, 20.89, 25.55,
                                30.10};
  if (std::is_sorted(prices.begin(), prices.end())) {
    std::cout << "The product prices are sorted in"
                 "ascending order.\n";
  } else {
    std::cout << "The product prices are not sorted.\n";
  }
  auto partitionPoint = std::partition(
      prices.begin(), prices.end(),
      [](double price) { return price < 20.0; });
  if (std::is_partitioned(
          prices.begin(), prices.end(),
          [](double price) { return price < 20.0; })) {
    std::cout << "Prices are partitioned with prices less "
                 "than $20 first.\n";
  } else {
    std::cout << "Prices are not partitioned based on the "
                 "given criteria.\n";
  }
  std::vector<double> shuffledPrices = {25.55, 5.99, 30.10,
                                        10.49, 20.89};
  // Using std::is_permutation to ascertain if
  // shuffledPrices is a reordering of prices
  if (std::is_permutation(prices.begin(), prices.end(),
                          shuffledPrices.begin())) {
    std::cout
        << "Sequences are permutations of each other.\n";
  } else {
    std::cout << "Sequences are not permutations of each "
                 "other.\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
The product prices are sorted in ascending order.
Prices are partitioned with prices less than $20 first.
Sequences are permutations of each other.
```

在这个例子中，我们使用描述的算法对一个产品价格的数据集进行了操作。首先检查价格是否已排序。然后，根据价格标准进行分区。最后，我们验证两个价格序列是否是彼此的排列。

利用这些条件检查函数不仅仅是将它们应用于数据集。真正的力量来自于构建有意义的谓词。通过利用 lambda 或函数对象的能力，你可以设计复杂的条件，精确地捕捉你的需求。无论是检查用户输入的有效性，验证处理前的数据，还是确保处理后的结果的神圣不可侵犯，基于谓词的函数是你的可靠工具。

但就像任何强大的工具包一样，这些函数必须谨慎使用。过度依赖检查可能导致性能开销，尤其是在大型数据集上。在验证和性能之间取得平衡至关重要。通常，了解数据的性质和应用的更广泛背景可以指导你有效地使用这些算法。

在总结对条件检查算法的探索时，很明显，它们是 STL 算法套件的重要组成部分。它们提供了一个强大的基础，可以在其上构建更高级的操作。随着我们继续前进，你会看到这些基础检查如何与其他算法，如计数和查找交织在一起，描绘出 C++ 魅力世界中数据处理的全景。

# 计数和查找

在我们日常处理的数据中，管理或验证数据，以及积极搜索、定位和量化其中的特定元素或模式，往往变得至关重要。STL 为开发者提供了一宝库精确的算法，用于计数和查找。

让我们从简单而强大的`std::count`及其孪生兄弟`std::count_if`开始。虽然`std::count`可以迅速告诉你特定值在范围内出现的次数，但`std::count_if`更进一步，允许你根据谓词来计数。想象一下，你有一个学生分数的集合，并希望找出有多少人得分超过 90。使用`std::count_if`，这就像走平地一样简单，如下所示：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> grades = {85, 90, 78, 92,
                             88, 76, 95, 89};
  const auto exact_count =
      std::count(grades.begin(), grades.end(), 90);
  std::cout << "Number of students who scored exactly 90:"
            << exact_count << "\n";
  const auto above_count =
      std::count_if(grades.begin(), grades.end(),
                    [](int grade) { return grade > 90; });
  std::cout << "Number of students who scored above 90:"
            << above_count << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
Number of students who scored exactly 90: 1
Number of students who scored above 90: 2
```

在这里，我们使用了`std::count`来检查得分恰好为 90 的学生数量，然后使用了`std::count_if`来计数得分超过 90 的学生。

除了计数之外，有时目标是要定位一个特定的元素。这就是`std::find`和`std::find_if`发挥作用的地方。相比之下，`std::find`寻找精确匹配，而`std::find_if`基于谓词进行搜索。在你渴望知道满足条件的第一个元素的位置时，这些函数是你的首选。

然而，生活并不总是关于第一个匹配项。有时，最后一个匹配项才是重要的。在这种情况下，`std::find_end`证明是无价的。特别是在定位较大序列中子序列的最后一个出现的情况下，这个函数确保你不会错过数据中的细微差别。

让我们看看一个使用`std::list`的代码示例，其中包含学生姓名和成绩的结构。然后，我们将使用`std::find_if`和`std::find_end`根据成绩定位学生，如下所示：

```cpp
#include <algorithm>
#include <iostream>
#include <list>
struct Student {
  std::string name;
  int grade{0};
  Student(std::string n, int g) : name(n), grade(g) {}
};
int main() {
  std::list<Student> students = {
      {"Lisa", 85},   {"Corbin", 92}, {"Aaron", 87},
      {"Daniel", 92}, {"Mandy", 78},  {"Regan", 92},
  };
  auto first_92 = std::find_if(
      students.begin(), students.end(),
      [](const Student &s) { return s.grade == 92; });
  if (first_92 != students.end()) {
    std::cout << first_92->name
              << "was the first to score 92.\n";
  }
  std::list<Student> searchFor = {{"", 92}};
  auto last_92 = std::find_end(
      students.begin(), students.end(), searchFor.begin(),
      searchFor.end(),
      [](const Student &s, const Student &value) {
        return s.grade == value.grade;
      });
  if (last_92 != students.end()) {
    std::cout << last_92->name
              << "was the last to score 92.\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
Corbin was the first to score 92.
Regan was the last to score 92.
```

在这个例子中，我们使用`std::find_if`来找到第一个得分 92 的学生。然后，我们使用`std::find_end`来找到最后一个得分 92 的学生。在这个情况下，`std::find_end`函数有点棘手，因为它旨在查找子序列，但通过提供一个单元素列表（作为我们的*子序列*），我们仍然可以使用它来找到特定成绩的最后一个出现。

对于那些处理排序数据的人来说，STL 并不会让人失望。通过`std::lower_bound`和`std::upper_bound`，你可以在排序序列中高效地找到等于给定值的值的范围的开始和结束。此外，`std::binary_search`让你能够快速确定一个元素是否存在于排序范围内。记住，这些函数利用了数据的排序特性，使它们的速度比它们的通用版本快得多。

让我们定义一个`Student`结构，并使用`Student`对象的`std::set`。我们将修改比较运算符，以便根据成绩进行排序，如下所示：

```cpp
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
struct Student {
  std::string name;
  int grade{0};
  bool operator<(const Student &other) const {
    return grade < other.grade; // Sorting based on grade
  }
};
int main() {
  std::set<Student> students = {
      {"Amanda", 68},  {"Claire", 72}, {"Aaron", 85},
      {"William", 85}, {"April", 92},  {"Bryan", 96},
      {"Chelsea", 98}};
  Student searchStudent{"", 85};
  const auto lb = std::lower_bound(
      students.begin(), students.end(), searchStudent);
  if (lb != students.end() && lb->grade == 85) {
    std::cout
        << lb->name
        << " is the first student with a grade of 85.\n";
  }
  const auto ub = std::upper_bound(
      students.begin(), students.end(), searchStudent);
  if (ub != students.end()) {
    std::cout << ub->name
              << " is the next student after the last one "
                 "with a grade of 85, with a grade of "
              << ub->grade << ".\n";
  }
  if (std::binary_search(students.begin(), students.end(),
                         searchStudent)) {
    std::cout << "There's at least one student with a "
                 "grade of 85.\n";
  } else {
    std::cout << "No student has scored an 85.\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
Aaron is the first student with a grade of 85.
April is the next student after the last one with a grade of 85, with a grade of 92.
There's at least one student with a grade of 85.
```

在这个例子中，`Student`结构根据成绩在`std::set`中排序。然后，在输出中使用姓名。

说到速度，邻接算法——`std::adjacent_find` 是一个典型的例子——允许快速定位序列中的连续重复项。想象一下一个传感器正在发送数据，而你希望快速识别是否存在连续重复的读数。这个函数就是你的首选解决方案。

让我们看看一个 `std::list` 结构体的例子，其中每个条目都有一个传感器读数（温度）和读取时间：

```cpp
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <list>
struct SensorData {
  int temperature{0};
  std::chrono::system_clock::time_point timestamp;
};
int main() {
  const auto now = std::chrono::system_clock::now();
  std::list<SensorData> sensorReadings = {
      {72, now - std::chrono::hours(10)},
      {73, now - std::chrono::hours(9)},
      {75, now - std::chrono::hours(8)},
      {75, now - std::chrono::hours(7)},
      {76, now - std::chrono::hours(6)},
      {78, now - std::chrono::hours(5)},
      {78, now - std::chrono::hours(4)},
      {79, now - std::chrono::hours(3)},
      {80, now - std::chrono::hours(2)},
      {81, now - std::chrono::hours(1)}};
  auto it = sensorReadings.begin();
  while (it != sensorReadings.end()) {
    it = std::adjacent_find(
        it, sensorReadings.end(),
        [](const SensorData &a, const SensorData &b) {
          return a.temperature == b.temperature;
        });
    if (it != sensorReadings.end()) {
      int duplicateValue = it->temperature;
      std::cout << "Found consecutive duplicate readings "
                   "of value: "
                << duplicateValue
                << " taken at the following times:\n";
      while (it != sensorReadings.end() &&
             it->temperature == duplicateValue) {
        const auto time =
            std::chrono::system_clock::to_time_t(
                it->timestamp);
        std::cout << "\t"
                  << std::put_time(std::localtime(&time),
                                   "%Y-%m-%d %H:%M:%S\n");
        ++it;
      }
    }
  }
  return 0;
}
```

在这个例子中，每个 `SensorData` 结构体包含一个温度及其记录的时间戳。我们使用 `std::adjacent_find` 和自定义比较器来检查连续重复的温度读数。当我们找到这样的读数时，我们会显示读取时间以及温度值。

下面是示例输出：

```cpp
Found consecutive duplicate readings of value: 75 taken at the following times:
    2099-10-01 03:14:51
    2099-10-01 04:14:51
Found consecutive duplicate readings of value: 78 taken at the following times:
    2099-10-01 06:14:51
    2099-10-01 07:14:51
```

就像所有工具一样，理解何时以及如何使用这些算法是至关重要的。虽然由于它们的速度，频繁地使用二分搜索可能很有吸引力，但它们仅适用于排序数据。否则，使用它们可能会导致错误的结果。同样，虽然计数出现次数可能看起来很简单，但根据你是否有特定值或条件，使用正确的计数函数可以显著影响你程序的清晰度和效率。

在 C++ 中处理数据时，计数和查找是基础且复杂的操作，为更高级的操作铺平了道路。对这些操作掌握得越好，你就越有可能熟练地处理最复杂的数据场景。鉴于我们的数据已排序，我们可以通过检查 STL 中的高效搜索和比较来进一步扩展我们的工具集。

# 搜索和比较

搜索数据是一个常见但至关重要的操作，大多数软件都需要。无论是尝试从数据库中检索特定用户详情，还是找到一本书在排序列表中的位置，强大的搜索技术都是至关重要的。STL 提供了多种方法来有效地搜索序列。此外，该库提供了直观的方式来比较序列和检索极值，使数据分析更加流畅。

当处理排序数据时，`std::binary_search` 是一个强大的工具。这是保持数据排序在可行范围内的重要性的证明。通过反复将数据集分成两半，它定位所需的元素，使其成为一个异常快速的工具。然而，这仅仅是一个布尔操作；它通知元素是否存在，但并不告知其位置。为此，我们依赖 `std::lower_bound` 和 `std::upper_bound`。这些函数检索指向元素首次出现和最后一次出现之后的迭代器。结合这两个函数可以给出表示排序序列中所有实例的值的范围。

然而，并非所有数据都是排序的，并非所有搜索都是精确匹配。STL 不会让你陷入困境。`std::find`和`std::find_if`等函数在这些情况下表现出色，提供了基于实际值或谓词进行搜索的灵活性。

在搜索之后，一个自然的步骤是比较元素。通常，我们需要确定一个序列在字典序上是否小于、大于或等于另一个序列。这就是`std::lexicographical_compare`发挥作用的地方，它允许你像字典排序一样比较两个序列。当处理字符串或自定义数据类型时，这是必不可少的，确保你可以快速地按需排序和排名数据。

这里有一个示例来展示`std::lexicographical_compare`的使用：

```cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
int main() {
  std::vector<char> seq1 = {'a', 'b', 'c'};
  std::vector<char> seq2 = {'a', 'b', 'd'};
  std::vector<char> seq3 = {'a', 'b', 'c', 'd'};
  if (std::lexicographical_compare(
          seq1.begin(), seq1.end(), seq2.begin(),
          seq2.end())) {
    std::cout << "Sequence 1 is lexicographically less"
                 "than Sequence 2"
              << "\n";
  } else {
    std::cout
        << "Sequence 1 is not lexicographically less"
           "than Sequence 2"
        << "\n";
  }
  if (std::lexicographical_compare(
          seq1.begin(), seq1.end(), seq3.begin(),
          seq3.end())) {
    std::cout << "Sequence 1 is lexicographically less"
                 "than Sequence 3"
              << "\n";
  } else {
    std::cout
        << "Sequence 1 is not lexicographically less"
           "than Sequence 3"
        << "\n";
  }
  // For strings
  std::string str1 = "apple";
  std::string str2 = "banana";
  if (std::lexicographical_compare(
          str1.begin(), str1.end(), str2.begin(),
          str2.end())) {
    std::cout << "String 1 (apple) is lexicographically "
                 "less than String 2 (banana)"
              << "\n";
  } else {
    std::cout << "String 1 (apple) is not "
                 "lexicographically less "
                 "than String 2 (banana)"
              << "\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
Sequence 1 is lexicographically less than Sequence 2
Sequence 1 is lexicographically less than Sequence 3
String 1 (apple) is lexicographically less than String 2 (banana)
```

这展示了如何使用`std::lexicographical_compare`来确定两个序列的相对顺序。

但如果你只对极端值感兴趣呢？也许你想要找到考试中的最高分或产品列表中的最低价格。在这里，`std::max_element`和`std::min_element`是你的得力助手。它们分别返回指向最大和最小元素的迭代器。如果你两者都要找，`std::minmax_element`一次就能给出一个迭代器对：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  std::vector<int> scores = {85, 93, 78, 90, 96, 82};
  const auto max_it =
      std::max_element(scores.begin(), scores.end());
  if (max_it != scores.end()) {
    std::cout << "The highest score is: "<< *max_it
              << "\n";
  }
  const auto min_it =
      std::min_element(scores.begin(), scores.end());
  if (min_it != scores.end()) {
    std::cout << "The lowest score is: "<< *min_it
              << "\n";
  }
  const auto minmax =
      std::minmax_element(scores.begin(), scores.end());
  if (minmax.first != scores.end() &&
      minmax.second != scores.end()) {
    std::cout << "The lowest and highest scores are: "
              << *minmax.first << " and " << *minmax.second
              << ", respectively.\n";
  }
  std::vector<double> productPrices = {99.99, 79.99, 49.99,
                                       59.99, 89.99};
  // Find the minimum and maximum prices
  auto minmaxPrices = std::minmax_element(
      productPrices.begin(), productPrices.end());
  if (minmaxPrices.first != productPrices.end() &&
      minmaxPrices.second != productPrices.end()) {
    std::cout
        << "The cheapest and priciest products cost: $"
        << *minmaxPrices.first << " and $"
        << *minmaxPrices.second << ", respectively.\n";
  }
  return 0;
}
```

这里是示例输出：

```cpp
The highest score is: 96
The lowest score is: 78
The lowest and highest scores are: 78 and 96, respectively.
The cheapest and priciest products cost: $49.99 and $99.99, respectively.
```

这展示了如何使用`std::max_element`、`std::min_element`和`std::minmax_element`在序列中找到极值。

总结来说，STL 中搜索和比较的力量不仅在于其函数的广度，还在于其适应性。有了迭代器和谓词，这些算法非常灵活，确保你可以根据各种场景调整它们。作为开发者，这些工具成为我们思考的延伸，引导我们走向高效且优雅的解决方案。随着我们进一步发展，请记住这些操作是更高级技术和最佳实践的基础，加强我们在 C++中处理数据和算法解决问题的能力。

# 最佳实践

C++ STL 的优雅之处在于其丰富的实用工具和优化潜力。然而，仅仅了解算法并不是最终目标。你如何使用它们，如何组合它们，以及如何做出细微的决策，这决定了程序是高效还是缓慢。因此，让我们深入研究最佳实践，确保你在 STL 中的探索是正确且效率最高的：

+   在一个基本排序的数组上使用`std::binary_search`可能是不切实际的，当`std::find`可以以更低的开销完成这项任务时。

+   `std::set`和`std::map`在搜索和插入元素方面具有固有的优势。然而，它们也可能导致陷阱。持续向此类容器添加元素可能并不高效，有时，批量插入后跟排序操作可能更优。

+   使用 `std::vector` 时，通过 `reserve` 方法，预先对大小进行合理的估计并保留内存至关重要。这样，当你调用 `push_back` 添加元素时，向量不需要频繁地重新分配内存，从而提供显著的性能提升。

+   `std::count_if` 和 `std::find_if` 允许设置自定义条件，这使得它们比非谓词对应版本更加灵活和适应更广泛的场景。此外，C++11 及以后的 lambda 表达式使得使用这些算法变得更加简洁和表达力丰富。

+   **警惕算法复杂度**：虽然 STL 提供了工具，但它并没有改变算法的基本性质。线性搜索始终是线性的，二分搜索将是对数的。认识到你算法的复杂度，并质疑这对你应用程序的需求是否最佳。

+   `std::array` 是栈分配的，由于缓存局部性，其访问速度可能比堆分配的对应版本更快。然而，这伴随着固定大小的权衡。因此，事先了解内存需求可以帮助找到正确的平衡点。

+   `std::vector` 可能会失效迭代器，导致未定义行为。

+   **基准测试和性能分析**：假设和最佳实践是起点，但真实性能指标来自对应用程序的性能分析。gprof、Valgrind 和 Celero 等工具在突出瓶颈并指导你进行正确的优化方面可能非常有价值。这些最佳实践概述了如何优化 C++ STL 的使用，强调了理解数据性质、利用排序数据结构、避免不必要的内存重新分配、优先选择具有谓词版本的算法、意识到算法复杂度、在适当的情况下选择栈分配而不是堆分配、谨慎使用迭代器以及基准测试和性能分析在识别性能瓶颈中的重要性。它们强调，虽然 STL 提供了强大的工具，但高效的编程取决于如何使用和组合这些工具。

# 摘要

在本章中，我们彻底研究了在 STL 容器上操作的核心算法及其在高效 C++ 编程中的作用。我们首先探讨了排序算法的基本原理，了解它们如何组织数据以实现更好的可访问性和性能。然后，我们深入探讨了检查容器条件的方法以及计数和查找元素的技术，这些对于数据分析和处理至关重要。

本章为您提供了有效搜索和比较元素的战略。我们还关注了确保这些操作以最优效率和最小错误率执行的最佳实践。

这些知识为您在中级到高级 C++ 开发中实现复杂算法、执行数据操作和日常任务提供了基础。

在下一章中，我们将进一步扩展我们对算法的理解。我们将学习在 STL 容器内进行复制和移动语义，**返回值优化**（**RVO**），以及填充、生成、删除和替换元素的技术。此外，我们还将探讨交换和反转元素细微差别，并以去重和抽样策略作为总结。这些主题将有助于全面理解数据操作和转换。
