

# 第二章：掌握 std::vector 中的迭代器

在本章中，我们将更深入地探索 `std::vector`，重点关注迭代的复杂性。本章将使我们掌握处理向量遍历的方方面面。掌握这些核心领域可以增强 C++ 代码的效率和可靠性，并深入了解动态数组行为的基础，这对于有效使用 C++ 至关重要。

在本章中，我们将涵盖以下主要主题：

+   STL 中的迭代器类型

+   使用 `std::vector` 的基本迭代技术

+   使用 `std::begin` 和 `std::end`

+   理解迭代器要求

+   基于范围的 `for` 循环

+   创建自定义迭代器

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# STL 中的迭代器类型

在 **标准模板库**（**STL**）中，迭代器通过连接算法和容器发挥着关键作用。它们为开发者提供了一种遍历、访问以及可能修改容器中元素的手段。迭代器是 STL 中高效数据操作的基本工具。然而，它们的函数并不统一。STL 将迭代器划分为五种主要类型，每种类型提供不同的访问和控制元素的能力。本节将深入探讨这些迭代器类型，详细阐述它们的独特功能和用途。

## 输入迭代器

输入迭代器（*LegacyInputIterator*）是探索迭代器类型的起点。它们代表了迭代器的基础类别。正如其名称所暗示的，输入迭代器专注于读取和遍历元素。它们使开发者能够前进到容器中的下一个元素并检索其值。需要注意的是，在移动输入迭代器之后，无法回退到先前元素，并且不允许修改当前元素。这个迭代器类别通常用于需要数据处理但不修改数据的算法中。

以下是一个使用 `std::vector` 及其输入迭代器的简单示例：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {10, 20, 30, 40, 50};
  for (auto it = numbers.begin(); it != numbers.end();
       ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  return 0;
}
```

在此示例中，我们使用 `std::vector<int>::const_iterator` 作为输入迭代器遍历向量并打印其元素。我们遵循输入迭代器的原则，不修改元素或移动迭代器向后。需要注意的是，使用输入迭代器无法更改元素或回到前一个元素。

## 输出迭代器

接下来，我们将探讨输出迭代器（*LegacyOutputIterator*）。尽管它们与输入迭代器有相似之处，但它们的主要功能不同：向元素写入。输出迭代器简化了对它们引用的元素的赋值。然而，通过迭代器直接读取这些元素是不支持的。它们通常用于设计用于在容器内生成和填充值序列的算法。

下面是一个使用`std::vector`演示输出迭代器使用的例子：

```cpp
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
int main() {
  std::vector<int> numbers;
  std::generate_n(std::back_inserter(numbers), 10,
                  [n = 0]() mutable { return ++n; });
  for (auto num : numbers) { std::cout << num << " "; }
  std::cout << "\n";
  return 0;
}
```

在前面的代码中，`std::back_inserter`是一个输出迭代器适配器，用于与`std::vector`等容器一起工作。它允许你向向量的末尾*写入*或推送新的值。我们使用`std::generate_n`算法生成并插入数字。这种模式完美地封装了输出迭代器的*只写*特性。我们不使用输出迭代器来读取。对于读取，我们使用常规迭代器。

## 正向迭代器

在掌握基础知识之后，让我们继续前进，了解正向迭代器（*LegacyForwardIterator*）。正向迭代器结合了输入迭代器和输出迭代器的功能。因此，它们支持读取、写入，并且正如其名称所暗示的——始终向前移动。正向迭代器永远不会改变其方向。它们的通用性使它们非常适合许多在单链表（即`std::forward_list`）上操作的计算算法。

`std::forward_list`是专门为单链表设计的，因此它是展示正向迭代器的理想选择。

下面是一个简单的代码示例来说明它们的使用：

```cpp
#include <forward_list>
#include <iostream>
int main() {
  std::forward_list<int> flist = {10, 20, 30, 40, 50};
  std::cout << "Original list: ";
  for (auto it = flist.begin(); it != flist.end(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  for (auto it = flist.begin(); it != flist.end(); ++it) {
    (*it)++;
  }
  std::cout << "Modified list: ";
  for (auto it = flist.begin(); it != flist.end(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Original list: 10 20 30 40 50
Modified list: 11 21 31 41 51
```

这段代码初始化了一个`std::forward_list`，使用正向迭代器遍历并显示其元素，然后递增每个元素 1，展示了正向迭代器的读取和写入能力。

## 反向迭代器

有时，你可能需要以相反的顺序遍历向量。这时就出现了`rbegin()`和`rend()`。这些函数返回反向迭代器，它们从向量的末尾开始，到开头结束。这种反向遍历在特定的算法和数据处理的任务中可能很有用。

注意，反向迭代器在技术上是一个迭代器适配器。`std::reverse_iterator`被分类为迭代器适配器。它接受一个给定的迭代器，该迭代器应该是*LegacyBidirectionalIterator*，或者从 C++20 开始遵守`bidirectional_iterator`标准。它反转其方向。当给定一个双向迭代器时，`std::reverse_iterator`产生一个新的迭代器，以相反的方向遍历序列——从末尾到开头。

## 双向迭代器

继续讨论，我们处理双向迭代器（*LegacyBidirectionalIterator*）。这些迭代器允许在容器内向前和向后遍历。继承所有正向迭代器的功能，它们引入了反向移动的能力。它们的设计特别有利于需要频繁双向遍历的数据结构，如双向链表。

下面是一个使用 `std::list` 和其双向迭代器的例子：

```cpp
#include <iostream>
#include <list>
int main() {
  std::list<int> numbers = {1, 2, 3, 4, 5};
  std::cout << "Traversing the list forwards:\n";
  for (std::list<int>::iterator it = numbers.begin();
       it != numbers.end(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  std::cout << "Traversing the list backwards:\n";
  for (std::list<int>::reverse_iterator rit =
           numbers.rbegin();
       rit != numbers.rend(); ++rit) {
    std::cout << *rit << " ";
  }
  std::cout << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Traversing the list forwards:
1 2 3 4 5
Traversing the list backward:
5 4 3 2 1
```

在这个例子中，我们创建了一个整数 `std::list`。然后，我们通过首先使用常规迭代器向前遍历列表，然后使用反向迭代器反向遍历，来演示双向迭代。

## 随机访问迭代器

在我们的迭代器分类中，我们介绍了随机访问迭代器（*LegacyRandomAccessIterator* 和 *LegacyContiguousIterator*）。这些迭代器代表了最高的通用性，不仅允许顺序访问。使用随机访问迭代器，开发者可以向前移动多个步骤，向后退，或直接访问元素而不需要顺序遍历。这些功能使它们非常适合允许直接元素访问的数据结构，如数组或向量。

下面是一个展示随机访问迭代器（`std::vector`）的灵活性和能力的例子：

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

这个例子展示了随机访问迭代器的各种功能。我们开始于直接访问，然后跳过位置，跳跃回退，计算距离，甚至以非线性的方式访问元素。

理解迭代器类型的选择并非任意选择至关重要。每个迭代器都是针对特定的用例设计的，选择正确的一个可以显著提高您 C++ 代码的效率和优雅性。当与 STL 算法和容器一起工作时，对不同的迭代器类型及其功能有扎实的掌握至关重要。这种知识不仅简化了编码过程，还有助于调试和优化应用程序的性能。

在探索 STL 的迭代器时，我们学习了六种核心类型：输入、输出、正向、反向、双向和随机访问。认识到每种类型的独特功能对于高效的 C++ 编程至关重要，因为它影响我们如何遍历和与 STL 容器交互。掌握这些差异不仅具有学术意义，而且具有实践意义。它使我们能够为任务选择正确的迭代器，例如使用 `std::vector` 的随机访问迭代器，以利用其快速元素访问能力。

在下一节中，我们将应用这些知识，我们将看到迭代在实际中的应用，强调使用常量迭代器进行只读目的，并强调迭代器在各种容器中的适应性，为编写健壮和通用的代码奠定基础。

# 使用 std::vector 的基本迭代技术

现在我们已经了解了可用的不同类型的迭代器，让我们来探索遍历数据结构的基本概念。迭代是编程中的一个基本技术，允许开发者高效地访问和操作数据结构中的每个元素。特别是对于 `std::vector`，由于其动态特性和在 C++ 应用程序中的广泛使用，迭代至关重要。通过掌握迭代，你可以充分利用 `std::vector` 的潜力，实现诸如搜索、排序和精确轻松地修改元素等操作。本节旨在加深你对为什么迭代是有效管理和利用数据结构的关键技能的理解，为你在程序中更高级的应用打下基础。

## 遍历 `std::vector`

`std::vector` 的一个强大功能是它允许无缝遍历其元素。无论你是访问单个元素还是遍历每一个元素，理解 `std::vector` 的迭代能力至关重要。迭代是编程中许多操作的基础，从数据处理到算法转换。随着你进入本节，你将熟悉如何在 C++ 中高效且有效地遍历向量。

C++ STL 中迭代的核心概念是迭代器。将迭代器想象成高级指针，引导你遍历容器中的每个元素，例如我们钟爱的 `std::vector`。有了迭代器，你可以向前、向后移动，跳转到开始或结束位置，并访问它们所指向的内容，这使得它们成为你的 C++ 工具箱中不可或缺的工具。

## 使用迭代器进行基本迭代

每个 `std::vector` 都提供了一组成员函数，这些函数返回迭代器。其中两个主要的是 `begin()` 和 `end()`。虽然我们将在下一节深入探讨这些函数，但请理解 `begin()` 返回一个指向第一个元素的迭代器，而 `end()` 返回一个指向最后一个元素之后的迭代器。

例如，要遍历名为 `values` 的向量，你通常会使用循环，如下面的代码所示：

```cpp
for(auto it = values.begin(); it != values.end(); ++it) {
  std::cout << *it << "\n";
}
```

在这个代码示例中，`it` 是一个迭代器，它遍历 `values` 中的每个元素。循环会一直继续，直到 `it` 达到 `values.end()` 指示的位置。

## 使用常量迭代器

当你确定在迭代过程中不会修改元素时，使用常量迭代器是一种良好的做法。它们确保在遍历过程中元素保持不变。

想象你是一名博物馆导游，向游客展示珍贵的文物。你希望他们欣赏和理解历史，但你不希望他们触摸或修改这些脆弱的物品。同样，在编程中，也有你希望遍历集合、展示（或读取）其内容但不更改它们的情况。这就是常量迭代器发挥作用的地方。

要使用常量迭代器，`std::vector` 提供了 `cbegin()` 和 `cend()` 成员函数：

```cpp
for(auto cit = values.cbegin(); cit != values.cend(); ++cit) {
  std::cout << *cit << "\n";
}
```

## 迭代的好处

为什么迭代如此关键？通过有效地遍历向量，你可以做以下事情：

+   **处理数据**：无论是规范化数据、过滤它还是执行任何转换，迭代都是这些操作的核心。

+   **搜索操作**：寻找特定元素？迭代允许你逐个检查每个项目，与条件或值进行比较。

+   `sort`、`find` 和 `transform` 需要迭代器来指定它们操作的范围。

`std::vector` 迭代的灵活性和效率使其成为开发者的首选选择。虽然数组也允许遍历，但向量提供了动态大小、对溢出的鲁棒性以及与 C++ STL 的集成，这使得它们在许多场景下成为首选。

总之，掌握 `std::vector` 的迭代是成为熟练的 C++ 开发者的基础。通过了解如何遍历这个动态数组，你将解锁一系列功能，使你能够利用算法的力量，高效地处理数据，并构建强大、高效的软件。随着我们不断深入，你将更深入地了解其他向量工具，从而巩固你在这一充满活力的语言中的知识和技能。

在本节中，我们使用迭代器导航 `std::vector` 遍历，学习按顺序访问元素并利用常量迭代器进行只读操作。理解这些技术对于编写灵活和优化的与各种容器类型兼容的 C++ 代码至关重要。迭代是 STL 中数据操作的基础；掌握它是发挥库全部潜力的关键。

接下来，我们转向“使用 std::begin 和 std::end”部分，以进一步扩展我们对迭代器的知识。我们将揭示这些函数如何在不同容器中标准化迭代的开始和结束，为更灵活和松耦合的代码铺平道路。

# 使用 std::begin 和 std::end

随着你对 `std::vector` 的用例了解更多，你将遇到一些情况，在这些情况下，超越成员函数是有利或甚至是必要的。这就是非成员函数，特别是 `std::begin` 和 `std::end` 走到聚光灯下的地方。这两个函数非常实用，提供了一种更通用的方式来访问容器的开始和结束，包括但不限于 `std::vector`。

为什么会有这种区别，你可能会问？难道没有像 `vector::begin()` 和 `vector::end()` 这样的成员函数吗？确实有。然而，非成员 `std::begin` 和 `std::end` 的美妙之处在于它们在不同容器类型中的更广泛适用性，这使得你的代码更加灵活和适应性强。

C++中的向量提供了一种强大的动态内存和连续存储的结合，使它们在许多编码场景中变得不可或缺。但要真正利用它们的潜力，了解它们与迭代器的交互至关重要。虽然`begin()`和`end()`成员函数经常成为焦点，但幕后还有两个多才多艺的演员：`std::begin`和`std::end`。

当使用 C++容器时，`std::begin`函数可能看起来是另一种开始遍历容器的方法。然而，它带来了一整套奇迹。虽然它主要获取指向容器第一个元素的迭代器，但其应用并不限于向量。

当你将`std::vector`传递给`std::begin`时，就像拥有了一张后台通行证。幕后，该函数通过调用向量的`begin()`成员函数来平滑地委派任务。这种直观的行为确保了即使在进入泛型编程时，过渡仍然无缝。

与其对应物相呼应，`std::end`不仅仅是一个返回指向最后一个元素之后迭代器的函数。它是 C++对一致性承诺的见证。正如`std::begin`依赖于`begin()`一样，当你与`std::end`交互时，它巧妙而高效地调用了容器的`end()`成员函数。

而这里的真正魔法在于：尽管这些非成员函数在`std::vector`中表现出色，但它们并不受其限制。它们的泛型特性意味着它们可以很好地与各种容器协同工作，从传统的数组到列表，使它们成为那些寻求代码适应性的不可或缺的工具。

让我们看看一个示例，它展示了`std::begin`和`std::end`非成员函数在对比其成员对应物时的实用性：

```cpp
#include <array>
#include <iostream>
#include <list>
#include <vector>
template <typename Container>
void displayElements(const Container &c) {
  for (auto it = std::begin(c); it != std::end(c); ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
}
int main() {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  std::list<int> lst = {6, 7, 8, 9, 10};
  std::array<int, 5> arr = {11, 12, 13, 14, 15};
  std::cout << "Elements in vector: ";
  displayElements(vec);
  std::cout << "Elements in list: ";
  displayElements(lst);
  std::cout << "Elements in array: ";
  displayElements(arr);
  return 0;
}
```

在这个先前的例子中，我们注意到以下几点：

+   我们有一个`displayElements`泛型函数，它接受任何容器并使用`std::begin`和`std::end`非成员函数来遍历其元素。

+   然后，我们创建了三个容器：一个`std::vector`，一个`std::list`和一个`std::array`。

+   我们为每个容器调用`displayElements`以显示其元素。

使用`std::begin`和`std::end`，我们的`displayElements`函数是多才多艺的，并且可以在不同的容器类型上工作。如果我们仅仅依赖于如`vector::begin()`和`vector::end()`这样的成员函数，这将不会那么简单，这强调了非成员函数的强大和灵活性。

想象一下，你被 handed 一个承诺不仅效率高而且适应性强的工具箱。这就是`std::vector`提供的，而`std::begin`和`std::end`等函数则完美地补充了这一点。它们不仅仅是函数，而是通往更类型无关的内存管理和遍历的门户。

我们已经看到`std::begin`和`std::end`如何通过扩展迭代能力到所有 STL 容器（而不仅仅是`std::vector`）来提升我们的代码。拥抱这些非成员函数是构建容器无关、可重用代码的关键——这是 C++中灵活算法实现的一个支柱。理解这一区别对于在 STL 中有效地使用迭代器至关重要。

展望未来，下一节将引导我们了解迭代器类别及其基本要素的细微差别。这种洞察对于将算法与适当的迭代器能力相匹配至关重要，反映了 C++的类型系统的深度及其与指针语义的紧密联系。

# 理解迭代器要求

C++中的迭代器为各种数据结构提供了一个一致的接口，例如容器，以及自 C++20 以来，范围。迭代器库提供了迭代器和相关特性、适配器和实用函数的定义。

由于迭代器扩展了指针的概念，它们在 C++中本质上采用了许多指针语义。因此，任何接受迭代器的函数模板也可以无缝地与常规指针一起工作。

迭代器被分为六种类型：*LegacyInputIterator*、*LegacyOutputIterator*、*LegacyForwardIterator*、*LegacyBidirectionalIterator*、*LegacyRandomAccessIterator*和*LegacyContiguousIterator*。这些类别不是由它们的内在类型决定的，而是由它们支持的运算来区分。例如，指针可以执行为*LegacyRandomAccessIterator*定义的所有运算，因此可以在需要*LegacyRandomAccessIterator*的地方使用。

这些迭代器类别（除了*LegacyOutputIterator*）可以按层次排列。更通用的迭代器类别，如*LegacyRandomAccessIterator*，包含了较不强大类别（如*LegacyInputIterator*）的能力。如果一个迭代器符合这些类别中的任何一个，并且也满足*LegacyOutputIterator*的标准，则称为可变迭代器，能够执行输入和输出函数。不可变的迭代器被称为常量迭代器。

在本节中，我们发现了迭代器作为 C++数据结构（包括容器和范围）统一接口的关键作用。我们探讨了 C++中的迭代器库如何定义迭代器类型、相关特性、适配器和实用函数，提供了一种标准化的方式来遍历这些结构。

我们了解到迭代器扩展了指针语义，允许任何接受迭代器的函数模板与指针无缝工作。我们进一步探讨了迭代器类别的层次结构——*LegacyInputIterator*、*LegacyOutputIterator*、*LegacyForwardIterator*、*LegacyBidirectionalIterator*、*LegacyRandomAccessIterator* 和 *LegacyContiguousIterator*。这些类别不是由它们的类型定义的，而是由它们支持的运算定义的，更高级的迭代器继承了简单迭代器的功能。

这项知识对我们至关重要，因为它告诉我们根据需要执行的操作来选择迭代器。了解每个迭代器类别的需求和功能使我们能够编写更高效和健壮的代码，因为我们可以选择满足我们需求的最弱迭代器，从而避免不必要的性能开销。

在下一节中，我们将从迭代器的理论基础过渡到实际应用，通过学习如何使用基于范围的 `for` 循环来迭代 `std::vector`，我们将了解这些循环如何在底层使用 `std::begin` 和 `std::end`，提供了一种更直观且更不易出错的元素访问和修改方法。

# 基于范围的 for 循环

在 C++ 中，基于范围的 `for` 循环为迭代容器如 `std::vector` 提供了一种简洁实用的机制。凭借对 `std::vector` 操作和 `std::begin` 以及 `std::end` 函数的了解，很明显，基于范围的 `for` 循环提供了一种简化的遍历技术。

在向量上使用传统的迭代需要声明一个迭代器，将其初始化为容器的开始位置，并更新它以进步到末尾。虽然这种方法可行，但它需要仔细的管理，并且容易出错。基于范围的 `for` 循环提供了一个更有效的解决方案。

## 基于范围的 for 循环概述

以下代码演示了基于范围的 `for` 循环的基本结构：

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
  std::cout << num << " ";
}
```

在这个例子中，`numbers` 向量中的每个整数都被打印出来。这种方法消除了显式迭代器和手动循环边界定义的需要。

### 内在机制

在内部，基于范围的 `for` 循环利用 `begin()` 和 `end()` 函数来导航容器。循环依次从容器中检索每个项目，并将其分配给循环变量（在这种情况下为 `num`）。

这种方法简化了迭代过程，使开发者能够专注于对每个元素执行的操作，而不是检索过程。

## 何时使用基于范围的 for 循环

基于范围的 `for` 循环在以下情况下特别有益：

+   `for` 循环对于完整向量遍历是最优的。

+   **直接迭代器访问不是必需的**：这些循环非常适合显示或修改元素。然而，如果需要访问迭代器本身（例如，在遍历过程中插入或删除元素），则传统的循环更为合适。

+   `for` 循环简洁地表达了操作每个容器元素的意图。

## 在迭代过程中修改元素

对于在迭代过程中需要修改向量元素的场景，使用引用作为循环变量是至关重要的，如下面的代码所示：

```cpp
for (auto &num : numbers) {
  num *= 2;
}
```

在这种情况下，numbers 向量中的每个整数都乘以二。如果没有引用 `(&)`，循环将改变复制的元素，而原始向量保持不变。

基于范围的 `for` 循环是 C++ 持续发展的证明，它在性能和可读性之间取得了平衡。它们为开发者提供了直接导航容器的途径，增强了代码的清晰度并最小化了潜在的错误。随着你在 C++ 中的进步，理解可用的工具并选择最适合你任务的工具至关重要。对 `std::vector` 函数和功能的彻底掌握确保了在多种情况下有效利用。

本节强调了基于范围的 `for` 循环在迭代 STL 容器时的优势，强调了其可读性和与传统 `for` 循环相比最小化的错误潜力。利用 `std::begin` 和 `std::end`，这些循环简化了迭代过程，让我们能够专注于元素级别的逻辑。它们在不需要直接迭代器控制时是最优的，这体现了现代 C++ 对高效和清晰的高层抽象的重视。

接下来，*创建自定义迭代器*这一部分将利用我们的迭代器进行高级抽象、数据转换或过滤数据视图。我们将探讨技术要求以及如何使我们的自定义迭代器与 STL 的分类保持一致。

# 创建自定义迭代器

C++ 的一个美丽之处在于其灵活性，赋予开发者根据需要塑造语言的能力。这种灵活性不仅限于容器迭代的内置功能。虽然 `std::vector` 附带了一组内置迭代器，但没有任何阻止我们创建自己的。但我们为什么想要这样做呢？

## 自定义迭代器的吸引力

让我们来看看你为什么想要实现一个自定义迭代器：

+   **增强抽象**：考虑一个以扁平格式存储矩阵的向量。通过行或列而不是单个元素来迭代是否更直观？自定义迭代器可以促进这一点。

+   **数据转换**：也许你希望迭代向量，但检索转换后的数据，如每个元素的平方值。而不是在检索前后或期间更改数据，自定义迭代器可以抽象这一点。

+   `std::vector`.

创建自定义 STL 迭代器可能看起来是一项艰巨的任务，但有了适当的指导，它就变得轻而易举！在其核心，迭代器是一个高级指针——一个引导你通过容器元素的向导。为了让你的迭代器与 STL 稳定地协同工作，你需要实现某些成员函数。

## 核心要求

这些函数的确切集合取决于你创建的迭代器类型，但其中一些是通用的。

1.  `value_type`：表示迭代器指向的元素类型。

1.  `difference_type`：表示两个迭代器之间的距离。

1.  `pointer` 和 `reference`：定义迭代器的指针和引用类型。

1.  `iterator_category`：将迭代器分类为输入、输出、前向、双向或随机访问等类别。每个类别都有其独特的特征，使迭代器变得灵活且有趣！

1.  `operator*`：解引用运算符，允许访问迭代器指向的元素。

1.  `operator++`：增量运算符！这些运算符将你的迭代器向前移动（无论是前缀增量还是后缀增量风格）。

1.  `operator==` 和 `operator!=`：装备了这些，你的迭代器可以进行比较，让算法知道它们是否到达了末尾或需要继续前进。

## 迭代器类别及其特性

迭代器有多种风味；每种风味（或类别）都有独特的要求：

+   `operator*`, `operator++`, `operator==`, 和 `operator!=`

+   `operator*` 和 `operator++`*   **前向迭代器**：它们结合输入和输出迭代器——读取、写入，并且始终向前移动。

    +   **基本要求**：所有核心要求*   `operator--` 以步退*   `std::vector`。

    +   `operator+`, `operator-`, `operator+=`, `operator-=`, `operator[]`, 和关系运算符如 `operator<`, `operator<=`, `operator>`, 和 `operator>=`

    C++ 中的随机访问迭代器是功能最强大的迭代器类别之一，需要几个函数和运算符才能完全与 STL 算法和容器兼容。

这里是一个为随机访问迭代器通常实现的函数和运算符列表：

+   `iterator_category` (应设置为 `std::random_access_iterator_tag`)

+   `value_type`

+   `difference_type`

+   `pointer`

+   `reference`

+   `operator*()` (解引用运算符)*   `operator->()` (箭头运算符)*   `operator++()` (前缀增量)*   `operator++(int)` (后缀增量)*   `operator--()` (前缀减量)*   `operator--(int)` (后缀减量)*   `ptrdiff_t`):

    +   `operator+(difference_type)` (通过某些数量向前移动迭代器)

    +   `operator-(difference_type)` (通过某些数量向后移动迭代器)

    +   `operator+=(difference_type)` (通过某些数量增加迭代器)

    +   `operator-=(difference_type)`（按某些量递减迭代器）*   `operator-(const RandomAccessIteratorType&)`*   `operator[](difference_type)`*   `operator==`（相等）*   `operator!=`（不等）*   `operator<`（小于）*   `operator<=`（小于或等于）*   `operator>`（大于）*   `operator>=`（大于或等于）*   **交换**（有时很有用，但不是迭代器本身的严格要求）：

    +   一个用于交换两个迭代器的交换函数

并非所有这些总是适用，特别是如果底层数据结构有限制或迭代器的特定使用场景不需要所有这些操作。然而，为了与 STL 的随机访问迭代器完全兼容，这是你想要考虑实现的一组完整函数和运算符。

## 自定义迭代器示例

让我们为`std::vector<int>`创建一个自定义迭代器，当解引用时，返回向量中值的平方：

```cpp
#include <iostream>
#include <iterator>
#include <vector>
class SquareIterator {
public:
  using iterator_category =
      std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using pointer = int *;
  using reference = int &;
  explicit SquareIterator(pointer ptr) : ptr(ptr) {}
  value_type operator*() const { return (*ptr) * (*ptr); }
  pointer operator->() { return ptr; }
  SquareIterator &operator++() {
    ++ptr;
    return *this;
  }
  SquareIterator operator++(int) {
    SquareIterator tmp = *this;
    ++ptr;
    return tmp;
  }
  SquareIterator &operator+=(difference_type diff) {
    ptr += diff;
    return *this;
  }
  SquareIterator operator+(difference_type diff) const {
    return SquareIterator(ptr + diff);
  }
  value_type operator[](difference_type diff) const {
    return *(ptr + diff) * *(ptr + diff);
  }
  bool operator!=(const SquareIterator &other) const {
    return ptr != other.ptr;
  }
private:
  pointer ptr;
};
int main() {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  SquareIterator begin(vec.data());
  SquareIterator end(vec.data() + vec.size());
  for (auto it = begin; it != end; ++it) {
    std::cout << *it << ' ';
  }
  SquareIterator it = begin + 2;
  std::cout << "\nValue at position 2: " << *it;
  std::cout
      << "\nValue at position 3 using subscript operator: "
      << it[1];
  return 0;
}
```

当运行此代码时，将输出以下内容：

```cpp
1 4 9 16 25
Value at position 2: 9
Value at position 3 using subscript operator: 16
```

代码中的迭代器可以非常类似于内置数组或`std::vector`迭代器使用，但在解引用时具有平方值的独特功能。

## 自定义迭代器的挑战和用例

创建自定义迭代器不仅仅是理解你的数据或用例；它还涉及到应对一些挑战：

+   **复杂性**：构建迭代器需要遵循某些迭代器概念。根据它是输入迭代器、正向迭代器、双向迭代器还是随机访问迭代器，必须满足不同的要求。

+   `push_back`或`erase`。确保自定义迭代器保持有效对于安全且可预测的行为至关重要。

+   **性能开销**：随着功能的增加，可能会带来额外的计算。确保迭代器的开销不会抵消其好处是至关重要的。

## 自定义迭代器的说明性用例

为了理解这个概念，让我们简要地看看几个自定义迭代器大放异彩的场景：

+   `std::vector`可能以线性方式存储图像的像素数据。自定义迭代器可以促进按行、通道或甚至感兴趣区域进行迭代。

+   `std::vector<char>`，迭代器可以被设计成从单词跳到单词或从句子跳到句子，忽略空白和标点符号。

+   **统计抽样**：对于存储在向量中的大数据集，迭代器可能会采样每*n*个元素，从而在不遍历每个元素的情况下提供快速概述。

创建自定义迭代器需要遵循特定的约定并定义一组必需的运算符，以赋予它迭代器的行为。

以下代码展示了如何为从存储在`std::vector`中的位图中提取 alpha 通道创建自定义迭代器：

```cpp
#include <iostream>
#include <iterator>
#include <vector>
struct RGBA {
  uint8_t r, g, b, a;
};
class AlphaIterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = uint8_t;
  using difference_type = std::ptrdiff_t;
  using pointer = uint8_t *;
  using reference = uint8_t &;
  explicit AlphaIterator(std::vector<RGBA>::iterator itr)
      : itr_(itr) {}
  reference operator*() { return itr_->a; }
  AlphaIterator &operator++() {
    ++itr_;
    return *this;
  }
  AlphaIterator operator++(int) {
    AlphaIterator tmp(*this);
    ++itr_;
    return tmp;
  }
  bool operator==(const AlphaIterator &other) const {
    return itr_ == other.itr_;
  }
  bool operator!=(const AlphaIterator &other) const {
    return itr_ != other.itr_;
  }
private:
  std::vector<RGBA>::iterator itr_;
};
int main() {
  std::vector<RGBA> bitmap = {
      {255, 0, 0, 128}, {0, 255, 0, 200}, {0, 0, 255, 255},
      // ... add more colors
  };
  std::cout << "Alpha values:\n";
  for (AlphaIterator it = AlphaIterator(bitmap.begin());
       it != AlphaIterator(bitmap.end()); ++it) {
    std::cout << static_cast<int>(*it) << " ";
  }
  std::cout << "\n";
  return 0;
}
```

在此示例中，我们定义了一个 `RGBA` 结构体来表示颜色。然后我们创建了一个自定义的 `AlphaIterator` 迭代器来导航 alpha 通道。接下来，迭代器使用底层的 `std::vector<RGBA>::iterator`，但在解引用时仅暴露 alpha 通道。最后，`main` 函数演示了使用此迭代器打印 alpha 值。

此自定义迭代器遵循 C++ 输入迭代器的约定，使其可用于各种算法和基于范围的 `for` 循环。示例中的 `AlphaIterator` 类演示了 C++ 中自定义输入迭代器的基本结构和行为。以下是关键成员函数及其对 STL 兼容性的重要性的分解：

+   `iterator_category`：定义迭代器的类型/类别。它帮助算法确定迭代器支持的操作。在此处，它定义为 `std::input_iterator_tag`，表示它是一个输入迭代器。

+   `value_type`：可以从底层容器中读取的数据类型。在此处，它是表示 alpha 通道的 `uint8_t`。

+   `difference_type`：用于表示两个迭代器相减的结果。通常用于随机访问迭代器。

+   `pointer` 和 `reference`：指向 `value_type` 的指针和引用类型。它们提供了对值的直接访问。

+   `explicit AlphaIterator`(`std::vector<RGBA>::iterator itr`): 此构造函数对于使用底层 `std::vector` 迭代器的实例初始化迭代器至关重要。*   `reference operator*()`：解引用运算符返回序列中当前项的引用。对于此迭代器，它返回 RGBA 值的 alpha 通道的引用。*   `AlphaIterator& operator++()`：前置增量运算符将迭代器向前推进到下一个元素。*   `AlphaIterator operator++(int)`：后置增量运算符将迭代器向前推进到下一个元素，但在增量之前返回当前元素的迭代器。这种行为对于 `it++` 等构造是必需的。*   `bool operator==(const AlphaIterator& other) const`：检查两个迭代器是否指向相同的位置。这对于比较和确定序列的末尾至关重要。*   `bool operator!=(const AlphaIterator& other) const`：前一个操作的相反：此操作检查两个迭代器是否不相等。

这些成员函数和类型别名对于使迭代器与 STL 兼容以及能够无缝使用各种 STL 算法和构造至关重要。它们定义了功能输入迭代器所需的基本接口和语义。

对于具有更多功能（如双向或随机访问）的迭代器，可能需要额外的操作。但对于在 `AlphaIterator` 中演示的输入迭代器，上述内容是核心组件。

这一节涵盖了自定义迭代器及其为特定需求（如数据抽象、转换和过滤）的创建。学会定义基本类型别名和实现关键运算符对于扩展 `std::vector` 的功能至关重要。这种知识使我们能够定制数据交互，确保我们的代码以精确的方式满足独特的领域需求。

# 摘要

在这一章中，我们全面探讨了迭代器在 C++ STL 中最灵活的容器之一的角色和用法。我们首先讨论了 STL 中可用的各种迭代器类型——输入、输出、正向、反向、双向和随机访问——以及它们的特定应用和支持操作。

然后，我们转向了实用的迭代技术，详细说明了如何使用标准迭代器和常量迭代器有效地遍历 `std::vector`。我们强调了选择正确的迭代器类型来完成手头任务的重要性，以编写干净、高效且具有容错能力的代码。

在使用 `std::begin` 和 `std::end` 的章节中，我们扩展了我们的工具箱，展示了这些非成员函数如何通过不紧密绑定到容器类型来使我们的代码更加灵活。我们还涵盖了迭代器的需求和分类，这是理解 STL 内部工作原理和实现自定义迭代器所必需的基本知识。

基于范围的 `for` 循环被引入作为一种现代 C++ 功能，它通过抽象迭代器管理的细节来简化迭代。我们学习了何时以及如何充分利用这些循环，特别是它们在迭代过程中修改元素时的便捷性。

最后，我们探讨了创建自定义迭代器的进阶主题。我们发现了背后的动机，例如提供更直观的导航或展示过滤后的数据视图。我们检查了自定义迭代器的核心要求、挑战和用例，从而完善了我们对其如何定制以满足特定需求的理解。

虽然与 `std::vector` 一起提供的标准迭代器覆盖了许多用例，但这并不是故事的终结。自定义迭代器提供了一条途径，可以扩展迭代可能性的边界，将遍历逻辑定制到特定需求。制作可靠的自定义迭代器的复杂性不容小觑。在我们结束这一章时，请记住，在正确的人手中，自定义迭代器可以是强大的工具。你可以通过对其工作原理的深入了解，做出关于何时以及如何使用它们的明智决策。

在这一章中获得的知识是有益的，因为它使我们能够创建更复杂、更健壮和性能更优的 C++ 应用程序。有效地理解和利用迭代器使我们能够充分利用 `std::vector` 的全部功能，并编写容器无关且高度优化的算法。

即将到来的章节，*使用 std::vector 精通内存和分配器*，建立在我们的现有知识之上，并将我们的关注点引向内存效率，这是高性能 C++ 编程的一个关键方面。我们将继续强调这些概念的实际、现实世界的应用，确保内容保持价值，并直接适用于我们作为中级 C++ 开发者的工作。
