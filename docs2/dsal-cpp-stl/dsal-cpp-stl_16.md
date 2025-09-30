

# 创建 STL-类型容器

开发者可以通过将自定义类型与 C++ **标准模板库**（**STL**）集成，利用无与伦比的互操作性、一致性和效率。本章重点介绍创建与 STL 算法无缝交互的自定义类型的必要方面，强调适当的操作符重载，并实现健壮的迭代器。到本章结束时，你将熟练于设计和实现自定义类型，确保它们充分利用 STL 的优势，并提高应用程序的整体有效性。

在本节中，我们将涵盖以下主题：

+   STL 兼容类型的优势

+   与 STL 算法交互

+   兼容性的基本要求

+   为自定义类型构建迭代器

+   有效的操作符重载

+   创建自定义哈希函数

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# STL 兼容类型的优势

在 C++中构建 STL 兼容类型为寻求提升编程能力的开发者提供了许多优势。其中最显著的原因是能够根据特定需求和性能要求定制容器。虽然 STL 提供了一套丰富的泛型容器，但自定义容器使我们能够在标准容器无法满足复杂应用需求或优化目标时，对数据结构进行精细调整。此外，创建自己的容器使我们能够对关键方面，如内存布局、分配策略和容器行为，拥有更大的控制权。这种细粒度的控制使我们能够优化内存使用并提高应用程序的效率。除了实际的好处之外，开始构建自定义容器的旅程是加深我们对 C++内部和复杂性的理解的无价机会。这是一条通往语言深度和精确度更高水平的专家知识的道路。

## 一种语言，一种方法

首先也是最重要的，使自定义类型与 STL 友好提供了一种不可否认的好处——一致性。考虑 STL 中大量算法和容器的多样性。从排序例程到复杂的数据结构，STL 是 C++开发的基石。通过使你的类型与 STL 保持一致，你确保它们可以与这个庞大的库无缝交互。

想象一下——一个刚接触您的代码库的开发者，已经熟悉 STL，当他们看到您的自定义类型遵循相同的模式时，会感到宾至如归。这种一致的方法显著降低了学习曲线，提供了熟悉且直观的体验。想象一下在您的自定义类型上使用 `std::for_each` 算法的便利性，就像使用 `std::vector` 或 `std::list` 一样。这种设计上的统一性提高了生产力，并促进了代码的可读性。

## 可重用性——源源不断的礼物

建立在统一性概念的基础上，关于 STL 兼容性的另一个同样有说服力的论点是可重用性。遵循 STL 规范可以使您的自定义类型在多种场景中可重用。想想 STL 提供的庞大算法集合。一旦您的类型与 STL 兼容，它就可以立即从所有这些算法中受益，无需重新发明轮子。

此外，可重用性不仅限于算法。如果您的类型与 STL 兼容，其他开发者可以轻松地在他们的项目中采用它。随着时间的推移，这鼓励了协作开发，并培养了一个更广泛的社区参与编写、共享、审查和改进代码的生态系统。

## 在熟悉的领域中提高效率

STL 的核心是对性能的承诺。该库经过精心优化以确保效率。通过使您的类型与 STL 兼容，您使它们能够利用这些优化。无论是排序例程还是复杂的关联容器，您都可以确信您的类型将受益于 STL 中的所有性能优化。

此外，STL 友好型设计通常引导开发者避免常见的陷阱。鉴于 STL 多年来经过测试和验证，与它的规范一致本质上鼓励了类型设计中的最佳实践。

## 为前进铺路

对 STL 兼容类型优点的明显认可使未来的旅程变得更加有趣。随着我们认识到统一性、可重用性和效率的价值，我们已经为 STL 兼容性做好了准备。接下来的章节将揭示确保您的自定义类型与 STL 保持一致并展现其独特性的复杂性。从与 STL 算法交互到定制迭代器的细微差别，路线图清晰可见——创建兼容性强、多功能性高的类型。

在本节中，我们探讨了使自定义类型 STL 兼容的优点。这段旅程使您理解了 STL 友好型设计不仅仅是一个选择，而且在 C++ 开发中是一个重要的进步。我们研究了统一性、可重用性和效率的优点，强调了这些品质如何提升您的自定义类型在 C++ 生态系统中的地位。

当我们进入下一节“与 STL 算法交互”时，我们将从“为什么”过渡到“如何”实现 STL 兼容性。即将到来的这一节将指导你了解迭代器在接口 STL 算法中的关键作用，调整你的自定义类型以满足算法预期，并有效地处理错误。

# 与 STL 算法交互

本节将专注于为你提供将自定义类型无缝集成到 STL 算法中的技能，这是高级 C++ 编程的一个关键方面。这种集成不仅仅是符合标准，而且是一种共生关系，其中自定义类型和 STL 算法相互增强彼此的能力。你将学习如何为你的自定义类型设计和实现健壮的迭代器，这对于与 STL 算法实现顺畅交互至关重要。了解不同 STL 算法的具体要求，并调整你的自定义类型以满足这些需求也是关键焦点。这包括支持各种操作，如复制、比较和算术运算，这些都是算法正确运行所必需的。

我们还将涵盖错误处理和反馈机制的细微差别，教你如何让你的自定义类型不仅能够促进 STL 算法的操作，而且能够适当地应对意外情况。强调算法效率，我们将引导你通过最佳实践来确保你的自定义类型不会成为性能瓶颈。到本节结束时，你将获得关于创建与 STL 算法兼容且性能优化的自定义类型的宝贵见解，这将使你的 C++ 编程更加有效，你的应用程序更加高效和易于维护。

## 迭代器的核心地位

迭代器是自定义类型和 STL 算法之间的桥梁。在核心上，STL 算法主要依赖于迭代器在容器内导航和操作数据。因此，任何旨在完美集成的自定义类型都必须优先考虑一个健壮的迭代器设计。虽然我们将在专门的章节中涉及迭代器的构建，但理解它们的关键作用是至关重要的。提供一系列迭代器——从正向迭代器到双向迭代器，甚至随机访问迭代器——扩展了你的自定义类型可以与之交互的 STL 算法范围。

## 适应算法预期

每个 STL 算法都有一套来自其交互的容器的需求或预期。例如，`std::sort` 算法与随机访问迭代器配合工作最为优化。因此，为了确保自定义类型与这种排序例程良好配合，它应该理想地支持随机访问迭代器。

但这种关系更为深入。一些算法期望能够复制元素，一些需要比较操作，而另一些可能需要算术操作。因此，理解你想要支持的算法的先决条件至关重要。你根据这些期望对自定义类型进行越精细的调整，协同作用就越好。

## 错误处理和反馈机制

一个健壮的自定义类型不仅能够促进算法的操作，还提供了反馈机制。假设 STL 算法在操作你的自定义类型时遇到意外情况。在这种情况下，你的类型将如何响应？实现处理潜在问题并提供有意义反馈的机制是至关重要的。这可能以异常或其他 C++ 支持的错误处理范例的形式出现。

让我们来看以下示例：

```cpp
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
class CustomType {
public:
  CustomType(int value = 0) : value_(value) {}
  // Comparison operation
  bool operator<(const CustomType &other) const {
    return value_ < other.value_;
  }
  // Arithmetic operation
  CustomType operator+(const CustomType &other) const {
    return CustomType(value_ + other.value_);
  }
  // Copy operation
  CustomType(const CustomType &other)
      : value_(other.value_) {}
private:
  int value_{0};
};
class CustomContainer {
public:
  using iterator = std::vector<CustomType>::iterator;
  using const_iterator =
      std::vector<CustomType>::const_iterator;
  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin(); }
  iterator end() { return data_.end(); }
  const_iterator end() const { return data_.end(); }
  void push_back(const CustomType &value) {
    data_.push_back(value);
  }
private:
  std::vector<CustomType> data_;
};
int main() {
  CustomContainer container;
  container.push_back(CustomType(3));
  container.push_back(CustomType(1));
  container.push_back(CustomType(2));
  try {
    std::sort(container.begin(), container.end());
  } catch (const std::exception &e) {
    // Handle potential issues and provide meaningful
    // feedback
    std::cerr << "An error occurred: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
```

上述示例中的 `CustomType` 支持比较、算术和复制操作。我们还有一个 `CustomContainer`，它提供了随机访问迭代器（通过底层的 `std::vector`）。使用 `std::sort` 算法对容器中的元素进行排序。如果在排序过程中发生错误，它将在 `catch` 块中被捕获和处理。

## 算法效率和你的类型

STL 算法以其性能著称，通常经过复杂的优化。然而，如果自定义类型没有考虑到性能，它可能会成为算法效率的瓶颈。考虑以下场景：算法可能需要访问元素或频繁遍历自定义容器。在这些基本操作中的任何延迟都可能在算法执行过程中放大。

作为最佳实践，当你的自定义类型受到 STL 算法的影响时，应持续对其性能进行基准测试。性能分析工具可以提供关于潜在瓶颈和指导优化的见解。

## 建立坚实的基础

实际上，使自定义类型成为 STL 算法友好的旅程是多方面的。从迭代器的基础元素开始，探索理解算法期望，强调错误处理，以及优先考虑效率，构成了这一努力的精髓。

在本节中，我们深入探讨了将自定义类型与 STL 算法集成的过程。这有助于我们的代码与 STL 形成共生关系，其中自定义类型和 STL 算法相互增强对方的功能。我们探讨了迭代器在自定义类型和 STL 算法之间的关键作用，作为连接两者的纽带，理解它们在数据流畅导航和处理中的必要性。此外，我们还学习了如何使自定义类型适应各种 STL 算法的特定要求，确保最佳性能和有效集成。

当我们进入下一节“兼容性的基本要求”时，我们的关注点将从与 STL 算法的广泛交互转移到实现真正 STL 兼容的具体要求和标准。

# 兼容性的基本要求

在本节中，我们关注使自定义类型真正与 STL 兼容的基础方面。理解和实现我们将概述的关键元素对于充分利用 STL 强大且多功能的工具包的潜力至关重要。我们将涵盖基本要素，例如迭代器的设计、遵循值语义、操作保证以及提供大小和容量信息，每个要素都在确保与 STL 算法的无缝集成中发挥着至关重要的作用。

此处的目标是使您的自定义类型不仅能够与 STL 算法交互，还能提高其效率和功能。这需要理解 STL 在性能、操作行为和异常安全性方面的期望。通过满足这些要求，您将能够创建不仅功能性强，而且在 STL 框架内针对性能和可靠性进行了优化的自定义类型。

## 兼容性的基石

进入 STL 兼容性的世界就像加入一个独家俱乐部。进入的关键是理解和遵守基础要求。一旦您掌握了这些，STL 的巨大好处就属于您了。让我们开始这段变革之旅，揭示无缝集成所必需的基本组件。

## 迭代器的重要性

与 STL 兼容的类型等同于迭代器。它们是向 STL 算法输送数据的脉络。然而，仅仅提供迭代器是不够的。您迭代器的性质和能力定义了哪些算法可以与您的自定义类型交互。正向迭代器可能提供基本功能，但若要利用更高级的算法，您需要双向甚至随机访问迭代器。确保您的自定义类型公开适当的迭代器，将为更广泛的算法交互打开大门。

## 拥抱值语义

C++ 及其 STL 依赖于值语义。这意味着对象明确理解复制、赋值和销毁。在构建与 STL 兼容的类型时，定义清晰且高效的复制构造函数、复制赋值运算符、移动操作和析构函数至关重要。定义良好的语义行为确保算法可以无缝地创建、修改或销毁您的自定义类型的实例，而不会出现未预见的后果。

## 操作保证

算法依赖于某些操作在可预测的时间框架内执行。例如，`std::vector`保证其元素可以以常数时间访问。如果你的自定义类型承诺类似的访问，它应该始终如一地履行这一承诺。提供准确的操作保证确保算法以最佳和预期的方式执行。

## 大小和容量查询

STL 算法通常需要关于容器大小或在某些情况下其容量的信息。你的自定义类型需要及时提供这些细节。`size()`、`empty()`和可能还有`capacity()`函数应该是你设计中的基本组成部分。

## 元素访问和操作

不仅要理解结构，STL 算法还需要访问和操作其内部的元素。这需要成员函数或运算符来简化直接访问、插入和删除。这些操作越灵活，你的自定义类型能够兼容的算法范围就越广。

## 异常安全性的一致性

异常安全性是确保你的代码在异常发生时不会泄露资源或变得未定义的保证。STL 采用了一种细微的异常安全性方法，通常分为“基本”和“强”等层次。将你的自定义类型的异常安全性保证与 STL 的保证相一致，确保更顺畅的交互并加强你类型的可靠性。

让我们来看一个例子：

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
// Custom type that is STL-compatible
class CustomType {
public:
  using iterator = std::vector<int>::iterator;
  using const_iterator = std::vector<int>::const_iterator;
  // Constructors
  CustomType() = default;
  CustomType(const CustomType &other) : data(other.data) {}
  CustomType(CustomType &&other) noexcept
      : data(std::move(other.data)) {}
  // Assignment operators
  CustomType &operator=(const CustomType &other) {
    if (this != &other) { data = other.data; }
    return *this;
  }
  CustomType &operator=(CustomType &&other) noexcept {
    if (this != &other) { data = std::move(other.data); }
    return *this;
  }
  ~CustomType() = default;
  // Size and capacity queries
  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  // Element access and manipulation
  int &operator[](size_t index) { return data[index]; }
  const int &operator[](size_t index) const {
    return data[index];
  }
  void push_back(int value) { data.push_back(value); }
  void pop_back() { data.pop_back(); }
  // Iterators
  iterator begin() { return data.begin(); }
  const_iterator begin() const { return data.begin(); }
  iterator end() { return data.end(); }
  const_iterator end() const { return data.end(); }
private:
  std::vector<int> data;
};
int main() {
  CustomType custom;
  // Fill with some data
  for (int i = 0; i < 10; ++i) { custom.push_back(i); }
  // Use STL algorithm with our custom type
  std::for_each(
      custom.begin(), custom.end(),
      [](int &value) { std::cout << value << ' '; });
  return 0;
}
```

下面是示例输出：

```cpp
0 1 2 3 4 5 6 7 8 9
```

此代码定义了一个与 STL 兼容的`CustomType`类。它提供了迭代器，并定义了拷贝构造函数、移动构造函数、赋值运算符和析构函数。它还提供了查询大小和容量以及访问和操作元素的函数。`main`函数演示了如何使用`CustomType`实例与 STL 算法（`std::for_each`）一起使用。

## 期待增强的集成

在掌握这些基础要求之后，你将朝着制作与 STL 和谐共鸣的类型迈进。记住，这是一个伙伴关系。虽然 STL 提供了无与伦比强大算法和工具，但你的自定义类型带来了独特的功能和细微差别。当这些世界在兼容性中碰撞时，结果就是编码魔法。

随着我们进入下一部分，我们将深化对创建迭代器的复杂艺术和运算符重载的微妙之处的理解。你每迈出的一步都巩固了你在 STL 集成精英俱乐部中的地位，解锁了更大的编程能力。

# 为自定义类型创建迭代器

迭代器无疑是 STL 数据访问世界的心脏。它们作为桥梁，将自定义数据结构与 STL 算法的广泛数组连接起来。一个精心设计的迭代器确保了无缝的数据访问和修改，使你的自定义类型感觉就像一直是 STL 家族的一员。

在 C++编程中，为自定义类型创建 STL 迭代器至关重要，因为它们作为基本桥梁，使得自定义类型与 STL 算法的众多算法之间能够无缝集成和交互。它们促进了自定义容器内数据的遍历和操作，确保这些类型能够充分利用 STL 算法的强大功能和效率。如果没有设计良好的迭代器，自定义类型将会孤立，无法利用 STL 提供的广泛和优化的功能。

## 选择正确的迭代器类型

有许多迭代器类型可供选择，每种类型都为其提供了独特的功能。正向迭代器允许通过序列进行单向移动，而双向迭代器则提供了反向遍历的能力。更进一步，随机访问迭代器允许在数据结构中的任何位置进行快速跳跃。当为自定义类型构建迭代器时，确定哪种类型与你的数据和希望支持的运算的性质相匹配至关重要。所选类型为可以使用的算法和这些操作的效率设定了舞台。

选择迭代器类型应该根据你数据结构的固有特性和你打算执行的操作的效率要求来指导。正向迭代器是最简单的，只支持单向遍历。它们适用于只需要顺序访问的数据结构，例如单链表。这种简单性可以导致此类任务性能的优化。

双向迭代器允许双向遍历，适用于如双向链表这样的结构，其中反向迭代与正向迭代一样基本。向后移动的额外灵活性伴随着轻微的复杂性增加，但如果你的数据结构和算法从双向遍历中受益，这是一个合理的选择。

随机访问迭代器提供了最大的灵活性，允许在常数时间内直接访问任何元素，类似于数组索引。它们对于向量、数组等需要此类功能的数据结构是必不可少的。然而，这种级别的功能对于所有数据类型来说并不是必需的，如果数据结构本身不支持快速随机访问，这可能会增加不必要的开销。

从本质上讲，虽然你可以设计一个数据结构来使用更高级的迭代器类型，如随机访问，但在不需要其功能的情况下这样做可能会导致效率低下。迭代器的选择应与数据结构的自然行为和需求相一致，以确保最佳性能和资源利用。这是在迭代器提供的功能与它打算用于的数据结构的本质之间找到正确平衡的问题。

## 构建基本组件

在本质上，迭代器必须支持一组基本操作来定义其行为。这包括解引用以访问底层数据，递增和可能递减以遍历数据，以及比较以确定两个迭代器的相对位置。有效地实现这些操作确保你的自定义类型的迭代器能够与 STL 算法良好协作。

## 使用类型特性处理迭代器类别

STL 算法，作为有洞察力的实体，经常寻找关于迭代器性质的线索。它们使用这些线索来优化其行为。这正是类型特性发挥作用的地方。通过为你的自定义迭代器特化 `std::iterator_traits`，你实际上是在算法耳边低语，告诉它预期什么。这种知识使算法能够在操作中做出最佳选择，确保最佳性能。

## 结束迭代器 – 标志着终点线

每次旅行都需要一个明确的终点，迭代器也不例外。除了允许访问数据的迭代器之外，提供一个*结束*迭代器至关重要。这个特殊的迭代器不指向有效数据，而是表示边界 – 超过最后一个有效元素的点。STL 算法依赖于这个哨兵来知道何时停止操作，使其成为任何迭代器套件的重要组成部分。

## 关于常量迭代器的考虑

正如图书馆提供常规书籍和仅作参考的文本一样，数据结构通常需要满足修改和仅查看的需求。**常量迭代器**满足后一种场景，允许数据被访问而不存在修改的风险。构建常量迭代器确保你的自定义类型可以在数据完整性至关重要的场景中安全使用。

让我们看看一个说明性的 C++ 代码示例，演示了为自定义数据结构创建自定义迭代器的过程：

```cpp
#include <iostream>
#include <iterator>
#include <vector>
// Define a custom data structure for custom iterators.
class MyContainer {
public:
  MyContainer(std::initializer_list<int> values)
      : data(values) {}
  // Custom iterator for MyContainer.
  class iterator {
  private:
    std::vector<int>::iterator it;
  public:
    iterator(std::vector<int>::iterator iter) : it(iter) {}
    // Dereferencing operator to access the underlying
    // data.
    int &operator*() { return *it; }
    // Increment operator to navigate through the data.
    iterator &operator++() {
      ++it;
      return *this;
    }
    // Comparison operator to determine the relative
    // positions of two iterators.
    bool operator==(const iterator &other) const {
      return it == other.it;
    }
    bool operator!=(const iterator &other) const {
      return it != other.it;
    }
  };
  // Begin and end functions to provide iterators.
  iterator begin() { return iterator(data.begin()); }
  iterator end() { return iterator(data.end()); }
  // Additional member functions for MyContainer as needed.
private:
  std::vector<int> data;
};
int main() {
  MyContainer container = {1, 2, 3, 4, 5};
  // Using custom iterators to iterate through the data.
  for (MyContainer::iterator it = container.begin();
       it != container.end(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
1 2 3 4 5
```

## 性能优化和高级技术

构建迭代器不仅仅是关于功能，还需要技巧。考虑内存缓存技术、预取和其他优化来提升性能。记住，迭代器是一个经常使用的组件，任何效率提升都可能对整体应用性能产生重大影响。

## 拥抱迭代精神

在深入了解迭代器世界之后，很明显，它们不仅仅是工具——它们是 STL 通用性和强大功能的证明。通过精心设计自定义类型的迭代器，你可以增强与 STL 算法的互操作性，提升用户体验，使数据访问直观且高效。在本节中，我们学习了为什么选择正确的迭代器类型很重要，如何编写基本的迭代器，以及构建常量迭代器时需要考虑的事项。在下一节中，我们将探讨操作符重载的细微差别，确保我们的自定义类型在 C++ 和 STL 的世界中真正感到舒适。

# 高效的操作符重载

接下来，让我们努力理解 C++ 中操作符重载的战略实现，这是一个显著增强自定义类型功能和集成特性的特性。操作符重载允许自定义类型模拟内置类型的行为，为 STL 算法提供一个无缝的接口，以便它们能够像处理原生 C++ 类型一样高效地处理这些类型。这一特性对于确保自定义类型不仅与 STL 算法兼容，而且针对其高效执行进行了优化至关重要。

此处重点在于设计操作符重载，以促进自定义类型集成到 STL 框架中。例如，重载算术操作符如 `+`、`-` 和 `*` 允许自定义类型直接参与执行数学运算的 STL 算法，如 `std::transform` 或 `std::accumulate`。同样，重载关系操作符如 `==`、`<` 和 `>` 使自定义类型能够有效地用于需要元素比较的 STL 算法，如 `std::sort` 或 `std::binary_search`。关键是要确保这些重载的操作符模仿内置类型对应操作符的行为，保持操作直观性并提高算法结果的可预测性。通过精心实现操作符重载，我们可以确保自定义类型不仅与 STL 算法无缝交互，而且有助于提高 C++ 程序的整体效率和可读性。

## C++ 中的操作符重载

操作符重载允许 C++ 中的自定义类型为标准操作符提供特定的行为。通过利用这一特性，开发者可以像使用内置类型一样直接实现自定义类型上的操作，从而提高代码的可读性和一致性。

## 重载时的考虑因素

尽管操作符重载可以使表达式更加丰富，但关键是要谨慎使用。主要目标应该是提高清晰度，而不是引起混淆。一个基本的指导原则是，重载的操作符应该与内置类型的对应操作符表现相似。偏离这个标准可能会产生难以理解和维护的代码。

## 实现自定义类型的算术运算符

对于自定义的数学向量类型，实现加法（`+`）、减法（`-`）和乘法（`*`）等操作是合理的。重载这些运算符确保开发者可以像处理原始数据类型一样操作您的自定义类型。

## 重载关系运算符以进行清晰的比较

关系运算符（`==`、`!=`、`<`、`<=`、`>`、`>=`）不仅限于原始类型。通过为自定义类型重载这些运算符，您提供了直接比较实例的方法。这种能力简化了如对自定义对象列表进行排序等任务。

考虑一个自定义的 `Product` 类，它重载了 `+`、`<`、`=` 和 `+=` 运算符。实现方式简单直观，为与类的交互提供了非常直观的方式：

```cpp
#include <iostream>
#include <string>
class Product {
public:
  std::string name;
  double price;
  Product(const std::string &n, double p)
      : name(n), price(p) {}
  // Overloading the addition operator (+) to combine
  // prices
  Product operator+(const Product &other) const {
    return Product(name + " and " + other.name,
                   price + other.price);
  }
  // Overloading the less than operator (<) to compare
  // prices
  bool operator<(const Product &other) const {
    return price < other.price;
  }
  // Overloading the assignment operator (=) to copy
  // products
  Product &operator=(const Product &other) {
    if (this == &other) { return *this; }
    name = other.name;
    price = other.price;
    return *this;
  }
  // Overloading the compound assignment operator (+=) to
  // add prices
  Product &operator+=(const Product &other) {
    price += other.price;
    return *this;
  }
};
int main() {
  Product widget("Widget", 25.99);
  Product gadget("Gadget", 19.95);
  // Using the overloaded operators
  Product combinedProduct = widget + gadget;
  // Using the compound assignment operator
  widget += gadget;
  bool widgetIsCheaper = widget < gadget;
  bool gadgetIsCheaper = gadget < widget;
  std::cout << "Combined Product: " << combinedProduct.name
            << " ($" << combinedProduct.price << ")"
            << "\n";
  std::cout << "Is Widget cheaper than Gadget? "
            << (widgetIsCheaper ? "Yes": "No") << "\n";
  std::cout << "Is Gadget cheaper than Widget? "
            << (gadgetIsCheaper ? "Yes": "No") << "\n";
  std::cout << "Updated widget: " << widget.name << " ($"
            << widget.price << ")"
            << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
Combined Product: Widget and Gadget ($45.94)
Is Widget cheaper than Gadget? No
Is Gadget cheaper than Widget? Yes
Updated widget: Widget ($45.94)
```

这个示例演示了如何利用自定义类型上的运算符重载。这些重载（尤其是比较）对于类型与各种 STL 算法兼容是必需的。

## 使用赋值和复合赋值简化任务

重载赋值运算符（`=`）和复合赋值运算符（`+=`、`-=`、`|=`、`>>=` 以及更多）为修改您的自定义类型的实例提供了一种简单的方法，消除了需要更长的函数调用的需求。

## 高效 I/O 的流运算符

I/O 操作对于大多数应用程序来说至关重要。重载流插入运算符（`<<`）和提取运算符（`>>`）使得自定义类型可以轻松地与 C++ 流一起工作，确保了统一的 I/O 接口。

## 重载中的运算符优先级和结合性

在定义运算符重载时，牢记 C++ 中已建立的优先级和结合性规则是至关重要的。这确保了涉及您的自定义类型的表达式按预期处理。

## C++ 中运算符重载的作用

运算符重载增强了自定义类型在 C++ 中的集成。它促进了简洁直观的操作，使得自定义类型能够很好地与 STL 算法和容器一起工作。通过深思熟虑地使用此功能，开发者可以创建提供功能和使用便利的自定义类型。

在后续章节中，我们将探讨可以优化您的 C++ 开发体验的工具和实践，旨在使应用程序开发既有效又简单。

# 创建自定义哈希函数

正如我们所见，STL 提供了大量的容器类，如 `std::unordered_map`、`std::unordered_set` 和 `std::unordered_multiset`，它们在很大程度上依赖于哈希函数以实现高效操作。当与自定义类型一起工作时，创建针对您的数据结构定制的自定义哈希函数是必不可少的。在本节中，我们将了解实现自定义哈希函数的重要性，探讨良好哈希函数的特征，并提供一个示例，说明如何使用自定义哈希函数将自定义类型与 STL 容器集成。

## 与 STL 容器的互操作性

STL 容器，如 `std::unordered_map` 或 `std::unordered_set`，使用哈希表来高效地存储和检索元素。为了使你的自定义类型与这些容器兼容，你需要提供一种方法，让它们能够计算哈希值，该值用于确定元素在容器中的存储位置。没有自定义哈希函数，STL 容器将不知道如何正确地哈希你的自定义对象。

通过实现自定义哈希函数，你可以确保你的自定义类型可以无缝地与 STL 容器交互，从而提供以下好处：

+   **效率**：自定义哈希函数可以针对你的特定数据结构进行优化，从而在 STL 容器中实现更快的访问和检索时间。这种优化可以显著提高应用程序的整体性能。

+   **一致性**：自定义哈希函数使你的自定义类型能够实现哈希一致性。没有它们，同一自定义类型的不同实例可能会产生不同的哈希值，导致从容器中检索元素时出现问题。

+   **正确性**：一个设计良好的自定义哈希函数确保你的自定义类型被正确哈希，防止冲突并保持容器内数据的完整性。

## 自定义类型语义

自定义类型通常具有独特的语义和内部结构，在哈希时需要特殊处理。STL 容器默认使用标准库提供的 `std::hash` 函数。这个函数可能无法充分处理你的自定义类型的复杂性。

通过精心设计你的自定义哈希函数，你可以根据数据结构的特定要求定制哈希过程。例如，你可能需要考虑自定义类型的内部状态，有选择地哈希某些成员而排除其他成员，或者甚至应用额外的转换以确保容器中元素的最佳分布。

## 良好哈希函数的特征

在创建自定义哈希函数时，遵循定义良好哈希函数的具体特征是至关重要的。一个良好的哈希函数应该具备以下属性：

+   **确定性**：哈希函数应该总是为输入产生相同的值。这个属性确保元素在容器内始终被放置在相同的位置。

+   **均匀分布**：理想情况下，哈希函数应该在所有可能的哈希值范围内均匀分布值。不均匀的分布可能导致性能问题，因为某些桶可能过载，而其他桶则未被充分利用。

+   **最小化冲突**：当两个不同的元素产生相同的哈希值时，会发生冲突。一个良好的哈希函数通过确保不同的输入生成不同的哈希值来最小化冲突，这减少了 STL 容器性能下降的可能性。

+   **高效率**：效率对于哈希函数至关重要，尤其是在处理大数据集时。一个好的哈希函数应该是计算效率高的，确保在计算哈希值时开销最小。

+   **混合良好**：哈希函数应该产生混合良好的哈希值，这意味着输入的微小变化应该导致哈希值有显著的不同。这一特性有助于在容器内保持元素的平衡分布。

## 自定义哈希函数创建示例

让我们通过一个示例来说明自定义哈希函数的创建。假设我们有一个具有姓名和年龄的自定义 `Person` 类。我们想使用 `std::unordered_map` 来存储 `Person` 对象，并且我们需要一个自定义哈希函数来实现这一点。以下是一个此类哈希函数的实现代码：

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
class Person {
public:
  Person(const std::string &n, int a) : name(n), age(a) {}
  std::string getName() const { return name; }
  int getAge() const { return age; }
  bool operator==(const Person &other) const {
    return name == other.name && age == other.age;
  }
private:
  std::string name;
  int age{0};
};
struct PersonHash {
  std::size_t operator()(const Person &person) const {
    // Combine the hash values of name and age using XOR
    std::size_t nameHash =
        std::hash<std::string>()(person.getName());
    std::size_t ageHash =
        std::hash<int>()(person.getAge());
    return nameHash ^ ageHash;
  }
};
int main() {
  std::unordered_map<Person, std::string, PersonHash>
      personMap;
  // Insert Person objects into the map
  Person person1("Alice", 30);
  Person person2("Bob", 25);
  personMap[person1] = "Engineer";
  personMap[person2] = "Designer";
  // Access values using custom Person objects
  std::cout << "Alice's profession: " << personMap[person1]
            << "\n";
  return 0;
}
```

在这个示例中，我们定义了一个具有自定义等价运算符和自定义哈希函数 `PersonHash` 的 `Person` 类。`PersonHash` 哈希函数结合了 `name` 和 `age` 成员的哈希值，使用 XOR 确保哈希结果混合良好。这个自定义哈希函数使我们能够将 `Person` 对象用作 `std::unordered_map` 中的键。

通过实现针对我们自定义类型特定需求的自定义哈希函数，我们使 STL 容器与自定义类型的平滑集成成为可能，并确保高效、一致和正确的操作。

总结来说，当在 STL 容器中使用自定义类型时，自定义哈希函数是必不可少的。它们促进了这些容器中元素的高效、一致和正确的存储和检索。遵循良好哈希函数的特征并创建一个适合自定义类型语义的哈希函数至关重要。我们提供的示例演示了如何为自定义类型创建自定义哈希函数并有效地与 STL 容器一起使用。这种知识使您能够充分利用 C++ STL 来处理自定义数据结构。

# 概述

在这一章中，我们探讨了在 C++ 中创建 STL 类型容器的根本方面。我们首先探讨了使用 STL 兼容类型的优势，强调了一致性、可重用性和效率的好处。这些优势为更顺畅和更高效的开发过程奠定了基础。

然后，我们讨论了如何与 STL 算法交互，强调了迭代器在导航和操作容器元素中的核心地位。我们强调了将自定义类型适应算法期望、优雅地处理错误以及优化算法效率的重要性。

我们还涵盖了兼容性的基本要求，包括迭代器的重要性、值语义、操作保证、大小和容量查询以及元素访问和操作。理解这些概念确保您的自定义类型能够无缝地与 STL 集成。

此外，我们还探讨了为自定义类型创建迭代器以及操作符重载的过程。最后，我们简要介绍了创建自定义哈希函数，这在你的自定义类型被用于关联容器，如`std::unordered_map`时是必不可少的。

本章提供的信息为你提供了创建与 STL 兼容的自定义容器所需的基础知识。它使你能够充分利用 C++ STL 在项目中的全部功能，从而实现更高效和可维护的代码。

在下一章中，我们将探索模板函数、重载、内联函数以及创建泛型算法的世界。你将更好地理解如何开发与各种自定义容器类型无缝工作的算法解决方案。我们将深入探讨函数模板、SFINAE、算法重载以及使用谓词和仿函数进行定制。到本章结束时，你将具备构建自己的与 STL 兼容的算法以及进一步提升你的 C++编程技能的充分准备。
