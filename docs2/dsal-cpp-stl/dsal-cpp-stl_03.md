

# 第三章：使用 `std::vector` 掌握内存和分配器

本章深入探讨了现代 C++ 编程中的关键内存管理概念。我们首先区分了 `std::vector` 的容量和大小，这对于编写高效代码至关重要。随着我们的进展，我们将了解内存预留和优化的机制，以及为什么这些操作在现实世界的应用中很重要。本章以彻底探讨自定义分配器结束，包括何时使用它们及其对容器性能的影响。它为我们提供了调整程序内存使用的专业知识。

在本章中，我们将涵盖以下主要主题：

+   理解容量与大小

+   调整和预留内存

+   自定义分配器基础知识

+   创建自定义分配器

+   分配器和容器性能

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 理解容量与大小

随着你深入到 C++ 编程艺术的殿堂，使用 `std::vector`，掌握向量的大小和容量的区别变得至关重要。虽然这两个术语密切相关，但在管理和优化动态数组时，它们扮演着不同的角色。理解它们将显著提高你代码的效率和清晰度。

## 回顾基础知识

回顾上一章的内容，向量的大小表示它当前包含的元素数量。当你添加或删除元素时，这个大小会相应调整。所以，如果你有一个包含五个整数的向量，其大小是 `5`。删除一个整数，大小变为 `4`。

但 `std::vector` 的一个引人注目的方面在于：虽然其大小根据其元素而变化，但它分配的内存并不总是立即跟随。为了彻底理解这一点，我们需要探索容量的概念。让我们在下一节中探讨它。

## 容量究竟是什么？

`std::vector` 指的是向量为自己预留的内存量——在重新分配内存之前它可以容纳的元素数量。这并不总是等于它当前持有的元素数量（即大小）。`std::vector` 通常分配比所需更多的内存，这是一种预防策略，以适应未来的元素。这就是 `std::vector` 的天才之处；过度分配减少了频繁的，可能还有计算成本高昂的重新分配的需求。

让我们用一个类比来使这个问题更直观。想象一下向量就像一列有隔间（内存块）的火车。当火车（向量）开始旅程时，它可能只有几个乘客（元素）。然而，预计在未来的车站会有更多的乘客，火车开始时有一些空隔间。火车的容量是隔间的总数，而大小是有乘客的隔间数量。

## 为什么这种区别很重要

你可能会想知道为什么我们不在每次添加新元素时仅仅扩展内存。答案在于计算效率。内存操作，尤其是重新分配，可能很耗时。向量通过分配比立即需要的更多内存来最小化这些操作，确保在大多数情况下添加元素仍然是一个快速操作。这种优化是 `std::vector` 成为 C++ 编程中必备工具的一个原因。

然而，也存在另一面。过度分配意味着一些内存可能暂时未被使用。如果内存使用是一个关键问题，那么理解和管理容量变得至关重要。在某些极端情况下，一个向量可能有 `10` 的大小，但容量为 `1000`！

## 查看内部结构

有时必须查看内部结构，以欣赏大小和容量的细微差别。考虑一个新初始化的 `std::vector<int> numbers;`。如果你逐个将 10 个整数推入它，并定期检查其容量，你可能会注意到一些有趣的事情：容量不会为每个整数增加一个！相反，它可能从 `1` 跳到 `2`，然后到 `4`，然后到 `8`，依此类推。这种指数增长策略是典型的实现方法，确保向量在用完空间时容量加倍。

让我们看看一个展示 `std::vector` 中大小和容量差异的代码示例：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> myVec;
  std::cout << "Initial size: " << myVec.size()
            << ", capacity: " << myVec.capacity() << "\n";
  for (auto i = 0; i < 10; ++i) {
    myVec.push_back(i);
    std::cout << "After adding " << i + 1
              << " integers, size: " << myVec.size()
              << ", capacity: " << myVec.capacity()
              << "\n";
  }
  myVec.resize(5);
  std::cout << "After resizing to 5 elements, size: "
            << myVec.size()
            << ", capacity: " << myVec.capacity() << "\n";
  myVec.shrink_to_fit();
  std::cout << "After shrinking to fit, size: "
            << myVec.size()
            << ", capacity: " << myVec.capacity() << "\n";
  myVec.push_back(5);
  std::cout << "After adding one more integer, size: "
            << myVec.size()
            << ", capacity: " << myVec.capacity() << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Initial size: 0, capacity: 0
After adding 1 integers, size: 1, capacity: 1
After adding 2 integers, size: 2, capacity: 2
After adding 3 integers, size: 3, capacity: 3
After adding 4 integers, size: 4, capacity: 4
After adding 5 integers, size: 5, capacity: 6
After adding 6 integers, size: 6, capacity: 6
After adding 7 integers, size: 7, capacity: 9
After adding 8 integers, size: 8, capacity: 9
After adding 9 integers, size: 9, capacity: 9
After adding 10 integers, size: 10, capacity: 13
After resizing to 5 elements, size: 5, capacity: 13
After shrinking to fit, size: 5, capacity: 5
After adding one more integer, size: 6, capacity: 7
```

下面是这个代码块的解释：

+   我们首先创建一个空的 `std::vector<int>` 命名为 `myVec`。

+   我们然后打印出初始的 `size` 和 `capacity`。由于它是空的，`size` 值将是 `0`。初始的 `capacity` 值可能因 C++ 的 `0` 而有所不同。

+   当我们逐个将整数推入向量时，我们可以看到大小和容量是如何变化的。对于每个添加的元素，`size` 值将始终增加一个。然而，`capacity` 值可能保持不变或增加，通常加倍，这取决于底层内存何时需要重新分配。

+   将向量的大小调整到五个元素表明，虽然 `size` 减少了，但 `capacity` 保持不变。这确保了之前分配的内存仍然为潜在的将来元素保留。

+   `shrink_to_fit()` 函数将向量的 `capacity` 减少以匹配其 `size`，从而释放未使用的内存。

+   我们可以通过在缩小后添加一个元素来再次观察容量是如何表现的。

当你运行这个示例时，你将亲身体验大小和容量之间的差异以及 `std::vector` 在后台如何管理内存。

通过理解大小和容量之间的关系，你可以优化内存使用并预防潜在的性能陷阱。这为即将到来的章节奠定了基础，我们将讨论如何使用向量进行手动内存管理以及如何有效地遍历它们。

本节加深了我们对于 `std::vector` 的大小和容量的理解。我们将这些概念与火车的车厢进行了比较，强调了容量规划如何防止频繁、昂贵的重新分配，并导致更高效的内存使用程序。掌握这一点对于性能敏感和内存受限的环境至关重要。

基于此，我们将接下来查看 `resize()`、`reserve()` 和 `shrink_to_fit()`，学习如何主动管理 `std::vector` 的内存占用以实现最佳性能和内存使用。

# 调整大小和预留内存

在我们对 `std::vector` 的探索中，理解如何有效地管理其内存是至关重要的。向量的美在于其动态性；它可以增长和缩小，适应我们应用程序不断变化的需求。然而，这种灵活性也带来了确保高效内存利用的责任。本节深入探讨了让我们可以操作向量大小及其预分配内存的操作：`resize`、`reserve` 和 `shrink_to_fit`。

在处理向量时，我们已经看到它们的容量（预分配内存）可能与其实际大小（元素数量）不同。管理这些方面的方法可能会显著影响你程序的性能和内存占用。

## `resize()` 的力量

假设你有一个包含五个元素的 `std::vector`。如果你突然需要它来保持八个元素，或者可能只需要三个元素，你将如何进行这种调整？`resize()` 函数就是你的答案。

`resize()` 用于改变向量的大小。如果你增加其大小，新元素将被默认初始化。例如，对于 `std::vector<int>`，新元素将具有值 `0`。相反，如果你减少其大小，额外的元素将被丢弃。

但请记住，调整大小并不总是影响容量。如果你将向量扩展到其当前容量之外，容量将会增长（通常比大小增长更多，以适应未来的增长）。然而，缩小向量的大小并不会减少其容量。

让我们看看一个示例，该示例演示了如何手动调整 `std::vector` 实例的容量：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {1, 2, 3, 4, 5};
  auto printVectorDetails = [&]() {
    std::cout << "Vector elements: ";
    for (auto num : numbers) { std::cout << num << " "; }
    std::cout << "\nSize: " << numbers.size() << "\n";
    std::cout << "Capacity: " << numbers.capacity()
              << "\n";
  };
  std::cout << "Initial vector:\n";
  printVectorDetails();
  numbers.resize(8);
  std::cout << "After resizing to 8 elements:\n";
  printVectorDetails();
  numbers.resize(3);
  std::cout << "After resizing to 3 elements:\n";
  printVectorDetails();
  std::cout << "Reducing size doesn't affect capacity:\n";
  std::cout << "Capacity after resize: "
            << numbers.capacity() << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Initial vector:
Vector elements: 1 2 3 4 5
Size: 5
Capacity: 5
After resizing to 8 elements:
Vector elements: 1 2 3 4 5 0 0 0
Size: 8
Capacity: 10
After resizing to 3 elements:
Vector elements: 1 2 3
Size: 3
Capacity: 10
Reducing size doesn't affect capacity:
Capacity after resize: 10
```

在这个例子中，我们看到了以下情况：

+   我们从一个包含五个元素的 `std::vector<int>` 开始。

+   打印实用工具 `printVectorDetails` lambda 函数显示向量的元素、大小和容量。

+   我们将向量的大小调整为容纳八个元素，并观察这些变化。

+   然后，我们将向量的大小调整为仅包含三个元素，并观察大小如何减少，但容量保持不变。

这展示了 `resize()` 函数的力量以及它如何影响大小但并不总是影响 `std::vector` 的容量。

## 进入 `reserve()`

有时候，我们对数据有所了解。比如说你知道你将向向量中插入 100 个元素。让向量随着元素的添加逐步调整其容量将是不高效的。这时 `reserve()` 函数就派上用场了。

通过调用 `reserve()`，你可以预先为向量预留一定量的内存。这就像提前订票一样。大小保持不变，但容量调整为至少指定的值。如果你预留的内存少于当前容量，调用将没有效果；你不能使用 `reserve()` 减少容量。

让我们通过以下示例来演示 `reserve()` 函数的实用性：

```cpp
#include <chrono>
#include <iostream>
#include <vector>
int main() {
  constexpr size_t numberOfElements = 100'000;
  std::vector<int> numbers1;
  auto start1 = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < numberOfElements; ++i) {
    numbers1.push_back(i);
  }
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "Time without reserve: " << elapsed1.count()
            << " seconds\n";
  std::vector<int> numbers2;
  numbers2.reserve(
      numberOfElements); // Reserve memory upfront.
  auto start2 = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < numberOfElements; ++i) {
    numbers2.push_back(i);
  }
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed2 = end2 - start2;
  std::cout << "Time with reserve:    " << elapsed2.count()
            << " seconds\n";
  return 0;
}
```

下面是示例输出：

```cpp
Time without reserve: 0.01195 seconds
Time with reserve:    0.003685 seconds
```

从前面的示例中，我们了解到以下内容：

+   我们打算向两个向量中插入许多元素（`numberOfElements`）。

+   在第一个向量（`numbers1`）中，我们直接插入元素，而没有预先保留任何内存。

+   在第二个向量（`numbers2`）中，我们在插入元素之前使用 `reserve()` 函数预先分配内存。

+   我们测量并比较了两种情况下插入元素所需的时间。

当你运行代码时，你可能会注意到使用 `reserve()` 的插入时间更短（通常显著），因为它减少了内存重新分配的次数。这个例子有效地展示了合理使用 `reserve()` 的性能优势。在这个例子中，使用 `reserve()` 比不调用 `reserve()` 快了 3 倍以上。

合理使用 `reserve()` 可以显著提高性能，尤其是在处理大数据集时。预分配内存意味着更少的内存重新分配，从而加快插入速度。

## 使用 `shrink_to_fit()`

虽然 `reserve()` 允许你扩展预分配的内存，但如果你想要做相反的事情怎么办？如果你在多次操作后发现一个向量的尺寸为 `10` 但容量为 `1000`，保留额外的内存可能是浪费的。

`shrink_to_fit()` 函数允许你请求向量减少其容量以匹配其大小。注意这个词 *request*。实现可能并不总是保证减少容量，但在大多数情况下，它们会遵守。在大量删除后或向量增长阶段结束时回收内存是减少向量容量的绝佳方式。

让我们通过以下简单的代码示例来说明 `shrink_to_fit()` 的用法：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers;
  numbers.reserve(1000);
  std::cout << "Initial capacity: " << numbers.capacity()
            << "\n";
  for (auto i = 0; i < 10; ++i) { numbers.push_back(i); }
  std::cout << "Size after adding 10 elements: "
            << numbers.size() << "\n";
  std::cout << "Capacity after adding 10 elements: "
            << numbers.capacity() << "\n";
  numbers.shrink_to_fit();
  std::cout << "Size after shrink_to_fit: "
            << numbers.size() << "\n";
  std::cout << "Capacity after shrink_to_fit: "
            << numbers.capacity() << "\n";
  return 0;
}
```

下面是示例输出：

```cpp
Initial capacity: 1000
Size after adding 10 elements: 10
Capacity after adding 10 elements: 1000
Size after shrink_to_fit: 10
Capacity after shrink_to_fit: 10
```

以下是前面示例的关键要点：

+   我们从 `std::vector<int>` 开始，并为 1000 个元素预留内存。

+   我们只向向量中添加了 10 个元素。

+   在这个阶段，向量的尺寸是 10，但其容量是 1000。

+   然后，我们调用 `shrink_to_fit()` 来将向量的容量减少到与大小完全匹配。

+   我们在调用 `shrink_to_fit()` 后显示了大小和容量。

在运行代码后，你应该观察到向量的容量已经减少到接近其大小，这说明了该函数在回收内存方面的实用性。

## 现实世界的相关性

理解大小和容量之间的区别以及如何操作它们具有深远的意义。对于性能至关重要的应用程序，如实时系统或高频交易平台，有效地管理内存至关重要。同样，确保在嵌入式系统或内存有限的设备中每个字节都得到有效使用也是至关重要的。

虽然`std::vector`提供了处理数组的动态和高效方法，但要熟练运用它需要深入了解其内存行为。通过有效地使用`resize`、`reserve`和`shrink_to_fit`，开发者可以调整内存使用以满足应用程序的精确需求，在性能和资源消耗之间实现最佳平衡。

要精通 C++，必须不仅仅是一个程序员；必须像建筑师一样思考，理解手头的材料，并构建经得起时间和负载考验的结构。随着我们继续前进，我们将深入研究迭代方法，使我们更接近于掌握`std::vector`。

本节加深了我们对于`std::vector`内存分配技术的理解。我们学习了如何通过`reserve()`策略性地分配内存以优化性能，而`shrink_to_fit()`可以通过释放未使用的空间来最小化内存占用。这些策略对于开发者提高应用程序效率和明智地管理资源至关重要。

接下来，我们将探讨分配器在内存管理中的核心作用。我们将剖析分配器接口和可能需要自定义分配器的场景，评估它们与标准实践相比对性能和内存使用的影响。

# 自定义分配器基础

`std::vector`（以及许多其他 STL 容器）中动态内存管理的奥秘在于一个可能不会立即引起你注意的组件：分配器。在核心上，一个`std::vector`可以在不绑定到特定内存源或分配策略的情况下运行。

## 分配器的角色和责任

分配器是内存管理的无名英雄。它们处理内存块的分配和释放，从而确保我们的数据结构能够优雅地增长和收缩。在这些任务之外，分配器还可以构建和销毁对象。它们在原始内存操作和高级对象管理之间架起桥梁。

但为什么我们需要这样的抽象？为什么不直接使用`new`和`delete`操作呢？答案在于灵活性。STL 赋予开发者通过解耦容器与特定内存操作来实现自定义内存策略的能力。对于性能关键的应用，这种灵活性是一种福音。

## 内部机制——分配器接口

默认的 `std::allocator` 提供了与其职责紧密相关的成员函数。让我们简要地看看这些成员函数：

+   `allocate()`: 分配一个适合容纳指定数量对象的内存块

+   `deallocate()`: 将分配器之前分配给系统的内存块返回给系统

+   `construct()`: 在给定的内存位置构造一个对象

+   `destroy()`: 在给定的内存位置调用对象的析构函数

记住，虽然 `std::allocator` 默认使用堆进行内存操作，但分配器接口的真正威力在于自定义分配器发挥作用时。

为了展示 `std::allocator` 的好处，让我们首先说明一个简单的自定义分配器可能的样子。这个自定义分配器将跟踪并打印其操作，使我们能够可视化其交互。

然后，我们将在这个代码块中使用 `std::vector` 和这个自定义分配器：

```cpp
#include <iostream>
#include <memory>
#include <vector>
template <typename T> class CustomAllocator {
public:
  using value_type = T;
  CustomAllocator() noexcept {}
  template <typename U>
  CustomAllocator(const CustomAllocator<U> &) noexcept {}
  T *allocate(std::size_t n) {
    std::cout << "Allocating " << n << " objects of size "
              << sizeof(T) << " bytes.\n";
    return static_cast<T *>(::operator new(n * sizeof(T)));
  }
  void deallocate(T *p, std::size_t) noexcept {
    std::cout << "Deallocating memory.\n";
    ::operator delete(p);
  }
  template <typename U, typename... Args>
  void construct(U *p, Args &&...args) {
    std::cout << "Constructing object.\n";
    new (p) U(std::forward<Args>(args)...);
  }
  template <typename U> void destroy(U *p) {
    std::cout << "Destroying object.\n";
    p->~U();
  }
};
int main() {
  std::vector<int, CustomAllocator<int>> numbers;
  std::cout << "Pushing back numbers 1 to 5:\n";
  for (int i = 1; i <= 5; ++i) { numbers.push_back(i); }
  std::cout << "\nClearing the vector:\n";
  numbers.clear();
  return 0;
}
```

下面是示例输出：

```cpp
Pushing back numbers 1 to 5:
Allocating 1 objects of size 4 bytes.
Constructing object.
Allocating 2 objects of size 4 bytes.
Constructing object.
Constructing object.
Destroying object.
Deallocating memory.
Allocating 4 objects of size 4 bytes.
Constructing object.
Constructing object.
Constructing object.
Destroying object.
Destroying object.
Deallocating memory.
Constructing object.
Allocating 8 objects of size 4 bytes.
Constructing object.
Constructing object.
Constructing object.
Constructing object.
Constructing object.
Destroying object.
Destroying object.
Destroying object.
Destroying object.
Deallocating memory.
Clearing the vector:
Destroying object.
Destroying object.
Destroying object.
Destroying object.
Destroying object.
Deallocating memory.
```

以下是从前面的例子中得出的关键要点：

+   我们创建了一个简单的 `CustomAllocator`，当它执行特定的操作，如分配、释放、构造和销毁时，会打印消息。它使用全局的 `new` 和 `delete` 操作符进行内存操作。

+   `main()` 函数中的 `std::vector` 使用我们的 `CustomAllocator`。

+   当我们将元素推入向量时，你会注意到指示内存分配和对象构造的消息。

+   清除向量将触发对象销毁和内存释放消息。

使用我们的自定义分配器，我们为 `std::vector` 的内存管理操作添加了自定义行为（在这种情况下是打印）。这展示了分配器在 STL 中提供的灵活性以及它们如何针对特定需求进行定制。

## 权衡和自定义分配器的需求

你可能想知道，如果 `std::allocator` 可以直接使用，为什么还要费心使用自定义分配器？就像软件开发中的许多事情一样，答案归结为权衡。

默认分配器的一般性质确保了广泛的适用性。然而，这种万能的解决方案可能不是特定场景的最佳选择。例如，频繁分配和释放小块内存的应用程序如果使用默认分配器可能会遭受碎片化。

此外，某些上下文可能具有独特的内存约束，例如内存有限的嵌入式系统或对性能要求严格的实时系统。在这些情况下，自定义分配器提供的控制和优化变得非常有价值。

## 选择 `std::allocator` 而不是 `new`、`delete` 和托管指针

关于 C++中的内存管理，开发者有多种机制可供选择。虽然使用`new`和`delete`或甚至智能指针如`std::shared_ptr`和`std::unique_ptr`可能看起来直观，但当与 STL 容器一起工作时，依赖`std::allocator`有很强的理由。让我们来探讨这些优势。

### 与 STL 容器的一致性

STL 中的容器设计时考虑了分配器。使用`std::allocator`确保了库中的一致性和兼容性。它确保你的定制或优化可以统一地应用于各种容器。

### 内存抽象和定制

原始内存操作甚至管理指针并不能直接提供定制内存分配策略的途径。另一方面，`std::allocator`（及其可定制的兄弟）提供了一个抽象层，为定制内存管理方法铺平了道路。这意味着你可以实施对抗碎片化、使用**内存池**或利用专用硬件的策略。

### 集中化内存操作

使用原始指针和手动内存管理时，分配和释放操作散布在代码中。这种分散化可能导致错误和不一致性。`std::allocator`封装了这些操作，确保内存管理保持一致和可追踪。

### 防止常见陷阱的安全措施

使用`new`和`delete`进行手动内存管理容易产生内存泄漏、重复删除和未定义行为等问题。即使使用智能指针，循环引用也可能成为头疼的问题。当与容器一起使用时，分配器通过自动化底层内存过程来减轻许多这些担忧。

### 与高级 STL 功能的更好协同

STL 中某些高级功能和优化，如分配器感知容器，直接利用了分配器的功能。使用`std::allocator`（或自定义分配器）确保你更好地利用这些增强功能。

虽然`new`、`delete`和管理指针在 C++编程中有其位置，但在基于容器的内存管理方面，`std::allocator`是一个明显的选择。它提供了一种定制、安全和效率的混合体，这是手动或半手动内存管理技术难以实现的。在你探索 C++开发的丰富领域时，让分配器成为你在动态内存中的忠实伴侣。

本节探讨了分配器及其在管理`std::vector`内存中的作用。我们揭示了分配器如何为 STL 容器中的内存操作提供抽象，并考察了分配器接口的工作原理。这种理解对于制定能够提升各种环境下应用程序性能的内存管理策略至关重要。

接下来，我们将探讨实现自定义分配器，研究内存池，并指导您创建一个用于`std::vector`的自定义分配器，展示个性化内存管理的优势。

# 创建自定义分配器

创建自定义分配器是增强内存管理的战略决策。当默认的内存分配策略与特定应用程序的独特性能要求或内存使用模式不一致时，这种方法尤其有价值。通过设计自定义分配器，开发者可以微调内存分配和释放过程，可能提高效率，减少开销，并确保更好地控制其应用程序中资源的管理。这种程度的定制对于标准分配方案可能无法满足特定需求或优化性能的应用程序至关重要。

## 自定义分配器——内存灵活性的核心

当您考虑 STL 容器如何处理内存时，表面之下隐藏着一种潜在的力量。例如，`std::vector`这样的容器通过分配器来满足内存需求。默认情况下，它们使用`std::allocator`，这是一个适用于大多数任务的通用分配器。然而，在某些情况下，您可能需要更多控制内存分配和释放策略。这就是自定义分配器发挥作用的地方。

## 理解自定义分配器的动机

初看之下，人们可能会 wonder 为什么需要比默认分配器更多的东西。毕竟，那不是足够了吗？虽然`std::allocator`很灵活，但它旨在满足广泛的用例。特定情况需要特定的内存策略。以下是一些动机：

+   **性能优化**：不同的应用程序有不同的内存访问模式。例如，图形应用程序可能会频繁地分配和释放小块内存。自定义分配器可以针对这种模式进行优化。

+   **缓解内存碎片**：碎片化可能导致内存使用效率低下，尤其是在长时间运行的应用程序中。自定义分配器可以采用策略来减少或甚至防止碎片化。

+   **专用硬件或内存区域**：有时，应用程序可能需要从特定区域或甚至专用硬件（如**图形处理单元**（**GPU**）内存）分配内存。自定义分配器提供了这种灵活性。

## 内存池——一种流行的自定义分配器策略

在自定义内存分配中，一个广受欢迎的策略是内存池的概念。内存池预先分配一大块内存，然后根据应用程序的需求以较小的块形式分配。内存池的卓越之处在于其简单性和效率。以下是它们有益的原因：

+   **更快的分配和释放**：由于已经预先分配了一大块内存，因此分配较小的内存块是快速的。

+   **减少碎片化**：内存池通过控制内存布局并确保连续块自然地减少了碎片化。

+   **可预测的行为**：内存池可以提供一定程度的可预测性，这在性能至关重要的实时系统中特别有益。

## 解锁自定义分配器的潜力

虽然深入研究自定义分配器可能看起来令人畏惧，但它们的益处是实实在在的。无论是为了性能提升、内存优化还是特定应用需求，理解自定义分配器的潜力是 C++ 开发者工具箱中的一个宝贵资产。随着你继续使用 `std::vector` 的旅程，请记住，分配器在每一个元素之下勤奋地管理内存，以实现高效的内存管理。通过自定义分配器，你可以根据应用需求定制这种管理。

本节介绍了 `std::vector` 中自定义分配器的设计和使用，强调了它们如何允许专门的内存管理，这对于优化具有独特内存使用模式的程序至关重要。有了这个见解，开发者可以超越 STL 的默认机制，通过定制的分配策略（如内存池）来提高性能。

我们接下来将检查分配器对 STL 容器性能的影响，仔细审查 `std::allocator` 的特性，确定自定义替代方案的场景，并强调 **性能分析** 在明智地选择分配器中的作用。

# 分配器和容器性能

每个容器的效率都源于其内存管理策略，对于 `std::vector` 来说，分配器扮演着至关重要的角色。虽然内存分配可能看起来很简单，但分配器设计中的细微差别可以带来各种性能影响。

## 为什么分配器在性能中很重要

在我们能够利用分配器的潜力之前，我们需要了解为什么它们很重要。内存分配不是一个一刀切的操作。根据应用程序的具体需求，分配的频率、内存块的大小以及这些分配的生存期可能会有很大的差异。

+   **分配和释放的速度**：分配和释放内存所需的时间可以是一个重要的因素。一些分配器可能会为了速度而牺牲内存开销，而另一些分配器可能会做相反的事情。

+   **内存开销**：开销包括分配器用于簿记或碎片化的额外内存。低开销可能意味着更快的分配器，但可能导致更高的碎片化。相反，高开销的分配器可能较慢，但可能导致较低的碎片化。

+   **内存访问模式**：内存的访问方式可以影响缓存性能。确保连续内存分配的分配器可以导致更好的缓存局部性，从而提高性能。

## std::allocator 的性能特性

默认的`std::allocator`旨在为通用情况提供平衡的性能。它是一个多面手，但可能不是特定用例的专家。以下是你可以期待的内容：

+   **通用效率**：它在各种场景中表现良好，使其成为许多应用的可靠选择

+   **低开销**：虽然开销最小，但内存碎片化风险较大，尤其是在频繁分配和释放不同大小的内存的场景中

+   **一致的行为**：由于它是标准库的一部分，其行为和性能在不同平台和编译器之间是一致的

## 考虑替代分配器的时机

由于`std::allocator`是一个可靠的通用选择，那么在什么情况下应该考虑替代方案？以下是一些突出的场景：

+   **特定工作负载**：如果你知道你的应用程序主要频繁分配小块内存，基于内存池的分配器可能更有效率

+   **实时系统**：对于具有可预测性能的系统，针对应用程序需求定制的自定义分配器可以产生影响

+   **硬件限制**：如果你在一个具有有限或专用内存的环境中工作，可以设计定制的分配器来适应这些限制

## 性能分析——做出明智决策的关键

虽然理解分配器性能的理论方面是有益的，但没有实际性能分析是无法替代的。使用不同的分配器测量应用程序的性能是最可靠的方法来确定最佳匹配。例如，Valgrind 或平台特定的分析工具可以提供有关内存使用模式、分配时间和碎片化的见解。

尽管内存管理通常在幕后，但它是高效 C++编程的基石。分配器作为默默无闻的英雄，提供了一种精细调整这一方面的方法。虽然`std::vector`提供了出色的通用性和性能，但了解分配器的角色和潜力使开发者能够将他们的应用程序提升到新的性能高度。随着我们结束这一章节，请记住，虽然理论提供方向，但性能分析提供清晰度。

在本节中，我们探讨了分配器如何影响`std::vector`的性能。我们发现分配器选择对容器效率有显著影响，并了解了 C++ STL 中的默认`std::allocator`，包括在某些情况下替代分配器可能更可取的场景。

这种知识使我们能够根据特定的性能需求定制容器的内存管理，确保我们的应用程序运行得更高效。

# 概述

在本章中，我们彻底考察了内存管理与`std::vector`使用之间的关系。我们首先回顾了容量与大小的基本概念，强调了它们各自的作用以及这种区别对于高效内存使用的重要性。然后探讨了`std::vector`容器内存分配的机制，阐明了当向量增长或缩小时内部发生的情况。

我们讨论了调整大小和预留内存的细微差别，介绍了`reserve()`和`shrink_to_fit()`等函数作为优化内存使用的工具。这些方法在实际应用中的相关性得到了强调，突出了它们在高性能应用中的实用性。

本章介绍了自定义分配器的基础知识，详细阐述了它们的作用并深入探讨了分配器接口。我们讨论了权衡利弊，并说明了为什么自定义分配器可能比直接使用`new`、`delete`和托管指针更可取。为`std::vector`创建和实现自定义内存池分配器演示了自定义分配器如何释放更大的内存灵活性。

最后，我们分析了分配器对容器性能的影响，详细说明了为什么分配器是性能调优的重要考虑因素。我们涵盖了`std::allocator`的性能特性，并讨论了何时应考虑使用替代分配器。性能分析被提出作为做出关于分配器使用的明智决策的关键。

本章的见解是无价的，为我们提供了掌握`std::vector`内存管理的复杂技术。这种知识使我们能够编写高性能的 C++应用程序，因为它允许对内存分配进行细粒度控制，这在内存约束严格或需要快速分配和释放周期的环境中尤为重要。

接下来，我们将关注在向量上操作的计算算法。我们将探索排序技术、搜索操作以及向量内容的操作，强调理解这些算法的效率和多功能性的重要性。我们将讨论使用自定义比较器和谓词以及它们如何被利用来对用户定义的数据类型执行复杂操作。下一章还将提供有关维护容器不变性和管理迭代器失效的指导，这对于确保在多线程场景中的健壮性和正确性至关重要。
