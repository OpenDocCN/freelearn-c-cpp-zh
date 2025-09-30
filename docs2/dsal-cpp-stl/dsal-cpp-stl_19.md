

# 第十九章：异常安全性

本章将引导你了解异常安全性的复杂性。它揭示了异常安全性的级别，区分了基本和强保证，强调了它们的重要性，并提供了实现它们的经过验证的策略。掌握这些高级主题使你能够创建更健壮、高效和适应性强的高性能 C++应用程序和数据结构。

在本章中，我们将涵盖以下主题：

+   基本的异常安全性

+   强异常安全性

+   `noexcept` 对 STL 容器的影响

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# 基本的异常安全性

**基本的异常安全性**，俗称为**保证**，承诺当发生异常且其不变性得到保留时，你的程序不会泄露资源。简单来说，软件不会陷入混乱。当发生未预见的异常时，操作可能会失败，但你的应用程序将继续运行，并且没有数据被损坏。

以下两个现实世界的例子展示了可以有效地管理未预见的异常，而不会导致资源泄露或数据损坏：

+   **数据处理过程中的文件操作失败**：考虑一个处理大型数据文件的应用程序。在这个过程中，应用程序可能会遇到意外的异常，例如由于磁盘 I/O 错误而无法读取文件的一部分。在这种情况下，基本的异常安全性确保应用程序不会泄露资源（如文件句柄或为数据处理分配的内存）。它维护任何涉及的数据结构的完整性。应用程序可能无法完成预期的文件处理。然而，它将优雅地处理异常，释放任何资源，并使应用程序处于稳定状态以继续运行。

+   **客户端-服务器应用程序中的网络通信中断**：在一个客户端-服务器应用程序中，如果在关键数据交换过程中网络连接突然丢失，可能会发生意外的异常。在这种情况下，基本的异常安全性确保应用程序不会最终处于部分或损坏的数据状态。系统可能无法完成当前操作（如更新记录或检索数据），但它将有效地管理资源，如网络套接字和内存缓冲区。应用程序将捕获异常，清理资源，并确保其核心功能保持完整并准备好进行后续操作。

## 程序不变量在 STL 中的关键作用

想象你正在构建一个复杂的应用程序，其核心是 C++的`std::vector`、`std::map`或其他，它们在特定的不变性下运行。例如，`std::vector`容器保证连续的内存。如果任何操作破坏了这些不变性，结果可能从性能损失到隐秘的 bug。

为了确保 STL 的基本异常安全性，你需要确保对这些容器的操作要么成功，要么在抛出异常时，不违反其不变性，使容器保持其原始状态。例如，如果`std::vector`上的`push_back`操作抛出异常，向量应该保持不变。

让我们看看如何使用基本异常安全性将数据推入`std::vector`的例子：

```cpp
// Adds an element to the vector, ensuring basic exception
// safety
void safePushBack(std::vector<int> &vec, int value) {
  try {
    // Attempt to add value to the vector
    vec.push_back(value);
  } catch (const std::exception &e) {
    // Handle any exception thrown by push_back
    std::cerr << "Exception caught: " << e.what() << "\n";
    // No additional action needed, vec is already in its
    // original state
  }
}
```

在这个例子中，如果发生异常（例如，由于系统内存不足导致的`bad_alloc`），`catch`块将处理它。重要的是，如果`push_back`抛出异常，它保证向量的状态（`vec`）保持不变，从而保持容器的不变性。

## 资源完整性——稳健软件的守护者

如果在内存分配或其他资源密集型任务期间抛出异常，如果没有正确管理，可能会造成灾难。然而，STL 提供了工具，当适当使用时，确保资源保持完整，即使异常即将发生。

STL 容器，如`std::vector`和`std::string`，处理它们的内存。如果在操作期间出现异常，容器确保不会发生内存泄漏。此外，**资源获取即初始化**（**RAII**），C++设计的标志，确保资源在对象创建时获取，并在它们超出作用域时释放。RAII 原则是防止资源泄漏的哨兵，尤其是在异常期间。

注意

RAII 是 C++中用于管理资源分配和释放的编程习惯。在 RAII 中，资源（如内存、文件句柄和网络连接）由对象获取和释放。当对象创建（初始化）时，它获取资源，当对象销毁（其生命周期结束）时，它释放资源。这确保了自动和异常安全的资源管理，防止资源泄漏，即使在面对异常的情况下也能确保资源干净释放。RAII 是 C++中有效资源管理的基本概念。

## 利用 STL 实现基本异常安全性

在拥有 STL 及其复杂性的知识后，实现基本异常安全性变得不那么令人畏惧。考虑以下最佳实践：

+   **利用复制和交换习惯用法**：在修改 STL 容器时，确保异常安全的一种常见技术是创建容器的副本，在副本上执行操作，然后与原始内容交换。如果出现异常，原始容器不受影响。

+   `std::shared_ptr`和`std::unique_ptr`不仅管理内存，而且在异常期间保证没有泄漏。

+   **受保护的操作**：在 STL 容器上进行任何不可逆操作之前，始终确保任何可能抛出异常的操作已经执行。

+   **通过 STL 文档保持信息更新**：熟悉 STL 函数和方法的异常保证。了解特定 STL 函数可能抛出的异常有助于构建健壮的软件。

使用 STL 拥抱基本的异常安全性为构建更具有弹性、可靠和健壮的软件奠定了基础。有了这种基础理解，你将能够应对 STL 的复杂性，确保即使遇到意外情况，你的软件也能坚定不移。但这只是开始，因为下一个层次，强大的异常安全性，在召唤，提供更多稳健的保证和策略，以优雅地运用 STL。

# 强大的异常安全性

当你进一步沉浸在 C++和 STL 错综复杂的领域中时，你会遇到术语*强大的异常安全性*。这不仅是一句华丽的辞藻，而且是 STL 异常处理的黄金标准。它向开发者提供了一种前所未有的保证——操作要么成功完成，要么在没有副作用的情况下恢复到之前的状态。这就像有一个安全网，确保无论发生什么情况，你的应用程序的完整性都完好无损。

## 带着强大保证导航 STL 容器

记得那些与`std::vector`、`std::map`和其他 STL 容器共度的动态日子吗？现在，想象一下添加元素、调整大小，甚至修改它们。当这些操作成功时，一切照常进行。但如果它们失败并抛出异常，强大的异常安全性保证容器保持原样，未受影响且未更改。

幸运的是，使用 STL 容器实现这一点并不需要超人的努力。许多 STL 容器操作自然提供强大的异常安全性。但当他们不提供时，像*复制和交换*这样的技巧可以拯救它们。通过在副本上操作，并在确定成功后才将内容与原始内容交换，你可以保证如果抛出异常，原始容器不会发生变化。

## 带着强大保证定制 STL 容器

当你进入创建自定义 STL 容器的领域时，确保强大异常安全性的责任就完全落在了你的肩上。实现这一目标的关键策略包括以下实践：

+   **本地化提交点**：通过将影响容器状态的任何更改推迟到最后时刻，并确保一旦开始这些更改就无异常，你可以巩固强大的保证。

+   **RAII 的重要性**：利用 RAII 的力量，特别是与资源管理相结合，至关重要。这确保了资源得到适当的处理，如果发生异常，容器保持不变。

+   **不可变操作**：尽可能设计不修改容器直到成功为止的操作。

为了说明创建具有强保证的自定义 STL 容器的概念，让我们考虑一个管理动态数组的自定义容器的例子。代码将演示局部提交点、RAII 习语和不可变操作，以提供强异常安全性。

首先，我们将创建 `CustomArray` 类。`CustomArray` 类是一个模板类，旨在管理指定数据类型 `T` 的动态数组。它提供了创建、复制、移动和管理动态数组的基本功能，并具有强异常保证。该类使用 RAII 原则，并利用 `std::unique_ptr` 进行资源管理，确保高效且安全的内存处理。它支持复制和移动语义，使其适用于各种场景，如动态数组操作和容器重新分配。让我们分部分来探讨这个问题。

我们将把这个例子分成几个部分在这里讨论。对于完整的代码示例，请参阅 GitHub 仓库。首先，我们将查看构造函数：

```cpp
template <typename T> class CustomArray {
public:
  explicit CustomArray(size_t size)
      : size(size), data(std::make_unique<T[]>(size)) {
    // Initialize with default values, assuming T can be
    // default constructed safely std::fill provides strong
    // guarantee
    std::fill(data.get(), data.get() + size, T());
  }
  // Copy constructor
  CustomArray(const CustomArray &other)
      : size(other.size),
        data(std::make_unique<T[]>(other.size)) {
    safeCopy(data.get(), other.data.get(), size);
  }
  // Move constructor - noexcept for strong guarantee
  // during container reallocation
  CustomArray(CustomArray &&other) noexcept
      : size(other.size), data(std::move(other.data)) {
    other.size = 0;
  }
  void safeCopy(T *destination, T *source, size_t size) {
    // std::copy provides strong guarantee
    std::copy(source, source + size, destination);
  }
```

我们为我们的类提供了三个构造函数：

+   `explicit CustomArray(size_t size)`: 这是 `CustomArray` 类的主要构造函数。它允许您通过指定动态数组的大小来创建类的实例。它将 `size` 成员变量初始化为提供的大小，并使用 `std::make_unique` 为动态数组分配内存。它还使用 `std::fill` 初始化数组的元素为默认值（假设类型 `T` 可以安全地进行默认构造），此构造函数被标记为 `explicit`，意味着它不能用于隐式类型转换。

+   `CustomArray(const CustomArray &other)`: 这是 `CustomArray` 类的复制构造函数。它允许您创建一个新的 `CustomArray` 对象，该对象是现有 `CustomArray` 对象 `other` 的副本。它将 `size` 成员初始化为 `other` 的大小，为动态数组分配内存，然后使用 `safeCopy` 函数从 `other` 到新对象执行深拷贝。当您想要创建现有对象的副本时，将使用此构造函数。

+   `CustomArray(CustomArray &&other)noexcept`：这是 `CustomArray` 类的移动构造函数。它使你能够高效地将数据的所有权从一 `CustomArray` 对象（通常是 `rvalue`）转移到另一个对象。它使用 `std::move` 将动态分配的数组所有权从 `other` 转移到当前对象，更新 `size` 成员，并将 `other` 的 `size` 设置为零，以表示它不再拥有数据。此构造函数标记为 `noexcept`，以确保在容器重新分配期间提供强保证，意味着它不会抛出异常。它用于将一个对象的内容移动到另一个对象中，通常用于优化目的。

接下来，让我们看看赋值运算符的重载：

```cpp
  // Copy assignment operator
  CustomArray &operator=(const CustomArray &other) {
    if (this != &other) {
      std::unique_ptr<T[]> newData(
          std::make_unique<T[]>(other.size));
      safeCopy(newData.get(), other.data.get(),
               other.size);
      size = other.size;
      data = std::move(
          newData); // Commit point, only change state here
    }
    return *this;
  }
  // Move assignment operator - noexcept for strong
  // guarantee during container reallocation
  CustomArray &operator=(CustomArray &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
      size = other.size;
      other.size = 0;
    }
    return *this;
  }
```

在这里，我们提供了赋值运算符的两个重载。这两个成员函数是 `CustomArray` 类的赋值运算符：

+   `CustomArray &operator=(const CustomArray &other)`：这是复制赋值运算符。它允许你将一个 `CustomArray` 对象的内容赋值给另一个相同类型的对象。它从 `other` 到当前对象执行数据的深度复制，确保两个对象都有数据的独立副本。它还更新 `size` 成员，并使用 `std::move` 转移新数据的所有权。运算符返回当前对象的引用，允许链式赋值。

+   `CustomArray &operator=(CustomArray &&other) noexcept`：这是移动赋值运算符。它允许你高效地将数据的所有权从一 `CustomArray` 对象（通常是 `rvalue`）转移到另一个对象。它将包含数据的 `std::unique_ptr` 从 `other` 移动到当前对象，更新 `size` 成员，并将 `other` 的 `size` 设置为零，以表示它不再拥有数据。此运算符标记为 `noexcept`，以确保在容器重新分配期间提供强保证，意味着它不会抛出异常。像复制赋值运算符一样，它返回当前对象的引用：

```cpp
int main() {
  try {
    // CustomArray managing an array of 5 integers
    CustomArray<int> arr(5);
    // ... Use the array
  } catch (const std::exception &e) {
    std::cerr << "An exception occurred: " << e.what()
              << '\n';
    // CustomArray destructor will clean up resources if an
    // exception occurs
  }
  return 0;
}
```

总结这个例子，`CustomArray` 类展示了以下原则：

+   `data`) 只在提交点改变，例如在复制赋值运算符的末尾，在所有可能抛出异常的操作成功之后。

+   `std::unique_ptr` 管理动态数组，确保当 `CustomArray` 对象超出作用域或发生异常时，内存会自动释放。

+   **不可变操作**：可能抛出异常的操作，如内存分配和复制，是在临时对象上执行的。只有当这些操作保证成功时，容器状态才会被修改。

此示例遵循 C++ 和 STL 最佳实践，并使用现代 C++ 功能，确保自定义容器遵守强异常安全性保证。

## 将异常安全性引入自定义 STL 算法

算法与数据和谐共舞。在 STL 中，确保自定义算法提供强异常安全性保证可能是高效应用程序和充满不可预测行为的应用程序之间的区别。

为了确保这一点，你应该牢记以下几点：

+   **操作副本**：在可能的情况下，操作数据的副本，确保如果抛出异常，原始数据保持未修改。

+   **原子操作**：设计算法，其中操作一旦开始，要么成功完成，要么可以无副作用地回滚。

## 异常安全性是构建健壮应用程序的途径

强异常安全性不仅仅是一个原则——它是对您应用程序可靠性和健壮性的承诺。当使用 STL、其容器和其算法或尝试创建自己的算法时，这一保证就像一道防线，抵御未预见的异常和不可预测的行为。

通过确保操作要么看到其成功完成，要么恢复到原始状态，强异常安全性不仅提高了应用程序的可靠性，而且使开发者对其软件能够经受异常风暴的考验充满信心，保持其数据和资源的完整性。

有了这一点，我们就结束了我们对 STL 中异常安全性的探索。在我们探讨了基本和强保证之后，希望你现在已经具备了构建健壮和可靠的 C++ 应用程序的知识和工具。并且记住，在软件开发的动态世界中，不仅是要防止异常，还要确保我们准备好应对它们的出现。接下来，我们将检查 STL 操作中使用 `noexcept` 的情况。

# noexcept 对 STL 操作的影响

C++ STL 提供了丰富的数据结构和算法，极大地简化了 C++ 的编程。异常安全性是健壮 C++ 编程的关键方面，`noexcept` 指定符在实现它方面发挥着关键作用。本节阐述了 noexcept 对 STL 操作的影响以及其正确应用如何提高基于 STL 的代码的可靠性和性能。

## noexcept 简介

在 C++11 中引入的 `noexcept` 是一个可以添加到函数声明中的指定符，表示该函数不期望抛出异常。当一个函数用 `noexcept` 声明时，它启用特定的优化并确保异常处理更加可预测。例如，当从 `noexcept` 函数抛出异常时，程序会调用 `std::terminate`，因为该函数违反了其不抛出异常的合同。因此，`noexcept` 是一个函数承诺遵守的承诺。

## 应用于 STL 数据类型

在 STL 数据类型中使用`noexcept`主要影响移动操作——移动构造函数和移动赋值运算符。这些操作对于 STL 容器的性能至关重要，因为它们允许在不进行昂贵的深拷贝的情况下将资源从一个对象转移到另一个对象。当这些操作是`noexcept`时，STL 容器可以安全地进行优化，例如在调整大小操作期间更有效地重新分配缓冲区。

考虑一个使用`std::vector`的场景，这是一个 STL 容器，它会在添加元素时动态调整大小。假设向量包含的对象类型具有`noexcept`的移动构造函数。在这种情况下，向量可以通过将对象移动到新数组来重新分配其内部数组，而无需处理潜在异常的开销。如果移动构造函数不是`noexcept`，则向量必须使用复制构造函数，这效率较低，可能会抛出异常，导致潜在的部分状态和强异常安全性的损失。

## 应用于 STL 算法

`noexcept`的影响不仅限于数据类型，还扩展到算法。当与`noexcept`函数一起工作时，STL 算法可以提供更强的保证并表现出更好的性能。例如，如果`std::sort`的比较函数不抛出异常，则它可以更有效地执行。算法可以优化其实施，因为它知道它不需要考虑由异常处理引起的复杂情况。

让我们以`std::for_each`算法为例，该算法将一个函数应用于一系列元素。如果使用的函数被标记为`noexcept`，则`std::for_each`可以在理解异常不会中断迭代的情况下操作。这可以导致更好的内联和减少开销，因为编译器不需要生成额外的代码来处理异常。

考虑以下示例：

```cpp
std::vector<int> data{1, 2, 3, 4, 5};
std::for_each(data.begin(), data.end(), [](int& value) noexcept {
    value *= 2;
});
```

在这个例子中，传递给`std::for_each`的 lambda 函数被声明为`noexcept`。这通知编译器和算法该函数保证不会抛出任何异常，从而允许潜在的性能优化。

`noexcept`指定符是 C++开发者的一项强大工具，它提供了性能优化和关于异常安全的语义保证。当明智地应用于 STL 操作时，`noexcept`使 STL 容器和算法能够更高效、更可靠地操作。对于希望编写高质量、异常安全代码的中级 C++开发者来说，理解和适当地使用`noexcept`是至关重要的。

# 概述

在本章中，我们试图通过 STL 来理解异常安全性的关键概念。我们探讨了不同级别的异常安全性，即基本和强保证，并概述了确保您的程序能够抵抗异常的策略。我们通过详细的讨论学习了如何维护程序不变性和资源完整性，主要关注 RAII 原则和受保护操作，以防止资源泄露并在异常期间保持容器状态。

理解异常安全性对于编写健壮的 C++应用程序至关重要。它确保即使在出现错误的情况下，您的软件的完整性保持完好，防止资源泄露并保持数据结构的有效性。这种知识是可靠和可维护代码的基石，因为它使我们能够保持强有力的保证，即我们的应用程序在异常情况下将表现出可预测的行为。

在下一章中，标题为《使用 STL 的线程安全和并发》，我们将基于异常安全性的基础来处理 C++中并发编程的复杂性。
