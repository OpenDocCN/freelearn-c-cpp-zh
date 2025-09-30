# 1

# std::vector 的基本原理

`std::vector`是 C++编程的一个基本组成部分。本章将探讨`std::vector`作为一个动态数组，讨论其在各种编程环境中的实用性。到本章结束时，你应该能够熟练地声明、初始化和操作向量。这些技能将使你能够在各种应用中有效地使用`std::vector`。它将为理解更广泛的数据结构和算法的**标准模板库**（**STL**）打下坚实的基础。

在本章中，我们将涵盖以下主要内容：

+   `std::vector`的重要性

+   声明和初始化`std::vector`

+   访问元素

+   添加和删除元素

# 技术要求

本章中的代码可以在 GitHub 上找到：

[`github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL`](https://github.com/PacktPublishing/Data-Structures-and-Algorithms-with-the-CPP-STL)

# `std::vector`的重要性

在 C++中，`std::vector`是一个常用的数据结构。虽然初学者可能会将其与 C 中的基本数组看到相似之处，但随着更深入的探索，`std::vector`的优势变得明显。此外，对`std::vector`的牢固掌握有助于更顺利地过渡到理解 STL 的其他组件。

向量和数组都作为元素集合的容器。它们之间的关键区别在于它们的灵活性和功能。数组在大小上是静态的，在声明时设置，之后不能更改。

相比之下，向量是动态的。它们可以根据对它们的操作进行扩展或收缩。与在声明时承诺一个固定内存块的数组不同，向量动态管理内存。它们经常分配额外的内存来预测未来的增长，优化效率和灵活性。虽然数组提供简单的基于索引的元素访问和修改，但向量提供更广泛的功能，包括插入、删除和定位元素的方法。

`std::vector`的主要优势是其动态调整大小和优化性能的结合。传统的 C++数组在编译时设置其大小。如果一个数组被声明为包含 10 个元素，它将受到该容量的限制。然而，在许多实际场景中，数据的量直到运行时才能确定。这正是`std::vector`大放异彩的地方。

## C 风格数组和 std::vector 的基本比较

作为动态数组，`std::vector`可以在程序执行期间调整其大小。它高效地管理其内存，不是为每个新添加的元素重新分配，而是在较大的块中重新分配，以保持性能和适应性的平衡。因此，`std::vector`动态地响应不断变化的数据需求。

这里有两个代码示例，展示了使用 C 风格数组和 `std::vector` 之间的对比。

以下代码演示了 C 风格数组的使用：

```cpp
#include <iostream>
int main() {
  int *cArray = new int[5];
  for (int i = 0; i < 5; ++i) { cArray[i] = i + 1; }
  for (int i = 0; i < 5; ++i) {
    std::cout << cArray[i] << " ";
  }
  std::cout << "\n";
  const int newSize = 7;
  int *newCArray = new int[newSize];
  for (int i = 0; i < 5; ++i) { newCArray[i] = cArray[i]; }
  delete[] cArray;
  cArray = newCArray;
  for (int i = 0; i < newSize; ++i) {
    std::cout << cArray[i] << " ";
  }
  std::cout << "\n";
  int arraySize = newSize;
  std::cout << "Size of cArray: " << arraySize << "\n";
  delete[] cArray;
  return 0;
}
```

这里是示例输出：

```cpp
1 2 3 4 5
1 2 3 4 5 0 0
Size of cArray: 7
```

在这个例子中，我们执行以下操作：

1.  声明一个大小为 `5` 的 C 风格动态数组。

1.  初始化动态数组。

1.  打印数组的内容。

1.  将数组调整到新大小（例如，`7`）。

1.  将旧数组中的元素复制到新数组中。

1.  释放旧数组。

1.  更新指针到新数组。

1.  打印调整大小后的数组的内容。

1.  获取调整大小后数组的大小。

1.  完成后，释放调整大小后的数组。

相比之下，以下代码演示了 `std::vector` 的使用：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> stlVector = {1, 2, 3, 4, 5};
  for (const int val : stlVector) {
    std::cout << val << " ";
  }
  std::cout << "\n";
  stlVector.resize(7);
  for (const int val : stlVector) {
    std::cout << val << " ";
  }
  std::cout << "\n";
  std::cout << "Size of stlVector: " << stlVector.size()
            << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
1 2 3 4 5
1 2 3 4 5 0 0
Size of stlVector: 7
```

与 C 风格版本进行对比，在这个例子中，我们执行以下操作：

1.  声明具有初始值的 `std::vector`。

1.  打印向量的内容。

1.  调整大小。使用 `std::vector` 进行此操作很容易。

1.  再次打印以查看变化。

1.  获取大小。这个操作通过 `size()` 成员函数很简单。

在初始示例中，C 风格数组受其固定大小的限制。修改其大小通常需要非平凡的程序。相反，`std::vector` 可以轻松调整其大小，并提供一个 `size()` 方法来确定它包含的元素数量。

除了其动态调整大小的能力之外，`std::vector` 与传统数组相比，进一步简化了内存管理。使用 `std::vector`，不需要显式进行内存分配或释放，因为它内部处理这些任务。这种方法最小化了内存泄漏的风险，并简化了开发过程。因此，许多 C++ 开发者，无论经验水平如何，都更喜欢使用 `std::vector` 而不是原始数组，以方便和安全。

让我们看看一个示例，对比传统的 C 风格数组如何管理内存，以及 `std::vector` 如何使这一过程更简单、更安全。

## C 风格数组和 `std::vector` 在内存管理方面的比较

首先，让我们考虑一个具有手动内存管理的 C 风格数组的例子。

在这个例子中，我们将使用动态内存分配（`new` 和 `delete`）来模拟 `std::vector` 的一些调整大小功能：

```cpp
#include <iostream>
int main() {
  int *cArray = new int[5];
  for (int i = 0; i < 5; ++i) { cArray[i] = i + 1; }
  int *temp = new int[10];
  for (int i = 0; i < 5; ++i) { temp[i] = cArray[i]; }
  delete[] cArray; // Important: free the old memory
  cArray = temp;
  for (int i = 5; i < 10; ++i) { cArray[i] = i + 1; }
  for (int i = 0; i < 10; ++i) {
    std::cout << cArray[i] << " ";
  }
  std::cout << "\n";
  delete[] cArray;
  return 0;
}
```

这里是示例输出：

```cpp
1 2 3 4 5 6 7 8 9 10
```

在这个例子中，我们执行以下操作：

1.  动态分配一个大小为 `5` 的 C 风格数组。

1.  填充数组。

1.  模拟调整大小：分配一个更大的数组并复制数据。

1.  填充新数组的其余部分。

1.  打印数组的内容。

1.  清理分配的内存。

现在，让我们考虑一个具有内置内存管理的 `std::vector` 的例子。

使用 `std::vector`，您不需要手动分配或释放内存；它由内部管理：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> myVector(5);
  for (int i = 0; i < 5; ++i) { myVector[i] = i + 1; }
  for (int i = 5; i < 10; ++i) {
    myVector.push_back(i + 1);
  }
  for (int val : myVector) { std::cout << val << " "; }
  std::cout << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
1 2 3 4 5 6 7 8 9 10
```

这个例子中的步骤包括以下内容：

1.  创建一个大小为 `5` 的 `std::vector`。

1.  填充向量。

1.  使用 `push_back()` 或 `resize()` 进行调整大小非常直接。

1.  打印向量的内容。

    没有必要进行显式的内存释放。

在第一个例子中，手动内存管理的挑战显而易见。未能适当地使用 `delete` 可能会导致内存泄漏。另一方面，第二个例子突出了 `std::vector` 的效率，它内部管理内存，消除了手动调整大小和内存操作的需求，并提高了开发过程。

传统数组提供了一套基本操作。相比之下，`std::vector` 提供了各种成员函数，这些函数提供了高级的数据操作和检索能力。这些函数将在后续章节中探讨。

在 C++ 开发中，`std::vector` 是一个基本工具。其灵活性使其成为从游戏开发到复杂软件项目等各种应用的优选选择。内置的防止常见内存问题的安全机制凸显了其价值。作为一个 STL 组件，`std::vector` 通过与其他 STL 元素的良好集成，鼓励一致的、最优的编码实践。

本节探讨了 C 风格数组和 `std::vector` 之间的基本区别。与静态的 C 风格数组不同，我们了解到 `std::vector` 提供了动态调整大小和强大的内存管理，这对于开发灵活和高效的应用程序至关重要。比较详细地说明了 `std::vector` 如何抽象出低级内存处理，从而最小化与手动内存管理相关的常见错误。

理解 `std::vector` 是有益的，因为它是在 C++ 编程中最广泛使用的序列容器之一。`std::vector` 支持在连续分配的内存中动态增长，支持随机访问迭代，并且与 STL 中的多种算法兼容。我们还讨论了 `std::vector` 如何提供更安全、更直观的接口来管理对象集合。

以下章节将在此基础上构建。我们将学习声明 `std::vector` 的语法以及初始化它的各种方法。这包括对默认、复制和移动语义的考察，这些语义与向量相关。

# 声明和初始化 std::vector

在 C++ 开发中建立了 `std::vector` 的基础知识后，是时候深入探讨其实际应用了——具体来说，是如何声明和初始化向量的。

`std::vector` 的本质在于其动态性。与固定大小的传统数组不同，向量可以根据需要增长或缩小，这使得它们成为开发者手中多才多艺的工具。

## 声明一个向量

`std::vector`的性能源于其设计，它结合了连续内存布局（如数组）的优点和动态调整大小的灵活性。当以指定的大小初始化时，它会预留足够的内存来存储这些元素。但如果向量填满并且需要更多容量，它会分配一个更大的内存块，转移现有元素，并释放旧内存。这个动态调整大小的过程被优化以减少开销，确保向量保持高效。连续存储和自动内存管理的结合使`std::vector`成为 C++生态系统中的基本组件。

要声明一个基本的`std::vector`，请使用以下：

```cpp
std::vector<int> vec;
```

这行代码初始化了一个名为`vec`的空`std::vector`，专门设计用来存储`int`类型的值。（`int`是`std::vector`类型模板参数`<>`内的内容。）`std::vector`是一个动态数组，这意味着尽管`vec`的初始大小为`0`，但其容量可以根据需要增长。当你向`vec`中插入整数时，容器将自动分配内存以适应元素数量的增加。这种动态调整大小使得`std::vector`成为 C++中一种多用途且广泛使用的容器，适用于元素数量事先未知或可能随时间变化的情况。

当创建`std::vector`时，可以指定其初始大小。如果你事先知道需要存储的元素数量，这可能会很有益：

```cpp
std::vector<int> vec(10);
```

在前面的代码中，名为`vec`的`std::vector`被初始化为有 10 个整数的空间。默认情况下，这些整数将被值初始化，这意味着对于`int`这样的基本数据类型，它们将被设置为`0`。

如果你希望使用特定的值初始化元素，可以在构造向量时提供第二个参数：

```cpp
std::vector<int> vec(10, 5);
```

在这里，使用 10 个整数声明了`std::vector`，并且这 10 个整数都被初始化为`5`的值。这种方法确保了在单步中高效地分配内存和初始化所需值。

## 初始化向量

在 C++11 及以后的版本中，随着初始化列表的引入，`std::vector`的初始化变得更加简单。这允许开发者在花括号内直接指定向量的初始值：

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
```

前面的语句创建了一个名为`vec`的`std::vector`实例，并使用五个整数对其进行初始化。这种方法提供了一种简洁的方式来声明和填充向量，这是初始化`std::vector`的一种方法。根据你的需求，还有许多其他方法可以实现这一点：

```cpp
// Method 1: Declare a vector and then add elements using
// push_back (Add integers from 0 to 4)
std::vector<int> vec1;
for (int i = 0; i < 5; ++i) { vec1.push_back(i); }
// Method 2: Initialize a vector with a specific size and
// default value (5 elements with the value 10)
std::vector<int> vec2(5, 10);
// Method 3: List initialization with braced initializers
// Initialize with a list of integers
std::vector<int> vec3 = {1, 2, 3, 4, 5};
// Method 4: Initialize a vector using the fill
// constructor Default-initializes the five elements (with
// zeros)
std::vector<int> vec4(5);
// Method 5: Using std::generate with a lambda function
std::vector<int> vec5(5);
int value = 0;
std::generate(vec5.begin(), vec5.end(),
              [&value]() { return value++; });
```

`std::vector`是一个多功能的模板容器，能够存储各种数据类型，而不仅仅是像`int`这样的原始类型。你可以存储自定义类的对象、其他标准库类型和指针。这种适应性使得`std::vector`适用于广泛的用途和场景。

此外，向量提供了一种将一个向量的内容复制到另一个向量的简单机制。这被称为 **复制初始化**。以下代码演示了这一点：

```cpp
std::vector<int> vec1 = {1, 2, 3, 4, 5};
std::vector<int> vec2(vec1);
```

在这个例子中，`vec2` 被初始化为 `vec1` 的精确副本，这意味着 `vec2` 将包含与 `vec1` 相同的元素。这种复制初始化确保原始向量（`vec1`）保持不变，并且新向量（`vec2`）提供了数据的单独副本。

STL 容器的真正优势之一是它们能够无缝地处理用户定义的类型，而不仅仅是像 `int` 或 `double` 这样的原始数据类型。这种灵活性是对其模板设计的证明，它允许它适应各种数据类型同时保持类型安全。在接下来的示例中，我们通过使用自定义类来展示这种多功能性：

```cpp
#include <iostream>
#include <string>
#include <vector>
class Person {
public:
  Person() = default;
  Person(std::string_view n, int a) : name(n), age(a) {}
  void display() const {
    std::cout << "Name: " << name << ", Age: " << age
              << "\n";
  }
private:
  std::string name;
  int age{0};
};
int main() {
  std::vector<Person> people;
  people.push_back(Person("Lisa", 30));
  people.push_back(Person("Corbin", 25));
  people.resize(3);
  people[2] = Person("Aaron", 28);
  for (const auto &person : people) { person.display(); }
  return 0;
}
```

这里是示例输出：

```cpp
Name: Lisa, Age: 30
Name: Corbin, Age: 25
Name: Aaron, Age: 28
```

在这个例子中，首先使用 `std::vector` 来管理自定义 `Person` 类的对象。它展示了 `std::vector` 如何轻松地容纳和管理内置类型和用户定义类型的内存。

在 C++ 中，尽管静态数组有其用途，但它们具有固定的大小，有时可能受到限制。另一方面，`std::vector` 提供了一种动态和灵活的替代方案。

理解向量的声明和初始化对于有效的 C++ 编程至关重要。`std::vector` 是一种多用途的工具，适用于从实现复杂算法到开发大型应用程序的各种任务。将 `std::vector` 纳入编程实践可以提高代码的效率和可维护性。

在本节中，我们涵盖了与 `std::vector` 一起工作的语法方面。具体来说，我们深入探讨了声明不同类型 `std::vector` 的正确技术以及初始化这些向量以适应不同编程场景的多种策略。

我们了解到声明 `std::vector` 需要指定它将包含的元素类型，以及可选的初始大小和元素的默认值。我们发现了多种初始化方法，包括直接列表初始化和使用特定值范围的初始化。本节强调了 `std::vector` 的灵活性，展示了它如何从预定义的元素集开始或从现有集合中构建。

这些信息对于实际的 C++ 开发至关重要，因为它为有效地使用 `std::vector` 提供了基础。适当的初始化可以导致性能优化并确保向量处于适合其预期用途的有效状态。能够简洁且正确地声明和初始化向量是利用 STL 在实际 C++ 应用程序中发挥其强大功能的基础技能。

在下一节“访问元素”中，我们将关注允许我们检索和修改`std::vector`内容的操作。我们将学习随机访问，它允许高效地检索和修改向量中任何位置的元素。此外，我们还将探讨如何访问第一个和最后一个元素，以及理解和管理向量大小的重要性，以确保代码健壮且无错误。

# 访问元素

在讨论了`std::vector`的声明和初始化之后，我们的重点现在转向访问和操作包含的数据。C++中的多个方法允许您以速度和安全的方式访问向量元素。

## 随机访问

下标`[]`操作符允许通过索引直接访问元素，类似于数组。在以下示例中，给定一个向量，表达式`numbers[1]`返回值`20`。然而，使用此操作符不涉及边界检查。超出范围的索引，如`numbers[10]`，会导致未定义的行为，从而导致不可预测的结果。

这在以下示例中显示：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {10, 20, 30, 40, 50};
  const auto secondElement = numbers[1];
  std::cout << "The second element is: " << secondElement
            << "\n";
  // Beware: The following line can cause undefined
  // behavior!
  const auto outOfBoundsElement = numbers[10];
  std::cout << "Accessing an out-of-bounds index: "
            << outOfBoundsElement << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
The second element is: 20
Accessing an out-of-bounds index: 0
```

为了更安全地基于索引访问，`std::vector`提供了`at()`成员函数。它执行索引边界检查，并在无效索引上抛出`out_of_range`异常。

这里是此示例的一个例子：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {10, 20, 30, 40, 50};
  try {
    const auto secondElement = numbers.at(1);
    std::cout << "The second element is: " << secondElement
              << "\n";
  } catch (const std::out_of_range &e) {
    std::cerr << "Error: " << e.what() << "\n";
  }
  try {
    const auto outOfBoundsElement = numbers.at(10);
    std::cout << "Accessing an out-of-bounds index: "
              << outOfBoundsElement << "\n";
  } catch (const std::out_of_range &e) {
    std::cerr << "Error: " << e.what() << "\n";
  }
  return 0;
}
```

访问向量元素时，谨慎至关重要。虽然 C++优先考虑性能，但它经常绕过安全检查，例如下标操作符所示。因此，开发者必须通过仔细的索引管理或采用更安全的方法（如`at()`）来确保有效的访问。

## 访问第一个和最后一个元素

可以使用`front()`和`back()`分别访问第一个和最后一个元素。

这在以下示例中显示：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {10, 20, 30, 40, 50};
  const auto firstElement = numbers.front();
  std::cout << "The first element is: " << firstElement
            << "\n";
  const auto lastElement = numbers.back();
  std::cout << "The last element is: " << lastElement
            << "\n";
  return 0;
}
```

示例输出如下：

```cpp
The first element is: 10
The last element is: 50
```

## 向量大小

使用`std::vector`时，理解其结构和包含的数据量是至关重要的。`size()`成员函数提供了向量中当前存储的元素数量。在`std::vector`实例上调用此函数将返回它持有的元素数量。这个计数代表活动元素，可以用来确定有效索引的范围。返回值是`size_t`类型，这是一种无符号整数类型，适合表示大小和计数。当遍历向量、执行大小比较或根据向量元素数量分配空间时，这很有用。

让我们看看以下代码：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> data = {1, 2, 3, 4, 5};
  const auto elementCount = data.size();
  std::cout << "Vector contains " << elementCount
            << " elements.\n";
  return 0;
}
```

在前面的代码中，对数据向量调用`size()`函数以检索和显示它包含的元素数量。结果正如预期的那样，表明向量中有五个元素。

总结来说，`std::vector`提供了一套工具，从高效的索引操作符到更安全的`at()`方法，再到方便的`front()`和`back()`方法。理解这些工具对于有效地和安全地访问和操作向量中的数据至关重要。

在本节中，我们专注于检索和检查`std::vector`内容的方法。我们了解了`std::vector`提供随机访问其元素的能力，这使得我们可以使用其索引以常数时间复杂度直接访问任何元素。本节还详细介绍了通过`front()`和`back()`成员函数分别访问向量第一个和最后一个元素的方法。

此外，我们讨论了理解并利用`size()`成员函数的重要性，以确定当前存储在`std::vector`中的元素数量。这种理解对于确保我们的访问模式保持在向量的界限内至关重要，从而防止越界错误和未定义的行为。

从本节中获得的能力是至关重要的，因为它们是交互`std::vector`内容的基础。这些访问模式是使用向量在 C++应用程序中有效使用的关键，无论是读取还是修改元素。直接访问向量中的元素可以导致高效的算法并支持广泛的日常编程任务。

以下部分将通过解决如何修改`std::vector`的大小和内容来进一步扩展我们的知识。我们将探讨如何向向量中添加元素以及删除它们的各种方法。这包括理解向量如何管理其容量及其对性能的影响。我们将学习为什么以及如何使用`.empty()`作为检查大小是否为`0`的更高效替代方案，并且我们将深入了解如何从向量中清除所有元素。

# 添加和删除元素

与传统的数组相比，`std::vector`的一个优点是它能够动态地调整大小。随着应用程序的发展，数据需求也在变化；静态数据结构不再适用。在本节中，我们将探索使用`std::vector`进行动态数据管理，学习无缝地向向量中添加和删除元素，同时确保我们的操作是安全的。

## 添加元素

让我们从添加元素开始。`push_back()`成员函数可能是向向量末尾添加元素最直接的方法。假设你有`std::vector<int> scores;`并希望添加一个新的分数，比如`95`。你只需调用`scores.push_back(95);`，哇，你的分数就添加成功了。

下面是一个简单的示例代码：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> scores;
  std::cout << "Initial size of scores: " << scores.size()
            << "\n";
  scores.push_back(95);
  std::cout << "Size after adding one score:"
            << scores.size() << "\n";
  std::cout << "Recently added score: " << scores[0]
            << "\n";
  return 0;
}
```

当运行此程序时，它将显示在添加分数之前和之后的向量大小以及分数本身，从而演示了`push_back()`函数的实际应用。

如果您需要在特定位置插入一个分数，而不仅仅是末尾，怎么办？`insert()`函数将成为您的最佳助手。如果您想在第三位置插入分数`85`，您将使用迭代器来指定位置：

`scores.insert(scores.begin() + 2, 85);`

记住，向量索引从`0`开始；`+ 2`是为了第三位。

让我们通过在以下代码中结合使用`insert()`函数来扩展前面的例子：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> scores = {90, 92, 97};
  std::cout << "Initial scores: ";
  for (int score : scores) { std::cout << " " << score; }
  std::cout << "\n";
  scores.push_back(95);
  std::cout << "Scores after adding 95 to the end: ";
  for (int score : scores) { std::cout << " " << score; }
  std::cout << "\n";
  scores.insert(scores.begin() + 2, 85);
  std::cout << "Scores after inserting 85 at the third "
               "position:";
  for (int score : scores) { std::cout << " " << score; }
  std::cout << "\n";
  return 0;
}
```

这个程序将展示原始分数，显示在末尾追加一个分数后的分数，以及在第三位置插入一个分数。它展示了`push_back()`和`insert()`函数的作用。

向量并没有停止在这里。`emplace_back()`和`emplace()`函数允许在向量内部直接构造元素。这意味着更少的临时对象，并且可能提高性能，尤其是在复杂的数据类型中。

让我们考虑一个具有几个数据成员的`Person`类。要创建一个新的`Person`对象，会执行字符串连接操作。使用`emplace_back()`和`emplace()`将避免`push_back()`可能引起的额外临时对象和复制/移动操作，从而提高性能。以下代码演示了这一点：

```cpp
#include <iostream>
#include <string>
#include <vector>
class Person {
public:
  Person(const std::string &firstName,
         const std::string &lastName)
      : fullName(firstName + " " + lastName) {}
  const std::string &getName() const { return fullName; }
private:
  std::string fullName;
};
int main() {
  std::vector<Person> people;
  people.emplace_back("John", "Doe");
  people.emplace(people.begin(), "Jane", "Doe");
  for (const auto &person : people) {
    std::cout << person.getName() << "\n";
  }
  return 0;
}
```

这个例子说明了`emplace_back()`和`emplace()`如何允许在向量内部直接构造对象。使用`push_back()`可能会创建临时的`Person`对象。使用`emplace_back()`直接在原地构造对象，可能避免临时对象的创建。使用`insert()`可能会创建临时的`Person`对象。使用`emplace()`直接在指定位置构造对象。这对于像`Person`这样的类型尤其有益，其构造函数可能涉及资源密集型操作（如字符串连接）。在这种情况下，`emplace`方法相对于它们的`push`对应方法的性能优势变得明显。

## 删除元素

但生活不仅仅是加法。有时，我们需要删除数据。`pop_back()`函数从向量中删除最后一个元素，将其大小减少一个。然而，如果您想从特定位置或一系列位置删除，`erase()`函数将是您的首选。

### erase-remove 惯用法

在 C++及其 STL 中，有经验开发者经常使用的编码模式。一个值得注意的模式是**erase-remove**惯用法，它根据定义的准则从容器中删除特定元素。本节将详细说明这个惯用法的功能，并讨论 C++20 中引入的新替代方案。

STL 容器，特别是`std::vector`，没有提供一种直接根据谓词删除元素的方法。相反，它们提供了单独的方法：一个用于重新排列元素（使用`std::remove`和`std::remove_if`），另一个用于删除它们。

下面是如何实现 erase-remove 惯用法的：

1.  使用`std::remove`或`std::remove_if`对容器的元素进行重新排序。需要移除的元素被移动到末尾。

1.  这些算法返回一个指向被移除元素起始位置的迭代器。

1.  然后使用容器的`erase`方法从容器中物理移除元素。

一个经典的例子是从`std::vector<int>`中移除所有`0`的实例：

```cpp
std::vector<int> numbers = {1, 0, 3, 0, 5};
auto end = std::remove(numbers.begin(), numbers.end(), 0);
numbers.erase(end, numbers.end());
```

### 使用 std::erase 和 std::erase_if 进行现代化

认识到 erase-remove 惯用语的普遍性和某种程度上反直觉的特性，C++20 引入了直接实用函数来简化此操作：`std::erase`和`std::erase_if`。这些函数将两步过程合并为一步，提供了一种更直观且更不易出错的解决方案。

使用之前的例子，在 C++20 中移除所有`0`的实例变得如下：

```cpp
std::vector<int> numbers = {1, 0, 3, 0, 5};
std::erase(numbers, 0);
```

现在不再需要调用单独的算法并记住处理过程的两个阶段。同样，为了根据谓词移除元素，您将执行以下操作：

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};
std::erase_if(numbers, [](int x){ return x % 2 == 0; });
```

虽然 erase-remove 惯用语多年来一直是基于 STL 的 C++编程的基石，但现代 C++仍在不断发展和简化常见模式。有了`std::erase`和`std::erase_if`，开发者现在有更直接的工具来移除容器元素，从而产生更干净、更易读的代码。这是 C++社区持续致力于增强语言的用户友好性、同时保留其强大和表达力的承诺的证明。

注意，`std::vector`已被巧妙地设计以优化内存操作。虽然人们可能会直观地期望在添加或移除元素时，底层数组会进行大小调整，但这并不是事实。相反，当向量增长时，它通常会分配比立即需要的更多内存，以预测未来的添加。这种策略最小化了频繁的内存重新分配，这可能非常昂贵。相反，当移除元素时，向量并不总是立即缩小其分配的内存。这种行为在内存使用和性能之间提供了平衡。然而，值得注意的是，这些内存管理决策的具体细节可能因 C++库实现而异。因此，虽然接口方面的行为在实现之间是一致的，但内部内存管理的细微差别可能不同。

## 容量

您可以使用`capacity()`成员函数来了解已分配了多少内存。`std::vector::capacity()`成员函数返回为向量分配的内存量，这可能会大于其实际大小。此值表示向量在重新分配内存之前可以容纳的最大元素数，确保在无需频繁内存操作的情况下高效增长。

这可以在以下内容中看到：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers;
  std::cout << "Initial size: " << numbers.size() << "\n";
  std::cout << "Initial capacity: " << numbers.capacity()
            << "\n";
  for (auto i = 1; i <= 10; ++i) { numbers.push_back(i); }
  std::cout << "Size after adding 10 elements: "
            << numbers.size() << "\n";
  std::cout << "Capacity after adding 10 elements: "
            << numbers.capacity() << "\n";
  for (auto i = 11; i <= 20; ++i) { numbers.push_back(i); }
  std::cout << "Size after adding 20 elements: "
            << numbers.size() << "\n";
  std::cout << "Capacity after adding 20 elements: "
            << numbers.capacity() << "\n";
  for (auto i = 0; i < 5; ++i) { numbers.pop_back(); }
  std::cout << "Size after removing 5 elements: "
            << numbers.size() << "\n";
  std::cout << "Capacity after removing 5 elements: "
            << numbers.capacity() << "\n";
  return 0;
}
```

具体的输出可能因编译器而异，但以下是一个示例输出：

```cpp
Initial size: 0
Initial capacity: 0
Size after adding 10 elements: 10
Capacity after adding 10 elements: 16
Size after adding 20 elements: 20
Capacity after adding 20 elements: 32
Size after removing 5 elements: 15
Capacity after removing 5 elements: 32
```

这个例子说明了随着元素的添加和删除，`std::vector` 实例的大小和容量如何变化。检查输出显示，容量通常并不直接与大小相对应，突出了内存优化技术。

## 当可能时，优先使用 empty()

在 C++中，当主要目的是检查容器是否为空时，建议使用 `.empty()` 成员函数而不是将 `.size()` 或 `.capacity()` 与 `0` 进行比较。`.empty()` 函数提供了一种直接确定容器是否有元素的方法，并且在许多实现中，它可以提供性能优势。具体来说，`.empty()` 通常具有常数时间复杂度 `O(1)`，而 `.size()` 对于某些容器类型可能具有线性时间复杂度 `O(n)`，这使得 `.empty()` 对于简单的空检查来说是一个更有效的选择。使用 `.empty()` 可以使代码更加简洁，并可能更快，尤其是在性能关键部分。

## 清除所有元素

`std::vector` 的 `clear()` 函数是一个强大的实用工具，可以迅速删除容器内的所有元素。调用此函数后，向量的 `size()` 将返回 `0`，表示其现在为空状态。然而，需要注意的一个关键方面是，任何之前指向向量内元素的引用、指针或迭代器都将因这次操作而失效。这也适用于任何超出范围的迭代器。有趣的是，尽管 `clear()` 清除了所有元素，但它并不改变向量的容量。这意味着为向量分配的内存保持不变，允许在无需立即重新分配的情况下进行高效的后续插入。

`std::vector` 的 `clear()` 成员函数从向量中删除所有元素，有效地将其大小减少到 `0`。以下是一个简单的示例，演示其用法：

```cpp
#include <iostream>
#include <vector>
int main() {
  std::vector<int> numbers = {1, 2, 3, 4, 5};
  std::cout << "Original numbers: ";
  for (const auto num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  numbers.clear();
  std::cout << "After using clear(): ";
  // This loop will produce no output.
  for (const auto num : numbers) {
    std::cout << num << " ";
  }
  std::cout << "\n";
  std::cout << "Size of vector after clear(): "
            << numbers.size() << "\n";
  return 0;
}
```

这里是示例输出：

```cpp
Original numbers: 1 2 3 4 5
After using clear():
Size of vector after clear(): 0
```

这个例子强调了 `std::vector` 在使用单个函数调用处理大量删除时的效率，使数据管理更加直接。

动态调整大小是 `std::vector` 的一个显著特性，但它需要谨慎管理以保持效率。当向量的内容超过其容量时，需要进行内存重新分配，这涉及到分配一个新的内存块、复制现有元素和释放旧内存。这个过程可能会引入性能开销，尤其是在向量通过小量重复增长时。如果你可以预测最大大小，可以使用 `reserve()` 函数预先分配内存以减轻这种低效。

例如，调用 `scores.reserve(100);` 为 100 个元素分配内存，减少到该限制之前的频繁重新分配需求。

`std::vector`提供了一套针对动态数据管理的综合函数。它使得快速添加元素、在中间插入元素或从各种位置删除元素变得容易。结合其高效的内存管理，`std::vector`作为一个灵活且注重性能的容器脱颖而出。随着你对 C++的深入了解，`std::vector`的实用性将越来越明显，因为它有效地解决了广泛的编程场景。

在本节中，我们探讨了`std::vector`的动态特性，这使得我们可以修改其内容和大小。我们学习了如何使用`push_back`、`emplace_back`等方法向向量中添加元素，以及如何使用迭代器在特定位置插入元素。我们还考察了删除元素的过程，无论是特定位置的单一元素、一系列元素，还是按值删除元素。

我们讨论了容量概念，即向量中元素预分配的空间量，以及它与大小（即向量中当前实际元素的数量）的区别。理解这一区别对于编写内存和性能高效的程序至关重要。

我们还强调了使用`empty()`作为检查向量是否包含任何元素的推荐方法。我们讨论了`empty()`相对于检查`size()`是否返回`0`的优势，特别是在清晰度和潜在的性能优势方面。

此外，我们还介绍了`clear()`函数的重要性，该函数从向量中删除所有元素，有效地将其大小重置为`0`，而无需改变其容量。

本节的信息非常实用，因为它使我们能够积极且高效地管理`std::vector`的内容。了解添加和删除元素对于实现需要动态数据操作的算法至关重要，这在软件开发中是一个常见的场景。

# 摘要

在本章中，你学习了 C++ STL 中`std::vector`的基础知识。本章首先解释了`std::vector`的重要性，强调了它相对于 C 风格数组的优势，尤其是在内存管理和易用性方面。本章详细比较了 C 风格数组和`std::vector`，展示了`std::vector`如何促进动态大小调整和更安全的内存操作。

接下来，你被引导了解了声明和初始化向量的过程。你学习了如何声明`std::vector`以及使用不同方法初始化这些实例。然后，章节探讨了访问`std::vector`中元素的各种方法，从随机访问到访问第一个和最后一个元素，并强调了理解向量大小的重要性。

此外，本章深入探讨了添加和删除元素的内情。本节阐明了修改向量内容时的最佳实践，包括何时使用 `empty()` 而不是检查大小为 `0`，以及理解向量容量的重要性。

本章所提供的信息极为宝贵，因为它构建了在多种编程场景中有效利用 `std::vector`（以及许多其他 STL 数据类型）所需的基础知识。掌握 `std::vector` 允许编写更高效、更易于维护的代码，使 C++ 开发者能够充分利用 STL 在动态数组操作方面的全部潜力。

在下一章中，你将通过学习迭代器来提高你对 `std::vector` 的理解，迭代器是导航 STL 容器中元素的关键。
