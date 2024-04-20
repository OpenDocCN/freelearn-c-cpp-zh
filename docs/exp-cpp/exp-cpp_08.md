# 第六章：深入 STL 中的数据结构和算法

掌握数据结构对程序员至关重要。大多数情况下，数据存储方式定义了应用程序的整体效率。例如，考虑一个电子邮件客户端。您可以设计一个显示最新 10 封电子邮件的电子邮件客户端，并且它可能具有最佳的用户界面；在几乎任何设备上都可以顺畅地显示最近的 10 封电子邮件。您的电子邮件应用程序的用户在使用您的应用程序两年后可能会收到数十万封电子邮件。当用户需要搜索电子邮件时，您的数据结构知识将发挥重要作用。您存储数十万封电子邮件的方式以及您用于排序和搜索它们的方法（算法）将是您的程序与其他所有程序的区别所在。

程序员在项目中努力寻找每日问题的最佳解决方案。使用经过验证的数据结构和算法可以极大地改善程序员的工作。一个好程序最重要的特性之一是速度，通过设计新的算法或使用现有算法来获得速度。

最后，C++20 引入了用于定义**元类型**的**概念**，即描述其他类型的类型。语言的这一强大特性使数据架构完整。

C++的**标准模板库**（**STL**）涵盖了大量的数据结构和算法。我们将探索使用 STL 容器来高效组织数据的方法。然后我们将深入研究 STL 提供的算法实现。理解并使用 STL 容器中的概念至关重要，因为 C++20 通过引入迭代器概念来大幅改进迭代器。

本章将涵盖以下主题：

+   数据结构

+   STL 容器

+   概念和迭代器

+   掌握算法

+   探索树和图

# 技术要求

本章中使用带有选项`-std=c++2a`的 g++编译器来编译示例。您可以在本书的 GitHub 存储库中找到本章中使用的源文件[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# 数据结构

作为程序员，您可能熟悉使用数组来存储和排序数据集。程序员在项目中除了数组之外还会大量使用其他数据结构。了解并应用适当的数据结构可能在程序性能中发挥重要作用。要选择正确的数据结构，您需要更好地了解它们。一个明显的问题可能会出现，即我们是否需要研究数据结构的动物园——向量、链表、哈希表、图、树等等。为了回答这个问题，让我们假设一个想要更好的数据结构的必要性自然而然地显现出来的想象场景。

在介绍内容中，我们提到了设计一个电子邮件客户端。让我们对其设计和实现过程中的基本任务有一个一般的了解。

电子邮件客户端是一个列出来自各个发件人的电子邮件的应用程序。我们可以将其安装在台式电脑或智能手机上，或者使用浏览器版本。电子邮件客户端应用程序的主要任务包括发送和接收电子邮件。现在假设我们正在设计一个足够简单的电子邮件客户端。就像在编程书籍中经常发生的那样，假设我们使用了一些封装了发送和接收电子邮件工作的库。我们更愿意集中精力设计专门用于存储和检索电子邮件的机制。电子邮件客户端用户应该能够查看**收件箱**部分中的电子邮件列表。我们还应该考虑用户可能想要对电子邮件执行的操作。他们可以逐个删除电子邮件，也可以一次删除多封。他们可以选择任意选定的电子邮件并回复给发件人或将电子邮件转发给其他人。

我们在第十章中讨论了软件设计过程和最佳实践，*设计真实世界应用程序*。现在，让我们草拟一个描述电子邮件对象的简单结构，如下所示：

```cpp
struct Email
{
  std::string subject;
  std::string body;
  std::string from;
  std::chrono::time_point datetime;
};
```

我们应该关心的第一件事是将电子邮件集合存储在一个易于访问的结构中。数组听起来可能不错。假设我们将所有收到的电子邮件存储在一个数组中，如下面的代码块所示：

```cpp
// let's suppose a million emails is the max for anyone
const int MAX_EMAILS = 1'000'000; 
Email inbox[MAX_EMAILS];
```

我们可以以任何形式存储 10 封电子邮件-这不会影响应用程序的性能。然而，显而易见的是，随着时间的推移，电子邮件的数量将增加。对于每封新收到的电子邮件，我们将`Email`对象与相应的字段推送到`inbox`数组中。数组的最后一个元素表示最近收到的电子邮件。因此，要显示最近的十封电子邮件列表，我们需要读取并返回数组的最后十个元素。

当我们尝试操作存储在`inbox`数组中的成千上万封电子邮件时，问题就出现了。如果我们想在所有电子邮件中搜索单词`friend`，我们必须扫描数组中的所有电子邮件，并将包含单词`friend`的电子邮件收集到一个单独的数组中。看看下面的伪代码：

```cpp
std::vector<Email> search(const std::string& word) {
  std::vector<Email> search_results;  
  for (all-million-emails) {
    if (inbox[i].subject.contains(word)) {
      search_results.push_back(inbox[i]);
    }
  }
  return search_results;
}
```

使用数组存储所有数据对于小集合来说已经足够了。在处理更大的数据集的真实世界应用程序中，情况会发生巨大变化。使用特定的数据结构的目的是使应用程序运行更加流畅。前面的例子展示了一个简单的问题：在电子邮件列表中搜索匹配特定值。在一封电子邮件中找到该值需要合理的时间。

如果我们假设电子邮件的主题字段可能包含多达十个单词，那么在电子邮件主题中搜索特定单词需要将该单词与主题中的所有单词进行比较。在*最坏的情况*下，没有匹配。我们强调最坏的情况，因为只有在查找需要检查主题中的每个单词时才会出现这种情况。对成千上万甚至数十万封电子邮件做同样的操作将使用户等待时间过长。

选择适合特定问题的数据结构对于应用程序的效率至关重要。例如，假设我们使用哈希表将单词映射到电子邮件对象。每个单词将被映射到包含该单词的电子邮件对象列表。这种方法将提高搜索操作的效率，如下图所示：

![](img/f7318cbc-fefa-41f1-a377-9bf8ebd60b26.png)

`search()`函数将返回哈希表键引用的列表：

```cpp
std::vector<Email> search(const std::string& word) {
  return table[word];
}
```

这种方法只需要处理每封接收到的电子邮件，将其拆分为单词并更新哈希表。

为了简单起见，我们使用`Email`对象作为值而不是引用。请注意，最好将指针存储在向量中指向`Email`。

现在让我们来看看不同的数据结构及其应用。

# 顺序数据结构

开发人员最常用的数据结构之一是动态增长的一维数组，通常称为向量。STL 提供了一个同名的容器：`std::vector`。向量背后的关键思想是它包含相同类型的项目按顺序放置在内存中。例如，由 4 字节整数组成的向量将具有以下内存布局。向量的索引位于以下图表的右侧：

![](img/8f1961c0-dcd6-481f-8ae8-3ba43902ba49.png)

向量的物理结构允许实时访问其任何元素。

我们应该根据容器的操作来区分它们，以便在特定问题中正确应用它们。为此，我们通常定义容器中的操作与容器中元素数量的运行时间复杂度的关系。例如，向量的元素访问被定义为常数时间操作，这意味着获取向量项需要相同数量的指令，无论向量长度如何。

访问向量的第一个元素和访问向量的第 100 个元素需要相同的工作量，因此我们称之为常数时间操作，也称为***O(1)操作***。

虽然向量中的元素访问速度很快，但添加新元素有些棘手。每当我们在向量的末尾插入新项时，我们还应该考虑向量的容量。当没有为向量分配更多空间时，它应该动态增长。看一下下面的`Vector`类及其`push_back()`函数：

```cpp
template <typename T>
class Vector
{
public:
  Vector() : buffer_{nullptr}, capacity_{2}, size_{0}
  {
    buffer_ = new T[capacity_]; // initializing an empty array
  }
  ~Vector() { delete [] buffer_; }
  // code omitted for brevity

public:
  void push_back(const T& item)
 {
 if (size_ == capacity_) {
 // resize
 }
 buffer_[size_++] = item;
 }
  // code omitted for brevity
};
```

在深入实现`push_back()`函数之前，让我们看一下下面的图表：

![](img/11cb3eec-b2a8-4166-8fdf-a58cf516bf90.png)

我们应该分配一个全新的数组，将旧数组的所有元素复制到新数组中，然后将新插入的元素添加到新数组末尾的下一个空闲槽中。这在下面的代码片段中显示：

```cpp
template <typename T>
class Vector
{
public:
  // code omitted for brevity
  void push_back(const T& item)
  {
    if (size_ == capacity_) {
 capacity_ *= 2; // increase the capacity of the vector twice
 T* temp_buffer = new T[capacity_];
      // copy elements of the old into the new
 for (int ix = 0; ix < size_; ++ix) {
 temp_buffer[ix] = buffer_[ix];
 }
 delete [] buffer_; // free the old array
 buffer_ = temp_buffer; // point the buffer_ to the new array
 }
    buffer_[size_++] = item;
  }
  // code omitted for brevity
};
```

调整因子可以选择不同 - 我们将其设置为`2`，这样每当向量满时，向量的大小就会增长两倍。因此，我们可以坚持认为，大多数情况下，在向量的末尾插入新项需要常数时间。它只是在空闲槽中添加项目并增加其`private size_`变量。不时地，添加新元素将需要分配一个新的、更大的向量，并将旧的向量复制到新的向量中。对于这样的情况，该操作被称为**摊销**常数时间完成。

当我们在向量的前面添加元素时，情况就不一样了。问题在于，所有其他元素都应该向右移动一个位置，以便为新元素腾出一个位置，如下图所示：

![](img/0f4021af-1ec3-4d9d-85ca-891a7e16e42a.png)

这是我们在`Vector`类中如何实现它的方式：

```cpp
// code omitted for brevity
void push_front(const T& item)
{
  if (size_ == capacity_) {
    // resizing code omitted for brevity
  }
  // shifting all the elements to the right
 for (int ix = size_ - 1; ix > 0; --ix) {
 buffer_[ix] = buffer[ix - 1];
 }
  // adding item at the front buffer_[0] = item;
  size_++;
}
```

在需要仅在容器的前面插入新元素的情况下，选择向量并不是一个好的选择。这是其他容器应该被考虑的例子之一。

# 基于节点的数据结构

基于节点的数据结构不占用连续的内存块。基于节点的数据结构为其元素分配节点，没有任何顺序 - 它们可能随机分布在内存中。我们将每个项目表示为链接到其他节点的节点。

最流行和最基础的基于节点的数据结构是链表。下图显示了双向链表的可视结构：

![](img/de263cb6-41ed-4f47-a59e-1a9e01261f64.png)

链表与向量非常不同。它的一些操作速度更快，尽管它缺乏向量的紧凑性。

为了简洁起见，让我们在列表的前面实现元素插入。我们将每个节点都保留为一个结构：

```cpp
template <typename T>
struct node 
{
  node(const T& it) : item{it}, next{nullptr}, prev{nullptr} {}
  T item;
  node<T>* next;
  node<T>* prev;
};
```

注意`next`成员 - 它指向相同的结构，这样可以允许节点链接在一起，如前面的插图所示。

要实现一个链表，我们只需要保留指向其第一个节点的指针，通常称为链表的头。在列表的前面插入元素很简单：

```cpp
template <typename T>
class LinkedList 
{
  // code omitted for brevity
public:
  void push_front(const T& item) 
 {
 node<T>* new_node = new node<T>{item};
 if (head_ != nullptr) {
 new_node->next = head_->next;
 if (head_->next != nullptr) {
 head_->next->prev = new_node;
 }
 }
 new_node->next = head_;
 head_ = new_node;
 }
private:
  node<T>* head_; 
};
```

在向列表中插入元素时，我们应该考虑三种情况：

+   如前所述，在列表前面插入元素需要以下步骤：

![](img/06be3736-adbe-4388-9396-677b0a094a7f.png)

+   在列表末尾插入元素如下图所示：

![](img/6f88bf92-0a38-448d-a32c-8a92883f53ab.png)

+   最后，在列表中间插入元素的操作如下所示：

![](img/75876dd0-13a8-4b23-a1be-68ac50c50dd0.png)

在前面的图中，向向量插入元素显然与向列表插入元素不同。您将如何在向量和列表之间进行选择？您应该专注于操作及其速度。例如，从向量中读取任何元素都需要恒定的时间。我们可以在向量中存储一百万封电子邮件，并在不需要任何额外工作的情况下检索位置为 834,000 的电子邮件。对于链表，操作是线性的。因此，如果您需要存储的数据集大部分是读取而不是写入，那么显然使用向量是一个合理的选择。

在列表中的任何位置插入元素都是一个常量时间的操作，而向量会努力在随机位置插入元素。因此，当您需要一个可以频繁添加/删除数据的对象集合时，更好的选择将是链表。

我们还应该考虑缓存内存。向量具有良好的数据局部性。读取向量的第一个元素涉及将前*N*个元素复制到缓存中。进一步读取向量元素将更快。我们不能说链表也是如此。要找出原因，让我们继续比较向量和链表的内存布局。

# 内存中的容器

正如您从前几章已经知道的那样，对象占用内存空间在进程提供的内存段之一上。大多数情况下，我们对堆栈或堆内存感兴趣。自动对象占用堆栈上的空间。以下两个声明都驻留在堆栈上：

```cpp
struct Email 
{
  // code omitted for brevity
};

int main() {
  Email obj;
  Email* ptr;
}
```

尽管`ptr`表示指向`Email`对象的指针，但它占用堆栈上的空间。它可以指向在堆上分配的内存位置，但指针本身（存储内存位置地址的变量）驻留在堆栈上。在继续使用向量和列表之前，这一点是至关重要的。

正如我们在本章前面看到的，实现向量涉及封装指向表示指定类型的元素数组的内部缓冲区的指针。当我们声明一个`Vector`对象时，它需要足够的堆栈内存来存储其成员数据。`Vector`类有以下三个成员：

```cpp
template <typename T>
class Vector
{
public:
  // code omitted for brevity

private:
  int capacity_;
  int size_;
  T* buffer_;
};
```

假设整数占用 4 个字节，指针占用 8 个字节，那么以下`Vector`对象声明将至少占用 16 个字节的堆栈内存：

```cpp
int main()
{
  Vector<int> v;
}
```

这是我们对前面代码的内存布局的想象：

![](img/5b07753c-2089-4701-a865-3e98d597197f.png)

插入元素后，堆栈上的向量大小将保持不变。堆出现了。`buffer_`数组指向使用`new[]`运算符分配的内存位置。例如，看看以下代码：

```cpp
// we continue the code from previous listing
v.push_back(17);
v.push_back(21);
v.push_back(74);
```

我们推送到向量的每个新元素都将占用堆上的空间，如下图所示：

![](img/ffb6f27e-00ca-4b30-86b7-4cdfd6c1530e.png)

每个新插入的元素都驻留在`buffer_`数组的最后一个元素之后。这就是为什么我们可以说向量是一个友好的缓存容器。

声明链表对象也会为其数据成员占用堆栈上的内存空间。如果我们讨论的是仅存储`head_`指针的简单实现，那么以下链表对象声明将至少占用 8 个字节的内存（仅用于`head_`指针）：

```cpp
int main()
{
  LinkedList<int> list;
}
```

以下插图描述了前面代码的内存布局：

![](img/62daaf10-d88c-4439-a8e8-4bb85feb15e4.png)

插入新元素会在堆上创建一个`node`类型的对象。看看以下行：

```cpp
list.push_back(19);
```

在插入新元素后，内存插图将如下所示改变：

![](img/62c34472-4720-43f8-80c1-71ac1b5ab204.png)

要注意的是，节点及其所有数据成员都驻留在堆上。该项存储我们插入的值。当我们插入另一个元素时，将再次创建一个新节点。这次，第一个节点的下一个指针将指向新插入的元素。而新插入的节点的 prev 指针将指向列表的前一个节点。下图描述了在插入第二个元素后链表的内存布局：

![](img/080ab163-ffd0-4b7b-8ff8-2ba3e9dfed60.png)

当我们在向列表中插入元素之间在堆上分配一些随机对象时，会发生有趣的事情。例如，以下代码将一个节点插入列表，然后为一个整数（与列表无关）分配空间。最后，再次向列表中插入一个元素：

```cpp
int main()
{
  LinkedList<int> list;
  list.push_back(19);
  int* random = new int(129);
  list.push_back(22);
}
```

这个中间的随机对象声明破坏了列表元素的顺序，如下图所示：

![](img/bff12ecb-958e-4b6b-95b1-d731f5a627a6.png)

前面的图表提示我们，列表不是一个友好的缓存容器，因为它的结构和其元素的分配。

注意通过将每个新节点合并到代码中所创建的内存开销。我们为一个元素额外支付 16 个字节（考虑到指针占用 8 个字节的内存）。因此，列表在最佳内存使用方面输给了向量。

我们可以尝试通过在列表中引入预分配的缓冲区来解决这个问题。然后每个新节点的创建将通过**placement new**操作符进行。然而，更明智的选择是选择更适合感兴趣问题的数据结构。

在实际应用程序开发中，程序员很少实现自己的向量或链表。他们通常使用经过测试和稳定的库版本。C++为向量和链表提供了标准容器。此外，它为单链表和双链表提供了两个单独的容器。

# STL 容器

STL 是一个强大的算法和容器集合。虽然理解和实现数据结构是程序员的一项重要技能，但你不必每次在项目中需要时都要实现它们。库提供者负责为我们实现稳定和经过测试的数据结构和算法。通过理解数据结构和算法的内部细节，我们在解决问题时能够更好地选择 STL 容器和算法。

先前讨论的向量和链表在 STL 中分别实现为`std::vector<T>`和`std::list<T>`，其中`T`是集合中每个元素的类型。除了类型，容器还以分配器作为第二个默认`template`参数。例如，`std::vector`声明如下：

```cpp
template <typename T, typename Allocator = std::allocator<T> >
class vector;
```

在上一章中介绍过，分配器处理容器元素的高效分配/释放。`std::allocator` 是 STL 中所有标准容器的默认分配器。一个更复杂的分配器，根据内存资源的不同而表现不同，是`std::pmr::polymorphic_allocator`。STL 提供了`std::pmr::vector`作为使用多态分配器的别名模板，定义如下：

```cpp
namespace pmr {
  template <typename T>
  using vector = std::vector<T, std::pmr::polymorphic_allocator<T>>;
}
```

现在让我们更仔细地看看`std::vector`和`std::list`。

# 使用 std::vector 和 std::list

`std::vector`在`<vector>`头文件中定义。以下是最简单的使用示例：

```cpp
#include <vector>

int main()
{
  std::vector<int> vec;
  vec.push_back(4);
  vec.push_back(2);
  for (const auto& elem : vec) {
    std::cout << elem;
  }
}
```

`std::vector`是动态增长的。我们应该考虑增长因子。在声明一个向量时，它有一些默认容量，然后在插入元素时会增长。每当元素的数量超过向量的容量时，它会以给定的因子增加其容量（通常情况下，它会将其容量加倍）。如果我们知道我们将需要的向量中元素的大致数量，我们可以通过使用`reserve()`方法来为向量最初分配该容量来优化其使用。例如，以下代码保留了一个包含 10,000 个元素的容量：

```cpp
std::vector<int> vec;
vec.reserve(10000);
```

它强制向量为 10,000 个元素分配空间，从而避免在插入元素时进行调整大小（除非达到 10,000 个元素的阈值）。

另一方面，如果我们遇到容量远大于向量中实际元素数量的情况，我们可以缩小向量以释放未使用的内存。我们需要调用`shrink_to_fit()`函数，如下例所示：

```cpp
vec.shrink_to_fit();
```

这减少了容量以适应向量的大小。

访问向量元素的方式与访问常规数组的方式相同，使用`operator[]`。然而，`std::vector`提供了两种访问其元素的选项。其中一种被认为是安全的方法，通过`at()`函数进行，如下所示：

```cpp
std::cout << vec.at(2);
// is the same as
std::cout << vec[2];
// which is the same as
std::cout << vec.data()[2];
```

`at()`和`operator[]`之间的区别在于，`at()`通过边界检查访问指定的元素；也就是说，以下行会抛出`std::out_of_range`异常：

```cpp
try {
  vec.at(999999);
} catch (std::out_of_range& e) { }
```

我们几乎以相同的方式使用`std::list`。这些列表大多有相似的公共接口。在本章后面，我们将讨论迭代器，允许从特定容器中抽象出来，这样我们可以用一个向量替换一个列表而几乎没有任何惩罚。在此之前，让我们看看列表和向量的公共接口之间的区别。

除了两个容器都支持的标准函数集，如`size()`、`resize()`、`empty()`、`clear()`、`erase()`等，列表还有`push_front()`函数，它在列表的前面插入一个元素。这样做是有效的，因为`std::list`表示一个双向链表。如下所示，`std::list`也支持`push_back()`：

```cpp
std::list<double> lst;
lst.push_back(4.2);
lst.push_front(3.14);
// the list contains: "3.14 -> 4.2"
```

列表支持许多在许多情况下非常有用的附加操作。例如，要合并两个排序列表，我们使用`merge()`方法。它接受另一个列表作为参数，并将其所有元素移动到当前列表。传递给`merge()`方法的列表在操作后变为空。

STL 还提供了一个单向链表，由`std::forward_list`表示。要使用它，应该包含`<forward_list>`头文件。由于单向链表节点只有一个指针，所以在内存方面比双向链表更便宜。

`splice()`方法与`merge()`有些相似，不同之处在于它移动作为参数提供的列表的一部分。所谓移动，是指重新指向内部指针以指向正确的列表节点。这对于`merge()`和`splice()`都是成立的。

当我们使用容器存储和操作复杂对象时，复制元素的代价在程序性能中起着重要作用。考虑以下表示三维点的结构体：

```cpp
struct Point
{
  float x;
  float y;
  float z;

  Point(float px, float py, float pz)
    : x(px), y(py), z(pz)
  {}

  Point(Point&& p)
    : x(p.x), y(p.y), z(p.z)
  {}
};
```

现在，看看以下代码，它将一个`Point`对象插入到一个向量中：

```cpp
std::vector<Point> points;
points.push_back(Point(1.1, 2.2, 3.3));
```

首先构造一个临时对象，然后将其移动到向量的相应插槽中。我们可以用以下方式进行可视化表示：

![](img/d94643e3-cbfa-4816-8059-4ac126c1bbcb.png)

显然，向量事先占用更多空间，以尽可能延迟调整大小操作。当我们插入一个新元素时，向量将其复制到下一个可用插槽（如果已满，则重新分配更多空间）。我们可以利用该未初始化空间来创建一个新元素。向量提供了`emplace_back()`函数来实现这一目的。以下是我们如何使用它：

```cpp
points.emplace_back(1.1, 2.2, 3.3);
```

注意我们直接传递给函数的参数。以下插图描述了`emplace_back()`的使用：

![](img/47e01350-abf8-4a83-8eba-70afe1301af7.png)

`emplace_back()`通过`std::allocator_traits::construct()`构造元素。后者通常使用新操作符的放置来在已分配但未初始化的空间中构造元素。

`std::list`还提供了一个`emplace_front()`方法。这两个函数都返回插入的元素的引用。唯一的要求是元素的类型必须是`EmplaceConstructible`。对于向量，类型还应该是`MoveInsertable`。

# 使用容器适配器

你可能已经遇到了关于堆栈和队列的描述，它们被称为数据结构（或者在 C++术语中称为*容器*）。从技术上讲，它们不是数据结构，而是数据结构适配器。在 STL 中，`std::stack`和`std::queue`通过提供特殊的接口来访问容器来适配容器。术语*堆栈*几乎无处不在。到目前为止，我们已经用它来描述具有自动存储期限的对象的内存段。该段采用*堆栈*的名称，因为它的分配/释放策略。

我们说每次声明对象时，对象都会被推送到堆栈上，并在销毁时弹出。对象以它们被推送的相反顺序弹出。这就是称内存段为堆栈的原因。相同的**后进先出**（**LIFO**）方法适用于堆栈适配器。`std::stack`提供的关键函数如下：

```cpp
void push(const value_type& value);
void push(value_type&& value);
```

`push()`函数有效地调用基础容器的`push_back()`。通常，堆栈是使用向量实现的。我们已经在第三章中讨论过这样的情况，*面向对象编程的细节*，当我们介绍了受保护的继承。`std::stack`有两个模板参数；其中一个是容器。你选择什么并不重要，但它必须有一个`push_back()`成员函数。`std::stack`和`std::queue`的默认容器是`std::deque`。

`std::deque`允许在其开头和结尾快速插入。它是一个类似于`std::vector`的索引顺序容器。deque 的名称代表*双端队列*。

让我们看看堆栈的运行情况：

```cpp
#include <stack>

int main()
{
  std::stack<int> st;
  st.push(1); // stack contains: 1
  st.push(2); // stack contains: 2 1
  st.push(3); // stack contains: 3 2 1
}
```

`push()`函数的一个更好的替代方法是`emplace()`。它调用基础容器的`emplace_back()`，因此在原地构造元素。

要取出元素，我们调用`pop()`函数。它不接受任何参数，也不返回任何内容，只是从堆栈中移除顶部元素。要访问堆栈的顶部元素，我们调用`top()`函数。让我们修改前面的示例，在弹出元素之前打印所有堆栈元素：

```cpp
#include <stack>

int main()
{
  std::stack<int> st;
  st.push(1);
  st.push(2);
  st.push(3);
  std::cout << st.top(); // prints 3
  st.pop();
  std::cout << st.top(); // prints 2
  st.pop();
  std::cout << st.top(); // prints 1
  st.pop();
  std::cout << st.top(); // crashes application
}
```

`top()`函数返回对顶部元素的引用。它调用基础容器的`back()`函数。在空堆栈上调用`top()`函数时要注意。我们建议在对空堆栈调用`top()`之前检查堆栈的大小使用`size()`。

`queue`是另一个适配器，其行为与堆栈略有不同。队列背后的逻辑是它首先返回插入的第一个元素：它遵循**先进先出**（**FIFO**）原则。看下面的图表：

![](img/04d19255-e43e-485b-af2f-6269d220bd0e.png)

队列中插入和检索操作的正式名称是**enqeue**和**dequeue**。`std::queue`保持一致的方法，并提供`push()`和`pop()`函数。要访问队列的第一个和最后一个元素，应该使用`front()`和`back()`。两者都返回元素的引用。这里是一个简单的使用示例：

```cpp
#include <queue>

int main()
{
 std::queue<char> q;
  q.push('a');
  q.push('b');
  q.push('c');
  std::cout << q.front(); // prints 'a'
  std::cout << q.back(); // prints 'c'
  q.pop();
  std::cout << q.front(); // prints 'b'
}
```

当你正确应用它们时，了解各种容器和适配器是有用的。在选择所有类型问题的正确容器时，并没有银弹。许多编译器使用堆栈来解析代码表达式。例如，使用堆栈很容易验证以下表达式中的括号：

```cpp
int r = (a + b) + (((x * y) - (a / b)) / 4);
```

尝试练习一下。编写一个小程序，使用堆栈验证前面的表达式。

队列的应用更加广泛。我们将在第十一章中看到其中之一，*使用设计模式设计策略游戏*，在那里我们设计了一个策略游戏。

另一个容器适配器是`std::priority_queue`。优先队列通常适配平衡的、基于节点的数据结构，例如最大堆或最小堆。我们将在本章末尾讨论树和图，并看看优先队列在内部是如何工作的。

# 迭代容器

一个不可迭代的容器的概念就像一辆无法驾驶的汽车一样。毕竟，容器是物品的集合。迭代容器元素的常见方法之一是使用普通的`for`循环：

```cpp
std::vector<int> vec{1, 2, 3, 4, 5};
for (int ix = 0; ix < vec.size(); ++ix) {
  std::cout << vec[ix];
}
```

容器提供了一组不同的元素访问操作。例如，向量提供了`operator[]`，而列表则没有。`std::list`有`front()`和`back()`方法，分别返回第一个和最后一个元素。另外，正如前面讨论的，`std::vector`还提供了`at()`和`operator[]`。

这意味着我们不能使用前面的循环来迭代列表元素。但我们可以使用基于范围的`for`循环来遍历列表（和向量），如下所示：

```cpp
std::list<double> lst{1.1, 2.2, 3.3, 4.2};
for (auto& elem : lst) {
  std::cout << elem;
} 
```

这可能看起来令人困惑，但诀窍隐藏在基于范围的`for`实现中。它使用`std::begin()`函数检索指向容器第一个元素的迭代器。

**迭代器**是指向容器元素的对象，并且可以根据容器的物理结构前进到下一个元素。以下代码声明了一个`vector`迭代器，并用指向`vector`开头的迭代器进行初始化：

```cpp
std::vector<int> vec{1, 2, 3, 4};
std::vector<int>::iterator it{vec.begin()};
```

容器提供两个成员函数`begin()`和`end()`，分别返回指向容器开头和结尾的迭代器。以下图表显示了我们如何处理容器的开头和结尾：

![](img/4a058f5f-c5de-47fb-94e4-a5e25dbf0440.png)

使用基于范围的`for`迭代列表元素的先前代码可以被视为以下内容：

```cpp
auto it_begin = std::begin(lst);
auto it_end = std::end(lst);
for ( ; it_begin != it_end; ++it_begin) {
  std::cout << *it_begin;
}
```

注意我们在先前代码中使用的`*`运算符，通过迭代器访问底层元素。我们认为迭代器是对容器元素的*巧妙*指针。

`std::begin()`和`std::end()`函数通常调用容器的`begin()`和`end()`方法，但它们也适用于常规数组。

容器迭代器确切地知道如何处理容器元素。例如，向前推进向量迭代器会将其移动到数组的下一个槽位，而向前推进列表迭代器会使用相应的指针将其移动到下一个节点，如下面的代码所示：

```cpp
std::vector<int> vec;
vec.push_back(4);
vec.push_back(2);
std::vector<int>::iterator it = vec.begin();
std::cout << *it; // 4
it++;
std::cout << *it; // 2

std::list<int> lst;
lst.push_back(4);
lst.push_back(2);
std::list<int>::iterator lit = lst.begin();
std::cout << *lit; // 4
lit++;
std::cout << *lit; // 2
```

每个容器都有自己的迭代器实现；这就是为什么列表和向量迭代器有相同的接口但行为不同。迭代器的行为由其*类别*定义。例如，向量的迭代器是随机访问迭代器，这意味着我们可以使用迭代器随机访问任何元素。以下代码通过向量的迭代器访问第四个元素，方法是将`3`添加到迭代器上：

```cpp
auto it = vec.begin();
std::cout << *(it + 3);
```

STL 中有六种迭代器类别：

+   输入

+   输出（与输入相同，但支持写访问）

+   前向

+   双向

+   随机访问

+   连续

**输入迭代器**提供读取访问（通过调用`*`运算符）并使用前缀和后缀递增运算符向前推进迭代器位置。输入迭代器不支持多次遍历，也就是说，我们只能使用迭代器对容器进行一次遍历。另一方面，**前向迭代器**支持多次遍历。多次遍历支持意味着我们可以通过迭代器多次读取元素的值。

**输出迭代器**不提供对元素的访问，但它允许为其分配新值。具有多次遍历特性的输入迭代器和输出迭代器的组合构成了前向迭代器。然而，前向迭代器仅支持递增操作，而**双向迭代器**支持将迭代器移动到任何位置。它们支持递减操作。例如，`std::list`支持双向迭代器。

最后，**随机访问迭代器**允许通过向迭代器添加/减去一个数字来*跳跃*元素。迭代器将跳转到由算术操作指定的位置。`std::vector`提供了随机访问迭代器。

每个类别都定义了可以应用于迭代器的操作集。例如，输入迭代器可用于读取元素的值并通过递增迭代器前进到下一个元素。另一方面，随机访问迭代器允许以任意值递增和递减迭代器，读取和写入元素的值等。

到目前为止在本节中描述的所有特性的组合都属于**连续迭代器**类别，它也期望容器是一个连续的。这意味着容器元素保证紧邻在一起。`std::array`就是一个连续的容器的例子。

诸如`distance()`的函数使用迭代器的信息来实现最快的执行结果。例如，两个双向迭代器之间的`distance()`函数需要线性执行时间，而随机访问迭代器的相同函数在常数时间内运行。

以下伪代码演示了一个示例实现：

```cpp
template <typename Iter>
std::size_type distance(Iter first, Iter second) {
  if (Iter is a random_access_iterator) {
    return second - first; 
  }
  std::size_type count = 0;
  for ( ; first != last; ++count, first++) {}
  return count;
}
```

尽管前面示例中显示的伪代码运行良好，但我们应该考虑在运行时检查迭代器的类别不是一个选项。它是在编译时定义的，因此我们需要使用模板特化来生成随机访问迭代器的`distance()`函数。更好的解决方案是使用`<type_traits>`中定义的`std::is_same`类型特征：

```cpp
#include <iterator>
#include <type_traits>

template <typename Iter>
typename std::iterator_traits<Iter>::difference_type distance(Iter first, Iter last)
{
  using category = std::iterator_traits<Iter>::iterator_category;
  if constexpr (std::is_same_v<category, std::random_access_iterator_tag>) {
    return last - first;
  }
  typename std::iterator_traits<Iter>::difference_type count;
  for (; first != last; ++count, first++) {}
  return count;
}
```

`std::is_same_v`是`std::is_same`的辅助模板，定义如下：

```cpp
template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;
```

迭代器最重要的特性是提供了容器和算法之间的松耦合：

![](img/2d7a6c25-7b1f-4a4d-a3c1-80259c833393.png)

STL 基于这三个概念：容器、算法和迭代器。虽然向量、列表或任何其他容器都不同，它们都有相同的目的：存储数据。

另一方面，算法是处理数据的函数；它们大部分时间都与数据集合一起工作。算法定义通常代表了指定应采取哪些步骤来处理容器元素的通用方式。例如，排序算法按升序或降序对容器元素进行排序。

向量是连续的容器，而列表是基于节点的容器。对它们进行排序将需要更深入地了解特定容器的物理结构。为了正确地对向量进行排序，应该为它实现一个单独的排序函数。相同的逻辑也适用于列表。

迭代器将这种多样性的实现提升到了一个通用级别。它们为库设计者提供了实现只需处理迭代器的排序函数的能力，抽象出容器类型。在 STL 中，`sort()`算法（在`<algorithm>`中定义）处理迭代器，我们可以使用相同的函数对向量和列表进行排序：

```cpp
#include <algorithm>
#include <vector>
#include <list>
...
std::vector<int> vec;
// insert elements into the vector
std::list<int> lst;
// insert elements into the list

std::sort(vec.begin(), vec.end());
std::sort(lst.begin(), lst.end());
```

本节中描述的迭代器现在被认为是遗留特性。C++20 引入了基于**概念**的新迭代器系统。

# 概念和迭代器

C++20 将**概念**作为其主要特性之一引入。除了概念，C++20 还有基于概念的新迭代器。尽管本章讨论的迭代器现在被认为是遗留特性，但已经有大量的代码使用它们。这就是为什么我们在继续介绍新的迭代器概念之前首先介绍它们的原因。现在，让我们了解一下概念是什么，以及如何使用它们。

# 理解概念

抽象在计算机编程中是至关重要的。我们在第三章中引入了类，*面向对象编程的细节*，作为一种将数据和操作表示为抽象实体的方式。之后，在第四章中，*理解和设计模板*，我们深入研究了模板，并看到如何通过重用它们来使类变得更加灵活，以适用于各种聚合类型。模板不仅提供了对特定类型的抽象，还实现了实体和聚合类型之间的松耦合。例如，`std::vector`。它提供了一个通用接口来存储和操作对象的集合。我们可以轻松地声明三个包含三种不同类型对象的不同向量，如下所示：

```cpp
std::vector<int> ivec;
std::vector<Person> persons;
std::vector<std::vector<double>> float_matrix;
```

如果没有模板，我们将不得不对前面的代码做如下处理：

```cpp
std::int_vector ivec;
std::custom_vector persons; // supposing the custom_vector stores void* 
std::double_vector_vector float_matrix;
```

尽管前面的代码是不可接受的，但我们应该同意模板是泛型编程的基础。概念为泛型编程引入了更多的灵活性。现在可以对模板参数设置限制，检查约束，并在编译时发现不一致的行为。模板类声明的形式如下：

```cpp
template <typename T>
class Wallet
{
  // the body of the class using the T type
};
```

请注意前面代码块中的`typename`关键字。概念甚至更进一步：它们允许用描述模板参数的类型描述来替换它。假设我们希望`Wallet`能够处理可以相加的类型，也就是说，它们应该是*可加的*。以下是如何使用概念来帮助我们在代码中实现这一点：

```cpp
template <addable T>
class Wallet
{
  // the body of the class using addable T's
};
```

因此，现在我们可以通过提供可相加的类型来创建`Wallet`实例。每当类型不满足约束时，编译器将抛出错误。这看起来有点超自然。以下代码片段声明了两个`Wallet`对象：

```cpp
class Book 
{
  // doesn't have an operator+
  // the body is omitted for brevity
};

constexpr bool operator+(const Money& a, const Money& b) { 
  return Money{a.value_ + b.value_}; 
}

class Money
{
  friend constexpr bool operator+(const Money&, const Money&);
  // code omitted for brevity
private:
  double value_;
};

Wallet<Money> w; // works fine
Wallet<Book> g; // compile error
```

`Book`类没有`+`运算符，因此由于`template`参数类型限制，`g`的构造将失败。

使用`concept`关键字来声明概念，形式如下：

```cpp
template <*parameter-list*>
concept *name-of-the-concept* = *constraint-expression*;
```

正如你所看到的，概念也是使用模板来声明的。我们可以将它们称为描述其他类型的类型。概念在**约束**上有很大的依赖。约束是指定模板参数要求的一种方式，因此概念是一组约束。以下是我们如何实现前面的可加概念：

```cpp
template <typename T>
concept addable = requires (T obj) { obj + obj; }
```

标准概念在`<concepts>`头文件中定义。

我们还可以通过要求新概念支持其他概念来将几个概念合并为一个。为了实现这一点，我们使用`&&`运算符。让我们看看迭代器如何利用概念，并举例说明一个将其他概念结合在一起的`incrementable`迭代器概念。

# 在 C++20 中使用迭代器

在介绍概念之后，显而易见的是迭代器是首先充分利用它们的。迭代器及其类别现在被认为是遗留的，因为从 C++20 开始，我们使用迭代器概念，如**`readable`**（指定类型可通过应用`*`运算符进行读取）和`writable`（指定可以向迭代器引用的对象写入值）。正如承诺的那样，让我们看看`incrementable`在`<iterator>`头文件中是如何定义的：

```cpp
template <typename T>
concept incrementable = std::regular<T> && std::weakly_incrementable<T>
            && requires (T t) { {t++} -> std::same_as<T>; };
```

因此，可递增的概念要求类型为 std::regular。这意味着它应该可以通过默认方式构造，并且具有复制构造函数和 operator==()。除此之外，可递增的概念要求类型为 weakly_incrementable，这意味着该类型支持前置和后置递增运算符，除了不需要该类型是可比较相等的。这就是为什么可递增加入 std::regular 要求类型是可比较相等的。最后，附加的 requires 约束指出类型在递增后不应更改，也就是说，它应该与之前的类型相同。尽管 std::same_as 被表示为一个概念（在<concepts>中定义），在以前的版本中我们使用的是在<type_traits>中定义的 std::is_same。它们基本上做同样的事情，但是 C++17 版本的 std::is_same_v 很啰嗦，带有额外的后缀。

因此，现在我们不再提到迭代器类别，而是提到迭代器概念。除了我们之前介绍的概念，还应该考虑以下概念：

+   输入迭代器指定该类型允许读取其引用值，并且可以进行前置和后置递增。

+   输出迭代器指定该类型的值可以被写入，并且该类型可以进行前置和后置递增。

+   输入或输出迭代器，除了名称过长之外，指定该类型是可递增的，并且可以被解引用。

+   前向迭代器指定该类型是一个输入迭代器，此外还支持相等比较和多遍历。

+   双向迭代器指定该类型支持前向迭代器，并且还支持向后移动。

+   随机访问迭代器指定该类型为双向迭代器，支持常数时间的前进和下标访问。

+   连续迭代器指定该类型是一个随机访问迭代器，指的是内存中连续的元素。

它们几乎重复了我们之前讨论的传统迭代器，但现在它们可以在声明模板参数时使用，这样编译器将处理其余部分。

# 掌握算法

正如前面提到的，算法是接受一些输入，处理它，并返回输出的函数。通常，在 STL 的上下文中，算法意味着处理数据集合的函数。数据集合以容器的形式呈现，例如 std::vector、std::list 等。

选择高效的算法是程序员日常工作中的常见任务。例如，使用二分搜索算法搜索排序后的向量将比使用顺序搜索更有效。为了比较算法的效率，进行所谓的渐近分析，考虑算法速度与输入数据大小的关系。这意味着我们实际上不应该将两个算法应用于一个包含十个或一百个元素的容器进行比较。

算法的实际差异在应用于足够大的容器时才会显现，比如有一百万甚至十亿条记录的容器。衡量算法的效率也被称为验证其复杂性。您可能遇到过 O(n)算法或 O(log N)算法。O()函数（读作 big-oh）定义了算法的复杂性。

让我们来看看搜索算法，并比较它们的复杂性。

# 搜索

在容器中搜索元素是一个常见的任务。让我们实现在向量中进行顺序搜索元素。

```cpp
template <typename T>
int search(const std::vector<T>& vec, const T& item)
{
  for (int ix = 0; ix < vec.size(); ++ix) {
    if (vec[ix] == item) {
      return ix;
    }
  }
  return -1; // not found
}
```

这是一个简单的算法，它遍历向量并返回元素等于作为搜索键传递的值的索引。我们称之为顺序搜索，因为它按顺序扫描向量元素。它的复杂性是线性的：*O(n)*。为了衡量它，我们应该以某种方式定义算法找到结果所需的操作数。假设向量包含 *n* 个元素，下面的代码在搜索函数的每一行都有关于其操作的注释：

```cpp
template <typename T>
int search(const std::vector<T>& vec, const T& item)
{
  for (int ix = 0;           // 1 copy
       ix < vec.size;        // n + 1 comparisons 
       ++ix)                 // n + 1 increments
  {  
    if (vec[ix] == item) {   // n comparisons
      return ix;             // 1 copy
    }
  }
  return -1;                 // 1 copy
}
```

我们有三种复制操作，*n + 1* 和 *n*（也就是 *2n + 1*）次比较，以及 *n + 1* 次增量操作。如果所需元素在向量的第一个位置怎么办？那么，我们只需要扫描向量的第一个元素并从函数中返回。

然而，这并不意味着我们的算法非常高效，只需要一步就能完成任务。为了衡量算法的复杂性，我们应该考虑最坏情况：所需元素要么不存在于向量中，要么位于向量的最后位置。下图显示了我们即将找到的元素的三种情况：

![](img/89c34b2d-9597-4ea4-ac1a-a32ef5031eb7.png)

我们只需要考虑最坏情况，因为它也涵盖了所有其他情况。如果我们为最坏情况定义算法的复杂性，我们可以确保它永远不会比那更慢。

为了找出算法的复杂性，我们应该找到操作次数和输入大小之间的关系。在这种情况下，输入的大小是容器的长度。让我们将复制记为 A，比较记为 C，增量操作记为 I，这样我们就有 3A + (2n + 1)C + (n + 1)I 次操作。算法的复杂性将定义如下：

*O(3A + (2n + 1)C + (n + 1)I)*

这可以以以下方式简化：

+   *O(3A + (2n + 1)C + (n + 1)I) =*

+   *O(3A + 2nC + C + nI + I) = *

+   *O(n(2C + I) + (3A + C + I)) = *

+   *O(n(2C + I))*

最后，*O()*的属性使我们可以摆脱常数系数和较小的成员，因为实际算法的复杂性只与输入的大小有关，即 *n*，我们得到最终复杂性等于 *O(n)*。换句话说，顺序搜索算法具有线性时间复杂性。

正如前面提到的，STL 的本质是通过迭代器连接容器和算法。这就是为什么顺序搜索实现不被认为是 STL 兼容的：因为它对输入参数有严格的限制。为了使其通用，我们应该考虑仅使用迭代器来实现它。为了涵盖各种容器类型，使用前向迭代器。下面的代码使用了`Iter`类型的操作符，假设它是一个前向迭代器：

```cpp
template <typename Iter, typename T>
int search(Iter first, Iter last, const T& elem)
{
  for (std::size_t count = 0; first != last; first++, ++count) {
    if (*first == elem) return count;
  }
  return -1;
}
...
std::vector<int> vec{4, 5, 6, 7, 8};
std::list<double> lst{1.1, 2.2, 3.3, 4.4};

std::cout << search(vec.begin(), vec.end(), 5);
std::cout << search(lst.begin(), lst.end(), 5.5);
```

实际上，任何类型的迭代器都可以传递给`search()`函数。我们通过对迭代器本身应用操作来确保使用前向迭代器。我们只使用增量（向前移动）、读取（`*`运算符）和严格比较（`==`和`!=`），这些操作都受前向迭代器支持。

# 二分搜索

另一方面是二分搜索算法，这个算法很容易解释。首先，它查找向量的中间元素并将搜索键与之进行比较，如果相等，算法就结束了：它返回索引。否则，如果搜索键小于中间元素，算法继续向向量的左侧进行。如果搜索键大于中间元素，算法继续向右侧子向量进行。

为了使二分搜索在向量中正确工作，它应该是排序的。二分搜索的核心是将搜索键与向量元素进行比较，并继续到左侧或右侧子向量，每个子向量都包含与向量中间元素相比较的较小或较大的元素。看一下下面的图表，它描述了二分搜索算法的执行过程：

![](img/c478e0fd-ae7e-4b99-8bec-7c288cd13272.png)

二分搜索算法有一个优雅的递归实现（尽管最好使用迭代实现）-在下面的代码中看一下：

```cpp
template <typename T>
std::size_t binsearch(const std::vector<T>& vec, const T& item, int start, int end)
{
  if (start > end) return -1;
  int mid = start + (end - start) / 2;
  if (vec[mid] == item) {
    return mid; // found
  }
  if (vec[mid] > item) {
    return binsearch(vec, item, start, mid - 1);
  }
  return binsearch(vec, item, mid + 1, end);
}
```

注意中间元素的计算。我们使用了`start + (end - start) / 2;`技术，而不是`(start + end) / 2;`，只是为了避免二分搜索实现中的著名错误（假设我们没有留下其他错误）。关键是对于 start 和 end 的大值，它们的和（*start + end*）会产生整数溢出，这将导致程序在某个时刻崩溃。

现在让我们找到二分搜索的复杂度。很明显，在执行的每一步中，源数组都会减半，这意味着我们在下一步中处理它的较小或较大的一半。这意味着最坏情况是将向量分割到只剩下一个或没有元素的情况。为了找到算法的步数，我们应该根据向量的大小找到分割的次数。如果向量有 10 个元素，那么我们将它分成一个包含五个元素的子向量；再次分割，我们得到一个包含两个元素的子向量，最后，再次分割将带我们到一个单一元素。因此，对于包含 10 个元素的向量，分割的次数是 3。对于包含*n*个元素的向量，分割的次数是*log(n)*，因为在每一步中，*n*变为*n/2*，然后变为*n/4*，依此类推。二分搜索的复杂度是*O(logn)*（即对数）。

STL 算法定义在`<algorithm>`头文件中；二分搜索的实现也在其中。STL 实现如果元素存在于容器中则返回 true。看一下它的原型：

```cpp
template <typename Iter, typename T>
bool binary_search(Iter start, Iter end, const T& elem);
```

STL 算法不直接与容器一起工作，而是与迭代器一起工作。这使我们能够抽象出特定的容器，并使用`binary_search()`来支持前向迭代器的所有容器。下面的示例调用了`binary_search()`函数，用于向量和列表：

```cpp
#include <vector>
#include <list>
#include <algorithm>
...
std::vector<int> vec{1, 2, 3, 4, 5};
std::list<int> lst{1, 2, 3, 4};
binary_search(vec.begin(), vec.end(), 8);
binary_search(lst.begin(), lst.end(), 3);
```

`binary_search()`检查迭代器的类别，在随机访问迭代器的情况下，它使用二分搜索算法的全部功能（否则，它将退回到顺序搜索）。

# 排序

二分搜索算法仅适用于排序的容器。对于计算机程序员来说，排序是一个众所周知的古老任务，现在他们很少编写自己的排序算法实现。你可能多次使用了`std::sort()`而不关心它的实现。基本上，排序算法接受一个集合作为输入，并返回一个新的排序集合（按照算法用户定义的顺序）。

在众多的排序算法中，最流行的（甚至是最快的）是**快速排序**。任何排序算法的基本思想都是找到较小（或较大）的元素，并将它们与较大（或较小）的元素交换，直到整个集合排序。例如，选择排序逻辑上将集合分为两部分，已排序和未排序，其中已排序的子数组最初为空，如下所示：

![](img/303865f2-ae26-44a9-bce2-0f8cefb9cc6f.png)

算法开始在未排序的子数组中寻找最小的元素，并通过与未排序的子数组的第一个元素交换将其放入已排序的子数组中。每一步之后，已排序子数组的长度增加了一个，而未排序子数组的长度减少了，如下所示：

![](img/995dc143-d05a-4a3d-b808-de3d058583c5.png)

该过程持续进行，直到未排序的子数组变为空。

STL 提供了`std::sort()`函数，接受两个随机访问迭代器：

```cpp
#include <vector>
#include <algorithm>
...
std::vector<int> vec{4, 7, -1, 2, 0, 5};
std::sort(vec.begin(), vec.end());
// -1, 0, 2, 4, 5, 7
```

`sort`函数不能应用于`std::list`，因为它不支持随机访问迭代器。相反，应该调用列表的`sort()`成员函数。尽管这与 STL 具有通用函数的想法相矛盾，但出于效率考虑而这样做。

`sort()`函数有一个第三个参数：一个比较函数，用于比较容器元素。假设我们在向量中存储`Product`对象：

```cpp
struct Product
{
  int price;
  bool available;
  std::string title;
};

std::vector<Product> products;
products.push_back({5, false, "Product 1"});
products.push_back({12, true, "Product 2"});
```

为了正确排序容器，其元素必须支持小于运算符，或`<`。我们应该为我们的自定义类型定义相应的运算符。但是，如果我们为我们的自定义类型创建一个单独的比较函数，就可以省略运算符定义，如下面的代码块所示：

```cpp
class ProductComparator
{
public:
 bool operator()(const Product& a, const Product& b) {
 return a.price > b.price;
 }
};
```

将`ProductComparator`传递给`std::sort()`函数允许它比较向量元素，而无需深入了解其元素的类型，如下所示：

```cpp
std::sort(products.begin(), products.end(), ProductComparator{});
```

虽然这是一个不错的技术，但更优雅的做法是使用 lambda 函数，它们是匿名函数，非常适合前面提到的场景。以下是我们如何覆盖它的方法：

```cpp
std::sort(products.begin(), products.end(), 
  [](const Product& a, const Product& b) { return a.price > b.price; })
```

上述代码允许省略`ProductComparator`的声明。

# 探索树和图

二叉搜索算法和排序算法结合在一起，引出了默认按排序方式保持项目的容器的想法。其中一个这样的容器是基于平衡树的`std::set`。在讨论平衡树本身之前，让我们先看看二叉搜索树，这是一个快速查找的完美候选者。

二叉搜索树的思想是，节点的左子树的值小于节点的值。相比之下，节点的右子树的值大于节点的值。以下是一个二叉搜索树的示例：

![](img/9c361b58-ff03-4ab1-bffc-dd29595f2378.png)

如前面的图表所示，值为 15 的元素位于左子树中，因为它小于 30（根元素）。另一方面，值为 60 的元素位于右子树中，因为它大于根元素。相同的逻辑适用于树的其余元素。

二叉树节点表示为一个包含项目和指向每个子节点的两个指针的结构。以下是树节点的示例代码表示：

```cpp
template <typename T>
struct tree_node
{
  T item;
  tree_node<T>* left;
  tree_node<T>* right;
};
```

在完全平衡的二叉搜索树中，搜索、插入或删除元素需要*O(logn)*的时间。STL 没有为树提供单独的容器，但它有基于树实现的类似容器。例如，`std::set`容器是基于平衡树的，可以按排序顺序唯一存储元素：

```cpp
#include <set>
...
std::set<int> s{1, 5, 2, 4, 4, 4, 3};
// s has {1, 2, 3, 4, 5}
```

`std::map`也是基于平衡树，但它提供了一个将键映射到某个值的容器，例如：

```cpp
#include <map>
...
std::map<int, std::string> numbers;
numbers[3] = "three";
numbers[4] = "four";
...
```

如前面的代码所示，`map` `numbers`函数将整数映射到字符串。因此，当我们告诉地图将`3`的值存储为键，`three`的字符串作为值时，它会向其内部树添加一个新节点，其键等于`3`，值等于`three`。

`set`和`map`操作是对数的，这使得它在大多数情况下成为非常高效的数据结构。然而，更高效的数据结构接下来就要出现。

# 哈希表

哈希表是最快的数据结构。它基于一个简单的向量索引的想法。想象一个包含指向列表的指针的大向量：

```cpp
std::vector<std::list<T> > hash_table;
```

访问向量元素需要常数时间。这是向量的主要优势。哈希表允许我们使用任何类型作为容器的键。哈希表的基本思想是使用精心策划的哈希函数，为输入键生成唯一的索引。例如，当我们使用字符串作为哈希表键时，哈希表使用哈希函数将哈希作为底层向量的索引值：

```cpp
template <typename T>
int hash(const T& key)
{
  // generate and return and efficient
  // hash value from key based on the key's type
}

template <typename T, typename U>
void insert_into_hashtable(const T& key, const U& value)
{
  int index = hash(key);
  hash_table[index].push_back(value); // insert into the list
}
```

以下是我们如何说明哈希表：

![](img/5623724b-8217-4b70-8fac-b52b713d8435.png)

访问哈希表需要常数时间，因为它是基于向量操作的。虽然可能会有不同的键导致相同的哈希值，但这些冲突通过使用值列表作为向量元素来解决（如前图所示）。

STL 支持名为`std::unordered_map`的哈希表：

```cpp
#include <unordered_map>
...
std::unordered_map<std::string, std::string> hashtable;
hashtable["key1"] = "value 1";
hashtable["key2"] = "value 2";
...
```

为了为提供的键生成哈希值，函数`std::unordered_map`使用`<functional>`头文件中定义的`std::hash()`函数。您可以为哈希函数指定自定义实现。`std::unordered_map`的第三个`template`参数是哈希函数，默认为`std::hash`。

# 图

二叉搜索树的平衡性是基于许多搜索索引实现的。例如，数据库系统使用称为 B 树的平衡树进行表索引。B 树不是*二叉*树，但它遵循相同的平衡逻辑，如下图所示：

![](img/92b32da1-8667-447b-b087-481c79ac0dc4.png)

另一方面，图表示没有适当顺序的连接节点：

![](img/84949559-6f1a-41a6-8b34-746c60392218.png)

假设我们正在构建一个最终将击败 Facebook 的社交网络。社交网络中的用户可以互相关注，这可以表示为图。例如，如果 A 关注 B，B 关注 C，C 既关注 B 又同时关注 A，那么我们可以将关系表示为以下图：

![](img/afd3b4ab-05ed-448a-8612-d596cce84d88.png)

在图中，一个节点被称为**顶点**。两个节点之间的链接被称为**边**。实际上并没有固定的图表示，所以我们应该从几种选择中进行选择。让我们想想我们的社交网络 - 我们如何表示用户 A 关注用户 B 的信息？

这里最好的选择之一是使用哈希表。我们可以将每个用户映射到他们关注的所有用户：

![](img/ff9a742f-9d83-4b1d-b655-77d3f57fe938.png)

图的实现变成了混合容器：

```cpp
#include <list>
#include <unordered_map>

template <typename T>
class Graph
{
public: 
  Graph();
  ~Graph();
  // copy, move constructors and assignment operators omitted for brevity

public:
  void insert_edge(const T& source, const T& target);
  void remove_edge(const T& source, const T& target);

  bool connected(const T& source, const T& target);

private:
  std::unordered_map<T, std::list<T> > hashtable_;
};
```

为了使其成为 STL 兼容的容器，让我们为图添加一个迭代器。虽然迭代图不是一个好主意，但添加迭代器并不是一个坏主意。

# 字符串

字符串类似于向量：它们存储字符，公开迭代器，并且它们是容器。但是，它们有些不同，因为它们专门表示一种数据：字符串。下图描述了字符串**hello, C++**作为以特殊**\0**字符结尾的字符数组：

![](img/c99154de-7f04-4bfe-8ae5-b8ba7ed23ecb.png)

特殊的**\0**字符（也称为空字符）用作字符串终止符。编译器会依次读取字符，直到遇到空字符为止。

字符串的实现方式与我们在本章开头实现向量的方式相同：

```cpp
class my_string
{
public:
 my_string();
 // code omitted for brevity

public:
 void insert(char ch);
 // code omitted for brevity

private:
 char* buffer_;
 int size_;
 int capacity_;
};
```

C++有其强大的`std::string`类，提供了一堆用于处理的函数。除了`std::string`成员函数外，`<algorithm>`中定义的算法也适用于字符串。

# 摘要

数据结构和算法在开发高效软件方面至关重要。通过理解和利用本章讨论的数据结构，您将充分利用 C++20 的功能，使程序运行更快。程序员具有强大的问题解决能力在市场上更受欢迎，这并不是秘密。首先要通过深入理解基本算法和数据结构来获得问题解决能力。正如您在本章中已经看到的，使用二分搜索算法在搜索任务中使代码运行速度比顺序搜索快得多。高效的软件节省时间并提供更好的用户体验，最终使您的软件成为现有软件的杰出替代品。

在本章中，我们讨论了基本数据结构及其区别。我们学会了根据问题分析来使用它们。例如，在需要随机查找的问题中应用链表被认为是耗时的，因为链表元素访问操作的复杂性。在这种情况下，使用动态增长的向量更合适，因为它具有常数时间的元素访问。相反，在需要在容器的前面快速插入的问题中使用向量比如列表更昂贵。

本章还介绍了算法以及衡量它们效率的方法。我们比较了几个问题，以应用更好的算法更有效地解决它们。

在下一章中，我们将讨论 C++中的函数式编程。在学习了 STL 的基本知识后，我们现在将在容器上应用函数式编程技术。

# 问题

1.  描述将元素插入动态增长的向量。

1.  在链表的前面插入元素和在向量的前面插入元素有什么区别？

1.  实现一个混合数据结构，它将元素存储在向量和列表中。对于每个操作，选择具有最快实现该操作的基础数据结构。

1.  如果我们按顺序插入 100 个元素，二叉搜索树会是什么样子呢？

1.  选择排序和插入排序算法有什么区别？

1.  实现本章描述的排序算法，即计数排序。

# 进一步阅读

有关更多信息，请参考以下资源：

+   *Jon Bentley 著的* *Programming Pearls* ，可从[`www.amazon.com/Programming-Pearls-2nd-Jon-Bentley/dp/0201657880/`](https://www.amazon.com/Programming-Pearls-2nd-Jon-Bentley/dp/0201657880/)获取。

+   *Data Abstraction and Problem Solving Using C++: Walls and Mirrors* by Frank Carrano 和 Timothy Henry，可从[`www.amazon.com/Data-Abstraction-Problem-Solving-Mirrors/dp/0134463978/`](https://www.amazon.com/Data-Abstraction-Problem-Solving-Mirrors/dp/0134463978/)获取。

+   *Cormen, Leiserson, Rivest, and Stein 著的* *Introduction to Algorithms* ，可从[`www.amazon.com/Introduction-Algorithms-3rd-MIT-Press/dp/0262033844/`](https://www.amazon.com/Introduction-Algorithms-3rd-MIT-Press/dp/0262033844/)获取。

+   *Wisnu Anggoro 著的* *C++ Data Structures and Algorithms* ，可从[`www.packtpub.com/application-development/c-data-structures-and-algorithms`](https://www.packtpub.com/application-development/c-data-structures-and-algorithms)获取。
