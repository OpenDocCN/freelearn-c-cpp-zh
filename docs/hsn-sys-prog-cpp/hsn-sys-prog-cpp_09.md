# 第九章：分配器的实践方法

在第七章中，*全面了解内存管理*，我们学习了如何使用 C++特定的技术来分配和释放内存，包括使用`std::unique_ptr`和`std::shared_ptr`。此外，我们还了解了碎片化以及根据内存分配和后续释放的方式可能浪费大量内存。系统程序员经常需要从不同的池中分配内存（有时来自不同的来源），并处理碎片以防止系统在运行过程中耗尽内存。这对于嵌入式程序员来说尤其如此。可以使用放置`new()`来解决这些问题，但基于放置 new 的实现通常很难创建，甚至更难维护。放置`new()`也只能从用户定义的代码中访问，无法控制源自 C++标准库 API（如`std::list`和`std::map`）的分配。

为了解决这些问题，C++提供了一个称为**分配器**的概念。C++分配器定义了如何为特定类型 T 分配和释放内存。在本章中，您将学习如何创建自己的分配器，同时涵盖 C++分配器概念的复杂细节。本章将以两个不同的示例结束；第一个示例将演示如何创建一个简单的、缓存对齐的无状态分配器，而第二个示例将提供一个有状态对象分配器的功能示例，该分配器维护一个用于快速分配的空闲池。

本章的目标如下：

+   介绍 C++分配器

+   研究无状态的、缓存对齐的分配器的示例

+   研究有状态的、内存池分配器的示例

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   一个能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请参阅以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter09`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter09)。

# 介绍 C++分配器

C++分配器定义了一个模板类，为特定类型 T 分配内存，并由分配器概念定义。有两种不同类型的分配器：

+   相等的分配器

+   不相等的分配器

相等的分配器是指可以从一个分配器中分配内存并从另一个分配器中释放内存的分配器，例如：

```cpp
myallocator<myclass> myalloc1;
myallocator<myclass> myalloc2;

auto ptr = myalloc1.allocate(1);
myalloc2.deallocate(ptr, 1);
```

在前面的例子中，我们创建了两个`myallocator{}`的实例。我们从一个分配器中分配内存，然后从另一个分配器中释放内存。为了使这有效，分配器必须是相等的：

```cpp
myalloc1 == myalloc2; // true
```

如果这不成立，分配器被认为是不相等的，这极大地复杂了分配器的使用方式。不相等的分配器通常是有状态的分配器，这意味着它在自身内部存储了一个状态，阻止了一个分配器从另一个相同分配器的实例中释放内存（因为状态不同）。

# 学习基本分配器

在我们深入研究有状态的、不相等的分配器的细节之前，让我们回顾一下最基本的分配器，即无状态的、相等的分配器。这个最基本的分配器采用以下形式：

```cpp
template<typename T>
class myallocator
{
public:

 using value_type = T;
 using pointer = T *;
 using size_type = std::size_t;

public:

 myallocator() = default;

 template <typename U>
 myallocator(const myallocator<U> &other) noexcept
 { (void) other; }

 pointer allocate(size_type n)
 {
 if (auto ptr = static_cast<pointer>(malloc(sizeof(T) * n))) {
 return ptr;
 }

 throw std::bad_alloc();
 }

 void deallocate(pointer p, size_type n)
 { (void) n; return free(p); }
};

template <typename T1, typename T2>
bool operator==(const myallocator<T1> &, const myallocator<T2> &)
{ return true; }

template <typename T1, typename T2>
bool operator!=(const myallocator<T1> &, const myallocator<T2> &)
{ return false; }
```

首先，所有分配器都是模板类，如下所示：

```cpp
template<typename T>
class myallocator
```

应该注意，分配器可以具有任意数量的模板参数，但至少需要一个来定义分配器将分配和释放的类型。在我们的示例中，我们使用以下别名：

```cpp
using value_type = T;
using pointer = T *;
using size_type = std::size_t;
```

从技术上讲，唯一需要的别名是以下内容：

```cpp
using value_type = T;
```

然而，由于需要`T*`和`std::size_t`来创建最小的分配器，这些别名也可以添加以提供更完整的实现。可选的别名包括以下内容：

```cpp
using value_type = T;
using pointer = T *;
using const_pointer = const T *;
using void_pointer = void *;
using const_void_pointer = const void *;
using size_type = std::size_t;
using difference_type = std::ptrdiff_t;
```

如果自定义分配器没有提供这些内容，将为您提供前面的默认值。

如所示，所有分配器必须提供默认构造函数。这是因为 C++容器将自行创建分配器，在某些情况下可能会多次创建，并且它们将使用默认构造函数来执行此操作，这意味着必须能够在不需要额外参数的情况下构造分配器。

我们示例中的`allocate()`函数如下：

```cpp
pointer allocate(size_type n)
{
    if (auto ptr = static_cast<pointer>(malloc(sizeof(T) * n))) {
        return ptr;
    }

    throw std::bad_alloc();
}
```

与本示例中解释的所有函数一样，`allocate()`函数的函数签名由分配器概念定义，这意味着分配器中的每个函数必须采用特定的签名；否则，在现有容器使用时，分配器将无法正确编译。

在前面的示例中，使用`malloc()`来分配一些内存，如果`malloc`没有返回`nullptr`，则返回结果指针。由于分配器分配`T*`类型的指针，而不是`void *`，我们必须在返回指针之前对`malloc()`的结果进行静态转换。提供给`malloc()`的字节数等于`sizeof(T) * n`。这是因为`n`参数定义了分配器必须分配的对象总数——因为一些容器将一次分配多个对象，并且期望被分配的对象在内存中是连续的。这包括`std::deque`和`std::vector`的示例，分配器必须确保这些规则在内存中成立。最后，如果`malloc()`返回`nullptr`，表示无法分配请求的内存，我们会抛出`std::bad_alloc()`。

应该注意的是，在我们的示例中，我们使用`malloc()`而不是`new()`。在这里，应该使用`malloc()`而不是`new()`，因为容器将为您构造被分配的对象。因此，我们不希望使用`new()`，因为它也会构造对象，这意味着对象将被构造两次，这将导致损坏和未定义的行为。因此，`new()`和`delete()`不应该在分配器中使用。

`deallocate`函数执行与`allocate`函数相反的操作，释放内存并将其释放回操作系统：

```cpp
void deallocate(pointer p, size_type n)
{ (void) n; free(p); }
```

在前面的示例中，要释放内存，我们只需要调用`free()`。请注意，我们创建了一个*相等*的分配器，这意味着`ptr`不需要来自执行解除分配的相同分配器。然而，分配的数量`n`必须与原始分配相匹配，在我们的情况下可能可以安全地忽略，因为我们使用的是`malloc()`和`free()`，它们会自动为我们跟踪原始分配的大小。并非所有的分配器都具有这个属性。

在我们的简单示例中，有两个额外的要求，以符合 C++分配器，这些要求在其目的方面远不那么明显。第一个是使用模板类型`U`的复制构造函数，如下所示：

```cpp
template <typename U>
myallocator(const myallocator<U> &other) noexcept
{ (void) other; }
```

这是因为当您在容器的定义中使用分配器时，您会指定容器中的类型，例如：

```cpp
std::list<myclass, myallocator<myclass>> mylist;
```

在前面的示例中，我们创建了一个`myclass{}`类型的`std::list`，使用一个分配器来分配和释放`myclass{}`对象。问题是，`std::list`有自己的内部数据结构，也必须进行分配。具体来说，`std::list`实现了一个链表，因此`std::list`必须能够分配和释放链表节点。在前面的定义中，我们定义了一个分配器，用于分配和释放`myclass{}`对象，但`std::list`实际上将分配和释放节点，这两种类型并不相同。为了解决这个问题，`std::list`将使用复制构造函数的模板版本创建`myclass{}`分配器的副本，从而使`std::list`能够使用最初提供的分配器来创建自己的节点分配器。因此，完全功能的分配器需要模板版本的复制构造函数。

前面示例中前面的奇怪添加是使用相等运算符，如下所示：

```cpp
template <typename T1, typename T2>
bool operator==(const myallocator<T1> &, const myallocator<T2> &)
{ return true; }

template <typename T1, typename T2>
bool operator!=(const myallocator<T1> &, const myallocator<T2> &)
{ return false; }
```

相等运算符定义了分配器是*相等*还是*不相等*。在前面的示例中，我们创建了一个无状态的分配器，这意味着以下是有效的：

```cpp
myallocator<int> myalloc1;
myallocator<int> myalloc2;

auto ptr = myalloc1.allocate(1);
myalloc2.deallocate(ptr, 1);
```

如果前面的属性成立，那么分配器是相等的。由于在我们的示例中，`myalloc1{}`在分配时调用`malloc()`，在释放时调用`free()`，我们知道它们是可以互换的，这意味着前面的属性成立，我们的示例实现了一个*相等*的分配器。前面的相等运算符只是正式陈述了这种相等关系，为 C++容器等提供了根据需要创建新分配器的 API。

# 了解分配器的属性和选项

我们刚刚讨论的基本分配器仅提供了使用现有 C++数据结构（以及利用对象分配的其他用户定义类型）的分配器所需的功能。除了我们讨论的可选别名之外，还有几个其他选项和属性构成了 C++分配器。

# 学习属性

C++分配器必须遵守一定的属性集，其中大多数要么是显而易见的，要么很容易遵守。

# 值指针类型

第一组属性确保分配器返回的指针类型实际上是一个指针：

```cpp
myallocator<myclass> myalloc;

myclass *ptr = myalloc.allocate(1);
const myclass *cptr = myalloc.allocate(1);

std::cout << (*ptr).data1 << '\n';
std::cout << (*cptr).data2 << '\n';

std::cout << ptr->data1 << '\n';
std::cout << cptr->data2 << '\n';

// 0
// 32644
// 0
// 32644
```

如果分配器返回的指针确实是一个指针，就可以对指针进行解引用以访问其指向的内存，如前面的示例所示。还应该注意，在这个例子中，当尝试将分配的内存输出到`stdout`时，返回的值是相对随机的。这是因为分配器没有要求将内存清零，因为使用这个内存的容器会为我们执行此操作，这样更高效。

# 相等性

如前所述，如果比较时分配器相等，则返回`true`，如下所示：

```cpp
myallocator<myclass> myalloc1;
myallocator<myclass> myalloc2;

std::cout << std::boolalpha;
std::cout << (myalloc1 == myalloc2) << '\n';
std::cout << (myalloc1 != myalloc2) << '\n';

// true
// false
```

如果同一类型的两个分配器返回`true`，这意味着使用此分配器的容器可以自由地使用不同实例的相同分配器来分配和释放内存，从而最终实现了某些优化的使用。例如，容器可以从不实际存储分配器的内部引用，而是只在需要分配内存时创建一个分配器。从那时起，容器在内部管理内存，并且只在销毁时释放内存，此时容器将再次创建另一个分配器来执行释放操作，再次假设分配器相等。

正如我们所讨论的，分配器的相等通常与状态有关。通常，有状态的分配器不相等，而无状态的分配器相等；但这个规则并不总是成立，特别是当对有状态的分配器进行复制时，规范要求提供相等性（或者至少能够释放从副本分配的先前分配的内存）。当我们涉及有状态的分配器时，我们将提供更多细节。

在 C++17 之前，分配器存在一个问题，即容器没有简单的方法来确定分配器是否相等，而不是在初始化时首先创建两个相同分配器的实例，进行比较，然后根据结果设置内部状态。由于 C++分配器概念的这种限制，容器要么假定是无状态的分配器（这是旧版本 C++库的情况），要么假定所有分配器都是有状态的，从而消除了优化的可能性。

为了克服这一问题，C++17 引入了以下内容：

```cpp
using is_always_equal = std::true_type;
```

如果您的分配器没有提供这个功能，就像前面的例子一样，默认值是`std::empty`，告诉容器需要使用旧式比较来确定相等性。如果提供了这个别名，容器将知道如何对自身进行优化。

# 不同的分配类型

容器如何分配内存完全取决于容器的类型，因此，分配器必须能够支持不同的分配类型，例如以下内容：

+   分配器的所有分配必须在内存中是连续的。不要求一个分配在内存中与另一个分配是连续的，但每个单独的分配必须是连续的。

+   分配器必须能够在单个分配中分配多个元素。这有时可能会有问题，这取决于分配器。

为了探讨这些属性，让我们使用以下示例：

```cpp
template<typename T>
class myallocator
{
public:

    using value_type = T;
    using pointer = T *;
    using size_type = std::size_t;
    using is_always_equal = std::true_type;

public:

    myallocator()
    {
        std::cout << this << " constructor, sizeof(T): "
                  << sizeof(T) << '\n';
    }

    template <typename U>
    myallocator(const myallocator<U> &other) noexcept
    { (void) other; }

    pointer allocate(size_type n)
    {
        if (auto ptr = static_cast<pointer>(malloc(sizeof(T) * n))) {
            std::cout << this << " A [" << n << "]: " << ptr << '\n';
            return ptr;
        }

        throw std::bad_alloc();
    }

    void deallocate(pointer p, size_type n)
    {
        (void) n;

        std::cout << this << " D [" << n << "]: " << p << '\n';
        free(p);
    }
};

template <typename T1, typename T2>
bool operator==(const myallocator<T1> &, const myallocator<T2> &)
{ return true; }

template <typename T1, typename T2>
bool operator!=(const myallocator<T1> &, const myallocator<T2> &)
{ return false; }
```

前面的分配器与第一个分配器相同，唯一的区别是在构造函数和分配和释放函数中添加了调试语句，这样我们就可以看到容器是如何分配内存的。

让我们来看一个简单的`std::list`的例子：

```cpp
std::list<int, myallocator<int>> mylist;
mylist.emplace_back(42);

// 0x7ffe97b0e8e0 constructor, sizeof(T): 24
// 0x7ffe97b0e8e0 A [1]: 0x55c0793e8580
// 0x7ffe97b0e8e0 D [1]: 0x55c0793e8580
```

正如我们所看到的，分配器只进行了一次分配和释放。尽管提供的类型是 4 字节的 int，但分配器分配了 24 字节的内存。这是因为`std::list`分配了链表节点，这种情况下是 24 字节。分配器位于`0x7ffe97b0e8e0`，分配位于`0x55c0793e8580`。此外，如所示，每次调用分配函数时分配的元素数量为 1。这是因为`std::list`实现了一个链表，对于添加到列表中的每个元素都进行了动态分配。尽管在使用自定义分配器时这似乎非常浪费，但在进行系统编程时，这可能非常有用，因为有时候一次只分配一个元素（而不是多个）时更容易处理内存。

现在让我们来看一下`std::vector`，如下所示：

```cpp
std::vector<int, myallocator<int>> myvector;
myvector.emplace_back(42);
myvector.emplace_back(42);
myvector.emplace_back(42);

// 0x7ffe1db8e2d0 constructor, sizeof(T): 4
// 0x7ffe1db8e2d0 A [1]: 0x55bf9dbdd550
// 0x7ffe1db8e2d0 A [2]: 0x55bf9dbebe90
// 0x7ffe1db8e2d0 D [1]: 0x55bf9dbdd550
// 0x7ffe1db8e2d0 A [4]: 0x55bf9dbdd550
// 0x7ffe1db8e2d0 D [2]: 0x55bf9dbebe90
// 0x7ffe1db8e2d0 D [4]: 0x55bf9dbdd550
```

在前面的例子中，我们使用我们的客户分配器创建了`std::vector`，然后，与之前的例子不同，我们向向量中添加了三个整数，而不是一个。这是因为`std::vector`必须维护连续的内存，而不管向量中的元素数量如何（这是`std::vector`的主要属性之一）。因此，如果`std::vector`填满（即，内存用完了），`std::vector`必须为`std::vector`中的所有元素分配一个全新的连续内存块，将`std::vector`从旧内存复制到新内存，然后释放先前的内存块，因为它不再足够大。

为了演示这是如何工作的，我们向`std::vector`添加了三个元素：

+   第一个元素分配了一个四个字节大小的内存块（`n == 1`和`sizeof(T) == 4`）。

+   第二次向`std::vector`添加数据时，当前的内存块已满（因为第一次只分配了四个字节），所以`std::vector`必须释放先前分配的内存，分配一个新的内存块，然后复制`std::vector`的旧内容。然而，这一次分配设置了`n == 2`，所以分配了八个字节。

+   第三次添加元素时，`std::vector`再次用完内存，重复这个过程，但是`n == 4`，这意味着分配了 16 个字节。

顺便说一句，第一次分配从`0x55bf9dbdd550`开始，这也恰好是第三次分配的位置。这是因为`malloc()`分配的内存是按 16 字节对齐的，这意味着第一次分配，虽然只有 4 个字节，实际上分配了 16 个字节，这对于第一次就足够了（也就是说，由 GCC 提供的`std::vector`的实现可以使用优化）。由于第一次分配在第二次向`std::vector`添加内存时被释放，所以这块内存可以在第三次使用元素时被释放，因为原始分配仍然足够请求的数量。

显然，看到分配器的使用方式，除非你真的需要连续的内存，否则`std::vector`不是存储列表的好选择，因为它很慢。然而，`std::list`占用了大量额外的内存，因为每个元素是 24 个字节，而不是 4 个字节。接下来要观察的下一个和最后一个容器是`std::deque`，它在`std::vector`和`std::list`之间找到了一个合适的平衡点：

```cpp
std::deque<int, myallocator<int>> mydeque;
mydeque.emplace_back(42);
mydeque.emplace_back(42);
mydeque.emplace_back(42);

// constructor, sizeof(T): 4
// 0x7ffdea986e67 A [8]: 0x55d6822b0da0
// 0x7ffdea986f30 A [128]: 0x55d6822afaf0
// 0x7ffdea986f30 D [128]: 0x55d6822afaf0
// 0x7ffdea986e67 D [8]: 0x55d6822b0da0
```

`std::deque`创建了一个内存块的链表，可以用来存储多个元素。换句话说，`std::deque`是`std::vectors`的`std::list`。像`std::list`一样，内存不是连续的，但像`std::vector`一样，每个元素只占用四个字节，并且不需要为每个添加的元素进行动态内存分配。如所示，`sizeof(T) == 4`字节，在创建`std::deque`时，分配了一个大的内存缓冲区来存储多个元素（具体来说是`128`个元素）。第二个较小的分配用于内部记录。

为了进一步探索`std::deque`，让我们向`std::deque`添加大量元素：

```cpp
std::deque<int, myallocator<int>> mydeque;

for (auto i = 0; i < 127; i++)
    mydeque.emplace_back(42);

for (auto i = 0; i < 127; i++)
    mydeque.emplace_back(42);

for (auto i = 0; i < 127; i++)
    mydeque.emplace_back(42);

// constructor, sizeof(T): 4
// 0x7ffc5926b1b7 A [8]: 0x560285cc0da0
// 0x7ffc5926b280 A [128]: 0x560285cbfaf0
// 0x7ffc5926b280 A [128]: 0x560285cc1660
// 0x7ffc5926b280 A [128]: 0x560285cc1bc0
// 0x7ffc5926b280 D [128]: 0x560285cbfaf0
// 0x7ffc5926b280 D [128]: 0x560285cc1660
// 0x7ffc5926b280 D [128]: 0x560285cc1bc0
// 0x7ffc5926b1b7 D [8]: 0x560285cc0da0
```

在上面的例子中，我们三次添加了`127`个元素。这是因为每次分配都足够存储`128`个元素，其中一个元素用于记录。如所示，`std::deque`分配了三个内存块。

# 复制相等的分配器

具有相等分配器的容器的复制是直接的，因为分配器是可互换的。为了探索这一点，让我们在先前的分配器中添加以下重载，以便我们可以观察到额外的操作：

```cpp
myallocator(myallocator &&other) noexcept
{
    (void) other;
    std::cout << this << " move constructor, sizeof(T): "
                << sizeof(T) << '\n';
}

myallocator &operator=(myallocator &&other) noexcept
{
    (void) other;
    std::cout << this << " move assignment, sizeof(T): "
                << sizeof(T) << '\n';
    return *this;
}

myallocator(const myallocator &other) noexcept
{
    (void) other;
    std::cout << this << " copy constructor, sizeof(T): "
                << sizeof(T) << '\n';
}

myallocator &operator=(const myallocator &other) noexcept
{
    (void) other;
    std::cout << this << " copy assignment, sizeof(T): "
                << sizeof(T) << '\n';
    return *this;
}
```

前面的代码添加了一个复制构造函数、`复制赋值`运算符、移动构造函数和一个`移动赋值`运算符，所有这些都有调试语句，以便我们可以看到容器在做什么。通过前面的添加，我们将能够看到分配器的复制是何时进行的。现在让我们在一个被复制的容器中使用这个分配器：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = mylist1;
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

在上面的例子中，我们创建了两个列表。在第一个`std::list`中，我们向列表添加了两个元素，然后将列表复制到第二个`std::list`。最后，我们向第二个`std::list`添加了两个元素。输出如下：

```cpp
0x7fff866d1e50 constructor, sizeof(T): 24
0x7fff866d1e70 constructor, sizeof(T): 24
0x7fff866d1e50 A [1]: 0x557c430ec550
0x7fff866d1e50 A [1]: 0x557c430fae90
----------------------------------------
0x7fff866d1d40 copy constructor, sizeof(T): 24
0x7fff866d1d40 A [1]: 0x557c430e39a0
0x7fff866d1d40 A [1]: 0x557c430f14a0
----------------------------------------
0x7fff866d1e70 A [1]: 0x557c430f3b30
0x7fff866d1e70 A [1]: 0x557c430ec4d0
0x7fff866d1e70 D [1]: 0x557c430e39a0
0x7fff866d1e70 D [1]: 0x557c430f14a0
0x7fff866d1e70 D [1]: 0x557c430f3b30
0x7fff866d1e70 D [1]: 0x557c430ec4d0
0x7fff866d1e50 D [1]: 0x557c430ec550
0x7fff866d1e50 D [1]: 0x557c430fae90
```

正如预期的那样，每个列表都创建了它打算使用的分配器，分配器创建了 24 字节的`std::list`节点。然后我们看到第一个分配器为添加到第一个列表中的两个元素分配内存。第二个列表在复制第一个列表之前仍然是空的，因此第二个容器创建了第三个临时分配器，它可以专门用于复制列表。完成这些操作后，我们将最后两个元素添加到第二个列表，我们可以看到第二个列表使用其原始分配器执行分配。

`std::list`可以自由地从一个分配器分配内存，然后从另一个分配器释放内存，这在释放内存时可以看到，这就是为什么`std::list`在复制期间创建临时分配器的原因。容器是否应该创建临时分配器并不是重点（尽管这可能是一个值得讨论的优化）。

# 移动相等的分配器

移动容器与复制容器类似，如果分配器相等。这是因为容器没有规则要做什么，因为容器可以使用其原始分配器来处理任何内存，如果需要，它可以创建一个新的分配器，如下所示：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = std::move(mylist1);
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

在前面的例子中，我们不是复制第一个容器，而是移动它。因此，移动后的第一个容器不再有效，第二个容器现在拥有来自第一个容器的内存。

这个例子的输出如下：

```cpp
0x7ffe582e2850 constructor, sizeof(T): 24
0x7ffe582e2870 constructor, sizeof(T): 24
0x7ffe582e2850 A [1]: 0x56229562d550
0x7ffe582e2850 A [1]: 0x56229563be90
----------------------------------------
----------------------------------------
0x7ffe582e2870 A [1]: 0x5622956249a0
0x7ffe582e2870 A [1]: 0x5622956324a0
0x7ffe582e2870 D [1]: 0x56229562d550
0x7ffe582e2870 D [1]: 0x56229563be90
0x7ffe582e2870 D [1]: 0x5622956249a0
0x7ffe582e2870 D [1]: 0x5622956324a0
```

与复制示例类似，两个列表被创建，每个`std::list`创建一个管理 24 字节的`std::list`节点的分配器。两个元素被添加到第一个列表，然后第一个列表被移动到第二个列表。因此，属于第一个列表的内存现在由第二个容器拥有，并且不执行任何副本。第二个列表的第二个分配是由它自己的分配器执行的，所有的释放也是如此，因为可以使用第二个分配器来释放从第一个分配器分配的内存。

# 探索一些可选属性

C++分配器提供了一些额外的属性，这些属性超出了`is_always_equal`。具体来说，C++分配器的作者可以选择定义以下内容：

+   +   `propagate_on_container_copy_assignment`

+   `propagate_on_container_move_assignment`

+   `propagate_on_container_swap`

可选属性告诉容器在特定操作（即复制、移动和交换）期间应如何处理分配器。具体来说，当容器被复制、移动或交换时，分配器不会被触及，这可能导致低效。传播属性告诉容器将操作传播到分配器。例如，如果`propagate_on_container_copy_assignment`设置为`std::true_type`并且正在复制容器，则在通常情况下不会复制分配器时，也必须复制分配器。

为了更好地探索这些属性，让我们创建我们的第一个不相等的分配器（即，相同分配器的两个不同实例可能不相等）。正如所述，大多数不相等的分配器是有状态的。在这个例子中，我们将创建一个无状态的不相等分配器，以保持简单。本章的最后一个例子将创建一个不相等的、有状态的分配器。

要开始我们的示例，我们首先需要为我们的分配器类创建一个托管对象，如下所示：

```cpp
class myallocator_object
{
public:

    using size_type = std::size_t;

public:

    void *allocate(size_type size)
    {
        if (auto ptr = malloc(size)) {
            std::cout << this << " A " << ptr << '\n';
            return ptr;
        }

        throw std::bad_alloc();
    }

    void deallocate(void *ptr)
    {
        std::cout << this << " D " << ptr << '\n';
        free(ptr);
    }
};
```

不相等的分配器必须遵守以下属性：

+   所有分配器的副本必须相等。这意味着即使我们创建了一个不相等的分配器，分配器的副本仍必须相等。当使用重新绑定复制构造函数时，这会变得棘手，因为这个属性仍然成立（即使两个分配器可能不具有相同的类型，如果一个是另一个的副本，它们仍可能相等）。

+   所有相等的分配器必须能够释放彼此的内存。再次，当使用重新绑定复制构造函数时，这变得棘手。具体来说，这意味着管理`int`对象的分配器可能必须从管理`std::list`节点的分配器中释放内存。

为了支持这两条规则，大多数不相等的分配器最终都成为受控对象的包装器。也就是说，创建了一个可以分配和释放内存的对象，并且每个分配器都存储指向此对象的指针。在前面的示例中，`myallocator_object{}`是能够分配和释放内存的受控对象。要创建此对象，我们所做的就是将`malloc()`和`free()`从分配器本身移动到此`myallocator_object{}`中；代码是相同的。添加到`myallocator_object{}`的唯一附加逻辑是以下内容：

+   构造函数接受一个大小。这是因为我们无法将受控对象创建为模板类。具体来说，受控对象需要能够更改其管理的内存类型（根据所述规则）。不久将介绍此特定需求。

+   添加了一个`rebind()`函数，专门用于更改受控对象管理的内存大小。再次，这使我们能够更改`myallocator_object{}`执行的分配大小。

接下来，我们需要定义分配器本身，如下所示：

```cpp
template<typename T>
class myallocator
{
```

分配器的第一部分与其他分配器相同，需要使用为某个`T`类型分配内存的模板类：

```cpp
public:

    using value_type = T;
    using pointer = T *;
    using size_type = std::size_t;
    using is_always_equal = std::false_type;
```

我们分配器的下一部分定义了我们的类型别名和可选属性。如图所示，所有三个传播函数都未定义，这告诉使用此分配器的任何容器，当容器发生复制、移动或交换时，分配器也不会被复制、移动或交换（容器应继续使用在构造时给定的相同分配器）。

接下来的一组函数定义了我们的构造函数和运算符。让我们从默认构造函数开始：

```cpp
myallocator() :
    m_object{std::make_shared<myallocator_object>()}
{
    std::cout << this << " constructor, sizeof(T): "
                << sizeof(T) << '\n';
}
```

与所有构造函数和运算符一样，我们输出`stdout`一些调试信息，以便观察容器对分配器的操作。如图所示，默认构造函数分配`myallocator_object{}`并将其存储为`std::shared_ptr`。我们利用`std::shared_ptr`，因为每个分配器的副本都必须相等，因此每个副本必须共享相同的受控对象（以便可以从一个分配器分配的内存可以从副本中释放）。由于任何分配器都可能在任何时间被销毁，因此*拥有*受控对象，因此`std::shared_ptr`是更合适的智能指针。

接下来的两个函数是移动构造函数和赋值运算符：

```cpp
myallocator(myallocator &&other) noexcept :
    m_object{std::move(other.m_object)}
{
    std::cout << this << " move constructor, sizeof(T): "
                << sizeof(T) << '\n';
}

myallocator &operator=(myallocator &&other) noexcept
{
    std::cout << this << " move assignment, sizeof(T): "
                << sizeof(T) << '\n';

    m_object = std::move(other.m_object);
    return *this;
}
```

在这两种情况下，由于移动操作的结果，我们需要`std::move()`我们的受控对象。对于复制也是一样的：

```cpp
myallocator(const myallocator &other) noexcept :
    m_object{other.m_object}
{
    std::cout << this << " copy constructor, sizeof(T): "
                << sizeof(T) << '\n';
}

myallocator &operator=(const myallocator &other) noexcept
{
    std::cout << this << " copy assignment, sizeof(T): "
                << sizeof(T) << '\n';

    m_object = other.m_object;
    return *this;
}
```

如图所示，如果对分配器进行复制，我们也必须复制受控对象。因此，分配器的副本利用相同的受控对象，这意味着副本可以从原始对象中释放内存。

下一个函数是使不相等的分配器如此困难的原因：

```cpp
template <typename U>
myallocator(const myallocator<U> &other) noexcept :
    m_object{other.m_object}
{
    std::cout << this << " copy constructor (U), sizeof(T): "
                << sizeof(T) << '\n';
}
```

前面的函数是重新绑定复制构造函数。此构造函数的目的是创建不同类型的另一个分配器的副本。例如，`std::list`从`myallocator<int>{}`开始，但实际上需要的是`myallocator<std::list::node>{}`类型的分配器，而不是`myallocator<int>{}`。为了克服这一点，前面的函数允许容器执行以下操作：

```cpp
myallocator<int> alloc1;
myallocator<std::list::node> alloc2(alloc1);
```

在上面的例子中，`alloc2`是`alloc1`的副本，即使`alloc1`和`alloc2`的`T`类型不相同。问题是，一个`int`是四个字节，而在我们的例子中，`std::list::node`有 24 个字节，这意味着前面的函数不仅能够创建一个相等的不同类型的分配器的副本，还必须能够创建一个能够释放不同类型内存的副本（特别是在这种情况下，`alloc2`必须能够释放`int`，即使它管理`std::list::node`元素）。在我们的例子中，这不是问题，因为我们使用`malloc()`和`free()`，但正如我们将在最后的例子中展示的那样，一些有状态的分配器，比如内存池，不太符合这个要求。

`allocate`和`deallocate`函数定义如下：

```cpp
pointer allocate(size_type n)
{
    auto ptr = m_object->allocate(sizeof(T) * n);
    return static_cast<pointer>(ptr);
}

void deallocate(pointer p, size_type n)
{
    (void) n;
    return m_object->deallocate(p);
}
```

由于我们的托管对象只调用`malloc()`和`free()`，我们可以将对象的`allocate()`和`deallocate()`函数视为`malloc()`和`free()`，因此，实现很简单。

我们`allocator`类中的私有逻辑如下：

```cpp
std::shared_ptr<myallocator_object> m_object;

template <typename T1, typename T2>
friend bool operator==(const myallocator<T1> &lhs, const myallocator<T2> &rhs);

template <typename T1, typename T2>
friend bool operator!=(const myallocator<T1> &lhs, const myallocator<T2> &rhs);
```

如前所述，我们存储了一个指向托管对象的智能指针，这允许我们创建分配器的副本。我们还声明我们的平等函数是友元的，尽管我们将这些友元函数放在类的私有部分，但我们可以将它们放在任何地方，因为友元声明不受公共/受保护/私有声明的影响。

最后，平等函数如下：

```cpp
template <typename T1, typename T2>
bool operator==(const myallocator<T1> &lhs, const myallocator<T2> &rhs)
{ return lhs.m_object.get() == rhs.m_object.get(); }

template <typename T1, typename T2>
bool operator!=(const myallocator<T1> &lhs, const myallocator<T2> &rhs)
{ return lhs.m_object.get() != rhs.m_object.get(); }
```

我们的*equal*分配器示例只是对`operator==`返回 true，对`operator!=`返回 false，这表明分配器是相等的（除了使用`is_always_equal`）。在这个例子中，`is_always_equal`设置为`false`，在我们的相等运算符中，我们比较了托管对象。每次创建一个新的分配器，都会创建一个新的托管对象，因此，分配器不相等（也就是说，它们是不相等的分配器）。问题是，我们不能简单地总是对`operator==`返回`false`，因为根据规范，分配器的副本必须始终等于原始分配器，这就是我们使用`std::shared_ptr`的原因。每个分配器的副本都创建了一个`std::shared_ptr`的副本，因此，如果复制了分配器，我们比较托管对象的地址，复制和原始对象有相同的托管对象，因此返回`true`（也就是说，它们是相等的）。虽然可能不使用`std::shared_ptr`，但大多数不相等的分配器都是这样实现的，因为它提供了一种简单的处理相等和不相等分配器之间差异的方法，根据分配器是否已被复制。

现在我们有了一个分配器，让我们来测试一下：

```cpp
std::list<int, myallocator<int>> mylist;
mylist.emplace_back(42);

// 0x7ffce60fbd10 constructor, sizeof(T): 24
// 0x561feb431590 A [1]: 0x561feb43fec0
// 0x561feb431590 D [1]: 0x561feb43fec0
```

如您所见，我们的分配器能够分配和释放内存。在上面的例子中，分配器位于`0x561feb431590`，而由`std::list`容器分配的元素位于`0x561feb43fec0`。

复制一个具有传播属性设置为`false`的不相等容器很简单，如下所示：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

mylist2.emplace_back(42);
mylist2.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = mylist1;
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

如前面的例子所示，我们创建了两个列表，并将两个列表都填充了两个元素。一旦列表填充完毕，我们就将第一个容器复制到第二个容器中，并输出到`stdout`，以便我们可以看到容器如何处理这个复制。最后，我们向刚刚复制的容器添加了两个元素。

这个例子的输出如下：

```cpp
// 0x7ffd65a15cb0 constructor, sizeof(T): 24
// 0x7ffd65a15ce0 constructor, sizeof(T): 24
// 0x55c4867c3a80 A [1]: 0x55c4867b9210  <--- add to list #1
// 0x55c4867c3a80 A [1]: 0x55c4867baec0  <--- add to list #1
// 0x55c4867d23c0 A [1]: 0x55c4867c89c0  <--- add to list #2
// 0x55c4867d23c0 A [1]: 0x55c4867cb050  <--- add to list #2
// ----------------------------------------
// ----------------------------------------
// 0x55c4867d23c0 A [1]: 0x55c4867c39f0  <--- add to list #2 after copy
// 0x55c4867d23c0 A [1]: 0x55c4867c3a10  <--- add to list #2 after copy
// 0x55c4867d23c0 D [1]: 0x55c4867c89c0  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867cb050  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867c39f0  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867c3a10  <--- deallocate list #2
// 0x55c4867c3a80 D [1]: 0x55c4867b9210  <--- deallocate list #1
// 0x55c4867c3a80 D [1]: 0x55c4867baec0  <--- deallocate list #1
```

如图所示，复制容器不涉及分配器。当发生复制时，列表 2 保留它已经拥有的两个分配，覆盖前两个元素的值。由于传播属性为`false`，第二个容器保留了它最初给定的分配器，并在复制后使用分配器来分配另外两个元素，但在列表失去作用域时也释放了之前分配的所有元素。

这种方法的问题在于容器需要循环遍历每个元素并执行手动复制。对于整数来说，这种类型的复制是可以的，但是我们可能已经在列表中存储了大型结构，因此复制容器将导致复制容器中的每个元素，这是浪费和昂贵的。由于传播属性为`false`，容器没有选择，因为它不能使用第一个列表的分配器，也不能使用自己的分配器来复制在第一个列表中分配的元素（因为分配器不相等）。尽管这是浪费的，但如将会展示的，这种方法可能仍然是最快的方法。

移动列表存在类似的问题：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

mylist2.emplace_back(42);
mylist2.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = std::move(mylist1);
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

在前面的示例中，我们做了与之前示例中相同的事情。我们创建了两个列表，并在将一个列表移动到另一个列表之前向每个列表添加了两个元素。

这个示例的结果如下：

```cpp
// 0x7ffd65a15cb0 constructor, sizeof(T): 24
// 0x7ffd65a15ce0 constructor, sizeof(T): 24
// 0x55c4867c3a80 A [1]: 0x55c4867c3a10  <--- add to list #1
// 0x55c4867c3a80 A [1]: 0x55c4867c39f0  <--- add to list #1
// 0x55c4867d23c0 A [1]: 0x55c4867c0170  <--- add to list #2
// 0x55c4867d23c0 A [1]: 0x55c4867c0190  <--- add to list #2
// ----------------------------------------
// ----------------------------------------
// 0x55c4867d23c0 A [1]: 0x55c4867b9c90  <--- add to list #2 after move
// 0x55c4867d23c0 A [1]: 0x55c4867b9cb0  <--- add to list #2 after move
// 0x55c4867d23c0 D [1]: 0x55c4867c0170  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867c0190  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867b9c90  <--- deallocate list #2
// 0x55c4867d23c0 D [1]: 0x55c4867b9cb0  <--- deallocate list #2
// 0x55c4867c3a80 D [1]: 0x55c4867c3a10  <--- deallocate list #1
// 0x55c4867c3a80 D [1]: 0x55c4867c39f0  <--- deallocate list #1
```

在前面的示例中，我们可以看到相同的低效性。由于传播属性为`false`，容器不能使用第一个列表的分配器，而必须继续使用它已经拥有的分配器。因此，移动操作不能简单地将内部容器从一个列表移动到另一个列表，而必须循环遍历整个容器，在每个单独的元素上执行`std::move()`，以便与列表中的每个节点相关联的内存仍然由第二个列表的原始分配器管理。

为了克服这些问题，我们将向我们的分配器添加以下内容：

```cpp
using propagate_on_container_copy_assignment = std::true_type;
using propagate_on_container_move_assignment = std::true_type;
using propagate_on_container_swap = std::true_type;
```

这些属性告诉使用这个分配器的任何容器，如果容器发生复制、移动或交换，分配器也应该执行相同的操作。例如，如果我们复制`std::list`，容器不仅必须*复制*元素，还必须复制分配器。

让我们看一下以下复制示例：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

mylist2.emplace_back(42);
mylist2.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = mylist1;
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

这个复制示例与我们之前的复制示例相同。我们创建两个列表，并向每个列表添加两个元素。然后我们将第一个列表复制到第二个列表，然后在完成之前向第二个列表添加两个额外的元素（最终将释放列表）。

这个示例的结果如下。应该注意，这个输出有点复杂，所以我们将一步一步地进行：

```cpp
// 0x7ffc766ec580 constructor, sizeof(T): 24
// 0x7ffc766ec5b0 constructor, sizeof(T): 24
// 0x5638419d9720 A [1]: 0x5638419d0b60  <--- add to list #1
// 0x5638419d9720 A [1]: 0x5638419de660  <--- add to list #1
// 0x5638419e8060 A [1]: 0x5638419e0cf0  <--- add to list #2
// 0x5638419e8060 A [1]: 0x5638419d9690  <--- add to list #2
```

在前面的输出中，两个列表都被创建，并且向每个容器添加了两个元素。接下来，输出将展示当我们将第二个容器复制到第一个容器时会发生什么：

```cpp
// 0x5638419e8060 D [1]: 0x5638419e0cf0
// 0x5638419e8060 D [1]: 0x5638419d9690
// 0x7ffc766ec5b0 copy assignment, sizeof(T): 24
// 0x7ffc766ec450 copy constructor (U), sizeof(T): 4
// 0x7ffc766ec3f0 copy constructor (U), sizeof(T): 24
// 0x7ffc766ec460 copy constructor, sizeof(T): 24
// 0x5638419d9720 A [1]: 0x5638419e8050
// 0x5638419d9720 A [1]: 0x5638419d9690
```

由于我们将传播属性设置为`false`，容器现在可以选择保留第一个容器使用的内存（例如，实现写时复制）。这是因为容器应该创建分配器的副本，任何两个分配器的副本都是相等的（即，它们可以释放彼此的内存）。glibc 的这种实现并不这样做。相反，它试图创建一个干净的内存视图。两个列表的分配器不相等，这意味着一旦复制发生，容器将不再能够释放自己先前分配的内存（因为它可能不再能够访问其原始分配器）。因此，容器首先删除它先前分配的所有内存。然后，它使用第一个列表分配器的重新绑定副本创建一个临时分配器（这似乎是未使用的），然后创建第一个列表分配器的直接副本，并使用它来为将要复制的元素分配新的内存。

最后，现在复制完成，最后两个元素可以添加到第二个列表中，每个列表在失去作用域时都可以被销毁：

```cpp
// 0x5638419d9720 A [1]: 0x5638419d96b0  <--- add to list #2 after copy
// 0x5638419d9720 A [1]: 0x5638419d5e10  <--- add to list #2 after copy
// 0x5638419d9720 D [1]: 0x5638419e8050  <--- deallocate list #2
// 0x5638419d9720 D [1]: 0x5638419d9690  <--- deallocate list #2
// 0x5638419d9720 D [1]: 0x5638419d96b0  <--- deallocate list #2
// 0x5638419d9720 D [1]: 0x5638419d5e10  <--- deallocate list #2
// 0x5638419d9720 D [1]: 0x5638419d0b60  <--- deallocate list #1
// 0x5638419d9720 D [1]: 0x5638419de660  <--- deallocate list #1
```

正如所示，由于分配器被传播，因此相同的分配器用于从两个列表中释放元素。这是因为一旦复制完成，两个列表现在都使用相同的分配器（因为任何两个分配器的副本必须相等，我们选择实现的方式是在发生复制时创建相同基本分配器对象的副本）。还应该注意，glibc 实现没有选择实现写时复制方案，这意味着实现不仅未能利用传播属性提供的可能优化，而且复制的实现实际上更慢，因为复制不仅必须逐个元素复制，还必须为复制分配新的内存。

现在让我们看一个移动示例：

```cpp
std::list<int, myallocator<int>> mylist1;
std::list<int, myallocator<int>> mylist2;

mylist1.emplace_back(42);
mylist1.emplace_back(42);

mylist2.emplace_back(42);
mylist2.emplace_back(42);

std::cout << "----------------------------------------\n";
mylist2 = std::move(mylist1);
std::cout << "----------------------------------------\n";

mylist2.emplace_back(42);
mylist2.emplace_back(42);
```

就像我们之前的移动示例一样，这创建了两个列表，并在将第一个列表移动到第二个列表之前向每个列表添加了两个元素。最后，我们的示例在第二个列表（现在是第一个列表）中添加了两个元素，然后在失去作用域时完成并释放了两个列表。

这个示例的输出结果如下：

```cpp
// 0x7ffc766ec580 constructor, sizeof(T): 24
// 0x7ffc766ec5b0 constructor, sizeof(T): 24
// 0x5638419d9720 A [1]: 0x5638419d96b0  <--- add to list #1
// 0x5638419d9720 A [1]: 0x5638419d9690  <--- add to list #1
// 0x5638419d5e20 A [1]: 0x5638419e8050  <--- add to list #2
// 0x5638419d5e20 A [1]: 0x5638419d5e30  <--- add to list #2
// ----------------------------------------
// 0x5638419d5e20 D [1]: 0x5638419e8050  <--- deallocate list #2
// 0x5638419d5e20 D [1]: 0x5638419d5e30  <--- deallocate list #2
// 0x7ffc766ec5b0 move assignment, sizeof(T): 24
// ----------------------------------------
// 0x5638419d9720 A [1]: 0x5638419d5e10
// 0x5638419d9720 A [1]: 0x5638419e8050
// 0x5638419d9720 D [1]: 0x5638419d96b0  <--- deallocate list #1
// 0x5638419d9720 D [1]: 0x5638419d9690  <--- deallocate list #1
// 0x5638419d9720 D [1]: 0x5638419d5e10  <--- deallocate list #2
// 0x5638419d9720 D [1]: 0x5638419e8050  <--- deallocate list #2
```

就像之前的示例一样，你可以看到列表被创建，并且第一个元素被添加到每个列表中。一旦移动发生，第二个列表将删除与其先前添加的元素相关联的内存。这是因为一旦移动发生，与第二个列表相关联的内存就不再需要了（因为它将被第一个列表分配的内存替换）。这是可能的，因为第一个列表的分配器将被移动到第二个列表（因为传播属性被设置为`true`），因此第二个列表现在将拥有第一个列表的所有内存。

最后，两个元素被添加到列表中，列表失去作用域并释放所有内存。正如所示，这是最优化的实现。不需要额外的内存分配，也不需要逐个元素的移动。移动操作只是将内存和分配器从一个容器移动到另一个容器。此外，由于没有复制分配器，这对于任何分配器来说都是一个简单的操作，因此，这个属性应该始终设置为 true。

# 可选函数

除了属性之外，还有几个可选的函数，可以为容器提供有关所提供的分配器类型的附加信息。一个可选的函数如下：

```cpp
size_type myallocator::max_size();
```

`max_size()` 函数告诉容器分配器可以分配的最大大小“n”。在 C++17 中，此函数已被弃用。`max_size()` 函数返回分配器可以执行的最大可能分配。耐人寻味的是，在 C++17 中，这默认为 `std::numeric_limits<size_type>::max() / sizeof(value_type)`，在大多数情况下可能不是一个有效的答案，因为大多数系统根本没有这么多可用的 RAM，这表明这个函数在实践中提供的价值很小。相反，就像 C++中的其他分配方案一样，如果分配失败，将抛出`std::bad_alloc`，表示容器尝试执行的分配是不可能的。

C++中的另一组可选函数如下：

```cpp
template<typename T, typename... Args>
static void myallocator::construct(T* ptr, Args&&... args);

template<typename T>
static void myallocator::destroy(T* ptr);
```

就像`max_size()`函数一样，构造和析构函数在 C++17 中已被弃用。在 C++17 之前，这些函数可以用于构造和析构与`ptr`提供的对象相关联的对象。应该注意的是，这就是为什么在构造函数中分配内存时我们不使用 new 和 delete，而是使用`malloc()`和`free()`。如果我们使用`new()`和`delete()`，我们会意外地调用对象的构造函数和/或析构函数两次，这将导致未定义的行为。

# 研究一个无状态、缓存对齐的分配器的示例

在这个例子中，我们将创建一个无状态的、相等的分配器，旨在分配对齐缓存的内存。这个分配器的目标是展示一个可以利用的 C++17 分配器，以增加容器存储的对象（例如链表）的效率，因为缓存抖动不太可能发生。

首先，我们将定义分配器如下：

```cpp
template<typename T, std::size_t Alignment = 0x40>
class myallocator
{
public:

    using value_type = T;
    using pointer = T *;
    using size_type = std::size_t;
    using is_always_equal = std::true_type;

    template<typename U> struct rebind {
        using other = myallocator<U, Alignment>;
    };

public:

    myallocator()
    { }

    template <typename U>
    myallocator(const myallocator<U, Alignment> &other) noexcept
    { (void) other; }

    pointer allocate(size_type n)
    {
        if (auto ptr = aligned_alloc(Alignment, sizeof(T) * n)) {
            return static_cast<pointer>(ptr);
        }

        throw std::bad_alloc();
    }

    void deallocate(pointer p, size_type n)
    {
        (void) n;
        free(p);
    }
};
```

前面的分配器类似于本章中创建的其他相等分配器。有一些显著的不同之处：

+   分配器的模板签名不同。我们不仅定义了分配器类型`T`，还添加了一个`Alignment`参数，并将默认值设置为`0x40`（即，分配将是 64 字节对齐的，这是 Intel CPU 上典型的缓存行大小）。

+   我们还提供了自己的重新绑定结构。通常，这个结构是为我们提供的，但由于我们的分配器有多个模板参数，我们必须提供我们自己版本的重新绑定结构。这个结构被容器使用，比如`std::list`，来创建容器需要的任何分配器，而不必创建一个副本（相反，它可以在初始化期间直接创建一个分配器）。在我们的这个重新绑定结构版本中，我们传递了原始分配器提供的`Alignment`参数。

+   重新绑定复制构造函数还必须定义`Alignment`变量。在这种情况下，如果要进行重新绑定，我们强制`Alignment`保持相同，这将是情况，因为重新绑定结构提供了`Alignment`（也是相同的）。

为了测试我们的例子，让我们创建分配器并输出一个分配的地址，以确保内存对齐：

```cpp
myallocator<int> myalloc;

auto ptr = myalloc.allocate(1);
std::cout << ptr << '\n';
myalloc.deallocate(ptr, 1);

// 0x561d512b6500
```

如图所示，分配的内存至少是 64 字节对齐的。多次分配也是如此：

```cpp
myallocator<int> myalloc;

auto ptr = myalloc.allocate(42);
std::cout << ptr << '\n';
myalloc.deallocate(ptr, 42);

// 0x55dcdcb41500
```

如图所示，分配的内存也至少是 64 字节对齐的。我们还可以将这个分配器与一个容器一起使用：

```cpp
std::vector<int, myallocator<int>> myvector;
myvector.emplace_back(42);

std::cout << myvector.data() << '\n';

// 0x55f875a0f500
```

而且，内存仍然是正确对齐的。

# 编译和测试

要编译这段代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter09/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter09/CMakeLists.txt)。

有了这段代码，我们可以使用以下方法编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter09/
> mkdir build
> cd build

> cmake ..
> make
```

要执行这个例子，运行以下命令：

```cpp
> ./example6
```

输出应该类似于以下内容：

```cpp
0x55aec04dbd00
0x55aec04e8f40
0x55aec04d5d00
===============================================================================
test cases: 3 | 3 passed
assertions: - none -
```

如前面的片段所示，我们能够分配不同类型的内存，以及释放这些内存，所有的地址都是 64 字节对齐的。

# 研究一个有状态的、内存池分配器的例子

在这个例子中，我们将创建一个更复杂的分配器，称为**内存池分配器**。内存池分配器的目标是快速为固定大小的类型分配内存，同时（更重要的是）减少内存的内部碎片（即，每个分配浪费的内存量，即使分配大小不是二的倍数或其他优化的分配大小）。

内存池分配器是如此有用，以至于一些 C++的实现已经包含了内存池分配器。此外，C++17 在技术上支持一种称为**多态分配器**的内存池分配器（本书未涵盖，因为在撰写时，没有主要的 C++17 实现支持多态分配器），大多数操作系统在内核中利用内存池分配器来减少内部碎片。

内存池分配器的主要优点如下：

+   使用`malloc()`是慢的。有时`free()`也很慢，但对于一些实现，`free()`就像翻转一个位一样简单，这样它可以实现非常快的速度。

+   大多数池分配器利用 deque 结构，这意味着池分配器分配了一个大的内存*块*，然后将这个内存分割为分配。每个内存*块*都使用链表链接，以便根据需要向池中添加更多内存。

池分配器还具有一个有趣的特性，即块大小越大，内部碎片的减少就越大。这种优化的代价是，如果池没有完全利用，那么随着块大小的增加，浪费的内存量也会增加，因此池分配器应该根据应用程序的需求进行定制。

为了开始我们的示例，我们首先创建一个管理*块*列表并从*块*中分配内存的`pool`类。*块*列表将存储在一个永远增长的堆栈中（也就是说，在这个示例中，我们将尝试对*块*中的内存进行碎片整理，或者如果*块*中的所有内存都已被释放，则从堆栈中移除*块*）。每次我们向池中添加一个内存块时，我们将将内存块分割为`sizeof(T)`大小的块，并将每个块的地址添加到称为地址堆栈的第二个堆栈中。当分配内存时，我们将从地址堆栈中弹出一个地址，当释放内存时，我们将地址推回堆栈。

我们池的开始如下：

```cpp
class pool
{
public:

    using size_type = std::size_t;

public:

    explicit pool(size_type size) :
        m_size{size}
    { }
```

池将充当我们不均匀分配器的托管对象，就像我们以前的不均匀分配器示例一样。因此，池不是一个模板类，因为如果使用重新绑定复制构造函数，我们将需要更改池的大小（更多关于这个特定主题的内容即将到来）。如图所示，在我们的构造函数中，我们存储了池的大小，但我们并没有尝试预加载池。

要分配，我们从地址堆栈中弹出一个地址并返回它。如果地址堆栈为空，我们通过分配另一个内存块并将其添加到块堆栈中，将内存分割成块，并将分割的块添加到地址堆栈中，如下所示：

```cpp
    void *allocate()
    {
        if (m_addrs.empty()) 
        {
            this->add_addrs();
        }

        auto ptr = m_addrs.top();
        m_addrs.pop();

        return ptr;
    }
```

为了释放内存，我们将提供的地址推送到地址堆栈中，以便以后可以重新分配。使用这种方法，为容器分配和释放内存就像从单个堆栈中弹出和推送地址一样简单：

```cpp
    void deallocate(void *ptr)
    { 
        m_addrs.push(ptr); 
    }
```

如果使用重新绑定复制构造函数，则需要更改池的大小。这种类型的复制只有在尝试将`int`类型的分配器创建为`std::list::node`类型的分配器时才会发生，这意味着要复制的分配器尚未被使用，这意味着可以调整大小。如果分配器已经被使用，这意味着分配器已经分配了不同大小的内存，因此在这种实现中重新绑定是不可能的。考虑以下代码：

```cpp
    void rebind(size_type size)
    {
        if (!m_addrs.empty() || !m_blocks.empty()) 
        {
            std::cerr << "rebind after alloc unsupported\n";
            abort();
        }

        m_size = size;
    }
```

应该指出，还有其他处理这个特定问题的方法。例如，可以创建一个不尝试使用重新绑定复制构造函数的`std::list`。还可以创建一个能够管理多个内存池的分配器，每个池都能够分配和释放特定类型的内存（当然，这将导致性能下降）。

在我们的私有部分，我们有`add_addrs()`函数，这个函数在`allocate`函数中看到过。`this`函数的目标是重新填充地址堆栈。为此，`this`函数分配另一个内存块，将内存分割，并将其添加到地址堆栈中：

```cpp
    void add_addrs()
    {
        constexpr const auto block_size = 0x1000;
        auto block = std::make_unique<uint8_t[]>(block_size);

        auto v = gsl::span<uint8_t>(
            block.get(), block_size
        );

        auto total_size =
            v.size() % m_size == 0 ? v.size() : v.size() - m_size;

        for (auto i = 0; i < total_size; i += m_size) 
        {
            m_addrs.push(&v.at(i));
        }

        m_blocks.push(std::move(block));
    }
```

最后，我们有私有成员变量，其中包括池的大小、地址堆栈和块堆栈。请注意，我们使用`std::stack`。`std::stack`使用`std::deque`来实现堆栈，尽管可以编写一个不使用迭代器的更有效的堆栈，但在测试中，`std::stack`的性能几乎一样好：

```cpp
    size_type m_size;

    std::stack<void *> m_addrs{};
    std::stack<std::unique_ptr<uint8_t[]>> m_blocks{};
```

分配器本身与我们已经定义的先前的不平等分配器几乎完全相同：

```cpp
template<typename T>
class myallocator
{
public:

    using value_type = T;
    using pointer = T *;
    using size_type = std::size_t;
    using is_always_equal = std::false_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
```

一个区别是我们将`propagate_on_container_copy_assignment`定义为`false`，特意防止分配器尽可能少地被复制。这个选择也得到了支持，因为我们已经确定 glibc 在使用不平等分配器时并不会提供很大的好处。

构造函数与先前定义的相同：

```cpp
    myallocator() :
        m_pool{std::make_shared<pool>(sizeof(T))}
    {
        std::cout << this << " constructor, sizeof(T): "
                  << sizeof(T) << '\n';
    }

    template <typename U>
    myallocator(const myallocator<U> &other) noexcept :
        m_pool{other.m_pool}
    {
        std::cout << this << " copy constructor (U), sizeof(T): "
                  << sizeof(T) << '\n';

        m_pool->rebind(sizeof(T));
    }

    myallocator(myallocator &&other) noexcept :
        m_pool{std::move(other.m_pool)}
    {
        std::cout << this << " move constructor, sizeof(T): "
                  << sizeof(T) << '\n';
    }

    myallocator &operator=(myallocator &&other) noexcept
    {
        std::cout << this << " move assignment, sizeof(T): "
                  << sizeof(T) << '\n';

        m_pool = std::move(other.m_pool);
        return *this;
    }

    myallocator(const myallocator &other) noexcept :
        m_pool{other.m_pool}
    {
        std::cout << this << " copy constructor, sizeof(T): "
                  << sizeof(T) << '\n';
    }

    myallocator &operator=(const myallocator &other) noexcept
    {
        std::cout << this << " copy assignment, sizeof(T): "
                  << sizeof(T) << '\n';

        m_pool = other.m_pool;
        return *this;
    }
```

`allocate`和`deallocate`函数与先前定义的相同，调用池的分配函数。一个区别是我们的池只能分配单个块的内存（也就是说，池分配器不能分配多个地址同时保持连续性）。因此，如果`n`不是`1`（也就是说，容器不是`std::list`或`std::map`），我们将退回到`malloc()`/`free()`实现，这通常是默认实现：

```cpp
    pointer allocate(size_type n)
    {
        if (n != 1) {
            return static_cast<pointer>(malloc(sizeof(T) * n));
        }

        return static_cast<pointer>(m_pool->allocate());
    }

    void deallocate(pointer ptr, size_type n)
    {
        if (n != 1) {
            free(ptr);
        }

        m_pool->deallocate(ptr);
    }
```

分配器的其余部分与先前定义的相同：

```cpp
private:

    std::shared_ptr<pool> m_pool;

    template <typename T1, typename T2>
    friend bool operator==(const myallocator<T1> &lhs, const myallocator<T2> &rhs);

    template <typename T1, typename T2>
    friend bool operator!=(const myallocator<T1> &lhs, const myallocator<T2> &rhs);

    template <typename U>
    friend class myallocator;
};

template <typename T1, typename T2>
bool operator==(const myallocator<T1> &lhs, const myallocator<T2> &rhs)
{ return lhs.m_pool.get() == rhs.m_pool.get(); }

template <typename T1, typename T2>
bool operator!=(const myallocator<T1> &lhs, const myallocator<T2> &rhs)
{ return lhs.m_pool.get() != rhs.m_pool.get(); }
```

最后，在测试我们的分配器之前，我们需要定义一个基准测试函数，能够给我们一个特定操作所需时间的指示。这个函数将在第十一章中更详细地定义，*Unix 中的时间接口*。目前，最重要的是要理解这个函数将一个回调函数作为输入（在我们的情况下是 Lambda），并返回一个数字。返回的数字越高，回调函数执行的时间越长：

```cpp
template<typename FUNC>
auto benchmark(FUNC func) {
    auto stime = std::chrono::high_resolution_clock::now();
    func();
    auto etime = std::chrono::high_resolution_clock::now();

    return (etime - stime).count();
}
```

我们将进行的第一个测试是创建两个列表，并向每个列表添加元素，同时计算添加所有元素到列表所需的时间。由于每次添加到列表都需要分配，执行此测试将使我们大致比较我们的分配器在分配内存方面与 glibc 提供的默认分配器相比有多好。

```cpp
constexpr const auto num = 100000;

std::list<int> mylist1;
std::list<int, myallocator<int>> mylist2;

auto time1 = benchmark([&]{
    for (auto i = 0; i < num; i++) {
        mylist1.emplace_back(42);
    }
});

auto time2 = benchmark([&]{
    for (auto i = 0; i < num; i++) {
        mylist2.emplace_back(42);
    }
});

std::cout << "[TEST] add many:\n";
std::cout << " - time1: " << time1 << '\n';
std::cout << " - time2: " << time2 << '\n';
```

如上所述，对于每个列表，我们向列表中添加`100000`个整数，并计算所需的时间，从而使我们能够比较分配器。结果如下：

```cpp
0x7ffca71d7a00 constructor, sizeof(T): 24
[TEST] add many:
  - time1: 3921793
  - time2: 1787499
```

如图所示，我们的分配器在分配内存方面比默认分配器快 219%。

在我们的下一个测试中，我们将比较我们的分配器与默认分配器在释放内存方面的表现。为了执行此测试，我们将做与之前相同的事情，但是不是计时我们的分配，而是计时从每个列表中删除元素所需的时间：

```cpp
constexpr const auto num = 100000;

std::list<int> mylist1;
std::list<int, myallocator<int>> mylist2;

for (auto i = 0; i < num; i++) {
    mylist1.emplace_back(42);
    mylist2.emplace_back(42);
}

auto time1 = benchmark([&]{
    for (auto i = 0; i < num; i++) {
        mylist1.pop_front();
    }
});

auto time2 = benchmark([&]{
    for (auto i = 0; i < num; i++) {
        mylist2.pop_front();
    }
});

std::cout << "[TEST] remove many:\n";
std::cout << " - time1: " << time1 << '\n';
std::cout << " - time2: " << time2 << '\n';
```

`this`函数的结果如下：

```cpp
0x7fff14709720 constructor, sizeof(T): 24
[TEST] remove many:
  - time1: 1046463
  - time2: 1285248
```

如图所示，我们的分配器只有默认分配器的 81%那么快。这可能是因为`free()`函数更有效率，这并不奇怪，因为理论上推送到堆栈可能比某些`free()`的实现更慢。即使我们的`free()`函数较慢，与分配和碎片化改进相比，差异微不足道。还要注意的是，这种实现的分配和释放速度几乎相同，这是我们所期望的。

为了确保我们正确编写了分配器，以下将再次运行我们的测试，但是不是计算向列表添加元素所需的时间，而是计算列表中每个值的总和。如果我们的总和符合预期，我们将知道分配和释放已正确执行：

```cpp
constexpr const auto num = 100000;

std::list<int, myallocator<int>> mylist;

for (auto i = 0; i < num; i++) {
    mylist.emplace_back(i);
}

uint64_t total1{};
uint64_t total2{};

for (auto i = 0; i < num; i++) {
    total1 += i;
    total2 += mylist.back();
    mylist.pop_back();
}

std::cout << "[TEST] verify: ";
if (total1 == total2) {
    std::cout << "success\n";
}
else {
    std::cout << "failure\n";
    std::cout << " - total1: " << total1 << '\n';
    std::cout << " - total2: " << total2 << '\n';
}
```

正如预期的那样，我们的测试输出是“成功”。

# 编译和测试

要编译这段代码，我们利用了与其他示例相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter09/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter09/CMakeLists.txt)。

有了这段代码，我们可以使用以下方式编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter09/
> mkdir build
> cd build

> cmake -DCMAKE_BUILD_TYPE=Release ..
> make
```

要执行示例，请运行以下命令：

```cpp
> ./example7
```

输出应该类似于以下内容：

```cpp
0x7ffca71d7a00 constructor, sizeof(T): 24
[TEST] add many:
  - time1: 3921793
  - time2: 1787499
0x7fff14709720 constructor, sizeof(T): 24
[TEST] remove many:
  - time1: 1046463
  - time2: 1285248
0x7fff5d8ad040 constructor, sizeof(T): 24
[TEST] verify: success
===============================================================================
test cases: 5 | 5 passed
assertions: - none -
```

正如你所看到的，我们的示例输出与我们之前提供的输出相匹配。需要注意的是，你的结果可能会根据硬件或已在系统上运行的内容等因素而有所不同。

# 总结

在本章中，我们看了如何创建自己的分配器，并涵盖了 C++分配器概念的复杂细节。主题包括相等和不相等分配器之间的区别，容器传播的处理方式，重新绑定以及有状态分配器可能出现的问题。最后，我们用两个不同的例子总结了。第一个例子演示了如何创建一个简单的、缓存对齐的无状态分配器，而第二个例子提供了一个有状态对象分配器的功能示例，该分配器维护一个用于快速分配的空闲池。

在下一章中，我们将使用几个示例来演示如何使用 C++编程 POSIX 套接字（即网络编程）。

# 问题

1.  `is_always_equal`是什么意思？

1.  什么决定了分配器是相等还是不相等？

1.  一个有状态的分配器可以是相等的吗？

1.  一个无状态的分配器可以是相等的吗？

1.  `propagate_on_container_copy_assignment`是做什么的？

1.  对于容器，rebind 复制构造函数的作用是什么？

1.  关于传递给 allocate 函数的`n`变量，`std::list`和`std::vector`有什么区别？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
