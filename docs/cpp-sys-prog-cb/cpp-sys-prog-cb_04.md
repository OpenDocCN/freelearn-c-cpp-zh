# 第四章：深入了解内存管理

内存在处理系统开发时是核心概念之一。分配、释放、学习内存管理方式，以及了解 C++可以提供什么来简化和管理内存，都是至关重要的。本章将通过学习如何使用 C++智能指针、对齐内存、内存映射 I/O 和分配器来帮助您理解内存的工作原理。

本章将涵盖以下主题：

+   学习自动与动态内存

+   学习何时使用`unique_ptr`，以及对大小的影响

+   学习何时使用`shared_ptr`，以及对大小的影响

+   分配对齐内存

+   检查分配的内存是否对齐

+   处理内存映射 I/O

+   亲自处理分配器

# 技术要求

为了让您立即尝试这些程序，我们设置了一个 Docker 镜像，其中包含本书中将需要的所有工具和库。这是基于 Ubuntu 19.04 的。

为了设置它，请按照以下步骤进行：

1.  从[www.docke](https://www.docker.com/)[r.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  通过运行以下命令从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  现在应该可以使用该镜像。键入以下命令查看镜像：`docker images`。

1.  现在您应该至少有这个镜像：`kasperondocker/system_programming_cookbook`。

1.  通过以下命令以交互式 shell 运行 Docker 镜像：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。键入`root@39a5a8934370/# cd /BOOK/`以获取按章节开发的所有程序。

需要`--cap-add sys_ptrace`参数以允许 Docker 容器中的 GNU Project Debugger（GDB）设置断点，默认情况下 Docker 不允许。

**免责声明**：C++20 标准已经在二月底的布拉格会议上由 WG21 批准（即技术上最终确定）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 的代码。GCC 将最新功能的开发保留在分支中（您必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励您自行尝试。因此，请克隆并探索 GCC 合同和模块分支，并尽情玩耍。

# 学习自动与动态内存

本教程将重点介绍 C++提供的两种主要策略来分配内存：**自动**和**动态**内存分配。当变量的作用域持续到其定义的块的持续时间时，变量是自动的，并且其分配和释放是自动的（即不由开发人员决定）。变量分配在堆栈上。

如果变量在内存的动态部分（自由存储区，通常称为*堆*）中分配，并且分配和释放由开发人员决定，则变量是动态的。动态内存分配提供的更大灵活性伴随着更多的工作量，以避免内存泄漏、悬空指针等。

# 如何做...

本节将展示自动和动态变量分配的两个示例。

1.  让我们创建一个我们需要的实用类：

```cpp
class User
{
public:
    User(){
        std::cout << "User constructor" << std::endl;
    };
    ~User(){
        std::cout << "User Destructor" << std::endl;
    };

    void cheers() 
    {
        std::cout << " hello!" << std::endl;};
    };
};
```

1.  现在，让我们创建`main`模块来显示自动内存使用情况：

```cpp
#include <iostream>

int main()
{
    std::cout << "Start ... " << std::endl;
    {
        User developer;
        developer.cheers();
    }
    std::cout << "End ... " << std::endl;
}
```

1.  现在，我们将为动态内存使用编写`main`模块：

```cpp
#include <iostream>

int main()
{
    std::cout << "Start ... " << std::endl;
    {
        User* developer = new User();
        developer->cheers();
        delete developer;
    }
    std::cout << "End ... " << std::endl;
}
```

这两个程序，尽管结果相同，但展示了处理内存的两种不同方式。

# 工作原理...

在第一步中，我们定义了一个`User`类，用于展示自动和动态内存分配之间的区别。它的构造函数和析构函数将用于显示类何时分配和释放。

在*步骤 2*中，我们可以看到变量只是定义为`User developer;`。C++运行时将负责在堆栈上分配内存并释放内存，而开发人员无需额外工作。这种类型的内存管理更快，更容易，但有两个主要成本：

+   内存量是有限的。

+   变量仅在内部`{ }`块中有效和可见，其中它被分配。

在*步骤 3*中，相同的对象分配在动态内存（即**堆**）上。主要区别在于现在开发人员负责分配和释放所需的内存量。如果内存没有被释放（使用`free`），就会发生泄漏。动态管理内存的优点如下：

+   灵活性：指针引用分配的内存（`developer`变量）可以在整个程序中使用。

+   可用的内存量远远超过自动内存管理的内存量。

# 还有更多...

使用更新的 C++标准（从版本 11 开始），可以安全地避免使用`new`和`delete`，而使用智能指针（`shared_ptr`和`unique_ptr`）。这两个工具将在不再使用内存时负责释放内存。第二章，*重温 C++*，提供了智能指针的复习。

# 另请参阅

接下来的两个配方将展示何时使用`unique_ptr`和`shared_ptr`。

# 学习何时使用`unique_ptr`，以及大小的影响

在上一个配方中，我们已经学习了 C++中管理内存的两种基本方式：自动和动态。我们还了解到，与自动内存（即从堆栈中可用）相比，动态内存对开发人员的数量更多，并提供了更大的灵活性。另一方面，处理动态内存可能是一种不愉快的体验：

+   指针不指示它指向数组还是单个对象。

+   释放分配的内存时，您不知道是否必须使用`delete`还是`delete[]`，因此您必须查看变量的定义方式。

+   没有明确的方法告诉指针是否悬空。

这些只是您在处理动态内存以及`new`和`delete`时可能遇到的一些问题。`unique_ptr`是一个智能指针，这意味着它知道何时应该释放内存，从而减轻了开发人员的负担。在本配方中，您将学习如何正确使用`unique_ptr`和`make_unique`。

# 如何做...

在本节中，我们将开发一个程序，以了解为什么`unique_ptr`是处理动态内存的便捷方式；第二个方面是了解`unique_ptr`是否与原始指针大小相同：

1.  我们将重用上一个配方中开发的`User`类。

1.  让我们编写`main`程序，使用`make_unique`分配`User`对象并使用`unique_ptr`：

```cpp
#include <iostream>

int main()
{
    std::cout << "Start ... " << std::endl;
    {
        auto developer = std::make_unique<User>();
        developer->cheers();
    }
    std::cout << "End ... " << std::endl;
}
```

1.  让我们看看内存的影响：

```cpp
auto developer = std::make_unique<User>();
developer->cheers();

User* developer2 = new User();
std::cout << "developer size = " << sizeof (developer) << std::endl;
std::cout << "developer2 size = " << sizeof (developer2) << std::endl;
delete developer2;
```

您认为`developer`和`developer2`之间的大小差异是多少？

# 它是如何工作的...

在*步骤 2*中，我们使用`unique_ptr`来定义使用`std::make_unique`分配的变量。一旦分配了变量，由于析构函数会自动为我们释放内存，因此不会有内存泄漏的风险。输出如下：

![](img/a9d63859-8852-406f-8c0e-3474395f5d97.png)

在*步骤 3*中，我们想要检查`unique_ptr`是否与原始指针相比增加了任何内存。好消息是，`unique_ptr`与原始指针版本的大小相同。此步骤的输出如下：

![](img/541b1ef5-331f-4f88-b69d-1e6006d78c37.png)

`developer`和`developer2`变量的大小相同，开发人员可以以相同的方式处理它们。

一个经验法则是仅对具有**独占所有权的资源**使用`unique_ptr`，这代表了大多数开发人员的用例。

# 还有更多...

默认情况下，`unique_ptr`调用对象的默认`delete`析构函数，但可以指定自定义的`delete`析构函数。如果指针变量不代表独占所有权，而是共享所有权，将其转换为`shared_ptr`很容易。

重要的一点要强调的是，`make_unique`不是 C++11 标准库的一部分，而是 C++14 库的一部分。如果你使用的是 C++11 标准库，它的实现是非常简单的。

# 另请参阅

第二章，*重温 C++*有一个专门讨论智能指针的配方，其中有一个关于共享和独特指针的配方。建议阅读的是 Scott Meyers 的*Effective Modern C++*。

# 学习何时使用 shared_ptr，以及大小的影响

在前面的配方中，我们已经学会了如何以一种非常方便的方式管理动态内存（在堆上分配），使用`unique_ptr`。我们也学到了`unique_ptr`必须在内存的独占所有权或由内存管理的资源的情况下使用。但是，如果我们有一个资源是由多个实体共同拥有的呢？如果我们必须在所有者完成工作后释放要管理的内存呢？好吧，这正是`shared_ptr`的用例。就像`unique_ptr`一样，对于`shared_ptr`，我们不必使用`new`来分配内存，但是有一个模板函数（C++标准库的一部分），`make_shared`。 

# 如何做到...

在本节中，我们将开发一个程序来展示如何使用`shared_ptr`。您将了解到只有在所有者不再使用内存时，内存才会被释放：

1.  我们将重用第一个配方中开发的`User`类。现在让我们编写`main`模块：

```cpp
int main()
{
    std::cout << "Start ... " << std::endl;
    auto shared1 = std::make_shared<User>();
    {
        auto shared2 = shared1;
        shared2->cheers(); std::cout << " from shared2"
            << std::endl;
        shared1->cheers(); std::cout << " from shared1"
            << std::endl;
    }
    std::cout << "End ... " << std::endl;
}
```

1.  现在，让我们通过编写这个程序来看一下`shared_ptr`使用的内存：

```cpp
int main()
{
    std::cout << "Start ... " << std::endl;
    auto shared1 = std::make_shared<User>();
   {
        auto shared2 = shared1;
        User* newAllocation = new User();
        auto uniqueAllocation = std::make_unique<User>();

        std::cout << "shared2 size = " << sizeof (shared2)
            << std::endl;
        std::cout << "newAllocation size = " <<
            sizeof (newAllocation) << std::endl;
        std::cout << "uniqueAllocation size = " <<
            sizeof (uniqueAllocation) << std::endl;

        delete newAllocation;
    }
    std::cout << "End ... " << std::endl;
}
```

在这一点上，我们应该知道`unique_ptr`的大小与原始指针相比（正如我们在*学习何时使用 unique_ptr 以及大小的影响*配方中所学到的）。`shared_ptr`变量的大小是多少？还是一样的？在下一节中，我们将了解这个重要的方面。

# 它是如何工作的...

在前面的第一个程序中，我们展示了如何使用`shared_ptr`。首先，我们分配了一个内存块，其中包含了一个类型为`User`的对象，`auto shared1 = std::make_shared<User>();`。到目前为止，`User`资源由`shared1`变量拥有。接下来，我们将`shared1`变量分配给`shared2`，通过`auto shared2 = shared1;`。这意味着包含`User`对象的内存现在由`shared1`和`shared2`指向。使用构造函数复制`auto shared2 (shared1);`也可以达到相同的目标。由于`User`现在由两个变量指向，所以使用的内存只有在所有变量超出范围时才会被释放。事实上，输出证明了内存在主块结束时被释放（`User`的析构函数被调用），而不是在内部块结束时，就像`unique_ptr`一样。

![](img/a75c73de-d7b4-4ff1-9412-af044422965d.png)

`shared_ptr`对内存的影响与`unique_ptr`不同。原因是`shared_ptr`的实现需要一个原始指针来跟踪内存（与`unique_ptr`一样），以及另一个原始指针用于资源的引用计数。

这个引用计数变量必须是原子的，因为它可以被不同的线程增加和减少：

![](img/2bbb9914-5677-489e-adc5-d9acef0e9550.png)

`shared_ptr`变量的内存大小通常是原始指针的两倍，正如在运行第二个程序时在前面的输出中所看到的。

# 还有更多...

另一个有趣的点不容忽视的是，由于`shared_ptr`包含原子变量，它通常比普通变量慢。

# 另请参阅

第二章，*重温 C++*，有一个专门介绍智能指针的示例，其中包括一个关于共享指针和唯一指针的示例。建议阅读 Scott Meyers 的*Effective Modern C++*。

# 分配对齐内存

编写系统程序可能需要使用在内存中对齐的数据，以便有效地访问硬件（在某些情况下，甚至是访问硬件）。例如，在 32 位架构机器上，我们将内存分配对齐到 4 字节边界。在这个示例中，您将学习如何使用 C++11 的`std::aligned_storage`来分配对齐内存。当然，还有其他更传统的机制来分配对齐内存，但本书的目标是尽可能使用 C++标准库工具。

# 如何做...

在本节中，我们将编写一个程序，该程序将使用使用`std::aligned_storage`分配的内存，并将展示`std::alignment_of`的使用：

1.  让我们从编写一个程序开始，检查当前计算机上整数和双精度浮点数的默认对齐边界是多少：

```cpp
#include <type_traits>
#include <iostream>
int main()
{
    std::cout << "int alignment = " << std::alignment_of<int>
        ::value << std::endl;
    std::cout << "double alignment = " << 
        std::alignment_of<double>::value << std::endl;
    return (0);
}
```

1.  现在，让我们编写一个程序来分配对齐到特定大小的内存。为此，让我们使用`std::aligned_storage`：

```cpp
#include <type_traits>
#include <iostream>
typedef std::aligned_storage<sizeof(int), 8>::type intAligned;
int main()
{
    intAligned i, j;
    new (&i) int();
    new (&j) int();

    int* iu = &reinterpret_cast<int&>(i);
    *iu = 12;
    int* ju = &reinterpret_cast<int&>(j);
    *ju = 13;

    std::cout << "alignment = " << std::alignment
        _of<intAligned>::value << std::endl;
    std::cout << "value = " << *iu << std::endl;
    std::cout << "value2 = " << reinterpret_cast<int&>(i)
        << std::endl;
    return (0);
}
```

分配对齐内存可能会很棘手，C++标准库（从第 11 版开始）提供了这两个功能（`std::alignment_of`，`std::aligned_storage`）来简化它。下一节将描述其背后的机制。

# 它是如何工作的...

第一个程序非常简单，通过`std::alignment_of`显示了两种原始类型在内存中的自然对齐。通过编译（`g++ alignedStorage.cpp`）并运行程序，我们得到以下输出：

![](img/cb15931f-34a9-47f0-8177-6b312346afba.png)

这意味着每个整数将在`4`字节的边界上对齐，并且浮点类型将在`8`字节处对齐。

在第二个程序中，我们需要一个对齐到`8`字节的整数。通过编译并运行可执行文件，输出将类似于这样：

![](img/9f9a06a2-f91c-41b7-a846-570c3b5837e4.png)

你可能已经注意到，我已经使用了`-g`选项进行了编译（添加调试符号）。我们这样做是为了在 GDB 中的内存转储中显示整数的内存正确地对齐在`8`字节处：

![](img/71381006-7525-4b1d-9919-2163adf644e0.png)

从调试会话中，我们可以看到通过`x/20bd iu`（`x`=*内存转储*）命令，我们在`iu`变量地址之后转储了`20`字节的内存。我们可以看到这里有一些有趣的东西：`iu`和`ju`变量都对齐在`8`字节处。每个内存行显示`8`字节（测试一下：`0x7ffc57654470`* - * `0x7ffc57654468` = `8`）。

# 还有更多...

玩弄内存总是有风险的，这些新的 C++特性（以及`std`命名空间中的其他可用特性）帮助我们**玩得更安全**。建议仍然是一样的：过早的优化必须谨慎使用；只有在必要时才进行优化（即使用对齐内存）。最后一个建议：不建议使用`reinterpret_cast`，因为它在低级别操纵内存。在使用它时，您需要知道自己在做什么。

# 另请参阅

Bjarne Stroustrup 的*The C++ Programming Language, Fourth Edition*的最新版本有一段关于*内存对齐*（*6.2.9*）和*aligned_storage*（*35.4.1*）的段落。

# 检查分配的内存是否对齐

在前一个示例中，您已经学会了如何使用 C++11 来分配对齐内存。现在的问题是：我们如何知道内存是否正确对齐？这个示例将教会您这一点。

# 如何做...

我们将使用前面的程序，并稍作修改，看看如何检查指针是否对齐：

1.  让我们修改前面的程序，如下所示：

```cpp
#include <type_traits>
#include <iostream>

using intAligned8 = std::aligned_storage<sizeof(int), 8>::type;
using intAligned4 = std::aligned_storage<sizeof(int), 4>::type;

int main()
{
    intAligned8 i; new(&i) int();
    intAligned4 j; new (&j) int();

    int* iu = &reinterpret_cast<int&>(i);
    *iu = 12;
    int* ju = &reinterpret_cast<int&>(j);
    *ju = 13;

    if (reinterpret_cast<unsigned long>(iu) % 8 == 0)
        std::cout << "memory pointed by the <iu> variable 
        aligned to 8 byte" << std::endl;
    else
        std::cout << "memory pointed by the <iu> variable NOT 
        aligned to 8 bytes" << std::endl;
    if (reinterpret_cast<unsigned long>(ju) % 8 == 0)
        std::cout << "memory pointed by the <ju> variable aligned to 
        8 bytes" << std::endl;
    else
        std::cout << "memory pointed by the <ju> variable NOT 
        aligned to 8 bytes" << std::endl;

    return (0);
}
```

我们特意创建了两个 typedef，一个用于对齐到`8`字节（`intAligned8`），一个用于对齐到`4`字节（`intAligned4`）。

# 它是如何工作的...

在程序中，我们定义了两个变量`i`和`j`，分别为`intAligned8`和`intAligned4`类型。借助这两个变量（分别对齐到`8`和`4`字节），我们可以通过检查除以`8`的结果是否为`0`来验证它们是否正确对齐：`((unsigned long)iu % 8 == 0)`。这确保了`iu`指针对齐到`8`字节。对`ju`变量也是同样的操作。通过运行前面的程序，我们将得到这个结果：

![](img/d2ddef14-28cf-4cc5-9991-e6f703144054.png)

预期的结果：`iu`正确对齐到`8`字节，而`ju`没有。

# 还有更多...

正如您可能已经注意到的，我们使用`reinterpret_cast`来允许模数（`%`）运算符，而不是 C 风格的转换`((unsigned long)iu % 8 == 0)`。如果您在 C++中开发，建议使用命名转换（`static_cast`、`reinterpret_cast`、`const_cast`、`dynamic_cast`）有两个基本原因：

+   允许程序员表达转换的意图

+   使转换安全

# 参见

有关此主题的更多信息可以在 W. Richard Stevens 和 Stephen A. Rago 的*UNIX 环境高级编程*中找到。

当一部分内存对齐时，编译器可以进行很好的优化。编译器无法知道这一点，因此无法进行任何优化。最新的 C++20 标准添加了`std::assume_aligned`功能。这告诉编译器指针的值是对齐到一定字节数的内存地址。可能发生的情况是，当我们分配一些对齐的内存时，该内存的指针会传递给其他函数。

`std::assume_aligned`功能告诉编译器假定指针指向的内存已经对齐，因此可以进行优化：

```cpp
void myFunc (int* p)
{
    int* pAligned = std::assume_aligned<64>(p);
    // using pAligned from now on.
}

```

`std::assume_aligned<64>(p);`功能告诉编译器`p`已经对齐到至少`64`字节。如果内存未对齐，将会得到未定义的行为。

# 处理内存映射 I/O

有时，我们需要以非常规或者说不常见的方式操作内存。正如我们所见，内存是使用`new`分配的，并使用`delete`（或者更好的是`make_unique`和`make_shared`）释放的。可能存在需要跳过某些层的情况——也就是说，使用 Linux 系统调用；出于性能考虑；或者因为我们无法使用 C++标准库来映射自定义行为。这就是`mmap` Linux 系统调用的情况（`man 2 mmap`）。`mmap`是一个符合 POSIX 标准的系统调用，允许程序员将文件映射到内存的一部分。除其他功能外，`mmap`还允许分配内存，本教程将教您如何实现。

# 如何做...

本节将展示两个`mmap`用例：第一个是如何将文件映射到内存的一部分；第二个是如何使用`mmap`分配内存。让我们首先编写一个将文件映射到内存的程序。

1.  在 shell 中，让我们创建一个名为`mmap_write.cpp`的新源文件。我们需要打开一个文件进行映射：

```cpp
 int fd = open(FILEPATH, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
 if (fd == -1)
 {
    std::cout << "Error opening file " << FILEPATH << std::endl;
    return 1;
 }
```

1.  其次，我们需要在文件中创建一个空间，以便以后使用（`mmap`不会执行此操作）：

```cpp
int result = lseek(fd, FILESIZE-1, SEEK_SET);
if (result == -1)
{
    close(fd);
    std::cout << "Error calling lseek " << std::endl;
    return 2;
}

result = write(fd, "", 1);
if (result != 1)
{
    close(fd);
    std::cout << "Error writing into the file " << std::endl;
    return 3;
}
```

1.  然后，我们可以将文件（由`fd`文件描述符表示）映射到`map`变量：

```cpp
 int* map = (int*) mmap(0, FILESIZE, PROT_READ | PROT_WRITE, 
     MAP_SHARED, fd, 0);
 if (map == MAP_FAILED)
 {
     close(fd);
     std::cout << "Error mapping the file " << std::endl;
     return 4;
 }
```

1.  最后，我们需要向其中写入一些值：

```cpp
for (int i = 1; i <=NUM_OF_ITEMS_IN_FILE; ++i)
    map[i] = 2 * i;
```

1.  不要忘记关闭使用的资源：

```cpp
if (munmap(map, FILESIZE) == -1)
    std::cout << "Error un-mapping" << std::endl;

close(fd);
```

1.  到目前为止所看到的步骤都与使用`mmap`写入文件有关。为了完整起见，在这一步中，我们将开发一个读取名为`mmap_read.cpp`的文件的程序，它与我们之前看到的非常相似。在这里，我们只会看到重要的部分（Docker 镜像包含读取器和写入器的完整版本）：

```cpp
int* map = (int*) mmap(0, FILESIZE, PROT_READ, MAP_SHARED, fd, 0);
if (map == MAP_FAILED)
{
    close(fd);
    std::cout << "Error mapping the file " << std::endl;
    return 4;
}

for (int i = 1; i <= NUM_OF_ITEMS_IN_FILE; ++i)
    std::cout << "i = " << map[i] << std::endl;
```

现在让我们学习如何使用`mmap`来分配内存。

1.  现在让我们使用`mmap`分配内存：

```cpp
#include <sys/mman.h>
#include <iostream>
#include <cstring>

constexpr auto SIZE = 1024;

int main(int argc, char *argv[])
{
    auto* mapPtr = (char*) mmap(0, SIZE, 
                                PROT_READ | PROT_WRITE, 
                                MAP_PRIVATE | MAP_ANONYMOUS, 
                                -1, 0);
 if (mapPtr == MAP_FAILED)
 {
     std::cout << "Error mapping memory " << std::endl;
     return 1;
 }
 std::cout << "memory allocated available from: " << mapPtr
   << std::endl;

 strcpy (mapPtr, "this is a string!");
 std::cout << "mapPtr val = " << mapPtr << std::endl;

 if (munmap(mapPtr, SIZE) == -1)
     std::cout << "Error un-mapping" << std::endl;

 return 0;
}
```

尽管简单，这两个程序向您展示了如何使用`mmap`分配内存和管理文件。在下一节中，我们将看到它是如何工作的。

# 它是如何工作的...

在第一个程序中，我们学习了`mmap`的最常见用法：将文件映射到内存的一部分。由于在 Linux 中几乎可以将任何资源映射到文件，这意味着我们可以使用`mmap`将几乎任何东西映射到内存中。它确实接受文件描述符。通过首先编译和运行`mmap_write.cpp`程序，我们能够在内存中写入一个整数列表的文件。生成的文件将被命名为`mmapped.txt`。有趣的部分是运行`mmap_read.cpp`读取程序。让我们编译并运行它：

![](img/04bbe152-7fff-49a6-a27f-fc23edc804d7.png)

正如我们所看到的，它正确地从文件中打印出所有的整数。

严格来说，`mmap`并不在堆内存或堆栈上分配内存。它是一个单独的内存区域，仍然在进程的虚拟空间中。`munmap`则相反：它释放映射的内存，并将数据刷新到文件（这种行为可以通过`msync`系统调用来控制）。

第二个程序展示了`mmap`的第二种用法：以一种替代`new`和`malloc`的方式分配内存。我们可以看到在调用`mmap`时有一些不同之处：

+   `MAP_PRIVATE`：修改是私有的。对内存所做的任何修改都不会反映到文件或其他映射中。文件被映射为写时复制。

+   `MAP_ANONYMOUS`：表示将分配大小为`SIZE`的一部分内存，并且不与任何特定文件关联。

+   我们传递了第五个参数`-1`，因为我们想要分配内存（即没有文件描述符）。

我们分配了 1KB 的内存并使用了一个字符串。输出如下：

![](img/b4a4ef22-fc72-4027-89d9-435480e7b79c.png)

同样，当我们使用`free`或`delete`释放内存时，我们需要使用`munmap`释放映射的内存。

# 还有更多...

有几个值得一提的优点关于`mmap`：

1.  从内存映射文件读取和写入避免了使用`mmap`与`MAP_SHARED`或`MAP_SHARED_VALIDATE`标志时`read()`和`write()`所需的复制。实际上，当我们向文件写入一块数据时，缓冲区从用户空间移动到内核空间，当读取一块数据时也是如此。

1.  读写内存映射文件实际上是一个简单的内存访问。内存映射文件只在内存中读写；在`munmap`调用时，内存被刷新回文件。这种行为可以通过`msync`系统调用的`MS_SYNC`、`MS_ASYNC`和`MS_INVALIDATE`标志参数来控制。

1.  非常方便的是，当多个进程将同一文件映射到内存中时，数据在所有进程之间共享（`MAP_SHARED`）。

# 另请参阅

查看`man 2 mmap`以获取更多信息。更多信息可以在 Robert Love 的《Linux 系统编程，第二版》中找到。

# 实际操作分配器

C++ **标准模板库**（**STL**）容器是管理资源的一种简单有效的方式。容器的一个巨大优势是它们可以管理（几乎）任何类型的数据。然而，在处理系统编程时，我们可能需要为容器提供一种替代的内存管理方式。分配器正是这样的：它们为容器提供了自定义实现。

# 如何做...

在本教程中，您将学习实现自己的自定义分配器（在本例中基于`mmap`）以提供给标准库容器（`std::vector`）：

1.  让我们首先创建一个空的分配器模板：

```cpp
template<typename T>
class mmap_allocator
{
public:
    using value_type = T;

    template<typename U> struct rebind {
        using alloc = mmap_allocator<U>;
    };

    mmap_allocator(){};
    template <typename U>
    mmap_allocator(const mmap_allocator<U> &alloc) noexcept {};

    T* allocate(std::size_t n){};

    void deallocate(T* p, std::size_t n) {}
};
```

1.  正如您所看到的，有复制构造函数、`allocate`和`deallocate`方法需要实现。让我们逐一实现它们（在这种情况下不需要实现默认构造函数）：

```cpp
    mmap_allocator(const mmap_allocator<U> &alloc) noexcept {
      (void) alloc;};
```

1.  接下来，实现`allocate`方法：

```cpp
    std::cout << "allocating ... n = " << n << std::endl;
    auto* mapPtr = static_cast<T*> (mmap(0, sizeof(T) * n, 
                                    PROT_READ | PROT_WRITE, 
                                    MAP_PRIVATE | MAP_ANONYMOUS, 
                                    -1, 0));
    if (mapPtr != MAP_FAILED)
        return static_cast<T*>(mapPtr);
    throw std::bad_alloc();
```

1.  最后，实现`deallocate`方法：

```cpp
    std::cout << "deallocating ... n = " << n << std::endl;
    (void) n;
    munmap(p, sizeof(T) * n);
```

1.  `main`方法如下：

```cpp
int main ()
{
    std::vector<int, mmap_allocator<int>> mmap_vector = {1, 2,
        3, 4, 5};

    for (auto i : mmap_vector)
        std::cout << i << std::endl;

    return 0;
}
```

正如你所看到的，使用`std::vector`对用户来说是无缝的。唯一的区别是要指定我们想要使用的分配器。这个容器将使用`mmap`和`munmap`来分配和释放内存，而不是基于`new`和`delete`的默认实现。

# 它是如何工作的...

这个程序的核心部分是两个方法：`allocate`，它返回表示分配的内存的指针，和`deallocate`，它接受要释放的内存的指针。

在第一步中，我们勾画了我们将用于分配和释放内存的接口。它是一个模板类，因为我们希望它对任何类型都有效。正如之前讨论的，我们必须实现的两种方法是`allocate`和`deallocate`。

在第二步中，我们开发了复制构造函数，当我们想要构造一个对象并传入相同类型的对象的输入时，它将被调用。我们只是返回一个`typedef`，它将指定新对象使用的分配器。

在第三步中，我们实现了构造函数，它基本上使用`mmap`为类型为`T`的对象`n`分配空间。我们已经在上一个示例中看到了`mmap`的使用，所以你可以再次阅读那个示例。

在第四步中，我们实现了`deallocate`方法，这种情况下它调用`munmap`方法，用于删除指定地址范围的映射。

最后，`main`方法展示了如何在`std::vector`中使用我们的自定义分配器（也可以是任何容器，例如 list）。在变量`mmap_vector`的定义中，我们传递了两个参数：第一个是`int`，用于告诉编译器它将是一个整数向量，第二个是`mmap_allocator<int>`，用于指示使用我们的自定义分配器`mmap_allocator`，而不是默认的分配器。

# 还有更多...

在系统编程中，有一个预先分配的内存**池**的概念，系统预留并且必须在资源的整个生命周期中使用。在这个示例中看到的`map_allocator`类可以很容易地修改为在构造函数中预先分配一部分内存，并且从内存池中获取和释放它，而不影响系统内存。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《The C++ Programming Language》详细介绍了这些主题。有关`mmap`的更多细节，请参阅*处理内存映射 I/O*示例。
