# 第二章：重温 C++

本章作为 C++ 11-20 的复习，将贯穿本书。我们将解释为什么 C++代表了一个绝佳的机会，不容错过，当涉及编写比以往更简洁和更具可移植性的高质量代码时。

本章不包含 C++（11 到 20）引入的*所有*新功能，只包括本书其余部分将使用的功能。具体来说，您将复习（如果您已经知道）或学习（如果您是新手）编写现代代码所需的最基本的新 C++技能。您将亲自动手使用 lambda 表达式、原子操作和移动语义等。

本章将涵盖以下示例：

+   理解 C++原始类型

+   Lambda 表达式

+   自动类型推断和`decltype`

+   学习原子操作的工作原理

+   学习`nullptr`的工作原理

+   智能指针 - `unique_ptr` 和 `shared_ptr`

+   学习语义的工作原理

+   理解并发性

+   理解文件系统

+   C++核心指南

+   将 GSL 添加到您的 makefile

+   理解概念

+   使用 span

+   学习范围如何工作

+   学习模块的工作原理

# 技术要求

为了让您立即尝试本章中的程序，我们设置了一个 Docker 镜像，其中包含本书中将需要的所有工具和库。它基于 Ubuntu 19.04。

为了设置它，请按照以下步骤进行：

1.  从[www.docker.com](http://www.docker.com)下载并安装 Docker Engine。

1.  从 Docker Hub 拉取镜像：`docker pull kasperondocker/system_programming_cookbook:latest`。

1.  现在应该可以使用该镜像。输入以下命令查看镜像：`docker images`。

1.  现在，您应该有以下镜像：`kasperondocker/system_programming_cookbook`。

1.  使用以下命令运行 Docker 镜像并打开交互式 shell：`docker run -it --cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`。

1.  正在运行的容器上的 shell 现在可用。使用`root@39a5a8934370/# cd /BOOK/`获取为本书章节开发的所有程序。

需要`--cap-add sys_ptrace`参数以允许 GDB 在 Docker 容器中设置断点，默认情况下 Docker 不允许。

**免责声明**：C++20 标准已经在二月底的布拉格会议上得到批准（即技术上已经最终确定）。这意味着本书使用的 GCC 编译器版本 8.3.0 不包括（或者对 C++20 的新功能支持非常有限）。因此，Docker 镜像不包括 C++20 示例代码。GCC 将最新功能的开发保留在分支中（您必须使用适当的标志，例如`-std=c++2a`）；因此，鼓励您自己尝试。因此，请克隆并探索 GCC 合同和模块分支，并尽情玩耍。

# 理解 C++原始类型

这个示例将展示 C++标准定义的所有原始数据类型，以及它们的大小。

# 如何做...

在本节中，我们将更仔细地查看 C++标准定义的原始类型以及其他重要信息。我们还将了解到，尽管标准没有为每个类型定义大小，但它定义了另一个重要参数：

1.  首先，打开一个新的终端并输入以下程序：

```cpp
#include <iostream>
#include <limits>

int main ()
 {
    // integral types section
    std::cout << "char " << int(std::numeric_limits<char>::min())
              << "-" << int(std::numeric_limits<char>::max())
              << " size (Byte) =" << sizeof (char) << std::endl;
    std::cout << "wchar_t " << std::numeric_limits<wchar_t>::min()
              << "-" <<  std::numeric_limits<wchar_t>::max()
              << " size (Byte) ="
              << sizeof (wchar_t) << std::endl;
    std::cout << "int " << std::numeric_limits<int>::min() << "-"
              << std::numeric_limits<int>::max() << " size
                  (Byte) ="
              << sizeof (int) << std::endl;
    std::cout << "bool " << std::numeric_limits<bool>::min() << "-"
              << std::numeric_limits<bool>::max() << "
                  size (Byte) ="
              << sizeof (bool) << std::endl;

    // floating point types
    std::cout << "float " << std::numeric_limits<float>::min() <<    
                  "-"
              << std::numeric_limits<float>::max() << " size
                  (Byte) ="
              << sizeof (float) << std::endl;
    std::cout << "double " << std::numeric_limits<double>::min()
                  << "-"
              << std::numeric_limits<double>::max() << " size
                  (Byte) ="
              << sizeof (double) << std::endl;
    return 0;
 }
```

1.  接下来，构建（编译和链接）`g++ primitives.cpp`。

1.  这将生成一个可执行文件，名称为`a.out`（默认）。

# 它是如何工作的...

前面程序的输出将类似于这样：

![](img/17a5c520-563d-45b6-b17d-5e3c197d535a.png)

这代表了类型可以表示的最小和最大值，以及当前平台的字节大小。

C++标准**不**定义每种类型的大小，但它定义了最小**宽度**：

+   `char`: 最小宽度= 8

+   `short int`: 最小宽度= 16

+   `int`: 最小宽度= 16

+   `long int`: 最小宽度= 32

+   `long int int`: 最小宽度= 64

这一点有着巨大的影响，因为不同的平台可能有不同的大小，程序员应该应对这一点。为了帮助我们获得关于数据类型的一些指导，有一个数据模型的概念。**数据模型**是每个实现（编译器和操作系统遵循的体系结构的 psABI）所做的一组选择（每种类型的特定大小）来定义所有原始数据类型。以下表格显示了存在的各种类型和数据模型的子集：

| **数据类型** | **LP32** | **ILP32** | **LLP64** | **LP64** |
| --- | --- | --- | --- | --- |
| `char` | 8 | 8 | 8 | 8 |
| `short int` | 16 | 16 | 16 | 16 |
| `int`  | 16 | 32 | 32 | 32 |
| `long` | 32 | 32 | 32 | 64 |
| `指针` | 32 | 32 | 64 | 64 |

Linux 内核对 64 位体系结构（x86_64）使用 LP64 数据模型。

我们简要地提到了 psABI 主题（**特定于平台的应用程序二进制接口**（**ABIs**）的缩写）。每个体系结构（例如 x86_64）都有一个 psABI 规范，操作系统遵循这个规范。**GNU 编译器集合**（**GCC**）必须知道这些细节，因为它必须知道它编译的原始类型的大小。`i386.h` GCC 头文件包含了该体系结构的原始数据类型的大小：

```cpp
root@453eb8a8d60a:~# uname -a
 Linux 453eb8a8d60a 4.9.125-linuxkit #1 SMP Fri Sep 7 08:20:28 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
```

程序输出显示，当前操作系统（实际上是我们正在运行的 Ubuntu 镜像）使用了 LP64 数据模型，这是预期的，并且机器的体系结构是 x86_64。

# 还有更多...

正如我们所见，C++标准定义了以下原始数据类型：

+   整数：`int`

+   字符：`char`

+   布尔值：`bool`

+   浮点数：`float`

+   双精度浮点数：`double`

+   空：`void`

+   宽字符：`wchar_t`

+   空指针：`nullptr_­t`

数据类型可以包含其他信息，以便定义它们的类型：

+   修饰符：`signed`、`unsigned`、`long`和`short`

+   限定词：`const`和`restrict`

+   存储类型：`auto`、`static`、`extern`和`mutable`

显然，并非所有这些附加属性都可以应用于所有类型；例如，`unsigned`不能应用于`float`和`double`类型（它们各自的 IEEE 标准不允许这样做）。

# 另请参阅

特别是对于 Linux，Linux 内核文档通常是深入研究这个问题的好地方：[`www.kernel.org/doc/html/latest`](https://www.kernel.org/doc/html/latest/)。GCC 源代码显示了每个支持的体系结构的原始数据类型的大小。请参考以下链接以了解更多信息：[`github.com/gcc-mirror/gcc`](https://github.com/gcc-mirror/gcc)。

# Lambda 表达式

**lambda 表达式**（或**lambda** **函数**）是一种方便的方式，用于定义一个匿名的、小型的、一次性使用的函数，以便在需要的地方使用。Lambda 在**标准模板库**（**STL**）中特别有用，我们将会看到。

# 如何做...

在本节中，我们将编写一些代码，以便熟悉 lambda 表达式。尽管机制很重要，但要特别注意 lambda 的代码可读性，特别是与 STL 结合使用。按照以下步骤：

1.  在这个程序中，lambda 函数获取一个整数并将其打印到标准输出。让我们打开一个名为`lambda_01.cpp`的文件，并在其中写入以下代码：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main ()
{
    std::vector<int> v {1, 2, 3, 4, 5, 6};
    for_each (begin(v), end(v), [](int x) {std::cout << x
        << std::endl;});
    return 0;
}
```

1.  在这第二个程序中，lambda 函数通过引用捕获一个前缀，并将其添加到标准输出的整数前面。让我们在一个名为`lambda_02.cpp`的文件中写入以下代码：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main ()
{
    std::vector<int> v {1, 2, 3, 4, 5, 6};
    std::string prefix ("0");
    for_each (begin(v), end(v), &prefix {std::cout
        << prefix << x << std::endl;});
    return 0;
}
```

1.  最后，我们用`g++ lambda_02.cpp`编译它。

# 它是如何工作的...

在第一个例子中，lambda 函数只是获取一个整数作为输入并打印它。请注意，代码简洁且可读。Lambda 可以通过引用`&`或值`=`捕获作用域中的变量。

第二个程序的输出如下：

![](img/271646b2-f3b5-450a-ad5c-ed95229b6c34.png)

在第二个例子中，lambda 通过引用**捕获**了变量前缀，使其对 lambda 可见。在这里，我们通过引用捕获了`prefix`变量，但我们也可以捕获以下任何一个：

+   所有变量按引用`[&]`

+   所有变量按值`[=]`

+   指定*要捕获的变量*和*如何捕获它们*`[&var1, =var2]`

有些情况下，我们必须明确指定要返回的类型，就像这种情况：

```cpp
[](int x) -> std::vector<int>{
             if (x%2)
                 return {1, 2};
             else
                 return {3, 4};
 });
```

`-> std::vector<int>`运算符，称为**尾返回类型**，告诉编译器这个 lambda 将返回一个整数向量。

# 还有更多...

Lambda 可以分解为六个部分：

1.  捕获子句：`[]`

1.  参数列表：`()`

1.  可变规范：`mutable`

1.  异常规范：`noexcept`

1.  尾返回类型：`-> type`

1.  主体：`{}`

在这里，*1*、*2*和*6*是强制性的。

虽然可选，但可变规范和异常规范值得一提，因为它们在某些情况下可能很方便。可变规范允许通过 lambda 主体修改按值传递的参数。参数列表中的变量通常是以*const-by-value*方式捕获的，因此`mutable`规范只是去除了这个限制。第二种情况是异常规范，我们可以用它来指定 lambda 可能抛出的异常。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。

# 自动类型推断和 decltype

C++提供了两种从表达式中推断类型的机制：`auto`和`decltype()`。`auto`用于从其初始化程序推断类型，而`decltype()`用于更复杂的情况推断类型。本文将展示如何使用这两种机制的示例。

# 如何做...

避免明确指定将使用的变量类型可能很方便（实际上确实如此），特别是当它特别长并且在本地使用时：

1.  让我们从一个典型的例子开始：

```cpp
std::map<int, std::string> payslips;
// ... 
for (std::map<int, 
     std::string>::const_iterator iter = payslips.begin(); 
     iter !=payslips.end(); ++iter) 
{
 // ... 
}
```

1.  现在，让我们用`auto`来重写它：

```cpp
std::map<int, std::string> payslips;
// ... 
for (auto iter = payslips.begin(); iter !=payslips.end(); ++iter) 
{
    // ... 
}
```

1.  让我们看另一个例子：

```cpp
auto speed = 123;         // speed is an int
auto height = calculate ();    // height will be of the
                         // type returned by calculate()
```

`decltype()`是 C++提供的另一种机制，可以在表达式比`auto`更复杂的情况下推断表达式的类型。

1.  让我们用一个例子来看看：

```cpp
decltype(a) y = x + 1;  // deducing the type of a
decltype(str->x) y;     // deducing the type of str->x, where str is 
                        // a struct and x 
                        // an int element of that struct
```

在这两个例子中，我们能否使用`auto`代替`decltype()`？我们将在下一节中看一看。

# 它是如何工作的...

第一个使用`auto`的例子显示，类型是在编译时从右侧参数推断出来的。`auto`用于简单的情况。

`decltype()`推断表达式的类型。在这个例子中，它定义了`y`变量，使其与`a`的类型相同。正如你可以想象的那样，这是不可能用`auto`来实现的。为什么？这很简单：`decltype()`告诉编译器*定义一个特定类型的变量*；在第一个例子中，`y`是一个与`a`相同类型的变量。而使用`auto`，类型会自动推断。

我们应该在不必显式指定变量类型的情况下使用`auto`和`decltype()`；例如，当我们需要`double`类型（而不是`float`）时。值得一提的是，`auto`和`decltype()`都推断编译器已知的表达式的类型，因此它们不是运行时机制。

# 还有更多...

有一个特殊情况必须提到。当`auto`使用`{}`（统一初始化程序）进行类型推断时，它可能会引起一些麻烦（或者至少是我们不会预期的行为）。让我们看一个例子：

```cpp
auto fuelLevel {0, 1, 2, 3, 4, 5};
```

在这种情况下，被推断的类型是`initializer_list<T>`，而不是我们可能期望的整数数组。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。

# 学习原子操作的工作原理

传统上，C 和 C++在系统编程中有着悠久的可移植代码传统。C++11 标准引入的`atomic`特性通过本地添加了操作被其他线程视为原子的保证，进一步加强了这一点。原子是一个模板，例如`template <class T> struct atomic;`或`template <class T> struct atomic<T*>;`。C++20 已经将`shared_ptr`和`weak_ptr`添加到了`T`和`T*`。现在对`atomic`变量执行的任何操作都受到其他线程的保护。

# 如何做...

`std::atomic`是现代 C++处理并发的重要方面。让我们编写一些代码来掌握这个概念：

1.  第一段代码片段展示了原子操作的基础知识。现在让我们写下这个：

```cpp
std::atomic<int> speed (0);         // Other threads have access to the speed variable
auto currentSpeed = speed.load();   // default memory order: memory_order_seq_cst
```

1.  在第二个程序中，我们可以看到`is_lock_free()`方法在实现是无锁的或者使用锁实现时返回`true`。让我们编写这段代码：

```cpp
#include <iostream>
#include <utility>
#include <atomic>
struct MyArray { int z[50]; };
struct MyStr { int a, b; };
int main()
{
     std::atomic<MyArray> myArray;
     std::atomic<MyStr> myStr;
     std::cout << std::boolalpha
               << "std::atomic<myArray> is lock free? "
               << std::atomic_is_lock_free(&myArray) << std::endl
               << "std::atomic<myStr> is lock free? "
               << std::atomic_is_lock_free(&myStr) << std::endl;
}               
```

1.  让我们编译程序。在这样做时，您可能需要向 g++添加`atomic`库（由于 GCC 的一个错误）：`g++ atomic.cpp -latomic`。

# 它是如何工作的...

`std::atomic<int> speed (0);`将`speed`变量定义为原子整数。尽管变量是原子的，但这种初始化**不是原子的**！相反，以下代码：`speed +=10;`原子地增加了`10`的速度。这意味着不会发生竞争条件。根据定义，当访问变量的线程中至少有 1 个是写入者时，就会发生竞争条件。

`std::cout << "current speed is: " << speed;`指令自动读取当前速度的值。请注意，从速度中读取值是原子的，但接下来发生的事情不是原子的（也就是说，通过`cout`打印它）。规则是读取和写入是原子的，但周围的操作不是，正如我们所见。

第二个程序的输出如下：

![](img/878ed611-133b-41a3-8388-b49f0f8a688e.png)

原子的基本操作是`load`、`store`、`swap`和`cas`（`compare and swap`的缩写），适用于所有类型的原子。根据类型，还有其他操作可用（例如`fetch_add`）。

然而，还有一个问题没有解决。为什么`myArray`使用锁而`myStr`是无锁的？原因很简单：C++为所有原始类型提供了无锁实现，而`MyStr`内部的变量是原始类型。用户将设置`myStr.a`和`myStr.b`。另一方面，`MyArray`不是基本类型，因此底层实现将使用锁。

标准保证是对于每个原子操作，每个线程都会取得进展。需要牢记的一个重要方面是，编译器经常进行代码优化。使用原子会对编译器施加关于代码如何重新排序的限制。一个限制的例子是，不能将写入`atomic`变量之前的任何代码移动到*之后*的原子写入。

# 还有更多...

在这个示例中，我们使用了名为`memory_order_seq_cst`的默认内存模型。其他可用的内存模型包括：

+   `memory_order_relaxed`：只保证当前操作的原子性。也就是说，没有保证不同线程中的内存访问与原子操作的顺序有关。

+   `memory_order_consume`：操作被排序在释放线程上所有对释放操作有依赖的内存访问发生后。

+   `memory_order_acquire`：操作被排序在释放线程上所有对内存的访问发生后。

+   `memory_order_release`：操作被排序在发生在消费或获取操作之前。

+   `memory_order_seq_cst`：操作是顺序一致的。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。此外，Herb Sutter 的*原子武器*演讲在 YouTube 上免费提供（[`www.youtube.com/watch?v=A8eCGOqgvH4`](https://www.youtube.com/watch?v=A8eCGOqgvH4)），是一个很好的介绍。

# 学习`nullptr`的工作原理

在 C++11 之前，`NULL`标识符是用于指针的。在这个示例中，我们将看到为什么这是一个问题，以及 C++11 是如何解决它的。

# 如何做...

要理解为什么`nullptr`很重要，让我们看看`NULL`的问题：

1.  让我们写下以下代码：

```cpp
bool speedUp (int speed);
bool speedUp (char* speed);
int main()  
{
    bool ok = speedUp (NULL);
}
```

1.  现在，让我们使用`nullptr`重写前面的代码：

```cpp
bool speedUp (int speed);
bool speedUp (char* speed);
int main()  
{
    bool ok = speedUp (nullptr);
}
```

# 它是如何工作的...

第一个程序可能无法编译，或者（如果可以）调用错误的方法。我们希望它调用`bool speedUp (char* speed);`。`NULL`的问题正是这样：`NULL`被定义为`0`，这是一个整数类型，并且被**预处理器**使用（替换所有`NULL`的出现）。这是一个巨大的区别，因为`nullptr`现在是 C++原始类型之一，并由**编译器**管理。

对于第二个程序，使用`char*`指针调用了`speedUp`（重载）方法。这里没有歧义 - 我们调用了`char*`类型的版本。

# 还有更多...

`nullptr`代表*不指向任何对象的指针*：

```cpp
int* p = nullptr;
```

由于这个，就没有歧义，这意味着可读性得到了提高。另一个提高可读性的例子如下：

```cpp
if (x == nullptr) 
{
    // ...\
}
```

这使得代码更易读，并清楚地表明我们正在比较一个指针。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。

# 智能指针 - unique_ptr 和 shared_ptr

这个示例将展示`unique_ptr`和`shared_ptr`的基本用法。这些智能指针是程序员的主要帮手，他们不想手动处理内存释放。一旦你学会了如何正确使用它们，这将节省头痛和夜间调试会话。

# 如何做...

在本节中，我们将看一下两个智能指针`std::unique_ptr`和`std::shared_ptr`的基本用法：

1.  让我们通过开发以下类来开发一个`unique_ptr`示例：

```cpp
#include <iostream>
#include <memory>
class CruiseControl
{
public:
    CruiseControl()
    {
        std::cout << "CruiseControl object created" << std::endl;
    };
    ~CruiseControl()
    {
        std::cout << "CruiseControl object destroyed" << std::endl;
    }
    void increaseSpeedTo(int speed)
    {
        std::cout << "Speed at " << speed << std::endl;
    };
};
```

1.  现在，让我们通过调用前面的类来开发一个`main`类：

```cpp
int main ()
{
    std::cout << "unique_ptr test started" << std::endl;
    std::unique_ptr<CruiseControl> cruiseControl =
    std::make_unique<CruiseControl>();
    cruiseControl->increaseSpeedTo(12);
    std::cout << "unique_ptr test finished" << std::endl;
}
```

1.  让我们编译`g++ unique_ptr_01.cpp`。

1.  另一个`unique_ptr`的例子展示了它在数组中的行为。让我们重用相同的类（`CruiseControl`）：

```cpp
int main ()
{
    std::cout << "unique_ptr test started" << std::endl;
    std::unique_ptr<CruiseControl[]> cruiseControl = 
        std::make_unique<CruiseControl[]>(3);
    cruiseControl[1].increaseSpeedTo(12); 
    std::cout << "unique_ptr test finished" << std::endl;
}
```

1.  让我们看看一个小程序中`std::shared_ptr`的实际应用：

```cpp
#include <iostream>
 #include <memory>
class CruiseControl
{
public:
    CruiseControl()
    {
        std::cout << "CruiseControl object created" << std::endl;
    };
    ~CruiseControl()
    {
        std::cout << "CruiseControl object destroyed" << std::endl;
    }
    void increaseSpeedTo(int speed)
    {
        std::cout << "Speed at " << speed << std::endl;
    };
};
```

`main`看起来像这样：

```cpp
int main ()
{
    std::cout << "shared_ptr test started" << std::endl;
    std::shared_ptr<CruiseControl> cruiseControlMaster(nullptr);
    {
        std::shared_ptr<CruiseControl> cruiseControlSlave = 
           std::make_shared<CruiseControl>();
        cruiseControlMaster = cruiseControlSlave;
    }
    std::cout << "shared_ptr test finished" << std::endl;
}
```

*它是如何工作的...*部分将详细描述这三个程序。

# 它是如何工作的...

通过运行第一个`unique_ptr`程序，即`./a.out`，我们得到以下输出：

![](img/a50a8dd3-47ed-411d-bb58-ac9a532dff0c.png)

`unique_ptr`是一个**智能指针**，体现了独特所有权的概念。独特所有权简单来说意味着只有一个变量可以*拥有*一个指针。这个概念的第一个结果是不允许在两个独特指针变量上使用复制运算符。只允许`move`，其中所有权从一个变量转移到另一个变量。运行的可执行文件显示，对象在当前作用域结束时被释放（在这种情况下是`main`函数）：`CruiseControl object destroyed`。开发人员不需要记得在需要时调用`delete`，但仍然可以控制内存，这是 C++相对于基于垃圾收集器的语言的主要优势之一。

在第二个`unique_ptr`示例中，使用数组，有三个`CruiseControl`类型的对象被分配然后释放。因此，输出如下：

![](img/54984bcf-dcb3-49ff-aeaa-a0c0aac1599c.png)

第三个例子展示了`shared_ptr`的用法。程序的输出如下：

![](img/9aecd0d1-6647-41a7-9f60-fcc91164b7aa.png)

`shared_ptr`智能指针代表一个对象被多个变量指向的概念（即，由所有者指向）。在这种情况下，我们谈论的是共享所有权。很明显，规则与`unique_ptr`的情况不同。一个对象**不能被释放**，直到至少有一个变量在使用它。在这个例子中，我们定义了一个指向`nullptr`的`cruiseControlMaster`变量。然后，我们定义了一个块，在该块中，我们定义了另一个变量：`cruiseControlSlave`。到目前为止一切顺利！然后，在块内部，我们将`cruiseControlSlave`指针分配给`cruiseControlMaster`。此时，分配的对象有两个指针：`cruiseControlMaster`和`cruiseControlSlave`。当此块关闭时，`cruiseControlSlave`析构函数被调用，但对象没有被释放，因为它仍然被另一个对象使用：`cruiseControlMaster`！当程序结束时，我们看到`shared_ptr test finished`日志，紧接着是`cruiseControlMaster`，因为它是唯一指向`CruiseControl`对象释放的对象，然后调用构造函数，如`CruiseControl object destroyed`日志所述。

显然，`shared_ptr`数据类型具有**引用计数**的概念来跟踪指针的数量。这些引用在构造函数（并非总是；`move`构造函数不是）和复制赋值运算符中增加，并在析构函数中减少。

引用计数变量是否可以安全地增加和减少？指向同一对象的指针可能在不同的线程中，因此操纵这个变量可能会有问题。这不是问题，因为引用计数变量是原子管理的（即，它是原子变量）。

关于大小的最后一点。`unique_ptr`的大小与原始指针一样大，而`shared_ptr`的大小通常是`unique_ptr`的两倍，因为有引用计数变量。

# 还有更多...

我强烈建议始终使用`std::make_unique`和`std::make_shared`。它们的使用消除了代码重复，并提高了异常安全性。想要更多细节吗？`shared_ptr.h`（[`github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/shared_ptr.h`](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/shared_ptr.h)）和`shared_ptr_base.h`（[`github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/shared_ptr_base.h`](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/shared_ptr_base.h)）包含了 GCC `shared_ptr`的实现，这样我们就可以看到引用计数是如何被操纵的。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。

# 学习移动语义的工作原理

我们知道复制是昂贵的，特别是对于重型对象。C++11 引入的移动语义帮助我们避免昂贵的复制。`std::move`和`std::forward`背后的基本概念是**右值引用**。这个示例将向您展示如何使用`std::move`。

# 如何做...

让我们开发三个程序来学习`std::move`及其通用引用：

1.  让我们从开发一个简单的程序开始：

```cpp
#include <iostream>
#include <vector>
int main () 
{
    std::vector<int> a = {1, 2, 3, 4, 5};
    auto b = std::move(a);
    std::cout << "a: " << a.size() << std::endl;
    std::cout << "b: " << b.size() << std::endl;
}
```

1.  让我们开发第二个例子：

```cpp
#include <iostream>
#include <vector>
void print (std::string &&s)
{
    std::cout << "print (std::string &&s)" << std::endl;
    std::string str (std::move(s));
    std::cout << "universal reference ==> str = " << str
              << std::endl;
    std::cout << "universal reference ==> s = " << s << std::endl;
}
void print (std::string &s)
{
    std::cout << "print (std::string &s)" << std::endl;
}
int main()
{
    std::string str ("this is a string");
    print (str);
    std::cout << "==> str = " << str << std::endl;
    return 0;
}
```

1.  让我们看一个通用引用的例子：

```cpp
#include <iostream>
void print (std::string &&s)
{
    std::cout << "print (std::string &&s)" << std::endl;
    std::string str (std::move(s));
    std::cout << "universal reference ==> str = " << str
              << std::endl;
    std::cout << "universal reference ==> s = " << s << std::endl;
}
void print (std::string &s)
{
    std::cout << "print (std::string &s)" << std::endl;
}
int main()
{
    print ("this is a string");
    return 0;
}
```

下一节将详细描述这三个程序。

# 工作原理...

第一个程序的输出如下（`g++ move_01.cpp`和`./a.out`）：

![](img/863d862f-50f8-46c8-894c-f4b94345d9ae.png)

在这个程序中，`auto b = std::move(a);`做了一些事情：

1.  它将向量`a`转换为**右值引用**。

1.  由于它是右值引用，所以调用了向量的移动构造函数，将`a`向量的内容移动到`b`向量中。

1.  `a`不再具有原始数据，`b`有。

第二个程序的输出如下（`g++ moveSemantics2.cpp`和`./a.out`）：

![](img/0289ab6d-50b2-4b65-9cac-6cf1cddacdbe.png)

在第二个例子中，我们传递给`print`方法的`str`字符串是一个**左值引用**（也就是说，我们可以取该变量的地址），因此它是通过引用传递的。

第三个程序的输出如下（`g++ moveSemantics3.cpp`和`./a.out`）：

![](img/bd79797d-ed44-45d1-9215-35b82981f9b3.png)

在第三个例子中，被调用的方法是带有**通用引用**作为参数的方法：`print (std::string &&s)`。这是因为我们无法取`this is a string`的地址，这意味着它是一个右值引用。

现在应该清楚了，`std::move`并没有**实际**移动任何东西-它是一个函数模板，**执行无条件转换**为右值，正如我们在第一个例子中看到的那样。这使我们能够将数据移动（而不是复制）到目标并使源无效。`std::move`的好处是巨大的，特别是每当我们看到一个方法（`T&&`）的右值引用参数，在语言的以前版本（C++98 及以前）中可能*是一个复制。

*可能：这取决于编译器的优化。

# 还有更多...

`std::forward`有些类似（但目的不同）。它是对右值引用的条件转换。您可以通过阅读下一节中引用的书籍来了解更多关于`std::forward`、右值和左值的知识。

# 另请参阅

Scott Meyers 的*Effective Modern C++*和 Bjarne Stroustrup 的*The C++ Programming Language*详细介绍了这些主题。

# 理解并发

过去，C++开发人员通常使用线程库或本地线程机制（例如`pthread`、Windows 线程）编写程序。自 C++11 以来，这已经发生了巨大的变化，并发是另一个重要的功能，它朝着一个自洽的语言方向发展。我们将在这个配方中看到的两个新特性是`std::thread`和`std::async`。

# 如何做...

在本节中，我们将学习如何在基本场景（创建和加入）中使用`std::thread`，以及如何向其传递和接收参数：

1.  `std::thread`：通过使用基本的线程方法，`create`和`join`，编写以下代码：

```cpp
#include <iostream>
#include <thread>
void threadFunction1 ();
int main()
{
    std::thread t1 {threadFunction1};
    t1.join();
    return 0;
}
void threadFunction1 ()
{
    std::cout << "starting thread 1 ... " << std::endl;
    std::cout << "end thread 1 ... " << std::endl;
}
```

1.  使用`g++ concurrency_01.cpp -lpthread`进行编译。

第二个例子与前一个例子类似，但在这种情况下，我们传递和获取参数：

1.  `std::thread`：创建和加入一个线程，传递一个参数并获取结果。编写以下代码：

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
void threadFunction (std::vector<int> &speeds, int& res);
int main()
{
    std::vector<int> speeds = {1, 2, 3, 4, 5};
    int result = 0;
    std::thread t1 (threadFunction, std::ref(speeds), 
                    std::ref(result));
    t1.join();
    std::cout << "Result = " << result << std::endl;
    return 0;
}
void threadFunction (std::vector<int> &speeds, int& res)
{
    std::cout << "starting thread 1 ... " << std::endl;
    for_each(begin(speeds), end(speeds), [](int speed) 
    {
        std::cout << "speed is " << speed << std::endl;
    });
    res = 10;
    std::cout << "end thread 1 ... " << std::endl;
}
```

1.  使用`g++ concurrency_02.cpp -lpthread`进行编译。

第三个例子使用**async**来创建一个任务，执行它，并获取结果，如下所示：

1.  `std::async`：在这里，我们可以看到为什么 async 被称为**基于任务的线程**。编写以下代码：

```cpp
root@b6e74d5cf049:/Chapter2# cat concurrency_03.cpp
#include <iostream>
#include <future>
int asyncFunction ();
int main()
{
    std::future<int> fut = std::async(asyncFunction);
    std::cout << "max = " << fut.get() << std::endl;
    return 0;
}
int asyncFunction()
{
    std::cout << "starting asyncFunction ... " << std::endl;
    int max = 0;
    for (int i = 0; i < 100000; ++i)
    {
        max += i;
    }
    std::cout << " Finished asyncFunction ..." << std::endl;
    return max;
}
```

1.  现在，我们需要编译程序。这里有一个问题。由于我们使用了线程机制，编译器依赖于本地实现，而在我们的情况下，结果是`pthread`。为了编译和链接而不出现错误（我们会得到一个未定义的引用），我们需要包含`-lpthread`：

```cpp
g++ concurrency_03.cpp -lpthread
```

在第四个例子中，`std::async`与`std::promise`和`std::future`结合使用是使两个任务相互通信的一种好而简单的方法。让我们来看一下：

1.  `std::async`：这是另一个`std::async`示例，展示了基本的通信机制。让我们编写它：

```cpp
#include <iostream>
#include <future>
void asyncProducer(std::promise<int> &prom);
void asyncConsumer(std::future<int> &fut);
int main()
{
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    std::async(asyncProducer, std::ref(prom));
    std::async(asyncConsumer, std::ref(fut));
    std::cout << "Async Producer-Consumer ended!" << std::endl;
    return 0;
}
void asyncConsumer(std::future<int> &fut)
{
    std::cout << "Got " << fut.get() << " from the producer ... "
        << std::endl;
}
void asyncProducer(std::promise<int> &prom)
{
    std::cout << " sending 5 to the consumer ... " << std::endl;
    prom.set_value (5);
}
```

1.  最后，编译它：`g++ concurrency_04.cpp -lpthread`

# 它是如何工作的...

让我们分析前面的四个程序：

1.  `std::thread`：下面的程序展示了基本的线程使用方法，用于创建和加入：

![](img/d24a2f92-5ce9-46f7-ab4f-7c3b1cba03ab.png)

在这个第一个测试中并没有什么复杂的。`std::thread`通过统一初始化用函数初始化，并加入（等待线程完成）。线程将接受一个函数对象：

```cpp
struct threadFunction 
{
    int speed;
    void operator ()();
}
std::thread t(threadFunction);
```

1.  `std::thread`：创建和加入一个线程，传递一个参数并获取结果：

![](img/ae0e61f3-9191-417d-b82f-9b9789c85852.png)

这第二个测试展示了如何通过`std::vector<int>& speeds`将参数传递给线程，并获取返回参数`int& ret`。这个测试展示了如何向线程传递参数，并且*不是*多线程代码（也就是说，如果*至少有一个*线程将对它们进行写入，那么向其他线程传递相同的参数将导致竞争条件）！

1.  `std::async`：在这里，我们可以看到为什么`async`被称为**基于任务的**线程：

![](img/2c7dfa24-2b7e-420a-a868-5dca78c347a6.png)

请注意，当我们调用`std::async(asyncFunction);`时，我们可以使用`auto fut = std::async(asyncFunction);`在编译时推断出`std::async`的返回类型。

1.  `std::async`：这是另一个`std::async`示例，展示了一种基本的通信机制：

![](img/3659b664-f69d-4cda-a2f7-1a24654284c2.png)

消费者`void asyncConsumer(std::future<int> &fut)`调用`get()`方法来获取由生产者通过`promise`的`set_value()`方法设置的值。`fut.get()`等待值的计算，如果需要的话（也就是说，这是一个阻塞调用）。

# 还有更多...

C++并发库不仅包括本示例中显示的功能，尽管这些是基础功能。您可以通过查看 Bjarne Stroustrup 的《C++程序设计语言》*第五章*第三段来探索可用的完整并发工具集。

# 另请参阅

Scott Meyers 的《Effective Modern C++》和 Bjarne Stroustrup 的《C++程序设计语言》详细介绍了这些主题。

# 理解文件系统

C++17 标志着另一个新功能方面的重大里程碑。`filesystem`库提供了一种更简单的与文件系统交互的方式。它受到了自 2003 年以来就可用的`Boost.Filesystem`的启发。本示例将展示其基本功能。

# 如何做到的...

在本节中，我们将通过使用`directory_iterator`和`create_directories`来展示`filesystem`库的两个示例。尽管在这个命名空间下肯定还有更多内容，但这两个片段的目标是突出它们的简单性：

1.  `std::filesystem::directory_iterator`：让我们编写以下代码：

```cpp
#include <iostream>
#include <filesystem>
int main()
{
    for(auto& p: std::filesystem::directory_iterator("/"))
    std::cout << p << std::endl;
}
```

1.  现在，使用`g++ filesystem_01.cpp -std=c++17 -lstdc++fs`进行编译，其中**`-std=c++17`**告诉编译器使用 C++17 标准，`-lstdc++fs`告诉编译器使用`filesystem`库。

第二个示例是关于创建目录和文件：

1.  `std::filesystem::create_directories`：编写以下代码：

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>
int main()
{
    std::filesystem::create_directories("test/src/config");
    std::ofstream("test/src/file.txt") << "This is an example!"
                                       << std::endl;
}
```

1.  编译与前面的示例相同：`g++ filesystem_02.cpp -std=c++17 -lstdc++fs`。

只需两行代码，我们就创建了一个文件夹结构、一个文件，并且还对其进行了写入！就是这么简单（而且可移植）。

# 它是如何工作的...

`filesystem`库位于`std::filesystem`命名空间下的`<filesystem>`头文件中。尽管这两个测试非常简单，但它们需要展示`filesystem`库的强大之处。第一个程序的输出如下：

![](img/e3d7f330-c990-493c-aac9-28ea974e1a71.png)

可以在这里找到`std::filesystem`方法的完整列表：[`en.cppreference.com/w/cpp/header/filesystem`](https://en.cppreference.com/w/cpp/header/filesystem)。

`std::filesystem::create_directories`在当前文件夹中创建一个目录（如果`test/src`不存在，则递归创建），在这种情况下。当然，绝对路径也是可以的，当前行也是完全有效的，即`std::filesystem::create_directories("/usr/local/test/config");`。

源代码的第二行使用`ofstream`来创建一个名为`test/src/file.txt`的输出文件流，并将`<<`附加到字符串：`This is an example!`*.*

# 还有更多...

`filesystem`库受`Boost.Filesystem`的启发，自 2003 年以来一直可用。如果你想要尝试和调试一下，只需在编译器中添加`-g`选项（将调试符号添加到二进制文件）：`g++ **-g** fs.cpp -std=c++17 -lstdc++fs`。

# 另请参阅

Scott Meyers 的书*Effective Modern C++*和 Bjarne Stroustrup 的书*The C++ Programming Language*详细介绍了这些主题。

# C++核心指南

C++核心指南是由 Bjarne Stroustrup 领导的协作努力，就像 C++语言本身一样。它们是多年来在许多组织中进行讨论和设计的结果。它们的设计鼓励普遍适用性和广泛采用，但可以自由复制和修改以满足您组织的需求。更准确地说，这些指南是指 C++14 标准。

# 准备就绪

前往 GitHub 并转到 C++核心指南文档（[`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)），以及 GitHub 项目页面：[`github.com/isocpp/CppCoreGuidelines`](https://github.com/isocpp/CppCoreGuidelines)。

# 如何做...

C++核心指南分为易于浏览的各个部分。这些部分包括类和类层次结构、资源管理、性能和错误处理。C++核心指南是由 Bjarne Stroustrup 和 Herb Sutter 领导的协作努力，但总共涉及 200 多名贡献者（要了解更多信息，请访问[`github.com/isocpp/CppCoreGuidelines/graphs/contributors`](https://github.com/isocpp/CppCoreGuidelines/graphs/contributors)）。他们提出的质量、建议和最佳实践令人难以置信。

# 它是如何工作的...

使用 C++核心指南的最常见方法是在 GitHub 页面上保持一个浏览器标签，并持续查阅它以完成日常任务。

# 还有更多...

如果您想为已提供的问题做出贡献，GitHub 页面包含许多可供选择的项目。有关更多信息，请访问[`github.com/isocpp/CppCoreGuidelines/issues`](https://github.com/isocpp/CppCoreGuidelines/issues)。

# 另请参阅

本章的*在 makefile 中添加 GSL*配方将非常有帮助。

# 在 makefile 中添加 GSL

*“GSL 是这些指南中指定的一小组类型和别名。在撰写本文时，它们的规范还不够详细；我们计划添加一个 WG21 风格的接口规范，以确保不同的实现达成一致，并提议作为可能标准化的贡献，通常受委员会决定接受/改进/更改/拒绝的影响。”* - C++核心指南的 FAQ.50。

# 准备就绪

前往 GitHub 并转到 C++核心指南文档：[`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)。

# 如何做...

在本节中，我们将通过修改 makefile 将**指南支持库**（`gsl`）集成到程序中：

1.  下载并复制`gsl`实现（例如[`github.com/microsoft/GSL`](https://github.com/microsoft/GSL)）。

1.  将`gsl`文件夹复制到您的项目中。

1.  在 makefile 中添加包含：`-I$HOME/dev/GSL/include`。

1.  在您的源文件中，包含`#include <gsl/gsl>`。

`gsl`目前提供以下内容：

+   `GSL.view`

+   `GSL.owner`

+   `GSL.assert: Assertions`

+   `GSL.util: Utilities`

+   `GSL.concept: Concepts`

# 它是如何工作的...

您可能已经注意到，要使`gsl`工作，只需在 makefile 中指定头文件夹路径，即`-I$HOME/dev/GSL/include`。还要注意的一点是，在 makefile 中没有指定任何库。

这是因为整个实现都是在`gsl`文件夹下的头文件中提供的*内联*。

# 还有更多...

Microsoft GSL ([`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)) 只是由 Microsoft 维护的一个实现。您可以在这里找到另一个实现：[`github.com/martinmoene/gsl-lite`](https://github.com/martinmoene/gsl-lite)。这两个实现都是以 MIT 许可类型发布的。

# 另请参阅

本章的《C++核心指南》示例。

# 理解概念

**概念**是与模板一起使用的编译时谓词。C++20 标准通过提供更多的编译时机会，使开发人员能够更多地传达其意图，从而明显提升了通用编程。我们可以将概念视为模板使用者必须遵守的要求（或约束）。我们为什么需要概念？您需要自己定义概念吗？这个示例将回答这些问题以及更多问题。

# 如何做...

在本节中，我们将使用`概念`开发一个具体的模板示例：

1.  我们想要创建自己版本的 C++标准库中的`std::sort`模板函数。让我们从在`.cpp`文件中编写以下代码开始：

```cpp
#include <algorithm>
#include <concepts>

namespace sp
{
    template<typename T>
        requires Sortable<T>
    void sort(T& container)
    {
        std::sort (begin(container), end(container));
    };
}
```

1.  现在，让我们使用我们的新模板类，并约束我们传递的类型，即`std::vector`必须是可排序的；否则，编译器会通知我们：

```cpp
int main()
{
    std::vector<int> myVec {2,1,4,3};
    sp::sort(vec);

    return 0;
}
```

我们将在下一节中详细讨论。

# 它是如何工作的...

我坚信`概念`是缺失的特性。在它们之前，模板没有明确定义的要求集，也没有在编译错误的情况下对其进行简单和简要的描述。这些是驱动`概念`特性设计的两个支柱。

*步骤 1*包括`std::sort`方法的`algorithms` `include`和`concepts`头文件。为了不让编译器和我们自己感到困惑，我们将新模板封装在一个命名空间`sp`中。正如您所看到的，与我们过去使用的经典模板相比，几乎没有什么区别，唯一的区别是使用了`requires`关键字。

`requires`向编译器（以及模板使用者）传达，这个模板只有在`T Sortable`类型（`Sortable<T>`）有效时才有效。好的；`Sortable`是什么？这是一个只有在评估为 true 时才满足的谓词。还有其他指定约束的方法，如下所示：

+   使用尾随`requires`：

```cpp
template<typename T>
void sort(T& container) requires Sortable<T>;
```

+   作为`模板`参数：

```cpp
template<Sortable T>
void sort(T& container)
```

我个人更喜欢*如何做...*部分的风格，因为它更符合惯用法，更重要的是，它允许我们将所有的`requires`放在一起，就像这样：

```cpp
template<typename T>
 requires Sortable<T> && Integral<T>
void sort(T& container)
{
    std::sort (begin(container), end(container));
}; 
```

在这个示例中，我们想要传达我们的`sp::sort`方法对类型`T`有效，这个类型是`Sortable`和`Integral`，出于任何原因。

*步骤 2*只是使用我们的新定制版本的 sort。为此，我们实例化了一个（`Sortable`！）向`sp::sort`方法传入输入的向量。

# 还有更多...

可能有情况需要创建自己的概念。标准库包含了大量的概念，因此您可能不需要自己创建概念。正如我们在前一节中学到的，概念只有在评估为 true 时才是谓词。将概念定义为两个现有概念的组合可能如下所示：

```cpp
template <typename T>
concept bool SignedSwappable() 
{
    return SignedIntegral<T>() && Swappable<T>();
}

```

在这里，我们可以使用`sort`方法：

```cpp
template<typename T>
 requires SignedSwappable<T>
void sort(T& container)
{
    std::sort (begin(container), end(container));
}; 
```

为什么这很酷？有几个原因：

+   它让我们立即知道模板期望什么，而不会迷失在实现细节中（也就是说，要求或约束是明确的）。

+   在编译时，编译器将评估约束是否已满足。

# 另请参阅

+   《C++之旅，第二版》，B. Stroustrup：*第 7.2 章*和*第 12.7 章*，列出了标准库中定义的概念的完整列表。

+   [`gcc.gnu.org/projects/cxx-status.html`](https://gcc.gnu.org/projects/cxx-status.html) 以获取与 GCC 版本和状态映射的 C++20 功能列表。

# 使用 span

我们可能会遇到这样的情况，我们需要编写一个方法，但我们希望能够接受普通数组或 STL 容器作为输入。`std::span`解决了这个问题。它为用户提供了对连续元素序列的视图。这个食谱将教会你如何使用它。

# 如何做...

在这个食谱中，我们将编写一个带有一个参数（`std::span`）的方法，可以在不同的上下文中使用。然后，我们将强调它提供的灵活性：

1.  让我们首先添加我们需要的包含文件。然后，我们需要通过传递`std::span`类型的`container`变量来定义`print`方法：

```cpp
#include <iostream>
#include <vector>
#include <array>
#include <span>

void print(std::span<int> container)
{
    for(const auto &c : container) 
        std::cout << c << "-";
}
```

1.  在`main`中，我们想通过调用`print`方法打印我们的数组：

```cpp
int main()
{
    int elems[]{4, 2, 43, 12};
    print(elems);

    std::vector vElems{4, 2, 43, 12};
    print(vElems);
}
```

让我们看看这是如何工作的。

# 它是如何工作的...

`std::span`描述了一个引用连续元素序列的对象。C++标准将数组定义为具有连续内存部分。这绝对简化了`std::span`的实现，因为典型的实现包括指向序列第一个元素的指针和大小。

*步骤 1*定义了通过`std::span`传递的`print`方法，我们可以将其视为整数序列。任何具有连续内存的数组类型都将从该方法中看到为序列。

*步骤 2*使用`print`方法与两个不同的数组，一个是 C 风格的，另一个是 STL 库的`std::vector`。由于这两个数组都在连续的内存部分中定义，`std::span`能够无缝地管理它们。

# 还有更多...

我们的方法考虑了带有`int`类型的`std::span`。您可能需要使该方法通用。在这种情况下，您需要编写类似于以下内容：

```cpp
template <typename T>
void print(std::span<T> container)
{
    for(const auto &c : container) 
        std::cout << c << "-";
}
```

正如我们在*理解概念*食谱中所学到的，为这个模板指定一些要求是明智的。因此，我们可能会写成以下内容：

```cpp
template <typename T>
    requires Integral<T>
void print(std::span<T> container)
{
    for(const auto &c : container) 
        std::cout << c << "-";
}
```

`requires Integral<T>`将明确指出模板需要`Integral`类型。

# 另请参阅

+   *理解概念*食谱回顾如何使用模板编写概念并将其应用于`std::span`。

+   [`gcc.gnu.org/projects/cxx-status.html`](https://gcc.gnu.org/projects/cxx-status.html)列出了与 GCC 版本及其状态映射的 C++20 功能列表。

# 学习 Ranges 的工作原理

C++20 标准添加了 Ranges，它们是对容器的抽象，允许程序统一地操作容器的元素。此外，Ranges 代表了一种非常现代和简洁的编写表达性代码的方式。我们将了解到，这种表达性在使用管道和适配器时甚至更加强大。

# 如何做...

在本节中，我们将编写一个程序，帮助我们学习 Ranges 与管道和适配器结合的主要用例。给定一个温度数组，我们想要过滤掉负数，并将正数（温暖的温度）转换为华氏度：

1.  在一个新的源文件中，输入以下代码。正如你所看到的，两个 lambda 函数和一个`for`循环完成了工作：

```cpp
#include <vector>
#include <iostream>
#include <ranges>

int main()
{
    auto temperatures{28, 25, -8, -3, 15, 21, -1};
    auto minus = [](int i){ return i <= 0; };
    auto toFahrenheit = [](int i) { return (i*(9/5)) + 32; };
    for (int t : temperatures | std::views::filter(minus) 
                              | std::views::transform(toFahrenheit)) 
        std::cout << t << ' ';  // 82.4 77 59 69.8
}
```

我们将在下一节分析 Ranges 的背后是什么。我们还将了解到 Ranges 是`concepts`的第一个用户。

# 它是如何工作的...

`std::ranges`代表了一种非常现代的方式来以可读的格式描述容器上的一系列操作。这是一种语言提高可读性的情况之一。

*步骤 1*定义了包含一些数据的`temperatures`向量。然后，我们定义了一个 lambda 函数，如果输入`i`大于或等于零，则返回 true。我们定义的第二个 lambda 将`i`转换为华氏度。然后，我们循环遍历`temperatures`（`viewable_range`），并将其传递给`filter`（在 Ranges 范围内称为`adaptor`），它根据`minus` lambda 函数删除了负温度。输出被传递给另一个适配器，它转换容器的每个单个项目，以便最终循环可以进行并打印到标准输出。

C++20 提供了另一个层次，用于迭代容器元素的层次更现代和成语化。通过将`viewable_range`与适配器结合使用，代码更加简洁、紧凑和可读。

C++20 标准库提供了许多遵循相同逻辑的适配器，包括`std::views::all`、`std::views::take`和`std::views::split`。

# 还有更多...

所有这些适配器都是使用概念来定义特定适配器需要的要求的模板。一个例子如下：

```cpp
template<ranges::input_range V,                  std::indirect_unary_predicate<ranges::iterator_t<V>> Pred >
    requires ranges::view<V> && std::is_object_v<Pred>
class filter_view : public ranges::view_interface<filter_view<V, Pred>>
```

这个模板是我们在这个配方中使用的`std::views::filter`。这个模板需要两种类型：第一种是`V`，输入范围（即容器），而第二种是`Pred`（在我们的情况下是 lambda 函数）。我们为这个模板指定了两个约束：

+   `V`必须是一个视图

+   谓词必须是对象类型：函数、lambda 等等

# 另请参阅

+   *理解概念*配方来审查概念。

+   访问[`github.com/ericniebler/range-v3`](https://github.com/ericniebler/range-v3)以查看 C++20 库提案作者（Eric Niebler）的`range`实现。

+   在第一章的*学习 Linux 基础知识-Shell*配方中，注意 C++20 范围管道与我们在 shell 上看到的管道概念非常相似。

+   要了解有关`std::is_object`的更多信息，请访问以下链接：[`en.cppreference.com/w/cpp/types/is_object`](https://en.cppreference.com/w/cpp/types/is_object)。

# 学习模块如何工作

在 C++20 之前，构建程序的唯一方法是通过`#include`指令（由预编译器解析）。最新标准添加了另一种更现代的方法来实现相同的结果，称为**模块**。这个配方将向您展示如何使用模块编写代码以及`#include`和模块之间的区别。

# 如何做...

在本节中，我们将编写一个由两个模块组成的程序。这个程序是我们在*学习范围如何工作*配方中开发的程序的改进。我们将把温度代码封装在一个模块中，并在客户端模块中使用它。让我们开始吧：

1.  让我们创建一个名为`temperature.cpp`的新`.cpp`源文件，并键入以下代码：

```cpp
export module temperature_engine;
import std.core
#include <ranges>

export 
std::vector<int> toFahrenheitFromCelsius(std::vector<int>& celsius)
{
    std::vector<int> fahrenheit;
    auto toFahrenheit = [](int i) { return (i*(9/5)) + 32; };
    for (int t : celsius | std::views::transform(toFahrenheit)) 
        fahrenheit.push_back(t);

    return fahrenheit;
}
```

1.  现在，我们必须使用它。创建一个新文件（例如`temperature_client.cpp`）并包含以下代码：

```cpp
import temperature_engine;
import std.core;  // instead of iostream, containers 
                  // (vector, etc) and algorithm
int main()
{ 
    auto celsius = {28, 25, -8, -3, 15, 21, -1};
    auto fahrenheit = toFahrenheitFromCelsius(celsius);
    std::for_each(begin(fahrenheit), end(fahrenheit),
        &fahrenheit
    {
        std::cout << i << ";";
    });
}
```

下一节将解释模块如何工作，它们与命名空间的关系以及它们相对于`#include`预编译指令的优势。

# 工作原理...

模块是 C++20 对（可能）`#include`指令的解决方案。这里可能是强制性的，因为数百万行的遗留代码不可能一夜之间转换为使用模块。

*步骤 1*的主要目标是定义我们的`temperature_engine`模块。第一行`export module temperature_engine;`定义了我们要导出的模块。接下来，我们有`import std.core`。这是 C++20 引入的最大区别之一：不再需要使用`#include`。具体来说，`import std.core`等同于`#include <iostream>`。我们还`#include`了范围。在这种情况下，我们以*旧方式*做到了这一点，以向您展示可以混合旧和新解决方案的代码。这一点很重要，因为它将使我们更好地了解如何管理到模块的过渡。每当我们想要从我们的模块中导出东西时，我们只需要用`export`关键字作为前缀，就像我们对`toFahrenheitFromCelsius`方法所做的那样。方法的实现不受影响，因此它的逻辑不会改变。

*步骤 2*包含使用`temperature_engine`的模块客户端的代码。与上一步一样，我们只需要使用`import temperature_engine`并使用导出的对象。我们还使用`import std.core`来替换`#include <iostream>`。现在，我们可以像通常一样使用导出的方法，调用`toFahrenheitFromCelsius`并传递预期的输入参数。`toFahrenheitFromCelsius`方法返回一个整数向量，表示转换后的华氏温度，这意味着我们只需要使用`for_each`模板方法通过**`import std.core`**打印值，而我们通常会使用`#include <algorithm>`。

此时的主要问题是：为什么我们应该使用模块而不是`#include`？`模块`不仅代表了一种语法上的差异 - 它比那更深刻：

+   模块只编译一次，而`#include`不会。要使`#include`只编译一次，我们需要使用`#ifdef` `#define`和`#endif`预编译器。

+   模块可以以任何顺序导入，而不会影响含义。这对`#include`来说并非如此。

+   如果一个符号没有从模块中导出，客户端代码将无法使用它，如果用户这样做，编译器将通知错误。

+   与包含不同，模块不是传递的。将模块`A`导入模块`B`，当模块`C`使用模块`B`时，并不意味着它自动获得对模块`A`的访问权限。

这对可维护性、代码结构和编译时间有很大影响。

# 还有更多...

一个经常出现的问题是，模块与命名空间是否冲突（或重叠）？这是一个很好的问题，答案是否定的。命名空间和模块解决了两个不同的问题。命名空间是另一种表达意图将一些声明分组在一起的机制。将声明分组在一起的其他机制包括函数和类。如果两个类冲突怎么办？我们可以将其中一个封装到命名空间中。您可以在*理解概念*配方中看到一个例子，我们在那里创建了我们自己的版本的 sort，称为`sp::sort`。另一方面，模块是一组逻辑功能。这两个概念是**正交**的，这意味着我可以将我的命名空间分布在更多的模块上。一个具体的例子是`std::vector`和`std::list`容器，它们位于两个不同的模块中，但在相同的`namespace`：`std`。

值得强调的另一件事是，模块允许我们将模块的一部分设置为`private`，使其对其他**翻译单元**（**TUs**）不可访问。如果要将符号导出为不完整类型，这将非常有用。

```cpp
export module temperature_engine;
import std.core
#include <ranges>

export struct ConversionFactors;  //exported as incomplete type

export 
void myMethod(ConversionFactors& factors)
{
    // ...
}

module: private;
struct ConversionFactors
{
    int toFahrenheit;
    int toCelsius;
};
```

# 另请参阅

+   转到[`gcc.gnu.org/projects/cxx-status.html`](https://gcc.gnu.org/projects/cxx-status.html)检查模块（以及其他 C++20 功能）支持时间表。

+   有关 lambda 表达式的刷新，请参阅*Lambda 表达式*配方。
