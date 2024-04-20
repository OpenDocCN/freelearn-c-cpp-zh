# 第七章：内存管理

在阅读了前面的章节之后，应该不会再感到惊讶，我们处理内存的方式对性能有很大影响。CPU 花费大量时间在 CPU 寄存器和主内存之间传输数据（加载和存储数据到主内存和从主内存中读取数据）。正如在*第四章*，*数据结构*中所示，CPU 使用内存缓存来加速对内存的访问，程序需要对缓存友好才能运行得快。

本章将揭示更多关于计算机如何处理内存的方面，以便您知道在调整内存使用时必须考虑哪些事项。此外，本章还涵盖了：

+   自动内存分配和动态内存管理。

+   C++对象的生命周期以及如何管理对象所有权。

+   高效的内存管理。有时，存在严格的内存限制，迫使我们保持数据表示紧凑，有时我们有大量的可用内存，但需要通过使内存管理更高效来加快程序运行速度。

+   如何最小化动态内存分配。分配和释放动态内存相对昂贵，有时我们需要避免不必要的分配以使程序运行更快。

我们将从解释一些概念开始这一章，这些概念在我们深入研究 C++内存管理之前需要理解。这个介绍将解释虚拟内存和虚拟地址空间，堆内存与栈内存，分页和交换空间。

# 计算机内存

计算机的物理内存是所有运行在系统上的进程共享的。如果一个进程使用了大量内存，其他进程很可能会受到影响。但从程序员的角度来看，我们通常不必担心其他进程正在使用的内存。这种内存的隔离是因为今天的大多数操作系统都是**虚拟内存**操作系统，它们提供了一个假象，即一个进程拥有了所有的内存。每个进程都有自己的**虚拟地址空间**。

## 虚拟地址空间

程序员看到的虚拟地址空间中的地址由操作系统和处理器的**内存管理单元**（**MMU**）映射到物理地址。每次访问内存地址时都会发生这种映射或转换。

这种额外的间接层使操作系统能够使用物理内存来存储进程当前正在使用的部分，并将其余的虚拟内存备份到磁盘上。在这个意义上，我们可以把物理主内存看作是虚拟内存空间的缓存，而虚拟内存空间位于辅助存储上。通常用于备份内存页面的辅助存储区域通常称为**交换空间**、**交换文件**或简单地称为**页面文件**，具体取决于操作系统。

虚拟内存使进程能够拥有比物理地址空间更大的虚拟地址空间，因为未使用的虚拟内存不需要占用物理内存。

## 内存页面

实现虚拟内存的最常见方式是将地址空间划分为称为**内存页面**的固定大小块。当一个进程访问虚拟地址处的内存时，操作系统会检查内存页面是否由物理内存（页面帧）支持。如果内存页面没有映射到主内存中，将会发生硬件异常，并且页面将从磁盘加载到内存中。这种硬件异常称为**页面错误**。这不是错误，而是为了从磁盘加载数据到内存而必要的中断。不过，正如你可能已经猜到的那样，这与读取已经驻留在内存中的数据相比非常慢。

当主内存中没有更多可用的页面帧时，必须驱逐一个页面帧。如果要驱逐的页面是脏的，也就是说，自从上次从磁盘加载以来已经被修改，那么它需要被写入磁盘才能被替换。这种机制称为**分页**。如果内存页面没有被修改，那么内存页面就会被简单地驱逐。

并非所有支持虚拟内存的操作系统都支持分页。例如，iOS 具有虚拟内存，但脏页面永远不会存储在磁盘上；只有干净的页面才能从内存中驱逐。如果主内存已满，iOS 将开始终止进程，直到再次有足够的空闲内存。Android 使用类似的策略。不将内存页面写回移动设备的闪存存储的原因之一是它会消耗电池电量，还会缩短闪存存储本身的寿命。

下图显示了两个运行中的进程。它们都有自己的虚拟内存空间。一些页面映射到物理内存，而另一些则没有。如果进程 1 需要使用从地址 0x1000 开始的内存页面，就会发生页面错误。然后该内存页面将被映射到一个空闲的内存帧。还要注意虚拟内存地址与物理地址不同。进程 1 的第一个内存页面，从虚拟地址 0x0000 开始，映射到从物理地址 0x4000 开始的内存帧：

![](img/B15619_07_01.png)

图 7.1：虚拟内存页面，映射到物理内存中的内存帧。未使用的虚拟内存页面不必占用物理内存。

## 抖动

**抖动**可能发生在系统的物理内存不足且不断分页的情况下。每当一个进程在 CPU 上被调度时，它试图访问已被分页出去的内存。加载新的内存页面意味着其他页面首先必须存储在磁盘上。在磁盘和内存之间来回移动数据通常非常缓慢；在某些情况下，这几乎会使计算机停滞，因为系统花费了所有的时间在分页上。查看系统的页面错误频率是确定程序是否开始抖动的好方法。

了解硬件和操作系统如何处理内存的基础知识对于优化性能很重要。接下来，我们将看到在执行 C++程序时内存是如何处理的。

# 进程内存

堆栈和堆是 C++程序中最重要的两个内存段。还有静态存储和线程本地存储，但我们稍后会更多地讨论这些。实际上，严格来说，C++并不谈论堆栈和堆；相反，它谈论自由存储、存储类和对象的存储持续时间。然而，由于堆栈和堆的概念在 C++社区中被广泛使用，并且我们所知道的所有 C++实现都使用堆栈来实现函数调用和管理局部变量的自动存储，因此了解堆栈和堆是很重要的。

在本书中，我还将使用术语*堆栈*和*堆*而不是对象的存储持续时间。我将使用术语*堆*和*自由存储*互换使用，并不会对它们进行区分。

堆栈和堆都驻留在进程的虚拟内存空间中。堆栈是所有局部变量驻留的地方；这也包括函数的参数。每次调用函数时，堆栈都会增长，并在函数返回时收缩。每个线程都有自己的堆栈，因此堆栈内存可以被视为线程安全。另一方面，堆是一个在运行进程中所有线程之间共享的全局内存区域。当我们使用`new`（或 C 库函数`malloc()`和`calloc()`）分配内存时，堆会增长，并在使用`delete`（或`free()`）释放内存时收缩。通常，堆从低地址开始增长，向上增长，而堆栈从高地址开始增长，向下增长。*图 7.2*显示了堆栈和堆在虚拟地址空间中以相反方向增长：

![](img/B15619_07_02.png)

图 7.2：进程的地址空间。堆栈和堆以相反方向增长。

接下来的部分将提供有关堆栈和堆的更多细节，并解释在我们编写的 C++程序中何时使用这些内存区域。

## 堆栈内存

堆栈在许多方面与堆不同。以下是堆栈的一些独特属性：

+   堆栈是一个连续的内存块。

+   它有一个固定的最大大小。如果程序超出最大堆栈大小，程序将崩溃。这种情况称为堆栈溢出。

+   堆栈内存永远不会变得分散。

+   从堆栈中分配内存（几乎）总是很快的。页面错误可能会发生，但很少见。

+   程序中的每个线程都有自己的堆栈。

本节中接下来的代码示例将检查其中一些属性。让我们从分配和释放开始，以了解堆栈在程序中的使用方式。

通过检查堆栈分配的数据的地址，我们可以轻松找出堆栈增长的方向。以下示例代码演示了进入和离开函数时堆栈的增长和收缩：

```cpp
void func1() {
  auto i = 0;
  std::cout << "func1(): " << std::addressof(i) << '\n';
}
void func2() {
  auto i = 0;
  std::cout << "func2(): " << std::addressof(i) << '\n';
  func1();
}

int main() { 
  auto i = 0; 
  std::cout << "main():  " << std::addressof(i) << '\n'; 
  func2();
  func1(); 
} 
```

运行程序时可能的输出如下：

```cpp
main():  0x7ea075ac 
func2(): 0x7ea07594 
func1(): 0x7ea0757c 
func1(): 0x7ea07594 
```

通过打印堆栈分配的整数的地址，我们可以确定堆栈在我的平台上增长了多少，以及增长的方向。每次我们进入`func1()`或`func2()`时，堆栈都会增加 24 个字节。整数`i`将分配在堆栈上，长度为 4 个字节。剩下的 20 个字节包含在函数结束时需要的数据，例如返回地址，可能还有一些用于对齐的填充。

以下图示说明了程序执行期间堆栈的增长和收缩。第一个框说明了程序刚进入`main()`函数时内存的样子。第二个框显示了当我们执行`func1()`时堆栈的增加，依此类推：

![](img/B15619_07_03.png)

图 7.3：当进入函数时，堆栈增长和收缩

堆栈分配的总内存是在线程启动时创建的固定大小的连续内存块。那么，堆栈有多大，当我们达到堆栈的限制时会发生什么呢？

如前所述，每次程序进入函数时，堆栈都会增长，并在函数返回时收缩。每当我们在同一函数内创建新的堆栈变量时，堆栈也会增长，并在此类变量超出范围时收缩。堆栈溢出的最常见原因是深度递归调用和/或在堆栈上使用大型自动变量。堆栈的最大大小在不同平台之间有所不同，并且还可以为单个进程和线程进行配置。

让我们看看是否可以编写一个程序来查看默认情况下系统的堆栈有多大。我们将首先编写一个名为`func()`的函数，该函数将无限递归。在每个函数的开始，我们将分配一个 1 千字节的变量，每次进入`func()`时都会将其放入堆栈。每次执行`func()`时，我们打印堆栈的当前大小：

```cpp
void func(std::byte* stack_bottom_addr) { 
  std::byte data[1024];     
  std::cout << stack_bottom_addr - data << '\n'; 
  func(stack_bottom_addr); 
} 

int main() { 
  std::byte b; 
  func(&b); 
} 
```

堆栈的大小只是一个估计值。我们通过从`main()`中定义的第一个局部变量的地址减去`func()`中定义的第一个局部变量的地址来计算它。

当我用 Clang 编译代码时，我收到一个警告，即`func()`永远不会返回。通常，这是一个我们不应该忽略的警告，但这次，这正是我们想要的结果，所以我们忽略了警告并运行了程序。程序在堆栈达到其限制后不久崩溃。在程序崩溃之前，它设法打印出数千行堆栈的当前大小。输出的最后几行看起来像这样：

```cpp
... 
8378667 
8379755 
8380843 
```

由于我们在减去`std::byte`指针，所以大小以字节为单位，因此在我的系统上，堆栈的最大大小似乎约为 8 MB。在类 Unix 系统上，可以使用`ulimit`命令和选项`-s`来设置和获取进程的堆栈大小：

```cpp
$ ulimit -s
$ 8192 
```

`ulimit`（用户限制的缩写）返回以千字节为单位的最大堆栈大小的当前设置。`ulimit`的输出证实了我们实验的结果：如果我没有显式配置，我的 Mac 上的堆栈大约为 8 MB。

在 Windows 上，默认的堆栈大小通常设置为 1 MB。如果堆栈大小没有正确配置，那么在 Windows 上运行良好的程序在 macOS 上可能会因堆栈溢出而崩溃。

通过这个例子，我们还可以得出结论，我们不希望用尽堆栈内存，因为当发生这种情况时，程序将崩溃。在本章的后面，我们将看到如何实现一个基本的内存分配器来处理固定大小的分配。然后我们将了解到，堆栈只是另一种类型的内存分配器，可以非常高效地实现，因为使用模式总是顺序的。我们总是在堆栈的顶部（连续内存的末尾）请求和释放内存。这确保了堆栈内存永远不会变得碎片化，并且我们可以通过仅移动堆栈指针来分配和释放内存。

## 堆内存

堆（或者更正确的术语是自由存储区，在 C++中）是动态存储数据的地方。如前所述，堆在多个线程之间共享，这意味着堆的内存管理需要考虑并发性。这使得堆中的内存分配比堆栈分配更复杂，因为堆中的内存分配是每个线程的本地分配。

堆栈内存的分配和释放模式是顺序的，即内存总是按照分配的相反顺序进行释放。另一方面，对于动态内存，分配和释放可以任意发生。对象的动态生命周期和内存分配的变量大小增加了**内存碎片**的风险。

理解内存碎片问题的简单方法是通过一个示例来说明内存如何发生碎片化。假设我们有一个小的连续内存块，大小为 16 KB，我们正在从中分配内存。我们正在分配两种类型的对象：类型**A**，大小为 1 KB，和类型**B**，大小为 2 KB。我们首先分配一个类型**A**的对象，然后是一个类型**B**的对象。这样重复，直到内存看起来像下面的图像：

![](img/B15619_07_04.png)

图 7.4：分配类型 A 和 B 对象后的内存

接下来，所有类型**A**的对象都不再需要，因此它们可以被释放。内存现在看起来像这样：

![](img/B15619_07_05.png)

图 7.5：释放类型 A 对象后的内存

现在有 10KB 的内存正在使用，还有 6KB 可用。现在，假设我们想要分配一个类型为**B**的新对象，它占用 2KB。尽管有 6KB 的空闲内存，但我们找不到 2KB 的内存块，因为内存已经变得碎片化。

现在您已经对计算机内存在运行过程中的结构和使用有了很好的理解，现在是时候探索 C++对象在内存中的生存方式了。

# 内存中的对象

我们在 C++程序中使用的所有对象都驻留在内存中。在这里，我们将探讨如何在内存中创建和删除对象，并描述对象在内存中的布局方式。

## 创建和删除对象

在本节中，我们将深入探讨使用`new`和`delete`的细节。考虑以下使用`new`在自由存储器上创建对象，然后使用`delete`删除它的方式：

```cpp
auto* user = new User{"John"};  // allocate and construct 
user->print_name();             // use object 
delete user;                    // destruct and deallocate 
```

我不建议以这种方式显式调用`new`和`delete`，但现在让我们忽略这一点。让我们来重点讨论一下；正如注释所建议的那样，`new`实际上做了两件事，即：

+   分配内存以容纳`User`类型的新对象

+   通过调用`User`类的构造函数在分配的内存空间中构造一个新的`User`对象

同样的事情也适用于`delete`，它：

+   通过调用其析构函数来销毁`User`对象

+   释放`User`对象所在的内存

实际上，在 C++中可以将这两个操作（内存分配和对象构造）分开。这很少使用，但在编写库组件时有一些重要和合法的用例。

### 放置 new

C++允许我们将内存分配与对象构造分开。例如，我们可以使用`malloc()`分配一个字节数组，并在该内存区域中构造一个新的`User`对象。看一下以下代码片段：

```cpp
auto* memory = std::malloc(sizeof(User));
auto* user = ::new (memory) User("john"); 
```

使用`::new (memory)`的可能不熟悉的语法称为**放置 new**。这是`new`的一种非分配形式，它只构造一个对象。`::`前面的双冒号确保了从全局命名空间进行解析，以避免选择`operator new`的重载版本。

在前面的示例中，放置 new 构造了`User`对象，并将其放置在指定的内存位置。由于我们使用`std::malloc()`为单个对象分配内存，所以它保证了正确的对齐（除非`User`类已声明为过对齐）。稍后，我们将探讨在使用放置 new 时必须考虑对齐的情况。

没有放置删除，因此为了销毁对象并释放内存，我们需要显式调用析构函数，然后释放内存：

```cpp
user->~User();
std::free(memory); 
```

这是您应该显式调用析构函数的唯一时机。除非您使用放置 new 创建了一个对象，否则永远不要这样调用析构函数。

C++17 在`<memory>`中引入了一组实用函数，用于在不分配或释放内存的情况下构造和销毁对象。因此，现在可以使用一些以`std::uninitialized_`开头的函数来构造、复制和移动对象到未初始化的内存区域，而不是调用放置 new。而且，现在可以使用`std::destroy_at()`在特定内存地址上销毁对象，而无需释放内存。

前面的示例可以使用这些新函数重写。下面是它的样子：

```cpp
auto* memory = std::malloc(sizeof(User));
auto* user_ptr = reinterpret_cast<User*>(memory);
std::uninitialized_fill_n(user_ptr, 1, User{"john"});
std::destroy_at(user_ptr);
std::free(memory); 
```

C++20 还引入了`std::construct_at()`，它使得可以用它来替换`std::uninitialized_fill_n()`的调用：

```cpp
std::construct_at(user_ptr, User{"john"});        // C++20 
```

请记住，我们展示这些裸露的低级内存设施是为了更好地理解 C++中的内存管理。在 C++代码库中，使用`reinterpret_cast`和这里演示的内存实用程序应该保持绝对最低限度。

接下来，您将看到当我们使用`new`和`delete`表达式时调用了哪些操作符。

### new 和 delete 操作符

函数 `operator new` 负责在调用 `new` 表达式时分配内存。`new` 运算符可以是全局定义的函数，也可以是类的静态成员函数。可以重载全局运算符 `new` 和 `delete`。在本章后面，我们将看到在分析内存使用情况时，这可能是有用的。

以下是如何做到这一点：

```cpp
auto operator new(size_t size) -> void* { 
  void* p = std::malloc(size); 
  std::cout << "allocated " << size << " byte(s)\n"; 
  return p; 
} 

auto operator delete(void* p) noexcept -> void { 
  std::cout << "deleted memory\n"; 
  return std::free(p); 
} 
```

我们可以验证我们重载的运算符在创建和删除 `char` 对象时是否真的被使用：

```cpp
auto* p = new char{'a'}; // Outputs "allocated 1 byte(s)"
delete p;                // Outputs "deleted memory" 
```

使用 `new[]` 和 `delete[]` 表达式创建和删除对象数组时，还使用了另一对运算符，即 `operator new[]` 和 `operator delete[]`。我们可以以相同的方式重载这些运算符：

```cpp
auto operator new[](size_t size) -> void* {
  void* p = std::malloc(size); 
  std::cout << "allocated " << size << " byte(s) with new[]\n"; 
  return p; 
} 

auto operator delete[](void* p) noexcept -> void { 
  std::cout << "deleted memory with delete[]\n"; 
  return std::free(p); 
} 
```

请记住，如果重载了 `operator new`，还应该重载 `operator delete`。分配和释放内存的函数是成对出现的。内存应该由分配该内存的分配器释放。例如，使用 `std::malloc()` 分配的内存应始终使用 `std::free()` 释放，而使用 `operator new[]` 分配的内存应使用 `operator delete[]` 释放。

还可以覆盖特定于类的 `operator new` 或 `operator delete`。这可能比重载全局运算符更有用，因为更有可能需要为特定类使用自定义动态内存分配器。

在这里，我们正在为 `Document` 类重载 `operator new` 和 `operator delete`：

```cpp
class Document { 
// ...
public:  
  auto operator new(size_t size) -> void* {
    return ::operator new(size);
  } 
  auto operator delete(void* p) -> void {
    ::operator delete(p); 
  } 
}; 
```

当我们创建新的动态分配的 `Document` 对象时，将使用特定于类的 `new` 版本：

```cpp
auto* p = new Document{}; // Uses class-specific operator new
delete p; 
```

如果我们希望使用全局 `new` 和 `delete`，仍然可以通过使用全局作用域 (`::`) 来实现：

```cpp
auto* p = ::new Document{}; // Uses global operator new
::delete p; 
```

我们将在本章后面讨论内存分配器，然后我们将看到重载的 `new` 和 `delete` 运算符的使用。

迄今为止，总结一下，`new`表达式涉及两个方面：分配和构造。`operator new`分配内存，您可以全局或按类重载它以自定义动态内存管理。放置 new 可用于在已分配的内存区域中构造对象。

另一个重要但相当低级的主题是我们需要了解以有效使用内存的**内存对齐**。

## 内存对齐

CPU 每次从内存中读取一个字时，将其读入寄存器。64 位架构上的字大小为 64 位，32 位架构上为 32 位，依此类推。为了使 CPU 在处理不同数据类型时能够高效工作，它对不同类型的对象所在的地址有限制。C++ 中的每种类型都有一个对齐要求，定义了内存中应该位于某种类型对象的地址。

如果类型的对齐方式为 1，则表示该类型的对象可以位于任何字节地址。如果类型的对齐方式为 2，则表示允许地址之间的字节数为 2。或者引用 C++ 标准的说法：

> "对齐是一个实现定义的整数值，表示给定对象可以分配的连续地址之间的字节数。"

我们可以使用 `alignof` 来查找类型的对齐方式：

```cpp
// Possible output is 4  
std::cout << alignof(int) << '\n'; 
```

当我运行此代码时，输出为 `4`，这意味着在我的平台上，类型 `int` 的对齐要求为 4 字节。

以下图示显示了来自具有 64 位字的系统的内存的两个示例。上排包含三个 4 字节整数，它们位于 4 字节对齐的地址上。CPU 可以以高效的方式将这些整数加载到寄存器中，并且在访问其中一个 `int` 成员时永远不需要读取多个字。将其与第二排进行比较，其中包含两个 `int` 成员，它们位于不对齐的地址上。第二个 `int` 甚至跨越了两个字的边界。在最好的情况下，这只是低效，但在某些平台上，程序将崩溃：

![](img/B15619_07_06.png)

图 7.6：包含整数的内存的两个示例，分别位于对齐和不对齐的内存地址

假设我们有一个对齐要求为 2 的类型。C++标准没有规定有效地址是 1、3、5、7...还是 0、2、4、6...。我们所知道的所有平台都是从 0 开始计算地址，因此实际上我们可以通过使用取模运算符（`%`）来检查对象是否正确对齐。

但是，如果我们想编写完全可移植的 C++代码，我们需要使用`std::align()`而不是取模来检查对象的对齐。`std::align()`是来自`<memory>`的一个函数，它将根据我们传递的对齐方式调整指针。如果我们传递给它的内存地址已经对齐，指针将不会被调整。因此，我们可以使用`std::align()`来实现一个名为`is_aligned()`的小型实用程序函数，如下所示：

```cpp
bool is_aligned(void* ptr, std::size_t alignment) {
  assert(ptr != nullptr);
  assert(std::has_single_bit(alignment)); // Power of 2
  auto s = std::numeric_limits<std::size_t>::max();
  auto aligned_ptr = ptr;
  std::align(alignment, 1, aligned_ptr, s);
  return ptr == aligned_ptr;
} 
```

首先，我们确保`ptr`参数不为空，并且`alignment`是 2 的幂，这是 C++标准中规定的要求。我们使用 C++20 `<bit>`头文件中的`std::has_single_bit()`来检查这一点。接下来，我们调用`std::align()`。`std::align()`的典型用法是当我们有一定大小的内存缓冲区，我们想要在其中存储具有一定对齐要求的对象。在这种情况下，我们没有缓冲区，也不关心对象的大小，因此我们说对象的大小为 1，缓冲区是`std::size_t`的最大值。然后，我们可以比较原始的`ptr`和调整后的`aligned_ptr`，以查看原始指针是否已经对齐。我们将在接下来的示例中使用这个实用程序。

使用`new`或`std::malloc()`分配内存时，我们获得的内存应正确对齐为我们指定的类型。以下代码显示，为`int`分配的内存在我的平台上至少是 4 字节对齐的：

```cpp
auto* p = new int{};
assert(is_aligned(p, 4ul)); // True 
```

实际上，`new`和`malloc()`保证始终返回适合任何标量类型的内存（如果它成功返回内存的话）。`<cstddef>`头文件为我们提供了一个名为`std::max_align_t`的类型，其对齐要求至少与所有标量类型一样严格。稍后，我们将看到在编写自定义内存分配器时，这种类型是有用的。因此，即使我们只请求自由存储器上的`char`内存，它也将适合于`std::max_align_t`。

以下代码显示，从`new`返回的内存对于`std::max_align_t`和任何标量类型都是正确对齐的：

```cpp
auto* p = new char{}; 
auto max_alignment = alignof(std::max_align_t);
assert(is_aligned(p, max_alignment)); // True 
```

让我们使用`new`连续两次分配`char`：

```cpp
auto* p1 = new char{'a'};
auto* p2 = new char{'b'}; 
```

然后，内存可能看起来像这样：

![](img/B15619_07_07.png)

图 7.7：分配两个单独的 char 后的内存布局

`p1`和`p2`之间的空间取决于`std::max_align_t`的对齐要求。在我的系统上，它是`16`字节，因此每个`char`实例之间有 15 个字节，即使`char`的对齐只有 1。

在使用`alignas`指定符声明变量时，可以指定比默认对齐更严格的自定义对齐要求。假设我们的缓存行大小为 64 字节，并且出于某种原因，我们希望确保两个变量位于不同的缓存行上。我们可以这样做：

```cpp
alignas(64) int x{};
alignas(64) int y{};
// x and y will be placed on different cache lines 
```

在定义类型时，也可以指定自定义对齐。以下是一个在使用时将占用一整个缓存行的结构体：

```cpp
struct alignas(64) CacheLine {
    std::byte data[64];
}; 
```

现在，如果我们创建一个类型为`CacheLine`的栈变量，它将根据 64 字节的自定义对齐进行对齐：

```cpp
int main() {
  auto x = CacheLine{};
  auto y = CacheLine{};
  assert(is_aligned(&x, 64));
  assert(is_aligned(&y, 64));
  // ...
} 
```

在堆上分配对象时，也满足了更严格的对齐要求。为了支持具有非默认对齐要求的类型的动态分配，C++17 引入了`operator new()`和`operator delete()`的新重载，它们接受`std::align_val_t`类型的对齐参数。在`<cstdlib>`中还定义了一个`aligned_alloc()`函数，可以用于手动分配对齐的堆内存。

以下是一个示例，我们在其中分配一个应该占用一个内存页面的堆内存块。在这种情况下，使用`new`和`delete`时将调用对齐感知版本的`operator new()`和`operator delete()`：

```cpp
constexpr auto ps = std::size_t{4096};      // Page size
struct alignas(ps) Page {
    std::byte data_[ps];
};
auto* page = new Page{};                    // Memory page
assert(is_aligned(page, ps));               // True
// Use page ...
delete page; 
```

内存页面不是 C++抽象机器的一部分，因此没有可移植的方法来以编程方式获取当前运行系统的页面大小。但是，您可以在 Unix 系统上使用`boost::mapped_region::get_page_size()`或特定于平台的系统调用，如`getpagesize()`。

要注意的最后一个警告是，支持的对齐集由您使用的标准库的实现定义，而不是 C++标准。

## 填充

编译器有时需要为我们定义的用户定义类型添加额外的字节，**填充**。当我们在类或结构中定义数据成员时，编译器被迫按照我们定义它们的顺序放置成员。

然而，编译器还必须确保类内的数据成员具有正确的对齐方式；因此，如果需要，它需要在数据成员之间添加填充。例如，假设我们有一个如下所示的类：

```cpp
class Document { 
  bool is_cached_{}; 
  double rank_{}; 
  int id_{}; 
};
std::cout << sizeof(Document) << '\n'; // Possible output is 24 
```

可能输出为 24 的原因是，编译器在`bool`和`int`之后插入填充，以满足各个数据成员和整个类的对齐要求。编译器将`Document`类转换为类似于这样的形式：

```cpp
class Document {
  bool is_cached_{};
  std::byte padding1[7]; // Invisible padding inserted by compiler
  double rank_{};
  int id_{};
  std::byte padding2[4]; // Invisible padding inserted by compiler
}; 
```

`bool`和`double`之间的第一个填充为 7 字节，因为`double`类型的`rank_`数据成员具有 8 字节的对齐。在`int`之后添加的第二个填充为 4 字节。这是为了满足`Document`类本身的对齐要求。具有最大对齐要求的成员也决定了整个数据结构的对齐要求。在我们的示例中，这意味着`Document`类的总大小必须是 8 的倍数，因为它包含一个 8 字节对齐的`double`值。

我们现在意识到，我们可以重新排列`Document`类中数据成员的顺序，以最小化编译器插入的填充，方法是从具有最大对齐要求的类型开始。让我们创建`Document`类的新版本：

```cpp
// Version 2 of Document class
class Document {
  double rank_{}; // Rearranged data members
  int id_{};
  bool is_cached_{};
}; 
```

通过重新排列成员，编译器现在只需要在`is_cached_`数据成员之后填充，以调整`Document`的对齐方式。这是填充后类的样子：

```cpp
// Version 2 of Document class after padding
class Document { 
  double rank_{}; 
  int id_{}; 
  bool is_cached_{}; 
  std::byte padding[3]; // Invisible padding inserted by compiler 
}; 
```

新的`Document`类的大小现在只有 16 字节，而第一个版本为 24 字节。这里的见解应该是，对象的大小可以通过更改成员声明的顺序而改变。我们还可以通过在我们更新的`Document`版本上再次使用`sizeof`运算符来验证这一点：

```cpp
std::cout << sizeof(Document) << '\n'; // Possible output is 16 
```

以下图片显示了`Document`类版本 1 和版本 2 的内存布局：

![](img/B15619_07_08.png)

图 7.8：`Document`类的两个版本的内存布局。对象的大小可以通过更改成员声明的顺序而改变。

一般规则是，将最大的数据成员放在开头，最小的成员放在末尾。这样，您可以最小化填充引起的内存开销。稍后，我们将看到，在将对象放置在我们已分配的内存区域时，我们需要考虑对齐，然后才能知道我们正在创建的对象的对齐方式。

从性能的角度来看，也可能存在一些情况，你希望将对象对齐到缓存行，以最小化对象跨越的缓存行数量。在谈论缓存友好性时，还应该提到，将频繁一起使用的多个数据成员放在一起可能是有益的。

保持数据结构紧凑对性能很重要。许多应用程序受到内存访问时间的限制。内存管理的另一个重要方面是永远不要泄漏或浪费不再需要的对象的内存。通过清晰和明确地表达资源的所有权，我们可以有效地避免各种资源泄漏。这是接下来章节的主题。

# 内存所有权

资源的所有权是编程时需要考虑的基本方面。资源的所有者负责在不再需要资源时释放资源。资源通常是一块内存，但也可能是数据库连接、文件句柄等。无论使用哪种编程语言，所有权都很重要。然而，在诸如 C 和 C++之类的语言中更为明显，因为动态内存不会默认进行垃圾回收。每当我们在 C++中分配动态内存时，都必须考虑该内存的所有权。幸运的是，语言中现在有非常好的支持，可以通过使用智能指针来表达各种所有权类型，我们将在本节后面介绍。

标准库中的智能指针帮助我们指定动态变量的所有权。其他类型的变量已经有了定义的所有权。例如，局部变量由当前作用域拥有。当作用域结束时，在作用域内创建的对象将被自动销毁：

```cpp
{
  auto user = User{};
} // user automatically destroys when it goes out of scope 
```

静态和全局变量由程序拥有，并将在程序终止时被销毁：

```cpp
static auto user = User{}; 
```

数据成员由它们所属的类的实例拥有：

```cpp
class Game {
  User user; // A Game object owns the User object
  // ...
}; 
```

只有动态变量没有默认所有者，程序员需要确保所有动态分配的变量都有一个所有者来控制变量的生命周期：

```cpp
auto* user = new User{}; // Who owns user now? 
```

在现代 C++中，我们可以在大部分代码中不显式调用`new`和`delete`，这是一件好事。手动跟踪`new`和`delete`的调用很容易成为内存泄漏、双重删除和其他令人讨厌的错误的问题。原始指针不表达任何所有权，如果我们只使用原始指针引用动态内存，所有权很难跟踪。

我建议你清晰和明确地表达所有权，但努力最小化手动内存管理。通过遵循一些相当简单的规则来处理内存的所有权，你将增加代码干净和正确的可能性，而不会泄漏资源。接下来的章节将指导你通过一些最佳实践来实现这一目的。

## 隐式处理资源

首先，使你的对象隐式处理动态内存的分配/释放：

```cpp
auto func() {
  auto v = std::vector<int>{1, 2, 3, 4, 5};
} 
```

在前面的例子中，我们同时使用了栈和动态内存，但我们不必显式调用`new`和`delete`。我们创建的`std::vector`对象是一个自动对象，将存储在栈上。由于它由作用域拥有，当函数返回时将自动销毁。`std::vector`对象本身使用动态内存来存储整数元素。当`v`超出作用域时，它的析构函数可以安全地释放动态内存。让析构函数释放动态内存的这种模式使得避免内存泄漏相当容易。

当我们谈论释放资源时，我认为提到 RAII 是有意义的。**RAII**是一个众所周知的 C++技术，缩写为**Resource Acquisition Is Initialization**，其中资源的生命周期由对象的生命周期控制。这种模式简单但对于处理资源（包括内存）非常有用。但是，假设我们需要的资源是用于发送请求的某种连接。每当我们使用连接完成后，我们（所有者）必须记得关闭它。以下是我们手动打开和关闭连接以发送请求时的示例：

```cpp
auto send_request(const std::string& request) { 
  auto connection = open_connection("http://www.example.com/"); 
  send_request(connection, request); 
  close(connection); 
} 
```

正如你所看到的，我们必须记得在使用完连接后关闭它，否则连接将保持打开（泄漏）。在这个例子中，似乎很难忘记，但一旦代码在插入适当的错误处理和多个退出路径后变得更加复杂，就很难保证连接总是关闭。RAII 通过依赖自动变量的生命周期以可预测的方式处理这个问题。我们需要的是一个对象，它的生命周期与我们从`open_connection()`调用中获得的连接相同。我们可以为此创建一个名为`RAIIConnection`的类：

```cpp
class RAIIConnection { 
public: 
  explicit RAIIConnection(const std::string& url) 
      : connection_{open_connection(url)} {} 
  ~RAIIConnection() { 
    try { 
      close(connection_);       
    } 
    catch (const std::exception&) { 
      // Handle error, but never throw from a destructor 
    } 
  }
  auto& get() { return connection_; } 

private:  
  Connection connection_; 
}; 
```

`Connection`对象现在被包装在一个控制连接（资源）生命周期的类中。现在我们可以让`RAIIConnection`来处理关闭连接，而不是手动关闭连接：

```cpp
auto send_request(const std::string& request) { 
  auto connection = RAIIConnection("http://www.example.com/"); 
  send_request(connection.get(), request); 
  // No need to close the connection, it is automatically handled 
  // by the RAIIConnection destructor 
} 
```

RAII 使我们的代码更安全。即使`send_request()`在这里抛出异常，连接对象仍然会被销毁并关闭连接。我们可以将 RAII 用于许多类型的资源，不仅仅是内存、文件句柄和连接。另一个例子是来自 C++标准库的`std::scoped_lock`。它在创建时尝试获取锁（互斥锁），然后在销毁时释放锁。您可以在*第十一章* *并发*中了解更多关于`std::scoped_lock`的信息。

现在，我们将探索更多使内存所有权在 C++中变得明确的方法。

## 容器

您可以使用标准容器来处理对象的集合。您使用的容器将拥有存储在其中的对象所需的动态内存。这是一种在代码中最小化手动`new`和`delete`表达式的非常有效的方法。

还可以使用`std::optional`来处理可能存在或可能不存在的对象的生命周期。`std::optional`可以被视为一个最大大小为 1 的容器。

我们不会在这里再讨论容器，因为它们已经在*第四章* *数据结构*中涵盖过了。

## 智能指针

标准库中的智能指针包装了一个原始指针，并明确了对象的所有权。当正确使用时，没有疑问谁负责删除动态对象。三种智能指针类型是：`std::unique_ptr`、`std::shared_ptr`和`std::weak_ptr`。正如它们的名称所暗示的那样，它们代表对象的三种所有权类型：

+   独占所有权表示我，只有我，拥有这个对象。当我使用完它后，我会删除它。

+   共享所有权表示我和其他人共同拥有对象。当没有人再需要这个对象时，它将被删除。

+   弱所有权表示如果对象存在，我会使用它，但不会仅仅为了我而保持它的生存。

我们将分别在以下各节中处理这些类型。

### 独占指针

最安全和最不复杂的所有权是独占所有权，当考虑智能指针时，应该首先想到的是独占所有权。独占指针表示独占所有权；也就是说，一个资源只被一个实体拥有。独占所有权可以转移给其他人，但不能被复制，因为那样会破坏其独特性。以下是如何使用`std::unique_ptr`：

```cpp
auto owner = std::make_unique<User>("John");
auto new_owner = std::move(owner); // Transfer ownership 
```

独占指针也非常高效，因为与普通原始指针相比，它们几乎没有性能开销。轻微的开销是由于`std::unique_ptr`具有非平凡的析构函数，这意味着（与原始指针不同）在传递给函数时无法将其传递到 CPU 寄存器中。这使它们比原始指针慢。

### 共享指针

共享所有权意味着一个对象可以有多个所有者。当最后一个所有者不存在时，对象将被删除。这是一种非常有用的指针类型，但也比独占指针更复杂。

`std::shared_ptr`对象使用引用计数来跟踪对象的所有者数量。当计数器达到 0 时，对象将被删除。计数器需要存储在某个地方，因此与独占指针相比，它确实具有一些内存开销。此外，`std::shared_ptr`在内部是线程安全的，因此需要原子方式更新计数器以防止竞争条件。

创建由共享指针拥有的对象的推荐方式是使用`std::make_shared<T>()`。这既更安全（从异常安全性的角度来看），也比手动使用`new`创建对象，然后将其传递给`std::shared_ptr`构造函数更有效。通过再次重载`operator new()`和`operator delete()`来跟踪分配，我们可以进行实验，找出为什么使用`std::make_shared<T>()`更有效：

```cpp
auto operator new(size_t size) -> void* { 
  void* p = std::malloc(size); 
  std::cout << "allocated " << size << " byte(s)" << '\n'; 
  return p; 
} 
auto operator delete(void* p) noexcept -> void { 
  std::cout << "deleted memory\n"; 
  return std::free(p); 
} 
```

现在，让我们首先尝试推荐的方式，使用`std::make_shared()`：

```cpp
int main() { 
  auto i = std::make_shared<double>(42.0); 
  return 0; 
} 
```

运行程序时的输出如下：

```cpp
allocated 32 bytes 
deleted memory 
```

现在，让我们通过使用`new`显式分配`int`值，然后将其传递给`std::shared_ptr`构造函数：

```cpp
int main() { 
  auto i = std::shared_ptr<double>{new double{42.0}}; 
  return 0; 
} 
```

程序将生成以下输出：

```cpp
allocated 4 bytes 
allocated 32 bytes 
deleted memory 
deleted memory 
```

我们可以得出结论，第二个版本需要两次分配，一次是为`double`，一次是为`std::shared_ptr`，而第一个版本只需要一次分配。这也意味着，通过使用`std::make_shared()`，我们的代码将更加友好地利用缓存，因为具有空间局部性。

### 弱指针

弱所有权不会保持任何对象存活；它只允许我们在其他人拥有对象时使用对象。为什么要使用这种模糊的弱所有权？使用弱指针的一个常见原因是打破引用循环。引用循环发生在两个或多个对象使用共享指针相互引用时。即使所有外部`std::shared_ptr`构造函数都消失了，对象仍然通过相互引用而保持存活。

为什么不只使用原始指针？弱指针难道不就是原始指针已经是的东西吗？一点也不是。弱指针是安全的，因为除非对象实际存在，否则我们无法引用该对象，而悬空的原始指针并非如此。一个例子将澄清这一点：

```cpp
auto i = std::make_shared<int>(10); 
auto weak_i = std::weak_ptr<int>{i};

// Maybe i.reset() happens here so that the int is deleted... 
if (auto shared_i = weak_i.lock()) { 
  // We managed to convert our weak pointer to a shared pointer 
  std::cout << *shared_i << '\n'; 
} 
else { 
  std::cout << "weak_i has expired, shared_ptr was nullptr\n"; 
} 
```

每当我们尝试使用弱指针时，我们需要首先使用成员函数`lock()`将其转换为共享指针。如果对象尚未过期，共享指针将是指向该对象的有效指针；否则，我们将得到一个空的`std::shared_ptr`。这样，我们可以避免在使用`std::weak_ptr`时出现悬空指针，而不是使用原始指针。

这将结束我们关于内存中对象的部分。C++在处理内存方面提供了出色的支持，无论是关于低级概念，如对齐和填充，还是高级概念，如对象所有权。

对所有权、RAII 和引用计数有着清晰的理解在使用 C++时非常重要。对于新手来说，如果之前没有接触过这些概念，可能需要一些时间才能完全掌握。与此同时，这些概念并不是 C++独有的。在大多数语言中，它们更加普遍，但在其他一些语言中，它们甚至更加突出（Rust 就是后者的一个例子）。因此，一旦掌握，它将提高您在其他语言中的编程技能。思考对象所有权将对您编写的程序的设计和架构产生积极影响。

现在，我们将继续介绍一种优化技术，它将减少动态内存分配的使用，并在可能的情况下使用堆栈。

# 小对象优化

像`std::vector`这样的容器的一个很大的优点是，它们在需要时会自动分配动态内存。然而，有时为只包含少量小元素的容器对象使用动态内存会影响性能。将元素保留在容器本身，并且只使用堆栈内存，而不是在堆上分配小的内存区域，会更有效率。大多数现代的`std::string`实现都会利用这样一个事实：在正常程序中，很多字符串都很短，而且短字符串在不使用堆内存的情况下更有效率。

一种选择是在字符串类本身中保留一个小的单独缓冲区，当字符串的内容很短时可以使用。即使短缓冲区没有被使用，这也会增加字符串类的大小。

因此，一个更节省内存的解决方案是使用一个联合，当字符串处于短模式时可以容纳一个短缓冲区，否则，它将容纳它需要处理动态分配缓冲区的数据成员。用于优化处理小数据的容器的技术通常被称为字符串的小字符串优化，或者其他类型的小对象优化和小缓冲区优化。我们对我们喜欢的事物有很多名称。

一个简短的代码示例将演示在我的 64 位系统上，来自 LLVM 的 libc++中的`std::string`的行为：

```cpp
auto allocated = size_t{0}; 
// Overload operator new and delete to track allocations 
void* operator new(size_t size) {  
  void* p = std::malloc(size); 
  allocated += size; 
  return p; 
} 

void operator delete(void* p) noexcept { 
  return std::free(p); 
} 

int main() { 
  allocated = 0; 
  auto s = std::string{""}; // Elaborate with different string sizes 

  std::cout << "stack space = " << sizeof(s) 
    << ", heap space = " << allocated 
    << ", capacity = " << s.capacity() << '\n'; 
} 
```

代码首先通过重载全局的`operator new`和`operator delete`来跟踪动态内存分配。现在我们可以开始测试不同大小的字符串`s`，看看`std::string`的行为。在我的系统上以发布模式构建和运行前面的示例时，它生成了以下输出：

```cpp
stack space = 24, heap space = 0, capacity = 22 
```

这个输出告诉我们，`std::string`在堆栈上占用 24 个字节，并且在不使用任何堆内存的情况下，它的容量为 22 个字符。让我们通过用一个包含 22 个字符的字符串来验证这一点：

```cpp
auto s = std::string{"1234567890123456789012"}; 
```

程序仍然产生相同的输出，并验证没有分配动态内存。但是当我们增加字符串以容纳 23 个字符时会发生什么呢？

```cpp
auto s = std::string{"12345678901234567890123"}; 
```

现在运行程序会产生以下输出：

```cpp
stack space = 24, heap space = 32, capacity = 31 
```

`std::string`类现在被强制使用堆来存储字符串。它分配了 32 个字节，并报告容量为 31。这是因为 libc++总是在内部存储一个以空字符结尾的字符串，因此需要在末尾额外的一个字节来存储空字符。令人惊讶的是，字符串类可以只占用 24 个字节，并且可以容纳长度为 22 个字符的字符串而不分配任何内存。它是如何做到的呢？如前所述，通常通过使用具有两种不同布局的联合来节省内存：一种用于短模式，一种用于长模式。在真正的 libc++实现中有很多巧妙之处，以充分利用可用的 24 个字节。这里的代码是为了演示这个概念而简化的。长模式的布局如下：

```cpp
struct Long { 
  size_t capacity_{}; 
  size_t size_{}; 
  char* data_{}; 
}; 
```

长布局中的每个成员占用 8 个字节，因此总大小为 24 个字节。`data_`指针是指向将容纳长字符串的动态分配内存的指针。短模式的布局看起来像这样：

```cpp
struct Short { 
  unsigned char size_{};
  char data_[23]{}; 
}; 
```

在短模式下，不需要使用一个变量来存储容量，因为它是一个编译时常量。在这种布局中，`size_`数据成员也可以使用更小的类型，因为我们知道如果是短字符串，字符串的长度只能在 0 到 22 之间。

这两种布局使用一个联合结合起来：

```cpp
union u_ { 
  Short short_layout_; 
  Long long_layout_; 
}; 
```

然而，还有一个缺失的部分：字符串类如何知道它当前是存储短字符串还是长字符串？需要一个标志来指示这一点，但它存储在哪里？事实证明，libc++在长模式下使用`capacity_`数据成员的最低有效位，而在短模式下使用`size_`数据成员的最低有效位。对于长模式，这个位是多余的，因为字符串总是分配 2 的倍数的内存大小。在短模式下，可以只使用 7 位来存储大小，以便一个位可以用于标志。当编写此代码以处理大端字节顺序时，情况变得更加复杂，因为无论我们使用联合的短结构还是长结构，位都需要放置在内存的相同位置。您可以在[`github.com/llvm/llvm-project/tree/master/libcxx`](https://github.com/llvm/llvm-project/tree/master/libcxx)上查看 libc++实现的详细信息。

*图 7.9*总结了我们简化的（但仍然相当复杂）内存布局，该布局由高效实现小字符串优化的联合使用：

![](img/B15619_07_09.png)

图 7.9：用于处理短字符串和长字符串的两种不同布局的并集

像这样的巧妙技巧是您应该在尝试自己编写之前，努力使用标准库提供的高效且经过充分测试的类的原因。然而，了解这些优化以及它们的工作原理是重要且有用的，即使您永远不需要自己编写一个。

# 自定义内存管理

在本章中，我们已经走了很长的路。我们已经介绍了虚拟内存、堆栈和堆、`new`和`delete`表达式、内存所有权以及对齐和填充的基础知识。但在结束本章之前，我们将展示如何在 C++中自定义内存管理。我们将看到，在编写自定义内存分配器时，本章前面介绍的部分将会派上用场。

但首先，什么是自定义内存管理器，为什么我们需要它？

使用`new`或`malloc()`来分配内存时，我们使用 C++中的内置内存管理系统。大多数`operator new`的实现都使用`malloc()`，这是一个通用的内存分配器。设计和构建通用内存管理器是一项复杂的任务，已经有许多人花了很多时间研究这个主题。然而，有几个原因可能会导致您想要编写自定义内存管理器。以下是一些例子：

+   **调试和诊断**：在本章中，我们已经几次通过重载`operator new`和`operator delete`来打印一些调试信息。

+   **沙盒**：自定义内存管理器可以为不允许分配不受限制内存的代码提供一个沙盒。沙盒还可以跟踪内存分配，并在沙盒代码执行完毕时释放内存。

+   **性能**：如果我们需要动态内存并且无法避免分配，可能需要编写一个针对特定需求性能更好的自定义内存管理器。稍后，我们将介绍一些情况，我们可以利用它们来超越`malloc()`。

尽管如此，许多有经验的 C++程序员从未遇到过实际需要定制系统提供的标准内存管理器的问题。这表明了通用内存管理器实际上有多么好，尽管它们必须在不了解我们的具体用例的情况下满足所有要求。我们对应用程序中的内存使用模式了解得越多，我们就越有可能编写比`malloc()`更有效的东西。例如，记得堆栈吗？与堆相比，从堆栈分配和释放内存非常快，这要归功于它不需要处理多个线程，而且释放总是保证以相反的顺序发生。

构建自定义内存管理器通常始于分析确切的内存使用模式，然后实现一个竞技场。

## 建立一个竞技场

在使用内存分配器时经常使用的两个术语是**竞技场**和**内存池**。在本书中，我们不会区分这些术语。我所说的竞技场是指一块连续的内存，包括分配和稍后回收该内存的策略。

竞技场在技术上也可以被称为*内存资源*或*分配器*，但这些术语将用于指代标准库中的抽象。我们稍后将开发的自定义分配器将使用我们在这里创建的竞技场。

在设计一个竞技场时，有一些通用策略可以使分配和释放内存的性能优于`malloc()`和`free()`：

+   单线程：如果我们知道一个竞技场只会从一个线程使用，就不需要用同步原语（如锁或原子操作）保护数据。客户端使用竞技场不会被其他线程阻塞的风险，这在实时环境中很重要。

+   固定大小的分配：如果竞技场只分配固定大小的内存块，那么使用自由列表可以相对容易地高效地回收内存，避免内存碎片化。

+   有限的生命周期：如果你知道从竞技场分配的对象只需要在有限且明确定义的生命周期内存在，竞技场可以推迟回收并一次性释放所有内存。一个例子可能是在服务器应用程序中处理请求时创建的对象。当请求完成时，可以一次性回收在请求期间分配的所有内存。当然，竞技场需要足够大，以便在不断回收内存的情况下处理请求期间的所有分配；否则，这种策略将不起作用。

我不会详细介绍这些策略，但在寻找改进程序中的内存管理方法时，了解可能性是很好的。与优化软件一样，关键是了解程序运行的环境，并分析特定的内存使用模式。我们这样做是为了找到比通用内存管理器更有效的自定义内存管理器的方法。

接下来，我们将看一个简单的竞技场类模板，它可以用于需要动态存储期的小型或少量对象，但它通常需要的内存量很小，可以放在堆栈上。这段代码基于 Howard Hinnant 的`short_alloc`，发布在[`howardhinnant.github.io/stack_alloc.html`](https://howardhinnant.github.io/stack_alloc.html)。如果你想深入了解自定义内存管理，这是一个很好的起点。我认为这是一个很好的示例，因为它可以处理需要正确对齐的多种大小的对象。

但是，请记住，这只是一个简化版本，用于演示概念，而不是为您提供生产就绪的代码：

```cpp
template <size_t N> 
class Arena { 
  static constexpr size_t alignment = alignof(std::max_align_t); 
public: 
  Arena() noexcept : ptr_(buffer_) {} 
  Arena(const Arena&) = delete; 
  Arena& operator=(const Arena&) = delete; 

  auto reset() noexcept { ptr_ = buffer_; } 
  static constexpr auto size() noexcept { return N; } 
  auto used() const noexcept {
    return static_cast<size_t>(ptr_ - buffer_); 
  } 
  auto allocate(size_t n) -> std::byte*; 
  auto deallocate(std::byte* p, size_t n) noexcept -> void; 

private: 
  static auto align_up(size_t n) noexcept -> size_t { 
    return (n + (alignment-1)) & ~(alignment-1); 
  } 
  auto pointer_in_buffer(const std::byte* p) const noexcept -> bool {
    return std::uintptr_t(p) >= std::uintptr_t(buffer_) &&
           std::uintptr_t(p) < std::uintptr_t(buffer_) + N;
  } 
  alignas(alignment) std::byte buffer_[N]; 
  std::byte* ptr_{}; 
}; 
```

区域包含一个`std::byte`缓冲区，其大小在编译时确定。这使得可以在堆栈上或作为具有静态或线程局部存储期的变量创建区域对象。对于除`char`之外的类型，对齐可能在堆栈上分配；因此，除非我们对数组应用`alignas`说明符，否则不能保证它对齐。如果你不习惯位操作，辅助函数`align_up()`可能看起来很复杂。然而，它基本上只是将其舍入到我们使用的对齐要求。这个版本分配的内存将与使用`malloc()`时一样，适用于任何类型。如果我们使用区域来处理具有较小对齐要求的小类型，这会有点浪费，但我们在这里忽略这一点。

在回收内存时，我们需要知道被要求回收的指针是否实际属于我们的区域。`pointer_in_buffer()`函数通过比较指针地址与区域的地址范围来检查这一点。顺便说一句，对不相交对象的原始指针进行关系比较是未定义行为；这可能被优化编译器使用，并导致意想不到的效果。为了避免这种情况，我们在比较地址之前将指针转换为`std::uintptr_t`。如果你对此背后的细节感兴趣，你可以在 Raymond Chen 的文章*如何检查指针是否在内存范围内*中找到详细的解释，链接为[`devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095`](https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095)。

接下来，我们需要实现分配和释放：

```cpp
template<size_t N> 
auto Arena<N>::allocate(size_t n) -> std::byte* { 
  const auto aligned_n = align_up(n); 
  const auto available_bytes =  
    static_cast<decltype(aligned_n)>(buffer_ + N - ptr_); 
  if (available_bytes >= aligned_n) { 
    auto* r = ptr_; 
    ptr_ += aligned_n; 
    return r; 
  } 
  return static_cast<std::byte*>(::operator new(n)); 
} 
```

`allocate()`函数返回一个指向指定大小`n`的正确对齐内存的指针。如果缓冲区中没有足够的空间来满足请求的大小，它将退而使用`operator new`。

以下的`deallocate()`函数首先检查要释放内存的指针是否来自缓冲区，或者是使用`operator new`分配的。如果不是来自缓冲区，我们就简单地使用`operator delete`删除它。否则，我们检查要释放的内存是否是我们从缓冲区分配的最后一块内存，然后通过移动当前的`ptr_`来回收它，就像栈一样。我们简单地忽略其他尝试回收内存的情况：

```cpp
template<size_t N> 
auto Arena<N>::deallocate(std::byte* p, size_t n) noexcept -> void { 
  if (pointer_in_buffer(p)) { 
    n = align_up(n); 
    if (p + n == ptr_) { 
      ptr_ = p; 
    } 
  } 
  else { 
    ::operator delete(p);
  }
} 
```

就是这样；我们的区域现在可以使用了。让我们在分配`User`对象时使用它：

```cpp
auto user_arena = Arena<1024>{}; 

class User { 
public: 
  auto operator new(size_t size) -> void* { 
    return user_arena.allocate(size); 
  } 
  auto operator delete(void* p) -> void { 
    user_arena.deallocate(static_cast<std::byte*>(p), sizeof(User)); 
  } 
  auto operator new[](size_t size) -> void* { 
    return user_arena.allocate(size); 
  } 
  auto operator delete[](void* p, size_t size) -> void { 
    user_arena.deallocate(static_cast<std::byte*>(p), size); 
  } 
private:
  int id_{};
}; 

int main() { 
  // No dynamic memory is allocated when we create the users 
  auto user1 = new User{}; 
  delete user1; 

  auto users = new User[10]; 
  delete [] users; 

  auto user2 = std::make_unique<User>(); 
  return 0; 
} 
```

在这个例子中创建的`User`对象都将驻留在`user_area`对象的缓冲区中。也就是说，当我们在这里调用`new`或`make_unique()`时，不会分配动态内存。但是在 C++中有其他创建`User`对象的方式，这个例子没有展示。我们将在下一节中介绍它们。

## 自定义内存分配器

当尝试使用特定类型的自定义内存管理器时，效果很好！但是有一个问题。事实证明，类特定的`operator new`并没有在我们可能期望的所有场合被调用。考虑以下代码：

```cpp
auto user = std::make_shared<User>(); 
```

当我们想要有一个包含 10 个用户的`std::vector`时会发生什么？

```cpp
auto users = std::vector<User>{};
users.reserve(10); 
```

在这两种情况下都没有使用我们的自定义内存管理器。为什么？从共享指针开始，我们必须回到之前的例子，我们在那里看到`std::make_shared()`实际上为引用计数数据和应该指向的对象分配内存。`std::make_shared()`无法使用诸如`new User()`这样的表达式来创建用户对象和只进行一次分配的计数器。相反，它分配内存并使用就地 new 构造用户对象。

`std::vector`对象也是类似的。当我们调用`reserve()`时，默认情况下它不会在数组中构造 10 个对象。这将需要所有类都有默认构造函数才能与向量一起使用。相反，它分配内存，可以用于添加 10 个用户对象时使用。再次，放置 new 是使这成为可能的工具。

幸运的是，我们可以为`std::vector`和`std::shared_ptr`提供自定义内存分配器，以便它们使用我们的自定义内存管理器。标准库中的其他容器也是如此。如果我们不提供自定义分配器，容器将使用默认的`std::allocator<T>`类。因此，为了使用我们的内存池，我们需要编写一个可以被容器使用的分配器。

自定义分配器在 C++社区中长期以来一直是一个备受争议的话题。许多自定义容器已经被实现，用于控制内存的管理，而不是使用具有自定义分配器的标准容器，这可能是有充分理由的。

然而，在 C++11 中，编写自定义分配器的支持和要求得到了改进，现在要好得多。在这里，我们将只关注 C++11 及以后的分配器。

C++11 中的最小分配器现在看起来是这样的：

```cpp
template<typename T> 
struct Alloc {  
  using value_type = T; 
  Alloc(); 
  template<typename U> Alloc(const Alloc<U>&); 
  T* allocate(size_t n); 
  auto deallocate(T*, size_t) const noexcept -> void; 
}; 
template<typename T> 
auto operator==(const Alloc<T>&, const Alloc<T>&) -> bool;   
template<typename T> 
auto operator!=(const Alloc<T>&, const Alloc<T>&) -> bool; 
```

由于 C++11 的改进，现在代码量确实不那么多了。使用分配器的容器实际上使用了`std::allocator_traits`，它提供了合理的默认值，如果分配器省略了它们。我建议您查看`std::allocator_traits`，看看可以配置哪些特性以及默认值是什么。

通过使用`malloc()`和`free()`，我们可以相当容易地实现一个最小的自定义分配器。在这里，我们将展示老式而著名的`Mallocator`，首次由 Stephan T. Lavavej 在博客文章中发布，以演示如何使用`malloc()`和`free()`编写一个最小的自定义分配器。自那时以来，它已经更新为 C++11，使其更加精简。它是这样的：

```cpp
template <class T>  
struct Mallocator { 

  using value_type = T; 
  Mallocator() = default;

  template <class U>  
  Mallocator(const Mallocator<U>&) noexcept {} 

  template <class U>  
  auto operator==(const Mallocator<U>&) const noexcept {  
    return true;  
  } 

  template <class U>  
  auto operator!=(const Mallocator<U>&) const noexcept {  
    return false;  
  } 

  auto allocate(size_t n) const -> T* { 
    if (n == 0) {  
      return nullptr;  
    } 
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) { 
      throw std::bad_array_new_length{}; 
    } 
    void* const pv = malloc(n * sizeof(T)); 
    if (pv == nullptr) {  
      throw std::bad_alloc{};  
    } 
    return static_cast<T*>(pv); 
  } 
  auto deallocate(T* p, size_t) const noexcept -> void { 
    free(p); 
  } 
}; 
```

`Mallocator`是一个**无状态的分配器**，这意味着分配器实例本身没有任何可变状态；相反，它使用全局函数进行分配和释放，即`malloc()`和`free()`。无状态的分配器应该始终与相同类型的分配器相等。这表明使用`Mallocator`分配的内存也应该使用`Mallocator`释放，而不管`Mallocator`实例如何。无状态的分配器是最简单的分配器，但也是有限的，因为它依赖于全局状态。

为了将我们的内存池作为一个栈分配的对象使用，我们将需要一个**有状态的分配器**，它可以引用内存池实例。在这里，我们实现的内存池类真正开始变得有意义。比如，假设我们想在一个函数中使用标准容器进行一些处理。我们知道，大多数情况下，我们处理的数据量非常小，可以放在栈上。但一旦我们使用标准库中的容器，它们将从堆中分配内存，这在这种情况下会影响我们的性能。

使用栈来管理数据并避免不必要的堆分配的替代方案是什么？一个替代方案是构建一个自定义容器，它使用了我们为`std::string`所研究的小对象优化的变体。

也可以使用 Boost 中的容器，比如`boost::container::small_vector`，它基于 LLVM 的小向量。如果您还没有使用过，我们建议您查看：[`www.boost.org/doc/libs/1_74_0/doc/html/container/non_standard_containers.html`](http://www.boost.org/doc/libs/1_74_0/doc/html/container/non_standard_containers.html)。

然而，另一种选择是使用自定义分配器，我们将在下面探讨。由于我们已经准备好了一个竞技场模板类，我们可以简单地在堆栈上创建一个竞技场实例，并让自定义分配器使用它进行分配。然后我们需要实现一个有状态的分配器，它可以持有对堆栈分配的竞技场对象的引用。

再次强调，我们将实现的这个自定义分配器是 Howard Hinnant 的`short_alloc`的简化版本：

```cpp
template <class T, size_t N> 
struct ShortAlloc { 

  using value_type = T; 
  using arena_type = Arena<N>; 

  ShortAlloc(const ShortAlloc&) = default; 
  ShortAlloc& operator=(const ShortAlloc&) = default; 

  ShortAlloc(arena_type& arena) noexcept : arena_{&arena} { }

  template <class U>
  ShortAlloc(const ShortAlloc<U, N>& other) noexcept
      : arena_{other.arena_} {}

  template <class U> struct rebind {
    using other = ShortAlloc<U, N>;
  };
  auto allocate(size_t n) -> T* {
    return reinterpret_cast<T*>(arena_->allocate(n*sizeof(T)));
  }
  auto deallocate(T* p, size_t n) noexcept -> void {
    arena_->deallocate(reinterpret_cast<std::byte*>(p), n*sizeof(T));
  }
  template <class U, size_t M>
  auto operator==(const ShortAlloc<U, M>& other) const noexcept {
    return N == M && arena_ == other.arena_;
  }
  template <class U, size_t M>
  auto operator!=(const ShortAlloc<U, M>& other) const noexcept {
    return !(*this == other);
  }
  template <class U, size_t M> friend struct ShortAlloc;
private:
  arena_type* arena_;
}; 
```

分配器持有对竞技场的引用。这是分配器唯一的状态。函数`allocate()`和`deallocate()`只是将它们的请求转发到竞技场。比较运算符确保`ShortAlloc`类型的两个实例使用相同的竞技场。

现在，我们实现的分配器和竞技场可以与标准容器一起使用，以避免动态内存分配。当我们使用小数据时，我们可以使用堆栈处理所有分配。让我们看一个使用`std::set`的例子：

```cpp
int main() { 

  using SmallSet =  
    std::set<int, std::less<int>, ShortAlloc<int, 512>>; 

  auto stack_arena = SmallSet::allocator_type::arena_type{}; 
  auto unique_numbers = SmallSet{stack_arena}; 

  // Read numbers from stdin 
  auto n = int{}; 
  while (std::cin >> n)
    unique_numbers.insert(n); 

  // Print unique numbers  
  for (const auto& number : unique_numbers)
    std::cout << number << '\n'; 
} 
```

该程序从标准输入读取整数，直到达到文件结尾（在类 Unix 系统上为 Ctrl + D，在 Windows 上为 Ctrl + Z）。然后按升序打印唯一的数字。根据从`stdin`读取的数字数量，程序将使用堆栈内存或动态内存，使用我们的`ShortAlloc`分配器。

## 使用多态内存分配器

如果您已经阅读了本章，现在您知道如何实现一个自定义分配器，可以与包括标准库在内的任意容器一起使用。假设我们想要在我们的代码库中使用我们的新分配器来处理`std::vector<int>`类型的缓冲区的一些代码，就像这样：

```cpp
void process(std::vector<int>& buffer) {
  // ...
}
auto some_func() {
  auto vec = std::vector<int>(64);
  process(vec); 
  // ...
} 
```

我们迫不及待地想尝试一下我们的新分配器，它正在利用堆栈内存，并尝试像这样注入它：

```cpp
using MyAlloc = ShortAlloc<int, 512>;  // Our custom allocator
auto some_func() {
  auto arena = MyAlloc::arena_type();
  auto vec = std::vector<int, MyAlloc>(64, arena);
  process(vec);
  // ...
} 
```

在编译时，我们痛苦地意识到`process()`是一个期望`std::vector<int>`的函数，而我们的`vec`变量现在是另一种类型。GCC 给了我们以下错误：

```cpp
error: invalid initialization of reference of type 'const std::vector<int>&' from expression of type 'std::vector<int, ShortAlloc<int, 512> > 
```

类型不匹配的原因是我们想要使用的自定义分配器`MyAlloc`作为模板参数传递给`std::vector`，因此成为我们实例化的类型的一部分。因此，`std::vector<int>`和`std::vector<int, MyAlloc>`不能互换。

这可能对您正在处理的用例有影响，您可以通过使`process()`函数接受`std::span`或使其成为使用范围而不是要求`std::vector`的通用函数来解决这个问题。无论如何，重要的是要意识到，当使用标准库中的支持分配器的模板类时，分配器实际上成为类型的一部分。

`std::vector<int>`使用的是什么分配器？答案是`std::vector<int>`使用默认模板参数`std::allocator`。因此，编写`std::vector<int>`等同于`std::vector<int, std::allocator<int>>`。模板类`std::allocator`是一个空类，当它满足容器的分配和释放请求时，它使用全局`new`和全局`delete`。这也意味着使用空分配器的容器的大小比使用自定义分配器的容器要小：

```cpp
std::cout << sizeof(std::vector<int>) << '\n';
// Possible output: 24
std::cout << sizeof(std::vector<int, MyAlloc>) << '\n';
// Possible output: 32 
```

检查来自 libc++的`std::vector`的实现，我们可以看到它使用了一个称为**compressed pair**的巧妙类型，这又基于*空基类优化*来摆脱通常由空类成员占用的不必要存储空间。我们不会在这里详细介绍，但如果您感兴趣，可以查看`compressed_pair`的 boost 版本，该版本在[`www.boost.org/doc/libs/1_74_0/libs/utility/doc/html/compressed_pair.html`](https://www.boost.org/doc/libs/1_74_0/libs/utility/doc/html/compressed_pair.html)中有文档。

在 C++17 中，使用不同的分配器时出现了不同类型的问题，通过引入额外的间接层来解决；在`std::pmr`命名空间下的所有标准容器都使用相同的分配器，即`std::pmr::polymorphic_allocator`，它将所有分配/释放请求分派给一个**内存资源**类。因此，我们可以使用通用的多态内存分配器`std::pmr::polymorphic_allocator`，而不是编写新的自定义内存分配器，并在构造过程中使用新的自定义内存资源。内存资源类似于我们的`Arena`类，而`polymorphic_allocator`是额外的间接层，其中包含指向资源的指针。

以下图表显示了向量委托给其分配器实例，然后分配器再委托给其指向的内存资源的控制流程。

![](img/B15619_07_10.png)

图 7.10：使用多态分配器分配内存

要开始使用多态分配器，我们需要将命名空间从`std`更改为`std::pmr`：

```cpp
auto v1 = std::vector<int>{};             // Uses std::allocator
auto v2 = std::pmr::vector<int>{/*...*/}; // Uses polymorphic_allocator 
```

编写自定义内存资源相对比较简单，特别是对于了解内存分配器和区域的知识。但为了实现我们想要的功能，我们甚至可能不需要编写自定义内存资源。C++已经为我们提供了一些有用的实现，在编写自己的实现之前，我们应该考虑一下。所有内存资源都派生自基类`std::pmr::memory_resource`。以下内存资源位于`<memory_resource>`头文件中：

+   `std::pmr::monotonic_buffer_resource`: 这与我们的`Arena`类非常相似。在我们创建许多寿命短的对象时，这个类是首选。只有在`monotonic_buffer_resource`实例被销毁时，内存才会被释放，这使得分配非常快。

+   `std::pmr::unsynchronized_pool_resource`: 这使用包含固定大小内存块的内存池（也称为“slabs”），避免了每个池内的碎片。每个池为特定大小的对象分配内存。如果您正在创建多个不同大小的对象，这个类可以很有益。这个内存资源不是线程安全的，除非提供外部同步，否则不能从多个线程使用。

+   `std::pmr::synchronized_pool_resource`: 这是`unsynchronized_pool_resource`的线程安全版本。

内存资源可以被链接。在创建内存资源的实例时，我们可以为其提供一个**上游内存资源**。如果当前资源无法处理请求（类似于我们在`ShortAlloc`中使用`malloc()`一旦我们的小缓冲区已满），或者当资源本身需要分配内存时（例如当`monotonic_buffer_resource`需要分配其下一个缓冲区时），将使用此上游资源。`<memory_resource>`头文件为我们提供了一些自由函数，返回指向全局资源对象的指针，这些在指定上游资源时非常有用：

+   `std::pmr::new_delete_resource()`: 使用全局的`operator new`和`operator delete`。

+   `std::pmr::null_memory_resource()`: 一个资源，每当被要求分配内存时总是抛出`std::bad_alloc`。

+   `std::pmr::get_default_resource()`: 返回一个全局默认的内存资源，可以在运行时通过`set_default_resource()`进行设置。初始默认资源是`new_delete_resource()`。

让我们看看如何重新编写上一节中的示例，但这次使用`std::pmr::set`：

```cpp
int main() {
  auto buffer = std::array<std::byte, 512>{};
  auto resource = std::pmr::monotonic_buffer_resource{
    buffer.data(), buffer.size(), std::pmr::new_delete_resource()};
  auto unique_numbers = std::pmr::set<int>{&resource};
  auto n = int{};
  while (std::cin >> n) {
    unique_numbers.insert(n);
  }
  for (const auto& number : unique_numbers) {
    std::cout << number << '\n';
  }
} 
```

我们将一个栈分配的缓冲区传递给内存资源，然后为其提供从`new_delete_resource()`返回的对象作为上游资源，以便在缓冲区变满时使用。如果我们省略了上游资源，它将使用默认内存资源，在这种情况下，由于我们的代码不会更改默认内存资源，因此默认内存资源将是相同的。

## 实现自定义内存资源

实现自定义内存资源相当简单。我们需要公开继承自`std::pmr::memory_resource`，然后实现三个纯虚函数，这些函数将被基类（`std::pmr::memory_resource`）调用。让我们实现一个简单的内存资源，它打印分配和释放，然后将请求转发到默认内存资源：

```cpp
class PrintingResource : public std::pmr::memory_resource {
public:
  PrintingResource() : res_{std::pmr::get_default_resource()} {}
private:
  void* do_allocate(std::size_t bytes, std::size_t alignment)override {
    std::cout << "allocate: " << bytes << '\n';
    return res_->allocate(bytes, alignment);
  }
  void do_deallocate(void* p, std::size_t bytes,
                     std::size_t alignment) override {
    std::cout << "deallocate: " << bytes << '\n';
    return res_->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) 
    const noexcept override {
    return (this == &other);
  }
  std::pmr::memory_resource* res_;  // Default resource
}; 
```

请注意，我们在构造函数中保存了默认资源，而不是直接从`do_allocate()`和`do_deallocate()`中直接调用`get_default_resource()`。原因是在分配和释放之间的时间内，某人可能通过调用`set_default_resource()`来更改默认资源。

我们可以使用自定义内存资源来跟踪`std::pmr`容器所做的分配。以下是使用`std::pmr::vector`的示例：

```cpp
auto res = PrintingResource{};
auto vec = std::pmr::vector<int>{&res};
vec.emplace_back(1);
vec.emplace_back(2); 
```

运行程序时可能的输出是：

```cpp
allocate: 4
allocate: 8
deallocate: 4
deallocate: 8 
```

在使用多态分配器时需要非常小心的一点是，我们传递的是原始的非拥有指针到内存资源。这不是特定于多态分配器；我们在`Arena`类和`ShortAlloc`中也有同样的问题，但是在使用`std::pmr`容器时可能更容易忘记，因为这些容器使用相同的分配器类型。考虑以下示例：

```cpp
auto create_vec() -> std::pmr::vector<int> {
  auto resource = PrintingResource{};
  auto vec = std::pmr::vector<int>{&resource}; // Raw pointer
  return vec;                                  // Ops! resource
}                                              // destroyed here 
auto vec = create_vec();
vec.emplace_back(1);                           // Undefined behavior 
```

由于资源在`create_vec()`结束时超出范围而被销毁，我们新创建的`std::pmr::vector`是无用的，很可能在使用时崩溃。

这结束了我们关于自定义内存管理的部分。这是一个复杂的主题，如果您想要使用自定义内存分配器来提高性能，我鼓励您在使用和/或实现自定义分配器之前仔细测量和分析应用程序中的内存访问模式。通常，应用程序中只有一小部分类或对象真正需要使用自定义分配器进行调整。同时，在应用程序中减少动态内存分配的数量或将对象组合在一起，可以对性能产生显著影响。

# 总结

本章涵盖了很多内容，从虚拟内存的基础开始，最终实现了可以被标准库中的容器使用的自定义分配器。了解程序如何使用内存是很重要的。过度使用动态内存可能成为性能瓶颈，您可能需要优化掉它。

在开始实现自己的容器或自定义内存分配器之前，请记住，您之前可能有很多人面临过与您可能面临的非常相似的内存问题。因此，很有可能您的正确工具已经存在于某个库中。构建快速、安全和健壮的自定义内存管理器是一个挑战。

在下一章中，您将学习如何从 C++概念中受益，以及如何使用模板元编程让编译器为我们生成代码。
