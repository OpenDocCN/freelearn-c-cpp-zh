# 7

# 重载内存分配运算符

到目前为止，你过得愉快吗？我希望你是！我们现在已经掌握了所有的钥匙，可以开始做这本书所宣传的事情，更详细地看看 C++中内存管理是如何工作的。这不是一个简单的话题，也不是一件微不足道的事情，所以我们需要确保我们已经准备好了……但现在我们已经准备好了，让我们开始吧！

*第五章*和*第六章*探讨了可以使用标准工具将动态分配资源的责任封装到 C++类型系统中的方法，这些工具包括标准提供的以及我们可以编写的以填补其他空白。使用智能指针而不是原始指针作为数据成员和函数返回类型，往往可以简化（并阐明）C++程序中大量内存管理任务。

有时候，我们希望在这个级别以下工作，并控制当有人编写`new X`时会发生什么。想要这种控制的原因有很多，在这本书中我们将探讨其中的一些，但在这章中，我们将专注于内存管理函数的基本知识以及如何在 C++中控制这些机制。

在这些基础知识被覆盖之后，我们将进行以下操作：

+   看看我们对 C++内存分配机制的了解如何让我们在*第八章*中编写一个简单的（但有效的）泄漏检测器

+   在*第九章*中检查如何在 C++中管理典型（持久、共享等）内存

+   在*第十章*中编写基于竞技场的内存分配，以确保确定性的时间分配和释放，当上下文允许时，这将导致`new`和`delete`的快速实现

后续章节将使用本章以及后续章节中获得的知识来编写高效的容器和延迟回收机制，这些机制类似于垃圾回收器。超过这一点，我们将探讨容器如何使用这些设施，包括和不包括分配器的情况。

# 为什么会重载分配函数？

在我们开始讨论如何重载内存分配机制之前，让我们退一步，看看为什么有人想要这样做。确实，大多数程序员（即使是经验丰富的程序员）最终都没有做过这样的事情，我们可以打赌，大多数程序员从未想过他们有理由这样做。然而，我们将分配（！）几个章节来讨论这个话题。肯定有一个原因……

关于内存分配的事情是，在一般情况下，没有完美的解决方案来解决这个问题；平均来说，有许多好的解决方案，对于更专业的问题版本，也有非常好的解决方案。在编程语言 A 中构成良好解决方案的某个特定用例，可能不适合另一个用例或在编程语言 B 中。

以例如 Java 或 C#中动态分配大量小对象为习惯的语言为例。在这样的语言中，人们可以期望分配策略针对这种使用模式进行了优化。在 C 这样的语言中，人们可能会在对象太大而无法放在栈上或使用基于节点的数据结构（例如）时进行分配，最佳的动态内存分配策略可能完全不同。在第*第十章*中，我们将看到一个分配过程从分配的对象都是相同大小和对齐的事实中受益的例子，另一个有趣的用例。

C++强调控制并提供给程序员复杂且多功能的工具。当我们知道分配将在何种上下文中执行时，我们有时可以使用这些工具做得更好（甚至*好得多*，正如我们将在*第十一章*中看到的那样！）并且对于许多指标：更好的执行时间、更确定的执行时间、减少内存碎片等等。

# C 语言分配函数的简要概述

在我们了解 C++的内存分配机制之前，让我们先简要地看一下 C 系列内存分配函数，通过其最杰出的代表：`malloc()`和`free()`。当然，还有许多其他与内存分配相关的函数，如`calloc()`、`realloc()`和`aligned_alloc()`，不计操作系统特定的服务，这些服务为特定的用例执行类似任务，但这些都很好地服务于我们的讨论。

注意，由于这是一本关于 C++内存管理的书，我将使用这些函数的 C++版本（从`<cstdlib>`而不是`<stdlib.h>`），这实际上对我们的代码没有任何影响，除了在 C++中，这些函数位于`std`命名空间的事实。

这两个函数的签名如下：

```cpp
void* malloc(size_t n);
void free(void *p);
```

`malloc(n)`的作用是在至少有`n`个连续字节可用的位置找到位置，可能将该位置标记为“已占用”，并返回指向该内存块开始的抽象指针（`void*`）。请注意，返回的指针必须适合给定机器最坏的自然情况，这意味着它必须满足`std::max_align_t`的对齐要求。在大多数机器上，这种类型是`double`的别名。

有趣的是，调用`malloc()`时`n==0`是合法的，但此类调用的结果由实现定义：对`malloc(0)`的调用可能返回`nullptr`，也可能返回非空指针。请注意，无论指针是否为空，都不应取消引用`malloc(0)`返回的指针。

如果 `malloc()` 无法分配内存，它返回 `nullptr`，因为 C 语言不支持 C++意义上的异常。在当代 C（自 C11 起），`malloc()` 实现必须是线程安全的，并且如果它们被并发调用，包括与 `free()` 一起调用，必须适当地与其他 C 分配函数同步。

`free(p)` 的作用是确保由 `p` 指向的内存变为可用，以便进一步分配请求，只要 `p` 指向的是通过 `malloc()` 等内存分配函数分配的块，并且尚未释放。不要对通过这种分配函数未分配的地址调用 `free()`… 不要这样做！另外，要知道一旦内存被释放，它就不再被视为已分配，因此以下代码会导致未定义行为（UB）：

```cpp
#include <cstdlib>
int main() {
   using std::malloc, std::free;
   int *p = static_cast<int*>(malloc(sizeof(int)));
   free(p); // fine since it comes from malloc()
   free(p); // NOOOOOO unless (stroke of luck?) p is null
}
```

如前例所述，`free(nullptr)` 不会做任何事情，并且自本文写作以来已经定义为不做任何事情几十年了。如果你的代码库中有在调用 `free()` 之前验证 `p!=nullptr` 的代码 – 例如，`if(p) free(p)` – 你可以安全地移除那个测试。

我们有时（不一定总是）会使用这些 C 函数来实现我们自制的 C++分配函数。它们是有效的，它们被很好地理解，并且是我们可以利用来构建高级抽象的低级抽象。 

# C++分配运算符概述

在 C++中，内存分配运算符有许多（无限多！）版本，但在编写自己的版本时必须遵循规则。当前章节主要关于这些规则；接下来的章节将探讨利用 C++赋予我们的这种自由的方法：

+   C++允许我们重载 `new int` 将使用我们自制的版本。在这里必须小心，因为小小的错误可能会对代码执行产生重大影响：如果你的 `operator new()` 实现很慢，你将减慢程序中大多数内存分配的速度！我们将在*第八章*中编写一个简单但有效的内存泄漏检测器时使用这种方法。

+   C++允许我们重载内存分配运算符的**成员函数版本**。如果我们这样做，那么全局版本（重载与否）通常适用，但成员函数版本适用于特定类型。这在我们对某些类型的用法模式有特定知识但不是对其他类型时很有用。我们将在*第十章*中利用这一点。

+   C++允许我们重载 `nothrow` 版本和（极其重要的）**placement new** 相关的版本。我们还可以利用这个特性来利用“奇异”内存，例如共享内存或持久内存，正如我们将在*第九章*中看到的那样。

在每种情况下，内存分配函数都分为四组：`operator new()`、`operator new[]()`、`operator delete()` 和 `operator delete[]()`。虽然有一些例外，但这个规则通常成立。如果我们至少重载这些函数中的一个，那么重载所有四个以保持程序行为一致是很重要的。当与这种低级设施（如本例所示）玩耍时，错误往往会更加严重，这也解释了为什么我们在 *第二章* 和 *第三章* 中如此小心地解释了我们可能会遇到麻烦的方式……以及如何同时遵守规则。

内存分配与对象模型（参见 *第一章* 中的基础知识）和异常安全性（本书中无处不在的主题）密切相关，所以请确保在接下来的页面和章节中掌握这些交互。它们将帮助您充分利用您在这里阅读的内容。

关于堆分配优化（HALO）的一个说明

了解这一点很重要，即不重载内存分配运算符也有好处。其中之一是您的库供应商默认提供了非常好的实现；另一个好处是，如果您不重载内存分配运算符，编译器可以假设您所做的分配数量是不可观察的。这意味着可以替换 *n* 次对 `new` 的调用，用一个一次性分配所有内容的调用，然后像执行了许多分配一样管理结果。这在实践中可能导致一些惊人的优化，包括从生成的代码中完全移除 `new` 和 `delete` 调用，即使它们出现在源代码中！如果有疑问，请确保在将优化提交并用于生产代码之前，它们提供了可衡量的好处。

注意，在本章中我们将看到的分配运算符重载，您需要包含 `<new>` 头文件，因为这是 `std::bad_alloc` 被声明的位置，以及其他一些内容，并且这是分配函数通常用来报告分配失败的类型。

## 全局分配运算符

假设我们想要控制 C++ 中的全局分配运算符版本。为了展示这是如何工作的，我们将简单地使用它们来委托给 `malloc()` 和 `free()`，现在，并在 *第八章* 中展示一个更详细的例子。

如果我们坚持这些运算符的基本形式，我们想要重载……嗯，在 C++11 之前是四个函数，从那时起是六个函数。当然，这本书假设我们已经超过十年没有使用 C++14，所以我们将相应地进行。

我们想要重载的签名如下：

```cpp
void *operator new(std::size_t);
void *operator new[](std::size_t);
void operator delete(void *) noexcept;
void operator delete[](void *) noexcept;
// since C++14
void operator delete(void *, std::size_t) noexcept;
void operator delete[](void *, std::size_t) noexcept;
```

我同意，这确实很多，但掌握内存管理工具是专业的工作。一旦你编写了这些函数之一，你就正式替换了为你提供的标准库中的那些函数，并且该函数将负责通过该渠道传入的分配（或释放）请求。替换分配函数需要你使用与原始函数完全相同的签名。

如果你至少重载了一个函数，那么重载整个函数集之所以重要，是因为这些函数形成了一个一致的整体。例如，如果你改变了`new`的行为方式，但忽略了标准库提供的`delete`执行其任务的方式，那么预测你的程序将遭受多少损害基本上是不可能的。正如一位著名的流行漫画书英雄多次所说的，“*权力越大，责任越大。”*要小心，要严谨，并遵循规则。

注意这些函数的签名，因为它们提供了有趣的信息...

### 关于 new 和 new[]操作符

函数`operator new()`和`operator new[]()`都接受一个`std::size_t`对象作为参数，并且都返回`void*`。在两种情况下，参数都是要分配的最小连续字节数。因此，它们的签名类似于`std::malloc()`。这常常让人惊讶；如果`new`不是一个`模板`并且不知道要创建什么，那么`new X`表达式是如何创建`X`对象的呢？

事情是这样的：`new`并不创建对象。`new`所做的就是找到将要构造对象的位置。是构造函数将`new`找到的原始内存转换成对象。在实践中，你可以编写如下内容：

```cpp
X *p = new X{ /* ... args ... */ };
```

你所写的是一个两步操作：

```cpp
// allocate enough space to put an X object
void * buf = operator new(sizeof(X));
// construct an X object at that location
X *p = ... // apply X::X( /* ... args ... */ ) on buf
```

这意味着构造函数就像是一层涂在内存块上的油漆，将那块内存转换成对象。这也意味着，例如`new X`这样的表达式可能会在`operator new()`失败时失败，如果分配请求无法成功，或者在`X::X()`失败，因为构造函数以某种方式失败了。只有当这两个步骤都成功时，客户端代码才对指向的对象负责。

关于这些操作符的命名

你可能已经注意到在前面的例子中，我们有时写`new X`，有时写`operator new(sizeof(X))`。第一种形式——*操作符形式*——将执行分配后跟构造的两个步骤，而第二种形式——*函数形式*——直接调用分配函数而不调用构造函数。这种区别也适用于`operator delete()`。

与`operator new[]`的情况类似：传递给函数的字节数是数组的总字节数，因此分配函数本身并不知道将要创建的对象的类型、元素的数量或对象的单个大小。实际上，对`new X[N]`的调用将调用`operator new[](N*sizeof(X))`以找到放置将要构造的数组的空间，然后对数组中每个大小为`sizeof(X)`的`N`个块调用`X::X()`。只有当整个序列成功完成时，客户端代码才负责结果数组。

通过`operator new`无法分配标量应该导致抛出与`std::bad_alloc`匹配的东西。对于`operator new[]()`，如果请求的大小有问题，也可以抛出`std::bad_array_new_length`（从`std::bad_alloc`派生），通常是因为它超过了实现定义的限制。

### 关于`delete`和`delete[]`运算符

与 C 语言的`free()`函数类似，`delete()`和`delete[]()`运算符都接受一个`void*`作为参数。这意味着它们不能销毁你的对象…当它们被调用时，对象已经被销毁了！实际上，你可以写出以下内容：

```cpp
delete p; // suppose that p is of type X*
```

这实际上是一个两步操作，相当于以下操作：

```cpp
p->~X(); // destroy the pointed-to object
operator delete(p); // free the associated memory
```

在 C++中，你的析构函数和`operator delete()`都不应该抛出异常。如果它们抛出异常，程序基本上会被终止，原因将在*第十二章*中变得显而易见。

`operator delete()`和`operator delete[]()`的大小感知版本是在 C++14 中引入的，并且现在通常除了这些函数的经典版本之外，还会实现它们。其想法是`operator new()`知道要分配的块的大小，但`operator delete()`不知道，这要求实现方面进行不必要的杂技表演，例如用某个值填充内存块以试图隐藏该位置存储的内容。这些函数的现代实现要求我们编写一个版本，它除了经典版本外还接受指向对象的尺寸；如果实现不需要该尺寸，可以直接从大小感知版本调用经典版本，然后完成。

关于大小感知版本的`operator delete[]()`重载的说明

如果你追踪你重载的执行过程，你可能会惊讶地发现，对于某些类型，`operator delete[]()`的大小版本并不一定被调用。确实，如果你有一个由平凡可销毁类型对象组成的数组`arr`，标准并未指定在编写`delete [] arr`时，将使用`operator delete[]()`的大小版本还是非大小版本。请放心，这并不是一个错误。

这些函数的一个完整但简单的实现是将工作委托给 C 分配函数，如下所示：

```cpp
#include <iostream>
#include <cstdlib>
#include <new>
void *operator new(std::size_t n) {
    std::cout << "operator new(" << n << ")\n";
    auto p = std::malloc(n);
    if(!p) throw std::bad_alloc{};
    return p;
}
void operator delete(void *p) noexcept {
    std::cout << "operator delete(...)\n";
    std::free(p);
}
void operator delete(void *p, std::size_t n) noexcept {
    std::cout << "operator delete(..., " << n << ")\n";
    ::operator delete(p);
}
void *operator new[](std::size_t n) {
    std::cout << "operator new[](" << n << ")\n";
    auto p = std::malloc(n);
    if(!p) throw std::bad_alloc{};
    return p;
}
void operator delete[](void *p) noexcept {
    std::cout << "operator delete[](...)\n";
    std::free(p);
}
void operator delete[](void *p, std::size_t n) noexcept {
    std::cout << "operator delete[](..., " << n << ")\n";
    ::operator delete[](p);
}
int main() {
   auto p = new int{ 3 };
   delete p;
   p = new int[10];
   delete []p;
}
```

如此看来，当 `operator new()` 和 `operator new[]()` 无法满足其后置条件并且实际上分配了请求的内存量时，默认行为是抛出 `std::bad_alloc` 或者在适当的情况下抛出 `std::bad_array_new_length`。由于分配之后是构造，客户端代码也可能面临构造函数抛出的任何异常。我们将在编写自定义容器时探讨如何处理这些情况，见*第十二章*。

在某些应用领域，异常处理不是一个可选项。这可能是由于内存限制；大多数异常处理器会使程序略微增大，这在嵌入式系统等领域的应用中可能是不被接受的。也可能是由于速度限制；`try`块中的代码通常运行得很快，因为这些块代表“正常”的执行路径，但`catch`块中的代码通常被视为罕见的（“异常”）路径，执行速度可能会显著减慢。当然，有些人可能仅仅出于哲学原因而避免使用异常，这也是可以的。

幸运的是，有一种方法可以在不使用异常来指示失败的情况下执行动态内存分配。

## 非抛出异常的分配操作符版本

也有不抛出异常的分配操作符版本。这些函数的签名如下：

```cpp
void *operator new(std::size_t, const std::nothrow_t&);
void *operator new[](std::size_t, const std::nothrow_t&);
void operator delete(void *, const std::nothrow_t&)
   noexcept;
void operator delete[](void *, const std::nothrow_t&)
   noexcept;
// since C++14
void operator delete
   (void *, std::size_t, const std::nothrow_t&) noexcept;
void operator delete[]
   (void *, std::size_t, nullptr than to just write it as if no failure occurred! The fact is that there are costs to using exceptions in one’s programs: it can make binaries slightly bigger, and it can slow down code execution, particularly when exceptions are caught (there are also issues of style involved; some people would not use exceptions even if they led to faster code, and that’s just part of life). For that reason, application domains such as games or embedded systems often shun exceptions and go to some lengths to write code that does not depend on them. The non-throwing versions of the allocation functions target these domains.
Type `std::nothrow_t` is what is called a `std::nothrow` object) can be used to guide the compiler when generating code. Note that these function signatures require the `std::nothrow_t` arguments to be passed by `const` reference, not by value, so make sure you respect this signature if you seek to replace them.
An example usage of these functions would be as follows:

```

X *p = new (nothrow) X{ /* ... args ... */ };

if(p) {

// ... 使用 *p

// note: 这不是 delete 的 nothrow 版本

delete p; // 即使 !p 也会是正确的

}

```cpp

 You might be surprised about the position of `nothrow` in the `new` expression, but if you think about it, it’s essentially the only syntactic space for additional arguments passed to `operator new()`; the first argument passed to the function is the number of contiguous bytes to allocate (here: `sizeof(X)`), and in expression `new X { ...args... }`, what follows the type of object to construct is the list of arguments passed to its constructor. Thus, the place to specify the additional arguments to `operator new()` itself is between `new` and the type of the object to construct, between parentheses.
A word on the position of additional arguments to operator new()
To illustrate this better with an artificially crafted example, one could write the following `operator` `new()` overload:
`void* operator new(std::size_t,` `);`
Then, a possible call to that hypothetical operator would be as follows:
`X *p = new (3, 1.5) X{ /* ... */ };`
Here, we can see how two additional arguments, an `int` argument and a `double` argument, are passed by client code.
Returning to the `nothrow` version of `operator new()` and `operator new[]()`, one thing that is subtle and needs to be understood is why one needs to write overloads of `operator delete()` and `operator delete[]()`. After all, even with client code that uses the `nothrow` version of `new`, as was the case in our example, it’s highly probable that the “normal” version of `operator delete()` will be used to end the life of that object. Why, then, write a `nothrow` version of `operator delete()`?
The reason is `operator new()`? Well, remember that memory allocation through `operator new()` is a two-step operation: find the location to place the object, then construct the object at that location. Thus, even if `operator new()` does not throw, we do not know whether the constructor that will be called will throw. Our code will obtain the pointer only after both the allocation *and* the construction that follows have successfully completed execution; as such, client code cannot manage exceptions that occur after allocation succeeded but during the construction of the object, at least not in such a way as to deallocate the memory… It’s difficult to deallocate a pointer your code has not yet seen!
For that reason, it falls on the C++ runtime to perform the deallocation if an exception is thrown by the constructor, and this is true for all versions of `operator new()`, not just the `nothrow` ones. The algorithm (informally) is as follows:

```

// 第 1 步，尝试为某些 T 对象执行分配

p = operator new(n, ... maybe additional arguments ...)

// 以下行仅用于 nothrow new

if(!p) return p

try {

// 第 2 步，在地址 p 处构造对象

在地址 p 处应用 T 的构造函数 // 可能会抛出

} catch(...) { // 构造函数抛出了异常

deallocate p // 这是我们这里关心的问题

re-throw the exception, whatever it was

}

return p // p 指向一个完全构造的对象

// 只有在这一点之后，客户端代码才会看到 p

```cpp

 As this algorithm shows, the C++ runtime has to deallocate the memory for us when the constructor throws an exception. But how does it do so? Well, it will use the `operator delete()` (or `operator delete[]()`) whose signature matches that of the version of `new` or `new[]` that was used to perform the allocation. For example, if we use `operator new(size_t,``)` to allocate and the constructor fails, it will use `operator delete(void*,``)` to perform the implicit deallocation.
That is the reason why, if we overload the `nothrow` versions of `new` and `new[]`, we have to overload the `nothrow` versions of `delete` and `delete[]` (they will be used for deallocation if a constructor throws), and why we also have to overload the “normal” throwing versions of `new`, `new[]`, `delete`, and `delete[]`. Expressed informally, code that uses `X *p = new(nothrow)X;` will usually call `delete p;` to end the life of the pointee, and as such, the `nothrow` and throwing versions of the allocation functions have to be coherent with one another.
Here is a full, yet naïve implementation where the throwing versions delegate to the non-throwing ones to reduce repetition:

```

#include <iostream>

#include <cstdlib>

#include <new>

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {

return std::malloc(n);

}

void* operator new(std::size_t n) {

auto p = operator new(n, std::nothrow);

if (!p) throw std::bad_alloc{};

return p;

}

void operator delete(void* p, const std::nothrow_t&)

noexcept {

std::free(p);

}

void operator delete(void* p) noexcept {

operator delete(p, std::nothrow);

}

void operator delete(void* p, std::size_t) noexcept {

operator delete (p, std::nothrow);

}

void* operator new[](std::size_t n,

const std::nothrow_t&) noexcept {

return std::malloc(n);

}

void* operator new[](std::size_t n) {

auto p = operator new[](n, std::nothrow);

if (!p) throw std::bad_alloc{};

return p;

}

void operator delete[](void* p, const std::nothrow_t&)

noexcept {

std::free(p);

}

void operator delete[](void* p) noexcept {

operator delete[](p, std::nothrow);

}

void operator delete[](void* p, std::size_t) noexcept {

operator delete[](p, std::nothrow);

}

int main() {

using std::nothrow;

auto p = new (nothrow) int{ 3 };

delete p;

p = new (nothrow) int[10];

delete[]p;

}

```cpp

 As you can see, there are quite a few functions to write to get a full, cohesive set of allocation operators if we want to cover both the throwing and the non-throwing versions of this mechanism.
We still have a lot to cover. For example, we mentioned a few times already the idea of placing an object at a specific memory location, in particular at the second of the two-step process modeled by calls to `new`. Let’s see how this is done.
The most important operator new: placement new
The most important version of `operator new()` and friends is not one you can replace, but even if you could… well, let’s just state that it would be difficult to achieve something more efficient:

```

// note: 这些存在，你可以使用它们，但你不能

// 替换它们

void *operator new(std::size_t, void *p) { return p; }

void *operator new[](std::size_t, void *p) { return p; }

void operator delete(void*, void*) noexcept { }

void operator delete[](void*, void*) noexcept { }

```cpp

 We call these the placement allocation functions, mostly known as **placement new** by the programming community.
What is the purpose of these functions? You might remember, at the beginning of our discussion of the global versions of the allocation operators, that we stated: “What `new` does is find the location where an object will be constructed.” This does not necessarily mean that `new` will allocate memory, and indeed, placement `new` does not allocate; it simply yields back the address it has been given as argument. *This allows us to place an object wherever we want in memory*… as long as we have the right to write the memory at that location.
Placement `new` serves many purposes:

*   If we have sufficient rights, it can let us map an object onto a piece of memory-mapped hardware, giving us an *extremely* thin layer of abstraction over that device.
*   It enables us to decouple allocation from construction, leading to significant speed improvements when writing containers.
*   It opens up options to implement important facilities such as types `optional<T>` (that might or might not store a `T` object) and `variant<T0,T1,...,Tn>` (that stores an object of one of types `T0`,`T1`,...,`Tn`), or even `std::string` and `std::function` that sometimes allocate external memory, but sometimes use their internal data structures and avoid allocation altogether. Placement `new` is not the only way to do this, but it is one of the options in our toolbox.

One important benefit of placement `new` is most probably in the implementation of containers and the interaction between containers and allocators, themes we will explore from *Chapter 12* to *Chapter 14* of this book. For now, we will limit ourselves to a simple, artificial example that’s meant as an illustration of how placement `new` works its magic, not as an example of something you should do (indeed, you should *not* do what the following example does!).
Suppose that you want to compute the length of a null-delimited character string and cannot remember the name of the C function that efficiently computes its length (better known as `std::strlen()`). One way to achieve similar results but *much* less efficiently would be to write the following:

```

auto string_length(const char *p) {

return std::string{ p }.size(); // 啊！但它有效...

}

```cpp

 That’s inefficient because the `std::string` constructor might allocate memory. We just wanted to count the characters until the first occurrence of a zero in the sequence, but it works (note: if you do the same maneuver with a `std::string_view` instead of with a `std::string`, its performance will actually be quite reasonable!). Now, suppose you want to show off to your friends the fact that you can place an object where you want in memory, and then use that object’s data members to do what you set out to do. You can (but should not) write the following:

```

auto string_length(const char *p) {

using std::string;

// A) 制作正确大小的局部缓冲区

// 字符串对象的对齐

alignas(string) char buf[sizeof(string)];

// B) 在该缓冲区中“绘制”字符串对象

// (注意：那个对象可能会分配其

// 使用外部数据，但那不是

// 我们的关注点在这里)

string *s = new (static_cast<void*>(buf)) string{ p };

// C) 使用该对象来计算大小

const auto sz = s->size();

// D) 销毁对象而不释放内存

// 对于缓冲区（它不是动态分配的，

// 它只是局部存储)

s->~string(); // 是的，你可以这样做

return sz;

}

```cpp

 What are the benefits of the complicated version in comparison to the simple one? None whatsoever, but it shows the intricacies of doing this sort of low-level memory management maneuver. From the comments in the code example, the steps work as follows:

*   Step `A)` makes sure that the location where the object will be constructed is of the right size and shape: it’s a buffer of bytes (type `char`), aligned in memory as a `std::string` object should be, and of sufficient size to hold a `std::string` object.
*   Step `B)` paints a `std::string` object in that buffer. That’s what a constructor does, really: it (conceptually) transforms raw memory into an object and initializes the state of that object. If the `std::string` constructor throws an exception, then the object has never been constructed and our `string_length()` function concludes without satisfying its postconditions. There is no memory allocation involved here unless the constructor itself allocates, but that’s fair (the object does what it has to do).
*   Step `C)` uses the newly constructed object; in our case, it’s just a matter of querying the size of that character string, but we could do whatever we want here. Do note, however, that (a) the object’s lifetime is tied to the buffer in which it is located, and (b) since we explicitly called the constructor, we will need to explicitly destroy it, which means that if an exception is thrown when we use the object, we will need to make sure the object’s destructor is called somehow.
*   Step `D)` destroys the object before we leave the function, as not doing so would lead to a possible leak of resources. If the buffer’s lifetime ends at a point where the object is not yet destroyed, things will be very wrong: either the destructor of the object we put in that buffer will never be called and code will leak, or someone might try to use the object even though the storage for that object is not ours anymore, leading to UB. Note the syntax, `s->~string()`, which calls the destructor but does not deallocate the storage for `*s`.

This is a bad example of placement `new` usage, but it is explicit and (hopefully) instructive. We will use this feature in much more reasonable ways in order to gain significant speed advantages when we write containers with explicit memory management in *Chapter 12*.
A note on make_shared<T>(args...)
We mentioned in *Chapter 6* that `make_shared<T>(args...)` usually leads to a better memory layout than `shared_ptr<T>{ new T(args...) }` would, at least with respect to cache usage. We can start to see why that is so.
Calling `shared_ptr<T>::shared_ptr(T*)` makes the object responsible for a preexisting pointee, the one whose address is passed as argument. Since that object has been constructed, the `shared_ptr<T>` object has to allocate a reference counter separately, ending up with two separate allocations, probably on different cache lines. In most programs, this worsened locality may induce slowdowns at runtime.
On the other hand, calling `make_shared<T>(args...)` makes this factory function responsible for creating a block of memory whose layout accommodates the `T` object and the reference counter, respecting the size and alignment constraints of both. There’s more than one way to do this, of course, including (a) resorting to a `union` where “coexist” a pair of pointers and a single pointer to a block that contains a counter and a `T` object, and (b) resorting to a byte buffer of appropriate size and alignment, then performing placement `new` for both objects in the appropriate locations within that buffer. In the latter case, we end up with a single allocation for a contiguous block of memory able to host both objects and two placement `new` calls.
Member versions of the allocation operators
Sometimes, we have special knowledge of the needs and requirements of specific types with respect to dynamic memory allocation. A full example that goes into detail about a real-life (but simplified) use case of such type-specific knowledge is given in *Chapter 10*, where we discuss arena-based allocation.
For now, we will limit ourselves to covering the syntax and the effect of a member function overload of the allocation operators. In the example that follows, we suppose class `X` would somehow benefit from a per-class specialization of these mechanisms, and show that client code will call these specializations when we call `new X` but not when we call `new int`:

```

#include <iostream>

#include <new>

class X {

// ...

public:

X() { std::cout << "X::X()\n"; }

~X() { std::cout << "X::~X()\n"; }

void *operator new(std::size_t);

void *operator new[](std::size_t);

void operator delete(void*);

void operator delete[](void*);

// ...

};

// ...

void* X::operator new(std::size_t n) {

std::cout << "Some X::operator new() magic\n";

return ::operator new(n);

}

void* X::operator new[](std::size_t n) {

std::cout << "Some X::operator new[]() magic\n";

return ::operator new[](n);

}

void X::operator delete(void *p) {

std::cout << "Some X::operator delete() magic\n";

return ::operator delete(p);

}

void X::operator delete[](void *p) {

std::cout << "Some X::operator delete[]() magic\n";

return ::operator delete[](p);

}

int main() {

std::cout << "p = new int{3}\n";

int *p = new int{ 3 }; // 全局操作符 new

std::cout << "q = new X\n";

X *q = new X; // X::operator new

std::cout << "delete p\n";

delete p; // 全局操作符 delete

std::cout << "delete q\n";

delete q; // X::operator delete

}

```cpp

 One important detail to mention is that these overloaded operators will be inherited by derived classes, which means that if the implementation of these operators somehow depends on details specific to that class – for example, its size of alignment or anything else that might be invalidated in derived classes through such seemingly inconspicuous details as adding a data member – consider marking the class that overloads these operators as `final`.
Alignment-aware versions of the allocation operators
When designing C++17, a fundamental problem with the memory allocation process was fixed with respect to what we call `std::max_align_t`.
There are many reasons for this, but a simple example would be when communicating with specialized hardware with requirements that differ from the ones on our computer. Suppose the following `Float4` type is such a type. Its size is `4*sizeof(float)`, and we require a `Float4` to be aligned on a 16-byte boundary:

```

struct alignas(16) Float4 { float vals[4]; };

```cpp

 In this example, if we remove `alignas(16)` from the type declaration, the natural alignment of type `Float4` would be `alignof(float)`, which is probably 4 on most platforms.
The problem with such types before C++17 is that variables generated by the compiler would respect our alignment requirements, but those located in dynamically allocated storage would, by default, end up with an alignment of `std::max_align_t`, which would be incorrect. That makes sense, of course; functions such as `malloc()` and `operator new()` will, by default, cover the “worst-case scenario” of the platform, not knowing what will be constructed in the allocated storage, but they cannot be assumed to implicitly cover even worse scenarios than this.
Since C++17, we can specify `operator new()` or `operator new[]()` by passing an additional argument of type `std::align_val_t`, an integral type. This has to be done explicitly at the call site, as the following example shows:

```

#include <iostream>

#include <new>

#include <cstdlib>

#include <type_traits>

void* operator new(std::size_t n, std::align_val_t al) {

std::cout << "new(" << n << ", align: "

<< static_cast<std::underlying_type_t<

std::align_val_t

>>(al) << ")\n";

return std::aligned_alloc(

static_cast<std::size_t>(al), n

);

}

// (其他省略以节省篇幅)

struct alignas(16) Float4 { float vals[4]; };

int main() {

auto p = new Float4; // 调用 operator new(size_t)

// 调用 operator new(size_t, align_val_t)

auto q = new(std::align_val_t{ 16 }) Float4;

// 泄露，当然，但这不是重点

}

```cpp

 The memory block allocated for `p` in this example will be aligned on a boundary of `std::max_align_t`, whereas the memory block allocated for `q` will be aligned on a 16-byte boundary. The former might satisfy the requirements of our type if we’re lucky and cause chaos otherwise; the latter will respect our constraints if the allocation operator overload is implemented correctly.
Destroying delete
C++20 brings a novel and highly specialized feature called destroying `delete`. The use case targeted here is a member function overload that benefits from specific knowledge of the type of object being destroyed in order to better perform the destruction process. When that member function is defined for some type `T`, it is preferred over other options when `delete` is invoked on a `T*`, even if `T` exposes another overload of `operator delete()`. To use destroying `delete` for some type `X`, one must implement the following member function:

```

class X {

// ...

public:

void operator delete(X*, std::destroying_delete_t);

// ...

};

```cpp

 Here, `std::destroying_delete_t` is a tag type like `std::nothrow_t`, which we saw earlier in this chapter. Note that the first argument of the destroying `delete` for class `X` is an `X*`, not a `void*`, as the destroying `delete` has the double role of destroying the object and deallocating memory… hence its name!
How does that work, and why is that useful? Let’s look at a concrete example with the following `Wrapper` class. In this example, an object of type `Wrapper` hides one of two implementations, modeled by `Wrapper::ImplA` and `Wrapper::ImplB`. The implementation is selected at construction time based on an enumerated value of type `Wrapper::Kind`. The intent is to remove the need for `virtual` functions from this class, replacing them with `if` statements based on the kind of implementation that was chosen. Of course, in this (admittedly) small example, there’s still only one `virtual` function (`Impl::f()`) as we aim to minimize the example’s complexity. There is also a wish to keep the destructor of class `Wrapper` trivial, a property that can be useful on occasion.
We will look at this example step by step as it is a bit more elaborate than the previous ones. First, let’s examine the basic structure of `Wrapper` including `Wrapper::Kind`, `Wrapper::Impl`, and its derived classes:

```

#include <new>

#include <iostream>

class Wrapper {

public:

enum class Kind { A, B };

private:

struct Impl {

virtual int f() const = 0;

};

struct ImplA final : Impl {

int f() const override { return 3; }

~ImplA() { std::cout << "Kind A\n"; }

};

struct ImplB final : Impl {

int f() const override { return 4; }

~ImplB() { std::cout << "Kind B\n"; }

};

Impl *p;

Kind kind;

// ...

```cpp

 Visibly, `Wrapper::Impl` does not have a `virtual` destructor, yet `Wrapper` keeps as a data member an `Impl*` named `p`, which means that simply calling `delete p` might not call the appropriate destructor for the pointed-to object.
The `Wrapper` class exposes a constructor that takes a `Kind` as argument, then calls `Wrapper::create()` to construct the appropriate implementation, modeled by a type derived from `Impl`:

```

// ...

static Impl *create(Kind kind) {

switch(kind) {

using enum Kind;

case A: return new ImplA;

case B: return new ImplB;

}

throw 0;

}

public:

Wrapper(Kind kind)

: p{ create(kind) }, kind{ kind } {

}

// ...

```cpp

 Now comes the destroying `delete`. Since we know by construction that the only possible implementations would be `ImplA` and `ImplB`, we test `p->kind` to know which one was chosen for `p`, then directly call the appropriate destructor. Once that is done, the `Wrapper` object itself is finalized and memory is freed through a direct call to `operator delete()`:

```

// ...

void operator delete(Wrapper *p,

std::destroying_delete_t) {

if(p->kind == Kind::A) {

delete static_cast<ImplA*>(p->p);

} else {

delete static_cast<ImplB*>(p->p);

}

p->~Wrapper();

::operator delete(p);

}

int f() const { return p->f(); }

};

```cpp

 For client code, the fact that we decided to use a destroying `delete` is completely transparent:

```

int main() {

using namespace std;

auto p = new Wrapper{ Wrapper::Kind::A };

cout << p->f() << endl;

删除 p;

p = new Wrapper{ Wrapper::Kind::B };

cout << p->f() << endl;

删除 p;

}

```cpp

 The destroying `delete` is a recent C++ facility as of this writing, but it is a tool that can let us get more control over the destruction process of our objects. Most of your types probably do not need this feature, but it’s good to know it exists for those cases where you need that extra bit of control over execution speed and program size. As always, measure the results of your efforts to ensure that they bring the desired benefits.
Summary
Whew, that was quite the ride! Now that we have the basics of memory allocation operator overloading handy, we will start to use them to our advantage. Our first application will be a leak detector (*Chapter 8*) using the global forms of these operators, followed by simplified examples of exotic memory management (*Chapter 9*) using specialized, custom forms of the global operators, and arena-based memory management (*Chapter 10*) with member versions of the operators that will perform very satisfying optimizations.

```
